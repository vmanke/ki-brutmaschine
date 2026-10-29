//! brut-breed: breed Cascades of model Workers on grid puzzles (Stage 2c, ADR 0005).
//!
//! The library (`brutmaschine::cascade`) does all computation; this binary does the I/O:
//! the run directory, the hash-chained event log, checkpoints after every generation,
//! the kill switch, the wall-clock limit, the files that carry LLM requests and answers,
//! and the promotion of the best Cascades into a pull-request draft (ADR 0004).
//!
//! Only the simulated models are built in. Real models (Ollama on a rented host) come
//! through the separate adapter crate of docs/STAGE2_BACKEND_CONTRACT.md (ADR 0003).
//!
//! Usage:
//!   brut-breed --out <dir> [--config <json>] [--seed N] [--generations G]
//!              [--children C] [--search P] [--select S] [--max-calls M]
//!              [--max-seconds T] [--llm-every K] [--llm-children N] [--no-pause]
//!              [--llm-answers <dir>]
//!   brut-breed --out <dir> --resume [--generations G] [--max-calls M] [--max-seconds T]
//!              [--llm-answers <dir>]
//!   brut-breed --out <dir> --promote
//!   brut-breed --out <dir> --verify
//!
//! Exit codes: 0 done or cleanly stopped, 1 error, 2 wrong usage, 3 waiting for LLM
//! answers (see <dir>/llm/requests/).
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use brutmaschine::cascade::arena::{
    AnswerBook, BreedConfig, Breeder, Event, ModelPool, Report, Snapshot, Status, request_key,
};
use brutmaschine::cascade::sim::SimModel;
use brutmaschine::util::{BrutError, Result, sha256_hex, write_atomic};

const EVENTS: &str = "events.jsonl";
const SNAPSHOT: &str = "snapshot.json";
const CONFIG: &str = "config.json";
const REPORT: &str = "report.json";
const STOP: &str = "STOP";
const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// Exit code: the run waits for LLM answers.
const EXIT_PENDING: i32 = 3;

fn main() {
    let code = match run() {
        Ok(code) => code,
        Err(BrutError::Invalid(m)) if m.starts_with("usage:") => {
            eprintln!("brut-breed: {m}");
            2
        }
        Err(e) => {
            eprintln!("brut-breed: {e}");
            1
        }
    };

    process::exit(code);
}

/// One line of the event log: the event plus its place in the hash chain.
#[derive(Serialize, Deserialize)]
struct LogLine {
    seq: u64,
    prev: String,
    hash: String,
    event: Event,
}

/// The checkpoint: breeder state plus where the log stood when it was taken.
#[derive(Serialize, Deserialize)]
struct Checkpoint {
    snapshot: Snapshot,
    log_len: u64,
    log_head: String,
}

/// Hash of one log entry: SHA-256 over the previous hash, a newline, and the event JSON.
fn entry_hash(prev: &str, event_json: &str) -> String {
    sha256_hex(format!("{prev}\n{event_json}").as_bytes())
}

/// Append-only, hash-chained event log.
struct Log {
    path: PathBuf,
    len: u64,
    head: String,
}

impl Log {
    /// Read and verify the log at `path`; keep only its first `keep` entries when given
    /// (dropping events of a generation that never reached its checkpoint).
    fn open(path: &Path, keep: Option<u64>) -> Result<Log> {
        let mut log = Log {
            path: path.to_path_buf(),
            len: 0,
            head: GENESIS.to_string(),
        };

        if !path.exists() {
            return Ok(log);
        }

        let text = std::fs::read_to_string(path)?;
        let mut kept = String::new();

        for line in text.lines() {
            if keep.is_some_and(|k| log.len >= k) {
                break;
            }

            let entry: LogLine = serde_json::from_str(line)?;
            let event_json = serde_json::to_string(&entry.event)?;

            if entry.seq != log.len
                || entry.prev != log.head
                || entry.hash != entry_hash(&log.head, &event_json)
            {
                return Err(BrutError::Integrity(format!(
                    "{}: entry {} breaks the hash chain",
                    path.display(),
                    log.len
                )));
            }

            log.head = entry.hash;
            log.len += 1;
            kept.push_str(line);
            kept.push('\n');
        }

        if let Some(k) = keep {
            if log.len != k {
                return Err(BrutError::Integrity(format!(
                    "{}: has {} entries, the checkpoint expects {k}",
                    path.display(),
                    log.len
                )));
            }

            write_atomic(path, kept.as_bytes())?;
        }

        Ok(log)
    }

    fn append(&mut self, events: &[Event]) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }

        let mut out = String::new();

        for event in events {
            let event_json = serde_json::to_string(event)?;
            let hash = entry_hash(&self.head, &event_json);
            let line = LogLine {
                seq: self.len,
                prev: self.head.clone(),
                hash: hash.clone(),
                event: event.clone(),
            };

            out.push_str(&serde_json::to_string(&line)?);
            out.push('\n');
            self.head = hash;
            self.len += 1;
        }

        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;

        f.write_all(out.as_bytes())?;
        f.sync_all()?;

        Ok(())
    }
}

fn run() -> Result<i32> {
    let args = parse_args()?;
    let out = PathBuf::from(
        args.get("out")
            .ok_or_else(|| usage("--out <dir> is required"))?,
    );

    if args.contains_key("verify") {
        let log = Log::open(&out.join(EVENTS), None)?;

        println!("brut-breed: {} events, hash chain intact, head {}", log.len, log.head);

        return Ok(0);
    }

    if args.contains_key("promote") {
        return promote(&out);
    }

    let resume = args.contains_key("resume");
    let cfg = if resume {
        let mut cfg: BreedConfig = read_json(&out.join(CONFIG))?;

        // Only these may change on resume (BreedConfig::fixed_hash).
        if let Some(v) = args.get("generations") {
            cfg.generations = num(v, "--generations")?;
        }

        if let Some(v) = args.get("max-calls") {
            cfg.max_calls = num(v, "--max-calls")?;
        }

        cfg
    } else {
        fresh_config(&args)?
    };

    let max_seconds: Option<u64> = args.get("max-seconds").map(|v| num(v, "--max-seconds")).transpose()?;
    let pool = sim_pool();
    let answers_dir = args
        .get("llm-answers")
        .map(PathBuf::from)
        .unwrap_or_else(|| out.join("llm").join("answers"));
    let book = load_answers(&answers_dir)?;

    let (mut breeder, mut log) = if resume {
        let cp: Checkpoint = read_json(&out.join(SNAPSHOT))?;
        let log = Log::open(&out.join(EVENTS), Some(cp.log_len))?;

        if log.head != cp.log_head {
            return Err(BrutError::Integrity(
                "event log head does not match the checkpoint".to_string(),
            ));
        }

        (Breeder::resume(cfg.clone(), &pool, Some(&book), cp.snapshot)?, log)
    } else {
        if out.join(EVENTS).exists() {
            return Err(usage(&format!(
                "{} already holds a run; use --resume or a fresh --out",
                out.display()
            )));
        }

        std::fs::create_dir_all(&out)?;

        let b = Breeder::new(cfg.clone(), &pool, Some(&book))?;

        (b, Log::open(&out.join(EVENTS), None)?)
    };

    write_atomic(&out.join(CONFIG), serde_json::to_string_pretty(&cfg)?.as_bytes())?;

    if !book.is_empty() {
        println!("brut-breed: {} recorded LLM answer(s) loaded", book.len());
    }

    let start = Instant::now();
    let limit = max_seconds.map(Duration::from_secs);
    let stop_file = out.join(STOP);
    let stop = || stop_file.exists() || limit.is_some_and(|l| start.elapsed() >= l);

    let reason = loop {
        let (status, events) = breeder.step(&stop)?;

        match status {
            Status::Continue => {
                log.append(&events)?;
                checkpoint(&out, &breeder, &log)?;
                print_generation(&events);
            }
            Status::Done(reason) => break reason,
            Status::Pending(requests) => {
                let dir = out.join("llm").join("requests");

                std::fs::create_dir_all(&dir)?;

                for req in &requests {
                    let path = dir.join(format!("{}.json", request_key(req)));

                    write_atomic(&path, serde_json::to_string_pretty(req)?.as_bytes())?;
                }

                println!(
                    "brut-breed: generation {} waits for {} LLM answer(s).\n\
                     Requests: {}\n\
                     Write each answer as {}/<key>.txt, then run: brut-breed --out {} --resume",
                    breeder.next_generation(),
                    requests.len(),
                    dir.display(),
                    answers_dir.display(),
                    out.display()
                );

                return Ok(EXIT_PENDING);
            }
        }
    };

    if reason == "stopped" {
        println!(
            "brut-breed: stopped after generation {}; resume with --resume",
            breeder.next_generation().saturating_sub(1)
        );

        return Ok(0);
    }

    let (report, events) = breeder.finish(&reason)?;

    log.append(&events)?;
    write_atomic(&out.join(REPORT), serde_json::to_string_pretty(&report)?.as_bytes())?;
    print_report(&report);

    Ok(0)
}

/// Write the checkpoint for the generation just committed.
fn checkpoint(out: &Path, breeder: &Breeder, log: &Log) -> Result<()> {
    let cp = Checkpoint {
        snapshot: breeder.snapshot(),
        log_len: log.len,
        log_head: log.head.clone(),
    };

    write_atomic(&out.join(SNAPSHOT), serde_json::to_string(&cp)?.as_bytes())
}

/// Every allow-listed simulated model; a real model tag fails `ModelPool::covers`.
fn sim_pool() -> ModelPool {
    let mut pool = ModelPool::new();

    for m in [SimModel::small(), SimModel::large()] {
        let tag = m.tag().to_string();

        pool.insert(&tag, Box::new(m));
    }

    pool
}

/// Recorded LLM answers: every `<key>.txt` in `dir`.
fn load_answers(dir: &Path) -> Result<AnswerBook> {
    let mut book = AnswerBook::new();

    if !dir.exists() {
        return Ok(book);
    }

    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();

        if path.extension().is_some_and(|x| x == "txt")
            && let Some(key) = path.file_stem().and_then(|s| s.to_str())
        {
            book.insert(key, std::fs::read_to_string(&path)?);
        }
    }

    Ok(book)
}

fn fresh_config(args: &BTreeMap<String, String>) -> Result<BreedConfig> {
    let mut cfg = match args.get("config") {
        Some(path) => read_json(Path::new(path))?,
        None => BreedConfig::default(),
    };

    let set = |key: &str, target: &mut usize| -> Result<()> {
        if let Some(v) = args.get(key) {
            *target = num(v, &format!("--{key}"))?;
        }

        Ok(())
    };

    set("generations", &mut cfg.generations)?;
    set("children", &mut cfg.children_per_generation)?;
    set("search", &mut cfg.search_puzzles)?;
    set("select", &mut cfg.select_puzzles)?;
    set("llm-every", &mut cfg.llm_every)?;
    set("llm-children", &mut cfg.llm_children)?;

    if let Some(v) = args.get("seed") {
        cfg.seed = num(v, "--seed")?;
    }

    if let Some(v) = args.get("max-calls") {
        cfg.max_calls = num(v, "--max-calls")?;
    }

    if args.contains_key("no-pause") {
        cfg.pause_for_llm = false;
    }

    cfg.check()?;

    Ok(cfg)
}

/// Write the two best non-seed finalists (by select panel, then search panel) as
/// candidate files plus a provenance manifest, ready for `scripts/propose-pr.sh`.
fn promote(out: &Path) -> Result<i32> {
    let report: Report = read_json(&out.join(REPORT))?;
    let cfg: BreedConfig = read_json(&out.join(CONFIG))?;
    let cp: Checkpoint = read_json(&out.join(SNAPSHOT))?;
    let mut finalists: Vec<_> = report.finalists.iter().filter(|f| !f.is_seed).collect();

    finalists.sort_by(|a, b| {
        b.select_solved
            .cmp(&a.select_solved)
            .then(b.search_solved_rate.total_cmp(&a.search_solved_rate))
            .then(a.id.cmp(&b.id))
    });

    if finalists.len() < 2 {
        return Err(BrutError::Invalid(
            "promotion needs two non-seed finalists (ADR 0004: two candidates per pull request)"
                .to_string(),
        ));
    }

    let dir = out.join("promote").join("candidates").join("cascades");
    let mut provenance = Vec::new();

    std::fs::create_dir_all(&dir)?;

    for f in finalists.iter().take(2) {
        let m = cp
            .snapshot
            .archive
            .iter()
            .find(|m| m.id == f.id)
            .ok_or_else(|| BrutError::Invalid(format!("finalist {} not in the archive", f.id)))?;
        let path = dir.join(format!("{}.json", m.id));

        write_atomic(&path, serde_json::to_string_pretty(&m.cascade)?.as_bytes())?;
        println!("wrote {}", path.display());

        provenance.push(serde_json::json!({
            "child_id": m.id,
            "parents": m.parents,
            "op": m.op,
            "generation": m.generation,
            "search_solved_rate": f.search_solved_rate,
        }));
    }

    let manifest = serde_json::json!({
        "tool": "brut-breed",
        "adapter": "cascade",
        "master_seed": cfg.seed,
        "config_hash": cfg.fixed_hash(),
        "candidates": provenance,
    });
    let path = out.join("promote").join("proposal.json");

    write_atomic(&path, serde_json::to_string_pretty(&manifest)?.as_bytes())?;
    println!(
        "wrote {}\nnext: scripts/propose-pr.sh {}",
        path.display(),
        out.join("promote").display()
    );

    Ok(0)
}

fn print_generation(events: &[Event]) {
    for e in events {
        if let Event::Generation {
            generation,
            best_id,
            best_fitness,
            best_solved,
            archive_size,
            calls_used,
        } = e
        {
            println!(
                "gen {generation:>3}  best {best_id}  fitness {best_fitness:.4}  solved {best_solved}  archive {archive_size}  calls {calls_used}"
            );
        }
    }
}

fn print_report(r: &Report) {
    println!(
        "\nbrut-breed: {} ({} generations, {} Cascades, {} Worker calls, {} peer comparisons, {} LLM answers used)",
        r.stop_reason,
        r.generations_completed,
        r.archive_size,
        r.calls_used,
        r.comparisons,
        r.mutator_answers_used
    );

    for t in r.top.iter().take(5) {
        println!(
            "  {}  gen {:>3}  {:<17}  fitness {:.4}  solved {}/{}  judge {:.2}  rating {:+.2}  {} stage(s), {} worker(s), {:.0} tokens/puzzle",
            t.id,
            t.generation,
            format!("{:?}", t.op),
            t.fitness,
            t.solved,
            t.puzzles,
            t.judge_skill,
            t.rating,
            t.stages,
            t.workers,
            t.completion_tokens_per_puzzle
        );
    }

    println!(
        "select panel (unseen during search): best finalist {:.3}, best seed {:.3}",
        r.best_finalist_select_rate, r.best_seed_select_rate
    );
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| BrutError::Invalid(format!("{}: {e}", path.display())))?;

    Ok(serde_json::from_str(&text)?)
}

fn num<T: std::str::FromStr>(v: &str, flag: &str) -> Result<T> {
    v.parse()
        .map_err(|_| usage(&format!("{flag} needs a number, got {v:?}")))
}

fn usage(m: &str) -> BrutError {
    BrutError::Invalid(format!("usage: {m}"))
}

/// `--key value` pairs; `--resume`, `--promote`, `--verify` and `--no-pause` take none.
fn parse_args() -> Result<BTreeMap<String, String>> {
    const FLAGS: [&str; 4] = ["resume", "promote", "verify", "no-pause"];
    const KEYS: [&str; 12] = [
        "out",
        "config",
        "seed",
        "generations",
        "children",
        "search",
        "select",
        "max-calls",
        "max-seconds",
        "llm-every",
        "llm-children",
        "llm-answers",
    ];

    let mut map = BTreeMap::new();
    let mut it = std::env::args().skip(1);

    while let Some(tok) = it.next() {
        let key = tok
            .strip_prefix("--")
            .ok_or_else(|| usage(&format!("expected --key, got {tok}")))?;

        if FLAGS.contains(&key) {
            map.insert(key.to_string(), String::new());
        } else if KEYS.contains(&key) {
            let val = it
                .next()
                .ok_or_else(|| usage(&format!("--{key} needs a value")))?;

            map.insert(key.to_string(), val);
        } else {
            return Err(usage(&format!("unknown option --{key}")));
        }
    }

    Ok(map)
}
