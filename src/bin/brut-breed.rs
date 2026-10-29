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
//! Exit codes: 0 done or cleanly stopped, 1 error, 2 wrong usage (also: another process
//! holds the run directory), 3 waiting for LLM answers (see <dir>/llm/requests/).
//!
//! Crash safety: before a generation or the select phase makes its first call, the
//! checkpoint reserves its worst-case calls; a resume after a kill counts them as spent,
//! so `max_calls` holds even then. One process at a time holds `<dir>/LOCK`.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs::{File, TryLockError};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use brutmaschine::cascade::arena::{
    AnswerBook, BreedConfig, Breeder, Event, Finalist, ModelPool, Report, Snapshot, Status,
    request_key,
};
use brutmaschine::cascade::sim::SimModel;
use brutmaschine::util::{BrutError, Result, sha256_hex, write_atomic};

const EVENTS: &str = "events.jsonl";
const SNAPSHOT: &str = "snapshot.json";
const CONFIG: &str = "config.json";
const REPORT: &str = "report.json";
const STOP: &str = "STOP";
const LOCK: &str = "LOCK";
const PROMOTE: &str = "promote";
const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// Options each mode accepts. A resume may change only what `BreedConfig::fixed_hash`
/// leaves out, plus the per-process wall-clock limit and the answers directory.
const RESUME_KEYS: [&str; 6] = ["out", "resume", "generations", "max-calls", "max-seconds", "llm-answers"];
const READ_KEYS: [&str; 3] = ["out", "promote", "verify"];

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

/// Append-only, hash-chained event log. Opening it only reads and verifies; entries a
/// resume drops (those after its checkpoint) leave the file with the next append.
struct Log {
    path: PathBuf,
    /// Every verified entry on disk: its line (without the newline) and its event.
    lines: Vec<String>,
    events: Vec<Event>,
    /// Hash of each entry, in order.
    hashes: Vec<String>,
    /// Entries that count; the rest are dropped from disk by the next append.
    len: u64,
    /// True when the file ends in an incomplete line (a write the process did not finish).
    torn: bool,
}

impl Log {
    /// Read and verify the whole log at `path`; an absent file is an empty log. Any
    /// complete line that is not canonical or breaks the chain is an error. Nothing on
    /// disk changes.
    fn open(path: &Path) -> Result<Log> {
        let mut log = Log {
            path: path.to_path_buf(),
            lines: Vec::new(),
            events: Vec::new(),
            hashes: Vec::new(),
            len: 0,
            torn: false,
        };

        if !path.exists() {
            return Ok(log);
        }

        let bytes = std::fs::read(path)?;

        for piece in bytes.split_inclusive(|&b| b == b'\n') {
            let Some(raw) = piece.strip_suffix(b"\n") else {
                // Only the last piece can lack its newline: a write cut short.
                log.torn = true;
                break;
            };
            let bad = |what: &str| {
                BrutError::Integrity(format!("{}: entry {} {what}", path.display(), log.lines.len()))
            };
            let line = std::str::from_utf8(raw).map_err(|_| bad("is not UTF-8"))?;
            let entry: LogLine = serde_json::from_str(line).map_err(|_| bad("is not a log entry"))?;
            let event_json = serde_json::to_string(&entry.event)?;
            let head = log.head().to_string();

            // Lines are written canonically, so a line that does not re-serialise to
            // itself was edited (whitespace, reordered or unknown fields included).
            if serde_json::to_string(&entry)? != line {
                return Err(bad("is not in canonical form"));
            }

            if entry.seq != log.len || entry.prev != head || entry.hash != entry_hash(&head, &event_json) {
                return Err(bad("breaks the hash chain"));
            }

            log.lines.push(line.to_string());
            log.events.push(entry.event);
            log.hashes.push(entry.hash);
            log.len += 1;
        }

        Ok(log)
    }

    /// Hash of the last counted entry (GENESIS for none).
    fn head(&self) -> &str {
        match self.len {
            0 => GENESIS,
            n => &self.hashes[n as usize - 1],
        }
    }

    /// The counted events.
    fn events(&self) -> &[Event] {
        &self.events[..self.len as usize]
    }

    /// Error unless the log holds at least `len` entries and the hash of entry `len - 1`
    /// is `head` (GENESIS for `len` 0): the anchor a checkpoint records.
    fn check_anchor(&self, len: u64, head: &str) -> Result<()> {
        let at = match len {
            0 => Some(GENESIS),
            n => self.hashes.get(n as usize - 1).map(String::as_str),
        };

        if at != Some(head) {
            return Err(BrutError::Integrity(format!(
                "{}: holds {} entries and does not match the checkpoint (entry {len}, head {head})",
                self.path.display(),
                self.hashes.len()
            )));
        }

        Ok(())
    }

    /// Count only the first `len` entries, after checking the anchor; `flush` (also the
    /// one in `append`) drops the rest from disk.
    fn keep(&mut self, len: u64, head: &str) -> Result<()> {
        self.check_anchor(len, head)?;
        self.len = len;

        Ok(())
    }

    /// Drop from disk, in one atomic step, the entries after the counted ones and a torn
    /// last line. Nothing happens when there are none.
    fn flush(&mut self) -> Result<()> {
        if !self.torn && self.len == self.lines.len() as u64 {
            return Ok(());
        }

        self.lines.truncate(self.len as usize);
        self.events.truncate(self.len as usize);
        self.hashes.truncate(self.len as usize);

        let mut all = String::new();

        for line in &self.lines {
            all.push_str(line);
            all.push('\n');
        }

        write_atomic(&self.path, all.as_bytes())?;
        self.torn = false;

        Ok(())
    }

    fn append(&mut self, events: &[Event]) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }

        self.flush()?;

        let mut out = String::new();

        for event in events {
            let event_json = serde_json::to_string(event)?;
            let prev = self.head().to_string();
            let hash = entry_hash(&prev, &event_json);
            let line = serde_json::to_string(&LogLine {
                seq: self.len,
                prev,
                hash: hash.clone(),
                event: event.clone(),
            })?;

            out.push_str(&line);
            out.push('\n');
            self.lines.push(line);
            self.events.push(event.clone());
            self.hashes.push(hash);
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

/// What a resume changes on disk, applied once, right before its first write: it takes
/// over the settings it was given, removes an earlier report and promotion (the run is
/// unfinished again; a completed select phase writes the report anew), and drops the log
/// entries after the checkpoint. A resume refused before that leaves the run's records
/// as they were.
struct Settle<'a> {
    out: &'a Path,
    cfg: &'a BreedConfig,
    done: bool,
}

impl Settle<'_> {
    fn now(&mut self, log: &mut Log) -> Result<()> {
        if self.done {
            return Ok(());
        }

        log.flush()?;
        write_atomic(&self.out.join(CONFIG), serde_json::to_string_pretty(self.cfg)?.as_bytes())?;

        if self.out.join(REPORT).exists() {
            std::fs::remove_file(self.out.join(REPORT))?;
        }

        if self.out.join(PROMOTE).exists() {
            std::fs::remove_dir_all(self.out.join(PROMOTE))?;
        }

        self.done = true;

        Ok(())
    }
}

fn run() -> Result<i32> {
    let args = parse_args()?;
    let out = PathBuf::from(
        args.get("out")
            .ok_or_else(|| usage("--out <dir> is required"))?,
    );
    let modes = ["resume", "promote", "verify"].iter().filter(|m| args.contains_key(**m)).count();

    if modes > 1 {
        return Err(usage("use at most one of --resume, --promote and --verify"));
    }

    let resume = args.contains_key("resume");
    let read_only = args.contains_key("verify") || args.contains_key("promote");

    // A resume may change only what fixed_hash leaves out; anything else would silently
    // be ignored, so it is refused.
    let allowed: Option<(&[&str], &str)> = if resume {
        Some((
            &RESUME_KEYS,
            "a resume may change only --generations, --max-calls, --max-seconds and --llm-answers; the rest defines the experiment",
        ))
    } else if read_only {
        Some((&READ_KEYS, "--verify and --promote take only --out"))
    } else {
        None
    };

    if let Some((keys, why)) = allowed
        && let Some(key) = args.keys().find(|k| !keys.contains(&k.as_str()))
    {
        return Err(usage(&format!("--{key} cannot be given here: {why}")));
    }

    if read_only || resume {
        if !out.join(SNAPSHOT).exists() {
            return Err(usage(&format!("{} is not a run directory (no {SNAPSHOT})", out.display())));
        }
    } else {
        std::fs::create_dir_all(&out)?;
    }

    // Held until the process ends; the OS releases it even when the process is killed.
    let _lock = lock_run(&out)?;

    if args.contains_key("verify") {
        return verify(&out);
    }

    if args.contains_key("promote") {
        return promote(&out);
    }

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

    // Every check runs before anything on disk changes (see Settle), so a refused resume
    // leaves the run's records as they were.
    let (mut breeder, mut log) = if resume {
        let cp: Checkpoint = read_json(&out.join(SNAPSHOT))?;
        let mut log = Log::open(&out.join(EVENTS))?;

        log.keep(cp.log_len, &cp.log_head)?;

        if cp.snapshot.reserved > 0 {
            println!(
                "brut-breed: the last process ended inside generation {} or the select phase; its {} reserved call(s) count as spent",
                cp.snapshot.next_generation, cp.snapshot.reserved
            );
        }

        (Breeder::resume(cfg.clone(), &pool, Some(&book), cp.snapshot)?, log)
    } else {
        if out.join(EVENTS).exists() || out.join(SNAPSHOT).exists() {
            return Err(usage(&format!(
                "{} already holds a run; use --resume or a fresh --out",
                out.display()
            )));
        }

        let b = Breeder::new(cfg.clone(), &pool, Some(&book))?;
        let log = Log::open(&out.join(EVENTS))?;

        (b, log)
    };

    let mut settle = Settle {
        out: &out,
        cfg: &cfg,
        done: !resume,
    };

    if !resume {
        // Settings first, then the checkpoint before generation 0, so every checkpoint
        // on disk has its settings next to it.
        write_atomic(&out.join(CONFIG), serde_json::to_string_pretty(&cfg)?.as_bytes())?;
        checkpoint(&out, breeder.snapshot(), &log)?;
    }

    if !book.is_empty() {
        println!("brut-breed: {} recorded LLM answer(s) loaded", book.len());
    }

    let start = Instant::now();
    let limit = max_seconds.map(Duration::from_secs);
    let stop_file = out.join(STOP);
    let stop = || stop_file.exists() || limit.is_some_and(|l| start.elapsed() >= l);
    let resume_cmd = format!(
        "brut-breed --out {} --resume{}",
        out.display(),
        args.get("llm-answers").map(|d| format!(" --llm-answers {d}")).unwrap_or_default()
    );
    let stopped = |phase: &str| {
        if stop_file.exists() {
            println!(
                "brut-breed: stopped by {} {phase}; delete {}, then run: {resume_cmd}",
                stop_file.display(),
                stop_file.display()
            );
        } else {
            println!("brut-breed: stopped at the time limit {phase}; run: {resume_cmd}");
        }
    };

    let reason = loop {
        let (status, events) = breeder.step_with(&stop, &mut |snap| {
            settle.now(&mut log)?;
            checkpoint(&out, snap.clone(), &log)
        })?;

        match status {
            Status::Continue => {
                settle.now(&mut log)?;
                log.append(&events)?;
                checkpoint(&out, breeder.snapshot(), &log)?;
                print_generation(&events);
            }
            Status::Done(reason) => break reason,
            Status::Pending(requests) => {
                settle.now(&mut log)?;

                let dir = out.join("llm").join("requests");

                std::fs::create_dir_all(&dir)?;

                for req in &requests {
                    let path = dir.join(format!("{}.json", request_key(req)));

                    write_atomic(&path, serde_json::to_string_pretty(req)?.as_bytes())?;
                }

                println!(
                    "brut-breed: generation {} waits for {} LLM answer(s).\n\
                     Requests: {}\n\
                     Write each answer as {}/<key>.txt, then run: {resume_cmd}",
                    breeder.next_generation(),
                    requests.len(),
                    dir.display(),
                    answers_dir.display()
                );

                return Ok(EXIT_PENDING);
            }
        }
    };

    if reason == "stopped" {
        // Nothing of the interrupted generation is kept except the calls it spent.
        settle.now(&mut log)?;
        checkpoint(&out, breeder.snapshot(), &log)?;
        stopped(&format!("before generation {} completed", breeder.next_generation()));

        return Ok(0);
    }

    // A refusal here (the select phase does not fit the budget) comes before any write.
    let finished = breeder.finish_with(&stop, &reason, &mut |snap| {
        settle.now(&mut log)?;
        checkpoint(&out, snap.clone(), &log)
    })?;
    let Some((report, events)) = finished else {
        // The answers obtained so far are kept; a resume scores only the rest.
        settle.now(&mut log)?;
        checkpoint(&out, breeder.snapshot(), &log)?;
        stopped("during the select phase");

        return Ok(0);
    };

    // Log first, then the checkpoint, so it covers the finalist events: a later resume
    // keeps them as history (a run finished before returns no new ones, but its log is
    // still cut back to the checkpoint).
    settle.now(&mut log)?;
    log.append(&events)?;
    checkpoint(&out, breeder.snapshot(), &log)?;
    write_atomic(&out.join(REPORT), serde_json::to_string_pretty(&report)?.as_bytes())?;
    print_report(&report);

    Ok(0)
}

/// Write the checkpoint: `snapshot` plus where the log stands.
fn checkpoint(out: &Path, snapshot: Snapshot, log: &Log) -> Result<()> {
    let cp = Checkpoint {
        snapshot,
        log_len: log.len,
        log_head: log.head().to_string(),
    };

    write_atomic(&out.join(SNAPSHOT), serde_json::to_string(&cp)?.as_bytes())
}

/// Take `<out>/LOCK` exclusively for the life of the process, so two processes never
/// write one run (or read it while it is written). The OS releases the lock when the
/// process ends, also when it is killed.
fn lock_run(out: &Path) -> Result<File> {
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(out.join(LOCK))?;

    match lock.try_lock() {
        Ok(()) => Ok(lock),
        Err(TryLockError::WouldBlock) => Err(usage(&format!(
            "another brut-breed process is using {}",
            out.display()
        ))),
        Err(TryLockError::Error(e)) => Err(e.into()),
    }
}

/// A run as recorded: the log verified in full and against the checkpoint's anchor, and
/// the settings checked against the checkpoint and the log's start.
fn open_run(out: &Path) -> Result<(Log, Checkpoint, BreedConfig)> {
    let cp: Checkpoint = read_json(&out.join(SNAPSHOT))?;
    let cfg: BreedConfig = read_json(&out.join(CONFIG))?;
    let log = Log::open(&out.join(EVENTS))?;

    if log.torn {
        return Err(BrutError::Integrity(format!(
            "{}: the last line is incomplete (the process ended while writing it); a resumed run drops it with its next write to the log",
            out.join(EVENTS).display()
        )));
    }

    log.check_anchor(cp.log_len, &cp.log_head)?;

    let hash = cfg.fixed_hash();
    let started = log.events().iter().find_map(|e| match e {
        Event::RunStart { config_hash, .. } => Some(config_hash),
        _ => None,
    });

    if hash != cp.snapshot.config_hash || started.is_some_and(|h| *h != hash) {
        return Err(BrutError::Integrity(format!(
            "{} does not match the recorded run",
            out.join(CONFIG).display()
        )));
    }

    Ok((log, cp, cfg))
}

/// The Finalist events after the last Generation event: the select phase of the archive
/// as it stands (earlier sets stay in the log as history).
fn logged_finalists(events: &[Event]) -> Vec<Finalist> {
    let start = events
        .iter()
        .rposition(|e| matches!(e, Event::Generation { .. }))
        .map_or(0, |i| i + 1);

    events[start..]
        .iter()
        .filter_map(|e| match e {
            Event::Finalist(f) => Some(f.clone()),
            _ => None,
        })
        .collect()
}

/// Error unless `report` lists exactly the finalists the log holds last.
fn check_report(report: &Report, log: &Log) -> Result<()> {
    if report.finalists != logged_finalists(log.events()) {
        return Err(BrutError::Integrity(
            "report.json does not match the finalists in the event log; resume the run to finish it"
                .to_string(),
        ));
    }

    Ok(())
}

/// Check a run directory: the hash chain, the checkpoint anchor, the settings, and the
/// report against the log.
fn verify(out: &Path) -> Result<i32> {
    let (log, cp, _) = open_run(out)?;

    if out.join(REPORT).exists() {
        check_report(&read_json(&out.join(REPORT))?, &log)?;
    }

    println!(
        "brut-breed: {} events, hash chain intact, head {}; checkpoint at entry {} matches; settings match",
        log.len,
        log.head(),
        cp.log_len
    );

    Ok(0)
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
/// candidate files plus a provenance manifest, ready for `scripts/propose-pr.sh`. Only
/// a finished run whose report matches its verified log is promoted.
fn promote(out: &Path) -> Result<i32> {
    let (log, cp, cfg) = open_run(out)?;

    if !out.join(REPORT).exists() {
        return Err(BrutError::Invalid(format!(
            "{} has no report: the run is not finished; resume it first",
            out.display()
        )));
    }

    let report: Report = read_json(&out.join(REPORT))?;

    check_report(&report, &log)?;

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

    let promote_dir = out.join(PROMOTE);
    let dir = promote_dir.join("candidates").join("cascades");
    let mut provenance = Vec::new();

    // Generated output only: drop an earlier promotion so the pull request carries
    // exactly this pair and its manifest.
    if promote_dir.exists() {
        std::fs::remove_dir_all(&promote_dir)?;
    }

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
    let path = out.join(PROMOTE).join("proposal.json");

    write_atomic(&path, serde_json::to_string_pretty(&manifest)?.as_bytes())?;
    println!(
        "wrote {}\nnext: scripts/propose-pr.sh {}",
        path.display(),
        out.join(PROMOTE).display()
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
