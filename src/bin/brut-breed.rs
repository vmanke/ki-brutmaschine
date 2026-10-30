//! brut-breed: breed Cascades of model Workers on grid puzzles (Stage 2c, ADR 0005) or on
//! quantum circuits (step 1, ADR 0006).
//!
//! The library (`brutmaschine::cascade`) does all computation; this binary does the I/O:
//! the run directory, the hash-chained event log, checkpoints after every generation,
//! the kill switch, the wall-clock limit, the files that carry LLM requests and answers,
//! and the promotion of the best Cascades into a pull-request draft (ADR 0004).
//!
//! Only the simulated models are built in, under the tags sim:small and sim:large: the
//! simulated solvers on grid puzzles, the simulated circuit designer on circuits. Real
//! models (Ollama on a rented host) come through the separate adapter crate of
//! docs/STAGE2_BACKEND_CONTRACT.md (ADR 0003).
//!
//! Usage:
//!   brut-breed --out <dir> [--config <json>] [--seed N] [--generations G]
//!              [--children C] [--search P] [--select S] [--max-calls M]
//!              [--max-seconds T] [--llm-every K] [--llm-children N] [--no-pause]
//!              [--llm-answers <dir>] [--domain grid|circuits] [--selection dgm|uniform]
//!   brut-breed --out <dir> --resume [--generations G] [--max-calls M] [--max-seconds T]
//!              [--llm-answers <dir>]
//!   brut-breed --out <dir> --promote
//!   brut-breed --out <dir> --verify
//!
//! --domain names the Domain a run breeds on (step 1, ADR 0006): grid (the default) or
//! circuits. A circuit run takes its circuit settings (targets, dressing, gate cap) from
//! --config; --domain circuits alone uses the defaults: every verified library target,
//! dressed, with at most 64 gates per answer. --selection says how parents are drawn: dgm
//! (the default) or uniform, the control arm. Both define the experiment, so a resume
//! cannot change them.
//!
//! A finished circuit run also writes `<dir>/rediscovery.json`, the pre-registered count
//! of step 1: per library target, what the seed Cascades and the bred ones solved on the
//! search panel, the share D of the targets the seeds miss that bred Cascades solve, the
//! calibration gate, and the cheapest invention candidate per target (plan (g)). It is
//! written before the report, so a run with a report has it too. Whenever --verify
//! recomputes the report, it also checks that the checkpoint's archive holds one member
//! per Born event of the log, in its order, and that each member, written as a Born event,
//! is the logged one (the same id, Cascade, parents, generation, operator, solved count,
//! graded value, usage and answers). Then it recomputes the count from that archive.
//! --promote refuses circuit runs: promotion of circuit Cascades is not part of step 1.
//!
//! Exit codes: 0 done or cleanly stopped, 1 error, 2 wrong usage (also: another process
//! holds the run directory), 3 waiting for LLM answers (see <dir>/llm/requests/).
//!
//! Crash safety: before a generation makes its first call, and before each Cascade the
//! select phase scores, the checkpoint reserves the worst-case calls of that work; a
//! resume after a kill counts them as spent, so `max_calls` holds even then. One writer
//! at a time holds `<dir>/LOCK` (--verify shares it).
//!
//! Wall time: `<dir>/timings.jsonl` gets one line `{"generation":g,"seconds":s}` for each
//! committed generation. It is not part of the record: no hash covers it, and --verify
//! does not read it.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs::{File, TryLockError};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{Duration, Instant};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use brutmaschine::cascade::arena::{
    AnswerBook, BreedConfig, Breeder, Event, Finalist, ModelPool, ParentSelection, Report,
    Snapshot, Status, request_key,
};
use brutmaschine::cascade::circuit::{Answered, CircuitDomain, Rediscovery, rediscovery};
use brutmaschine::cascade::circuit_genes::circuit_pool;
use brutmaschine::cascade::designer::SimDesigner;
use brutmaschine::cascade::domain::{CircuitConfig, Domain, DomainConfig, GenePool, GridDomain};
use brutmaschine::cascade::puzzle::Grid;
use brutmaschine::cascade::sim::SimModel;
use brutmaschine::cascade::variation::Op;
use brutmaschine::util::{BrutError, Result, sha256_hex, write_atomic};

const EVENTS: &str = "events.jsonl";
const SNAPSHOT: &str = "snapshot.json";
const CONFIG: &str = "config.json";
const REPORT: &str = "report.json";
const REDISCOVERY: &str = "rediscovery.json";
const TIMINGS: &str = "timings.jsonl";
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

/// The checkpoint: breeder state plus where the log stood when it was taken. `A` is the
/// answer type of the run's domain.
#[derive(Serialize, Deserialize)]
struct Checkpoint<A = Grid> {
    snapshot: Snapshot<A>,
    log_len: u64,
    log_head: String,
}

/// What a Domain brings to the binary besides itself: the gene pool its Cascades are made
/// from, the models they call, and what it writes once a select phase is complete.
struct Family<D: Domain> {
    genes: GenePool,
    pool: ModelPool,
    /// Called once a completed select phase is logged and checkpointed, before its report
    /// is written, so an error here leaves no report.
    after_finish: fn(&Path, &Breeder<'_, D>) -> Result<()>,
    /// For --verify: error unless what `after_finish` wrote is what the finished run gives,
    /// rebuilt from its checkpoint, and that checkpoint's archive is the one the logged
    /// events it covers record. Returns the words the verify line adds.
    check_finish: fn(&Path, &[Event], Finish<'_, '_, D>) -> Result<String>,
}

/// What --verify hands a family's check of the files a finished run writes.
enum Finish<'r, 'a, D: Domain> {
    /// The report was recomputed from the checkpoint and is equal: the run rebuilt from
    /// that checkpoint.
    Rebuilt(&'r Breeder<'a, D>),
    /// The report was not recomputed, for this reason.
    Unchecked(&'static str),
}

/// One line of `timings.jsonl`: the wall time of one committed generation.
#[derive(Serialize)]
struct Timing {
    generation: usize,
    seconds: f64,
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
/// over the settings it was given, removes an earlier report, rediscovery count and
/// promotion (the run is unfinished again; a completed select phase writes them anew), and
/// drops the log entries after the checkpoint. A resume refused before that leaves the
/// run's records as they were.
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

        // The report goes first, so a run that still has one has the rest too.
        for name in [REPORT, REDISCOVERY] {
            if self.out.join(name).exists() {
                std::fs::remove_file(self.out.join(name))?;
            }
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
    // --verify only reads, so it shares the lock (verifies run side by side, never while
    // a writer writes) and works on a read-only run directory too.
    let _lock = lock_run(&out, !args.contains_key("verify"))?;

    // The settings name the Domain, so every mode reads them first: the recorded
    // ones (a resume may change two of them) or those of a fresh run.
    let cfg = if resume || read_only {
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

    if args.contains_key("promote") {
        return promote(&out, &cfg);
    }

    let verifying = args.contains_key("verify");

    match cfg.domain {
        DomainConfig::Grid if verifying => verify(&out, &cfg, &grid_family()),
        DomainConfig::Grid => drive(cfg, &grid_family(), &args, &out, resume),
        DomainConfig::Circuits(_) if verifying => verify(&out, &cfg, &circuit_family()),
        DomainConfig::Circuits(_) => drive(cfg, &circuit_family(), &args, &out, resume),
    }
}

/// Breed a run on the domain `D` until it is done, stopped, or waits for LLM answers: the
/// run loop of Stage 2c for any Domain. `cfg` holds a fresh run's settings, or the
/// recorded ones with a resume's changes.
fn drive<D: Domain>(
    cfg: BreedConfig,
    family: &Family<D>,
    args: &BTreeMap<String, String>,
    out: &Path,
    resume: bool,
) -> Result<i32> {
    let max_seconds: Option<u64> = args.get("max-seconds").map(|v| num(v, "--max-seconds")).transpose()?;
    let answers_dir = args
        .get("llm-answers")
        .map(PathBuf::from)
        .unwrap_or_else(|| out.join("llm").join("answers"));
    let book = load_answers(&answers_dir)?;

    // Every check runs before anything on disk changes (see Settle), so a refused resume
    // leaves the run's records as they were.
    let (mut breeder, mut log) = if resume {
        let cp: Checkpoint<D::Answer> = read_checkpoint(out, &cfg)?;
        let mut log = Log::open(&out.join(EVENTS))?;

        log.keep(cp.log_len, &cp.log_head)?;

        if cp.snapshot.reserved > 0 {
            println!(
                "brut-breed: the last process ended inside generation {} or the select phase; its {} reserved call(s) count as spent",
                cp.snapshot.next_generation, cp.snapshot.reserved
            );
        }

        let b = Breeder::<D>::resume_from(
            cfg.clone(),
            family.genes,
            &family.pool,
            Some(&book),
            cp.snapshot,
        )?;

        (b, log)
    } else {
        if out.join(EVENTS).exists() || out.join(SNAPSHOT).exists() {
            return Err(usage(&format!(
                "{} already holds a run; use --resume or a fresh --out",
                out.display()
            )));
        }

        let b = Breeder::<D>::start(cfg.clone(), family.genes, &family.pool, Some(&book))?;
        let log = Log::open(&out.join(EVENTS))?;

        (b, log)
    };

    let mut settle = Settle {
        out,
        cfg: &cfg,
        done: !resume,
    };

    if !resume {
        // Settings first, then the checkpoint before generation 0, so every checkpoint
        // on disk has its settings next to it.
        write_atomic(&out.join(CONFIG), serde_json::to_string_pretty(&cfg)?.as_bytes())?;
        checkpoint(out, breeder.snapshot(), &log)?;
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
        let generation = breeder.next_generation();
        let began = Instant::now();
        let (status, events) = breeder.step_with(&stop, &mut |snap| {
            settle.now(&mut log)?;
            checkpoint(out, snap.clone(), &log)
        })?;

        match status {
            Status::Continue => {
                settle.now(&mut log)?;
                log.append(&events)?;
                checkpoint(out, breeder.snapshot(), &log)?;
                // The time from the step's start until its checkpoint is written.
                record_timing(out, generation, began.elapsed())?;
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
        checkpoint(out, breeder.snapshot(), &log)?;
        stopped(&format!("before generation {} completed", breeder.next_generation()));

        return Ok(0);
    }

    // A refusal here (the select phase does not fit the budget) comes before any write.
    let finished = breeder.finish_with(&stop, &reason, &mut |snap| {
        settle.now(&mut log)?;
        checkpoint(out, snap.clone(), &log)
    })?;
    let Some((report, events)) = finished else {
        // The answers obtained so far are kept; a resume scores only the rest.
        settle.now(&mut log)?;
        checkpoint(out, breeder.snapshot(), &log)?;
        stopped("during the select phase");

        return Ok(0);
    };

    // Log first, then the checkpoint, so it covers the finalist events: a later resume
    // keeps them as history (a run finished before returns no new ones, but its log is
    // still cut back to the checkpoint). The report comes last, so a run that has one has
    // everything else a finished run writes.
    settle.now(&mut log)?;
    log.append(&events)?;
    checkpoint(out, breeder.snapshot(), &log)?;
    (family.after_finish)(out, &breeder)?;
    write_atomic(&out.join(REPORT), serde_json::to_string_pretty(&report)?.as_bytes())?;
    print_report(&report);

    Ok(0)
}

/// Write the checkpoint: `snapshot` plus where the log stands.
fn checkpoint<A: Serialize>(out: &Path, snapshot: Snapshot<A>, log: &Log) -> Result<()> {
    let cp = Checkpoint {
        snapshot,
        log_len: log.len,
        log_head: log.head().to_string(),
    };

    write_atomic(&out.join(SNAPSHOT), serde_json::to_string(&cp)?.as_bytes())
}

/// Append the wall time of the committed generation `generation` to `timings.jsonl`.
/// The file is not part of the record: no hash covers it, and --verify does not read it.
fn record_timing(out: &Path, generation: usize, took: Duration) -> Result<()> {
    let line = serde_json::to_string(&Timing {
        generation,
        seconds: took.as_secs_f64(),
    })?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out.join(TIMINGS))?;

    f.write_all(format!("{line}\n").as_bytes())?;

    Ok(())
}

/// Take `<out>/LOCK` for the life of the process, so two processes never write one run
/// (or read it while it is written): exclusively to write, shared to read. The OS
/// releases the lock when the process ends, also when it is killed. A reader that can
/// neither open nor create LOCK (a read-only copy of a run) goes on without it.
fn lock_run(out: &Path, write: bool) -> Result<Option<File>> {
    let path = out.join(LOCK);
    let opened = if write {
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
    } else {
        File::open(&path).or_else(|_| {
            std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)
        })
    };
    let lock = match opened {
        Ok(f) => f,
        Err(_) if !write => return Ok(None),
        Err(e) => {
            return Err(BrutError::Invalid(format!("{}: {e}", path.display())));
        }
    };
    let taken = if write {
        lock.try_lock()
    } else {
        lock.try_lock_shared()
    };

    match taken {
        Ok(()) => Ok(Some(lock)),
        Err(TryLockError::WouldBlock) => Err(usage(&format!(
            "another brut-breed process is using {}",
            out.display()
        ))),
        Err(TryLockError::Error(e)) => {
            Err(BrutError::Invalid(format!("{}: {e}", path.display())))
        }
    }
}

/// The checkpoint in `out`, its answers read as `A`. When it does not read, and the
/// settings it records are not those of `cfg`, it is refused for that reason: the settings
/// name the domain, so a checkpoint of another domain (whose answers do not read as `A`)
/// is a run whose recorded settings were changed.
fn read_checkpoint<A: DeserializeOwned>(out: &Path, cfg: &BreedConfig) -> Result<Checkpoint<A>> {
    let path = out.join(SNAPSHOT);
    let text = std::fs::read_to_string(&path)
        .map_err(|e| BrutError::Invalid(format!("{}: {e}", path.display())))?;
    let typed = serde_json::from_str::<Checkpoint<A>>(&text);

    if typed.is_err() {
        let recorded = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|v| v["snapshot"]["config_hash"].as_str().map(str::to_string));

        if recorded.is_some_and(|h| h != cfg.fixed_hash()) {
            return Err(BrutError::Integrity(format!(
                "{} does not match the recorded run",
                out.join(CONFIG).display()
            )));
        }
    }

    Ok(typed?)
}

/// A run on the domain `D` as recorded: the log verified in full and against the
/// checkpoint's anchor, and the settings `cfg` from the run directory checked against
/// the checkpoint and the log's start.
fn open_run<D: Domain>(out: &Path, cfg: &BreedConfig) -> Result<(Log, Checkpoint<D::Answer>)> {
    let cp: Checkpoint<D::Answer> = read_checkpoint(out, cfg)?;
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

    Ok((log, cp))
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

/// Error unless `report` is what the recorded run gives. Its finalists must be the last
/// ones in the log. When the checkpoint keeps every select-panel answer (a finished
/// run), the whole report is recomputed from it, without a model call, and must be equal;
/// returns whether that happened.
fn check_report<D: Domain>(
    report: &Report,
    log: &Log,
    cp: &Checkpoint<D::Answer>,
    cfg: &BreedConfig,
    family: &Family<D>,
) -> Result<bool> {
    let mismatch = |what: &str| {
        BrutError::Integrity(format!(
            "report.json does not match {what}; resume the run to finish it"
        ))
    };

    if report.finalists != logged_finalists(log.events()) {
        return Err(mismatch("the finalists in the event log"));
    }

    if !cp.snapshot.finished {
        return Ok(false);
    }

    // The report's own stop reason is not trusted. This binary finishes a run only when
    // every generation is done or the next one no longer fits the call budget (a stop
    // returns before the select phase), so the checkpoint tells which.
    let reason = if cp.snapshot.next_generation > cfg.generations {
        "all generations done"
    } else {
        "call budget reached"
    };

    if report.stop_reason != reason {
        return Err(mismatch("the stop reason the checkpoint implies"));
    }

    let mut breeder = Breeder::<D>::resume_from(
        cfg.clone(),
        family.genes,
        &family.pool,
        None,
        cp.snapshot.clone(),
    )?;

    if breeder.select_need() > 0 {
        return Ok(false);
    }

    // Every answer is kept, so nothing runs; stopping at once guards that anyway.
    let at_once = || true;
    let again = breeder.finish(&at_once, reason)?;

    match again {
        Some((r, _)) if r == *report => Ok(true),
        _ => Err(mismatch("the report recomputed from the checkpoint")),
    }
}

/// Check a run directory on the domain `D` with the settings `cfg` read from it: the hash
/// chain, the checkpoint anchor, the settings, the report (recomputed from the checkpoint
/// for a finished run), and what else a finished run of the family writes (also
/// recomputed, from an archive checked against the log). `timings.jsonl` is not part of
/// the record.
fn verify<D: Domain>(out: &Path, cfg: &BreedConfig, family: &Family<D>) -> Result<i32> {
    let (log, cp) = open_run::<D>(out, cfg)?;
    let has_report = out.join(REPORT).exists();
    let (report, recomputed) = if has_report {
        if check_report(&read_json(&out.join(REPORT))?, &log, &cp, cfg, family)? {
            ("; report recomputed and equal", true)
        } else {
            ("; report finalists match the log", false)
        }
    } else {
        ("", false)
    };
    // The rest of what a finished run writes is checked exactly when its report is: the
    // report comes last, so a run with one has the rest too. The checkpoint alone does not
    // tell: a resume to more generations removes these files before its first commit,
    // and until then its checkpoint still holds the finished archive.
    let finished = if recomputed {
        let rebuilt = Breeder::<D>::resume_from(
            cfg.clone(),
            family.genes,
            &family.pool,
            None,
            cp.snapshot.clone(),
        )?;

        Some(rebuilt)
    } else {
        None
    };
    let finish = match &finished {
        Some(rebuilt) => Finish::Rebuilt(rebuilt),
        None if !cp.snapshot.finished => Finish::Unchecked("the run is not finished"),
        // A process that ended between its checkpoint and its report leaves this, and so
        // does a resume to more generations that stopped before its first commit.
        None if !has_report => {
            Finish::Unchecked("there is no report.json; resume the run to write it")
        }
        None => Finish::Unchecked("the checkpoint lacks select-panel answers"),
    };
    // The archive of the checkpoint stands for the events up to its anchor; the log may
    // hold more (a process that ended after logging a generation, before its checkpoint).
    let covered = log.events().get(..cp.log_len as usize).unwrap_or_default();
    let extra = (family.check_finish)(out, covered, finish)?;

    println!(
        "brut-breed: {} events, hash chain intact, head {}; checkpoint at entry {} matches; settings match{report}{extra}",
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

/// The grid puzzles of Stage 2c: the grid gene pool and the simulated models. A finished
/// grid run writes nothing besides its report.
fn grid_family() -> Family<GridDomain> {
    Family {
        genes: GenePool::grid(),
        pool: sim_pool(),
        after_finish: |_, _| Ok(()),
        check_finish: |_, _, _| Ok(String::new()),
    }
}

/// Every allow-listed simulated circuit designer; a real model tag fails
/// `ModelPool::covers`.
fn designer_pool() -> ModelPool {
    let mut pool = ModelPool::new();

    for m in [SimDesigner::small(), SimDesigner::large()] {
        let tag = m.tag().to_string();

        pool.insert(&tag, Box::new(m));
    }

    pool
}

/// The quantum circuits of step 1 (ADR 0006): the circuit gene pool and the simulated
/// designers. A finished circuit run also writes `rediscovery.json`.
fn circuit_family() -> Family<CircuitDomain> {
    Family {
        genes: circuit_pool(),
        pool: designer_pool(),
        after_finish: write_rediscovery,
        check_finish: check_rediscovery,
    }
}

/// The rediscovery count of a circuit run (plan (g)): every archived Cascade with its
/// search-panel answers, in archive order, the hand-written seeds of generation 0 marked.
/// Err(Integrity) when an exact answer is below a Proven or Derived reference.
fn rediscovery_of(breeder: &Breeder<'_, CircuitDomain>) -> Result<Rediscovery> {
    let members: Vec<Answered<'_>> = breeder
        .archive()
        .iter()
        .map(|m| Answered {
            id: &m.id,
            seed: m.op == Op::Seed && m.generation == 0,
            generation: m.generation,
            answers: &m.answers,
        })
        .collect();

    rediscovery(breeder.search_panel(), &members)
}

/// Write `<out>/rediscovery.json` for a finished circuit run, as pretty JSON, and print its
/// headline. An integrity alarm fails the run before its report is written.
fn write_rediscovery(out: &Path, breeder: &Breeder<'_, CircuitDomain>) -> Result<()> {
    let found = rediscovery_of(breeder)?;
    let path = out.join(REDISCOVERY);

    write_atomic(&path, serde_json::to_string_pretty(&found)?.as_bytes())?;

    let open = found.library.saturating_sub(found.reached_by_seeds);
    // D > 1/2 in integers, whatever the gate says; `passes` needs both.
    let above_half = 2 * found.reached_by_bred_only > open;
    let gate = if found.calibrated {
        "calibrated"
    } else {
        "not calibrated: the seeds solve more than a third of the targets"
    };

    println!(
        "brut-breed: wrote {}: the seeds solve {} of {} targets, bred Cascades {} of the other {open}; D > 1/2: {}; {gate}",
        path.display(),
        found.reached_by_seeds,
        found.library,
        found.reached_by_bred_only,
        if above_half { "yes" } else { "no" }
    );

    for t in &found.targets {
        if let Some(c) = &t.candidate {
            println!(
                "  invention candidate on {}: T {}, CNOT {} below a best known count; the literature is not checked yet",
                t.target, c.cost.t, c.cost.cnot
            );
        }
    }

    Ok(())
}

/// --verify on a circuit run. For a finished run, rebuilt from its checkpoint: error unless
/// the checkpoint's archive is the one the `events` it covers record (`check_archive`),
/// and `<out>/rediscovery.json` holds, byte for byte, what that archive gives; a missing
/// file is an error too. Otherwise the file is not checked, and the line says why.
fn check_rediscovery(
    out: &Path,
    events: &[Event],
    finish: Finish<'_, '_, CircuitDomain>,
) -> Result<String> {
    let breeder = match finish {
        Finish::Rebuilt(b) => b,
        Finish::Unchecked(why) => return Ok(format!("; {REDISCOVERY} not checked: {why}")),
    };

    check_archive(out, breeder, events)?;

    let expected = serde_json::to_string_pretty(&rediscovery_of(breeder)?)?;
    let path = out.join(REDISCOVERY);

    if std::fs::read(&path).ok().as_deref() != Some(expected.as_bytes()) {
        return Err(BrutError::Integrity(format!(
            "{} does not match the rediscovery count recomputed from the checkpoint; resume the run to write it anew",
            path.display()
        )));
    }

    Ok("; rediscovery recomputed and equal".to_string())
}

/// Error unless the archive of `breeder`, rebuilt from the checkpoint, holds one member per
/// Born event among `events`, in their order, and each member, written as a Born event,
/// is the logged one: the same id, Cascade, parents, generation, operator, solved count,
/// graded value, usage and answers. `rediscovery` counts by the id, generation, operator
/// and answers; the rest shapes what a resume to more generations breeds. No hash covers
/// the checkpoint, so only this ties the archive to the hash-chained log: an answer edited
/// in the checkpoint would otherwise enter the count with the next resume. What a member
/// gains later (children, judgments, fitness, rating) is not in its Born event, and this
/// does not check it.
fn check_archive(out: &Path, breeder: &Breeder<'_, CircuitDomain>, events: &[Event]) -> Result<()> {
    let tasks = breeder.search_panel().len();
    let born: Vec<&Event> = events
        .iter()
        .filter(|e| matches!(e, Event::Born { .. }))
        .collect();
    let archive = breeder.archive();
    let differs = |what: String| {
        BrutError::Integrity(format!(
            "{}: {what} differs from what the Born events of {} record",
            out.join(SNAPSHOT).display(),
            out.join(EVENTS).display()
        ))
    };

    if born.len() != archive.len() {
        return Err(differs(format!(
            "the archive of {} Cascades ({} Born events)",
            archive.len(),
            born.len()
        )));
    }

    for (m, logged) in archive.iter().zip(born) {
        // The member as the breeder logs it at its birth: the answers as text, left out
        // when no answer has one.
        let answers: Vec<Option<String>> = m
            .answers
            .iter()
            .map(|a| a.as_ref().and_then(|c| breeder.domain().artefact(c)))
            .collect();
        let written = Event::Born {
            generation: m.generation,
            id: m.id.clone(),
            parents: m.parents.clone(),
            op: m.op,
            cascade: m.cascade.clone(),
            solved: m.solved,
            puzzles: tasks,
            usage: m.usage,
            quality: m.quality,
            answers: answers.iter().any(Option::is_some).then_some(answers),
        };

        if written != *logged {
            return Err(differs(format!("archive member {}", m.id)));
        }
    }

    Ok(())
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

    // --domain circuits keeps the circuit settings a config file brings. --domain grid next
    // to such settings is refused, so they are never dropped unnoticed.
    if let Some(v) = args.get("domain") {
        match (v.as_str(), &cfg.domain) {
            ("grid", DomainConfig::Grid) | ("circuits", DomainConfig::Circuits(_)) => {}
            ("circuits", DomainConfig::Grid) => {
                cfg.domain = DomainConfig::Circuits(CircuitConfig::default());
            }
            ("grid", DomainConfig::Circuits(_)) => {
                return Err(usage(
                    "--domain grid contradicts the circuit settings in --config",
                ));
            }
            _ => {
                return Err(usage(&format!(
                    "--domain must be grid or circuits, got {v:?}"
                )));
            }
        }
    }

    if let Some(v) = args.get("selection") {
        cfg.selection = match v.as_str() {
            "dgm" => ParentSelection::Dgm,
            "uniform" => ParentSelection::Uniform,
            _ => {
                return Err(usage(&format!(
                    "--selection must be dgm or uniform, got {v:?}"
                )));
            }
        };
    }

    cfg.check()?;

    Ok(cfg)
}

/// Write the two best non-seed finalists (by select panel, then search panel) as
/// candidate files plus a provenance manifest, ready for `scripts/propose-pr.sh`. Only
/// a finished grid run whose report matches its verified log is promoted; `cfg` holds the
/// run's recorded settings.
fn promote(out: &Path, cfg: &BreedConfig) -> Result<i32> {
    if !cfg.domain.is_grid() {
        return Err(BrutError::Invalid(
            "promotion of circuit Cascades is not part of step 1 (ADR 0006)".to_string(),
        ));
    }

    let family = grid_family();
    let (log, cp) = open_run::<GridDomain>(out, cfg)?;

    if !out.join(REPORT).exists() {
        return Err(BrutError::Invalid(format!(
            "{} has no report: the run is not finished; resume it first",
            out.display()
        )));
    }

    let report: Report = read_json(&out.join(REPORT))?;

    check_report(&report, &log, &cp, cfg, &family)?;

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
    const KEYS: [&str; 14] = [
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
        "domain",
        "selection",
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
