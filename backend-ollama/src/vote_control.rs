//! vote-control: the compute-equal majority-vote control of a finished grid breeding run
//! (module `vote`). It samples one seed Cascade N times per select puzzle, keeps the most
//! frequent grid, scores it exactly, and sets it beside the bred champion's select result.
//!
//! Usage:
//!   vote-control --run <dir> [--samples N] [--cascade ID] [--temperature T]
//!                [--puzzles K] [--jobs J] [--dry-run]
//!
//! Without --samples, N is the champion's average Worker calls per search puzzle divided by
//! the sampled Cascade's own, rounded. --cascade samples another archive member than the
//! seed finalist with the best search fitness; --temperature sets every Worker's
//! temperature (another Cascade then, with its own id). --puzzles K scores only the first K
//! select puzzles, a smoke test: the result goes to stdout only and says `partial`.
//! --jobs J runs J samples at a time (default 16). --dry-run prints the plan and makes no
//! call; it also works on an unfinished run.
//!
//! The result goes to stdout and to `<dir>/vote-control.json`. Real models are recorded in
//! `<dir>/vote-control/worker-calls.jsonl` (failed calls in `worker-errors.jsonl` beside
//! it), apart from the run's own record; a repeated control replays them. `VLLM_URL`,
//! `OLLAMA_URL` and `OLLAMA_TIMEOUT_SECS` work as for brut-breed-ollama.
//!
//! Exit codes: 0 done, 1 error or failed calls (the result is still written; run again to
//! retry the failed calls), 2 wrong usage (also: an option the run cannot take, the run is
//! unfinished, or a brut-breed process is writing it).
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};
use std::process;

use backend_ollama::vote::{
    Endpoints, OUTPUT, Options, RECORD, build_pool, check_panel, check_run_start, load_run, plan,
    select_panel, vote,
};
use brutmaschine::util::{BrutError, Result, write_atomic};

/// Samples run at the same time unless --jobs says otherwise.
const DEFAULT_JOBS: usize = 16;

fn main() {
    let code = match run() {
        Ok(code) => code,
        Err(BrutError::Invalid(m)) if m.starts_with("usage:") => {
            eprintln!("vote-control: {m}");
            2
        }
        Err(e) => {
            eprintln!("vote-control: {e}");
            1
        }
    };

    process::exit(code);
}

fn usage(m: &str) -> BrutError {
    BrutError::Invalid(format!("usage: {m}"))
}

fn parse_args() -> Result<BTreeMap<String, String>> {
    const FLAGS: [&str; 1] = ["dry-run"];
    const KEYS: [&str; 6] = ["run", "samples", "cascade", "temperature", "puzzles", "jobs"];

    let mut map = BTreeMap::new();
    let mut it = std::env::args().skip(1);

    while let Some(tok) = it.next() {
        let key = tok
            .strip_prefix("--")
            .ok_or_else(|| usage(&format!("expected --key, got {tok}")))?;

        if FLAGS.contains(&key) {
            map.insert(key.to_string(), String::new());
        } else if KEYS.contains(&key) {
            let val = it.next().ok_or_else(|| usage(&format!("--{key} needs a value")))?;

            map.insert(key.to_string(), val);
        } else {
            return Err(usage(&format!("unknown option --{key}")));
        }
    }

    Ok(map)
}

fn num<T: std::str::FromStr>(args: &BTreeMap<String, String>, key: &str) -> Result<Option<T>> {
    args.get(key)
        .map(|v| v.parse().map_err(|_| usage(&format!("--{key} needs a number, got {v:?}"))))
        .transpose()
}

/// Share `<dir>/LOCK` with other readers for the life of the process, so the control
/// never reads a run while brut-breed writes it, and a resume waits until it is done.
fn lock_shared(dir: &Path) -> Result<Option<File>> {
    let path = dir.join("LOCK");
    let opened = File::open(&path).or_else(|_| {
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
    });
    let Ok(lock) = opened else {
        // A read-only copy of a run: nothing can write it either.
        return Ok(None);
    };

    match lock.try_lock_shared() {
        Ok(()) => Ok(Some(lock)),
        Err(TryLockError::WouldBlock) => Err(usage(&format!(
            "a brut-breed process is writing {}",
            dir.display()
        ))),
        Err(TryLockError::Error(e)) => Err(BrutError::Invalid(format!("{}: {e}", path.display()))),
    }
}

fn run() -> Result<i32> {
    let args = parse_args()?;
    let dir = PathBuf::from(args.get("run").ok_or_else(|| usage("--run <dir> is required"))?);
    let dry = args.contains_key("dry-run");
    let jobs: usize = num(&args, "jobs")?.unwrap_or(DEFAULT_JOBS);

    if jobs == 0 {
        return Err(usage("--jobs must be at least 1"));
    }

    let opts = Options {
        cascade: args.get("cascade").cloned(),
        samples: num(&args, "samples")?,
        temperature: num(&args, "temperature")?,
        puzzles: num(&args, "puzzles")?,
    };
    let _lock = if dry { None } else { lock_shared(&dir)? };
    let run = load_run(&dir)?;
    let panel = select_panel(&run.cfg)?;

    check_panel(&panel, &run)?;
    check_run_start(&dir, &panel)?;

    // Every refusal of the plan concerns an option (or the lack of one).
    let (plan, voted) = plan(&run, &opts).map_err(|e| match e {
        BrutError::Invalid(m) => usage(&m),
        e => e,
    })?;

    if dry {
        println!("{}", serde_json::to_string_pretty(&plan)?);
        return Ok(0);
    }

    if !run.snapshot.finished {
        return Err(usage(&format!(
            "{} is not finished (its select phase has not run); the control needs the final champion",
            dir.display()
        )));
    }

    let (pool, filed) = build_pool(&voted, &dir.join(RECORD), &Endpoints::from_env())?;
    let before: usize = filed.iter().map(|f| f.filed()).sum();

    eprintln!(
        "vote-control: {} x {} samples of {} ({} call(s) on file), at most {} calls",
        plan.select_puzzles, plan.samples_per_puzzle, plan.cascade.id, before, plan.max_calls
    );

    let progress = |done: usize, total: usize| {
        let step = (total / 20).max(1);

        if done.is_multiple_of(step) || done == total {
            eprintln!("vote-control: {done}/{total} runs");
        }
    };
    let mut report = vote(&pool, &run, plan, &voted, &panel, jobs, &progress)?;

    if !filed.is_empty() {
        let after: usize = filed.iter().map(|f| f.filed()).sum();

        report.new_calls = Some(after.saturating_sub(before) as u64);
    }

    report.failed_calls = filed.iter().map(|f| f.failed()).sum();

    let json = serde_json::to_string_pretty(&report)?;

    println!("{json}");

    if !report.plan.partial {
        write_atomic(&dir.join(OUTPUT), format!("{json}\n").as_bytes())?;
    }

    let t = &report.totals;

    eprintln!(
        "vote-control: vote {}/{} right, champion {}, sign test p {}",
        t.vote_correct,
        t.puzzles,
        t.champion_correct.map_or("n/a".to_string(), |c| format!("{c}/{}", t.puzzles)),
        t.sign_test_p.map_or("n/a".to_string(), |p| format!("{p:.4}"))
    );

    if report.failed_calls > 0 {
        eprintln!(
            "vote-control: {} call(s) failed and counted as abstentions; run again to retry them",
            report.failed_calls
        );

        return Ok(1);
    }

    Ok(0)
}
