//! brut-mutator: answer the Mutator requests of a paused breeding run with a local model.
//!
//! A run bred with `--llm-every K` stops with exit code 3 when an LLM round lacks answers,
//! and leaves the requests in `<out>/llm/requests/<key>.json`. This program sends every
//! request that has no answer yet, unchanged, to the model of `--tag`, and writes the
//! answer text as `<answers>/<key>.txt`, where `brut-breed-ollama --out <out> --resume`
//! reads it. Every call and its full answer are recorded in `<out>/llm/mutator-calls.jsonl`
//! (failed calls in `mutator-errors.jsonl`); a request on file there is answered from the
//! record without a new call (ADR 0002, contract I7). An existing answer is never
//! overwritten, and nothing is read or written outside `<out>`. The core parses the answers
//! itself and rejects those that are no valid Cascade; this program checks none of them.
//!
//! Usage:
//!   brut-mutator --out <dir> --tag <vllm:<model>|ollama:<model>> [--llm-answers <dir>]
//!                [--max-requests N] [--parallel P]
//!
//! --llm-answers names the directory the run reads its answers from (the value given to
//! brut-breed, default <dir>/llm/answers); it must lie inside <dir>. --max-requests
//! (default 64) caps the requests of one round; the rest waits for the next round.
//! --parallel (default 4) is the number of requests in flight; vLLM decodes them in one
//! batch. The program holds <dir>/LOCK while it runs and asks no further model once
//! <dir>/STOP exists.
//!
//! Environment as for brut-breed-ollama: VLLM_URL (default http://127.0.0.1:8000),
//! OLLAMA_URL (default http://127.0.0.1:11434), both loopback only, and
//! OLLAMA_TIMEOUT_SECS (default 300, per request).
//!
//! Exit codes: 0 every pending request (up to --max-requests) answered, or stopped
//! cleanly by <dir>/STOP; 1 error (a call failed or a request file was refused; the
//! answers already written stay); 2 wrong usage, a path outside the run, or another
//! process holds <dir>/LOCK; 4 nothing to answer.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::path::Path;
use std::process;
use std::time::Duration;

use backend_ollama::mutator::{self, Pending, RunDirs};
use backend_ollama::vllm::{self, VllmBackend};
use backend_ollama::{DEFAULT_BASE, OllamaBackend, Recorded, TAG_PREFIX};
use brutmaschine::backend::ModelBackend;
use brutmaschine::util::{BrutError, Result};

/// Exit code: no request waits for an answer.
const EXIT_NOTHING: i32 = 4;

const KEYS: [&str; 5] = ["out", "tag", "llm-answers", "max-requests", "parallel"];

fn main() {
    let code = match run() {
        Ok(code) => code,
        Err(BrutError::Invalid(m)) if m.starts_with("usage:") => {
            eprintln!("brut-mutator: {m}");
            2
        }
        Err(e) => {
            eprintln!("brut-mutator: {e}");
            1
        }
    };

    process::exit(code);
}

fn usage(m: &str) -> BrutError {
    BrutError::Invalid(format!("usage: {m}"))
}

/// A refusal (wrong path, wrong tag) as wrong usage; other errors stay as they are.
fn as_usage(e: BrutError) -> BrutError {
    match e {
        BrutError::Invalid(m) if !m.starts_with("usage:") => usage(&m),
        other => other,
    }
}

fn run() -> Result<i32> {
    let args = parse_args()?;
    let out = args
        .get("out")
        .ok_or_else(|| usage("--out <dir> is required"))?;
    let tag = args
        .get("tag")
        .ok_or_else(|| usage("--tag vllm:<model> or ollama:<model> is required"))?;
    let max_requests = count(&args, "max-requests", 64, 1_000_000)?;
    let parallel = count(&args, "parallel", 4, 64)?;
    let dirs = RunDirs::new(Path::new(out), args.get("llm-answers").map(Path::new)).map_err(as_usage)?;
    let timeout = Duration::from_secs(
        std::env::var("OLLAMA_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300),
    );

    if tag.starts_with(TAG_PREFIX) {
        let base = std::env::var("OLLAMA_URL").unwrap_or_else(|_| DEFAULT_BASE.to_string());

        round(OllamaBackend::new(tag, &base, timeout), &dirs, max_requests, parallel)
    } else if tag.starts_with(vllm::TAG_PREFIX) {
        let base = std::env::var("VLLM_URL").unwrap_or_else(|_| vllm::DEFAULT_BASE.to_string());

        round(VllmBackend::new(tag, &base, timeout), &dirs, max_requests, parallel)
    } else {
        Err(usage(&format!(
            "--tag {tag:?} names no local model; use {}<model> or {TAG_PREFIX}<model>",
            vllm::TAG_PREFIX
        )))
    }
}

/// One round: answer the pending requests of the run in `dirs` with `backend`.
fn round<B: ModelBackend>(
    backend: Result<B>,
    dirs: &RunDirs,
    max_requests: usize,
    parallel: usize,
) -> Result<i32> {
    // The allow-list and the loopback check refuse before any call (I3, I6).
    let backend = backend.map_err(as_usage)?;
    let _lock = mutator::lock_run(dirs)?;
    let stop_file = dirs.stop_file();

    if stop_file.exists() {
        println!(
            "brut-mutator: {} exists; no model asked. Delete it to go on.",
            stop_file.display()
        );

        return Ok(0);
    }

    let scan = mutator::scan(dirs)?;

    for r in &scan.refused {
        eprintln!("brut-mutator: refused {}: {}", r.path.display(), r.reason);
    }

    if scan.pending.is_empty() {
        println!(
            "brut-mutator: nothing to answer in {} ({} request(s) have an answer in {})",
            dirs.requests.display(),
            scan.answered,
            dirs.answers.display()
        );

        return Ok(if scan.refused.is_empty() { EXIT_NOTHING } else { 1 });
    }

    let recorded = Recorded::open(backend, &dirs.record)?;
    let waiting = scan.pending.len();
    let todo: Vec<Pending> = scan.pending.into_iter().take(max_requests).collect();
    let replays = todo.iter().filter(|p| recorded.contains(&p.request)).count();

    println!(
        "brut-mutator: {waiting} request(s) wait; {} go to {} ({replays} answered from {}), {parallel} at a time",
        todo.len(),
        recorded.name(),
        dirs.record.display()
    );

    let stop = || stop_file.exists();
    let r = mutator::answer(&recorded, &dirs.answers, &todo, parallel, &stop);

    for key in &r.cut {
        eprintln!(
            "brut-mutator: {key}: the answer reached max_tokens; written as it came, the core parses or rejects it"
        );
    }

    for key in &r.kept {
        println!("brut-mutator: {key}: an answer appeared meanwhile and stays as it was");
    }

    for (key, e) in &r.failed {
        eprintln!("brut-mutator: {key}: no answer: {e}");
    }

    println!(
        "brut-mutator: {} answer(s) written to {}",
        r.written.len(),
        dirs.answers.display()
    );

    if r.skipped > 0 {
        println!(
            "brut-mutator: stopped by {}; {} request(s) not asked",
            stop_file.display(),
            r.skipped
        );
    }

    if waiting > todo.len() {
        println!(
            "brut-mutator: {} request(s) wait for the next round (--max-requests {max_requests})",
            waiting - todo.len()
        );
    }

    Ok(if r.failed.is_empty() && scan.refused.is_empty() { 0 } else { 1 })
}

/// The number under `--key`, `default` if absent, refused outside 1..=`max`.
fn count(args: &BTreeMap<String, String>, key: &str, default: usize, max: usize) -> Result<usize> {
    let Some(v) = args.get(key) else {
        return Ok(default);
    };

    match v.parse::<usize>() {
        Ok(n) if (1..=max).contains(&n) => Ok(n),
        _ => Err(usage(&format!("--{key} needs a number from 1 to {max}, got {v:?}"))),
    }
}

/// `--key value` pairs, each key at most once.
fn parse_args() -> Result<BTreeMap<String, String>> {
    let mut map = BTreeMap::new();
    let mut it = std::env::args().skip(1);

    while let Some(tok) = it.next() {
        let key = tok
            .strip_prefix("--")
            .ok_or_else(|| usage(&format!("expected --key, got {tok}")))?;

        if !KEYS.contains(&key) {
            return Err(usage(&format!("unknown option --{key}")));
        }

        let val = it
            .next()
            .ok_or_else(|| usage(&format!("--{key} needs a value")))?;

        if map.insert(key.to_string(), val).is_some() {
            return Err(usage(&format!("--{key} given twice")));
        }
    }

    Ok(map)
}
