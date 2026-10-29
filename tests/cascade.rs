//! Stage 2c tests (ADR 0005): hygiene of `src/cascade/` and end-to-end runs of
//! `brut-breed` with the simulated models: determinism across processes, kill switch and
//! resume, the hash-chained log, the pause for LLM answers, and promotion into the
//! candidate gate.
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

const BREED: &str = env!("CARGO_BIN_EXE_brut-breed");
const INGEST: &str = env!("CARGO_BIN_EXE_brut-ingest");

/// Strings `src/cascade/**` must never contain: the module is pure computation (no
/// files, network, processes, environment or clock, also not through crate helpers).
const IO_MARKERS: [&str; 18] = [
    "std::fs",
    "std::net",
    "std::process",
    "std::env",
    "std::io",
    "std::time",
    "File::",
    "Command",
    "TcpStream",
    "include_str",
    "include_bytes",
    "Instant",
    "SystemTime",
    "env!(",
    "option_env!(",
    "now_unix_ms",
    "write_atomic",
    "parse_env_file",
];

/// Modules a grouped `use std::{..}` must not name.
const IO_MODULES: [&str; 6] = ["fs", "net", "process", "env", "io", "time"];

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// Every `.rs` file under `src/cascade`, subdirectories included.
fn cascade_sources() -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut pending = vec![crate_path("src/cascade")];

    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).expect("src/cascade") {
            let path = entry.expect("entry").path();

            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|x| x == "rs") {
                let text = std::fs::read_to_string(&path).expect("read");

                out.push((path.display().to_string(), text));
            }
        }
    }

    out.sort();

    out
}

/// First path segment of every item in each `std::{...}` group of whitespace-free
/// source (as in tests/learner_security.rs).
fn grouped_std_imports(compact: &str) -> Vec<String> {
    let mut out = Vec::new();

    for (start, marker) in compact.match_indices("std::{") {
        let body = &compact[start + marker.len()..];
        let mut depth = 0usize;
        let mut item = String::new();

        for c in body.chars() {
            match c {
                '{' => {
                    depth += 1;
                    item.push(c);
                }
                '}' if depth == 0 => break,
                '}' => {
                    depth -= 1;
                    item.push(c);
                }
                ',' if depth == 0 => {
                    out.push(first_segment(&item));
                    item.clear();
                }
                _ => item.push(c),
            }
        }

        out.push(first_segment(&item));
    }

    out
}

/// First path segment of one item in a `use` group, e.g. "fs" for "fs::File".
fn first_segment(item: &str) -> String {
    item.split("::").next().unwrap_or("").to_string()
}

#[test]
fn cascade_module_is_pure_and_forbids_unsafe() {
    let mut violations = Vec::new();

    for (path, text) in cascade_sources() {
        let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();

        for marker in IO_MARKERS {
            if text.contains(marker) || compact.contains(&marker.replace(' ', "")) {
                violations.push(format!("{path}: mentions {marker}"));
            }
        }

        for module in grouped_std_imports(&compact) {
            if IO_MODULES.contains(&module.as_str()) {
                violations.push(format!("{path}: imports std::{module} in a group"));
            }
        }

        let first = text
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty() && !l.starts_with("//"));

        if first != Some("#![forbid(unsafe_code)]") {
            violations.push(format!("{path}: first code line is not #![forbid(unsafe_code)]"));
        }

        if text.contains("unsafe {") || compact.contains("unsafe{") {
            violations.push(format!("{path}: unsafe block"));
        }
    }

    assert!(violations.is_empty(), "{violations:#?}");
}

/// A fresh, empty directory under the test target directory.
fn fresh(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cascade").join(name);

    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("clean");
    }

    std::fs::create_dir_all(dir.parent().expect("parent")).expect("mkdir");

    dir
}

fn breed(args: &[&str]) -> Output {
    Command::new(BREED).args(args).output().expect("run brut-breed")
}

fn ok(out: &Output) {
    assert!(
        out.status.success(),
        "exit {:?}\nstdout:\n{}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

const SMALL: [&str; 10] = [
    "--seed",
    "5",
    "--generations",
    "3",
    "--children",
    "4",
    "--search",
    "12",
    "--select",
    "12",
];

fn small_run(dir: &Path, extra: &[&str]) -> Output {
    let out = dir.to_str().expect("utf-8 path");
    let mut args = vec!["--out", out];

    args.extend_from_slice(&SMALL);
    args.extend_from_slice(extra);

    breed(&args)
}

#[test]
fn breed_runs_are_reproducible_across_processes_and_verifiable() {
    let a = fresh("repro-a");
    let b = fresh("repro-b");

    ok(&small_run(&a, &[]));
    ok(&small_run(&b, &[]));

    let ra = std::fs::read_to_string(a.join("report.json")).expect("report a");
    let rb = std::fs::read_to_string(b.join("report.json")).expect("report b");
    let ea = std::fs::read_to_string(a.join("events.jsonl")).expect("events a");
    let eb = std::fs::read_to_string(b.join("events.jsonl")).expect("events b");

    assert_eq!(ra, rb);
    assert_eq!(ea, eb);

    ok(&breed(&["--out", a.to_str().expect("path"), "--verify"]));

    // A fresh run into a used directory is refused as wrong usage.
    assert_eq!(small_run(&a, &[]).status.code(), Some(2));

    // Any edit fails verification: a changed value breaks the hash chain, and even a
    // whitespace-only edit breaks the canonical form.
    let verify_fails = |text: &str| {
        std::fs::write(a.join("events.jsonl"), text).expect("tamper");
        breed(&["--out", a.to_str().expect("path"), "--verify"]).status.code() == Some(1)
    };
    let value = ea.replacen("\"puzzles\":12", "\"puzzles\":13", 1);
    let spacing = ea.replacen("\"solved\":", "\"solved\": ", 1);

    assert_ne!(value, ea);
    assert_ne!(spacing, ea);
    assert!(verify_fails(&value), "a changed value must fail");
    assert!(verify_fails(&spacing), "a whitespace edit must fail");

    // A cut log is a valid chain, but it no longer reaches the checkpoint's anchor; a
    // torn last line (a write the process did not finish) fails too.
    let head: String = ea.lines().take(10).map(|l| format!("{l}\n")).collect();
    let torn = format!("{ea}{{\"seq\":");

    assert!(verify_fails(&head), "a cut log must fail");
    assert!(verify_fails(&torn), "a torn last line must fail");

    // The report of a finished run is recomputed from the checkpoint: a changed headline
    // number fails as well as changed finalists.
    std::fs::write(a.join("events.jsonl"), &ea).expect("restore");

    let checked = breed(&["--out", a.to_str().expect("path"), "--verify"]);

    ok(&checked);
    assert!(String::from_utf8_lossy(&checked.stdout).contains("report recomputed and equal"));

    let mut r = report(&a);

    r["best_finalist_select_rate"] = serde_json::json!(0.99);
    std::fs::write(a.join("report.json"), serde_json::to_string_pretty(&r).expect("json")).expect("tamper");
    assert_eq!(breed(&["--out", a.to_str().expect("path"), "--verify"]).status.code(), Some(1));
    std::fs::write(a.join("report.json"), &ra).expect("restore report");
    tamper_report(&a);
    assert_eq!(breed(&["--out", a.to_str().expect("path"), "--verify"]).status.code(), Some(1));

    // A directory without a run is wrong usage, not an empty success.
    let none = fresh("repro-none");

    assert_eq!(breed(&["--out", none.to_str().expect("path"), "--verify"]).status.code(), Some(2));
}

#[test]
fn verify_takes_the_stop_reason_from_the_checkpoint_not_the_report() {
    // A finished run ended either after its last generation or because the next one no
    // longer fit the call budget; the checkpoint shows which. A report that relabels one
    // as the other fails --verify and --promote.
    let full = fresh("reason-full");
    let capped = fresh("reason-capped");

    ok(&small_run(&full, &[]));
    ok(&small_run(&capped, &["--max-calls", "400"]));

    for (dir, honest, forged) in [
        (&full, "all generations done", "call budget reached"),
        (&capped, "call budget reached", "all generations done"),
    ] {
        let path = dir.to_str().expect("path");
        let text = std::fs::read_to_string(dir.join("report.json")).expect("report");
        let mut r = report(dir);

        assert_eq!(r["stop_reason"], honest);
        ok(&breed(&["--out", path, "--verify"]));

        r["stop_reason"] = serde_json::json!(forged);
        std::fs::write(dir.join("report.json"), serde_json::to_string_pretty(&r).expect("json"))
            .expect("forge report");

        for mode in ["--verify", "--promote"] {
            let refused = breed(&["--out", path, mode]);

            assert_eq!(refused.status.code(), Some(1), "{mode}: {honest} relabelled as {forged}");
            assert!(String::from_utf8_lossy(&refused.stderr).contains("stop reason"), "{mode}");
        }

        std::fs::write(dir.join("report.json"), text).expect("restore report");
        ok(&breed(&["--out", path, "--verify"]));
    }
}

/// The events of a run's log, as JSON values.
fn events(dir: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(dir.join("events.jsonl"))
        .expect("log")
        .lines()
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).expect("json");

            v["event"].clone()
        })
        .collect()
}

fn count(events: &[serde_json::Value], kind: &str) -> usize {
    events.iter().filter(|e| e["kind"] == kind).count()
}

/// What a run decided: every Cascade and judgment, plus the finalists after the last
/// generation (earlier finalist sets are history), without cumulative call counts.
fn lineage(dir: &Path) -> Vec<String> {
    let all = events(dir);
    let last = all
        .iter()
        .rposition(|e| e["kind"] == "generation")
        .map_or(0, |i| i + 1);
    let mut out: Vec<String> = all
        .iter()
        .filter(|e| e["kind"] == "born" || e["kind"] == "compared")
        .map(|e| e.to_string())
        .collect();

    out.extend(all[last..].iter().filter(|e| e["kind"] == "finalist").map(|e| e.to_string()));

    out
}

fn report(dir: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(dir.join("report.json")).expect("report");

    serde_json::from_str(&text).expect("json")
}

/// Raise the first finalist's select score in report.json by one.
fn tamper_report(dir: &Path) {
    let mut r = report(dir);
    let solved = r["finalists"][0]["select_solved"].as_u64().expect("select_solved");

    r["finalists"][0]["select_solved"] = serde_json::json!(solved + 1);
    std::fs::write(dir.join("report.json"), serde_json::to_string_pretty(&r).expect("json"))
        .expect("tamper report");
}

#[test]
fn kill_switch_then_resume_equals_a_straight_run() {
    let straight = fresh("straight");
    let split = fresh("split");
    let path = split.to_str().expect("path");

    ok(&small_run(&straight, &[]));

    // With STOP present the run stops before committing generation 0: a checkpoint
    // exists (written before generation 0), but no event and no report. The message
    // says to delete STOP before resuming.
    std::fs::create_dir_all(&split).expect("mkdir");
    std::fs::write(split.join("STOP"), "").expect("stop");

    let stopped = small_run(&split, &[]);

    ok(&stopped);
    assert!(String::from_utf8_lossy(&stopped.stdout).contains("delete"));
    assert!(split.join("snapshot.json").exists());
    assert!(!split.join("events.jsonl").exists());
    assert!(!split.join("report.json").exists());

    // A fresh start into that directory is refused; resume runs a short run to its end,
    // and a second resume continues it with more generations.
    std::fs::remove_file(split.join("STOP")).expect("unstop");
    assert_eq!(small_run(&split, &[]).status.code(), Some(2));
    ok(&breed(&["--out", path, "--resume", "--generations", "1"]));

    let short_finalists = count(&events(&split), "finalist");

    assert!(short_finalists > 0);
    ok(&breed(&["--out", path, "--resume", "--generations", "3"]));

    // The short run's finalists stay in the log as history. Every Cascade, every
    // judgment and the final finalists match the straight run. The short run's select
    // phase really spent calls; answers it can reuse are not paid for again.
    assert_eq!(lineage(&straight), lineage(&split));
    assert_eq!(
        count(&events(&split), "finalist"),
        short_finalists + count(&events(&straight), "finalist")
    );

    let (mut a, mut b) = (report(&straight), report(&split));

    assert!(b["calls_used"].as_u64() >= a["calls_used"].as_u64());

    a["calls_used"] = serde_json::Value::Null;
    b["calls_used"] = serde_json::Value::Null;
    assert_eq!(a, b);
    ok(&breed(&["--out", path, "--verify"]));
}

#[test]
fn a_finished_run_resumes_without_change_and_a_refused_resume_touches_nothing() {
    let dir = fresh("finished");
    let path = dir.to_str().expect("path");

    ok(&small_run(&dir, &[]));

    let log = std::fs::read(dir.join("events.jsonl")).expect("log");
    let rep = std::fs::read(dir.join("report.json")).expect("report");
    let config = std::fs::read_to_string(dir.join("config.json")).expect("config");

    // Options that define the experiment cannot change on resume.
    assert_eq!(breed(&["--out", path, "--resume", "--seed", "9"]).status.code(), Some(2));
    assert_eq!(breed(&["--out", path, "--resume", "--no-pause"]).status.code(), Some(2));
    assert_eq!(breed(&["--out", path, "--verify", "--seed", "9"]).status.code(), Some(2));

    // A resume refused for changed settings leaves every file as it was.
    let edited = config.replacen("\"judge_weight\": 0.2", "\"judge_weight\": 0.3", 1);

    assert_ne!(edited, config);
    std::fs::write(dir.join("config.json"), edited).expect("edit");
    assert_eq!(breed(&["--out", path, "--resume"]).status.code(), Some(1));
    assert_eq!(std::fs::read(dir.join("events.jsonl")).expect("log"), log);
    assert_eq!(std::fs::read(dir.join("report.json")).expect("report"), rep);
    std::fs::write(dir.join("config.json"), &config).expect("restore");

    // Resuming the finished run gives the same report with no new calls and no new
    // events (its finalists are logged already).
    ok(&breed(&["--out", path, "--resume"]));
    assert_eq!(std::fs::read(dir.join("events.jsonl")).expect("log"), log);
    assert_eq!(std::fs::read(dir.join("report.json")).expect("report"), rep);

    // A finished run needs no call to report again, so even max_calls 1 (below the calls
    // already spent) gives the same report and the same log.
    ok(&breed(&["--out", path, "--resume", "--max-calls", "1"]));
    assert_eq!(std::fs::read(dir.join("events.jsonl")).expect("log"), log);
    assert_eq!(std::fs::read(dir.join("report.json")).expect("report"), rep);

    // Promotion needs a report that matches the log.
    ok(&breed(&["--out", path, "--promote"]));
    tamper_report(&dir);
    assert_eq!(breed(&["--out", path, "--promote"]).status.code(), Some(1));
}

/// Next generation and reserved calls of the stored checkpoint.
fn progress(dir: &Path) -> (u64, u64) {
    std::fs::read_to_string(dir.join("snapshot.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .map(|v| {
            let s = &v["snapshot"];

            (
                s["next_generation"].as_u64().unwrap_or(0),
                s["reserved"].as_u64().unwrap_or(0),
            )
        })
        .unwrap_or((0, 0))
}

#[test]
fn a_resume_refused_by_the_select_budget_changes_nothing() {
    // A short generation 0 and a long select phase, so the kill lands in the select phase.
    let dir = fresh("select-kill");
    let path = dir.to_str().expect("path");
    let mut child = Command::new(BREED)
        .args(["--out", path, "--generations", "0", "--search", "12", "--select", "20000"])
        .args(["--max-calls", "1000000"])
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn");

    assert!(
        wait_for(120, || {
            let (next, reserved) = progress(&dir);

            next == 1 && reserved > 0
        }),
        "the select phase reserved its first Cascade"
    );

    child.kill().expect("kill");
    child.wait().expect("wait");

    let (_, reserved) = progress(&dir);
    let read = |name: &str| std::fs::read(dir.join(name)).expect(name);
    let before = ["snapshot.json", "config.json", "events.jsonl"].map(read);
    let calls = serde_json::from_slice::<serde_json::Value>(&before[0]).expect("json")["snapshot"]["calls"]
        .as_u64()
        .expect("calls");

    assert!(reserved > 0, "killed inside the select phase");

    // The cap leaves no room for the scoring still to do: refused before any write.
    let cap = (calls + reserved).to_string();

    assert_eq!(breed(&["--out", path, "--resume", "--max-calls", &cap]).status.code(), Some(1));
    assert_eq!(["snapshot.json", "config.json", "events.jsonl"].map(read), before);
    assert!(!dir.join("report.json").exists());

    // With the recorded cap, the resume scores the rest and finishes.
    ok(&breed(&["--out", path, "--resume"]));
    assert!(dir.join("report.json").exists());
    ok(&breed(&["--out", path, "--verify"]));
}

#[test]
fn a_resume_drops_an_uncommitted_log_tail_even_without_new_events() {
    let finished = fresh("tail-finished");
    let longer = fresh("tail-longer");

    ok(&small_run(&finished, &[]));
    ok(&small_run(&longer, &[]));
    ok(&breed(&["--out", longer.to_str().expect("path"), "--resume", "--generations", "4"]));

    // The state a crash leaves when it hits after a generation was logged but before its
    // checkpoint: the longer log (with a torn last line) next to the finished run's
    // checkpoint and settings. The resume has nothing to run and its finalists are
    // logged already, so it returns no events; it must still cut the log back.
    let log = std::fs::read_to_string(finished.join("events.jsonl")).expect("log");
    let tail = std::fs::read_to_string(longer.join("events.jsonl")).expect("longer log");

    assert!(tail.starts_with(&log) && tail.len() > log.len());

    for name in ["snapshot.json", "config.json", "report.json"] {
        std::fs::copy(finished.join(name), longer.join(name)).expect("copy");
    }

    std::fs::write(longer.join("events.jsonl"), format!("{tail}{{\"seq\":")).expect("torn");

    let path = longer.to_str().expect("path");

    assert_eq!(breed(&["--out", path, "--verify"]).status.code(), Some(1), "torn line");
    ok(&breed(&["--out", path, "--resume"]));
    assert_eq!(std::fs::read_to_string(longer.join("events.jsonl")).expect("log"), log);
    assert_eq!(report(&longer), report(&finished));
    ok(&breed(&["--out", path, "--verify"]));
    ok(&breed(&["--out", path, "--promote"]));
}

/// Wait until `cond` holds, polling every 10 ms, for at most `secs` seconds.
fn wait_for(secs: u64, mut cond: impl FnMut() -> bool) -> bool {
    let end = Instant::now() + Duration::from_secs(secs);

    while Instant::now() < end {
        if cond() {
            return true;
        }

        std::thread::sleep(Duration::from_millis(10));
    }

    false
}

/// Calls the stored checkpoint reserves for work in flight (0 when there is none).
fn reserved(dir: &Path) -> u64 {
    std::fs::read_to_string(dir.join("snapshot.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v["snapshot"]["reserved"].as_u64())
        .unwrap_or(0)
}

#[test]
fn a_killed_run_counts_its_reservation_and_holds_the_lock() {
    // One long generation 0 (the four seeds on a large panel), so the kill lands in it.
    let long = |dir: &Path| -> Vec<String> {
        let mut args = vec!["--out".to_string(), dir.display().to_string()];

        for a in ["--generations", "0", "--search", "20000", "--select", "12", "--max-calls", "1000000"] {
            args.push(a.to_string());
        }

        args
    };
    let straight = fresh("kill-straight");
    let killed = fresh("kill");
    let path = killed.to_str().expect("path");

    ok(&Command::new(BREED).args(long(&straight)).output().expect("straight run"));

    let mut child = Command::new(BREED)
        .args(long(&killed))
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn");

    assert!(wait_for(120, || reserved(&killed) > 0), "reservation stored before the first call");

    // While it runs, a second process on the same directory is refused.
    let second = breed(&["--out", path, "--verify"]);

    assert_eq!(second.status.code(), Some(2), "{}", String::from_utf8_lossy(&second.stderr));

    child.kill().expect("kill");
    child.wait().expect("wait");

    // The resume counts the killed attempt's worst case as spent, then repeats the
    // generation exactly.
    let r = reserved(&killed);

    assert!(r > 0, "killed inside the run");
    ok(&breed(&["--out", path, "--resume"]));

    let calls = |dir: &Path| report(dir)["calls_used"].as_u64().expect("calls");

    assert_eq!(calls(&killed), calls(&straight) + r);
    assert_eq!(lineage(&straight), lineage(&killed));
    ok(&breed(&["--out", path, "--verify"]));
}

#[test]
fn llm_round_pauses_for_answers_and_promotion_passes_the_gate() {
    let dir = fresh("llm");
    let answers = dir.join("my-answers");
    let answers_arg = answers.to_str().expect("path");
    let out = small_run(
        &dir,
        &["--llm-every", "2", "--llm-children", "2", "--llm-answers", answers_arg],
    );

    assert_eq!(out.status.code(), Some(3), "{}", String::from_utf8_lossy(&out.stderr));

    // The printed resume command names the answers directory.
    let printed = String::from_utf8_lossy(&out.stdout).to_string();

    assert!(printed.contains(&format!("--resume --llm-answers {answers_arg}")), "{printed}");

    let requests: Vec<PathBuf> = std::fs::read_dir(dir.join("llm").join("requests"))
        .expect("requests")
        .map(|e| e.expect("entry").path())
        .collect();

    assert_eq!(requests.len(), 2);

    // Answer every request with a valid Cascade (a seed with a larger thinking budget,
    // distinct per request) wrapped in prose, as a Claude session would.
    std::fs::create_dir_all(&answers).expect("answers dir");

    for (i, path) in requests.iter().enumerate() {
        let key = path.file_stem().and_then(|s| s.to_str()).expect("key");
        let think = 512 * (i + 1);
        let cascade = format!(
            r#"{{"stages":[{{"workers":[{{"model":"sim:large","sections":[
                {{"name":"role","text":"You solve grid puzzles."}},
                {{"name":"strategy","text":"Look for mirror images, rotations, transposition, gravity and colour changes; combine two steps."}},
                {{"name":"format","text":"Answer with the line ANSWER followed by the output grid."}}],
              "traits":{{"think_tokens":{think},"temperature":0.0,"answer_tokens":512,"self_check":true}}}}]}}]}}"#
        );

        std::fs::write(answers.join(format!("{key}.txt")), format!("New Cascade:\n```json\n{cascade}\n```\n"))
            .expect("answer");
    }

    let resumed = breed(&["--out", dir.to_str().expect("path"), "--resume", "--llm-answers", answers_arg]);

    ok(&resumed);

    let log = std::fs::read_to_string(dir.join("events.jsonl")).expect("log");
    let accepted = log
        .lines()
        .filter(|l| l.contains("\"kind\":\"llm_exchange\"") && l.contains("\"accepted\":true"))
        .count();

    assert_eq!(accepted, 2, "both recorded answers were used");

    ok(&breed(&["--out", dir.to_str().expect("path"), "--promote"]));

    let promoted = dir.join("promote").join("candidates").join("cascades");
    let files = std::fs::read_dir(&promoted).expect("promoted").count();

    assert_eq!(files, 2);

    let gate = Command::new(INGEST)
        .args(["--cascades", promoted.to_str().expect("path")])
        .output()
        .expect("run brut-ingest");

    ok(&gate);
}
