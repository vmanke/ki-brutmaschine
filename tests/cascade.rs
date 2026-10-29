//! Stage 2c tests (ADR 0005): hygiene of `src/cascade/` and end-to-end runs of
//! `brut-breed` with the simulated models: determinism across processes, kill switch and
//! resume, the hash-chained log, the pause for LLM answers, and promotion into the
//! candidate gate.
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BREED: &str = env!("CARGO_BIN_EXE_brut-breed");
const INGEST: &str = env!("CARGO_BIN_EXE_brut-ingest");

/// Strings `src/cascade/**` must never contain: the module is pure computation.
const IO_MARKERS: [&str; 10] = [
    "std::fs",
    "std::net",
    "std::process",
    "std::env",
    "File::",
    "Command",
    "TcpStream",
    "include_str",
    "include_bytes",
    "Instant",
];

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn cascade_sources() -> Vec<(String, String)> {
    let dir = crate_path("src/cascade");
    let mut out: Vec<(String, String)> = std::fs::read_dir(&dir)
        .expect("src/cascade")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .map(|p| {
            let text = std::fs::read_to_string(&p).expect("read");

            (p.display().to_string(), text)
        })
        .collect();

    out.sort();

    out
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

    // Changing one byte of the log breaks the chain.
    let tampered = ea.replacen("\"solved\":", "\"solved\": ", 1);

    assert_ne!(tampered, ea);
    std::fs::write(a.join("events.jsonl"), tampered).expect("tamper");
    assert_eq!(
        breed(&["--out", a.to_str().expect("path"), "--verify"]).status.code(),
        Some(1)
    );
}

#[test]
fn kill_switch_then_resume_equals_a_straight_run() {
    let straight = fresh("straight");
    let split = fresh("split");

    ok(&small_run(&straight, &[]));

    // With STOP present the run stops before committing generation 0: no log, no report.
    std::fs::create_dir_all(&split).expect("mkdir");
    std::fs::write(split.join("STOP"), "").expect("stop");
    ok(&small_run(&split, &[]));
    assert!(!split.join("events.jsonl").exists());
    assert!(!split.join("report.json").exists());

    // Nothing was committed, so the same directory starts afresh; then a short run is
    // resumed with more generations.
    std::fs::remove_file(split.join("STOP")).expect("unstop");
    ok(&small_run(&split, &["--generations", "1"]));
    ok(&breed(&[
        "--out",
        split.to_str().expect("path"),
        "--resume",
        "--generations",
        "3",
    ]));

    // Resume cut the log back to the last checkpoint (dropping the finalist lines of the
    // short run), so every generation event matches the straight run. The run-start
    // event differs only in the recorded number of generations.
    let generations = |dir: &Path| -> Vec<String> {
        std::fs::read_to_string(dir.join("events.jsonl"))
            .expect("log")
            .lines()
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).expect("json");

                v["event"].clone()
            })
            .filter(|e| e["kind"] != "run_start")
            .map(|e| e.to_string())
            .collect()
    };

    assert_eq!(generations(&straight), generations(&split));
    assert_eq!(
        std::fs::read_to_string(straight.join("report.json")).expect("a"),
        std::fs::read_to_string(split.join("report.json")).expect("b")
    );
}

#[test]
fn llm_round_pauses_for_answers_and_promotion_passes_the_gate() {
    let dir = fresh("llm");
    let out = small_run(&dir, &["--llm-every", "2", "--llm-children", "2"]);

    assert_eq!(out.status.code(), Some(3), "{}", String::from_utf8_lossy(&out.stderr));

    let requests: Vec<PathBuf> = std::fs::read_dir(dir.join("llm").join("requests"))
        .expect("requests")
        .map(|e| e.expect("entry").path())
        .collect();

    assert_eq!(requests.len(), 2);

    // Answer every request with a valid Cascade (a seed with a larger thinking budget,
    // distinct per request) wrapped in prose, as a Claude session would.
    let answers = dir.join("llm").join("answers");

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

    let resumed = breed(&["--out", dir.to_str().expect("path"), "--resume"]);

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
