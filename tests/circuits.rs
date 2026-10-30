//! Step 1 integration tests (ADR 0006, plan (g) "Integration"): circuit runs of the
//! `brut-breed` binary end to end, the rediscovery count against the logged answers, the
//! source scan that keeps the simulated designer away from the target library, the pinned
//! calibration of the positive control and two of its routes at the ceiling of the
//! Traits, and a cross-check of the exact checker against the floating-point simulator of
//! `src/quantum.rs`, which shares no code with it.
//!
//! No test here runs a counted seed of the experiment (1 to 10) or a pilot seed (101,
//! 102). The calibration uses the search panel the pilot seed 101 would draw, without a
//! run.
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::FRAC_PI_4;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use brutmaschine::backend::{ModelBackend, StopReason};
use brutmaschine::cascade::arena::{BreedConfig, ModelPool, run_cascade_in, worker_request_with};
use brutmaschine::cascade::circuit::{
    Answered, CircuitDomain, CircuitTask, LIBRARY, Rediscovery, grade_circuit, library_target,
    rediscovery, verified,
};
use brutmaschine::cascade::circuit_genes::{CIRCUIT_PHRASES, circuit_seed_cascades, no_gate_lists};
use brutmaschine::cascade::circuit_text::{
    CIRCUIT_SELF_CHECK_TEXT, parse_circuit_answer, render_circuit_solve_prompt,
};
use brutmaschine::cascade::designer::{SimDesigner, knowledge};
use brutmaschine::cascade::domain::{CircuitConfig, Domain, DomainConfig};
use brutmaschine::cascade::genome::{
    Cascade, CascadeLimits, Section, Stage, Traits, Worker, adapt_seeds,
};
use brutmaschine::cascade::qubit::{self, Circuit, Dressing, Gate};
use brutmaschine::quantum;
use brutmaschine::util::derive_seed;

const BREED: &str = env!("CARGO_BIN_EXE_brut-breed");

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// A fresh, empty directory under the test target directory.
fn fresh(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("circuits")
        .join(name);

    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("clean");
    }

    std::fs::create_dir_all(dir.parent().expect("parent")).expect("mkdir");

    dir
}

fn breed(args: &[&str]) -> Output {
    Command::new(BREED)
        .args(args)
        .output()
        .expect("run brut-breed")
}

fn ok(out: &Output) {
    assert!(
        out.status.success(),
        "exit {:?}\nstdout:\n{}\nstderr:\n{}",
        out.status.code(),
        stdout(out),
        stderr(out)
    );
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

/// A small circuit run: every verified target on one dressed task each.
const SMALL: [&str; 12] = [
    "--domain",
    "circuits",
    "--seed",
    "31",
    "--generations",
    "3",
    "--children",
    "4",
    "--search",
    "11",
    "--select",
    "11",
];

/// A longer circuit run, still well under a second, in which bred Cascades solve targets
/// the seed Cascades miss.
const RICH: [&str; 12] = [
    "--domain",
    "circuits",
    "--seed",
    "33",
    "--generations",
    "20",
    "--children",
    "6",
    "--search",
    "11",
    "--select",
    "11",
];

/// A fresh run with the settings `settings` into `dir`.
fn run(dir: &Path, settings: &[&str]) -> Output {
    let out = dir.to_str().expect("utf-8 path");
    let mut args = vec!["--out", out];

    args.extend_from_slice(settings);

    breed(&args)
}

fn verify(dir: &Path) -> Output {
    breed(&["--out", dir.to_str().expect("path"), "--verify"])
}

/// The bytes of a file of a run.
fn read(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
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

/// The events of one kind, in order.
fn of_kind(events: &[serde_json::Value], kind: &str) -> Vec<serde_json::Value> {
    events
        .iter()
        .filter(|e| e["kind"] == kind)
        .cloned()
        .collect()
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

    out.extend(
        all[last..]
            .iter()
            .filter(|e| e["kind"] == "finalist")
            .map(|e| e.to_string()),
    );

    out
}

fn json(dir: &Path, name: &str) -> serde_json::Value {
    serde_json::from_slice(&read(dir, name)).expect("json")
}

/// The files that record a finished run.
const RECORD: [&str; 5] = [
    "config.json",
    "events.jsonl",
    "snapshot.json",
    "report.json",
    "rediscovery.json",
];

/// The settings of the SMALL run as a config file in `dir`.
fn small_config_file(dir: &Path) -> PathBuf {
    let file = dir.join("circuits.json");
    let cfg = BreedConfig {
        seed: 31,
        generations: 3,
        children_per_generation: 4,
        search_puzzles: 11,
        select_puzzles: 11,
        domain: DomainConfig::Circuits(CircuitConfig::default()),
        ..BreedConfig::default()
    };

    std::fs::create_dir_all(dir).expect("mkdir");
    std::fs::write(&file, serde_json::to_string_pretty(&cfg).expect("json")).expect("write");

    file
}

#[test]
fn a_circuit_run_is_reproducible_and_passes_verify() {
    let flag = fresh("repro-flag");
    let files = fresh("repro-file");
    let file = small_config_file(&files);
    let from_file = files.join("run");
    let first = run(&flag, &SMALL);

    ok(&first);
    assert!(
        stdout(&first).contains("rediscovery.json"),
        "{}",
        stdout(&first)
    );

    // The same settings by a config file, in another process, give the same run, byte for
    // byte.
    ok(&breed(&[
        "--out",
        from_file.to_str().expect("path"),
        "--config",
        file.to_str().expect("path"),
    ]));

    for name in RECORD {
        assert_eq!(read(&flag, name), read(&from_file, name), "{name}");
    }

    // The settings record the circuit domain, and the run start names the search tasks.
    let config = json(&flag, "config.json");
    let start = &events(&flag)[0];
    let defaults = serde_json::json!({ "targets": [], "dress": true, "max_gates": 64 });

    assert_eq!(
        config["domain"],
        serde_json::json!({ "circuits": defaults })
    );
    assert_eq!(start["kind"], "run_start");
    assert_eq!(start["search"].as_array().map(Vec::len), Some(11));

    // A finished circuit run recomputes both its report and its rediscovery count.
    let checked = verify(&flag);

    ok(&checked);
    assert!(
        stdout(&checked).contains("report recomputed and equal"),
        "{}",
        stdout(&checked)
    );
    assert!(
        stdout(&checked).contains("rediscovery recomputed and equal"),
        "{}",
        stdout(&checked)
    );

    // A fresh run into a used directory is refused as wrong usage.
    assert_eq!(run(&flag, &SMALL).status.code(), Some(2));
}

#[test]
fn a_stopped_and_resumed_circuit_run_equals_a_straight_one() {
    let straight = fresh("straight");
    let split = fresh("split");
    let path = split.to_str().expect("path");

    ok(&run(&straight, &RICH));

    // With STOP present the run stops before committing generation 0: a checkpoint
    // exists, but no event, no report and no rediscovery count. --verify says that the
    // count is not checked yet.
    std::fs::create_dir_all(&split).expect("mkdir");
    std::fs::write(split.join("STOP"), "").expect("stop");

    let stopped = run(&split, &RICH);

    ok(&stopped);
    assert!(stdout(&stopped).contains("delete"), "{}", stdout(&stopped));
    assert!(split.join("snapshot.json").exists());

    for name in ["events.jsonl", "report.json", "rediscovery.json"] {
        assert!(!split.join(name).exists(), "{name}");
    }

    std::fs::remove_file(split.join("STOP")).expect("unstop");

    let unfinished = verify(&split);

    ok(&unfinished);
    assert!(
        stdout(&unfinished).contains("rediscovery.json not checked"),
        "{}",
        stdout(&unfinished)
    );

    // A short run to its end, then a resume to all generations. The short run's count is
    // replaced by the final one.
    ok(&breed(&["--out", path, "--resume", "--generations", "10"]));
    assert!(split.join("rediscovery.json").exists());
    ok(&verify(&split));
    ok(&breed(&["--out", path, "--resume", "--generations", "20"]));

    // Every Cascade, every judgment, the final finalists, every state line and the count
    // match the straight run. The report matches except for the calls the short run's
    // select phase spent.
    let (a, b) = (events(&straight), events(&split));

    assert_eq!(lineage(&straight), lineage(&split));
    assert_eq!(of_kind(&a, "state").len(), 21);
    assert_eq!(of_kind(&a, "state"), of_kind(&b, "state"));
    assert_eq!(
        read(&straight, "rediscovery.json"),
        read(&split, "rediscovery.json")
    );

    let (mut ra, mut rb) = (json(&straight, "report.json"), json(&split, "report.json"));

    assert!(rb["calls_used"].as_u64() >= ra["calls_used"].as_u64());
    ra["calls_used"] = serde_json::Value::Null;
    rb["calls_used"] = serde_json::Value::Null;
    assert_eq!(ra, rb);

    let checked = verify(&split);

    ok(&checked);
    assert!(
        stdout(&checked).contains("rediscovery recomputed and equal"),
        "{}",
        stdout(&checked)
    );
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

/// The checkpoint's next generation, and the calls it reserves for work in flight (0
/// when there is none); (0, 0) while there is no checkpoint.
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
fn rediscovery_json_comes_only_with_a_completed_select_phase() {
    // A run that waits for LLM answers has committed generations 0 and 1 but no select
    // phase: it has neither a report nor a count, and --verify checks neither.
    let paused = fresh("paused");
    let waiting = run(&paused, &[&SMALL[..], &["--llm-every", "2"]].concat());

    assert_eq!(waiting.status.code(), Some(3), "{}", stdout(&waiting));
    assert_eq!(progress(&paused), (2, 0));

    for name in ["report.json", "rediscovery.json"] {
        assert!(!paused.join(name).exists(), "{name}");
    }

    let unchecked = verify(&paused);

    ok(&unchecked);
    assert!(
        stdout(&unchecked).contains("rediscovery.json not checked: the run is not finished"),
        "{}",
        stdout(&unchecked)
    );

    // Generation 0 alone, then a long select phase. STOP lands in it once the checkpoint
    // reserves the calls of its first Cascade.
    let long = [
        "--domain",
        "circuits",
        "--seed",
        "31",
        "--generations",
        "0",
        "--search",
        "11",
        "--select",
        "20000",
        "--max-calls",
        "1000000",
    ];
    let straight = fresh("select-straight");
    let split = fresh("select-split");
    let path = split.to_str().expect("path");
    let child = Command::new(BREED)
        .args(["--out", path])
        .args(long)
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn");

    assert!(
        wait_for(120, || {
            let (next, reserved) = progress(&split);

            next == 1 && reserved > 0
        }),
        "the select phase reserved the calls of its first Cascade"
    );
    std::fs::write(split.join("STOP"), "").expect("stop");

    let stopped = child.wait_with_output().expect("wait");

    ok(&stopped);
    assert!(
        stdout(&stopped).contains("during the select phase"),
        "{}",
        stdout(&stopped)
    );

    // The select phase is not complete: neither a report nor a count, and --verify checks
    // neither.
    for name in ["report.json", "rediscovery.json"] {
        assert!(!split.join(name).exists(), "{name}");
    }

    std::fs::remove_file(split.join("STOP")).expect("unstop");

    let unfinished = verify(&split);

    ok(&unfinished);
    assert!(
        stdout(&unfinished).contains("rediscovery.json not checked: the run is not finished"),
        "{}",
        stdout(&unfinished)
    );

    // The resume scores the rest of the select panel and writes both. They equal those of
    // a straight run, apart from the calls the stopped attempt spent.
    ok(&breed(&["--out", path, "--resume"]));
    ok(&run(&straight, &long));
    assert_eq!(lineage(&straight), lineage(&split));
    assert_eq!(
        read(&straight, "rediscovery.json"),
        read(&split, "rediscovery.json")
    );

    let (mut ra, mut rb) = (json(&straight, "report.json"), json(&split, "report.json"));

    assert!(rb["calls_used"].as_u64() >= ra["calls_used"].as_u64());
    ra["calls_used"] = serde_json::Value::Null;
    rb["calls_used"] = serde_json::Value::Null;
    assert_eq!(ra, rb);

    let checked = verify(&split);

    ok(&checked);
    assert!(
        stdout(&checked).contains("rediscovery recomputed and equal"),
        "{}",
        stdout(&checked)
    );
}

#[test]
fn the_thread_count_changes_no_byte_of_a_circuit_run() {
    let at = |threads: &str| {
        let dir = fresh(&format!("threads-{threads}"));
        let out = Command::new(BREED)
            .args(["--out", dir.to_str().expect("path")])
            .args(RICH)
            .env("RAYON_NUM_THREADS", threads)
            .output()
            .expect("run brut-breed");

        ok(&out);

        RECORD.map(|name| read(&dir, name))
    };

    assert_eq!(at("1"), at("4"));
}

#[test]
fn promotion_of_a_circuit_run_is_refused() {
    let dir = fresh("promote");

    ok(&run(&dir, &SMALL));

    let record = || RECORD.map(|name| read(&dir, name));
    let before = record();
    let refused = breed(&["--out", dir.to_str().expect("path"), "--promote"]);

    assert_eq!(refused.status.code(), Some(1));
    assert!(
        stderr(&refused).contains("promotion of circuit Cascades is not part of step 1 (ADR 0006)"),
        "{}",
        stderr(&refused)
    );
    assert!(!dir.join("promote").exists());
    assert_eq!(record(), before);
}

#[test]
fn the_headline_says_whether_d_is_above_half_whatever_the_gate_says() {
    // A run on swap, cs and cz only: the seed Cascades solve two of the three targets, more
    // than the gate allows, and bred Cascades solve the third.
    let files = fresh("narrow");
    let file = files.join("narrow.json");
    let cfg = BreedConfig {
        seed: 57,
        generations: 5,
        children_per_generation: 8,
        search_puzzles: 9,
        select_puzzles: 6,
        max_calls: 400_000,
        domain: DomainConfig::Circuits(CircuitConfig {
            targets: ["swap", "cs", "cz"].map(String::from).to_vec(),
            ..CircuitConfig::default()
        }),
        ..BreedConfig::default()
    };
    let dir = files.join("run");

    std::fs::create_dir_all(&files).expect("mkdir");
    std::fs::write(&file, serde_json::to_string_pretty(&cfg).expect("json")).expect("write");

    let out = breed(&[
        "--out",
        dir.to_str().expect("path"),
        "--config",
        file.to_str().expect("path"),
    ]);

    ok(&out);

    // D = 1 but the run fails the gate, so it does not pass; the headline says both.
    let found = json(&dir, "rediscovery.json");

    assert_eq!(found["share"], 1.0);
    assert_eq!(found["calibrated"], false);
    assert_eq!(found["passes"], false);
    assert!(
        stdout(&out).contains(
            "the seeds solve 2 of 3 targets, bred Cascades 1 of the other 1; D > 1/2: yes; not calibrated"
        ),
        "{}",
        stdout(&out)
    );
}

/// One archived Cascade as its Born event logs it.
struct Logged {
    id: String,
    seed: bool,
    generation: usize,
    answers: Vec<Option<Circuit>>,
}

/// The rediscovery count of the run in `dir`, recomputed from its log alone: the search
/// panel rebuilt from the recorded settings (checked against the run start), and every
/// Born event's answers read back into circuits.
fn recount(dir: &Path) -> Rediscovery {
    let cfg: BreedConfig = serde_json::from_slice(&read(dir, "config.json")).expect("config");
    let domain = CircuitDomain::from_config(&cfg).expect("a circuit run");
    let panel = domain.panel(derive_seed(cfg.seed, "panel/search"), cfg.search_puzzles);
    let log = events(dir);
    let ids: Vec<&str> = panel.iter().map(|t| t.id.as_str()).collect();
    let started: Vec<&str> = log[0]["search"]
        .as_array()
        .expect("search ids")
        .iter()
        .map(|v| v.as_str().expect("id"))
        .collect();

    assert_eq!(ids, started, "the rebuilt panel is the logged one");

    let born: Vec<Logged> = of_kind(&log, "born")
        .iter()
        .map(|e| {
            let answers = match e["answers"].as_array() {
                Some(texts) => texts
                    .iter()
                    .map(|t| {
                        t.as_str()
                            .map(|s| s.parse::<Circuit>().expect("a logged circuit"))
                    })
                    .collect(),
                None => vec![None; panel.len()],
            };

            Logged {
                id: e["id"].as_str().expect("id").to_string(),
                seed: e["op"] == "seed" && e["generation"] == 0,
                generation: e["generation"].as_u64().expect("generation") as usize,
                answers,
            }
        })
        .collect();
    let members: Vec<Answered<'_>> = born
        .iter()
        .map(|m| Answered {
            id: &m.id,
            seed: m.seed,
            generation: m.generation,
            answers: &m.answers,
        })
        .collect();

    rediscovery(&panel, &members).expect("no integrity alarm")
}

#[test]
fn rediscovery_json_is_the_count_of_the_logged_answers_and_verify_guards_it() {
    let dir = fresh("rediscovery");
    let path = dir.to_str().expect("path");

    ok(&run(&dir, &RICH));

    let text = String::from_utf8(read(&dir, "rediscovery.json")).expect("utf-8");
    let recounted = recount(&dir);

    assert_eq!(
        text,
        serde_json::to_string_pretty(&recounted).expect("json")
    );

    // The count is not trivial: every verified target is on the panel, and bred Cascades
    // solve a target the seeds miss, in a known generation. (If a deliberate change of the
    // designer or the genes moves this, choose another seed for RICH.)
    assert_eq!(recounted.library, verified().len());
    assert!(recounted.reached_by_bred_only > 0, "{text}");
    assert!(
        recounted
            .targets
            .iter()
            .any(|t| t.reached_by_bred && !t.reached_by_seed && t.first_generation.is_some()),
        "{text}"
    );

    // Any edit of the file fails --verify, a changed number as well as a whitespace edit,
    // and so does its removal. Restored, the run verifies again.
    let mut edited: serde_json::Value = serde_json::from_str(&text).expect("json");
    let open = edited["reached_by_bred_only"].as_u64().expect("count");

    edited["reached_by_bred_only"] = serde_json::json!(open + 1);

    for forged in [
        serde_json::to_string_pretty(&edited).expect("json"),
        text.replacen("\"library\": ", "\"library\":  ", 1),
    ] {
        assert_ne!(forged, text);
        std::fs::write(dir.join("rediscovery.json"), &forged).expect("forge");

        let refused = verify(&dir);

        assert_eq!(refused.status.code(), Some(1));
        assert!(
            stderr(&refused).contains("integrity"),
            "{}",
            stderr(&refused)
        );
    }

    std::fs::remove_file(dir.join("rediscovery.json")).expect("remove");
    assert_eq!(verify(&dir).status.code(), Some(1));

    // A resume of the finished run writes the count anew, with no new event.
    let log = read(&dir, "events.jsonl");

    ok(&breed(&["--out", path, "--resume"]));
    assert_eq!(read(&dir, "rediscovery.json"), text.as_bytes());
    assert_eq!(read(&dir, "events.jsonl"), log);
    ok(&verify(&dir));

    // A resume to more generations that STOP halts before it commits one: the run is
    // unfinished again, so its report and count are gone and --verify checks neither,
    // although the archive of the checkpoint is still the finished one.
    std::fs::write(dir.join("STOP"), "").expect("stop");
    ok(&breed(&["--out", path, "--resume", "--generations", "21"]));
    assert!(!dir.join("report.json").exists());
    assert!(!dir.join("rediscovery.json").exists());

    let unfinished = verify(&dir);

    ok(&unfinished);
    assert!(
        stdout(&unfinished).contains("rediscovery.json not checked"),
        "{}",
        stdout(&unfinished)
    );
    std::fs::remove_file(dir.join("STOP")).expect("unstop");
    ok(&breed(&["--out", path, "--resume"]));

    let finished = verify(&dir);

    ok(&finished);
    assert!(
        stdout(&finished).contains("rediscovery recomputed and equal"),
        "{}",
        stdout(&finished)
    );
}

#[test]
fn verify_ties_the_counted_archive_to_the_logged_answers() {
    let dir = fresh("forged-archive");
    let path = dir.to_str().expect("path");

    ok(&run(&dir, &RICH));

    // Give a bred member, on a task it left unanswered, the answer another member gave
    // there. No hash covers the checkpoint, and the member's Born event still logs no
    // answer for that task.
    let mut cp = json(&dir, "snapshot.json");
    let archive = cp["snapshot"]["archive"]
        .as_array()
        .expect("archive")
        .clone();
    let (member, task, answer) = archive
        .iter()
        .enumerate()
        .filter(|(_, m)| m["op"] != "seed")
        .find_map(|(i, m)| {
            let answers = m["answers"].as_array().expect("answers");

            answers.iter().enumerate().find_map(|(j, a)| {
                let given = archive
                    .iter()
                    .map(|other| &other["answers"][j])
                    .find(|other| !other.is_null());

                given.filter(|_| a.is_null()).map(|g| (i, j, g.clone()))
            })
        })
        .expect("a bred member without an answer that another member gave");

    cp["snapshot"]["archive"][member]["answers"][task] = answer;
    std::fs::write(
        dir.join("snapshot.json"),
        serde_json::to_string(&cp).expect("json"),
    )
    .expect("forge");

    // --verify refuses the checkpoint. A resume writes the count anew from it, without a
    // new event, and --verify still refuses: the archive is not the logged one.
    assert_eq!(verify(&dir).status.code(), Some(1));

    let log = read(&dir, "events.jsonl");

    ok(&breed(&["--out", path, "--resume"]));
    assert_eq!(read(&dir, "events.jsonl"), log);

    let refused = verify(&dir);

    assert_eq!(refused.status.code(), Some(1));
    assert!(
        stderr(&refused).contains("integrity") && stderr(&refused).contains("Born events"),
        "{}",
        stderr(&refused)
    );
}

#[test]
fn verify_says_why_it_skips_the_count_of_a_finished_run() {
    let dir = fresh("no-report");
    let path = dir.to_str().expect("path");

    ok(&run(&dir, &SMALL));

    // A finished run without its report (a process that ended between its checkpoint and
    // its report leaves this): --verify does not check the count, and it does not call
    // the run unfinished.
    std::fs::remove_file(dir.join("report.json")).expect("remove");

    let skipped = verify(&dir);

    ok(&skipped);
    assert!(
        stdout(&skipped)
            .contains("rediscovery.json not checked: there is no report.json; resume the run"),
        "{}",
        stdout(&skipped)
    );
    assert!(
        !stdout(&skipped).contains("not finished"),
        "{}",
        stdout(&skipped)
    );

    // A resume writes both files again, and then both are checked.
    ok(&breed(&["--out", path, "--resume"]));

    let checked = verify(&dir);

    ok(&checked);
    assert!(
        stdout(&checked).contains("report recomputed and equal; rediscovery recomputed and equal"),
        "{}",
        stdout(&checked)
    );
}

/// Paths into the crate that `text` names: the module after `crate::`, or after
/// `crate::cascade::` for the cascade modules, with grouped paths opened.
fn crate_modules(text: &str) -> BTreeSet<String> {
    let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    let mut found = BTreeSet::new();

    for (at, marker) in compact.match_indices("crate::") {
        for path in paths_at(&compact[at + marker.len()..]) {
            let mut segments = path.split("::");
            let first = segments.next().unwrap_or("").to_string();
            let module = if first == "cascade" {
                segments.next().unwrap_or("").to_string()
            } else {
                first
            };

            found.insert(module);
        }
    }

    found
}

/// The paths that start at the head of `rest`, which holds no whitespace: one path, or
/// each path of a `{..}` group, nested groups opened too.
fn paths_at(rest: &str) -> Vec<String> {
    let head: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == ':')
        .collect();
    let Some(group) = rest[head.len()..].strip_prefix('{') else {
        return vec![head];
    };
    let mut items = Vec::new();
    let mut depth = 0usize;
    let mut item = String::new();

    for c in group.chars() {
        match c {
            '}' if depth == 0 => break,
            ',' if depth == 0 => items.push(std::mem::take(&mut item)),
            '{' => {
                depth += 1;
                item.push(c);
            }
            '}' => {
                depth -= 1;
                item.push(c);
            }
            _ => item.push(c),
        }
    }

    items.push(item);
    items
        .iter()
        .filter(|i| !i.is_empty())
        .flat_map(|i| paths_at(i).into_iter().map(|p| format!("{head}{p}")))
        .collect()
}

#[test]
fn the_designer_knows_nothing_of_the_library() {
    let text = std::fs::read_to_string(crate_path("src/cascade/designer.rs")).expect("designer.rs");

    // No name of the library, its references or its witnesses, and no path to circuit.rs.
    for word in ["library", "LIBRARY", "Reference", "witness", "circuit::"] {
        assert!(!text.contains(word), "designer.rs mentions {word}");
    }

    // It reaches the crate only through the simulator, the protocol, the backend types and
    // the utilities; `super::` is only the test module's glob of the designer itself.
    let allowed: BTreeSet<String> = ["qubit", "circuit_text", "backend", "util"]
        .map(String::from)
        .into();
    let used = crate_modules(&text);
    let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();

    assert!(used.is_subset(&allowed), "designer.rs uses {used:?}");
    assert!(
        used.contains("qubit") && used.contains("circuit_text"),
        "{used:?}"
    );
    assert!(
        compact
            .match_indices("super::")
            .all(|(at, m)| compact[at + m.len()..].starts_with('*')),
        "designer.rs reaches its parent module"
    );
    assert!(!text.contains("brutmaschine::"));

    // The scan itself sees grouped and nested paths, and paths outside `use` lines.
    assert_eq!(
        crate_modules("use crate::{backend::A, cascade::{qubit::{self, B}, circuit::C}};"),
        BTreeSet::from(["backend", "qubit", "circuit"].map(String::from))
    );
    assert_eq!(
        crate_modules("use crate::cascade::{qubit::B, circuit::C};\nlet x = crate::util::f();"),
        BTreeSet::from(["qubit", "circuit", "util"].map(String::from))
    );
}

/// The pilot seed whose search panel the calibration uses (the run itself is not made).
const PILOT_SEED: u64 = 101;

/// The pre-registered size of the search panel: three dressings of each verified target.
const SEARCH_TASKS: usize = 33;

/// The strong strategy of plan (e): every family in the order Hadamard, Phase, Linear,
/// Compose, Simplify, Search, Meet.
const STRONG: &str = "Use Hadamard gates to change the basis, so that bit flips become sign \
                      flips. Write the phase of each basis state as a sum over parities of \
                      the bits; one T or TDG per odd term. Treat each CNOT as a row \
                      operation: it adds one bit into another by XOR. Conjugate the target \
                      by one CNOT on both sides. Cancel inverse pairs and merge phase gates \
                      to simplify the circuit. Try short sequences, one gate at a time. Meet \
                      in the middle: grow sequences from both ends.";

/// The strong Cascade: one sim:large Worker with the seeds' role and format around the
/// strong strategy, 4096 thinking tokens, temperature 0 and self-check.
fn strong_cascade() -> Cascade {
    let (_, seed) = circuit_seed_cascades().swap_remove(0);
    let mut w = seed.stages[0].workers[0].clone();

    w.model = "sim:large".to_string();
    w.sections = w
        .sections
        .into_iter()
        .map(|s| match s.name.as_str() {
            "strategy" => Section {
                name: s.name,
                text: STRONG.to_string(),
            },
            _ => s,
        })
        .collect();
    w.traits = Traits {
        think_tokens: 4096,
        temperature: 0.0,
        answer_tokens: 512,
        self_check: true,
    };

    Cascade {
        stages: vec![Stage { workers: vec![w] }],
    }
}

/// The simulated designers of a circuit run.
fn designers() -> ModelPool {
    let mut pool = ModelPool::new();

    pool.insert("sim:small", Box::new(SimDesigner::small()));
    pool.insert("sim:large", Box::new(SimDesigner::large()));

    pool
}

/// The answer of the one Worker of `c` to `task`, from the request a run's search phase
/// sends it (run seed PILOT_SEED, the Cascade's id and the task's id), checked against
/// `run_cascade_in`, the path a run takes.
fn answer(domain: &CircuitDomain, c: &Cascade, task: &CircuitTask) -> Option<Circuit> {
    let w = &c.stages[0].workers[0];
    let task_seed = derive_seed(PILOT_SEED, &format!("search/{}/{}", c.id(), task.id));
    let prompt = domain.solve_prompt(task, &[]);
    let req = worker_request_with(
        w,
        prompt,
        derive_seed(task_seed, "stage0/worker0"),
        CIRCUIT_SELF_CHECK_TEXT,
    );
    let designer = SimDesigner::from_tag(&w.model).expect("a simulated designer");
    let reply = designer.generate(&req).expect("generate");
    let answer = (reply.stop_reason == StopReason::Stop)
        .then(|| domain.parse_answer(task, &reply.text))
        .flatten();
    let in_run = run_cascade_in(
        domain,
        &designers(),
        &CascadeLimits::default(),
        c,
        task,
        task_seed,
    );

    assert_eq!(in_run.answer, answer, "the request is the one a run sends");

    answer
}

#[test]
fn designer_calibration_is_pinned() {
    let domain = CircuitDomain::new(&CircuitConfig::default()).expect("domain");
    let panel = domain.panel(derive_seed(PILOT_SEED, "panel/search"), SEARCH_TASKS);
    let names: Vec<&str> = verified().iter().map(|t| t.name).collect();

    // Three dressed tasks of each verified target, round robin.
    assert_eq!(panel.len(), 3 * names.len());

    for (i, task) in panel.iter().enumerate() {
        assert_eq!(task.name, names[i % names.len()]);
    }

    // The seed Cascades of a run, each with the targets it solves on some task.
    let seeds = adapt_seeds(circuit_seed_cascades(), &CascadeLimits::default()).expect("seeds");
    let mut reached: BTreeMap<String, BTreeSet<&str>> = BTreeMap::new();

    for (name, cascade) in &seeds {
        let hits = reached.entry(name.clone()).or_default();

        for task in &panel {
            if let Some(c) = answer(&domain, cascade, task) {
                let grade = grade_circuit(task, &c).expect("no integrity alarm");

                if grade.solved {
                    hits.insert(task.name);
                }
            }
        }
    }

    let union: BTreeSet<&str> = reached.values().flatten().copied().collect();
    let pinned: BTreeMap<String, BTreeSet<&str>> = [
        ("linear", vec!["swap"]),
        ("phase", vec!["cs"]),
        ("hadamard", vec![]),
        ("search", vec![]),
    ]
    .into_iter()
    .map(|(name, hits)| (name.to_string(), hits.into_iter().collect()))
    .collect();

    assert_eq!(union, BTreeSet::from(["swap", "cs"]), "R_seed");
    assert_eq!(reached, pinned);
    // The calibration gate of the pre-registration: |R_seed| <= |L| / 3.
    assert!(3 * union.len() <= names.len());

    // The strong prompt passes the genome's rules and the admission rule. Its one
    // hand-written text is no pool phrase, so a run without LLM rounds cannot write it;
    // circuit_genes.rs shows that pool phrases reach every target. It names the families
    // in its order and reaches the reference on every task.
    let strong = strong_cascade();
    let w = &strong.stages[0].workers[0];

    strong
        .check(&CascadeLimits::default())
        .expect("a valid Cascade");
    no_gate_lists(&strong).expect("admitted");
    assert_eq!(
        knowledge(&worker_request_with(w, String::new(), 0, CIRCUIT_SELF_CHECK_TEXT).system),
        [
            "Hadamard", "Phase", "Linear", "Compose", "Simplify", "Search", "Meet"
        ]
    );

    for task in &panel {
        let c = answer(&domain, &strong, task)
            .unwrap_or_else(|| panic!("{} ({}): no answer", task.name, task.id));
        let grade = grade_circuit(task, &c).expect("no integrity alarm");

        assert!(
            grade.correct && grade.solved,
            "{} ({}): {} with cost {:?}, reference {:?}",
            task.name,
            task.id,
            c.render(),
            c.cost(),
            task.reference
        );
    }
}

#[test]
fn the_admission_rule_counts_gate_mentions_per_section() {
    // The eight gates of the QFT2 witness, two to a section: every section passes, so the
    // Cascade does. Three of them in one section are refused. Against a memorised circuit
    // spread like this stands the disguise of every task.
    let (_, seed) = circuit_seed_cascades().swap_remove(0);
    let with = |texts: &[&str]| {
        let mut c = seed.clone();

        for (name, text) in ["check", "style", "residual", "notes"].iter().zip(texts) {
            c.stages[0].workers[0].sections.push(Section {
                name: name.to_string(),
                text: text.to_string(),
            });
        }

        c
    };
    let spread = with(&[
        "Start with H 0 and then CNOT 0 1.",
        "Next comes CNOT 1 0, then TDG 1.",
        "Then CNOT 0 1 and T 1.",
        "End with T 0 and H 0.",
    ]);
    let listed = with(&["Start with H 0, then CNOT 0 1 and CNOT 1 0."]);

    spread
        .check(&CascadeLimits::default())
        .expect("a valid Cascade");
    assert_eq!(no_gate_lists(&spread), Ok(()));
    assert!(no_gate_lists(&listed).is_err());
}

/// Every dressing of a target on `n` qubits: each relabelling of the qubits, with X on
/// any set of inputs and on any set of outputs.
fn every_dressing(n: usize) -> Vec<Dressing> {
    let span = u8::try_from(n).expect("few qubits");
    let mut perms: Vec<Vec<u8>> = vec![Vec::new()];

    for _ in 0..n {
        let mut longer = Vec::new();

        for head in &perms {
            for q in (0..span).filter(|q| !head.contains(q)) {
                let mut perm = head.clone();

                perm.push(q);
                longer.push(perm);
            }
        }

        perms = longer;
    }

    let mut out = Vec::new();

    for perm in perms {
        for x_in in 0..1u32 << n {
            for x_out in 0..1u32 << n {
                out.push(Dressing {
                    perm: perm.clone(),
                    x_in,
                    x_out,
                });
            }
        }
    }

    out
}

/// The first pool phrase whose text names exactly the method families `families`.
fn phrase_naming(families: &[&str]) -> &'static str {
    let mut wanted = families.to_vec();

    wanted.sort();
    CIRCUIT_PHRASES
        .iter()
        .map(|(_, text)| *text)
        .find(|text| {
            let mut named = knowledge(text);

            named.sort();
            named == wanted
        })
        .expect("a pool phrase")
}

/// A Worker of the seeds' role and format around pool phrases in the sections `slots`,
/// with its Traits at the ceiling: sim:large, 4096 thinking tokens, temperature 0 and
/// self-check.
fn ceiling_worker(slots: &[(&str, &str)]) -> Worker {
    let (_, seed) = circuit_seed_cascades().swap_remove(0);
    let base = &seed.stages[0].workers[0];
    let section = |name: &str| {
        base.sections
            .iter()
            .find(|s| s.name == name)
            .expect("a seed section")
            .clone()
    };
    let mut sections = vec![section("role")];

    for (name, text) in slots {
        sections.push(Section {
            name: name.to_string(),
            text: text.to_string(),
        });
    }

    sections.push(section("format"));

    Worker {
        model: "sim:large".to_string(),
        sections,
        traits: Traits {
            think_tokens: 4096,
            temperature: 0.0,
            answer_tokens: 512,
            self_check: true,
        },
    }
}

/// On how many of the dressings of library target `name` the Worker `w` answers exactly
/// at the reference, with requests built as a run builds them; and how many there are.
fn solved_dressings(w: &Worker, name: &str) -> (usize, usize) {
    let entry = library_target(name).expect("a library target");
    let base = (entry.build)().expect("target");
    let reference = entry.reference.expect("a verified reference");
    let designer = SimDesigner::from_tag(&w.model).expect("a simulated designer");
    let dressings = every_dressing(base.qubits());
    let solved = (0u64..)
        .zip(&dressings)
        .filter(|(i, d)| {
            let target = base.dress(d);
            let prompt = render_circuit_solve_prompt(&target, 64, &[]);
            let req = worker_request_with(w, prompt, *i, CIRCUIT_SELF_CHECK_TEXT);
            let reply = designer.generate(&req).expect("generate");
            let answer = (reply.stop_reason == StopReason::Stop)
                .then(|| parse_circuit_answer(&reply.text, target.qubits(), 64))
                .flatten();

            answer.is_some_and(|c| {
                let checked = qubit::check(&target, &c).expect("check");

                checked.exact && reference.reached(&checked.cost)
            })
        })
        .count();

    (solved, dressings.len())
}

#[test]
fn fewer_families_reach_qft2_and_ccix_at_the_ceiling_of_the_traits() {
    // At the seeds' Traits, one phrase step from a seed reaches neither QFT2 nor CCiX, and
    // two steps that teach a seed Hadamard, Phase and Linear reach both (circuit_genes.rs).
    // At the ceiling of the Traits fewer families do, in some disguises: Meet alone reaches
    // QFT2, and Hadamard, Phase and Simplify reach CCiX. Both Workers differ from every
    // seed in model, thinking budget and self-check, and both are admitted.
    let meet = ceiling_worker(&[("strategy", phrase_naming(&["Meet"]))]);
    let hps = ceiling_worker(&[
        ("strategy", phrase_naming(&["Hadamard"])),
        ("check", phrase_naming(&["Phase"])),
        ("style", phrase_naming(&["Simplify"])),
    ]);
    let routes = [
        (&meet, vec!["Meet"]),
        (&hps, vec!["Hadamard", "Phase", "Simplify"]),
    ];

    for (w, families) in routes {
        let c = Cascade {
            stages: vec![Stage {
                workers: vec![w.clone()],
            }],
        };
        let system = worker_request_with(w, String::new(), 0, CIRCUIT_SELF_CHECK_TEXT).system;

        c.check(&CascadeLimits::default()).expect("a valid Cascade");
        no_gate_lists(&c).expect("admitted");
        assert_eq!(knowledge(&system), families);
    }

    assert_eq!(solved_dressings(&meet, "qft2"), (4, 32));
    assert_eq!(solved_dressings(&hps, "ccix"), (128, 384));
}

/// The same gate list for `src/quantum.rs`, whose qubit 0 is the least significant bit:
/// qubit q becomes n - 1 - q, so both simulators index basis states alike.
fn to_quantum(n: usize, c: &Circuit) -> quantum::Circuit {
    let flip = |q: u8| n - 1 - usize::from(q);
    let gates = c
        .gates()
        .iter()
        .map(|&g| match g {
            Gate::H(q) => quantum::Gate::H(flip(q)),
            Gate::S(q) => quantum::Gate::S(flip(q)),
            Gate::Sdg(q) => quantum::Gate::Sdg(flip(q)),
            Gate::T(q) => quantum::Gate::T(flip(q)),
            Gate::Tdg(q) => quantum::Gate::Tdg(flip(q)),
            Gate::X(q) => quantum::Gate::X(flip(q)),
            Gate::Z(q) => quantum::Gate::Z(flip(q)),
            Gate::Cnot { control, target } => quantum::Gate::Cnot {
                control: flip(control),
                target: flip(target),
            },
        })
        .collect();

    quantum::Circuit::from_gates(n, gates).expect("a valid quantum.rs circuit")
}

/// The unitary of `c` on `n` qubits by `src/quantum.rs`.
fn quantum_unitary(n: usize, c: &Circuit) -> quantum::Unitary {
    to_quantum(n, c).unitary().expect("unitary")
}

/// The exact decision of `qubit.rs`: equal up to one global phase.
fn exactly_equal(n: usize, a: &Circuit, b: &Circuit) -> bool {
    let key = |c: &Circuit| qubit::canonical_key(&qubit::columns(n, c).expect("columns"));

    key(a) == key(b)
}

/// The decision of `src/quantum.rs` at its fidelity tolerance.
fn equivalent_by_fidelity(n: usize, a: &Circuit, b: &Circuit) -> bool {
    quantum_unitary(n, a)
        .equivalent(&quantum_unitary(n, b), quantum::DEFAULT_FIDELITY_TOL)
        .expect("same size")
}

fn qubit_index(n: usize, rng: &mut ChaCha8Rng) -> u8 {
    u8::try_from(rng.random_range(0..n)).expect("few qubits")
}

/// A uniformly drawn gate on `n` qubits; a CNOT only from two qubits on.
fn random_gate(n: usize, rng: &mut ChaCha8Rng) -> Gate {
    let q = qubit_index(n, rng);
    let kinds = if n >= 2 { 8 } else { 7 };

    match rng.random_range(0..kinds) {
        0 => Gate::H(q),
        1 => Gate::S(q),
        2 => Gate::Sdg(q),
        3 => Gate::T(q),
        4 => Gate::Tdg(q),
        5 => Gate::X(q),
        6 => Gate::Z(q),
        _ => {
            let shift = u8::try_from(rng.random_range(1..n)).expect("few qubits");
            let span = u8::try_from(n).expect("few qubits");

            Gate::Cnot {
                control: q,
                target: (q + shift) % span,
            }
        }
    }
}

fn random_gates(n: usize, rng: &mut ChaCha8Rng) -> Vec<Gate> {
    let len = rng.random_range(0..=24);

    (0..len).map(|_| random_gate(n, rng)).collect()
}

/// A word equal to the identity up to a global phase, on a drawn qubit (and a drawn second
/// qubit for a CNOT pair).
fn identity_word(n: usize, rng: &mut ChaCha8Rng) -> Vec<Gate> {
    use Gate::{H, S, Sdg, T, Tdg, X, Z};

    let q = qubit_index(n, rng);
    let words: Vec<Vec<Gate>> = vec![
        vec![X(q), X(q)],
        vec![H(q), H(q)],
        vec![Z(q), Z(q)],
        vec![S(q), Sdg(q)],
        vec![Tdg(q), T(q)],
        // T T = S, then SDG undoes it; S S = Z, then Z undoes it.
        vec![T(q), T(q), Sdg(q)],
        vec![S(q), S(q), Z(q)],
        // (X Z)^2 = -I and (S H)^3 = w I: identities only up to the global phase.
        vec![X(q), Z(q), X(q), Z(q)],
        vec![S(q), H(q), S(q), H(q), S(q), H(q)],
    ];

    if n >= 2 && rng.random_bool(0.25) {
        return vec![random_cnot(n, rng); 2];
    }

    words[rng.random_range(0..words.len())].clone()
}

fn random_cnot(n: usize, rng: &mut ChaCha8Rng) -> Gate {
    loop {
        if let g @ Gate::Cnot { .. } = random_gate(n, rng) {
            return g;
        }
    }
}

/// A word with exactly the unitary of `g`, up to a global phase, other than `g` itself.
fn rewrite(g: Gate) -> Vec<Gate> {
    use Gate::{H, S, Sdg, T, Tdg, X, Z};

    match g {
        S(q) => vec![T(q), T(q)],
        Sdg(q) => vec![Tdg(q), Tdg(q)],
        Z(q) => vec![H(q), X(q), H(q)],
        X(q) => vec![H(q), Z(q), H(q)],
        // S H S H S = w H, since (S H)^3 = w I.
        H(q) => vec![S(q), H(q), S(q), H(q), S(q)],
        T(q) => vec![Tdg(q), S(q)],
        Tdg(q) => vec![T(q), Sdg(q)],
        // H on both qubits reverses a CNOT.
        Gate::Cnot { control, target } => vec![
            H(control),
            H(target),
            Gate::Cnot {
                control: target,
                target: control,
            },
            H(control),
            H(target),
        ],
    }
}

/// True for a gate that is diagonal: it commutes with every other diagonal gate.
fn diagonal(g: Gate) -> bool {
    matches!(
        g,
        Gate::S(_) | Gate::Sdg(_) | Gate::T(_) | Gate::Tdg(_) | Gate::Z(_)
    )
}

/// True when `a` and `b` commute for a reason this test knows: disjoint qubits, both
/// diagonal, a diagonal gate on a CNOT's control, an X on a CNOT's target, or two CNOTs
/// that share only their control or only their target.
fn commute(a: Gate, b: Gate) -> bool {
    let qubits = |g: Gate| {
        let (x, y) = g.qubits();

        [Some(x), y].into_iter().flatten().collect::<BTreeSet<u8>>()
    };
    let one_way = |a: Gate, b: Gate| match (a, b) {
        (d, Gate::Cnot { control, .. }) if diagonal(d) => d.qubits().0 == control,
        (Gate::X(q), Gate::Cnot { target, .. }) => q == target,
        (
            Gate::Cnot {
                control: c1,
                target: t1,
            },
            Gate::Cnot {
                control: c2,
                target: t2,
            },
        ) => (c1 == c2) != (t1 == t2) && c1 != t2 && c2 != t1,
        _ => false,
    };

    qubits(a).is_disjoint(&qubits(b))
        || (diagonal(a) && diagonal(b))
        || one_way(a, b)
        || one_way(b, a)
}

fn circuit(gates: Vec<Gate>) -> Circuit {
    Circuit::new(gates).expect("within the caps")
}

/// A circuit and a second one with the same unitary up to a global phase: an identity
/// word inserted, a gate rewritten, or two commuting neighbours swapped.
fn equal_pair(n: usize, rng: &mut ChaCha8Rng) -> (Circuit, Circuit) {
    let gates = random_gates(n, rng);
    let mut other = gates.clone();
    let swappable: Vec<usize> = (1..gates.len())
        .filter(|&i| gates[i - 1] != gates[i] && commute(gates[i - 1], gates[i]))
        .collect();

    match rng.random_range(0..3) {
        0 if !gates.is_empty() => {
            let at = rng.random_range(0..gates.len());

            other.splice(at..=at, rewrite(gates[at]));
        }
        1 if !swappable.is_empty() => {
            let at = swappable[rng.random_range(0..swappable.len())];

            other.swap(at - 1, at);
        }
        _ => {
            let at = rng.random_range(0..=gates.len());

            other.splice(at..at, identity_word(n, rng));
        }
    }

    (circuit(gates), circuit(other))
}

/// A circuit and a second one whose unitary differs from it by more than a global phase:
/// one gate G inserted, or one gate G' replaced by another gate G. No gate of the set is a
/// phase times the identity, and no two distinct gates differ only by a phase. So, with A
/// and B the gates after and before, A G B is never a phase times A B or A G' B.
fn different_pair(n: usize, rng: &mut ChaCha8Rng) -> (Circuit, Circuit) {
    let gates = random_gates(n, rng);
    let mut other = gates.clone();

    if gates.is_empty() || rng.random_bool(0.5) {
        let at = rng.random_range(0..=gates.len());

        other.insert(at, random_gate(n, rng));
    } else {
        let at = rng.random_range(0..gates.len());
        let mut g = random_gate(n, rng);

        while g == gates[at] {
            g = random_gate(n, rng);
        }

        other[at] = g;
    }

    (circuit(gates), circuit(other))
}

#[test]
fn the_exact_checker_agrees_with_quantum_rs_on_random_pairs() {
    let mut rng = ChaCha8Rng::seed_from_u64(20261029);
    let mut seen = BTreeMap::new();

    for i in 0..900 {
        let n = 1 + i % 4;
        let (kind, (a, b)) = match i % 3 {
            0 => ("equal", equal_pair(n, &mut rng)),
            1 => ("different", different_pair(n, &mut rng)),
            _ => (
                "independent",
                (
                    circuit(random_gates(n, &mut rng)),
                    circuit(random_gates(n, &mut rng)),
                ),
            ),
        };
        let exact = exactly_equal(n, &a, &b);
        let close = equivalent_by_fidelity(n, &a, &b);
        let pair = format!("{n} qubits: [{}] vs [{}]", a.render(), b.render());

        assert_eq!(exact, close, "the simulators disagree on {pair}");

        match kind {
            "equal" => assert!(exact, "equal by construction: {pair}"),
            "different" => assert!(!exact, "different by construction: {pair}"),
            _ => {}
        }

        *seen.entry((kind, exact)).or_insert(0usize) += 1;
    }

    // 600 pairs whose answer is known by construction, plus 300 independent ones.
    assert_eq!(seen.get(&("equal", true)), Some(&300), "{seen:?}");
    assert_eq!(seen.get(&("different", false)), Some(&300), "{seen:?}");
}

/// The Nielsen-Chuang Toffoli (controls 0 and 1, target 2): 7 T and 6 CNOT.
const TOFFOLI_NC: &str = "H 2; CNOT 1 2; TDG 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; CNOT 0 2; \
                          T 1; T 2; H 2; CNOT 0 1; T 0; TDG 1; CNOT 0 1";

/// The witness circuits of the WP3 tests in `circuit.rs`, one per library entry, with
/// their T and CNOT counts. Most come from the literature; the research for step 1 built
/// and checked the others: QFT2, the W gate, sqrt(SWAP), the quantum OR, the 1-bit adder
/// and the explicit form of the Fredkin circuit. Gates are in time order, and qubit 0 is
/// the most significant bit.
fn witnesses() -> Vec<(&'static str, Circuit, (u32, u32))> {
    let parse = |text: &str| -> Circuit { text.parse().expect("circuit text") };
    let nc: Vec<&str> = TOFFOLI_NC.split(';').map(str::trim).collect();
    let ccz: Vec<&str> = nc.iter().copied().filter(|g| *g != "H 2").collect();
    let fredkin = format!(
        "S 1; CNOT 2 1; SDG 1; H 2; S 2; H 2; S 2; {}; CNOT 2 1",
        nc[2..].join("; ")
    );
    // The adder's Toffoli has controls 1 and 2 and target 3.
    let inner = parse(TOFFOLI_NC).relabel(&[1, 2, 3]).expect("relabel");
    let sqrt_swap = "SDG 0; CNOT 0 1; S 1; T 1; SDG 0; H 0; SDG 0; TDG 0; H 0; S 0; \
                     CNOT 1 0; SDG 0; H 0; S 0; T 0; H 0; S 0; CNOT 0 1; S 1";
    let table = [
        ("cz", "H 1; CNOT 0 1; H 1".to_string(), (0, 1)),
        ("swap", "CNOT 0 1; CNOT 1 0; CNOT 0 1".to_string(), (0, 3)),
        (
            "cs",
            "T 0; T 1; CNOT 0 1; TDG 1; CNOT 0 1".to_string(),
            (3, 2),
        ),
        (
            "ch",
            "S 1; H 1; T 1; CNOT 0 1; TDG 1; H 1; SDG 1".to_string(),
            (2, 1),
        ),
        (
            "qft2",
            "H 0; CNOT 0 1; CNOT 1 0; TDG 1; CNOT 0 1; T 1; T 0; H 0".to_string(),
            (3, 3),
        ),
        (
            "w_gate",
            "CNOT 0 1; SDG 0; H 0; TDG 0; CNOT 1 0; T 0; H 0; CNOT 0 1; S 0".to_string(),
            (2, 3),
        ),
        ("sqrt_swap", sqrt_swap.to_string(), (3, 3)),
        ("ccz", ccz.join("; "), (7, 6)),
        ("toffoli", TOFFOLI_NC.to_string(), (7, 6)),
        ("peres", nc[..14].join("; "), (7, 5)),
        ("fredkin", fredkin, (7, 7)),
        (
            "quantum_or",
            format!("X 0; X 1; {TOFFOLI_NC}; X 0; X 1; X 2"),
            (7, 6),
        ),
        (
            "rtof",
            "H 2; T 2; CNOT 1 2; TDG 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; H 2".to_string(),
            (4, 3),
        ),
        (
            "ccix",
            "H 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; H 2".to_string(),
            (4, 4),
        ),
        (
            "full_adder",
            format!(
                "CNOT 0 1; CNOT 0 2; {}; CNOT 0 3; CNOT 1 2; CNOT 0 2",
                inner.render()
            ),
            (7, 11),
        ),
    ];

    table
        .into_iter()
        .map(|(name, text, counts)| (name, parse(&text), counts))
        .collect()
}

/// A complex number as (re, im).
type Amp = (f64, f64);

fn mul_conj(a: Amp, b: Amp) -> Amp {
    (a.0 * b.0 + a.1 * b.1, a.0 * b.1 - a.1 * b.0)
}

/// |sum over all entries of conj(u) v|^2 / d^2: the fidelity of two d x d matrices given
/// column by column, 1 when they agree up to a global phase.
fn fidelity(u: &[Vec<Amp>], v: &[Vec<Amp>]) -> f64 {
    let d = u.len() as f64;
    let (re, im) = u
        .iter()
        .flatten()
        .zip(v.iter().flatten())
        .map(|(&a, &b)| mul_conj(a, b))
        .fold((0.0, 0.0), |s, x| (s.0 + x.0, s.1 + x.1));

    (re * re + im * im) / (d * d)
}

#[test]
fn every_library_witness_agrees_between_both_simulators() {
    let witnesses = witnesses();
    let names: BTreeSet<&str> = witnesses.iter().map(|w| w.0).collect();

    assert_eq!(
        names,
        LIBRARY.iter().map(|t| t.name).collect::<BTreeSet<&str>>()
    );

    for (name, c, (t, cnot)) in witnesses {
        let entry = library_target(name).expect("in the library");
        let n = entry.qubits;
        let target = (entry.build)().expect("target");
        let dim = 1usize << n;

        // Column j of each matrix: the image of input j.
        let exact: Vec<Vec<Amp>> = qubit::columns(n, &c)
            .expect("columns")
            .iter()
            .map(|s| s.to_c64().iter().map(|a| (a.re, a.im)).collect())
            .collect();
        let u = quantum_unitary(n, &c);
        let float: Vec<Vec<Amp>> = (0..dim)
            .map(|j| {
                (0..dim)
                    .map(|i| {
                        let a = u.get(i, j).expect("entry");

                        (a.re, a.im)
                    })
                    .collect()
            })
            .collect();
        let wanted: Vec<Vec<Amp>> = target
            .columns()
            .iter()
            .map(|column| {
                let terms = column.terms();
                let scale = (terms.len() as f64).sqrt().recip();
                let mut col = vec![(0.0, 0.0); dim];

                for &(b, p) in terms {
                    let angle = f64::from(p) * FRAC_PI_4;

                    col[b as usize] = (scale * angle.cos(), scale * angle.sin());
                }

                col
            })
            .collect();
        let gap = exact
            .iter()
            .flatten()
            .zip(float.iter().flatten())
            .map(|(a, b)| (a.0 - b.0).abs().max((a.1 - b.1).abs()))
            .fold(0.0, f64::max);

        // The same circuit gives the same matrix in both simulators, entry by entry, and
        // a fidelity of 1.
        assert!(gap <= 1e-12, "{name}: entries differ by {gap}");
        assert!((fidelity(&float, &exact) - 1.0).abs() <= 1e-12, "{name}");

        // quantum.rs, too, finds that the witness implements its library target, which
        // was built from its definition.
        assert!(
            (fidelity(&float, &wanted) - 1.0).abs() <= 1e-12,
            "{name}: not the target"
        );

        let checked = qubit::check(&target, &c).expect("check");

        assert!(checked.exact, "{name}");
        assert_eq!((checked.cost.t, checked.cost.cnot), (t, cnot), "{name}");
    }
}
