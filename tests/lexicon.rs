//! Tests of the Lexicon and the brevity case on the `brut-breed` binary (arms L and R,
//! `.scratch/eigene-sprache/spec.md`, ticket 01). With both settings off nothing a Run writes
//! changes, which the golden tests of `tests/universe.rs`, the drafting digests of `arena.rs`
//! and the release tests of `tests/soul.rs` check. Here: the settings are left out when off
//! and recorded when on; Runs of 60 generations in the arms of the screen of ticket 02 (N with
//! brevity alone, R and L with a Lexicon too) write the same bytes at 1 and at 4 threads, pass
//! `--verify`, and resume at generation 30 like straight Runs; `--verify` and a resume find a
//! changed word; and the settings are refused where they cannot work.
//!
//! No test here runs a seed of a screen or of the pilots: the screen of ticket 02 uses seeds
//! 6001 to 6008, and these tests use seeds above 6400.
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde::{Deserialize, Serialize};

use brutmaschine::cascade::arena::{BreedConfig, Event};
use brutmaschine::cascade::circuit_genes::LEXICON_TERMS;
use brutmaschine::cascade::designer::{FAMILY_STEMS, named_families};
use brutmaschine::cascade::lexicon::{LexiconConfig, phrase};
use brutmaschine::util::sha256_hex;

const BREED: &str = env!("CARGO_BIN_EXE_brut-breed");

/// The five record files of a finished circuit Run.
const RECORD: [&str; 5] = [
    "config.json",
    "events.jsonl",
    "snapshot.json",
    "report.json",
    "rediscovery.json",
];

/// A fresh, empty directory under the test target directory.
fn fresh(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("lexicon")
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

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
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

fn path(dir: &Path) -> &str {
    dir.to_str().expect("utf-8 path")
}

fn read(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn json(dir: &Path, name: &str) -> serde_json::Value {
    serde_json::from_slice(&read(dir, name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// The events of a Run's log, as JSON values.
fn logged_events(dir: &Path) -> Vec<serde_json::Value> {
    String::from_utf8(read(dir, "events.jsonl"))
        .expect("utf-8 log")
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).expect("json")["event"].clone())
        .collect()
}

fn of_kind<'e>(events: &'e [serde_json::Value], kind: &str) -> Vec<&'e serde_json::Value> {
    events.iter().filter(|e| e["kind"] == kind).collect()
}

/// The flags of the screen of ticket 02, for 60 generations on `seed`.
fn screen(seed: &str) -> Vec<&str> {
    vec![
        "--domain",
        "circuits",
        "--search",
        "33",
        "--select",
        "22",
        "--children",
        "8",
        "--generations",
        "60",
        "--max-calls",
        "13000000",
        "--selection",
        "lexicase",
        "--seed",
        seed,
    ]
}

/// The config file of arm `arm` (N, R or L) in `dir`, written as ticket 02 writes it: the
/// `config.json` that the binary writes with the screen's flags, here from a Run stopped
/// before its generation 0, with `"brevity": true`, and for R and L the Lexicon.
fn arm_config(dir: &Path, arm: &str) -> PathBuf {
    let plain = dir.join("plain");

    if !plain.join("config.json").exists() {
        std::fs::create_dir_all(&plain).expect("mkdir");
        std::fs::write(plain.join("STOP"), "").expect("stop");

        let mut args = vec!["--out", path(&plain)];

        args.extend(screen("6400"));
        ok(&breed(&args));
    }

    let mut cfg = json(&plain, "config.json");

    cfg["brevity"] = serde_json::json!(true);

    let lexicon = match arm {
        "R" => Some(LexiconConfig::arm_r()),
        "L" => Some(LexiconConfig::arm_l()),
        _ => None,
    };

    if let Some(lexicon) = lexicon {
        cfg["lexicon"] = serde_json::to_value(lexicon).expect("json");
    }

    let file = dir.join(format!("config-{arm}.json"));

    std::fs::write(&file, serde_json::to_string_pretty(&cfg).expect("json")).expect("write");

    file
}

/// A Run of arm `arm` with the screen's flags on `seed` into `dir`, at `threads` threads.
fn arm_run(dir: &Path, arm: &str, seed: &str, threads: &str, generations: &str) -> Output {
    let config = arm_config(dir.parent().expect("parent"), arm);
    let mut args = vec!["--out", path(dir), "--config", path(&config)];
    let mut flags = screen(seed);

    flags[9] = generations;
    args.extend(flags);

    Command::new(BREED)
        .args(&args)
        .env("RAYON_NUM_THREADS", threads)
        .output()
        .expect("run brut-breed")
}

/// The places of the families named `names`, in the order of `designer::FAMILY_STEMS`.
fn places(names: &serde_json::Value) -> Vec<usize> {
    names
        .as_array()
        .expect("a meaning")
        .iter()
        .map(|n| {
            FAMILY_STEMS
                .iter()
                .position(|(family, _)| n == family)
                .expect("a family")
        })
        .collect()
}

#[test]
fn lexicon_and_brevity_are_left_out_when_off() {
    let off = BreedConfig::default();
    let text = serde_json::to_string(&off).expect("json");

    assert!(off.lexicon.is_none() && !off.brevity);
    assert!(
        !text.contains("lexicon") && !text.contains("brevity"),
        "{text}"
    );

    // On, they are recorded at the end of the settings and define the experiment.
    let on = BreedConfig {
        lexicon: Some(LexiconConfig::arm_l()),
        brevity: true,
        ..BreedConfig::default()
    };
    let written = serde_json::to_string(&on).expect("json");

    assert!(
        written.ends_with(
            r#","lexicon":{"every":20,"best":22,"max_words":8,"meaning":"compression"},"brevity":true}"#
        ),
        "{written}"
    );
    assert_eq!(
        serde_json::from_str::<BreedConfig>(&written).expect("read"),
        on
    );
    assert_ne!(on.fixed_hash(), off.fixed_hash());

    let random = BreedConfig {
        lexicon: Some(LexiconConfig::arm_r()),
        ..on.clone()
    };

    for one in [
        BreedConfig {
            brevity: false,
            ..on.clone()
        },
        BreedConfig {
            lexicon: None,
            ..on.clone()
        },
        random,
    ] {
        assert_ne!(one.fixed_hash(), off.fixed_hash());
        assert_ne!(one.fixed_hash(), on.fixed_hash());
    }

    // A brevity of false is off: it is left out again, and the hash is that of a Run
    // without it.
    let spelled = format!("{},\"brevity\":false}}", &text[..text.len() - 1]);
    let read: BreedConfig = serde_json::from_str(&spelled).expect("read");

    assert!(spelled.ends_with(r#","brevity":false}"#), "{spelled}");
    assert_eq!(read, off);
    assert_eq!(read.fixed_hash(), off.fixed_hash());

    // An unknown or a missing field of the Lexicon is refused, and so is a brevity that is
    // not a flag.
    for bad in [
        written.replace(r#""max_words":8"#, r#""max_words":8,"length":3"#),
        written.replace(r#","max_words":8"#, ""),
        written.replace(r#""compression""#, r#""zipf""#),
        written.replace(r#""brevity":true"#, r#""brevity":1"#),
    ] {
        assert_ne!(bad, written);
        assert!(serde_json::from_str::<BreedConfig>(&bad).is_err(), "{bad}");
    }
}

#[test]
fn a_run_with_a_lexicon_or_brevity_reproduces() {
    // 60 generations with the screen's flags in each arm: --verify passes, and the five
    // record files are byte-equal at 1 and at 4 threads.
    let root = fresh("reproduce");

    for arm in ["N", "R", "L"] {
        let at = |threads: &str| {
            let dir = root.join(format!("{arm}-{threads}"));

            ok(&arm_run(&dir, arm, "6411", threads, "60"));

            let verified = breed(&["--out", path(&dir), "--verify"]);

            ok(&verified);
            assert!(
                stdout(&verified).contains("rediscovery recomputed and equal"),
                "{arm}: {}",
                stdout(&verified)
            );

            (RECORD.map(|name| read(&dir, name)), dir)
        };
        let (four, dir) = at("4");

        assert_eq!(four, at("1").0, "arm {arm}: 1 and 4 threads");

        let events = logged_events(&dir);
        let config = json(&dir, "config.json");
        let states = of_kind(&events, "state");
        let coined = of_kind(&events, "coined");
        let retired = of_kind(&events, "retired");

        assert_eq!(states.len(), 61, "arm {arm}");
        assert_eq!(config["brevity"], true, "arm {arm}");

        if arm == "N" {
            assert!(config.get("lexicon").is_none());
            assert!(coined.is_empty() && retired.is_empty());
            assert!(states.iter().all(|s| s.get("lexicon").is_none()));
            continue;
        }

        let meaning = if arm == "R" { "random" } else { "compression" };

        assert_eq!(config["lexicon"]["meaning"], meaning);

        // Words were coined, each at a coining step, and each names exactly its meaning, which
        // is logged by name in index order. Arm R's phrase names its families in index order.
        assert!(!coined.is_empty(), "arm {arm}: no word coined");

        for word in coined.iter().chain(&retired) {
            let g = word["generation"].as_u64().expect("generation");

            assert!(g > 0 && g % 20 == 0, "arm {arm}: {word}");
        }

        for word in &coined {
            let text = word["phrase"].as_str().expect("phrase");
            let meaning = places(&word["meaning"]);

            assert!((2..=3).contains(&meaning.len()), "{word}");
            assert!(meaning.windows(2).all(|w| w[0] < w[1]), "{word}");
            assert_eq!(named_families(text), meaning, "arm {arm}: {word}");

            if arm == "R" {
                assert_eq!(text, phrase(&LEXICON_TERMS, &meaning), "{word}");
            } else {
                assert!(word["saving"].as_i64() > Some(0), "{word}");
            }
        }

        // Every state line holds the group; the last one counts every word.
        let groups: Vec<&serde_json::Value> = states.iter().map(|s| &s["lexicon"]).collect();
        let last = groups.last().expect("a state");

        assert!(groups.iter().all(|g| g.is_object()), "arm {arm}");
        assert_eq!(groups[19]["coined"], 0, "arm {arm}");
        assert_eq!(last["coined"].as_u64(), Some(coined.len() as u64));
        assert_eq!(last["retired"].as_u64(), Some(retired.len() as u64));
        assert_eq!(
            last["words"].as_u64(),
            Some((coined.len() - retired.len()) as u64)
        );
    }
}

/// The events of `events` as a Run resumed to more generations and a straight Run share
/// them: without the finalists of every select phase but the last, without the cumulative
/// calls of the generation events, which count the calls of an earlier select phase too, and
/// without the number of generations the first process was given.
fn shared(events: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
    let last = events
        .iter()
        .rposition(|e| e["kind"] == "generation")
        .expect("a generation");

    events
        .into_iter()
        .enumerate()
        .filter(|(i, e)| *i > last || e["kind"] != "finalist")
        .map(|(_, mut e)| {
            if let Some(event) = e.as_object_mut() {
                event.remove("calls_used");
            }

            if let Some(config) = e.get_mut("config").and_then(|c| c.as_object_mut()) {
                config.remove("generations");
            }

            e
        })
        .collect()
}

#[test]
fn a_resumed_run_equals_a_straight_one() {
    // In each arm, 60 generations straight, and 30 generations that are then resumed to 60:
    // the same events but the first select phase's finalists, among them every coined and
    // retired word, the same archive, the same rediscovery count, and the same report but for
    // the calls.
    let root = fresh("resume");

    for arm in ["N", "R", "L"] {
        let straight = root.join(format!("{arm}-straight"));
        let split = root.join(format!("{arm}-split"));

        ok(&arm_run(&straight, arm, "6417", "4", "60"));
        ok(&arm_run(&split, arm, "6417", "4", "30"));
        ok(&breed(&[
            "--out",
            path(&split),
            "--resume",
            "--generations",
            "60",
        ]));
        ok(&breed(&["--out", path(&split), "--verify"]));

        let (a, b) = (logged_events(&straight), logged_events(&split));

        assert!(b.len() > a.len(), "arm {arm}");

        // The resumed Run coined words after generation 30 too, which its Lexicon, replayed
        // from the archive, decided.
        if arm != "N" {
            let late = of_kind(&b, "coined")
                .iter()
                .chain(&of_kind(&b, "retired"))
                .filter(|e| e["generation"].as_u64() > Some(30))
                .count();

            assert!(
                late > 0,
                "arm {arm}: no coining step after the resume did anything"
            );
        }

        assert_eq!(shared(a), shared(b), "arm {arm}");
        assert_eq!(
            read(&straight, "rediscovery.json"),
            read(&split, "rediscovery.json"),
            "arm {arm}"
        );
        assert_eq!(read(&straight, "config.json"), read(&split, "config.json"));

        let snapshot = |dir: &Path| json(dir, "snapshot.json")["snapshot"].clone();
        let (sa, sb) = (snapshot(&straight), snapshot(&split));

        for key in ["archive", "comparisons", "next_generation", "finished"] {
            assert_eq!(sa[key], sb[key], "arm {arm}: {key}");
        }

        let (mut ra, mut rb) = (json(&straight, "report.json"), json(&split, "report.json"));

        assert!(rb["calls_used"].as_u64() > ra["calls_used"].as_u64());
        ra["calls_used"] = serde_json::Value::Null;
        rb["calls_used"] = serde_json::Value::Null;
        assert_eq!(ra, rb, "arm {arm}");
    }
}

/// One line of the event log, as the binary writes it.
#[derive(Serialize, Deserialize)]
struct LogLine {
    seq: u64,
    prev: String,
    hash: String,
    event: Event,
}

/// The log of `dir` with `change` applied to its events and the hash chain written anew,
/// and the checkpoint's anchor moved to the new chain: a log that only the Lexicon's replay
/// can tell from the record.
fn rewrite_log(dir: &Path, change: impl Fn(&mut Event)) {
    let text = String::from_utf8(read(dir, "events.jsonl")).expect("utf-8 log");
    let mut prev = "0000000000000000000000000000000000000000000000000000000000000000".to_string();
    let mut out = String::new();
    let mut hashes = Vec::new();

    for (seq, line) in text.lines().enumerate() {
        let mut entry: LogLine = serde_json::from_str(line).expect("a log entry");

        change(&mut entry.event);

        let event_json = serde_json::to_string(&entry.event).expect("json");
        let hash = sha256_hex(format!("{prev}\n{event_json}").as_bytes());

        entry.seq = seq as u64;
        entry.prev = prev.clone();
        entry.hash = hash.clone();
        out.push_str(&serde_json::to_string(&entry).expect("json"));
        out.push('\n');
        hashes.push(hash.clone());
        prev = hash;
    }

    std::fs::write(dir.join("events.jsonl"), out).expect("write log");

    let mut checkpoint = json(dir, "snapshot.json");
    let len = checkpoint["log_len"].as_u64().expect("log_len") as usize;

    checkpoint["log_head"] = serde_json::json!(hashes[len - 1]);
    std::fs::write(
        dir.join("snapshot.json"),
        serde_json::to_string(&checkpoint).expect("json"),
    )
    .expect("write checkpoint");
}

/// A named change to the events of a log.
type Change = (&'static str, fn(&mut Event));

#[test]
fn verify_and_a_resume_find_a_changed_word() {
    let root = fresh("verify");
    let dir = root.join("run");

    ok(&arm_run(&dir, "L", "6421", "4", "40"));
    ok(&breed(&["--out", path(&dir), "--verify"]));
    assert!(
        !of_kind(&logged_events(&dir), "coined").is_empty(),
        "no word coined"
    );

    let original = RECORD.map(|name| read(&dir, name));
    let changes: [Change; 3] = [
        ("saving", |e| {
            if let Event::Coined { saving, .. } = e {
                *saving += 1;
            }
        }),
        ("section", |e| {
            if let Event::Coined { section, .. } = e {
                section.push('s');
            }
        }),
        ("phrase", |e| {
            if let Event::Coined { phrase, .. } = e {
                *phrase = phrase.replacen("Use ", "Try ", 1);
            }
        }),
    ];

    for (what, change) in changes {
        rewrite_log(&dir, change);

        // The rewritten log keeps its hash chain and its anchor, and the report and the
        // archive still match it.
        let refused = breed(&["--out", path(&dir), "--verify"]);

        assert_eq!(
            refused.status.code(),
            Some(1),
            "{what}: {}",
            stdout(&refused)
        );
        assert!(
            stderr(&refused).contains("integrity:")
                && stderr(&refused).contains("coined and retired events differ"),
            "{what}: {}",
            stderr(&refused)
        );

        // A resume replays the Lexicon from the checkpoint and refuses to go on, before it
        // changes a file.
        let before = RECORD.map(|name| read(&dir, name));
        let resumed = breed(&["--out", path(&dir), "--resume", "--generations", "41"]);

        assert_eq!(
            resumed.status.code(),
            Some(1),
            "{what}: {}",
            stdout(&resumed)
        );
        assert!(
            stderr(&resumed).contains("coined and retired events differ"),
            "{what}: {}",
            stderr(&resumed)
        );
        assert_eq!(RECORD.map(|name| read(&dir, name)), before, "{what}");

        for (name, bytes) in RECORD.iter().zip(&original) {
            std::fs::write(dir.join(name), bytes).expect("restore");
        }
    }

    ok(&breed(&["--out", path(&dir), "--verify"]));
}

/// The names of the files in `dir`.
fn left_in(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn the_settings_are_refused_where_they_cannot_work() {
    // Each refusal comes before the Run writes a record: as with every other refusal of a
    // fresh Run's settings, only the empty run directory and its LOCK file are left.
    let root = fresh("refused");
    let write = |name: &str, cfg: &BreedConfig| {
        let file = root.join(format!("{name}.json"));

        std::fs::create_dir_all(&root).expect("mkdir");
        std::fs::write(&file, serde_json::to_string_pretty(cfg).expect("json")).expect("write");

        file
    };
    let lexicon = write(
        "lexicon",
        &BreedConfig {
            lexicon: Some(LexiconConfig::arm_l()),
            ..BreedConfig::default()
        },
    );
    let brevity = write(
        "brevity",
        &BreedConfig {
            brevity: true,
            ..BreedConfig::default()
        },
    );
    let cases: [(&str, &Path, &[&str], &str); 5] = [
        (
            "a lexicon on grids",
            &lexicon,
            &["--domain", "grid"],
            "lexicon needs a gene pool",
        ),
        (
            "brevity with dgm",
            &brevity,
            &["--domain", "circuits"],
            "brevity needs lexicase selection",
        ),
        (
            "brevity with tournaments",
            &brevity,
            &["--domain", "circuits", "--selection", "tournament"],
            "brevity needs lexicase selection",
        ),
        (
            "brevity with an epsilon of 1",
            &brevity,
            &[
                "--domain",
                "circuits",
                "--selection",
                "lexicase",
                "--lexicase-epsilon",
                "1",
            ],
            "brevity needs lexicase selection with a lexicase_epsilon below 1",
        ),
        (
            "brevity on grids with an epsilon of 1",
            &brevity,
            &["--selection", "lexicase", "--lexicase-epsilon", "1.5"],
            "lexicase_epsilon below 1",
        ),
    ];

    for (k, (what, config, flags, reason)) in cases.into_iter().enumerate() {
        let run = root.join(format!("run-{k}"));
        let mut args = vec!["--out", path(&run), "--config", path(config)];

        args.extend_from_slice(flags);

        let refused = breed(&args);

        assert_eq!(
            refused.status.code(),
            Some(1),
            "{what}: {}",
            stderr(&refused)
        );
        assert!(
            stderr(&refused).contains(reason),
            "{what}: {}",
            stderr(&refused)
        );
        assert!(
            left_in(&run).iter().all(|name| name == "LOCK"),
            "{what}: {:?}",
            left_in(&run)
        );
    }

    // On circuits with lexicase selection both start, and record their settings; brevity
    // starts on grids too.
    for (config, key, domain) in [
        (&lexicon, "lexicon", "circuits"),
        (&brevity, "brevity", "circuits"),
        (&brevity, "brevity", "grid"),
    ] {
        let run = root.join(format!("starts-{key}-{domain}"));

        std::fs::create_dir_all(&run).expect("mkdir");
        std::fs::write(run.join("STOP"), "").expect("stop");
        ok(&breed(&[
            "--out",
            path(&run),
            "--config",
            path(config),
            "--domain",
            domain,
            "--selection",
            "lexicase",
        ]));
        assert!(!json(&run, "config.json")[key].is_null(), "{key}");
    }
}
