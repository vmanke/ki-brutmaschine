//! Tests of experience, imagination and changing goals on the `brut-breed` binary (arms I, G
//! and IG, `.scratch/seelen-modell/spec-imagination.md`, tickets 04 and 05). With both
//! settings off nothing a Run writes changes, which the golden tests of `tests/universe.rs`,
//! the drafting digests of `arena.rs` and the release tests of `tests/soul.rs` check. Here:
//! the settings are left out when off and recorded when on; Runs of 60 generations with the
//! flags of the screen of ticket 06 write the same bytes at 1 and at 4 threads, pass
//! `--verify`, and resume like straight Runs; `--verify` finds changed values; and the
//! settings are refused where they cannot work.
//!
//! No test here runs a seed of a screen or of the pilots: the screen of ticket 06 uses seeds
//! 5001 to 5008, and these tests use seeds above 6200.
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use brutmaschine::cascade::arena::BreedConfig;
use brutmaschine::cascade::circuit::{Answered, CircuitDomain, Rediscovery, rediscovery};
use brutmaschine::cascade::domain::Domain;
use brutmaschine::cascade::mind::{GoalsConfig, ImaginationConfig};
use brutmaschine::cascade::qubit::Circuit;
use brutmaschine::util::derive_seed;

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
        .join("mind")
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

/// The flags of the screen of ticket 06, for 60 generations on `seed`.
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

/// The config file of arm `arm` (N, G, I or IG) in `dir`, written as ticket 06 writes it: the
/// `config.json` that the binary writes for arm N with the screen's flags, here from a Run
/// stopped before its generation 0, with `goals` (G), `imagination` (I), or both (IG).
fn arm_config(dir: &Path, arm: &str) -> PathBuf {
    let n = dir.join("arm-N");

    if !n.join("config.json").exists() {
        std::fs::create_dir_all(&n).expect("mkdir");
        std::fs::write(n.join("STOP"), "").expect("stop");

        let mut args = vec!["--out", path(&n)];

        args.extend(screen("6200"));
        ok(&breed(&args));
    }

    let mut cfg = json(&n, "config.json");

    if arm.contains('G') {
        cfg["goals"] = serde_json::to_value(GoalsConfig::arm_g()).expect("json");
    }

    if arm.contains('I') {
        cfg["imagination"] = serde_json::to_value(ImaginationConfig::arm_i()).expect("json");
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

/// The rediscovery count of the Run in `dir`, recomputed from its log alone, as
/// `tests/circuits.rs` does it.
fn recount(dir: &Path) -> Rediscovery {
    let cfg: BreedConfig = serde_json::from_slice(&read(dir, "config.json")).expect("config");
    let domain = CircuitDomain::from_config(&cfg).expect("a circuit run");
    let panel = domain.panel(derive_seed(cfg.seed, "panel/search"), cfg.search_puzzles);
    let log = logged_events(dir);
    let born: Vec<(String, bool, usize, Vec<Option<Circuit>>)> = of_kind(&log, "born")
        .iter()
        .map(|e| {
            let answers = match e["answers"].as_array() {
                Some(texts) => texts
                    .iter()
                    .map(|t| t.as_str().map(|s| s.parse::<Circuit>().expect("a circuit")))
                    .collect(),
                None => vec![None; panel.len()],
            };

            (
                e["id"].as_str().expect("id").to_string(),
                e["op"] == "seed" && e["generation"] == 0,
                e["generation"].as_u64().expect("generation") as usize,
                answers,
            )
        })
        .collect();
    let members: Vec<Answered<'_>> = born
        .iter()
        .map(|(id, seed, generation, answers)| Answered {
            id,
            seed: *seed,
            generation: *generation,
            answers,
        })
        .collect();

    rediscovery(&panel, &members).expect("no integrity alarm")
}

#[test]
fn imagination_and_goals_are_left_out_when_off() {
    let off = BreedConfig::default();
    let text = serde_json::to_string(&off).expect("json");

    assert!(off.imagination.is_none() && off.goals.is_none());
    assert!(
        !text.contains("imagination") && !text.contains("goals"),
        "{text}"
    );

    // On, they are recorded at the end of the settings and define the experiment.
    let on = BreedConfig {
        imagination: Some(ImaginationConfig::arm_i()),
        goals: Some(GoalsConfig::arm_g()),
        ..BreedConfig::default()
    };
    let written = serde_json::to_string(&on).expect("json");

    assert!(
        written.ends_with(r#","imagination":{"draws":4,"rate":250},"goals":{"length":20}}"#),
        "{written}"
    );
    assert_eq!(
        serde_json::from_str::<BreedConfig>(&written).expect("read"),
        on
    );
    assert_ne!(on.fixed_hash(), off.fixed_hash());

    for one in [
        BreedConfig {
            goals: None,
            ..on.clone()
        },
        BreedConfig {
            imagination: None,
            ..on.clone()
        },
    ] {
        assert_ne!(one.fixed_hash(), off.fixed_hash());
        assert_ne!(one.fixed_hash(), on.fixed_hash());
    }

    // An unknown or a missing field of either setting is refused.
    for bad in [
        written.replace(r#""rate":250"#, r#""rate":250,"depth":2"#),
        written.replace(r#","rate":250"#, ""),
        written.replace(r#""length":20"#, r#""length":20,"start":1"#),
        written.replace(r#"{"length":20}"#, "{}"),
    ] {
        assert_ne!(bad, written);
        assert!(serde_json::from_str::<BreedConfig>(&bad).is_err(), "{bad}");
    }
}

#[test]
fn a_run_with_imagination_or_goals_reproduces() {
    // 60 generations with the screen's flags in each arm: --verify passes, the five record
    // files are byte-equal at 1 and at 4 threads, and rediscovery.json is the count of the
    // logged answers, as without the settings.
    let root = fresh("reproduce");

    for arm in ["I", "G", "IG"] {
        let at = |threads: &str| {
            let dir = root.join(format!("{arm}-{threads}"));

            ok(&arm_run(&dir, arm, "6211", threads, "60"));

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
        let minds: Vec<&serde_json::Value> = states.iter().map(|s| &s["mind"]).collect();
        let learned = of_kind(&events, "learned");
        let born = of_kind(&events, "born");

        assert_eq!(states.len(), 61, "arm {arm}");
        assert_eq!(
            read(&dir, "rediscovery.json"),
            serde_json::to_string_pretty(&recount(&dir))
                .expect("json")
                .as_bytes(),
            "arm {arm}: rediscovery.json"
        );

        // The goal of each generation: all tasks to generation 20, then the targets that do
        // not name Linear, then all tasks again. Without goals it is always all tasks.
        for (g, mind) in minds.iter().enumerate() {
            let goal = if arm.contains('G') && (21..=40).contains(&g) {
                "-Linear"
            } else {
                "all"
            };

            assert_eq!(mind["goal"], goal, "arm {arm}, generation {g}");
        }

        if arm.contains('I') {
            assert_eq!(config["imagination"]["draws"], 4);
            assert_eq!(learned.len(), 61, "arm {arm}");
            assert!(born.iter().all(|b| b["values"].is_array()), "arm {arm}");

            // Imagination changed picks, never more than it made choices.
            let count = |field: &str| -> u64 {
                minds
                    .iter()
                    .map(|m| m[field].as_u64().unwrap_or_else(|| panic!("{field}")))
                    .sum()
            };

            assert!(count("imagination_changed") > 0, "arm {arm}");
            assert!(count("imagination_changed") <= count("imagination_choices"));
            assert!(born.iter().any(|b| b["effect"]["draw"].as_u64() > Some(1)));
        } else {
            assert!(config.get("imagination").is_none());
            assert!(learned.is_empty());
            assert!(
                born.iter()
                    .all(|b| b.get("values").is_none() && b.get("effect").is_none())
            );
            assert!(minds.iter().all(|m| m.get("imagination_choices").is_none()));
        }

        if arm.contains('G') {
            assert_eq!(config["goals"]["length"], 20);
        } else {
            assert!(config.get("goals").is_none());
        }
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
    // the same events but the first select phase's finalists, the same archive with its
    // learned values, the same rediscovery count, and the same report but for the calls.
    let root = fresh("resume");

    for arm in ["I", "G", "IG"] {
        let straight = root.join(format!("{arm}-straight"));
        let split = root.join(format!("{arm}-split"));

        ok(&arm_run(&straight, arm, "6217", "4", "60"));
        ok(&arm_run(&split, arm, "6217", "4", "30"));
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

/// A named change to an archive member of a checkpoint.
type Tamper = (&'static str, fn(&mut serde_json::Value));

/// One thousandth more, or less at the top of the range: a learned value still in range.
fn bump(v: &mut serde_json::Value) {
    let x = v.as_i64().expect("a value");

    *v = serde_json::json!(if x < 1000 { x + 1 } else { x - 1 });
}

#[test]
fn verify_finds_changed_values() {
    let root = fresh("verify");
    let dir = root.join("run");
    let config = arm_config(&root, "IG");

    ok(&breed(&[
        "--out",
        path(&dir),
        "--config",
        path(&config),
        "--generations",
        "6",
        "--search",
        "11",
        "--select",
        "11",
        "--seed",
        "6221",
    ]));
    ok(&breed(&["--out", path(&dir), "--verify"]));

    let file = dir.join("snapshot.json");
    let original = std::fs::read_to_string(&file).expect("snapshot");
    let checkpoint: serde_json::Value = serde_json::from_str(&original).expect("json");
    let archive = checkpoint["snapshot"]["archive"]
        .as_array()
        .expect("archive");
    let moved: Vec<(String, usize)> = of_kind(&logged_events(&dir), "learned")
        .iter()
        .flat_map(|e| e["changed"].as_array().expect("changed").clone())
        .map(|c| {
            (
                c["member"].as_str().expect("member").to_string(),
                c["worker"].as_u64().expect("worker") as usize,
            )
        })
        .collect();

    assert!(!moved.is_empty(), "no learned event moved a value");

    // A member whose first Worker a learned event moved, one whose values never moved, and
    // one whose Born event holds what its learning read.
    let place = |id: &str| {
        archive
            .iter()
            .position(|m| m["id"] == id)
            .expect("an archived member")
    };
    let was_moved = moved.iter().find(|(_, k)| *k == 0).map(|(id, _)| place(id));
    let never_moved = archive.iter().position(|m| {
        !moved
            .iter()
            .any(|(id, _)| m["id"].as_str() == Some(id.as_str()))
    });
    let with_effect = archive.iter().position(|m| m["effect"].is_object());
    let tampers: [(Option<usize>, Tamper); 4] = [
        (
            was_moved,
            ("moved current values", |m| {
                bump(&mut m["values_now"][0][0]);
            }),
        ),
        (
            never_moved,
            ("current values never moved", |m| {
                bump(&mut m["values_now"][0][7]);
            }),
        ),
        (
            never_moved,
            ("values at birth", |m| {
                bump(&mut m["values"][0][7]);
                bump(&mut m["values_now"][0][7]);
            }),
        ),
        (
            with_effect,
            ("the prediction", |m| {
                let p = m["effect"]["p"].as_i64().expect("p");

                m["effect"]["p"] = serde_json::json!(p + 1);
            }),
        ),
    ];

    for (at, (what, tamper)) in tampers {
        let i = at.unwrap_or_else(|| panic!("no member for {what}"));
        let id = archive[i]["id"].as_str().expect("id").to_string();
        let mut changed = checkpoint.clone();

        tamper(&mut changed["snapshot"]["archive"][i]);
        assert_ne!(changed, checkpoint, "{what}");
        std::fs::write(&file, serde_json::to_string(&changed).expect("json")).expect("write");

        let refused = breed(&["--out", path(&dir), "--verify"]);

        assert_eq!(
            refused.status.code(),
            Some(1),
            "{what}: {}",
            stdout(&refused)
        );
        assert!(
            stderr(&refused).contains("integrity:") && stderr(&refused).contains(&id),
            "{what}: {}",
            stderr(&refused)
        );
    }

    std::fs::write(&file, original).expect("restore");
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
    let imagining = write(
        "imagining",
        &BreedConfig {
            imagination: Some(ImaginationConfig::arm_i()),
            ..BreedConfig::default()
        },
    );
    let goals = write(
        "goals",
        &BreedConfig {
            goals: Some(GoalsConfig::arm_g()),
            ..BreedConfig::default()
        },
    );
    let cases: [(&str, &Path, &[&str], &str); 4] = [
        (
            "imagination on grids",
            &imagining,
            &["--domain", "grid"],
            "imagination needs a gene pool",
        ),
        (
            "goals with dgm",
            &goals,
            &["--domain", "circuits"],
            "goals need lexicase selection",
        ),
        (
            "goals with tournaments",
            &goals,
            &["--domain", "circuits", "--selection", "tournament"],
            "goals need lexicase selection",
        ),
        (
            "goals on grids",
            &goals,
            &["--domain", "grid", "--selection", "lexicase"],
            "goals need lexicase selection and the circuits Domain",
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

    // On circuits with lexicase selection both start, and record their settings.
    for (config, key) in [(&imagining, "imagination"), (&goals, "goals")] {
        let run = root.join(format!("starts-{key}"));

        std::fs::create_dir_all(&run).expect("mkdir");
        std::fs::write(run.join("STOP"), "").expect("stop");
        ok(&breed(&[
            "--out",
            path(&run),
            "--config",
            path(config),
            "--domain",
            "circuits",
            "--selection",
            "lexicase",
        ]));
        assert!(json(&run, "config.json")[key].is_object(), "{key}");
    }
}
