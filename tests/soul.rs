//! Step 2 tests (Soul and Coupling strength, `.scratch/seele-und-kappa/`). With the default
//! settings there is no Soul, and nothing a Run writes may change (spec section 14). The
//! end-to-end tests of ticket 09 run `brut-breed` with `--soul`, check `--verify` on Runs
//! with Souls, and check `examples/soul_character.rs` against the Character of the record.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use brutmaschine::cascade::arena::{BreedConfig, worker_request_with};
use brutmaschine::cascade::circuit::{
    Answered, CircuitDomain, CircuitTask, Rediscovery, rediscovery,
};
use brutmaschine::cascade::circuit_genes::circuit_pool;
use brutmaschine::cascade::domain::{CircuitConfig, Domain, DomainConfig, GenePool, GridDomain};
use brutmaschine::cascade::genome::Cascade;
use brutmaschine::cascade::puzzle::Rule;
use brutmaschine::cascade::qubit::{Circuit, Gate};
use brutmaschine::cascade::soul::{SoulConfig, Strengths, character, character_norm};
use brutmaschine::util::{derive_seed, sha256_hex};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/universe")
        .join(format!("{name}.txt"))
}

/// The ids of the seed Cascades of both gene pools, recorded before step 2 changed `Traits`.
const GRID_SEED_IDS: [&str; 4] = [
    "f39cb429fa9b9363",
    "45379dd30abebb66",
    "3db3dfb2e3603931",
    "3e52074d3f6376e6",
];
const CIRCUIT_SEED_IDS: [&str; 4] = [
    "c32087caaa6dcfcc",
    "b3c5bb0d0eef2a79",
    "1564f914d36794c7",
    "2f72c0c6dd7d2423",
];

/// The digests of `prompt_digest`, recorded before step 2 gave the Domains their families
/// and strengths.
const GRID_PROMPTS: &str = "85070b8ce13723e1f6e08c3a618ac8604dbe4f924dc0ee386788617e6b1be0f8";
const CIRCUIT_PROMPTS: &str = "d7dbfa5395805d4f3c607537e75a3f03dc1d0a69e913ddbf6c95f4ff9e9b7c73";

fn ids(seeds: &[(String, Cascade)]) -> Vec<String> {
    seeds.iter().map(|(_, c)| c.id()).collect()
}

/// The ids of the `born` events of generation 0 in a golden digest of `tests/universe.rs`.
fn seed_ids_in(name: &str) -> Vec<String> {
    std::fs::read_to_string(fixture(name))
        .expect("fixture")
        .lines()
        .filter_map(|l| l.strip_prefix("born g=0 id="))
        .map(|rest| rest.split(' ').next().unwrap_or_default().to_string())
        .collect()
}

#[test]
fn old_traits_write_back_byte_equal() {
    let grid = (GenePool::grid().seeds)();
    let circuits = (circuit_pool().seeds)();

    assert_eq!(ids(&grid), GRID_SEED_IDS, "grid seeds");
    assert_eq!(ids(&circuits), CIRCUIT_SEED_IDS, "circuit seeds");

    // The golden runs of `tests/universe.rs` were recorded before step 1; their seeds keep
    // their ids.
    for name in ["grid_run", "grid_llm_run"] {
        assert_eq!(seed_ids_in(name), ids(&grid), "{name}");
    }

    for (name, c) in grid.iter().chain(&circuits) {
        for json in [
            serde_json::to_string(c).expect("json"),
            serde_json::to_string_pretty(c).expect("pretty json"),
        ] {
            let back: Cascade = serde_json::from_str(&json).expect("parse");

            assert_eq!(&back, c, "{name}");
            assert_eq!(back.id(), c.id(), "{name}");
            assert_eq!(
                serde_json::to_string(&back).expect("json"),
                serde_json::to_string(c).expect("json"),
                "{name}"
            );
        }
    }
}

/// One digest over every prompt and Worker request a search panel of `domain` gives. Per
/// task: the solve prompt without and with an earlier answer and the judge prompt for the
/// answers of `pair`, each with the request of every seed Worker, with and without
/// self-check.
fn prompt_digest<D: Domain>(
    domain: &D,
    panel: &[D::Task],
    pair: impl Fn(&D::Task) -> (D::Answer, D::Answer),
    seeds: &[(String, Cascade)],
) -> String {
    let mut all = String::new();

    for task in panel {
        let (a, b) = pair(task);
        let prompts = [
            domain.solve_prompt(task, &[]),
            domain.solve_prompt(task, std::slice::from_ref(&a)),
            domain.judge_prompt(task, &a, &b),
        ];

        for prompt in prompts {
            all.push_str(&prompt);
            all.push('\n');

            for w in seeds
                .iter()
                .flat_map(|(_, c)| c.stages.iter().flat_map(|s| &s.workers))
            {
                for check in [false, true] {
                    let mut w = w.clone();

                    w.traits.self_check = check;

                    let req = worker_request_with(&w, prompt.clone(), 7, domain.self_check_text());

                    all.push_str(&serde_json::to_string(&req).expect("request json"));
                    all.push('\n');
                }
            }
        }
    }

    sha256_hex(all.as_bytes())
}

#[test]
fn strengths_never_reach_a_prompt() {
    // The search panel of the default grid settings (seed 1, 40 puzzles).
    let grid = GridDomain;
    let puzzles = grid.panel(derive_seed(1, "panel/search"), 40);
    let grid_digest = prompt_digest(
        &grid,
        &puzzles,
        |p| (p.test_output.clone(), p.test_input.clone()),
        &(GenePool::grid().seeds)(),
    );

    // The search panel of the step 1 pilot (seed 101, 33 tasks, every verified target).
    let circuits = CircuitDomain::new(&CircuitConfig::default()).expect("circuit domain");
    let tasks = circuits.panel(derive_seed(101, "panel/search"), 33);
    let circuit = |gates: Vec<Gate>| Circuit::new(gates).expect("circuit");
    let circuit_digest = prompt_digest(
        &circuits,
        &tasks,
        |_| {
            (
                circuit(vec![Gate::Cnot {
                    control: 0,
                    target: 1,
                }]),
                circuit(vec![Gate::H(0), Gate::T(1)]),
            )
        },
        &(circuit_pool().seeds)(),
    );

    assert_eq!(grid_digest, GRID_PROMPTS, "grid prompts");
    assert_eq!(circuit_digest, CIRCUIT_PROMPTS, "circuit prompts");
}

/// The head hash of the event log of the step 1 pilot Run of arm A on seed 101
/// (`docs/pilot-step1/pilot.json`).
const PILOT_A_101_HEAD: &str = "e93ed5212dfa6eef09cdfe56fd5e3363f9865995db3f18d79ef4898bd1920ed9";

const BREED: &str = env!("CARGO_BIN_EXE_brut-breed");

/// A fresh, empty directory under the test target directory.
fn fresh(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("soul")
        .join(name);

    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("clean");
    }

    std::fs::create_dir_all(dir.parent().expect("parent")).expect("mkdir");

    dir
}

/// The last line of a Run's event log, as JSON.
fn last_log_line(dir: &Path) -> serde_json::Value {
    let log = std::fs::read_to_string(dir.join("events.jsonl")).expect("log");
    let last = log.lines().last().expect("a log line");

    serde_json::from_str(last).expect("json")
}

fn breed(args: &[&str]) -> Output {
    Command::new(BREED)
        .args(args)
        .output()
        .expect("run brut-breed")
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

/// A Run with the settings of arm S (spec 12.2).
fn souled_config() -> BreedConfig {
    BreedConfig {
        soul: Some(SoulConfig::arm_s()),
        ..BreedConfig::default()
    }
}

#[test]
fn soul_settings_are_left_out_when_off() {
    let cfg = BreedConfig::default();
    let json = serde_json::to_string(&cfg).expect("json");
    let pretty = serde_json::to_string_pretty(&cfg).expect("pretty json");
    let recorded = std::fs::read_to_string(fixture("settings")).expect("settings fixture");

    assert!(cfg.soul.is_none());
    assert!(!json.contains("soul"), "{json}");
    assert_eq!(
        recorded,
        format!(
            "fixed_hash={}\npretty_sha={}\n",
            cfg.fixed_hash(),
            sha256_hex(pretty.as_bytes())
        )
    );

    // With Souls the settings are recorded, and they define the experiment.
    let souled = souled_config();
    let text = serde_json::to_string(&souled).expect("json");

    assert!(text.contains(r#""soul":{"readout":"complex","#), "{text}");
    assert_eq!(
        serde_json::from_str::<BreedConfig>(&text).expect("read"),
        souled
    );
    assert_ne!(souled.fixed_hash(), cfg.fixed_hash());
    assert!(souled.check().is_ok());

    // A soul object with an unknown or a missing field is refused.
    let unknown = text.replace(r#""teach":false"#, r#""teach":false,"lessons":true"#);
    let missing = text.replace(r#","teach":false"#, "");

    for bad in [unknown, missing] {
        assert_ne!(bad, text);
        assert!(serde_json::from_str::<BreedConfig>(&bad).is_err(), "{bad}");
    }

    // A setting out of range fails the check of the whole config.
    let wild = BreedConfig {
        soul: Some(SoulConfig {
            kappa_seed: 1001,
            ..SoulConfig::arm_s()
        }),
        ..BreedConfig::default()
    };

    assert!(wild.check().is_err());
}

/// The settings of arm S as a config file in `dir`.
fn souled_config_file(dir: &Path) -> PathBuf {
    let config = dir.join("souled.json");

    std::fs::create_dir_all(dir).expect("mkdir");
    std::fs::write(
        &config,
        serde_json::to_string_pretty(&souled_config()).expect("json"),
    )
    .expect("write config");

    config
}

/// A short Run of arm S into `dir/run` on the Domain `domain`.
fn souled_run(dir: &Path, domain: &str) -> (PathBuf, Output) {
    let config = souled_config_file(dir);
    let run = dir.join("run");
    let out = breed(&[
        "--out",
        run.to_str().expect("utf-8 path"),
        "--config",
        config.to_str().expect("utf-8 path"),
        "--domain",
        domain,
        "--generations",
        "2",
        "--children",
        "4",
        "--search",
        "11",
        "--select",
        "11",
    ]);

    (run, out)
}

/// The events of a Run's log, as JSON values.
fn logged_events(dir: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(dir.join("events.jsonl"))
        .expect("log")
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).expect("json")["event"].clone())
        .collect()
}

#[test]
fn a_run_with_souls_starts_on_both_domains() {
    for (domain, families) in [("grid", 7), ("circuits", 8)] {
        let (run, out) = souled_run(&fresh(&format!("souled-{domain}")), domain);

        assert!(out.status.success(), "{domain}: {}", stderr(&out));

        let config = std::fs::read_to_string(run.join("config.json")).expect("config");

        assert!(config.contains(r#""soul": {"#), "{config}");

        // Every Worker of every newborn has a Soul of a phase per family, and its Born
        // event says where each came from.
        let born: Vec<serde_json::Value> = logged_events(&run)
            .into_iter()
            .filter(|e| e["kind"] == "born")
            .collect();

        assert!(born.len() > 4, "{domain}");

        for b in &born {
            let workers: Vec<&serde_json::Value> = b["cascade"]["stages"]
                .as_array()
                .expect("stages")
                .iter()
                .flat_map(|s| s["workers"].as_array().expect("workers"))
                .collect();
            let sources = b["soul_sources"].as_array().expect("soul sources");

            assert_eq!(sources.len(), workers.len(), "{domain}");

            for w in workers {
                let phases = w["traits"]["phase_state"].as_array().expect("a Soul");

                assert_eq!(phases.len(), families, "{domain}");
            }
        }

        // The circuit Run's checkpoint matches its log, Soul records included.
        let verified = breed(&["--out", run.to_str().expect("utf-8 path"), "--verify"]);

        assert!(verified.status.success(), "{domain}: {}", stderr(&verified));
    }
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
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

/// A fresh Run into `dir` on the Domain `domain` with the preset `--soul arm` and the
/// settings `settings`.
fn arm_run(dir: &Path, domain: &str, arm: &str, settings: &[&str]) -> Output {
    let mut args = vec!["--out", path(dir), "--domain", domain, "--soul", arm];

    args.extend_from_slice(settings);

    breed(&args)
}

/// A short Run: two generations of four children on 11 search and 11 select tasks.
const SHORT: [&str; 8] = [
    "--generations",
    "2",
    "--children",
    "4",
    "--search",
    "11",
    "--select",
    "11",
];

/// The `soul` object of arm S, V or K, written out from the table of spec 12.2: arm V reads
/// signs, arm K fixes κ at 0.
fn spec_arm(arm: &str) -> serde_json::Value {
    let (readout, kappa_seed, kappa_fixed) = match arm {
        "s" => ("complex", 200, false),
        "v" => ("real", 200, false),
        "k" => ("complex", 0, true),
        _ => panic!("no arm {arm}"),
    };

    serde_json::json!({
        "readout": readout,
        "strengths": "domain",
        "vote": true,
        "simulator": true,
        "observe_vote": true,
        "observe_peer": true,
        "kappa_seed": kappa_seed,
        "kappa_fixed": kappa_fixed,
        "kappa_drift": 20,
        "copy": true,
        "mix": true,
        "phase_drift": 512,
        "new_soul_theta": 1.0,
        "imprint": true,
        "teach": false
    })
}

fn config_json(dir: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).expect("config"))
        .expect("json")
}

#[test]
fn the_presets_are_the_arms() {
    // A STOP file in place before the start ends the Run before its generation 0, after it
    // wrote its settings.
    let stopped = |name: &str, args: &[&str]| {
        let dir = fresh(name);

        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("STOP"), "").expect("stop");

        let mut all = vec!["--out", path(&dir)];

        all.extend_from_slice(args);
        ok(&breed(&all));

        dir
    };
    let plain = config_json(&stopped("presets-none", &["--domain", "circuits"]));

    assert!(plain.get("soul").is_none(), "{plain}");

    for arm in ["s", "v", "k"] {
        let dir = stopped(
            &format!("presets-{arm}"),
            &["--domain", "circuits", "--soul", arm],
        );
        let mut cfg = config_json(&dir);

        assert_eq!(cfg["soul"], spec_arm(arm), "arm {arm}");

        // --soul sets the soul field and nothing else.
        cfg.as_object_mut().expect("object").remove("soul");
        assert_eq!(cfg, plain, "arm {arm}");

        // It defines the experiment: a resume refuses it and leaves the Run as it was.
        let before = std::fs::read(dir.join("config.json")).expect("config");
        let resumed = breed(&["--out", path(&dir), "--resume", "--soul", arm]);

        assert_eq!(resumed.status.code(), Some(2), "{}", stderr(&resumed));
        assert!(
            stderr(&resumed).contains("a resume may change only"),
            "{}",
            stderr(&resumed)
        );
        assert_eq!(
            std::fs::read(dir.join("config.json")).expect("config"),
            before
        );

        // --verify and --promote take only --out.
        for mode in ["--verify", "--promote"] {
            let refused = breed(&["--out", path(&dir), mode, "--soul", arm]);

            assert_eq!(
                refused.status.code(),
                Some(2),
                "{mode}: {}",
                stderr(&refused)
            );
        }
    }

    // A --config that holds soul settings next to --soul is wrong usage, and so is an
    // unknown arm.
    let files = fresh("presets-files");
    let souled = souled_config_file(&files);

    for (i, args) in [
        vec![
            "--config",
            souled.to_str().expect("utf-8 path"),
            "--soul",
            "s",
        ],
        vec!["--soul", "r"],
        vec!["--soul", "S"],
    ]
    .into_iter()
    .enumerate()
    {
        let dir = fresh(&format!("presets-refused-{i}"));
        let mut all = vec!["--out", path(&dir)];

        all.extend_from_slice(&args);

        let refused = breed(&all);

        assert_eq!(
            refused.status.code(),
            Some(2),
            "{args:?}: {}",
            stderr(&refused)
        );
        assert!(stderr(&refused).contains("--soul"), "{}", stderr(&refused));
        assert!(!dir.join("config.json").exists(), "{args:?}");
    }

    // A --config without soul settings brings the rest, and --soul the Souls.
    let other = files.join("other.json");

    std::fs::write(
        &other,
        serde_json::to_string_pretty(&BreedConfig {
            seed: 7,
            ..BreedConfig::default()
        })
        .expect("json"),
    )
    .expect("write config");

    let dir = stopped(
        "presets-config",
        &[
            "--config",
            other.to_str().expect("utf-8 path"),
            "--soul",
            "k",
        ],
    );
    let cfg = config_json(&dir);

    assert_eq!(cfg["seed"], 7);
    assert_eq!(cfg["soul"], spec_arm("k"));
}

/// A short circuit Run in which the newborns hold complex Stage votes (6 in each arm).
const VOTING: [&str; 12] = [
    "--seed",
    "41",
    "--generations",
    "20",
    "--children",
    "6",
    "--search",
    "11",
    "--select",
    "11",
    "--max-calls",
    "200000",
];

/// The complex Stage votes of all newborns in a log.
fn complex_votes(events: &[serde_json::Value]) -> u64 {
    events
        .iter()
        .filter(|e| e["kind"] == "born")
        .map(|e| e["votes"]["complex"].as_u64().expect("a vote tally"))
        .sum()
}

#[test]
fn a_souled_circuit_run_reproduces() {
    for arm in ["s", "v", "k"] {
        let at = |name: &str, threads: &str| {
            let dir = fresh(&format!("reproduce-{arm}-{name}"));
            let out = Command::new(BREED)
                .args(["--out", path(&dir), "--domain", "circuits", "--soul", arm])
                .args(VOTING)
                .env("RAYON_NUM_THREADS", threads)
                .output()
                .expect("run brut-breed");

            ok(&out);
            ok(&breed(&["--out", path(&dir), "--verify"]));

            (
                RECORD.map(|name| {
                    std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
                }),
                logged_events(&dir),
            )
        };
        let (first, events) = at("first", "4");

        // The Runs hold complex votes, so their amplitudes are in the bytes compared, and a
        // `souls` event after each of the 21 generations.
        assert!(complex_votes(&events) > 0, "arm {arm}: no complex vote");
        assert_eq!(
            events.iter().filter(|e| e["kind"] == "souls").count(),
            21,
            "arm {arm}"
        );
        assert_eq!(
            first,
            at("again", "4").0,
            "arm {arm}: the same command twice"
        );
        assert_eq!(first, at("one-thread", "1").0, "arm {arm}: 1 and 4 threads");
    }
}

/// A named change to a part of a checkpoint: an archive member, or the whole snapshot.
type Tamper = (&'static str, fn(&mut serde_json::Value));

/// One phase more, modulo a whole turn.
fn turn_once(phase: &mut serde_json::Value) {
    let p = phase.as_u64().expect("a phase");

    *phase = serde_json::json!((p + 1) % 65536);
}

#[test]
fn verify_refuses_tampered_souls() {
    for domain in ["circuits", "grid"] {
        let dir = fresh(&format!("tampered-{domain}"));

        ok(&arm_run(&dir, domain, "s", &SHORT));
        ok(&breed(&["--out", path(&dir), "--verify"]));

        let file = dir.join("snapshot.json");
        let original = std::fs::read_to_string(&file).expect("snapshot");
        let checkpoint: serde_json::Value = serde_json::from_str(&original).expect("json");
        let archive = checkpoint["snapshot"]["archive"]
            .as_array()
            .expect("archive");

        // A child, whose first Worker has a source.
        let i = archive
            .iter()
            .position(|m| {
                m["soul_sources"][0]["sources"]
                    .as_array()
                    .is_some_and(|s| !s.is_empty())
            })
            .expect("a child");
        let id = archive[i]["id"].as_str().expect("id").to_string();
        let tampers: [Tamper; 4] = [
            ("birth Soul", |m| {
                turn_once(&mut m["cascade"]["stages"][0]["workers"][0]["traits"]["phase_state"][0])
            }),
            ("imprinted Soul", |m| turn_once(&mut m["imprinted"][0][0])),
            ("source map", |m| {
                let w = &mut m["soul_sources"][0]["sources"][0]["worker"];

                *w = serde_json::json!(w.as_u64().expect("a Worker") + 1);
            }),
            ("current Soul", |m| turn_once(&mut m["soul_now"][0][0])),
        ];

        for (what, tamper) in tampers {
            let mut changed = checkpoint.clone();

            tamper(&mut changed["snapshot"]["archive"][i]);
            assert_ne!(changed, checkpoint, "{domain}: {what}");
            std::fs::write(&file, serde_json::to_string(&changed).expect("json")).expect("write");

            let refused = breed(&["--out", path(&dir), "--verify"]);

            assert_eq!(
                refused.status.code(),
                Some(1),
                "{domain}: {what}: {}",
                stdout(&refused)
            );
            assert!(
                stderr(&refused).contains("integrity:") && stderr(&refused).contains(&id),
                "{domain}: {what}: {}",
                stderr(&refused)
            );
        }

        // A changed current Soul cannot hide behind a checkpoint that says the Run is not
        // finished, or that lost its select-panel answers. A Run with Souls keeps all of
        // them, so a report its checkpoint cannot recompute is refused, on grids too.
        let cuts: [Tamper; 2] = [
            ("not finished", |s| s["finished"] = serde_json::json!(false)),
            ("no select answers", |s| {
                s["select_answers"] = serde_json::json!({})
            }),
        ];

        for (what, cut) in cuts {
            let mut changed = checkpoint.clone();

            turn_once(&mut changed["snapshot"]["archive"][i]["soul_now"][0][0]);
            cut(&mut changed["snapshot"]);
            std::fs::write(&file, serde_json::to_string(&changed).expect("json")).expect("write");

            let refused = breed(&["--out", path(&dir), "--verify"]);

            assert_eq!(
                refused.status.code(),
                Some(1),
                "{domain}: current Soul, {what}: {}",
                stdout(&refused)
            );
            assert!(
                stderr(&refused).contains("integrity:")
                    && stderr(&refused).contains("cannot be recomputed from the checkpoint"),
                "{domain}: current Soul, {what}: {}",
                stderr(&refused)
            );
        }

        std::fs::write(&file, original).expect("restore");
        ok(&breed(&["--out", path(&dir), "--verify"]));
    }
}

/// What `--verify` printed before ticket 09 on a grid Run without Souls of the settings
/// `GRID_SHORT`: finished, and stopped before its generation 0.
const GRID_VERIFIED: &str = "brut-breed: 93 events, hash chain intact, head 13c8f836ab3500222da8771d01b5f0daeb2afe63eeef3b836c1f92e82e7eceaf; checkpoint at entry 93 matches; settings match; report recomputed and equal\n";
const GRID_STOPPED: &str = "brut-breed: 0 events, hash chain intact, head 0000000000000000000000000000000000000000000000000000000000000000; checkpoint at entry 0 matches; settings match\n";

#[test]
fn grid_verify_is_unchanged_without_souls() {
    let finished = fresh("grid-verify-finished");
    let mut args = vec!["--out", path(&finished), "--domain", "grid"];

    args.extend_from_slice(&SHORT);
    ok(&breed(&args));

    let verified = breed(&["--out", path(&finished), "--verify"]);

    ok(&verified);
    assert_eq!(stdout(&verified), GRID_VERIFIED);
    assert_eq!(stderr(&verified), "");

    // A grid Run without Souls may come from before the select answers were kept, so a
    // checkpoint without them still falls back to the check of the finalists.
    let file = finished.join("snapshot.json");
    let original = std::fs::read_to_string(&file).expect("snapshot");
    let mut cut: serde_json::Value = serde_json::from_str(&original).expect("json");

    cut["snapshot"]["select_answers"] = serde_json::json!({});
    std::fs::write(&file, serde_json::to_string(&cut).expect("json")).expect("write");

    let fallback = breed(&["--out", path(&finished), "--verify"]);

    ok(&fallback);
    assert_eq!(
        stdout(&fallback),
        GRID_VERIFIED.replace(
            "report recomputed and equal",
            "report finalists match the log"
        )
    );
    std::fs::write(&file, original).expect("restore");

    let stopped = fresh("grid-verify-stopped");

    std::fs::create_dir_all(&stopped).expect("mkdir");
    std::fs::write(stopped.join("STOP"), "").expect("stop");
    args[1] = path(&stopped);
    ok(&breed(&args));

    let verified = breed(&["--out", path(&stopped), "--verify"]);

    ok(&verified);
    assert_eq!(stdout(&verified), GRID_STOPPED);
    assert_eq!(stderr(&verified), "");
}

#[test]
fn grid_verify_with_souls_says_what_it_checked() {
    let finished = fresh("grid-souled-verify-finished");

    ok(&arm_run(&finished, "grid", "s", &SHORT));

    let verified = breed(&["--out", path(&finished), "--verify"]);

    ok(&verified);
    assert!(
        stdout(&verified)
            .ends_with("; report recomputed and equal; archive and Souls match the log\n"),
        "{}",
        stdout(&verified)
    );

    // Stopped before its generation 0, the Run is not finished, and the line says why its
    // archive was not checked.
    let stopped = fresh("grid-souled-verify-stopped");

    std::fs::create_dir_all(&stopped).expect("mkdir");
    std::fs::write(stopped.join("STOP"), "").expect("stop");
    ok(&arm_run(&stopped, "grid", "s", &SHORT));

    let verified = breed(&["--out", path(&stopped), "--verify"]);

    ok(&verified);
    assert!(
        stdout(&verified).ends_with(
            "; settings match; archive and Souls not checked: the run is not finished\n"
        ),
        "{}",
        stdout(&verified)
    );
}

#[test]
fn promote_refuses_souls() {
    for arm in ["s", "v", "k"] {
        let dir = fresh(&format!("promote-{arm}"));

        ok(&arm_run(&dir, "grid", arm, &SHORT));
        assert!(dir.join("report.json").exists(), "a finished Run");

        let refused = breed(&["--out", path(&dir), "--promote"]);

        assert_eq!(refused.status.code(), Some(1), "{}", stderr(&refused));
        assert!(stderr(&refused).contains("Souls"), "{}", stderr(&refused));
        assert!(!dir.join("promote").exists());
    }
}

/// One archived Cascade as its Born event logs it.
struct Logged {
    id: String,
    seed: bool,
    generation: usize,
    answers: Vec<Option<Circuit>>,
}

/// The rediscovery count of the circuit Run in `dir`, recomputed from its log alone, as
/// `tests/circuits.rs` recounts a Run without Souls: the search panel rebuilt from the
/// recorded settings (checked against the run start), and every Born event's answers read
/// back into circuits.
fn recount(dir: &Path) -> Rediscovery {
    let cfg: BreedConfig =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config"))
            .expect("config json");
    let domain = CircuitDomain::from_config(&cfg).expect("a circuit run");
    let panel = domain.panel(derive_seed(cfg.seed, "panel/search"), cfg.search_puzzles);
    let log = logged_events(dir);
    let ids: Vec<&str> = panel.iter().map(|t| t.id.as_str()).collect();
    let started: Vec<&str> = log[0]["search"]
        .as_array()
        .expect("search ids")
        .iter()
        .map(|v| v.as_str().expect("id"))
        .collect();

    assert_eq!(ids, started, "the rebuilt panel is the logged one");

    let born: Vec<Logged> = log
        .iter()
        .filter(|e| e["kind"] == "born")
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
fn rediscovery_is_unchanged_by_souls() {
    // The settings of the RICH Run of `tests/circuits.rs`, with the Souls of arm S.
    let dir = fresh("rediscovery-souled");

    ok(&arm_run(
        &dir,
        "circuits",
        "s",
        &[
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
        ],
    ));

    let text = std::fs::read_to_string(dir.join("rediscovery.json")).expect("count");

    assert_eq!(
        text,
        serde_json::to_string_pretty(&recount(&dir)).expect("json")
    );
    assert!(
        logged_events(&dir).iter().any(|e| e["kind"] == "souls"),
        "a Run with Souls"
    );
    ok(&breed(&["--out", path(&dir), "--verify"]));
}

#[allow(dead_code)]
#[path = "../examples/soul_character.rs"]
mod soul_character;

/// The label of a grid rule in the example's output: its geometric step, and "+colours"
/// when it carries a colour map.
fn rule_label(rule: &Rule) -> String {
    match rule.colours {
        Some(_) => format!("{:?}+colours", rule.geo),
        None => format!("{:?}", rule.geo),
    }
}

/// The rows of the example's output, by member, Worker and target or rule: the Character
/// and the normalised Character. The header names the column of the targets or rules.
fn character_rows(text: &str, column: &str) -> BTreeMap<(String, usize, String), (f64, f64)> {
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let header = format!("member\tworker\t{column}\tcharacter\tcharacter_norm");

    assert_eq!(lines.next(), Some(header.as_str()), "{text}");

    let mut rows = BTreeMap::new();

    for line in lines {
        let cells: Vec<&str> = line.split('\t').collect();

        assert_eq!(cells.len(), 5, "{line}");

        let key = (
            cells[0].to_string(),
            cells[1].parse().expect("a Worker"),
            cells[2].to_string(),
        );
        let values = (
            cells[3].parse().expect("a Character"),
            cells[4].parse().expect("a normalised Character"),
        );

        assert!(rows.insert(key, values).is_none(), "twice: {line}");
    }

    rows
}

/// Error unless the example's rows for the Run in `dir` hold, for every member and Worker of
/// the checkpoint and every search task, the Character and the normalised Character of the
/// Worker's current Soul with the task's strengths, under the task's label, and nothing
/// else. Returns the number of rows.
fn check_characters<D: Domain>(
    dir: &Path,
    domain: &D,
    label: impl Fn(&D::Task) -> String,
    column: &str,
) -> usize {
    let cfg: BreedConfig =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config"))
            .expect("config json");
    let soul = cfg.soul.clone().expect("a Run with Souls");
    let panel = domain.panel(derive_seed(cfg.seed, "panel/search"), cfg.search_puzzles);
    let checkpoint: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("snapshot.json")).expect("snapshot"))
            .expect("json");
    let text = soul_character::run(dir).expect("the example reads the Run");
    let rows = character_rows(&text, column);
    let mut expected = BTreeSet::new();

    for m in checkpoint["snapshot"]["archive"]
        .as_array()
        .expect("archive")
    {
        let id = m["id"].as_str().expect("id");

        for (k, now) in m["soul_now"]
            .as_array()
            .expect("current Souls")
            .iter()
            .enumerate()
        {
            let phases: Vec<u16> = serde_json::from_value(now.clone()).expect("phases");

            for task in &panel {
                let key = (id.to_string(), k, label(task));
                let s = soul.strengths.of(domain, task);
                let want = (
                    character(&phases, &s, soul.readout),
                    character_norm(&phases, &s, soul.readout),
                );

                assert_eq!(rows.get(&key), Some(&want), "{key:?}");
                expected.insert(key);
            }
        }
    }

    assert_eq!(rows.len(), expected.len(), "rows beyond the archive");

    rows.len()
}

#[test]
fn the_character_example_agrees() {
    // Arm V on grids, arm S on circuits, and arm S with equal strengths on circuits.
    let grid = fresh("character-grid");

    ok(&arm_run(&grid, "grid", "v", &SHORT));
    assert!(check_characters(&grid, &GridDomain, |p| rule_label(&p.rule), "rule") > 0);

    let circuits = CircuitDomain::new(&CircuitConfig::default()).expect("circuit domain");
    let named = |t: &CircuitTask| t.name.to_string();
    let complex = fresh("character-circuits");

    ok(&arm_run(&complex, "circuits", "s", &SHORT));
    assert!(check_characters(&complex, &circuits, named, "target") > 0);

    let equal = fresh("character-equal");
    let config = equal.join("equal.json");

    std::fs::create_dir_all(&equal).expect("mkdir");
    std::fs::write(
        &config,
        serde_json::to_string_pretty(&BreedConfig {
            soul: Some(SoulConfig {
                strengths: Strengths::Equal,
                ..SoulConfig::arm_s()
            }),
            domain: DomainConfig::Circuits(CircuitConfig::default()),
            ..BreedConfig::default()
        })
        .expect("json"),
    )
    .expect("write config");

    let run = equal.join("run");
    let mut args = vec![
        "--out",
        path(&run),
        "--config",
        config.to_str().expect("utf-8 path"),
    ];

    args.extend_from_slice(&SHORT);
    ok(&breed(&args));
    assert!(check_characters(&run, &circuits, named, "target") > 0);

    // A Run without Souls has no Character to show.
    let none = fresh("character-none");
    let mut args = vec!["--out", path(&none), "--domain", "grid"];

    args.extend_from_slice(&SHORT);
    ok(&breed(&args));
    assert!(soul_character::run(&none).is_err());
}

/// The files that record a finished circuit Run.
const RECORD: [&str; 5] = [
    "config.json",
    "events.jsonl",
    "snapshot.json",
    "report.json",
    "rediscovery.json",
];

#[test]
fn a_souled_run_does_not_depend_on_threads() {
    let at = |threads: &str| {
        let dir = fresh(&format!("souled-threads-{threads}"));
        let config = souled_config_file(&dir);
        let run = dir.join("run");
        let out = Command::new(BREED)
            .args(["--out", run.to_str().expect("utf-8 path")])
            .args(["--config", config.to_str().expect("utf-8 path")])
            .args([
                "--domain",
                "circuits",
                "--seed",
                "41",
                "--generations",
                "20",
            ])
            .args(["--children", "6", "--search", "11", "--select", "11"])
            .args(["--max-calls", "200000"])
            .env("RAYON_NUM_THREADS", threads)
            .output()
            .expect("run brut-breed");

        assert!(out.status.success(), "{threads} threads: {}", stderr(&out));

        RECORD.map(|name| std::fs::read(run.join(name)).unwrap_or_else(|e| panic!("{name}: {e}")))
    };
    let one = at("1");

    // The Run holds complex votes, so their amplitudes are in the bytes compared.
    let complex: u64 = String::from_utf8(one[1].clone())
        .expect("utf-8 log")
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).expect("json")["event"].clone())
        .filter(|e| e["kind"] == "born")
        .map(|e| e["votes"]["complex"].as_u64().expect("a vote tally"))
        .sum();

    assert!(complex > 0, "no complex vote");
    assert_eq!(one, at("4"));
}

/// The Workers the `souls` events of a log moved, by member id and Worker index.
fn moved_workers(events: &[serde_json::Value]) -> BTreeSet<(String, u64)> {
    events
        .iter()
        .filter(|e| e["kind"] == "souls")
        .flat_map(|e| e["changed"].as_array().expect("changed").clone())
        .map(|c| {
            (
                c["member"].as_str().expect("member").to_string(),
                c["worker"].as_u64().expect("worker"),
            )
        })
        .collect()
}

#[test]
fn souls_do_not_depend_on_threads() {
    let at = |threads: &str| {
        let dir = fresh(&format!("souls-threads-{threads}"));
        let config = souled_config_file(&dir);
        let run = dir.join("run");
        let out = Command::new(BREED)
            .args(["--out", run.to_str().expect("utf-8 path")])
            .args(["--config", config.to_str().expect("utf-8 path")])
            .args(["--domain", "grid", "--generations", "3"])
            .args(["--children", "6", "--search", "12", "--select", "12"])
            .env("RAYON_NUM_THREADS", threads)
            .output()
            .expect("run brut-breed");

        assert!(out.status.success(), "{threads} threads: {}", stderr(&out));

        let events = logged_events(&run);

        (
            ["events.jsonl", "snapshot.json"].map(|name| {
                std::fs::read(run.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
            }),
            events,
        )
    };
    let (one, events) = at("1");

    // One `souls` event per committed generation, and the Observations moved Souls.
    assert_eq!(events.iter().filter(|e| e["kind"] == "souls").count(), 4);
    assert!(!moved_workers(&events).is_empty());
    assert_eq!(one, at("4").0);
}

#[test]
fn verify_finds_a_changed_soul() {
    let (run, out) = souled_run(&fresh("verify-soul"), "circuits");
    let dir = run.to_str().expect("utf-8 path");

    assert!(out.status.success(), "{}", stderr(&out));

    let verified = breed(&["--out", dir, "--verify"]);

    assert!(verified.status.success(), "{}", stderr(&verified));

    // One changed phase in a current Soul of the checkpoint: of a Worker the `souls` events
    // moved, and of one they never moved, whose current Soul is its birth Soul.
    let path = run.join("snapshot.json");
    let original = std::fs::read_to_string(&path).expect("snapshot");
    let checkpoint: serde_json::Value = serde_json::from_str(&original).expect("json");
    let moved = &moved_workers(&logged_events(&run));
    let archive = checkpoint["snapshot"]["archive"]
        .as_array()
        .expect("archive");
    let workers: Vec<(usize, u64, bool)> = archive
        .iter()
        .enumerate()
        .flat_map(|(i, m)| {
            let id = m["id"].as_str().expect("id").to_string();
            let n = m["soul_now"].as_array().expect("current Souls").len() as u64;

            (0..n).map(move |k| (i, k, moved.contains(&(id.clone(), k))))
        })
        .collect();
    let picks: Vec<(usize, u64)> = [true, false]
        .iter()
        .filter_map(|&was_moved| {
            workers
                .iter()
                .find(|w| w.2 == was_moved)
                .map(|&(i, k, _)| (i, k))
        })
        .collect();

    assert_eq!(picks.len(), 2, "a moved and an unmoved Worker");

    for (i, k) in picks {
        let mut tampered = checkpoint.clone();
        let phase = &mut tampered["snapshot"]["archive"][i]["soul_now"][k as usize][0];
        let changed = (phase.as_u64().expect("a phase") + 1) % 65536;

        *phase = serde_json::json!(changed);
        std::fs::write(&path, serde_json::to_string(&tampered).expect("json")).expect("write");

        let refused = breed(&["--out", dir, "--verify"]);

        assert_eq!(refused.status.code(), Some(1), "{}", stderr(&refused));
        assert!(
            stderr(&refused).contains("integrity:") && stderr(&refused).contains("current Soul"),
            "{}",
            stderr(&refused)
        );
    }

    std::fs::write(&path, original).expect("restore");

    assert!(breed(&["--out", dir, "--verify"]).status.success());
}

#[test]
fn the_backend_contract_is_v3() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs/STAGE2_BACKEND_CONTRACT.md");
    let contract = std::fs::read_to_string(path).expect("contract");
    let flat = contract.split_whitespace().collect::<Vec<_>>().join(" ");

    assert!(flat.contains("Version: contract-v3,"), "{flat}");

    for needle in [
        "phases: Option<Vec<u16>>",
        "soul_decided: bool",
        "fn reads_phases(&self) -> bool { false }",
        "`GenRequest::phases`",
        "serde default `None` and is left out of the JSON when `None`",
        "never in a peer judgment",
        "only for a backend whose `reads_phases()` is true",
        "`GenResponse::soul_decided`",
        "serde default `false` and is left out of the JSON when false",
        "`ModelBackend::reads_phases()` is false by default",
    ] {
        assert!(flat.contains(needle), "{needle}");
    }
}

/// The 60-generation Run of pilot seed 101 in arm A of step 1, run as the pilot ran it. A
/// release build takes a few seconds: `cargo test --release -- --ignored`.
#[test]
#[ignore = "release test: cargo test --release -- --ignored"]
fn souls_off_reproduces_the_step1_pilot() {
    let dir = fresh("pilot-A-101");
    let out = Command::new(BREED)
        .args([
            "--out",
            dir.to_str().expect("utf-8 path"),
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
            "--seed",
            "101",
            "--max-calls",
            "200000",
        ])
        .output()
        .expect("run brut-breed");

    assert!(
        out.status.success(),
        "exit {:?}\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );

    let last = last_log_line(&dir);

    assert_eq!(last["seq"], 1361, "1362 events");
    assert_eq!(last["hash"], PILOT_A_101_HEAD);
}

/// The head hash of the README's Run on pilot seed 901 in arm S (`README.md`, section "The
/// Soul (step 2)"), recorded before the setting `variation` existed.
const README_S_901_HEAD: &str = "e38339538f19d85fccd1d1a7a1c3a0435e1989eec7838e6b6e0c5a46e0e692a1";

/// The README's Run of arm S on pilot seed 901, as the README gives it. With `variation` off
/// it writes the same bytes as before arm W. A release build takes a few seconds: `cargo
/// test --release -- --ignored`.
#[test]
#[ignore = "release test: cargo test --release -- --ignored"]
fn the_readme_run_of_arm_s_is_unchanged() {
    let dir = fresh("readme-S-901");

    ok(&breed(&[
        "--out",
        path(&dir),
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
        "--seed",
        "901",
        "--max-calls",
        "200000",
        "--soul",
        "s",
    ]));

    let verified = breed(&["--out", path(&dir), "--verify"]);

    ok(&verified);
    assert!(
        stdout(&verified).contains(&format!(
            "2046 events, hash chain intact, head {README_S_901_HEAD}"
        )),
        "{}",
        stdout(&verified)
    );
}

/// The settings of arm S with `variation` set to `variation` (arm W: prefer; arm WF:
/// fixed), as a config file in `dir`.
fn varied_config_file(dir: &Path, variation: &str) -> PathBuf {
    let mut cfg = serde_json::to_value(souled_config()).expect("json");
    let config = dir.join(format!("{variation}.json"));

    cfg["soul"]["variation"] = serde_json::json!(variation);
    std::fs::create_dir_all(dir).expect("mkdir");
    std::fs::write(&config, serde_json::to_string_pretty(&cfg).expect("json"))
        .expect("write config");

    config
}

#[test]
fn a_run_whose_soul_steers_variation_reproduces() {
    // 60 generations of arm W on circuits with the pilot's panels: --verify passes, and the
    // five record files are byte-equal at 1 and at 4 threads.
    let at = |threads: &str| {
        let dir = fresh(&format!("variation-threads-{threads}"));
        let config = varied_config_file(&dir, "prefer");
        let run = dir.join("run");
        let out = Command::new(BREED)
            .args(["--out", path(&run), "--config", path(&config)])
            .args(["--domain", "circuits", "--search", "33", "--select", "22"])
            .args(["--children", "8", "--generations", "60", "--seed", "41"])
            .args(["--max-calls", "13000000"])
            .env("RAYON_NUM_THREADS", threads)
            .output()
            .expect("run brut-breed");

        ok(&out);
        ok(&breed(&["--out", path(&run), "--verify"]));

        (
            RECORD.map(|name| {
                std::fs::read(run.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
            }),
            logged_events(&run),
            config_json(&run),
        )
    };
    let (four, events, config) = at("4");
    let states: Vec<&serde_json::Value> = events.iter().filter(|e| e["kind"] == "state").collect();
    let count = |field: &str| -> u64 {
        states
            .iter()
            .map(|e| {
                e["soul"][field]
                    .as_u64()
                    .unwrap_or_else(|| panic!("{field}"))
            })
            .sum()
    };

    // The setting is recorded, every state line counts the choices, and the weights
    // changed some.
    assert_eq!(config["soul"]["variation"], "prefer");
    assert_eq!(states.len(), 61);
    assert!(count("variation_changed") > 0);
    assert!(count("variation_changed") <= count("variation_choices"));
    assert_eq!(four, at("1").0, "1 and 4 threads");
}

#[test]
fn variation_needs_circuits() {
    // The grid gene pool names no families of a text, so a grid Run whose Soul steers
    // variation is refused before it writes a record. As with every other refusal of a
    // fresh Run's settings, only the empty run directory and its LOCK file are left; a
    // later Run can use that directory.
    for variation in ["prefer", "fixed"] {
        let dir = fresh(&format!("variation-grid-{variation}"));
        let config = varied_config_file(&dir, variation);
        let run = dir.join("run");
        let refused = breed(&[
            "--out",
            path(&run),
            "--config",
            path(&config),
            "--domain",
            "grid",
        ]);

        assert_eq!(refused.status.code(), Some(1), "{}", stderr(&refused));
        assert!(
            stderr(&refused).contains("variation needs a gene pool"),
            "{}",
            stderr(&refused)
        );
        assert!(!run.join("config.json").exists());

        let left: Vec<String> = std::fs::read_dir(&run)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();

        assert!(left.iter().all(|name| name == "LOCK"), "{left:?}");
    }

    // Arm WF on circuits starts and records its setting.
    let dir = fresh("variation-fixed");
    let config = varied_config_file(&dir, "fixed");
    let run = dir.join("run");

    std::fs::create_dir_all(&run).expect("mkdir");
    std::fs::write(run.join("STOP"), "").expect("stop");
    ok(&breed(&[
        "--out",
        path(&run),
        "--config",
        path(&config),
        "--domain",
        "circuits",
    ]));
    assert_eq!(config_json(&run)["soul"]["variation"], "fixed");
}
