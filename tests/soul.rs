//! Step 2 tests (Soul and Coupling strength, `.scratch/seele-und-kappa/`). With the default
//! settings there is no Soul, and nothing a Run writes may change (spec section 14).

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use brutmaschine::cascade::arena::{BreedConfig, worker_request_with};
use brutmaschine::cascade::circuit::CircuitDomain;
use brutmaschine::cascade::circuit_genes::circuit_pool;
use brutmaschine::cascade::domain::{CircuitConfig, Domain, GenePool, GridDomain};
use brutmaschine::cascade::genome::Cascade;
use brutmaschine::cascade::qubit::{Circuit, Gate};
use brutmaschine::cascade::soul::SoulConfig;
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

#[test]
fn promote_refuses_a_run_with_souls() {
    let (run, out) = souled_run(&fresh("promote-souled"), "grid");

    assert!(out.status.success(), "{}", stderr(&out));
    assert!(run.join("report.json").exists(), "a finished Run");

    let refused = breed(&["--out", run.to_str().expect("utf-8 path"), "--promote"]);

    assert_eq!(refused.status.code(), Some(1), "{}", stderr(&refused));
    assert!(stderr(&refused).contains("Souls"), "{}", stderr(&refused));
    assert!(!run.join("promote").exists());
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
