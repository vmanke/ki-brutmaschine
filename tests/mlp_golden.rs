//! Golden tests of the MLP mode: the regression net of the CNN change
//! (`.scratch/cnn-basismodelle/`, ticket 02). The fixtures under `tests/fixtures/mlp_golden/`
//! were recorded on the code before that change. An MLP run must stay byte-identical, so
//! these values must not change:
//!
//! 1. A tiny evolution run with a fixed seed and a fixed hidden seed: the SHA-256 of every
//!    file in its run directory, after the wall-clock fields are set to 0 (`mask_file`
//!    lists them), and its summary. Two equal runs and a run with another thread count must
//!    give the same values, which shows that the mask is complete.
//! 2. The SHA-256 of `task::generate` for every task and every split, at two seeds.
//! 3. The exact JSON text of a genome, a trained model file, every task spec, the default
//!    config, the tiny config, a `TrainStats` and an eval request as `brut-eval` reads it.
//!    Each fixture also parses back to an equal value.
//!
//! The fixtures were recorded on x86-64 Linux with glibc. Data, training and scoring call
//! `sin`, `cos`, `exp`, `ln`, `powf` and `log10` of the platform's math library. Another
//! library may round one of them differently in the last bit; then the values differ
//! although the code did not change.
//!
//! To regenerate, run `MLP_GOLDEN_OUT=<dir> cargo test --test mlp_golden`. Every test then
//! writes its actual values into `<dir>`, one file per fixture name, and still compares them
//! with the fixtures. Copy them into `tests/fixtures/mlp_golden/` only when a change is meant
//! to alter MLP behaviour, and name that change in the pull request. Any other change, the
//! CNN change included, must pass with the fixtures as they are. Such a change may need to
//! adapt the code of this file to a new API, but never the fixtures.
#![forbid(unsafe_code)]

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

use brutmaschine::config::{Config, Limits};
use brutmaschine::evolve::{self, ARCHIVE_FILE, CONFIG_FILE, EvalSetup, RunSummary};
use brutmaschine::genome::{Activation, Arch, Genome, Optimizer};
use brutmaschine::hidden_eval::{AnyEvalRequest, EvalRequest, HIDDEN_SEED_KEY};
use brutmaschine::model::Model;
use brutmaschine::nn::Mlp;
use brutmaschine::task::{self, Dataset, Split, Targets, TaskKind, TaskSpec};
use brutmaschine::train::{self, TrainStats};

const EVAL_BIN: &str = env!("CARGO_BIN_EXE_brut-eval");
const BRUT_BIN: &str = env!("CARGO_BIN_EXE_brut");

/// Directory that receives the actual values when set (see the module doc).
const OUT_ENV: &str = "MLP_GOLDEN_OUT";

/// Head anchor of the archive, next to `archive.jsonl`.
const HEAD_FILE: &str = "archive.head";

/// `prev_hash` of the first record. Masked hashes are set to this value too.
const ZERO_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// Hidden seed of the golden run. `create_secret` draws a random one, which would change
/// the hidden scores in the archive from run to run.
const HIDDEN_SEED: u64 = 4_242;

/// Every task that exists before the CNN change.
const TASKS: [TaskKind; 4] = [
    TaskKind::TwoSpirals,
    TaskKind::Checkerboard,
    TaskKind::Parity,
    TaskKind::SineRegression,
];

const SPLITS: [Split; 3] = [Split::Train, Split::Valid, Split::Hidden];

/// Seeds of the dataset hashes: a small one and one above 2^53.
const DATA_SEEDS: [u64; 2] = [1, 0x9e37_79b9_7f4a_7c15];

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/mlp_golden")
        .join(name)
}

fn read_text(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Compare `actual` with the fixture `name`, byte for byte (see `golden_all`).
fn golden(name: &str, actual: &str) {
    golden_all(&[(name, actual)]);
}

/// Compare each `(name, actual)` with its fixture, byte for byte.
///
/// With `MLP_GOLDEN_OUT` set, all actual values are first written into that directory, so
/// a difference in the first one does not keep the others from being written. On a
/// difference, the whole actual text goes to stderr and the panic names the first line
/// that differs.
fn golden_all(values: &[(&str, &str)]) {
    if let Some(dir) = std::env::var_os(OUT_ENV) {
        let dir = PathBuf::from(dir);

        std::fs::create_dir_all(&dir).expect("create the output directory");

        for (name, actual) in values {
            std::fs::write(dir.join(name), actual).expect("write the actual value");
        }
    }

    for &(name, actual) in values {
        let expected = std::fs::read_to_string(fixture(name)).unwrap_or_default();

        if expected == actual {
            continue;
        }

        eprintln!("---- actual {name} ----\n{actual}\n---- end {name} ----");

        let first = expected
            .lines()
            .zip(actual.lines())
            .position(|(e, a)| e != a)
            .unwrap_or_else(|| expected.lines().count().min(actual.lines().count()));

        panic!(
            "{name} differs from its fixture at line {}: expected {:?}, got {:?} \
             ({} vs {} bytes)",
            first + 1,
            expected.lines().nth(first),
            actual.lines().nth(first),
            expected.len(),
            actual.len()
        );
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes).as_slice())
}

/// A fresh, empty directory for one test.
fn fresh_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("mlp_golden")
        .join(name);

    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test dir");

    dir
}

/// `tiny_config` of `tests/e2e.rs` with seed 1, but `threads` 0.
///
/// With 0 the pool takes its size from `RAYON_NUM_THREADS`, else from the core count. So a
/// run with another thread count writes the same `config.json` and `RunStarted` record.
fn tiny_config() -> Config {
    Config {
        task: TaskKind::TwoSpirals,
        master_seed: 1,
        population_init: 4,
        generations: 2,
        children_per_generation: 3,
        crossover_rate: 0.3,
        limits: Limits {
            max_hidden_layers: 2,
            max_width: 12,
            max_epochs: 20,
            ..Limits::default()
        },
        hidden_eval_top_k: 2,
        threads: 0,
        ..Config::default()
    }
}

/// Where `brut init` puts the secret file.
fn secret_path(dir: &Path) -> PathBuf {
    dir.join("secret").join("hidden.env")
}

/// A fresh run directory with `config.json` and a secret file that holds `HIDDEN_SEED`.
fn prepare_run(name: &str, config: &Config) -> PathBuf {
    let dir = fresh_dir(name);
    let secret = secret_path(&dir);

    config.save(&dir.join(CONFIG_FILE)).expect("save config");
    std::fs::create_dir_all(secret.parent().expect("secret dir")).expect("create secret dir");

    // Mode 0600 as `create_secret` makes it: brut-eval refuses a secret that others can read.
    let mut options = std::fs::OpenOptions::new();

    options.write(true).create_new(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        options.mode(0o600);
    }

    let mut file = options.open(&secret).expect("create secret");

    writeln!(file, "{HIDDEN_SEED_KEY}={HIDDEN_SEED}").expect("write secret");

    dir
}

/// The summary as `brut run` prints it: pretty JSON and a newline.
fn summary_text(summary: &RunSummary) -> String {
    let json = serde_json::to_string_pretty(summary).expect("summary json");

    format!("{json}\n")
}

/// Run the evolution loop in this process, with the hidden evaluation.
fn run_in_process(dir: &Path) -> String {
    let eval = EvalSetup {
        eval_bin: PathBuf::from(EVAL_BIN),
        secret_env: secret_path(dir),
    };
    let summary = evolve::run(dir, Some(&eval)).expect("run");

    summary_text(&summary)
}

/// Run `brut run <dir>` with `threads` rayon threads and return what it prints.
fn run_cli(dir: &Path, threads: usize) -> String {
    let out = Command::new(BRUT_BIN)
        .arg("run")
        .arg(dir)
        .env("RAYON_NUM_THREADS", threads.to_string())
        .output()
        .expect("start brut");

    assert!(
        out.status.success(),
        "brut run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    String::from_utf8(out.stdout).expect("utf-8 summary")
}

/// A thread count that differs from the pool size of the runs in this process.
///
/// Those runs use rayon's default size (`threads` 0), the size of the global pool.
fn other_thread_count() -> usize {
    if rayon::current_num_threads() == 1 {
        2
    } else {
        1
    }
}

/// Replace the value of every `"key":` in `text` with `zero`.
///
/// A value is a JSON number or a string without escapes. Spaces after the colon (pretty
/// JSON) are kept.
fn mask(text: &str, key: &str, zero: &str) -> String {
    let pattern = format!("\"{key}\":");
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(at) = rest.find(&pattern) {
        let after = &rest[at + pattern.len()..];
        let value = after.trim_start_matches(' ');

        let len = match value.strip_prefix('"') {
            Some(inner) => inner.find('"').expect("closing quote") + 2,
            None => value
                .find(|c: char| !(c.is_ascii_digit() || "+-.eE".contains(c)))
                .unwrap_or(value.len()),
        };

        out.push_str(&rest[..rest.len() - value.len()]);
        out.push_str(zero);
        rest = &value[len..];
    }

    out.push_str(rest);

    out
}

/// The wall-clock fields of a run's files, set to 0 before hashing (hashes to 64 zeros):
///
/// - `archive.jsonl`, every record: `unix_ms`, the time the record was appended.
/// - `archive.jsonl`, every `Candidate` record: `wall_seconds` in `train`, the training time.
/// - `archive.jsonl`, every record: `prev_hash` and `hash`. They are SHA-256 values over
///   record text that holds `unix_ms`. `check_chain` checks them before they are masked.
/// - `archive.head`: `last_hash`, the `hash` of the last record.
///
/// The other files hold no clock value: `config.json`, the model files,
/// `secret/hidden.env` (written by `prepare_run`), `secret/eval_state.json` and the empty
/// `secret/eval.lock`. The run summary, which is no file, holds one: `wall_seconds`.
fn mask_file(rel: &str, bytes: Vec<u8>) -> Vec<u8> {
    let zero_hash = format!("\"{ZERO_HASH}\"");

    match rel {
        ARCHIVE_FILE => {
            let text = String::from_utf8(bytes).expect("utf-8 archive");
            let text = mask(&text, "unix_ms", "0");
            let text = mask(&text, "wall_seconds", "0.0");
            let text = mask(&text, "prev_hash", &zero_hash);

            mask(&text, "hash", &zero_hash).into_bytes()
        }
        HEAD_FILE => {
            let text = String::from_utf8(bytes).expect("utf-8 head");

            mask(&text, "last_hash", &zero_hash).into_bytes()
        }
        _ => bytes,
    }
}

/// Check the hash chain on the real text: each `hash` is the SHA-256 of its line with
/// `"hash":""`, each `prev_hash` is the `hash` of the line before (`ZERO_HASH` for the
/// first line), and the head names the record count and the last hash.
fn check_chain(archive: &str, head: &str) {
    let mut prev = ZERO_HASH.to_string();
    let mut records = 0;

    for line in archive.lines() {
        let at = line.rfind(",\"hash\":\"").expect("hash field");
        let hash = line[at..]
            .strip_prefix(",\"hash\":\"")
            .and_then(|h| h.strip_suffix("\"}"))
            .expect("hash is the last field");
        let unhashed = format!("{},\"hash\":\"\"}}", &line[..at]);

        assert_eq!(
            sha256_hex(unhashed.as_bytes()),
            hash,
            "hash of record {records}"
        );
        assert!(
            line.contains(&format!(",\"prev_hash\":\"{prev}\",")),
            "prev_hash of record {records}"
        );

        prev = hash.to_string();
        records += 1;
    }

    assert_eq!(
        head,
        format!("{{\"records\":{records},\"last_hash\":\"{prev}\"}}\n")
    );
}

/// Every file below `dir`, as paths relative to `root` with `/` between the parts.
fn list_files(root: &Path, dir: &Path, out: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).expect("read dir") {
        let path = entry.expect("dir entry").path();

        if path.is_dir() {
            list_files(root, &path, out);

            continue;
        }

        let rel = path.strip_prefix(root).expect("inside the run directory");
        let parts: Vec<String> = rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();

        out.push(parts.join("/"));
    }
}

/// One `<sha256>  <path>` line per file of the run directory, sorted by path, after the
/// chain check and the mask.
fn digest_run(dir: &Path) -> String {
    let mut files = Vec::new();

    list_files(dir, dir, &mut files);
    files.sort();

    check_chain(
        &read_text(&dir.join(ARCHIVE_FILE)),
        &read_text(&dir.join(HEAD_FILE)),
    );

    let mut out = String::new();

    for rel in &files {
        let bytes = std::fs::read(dir.join(rel)).expect("read run file");

        writeln!(out, "{}  {rel}", sha256_hex(&mask_file(rel, bytes))).expect("write line");
    }

    out
}

#[test]
fn a_tiny_run_writes_the_same_files() {
    let config = tiny_config();
    let first = prepare_run("first", &config);
    let second = prepare_run("second", &config);
    let other = prepare_run("other-threads", &config);

    let first_summary = mask(&run_in_process(&first), "wall_seconds", "0.0");
    let second_summary = mask(&run_in_process(&second), "wall_seconds", "0.0");
    let other_summary = mask(
        &run_cli(&other, other_thread_count()),
        "wall_seconds",
        "0.0",
    );

    let files = digest_run(&first);

    // Equal runs agree after the mask, so no other field carries a clock value.
    assert_eq!(digest_run(&second), files, "two equal runs differ");
    assert_eq!(
        digest_run(&other),
        files,
        "another thread count changed files"
    );
    assert_eq!(second_summary, first_summary, "two equal runs differ");
    assert_eq!(
        other_summary, first_summary,
        "another thread count changed the summary"
    );

    golden_all(&[
        ("run_files.txt", files.as_str()),
        ("run_summary.json", first_summary.as_str()),
    ]);
}

/// Sizes, then the SHA-256 of the inputs and of the targets as little-endian bytes.
fn dataset_digest(data: &Dataset) -> String {
    let x: Vec<u8> = data.x.iter().flat_map(|v| v.to_le_bytes()).collect();

    let (kind, targets): (&str, Vec<u8>) = match &data.targets {
        Targets::Labels(labels) => (
            "labels",
            labels.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        Targets::Values(values) => (
            "values",
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
    };

    format!(
        "n={} n_in={} n_out={} x={} {kind}={}",
        data.n,
        data.n_in,
        data.n_out,
        sha256_hex(&x),
        sha256_hex(&targets)
    )
}

#[test]
fn generated_datasets_keep_their_bytes() {
    let mut text = String::new();

    for kind in TASKS {
        let spec = kind.spec();

        for split in SPLITS {
            for seed in DATA_SEEDS {
                let data = task::generate(&spec, split, seed);

                writeln!(
                    text,
                    "{kind:?} {split:?} seed={seed} {}",
                    dataset_digest(&data)
                )
                .expect("write line");
            }
        }
    }

    golden("datasets.txt", &text);
}

/// A hand-made genome. Its floats are not exact in binary and its seed lies above 2^53,
/// so its JSON text pins the float and the integer format.
fn golden_genome() -> Genome {
    Genome {
        arch: Arch::Mlp,
        hidden: vec![6, 5],
        activation: Activation::Tanh,
        optimizer: Optimizer::Adam,
        learning_rate: 0.0123,
        l2: 3.0e-5,
        batch_size: 32,
        epochs: 8,
        init_scale: 1.1,
        seed: 0xdead_beef_0123_4567,
    }
}

/// `golden_genome` trained on the Train split of `TwoSpirals` (data seed 1).
fn trained() -> (Model, TrainStats) {
    let spec = TaskKind::TwoSpirals.spec();
    let data = task::generate(&spec, Split::Train, 1);

    train::train(&golden_genome(), &spec, &data, &Limits::default()).expect("train")
}

#[test]
fn a_genome_keeps_its_json() {
    let genome = golden_genome();
    let text = serde_json::to_string(&genome).expect("json");

    golden("genome.json", &text);

    let parsed: Genome = serde_json::from_str(&read_text(&fixture("genome.json"))).expect("parse");

    assert_eq!(parsed, genome);

    // The id, which names the model files, is the start of the SHA-256 of this text.
    assert_eq!(genome.id(), sha256_hex(text.as_bytes())[..16]);
}

#[test]
fn a_trained_model_file_keeps_its_json() {
    let (model, _) = trained();

    // The bytes `evolve` writes to `models/<id>.json`.
    let bytes = serde_json::to_vec(&model).expect("json");
    let text = String::from_utf8(bytes).expect("utf-8 model");

    golden("model.json", &text);

    let parsed: Mlp = serde_json::from_str(&read_text(&fixture("model.json"))).expect("parse");

    assert_eq!(Model::Mlp(parsed), model);
}

#[test]
fn train_stats_keep_their_json() {
    let (_, mut stats) = trained();

    // The training time, the one field that differs from run to run.
    stats.wall_seconds = 0.0;

    let text = serde_json::to_string(&stats).expect("json");

    golden("train_stats.json", &text);

    let parsed: TrainStats =
        serde_json::from_str(&read_text(&fixture("train_stats.json"))).expect("parse");

    assert_eq!(parsed, stats);
}

#[test]
fn task_specs_keep_their_json() {
    let specs: Vec<TaskSpec> = TASKS.iter().map(|k| k.spec()).collect();
    let mut text = String::new();

    for spec in &specs {
        writeln!(text, "{}", serde_json::to_string(spec).expect("json")).expect("write line");
    }

    golden("task_specs.jsonl", &text);

    let parsed: Vec<TaskSpec> = read_text(&fixture("task_specs.jsonl"))
        .lines()
        .map(|line| serde_json::from_str(line).expect("parse"))
        .collect();

    assert_eq!(parsed, specs);
}

/// Write `config` with `Config::save`, which writes `config.json`. Compare the bytes with
/// the fixture `name`, then load the fixture back.
fn check_config(name: &str, config: &Config) {
    let path = fresh_dir(name.trim_end_matches(".json")).join(CONFIG_FILE);

    config.save(&path).expect("save config");
    golden(name, &read_text(&path));

    assert_eq!(&Config::load(&fixture(name)).expect("load"), config);
}

#[test]
fn the_default_config_keeps_its_json() {
    check_config("config_default.json", &Config::default());
}

#[test]
fn the_tiny_config_keeps_its_json() {
    check_config("config_tiny.json", &tiny_config());
}

#[test]
fn an_eval_request_keeps_its_json() {
    let id = golden_genome().id();
    let req = EvalRequest {
        candidate_id: id.clone(),
        model_path: PathBuf::from(format!("/runs/golden/models/{id}.json")),
        task: TaskKind::TwoSpirals,
    };

    // The bytes the orchestrator writes to the stdin of brut-eval (`request_hidden_eval`).
    let bytes = serde_json::to_vec(&req).expect("json");
    let text = String::from_utf8(bytes).expect("utf-8 request");

    golden("eval_request.json", &text);

    // brut-eval reads its stdin with `AnyEvalRequest::from_json`.
    let parsed = AnyEvalRequest::from_json(&read_text(&fixture("eval_request.json")));

    assert_eq!(parsed.expect("parse"), AnyEvalRequest::Mlp(req.clone()));
    assert_eq!(AnyEvalRequest::Mlp(req).to_json().expect("json"), text);
}
