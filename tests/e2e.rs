//! End-to-end tests: tiny real runs with the hidden evaluator, resume, kill switch, tamper detection.

use std::collections::HashSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use brutmaschine::archive::{Archive, CandidateStatus, RecordKind};
use brutmaschine::cnn::families::Family;
use brutmaschine::cnn::genome::{
    Block, CnnActivation, CnnGenome, Dims, Downsample, Head, Norm, Stage, Stem,
};
use brutmaschine::config::{Config, Limits, ModelKinds, ScoreWeights};
use brutmaschine::evolve::{
    self, ARCHIVE_FILE, BASE_GENERATOR, BASE_MODEL_GENERATOR, CONFIG_FILE, EvalSetup,
    INIT_GENERATOR, MODELS_DIR, PROMOTED_GENERATOR, RunSummary,
};
use brutmaschine::genome::{Activation, Arch, Genome, Optimizer};
use brutmaschine::hidden_eval::{HIDDEN_SEED_KEY, create_secret};
use brutmaschine::hitl::{
    BaseModelScore, CandidateAdapter, LineageStep, MAX_CANDIDATE_FILE_BYTES, MODEL_DIR,
    ModelAdapter, ModelProvenance, PromotedModel, model_seed_ids,
};
use brutmaschine::model::Model;
use brutmaschine::nn::Mlp;
use brutmaschine::promote::{self, PROMOTE_DIR, PromotionManifest};
use brutmaschine::safety;
use brutmaschine::score;
use brutmaschine::task::{self, Split, TaskKind};
use brutmaschine::util::{BrutError, derive_seed};
use brutmaschine::variation::CNN_GENERATOR;

const EVAL_BIN: &str = env!("CARGO_BIN_EXE_brut-eval");
const BRUT_BIN: &str = env!("CARGO_BIN_EXE_brut");
const INGEST_BIN: &str = env!("CARGO_BIN_EXE_brut-ingest");

/// A fresh, empty directory for one test.
fn fresh_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("e2e")
        .join(name);

    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test dir");

    dir
}

/// Small config: 2 generations, small population, short training.
fn tiny_config(seed: u64) -> Config {
    Config {
        task: TaskKind::TwoSpirals,
        master_seed: seed,
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
        threads: 2,
        ..Config::default()
    }
}

/// Write config and secret into `dir`.
fn setup(dir: &Path, config: &Config) -> EvalSetup {
    config.save(&dir.join(CONFIG_FILE)).expect("save config");

    let secret_env = dir.join("secret").join("hidden.env");

    create_secret(&secret_env).expect("create secret");

    EvalSetup {
        eval_bin: PathBuf::from(EVAL_BIN),
        secret_env,
    }
}

fn fitness_list(dir: &Path) -> Vec<(String, f64)> {
    let archive = Archive::open(&dir.join(ARCHIVE_FILE)).expect("open archive");

    archive
        .candidates()
        .into_iter()
        .map(|c| (c.id, c.fitness))
        .collect()
}

fn run_brut(args: &[&str]) -> std::process::Output {
    Command::new(BRUT_BIN)
        .args(args)
        .output()
        .expect("run brut")
}

#[test]
fn tiny_run_with_hidden_eval() {
    let dir = fresh_dir("tiny");
    let config = tiny_config(1);
    let eval = setup(&dir, &config);
    let summary: RunSummary = evolve::run(&dir, Some(&eval)).expect("run");

    // Summary is sane.
    assert_eq!(summary.generations_done, 2);
    assert_eq!(summary.candidates, 4 + 2 * 3);
    assert_eq!(summary.stop_reason, "completed");
    assert!(summary.best_id.is_some());
    assert!(summary.best_fitness.is_finite() && summary.best_fitness > 0.0);
    assert!((0.0..=1.0).contains(&summary.best_valid_primary));
    assert!(summary.flops_used > 0);
    assert!(summary.wall_seconds >= 0.0);

    // Hidden results are present, rounded and consistent with the archive.
    assert!(!summary.hidden.is_empty() && summary.hidden.len() <= 2);

    for h in &summary.hidden {
        assert!((0.0..=1.0).contains(&h.hidden_primary));
        assert!(((h.hidden_primary * 100.0).round() - h.hidden_primary * 100.0).abs() < 1e-6);
    }

    // Archive verifies; records are complete.
    let n = Archive::verify(&dir.join(ARCHIVE_FILE)).expect("verify");
    let archive = Archive::open(&dir.join(ARCHIVE_FILE)).expect("open");

    assert_eq!(n, archive.records().len());
    assert!(matches!(
        archive.records()[0].kind,
        RecordKind::RunStarted { .. }
    ));
    assert!(matches!(
        archive.records().last().map(|r| &r.kind),
        Some(RecordKind::RunFinished { .. })
    ));

    // Every Ok candidate has a model file; others do not.
    for c in archive.candidates() {
        match c.status {
            CandidateStatus::Ok | CandidateStatus::Quarantined => {
                let file = c.model_file.expect("model file");

                assert!(dir.join(file).is_file());
            }
            _ => assert!(c.model_file.is_none()),
        }
    }

    // Gen 1+ children name their parents, which exist in the archive.
    for c in archive
        .candidates()
        .into_iter()
        .filter(|c| c.generation > 0)
    {
        for p in &c.parents {
            assert!(archive.get(p).is_some());
        }
    }

    // The evaluator counted its queries next to the secret.
    let state = std::fs::read_to_string(dir.join("secret").join("eval_state.json")).expect("state");

    assert!(state.contains("queries_used"));
}

/// An MLP run works on every image and sequence task. The MLP reads each sample flattened,
/// and the hidden evaluator scores it on the hidden split of that task.
#[test]
fn tiny_runs_on_the_image_and_sequence_tasks() {
    let tasks = [
        TaskKind::Shapes,
        TaskKind::BarsAndStripes,
        TaskKind::BlobCount,
        TaskKind::MirrorSymmetry,
        TaskKind::Motif1d,
    ];

    for task in tasks {
        let dir = fresh_dir(&format!("grid-{}", task.name()));
        let config = Config {
            task,
            population_init: 2,
            generations: 1,
            children_per_generation: 2,
            ..tiny_config(3)
        };
        let eval = setup(&dir, &config);
        let summary = evolve::run(&dir, Some(&eval)).expect("run");

        assert_eq!(summary.stop_reason, "completed", "{task:?}");
        assert_eq!(summary.candidates, 2 + 2, "{task:?}");
        assert!(summary.best_fitness.is_finite(), "{task:?}");
        assert!(!summary.hidden.is_empty(), "{task:?}");

        for h in &summary.hidden {
            assert!((0.0..=1.0).contains(&h.hidden_primary), "{task:?}");
        }

        let n = Archive::verify(&dir.join(ARCHIVE_FILE)).expect("verify");
        let archive = Archive::open(&dir.join(ARCHIVE_FILE)).expect("open");
        let best = summary.best_id.expect("best id");
        let file = archive
            .get(&best)
            .and_then(|c| c.model_file)
            .expect("model file");
        let model: Mlp = serde_json::from_slice(&std::fs::read(dir.join(file)).expect("read"))
            .expect("model json");
        let spec = task.spec();

        assert_eq!(n, archive.records().len());
        assert_eq!(model.sizes.first(), Some(&spec.n_in), "{task:?}");
        assert_eq!(model.sizes.last(), Some(&spec.n_out), "{task:?}");

        // `brut init` takes the task's CLI name.
        let run_dir = dir.join("cli");
        let run = run_dir.to_str().expect("utf8 path");

        assert_eq!(
            run_brut(&["init", run, "--task", task.name()])
                .status
                .code(),
            Some(0)
        );
        assert_eq!(
            Config::load(&run_dir.join(CONFIG_FILE)).expect("load").task,
            task
        );
    }

    // The help names every task.
    let help = run_brut(&["help"]);
    let usage = String::from_utf8_lossy(&help.stdout);

    for task in TaskKind::ALL {
        assert!(usage.contains(task.name()), "help lacks {}", task.name());
    }
}

/// A small CNN for a 1 x 16 x 16 image: a 3x3 convolution of 4 channels with batch norm,
/// ReLU and 2x2 max pooling, a 3x3 convolution of 8 channels with ReLU and 2x2 average
/// pooling, then flatten and the output layer. 598 parameters for 2 classes.
fn small_cnn() -> Genome {
    let stage = Stage {
        block: Block::Plain,
        width: 8,
        kernel: 3,
        stride: 2,
        down: Downsample::AvgPool,
        dilation: 1,
        groups: 1,
        se: false,
        residual: false,
        norm: Norm::None,
        activation: CnnActivation::Relu,
    };

    Genome {
        arch: Arch::Cnn(CnnGenome {
            dims: Dims::TwoD,
            stem: Stem {
                width: 4,
                kernel: 3,
                stride: 2,
                down: Downsample::MaxPool,
                norm: Norm::Batch,
                activation: CnnActivation::Relu,
            },
            stages: vec![stage],
            head: Head::Flatten,
        }),
        hidden: Vec::new(),
        activation: Activation::Relu,
        optimizer: Optimizer::Adam,
        learning_rate: 0.01,
        l2: 0.0,
        batch_size: 64,
        epochs: 5,
        init_scale: 1.0,
        seed: 5,
    }
}

/// `tiny_config` on bars and stripes with `small_cnn` as a base genome: generation 0 holds
/// the CNN and 1 random MLP, generation 1 two children by mutation. All 4 candidates go to
/// the hidden evaluation. With master seed 16 both children have the CNN as their parent
/// (on the machine that wrote this test; seeds 13 to 19 all gave at least one).
fn cnn_config(threads: usize) -> Config {
    Config {
        task: TaskKind::BarsAndStripes,
        population_init: 1,
        generations: 1,
        children_per_generation: 2,
        crossover_rate: 0.0,
        hidden_eval_top_k: 4,
        threads,
        base_genomes: vec![small_cnn()],
        ..tiny_config(16)
    }
}

/// A CNN through the whole run: the base genome trains, its model file is a CNN file that
/// gives back its validation score, and `brut-eval` loads and scores it on the hidden split.
#[test]
fn a_cnn_base_genome_trains_is_saved_and_scored_in_a_run() {
    let dir = fresh_dir("cnn-run");
    let config = cnn_config(2);
    let eval = setup(&dir, &config);
    let summary = evolve::run(&dir, Some(&eval)).expect("run");
    let spec = config.task.spec();
    let genome = small_cnn();

    assert_eq!(summary.stop_reason, "completed");
    assert_eq!(summary.candidates, 1 + 1 + 2);
    assert_eq!(
        Archive::verify(&dir.join(ARCHIVE_FILE)).expect("verify"),
        Archive::open(&dir.join(ARCHIVE_FILE))
            .expect("open")
            .records()
            .len()
    );

    let archive = Archive::open(&dir.join(ARCHIVE_FILE)).expect("open");
    let base = archive
        .candidates()
        .into_iter()
        .find(|c| c.generator == BASE_GENERATOR)
        .expect("the base genome is a candidate");

    assert_eq!(base.generation, 0);
    assert_eq!(base.genome, genome);
    assert_eq!(base.id, genome.id());
    assert_eq!(base.status, CandidateStatus::Ok);
    assert_eq!(base.params, genome.param_count(&spec));
    assert_eq!(base.train.flops, genome.train_flops(&spec));

    // The model file is a CNN file, and it gives back the archived validation score.
    let bytes = std::fs::read(dir.join(base.model_file.as_deref().expect("model file")))
        .expect("read model");

    assert!(bytes.starts_with(b"{\"kind\":\"cnn\","));

    let model = Model::from_json(&bytes).expect("parse");
    let Model::Cnn(cnn) = &model else {
        panic!("the CNN was saved as another kind");
    };
    let valid = task::generate(&spec, Split::Valid, derive_seed(config.master_seed, "data"));

    assert_eq!(cnn.param_count(), base.params);
    assert_eq!(score::evaluate(&model, &spec, &valid), base.valid);
    assert!(base.valid.primary > 0.6, "{:?}", base.valid);

    // brut-eval loaded and scored it.
    let hidden = summary
        .hidden
        .iter()
        .find(|h| h.candidate_id == base.id)
        .expect("a hidden result for the CNN");

    assert!((0.0..=1.0).contains(&hidden.hidden_primary));
    assert_eq!(summary.hidden.len(), 4);

    // A child of a CNN is a CNN, and it trains and gets its CNN file as well.
    let children: Vec<_> = archive
        .candidates()
        .into_iter()
        .filter(|c| c.generation == 1 && c.parents == [base.id.clone()])
        .collect();

    assert!(!children.is_empty(), "no child of the CNN");

    for child in children {
        let file = child.model_file.as_deref().expect("model file");
        let bytes = std::fs::read(dir.join(file)).expect("read model");

        assert!(matches!(child.genome.arch, Arch::Cnn(_)), "{}", child.id);
        assert_eq!(child.status, CandidateStatus::Ok);
        assert!(bytes.starts_with(b"{\"kind\":\"cnn\","));
    }

    // The base genome is part of the stored experiment.
    let stored = std::fs::read_to_string(dir.join(CONFIG_FILE)).expect("config");

    assert!(stored.contains("\"base_genomes\""));
}

/// The id and the bytes of every model file of the run in `dir`.
fn model_files(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let archive = Archive::open(&dir.join(ARCHIVE_FILE)).expect("open");

    archive
        .candidates()
        .into_iter()
        .filter_map(|c| c.model_file.map(|f| (c.id, f)))
        .map(|(id, f)| (id, std::fs::read(dir.join(f)).expect("read")))
        .collect()
}

/// The same CNN run trains to the same candidates and model files with 1 or 2 worker
/// threads (`Config::threads`, the pool that trains candidates side by side).
///
/// This does not change the threads of candle's matrix products; see
/// `a_cnn_run_gives_the_same_files_at_1_and_3_matrix_threads`.
#[test]
fn a_cnn_run_does_not_depend_on_the_worker_threads() {
    let files = model_files;
    let a = fresh_dir("cnn-threads-a");
    let b = fresh_dir("cnn-threads-b");

    setup(&a, &cnn_config(2));
    setup(&b, &cnn_config(1));
    evolve::run(&a, None).expect("run a");
    evolve::run(&b, None).expect("run b");

    assert_eq!(fitness_list(&a), fitness_list(&b));
    assert_eq!(files(&a), files(&b));
    assert!(
        files(&a)
            .iter()
            .any(|(_, bytes)| bytes.starts_with(b"{\"kind\":\"cnn\","))
    );
}

/// A CNN with every norm and pool of ticket 02, for a 16 x 16 image: a stem with batch norm
/// and max pooling, a dilated group convolution with group norm and SiLU, and a convolution
/// with layer norm, GELU and average pooling.
fn normed_cnn() -> Genome {
    let stage = |width: usize, stride: usize, down: Downsample, norm: Norm| Stage {
        block: Block::Plain,
        width,
        kernel: 3,
        stride,
        down,
        dilation: 1,
        groups: 1,
        se: false,
        residual: false,
        norm,
        activation: CnnActivation::Gelu,
    };

    Genome {
        arch: Arch::Cnn(CnnGenome {
            dims: Dims::TwoD,
            stem: Stem {
                width: 8,
                kernel: 3,
                stride: 2,
                down: Downsample::MaxPool,
                norm: Norm::Batch,
                activation: CnnActivation::Relu,
            },
            stages: vec![
                Stage {
                    dilation: 2,
                    groups: 2,
                    activation: CnnActivation::Silu,
                    ..stage(8, 1, Downsample::Conv, Norm::Group)
                },
                stage(12, 2, Downsample::AvgPool, Norm::Layer),
            ],
            head: Head::GlobalAvgPool,
        }),
        hidden: vec![6],
        epochs: 5,
        ..small_cnn()
    }
}

/// A CNN run gives the same candidates and model files when candle splits its matrix
/// products into 1 or 3 parts.
///
/// candle reads that number from `RAYON_NUM_THREADS` when a product starts, not from the
/// rayon pool around it, and 1 means no split at all. A test cannot change the
/// environment of its own process, so each run is a child process `brut run --no-hidden`
/// with its own value. One worker thread trains the candidates in both.
#[test]
fn a_cnn_run_gives_the_same_files_at_1_and_3_matrix_threads() {
    let run = |name: &str, threads: &str| {
        let dir = fresh_dir(name);
        let config = Config {
            base_genomes: vec![small_cnn(), normed_cnn()],
            ..cnn_config(1)
        };

        config.save(&dir.join(CONFIG_FILE)).expect("save config");

        let out = Command::new(BRUT_BIN)
            .arg("run")
            .arg(&dir)
            .arg("--no-hidden")
            .env("RAYON_NUM_THREADS", threads)
            .output()
            .expect("start brut");

        assert!(
            out.status.success(),
            "brut run failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );

        dir
    };
    let one = run("cnn-matrix-1", "1");
    let three = run("cnn-matrix-3", "3");
    let cnn_files = model_files(&one)
        .iter()
        .filter(|(_, bytes)| bytes.starts_with(b"{\"kind\":\"cnn\","))
        .count();

    assert_eq!(fitness_list(&one), fitness_list(&three));
    assert_eq!(model_files(&one), model_files(&three));
    assert!(cnn_files >= 3, "{cnn_files} CNN files");
}

/// A short run with CNNs on the shapes task (ticket 05): the base models that fit
/// `max_params` of 1100 (GoogLeNet, Xception and ShuffleNet; the 13 others stay out with a
/// log line), 2 random CNNs, and 3 children. A clock of 600 s per candidate keeps a busy
/// machine from stopping a training, which would change the results.
fn short_cnn_config() -> Config {
    Config {
        task: TaskKind::Shapes,
        master_seed: 1,
        population_init: 2,
        generations: 1,
        children_per_generation: 3,
        crossover_rate: 0.5,
        limits: Limits {
            max_params: 1100,
            max_candidate_seconds: 600.0,
            max_total_seconds: 3600.0,
            ..Limits::default()
        },
        threads: 0,
        models: ModelKinds::Cnn,
        ..Config::default()
    }
}

/// Set the value of every `"key":` in `text` to `zero`. A value is a JSON number or a string
/// without escapes, as in `tests/mlp_golden.rs`.
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

/// Every file of the run in `dir`, sorted by path, as text, with the clock fields set to 0:
/// in the archive the time and the training time of each record and the hashes over them, in
/// its head the last hash. Nothing else in a run directory without a secret holds the clock.
fn run_files(dir: &Path) -> Vec<(String, String)> {
    let mut paths = vec![
        CONFIG_FILE.to_string(),
        ARCHIVE_FILE.to_string(),
        "archive.head".to_string(),
    ];
    let mut models: Vec<String> = std::fs::read_dir(dir.join(MODELS_DIR))
        .expect("models dir")
        .map(|entry| {
            let name = entry.expect("dir entry").file_name();

            format!("{MODELS_DIR}/{}", name.to_string_lossy())
        })
        .collect();

    models.sort();
    paths.extend(models);

    paths
        .into_iter()
        .map(|rel| {
            let text = std::fs::read_to_string(dir.join(&rel)).expect("read run file");
            let text = mask(&text, "unix_ms", "0");
            let text = mask(&text, "wall_seconds", "0.0");
            let text = mask(&text, "prev_hash", "\"\"");
            let text = mask(&text, "last_hash", "\"\"");

            (rel, mask(&text, "hash", "\"\""))
        })
        .collect()
}

/// A short evolution run with CNNs on the shapes task gives the same files, byte for byte
/// once the clock fields are masked, from the same seed: in two child processes `brut run
/// --no-hidden` with `RAYON_NUM_THREADS` at 1 and 2, which sets both the worker threads that
/// train candidates side by side and the threads of candle's matrix products. Both run at
/// the same time, about a minute at one thread.
#[test]
fn a_short_cnn_run_on_shapes_is_reproducible_byte_for_byte() {
    let config = short_cnn_config();
    let start = |name: &str, threads: &str| {
        let dir = fresh_dir(name);

        config.save(&dir.join(CONFIG_FILE)).expect("save config");

        let child = Command::new(BRUT_BIN)
            .arg("run")
            .arg(&dir)
            .arg("--no-hidden")
            .env("RAYON_NUM_THREADS", threads)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("start brut");

        (dir, child)
    };
    let (one, first) = start("cnn-short-1", "1");
    let (two, second) = start("cnn-short-2", "2");
    let summaries: Vec<String> = [first, second]
        .into_iter()
        .map(|child| {
            let out = child.wait_with_output().expect("brut run");

            assert!(
                out.status.success(),
                "brut run failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );

            mask(
                &String::from_utf8(out.stdout).expect("utf-8 summary"),
                "wall_seconds",
                "0.0",
            )
        })
        .collect();
    let files = run_files(&one);

    assert_eq!(run_files(&two), files);
    assert_eq!(summaries[0], summaries[1]);

    let archive = Archive::open(&one.join(ARCHIVE_FILE)).expect("open");
    let candidates = archive.candidates();
    let first_generation: Vec<&str> = candidates
        .iter()
        .filter(|c| c.generation == 0)
        .map(|c| c.generator.as_str())
        .collect();
    let left_out = archive
        .records()
        .iter()
        .filter(
            |r| matches!(&r.kind, RecordKind::SafetyEvent { what } if what.contains("left out")),
        )
        .count();

    // Generation 0: the base models that fit, and the random CNNs.
    for family in ["googlenet", "xception", "shufflenet"] {
        let name = format!("{BASE_MODEL_GENERATOR}/{family}");

        assert!(
            first_generation.contains(&name.as_str()),
            "{first_generation:?}"
        );
    }

    assert_eq!(first_generation.len(), 3 + 2, "{first_generation:?}");
    assert_eq!(
        first_generation
            .iter()
            .filter(|g| **g == INIT_GENERATOR)
            .count(),
        2
    );
    assert_eq!(left_out, 13);
    assert_eq!(files.len(), 3 + candidates.len());

    // Every candidate is a CNN that trained to its end; the children come from the CNN
    // operators. On this seed one of them exchanged the stages of the base models of two
    // families (ShuffleNet and GoogLeNet, on the machine that wrote this test).
    let children: Vec<_> = candidates.iter().filter(|c| c.generation == 1).collect();

    for c in &candidates {
        assert!(matches!(c.genome.arch, Arch::Cnn(_)), "{}", c.id);
        assert_eq!(c.status, CandidateStatus::Ok, "{}", c.id);
        assert!(!c.train.stopped_by_budget, "{}", c.id);
    }

    assert_eq!(children.len(), 3);
    assert!(
        children
            .iter()
            .all(|c| c.generator.starts_with(&format!("{CNN_GENERATOR}/"))),
        "{:?}",
        children.iter().map(|c| &c.generator).collect::<Vec<_>>()
    );
    assert!(
        children
            .iter()
            .any(|c| c.generator == format!("{CNN_GENERATOR}/crossover") && c.parents.len() == 2)
    );
}

/// The commit that the promotion tests name. A promotion checks only its form.
const TEST_COMMIT: &str = "4613e2a";

/// Write `config` and a secret with this hidden seed into `dir`. `create_secret` draws a
/// random seed, which would change the hidden scores, and so what a run promotes, from one
/// test run to the next.
fn setup_with_hidden_seed(dir: &Path, config: &Config, hidden_seed: u64) -> EvalSetup {
    config.save(&dir.join(CONFIG_FILE)).expect("save config");

    let secret_env = dir.join("secret").join("hidden.env");

    std::fs::create_dir_all(secret_env.parent().expect("secret dir")).expect("create secret dir");

    // Mode 0600 as `create_secret` makes it: brut-eval refuses a secret that others can read.
    let mut options = std::fs::OpenOptions::new();

    options.write(true).create_new(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        options.mode(0o600);
    }

    let mut file = options.open(&secret_env).expect("create secret");

    writeln!(file, "{HIDDEN_SEED_KEY}={hidden_seed}").expect("write secret");

    EvalSetup {
        eval_bin: PathBuf::from(EVAL_BIN),
        secret_env,
    }
}

/// A linear model that trains for the fewest epochs: a weak base model for a winner to beat.
fn weak_base_genome() -> Genome {
    Genome {
        arch: Arch::Mlp,
        hidden: Vec::new(),
        activation: Activation::Relu,
        optimizer: Optimizer::Sgd,
        learning_rate: 0.001,
        l2: 0.0,
        batch_size: 256,
        epochs: 5,
        init_scale: 1.0,
        seed: 3,
    }
}

/// The `*.json` files in `dir`, sorted by name.
fn json_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read dir")
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| path.extension().is_some_and(|x| x == "json"))
        .collect();

    files.sort();

    files
}

/// The way of ticket 07: a finished run promotes its winners with `brut promote`, the gate
/// of the pool accepts them, a second promotion of the same run adds nothing once the pool
/// holds them, and `brut init --promoted` puts them into generation 0 of a new run, which
/// leaves out a promoted model that fails its limits, with a log line.
///
/// The first run is an MLP run that starts from a weak promoted model, so that it can
/// promote (`Config::can_promote`) and stays fast.
#[test]
fn promoted_models_go_from_a_finished_run_into_the_first_generation_of_a_new_run() {
    let adapter = ModelAdapter::new();
    let from = fresh_dir("promote-from");
    let config = Config {
        promoted: vec![weak_base_genome()],
        ..tiny_config(5)
    };
    let eval = setup_with_hidden_seed(&from, &config, 4_242);

    evolve::run(&from, Some(&eval)).expect("run");

    // The hidden set holds the two best by fitness and the weak model, the only base model.
    let archive = Archive::open(&from.join(ARCHIVE_FILE)).expect("open");
    let base = archive
        .candidates()
        .into_iter()
        .find(|c| c.generator == PROMOTED_GENERATOR)
        .expect("the weak promoted model trained");

    assert!(
        archive
            .hidden_evals()
            .iter()
            .any(|h| h.candidate_id == base.id)
    );

    let empty = fresh_dir("promote-empty-pool");
    let path = |p: &Path| p.to_str().expect("utf-8 path").to_string();
    let out = run_brut(&[
        "promote",
        &path(&from),
        "--commit",
        TEST_COMMIT,
        "--pool",
        &path(&empty),
    ]);

    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // The files are what `promote::choose` gives: at most two, each beats the base genome.
    let expected =
        promote::choose(&archive, TEST_COMMIT, &model_seed_ids(), &adapter).expect("promotion");
    let promoted: Vec<&PromotedModel> = expected.promoted().collect();
    let written = from.join(PROMOTE_DIR).join(MODEL_DIR);
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();

    assert!(!promoted.is_empty(), "{stdout}");
    assert!(promoted.len() <= 2);
    assert_eq!(expected.best_base_model.id, base.id);
    assert_eq!(json_files(&written).len(), promoted.len());

    for m in &promoted {
        let id = adapter.fingerprint(m);
        let text = std::fs::read_to_string(written.join(format!("{id}.json"))).expect("read");

        assert_eq!(text, format!("{}\n", adapter.to_json(m).expect("json")));
        assert!(m.provenance.hidden_primary > expected.best_base_model.hidden_primary);
        assert_eq!(m.provenance.run_id, archive.run_id().expect("run id"));
        assert!(stdout.contains(&format!("promoted      {id}")), "{stdout}");
    }

    let manifest: PromotionManifest = serde_json::from_str(
        &std::fs::read_to_string(from.join(PROMOTE_DIR).join("proposal.json"))
            .expect("read manifest"),
    )
    .expect("manifest json");

    assert_eq!(manifest, expected.manifest());
    assert_eq!(manifest.adapter, "model");

    // The gate of the pool, which CI runs on a pull request, accepts the files.
    let gate = Command::new(INGEST_BIN)
        .arg("--models")
        .arg(&written)
        .output()
        .expect("run brut-ingest");

    assert!(
        gate.status.success(),
        "{}",
        String::from_utf8_lossy(&gate.stdout)
    );

    // A human merges them into the pool. A second promotion of the run adds nothing.
    let pool = fresh_dir("promote-pool");

    for file in json_files(&written) {
        std::fs::copy(&file, pool.join(file.file_name().expect("name"))).expect("copy");
    }

    let again = run_brut(&[
        "promote",
        &path(&from),
        "--commit",
        TEST_COMMIT,
        "--pool",
        &path(&pool),
    ]);

    assert!(again.status.success());
    assert!(String::from_utf8_lossy(&again.stdout).contains("nothing to promote"));
    assert!(!from.join(PROMOTE_DIR).exists());

    // A valid pool file whose model trains longer than the next run allows.
    let mut long = promoted[0].clone();

    long.genome.epochs = 50;
    long.provenance.lineage[0].id = long.genome.id();

    assert!(adapter.validate(&long).is_ok());

    std::fs::write(
        pool.join(format!("{}.json", adapter.fingerprint(&long))),
        adapter.to_json(&long).expect("json"),
    )
    .expect("write");

    // A new run takes the pool into its config; this test then shrinks the run.
    let into = fresh_dir("promote-into");
    let init = run_brut(&[
        "init",
        &path(&into),
        "--seed",
        "6",
        "--promoted",
        &path(&pool),
    ]);

    assert!(
        init.status.success(),
        "{}",
        String::from_utf8_lossy(&init.stderr)
    );
    assert!(
        String::from_utf8_lossy(&init.stdout)
            .contains(&format!("promoted: {} models", promoted.len() + 1))
    );

    let stored = Config::load(&into.join(CONFIG_FILE)).expect("config");
    let config = Config {
        promoted: stored.promoted.clone(),
        ..tiny_config(6)
    };

    assert_eq!(stored.promoted.len(), promoted.len() + 1);
    config.save(&into.join(CONFIG_FILE)).expect("save");
    evolve::run(&into, None).expect("run");

    let archive = Archive::open(&into.join(ARCHIVE_FILE)).expect("open");
    let entered: HashSet<String> = archive
        .candidates()
        .into_iter()
        .filter(|c| c.generation == 0 && c.generator == PROMOTED_GENERATOR)
        .map(|c| c.id)
        .collect();
    let wanted: HashSet<String> = promoted.iter().map(|m| m.genome.id()).collect();
    let left_out = format!(
        "generation 0: promoted model {} left out: invalid: genome: epochs 50 outside 5..=20",
        long.genome.id()
    );

    assert_eq!(entered, wanted);
    assert!(
        archive
            .records()
            .iter()
            .any(|r| matches!(&r.kind, RecordKind::SafetyEvent { what } if *what == left_out)),
        "no log line: {left_out}"
    );
}

/// A run with the model kinds mlp and no promoted models keeps the hidden set of the runs
/// before CNNs, even with a base genome in its config: the weak base genome, the least fit
/// candidate, gets no hidden query, and `brut promote` refuses the run.
#[test]
fn an_mlp_run_with_base_genomes_keeps_its_hidden_set_and_promotes_nothing() {
    let dir = fresh_dir("mlp-base-hidden");
    let config = Config {
        base_genomes: vec![weak_base_genome()],
        ..tiny_config(5)
    };

    assert!(!config.can_promote());

    let eval = setup_with_hidden_seed(&dir, &config, 4_242);

    evolve::run(&dir, Some(&eval)).expect("run");

    let archive = Archive::open(&dir.join(ARCHIVE_FILE)).expect("open");
    let mut trained: Vec<_> = archive
        .candidates()
        .into_iter()
        .filter(|c| matches!(c.status, CandidateStatus::Ok | CandidateStatus::Quarantined))
        .collect();

    trained.sort_by(|a, b| {
        b.fitness
            .total_cmp(&a.fitness)
            .then_with(|| a.id.cmp(&b.id))
    });

    let base = trained
        .iter()
        .find(|c| c.generator == BASE_GENERATOR)
        .expect("the base genome trained")
        .id
        .clone();
    let best: HashSet<String> = trained
        .iter()
        .take(config.hidden_eval_top_k)
        .map(|c| c.id.clone())
        .collect();
    let queried: HashSet<String> = archive
        .hidden_evals()
        .into_iter()
        .map(|h| h.candidate_id)
        .collect();

    assert!(
        !best.contains(&base),
        "the weak base genome is among the best"
    );
    assert_eq!(queried, best);

    let out = run_brut(&[
        "promote",
        dir.to_str().expect("utf-8 path"),
        "--commit",
        TEST_COMMIT,
        "--pool",
        fresh_dir("mlp-base-pool").to_str().expect("utf-8 path"),
    ]);

    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("no promoted models"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The gate of the pool (`brut-ingest --models`, which CI runs on a pull request) reads only
/// regular files of at most `MAX_CANDIDATE_FILE_BYTES`, and each line it prints stays one
/// line: a FIFO, a symlink to a device and a huge file are rejected at once, and neither a
/// file name nor a provenance with line breaks can forge a line of its log.
#[test]
fn the_model_gate_rejects_unreadable_files_and_keeps_its_log_lines() {
    let dir = fresh_dir("gate-files");
    let adapter = ModelAdapter::new();
    let mut genome = Family::Vgg.base_model();

    // Another seed: no longer a base model of a family, which would be a duplicate.
    genome.seed += 1;

    let model = |generator: &str| PromotedModel {
        genome: genome.clone(),
        provenance: ModelProvenance {
            run_id: "cd".repeat(32),
            task: TaskKind::Shapes,
            master_seed: 4,
            commit: TEST_COMMIT.to_string(),
            valid_primary: 0.9,
            hidden_primary: 0.9,
            best_base_model: BaseModelScore {
                id: "0123456789abcdef".to_string(),
                generator: generator.to_string(),
                valid_primary: 0.8,
                hidden_primary: 0.8,
            },
            lineage: vec![LineageStep {
                id: genome.id(),
                generation: 0,
                generator: INIT_GENERATOR.to_string(),
                parents: Vec::new(),
            }],
        },
    };
    let forged = "base-model/\n::error::forged\nok    0123456789abcdef  forged";

    assert!(adapter.validate(&model("base-model/resnet")).is_ok());
    std::fs::write(
        dir.join(format!("{}.json", genome.id())),
        adapter.to_json(&model(forged)).expect("json"),
    )
    .expect("write");
    std::fs::write(dir.join("x\nok    0123456789abcdef  forged.json"), "{}").expect("write");
    std::fs::File::create(dir.join("cccccccccccccccc.json"))
        .and_then(|file| file.set_len(MAX_CANDIDATE_FILE_BYTES + 1))
        .expect("a sparse file above the cap");

    let mut files = 3;

    #[cfg(unix)]
    {
        std::os::unix::fs::symlink("/dev/zero", dir.join("bbbbbbbbbbbbbbbb.json"))
            .expect("symlink");
        files += 1;

        let made = Command::new("mkfifo")
            .arg(dir.join("aaaaaaaaaaaaaaaa.json"))
            .status()
            .is_ok_and(|status| status.success());

        if made {
            files += 1;
        }
    }

    let mut child = Command::new(INGEST_BIN)
        .arg("--models")
        .arg(&dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start brut-ingest");
    let started = Instant::now();

    while child.try_wait().expect("wait").is_none() {
        if started.elapsed() > Duration::from_secs(60) {
            let _ = child.kill();

            panic!("brut-ingest did not finish within 60 s");
        }

        std::thread::sleep(Duration::from_millis(20));
    }

    let out = child.wait_with_output().expect("output");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let lines: Vec<&str> = stdout.lines().collect();

    assert_eq!(out.status.code(), Some(1), "{stdout}");
    assert!(
        lines
            .iter()
            .all(|line| line.starts_with("REJECT ") || line.starts_with("brut-ingest: ")),
        "{stdout}"
    );
    assert_eq!(
        lines.iter().filter(|l| l.starts_with("REJECT ")).count(),
        files,
        "{stdout}"
    );
    assert!(stdout.contains(&format!("brut-ingest: 0 accepted, {files} rejected")));

    // The forged generator stays inside the line that rejects its file.
    let line = lines
        .iter()
        .find(|line| line.contains(&format!("{}.json", genome.id())))
        .expect("a line for the forged model");

    assert!(
        line.contains("best_base_model must be a base model"),
        "{line}"
    );
    assert!(line.contains("::error::forged"), "{line}");
}

/// `small_cnn` and two MLPs as the promoted models of a short run of both kinds on bars and
/// stripes. `max_params` of 700 leaves out every base model of the CNN families (the
/// smallest, ShuffleNet, has 710) and the wide MLP (4146), so generation 0 holds the CNN, the
/// narrow MLP and 2 random genomes, an MLP and a CNN; 3 children follow. A clock of 600 s per
/// candidate keeps a busy machine from stopping a training.
fn promoted_config() -> Config {
    let narrow = Genome {
        arch: Arch::Mlp,
        hidden: vec![2],
        activation: Activation::Tanh,
        optimizer: Optimizer::Adam,
        learning_rate: 0.01,
        l2: 0.0,
        batch_size: 64,
        epochs: 5,
        init_scale: 1.0,
        seed: 12,
    };
    let wide = Genome {
        hidden: vec![16],
        ..narrow.clone()
    };

    Config {
        task: TaskKind::BarsAndStripes,
        master_seed: 3,
        population_init: 2,
        generations: 1,
        children_per_generation: 3,
        crossover_rate: 0.5,
        limits: Limits {
            max_params: 700,
            max_candidate_seconds: 600.0,
            max_total_seconds: 3600.0,
            ..Limits::default()
        },
        threads: 0,
        models: ModelKinds::Both,
        promoted: vec![small_cnn(), narrow, wide],
        ..Config::default()
    }
}

/// A run that starts from promoted models gives the same files, byte for byte once the clock
/// fields are masked, from the same seed: `brut run --no-hidden` in two child processes with
/// `RAYON_NUM_THREADS` at 1 and 2, as for the CNN run of ticket 05.
#[test]
fn a_run_with_promoted_models_is_reproducible_byte_for_byte() {
    let config = promoted_config();
    let start = |name: &str, threads: &str| {
        let dir = fresh_dir(name);

        config.save(&dir.join(CONFIG_FILE)).expect("save config");

        let child = Command::new(BRUT_BIN)
            .arg("run")
            .arg(&dir)
            .arg("--no-hidden")
            .env("RAYON_NUM_THREADS", threads)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("start brut");

        (dir, child)
    };
    let (one, first) = start("promoted-1", "1");
    let (two, second) = start("promoted-2", "2");
    let summaries: Vec<String> = [first, second]
        .into_iter()
        .map(|child| {
            let out = child.wait_with_output().expect("brut run");

            assert!(
                out.status.success(),
                "brut run failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );

            mask(
                &String::from_utf8(out.stdout).expect("utf-8 summary"),
                "wall_seconds",
                "0.0",
            )
        })
        .collect();
    let files = run_files(&one);

    assert_eq!(run_files(&two), files);
    assert_eq!(summaries[0], summaries[1]);

    let archive = Archive::open(&one.join(ARCHIVE_FILE)).expect("open");
    let candidates = archive.candidates();
    let mut first_generation: Vec<(&str, bool)> = candidates
        .iter()
        .filter(|c| c.generation == 0)
        .map(|c| (c.generator.as_str(), c.genome.arch.is_mlp()))
        .collect();
    let logged = |what: &str| {
        archive
            .records()
            .iter()
            .filter(|r| matches!(&r.kind, RecordKind::SafetyEvent { what: w } if w.contains(what)))
            .count()
    };
    let wide = &config.promoted[2];

    // The archive orders a generation by id.
    first_generation.sort();

    assert_eq!(
        first_generation,
        vec![
            (INIT_GENERATOR, false),
            (INIT_GENERATOR, true),
            (PROMOTED_GENERATOR, false),
            (PROMOTED_GENERATOR, true),
        ]
    );

    for genome in &config.promoted[..2] {
        let entry = archive.get(&genome.id()).expect("a promoted model trained");

        assert_eq!(entry.generator, PROMOTED_GENERATOR);
        assert_eq!(&entry.genome, genome);
    }
    assert_eq!(logged("generation 0: base model "), 16);
    assert_eq!(
        logged(&format!(
            "generation 0: promoted model {} left out: invalid: genome: 4146 params > max 700",
            wide.id()
        )),
        1
    );
    assert_eq!(files.len(), 3 + candidates.len());
    assert_eq!(candidates.iter().filter(|c| c.generation == 1).count(), 3);

    for c in &candidates {
        assert_eq!(c.status, CandidateStatus::Ok, "{}", c.id);
        assert!(!c.train.stopped_by_budget, "{}", c.id);
    }
}

/// A run directory that `brut run --no-hidden` wrote at commit bec9aa1, before CNNs: the
/// config of `tiny_config(3)` with 1 thread, its archive and its 10 model files.
const PRE_CNN_RUN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/pre_cnn_run");

/// A run written before CNNs still opens and verifies, its model files give back their
/// archived validation scores, and it resumes for one more generation.
#[test]
fn a_run_written_before_cnns_opens_scores_and_resumes() {
    let dir = fresh_dir("pre-cnn");
    let fixture = Path::new(PRE_CNN_RUN);

    std::fs::create_dir_all(dir.join(MODELS_DIR)).expect("models dir");

    for name in [CONFIG_FILE, ARCHIVE_FILE, "archive.head"] {
        std::fs::copy(fixture.join(name), dir.join(name)).expect("copy");
    }

    for entry in std::fs::read_dir(fixture.join(MODELS_DIR)).expect("models") {
        let path = entry.expect("entry").path();
        let name = path.file_name().expect("file name");

        std::fs::copy(&path, dir.join(MODELS_DIR).join(name)).expect("copy model");
    }

    let records = Archive::verify(&dir.join(ARCHIVE_FILE)).expect("verify");
    let archive = Archive::open(&dir.join(ARCHIVE_FILE)).expect("open");
    let config = Config::load(&dir.join(CONFIG_FILE)).expect("config");
    let spec = config.task.spec();
    let valid = task::generate(&spec, Split::Valid, derive_seed(config.master_seed, "data"));
    let candidates = archive.candidates();

    assert_eq!(records, 12);
    assert_eq!(candidates.len(), 4 + 2 * 3);
    assert!(config.base_genomes.is_empty());

    let mut scored = 0;

    for c in &candidates {
        assert!(c.genome.arch.is_mlp(), "{}", c.id);

        if let Some(file) = &c.model_file {
            let model = Model::from_json(&std::fs::read(dir.join(file)).expect("read"))
                .expect("an old model file parses");

            assert!(matches!(model, Model::Mlp(_)), "{}", c.id);
            assert_eq!(score::evaluate(&model, &spec, &valid), c.valid, "{}", c.id);
            scored += 1;
        }
    }

    assert_eq!(scored, 10);

    // One more generation on top of the old archive.
    let more = Config {
        generations: 3,
        ..config
    };

    more.save(&dir.join(CONFIG_FILE)).expect("save");

    let summary = evolve::run(&dir, None).expect("resume");
    let archive = Archive::open(&dir.join(ARCHIVE_FILE)).expect("open");

    assert_eq!(summary.stop_reason, "completed");
    assert_eq!(summary.candidates, 10 + 3);
    assert!(archive.candidates().iter().any(|c| c.generation == 3));
    assert!(Archive::verify(&dir.join(ARCHIVE_FILE)).is_ok());
}

#[test]
fn same_seed_gives_same_results() {
    let a = fresh_dir("det-a");
    let b = fresh_dir("det-b");
    let config = tiny_config(7);

    let ea = setup(&a, &config);
    let eb = setup(&b, &config);

    evolve::run(&a, Some(&ea)).expect("run a");
    evolve::run(&b, Some(&eb)).expect("run b");

    let fa = fitness_list(&a);
    let fb = fitness_list(&b);

    assert!(!fa.is_empty());
    assert_eq!(fa, fb);

    // A different seed explores different candidates.
    let c = fresh_dir("det-c");
    let ec = setup(&c, &tiny_config(8));

    evolve::run(&c, Some(&ec)).expect("run c");

    assert_ne!(fa, fitness_list(&c));
}

#[test]
fn thread_count_does_not_change_results() {
    let one = fresh_dir("threads-1");
    let three = fresh_dir("threads-3");

    for (dir, threads) in [(&one, 1), (&three, 3)] {
        let config = Config {
            threads,
            ..tiny_config(9)
        };

        setup(dir, &config);
        evolve::run(dir, None).expect("run");
    }

    let a = fitness_list(&one);

    assert!(!a.is_empty());
    assert_eq!(a, fitness_list(&three));
}

#[test]
fn stop_file_stops_cleanly() {
    let dir = fresh_dir("stop");
    let eval = setup(&dir, &tiny_config(2));

    safety::request_stop(&dir).expect("stop");

    let summary = evolve::run(&dir, Some(&eval)).expect("a stop is not an error");

    assert_eq!(summary.candidates, 0);
    assert!(
        summary.stop_reason.contains("kill switch"),
        "{}",
        summary.stop_reason
    );
    assert!(summary.hidden.is_empty());

    let archive = Archive::open(&dir.join(ARCHIVE_FILE)).expect("open");
    let events = archive
        .records()
        .iter()
        .filter(|r| matches!(r.kind, RecordKind::SafetyEvent { .. }))
        .count();

    assert!(events >= 1);
    assert!(matches!(
        archive.records().last().map(|r| &r.kind),
        Some(RecordKind::RunFinished { .. })
    ));

    // Without the STOP file the run proceeds.
    std::fs::remove_file(dir.join(safety::STOP_FILE)).expect("remove stop");

    let summary = evolve::run(&dir, Some(&eval)).expect("run");

    assert_eq!(summary.stop_reason, "completed");
    assert_eq!(summary.generations_done, 2);
}

/// Poll `ready` every few milliseconds until it holds. False if `limit` passes first.
fn wait_until(limit: Duration, ready: impl Fn() -> bool) -> bool {
    let started = Instant::now();

    while !ready() {
        if started.elapsed() > limit {
            return false;
        }

        std::thread::sleep(Duration::from_millis(5));
    }

    true
}

#[test]
fn stop_file_interrupts_a_running_generation() {
    let dir = fresh_dir("stop-mid");

    // One thread and long-trained genomes (up to 10 000 epochs). Children train largest
    // first, and the largest alone takes 5.5 s in a release build and 8.9 s in the dev
    // profile (measured), so it is still training when the STOP file appears.
    let config = Config {
        population_init: 16,
        generations: 0,
        threads: 1,
        limits: Limits {
            max_epochs: 10_000,
            max_train_flops: 100_000_000_000,
            ..Limits::default()
        },
        ..tiny_config(6)
    };

    setup(&dir, &config);

    let (summary, seconds) = std::thread::scope(|scope| {
        let stopper = scope.spawn(|| {
            // The run creates its models directory just before generation 0, and the first
            // child starts training about 0.5 ms later (measured). STOP written 200 ms after
            // the directory appears therefore lands inside that child's training, not before
            // the generation starts.
            let ready = wait_until(Duration::from_secs(30), || dir.join(MODELS_DIR).is_dir());

            assert!(ready, "the run did not create {MODELS_DIR} within 30 s");
            std::thread::sleep(Duration::from_millis(200));
            safety::request_stop(&dir).expect("stop");

            Instant::now()
        });

        let summary = evolve::run(&dir, None).expect("a stop is not an error");
        let finished = Instant::now();
        let stopped_at = stopper.join().expect("stop writer");

        (
            summary,
            finished.saturating_duration_since(stopped_at).as_secs_f64(),
        )
    });

    assert!(seconds < 10.0, "took {seconds:.1} s after STOP");
    assert!(summary.candidates < 16, "{} candidates", summary.candidates);
    assert!(
        summary.stop_reason.contains("kill switch")
            && summary.stop_reason.contains("during generation 0"),
        "{}",
        summary.stop_reason
    );

    let archive = Archive::open(&dir.join(ARCHIVE_FILE)).expect("open");
    let stopped: Vec<_> = archive
        .candidates()
        .into_iter()
        .filter(|c| c.train.stopped_by_kill_switch)
        .collect();

    // The child in training when STOP came was cut short, not finished or skipped.
    assert!(
        !stopped.is_empty(),
        "no candidate was stopped during training"
    );

    for c in stopped {
        assert_eq!(c.status, CandidateStatus::Stopped);
        assert!(c.model_file.is_none());
    }
}

#[test]
fn resume_continues_generations() {
    let dir = fresh_dir("resume");
    let mut config = tiny_config(3);

    config.generations = 1;

    // Stages without the hidden evaluation, which only comes at the very end.
    let eval = setup(&dir, &config);
    let first = evolve::run(&dir, None).expect("first run");

    assert_eq!(first.generations_done, 1);
    assert_eq!(first.candidates, 4 + 3);
    assert!(first.hidden.is_empty());

    let before = fitness_list(&dir);

    config.generations = 3;
    config.save(&dir.join(CONFIG_FILE)).expect("save");

    let second = evolve::run(&dir, Some(&eval)).expect("second run");

    assert_eq!(second.generations_done, 3);
    assert_eq!(second.candidates, 4 + 3 * 3);
    assert!(!second.hidden.is_empty());

    // Old records are kept unchanged and the FLOP count carries over.
    assert_eq!(fitness_list(&dir)[..before.len()], before[..]);
    assert!(second.flops_used > first.flops_used);

    let archive = Archive::open(&dir.join(ARCHIVE_FILE)).expect("open");
    let starts = archive
        .records()
        .iter()
        .filter(|r| matches!(r.kind, RecordKind::RunStarted { .. }))
        .count();

    assert_eq!(starts, 2);

    // Ids stay unique across both runs.
    let ids: std::collections::HashSet<String> =
        archive.candidates().into_iter().map(|c| c.id).collect();

    assert_eq!(ids.len(), second.candidates);
}

fn record_count(dir: &Path) -> usize {
    Archive::open(&dir.join(ARCHIVE_FILE))
        .expect("open archive")
        .records()
        .len()
}

#[test]
fn resume_refuses_a_changed_experiment() {
    let dir = fresh_dir("resume-changed");
    let mut config = tiny_config(3);

    config.generations = 1;
    setup(&dir, &config);
    evolve::run(&dir, None).expect("first run");

    let before = record_count(&dir);

    // Each of these changes what the archived candidates mean.
    let changed = [
        Config {
            task: TaskKind::Parity,
            generations: 2,
            ..config.clone()
        },
        Config {
            master_seed: 99,
            generations: 2,
            ..config.clone()
        },
        Config {
            score: ScoreWeights {
                param_penalty: 0.5,
                ..config.score
            },
            generations: 2,
            ..config.clone()
        },
        Config {
            limits: Limits {
                max_width: 8,
                ..config.limits
            },
            generations: 2,
            ..config.clone()
        },
    ];

    for c in changed {
        c.save(&dir.join(CONFIG_FILE)).expect("save");

        let result = evolve::run(&dir, None);

        assert!(
            matches!(&result, Err(BrutError::Invalid(m)) if m.contains("first run")),
            "{:?}",
            result.err()
        );
        assert_eq!(record_count(&dir), before, "archive must stay unchanged");
    }

    // More generations, other threads and other run-time limits are fine.
    let allowed = Config {
        generations: 2,
        threads: 1,
        limits: Limits {
            max_candidate_seconds: 20.0,
            max_total_seconds: 200.0,
            max_total_flops: config.limits.max_total_flops / 2,
            ..config.limits
        },
        ..config.clone()
    };

    allowed.save(&dir.join(CONFIG_FILE)).expect("save");

    let summary = evolve::run(&dir, None).expect("resume with allowed changes");

    assert_eq!(summary.generations_done, 2);
}

#[test]
fn hidden_budget_exhaustion_is_logged_not_fatal() {
    let dir = fresh_dir("budget");
    let config = Config {
        hidden_query_budget: 1,
        hidden_eval_top_k: 3,
        ..tiny_config(4)
    };
    let eval = setup(&dir, &config);
    let summary = evolve::run(&dir, Some(&eval)).expect("run");

    assert_eq!(summary.hidden.len(), 1);

    let archive = Archive::open(&dir.join(ARCHIVE_FILE)).expect("open");
    let failures = archive
        .records()
        .iter()
        .filter(|r| matches!(&r.kind, RecordKind::SafetyEvent { what } if what.contains("budget exhausted")))
        .count();

    assert_eq!(failures, 2);
}

/// Overwrite a candidate's model file with zero weights. It then predicts one class for
/// every input, so the hidden evaluator scores it 0.5 on the balanced spirals task, while its
/// archived validation score stays as it was.
fn sabotage_model(dir: &Path, id: &str) {
    let archive = Archive::open(&dir.join(ARCHIVE_FILE)).expect("open");
    let file = archive
        .get(id)
        .and_then(|c| c.model_file)
        .expect("model file");
    let path = dir.join(file);
    let mut model: Mlp =
        serde_json::from_slice(&std::fs::read(&path).expect("read model")).expect("model json");

    for w in model.weights.iter_mut().chain(model.biases.iter_mut()) {
        w.fill(0.0);
    }

    std::fs::write(&path, serde_json::to_vec(&model).expect("json")).expect("write model");
}

fn queries_used(dir: &Path) -> u64 {
    let text = std::fs::read_to_string(dir.join("secret").join("eval_state.json")).expect("state");
    let value: serde_json::Value = serde_json::from_str(&text).expect("state json");

    value["queries_used"].as_u64().expect("queries_used")
}

#[test]
fn quarantine_is_final_and_does_not_shift_the_hidden_set() {
    let dir = fresh_dir("quarantine");
    let config = tiny_config(5);
    let eval = setup(&dir, &config);

    // Search without hidden evaluation, then break the best model on disk.
    let search = evolve::run(&dir, None).expect("search");
    let best = search.best_id.clone().expect("best id");

    sabotage_model(&dir, &best);

    // The hidden evaluation scores the broken model 0.5 and quarantines it.
    let hidden = evolve::run(&dir, Some(&eval)).expect("hidden phase");
    let h = hidden
        .hidden
        .iter()
        .find(|h| h.candidate_id == best)
        .expect("best was evaluated");

    assert_eq!(h.hidden_primary, 0.5);
    assert!(h.valid_primary > 0.5 + config.hidden_gap_threshold);
    assert!(h.quarantined);
    assert_eq!(hidden.hidden.len(), config.hidden_eval_top_k);
    assert_ne!(hidden.best_id.as_deref(), Some(best.as_str()));

    let archive = Archive::open(&dir.join(ARCHIVE_FILE)).expect("open");

    assert_eq!(
        archive.get(&best).map(|c| c.status),
        Some(CandidateStatus::Quarantined)
    );

    // Running again sends nobody new: the hidden set is the fixed top k by validation
    // fitness, quarantined candidates included. Moving down the ranking would turn the
    // hidden data into selection data.
    let used = queries_used(&dir);
    let again = evolve::run(&dir, Some(&eval)).expect("run again");

    assert_eq!(again.hidden, hidden.hidden);
    assert_eq!(queries_used(&dir), used);

    // More generations after a hidden evaluation are refused, with or without the evaluator.
    let more = Config {
        generations: config.generations + 1,
        ..config.clone()
    };

    more.save(&dir.join(CONFIG_FILE)).expect("save");

    let before = record_count(&dir);

    for setup in [Some(&eval), None] {
        let result = evolve::run(&dir, setup);

        assert!(
            matches!(&result, Err(BrutError::Invalid(m)) if m.contains("hidden evaluation")),
            "{:?}",
            result.err()
        );
        assert_eq!(record_count(&dir), before);
    }
}

#[test]
fn cli_full_cycle_and_tamper_detection() {
    let dir = fresh_dir("cli");
    let run_dir = dir.join("run");
    let run = run_dir.to_str().expect("utf8 path");

    // Usage errors give exit code 2.
    assert_eq!(run_brut(&[]).status.code(), Some(2));
    assert_eq!(run_brut(&["frobnicate"]).status.code(), Some(2));
    assert_eq!(
        run_brut(&["init", run, "--task", "nope"]).status.code(),
        Some(2)
    );
    assert_eq!(run_brut(&["help"]).status.code(), Some(0));

    // init, and refuse a second init.
    assert_eq!(
        run_brut(&["init", run, "--task", "parity", "--seed", "11"])
            .status
            .code(),
        Some(0)
    );
    assert_eq!(run_brut(&["init", run]).status.code(), Some(1));

    // Shrink the default config so the test stays fast.
    let mut config = Config::load(&run_dir.join(CONFIG_FILE)).expect("load");

    assert_eq!(config.task, TaskKind::Parity);
    assert_eq!(config.master_seed, 11);

    config.population_init = 3;
    config.generations = 1;
    config.children_per_generation = 2;
    config.limits = Limits {
        max_hidden_layers: 2,
        max_width: 12,
        max_epochs: 30,
        ..Limits::default()
    };
    config.hidden_eval_top_k = 1;
    config.save(&run_dir.join(CONFIG_FILE)).expect("save");

    let out = run_brut(&["run", run]);

    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let summary: RunSummary = serde_json::from_slice(&out.stdout).expect("summary json");

    assert_eq!(summary.hidden.len(), 1);

    // status and lineage work.
    let status = run_brut(&["status", run]);

    assert_eq!(status.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&status.stdout).contains("hidden evaluations: 1"));

    let best = summary.best_id.clone().expect("best id");

    assert_eq!(run_brut(&["lineage", run, &best]).status.code(), Some(0));
    assert_eq!(
        run_brut(&["lineage", run, "0000000000000000"])
            .status
            .code(),
        Some(1)
    );

    // approve: needs --by; refuses a candidate without hidden evaluation.
    let evaluated = summary.hidden[0].candidate_id.clone();

    assert_eq!(
        run_brut(&["approve", run, &evaluated]).status.code(),
        Some(2)
    );

    let archive = Archive::open(&run_dir.join(ARCHIVE_FILE)).expect("open");
    let unevaluated = archive
        .candidates()
        .into_iter()
        .find(|c| c.id != evaluated)
        .map(|c| c.id)
        .expect("other");

    assert_eq!(
        run_brut(&["approve", run, &unevaluated, "--by", "tester"])
            .status
            .code(),
        Some(1)
    );

    let approve = run_brut(&[
        "approve",
        run,
        &evaluated,
        "--by",
        "tester",
        "--note",
        "looks fine",
    ]);

    if summary.hidden[0].quarantined {
        assert_eq!(approve.status.code(), Some(1));
    } else {
        assert_eq!(
            approve.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&approve.stderr)
        );
        assert_eq!(
            run_brut(&["approve", run, &evaluated, "--by", "tester"])
                .status
                .code(),
            Some(1)
        );
    }

    // verify: OK, then detect an edited record.
    let ok = run_brut(&["verify", run]);

    assert_eq!(ok.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&ok.stdout).starts_with("OK "));

    let path = run_dir.join(ARCHIVE_FILE);
    let text = std::fs::read_to_string(&path).expect("read archive");

    // Removing the last record (e.g. a HiddenEval or Approved record) is detected.
    let lines: Vec<&str> = text.lines().collect();
    let shorter: String = lines[..lines.len() - 1]
        .iter()
        .map(|l| format!("{l}\n"))
        .collect();

    std::fs::write(&path, shorter).expect("truncate");

    let truncated = run_brut(&["verify", run]);

    assert_eq!(truncated.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&truncated.stderr).contains("removed from the end"));

    std::fs::write(&path, &text).expect("restore");
    assert_eq!(run_brut(&["verify", run]).status.code(), Some(0));
    let tampered = text.replacen("\"generator\":\"init\"", "\"generator\":\"evil\"", 1);

    assert_ne!(tampered, text);
    std::fs::write(&path, tampered).expect("tamper");

    let bad = run_brut(&["verify", run]);

    assert_eq!(bad.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&bad.stderr).contains("integrity"));

    // stop sets the kill switch.
    assert_eq!(run_brut(&["stop", run]).status.code(), Some(0));
    assert!(run_dir.join(safety::STOP_FILE).exists());
}

#[test]
fn orchestrator_never_touches_the_secret() {
    // Orchestrator files of both modes (learner mode: contract S1).
    let sources = [
        ("src/evolve.rs", include_str!("../src/evolve.rs")),
        ("src/main.rs", include_str!("../src/main.rs")),
        ("src/meta_evolve.rs", include_str!("../src/meta_evolve.rs")),
        ("src/meta_report.rs", include_str!("../src/meta_report.rs")),
        (
            "src/lineage_store.rs",
            include_str!("../src/lineage_store.rs"),
        ),
    ];

    // Evaluator-side names, secret keys and paths of the external evaluation sets.
    let forbidden = [
        "HiddenEvaluator",
        "LearnerEvaluator",
        "parse_env_file",
        "HIDDEN_SEED_KEY",
        "LEARNER_SEED_KEY",
        "HiddenToken",
        "hidden_tier_tasks",
        "hidden_grammar_pool",
        "BRUT_ARC_DATA_ROOT",
        "BRUT_ARC_MANIFEST",
        "data/evaluation",
        "arc-agi-2",
        "ConceptARC/corpus",
    ];

    for (name, text) in sources {
        for word in forbidden {
            assert!(!text.contains(word), "{name} references {word}");
        }
    }
}
