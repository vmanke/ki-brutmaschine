//! End-to-end tests: tiny real runs with the hidden evaluator, resume, kill switch, tamper detection.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use brutmaschine::archive::{Archive, CandidateStatus, RecordKind};
use brutmaschine::cnn::genome::{
    Block, CnnActivation, CnnGenome, Dims, Downsample, Head, Norm, Stage, Stem,
};
use brutmaschine::config::{Config, Limits, ScoreWeights};
use brutmaschine::evolve::{
    self, ARCHIVE_FILE, BASE_GENERATOR, CONFIG_FILE, EvalSetup, MODELS_DIR, RunSummary,
};
use brutmaschine::genome::{Activation, Arch, Genome, Optimizer};
use brutmaschine::hidden_eval::create_secret;
use brutmaschine::model::Model;
use brutmaschine::nn::Mlp;
use brutmaschine::safety;
use brutmaschine::score;
use brutmaschine::task::{self, Split, TaskKind};
use brutmaschine::util::{BrutError, derive_seed};

const EVAL_BIN: &str = env!("CARGO_BIN_EXE_brut-eval");
const BRUT_BIN: &str = env!("CARGO_BIN_EXE_brut");

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

/// The same CNN run trains to the same candidates and model files with 1 or 2 worker threads.
#[test]
fn a_cnn_run_does_not_depend_on_the_thread_count() {
    let files = |dir: &Path| -> Vec<(String, Vec<u8>)> {
        let archive = Archive::open(&dir.join(ARCHIVE_FILE)).expect("open");

        archive
            .candidates()
            .into_iter()
            .filter_map(|c| c.model_file.map(|f| (c.id, f)))
            .map(|(id, f)| (id, std::fs::read(dir.join(f)).expect("read")))
            .collect()
    };
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
