//! Run configuration and hard resource limits.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::genome::{Activation, Arch, Genome, MIN_EPOCHS, Optimizer};
use crate::hidden_eval::MIN_HIDDEN_SCORE_STEP;
use crate::task::TaskKind;
use crate::util::{BrutError, Result, write_atomic};

/// Upper bound for `threads` (guards against typos like 10000).
pub const MAX_THREADS: usize = 1024;

/// Hard cap on `max_hidden_layers`. `brut-eval` also refuses deeper models.
pub const HARD_MAX_HIDDEN_LAYERS: usize = 64;

/// Hard cap on `max_width`. `brut-eval` also refuses wider layers.
pub const HARD_MAX_WIDTH: usize = 4096;

/// Hard cap on `max_params`. `brut-eval` also refuses larger models,
/// so every model a run can train can also be scored.
pub const HARD_MAX_PARAMS: usize = 2_000_000;

/// Hard cap on `max_epochs`.
pub const HARD_MAX_EPOCHS: usize = 100_000;

/// Hard cap on `population_init` and `children_per_generation`.
pub const HARD_MAX_POPULATION: usize = 10_000;

/// Hard limits. A candidate or a run that would exceed them is stopped.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    /// Max number of hidden layers in one network.
    pub max_hidden_layers: usize,
    /// Max neurons in one hidden layer.
    pub max_width: usize,
    /// Max trainable parameters in one network.
    pub max_params: usize,
    /// Max training epochs for one candidate.
    pub max_epochs: usize,
    /// Max estimated training FLOPs for one candidate.
    pub max_train_flops: u64,
    /// Max wall-clock seconds for training one candidate.
    pub max_candidate_seconds: f64,
    /// Max wall-clock seconds for the whole run.
    pub max_total_seconds: f64,
    /// Max estimated FLOPs for the whole run (all candidates).
    pub max_total_flops: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_hidden_layers: 4,
            max_width: 64,
            max_params: 20_000,
            max_epochs: 400,
            max_train_flops: 20_000_000_000,
            max_candidate_seconds: 30.0,
            max_total_seconds: 300.0,
            max_total_flops: 2_000_000_000_000,
        }
    }
}

/// Weights of the fitness function. See `score::fitness`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreWeights {
    /// Penalty per log10(parameter count).
    pub param_penalty: f64,
    /// Penalty per log10(training FLOPs).
    pub flop_penalty: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        ScoreWeights {
            param_penalty: 0.005,
            flop_penalty: 0.002,
        }
    }
}

/// Everything one run needs. Stored as `config.json` in the run directory.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub task: TaskKind,
    /// Seeds all randomness except the hidden test split.
    pub master_seed: u64,
    /// Random genomes created before evolution starts.
    pub population_init: usize,
    /// Number of evolution steps.
    pub generations: usize,
    /// New candidates per generation.
    pub children_per_generation: usize,
    /// Probability that a child is made by crossover of two parents (else mutation of one).
    pub crossover_rate: f64,
    pub limits: Limits,
    pub score: ScoreWeights,
    /// How many top candidates go to the hidden evaluation at the end.
    pub hidden_eval_top_k: usize,
    /// Quarantine a candidate when validation score minus hidden score exceeds this.
    /// Must be at least twice `hidden_score_step`.
    pub hidden_gap_threshold: f64,
    /// Total hidden-evaluation queries allowed for this run directory.
    pub hidden_query_budget: usize,
    /// Hidden scores are released rounded down to this step (limits leakage).
    /// At least `MIN_HIDDEN_SCORE_STEP` (0.01); `brut-eval` enforces that floor as well.
    pub hidden_score_step: f64,
    /// Worker threads for parallel training. 0 means all cores.
    pub threads: usize,
    /// Genomes that join generation 0 before the `population_init` random ones, such as
    /// CNNs, which no random genome is. Each must pass `Genome::validate` for `limits` and
    /// the task. The JSON leaves an empty list out, so configs without base genomes keep
    /// their bytes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub base_genomes: Vec<Genome>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            task: TaskKind::TwoSpirals,
            master_seed: 1,
            population_init: 16,
            generations: 30,
            children_per_generation: 16,
            crossover_rate: 0.25,
            limits: Limits::default(),
            score: ScoreWeights::default(),
            hidden_eval_top_k: 3,
            hidden_gap_threshold: 0.05,
            hidden_query_budget: 30,
            hidden_score_step: 0.01,
            threads: 0,
            base_genomes: Vec::new(),
        }
    }
}

impl Config {
    /// Read and validate a JSON config file.
    pub fn load(path: &Path) -> Result<Config> {
        let text = std::fs::read_to_string(path)?;
        let config: Config = serde_json::from_str(&text)?;

        config.validate()?;

        Ok(config)
    }

    /// Write this config as pretty JSON (atomic write).
    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;

        let mut bytes = serde_json::to_vec_pretty(self)?;

        bytes.push(b'\n');

        write_atomic(path, &bytes)
    }

    /// Fields that differ from `earlier` and may not change when a run is resumed.
    ///
    /// Only `generations`, `threads` and the run-time limits (`max_candidate_seconds`,
    /// `max_total_seconds`, `max_total_flops`) may change. Every other field defines the
    /// experiment: the task, the seeds, the score weights and the search space. Changing one
    /// would mix candidates that were trained, scored or selected under different rules.
    /// Returns dotted field names such as `"task"` or `"limits.max_width"`; empty means compatible.
    pub fn resume_conflicts(&self, earlier: &Config) -> Vec<String> {
        let neutral = |c: &Config| Config {
            generations: 0,
            threads: 0,
            limits: Limits {
                max_candidate_seconds: 0.0,
                max_total_seconds: 0.0,
                max_total_flops: 0,
                ..c.limits
            },
            ..c.clone()
        };

        let (Ok(now), Ok(before)) = (
            serde_json::to_value(neutral(self)),
            serde_json::to_value(neutral(earlier)),
        ) else {
            return vec!["config".to_string()];
        };

        let mut out = Vec::new();

        diff_fields(&now, &before, "", &mut out);

        out
    }

    /// Reject nonsense values: zero sizes, NaN, rates outside [0, 1], sizes above the
    /// hard caps (`HARD_MAX_*`, so a valid config can neither hang nor exhaust memory), etc.
    ///
    /// Also checks feasibility: the smallest genome (linear model, `MIN_EPOCHS`)
    /// must fit `max_params` and `max_train_flops`, so random genomes always exist.
    pub fn validate(&self) -> Result<()> {
        let fail = |m: String| Err(BrutError::Invalid(format!("config: {m}")));
        let l = &self.limits;

        if self.population_init == 0 {
            return fail("population_init must be >= 1".into());
        }

        if self.children_per_generation == 0 {
            return fail("children_per_generation must be >= 1".into());
        }

        if self.population_init > HARD_MAX_POPULATION
            || self.children_per_generation > HARD_MAX_POPULATION
        {
            return fail(format!(
                "population_init and children_per_generation must be <= {HARD_MAX_POPULATION}"
            ));
        }

        if !(0.0..=1.0).contains(&self.crossover_rate) {
            return fail(format!(
                "crossover_rate {} must be in [0, 1]",
                self.crossover_rate
            ));
        }

        if self.threads > MAX_THREADS {
            return fail(format!("threads {} > {MAX_THREADS}", self.threads));
        }

        if l.max_params == 0 || l.max_train_flops == 0 || l.max_total_flops == 0 {
            return fail("max_params, max_train_flops and max_total_flops must be >= 1".into());
        }

        if l.max_epochs < MIN_EPOCHS {
            return fail(format!("max_epochs {} < {MIN_EPOCHS}", l.max_epochs));
        }

        let caps = [
            (
                "max_hidden_layers",
                l.max_hidden_layers,
                HARD_MAX_HIDDEN_LAYERS,
            ),
            ("max_width", l.max_width, HARD_MAX_WIDTH),
            ("max_params", l.max_params, HARD_MAX_PARAMS),
            ("max_epochs", l.max_epochs, HARD_MAX_EPOCHS),
        ];

        for (name, value, cap) in caps {
            if value > cap {
                return fail(format!("{name} {value} > hard cap {cap}"));
            }
        }

        if l.max_hidden_layers > 0 && l.max_width == 0 {
            return fail("max_width must be >= 1 when hidden layers are allowed".into());
        }

        let seconds_ok = |v: f64| v.is_finite() && v > 0.0;

        if !seconds_ok(l.max_candidate_seconds) || !seconds_ok(l.max_total_seconds) {
            return fail("time limits must be finite and > 0".into());
        }

        let weight_ok = |v: f64| v.is_finite() && v >= 0.0;

        if !weight_ok(self.score.param_penalty) || !weight_ok(self.score.flop_penalty) {
            return fail("score weights must be finite and >= 0".into());
        }

        if !weight_ok(self.hidden_gap_threshold) {
            return fail("hidden_gap_threshold must be finite and >= 0".into());
        }

        if !(MIN_HIDDEN_SCORE_STEP..1.0).contains(&self.hidden_score_step) {
            return fail(format!(
                "hidden_score_step {} must be in [{MIN_HIDDEN_SCORE_STEP}, 1)",
                self.hidden_score_step
            ));
        }

        // The hidden score is rounded down by up to one step, the validation score is not.
        // At most half the threshold may come from rounding, so a model that does as well on
        // the hidden data as on validation is never quarantined.
        if 2.0 * self.hidden_score_step > self.hidden_gap_threshold {
            return fail(format!(
                "hidden_score_step {} must be at most half of hidden_gap_threshold {}",
                self.hidden_score_step, self.hidden_gap_threshold
            ));
        }

        let spec = self.task.spec();
        let smallest = Genome {
            arch: Arch::Mlp,
            hidden: Vec::new(),
            activation: Activation::Relu,
            optimizer: Optimizer::Sgd,
            learning_rate: 0.01,
            l2: 0.0,
            batch_size: 32,
            epochs: MIN_EPOCHS,
            init_scale: 1.0,
            seed: 0,
        };

        if smallest.param_count(&spec) > l.max_params
            || smallest.train_flops(&spec) > l.max_train_flops
        {
            return fail(
                "limits too small: even a linear model with MIN_EPOCHS does not fit".into(),
            );
        }

        if self.base_genomes.len() > HARD_MAX_POPULATION {
            return fail(format!(
                "{} base genomes > {HARD_MAX_POPULATION}",
                self.base_genomes.len()
            ));
        }

        for (i, genome) in self.base_genomes.iter().enumerate() {
            if let Err(e) = genome.validate(l, &spec) {
                return fail(format!("base genome {}: {e}", i + 1));
            }
        }

        Ok(())
    }
}

/// Collect the dotted paths where two JSON values differ (objects are compared key by key).
pub(crate) fn diff_fields(
    a: &serde_json::Value,
    b: &serde_json::Value,
    prefix: &str,
    out: &mut Vec<String>,
) {
    match (a, b) {
        (serde_json::Value::Object(ma), serde_json::Value::Object(mb)) => {
            let mut keys: Vec<&String> = ma.keys().chain(mb.keys()).collect();

            keys.sort();
            keys.dedup();

            for key in keys {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };

                match (ma.get(key), mb.get(key)) {
                    (Some(va), Some(vb)) => diff_fields(va, vb, &path, out),
                    _ => out.push(path),
                }
            }
        }
        _ => {
            if a != b {
                out.push(prefix.to_string());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_genomes_are_left_out_when_empty_and_checked_when_given() {
        let plain = Config::default();
        let text = serde_json::to_string(&plain).expect("json");

        assert!(!text.contains("base_genomes"), "{text}");

        let cnn = crate::cnn::genome::testing::lenet();
        let with_cnn = Config {
            task: TaskKind::Shapes,
            base_genomes: vec![cnn.clone()],
            ..Config::default()
        };
        let text = serde_json::to_string(&with_cnn).expect("json");

        assert!(with_cnn.validate().is_ok());
        assert_eq!(
            serde_json::from_str::<Config>(&text).expect("parse"),
            with_cnn
        );
        assert_eq!(
            with_cnn.resume_conflicts(&Config {
                task: TaskKind::Shapes,
                ..Config::default()
            }),
            vec!["base_genomes".to_string()]
        );

        // A CNN does not fit the flat inputs of two spirals.
        let message = Config {
            base_genomes: vec![cnn.clone()],
            ..Config::default()
        }
        .validate()
        .expect_err("refused")
        .to_string();

        assert!(message.contains("base genome 1: "), "{message}");

        let mut over = cnn;

        over.epochs = Limits::default().max_epochs + 1;

        assert!(
            Config {
                task: TaskKind::Shapes,
                base_genomes: vec![over],
                ..Config::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn resume_conflicts_names_experiment_fields_only() {
        let base = Config::default();
        let allowed = Config {
            generations: 99,
            threads: 3,
            limits: Limits {
                max_candidate_seconds: 1.0,
                max_total_seconds: 2.0,
                max_total_flops: 3,
                ..base.limits
            },
            ..base.clone()
        };

        assert!(allowed.resume_conflicts(&base).is_empty());

        let changed = Config {
            task: TaskKind::Parity,
            master_seed: 2,
            limits: Limits {
                max_width: 8,
                ..base.limits
            },
            ..base.clone()
        };

        assert_eq!(
            changed.resume_conflicts(&base),
            vec!["limits.max_width", "master_seed", "task"]
        );
    }

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("brut-config-{name}-{}", std::process::id()));

        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");

        dir
    }

    #[test]
    fn default_is_valid() {
        assert!(Config::default().validate().is_ok());
    }

    #[test]
    fn score_step_may_be_half_the_gap_threshold() {
        let c = Config {
            hidden_score_step: 0.025,
            hidden_gap_threshold: 0.05,
            ..Config::default()
        };

        assert!(c.validate().is_ok());
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir = tmp_dir("roundtrip");
        let path = dir.join("config.json");
        let config = Config {
            task: TaskKind::Parity,
            master_seed: 77,
            ..Config::default()
        };

        config.save(&path).expect("save");

        assert_eq!(Config::load(&path).expect("load"), config);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_rejects_unknown_fields_and_bad_values() {
        let dir = tmp_dir("bad");
        let path = dir.join("config.json");
        let mut value = serde_json::to_value(Config::default()).expect("to value");

        value["surprise"] = serde_json::json!(1);
        std::fs::write(&path, value.to_string()).expect("write");
        assert!(Config::load(&path).is_err());

        let mut value = serde_json::to_value(Config::default()).expect("to value");

        value["crossover_rate"] = serde_json::json!(1.5);
        std::fs::write(&path, value.to_string()).expect("write");
        assert!(Config::load(&path).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn validate_rejects_nonsense() {
        let base = Config::default();
        let bad = [
            Config {
                population_init: 0,
                ..base.clone()
            },
            Config {
                children_per_generation: 0,
                ..base.clone()
            },
            Config {
                crossover_rate: f64::NAN,
                ..base.clone()
            },
            Config {
                crossover_rate: -0.1,
                ..base.clone()
            },
            Config {
                threads: MAX_THREADS + 1,
                ..base.clone()
            },
            Config {
                hidden_gap_threshold: f64::INFINITY,
                ..base.clone()
            },
            Config {
                hidden_score_step: 1.0,
                ..base.clone()
            },
            Config {
                hidden_score_step: 0.0,
                ..base.clone()
            },
            Config {
                hidden_score_step: 0.001,
                ..base.clone()
            },
            // Rounding down alone could exceed the gap threshold.
            Config {
                hidden_score_step: 0.1,
                hidden_gap_threshold: 0.05,
                ..base.clone()
            },
            Config {
                hidden_score_step: 0.03,
                hidden_gap_threshold: 0.05,
                ..base.clone()
            },
            Config {
                hidden_gap_threshold: 0.0,
                ..base.clone()
            },
            Config {
                limits: Limits {
                    max_epochs: 1,
                    ..base.limits
                },
                ..base.clone()
            },
            Config {
                limits: Limits {
                    max_total_seconds: 0.0,
                    ..base.limits
                },
                ..base.clone()
            },
            Config {
                limits: Limits {
                    max_candidate_seconds: f64::NAN,
                    ..base.limits
                },
                ..base.clone()
            },
            Config {
                limits: Limits {
                    max_params: 3,
                    ..base.limits
                },
                ..base.clone()
            },
            Config {
                limits: Limits {
                    max_train_flops: 10,
                    ..base.limits
                },
                ..base.clone()
            },
            Config {
                score: ScoreWeights {
                    param_penalty: -1.0,
                    flop_penalty: 0.0,
                },
                ..base.clone()
            },
        ];

        for c in bad {
            assert!(c.validate().is_err(), "{c:?}");
        }
    }

    #[test]
    fn validate_rejects_limits_above_the_hard_caps() {
        let base = Config::default();
        let limits = |l: Limits| Config {
            limits: l,
            ..base.clone()
        };
        let bad = [
            limits(Limits {
                max_hidden_layers: 200_000,
                ..base.limits
            }),
            limits(Limits {
                max_width: 1 << 20,
                ..base.limits
            }),
            limits(Limits {
                max_params: 1_000_000_000,
                ..base.limits
            }),
            limits(Limits {
                max_epochs: usize::MAX,
                ..base.limits
            }),
            Config {
                population_init: usize::MAX,
                ..base.clone()
            },
            Config {
                children_per_generation: 1 << 40,
                ..base.clone()
            },
        ];

        for c in bad {
            assert!(c.validate().is_err(), "{:?}", c.limits);
        }
    }

    #[test]
    fn random_genomes_at_the_hard_caps_are_quick() {
        let config = Config {
            limits: Limits {
                max_hidden_layers: HARD_MAX_HIDDEN_LAYERS,
                max_width: HARD_MAX_WIDTH,
                max_params: HARD_MAX_PARAMS,
                max_epochs: HARD_MAX_EPOCHS,
                max_train_flops: u64::MAX,
                ..Limits::default()
            },
            ..Config::default()
        };

        assert!(config.validate().is_ok());

        let spec = config.task.spec();
        let mut rng = rand::SeedableRng::seed_from_u64(1);
        let started = std::time::Instant::now();

        for _ in 0..50 {
            let g = Genome::random(&mut rng, &config.limits, &spec);

            assert!(g.validate(&config.limits, &spec).is_ok());
        }

        assert!(started.elapsed().as_secs_f64() < 10.0);
    }
}
