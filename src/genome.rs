//! Genome: the full, data-only description of one candidate model.
//!
//! A genome is plain data. It is never executed as code.
//! This is a deliberate safety property of the prototype.

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::config::Limits;
use crate::task::TaskSpec;
use crate::util::{BrutError, Result, log_uniform_f32, sha256_hex};
use crate::variation::repair;

/// Allowed ranges for continuous genes. Mutation and validation use them.
pub const LR_MIN: f32 = 1e-4;
pub const LR_MAX: f32 = 0.3;
pub const L2_MIN: f32 = 0.0;
pub const L2_MAX: f32 = 1e-2;
pub const INIT_SCALE_MIN: f32 = 0.25;
pub const INIT_SCALE_MAX: f32 = 2.0;
pub const BATCH_SIZES: [usize; 5] = [16, 32, 64, 128, 256];
pub const MIN_EPOCHS: usize = 5;

/// Smallest width a random genome starts with (if `max_width` allows it).
const RANDOM_MIN_WIDTH: usize = 4;

/// Every activation, for random choice.
pub const ACTIVATIONS: [Activation; 4] = [
    Activation::Relu,
    Activation::Tanh,
    Activation::Gelu,
    Activation::Sine,
];

/// Every optimizer, for random choice.
pub const OPTIMIZERS: [Optimizer; 3] = [Optimizer::Sgd, Optimizer::Momentum, Optimizer::Adam];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Activation {
    Relu,
    Tanh,
    Gelu,
    Sine,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Optimizer {
    Sgd,
    Momentum,
    Adam,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Genome {
    /// Width of each hidden layer, input to output order. May be empty (linear model).
    pub hidden: Vec<usize>,
    pub activation: Activation,
    pub optimizer: Optimizer,
    pub learning_rate: f32,
    /// L2 weight decay factor.
    pub l2: f32,
    pub batch_size: usize,
    pub epochs: usize,
    /// Multiplier on the standard (He or Xavier) init scale.
    pub init_scale: f32,
    /// Seed for weight init and batch shuffling.
    pub seed: u64,
}

impl Genome {
    /// A random genome that passes `validate` for these limits and task.
    ///
    /// Requires feasible limits (see `Config::validate`): a linear model with
    /// `MIN_EPOCHS` epochs must fit `max_params` and `max_train_flops`.
    pub fn random(rng: &mut ChaCha8Rng, limits: &Limits, spec: &TaskSpec) -> Genome {
        let max_width = limits.max_width;
        let max_layers = if max_width == 0 {
            0
        } else {
            limits.max_hidden_layers
        };

        let n_layers = if max_layers == 0 {
            0
        } else {
            rng.random_range(1..=max_layers)
        };
        let min_width = RANDOM_MIN_WIDTH.min(max_width).max(1);
        let hidden: Vec<usize> = (0..n_layers)
            .map(|_| rng.random_range(min_width..=max_width))
            .collect();

        let activation = ACTIVATIONS[rng.random_range(0..ACTIVATIONS.len())];
        let optimizer = OPTIMIZERS[rng.random_range(0..OPTIMIZERS.len())];
        let learning_rate = log_uniform_f32(rng, LR_MIN, LR_MAX);

        let l2 = if rng.random_bool(0.5) {
            0.0
        } else {
            log_uniform_f32(rng, 1e-6, L2_MAX)
        };

        let batch_size = BATCH_SIZES[rng.random_range(0..BATCH_SIZES.len())];
        let max_epochs = limits.max_epochs.max(MIN_EPOCHS);
        let epochs = log_uniform_f32(rng, MIN_EPOCHS as f32, max_epochs as f32).round() as usize;
        let init_scale = log_uniform_f32(rng, 0.5, 1.5);
        let seed = rng.random();

        let g = Genome {
            hidden,
            activation,
            optimizer,
            learning_rate,
            l2,
            batch_size,
            epochs,
            init_scale,
            seed,
        };

        repair(g, limits, spec)
    }

    /// Layer sizes including input and output: [n_in, hidden.., n_out].
    pub fn layer_sizes(&self, spec: &TaskSpec) -> Vec<usize> {
        let mut sizes = Vec::with_capacity(self.hidden.len() + 2);

        sizes.push(spec.n_in);
        sizes.extend_from_slice(&self.hidden);
        sizes.push(spec.n_out);

        sizes
    }

    /// Trainable parameters (weights + biases) of the resulting MLP.
    pub fn param_count(&self, spec: &TaskSpec) -> usize {
        let sizes = self.layer_sizes(spec);

        sizes
            .windows(2)
            .map(|w| w[0].saturating_mul(w[1]).saturating_add(w[1]))
            .fold(0usize, |acc, p| acc.saturating_add(p))
    }

    /// Estimated training FLOPs: 6 * params * n_train * epochs
    /// (forward ~2, backward ~4 FLOPs per parameter per sample).
    pub fn train_flops(&self, spec: &TaskSpec) -> u64 {
        6u64.saturating_mul(self.param_count(spec) as u64)
            .saturating_mul(spec.n_train as u64)
            .saturating_mul(self.epochs as u64)
    }

    /// Check every gene against the constants above and `limits`.
    pub fn validate(&self, limits: &Limits, spec: &TaskSpec) -> Result<()> {
        let fail = |m: String| Err(BrutError::Invalid(format!("genome: {m}")));

        if spec.n_in == 0 || spec.n_out == 0 {
            return fail("task has zero inputs or outputs".to_string());
        }

        if self.hidden.len() > limits.max_hidden_layers {
            return fail(format!(
                "{} hidden layers > max {}",
                self.hidden.len(),
                limits.max_hidden_layers
            ));
        }

        if let Some(w) = self
            .hidden
            .iter()
            .find(|&&w| w == 0 || w > limits.max_width)
        {
            return fail(format!("layer width {w} outside 1..={}", limits.max_width));
        }

        if !(LR_MIN..=LR_MAX).contains(&self.learning_rate) {
            return fail(format!(
                "learning rate {} outside [{LR_MIN}, {LR_MAX}]",
                self.learning_rate
            ));
        }

        if !(L2_MIN..=L2_MAX).contains(&self.l2) {
            return fail(format!("l2 {} outside [{L2_MIN}, {L2_MAX}]", self.l2));
        }

        if !(INIT_SCALE_MIN..=INIT_SCALE_MAX).contains(&self.init_scale) {
            return fail(format!(
                "init scale {} outside [{INIT_SCALE_MIN}, {INIT_SCALE_MAX}]",
                self.init_scale
            ));
        }

        if !BATCH_SIZES.contains(&self.batch_size) {
            return fail(format!(
                "batch size {} not in {BATCH_SIZES:?}",
                self.batch_size
            ));
        }

        if self.epochs < MIN_EPOCHS || self.epochs > limits.max_epochs {
            return fail(format!(
                "epochs {} outside {MIN_EPOCHS}..={}",
                self.epochs, limits.max_epochs
            ));
        }

        let params = self.param_count(spec);

        if params > limits.max_params {
            return fail(format!("{params} params > max {}", limits.max_params));
        }

        let flops = self.train_flops(spec);

        if flops > limits.max_train_flops {
            return fail(format!(
                "{flops} train FLOPs > max {}",
                limits.max_train_flops
            ));
        }

        Ok(())
    }

    /// Stable content id: first 16 hex chars of SHA-256 over canonical JSON.
    pub fn id(&self) -> String {
        // Serializing this plain struct cannot fail; the Debug fallback only keeps the code panic-free.
        let bytes = match serde_json::to_vec(self) {
            Ok(b) => b,
            Err(_) => format!("{self:?}").into_bytes(),
        };

        let mut hex = sha256_hex(&bytes);

        hex.truncate(16);

        hex
    }

    /// One-line human summary, e.g. for `brut lineage`.
    pub fn summary(&self) -> String {
        format!(
            "hidden={:?} act={:?} opt={:?} lr={:.2e} l2={:.1e} bs={} ep={} init={:.2}",
            self.hidden,
            self.activation,
            self.optimizer,
            self.learning_rate,
            self.l2,
            self.batch_size,
            self.epochs,
            self.init_scale
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::TaskKind;
    use rand::SeedableRng;

    fn sample() -> Genome {
        Genome {
            hidden: vec![8, 4],
            activation: Activation::Tanh,
            optimizer: Optimizer::Adam,
            learning_rate: 0.01,
            l2: 0.0,
            batch_size: 32,
            epochs: 10,
            init_scale: 1.0,
            seed: 9,
        }
    }

    #[test]
    fn sizes_params_flops() {
        let spec = TaskKind::TwoSpirals.spec();
        let g = sample();

        assert_eq!(g.layer_sizes(&spec), vec![2, 8, 4, 2]);

        // (2*8+8) + (8*4+4) + (4*2+2) = 24 + 36 + 10
        assert_eq!(g.param_count(&spec), 70);
        assert_eq!(g.train_flops(&spec), 6 * 70 * spec.n_train as u64 * 10);
    }

    #[test]
    fn linear_model_is_allowed() {
        let spec = TaskKind::TwoSpirals.spec();
        let mut g = sample();

        g.hidden.clear();

        assert_eq!(g.param_count(&spec), 6);
        assert!(g.validate(&Limits::default(), &spec).is_ok());
    }

    #[test]
    fn validate_rejects_each_bad_gene() {
        let spec = TaskKind::TwoSpirals.spec();
        let limits = Limits::default();

        assert!(sample().validate(&limits, &spec).is_ok());

        let mut bad: Vec<Genome> = Vec::new();
        let mut g = sample();

        g.learning_rate = f32::NAN;
        bad.push(g);

        let mut g = sample();
        g.learning_rate = 1.0;
        bad.push(g);

        let mut g = sample();
        g.l2 = -1.0;
        bad.push(g);

        let mut g = sample();
        g.init_scale = 5.0;
        bad.push(g);

        let mut g = sample();
        g.batch_size = 17;
        bad.push(g);

        let mut g = sample();
        g.epochs = 1;
        bad.push(g);

        let mut g = sample();
        g.epochs = limits.max_epochs + 1;
        bad.push(g);

        let mut g = sample();
        g.hidden = vec![0];
        bad.push(g);

        let mut g = sample();
        g.hidden = vec![limits.max_width + 1];
        bad.push(g);

        let mut g = sample();
        g.hidden = vec![4; limits.max_hidden_layers + 1];
        bad.push(g);

        for g in bad {
            assert!(g.validate(&limits, &spec).is_err(), "{g:?}");
        }
    }

    #[test]
    fn validate_checks_params_and_flops() {
        let spec = TaskKind::TwoSpirals.spec();
        let g = sample();
        let limits = Limits {
            max_params: 69,
            ..Limits::default()
        };

        assert!(g.validate(&limits, &spec).is_err());

        let limits = Limits {
            max_train_flops: g.train_flops(&spec) - 1,
            ..Limits::default()
        };

        assert!(g.validate(&limits, &spec).is_err());
    }

    #[test]
    fn random_genomes_are_valid_and_deterministic() {
        let limits = Limits::default();

        for kind in [
            TaskKind::TwoSpirals,
            TaskKind::Parity,
            TaskKind::SineRegression,
        ] {
            let spec = kind.spec();
            let mut rng = ChaCha8Rng::seed_from_u64(1);
            let mut rng2 = ChaCha8Rng::seed_from_u64(1);

            for _ in 0..200 {
                let g = Genome::random(&mut rng, &limits, &spec);

                assert!(g.validate(&limits, &spec).is_ok(), "{g:?}");
                assert_eq!(g, Genome::random(&mut rng2, &limits, &spec));
            }
        }
    }

    #[test]
    fn random_respects_tight_limits() {
        let spec = TaskKind::TwoSpirals.spec();
        let limits = Limits {
            max_hidden_layers: 1,
            max_width: 3,
            max_params: 50,
            max_epochs: 6,
            max_train_flops: 6 * 50 * spec.n_train as u64 * 6,
            ..Limits::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(2);

        for _ in 0..100 {
            let g = Genome::random(&mut rng, &limits, &spec);

            assert!(g.validate(&limits, &spec).is_ok(), "{g:?}");
        }
    }

    #[test]
    fn id_is_stable_and_content_sensitive() {
        let a = sample();
        let mut b = sample();

        assert_eq!(a.id(), sample().id());
        assert_eq!(a.id().len(), 16);

        b.seed = 10;
        assert_ne!(a.id(), b.id());
    }
}
