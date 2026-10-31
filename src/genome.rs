//! Genome: the full, data-only description of one candidate model.
//!
//! A genome is plain data. It is never executed as code.
//! This is a deliberate safety property of the prototype.

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::cnn::genome::CnnGenome;
use crate::cnn::shape::{Plan, TIME_BUDGET_FLOPS_PER_SECOND};
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

/// The kind of network a genome describes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Arch {
    /// A dense network. `Genome::hidden` gives its hidden layers.
    #[default]
    Mlp,
    /// A convolutional network: a stem, stages and a head. `Genome::hidden` gives the
    /// dense layers of its head.
    Cnn(CnnGenome),
}

impl Arch {
    /// True for `Mlp`, the default. A genome leaves an MLP arch out of its JSON, so MLP
    /// genomes keep their bytes and the genomes written before CNNs still parse.
    pub fn is_mlp(&self) -> bool {
        matches!(self, Arch::Mlp)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Genome {
    /// The kind of network: an MLP, or a CNN with its stem, stages and head.
    #[serde(default, skip_serializing_if = "Arch::is_mlp")]
    pub arch: Arch,
    /// Width of each hidden layer, input to output order. May be empty (linear model).
    /// For a CNN: the dense layers of its head, between the pooling and the output layer.
    pub hidden: Vec<usize>,
    /// Activation of the hidden layers; for a CNN, of the dense layers of its head.
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
    /// A random MLP genome that passes `validate` for these limits and task.
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
            arch: Arch::Mlp,
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

    /// Layer sizes of an MLP including input and output: [n_in, hidden.., n_out].
    ///
    /// An error for a CNN genome: its `hidden` holds only the dense layers of its head, so
    /// these sizes would describe the wrong network.
    pub fn layer_sizes(&self, spec: &TaskSpec) -> Result<Vec<usize>> {
        match self.arch {
            Arch::Mlp => Ok(self.mlp_sizes(spec)),
            Arch::Cnn(_) => Err(BrutError::Invalid(
                "genome: layer sizes describe an MLP, and this genome is a CNN".to_string(),
            )),
        }
    }

    /// [n_in, hidden.., n_out], read as the layers of an MLP.
    fn mlp_sizes(&self, spec: &TaskSpec) -> Vec<usize> {
        let mut sizes = Vec::with_capacity(self.hidden.len() + 2);

        sizes.push(spec.n_in);
        sizes.extend_from_slice(&self.hidden);
        sizes.push(spec.n_out);

        sizes
    }

    /// The layers of a CNN genome on this task, with their shapes and costs
    /// (`CnnGenome::plan`). An error for an MLP genome, or when a shape rule breaks.
    pub fn cnn_plan(&self, spec: &TaskSpec) -> Result<Plan> {
        match &self.arch {
            Arch::Mlp => Err(BrutError::Invalid(
                "genome: an MLP genome has no CNN plan".to_string(),
            )),
            Arch::Cnn(cnn) => cnn.plan(&self.hidden, self.activation, spec),
        }
    }

    /// Trainable parameters of the network: weights and biases, and for a CNN also the
    /// scales and shifts of its normalizations.
    ///
    /// For a CNN whose shapes do not fit the task the count is `usize::MAX`, so every
    /// limit rejects it; `validate` says which rule it breaks.
    pub fn param_count(&self, spec: &TaskSpec) -> usize {
        match self.arch {
            Arch::Mlp => self
                .mlp_sizes(spec)
                .windows(2)
                .map(|w| w[0].saturating_mul(w[1]).saturating_add(w[1]))
                .fold(0usize, |acc, p| acc.saturating_add(p)),
            Arch::Cnn(_) => self.cnn_plan(spec).map_or(usize::MAX, |plan| plan.params),
        }
    }

    /// Estimated training FLOPs for `epochs` passes over the `n_train` samples.
    ///
    /// MLP: 6 * params * n_train * epochs (forward ~2, backward ~4 FLOPs per parameter per
    /// sample). CNN: 3 * forward FLOPs per sample * n_train * epochs, with the forward FLOPs
    /// counted layer by layer (`cnn::shape`); for a dense layer that is the MLP rule.
    /// `u64::MAX` for a CNN whose shapes do not fit the task.
    pub fn train_flops(&self, spec: &TaskSpec) -> u64 {
        let per_sample = match self.arch {
            Arch::Mlp => 6u64.saturating_mul(self.param_count(spec) as u64),
            Arch::Cnn(_) => match self.cnn_plan(spec) {
                Ok(plan) => plan.train_flops_per_sample(),
                Err(_) => return u64::MAX,
            },
        };

        per_sample
            .saturating_mul(spec.n_train as u64)
            .saturating_mul(self.epochs as u64)
    }

    /// Check every gene against the constants above and `limits`.
    ///
    /// The training genes and the hidden layers follow the same rules for both kinds. A
    /// CNN must also pass the shape check (`CnnGenome::plan`). `max_params` and
    /// `max_train_flops` hold for both kinds. A CNN's estimated training time must also
    /// stay within `max_train_flops / TIME_BUDGET_FLOPS_PER_SECOND` seconds (`cnn::shape`).
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

        let samples = (spec.n_train as f64) * (self.epochs as f64);
        let (params, flops, seconds) = match &self.arch {
            Arch::Mlp => (self.param_count(spec), self.train_flops(spec), 0.0),
            Arch::Cnn(cnn) => match cnn.plan(&self.hidden, self.activation, spec) {
                Ok(plan) => (
                    plan.params,
                    plan.train_flops_per_sample()
                        .saturating_mul(spec.n_train as u64)
                        .saturating_mul(self.epochs as u64),
                    plan.train_nanos_per_sample() * samples / 1e9,
                ),
                Err(BrutError::Invalid(m)) => return fail(m),
                Err(e) => return Err(e),
            },
        };

        if params > limits.max_params {
            return fail(format!("{params} params > max {}", limits.max_params));
        }

        if flops > limits.max_train_flops {
            return fail(format!(
                "{flops} train FLOPs > max {}",
                limits.max_train_flops
            ));
        }

        // A CNN must also fit the time that its FLOP budget buys, so that the FLOP limit
        // ends its training, not the clock (`cnn::shape`, "Time").
        let budget = limits.max_train_flops as f64 / TIME_BUDGET_FLOPS_PER_SECOND;

        if seconds > budget {
            return fail(format!(
                "estimated training time {seconds:.1} s > {budget:.1} s, the time \
                 max_train_flops buys a CNN"
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

    /// One-line human summary, e.g. for `brut lineage`. A CNN's starts with its
    /// architecture (`CnnGenome::summary`); its `hidden` are the dense layers of its head.
    pub fn summary(&self) -> String {
        let text = format!(
            "hidden={:?} act={:?} opt={:?} lr={:.2e} l2={:.1e} bs={} ep={} init={:.2}",
            self.hidden,
            self.activation,
            self.optimizer,
            self.learning_rate,
            self.l2,
            self.batch_size,
            self.epochs,
            self.init_scale
        );

        match &self.arch {
            Arch::Mlp => text,
            Arch::Cnn(cnn) => format!("{} {text}", cnn.summary()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnn::genome::testing;
    use crate::task::TaskKind;
    use rand::SeedableRng;

    fn sample() -> Genome {
        Genome {
            arch: Arch::Mlp,
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

        assert_eq!(g.layer_sizes(&spec).expect("an MLP"), vec![2, 8, 4, 2]);

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
    fn an_mlp_genome_leaves_its_arch_out_of_the_json() {
        let g = sample();
        let text = serde_json::to_string(&g).expect("json");

        assert_eq!(
            text,
            concat!(
                r#"{"hidden":[8,4],"activation":"Tanh","optimizer":"Adam","learning_rate":0.01,"#,
                r#""l2":0.0,"batch_size":32,"epochs":10,"init_scale":1.0,"seed":9}"#
            )
        );

        let parsed: Genome = serde_json::from_str(&text).expect("parse");

        assert_eq!(parsed, g);
        assert!(parsed.arch.is_mlp());
    }

    #[test]
    fn a_cnn_genome_keeps_its_json_and_its_id() {
        let g = testing::lenet();
        let text = serde_json::to_string(&g).expect("json");

        assert_eq!(
            text,
            concat!(
                r#"{"arch":{"Cnn":{"dims":"TwoD","#,
                r#""stem":{"width":6,"kernel":5,"stride":2,"down":"AvgPool","#,
                r#""norm":"None","activation":"Tanh"},"#,
                r#""stages":[{"block":"Plain","width":16,"kernel":5,"stride":2,"down":"AvgPool","#,
                r#""dilation":1,"groups":1,"se":false,"residual":false,"#,
                r#""norm":"None","activation":"Tanh"}],"#,
                r#""head":"Flatten"}},"#,
                r#""hidden":[32],"activation":"Tanh","optimizer":"Adam","learning_rate":0.01,"#,
                r#""l2":0.0,"batch_size":32,"epochs":10,"init_scale":1.0,"seed":7}"#
            )
        );

        let parsed: Genome = serde_json::from_str(&text).expect("parse");

        assert_eq!(parsed, g);
        assert_eq!(serde_json::to_string(&parsed).expect("json"), text);
        assert_eq!(g.id(), "cbff19d9f32271bf");
        assert_eq!(g.id(), sha256_hex(text.as_bytes())[..16]);
    }

    #[test]
    fn a_cnn_genome_with_an_unknown_field_does_not_parse() {
        let text = serde_json::to_string(&testing::lenet()).expect("json");
        let typo = text.replacen("\"kernel\":5", "\"kernel\":5,\"kernal\":5", 1);

        assert_ne!(typo, text);
        assert!(serde_json::from_str::<Genome>(&typo).is_err());
    }

    #[test]
    fn mlp_only_functions_refuse_a_cnn() {
        let spec = TaskKind::Shapes.spec();
        let cnn = testing::lenet();
        let mlp = sample();

        let message = cnn.layer_sizes(&spec).expect_err("a CNN has no MLP sizes");

        assert!(
            message.to_string().contains("this genome is a CNN"),
            "{message}"
        );
        assert!(mlp.cnn_plan(&spec).is_err());
        assert_eq!(mlp.layer_sizes(&spec).expect("an MLP"), vec![256, 8, 4, 4]);

        // The shared counts follow the kind: the CNN's come from its plan, not from
        // reading its head as an MLP.
        let plan = cnn.cnn_plan(&spec).expect("a plan");
        let mut head_as_mlp = cnn.clone();

        head_as_mlp.arch = Arch::Mlp;

        assert_eq!(cnn.param_count(&spec), plan.params);
        assert_ne!(cnn.param_count(&spec), head_as_mlp.param_count(&spec));
        assert_eq!(
            cnn.train_flops(&spec),
            plan.train_flops_per_sample() * spec.n_train as u64 * cnn.epochs as u64
        );
    }

    #[test]
    fn summaries_name_the_architecture() {
        assert_eq!(
            sample().summary(),
            "hidden=[8, 4] act=Tanh opt=Adam lr=1.00e-2 l2=0.0e0 bs=32 ep=10 init=1.00"
        );
        assert_eq!(
            testing::lenet().summary(),
            "cnn2d stem(w6 k5 s2avg tanh) plain(w16 k5 s2avg tanh) head=flatten \
             hidden=[32] act=Tanh opt=Adam lr=1.00e-2 l2=0.0e0 bs=32 ep=10 init=1.00"
        );
        assert_eq!(
            testing::grouped_dilated().summary(),
            "cnn2d stem(w8 k3 relu) plain(w16 k3 s2conv d2 g4 gn silu) \
             plain(w16 k3 d3 g16 ln relu6) head=gap \
             hidden=[8] act=Gelu opt=Adam lr=1.00e-2 l2=0.0e0 bs=32 ep=10 init=1.00"
        );
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
