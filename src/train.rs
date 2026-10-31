//! Training of one candidate under hard budgets.
//!
//! An MLP trains with the hand-written backprop of `nn`, a CNN through candle
//! (`cnn::train`). Both run the same loop (`run_epochs`): mini-batches in an order shuffled
//! from the genome's seed each epoch, the same cosine decay of the learning rate, the same
//! FLOP and time budgets, kill switch and divergence rules, and the same optimizers.

use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::cnn::train::CnnLearner;
use crate::config::Limits;
use crate::genome::{Arch, Genome};
use crate::model::Model;
use crate::nn::{Gradients, Mlp, OptimizerState, TargetsRef, Workspace};
use crate::task::{Dataset, Objective, Targets, TaskSpec};
use crate::util::{BrutError, Result, derive_seed};

/// Final learning rate as a fraction of the start value (cosine decay).
pub const LR_FLOOR: f32 = 0.05;

/// Check the wall clock every this many optimizer steps.
const TIME_CHECK_STEPS: u64 = 32;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainStats {
    pub epochs_done: usize,
    pub steps: u64,
    /// Estimated FLOPs actually spent: the training FLOPs per sample times the samples
    /// seen. For an MLP 6 * params per sample, for a CNN 3 times the forward FLOPs of its
    /// plan (`cnn::shape`).
    pub flops: u64,
    pub wall_seconds: f64,
    /// Always finite: f32::MAX when training diverged (JSON cannot store NaN).
    pub final_train_loss: f32,
    /// Loss or weights became NaN/inf. Training stopped early.
    pub diverged: bool,
    /// Stopped by `max_candidate_seconds` or `max_train_flops`.
    pub stopped_by_budget: bool,
    /// Stopped early because the kill switch (STOP file) was set.
    #[serde(default)]
    pub stopped_by_kill_switch: bool,
}

/// Train the network described by `genome` on `data` (the Train split).
///
/// Errors if the genome is invalid for `limits` (call `Genome::validate` first).
/// Mini-batch training with per-epoch shuffling from `genome.seed`.
/// Checks the wall-clock and FLOP budgets at least once per epoch and stops cleanly.
/// Deterministic for equal inputs (wall-clock stops aside). A CNN is deterministic on one
/// machine at any thread count; another CPU can give other bits (see `cnn::train`).
///
/// The learning rate follows a cosine decay from `learning_rate` down to `LR_FLOOR` times it.
pub fn train(
    genome: &Genome,
    spec: &TaskSpec,
    data: &Dataset,
    limits: &Limits,
) -> Result<(Model, TrainStats)> {
    train_with_stop(genome, spec, data, limits, &|| false)
}

/// Like `train`, but also asks `stop` at the start of each epoch and every
/// `TIME_CHECK_STEPS` steps. When it answers true, training ends cleanly with
/// `stopped_by_kill_switch` set. `stop` should be cheap (see `safety::StopSignal`).
pub fn train_with_stop(
    genome: &Genome,
    spec: &TaskSpec,
    data: &Dataset,
    limits: &Limits,
    stop: &dyn Fn() -> bool,
) -> Result<(Model, TrainStats)> {
    train_inner(genome, spec, data, limits, stop, None)
}

/// Like `train`, and also returns the loss of every step, for tests.
#[cfg(test)]
pub(crate) fn train_logged(
    genome: &Genome,
    spec: &TaskSpec,
    data: &Dataset,
    limits: &Limits,
) -> Result<(Model, TrainStats, Vec<f32>)> {
    let mut losses = Vec::new();
    let (model, stats) = train_inner(genome, spec, data, limits, &|| false, Some(&mut losses))?;

    Ok((model, stats, losses))
}

/// `train_with_stop`; pushes the loss of every step to `losses` when given.
fn train_inner(
    genome: &Genome,
    spec: &TaskSpec,
    data: &Dataset,
    limits: &Limits,
    stop: &dyn Fn() -> bool,
    losses: Option<&mut Vec<f32>>,
) -> Result<(Model, TrainStats)> {
    genome.validate(limits, spec)?;
    check_data(spec, data)?;

    let started = Instant::now();

    match &genome.arch {
        Arch::Mlp => {
            let mut learner = MlpLearner::new(genome, spec, data)?;
            let stats = run_epochs(genome, data.n, limits, stop, started, &mut learner, losses)?;

            Ok((Model::Mlp(learner.model), stats))
        }
        Arch::Cnn(cnn) => {
            let mut learner = CnnLearner::new(genome, cnn, spec, data)?;
            let stats = run_epochs(genome, data.n, limits, stop, started, &mut learner, losses)?;

            Ok((Model::Cnn(learner.into_model()?), stats))
        }
    }
}

/// A network under training, as the loop of `run_epochs` drives it.
pub(crate) trait Learner {
    /// Training FLOPs per sample (forward and backward pass), for the budget.
    fn flops_per_sample(&self) -> u64;

    /// The mean loss of the train rows `rows` (one mini-batch) and its gradients, which
    /// the learner keeps for `update`. A non-finite loss means the training diverged.
    fn loss_and_grad(&mut self, rows: &[usize]) -> Result<f32>;

    /// One optimizer step with learning rate `lr` and weight decay `l2`, from the
    /// gradients of the last `loss_and_grad`.
    fn update(&mut self, lr: f32, l2: f32) -> Result<()>;

    /// True if every trained value is finite.
    fn is_finite(&self) -> Result<bool>;
}

/// The training loop: `genome.epochs` epochs of mini-batches over the `n` train rows.
///
/// Each epoch first checks the budgets and the kill switch, then shuffles the rows with
/// the stream `derive_seed(genome.seed, "shuffle")`. A non-finite loss, or a non-finite
/// value after an epoch, ends the training as diverged.
fn run_epochs<L: Learner>(
    genome: &Genome,
    n: usize,
    limits: &Limits,
    stop: &dyn Fn() -> bool,
    started: Instant,
    learner: &mut L,
    mut losses: Option<&mut Vec<f32>>,
) -> Result<TrainStats> {
    let mut shuffle_rng = ChaCha8Rng::seed_from_u64(derive_seed(genome.seed, "shuffle"));
    let batch = genome.batch_size.min(n).max(1);
    let flops_per_sample = learner.flops_per_sample();
    let flops_per_epoch = flops_per_sample.saturating_mul(n as u64);
    let total_steps = (genome.epochs * n.div_ceil(batch)).max(1) as f32;
    let mut order: Vec<usize> = (0..n).collect();

    let mut stats = TrainStats {
        epochs_done: 0,
        steps: 0,
        flops: 0,
        wall_seconds: 0.0,
        final_train_loss: f32::MAX,
        diverged: false,
        stopped_by_budget: false,
        stopped_by_kill_switch: false,
    };

    'epochs: for _ in 0..genome.epochs {
        if out_of_budget(&stats, flops_per_epoch, limits, &started) {
            stats.stopped_by_budget = true;
            break;
        }

        if stop() {
            stats.stopped_by_kill_switch = true;
            break;
        }

        shuffle(&mut order, &mut shuffle_rng);

        let mut epoch_loss = 0.0f64;

        for chunk in order.chunks(batch) {
            let m = chunk.len();
            let loss = learner.loss_and_grad(chunk)?;

            if let Some(losses) = losses.as_deref_mut() {
                losses.push(loss);
            }

            if !loss.is_finite() {
                stats.diverged = true;
                break 'epochs;
            }

            let progress = stats.steps as f32 / total_steps;
            let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            let lr = genome.learning_rate * (LR_FLOOR + (1.0 - LR_FLOOR) * cosine);

            learner.update(lr, genome.l2)?;

            stats.steps += 1;
            stats.flops = stats
                .flops
                .saturating_add(flops_per_sample.saturating_mul(m as u64));
            epoch_loss += loss as f64 * m as f64;

            if stats.steps.is_multiple_of(TIME_CHECK_STEPS) {
                if started.elapsed().as_secs_f64() > limits.max_candidate_seconds {
                    stats.stopped_by_budget = true;
                    break 'epochs;
                }

                if stop() {
                    stats.stopped_by_kill_switch = true;
                    break 'epochs;
                }
            }
        }

        if !learner.is_finite()? {
            stats.diverged = true;
            break;
        }

        stats.epochs_done += 1;
        stats.final_train_loss = finite_or_max((epoch_loss / n as f64) as f32);
    }

    if stats.diverged {
        stats.final_train_loss = f32::MAX;
    }

    stats.wall_seconds = started.elapsed().as_secs_f64();

    Ok(stats)
}

/// An MLP under training: the model, its optimizer and the reusable buffers.
struct MlpLearner<'a> {
    model: Mlp,
    opt: OptimizerState,
    data: &'a Dataset,
    objective: Objective,
    ws: Workspace,
    grads: Gradients,
    xb: Vec<f32>,
    lb: Vec<u32>,
    vb: Vec<f32>,
}

impl<'a> MlpLearner<'a> {
    /// The MLP of `genome` on `spec`, with its initial weights drawn from the stream
    /// `derive_seed(genome.seed, "init")`.
    fn new(genome: &Genome, spec: &TaskSpec, data: &'a Dataset) -> Result<MlpLearner<'a>> {
        let sizes = genome.layer_sizes(spec)?;
        let mut init_rng = ChaCha8Rng::seed_from_u64(derive_seed(genome.seed, "init"));
        let model = Mlp::new(&sizes, genome.activation, genome.init_scale, &mut init_rng);
        let opt = OptimizerState::new(genome.optimizer, &model);
        let batch = genome.batch_size.min(data.n).max(1);
        let grads = Gradients::zeros_like(&model);

        Ok(MlpLearner {
            model,
            opt,
            data,
            objective: spec.objective,
            ws: Workspace::new(),
            grads,
            xb: vec![0.0f32; batch * data.n_in],
            lb: vec![0u32; batch],
            vb: vec![0.0f32; batch * data.n_out],
        })
    }
}

impl Learner for MlpLearner<'_> {
    fn flops_per_sample(&self) -> u64 {
        6u64.saturating_mul(self.model.param_count() as u64)
    }

    fn loss_and_grad(&mut self, rows: &[usize]) -> Result<f32> {
        let m = rows.len();
        let n_in = self.data.n_in;
        let n_out = self.data.n_out;

        // Gather the batch into the reusable buffers.
        for (row, &i) in rows.iter().enumerate() {
            self.xb[row * n_in..(row + 1) * n_in]
                .copy_from_slice(&self.data.x[i * n_in..(i + 1) * n_in]);
        }

        let targets = match &self.data.targets {
            Targets::Labels(labels) => {
                for (row, &i) in rows.iter().enumerate() {
                    self.lb[row] = labels[i];
                }

                TargetsRef::Labels(&self.lb[..m])
            }
            Targets::Values(values) => {
                for (row, &i) in rows.iter().enumerate() {
                    self.vb[row * n_out..(row + 1) * n_out]
                        .copy_from_slice(&values[i * n_out..(i + 1) * n_out]);
                }

                TargetsRef::Values(&self.vb[..m * n_out])
            }
        };

        Ok(self.model.loss_and_grad_into(
            &self.xb[..m * n_in],
            targets,
            m,
            self.objective,
            &mut self.ws,
            &mut self.grads,
        ))
    }

    fn update(&mut self, lr: f32, l2: f32) -> Result<()> {
        self.opt.step(&mut self.model, &self.grads, lr, l2);

        Ok(())
    }

    fn is_finite(&self) -> Result<bool> {
        Ok(self.model.is_finite())
    }
}

/// True if one more epoch would break the FLOP or time limit.
fn out_of_budget(
    stats: &TrainStats,
    flops_per_epoch: u64,
    limits: &Limits,
    started: &Instant,
) -> bool {
    let flops_next = stats.flops.saturating_add(flops_per_epoch);
    let too_many_flops = flops_next > limits.max_train_flops;
    let too_slow = started.elapsed().as_secs_f64() > limits.max_candidate_seconds;

    too_many_flops || too_slow
}

/// Fisher-Yates shuffle.
fn shuffle(order: &mut [usize], rng: &mut ChaCha8Rng) {
    for i in (1..order.len()).rev() {
        let j = rng.random_range(0..=i);

        order.swap(i, j);
    }
}

fn finite_or_max(v: f32) -> f32 {
    if v.is_finite() { v } else { f32::MAX }
}

/// Reject data that does not fit the task spec.
fn check_data(spec: &TaskSpec, data: &Dataset) -> Result<()> {
    let fail = |m: &str| Err(BrutError::Invalid(format!("train data: {m}")));

    if data.n == 0 {
        return fail("empty dataset");
    }

    if data.n_in != spec.n_in || data.n_out != spec.n_out {
        return fail("sizes do not match the task");
    }

    if data.x.len() != data.n * data.n_in {
        return fail("input length does not match n * n_in");
    }

    if !data.x.iter().all(|v| v.is_finite()) {
        return fail("non-finite input");
    }

    match (&data.targets, spec.objective) {
        (Targets::Labels(labels), Objective::Classification { classes }) => {
            if classes != spec.n_out || labels.len() != data.n {
                return fail("label count or class count wrong");
            }

            if labels.iter().any(|&c| c as usize >= classes) {
                return fail("label out of range");
            }
        }
        (Targets::Values(values), Objective::Regression) => {
            if values.len() != data.n * data.n_out {
                return fail("target length does not match n * n_out");
            }

            if !values.iter().all(|v| v.is_finite()) {
                return fail("non-finite target");
            }
        }
        _ => return fail("targets do not match the objective"),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{Activation, Arch, Optimizer};
    use crate::task::{Split, TaskKind, generate};

    fn genome() -> Genome {
        Genome {
            arch: Arch::Mlp,
            hidden: vec![16],
            activation: Activation::Tanh,
            optimizer: Optimizer::Adam,
            learning_rate: 0.02,
            l2: 0.0,
            batch_size: 32,
            epochs: 30,
            init_scale: 1.0,
            seed: 3,
        }
    }

    /// The MLP inside `model`.
    fn mlp(model: Model) -> Mlp {
        match model {
            Model::Mlp(mlp) => mlp,
            Model::Cnn(_) => panic!("expected an MLP"),
        }
    }

    /// Accuracy computed by hand, so this test does not depend on `score`.
    fn accuracy(model: &Mlp, data: &Dataset) -> f64 {
        let out = model.forward(&data.x, data.n);
        let Targets::Labels(labels) = &data.targets else {
            return 0.0;
        };
        let correct = out
            .chunks_exact(data.n_out)
            .zip(labels)
            .filter(|(row, l)| {
                let best = if row[1] > row[0] { 1 } else { 0 };

                best == **l
            })
            .count();

        correct as f64 / data.n as f64
    }

    #[test]
    fn training_learns_parity() {
        let spec = TaskKind::Parity.spec();
        let data = generate(&spec, Split::Train, 1);
        let mut g = genome();

        g.hidden = vec![32, 32];
        g.epochs = 150;
        g.learning_rate = 0.01;

        let (model, stats) = train(&g, &spec, &data, &Limits::default()).expect("valid genome");

        assert!(!stats.diverged);
        assert!(!stats.stopped_by_budget);
        assert_eq!(stats.epochs_done, 150);
        assert!(stats.final_train_loss < 0.3, "{}", stats.final_train_loss);
        assert!(accuracy(&mlp(model), &data) > 0.9);
    }

    #[test]
    fn training_is_deterministic() {
        let spec = TaskKind::TwoSpirals.spec();
        let data = generate(&spec, Split::Train, 1);
        let (a, sa) = train(&genome(), &spec, &data, &Limits::default()).expect("ok");
        let (b, sb) = train(&genome(), &spec, &data, &Limits::default()).expect("ok");

        assert_eq!(a, b);
        assert_eq!(sa.final_train_loss, sb.final_train_loss);
        assert_eq!(sa.flops, sb.flops);
    }

    #[test]
    fn flops_match_estimate() {
        let spec = TaskKind::TwoSpirals.spec();
        let data = generate(&spec, Split::Train, 1);
        let g = genome();
        let (_, stats) = train(&g, &spec, &data, &Limits::default()).expect("ok");

        assert_eq!(stats.flops, g.train_flops(&spec));
        assert_eq!(stats.steps, (30 * spec.n_train.div_ceil(32)) as u64);
    }

    #[test]
    fn invalid_genome_is_an_error() {
        let spec = TaskKind::TwoSpirals.spec();
        let data = generate(&spec, Split::Train, 1);
        let mut g = genome();

        g.batch_size = 3;

        assert!(train(&g, &spec, &data, &Limits::default()).is_err());
    }

    #[test]
    fn a_cnn_genome_trains_as_a_cnn() {
        // Its `hidden` holds only the dense layers of its head, so an MLP built from them
        // would be the wrong network. `train` builds the CNN of its plan instead.
        let spec = TaskKind::Shapes.spec();
        let data = crate::cnn::train::testing::head(&generate(&spec, Split::Train, 1), 96);
        let mut g = crate::cnn::genome::testing::lenet();

        g.epochs = 5;

        let (model, stats) = train(&g, &spec, &data, &Limits::default()).expect("trained");
        let Model::Cnn(cnn) = model else {
            panic!("a CNN genome gave an MLP");
        };
        let plan = g.cnn_plan(&spec).expect("plan");

        assert_eq!(cnn.param_count(), plan.params);
        assert_eq!(cnn.param_count(), g.param_count(&spec));
        assert_eq!(stats.epochs_done, 5);
        assert_eq!(stats.steps, 5 * 3);
        assert_eq!(stats.flops, plan.train_flops_per_sample() * 96 * 5);
        assert!(!stats.diverged && stats.final_train_loss.is_finite());
    }

    #[test]
    fn bad_data_is_an_error() {
        let spec = TaskKind::TwoSpirals.spec();
        let mut data = generate(&spec, Split::Train, 1);

        data.targets = Targets::Labels(vec![5; data.n]);

        assert!(train(&genome(), &spec, &data, &Limits::default()).is_err());

        let mut data = generate(&spec, Split::Train, 1);

        data.x.pop();

        assert!(train(&genome(), &spec, &data, &Limits::default()).is_err());
    }

    #[test]
    fn divergence_is_reported_with_finite_loss() {
        let spec = TaskKind::SineRegression.spec();
        let data = generate(&spec, Split::Train, 1);
        let mut g = genome();

        g.optimizer = Optimizer::Sgd;
        g.activation = Activation::Relu;
        g.learning_rate = 0.3;
        g.init_scale = 2.0;
        g.hidden = vec![64, 64, 64, 64];
        g.batch_size = 16;

        let (_, stats) = train(&g, &spec, &data, &Limits::default()).expect("ok");

        assert!(stats.final_train_loss.is_finite());

        if stats.diverged {
            assert_eq!(stats.final_train_loss, f32::MAX);
        }
    }

    #[test]
    fn kill_switch_stops_training_between_steps() {
        let spec = TaskKind::TwoSpirals.spec();
        let data = generate(&spec, Split::Train, 1);
        let calls = std::cell::Cell::new(0u32);

        // Answers "stop" from the third check on.
        let stop = || {
            calls.set(calls.get() + 1);
            calls.get() >= 3
        };

        let (model, stats) =
            train_with_stop(&genome(), &spec, &data, &Limits::default(), &stop).expect("ok");

        assert!(stats.stopped_by_kill_switch);
        assert!(!stats.stopped_by_budget);
        assert!(stats.epochs_done < genome().epochs);
        assert!(stats.final_train_loss.is_finite());
        assert!(model.is_finite());

        // Without a stop request the result equals plain `train`.
        let (a, sa) = train(&genome(), &spec, &data, &Limits::default()).expect("ok");
        let (b, sb) =
            train_with_stop(&genome(), &spec, &data, &Limits::default(), &|| false).expect("ok");

        assert_eq!(a, b);
        assert_eq!(sa.final_train_loss, sb.final_train_loss);
        assert!(!sb.stopped_by_kill_switch);
    }

    #[test]
    fn time_budget_stops_cleanly() {
        let spec = TaskKind::TwoSpirals.spec();
        let data = generate(&spec, Split::Train, 1);
        let limits = Limits {
            max_candidate_seconds: 0.0,
            ..Limits::default()
        };
        let (_, stats) = train(&genome(), &spec, &data, &limits).expect("ok");

        assert!(stats.stopped_by_budget);
        assert_eq!(stats.epochs_done, 0);
        assert!(stats.wall_seconds.is_finite());
    }
}
