//! Pre-registered baselines: equal-effort constant tuning and the compute-matched MLP.
//!
//! Owner: WP2. Contract section 5.9 and errata E8. Pure: no I/O.
//!
//! Task scores inside one tuning trial are computed in parallel (rayon) and summed in task
//! order, so results do not depend on the thread count.
#![forbid(unsafe_code)]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::genome::{ACTIVATIONS, Optimizer};
use crate::learner::dsl::{self, ConstKind};
use crate::learner::families;
use crate::learner::protocol::{
    self, Stream, TaskPrep, aulc, label_values, quality, run_learner_prepared,
};
use crate::learner::seeds;
use crate::learner::stats::{normal_f64, uniform_index, unit_f64};
use crate::learner::types::{
    Abort, BaselineId, DslLimits, EvalSettings, LearnKind, LearnerGenome, MlpRecipe, SelectConfig,
    TaskInstance, TaskResult, TunedBaselines,
};
use crate::nn::{Gradients, Mlp, OptimizerState, TargetsRef, Workspace};
use crate::task::Objective;
use crate::train::LR_FLOOR;
use crate::util::{BrutError, Result, derive_seed, sha256_hex};

/// Trials of the pre-registered tuning (contract 5.9; also in the protocol hash).
pub const TUNING_TRIALS: usize = 200;
/// MLP search space.
pub const MLP_HIDDEN_CHOICES: [usize; 4] = [8, 16, 32, 64];
pub const MLP_L2_CHOICES: [f32; 3] = [0.0, 1e-4, 1e-3];
pub const MLP_LR_MIN: f32 = 1e-3;
pub const MLP_LR_MAX: f32 = 1e-1;
/// Most epochs one MLP checkpoint fit may run.
pub const MAX_MLP_EPOCHS: u64 = 100_000;

/// Standard deviation of the log-normal constant perturbation.
const PERTURB_SD: f64 = 0.5;

/// Program baselines in `TunedBaselines::programs` order, with their seed names.
const PROGRAM_BASELINES: [(BaselineId, &str); 4] = [
    (BaselineId::LinearSgd, "linear-sgd"),
    (BaselineId::TwoLayerNn, "nn2-sgd"),
    (BaselineId::Prototype, "prototype"),
    (BaselineId::Perceptron, "perceptron"),
];

/// Recipe used when no MLP trial ran.
const DEFAULT_RECIPE: MlpRecipe = MlpRecipe {
    hidden: 16,
    activation: crate::genome::Activation::Relu,
    lr: 0.01,
    l2: 0.0,
};

/// Equal-effort constant tuning: `trials` random perturbations of all Value/Scale
/// constants (each multiplied by exp(N(0, 0.5²)), kept if the mean AULC on `tasks` improves).
/// Structure never changes (same_structure holds).
///
/// A (1+1) hill climb from `g`; constants are clamped to the default |c| <= 1e3 limit.
/// Returns the best program and its mean AULC (attempt 0, never stopped).
pub fn tune_constants(
    g: &LearnerGenome,
    tasks: &[TaskInstance],
    trials: usize,
    rng: &mut ChaCha8Rng,
    settings: &EvalSettings,
) -> (LearnerGenome, f64) {
    let tuned = tune_program(g, tasks, trials, rng, settings, &|| false);

    (tuned.best, tuned.score)
}

/// Outcome of one tuning: the best program or recipe so far, its mean AULC, whether a stop
/// cut it short, and the FLOPs of every task run (static counts, as charged in the search).
struct Tuned<T> {
    best: T,
    score: f64,
    stopped: bool,
    flops: u64,
}

/// `tune_constants` with a stop flag, checked after every trial.
fn tune_program(
    g: &LearnerGenome,
    tasks: &[TaskInstance],
    trials: usize,
    rng: &mut ChaCha8Rng,
    settings: &EvalSettings,
    stop: &dyn Fn() -> bool,
) -> Tuned<LearnerGenome> {
    let prep: Vec<TaskPrep> = tasks.iter().map(protocol::prepare).collect();
    let (score, flops) = mean_aulc(g, tasks, &prep, settings);
    let mut tuned = Tuned {
        best: g.clone(),
        score,
        stopped: false,
        flops,
    };

    for _ in 0..trials {
        if stop() {
            tuned.stopped = true;

            return tuned;
        }

        let mut candidate = tuned.best.clone();

        perturb_constants(&mut candidate, rng);

        let (score, flops) = mean_aulc(&candidate, tasks, &prep, settings);

        tuned.flops = tuned.flops.saturating_add(flops);

        if score > tuned.score {
            tuned.best = candidate;
            tuned.score = score;
        }
    }

    tuned
}

/// Multiply every Value/Scale constant by exp(N(0, 0.5²)), in program order
/// (setup, predict, learn), clamped to the default constant limit.
fn perturb_constants(g: &mut LearnerGenome, rng: &mut ChaCha8Rng) {
    let limit = DslLimits::default().const_abs_max;

    for instr in g
        .setup
        .iter_mut()
        .chain(g.predict.iter_mut())
        .chain(g.learn.iter_mut())
    {
        if matches!(dsl::sig(instr.op).c, ConstKind::Value | ConstKind::Scale) {
            let factor = (PERTURB_SD * normal_f64(rng)).exp() as f32;

            instr.c = (instr.c * factor).clamp(-limit, limit);
        }
    }
}

/// Mean AULC over `tasks` with their preparations (attempt 0, never stopped; 0 for no
/// tasks) and the FLOPs of the runs.
fn mean_aulc(
    g: &LearnerGenome,
    tasks: &[TaskInstance],
    prep: &[TaskPrep],
    settings: &EvalSettings,
) -> (f64, u64) {
    if tasks.is_empty() {
        return (0.0, 0);
    }

    let results: Vec<TaskResult> = tasks
        .par_iter()
        .zip(prep)
        .map(|(t, p)| run_learner_prepared(g, t, p, 0, settings, &|| false))
        .collect();

    mean_and_flops(&results, tasks.len())
}

/// Mean AULC over `n` tasks and the summed FLOPs of their results.
fn mean_and_flops(results: &[TaskResult], n: usize) -> (f64, u64) {
    let total: f64 = results.iter().map(|r| f64::from(r.aulc)).sum();
    let flops = results
        .iter()
        .fold(0u64, |acc, r| acc.saturating_add(r.flops));

    (total / n.max(1) as f64, flops)
}

/// Random-search tuning of the MLP recipe over hidden ∈ {8,16,32,64}, the 4 activations,
/// lr log-uniform [1e-3, 1e-1], l2 ∈ {0, 1e-4, 1e-3}; `trials` draws, FLOP budget per task =
/// reference NN task cost.
///
/// The first best trial wins ties. With trials = 0 the default recipe (16, ReLU, 0.01, 0)
/// and its score are returned.
pub fn tune_mlp(tasks: &[TaskInstance], trials: usize, rng: &mut ChaCha8Rng) -> (MlpRecipe, f64) {
    let tuned = tune_mlp_inner(tasks, trials, rng, &|| false);

    (tuned.best, tuned.score)
}

fn tune_mlp_inner(
    tasks: &[TaskInstance],
    trials: usize,
    rng: &mut ChaCha8Rng,
    stop: &dyn Fn() -> bool,
) -> Tuned<MlpRecipe> {
    let reference = seeds::reference_nn();
    let budgets: Vec<u64> = tasks
        .iter()
        .map(|t| reference.task_cost(t.f, t.steps, t.n_valid))
        .collect();
    let mut best: Option<(MlpRecipe, f64)> = None;
    let mut flops = 0u64;

    for _ in 0..trials {
        if stop() {
            let (recipe, score) = best.unwrap_or((DEFAULT_RECIPE, 0.0));

            return Tuned {
                best: recipe,
                score,
                stopped: true,
                flops,
            };
        }

        let recipe = draw_recipe(rng);
        let (score, spent) = mean_mlp_aulc(&recipe, tasks, &budgets);

        flops = flops.saturating_add(spent);

        if best.is_none_or(|(_, s)| score > s) {
            best = Some((recipe, score));
        }
    }

    let (best, score) = match best {
        Some(found) => found,
        None => {
            let (score, spent) = mean_mlp_aulc(&DEFAULT_RECIPE, tasks, &budgets);

            flops = flops.saturating_add(spent);

            (DEFAULT_RECIPE, score)
        }
    };

    Tuned {
        best,
        score,
        stopped: false,
        flops,
    }
}

/// One random recipe; draw order hidden, activation, lr, l2.
fn draw_recipe(rng: &mut ChaCha8Rng) -> MlpRecipe {
    let hidden = MLP_HIDDEN_CHOICES[uniform_index(rng, MLP_HIDDEN_CHOICES.len())];
    let activation = ACTIVATIONS[uniform_index(rng, ACTIVATIONS.len())];
    let (lo, hi) = (f64::from(MLP_LR_MIN).ln(), f64::from(MLP_LR_MAX).ln());
    let lr = ((lo + (hi - lo) * unit_f64(rng)).exp() as f32).clamp(MLP_LR_MIN, MLP_LR_MAX);
    let l2 = MLP_L2_CHOICES[uniform_index(rng, MLP_L2_CHOICES.len())];

    MlpRecipe {
        hidden,
        activation,
        lr,
        l2,
    }
}

/// Mean AULC of the recipe's matched fits over `tasks` and the FLOPs they spent.
fn mean_mlp_aulc(recipe: &MlpRecipe, tasks: &[TaskInstance], budgets: &[u64]) -> (f64, u64) {
    if tasks.is_empty() {
        return (0.0, 0);
    }

    let results: Vec<TaskResult> = tasks
        .par_iter()
        .zip(budgets.par_iter())
        .map(|(t, &b)| mlp_matched(t, recipe, b))
        .collect();

    mean_and_flops(&results, tasks.len())
}

/// The tuning task set (errata E8): a SELECT-style set built with SelectConfig::default()
/// and seeds derive_seed(0, "baselines/tune/<family>/<i>"); the even indices
/// (20 hand + 16 grammar = 36 tasks).
pub fn tuning_tasks() -> Vec<TaskInstance> {
    let seed_of = |family: &str, i: usize| derive_seed(0, &format!("baselines/tune/{family}/{i}"));

    families::select_like(&SelectConfig::default(), &seed_of)
        .into_iter()
        .step_by(2)
        .collect()
}

/// Deterministic tuned baselines: tasks = the even indices (36 tasks) of a SELECT task set
/// built like select_tasks but seeded with derive_seed(0, "baselines/tune/<family>/<i>");
/// rng = derive_seed(0, "baselines/tune"), trials 200. Orchestrator and brut-eval both call
/// this and MUST get identical `hash`.
///
/// One rng serves the four programs (LinearSgd, TwoLayerNn, Prototype, Perceptron) and
/// then the MLP recipe, in this order. The hash is identical on both sides as long as no
/// task hits `settings.max_seconds_per_task` (a few ms per task in release builds).
pub fn tuned_baselines(settings: &EvalSettings) -> TunedBaselines {
    tune_all(settings, TUNING_TRIALS, &|| false).baselines
}

/// `tuned_baselines` that checks `stop` between trials (errata E6: STOP aborts the
/// one-time tuning) and also returns the FLOPs it spent (static counts of every task run
/// and MLP fit). Err(Stopped) when it stopped; a partial result is never returned.
pub fn tuned_baselines_with_stop(
    settings: &EvalSettings,
    stop: &dyn Fn() -> bool,
) -> Result<(TunedBaselines, u64)> {
    let tuned = tune_all(settings, TUNING_TRIALS, stop);

    if tuned.stopped {
        return Err(BrutError::Stopped("baseline tuning stopped".into()));
    }

    Ok((tuned.baselines, tuned.flops))
}

/// All baselines after tuning, whether a stop cut the tuning short, and its FLOPs.
struct TuningRun {
    baselines: TunedBaselines,
    stopped: bool,
    flops: u64,
}

/// Tune all baselines with `trials` trials each.
fn tune_all(settings: &EvalSettings, trials: usize, stop: &dyn Fn() -> bool) -> TuningRun {
    let tasks = tuning_tasks();
    let seed_list = seeds::seed_programs();
    let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(0, "baselines/tune"));
    let mut programs = Vec::with_capacity(PROGRAM_BASELINES.len());
    let mut flops = 0u64;

    for (id, name) in PROGRAM_BASELINES {
        let start = seed_list
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, g)| g.clone())
            .unwrap_or_default();
        let tuned = tune_program(&start, &tasks, trials, &mut rng, settings, stop);

        flops = flops.saturating_add(tuned.flops);
        programs.push((id, tuned.best));

        if tuned.stopped {
            return TuningRun {
                baselines: with_hash(programs, DEFAULT_RECIPE),
                stopped: true,
                flops,
            };
        }
    }

    let mlp = tune_mlp_inner(&tasks, trials, &mut rng, stop);

    TuningRun {
        baselines: with_hash(programs, mlp.best),
        stopped: mlp.stopped,
        flops: flops.saturating_add(mlp.flops),
    }
}

fn with_hash(programs: Vec<(BaselineId, LearnerGenome)>, mlp: MlpRecipe) -> TunedBaselines {
    let mut tb = TunedBaselines {
        programs,
        mlp,
        hash: String::new(),
    };

    tb.hash = baselines_hash(&tb);

    tb
}

/// SHA-256 of the canonical JSON of `tb` with `hash` set to "".
pub fn baselines_hash(tb: &TunedBaselines) -> String {
    let mut copy = tb.clone();

    copy.hash = String::new();

    sha256_hex(serde_json::to_string(&copy).unwrap_or_default().as_bytes())
}

/// Per-checkpoint accounting of the matched MLP.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MlpCheckpoint {
    /// FLOP budget of this checkpoint (the candidate's cost up to it).
    pub budget: u64,
    /// Distinct training examples seen up to this checkpoint.
    pub examples: u64,
    /// Full-batch epochs run.
    pub epochs: u64,
    /// Training FLOPs spent: epochs * 6 * P * examples.
    pub spent: u64,
}

/// A candidate's training cost on a task of width f: Setup once, then Predict + Learn per
/// stream step (validation predictions excluded). The MlpMatched budget (contract 5.9).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TrainingCost {
    pub setup: u64,
    pub step: u64,
}

impl TrainingCost {
    /// Static costs of `g` at width f.
    pub fn of(g: &LearnerGenome, f: usize) -> TrainingCost {
        TrainingCost {
            setup: g.setup_cost(f),
            step: g.step_cost(f),
        }
    }

    /// Setup plus the first `steps` stream steps (saturating).
    pub fn until(self, steps: usize) -> u64 {
        self.step
            .saturating_mul(steps as u64)
            .saturating_add(self.setup)
    }
}

/// Compute-matched MLP (nn.rs; Adam; cosine decay to 5 %; full batch). At each checkpoint c
/// it trains from scratch on the distinct examples seen in stream steps [0, step_c) with a
/// FLOP budget equal to the candidate's training cost up to step_c (6·P·n per epoch;
/// at least one epoch), then predicts the validation set. Uses nn::Mlp::new,
/// nn::Mlp::loss_and_grad_into and nn::OptimizerState; does not modify nn.rs or train.rs.
///
/// This is the baseline in B(g) (COR-1): the budget at checkpoint c is exactly
/// `cost.until(step_c)`. Epochs = round(budget / (6 P n)), at least 1, at most
/// MAX_MLP_EPOCHS; so the spend is within half an epoch of the budget, or one epoch when
/// the budget is smaller than that (then the MLP gets more than the candidate). Examples
/// come from the same stream permutation as the candidate's; a drift example carries the
/// label of its latest appearance. Same init (derive_seed(task.seed, "mlp-matched/init"))
/// at every checkpoint. Binary uses a two-logit softmax, so its output is P(class 1).
/// Divergence stops training early. `flops` adds 2 P per validation prediction. A
/// malformed task gives abort NonFinite.
pub fn mlp_matched_training(
    task: &TaskInstance,
    recipe: &MlpRecipe,
    cost: TrainingCost,
) -> TaskResult {
    mlp_matched_training_detail(task, recipe, cost).0
}

/// `mlp_matched_training` plus the per-checkpoint accounting.
pub fn mlp_matched_training_detail(
    task: &TaskInstance,
    recipe: &MlpRecipe,
    cost: TrainingCost,
) -> (TaskResult, [MlpCheckpoint; 3]) {
    fit_matched(task, recipe, &|until| cost.until(until))
}

/// The same fit with a budget prorated from one total: total * step_c / steps. Only the
/// recipe tuning uses it (contract 5.9: "FLOP budget per task = reference NN task cost").
pub fn mlp_matched(
    task: &TaskInstance,
    recipe: &MlpRecipe,
    candidate_task_cost: u64,
) -> TaskResult {
    mlp_matched_detail(task, recipe, candidate_task_cost).0
}

/// `mlp_matched` plus the per-checkpoint accounting (budget, examples, epochs, spend).
pub fn mlp_matched_detail(
    task: &TaskInstance,
    recipe: &MlpRecipe,
    candidate_task_cost: u64,
) -> (TaskResult, [MlpCheckpoint; 3]) {
    let steps = task.steps;

    fit_matched(task, recipe, &|until| {
        prorate(candidate_task_cost, until, steps)
    })
}

/// The MLP fits at the three checkpoints; `budget_at(step_c)` is the FLOP budget of each.
fn fit_matched(
    task: &TaskInstance,
    recipe: &MlpRecipe,
    budget_at: &dyn Fn(usize) -> u64,
) -> (TaskResult, [MlpCheckpoint; 3]) {
    let mut details = [MlpCheckpoint::default(); 3];

    if protocol::validate_task(task).is_err() {
        return (protocol::aborted(Abort::NonFinite, 0, 0), details);
    }

    let f = task.f;
    let kind = task.kind;
    let (n_out, objective) = match kind {
        LearnKind::Regression => (1, Objective::Regression),
        LearnKind::Binary => (2, Objective::Classification { classes: 2 }),
        LearnKind::Multiclass { k } => (
            k as usize,
            Objective::Classification {
                classes: k as usize,
            },
        ),
    };
    let sizes = [f, recipe.hidden.clamp(1, 4_096), n_out];
    let lr = if recipe.lr.is_finite() {
        recipe.lr.max(0.0)
    } else {
        0.0
    };
    let l2 = if recipe.l2.is_finite() {
        recipe.l2.max(0.0)
    } else {
        0.0
    };

    let checkpoints = protocol::checkpoint_steps(task.steps);
    let labels = label_values(&task.y_train);
    let late = task.y_train_late.as_ref().map(label_values);
    let drift_at = match (&late, task.drift_at) {
        (Some(_), Some(at)) => at,
        _ => usize::MAX,
    };

    let mut stream = Stream::new(task.seed, task.n_train);
    let mut last_seen: Vec<Option<usize>> = vec![None; task.n_train];
    let mut walked = 0usize;
    let mut q = [0.0f32; 3];
    let mut flops = 0u64;
    let mut ws = Workspace::new();
    let mut xb: Vec<f32> = Vec::with_capacity(task.n_train * f);
    let mut values: Vec<f32> = Vec::with_capacity(task.n_train);
    let mut classes: Vec<u32> = Vec::with_capacity(task.n_train);
    let mut outputs = vec![0.0f32; task.n_valid * kind.k_out()];

    for (c, &until) in checkpoints.iter().enumerate() {
        while walked < until {
            last_seen[stream.index(walked)] = Some(walked);
            walked += 1;
        }

        xb.clear();
        values.clear();
        classes.clear();

        for (i, seen) in last_seen.iter().enumerate() {
            let Some(step) = *seen else {
                continue;
            };
            let y = match &late {
                Some(l) if step >= drift_at => l[i],
                _ => labels[i],
            };

            xb.extend_from_slice(&task.x_train[i * f..(i + 1) * f]);

            match kind {
                LearnKind::Regression => values.push(y),
                _ => classes.push(y as u32),
            }
        }

        let n = xb.len() / f;
        let budget = budget_at(until);
        let mut init = ChaCha8Rng::seed_from_u64(derive_seed(task.seed, "mlp-matched/init"));
        let mut model = Mlp::new(&sizes, recipe.activation, 1.0, &mut init);
        let params = model.param_count() as u64;
        let per_epoch = 6u64.saturating_mul(params).saturating_mul(n as u64);
        let epochs = if n == 0 {
            0
        } else {
            epochs_for(budget, per_epoch)
        };
        let mut grads = Gradients::zeros_like(&model);
        let mut opt = OptimizerState::new(Optimizer::Adam, &model);
        let mut done = 0u64;

        for e in 0..epochs {
            let targets = match kind {
                LearnKind::Regression => TargetsRef::Values(&values),
                _ => TargetsRef::Labels(&classes),
            };
            let loss = model.loss_and_grad_into(&xb, targets, n, objective, &mut ws, &mut grads);

            if !loss.is_finite() {
                break;
            }

            let progress = e as f32 / epochs as f32;
            let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            let step_lr = lr * (LR_FLOOR + (1.0 - LR_FLOOR) * cosine);

            opt.step(&mut model, &grads, step_lr, l2);
            done += 1;
        }

        let spent = done.saturating_mul(per_epoch);
        let predict = 2u64
            .saturating_mul(params)
            .saturating_mul(task.n_valid as u64);

        q[c] = match model.forward_into(&task.x_valid, task.n_valid, &mut ws) {
            Some(raw) => {
                link_rows(kind, raw, n_out, &mut outputs);

                quality(kind, &outputs, &task.y_valid)
            }
            None => 0.0,
        };

        flops = flops.saturating_add(spent).saturating_add(predict);
        details[c] = MlpCheckpoint {
            budget,
            examples: n as u64,
            epochs: done,
            spent,
        };
    }

    let result = TaskResult {
        q,
        aulc: aulc(q),
        abort: None,
        flops,
        steps_done: task.steps.min(u32::MAX as usize) as u32,
    };

    (result, details)
}

/// cost * until / steps in u128 (0 for steps = 0).
fn prorate(cost: u64, until: usize, steps: usize) -> u64 {
    if steps == 0 {
        return 0;
    }

    let value = u128::from(cost) * until as u128 / steps as u128;

    value.min(u128::from(u64::MAX)) as u64
}

/// Epochs for a budget: round(budget / per_epoch), at least 1, at most MAX_MLP_EPOCHS.
fn epochs_for(budget: u64, per_epoch: u64) -> u64 {
    if per_epoch == 0 {
        return 1;
    }

    let rounded = (u128::from(budget) + u128::from(per_epoch) / 2) / u128::from(per_epoch);

    (rounded as u64).clamp(1, MAX_MLP_EPOCHS)
}

/// Raw MLP rows -> protocol outputs: identity (regression), P(class 1) from two logits
/// (binary), softmax (multiclass, max-subtracted).
fn link_rows(kind: LearnKind, raw: &[f32], n_out: usize, outputs: &mut [f32]) {
    match kind {
        LearnKind::Regression => {
            for (o, r) in outputs.iter_mut().zip(raw.chunks_exact(n_out)) {
                *o = r[0];
            }
        }
        LearnKind::Binary => {
            for (o, r) in outputs.iter_mut().zip(raw.chunks_exact(n_out)) {
                *o = 1.0 / (1.0 + (r[0] - r[1]).exp());
            }
        }
        LearnKind::Multiclass { .. } => {
            for (o, r) in outputs.chunks_exact_mut(n_out).zip(raw.chunks_exact(n_out)) {
                let max = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;

                for (oi, &ri) in o.iter_mut().zip(r) {
                    *oi = (ri - max).exp();
                    sum += *oi;
                }

                for oi in o.iter_mut() {
                    *oi /= sum;
                }
            }
        }
    }
}

/// Baselines in B(g) for a candidate with per-step cost `cand_step_cost_f16` at F = 16:
/// every program baseline whose step cost at F = 16 is <= the candidate's, plus MlpMatched.
/// Program baselines keep their `tb.programs` order; MlpMatched comes last.
pub fn compute_matched_set(tb: &TunedBaselines, cand_step_cost_f16: u64) -> Vec<BaselineId> {
    let mut set: Vec<BaselineId> = tb
        .programs
        .iter()
        .filter(|(_, g)| g.step_cost(16) <= cand_step_cost_f16)
        .map(|(id, _)| *id)
        .collect();

    set.push(BaselineId::MlpMatched);

    set
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::genome::Activation;
    use crate::learner::families::{InstanceParams, hand_instance};
    use crate::learner::types::{FamilyId, Labels};

    const RELAXED: EvalSettings = EvalSettings {
        max_seconds_per_task: 1.0e9,
    };

    fn task(name: &str, f: usize, n_train: usize, seed: u64) -> TaskInstance {
        let p = InstanceParams {
            f,
            n_train,
            n_valid: 64,
            level: 1,
        };

        hand_instance(&FamilyId(name.into()), seed, &p).expect("known family")
    }

    fn recipe() -> MlpRecipe {
        MlpRecipe {
            hidden: 16,
            activation: Activation::Tanh,
            lr: 0.03,
            l2: 0.0,
        }
    }

    /// T-BAS-2: spend within ±10 % of the budget per checkpoint when the budget covers at
    /// least five epochs; the at-least-one-epoch rule otherwise.
    #[test]
    fn mlp_matched_spends_the_budget() {
        for (name, f, n_train) in [
            ("lin-reg", 8, 16),
            ("lin-cls", 16, 64),
            ("multiclass-lin", 32, 256),
        ] {
            let t = task(name, f, n_train, 4);
            let n_out = match t.kind {
                LearnKind::Regression => 1,
                LearnKind::Binary => 2,
                LearnKind::Multiclass { k } => k as usize,
            };
            let params = (f * 16 + 16 + 16 * n_out + n_out) as u64;
            let big = 400 * 6 * params * n_train as u64;
            let (r, details) = mlp_matched_detail(&t, &recipe(), big);

            assert_eq!(r.abort, None);

            for d in details {
                assert!(d.examples > 0 && d.examples <= n_train as u64);

                if d.budget >= 5 * 6 * params * d.examples {
                    let ratio = d.spent as f64 / d.budget as f64;

                    assert!((0.9..=1.1).contains(&ratio), "{name}: {d:?}");
                } else {
                    assert_eq!(d.epochs, 1, "{name}: {d:?}");
                }
            }

            assert_eq!(details[0].budget, big * 24 / 384);
            assert_eq!(details[2].budget, big);
            assert_eq!(details[2].examples, n_train as u64);
        }

        // A tiny budget still trains one epoch per checkpoint.
        let t = task("lin-reg", 8, 64, 1);
        let (_, details) = mlp_matched_detail(&t, &recipe(), 10);

        assert!(details.iter().all(|d| d.epochs == 1 && d.spent > d.budget));
    }

    /// COR-1 (contract 5.9): the budget at checkpoint c is the candidate's training cost up
    /// to step_c, without the validation predictions that `task_cost` also counts.
    #[test]
    fn matched_budget_is_the_training_cost_only() {
        let g = seeds::seed_programs()[0].1.clone();
        let p = InstanceParams {
            f: 16,
            n_train: 64,
            n_valid: 256,
            level: 1,
        };
        let t = hand_instance(&FamilyId("lin-cls".into()), 7, &p).expect("known family");
        let cost = TrainingCost::of(&g, t.f);

        assert_eq!(cost.setup, g.setup_cost(16));
        assert_eq!(cost.step, g.step_cost(16));

        let (r, details) = mlp_matched_training_detail(&t, &recipe(), cost);

        assert_eq!(r.abort, None);

        for (d, until) in details.iter().zip(protocol::checkpoint_steps(t.steps)) {
            assert_eq!(d.budget, g.setup_cost(16) + until as u64 * g.step_cost(16));
        }

        // The full task cost would almost double the last budget at n_valid = 256.
        let full = g.task_cost(16, t.steps, t.n_valid);

        assert!(details[2].budget * 3 < full * 2, "{details:?} vs {full}");
        assert_eq!(cost.until(0), cost.setup);
        assert_eq!(
            TrainingCost {
                setup: 1,
                step: u64::MAX
            }
            .until(2),
            u64::MAX
        );
    }

    #[test]
    fn mlp_matched_learns_and_is_deterministic() {
        let t = task("lin-reg", 16, 256, 2);
        let budget = 20_000_000;
        let a = mlp_matched(&t, &recipe(), budget);
        let b = mlp_matched(&t, &recipe(), budget);

        assert_eq!(a, b);
        assert!(a.q[2] > 0.5, "MLP should fit lin-reg: {:?}", a.q);
        assert!(a.q[0] <= a.q[2] + 0.1);
        assert!(a.flops > 0);

        // Classification paths run and stay in range.
        for name in ["lin-cls", "multiclass-lin", "drift-cls"] {
            let r = mlp_matched(&task(name, 16, 64, 3), &recipe(), budget);

            assert!(
                r.q.iter().all(|q| (0.0..=1.0).contains(q)),
                "{name}: {:?}",
                r.q
            );
            assert_eq!(r.abort, None);
        }
    }

    #[test]
    fn mlp_matched_sees_only_streamed_examples() {
        // n_train 256: by step 24 at most 24 distinct examples, by 96 at most 96.
        let t = task("lin-cls", 8, 256, 5);
        let (_, details) = mlp_matched_detail(&t, &recipe(), 1_000_000);

        assert_eq!(details[0].examples, 24);
        assert_eq!(details[1].examples, 96);
        assert_eq!(details[2].examples, 256);

        // n_train 16: every example appears within the first epoch.
        let t = task("lin-cls", 8, 16, 5);
        let (_, details) = mlp_matched_detail(&t, &recipe(), 1_000_000);

        assert!(details.iter().all(|d| d.examples == 16));
    }

    #[test]
    fn mlp_matched_refuses_malformed_tasks() {
        let mut t = task("lin-reg", 8, 16, 1);

        t.y_train = Labels::Classes(vec![0; 16]);

        let r = mlp_matched(&t, &recipe(), 1_000);

        assert_eq!(r.abort, Some(Abort::NonFinite));
        assert_eq!(r.aulc, 0.0);
    }

    #[test]
    fn epochs_and_prorating() {
        assert_eq!(epochs_for(1_000, 100), 10);
        assert_eq!(epochs_for(1_049, 100), 10);
        assert_eq!(epochs_for(1_050, 100), 11);
        assert_eq!(epochs_for(1, 100), 1);
        assert_eq!(epochs_for(5, 0), 1);
        assert_eq!(epochs_for(u64::MAX, 1), MAX_MLP_EPOCHS);
        assert_eq!(prorate(384, 24, 384), 24);
        assert_eq!(prorate(u64::MAX, 384, 384), u64::MAX);
        assert_eq!(prorate(10, 1, 0), 0);
    }

    #[test]
    fn links_match_their_definitions() {
        let mut out = [0.0f32; 2];

        link_rows(LearnKind::Binary, &[0.0, 0.0, 2.0, 0.0], 2, &mut out);

        assert!((out[0] - 0.5).abs() < 1e-7);
        assert!((out[1] - 1.0 / (1.0 + 2.0f32.exp())).abs() < 1e-7);

        let mut soft = [0.0f32; 3];

        link_rows(
            LearnKind::Multiclass { k: 3 },
            &[1.0, 1.0, 1.0],
            3,
            &mut soft,
        );

        assert!(soft.iter().all(|&p| (p - 1.0 / 3.0).abs() < 1e-6));
    }

    #[test]
    fn baselines_hash_ignores_the_hash_field() {
        let tb = with_hash(
            vec![(BaselineId::LinearSgd, LearnerGenome::default())],
            recipe(),
        );
        let mut other = tb.clone();

        other.hash = "x".into();

        assert_eq!(baselines_hash(&other), tb.hash);
        assert_eq!(tb.hash.len(), 64);

        other.mlp.lr = 0.5;

        assert_ne!(baselines_hash(&other), tb.hash);
    }

    #[test]
    fn tuning_tasks_are_the_even_indices() {
        let tasks = tuning_tasks();
        let hand = tasks
            .iter()
            .filter(|t| !t.family.0.starts_with("grammar/"))
            .count();

        assert_eq!(tasks.len(), 36);
        assert_eq!(hand, 20);
        assert_eq!(tasks, tuning_tasks());
        assert_eq!(tasks[0].seed, derive_seed(0, "baselines/tune/lin-reg/0"));
        assert_eq!(tasks[1].seed, derive_seed(0, "baselines/tune/lin-reg/2"));
    }

    #[test]
    fn draw_recipe_stays_in_the_search_space() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);

        for _ in 0..500 {
            let r = draw_recipe(&mut rng);

            assert!(MLP_HIDDEN_CHOICES.contains(&r.hidden));
            assert!((MLP_LR_MIN..=MLP_LR_MAX).contains(&r.lr));
            assert!(MLP_L2_CHOICES.contains(&r.l2));
        }
    }

    // ---- Tests below need the WP1 interpreter (skipped while it is a stub). ----

    /// T-BAS-1 (structure): tuning only moves Value/Scale constants and never gets worse.
    #[test]
    fn tune_constants_keeps_structure_and_improves() {
        let tasks: Vec<TaskInstance> = tuning_tasks().into_iter().take(6).collect();
        let g = seeds::reference_nn();
        let prep: Vec<TaskPrep> = tasks.iter().map(protocol::prepare).collect();
        let (start, _) = mean_aulc(&g, &tasks, &prep, &RELAXED);
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let (tuned, score) = tune_constants(&g, &tasks, 8, &mut rng, &RELAXED);
        let mut again_rng = ChaCha8Rng::seed_from_u64(3);
        let (again, again_score) = tune_constants(&g, &tasks, 8, &mut again_rng, &RELAXED);

        assert!(tuned.same_structure(&g));
        assert!(score >= start);
        assert_eq!((tuned.clone(), score), (again, again_score));
        assert!(tuned.check(&DslLimits::default()).is_ok());
    }

    /// T-BAS-1 (determinism): the same hash for two calls and for 1 vs 4 threads.
    /// Uses 2 trials per baseline so that debug builds stay fast; the full 200-trial run is
    /// `full_tuned_baselines_are_reproducible` (ignored, release only).
    #[test]
    fn tuned_baselines_are_thread_count_independent() {
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool")
                .install(|| tune_all(&RELAXED, 2, &|| false))
        };

        let one_run = run(1);
        let four_run = run(4);

        assert!(!one_run.stopped && !four_run.stopped);

        // SCI-R5: the tuning is metered, the same way on any thread count.
        assert!(one_run.flops > 0);
        assert_eq!(one_run.flops, four_run.flops);

        let (one, four) = (one_run.baselines, four_run.baselines);

        assert_eq!(one, four);
        assert_eq!(one.hash, baselines_hash(&one));
        assert_eq!(one.programs.len(), 4);
        assert_eq!(
            one.programs.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            PROGRAM_BASELINES
                .iter()
                .map(|(id, _)| *id)
                .collect::<Vec<_>>()
        );

        for ((_, tuned), (_, seed)) in one.programs.iter().zip(seeds::seed_programs()) {
            assert!(tuned.same_structure(&seed));
        }
    }

    #[test]
    fn stop_aborts_tuning() {
        let result = tuned_baselines_with_stop(&RELAXED, &|| true);

        assert!(matches!(result, Err(BrutError::Stopped(_))));
    }

    /// T-BAS-1 in full: 200 trials, two calls, 1 vs 4 threads. About 3e10 FLOPs per call:
    /// run with `cargo test --release -- --ignored full_tuned_baselines`.
    #[test]
    #[ignore = "slow: about 3e10 FLOPs per call; run in release"]
    fn full_tuned_baselines_are_reproducible() {
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool")
                .install(|| tuned_baselines(&RELAXED))
        };

        let a = run(4);
        let b = run(1);

        assert_eq!(a.hash, b.hash);
        assert_eq!(a, b);

        eprintln!("tuned baselines hash {}", a.hash);
    }

    #[test]
    fn compute_matched_set_filters_by_cost() {
        let programs: Vec<(BaselineId, LearnerGenome)> = PROGRAM_BASELINES
            .iter()
            .zip(seeds::seed_programs())
            .map(|((id, _), (_, g))| (*id, g))
            .collect();
        let costs: Vec<u64> = programs.iter().map(|(_, g)| g.step_cost(16)).collect();
        let tb = with_hash(programs, recipe());
        let nn_cost = seeds::reference_nn().step_cost(16);

        // A candidate as costly as the NN seed may face every program baseline at most
        // that costly; a free candidate faces only the MLP.
        let set = compute_matched_set(&tb, nn_cost);
        let expected: Vec<BaselineId> = PROGRAM_BASELINES
            .iter()
            .zip(&costs)
            .filter(|(_, c)| **c <= nn_cost)
            .map(|((id, _), _)| *id)
            .chain([BaselineId::MlpMatched])
            .collect();

        assert_eq!(set, expected);
        assert!(set.contains(&BaselineId::TwoLayerNn));
        assert_eq!(compute_matched_set(&tb, 0), vec![BaselineId::MlpMatched]);
        assert_eq!(compute_matched_set(&tb, u64::MAX).len(), 5);
    }
}
