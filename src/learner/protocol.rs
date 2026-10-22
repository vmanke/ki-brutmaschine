//! Per-task protocol: run a learner on a task, quality, AULC, fingerprint, protocol hash.
//!
//! Owner: WP2. Contract sections 5.6, 6.2, 6.3, 6.8 and errata E11. Pure: no I/O.
//!
//! The training loop is the hot path of the search. Per step it copies one input row into
//! v0, runs Predict and Learn through the interpreter, and checks at most k_out outputs.
//! Nothing allocates per step: the label vector, the output buffer and the stream
//! permutation are allocated once per task.
#![forbid(unsafe_code)]

use std::time::Instant;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::learner::dsl::{self, ALL_OPS};
use crate::learner::families::{self, InstanceParams};
use crate::learner::grammar::{self, Expr, GrammarFamily};
use crate::learner::interp::{self, Memory};
use crate::learner::seeds;
use crate::learner::stats::{normal_f64, shuffle};
use crate::learner::types::{
    AULC_WEIGHTS, Abort, CHECK_EVERY_STEPS, CHECKPOINT_DIVISORS, Decision, EvalSettings, FamilyId,
    Group, Labels, LearnKind, LearnerGenome, MAX_ABS_TRAIN_ERROR, MAX_K, PROTOCOL_VERSION,
    STEPS_PER_TASK, SUPPORTED_F, TaskInstance, TaskResult, Tier,
};
use crate::util::{BrutError, Result, derive_seed, sha256_hex};

/// Validation rows of every hidden-tier task.
pub const N_VALID_HIDDEN: usize = 256;

/// Hidden tiers: (tier, families, instances per family). Contract section 5.15.
pub const HIDDEN_TIER_SIZES: [(Tier, usize, usize); 4] = [
    (Tier::HiddenInstance, 10, 10),
    (Tier::HiddenNear, 5, 20),
    (Tier::HiddenFar, 4, 25),
    (Tier::HiddenGrammar, 40, 5),
];

/// Pre-registered baseline tuning (part of the protocol hash).
pub const BASELINE_TUNING: &str = "trials=200;tasks=36;seed=derive_seed(0,\"baselines/tune\")";

/// Pre-registered primary test (part of the protocol hash). Contract section 6.8.
pub const PRIMARY_TEST: &str = "sign-flip;reps=100000;alpha=0.05;min_delta=0.01;coupling=0.05";

/// Parameters of the primary decision; they match `PRIMARY_TEST`.
pub const PRIMARY_REPS: usize = 100_000;
pub const PRIMARY_ALPHA: f64 = 0.05;
pub const PRIMARY_MIN_DELTA: f64 = 0.01;
pub const PRIMARY_COUPLING: f64 = 0.05;

/// Probe tasks of the fingerprint: width, rows per split, stream length.
pub const PROBE_F: usize = 16;
pub const PROBE_ROWS: usize = 10;
pub const PROBE_STEPS: usize = 10;

/// Training steps after which the three checkpoints are evaluated: steps / 16, / 4, / 1.
pub fn checkpoint_steps(steps: usize) -> [usize; 3] {
    CHECKPOINT_DIVISORS.map(|d| steps / d.max(1))
}

/// Run learner `g` on `task` (attempt 0 for search; ARC uses attempts 0 and 1).
/// Implements section 6.2 exactly. The time limit and `stop` are polled every
/// CHECK_EVERY_STEPS training steps, starting at step 0 (errata E11), and every
/// CHECK_EVERY_STEPS validation rows of each checkpoint, starting at its first row (SEC-2).
/// Refuses (abort OverCost, flops 0) when g.check_cost() fails. A malformed task (see
/// `validate_task`) gives abort NonFinite with flops 0. Aborted tasks have q = [0, 0, 0] and
/// aulc = 0.
pub fn run_learner(
    g: &LearnerGenome,
    task: &TaskInstance,
    attempt: u32,
    settings: &EvalSettings,
    stop: &dyn Fn() -> bool,
) -> TaskResult {
    run_learner_with_outputs(g, task, attempt, settings, stop).0
}

/// Like run_learner, but also returns the linked validation outputs at the final checkpoint
/// (n_valid rows of k_out values). Used by ARC solving and the wire oracle.
/// The outputs are empty when the run was aborted or refused.
pub fn run_learner_with_outputs(
    g: &LearnerGenome,
    task: &TaskInstance,
    attempt: u32,
    settings: &EvalSettings,
    stop: &dyn Fn() -> bool,
) -> (TaskResult, Vec<f32>) {
    if g.check_cost().is_err() {
        return (aborted(Abort::OverCost, 0, 0), Vec::new());
    }

    if validate_task(task).is_err() {
        return (aborted(Abort::NonFinite, 0, 0), Vec::new());
    }

    let prep = prepare(task);

    match task.f {
        8 => run_width::<8>(g, task, &prep, attempt, settings, stop),
        16 => run_width::<16>(g, task, &prep, attempt, settings, stop),
        32 => run_width::<32>(g, task, &prep, attempt, settings, stop),
        _ => (aborted(Abort::NonFinite, 0, 0), Vec::new()),
    }
}

/// What every run of a task recomputes and what does not depend on the program: the
/// example order of the training stream and the interpreter seed of attempt 0 (PERF-3).
/// Built only by `prepare`, so it always matches the task it was made for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskPrep {
    /// Example index of every stream step (`steps` entries, each < n_train).
    order: Vec<u32>,
    /// derive_seed(task.seed, "interp/0").
    interp_seed0: u64,
    /// The task fields the preparation depends on.
    seed: u64,
    n_train: usize,
    steps: usize,
}

impl TaskPrep {
    fn fits(&self, task: &TaskInstance) -> bool {
        self.seed == task.seed && self.n_train == task.n_train && self.steps == task.steps
    }

    fn interp_seed(&self, attempt: u32) -> u64 {
        if attempt == 0 {
            self.interp_seed0
        } else {
            derive_seed(self.seed, &format!("interp/{attempt}"))
        }
    }
}

/// Precompute the program-independent part of a task run (same stream as `Stream`).
pub fn prepare(task: &TaskInstance) -> TaskPrep {
    let mut stream = Stream::new(task.seed, task.n_train);
    let order = (0..task.steps).map(|s| stream.index(s) as u32).collect();

    TaskPrep {
        order,
        interp_seed0: derive_seed(task.seed, "interp/0"),
        seed: task.seed,
        n_train: task.n_train,
        steps: task.steps,
    }
}

/// `run_learner` with a preparation of the same task (bit-identical results). A
/// preparation of another task is ignored.
pub fn run_learner_prepared(
    g: &LearnerGenome,
    task: &TaskInstance,
    prep: &TaskPrep,
    attempt: u32,
    settings: &EvalSettings,
    stop: &dyn Fn() -> bool,
) -> TaskResult {
    run_prepared_with_outputs(g, task, prep, attempt, settings, stop).0
}

/// `run_learner_with_outputs` with a preparation of the same task.
pub fn run_prepared_with_outputs(
    g: &LearnerGenome,
    task: &TaskInstance,
    prep: &TaskPrep,
    attempt: u32,
    settings: &EvalSettings,
    stop: &dyn Fn() -> bool,
) -> (TaskResult, Vec<f32>) {
    if g.check_cost().is_err() {
        return (aborted(Abort::OverCost, 0, 0), Vec::new());
    }

    if validate_task(task).is_err() {
        return (aborted(Abort::NonFinite, 0, 0), Vec::new());
    }

    if !prep.fits(task) {
        return run_learner_with_outputs(g, task, attempt, settings, stop);
    }

    match task.f {
        8 => run_width::<8>(g, task, prep, attempt, settings, stop),
        16 => run_width::<16>(g, task, prep, attempt, settings, stop),
        32 => run_width::<32>(g, task, prep, attempt, settings, stop),
        _ => (aborted(Abort::NonFinite, 0, 0), Vec::new()),
    }
}

fn run_width<const F: usize>(
    g: &LearnerGenome,
    task: &TaskInstance,
    prep: &TaskPrep,
    attempt: u32,
    settings: &EvalSettings,
    stop: &dyn Fn() -> bool,
) -> (TaskResult, Vec<f32>) {
    let (result, outputs, _, _) = run_task::<F>(g, task, prep, attempt, settings, stop, true);

    (result, outputs)
}

/// A finished run: the result, the final-checkpoint outputs (empty when aborted), the
/// training memory after the last completed step, and the first validation output at each
/// evaluated checkpoint (the last two are for tests).
type Run<const F: usize> = (TaskResult, Vec<f32>, Memory<F>, [f32; 3]);

/// The section 6.2 loop at width F (task already validated, f == F, `prep` fits the task).
/// `evaluate = false` skips the checkpoints (tests compare memories with and without).
fn run_task<const F: usize>(
    g: &LearnerGenome,
    task: &TaskInstance,
    prep: &TaskPrep,
    attempt: u32,
    settings: &EvalSettings,
    stop: &dyn Fn() -> bool,
    evaluate: bool,
) -> Run<F> {
    let started = Instant::now();
    let kind = task.kind;
    let k_out = kind.k_out().min(F);
    let checkpoints = checkpoint_steps(task.steps);
    let labels = label_values(&task.y_train);
    let late = task.y_train_late.as_ref().map(label_values);
    let drift_at = match (&late, task.drift_at) {
        (Some(_), Some(at)) => at,
        _ => usize::MAX,
    };

    let poll = Poll {
        started,
        limit: settings.max_seconds_per_task,
        stop,
    };
    let mut mem = Memory::<F>::new(prep.interp_seed(attempt));
    let mut outputs = vec![0.0f32; task.n_valid * k_out];
    let mut q = [0.0f32; 3];
    let mut first_outputs = [0.0f32; 3];
    let mut evaluated = 0usize;
    let mut steps_done = 0usize;
    let mut partial_predicts = 0u64;
    let mut abort = None;

    interp::run(&g.setup, &mut mem);

    while abort.is_none() && evaluated < 3 && checkpoints[evaluated] == 0 {
        if evaluate {
            match evaluate_checkpoint(g, task, &mem, k_out, &mut outputs, &poll) {
                Ok(quality) => q[evaluated] = quality,
                Err((a, rows)) => {
                    abort = Some(a);
                    partial_predicts = rows as u64;
                    break;
                }
            }

            first_outputs[evaluated] = outputs.first().copied().unwrap_or(0.0);
        }

        evaluated += 1;
    }

    for (s, &example) in prep.order.iter().enumerate() {
        if abort.is_some() {
            break;
        }

        if s % CHECK_EVERY_STEPS == 0
            && let Some(a) = poll.check()
        {
            abort = Some(a);
            break;
        }

        let i = example as usize;

        mem.set_input(&task.x_train[i * F..(i + 1) * F]);
        mem.clear_label();
        interp::run(&g.predict, &mut mem);
        mem.apply_link(kind);

        let y = if s >= drift_at {
            late.as_ref().map_or(labels[i], |l| l[i])
        } else {
            labels[i]
        };

        if mem.v[1][..k_out].iter().any(|v| !v.is_finite()) {
            partial_predicts = 1;
            abort = Some(Abort::NonFinite);
            break;
        }

        if kind == LearnKind::Regression && (mem.v[1][0] - y).abs() > MAX_ABS_TRAIN_ERROR {
            partial_predicts = 1;
            abort = Some(Abort::ErrorTooLarge);
            break;
        }

        mem.set_label(kind, y);
        interp::run(&g.learn, &mut mem);

        steps_done = s + 1;

        while evaluated < 3 && checkpoints[evaluated] == steps_done {
            if evaluate {
                match evaluate_checkpoint(g, task, &mem, k_out, &mut outputs, &poll) {
                    Ok(quality) => q[evaluated] = quality,
                    Err((a, rows)) => {
                        abort = Some(a);
                        partial_predicts = rows as u64;
                        break;
                    }
                }

                first_outputs[evaluated] = outputs.first().copied().unwrap_or(0.0);
            }

            evaluated += 1;
        }
    }

    let flops = static_flops(
        g.setup_cost(F),
        g.predict_cost(F),
        g.learn_cost(F),
        steps_done,
        partial_predicts,
        evaluated,
        task.n_valid,
    );

    let result = match abort {
        Some(a) => aborted(a, flops, steps_done),
        None => TaskResult {
            q,
            aulc: aulc(q),
            abort: None,
            flops,
            steps_done: steps_done as u32,
        },
    };

    if result.abort.is_some() {
        outputs.clear();
    }

    (result, outputs, mem, first_outputs)
}

/// The wall-clock limit and the stop flag of one task run.
struct Poll<'a> {
    started: Instant,
    limit: f64,
    stop: &'a dyn Fn() -> bool,
}

impl Poll<'_> {
    /// Timeout when the limit is reached (NaN limits time out at once, infinite ones never),
    /// else Stopped when the flag is set.
    fn check(&self) -> Option<Abort> {
        let within = self.started.elapsed().as_secs_f64() < self.limit;

        if !within {
            return Some(Abort::Timeout);
        }

        if (self.stop)() {
            return Some(Abort::Stopped);
        }

        None
    }
}

/// Clone the trained memory once and predict every validation row in order (section 6.2.3).
/// Polls every CHECK_EVERY_STEPS rows; an abort returns how many rows were predicted.
fn evaluate_checkpoint<const F: usize>(
    g: &LearnerGenome,
    task: &TaskInstance,
    mem: &Memory<F>,
    k_out: usize,
    outputs: &mut [f32],
    poll: &Poll<'_>,
) -> std::result::Result<f32, (Abort, usize)> {
    let mut eval = mem.clone();

    for (i, (row, out)) in task
        .x_valid
        .chunks_exact(F)
        .zip(outputs.chunks_exact_mut(k_out.max(1)))
        .enumerate()
    {
        if i % CHECK_EVERY_STEPS == 0
            && let Some(a) = poll.check()
        {
            return Err((a, i));
        }

        eval.set_input(row);
        eval.clear_label();
        interp::run(&g.predict, &mut eval);
        eval.apply_link(task.kind);
        out.copy_from_slice(&eval.v[1][..k_out]);
    }

    Ok(quality(task.kind, outputs, &task.y_valid))
}

/// Static FLOPs of the work done: setup, completed steps, a Predict of an aborted step,
/// and n_valid Predicts per evaluated checkpoint. Saturating.
pub(crate) fn static_flops(
    setup: u64,
    predict: u64,
    learn: u64,
    steps_done: usize,
    partial_predicts: u64,
    checkpoints: usize,
    n_valid: usize,
) -> u64 {
    let per_step = predict.saturating_add(learn);
    let training = per_step.saturating_mul(steps_done as u64);
    let partial = predict.saturating_mul(partial_predicts);
    let validation = predict
        .saturating_mul(n_valid as u64)
        .saturating_mul(checkpoints as u64);

    setup
        .saturating_add(training)
        .saturating_add(partial)
        .saturating_add(validation)
}

/// Result of an aborted or refused run: A_t = 0.
pub(crate) fn aborted(abort: Abort, flops: u64, steps_done: usize) -> TaskResult {
    TaskResult {
        q: [0.0; 3],
        aulc: 0.0,
        abort: Some(abort),
        flops,
        steps_done: steps_done as u32,
    }
}

/// Labels as f32 per example (class index for classification).
pub(crate) fn label_values(labels: &Labels) -> Vec<f32> {
    match labels {
        Labels::Values(v) => v.clone(),
        Labels::Classes(c) => c.iter().map(|&y| f32::from(y)).collect(),
    }
}

/// Example order of the training stream: epoch e is a Fisher-Yates permutation of 0..n
/// drawn from derive_seed(task.seed, "stream/<e>"); step s uses epoch s / n, position s % n.
pub(crate) struct Stream {
    seed: u64,
    epoch: usize,
    perm: Vec<u32>,
}

impl Stream {
    pub(crate) fn new(seed: u64, n: usize) -> Stream {
        Stream {
            seed,
            epoch: usize::MAX,
            perm: (0..n as u32).collect(),
        }
    }

    /// Example index of stream step s (0 for an empty training set).
    pub(crate) fn index(&mut self, step: usize) -> usize {
        let n = self.perm.len();

        if n == 0 {
            return 0;
        }

        let epoch = step / n;

        if epoch != self.epoch {
            for (i, p) in self.perm.iter_mut().enumerate() {
                *p = i as u32;
            }

            let mut rng =
                ChaCha8Rng::seed_from_u64(derive_seed(self.seed, &format!("stream/{epoch}")));

            shuffle(&mut rng, &mut self.perm);
            self.epoch = epoch;
        }

        self.perm[step % n] as usize
    }
}

/// Structural check of a task: supported width, k_out <= f (errata E3), Multiclass k in
/// 3..=10, sizes >= 1, matrix and label lengths, label type matching the kind, class
/// labels < k, drift labels and drift step given together, mask length, steps <= u32::MAX.
/// Values are not scanned for finiteness (non-finite data cannot cause a panic).
pub fn validate_task(t: &TaskInstance) -> Result<()> {
    let bad = |what: &str| Err(BrutError::Invalid(format!("task: {what}")));

    if !SUPPORTED_F.contains(&t.f) {
        return bad("unsupported width");
    }

    if let LearnKind::Multiclass { k } = t.kind
        && !(3..=MAX_K).contains(&(k as usize))
    {
        return bad("class count out of range");
    }

    if t.kind.k_out() > t.f {
        return bad("more outputs than the vector width");
    }

    if t.n_train == 0 || t.n_valid == 0 {
        return bad("empty split");
    }

    if t.steps > u32::MAX as usize {
        return bad("stream too long");
    }

    if t.n_train.checked_mul(t.f) != Some(t.x_train.len())
        || t.n_valid.checked_mul(t.f) != Some(t.x_valid.len())
    {
        return bad("input matrix has the wrong length");
    }

    check_labels(t.kind, &t.y_train, t.n_train)?;
    check_labels(t.kind, &t.y_valid, t.n_valid)?;

    match (&t.y_train_late, t.drift_at) {
        (Some(late), Some(_)) => check_labels(t.kind, late, t.n_train)?,
        (None, None) => {}
        _ => return bad("drift labels and drift step must come together"),
    }

    if let Some(mask) = &t.changed_mask
        && mask.len() != t.n_valid
    {
        return bad("changed mask has the wrong length");
    }

    Ok(())
}

fn check_labels(kind: LearnKind, labels: &Labels, n: usize) -> Result<()> {
    let classes = match kind {
        LearnKind::Regression => 0,
        LearnKind::Binary => 2,
        LearnKind::Multiclass { k } => k as usize,
    };

    let ok = match (kind, labels) {
        (LearnKind::Regression, Labels::Values(v)) => v.len() == n,
        (LearnKind::Binary | LearnKind::Multiclass { .. }, Labels::Classes(c)) => {
            c.len() == n && c.iter().all(|&y| (y as usize) < classes)
        }
        _ => false,
    };

    if ok {
        Ok(())
    } else {
        Err(BrutError::Invalid(
            "task: labels do not match the kind or the size".into(),
        ))
    }
}

/// q in [0, 1] for one checkpoint (formulas in section 6.3).
///
/// Regression: clip(1 - MSE / Var(y_valid), 0, 1), 0 when Var < 1e-12.
/// Binary: class = (output >= 0.5). Multiclass: argmax, lowest index on ties.
/// Classification: clip((acc - a0) / (1 - a0), 0, 1), a0 = majority share, 0 when a0 = 1.
/// Rows are `kind.k_out()` values wide. Mismatched kinds or lengths, any non-finite output
/// (COR-2: a NaN row would otherwise count as a class-0 prediction) and non-finite results
/// give 0. Sums run in f64.
pub fn quality(kind: LearnKind, outputs: &[f32], labels: &Labels) -> f32 {
    if outputs.iter().any(|v| !v.is_finite()) {
        return 0.0;
    }

    let q = match (kind, labels) {
        (LearnKind::Regression, Labels::Values(y)) => regression_quality(outputs, y),
        (LearnKind::Binary, Labels::Classes(y)) => classification_quality(outputs, y, 1),
        (LearnKind::Multiclass { k }, Labels::Classes(y)) => {
            classification_quality(outputs, y, k as usize)
        }
        _ => 0.0,
    };

    if q.is_finite() {
        q.clamp(0.0, 1.0) as f32
    } else {
        0.0
    }
}

fn regression_quality(outputs: &[f32], y: &[f32]) -> f64 {
    let n = y.len();

    if n == 0 || outputs.len() != n {
        return 0.0;
    }

    let mean = y.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
    let var = y
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n as f64;

    if var.is_nan() || var < 1e-12 {
        return 0.0;
    }

    let mse = outputs
        .iter()
        .zip(y)
        .map(|(&o, &v)| (f64::from(o) - f64::from(v)).powi(2))
        .sum::<f64>()
        / n as f64;

    1.0 - mse / var
}

/// `width` = 1 means Binary (threshold 0.5); otherwise argmax over `width` outputs.
fn classification_quality(outputs: &[f32], y: &[u8], width: usize) -> f64 {
    let n = y.len();

    if n == 0 || outputs.len() != n * width.max(1) {
        return 0.0;
    }

    let mut counts = [0usize; 256];

    for &label in y {
        counts[label as usize] += 1;
    }

    let a0 = counts.iter().copied().max().unwrap_or(0) as f64 / n as f64;

    if a0 >= 1.0 {
        return 0.0;
    }

    let correct = outputs
        .chunks_exact(width.max(1))
        .zip(y)
        .filter(|&(row, &label)| predicted_class(row) == label as usize)
        .count();
    let acc = correct as f64 / n as f64;

    (acc - a0) / (1.0 - a0)
}

/// Binary rows (one value): 1 when >= 0.5 (NaN gives 0). Wider rows: argmax with strict
/// comparisons, so ties and NaN keep the lowest index.
fn predicted_class(row: &[f32]) -> usize {
    if row.len() == 1 {
        return usize::from(row[0] >= 0.5);
    }

    let mut best = 0;

    for (i, &v) in row.iter().enumerate().skip(1) {
        if v > row[best] {
            best = i;
        }
    }

    best
}

/// 0.25 q0 + 0.25 q1 + 0.5 q2.
pub fn aulc(q: [f32; 3]) -> f32 {
    AULC_WEIGHTS[0] * q[0] + AULC_WEIGHTS[1] * q[1] + AULC_WEIGHTS[2] * q[2]
}

/// The two fixed probe tasks (F = 16; one regression, one binary; 10 train, 10 valid),
/// seeds derive_seed(0, "fec/probe/0") and ".../1".
///
/// Built directly (not from a family) so that they never change with the task families:
/// x ~ N(0, I), w ~ N(0, I); regression y = w·x / 4 + N(0, 0.1²), binary y = 1[w·x > 0].
/// Stream length 10, so the fingerprint sees exactly one epoch.
pub fn probe_tasks() -> [TaskInstance; 2] {
    [probe_task(0), probe_task(1)]
}

fn probe_task(index: usize) -> TaskInstance {
    let regression = index == 0;
    let seed = derive_seed(0, &format!("fec/probe/{index}"));
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let w: Vec<f64> = (0..PROBE_F).map(|_| normal_f64(&mut rng)).collect();

    let split = |rng: &mut ChaCha8Rng| {
        let mut x = Vec::with_capacity(PROBE_ROWS * PROBE_F);
        let mut y = Vec::with_capacity(PROBE_ROWS);

        for _ in 0..PROBE_ROWS {
            let row: Vec<f64> = (0..PROBE_F).map(|_| normal_f64(rng)).collect();
            let s: f64 = row.iter().zip(&w).map(|(a, b)| a * b).sum();
            let label = if regression {
                s / 4.0 + 0.1 * normal_f64(rng)
            } else if s > 0.0 {
                1.0
            } else {
                0.0
            };

            x.extend(row.iter().map(|&v| v as f32));
            y.push(label);
        }

        (x, y)
    };

    let (x_train, y_train) = split(&mut rng);
    let (x_valid, y_valid) = split(&mut rng);

    let (kind, family, y_train, y_valid) = if regression {
        (
            LearnKind::Regression,
            "fec-probe/regression",
            Labels::Values(y_train.iter().map(|&v| v as f32).collect()),
            Labels::Values(y_valid.iter().map(|&v| v as f32).collect()),
        )
    } else {
        (
            LearnKind::Binary,
            "fec-probe/binary",
            Labels::Classes(y_train.iter().map(|&v| v as u8).collect()),
            Labels::Classes(y_valid.iter().map(|&v| v as u8).collect()),
        )
    };

    let params = InstanceParams {
        f: PROBE_F,
        n_train: PROBE_ROWS,
        n_valid: PROBE_ROWS,
        level: 0,
    };

    TaskInstance {
        id: families::instance_id(family, seed, &params),
        family: FamilyId(family.to_string()),
        group: Group::Linear,
        f: PROBE_F,
        kind,
        x_train,
        y_train,
        y_train_late: None,
        drift_at: None,
        x_valid,
        y_valid,
        n_train: PROBE_ROWS,
        n_valid: PROBE_ROWS,
        steps: PROBE_STEPS,
        seed,
        changed_mask: None,
    }
}

/// Fingerprint of every over-cost program: they all score 0 everywhere, so sharing one
/// cache key is correct, and they never reach the interpreter (AC-6).
fn over_cost_fingerprint() -> u64 {
    derive_seed(0, "fec/over-cost")
}

/// Functional fingerprint: run Setup + 10 steps + 10 validation predictions on each probe
/// (checkpoint only at the end), round the 20 outputs to 1e-3 (NaN -> i64::MIN,
/// +-inf -> i64::MAX/MIN+1), SHA-256 over the LE i64s, first 8 bytes LE.
///
/// The probe run uses the protocol's streams and links but never aborts, so every program
/// has a fingerprint. Over-cost programs are not run (one shared fingerprint).
pub fn fingerprint(g: &LearnerGenome, probes: &[TaskInstance; 2]) -> u64 {
    if g.check_cost().is_err() {
        return over_cost_fingerprint();
    }

    let mut hasher = Sha256::new();

    for probe in probes {
        for v in probe_outputs(g, probe) {
            hasher.update(round_key(v).to_le_bytes());
        }
    }

    let digest = hasher.finalize();
    let mut first = [0u8; 8];

    first.copy_from_slice(&digest[..8]);

    u64::from_le_bytes(first)
}

/// 1e-3 rounding key of one output.
fn round_key(v: f32) -> i64 {
    if v.is_nan() {
        i64::MIN
    } else if v == f32::INFINITY {
        i64::MAX
    } else if v == f32::NEG_INFINITY {
        i64::MIN + 1
    } else {
        // `as` saturates for finite values beyond the i64 range.
        (f64::from(v) * 1000.0).round() as i64
    }
}

/// Linked validation outputs after the probe run (empty for a malformed probe).
fn probe_outputs(g: &LearnerGenome, probe: &TaskInstance) -> Vec<f32> {
    if validate_task(probe).is_err() {
        return Vec::new();
    }

    match probe.f {
        8 => probe_outputs_width::<8>(g, probe),
        16 => probe_outputs_width::<16>(g, probe),
        32 => probe_outputs_width::<32>(g, probe),
        _ => Vec::new(),
    }
}

fn probe_outputs_width<const F: usize>(g: &LearnerGenome, task: &TaskInstance) -> Vec<f32> {
    let kind = task.kind;
    let k_out = kind.k_out().min(F);
    let labels = label_values(&task.y_train);
    let mut mem = Memory::<F>::new(derive_seed(task.seed, "interp/0"));
    let mut stream = Stream::new(task.seed, task.n_train);
    let mut outputs = vec![0.0f32; task.n_valid * k_out];

    interp::run(&g.setup, &mut mem);

    for s in 0..task.steps {
        let i = stream.index(s);

        mem.set_input(&task.x_train[i * F..(i + 1) * F]);
        mem.clear_label();
        interp::run(&g.predict, &mut mem);
        mem.apply_link(kind);
        mem.set_label(kind, labels[i]);
        interp::run(&g.learn, &mut mem);
    }

    let never = Poll {
        started: Instant::now(),
        limit: f64::INFINITY,
        stop: &|| false,
    };

    // Never aborts: the limit is infinite and the flag never set.
    let _ = evaluate_checkpoint(g, task, &mem, k_out, &mut outputs, &never);

    outputs
}

/// The pre-registered primary decision (section 6.8): Pass iff p < 0.05, Δ_T >= 0.01 and
/// Δ32 >= Δ816 - 0.05. Non-finite inputs fail.
pub fn primary_decision(p: f64, delta: f64, delta32: f64, delta816: f64) -> Decision {
    let pass =
        p < PRIMARY_ALPHA && delta >= PRIMARY_MIN_DELTA && delta32 >= delta816 - PRIMARY_COUPLING;

    if pass { Decision::Pass } else { Decision::Fail }
}

/// Everything that defines the measurement, as plain data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProtocolDescriptor {
    pub version: String,
    /// (family, group, tier, developer_seen).
    pub families: Vec<(FamilyId, Group, Tier, bool)>,
    pub grammar_search_pool_hash: String,
    pub grammar_select_pool_hash: String,
    pub steps_per_task: usize,
    pub checkpoint_divisors: [usize; 3],
    pub aulc_weights: [f32; 3],
    pub n_valid_hidden: usize,
    /// (tier, families, instances each).
    pub hidden_tier_sizes: Vec<(Tier, usize, usize)>,
    pub seeds_hash: String,
    /// "trials=200;tasks=36;seed=derive_seed(0,\"baselines/tune\")"
    pub baseline_tuning: String,
    /// "sign-flip;reps=100000;alpha=0.05;min_delta=0.01;coupling=0.05"
    pub primary_test: String,
    /// SHA-256 over ALL_OPS names, sigs and costs.
    pub dsl_ops_hash: String,
}

/// The descriptor of this build: split-v1 families in registry order, public pool hashes,
/// protocol constants, hidden tier sizes, seeds hash and DSL op table hash.
pub fn protocol_descriptor() -> ProtocolDescriptor {
    let families = families::registry()
        .iter()
        .zip(families::split_v1())
        .map(|(spec, (id, tier))| (id, spec.group, tier, spec.developer_seen))
        .collect();

    ProtocolDescriptor {
        version: PROTOCOL_VERSION.to_string(),
        families,
        grammar_search_pool_hash: pool_hash(grammar::public_pool_slice(Tier::Search)),
        grammar_select_pool_hash: pool_hash(grammar::public_pool_slice(Tier::Select)),
        steps_per_task: STEPS_PER_TASK,
        checkpoint_divisors: CHECKPOINT_DIVISORS,
        aulc_weights: AULC_WEIGHTS,
        n_valid_hidden: N_VALID_HIDDEN,
        hidden_tier_sizes: HIDDEN_TIER_SIZES.to_vec(),
        seeds_hash: seeds::seeds_hash(),
        baseline_tuning: BASELINE_TUNING.to_string(),
        primary_test: PRIMARY_TEST.to_string(),
        dsl_ops_hash: dsl_ops_hash(),
    }
}

/// SHA-256 hex over the JSON list of (id, signature, kind, expr) of a pool.
pub fn pool_hash(pool: &[GrammarFamily]) -> String {
    let rows: Vec<(&str, &str, LearnKind, &Expr)> = pool
        .iter()
        .map(|f| (f.id.0.as_str(), f.signature.as_str(), f.kind, &f.expr))
        .collect();
    let text = serde_json::to_string(&rows).unwrap_or_default();

    sha256_hex(text.as_bytes())
}

/// SHA-256 hex over one line per opcode: name, operand slots, constant kind and the cost
/// at every supported width.
pub fn dsl_ops_hash() -> String {
    let mut text = String::new();

    for op in ALL_OPS {
        let sig = dsl::sig(op);
        let costs: Vec<String> = SUPPORTED_F
            .iter()
            .map(|&f| dsl::cost(op, f as u64).to_string())
            .collect();

        text.push_str(&format!(
            "{op:?}|{:?}|{:?}|{:?}|{:?}|{}\n",
            sig.a,
            sig.b,
            sig.out,
            sig.c,
            costs.join(",")
        ));
    }

    sha256_hex(text.as_bytes())
}

/// SHA-256 hex over serde_json::to_string(&protocol_descriptor()).
pub fn protocol_hash() -> String {
    descriptor_hash(&protocol_descriptor())
}

/// SHA-256 hex over the JSON of a descriptor.
pub fn descriptor_hash(d: &ProtocolDescriptor) -> String {
    let text = serde_json::to_string(d).unwrap_or_default();

    sha256_hex(text.as_bytes())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use crate::learner::families::{hand_instance, public_reference_tasks};
    use crate::learner::types::{Instr, OpCode};

    const RELAXED: EvalSettings = EvalSettings {
        max_seconds_per_task: 1.0e9,
    };

    fn never() -> bool {
        false
    }

    fn ins(op: OpCode, a: u8, b: u8, out: u8, c: f32) -> Instr {
        Instr { op, a, b, out, c }
    }

    fn values(v: &[f32]) -> Labels {
        Labels::Values(v.to_vec())
    }

    fn classes(v: &[u8]) -> Labels {
        Labels::Classes(v.to_vec())
    }

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn task(name: &str, f: usize, n_train: usize, seed: u64) -> TaskInstance {
        let p = InstanceParams {
            f,
            n_train,
            n_valid: 64,
            level: 1,
        };

        hand_instance(&FamilyId(name.into()), seed, &p).expect("known family")
    }

    #[test]
    fn regression_quality_hand_values() {
        // y = [1, 2, 3, 4]: Var = 1.25. Outputs [1, 2, 3, 5]: MSE = 0.25 -> q = 0.8.
        let y = values(&[1.0, 2.0, 3.0, 4.0]);

        assert!(close(
            quality(LearnKind::Regression, &[1.0, 2.0, 3.0, 5.0], &y),
            0.8,
            1e-6
        ));
        assert_eq!(
            quality(LearnKind::Regression, &[1.0, 2.0, 3.0, 4.0], &y),
            1.0
        );

        // Predicting the mean gives 0; worse than the mean clips to 0.
        assert_eq!(quality(LearnKind::Regression, &[2.5; 4], &y), 0.0);
        assert_eq!(quality(LearnKind::Regression, &[9.0; 4], &y), 0.0);

        // Constant labels: Var < 1e-12 -> 0, even for a perfect prediction.
        assert_eq!(
            quality(LearnKind::Regression, &[3.0; 3], &values(&[3.0; 3])),
            0.0
        );

        // Non-finite outputs and length mismatch give 0.
        assert_eq!(
            quality(LearnKind::Regression, &[f32::NAN, 2.0, 3.0, 4.0], &y),
            0.0
        );
        assert_eq!(quality(LearnKind::Regression, &[1.0, 2.0], &y), 0.0);
    }

    #[test]
    fn binary_quality_hand_values() {
        // Labels 0,0,0,1,1 -> a0 = 0.6. Predictions >= 0.5: 0,0,1,1,1 -> acc 0.8.
        // q = (0.8 - 0.6) / 0.4 = 0.5.
        let y = classes(&[0, 0, 0, 1, 1]);
        let out = [0.1, 0.49, 0.5, 0.9, 0.7];

        assert!(close(quality(LearnKind::Binary, &out, &y), 0.5, 1e-6));

        // Majority guessing scores 0.
        assert_eq!(quality(LearnKind::Binary, &[0.0; 5], &y), 0.0);

        // COR-2: a non-finite output anywhere in the checkpoint scores 0 (it used to count
        // as a class-0 prediction and earn accuracy on class-0 rows).
        assert_eq!(
            quality(LearnKind::Binary, &[f32::NAN, 0.0, 0.0, 1.0, 1.0], &y),
            0.0
        );

        let mostly_nan = [
            0.9,
            0.9,
            0.9,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
        ];

        assert_eq!(
            quality(
                LearnKind::Binary,
                &mostly_nan,
                &classes(&[1, 1, 1, 0, 0, 0, 0, 0])
            ),
            0.0
        );

        // One class only: a0 = 1 -> 0.
        assert_eq!(
            quality(LearnKind::Binary, &[1.0; 3], &classes(&[1, 1, 1])),
            0.0
        );
    }

    #[test]
    fn multiclass_quality_hand_values() {
        // k = 3, labels 0,1,2,2 -> a0 = 0.5.
        // Rows: argmax 0 (ok), tie 1/2 -> 1 (ok), argmax 2 (ok), argmax 0 (wrong): acc 0.75.
        // q = (0.75 - 0.5) / 0.5 = 0.5.
        let y = classes(&[0, 1, 2, 2]);
        let out = [
            0.8, 0.1, 0.1, //
            0.1, 0.45, 0.45, //
            0.0, 0.2, 0.8, //
            0.5, 0.3, 0.2,
        ];
        let kind = LearnKind::Multiclass { k: 3 };

        assert!(close(quality(kind, &out, &y), 0.5, 1e-6));

        // COR-2: NaN rows would argmax to class 0; any non-finite output scores 0.
        let mut with_nan = out;

        with_nan[0] = f32::NAN;
        with_nan[1] = f32::NAN;
        with_nan[2] = f32::NAN;
        assert_eq!(quality(kind, &with_nan, &y), 0.0);
        assert_eq!(predicted_class(&[f32::NAN, 1.0, 2.0]), 0);
        assert_eq!(predicted_class(&[0.2, 0.2, 0.2]), 0);
        assert_eq!(quality(kind, &out[..9], &y), 0.0);
        assert_eq!(quality(kind, &out, &values(&[0.0; 4])), 0.0);
    }

    #[test]
    fn aulc_weights() {
        assert_eq!(aulc([0.0, 0.0, 1.0]), 0.5);
        assert_eq!(aulc([1.0, 1.0, 1.0]), 1.0);
        assert!(close(aulc([0.2, 0.4, 0.8]), 0.05 + 0.1 + 0.4, 1e-6));
    }

    #[test]
    fn checkpoints_are_24_96_384() {
        assert_eq!(checkpoint_steps(STEPS_PER_TASK), [24, 96, 384]);
        assert_eq!(checkpoint_steps(8192), [512, 2048, 8192]);
        assert_eq!(checkpoint_steps(10), [0, 2, 10]);
    }

    #[test]
    fn stream_epochs_are_permutations() {
        let mut stream = Stream::new(42, 16);
        let first: Vec<usize> = (0..16).map(|s| stream.index(s)).collect();
        let second: Vec<usize> = (16..32).map(|s| stream.index(s)).collect();
        let mut sorted = first.clone();

        sorted.sort_unstable();

        assert_eq!(sorted, (0..16).collect::<Vec<_>>());
        assert_ne!(first, second);

        // Same seed, same order; independent of the access pattern.
        let mut again = Stream::new(42, 16);

        assert_eq!(again.index(20), second[4]);
        assert_eq!(again.index(3), first[3]);

        // Pinned to stream/<e> with the shared Fisher-Yates.
        let mut perm: Vec<u32> = (0..16).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(42, "stream/1"));

        shuffle(&mut rng, &mut perm);

        assert_eq!(second, perm.iter().map(|&p| p as usize).collect::<Vec<_>>());
        assert_eq!(Stream::new(1, 0).index(5), 0);
    }

    /// PERF-3: a prepared run is bit-identical to a plain run; a preparation made for
    /// another task is never used.
    #[test]
    fn prepared_runs_equal_plain_runs() {
        let tasks = [
            task("lin-cls", 8, 16, 1),
            task("lin-reg", 16, 64, 2),
            task("drift-cls", 32, 256, 3),
        ];
        let programs = [nn2(), seeds::seed_programs()[0].1.clone(), over_cost()];

        for t in &tasks {
            let prep = prepare(t);

            assert_eq!(prep.order.len(), t.steps);

            for g in &programs {
                let plain = run_learner_with_outputs(g, t, 0, &RELAXED, &never);
                let fast = run_prepared_with_outputs(g, t, &prep, 0, &RELAXED, &never);

                assert_eq!(plain, fast);
                assert_eq!(run_learner(g, t, 1, &RELAXED, &never), {
                    run_learner_prepared(g, t, &prep, 1, &RELAXED, &never)
                });
            }
        }

        let wrong = prepare(&tasks[0]);

        assert_eq!(
            run_learner_prepared(&nn2(), &tasks[1], &wrong, 0, &RELAXED, &never),
            run_learner(&nn2(), &tasks[1], 0, &RELAXED, &never)
        );
    }

    #[test]
    fn validate_task_catches_shape_errors() {
        let good = task("lin-cls", 8, 16, 1);

        assert!(validate_task(&good).is_ok());

        let mut t = good.clone();
        t.f = 12;
        assert!(validate_task(&t).is_err());

        let mut t = good.clone();
        t.x_train.pop();
        assert!(validate_task(&t).is_err());

        let mut t = good.clone();
        t.y_valid = values(&[0.0; 64]);
        assert!(validate_task(&t).is_err());

        let mut t = good.clone();
        t.y_train = classes(&[2; 16]);
        assert!(validate_task(&t).is_err());

        let mut t = good.clone();
        t.drift_at = Some(5);
        assert!(validate_task(&t).is_err());

        let mut t = good.clone();
        t.kind = LearnKind::Multiclass { k: 11 };
        assert!(validate_task(&t).is_err());

        let mut t = good.clone();
        t.changed_mask = Some(vec![true; 3]);
        assert!(validate_task(&t).is_err());

        let mut t = good;
        t.n_valid = 0;
        t.x_valid.clear();
        t.y_valid = classes(&[]);
        assert!(validate_task(&t).is_err());

        // Every builder output is valid, including drift tasks.
        assert!(validate_task(&task("drift-cls", 32, 256, 2)).is_ok());
        assert!(probe_tasks().iter().all(|p| validate_task(p).is_ok()));
    }

    #[test]
    fn probe_tasks_are_fixed() {
        let [a, b] = probe_tasks();

        assert_eq!(probe_tasks(), [a.clone(), b.clone()]);
        assert_eq!(a.kind, LearnKind::Regression);
        assert_eq!(b.kind, LearnKind::Binary);
        assert_eq!((a.f, a.n_train, a.n_valid, a.steps), (16, 10, 10, 10));
        assert_eq!(a.seed, derive_seed(0, "fec/probe/0"));
        assert_eq!(b.seed, derive_seed(0, "fec/probe/1"));

        let Labels::Classes(c) = &b.y_train else {
            panic!("classes expected");
        };

        assert!(c.contains(&0) && c.contains(&1));
    }

    #[test]
    fn round_keys() {
        assert_eq!(round_key(f32::NAN), i64::MIN);
        assert_eq!(round_key(f32::INFINITY), i64::MAX);
        assert_eq!(round_key(f32::NEG_INFINITY), i64::MIN + 1);
        assert_eq!(round_key(0.0012), 1);
        assert_eq!(round_key(-0.0016), -2);
        assert_eq!(round_key(3.0e38), i64::MAX);
    }

    #[test]
    fn static_flops_counts_every_part() {
        // setup 5, predict 3, learn 7, 10 steps, one partial predict, 2 checkpoints of 4.
        assert_eq!(static_flops(5, 3, 7, 10, 1, 2, 4), 5 + 100 + 3 + 24);
        assert_eq!(static_flops(u64::MAX, 1, 1, 1, 0, 0, 0), u64::MAX);
    }

    #[test]
    fn primary_decision_table() {
        assert_eq!(primary_decision(0.01, 0.02, 0.0, 0.04), Decision::Pass);
        assert_eq!(primary_decision(0.05, 0.02, 0.0, 0.0), Decision::Fail);
        assert_eq!(primary_decision(0.01, 0.009, 0.0, 0.0), Decision::Fail);
        assert_eq!(primary_decision(0.01, 0.02, 0.0, 0.06), Decision::Fail);
        assert_eq!(primary_decision(f64::NAN, 0.02, 0.0, 0.0), Decision::Fail);
    }

    #[test]
    fn pool_hashes_are_stable_and_distinct() {
        let a = pool_hash(grammar::public_pool_slice(Tier::Search));
        let b = pool_hash(grammar::public_pool_slice(Tier::Search));
        let c = pool_hash(grammar::public_pool_slice(Tier::Select));

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 64);
    }

    // ---- Tests below need the WP1 interpreter (skipped while it is a stub). ----

    pub(crate) fn nn2() -> LearnerGenome {
        seeds::reference_nn()
    }

    /// nn2-sgd with its Learn part repeated 8 times: 4F² + 8 * 6F² > 4 * (10F² + 5F).
    pub(crate) fn over_cost() -> LearnerGenome {
        let mut heavy = nn2();

        for _ in 0..7 {
            heavy.learn.extend(nn2().learn);
        }

        heavy
    }

    /// Predict copies the label registers (v2, s0) into the output.
    fn cheater() -> LearnerGenome {
        LearnerGenome {
            setup: vec![],
            predict: vec![
                ins(OpCode::VAdd, 2, 2, 1, 0.0),
                ins(OpCode::VScale, 0, 1, 1, 0.0),
                ins(OpCode::VBroadcast, 0, 0, 4, 0.0),
                ins(OpCode::VAdd, 1, 4, 1, 0.0),
            ],
            learn: vec![],
        }
    }

    /// T-PROT-1: a learner that reads the label in Predict gets nothing.
    #[test]
    fn cheating_program_scores_nothing() {
        let tasks = public_reference_tasks(1);
        let total: f32 = tasks
            .iter()
            .map(|t| run_learner(&cheater(), t, 0, &RELAXED, &never).aulc)
            .sum();
        let mean = total / tasks.len() as f32;

        assert_eq!(tasks.len(), 100);
        assert!(mean <= 0.02, "cheater mean AULC {mean}");
    }

    /// T-PROT-2: validation never touches the training memory, its RNG included.
    #[test]
    fn validation_is_side_effect_free() {
        let mut g = nn2();

        // Random draws in Predict feed Learn: a shared RNG would change training.
        g.predict.push(ins(OpCode::VGaussian, 0, 0, 7, 0.01));
        g.learn.push(ins(OpCode::VAdd, 6, 7, 6, 0.0));

        for name in ["lin-cls", "relu-teacher-reg", "multiclass-lin", "drift-cls"] {
            let t = task(name, 16, 64, 5);
            let with = run_task::<16>(&g, &t, &prepare(&t), 0, &RELAXED, &never, true);
            let without = run_task::<16>(&g, &t, &prepare(&t), 0, &RELAXED, &never, false);

            assert_eq!(with.0.steps_done, without.0.steps_done);
            assert_eq!(with.2.s, without.2.s, "{name}");
            assert_eq!(with.2.v, without.2.v, "{name}");
            assert_eq!(with.2.m, without.2.m, "{name}");
        }
    }

    /// T-PROT-3: aborts and their zero score.
    #[test]
    fn aborts_score_zero() {
        let reg = task("lin-reg", 8, 16, 3);
        let cls = task("lin-cls", 8, 16, 3);

        // NaN output: s4 = -1, s5 = ln(s4), v1 = s5.
        let nan = LearnerGenome {
            setup: vec![ins(OpCode::SConst, 0, 0, 4, -1.0)],
            predict: vec![
                ins(OpCode::SLog, 4, 0, 5, 0.0),
                ins(OpCode::VBroadcast, 5, 0, 1, 0.0),
            ],
            learn: vec![],
        };
        let r = run_learner(&nan, &cls, 0, &RELAXED, &never);

        assert_eq!(r.abort, Some(Abort::NonFinite));
        assert_eq!((r.aulc, r.q, r.steps_done), (0.0, [0.0; 3], 0));
        assert!(r.flops > 0);

        // Regression error > 100: predict 1000.
        let big = LearnerGenome {
            setup: vec![ins(OpCode::SConst, 0, 0, 4, 1000.0)],
            predict: vec![ins(OpCode::VBroadcast, 4, 0, 1, 0.0)],
            learn: vec![],
        };
        let r = run_learner(&big, &reg, 0, &RELAXED, &never);

        assert_eq!(r.abort, Some(Abort::ErrorTooLarge));
        assert_eq!(r.aulc, 0.0);

        // The same program on a classification task is linked by the sigmoid: no abort.
        assert_eq!(run_learner(&big, &cls, 0, &RELAXED, &never).abort, None);

        // Timeout and stop are checked at step 0 (errata E11).
        let zero = EvalSettings {
            max_seconds_per_task: 0.0,
        };
        let r = run_learner(&nn2(), &cls, 0, &zero, &never);

        assert_eq!(r.abort, Some(Abort::Timeout));
        assert_eq!((r.aulc, r.steps_done), (0.0, 0));

        let r = run_learner(&nn2(), &cls, 0, &RELAXED, &|| true);

        assert_eq!(r.abort, Some(Abort::Stopped));
        assert_eq!(r.aulc, 0.0);

        // Over-cost programs are refused before running.
        let heavy = over_cost();
        let (r, out) = run_learner_with_outputs(&heavy, &cls, 0, &RELAXED, &never);

        assert_eq!(r.abort, Some(Abort::OverCost));
        assert_eq!((r.flops, r.aulc), (0, 0.0));
        assert!(out.is_empty());

        // A healthy run is not aborted and reports the full static cost.
        let g = nn2();
        let r = run_learner(&g, &cls, 0, &RELAXED, &never);

        assert_eq!(r.abort, None);
        assert_eq!(r.steps_done, 384);
        assert_eq!(r.flops, g.task_cost(8, 384, 64));
    }

    /// A stop flag that turns true from its `n`-th poll on, and counts the polls.
    fn stop_after(n: usize) -> (std::cell::Cell<usize>, usize) {
        (std::cell::Cell::new(0), n)
    }

    /// SEC-2 (T-PROT-3): a 384-step task is polled during training and during validation,
    /// not only before step 0, so a stop or a time limit cuts it short.
    #[test]
    fn stop_and_time_are_polled_inside_a_task() {
        let cls = task("lin-cls", 16, 64, 3);
        let poll = |counter: &(std::cell::Cell<usize>, usize)| {
            counter.0.set(counter.0.get() + 1);

            counter.0.get() >= counter.1
        };

        // Training: polls at step 0 and before the first checkpoint pass; the third (at step
        // CHECK_EVERY_STEPS) stops the task.
        let training = stop_after(3);
        let r = run_learner(&nn2(), &cls, 0, &RELAXED, &|| poll(&training));

        assert_eq!(r.abort, Some(Abort::Stopped));
        assert!(
            r.steps_done > 0 && (r.steps_done as usize) < cls.steps,
            "{r:?}"
        );
        assert_eq!(r.aulc, 0.0);

        // Validation: the first checkpoint (after step 24) polls before its first row.
        let validation = stop_after(2);
        let r = run_learner(&nn2(), &cls, 0, &RELAXED, &|| poll(&validation));

        assert_eq!(r.abort, Some(Abort::Stopped));
        assert_eq!(r.steps_done, 24);

        // A limit far below the task's run time fires inside the task.
        let tight = EvalSettings {
            max_seconds_per_task: 1e-7,
        };
        let r = run_learner(&nn2(), &task("lin-cls", 32, 256, 4), 0, &tight, &never);

        assert_eq!(r.abort, Some(Abort::Timeout));
        assert!((r.steps_done as usize) < STEPS_PER_TASK);
        assert!(training.0.get() >= 3 && validation.0.get() >= 2);
    }

    /// AC-6: an over-cap program never reaches the interpreter, neither in a task run nor
    /// in the fingerprint. The per-thread run counter stays 0.
    #[test]
    fn over_cap_program_never_reaches_the_interpreter() {
        use crate::learner::interp::test_counters;

        let heavy = over_cost();
        let cls = task("lin-cls", 8, 16, 3);
        let probes = probe_tasks();

        assert!(heavy.check_cost().is_err());

        test_counters::start();

        let (r, _) = run_learner_with_outputs(&heavy, &cls, 0, &RELAXED, &never);
        let fp = fingerprint(&heavy, &probes);

        assert_eq!(test_counters::take(), (0, 0));
        assert_eq!(r.abort, Some(Abort::OverCost));
        assert_eq!(fp, over_cost_fingerprint());

        // The counter works: a valid program does reach the interpreter.
        test_counters::start();

        let _ = run_learner(&nn2(), &cls, 0, &RELAXED, &never);

        assert!(test_counters::take().0 > 0);
    }

    /// T-PROT-4: checkpoints after 24, 96 and 384 steps; common random numbers.
    #[test]
    fn checkpoints_and_common_random_numbers() {
        // Learn adds 0.01 to s5 per step; Predict outputs s5.
        let counter = LearnerGenome {
            setup: vec![ins(OpCode::SConst, 0, 0, 6, 0.01)],
            predict: vec![ins(OpCode::VBroadcast, 5, 0, 1, 0.0)],
            learn: vec![ins(OpCode::SAdd, 5, 6, 5, 0.0)],
        };
        let t = task("lin-reg", 16, 64, 9);
        let (result, _, _, first) =
            run_task::<16>(&counter, &t, &prepare(&t), 0, &RELAXED, &never, true);

        assert_eq!(result.abort, None);
        assert!(close(first[0], 0.24, 1e-4), "{first:?}");
        assert!(close(first[1], 0.96, 1e-4), "{first:?}");
        assert!(close(first[2], 3.84, 1e-3), "{first:?}");

        // Two programs that only read random draws (in different registers) see the same
        // stream; another attempt sees a different one.
        let a = LearnerGenome {
            setup: vec![],
            predict: vec![
                ins(OpCode::SUniform, 0, 0, 3, 1.0),
                ins(OpCode::VBroadcast, 3, 0, 1, 0.0),
            ],
            learn: vec![],
        };
        let b = LearnerGenome {
            setup: vec![],
            predict: vec![
                ins(OpCode::SUniform, 0, 0, 8, 1.0),
                ins(OpCode::VBroadcast, 8, 0, 1, 0.0),
            ],
            learn: vec![],
        };
        let (ra, oa) = run_learner_with_outputs(&a, &t, 0, &RELAXED, &never);
        let (rb, ob) = run_learner_with_outputs(&b, &t, 0, &RELAXED, &never);
        let (_, oc) = run_learner_with_outputs(&a, &t, 1, &RELAXED, &never);

        assert_eq!(ra, rb);
        assert_eq!(oa, ob);
        assert_ne!(oa, oc);
        assert_eq!(oa.len(), 64);
    }

    #[test]
    fn runs_are_deterministic_and_the_nn_seed_learns() {
        let g = nn2();
        let t = task("lin-reg", 16, 256, 2);
        let (a, oa) = run_learner_with_outputs(&g, &t, 0, &RELAXED, &never);
        let (b, ob) = run_learner_with_outputs(&g, &t, 0, &RELAXED, &never);

        assert_eq!(a, b);
        assert_eq!(oa, ob);
        assert!(a.q[2] > 0.5, "nn2-sgd should fit lin-reg: {:?}", a.q);
        assert!(a.q.iter().all(|q| (0.0..=1.0).contains(q)));
        assert!(close(a.aulc, aulc(a.q), 0.0));
    }

    #[test]
    fn fingerprint_separates_behaviour_not_syntax() {
        let probes = probe_tasks();
        let g = nn2();
        let mut padded = g.clone();

        // Dead code (writes a register nobody reads) keeps the behaviour.
        padded.learn.push(ins(OpCode::SConst, 0, 0, 9, 3.0));

        let mut faster = g.clone();

        faster.setup[2].c *= 2.0;

        assert_eq!(fingerprint(&g, &probes), fingerprint(&g, &probes));
        assert_eq!(fingerprint(&g, &probes), fingerprint(&padded, &probes));
        assert_ne!(fingerprint(&g, &probes), fingerprint(&faster, &probes));

        // Over-cost programs share one fingerprint.
        assert!(over_cost().check_cost().is_err());
        assert_eq!(fingerprint(&over_cost(), &probes), over_cost_fingerprint());
    }

    /// T-PROT-5: the protocol hash is stable and reacts to every field.
    #[test]
    fn protocol_hash_is_stable_and_sensitive() {
        let d = protocol_descriptor();
        let h = protocol_hash();

        assert_eq!(h, protocol_hash());
        assert_eq!(h, descriptor_hash(&d));
        assert_eq!(d.version, PROTOCOL_VERSION);
        assert_eq!(d.families.len(), 24);

        let mut changes: Vec<ProtocolDescriptor> = Vec::new();
        let mut c = d.clone();

        // lin-reg is a SELECT family; move it to SEARCH.
        c.families[0].2 = Tier::Search;
        changes.push(c);

        let mut c = d.clone();

        c.families.pop();
        changes.push(c);

        let mut c = d.clone();

        c.seeds_hash.push('0');
        changes.push(c);

        let mut c = d.clone();

        c.steps_per_task += 1;
        changes.push(c);

        let mut c = d.clone();

        c.aulc_weights[0] = 0.3;
        changes.push(c);

        let mut c = d.clone();

        c.grammar_search_pool_hash = pool_hash(&grammar::public_pool_slice(Tier::Search)[1..]);
        changes.push(c);

        let mut c = d.clone();

        c.hidden_tier_sizes[3].1 = 41;
        changes.push(c);

        for changed in changes {
            assert_ne!(descriptor_hash(&changed), h);
        }
    }
}
