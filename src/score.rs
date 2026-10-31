//! Scoring: metrics on a split and the fitness used for selection.

use serde::{Deserialize, Serialize};

use crate::cnn::model::CnnModel;
use crate::cnn::shape::Plan;
use crate::config::ScoreWeights;
use crate::model::Model;
use crate::nn::{Mlp, Workspace};
use crate::task::{Dataset, Objective, Targets, TaskSpec};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// Mean loss (cross-entropy or MSE).
    pub loss: f64,
    /// Classification only.
    pub accuracy: Option<f64>,
    /// Regression only: coefficient of determination.
    pub r2: Option<f64>,
    /// Main score in [0, 1], higher is better: accuracy, or max(0, r2).
    pub primary: f64,
}

/// Worst possible fitness. A finite number, because JSON cannot store NaN or infinity.
pub const WORST_FITNESS: f64 = -1.0e9;

/// Largest loss value stored. Larger or non-finite losses are clipped to this.
pub const MAX_LOSS: f64 = 1.0e9;

/// Most rows per forward chunk in `evaluate` (bounds memory for large splits).
const EVAL_CHUNK: usize = 512;

/// Most activation floats per chunk (rows times the sum of the layer widths), so wide
/// or deep models use smaller chunks. About 8 MiB for the pre-activations and activations
/// of an MLP; a CNN holds fewer at a time (`cnn::model::Evaluator::values_per_row`).
const EVAL_MAX_FLOATS: usize = 1 << 20;

/// Rows per chunk for an MLP with these layer sizes (see `rows_per_chunk`).
fn chunk_rows(sizes: &[usize]) -> usize {
    let per_row = sizes
        .iter()
        .skip(1)
        .fold(0usize, |acc, &w| acc.saturating_add(w));

    rows_per_chunk(per_row)
}

/// Rows per chunk for a model that keeps `per_row` activation values per row: at most
/// `EVAL_CHUNK`, at least 1, and small enough that one chunk needs at most
/// `EVAL_MAX_FLOATS` activation values.
///
/// For an MLP, chunking does not change the result: rows are scored independently and
/// summed in order. For a CNN it does. candle's matrix products (gemm) may sum in another
/// order when the number of rows changes, so a row's outputs can differ in the last bits
/// between chunk sizes; the dense layers of the head show this. The scores of a CNN are
/// therefore reproducible for this rule only: `cnn_chunk_rows`, with `EVAL_CHUNK` and
/// `EVAL_MAX_FLOATS`. A change of any of them changes the bits of CNN validation and hidden
/// scores, not those of MLPs.
fn rows_per_chunk(per_row: usize) -> usize {
    (EVAL_MAX_FLOATS / per_row.max(1)).clamp(1, EVAL_CHUNK)
}

/// Rows per chunk for a CNN of this plan: `rows_per_chunk` of `Plan::values_per_row`.
///
/// `evaluate` scores a CNN in chunks of this size, and `brut-eval` estimates the time of
/// scoring with it before it scores (`Plan::eval_nanos`).
pub(crate) fn cnn_chunk_rows(plan: &Plan) -> usize {
    rows_per_chunk(plan.values_per_row())
}

impl Metrics {
    /// Metrics of a model that could not be scored.
    pub fn failed(objective: Objective) -> Metrics {
        let classification = matches!(objective, Objective::Classification { .. });

        Metrics {
            loss: MAX_LOSS,
            accuracy: classification.then_some(0.0),
            r2: (!classification).then_some(0.0),
            primary: 0.0,
        }
    }
}

/// Evaluate `model` on `data`. Non-finite outputs give primary = 0 and loss = MAX_LOSS.
/// All returned numbers are finite. A model that does not fit the task or the data gives
/// `Metrics::failed`.
pub fn evaluate(model: &Model, spec: &TaskSpec, data: &Dataset) -> Metrics {
    match model {
        Model::Mlp(mlp) => evaluate_mlp(mlp, spec, data),
        Model::Cnn(cnn) => evaluate_cnn(cnn, spec, data),
    }
}

/// `evaluate` for an MLP.
fn evaluate_mlp(model: &Mlp, spec: &TaskSpec, data: &Dataset) -> Metrics {
    let n = data.n;
    let n_in = data.n_in;
    let n_out = data.n_out;

    let shapes_ok = n > 0
        && model.check_shape().is_ok()
        && model.n_in() == n_in
        && model.n_out() == n_out
        && data.x.len() == n * n_in;

    if !shapes_ok {
        return Metrics::failed(spec.objective);
    }

    let rows = chunk_rows(&model.sizes);
    let mut ws = Workspace::new();

    metrics(spec, data, rows, |x, rows| {
        model.forward_into(x, rows, &mut ws).map(<[f32]>::to_vec)
    })
}

/// `evaluate` for a CNN: the forward pass in evaluation mode, after `CnnModel::check`.
fn evaluate_cnn(model: &CnnModel, spec: &TaskSpec, data: &Dataset) -> Metrics {
    let n = data.n;
    let shapes_ok = n > 0
        && data.n_in == spec.n_in
        && data.n_out == spec.n_out
        && data.x.len() == n * data.n_in;

    let evaluator = match model.evaluator(spec) {
        Ok(evaluator) if shapes_ok => evaluator,
        _ => return Metrics::failed(spec.objective),
    };
    // The same rule as `cnn_chunk_rows`, from the plan of the checked model.
    let rows = rows_per_chunk(evaluator.values_per_row());

    metrics(spec, data, rows, |x, rows| evaluator.outputs(x, rows).ok())
}

/// Metrics from the raw outputs of a model. `forward` maps `rows` input rows to their
/// outputs, row-major, or to None when it fails; it is called on chunks of at most `rows`.
fn metrics(
    spec: &TaskSpec,
    data: &Dataset,
    rows: usize,
    forward: impl FnMut(&[f32], usize) -> Option<Vec<f32>>,
) -> Metrics {
    let n = data.n;
    let n_out = data.n_out;

    let result = match (spec.objective, &data.targets) {
        (Objective::Classification { classes }, Targets::Labels(labels))
            if classes == n_out && labels.len() == n =>
        {
            classification(forward, rows, data, labels)
        }
        (Objective::Regression, Targets::Values(values)) if values.len() == n * n_out => {
            regression(forward, rows, data, values)
        }
        _ => None,
    };

    result.unwrap_or(Metrics::failed(spec.objective))
}

/// Cross-entropy and accuracy. None on non-finite output or bad label.
fn classification(
    mut forward: impl FnMut(&[f32], usize) -> Option<Vec<f32>>,
    rows: usize,
    data: &Dataset,
    labels: &[u32],
) -> Option<Metrics> {
    let n_in = data.n_in;
    let n_out = data.n_out;
    let mut loss = 0.0f64;
    let mut correct = 0usize;

    for (x_chunk, l_chunk) in data.x.chunks(rows * n_in).zip(labels.chunks(rows)) {
        let rows = l_chunk.len();
        let out = forward(x_chunk, rows)?;

        if out.len() != rows * n_out {
            return None;
        }

        for (row, &label) in out.chunks_exact(n_out).zip(l_chunk) {
            let label = label as usize;

            if label >= n_out || !row.iter().all(|v| v.is_finite()) {
                return None;
            }

            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = row.iter().map(|&o| (o as f64 - max).exp()).sum();

            loss += max + sum.ln() - row[label] as f64;

            // First index of the maximum, so ties are deterministic.
            let mut best = 0;

            for (j, &v) in row.iter().enumerate() {
                if v > row[best] {
                    best = j;
                }
            }

            if best == label {
                correct += 1;
            }
        }
    }

    let n = data.n as f64;
    let accuracy = correct as f64 / n;

    Some(Metrics {
        loss: clip_loss(loss / n),
        accuracy: Some(accuracy),
        r2: None,
        primary: accuracy,
    })
}

/// MSE and R^2. None on non-finite output.
fn regression(
    mut forward: impl FnMut(&[f32], usize) -> Option<Vec<f32>>,
    rows: usize,
    data: &Dataset,
    values: &[f32],
) -> Option<Metrics> {
    let n_in = data.n_in;
    let n_out = data.n_out;
    let mut means = vec![0.0f64; n_out];

    for row in values.chunks_exact(n_out) {
        for (m, &v) in means.iter_mut().zip(row) {
            *m += v as f64;
        }
    }

    for m in means.iter_mut() {
        *m /= data.n as f64;
    }

    let mut ss_res = 0.0f64;
    let mut ss_tot = 0.0f64;

    for (x_chunk, v_chunk) in data.x.chunks(rows * n_in).zip(values.chunks(rows * n_out)) {
        let rows = v_chunk.len() / n_out;
        let out = forward(x_chunk, rows)?;

        if out.len() != rows * n_out {
            return None;
        }

        for (o_row, t_row) in out.chunks_exact(n_out).zip(v_chunk.chunks_exact(n_out)) {
            for ((&o, &t), &m) in o_row.iter().zip(t_row).zip(&means) {
                if !o.is_finite() {
                    return None;
                }

                let diff = o as f64 - t as f64;
                let dev = t as f64 - m;

                ss_res += diff * diff;
                ss_tot += dev * dev;
            }
        }
    }

    let count = (data.n * n_out) as f64;

    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else if ss_res == 0.0 {
        1.0
    } else {
        0.0
    };

    let r2 = if r2.is_finite() {
        r2.max(-MAX_LOSS)
    } else {
        -MAX_LOSS
    };

    Some(Metrics {
        loss: clip_loss(ss_res / count),
        accuracy: None,
        r2: Some(r2),
        primary: r2.clamp(0.0, 1.0),
    })
}

fn clip_loss(loss: f64) -> f64 {
    if loss.is_finite() {
        loss.min(MAX_LOSS)
    } else {
        MAX_LOSS
    }
}

/// Fitness for selection, higher is better:
/// primary - param_penalty * log10(max(params, 1)) - flop_penalty * log10(max(flops, 1)).
/// Returns WORST_FITNESS if any input is not finite. The result is always finite.
pub fn fitness(metrics: &Metrics, params: usize, flops: u64, w: &ScoreWeights) -> f64 {
    let inputs_finite =
        metrics.primary.is_finite() && w.param_penalty.is_finite() && w.flop_penalty.is_finite();

    if !inputs_finite {
        return WORST_FITNESS;
    }

    let param_term = w.param_penalty * (params.max(1) as f64).log10();
    let flop_term = w.flop_penalty * (flops.max(1) as f64).log10();
    let f = metrics.primary - param_term - flop_term;

    if f.is_finite() {
        f.max(WORST_FITNESS)
    } else {
        WORST_FITNESS
    }
}

/// Indices of the Pareto front for (quality: maximize, cost: minimize).
/// Ties: a point is dominated only if another is >= in quality, <= in cost, and strictly better in one.
/// Points with a non-finite coordinate are never on the front.
pub fn pareto_front(points: &[(f64, f64)]) -> Vec<usize> {
    let finite = |p: &(f64, f64)| p.0.is_finite() && p.1.is_finite();

    (0..points.len())
        .filter(|&i| {
            let (qi, ci) = points[i];

            finite(&points[i])
                && !points.iter().any(|p| {
                    let (qj, cj) = *p;

                    finite(p) && qj >= qi && cj <= ci && (qj > qi || cj < ci)
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Activation;

    /// Identity-like linear model: output = input (2 -> 2).
    fn identity() -> Mlp {
        Mlp {
            sizes: vec![2, 2],
            activation: Activation::Relu,
            weights: vec![vec![1.0, 0.0, 0.0, 1.0]],
            biases: vec![vec![0.0, 0.0]],
        }
    }

    fn cls_spec() -> TaskSpec {
        TaskSpec {
            kind: crate::task::TaskKind::TwoSpirals,
            n_in: 2,
            input: crate::task::InputShape::Flat,
            n_out: 2,
            objective: Objective::Classification { classes: 2 },
            n_train: 4,
            n_valid: 4,
            n_hidden: 4,
            noise: 0.0,
        }
    }

    fn cls_data() -> Dataset {
        Dataset {
            x: vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0],
            targets: Targets::Labels(vec![0, 1, 0, 0]),
            n: 4,
            n_in: 2,
            n_out: 2,
        }
    }

    #[test]
    fn classification_metrics() {
        let m = evaluate(&Model::Mlp(identity()), &cls_spec(), &cls_data());

        assert_eq!(m.accuracy, Some(0.75));
        assert_eq!(m.primary, 0.75);
        assert_eq!(m.r2, None);
        assert!(m.loss.is_finite() && m.loss > 0.0);
    }

    #[test]
    fn regression_metrics() {
        let spec = TaskSpec {
            n_in: 1,
            n_out: 1,
            objective: Objective::Regression,
            ..cls_spec()
        };
        let model = Mlp {
            sizes: vec![1, 1],
            activation: Activation::Tanh,
            weights: vec![vec![1.0]],
            biases: vec![vec![0.0]],
        };
        let perfect = Dataset {
            x: vec![1.0, 2.0, 3.0],
            targets: Targets::Values(vec![1.0, 2.0, 3.0]),
            n: 3,
            n_in: 1,
            n_out: 1,
        };
        let m = evaluate(&Model::Mlp(model.clone()), &spec, &perfect);

        assert_eq!(m.r2, Some(1.0));
        assert_eq!(m.primary, 1.0);
        assert_eq!(m.loss, 0.0);

        let bad = Dataset {
            targets: Targets::Values(vec![3.0, 2.0, 1.0]),
            ..perfect
        };
        let m = evaluate(&Model::Mlp(model), &spec, &bad);

        // ss_res = 8, ss_tot = 2 -> r2 = -3, primary clipped to 0.
        assert_eq!(m.r2, Some(-3.0));
        assert_eq!(m.primary, 0.0);
    }

    #[test]
    fn non_finite_output_fails_safely() {
        let mut model = identity();

        model.biases[0][0] = f32::NAN;

        let m = evaluate(&Model::Mlp(model), &cls_spec(), &cls_data());

        assert_eq!(m.primary, 0.0);
        assert_eq!(m.loss, MAX_LOSS);
    }

    #[test]
    fn shape_mismatch_fails_safely() {
        let mut model = identity();

        model.sizes = vec![3, 2];

        assert_eq!(
            evaluate(&Model::Mlp(model), &cls_spec(), &cls_data()).primary,
            0.0
        );

        let mut data = cls_data();

        data.targets = Targets::Labels(vec![0, 9, 0, 0]);

        assert_eq!(
            evaluate(&Model::Mlp(identity()), &cls_spec(), &data).primary,
            0.0
        );
    }

    #[test]
    fn large_split_uses_chunks_correctly() {
        let n = 1300;
        let x: Vec<f32> = (0..n)
            .flat_map(|i| if i % 3 == 0 { [1.0, 0.0] } else { [0.0, 1.0] })
            .collect();
        let labels: Vec<u32> = (0..n).map(|i| if i % 3 == 0 { 0 } else { 1 }).collect();
        let data = Dataset {
            x,
            targets: Targets::Labels(labels),
            n,
            n_in: 2,
            n_out: 2,
        };

        assert_eq!(
            evaluate(&Model::Mlp(identity()), &cls_spec(), &data).primary,
            1.0
        );
    }

    #[test]
    fn eval_chunk_bounds_activation_memory() {
        assert_eq!(chunk_rows(&[2, 64, 64, 2]), EVAL_CHUNK);

        for sizes in [vec![2, 4096, 1, 4096, 1, 4096, 2], vec![2, 4096, 2]] {
            let rows = chunk_rows(&sizes);
            let floats: usize = sizes[1..].iter().sum();

            assert!((1..=EVAL_CHUNK).contains(&rows));
            assert!(rows * floats <= EVAL_MAX_FLOATS, "{sizes:?}: {rows} rows");
        }

        assert_eq!(chunk_rows(&[2, EVAL_MAX_FLOATS * 2, 2]), 1);
        assert_eq!(chunk_rows(&[]), EVAL_CHUNK);
    }

    #[test]
    fn fitness_formula_and_guards() {
        let w = ScoreWeights {
            param_penalty: 0.01,
            flop_penalty: 0.001,
        };
        let m = Metrics {
            loss: 0.1,
            accuracy: Some(0.9),
            r2: None,
            primary: 0.9,
        };
        let f = fitness(&m, 100, 1_000_000, &w);

        assert!((f - (0.9 - 0.02 - 0.006)).abs() < 1e-12);
        assert_eq!(fitness(&m, 0, 0, &w), 0.9);

        let nan = Metrics {
            primary: f64::NAN,
            ..m
        };

        assert_eq!(fitness(&nan, 1, 1, &w), WORST_FITNESS);

        let bad_w = ScoreWeights {
            param_penalty: f64::INFINITY,
            flop_penalty: 0.0,
        };

        assert_eq!(fitness(&m, 1, 1, &bad_w), WORST_FITNESS);
    }

    #[test]
    fn pareto_front_basic_and_ties() {
        let points = [
            (0.9, 10.0),
            (0.8, 5.0),
            (0.7, 6.0),
            (0.9, 10.0),
            (0.95, 20.0),
            (f64::NAN, 1.0),
        ];
        let front = pareto_front(&points);

        // (0.7, 6) is dominated by (0.8, 5). Exact duplicates do not dominate each other.
        assert_eq!(front, vec![0, 1, 3, 4]);
        assert!(pareto_front(&[]).is_empty());
    }
}
