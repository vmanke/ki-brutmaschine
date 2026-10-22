//! Minimal, fast multi-layer perceptron on f32 with manual backprop.
//!
//! Single-threaded per model. Parallelism happens across candidates.
//! Hot loops use a reusable `Workspace`, so training allocates nothing per batch.

use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::genome::{Activation, Optimizer};
use crate::task::Objective;
use crate::util::{BrutError, Result, normal_f32};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mlp {
    /// [n_in, hidden.., n_out]
    pub sizes: Vec<usize>,
    pub activation: Activation,
    /// Per layer: row-major (out x in) weight matrix.
    pub weights: Vec<Vec<f32>>,
    /// Per layer: bias vector of length out.
    pub biases: Vec<Vec<f32>>,
}

/// Gradients with the same shape as the model.
#[derive(Clone, Debug)]
pub struct Gradients {
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<Vec<f32>>,
}

/// Borrowed batch targets.
#[derive(Clone, Copy, Debug)]
pub enum TargetsRef<'a> {
    Labels(&'a [u32]),
    Values(&'a [f32]),
}

/// Optimizer state (momentum buffers, Adam moments, step counter).
#[derive(Clone, Debug)]
pub struct OptimizerState {
    pub kind: Optimizer,
    pub step: u64,
    pub m_w: Vec<Vec<f32>>,
    pub m_b: Vec<Vec<f32>>,
    pub v_w: Vec<Vec<f32>>,
    pub v_b: Vec<Vec<f32>>,
}

/// Reusable buffers for forward and backward passes. Grows on demand, never shrinks.
#[derive(Clone, Debug, Default)]
pub struct Workspace {
    /// Per layer: pre-activations, rows x sizes[l + 1]. The last one is the network output.
    zs: Vec<Vec<f32>>,
    /// Per hidden layer: activations, rows x sizes[l + 1].
    acts: Vec<Vec<f32>>,
    delta: Vec<f32>,
    delta_prev: Vec<f32>,
}

impl Workspace {
    pub fn new() -> Workspace {
        Workspace::default()
    }

    /// Size the buffers for `rows` samples of a model with these layer sizes.
    /// The two delta buffers are only needed (and only sized) when `backward` is true.
    fn prepare(&mut self, sizes: &[usize], rows: usize, backward: bool) {
        let layers = sizes.len().saturating_sub(1);
        let widest = sizes.iter().copied().max().unwrap_or(0);

        self.zs.resize_with(layers, Vec::new);
        self.acts.resize_with(layers.saturating_sub(1), Vec::new);

        for (l, z) in self.zs.iter_mut().enumerate() {
            z.resize(rows * sizes[l + 1], 0.0);
        }

        for (l, a) in self.acts.iter_mut().enumerate() {
            a.resize(rows * sizes[l + 1], 0.0);
        }

        if backward {
            self.delta.resize(rows * widest, 0.0);
            self.delta_prev.resize(rows * widest, 0.0);
        }
    }
}

impl Gradients {
    /// Zero gradients shaped like `model`.
    pub fn zeros_like(model: &Mlp) -> Gradients {
        Gradients {
            weights: model.weights.iter().map(|w| vec![0.0; w.len()]).collect(),
            biases: model.biases.iter().map(|b| vec![0.0; b.len()]).collect(),
        }
    }

    fn fill_zero(&mut self) {
        for w in self.weights.iter_mut() {
            w.fill(0.0);
        }

        for b in self.biases.iter_mut() {
            b.fill(0.0);
        }
    }
}

impl Mlp {
    /// He init for ReLU/GELU, Xavier for Tanh/Sine, times `init_scale`. Zero biases.
    pub fn new(
        sizes: &[usize],
        activation: Activation,
        init_scale: f32,
        rng: &mut ChaCha8Rng,
    ) -> Mlp {
        let mut weights = Vec::with_capacity(sizes.len().saturating_sub(1));
        let mut biases = Vec::with_capacity(sizes.len().saturating_sub(1));

        for pair in sizes.windows(2) {
            let (fan_in, fan_out) = (pair[0], pair[1]);
            let std = match activation {
                Activation::Relu | Activation::Gelu => (2.0 / fan_in.max(1) as f32).sqrt(),
                Activation::Tanh | Activation::Sine => {
                    (2.0 / (fan_in + fan_out).max(1) as f32).sqrt()
                }
            };
            let scale = std * init_scale;
            let w: Vec<f32> = (0..fan_in * fan_out)
                .map(|_| scale * normal_f32(rng))
                .collect();

            weights.push(w);
            biases.push(vec![0.0; fan_out]);
        }

        Mlp {
            sizes: sizes.to_vec(),
            activation,
            weights,
            biases,
        }
    }

    pub fn param_count(&self) -> usize {
        let w: usize = self.weights.iter().map(Vec::len).sum();
        let b: usize = self.biases.iter().map(Vec::len).sum();

        w + b
    }

    /// Ok if sizes, weights and biases fit together (needed before any pass).
    pub fn check_shape(&self) -> Result<()> {
        let layers = self.sizes.len().saturating_sub(1);

        if self.sizes.len() < 2 {
            return Err(BrutError::Invalid(
                "model needs at least input and output size".into(),
            ));
        }

        if self.sizes.contains(&0) {
            return Err(BrutError::Invalid("model has a layer of size 0".into()));
        }

        if self.weights.len() != layers || self.biases.len() != layers {
            return Err(BrutError::Invalid(
                "model layer count does not match sizes".into(),
            ));
        }

        for (l, pair) in self.sizes.windows(2).enumerate() {
            let expected = pair[0].checked_mul(pair[1]);

            if expected != Some(self.weights[l].len()) || self.biases[l].len() != pair[1] {
                return Err(BrutError::Invalid(format!(
                    "model layer {l} has wrong shape"
                )));
            }
        }

        Ok(())
    }

    pub fn n_in(&self) -> usize {
        self.sizes.first().copied().unwrap_or(0)
    }

    pub fn n_out(&self) -> usize {
        self.sizes.last().copied().unwrap_or(0)
    }

    /// Forward pass for `n` rows. Returns row-major n x n_out raw outputs (logits for classification).
    /// A malformed model or too-short input gives NaN outputs (never a panic).
    pub fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut ws = Workspace::new();

        match self.forward_into(x, n, &mut ws) {
            Some(out) => out.to_vec(),
            None => vec![f32::NAN; n.saturating_mul(self.n_out())],
        }
    }

    /// Forward pass into `ws`. Returns the n x n_out output, or None on bad shapes.
    pub fn forward_into<'w>(
        &self,
        x: &[f32],
        n: usize,
        ws: &'w mut Workspace,
    ) -> Option<&'w [f32]> {
        if self.check_shape().is_err() || x.len() < n.saturating_mul(self.n_in()) {
            return None;
        }

        ws.prepare(&self.sizes, n, false);
        self.forward_ws(x, n, ws);

        let out_len = n * self.n_out();

        ws.zs.last().map(|z| &z[..out_len])
    }

    /// Forward pass; shapes are already checked and `ws` is prepared.
    fn forward_ws(&self, x: &[f32], n: usize, ws: &mut Workspace) {
        let layers = self.weights.len();
        let Workspace { zs, acts, .. } = ws;

        for l in 0..layers {
            let n_in = self.sizes[l];
            let n_out = self.sizes[l + 1];
            let input: &[f32] = if l == 0 {
                &x[..n * n_in]
            } else {
                &acts[l - 1][..n * n_in]
            };

            linear(
                &self.weights[l],
                &self.biases[l],
                input,
                n_in,
                n_out,
                &mut zs[l][..n * n_out],
            );

            if l + 1 < layers {
                activate(
                    self.activation,
                    &zs[l][..n * n_out],
                    &mut acts[l][..n * n_out],
                );
            }
        }
    }

    /// Mean loss over the batch and its gradients (L2 not included).
    pub fn loss_and_grad(
        &self,
        x: &[f32],
        targets: TargetsRef<'_>,
        n: usize,
        objective: Objective,
    ) -> (f32, Gradients) {
        let mut ws = Workspace::new();
        let mut grads = Gradients::zeros_like(self);
        let loss = self.loss_and_grad_into(x, targets, n, objective, &mut ws, &mut grads);

        (loss, grads)
    }

    /// Like `loss_and_grad`, but reuses `ws` and `grads` (no allocation once sized).
    /// Returns NaN (and zero gradients) on bad shapes or labels.
    pub fn loss_and_grad_into(
        &self,
        x: &[f32],
        targets: TargetsRef<'_>,
        n: usize,
        objective: Objective,
        ws: &mut Workspace,
        grads: &mut Gradients,
    ) -> f32 {
        let n_in = self.n_in();
        let n_out = self.n_out();

        let shapes_ok = grads.weights.len() == self.weights.len()
            && grads.biases.len() == self.biases.len()
            && grads
                .weights
                .iter()
                .zip(&self.weights)
                .all(|(g, w)| g.len() == w.len())
            && grads
                .biases
                .iter()
                .zip(&self.biases)
                .all(|(g, b)| g.len() == b.len());

        if shapes_ok {
            grads.fill_zero();
        } else {
            *grads = Gradients::zeros_like(self);
        }

        if n == 0 || self.check_shape().is_err() || x.len() < n * n_in {
            return f32::NAN;
        }

        ws.prepare(&self.sizes, n, true);
        self.forward_ws(x, n, ws);

        let layers = self.weights.len();
        let Workspace {
            zs,
            acts,
            delta,
            delta_prev,
        } = ws;
        let output = &zs[layers - 1][..n * n_out];
        let inv_n = 1.0 / n as f32;

        // Output delta = dLoss/dOutput, already divided by n.
        let loss = match (objective, targets) {
            (Objective::Classification { classes }, TargetsRef::Labels(labels)) => {
                if classes != n_out || labels.len() < n {
                    return f32::NAN;
                }

                let mut total = 0.0f64;

                for ((row, d_row), &label) in output
                    .chunks_exact(n_out)
                    .zip(delta.chunks_exact_mut(n_out))
                    .zip(labels)
                {
                    let label = label as usize;

                    if label >= n_out {
                        return f32::NAN;
                    }

                    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;

                    for (d, &o) in d_row.iter_mut().zip(row) {
                        *d = (o - max).exp();
                        sum += *d;
                    }

                    let inv_sum = 1.0 / sum;

                    for d in d_row.iter_mut() {
                        *d *= inv_sum * inv_n;
                    }

                    d_row[label] -= inv_n;
                    total += (max + sum.ln() - row[label]) as f64;
                }

                (total / n as f64) as f32
            }
            (Objective::Regression, TargetsRef::Values(values)) => {
                if values.len() < n * n_out {
                    return f32::NAN;
                }

                let count = (n * n_out) as f32;
                let scale = 2.0 / count;
                let mut total = 0.0f64;

                for ((d, &o), &t) in delta.iter_mut().zip(output).zip(values) {
                    let diff = o - t;

                    *d = scale * diff;
                    total += (diff * diff) as f64;
                }

                (total / count as f64) as f32
            }
            _ => return f32::NAN,
        };

        // Backward pass, layer by layer.
        for l in (0..layers).rev() {
            let l_in = self.sizes[l];
            let l_out = self.sizes[l + 1];
            let input: &[f32] = if l == 0 {
                &x[..n * l_in]
            } else {
                &acts[l - 1][..n * l_in]
            };
            let d = &delta[..n * l_out];

            accumulate_grads(
                &mut grads.weights[l],
                &mut grads.biases[l],
                d,
                input,
                l_in,
                l_out,
            );

            if l == 0 {
                break;
            }

            let dp = &mut delta_prev[..n * l_in];

            backprop_delta(dp, d, &self.weights[l], l_in, l_out);

            activation_backward(
                self.activation,
                &zs[l - 1][..n * l_in],
                &acts[l - 1][..n * l_in],
                dp,
            );
            std::mem::swap(delta, delta_prev);
        }

        loss
    }

    /// True if all weights and biases are finite.
    pub fn is_finite(&self) -> bool {
        let w = self.weights.iter().all(|w| w.iter().all(|v| v.is_finite()));
        let b = self.biases.iter().all(|b| b.iter().all(|v| v.is_finite()));

        w && b
    }
}

/// out[i, j] = b[j] + sum_k w[j, k] * input[i, k]
fn linear(w: &[f32], b: &[f32], input: &[f32], n_in: usize, n_out: usize, out: &mut [f32]) {
    for (in_row, out_row) in input.chunks_exact(n_in).zip(out.chunks_exact_mut(n_out)) {
        for ((o, w_row), &bj) in out_row.iter_mut().zip(w.chunks_exact(n_in)).zip(b) {
            *o = bj + dot(w_row, in_row);
        }
    }
}

/// Dot product with 8 independent accumulators, so the compiler can vectorize it.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let ra = ca.remainder();
    let rb = cb.remainder();
    let mut acc = [0.0f32; 8];

    for (x, y) in ca.zip(cb) {
        for ((s, &xi), &yi) in acc.iter_mut().zip(x).zip(y) {
            *s += xi * yi;
        }
    }

    let mut sum = acc.iter().sum::<f32>();

    for (&xi, &yi) in ra.iter().zip(rb) {
        sum += xi * yi;
    }

    sum
}

/// gw[j, k] += sum_i d[i, j] * x[i, k] and gb[j] += sum_i d[i, j] over all samples i.
///
/// Four samples per pass: each gradient row is loaded and stored once per four products
/// instead of once per product. Leftover samples go one at a time. The order of the sums is
/// fixed, so results are deterministic (but not bitwise equal to a one-sample loop).
fn accumulate_grads(
    gw: &mut [f32],
    gb: &mut [f32],
    d: &[f32],
    x: &[f32],
    l_in: usize,
    l_out: usize,
) {
    let mut d_blocks = d.chunks_exact(4 * l_out);
    let mut x_blocks = x.chunks_exact(4 * l_in);

    for (db, xb) in (&mut d_blocks).zip(&mut x_blocks) {
        let (d0, rest) = db.split_at(l_out);
        let (d1, rest) = rest.split_at(l_out);
        let (d2, d3) = rest.split_at(l_out);
        let (x0, rest) = xb.split_at(l_in);
        let (x1, rest) = rest.split_at(l_in);
        let (x2, x3) = rest.split_at(l_in);

        let rows = gw
            .chunks_exact_mut(l_in)
            .zip(gb.iter_mut())
            .zip(d0)
            .zip(d1)
            .zip(d2)
            .zip(d3);

        for (((((gw_row, gbj), &a0), &a1), &a2), &a3) in rows {
            let cols = gw_row.iter_mut().zip(x0).zip(x1).zip(x2).zip(x3);

            for ((((g, &v0), &v1), &v2), &v3) in cols {
                *g += a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3;
            }

            *gbj += a0 + a1 + a2 + a3;
        }
    }

    let d_rest = d_blocks.remainder().chunks_exact(l_out);
    let x_rest = x_blocks.remainder().chunks_exact(l_in);

    for (d_row, in_row) in d_rest.zip(x_rest) {
        for ((&dj, gw_row), gbj) in d_row
            .iter()
            .zip(gw.chunks_exact_mut(l_in))
            .zip(gb.iter_mut())
        {
            axpy(gw_row, dj, in_row);
            *gbj += dj;
        }
    }
}

/// dp[i, k] = sum_j d[i, j] * w[j, k], with `dp` zeroed first.
///
/// Four rows of `w` per pass, so each `dp` row is loaded and stored a quarter as often.
/// Leftover rows go one at a time. Deterministic order of the sums.
fn backprop_delta(dp: &mut [f32], d: &[f32], w: &[f32], l_in: usize, l_out: usize) {
    dp.fill(0.0);

    for (d_row, dp_row) in d.chunks_exact(l_out).zip(dp.chunks_exact_mut(l_in)) {
        let mut d_quads = d_row.chunks_exact(4);
        let mut w_quads = w.chunks_exact(4 * l_in);

        for (dq, wq) in (&mut d_quads).zip(&mut w_quads) {
            let &[a0, a1, a2, a3] = dq else {
                continue;
            };
            let (w0, rest) = wq.split_at(l_in);
            let (w1, rest) = rest.split_at(l_in);
            let (w2, w3) = rest.split_at(l_in);
            let cols = dp_row.iter_mut().zip(w0).zip(w1).zip(w2).zip(w3);

            for ((((p, &v0), &v1), &v2), &v3) in cols {
                *p += a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3;
            }
        }

        let w_rest = w_quads.remainder().chunks_exact(l_in);

        for (&dj, w_row) in d_quads.remainder().iter().zip(w_rest) {
            axpy(dp_row, dj, w_row);
        }
    }
}

/// y += a * x
#[inline]
fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

const GELU_C: f32 = 0.797_884_6; // sqrt(2 / pi)
const GELU_A: f32 = 0.044_715;

/// Beyond this input, `fast_tanh` returns its value at the clamp (1 - 2.7e-7).
/// Chosen so that the result never exceeds 1 without fused multiply-add.
const TANH_CLAMP: f32 = 7.905_311;

/// tanh as a rational function p(x²)·x / q(x²) (the float coefficients of the Eigen library).
///
/// Max absolute error about 5e-7 against exact tanh. Unlike `f32::tanh` (a libm call),
/// it has no branches, so loops over it vectorize. Odd, |result| <= 1, NaN stays NaN.
#[inline(always)]
fn fast_tanh(x: f32) -> f32 {
    // clamp keeps NaN as NaN (a diverged model must stay visible).
    let x = x.clamp(-TANH_CLAMP, TANH_CLAMP);
    let x2 = x * x;

    let mut p = -2.760_768_5e-16;

    p = p * x2 + 2.000_188e-13;
    p = p * x2 - 8.604_672e-11;
    p = p * x2 + 5.122_297e-8;
    p = p * x2 + 1.485_722_4e-5;
    p = p * x2 + 6.372_619_3e-4;
    p = p * x2 + 4.893_524_6e-3;

    let mut q = 1.198_258_4e-6;

    q = q * x2 + 1.185_347_1e-4;
    q = q * x2 + 2.268_434_6e-3;
    q = q * x2 + 4.893_525e-3;

    x * p / q
}

fn activate(act: Activation, z: &[f32], a: &mut [f32]) {
    match act {
        Activation::Relu => {
            for (ai, &zi) in a.iter_mut().zip(z) {
                *ai = zi.max(0.0);
            }
        }
        Activation::Tanh => {
            for (ai, &zi) in a.iter_mut().zip(z) {
                *ai = fast_tanh(zi);
            }
        }
        Activation::Gelu => {
            for (ai, &zi) in a.iter_mut().zip(z) {
                let t = fast_tanh(GELU_C * (zi + GELU_A * zi * zi * zi));

                *ai = 0.5 * zi * (1.0 + t);
            }
        }
        Activation::Sine => {
            for (ai, &zi) in a.iter_mut().zip(z) {
                *ai = zi.sin();
            }
        }
    }
}

/// d *= act'(z), using the stored activation `a` where cheaper.
fn activation_backward(act: Activation, z: &[f32], a: &[f32], d: &mut [f32]) {
    match act {
        Activation::Relu => {
            for (di, &zi) in d.iter_mut().zip(z) {
                if zi <= 0.0 {
                    *di = 0.0;
                }
            }
        }
        Activation::Tanh => {
            for (di, &ai) in d.iter_mut().zip(a) {
                *di *= 1.0 - ai * ai;
            }
        }
        Activation::Gelu => {
            for (di, &zi) in d.iter_mut().zip(z) {
                let t = fast_tanh(GELU_C * (zi + GELU_A * zi * zi * zi));
                let dt = (1.0 - t * t) * GELU_C * (1.0 + 3.0 * GELU_A * zi * zi);

                *di *= 0.5 * (1.0 + t) + 0.5 * zi * dt;
            }
        }
        Activation::Sine => {
            for (di, &zi) in d.iter_mut().zip(z) {
                *di *= zi.cos();
            }
        }
    }
}

impl OptimizerState {
    pub fn new(kind: Optimizer, model: &Mlp) -> OptimizerState {
        let zeros_w =
            || -> Vec<Vec<f32>> { model.weights.iter().map(|w| vec![0.0; w.len()]).collect() };
        let zeros_b =
            || -> Vec<Vec<f32>> { model.biases.iter().map(|b| vec![0.0; b.len()]).collect() };

        let (m_w, m_b) = match kind {
            Optimizer::Sgd => (Vec::new(), Vec::new()),
            Optimizer::Momentum | Optimizer::Adam => (zeros_w(), zeros_b()),
        };

        let (v_w, v_b) = match kind {
            Optimizer::Adam => (zeros_w(), zeros_b()),
            Optimizer::Sgd | Optimizer::Momentum => (Vec::new(), Vec::new()),
        };

        OptimizerState {
            kind,
            step: 0,
            m_w,
            m_b,
            v_w,
            v_b,
        }
    }

    /// One update. Applies decoupled L2 weight decay `l2` to weights only.
    /// SGD: w -= lr * g. Momentum: beta 0.9. Adam: beta1 0.9, beta2 0.999, eps 1e-8, bias-corrected.
    pub fn step(&mut self, model: &mut Mlp, grads: &Gradients, lr: f32, l2: f32) {
        self.step = self.step.saturating_add(1);

        let decay = 1.0 - lr * l2;

        for w in model.weights.iter_mut() {
            for wi in w.iter_mut() {
                *wi *= decay;
            }
        }

        match self.kind {
            Optimizer::Sgd => {
                sgd(&mut model.weights, &grads.weights, lr);
                sgd(&mut model.biases, &grads.biases, lr);
            }
            Optimizer::Momentum => {
                momentum(&mut model.weights, &grads.weights, &mut self.m_w, lr);
                momentum(&mut model.biases, &grads.biases, &mut self.m_b, lr);
            }
            Optimizer::Adam => {
                let t = self.step as f64;
                let bc1 = (1.0 - 0.9f64.powf(t)) as f32;
                let bc2 = (1.0 - 0.999f64.powf(t)) as f32;
                let consts = AdamConsts { lr, bc1, bc2 };

                adam(
                    &mut model.weights,
                    &grads.weights,
                    &mut self.m_w,
                    &mut self.v_w,
                    consts,
                );
                adam(
                    &mut model.biases,
                    &grads.biases,
                    &mut self.m_b,
                    &mut self.v_b,
                    consts,
                );
            }
        }
    }
}

fn sgd(params: &mut [Vec<f32>], grads: &[Vec<f32>], lr: f32) {
    for (p, g) in params.iter_mut().zip(grads) {
        axpy(p, -lr, g);
    }
}

/// Optimizer buffer values smaller than this in magnitude are stored as 0.
///
/// Moments of a parameter whose gradient stays 0 (for example a dead ReLU unit) decay
/// geometrically into the subnormal range, where x86 floating point is about 100 times
/// slower. Flushing them changes an update by at most lr * 1e-30 for momentum, and for Adam
/// sqrt(1e-30) is far below eps = 1e-8, so results do not change in practice.
const FLUSH_BELOW: f32 = 1e-30;

/// `v`, or 0 when |v| < `FLUSH_BELOW`. NaN stays NaN. Compiles to a branch-free select.
#[inline(always)]
fn flush_tiny(v: f32) -> f32 {
    if v.abs() < FLUSH_BELOW { 0.0 } else { v }
}

fn momentum(params: &mut [Vec<f32>], grads: &[Vec<f32>], buf: &mut [Vec<f32>], lr: f32) {
    for ((p, g), m) in params.iter_mut().zip(grads).zip(buf.iter_mut()) {
        for ((pi, &gi), mi) in p.iter_mut().zip(g).zip(m.iter_mut()) {
            *mi = flush_tiny(0.9 * *mi + gi);
            *pi -= lr * *mi;
        }
    }
}

#[derive(Clone, Copy)]
struct AdamConsts {
    lr: f32,
    bc1: f32,
    bc2: f32,
}

fn adam(
    params: &mut [Vec<f32>],
    grads: &[Vec<f32>],
    m: &mut [Vec<f32>],
    v: &mut [Vec<f32>],
    c: AdamConsts,
) {
    let step = c.lr / c.bc1;
    let inv_bc2 = 1.0 / c.bc2;

    for (((p, g), m), v) in params
        .iter_mut()
        .zip(grads)
        .zip(m.iter_mut())
        .zip(v.iter_mut())
    {
        for (((pi, &gi), mi), vi) in p.iter_mut().zip(g).zip(m.iter_mut()).zip(v.iter_mut()) {
            *mi = flush_tiny(0.9 * *mi + 0.1 * gi);
            *vi = flush_tiny(0.999 * *vi + 0.001 * gi * gi);
            *pi -= step * *mi / ((*vi * inv_bc2).sqrt() + 1e-8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn model(act: Activation) -> Mlp {
        let mut rng = ChaCha8Rng::seed_from_u64(1);

        Mlp::new(&[3, 5, 4, 2], act, 1.0, &mut rng)
    }

    fn batch() -> (Vec<f32>, Vec<u32>, Vec<f32>) {
        let x = vec![
            0.5, -0.2, 0.1, -0.7, 0.3, 0.9, 0.2, 0.2, -0.4, 1.0, -1.0, 0.5,
        ];
        let labels = vec![0, 1, 1, 0];
        let values = vec![0.1, -0.3, 0.7, 0.2, -0.5, 0.0, 0.4, 0.9];

        (x, labels, values)
    }

    /// Compare analytic gradients with central finite differences.
    fn grad_check(act: Activation, objective: Objective) {
        let mut m = model(act);
        let (x, labels, values) = batch();
        let targets = match objective {
            Objective::Classification { .. } => TargetsRef::Labels(&labels),
            Objective::Regression => TargetsRef::Values(&values),
        };

        // Avoid the ReLU kink: shift biases a little so no pre-activation sits at 0.
        for b in m.biases.iter_mut() {
            for v in b.iter_mut() {
                *v += 0.01;
            }
        }

        let (_, grads) = m.loss_and_grad(&x, targets, 4, objective);
        let eps = 1e-2f32;

        for l in 0..m.weights.len() {
            for k in 0..m.weights[l].len() {
                let orig = m.weights[l][k];

                m.weights[l][k] = orig + eps;
                let (up, _) = m.loss_and_grad(&x, targets, 4, objective);
                m.weights[l][k] = orig - eps;
                let (down, _) = m.loss_and_grad(&x, targets, 4, objective);
                m.weights[l][k] = orig;

                let numeric = (up - down) / (2.0 * eps);
                let analytic = grads.weights[l][k];

                assert!(
                    (numeric - analytic).abs() < 2e-3 + 2e-2 * analytic.abs(),
                    "{act:?} w[{l}][{k}]: {numeric} vs {analytic}"
                );
            }

            for k in 0..m.biases[l].len() {
                let orig = m.biases[l][k];

                m.biases[l][k] = orig + eps;
                let (up, _) = m.loss_and_grad(&x, targets, 4, objective);
                m.biases[l][k] = orig - eps;
                let (down, _) = m.loss_and_grad(&x, targets, 4, objective);
                m.biases[l][k] = orig;

                let numeric = (up - down) / (2.0 * eps);
                let analytic = grads.biases[l][k];

                assert!(
                    (numeric - analytic).abs() < 2e-3 + 2e-2 * analytic.abs(),
                    "{act:?} b[{l}][{k}]: {numeric} vs {analytic}"
                );
            }
        }
    }

    #[test]
    fn gradients_match_finite_differences() {
        for act in [
            Activation::Relu,
            Activation::Tanh,
            Activation::Gelu,
            Activation::Sine,
        ] {
            grad_check(act, Objective::Classification { classes: 2 });
            grad_check(act, Objective::Regression);
        }
    }

    #[test]
    fn fast_tanh_is_accurate_bounded_and_odd() {
        let mut max_err = 0.0f64;
        let mut x = -12.0f64;

        while x <= 12.0 {
            let xf = x as f32;
            let t = fast_tanh(xf);

            max_err = max_err.max((t as f64 - (xf as f64).tanh()).abs());

            assert!(t.abs() <= 1.0, "fast_tanh({xf}) = {t}");
            assert_eq!(fast_tanh(-xf), -t);

            x += 1e-4;
        }

        assert!(max_err < 1e-6, "max error {max_err}");
        assert_eq!(fast_tanh(0.0), 0.0);
        assert_eq!(fast_tanh(f32::INFINITY), fast_tanh(100.0));
        assert!(fast_tanh(f32::NAN).is_nan());
    }

    #[test]
    fn forward_pass_allocates_no_backward_buffers() {
        let m = model(Activation::Relu);
        let (x, labels, _) = batch();
        let mut ws = Workspace::new();

        assert!(m.forward_into(&x, 4, &mut ws).is_some());
        assert!(ws.delta.is_empty() && ws.delta_prev.is_empty());

        // A training pass with the same workspace still sizes them.
        let mut grads = Gradients::zeros_like(&m);
        let loss = m.loss_and_grad_into(
            &x,
            TargetsRef::Labels(&labels),
            4,
            Objective::Classification { classes: 2 },
            &mut ws,
            &mut grads,
        );

        assert!(loss.is_finite());
        assert!(!ws.delta.is_empty());
    }

    /// Loss gradients in f64 by plain loops, as a reference for the blocked kernels.
    fn reference_grads(m: &Mlp, x: &[f32], labels: &[u32], n: usize) -> Vec<Vec<f64>> {
        let layers = m.weights.len();
        let mut acts: Vec<Vec<f64>> = vec![x[..n * m.n_in()].iter().map(|&v| v as f64).collect()];
        let mut zs: Vec<Vec<f64>> = Vec::new();

        for l in 0..layers {
            let (l_in, l_out) = (m.sizes[l], m.sizes[l + 1]);
            let input = &acts[l];
            let mut z = vec![0.0f64; n * l_out];

            for i in 0..n {
                for j in 0..l_out {
                    let mut s = m.biases[l][j] as f64;

                    for k in 0..l_in {
                        s += m.weights[l][j * l_in + k] as f64 * input[i * l_in + k];
                    }

                    z[i * l_out + j] = s;
                }
            }

            acts.push(z.iter().map(|&v| v.tanh()).collect());
            zs.push(z);
        }

        // Softmax cross-entropy on the raw output (the last z).
        let out = &zs[layers - 1];
        let n_out = m.n_out();
        let mut delta = vec![0.0f64; n * n_out];

        for i in 0..n {
            let row = &out[i * n_out..(i + 1) * n_out];
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = row.iter().map(|v| (v - max).exp()).sum();

            for j in 0..n_out {
                delta[i * n_out + j] = (row[j] - max).exp() / sum / n as f64;
            }

            delta[i * n_out + labels[i] as usize] -= 1.0 / n as f64;
        }

        let mut grads = vec![Vec::new(); layers];

        for l in (0..layers).rev() {
            let (l_in, l_out) = (m.sizes[l], m.sizes[l + 1]);
            let input = &acts[l];
            let mut g = vec![0.0f64; l_in * l_out];

            for i in 0..n {
                for j in 0..l_out {
                    for k in 0..l_in {
                        g[j * l_in + k] += delta[i * l_out + j] * input[i * l_in + k];
                    }
                }
            }

            grads[l] = g;

            if l > 0 {
                let mut prev = vec![0.0f64; n * l_in];

                for i in 0..n {
                    for k in 0..l_in {
                        let mut s = 0.0;

                        for j in 0..l_out {
                            s += delta[i * l_out + j] * m.weights[l][j * l_in + k] as f64;
                        }

                        let a = acts[l][i * l_in + k];

                        prev[i * l_in + k] = s * (1.0 - a * a);
                    }
                }

                delta = prev;
            }
        }

        grads
    }

    #[test]
    fn blocked_backward_matches_a_plain_reference() {
        let mut rng = ChaCha8Rng::seed_from_u64(9);

        // Layer widths that are and are not multiples of 4, batch sizes with every remainder.
        let m = Mlp::new(&[3, 9, 8, 7, 2], Activation::Tanh, 1.0, &mut rng);
        let x: Vec<f32> = (0..3 * 11)
            .map(|i| ((i * 37 % 19) as f32 - 9.0) / 7.0)
            .collect();
        let labels: Vec<u32> = (0..11).map(|i| (i * 5 % 2) as u32).collect();

        for n in 1..=11 {
            let (_, grads) = m.loss_and_grad(
                &x,
                TargetsRef::Labels(&labels),
                n,
                Objective::Classification { classes: 2 },
            );
            let reference = reference_grads(&m, &x, &labels, n);

            for (l, (g, r)) in grads.weights.iter().zip(&reference).enumerate() {
                for (k, (&a, &b)) in g.iter().zip(r).enumerate() {
                    assert!(
                        (a as f64 - b).abs() < 1e-5 + 1e-4 * b.abs(),
                        "n {n} layer {l} weight {k}: {a} vs {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn shapes_and_params() {
        let m = model(Activation::Tanh);

        assert_eq!(m.param_count(), (3 * 5 + 5) + (5 * 4 + 4) + (4 * 2 + 2));
        assert!(m.check_shape().is_ok());
        assert!(m.is_finite());
        assert_eq!(m.forward(&[0.0; 6], 2).len(), 4);
    }

    #[test]
    fn forward_into_matches_forward() {
        let m = model(Activation::Gelu);
        let (x, _, _) = batch();
        let mut ws = Workspace::new();
        let a = m.forward(&x, 4);
        let b = m.forward_into(&x, 4, &mut ws).map(|o| o.to_vec());

        assert_eq!(Some(a), b);
    }

    #[test]
    fn malformed_model_gives_nan_not_panic() {
        let mut m = model(Activation::Relu);

        m.weights[1].pop();

        assert!(m.check_shape().is_err());
        assert!(m.forward(&[0.0; 3], 1).iter().all(|v| v.is_nan()));

        let (loss, _) = m.loss_and_grad(
            &[0.0; 3],
            TargetsRef::Labels(&[0]),
            1,
            Objective::Classification { classes: 2 },
        );

        assert!(loss.is_nan());
    }

    #[test]
    fn short_input_and_bad_label_give_nan() {
        let m = model(Activation::Relu);

        assert!(m.forward(&[0.0; 2], 1).iter().all(|v| v.is_nan()));

        let (loss, _) = m.loss_and_grad(
            &[0.0; 3],
            TargetsRef::Labels(&[7]),
            1,
            Objective::Classification { classes: 2 },
        );

        assert!(loss.is_nan());

        let (loss, _) = m.loss_and_grad(
            &[0.0; 3],
            TargetsRef::Values(&[1.0]),
            1,
            Objective::Classification { classes: 2 },
        );

        assert!(loss.is_nan());
    }

    #[test]
    fn non_finite_weights_are_detected() {
        let mut m = model(Activation::Relu);

        m.biases[0][0] = f32::INFINITY;

        assert!(!m.is_finite());
    }

    #[test]
    fn every_optimizer_reduces_loss() {
        let (x, labels, _) = batch();
        let objective = Objective::Classification { classes: 2 };

        for kind in [Optimizer::Sgd, Optimizer::Momentum, Optimizer::Adam] {
            let mut m = model(Activation::Tanh);
            let mut opt = OptimizerState::new(kind, &m);
            let mut ws = Workspace::new();
            let mut grads = Gradients::zeros_like(&m);
            let lr = if kind == Optimizer::Adam { 0.01 } else { 0.1 };
            let first = m.loss_and_grad_into(
                &x,
                TargetsRef::Labels(&labels),
                4,
                objective,
                &mut ws,
                &mut grads,
            );

            for _ in 0..200 {
                m.loss_and_grad_into(
                    &x,
                    TargetsRef::Labels(&labels),
                    4,
                    objective,
                    &mut ws,
                    &mut grads,
                );
                opt.step(&mut m, &grads, lr, 0.0);
            }

            let last = m.loss_and_grad_into(
                &x,
                TargetsRef::Labels(&labels),
                4,
                objective,
                &mut ws,
                &mut grads,
            );

            assert!(last < first * 0.5, "{kind:?}: {first} -> {last}");
            assert_eq!(opt.step, 200);
        }
    }

    #[test]
    fn optimizer_buffers_never_hold_subnormal_floats() {
        for kind in [Optimizer::Momentum, Optimizer::Adam] {
            let mut m = model(Activation::Relu);
            let mut opt = OptimizerState::new(kind, &m);
            let zero = Gradients::zeros_like(&m);

            // Start every buffer just above the subnormal range, as for a dead ReLU unit
            // whose moments have decayed for a long time.
            for buf in opt
                .m_w
                .iter_mut()
                .chain(opt.m_b.iter_mut())
                .chain(opt.v_w.iter_mut())
                .chain(opt.v_b.iter_mut())
            {
                buf.fill(1.2e-38);
            }

            for _ in 0..100 {
                opt.step(&mut m, &zero, 0.01, 0.0);

                let mut all = opt
                    .m_w
                    .iter()
                    .chain(&opt.m_b)
                    .chain(&opt.v_w)
                    .chain(&opt.v_b)
                    .flatten();

                assert!(all.all(|v| !v.is_subnormal()), "{kind:?}");
            }

            assert!(m.is_finite());
        }
    }

    #[test]
    fn weight_decay_shrinks_weights() {
        let mut m = model(Activation::Relu);
        let before: f32 = m.weights[0].iter().map(|w| w.abs()).sum();
        let grads = Gradients::zeros_like(&m);
        let mut opt = OptimizerState::new(Optimizer::Sgd, &m);

        opt.step(&mut m, &grads, 0.1, 0.01);

        let after: f32 = m.weights[0].iter().map(|w| w.abs()).sum();

        assert!((after - before * 0.999).abs() < 1e-4);
    }

    #[test]
    fn init_is_deterministic_and_scaled() {
        let mut r1 = ChaCha8Rng::seed_from_u64(3);
        let mut r2 = ChaCha8Rng::seed_from_u64(3);
        let a = Mlp::new(&[64, 64, 2], Activation::Relu, 1.0, &mut r1);
        let b = Mlp::new(&[64, 64, 2], Activation::Relu, 1.0, &mut r2);

        assert_eq!(a, b);

        let w = &a.weights[0];
        let var = w.iter().map(|v| (v * v) as f64).sum::<f64>() / w.len() as f64;

        // He init: var = 2 / fan_in = 0.03125.
        assert!((var - 0.03125).abs() < 0.005, "{var}");
        assert!(a.biases.iter().all(|b| b.iter().all(|&v| v == 0.0)));
    }
}
