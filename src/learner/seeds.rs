//! The four hand-written warm-start programs.
//!
//! Owner: WP1. Contract section 5.3. Pure: no I/O.
//!
//! Register use (section 6.2): v0 = input, v1 = prediction (linked before Learn),
//! v2 / s0 = label (Learn only), s3 / s4 = learning-rate style constants.
#![forbid(unsafe_code)]

use crate::learner::types::{Instr, LearnerGenome, OpCode};
use crate::util::sha256_hex;

/// Shorthand for one instruction.
fn op(op: OpCode, a: u8, b: u8, out: u8, c: f32) -> Instr {
    Instr { op, a, b, out, c }
}

/// linear-sgd: one linear layer, delta rule on the linked output.
fn linear_sgd() -> LearnerGenome {
    LearnerGenome {
        setup: vec![op(OpCode::SConst, 0, 0, 3, 0.05)],
        predict: vec![op(OpCode::MatVec, 0, 0, 1, 0.0)],
        learn: vec![
            op(OpCode::VSub, 2, 1, 3, 0.0),
            op(OpCode::VScale, 3, 3, 3, 0.0),
            op(OpCode::MOuter, 3, 0, 1, 0.0),
            op(OpCode::MAdd, 0, 1, 0, 0.0),
        ],
    }
}

/// nn2-sgd: two F x F layers with ReLU, online backpropagation.
fn nn2_sgd() -> LearnerGenome {
    LearnerGenome {
        setup: vec![
            op(OpCode::MGaussian, 0, 0, 0, 0.3),
            op(OpCode::MGaussian, 0, 0, 1, 0.3),
            op(OpCode::SConst, 0, 0, 3, 0.02),
        ],
        predict: vec![
            op(OpCode::MatVec, 0, 0, 4, 0.0),
            op(OpCode::VMax0, 4, 0, 5, 0.0),
            op(OpCode::MatVec, 1, 5, 1, 0.0),
        ],
        learn: vec![
            op(OpCode::VSub, 2, 1, 3, 0.0),
            op(OpCode::VScale, 3, 3, 3, 0.0),
            op(OpCode::MOuter, 3, 5, 2, 0.0),
            op(OpCode::MatTVec, 1, 3, 6, 0.0),
            op(OpCode::VHeaviside, 4, 0, 7, 0.0),
            op(OpCode::VMul, 6, 7, 6, 0.0),
            op(OpCode::MAdd, 1, 2, 1, 0.0),
            op(OpCode::MOuter, 6, 0, 3, 0.0),
            op(OpCode::MAdd, 0, 3, 0, 0.0),
        ],
    }
}

/// prototype: Hebbian accumulation of label x input (class means).
fn prototype() -> LearnerGenome {
    LearnerGenome {
        setup: vec![op(OpCode::SConst, 0, 0, 3, 0.05)],
        predict: vec![op(OpCode::MatVec, 0, 0, 1, 0.0)],
        learn: vec![
            op(OpCode::MOuter, 2, 0, 1, 0.0),
            op(OpCode::MScale, 3, 1, 1, 0.0),
            op(OpCode::MAdd, 0, 1, 0, 0.0),
        ],
    }
}

/// perceptron: delta rule only where |label - output| > 0.5 (mistakes).
fn perceptron() -> LearnerGenome {
    LearnerGenome {
        setup: vec![
            op(OpCode::SConst, 0, 0, 3, 0.1),
            op(OpCode::SConst, 0, 0, 4, 0.5),
        ],
        predict: vec![op(OpCode::MatVec, 0, 0, 1, 0.0)],
        learn: vec![
            op(OpCode::VSub, 2, 1, 3, 0.0),
            op(OpCode::VAbs, 3, 0, 4, 0.0),
            op(OpCode::VBroadcast, 4, 0, 5, 0.0),
            op(OpCode::VSub, 4, 5, 4, 0.0),
            op(OpCode::VHeaviside, 4, 0, 4, 0.0),
            op(OpCode::VMul, 3, 4, 3, 0.0),
            op(OpCode::VScale, 3, 3, 3, 0.0),
            op(OpCode::MOuter, 3, 0, 1, 0.0),
            op(OpCode::MAdd, 0, 1, 0, 0.0),
        ],
    }
}

/// The four warm-start programs, in this fixed order:
/// "linear-sgd", "nn2-sgd", "prototype", "perceptron".
pub fn seed_programs() -> Vec<(&'static str, LearnerGenome)> {
    vec![
        ("linear-sgd", linear_sgd()),
        ("nn2-sgd", nn2_sgd()),
        ("prototype", prototype()),
        ("perceptron", perceptron()),
    ]
}

/// The 2-layer ReLU NN seed ("nn2-sgd"); its step cost is reference_step_cost().
pub fn reference_nn() -> LearnerGenome {
    nn2_sgd()
}

/// SHA-256 over the canonical JSON of all seeds (part of the protocol hash).
/// The JSON is `[["linear-sgd",{...}],["nn2-sgd",{...}],...]` in seed order.
pub fn seeds_hash() -> String {
    // Serializing plain data cannot fail; the Debug fallback only keeps it panic-free.
    let seeds = seed_programs();
    let text = match serde_json::to_string(&seeds) {
        Ok(t) => t,
        Err(_) => format!("{seeds:?}"),
    };

    sha256_hex(text.as_bytes())
}

#[cfg(test)]
mod tests {
    use std::ops::{Add, Div, Mul, Sub};

    use super::*;
    use crate::learner::dsl::reference_step_cost;
    use crate::learner::interp::{Memory, run, testkit};
    use crate::learner::types::{
        CHECKPOINT_DIVISORS, DslLimits, Labels, LearnKind, MAX_ABS_TRAIN_ERROR, SUPPORTED_F,
        TaskInstance,
    };
    use crate::util::derive_seed;

    #[test]
    fn seeds_are_valid_and_ordered() {
        let names: Vec<&str> = seed_programs().iter().map(|(n, _)| *n).collect();

        assert_eq!(names, ["linear-sgd", "nn2-sgd", "prototype", "perceptron"]);
        assert_eq!(reference_nn(), seed_programs()[1].1);

        for (name, g) in seed_programs() {
            assert!(g.check(&DslLimits::default()).is_ok(), "{name}");
            assert_eq!(g.simplify(), g, "{name} has no dead code");
        }

        let hash = seeds_hash();

        assert_eq!(hash, seeds_hash());
        assert_eq!(hash.len(), 64);

        // Golden value: a change here changes the protocol hash.
        assert_eq!(
            hash,
            "e2706c7552403774627ccf470f8df9a2064414ae77cc474ce45019278941485f"
        );
    }

    /// T-SEED-2: nn2-sgd step cost == 10 F^2 + 5 F.
    #[test]
    fn t_seed_2_reference_step_cost() {
        let nn = reference_nn();

        for f in SUPPORTED_F {
            let fu = f as u64;

            assert_eq!(nn.step_cost(f), 10 * fu * fu + 5 * fu);
            assert_eq!(nn.step_cost(f), reference_step_cost(f));
        }
    }

    /// Number type of the plain-Rust reference learners.
    trait Real:
        Copy
        + PartialOrd
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
    {
        const ZERO: Self;
        const ONE: Self;

        fn of(x: f32) -> Self;
        fn wide(self) -> f64;
        fn exp(self) -> Self;
        fn finite(self) -> bool;

        /// Dot product. f32 uses the interpreter's eight partial sums, f64 a plain sum.
        fn dot(x: &[Self], y: &[Self]) -> Self;
    }

    impl Real for f32 {
        const ZERO: Self = 0.0;
        const ONE: Self = 1.0;

        fn of(x: f32) -> Self {
            x
        }

        fn wide(self) -> f64 {
            self as f64
        }

        fn exp(self) -> Self {
            f32::exp(self)
        }

        fn finite(self) -> bool {
            self.is_finite()
        }

        fn dot(x: &[Self], y: &[Self]) -> Self {
            let mut lanes = [0.0f32; 8];
            let full = x.len() / 8 * 8;

            for (i, (p, q)) in x[..full].iter().zip(&y[..full]).enumerate() {
                lanes[i % 8] += p * q;
            }

            let quad = [
                lanes[0] + lanes[4],
                lanes[1] + lanes[5],
                lanes[2] + lanes[6],
                lanes[3] + lanes[7],
            ];
            let mut total = (quad[0] + quad[2]) + (quad[1] + quad[3]);

            for (p, q) in x[full..].iter().zip(&y[full..]) {
                total += p * q;
            }

            total
        }
    }

    impl Real for f64 {
        const ZERO: Self = 0.0;
        const ONE: Self = 1.0;

        fn of(x: f32) -> Self {
            x as f64
        }

        fn wide(self) -> f64 {
            self
        }

        fn exp(self) -> Self {
            f64::exp(self)
        }

        fn finite(self) -> bool {
            self.is_finite()
        }

        fn dot(x: &[Self], y: &[Self]) -> Self {
            x.iter().zip(y).map(|(p, q)| p * q).sum()
        }
    }

    /// Plain-Rust learner: the textbook form of linear-sgd or nn2-sgd.
    #[derive(Clone)]
    enum Reference<T> {
        Linear {
            w: Vec<Vec<T>>,
            lr: T,
        },
        Nn {
            w1: Vec<Vec<T>>,
            w2: Vec<Vec<T>>,
            lr: T,
            hidden: Vec<T>,
            active: Vec<T>,
        },
    }

    fn mat_vec<T: Real>(w: &[Vec<T>], x: &[T]) -> Vec<T> {
        w.iter().map(|row| T::dot(row, x)).collect()
    }

    /// w += u (x) x, element by element.
    fn add_outer<T: Real>(w: &mut [Vec<T>], u: &[T], x: &[T]) {
        for (row, &ui) in w.iter_mut().zip(u) {
            for (wij, &xj) in row.iter_mut().zip(x) {
                *wij = *wij + ui * xj;
            }
        }
    }

    impl<T: Real> Reference<T> {
        /// Raw output (before the link).
        fn predict(&mut self, x: &[T]) -> Vec<T> {
            match self {
                Reference::Linear { w, .. } => mat_vec(w, x),

                Reference::Nn {
                    w1,
                    w2,
                    hidden,
                    active,
                    ..
                } => {
                    *hidden = mat_vec(w1, x);
                    *active = hidden
                        .iter()
                        .map(|&h| if h > T::ZERO { h } else { T::ZERO })
                        .collect();

                    mat_vec(w2, active)
                }
            }
        }

        /// One SGD step on the squared error of the linked output.
        fn learn(&mut self, x: &[T], linked: &[T], target: &[T]) {
            match self {
                Reference::Linear { w, lr } => {
                    let err: Vec<T> = target
                        .iter()
                        .zip(linked)
                        .map(|(&t, &p)| *lr * (t - p))
                        .collect();

                    add_outer(w, &err, x);
                }

                Reference::Nn {
                    w1,
                    w2,
                    lr,
                    hidden,
                    active,
                } => {
                    let err: Vec<T> = target
                        .iter()
                        .zip(linked)
                        .map(|(&t, &p)| *lr * (t - p))
                        .collect();

                    // Backpropagate through the old w2, gated by the ReLU.
                    let mut back = vec![T::ZERO; err.len()];

                    for (row, &e) in w2.iter().zip(&err) {
                        for (b, &wij) in back.iter_mut().zip(row) {
                            *b = *b + wij * e;
                        }
                    }

                    for (b, &h) in back.iter_mut().zip(hidden.iter()) {
                        let gate = if h > T::ZERO { T::ONE } else { T::ZERO };

                        *b = *b * gate;
                    }

                    add_outer(w2, &err, active);
                    add_outer(w1, &back, x);
                }
            }
        }
    }

    fn link<T: Real>(kind: LearnKind, y: &mut [T]) {
        match kind {
            LearnKind::Regression => {}
            LearnKind::Binary => y[0] = T::ONE / (T::ONE + (T::ZERO - y[0]).exp()),

            LearnKind::Multiclass { k } => {
                let k = k as usize;
                let top = y[..k]
                    .iter()
                    .fold(T::of(f32::NEG_INFINITY), |a, &b| if b > a { b } else { a });
                let mut total = T::ZERO;

                for v in y[..k].iter_mut() {
                    *v = (*v - top).exp();
                    total = total + *v;
                }

                for v in y[..k].iter_mut() {
                    *v = *v / total;
                }
            }
        }
    }

    /// The label vector v2: value in slot 0, or one-hot.
    fn target<T: Real>(kind: LearnKind, y: f32, f: usize) -> Vec<T> {
        let mut t = vec![T::ZERO; f];

        match kind {
            LearnKind::Multiclass { .. } => t[y as usize] = T::ONE,
            _ => t[0] = T::of(y),
        }

        t
    }

    fn row<T: Real>(x: &[f32], i: usize, f: usize) -> Vec<T> {
        x[i * f..(i + 1) * f].iter().map(|&v| T::of(v)).collect()
    }

    /// The section 6.2 stream with a plain-Rust learner: (recorded outputs, aborted).
    fn reference_drive<T: Real>(mut model: Reference<T>, task: &TaskInstance) -> (Vec<f64>, bool) {
        let f = task.f;
        let k = task.kind.k_out();
        let n = task.n_train;
        let checkpoints = CHECKPOINT_DIVISORS.map(|d| task.steps / d);
        let mut outputs = Vec::new();
        let mut order = Vec::new();

        for s in 0..task.steps {
            if s % n == 0 {
                let seed = derive_seed(task.seed, &format!("stream/{}", s / n));

                order = testkit::permutation(n, seed);
            }

            let i = order[s % n];
            let x = row::<T>(&task.x_train, i, f);
            let mut y = model.predict(&x);

            link(task.kind, &mut y);

            if !y[..k].iter().all(|v| v.finite()) {
                return (outputs, true);
            }

            let label = testkit::label(&task.y_train, i);
            let error = (y[0].wide() - label as f64).abs();

            if task.kind == LearnKind::Regression && error > MAX_ABS_TRAIN_ERROR as f64 {
                return (outputs, true);
            }

            outputs.extend(y[..k].iter().map(|v| v.wide()));
            model.learn(&x, &y, &target(task.kind, label, f));

            for _ in checkpoints.iter().filter(|&&c| c == s + 1) {
                let mut probe = model.clone();

                for j in 0..task.n_valid {
                    let mut y = probe.predict(&row(&task.x_valid, j, f));

                    link(task.kind, &mut y);

                    if !y[..k].iter().all(|v| v.finite()) {
                        return (outputs, true);
                    }

                    outputs.extend(y[..k].iter().map(|v| v.wide()));
                }
            }
        }

        (outputs, false)
    }

    /// Start state of the reference learner `name`. nn2-sgd takes its initial weights from
    /// running its Setup exactly as the protocol does (same random stream).
    fn start<T: Real, const F: usize>(name: &str, task: &TaskInstance) -> Reference<T> {
        let matrix = |m: &[[f32; F]; F]| -> Vec<Vec<T>> {
            m.iter()
                .map(|r| r.iter().map(|&v| T::of(v)).collect())
                .collect()
        };

        if name == "linear-sgd" {
            return Reference::Linear {
                w: vec![vec![T::ZERO; F]; F],
                lr: T::of(0.05),
            };
        }

        let mut mem = Memory::<F>::new(derive_seed(task.seed, "interp/0"));

        run(&reference_nn().setup, &mut mem);

        Reference::Nn {
            w1: matrix(&mem.m[0]),
            w2: matrix(&mem.m[1]),
            lr: T::of(0.02),
            hidden: Vec::new(),
            active: Vec::new(),
        }
    }

    fn start_any<T: Real>(name: &str, task: &TaskInstance) -> Reference<T> {
        match task.f {
            8 => start::<T, 8>(name, task),
            16 => start::<T, 16>(name, task),
            _ => start::<T, 32>(name, task),
        }
    }

    fn seed(name: &str) -> LearnerGenome {
        seed_programs()
            .into_iter()
            .find(|(n, _)| *n == name)
            .map(|(_, g)| g)
            .unwrap_or_default()
    }

    /// The 20 test tasks: F cycles 8, 16, 32; four kinds; n_train 16 or 64; 384 steps.
    fn tasks() -> Vec<TaskInstance> {
        let kinds = [
            LearnKind::Regression,
            LearnKind::Binary,
            LearnKind::Multiclass { k: 3 },
            LearnKind::Multiclass { k: 5 },
        ];

        (0..20usize)
            .map(|t| {
                let f = SUPPORTED_F[t % 3];
                let n_train = [16, 64][t % 2];

                testkit::task(1000 + t as u64, f, kinds[t % 4], n_train, 16, 384)
            })
            .collect()
    }

    /// Largest |interpreter - reference| over all recorded outputs; asserts equal length and
    /// equal abort status.
    fn compare<T: Real>(name: &str, task: &TaskInstance) -> f64 {
        let got = testkit::drive(&seed(name), task, 0);
        let (want, aborted) = reference_drive(start_any::<T>(name, task), task);

        assert_eq!(
            got.abort.is_some(),
            aborted,
            "{name} task {}: abort status",
            task.seed
        );
        assert_eq!(got.outputs.len(), want.len(), "{name} task {}", task.seed);

        got.outputs
            .iter()
            .zip(&want)
            .map(|(g, w)| (*g as f64 - w).abs())
            .fold(0.0, f64::max)
    }

    /// T-SEED-1: linear-sgd and nn2-sgd equal a plain-Rust reference (abs <= 1e-5) on 20
    /// tasks at F = 8, 16, 32. The reference works in f32 and sums dot products in the
    /// interpreter's order, because the untuned seeds are numerically unstable on some F = 32
    /// tasks, where any other rounding order drifts apart (see the f64 test below).
    #[test]
    fn t_seed_1_seeds_match_plain_rust() {
        for task in tasks() {
            for name in ["linear-sgd", "nn2-sgd"] {
                let worst = compare::<f32>(name, &task);

                assert!(worst <= 1e-5, "{name} task {}: {worst}", task.seed);
            }
        }
    }

    /// Order-independent check: an f64 reference with plain sums agrees within 1e-5 on the
    /// same kind of tasks wherever the untuned seed is numerically stable (F = 8, 16).
    #[test]
    fn seeds_match_an_f64_reference_where_stable() {
        let mut checked = 0;

        for task in tasks().into_iter().filter(|t| t.f <= 16) {
            for name in ["linear-sgd", "nn2-sgd"] {
                let worst = compare::<f64>(name, &task);

                assert!(worst <= 1e-5, "{name} task {}: {worst}", task.seed);

                checked += 1;
            }
        }

        assert_eq!(checked, 28);
    }

    /// Sanity: three seeds learn a linearly separable binary task well above chance (0.7).
    #[test]
    fn seeds_learn_a_linear_binary_task() {
        for t in 0..4 {
            let task = testkit::task(500 + t, 16, LearnKind::Binary, 64, 64, 384);

            let labels = match &task.y_valid {
                Labels::Classes(c) => c.clone(),
                Labels::Values(_) => Vec::new(),
            };

            for name in ["linear-sgd", "nn2-sgd", "prototype"] {
                let trace = testkit::drive(&seed(name), &task, 0);
                let last = &trace.outputs[trace.outputs.len() - task.n_valid..];
                let hits = last
                    .iter()
                    .zip(&labels)
                    .filter(|(p, y)| u8::from(**p >= 0.5) == **y)
                    .count();
                let accuracy = hits as f64 / task.n_valid as f64;

                assert!(trace.abort.is_none(), "{name}");
                assert!(accuracy >= 0.7, "{name} task {t}: accuracy {accuracy}");
            }
        }
    }

    /// Documents a property of the normative perceptron seed: on Binary tasks it never
    /// updates from its zero start, since sigmoid(0) = 0.5 gives |y - p| = 0.5, which is not
    /// above its 0.5 margin. It does learn on multiclass tasks.
    #[test]
    fn perceptron_stalls_on_binary_but_learns_multiclass() {
        let p = seed("perceptron");
        let binary = testkit::task(600, 16, LearnKind::Binary, 64, 16, 384);
        let trace = testkit::drive(&p, &binary, 0);

        assert!(trace.outputs.iter().all(|&o| o == 0.5));

        let multi = testkit::task(601, 16, LearnKind::Multiclass { k: 3 }, 64, 64, 384);
        let trace = testkit::drive(&p, &multi, 0);
        let labels = match &multi.y_valid {
            Labels::Classes(c) => c.clone(),
            Labels::Values(_) => Vec::new(),
        };
        let last = &trace.outputs[trace.outputs.len() - 3 * multi.n_valid..];
        let hits = last
            .chunks(3)
            .zip(&labels)
            .filter(|(probs, y)| {
                let best = (0..3).fold(0, |b, j| if probs[j] > probs[b] { j } else { b });

                best == **y as usize
            })
            .count();

        assert!(hits as f64 / multi.n_valid as f64 >= 0.6, "hits {hits}");
    }
}
