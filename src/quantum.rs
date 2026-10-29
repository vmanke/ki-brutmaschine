//! Exact state-vector simulator for small qubit circuits (issue 02 of
//! `.scratch/universum-erfinden/`, decisions Q39 and Q41).
//!
//! Pure Rust, no new dependency: a small `Complex` type, a state vector, a gate
//! list, and the full unitary for small `n`. The gate set is Clifford+T
//! (`H`, `S`, `S-dagger`, `T`, `T-dagger`, `X`, `Z`, `CNOT`); the cost measures
//! are T-count, CNOT-count and depth (Q41). Equivalence of two circuits is
//! tested up to a global phase with a fidelity threshold of `1 - 1e-9` (Q38).
//!
//! Qubit `0` is the least significant bit: basis state `k` has qubit `b` set
//! when `(k >> b) & 1 == 1`. Everything here is deterministic; the simulator
//! holds no randomness.

use std::ops::{Add, Mul, Sub};

use serde::{Deserialize, Serialize};

use crate::util::{BrutError, Result};

/// Largest qubit count the simulator accepts here. The host runs a wider cap
/// (about 25, Q39); a state vector for 12 qubits is 4096 amplitudes.
pub const MAX_QUBITS: usize = 12;

/// Largest number of gates in one circuit. A hard ceiling that bounds work and
/// keeps a bred gate list from growing without limit.
pub const MAX_GATES: usize = 1 << 16;

/// Largest qubit count for which the full unitary is built. The matrix holds
/// `4^n` complex numbers, so 10 qubits is about 16 MiB and the ceiling here.
pub const MAX_UNITARY_QUBITS: usize = 10;

/// Default fidelity tolerance for "equal up to a global phase" (Q38).
pub const DEFAULT_FIDELITY_TOL: f64 = 1e-9;

/// A complex number. Own type so the crate gains no dependency (Q39).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    pub const ONE: Complex = Complex { re: 1.0, im: 0.0 };
    pub const I: Complex = Complex { re: 0.0, im: 1.0 };

    pub const fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }

    /// Complex conjugate.
    pub fn conj(self) -> Self {
        Complex::new(self.re, -self.im)
    }

    /// Squared magnitude `|z|^2`. Always finite for finite inputs.
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Multiply by a real scalar.
    pub fn scale(self, s: f64) -> Self {
        Complex::new(self.re * s, self.im * s)
    }
}

impl Add for Complex {
    type Output = Complex;

    fn add(self, o: Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}

impl Sub for Complex {
    type Output = Complex;

    fn sub(self, o: Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}

impl Mul for Complex {
    type Output = Complex;

    fn mul(self, o: Complex) -> Complex {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

/// One gate on named qubit lines. A circuit is a list of these (Q41).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Gate {
    H(usize),
    S(usize),
    Sdg(usize),
    T(usize),
    Tdg(usize),
    X(usize),
    Z(usize),
    Cnot { control: usize, target: usize },
}

impl Gate {
    /// The 2x2 matrix of a single-qubit gate, row-major `[m00, m01, m10, m11]`.
    /// `CNOT` has no single-qubit matrix and returns `None`.
    fn matrix(self) -> Option<[Complex; 4]> {
        let c = std::f64::consts::FRAC_1_SQRT_2;

        let m = match self {
            Gate::H(_) => [
                Complex::new(c, 0.0),
                Complex::new(c, 0.0),
                Complex::new(c, 0.0),
                Complex::new(-c, 0.0),
            ],
            Gate::S(_) => [Complex::ONE, Complex::ZERO, Complex::ZERO, Complex::I],
            Gate::Sdg(_) => [
                Complex::ONE,
                Complex::ZERO,
                Complex::ZERO,
                Complex::new(0.0, -1.0),
            ],
            Gate::T(_) => [
                Complex::ONE,
                Complex::ZERO,
                Complex::ZERO,
                Complex::new(c, c),
            ],
            Gate::Tdg(_) => [
                Complex::ONE,
                Complex::ZERO,
                Complex::ZERO,
                Complex::new(c, -c),
            ],
            Gate::X(_) => [Complex::ZERO, Complex::ONE, Complex::ONE, Complex::ZERO],
            Gate::Z(_) => [
                Complex::ONE,
                Complex::ZERO,
                Complex::ZERO,
                Complex::new(-1.0, 0.0),
            ],
            Gate::Cnot { .. } => return None,
        };

        Some(m)
    }

    /// The highest qubit index the gate touches, used for bounds checks.
    fn max_qubit(self) -> usize {
        match self {
            Gate::H(q)
            | Gate::S(q)
            | Gate::Sdg(q)
            | Gate::T(q)
            | Gate::Tdg(q)
            | Gate::X(q)
            | Gate::Z(q) => q,
            Gate::Cnot { control, target } => control.max(target),
        }
    }
}

/// A pure quantum state over `n` qubits: `2^n` complex amplitudes.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    n: usize,
    amps: Vec<Complex>,
}

impl State {
    /// The all-zero state `|0...0>` over `n` qubits.
    pub fn zero(n: usize) -> Result<Self> {
        Self::basis(n, 0)
    }

    /// The computational basis state `|k>` over `n` qubits.
    pub fn basis(n: usize, k: usize) -> Result<Self> {
        if n > MAX_QUBITS {
            return Err(BrutError::Invalid(format!(
                "quantum: {n} qubits over the cap of {MAX_QUBITS}"
            )));
        }

        let dim = 1usize << n;

        if k >= dim {
            return Err(BrutError::Invalid(format!(
                "quantum: basis index {k} outside 0..{dim}"
            )));
        }

        let mut amps = vec![Complex::ZERO; dim];
        amps[k] = Complex::ONE;

        Ok(State { n, amps })
    }

    pub fn n_qubits(&self) -> usize {
        self.n
    }

    pub fn amps(&self) -> &[Complex] {
        &self.amps
    }

    /// Apply one gate in place. Validates qubit indices against `n`.
    pub fn apply(&mut self, gate: Gate) -> Result<()> {
        if gate.max_qubit() >= self.n {
            return Err(BrutError::Invalid(format!(
                "quantum: gate {gate:?} touches a qubit outside 0..{}",
                self.n
            )));
        }

        match gate {
            Gate::Cnot { control, target } => {
                if control == target {
                    return Err(BrutError::Invalid(
                        "quantum: CNOT needs distinct control and target".to_string(),
                    ));
                }

                self.apply_cnot(control, target);
            }
            single => {
                // `single` is never CNOT here, so `matrix` is always `Some`.
                let Some(m) = single.matrix() else {
                    return Err(BrutError::Invalid(format!(
                        "quantum: {single:?} has no single-qubit matrix"
                    )));
                };
                let q = single.max_qubit();

                self.apply_1q(q, m);
            }
        }

        Ok(())
    }

    /// Apply a single-qubit matrix `m` (row-major) to qubit `q`.
    fn apply_1q(&mut self, q: usize, m: [Complex; 4]) {
        let bit = 1usize << q;

        for i in 0..self.amps.len() {
            if i & bit == 0 {
                let j = i | bit;
                let a0 = self.amps[i];
                let a1 = self.amps[j];

                self.amps[i] = m[0] * a0 + m[1] * a1;
                self.amps[j] = m[2] * a0 + m[3] * a1;
            }
        }
    }

    /// Apply `CNOT`: where the control bit is set, flip the target bit.
    fn apply_cnot(&mut self, control: usize, target: usize) {
        let cb = 1usize << control;
        let tb = 1usize << target;

        for i in 0..self.amps.len() {
            if i & cb != 0 && i & tb == 0 {
                self.amps.swap(i, i | tb);
            }
        }
    }
}

/// A list of gates over a fixed number of qubits.
#[derive(Clone, Debug, PartialEq)]
pub struct Circuit {
    n: usize,
    gates: Vec<Gate>,
}

impl Circuit {
    /// An empty circuit over `n` qubits (the identity).
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 || n > MAX_QUBITS {
            return Err(BrutError::Invalid(format!(
                "quantum: {n} qubits outside 1..={MAX_QUBITS}"
            )));
        }

        Ok(Circuit {
            n,
            gates: Vec::new(),
        })
    }

    /// Build a circuit from a gate list, validating every gate (Q41).
    pub fn from_gates(n: usize, gates: Vec<Gate>) -> Result<Self> {
        let mut c = Circuit::new(n)?;

        for g in gates {
            c.push(g)?;
        }

        Ok(c)
    }

    /// Append one gate, checking the gate cap and the qubit indices.
    pub fn push(&mut self, gate: Gate) -> Result<&mut Self> {
        if self.gates.len() >= MAX_GATES {
            return Err(BrutError::Invalid(format!(
                "quantum: circuit over the gate cap of {MAX_GATES}"
            )));
        }

        if gate.max_qubit() >= self.n {
            return Err(BrutError::Invalid(format!(
                "quantum: gate {gate:?} touches a qubit outside 0..{}",
                self.n
            )));
        }

        if let Gate::Cnot { control, target } = gate
            && control == target
        {
            return Err(BrutError::Invalid(
                "quantum: CNOT needs distinct control and target".to_string(),
            ));
        }

        self.gates.push(gate);

        Ok(self)
    }

    pub fn n_qubits(&self) -> usize {
        self.n
    }

    pub fn gates(&self) -> &[Gate] {
        &self.gates
    }

    /// Run the circuit on a start state and return the result.
    pub fn run(&self, start: &State) -> Result<State> {
        if start.n_qubits() != self.n {
            return Err(BrutError::Invalid(format!(
                "quantum: state has {} qubits, circuit has {}",
                start.n_qubits(),
                self.n
            )));
        }

        let mut st = start.clone();

        for g in &self.gates {
            st.apply(*g)?;
        }

        Ok(st)
    }

    /// Run the circuit on `|0...0>`.
    pub fn state(&self) -> Result<State> {
        self.run(&State::zero(self.n)?)
    }

    /// The full `2^n x 2^n` unitary, column `k` being the circuit applied to
    /// `|k>`. Refused above `MAX_UNITARY_QUBITS` to bound memory.
    pub fn unitary(&self) -> Result<Unitary> {
        if self.n > MAX_UNITARY_QUBITS {
            return Err(BrutError::Invalid(format!(
                "quantum: unitary for {} qubits over the cap of {MAX_UNITARY_QUBITS}",
                self.n
            )));
        }

        let dim = 1usize << self.n;
        let mut data = vec![Complex::ZERO; dim * dim];

        for k in 0..dim {
            let col = self.run(&State::basis(self.n, k)?)?;

            for (row, amp) in col.amps().iter().enumerate() {
                data[row * dim + k] = *amp;
            }
        }

        Ok(Unitary {
            n: self.n,
            dim,
            data,
        })
    }

    /// Number of `T` and `T-dagger` gates (the Clifford+T cost measure, Q41).
    pub fn t_count(&self) -> usize {
        self.gates
            .iter()
            .filter(|g| matches!(g, Gate::T(_) | Gate::Tdg(_)))
            .count()
    }

    /// Number of `CNOT` gates.
    pub fn cnot_count(&self) -> usize {
        self.gates
            .iter()
            .filter(|g| matches!(g, Gate::Cnot { .. }))
            .count()
    }

    /// Circuit depth: the longest chain of gates that share a qubit, computed by
    /// as-soon-as-possible layering. An empty circuit has depth 0.
    pub fn depth(&self) -> usize {
        let mut layer = vec![0usize; self.n];

        for g in &self.gates {
            match *g {
                Gate::Cnot { control, target } => {
                    let next = layer[control].max(layer[target]) + 1;
                    layer[control] = next;
                    layer[target] = next;
                }
                single => {
                    let q = single.max_qubit();
                    layer[q] += 1;
                }
            }
        }

        layer.into_iter().max().unwrap_or(0)
    }
}

/// A dense `2^n x 2^n` unitary matrix, row-major.
#[derive(Clone, Debug)]
pub struct Unitary {
    n: usize,
    dim: usize,
    data: Vec<Complex>,
}

impl Unitary {
    pub fn n_qubits(&self) -> usize {
        self.n
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Entry at `row`, `col`, or `None` if out of range.
    pub fn get(&self, row: usize, col: usize) -> Option<Complex> {
        if row < self.dim && col < self.dim {
            Some(self.data[row * self.dim + col])
        } else {
            None
        }
    }

    /// Fidelity to another operator, ignoring a global phase:
    /// `|<U, V>|^2 / d^2` with the Frobenius inner product `<U, V> = Tr(U^dag V)`.
    /// It is `1` exactly when the two agree up to a global phase, and both must
    /// act on the same number of qubits.
    pub fn fidelity(&self, other: &Unitary) -> Result<f64> {
        if self.dim != other.dim {
            return Err(BrutError::Invalid(format!(
                "quantum: fidelity of a {}-qubit and a {}-qubit operator",
                self.n, other.n
            )));
        }

        let mut acc = Complex::ZERO;

        for (a, b) in self.data.iter().zip(other.data.iter()) {
            acc = acc + a.conj() * *b;
        }

        let d = self.dim as f64;

        Ok(acc.norm_sqr() / (d * d))
    }

    /// Whether two circuits' unitaries agree up to a global phase, within `tol`
    /// of fidelity 1 (Q38; `DEFAULT_FIDELITY_TOL` is the standard value).
    pub fn equivalent(&self, other: &Unitary, tol: f64) -> Result<bool> {
        Ok(self.fidelity(other)? >= 1.0 - tol)
    }
}

/// State fidelity `|<a|b>|^2`, ignoring a global phase. Both states must have
/// the same number of qubits.
pub fn state_fidelity(a: &State, b: &State) -> Result<f64> {
    if a.n_qubits() != b.n_qubits() {
        return Err(BrutError::Invalid(format!(
            "quantum: state fidelity of a {}-qubit and a {}-qubit state",
            a.n_qubits(),
            b.n_qubits()
        )));
    }

    let mut acc = Complex::ZERO;

    for (x, y) in a.amps().iter().zip(b.amps().iter()) {
        acc = acc + x.conj() * *y;
    }

    Ok(acc.norm_sqr())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = DEFAULT_FIDELITY_TOL;

    fn unitary_of(n: usize, gates: &[Gate]) -> Unitary {
        Circuit::from_gates(n, gates.to_vec())
            .expect("valid circuit")
            .unitary()
            .expect("unitary within cap")
    }

    #[test]
    fn complex_arithmetic_is_standard() {
        let a = Complex::new(1.0, 1.0);
        let b = Complex::new(1.0, -1.0);

        assert_eq!(a * b, Complex::new(2.0, 0.0));
        assert_eq!(a.conj(), b);
        assert!((a.norm_sqr() - 2.0).abs() < 1e-12);
        assert_eq!(Complex::I * Complex::I, Complex::new(-1.0, 0.0));
    }

    #[test]
    fn x_flips_the_basis() {
        let st = Circuit::from_gates(1, vec![Gate::X(0)])
            .expect("circuit")
            .state()
            .expect("state");

        assert_eq!(st.amps(), &[Complex::ZERO, Complex::ONE]);
    }

    #[test]
    fn h_builds_an_equal_superposition() {
        let st = Circuit::from_gates(1, vec![Gate::H(0)])
            .expect("circuit")
            .state()
            .expect("state");
        let c = std::f64::consts::FRAC_1_SQRT_2;

        assert!((st.amps()[0].re - c).abs() < 1e-12);
        assert!((st.amps()[1].re - c).abs() < 1e-12);

        let norm: f64 = st.amps().iter().map(|z| z.norm_sqr()).sum();

        assert!((norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn hzh_equals_x() {
        let hzh = unitary_of(1, &[Gate::H(0), Gate::Z(0), Gate::H(0)]);
        let x = unitary_of(1, &[Gate::X(0)]);

        assert!(hzh.equivalent(&x, TOL).expect("same size"));
    }

    #[test]
    fn s_squared_equals_z() {
        let ss = unitary_of(1, &[Gate::S(0), Gate::S(0)]);
        let z = unitary_of(1, &[Gate::Z(0)]);

        assert!(ss.equivalent(&z, TOL).expect("same size"));
    }

    #[test]
    fn t_squared_equals_s() {
        let tt = unitary_of(1, &[Gate::T(0), Gate::T(0)]);
        let s = unitary_of(1, &[Gate::S(0)]);

        assert!(tt.equivalent(&s, TOL).expect("same size"));
    }

    #[test]
    fn t_to_the_eighth_is_identity() {
        let t8 = unitary_of(1, &[Gate::T(0); 8]);
        let id = unitary_of(1, &[]);

        assert!(t8.equivalent(&id, TOL).expect("same size"));
    }

    #[test]
    fn sdg_and_tdg_invert_s_and_t() {
        let id = unitary_of(1, &[]);
        let s_then_sdg = unitary_of(1, &[Gate::S(0), Gate::Sdg(0)]);
        let t_then_tdg = unitary_of(1, &[Gate::T(0), Gate::Tdg(0)]);

        assert!(s_then_sdg.equivalent(&id, TOL).expect("same size"));
        assert!(t_then_tdg.equivalent(&id, TOL).expect("same size"));
    }

    #[test]
    fn cnot_truth_table() {
        // control = qubit 0, target = qubit 1. Mapping of basis indices:
        // 0->0, 1->3, 2->2, 3->1.
        let cnot = Circuit::from_gates(
            2,
            vec![Gate::Cnot {
                control: 0,
                target: 1,
            }],
        )
        .expect("circuit");

        for (input, want) in [(0usize, 0usize), (1, 3), (2, 2), (3, 1)] {
            let start = State::basis(2, input).expect("basis");
            let out = cnot.run(&start).expect("run");
            let expect = State::basis(2, want).expect("basis");

            assert!(
                (state_fidelity(&out, &expect).expect("same size") - 1.0).abs() < 1e-12,
                "CNOT sent |{input}> to the wrong state",
            );
        }
    }

    #[test]
    fn cnot_reverses_under_hadamards() {
        // H on both lines around a CNOT swaps control and target.
        let conj = unitary_of(
            2,
            &[
                Gate::H(0),
                Gate::H(1),
                Gate::Cnot {
                    control: 0,
                    target: 1,
                },
                Gate::H(0),
                Gate::H(1),
            ],
        );
        let reversed = unitary_of(
            2,
            &[Gate::Cnot {
                control: 1,
                target: 0,
            }],
        );

        assert!(conj.equivalent(&reversed, TOL).expect("same size"));
    }

    #[test]
    fn global_phase_is_ignored() {
        // Multiplying every entry by e^{i pi/4} must not change the fidelity.
        let u = unitary_of(
            2,
            &[
                Gate::H(0),
                Gate::Cnot {
                    control: 0,
                    target: 1,
                },
            ],
        );
        let phase = Complex::new(
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        );
        let mut shifted = u.clone();

        for z in shifted.data.iter_mut() {
            *z = *z * phase;
        }

        assert!(u.equivalent(&shifted, TOL).expect("same size"));
    }

    #[test]
    fn distinct_gates_have_low_fidelity() {
        let x = unitary_of(1, &[Gate::X(0)]);
        let z = unitary_of(1, &[Gate::Z(0)]);

        // X and Z are orthogonal, so the fidelity is 0.
        assert!(x.fidelity(&z).expect("same size") < 1e-9);
    }

    #[test]
    fn costs_count_correctly() {
        let c = Circuit::from_gates(
            2,
            vec![
                Gate::H(0),
                Gate::T(0),
                Gate::Cnot {
                    control: 0,
                    target: 1,
                },
                Gate::T(1),
                Gate::H(0),
            ],
        )
        .expect("circuit");

        assert_eq!(c.t_count(), 2);
        assert_eq!(c.cnot_count(), 1);
        // Depth: q0 runs H,T,CNOT,H (CNOT syncs to layer 3), then H on q0 -> 4.
        assert_eq!(c.depth(), 4);
    }

    #[test]
    fn empty_circuit_has_zero_depth() {
        let c = Circuit::new(3).expect("circuit");

        assert_eq!(c.depth(), 0);
        assert_eq!(c.t_count(), 0);
        assert_eq!(c.cnot_count(), 0);
    }

    #[test]
    fn simulation_is_deterministic() {
        let c = Circuit::from_gates(
            3,
            vec![
                Gate::H(0),
                Gate::Cnot {
                    control: 0,
                    target: 1,
                },
                Gate::T(2),
                Gate::Cnot {
                    control: 1,
                    target: 2,
                },
            ],
        )
        .expect("circuit");

        let a = c.state().expect("state");
        let b = c.state().expect("state");

        assert_eq!(a.amps(), b.amps());
    }

    #[test]
    fn caps_and_bounds_are_enforced() {
        assert!(Circuit::new(0).is_err());
        assert!(Circuit::new(MAX_QUBITS + 1).is_err());

        // Gate index outside the register.
        let mut c = Circuit::new(2).expect("circuit");
        assert!(c.push(Gate::X(2)).is_err());

        // CNOT with equal control and target.
        assert!(
            c.push(Gate::Cnot {
                control: 0,
                target: 0
            })
            .is_err()
        );

        // Unitary refused above the memory cap.
        let wide = Circuit::new(MAX_UNITARY_QUBITS + 1).expect("circuit");
        assert!(wide.unitary().is_err());

        // Basis index outside the state space.
        assert!(State::basis(2, 4).is_err());
    }

    #[test]
    fn fidelity_rejects_mismatched_sizes() {
        let one = unitary_of(1, &[Gate::X(0)]);
        let two = unitary_of(2, &[Gate::X(0)]);

        assert!(one.fidelity(&two).is_err());
    }
}
