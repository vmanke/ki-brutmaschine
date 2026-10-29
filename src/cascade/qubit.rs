//! An exact simulator of Clifford+T circuits (step 1, ticket 02; decisions Q38, Q39).
//!
//! An amplitude is x / sqrt2^k with x = c0 + c1 w + c2 w^2 + c3 w^3 in Z[w], where
//! w = e^(i pi/4). The four coefficients are i64, and one k is shared by a whole state.
//! Every gate is integer arithmetic, so equality is exact: a circuit implements a target
//! when all its columns equal the target's up to one global phase w^g. A fidelity is
//! computed in f64 too, but it is only reported.
//!
//! Conventions: qubit q is bit n - 1 - q of a basis index, so qubit 0 is the most
//! significant bit and the leftmost printed bit. A phase is a u8 k mod 8 that stands for
//! w^k. Gate text is "NAME q" or "CNOT c t", control first. A circuit's text joins its
//! gates with "; ".
//!
//! Why i64 is enough: let x hold the coefficients of a state that gates of this set reached
//! from a basis state, with exponent k. Each of the four embeddings of Z[w] into the
//! complex numbers (w to w, w^3, w^5 or w^7) turns every gate of the set into a unitary,
//! so it maps x to a vector of norm 2^(k/2). A coefficient c is the mean of four embedded
//! values times roots of unity, so |c| <= 2^(k/2). Only H raises k, by one, and a circuit
//! has at most MAX_H of them, so |c| <= 2^60 < i64::MAX. The sum inside H is a
//! coefficient of the next state, so the bound covers it too. All arithmetic is checked
//! anyway: an overflow is an error, never a panic.
//!
//! Circuits and targets are data, never executed as code. The caps are constants in code,
//! not genes.
#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::f64::consts::FRAC_1_SQRT_2;
use std::fmt;
use std::ops::{Add, Mul, Sub};
use std::str::FromStr;

use rand::Rng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::util::sha256_hex;

/// Most qubits of a state vector: 4096 amplitudes, 128 KiB (Q39). A larger value comes only
/// by a reviewed pull request.
pub const MAX_QUBITS: usize = 12;

/// Most qubits of a target and of a full unitary check: 64 columns.
pub const MAX_UNITARY_QUBITS: usize = 6;

/// Most gates in one circuit.
pub const MAX_GATES: usize = 256;

/// Most H gates in one circuit. It bounds every coefficient by 2^60 (see above).
pub const MAX_H: usize = 120;

/// The fidelity threshold of Q38. It is reported only: exact equality decides, and exact
/// equality gives a fidelity of 1 up to rounding.
pub const FIDELITY_MIN: f64 = 1.0 - 1e-9;

/// An element c0 + c1 w + c2 w^2 + c3 w^3 of Z[w], with w = e^(i pi/4).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Zw(pub [i64; 4]);

impl Zw {
    pub const ZERO: Zw = Zw([0; 4]);
    pub const ONE: Zw = Zw([1, 0, 0, 0]);

    /// w^k, with k taken mod 8. Since w^4 = -1, w^k is -w^(k - 4) for k >= 4.
    pub fn unit(k: u8) -> Zw {
        let k = usize::from(k % 8);
        let mut c = [0; 4];

        c[k % 4] = if k < 4 { 1 } else { -1 };

        Zw(c)
    }

    pub fn is_zero(self) -> bool {
        self == Zw::ZERO
    }

    /// The sum, or None when a coefficient leaves the i64 range.
    pub fn checked_add(self, o: Zw) -> Option<Zw> {
        let [a0, a1, a2, a3] = self.0;
        let [b0, b1, b2, b3] = o.0;

        Some(Zw([
            a0.checked_add(b0)?,
            a1.checked_add(b1)?,
            a2.checked_add(b2)?,
            a3.checked_add(b3)?,
        ]))
    }

    /// The difference, or None when a coefficient leaves the i64 range.
    pub fn checked_sub(self, o: Zw) -> Option<Zw> {
        let [a0, a1, a2, a3] = self.0;
        let [b0, b1, b2, b3] = o.0;

        Some(Zw([
            a0.checked_sub(b0)?,
            a1.checked_sub(b1)?,
            a2.checked_sub(b2)?,
            a3.checked_sub(b3)?,
        ]))
    }

    /// self * w^k, with k taken mod 8, or None on overflow. A coefficient that moves past
    /// w^3 changes sign, because w^4 = -1.
    pub fn rotate(self, k: u8) -> Option<Zw> {
        let k = usize::from(k % 8);
        let mut out = [0; 4];

        for (j, c) in self.0.into_iter().enumerate() {
            let to = j + k;

            out[to % 4] = if (to / 4) % 2 == 1 {
                c.checked_neg()?
            } else {
                c
            };
        }

        Some(Zw(out))
    }

    /// The product, or None on overflow. Partial sums are exact in i128.
    pub fn checked_mul(self, o: Zw) -> Option<Zw> {
        let mut sum = [0i128; 4];

        for (i, a) in self.0.into_iter().enumerate() {
            for (j, b) in o.0.into_iter().enumerate() {
                // |a * b| <= 2^126, so the product itself fits an i128.
                let product = i128::from(a) * i128::from(b);
                let slot = &mut sum[(i + j) % 4];

                *slot = if i + j >= 4 {
                    slot.checked_sub(product)?
                } else {
                    slot.checked_add(product)?
                };
            }
        }

        narrow(sum)
    }

    /// self / sqrt2 when that is again in Z[w] (c0 + c2 and c1 + c3 both even), else None.
    pub fn div_sqrt2(self) -> Option<Zw> {
        let doubled = times_sqrt2(self);

        if doubled.iter().any(|c| c % 2 != 0) {
            return None;
        }

        narrow(doubled.map(|c| c / 2))
    }

    /// The complex value of self / sqrt2^k.
    pub fn to_c64(self, k: u32) -> C64 {
        let [c0, c1, c2, c3] = self.0.map(i128::from);
        let re = c0 as f64 + (c1 - c3) as f64 * FRAC_1_SQRT_2;
        let im = c2 as f64 + (c1 + c3) as f64 * FRAC_1_SQRT_2;
        let scale = inv_sqrt2_pow(k);

        C64::new(re * scale, im * scale)
    }

    /// self * sqrt2, or None on overflow.
    fn mul_sqrt2(self) -> Option<Zw> {
        narrow(times_sqrt2(self))
    }

    /// The p with self = w^p, if self is such a unit.
    fn unit_power(self) -> Option<u8> {
        (0..8).find(|&p| Zw::unit(p) == self)
    }
}

/// x * sqrt2, exact in i128: sqrt2 = w - w^3.
fn times_sqrt2(x: Zw) -> [i128; 4] {
    let [c0, c1, c2, c3] = x.0.map(i128::from);

    [c1 - c3, c0 + c2, c1 + c3, c2 - c0]
}

/// Back to i64 coefficients, or None if one does not fit.
fn narrow(c: [i128; 4]) -> Option<Zw> {
    let [c0, c1, c2, c3] = c;

    Some(Zw([
        i64::try_from(c0).ok()?,
        i64::try_from(c1).ok()?,
        i64::try_from(c2).ok()?,
        i64::try_from(c3).ok()?,
    ]))
}

/// 1 / sqrt2^k in f64, by exact halvings. Beyond 2^-1100 the value is 0 in f64 anyway, so
/// the loop stops there.
fn inv_sqrt2_pow(k: u32) -> f64 {
    let mut scale = 1.0;

    for _ in 0..(k / 2).min(1100) {
        scale *= 0.5;
    }

    if k % 2 == 1 {
        scale *= FRAC_1_SQRT_2;
    }

    scale
}

/// A complex number in f64. It serves reports and cross-checks only; no decision uses it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub const fn new(re: f64, im: f64) -> C64 {
        C64 { re, im }
    }

    /// The complex conjugate.
    pub fn conj(self) -> C64 {
        C64::new(self.re, -self.im)
    }

    /// |z|^2.
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

impl Add for C64 {
    type Output = C64;

    fn add(self, o: C64) -> C64 {
        C64::new(self.re + o.re, self.im + o.im)
    }
}

impl Sub for C64 {
    type Output = C64;

    fn sub(self, o: C64) -> C64 {
        C64::new(self.re - o.re, self.im - o.im)
    }
}

impl Mul for C64 {
    type Output = C64;

    fn mul(self, o: C64) -> C64 {
        C64::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

/// Why a circuit, a state or a target is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QError {
    /// The qubit count is 0 or above its cap (MAX_QUBITS for states, MAX_UNITARY_QUBITS for
    /// targets and unitary checks).
    TooManyQubits,
    /// More than MAX_GATES gates.
    TooManyGates,
    /// More than MAX_H H gates.
    TooManyH,
    /// A qubit index outside the register, or a state and a target of different sizes.
    QubitIndex,
    /// A CNOT whose control is its target.
    ControlIsTarget,
    /// A column that does not fit: no terms, a term count that is not a power of two, a
    /// repeated basis state, a basis state outside the register, or a wrong column count.
    BadColumn,
    /// Columns that are not orthogonal.
    NotUnitary,
    /// An exact coefficient left the i64 range.
    Overflow,
}

impl fmt::Display for QError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QError::TooManyQubits => write!(
                f,
                "the qubit count is 0 or above its cap ({MAX_QUBITS} for states, {MAX_UNITARY_QUBITS} for targets)"
            ),
            QError::TooManyGates => write!(f, "more than {MAX_GATES} gates"),
            QError::TooManyH => write!(f, "more than {MAX_H} H gates"),
            QError::QubitIndex => write!(f, "a qubit index outside the register"),
            QError::ControlIsTarget => write!(f, "a CNOT whose control is its target"),
            QError::BadColumn => write!(
                f,
                "a column that is not an equal-weight sum of distinct basis states in range"
            ),
            QError::NotUnitary => write!(f, "the target's columns are not orthogonal"),
            QError::Overflow => write!(f, "an exact coefficient left the i64 range"),
        }
    }
}

impl std::error::Error for QError {}

/// One gate of the set H, S, SDG, T, TDG, X, Z and CNOT, on qubit indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Gate {
    H(u8),
    S(u8),
    Sdg(u8),
    T(u8),
    Tdg(u8),
    X(u8),
    Z(u8),
    Cnot { control: u8, target: u8 },
}

/// Builds a one-qubit gate on a qubit index.
type OneQubit = fn(u8) -> Gate;

/// One-qubit gate names in text (upper case), aliases included.
const SINGLE_NAMES: [(&str, OneQubit); 9] = [
    ("H", Gate::H),
    ("S", Gate::S),
    ("SDG", Gate::Sdg),
    ("SDAG", Gate::Sdg),
    ("T", Gate::T),
    ("TDG", Gate::Tdg),
    ("TDAG", Gate::Tdg),
    ("X", Gate::X),
    ("Z", Gate::Z),
];

/// Names of the CNOT in text (upper case), alias included.
const CNOT_NAMES: [&str; 2] = ["CNOT", "CX"];

impl Gate {
    /// The inverse gate: S and SDG swap, T and TDG swap, the rest are their own inverse.
    pub fn inverse(self) -> Gate {
        match self {
            Gate::S(q) => Gate::Sdg(q),
            Gate::Sdg(q) => Gate::S(q),
            Gate::T(q) => Gate::Tdg(q),
            Gate::Tdg(q) => Gate::T(q),
            other => other,
        }
    }

    /// The qubit of a one-qubit gate, or the control and the target of a CNOT.
    pub fn qubits(self) -> (u8, Option<u8>) {
        match self {
            Gate::H(q)
            | Gate::S(q)
            | Gate::Sdg(q)
            | Gate::T(q)
            | Gate::Tdg(q)
            | Gate::X(q)
            | Gate::Z(q) => (q, None),
            Gate::Cnot { control, target } => (control, Some(target)),
        }
    }

    /// The same gate with qubit q moved to perm[q]. None if a qubit has no image, or if a
    /// CNOT's control and target would meet.
    pub fn relabel(self, perm: &[u8]) -> Option<Gate> {
        let image = |q: u8| perm.get(usize::from(q)).copied();

        match self {
            Gate::Cnot { control, target } => {
                let (c, t) = (image(control)?, image(target)?);

                (c != t).then_some(Gate::Cnot {
                    control: c,
                    target: t,
                })
            }
            single => Some(single.on(image(single.qubits().0)?)),
        }
    }

    /// True for T and TDG.
    pub fn is_t(self) -> bool {
        matches!(self, Gate::T(_) | Gate::Tdg(_))
    }

    /// The gate as text: "H 0", "SDG 2", "CNOT 0 1" (control first).
    pub fn render(self) -> String {
        match self {
            Gate::Cnot { control, target } => format!("CNOT {control} {target}"),
            single => format!("{} {}", single.name(), single.qubits().0),
        }
    }

    /// Read one gate from text: a name and its qubit indices, separated by whitespace.
    /// Names are case-insensitive, and CX, SDAG and TDAG are aliases. An index is written 3
    /// or q3 and must fit a u8. Anything else gives None, a CNOT on one qubit included.
    pub fn parse(text: &str) -> Option<Gate> {
        let mut words = text.split_whitespace();
        let name = words.next()?.to_ascii_uppercase();
        let first = qubit_word(words.next()?)?;
        let gate = if CNOT_NAMES.contains(&name.as_str()) {
            let target = qubit_word(words.next()?)?;

            (first != target).then_some(Gate::Cnot {
                control: first,
                target,
            })?
        } else {
            let (_, make) = SINGLE_NAMES.iter().find(|(n, _)| *n == name)?;

            make(first)
        };

        words.next().is_none().then_some(gate)
    }

    /// True when `word` is a gate name or an alias, in any case.
    pub fn is_name(word: &str) -> bool {
        let name = word.to_ascii_uppercase();

        CNOT_NAMES.contains(&name.as_str()) || SINGLE_NAMES.iter().any(|(n, _)| *n == name)
    }

    /// The canonical name.
    fn name(self) -> &'static str {
        match self {
            Gate::H(_) => "H",
            Gate::S(_) => "S",
            Gate::Sdg(_) => "SDG",
            Gate::T(_) => "T",
            Gate::Tdg(_) => "TDG",
            Gate::X(_) => "X",
            Gate::Z(_) => "Z",
            Gate::Cnot { .. } => "CNOT",
        }
    }

    /// The same one-qubit gate on qubit q (a CNOT stays as it is).
    fn on(self, q: u8) -> Gate {
        match self {
            Gate::H(_) => Gate::H(q),
            Gate::S(_) => Gate::S(q),
            Gate::Sdg(_) => Gate::Sdg(q),
            Gate::T(_) => Gate::T(q),
            Gate::Tdg(_) => Gate::Tdg(q),
            Gate::X(_) => Gate::X(q),
            Gate::Z(_) => Gate::Z(q),
            cnot @ Gate::Cnot { .. } => cnot,
        }
    }
}

/// A qubit index written 3 or q3: ASCII digits that fit a u8.
fn qubit_word(word: &str) -> Option<u8> {
    let digits = word.strip_prefix(['q', 'Q']).unwrap_or(word);

    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }

    digits.parse().ok()
}

/// What a circuit costs. The field order makes the derived order lexicographic in T, then
/// CNOT.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct Cost {
    /// T and TDG gates.
    pub t: u32,
    pub cnot: u32,
    pub h: u32,
    /// All gates.
    pub gates: u32,
    /// Layers when every gate starts as soon as its qubits are free; a CNOT holds both.
    pub depth: u32,
    /// The same layering with only T and TDG layers counted (Amy et al. 2013).
    pub t_depth: u32,
}

/// A gate list in time order, valid by construction: at most MAX_GATES gates and MAX_H H
/// gates, every index below MAX_QUBITS, no CNOT on one qubit. It is data, never executed
/// as code. Its serde form is its text, such as "H 0; CNOT 0 1".
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Circuit {
    gates: Vec<Gate>,
}

impl Circuit {
    /// A circuit from gates in time order, refused when it breaks a cap.
    pub fn new(gates: Vec<Gate>) -> Result<Circuit, QError> {
        if gates.len() > MAX_GATES {
            return Err(QError::TooManyGates);
        }

        for g in &gates {
            let (a, b) = g.qubits();

            if usize::from(a.max(b.unwrap_or(0))) >= MAX_QUBITS {
                return Err(QError::QubitIndex);
            }

            if b == Some(a) {
                return Err(QError::ControlIsTarget);
            }
        }

        if gates.iter().filter(|g| matches!(g, Gate::H(_))).count() > MAX_H {
            return Err(QError::TooManyH);
        }

        Ok(Circuit { gates })
    }

    pub fn gates(&self) -> &[Gate] {
        &self.gates
    }

    /// One more than the highest qubit index; 0 for the empty circuit.
    pub fn qubits_used(&self) -> usize {
        self.gates
            .iter()
            .map(|g| {
                let (a, b) = g.qubits();

                usize::from(a.max(b.unwrap_or(0))) + 1
            })
            .max()
            .unwrap_or(0)
    }

    /// Gate counts, depth and T-depth.
    pub fn cost(&self) -> Cost {
        let mut cost = Cost::default();
        let mut ready = [0u32; MAX_QUBITS];
        let mut t_ready = [0u32; MAX_QUBITS];

        for &g in &self.gates {
            let (a, b) = g.qubits();
            let a = usize::from(a);

            cost.gates += 1;

            match b {
                Some(b) => {
                    let b = usize::from(b);
                    let level = ready[a].max(ready[b]) + 1;
                    let t_level = t_ready[a].max(t_ready[b]);

                    (ready[a], ready[b]) = (level, level);
                    (t_ready[a], t_ready[b]) = (t_level, t_level);
                    cost.cnot += 1;
                }
                None => {
                    ready[a] += 1;

                    if g.is_t() {
                        t_ready[a] += 1;
                        cost.t += 1;
                    }

                    if matches!(g, Gate::H(_)) {
                        cost.h += 1;
                    }
                }
            }
        }

        cost.depth = ready.into_iter().max().unwrap_or(0);
        cost.t_depth = t_ready.into_iter().max().unwrap_or(0);

        cost
    }

    /// The inverse circuit: the gates in reverse order, each inverted. It has the same
    /// counts, so it is valid too.
    pub fn inverse(&self) -> Circuit {
        Circuit {
            gates: self.gates.iter().rev().map(|g| g.inverse()).collect(),
        }
    }

    /// This circuit, then `next`; refused when the result breaks a cap.
    pub fn then(&self, next: &Circuit) -> Result<Circuit, QError> {
        Circuit::new(self.gates.iter().chain(&next.gates).copied().collect())
    }

    /// Every gate relabelled through `perm` (qubit q to perm[q]). None if a gate cannot be
    /// relabelled or the result breaks a cap.
    pub fn relabel(&self, perm: &[u8]) -> Option<Circuit> {
        let gates = self
            .gates
            .iter()
            .map(|g| g.relabel(perm))
            .collect::<Option<Vec<Gate>>>()?;

        Circuit::new(gates).ok()
    }

    /// The circuit as one line of text: its gates joined by "; ".
    pub fn render(&self) -> String {
        self.gates
            .iter()
            .map(|g| g.render())
            .collect::<Vec<_>>()
            .join("; ")
    }
}

impl FromStr for Circuit {
    type Err = String;

    /// Gates separated by ';', each read by `Gate::parse`. A blank text is the empty
    /// circuit; an empty piece, such as after a trailing ';', is refused.
    fn from_str(text: &str) -> Result<Circuit, String> {
        if text.trim().is_empty() {
            return Ok(Circuit::default());
        }

        let gates = text
            .split(';')
            .map(|piece| {
                Gate::parse(piece).ok_or_else(|| format!("not a gate: {:?}", piece.trim()))
            })
            .collect::<Result<Vec<Gate>, String>>()?;

        Circuit::new(gates).map_err(|e| e.to_string())
    }
}

impl TryFrom<String> for Circuit {
    type Error = String;

    fn try_from(text: String) -> Result<Circuit, String> {
        text.parse()
    }
}

impl From<Circuit> for String {
    fn from(c: Circuit) -> String {
        c.render()
    }
}

/// One column of a target: an equal-weight sum of 2^m distinct basis states, each with a
/// phase, so the amplitude of term (b, k) is w^k / sqrt2^m. Terms are sorted by basis.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Column {
    terms: Vec<(u32, u8)>,
}

impl Column {
    /// A column from (basis, phase) terms in any order, phases taken mod 8. Refused
    /// (BadColumn) unless the count is a power of two up to 2^MAX_QUBITS and no basis state
    /// repeats.
    pub fn new(mut terms: Vec<(u32, u8)>) -> Result<Column, QError> {
        let count = terms.len();

        if count == 0 || !count.is_power_of_two() || count > 1 << MAX_QUBITS {
            return Err(QError::BadColumn);
        }

        for term in &mut terms {
            term.1 %= 8;
        }

        terms.sort_unstable();

        if terms
            .windows(2)
            .any(|pair| matches!(pair, [a, b] if a.0 == b.0))
        {
            return Err(QError::BadColumn);
        }

        Ok(Column { terms })
    }

    /// The column w^phase |basis>.
    pub fn single(basis: u32, phase: u8) -> Column {
        Column {
            terms: vec![(basis, phase % 8)],
        }
    }

    pub fn terms(&self) -> &[(u32, u8)] {
        &self.terms
    }

    /// m, where the column has 2^m terms.
    fn m(&self) -> u32 {
        self.terms.len().trailing_zeros()
    }

    /// The column as an exact state on `qubits` qubits (k = m, already reduced).
    fn state(&self, qubits: usize) -> ExactState {
        let mut amps = vec![Zw::ZERO; 1 << qubits];

        for &(b, p) in &self.terms {
            if let Some(slot) = amps.get_mut(b as usize) {
                *slot = Zw::unit(p);
            }
        }

        ExactState {
            qubits,
            k: self.m(),
            amps,
        }
    }
}

/// True when two columns are orthogonal: the sum of w^(q - p) over their shared basis
/// states is exactly 0. Each column has norm 1 by construction.
fn orthogonal(a: &Column, b: &Column) -> bool {
    let mut sum = Some(Zw::ZERO);
    let (mut i, mut j) = (0, 0);

    while let (Some(&(ba, pa)), Some(&(bb, pb))) = (a.terms.get(i), b.terms.get(j)) {
        match ba.cmp(&bb) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                sum = sum.and_then(|s| s.checked_add(Zw::unit(pb + 8 - pa)));
                i += 1;
                j += 1;
            }
        }
    }

    sum.is_some_and(Zw::is_zero)
}

/// A target unitary on 1..=MAX_UNITARY_QUBITS qubits: the images of all 2^n basis inputs,
/// in order. Qubit 0 is the most significant bit of an index.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Target {
    qubits: usize,
    columns: Vec<Column>,
}

impl Target {
    /// A target from its 2^n columns, checked to be exactly unitary.
    pub fn new(qubits: usize, columns: Vec<Column>) -> Result<Target, QError> {
        if qubits == 0 || qubits > MAX_UNITARY_QUBITS {
            return Err(QError::TooManyQubits);
        }

        let dim = 1usize << qubits;
        let in_range = columns
            .iter()
            .flat_map(|c| &c.terms)
            .all(|&(b, _)| (b as usize) < dim);

        if columns.len() != dim || !in_range {
            return Err(QError::BadColumn);
        }

        for (i, a) in columns.iter().enumerate() {
            if columns.iter().skip(i + 1).any(|b| !orthogonal(a, b)) {
                return Err(QError::NotUnitary);
            }
        }

        Ok(Target { qubits, columns })
    }

    /// A target that maps input j to w^p |b>, where (b, p) = image(j): a permutation of the
    /// basis with phases.
    pub fn monomial(qubits: usize, image: impl Fn(u32) -> (u32, u8)) -> Result<Target, QError> {
        if qubits == 0 || qubits > MAX_UNITARY_QUBITS {
            return Err(QError::TooManyQubits);
        }

        let columns = (0..1u32 << qubits)
            .map(|j| {
                let (b, p) = image(j);

                Column::single(b, p)
            })
            .collect();

        Target::new(qubits, columns)
    }

    pub fn qubits(&self) -> usize {
        self.qubits
    }

    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// The columns as exact states, reduced.
    pub fn states(&self) -> Vec<ExactState> {
        self.columns.iter().map(|c| c.state(self.qubits)).collect()
    }

    /// The dressed target X^out . P U P^-1 . X^in. A dressing that does not fit this target
    /// leaves it unchanged (and `Dressing::dress_circuit` refuses it).
    pub fn dress(&self, d: &Dressing) -> Target {
        if d.perm.len() != self.qubits || !d.fits() {
            return self.clone();
        }

        let columns = (0..1u32 << self.qubits)
            .map(|j| {
                let source = self.columns.get(d.unmove(j ^ d.x_in) as usize)?;
                let mut terms: Vec<(u32, u8)> = source
                    .terms
                    .iter()
                    .map(|&(b, p)| (d.move_index(b) ^ d.x_out, p))
                    .collect();

                terms.sort_unstable();

                Some(Column { terms })
            })
            .collect::<Option<Vec<Column>>>();

        match columns {
            Some(columns) => Target {
                qubits: self.qubits,
                columns,
            },
            None => self.clone(),
        }
    }

    /// 16 hex digits of SHA-256 over the canonical text of the target. It is invariant under
    /// a global phase: every phase is taken relative to the first term of column 0.
    pub fn fingerprint(&self) -> String {
        let g = self
            .columns
            .first()
            .and_then(|c| c.terms.first())
            .map_or(0, |&(_, p)| p);
        let mut text = format!("qubits {}", self.qubits);

        for column in &self.columns {
            text.push_str("\ncolumn");

            for &(b, p) in &column.terms {
                text.push_str(&format!(" {b}:{}", (p + 8 - g) % 8));
            }
        }

        sha256_hex(text.as_bytes())[..16].to_string()
    }
}

/// A relabelling of qubits plus X flips on the inputs and outputs:
/// U' = X^out . P U P^-1 . X^in. P moves qubit q to qubit perm[q]. A mask XORs the basis
/// index, so its bit n - 1 - q flips qubit q. Any circuit for U gives one for U' with the
/// same T and CNOT counts, and back, so dressing keeps optimal costs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Dressing {
    pub perm: Vec<u8>,
    pub x_in: u32,
    pub x_out: u32,
}

impl Dressing {
    /// The dressing that changes nothing.
    pub fn identity(qubits: usize) -> Dressing {
        Dressing {
            perm: (0..=u8::MAX).take(qubits).collect(),
            x_in: 0,
            x_out: 0,
        }
    }

    /// A uniform relabelling, then uniform input and output masks, drawn from `rng` in that
    /// order. Above MAX_QUBITS qubits it is the identity and draws nothing.
    pub fn draw(qubits: usize, rng: &mut ChaCha8Rng) -> Dressing {
        let mut d = Dressing::identity(qubits);

        if qubits > MAX_QUBITS {
            return d;
        }

        let span = 1u32 << qubits;

        d.perm.shuffle(rng);
        d.x_in = rng.random_range(0..span);
        d.x_out = rng.random_range(0..span);

        d
    }

    /// The circuit for the dressed target: X on the x_in qubits, then `c` relabelled, then X
    /// on the x_out qubits. It has the T and CNOT counts of `c`. None if the dressing does
    /// not fit `c` or the result breaks a cap.
    pub fn dress_circuit(&self, c: &Circuit) -> Option<Circuit> {
        if !self.fits() {
            return None;
        }

        let n = self.perm.len();
        let flips = |mask: u32| {
            (0..n)
                .filter(move |&q| (mask >> (n - 1 - q)) & 1 == 1)
                .filter_map(|q| u8::try_from(q).ok())
                .map(Gate::X)
        };
        let mut gates: Vec<Gate> = flips(self.x_in).collect();

        gates.extend_from_slice(c.relabel(&self.perm)?.gates());
        gates.extend(flips(self.x_out));

        Circuit::new(gates).ok()
    }

    /// True when perm is a permutation of 0..n with n <= MAX_QUBITS and both masks fit in
    /// n bits.
    fn fits(&self) -> bool {
        let n = self.perm.len();
        let mut seen = [false; MAX_QUBITS];

        n <= MAX_QUBITS
            && self.perm.iter().all(|&p| {
                seen.get_mut(usize::from(p))
                    .is_some_and(|s| usize::from(p) < n && !std::mem::replace(s, true))
            })
            && self.x_in >> n == 0
            && self.x_out >> n == 0
    }

    /// P on a basis index: the bit of qubit q moves to qubit perm[q]. Needs `fits`.
    fn move_index(&self, index: u32) -> u32 {
        let n = self.perm.len();

        self.perm.iter().enumerate().fold(0, |out, (q, &to)| {
            out | (((index >> (n - 1 - q)) & 1) << (n - 1 - usize::from(to)))
        })
    }

    /// P^-1 on a basis index: qubit q takes the bit of qubit perm[q]. Needs `fits`.
    fn unmove(&self, index: u32) -> u32 {
        let n = self.perm.len();

        self.perm.iter().enumerate().fold(0, |out, (q, &from)| {
            out | (((index >> (n - 1 - usize::from(from))) & 1) << (n - 1 - q))
        })
    }
}

/// An exact state on 1..=MAX_QUBITS qubits: amplitude i is amps[i] / sqrt2^k.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactState {
    qubits: usize,
    k: u32,
    amps: Vec<Zw>,
}

impl ExactState {
    /// The basis state |index>. Err for 0 or more than MAX_QUBITS qubits (TooManyQubits), or
    /// for an index outside the register (BadColumn).
    pub fn basis(qubits: usize, index: u32) -> Result<ExactState, QError> {
        if qubits == 0 || qubits > MAX_QUBITS {
            return Err(QError::TooManyQubits);
        }

        let mut amps = vec![Zw::ZERO; 1 << qubits];
        let slot = amps.get_mut(index as usize).ok_or(QError::BadColumn)?;

        *slot = Zw::ONE;

        Ok(ExactState { qubits, k: 0, amps })
    }

    /// Apply one gate. Nothing changes when it fails: an index outside the register, a CNOT
    /// on one qubit, or an overflow.
    pub fn apply(&mut self, g: Gate) -> Result<(), QError> {
        match g {
            Gate::H(q) => {
                let bit = self.place(q)?;
                let k = self.k.checked_add(1).ok_or(QError::Overflow)?;

                self.map_pairs(bit, |a, b| Some((a.checked_add(b)?, a.checked_sub(b)?)))?;
                self.k = k;

                Ok(())
            }
            Gate::S(q) => self.phase(q, 2),
            Gate::Sdg(q) => self.phase(q, 6),
            Gate::T(q) => self.phase(q, 1),
            Gate::Tdg(q) => self.phase(q, 7),
            Gate::Z(q) => self.phase(q, 4),
            Gate::X(q) => {
                let bit = self.place(q)?;

                for block in self.amps.chunks_exact_mut(2 * bit) {
                    let (low, high) = block.split_at_mut(bit);

                    low.swap_with_slice(high);
                }

                Ok(())
            }
            Gate::Cnot { control, target } => {
                if control == target {
                    return Err(QError::ControlIsTarget);
                }

                let (c, t) = (self.place(control)?, self.place(target)?);

                for i in 0..self.amps.len() {
                    if i & c != 0 && i & t == 0 {
                        self.amps.swap(i, i | t);
                    }
                }

                Ok(())
            }
        }
    }

    /// Run a whole circuit. On an error the state stays as it was before the run. A valid
    /// circuit run from a basis state never overflows (see the module comment).
    pub fn run(&mut self, c: &Circuit) -> Result<(), QError> {
        let mut next = self.clone();

        for &g in c.gates() {
            next.apply(g)?;
        }

        *self = next;

        Ok(())
    }

    /// Divide every amplitude by sqrt2 while all of them allow it and k > 0. The result is
    /// the unique form with the smallest k, so equal states reduce to equal values.
    pub fn reduce(&mut self) {
        while self.k > 0 {
            let halved: Option<Vec<Zw>> = self.amps.iter().map(|a| a.div_sqrt2()).collect();

            match halved {
                Some(amps) => {
                    self.amps = amps;
                    self.k -= 1;
                }
                None => break,
            }
        }
    }

    pub fn qubits(&self) -> usize {
        self.qubits
    }

    pub fn k(&self) -> u32 {
        self.k
    }

    pub fn amplitudes(&self) -> &[Zw] {
        &self.amps
    }

    /// (b, p) when the state is exactly w^p |b>, whether or not it is reduced.
    pub fn as_unit(&self) -> Option<(u32, u8)> {
        let mut nonzero = self.amps.iter().enumerate().filter(|(_, a)| !a.is_zero());
        let (b, &a) = nonzero.next()?;

        if nonzero.next().is_some() {
            return None;
        }

        let mut x = a;

        for _ in 0..self.k {
            x = x.div_sqrt2()?;
        }

        Some((u32::try_from(b).ok()?, x.unit_power()?))
    }

    /// The amplitudes as complex numbers.
    pub fn to_c64(&self) -> Vec<C64> {
        self.amps.iter().map(|a| a.to_c64(self.k)).collect()
    }

    /// The place value of qubit q's bit in a basis index.
    fn place(&self, q: u8) -> Result<usize, QError> {
        let q = usize::from(q);

        if q >= self.qubits {
            return Err(QError::QubitIndex);
        }

        Ok(1 << (self.qubits - 1 - q))
    }

    /// Multiply the amplitudes whose qubit q is 1 by w^power.
    fn phase(&mut self, q: u8, power: u8) -> Result<(), QError> {
        let bit = self.place(q)?;

        self.map_pairs(bit, |a, b| Some((a, b.rotate(power)?)))
    }

    /// Replace every pair (qubit at 0, qubit at 1) of amplitudes by `f` of it. Nothing
    /// changes unless every pair succeeds.
    fn map_pairs(
        &mut self,
        bit: usize,
        f: impl Fn(Zw, Zw) -> Option<(Zw, Zw)>,
    ) -> Result<(), QError> {
        let mut next = self.amps.clone();

        for block in next.chunks_exact_mut(2 * bit) {
            let (low, high) = block.split_at_mut(bit);

            for (a, b) in low.iter_mut().zip(high.iter_mut()) {
                (*a, *b) = f(*a, *b).ok_or(QError::Overflow)?;
            }
        }

        self.amps = next;

        Ok(())
    }
}

/// The target applied to an exact state, reduced. Err if their qubit counts differ, or on
/// overflow.
pub fn apply_target(target: &Target, state: &ExactState) -> Result<ExactState, QError> {
    if target.qubits != state.qubits {
        return Err(QError::QubitIndex);
    }

    // Every column is brought to the common denominator sqrt2^top.
    let top = target.columns.iter().map(Column::m).max().unwrap_or(0);
    let mut amps = vec![Zw::ZERO; state.amps.len()];

    for (x, column) in state.amps.iter().zip(&target.columns) {
        if x.is_zero() {
            continue;
        }

        let mut scaled = *x;

        for _ in column.m()..top {
            scaled = scaled.mul_sqrt2().ok_or(QError::Overflow)?;
        }

        for &(b, p) in &column.terms {
            let term = scaled.rotate(p).ok_or(QError::Overflow)?;
            let slot = amps.get_mut(b as usize).ok_or(QError::BadColumn)?;

            *slot = slot.checked_add(term).ok_or(QError::Overflow)?;
        }
    }

    let k = state.k.checked_add(top).ok_or(QError::Overflow)?;
    let mut out = ExactState {
        qubits: state.qubits,
        k,
        amps,
    };

    out.reduce();

    Ok(out)
}

/// All 2^n columns of a circuit's unitary on `qubits` qubits, reduced. Err above
/// MAX_UNITARY_QUBITS qubits, or when the circuit uses a qubit outside them.
pub fn columns(qubits: usize, c: &Circuit) -> Result<Vec<ExactState>, QError> {
    if qubits == 0 || qubits > MAX_UNITARY_QUBITS {
        return Err(QError::TooManyQubits);
    }

    (0..1u32 << qubits)
        .map(|j| {
            let mut state = ExactState::basis(qubits, j)?;

            state.run(c)?;
            state.reduce();

            Ok(state)
        })
        .collect()
}

/// A key equal for two lists of columns exactly when they are equal up to one global phase
/// w^g. Each column is reduced; then all are turned by the w^g that makes the first
/// non-zero entry largest. The 8 turns of a non-zero entry differ, so g is unique.
pub fn canonical_key(columns: &[ExactState]) -> Vec<i64> {
    let reduced: Vec<ExactState> = columns
        .iter()
        .map(|c| {
            let mut c = c.clone();

            c.reduce();
            c
        })
        .collect();
    let first = reduced
        .iter()
        .flat_map(|c| &c.amps)
        .find(|a| !a.is_zero())
        .copied()
        .unwrap_or(Zw::ZERO);
    let g = (0..8u8).max_by_key(|&g| first.rotate(g)).unwrap_or(0);
    let mut key = Vec::new();

    for c in &reduced {
        key.push(c.qubits as i64);
        key.push(i64::from(c.k));

        for a in &c.amps {
            // Only an entry at the very edge of i64 cannot turn; it is kept as it is.
            key.extend(a.rotate(g).unwrap_or(*a).0);
        }
    }

    key
}

/// The outcome of an exact check.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Check {
    /// The circuit equals the target up to one global phase w^g, exactly.
    pub exact: bool,
    /// |sum over j of <t_j|U|j>|^2 / d^2 in f64 (Q38), reported only.
    pub fidelity: f64,
    pub cost: Cost,
}

/// Run `c` on every input of `target` and compare exactly. Column j must have the target's
/// k, support and phases, all turned by one w^g, which the first term of column 0 fixes.
/// Err if the circuit uses a qubit the target lacks, or on overflow (impossible within the
/// caps).
pub fn check(target: &Target, c: &Circuit) -> Result<Check, QError> {
    if c.qubits_used() > target.qubits {
        return Err(QError::QubitIndex);
    }

    let mut phase = None;
    let mut exact = true;
    let mut overlaps = Vec::with_capacity(target.columns.len());

    for (j, column) in (0u32..).zip(&target.columns) {
        let mut state = ExactState::basis(target.qubits, j)?;

        state.run(c)?;
        state.reduce();
        exact = exact && matches_column(&state, column, &mut phase);
        overlaps.push(overlap(column, &state));
    }

    let d = overlaps.len() as f64;
    let fidelity = pairwise_sum(&overlaps).norm_sqr() / (d * d);

    Ok(Check {
        exact,
        fidelity,
        cost: c.cost(),
    })
}

/// Spot checks: true when `c` maps each listed input to the target's column up to one
/// common global phase, which the first listed input fixes. Unlisted columns are not
/// looked at, so a wrong circuit can pass.
pub fn agrees_on(target: &Target, c: &Circuit, inputs: &[u32]) -> Result<bool, QError> {
    if c.qubits_used() > target.qubits {
        return Err(QError::QubitIndex);
    }

    if inputs.iter().any(|&j| j as usize >= target.columns.len()) {
        return Err(QError::BadColumn);
    }

    let mut phase = None;

    for &j in inputs {
        let column = target.columns.get(j as usize).ok_or(QError::BadColumn)?;
        let mut state = ExactState::basis(target.qubits, j)?;

        state.run(c)?;
        state.reduce();

        if !matches_column(&state, column, &mut phase) {
            return Ok(false);
        }
    }

    Ok(true)
}

/// True when the reduced `state` is w^g times `column`. g is `phase` when set; else the
/// first term fixes it and it is stored there.
fn matches_column(state: &ExactState, column: &Column, phase: &mut Option<u8>) -> bool {
    let nonzero = state.amps.iter().filter(|a| !a.is_zero()).count();

    if state.k != column.m() || nonzero != column.terms.len() {
        return false;
    }

    for &(b, p) in &column.terms {
        let Some(q) = state.amps.get(b as usize).and_then(|a| a.unit_power()) else {
            return false;
        };
        let g = (q + 8 - p) % 8;

        match *phase {
            Some(fixed) if fixed != g => return false,
            Some(_) => {}
            None => *phase = Some(g),
        }
    }

    true
}

/// <t|s> for a target column t and a state s, in f64, summed pairwise in term order.
fn overlap(column: &Column, state: &ExactState) -> C64 {
    let terms: Vec<C64> = column
        .terms
        .iter()
        .map(|&(b, p)| {
            let t = Zw::unit(p).to_c64(column.m()).conj();
            let s = state.amps.get(b as usize).copied().unwrap_or(Zw::ZERO);

            t * s.to_c64(state.k)
        })
        .collect();

    pairwise_sum(&terms)
}

/// A sum in a fixed order: halves first, then their sum.
fn pairwise_sum(values: &[C64]) -> C64 {
    match values {
        [] => C64::default(),
        [one] => *one,
        _ => {
            let (left, right) = values.split_at(values.len() / 2);

            pairwise_sum(left) + pairwise_sum(right)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_4, SQRT_2};

    use rand::SeedableRng;

    use super::*;

    /// The Nielsen-Chuang Toffoli (controls 0 and 1, target 2): 7 T and 6 CNOT.
    const TOFFOLI_NC: &str = "H 2; CNOT 1 2; TDG 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; CNOT 0 2; \
                              T 1; T 2; H 2; CNOT 0 1; T 0; TDG 1; CNOT 0 1";

    /// A global phase w as a circuit: (SH)^3 = w I.
    const OMEGA: &str = "H 0; S 0; H 0; S 0; H 0; S 0";

    /// A circuit from its text, such as "H 0; CNOT 0 1".
    fn circuit(text: &str) -> Circuit {
        text.parse().expect("circuit text")
    }

    /// The g with U_a = w^g U_b on n qubits, if there is one.
    fn phase_between(n: usize, a: &Circuit, b: &Circuit) -> Option<u8> {
        let ca = columns(n, a).expect("columns of a");
        let cb = columns(n, b).expect("columns of b");

        (0..8).find(|&g| {
            ca.iter().zip(&cb).all(|(x, y)| {
                x.k == y.k
                    && x.amps
                        .iter()
                        .zip(&y.amps)
                        .all(|(p, q)| q.rotate(g) == Some(*p))
            })
        })
    }

    /// An element with every coefficient in -2^bits..2^bits.
    fn random_zw(rng: &mut ChaCha8Rng, bits: u32) -> Zw {
        let bound = 1i64 << bits;

        Zw([0; 4].map(|_: i64| rng.random_range(-bound..bound)))
    }

    /// A random circuit of `len` gates on n qubits (CNOTs only when n > 1).
    fn random_circuit(rng: &mut ChaCha8Rng, n: usize, len: usize) -> Circuit {
        let kinds = if n > 1 { 8 } else { 7 };
        let gates = (0..len)
            .map(|_| {
                let q = rng.random_range(0..n);
                let qb = u8::try_from(q).expect("small index");

                match rng.random_range(0..kinds) {
                    0 => Gate::H(qb),
                    1 => Gate::S(qb),
                    2 => Gate::Sdg(qb),
                    3 => Gate::T(qb),
                    4 => Gate::Tdg(qb),
                    5 => Gate::X(qb),
                    6 => Gate::Z(qb),
                    _ => {
                        let other = (q + rng.random_range(1..n)) % n;

                        Gate::Cnot {
                            control: qb,
                            target: u8::try_from(other).expect("small index"),
                        }
                    }
                }
            })
            .collect();

        Circuit::new(gates).expect("within the caps")
    }

    /// A dense complex matrix, row by row, as (re, im) pairs. The independent cross-check
    /// builds gates by Kronecker products and shares no arithmetic with the exact engine.
    type Dense = Vec<Vec<(f64, f64)>>;

    fn c_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    }

    fn c_add(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
        (a.0 + b.0, a.1 + b.1)
    }

    fn dense_identity(d: usize) -> Dense {
        (0..d)
            .map(|i| {
                (0..d)
                    .map(|j| if i == j { (1.0, 0.0) } else { (0.0, 0.0) })
                    .collect()
            })
            .collect()
    }

    /// The Kronecker product of two square matrices.
    fn kron(a: &Dense, b: &Dense) -> Dense {
        let (ra, rb) = (a.len(), b.len());

        (0..ra * rb)
            .map(|i| {
                (0..ra * rb)
                    .map(|j| c_mul(a[i / rb][j / rb], b[i % rb][j % rb]))
                    .collect()
            })
            .collect()
    }

    fn matmul(a: &Dense, b: &Dense) -> Dense {
        let d = a.len();

        (0..d)
            .map(|i| {
                (0..d)
                    .map(|j| (0..d).fold((0.0, 0.0), |acc, k| c_add(acc, c_mul(a[i][k], b[k][j]))))
                    .collect()
            })
            .collect()
    }

    /// The 2x2 matrix of a one-qubit gate, from its textbook definition.
    fn small(g: Gate) -> Dense {
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let (one, zero) = ((1.0, 0.0), (0.0, 0.0));
        let diag = |p: (f64, f64)| vec![vec![one, zero], vec![zero, p]];

        match g {
            Gate::H(_) => vec![vec![(r, 0.0), (r, 0.0)], vec![(r, 0.0), (-r, 0.0)]],
            Gate::S(_) => diag((0.0, 1.0)),
            Gate::Sdg(_) => diag((0.0, -1.0)),
            Gate::T(_) => diag((r, r)),
            Gate::Tdg(_) => diag((r, -r)),
            Gate::X(_) => vec![vec![zero, one], vec![one, zero]],
            Gate::Z(_) => diag((-1.0, 0.0)),
            Gate::Cnot { .. } => dense_identity(2),
        }
    }

    /// The 2^n x 2^n matrix of one gate: a Kronecker product with qubit 0 leftmost.
    fn dense_gate(n: usize, g: Gate) -> Dense {
        let product = |factor: &dyn Fn(usize) -> Dense| {
            (0..n).fold(dense_identity(1), |m, q| kron(&m, &factor(q)))
        };

        match g {
            Gate::Cnot { control, target } => {
                let (c, t) = (usize::from(control), usize::from(target));
                let (one, zero) = ((1.0, 0.0), (0.0, 0.0));
                let off = vec![vec![one, zero], vec![zero, zero]];
                let on = vec![vec![zero, zero], vec![zero, one]];
                let kept = product(&|q| {
                    if q == c {
                        off.clone()
                    } else {
                        dense_identity(2)
                    }
                });
                let flipped = product(&|q| {
                    if q == c {
                        on.clone()
                    } else if q == t {
                        small(Gate::X(0))
                    } else {
                        dense_identity(2)
                    }
                });

                kept.iter()
                    .zip(&flipped)
                    .map(|(x, y)| x.iter().zip(y).map(|(&a, &b)| c_add(a, b)).collect())
                    .collect()
            }
            single => {
                let at = usize::from(single.qubits().0);

                product(&|q| {
                    if q == at {
                        small(single)
                    } else {
                        dense_identity(2)
                    }
                })
            }
        }
    }

    /// The whole circuit as one dense matrix.
    fn dense_circuit(n: usize, c: &Circuit) -> Dense {
        c.gates().iter().fold(dense_identity(1 << n), |u, &g| {
            matmul(&dense_gate(n, g), &u)
        })
    }

    /// Largest difference between exact columns and the columns of a dense matrix.
    fn distance(exact: &[ExactState], dense: &Dense) -> f64 {
        let mut worst = 0.0f64;

        for (j, column) in exact.iter().enumerate() {
            for (i, z) in column.to_c64().into_iter().enumerate() {
                let (re, im) = dense[i][j];

                worst = worst.max((z.re - re).abs()).max((z.im - im).abs());
            }
        }

        worst
    }

    /// Targets built from their definitions here (no library), each with a witness from the
    /// literature and its (T, CNOT) counts.
    fn witnesses() -> Vec<(&'static str, Target, Circuit, (u32, u32))> {
        let column = |terms: Vec<(u32, u8)>| Column::new(terms).expect("column");
        let ch = vec![
            Column::single(0, 0),
            Column::single(1, 0),
            column(vec![(2, 0), (3, 0)]),
            column(vec![(2, 0), (3, 4)]),
        ];
        let qft2 = (0..4u32)
            .map(|j| {
                column(
                    (0..4u32)
                        .map(|b| (b, u8::try_from(2 * j * b % 8).expect("phase")))
                        .collect(),
                )
            })
            .collect();
        let nc: Vec<&str> = TOFFOLI_NC.split(';').map(str::trim).collect();
        let fredkin = format!(
            "S 1; CNOT 2 1; SDG 1; H 2; S 2; H 2; S 2; {}; CNOT 2 1",
            nc[2..].join("; ")
        );
        let table = [
            (
                "cz",
                Target::monomial(2, |j| (j, if j == 3 { 4 } else { 0 })),
                "H 1; CNOT 0 1; H 1".to_string(),
                (0, 1),
            ),
            (
                "swap",
                Target::monomial(2, |j| (((j & 1) << 1) | (j >> 1), 0)),
                "CNOT 0 1; CNOT 1 0; CNOT 0 1".to_string(),
                (0, 3),
            ),
            (
                "cs",
                Target::monomial(2, |j| (j, if j == 3 { 2 } else { 0 })),
                "T 0; T 1; CNOT 0 1; TDG 1; CNOT 0 1".to_string(),
                (3, 2),
            ),
            (
                "ch",
                Target::new(2, ch),
                "S 1; H 1; T 1; CNOT 0 1; TDG 1; H 1; SDG 1".to_string(),
                (2, 1),
            ),
            (
                "qft2",
                Target::new(2, qft2),
                "H 0; CNOT 0 1; CNOT 1 0; TDG 1; CNOT 0 1; T 1; T 0; H 0".to_string(),
                (3, 3),
            ),
            (
                "ccz",
                Target::monomial(3, |j| (j, if j == 7 { 4 } else { 0 })),
                format!("{}; {}", nc[1..10].join("; "), nc[11..].join("; ")),
                (7, 6),
            ),
            (
                "toffoli",
                Target::monomial(3, |j| (if j >= 6 { j ^ 1 } else { j }, 0)),
                TOFFOLI_NC.to_string(),
                (7, 6),
            ),
            (
                "peres",
                Target::monomial(3, |j| {
                    let flipped = if j >= 6 { j ^ 1 } else { j };

                    (if j >= 4 { flipped ^ 2 } else { flipped }, 0)
                }),
                nc[..14].join("; "),
                (7, 5),
            ),
            (
                "rtof",
                Target::monomial(3, |j| match j {
                    5 => (5, 4),
                    6 => (7, 2),
                    7 => (6, 6),
                    _ => (j, 0),
                }),
                "H 2; T 2; CNOT 1 2; TDG 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; H 2".to_string(),
                (4, 3),
            ),
            (
                "ccix",
                Target::monomial(3, |j| match j {
                    6 => (7, 2),
                    7 => (6, 2),
                    _ => (j, 0),
                }),
                "H 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; H 2"
                    .to_string(),
                (4, 4),
            ),
            (
                "fredkin",
                Target::monomial(3, |j| match j {
                    5 => (6, 0),
                    6 => (5, 0),
                    _ => (j, 0),
                }),
                fredkin,
                (7, 7),
            ),
        ];

        table
            .into_iter()
            .map(|(name, target, text, counts)| (name, target.expect(name), circuit(&text), counts))
            .collect()
    }

    fn witness(name: &str) -> (Target, Circuit) {
        witnesses()
            .into_iter()
            .find(|w| w.0 == name)
            .map(|(_, t, c, _)| (t, c))
            .expect("known witness")
    }

    #[test]
    fn units_follow_the_table() {
        let table = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
        ];

        for (k, coefficients) in (0u8..).zip(table) {
            let angle = f64::from(k) * FRAC_PI_4;
            let z = Zw::unit(k).to_c64(0);

            assert_eq!(Zw::unit(k), Zw(coefficients), "w^{k}");
            assert_eq!(Zw::unit(k + 8), Zw::unit(k), "mod 8");
            assert_eq!(Zw::unit(k).unit_power(), Some(k));
            assert!((z.re - angle.cos()).abs() < 1e-15 && (z.im - angle.sin()).abs() < 1e-15);

            for j in 0..8 {
                assert_eq!(Zw::unit(k).checked_mul(Zw::unit(j)), Some(Zw::unit(k + j)));
                assert_eq!(Zw::unit(k).rotate(j), Some(Zw::unit(k + j)));
            }
        }

        let sqrt2 = Zw([0, 1, 0, -1]);

        assert_eq!(Zw::ONE, Zw::unit(0));
        assert!(Zw::ZERO.is_zero() && !Zw::ONE.is_zero());
        assert_eq!(Zw([2, 1, 0, 0]).unit_power(), None);
        assert_eq!(sqrt2.checked_mul(sqrt2), Some(Zw([2, 0, 0, 0])));
        assert_eq!(Zw::ONE.mul_sqrt2(), Some(sqrt2));
        assert!((sqrt2.to_c64(0).re - SQRT_2).abs() < 1e-15);
        assert_eq!(Zw([2, 0, 0, 0]).to_c64(2), C64::new(1.0, 0.0));
        assert_eq!(Zw::ONE.to_c64(4), C64::new(0.25, 0.0));
    }

    #[test]
    fn rotation_composes_and_matches_complex_products() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);

        for _ in 0..500 {
            let x = random_zw(&mut rng, 30);
            let z = x.to_c64(60);

            assert_eq!(x.rotate(4), Some(Zw(x.0.map(|c| -c))));
            assert_eq!(x.rotate(8), Some(x));

            for a in 0..8u8 {
                let turned = x.rotate(a).expect("small coefficients");
                let expected = z * Zw::unit(a).to_c64(0);

                assert_eq!(x.checked_mul(Zw::unit(a)), Some(turned));
                assert!((turned.to_c64(60) - expected).norm_sqr() < 1e-24);

                for b in 0..8u8 {
                    assert_eq!(turned.rotate(b), x.rotate(a + b));
                }
            }

            let y = random_zw(&mut rng, 30);
            let product = x.checked_mul(y).expect("fits");

            assert!((product.to_c64(120) - z * y.to_c64(60)).norm_sqr() < 1e-24);
            assert_eq!(x.checked_add(y).and_then(|s| s.checked_sub(y)), Some(x));
        }
    }

    #[test]
    fn div_sqrt2_round_trips_on_random_elements() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let sqrt2 = Zw([0, 1, 0, -1]);
        let mut divisible = 0;

        for _ in 0..2000 {
            let x = random_zw(&mut rng, 61);
            let doubled = x.mul_sqrt2().expect("below 2^62 fits");

            assert_eq!(doubled.div_sqrt2(), Some(x));

            let [c0, c1, c2, c3] = x.0;
            let even = (c0 + c2) % 2 == 0 && (c1 + c3) % 2 == 0;

            match x.div_sqrt2() {
                Some(half) => {
                    divisible += 1;
                    assert!(even);
                    assert_eq!(half.mul_sqrt2(), Some(x));
                }
                None => assert!(!even),
            }

            let small = random_zw(&mut rng, 40);

            assert_eq!(small.checked_mul(sqrt2), small.mul_sqrt2());
        }

        assert!(
            (300..700).contains(&divisible),
            "{divisible} of 2000 divisible"
        );
    }

    #[test]
    fn arithmetic_is_checked_at_the_i64_edges() {
        let max = Zw([i64::MAX, 0, 0, 0]);
        let min = Zw([i64::MIN, 0, 0, 0]);
        let top = Zw([0, 0, 0, i64::MIN]);

        assert_eq!(max.checked_add(Zw::ONE), None);
        assert_eq!(max.checked_add(Zw::ZERO), Some(max));
        assert_eq!(min.checked_sub(Zw::ONE), None);
        assert_eq!(Zw([0, i64::MAX, 0, 0]).checked_sub(Zw([0, -1, 0, 0])), None);
        assert_eq!(min.rotate(4), None, "-(i64::MIN) does not fit");
        assert_eq!(min.rotate(1), Some(Zw([0, i64::MIN, 0, 0])));
        assert_eq!(top.rotate(1), None);
        assert_eq!(top.rotate(5), Some(min), "two sign changes cancel");
        assert_eq!(max.checked_mul(Zw::ONE), Some(max));
        assert_eq!(min.checked_mul(Zw::unit(4)), None);
        assert_eq!(
            Zw([1 << 32, 0, 0, 0]).checked_mul(Zw([1 << 31, 0, 0, 0])),
            None
        );
        assert_eq!(
            Zw([1 << 31, 0, 0, 0]).checked_mul(Zw([1 << 31, 0, 0, 0])),
            Some(Zw([1 << 62, 0, 0, 0]))
        );
        assert_eq!(Zw([i64::MAX, 0, 1, 0]).mul_sqrt2(), None);
        assert_eq!(
            Zw([i64::MAX, 0, i64::MAX, 0]).div_sqrt2(),
            Some(Zw([0, i64::MAX, 0, 0]))
        );
        assert_eq!(
            Zw([i64::MIN, 0, i64::MIN, 0]).div_sqrt2(),
            Some(Zw([0, i64::MIN, 0, 0]))
        );
        assert_eq!(Zw([i64::MAX, 0, 0, 0]).div_sqrt2(), None, "odd");
        assert_eq!(Zw::ONE.to_c64(u32::MAX), C64::new(0.0, 0.0));
    }

    #[test]
    fn an_overflow_is_an_error_and_changes_nothing() {
        let big = Zw([i64::MAX, 0, 0, 0]);
        let mut s = ExactState {
            qubits: 1,
            k: 0,
            amps: vec![big, big],
        };
        let before = s.clone();

        assert_eq!(s.apply(Gate::H(0)), Err(QError::Overflow));
        assert_eq!(s, before);

        let mut t = ExactState {
            qubits: 1,
            k: 0,
            amps: vec![Zw::ONE, Zw([0, 0, 0, i64::MIN])],
        };
        let before = t.clone();

        assert_eq!(t.apply(Gate::T(0)), Err(QError::Overflow));
        assert_eq!(t, before);
        assert_eq!(t.run(&circuit("X 0; X 0; T 0")), Err(QError::Overflow));
        assert_eq!(t, before, "a failed run keeps the state as it was");

        let mut deep = ExactState {
            qubits: 1,
            k: u32::MAX,
            amps: vec![Zw::ONE, Zw::ZERO],
        };

        assert_eq!(deep.apply(Gate::H(0)), Err(QError::Overflow));
    }

    #[test]
    fn gates_read_and_write_their_text() {
        let cnot = |control, target| Some(Gate::Cnot { control, target });
        let cases = [
            ("H 0", Some(Gate::H(0))),
            ("s 1", Some(Gate::S(1))),
            ("SDG 2", Some(Gate::Sdg(2))),
            ("sdag 2", Some(Gate::Sdg(2))),
            ("Sdg q3", Some(Gate::Sdg(3))),
            ("T Q4", Some(Gate::T(4))),
            ("TDG 5", Some(Gate::Tdg(5))),
            ("tDaG 5", Some(Gate::Tdg(5))),
            ("X 11", Some(Gate::X(11))),
            ("z 0", Some(Gate::Z(0))),
            ("H 255", Some(Gate::H(255))),
            ("H 007", Some(Gate::H(7))),
            ("CNOT 0 1", cnot(0, 1)),
            ("cx q1 q0", cnot(1, 0)),
            ("  CNOT   2\t3  ", cnot(2, 3)),
            ("CNOT 1 1", None),
            ("CX q2 2", None),
            ("CNOT 0", None),
            ("CNOT 0 1 2", None),
            ("H", None),
            ("H 0 1", None),
            ("H -1", None),
            ("H +1", None),
            ("H 256", None),
            ("H 1.0", None),
            ("H q", None),
            ("H qq1", None),
            ("H0", None),
            ("Y 0", None),
            ("CZ 0 1", None),
            ("Ｈ 0", None),
            ("", None),
        ];

        for (text, expected) in cases {
            assert_eq!(Gate::parse(text), expected, "{text:?}");
        }

        let all = [
            Gate::H(0),
            Gate::S(1),
            Gate::Sdg(2),
            Gate::T(3),
            Gate::Tdg(4),
            Gate::X(5),
            Gate::Z(6),
            Gate::Cnot {
                control: 7,
                target: 0,
            },
        ];

        for g in all {
            assert_eq!(Gate::parse(&g.render()), Some(g));
            assert_eq!(g.inverse().inverse(), g);
            assert_eq!(g.is_t(), matches!(g, Gate::T(_) | Gate::Tdg(_)));
        }

        assert_eq!(Gate::Sdg(3).render(), "SDG 3");
        assert_eq!(
            Gate::Cnot {
                control: 0,
                target: 1
            }
            .render(),
            "CNOT 0 1"
        );
        assert_eq!(Gate::T(2).inverse(), Gate::Tdg(2));
        assert_eq!(Gate::S(2).inverse(), Gate::Sdg(2));
        assert_eq!(Gate::H(2).inverse(), Gate::H(2));
        assert_eq!(
            Gate::Cnot {
                control: 4,
                target: 1
            }
            .qubits(),
            (4, Some(1))
        );
        assert_eq!(Gate::Z(9).qubits(), (9, None));

        for (word, named) in [
            ("h", true),
            ("CX", true),
            ("Tdag", true),
            ("cnot", true),
            ("Y", false),
            ("", false),
            ("HH", false),
        ] {
            assert_eq!(Gate::is_name(word), named, "{word:?}");
        }
    }

    #[test]
    fn relabel_then_and_inverse_keep_circuits_valid() {
        let c = circuit("H 0; CNOT 0 1; T 1");

        assert_eq!(c.relabel(&[2, 5]), Some(circuit("H 2; CNOT 2 5; T 5")));
        assert_eq!(c.relabel(&[1, 1]), None, "a CNOT on one qubit");
        assert_eq!(c.relabel(&[0]), None, "qubit 1 has no image");
        assert_eq!(c.relabel(&[0, 12]), None, "outside the register");
        assert_eq!(c.inverse(), circuit("TDG 1; CNOT 0 1; H 0"));
        assert_eq!(c.then(&c.inverse()).map(|both| both.gates().len()), Ok(6));
        assert_eq!(c.qubits_used(), 2);
        assert_eq!(circuit("CNOT 3 0").qubits_used(), 4);
        assert_eq!(Circuit::default().qubits_used(), 0);
        assert_eq!(Circuit::default().render(), "");
    }

    #[test]
    fn caps_are_enforced() {
        let hs = |count: usize| vec![Gate::H(0); count];

        assert_eq!(Circuit::new(hs(MAX_H)).map(|c| c.cost().h), Ok(120));
        assert_eq!(Circuit::new(hs(MAX_H + 1)), Err(QError::TooManyH));
        assert_eq!(
            Circuit::new(vec![Gate::X(0); MAX_GATES]).map(|c| c.gates().len()),
            Ok(256)
        );
        assert_eq!(
            Circuit::new(vec![Gate::X(0); MAX_GATES + 1]),
            Err(QError::TooManyGates)
        );
        assert_eq!(
            Circuit::new(vec![Gate::Cnot {
                control: 3,
                target: 3
            }]),
            Err(QError::ControlIsTarget)
        );
        assert_eq!(
            Circuit::new(vec![Gate::X(11)]).map(|c| c.qubits_used()),
            Ok(12)
        );
        assert_eq!(Circuit::new(vec![Gate::X(12)]), Err(QError::QubitIndex));
        assert_eq!(
            Circuit::new(vec![Gate::Cnot {
                control: 0,
                target: 12
            }]),
            Err(QError::QubitIndex)
        );

        let half = Circuit::new(hs(60)).expect("60 H");
        let full = half.then(&half).expect("120 H");

        assert_eq!(full.then(&circuit("H 1")), Err(QError::TooManyH));
        assert!(ExactState::basis(12, 4095).is_ok());
        assert_eq!(ExactState::basis(13, 0), Err(QError::TooManyQubits));
        assert_eq!(ExactState::basis(0, 0), Err(QError::TooManyQubits));
        assert_eq!(ExactState::basis(2, 4), Err(QError::BadColumn));
        assert_eq!(
            columns(7, &Circuit::default()).map(|c| c.len()),
            Err(QError::TooManyQubits)
        );
        assert_eq!(columns(6, &Circuit::default()).map(|c| c.len()), Ok(64));
        assert_eq!(Target::monomial(7, |j| (j, 0)), Err(QError::TooManyQubits));

        let mut s = ExactState::basis(2, 0).expect("state");

        assert_eq!(s.apply(Gate::H(2)), Err(QError::QubitIndex));
        assert_eq!(
            s.apply(Gate::Cnot {
                control: 1,
                target: 1
            }),
            Err(QError::ControlIsTarget)
        );
        assert_eq!(s.run(&circuit("H 0; X 2")), Err(QError::QubitIndex));
        assert_eq!(s, ExactState::basis(2, 0).expect("state"));

        let one = Target::monomial(1, |j| (j, 0)).expect("identity");

        assert_eq!(check(&one, &circuit("X 1")), Err(QError::QubitIndex));
        assert_eq!(
            columns(2, &circuit("X 2")).map(|c| c.len()),
            Err(QError::QubitIndex)
        );
    }

    #[test]
    fn textbook_identities_hold_exactly() {
        let one_qubit = [
            ("H 0; X 0; H 0", "Z 0", 0),
            ("H 0; Z 0; H 0", "X 0", 0),
            ("H 0; H 0", "", 0),
            ("S 0; S 0", "Z 0", 0),
            ("T 0; T 0", "S 0", 0),
            ("T 0; T 0; T 0; T 0; T 0; T 0; T 0; T 0", "", 0),
            ("S 0; SDG 0", "", 0),
            ("T 0; TDG 0", "", 0),
            ("X 0; Z 0; X 0", "Z 0", 4),
            (OMEGA, "", 1),
        ];

        for (a, b, g) in one_qubit {
            assert_eq!(
                phase_between(1, &circuit(a), &circuit(b)),
                Some(g),
                "{a} = w^{g} {b}"
            );
        }

        assert_eq!(phase_between(1, &circuit("T 0"), &Circuit::default()), None);
        assert_eq!(phase_between(1, &circuit("H 0"), &circuit("X 0")), None);

        let two_qubits = [
            ("CNOT 0 1; X 0; CNOT 0 1", "X 0; X 1"),
            ("CNOT 0 1; X 1; CNOT 0 1", "X 1"),
            ("CNOT 0 1; Z 0; CNOT 0 1", "Z 0"),
            ("CNOT 0 1; Z 1; CNOT 0 1", "Z 0; Z 1"),
            ("H 0; H 1; CNOT 0 1; H 0; H 1", "CNOT 1 0"),
        ];

        for (a, b) in two_qubits {
            assert_eq!(
                phase_between(2, &circuit(a), &circuit(b)),
                Some(0),
                "{a} = {b}"
            );
        }

        let swap = Target::monomial(2, |j| (((j & 1) << 1) | (j >> 1), 0)).expect("swap");

        assert!(
            check(&swap, &circuit("CNOT 0 1; CNOT 1 0; CNOT 0 1"))
                .expect("check")
                .exact
        );
        assert!(
            !check(&swap, &circuit("CNOT 0 1; CNOT 1 0"))
                .expect("check")
                .exact
        );
    }

    #[test]
    fn engine_agrees_with_an_independent_dense_simulator() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut worst = 0.0f64;
        let mut largest = 0;

        for _ in 0..500 {
            let n = rng.random_range(1..=5usize);
            let len = rng.random_range(0..=60usize);
            let c = random_circuit(&mut rng, n, len);

            largest = largest.max(n);
            worst = worst.max(distance(
                &columns(n, &c).expect("columns"),
                &dense_circuit(n, &c),
            ));
        }

        assert_eq!(largest, 5);
        assert!(worst < 1e-12, "largest difference {worst:e}");
    }

    #[test]
    fn a_global_phase_passes_and_a_relative_phase_fails() {
        let (toffoli, nc) = witness("toffoli");
        let then = |more: &str| nc.then(&circuit(more)).expect("within the caps");

        for c in [
            nc.clone(),
            then("X 0; Z 0; X 0; Z 0"),
            then(OMEGA),
            then("Z 1; Z 1"),
        ] {
            let r = check(&toffoli, &c).expect("check");

            assert!(r.exact && r.fidelity >= 1.0 - 1e-12, "{}", c.render());
        }

        for more in ["Z 0", "S 2", "T 1", "X 2; Z 2; X 2"] {
            let r = check(&toffoli, &then(more)).expect("check");

            assert!(
                !r.exact && r.fidelity < FIDELITY_MIN,
                "{more}: {}",
                r.fidelity
            );
        }
    }

    #[test]
    fn literature_witnesses_are_exact_at_their_counts() {
        for (name, target, c, (t, cnot)) in witnesses() {
            let r = check(&target, &c).expect("check");

            assert!(r.exact, "{name}");
            assert_eq!((r.cost.t, r.cost.cnot), (t, cnot), "{name}");
            assert!((r.fidelity - 1.0).abs() < 1e-12, "{name}: {}", r.fidelity);
        }
    }

    #[test]
    fn fidelity_is_reported_but_never_decides() {
        let identity = Target::monomial(1, |j| (j, 0)).expect("identity");
        let r = check(&identity, &circuit("T 0")).expect("check");

        // T against the identity: |1 + w|^2 / 4 = (2 + sqrt2) / 4.
        assert!(!r.exact);
        assert!((r.fidelity - (2.0 + SQRT_2) / 4.0).abs() < 1e-15);
        assert!(r.fidelity < FIDELITY_MIN);

        // A global phase changes neither number: w I is the identity up to w.
        let turned = Target::monomial(1, |j| (j, 1)).expect("w I");

        for target in [&identity, &turned] {
            let r = check(target, &Circuit::default()).expect("check");

            assert!(r.exact && (r.fidelity - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn ht_120_times_never_overflows() {
        let ht = |q: u8| [Gate::H(q), Gate::T(q)];
        let one = Circuit::new((0..120).flat_map(|_| ht(0)).collect()).expect("240 gates");

        // Unreduced, as during a run: k = 120 and coefficients close to the 2^60 bound.
        let unreduced: Vec<ExactState> = (0..2)
            .map(|j| {
                let mut s = ExactState::basis(1, j).expect("basis");

                s.run(&one).expect("no overflow");
                s
            })
            .collect();
        let largest = unreduced
            .iter()
            .flat_map(|s| s.amplitudes().iter().flat_map(|z| z.0))
            .map(i64::unsigned_abs)
            .max()
            .unwrap_or(0);

        assert!(unreduced.iter().all(|s| s.k() == 120));
        assert!(largest <= 1 << 60, "bound: {largest}");
        assert!(
            largest >= 1 << 58,
            "the test reaches large coefficients: {largest}"
        );
        assert!(distance(&unreduced, &dense_circuit(1, &one)) < 1e-9);

        let exact = columns(1, &one).expect("no overflow");

        assert!(distance(&exact, &dense_circuit(1, &one)) < 1e-9);
        assert!(exact.iter().all(|s| s.k() < 120), "reduced");

        // On qubit 0 of twelve: a product state, the same numbers as on one qubit.
        let mut on_zero = ExactState::basis(12, 0).expect("state");

        on_zero.run(&one).expect("no overflow");
        on_zero.reduce();
        assert_eq!(on_zero.k(), exact[0].k());

        for (i, z) in on_zero.amplitudes().iter().enumerate() {
            let expected = match i {
                0 => exact[0].amplitudes()[0],
                2048 => exact[0].amplitudes()[1],
                _ => Zw::ZERO,
            };

            assert_eq!(*z, expected, "amplitude {i}");
        }

        // Spread over all twelve qubits: each carries (HT)^10, checked against f64.
        let spread =
            Circuit::new((0..120u8).flat_map(|i| ht(i % 12)).collect()).expect("240 gates");
        let ten = Circuit::new((0..10).flat_map(|_| ht(0)).collect()).expect("20 gates");
        let v = dense_circuit(1, &ten);
        let mut s = ExactState::basis(12, 0).expect("state");

        s.run(&spread).expect("no overflow");
        assert_eq!(s.k(), 120);

        for (i, z) in s.to_c64().into_iter().enumerate() {
            let expected = (0..12).fold((1.0, 0.0), |acc, q| c_mul(acc, v[(i >> (11 - q)) & 1][0]));

            assert!(
                (z.re - expected.0).abs() < 1e-9 && (z.im - expected.1).abs() < 1e-9,
                "{i}"
            );
        }
    }

    #[test]
    fn ghz12_amplitudes_are_exact() {
        let mut gates = vec![Gate::H(0)];

        gates.extend((0..11u8).map(|q| Gate::Cnot {
            control: q,
            target: q + 1,
        }));

        let mut s = ExactState::basis(12, 0).expect("state");

        s.run(&Circuit::new(gates).expect("circuit")).expect("run");
        s.reduce();
        assert_eq!(s.k(), 1);

        for (i, z) in s.amplitudes().iter().enumerate() {
            let expected = if i == 0 || i == 4095 {
                Zw::ONE
            } else {
                Zw::ZERO
            };

            assert_eq!(*z, expected, "amplitude {i}");
        }

        let f = s.to_c64();

        assert_eq!(f[0], f[4095]);
        assert!((f[0].re - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-16);
    }

    #[test]
    fn costs_count_gates_depth_and_t_depth() {
        let cost = |text: &str| circuit(text).cost();
        let c = |t, cnot, h, gates, depth, t_depth| Cost {
            t,
            cnot,
            h,
            gates,
            depth,
            t_depth,
        };

        assert_eq!(cost(""), Cost::default());
        assert_eq!(cost("H 0; H 1"), c(0, 0, 2, 2, 1, 0));
        assert_eq!(cost("T 0; T 0; T 0"), c(3, 0, 0, 3, 3, 3));
        assert_eq!(cost("T 0; T 1; T 2"), c(3, 0, 0, 3, 1, 1));
        assert_eq!(cost("T 0; H 1; T 1"), c(2, 0, 1, 3, 2, 1));
        assert_eq!(cost("T 0; CNOT 0 1; T 1"), c(2, 1, 0, 3, 3, 2));
        assert_eq!(cost("H 0; H 0; CNOT 0 1; X 1"), c(0, 1, 2, 4, 4, 0));
        assert_eq!(
            cost("T 0; T 1; CNOT 0 1; TDG 1; CNOT 0 1"),
            c(3, 2, 0, 5, 4, 2)
        );
        assert_eq!(cost(TOFFOLI_NC), c(7, 6, 2, 15, 11, 4));

        // The derived order is lexicographic in T, then CNOT.
        assert!(c(3, 9, 9, 9, 9, 9) < c(4, 0, 0, 0, 0, 0));
        assert!(c(3, 2, 9, 9, 9, 9) < c(3, 3, 0, 0, 0, 0));
    }

    #[test]
    fn canonical_keys_ignore_global_phase_and_unreduced_k() {
        let mut rng = ChaCha8Rng::seed_from_u64(4);

        for _ in 0..50 {
            let n = rng.random_range(1..=3usize);
            let c = random_circuit(&mut rng, n, 30);
            let cols = columns(n, &c).expect("columns");
            let key = canonical_key(&cols);

            for g in 0..8u8 {
                let turned: Vec<ExactState> = cols
                    .iter()
                    .map(|s| ExactState {
                        amps: s.amps.iter().map(|z| z.rotate(g).expect("small")).collect(),
                        ..s.clone()
                    })
                    .collect();

                assert_eq!(canonical_key(&turned), key, "turned by w^{g}");
            }

            let unreduced: Vec<ExactState> = cols
                .iter()
                .map(|s| ExactState {
                    qubits: s.qubits,
                    k: s.k + 3,
                    amps: s
                        .amps
                        .iter()
                        .map(|z| z.checked_mul(Zw([0, 2, 0, -2])).expect("small"))
                        .collect(),
                })
                .collect();

            assert_eq!(canonical_key(&unreduced), key, "k three too high");

            let phased = c.then(&circuit(OMEGA)).expect("within the caps");
            let relative = c.then(&circuit("T 0")).expect("within the caps");

            assert_eq!(canonical_key(&columns(n, &phased).expect("columns")), key);
            assert_ne!(canonical_key(&columns(n, &relative).expect("columns")), key);
        }
    }

    #[test]
    fn a_circuit_then_its_inverse_has_the_identity_key() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);

        for _ in 0..50 {
            let n = rng.random_range(1..=4usize);
            let c = random_circuit(&mut rng, n, 40);
            let both = c.then(&c.inverse()).expect("within the caps");
            let identity = columns(n, &Circuit::default()).expect("identity");

            assert_eq!(
                canonical_key(&columns(n, &both).expect("columns")),
                canonical_key(&identity)
            );
            assert_eq!(c.inverse().inverse(), c);
            assert_eq!(
                (c.inverse().cost().t, c.inverse().cost().cnot),
                (c.cost().t, c.cost().cnot)
            );
        }
    }

    #[test]
    fn targets_must_be_unitary_and_columns_well_formed() {
        let single = Column::single;
        let plus = Column::new(vec![(0, 0), (1, 0)]).expect("plus");
        let minus = Column::new(vec![(1, 4), (0, 0)]).expect("minus");

        assert_eq!(Column::new(vec![]), Err(QError::BadColumn));
        assert_eq!(
            Column::new(vec![(0, 0), (1, 0), (2, 0)]),
            Err(QError::BadColumn)
        );
        assert_eq!(Column::new(vec![(1, 0), (1, 4)]), Err(QError::BadColumn));
        assert_eq!(minus.terms(), &[(0, 0), (1, 4)], "sorted by basis");
        assert_eq!(
            Column::new(vec![(3, 9), (1, 0)]).map(|c| c.terms().to_vec()),
            Ok(vec![(1, 0), (3, 1)])
        );
        assert_eq!(single(2, 11).terms(), &[(2, 3)]);

        assert_eq!(
            Target::new(0, vec![single(0, 0)]),
            Err(QError::TooManyQubits)
        );
        assert_eq!(Target::new(1, vec![single(0, 0)]), Err(QError::BadColumn));
        assert_eq!(
            Target::new(1, vec![single(0, 0), single(2, 0)]),
            Err(QError::BadColumn)
        );
        assert_eq!(
            Target::new(1, vec![single(0, 0), single(0, 3)]),
            Err(QError::NotUnitary)
        );
        assert_eq!(
            Target::new(1, vec![plus.clone(), plus.clone()]),
            Err(QError::NotUnitary)
        );
        assert_eq!(
            Target::new(1, vec![plus.clone(), single(1, 0)]),
            Err(QError::NotUnitary)
        );
        assert!(Target::new(1, vec![plus.clone(), minus]).is_ok(), "H");

        // (|0> + i|1>) and (|0> - i|1>) are orthogonal; (|0> + i|1>) and (|0> + w|1>) are not.
        let column = |terms: Vec<(u32, u8)>| Column::new(terms).expect("column");

        assert!(
            Target::new(
                1,
                vec![column(vec![(0, 0), (1, 2)]), column(vec![(0, 0), (1, 6)])]
            )
            .is_ok()
        );
        assert_eq!(
            Target::new(
                1,
                vec![column(vec![(0, 0), (1, 2)]), column(vec![(0, 0), (1, 1)])]
            ),
            Err(QError::NotUnitary)
        );

        assert_eq!(Target::monomial(2, |j| (j + 1, 0)), Err(QError::BadColumn));
        assert_eq!(Target::monomial(2, |_| (0, 0)), Err(QError::NotUnitary));

        let (ch, _) = witness("ch");

        assert_eq!(ch.qubits(), 2);
        assert_eq!(ch.columns().len(), 4);
        assert_eq!(Target::new(2, ch.columns().to_vec()), Ok(ch.clone()));
    }

    #[test]
    fn fingerprints_ignore_a_global_phase_only() {
        let all = witnesses();
        let mut seen = std::collections::BTreeSet::new();

        for (name, target, _, _) in &all {
            let print = target.fingerprint();
            let turned = Target {
                qubits: target.qubits,
                columns: target
                    .columns
                    .iter()
                    .map(|c| Column {
                        terms: c.terms.iter().map(|&(b, p)| (b, (p + 3) % 8)).collect(),
                    })
                    .collect(),
            };

            assert_eq!(print.len(), 16);
            assert!(print.bytes().all(|b| b.is_ascii_hexdigit()));
            assert_eq!(turned.fingerprint(), print, "{name}");
            assert!(
                seen.insert(print),
                "{name}: fingerprints differ between targets"
            );
        }
    }

    #[test]
    fn states_and_apply_target_agree_with_the_columns() {
        let mut rng = ChaCha8Rng::seed_from_u64(6);

        for (name, target, witness, _) in witnesses() {
            let n = target.qubits();

            for (j, column) in (0u32..).zip(target.states()) {
                let input = ExactState::basis(n, j).expect("basis");

                assert_eq!(
                    apply_target(&target, &input).as_ref(),
                    Ok(&column),
                    "{name} column {j}"
                );
            }

            // On superpositions too: the target after a random prefix equals the witness after
            // it, up to the witness's one global phase.
            let mut phases = std::collections::BTreeSet::new();

            for _ in 0..10 {
                let prefix = random_circuit(&mut rng, n, 12);
                let mut s = ExactState::basis(n, 0).expect("basis");

                s.run(&prefix).expect("run");

                let by_target = apply_target(&target, &s).expect("apply");

                s.run(&witness).expect("run");
                s.reduce();

                let g = (0..8u8).find(|&g| {
                    s.k == by_target.k
                        && s.amps
                            .iter()
                            .zip(&by_target.amps)
                            .all(|(a, b)| b.rotate(g) == Some(*a))
                });

                phases.insert(g.expect("equal up to phase"));
            }

            assert_eq!(phases.len(), 1, "{name}: one global phase");
        }

        let (cz, _) = witness("cz");

        assert_eq!(
            apply_target(&cz, &ExactState::basis(3, 0).expect("basis")),
            Err(QError::QubitIndex)
        );
    }

    #[test]
    fn as_unit_reads_basis_states_with_a_phase() {
        let mut s = ExactState::basis(2, 1).expect("basis");

        assert_eq!(s.as_unit(), Some((1, 0)));
        s.run(&circuit("H 0; H 0; T 1; X 0")).expect("run");
        assert_eq!(s.k(), 2, "not reduced");
        assert_eq!(s.as_unit(), Some((3, 1)));
        s.run(&circuit("H 1")).expect("run");
        assert_eq!(s.as_unit(), None);
        assert_eq!(s.qubits(), 2);
    }

    #[test]
    fn spot_checks_can_be_fooled_but_the_full_check_cannot() {
        let (cz, witness) = witness("cz");
        let fake = circuit("Z 0");

        assert_eq!(agrees_on(&cz, &fake, &[0, 3]), Ok(true));
        assert_eq!(agrees_on(&cz, &fake, &[0, 1, 2, 3]), Ok(false));
        assert!(!check(&cz, &fake).expect("check").exact);
        assert_eq!(agrees_on(&cz, &witness, &[0, 1, 2, 3]), Ok(true));
        assert_eq!(agrees_on(&cz, &fake, &[]), Ok(true));
        assert_eq!(agrees_on(&cz, &fake, &[1, 4]), Err(QError::BadColumn));
        assert_eq!(
            agrees_on(&cz, &circuit("X 2"), &[0]),
            Err(QError::QubitIndex)
        );

        // One global phase for all listed columns: Z 0 gives -|10>, which alone is CZ's column
        // up to -1, but not with the phase column 0 fixes.
        assert_eq!(agrees_on(&cz, &fake, &[2]), Ok(true));
        assert_eq!(agrees_on(&cz, &fake, &[0, 2]), Ok(false));
    }

    #[test]
    fn dressing_keeps_exactness_and_t_and_cnot_counts() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);

        for (name, target, witness, _) in witnesses() {
            let n = target.qubits();

            for _ in 0..20 {
                let d = Dressing::draw(n, &mut rng);
                let dressed = target.dress(&d);
                let c = d.dress_circuit(&witness).expect("fits");
                let r = check(&dressed, &c).expect("check");
                let base = witness.cost();

                assert!(r.exact, "{name} {d:?}");
                assert_eq!((r.cost.t, r.cost.cnot), (base.t, base.cnot), "{name}");
                assert_eq!(
                    r.cost.gates,
                    base.gates + d.x_in.count_ones() + d.x_out.count_ones()
                );
                assert_eq!(
                    Target::new(n, dressed.columns().to_vec()).as_ref(),
                    Ok(&dressed)
                );

                let mut perm = d.perm.clone();

                perm.sort_unstable();
                assert_eq!(perm, (0..=u8::MAX).take(n).collect::<Vec<u8>>());
                assert!(d.x_in < 1 << n && d.x_out < 1 << n);
            }

            let id = Dressing::identity(n);

            assert_eq!(target.dress(&id), target);
            assert_eq!(id.dress_circuit(&witness), Some(witness.clone()));
        }

        let draw = |seed| Dressing::draw(3, &mut ChaCha8Rng::seed_from_u64(seed));

        assert_eq!(draw(9), draw(9));
        assert!((10..30).any(|seed| draw(seed) != draw(9)));

        // A dressing that does not fit leaves the target as it is and dresses no circuit.
        let (toffoli, nc) = witness("toffoli");
        let unfit = [
            Dressing {
                perm: vec![0, 0, 1],
                x_in: 0,
                x_out: 0,
            },
            Dressing {
                perm: vec![0, 1, 3],
                x_in: 0,
                x_out: 0,
            },
            Dressing {
                perm: vec![0, 1, 2],
                x_in: 8,
                x_out: 0,
            },
            Dressing {
                perm: vec![0, 1, 2],
                x_in: 0,
                x_out: 9,
            },
        ];

        for d in &unfit {
            assert_eq!(toffoli.dress(d), toffoli, "{d:?}");
            assert_eq!(d.dress_circuit(&nc), None, "{d:?}");
        }

        let narrow = Dressing::identity(2);

        assert_eq!(toffoli.dress(&narrow), toffoli, "another size");
        assert_eq!(narrow.dress_circuit(&nc), None, "qubit 2 has no image");
        assert_eq!(Dressing::draw(13, &mut rng), Dressing::identity(13));
    }

    #[test]
    fn circuits_serialise_as_text() {
        let c = circuit("H 0; CNOT 0 1; TDG 1");
        let json = serde_json::to_string(&c).expect("json");

        assert_eq!(json, "\"H 0; CNOT 0 1; TDG 1\"");
        assert_eq!(serde_json::from_str::<Circuit>(&json).expect("back"), c);
        assert_eq!(
            serde_json::to_string(&Circuit::default()).expect("json"),
            "\"\""
        );
        assert_eq!(
            serde_json::from_str::<Circuit>("\"  \"").expect("blank"),
            Circuit::default()
        );
        assert_eq!(
            circuit("h q0 ;cx 0 1;  tdag 1").render(),
            "H 0; CNOT 0 1; TDG 1"
        );
        assert_eq!(String::from(c.clone()), c.render());
        assert_eq!(Circuit::try_from(c.render()), Ok(c.clone()));

        let many_h = format!("\"{}\"", vec!["H 0"; MAX_H + 1].join("; "));

        for bad in [
            "\"H 0;\"",
            "\"; H 0\"",
            "\"FOO 1\"",
            "\"CNOT 1 1\"",
            "\"H 12\"",
            "3",
            "\"H 0\\nH 1\"",
            &many_h,
        ] {
            assert!(serde_json::from_str::<Circuit>(bad).is_err(), "{bad}");
        }

        assert_eq!(
            serde_json::to_string(&c.cost()).expect("json"),
            r#"{"t":1,"cnot":1,"h":1,"gates":3,"depth":3,"t_depth":1}"#
        );

        let d = Dressing {
            perm: vec![2, 0, 1],
            x_in: 5,
            x_out: 1,
        };
        let text = serde_json::to_string(&d).expect("json");

        assert_eq!(text, r#"{"perm":[2,0,1],"x_in":5,"x_out":1}"#);
        assert_eq!(serde_json::from_str::<Dressing>(&text).expect("back"), d);
    }

    #[test]
    fn errors_say_what_is_wrong() {
        assert_eq!(QError::TooManyH.to_string(), "more than 120 H gates");
        assert_eq!(QError::TooManyGates.to_string(), "more than 256 gates");
        assert!(QError::TooManyQubits.to_string().contains("12"));
        assert!(QError::Overflow.to_string().contains("i64"));
    }
}
