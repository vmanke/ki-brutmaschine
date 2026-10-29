//! A simulated circuit designer, the positive control of circuit breeding (step 1, ticket
//! 03).
//!
//! `SimDesigner` is not a language model. It reads the task from the prompt text alone: a
//! prompt shows the target's table, the gate cap and earlier answers, never a task's name,
//! id or known cost. It builds circuits with the method families its system prompt names,
//! within its thinking budget, so its skill depends only on the system prompt and the
//! Traits. We know which prompts and Traits make it strong, so a search that cannot find
//! them is broken.
//!
//! It stores no circuit, template or target list. Every candidate comes from synthesis
//! within the step budget, and a candidate counts only after it is checked.
//!
//! How it answers, deterministically from the request:
//! 1. Knowledge: stems in the lower-cased system prompt name the families (`FAMILY_STEMS`),
//!    in the order of their first mention. That order is the method order.
//! 2. Thinking: all work costs steps, paid before the work is done. A call has 16 free
//!    steps plus one step per `step_tokens` thinking tokens, never more than 2^20 steps in
//!    all. Once a payment fails, thinking stops for the rest of the call. The steps used
//!    beyond the free ones are reported as thinking tokens.
//! 3. Self-check: with `CIRCUIT_SELF_CHECK_TEXT` in the system prompt, a candidate is
//!    verified on every column of the target. Without it, only the first and the last
//!    column are checked, and a wrong circuit can pass.
//! 4. Temperature: with probability 0.5 * temperature / 1.5 the method order is shuffled.
//! 5. Solve: each method runs until it is done or the budget is spent. The reply is the
//!    cheapest verified candidate by (T, CNOT, gates), else the latest earlier answer that
//!    fits the gate cap, else a line without ANSWER.
//! 6. Judge: verify both candidates. When exactly one passes, it wins; otherwise the stated
//!    costs decide (T, then CNOT), and a seeded coin breaks a tie. So without self-check,
//!    a cheap wrong circuit can fool the judge.
//!
//! The methods are generic, published algorithms:
//! - Linear: a fewest-CNOT network for the linear part of an affine permutation, by
//!   breadth-first search; the X gates of the shift are free.
//! - Phase: the phase polynomial of the diagonal part (Amy and Mosca 2019), from the Walsh
//!   transform. With Linear, the network search must also visit every parity that carries
//!   a phase (parity networks, cf. Amy, Azimzadeh and Mosca 2018). Without Linear, each
//!   parity is computed, given its phase and uncomputed, which needs an identity linear
//!   part.
//! - Hadamard: a change of basis by H on a subset of the qubits, accepted when the target
//!   becomes an affine permutation with phases that the families above can build. Without
//!   Phase, a Z gate that the change of basis makes from an X is built as H X H.
//! - Compose: conjugation by one CNOT, every ordered pair, then the methods above.
//! - Search: breadth-first search over gate words on the target's qubits, deduplicated by
//!   the canonical key of the checked columns. X gates on the inputs and outputs are free,
//!   so the search compares words under X flips at both ends. It stops at its first match,
//!   which is short but not always the cheapest circuit. It works on up to three qubits.
//! - Meet: the search grows words from both ends and matches them in the middle (Amy,
//!   Maslov, Mosca and Roetteler 2013). Meet implies Search; there is one search per call.
//! - Simplify: exact rewrites of the cheapest candidate so far and of the earlier answers:
//!   peephole rules (inverse pairs cancel, phase gates on one qubit merge, gates move only
//!   past gates they commute with), phase folding (Nam, Ross, Su, Childs and Maslov 2018),
//!   and two-qubit Clifford windows rebuilt with fewer CNOTs from a table of Cliffords that
//!   a breadth-first search in order of CNOT count builds within the budget. Every result is
//!   verified like any other candidate.
//! - Previous: verify the answers of earlier stages and keep the cheapest that passes.
#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::backend::{GenRequest, GenResponse, ModelBackend, StopReason, approx_tokens};
use crate::cascade::circuit_text::{
    CIRCUIT_SELF_CHECK_TEXT, ParsedCircuitPrompt, parse_circuit_prompt, render_circuit_answer,
};
use crate::cascade::qubit::{
    Circuit, ExactState, Gate, MAX_UNITARY_QUBITS, Target, agrees_on, apply_target, canonical_key,
    columns,
};
use crate::util::{Result, derive_seed};

/// The reply to a prompt that does not follow the circuit protocol.
const CANNOT_READ: &str = "I cannot read this task.";

/// The reply when no candidate passed and no earlier answer can be passed on.
const NO_CIRCUIT: &str = "I found no circuit for this target.";

/// Steps every call may use without thinking tokens.
const FREE_STEPS: u64 = 16;

/// The most steps one call may use, whatever its thinking budget. With at most 4096
/// thinking tokens (the Traits' ceiling) a call has at most 16 + 4096 / 2 = 2,064 steps, so
/// there the ceiling never binds. It bounds the work of a direct request that asks for far
/// more thinking than a Worker can have.
const MAX_STEPS: u64 = 1 << 20;

/// States a network search reaches for one step.
const STATES_PER_STEP: usize = 16;

/// The most states one network search holds, whatever the budget. It caps memory only: with
/// at most 4096 thinking tokens (the Traits' ceiling) a search reaches at most 33,024
/// states, so there the budget always binds first.
const MAX_NETWORK_STATES: usize = 1 << 18;

/// The temperature at which the method order is shuffled with probability one half: the
/// ceiling of the temperature Trait.
const TEMPERATURE_CEILING: f32 = 1.5;

/// Columns of new words a word search computes for one step: 4 words on two qubits, 2 on
/// three. A word on more qubits costs more work, since it holds more columns and is compared
/// under more input flips.
const COLUMNS_PER_STEP: usize = 16;

/// Classes the table of two-qubit Cliffords gains for one step.
const CLASSES_PER_STEP: usize = 4;

/// The most qubits a word search works on. A node holds all 2^n columns and is compared
/// under all 2^n input flips, so the work per node grows as 4^n; on larger targets Search
/// and Meet contribute no candidates.
const SEARCH_QUBITS: usize = 3;

/// The most columns the words of one search hold, whatever the budget: 8,192 words on three
/// qubits. It caps memory only: with at most 4096 thinking tokens (the Traits' ceiling) a
/// search computes at most 33,024 columns, so there the budget always binds first.
const MAX_SEARCH_COLUMNS: usize = 1 << 16;

/// Each family's name with the stems that name it, matched in the lower-cased system
/// prompt, in English and German.
pub const FAMILY_STEMS: [(&str, &[&str]); 8] = [
    (
        "Linear",
        &["xor", "row operation", "linear", "zeilenoperation"],
    ),
    ("Phase", &["parit", "phase polynomial", "phasenpolynom"]),
    (
        "Hadamard",
        &[
            "hadamard",
            "change the basis",
            "basis change",
            "basiswechsel",
        ],
    ),
    ("Compose", &["conjugat", "sandwich", "konjug", "einrahm"]),
    (
        "Search",
        &[
            "search",
            "try every",
            "one gate at a time",
            "durchsuch",
            "durchprobier",
        ],
    ),
    ("Meet", &["meet in the middle", "both ends", "beiden enden"]),
    (
        "Simplify",
        &["cancel", "simplif", "merge", "kürz", "vereinfach"],
    ),
    (
        "Previous",
        &[
            "previous answer",
            "previous circuit",
            "earlier answer",
            "earlier circuit",
            "vorige antwort",
            "vorige schaltung",
        ],
    ),
];

/// A family of methods a system prompt can name, in the order of `FAMILY_STEMS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Family {
    /// CNOT and X: a fewest-CNOT network for an affine permutation.
    Linear,
    /// T, TDG, S, SDG and Z on parities: phase polynomials.
    Phase,
    /// H: a change of basis on a subset of the qubits.
    Hadamard,
    /// Conjugation by one CNOT, then the methods above.
    Compose,
    /// Breadth-first search over gate words, one gate at a time.
    Search,
    /// Search from both ends of the circuit, which meet in the middle. It implies Search.
    Meet,
    /// Exact rewrites of the circuits at hand: peephole rules, phase folding, and two-qubit
    /// Clifford windows rebuilt with fewer CNOTs.
    Simplify,
    /// Verify the answers of earlier stages.
    Previous,
}

impl Family {
    /// Every family, in the order of `FAMILY_STEMS`.
    const ALL: [Family; 8] = [
        Family::Linear,
        Family::Phase,
        Family::Hadamard,
        Family::Compose,
        Family::Search,
        Family::Meet,
        Family::Simplify,
        Family::Previous,
    ];

    /// The family's name, as in `FAMILY_STEMS`.
    fn name(self) -> &'static str {
        match self {
            Family::Linear => "Linear",
            Family::Phase => "Phase",
            Family::Hadamard => "Hadamard",
            Family::Compose => "Compose",
            Family::Search => "Search",
            Family::Meet => "Meet",
            Family::Simplify => "Simplify",
            Family::Previous => "Previous",
        }
    }
}

/// The families a system prompt names, in the order of their first stem.
fn known(system: &str) -> Vec<Family> {
    let lower = system.to_lowercase();
    let mut named: Vec<(usize, Family)> = FAMILY_STEMS
        .iter()
        .zip(Family::ALL)
        .filter_map(|(&(_, stems), family)| {
            let first = stems.iter().filter_map(|s| lower.find(s)).min();

            first.map(|at| (at, family))
        })
        .collect();

    named.sort();
    named.into_iter().map(|(_, family)| family).collect()
}

/// The names of the families a system prompt names, in the order of their first mention.
pub fn knowledge(system: &str) -> Vec<&'static str> {
    known(system).into_iter().map(Family::name).collect()
}

/// What the known families can build, whatever their order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Skills {
    linear: bool,
    phase: bool,
    hadamard: bool,
}

impl Skills {
    fn of(families: &[Family]) -> Skills {
        Skills {
            linear: families.contains(&Family::Linear),
            phase: families.contains(&Family::Phase),
            hadamard: families.contains(&Family::Hadamard),
        }
    }
}

/// The thinking budget of one call, counted in steps. Work is paid before it is done, so
/// the budget also bounds the time a call takes.
#[derive(Clone, Debug)]
struct Meter {
    budget: u64,
    used: u64,
    exhausted: bool,
}

impl Meter {
    /// FREE_STEPS plus one step per `step_tokens` thinking tokens, at most MAX_STEPS.
    fn new(think_tokens: u32, step_tokens: u32) -> Meter {
        let paid = u64::from(think_tokens) / u64::from(step_tokens.max(1));

        Meter {
            budget: (FREE_STEPS + paid).min(MAX_STEPS),
            used: 0,
            exhausted: false,
        }
    }

    /// Pay for `steps` steps of work. Once the budget cannot cover a payment, thinking
    /// stops for the rest of the call.
    fn pay(&mut self, steps: u64) -> bool {
        if self.exhausted {
            return false;
        }

        match self.used.checked_add(steps) {
            Some(total) if total <= self.budget => {
                self.used = total;
                true
            }
            _ => {
                self.exhausted = true;
                false
            }
        }
    }

    /// The thinking tokens to report: the steps used beyond the free ones, at the model's
    /// rate. Never more than the request's thinking budget.
    fn thinking(&self, step_tokens: u32) -> u32 {
        let paid = self.used.saturating_sub(FREE_STEPS);

        u32::try_from(paid.saturating_mul(u64::from(step_tokens))).unwrap_or(u32::MAX)
    }
}

/// 2^n, the number of columns of an n-qubit target.
fn dim(n: usize) -> u32 {
    1 << n.min(MAX_UNITARY_QUBITS)
}

/// The qubit indices 0..n.
fn qubits(n: usize) -> impl Iterator<Item = u8> {
    (0u8..).take(n.min(MAX_UNITARY_QUBITS))
}

/// Every ordered pair (control, target) of distinct qubits, control first.
fn pairs(n: usize) -> Vec<(u8, u8)> {
    qubits(n)
        .flat_map(|c| qubits(n).filter(move |&t| t != c).map(move |t| (c, t)))
        .collect()
}

/// True when the parity z of the basis-index bits is 1 on input x.
fn odd(z: u32, x: u32) -> bool {
    (z & x).count_ones() % 2 == 1
}

/// The basis-index bit of qubit q < n: qubit 0 is the most significant bit.
fn place(n: usize, q: u8) -> u32 {
    1 << (n - 1 - usize::from(q))
}

/// The cost order of candidates: T, then CNOT, then all gates.
fn rank(c: &Circuit) -> (u32, u32, u32) {
    let cost = c.cost();

    (cost.t, cost.cnot, cost.gates)
}

/// The columns verification checks: all of them with self-check, else the first and the
/// last.
fn checked_columns(n: usize, self_check: bool) -> Vec<u32> {
    let last = dim(n) - 1;

    if self_check {
        (0..=last).collect()
    } else {
        vec![0, last]
    }
}

/// Verify `c` on the checked columns, at one step per column, paid before the check. None
/// when the budget cannot pay; else whether `c` agrees with the target there, up to one
/// global phase.
fn verify(target: &Target, c: &Circuit, checked: &[u32], meter: &mut Meter) -> Option<bool> {
    let steps = u64::try_from(checked.len()).unwrap_or(u64::MAX);

    if !meter.pay(steps) {
        return None;
    }

    Some(agrees_on(target, c, checked).unwrap_or(false))
}

/// The candidates of one solve call. A candidate counts only after it is verified, and
/// only the cheapest verified one is kept; on a tie the earlier one stays.
struct Pool<'t> {
    target: &'t Target,
    checked: Vec<u32>,
    max_gates: usize,
    best: Option<Circuit>,
}

impl<'t> Pool<'t> {
    fn new(target: &'t Target, self_check: bool, max_gates: usize) -> Self {
        Pool {
            target,
            checked: checked_columns(target.qubits(), self_check),
            max_gates,
            best: None,
        }
    }

    /// Offer a candidate. It is dropped without cost when it has no gates, breaks the gate
    /// cap, or is no cheaper than the best so far; otherwise it is verified and kept when
    /// it passes. True when it was kept.
    fn offer(&mut self, c: Circuit, meter: &mut Meter) -> bool {
        let fits = !c.gates().is_empty() && c.gates().len() <= self.max_gates;
        let cheaper = self.best.as_ref().is_none_or(|b| rank(&c) < rank(b));

        if !fits || !cheaper || verify(self.target, &c, &self.checked, meter) != Some(true) {
            return false;
        }

        self.best = Some(c);

        true
    }
}

/// What each wire holds: row q is the parity of the input bits that qubit q carries, as a
/// mask over basis-index bits. Rows past the qubit count stay 0.
type Rows = [u8; MAX_UNITARY_QUBITS];

/// The rows of the identity: qubit q holds its own input bit.
fn identity_rows(n: usize) -> Rows {
    let mut rows = [0; MAX_UNITARY_QUBITS];

    for (q, row) in qubits(n).zip(rows.iter_mut()) {
        *row = u8::try_from(place(n, q)).unwrap_or(0);
    }

    rows
}

/// The basis index that the rows map input x to.
fn apply_rows(rows: &Rows, n: usize, x: u32) -> u32 {
    qubits(n)
        .zip(rows)
        .filter(|&(_, &row)| odd(u32::from(row), x))
        .fold(0, |out, (q, _)| out | place(n, q))
}

/// The bit of parity z in a set of parities.
fn bit(z: u8) -> u64 {
    1u64 << (z % 64)
}

/// A change of frame around the target: an optional CNOT, then H on a subset of the
/// qubits (bit q of `hadamard` is qubit q). The core is the target seen through the
/// frame; the answer is the frame, the core's circuit, and the frame undone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Frame {
    cnot: Option<(u8, u8)>,
    hadamard: u32,
}

impl Frame {
    /// The frame that changes nothing.
    const PLAIN: Frame = Frame {
        cnot: None,
        hadamard: 0,
    };

    /// The frame's gates in time order: the CNOT first, then the H gates.
    fn gates(self, n: usize) -> Vec<Gate> {
        let cnot = self
            .cnot
            .map(|(control, target)| Gate::Cnot { control, target });
        let hadamards = qubits(n)
            .filter(|&q| (self.hadamard >> q) & 1 == 1)
            .map(Gate::H);

        cnot.into_iter().chain(hadamards).collect()
    }
}

/// The frames a family's method tries, in order. Linear and Phase look at the target as it
/// is, Hadamard through every nonempty subset, and Compose through every CNOT, alone and
/// (with Hadamard known) followed by every subset.
fn frames(family: Family, skills: Skills, n: usize) -> Vec<Frame> {
    let subsets = 1..dim(n);

    match family {
        Family::Linear | Family::Phase => vec![Frame::PLAIN],
        Family::Hadamard => subsets
            .map(|hadamard| Frame {
                cnot: None,
                hadamard,
            })
            .collect(),
        Family::Compose => pairs(n)
            .into_iter()
            .flat_map(|cnot| {
                let hadamards = if skills.hadamard {
                    subsets.clone()
                } else {
                    1..1
                };

                std::iter::once(0)
                    .chain(hadamards)
                    .map(move |hadamard| Frame {
                        cnot: Some(cnot),
                        hadamard,
                    })
            })
            .collect(),
        Family::Search | Family::Meet | Family::Simplify | Family::Previous => Vec::new(),
    }
}

/// An affine permutation with phases: input x goes to w^phases[x] |A x xor shift>, where
/// row q of A is `rows[q]`. Phases are taken relative to input 0.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Monomial {
    rows: Rows,
    shift: u32,
    phases: Vec<u8>,
}

/// The core of `target` in `frame`: the frame undone, the target, then the frame. Some when
/// it maps every basis state to one basis state with a phase and that basis map is affine.
fn core(target: &Target, frame: Frame) -> Option<Monomial> {
    let n = target.qubits();
    let outer = frame.gates(n);
    let before = Circuit::new(outer.iter().rev().copied().collect()).ok()?;
    let after = Circuit::new(outer).ok()?;
    let mut image = Vec::new();

    for j in 0..dim(n) {
        let mut state = ExactState::basis(n, j).ok()?;

        state.run(&before).ok()?;

        let mut state = apply_target(target, &state).ok()?;

        state.run(&after).ok()?;
        state.reduce();
        image.push(state.as_unit()?);
    }

    affine(n, &image)
}

/// The affine structure of a basis map with phases, given as the image (basis, phase) of
/// every input in order. None when the basis map is not affine.
fn affine(n: usize, image: &[(u32, u8)]) -> Option<Monomial> {
    let &(shift, first_phase) = image.first()?;
    let mut rows = [0; MAX_UNITARY_QUBITS];

    // Input bit i adds column A e_i, the image of that bit alone without the shift.
    for i in 0..n {
        let &(b, _) = image.get(1 << i)?;
        let column = b ^ shift;

        for (q, row) in qubits(n).zip(rows.iter_mut()) {
            if column & place(n, q) != 0 {
                *row |= 1 << i;
            }
        }
    }

    let affine = (0u32..)
        .zip(image)
        .all(|(x, &(b, _))| apply_rows(&rows, n, x) ^ shift == b);

    affine.then(|| Monomial {
        rows,
        shift,
        phases: image
            .iter()
            .map(|&(_, p)| (p + 8 - first_phase) % 8)
            .collect(),
    })
}

/// The phase polynomial of a phase table (phases[0] = 0): coefficients a_z in Z8 with
/// phases[x] = sum over z of a_z (z.x) mod 8, where z.x is the parity of the input bits
/// in z. Index 0 is unused and 0. From the Walsh transform,
/// a_z = -2^(1-n) sum_x (-1)^(z.x) phases[x] mod 8, with the table read as integers 0..7.
/// None when a transform value does not divide by 2^(n-1). For n <= 4 it always divides
/// when the table has a phase polynomial at all; for 5 or 6 qubits it may not, and the
/// designer then knows no phase polynomial.
fn phase_polynomial(phases: &[u8], n: usize) -> Option<Vec<u8>> {
    let half = 1i64 << (n.clamp(1, MAX_UNITARY_QUBITS) - 1);

    (0u32..)
        .zip(phases)
        .map(|(z, _)| {
            if z == 0 {
                return Some(0);
            }

            let sum: i64 = (0u32..)
                .zip(phases)
                .map(|(x, &p)| {
                    if odd(z, x) {
                        -i64::from(p)
                    } else {
                        i64::from(p)
                    }
                })
                .sum();

            if sum % half != 0 {
                return None;
            }

            u8::try_from((-(sum / half)).rem_euclid(8)).ok()
        })
        .collect()
}

/// Coefficient vectors that add no phase to any input, on n qubits. Adding one to a phase
/// polynomial gives another polynomial with the same phases. Two kinds are used:
/// 4 on the three parities a, b, a xor b of a plane (p_a + p_b - p_(a xor b) = 2 p_a p_b,
/// so the sum is 8 (p_a p_b + p_(a xor b)) = 0 mod 8), and 2 on the seven nonzero parities
/// of a three-dimensional space (on any input either none or four of them are 1). For
/// n <= 3 their sums are the whole kernel, and each one is even, so the odd terms of a
/// polynomial (its T gates) never change. For more qubits only the zero vector is used.
fn kernel(n: usize) -> Vec<Vec<u8>> {
    let size = 1usize << n.min(MAX_UNITARY_QUBITS);
    let zero = vec![0u8; size];

    if n > 3 {
        return vec![zero];
    }

    let mut generators = Vec::new();

    for a in 1..size {
        for b in a + 1..size {
            if a ^ b > b {
                let mut g = zero.clone();

                for z in [a, b, a ^ b] {
                    g[z] = 4;
                }

                generators.push(g);
            }
        }
    }

    if n == 3 {
        let mut g = vec![2; size];

        g[0] = 0;
        generators.push(g);
    }

    let mut group = BTreeSet::from([zero]);

    loop {
        let grown: BTreeSet<Vec<u8>> = group
            .iter()
            .flat_map(|k| generators.iter().map(move |g| add(k, g)))
            .chain(group.iter().cloned())
            .collect();

        if grown.len() == group.len() {
            return group.into_iter().collect();
        }

        group = grown;
    }
}

/// The sum of two coefficient vectors mod 8.
fn add(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b).map(|(x, y)| (x + y) % 8).collect()
}

/// The parities a coefficient vector puts a phase on.
fn support(coefficients: &[u8]) -> u64 {
    (0u8..)
        .zip(coefficients)
        .filter(|&(_, &k)| k % 8 != 0)
        .fold(0, |set, (z, _)| set | bit(z))
}

/// The supports no other support lies inside: a network that visits one of them can place
/// the phases.
fn minimal_supports(representatives: &[Vec<u8>]) -> Vec<u64> {
    let all: BTreeSet<u64> = representatives.iter().map(|r| support(r)).collect();

    all.iter()
        .copied()
        .filter(|&s| !all.iter().any(|&t| t != s && t & !s == 0))
        .collect()
}

/// The gates that put w^k on qubit q: one gate, or two for k = 3 and k = 5.
fn phase_gates(q: u8, k: u8) -> Vec<Gate> {
    match k % 8 {
        1 => vec![Gate::T(q)],
        2 => vec![Gate::S(q)],
        3 => vec![Gate::S(q), Gate::T(q)],
        4 => vec![Gate::Z(q)],
        5 => vec![Gate::Z(q), Gate::T(q)],
        6 => vec![Gate::Sdg(q)],
        7 => vec![Gate::Tdg(q)],
        _ => Vec::new(),
    }
}

/// The phase gates a coefficient vector needs in all.
fn phase_gate_count(coefficients: &[u8]) -> usize {
    coefficients.iter().map(|&k| phase_gates(0, k).len()).sum()
}

/// X on every qubit the shift flips.
fn shift_gates(n: usize, shift: u32) -> impl Iterator<Item = Gate> {
    qubits(n)
        .filter(move |&q| shift & place(n, q) != 0)
        .map(Gate::X)
}

/// The qubits of a parity, in order.
fn members(n: usize, z: u8) -> Vec<u8> {
    qubits(n)
        .filter(|&q| u32::from(z) & place(n, q) != 0)
        .collect()
}

/// CNOTs the naive network spends: two per extra qubit of every parity with a phase.
fn naive_cnots(n: usize, coefficients: &[u8]) -> usize {
    (0u8..)
        .zip(coefficients)
        .filter(|&(_, &k)| k % 8 != 0)
        .map(|(z, _)| 2 * members(n, z).len().saturating_sub(1))
        .sum()
}

/// The naive phase network, for an identity linear part: each parity with a phase is
/// computed onto its last qubit, given its phase and uncomputed; then the shift's X gates.
fn naive(n: usize, coefficients: &[u8], shift: u32) -> Vec<Gate> {
    let mut gates = Vec::new();

    for (z, &k) in (0u8..).zip(coefficients) {
        let qs = members(n, z);
        let Some((&last, rest)) = qs.split_last() else {
            continue;
        };

        if k % 8 == 0 {
            continue;
        }

        let ladder: Vec<Gate> = rest
            .iter()
            .map(|&control| Gate::Cnot {
                control,
                target: last,
            })
            .collect();

        gates.extend(&ladder);
        gates.extend(phase_gates(last, k));
        gates.extend(ladder.iter().rev());
    }

    gates.extend(shift_gates(n, shift));
    gates
}

/// What a network search looks for: the rows to end on, and parity sets of which it must
/// visit at least one, so the phases can be placed.
struct Goal {
    rows: Rows,
    supports: Vec<u64>,
}

/// One state of a network search: the rows, the parities visited so far (only those a
/// support holds), and the state and CNOT it was reached from.
struct Node {
    rows: Rows,
    visited: u64,
    from: Option<(usize, (u8, u8))>,
}

/// The fewest CNOTs that end on the goal's rows and visit one of its supports, by
/// breadth-first search over (rows, visited parities) from the identity, where the input
/// parities count as visited. One step pays for every STATES_PER_STEP states reached.
/// None when the budget runs out or the search exceeds MAX_NETWORK_STATES.
fn network(n: usize, goal: &Goal, meter: &mut Meter) -> Option<Vec<(u8, u8)>> {
    let relevant = goal.supports.iter().fold(0, |all, s| all | s);
    let reached = |rows: &Rows, visited: u64| {
        *rows == goal.rows && goal.supports.iter().any(|&s| s & !visited == 0)
    };
    let start = identity_rows(n);
    let visited = start.iter().fold(0, |set, &z| set | bit(z)) & relevant;
    let moves = pairs(n);
    let mut nodes = Vec::new();
    let mut seen = BTreeSet::new();
    let mut queue = VecDeque::new();

    if !meter.pay(1) {
        return None;
    }

    if reached(&start, visited) {
        return Some(Vec::new());
    }

    nodes.push(Node {
        rows: start,
        visited,
        from: None,
    });
    seen.insert((start, visited));
    queue.push_back(0);

    while let Some(at) = queue.pop_front() {
        let Some(&Node { rows, visited, .. }) = nodes.get(at) else {
            break;
        };

        for &(c, t) in &moves {
            let mut next = rows;

            next[usize::from(t)] ^= next[usize::from(c)];

            let now = (visited | bit(next[usize::from(t)])) & relevant;

            if !seen.insert((next, now)) {
                continue;
            }

            if nodes.len() >= MAX_NETWORK_STATES
                || (nodes.len() % STATES_PER_STEP == 0 && !meter.pay(1))
            {
                return None;
            }

            nodes.push(Node {
                rows: next,
                visited: now,
                from: Some((at, (c, t))),
            });

            if reached(&next, now) {
                return Some(path(&nodes, nodes.len() - 1));
            }

            queue.push_back(nodes.len() - 1);
        }
    }

    None
}

/// The CNOTs that lead from the start to node `end`, in time order.
fn path(nodes: &[Node], end: usize) -> Vec<(u8, u8)> {
    let mut moves = Vec::new();
    let mut at = end;

    while let Some((parent, step)) = nodes.get(at).and_then(|node| node.from) {
        moves.push(step);
        at = parent;
    }

    moves.reverse();
    moves
}

/// The parities a CNOT network puts on some wire at some time, the inputs included.
fn visited_by(n: usize, moves: &[(u8, u8)]) -> u64 {
    let mut rows = identity_rows(n);
    let mut visited = rows.iter().fold(0, |set, &z| set | bit(z));

    for &(c, t) in moves {
        rows[usize::from(t)] ^= rows[usize::from(c)];
        visited |= bit(rows[usize::from(t)]);
    }

    visited
}

/// The core's circuit from a network: the CNOTs in order, each phase on the first wire
/// that holds its parity, then the shift's X gates. None when a phase finds no wire.
fn assemble(n: usize, moves: &[(u8, u8)], coefficients: &[u8], shift: u32) -> Option<Vec<Gate>> {
    let mut rows = identity_rows(n);
    let mut pending = coefficients.to_vec();
    let mut gates = Vec::new();
    let mut put = |gates: &mut Vec<Gate>, q: u8, z: u8| {
        if let Some(k) = pending.get_mut(usize::from(z)) {
            gates.extend(phase_gates(q, *k));
            *k = 0;
        }
    };

    for (q, &z) in qubits(n).zip(&rows) {
        put(&mut gates, q, z);
    }

    for &(c, t) in moves {
        gates.push(Gate::Cnot {
            control: c,
            target: t,
        });
        rows[usize::from(t)] ^= rows[usize::from(c)];
        put(&mut gates, t, rows[usize::from(t)]);
    }

    if support(&pending) != 0 {
        return None;
    }

    gates.extend(shift_gates(n, shift));

    Some(gates)
}

/// The input bits c with phases[x] = 4 (c.x) for every input x: phases that are Z gates on
/// the inputs. None for any other phase table.
fn z_flips(phases: &[u8], n: usize) -> Option<u32> {
    let flips = (0..n)
        .filter(|&i| phases.get(1 << i) == Some(&4))
        .fold(0, |c, i| c | (1 << i));
    let pauli = (0u32..)
        .zip(phases)
        .all(|(x, &p)| p == if odd(flips, x) { 4 } else { 0 });

    pauli.then_some(flips)
}

/// The circuit of a core with the known skills: a network search with Linear, the naive
/// phase network without it. Without Phase, Hadamard still builds Z gates on the inputs,
/// as H X H. None when the core needs a skill the Worker lacks (Linear for a linear part
/// other than the identity, Phase for other phases), when it has no phase polynomial, or
/// when the budget runs out. The Walsh transform costs 2^n steps.
fn build_core(m: &Monomial, n: usize, skills: Skills, meter: &mut Meter) -> Option<Vec<Gate>> {
    let identity = m.rows == identity_rows(n);
    let flat = m.phases.iter().all(|&p| p == 0);

    if !identity && !skills.linear {
        return None;
    }

    let flips = if flat || skills.phase {
        0
    } else if skills.hadamard {
        z_flips(&m.phases, n)?
    } else {
        return None;
    };

    let representatives = if flips != 0 || flat {
        vec![vec![0; m.phases.len()]]
    } else {
        if !meter.pay(u64::from(dim(n))) {
            return None;
        }

        let polynomial = phase_polynomial(&m.phases, n)?;

        kernel(n).iter().map(|k| add(&polynomial, k)).collect()
    };

    let mut gates: Vec<Gate> = shift_gates(n, flips)
        .flat_map(|x| match x {
            Gate::X(q) => vec![Gate::H(q), x, Gate::H(q)],
            other => vec![other],
        })
        .collect();

    if skills.linear {
        let goal = Goal {
            rows: m.rows,
            supports: minimal_supports(&representatives),
        };
        let moves = network(n, &goal, meter)?;
        let visited = visited_by(n, &moves);
        let fewest = representatives
            .iter()
            .filter(|r| support(r) & !visited == 0)
            .min_by_key(|r| phase_gate_count(r))?;

        gates.extend(assemble(n, &moves, fewest, m.shift)?);
    } else {
        let cheapest = representatives
            .iter()
            .min_by_key(|r| (naive_cnots(n, r), phase_gate_count(r)))?;

        gates.extend(naive(n, cheapest, m.shift));
    }

    Some(gates)
}

/// The candidate a frame gives: the frame, the core's circuit, and the frame undone. The
/// core test costs 2^n steps. None when the core is not an affine permutation with
/// phases, or when the known families cannot build it.
fn through(target: &Target, frame: Frame, skills: Skills, meter: &mut Meter) -> Option<Circuit> {
    let n = target.qubits();

    if !meter.pay(u64::from(dim(n))) {
        return None;
    }

    let m = core(target, frame)?;
    let inner = build_core(&m, n, skills, meter)?;
    let outer = frame.gates(n);
    let gates = outer
        .iter()
        .chain(&inner)
        .chain(outer.iter().rev())
        .copied()
        .collect();

    Circuit::new(gates).ok()
}

/// Steps for `count` units of work, at one step per unit.
fn steps(count: usize) -> u64 {
    u64::try_from(count).unwrap_or(u64::MAX)
}

/// Every gate on n qubits, in a fixed order: H, S, SDG, T, TDG, X and Z on each qubit, then
/// every CNOT.
fn word_moves(n: usize) -> Vec<Gate> {
    let singles: [fn(u8) -> Gate; 7] = [
        Gate::H,
        Gate::S,
        Gate::Sdg,
        Gate::T,
        Gate::Tdg,
        Gate::X,
        Gate::Z,
    ];
    let mut moves: Vec<Gate> = qubits(n)
        .flat_map(|q| singles.iter().map(move |make| make(q)))
        .collect();

    moves.extend(
        pairs(n)
            .into_iter()
            .map(|(control, target)| Gate::Cnot { control, target }),
    );
    moves
}

/// The canonical key of the checked columns of U X^mask, where `columns` are those of U:
/// column j of U X^mask is column j xor mask of U.
fn flipped_key(columns: &[ExactState], checked: &[u32], mask: u32) -> Vec<i64> {
    let picked: Vec<ExactState> = checked
        .iter()
        .filter_map(|&j| columns.get((j ^ mask) as usize).cloned())
        .collect();

    canonical_key(&picked)
}

/// The end of the circuit a side of a word search grows from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum End {
    /// Words that start at the identity and grow towards the output.
    Input,
    /// Words that start at the target and grow towards the input.
    Output,
}

/// Where a word of a search hangs: its parent and its last gate (None for a root), and the X
/// mask of its root.
type Link = (Option<(usize, Gate)>, u32);

/// One side of a word search: a tree of words, the keys it has seen, and the words of its
/// newest layer with all their columns.
#[derive(Default)]
struct Words {
    tree: Vec<Link>,
    seen: BTreeMap<Vec<i64>, usize>,
    layer: Vec<(usize, Vec<ExactState>)>,
}

impl Words {
    /// The gates of word `at` in the order the search added them, from its root on.
    fn path(&self, at: usize) -> Vec<Gate> {
        let mut gates = Vec::new();
        let mut at = at;

        while let Some(&(Some((parent, g)), _)) = self.tree.get(at) {
            gates.push(g);
            at = parent;
        }

        gates.reverse();
        gates
    }

    /// The X mask of the root of word `at`.
    fn root(&self, at: usize) -> u32 {
        self.tree.get(at).map_or(0, |&(_, mask)| mask)
    }
}

/// What adding words to a search did.
enum Added {
    /// Nothing that fits was met: the word was known or is new, or the layer is complete.
    Nothing,
    /// A new word met a word of the other side: the circuit they make.
    Met(Circuit),
    /// The budget or the column cap stops the search.
    Stopped,
}

/// A breadth-first search over gate words for a target U on n qubits, deduplicated by the
/// canonical key of the checked columns.
///
/// X gates cost no T and no CNOT, so X on the inputs and the outputs is free. Words grow
/// from the input end, starting at the identity, and (for Meet) from the output end,
/// starting at X^c U for every X mask c; a word B from the output end stands for B^-1 X^c U.
/// A word A meets B when A X^b equals B^-1 X^c U up to a global phase on the checked
/// columns, for some mask b, and the circuit is then X^b, A, B, X^c. Without Meet, words grow
/// from the input end only and meet the starting points X^c U.
///
/// A new word from the output end is matched under the masks b that map the checked columns
/// onto themselves: every mask with self-check, only 0 and 2^n - 1 without it. Its match
/// compares A with B^-1 X^c U X^b on the checked columns, which shifts them by b.
struct WordSearch<'a> {
    n: usize,
    checked: &'a [u32],
    max_gates: usize,
    input: Words,
    output: Words,
    /// Words added so far, on both sides.
    words: usize,
}

impl WordSearch<'_> {
    fn side(&self, end: End) -> &Words {
        match end {
            End::Input => &self.input,
            End::Output => &self.output,
        }
    }

    fn side_mut(&mut self, end: End) -> &mut Words {
        match end {
            End::Input => &mut self.input,
            End::Output => &mut self.output,
        }
    }

    /// True when X^mask maps the checked columns onto themselves.
    fn keeps_checked(&self, mask: u32) -> bool {
        self.checked
            .iter()
            .all(|&j| self.checked.contains(&(j ^ mask)))
    }

    /// Add a word with its columns to one side, unless its key is known there. One step
    /// pays for every COLUMNS_PER_STEP columns of new words, the starting points included.
    fn admit(
        &mut self,
        end: End,
        link: Link,
        columns: Vec<ExactState>,
        meter: &mut Meter,
    ) -> Added {
        let key = flipped_key(&columns, self.checked, 0);

        if self.side(end).seen.contains_key(&key) {
            return Added::Nothing;
        }

        let words_per_step = (COLUMNS_PER_STEP >> self.n).max(1);

        if self.words >= MAX_SEARCH_COLUMNS >> self.n
            || (self.words.is_multiple_of(words_per_step) && !meter.pay(1))
        {
            return Added::Stopped;
        }

        let other = match end {
            End::Input => End::Output,
            End::Output => End::Input,
        };

        // A word A from the input end is compared as A X^mask, so the circuit agrees with U
        // on the checked columns. A word from the output end is compared as B^-1 X^c U X^mask,
        // so the circuit agrees with U on the checked columns shifted by the mask: only masks
        // that keep them in place may match.
        let found: Vec<(u32, usize)> = (0..dim(self.n))
            .filter(|&mask| end == End::Input || self.keeps_checked(mask))
            .filter_map(|mask| {
                let flipped = if mask == 0 {
                    self.side(other).seen.get(&key)
                } else {
                    let key = flipped_key(&columns, self.checked, mask);

                    self.side(other).seen.get(&key)
                };

                flipped.map(|&found| (mask, found))
            })
            .collect();
        let at = self.side(end).tree.len();
        let side = self.side_mut(end);

        side.tree.push(link);
        side.seen.insert(key, at);
        side.layer.push((at, columns));
        self.words += 1;

        let met = found.into_iter().find_map(|(mask, found)| match end {
            End::Input => self.circuit(at, found, mask),
            End::Output => self.circuit(found, at, mask),
        });

        met.map_or(Added::Nothing, Added::Met)
    }

    /// The circuit X^mask, input word `input`, output word `output`, then X^c for the root c
    /// of the output word, when it has gates and fits the gate cap. Around two empty words
    /// the X gates merge.
    fn circuit(&self, input: usize, output: usize, mask: u32) -> Option<Circuit> {
        let mut first = self.input.path(input);
        let mut second = self.output.path(output);
        let root = self.output.root(output);

        // The output word was grown backwards: its first gate is the circuit's last.
        second.reverse();

        let gates: Vec<Gate> = if first.is_empty() && second.is_empty() {
            shift_gates(self.n, mask ^ root).collect()
        } else {
            let mut gates: Vec<Gate> = shift_gates(self.n, mask).collect();

            gates.append(&mut first);
            gates.append(&mut second);
            gates.extend(shift_gates(self.n, root));
            gates
        };

        if gates.is_empty() || gates.len() > self.max_gates {
            return None;
        }

        Circuit::new(gates).ok()
    }

    /// Add the next layer of one side: each move after every word of its newest layer (from
    /// the output end, before it), in order.
    fn grow(&mut self, end: End, moves: &[Gate], meter: &mut Meter) -> Added {
        let layer = std::mem::take(&mut self.side_mut(end).layer);

        for (at, states) in layer {
            let root = self.side(end).root(at);

            for &g in moves {
                let applied = match end {
                    End::Input => g,
                    End::Output => g.inverse(),
                };
                let mut child = states.clone();

                if child.iter_mut().any(|s| s.apply(applied).is_err()) {
                    continue;
                }

                child.iter_mut().for_each(ExactState::reduce);

                match self.admit(end, (Some((at, g)), root), child, meter) {
                    Added::Nothing => {}
                    done => return done,
                }
            }
        }

        Added::Nothing
    }
}

/// The first circuit a word search finds for `target` (see `WordSearch`). With `meet` the
/// words grow from both ends, the side with the smaller newest layer first; otherwise from
/// the input end only. None when the budget or the column cap runs out first, or when the
/// target has more than SEARCH_QUBITS qubits.
fn word_search(
    target: &Target,
    checked: &[u32],
    meet: bool,
    max_gates: usize,
    meter: &mut Meter,
) -> Option<Circuit> {
    let n = target.qubits();

    if n > SEARCH_QUBITS {
        return None;
    }

    let mut search = WordSearch {
        n,
        checked,
        max_gates,
        input: Words::default(),
        output: Words::default(),
        words: 0,
    };
    let identity = (0..dim(n))
        .map(|j| ExactState::basis(n, j).ok())
        .collect::<Option<Vec<ExactState>>>()?;
    let mut starts = vec![(End::Input, 0, identity)];

    for mask in 0..dim(n) {
        let mut states = target.states();

        for state in &mut states {
            for x in shift_gates(n, mask) {
                state.apply(x).ok()?;
            }
        }

        starts.push((End::Output, mask, states));
    }

    for (end, mask, states) in starts {
        match search.admit(end, (None, mask), states, meter) {
            Added::Nothing => {}
            Added::Met(c) => return Some(c),
            Added::Stopped => return None,
        }
    }

    let moves = word_moves(n);

    loop {
        let (inputs, outputs) = (search.input.layer.len(), search.output.layer.len());
        let end = match (meet, inputs, outputs) {
            (false, _, _) => End::Input,
            (true, 0, _) => End::Output,
            (true, i, o) if o == 0 || i <= o => End::Input,
            _ => End::Output,
        };

        if search.side(end).layer.is_empty() {
            return None;
        }

        match search.grow(end, &moves, meter) {
            Added::Nothing => {}
            Added::Met(c) => return Some(c),
            Added::Stopped => return None,
        }
    }
}

/// The qubit and the power k of a phase gate, which puts w^k on |1>: Z 4, S 2, SDG 6, T 1,
/// TDG 7. None for H, X and CNOT.
fn power(g: Gate) -> Option<(u8, u8)> {
    match g {
        Gate::Z(q) => Some((q, 4)),
        Gate::S(q) => Some((q, 2)),
        Gate::Sdg(q) => Some((q, 6)),
        Gate::T(q) => Some((q, 1)),
        Gate::Tdg(q) => Some((q, 7)),
        Gate::H(_) | Gate::X(_) | Gate::Cnot { .. } => None,
    }
}

/// True when two gates share a qubit.
fn shares(a: Gate, b: Gate) -> bool {
    let (a0, a1) = a.qubits();
    let (b0, b1) = b.qubits();
    let on_b = |q: u8| q == b0 || Some(q) == b1;

    on_b(a0) || a1.is_some_and(on_b)
}

/// True when a then b equals b then a, up to a global phase, by a rule that is always valid:
/// gates on different qubits, two equal gates, two phase gates, a phase gate on the control
/// of a CNOT, an X on its target, and two CNOTs where neither one's control is the other's
/// target.
fn commutes(a: Gate, b: Gate) -> bool {
    if !shares(a, b) || a == b || (power(a).is_some() && power(b).is_some()) {
        return true;
    }

    match (a, b) {
        (Gate::Cnot { control, target }, other) | (other, Gate::Cnot { control, target }) => {
            match other {
                Gate::Cnot {
                    control: c,
                    target: t,
                } => control != t && target != c,
                Gate::X(q) => q == target,
                g => power(g).is_some_and(|(q, _)| q == control),
            }
        }
        _ => false,
    }
}

/// What a then b on the same qubits makes when it is at most one gate: nothing for an
/// inverse pair, one phase gate for two phase gates whose powers add up to one gate. None
/// otherwise.
fn combine(a: Gate, b: Gate) -> Option<Option<Gate>> {
    if a.inverse() == b {
        return Some(None);
    }

    let ((qa, ka), (qb, kb)) = (power(a)?, power(b)?);

    if qa != qb {
        return None;
    }

    match phase_gates(qa, ka + kb).as_slice() {
        [] => Some(None),
        [one] => Some(Some(*one)),
        _ => None,
    }
}

/// One pass of the peephole rules. Each gate moves later past the gates it commutes with
/// until it meets one it combines with (see `combine`): an inverse pair cancels, and two
/// phase gates on one qubit merge. Some with the new gates when anything changed.
fn peephole_pass(gates: &[Gate]) -> Option<Vec<Gate>> {
    let mut slots: Vec<Option<Gate>> = gates.iter().copied().map(Some).collect();
    let mut changed = false;

    for i in 0..slots.len() {
        let Some(Some(g)) = slots.get(i).copied() else {
            continue;
        };

        for j in i + 1..slots.len() {
            let Some(Some(h)) = slots.get(j).copied() else {
                continue;
            };

            if !shares(g, h) {
                continue;
            }

            if let Some(merged) = combine(g, h) {
                slots[i] = None;
                slots[j] = merged;
                changed = true;
                break;
            }

            if !commutes(g, h) {
                break;
            }
        }
    }

    changed.then(|| slots.into_iter().flatten().collect())
}

/// Peephole passes until nothing changes, at one step per gate for each pass. None when
/// the budget runs out.
fn peephole(gates: Vec<Gate>, meter: &mut Meter) -> Option<Vec<Gate>> {
    let mut gates = gates;

    loop {
        if !meter.pay(steps(gates.len())) {
            return None;
        }

        match peephole_pass(&gates) {
            Some(next) => gates = next,
            None => return Some(gates),
        }
    }
}

/// An affine parity of the path variables a wire holds: the variables as a bit set, and a
/// constant that X gates flip.
type Parity = (u128, bool);

/// The wires' parities after `g`. A fresh path variable `fresh` starts on the wire of an H.
fn advance(wires: &mut [Parity], g: Gate, fresh: &mut u32) -> Option<()> {
    match g {
        Gate::H(q) => {
            let wire = wires.get_mut(usize::from(q))?;

            *wire = (1u128.checked_shl(*fresh)?, false);
            *fresh += 1;
        }
        Gate::X(q) => wires.get_mut(usize::from(q))?.1 ^= true,
        Gate::Cnot { control, target } => {
            let (mask, flip) = *wires.get(usize::from(control))?;
            let wire = wires.get_mut(usize::from(target))?;

            wire.0 ^= mask;
            wire.1 ^= flip;
        }
        _ => {}
    }

    Some(())
}

/// Phase folding (rotation merging, Nam, Ross, Su, Childs and Maslov 2018). Written as a sum
/// over paths, a circuit has one path variable per input and one more after each H, and
/// every wire holds an affine parity of them. A phase gate puts w^k on the parity its wire
/// holds, and since only the sum of these terms matters, the terms on one parity add up and
/// the sum may sit wherever a wire holds that parity. Each sum goes to the first place where
/// its parity is held, or the last with `last`; a parity with the constant 1 takes the
/// negated power, up to a global phase. The other gates keep their order, so this is exact,
/// and it never adds a T gate. None when the path variables do not fit 128 bits.
fn fold(gates: &[Gate], n: usize, last: bool) -> Option<Vec<Gate>> {
    let start: Vec<Parity> = (0..n)
        .map(|q| Some((1u128.checked_shl(u32::try_from(q).ok()?)?, false)))
        .collect::<Option<_>>()?;
    let first_fresh = u32::try_from(n).ok()?;
    let negated = |k: u8, flip: bool| if flip { (8 - k % 8) % 8 } else { k % 8 };

    // The summed power on each parity.
    let mut wires = start.clone();
    let mut fresh = first_fresh;
    let mut sums: BTreeMap<u128, u8> = BTreeMap::new();

    for &g in gates {
        match power(g) {
            Some((q, k)) => {
                let (mask, flip) = *wires.get(usize::from(q))?;
                let sum = sums.entry(mask).or_insert(0);

                *sum = (*sum + negated(k, flip)) % 8;
            }
            None => advance(&mut wires, g, &mut fresh)?,
        }
    }

    // Where each sum goes: after `position` of the other gates, on a wire that holds it.
    let others: Vec<Gate> = gates
        .iter()
        .copied()
        .filter(|&g| power(g).is_none())
        .collect();
    let mut wires = start;
    let mut fresh = first_fresh;
    let mut places: BTreeMap<u128, (usize, u8, bool)> = BTreeMap::new();

    for position in 0..=others.len() {
        for (q, &(mask, flip)) in (0u8..).zip(&wires) {
            let wanted = sums.get(&mask).is_some_and(|&k| k != 0);

            if wanted && (last || !places.contains_key(&mask)) {
                places.insert(mask, (position, q, flip));
            }
        }

        if let Some(&g) = others.get(position) {
            advance(&mut wires, g, &mut fresh)?;
        }
    }

    let mut phases: BTreeMap<usize, Vec<Gate>> = BTreeMap::new();

    for (mask, &(position, q, flip)) in &places {
        let k = sums.get(mask).copied().unwrap_or(0);

        phases
            .entry(position)
            .or_default()
            .extend(phase_gates(q, negated(k, flip)));
    }

    let mut out = Vec::with_capacity(gates.len());

    for position in 0..=others.len() {
        if let Some(placed) = phases.get(&position) {
            out.extend(placed);
        }

        if let Some(&g) = others.get(position) {
            out.push(g);
        }
    }

    Some(out)
}

/// The gates of a window: a CNOT, then every later gate that acts on its two qubits only, is
/// Clifford, and commutes with each gate between them that was left out; the window's gates
/// can therefore be gathered right after the CNOT. Returns the two qubits (lower first) and
/// the window's indices, the CNOT's first; None when the gate at `start` is not a CNOT.
fn window_at(gates: &[Gate], start: usize) -> Option<((u8, u8), Vec<usize>)> {
    let Some(&Gate::Cnot { control, target }) = gates.get(start) else {
        return None;
    };
    let pair = (control.min(target), control.max(target));
    let inside = |q: u8| q == pair.0 || q == pair.1;
    let mut members = vec![start];
    let mut left_out: Vec<Gate> = Vec::new();

    for (i, &g) in gates.iter().enumerate().skip(start + 1) {
        let (a, b) = g.qubits();

        if !inside(a) && !b.is_some_and(inside) {
            continue;
        }

        let fits = inside(a) && b.is_none_or(inside) && !g.is_t();

        if fits && left_out.iter().all(|&o| commutes(o, g)) {
            members.push(i);
        } else {
            left_out.push(g);
        }
    }

    Some((pair, members))
}

/// The moves of the Clifford table on qubits 0 and 1 that keep the CNOT count: H and S on
/// each qubit. SDG, X and Z are left out, since they differ from S or from nothing by a
/// Pauli gate.
const LOCAL_MOVES: [Gate; 4] = [Gate::H(0), Gate::H(1), Gate::S(0), Gate::S(1)];

/// The moves of the Clifford table that add one CNOT.
const CNOT_MOVES: [Gate; 2] = [
    Gate::Cnot {
        control: 0,
        target: 1,
    },
    Gate::Cnot {
        control: 1,
        target: 0,
    },
];

/// The Pauli gate P on qubits 0 and 1 with index p < 16: X on qubit q when bit q of p is
/// set, then Z on qubit q when bit q + 2 is set.
fn pauli(p: u32) -> impl Iterator<Item = Gate> {
    let xs = (0u8..2).filter(move |&q| (p >> q) & 1 == 1).map(Gate::X);
    let zs = (0u8..2)
        .filter(move |&q| (p >> (q + 2)) & 1 == 1)
        .map(Gate::Z);

    xs.chain(zs)
}

/// A word of the Clifford table: its parent and last gate, its CNOT count and its columns.
type CliffordWord = (Option<(usize, Gate)>, u32, Vec<ExactState>);

/// The two-qubit Cliffords up to a Pauli gate after them, each with a word of fewest CNOTs.
/// This is a Clifford meet in the middle under a CNOT cap: the identity side is enumerated
/// breadth-first in order of CNOT count, shared by all windows of a call, and grown only as
/// far as the windows need; the window side is the window itself, matched under all 16
/// Paulis. One step pays for every CLASSES_PER_STEP classes.
#[derive(Default)]
struct Cliffords {
    words: Vec<CliffordWord>,
    /// The canonical key of P V for every word V and Pauli P, with the word and P's index.
    variants: BTreeMap<Vec<i64>, (usize, u32)>,
    /// The words with the most CNOTs so far.
    layer: Vec<usize>,
    /// How many CNOT counts are complete: every class with fewer CNOTs is known.
    complete: u32,
}

impl Cliffords {
    /// Add a word unless its class is known: then its key is one of the variants. The new
    /// class's 16 variants are walked in Gray code order, one Pauli gate apart; the order of
    /// the X and Z gates changes only a global phase. None when the budget runs out.
    fn admit(
        &mut self,
        from: Option<(usize, Gate)>,
        columns: Vec<ExactState>,
        cnots: u32,
        meter: &mut Meter,
    ) -> Option<Option<usize>> {
        let key = canonical_key(&columns);

        if self.variants.contains_key(&key) {
            return Some(None);
        }

        if self.words.len().is_multiple_of(CLASSES_PER_STEP) && !meter.pay(1) {
            return None;
        }

        let at = self.words.len();
        let flips = [Gate::X(0), Gate::X(1), Gate::Z(0), Gate::Z(1)];
        let mut walked = columns.clone();

        self.variants.insert(key, (at, 0));

        for i in 1u32..16 {
            let g = *flips.get(i.trailing_zeros() as usize)?;

            for state in &mut walked {
                state.apply(g).ok()?;
            }

            self.variants
                .entry(canonical_key(&walked))
                .or_insert((at, i ^ (i >> 1)));
        }

        self.words.push((from, cnots, columns));

        Some(Some(at))
    }

    /// The words reached from `seeds` by local moves, breadth-first, all with `cnots`
    /// CNOTs; the seeds included. None when the budget runs out.
    fn close(&mut self, seeds: Vec<usize>, cnots: u32, meter: &mut Meter) -> Option<Vec<usize>> {
        let mut reached = seeds.clone();
        let mut queue: VecDeque<usize> = seeds.into();

        while let Some(at) = queue.pop_front() {
            for g in LOCAL_MOVES {
                let Some(next) = self.child(at, g) else {
                    continue;
                };

                if let Some(new) = self.admit(Some((at, g)), next, cnots, meter)? {
                    reached.push(new);
                    queue.push_back(new);
                }
            }
        }

        Some(reached)
    }

    /// The columns of word `at` followed by gate g.
    fn child(&self, at: usize, g: Gate) -> Option<Vec<ExactState>> {
        let mut columns = self.words.get(at)?.2.clone();

        for state in &mut columns {
            state.apply(g).ok()?;
            state.reduce();
        }

        Some(columns)
    }

    /// Complete the classes up to `cnots` CNOTs. False when the budget runs out first.
    fn grow_to(&mut self, cnots: u32, meter: &mut Meter) -> bool {
        while self.complete <= cnots {
            let layer = if self.complete == 0 {
                let Some(identity) = (0..4)
                    .map(|j| ExactState::basis(2, j).ok())
                    .collect::<Option<Vec<ExactState>>>()
                else {
                    return false;
                };
                let Some(Some(root)) = self.admit(None, identity, 0, meter) else {
                    return false;
                };

                self.close(vec![root], 0, meter)
            } else {
                let mut seeds = Vec::new();
                let layer = std::mem::take(&mut self.layer);

                for at in layer {
                    for g in CNOT_MOVES {
                        let Some(next) = self.child(at, g) else {
                            continue;
                        };

                        match self.admit(Some((at, g)), next, self.complete, meter) {
                            Some(Some(new)) => seeds.push(new),
                            Some(None) => {}
                            None => return false,
                        }
                    }
                }

                self.close(seeds, self.complete, meter)
            };

            match layer {
                Some(layer) => self.layer = layer,
                None => return false,
            }

            self.complete += 1;
        }

        true
    }

    /// A word for the two-qubit Clifford `window` with fewer than `cnots` CNOTs, if one
    /// exists: the word of its class, then the Pauli gate that turns it into the window.
    fn rebuild(&mut self, window: &[Gate], cnots: u32, meter: &mut Meter) -> Option<Vec<Gate>> {
        let cap = cnots.checked_sub(1)?.min(3);
        let unitary = columns(2, &Circuit::new(window.to_vec()).ok()?).ok()?;

        if !self.grow_to(cap, meter) {
            return None;
        }

        let &(at, p) = self.variants.get(&canonical_key(&unitary))?;

        if self.words.get(at)?.1 > cap {
            return None;
        }

        let mut gates = Vec::new();
        let mut step = at;

        while let Some(&(Some((parent, g)), _, _)) = self.words.get(step) {
            gates.push(g);
            step = parent;
        }

        gates.reverse();
        gates.extend(pauli(p));

        Some(gates)
    }
}

/// Rebuild two-qubit Clifford windows (see `window_at`) with fewer CNOTs from the Clifford
/// table, the first window that improves at a time, until none does. Each scan of the
/// circuit costs one step per gate, and each rebuilt circuit is cleaned by peephole passes.
fn rewindow(gates: Vec<Gate>, table: &mut Cliffords, meter: &mut Meter) -> Vec<Gate> {
    let mut gates = gates;

    'scan: loop {
        if !meter.pay(steps(gates.len())) {
            return gates;
        }

        for start in 0..gates.len() {
            let Some(((p, q), members)) = window_at(&gates, start) else {
                continue;
            };
            let cnots = members
                .iter()
                .filter(|&&i| matches!(gates.get(i), Some(Gate::Cnot { .. })))
                .count();

            if cnots < 2 {
                continue;
            }

            // The window on qubits 0 and 1, and the way back.
            let mut inward = vec![u8::MAX; usize::from(p.max(q)) + 1];

            inward[usize::from(p)] = 0;
            inward[usize::from(q)] = 1;

            let local: Option<Vec<Gate>> = members
                .iter()
                .map(|&i| gates.get(i)?.relabel(&inward))
                .collect();
            let Some(local) = local else {
                continue;
            };
            let Some(word) = table.rebuild(&local, u32::try_from(cnots).unwrap_or(u32::MAX), meter)
            else {
                if meter.exhausted {
                    return gates;
                }

                continue;
            };
            let Some(outward) = word
                .iter()
                .map(|g| g.relabel(&[p, q]))
                .collect::<Option<Vec<Gate>>>()
            else {
                continue;
            };
            let mut next: Vec<Gate> = gates.iter().take(start).copied().collect();

            next.extend(outward);
            next.extend(
                gates
                    .iter()
                    .enumerate()
                    .skip(start + 1)
                    .filter(|(i, _)| !members.contains(i))
                    .map(|(_, &g)| g),
            );

            match peephole(next.clone(), meter) {
                Some(cleaned) => gates = cleaned,
                None => return next,
            }

            continue 'scan;
        }

        return gates;
    }
}

/// Simplify one circuit: peephole passes, then phase folding to the first and to the last
/// places with window rebuilding after each. Every result is offered to the pool, which
/// verifies it, as soon as it exists. Folding costs one step per gate.
fn simplify(c: &Circuit, n: usize, pool: &mut Pool<'_>, table: &mut Cliffords, meter: &mut Meter) {
    let offer = |gates: Vec<Gate>, pool: &mut Pool<'_>, meter: &mut Meter| {
        if let Ok(c) = Circuit::new(gates) {
            pool.offer(c, meter);
        }
    };
    let Some(base) = peephole(c.gates().to_vec(), meter) else {
        return;
    };

    offer(base.clone(), pool, meter);

    for last in [false, true] {
        if !meter.pay(steps(base.len())) {
            return;
        }

        let Some(folded) = fold(&base, n, last) else {
            continue;
        };
        let Some(folded) = peephole(folded, meter) else {
            return;
        };

        offer(rewindow(folded, table, meter), pool, meter);
    }
}

/// Run the methods in order, offer every candidate to a pool, and return the cheapest
/// verified one.
fn design(
    task: &ParsedCircuitPrompt,
    order: &[Family],
    self_check: bool,
    meter: &mut Meter,
) -> Option<Circuit> {
    let target = &task.target;
    let n = target.qubits();
    let skills = Skills::of(order);
    let meet = order.contains(&Family::Meet);
    let mut pool = Pool::new(target, self_check, task.max_gates);
    let mut tried = BTreeSet::new();
    let mut searched = false;
    let mut table = Cliffords::default();

    for &family in order {
        match family {
            Family::Linear | Family::Phase | Family::Hadamard | Family::Compose => {
                for frame in frames(family, skills, n) {
                    if !tried.insert(frame) {
                        continue;
                    }

                    if let Some(c) = through(target, frame, skills, meter) {
                        pool.offer(c, meter);
                    }
                }
            }
            // One word search, at the first of the two: meet in the middle when Meet is
            // known.
            Family::Search | Family::Meet if !searched => {
                searched = true;

                let checked = pool.checked.clone();

                if let Some(c) = word_search(target, &checked, meet, task.max_gates, meter) {
                    pool.offer(c, meter);
                }
            }
            Family::Search | Family::Meet => {}
            // The cheapest candidate so far, then the earlier answers, latest first.
            Family::Simplify => {
                let mut inputs: Vec<Circuit> = pool.best.iter().cloned().collect();

                for c in task.previous.iter().rev() {
                    if !inputs.contains(c) {
                        inputs.push(c.clone());
                    }
                }

                for c in &inputs {
                    simplify(c, n, &mut pool, &mut table, meter);
                }
            }
            Family::Previous => {
                for c in task.previous.iter().rev() {
                    pool.offer(c.clone(), meter);
                }
            }
        }
    }

    pool.best
}

/// The judge's verdict, "VERDICT A" or "VERDICT B". Both candidates are verified within the
/// budget; when both checks ran and exactly one passed, it wins. Otherwise the stated costs
/// decide (T, then CNOT), and a seeded coin breaks a tie.
fn judge(
    target: &Target,
    (a, b): (&Circuit, &Circuit),
    self_check: bool,
    meter: &mut Meter,
    rng: &mut ChaCha8Rng,
) -> &'static str {
    let checked = checked_columns(target.qubits(), self_check);
    let verdict_a = verify(target, a, &checked, meter);
    let verdict_b = verify(target, b, &checked, meter);

    if let (Some(a_passes), Some(b_passes)) = (verdict_a, verdict_b)
        && a_passes != b_passes
    {
        return if a_passes { "VERDICT A" } else { "VERDICT B" };
    }

    let stated = |c: &Circuit| {
        let cost = c.cost();

        (cost.t, cost.cnot)
    };

    match stated(a).cmp(&stated(b)) {
        Ordering::Less => "VERDICT A",
        Ordering::Greater => "VERDICT B",
        Ordering::Equal if rng.random_bool(0.5) => "VERDICT A",
        Ordering::Equal => "VERDICT B",
    }
}

/// Chance that temperature shuffles the method order: none at 0, one half at the ceiling.
fn shuffle_chance(temperature: f32) -> f64 {
    let t = f64::from(temperature) / f64::from(TEMPERATURE_CEILING);

    if t.is_nan() {
        0.0
    } else {
        0.5 * t.clamp(0.0, 1.0)
    }
}

/// The solve reply cut to `max_tokens` by dropping trailing gates; the ANSWER line stays as
/// long as it fits at all.
fn answer_reply(answer: &Circuit, max_tokens: u32) -> (String, StopReason) {
    let gates = answer.gates();

    for keep in (0..=gates.len()).rev() {
        let Ok(head) = Circuit::new(gates.iter().take(keep).copied().collect()) else {
            continue;
        };
        let text = render_circuit_answer(&head);

        if approx_tokens(&text) <= max_tokens {
            let stop = if keep == gates.len() {
                StopReason::Stop
            } else {
                StopReason::Length
            };

            return (text, stop);
        }
    }

    (String::new(), StopReason::Length)
}

/// Any other reply cut to `max_tokens` words.
fn word_reply(text: &str, max_tokens: u32) -> (String, StopReason) {
    if approx_tokens(text) <= max_tokens {
        return (text.to_string(), StopReason::Stop);
    }

    let words = usize::try_from(max_tokens).unwrap_or(usize::MAX);
    let kept: Vec<&str> = text.split_whitespace().take(words).collect();

    (kept.join(" "), StopReason::Length)
}

/// A simulated circuit designer. NOT a language model: a positive control whose skill
/// depends on the system prompt and Traits only.
#[derive(Clone, Debug)]
pub struct SimDesigner {
    tag: String,
    /// Thinking tokens one step of work costs.
    step_tokens: u32,
}

impl SimDesigner {
    /// The small model "sim:small": 4 thinking tokens per step.
    pub fn small() -> SimDesigner {
        SimDesigner {
            tag: "sim:small".to_string(),
            step_tokens: 4,
        }
    }

    /// The large model "sim:large": 2 thinking tokens per step, so the same budget does
    /// twice the work.
    pub fn large() -> SimDesigner {
        SimDesigner {
            tag: "sim:large".to_string(),
            step_tokens: 2,
        }
    }

    /// The simulated designer with this tag, if there is one.
    pub fn from_tag(tag: &str) -> Option<SimDesigner> {
        [SimDesigner::small(), SimDesigner::large()]
            .into_iter()
            .find(|m| m.tag == tag)
    }

    /// The model tag, as on the allow-list of `CascadeLimits`.
    pub fn tag(&self) -> &str {
        &self.tag
    }

    fn response(
        &self,
        text: String,
        stop: StopReason,
        prompt_tokens: u32,
        thinking: u32,
    ) -> GenResponse {
        GenResponse {
            completion_tokens: thinking.saturating_add(approx_tokens(&text)),
            prompt_tokens,
            text,
            stop_reason: stop,
            backend: self.tag.clone(),
        }
    }
}

impl ModelBackend for SimDesigner {
    fn name(&self) -> &str {
        &self.tag
    }

    fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
        let prompt_tokens = approx_tokens(&req.system).saturating_add(approx_tokens(&req.prompt));

        let Some(task) = parse_circuit_prompt(&req.prompt) else {
            let (text, stop) = word_reply(CANNOT_READ, req.max_tokens);

            return Ok(self.response(text, stop, prompt_tokens, 0));
        };

        let mut order = known(&req.system);
        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(req.seed, "designer"));

        if rng.random::<f64>() < shuffle_chance(req.temperature) {
            order.shuffle(&mut rng);
        }

        let self_check = req.system.contains(CIRCUIT_SELF_CHECK_TEXT);
        let mut meter = Meter::new(req.think_tokens, self.step_tokens);

        let (text, stop) = match &task.candidates {
            Some((a, b)) => {
                let verdict = judge(&task.target, (a, b), self_check, &mut meter, &mut rng);

                word_reply(verdict, req.max_tokens)
            }
            None => {
                let fallback = || {
                    task.previous
                        .iter()
                        .rev()
                        .find(|c| !c.gates().is_empty() && c.gates().len() <= task.max_gates)
                        .cloned()
                };

                match design(&task, &order, self_check, &mut meter).or_else(fallback) {
                    Some(answer) => answer_reply(&answer, req.max_tokens),
                    None => word_reply(NO_CIRCUIT, req.max_tokens),
                }
            }
        };

        Ok(self.response(text, stop, prompt_tokens, meter.thinking(self.step_tokens)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cascade::circuit_text::{
        parse_circuit_answer, render_circuit_judge_prompt, render_circuit_solve_prompt,
    };
    use crate::cascade::qubit::{Column, Dressing, check};

    /// Strategy texts like those of the seed Workers: each names one family.
    const LINEAR: &str = "Treat each CNOT as a row operation: it adds one bit into another by XOR.";
    const PHASE: &str = "Write the phase of each basis state as a sum over parities of the bits; one T or TDG per odd term.";
    const HADAMARD: &str =
        "Use Hadamard gates to change the basis, so that bit flips become sign flips.";
    const SEARCH: &str = "Try short sequences, one gate at a time.";

    /// Strategy texts of the other families.
    const COMPOSE: &str = "Conjugate the target by one CNOT on both sides.";
    const PREVIOUS: &str = "Check the previous answer before you build your own.";
    const MEET: &str = "Meet in the middle: grow sequences from both ends.";
    const SIMPLIFY: &str = "Cancel inverse pairs and merge phase gates to simplify the circuit.";

    /// The strategy texts joined, with self-check.
    fn checking(families: &[&str]) -> String {
        format!("{} {CIRCUIT_SELF_CHECK_TEXT}", families.join(" "))
    }

    /// The strong prompt of the calibration: the families in the order Hadamard, Phase,
    /// Linear, Compose, Simplify, Search, Meet, with self-check.
    fn strong() -> String {
        checking(&[HADAMARD, PHASE, LINEAR, COMPOSE, SIMPLIFY, SEARCH, MEET])
    }

    /// The role and format texts of the seeds; they name no family.
    const ROLE: &str =
        "You design quantum circuits from the gates H, S, SDG, T, TDG, X, Z and CNOT.";
    const FORMAT: &str =
        "Answer with the line ANSWER, then one gate per line, such as H 0 or CNOT 0 1.";

    fn monomial(n: usize, image: impl Fn(u32) -> (u32, u8)) -> Target {
        Target::monomial(n, image).expect("unitary")
    }

    fn column(terms: Vec<(u32, u8)>) -> Column {
        Column::new(terms).expect("column")
    }

    // The eleven targets, from their definitions. Qubit 0 is the most significant bit, so
    // on three qubits the input |abc> is 4a + 2b + c.

    /// diag(1, 1, 1, -1).
    fn cz() -> Target {
        monomial(2, |j| (j, if j == 3 { 4 } else { 0 }))
    }

    /// |ab> -> |ba>.
    fn swap() -> Target {
        monomial(2, |j| (((j & 1) << 1) | (j >> 1), 0))
    }

    /// diag(1, 1, 1, i).
    fn cs() -> Target {
        monomial(2, |j| (j, if j == 3 { 2 } else { 0 }))
    }

    /// |0><0| x I + |1><1| x H: H on qubit 1, controlled by qubit 0.
    fn ch() -> Target {
        let columns = vec![
            Column::single(0, 0),
            Column::single(1, 0),
            column(vec![(2, 0), (3, 0)]),
            column(vec![(2, 0), (3, 4)]),
        ];

        Target::new(2, columns).expect("unitary")
    }

    /// The two-qubit Fourier transform: entry (b, j) is i^(b j) / 2.
    fn qft2() -> Target {
        let columns = (0..4u32)
            .map(|j| {
                let terms = (0..4u32)
                    .map(|b| (b, u8::try_from(2 * j * b % 8).expect("phase")))
                    .collect();

                column(terms)
            })
            .collect();

        Target::new(2, columns).expect("unitary")
    }

    /// diag(1, ..., 1, -1).
    fn ccz() -> Target {
        monomial(3, |j| (j, if j == 7 { 4 } else { 0 }))
    }

    /// |a,b,c> -> |a,b,c xor ab>.
    fn toffoli() -> Target {
        monomial(3, |j| (if j >= 6 { j ^ 1 } else { j }, 0))
    }

    /// |a,b,c> -> |a, a xor b, c xor ab>.
    fn peres() -> Target {
        monomial(3, |j| {
            let (a, b, c) = (j >> 2, (j >> 1) & 1, j & 1);

            ((a << 2) | ((a ^ b) << 1) | (c ^ (a & b)), 0)
        })
    }

    /// a = 1 swaps b and c: |101> and |110> trade places.
    fn fredkin() -> Target {
        monomial(3, |j| (if j == 5 || j == 6 { j ^ 3 } else { j }, 0))
    }

    /// The identity except |101> -> -|101>, |110> -> i|111>, |111> -> -i|110>.
    fn rtof() -> Target {
        monomial(3, |j| match j {
            5 => (5, 4),
            6 => (7, 2),
            7 => (6, 6),
            _ => (j, 0),
        })
    }

    /// The identity except |110> -> i|111>, |111> -> i|110>.
    fn ccix() -> Target {
        monomial(3, |j| match j {
            6 => (7, 2),
            7 => (6, 2),
            _ => (j, 0),
        })
    }

    fn circuit(text: &str) -> Circuit {
        text.parse().expect("circuit text")
    }

    fn rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }

    /// A complex amplitude (re, im) of the dense check below.
    type Amp = (f64, f64);

    /// Apply one gate to a dense state vector in f64. It shares no code with `qubit.rs`, so
    /// a bug there cannot hide here.
    fn dense_gate(n: usize, g: Gate, v: &mut [Amp]) {
        use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4};

        let len = v.len();
        let place = |q: u8| 1usize << (n - 1 - usize::from(q));
        let low = |p: usize| -> Vec<usize> { (0..len).filter(|i| i & p == 0).collect() };
        let turn = |v: &mut [Amp], q: u8, eighths: f64| {
            let (c, s) = ((eighths * FRAC_PI_4).cos(), (eighths * FRAC_PI_4).sin());

            for (i, a) in v.iter_mut().enumerate() {
                if i & place(q) != 0 {
                    *a = (a.0 * c - a.1 * s, a.0 * s + a.1 * c);
                }
            }
        };

        match g {
            Gate::H(q) => {
                for i in low(place(q)) {
                    let (a, b) = (v[i], v[i | place(q)]);

                    v[i] = ((a.0 + b.0) * FRAC_1_SQRT_2, (a.1 + b.1) * FRAC_1_SQRT_2);
                    v[i | place(q)] = ((a.0 - b.0) * FRAC_1_SQRT_2, (a.1 - b.1) * FRAC_1_SQRT_2);
                }
            }
            Gate::S(q) => turn(v, q, 2.0),
            Gate::Sdg(q) => turn(v, q, -2.0),
            Gate::T(q) => turn(v, q, 1.0),
            Gate::Tdg(q) => turn(v, q, -1.0),
            Gate::Z(q) => turn(v, q, 4.0),
            Gate::X(q) => {
                for i in low(place(q)) {
                    v.swap(i, i | place(q));
                }
            }
            Gate::Cnot { control, target } => {
                for i in low(place(target)) {
                    if i & place(control) != 0 {
                        v.swap(i, i | place(target));
                    }
                }
            }
        }
    }

    /// True when `c` equals `target` up to one global phase, within 1e-9, by dense f64
    /// simulation of every column.
    fn dense_agrees(target: &Target, c: &Circuit) -> bool {
        let n = target.qubits();
        let mut got = Vec::new();
        let mut want = Vec::new();

        for (j, col) in target.columns().iter().enumerate() {
            let mut v = vec![(0.0, 0.0); 1 << n];
            let mut t = vec![(0.0, 0.0); 1 << n];
            let scale = 1.0 / (col.terms().len() as f64).sqrt();

            v[j] = (1.0, 0.0);

            for &g in c.gates() {
                dense_gate(n, g, &mut v);
            }

            for &(b, p) in col.terms() {
                let angle = f64::from(p) * std::f64::consts::FRAC_PI_4;

                t[b as usize] = (angle.cos() * scale, angle.sin() * scale);
            }

            got.extend(v);
            want.extend(t);
        }

        let Some(k) = want.iter().position(|a| a.0.hypot(a.1) > 0.1) else {
            return false;
        };
        let ((gr, gi), (wr, wi)) = (got[k], want[k]);
        let norm = wr * wr + wi * wi;
        let (pr, pi) = ((gr * wr + gi * wi) / norm, (gi * wr - gr * wi) / norm);

        got.iter().zip(&want).all(|(g, w)| {
            let turned = (pr * w.0 - pi * w.1, pr * w.1 + pi * w.0);

            (g.0 - turned.0).abs() < 1e-9 && (g.1 - turned.1).abs() < 1e-9
        })
    }

    fn request(system: &str, prompt: String, think_tokens: u32) -> GenRequest {
        GenRequest {
            system: system.to_string(),
            prompt,
            max_tokens: 512,
            temperature: 0.0,
            seed: 1,
            think_tokens,
        }
    }

    fn solve_prompt(target: &Target, previous: &[Circuit]) -> String {
        render_circuit_solve_prompt(target, 64, previous)
    }

    /// The circuit `model` answers to a solve prompt, if its reply parses.
    fn answer(model: &SimDesigner, system: &str, prompt: String, think: u32) -> Option<Circuit> {
        let qubits = parse_circuit_prompt(&prompt)
            .expect("prompt")
            .target
            .qubits();
        let reply = model
            .generate(&request(system, prompt, think))
            .expect("generate");

        parse_circuit_answer(&reply.text, qubits, 64)
    }

    /// (exact, T, CNOT) of the answer to `target`. The exact checker and the dense check
    /// must agree.
    fn designed(
        model: &SimDesigner,
        system: &str,
        target: &Target,
        think: u32,
    ) -> Option<(bool, u32, u32)> {
        let c = answer(model, system, solve_prompt(target, &[]), think)?;
        let checked = check(target, &c).expect("check");

        assert_eq!(checked.exact, dense_agrees(target, &c), "{}", c.render());

        Some((checked.exact, checked.cost.t, checked.cost.cnot))
    }

    /// The large designer's verdict on candidates a and b.
    fn verdict(
        system: &str,
        target: &Target,
        (a, b): (&str, &str),
        think: u32,
        seed: u64,
    ) -> String {
        let prompt = render_circuit_judge_prompt(target, 64, &circuit(a), &circuit(b));
        let req = GenRequest {
            seed,
            ..request(system, prompt, think)
        };

        SimDesigner::large().generate(&req).expect("generate").text
    }

    #[test]
    fn knowledge_follows_the_order_of_first_mention() {
        let english = "Try every short circuit, cancel pairs, use XOR row operations and Hadamard basis changes.";
        let german = "Nutze einen Basiswechsel, dann ein Phasenpolynom über Paritäten, Zeilenoperationen; \
                      konjugiere; durchsuche von beiden Enden; kürze; prüfe die vorige Schaltung.";

        assert_eq!(
            knowledge(english),
            ["Search", "Simplify", "Linear", "Hadamard"]
        );
        assert_eq!(
            knowledge(german),
            [
                "Hadamard", "Phase", "Linear", "Compose", "Search", "Meet", "Simplify", "Previous"
            ]
        );
        assert_eq!(knowledge("Meet in the middle."), ["Meet"]);
        assert!(knowledge("Solve it.").is_empty());

        // The seed strategies name one family each; role, format and check name none.
        assert_eq!(knowledge(LINEAR), ["Linear"]);
        assert_eq!(knowledge(PHASE), ["Phase"]);
        assert_eq!(knowledge(HADAMARD), ["Hadamard"]);
        assert_eq!(knowledge(SEARCH), ["Search"]);

        for neutral in [ROLE, FORMAT, CIRCUIT_SELF_CHECK_TEXT] {
            assert!(knowledge(neutral).is_empty(), "{neutral}");
        }

        // The other test strategies name one family each too, and the strong prompt names
        // seven in its order.
        assert_eq!(knowledge(COMPOSE), ["Compose"]);
        assert_eq!(knowledge(PREVIOUS), ["Previous"]);
        assert_eq!(knowledge(MEET), ["Meet"]);
        assert_eq!(knowledge(SIMPLIFY), ["Simplify"]);
        assert_eq!(
            knowledge(&strong()),
            [
                "Hadamard", "Phase", "Linear", "Compose", "Simplify", "Search", "Meet"
            ]
        );
    }

    #[test]
    fn the_stems_are_lower_case_and_name_the_families_in_order() {
        let names: Vec<&str> = FAMILY_STEMS.iter().map(|&(name, _)| name).collect();
        let families: Vec<&str> = Family::ALL.iter().map(|f| f.name()).collect();

        assert_eq!(names, families);

        for (name, stems) in FAMILY_STEMS {
            assert!(!stems.is_empty(), "{name}");

            for stem in stems {
                assert_eq!(*stem, stem.to_lowercase(), "{name}");
                assert_eq!(knowledge(stem), [name], "{stem} names only {name}");
            }
        }
    }

    #[test]
    fn the_targets_match_published_circuits() {
        let nielsen_chuang = "H 2; CNOT 1 2; TDG 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; CNOT 0 2; \
                              T 1; T 2; H 2; CNOT 0 1; T 0; TDG 1; CNOT 0 1";
        let ccz_text = nielsen_chuang
            .replacen("H 2; ", "", 1)
            .replacen("H 2; ", "", 1);
        let peres_text = nielsen_chuang.trim_end_matches("; CNOT 0 1").to_string();
        let tail = nielsen_chuang
            .splitn(3, "; ")
            .nth(2)
            .expect("third gate on");
        let fredkin_text = format!("S 1; CNOT 2 1; SDG 1; H 2; S 2; H 2; S 2; {tail}; CNOT 2 1");

        assert_eq!(circuit(&fredkin_text).gates().len(), 21);

        let cases = [
            (cz(), "H 1; CNOT 0 1; H 1".to_string(), (0, 1)),
            (swap(), "CNOT 0 1; CNOT 1 0; CNOT 0 1".to_string(), (0, 3)),
            (
                cs(),
                "T 0; T 1; CNOT 0 1; TDG 1; CNOT 0 1".to_string(),
                (3, 2),
            ),
            (
                ch(),
                "S 1; H 1; T 1; CNOT 0 1; TDG 1; H 1; SDG 1".to_string(),
                (2, 1),
            ),
            (
                qft2(),
                "H 0; CNOT 0 1; CNOT 1 0; TDG 1; CNOT 0 1; T 1; T 0; H 0".to_string(),
                (3, 3),
            ),
            (ccz(), ccz_text, (7, 6)),
            (toffoli(), nielsen_chuang.to_string(), (7, 6)),
            (peres(), peres_text, (7, 5)),
            (fredkin(), fredkin_text, (7, 7)),
            (
                rtof(),
                "H 2; T 2; CNOT 1 2; TDG 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; H 2".to_string(),
                (4, 3),
            ),
            (
                ccix(),
                "H 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; H 2"
                    .to_string(),
                (4, 4),
            ),
        ];

        for (target, text, (t, cnot)) in cases {
            let c = circuit(&text);
            let checked = check(&target, &c).expect("check");

            assert!(checked.exact && dense_agrees(&target, &c), "{text}");
            assert_eq!((checked.cost.t, checked.cost.cnot), (t, cnot), "{text}");
        }
    }

    /// A route of the calibration: a label, the system prompt, the target, and the (T, CNOT)
    /// of the exact answer, or None for a miss.
    type Route = (&'static str, String, Target, Option<(u32, u32)>);

    #[test]
    fn each_route_reaches_its_cost_on_plain_and_dressed_targets() {
        let basis = checking(&[HADAMARD, PHASE, LINEAR]);
        let cases: Vec<Route> = vec![
            ("swap by Linear", LINEAR.to_string(), swap(), Some((0, 3))),
            ("cs by Phase", PHASE.to_string(), cs(), Some((3, 2))),
            ("cz by Phase alone", PHASE.to_string(), cz(), Some((0, 2))),
            (
                "cz by Hadamard and Linear",
                checking(&[HADAMARD, LINEAR]),
                cz(),
                Some((0, 1)),
            ),
            (
                "ccz by Phase alone",
                PHASE.to_string(),
                ccz(),
                Some((7, 10)),
            ),
            (
                "ccz by Phase and Linear",
                checking(&[PHASE, LINEAR]),
                ccz(),
                Some((7, 6)),
            ),
            ("qft2", basis.clone(), qft2(), Some((3, 3))),
            ("toffoli", basis.clone(), toffoli(), Some((7, 6))),
            ("peres", basis.clone(), peres(), Some((7, 5))),
            ("rtof", basis.clone(), rtof(), Some((4, 3))),
            ("ccix", basis.clone(), ccix(), Some((4, 4))),
            (
                "fredkin by Compose",
                checking(&[COMPOSE, HADAMARD, PHASE, LINEAR]),
                fredkin(),
                Some((7, 8)),
            ),
            (
                "fredkin by Compose and Simplify",
                checking(&[COMPOSE, HADAMARD, PHASE, LINEAR, SIMPLIFY]),
                fredkin(),
                Some((7, 7)),
            ),
            ("cz by Search", checking(&[SEARCH]), cz(), Some((0, 1))),
            (
                "ch by Search and Meet",
                checking(&[SEARCH, MEET]),
                ch(),
                Some((2, 1)),
            ),
            ("ch by Meet alone", checking(&[MEET]), ch(), Some((2, 1))),
            // Misses: a family short. Search without Meet stops long before the seven
            // gates of CH.
            ("fredkin without Compose", basis.clone(), fredkin(), None),
            (
                "toffoli without Hadamard",
                checking(&[PHASE, LINEAR]),
                toffoli(),
                None,
            ),
            ("swap by Phase", PHASE.to_string(), swap(), None),
            ("cz by Hadamard alone", HADAMARD.to_string(), cz(), None),
            ("ch by Search alone", checking(&[SEARCH]), ch(), None),
            (
                "ch",
                checking(&[HADAMARD, PHASE, LINEAR, COMPOSE, PREVIOUS, SIMPLIFY]),
                ch(),
                None,
            ),
        ];

        for (label, system, base, cost) in cases {
            for seed in 0..6 {
                let target = if seed == 0 {
                    base.clone()
                } else {
                    base.dress(&Dressing::draw(base.qubits(), &mut rng(seed)))
                };
                let want = cost.map(|(t, cnot)| (true, t, cnot));

                assert_eq!(
                    designed(&SimDesigner::large(), &system, &target, 4096),
                    want,
                    "{label}, dressing {seed}"
                );
            }
        }
    }

    #[test]
    fn hadamard_builds_z_gates_as_h_x_h() {
        // X on both inputs of CZ: through H on qubit 0, the X there turns into a Z, which
        // a Worker without Phase builds as H X H.
        let flipped = cz().dress(&Dressing {
            perm: vec![0, 1],
            x_in: 3,
            x_out: 0,
        });
        let system = format!("{HADAMARD} {LINEAR}");
        let c = answer(
            &SimDesigner::small(),
            &system,
            solve_prompt(&flipped, &[]),
            0,
        );

        assert_eq!(z_flips(&[0, 4, 0, 4], 2), Some(1));
        assert_eq!(z_flips(&[0, 4, 4, 4], 2), None);
        assert_eq!(
            designed(&SimDesigner::small(), &system, &flipped, 0),
            Some((true, 0, 1))
        );
        // The frame's H 0 and the first H of H X H stay uncancelled without Simplify; with
        // it, the pair cancels.
        assert_eq!(
            c.expect("answer").render(),
            "H 0; H 0; X 0; H 0; CNOT 1 0; X 0; X 1; H 0"
        );

        let simplifying = format!("{system} {SIMPLIFY}");
        let simplified = answer(
            &SimDesigner::small(),
            &simplifying,
            solve_prompt(&flipped, &[]),
            4096,
        );

        assert_eq!(
            simplified.expect("answer").render(),
            "X 0; H 0; CNOT 1 0; X 0; X 1; H 0"
        );
    }

    #[test]
    fn kernel_shifts_save_cnots_on_relative_phase_toffolis() {
        // Through H on qubit 2, both targets become an affine permutation with phases. The
        // Walsh polynomial alone puts phases on parities a short network never visits; a
        // kernel shift moves them away.
        for (target, plain, shifted) in [(rtof(), 5, 3), (ccix(), 6, 4)] {
            let m = core(
                &target,
                Frame {
                    cnot: None,
                    hadamard: 4,
                },
            )
            .expect("monomial core");
            let polynomial = phase_polynomial(&m.phases, 3).expect("polynomial");
            let representatives: Vec<Vec<u8>> =
                kernel(3).iter().map(|k| add(&polynomial, k)).collect();
            let cnots = |supports: Vec<u64>| {
                let goal = Goal {
                    rows: m.rows,
                    supports,
                };

                network(3, &goal, &mut Meter::new(4096, 2)).map(|moves| moves.len())
            };

            assert_eq!(cnots(vec![support(&polynomial)]), Some(plain));
            assert_eq!(cnots(minimal_supports(&representatives)), Some(shifted));
        }
    }

    /// The phase table of a coefficient vector: sum over z of a_z (z.x) mod 8.
    fn phases_of(coefficients: &[u8]) -> Vec<u8> {
        (0u32..)
            .take(coefficients.len())
            .map(|x| {
                let sum: u32 = (0u32..)
                    .zip(coefficients)
                    .filter(|&(z, _)| odd(z, x))
                    .map(|(_, &k)| u32::from(k))
                    .sum();

                u8::try_from(sum % 8).expect("phase")
            })
            .collect()
    }

    #[test]
    fn the_kernel_is_complete_up_to_three_qubits() {
        for n in 1..=3 {
            let size = 1usize << n;

            // Every coefficient vector in Z8 that adds no phase: none has an odd entry.
            let brute: BTreeSet<Vec<u8>> = (0..1u32 << (3 * (size - 1)))
                .map(|code| {
                    (0..size)
                        .map(|z| {
                            if z == 0 {
                                0
                            } else {
                                ((code >> (3 * (z - 1))) & 7) as u8
                            }
                        })
                        .collect::<Vec<u8>>()
                })
                .filter(|k| phases_of(k).iter().all(|&p| p == 0))
                .collect();
            let found: BTreeSet<Vec<u8>> = kernel(n).into_iter().collect();

            assert_eq!(found, brute, "n = {n}");
        }

        assert_eq!(
            [1, 2, 3].map(|n| kernel(n).len()),
            [1, 2, 32],
            "the kernel sizes counted by hand"
        );
        assert_eq!(kernel(4), [vec![0; 16]]);
    }

    #[test]
    fn the_walsh_transform_inverts_phase_polynomials() {
        let mut r = rng(5);

        for n in 1..=4 {
            let size = 1usize << n;

            for _ in 0..200 {
                let mut a: Vec<u8> = (0..size).map(|_| r.random_range(0..8)).collect();

                a[0] = 0;

                let phases = phases_of(&a);
                let found = phase_polynomial(&phases, n).expect("divides");

                assert_eq!(phases_of(&found), phases);

                if n <= 3 {
                    // The two differ by a kernel vector, so their odd terms are the same.
                    let difference: Vec<u8> =
                        found.iter().zip(&a).map(|(x, y)| (x + 8 - y) % 8).collect();

                    assert!(kernel(n).contains(&difference));
                }
            }
        }

        // On six qubits the lifted table need not divide: TDG on every qubit has a phase
        // polynomial, but the transform of its table 0..7 does not give it.
        let mut tdg = vec![0; 64];

        for q in 0..6 {
            tdg[1 << q] = 7;
        }

        let tdg_everywhere = phases_of(&tdg);

        assert!(
            tdg_everywhere
                .iter()
                .zip(0u32..)
                .all(|(&p, x)| u32::from(p) == 7 * x.count_ones() % 8)
        );
        assert_eq!(phase_polynomial(&tdg_everywhere, 6), None);
    }

    #[test]
    fn one_step_short_misses_and_enough_thinking_hits() {
        let (small, large) = (SimDesigner::small(), SimDesigner::large());
        let basis = format!("{HADAMARD} {PHASE} {LINEAR} {CIRCUIT_SELF_CHECK_TEXT}");
        let framing = format!("{HADAMARD} {PHASE} {LINEAR} {COMPOSE} {CIRCUIT_SELF_CHECK_TEXT}");

        // Phase alone on CCZ: the core test (8 steps) and the Walsh transform (8) fill the
        // 16 free steps, and verifying two columns takes 2 more, at 4 tokens each.
        // Meet on CH: the 6,293rd word meets. At 4 words per step that is 1,574 steps, and
        // with 4 columns verified 1,578, of which 16 are free: 3,124 tokens on sim:large.
        let cases = [
            (&small, PHASE.to_string(), ccz(), 8, (7, 10)),
            (&small, format!("{PHASE} {LINEAR}"), ccz(), 196, (7, 6)),
            (&large, basis.clone(), toffoli(), 158, (7, 6)),
            (&large, basis.clone(), peres(), 124, (7, 5)),
            (&large, basis.clone(), rtof(), 78, (4, 3)),
            (&large, basis.clone(), ccix(), 92, (4, 4)),
            (&large, framing, fredkin(), 654, (7, 8)),
            (&large, checking(&[SEARCH]), cz(), 104, (0, 1)),
            (&large, checking(&[SEARCH, MEET]), ch(), 3124, (2, 1)),
            (
                &large,
                checking(&[COMPOSE, HADAMARD, PHASE, LINEAR, SIMPLIFY]),
                fredkin(),
                1566,
                (7, 7),
            ),
        ];

        for (model, system, target, think, (t, cnot)) in cases {
            let hit = Some((true, t, cnot));

            assert_eq!(designed(model, &system, &target, think), hit, "{think}");
            assert_ne!(
                designed(model, &system, &target, think - 1),
                hit,
                "{think} - 1"
            );
        }

        // Two-qubit routes fit the free steps.
        assert_eq!(designed(&small, LINEAR, &swap(), 0), Some((true, 0, 3)));
        assert_eq!(designed(&small, PHASE, &cs(), 0), Some((true, 3, 2)));
    }

    #[test]
    fn self_check_resists_a_circuit_that_passes_spot_checks() {
        let fake = circuit("Z 0");
        let target = cz();
        let checking = format!("{PREVIOUS} {PHASE} {CIRCUIT_SELF_CHECK_TEXT}");
        let trusting = format!("{PREVIOUS} {PHASE}");
        let prompt = solve_prompt(&target, std::slice::from_ref(&fake));

        // Z 0 agrees with CZ on |00> and |11>, but not on |10>.
        assert!(agrees_on(&target, &fake, &[0, 3]).expect("spot check"));
        assert!(!check(&target, &fake).expect("check").exact);

        let model = SimDesigner::large();
        let fooled = answer(&model, &trusting, prompt.clone(), 4096).expect("answer");
        let right = answer(&model, &checking, prompt, 4096).expect("answer");

        assert_eq!(fooled, fake, "the cheap fake beats the real (0, 2) circuit");
        assert!(check(&target, &right).expect("check").exact);
        assert_eq!(right.cost().cnot, 2);
    }

    #[test]
    fn the_judge_is_fooled_only_without_self_check() {
        let (fake, real) = ("Z 0", "H 1; CNOT 0 1; H 1");
        let checking = CIRCUIT_SELF_CHECK_TEXT;

        for seed in 0..16 {
            assert_eq!(
                verdict("", &cz(), (fake, real), 0, seed),
                "VERDICT A",
                "fooled"
            );
            assert_eq!(
                verdict("", &cz(), (real, fake), 0, seed),
                "VERDICT B",
                "fooled"
            );
            assert_eq!(verdict(checking, &cz(), (fake, real), 0, seed), "VERDICT B");
            assert_eq!(verdict(checking, &cz(), (real, fake), 0, seed), "VERDICT A");
        }

        // Two exact circuits: fewer T gates win, then fewer CNOTs.
        let phase_network = "S 0; S 1; CNOT 0 1; SDG 1; CNOT 0 1";

        assert_eq!(
            verdict(checking, &cz(), (phase_network, real), 0, 3),
            "VERDICT B"
        );
        assert_eq!(
            verdict(
                checking,
                &cs(),
                (
                    "T 0; T 1; CNOT 0 1; TDG 1; CNOT 0 1",
                    "T 0; T 1; CNOT 0 1; TDG 1; CNOT 0 1; CNOT 0 1; CNOT 0 1"
                ),
                0,
                3
            ),
            "VERDICT A"
        );
    }

    #[test]
    fn a_judge_that_cannot_verify_trusts_stated_costs_then_a_coin() {
        // CZ on the first two of four qubits: checking one candidate on all 16 columns
        // takes the 16 free steps, so the second needs 16 more (32 tokens on sim:large).
        let wide = monomial(4, |j| (j, if j >> 2 == 3 { 4 } else { 0 }));
        let (fake, real) = ("Z 0", "H 1; CNOT 0 1; H 1");
        let checking = CIRCUIT_SELF_CHECK_TEXT;

        assert_eq!(verdict(checking, &wide, (fake, real), 0, 1), "VERDICT A");
        assert_eq!(verdict(checking, &wide, (fake, real), 31, 1), "VERDICT A");
        assert_eq!(verdict(checking, &wide, (fake, real), 32, 1), "VERDICT B");

        // Neither candidate passes and their costs tie: a seeded coin decides.
        let flips: BTreeSet<String> = (0..32)
            .map(|seed| verdict(checking, &cz(), ("X 0", "X 1"), 0, seed))
            .collect();

        assert_eq!(
            flips.into_iter().collect::<Vec<_>>(),
            ["VERDICT A", "VERDICT B"]
        );
    }

    #[test]
    fn previous_answers_are_kept_or_dropped() {
        let model = SimDesigner::large();
        let real = circuit("H 1; CNOT 0 1; H 1");
        let junk = circuit("X 0");
        let verifying = format!("{PREVIOUS} {PHASE} {CIRCUIT_SELF_CHECK_TEXT}");
        let solve = |system: &str, previous: &[Circuit]| {
            answer(&model, system, solve_prompt(&cz(), previous), 4096).expect("answer")
        };

        // A verified earlier answer is kept when it is cheaper than the Worker's own; an
        // older one too. A Worker that does not verify overrides it.
        assert_eq!(solve(&verifying, std::slice::from_ref(&real)), real);
        assert_eq!(solve(&verifying, &[real.clone(), junk.clone()]), real);
        assert_eq!(solve(PHASE, std::slice::from_ref(&real)).cost().cnot, 2);

        // A costlier earlier answer is dropped for the Worker's own.
        let both = format!("{PREVIOUS} {PHASE} {LINEAR} {CIRCUIT_SELF_CHECK_TEXT}");
        let naive = answer(&model, PHASE, solve_prompt(&ccz(), &[]), 4096).expect("naive");
        let better = answer(
            &model,
            &both,
            solve_prompt(&ccz(), std::slice::from_ref(&naive)),
            4096,
        );

        assert_eq!(naive.cost().cnot, 10);
        assert_eq!(better.map(|c| c.cost().cnot), Some(6));

        // With nothing of its own (Hadamard alone builds nothing for CZ), a Worker passes
        // the latest earlier answer on, unchecked, as long as it fits the gate cap.
        assert_eq!(solve(HADAMARD, &[real.clone(), junk.clone()]), junk);

        let capped = render_circuit_solve_prompt(&cz(), 2, &[junk.clone(), real.clone()]);

        assert_eq!(answer(&model, HADAMARD, capped, 4096), Some(junk));
    }

    #[test]
    fn the_gate_cap_is_respected() {
        let model = SimDesigner::large();
        let both = format!("{PHASE} {LINEAR}");
        let reply = |system: &str, cap: usize| {
            let req = request(system, render_circuit_solve_prompt(&ccz(), cap, &[]), 4096);

            model.generate(&req).expect("generate").text
        };

        // The naive CCZ circuit has 17 gates, the network one 13.
        assert_eq!(reply(PHASE, 16), NO_CIRCUIT);
        assert_eq!(
            parse_circuit_answer(&reply(PHASE, 17), 3, 17).map(|c| c.gates().len()),
            Some(17)
        );
        assert_eq!(reply(&both, 12), NO_CIRCUIT);
        assert_eq!(
            parse_circuit_answer(&reply(&both, 13), 3, 13).map(|c| c.gates().len()),
            Some(13)
        );
        assert_eq!(reply(PHASE, 0), NO_CIRCUIT);
    }

    #[test]
    fn temperature_shuffles_the_method_order() {
        // Phase first gives (0, 2) and leaves too little of the 16 free steps for Hadamard;
        // Hadamard first gives (0, 1).
        let system = "Use parities, then hadamard gates, then xor.";
        let prompt = solve_prompt(&cz(), &[]);
        let answers = |temperature: f32| -> BTreeSet<Option<(u32, u32)>> {
            (0..64)
                .map(|seed| {
                    let req = GenRequest {
                        temperature,
                        seed,
                        ..request(system, prompt.clone(), 0)
                    };
                    let text = SimDesigner::small().generate(&req).expect("generate").text;

                    parse_circuit_answer(&text, 2, 64).map(|c| (c.cost().t, c.cost().cnot))
                })
                .collect()
        };

        assert_eq!(answers(0.0).into_iter().collect::<Vec<_>>(), [Some((0, 2))]);
        assert_eq!(answers(f32::NAN).len(), 1, "NaN counts as cold");
        assert_eq!(
            answers(TEMPERATURE_CEILING).into_iter().collect::<Vec<_>>(),
            [Some((0, 1)), Some((0, 2))],
            "hot: sometimes Hadamard first"
        );
    }

    #[test]
    fn responses_depend_on_the_request_only() {
        let system = format!("{HADAMARD} {PHASE} {LINEAR} {COMPOSE}");
        let solve = GenRequest {
            temperature: 1.0,
            ..request(&system, solve_prompt(&toffoli(), &[]), 512)
        };
        let judge = request(
            &system,
            render_circuit_judge_prompt(&cz(), 64, &circuit("Z 0"), &circuit("X 1")),
            512,
        );
        let searching = GenRequest {
            temperature: 1.0,
            ..request(&strong(), solve_prompt(&ch(), &[]), 4096)
        };
        let simplifying = request(
            &strong(),
            solve_prompt(
                &fredkin(),
                &[circuit("CNOT 1 2; H 1; CNOT 2 1; H 1; CNOT 1 2")],
            ),
            4096,
        );

        for model in [SimDesigner::small(), SimDesigner::large()] {
            for req in [&solve, &judge, &searching, &simplifying] {
                assert_eq!(
                    model.generate(req).expect("first"),
                    model.generate(req).expect("again")
                );
            }
        }
    }

    #[test]
    fn token_counts_stay_within_the_budgets() {
        let systems = [
            PHASE.to_string(),
            format!("{HADAMARD} {PHASE} {LINEAR} {COMPOSE} {PREVIOUS} {CIRCUIT_SELF_CHECK_TEXT}"),
            strong(),
        ];

        for model in [SimDesigner::small(), SimDesigner::large()] {
            for target in [cz(), swap(), ch(), toffoli(), fredkin()] {
                for think in [0, 5, 64, 300, 4096] {
                    for system in &systems {
                        let req = request(system, solve_prompt(&target, &[]), think);
                        let reply = model.generate(&req).expect("generate");
                        let text = approx_tokens(&reply.text);
                        let prompt = approx_tokens(&req.system) + approx_tokens(&req.prompt);

                        assert!((text..=text + think).contains(&reply.completion_tokens));
                        assert_eq!(reply.prompt_tokens, prompt);
                        assert_eq!(reply.stop_reason, StopReason::Stop);
                        assert_eq!(reply.backend, model.tag());
                    }
                }
            }
        }

        // Phase alone on CCZ uses 18 steps, 2 beyond the free ones.
        let req = request(PHASE, solve_prompt(&ccz(), &[]), 4096);
        let thinking = |m: SimDesigner| {
            let reply = m.generate(&req).expect("generate");

            reply.completion_tokens - approx_tokens(&reply.text)
        };

        assert_eq!(thinking(SimDesigner::small()), 2 * 4);
        assert_eq!(thinking(SimDesigner::large()), 2 * 2);
    }

    #[test]
    fn a_tiny_answer_budget_cuts_gates_and_reports_length() {
        let reply = |prompt: String, max_tokens: u32| {
            let req = GenRequest {
                max_tokens,
                ..request(PHASE, prompt, 0)
            };
            let r = SimDesigner::small().generate(&req).expect("generate");

            assert!(
                r.completion_tokens <= max_tokens,
                "no thinking, so all of it is text"
            );
            (r.text, r.stop_reason)
        };
        let solve = |max_tokens| reply(solve_prompt(&cs(), &[]), max_tokens);
        let full = circuit("T 1; T 0; CNOT 0 1; TDG 1; CNOT 0 1");
        let four = circuit("T 1; T 0; CNOT 0 1; TDG 1");
        let judge = render_circuit_judge_prompt(&cs(), 64, &full, &four);

        // ANSWER, two words per one-qubit gate and three per CNOT: 13 tokens.
        assert_eq!(solve(13), (render_circuit_answer(&full), StopReason::Stop));
        assert_eq!(
            solve(12),
            (render_circuit_answer(&four), StopReason::Length)
        );
        assert_eq!(solve(1), ("ANSWER".to_string(), StopReason::Length));
        assert_eq!(solve(0), (String::new(), StopReason::Length));
        assert_eq!(reply(judge, 1), ("VERDICT".to_string(), StopReason::Length));
    }

    #[test]
    fn an_unreadable_prompt_gets_a_plain_reply() {
        let grid = "TASK\nEXAMPLE 1\nINPUT\n1 2\nOUTPUT\n2 1\nTEST INPUT\n3 4\nEND";
        let full = solve_prompt(&cz(), &[]);
        let cut = full.trim_end_matches("END").to_string();
        let not_unitary = full.replace("|11> -> -|11>", "|11> -> -|10>");

        for prompt in ["", "Hello", "ANSWER\nH 0", grid, &cut, &not_unitary] {
            let req = request(PHASE, prompt.to_string(), 4096);
            let reply = SimDesigner::large().generate(&req).expect("generate");

            assert_eq!(
                (reply.text.as_str(), reply.stop_reason),
                (CANNOT_READ, StopReason::Stop),
                "{prompt:?}"
            );
            assert_eq!(
                reply.completion_tokens,
                approx_tokens(CANNOT_READ),
                "no thinking"
            );
        }
    }

    #[test]
    fn model_tags_name_the_two_designers() {
        for tag in ["sim:small", "sim:large"] {
            let model = SimDesigner::from_tag(tag).expect("designer");

            assert_eq!((model.tag(), model.name()), (tag, tag));
        }

        assert_eq!(SimDesigner::small().step_tokens, 4);
        assert_eq!(SimDesigner::large().step_tokens, 2);
        assert!(SimDesigner::from_tag("sim:medium").is_none());
    }

    #[test]
    fn garbage_never_panics() {
        let solve = solve_prompt(&ccix(), &[circuit("H 2; CNOT 0 2")]);
        let judge = render_circuit_judge_prompt(&qft2(), 64, &circuit("H 0"), &circuit("T 1"));
        let six = render_circuit_solve_prompt(
            &monomial(6, |j| {
                (j ^ 21, u8::try_from(j.count_ones() % 8).expect("phase"))
            }),
            256,
            &[],
        );
        let five = render_circuit_solve_prompt(
            &monomial(5, |j| (j.rotate_left(1) & 31 | j >> 4, 4 * (j & 1) as u8)),
            64,
            &[],
        );
        let mut texts: Vec<String> = [
            "",
            "\n",
            "CIRCUIT TASK",
            "CIRCUIT JUDGE\nEND",
            "ANSWER\nH 0",
        ]
        .map(String::from)
        .to_vec();

        texts.extend([solve.clone(), judge.clone(), six, five]);

        for text in [&solve, &judge] {
            texts.extend(text.char_indices().map(|(i, _)| text[..i].to_string()));
        }

        let mut r = rng(17);
        let edits = ['0', '1', '|', '+', '-', ' ', '\n', 'x', 'w', 'i', '9'];

        for _ in 0..300 {
            let base = if r.random_bool(0.5) { &solve } else { &judge };
            let mut chars: Vec<char> = base.chars().collect();
            let at = r.random_range(0..chars.len());

            chars[at] = edits[r.random_range(0..edits.len())];
            texts.push(chars.into_iter().collect());
        }

        let systems = [
            String::new(),
            PHASE.to_string(),
            checking(&[
                HADAMARD, PHASE, LINEAR, COMPOSE, PREVIOUS, SIMPLIFY, SEARCH, MEET,
            ]),
        ];
        let temperatures = [0.0, 1.5, f32::NAN, f32::INFINITY, -1.0];

        for (i, text) in texts.iter().enumerate() {
            let req = GenRequest {
                system: systems[i % systems.len()].clone(),
                prompt: text.clone(),
                max_tokens: [0, 3, 512][i % 3],
                temperature: temperatures[i % temperatures.len()],
                seed: i as u64,
                think_tokens: [0, 4096][i % 2],
            };

            for model in [SimDesigner::small(), SimDesigner::large()] {
                let reply = model.generate(&req).expect("generate");

                assert!(reply.completion_tokens >= approx_tokens(&reply.text));
                assert!(approx_tokens(&reply.text) <= req.max_tokens);
            }
        }
    }

    /// The dense columns of a circuit's unitary on n qubits, in f64.
    fn dense_unitary(n: usize, c: &Circuit) -> Vec<Amp> {
        let mut out = Vec::new();

        for j in 0..1usize << n {
            let mut v = vec![(0.0, 0.0); 1 << n];

            v[j] = (1.0, 0.0);

            for &g in c.gates() {
                dense_gate(n, g, &mut v);
            }

            out.extend(v);
        }

        out
    }

    /// True when two circuits on n qubits have one unitary up to a global phase: exactly by
    /// their canonical keys, and within 1e-9 by dense simulation. The two must agree.
    fn same(n: usize, a: &Circuit, b: &Circuit) -> bool {
        let exact = canonical_key(&columns(n, a).expect("columns"))
            == canonical_key(&columns(n, b).expect("columns"));
        let (x, y) = (dense_unitary(n, a), dense_unitary(n, b));
        let k = y
            .iter()
            .position(|v| v.0.hypot(v.1) > 0.1)
            .expect("a nonzero entry");
        let ((xr, xi), (yr, yi)) = (x[k], y[k]);
        let norm = yr * yr + yi * yi;
        let (pr, pi) = ((xr * yr + xi * yi) / norm, (xi * yr - xr * yi) / norm);
        let dense = x.iter().zip(&y).all(|(u, v)| {
            let turned = (pr * v.0 - pi * v.1, pr * v.1 + pi * v.0);

            (u.0 - turned.0).abs() < 1e-9 && (u.1 - turned.1).abs() < 1e-9
        });

        assert_eq!(exact, dense, "{} versus {}", a.render(), b.render());

        exact
    }

    /// A random circuit of `len` gates drawn from `moves`.
    fn random_circuit(r: &mut ChaCha8Rng, moves: &[Gate], len: usize) -> Circuit {
        let gates = (0..len)
            .map(|_| moves[r.random_range(0..moves.len())])
            .collect();

        Circuit::new(gates).expect("valid")
    }

    /// A meter that never runs out in these tests.
    fn plenty() -> Meter {
        Meter::new(u32::MAX, 1)
    }

    #[test]
    fn a_search_without_self_check_takes_a_spot_check_fake() {
        // Z 0 agrees with CZ on |00> and |11>, and it is the first word that does.
        let model = SimDesigner::large();
        let fooled = answer(&model, SEARCH, solve_prompt(&cz(), &[]), 0).expect("answer");
        let right =
            answer(&model, &checking(&[SEARCH]), solve_prompt(&cz(), &[]), 4096).expect("answer");

        assert_eq!(fooled.render(), "Z 0");
        assert!(!check(&cz(), &fooled).expect("check").exact);
        assert!(check(&cz(), &right).expect("check").exact);
        assert_eq!((right.cost().t, right.cost().cnot), (0, 1));
    }

    #[test]
    fn a_search_without_self_check_matches_on_the_checked_columns() {
        // |a,b> -> |1 xor b, a>: SWAP, then X on qubit 0. X 1 agrees with it on |01> and
        // |10> only, X 0 on |00> and |11>, the columns a spot check looks at.
        let target = monomial(2, |j| ((((j & 1) ^ 1) << 1) | (j >> 1), 0));

        assert!(agrees_on(&target, &circuit("X 0"), &[0, 3]).expect("spot check"));
        assert!(!agrees_on(&target, &circuit("X 1"), &[0, 3]).expect("spot check"));

        for meet in [false, true] {
            let found = word_search(&target, &[0, 3], meet, 64, &mut plenty()).expect("a word");

            assert!(
                agrees_on(&target, &found, &[0, 3]).expect("spot check"),
                "{}",
                found.render()
            );
        }

        // So a Worker without self-check answers with a spot-check fake, not with nothing.
        let model = SimDesigner::large();

        for system in [SEARCH, MEET] {
            let fooled = answer(&model, system, solve_prompt(&target, &[]), 4096).expect("answer");

            assert!(agrees_on(&target, &fooled, &[0, 3]).expect("spot check"));
            assert!(!check(&target, &fooled).expect("check").exact);
        }
    }

    #[test]
    fn x_flips_cost_a_word_search_nothing() {
        // X on the inputs and outputs is free, so every X dressing of CZ (qubits kept in
        // place) is found after as many words as CZ itself: one step short misses.
        let model = SimDesigner::large();
        let system = checking(&[SEARCH]);

        for x_in in 0..4 {
            for x_out in 0..4 {
                let target = cz().dress(&Dressing {
                    perm: vec![0, 1],
                    x_in,
                    x_out,
                });

                assert_eq!(
                    designed(&model, &system, &target, 104),
                    Some((true, 0, 1)),
                    "{x_in} {x_out}"
                );
                assert_eq!(designed(&model, &system, &target, 102), None);
            }
        }

        // Meet gets the X gates at both ends for free too: X on qubit 1 before CH, X on
        // qubit 0 after it.
        let flipped = ch().dress(&Dressing {
            perm: vec![0, 1],
            x_in: 1,
            x_out: 2,
        });

        assert_eq!(
            designed(&model, &checking(&[MEET]), &flipped, 3124),
            Some((true, 2, 1))
        );
    }

    #[test]
    fn word_search_works_on_up_to_three_qubits() {
        // CZ on the first two of three qubits is found; on the first two of four, Search
        // does not try.
        let model = SimDesigner::large();
        let system = checking(&[SEARCH]);
        let three = monomial(3, |j| (j, if j >> 1 == 3 { 4 } else { 0 }));
        let four = monomial(4, |j| (j, if j >> 2 == 3 { 4 } else { 0 }));

        assert_eq!(designed(&model, &system, &three, 4096), Some((true, 0, 1)));
        assert_eq!(designed(&model, &system, &four, 4096), None);
        assert_eq!(word_search(&four, &[0, 15], true, 64, &mut plenty()), None);
    }

    #[test]
    fn peephole_rules_move_gates_only_past_gates_they_commute_with() {
        let cases = [
            ("H 0; H 0", ""),
            ("T 0; T 0", "S 0"),
            ("S 0; Z 0", "SDG 0"),
            ("T 0; S 0", "T 0; S 0"),
            // A phase gate passes the control of a CNOT, an X its target.
            ("T 0; CNOT 0 1; T 0", "CNOT 0 1; S 0"),
            ("X 1; CNOT 0 1; X 1", "CNOT 0 1"),
            ("T 1; CNOT 0 1; TDG 1", "T 1; CNOT 0 1; TDG 1"),
            ("X 0; CNOT 0 1; X 0", "X 0; CNOT 0 1; X 0"),
            // Two CNOTs commute when they share only a control or only a target.
            ("CNOT 0 1; CNOT 0 2; CNOT 0 1", "CNOT 0 2"),
            ("CNOT 0 2; CNOT 1 2; CNOT 0 2", "CNOT 1 2"),
            (
                "CNOT 0 1; CNOT 1 2; CNOT 0 1",
                "CNOT 0 1; CNOT 1 2; CNOT 0 1",
            ),
            ("H 0; T 0; H 0", "H 0; T 0; H 0"),
            ("H 0; X 1; H 0; S 1; SDG 1", "X 1"),
        ];

        for (before, after) in cases {
            let c = circuit(before);
            let simplified = peephole(c.gates().to_vec(), &mut plenty()).expect("budget");
            let simplified = Circuit::new(simplified).expect("valid");

            assert_eq!(simplified.render(), after, "{before}");
            assert!(same(3, &c, &simplified), "{before}");
        }

        // A pass costs one step per gate: 2 for H 0; H 0, and nothing for the pass over the
        // empty result, which finds no more rewrites.
        let mut meter = Meter::new(0, 1);

        assert!(peephole(circuit("H 0; H 0").gates().to_vec(), &mut meter).is_some());
        assert_eq!(meter.used, 2);
    }

    #[test]
    fn phase_folding_adds_up_the_phases_on_each_parity() {
        let fold_text = |text: &str, last: bool| {
            let gates = fold(circuit(text).gates(), 3, last).expect("fold");

            Circuit::new(gates).expect("valid").render()
        };

        // T on qubit 1 before and after a CNOT pair: one parity, so one S.
        assert_eq!(
            fold_text("T 1; CNOT 0 1; CNOT 0 1; T 1", false),
            "S 1; CNOT 0 1; CNOT 0 1"
        );
        assert_eq!(
            fold_text("T 1; CNOT 0 1; CNOT 0 1; T 1", true),
            "CNOT 0 1; CNOT 0 1; S 1"
        );
        // T on x0 and on x0 xor 1 cancel up to a global phase.
        assert_eq!(fold_text("T 0; X 0; T 0", false), "X 0");
        // An H starts a new path variable: nothing merges across it.
        assert_eq!(fold_text("T 0; H 0; T 0", false), "T 0; H 0; T 0");
        // The parity x0 xor x1 is held by qubit 1 between the first two CNOTs, and later by
        // qubit 0.
        let moved = "CNOT 0 1; T 1; CNOT 0 1; CNOT 1 0; T 0";

        assert_eq!(fold_text(moved, false), "CNOT 0 1; S 1; CNOT 0 1; CNOT 1 0");
        assert_eq!(fold_text(moved, true), "CNOT 0 1; CNOT 0 1; CNOT 1 0; S 0");
    }

    #[test]
    fn every_rewrite_is_exact_on_random_circuits() {
        let mut r = rng(23);
        let mut table = Cliffords::default();
        let clifford: Vec<Gate> = word_moves(2).into_iter().filter(|g| !g.is_t()).collect();

        for round in 0..300 {
            let n = 1 + round % 3;
            let moves = if round % 2 == 0 && n == 2 {
                clifford.clone()
            } else {
                word_moves(n)
            };
            let len = r.random_range(0..24);
            let c = random_circuit(&mut r, &moves, len);
            let cost = c.cost();
            let rewrites = [
                peephole(c.gates().to_vec(), &mut plenty()),
                fold(c.gates(), n, false),
                fold(c.gates(), n, true),
                Some(rewindow(c.gates().to_vec(), &mut table, &mut plenty())),
            ];

            for (i, rewrite) in rewrites.into_iter().enumerate() {
                let out = Circuit::new(rewrite.expect("rewrite")).expect("valid");
                let after = out.cost();

                assert!(same(n, &c, &out), "rewrite {i} of {}", c.render());
                assert!(after.t <= cost.t && after.cnot <= cost.cnot, "rewrite {i}");
                assert!(after.gates <= cost.gates || i == 3, "rewrite {i}");
            }
        }
    }

    #[test]
    fn the_clifford_table_holds_every_two_qubit_clifford() {
        // The two-qubit Clifford group has 11,520 elements up to a global phase; up to a
        // Pauli gate, 720. By fewest CNOTs: 36 local ones, 324 with one CNOT, 324 with two,
        // and 36 that need three (SWAP times local gates).
        let mut table = Cliffords::default();

        assert!(table.grow_to(3, &mut plenty()));

        let by_cnots: Vec<usize> = (0..4)
            .map(|c| table.words.iter().filter(|w| w.1 == c).count())
            .collect();

        assert_eq!(by_cnots, [36, 324, 324, 36]);
        assert_eq!(table.variants.len(), 11_520);

        // Up to one CNOT there are 360 classes: 90 steps at four classes per step, more than
        // the free ones.
        let mut meter = plenty();

        assert!(Cliffords::default().grow_to(1, &mut meter));
        assert_eq!(meter.used, 90);
        assert!(!Cliffords::default().grow_to(1, &mut Meter::new(0, 1)));
    }

    #[test]
    fn a_cnot_h_cnot_window_needs_one_cnot() {
        let mut table = Cliffords::default();
        let cnots = [
            Gate::Cnot {
                control: 0,
                target: 1,
            },
            Gate::Cnot {
                control: 1,
                target: 0,
            },
        ];

        for first in cnots {
            for h in [Gate::H(0), Gate::H(1)] {
                for last in cnots {
                    let window = [first, h, last];
                    let word = table.rebuild(&window, 2, &mut plenty()).expect("one CNOT");
                    let rebuilt = Circuit::new(word).expect("valid");

                    assert_eq!(rebuilt.cost().cnot, 1);
                    assert!(same(
                        2,
                        &Circuit::new(window.to_vec()).expect("valid"),
                        &rebuilt
                    ));
                }
            }
        }

        // The block on which the seven-CNOT Fredkin circuit (Niroula and Nam 2021) saves a
        // CNOT, against a one-CNOT form of it from the calibration notes.
        let block = circuit("CNOT 1 0; H 1; CNOT 0 1");

        assert!(same(
            2,
            &block,
            &circuit("S 0; CNOT 1 0; SDG 0; SDG 1; H 1")
        ));

        // S on the parity of both qubits is CZ with S on each: one CNOT.
        let parity = circuit("CNOT 0 1; S 1; CNOT 0 1");
        let rebuilt = table
            .rebuild(parity.gates(), 2, &mut plenty())
            .expect("one CNOT");

        assert!(same(2, &parity, &Circuit::new(rebuilt).expect("valid")));

        // Two CNOTs in turn (locally equal to iSWAP) need both, and SWAP needs its three.
        assert_eq!(
            table.rebuild(circuit("CNOT 0 1; CNOT 1 0").gates(), 2, &mut plenty()),
            None
        );
        assert_eq!(
            table.rebuild(
                circuit("CNOT 0 1; CNOT 1 0; CNOT 0 1").gates(),
                3,
                &mut plenty()
            ),
            None
        );
    }

    #[test]
    fn a_window_gathers_the_gates_that_commute_with_those_it_leaves_out() {
        let window = |text: &str, start: usize| {
            let gates = circuit(text).gates().to_vec();

            window_at(&gates, start)
        };

        // T 0 is on another qubit; T 2 on the target of the second CNOT stays out of the
        // window and blocks the gates behind it on qubit 2.
        assert_eq!(
            window("CNOT 1 2; H 1; T 0; CNOT 2 1; T 2; CNOT 1 2", 0),
            Some(((1, 2), vec![0, 1, 3]))
        );
        // CNOT 0 1 is left out; the last CNOT 0 2 shares its control, so it joins. Behind
        // CNOT 1 0, whose target is that control, it cannot.
        assert_eq!(
            window("CNOT 0 2; H 0; CNOT 0 1; CNOT 0 2", 0),
            Some(((0, 2), vec![0, 1, 3]))
        );
        assert_eq!(
            window("CNOT 0 2; H 0; CNOT 1 0; CNOT 0 2", 0),
            Some(((0, 2), vec![0, 1]))
        );
        assert_eq!(window("H 0; CNOT 0 1", 0), None);
    }

    #[test]
    fn simplify_improves_an_earlier_answer() {
        // A second stage that knows only Simplify takes the first stage's (7, 8) Fredkin
        // down to (7, 7); one that only checks earlier answers keeps it.
        let model = SimDesigner::large();
        let first = answer(
            &model,
            &checking(&[COMPOSE, HADAMARD, PHASE, LINEAR]),
            solve_prompt(&fredkin(), &[]),
            4096,
        )
        .expect("first stage");
        let earlier = std::slice::from_ref(&first);
        let simplified = answer(
            &model,
            &checking(&[SIMPLIFY]),
            solve_prompt(&fredkin(), earlier),
            4096,
        )
        .expect("second stage");
        let kept = answer(
            &model,
            &checking(&[PREVIOUS]),
            solve_prompt(&fredkin(), earlier),
            4096,
        );

        assert_eq!((first.cost().t, first.cost().cnot), (7, 8));
        assert!(check(&fredkin(), &simplified).expect("check").exact);
        assert!(dense_agrees(&fredkin(), &simplified));
        assert_eq!((simplified.cost().t, simplified.cost().cnot), (7, 7));
        assert_eq!(kept, Some(first));
    }

    #[test]
    fn the_strong_prompt_reaches_every_target_cost() {
        let cases = [
            (cz(), (0, 1)),
            (swap(), (0, 3)),
            (cs(), (3, 2)),
            (ch(), (2, 1)),
            (qft2(), (3, 3)),
            (ccz(), (7, 6)),
            (toffoli(), (7, 6)),
            (peres(), (7, 5)),
            (fredkin(), (7, 7)),
            (rtof(), (4, 3)),
            (ccix(), (4, 4)),
        ];

        for (base, (t, cnot)) in cases {
            for seed in 0..4 {
                let target = if seed == 0 {
                    base.clone()
                } else {
                    base.dress(&Dressing::draw(base.qubits(), &mut rng(seed)))
                };

                assert_eq!(
                    designed(&SimDesigner::large(), &strong(), &target, 4096),
                    Some((true, t, cnot)),
                    "{} qubits, dressing {seed}",
                    base.qubits()
                );
            }
        }
    }

    #[test]
    fn the_column_cap_bounds_a_search_whatever_the_budget() {
        // On three qubits the cap is 8,192 words, two per step: 4,096 steps, 16 of them free,
        // however many thinking tokens the request allows.
        let req = request(
            &checking(&[SEARCH, MEET]),
            solve_prompt(&ccz(), &[]),
            u32::MAX,
        );
        let reply = SimDesigner::large().generate(&req).expect("generate");

        assert_eq!(reply.text, NO_CIRCUIT);
        assert_eq!(
            reply.completion_tokens - approx_tokens(&reply.text),
            (4096 - 16) * 2
        );
    }

    #[test]
    fn a_call_never_uses_more_than_max_steps() {
        // Within the Traits' ceiling of 4096 thinking tokens the budget is untouched.
        assert_eq!(MAX_STEPS, 1 << 20);
        assert_eq!(Meter::new(4096, 4).budget, FREE_STEPS + 1024);
        assert_eq!(Meter::new(4096, 2).budget, FREE_STEPS + 2048);

        // An absurd budget is cut to the ceiling.
        for step_tokens in [1, 2, 4] {
            assert_eq!(Meter::new(u32::MAX, step_tokens).budget, MAX_STEPS);
        }

        // A direct request whose work would take more: 20,000 earlier answers on six qubits,
        // each verified on all 64 columns at one step per column. The ceiling pays for
        // 16,384 of them, and the reply passes the latest one on unchecked.
        let target = monomial(6, |j| (((j << 1) | (j >> 5)) & 63, 0));
        let wrong = vec![circuit("X 0"); 20_000];
        let req = request(
            &format!("{PREVIOUS} {CIRCUIT_SELF_CHECK_TEXT}"),
            render_circuit_solve_prompt(&target, 256, &wrong),
            u32::MAX,
        );
        let reply = SimDesigner::large().generate(&req).expect("generate");
        let thinking = reply.completion_tokens - approx_tokens(&reply.text);

        assert_eq!(reply.text, render_circuit_answer(&circuit("X 0")));
        assert_eq!(u64::from(thinking), (MAX_STEPS - FREE_STEPS) * 2);
    }

    #[test]
    fn the_seed_strategies_reach_only_swap_and_cs() {
        // The four seed Workers: sim:small, 256 thinking tokens, temperature 0.2, no
        // self-check, role, one strategy and format. Of the eleven targets, plain and
        // dressed, only SWAP (Linear) and CS (Phase) are reached at their costs.
        let targets = [
            ("cz", cz(), (0, 1)),
            ("swap", swap(), (0, 3)),
            ("cs", cs(), (3, 2)),
            ("ch", ch(), (2, 1)),
            ("qft2", qft2(), (3, 3)),
            ("ccz", ccz(), (7, 6)),
            ("toffoli", toffoli(), (7, 6)),
            ("peres", peres(), (7, 5)),
            ("fredkin", fredkin(), (7, 7)),
            ("rtof", rtof(), (4, 3)),
            ("ccix", ccix(), (4, 4)),
        ];
        let mut reached = BTreeSet::new();

        for (strategy, label) in [
            (LINEAR, "linear"),
            (PHASE, "phase"),
            (HADAMARD, "hadamard"),
            (SEARCH, "search"),
        ] {
            let system = format!("{ROLE} {strategy} {FORMAT}");

            for (name, base, (t, cnot)) in &targets {
                for seed in 0..4 {
                    let target = if seed == 0 {
                        base.clone()
                    } else {
                        base.dress(&Dressing::draw(base.qubits(), &mut rng(seed)))
                    };
                    let req = GenRequest {
                        temperature: 0.2,
                        seed: 100 + seed,
                        ..request(&system, solve_prompt(&target, &[]), 256)
                    };
                    let text = SimDesigner::small().generate(&req).expect("generate").text;
                    let Some(c) = parse_circuit_answer(&text, target.qubits(), 64) else {
                        continue;
                    };
                    let checked = check(&target, &c).expect("check");

                    if checked.exact && checked.cost.t <= *t && checked.cost.cnot <= *cnot {
                        reached.insert((label, *name));
                    }
                }
            }
        }

        assert_eq!(
            reached.into_iter().collect::<Vec<_>>(),
            [("linear", "swap"), ("phase", "cs")]
        );
    }
}
