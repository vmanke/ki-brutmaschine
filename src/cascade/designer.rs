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
//!    steps plus one step per `step_tokens` thinking tokens. Once a payment fails, thinking
//!    stops for the rest of the call. The steps used beyond the free ones are reported as
//!    thinking tokens.
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
//! - Previous: verify the answers of earlier stages and keep the cheapest that passes.
//! - Search, Meet and Simplify are recognised, but contribute no candidates yet.
#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::{BTreeSet, VecDeque};

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::backend::{GenRequest, GenResponse, ModelBackend, StopReason, approx_tokens};
use crate::cascade::circuit_text::{
    CIRCUIT_SELF_CHECK_TEXT, ParsedCircuitPrompt, parse_circuit_prompt, render_circuit_answer,
};
use crate::cascade::qubit::{
    Circuit, ExactState, Gate, MAX_UNITARY_QUBITS, Target, agrees_on, apply_target,
};
use crate::util::{Result, derive_seed};

/// The reply to a prompt that does not follow the circuit protocol.
const CANNOT_READ: &str = "I cannot read this task.";

/// The reply when no candidate passed and no earlier answer can be passed on.
const NO_CIRCUIT: &str = "I found no circuit for this target.";

/// Steps every call may use without thinking tokens.
const FREE_STEPS: u64 = 16;

/// States a network search reaches for one step.
const STATES_PER_STEP: usize = 16;

/// The most states one network search holds, whatever the budget. It caps memory only: with
/// at most 4096 thinking tokens (the Traits' ceiling) a search reaches at most 33,024
/// states, so there the budget always binds first.
const MAX_NETWORK_STATES: usize = 1 << 18;

/// The temperature at which the method order is shuffled with probability one half: the
/// ceiling of the temperature Trait.
const TEMPERATURE_CEILING: f32 = 1.5;

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
    /// Word search, one gate at a time. Recognised, but it contributes no candidates yet.
    Search,
    /// Search from both ends. Recognised, but it contributes no candidates yet.
    Meet,
    /// Peephole rules and window resynthesis. Recognised, but it contributes no candidates
    /// yet.
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
    /// FREE_STEPS plus one step per `step_tokens` thinking tokens.
    fn new(think_tokens: u32, step_tokens: u32) -> Meter {
        let paid = u64::from(think_tokens) / u64::from(step_tokens.max(1));

        Meter {
            budget: FREE_STEPS + paid,
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
    let mut pool = Pool::new(target, self_check, task.max_gates);
    let mut tried = BTreeSet::new();

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
            Family::Previous => {
                for c in task.previous.iter().rev() {
                    pool.offer(c.clone(), meter);
                }
            }
            // Recognised, but they contribute no candidates yet.
            Family::Search | Family::Meet | Family::Simplify => {}
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

    /// Strategy texts of two more families.
    const COMPOSE: &str = "Conjugate the target by one CNOT on both sides.";
    const PREVIOUS: &str = "Check the previous answer before you build your own.";

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
        let checking =
            |families: &[&str]| format!("{} {CIRCUIT_SELF_CHECK_TEXT}", families.join(" "));
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
            // Misses: a family short. CH needs Search, which contributes nothing yet.
            ("fredkin without Compose", basis.clone(), fredkin(), None),
            (
                "toffoli without Hadamard",
                checking(&[PHASE, LINEAR]),
                toffoli(),
                None,
            ),
            ("swap by Phase", PHASE.to_string(), swap(), None),
            ("cz by Hadamard alone", HADAMARD.to_string(), cz(), None),
            ("cs by Search", SEARCH.to_string(), cs(), None),
            (
                "ch",
                checking(&[HADAMARD, PHASE, LINEAR, COMPOSE, PREVIOUS]),
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
        // The frame's H 0 and the first H of H X H stay uncancelled: no Simplify yet.
        assert_eq!(
            c.expect("answer").render(),
            "H 0; H 0; X 0; H 0; CNOT 1 0; X 0; X 1; H 0"
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
        let strong = format!("{HADAMARD} {PHASE} {LINEAR} {COMPOSE} {CIRCUIT_SELF_CHECK_TEXT}");

        // Phase alone on CCZ: the core test (8 steps) and the Walsh transform (8) fill the
        // 16 free steps, and verifying two columns takes 2 more, at 4 tokens each.
        let cases = [
            (&small, PHASE.to_string(), ccz(), 8, (7, 10)),
            (&small, format!("{PHASE} {LINEAR}"), ccz(), 196, (7, 6)),
            (&large, basis.clone(), toffoli(), 158, (7, 6)),
            (&large, basis.clone(), peres(), 124, (7, 5)),
            (&large, basis.clone(), rtof(), 78, (4, 3)),
            (&large, basis.clone(), ccix(), 92, (4, 4)),
            (&large, strong, fredkin(), 654, (7, 8)),
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

        // With nothing of its own, a Worker passes the latest earlier answer on, unchecked,
        // as long as it fits the gate cap.
        assert_eq!(solve(SEARCH, &[real.clone(), junk.clone()]), junk);

        let capped = render_circuit_solve_prompt(&cz(), 2, &[junk.clone(), real.clone()]);

        assert_eq!(answer(&model, SEARCH, capped, 4096), Some(junk));
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

        for model in [SimDesigner::small(), SimDesigner::large()] {
            for req in [&solve, &judge] {
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
            format!(
                "{HADAMARD} {PHASE} {LINEAR} {COMPOSE} {PREVIOUS} {SEARCH} {CIRCUIT_SELF_CHECK_TEXT}"
            ),
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
}
