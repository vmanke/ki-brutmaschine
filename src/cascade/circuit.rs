//! The circuit task family (step 1, ticket 03; ADR 0006): a library of target unitaries
//! with costs from the literature, panels of dressed tasks, exact grading, and the
//! pre-registered rediscovery count.
//!
//! Every target is built from its mathematical definition, never from a circuit, so no
//! circuit decides what is right. A verified target carries the T and CNOT counts that the
//! research read in a cited source, and each number has a status. A Proven or Derived
//! number is a lower bound: no exact circuit goes below it. A BestKnown number is only the
//! best the literature knows. Four entries have no verified numbers. They stay in the
//! library as test vectors for the checker and never enter a panel.
//!
//! A task is a verified target in a dressing: its qubits relabelled, and X gates on some
//! inputs and outputs. A dressing keeps the optimal T and CNOT counts, so the reference
//! still holds, but a memorised circuit no longer fits. A prompt shows the dressed target
//! only; a task's name, id, sources and reference never reach it.
//!
//! The exact checker of `qubit.rs` decides correctness, in integers. An exact answer below
//! a Proven or Derived number means that the checker or the reference is wrong, so it
//! raises an integrity alarm instead of earning a score. An exact answer below a BestKnown
//! number is an invention candidate. The rediscovery count keeps the cheapest one of each
//! target, marked as not yet checked. A human checks the literature before anyone calls it
//! new (Q38).
//!
//! Pure computation: no files, no network, no clock.
#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::cascade::arena::BreedConfig;
use crate::cascade::circuit_text::{
    CIRCUIT_SELF_CHECK_TEXT, parse_circuit_answer, render_circuit_judge_prompt,
    render_circuit_solve_prompt,
};
use crate::cascade::domain::{CircuitConfig, Domain, DomainConfig, Grade};
use crate::cascade::qubit::{Circuit, Column, Cost, Dressing, MAX_GATES, QError, Target, check};
use crate::util::{BrutError, Result, derive_seed};

/// How firmly a reference number is established.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bound {
    /// A cited source proves that no circuit does better.
    Proven,
    /// No circuit does better either, but the research derived this from published
    /// results and criteria; no source states it.
    Derived,
    /// The best number the literature knows. Whether it can be beaten is open.
    BestKnown,
}

impl Bound {
    /// True for Proven and Derived: no exact circuit may go below the number.
    fn is_lower_bound(self) -> bool {
        !matches!(self, Bound::BestKnown)
    }
}

/// The literature cost of a target: a T count and a CNOT count that one circuit reaches
/// together, each with its status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reference {
    pub t: u32,
    pub cnot: u32,
    pub t_bound: Bound,
    pub cnot_bound: Bound,
}

impl Reference {
    /// True when both counts are at or below the reference: the task is solved.
    pub fn reached(&self, c: &Cost) -> bool {
        c.t <= self.t && c.cnot <= self.cnot
    }

    /// True when a count is below a Proven or Derived number. No exact circuit can do
    /// that, so for an exact circuit it means that the checker or the reference is wrong.
    pub fn below_lower_bound(&self, c: &Cost) -> bool {
        (self.t_bound.is_lower_bound() && c.t < self.t)
            || (self.cnot_bound.is_lower_bound() && c.cnot < self.cnot)
    }

    /// True when a count is below a BestKnown number: for an exact circuit, an invention
    /// candidate. Each count is looked at alone, as in `below_lower_bound`, so fewer CNOT
    /// gates than a BestKnown CNOT count make a candidate even with more T gates.
    pub fn below_best_known(&self, c: &Cost) -> bool {
        (!self.t_bound.is_lower_bound() && c.t < self.t)
            || (!self.cnot_bound.is_lower_bound() && c.cnot < self.cnot)
    }
}

/// One entry of the target library.
#[derive(Clone, Copy, Debug)]
pub struct LibraryTarget {
    /// A short lower-case name for configs and reports. It never reaches a prompt.
    pub name: &'static str,
    pub qubits: usize,
    /// Builds the target from its definition, never from a circuit.
    pub build: fn() -> std::result::Result<Target, QError>,
    /// The verified T and CNOT counts. None when the research could not read both numbers
    /// in a source; such an entry never enters a panel.
    pub reference: Option<Reference>,
    /// Where the numbers come from: authors, venue, year, and figure or theorem.
    pub sources: &'static str,
    /// One plain line about the numbers.
    pub note: &'static str,
}

/// What a target builder returns.
type Built = std::result::Result<Target, QError>;

/// A reference from its two counts and their statuses.
const fn counts(t: u32, t_bound: Bound, cnot: u32, cnot_bound: Bound) -> Option<Reference> {
    Some(Reference {
        t,
        cnot,
        t_bound,
        cnot_bound,
    })
}

/// The library in the order of the research notes: two-qubit targets, then three-qubit
/// targets, then the four-qubit adder. Its 11 verified entries, in this order, are the
/// round-robin order of a panel.
pub static LIBRARY: [LibraryTarget; 15] = [
    LibraryTarget {
        name: "cz",
        qubits: 2,
        build: cz,
        reference: counts(0, Bound::Proven, 1, Bound::Proven),
        sources: "Amy, Maslov, Mosca, Roetteler, IEEE TCAD 32(6):818-830 (2013), Fig. 4(c)",
        note: "A Clifford gate needs no T gate, and an entangling gate needs at least one CNOT.",
    },
    LibraryTarget {
        name: "swap",
        qubits: 2,
        build: swap,
        reference: counts(0, Bound::Proven, 3, Bound::Proven),
        sources: "Vatan, Williams, Phys. Rev. A 69, 032315 (2004), Sec. VI, Thm 6; \
                  Shende, Bullock, Markov, Phys. Rev. A 70, 012310 (2004)",
        note: "A Clifford gate needs no T gate, and SWAP needs three CNOTs by its entangling \
               power.",
    },
    LibraryTarget {
        name: "cs",
        qubits: 2,
        build: cs,
        reference: counts(3, Bound::Proven, 2, Bound::Derived),
        sources: "Amy, Maslov, Mosca, Roetteler, IEEE TCAD 32(6) (2013), Fig. 5(b) and Thm 1; \
                  Gheorghiu, Mosca, Mukhopadhyay, npj Quantum Inf. 8, 141 (2022), Table 2",
        note: "T 3 is proven; CNOT 2 is derived from the Shende-Bullock-Markov test.",
    },
    LibraryTarget {
        name: "ch",
        qubits: 2,
        build: ch,
        reference: counts(2, Bound::Derived, 1, Bound::Proven),
        sources: "Amy, Maslov, Mosca, Roetteler, IEEE TCAD 32(6) (2013), Fig. 5(a), Fig. 12 \
                  and Thm 1",
        note: "One CNOT is trivially optimal; T 2 is derived, and an exhaustive check found \
               no circuit with one T gate.",
    },
    LibraryTarget {
        name: "qft2",
        qubits: 2,
        build: qft2,
        reference: counts(3, Bound::Proven, 3, Bound::Proven),
        sources: "Gheorghiu, Mosca, Mukhopadhyay, npj Quantum Inf. 8, 141 (2022), Table 3; \
                  Shende, Bullock, Markov, Phys. Rev. A 70, 012310 (2004), Example IV.3",
        note: "Both numbers are proven; the research found one circuit that reaches both.",
    },
    LibraryTarget {
        name: "w_gate",
        qubits: 2,
        build: w_gate,
        reference: None,
        sources: "Amy, Maslov, Mosca, Roetteler, IEEE TCAD 32(6) (2013), Fig. 6; CNOT bound \
                  from the criterion of Shende, Bullock, Markov, Phys. Rev. A 70, 012310 (2004)",
        note: "Unverified: T 2 was read in a source, CNOT 3 is derived; never on a panel.",
    },
    LibraryTarget {
        name: "sqrt_swap",
        qubits: 2,
        build: sqrt_swap,
        reference: None,
        sources: "Vatan, Williams, Phys. Rev. A 69, 032315 (2004), Fig. 6 (template); \
                  Shende, Bullock, Markov, Phys. Rev. A 70, 012310 (2004) (CNOT criterion)",
        note: "Unverified: T 3 and CNOT 3 are both derived, and no published T count was \
               found; never on a panel.",
    },
    LibraryTarget {
        name: "ccz",
        qubits: 3,
        build: ccz,
        reference: counts(7, Bound::Proven, 6, Bound::Proven),
        sources: "Gosset, Kliuchnikov, Mosca, Russo, Quantum Inf. Comput. 14(15-16):1261-1276 \
                  (2014); Shende, Markov, Quantum Inf. Comput. 9(5-6):461-486 (2009), Thm 15 \
                  and Cor. 34; Selinger, Phys. Rev. A 87, 042302 (2013), eqs. 4-6; Amy, \
                  Mosca, IEEE Trans. Inf. Theory 65(8) (2019), Ex. 3.1",
        note: "Both numbers are proven; the T count carries over from the Toffoli, which is \
               CCZ between two H gates on the target.",
    },
    LibraryTarget {
        name: "toffoli",
        qubits: 3,
        build: toffoli,
        reference: counts(7, Bound::Proven, 6, Bound::Proven),
        sources: "Gosset, Kliuchnikov, Mosca, Russo, Quantum Inf. Comput. 14 (2014), Sec. 5.3; \
                  Shende, Markov, Quantum Inf. Comput. 9 (2009), Thm 1; Maslov, Phys. Rev. A \
                  93, 022311 (2016), eq. 3",
        note: "Both numbers are proven, and the Nielsen-Chuang circuit reaches them together.",
    },
    LibraryTarget {
        name: "peres",
        qubits: 3,
        build: peres,
        reference: counts(7, Bound::Proven, 5, Bound::Proven),
        sources: "Shende, Markov, Quantum Inf. Comput. 9 (2009), Cor. 23 and Cor. 34; Mosca, \
                  Mukhopadhyay, Quantum Sci. Technol. 7, 015003 (2022), Table 1; Di Matteo, \
                  Mosca, Quantum Sci. Technol. 1, 015003 (2016), Table 1",
        note: "Both numbers are proven; T 7 holds because the Peres gate is \
               Clifford-equivalent to the Toffoli.",
    },
    LibraryTarget {
        name: "fredkin",
        qubits: 3,
        build: fredkin,
        reference: counts(7, Bound::Proven, 7, Bound::BestKnown),
        sources: "Gosset, Kliuchnikov, Mosca, Russo, Quantum Inf. Comput. 14 (2014); Niroula, \
                  Nam, npj Quantum Inf. 7, 37 (2021), Supplementary Note 1; Cruz, Murta, APL \
                  Quantum 1, 016105 (2024)",
        note: "BestKnown CNOT: an exact answer below it is an invention candidate (Q38), \
               never an automatic claim.",
    },
    LibraryTarget {
        name: "quantum_or",
        qubits: 3,
        build: quantum_or,
        reference: None,
        sources: "Mosca, Mukhopadhyay, Quantum Sci. Technol. 7, 015003 (2022), Table 1; Di \
                  Matteo, Mosca, Quantum Sci. Technol. 1, 015003 (2016), Table 1; Amy, Maslov, \
                  Mosca, Roetteler, IEEE TCAD 32(6) (2013), Fig. 7(c)",
        note: "Unverified: T 7 was read in a source, CNOT 6 is derived from local equivalence \
               to the Toffoli; never on a panel.",
    },
    LibraryTarget {
        name: "rtof",
        qubits: 3,
        build: rtof,
        reference: counts(4, Bound::Derived, 3, Bound::Proven),
        sources: "Maslov, Phys. Rev. A 93, 022311 (2016), Fig. 3 and Sec. IV",
        note: "CNOT 3 is optimal per Maslov; T 4 is derived, because the Toffoli is this gate \
               times a diagonal gate of T count 3.",
    },
    LibraryTarget {
        name: "ccix",
        qubits: 3,
        build: ccix,
        reference: counts(4, Bound::Derived, 4, Bound::BestKnown),
        sources: "Selinger, Phys. Rev. A 87, 042302 (2013), eqs. 7-11; Maslov, Phys. Rev. A \
                  93, 022311 (2016), Fig. 3",
        note: "BestKnown CNOT: an exact answer below it is an invention candidate (Q38), \
               never an automatic claim.",
    },
    LibraryTarget {
        name: "full_adder",
        qubits: 4,
        build: full_adder,
        reference: None,
        sources: "Di Matteo, Mosca, Quantum Sci. Technol. 1, 015003 (2016), Sec. 5.3; Mosca, \
                  Mukhopadhyay, Quantum Sci. Technol. 7, 015003 (2022), Table 1; Amy, Maslov, \
                  Mosca, Roetteler, IEEE TCAD 32(6) (2013), Fig. 11",
        note: "Unverified: T 7 was read in a source, but no source gives the CNOT count of a \
               7-T circuit; never on a panel.",
    },
];

/// The bits of basis index j on N qubits, qubit 0 first.
fn bits<const N: usize>(j: u32) -> [u32; N] {
    std::array::from_fn(|q| (j >> (N - 1 - q)) & 1)
}

/// The basis index of bits listed qubit 0 first.
fn index(bits: &[u32]) -> u32 {
    bits.iter().fold(0, |acc, &b| (acc << 1) | b)
}

/// CZ: diag(1, 1, 1, -1).
fn cz() -> Built {
    Target::monomial(2, |j| {
        let [a, b] = bits(j);

        (j, if a == 1 && b == 1 { 4 } else { 0 })
    })
}

/// SWAP: |a b> to |b a>.
fn swap() -> Built {
    Target::monomial(2, |j| {
        let [a, b] = bits(j);

        (index(&[b, a]), 0)
    })
}

/// CS: diag(1, 1, 1, i).
fn cs() -> Built {
    Target::monomial(2, |j| {
        let [a, b] = bits(j);

        (j, if a == 1 && b == 1 { 2 } else { 0 })
    })
}

/// CH: |0><0| (x) I + |1><1| (x) H, with qubit 0 as the control.
fn ch() -> Built {
    let columns = vec![
        Column::single(0b00, 0),
        Column::single(0b01, 0),
        Column::new(vec![(0b10, 0), (0b11, 0)])?,
        Column::new(vec![(0b10, 0), (0b11, 4)])?,
    ];

    Target::new(2, columns)
}

/// QFT2: entry (b, j) is i^(b j) / 2, the four-point Fourier transform with the bit
/// reversal included. As a power of w, i^(b j) is w^(2 b j).
fn qft2() -> Built {
    let columns = (0..4u8)
        .map(|j| Column::new((0..4u8).map(|b| (u32::from(b), 2 * b * j)).collect()))
        .collect::<std::result::Result<Vec<Column>, QError>>()?;

    Target::new(2, columns)
}

/// The W gate: a Hadamard on the span of |01> and |10>, and the identity on |00> and |11>.
fn w_gate() -> Built {
    let columns = vec![
        Column::single(0b00, 0),
        Column::new(vec![(0b01, 0), (0b10, 0)])?,
        Column::new(vec![(0b01, 0), (0b10, 4)])?,
        Column::single(0b11, 0),
    ];

    Target::new(2, columns)
}

/// sqrt(SWAP): the principal square root (1 + i)/2 I + (1 - i)/2 SWAP. (1 + i)/2 is
/// w / sqrt2, and (1 - i)/2 is w^7 / sqrt2.
fn sqrt_swap() -> Built {
    let columns = vec![
        Column::single(0b00, 0),
        Column::new(vec![(0b01, 1), (0b10, 7)])?,
        Column::new(vec![(0b01, 7), (0b10, 1)])?,
        Column::single(0b11, 0),
    ];

    Target::new(2, columns)
}

/// CCZ: diag(1, ..., 1, -1).
fn ccz() -> Built {
    Target::monomial(3, |j| {
        let [a, b, c] = bits(j);

        (j, if a == 1 && b == 1 && c == 1 { 4 } else { 0 })
    })
}

/// Toffoli: |a b c> to |a, b, c xor ab>.
fn toffoli() -> Built {
    Target::monomial(3, |j| {
        let [a, b, c] = bits(j);

        (index(&[a, b, c ^ (a & b)]), 0)
    })
}

/// Peres: |a b c> to |a, a xor b, c xor ab>.
fn peres() -> Built {
    Target::monomial(3, |j| {
        let [a, b, c] = bits(j);

        (index(&[a, a ^ b, c ^ (a & b)]), 0)
    })
}

/// Fredkin: when qubit 0 is 1, qubits 1 and 2 swap.
fn fredkin() -> Built {
    Target::monomial(3, |j| {
        let [a, b, c] = bits(j);
        let out = if a == 1 { [a, c, b] } else { [a, b, c] };

        (index(&out), 0)
    })
}

/// Quantum OR: |a b c> to |a, b, c xor (a or b)>.
fn quantum_or() -> Built {
    Target::monomial(3, |j| {
        let [a, b, c] = bits(j);

        (index(&[a, b, c ^ (a | b)]), 0)
    })
}

/// Maslov's relative-phase Toffoli: the identity except |101> to -|101>, |110> to i|111>
/// and |111> to -i|110>.
fn rtof() -> Built {
    Target::monomial(3, |j| match j {
        0b101 => (0b101, 4),
        0b110 => (0b111, 2),
        0b111 => (0b110, 6),
        _ => (j, 0),
    })
}

/// Selinger's doubly-controlled iX: the identity except |110> to i|111> and |111> to
/// i|110>.
fn ccix() -> Built {
    Target::monomial(3, |j| match j {
        0b110 => (0b111, 2),
        0b111 => (0b110, 2),
        _ => (j, 0),
    })
}

/// The 1-bit full adder: |a b c d> to |a, a xor b, a xor b xor c, d xor MAJ(a, b, c)>.
fn full_adder() -> Built {
    Target::monomial(4, |j| {
        let [a, b, c, d] = bits(j);
        let majority = (a & b) ^ (a & c) ^ (b & c);

        (index(&[a, a ^ b, a ^ b ^ c, d ^ majority]), 0)
    })
}

/// The entries with a verified reference, in library order.
pub fn verified() -> Vec<&'static LibraryTarget> {
    LIBRARY.iter().filter(|t| t.reference.is_some()).collect()
}

/// The entry with exactly this name.
pub fn library_target(name: &str) -> Option<&'static LibraryTarget> {
    LIBRARY.iter().find(|t| t.name == name)
}

/// The verified names, comma-separated, for error messages.
fn verified_names() -> String {
    verified()
        .iter()
        .map(|t| t.name)
        .collect::<Vec<_>>()
        .join(", ")
}

/// One task: a verified library target in a dressing.
#[derive(Clone, Debug, PartialEq)]
pub struct CircuitTask {
    /// The dressed target's fingerprint, salted with the task's index when an earlier task
    /// of the panel has the same fingerprint. It seeds model calls and names the task in
    /// the log; it never reaches a prompt.
    pub id: String,
    /// The library name. It never reaches a prompt.
    pub name: &'static str,
    /// The dressed target: the only part of the task a prompt shows.
    pub target: Target,
    /// The library reference. Dressing keeps the optimal T and CNOT counts, so it holds
    /// for the dressed target too.
    pub reference: Reference,
    pub dressing: Dressing,
    /// The gate cap of an answer, at most MAX_GATES.
    pub max_gates: usize,
}

/// `n` tasks from `seed`. Task i uses the verified base i mod L, where L counts the bases
/// with a reference; a base without one is skipped, so it never enters a panel. With
/// `dress`, task i draws its dressing from its own seed `derive_seed(seed, "circuit/{i}")`,
/// so a longer panel extends a shorter one; without, every task keeps its base unchanged.
/// A task's id is the dressed target's fingerprint. When an earlier task already has that
/// fingerprint, the id is the fingerprint, "-" and i. A fingerprint holds no "-", so ids
/// are unique. The gate cap is at most MAX_GATES.
pub fn circuit_panel(
    bases: &[(&'static LibraryTarget, Target)],
    seed: u64,
    n: usize,
    dress: bool,
    max_gates: usize,
) -> Vec<CircuitTask> {
    let usable: Vec<(&'static LibraryTarget, Reference, &Target)> = bases
        .iter()
        .filter_map(|(entry, base)| Some((*entry, entry.reference?, base)))
        .collect();
    let mut ids = BTreeSet::new();
    let mut panel = Vec::new();

    for (i, &(entry, reference, base)) in usable.iter().cycle().take(n).enumerate() {
        let dressing = if dress {
            let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(seed, &format!("circuit/{i}")));

            Dressing::draw(base.qubits(), &mut rng)
        } else {
            Dressing::identity(base.qubits())
        };
        let target = base.dress(&dressing);
        let fingerprint = target.fingerprint();
        let id = if ids.contains(&fingerprint) {
            format!("{fingerprint}-{i}")
        } else {
            fingerprint
        };

        ids.insert(id.clone());
        panel.push(CircuitTask {
            id,
            name: entry.name,
            target,
            reference,
            dressing,
            max_gates: max_gates.min(MAX_GATES),
        });
    }

    panel
}

/// The value of an exact circuit: 0.5 + 0.5 / (1 + dT + dC / (1 + dC)), where dT and dC
/// are the T and CNOT counts above the reference (0 when at or below it). It is 1 exactly
/// when both counts are at or below the reference, and it stays above 0.5, so every exact
/// circuit is worth more than any wrong one (which gets 0). dC / (1 + dC) stays below 1,
/// so one more T gate always costs more than any number of extra CNOT gates: the order is
/// lexicographic in the excess, T first, and both counts still give a gradient.
pub fn score_value(cost: &Cost, r: &Reference) -> f64 {
    let dt = f64::from(cost.t.saturating_sub(r.t));
    let dc = f64::from(cost.cnot.saturating_sub(r.cnot));

    0.5 + 0.5 / (1.0 + dt + dc / (1.0 + dc))
}

/// The grade of a circuit that is not exact.
const WRONG: Grade = Grade {
    correct: false,
    solved: false,
    value: 0.0,
};

/// The cost of `c` when it implements the task's target exactly, else None; a circuit the
/// checker refuses is not exact. Err(Integrity) when an exact circuit goes below a Proven
/// or Derived number.
fn exact_cost(task: &CircuitTask, c: &Circuit) -> Result<Option<Cost>> {
    let cost = match check(&task.target, c) {
        Ok(found) if found.exact => found.cost,
        _ => return Ok(None),
    };
    let r = &task.reference;

    if r.below_lower_bound(&cost) {
        return Err(BrutError::Integrity(format!(
            "circuit task {} ({}): an exact answer with T {} and CNOT {} is below the \
             reference T {} ({:?}), CNOT {} ({:?}); the checker or the reference is wrong",
            task.id, task.name, cost.t, cost.cnot, r.t, r.t_bound, r.cnot, r.cnot_bound
        )));
    }

    Ok(Some(cost))
}

/// Grade an answer. Only the exact checker decides correctness: a circuit that is not
/// exact, or that the checker refuses, gets 0. An exact circuit is solved when it reaches
/// the reference, and its value is `score_value`. Err(Integrity) when an exact circuit is
/// below a Proven or Derived number, because then the checker or the reference is wrong
/// and nothing may be committed. Below a BestKnown number is allowed.
pub fn grade_circuit(task: &CircuitTask, c: &Circuit) -> Result<Grade> {
    let Some(cost) = exact_cost(task, c)? else {
        return Ok(WRONG);
    };

    Ok(Grade {
        correct: true,
        solved: task.reference.reached(&cost),
        value: score_value(&cost, &task.reference),
    })
}

/// Some(true) when `a` is strictly better than `b`, Some(false) when `b` is, and None when
/// neither is. An exact circuit beats one that is not; between two exact circuits, fewer
/// T gates win, then fewer CNOT gates. That is the order the judge prompt states. A tie
/// and two wrong circuits give None, so they never score a judge. This raises no alarm;
/// `grade_circuit` does, and the breeder grades every answer before it compares answers.
pub fn prefer_circuit(task: &CircuitTask, a: &Circuit, b: &Circuit) -> Option<bool> {
    let key = |c: &Circuit| match check(&task.target, c) {
        Ok(found) if found.exact => Some((found.cost.t, found.cost.cnot)),
        _ => None,
    };

    match (key(a), key(b)) {
        (Some(x), Some(y)) => (x != y).then_some(x < y),
        (Some(_), None) => Some(true),
        (None, Some(_)) => Some(false),
        (None, None) => None,
    }
}

/// The circuit task family as a domain of the breeder.
#[derive(Clone, Debug)]
pub struct CircuitDomain {
    /// The chosen verified targets, built once, in panel order.
    bases: Vec<(&'static LibraryTarget, Target)>,
    dress: bool,
    max_gates: usize,
}

impl CircuitDomain {
    /// The domain of `cfg`. An empty target list means every verified target, in library
    /// order; otherwise the list's order is kept. Refused: a name that is not in the
    /// library, an entry without a verified reference, a name given twice, and a gate cap
    /// of 0 or above MAX_GATES.
    pub fn new(cfg: &CircuitConfig) -> Result<CircuitDomain> {
        let bad = |m: String| BrutError::Invalid(format!("circuit config: {m}"));

        if cfg.max_gates == 0 || cfg.max_gates > MAX_GATES {
            return Err(bad(format!(
                "max_gates must lie in 1..={MAX_GATES}, not {}",
                cfg.max_gates
            )));
        }

        let names: Vec<&str> = if cfg.targets.is_empty() {
            verified().iter().map(|t| t.name).collect()
        } else {
            cfg.targets.iter().map(String::as_str).collect()
        };
        let mut seen = BTreeSet::new();
        let mut bases = Vec::new();

        for name in names {
            let entry = library_target(name).ok_or_else(|| {
                bad(format!(
                    "unknown target {name:?}; the verified targets are {}",
                    verified_names()
                ))
            })?;

            if entry.reference.is_none() {
                return Err(bad(format!(
                    "target {name:?} is unverified: it has no reference, so it never enters \
                     a panel"
                )));
            }

            if !seen.insert(name) {
                return Err(bad(format!("target {name:?} is listed twice")));
            }

            let target = (entry.build)().map_err(|e| {
                BrutError::Integrity(format!("library target {name} does not build: {e}"))
            })?;

            bases.push((entry, target));
        }

        Ok(CircuitDomain {
            bases,
            dress: cfg.dress,
            max_gates: cfg.max_gates,
        })
    }
}

impl Domain for CircuitDomain {
    type Task = CircuitTask;
    type Answer = Circuit;

    fn from_config(cfg: &BreedConfig) -> Result<Self> {
        match &cfg.domain {
            DomainConfig::Circuits(circuits) => CircuitDomain::new(circuits),
            DomainConfig::Grid => Err(BrutError::Invalid(
                "breed config: this run breeds on circuits, but the config names grid puzzles"
                    .to_string(),
            )),
        }
    }

    fn panel(&self, seed: u64, n: usize) -> Vec<CircuitTask> {
        circuit_panel(&self.bases, seed, n, self.dress, self.max_gates)
    }

    fn task_id<'t>(&self, task: &'t CircuitTask) -> &'t str {
        &task.id
    }

    fn solve_prompt(&self, task: &CircuitTask, previous: &[Circuit]) -> String {
        render_circuit_solve_prompt(&task.target, task.max_gates, previous)
    }

    fn judge_prompt(&self, task: &CircuitTask, a: &Circuit, b: &Circuit) -> String {
        render_circuit_judge_prompt(&task.target, task.max_gates, a, b)
    }

    fn parse_answer(&self, task: &CircuitTask, text: &str) -> Option<Circuit> {
        parse_circuit_answer(text, task.target.qubits(), task.max_gates)
    }

    fn grade(&self, task: &CircuitTask, answer: &Circuit) -> Result<Grade> {
        grade_circuit(task, answer)
    }

    fn prefer(&self, task: &CircuitTask, a: &Circuit, b: &Circuit) -> Option<bool> {
        prefer_circuit(task, a, b)
    }

    fn graded(&self) -> bool {
        true
    }

    fn artefact(&self, answer: &Circuit) -> Option<String> {
        Some(answer.render())
    }

    fn self_check_text(&self) -> &'static str {
        CIRCUIT_SELF_CHECK_TEXT
    }
}

/// One member's answers on the search panel, as `rediscovery` reads them.
#[derive(Clone, Copy, Debug)]
pub struct Answered<'a> {
    pub id: &'a str,
    /// True for a hand-written seed Cascade of generation 0.
    pub seed: bool,
    pub generation: usize,
    /// One answer per panel task, in panel order.
    pub answers: &'a [Option<Circuit>],
}

/// An invention candidate: an exact answer below a BestKnown number. It is no claim. A
/// human checks the literature first, and only then may `literature_checked` become true
/// (Q38).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    /// The id of the member that gave the answer.
    pub member: String,
    pub generation: usize,
    /// True when the member is a seed Cascade of generation 0.
    pub seed: bool,
    /// The id of the panel task the answer is for. That task holds the dressing.
    pub task: String,
    /// The answer as text. It implements the dressed target of its task, so the task's
    /// dressing has to be undone before the literature check.
    pub circuit: String,
    pub cost: Cost,
    /// Always false when `rediscovery` writes it: only a human sets it, after checking the
    /// literature.
    pub literature_checked: bool,
}

/// What a run found on one library target.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TargetOutcome {
    /// The library name.
    pub target: String,
    pub reference: Reference,
    /// The panel tasks of this target: its dressings.
    pub tasks: usize,
    /// The cheapest exact answer of a seed, by the order of `Cost`: T first, then CNOT,
    /// then the other counts.
    pub best_seed: Option<Cost>,
    /// The cheapest exact answer of a bred member, in the same order.
    pub best_bred: Option<Cost>,
    /// A seed solved one of the target's tasks: exact, and at or below the reference.
    pub reached_by_seed: bool,
    /// A bred member solved one of the target's tasks.
    pub reached_by_bred: bool,
    /// The earliest generation of a bred member that solved one of the target's tasks.
    pub first_generation: Option<usize>,
    /// The text of the cheapest exact bred answer. It answers its own dressed task, so it
    /// implements the dressed target of that task, not the library target as written. It
    /// need not be the invention candidate.
    pub circuit: Option<String>,
    /// Some exact answer, of a seed or of a bred member, is below a BestKnown number. True
    /// exactly when `candidate` is Some.
    pub below_best_known: bool,
    /// The cheapest invention candidate, of a seed or of a bred member, for the human
    /// literature check (Q38). Only T and then CNOT decide, because the reference speaks
    /// only of these two counts. On a tie the answer counted first stays: the earlier
    /// member, then the earlier task. None when no exact answer is below a BestKnown number.
    pub candidate: Option<Candidate>,
}

impl TargetOutcome {
    /// The outcome of `task`'s target before any answer is counted.
    fn empty(task: &CircuitTask) -> TargetOutcome {
        TargetOutcome {
            target: task.name.to_string(),
            reference: task.reference,
            tasks: 0,
            best_seed: None,
            best_bred: None,
            reached_by_seed: false,
            reached_by_bred: false,
            first_generation: None,
            circuit: None,
            below_best_known: false,
            candidate: None,
        }
    }

    /// Count one exact answer `c` of member `m` on `task`, with cost `cost`. A hit or a
    /// candidate needs only one answer, so a later answer never clears it. On equal costs
    /// the answer counted first stays the best.
    fn record(&mut self, m: &Answered<'_>, task: &CircuitTask, c: &Circuit, cost: Cost) {
        let solved = self.reference.reached(&cost);

        if self.reference.below_best_known(&cost) {
            self.below_best_known = true;
            self.keep_candidate(m, task, c, cost);
        }

        if m.seed {
            self.reached_by_seed |= solved;

            if self.best_seed.is_none_or(|best| cost < best) {
                self.best_seed = Some(cost);
            }

            return;
        }

        self.reached_by_bred |= solved;

        if solved {
            let first = self
                .first_generation
                .map_or(m.generation, |g| g.min(m.generation));

            self.first_generation = Some(first);
        }

        if self.best_bred.is_none_or(|best| cost < best) {
            self.best_bred = Some(cost);
            self.circuit = Some(c.render());
        }
    }

    /// Make the answer `c` of member `m` on `task` the candidate when it is cheaper than the
    /// candidate so far, by T and then CNOT. The caller has checked that it is below a
    /// BestKnown number. On a tie the candidate so far stays.
    fn keep_candidate(&mut self, m: &Answered<'_>, task: &CircuitTask, c: &Circuit, cost: Cost) {
        let cheaper = self
            .candidate
            .as_ref()
            .is_none_or(|best| (cost.t, cost.cnot) < (best.cost.t, best.cost.cnot));

        if cheaper {
            self.candidate = Some(Candidate {
                member: m.id.to_string(),
                generation: m.generation,
                seed: m.seed,
                task: task.id.clone(),
                circuit: c.render(),
                cost,
                literature_checked: false,
            });
        }
    }
}

/// The pre-registered count of step 1, over the search panel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rediscovery {
    /// One outcome per library target on the panel, in order of first appearance.
    pub targets: Vec<TargetOutcome>,
    /// |L|: the distinct targets on the panel.
    pub library: usize,
    /// |R_seed|: the targets a seed solved.
    pub reached_by_seeds: usize,
    /// |R_bred \ R_seed|: the targets that only bred members solved.
    pub reached_by_bred_only: usize,
    /// D = |R_bred \ R_seed| / |L \ R_seed|; None when the seeds solved every target.
    pub share: Option<f64>,
    /// The calibration gate |R_seed| <= |L| / 3, decided in integers.
    pub calibrated: bool,
    /// Calibrated, and D > 1/2.
    pub passes: bool,
}

/// The pre-registered rediscovery count (plan (g)): per target on the panel, what the
/// seeds and the bred members solved, then |L|, |R_seed|, |R_bred \ R_seed|, the share D
/// and the gates. A target counts as solved when an exact answer at or below its
/// reference exists on at least one of its dressed tasks. Each target also keeps its
/// cheapest invention candidate (plan decision 8). `members` lists every archived
/// Cascade, with answers in panel order. Answers are counted member by member, in the
/// given order, and each member's in panel order. Err(Invalid) when a member's answer
/// count differs from the panel's task count, or when tasks of one target carry different
/// references. Err(Integrity) when an exact answer is below a Proven or Derived number.
pub fn rediscovery(panel: &[CircuitTask], members: &[Answered<'_>]) -> Result<Rediscovery> {
    if let Some(m) = members.iter().find(|m| m.answers.len() != panel.len()) {
        return Err(BrutError::Invalid(format!(
            "rediscovery: member {} has {} answers for {} panel tasks",
            m.id,
            m.answers.len(),
            panel.len()
        )));
    }

    let mut targets: Vec<TargetOutcome> = Vec::new();
    // For each panel task, the index of its target's outcome.
    let mut owner = Vec::new();

    for task in panel {
        let at = match targets.iter().position(|t| t.target == task.name) {
            Some(at) => at,
            None => {
                targets.push(TargetOutcome::empty(task));
                targets.len() - 1
            }
        };

        if targets[at].reference != task.reference {
            return Err(BrutError::Invalid(format!(
                "rediscovery: the tasks of target {} carry different references",
                task.name
            )));
        }

        targets[at].tasks += 1;
        owner.push(at);
    }

    for m in members {
        for ((task, answer), &at) in panel.iter().zip(m.answers).zip(&owner) {
            let Some(c) = answer else {
                continue;
            };

            if let Some(cost) = exact_cost(task, c)? {
                targets[at].record(m, task, c, cost);
            }
        }
    }

    let library = targets.len();
    let seeds = targets.iter().filter(|t| t.reached_by_seed).count();
    let bred_only = targets
        .iter()
        .filter(|t| t.reached_by_bred && !t.reached_by_seed)
        .count();
    // |L \ R_seed|: the targets the seeds left open.
    let open = library - seeds;
    let calibrated = 3 * seeds <= library;

    Ok(Rediscovery {
        targets,
        library,
        reached_by_seeds: seeds,
        reached_by_bred_only: bred_only,
        share: (open > 0).then(|| bred_only as f64 / open as f64),
        calibrated,
        // D > 1/2 in integers: 2 |R_bred \ R_seed| > |L \ R_seed|.
        passes: calibrated && 2 * bred_only > open,
    })
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::collections::BTreeMap;

    use rand::Rng;

    use super::*;
    use crate::cascade::circuit_text::{parse_circuit_prompt, render_circuit_answer};

    /// The Nielsen-Chuang Toffoli (controls 0 and 1, target 2): 7 T and 6 CNOT.
    const TOFFOLI_NC: &str = "H 2; CNOT 1 2; TDG 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; CNOT 0 2; \
                              T 1; T 2; H 2; CNOT 0 1; T 0; TDG 1; CNOT 0 1";

    /// The entries without a verified reference.
    const UNVERIFIED: [&str; 4] = ["w_gate", "sqrt_swap", "quantum_or", "full_adder"];

    fn circuit(text: &str) -> Circuit {
        text.parse().expect("circuit text")
    }

    fn cost(t: u32, cnot: u32) -> Cost {
        Cost {
            t,
            cnot,
            ..Cost::default()
        }
    }

    fn reference(t: u32, t_bound: Bound, cnot: u32, cnot_bound: Bound) -> Reference {
        Reference {
            t,
            cnot,
            t_bound,
            cnot_bound,
        }
    }

    /// One witness per library entry, from the literature (plan (d) and the research
    /// notes), with the T and CNOT counts its sources state. Gates are in time order, and
    /// qubit 0 is the most significant bit.
    fn witnesses() -> Vec<(&'static str, Circuit, (u32, u32))> {
        let nc: Vec<&str> = TOFFOLI_NC.split(';').map(str::trim).collect();
        let ccz: Vec<&str> = nc.iter().copied().filter(|g| *g != "H 2").collect();
        let fredkin = format!(
            "S 1; CNOT 2 1; SDG 1; H 2; S 2; H 2; S 2; {}; CNOT 2 1",
            nc[2..].join("; ")
        );
        // The adder's Toffoli has controls 1 and 2 and target 3.
        let inner = circuit(TOFFOLI_NC).relabel(&[1, 2, 3]).expect("relabel");
        // The Vatan-Williams template (Fig. 6) with alpha = beta = gamma = -pi/8, in its sign
        // conventions, with every rotation written in Clifford+T up to a global phase:
        // Rz(pi/2) is SDG, Rz(-3pi/4) is S T, Ry(3pi/4) is SDG H SDG TDG H S, Ry(-3pi/4) is
        // SDG H S T H S, and Rz(-pi/2) is S.
        let sqrt_swap = "SDG 0; CNOT 0 1; S 1; T 1; SDG 0; H 0; SDG 0; TDG 0; H 0; S 0; \
                         CNOT 1 0; SDG 0; H 0; S 0; T 0; H 0; S 0; CNOT 0 1; S 1";
        let table = [
            ("cz", "H 1; CNOT 0 1; H 1".to_string(), (0, 1)),
            ("swap", "CNOT 0 1; CNOT 1 0; CNOT 0 1".to_string(), (0, 3)),
            (
                "cs",
                "T 0; T 1; CNOT 0 1; TDG 1; CNOT 0 1".to_string(),
                (3, 2),
            ),
            (
                "ch",
                "S 1; H 1; T 1; CNOT 0 1; TDG 1; H 1; SDG 1".to_string(),
                (2, 1),
            ),
            (
                "qft2",
                "H 0; CNOT 0 1; CNOT 1 0; TDG 1; CNOT 0 1; T 1; T 0; H 0".to_string(),
                (3, 3),
            ),
            (
                "w_gate",
                "CNOT 0 1; SDG 0; H 0; TDG 0; CNOT 1 0; T 0; H 0; CNOT 0 1; S 0".to_string(),
                (2, 3),
            ),
            ("sqrt_swap", sqrt_swap.to_string(), (3, 3)),
            ("ccz", ccz.join("; "), (7, 6)),
            ("toffoli", TOFFOLI_NC.to_string(), (7, 6)),
            ("peres", nc[..14].join("; "), (7, 5)),
            ("fredkin", fredkin, (7, 7)),
            (
                "quantum_or",
                format!("X 0; X 1; {TOFFOLI_NC}; X 0; X 1; X 2"),
                (7, 6),
            ),
            (
                "rtof",
                "H 2; T 2; CNOT 1 2; TDG 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; H 2".to_string(),
                (4, 3),
            ),
            (
                "ccix",
                "H 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; CNOT 0 2; T 2; CNOT 1 2; TDG 2; H 2"
                    .to_string(),
                (4, 4),
            ),
            (
                "full_adder",
                format!(
                    "CNOT 0 1; CNOT 0 2; {}; CNOT 0 3; CNOT 1 2; CNOT 0 2",
                    inner.render()
                ),
                (7, 11),
            ),
        ];

        table
            .into_iter()
            .map(|(name, text, counts)| (name, circuit(&text), counts))
            .collect()
    }

    fn witness(name: &str) -> Circuit {
        witnesses()
            .into_iter()
            .find(|w| w.0 == name)
            .map(|w| w.1)
            .expect("a witness")
    }

    /// The witness of a task's target, dressed like the task.
    fn dressed(task: &CircuitTask) -> Circuit {
        task.dressing
            .dress_circuit(&witness(task.name))
            .expect("the dressing fits")
    }

    /// `c` followed by `t_pairs` pairs "T 0; TDG 0" and `cnot_pairs` pairs "CNOT 0 1;
    /// CNOT 0 1": the same unitary with 2 more T or CNOT gates per pair.
    fn padded(c: &Circuit, t_pairs: usize, cnot_pairs: usize) -> Circuit {
        let mut text = c.render();

        for _ in 0..t_pairs {
            text.push_str("; T 0; TDG 0");
        }

        for _ in 0..cnot_pairs {
            text.push_str("; CNOT 0 1; CNOT 0 1");
        }

        circuit(&text)
    }

    /// A dressed domain on the named targets with the default gate cap.
    fn domain(targets: &[&str]) -> CircuitDomain {
        let cfg = CircuitConfig {
            targets: targets.iter().map(|t| t.to_string()).collect(),
            ..CircuitConfig::default()
        };

        CircuitDomain::new(&cfg).expect("domain")
    }

    /// The task of `name` on a dressed panel of every verified target.
    fn task(name: &str) -> CircuitTask {
        domain(&[])
            .panel(1, 11)
            .into_iter()
            .find(|t| t.name == name)
            .expect("on the panel")
    }

    #[test]
    fn library_entries_are_well_formed() {
        let mut names = BTreeSet::new();
        let mut fingerprints = BTreeSet::new();

        assert_eq!(LIBRARY.len(), 15);

        for entry in &LIBRARY {
            let name = entry.name;
            let target = (entry.build)().expect(name);

            assert!(names.insert(name), "{name} twice");
            assert!(
                name.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{name}"
            );
            assert_eq!(target.qubits(), entry.qubits, "{name}");
            assert_eq!(
                Target::new(entry.qubits, target.columns().to_vec()).as_ref(),
                Ok(&target),
                "{name} is exactly unitary"
            );
            assert!(
                fingerprints.insert(target.fingerprint()),
                "{name} repeats a target"
            );
            assert!(entry.sources.contains(" (20"), "{name}: a cited year");
            assert!(
                !entry.note.trim().is_empty() && !entry.note.contains('\n'),
                "{name}"
            );
            assert_eq!(
                entry.reference.is_none(),
                UNVERIFIED.contains(&name),
                "{name}: None exactly for the unverified entries"
            );
            assert_eq!(library_target(name).map(|t| t.name), Some(name));
        }

        let verified_names: Vec<&str> = verified().iter().map(|t| t.name).collect();

        assert_eq!(
            verified_names,
            [
                "cz", "swap", "cs", "ch", "qft2", "ccz", "toffoli", "peres", "fredkin", "rtof",
                "ccix"
            ]
        );
        assert!(library_target("nope").is_none());
        assert!(library_target("Toffoli").is_none(), "names match exactly");
    }

    #[test]
    fn references_follow_the_plan_table() {
        use Bound::{BestKnown, Derived, Proven};

        let table = [
            ("cz", reference(0, Proven, 1, Proven)),
            ("swap", reference(0, Proven, 3, Proven)),
            ("cs", reference(3, Proven, 2, Derived)),
            ("ch", reference(2, Derived, 1, Proven)),
            ("qft2", reference(3, Proven, 3, Proven)),
            ("ccz", reference(7, Proven, 6, Proven)),
            ("toffoli", reference(7, Proven, 6, Proven)),
            ("peres", reference(7, Proven, 5, Proven)),
            ("fredkin", reference(7, Proven, 7, BestKnown)),
            ("rtof", reference(4, Derived, 3, Proven)),
            ("ccix", reference(4, Derived, 4, BestKnown)),
        ];
        let found: Vec<(&str, Reference)> = verified()
            .iter()
            .filter_map(|t| Some((t.name, t.reference?)))
            .collect();

        assert_eq!(found, table);

        for (name, _) in table {
            assert!(
                library_target(name).is_some_and(
                    |t| t.note.contains("BestKnown") == (name == "fredkin" || name == "ccix")
                ),
                "{name}: the note names a BestKnown number exactly when there is one"
            );
        }
    }

    #[test]
    fn every_witness_is_exact_and_verified_ones_match_their_reference() {
        let all = witnesses();
        let covered: BTreeSet<&str> = all.iter().map(|w| w.0).collect();

        assert_eq!(covered.len(), LIBRARY.len(), "one witness per entry");

        for (name, c, counts) in all {
            let entry = library_target(name).expect(name);
            let target = (entry.build)().expect(name);
            let found = check(&target, &c).expect(name);

            assert!(
                found.exact,
                "{name}: the witness and the definition disagree"
            );
            assert!(found.fidelity >= 1.0 - 1e-12, "{name}");
            assert_eq!((found.cost.t, found.cost.cnot), counts, "{name}");

            if let Some(r) = entry.reference {
                assert_eq!(
                    (r.t, r.cnot),
                    counts,
                    "{name}: the witness reaches the reference"
                );
                assert!(r.reached(&found.cost) && !r.below_lower_bound(&found.cost));
                assert!(!r.below_best_known(&found.cost), "{name}");
            }
        }
    }

    #[test]
    fn references_tell_reached_lower_bound_and_best_known_apart() {
        let fredkin = reference(7, Bound::Proven, 7, Bound::BestKnown);

        assert!(fredkin.reached(&cost(7, 7)) && fredkin.reached(&cost(7, 6)));
        assert!(!fredkin.reached(&cost(8, 7)) && !fredkin.reached(&cost(7, 8)));
        assert!(
            fredkin.below_lower_bound(&cost(6, 9)),
            "T below a proven number"
        );
        assert!(
            !fredkin.below_lower_bound(&cost(7, 6)),
            "below a best-known number only"
        );
        assert!(fredkin.below_best_known(&cost(7, 6)));
        assert!(fredkin.below_best_known(&cost(9, 6)), "each count alone");
        assert!(!fredkin.below_best_known(&cost(7, 7)) && !fredkin.below_best_known(&cost(6, 7)));

        // A BestKnown T count is looked at alone too. T 7 is a candidate and no alarm. T 8 is
        // no candidate, and CNOT 4 below the Proven 5 is still an alarm.
        let open_t = reference(8, Bound::BestKnown, 5, Bound::Proven);

        assert!(open_t.below_best_known(&cost(7, 9)) && !open_t.below_lower_bound(&cost(7, 9)));
        assert!(!open_t.below_best_known(&cost(8, 4)) && open_t.below_lower_bound(&cost(8, 4)));

        let derived = reference(4, Bound::Derived, 3, Bound::Derived);

        assert!(derived.below_lower_bound(&cost(3, 9)) && derived.below_lower_bound(&cost(9, 2)));
        assert!(!derived.below_best_known(&cost(3, 2)));
        assert!(!derived.below_lower_bound(&cost(4, 3)));
    }

    #[test]
    fn panels_are_deterministic_extending_and_round_robin() {
        let d = domain(&[]);
        let names: Vec<&str> = verified().iter().map(|t| t.name).collect();
        let panel = d.panel(5, 40);
        let mut ids = BTreeSet::new();

        assert_eq!(panel.len(), 40);
        assert_eq!(panel, d.panel(5, 40), "deterministic");
        assert_eq!(
            d.panel(5, 17)[..],
            panel[..17],
            "a longer panel extends a shorter one"
        );
        assert_ne!(d.panel(6, 40), panel, "seed-sensitive");
        assert!(d.panel(5, 0).is_empty());

        for (i, task) in panel.iter().enumerate() {
            let entry = library_target(task.name).expect("library");
            let base = (entry.build)().expect("base");
            let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(5, &format!("circuit/{i}")));

            assert_eq!(task.name, names[i % names.len()], "round robin");
            assert_eq!(task.dressing, Dressing::draw(base.qubits(), &mut rng));
            assert_eq!(task.target, base.dress(&task.dressing));
            assert_eq!(Some(task.reference), entry.reference);
            assert_eq!(task.max_gates, 64);
            assert!(ids.insert(task.id.clone()), "unique ids");

            let fingerprint = task.target.fingerprint();

            assert!(
                task.id == fingerprint || task.id == format!("{fingerprint}-{i}"),
                "{}",
                task.id
            );
        }

        // Dressings vary over the tasks of one target.
        let toffolis: BTreeSet<(&[u8], u32, u32)> = panel
            .iter()
            .filter(|t| t.name == "toffoli")
            .map(|t| {
                (
                    t.dressing.perm.as_slice(),
                    t.dressing.x_in,
                    t.dressing.x_out,
                )
            })
            .collect();

        assert!(toffolis.len() > 1);
    }

    #[test]
    fn ids_are_fingerprints_salted_only_on_a_collision() {
        let cfg = CircuitConfig {
            targets: vec!["swap".to_string(), "cs".to_string()],
            dress: false,
            max_gates: 32,
        };
        let d = CircuitDomain::new(&cfg).expect("domain");
        let panel = d.panel(3, 6);
        let swap = (library_target("swap").expect("swap").build)().expect("swap");
        let ids: BTreeSet<&str> = panel.iter().map(|t| t.id.as_str()).collect();

        assert_eq!(ids.len(), 6, "unique");
        assert_eq!(panel[0].target, swap, "no dressing");
        assert_eq!(panel[0].dressing, Dressing::identity(2));
        assert_eq!(panel[0].id, swap.fingerprint());
        assert_eq!(panel[2].id, format!("{}-2", swap.fingerprint()));
        assert_eq!(panel[4].id, format!("{}-4", swap.fingerprint()));
        assert_eq!(panel[3].id, format!("{}-3", panel[1].target.fingerprint()));
        assert_eq!(panel[5].max_gates, 32);

        // A cap above MAX_GATES is cut to it.
        let cz = library_target("cz").expect("cz");
        let bases = [(cz, (cz.build)().expect("cz"))];

        assert_eq!(
            circuit_panel(&bases, 1, 1, false, 999)[0].max_gates,
            MAX_GATES
        );
    }

    #[test]
    fn unverified_targets_never_enter_a_panel() {
        let entry = |name| library_target(name).expect(name);
        let built = |name| (entry(name).build)().expect(name);
        let only_unverified: Vec<(&'static LibraryTarget, Target)> =
            UNVERIFIED.iter().map(|&n| (entry(n), built(n))).collect();

        assert!(circuit_panel(&only_unverified, 1, 10, true, 64).is_empty());
        assert!(circuit_panel(&[], 1, 10, true, 64).is_empty());

        let mut mixed = only_unverified;

        mixed.insert(1, (entry("cz"), built("cz")));

        let panel = circuit_panel(&mixed, 1, 10, true, 64);

        assert_eq!(panel.len(), 10);
        assert!(panel.iter().all(|t| t.name == "cz"));
    }

    #[test]
    fn dressed_witnesses_solve_every_task() {
        for task in domain(&[]).panel(7, 33) {
            let c = dressed(&task);
            let undressed = witness(task.name);

            assert_eq!(
                grade_circuit(&task, &c).expect("grade"),
                Grade {
                    correct: true,
                    solved: true,
                    value: 1.0
                },
                "{} {:?}",
                task.name,
                task.dressing
            );
            assert_eq!(
                (c.cost().t, c.cost().cnot),
                (undressed.cost().t, undressed.cost().cnot)
            );
            assert_eq!(prefer_circuit(&task, &c, &circuit("X 0")), Some(true));
        }
    }

    #[test]
    fn prompts_never_carry_name_id_or_reference() {
        let d = domain(&[]);

        for task in d.panel(3, 22) {
            let mut other = task.clone();

            other.id = "another id".to_string();
            other.name = "another";
            other.reference = reference(99, Bound::BestKnown, 99, Bound::BestKnown);
            other.dressing = Dressing::identity(5);

            let previous = [circuit("H 0"), dressed(&task)];
            let (a, b) = (dressed(&task), circuit("CNOT 0 1"));
            let solve = d.solve_prompt(&task, &previous);
            let judge = d.judge_prompt(&task, &a, &b);

            assert_eq!(d.solve_prompt(&task, &[]), d.solve_prompt(&other, &[]));
            assert_eq!(solve, d.solve_prompt(&other, &previous));
            assert_eq!(judge, d.judge_prompt(&other, &a, &b));

            let text = format!("{solve}\n{judge}").to_lowercase();
            let words: BTreeSet<&str> = text
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .collect();

            for entry in &LIBRARY {
                assert!(!words.contains(entry.name), "{} in a prompt", entry.name);
            }

            assert!(!text.contains(&task.id), "the id in a prompt");
            assert!(
                !text.contains("maslov") && !text.contains("gosset"),
                "a source"
            );
        }
    }

    #[test]
    fn a_prompt_depends_only_on_the_dressed_target() {
        let d = domain(&[]);
        let panel = d.panel(4, 11);
        let (first, second) = (&panel[6], &panel[9]);
        let mut swapped = first.clone();
        let (a, b) = (circuit("H 0"), circuit("CNOT 0 1"));

        swapped.target = second.target.clone();

        assert_eq!((first.name, second.name), ("toffoli", "rtof"));
        assert_ne!(d.solve_prompt(first, &[]), d.solve_prompt(second, &[]));
        assert_eq!(d.solve_prompt(&swapped, &[]), d.solve_prompt(second, &[]));
        assert_eq!(
            d.judge_prompt(&swapped, &a, &b),
            d.judge_prompt(second, &a, &b)
        );

        // The prompt shows exactly the dressed target and the gate cap.
        let shown = parse_circuit_prompt(&d.solve_prompt(first, &[])).expect("prompt");

        assert_eq!(shown.target, first.target);
        assert_eq!(shown.max_gates, first.max_gates);
    }

    #[test]
    fn score_values_follow_the_formula() {
        let r = reference(7, Bound::Proven, 6, Bound::Proven);

        assert_eq!(score_value(&cost(7, 6), &r), 1.0);
        assert_eq!(score_value(&cost(0, 0), &r), 1.0);
        assert_eq!(score_value(&cost(8, 6), &r), 0.75);
        assert_eq!(score_value(&cost(7, 7), &r), 0.5 + 0.5 / 1.5);
        assert_eq!(score_value(&cost(9, 9), &r), 0.5 + 0.5 / 3.75);
        assert!(
            score_value(&cost(8, 6), &r) < score_value(&cost(7, 250), &r),
            "one more T costs more than many more CNOT gates"
        );
        assert!(score_value(&cost(u32::MAX, u32::MAX), &r) > 0.5);
    }

    #[test]
    fn score_order_is_the_lexicographic_order_of_the_excess() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let cap = u32::try_from(MAX_GATES).expect("small");

        for _ in 0..20_000 {
            let r = reference(
                rng.random_range(0..10),
                Bound::Proven,
                rng.random_range(0..10),
                Bound::BestKnown,
            );
            let top = if rng.random_bool(0.5) { 16 } else { cap };
            let a = cost(rng.random_range(0..=top), rng.random_range(0..=top));
            let b = cost(rng.random_range(0..=top), rng.random_range(0..=top));
            let excess = |c: &Cost| (c.t.saturating_sub(r.t), c.cnot.saturating_sub(r.cnot));
            let (va, vb) = (score_value(&a, &r), score_value(&b, &r));

            assert_eq!(
                va.partial_cmp(&vb),
                Some(excess(&b).cmp(&excess(&a))),
                "{a:?} {b:?} {r:?}"
            );
            assert_eq!(va == 1.0, r.reached(&a));
            assert!(va > 0.5 && va <= 1.0);

            // At or above the reference, that is the plain (T, CNOT) order of `prefer`.
            let above = |c: &Cost| c.t >= r.t && c.cnot >= r.cnot;

            if above(&a) && above(&b) {
                assert_eq!(va.partial_cmp(&vb), Some((b.t, b.cnot).cmp(&(a.t, a.cnot))));
            }
        }
    }

    #[test]
    fn value_order_equals_the_prefer_order_on_exact_circuits() {
        let task = task("toffoli");
        let base = dressed(&task);
        let mut rng = ChaCha8Rng::seed_from_u64(12);

        for _ in 0..300 {
            let a = padded(&base, rng.random_range(0..6), rng.random_range(0..6));
            let b = padded(&base, rng.random_range(0..6), rng.random_range(0..6));
            let ga = grade_circuit(&task, &a).expect("grade");
            let gb = grade_circuit(&task, &b).expect("grade");
            let by_value = match ga.value.partial_cmp(&gb.value) {
                Some(Ordering::Greater) => Some(true),
                Some(Ordering::Less) => Some(false),
                _ => None,
            };

            assert!(ga.correct && gb.correct);
            assert_eq!(
                prefer_circuit(&task, &a, &b),
                by_value,
                "{} | {}",
                a.render(),
                b.render()
            );
        }
    }

    #[test]
    fn grades_come_from_the_exact_checker() {
        let task = task("cs");
        let exact = dressed(&task);
        let costly = padded(&exact, 1, 1);

        assert_eq!(grade_circuit(&task, &circuit("X 0")).expect("grade"), WRONG);
        assert_eq!(
            grade_circuit(&task, &circuit("H 3")).expect("grade"),
            WRONG,
            "the checker refuses a qubit outside the target"
        );

        let g = grade_circuit(&task, &costly).expect("grade");

        assert!(g.correct && !g.solved);
        assert_eq!(g.value, score_value(&costly.cost(), &task.reference));
        assert_eq!(g.value, 0.5 + 0.5 / (1.0 + 2.0 + 2.0 / 3.0));
    }

    #[test]
    fn an_exact_answer_below_a_proven_number_raises_the_integrity_alarm() {
        let mut task = task("toffoli");
        let c = dressed(&task);
        let solved = Grade {
            correct: true,
            solved: true,
            value: 1.0,
        };

        assert_eq!(grade_circuit(&task, &c).expect("grade"), solved);

        // A forged, test-only reference that claims T 8 is a proven lower bound.
        task.reference = reference(8, Bound::Proven, 6, Bound::Proven);

        match grade_circuit(&task, &c) {
            Err(BrutError::Integrity(text)) => {
                assert!(text.contains(&task.id), "{text}");
                assert!(text.contains("T 7 and CNOT 6"), "{text}");
                assert!(text.contains("T 8 (Proven), CNOT 6 (Proven)"), "{text}");
            }
            other => panic!("expected an integrity alarm, got {other:?}"),
        }

        // A Derived CNOT number is a lower bound too.
        task.reference = reference(7, Bound::Proven, 7, Bound::Derived);
        assert!(matches!(
            grade_circuit(&task, &c),
            Err(BrutError::Integrity(_))
        ));

        // Below a BestKnown number is allowed: solved, value 1, an invention candidate.
        task.reference = reference(7, Bound::Proven, 7, Bound::BestKnown);
        assert_eq!(grade_circuit(&task, &c).expect("grade"), solved);
        assert!(task.reference.below_best_known(&c.cost()));

        // A wrong answer never raises the alarm.
        task.reference = reference(50, Bound::Proven, 50, Bound::Proven);
        assert_eq!(grade_circuit(&task, &circuit("X 0")).expect("grade"), WRONG);
    }

    #[test]
    fn prefer_ranks_exactness_then_t_then_cnot() {
        let task = task("toffoli");
        let exact = dressed(&task);
        let more_cnot = padded(&exact, 0, 1);
        let more_t = padded(&exact, 1, 0);
        let longer = circuit(&format!("{}; X 0; X 0", exact.render()));
        let wrong = circuit("X 0");
        let refused = circuit("H 5");

        assert_eq!(prefer_circuit(&task, &exact, &wrong), Some(true));
        assert_eq!(prefer_circuit(&task, &wrong, &exact), Some(false));
        assert_eq!(prefer_circuit(&task, &refused, &exact), Some(false));
        assert_eq!(prefer_circuit(&task, &wrong, &refused), None, "both wrong");
        assert_eq!(prefer_circuit(&task, &wrong, &wrong), None);
        assert_eq!(
            prefer_circuit(&task, &exact, &longer),
            None,
            "a tie in T and CNOT"
        );
        assert_eq!(prefer_circuit(&task, &exact, &exact), None);
        assert_eq!(prefer_circuit(&task, &exact, &more_cnot), Some(true));
        assert_eq!(prefer_circuit(&task, &more_cnot, &exact), Some(false));
        assert_eq!(
            prefer_circuit(&task, &more_t, &more_cnot),
            Some(false),
            "fewer T gates win over fewer CNOT gates"
        );
    }

    #[test]
    fn circuit_domain_refuses_bad_configs() {
        let cfg = |targets: &[&str], max_gates: usize| CircuitConfig {
            targets: targets.iter().map(|t| t.to_string()).collect(),
            dress: true,
            max_gates,
        };
        let refused = [
            (cfg(&["toffoli", "nope"], 64), "unknown target \"nope\""),
            (cfg(&["w_gate"], 64), "unverified"),
            (cfg(&["full_adder"], 64), "unverified"),
            (cfg(&["cz", "swap", "cz"], 64), "listed twice"),
            (cfg(&[], 0), "max_gates"),
            (cfg(&[], MAX_GATES + 1), "max_gates"),
        ];

        for (bad, why) in refused {
            match CircuitDomain::new(&bad) {
                Err(BrutError::Invalid(text)) => assert!(text.contains(why), "{text}"),
                other => panic!("{bad:?}: expected a refusal, got {other:?}"),
            }
        }

        let all = CircuitDomain::new(&cfg(&[], MAX_GATES)).expect("every verified target");
        let names: Vec<&str> = all.bases.iter().map(|(t, _)| t.name).collect();
        let verified_names: Vec<&str> = verified().iter().map(|t| t.name).collect();

        assert_eq!(names, verified_names);
        assert_eq!(all.max_gates, MAX_GATES);

        let mut breed = BreedConfig::default();

        assert!(matches!(
            CircuitDomain::from_config(&breed),
            Err(BrutError::Invalid(_))
        ));

        breed.domain = DomainConfig::Circuits(cfg(&["swap", "cs"], 1));

        let two = CircuitDomain::from_config(&breed).expect("circuits");
        let names: Vec<&str> = two.bases.iter().map(|(t, _)| t.name).collect();

        assert_eq!(names, ["swap", "cs"], "the config's order");
        assert_eq!((two.max_gates, two.dress), (1, true));

        breed.domain = DomainConfig::Circuits(cfg(&["swap", "sqrt_swap"], 64));
        assert!(CircuitDomain::from_config(&breed).is_err());
    }

    #[test]
    fn circuit_domain_forwards_to_the_circuit_functions() {
        let d = domain(&[]);
        let panel = d.panel(9, 13);
        let task = &panel[2];
        let w = dressed(task);
        let wrong = circuit("X 0");

        assert_eq!(panel, circuit_panel(&d.bases, 9, 13, true, 64));
        assert_eq!(task.name, "cs");
        assert_eq!(d.task_id(task), task.id);
        assert_eq!(
            d.solve_prompt(task, std::slice::from_ref(&w)),
            render_circuit_solve_prompt(&task.target, 64, std::slice::from_ref(&w))
        );
        assert_eq!(
            d.judge_prompt(task, &w, &wrong),
            render_circuit_judge_prompt(&task.target, 64, &w, &wrong)
        );
        assert_eq!(
            d.parse_answer(task, &render_circuit_answer(&w)),
            Some(w.clone())
        );
        assert_eq!(
            d.parse_answer(task, "ANSWER\nH 2"),
            None,
            "a 2-qubit target"
        );

        let mut short = task.clone();

        short.max_gates = w.gates().len() - 1;
        assert_eq!(
            d.parse_answer(&short, &render_circuit_answer(&w)),
            None,
            "the cap"
        );
        assert_eq!(
            d.grade(task, &w).expect("grade"),
            grade_circuit(task, &w).expect("grade")
        );
        assert_eq!(d.prefer(task, &w, &wrong), Some(true));
        assert!(d.graded());
        assert_eq!(d.artefact(&w), Some(w.render()));
        assert_eq!(d.self_check_text(), CIRCUIT_SELF_CHECK_TEXT);
    }

    /// A panel of swap, cs and toffoli, two dressed tasks each, in that order.
    fn small_panel() -> Vec<CircuitTask> {
        let panel = domain(&["swap", "cs", "toffoli"]).panel(21, 6);
        let names: Vec<&str> = panel.iter().map(|t| t.name).collect();

        assert_eq!(names, ["swap", "cs", "toffoli", "swap", "cs", "toffoli"]);

        panel
    }

    fn member<'a>(id: &'a str, generation: usize, answers: &'a [Option<Circuit>]) -> Answered<'a> {
        Answered {
            id,
            seed: generation == 0,
            generation,
            answers,
        }
    }

    #[test]
    fn rediscovery_counts_seed_and_bred_hits() {
        let panel = small_panel();
        let mut seed = vec![None; 6];
        let mut late = vec![None; 6];
        let mut early = vec![None; 6];
        let mut costly = vec![None; 6];
        let mut again = vec![None; 6];
        // The dressed witness of task i followed by gates that change neither T nor CNOT.
        let with =
            |i: usize, gates: &str| circuit(&format!("{}; {gates}", dressed(&panel[i]).render()));

        // A seed solves swap, then answers swap's other task exactly but with CNOT 5. It
        // answers cs wrongly, and toffoli exactly but with T 9: first with two more H
        // gates, then without them.
        seed[0] = Some(dressed(&panel[0]));
        seed[1] = Some(circuit("X 0"));
        seed[2] = Some(padded(&with(2, "H 0; H 0"), 1, 0));
        seed[3] = Some(padded(&dressed(&panel[3]), 0, 1));
        seed[5] = Some(padded(&dressed(&panel[5]), 1, 0));
        // Bred members: cs with two more H gates and toffoli in generation 3; toffoli at
        // the same full cost in generation 2, listed after generation 3; an exact but
        // costly toffoli (T 9) in generation 1, listed after both hits; and in
        // generation 4, the seed's swap again and cs without the two H gates.
        late[4] = Some(with(4, "H 0; H 0"));
        late[5] = Some(with(5, "X 0; X 0"));
        early[5] = Some(with(5, "Z 0; Z 0"));
        costly[2] = Some(padded(&dressed(&panel[2]), 1, 0));
        again[1] = Some(dressed(&panel[1]));
        again[3] = Some(dressed(&panel[3]));

        assert_eq!(
            cost_of(&late[5]),
            cost_of(&early[5]),
            "a tie in the full cost"
        );
        assert_ne!(late[5], early[5]);

        // The answers of each pair agree in T and CNOT, and the later one has fewer H
        // gates. So only the full order of `Cost` makes the later answer the cheaper one.
        for (first, later) in [(&seed[2], &seed[5]), (&late[4], &again[1])] {
            let (a, b) = (cost_of(first), cost_of(later));

            assert_eq!((a.t, a.cnot), (b.t, b.cnot));
            assert!(b.h < a.h && b < a, "{a:?} {b:?}");
        }

        let members = [
            member("seed", 0, &seed),
            member("late", 3, &late),
            member("early", 2, &early),
            member("costly", 1, &costly),
            member("again", 4, &again),
        ];
        let found = rediscovery(&panel, &members).expect("rediscovery");
        let [swap, cs, toffoli] = &found.targets[..] else {
            panic!("three targets: {found:?}");
        };

        // A later answer above the reference keeps the seed's hit.
        assert_eq!((swap.target.as_str(), swap.tasks), ("swap", 2));
        assert_eq!(swap.reference, panel[0].reference);
        assert!(swap.reached_by_seed && swap.reached_by_bred);
        assert_eq!(swap.best_seed, Some(cost_of(&seed[0])));
        assert_eq!(swap.best_bred, Some(cost_of(&again[3])));
        assert_eq!(swap.first_generation, Some(4));
        assert_eq!(swap.circuit, Some(dressed(&panel[3]).render()));

        // The cheapest bred answer comes last, and the earliest hit first.
        assert!(!cs.reached_by_seed && cs.reached_by_bred);
        assert_eq!(cs.best_seed, None, "a wrong answer is no cost");
        assert_eq!(cs.best_bred, Some(cost_of(&again[1])));
        assert_eq!(cs.first_generation, Some(3));
        assert_eq!(cs.circuit, again[1].as_ref().map(Circuit::render));

        // Exact seed answers above the reference are no hit, and the second, cheaper one is
        // the best. The bred hits come before an exact miss, the earliest hit comes second,
        // and on the tie the answer counted first stays.
        assert!(!toffoli.reached_by_seed && toffoli.reached_by_bred);
        assert_eq!(toffoli.best_seed, Some(cost_of(&seed[5])));
        assert_eq!(toffoli.first_generation, Some(2), "the earliest bred hit");
        assert_eq!(toffoli.best_bred, Some(cost_of(&late[5])));
        assert_eq!(toffoli.circuit, late[5].as_ref().map(Circuit::render));
        assert!(!swap.below_best_known && !cs.below_best_known && !toffoli.below_best_known);
        assert!(found.targets.iter().all(|t| t.candidate.is_none()));

        // swap counts for the seeds only, although a bred member solves it too.
        assert_eq!(found.library, 3);
        assert_eq!((found.reached_by_seeds, found.reached_by_bred_only), (1, 2));
        assert_eq!(found.share, Some(1.0));
        assert!(found.calibrated && found.passes);

        // An exact answer above the reference is no hit, but it is the best bred cost.
        let found = rediscovery(&panel, &[members[0], members[3]]).expect("rediscovery");
        let swap = &found.targets[0];

        assert!(swap.reached_by_seed && !swap.reached_by_bred);
        assert_eq!((swap.best_bred, swap.first_generation), (None, None));
        assert_eq!(swap.circuit, None);
        assert!(!found.targets[2].reached_by_bred);
        assert_eq!(found.targets[2].best_bred, Some(cost_of(&costly[2])));
        assert_eq!(
            found.targets[2].circuit,
            costly[2].as_ref().map(Circuit::render)
        );
        assert_eq!(found.targets[2].first_generation, None);
        assert_eq!((found.reached_by_bred_only, found.share), (0, Some(0.0)));
        assert!(found.calibrated && !found.passes);

        // D = 1/2 does not pass: the gate is D > 1/2.
        let mut cs_only = vec![None; 6];

        cs_only[1] = Some(dressed(&panel[1]));

        let found =
            rediscovery(&panel, &[members[0], member("cs", 2, &cs_only)]).expect("rediscovery");

        assert_eq!(found.share, Some(0.5));
        assert!(found.calibrated && !found.passes);

        // Seeds that solve two of three targets fail the calibration gate.
        let mut strong = seed.clone();

        strong[4] = Some(dressed(&panel[4]));

        let found =
            rediscovery(&panel, &[member("strong", 0, &strong), members[2]]).expect("rediscovery");

        assert_eq!(found.reached_by_seeds, 2);
        assert_eq!(found.share, Some(1.0));
        assert!(!found.calibrated && !found.passes);
    }

    /// The cost of an answer that is there.
    fn cost_of(answer: &Option<Circuit>) -> Cost {
        answer.as_ref().expect("an answer").cost()
    }

    #[test]
    fn rediscovery_share_is_none_when_the_seeds_solve_every_target() {
        let panel = small_panel();
        let answers: Vec<Option<Circuit>> = panel.iter().map(|t| Some(dressed(t))).collect();
        let found = rediscovery(&panel, &[member("seed", 0, &answers)]).expect("rediscovery");

        assert_eq!((found.library, found.reached_by_seeds), (3, 3));
        assert_eq!(found.share, None);
        assert!(!found.calibrated && !found.passes);

        let found = rediscovery(&panel, &[]).expect("no members");

        assert_eq!((found.library, found.reached_by_seeds), (3, 0));
        assert_eq!(found.share, Some(0.0));
        assert!(
            found
                .targets
                .iter()
                .all(|t| t.tasks == 2 && t.best_bred.is_none())
        );

        let empty = rediscovery(&[], &[]).expect("empty panel");

        assert_eq!((empty.library, empty.share), (0, None));
        assert!(empty.targets.is_empty() && !empty.passes);
    }

    #[test]
    fn rediscovery_raises_the_alarm_and_refuses_misaligned_input() {
        let mut panel = small_panel();
        let mut answers = vec![None; 6];

        answers[5] = Some(dressed(&panel[5]));

        // An exact answer below a forged Proven number is an integrity alarm, for seeds and
        // bred members alike.
        for task in panel.iter_mut().filter(|t| t.name == "toffoli") {
            task.reference = reference(7, Bound::Proven, 7, Bound::Proven);
        }

        for generation in [0, 4] {
            match rediscovery(&panel, &[member("m", generation, &answers)]) {
                Err(BrutError::Integrity(text)) => assert!(text.contains(&panel[5].id), "{text}"),
                other => panic!("expected an integrity alarm, got {other:?}"),
            }
        }

        // Below a BestKnown number, the target is flagged as an invention candidate.
        for task in panel.iter_mut().filter(|t| t.name == "toffoli") {
            task.reference = reference(7, Bound::Proven, 7, Bound::BestKnown);
        }

        let found = rediscovery(&panel, &[member("m", 4, &answers)]).expect("rediscovery");

        assert!(found.targets[2].below_best_known && found.targets[2].reached_by_bred);
        assert!(!found.targets[0].below_best_known);
        assert_eq!(found.targets[0].candidate, None);

        let candidate = found.targets[2].candidate.as_ref().expect("a candidate");

        assert_eq!((candidate.member.as_str(), candidate.seed), ("m", false));
        assert_eq!(candidate.task, panel[5].id);

        // A seed's answer is a candidate as well.
        let found = rediscovery(&panel, &[member("s", 0, &answers)]).expect("rediscovery");

        assert!(found.targets[2].below_best_known && found.targets[2].reached_by_seed);
        assert!(
            found.targets[2]
                .candidate
                .as_ref()
                .is_some_and(|k| k.seed && k.member == "s")
        );

        // A hit and candidate (T 7, CNOT 6), then an exact answer that is neither (T 7,
        // CNOT 8): the later answer clears no flag, for seeds and bred members alike.
        let mut then_neither = vec![None; 6];

        then_neither[2] = Some(dressed(&panel[2]));
        then_neither[5] = Some(padded(&dressed(&panel[5]), 0, 1));

        for generation in [0, 4] {
            let found = rediscovery(&panel, &[member("m", generation, &then_neither)])
                .expect("rediscovery");
            let toffoli = &found.targets[2];

            assert!(toffoli.below_best_known, "generation {generation}");
            assert_eq!(
                (toffoli.reached_by_seed, toffoli.reached_by_bred),
                (generation == 0, generation > 0)
            );
            assert_eq!(
                toffoli.candidate.as_ref().map(|k| k.cost),
                Some(cost_of(&then_neither[2])),
                "generation {generation}"
            );
        }

        // A candidate needs no hit. An exact answer with T 9 and CNOT 6 misses the reference,
        // but its CNOT count is below the BestKnown 7. It flags the target for seeds and bred
        // members alike.
        let mut no_hit = vec![None; 6];

        no_hit[2] = Some(padded(&dressed(&panel[2]), 1, 0));

        let c = cost_of(&no_hit[2]);

        assert_eq!((c.t, c.cnot), (9, 6));

        for generation in [0, 4] {
            let found =
                rediscovery(&panel, &[member("m", generation, &no_hit)]).expect("rediscovery");
            let toffoli = &found.targets[2];

            assert!(toffoli.below_best_known, "generation {generation}");
            assert!(
                !toffoli.reached_by_seed && !toffoli.reached_by_bred,
                "generation {generation}"
            );
            assert_eq!(
                toffoli.candidate.as_ref().map(|k| (k.cost, k.seed)),
                Some((c, generation == 0)),
                "generation {generation}"
            );
        }

        // Answers that do not align with the panel are refused, too few and too many alike.
        for (id, count) in [("short", 5), ("long", 7)] {
            let misaligned = vec![None; count];

            match rediscovery(&panel, &[member(id, 2, &misaligned)]) {
                Err(BrutError::Invalid(text)) => assert!(text.contains(id), "{text}"),
                other => panic!("expected a refusal, got {other:?}"),
            }
        }

        // So are tasks of one target with different references.
        panel[2].reference = reference(7, Bound::Proven, 6, Bound::Proven);

        assert!(matches!(
            rediscovery(&panel, &[]),
            Err(BrutError::Invalid(_))
        ));
    }

    /// The toffoli outcome of a rediscovery over `members`. It also checks, on every target,
    /// that the flag agrees with the candidate.
    fn toffoli_outcome(panel: &[CircuitTask], members: &[Answered<'_>]) -> TargetOutcome {
        let found = rediscovery(panel, members).expect("rediscovery");

        for t in &found.targets {
            assert_eq!(t.below_best_known, t.candidate.is_some(), "{}", t.target);
        }

        found
            .targets
            .into_iter()
            .find(|t| t.target == "toffoli")
            .expect("toffoli on the panel")
    }

    #[test]
    fn rediscovery_keeps_the_cheapest_invention_candidate() {
        let mut panel = small_panel();

        // A forged, test-only toffoli reference: T 7 (Proven) and CNOT 9 (BestKnown). An
        // exact toffoli answer with fewer than 9 CNOT gates is then an invention candidate.
        for task in panel.iter_mut().filter(|t| t.name == "toffoli") {
            task.reference = reference(7, Bound::Proven, 9, Bound::BestKnown);
        }

        let with =
            |i: usize, gates: &str| circuit(&format!("{}; {gates}", dressed(&panel[i]).render()));
        let expected = |id: &str, generation: usize, i: usize, answer: &Option<Circuit>| {
            Some(Candidate {
                member: id.to_string(),
                generation,
                seed: generation == 0,
                task: panel[i].id.clone(),
                circuit: answer.as_ref().expect("an answer").render(),
                cost: cost_of(answer),
                literature_checked: false,
            })
        };
        let mut seed = vec![None; 6];
        let mut above = vec![None; 6];
        let mut more_cnot = vec![None; 6];
        let mut cheap = vec![None; 6];
        let mut tie = vec![None; 6];

        // The dressed toffoli witness has T 7 and CNOT 6. The seed adds two T gates on task
        // 5. The member "above" adds four CNOT gates on task 2: CNOT 10 is above every
        // reference, so that answer is no candidate, although it has fewer T gates than the
        // seed's. "more_cnot" adds two CNOT gates on task 2. "cheap" answers task 2 with
        // two more H gates, and task 5 with the witness alone. "tie" answers task 2 with the
        // witness alone, in an earlier generation.
        seed[5] = Some(padded(&dressed(&panel[5]), 1, 0));
        above[2] = Some(padded(&dressed(&panel[2]), 0, 2));
        more_cnot[2] = Some(padded(&dressed(&panel[2]), 0, 1));
        cheap[2] = Some(with(2, "H 0; H 0"));
        cheap[5] = Some(dressed(&panel[5]));
        tie[2] = Some(dressed(&panel[2]));

        let counts = |answer: &Option<Circuit>| {
            let c = cost_of(answer);

            (c.t, c.cnot)
        };

        assert_eq!(
            [&seed[5], &above[2], &more_cnot[2]].map(counts),
            [(9, 6), (7, 10), (7, 8)]
        );
        assert_eq!([&cheap[2], &cheap[5], &tie[2]].map(counts), [(7, 6); 3]);

        // The later answer of each tie is cheaper in the full order of Cost, so only a rule
        // that compares T and CNOT alone keeps the earlier one.
        assert!(cost_of(&cheap[5]) < cost_of(&cheap[2]) && cost_of(&tie[2]) < cost_of(&cheap[2]));

        let members = [
            member("seed", 0, &seed),
            member("above", 3, &above),
            member("more_cnot", 4, &more_cnot),
            member("cheap", 5, &cheap),
            member("tie", 1, &tie),
        ];

        // A seed's answer is a candidate, and so is a bred member's.
        assert_eq!(
            toffoli_outcome(&panel, &[members[0]]).candidate,
            expected("seed", 0, 5, &seed[5])
        );
        assert_eq!(
            toffoli_outcome(&panel, &[members[2]]).candidate,
            expected("more_cnot", 4, 2, &more_cnot[2])
        );

        // The cheaper of two candidates wins, whichever comes first: fewer T gates against
        // the seed, and fewer CNOT gates at equal T against "more_cnot".
        for other in [members[0], members[2]] {
            for pair in [[other, members[3]], [members[3], other]] {
                assert_eq!(
                    toffoli_outcome(&panel, &pair).candidate,
                    expected("cheap", 5, 2, &cheap[2]),
                    "{} first",
                    pair[0].id
                );
            }
        }

        // T decides before CNOT, whichever comes first: "more_cnot" with T 7 and CNOT 8 beats
        // the seed with T 9 and CNOT 6. The two counts disagree here, so a rule that looks at
        // CNOT first, or at their sum, would pick the seed at least once.
        for pair in [[members[0], members[2]], [members[2], members[0]]] {
            assert_eq!(
                toffoli_outcome(&panel, &pair).candidate,
                expected("more_cnot", 4, 2, &more_cnot[2]),
                "{} first",
                pair[0].id
            );
        }

        // A cheaper answer above every reference never becomes the candidate, although it is
        // the best bred answer.
        let alone = toffoli_outcome(&panel, &[members[1]]);
        let after_seed = toffoli_outcome(&panel, &[members[0], members[1]]);

        assert_eq!(alone.candidate, None);
        assert_eq!(after_seed.candidate, expected("seed", 0, 5, &seed[5]));
        assert_eq!(after_seed.best_bred, Some(cost_of(&above[2])));
        assert_eq!(after_seed.circuit, above[2].as_ref().map(Circuit::render));

        // On a tie in T and CNOT the answer counted first stays: the member listed first, and
        // within a member the earlier task. The generation does not matter.
        assert_eq!(
            toffoli_outcome(&panel, &[members[3], members[4]]).candidate,
            expected("cheap", 5, 2, &cheap[2])
        );
        assert_eq!(
            toffoli_outcome(&panel, &[members[4], members[3]]).candidate,
            expected("tie", 1, 2, &tie[2])
        );

        // Over all members, the candidate goes to rediscovery.json unchecked and comes back
        // unchanged.
        let found = rediscovery(&panel, &members).expect("rediscovery");
        let toffoli = &found.targets[2];
        let text = serde_json::to_string(toffoli).expect("json");

        assert_eq!(toffoli.candidate, expected("cheap", 5, 2, &cheap[2]));
        assert!(toffoli.below_best_known && !toffoli.reached_by_seed && toffoli.reached_by_bred);
        assert!(
            text.contains(r#""candidate":{"member":"cheap","generation":5,"seed":false,"#),
            "{text}"
        );
        assert!(text.contains(r#""literature_checked":false"#), "{text}");
        assert_eq!(
            serde_json::from_str::<TargetOutcome>(&text).expect("back"),
            *toffoli
        );

        let text = serde_json::to_string(&found).expect("json");

        assert_eq!(
            serde_json::from_str::<Rediscovery>(&text).expect("back"),
            found
        );
    }

    #[test]
    fn answers_and_outcomes_round_trip_through_json() {
        let panel = small_panel();
        let answers: Vec<Option<Circuit>> = vec![
            Some(circuit("H 0; CNOT 0 1")),
            None,
            Some(dressed(&panel[2])),
        ];
        let text = serde_json::to_string(&answers).expect("json");

        assert!(text.starts_with(r#"["H 0; CNOT 0 1",null,""#), "{text}");
        assert_eq!(
            serde_json::from_str::<Vec<Option<Circuit>>>(&text).expect("back"),
            answers
        );

        // Select answers as a snapshot keeps them: member id, then task id.
        let mut select: BTreeMap<String, BTreeMap<String, Option<Circuit>>> = BTreeMap::new();

        for (task, answer) in panel.iter().zip(&answers) {
            select
                .entry("m1".to_string())
                .or_default()
                .insert(task.id.clone(), answer.clone());
        }

        select.entry("m2".to_string()).or_default();

        let text = serde_json::to_string(&select).expect("json");

        assert_eq!(
            serde_json::from_str::<BTreeMap<String, BTreeMap<String, Option<Circuit>>>>(&text)
                .expect("back"),
            select
        );

        // The rediscovery file round-trips too, and bounds read in snake case.
        let found = rediscovery(
            &panel,
            &[member("seed", 0, &[None, None, None, None, None, None])],
        )
        .expect("rediscovery");
        let text = serde_json::to_string(&found).expect("json");

        assert_eq!(
            serde_json::from_str::<Rediscovery>(&text).expect("back"),
            found
        );
        assert!(text.contains(r#""t_bound":"proven""#), "{text}");
        assert_eq!(
            serde_json::to_string(&Bound::BestKnown).expect("json"),
            r#""best_known""#
        );
        assert!(serde_json::from_str::<Bound>(r#""guess""#).is_err());
    }
}
