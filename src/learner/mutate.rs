//! Mutation operators and the program-proposer hook.
//!
//! Owner: WP1. Contract section 5.4. Pure: no I/O.
#![forbid(unsafe_code)]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::learner::dsl::{ALL_OPS, ConstKind, Slot, sig, slot_len};
use crate::learner::seeds::{reference_nn, seed_programs};
use crate::learner::types::{Component, DslLimits, Instr, LearnerGenome, Mutation, VariationMode};
use crate::util::{derive_seed, log_uniform_f32};

/// Retries after the first attempt before the fallbacks run.
const RETRIES: usize = 8;

/// Fresh draws per constant in the Const fallback.
const FALLBACK_DRAWS: usize = 4;

/// Insert attempts in the last fallback.
const FALLBACK_INSERTS: usize = 32;

/// Log-uniform range of fresh Value and Scale constants.
const CONST_LO: f32 = 1.0e-3;
const CONST_HI: f32 = 1.0;

/// The three parts in execution order.
const COMPONENTS: [Component; 3] = [Component::Setup, Component::Predict, Component::Learn];

/// A uniformly random register index for `slot` (0 for an unused slot).
fn random_operand(rng: &mut ChaCha8Rng, slot: Slot) -> u8 {
    match slot_len(slot) {
        0 => 0,
        n => rng.random_range(0..n) as u8,
    }
}

/// A fresh constant of the given kind (0.0 for ConstKind::None).
fn random_const(rng: &mut ChaCha8Rng, kind: ConstKind) -> f32 {
    match kind {
        ConstKind::None => 0.0,

        ConstKind::Value => {
            let magnitude = log_uniform_f32(rng, CONST_LO, CONST_HI);

            if rng.random_bool(0.5) {
                -magnitude
            } else {
                magnitude
            }
        }

        ConstKind::Scale => log_uniform_f32(rng, CONST_LO, CONST_HI),
        ConstKind::Index => rng.random_range(0..=255u32) as f32,
    }
}

/// A uniformly random valid instruction for `comp` (op uniform over ALL_OPS, operands
/// uniform over their slot range, Value/Scale constants log-uniform in [1e-3, 1] with a
/// random sign for Value, Index uniform 0..=255).
///
/// All three components share one op set, so `comp` does not change the distribution.
pub fn random_instr(rng: &mut ChaCha8Rng, comp: Component) -> Instr {
    let _ = comp;

    let op = ALL_OPS[rng.random_range(0..ALL_OPS.len())];
    let s = sig(op);

    let a = random_operand(rng, s.a);
    let b = random_operand(rng, s.b);
    let out = random_operand(rng, s.out);
    let c = random_const(rng, s.c);

    Instr { op, a, b, out, c }
}

/// The program part of `g` for `comp`.
fn part(g: &LearnerGenome, comp: Component) -> &Vec<Instr> {
    match comp {
        Component::Setup => &g.setup,
        Component::Predict => &g.predict,
        Component::Learn => &g.learn,
    }
}

fn part_mut(g: &mut LearnerGenome, comp: Component) -> &mut Vec<Instr> {
    match comp {
        Component::Setup => &mut g.setup,
        Component::Predict => &mut g.predict,
        Component::Learn => &mut g.learn,
    }
}

fn max_len(lim: &DslLimits, comp: Component) -> usize {
    match comp {
        Component::Setup => lim.max_setup,
        Component::Predict => lim.max_predict,
        Component::Learn => lim.max_learn,
    }
}

/// A uniformly chosen component that satisfies `ok`, or None.
fn pick_component(
    rng: &mut ChaCha8Rng,
    g: &LearnerGenome,
    ok: impl Fn(&LearnerGenome, Component) -> bool,
) -> Option<Component> {
    let eligible: Vec<Component> = COMPONENTS.into_iter().filter(|&c| ok(g, c)).collect();

    match eligible.len() {
        0 => None,
        n => Some(eligible[rng.random_range(0..n)]),
    }
}

/// Positions (component, index) of all instructions that satisfy `ok`.
fn positions(g: &LearnerGenome, ok: impl Fn(&Instr) -> bool) -> Vec<(Component, usize)> {
    COMPONENTS
        .into_iter()
        .flat_map(|comp| {
            part(g, comp)
                .iter()
                .enumerate()
                .filter(|(_, ins)| ok(ins))
                .map(move |(i, _)| (comp, i))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Draw the mutation kind for one attempt.
fn draw_kind(mode: VariationMode, rng: &mut ChaCha8Rng) -> Mutation {
    if mode == VariationMode::ConstantsOnly {
        return Mutation::Const;
    }

    let u: f64 = rng.random();

    if u < 0.1 {
        Mutation::Insert
    } else if u < 0.3 {
        Mutation::Remove
    } else if u < 0.4 {
        Mutation::Randomize
    } else if u < 0.7 {
        Mutation::Arg
    } else {
        Mutation::Const
    }
}

/// |x| capped at `limit` with its sign kept; a NaN limit leaves x unchanged.
fn cap(x: f32, limit: f32) -> f32 {
    if x > limit {
        limit
    } else if x < -limit {
        -limit
    } else {
        x
    }
}

/// A new constant for a Const mutation: multiply by U[0.5, 2.0) (Value: sign flip with
/// p = 0.1), capped at const_abs_max; a zero constant gets a fresh draw instead, since
/// multiplying would never move it; Index constants are redrawn.
fn mutated_const(rng: &mut ChaCha8Rng, c: f32, kind: ConstKind, lim: &DslLimits) -> f32 {
    match kind {
        ConstKind::None => c,
        ConstKind::Index => random_const(rng, kind),
        ConstKind::Value | ConstKind::Scale if c == 0.0 => random_const(rng, kind),

        ConstKind::Value | ConstKind::Scale => {
            let factor = 0.5 + 1.5 * rng.random::<f32>();
            let mut x = c * factor;

            if kind == ConstKind::Value && rng.random_bool(0.1) {
                x = -x;
            }

            cap(x, lim.const_abs_max)
        }
    }
}

/// A different register index in the same slot range.
fn other_operand(rng: &mut ChaCha8Rng, slot: Slot, current: u8) -> u8 {
    let n = slot_len(slot);

    if n < 2 || current as usize >= n {
        return random_operand(rng, slot);
    }

    let r = rng.random_range(0..n - 1) as u8;

    if r >= current { r + 1 } else { r }
}

/// Apply one mutation of `kind`; None when it does not apply to `g`.
fn apply(
    g: &LearnerGenome,
    kind: Mutation,
    rng: &mut ChaCha8Rng,
    lim: &DslLimits,
) -> Option<LearnerGenome> {
    let mut child = g.clone();

    match kind {
        Mutation::Insert => {
            let comp = pick_component(rng, g, |g, c| part(g, c).len() < max_len(lim, c))?;
            let code = part_mut(&mut child, comp);
            let at = rng.random_range(0..=code.len());

            code.insert(at, random_instr(rng, comp));
        }

        Mutation::Remove => {
            let comp = pick_component(rng, g, |g, c| !part(g, c).is_empty())?;
            let code = part_mut(&mut child, comp);
            let at = rng.random_range(0..code.len());

            code.remove(at);
        }

        Mutation::Randomize => {
            let comp = pick_component(rng, g, |g, c| !part(g, c).is_empty())?;

            for ins in part_mut(&mut child, comp).iter_mut() {
                *ins = random_instr(rng, comp);
            }
        }

        Mutation::Arg => {
            let all = positions(g, |_| true);

            if all.is_empty() {
                return None;
            }

            let (comp, i) = all[rng.random_range(0..all.len())];
            let ins = &mut part_mut(&mut child, comp)[i];

            change_operand(rng, ins);
        }

        Mutation::Const => {
            let with_const = positions(g, |ins| sig(ins.op).c != ConstKind::None);

            if with_const.is_empty() {
                return None;
            }

            let (comp, i) = with_const[rng.random_range(0..with_const.len())];
            let ins = &mut part_mut(&mut child, comp)[i];

            ins.c = mutated_const(rng, ins.c, sig(ins.op).c, lim);
        }
    }

    Some(child)
}

/// Change one used operand field of `ins` to a different register.
fn change_operand(rng: &mut ChaCha8Rng, ins: &mut Instr) {
    let s = sig(ins.op);
    let used: Vec<usize> = [s.a, s.b, s.out]
        .iter()
        .enumerate()
        .filter(|(_, slot)| **slot != Slot::None)
        .map(|(k, _)| k)
        .collect();

    if used.is_empty() {
        return;
    }

    match used[rng.random_range(0..used.len())] {
        0 => ins.a = other_operand(rng, s.a, ins.a),
        1 => ins.b = other_operand(rng, s.b, ins.b),
        _ => ins.out = other_operand(rng, s.out, ins.out),
    }
}

/// True when `child` is a valid, different result for parent `g`.
fn acceptable(child: &LearnerGenome, g: &LearnerGenome, lim: &DslLimits) -> bool {
    child != g && child.check(lim).is_ok()
}

/// Const fallback: fresh constant draws at every constant position, from a random start.
fn fallback_const(
    g: &LearnerGenome,
    rng: &mut ChaCha8Rng,
    lim: &DslLimits,
) -> Option<LearnerGenome> {
    let spots = positions(g, |ins| sig(ins.op).c != ConstKind::None);

    if spots.is_empty() {
        return None;
    }

    let start = rng.random_range(0..spots.len());

    for k in 0..spots.len() {
        let (comp, i) = spots[(start + k) % spots.len()];

        for _ in 0..FALLBACK_DRAWS {
            let mut child = g.clone();
            let ins = &mut part_mut(&mut child, comp)[i];

            ins.c = random_const(rng, sig(ins.op).c);

            if acceptable(&child, g, lim) {
                return Some(child);
            }
        }
    }

    None
}

/// Arg fallback: one changed operand at the first position, from a random start, that works.
fn fallback_arg(g: &LearnerGenome, rng: &mut ChaCha8Rng, lim: &DslLimits) -> Option<LearnerGenome> {
    let spots = positions(g, |_| true);

    if spots.is_empty() {
        return None;
    }

    let start = rng.random_range(0..spots.len());

    for k in 0..spots.len() {
        let (comp, i) = spots[(start + k) % spots.len()];
        let mut child = g.clone();

        change_operand(rng, &mut part_mut(&mut child, comp)[i]);

        if acceptable(&child, g, lim) {
            return Some(child);
        }
    }

    None
}

/// Insert fallback (an empty program has nothing else to change).
fn fallback_insert(
    g: &LearnerGenome,
    rng: &mut ChaCha8Rng,
    lim: &DslLimits,
) -> Option<LearnerGenome> {
    for _ in 0..FALLBACK_INSERTS {
        if let Some(child) = apply(g, Mutation::Insert, rng, lim)
            && acceptable(&child, g, lim)
        {
            return Some(child);
        }
    }

    None
}

/// One child that differs from `g` and passes `check(lim)`.
/// Full: P(Insert)=0.1, P(Remove)=0.2, P(Randomize component, keep length)=0.1,
///       P(Arg)=0.3, P(Const)=0.3. Const: multiply by U[0.5, 2.0], flip sign with p = 0.1
///       (Value only); Index constants are redrawn. Up to 8 retries for a valid, different
///       child; then fall back to Const, then Arg.
/// ConstantsOnly: only Const. RandomSearch: same as Full (selection is skipped elsewhere).
///
/// Details: every attempt draws a new kind; Insert/Remove/Randomize pick a component
/// uniformly among those where the mutation applies; Arg/Const pick an instruction
/// uniformly among all that qualify. A zero Value/Scale constant is redrawn instead of
/// multiplied, and results are capped at const_abs_max. After Const and Arg, Full and
/// RandomSearch also try Insert (the only change an empty program allows). ConstantsOnly
/// never falls back to a structural change. When no valid different child exists (a parent
/// without constants under ConstantsOnly, or an invalid parent), an unchanged copy is
/// returned with Mutation::Const.
pub fn mutate(
    g: &LearnerGenome,
    mode: VariationMode,
    rng: &mut ChaCha8Rng,
    lim: &DslLimits,
) -> (LearnerGenome, Mutation) {
    for _ in 0..=RETRIES {
        let kind = draw_kind(mode, rng);

        if let Some(child) = apply(g, kind, rng, lim)
            && acceptable(&child, g, lim)
        {
            return (child, kind);
        }
    }

    if let Some(child) = fallback_const(g, rng, lim) {
        return (child, Mutation::Const);
    }

    if mode != VariationMode::ConstantsOnly {
        if let Some(child) = fallback_arg(g, rng, lim) {
            return (child, Mutation::Arg);
        }

        if let Some(child) = fallback_insert(g, rng, lim) {
            return (child, Mutation::Insert);
        }
    }

    (g.clone(), Mutation::Const)
}

/// Stage-2 hook (same role as variation::VariantGenerator). The result MUST pass check().
pub trait ProgramProposer: Send + Sync {
    fn name(&self) -> &str;

    fn propose(
        &self,
        parents: &[&LearnerGenome],
        rng: &mut ChaCha8Rng,
        lim: &DslLimits,
    ) -> (LearnerGenome, Mutation);
}

/// Default proposer wrapping `mutate`.
pub struct RandomProgramMutator {
    pub mode: VariationMode,
}

impl ProgramProposer for RandomProgramMutator {
    fn name(&self) -> &str {
        "random-program-mutator-v1"
    }

    /// Mutates the first parent; with no parent, mutates the reference NN seed.
    fn propose(
        &self,
        parents: &[&LearnerGenome],
        rng: &mut ChaCha8Rng,
        lim: &DslLimits,
    ) -> (LearnerGenome, Mutation) {
        match parents.first() {
            Some(parent) => mutate(parent, self.mode, rng, lim),
            None => mutate(&reference_nn(), self.mode, rng, lim),
        }
    }
}

/// Initial population member i (0-based) of a run: seed[i % 4] mutated min(i / 4, 3) times
/// with derive_seed(master, "init/<i>").
pub fn initial_member(
    i: usize,
    master: u64,
    mode: VariationMode,
    lim: &DslLimits,
) -> LearnerGenome {
    let mut seeds = seed_programs();
    let mut g = seeds.swap_remove(i % seeds.len()).1;
    let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(master, &format!("init/{i}")));

    for _ in 0..(i / 4).min(3) {
        g = mutate(&g, mode, &mut rng, lim).0;
    }

    g
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::learner::dsl::ConstKind;

    fn all_programs(lim: &DslLimits) -> Vec<LearnerGenome> {
        let mut out: Vec<LearnerGenome> = seed_programs().into_iter().map(|(_, g)| g).collect();

        out.extend((0..64).map(|i| initial_member(i, 3, VariationMode::Full, lim)));

        out
    }

    #[test]
    fn random_instr_is_always_valid() {
        let lim = DslLimits::default();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut ops = std::collections::HashSet::new();

        for k in 0..20_000 {
            let comp = COMPONENTS[k % 3];
            let ins = random_instr(&mut rng, comp);
            let g = LearnerGenome {
                setup: vec![ins],
                ..LearnerGenome::default()
            };

            assert!(g.validate(&lim).is_ok(), "{ins:?}");

            ops.insert(ins.op);

            match sig(ins.op).c {
                ConstKind::Value => assert!((1e-3..=1.0).contains(&ins.c.abs())),
                ConstKind::Scale => assert!((1e-3..=1.0).contains(&ins.c)),
                ConstKind::Index => assert!(ins.c == ins.c.trunc() && ins.c <= 255.0),
                ConstKind::None => assert_eq!(ins.c.to_bits(), 0),
            }
        }

        assert_eq!(ops.len(), ALL_OPS.len());
    }

    /// T-MUT-1: 10 000 mutations are always valid and always differ from the parent.
    #[test]
    fn t_mut_1_children_valid_and_different() {
        let lim = DslLimits::default();
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut g = seed_programs()[1].1.clone();
        let mut kinds: HashMap<Mutation, usize> = HashMap::new();

        for k in 0..10_000 {
            let mode = [VariationMode::Full, VariationMode::RandomSearch][k % 2];
            let (child, kind) = mutate(&g, mode, &mut rng, &lim);

            assert!(child.check(&lim).is_ok(), "invalid child after {kind:?}");
            assert_ne!(child, g, "unchanged child after {kind:?}");

            *kinds.entry(kind).or_default() += 1;

            // Random walk, restarted now and then so programs of every size appear.
            g = if k % 500 == 499 {
                seed_programs()[k % 4].1.clone()
            } else {
                child
            };
        }

        assert_eq!(kinds.len(), 5, "{kinds:?}");
    }

    /// T-MUT-2: ConstantsOnly changes only `c` (same_structure).
    #[test]
    fn t_mut_2_constants_only_keeps_structure() {
        let lim = DslLimits::default();
        let mut rng = ChaCha8Rng::seed_from_u64(3);

        for (name, seed) in seed_programs() {
            let mut g = seed.clone();

            for _ in 0..1000 {
                let (child, kind) = mutate(&g, VariationMode::ConstantsOnly, &mut rng, &lim);

                assert_eq!(kind, Mutation::Const);
                assert!(child.same_structure(&seed), "{name}");
                assert!(child.check(&lim).is_ok());
                assert_ne!(child, g);

                g = child;
            }
        }

        for i in 0..32 {
            let member = initial_member(i, 9, VariationMode::ConstantsOnly, &lim);

            assert!(member.same_structure(&seed_programs()[i % 4].1));
        }

        // No constants at all: an unchanged copy, never a structural change.
        let bare = LearnerGenome {
            predict: vec![Instr {
                op: crate::learner::types::OpCode::MatVec,
                a: 0,
                b: 0,
                out: 1,
                c: 0.0,
            }],
            ..LearnerGenome::default()
        };

        let (child, kind) = mutate(&bare, VariationMode::ConstantsOnly, &mut rng, &lim);

        assert_eq!(child, bare);
        assert_eq!(kind, Mutation::Const);
    }

    /// T-MUT-3: Remove happens about twice as often as Insert (chi-square, p > 0.001).
    #[test]
    fn t_mut_3_remove_twice_insert() {
        let lim = DslLimits::default();
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let programs = all_programs(&lim);
        let mut counts: HashMap<Mutation, f64> = HashMap::new();
        let n = 20_000;

        for k in 0..n {
            let (_, kind) = mutate(
                &programs[k % programs.len()],
                VariationMode::Full,
                &mut rng,
                &lim,
            );

            *counts.entry(kind).or_default() += 1.0;
        }

        let insert = counts.get(&Mutation::Insert).copied().unwrap_or(0.0);
        let remove = counts.get(&Mutation::Remove).copied().unwrap_or(0.0);
        let total = insert + remove;

        // H0: P(Remove | Insert or Remove) = 2/3. Chi-square with 1 dof; 10.83 is p = 0.001.
        let want_insert = total / 3.0;
        let want_remove = 2.0 * total / 3.0;
        let chi2 = (insert - want_insert).powi(2) / want_insert
            + (remove - want_remove).powi(2) / want_remove;

        assert!(
            chi2 < 10.83,
            "insert {insert}, remove {remove}, chi2 {chi2}"
        );

        // The other kinds come out near their nominal shares too.
        for (kind, p) in [
            (Mutation::Randomize, 0.1),
            (Mutation::Arg, 0.3),
            (Mutation::Const, 0.3),
        ] {
            let share = counts.get(&kind).copied().unwrap_or(0.0) / n as f64;

            assert!((share - p).abs() < 0.05, "{kind:?} share {share}");
        }
    }

    #[test]
    fn const_mutation_rules() {
        let lim = DslLimits::default();
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let mut flips = 0;

        for _ in 0..10_000 {
            let x = mutated_const(&mut rng, 0.2, ConstKind::Value, &lim);
            let ratio = x / 0.2;

            if ratio < 0.0 {
                flips += 1;
            }

            assert!((0.5..=2.0).contains(&ratio.abs()), "ratio {ratio}");

            let s = mutated_const(&mut rng, 0.2, ConstKind::Scale, &lim);

            assert!((0.1..=0.4).contains(&s));
        }

        // Sign flips with p = 0.1 (binomial sd = 30).
        assert!((850..=1150).contains(&flips), "flips {flips}");

        // Capped at the limit; zero gets a fresh value; Index is redrawn.
        assert!(mutated_const(&mut rng, 999.0, ConstKind::Value, &lim).abs() <= 1000.0);
        assert_ne!(mutated_const(&mut rng, 0.0, ConstKind::Value, &lim), 0.0);

        let index = mutated_const(&mut rng, 7.0, ConstKind::Index, &lim);

        assert!(index == index.trunc() && (0.0..=255.0).contains(&index));

        // A NaN limit never panics.
        let odd = DslLimits {
            const_abs_max: f32::NAN,
            ..lim
        };

        assert!(mutated_const(&mut rng, 1.0, ConstKind::Value, &odd).is_finite());
    }

    #[test]
    fn initial_members_follow_the_rule() {
        let lim = DslLimits::default();
        let seeds = seed_programs();

        for (i, (_, seed)) in seeds.iter().enumerate() {
            assert_eq!(&initial_member(i, 1, VariationMode::Full, &lim), seed);
        }

        for i in 0..64 {
            let a = initial_member(i, 1, VariationMode::Full, &lim);
            let b = initial_member(i, 1, VariationMode::Full, &lim);
            let c = initial_member(i, 2, VariationMode::Full, &lim);

            assert_eq!(a, b, "deterministic");
            assert!(a.check(&lim).is_ok());

            if i >= 4 {
                assert_ne!(a, seeds[i % 4].1);
                assert_ne!(a, c, "master seed matters at member {i}");
            }
        }
    }

    #[test]
    fn proposer_wraps_mutate() {
        let lim = DslLimits::default();
        let p = RandomProgramMutator {
            mode: VariationMode::Full,
        };
        let parent = seed_programs()[0].1.clone();

        assert_eq!(p.name(), "random-program-mutator-v1");

        let mut r1 = ChaCha8Rng::seed_from_u64(8);
        let mut r2 = ChaCha8Rng::seed_from_u64(8);

        assert_eq!(
            p.propose(&[&parent], &mut r1, &lim),
            mutate(&parent, VariationMode::Full, &mut r2, &lim)
        );

        let (orphan, _) = p.propose(&[], &mut r1, &lim);

        assert!(orphan.check(&lim).is_ok());

        // Trait object use, as the orchestrator will hold it.
        let boxed: Box<dyn ProgramProposer> = Box::new(p);

        assert!(
            boxed
                .propose(&[&parent, &parent], &mut r1, &lim)
                .0
                .check(&lim)
                .is_ok()
        );
    }

    #[test]
    fn full_components_and_tiny_limits_do_not_break_mutate() {
        let lim = DslLimits {
            max_setup: 1,
            max_predict: 1,
            max_learn: 4,
            const_abs_max: 1.0e3,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let mut g = seed_programs()[0].1.clone();

        for _ in 0..2000 {
            let (child, _) = mutate(&g, VariationMode::Full, &mut rng, &lim);

            assert!(child.check(&lim).is_ok());
            assert_ne!(child, g);

            g = child;
        }

        // An empty program can only grow.
        let (child, kind) = mutate(
            &LearnerGenome::default(),
            VariationMode::Full,
            &mut rng,
            &lim,
        );

        assert_eq!(kind, Mutation::Insert);
        assert!(child.check(&lim).is_ok());
    }
}
