//! DSL core: opcode signatures, costs, validation, ids and dead-code elimination.
//!
//! Owner: WP1. Contract section 5.1. Pure: no I/O.
#![forbid(unsafe_code)]

use crate::learner::types::{
    CHECKPOINT_DIVISORS, COST_CAP_RATIO, Component, DslLimits, Instr, LearnerGenome, N_M, N_S, N_V,
    OpCode, SUPPORTED_F,
};
use crate::util::{BrutError, Result, sha256_hex};

/// Operand slot type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    None,
    S,
    V,
    M,
}

/// Meaning of the constant field `c`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstKind {
    None,
    /// Any finite value with |c| <= const_abs_max.
    Value,
    /// Scale of a random op; |c| used; |c| <= const_abs_max.
    Scale,
    /// Integer index 0..=255 stored in c (c == c.trunc()); effective index = c as usize % F.
    Index,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpSig {
    pub a: Slot,
    pub b: Slot,
    pub out: Slot,
    pub c: ConstKind,
}

/// All 46 opcodes in declaration order.
pub const ALL_OPS: [OpCode; 46] = [
    OpCode::SConst,
    OpCode::SAdd,
    OpCode::SSub,
    OpCode::SMul,
    OpCode::SDiv,
    OpCode::SMin,
    OpCode::SMax,
    OpCode::SAbs,
    OpCode::SRecip,
    OpCode::SHeaviside,
    OpCode::SSin,
    OpCode::SCos,
    OpCode::SExp,
    OpCode::SLog,
    OpCode::SUniform,
    OpCode::VBroadcast,
    OpCode::VAdd,
    OpCode::VSub,
    OpCode::VMul,
    OpCode::VDiv,
    OpCode::VMin,
    OpCode::VMax,
    OpCode::VMax0,
    OpCode::VScale,
    OpCode::VAbs,
    OpCode::VHeaviside,
    OpCode::VSet,
    OpCode::VGaussian,
    OpCode::VUniform,
    OpCode::MatVec,
    OpCode::MatTVec,
    OpCode::MRowMean,
    OpCode::VDot,
    OpCode::VNorm,
    OpCode::VMean,
    OpCode::VStd,
    OpCode::VGet,
    OpCode::MOuter,
    OpCode::MAdd,
    OpCode::MSub,
    OpCode::MMul,
    OpCode::MScale,
    OpCode::MTranspose,
    OpCode::MGaussian,
    OpCode::MUniform,
    OpCode::MNorm,
];

/// Signature of an opcode (table in contract section 5.1).
pub fn sig(op: OpCode) -> OpSig {
    use ConstKind as C;
    use OpCode::*;
    use Slot::{M, S, V};

    let no = Slot::None;

    let (a, b, out, c) = match op {
        SConst => (no, no, S, C::Value),
        SAdd | SSub | SMul | SDiv | SMin | SMax => (S, S, S, C::None),
        SAbs | SRecip | SHeaviside | SSin | SCos | SExp | SLog => (S, no, S, C::None),
        SUniform => (no, no, S, C::Scale),
        VBroadcast => (S, no, V, C::None),
        VAdd | VSub | VMul | VDiv | VMin | VMax => (V, V, V, C::None),
        VMax0 | VAbs | VHeaviside => (V, no, V, C::None),
        VScale => (S, V, V, C::None),
        VSet => (S, no, V, C::Index),
        VGaussian | VUniform => (no, no, V, C::Scale),
        MatVec | MatTVec => (M, V, V, C::None),
        MRowMean => (M, no, V, C::None),
        VDot => (V, V, S, C::None),
        VNorm | VMean | VStd => (V, no, S, C::None),
        VGet => (V, no, S, C::Index),
        MOuter => (V, V, M, C::None),
        MAdd | MSub | MMul => (M, M, M, C::None),
        MScale => (S, M, M, C::None),
        MTranspose => (M, no, M, C::None),
        MGaussian | MUniform => (no, no, M, C::Scale),
        MNorm => (M, no, S, C::None),
    };

    OpSig { a, b, out, c }
}

/// FLOP-equivalent cost of one execution at width f (table in contract section 5.1).
///
/// Errata K1 (re-cost, learner-v3): the adversarial-FLOP bench measured these ops slower
/// per counted FLOP than MatVec, so their costs are raised to match their real time:
/// SSin/SCos x9 (10 -> 90), SExp/SLog x5 (10 -> 50), VGaussian x10 per element
/// (10 f -> 100 f) and MGaussian x10 (10 f^2 -> 100 f^2). SUniform, VUniform and MUniform
/// keep their old costs (the bench did not flag them). MatVec stays f^2 * 2.
pub fn cost(op: OpCode, f: u64) -> u64 {
    use OpCode::*;

    let f2 = f.saturating_mul(f);

    match op {
        SConst | SAdd | SSub | SMul | SDiv | SMin | SMax => 1,
        SAbs | SRecip | SHeaviside => 1,
        SSin | SCos => 90,
        SExp | SLog => 50,
        SUniform => 10,
        VBroadcast | VAdd | VSub | VMul | VDiv | VMin | VMax => f,
        VMax0 | VScale | VAbs | VHeaviside | VMean => f,
        VSet | VGet => 1,
        VGaussian => f.saturating_mul(100),
        VUniform => f.saturating_mul(10),
        VDot | VNorm => f.saturating_mul(2),
        VStd => f.saturating_mul(3),
        MatVec | MatTVec | MNorm => f2.saturating_mul(2),
        MRowMean | MOuter | MAdd | MSub | MMul | MScale | MTranspose => f2,
        MGaussian => f2.saturating_mul(100),
        MUniform => f2.saturating_mul(10),
    }
}

/// Per-step cost (predict + learn) of the reference 2-layer NN seed at width f:
/// 10 f^2 + 5 f. Defines the static cost cap.
pub fn reference_step_cost(f: usize) -> u64 {
    let f = f as u64;

    f.saturating_mul(f)
        .saturating_mul(10)
        .saturating_add(f.saturating_mul(5))
}

/// True for the ops that draw from the memory's random stream.
pub(crate) fn is_random(op: OpCode) -> bool {
    matches!(
        op,
        OpCode::SUniform
            | OpCode::VGaussian
            | OpCode::VUniform
            | OpCode::MGaussian
            | OpCode::MUniform
    )
}

/// Number of registers behind a slot (0 for an unused slot).
pub(crate) fn slot_len(slot: Slot) -> usize {
    match slot {
        Slot::None => 0,
        Slot::S => N_S,
        Slot::V => N_V,
        Slot::M => N_M,
    }
}

/// Lower-case name of a program part.
pub(crate) fn component_name(comp: Component) -> &'static str {
    match comp {
        Component::Setup => "setup",
        Component::Predict => "predict",
        Component::Learn => "learn",
    }
}

/// Summed op cost of one straight-line program at width f.
fn code_cost(code: &[Instr], f: usize) -> u64 {
    let f = f as u64;

    code.iter()
        .fold(0u64, |acc, ins| acc.saturating_add(cost(ins.op, f)))
}

/// Why one operand field is invalid, or None.
fn operand_error(name: &str, value: u8, slot: Slot) -> Option<String> {
    let len = slot_len(slot);

    if slot == Slot::None && value != 0 {
        return Some(format!("unused operand {name} must be 0, got {value}"));
    }

    if slot != Slot::None && value as usize >= len {
        return Some(format!(
            "operand {name} = {value} out of range for {slot:?} (< {len})"
        ));
    }

    None
}

/// Why the constant field is invalid, or None.
fn const_error(c: f32, kind: ConstKind, lim: &DslLimits) -> Option<String> {
    match kind {
        ConstKind::None => {
            // Only +0.0, so that every program has exactly one canonical JSON form.
            if c.to_bits() != 0 {
                return Some("unused constant must be 0.0".to_string());
            }
        }

        ConstKind::Value | ConstKind::Scale => {
            if !c.is_finite() {
                return Some("constant is not finite".to_string());
            }

            if c.abs() > lim.const_abs_max {
                return Some(format!(
                    "|constant| > const_abs_max = {}",
                    lim.const_abs_max
                ));
            }
        }

        ConstKind::Index => {
            let integral = c.is_finite() && c == c.trunc();
            let in_range = !c.is_sign_negative() && c <= 255.0;

            if !integral || !in_range {
                return Some("index constant must be an integer in 0..=255".to_string());
            }
        }
    }

    None
}

/// Why one instruction is structurally invalid, or None.
fn instr_error(ins: &Instr, lim: &DslLimits) -> Option<String> {
    let s = sig(ins.op);

    operand_error("a", ins.a, s.a)
        .or_else(|| operand_error("b", ins.b, s.b))
        .or_else(|| operand_error("out", ins.out, s.out))
        .or_else(|| const_error(ins.c, s.c, lim))
}

// ---------------------------------------------------------------------------------------------
// Liveness (dead-code elimination)
// ---------------------------------------------------------------------------------------------

/// Register set for liveness: bits 0..10 = s, 10..18 = v, 18..22 = m, bit 22 = random stream.
type RegSet = u32;

const V_BASE: u32 = N_S as u32;
const M_BASE: u32 = (N_S + N_V) as u32;
const RNG_BIT: RegSet = 1 << (N_S + N_V + N_M);

/// Bit of one register; indices are clamped exactly like the interpreter clamps them.
fn reg_bit(slot: Slot, index: u8) -> RegSet {
    let i = index as u32;

    match slot {
        Slot::None => 0,
        Slot::S => 1 << i.min(N_S as u32 - 1),
        Slot::V => 1 << (V_BASE + i.min(N_V as u32 - 1)),
        Slot::M => 1 << (M_BASE + i.min(N_M as u32 - 1)),
    }
}

/// s0, v0, v1 and v2 as register bits.
const S0: RegSet = 1;
const V0: RegSet = 1 << V_BASE;
const V1: RegSet = 1 << (V_BASE + 1);
const V2: RegSet = 1 << (V_BASE + 2);

/// Registers the protocol overwrites before every Predict (set_input, clear_label).
const KILLED_BEFORE_PREDICT: RegSet = V0 | V2 | S0;

/// Registers the protocol overwrites before every Learn (set_label).
const KILLED_BEFORE_LEARN: RegSet = V2 | S0;

/// Upper bound on fixed-point rounds; the lattice has 23 bits, so 64 is never reached.
const MAX_LIVENESS_ROUNDS: usize = 64;

/// Uses, definitions and full overwrites of one instruction.
struct Effect {
    uses: RegSet,
    defs: RegSet,
    kills: RegSet,
}

fn effect(ins: &Instr) -> Effect {
    let s = sig(ins.op);
    let out = reg_bit(s.out, ins.out);

    let mut uses = reg_bit(s.a, ins.a) | reg_bit(s.b, ins.b);
    let mut defs = out;
    let mut kills = out;

    // VSet writes one element only, so the rest of v[out] flows through.
    if ins.op == OpCode::VSet {
        uses |= out;
        kills = 0;
    }

    // Random ops read and advance the shared random stream.
    if is_random(ins.op) {
        uses |= RNG_BIT;
        defs |= RNG_BIT;
        kills |= RNG_BIT;
    }

    Effect { uses, defs, kills }
}

/// Backward strong-liveness pass. Returns the live-in set; fills `keep` when given.
fn backward(code: &[Instr], live_out: RegSet, mut keep: Option<&mut Vec<bool>>) -> RegSet {
    let mut live = live_out;

    if let Some(k) = keep.as_deref_mut() {
        k.clear();
        k.resize(code.len(), false);
    }

    for (pos, ins) in code.iter().enumerate().rev() {
        let e = effect(ins);
        let needed = e.defs & live != 0;

        if needed {
            live = (live & !e.kills) | e.uses;
        }

        if let Some(k) = keep.as_deref_mut() {
            k[pos] = needed;
        }
    }

    live
}

/// Live-out sets of (Setup, Predict, Learn) at the least fixed point.
///
/// Control flow: Setup -> Predict; Predict -> Learn (training);
/// Learn -> Predict (next step or a validation clone); Predict -> Predict (validation).
fn live_outs(g: &LearnerGenome) -> (RegSet, RegSet, RegSet) {
    let mut in_predict: RegSet = 0;
    let mut in_learn: RegSet = 0;

    for _ in 0..MAX_LIVENESS_ROUNDS {
        let out_learn = in_predict & !KILLED_BEFORE_PREDICT;
        let new_in_learn = backward(&g.learn, out_learn, None);

        let out_predict =
            V1 | (new_in_learn & !KILLED_BEFORE_LEARN) | (in_predict & !KILLED_BEFORE_PREDICT);
        let new_in_predict = backward(&g.predict, out_predict, None);

        let stable = new_in_learn == in_learn && new_in_predict == in_predict;

        in_learn = new_in_learn;
        in_predict = new_in_predict;

        if stable {
            break;
        }
    }

    let out_setup = in_predict & !KILLED_BEFORE_PREDICT;
    let out_learn = in_predict & !KILLED_BEFORE_PREDICT;
    let out_predict =
        V1 | (in_learn & !KILLED_BEFORE_LEARN) | (in_predict & !KILLED_BEFORE_PREDICT);

    (out_setup, out_predict, out_learn)
}

/// Instructions of `code` whose keep flag is set.
fn kept(code: &[Instr], keep: &[bool]) -> Vec<Instr> {
    code.iter()
        .zip(keep)
        .filter(|(_, k)| **k)
        .map(|(ins, _)| *ins)
        .collect()
}

/// One round of dead-code elimination.
fn simplify_once(g: &LearnerGenome) -> LearnerGenome {
    let (out_setup, out_predict, out_learn) = live_outs(g);
    let mut keep = Vec::new();

    backward(&g.setup, out_setup, Some(&mut keep));
    let setup = kept(&g.setup, &keep);

    backward(&g.predict, out_predict, Some(&mut keep));
    let predict = kept(&g.predict, &keep);

    backward(&g.learn, out_learn, Some(&mut keep));
    let learn = kept(&g.learn, &keep);

    LearnerGenome {
        setup,
        predict,
        learn,
    }
}

// ---------------------------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------------------------

/// Register name for a slot, e.g. "v3".
fn reg_name(slot: Slot, index: u8) -> String {
    match slot {
        Slot::None => String::new(),
        Slot::S => format!("s{index}"),
        Slot::V => format!("v{index}"),
        Slot::M => format!("m{index}"),
    }
}

/// One instruction as text, e.g. "MatVec v1 <- m0 . v0".
fn render_instr(ins: &Instr) -> String {
    use OpCode::*;

    let s = sig(ins.op);
    let a = reg_name(s.a, ins.a);
    let b = reg_name(s.b, ins.b);
    let out = reg_name(s.out, ins.out);
    let c = ins.c;
    let scale = c.abs();
    let index = c as usize;

    let rhs = match ins.op {
        SConst => format!("{c}"),
        SAdd | VAdd | MAdd => format!("{a} + {b}"),
        SSub | VSub | MSub => format!("{a} - {b}"),
        SMul | VMul | MMul | VScale | MScale => format!("{a} * {b}"),
        SDiv | VDiv => format!("{a} / {b}"),
        SMin | VMin => format!("min({a}, {b})"),
        SMax | VMax => format!("max({a}, {b})"),
        SAbs | VAbs => format!("abs({a})"),
        SRecip => format!("1 / {a}"),
        SHeaviside | VHeaviside => format!("heaviside({a})"),
        SSin => format!("sin({a})"),
        SCos => format!("cos({a})"),
        SExp => format!("exp({a})"),
        SLog => format!("log({a})"),
        SUniform | VUniform | MUniform => format!("uniform(-{scale}, {scale})"),
        VGaussian | MGaussian => format!("gaussian(0, {scale}^2)"),
        VBroadcast => format!("broadcast({a})"),
        VMax0 => format!("max({a}, 0)"),
        VSet => a,
        MatVec | VDot => format!("{a} . {b}"),
        MatTVec => format!("{a}^T . {b}"),
        MRowMean => format!("rowmean({a})"),
        VNorm | MNorm => format!("norm({a})"),
        VMean => format!("mean({a})"),
        VStd => format!("std({a})"),
        VGet => format!("{a}[{index}]"),
        MOuter => format!("{a} (x) {b}"),
        MTranspose => format!("{a}^T"),
    };

    let lhs = if ins.op == VSet {
        format!("{out}[{index}]")
    } else {
        out
    };

    format!("{:?} {lhs} <- {rhs}", ins.op)
}

// ---------------------------------------------------------------------------------------------
// LearnerGenome API
// ---------------------------------------------------------------------------------------------

impl LearnerGenome {
    /// The three parts with their component tags, in execution order.
    fn parts(&self) -> [(Component, &Vec<Instr>); 3] {
        [
            (Component::Setup, &self.setup),
            (Component::Predict, &self.predict),
            (Component::Learn, &self.learn),
        ]
    }

    /// Structural check: lengths <= limits, every operand index valid for its slot
    /// (S < N_S, V < N_V, M < N_M), unused fields 0, constants finite and within limits,
    /// Index constants integral in 0..=255. Does not check cost.
    ///
    /// Unused constants must be exactly +0.0 (not -0.0), so the canonical JSON is unique.
    pub fn validate(&self, lim: &DslLimits) -> Result<()> {
        for (comp, code) in self.parts() {
            let name = component_name(comp);

            let max = match comp {
                Component::Setup => lim.max_setup,
                Component::Predict => lim.max_predict,
                Component::Learn => lim.max_learn,
            };

            if code.len() > max {
                return Err(BrutError::Invalid(format!(
                    "{name} has {} instructions, max {max}",
                    code.len()
                )));
            }

            for (pos, ins) in code.iter().enumerate() {
                if let Some(why) = instr_error(ins, lim) {
                    return Err(BrutError::Invalid(format!(
                        "{name}[{pos}] {:?}: {why}",
                        ins.op
                    )));
                }
            }
        }

        Ok(())
    }

    /// Static cost cap: step_cost(f) <= COST_CAP_RATIO * reference_step_cost(f)
    /// for every f in SUPPORTED_F. Err(Budget) otherwise. Called before any execution.
    pub fn check_cost(&self) -> Result<()> {
        for f in SUPPORTED_F {
            let step = self.step_cost(f);
            let cap = COST_CAP_RATIO.saturating_mul(reference_step_cost(f));

            if step > cap {
                return Err(BrutError::Budget(format!(
                    "step cost {step} at F = {f} exceeds the cap {cap}"
                )));
            }
        }

        Ok(())
    }

    /// validate + check_cost.
    pub fn check(&self, lim: &DslLimits) -> Result<()> {
        self.validate(lim)?;

        self.check_cost()
    }

    /// First 16 hex chars of SHA-256 over canonical_json() (same scheme as Genome::id).
    pub fn id(&self) -> String {
        let mut hex = sha256_hex(self.canonical_json().as_bytes());

        hex.truncate(16);

        hex
    }

    /// serde_json::to_string of self (field order fixed by the struct).
    pub fn canonical_json(&self) -> String {
        // Serializing this plain struct cannot fail; the Debug fallback only keeps it panic-free.
        match serde_json::to_string(self) {
            Ok(s) => s,
            Err(_) => format!("{self:?}"),
        }
    }

    /// Dead-code elimination by backward liveness over Setup -> (Predict -> Learn)*,
    /// iterated to a fixed point. Live outputs: v1 after Predict and every register read
    /// later. MUST preserve predictions bit-for-bit (property test).
    ///
    /// The protocol's own writes are modelled: v0, v2 and s0 are overwritten before every
    /// Predict, v2 and s0 before every Learn. Validation runs Predict repeatedly on a clone,
    /// so Predict also flows into Predict. Random ops share one stream, which is treated as a
    /// register: a random op stays when any later random op stays.
    pub fn simplify(&self) -> LearnerGenome {
        let mut current = simplify_once(self);

        for _ in 0..MAX_LIVENESS_ROUNDS {
            let next = simplify_once(&current);

            if next == current {
                break;
            }

            current = next;
        }

        current
    }

    /// Instruction count of simplify().
    pub fn effective_len(&self) -> usize {
        let s = self.simplify();

        s.setup.len() + s.predict.len() + s.learn.len()
    }

    pub fn setup_cost(&self, f: usize) -> u64 {
        code_cost(&self.setup, f)
    }

    pub fn predict_cost(&self, f: usize) -> u64 {
        code_cost(&self.predict, f)
    }

    pub fn learn_cost(&self, f: usize) -> u64 {
        code_cost(&self.learn, f)
    }

    /// predict_cost + learn_cost.
    pub fn step_cost(&self, f: usize) -> u64 {
        self.predict_cost(f).saturating_add(self.learn_cost(f))
    }

    /// setup + steps * step + 3 * n_valid * predict.
    pub fn task_cost(&self, f: usize, steps: usize, n_valid: usize) -> u64 {
        let checkpoints = CHECKPOINT_DIVISORS.len() as u64;
        let training = (steps as u64).saturating_mul(self.step_cost(f));
        let validation = checkpoints
            .saturating_mul(n_valid as u64)
            .saturating_mul(self.predict_cost(f));

        self.setup_cost(f)
            .saturating_add(training)
            .saturating_add(validation)
    }

    /// Human-readable listing, one instruction per line ("MatVec v1 <- m0 . v0").
    /// Each part starts with a header line ("setup:"); Index constants are shown raw
    /// (the interpreter uses them modulo F).
    pub fn render(&self) -> String {
        let mut text = String::new();

        for (comp, code) in self.parts() {
            text.push_str(component_name(comp));
            text.push_str(":\n");

            for ins in code {
                text.push_str("  ");
                text.push_str(&render_instr(ins));
                text.push('\n');
            }
        }

        text
    }

    /// True when every instruction equals `other`'s except possibly `c`
    /// (used to accept constants-only variants of registered seeds).
    pub fn same_structure(&self, other: &LearnerGenome) -> bool {
        let same = |x: &[Instr], y: &[Instr]| {
            x.len() == y.len()
                && x.iter()
                    .zip(y)
                    .all(|(p, q)| p.op == q.op && p.a == q.a && p.b == q.b && p.out == q.out)
        };

        same(&self.setup, &other.setup)
            && same(&self.predict, &other.predict)
            && same(&self.learn, &other.learn)
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::learner::interp::testkit;
    use crate::learner::mutate::{mutate, random_instr};
    use crate::learner::seeds::seed_programs;
    use crate::learner::types::{LearnKind, VariationMode};

    fn ins(op: OpCode, a: u8, b: u8, out: u8, c: f32) -> Instr {
        Instr { op, a, b, out, c }
    }

    /// Independent transcription of the op table: signature text and cost as [1, F, F^2].
    fn table() -> Vec<(OpCode, &'static str, [u64; 3])> {
        use OpCode::*;

        vec![
            (SConst, "--S value", [1, 0, 0]),
            (SAdd, "SSS none", [1, 0, 0]),
            (SSub, "SSS none", [1, 0, 0]),
            (SMul, "SSS none", [1, 0, 0]),
            (SDiv, "SSS none", [1, 0, 0]),
            (SMin, "SSS none", [1, 0, 0]),
            (SMax, "SSS none", [1, 0, 0]),
            (SAbs, "S-S none", [1, 0, 0]),
            (SRecip, "S-S none", [1, 0, 0]),
            (SHeaviside, "S-S none", [1, 0, 0]),
            // Errata K1: SSin/SCos x9, SExp/SLog x5; SUniform unchanged.
            (SSin, "S-S none", [90, 0, 0]),
            (SCos, "S-S none", [90, 0, 0]),
            (SExp, "S-S none", [50, 0, 0]),
            (SLog, "S-S none", [50, 0, 0]),
            (SUniform, "--S scale", [10, 0, 0]),
            (VBroadcast, "S-V none", [0, 1, 0]),
            (VAdd, "VVV none", [0, 1, 0]),
            (VSub, "VVV none", [0, 1, 0]),
            (VMul, "VVV none", [0, 1, 0]),
            (VDiv, "VVV none", [0, 1, 0]),
            (VMin, "VVV none", [0, 1, 0]),
            (VMax, "VVV none", [0, 1, 0]),
            (VMax0, "V-V none", [0, 1, 0]),
            (VScale, "SVV none", [0, 1, 0]),
            (VAbs, "V-V none", [0, 1, 0]),
            (VHeaviside, "V-V none", [0, 1, 0]),
            (VSet, "S-V index", [1, 0, 0]),
            // Errata K1: VGaussian x10 per element; VUniform unchanged.
            (VGaussian, "--V scale", [0, 100, 0]),
            (VUniform, "--V scale", [0, 10, 0]),
            (MatVec, "MVV none", [0, 0, 2]),
            (MatTVec, "MVV none", [0, 0, 2]),
            (MRowMean, "M-V none", [0, 0, 1]),
            (VDot, "VVS none", [0, 2, 0]),
            (VNorm, "V-S none", [0, 2, 0]),
            (VMean, "V-S none", [0, 1, 0]),
            (VStd, "V-S none", [0, 3, 0]),
            (VGet, "V-S index", [1, 0, 0]),
            (MOuter, "VVM none", [0, 0, 1]),
            (MAdd, "MMM none", [0, 0, 1]),
            (MSub, "MMM none", [0, 0, 1]),
            (MMul, "MMM none", [0, 0, 1]),
            (MScale, "SMM none", [0, 0, 1]),
            (MTranspose, "M-M none", [0, 0, 1]),
            // Errata K1: MGaussian x10; MUniform unchanged.
            (MGaussian, "--M scale", [0, 0, 100]),
            (MUniform, "--M scale", [0, 0, 10]),
            (MNorm, "M-S none", [0, 0, 2]),
        ]
    }

    fn sig_text(s: OpSig) -> String {
        let slot = |x: Slot| match x {
            Slot::None => '-',
            Slot::S => 'S',
            Slot::V => 'V',
            Slot::M => 'M',
        };

        let c = match s.c {
            ConstKind::None => "none",
            ConstKind::Value => "value",
            ConstKind::Scale => "scale",
            ConstKind::Index => "index",
        };

        format!("{}{}{} {c}", slot(s.a), slot(s.b), slot(s.out))
    }

    /// T-DSL-1: signatures and costs of all 46 ops.
    #[test]
    fn t_dsl_1_signatures_and_cost_table() {
        let t = table();

        assert_eq!(ALL_OPS.len(), 46);
        assert_eq!(t.len(), 46);

        let mut seen = std::collections::HashSet::new();

        for op in ALL_OPS {
            assert!(seen.insert(op), "{op:?} listed twice");
        }

        for ((op, want_sig, coeff), listed) in t.iter().zip(ALL_OPS) {
            assert_eq!(*op, listed, "declaration order");
            assert_eq!(sig_text(sig(*op)), *want_sig, "{op:?} signature");

            for f in [1u64, 8, 16, 32] {
                let want = coeff[0] + coeff[1] * f + coeff[2] * f * f;

                assert_eq!(cost(*op, f), want, "{op:?} cost at F = {f}");
            }
        }

        for f in SUPPORTED_F {
            let fu = f as u64;

            assert_eq!(reference_step_cost(f), 10 * fu * fu + 5 * fu);
        }

        assert_eq!(cost(OpCode::MGaussian, u64::MAX), u64::MAX);
    }

    /// Errata K1: the re-costed ops. Their counted cost rises by the factor the
    /// adversarial-FLOP bench measured against MatVec (SSin/SCos x9, SExp/SLog x5,
    /// VGaussian/MGaussian x10), so counted cost now tracks real time. MatVec and the
    /// three uniform fills are unchanged.
    #[test]
    fn t_dsl_k1_recost() {
        use OpCode::*;

        // Scalars: a flat cost at every width.
        for f in [1u64, 8, 16, 32] {
            assert_eq!(cost(SSin, f), 90);
            assert_eq!(cost(SCos, f), 90);
            assert_eq!(cost(SExp, f), 50);
            assert_eq!(cost(SLog, f), 50);
            assert_eq!(cost(SUniform, f), 10, "SUniform is unchanged");
        }

        // The raised factor is exactly the old cost times the bench ratio.
        for f in [1u64, 8, 16, 32] {
            assert_eq!(cost(SSin, f), 9 * 10);
            assert_eq!(cost(SExp, f), 5 * 10);
            assert_eq!(
                cost(VGaussian, f),
                10 * (10 * f),
                "VGaussian x10 per element"
            );
            assert_eq!(cost(MGaussian, f), 10 * (10 * f * f), "MGaussian x10");

            // Unchanged ops and the MatVec reference.
            assert_eq!(cost(VUniform, f), 10 * f);
            assert_eq!(cost(MUniform, f), 10 * f * f);
            assert_eq!(cost(MatVec, f), 2 * f * f, "MatVec stays f^2 * 2");
        }
    }

    /// T-DSL-2: validate rejects every kind of bad field.
    #[test]
    fn t_dsl_2_validate_rejects_bad_fields() {
        let lim = DslLimits::default();

        let ok = LearnerGenome {
            setup: vec![ins(OpCode::SConst, 0, 0, 3, 0.05)],
            predict: vec![ins(OpCode::MatVec, 0, 0, 1, 0.0)],
            learn: vec![ins(OpCode::VSet, 2, 0, 3, 255.0)],
        };

        assert!(ok.validate(&lim).is_ok());

        let bad_learn = [
            // Index out of range per slot.
            ins(OpCode::SAdd, 10, 0, 0, 0.0),
            ins(OpCode::SAdd, 0, 10, 0, 0.0),
            ins(OpCode::SAdd, 0, 0, 10, 0.0),
            ins(OpCode::VAdd, 8, 0, 0, 0.0),
            ins(OpCode::VAdd, 0, 0, 8, 0.0),
            ins(OpCode::MAdd, 4, 0, 0, 0.0),
            ins(OpCode::MatVec, 0, 8, 1, 0.0),
            ins(OpCode::MatVec, 4, 0, 1, 0.0),
            ins(OpCode::MOuter, 0, 0, 4, 0.0),
            ins(OpCode::VGet, 0, 0, 255, 3.0),
            // Non-zero unused fields.
            ins(OpCode::SAbs, 0, 1, 0, 0.0),
            ins(OpCode::SConst, 1, 0, 0, 1.0),
            ins(OpCode::VGaussian, 0, 1, 0, 1.0),
            ins(OpCode::VAdd, 0, 0, 0, 1.0),
            ins(OpCode::VAdd, 0, 0, 0, -0.0),
            // Bad constants.
            ins(OpCode::SConst, 0, 0, 0, f32::NAN),
            ins(OpCode::SConst, 0, 0, 0, f32::INFINITY),
            ins(OpCode::SConst, 0, 0, 0, 1000.5),
            ins(OpCode::VUniform, 0, 0, 0, -1.0e4),
            ins(OpCode::MGaussian, 0, 0, 0, f32::NEG_INFINITY),
            ins(OpCode::VSet, 0, 0, 0, 1.5),
            ins(OpCode::VSet, 0, 0, 0, 256.0),
            ins(OpCode::VGet, 0, 0, 0, -1.0),
            ins(OpCode::VGet, 0, 0, 0, -0.0),
            ins(OpCode::VGet, 0, 0, 0, f32::NAN),
        ];

        for bad in bad_learn {
            let mut g = ok.clone();

            g.learn.push(bad);

            assert!(g.validate(&lim).is_err(), "accepted {bad:?}");
        }

        // Boundary constants are fine.
        let mut edge = ok.clone();

        edge.learn.push(ins(OpCode::SConst, 0, 0, 9, -1000.0));
        edge.learn.push(ins(OpCode::VGet, 7, 0, 9, 0.0));

        assert!(edge.validate(&lim).is_ok());

        // Too long, per component.
        let filler = ins(OpCode::SAdd, 0, 0, 0, 0.0);
        let maxima = [lim.max_setup, lim.max_predict, lim.max_learn];

        for (comp, max) in maxima.into_iter().enumerate() {
            let make = |len: usize| {
                let mut g = LearnerGenome::default();

                match comp {
                    0 => g.setup = vec![filler; len],
                    1 => g.predict = vec![filler; len],
                    _ => g.learn = vec![filler; len],
                }

                g
            };

            assert!(make(max).validate(&lim).is_ok());
            assert!(
                make(max + 1).validate(&lim).is_err(),
                "component {comp} too long"
            );
        }
    }

    /// T-DSL-3: the static cap accepts exactly 4 x ref and refuses 4 x ref + 1.
    #[test]
    fn t_dsl_3_cost_cap_boundary() {
        let lim = DslLimits::default();
        let matvec = ins(OpCode::MatVec, 0, 0, 1, 0.0);
        let vadd = ins(OpCode::VAdd, 1, 1, 3, 0.0);

        // 20 MatVec + 20 VAdd = 40 F^2 + 20 F = 4 * (10 F^2 + 5 F) at every F.
        let mut g = LearnerGenome {
            setup: vec![],
            predict: vec![matvec; 12],
            learn: vec![matvec; 8],
        };

        g.learn.extend(vec![vadd; 20]);

        for f in SUPPORTED_F {
            assert_eq!(g.step_cost(f), COST_CAP_RATIO * reference_step_cost(f));
        }

        assert!(g.check_cost().is_ok());
        assert!(g.check(&lim).is_ok());

        g.learn.push(ins(OpCode::SConst, 0, 0, 5, 1.0));

        for f in SUPPORTED_F {
            assert_eq!(g.step_cost(f), COST_CAP_RATIO * reference_step_cost(f) + 1);
        }

        assert!(matches!(g.check_cost(), Err(BrutError::Budget(_))));
        assert!(g.check(&lim).is_err());

        // Setup cost never counts against the per-step cap.
        let heavy_setup = LearnerGenome {
            setup: vec![ins(OpCode::MGaussian, 0, 0, 0, 1.0); 16],
            predict: vec![],
            learn: vec![],
        };

        assert!(heavy_setup.check_cost().is_ok());
    }

    /// A random program with short parts (valid fields; may fail the cost cap).
    fn random_program(rng: &mut ChaCha8Rng) -> LearnerGenome {
        let mut part = |comp| -> Vec<Instr> {
            let len = rng.random_range(0..=10);

            (0..len).map(|_| random_instr(rng, comp)).collect()
        };

        let setup = part(Component::Setup);
        let predict = part(Component::Predict);
        let learn = part(Component::Learn);

        LearnerGenome {
            setup,
            predict,
            learn,
        }
    }

    /// T-DSL-4: simplify preserves predictions bit-for-bit (1 000 programs x 3 tasks).
    #[test]
    fn t_dsl_4_simplify_preserves_predictions() {
        let lim = DslLimits::default();
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let seeds = seed_programs();

        let tasks = [
            testkit::task(11, 8, LearnKind::Regression, 16, 8, 48),
            testkit::task(12, 16, LearnKind::Binary, 16, 8, 48),
            testkit::task(13, 8, LearnKind::Multiclass { k: 4 }, 16, 8, 48),
        ];

        let mut programs = 0;
        let mut shrunk = 0;
        let mut removed = 0;

        while programs < 1000 {
            let g = if programs % 2 == 0 {
                random_program(&mut rng)
            } else {
                let mut g = seeds[rng.random_range(0..seeds.len())].1.clone();

                for _ in 0..rng.random_range(1..40) {
                    g = mutate(&g, VariationMode::Full, &mut rng, &lim).0;
                }

                g
            };

            if g.check(&lim).is_err() {
                continue;
            }

            programs += 1;

            let s = g.simplify();
            let before = g.setup.len() + g.predict.len() + g.learn.len();
            let after = s.setup.len() + s.predict.len() + s.learn.len();

            assert!(after <= before);
            assert_eq!(g.effective_len(), after);
            assert_eq!(s.simplify(), s, "simplify is idempotent");
            assert!(s.check(&lim).is_ok());

            if after < before {
                shrunk += 1;
                removed += before - after;
            }

            for t in &tasks {
                let full = testkit::drive(&g, t, 0);
                let lean = testkit::drive(&s, t, 0);

                assert_eq!(full.abort, lean.abort, "abort differs for\n{}", g.render());
                assert_eq!(full.steps_done, lean.steps_done);
                assert_eq!(full.outputs.len(), lean.outputs.len());

                for (x, y) in full.outputs.iter().zip(&lean.outputs) {
                    let same = x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());

                    assert!(
                        same,
                        "{x} != {y} for\n{}\nsimplified\n{}",
                        g.render(),
                        s.render()
                    );
                }
            }
        }

        // The property is only meaningful when simplify actually removes code.
        assert!(shrunk > 300, "only {shrunk} programs shrank");
        assert!(removed > 1000, "only {removed} instructions removed");
    }

    /// Hand-checked liveness cases, including the random stream and the protocol's writes.
    #[test]
    fn simplify_hand_cases() {
        // A dead write to s5 and a dead trailing random op go; the label read in Learn stays.
        let g = LearnerGenome {
            setup: vec![
                ins(OpCode::SConst, 0, 0, 5, 1.0),
                ins(OpCode::VGaussian, 0, 0, 6, 1.0),
                ins(OpCode::SConst, 0, 0, 3, 0.1),
            ],
            predict: vec![ins(OpCode::MatVec, 0, 0, 1, 0.0)],
            learn: vec![
                ins(OpCode::VSub, 2, 1, 3, 0.0),
                ins(OpCode::VScale, 3, 3, 3, 0.0),
                ins(OpCode::MOuter, 3, 0, 1, 0.0),
                ins(OpCode::MAdd, 0, 1, 0, 0.0),
                ins(OpCode::VAdd, 4, 4, 4, 0.0),
            ],
        };

        let s = g.simplify();

        assert_eq!(s.setup, vec![ins(OpCode::SConst, 0, 0, 3, 0.1)]);
        assert_eq!(s.predict, g.predict);
        assert_eq!(s.learn, g.learn[..4].to_vec());

        // A dead random op before a live one must stay (shared stream).
        let g = LearnerGenome {
            setup: vec![
                ins(OpCode::VUniform, 0, 0, 6, 1.0),
                ins(OpCode::VUniform, 0, 0, 1, 1.0),
            ],
            predict: vec![],
            learn: vec![],
        };

        assert_eq!(g.simplify(), g);

        // Writes to v0, v2 and s0 in Setup are dead: the protocol overwrites them.
        let g = LearnerGenome {
            setup: vec![
                ins(OpCode::SConst, 0, 0, 0, 1.0),
                ins(OpCode::VBroadcast, 0, 0, 0, 0.0),
                ins(OpCode::VBroadcast, 0, 0, 2, 0.0),
            ],
            predict: vec![ins(OpCode::VAdd, 0, 2, 1, 0.0)],
            learn: vec![],
        };

        assert!(g.simplify().setup.is_empty());

        // A loop-carried register that never reaches v1 is dead (strong liveness).
        let g = LearnerGenome {
            setup: vec![],
            predict: vec![ins(OpCode::VAdd, 0, 0, 1, 0.0)],
            learn: vec![ins(OpCode::SAdd, 7, 7, 7, 0.0)],
        };

        assert!(g.simplify().learn.is_empty());

        // VSet keeps the rest of the vector live.
        let g = LearnerGenome {
            setup: vec![ins(OpCode::VBroadcast, 3, 0, 1, 0.0)],
            predict: vec![ins(OpCode::VSet, 4, 0, 1, 2.0)],
            learn: vec![],
        };

        assert_eq!(g.simplify(), g);
    }

    /// T-DSL-5: id stable and content-sensitive; JSON round trip; unknown fields rejected.
    #[test]
    fn t_dsl_5_id_and_json() {
        let seeds = seed_programs();
        let linear = &seeds[0].1;

        assert_eq!(linear.id(), linear.clone().id());
        assert_eq!(linear.id().len(), 16);

        // Golden value: a change here changes every program id (a protocol change).
        assert_eq!(linear.id(), "07248a2658f07a8b");

        let mut other = linear.clone();

        other.setup[0].c = 0.06;

        assert_ne!(other.id(), linear.id());

        let mut ids = std::collections::HashSet::new();

        for (_, g) in &seeds {
            assert!(ids.insert(g.id()));

            let text = g.canonical_json();
            let back: LearnerGenome = serde_json::from_str(&text).expect("parse");

            assert_eq!(&back, g);
            assert_eq!(back.id(), g.id());
        }

        let head = "{\"setup\":[{\"op\":\"SConst\",\"a\":0,\"b\":0,\"out\":3,\"c\":0.05}]";

        assert!(linear.canonical_json().starts_with(head));

        let unknown_genome = r#"{"setup":[],"predict":[],"learn":[],"seed":1}"#;
        let unknown_instr =
            r#"{"setup":[{"op":"SConst","out":1,"c":1.0,"x":2}],"predict":[],"learn":[]}"#;
        let unknown_op = r#"{"setup":[{"op":"MatMul"}],"predict":[],"learn":[]}"#;

        assert!(serde_json::from_str::<LearnerGenome>(unknown_genome).is_err());
        assert!(serde_json::from_str::<LearnerGenome>(unknown_instr).is_err());
        assert!(serde_json::from_str::<LearnerGenome>(unknown_op).is_err());
    }

    #[test]
    fn render_and_same_structure() {
        let seeds = seed_programs();
        let linear = &seeds[0].1;
        let text = linear.render();

        assert!(text.contains("MatVec v1 <- m0 . v0"), "{text}");
        assert!(text.contains("SConst s3 <- 0.05"), "{text}");
        assert!(text.contains("MOuter m1 <- v3 (x) v0"), "{text}");
        assert_eq!(text.lines().count(), 3 + 6);

        for op in ALL_OPS {
            let line = render_instr(&ins(op, 0, 0, 0, 0.0));

            assert!(line.starts_with(&format!("{op:?} ")), "{line}");
        }

        let mut tuned = linear.clone();

        tuned.setup[0].c = 0.5;

        assert!(tuned.same_structure(linear));

        tuned.learn[0].out = 4;

        assert!(!tuned.same_structure(linear));
        assert!(!linear.same_structure(&seeds[1].1));
    }

    #[test]
    fn costs_of_parts_and_tasks() {
        let nn = &seed_programs()[1].1;

        for f in SUPPORTED_F {
            let ff = (f * f) as u64;
            let fu = f as u64;

            // Setup is 2 MGaussian + 1 SConst; errata K1 raised MGaussian to 100 f^2
            // (was 20 f^2 + 1). Predict and Learn use no re-costed op, so the step cost
            // (predict + learn) stays reference_step_cost = 10 f^2 + 5 f.
            assert_eq!(nn.setup_cost(f), 200 * ff + 1);
            assert_eq!(nn.predict_cost(f), 4 * ff + fu);
            assert_eq!(nn.learn_cost(f), 6 * ff + 4 * fu);
            assert_eq!(nn.step_cost(f), reference_step_cost(f));

            let want = nn.setup_cost(f) + 384 * nn.step_cost(f) + 3 * 64 * nn.predict_cost(f);

            assert_eq!(nn.task_cost(f, 384, 64), want);
        }

        assert_eq!(nn.task_cost(16, usize::MAX, usize::MAX), u64::MAX);
    }
}
