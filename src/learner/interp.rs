//! Interpreter: executes straight-line programs on a flat, fixed-size memory.
//!
//! Owner: WP1. Contract section 5.2. Pure: no I/O. This is the hot path.
//!
//! Design:
//! - `run` walks the instruction slice once and dispatches with one `match` on the
//!   one-byte opcode. Registers are fixed-size arrays, so no instruction allocates.
//! - Operand indices are clamped to their register file. The compiler can then drop the
//!   bounds checks, and even unvalidated code cannot panic.
//! - Inputs are copied (a vector or one matrix row at a time) before the output is written,
//!   so every op behaves as if its inputs were copied first (aliasing is allowed).
//! - Reductions use eight partial sums combined in a fixed order: deterministic on every
//!   build, and easy for the compiler to vectorise.
//! - min and max follow IEEE 754-2019 minimumNumber / maximumNumber: a NaN operand gives
//!   the other operand and -0 < +0, so operand order never matters (COR-4). max(x, 0) and
//!   Heaviside give 0 for NaN.
//! - Tiny values are stored as +0 (SEC-1). Scalars and vectors keep magnitudes >= 2^-34
//!   (about 5.8e-11), matrices >= 2^-68 (about 3.4e-21); every op flushes smaller results to
//!   +0, and so do inputs, labels, constants and random scales. x86 CPUs compute subnormal
//!   results (below 2^-126) up to 100 times slower than normal ones, which would make the
//!   static FLOP count meaningless as a time bound. With these floors no op forms a
//!   subnormal value: a vector times a vector is >= 2^-68, so MOuter needs no flush at all;
//!   a matrix entry times a vector entry is >= 2^-102 and a multiple of 2^-125, so the sums
//!   in MatVec and MatTVec stay normal too. The few ops whose factors can be smaller (MMul,
//!   MNorm, divisions) compute the product or quotient in f64 or flush their factors first.
//!   Such tiny values do not matter for learning on standardised data.
#![forbid(unsafe_code)]

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::learner::types::{Instr, LearnKind, MAX_K, N_M, N_S, N_V, OpCode};

/// Learner memory at width F. Zero-initialised at task start.
#[derive(Clone)]
pub struct Memory<const F: usize> {
    pub s: [f32; N_S],
    pub v: [[f32; F]; N_V],
    pub m: [[[f32; F]; F]; N_M],
    rng: rand_chacha::ChaCha8Rng,
}

impl<const F: usize> Memory<F> {
    /// Zeroed memory; `rng_seed` = derive_seed(task.seed, "interp/<attempt>").
    pub fn new(rng_seed: u64) -> Self {
        Memory {
            s: [0.0; N_S],
            v: [[0.0; F]; N_V],
            m: [[[0.0; F]; F]; N_M],
            rng: ChaCha8Rng::seed_from_u64(rng_seed),
        }
    }

    /// v0 = x (x.len() == F), tiny values flushed. A shorter x is zero-padded, a longer one
    /// truncated.
    pub fn set_input(&mut self, x: &[f32]) {
        if let Ok(row) = <[f32; F]>::try_from(x) {
            // A plain loop: `array::map` compiles to much slower code here.
            for (slot, value) in self.v[0].iter_mut().zip(row) {
                *slot = flush(value);
            }

            return;
        }

        for (i, slot) in self.v[0].iter_mut().enumerate() {
            *slot = flush(x.get(i).copied().unwrap_or(0.0));
        }
    }

    /// Zero v2 entirely and s0.
    pub fn clear_label(&mut self) {
        self.v[2] = [0.0; F];
        self.s[0] = 0.0;
    }

    /// Regression/Binary: v2[0] = y, rest 0. Multiclass: v2 = one-hot(y) over k. Then s0 = v2[0].
    /// A class index that is not an integer in 0..min(k, F) gives an all-zero v2.
    pub fn set_label(&mut self, kind: LearnKind, y: f32) {
        let v2 = &mut self.v[2];

        *v2 = [0.0; F];

        match kind {
            LearnKind::Regression | LearnKind::Binary => {
                if let Some(first) = v2.first_mut() {
                    *first = flush(y);
                }
            }

            LearnKind::Multiclass { k } => {
                let k = (k as usize).min(F);
                let integral = y.is_finite() && y >= 0.0 && y == y.trunc();

                if integral && (y as usize) < k {
                    v2[y as usize] = 1.0;
                }
            }
        }

        self.s[0] = v2.first().copied().unwrap_or(0.0);
    }

    /// Apply the link in place on v1[0..k_out]: identity, sigmoid, or softmax
    /// (max-subtracted). Values outside 0..k_out are left as they are.
    /// Finite inputs always give finite outputs; k is capped at F (errata E3).
    pub fn apply_link(&mut self, kind: LearnKind) {
        let v1 = &mut self.v[1];

        match kind {
            LearnKind::Regression => {}

            LearnKind::Binary => {
                if let Some(x) = v1.first_mut() {
                    *x = sigmoid(*x);
                }
            }

            LearnKind::Multiclass { k } => {
                let k = (k as usize).min(F);

                softmax(&mut v1[..k]);
            }
        }
    }

    /// Copy of v1[0..k]; entries from min(k, F) on are 0.
    pub fn output(&self, k: usize) -> [f32; MAX_K] {
        let mut out = [0.0; MAX_K];
        let n = k.min(F).min(MAX_K);

        out[..n].copy_from_slice(&self.v[1][..n]);

        out
    }
}

/// Execute one straight-line program on `mem`. Never panics: indices were validated;
/// out-of-range indices (impossible after validate) are clamped in every build.
pub fn run<const F: usize>(code: &[Instr], mem: &mut Memory<F>) {
    if F == 0 {
        return;
    }

    #[cfg(test)]
    test_counters::record(code, F);

    for ins in code {
        exec(ins, mem);
    }
}

// ---------------------------------------------------------------------------------------------
// Operand access
// ---------------------------------------------------------------------------------------------

#[inline(always)]
fn si(i: u8) -> usize {
    (i as usize).min(N_S - 1)
}

#[inline(always)]
fn vi(i: u8) -> usize {
    (i as usize).min(N_V - 1)
}

#[inline(always)]
fn mi(i: u8) -> usize {
    (i as usize).min(N_M - 1)
}

/// Effective vector index of an Index constant: (c as usize) % F. Needs F > 0.
#[inline(always)]
fn idx<const F: usize>(c: f32) -> usize {
    (c as usize) % F
}

// ---------------------------------------------------------------------------------------------
// Scalar semantics
// ---------------------------------------------------------------------------------------------

/// Smallest magnitude a scalar or vector register keeps: 2^-34.
pub const TINY: f32 = f32::from_bits(0x2E80_0000);

/// Smallest magnitude a matrix register keeps: 2^-68 = TINY².
pub const TINY_M: f32 = f32::from_bits(0x1D80_0000);

/// Floor for the factors of a square in MNorm: 2^-50 (squares >= 2^-100).
const SQUARE_FLOOR: f32 = f32::from_bits(0x2680_0000);

/// The f64 values that round to a magnitude >= TINY as f32: |q| >= 2^-34 - 2^-59 (the
/// midpoint below 2^-34 rounds up to it, ties to even).
const TINY_F64_CUTOFF: f64 = f64::from_bits(0x3DCF_FFFF_F000_0000);

/// The same for TINY_M: |q| >= 2^-68 - 2^-93.
const TINY_M_F64_CUTOFF: f64 = f64::from_bits(0x3BAF_FFFF_F000_0000);

/// Below this argument exp() is subnormal or tiny; the result is +0 then.
const EXP_MIN_ARG: f32 = -87.0;

/// |x| below the floor becomes +0; everything else, NaN and infinities included, passes.
/// Branch-free integer form: the magnitude bits order like the magnitudes (NaN above
/// infinity), and the sign of (magnitude - floor) is the mask. Compiles to a few vector
/// instructions inside the op loops.
#[inline(always)]
fn flush_below(x: f32, floor: f32) -> f32 {
    let bits = x.to_bits() as i32;
    let below = ((bits & i32::MAX) - floor.to_bits() as i32) >> 31;

    f32::from_bits((bits & !below) as u32)
}

/// Flush for scalar and vector results (floor TINY).
#[inline(always)]
fn flush(x: f32) -> f32 {
    flush_below(x, TINY)
}

/// Flush for matrix results (floor TINY_M).
#[inline(always)]
fn flush_m(x: f32) -> f32 {
    flush_below(x, TINY_M)
}

/// x * y with f32 rounding for a matrix result, without forming a subnormal product: the
/// f64 product of two f32 values is exact, so rounding it gives the f32 product.
#[inline(always)]
fn mul_m(x: f32, y: f32) -> f32 {
    let p = f64::from(x) * f64::from(y);

    if p.abs() < TINY_M_F64_CUTOFF {
        0.0
    } else {
        p as f32
    }
}

/// x / y with f32 rounding, flushed. Computed in f64, where the quotient of two f32 values
/// is never subnormal; rounding that quotient to f32 gives the correctly rounded f32
/// quotient (53 >= 2 * 24 + 2 bits, so double rounding is harmless).
#[inline(always)]
fn div(x: f32, y: f32) -> f32 {
    let q = f64::from(x) / f64::from(y);

    if q.abs() < TINY_F64_CUTOFF {
        0.0
    } else {
        q as f32
    }
}

/// exp(x), flushed, without ever computing a subnormal result.
#[inline(always)]
fn exp(x: f32) -> f32 {
    if x < EXP_MIN_ARG { 0.0 } else { flush(x.exp()) }
}

/// 1 / (1 + exp(-x)), flushed. Below -87 the result is tiny (so +0); above, the quotient
/// is a normal number, so plain f32 division gives the same bits as `div`.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    if x < EXP_MIN_ARG {
        0.0
    } else {
        flush(1.0 / (1.0 + exp(-x)))
    }
}

/// IEEE 754-2019 minimumNumber: NaN gives the other operand; -0 < +0.
#[inline(always)]
fn min(x: f32, y: f32) -> f32 {
    if x < y {
        x
    } else if y < x {
        y
    } else if x == y {
        f32::from_bits(x.to_bits() | y.to_bits())
    } else if x.is_nan() {
        y
    } else {
        x
    }
}

/// IEEE 754-2019 maximumNumber: NaN gives the other operand; +0 > -0.
#[inline(always)]
fn max(x: f32, y: f32) -> f32 {
    if x > y {
        x
    } else if y > x {
        y
    } else if x == y {
        f32::from_bits(x.to_bits() & y.to_bits())
    } else if x.is_nan() {
        y
    } else {
        x
    }
}

#[inline(always)]
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

#[inline(always)]
fn heaviside(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

/// Max-subtracted softmax in place. A NaN or +inf input yields NaN outputs (the protocol
/// then aborts); a -inf input next to finite ones gets probability 0. Tiny probabilities
/// are flushed.
fn softmax(xs: &mut [f32]) {
    if xs.is_empty() {
        return;
    }

    let top = xs.iter().fold(f32::NEG_INFINITY, |acc, &x| max(x, acc));
    let mut total = 0.0;

    for x in xs.iter_mut() {
        *x = exp(*x - top);
        total += *x;
    }

    // total >= 1 (the top entry is exp(0)) and every entry is 0 or >= 2^-34, so the
    // quotients are normal numbers or 0; the flush keeps the stored values above 2^-34.
    for x in xs.iter_mut() {
        *x = flush(*x / total);
    }
}

// ---------------------------------------------------------------------------------------------
// Reductions (eight fixed partial sums)
// ---------------------------------------------------------------------------------------------

const LANES: usize = 8;

#[inline(always)]
fn fold_lanes(acc: [f32; LANES]) -> f32 {
    let quad = [
        acc[0] + acc[4],
        acc[1] + acc[5],
        acc[2] + acc[6],
        acc[3] + acc[7],
    ];
    let pair = [quad[0] + quad[2], quad[1] + quad[3]];

    pair[0] + pair[1]
}

#[inline(always)]
fn dot(x: &[f32], y: &[f32]) -> f32 {
    let (xc, xt) = x.as_chunks::<LANES>();
    let (yc, yt) = y.as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];

    for (p, q) in xc.iter().zip(yc) {
        for ((a, &u), &w) in acc.iter_mut().zip(p).zip(q) {
            *a += u * w;
        }
    }

    let mut total = fold_lanes(acc);

    for (&u, &w) in xt.iter().zip(yt) {
        total += u * w;
    }

    total
}

#[inline(always)]
fn sum(x: &[f32]) -> f32 {
    let (chunks, tail) = x.as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];

    for p in chunks {
        for (a, &u) in acc.iter_mut().zip(p) {
            *a += u;
        }
    }

    let mut total = fold_lanes(acc);

    for &u in tail {
        total += u;
    }

    total
}

/// Population standard deviation (two passes). Deviations are flushed before squaring and
/// the variance is divided like `div`, so no step forms a subnormal product or quotient.
#[inline(always)]
fn std_dev(x: &[f32]) -> f32 {
    let n = x.len() as f32;
    let mean = sum(x) / n;
    let (chunks, tail) = x.as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];

    for p in chunks {
        for (a, &u) in acc.iter_mut().zip(p) {
            let d = flush(u - mean);

            *a += d * d;
        }
    }

    let mut total = fold_lanes(acc);

    for &u in tail {
        let d = flush(u - mean);

        total += d * d;
    }

    flush(div(total, n).sqrt())
}

// ---------------------------------------------------------------------------------------------
// Random streams
// ---------------------------------------------------------------------------------------------

/// 2^-23: 24 random bits map exactly onto f32.
const INV_2_23: f32 = 1.0 / 8_388_608.0;

/// Random words fetched per block by the vector and matrix fills.
const WORD_BLOCK: usize = 64;

/// Uniform in [-1, 1) from one 32-bit draw.
#[inline(always)]
fn sym_uniform(rng: &mut ChaCha8Rng) -> f32 {
    (rng.next_u32() >> 8) as f32 * INV_2_23 - 1.0
}

/// Rejection rounds before the polar method gives up; each round fails with p = 1 - pi/4,
/// so reaching the limit has probability below 1e-42.
const MAX_POLAR_ROUNDS: usize = 64;

/// Two independent standard normals (Marsaglia polar method). Always finite: 24-bit
/// uniforms keep s >= 2^-46, so |z| <= sqrt(-2 ln s) < 8.
#[inline(always)]
fn normal_pair(rng: &mut ChaCha8Rng) -> (f32, f32) {
    for _ in 0..MAX_POLAR_ROUNDS {
        let u = sym_uniform(rng);
        let v = sym_uniform(rng);
        let s = u * u + v * v;

        if s > 0.0 && s < 1.0 {
            let k = (-2.0 * s.ln() / s).sqrt();

            return (u * k, v * k);
        }
    }

    (0.0, 0.0)
}

/// out[i] = U[-scale, scale), in index order, flushed below `floor` (TINY for vectors,
/// TINY_M for matrices). The scale is a constant, flushed like a scalar first: a tiny
/// scale draws the same words and stores zeros. Draws whole blocks of words at once; the
/// stream is the same as one `sym_uniform` per value.
fn fill_uniform(rng: &mut ChaCha8Rng, out: &mut [f32], scale: f32, floor: f32) {
    let scale = flush(scale);
    let mut bits = [0u32; WORD_BLOCK];

    for chunk in out.chunks_mut(WORD_BLOCK) {
        let words = &mut bits[..chunk.len()];

        rng.fill(words);

        for (o, &w) in chunk.iter_mut().zip(words.iter()) {
            *o = flush_below(((w >> 8) as f32 * INV_2_23 - 1.0) * scale, floor);
        }
    }
}

/// out[i] = N(0, scale^2), filled pairwise in index order, flushed like `fill_uniform`.
/// |z| >= 2^-35 and scale >= 2^-34, so no product is subnormal.
fn fill_normal(rng: &mut ChaCha8Rng, out: &mut [f32], scale: f32, floor: f32) {
    let scale = flush(scale);
    let mut pairs = out.chunks_exact_mut(2);

    for pair in &mut pairs {
        let (z0, z1) = normal_pair(rng);

        pair[0] = flush_below(z0 * scale, floor);
        pair[1] = flush_below(z1 * scale, floor);
    }

    for o in pairs.into_remainder() {
        *o = flush_below(normal_pair(rng).0 * scale, floor);
    }
}

// ---------------------------------------------------------------------------------------------
// Op groups
// ---------------------------------------------------------------------------------------------

#[inline(always)]
fn s_zip<const F: usize>(mem: &mut Memory<F>, ins: &Instr, op: impl Fn(f32, f32) -> f32) {
    let x = mem.s[si(ins.a)];
    let y = mem.s[si(ins.b)];

    mem.s[si(ins.out)] = op(x, y);
}

#[inline(always)]
fn s_map<const F: usize>(mem: &mut Memory<F>, ins: &Instr, op: impl Fn(f32) -> f32) {
    let x = mem.s[si(ins.a)];

    mem.s[si(ins.out)] = op(x);
}

#[inline(always)]
fn v_zip<const F: usize>(mem: &mut Memory<F>, ins: &Instr, op: impl Fn(f32, f32) -> f32) {
    let x = mem.v[vi(ins.a)];
    let y = mem.v[vi(ins.b)];

    for ((o, u), w) in mem.v[vi(ins.out)].iter_mut().zip(x).zip(y) {
        *o = op(u, w);
    }
}

#[inline(always)]
fn v_map<const F: usize>(mem: &mut Memory<F>, ins: &Instr, op: impl Fn(f32) -> f32) {
    let x = mem.v[vi(ins.a)];

    for (o, u) in mem.v[vi(ins.out)].iter_mut().zip(x) {
        *o = op(u);
    }
}

/// v[out] = s[a] * v[b].
#[inline(always)]
fn v_scale<const F: usize>(mem: &mut Memory<F>, ins: &Instr) {
    let k = mem.s[si(ins.a)];
    let x = mem.v[vi(ins.b)];

    for (o, u) in mem.v[vi(ins.out)].iter_mut().zip(x) {
        *o = flush(k * u);
    }
}

/// Matrix kernels work on the flattened F x F values in blocks of this many. With the
/// flush in the loop, fixed blocks of 32 vectorise well; rows of 8 or 16 do not (SEC-1).
const BLOCK: usize = 32;

/// Blocks of a flattened F x F matrix (F <= 32 divides 32 for every supported width;
/// other widths take the element loop of the callers' fallbacks).
#[inline(always)]
fn blocked<const F: usize>() -> bool {
    F <= BLOCK && BLOCK.is_multiple_of(F)
}

/// Elementwise matrix op. Each block of both inputs is copied before the output block is
/// written (copy-first semantics under aliasing).
#[inline(never)]
fn m_zip<const F: usize>(mem: &mut Memory<F>, ins: &Instr, op: impl Fn(f32, f32) -> f32) {
    let (a, b, out) = (mi(ins.a), mi(ins.b), mi(ins.out));

    if !blocked::<F>() {
        for r in 0..F {
            let x = mem.m[a][r];
            let y = mem.m[b][r];

            for ((o, u), w) in mem.m[out][r].iter_mut().zip(x).zip(y) {
                *o = op(u, w);
            }
        }

        return;
    }

    for c in 0..F * F / BLOCK {
        let x = mem.m[a].as_flattened().as_chunks::<BLOCK>().0[c];
        let y = mem.m[b].as_flattened().as_chunks::<BLOCK>().0[c];
        let o = &mut mem.m[out].as_flattened_mut().as_chunks_mut::<BLOCK>().0[c];

        for ((o, u), w) in o.iter_mut().zip(x).zip(y) {
            *o = op(u, w);
        }
    }
}

/// m[out] = s[a] * m[b] (at least 2^-102, never subnormal), flushed, in blocks like `m_zip`.
#[inline(never)]
fn m_scale<const F: usize>(mem: &mut Memory<F>, ins: &Instr) {
    let k = mem.s[si(ins.a)];
    let (b, out) = (mi(ins.b), mi(ins.out));

    if !blocked::<F>() {
        for r in 0..F {
            let y = mem.m[b][r];

            for (o, w) in mem.m[out][r].iter_mut().zip(y) {
                *o = flush_m(k * w);
            }
        }

        return;
    }

    for c in 0..F * F / BLOCK {
        let y = mem.m[b].as_flattened().as_chunks::<BLOCK>().0[c];
        let o = &mut mem.m[out].as_flattened_mut().as_chunks_mut::<BLOCK>().0[c];

        for (o, w) in o.iter_mut().zip(y) {
            *o = flush_m(k * w);
        }
    }
}

/// v[out] = m[a] . v[b] (row dot products).
#[inline(always)]
fn mat_vec<const F: usize>(mem: &mut Memory<F>, ins: &Instr) {
    let x = mem.v[vi(ins.b)];
    let out = vi(ins.out);
    let m = &mem.m[mi(ins.a)];

    for (o, row) in mem.v[out].iter_mut().zip(m) {
        *o = flush(dot(row, &x));
    }
}

/// v[out] = m[a]^T . v[b] (rows scaled and accumulated in row order).
#[inline(always)]
fn mat_t_vec<const F: usize>(mem: &mut Memory<F>, ins: &Instr) {
    let x = mem.v[vi(ins.b)];
    let m = &mem.m[mi(ins.a)];
    let mut acc = [0.0f32; F];

    for (row, u) in m.iter().zip(x) {
        for (a, &w) in acc.iter_mut().zip(row) {
            *a += w * u;
        }
    }

    for (o, a) in mem.v[vi(ins.out)].iter_mut().zip(acc) {
        *o = flush(a);
    }
}

/// m[out] = v[a] (x) v[b]. Both factors are 0 or >= 2^-34, so every product is 0, non-finite
/// or >= 2^-68 = TINY_M: no flush needed (and none of the cost on this hot op).
#[inline(always)]
fn outer<const F: usize>(mem: &mut Memory<F>, ins: &Instr) {
    let x = mem.v[vi(ins.a)];
    let y = mem.v[vi(ins.b)];

    for (row, u) in mem.m[mi(ins.out)].iter_mut().zip(x) {
        for (o, w) in row.iter_mut().zip(y) {
            *o = u * w;
        }
    }
}

#[inline(always)]
fn transpose<const F: usize>(mem: &mut Memory<F>, ins: &Instr) {
    let src = mem.m[mi(ins.a)];

    for (i, row) in mem.m[mi(ins.out)].iter_mut().enumerate() {
        for (o, src_row) in row.iter_mut().zip(&src) {
            *o = src_row[i];
        }
    }
}

/// One instruction. Inlined into `run`'s loop.
#[inline(always)]
fn exec<const F: usize>(ins: &Instr, mem: &mut Memory<F>) {
    let c = ins.c;

    match ins.op {
        OpCode::SConst => mem.s[si(ins.out)] = flush(c),
        OpCode::SAdd => s_zip(mem, ins, |x, y| flush(x + y)),
        OpCode::SSub => s_zip(mem, ins, |x, y| flush(x - y)),
        OpCode::SMul => s_zip(mem, ins, |x, y| flush(x * y)),
        OpCode::SDiv => s_zip(mem, ins, div),
        OpCode::SMin => s_zip(mem, ins, min),
        OpCode::SMax => s_zip(mem, ins, max),
        OpCode::SAbs => s_map(mem, ins, f32::abs),
        OpCode::SRecip => s_map(mem, ins, |x| div(1.0, x)),
        OpCode::SHeaviside => s_map(mem, ins, heaviside),
        OpCode::SSin => s_map(mem, ins, |x| flush(x.sin())),
        OpCode::SCos => s_map(mem, ins, |x| flush(x.cos())),
        OpCode::SExp => s_map(mem, ins, exp),
        OpCode::SLog => s_map(mem, ins, |x| flush(x.ln())),
        OpCode::SUniform => {
            mem.s[si(ins.out)] = flush(sym_uniform(&mut mem.rng) * flush(c.abs()));
        }

        OpCode::VBroadcast => mem.v[vi(ins.out)] = [mem.s[si(ins.a)]; F],
        OpCode::VAdd => v_zip(mem, ins, |x, y| flush(x + y)),
        OpCode::VSub => v_zip(mem, ins, |x, y| flush(x - y)),
        OpCode::VMul => v_zip(mem, ins, |x, y| flush(x * y)),
        OpCode::VDiv => v_zip(mem, ins, div),
        OpCode::VMin => v_zip(mem, ins, min),
        OpCode::VMax => v_zip(mem, ins, max),
        OpCode::VMax0 => v_map(mem, ins, relu),
        OpCode::VAbs => v_map(mem, ins, f32::abs),
        OpCode::VHeaviside => v_map(mem, ins, heaviside),

        OpCode::VScale => v_scale(mem, ins),

        OpCode::VSet => mem.v[vi(ins.out)][idx::<F>(c)] = mem.s[si(ins.a)],
        OpCode::VGaussian => fill_normal(&mut mem.rng, &mut mem.v[vi(ins.out)], c.abs(), TINY),
        OpCode::VUniform => fill_uniform(&mut mem.rng, &mut mem.v[vi(ins.out)], c.abs(), TINY),
        OpCode::MatVec => mat_vec(mem, ins),
        OpCode::MatTVec => mat_t_vec(mem, ins),

        OpCode::MRowMean => {
            let out = vi(ins.out);
            let m = &mem.m[mi(ins.a)];

            for (o, row) in mem.v[out].iter_mut().zip(m) {
                *o = flush(sum(row) / F as f32);
            }
        }

        OpCode::VDot => mem.s[si(ins.out)] = flush(dot(&mem.v[vi(ins.a)], &mem.v[vi(ins.b)])),

        OpCode::VNorm => {
            let x = &mem.v[vi(ins.a)];

            mem.s[si(ins.out)] = flush(dot(x, x).sqrt());
        }

        OpCode::VMean => mem.s[si(ins.out)] = flush(sum(&mem.v[vi(ins.a)]) / F as f32),
        OpCode::VStd => mem.s[si(ins.out)] = std_dev(&mem.v[vi(ins.a)]),
        OpCode::VGet => mem.s[si(ins.out)] = mem.v[vi(ins.a)][idx::<F>(c)],

        OpCode::MOuter => outer(mem, ins),
        OpCode::MAdd => m_zip(mem, ins, |x, y| flush_m(x + y)),
        OpCode::MSub => m_zip(mem, ins, |x, y| flush_m(x - y)),
        OpCode::MMul => m_zip(mem, ins, mul_m),
        OpCode::MScale => m_scale(mem, ins),
        OpCode::MTranspose => transpose(mem, ins),

        OpCode::MGaussian => {
            let m = mem.m[mi(ins.out)].as_flattened_mut();

            fill_normal(&mut mem.rng, m, c.abs(), TINY_M);
        }

        OpCode::MUniform => {
            let m = mem.m[mi(ins.out)].as_flattened_mut();

            fill_uniform(&mut mem.rng, m, c.abs(), TINY_M);
        }

        OpCode::MNorm => {
            let m = &mem.m[mi(ins.a)];
            // Entries below 2^-50 are left out, so no square is subnormal (they would add
            // less than 2^-90 in total).
            let total = m.iter().fold(0.0f32, |acc, row| {
                let mut r = *row;

                for x in r.iter_mut() {
                    *x = flush_below(*x, SQUARE_FLOOR);
                }

                acc + dot(&r, &r)
            });

            mem.s[si(ins.out)] = flush(total.sqrt());
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Test support
// ---------------------------------------------------------------------------------------------

/// Test-only instrumentation of `run`, per thread so parallel tests do not interfere.
/// Calls are always counted; the executed static cost only after `start` (it costs a pass
/// over the code, which would distort benchmarks). Other modules' tests may use it, e.g. to
/// show that an over-cap program never reaches `run` (AC-6).
#[cfg(test)]
pub(crate) mod test_counters {
    use std::cell::Cell;

    use crate::learner::dsl::cost;
    use crate::learner::types::Instr;

    thread_local! {
        static CALLS: Cell<u64> = const { Cell::new(0) };
        static COST: Cell<u64> = const { Cell::new(0) };
        static COST_ON: Cell<bool> = const { Cell::new(false) };
    }

    /// Count one `run` call and, when enabled, the static cost of the code it executes.
    pub(crate) fn record(code: &[Instr], f: usize) {
        CALLS.with(|c| c.set(c.get() + 1));

        if !COST_ON.with(Cell::get) {
            return;
        }

        let executed = code
            .iter()
            .fold(0u64, |acc, ins| acc.saturating_add(cost(ins.op, f as u64)));

        COST.with(|c| c.set(c.get().saturating_add(executed)));
    }

    /// Reset both counters on this thread and start counting cost.
    pub(crate) fn start() {
        CALLS.with(|c| c.set(0));
        COST.with(|c| c.set(0));
        COST_ON.with(|c| c.set(true));
    }

    /// (run calls, executed cost) on this thread since the last start or take; resets both
    /// and stops counting cost.
    pub(crate) fn take() -> (u64, u64) {
        COST_ON.with(|c| c.set(false));

        (CALLS.with(|c| c.replace(0)), COST.with(|c| c.replace(0)))
    }
}

/// Test-only task builder and section 6.2 driver. WP1 cannot call the WP2 protocol, so its
/// tests use this small copy (no time or stop checks).
///
/// One deliberate choice beyond the contract: a non-finite validation output aborts with
/// NonFinite, as on a training step (section 6.2 leaves validation outputs unspecified).
#[cfg(test)]
pub(crate) mod testkit {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::{Memory, run};
    use crate::learner::types::{
        Abort, CHECKPOINT_DIVISORS, FamilyId, Group, Labels, LearnKind, LearnerGenome,
        MAX_ABS_TRAIN_ERROR, TaskInstance,
    };
    use crate::util::{derive_seed, normal_f32};

    /// A hand-built task: standard-normal features, labels from a random linear teacher.
    pub(crate) fn task(
        seed: u64,
        f: usize,
        kind: LearnKind,
        n_train: usize,
        n_valid: usize,
        steps: usize,
    ) -> TaskInstance {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let teacher: Vec<f32> = (0..kind.k_out() * f)
            .map(|_| normal_f32(&mut rng))
            .collect();

        let (x_train, y_train) = sample(&mut rng, &teacher, f, kind, n_train);
        let (x_valid, y_valid) = sample(&mut rng, &teacher, f, kind, n_valid);

        TaskInstance {
            id: seed,
            family: FamilyId("wp1-testkit".to_string()),
            group: Group::Linear,
            f,
            kind,
            x_train,
            y_train,
            y_train_late: None,
            drift_at: None,
            x_valid,
            y_valid,
            n_train,
            n_valid,
            steps,
            seed,
            changed_mask: None,
        }
    }

    fn sample(
        rng: &mut ChaCha8Rng,
        teacher: &[f32],
        f: usize,
        kind: LearnKind,
        n: usize,
    ) -> (Vec<f32>, Labels) {
        let k = kind.k_out();
        let x: Vec<f32> = (0..n * f).map(|_| normal_f32(rng)).collect();
        let norm = (f as f32).sqrt();

        let scores = |row: &[f32]| -> Vec<f32> {
            teacher
                .chunks(f)
                .take(k)
                .map(|w| w.iter().zip(row).map(|(a, b)| a * b).sum::<f32>() / norm)
                .collect()
        };

        let labels = match kind {
            LearnKind::Regression => Labels::Values(
                x.chunks(f)
                    .map(|row| scores(row)[0] + 0.1 * normal_f32(rng))
                    .collect(),
            ),

            LearnKind::Binary => Labels::Classes(
                x.chunks(f)
                    .map(|row| u8::from(scores(row)[0] > 0.0))
                    .collect(),
            ),

            LearnKind::Multiclass { .. } => Labels::Classes(
                x.chunks(f)
                    .map(|row| {
                        let s = scores(row);
                        let best = (0..s.len()).fold(0, |b, j| if s[j] > s[b] { j } else { b });

                        best as u8
                    })
                    .collect(),
            ),
        };

        (x, labels)
    }

    /// Everything the driver recorded: linked training predictions, then validation outputs.
    pub(crate) struct Trace {
        pub(crate) outputs: Vec<f32>,
        pub(crate) abort: Option<Abort>,
        pub(crate) steps_done: usize,
    }

    /// Run `g` on `task` as in section 6.2 (F must be 8, 16 or 32).
    pub(crate) fn drive(g: &LearnerGenome, task: &TaskInstance, attempt: u32) -> Trace {
        match task.f {
            8 => drive_f::<8>(g, task, attempt),
            16 => drive_f::<16>(g, task, attempt),
            32 => drive_f::<32>(g, task, attempt),
            _ => Trace {
                outputs: Vec::new(),
                abort: Some(Abort::OverCost),
                steps_done: 0,
            },
        }
    }

    pub(crate) fn permutation(n: usize, seed: u64) -> Vec<usize> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut p: Vec<usize> = (0..n).collect();

        for i in (1..n).rev() {
            p.swap(i, rng.random_range(0..=i));
        }

        p
    }

    pub(crate) fn label(labels: &Labels, i: usize) -> f32 {
        match labels {
            Labels::Values(v) => v[i],
            Labels::Classes(c) => c[i] as f32,
        }
    }

    fn drive_f<const F: usize>(g: &LearnerGenome, task: &TaskInstance, attempt: u32) -> Trace {
        let k = task.kind.k_out().min(F);
        let seed = derive_seed(task.seed, &format!("interp/{attempt}"));
        let checkpoints = CHECKPOINT_DIVISORS.map(|d| task.steps / d);
        let n = task.n_train;

        let mut mem = Memory::<F>::new(seed);
        let mut trace = Trace {
            outputs: Vec::new(),
            abort: None,
            steps_done: 0,
        };

        run(&g.setup, &mut mem);

        for _ in checkpoints.iter().filter(|&&c| c == 0) {
            if !validate(g, task, &mem, k, &mut trace) {
                return trace;
            }
        }

        let mut order = Vec::new();

        for s in 0..task.steps {
            if s % n == 0 {
                order = permutation(n, derive_seed(task.seed, &format!("stream/{}", s / n)));
            }

            let i = order[s % n];

            mem.set_input(&task.x_train[i * F..(i + 1) * F]);
            mem.clear_label();
            run(&g.predict, &mut mem);
            mem.apply_link(task.kind);

            let out = mem.output(k);

            if out[..k].iter().any(|x| !x.is_finite()) {
                trace.abort = Some(Abort::NonFinite);

                return trace;
            }

            let late = task.drift_at.is_some_and(|d| s >= d);
            let labels = match (&task.y_train_late, late) {
                (Some(l), true) => l,
                _ => &task.y_train,
            };
            let y = label(labels, i);

            if task.kind == LearnKind::Regression && (out[0] - y).abs() > MAX_ABS_TRAIN_ERROR {
                trace.abort = Some(Abort::ErrorTooLarge);

                return trace;
            }

            trace.outputs.extend_from_slice(&out[..k]);

            mem.set_label(task.kind, y);
            run(&g.learn, &mut mem);

            trace.steps_done = s + 1;

            for _ in checkpoints.iter().filter(|&&c| c == s + 1) {
                if !validate(g, task, &mem, k, &mut trace) {
                    return trace;
                }
            }
        }

        trace
    }

    /// One checkpoint on a clone of the trained memory. False after a NonFinite abort.
    fn validate<const F: usize>(
        g: &LearnerGenome,
        task: &TaskInstance,
        mem: &Memory<F>,
        k: usize,
        trace: &mut Trace,
    ) -> bool {
        let mut probe = mem.clone();

        for i in 0..task.n_valid {
            probe.set_input(&task.x_valid[i * F..(i + 1) * F]);
            probe.clear_label();
            run(&g.predict, &mut probe);
            probe.apply_link(task.kind);

            let out = probe.output(k);

            if out[..k].iter().any(|x| !x.is_finite()) {
                trace.abort = Some(Abort::NonFinite);

                return false;
            }

            trace.outputs.extend_from_slice(&out[..k]);
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::learner::dsl::{ALL_OPS, ConstKind, Slot, cost, sig, slot_len};
    use crate::learner::mutate::{initial_member, mutate, random_instr};
    use crate::learner::seeds::{reference_nn, seed_programs};
    use crate::learner::types::{Abort, Component, DslLimits, LearnerGenome, VariationMode};
    use crate::util::{derive_seed, normal_f32};

    /// Memory with standard-normal registers.
    fn random_memory<const F: usize>(rng: &mut ChaCha8Rng) -> Memory<F> {
        let mut mem = Memory::<F>::new(rng.random());

        for x in mem.s.iter_mut() {
            *x = normal_f32(rng);
        }

        for x in mem.v.as_flattened_mut() {
            *x = normal_f32(rng);
        }

        for x in mem.m.as_flattened_mut().as_flattened_mut() {
            *x = normal_f32(rng);
        }

        mem
    }

    /// Replays the random draws of one random op.
    type Draw = Box<dyn Fn(&mut ChaCha8Rng) -> Vec<f32>>;

    /// Expected output of one op: values and the error scale of each element.
    enum Want {
        S(f64, f64),
        V(Vec<(f64, f64)>),
        M(Vec<(f64, f64)>),
        Exact(Draw),
    }

    /// Naive f64 reference that reads only the state before the op (copy-first semantics).
    fn reference<const F: usize>(ins: &Instr, before: &Memory<F>) -> Want {
        use OpCode::*;

        let s = |i: u8| before.s[i as usize] as f64;
        let v = |i: u8| -> Vec<f64> { before.v[i as usize].iter().map(|&x| x as f64).collect() };
        let m = |i: u8| -> Vec<Vec<f64>> {
            before.m[i as usize]
                .iter()
                .map(|row| row.iter().map(|&x| x as f64).collect())
                .collect()
        };
        let s32 = |i: u8| before.s[i as usize];
        let exact = |x: f64| (x, x.abs());
        let elem = |x: Vec<f64>| x.into_iter().map(exact).collect::<Vec<_>>();
        let c = ins.c;
        let scale = c.abs();
        let (a, b) = (ins.a, ins.b);
        let n = F as f64;

        let dot = |x: &[f64], y: &[f64]| -> (f64, f64) {
            let value = x.iter().zip(y).map(|(p, q)| p * q).sum();
            let mag = x.iter().zip(y).map(|(p, q)| (p * q).abs()).sum();

            (value, mag)
        };

        match ins.op {
            SConst => Want::S(c as f64, (c as f64).abs()),
            SAdd => Want::S(s(a) + s(b), (s(a) + s(b)).abs()),
            SSub => Want::S(s(a) - s(b), (s(a) - s(b)).abs()),
            SMul => Want::S(s(a) * s(b), (s(a) * s(b)).abs()),
            SDiv => Want::S(s(a) / s(b), (s(a) / s(b)).abs()),
            SMin => Want::S(s(a).min(s(b)), s(a).min(s(b)).abs()),
            SMax => Want::S(s(a).max(s(b)), s(a).max(s(b)).abs()),
            SAbs => Want::S(s(a).abs(), s(a).abs()),
            SRecip => Want::S(1.0 / s(a), (1.0 / s(a)).abs()),
            SHeaviside => Want::S(if s(a) > 0.0 { 1.0 } else { 0.0 }, 1.0),
            SSin => Want::S(s32(a).sin() as f64, 1.0),
            SCos => Want::S(s32(a).cos() as f64, 1.0),
            SExp => Want::S(s32(a).exp() as f64, (s32(a).exp() as f64).abs()),
            SLog => Want::S(s32(a).ln() as f64, (s32(a).ln() as f64).abs()),

            VBroadcast => Want::V(elem(vec![s(a); F])),
            VAdd => Want::V(elem(v(a).iter().zip(v(b)).map(|(p, q)| p + q).collect())),
            VSub => Want::V(elem(v(a).iter().zip(v(b)).map(|(p, q)| p - q).collect())),
            VMul => Want::V(elem(v(a).iter().zip(v(b)).map(|(p, q)| p * q).collect())),
            VDiv => Want::V(elem(v(a).iter().zip(v(b)).map(|(p, q)| p / q).collect())),
            VMin => Want::V(elem(v(a).iter().zip(v(b)).map(|(p, q)| p.min(q)).collect())),
            VMax => Want::V(elem(v(a).iter().zip(v(b)).map(|(p, q)| p.max(q)).collect())),
            VMax0 => Want::V(elem(v(a).iter().map(|p| p.max(0.0)).collect())),
            VScale => Want::V(elem(v(b).iter().map(|q| s(a) * q).collect())),
            VAbs => Want::V(elem(v(a).iter().map(|p| p.abs()).collect())),
            VHeaviside => Want::V(elem(
                v(a).iter()
                    .map(|&p| if p > 0.0 { 1.0 } else { 0.0 })
                    .collect(),
            )),

            VSet => {
                let mut x = v(ins.out);

                x[(c as usize) % F] = s(a);

                Want::V(elem(x))
            }

            MatVec => {
                let x = v(b);

                Want::V(m(a).iter().map(|row| dot(row, &x)).collect())
            }

            MatTVec => {
                let x = v(b);
                let t = m(a);

                Want::V(
                    (0..F)
                        .map(|j| {
                            let col: Vec<f64> = (0..F).map(|i| t[i][j]).collect();

                            dot(&col, &x)
                        })
                        .collect(),
                )
            }

            MRowMean => Want::V(
                m(a).iter()
                    .map(|row| {
                        let total: f64 = row.iter().sum();
                        let mag: f64 = row.iter().map(|x| x.abs()).sum();

                        (total / n, mag / n)
                    })
                    .collect(),
            ),

            VDot => {
                let (value, mag) = dot(&v(a), &v(b));

                Want::S(value, mag)
            }

            VNorm => {
                let x = v(a);
                let norm = dot(&x, &x).0.sqrt();

                Want::S(norm, norm)
            }

            VMean => {
                let x = v(a);

                Want::S(
                    x.iter().sum::<f64>() / n,
                    x.iter().map(|p| p.abs()).sum::<f64>() / n,
                )
            }

            VStd => {
                let x = v(a);
                let mean = x.iter().sum::<f64>() / n;
                let var = x.iter().map(|p| (p - mean) * (p - mean)).sum::<f64>() / n;
                let rms = (x.iter().map(|p| p * p).sum::<f64>() / n).sqrt();

                Want::S(var.sqrt(), rms)
            }

            VGet => Want::S(v(a)[(c as usize) % F], 1.0),

            MOuter => {
                let (x, y) = (v(a), v(b));

                Want::M(elem(
                    x.iter()
                        .flat_map(|p| y.iter().map(move |q| p * q))
                        .collect(),
                ))
            }

            MAdd | MSub | MMul => {
                let (x, y) = (m(a), m(b));
                let f = |p: f64, q: f64| match ins.op {
                    MAdd => p + q,
                    MSub => p - q,
                    _ => p * q,
                };

                Want::M(elem(
                    x.iter()
                        .flatten()
                        .zip(y.iter().flatten())
                        .map(|(&p, &q)| f(p, q))
                        .collect(),
                ))
            }

            MScale => Want::M(elem(m(b).iter().flatten().map(|q| s(a) * q).collect())),

            MTranspose => {
                let x = m(a);

                Want::M(elem((0..F * F).map(|k| x[k % F][k / F]).collect()))
            }

            MNorm => {
                let x = m(a);
                let norm = x.iter().flatten().map(|p| p * p).sum::<f64>().sqrt();

                Want::S(norm, norm)
            }

            SUniform => Want::Exact(Box::new(move |r| vec![sym_uniform(r) * scale])),

            VUniform => Want::Exact(Box::new(move |r| {
                (0..F).map(|_| sym_uniform(r) * scale).collect()
            })),

            MUniform => Want::Exact(Box::new(move |r| {
                (0..F * F).map(|_| sym_uniform(r) * scale).collect()
            })),

            VGaussian | MGaussian => {
                let len = if ins.op == VGaussian { F } else { F * F };

                Want::Exact(Box::new(move |r| {
                    let mut out = Vec::new();

                    while out.len() < len {
                        let (z0, z1) = normal_pair(r);

                        out.push(z0 * scale);
                        out.push(z1 * scale);
                    }

                    out.truncate(len);

                    out
                }))
            }
        }
    }

    fn close(got: f32, want: f64, scale: f64) -> bool {
        if want.is_nan() {
            return got.is_nan();
        }

        let want32 = want as f32;

        if !want32.is_finite() || !got.is_finite() {
            return got == want32;
        }

        (got as f64 - want).abs() <= 1e-6 * scale.max(want.abs()).max(f64::MIN_POSITIVE)
    }

    /// Compare one executed op against the reference; every other register must be untouched.
    fn check_op<const F: usize>(ins: &Instr, before: &Memory<F>) {
        let mut after = before.clone();

        run(std::slice::from_ref(ins), &mut after);

        let s = sig(ins.op);
        let out = ins.out as usize;
        let want = reference(ins, before);

        let got: Vec<f32> = match s.out {
            Slot::S => vec![after.s[out]],
            Slot::V => after.v[out].to_vec(),
            Slot::M => after.m[out].as_flattened().to_vec(),
            Slot::None => Vec::new(),
        };

        let pairs: Vec<(f64, f64)> = match want {
            Want::S(x, mag) => vec![(x, mag)],
            Want::V(x) | Want::M(x) => x,

            Want::Exact(draw) => {
                let mut r = before.rng.clone();

                draw(&mut r)
                    .into_iter()
                    .map(|x| (x as f64, x.abs() as f64))
                    .collect()
            }
        };

        assert_eq!(got.len(), pairs.len(), "{ins:?}");

        for (k, (g, (w, mag))) in got.iter().zip(&pairs).enumerate() {
            assert!(
                close(*g, *w, *mag),
                "{ins:?} F = {F} element {k}: got {g}, want {w}"
            );
        }

        // Everything except the output register is bit-identical.
        let mut masked_before = before.clone();
        let mut masked_after = after.clone();

        match s.out {
            Slot::S => {
                masked_before.s[out] = 0.0;
                masked_after.s[out] = 0.0;
            }
            Slot::V => {
                masked_before.v[out] = [0.0; F];
                masked_after.v[out] = [0.0; F];
            }
            Slot::M => {
                masked_before.m[out] = [[0.0; F]; F];
                masked_after.m[out] = [[0.0; F]; F];
            }
            Slot::None => {}
        }

        let bits = |m: &Memory<F>| -> Vec<u32> {
            m.s.iter()
                .chain(m.v.as_flattened())
                .chain(m.m.as_flattened().as_flattened())
                .map(|x| x.to_bits())
                .collect()
        };

        assert_eq!(
            bits(&masked_before),
            bits(&masked_after),
            "{ins:?} wrote elsewhere"
        );
    }

    /// A random valid instruction for `op`; with `alias`, same-typed operands share an index.
    fn instr_for(op: OpCode, rng: &mut ChaCha8Rng, alias: bool) -> Instr {
        let s = sig(op);
        let mut pick = |slot: Slot| -> u8 {
            match slot_len(slot) {
                0 => 0,
                n => rng.random_range(0..n) as u8,
            }
        };

        let mut ins = Instr {
            op,
            a: pick(s.a),
            b: pick(s.b),
            out: pick(s.out),
            c: 0.0,
        };

        if alias {
            if s.b == s.a {
                ins.b = ins.a;
            }

            if s.out == s.a {
                ins.out = ins.a;
            }

            if s.out == s.b {
                ins.out = ins.b;
            }
        }

        ins.c = match s.c {
            ConstKind::None => 0.0,
            ConstKind::Value => normal_f32(rng) * 3.0,
            ConstKind::Scale => -0.7,
            ConstKind::Index => rng.random_range(0..=255u32) as f32,
        };

        ins
    }

    fn each_op_against_reference<const F: usize>(alias: bool, trials: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(100 + F as u64 + u64::from(alias));

        for op in ALL_OPS {
            for _ in 0..trials {
                let before = random_memory::<F>(&mut rng);
                let ins = instr_for(op, &mut rng, alias);

                check_op(&ins, &before);
            }
        }
    }

    /// T-INT-1: each op equals a naive reference (rel. err <= 1e-6) at F = 8, 16, 32.
    #[test]
    fn t_int_1_ops_match_reference() {
        each_op_against_reference::<8>(false, 30);
        each_op_against_reference::<16>(false, 30);
        each_op_against_reference::<32>(false, 15);
    }

    /// T-INT-2: aliasing (out == a == b) equals copy-first semantics.
    #[test]
    fn t_int_2_aliasing_is_copy_first() {
        each_op_against_reference::<8>(true, 20);
        each_op_against_reference::<16>(true, 20);
        each_op_against_reference::<32>(true, 10);
    }

    /// T-INT-3: same seed gives the same random-op outputs; different attempts differ.
    #[test]
    fn t_int_3_random_ops_follow_the_task_seed() {
        let code: Vec<Instr> = [
            OpCode::SUniform,
            OpCode::VGaussian,
            OpCode::VUniform,
            OpCode::MGaussian,
            OpCode::MUniform,
        ]
        .into_iter()
        .map(|op| Instr {
            op,
            a: 0,
            b: 0,
            out: 1,
            c: 0.5,
        })
        .collect();

        let draw = |attempt: u32| {
            let mut mem = Memory::<16>::new(derive_seed(77, &format!("interp/{attempt}")));

            run(&code, &mut mem);

            (mem.s[1], mem.v[1], mem.m[1])
        };

        assert_eq!(draw(0), draw(0));
        assert_eq!(draw(1), draw(1));

        let (s0, v0, m0) = draw(0);
        let (s1, v1, m1) = draw(1);

        assert_ne!(s0, s1);
        assert_ne!(v0, v1);
        assert_ne!(m0, m1);
    }

    #[test]
    fn random_streams_have_the_right_moments() {
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let n = 100_000;

        let normals: Vec<f64> = (0..n / 2)
            .flat_map(|_| {
                let (z0, z1) = normal_pair(&mut rng);

                [z0 as f64, z1 as f64]
            })
            .collect();

        let mean = normals.iter().sum::<f64>() / n as f64;
        let var = normals.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;

        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "var {var}");
        assert!(normals.iter().all(|x| x.is_finite() && x.abs() < 8.0));

        let uniforms: Vec<f32> = (0..n).map(|_| sym_uniform(&mut rng)).collect();
        let mean = uniforms.iter().map(|&x| x as f64).sum::<f64>() / n as f64;

        assert!(uniforms.iter().all(|x| (-1.0..1.0).contains(x)));
        assert!(mean.abs() < 0.01);

        // The largest allowed scale keeps every random op finite.
        let big = DslLimits::default().const_abs_max;
        let mut mem = Memory::<32>::new(1);
        let code = [
            Instr {
                op: OpCode::MGaussian,
                a: 0,
                b: 0,
                out: 0,
                c: -big,
            },
            Instr {
                op: OpCode::MUniform,
                a: 0,
                b: 0,
                out: 1,
                c: big,
            },
        ];

        run(&code, &mut mem);

        assert!(
            mem.m
                .as_flattened()
                .as_flattened()
                .iter()
                .all(|x| x.is_finite())
        );
    }

    #[test]
    fn labels_inputs_links_and_outputs() {
        let mut mem = Memory::<8>::new(0);

        mem.set_input(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(mem.v[0][7], 8.0);

        mem.set_input(&[1.0, 2.0]);
        assert_eq!(mem.v[0], [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        mem.set_label(LearnKind::Regression, -2.5);
        assert_eq!(mem.v[2][0], -2.5);
        assert_eq!(mem.s[0], -2.5);

        mem.set_label(LearnKind::Multiclass { k: 5 }, 3.0);
        assert_eq!(mem.v[2], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(mem.s[0], 0.0);

        mem.set_label(LearnKind::Multiclass { k: 5 }, 0.0);
        assert_eq!(mem.s[0], 1.0);

        for bad in [5.0, -1.0, 1.5, f32::NAN, f32::INFINITY] {
            mem.set_label(LearnKind::Multiclass { k: 5 }, bad);
            assert_eq!(mem.v[2], [0.0; 8], "label {bad}");
        }

        // k > F is capped at F (errata E3).
        mem.set_label(LearnKind::Multiclass { k: 10 }, 9.0);
        assert_eq!(mem.v[2], [0.0; 8]);

        mem.clear_label();
        assert_eq!(mem.v[2], [0.0; 8]);
        assert_eq!(mem.s[0], 0.0);

        mem.v[1] = [0.0, 1.0, 2.0, 3.0, 9.0, 9.0, 9.0, 9.0];
        mem.apply_link(LearnKind::Multiclass { k: 4 });

        let total: f32 = mem.v[1][..4].iter().sum();

        assert!((total - 1.0).abs() < 1e-6);
        assert!(mem.v[1][3] > mem.v[1][2]);
        assert_eq!(mem.v[1][4..], [9.0; 4]);

        mem.v[1][0] = 0.0;
        mem.apply_link(LearnKind::Binary);
        assert_eq!(mem.v[1][0], 0.5);

        let out = mem.output(4);

        assert_eq!(out[..4], mem.v[1][..4]);
        assert_eq!(out[4..], [0.0; MAX_K - 4]);
        assert_eq!(mem.output(99)[..8], mem.v[1]);
    }

    /// Finite inputs to the link always give finite outputs in range, even at the extremes.
    #[test]
    fn links_are_finite_on_finite_inputs() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let extremes = [
            f32::MAX,
            -f32::MAX,
            1e30,
            -1e30,
            88.8,
            -88.8,
            0.0,
            -0.0,
            1e-40,
        ];

        for _ in 0..5000 {
            let mut mem = Memory::<16>::new(0);

            for x in mem.v[1].iter_mut() {
                *x = if rng.random_bool(0.5) {
                    extremes[rng.random_range(0..extremes.len())]
                } else {
                    normal_f32(&mut rng) * 1e3
                };
            }

            let raw = mem.v[1];
            let k = rng.random_range(3..=10u8);

            mem.apply_link(LearnKind::Multiclass { k });

            let probs = &mem.v[1][..k as usize];
            let total: f32 = probs.iter().sum();

            assert!(
                probs
                    .iter()
                    .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
            );
            assert!((total - 1.0).abs() < 1e-5, "sum {total} for {raw:?}");

            mem.v[1] = raw;
            mem.apply_link(LearnKind::Binary);

            assert!((0.0..=1.0).contains(&mem.v[1][0]));
        }
    }

    /// Every scalar and vector register holds 0, a non-finite value or a magnitude >= 2^-34;
    /// every matrix register the same with 2^-68.
    fn assert_no_tiny<const F: usize>(mem: &Memory<F>, what: &str) {
        let ok = |x: f32, floor: f32| x == 0.0 || !x.is_finite() || x.abs() >= floor;
        let sv = mem.s.iter().chain(mem.v.as_flattened());

        for (k, &x) in sv.enumerate() {
            assert!(ok(x, TINY), "{what}: scalar/vector value {k} is {x:e}");
        }

        for (k, &x) in mem.m.as_flattened().as_flattened().iter().enumerate() {
            assert!(ok(x, TINY_M), "{what}: matrix value {k} is {x:e}");
        }
    }

    /// SEC-1: valid programs that fill registers with subnormal or tiny values (directly or
    /// by multiplying tiny normal numbers) store zeros instead, so later products never
    /// touch the slow subnormal range.
    #[test]
    fn tiny_and_subnormal_values_are_flushed() {
        let i = |op, a, b, out, c| Instr { op, a, b, out, c };
        let programs: Vec<(&str, Vec<Instr>)> = vec![
            ("subnormal fill", vec![i(OpCode::MUniform, 0, 0, 0, 1e-39)]),
            (
                "subnormal gaussian",
                vec![i(OpCode::VGaussian, 0, 0, 3, 1e-39)],
            ),
            (
                "subnormal constant",
                vec![i(OpCode::SConst, 0, 0, 3, -1e-40)],
            ),
            (
                "tiny products",
                vec![
                    i(OpCode::MUniform, 0, 0, 0, 1e-8),
                    i(OpCode::VUniform, 0, 0, 3, 1e-8),
                    i(OpCode::MOuter, 3, 3, 0, 0.0),
                    i(OpCode::MMul, 0, 0, 1, 0.0),
                    i(OpCode::MMul, 1, 1, 1, 0.0),
                    i(OpCode::MNorm, 1, 0, 6, 0.0),
                    i(OpCode::MatVec, 0, 3, 4, 0.0),
                    i(OpCode::MOuter, 3, 3, 2, 0.0),
                    i(OpCode::VMul, 3, 3, 5, 0.0),
                    i(OpCode::VDot, 3, 3, 4, 0.0),
                ],
            ),
            (
                "tiny quotients",
                vec![
                    // s1 = 1e-30, s2 = 1e12, s6 = 1e38.
                    i(OpCode::SConst, 0, 0, 1, 1e-30),
                    i(OpCode::SConst, 0, 0, 2, 1000.0),
                    i(OpCode::SMul, 2, 2, 2, 0.0),
                    i(OpCode::SMul, 2, 2, 2, 0.0),
                    i(OpCode::SMul, 2, 2, 6, 0.0),
                    i(OpCode::SMul, 6, 2, 6, 0.0),
                    i(OpCode::SConst, 0, 0, 7, 100.0),
                    i(OpCode::SMul, 6, 7, 6, 0.0),
                    i(OpCode::SDiv, 1, 2, 3, 0.0),
                    i(OpCode::SRecip, 6, 0, 4, 0.0),
                    i(OpCode::VBroadcast, 1, 0, 3, 0.0),
                    i(OpCode::VBroadcast, 2, 0, 4, 0.0),
                    i(OpCode::VDiv, 3, 4, 5, 0.0),
                ],
            ),
            (
                "tiny exponentials",
                vec![
                    i(OpCode::SConst, 0, 0, 1, -100.0),
                    i(OpCode::SExp, 1, 0, 2, 0.0),
                    i(OpCode::SConst, 0, 0, 3, -50.0),
                    i(OpCode::SExp, 3, 0, 4, 0.0),
                ],
            ),
            (
                "cancellation",
                vec![
                    i(OpCode::SConst, 0, 0, 1, 1e-4),
                    i(OpCode::SConst, 0, 0, 2, 1.0000001e-4),
                    i(OpCode::SSub, 2, 1, 3, 0.0),
                    i(OpCode::VBroadcast, 1, 0, 4, 0.0),
                    i(OpCode::VBroadcast, 2, 0, 5, 0.0),
                    i(OpCode::VSub, 5, 4, 6, 0.0),
                    i(OpCode::VStd, 5, 0, 4, 0.0),
                ],
            ),
        ];

        for (what, code) in &programs {
            let g = LearnerGenome {
                setup: code.clone(),
                predict: Vec::new(),
                learn: Vec::new(),
            };

            assert!(
                g.check(&DslLimits::default()).is_ok(),
                "{what} must be valid"
            );

            let mut mem = Memory::<16>::new(1);

            run(code, &mut mem);
            assert_no_tiny(&mem, what);
        }

        // Tiny inputs and labels are flushed too.
        let mut mem = Memory::<8>::new(0);

        mem.set_input(&[1e-39, 1e-25, 1.0, -1e-11, 0.0, 2.0, 3.0, 4.0]);
        mem.set_label(LearnKind::Regression, 1e-38);
        assert_no_tiny(&mem, "inputs");
        assert_eq!(mem.v[0][2], 1.0);

        // Links never store tiny probabilities.
        mem.v[1] = [-30.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0];
        mem.apply_link(LearnKind::Binary);
        assert_eq!(mem.v[1][0], 0.0);

        mem.v[1] = [-200.0, 0.0, 1.0, -30.0, 0.0, 0.0, 0.0, 0.0];
        mem.apply_link(LearnKind::Multiclass { k: 4 });
        assert_eq!((mem.v[1][0], mem.v[1][3]), (0.0, 0.0));
        assert_no_tiny(&mem, "links");
    }

    /// SEC-1: random programs on memories seeded with tiny values never store a tiny value.
    #[test]
    fn prop_no_op_stores_a_tiny_value() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let tiny = [
            1e-39f32, -3e-42, 1e-25, -2e-20, 5e-30, 1e-19, 3e-11, -1e-10, 1e-5,
        ];

        for trial in 0..400 {
            let mut mem = random_memory::<8>(&mut rng);

            for x in mem.v.as_flattened_mut().iter_mut().step_by(3) {
                *x = flush(*x * 1e-9);
            }

            for x in mem
                .m
                .as_flattened_mut()
                .as_flattened_mut()
                .iter_mut()
                .step_by(5)
            {
                *x = flush_m(*x * 1e-19);
            }

            let code: Vec<Instr> = (0..12)
                .map(|_| {
                    let op = ALL_OPS[rng.random_range(0..ALL_OPS.len())];
                    let alias = rng.random_bool(0.3);
                    let mut ins = instr_for(op, &mut rng, alias);

                    if matches!(sig(op).c, ConstKind::Value | ConstKind::Scale)
                        && rng.random_bool(0.5)
                    {
                        ins.c = tiny[rng.random_range(0..tiny.len())];
                    }

                    ins
                })
                .collect();

            run(&code, &mut mem);
            assert_no_tiny(&mem, &format!("trial {trial}"));
        }
    }

    /// SEC-1: division through f64 equals IEEE f32 division whenever the f32 quotient is not
    /// tiny, and is exactly 0 when it is.
    #[test]
    fn division_matches_f32_division_outside_the_tiny_range() {
        let mut rng = ChaCha8Rng::seed_from_u64(12);
        let specials = [
            0.0,
            -0.0,
            1.0,
            f32::MAX,
            f32::MIN_POSITIVE,
            TINY,
            f32::INFINITY,
            f32::NAN,
            3e38,
            1e-10,
            TINY_M,
        ];

        for k in 0..200_000 {
            let draw = |rng: &mut ChaCha8Rng| -> f32 {
                if rng.random_bool(0.05) {
                    specials[rng.random_range(0..specials.len())]
                } else {
                    let e = rng.random_range(-70i32..70);
                    let x = normal_f32(rng) * 2f32.powi(e);

                    flush(x)
                }
            };
            let x = draw(&mut rng);
            let y = if k % 7 == 0 {
                x * 1.0000001
            } else {
                draw(&mut rng)
            };
            let want = flush(x / y);
            let got = div(x, y);

            assert!(
                got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                "{x:e} / {y:e}: got {got:e}, want {want:e}"
            );
        }

        // The rounding boundary just below 2^-34 is honoured exactly.
        let below = f32::from_bits(TINY.to_bits() - 1);

        assert_eq!(div(below, 1.0), 0.0);
        assert_eq!(div(TINY, 1.0), TINY);
        assert_eq!(div(TINY * 3.0, 3.0), TINY);

        // mul_m: the f32 product, flushed below 2^-68, never formed in f32 when tiny.
        for k in 0..200_000u32 {
            let x = normal_f32(&mut rng) * 2f32.powi(rng.random_range(-70i32..60));
            let y = if k % 5 == 0 {
                specials[rng.random_range(0..specials.len())]
            } else {
                normal_f32(&mut rng) * 2f32.powi(rng.random_range(-70i32..60))
            };
            let want = flush_m(x * y);
            let got = mul_m(x, y);

            assert!(
                got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                "{x:e} * {y:e}: got {got:e}, want {want:e}"
            );
        }
    }

    /// COR-4: min and max are symmetric: a NaN operand yields the other operand, and
    /// -0 < +0 (IEEE 754-2019 minimumNumber / maximumNumber).
    #[test]
    fn min_and_max_do_not_depend_on_operand_order() {
        let pairs = [
            (f32::NAN, 1.0),
            (f32::NAN, -2.5),
            (-0.0, 0.0),
            (3.0, 3.0),
            (f32::NEG_INFINITY, f32::NAN),
            (-1.0, 2.0),
        ];

        for (x, y) in pairs {
            for (a, b) in [(min(x, y), min(y, x)), (max(x, y), max(y, x))] {
                assert_eq!(a.to_bits(), b.to_bits(), "{x} vs {y}");
            }
        }

        assert_eq!(min(f32::NAN, 1.0), 1.0);
        assert_eq!(max(1.0, f32::NAN), 1.0);
        assert!(min(f32::NAN, f32::NAN).is_nan());
        assert!(min(-0.0, 0.0).is_sign_negative());
        assert!(max(-0.0, 0.0).is_sign_positive());

        // Through the interpreter, scalar and vector forms.
        let i = |op, a, b, out| Instr {
            op,
            a,
            b,
            out,
            c: 0.0,
        };
        let mut mem = Memory::<8>::new(0);

        mem.s[1] = f32::NAN;
        mem.s[2] = 1.0;
        mem.v[1] = [f32::NAN; 8];
        mem.v[2] = [1.0; 8];

        run(
            &[
                i(OpCode::SMin, 1, 2, 3),
                i(OpCode::SMin, 2, 1, 4),
                i(OpCode::SMax, 1, 2, 5),
                i(OpCode::SMax, 2, 1, 6),
                i(OpCode::VMin, 1, 2, 3),
                i(OpCode::VMin, 2, 1, 4),
                i(OpCode::VMax, 1, 2, 5),
                i(OpCode::VMax, 2, 1, 6),
            ],
            &mut mem,
        );

        assert_eq!(mem.s[3..7], [1.0; 4]);
        assert_eq!([mem.v[3], mem.v[4], mem.v[5], mem.v[6]], [[1.0; 8]; 4]);
    }

    /// Property: arbitrary, unvalidated instructions never panic (indices are clamped).
    #[test]
    fn prop_garbage_code_never_panics() {
        let mut rng = ChaCha8Rng::seed_from_u64(6);
        let specials = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            -1e9,
            1e9,
            -0.0,
            255.0,
            4e9,
        ];

        for _ in 0..300 {
            let code: Vec<Instr> = (0..40)
                .map(|_| Instr {
                    op: ALL_OPS[rng.random_range(0..ALL_OPS.len())],
                    a: rng.random(),
                    b: rng.random(),
                    out: rng.random(),
                    c: if rng.random_bool(0.5) {
                        specials[rng.random_range(0..specials.len())]
                    } else {
                        f32::from_bits(rng.random())
                    },
                })
                .collect();

            let mut m8 = random_memory::<8>(&mut rng);
            let mut m16 = random_memory::<16>(&mut rng);
            let mut m32 = random_memory::<32>(&mut rng);

            m8.s[3] = f32::NAN;
            m16.v[2][0] = f32::INFINITY;

            for _ in 0..3 {
                run(&code, &mut m8);
                run(&code, &mut m16);
                run(&code, &mut m32);
            }
        }
    }

    /// A random valid program: either short random parts or a seed after many mutations.
    fn random_valid_program(rng: &mut ChaCha8Rng, lim: &DslLimits) -> LearnerGenome {
        let seeds = seed_programs();

        loop {
            let g = if rng.random_bool(0.5) {
                let mut part = |comp| -> Vec<Instr> {
                    let len = rng.random_range(0..=8);

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
            } else {
                let mut g = seeds[rng.random_range(0..seeds.len())].1.clone();

                for _ in 0..rng.random_range(1..60) {
                    g = mutate(&g, VariationMode::Full, rng, lim).0;
                }

                g
            };

            if g.check(lim).is_ok() {
                return g;
            }
        }
    }

    /// Property (section 6.2 driver): random valid programs never panic, every recorded
    /// output is finite and in range for its link, and the static task cost never
    /// under-estimates the cost the interpreter actually executes.
    #[test]
    fn prop_random_programs_record_finite_outputs_within_cost() {
        let lim = DslLimits::default();
        let mut rng = ChaCha8Rng::seed_from_u64(7);

        let kinds = [
            LearnKind::Regression,
            LearnKind::Binary,
            LearnKind::Multiclass { k: 3 },
            LearnKind::Multiclass { k: 8 },
        ];

        let mut aborts = 0;
        let mut completed = 0;

        for trial in 0..600 {
            let g = random_valid_program(&mut rng, &lim);
            let f = [8, 16, 32][trial % 3];
            let kind = kinds[rng.random_range(0..kinds.len())];
            let n_train = [4, 16][trial % 2];
            let steps = rng.random_range(1..=64);
            let n_valid = rng.random_range(1..=6);
            let task = testkit::task(trial as u64, f, kind, n_train, n_valid, steps);

            test_counters::start();

            let trace = testkit::drive(&g, &task, 0);
            let (_, executed) = test_counters::take();
            let planned = g.task_cost(f, steps, n_valid);

            assert!(
                executed <= planned,
                "executed {executed} > planned {planned}"
            );

            if trace.abort.is_none() {
                completed += 1;

                assert_eq!(
                    executed, planned,
                    "a full run executes exactly the planned cost"
                );
            } else {
                aborts += 1;
            }

            assert!(trace.outputs.iter().all(|x| x.is_finite()));

            let k = kind.k_out().min(f);

            for row in trace.outputs.chunks(k) {
                match kind {
                    LearnKind::Regression => {}
                    LearnKind::Binary => assert!((0.0..=1.0).contains(&row[0])),
                    LearnKind::Multiclass { .. } => {
                        let total: f32 = row.iter().sum();

                        assert!((total - 1.0).abs() < 1e-4, "softmax sums to {total}");
                    }
                }
            }
        }

        assert!(
            completed > 100 && aborts >= 5,
            "completed {completed}, aborted {aborts}"
        );
    }

    /// Different attempts see different random streams; the same attempt is repeatable.
    #[test]
    fn driver_is_deterministic_per_attempt() {
        let nn = reference_nn();
        let task = testkit::task(3, 16, LearnKind::Binary, 16, 8, 96);

        let a = testkit::drive(&nn, &task, 0);
        let b = testkit::drive(&nn, &task, 0);
        let c = testkit::drive(&nn, &task, 1);

        assert_eq!(a.outputs, b.outputs);
        assert_ne!(a.outputs, c.outputs);
        assert_eq!(a.abort, None);
    }

    #[test]
    fn counters_count_every_run() {
        let nn = reference_nn();
        let mut mem = Memory::<16>::new(0);

        test_counters::start();

        run(&nn.setup, &mut mem);
        run(&nn.predict, &mut mem);

        assert_eq!(
            test_counters::take(),
            (2, nn.setup_cost(16) + nn.predict_cost(16))
        );
        assert_eq!(test_counters::take(), (0, 0));

        assert_eq!(cost(OpCode::MatVec, 16), 512);

        let abort = testkit::drive(&nn, &testkit::task(1, 12, LearnKind::Binary, 4, 4, 4), 0);

        assert_eq!(abort.abort, Some(Abort::OverCost));
    }

    // -----------------------------------------------------------------------------------------
    // Benchmark (run: cargo test --release --lib bench_interpreter -- --ignored --nocapture)
    // -----------------------------------------------------------------------------------------

    /// Run `work` repeatedly for at least `min_secs`; returns (calls, seconds).
    fn timed(min_secs: f64, mut work: impl FnMut()) -> (u64, f64) {
        let start = std::time::Instant::now();
        let mut calls = 0u64;

        loop {
            work();
            calls += 1;

            let secs = start.elapsed().as_secs_f64();

            if secs >= min_secs {
                return (calls, secs);
            }
        }
    }

    /// Instructions executed by one full task run of `g`.
    fn task_instructions(g: &LearnerGenome, steps: usize, n_valid: usize) -> u64 {
        let setup = g.setup.len() as u64;
        let step = (g.predict.len() + g.learn.len()) as u64;
        let valid = (3 * n_valid * g.predict.len()) as u64;

        setup + steps as u64 * step + valid
    }

    fn bench_programs(label: &str, programs: &[LearnerGenome], f: usize, secs: f64) {
        let (steps, n_valid) = (384, 64);
        let task = testkit::task(42, f, LearnKind::Binary, 64, n_valid, steps);
        let mut next = 0;
        let mut flops = 0u64;
        let mut instrs = 0u64;

        let (calls, elapsed) = timed(secs, || {
            let g = &programs[next % programs.len()];

            next += 1;

            let trace = testkit::drive(g, &task, 0);
            let done = trace.steps_done;

            flops += g.task_cost(f, done, if trace.abort.is_none() { n_valid } else { 0 });
            instrs += task_instructions(g, done, if trace.abort.is_none() { n_valid } else { 0 });
        });

        println!(
            "{label:<28} F = {f:>2}: {:>9.0} tasks/s  {:>6.2} M instr/s  {:>5.2} GFLOP/s counted",
            calls as f64 / elapsed,
            instrs as f64 / elapsed / 1e6,
            flops as f64 / elapsed / 1e9,
        );
    }

    fn bench_ops<const F: usize>(secs: f64) {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let template = random_memory::<F>(&mut rng);
        let mut rows = Vec::new();

        for op in ALL_OPS {
            let s = sig(op);

            // Inputs from register 0/1, output to register 2/3: values never feed back.
            let ins = Instr {
                op,
                a: 0,
                b: if s.b == Slot::None { 0 } else { 1 },
                out: if s.out == Slot::M { 2 } else { 3 },
                c: match s.c {
                    ConstKind::None => 0.0,
                    ConstKind::Index => 5.0,
                    _ => 0.5,
                },
            };
            let code = vec![ins; 32];
            let mut mem = template.clone();

            let (calls, elapsed) = timed(secs, || run(&code, &mut mem));
            let ns = elapsed * 1e9 / (calls as f64 * 32.0);
            let counted = cost(op, F as u64) as f64;

            rows.push((op, ns, counted / ns));
        }

        let mut rates: Vec<f64> = rows.iter().map(|r| r.2).collect();

        rates.sort_by(f64::total_cmp);

        let median = rates[rates.len() / 2];

        println!("per-op timing at F = {F} (median {median:.2} counted FLOP/ns):");

        for (op, ns, rate) in rows {
            let flag = if rate < median / 4.0 {
                "  <- slow per counted FLOP"
            } else {
                ""
            };

            println!("  {op:<12?} {ns:>8.1} ns  {rate:>6.2} FLOP/ns{flag}");
        }
    }

    #[test]
    #[ignore = "benchmark: cargo test --release --lib bench_interpreter -- --ignored --nocapture"]
    fn bench_interpreter() {
        let secs = if cfg!(debug_assertions) { 0.5 } else { 1.0 };
        let lim = DslLimits::default();
        let nn = vec![reference_nn()];

        println!(
            "build: {}",
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
        );

        for f in [8, 16, 32] {
            bench_programs("nn2-sgd", &nn, f, secs);
        }

        // Random search-like programs: initial members 64.. (3 mutations) and deeper walks.
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut mixed: Vec<LearnerGenome> = (64..128)
            .map(|i| initial_member(i, 1, VariationMode::Full, &lim))
            .collect();

        for i in 0..64 {
            let mut g = seed_programs()[i % 4].1.clone();

            for _ in 0..30 {
                g = mutate(&g, VariationMode::Full, &mut rng, &lim).0;
            }

            mixed.push(g);
        }

        for f in [16, 32] {
            bench_programs("mutated programs (128)", &mixed, f, secs);
        }

        // Four threads, nn2-sgd at F = 16 (AC-2: >= 3 000 tasks/s).
        let task = testkit::task(42, 16, LearnKind::Binary, 64, 64, 384);
        let start = std::time::Instant::now();
        let total: u64 = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| scope.spawn(|| timed(secs, || drop(testkit::drive(&nn[0], &task, 0))).0))
                .collect();

            handles.into_iter().map(|h| h.join().unwrap_or(0)).sum()
        });
        let elapsed = start.elapsed().as_secs_f64();

        println!(
            "nn2-sgd F = 16, 4 threads:  {:>9.0} tasks/s  {:>5.2} GFLOP/s counted",
            total as f64 / elapsed,
            total as f64 * nn[0].task_cost(16, 384, 64) as f64 / elapsed / 1e9
        );

        bench_ops::<16>(secs / 10.0);
        bench_ops::<32>(secs / 10.0);
    }
}
