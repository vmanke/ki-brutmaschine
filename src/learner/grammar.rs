//! Expression-grammar task families: signatures, public pools, disjoint pools, instances.
//!
//! Owner: WP2. Contract section 5.8. Pure: no I/O.
//!
//! Two rules make the pools well defined (a WP2 deviation from contract 5.8; lead: confirm):
//! - `signature` is canonical: commutative operands are sorted and the projection labels
//!   t0..t2 are renamed to the smallest form, because the projections are exchangeable.
//!   Expressions that only collapse (abs(abs(x)), max(x, x), ...) are never sampled.
//! - Every pool holds distinct signatures. The public pools take their lengths from
//!   P(L) ∝ 2^-L with each length capped at a quarter of the signatures that exist, so
//!   the hidden pool (40 families, same length histogram) always has unseen signatures
//!   left. Without the cap, length 2 (4 signatures) and 3 (13) would run out.
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::learner::families::{self, InstanceParams};
use crate::learner::stats::{normal_f64, range_f64, shuffle, uniform_below};
use crate::learner::types::{
    FamilyId, Group, Labels, LearnKind, STEPS_PER_TASK, TaskInstance, Tier,
};
use crate::util::{BrutError, Result, derive_seed};

/// Expression over projections t_j = u_j · z (j < 3) of a 3-dim latent z.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    T(u8),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
    /// sin(w * e), w drawn per instance from [1, 3].
    Sin(Box<Expr>),
    Abs(Box<Expr>),
    Square(Box<Expr>),
    /// 1[e > t], t drawn per instance from N(0, 0.5²).
    Step(Box<Expr>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct GrammarFamily {
    pub id: FamilyId,
    pub expr: Expr,
    /// Canonical structure without constants, e.g. "add(mul(t0,t1),sin(t2))".
    pub signature: String,
    pub kind: LearnKind,
}

/// Latent dimension and number of projections.
pub const LATENT_DIM: usize = 3;
/// Deepest root-to-leaf path in edges.
pub const MAX_DEPTH: usize = 3;
/// Most nodes an expression of depth <= 3 can have (a full binary tree).
pub const MAX_EXPR_LEN: usize = 15;
/// Shortest and longest target length of the pools.
pub const MIN_LEN: usize = 2;
pub const MAX_LEN: usize = 9;
/// Pool sizes.
pub const SEARCH_POOL_SIZE: usize = 64;
pub const SELECT_POOL_SIZE: usize = 32;
/// Extra draws per instance for quantiles (classification) or standardisation (regression).
pub const QUANTILE_DRAWS: usize = 4_096;
/// Regression target noise (after standardisation).
pub const REGRESSION_NOISE: f64 = 0.1;
/// A pool fails when one family needs more than this many rejected draws.
pub const MAX_REJECTED_DRAWS: usize = 10_000;

/// Seed tags of the public pools (contract section 6.1).
const SEARCH_POOL_TAG: &str = "grammar/search/learner-v1";
const SELECT_POOL_TAG: &str = "grammar/select/learner-v1";

/// Most samples drawn to find one non-degenerate expression of a given length.
const MAX_DEGENERATE_RETRIES: usize = 1_000;

/// Per length L (index), the most families one public pool may have: a quarter of the
/// canonical, non-degenerate signatures of that length (4, 13, 63, 252, 828 for L = 2..6;
/// a unit test recounts them). Lengths 7..9 have thousands, so they are not capped.
const PUBLIC_CAP: [usize; MAX_LEN + 1] =
    [0, 0, 1, 3, 15, 63, 207, usize::MAX, usize::MAX, usize::MAX];

const UNARY_OPS: usize = 4;
const BINARY_OPS: usize = 3;

/// Identity relabelling of t0..t2.
const IDENTITY: [u8; LATENT_DIM] = [0, 1, 2];

/// All relabellings of t0..t2.
const PERMUTATIONS: [[u8; LATENT_DIM]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// Canonical signature: the smallest commutative-sorted form over all relabellings of
/// t0..t2. Equal for expressions that differ only by operand order of add/mul/max or by
/// renaming projections (which are exchangeable), so a "new" signature is a new structure.
pub fn signature(e: &Expr) -> String {
    PERMUTATIONS
        .iter()
        .map(|perm| form(e, perm))
        .min()
        .unwrap_or_default()
}

/// Commutative-sorted text form of `e` with projections renamed by `perm`.
fn form(e: &Expr, perm: &[u8; LATENT_DIM]) -> String {
    match e {
        Expr::T(j) => {
            let j = perm.get(*j as usize).copied().unwrap_or(*j);

            format!("t{j}")
        }
        Expr::Add(a, b) => binary_form("add", a, b, perm),
        Expr::Mul(a, b) => binary_form("mul", a, b, perm),
        Expr::Max(a, b) => binary_form("max", a, b, perm),
        Expr::Sin(c) => format!("sin({})", form(c, perm)),
        Expr::Abs(c) => format!("abs({})", form(c, perm)),
        Expr::Square(c) => format!("square({})", form(c, perm)),
        Expr::Step(c) => format!("step({})", form(c, perm)),
    }
}

fn binary_form(name: &str, a: &Expr, b: &Expr, perm: &[u8; LATENT_DIM]) -> String {
    let x = form(a, perm);
    let y = form(b, perm);

    if x <= y {
        format!("{name}({x},{y})")
    } else {
        format!("{name}({y},{x})")
    }
}

/// Semantic key of a family: its kind and the signature after two rewrites that leave the
/// task distribution exactly unchanged (SCI-R4). Families with equal keys are twins.
///
/// (a) In a scale-free context (only mul, abs and square between the node and the root), a
///     sum of distinct projections that occur nowhere else is one projection: the direction
///     of u_i + u_j is uniform and independent of the other projections, and its positive
///     length cancels in the regression standardisation or the classification quantiles.
/// (b) For classification, square at the root is abs: a strictly increasing map of |x|
///     keeps every quantile label.
///
/// Other exact equivalences may exist, so equal signatures and equal keys are only a lower
/// bound on twins.
pub fn semantic_key(e: &Expr, kind: LearnKind) -> String {
    let mut uses = [0usize; LATENT_DIM];

    count_leaves(e, &mut uses);

    let mut folded = fold_sums(e, true, &uses);

    if kind != LearnKind::Regression
        && let Expr::Square(c) = folded
    {
        folded = Expr::Abs(c);
    }

    let tag = match kind {
        LearnKind::Regression => "regression".to_string(),
        LearnKind::Binary => "binary".to_string(),
        LearnKind::Multiclass { k } => format!("multiclass{k}"),
    };

    format!("{tag}:{}", signature(&folded))
}

/// Semantic keys of a pool (each under its own kind).
pub fn semantic_keys(pool: &[GrammarFamily]) -> BTreeSet<String> {
    pool.iter().map(|f| semantic_key(&f.expr, f.kind)).collect()
}

/// Semantic keys of both public pools (the semantic `exclude` set of the hidden pool).
pub fn public_semantic_keys() -> BTreeSet<String> {
    let mut all = semantic_keys(public_pool_slice(Tier::Search));

    all.extend(semantic_keys(public_pool_slice(Tier::Select)));

    all
}

fn count_leaves(e: &Expr, uses: &mut [usize; LATENT_DIM]) {
    match e {
        Expr::T(j) => {
            if let Some(n) = uses.get_mut(*j as usize) {
                *n += 1;
            }
        }
        Expr::Add(a, b) | Expr::Mul(a, b) | Expr::Max(a, b) => {
            count_leaves(a, uses);
            count_leaves(b, uses);
        }
        Expr::Sin(c) | Expr::Abs(c) | Expr::Square(c) | Expr::Step(c) => count_leaves(c, uses),
    }
}

/// The leaves of a pure sum of projections (None if any node is not add or a leaf).
fn sum_leaves(e: &Expr, out: &mut Vec<u8>) -> bool {
    match e {
        Expr::T(j) => {
            out.push(*j);

            true
        }
        Expr::Add(a, b) => sum_leaves(a, out) && sum_leaves(b, out),
        _ => false,
    }
}

/// Rewrite (a) of `semantic_key`; `free` is true while only mul, abs and square lie above.
fn fold_sums(e: &Expr, free: bool, uses: &[usize; LATENT_DIM]) -> Expr {
    let boxed = |x: Expr| Box::new(x);

    match e {
        Expr::T(j) => Expr::T(*j),
        Expr::Add(a, b) => {
            let mut leaves = Vec::new();
            let single = sum_leaves(e, &mut leaves)
                && leaves
                    .iter()
                    .all(|&j| uses.get(j as usize).copied() == Some(1));

            if free && single {
                Expr::T(leaves[0])
            } else {
                Expr::Add(
                    boxed(fold_sums(a, false, uses)),
                    boxed(fold_sums(b, false, uses)),
                )
            }
        }
        Expr::Mul(a, b) => Expr::Mul(
            boxed(fold_sums(a, free, uses)),
            boxed(fold_sums(b, free, uses)),
        ),
        Expr::Max(a, b) => Expr::Max(
            boxed(fold_sums(a, false, uses)),
            boxed(fold_sums(b, false, uses)),
        ),
        Expr::Abs(c) => Expr::Abs(boxed(fold_sums(c, free, uses))),
        Expr::Square(c) => Expr::Square(boxed(fold_sums(c, free, uses))),
        Expr::Sin(c) => Expr::Sin(boxed(fold_sums(c, false, uses))),
        Expr::Step(c) => Expr::Step(boxed(fold_sums(c, false, uses))),
    }
}

/// Node count.
pub fn length(e: &Expr) -> usize {
    match e {
        Expr::T(_) => 1,
        Expr::Add(a, b) | Expr::Mul(a, b) | Expr::Max(a, b) => 1 + length(a) + length(b),
        Expr::Sin(c) | Expr::Abs(c) | Expr::Square(c) | Expr::Step(c) => 1 + length(c),
    }
}

/// Longest root-to-leaf path in edges (a single leaf has depth 0).
pub fn depth(e: &Expr) -> usize {
    match e {
        Expr::T(_) => 0,
        Expr::Add(a, b) | Expr::Mul(a, b) | Expr::Max(a, b) => 1 + depth(a).max(depth(b)),
        Expr::Sin(c) | Expr::Abs(c) | Expr::Square(c) | Expr::Step(c) => 1 + depth(c),
    }
}

/// True when the expression contains a part that collapses to a shorter one:
/// abs of a non-negative value (abs, square, step), square(abs(x)), square(step(x)),
/// step(step(x)), or a binary operation on two equal operands (max(x, x) = x,
/// add(x, x) = 2x, mul(x, x) = square(x)).
pub fn is_degenerate(e: &Expr) -> bool {
    match e {
        Expr::T(_) => false,
        Expr::Abs(c) => {
            matches!(**c, Expr::Abs(_) | Expr::Square(_) | Expr::Step(_)) || is_degenerate(c)
        }
        Expr::Square(c) => matches!(**c, Expr::Abs(_) | Expr::Step(_)) || is_degenerate(c),
        Expr::Step(c) => matches!(**c, Expr::Step(_)) || is_degenerate(c),
        Expr::Sin(c) => is_degenerate(c),
        Expr::Add(a, b) | Expr::Mul(a, b) | Expr::Max(a, b) => {
            form(a, &IDENTITY) == form(b, &IDENTITY) || is_degenerate(a) || is_degenerate(b)
        }
    }
}

/// counts[n][d]: number of expression trees (ordered children, labelled leaves) with
/// exactly n nodes and depth <= d.
fn tree_counts() -> [[u64; MAX_DEPTH + 1]; MAX_EXPR_LEN + 1] {
    let mut counts = [[0u64; MAX_DEPTH + 1]; MAX_EXPR_LEN + 1];

    counts[1] = [LATENT_DIM as u64; MAX_DEPTH + 1];

    for n in 2..=MAX_EXPR_LEN {
        for d in 1..=MAX_DEPTH {
            let mut total = UNARY_OPS as u64 * counts[n - 1][d - 1];

            for k in 1..n.saturating_sub(1) {
                total += BINARY_OPS as u64 * counts[k][d - 1] * counts[n - 1 - k][d - 1];
            }

            counts[n][d] = total;
        }
    }

    counts
}

/// Uniform tree with exactly n nodes and depth <= d (requires counts[n][d] > 0).
fn sample_tree(
    rng: &mut ChaCha8Rng,
    n: usize,
    d: usize,
    counts: &[[u64; MAX_DEPTH + 1]; MAX_EXPR_LEN + 1],
) -> Expr {
    if n <= 1 || d == 0 {
        return Expr::T(uniform_below(rng, LATENT_DIM as u64) as u8);
    }

    let child = counts[n - 1][d - 1];
    let unary = UNARY_OPS as u64 * child;
    let mut r = uniform_below(rng, counts[n][d]);

    if r < unary {
        let op = r / child.max(1);
        let c = Box::new(sample_tree(rng, n - 1, d - 1, counts));

        return match op {
            0 => Expr::Sin(c),
            1 => Expr::Abs(c),
            2 => Expr::Square(c),
            _ => Expr::Step(c),
        };
    }

    r -= unary;

    for k in 1..n - 1 {
        let block = counts[k][d - 1] * counts[n - 1 - k][d - 1];

        if r < BINARY_OPS as u64 * block {
            let op = r / block.max(1);
            let a = Box::new(sample_tree(rng, k, d - 1, counts));
            let b = Box::new(sample_tree(rng, n - 1 - k, d - 1, counts));

            return match op {
                0 => Expr::Add(a, b),
                1 => Expr::Mul(a, b),
                _ => Expr::Max(a, b),
            };
        }

        r -= BINARY_OPS as u64 * block;
    }

    // Unreachable while counts are consistent; a leaf keeps the function total.
    Expr::T(0)
}

/// Sample an expression of exactly `target_len` nodes, depth <= 3 (Err if impossible).
/// Uniform over the non-degenerate trees of that size (ordered children, labelled leaves).
pub fn sample_expr(rng: &mut ChaCha8Rng, target_len: usize) -> Result<Expr> {
    if target_len == 0 || target_len > MAX_EXPR_LEN {
        return Err(BrutError::Invalid(format!(
            "grammar: no expression of {target_len} nodes with depth <= {MAX_DEPTH}"
        )));
    }

    let counts = tree_counts();

    for _ in 0..MAX_DEGENERATE_RETRIES {
        let e = sample_tree(rng, target_len, MAX_DEPTH, &counts);

        if !is_degenerate(&e) {
            return Ok(e);
        }
    }

    Err(BrutError::Invalid(format!(
        "grammar: no non-degenerate expression of {target_len} nodes found"
    )))
}

/// Target length distribution: L in 2..=9 with P(L) ∝ 2^-L.
/// The pools use the same weights through `public_lengths`, which adds capacity caps.
pub fn sample_length(rng: &mut ChaCha8Rng) -> usize {
    // Weights 2^(9 - L) for L = 2..9 sum to 255.
    let mut r = uniform_below(rng, 255);

    for len in MIN_LEN..=MAX_LEN {
        let weight = 1u64 << (MAX_LEN - len);

        if r < weight {
            return len;
        }

        r -= weight;
    }

    MAX_LEN
}

/// Lengths of a public pool of `n` families: P(L) ∝ 2^-L over 2..=9, water-filled under
/// `PUBLIC_CAP` (a capped length keeps its cap; the rest is shared ∝ 2^-L among the
/// others), then largest-remainder rounding. Exact integer arithmetic; ascending order.
pub fn public_lengths(n: usize) -> Result<Vec<usize>> {
    let mut fixed: [Option<usize>; MAX_LEN + 1] = [None; MAX_LEN + 1];

    loop {
        let free: Vec<usize> = (MIN_LEN..=MAX_LEN)
            .filter(|&l| fixed[l].is_none())
            .collect();
        let used: usize = fixed.iter().flatten().sum();
        let remaining = n.saturating_sub(used) as u128;
        let total: u128 = free.iter().map(|&l| length_weight(l)).sum();

        if free.is_empty() {
            if remaining > 0 {
                return Err(BrutError::Invalid(format!(
                    "grammar: a public pool of {n} families exceeds the length caps"
                )));
            }

            break;
        }

        let mut capped = false;

        for &l in &free {
            if remaining * length_weight(l) > PUBLIC_CAP[l] as u128 * total {
                fixed[l] = Some(PUBLIC_CAP[l]);
                capped = true;
            }
        }

        if !capped {
            let ideal: Vec<(usize, u128, u128)> = free
                .iter()
                .map(|&l| (l, remaining * length_weight(l), total))
                .collect();

            for (l, count) in largest_remainder(&ideal, remaining as usize) {
                fixed[l] = Some(count);
            }

            break;
        }
    }

    let mut lengths = Vec::with_capacity(n);

    for (l, count) in fixed.iter().enumerate() {
        lengths.extend(std::iter::repeat_n(l, count.unwrap_or(0)));
    }

    Ok(lengths)
}

/// 2^(9 - L): the P(L) ∝ 2^-L weight on integers.
fn length_weight(len: usize) -> u128 {
    1u128 << (MAX_LEN - len)
}

/// Round the shares num/den (one per key, summing to `n`) to integers: floors first,
/// then +1 for the largest remainders (ties: smaller key first).
fn largest_remainder(shares: &[(usize, u128, u128)], n: usize) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = shares
        .iter()
        .map(|&(key, num, den)| (key, (num / den.max(1)) as usize))
        .collect();
    let assigned: usize = out.iter().map(|&(_, c)| c).sum();
    let mut order: Vec<usize> = (0..shares.len()).collect();

    // Remainders share no common denominator, so compare r_a / d_a with r_b / d_b exactly.
    order.sort_by(|&a, &b| {
        let (ka, na, da) = shares[a];
        let (kb, nb, db) = shares[b];
        let ra = na % da.max(1);
        let rb = nb % db.max(1);

        (rb * da.max(1)).cmp(&(ra * db.max(1))).then(ka.cmp(&kb))
    });

    for &i in order.iter().take(n.saturating_sub(assigned)) {
        out[i].1 += 1;
    }

    out
}

/// Pool of `n` families from `seed` whose signatures avoid `exclude` and whose length
/// histogram equals the histogram of `match_lengths` (scaled to n, largest-remainder
/// rounding, ties to the shorter length). Signatures are distinct inside the pool.
/// Err after 10 000 rejected draws for one family, for empty `match_lengths` (n > 0) and
/// for lengths outside 1..=15.
///
/// Draw order: the length list is shuffled, then per family an expression (rejecting
/// excluded or repeated signatures) and its kind. Ids are "<prefix>/<index:03>".
pub fn disjoint_pool(
    seed: u64,
    n: usize,
    exclude: &BTreeSet<String>,
    match_lengths: &[usize],
    prefix: &str,
) -> Result<Vec<GrammarFamily>> {
    draw_pool(seed, n, exclude, None, match_lengths, prefix)
}

/// `disjoint_pool` that also rejects twins (SCI-R4): a family whose semantic key (see
/// `semantic_key`) is in `exclude_keys` or already in the pool is redrawn, expression and
/// kind together, and counts as a rejected draw. Used for the hidden pool only; the public
/// pools keep their definition (and their hashes).
pub fn disjoint_pool_semantic(
    seed: u64,
    n: usize,
    exclude: &BTreeSet<String>,
    exclude_keys: &BTreeSet<String>,
    match_lengths: &[usize],
    prefix: &str,
) -> Result<Vec<GrammarFamily>> {
    draw_pool(seed, n, exclude, Some(exclude_keys), match_lengths, prefix)
}

/// The draw loop of both pool builders; `exclude_keys` switches the twin check on.
fn draw_pool(
    seed: u64,
    n: usize,
    exclude: &BTreeSet<String>,
    exclude_keys: Option<&BTreeSet<String>>,
    match_lengths: &[usize],
    prefix: &str,
) -> Result<Vec<GrammarFamily>> {
    if n == 0 {
        return Ok(Vec::new());
    }

    if match_lengths.is_empty() {
        return Err(BrutError::Invalid(
            "grammar: disjoint_pool needs a length histogram".into(),
        ));
    }

    let mut lengths = scaled_lengths(match_lengths, n);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut used: BTreeSet<String> = BTreeSet::new();
    let mut used_keys: BTreeSet<String> = BTreeSet::new();
    let mut pool = Vec::with_capacity(n);

    shuffle(&mut rng, &mut lengths);

    for (i, &len) in lengths.iter().enumerate() {
        let mut rejected = 0usize;

        loop {
            let reject = |rejected: &mut usize| -> Result<()> {
                *rejected += 1;

                if *rejected >= MAX_REJECTED_DRAWS {
                    return Err(BrutError::Invalid(format!(
                        "grammar: no unused signature of length {len} after \
                         {MAX_REJECTED_DRAWS} draws"
                    )));
                }

                Ok(())
            };

            let expr = sample_expr(&mut rng, len)?;
            let sig = signature(&expr);

            if exclude.contains(&sig) || used.contains(&sig) {
                reject(&mut rejected)?;

                continue;
            }

            let kind = sample_kind(&mut rng);

            if let Some(keys) = exclude_keys {
                let key = semantic_key(&expr, kind);

                if keys.contains(&key) || !used_keys.insert(key) {
                    reject(&mut rejected)?;

                    continue;
                }
            }

            used.insert(sig.clone());
            pool.push(GrammarFamily {
                id: FamilyId(format!("{prefix}/{i:03}")),
                expr,
                signature: sig,
                kind,
            });

            break;
        }
    }

    Ok(pool)
}

/// The histogram of `lengths` scaled to n entries (largest remainder), as an ascending list.
fn scaled_lengths(lengths: &[usize], n: usize) -> Vec<usize> {
    let mut hist: BTreeMap<usize, u128> = BTreeMap::new();

    for &l in lengths {
        *hist.entry(l).or_insert(0) += 1;
    }

    let m = lengths.len() as u128;
    let shares: Vec<(usize, u128, u128)> =
        hist.iter().map(|(&l, &h)| (l, n as u128 * h, m)).collect();
    let mut out = Vec::with_capacity(n);

    for (l, count) in largest_remainder(&shares, n) {
        out.extend(std::iter::repeat_n(l, count));
    }

    out
}

/// Regression or classification with equal probability; classification is Binary or
/// Multiclass with k in 3..=4, each equally likely.
fn sample_kind(rng: &mut ChaCha8Rng) -> LearnKind {
    match uniform_below(rng, 6) {
        0..=2 => LearnKind::Regression,
        3 => LearnKind::Binary,
        4 => LearnKind::Multiclass { k: 3 },
        _ => LearnKind::Multiclass { k: 4 },
    }
}

/// Both public pools, built once per process (deterministic).
fn public_pools() -> &'static (Vec<GrammarFamily>, Vec<GrammarFamily>) {
    static POOLS: OnceLock<(Vec<GrammarFamily>, Vec<GrammarFamily>)> = OnceLock::new();

    POOLS.get_or_init(build_public_pools)
}

fn build_public_pools() -> (Vec<GrammarFamily>, Vec<GrammarFamily>) {
    let search = public_lengths(SEARCH_POOL_SIZE)
        .and_then(|lengths| {
            disjoint_pool(
                derive_seed(0, SEARCH_POOL_TAG),
                SEARCH_POOL_SIZE,
                &BTreeSet::new(),
                &lengths,
                "grammar/search",
            )
        })
        .unwrap_or_default();

    let exclude = signatures(&search);

    let select = public_lengths(SELECT_POOL_SIZE)
        .and_then(|lengths| {
            disjoint_pool(
                derive_seed(0, SELECT_POOL_TAG),
                SELECT_POOL_SIZE,
                &exclude,
                &lengths,
                "grammar/select",
            )
        })
        .unwrap_or_default();

    (search, select)
}

/// Public pools: Search = 64 families from derive_seed(0, "grammar/search/learner-v1");
/// Select = 32 families from derive_seed(0, "grammar/select/learner-v1") with signatures
/// disjoint from Search. Kind: Regression or Classification (Binary or Multiclass k in 3..=4)
/// with equal probability. Panics never; deterministic. Other tiers give an empty list.
pub fn public_pool(tier: Tier) -> Vec<GrammarFamily> {
    public_pool_slice(tier).to_vec()
}

/// Borrowed form of `public_pool` (no copy).
pub fn public_pool_slice(tier: Tier) -> &'static [GrammarFamily] {
    let (search, select) = public_pools();

    match tier {
        Tier::Search => search,
        Tier::Select => select,
        _ => &[],
    }
}

/// Signatures of a pool.
pub fn signatures(pool: &[GrammarFamily]) -> BTreeSet<String> {
    pool.iter().map(|f| f.signature.clone()).collect()
}

/// Signatures of both public pools (the `exclude` set of the hidden pool).
pub fn public_signatures() -> BTreeSet<String> {
    let mut all = signatures(public_pool_slice(Tier::Search));

    all.extend(signatures(public_pool_slice(Tier::Select)));

    all
}

/// One postfix instruction of a compiled expression, constants drawn.
#[derive(Clone, Copy, Debug)]
enum Node {
    T(usize),
    Add,
    Mul,
    Max,
    Sin(f64),
    Abs,
    Square,
    Step(f64),
}

/// Compile to postfix; per-instance constants are drawn in pre-order (node, then children).
/// Leaf labels >= 3 (only possible in hand-made data) wrap around modulo 3.
fn compile(e: &Expr, rng: &mut ChaCha8Rng, out: &mut Vec<Node>) {
    match e {
        Expr::T(j) => out.push(Node::T(*j as usize % LATENT_DIM)),
        Expr::Add(a, b) | Expr::Mul(a, b) | Expr::Max(a, b) => {
            compile(a, rng, out);
            compile(b, rng, out);

            out.push(match e {
                Expr::Add(..) => Node::Add,
                Expr::Mul(..) => Node::Mul,
                _ => Node::Max,
            });
        }
        Expr::Sin(c) => {
            let w = range_f64(rng, 1.0, 3.0);

            compile(c, rng, out);
            out.push(Node::Sin(w));
        }
        Expr::Step(c) => {
            let t = 0.5 * normal_f64(rng);

            compile(c, rng, out);
            out.push(Node::Step(t));
        }
        Expr::Abs(c) => {
            compile(c, rng, out);
            out.push(Node::Abs);
        }
        Expr::Square(c) => {
            compile(c, rng, out);
            out.push(Node::Square);
        }
    }
}

/// Evaluate a compiled expression on the projections t. A malformed program gives 0.
fn eval(program: &[Node], t: &[f64; LATENT_DIM]) -> f64 {
    let mut stack = [0.0f64; MAX_EXPR_LEN + 1];
    let mut sp = 0usize;

    for node in program {
        match *node {
            Node::T(j) => {
                if sp >= stack.len() {
                    return 0.0;
                }

                stack[sp] = t[j % LATENT_DIM];
                sp += 1;
            }
            Node::Add | Node::Mul | Node::Max => {
                if sp < 2 {
                    return 0.0;
                }

                let b = stack[sp - 1];
                let a = stack[sp - 2];

                sp -= 1;
                stack[sp - 1] = match *node {
                    Node::Add => a + b,
                    Node::Mul => a * b,
                    _ => a.max(b),
                };
            }
            Node::Sin(_) | Node::Abs | Node::Square | Node::Step(_) => {
                if sp == 0 {
                    return 0.0;
                }

                let x = stack[sp - 1];

                stack[sp - 1] = match *node {
                    Node::Sin(w) => (w * x).sin(),
                    Node::Abs => x.abs(),
                    Node::Square => x * x,
                    Node::Step(t) => {
                        if x > t {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    _ => x,
                };
            }
        }
    }

    if sp == 1 { stack[0] } else { 0.0 }
}

/// How f(z) becomes a label: standardisation (regression) or quantile edges (classes).
enum Target {
    Regression { mean: f64, sd: f64 },
    Classes { edges: Vec<f64> },
}

impl Target {
    fn from_reference(kind: LearnKind, mut reference: Vec<f64>) -> Target {
        let n = reference.len().max(1) as f64;

        match kind {
            LearnKind::Regression => {
                let mean = reference.iter().sum::<f64>() / n;
                let var = reference
                    .iter()
                    .map(|v| (v - mean) * (v - mean))
                    .sum::<f64>()
                    / n;
                let sd = var.sqrt();

                Target::Regression {
                    mean,
                    sd: if sd > 1e-9 { sd } else { 1.0 },
                }
            }
            LearnKind::Binary | LearnKind::Multiclass { .. } => {
                let k = kind_classes(kind);

                reference.sort_by(f64::total_cmp);

                let edges = (1..k)
                    .filter_map(|i| reference.get(i * reference.len() / k).copied())
                    .collect();

                Target::Classes { edges }
            }
        }
    }

    /// Label of value v; regression draws one noise sample.
    fn label(&self, v: f64, rng: &mut ChaCha8Rng) -> f64 {
        match self {
            Target::Regression { mean, sd } => (v - mean) / sd + REGRESSION_NOISE * normal_f64(rng),
            Target::Classes { edges } => edges.iter().filter(|&&e| v > e).count() as f64,
        }
    }
}

/// Number of classes (Binary = 2; Regression = 1).
fn kind_classes(kind: LearnKind) -> usize {
    match kind {
        LearnKind::Regression => 1,
        LearnKind::Binary => 2,
        LearnKind::Multiclass { k } => k as usize,
    }
}

/// One latent point z ~ N(0, I_3) and its projections t = U z.
fn draw_projected(
    rng: &mut ChaCha8Rng,
    u: &[[f64; LATENT_DIM]; LATENT_DIM],
) -> ([f64; LATENT_DIM], [f64; LATENT_DIM]) {
    let mut z = [0.0; LATENT_DIM];

    for zi in z.iter_mut() {
        *zi = normal_f64(rng);
    }

    let mut t = [0.0; LATENT_DIM];

    for (tj, uj) in t.iter_mut().zip(u) {
        *tj = uj.iter().zip(&z).map(|(a, b)| a * b).sum();
    }

    (z, t)
}

/// Materialise one instance: classification = balanced quantile buckets of f(z) (quantiles
/// from 4 096 extra draws of the same instance), regression = standardised f(z) + N(0, 0.1²).
///
/// Projections u_j are independent random unit vectors in R^3; z is embedded like the hand
/// families (random orthonormal Q, Gaussian filler, train-split feature standardisation).
/// Unsupported f or out-of-range sizes are sanitised (f -> 16, sizes clamped to
/// 1..=MAX_ROWS); `level` is ignored except in the task id. Streams: "gen/params",
/// "gen/reference", "gen/train", "gen/valid" under the instance seed.
pub fn grammar_instance(fam: &GrammarFamily, seed: u64, p: &InstanceParams) -> TaskInstance {
    let p = families::sanitize_sizes(p);
    let f = p.f;
    let kind = fam.kind;

    let mut params = ChaCha8Rng::seed_from_u64(derive_seed(seed, "gen/params"));
    let mut u = [[0.0; LATENT_DIM]; LATENT_DIM];

    for uj in u.iter_mut() {
        uj.copy_from_slice(&families::unit_vector(&mut params, LATENT_DIM));
    }

    let mut program = Vec::with_capacity(MAX_EXPR_LEN);

    compile(&fam.expr, &mut params, &mut program);

    let q = families::orthonormal(&mut params, f);

    let mut reference_rng = ChaCha8Rng::seed_from_u64(derive_seed(seed, "gen/reference"));
    let reference: Vec<f64> = (0..QUANTILE_DRAWS)
        .map(|_| eval(&program, &draw_projected(&mut reference_rng, &u).1))
        .collect();
    let target = Target::from_reference(kind, reference);

    let make_split = |tag: &str, n: usize| {
        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(seed, tag));
        let mut x = vec![0.0; n * f];
        let mut y = Vec::with_capacity(n);
        let mut filler = vec![0.0; f];

        for row in x.chunks_exact_mut(f) {
            let (z, t) = draw_projected(&mut rng, &u);

            families::embed(&q, f, &z, &mut rng, &mut filler, row);
            y.push(target.label(eval(&program, &t), &mut rng));
        }

        (x, y)
    };

    let (x_train, y_train) = make_split("gen/train", p.n_train);
    let (x_valid, y_valid) = make_split("gen/valid", p.n_valid);
    let (x_train, x_valid) = families::standardize_features(&x_train, &x_valid, f);

    let (y_train, y_valid) = match kind {
        LearnKind::Regression => (
            Labels::Values(y_train.iter().map(|&v| v as f32).collect()),
            Labels::Values(y_valid.iter().map(|&v| v as f32).collect()),
        ),
        LearnKind::Binary | LearnKind::Multiclass { .. } => (
            Labels::Classes(y_train.iter().map(|&v| v as u8).collect()),
            Labels::Classes(y_valid.iter().map(|&v| v as u8).collect()),
        ),
    };

    TaskInstance {
        id: families::instance_id(&fam.id.0, seed, &p),
        family: fam.id.clone(),
        group: Group::Grammar,
        f,
        kind,
        x_train,
        y_train,
        y_train_late: None,
        drift_at: None,
        x_valid,
        y_valid,
        n_train: p.n_train,
        n_valid: p.n_valid,
        steps: STEPS_PER_TASK,
        seed,
        changed_mask: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(j: u8) -> Box<Expr> {
        Box::new(Expr::T(j))
    }

    /// All trees with n nodes and depth <= d (ordered children, labelled leaves).
    fn enumerate(n: usize, d: usize) -> Vec<Expr> {
        if n == 1 {
            return (0..3).map(Expr::T).collect();
        }

        if d == 0 {
            return Vec::new();
        }

        let mut out = Vec::new();

        for c in enumerate(n - 1, d - 1) {
            let c = Box::new(c);

            out.push(Expr::Sin(c.clone()));
            out.push(Expr::Abs(c.clone()));
            out.push(Expr::Square(c.clone()));
            out.push(Expr::Step(c));
        }

        for k in 1..n.saturating_sub(1) {
            let left = enumerate(k, d - 1);
            let right = enumerate(n - 1 - k, d - 1);

            for a in &left {
                for b in &right {
                    let (a, b) = (Box::new(a.clone()), Box::new(b.clone()));

                    out.push(Expr::Add(a.clone(), b.clone()));
                    out.push(Expr::Mul(a.clone(), b.clone()));
                    out.push(Expr::Max(a, b));
                }
            }
        }

        out
    }

    fn histogram(lengths: impl Iterator<Item = usize>) -> BTreeMap<usize, usize> {
        let mut h = BTreeMap::new();

        for l in lengths {
            *h.entry(l).or_insert(0) += 1;
        }

        h
    }

    fn params(f: usize, n_train: usize) -> InstanceParams {
        InstanceParams {
            f,
            n_train,
            n_valid: 64,
            level: 1,
        }
    }

    #[test]
    fn signature_is_canonical_under_relabelling_and_commutativity() {
        let a = Expr::Add(Box::new(Expr::Sin(t(0))), Box::new(Expr::Mul(t(1), t(2))));
        let b = Expr::Add(Box::new(Expr::Mul(t(0), t(2))), Box::new(Expr::Sin(t(1))));

        assert_eq!(signature(&a), signature(&b));
        assert_eq!(signature(&a), "add(mul(t0,t1),sin(t2))");
        assert_eq!(signature(&Expr::Sin(t(2))), "sin(t0)");
        assert_ne!(
            signature(&Expr::Add(t(0), t(1))),
            signature(&Expr::Mul(t(0), t(1)))
        );

        // Different structure stays different: max(t0, t0) vs max(t0, t1).
        assert_ne!(
            signature(&Expr::Max(t(0), t(0))),
            signature(&Expr::Max(t(0), t(1)))
        );
    }

    #[test]
    fn length_and_depth() {
        let e = Expr::Add(
            Box::new(Expr::Sin(Box::new(Expr::Abs(t(0))))),
            Box::new(Expr::Mul(t(1), t(2))),
        );

        assert_eq!(length(&e), 7);
        assert_eq!(depth(&e), 3);
        assert_eq!(length(&Expr::T(1)), 1);
        assert_eq!(depth(&Expr::T(1)), 0);
    }

    #[test]
    fn degenerate_patterns_are_detected() {
        assert!(is_degenerate(&Expr::Abs(Box::new(Expr::Square(t(0))))));
        assert!(is_degenerate(&Expr::Square(Box::new(Expr::Abs(t(0))))));
        assert!(is_degenerate(&Expr::Step(Box::new(Expr::Step(t(0))))));
        assert!(is_degenerate(&Expr::Max(t(1), t(1))));
        assert!(is_degenerate(&Expr::Add(
            Box::new(Expr::Mul(t(0), t(1))),
            Box::new(Expr::Mul(t(1), t(0)))
        )));
        assert!(!is_degenerate(&Expr::Add(t(0), t(1))));
        assert!(!is_degenerate(&Expr::Abs(Box::new(Expr::Sin(t(0))))));
        assert!(!is_degenerate(&Expr::Step(Box::new(Expr::Square(t(0))))));
    }

    #[test]
    fn tree_counts_match_enumeration() {
        let counts = tree_counts();

        for (n, row) in counts.iter().enumerate().take(7).skip(1) {
            assert_eq!(
                row[MAX_DEPTH] as usize,
                enumerate(n, MAX_DEPTH).len(),
                "n {n}"
            );
        }

        assert_eq!(counts[9][MAX_DEPTH], 907_362);
        assert!(counts[MAX_EXPR_LEN][MAX_DEPTH] > 0);
        assert_eq!(counts[2][0], 0);
    }

    /// The public length caps are at most a quarter of the canonical non-degenerate
    /// signatures of each capped length (recounted by enumeration).
    #[test]
    fn public_caps_leave_room_for_the_hidden_pool() {
        let expected = [4usize, 13, 63, 252, 828];

        for (len, &want) in (2..=6).zip(&expected) {
            let distinct: BTreeSet<String> = enumerate(len, MAX_DEPTH)
                .iter()
                .filter(|e| !is_degenerate(e))
                .map(signature)
                .collect();

            assert_eq!(distinct.len(), want, "length {len}");
            assert!(4 * PUBLIC_CAP[len] <= distinct.len(), "length {len}");
        }
    }

    #[test]
    fn sample_expr_hits_the_target_length_and_depth() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);

        for len in 1..=MAX_EXPR_LEN {
            for _ in 0..20 {
                let e = sample_expr(&mut rng, len).expect("feasible length");

                assert_eq!(length(&e), len);
                assert!(depth(&e) <= MAX_DEPTH);
                assert!(!is_degenerate(&e));
            }
        }

        assert!(sample_expr(&mut rng, 0).is_err());
        assert!(sample_expr(&mut rng, MAX_EXPR_LEN + 1).is_err());
    }

    #[test]
    fn sample_expr_is_uniform_over_signatures_of_length_two() {
        // Length 2 = unary(leaf): 12 trees, 4 canonical signatures, 3 trees each.
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();

        for _ in 0..12_000 {
            let e = sample_expr(&mut rng, 2).expect("length 2");

            *counts.entry(signature(&e)).or_insert(0) += 1;
        }

        assert_eq!(counts.len(), 4);
        assert!(
            counts.values().all(|&c| (2_700..=3_300).contains(&c)),
            "{counts:?}"
        );
    }

    #[test]
    fn sample_length_follows_two_to_the_minus_l() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let n = 51_000;
        let mut counts = [0usize; MAX_LEN + 1];

        for _ in 0..n {
            counts[sample_length(&mut rng)] += 1;
        }

        for (len, &count) in counts.iter().enumerate().skip(MIN_LEN) {
            let expected = n as f64 * length_weight(len) as f64 / 255.0;
            let sd = expected.sqrt().max(1.0);

            assert!(
                (count as f64 - expected).abs() < 5.0 * sd,
                "length {len}: {count} vs {expected}"
            );
        }

        assert_eq!(counts[0] + counts[1], 0);
    }

    #[test]
    fn public_lengths_respect_caps_and_weights() {
        let search = public_lengths(SEARCH_POOL_SIZE).expect("search lengths");
        let select = public_lengths(SELECT_POOL_SIZE).expect("select lengths");

        assert_eq!(search.len(), 64);
        assert_eq!(select.len(), 32);

        let hs = histogram(search.iter().copied());

        // Water-filled: L2..L4 hit their caps; the rest follows 2^-L.
        assert_eq!(
            hs.into_iter().collect::<Vec<_>>(),
            vec![
                (2, 1),
                (3, 3),
                (4, 15),
                (5, 23),
                (6, 12),
                (7, 6),
                (8, 3),
                (9, 1)
            ]
        );

        for (&l, &c) in histogram(select.iter().copied()).iter() {
            assert!(c <= PUBLIC_CAP[l]);
        }

        assert!(public_lengths(10_000).is_ok());
        assert_eq!(public_lengths(0).expect("empty"), Vec::<usize>::new());
    }

    #[test]
    fn largest_remainder_hand_case() {
        // 7 split as 1/3, 1/3, 1/3: floors 2, 2, 2; one extra to the smallest key.
        let shares = [(5, 7, 3), (6, 7, 3), (7, 7, 3)];

        assert_eq!(largest_remainder(&shares, 7), vec![(5, 3), (6, 2), (7, 2)]);

        // Histogram [2, 2, 3] scaled to 5: shares 10/3, 5/3 -> floors 3, 1; remainders
        // 1/3 and 2/3 -> length 3 gets the extra entry.
        assert_eq!(scaled_lengths(&[2, 2, 3], 5), vec![2, 2, 2, 3, 3]);
    }

    /// T-GRA-1 (part 1): pools are deterministic, distinct inside, Search ∩ Select = ∅.
    #[test]
    fn public_pools_are_deterministic_and_disjoint() {
        let search = public_pool(Tier::Search);
        let select = public_pool(Tier::Select);
        let (fresh_search, fresh_select) = build_public_pools();

        assert_eq!(search.len(), SEARCH_POOL_SIZE);
        assert_eq!(select.len(), SELECT_POOL_SIZE);
        assert_eq!(search, fresh_search);
        assert_eq!(select, fresh_select);

        let ss = signatures(&search);
        let sl = signatures(&select);

        assert_eq!(ss.len(), 64);
        assert_eq!(sl.len(), 32);
        assert!(ss.is_disjoint(&sl));
        assert!(public_pool(Tier::HiddenGrammar).is_empty());

        for (i, fam) in search.iter().enumerate() {
            assert_eq!(fam.id.0, format!("grammar/search/{i:03}"));
            assert_eq!(fam.signature, signature(&fam.expr));
            assert!((MIN_LEN..=MAX_LEN).contains(&length(&fam.expr)));
            assert!(depth(&fam.expr) <= MAX_DEPTH);
            assert!(!is_degenerate(&fam.expr));
        }

        // Kinds: roughly half regression.
        let regression = search
            .iter()
            .chain(&select)
            .filter(|f| f.kind == LearnKind::Regression)
            .count();

        assert!(
            (30..=66).contains(&regression),
            "regression families {regression}"
        );
    }

    /// T-GRA-1 (part 2) and the hidden-pool rule of contract 5.15: for many stand-in
    /// secret seeds the hidden pool avoids both public pools, has 40 distinct signatures
    /// and matches the search length histogram exactly.
    #[test]
    fn hidden_style_pools_never_collide_with_public_signatures() {
        let exclude = public_signatures();
        let search_lengths: Vec<usize> = public_pool_slice(Tier::Search)
            .iter()
            .map(|f| length(&f.expr))
            .collect();
        let want = histogram(scaled_lengths(&search_lengths, 40).into_iter());

        assert_eq!(exclude.len(), 96);

        for secret in 0..25u64 {
            let seed = derive_seed(secret.wrapping_mul(0x9e37_79b9), "grammar/hidden");
            let pool = disjoint_pool(seed, 40, &exclude, &search_lengths, "grammar/hidden")
                .expect("hidden pool is feasible");
            let sigs = signatures(&pool);

            assert_eq!(pool.len(), 40);
            assert_eq!(sigs.len(), 40);
            assert!(sigs.is_disjoint(&exclude), "secret {secret}");
            assert_eq!(histogram(pool.iter().map(|f| length(&f.expr))), want);
            assert_eq!(pool[3].id.0, "grammar/hidden/003");
        }
    }

    /// SCI-R4: the rewrites of `semantic_key` find the twins the review named, and nothing
    /// that changes the distribution.
    #[test]
    fn semantic_keys_find_exact_twins_only() {
        let add = |a, b| Box::new(Expr::Add(a, b));
        let key = semantic_key;
        let reg = LearnKind::Regression;
        let bin = LearnKind::Binary;

        // select/031 abs(add(t0,t1)) is regression abs(t0).
        assert_eq!(
            key(&Expr::Abs(add(t(0), t(1))), reg),
            key(&Expr::Abs(t(0)), reg)
        );

        // search/031 mul(sin(t0),t1) equals mul(add(t0,t1),sin(t2)).
        assert_eq!(
            key(&Expr::Mul(Box::new(Expr::Sin(t(0))), t(1)), reg),
            key(&Expr::Mul(add(t(0), t(1)), Box::new(Expr::Sin(t(2)))), reg)
        );

        // A whole sum of projections at the root, nested sums included.
        assert_eq!(
            key(&Expr::Add(add(t(0), t(1)), t(2)), bin),
            key(&Expr::T(0), bin)
        );

        // Classification: square at the root is abs; regression keeps them apart.
        let square = Expr::Square(t(0));
        let abs = Expr::Abs(t(0));

        assert_eq!(key(&square, bin), key(&abs, bin));
        assert_ne!(key(&square, reg), key(&abs, reg));
        assert_ne!(key(&abs, reg), key(&abs, bin));
        assert_ne!(
            key(&abs, LearnKind::Multiclass { k: 3 }),
            key(&abs, LearnKind::Multiclass { k: 4 })
        );

        // Not scale-free: under sin, max, step or add the length of u_i + u_j matters.
        for e in [
            Expr::Sin(add(t(0), t(1))),
            Expr::Max(add(t(0), t(1)), t(2)),
            Expr::Step(add(t(0), t(1))),
            Expr::Add(Box::new(Expr::Abs(add(t(0), t(1)))), t(2)),
        ] {
            assert_eq!(
                key(&e, reg),
                format!("regression:{}", signature(&e)),
                "{e:?}"
            );
        }

        // A projection used twice is not folded: mul(add(t0,t1), t0).
        let shared = Expr::Mul(add(t(0), t(1)), t(0));

        assert_eq!(
            key(&shared, reg),
            format!("regression:{}", signature(&shared))
        );
    }

    #[test]
    fn disjoint_pool_errors() {
        let empty = BTreeSet::new();

        // Only 4 signatures of length 2 exist: 5 distinct ones are impossible.
        assert!(disjoint_pool(1, 5, &empty, &[2], "p").is_err());
        assert!(disjoint_pool(1, 4, &empty, &[2], "p").is_ok());

        // Excluding all 4 leaves nothing.
        let all: BTreeSet<String> = disjoint_pool(1, 4, &empty, &[2], "p")
            .expect("four")
            .iter()
            .map(|f| f.signature.clone())
            .collect();

        assert!(disjoint_pool(2, 1, &all, &[2], "p").is_err());
        assert!(disjoint_pool(1, 3, &empty, &[], "p").is_err());
        assert!(disjoint_pool(1, 3, &empty, &[16], "p").is_err());
        assert_eq!(
            disjoint_pool(1, 0, &empty, &[], "p").expect("empty").len(),
            0
        );
    }

    #[test]
    fn eval_matches_hand_computation() {
        // max(t0 * t1, step(t2)) with the step threshold drawn from the rng.
        let e = Expr::Max(Box::new(Expr::Mul(t(0), t(1))), Box::new(Expr::Step(t(2))));
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut program = Vec::new();

        compile(&e, &mut rng, &mut program);

        // Postfix: t0 t1 mul t2 step max.
        let threshold = match program[4] {
            Node::Step(th) => th,
            _ => f64::NAN,
        };

        assert!(threshold.is_finite());
        assert_eq!(program.len(), 6);
        let tv = [2.0, -3.0, threshold + 0.1];

        assert_eq!(eval(&program, &tv), 1.0);
        assert_eq!(eval(&program, &[2.0, 3.0, threshold - 0.1]), 6.0);
        assert_eq!(eval(&[Node::Add], &tv), 0.0);
    }

    #[test]
    fn grammar_instances_are_deterministic_finite_and_balanced() {
        let pool = public_pool_slice(Tier::Search);

        for (i, fam) in pool.iter().enumerate().take(24) {
            let f = [8, 16, 32][i % 3];
            let p = params(f, [16, 64, 256][i % 3]);
            let a = grammar_instance(fam, 1_000 + i as u64, &p);
            let b = grammar_instance(fam, 1_000 + i as u64, &p);

            assert_eq!(a, b);
            assert_eq!(a.f, f);
            assert_eq!(a.group, Group::Grammar);
            assert_eq!(a.x_train.len(), p.n_train * f);
            assert_eq!(a.x_valid.len(), 64 * f);
            assert!(a.x_train.iter().chain(&a.x_valid).all(|v| v.is_finite()));

            match (&a.y_train, a.kind) {
                (Labels::Values(v), LearnKind::Regression) => {
                    assert!(v.iter().all(|y| y.is_finite()));
                }
                (Labels::Classes(c), kind) => {
                    let k = kind_classes(kind);

                    assert!(c.iter().all(|&y| (y as usize) < k));
                }
                _ => panic!("label type does not match kind"),
            }
        }
    }

    #[test]
    fn grammar_classes_are_balanced_on_continuous_targets() {
        let fam = GrammarFamily {
            id: FamilyId("grammar/test/000".into()),
            expr: Expr::Add(Box::new(Expr::Sin(t(0))), t(1)),
            signature: String::new(),
            kind: LearnKind::Multiclass { k: 4 },
        };
        let task = grammar_instance(&fam, 9, &params(16, 256));
        let Labels::Classes(c) = &task.y_train else {
            panic!("classes expected");
        };
        let mut counts = [0usize; 4];

        for &y in c {
            counts[y as usize] += 1;
        }

        assert!(counts.iter().all(|&n| (40..=90).contains(&n)), "{counts:?}");
    }

    #[test]
    fn grammar_regression_targets_are_standardised_plus_noise() {
        let fam = GrammarFamily {
            id: FamilyId("grammar/test/001".into()),
            expr: Expr::Mul(t(0), Box::new(Expr::Sin(t(1)))),
            signature: String::new(),
            kind: LearnKind::Regression,
        };
        let task = grammar_instance(&fam, 11, &params(8, 256));
        let Labels::Values(v) = &task.y_train else {
            panic!("values expected");
        };
        let n = v.len() as f64;
        let mean = v.iter().map(|&y| f64::from(y)).sum::<f64>() / n;
        let var = v
            .iter()
            .map(|&y| (f64::from(y) - mean).powi(2))
            .sum::<f64>()
            / n;

        // Unit-variance signal plus 0.01 noise variance, up to sampling error.
        assert!(mean.abs() < 0.25, "mean {mean}");
        assert!((0.6..=1.5).contains(&var), "var {var}");
    }

    #[test]
    fn grammar_instance_sanitises_bad_params() {
        let fam = &public_pool_slice(Tier::Search)[0];
        let bad = InstanceParams {
            f: 7,
            n_train: 0,
            n_valid: 0,
            level: 0,
        };
        let task = grammar_instance(fam, 1, &bad);

        assert_eq!(task.f, 16);
        assert_eq!(task.n_train, 1);
        assert_eq!(task.n_valid, 1);
        assert!(task.x_train.iter().all(|v| v.is_finite()));
    }
}
