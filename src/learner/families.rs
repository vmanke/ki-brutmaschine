//! The 24 hand-written task families, split-v1, search panels and SELECT task sets.
//!
//! Owner: WP2. Contract sections 5.7, 6.4 and errata E7, E12, E13. Pure: no I/O.
//!
//! Instance streams (all under the instance seed): "gen/params" draws the family's random
//! parameters and the embedding Q, "gen/train" and "gen/valid" draw the examples. These
//! tags never collide with "interp/<a>" and "stream/<e>" of the protocol.
#![forbid(unsafe_code)]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};

use crate::learner::grammar::{self, GrammarFamily};
use crate::learner::stats::{
    normal_f64, range_f64, shuffle, uniform_below, uniform_index, unit_f64,
};
use crate::learner::types::{
    FamilyId, Group, Labels, LearnKind, PanelConfig, STEPS_PER_TASK, SUPPORTED_F, SelectConfig,
    TaskInstance, Tier,
};
use crate::util::{BrutError, Result, derive_seed};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FamilySpec {
    pub id: &'static str,
    pub group: Group,
    /// True when the developer has seen this family's behaviour in pilot runs.
    pub developer_seen: bool,
}

/// n_train values that every fixed task set cycles through (errata E7).
pub const N_TRAIN_CYCLE: [usize; 3] = [16, 64, 256];
/// Widths of the sets that "cycle" (SELECT, hidden tiers except HiddenInstance, tuning).
pub const DIMS_CYCLE: [usize; 3] = [8, 16, 32];
/// Widths of the sets that "alternate" (public reference tasks, HiddenInstance).
pub const DIMS_ALTERNATE: [usize; 2] = [8, 16];
/// Validation rows of SELECT tasks.
pub const SELECT_N_VALID: usize = 64;
/// Validation rows of the public reference tasks (same as the hidden tiers).
pub const PUBLIC_REFERENCE_N_VALID: usize = 256;
/// Instances per SEARCH hand family in `public_reference_tasks`.
pub const PUBLIC_REFERENCE_PER_FAMILY: usize = 10;
/// Largest n_train or n_valid of one instance.
pub const MAX_ROWS: usize = 1 << 16;

/// Largest latent dimension (1 + level with level <= 3).
const MAX_D: usize = 4;
/// Train-label flip rates.
const LIN_CLS_FLIP: f64 = 0.05;
const NOISY_LABEL_FLIP: f64 = 0.30;
/// Regression noise of the families that state "+ noise".
const NOISE_SD: f64 = 0.1;
/// Input noise of the task.rs checkerboard and spirals (reimplemented).
const CHECKERBOARD_NOISE: f64 = 0.02;
const SPIRAL_NOISE: f64 = 0.03;
const SPIRAL_TURNS: f64 = 3.0;
/// Medians of the chi-square distribution with 2, 3 and 4 degrees of freedom.
const CHI2_MEDIAN: [f64; 3] = [
    1.386_294_361_119_890_4,
    2.365_973_884_375_337_7,
    3.356_693_980_033_320_6,
];

/// The 24 hand-written families in contract table order.
static REGISTRY: [FamilySpec; 24] = [
    spec("lin-reg", Group::Linear, false),
    spec("lin-cls", Group::Linear, false),
    spec("sparse-lin-reg", Group::Linear, false),
    spec("multiclass-lin", Group::Linear, false),
    spec("quad-reg", Group::Interaction, false),
    spec("xor-cls", Group::Interaction, true),
    spec("sign-parity-cls", Group::Interaction, true),
    spec("bilinear-reg", Group::Interaction, false),
    spec("sphere-cls", Group::Radial, false),
    spec("blobs-multiclass", Group::Radial, false),
    spec("rbf-reg", Group::Radial, false),
    spec("prototype-cls", Group::Radial, false),
    spec("sine-reg", Group::Periodic, true),
    spec("checkerboard-cls", Group::Periodic, true),
    spec("spirals-cls", Group::Periodic, true),
    spec("sawtooth-reg", Group::Periodic, false),
    spec("tree-cls", Group::Piecewise, false),
    spec("relu-teacher-reg", Group::Piecewise, false),
    spec("step-reg", Group::Piecewise, false),
    spec("dnf-cls", Group::Piecewise, false),
    spec("drift-cls", Group::Robustness, false),
    spec("noisy-label-cls", Group::Robustness, false),
    spec("heavy-tail-reg", Group::Robustness, false),
    spec("tanh-teacher-cls", Group::Robustness, false),
];

const fn spec(id: &'static str, group: Group, developer_seen: bool) -> FamilySpec {
    FamilySpec {
        id,
        group,
        developer_seen,
    }
}

/// The six mechanism groups of the hand families.
const HAND_GROUPS: [Group; 6] = [
    Group::Linear,
    Group::Interaction,
    Group::Radial,
    Group::Periodic,
    Group::Piecewise,
    Group::Robustness,
];

/// The 24 hand-written families (contract table in section 5.7), in table order.
pub fn registry() -> &'static [FamilySpec] {
    &REGISTRY
}

/// Serde (snake_case) name of a group, as hashed by split-v1.
fn group_name(g: Group) -> &'static str {
    match g {
        Group::Linear => "linear",
        Group::Interaction => "interaction",
        Group::Radial => "radial",
        Group::Periodic => "periodic",
        Group::Piecewise => "piecewise",
        Group::Robustness => "robustness",
        Group::Grammar => "grammar",
        Group::ArcCell => "arc_cell",
    }
}

fn split_digest(prefix: &str, name: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();

    hasher.update(prefix.as_bytes());
    hasher.update(name.as_bytes());

    hasher.finalize().into()
}

/// split-v1 (golden; a unit test pins it): hidden-far group = smallest
/// SHA-256("brut-split-v1/group/" + name); inside every other group the families ordered by
/// SHA-256("brut-split-v1/family/" + name): first two Search, third Select, fourth HiddenNear.
/// Returned in registry order.
pub fn split_v1() -> Vec<(FamilyId, Tier)> {
    split_tiers()
        .iter()
        .zip(REGISTRY.iter())
        .map(|(&tier, s)| (FamilyId(s.id.to_string()), tier))
        .collect()
}

/// Tier of every registry entry, by registry index.
fn split_tiers() -> [Tier; 24] {
    let far = HAND_GROUPS
        .iter()
        .copied()
        .min_by_key(|&g| split_digest("brut-split-v1/group/", group_name(g)));
    let mut tiers = [Tier::HiddenNear; 24];

    for g in HAND_GROUPS {
        let mut members: Vec<usize> = (0..REGISTRY.len())
            .filter(|&i| REGISTRY[i].group == g)
            .collect();

        if Some(g) == far {
            for i in members {
                tiers[i] = Tier::HiddenFar;
            }

            continue;
        }

        members.sort_by_key(|&i| split_digest("brut-split-v1/family/", REGISTRY[i].id));

        for (rank, i) in members.into_iter().enumerate() {
            tiers[i] = match rank {
                0 | 1 => Tier::Search,
                2 => Tier::Select,
                _ => Tier::HiddenNear,
            };
        }
    }

    tiers
}

/// Registry indices of a tier, in registry order.
fn hand_indices(tier: Tier) -> Vec<usize> {
    split_tiers()
        .iter()
        .enumerate()
        .filter(|&(_, &t)| t == tier)
        .map(|(i, _)| i)
        .collect()
}

/// Families of a tier among the hand-written ones, in registry order.
pub fn hand_families(tier: Tier) -> Vec<FamilyId> {
    hand_indices(tier)
        .into_iter()
        .map(|i| FamilyId(REGISTRY[i].id.to_string()))
        .collect()
}

/// Split tier of a hand family (None for unknown ids).
pub fn hand_tier(fam: &FamilyId) -> Option<Tier> {
    let tiers = split_tiers();

    registry_index(&fam.0).map(|i| tiers[i])
}

fn registry_index(id: &str) -> Option<usize> {
    REGISTRY.iter().position(|s| s.id == id)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstanceParams {
    pub f: usize,
    pub n_train: usize,
    pub n_valid: usize,
    pub level: u8,
}

/// Draw f from `dims`, n_train from `n_train_choices`, level l in {1,2,3} with P ∝ 2^-l.
/// Draw order: f, n_train, level. An empty `dims` gives 16, an empty `n_train_choices` 64.
pub fn sample_params(
    rng: &mut ChaCha8Rng,
    dims: &[usize],
    n_train_choices: &[usize],
    n_valid: usize,
) -> InstanceParams {
    let f = pick(rng, dims, 16);
    let n_train = pick(rng, n_train_choices, 64);

    // Weights 4 : 2 : 1 for levels 1, 2, 3.
    let level = match uniform_below(rng, 7) {
        0..=3 => 1,
        4 | 5 => 2,
        _ => 3,
    };

    InstanceParams {
        f,
        n_train,
        n_valid,
        level,
    }
}

/// Uniform element of `choices` (one draw), or `fallback` without drawing when empty.
fn pick(rng: &mut ChaCha8Rng, choices: &[usize], fallback: usize) -> usize {
    if choices.is_empty() {
        return fallback;
    }

    choices[uniform_index(rng, choices.len())]
}

/// Parameters of task k of a fixed set (errata E7): f = dims[k % len],
/// n_train = [16, 64, 256][(k / len) % 3], level from `sample_params` seeded with the
/// instance seed and single-value choices.
pub fn cycled_params(
    k: usize,
    dims: &[usize],
    n_valid: usize,
    instance_seed: u64,
) -> InstanceParams {
    let len = dims.len().max(1);
    let f = dims.get(k % len).copied().unwrap_or(16);
    let n_train = N_TRAIN_CYCLE[(k / len) % N_TRAIN_CYCLE.len()];
    let mut rng = ChaCha8Rng::seed_from_u64(instance_seed);
    let level = sample_params(&mut rng, &[f], &[n_train], n_valid).level;

    InstanceParams {
        f,
        n_train,
        n_valid,
        level,
    }
}

/// Ok when f is supported, 1 <= n_train, n_valid <= MAX_ROWS and level is 1..=3.
pub fn check_params(p: &InstanceParams) -> Result<()> {
    if !SUPPORTED_F.contains(&p.f) {
        return Err(BrutError::Invalid(format!(
            "task width {} is not supported",
            p.f
        )));
    }

    if !(1..=MAX_ROWS).contains(&p.n_train) || !(1..=MAX_ROWS).contains(&p.n_valid) {
        return Err(BrutError::Invalid(format!(
            "task sizes must be in 1..={MAX_ROWS}"
        )));
    }

    if !(1..=3).contains(&p.level) {
        return Err(BrutError::Invalid(format!(
            "task level {} is not in 1..=3",
            p.level
        )));
    }

    Ok(())
}

/// f -> 16 when unsupported; n_train and n_valid clamped to 1..=MAX_ROWS; level kept.
pub(crate) fn sanitize_sizes(p: &InstanceParams) -> InstanceParams {
    InstanceParams {
        f: if SUPPORTED_F.contains(&p.f) { p.f } else { 16 },
        n_train: p.n_train.clamp(1, MAX_ROWS),
        n_valid: p.n_valid.clamp(1, MAX_ROWS),
        level: p.level,
    }
}

/// First 8 bytes (LE) of SHA-256 over "family|seed|f|n_train|n_valid|level".
pub fn instance_id(family: &str, seed: u64, p: &InstanceParams) -> u64 {
    let text = format!(
        "{family}|{seed}|{}|{}|{}|{}",
        p.f, p.n_train, p.n_valid, p.level
    );

    first_u64(&Sha256::digest(text.as_bytes()))
}

fn first_u64(digest: &[u8]) -> u64 {
    let mut first = [0u8; 8];

    first.copy_from_slice(&digest[..8]);

    u64::from_le_bytes(first)
}

/// Random unit vector in R^d (direction uniform on the sphere).
pub(crate) fn unit_vector(rng: &mut ChaCha8Rng, d: usize) -> Vec<f64> {
    let mut v = vec![0.0; d];

    // A Gaussian vector is zero with probability 0; the cap only keeps this total.
    for _ in 0..64 {
        for x in v.iter_mut() {
            *x = normal_f64(rng);
        }

        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm > 1e-9 {
            for x in v.iter_mut() {
                *x /= norm;
            }

            return v;
        }
    }

    if let Some(first) = v.first_mut() {
        *first = 1.0;
    }

    v
}

/// Random F x F orthonormal matrix, row-major: modified Gram-Schmidt of Gaussian rows.
pub(crate) fn orthonormal(rng: &mut ChaCha8Rng, f: usize) -> Vec<f64> {
    let mut q = vec![0.0; f * f];

    for i in 0..f {
        for attempt in 0.. {
            let mut v: Vec<f64> = (0..f).map(|_| normal_f64(rng)).collect();

            for j in 0..i {
                let row = &q[j * f..(j + 1) * f];
                let dot: f64 = v.iter().zip(row).map(|(a, b)| a * b).sum();

                for (vk, rk) in v.iter_mut().zip(row) {
                    *vk -= dot * rk;
                }
            }

            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();

            // Gaussian rows are independent almost surely; the cap only keeps this total.
            if norm > 1e-6 || attempt >= 64 {
                for (qk, vk) in q[i * f..(i + 1) * f].iter_mut().zip(&v) {
                    *qk = vk / norm.max(1e-12);
                }

                break;
            }
        }
    }

    q
}

/// x = Q · [z; ξ] with ξ ~ N(0, I_{F-d}) drawn from `rng`. `filler` is scratch of length f.
pub(crate) fn embed(
    q: &[f64],
    f: usize,
    z: &[f64],
    rng: &mut ChaCha8Rng,
    filler: &mut [f64],
    out: &mut [f64],
) {
    let d = z.len().min(f);

    filler[..d].copy_from_slice(&z[..d]);

    for w in filler[d..f].iter_mut() {
        *w = normal_f64(rng);
    }

    for (r, x) in out.iter_mut().enumerate().take(f) {
        *x = q[r * f..(r + 1) * f]
            .iter()
            .zip(&filler[..f])
            .map(|(a, b)| a * b)
            .sum();
    }
}

/// Standardise every feature with the train split's mean and population std
/// (std < 1e-6 counts as 1) and apply the same map to both splits.
pub(crate) fn standardize_features(train: &[f64], valid: &[f64], f: usize) -> (Vec<f32>, Vec<f32>) {
    let n = (train.len() / f.max(1)).max(1) as f64;
    let mut mean = vec![0.0; f];
    let mut sd = vec![0.0; f];

    for row in train.chunks_exact(f) {
        for (m, &x) in mean.iter_mut().zip(row) {
            *m += x;
        }
    }

    for m in mean.iter_mut() {
        *m /= n;
    }

    for row in train.chunks_exact(f) {
        for ((s, &x), &m) in sd.iter_mut().zip(row).zip(&mean) {
            *s += (x - m) * (x - m);
        }
    }

    for s in sd.iter_mut() {
        *s = (*s / n).sqrt();

        if *s < 1e-6 {
            *s = 1.0;
        }
    }

    let apply = |data: &[f64]| -> Vec<f32> {
        data.chunks_exact(f)
            .flat_map(|row| {
                row.iter()
                    .zip(&mean)
                    .zip(&sd)
                    .map(|((&x, &m), &s)| ((x - m) / s) as f32)
            })
            .collect()
    };

    (apply(train), apply(valid))
}

/// Standardise regression targets with the train split's mean and std.
fn standardize_values(train: &[f64], valid: &[f64]) -> (Vec<f32>, Vec<f32>) {
    standardize_features(train, valid, 1)
}

/// How the latent point of one example is drawn.
#[derive(Clone, Debug)]
enum Latent {
    Gaussian,
    /// Student-t with 2 degrees of freedom per axis.
    StudentT,
    /// Uniform on [-1, 1]^2; the label comes from the clean point, then input noise.
    Checkerboard,
    /// task.rs two spirals; class alternates with the example index.
    Spirals,
    /// Class uniform in 0..k, then z = centre + sigma * N(0, I).
    Blobs {
        centres: Vec<Vec<f64>>,
        sigmas: Vec<f64>,
    },
}

/// The labelling rule of one instance (family parameters drawn).
#[derive(Clone, Debug)]
enum Rule {
    /// y = w·z + noise_sd * N (noise Student-t when `heavy`).
    Linear { w: Vec<f64>, heavy: bool },
    /// y = 1[w·z + b > 0].
    Halfspace { w: Vec<f64>, b: f64 },
    /// y = Σ w_j x_raw[c_j] + noise.
    SparseRaw { coords: Vec<(usize, f64)> },
    /// y = argmax_k (W z)_k.
    Argmax { w: Vec<Vec<f64>> },
    /// y = (a·z)² - 1 + noise.
    Quad { a: Vec<f64> },
    /// y = 1[(a·z)(b·z) > 0].
    Xor { a: Vec<f64>, b: Vec<f64> },
    /// Parity of the sign bits of the first `axes` latent axes.
    SignParity { axes: usize },
    /// y = (a·z)(b·z) + noise.
    Bilinear { a: Vec<f64>, b: Vec<f64> },
    /// y = 1[|z|² < r2].
    Sphere { r2: f64 },
    /// The class the latent sampler drew.
    FromLatent,
    /// y = Σ a_j exp(-|z - c_j|² / 2s²).
    Rbf {
        amps: Vec<f64>,
        centres: Vec<Vec<f64>>,
        s: f64,
    },
    /// Label of the nearest prototype.
    Prototype {
        protos: Vec<Vec<f64>>,
        labels: Vec<u8>,
    },
    /// y = A sin(ω a·z + φ).
    Sine {
        a: Vec<f64>,
        amp: f64,
        omega: f64,
        phi: f64,
    },
    /// y = frac(ω a·z + φ) - 0.5.
    Sawtooth { a: Vec<f64>, omega: f64, phi: f64 },
    /// Complete axis-aligned tree in heap order; leaves after the internal nodes.
    Tree {
        depth: usize,
        axes: Vec<usize>,
        thresholds: Vec<f64>,
        leaves: Vec<u8>,
    },
    /// y = Σ v_j relu(w_j·z + b_j).
    Relu { units: Vec<(f64, Vec<f64>, f64)> },
    /// y = Σ h_j 1[a_j·z > t_j].
    Steps { steps: Vec<(f64, Vec<f64>, f64)> },
    /// OR of terms; a term is an AND of (axis, sign) literals on z_axis > 0.
    Dnf { terms: Vec<[(usize, bool); 2]> },
    /// y = 1[Σ v_j tanh(w_j·z) > 0].
    Tanh { units: Vec<(f64, Vec<f64>)> },
}

/// Everything drawn once per instance before the examples.
struct Plan {
    d: usize,
    kind: LearnKind,
    latent: Latent,
    rule: Rule,
    /// Probability of flipping a binary train label.
    flip: f64,
    /// Drift: late labels use the negated weights.
    drift: bool,
}

/// Inner product of the common prefix.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Two orthonormal directions in R^d (d >= 2).
fn orthogonal_pair(rng: &mut ChaCha8Rng, d: usize) -> (Vec<f64>, Vec<f64>) {
    let q = orthonormal(rng, d);

    (q[..d].to_vec(), q[d..2 * d].to_vec())
}

/// `count` labels, half 0 and half 1 (an odd extra is 0), in random order.
fn balanced_labels(rng: &mut ChaCha8Rng, count: usize) -> Vec<u8> {
    let mut labels: Vec<u8> = (0..count).map(|i| (i % 2) as u8).collect();

    shuffle(rng, &mut labels);

    labels
}

/// Gaussian vector N(0, scale² I_d).
fn gaussian(rng: &mut ChaCha8Rng, d: usize, scale: f64) -> Vec<f64> {
    (0..d).map(|_| scale * normal_f64(rng)).collect()
}

/// Student-t(2) sample: N / sqrt(E) with E ~ Exp(1) = chi²_2 / 2 (E floored at 1e-12).
fn student_t2(rng: &mut ChaCha8Rng) -> f64 {
    let n = normal_f64(rng);
    let e = (-(1.0 - unit_f64(rng)).ln()).max(1e-12);

    n / e.sqrt()
}

/// Draw the family parameters of registry entry `idx` at latent dimension d.
fn plan(idx: usize, level: u8, f: usize, rng: &mut ChaCha8Rng) -> Plan {
    let id = REGISTRY[idx].id;
    let level = level.clamp(1, 3) as usize;
    let d = match id {
        "checkerboard-cls" | "spirals-cls" => 2,
        _ => 1 + level,
    };
    let k = (2 + level) as u8;

    let binary = |rule: Rule| Plan {
        d,
        kind: LearnKind::Binary,
        latent: Latent::Gaussian,
        rule,
        flip: 0.0,
        drift: false,
    };
    let regression = |rule: Rule| Plan {
        d,
        kind: LearnKind::Regression,
        latent: Latent::Gaussian,
        rule,
        flip: 0.0,
        drift: false,
    };

    match id {
        "lin-reg" => regression(Rule::Linear {
            w: unit_vector(rng, d),
            heavy: false,
        }),
        "lin-cls" | "noisy-label-cls" => {
            let w = unit_vector(rng, d);
            let b = 0.25 * normal_f64(rng);
            let flip = if id == "lin-cls" {
                LIN_CLS_FLIP
            } else {
                NOISY_LABEL_FLIP
            };

            Plan {
                flip,
                ..binary(Rule::Halfspace { w, b })
            }
        }
        "sparse-lin-reg" => {
            let count = 1 + uniform_index(rng, 2);
            let mut columns: Vec<usize> = (0..f).collect();

            shuffle(rng, &mut columns);

            let coords = columns
                .into_iter()
                .take(count)
                .map(|c| (c, normal_f64(rng)))
                .collect();

            regression(Rule::SparseRaw { coords })
        }
        "multiclass-lin" => Plan {
            kind: LearnKind::Multiclass { k },
            ..binary(Rule::Argmax {
                w: (0..k).map(|_| unit_vector(rng, d)).collect(),
            })
        },
        "quad-reg" => regression(Rule::Quad {
            a: unit_vector(rng, d),
        }),
        "xor-cls" => {
            let (a, b) = orthogonal_pair(rng, d);

            binary(Rule::Xor { a, b })
        }
        "sign-parity-cls" => binary(Rule::SignParity {
            axes: d.min(1 + level),
        }),
        "bilinear-reg" => {
            let (a, b) = orthogonal_pair(rng, d);

            regression(Rule::Bilinear { a, b })
        }
        "sphere-cls" => binary(Rule::Sphere {
            r2: CHI2_MEDIAN[(d - 2).min(2)],
        }),
        "blobs-multiclass" => {
            let centres = (0..k).map(|_| gaussian(rng, d, 2.0)).collect();
            let sigmas = (0..k).map(|_| range_f64(rng, 0.5, 1.5)).collect();

            Plan {
                kind: LearnKind::Multiclass { k },
                latent: Latent::Blobs { centres, sigmas },
                ..binary(Rule::FromLatent)
            }
        }
        "rbf-reg" => {
            let m = 1 + level;
            let amps = (0..m).map(|_| normal_f64(rng)).collect();
            let centres = (0..m).map(|_| gaussian(rng, d, 1.0)).collect();
            let s = range_f64(rng, 0.75, 1.5);

            regression(Rule::Rbf { amps, centres, s })
        }
        "prototype-cls" => {
            let m = 4 + uniform_index(rng, 5);
            let protos = (0..m).map(|_| gaussian(rng, d, 1.0)).collect();
            let labels = balanced_labels(rng, m);

            binary(Rule::Prototype { protos, labels })
        }
        "sine-reg" => regression(Rule::Sine {
            a: unit_vector(rng, d),
            amp: range_f64(rng, 0.1, 5.0),
            omega: range_f64(rng, 1.0, 3.0),
            phi: range_f64(rng, 0.0, std::f64::consts::PI),
        }),
        "checkerboard-cls" => Plan {
            latent: Latent::Checkerboard,
            ..binary(Rule::FromLatent)
        },
        "spirals-cls" => Plan {
            latent: Latent::Spirals,
            ..binary(Rule::FromLatent)
        },
        "sawtooth-reg" => regression(Rule::Sawtooth {
            a: unit_vector(rng, d),
            omega: range_f64(rng, 1.0, 3.0),
            phi: unit_f64(rng),
        }),
        "tree-cls" => {
            let depth = 1 + level;
            let internal = (1usize << depth) - 1;
            let axes = (0..internal).map(|_| uniform_index(rng, d)).collect();
            let thresholds = (0..internal).map(|_| 0.5 * normal_f64(rng)).collect();
            let leaves = balanced_labels(rng, 1 << depth);

            binary(Rule::Tree {
                depth,
                axes,
                thresholds,
                leaves,
            })
        }
        "relu-teacher-reg" => regression(Rule::Relu {
            units: (0..1 + level)
                .map(|_| (normal_f64(rng), unit_vector(rng, d), 0.5 * normal_f64(rng)))
                .collect(),
        }),
        "step-reg" => regression(Rule::Steps {
            steps: (0..level)
                .map(|_| (normal_f64(rng), unit_vector(rng, d), 0.5 * normal_f64(rng)))
                .collect(),
        }),
        "dnf-cls" => {
            let terms = (0..3)
                .map(|_| {
                    let first = uniform_index(rng, d);
                    let second = (first + 1 + uniform_index(rng, d - 1)) % d;

                    [
                        (first, uniform_below(rng, 2) == 1),
                        (second, uniform_below(rng, 2) == 1),
                    ]
                })
                .collect();

            binary(Rule::Dnf { terms })
        }
        "drift-cls" => Plan {
            drift: true,
            ..binary(Rule::Halfspace {
                w: unit_vector(rng, d),
                b: 0.0,
            })
        },
        "heavy-tail-reg" => Plan {
            latent: Latent::StudentT,
            ..regression(Rule::Linear {
                w: unit_vector(rng, d),
                heavy: true,
            })
        },
        // "tanh-teacher-cls" (the registry has no other ids).
        _ => binary(Rule::Tanh {
            units: (0..1 + level)
                .map(|_| (normal_f64(rng), gaussian(rng, d, 1.0)))
                .collect(),
        }),
    }
}

impl Latent {
    /// Draw z (first d entries) and, for samplers that decide the class, that class.
    fn draw(&self, i: usize, d: usize, rng: &mut ChaCha8Rng) -> ([f64; MAX_D], Option<f64>) {
        let mut z = [0.0; MAX_D];

        match self {
            Latent::Gaussian => {
                for zj in z.iter_mut().take(d) {
                    *zj = normal_f64(rng);
                }

                (z, None)
            }
            Latent::StudentT => {
                for zj in z.iter_mut().take(d) {
                    *zj = student_t2(rng);
                }

                (z, None)
            }
            Latent::Checkerboard => {
                let px = range_f64(rng, -1.0, 1.0);
                let py = range_f64(rng, -1.0, 1.0);
                let cx = (((px + 1.0) * 2.0).floor() as i64).clamp(0, 3);
                let cy = (((py + 1.0) * 2.0).floor() as i64).clamp(0, 3);

                z[0] = px + CHECKERBOARD_NOISE * normal_f64(rng);
                z[1] = py + CHECKERBOARD_NOISE * normal_f64(rng);

                (z, Some(((cx + cy) % 2) as f64))
            }
            Latent::Spirals => {
                let class = (i % 2) as f64;
                let t = unit_f64(rng).sqrt();
                let angle = t * SPIRAL_TURNS * std::f64::consts::TAU + class * std::f64::consts::PI;

                z[0] = t * angle.cos() + SPIRAL_NOISE * normal_f64(rng);
                z[1] = t * angle.sin() + SPIRAL_NOISE * normal_f64(rng);

                (z, Some(class))
            }
            Latent::Blobs { centres, sigmas } => {
                let c = uniform_index(rng, centres.len());

                for (j, zj) in z.iter_mut().enumerate().take(d) {
                    *zj = centres[c][j] + sigmas[c] * normal_f64(rng);
                }

                (z, Some(c as f64))
            }
        }
    }
}

impl Rule {
    /// Clean label (classification) or noisy target (regression) of one example.
    fn label(&self, z: &[f64], x_raw: &[f64], hint: Option<f64>, rng: &mut ChaCha8Rng) -> f64 {
        let indicator = |b: bool| if b { 1.0 } else { 0.0 };

        match self {
            Rule::Linear { w, heavy } => {
                let noise = if *heavy {
                    student_t2(rng)
                } else {
                    normal_f64(rng)
                };

                dot(w, z) + NOISE_SD * noise
            }
            Rule::Halfspace { w, b } => indicator(dot(w, z) + b > 0.0),
            Rule::SparseRaw { coords } => {
                let signal: f64 = coords.iter().map(|&(c, w)| w * x_raw[c]).sum();

                signal + NOISE_SD * normal_f64(rng)
            }
            Rule::Argmax { w } => {
                let mut best = 0;
                let mut best_value = f64::NEG_INFINITY;

                for (c, row) in w.iter().enumerate() {
                    let v = dot(row, z);

                    if v > best_value {
                        best = c;
                        best_value = v;
                    }
                }

                best as f64
            }
            Rule::Quad { a } => {
                let p = dot(a, z);

                p * p - 1.0 + NOISE_SD * normal_f64(rng)
            }
            Rule::Xor { a, b } => indicator(dot(a, z) * dot(b, z) > 0.0),
            Rule::SignParity { axes } => {
                let ones = z.iter().take(*axes).filter(|&&v| v > 0.0).count();

                (ones % 2) as f64
            }
            Rule::Bilinear { a, b } => dot(a, z) * dot(b, z) + NOISE_SD * normal_f64(rng),
            Rule::Sphere { r2 } => indicator(dot(z, z) < *r2),
            Rule::FromLatent => hint.unwrap_or(0.0),
            Rule::Rbf { amps, centres, s } => amps
                .iter()
                .zip(centres)
                .map(|(a, c)| {
                    let dist2: f64 = z.iter().zip(c).map(|(x, y)| (x - y) * (x - y)).sum();

                    a * (-dist2 / (2.0 * s * s)).exp()
                })
                .sum(),
            Rule::Prototype { protos, labels } => {
                let mut best = 0;
                let mut best_dist = f64::INFINITY;

                for (j, p) in protos.iter().enumerate() {
                    let dist: f64 = z.iter().zip(p).map(|(x, y)| (x - y) * (x - y)).sum();

                    if dist < best_dist {
                        best = j;
                        best_dist = dist;
                    }
                }

                f64::from(labels[best])
            }
            Rule::Sine { a, amp, omega, phi } => amp * (omega * dot(a, z) + phi).sin(),
            Rule::Sawtooth { a, omega, phi } => {
                let v = omega * dot(a, z) + phi;

                v - v.floor() - 0.5
            }
            Rule::Tree {
                depth,
                axes,
                thresholds,
                leaves,
            } => {
                let mut node = 0;

                for _ in 0..*depth {
                    node = if z[axes[node]] <= thresholds[node] {
                        2 * node + 1
                    } else {
                        2 * node + 2
                    };
                }

                f64::from(leaves[node - axes.len()])
            }
            Rule::Relu { units } => units
                .iter()
                .map(|(v, w, b)| v * (dot(w, z) + b).max(0.0))
                .sum(),
            Rule::Steps { steps } => steps
                .iter()
                .map(|(h, a, t)| h * indicator(dot(a, z) > *t))
                .sum(),
            Rule::Dnf { terms } => {
                let holds = |&(axis, positive): &(usize, bool)| (z[axis] > 0.0) == positive;

                indicator(terms.iter().any(|term| term.iter().all(holds)))
            }
            Rule::Tanh { units } => {
                indicator(units.iter().map(|(v, w)| v * dot(w, z).tanh()).sum::<f64>() > 0.0)
            }
        }
    }

    /// Drift tasks: the label under negated weights.
    fn late_label(&self, z: &[f64]) -> f64 {
        match self {
            Rule::Halfspace { w, b } if -dot(w, z) + b > 0.0 => 1.0,
            _ => 0.0,
        }
    }
}

/// Raw examples of one split.
struct SplitData {
    x: Vec<f64>,
    y: Vec<f64>,
    late: Vec<f64>,
}

fn generate_split(
    plan: &Plan,
    q: &[f64],
    f: usize,
    n: usize,
    train: bool,
    rng: &mut ChaCha8Rng,
) -> SplitData {
    let mut x = vec![0.0; n * f];
    let mut y = Vec::with_capacity(n);
    let mut late = Vec::with_capacity(if plan.drift { n } else { 0 });
    let mut filler = vec![0.0; f];

    for (i, row) in x.chunks_exact_mut(f).enumerate() {
        let (z, hint) = plan.latent.draw(i, plan.d, rng);
        let z = &z[..plan.d];

        embed(q, f, z, rng, &mut filler, row);

        let mut label = plan.rule.label(z, row, hint, rng);

        if train && plan.flip > 0.0 && unit_f64(rng) < plan.flip {
            label = 1.0 - label;
        }

        y.push(label);

        if plan.drift {
            late.push(plan.rule.late_label(z));
        }
    }

    SplitData { x, y, late }
}

fn classes(values: &[f64]) -> Labels {
    Labels::Classes(values.iter().map(|&v| v as u8).collect())
}

/// Materialise registry entry `idx` (params already valid).
fn build_hand(idx: usize, seed: u64, p: &InstanceParams) -> TaskInstance {
    let f = p.f;
    let mut params = ChaCha8Rng::seed_from_u64(derive_seed(seed, "gen/params"));
    let plan = plan(idx, p.level, f, &mut params);
    let q = orthonormal(&mut params, f);

    let mut train_rng = ChaCha8Rng::seed_from_u64(derive_seed(seed, "gen/train"));
    let mut valid_rng = ChaCha8Rng::seed_from_u64(derive_seed(seed, "gen/valid"));
    let train = generate_split(&plan, &q, f, p.n_train, true, &mut train_rng);
    let valid = generate_split(&plan, &q, f, p.n_valid, false, &mut valid_rng);

    let (x_train, x_valid) = standardize_features(&train.x, &valid.x, f);

    let (y_train, y_valid, y_train_late, drift_at) = match plan.kind {
        LearnKind::Regression => {
            let (t, v) = standardize_values(&train.y, &valid.y);

            (Labels::Values(t), Labels::Values(v), None, None)
        }
        _ if plan.drift => (
            classes(&train.y),
            classes(&valid.late),
            Some(classes(&train.late)),
            Some(STEPS_PER_TASK / 2),
        ),
        _ => (classes(&train.y), classes(&valid.y), None, None),
    };

    let family = REGISTRY[idx].id;

    TaskInstance {
        id: instance_id(family, seed, p),
        family: FamilyId(family.to_string()),
        group: REGISTRY[idx].group,
        f,
        kind: plan.kind,
        x_train,
        y_train,
        y_train_late,
        drift_at,
        x_valid,
        y_valid,
        n_train: p.n_train,
        n_valid: p.n_valid,
        steps: STEPS_PER_TASK,
        seed,
        changed_mask: None,
    }
}

/// Materialise one instance of a hand-written family (Err for unknown ids or bad params).
pub fn hand_instance(fam: &FamilyId, seed: u64, p: &InstanceParams) -> Result<TaskInstance> {
    let idx =
        registry_index(&fam.0).ok_or_else(|| BrutError::Invalid("unknown hand family".into()))?;

    check_params(p)?;

    Ok(build_hand(idx, seed, p))
}

/// One search panel (section 6.4).
#[derive(Clone, Debug, PartialEq)]
pub struct Panel {
    /// SHA-256-derived id over the task ids (changes on every rotation).
    pub id: u64,
    pub rotation: u64,
    pub tasks: Vec<TaskInstance>,
    pub stage_a: Vec<usize>,
}

/// Panel after `rotation` rotations (deterministic from master_seed).
///
/// Positions 0..hand_tasks: SEARCH hand families in order (wrapping if hand_tasks > 10).
/// Then pool[(p - hand_tasks + grammar_tasks * g) mod 64] (errata E12). Instance seed
/// derive_seed(master, "panel/<p>/<g>"); params from `sample_params` with that seed.
/// `rotate_every = 0` pins every position at generation 0. Bad params are sanitised
/// (MetaConfig::validate rejects them earlier). E2 (ARC cell at position 19) is off.
pub fn search_panel(
    master_seed: u64,
    rotation: u64,
    cfg: &PanelConfig,
    dims: &[usize],
    n_train_choices: &[usize],
    n_valid: usize,
) -> Panel {
    let hand = hand_indices(Tier::Search);
    let pool = grammar::public_pool_slice(Tier::Search);
    let len = cfg.hand_tasks + cfg.grammar_tasks;
    let mut tasks = Vec::with_capacity(len);

    for p in 0..len {
        let g = position_generation(p, rotation, cfg);
        let seed = derive_seed(master_seed, &format!("panel/{p}/{g}"));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let params = sample_params(&mut rng, dims, n_train_choices, n_valid);
        let sane = sanitize_sizes(&params);

        let task = if p >= cfg.hand_tasks && !pool.is_empty() {
            let offset = (p - cfg.hand_tasks) as u128 + cfg.grammar_tasks as u128 * u128::from(g);
            let j = (offset % pool.len() as u128) as usize;

            grammar::grammar_instance(&pool[j], seed, &sane)
        } else {
            build_hand(hand[p % hand.len()], seed, &sane)
        };

        tasks.push(task);
    }

    let mut hasher = Sha256::new();

    for t in &tasks {
        hasher.update(t.id.to_le_bytes());
    }

    let stage_a = cfg
        .stage_a_positions
        .iter()
        .copied()
        .filter(|&p| p < len)
        .collect();

    Panel {
        id: first_u64(&hasher.finalize()),
        rotation,
        tasks,
        stage_a,
    }
}

/// Generation of position p after `rotation` rotations: the number of r' in 1..=rotation
/// with p in rotation_positions(r'). Closed form over the cyclic sequence of swapped slots.
pub fn position_generation(p: usize, rotation: u64, cfg: &PanelConfig) -> u64 {
    let len = (cfg.hand_tasks + cfg.grammar_tasks) as u128;
    let swap = cfg.rotate_swap as u128;

    if cfg.rotate_every == 0 || rotation == 0 || len == 0 || swap == 0 || p as u128 >= len {
        return 0;
    }

    if swap >= len {
        return rotation;
    }

    // Rotation r' swaps sequence indices swap(r'-1) .. swap r' - 1 (mod len).
    let covered = swap * u128::from(rotation);
    let p = p as u128;

    if covered <= p {
        0
    } else {
        ((covered - 1 - p) / len + 1) as u64
    }
}

/// Positions swapped by rotation r >= 1: {(swap*(r-1) + j) mod len : j in 0..swap}.
/// Sorted, without repeats; empty for r = 0 or an empty panel.
pub fn rotation_positions(r: u64, cfg: &PanelConfig) -> Vec<usize> {
    let len = (cfg.hand_tasks + cfg.grammar_tasks) as u128;

    if r == 0 || len == 0 {
        return Vec::new();
    }

    let start = cfg.rotate_swap as u128 * u128::from(r - 1);
    let mut positions: Vec<usize> = (0..cfg.rotate_swap.min(len as usize))
        .map(|j| ((start + j as u128) % len) as usize)
        .collect();

    positions.sort_unstable();
    positions.dedup();

    positions
}

/// A family of a fixed task set.
#[derive(Clone, Copy, Debug)]
pub enum SetFamily<'a> {
    Hand(&'a FamilyId),
    Grammar(&'a GrammarFamily),
}

impl SetFamily<'_> {
    fn id(&self) -> &str {
        match self {
            SetFamily::Hand(id) => &id.0,
            SetFamily::Grammar(g) => &g.id.0,
        }
    }
}

/// Build a fixed task set by the cycling rule of errata E7: entries in order, `count`
/// instances each (i = 0..count), k = running index over the whole set,
/// seed = seed_of(family id, i), params = cycled_params(k, dims, n_valid, seed).
/// Shared by SELECT tasks, public reference tasks, the baseline tuning set and (WP4) the
/// hidden tiers. Err for unknown hand families or bad params.
pub fn cycled_set(
    entries: &[(SetFamily<'_>, usize)],
    dims: &[usize],
    n_valid: usize,
    seed_of: &dyn Fn(&str, usize) -> u64,
) -> Result<Vec<TaskInstance>> {
    let total: usize = entries.iter().map(|(_, c)| c).sum();
    let mut tasks = Vec::with_capacity(total);
    let mut k = 0;

    for (fam, count) in entries {
        for i in 0..*count {
            let seed = seed_of(fam.id(), i);
            let p = cycled_params(k, dims, n_valid, seed);

            let task = match fam {
                SetFamily::Hand(id) => hand_instance(id, seed, &p)?,
                SetFamily::Grammar(g) => {
                    check_params(&p)?;

                    grammar::grammar_instance(g, seed, &p)
                }
            };

            tasks.push(task);
            k += 1;
        }
    }

    Ok(tasks)
}

/// The SELECT-style set: `hand_per_family` instances of each SELECT hand family, then
/// `grammar_families` SELECT grammar families (one instance each; beyond the pool size the
/// pool repeats with i = 1, 2, ...). F cycles 8, 16, 32; n_valid 64.
pub fn select_like(cfg: &SelectConfig, seed_of: &dyn Fn(&str, usize) -> u64) -> Vec<TaskInstance> {
    let hand = hand_families(Tier::Select);
    let pool = grammar::public_pool_slice(Tier::Select);
    let mut entries: Vec<(SetFamily<'_>, usize)> = hand
        .iter()
        .map(|id| (SetFamily::Hand(id), cfg.hand_per_family))
        .collect();

    let len = pool.len().max(1);

    for (j, fam) in pool.iter().enumerate() {
        let count = cfg.grammar_families / len + usize::from(j < cfg.grammar_families % len);

        if count > 0 {
            entries.push((SetFamily::Grammar(fam), count));
        }
    }

    cycled_set(&entries, &DIMS_CYCLE, SELECT_N_VALID, seed_of).unwrap_or_default()
}

/// Fresh SELECT tasks for Stage C round `round` (hand: hand_per_family each with F cycling
/// 8,16,32; grammar: `grammar_families` from the SELECT pool, one each), n_valid 64.
/// Seeds derive_seed(master, "select/<round>/<family>/<i>").
pub fn select_tasks(master_seed: u64, round: u64, cfg: &SelectConfig) -> Vec<TaskInstance> {
    let seed_of =
        |family: &str, i: usize| derive_seed(master_seed, &format!("select/{round}/{family}/{i}"));

    select_like(cfg, &seed_of)
}

/// 100 fresh public instances of the 10 SEARCH hand families (10 each), F alternating 8/16,
/// n_train cycling 16/64/256, n_valid 256. Used for Δ_public (quarantine).
/// Seeds derive_seed(master, "pubref/<family>/<i>").
pub fn public_reference_tasks(master_seed: u64) -> Vec<TaskInstance> {
    let hand = hand_families(Tier::Search);
    let entries: Vec<(SetFamily<'_>, usize)> = hand
        .iter()
        .map(|id| (SetFamily::Hand(id), PUBLIC_REFERENCE_PER_FAMILY))
        .collect();
    let seed_of =
        |family: &str, i: usize| derive_seed(master_seed, &format!("pubref/{family}/{i}"));

    cycled_set(
        &entries,
        &DIMS_ALTERNATE,
        PUBLIC_REFERENCE_N_VALID,
        &seed_of,
    )
    .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn params(f: usize, n_train: usize, level: u8) -> InstanceParams {
        InstanceParams {
            f,
            n_train,
            n_valid: 64,
            level,
        }
    }

    fn ids(tier: Tier) -> Vec<String> {
        hand_families(tier).into_iter().map(|f| f.0).collect()
    }

    fn class_counts(labels: &Labels, k: usize) -> Vec<usize> {
        let mut counts = vec![0; k];

        if let Labels::Classes(c) = labels {
            for &y in c {
                counts[y as usize] += 1;
            }
        }

        counts
    }

    /// T-FAM-1: golden split-v1 table (plan section 4).
    #[test]
    fn split_v1_matches_the_golden_table() {
        assert_eq!(
            ids(Tier::Search),
            vec![
                "lin-cls",
                "multiclass-lin",
                "quad-reg",
                "bilinear-reg",
                "sphere-cls",
                "blobs-multiclass",
                "relu-teacher-reg",
                "dnf-cls",
                "drift-cls",
                "noisy-label-cls"
            ]
        );
        assert_eq!(
            ids(Tier::Select),
            vec![
                "lin-reg",
                "sign-parity-cls",
                "prototype-cls",
                "step-reg",
                "tanh-teacher-cls"
            ]
        );
        assert_eq!(
            ids(Tier::HiddenNear),
            vec![
                "sparse-lin-reg",
                "xor-cls",
                "rbf-reg",
                "tree-cls",
                "heavy-tail-reg"
            ]
        );
        assert_eq!(
            ids(Tier::HiddenFar),
            vec![
                "sine-reg",
                "checkerboard-cls",
                "spirals-cls",
                "sawtooth-reg"
            ]
        );
        assert!(ids(Tier::HiddenInstance).is_empty());
        assert!(ids(Tier::HiddenGrammar).is_empty());
        assert_eq!(split_v1().len(), 24);
        assert_eq!(
            hand_tier(&FamilyId("xor-cls".into())),
            Some(Tier::HiddenNear)
        );
        assert_eq!(hand_tier(&FamilyId("nope".into())), None);
    }

    #[test]
    fn registry_is_complete_and_flags_developer_seen() {
        let reg = registry();
        let distinct: BTreeSet<&str> = reg.iter().map(|s| s.id).collect();
        let seen: Vec<&str> = reg
            .iter()
            .filter(|s| s.developer_seen)
            .map(|s| s.id)
            .collect();

        assert_eq!(reg.len(), 24);
        assert_eq!(distinct.len(), 24);
        assert_eq!(
            seen,
            vec![
                "xor-cls",
                "sign-parity-cls",
                "sine-reg",
                "checkerboard-cls",
                "spirals-cls"
            ]
        );

        for g in HAND_GROUPS {
            assert_eq!(reg.iter().filter(|s| s.group == g).count(), 4);
        }
    }

    /// Leakage: the four tiers partition the hand families.
    #[test]
    fn tiers_are_disjoint_and_cover_the_registry() {
        let tiers = [
            Tier::Search,
            Tier::Select,
            Tier::HiddenNear,
            Tier::HiddenFar,
        ];
        let mut all = BTreeSet::new();

        for tier in tiers {
            for id in ids(tier) {
                assert!(all.insert(id), "family in two tiers");
            }
        }

        assert_eq!(all.len(), 24);
    }

    #[test]
    fn sample_params_distribution() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut levels = [0usize; 4];
        let mut dims = [0usize; 2];

        for _ in 0..7_000 {
            let p = sample_params(&mut rng, &[8, 16], &[16, 64, 256], 64);

            levels[p.level as usize] += 1;
            dims[usize::from(p.f == 16)] += 1;

            assert!([16, 64, 256].contains(&p.n_train));
            assert_eq!(p.n_valid, 64);
        }

        // P(level) = 4/7, 2/7, 1/7.
        assert!((3_800..=4_200).contains(&levels[1]), "{levels:?}");
        assert!((1_850..=2_150).contains(&levels[2]), "{levels:?}");
        assert!((880..=1_120).contains(&levels[3]), "{levels:?}");
        assert!((3_300..=3_700).contains(&dims[0]));

        let fallback = sample_params(&mut rng, &[], &[], 3);

        assert_eq!((fallback.f, fallback.n_train), (16, 64));
    }

    #[test]
    fn cycled_params_follow_errata_e7() {
        let got: Vec<(usize, usize)> = (0..9)
            .map(|k| {
                let p = cycled_params(k, &DIMS_CYCLE, 64, k as u64);

                (p.f, p.n_train)
            })
            .collect();

        assert_eq!(
            got,
            vec![
                (8, 16),
                (16, 16),
                (32, 16),
                (8, 64),
                (16, 64),
                (32, 64),
                (8, 256),
                (16, 256),
                (32, 256)
            ]
        );

        // Level: sample_params with single-value choices and the instance seed.
        let seed = 12_345;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let level = sample_params(&mut rng, &[16], &[64], 256).level;

        assert_eq!(cycled_params(4, &DIMS_CYCLE, 256, seed).level, level);
        assert_eq!(cycled_params(3, &DIMS_ALTERNATE, 256, seed).n_train, 64);
    }

    /// T-FAM-2: every family × F × n_train builds, is finite, standardised on train.
    #[test]
    fn every_family_builds_for_every_size() {
        for spec in registry() {
            let fam = FamilyId(spec.id.to_string());

            for f in SUPPORTED_F {
                for n_train in N_TRAIN_CYCLE {
                    for level in 1..=3u8 {
                        let seed = instance_id(spec.id, 7, &params(f, n_train, level));
                        let t = hand_instance(&fam, seed, &params(f, n_train, level))
                            .expect("known family");

                        assert_eq!(t.f, f);
                        assert_eq!(t.x_train.len(), n_train * f);
                        assert_eq!(t.x_valid.len(), 64 * f);
                        assert!(t.kind.k_out() <= f);
                        assert!(t.x_train.iter().chain(&t.x_valid).all(|v| v.is_finite()));

                        check_standardised(&t);
                        check_labels(&t);
                    }
                }
            }
        }
    }

    fn check_standardised(t: &TaskInstance) {
        let n = t.n_train as f64;

        for c in 0..t.f {
            let col: Vec<f64> = (0..t.n_train)
                .map(|r| f64::from(t.x_train[r * t.f + c]))
                .collect();
            let mean = col.iter().sum::<f64>() / n;
            let var = col.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;

            assert!(mean.abs() < 1e-4, "{} column {c} mean {mean}", t.family.0);
            assert!(
                (var - 1.0).abs() < 1e-3,
                "{} column {c} var {var}",
                t.family.0
            );
        }
    }

    fn check_labels(t: &TaskInstance) {
        match (&t.y_train, &t.y_valid, t.kind) {
            (Labels::Values(a), Labels::Values(b), LearnKind::Regression) => {
                assert_eq!(a.len(), t.n_train);
                assert_eq!(b.len(), t.n_valid);
                assert!(a.iter().chain(b).all(|v| v.is_finite()));

                let mean = a.iter().map(|&v| f64::from(v)).sum::<f64>() / t.n_train as f64;

                assert!(mean.abs() < 1e-4, "{} target mean {mean}", t.family.0);
            }
            (Labels::Classes(a), Labels::Classes(b), kind) => {
                let k = match kind {
                    LearnKind::Multiclass { k } => k as usize,
                    _ => 2,
                };

                assert_eq!(a.len(), t.n_train);
                assert_eq!(b.len(), t.n_valid);
                assert!(a.iter().chain(b).all(|&y| (y as usize) < k));
            }
            _ => panic!("{}: labels do not match kind", t.family.0),
        }
    }

    /// T-FAM-3: determinism, class balance and drift.
    #[test]
    fn instances_are_deterministic_and_seed_sensitive() {
        let fam = FamilyId("relu-teacher-reg".into());
        let p = params(16, 64, 2);
        let a = hand_instance(&fam, 99, &p).expect("build");
        let b = hand_instance(&fam, 99, &p).expect("build");
        let c = hand_instance(&fam, 100, &p).expect("build");

        assert_eq!(a, b);
        assert_ne!(a.x_train, c.x_train);
        assert_ne!(a.id, c.id);
        assert_eq!(a.id, instance_id("relu-teacher-reg", 99, &p));
        assert_eq!(a.steps, STEPS_PER_TASK);
        assert_eq!(a.seed, 99);
    }

    #[test]
    fn class_balance_is_within_bounds() {
        for spec in registry() {
            let fam = FamilyId(spec.id.to_string());
            let mut majority_sum = 0.0;
            let mut worst: f64 = 0.0;
            let mut count = 0.0;

            for s in 0..20u64 {
                let level = 1 + (s % 3) as u8;
                let t = hand_instance(&fam, 1_000 + s, &params(16, 256, level)).expect("build");
                let k = match t.kind {
                    LearnKind::Regression => continue,
                    LearnKind::Binary => 2,
                    LearnKind::Multiclass { k } => k as usize,
                };
                let train = class_counts(&t.y_train, k);
                let n: usize = train.iter().sum();
                let majority = *train.iter().max().unwrap_or(&0) as f64 / n as f64;

                majority_sum += majority;
                worst = worst.max(majority);
                count += 1.0;
            }

            if count > 0.0 {
                let mean = majority_sum / count;

                assert!(mean <= 0.80, "{}: mean majority share {mean}", spec.id);
                assert!(worst <= 0.97, "{}: worst majority share {worst}", spec.id);
            }
        }
    }

    #[test]
    fn label_noise_hits_train_only() {
        // Validation labels must equal the clean rule: rebuild the validation split from
        // the same streams and compare (validation draws no flip coins).
        let fam = FamilyId("noisy-label-cls".into());
        let p = params(8, 256, 1);
        let t = hand_instance(&fam, 5, &p).expect("build");
        let idx = registry_index("noisy-label-cls").expect("known");
        let mut params_rng = ChaCha8Rng::seed_from_u64(derive_seed(5, "gen/params"));
        let plan = plan(idx, 1, 8, &mut params_rng);
        let q = orthonormal(&mut params_rng, 8);
        let mut valid_rng = ChaCha8Rng::seed_from_u64(derive_seed(5, "gen/valid"));
        let clean_valid = generate_split(&plan, &q, 8, 64, false, &mut valid_rng);

        assert!((plan.flip - NOISY_LABEL_FLIP).abs() < 1e-12);
        assert_eq!(t.y_valid, classes(&clean_valid.y));
    }

    #[test]
    fn noisy_label_flip_rate_is_about_thirty_percent() {
        // A halfspace with flip probability 0.3: the share of train labels that disagree
        // with the clean rule is about 0.3 (recomputed from the same latent stream).
        let idx = registry_index("noisy-label-cls").expect("known");
        let mut disagree = 0usize;
        let mut total = 0usize;

        for s in 0..10u64 {
            let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(s, "gen/params"));
            let plan = plan(idx, 1, 8, &mut rng);
            let q = orthonormal(&mut rng, 8);
            let mut train_rng = ChaCha8Rng::seed_from_u64(derive_seed(s, "gen/train"));
            let data = generate_split(&plan, &q, 8, 256, true, &mut train_rng);

            // Recover z from x: x = Q [z; ξ], so z = first d entries of Q^T x.
            for (row, &y) in data.x.chunks_exact(8).zip(&data.y) {
                let z: Vec<f64> = (0..plan.d)
                    .map(|j| (0..8).map(|r| q[r * 8 + j] * row[r]).sum())
                    .collect();
                let clean = plan.rule.label(&z, row, None, &mut rng);

                disagree += usize::from(clean != y);
                total += 1;
            }
        }

        let rate = disagree as f64 / total as f64;

        assert!((0.26..=0.34).contains(&rate), "flip rate {rate}");
    }

    #[test]
    fn drift_labels_flip_at_half_the_stream() {
        let fam = FamilyId("drift-cls".into());
        let t = hand_instance(&fam, 3, &params(16, 64, 2)).expect("build");

        assert_eq!(t.drift_at, Some(STEPS_PER_TASK / 2));

        let (Labels::Classes(early), Some(Labels::Classes(late))) = (&t.y_train, &t.y_train_late)
        else {
            panic!("drift labels expected");
        };

        assert_eq!(early.len(), late.len());
        assert!(early.iter().zip(late).all(|(&a, &b)| a + b == 1));

        // Validation uses the post-drift rule: about half of them, not all equal.
        let Labels::Classes(valid) = &t.y_valid else {
            panic!("classes expected");
        };

        assert!(valid.contains(&0) && valid.contains(&1));
    }

    #[test]
    fn hand_instance_rejects_bad_input() {
        let fam = FamilyId("lin-reg".into());

        assert!(hand_instance(&FamilyId("unknown".into()), 1, &params(8, 16, 1)).is_err());
        assert!(hand_instance(&fam, 1, &params(7, 16, 1)).is_err());
        assert!(hand_instance(&fam, 1, &params(8, 0, 1)).is_err());
        assert!(hand_instance(&fam, 1, &params(8, MAX_ROWS + 1, 1)).is_err());
        assert!(hand_instance(&fam, 1, &params(8, 16, 0)).is_err());
        assert!(hand_instance(&fam, 1, &params(8, 16, 4)).is_err());
        assert!(hand_instance(&fam, 1, &params(8, 1, 1)).is_ok());
    }

    #[test]
    fn orthonormal_is_orthonormal() {
        let mut rng = ChaCha8Rng::seed_from_u64(4);

        for f in SUPPORTED_F {
            let q = orthonormal(&mut rng, f);

            for i in 0..f {
                for j in 0..f {
                    let d: f64 = (0..f).map(|k| q[i * f + k] * q[j * f + k]).sum();
                    let want = if i == j { 1.0 } else { 0.0 };

                    assert!((d - want).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn sphere_median_splits_gaussian_mass_in_half() {
        let mut rng = ChaCha8Rng::seed_from_u64(8);

        for d in 2..=4usize {
            let inside = (0..40_000)
                .filter(|_| {
                    let r2: f64 = (0..d).map(|_| normal_f64(&mut rng).powi(2)).sum();

                    r2 < CHI2_MEDIAN[d - 2]
                })
                .count();

            assert!((19_400..=20_600).contains(&inside), "d {d}: {inside}");
        }
    }

    /// T-FAM-4: rotation positions and generations (section 6.4).
    #[test]
    fn rotation_positions_and_generations() {
        let cfg = PanelConfig::default();

        assert!(rotation_positions(0, &cfg).is_empty());
        assert_eq!(rotation_positions(1, &cfg), vec![0, 1, 2, 3, 4]);
        assert_eq!(rotation_positions(2, &cfg), vec![5, 6, 7, 8, 9]);
        assert_eq!(rotation_positions(4, &cfg), vec![15, 16, 17, 18, 19]);
        assert_eq!(rotation_positions(5, &cfg), vec![0, 1, 2, 3, 4]);

        let odd = PanelConfig {
            rotate_swap: 7,
            ..PanelConfig::default()
        };

        // Rotation 3 starts at 14 and wraps: {14, ..., 19, 0}.
        assert_eq!(rotation_positions(3, &odd), vec![0, 14, 15, 16, 17, 18, 19]);

        // Brute force against the closed form for several swap sizes.
        for swap in [0, 1, 3, 5, 7, 19, 20, 33] {
            let cfg = PanelConfig {
                rotate_swap: swap,
                ..PanelConfig::default()
            };

            for p in 0..20 {
                let mut g = 0u64;

                for r in 0..=45u64 {
                    if r > 0 && rotation_positions(r, &cfg).contains(&p) {
                        g += 1;
                    }

                    assert_eq!(
                        position_generation(p, r, &cfg),
                        g,
                        "swap {swap} p {p} r {r}"
                    );
                }
            }
        }

        let fixed = PanelConfig {
            rotate_every: 0,
            ..PanelConfig::default()
        };

        assert_eq!(position_generation(3, 10, &fixed), 0);
    }

    #[test]
    fn panel_layout_and_rotation_changes_exactly_the_swapped_positions() {
        let cfg = PanelConfig::default();
        let search = hand_families(Tier::Search);
        let pool = grammar::public_pool_slice(Tier::Search);
        let p0 = search_panel(1, 0, &cfg, &[8, 16], &[16, 64, 256], 64);

        assert_eq!(p0.tasks.len(), 20);
        assert_eq!(p0.stage_a, vec![0, 3, 6, 10, 15]);
        assert_eq!(p0, search_panel(1, 0, &cfg, &[8, 16], &[16, 64, 256], 64));

        for (p, t) in p0.tasks.iter().enumerate() {
            if p < 10 {
                assert_eq!(t.family, search[p]);
            } else {
                assert_eq!(t.family, pool[p - 10].id);
            }

            assert!([8, 16].contains(&t.f));
            assert_eq!(t.n_valid, 64);
        }

        let mut previous = p0;

        for r in 1..=6u64 {
            let next = search_panel(1, r, &cfg, &[8, 16], &[16, 64, 256], 64);
            let swapped = rotation_positions(r, &cfg);

            for p in 0..20 {
                let changed = next.tasks[p].id != previous.tasks[p].id;

                assert_eq!(changed, swapped.contains(&p), "rotation {r} position {p}");

                if p >= 10 {
                    let g = position_generation(p, r, &cfg) as usize;

                    assert_eq!(
                        next.tasks[p].family,
                        pool[(p - 10 + 10 * g) % pool.len()].id
                    );
                }
            }

            assert_ne!(next.id, previous.id);
            assert_eq!(next.rotation, r);

            previous = next;
        }

        assert_ne!(
            search_panel(2, 0, &cfg, &[8, 16], &[16, 64, 256], 64).id,
            search_panel(1, 0, &cfg, &[8, 16], &[16, 64, 256], 64).id
        );
    }

    #[test]
    fn select_tasks_follow_the_contract() {
        let cfg = SelectConfig::default();
        let tasks = select_tasks(1, 0, &cfg);
        let select_hand: BTreeSet<String> = ids(Tier::Select).into_iter().collect();
        let select_pool: BTreeSet<FamilyId> = grammar::public_pool_slice(Tier::Select)
            .iter()
            .map(|f| f.id.clone())
            .collect();

        assert_eq!(tasks.len(), 5 * 8 + 32);
        assert_eq!(tasks, select_tasks(1, 0, &cfg));
        assert_ne!(tasks[0].id, select_tasks(1, 1, &cfg)[0].id);

        for (k, t) in tasks.iter().enumerate() {
            assert_eq!(t.n_valid, 64);
            assert_eq!(t.f, DIMS_CYCLE[k % 3]);
            assert_eq!(t.n_train, N_TRAIN_CYCLE[(k / 3) % 3]);

            if k < 40 {
                assert!(select_hand.contains(&t.family.0));
                assert_eq!(t.family.0, ids(Tier::Select)[k / 8]);
            } else {
                assert!(select_pool.contains(&t.family));
            }
        }

        let seed = derive_seed(1, "select/0/lin-reg/3");

        assert_eq!(tasks[3].seed, seed);
        assert_eq!(
            tasks[40].seed,
            derive_seed(1, "select/0/grammar/select/000/0")
        );
    }

    #[test]
    fn select_tasks_repeat_the_pool_when_asked_for_more_families() {
        let cfg = SelectConfig {
            hand_per_family: 1,
            grammar_families: 33,
            ..SelectConfig::default()
        };
        let tasks = select_tasks(1, 0, &cfg);
        let ids: BTreeSet<u64> = tasks.iter().map(|t| t.id).collect();

        assert_eq!(tasks.len(), 5 + 33);
        assert_eq!(ids.len(), tasks.len());
    }

    #[test]
    fn public_reference_tasks_follow_the_contract() {
        let tasks = public_reference_tasks(1);
        let search = ids(Tier::Search);

        assert_eq!(tasks.len(), 100);

        for (k, t) in tasks.iter().enumerate() {
            assert_eq!(t.family.0, search[k / 10]);
            assert_eq!(t.f, DIMS_ALTERNATE[k % 2]);
            assert_eq!(t.n_train, N_TRAIN_CYCLE[(k / 2) % 3]);
            assert_eq!(t.n_valid, 256);
            assert_eq!(
                t.seed,
                derive_seed(1, &format!("pubref/{}/{}", search[k / 10], k % 10))
            );
        }
    }

    /// Leakage: no public task set uses a SELECT or hidden family where it must not, and
    /// SEARCH-side sets never touch hidden families.
    #[test]
    fn public_task_sets_never_use_hidden_families() {
        let hidden: BTreeSet<String> = ids(Tier::HiddenNear)
            .into_iter()
            .chain(ids(Tier::HiddenFar))
            .collect();
        let select: BTreeSet<String> = ids(Tier::Select).into_iter().collect();
        let cfg = PanelConfig::default();
        let select_pool: BTreeSet<FamilyId> = grammar::public_pool_slice(Tier::Select)
            .iter()
            .map(|f| f.id.clone())
            .collect();

        for r in 0..12 {
            for t in search_panel(3, r, &cfg, &[8, 16], &[16, 64, 256], 64).tasks {
                assert!(!hidden.contains(&t.family.0));
                assert!(!select.contains(&t.family.0));
                assert!(!select_pool.contains(&t.family));
            }
        }

        for t in public_reference_tasks(3) {
            assert!(!hidden.contains(&t.family.0));
            assert!(!select.contains(&t.family.0));
        }

        for t in select_tasks(3, 0, &SelectConfig::default()) {
            assert!(!hidden.contains(&t.family.0));
        }
    }

    #[test]
    fn cycled_set_rejects_unknown_families() {
        let bad = FamilyId("nope".into());
        let seed_of = |_: &str, i: usize| i as u64;

        assert!(cycled_set(&[(SetFamily::Hand(&bad), 1)], &DIMS_CYCLE, 64, &seed_of).is_err());
        assert!(
            cycled_set(&[], &DIMS_CYCLE, 64, &seed_of)
                .expect("empty")
                .is_empty()
        );
    }
}
