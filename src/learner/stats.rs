//! Statistics: sign-flip test, cluster bootstrap, Holm, Spearman, rolling-quantile hurdle.
//! Also the small random-number helpers that the WP2 modules share.
//!
//! Owner: WP2. Contract section 5.10. Pure: no I/O.
//!
//! The helpers draw only through `RngCore::next_u64`, so every sample is pinned by the
//! ChaCha8 stream alone and does not depend on how `rand` turns bits into numbers.
#![forbid(unsafe_code)]

use std::collections::VecDeque;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::learner::types::HurdleConfig;

/// Largest sample size the sign-flip test enumerates exactly (2^20 sign patterns).
pub const EXACT_SIGN_FLIP_MAX: usize = 20;

/// Relative tolerance (times sum |d|) under which a permuted sum counts as tied with the
/// observed one. Guards against rounding making mathematically equal sums look smaller.
const TIE_TOLERANCE: f64 = 1e-9;

/// Slack subtracted from q * n before the ceiling in the nearest-rank quantile, so that
/// products such as 0.7 * 10 = 7.000000000000001 still give rank 7.
const RANK_SLACK: f64 = 1e-9;

/// Uniform integer in 0..n (Lemire's multiply-and-reject; unbiased). Returns 0 for n = 0.
pub fn uniform_below(rng: &mut ChaCha8Rng, n: u64) -> u64 {
    if n == 0 {
        return 0;
    }

    let threshold = n.wrapping_neg() % n;

    loop {
        let product = u128::from(rng.next_u64()) * u128::from(n);
        let low = product as u64;

        if low >= threshold {
            return (product >> 64) as u64;
        }
    }
}

/// Uniform index in 0..n (0 for n = 0).
pub fn uniform_index(rng: &mut ChaCha8Rng, n: usize) -> usize {
    uniform_below(rng, n as u64) as usize
}

/// Uniform f64 in [0, 1) with 53 random bits.
pub fn unit_f64(rng: &mut ChaCha8Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Uniform f64 in [lo, hi).
pub fn range_f64(rng: &mut ChaCha8Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * unit_f64(rng)
}

/// One standard normal sample (Box-Muller, cosine branch). Always finite.
pub fn normal_f64(rng: &mut ChaCha8Rng) -> f64 {
    let u1 = unit_f64(rng);
    let u2 = unit_f64(rng);

    // 1 - u1 is in (0, 1], so the logarithm stays finite.
    let radius = (-2.0 * (1.0 - u1).ln()).sqrt();

    radius * (std::f64::consts::TAU * u2).cos()
}

/// Fisher-Yates shuffle driven by `uniform_below`.
pub fn shuffle<T>(rng: &mut ChaCha8Rng, items: &mut [T]) {
    for i in (1..items.len()).rev() {
        let j = uniform_index(rng, i + 1);

        items.swap(i, j);
    }
}

/// One-sided sign-flip permutation p-value for H1: mean(d) > 0.
///
/// Exact enumeration of all 2^n sign patterns when n <= 20: p = share of patterns whose
/// sum is >= the observed sum (identity included). Otherwise `reps` Monte-Carlo patterns
/// from `seed` and p = (1 + #{patterns with sum >= observed}) / (1 + reps).
/// Empty input, non-finite values, or reps = 0 in the Monte-Carlo case give p = 1.
pub fn sign_flip_pvalue(d: &[f64], reps: usize, seed: u64) -> f64 {
    let n = d.len();

    if n == 0 || d.iter().any(|x| !x.is_finite()) {
        return 1.0;
    }

    let observed: f64 = d.iter().sum();
    let scale: f64 = d.iter().map(|x| x.abs()).sum();
    let bar = observed - TIE_TOLERANCE * scale;

    if n <= EXACT_SIGN_FLIP_MAX {
        let patterns = 1u64 << n;
        let mut hits = 0u64;

        for mask in 0..patterns {
            if flipped_sum(d, mask) >= bar {
                hits += 1;
            }
        }

        return hits as f64 / patterns as f64;
    }

    if reps == 0 {
        return 1.0;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut hits = 0u64;

    for _ in 0..reps {
        let mut sum = 0.0;
        let mut bits = 0u64;

        for (i, &x) in d.iter().enumerate() {
            if i % 64 == 0 {
                bits = rng.next_u64();
            }

            if bits & 1 == 1 {
                sum -= x;
            } else {
                sum += x;
            }

            bits >>= 1;
        }

        if sum >= bar {
            hits += 1;
        }
    }

    (hits as f64 + 1.0) / (reps as f64 + 1.0)
}

/// Sum of d with element i negated where bit i of `mask` is set (summed in index order).
fn flipped_sum(d: &[f64], mask: u64) -> f64 {
    let mut sum = 0.0;

    for (i, &x) in d.iter().enumerate() {
        if (mask >> i) & 1 == 1 {
            sum -= x;
        } else {
            sum += x;
        }
    }

    sum
}

/// Percentile CI of the mean over clusters (resample clusters with replacement).
///
/// The statistic is the mean of the cluster means (every cluster weighs the same, as S_T
/// in contract section 6.8). Non-finite values are dropped; empty clusters are skipped.
/// Percentiles interpolate linearly between order statistics (type 7).
/// Returns (mean, lo, hi); (0, 0, 0) when no cluster has data, (mean, mean, mean) for
/// reps = 0. Unreliable below about 20 clusters: few clusters give few distinct resamples.
pub fn cluster_bootstrap_ci(
    clusters: &[Vec<f64>],
    reps: usize,
    alpha: f64,
    seed: u64,
) -> (f64, f64, f64) {
    let means: Vec<f64> = clusters.iter().filter_map(|c| finite_mean(c)).collect();

    if means.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let m = means.len();
    let mean = means.iter().sum::<f64>() / m as f64;

    if reps == 0 {
        return (mean, mean, mean);
    }

    let alpha = if alpha.is_finite() {
        alpha.clamp(0.0, 1.0)
    } else {
        0.05
    };

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut boot = Vec::with_capacity(reps);

    for _ in 0..reps {
        let mut sum = 0.0;

        for _ in 0..m {
            sum += means[uniform_index(&mut rng, m)];
        }

        boot.push(sum / m as f64);
    }

    boot.sort_by(f64::total_cmp);

    let lo = percentile_sorted(&boot, alpha / 2.0);
    let hi = percentile_sorted(&boot, 1.0 - alpha / 2.0);

    (mean, lo, hi)
}

/// Mean of the finite values of `c`; None when there are none.
fn finite_mean(c: &[f64]) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;

    for &x in c.iter().filter(|x| x.is_finite()) {
        sum += x;
        count += 1;
    }

    (count > 0).then(|| sum / count as f64)
}

/// Type-7 percentile of an ascending slice: interpolate at h = (n - 1) p. 0 when empty.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();

    if n == 0 {
        return 0.0;
    }

    let h = (n - 1) as f64 * p.clamp(0.0, 1.0);
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = h - lo as f64;

    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Holm step-down; returns which hypotheses are rejected at family-wise level alpha.
///
/// Sort p ascending (ties by index); reject p_(i) while p_(i) <= alpha / (m - i) for
/// 0-based rank i; stop at the first failure. NaN p-values are never rejected.
pub fn holm(p: &[f64], alpha: f64) -> Vec<bool> {
    let m = p.len();
    let key = |i: usize| if p[i].is_nan() { f64::INFINITY } else { p[i] };
    let mut order: Vec<usize> = (0..m).collect();

    order.sort_by(|&a, &b| key(a).total_cmp(&key(b)).then(a.cmp(&b)));

    let mut rejected = vec![false; m];

    for (rank, &i) in order.iter().enumerate() {
        let threshold = alpha / (m - rank) as f64;

        if key(i) <= threshold {
            rejected[i] = true;
        } else {
            break;
        }
    }

    rejected
}

/// Spearman rank correlation (average ranks for ties).
///
/// Pairs with a non-finite member are dropped. Returns 0 when the lengths differ, fewer
/// than two pairs remain, or one side has no rank variance (the correlation is undefined).
pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let (xa, xb): (Vec<f64>, Vec<f64>) = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(&x, &y)| (x, y))
        .unzip();

    if xa.len() < 2 {
        return 0.0;
    }

    let ra = average_ranks(&xa);
    let rb = average_ranks(&xb);

    pearson(&ra, &rb).clamp(-1.0, 1.0)
}

/// 1-based ranks; tied values share the mean of their positions.
fn average_ranks(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut order: Vec<usize> = (0..n).collect();

    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(a.cmp(&b)));

    let mut ranks = vec![0.0; n];
    let mut start = 0;

    while start < n {
        let mut end = start + 1;

        while end < n && x[order[end]] == x[order[start]] {
            end += 1;
        }

        // Positions start..end (0-based) hold equal values: their mean 1-based rank.
        let rank = (start + end + 1) as f64 / 2.0;

        for &i in &order[start..end] {
            ranks[i] = rank;
        }

        start = end;
    }

    ranks
}

/// Pearson correlation; 0 when either side has zero variance.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for (&x, &y) in a.iter().zip(b) {
        cov += (x - mean_a) * (y - mean_b);
        var_a += (x - mean_a) * (x - mean_a);
        var_b += (y - mean_b) * (y - mean_b);
    }

    if var_a <= 0.0 || var_b <= 0.0 {
        return 0.0;
    }

    cov / (var_a * var_b).sqrt()
}

/// Rolling-quantile hurdle (AutoML-Zero). Deterministic.
#[derive(Clone, Debug, PartialEq)]
pub struct Hurdle {
    /// Most recent scores, oldest first; never longer than `window`.
    values: VecDeque<f64>,
    window: usize,
    /// In [0, 1].
    quantile: f64,
    warmup: usize,
}

impl Hurdle {
    /// Empty hurdle. A NaN quantile falls back to the default 0.75; others are clamped
    /// to [0, 1].
    pub fn new(cfg: &HurdleConfig) -> Self {
        let quantile = if cfg.quantile.is_nan() {
            HurdleConfig::default().quantile
        } else {
            cfg.quantile.clamp(0.0, 1.0)
        };

        Hurdle {
            values: VecDeque::with_capacity(cfg.window.min(1 << 16)),
            window: cfg.window,
            quantile,
            warmup: cfg.warmup,
        }
    }

    /// None during warm-up (everyone passes), else the current quantile (nearest-rank:
    /// the ceil(q * n)-th smallest value, at least the smallest).
    /// Warm-up lasts while the window holds fewer than `warmup` scores, and while empty.
    pub fn threshold(&self) -> Option<f64> {
        let n = self.values.len();

        if n == 0 || n < self.warmup {
            return None;
        }

        let mut sorted: Vec<f64> = self.values.iter().copied().collect();

        sorted.sort_by(f64::total_cmp);

        let rank = (self.quantile * n as f64 - RANK_SLACK).ceil().max(1.0) as usize;

        Some(sorted[rank.min(n) - 1])
    }

    /// Append a score and drop the oldest beyond the window. Non-finite scores are ignored.
    pub fn push(&mut self, score: f64) {
        if !score.is_finite() || self.window == 0 {
            return;
        }

        self.values.push_back(score);

        while self.values.len() > self.window {
            self.values.pop_front();
        }
    }

    /// Window contents, oldest first (for snapshots).
    pub fn values(&self) -> Vec<f64> {
        self.values.iter().copied().collect()
    }

    /// Rebuild from a snapshot: pushes `values` in order (keeps the newest `window`).
    pub fn from_values(cfg: &HurdleConfig, values: Vec<f64>) -> Self {
        let mut hurdle = Hurdle::new(cfg);

        for v in values {
            hurdle.push(v);
        }

        hurdle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn uniform_below_is_in_range_and_roughly_uniform() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut counts = [0usize; 7];

        for _ in 0..70_000 {
            counts[uniform_index(&mut rng, 7)] += 1;
        }

        assert!(counts.iter().all(|&c| (9_400..=10_600).contains(&c)));
        assert_eq!(uniform_below(&mut rng, 0), 0);
        assert_eq!(uniform_below(&mut rng, 1), 0);
    }

    #[test]
    fn normal_and_unit_helpers_have_the_right_moments() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let n = 40_000;
        let xs: Vec<f64> = (0..n).map(|_| normal_f64(&mut rng)).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;

        assert!(mean.abs() < 0.03);
        assert!((var - 1.0).abs() < 0.03);

        let us: Vec<f64> = (0..n).map(|_| unit_f64(&mut rng)).collect();

        assert!(us.iter().all(|u| (0.0..1.0).contains(u)));
        assert!(close(us.iter().sum::<f64>() / n as f64, 0.5, 0.01));
    }

    #[test]
    fn shuffle_is_a_permutation_and_deterministic() {
        let mut a: Vec<u32> = (0..50).collect();
        let mut b = a.clone();

        shuffle(&mut ChaCha8Rng::seed_from_u64(9), &mut a);
        shuffle(&mut ChaCha8Rng::seed_from_u64(9), &mut b);

        assert_eq!(a, b);
        assert_ne!(a, (0..50).collect::<Vec<u32>>());

        a.sort();

        assert_eq!(a, (0..50).collect::<Vec<u32>>());
    }

    #[test]
    fn sign_flip_exact_matches_hand_counts() {
        // d = [1, 2, 3]: sums by mask 6, 4, 2, 0, 0, -2, -4, -6; only the identity
        // reaches 6, so p = 1/8.
        assert!(close(
            sign_flip_pvalue(&[1.0, 2.0, 3.0], 0, 0),
            0.125,
            1e-15
        ));

        // d = [3, -1]: sums by mask 2, -4, 4, -2; {2, 4} are >= 2, so p = 2/4.
        assert!(close(sign_flip_pvalue(&[3.0, -1.0], 0, 0), 0.5, 1e-15));

        // d = [1, 1, 1, 1]: only all-plus reaches 4, so p = 1/16.
        assert!(close(sign_flip_pvalue(&[1.0; 4], 0, 0), 1.0 / 16.0, 1e-15));

        // d = [1, -1]: observed 0; sums 0, -2, 2, 0, so 3 of 4 are >= 0.
        assert!(close(sign_flip_pvalue(&[1.0, -1.0], 0, 0), 0.75, 1e-15));

        // All zeros: every pattern ties, p = 1.
        assert!(close(sign_flip_pvalue(&[0.0; 5], 0, 0), 1.0, 1e-15));

        // Negative effect: every pattern is >= -6, so p = 1.
        assert!(close(
            sign_flip_pvalue(&[-1.0, -2.0, -3.0], 0, 0),
            1.0,
            1e-15
        ));
    }

    #[test]
    fn sign_flip_ties_survive_rounding() {
        // d = [0.1, 0.2, -0.3, 0.3], observed sum 0.3. With x = +-0.1 +-0.2 and
        // y = 0.3 (s4 - s3), the patterns with x + y >= 0.3 are: y = 0.6 with any x (4)
        // and y = 0 with x = 0.3 (2). Three of them tie mathematically: p = 6/16.
        let p = sign_flip_pvalue(&[0.1, 0.2, -0.3, 0.3], 0, 0);

        assert!(close(p, 6.0 / 16.0, 1e-15));
    }

    #[test]
    fn sign_flip_bad_input_gives_one() {
        assert_eq!(sign_flip_pvalue(&[], 100, 1), 1.0);
        assert_eq!(sign_flip_pvalue(&[1.0, f64::NAN], 100, 1), 1.0);
        assert_eq!(sign_flip_pvalue(&[1.0; 30], 0, 1), 1.0);
    }

    #[test]
    fn sign_flip_monte_carlo_formula_and_determinism() {
        // 30 equal positive values: only the all-plus pattern reaches the observed sum,
        // which a random pattern hits with probability 2^-30, so p = 1 / (1 + reps).
        let p = sign_flip_pvalue(&[0.5; 30], 999, 7);

        assert!(close(p, 1.0 / 1000.0, 1e-15));

        let e: Vec<f64> = (0..25).map(|i| (i as f64 - 11.0) / 10.0).collect();

        assert_eq!(sign_flip_pvalue(&e, 5000, 3), sign_flip_pvalue(&e, 5000, 3));
    }

    #[test]
    fn sign_flip_monte_carlo_agrees_with_exact_on_an_overlap_case() {
        // Exact p for n = 20 against the Monte-Carlo estimate on the same data padded to
        // n = 21 with a zero, which leaves the null distribution of the sum unchanged.
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let d: Vec<f64> = (0..20).map(|_| normal_f64(&mut rng) + 0.3).collect();
        let exact = sign_flip_pvalue(&d, 0, 0);
        let mut padded = d.clone();

        padded.push(0.0);

        let mc = sign_flip_pvalue(&padded, 200_000, 5);
        let se = (exact * (1.0 - exact) / 200_000.0).sqrt();

        assert!(
            (exact - mc).abs() < 5.0 * se + 1e-5,
            "exact {exact} mc {mc}"
        );
    }

    /// T-STAT-1: null calibration of the decision rule p < 0.05 with 40 families.
    /// reps = 2 000 keeps debug builds fast; validity does not depend on reps.
    #[test]
    fn sign_flip_null_calibration() {
        let datasets = 2_000;
        let reps = 2_000;
        let mut data_rng = ChaCha8Rng::seed_from_u64(20_261_002);
        let mut false_positives = 0usize;

        for k in 0..datasets {
            let d: Vec<f64> = (0..40).map(|_| 0.05 * normal_f64(&mut data_rng)).collect();

            if sign_flip_pvalue(&d, reps, 1_000 + k as u64) < 0.05 {
                false_positives += 1;
            }
        }

        let rate = false_positives as f64 / datasets as f64;

        assert!(
            (0.035..=0.065).contains(&rate),
            "false-positive rate {rate}"
        );
    }

    /// T-STAT-2: power sanity with a clear effect.
    #[test]
    fn sign_flip_power_sanity() {
        let mut rng = ChaCha8Rng::seed_from_u64(77);
        let d: Vec<f64> = (0..40)
            .map(|_| 0.05 + 0.05 * normal_f64(&mut rng))
            .collect();

        let p = sign_flip_pvalue(&d, 100_000, 78);

        assert!(p < 0.001, "p = {p}");
    }

    #[test]
    fn bootstrap_two_clusters_has_known_support() {
        // Cluster means 0 and 1. A resample of 2 clusters has mean 0 (p 1/4), 0.5 (1/2)
        // or 1 (1/4), so the 2.5 % and 97.5 % percentiles are 0 and 1.
        let clusters = vec![vec![-1.0, 1.0], vec![1.0, 1.0, 1.0]];
        let (mean, lo, hi) = cluster_bootstrap_ci(&clusters, 4_000, 0.05, 3);

        assert!(close(mean, 0.5, 1e-15));
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 1.0);
    }

    #[test]
    fn bootstrap_weights_clusters_equally_and_skips_bad_values() {
        // Means 1 (one value) and 3 (three finite values) -> statistic 2.
        let clusters = vec![
            vec![1.0],
            vec![2.0, 4.0, f64::NAN, 3.0],
            vec![],
            vec![f64::INFINITY],
        ];

        assert_eq!(cluster_bootstrap_ci(&clusters, 0, 0.05, 1), (2.0, 2.0, 2.0));
        assert_eq!(cluster_bootstrap_ci(&[], 100, 0.05, 1), (0.0, 0.0, 0.0));
    }

    #[test]
    fn bootstrap_interval_matches_normal_theory_for_many_clusters() {
        // 400 clusters of one N(0, 1) value: the percentile CI half-width should be close
        // to 1.96 sd / sqrt(400) of the sample, and the interval contains the mean.
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let clusters: Vec<Vec<f64>> = (0..400).map(|_| vec![normal_f64(&mut rng)]).collect();
        let values: Vec<f64> = clusters.iter().map(|c| c[0]).collect();
        let m = values.iter().sum::<f64>() / 400.0;
        let sd = (values.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / 400.0).sqrt();
        let (mean, lo, hi) = cluster_bootstrap_ci(&clusters, 20_000, 0.05, 6);
        let expected_half = 1.96 * sd / 20.0;

        assert!(close(mean, m, 1e-12));
        assert!(lo < mean && mean < hi);
        assert!(close((hi - lo) / 2.0, expected_half, 0.1 * expected_half));
    }

    #[test]
    fn percentile_type7_hand_values() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(percentile_sorted(&x, 0.0), 1.0);
        assert_eq!(percentile_sorted(&x, 1.0), 5.0);
        assert_eq!(percentile_sorted(&x, 0.5), 3.0);
        // h = 4 * 0.1 = 0.4 -> 1 + 0.4 * (2 - 1) = 1.4.
        assert!(close(percentile_sorted(&x, 0.1), 1.4, 1e-12));
    }

    #[test]
    fn holm_hand_fixtures() {
        // m = 4, thresholds 0.0125, 0.01667, 0.025, 0.05 on the sorted p:
        // 0.01 (reject), 0.015 (reject), 0.03 > 0.025 (stop).
        assert_eq!(
            holm(&[0.03, 0.01, 0.04, 0.015], 0.05),
            vec![false, true, false, true]
        );

        assert_eq!(holm(&[0.001, 0.002, 0.003], 0.05), vec![true; 3]);

        // Step-down stops at the first failure, although 0.04 <= 0.05 alone.
        assert_eq!(holm(&[0.03, 0.04], 0.05), vec![false, false]);

        // Boundary: p equal to its threshold is rejected.
        assert_eq!(holm(&[0.025, 0.05], 0.05), vec![true, true]);

        // Ties, NaN and empty input.
        assert_eq!(holm(&[0.01, 0.01], 0.05), vec![true, true]);
        assert_eq!(holm(&[f64::NAN, 0.001], 0.05), vec![false, true]);
        assert!(holm(&[], 0.05).is_empty());
    }

    #[test]
    fn holm_is_less_conservative_than_bonferroni() {
        // Bonferroni at 0.05 / 3 rejects only 0.01; Holm also rejects 0.02 (threshold
        // 0.025) and 0.04 (threshold 0.05).
        assert_eq!(holm(&[0.01, 0.02, 0.04], 0.05), vec![true, true, true]);
    }

    #[test]
    fn spearman_known_values() {
        // Textbook example (IQ against hours of TV): rho = -29 / 165.
        let iq = [
            106.0, 100.0, 86.0, 101.0, 99.0, 103.0, 97.0, 113.0, 112.0, 110.0,
        ];
        let tv = [7.0, 27.0, 2.0, 50.0, 28.0, 29.0, 20.0, 12.0, 6.0, 17.0];

        assert!(close(spearman(&iq, &tv), -29.0 / 165.0, 1e-12));

        // Monotone transforms give +-1.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0, 8.0, 27.0, 64.0, 125.0];
        let c = [5.0, 4.0, 3.0, 2.0, 1.0];

        assert!(close(spearman(&a, &b), 1.0, 1e-12));
        assert!(close(spearman(&a, &c), -1.0, 1e-12));
    }

    #[test]
    fn spearman_with_ties_uses_average_ranks() {
        // Ranks a: 1, 2.5, 2.5, 4; b: 1, 2, 3, 4.
        // cov = 4.5, var_a = 4.5, var_b = 5 -> rho = 4.5 / sqrt(22.5).
        let a = [1.0, 2.0, 2.0, 3.0];
        let b = [10.0, 20.0, 30.0, 40.0];

        assert_eq!(average_ranks(&a), vec![1.0, 2.5, 2.5, 4.0]);
        assert!(close(spearman(&a, &b), 4.5 / 22.5f64.sqrt(), 1e-12));
    }

    #[test]
    fn spearman_degenerate_inputs_give_zero() {
        assert_eq!(spearman(&[1.0, 2.0], &[1.0]), 0.0);
        assert_eq!(spearman(&[1.0], &[1.0]), 0.0);
        assert_eq!(spearman(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]), 0.0);

        // The NaN pair is dropped; the rest is perfectly monotone.
        let rho = spearman(&[1.0, f64::NAN, 2.0, 3.0], &[1.0, 5.0, 2.0, 3.0]);

        assert!(close(rho, 1.0, 1e-12));
    }

    #[test]
    fn hurdle_nearest_rank_and_warmup() {
        let cfg = HurdleConfig {
            quantile: 0.75,
            window: 4,
            warmup: 3,
        };
        let mut h = Hurdle::new(&cfg);

        assert_eq!(h.threshold(), None);

        h.push(0.4);
        h.push(0.1);

        assert_eq!(h.threshold(), None);

        h.push(0.3);

        // n = 3: rank ceil(2.25) = 3, the largest of {0.1, 0.3, 0.4}.
        assert_eq!(h.threshold(), Some(0.4));

        h.push(0.2);

        // n = 4: rank 3 of {0.1, 0.2, 0.3, 0.4}.
        assert_eq!(h.threshold(), Some(0.3));

        h.push(0.9);

        // The window drops 0.4: rank 3 of {0.1, 0.2, 0.3, 0.9}.
        assert_eq!(h.values(), vec![0.1, 0.3, 0.2, 0.9]);
        assert_eq!(h.threshold(), Some(0.3));

        h.push(f64::NAN);

        assert_eq!(h.values().len(), 4);
    }

    #[test]
    fn hurdle_rank_slack_and_extremes() {
        // q = 0.7, n = 10: 0.7 * 10 is 7.000000000000001 in f64; the rank must be 7.
        let cfg = HurdleConfig {
            quantile: 0.7,
            window: 100,
            warmup: 0,
        };
        let h = Hurdle::from_values(&cfg, (1..=10).map(f64::from).collect());

        assert_eq!(h.threshold(), Some(7.0));

        let at = |quantile: f64| {
            let cfg = HurdleConfig {
                quantile,
                window: 10,
                warmup: 1,
            };

            Hurdle::from_values(&cfg, vec![3.0, 1.0, 2.0]).threshold()
        };

        assert_eq!(at(0.0), Some(1.0));
        assert_eq!(at(1.0), Some(3.0));
        assert_eq!(at(f64::NAN), Some(3.0));
    }

    #[test]
    fn hurdle_snapshot_round_trip_keeps_the_newest_window() {
        let cfg = HurdleConfig::default();
        let mut h = Hurdle::new(&cfg);

        for i in 0..1_500 {
            h.push(f64::from(i % 97) / 97.0);
        }

        let values = h.values();
        let back = Hurdle::from_values(&cfg, values.clone());

        assert_eq!(values.len(), 1_000);
        assert_eq!(back, h);
        assert_eq!(back.threshold(), h.threshold());

        let long: Vec<f64> = (0..1_200).map(f64::from).collect();
        let trimmed = Hurdle::from_values(&cfg, long);

        assert_eq!(trimmed.values().first(), Some(&200.0));
    }
}
