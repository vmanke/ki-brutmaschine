//! Parent selection strategies that look at single tasks or at ranks (research brief:
//! evolutionary selection improvements), next to the Darwin Gödel weights of `arena.rs`.
//!
//! * **Lexicase** (Spector 2012; La Cava et al. 2016 for the epsilon form): to pick one
//!   parent, shuffle the search tasks, then walk through them. On each task only the
//!   candidates that score within `epsilon` of the best remaining score stay. A single
//!   survivor wins; if the tasks run out first, a survivor is drawn uniformly. A candidate
//!   that is the only one to do well on a rare task can therefore be chosen although its
//!   mean is low; no aggregate score enters.
//! * **Tournament**: draw `size` members uniformly with replacement and take the one that
//!   ranks best (fitness, then peer rating, then id, as everywhere else).
//!
//! Everything here is a pure function of its inputs and the `rng` it is given, so a
//! selection event is reproducible from its seed. Scores that are NaN count as 0.
#![forbid(unsafe_code)]

use rand::Rng;
use rand::seq::{IndexedRandom, SliceRandom};
use rand_chacha::ChaCha8Rng;

/// Added to `epsilon` when comparing scores, so that sums of floats that differ in the last
/// bits still tie. With `epsilon` 0 this is the only tolerance: scores tie when they differ
/// by no more than 1e-12.
pub const TIE_SLACK: f64 = 1e-12;

/// Settings of the selection strategies beyond the choice of strategy. The defaults give
/// exact ties for lexicase and tournaments of three. Left out of a run's recorded settings
/// while they are at their defaults, so earlier configs and hashes do not change.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionParams {
    /// Lexicase: a candidate stays on a task when its score is at least the best remaining
    /// score minus this. In score units (a task is worth at most 1).
    #[serde(default)]
    pub lexicase_epsilon: f64,
    /// Tournament: members drawn per tournament.
    #[serde(default = "default_tournament_size")]
    pub tournament_size: usize,
}

fn default_tournament_size() -> usize {
    3
}

impl Default for SelectionParams {
    fn default() -> Self {
        SelectionParams {
            lexicase_epsilon: 0.0,
            tournament_size: default_tournament_size(),
        }
    }
}

impl SelectionParams {
    pub fn is_default(&self) -> bool {
        *self == SelectionParams::default()
    }

    /// An error text when the settings cannot give a meaningful selection.
    pub fn check(&self) -> Result<(), String> {
        if !(self.lexicase_epsilon >= 0.0 && self.lexicase_epsilon.is_finite()) {
            return Err("lexicase_epsilon must be finite and not negative".to_string());
        }

        if self.tournament_size == 0 {
            return Err("tournament_size must be at least 1".to_string());
        }

        if self.tournament_size > MAX_TOURNAMENT_SIZE {
            return Err(format!(
                "tournament_size must be at most {MAX_TOURNAMENT_SIZE}"
            ));
        }

        Ok(())
    }
}

/// The largest tournament. Every pick draws this many members, so the bound keeps a pick
/// cheap; with 1024 draws a tournament already picks the best member almost always.
pub const MAX_TOURNAMENT_SIZE: usize = 1024;

fn clean(x: f64) -> f64 {
    if x.is_nan() { 0.0 } else { x }
}

/// One lexicase selection among `pool` (indices into `scores`, one row per candidate and
/// one column per task). The task order is a fresh shuffle from `rng`. None when `pool` is
/// empty.
pub fn lexicase_pick(
    scores: &[Vec<f64>],
    pool: &[usize],
    epsilon: f64,
    rng: &mut ChaCha8Rng,
) -> Option<usize> {
    if pool.is_empty() {
        return None;
    }

    let tasks = scores.first().map_or(0, Vec::len);
    let mut order: Vec<usize> = (0..tasks).collect();

    order.shuffle(rng);

    let mut alive: Vec<usize> = pool.to_vec();

    for t in order {
        if alive.len() == 1 {
            break;
        }

        let at = |i: usize| clean(scores[i].get(t).copied().unwrap_or(0.0));
        let best = alive
            .iter()
            .map(|&i| at(i))
            .fold(f64::NEG_INFINITY, f64::max);
        let cut = best - epsilon - TIE_SLACK;

        alive.retain(|&i| at(i) >= cut);
    }

    alive.choose(rng).copied()
}

/// One tournament: the best-ranked of `size` uniform draws with replacement. `rank_of[i]`
/// is member i's position in the ranking, 0 for the best. None when there is no member.
pub fn tournament_pick(rank_of: &[usize], size: usize, rng: &mut ChaCha8Rng) -> Option<usize> {
    if rank_of.is_empty() {
        return None;
    }

    (0..size.max(1))
        .map(|_| rng.random_range(0..rank_of.len()))
        .min_by_key(|&i| rank_of[i])
}

/// The exact chance that a tournament of `size` picks each member: with the member at
/// position r of n, ((n - r) / n)^size - ((n - r - 1) / n)^size. The shares are taken
/// before the power, so large tournaments do not overflow; far down the ranking a weight
/// can underflow to 0.
pub fn tournament_weights(rank_of: &[usize], size: usize) -> Vec<f64> {
    let n = rank_of.len() as f64;
    let k = size.max(1) as f64;

    rank_of
        .iter()
        .map(|&r| {
            let above = (n - r as f64) / n;
            let below = (n - r as f64 - 1.0) / n;

            above.powf(k) - below.powf(k)
        })
        .collect()
}

/// Estimated chance that lexicase picks each member: the share of `draws` selections that
/// choose it. Used only to log how concentrated the next parent draw is.
pub fn lexicase_weights(
    scores: &[Vec<f64>],
    epsilon: f64,
    draws: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<f64> {
    let pool: Vec<usize> = (0..scores.len()).collect();
    let mut counts = vec![0.0; scores.len()];

    for _ in 0..draws {
        if let Some(i) = lexicase_pick(scores, &pool, epsilon, rng) {
            counts[i] += 1.0;
        }
    }

    counts
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    fn rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }

    #[test]
    fn tournament_size_is_bounded_so_a_pick_stays_cheap() {
        let at = SelectionParams {
            tournament_size: MAX_TOURNAMENT_SIZE,
            ..SelectionParams::default()
        };
        let above = SelectionParams {
            tournament_size: MAX_TOURNAMENT_SIZE + 1,
            ..SelectionParams::default()
        };

        assert!(at.check().is_ok());
        assert!(above.check().unwrap_err().contains("at most"));
    }

    /// A is better on average, B is the only one that solves task 2.
    fn specialists() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 1.0, 0.0, 1.0], // generalist, mean 0.75
            vec![0.0, 0.0, 1.0, 0.0], // specialist, mean 0.25
            vec![0.0, 0.0, 0.0, 0.0], // dominated
        ]
    }

    #[test]
    fn lexicase_keeps_a_specialist_that_aggregate_selection_drops() {
        let scores = specialists();
        let pool: Vec<usize> = (0..3).collect();
        let mut r = rng(1);
        let mut wins = [0usize; 3];

        for _ in 0..2000 {
            wins[lexicase_pick(&scores, &pool, 0.0, &mut r).unwrap()] += 1;
        }

        // The specialist wins exactly when task 2 comes before every task the generalist
        // solves: probability 1/4.
        assert_eq!(wins[2], 0, "a dominated candidate is never picked");
        assert!((400..600).contains(&wins[1]), "specialist wins {}", wins[1]);
        assert!(wins[0] > wins[1]);
    }

    #[test]
    fn lexicase_is_reproducible_from_its_seed_and_varies_with_it() {
        let scores = specialists();
        let pool: Vec<usize> = (0..3).collect();
        let run = |seed| {
            let mut r = rng(seed);

            (0..50)
                .map(|_| lexicase_pick(&scores, &pool, 0.0, &mut r).unwrap())
                .collect::<Vec<_>>()
        };

        assert_eq!(run(7), run(7));
        assert_ne!(run(7), run(8));
    }

    #[test]
    fn epsilon_makes_near_ties_equal_and_zero_epsilon_does_not() {
        let scores = vec![vec![1.0], vec![0.95]];
        let pool = [0usize, 1];
        let count = |eps: f64| {
            let mut r = rng(3);

            (0..400)
                .filter(|_| lexicase_pick(&scores, &pool, eps, &mut r) == Some(1))
                .count()
        };

        assert_eq!(count(0.0), 0);
        assert!(
            count(0.06) > 100,
            "a score within epsilon stays in the running"
        );
    }

    #[test]
    fn float_noise_below_the_slack_is_a_tie() {
        let scores = vec![vec![0.1 + 0.2], vec![0.3]];
        let mut r = rng(5);
        let hits = (0..200)
            .filter(|_| lexicase_pick(&scores, &[0, 1], 0.0, &mut r) == Some(1))
            .count();

        assert!(hits > 50 && hits < 150, "both candidates tie: {hits}");
    }

    #[test]
    fn lexicase_handles_degenerate_inputs() {
        let mut r = rng(1);

        assert_eq!(lexicase_pick(&[], &[], 0.0, &mut r), None);
        // No task at all: every candidate ties, so a uniform draw.
        assert!(lexicase_pick(&[vec![], vec![]], &[0, 1], 0.0, &mut r).is_some());
        // NaN counts as 0.
        let scores = vec![vec![f64::NAN], vec![0.5]];

        assert_eq!(lexicase_pick(&scores, &[0, 1], 0.0, &mut r), Some(1));
        // The pool restricts the choice.
        assert_eq!(lexicase_pick(&specialists(), &[2], 0.0, &mut r), Some(2));
    }

    #[test]
    fn tournament_prefers_rank_and_matches_its_exact_weights() {
        // Member i has rank i.
        let ranks: Vec<usize> = (0..5).collect();
        let weights = tournament_weights(&ranks, 3);

        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(weights.windows(2).all(|w| w[0] > w[1]));

        let mut r = rng(11);
        let mut counts = [0usize; 5];
        let n = 20_000;

        for _ in 0..n {
            counts[tournament_pick(&ranks, 3, &mut r).unwrap()] += 1;
        }

        for (c, w) in counts.iter().zip(&weights) {
            assert!((*c as f64 / n as f64 - w).abs() < 0.02);
        }

        // Size 1 is a uniform draw.
        let uniform = tournament_weights(&ranks, 1);

        assert!(uniform.iter().all(|w| (w - 0.2).abs() < 1e-12));
        assert_eq!(tournament_pick(&[], 3, &mut r), None);
    }

    #[test]
    fn tournament_weights_stay_finite_for_large_tournaments() {
        // 100 members, tournaments of 200: 100^200 alone is beyond the range of f64.
        let ranks: Vec<usize> = (0..100).collect();
        let weights = tournament_weights(&ranks, 200);

        assert!(weights.iter().all(|w| w.is_finite()), "{weights:?}");
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        // The weights fall with rank; far down the ranking they reach 0 and stay there.
        assert!(
            weights
                .windows(2)
                .all(|w| w[0] > w[1] || (w[0] == 0.0 && w[1] == 0.0))
        );
        assert!(weights[0] > weights[99]);

        // A size beyond i32 is taken as it is: the best member wins every tournament.
        let huge = tournament_weights(&ranks[..5], usize::MAX);

        assert_eq!(huge, [1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn params_check_and_default() {
        assert!(SelectionParams::default().is_default());
        assert!(SelectionParams::default().check().is_ok());
        assert!(
            SelectionParams {
                lexicase_epsilon: -0.1,
                ..Default::default()
            }
            .check()
            .is_err()
        );
        assert!(
            SelectionParams {
                lexicase_epsilon: f64::NAN,
                ..Default::default()
            }
            .check()
            .is_err()
        );
        assert!(
            SelectionParams {
                tournament_size: 0,
                ..Default::default()
            }
            .check()
            .is_err()
        );
        assert!(serde_json::from_str::<SelectionParams>(r#"{"bogus":1}"#).is_err());
    }
}
