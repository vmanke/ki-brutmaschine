//! Experience, imagination and changing goals (arms I, G and IG,
//! `.scratch/seelen-modell/spec-imagination.md`, section 3). Both settings are off by
//! default; a Run without them writes every byte as before.
//!
//! - Experience (spec 3.1): with `imagination` on, every Worker carries a learned value per
//!   family of its Domain. The values live in the member, never in the genome, because they
//!   change after birth, so the genome and the Cascade id stay as they are. They are integers
//!   in thousandths, so no byte and no hash depends on how a float prints. A child Worker
//!   copies the values of its source Worker; a Worker without a source starts at 0.
//! - Imagination (spec 3.2): an operator that can change the families a Worker names is drawn
//!   several times on the same parent, and the draw with the highest prediction becomes the
//!   child. The prediction of a draw is the sum, over the Workers it changes, of the values of
//!   the families a Worker gains minus those it loses, each read in the values of the Worker's
//!   source. It costs no model call.
//! - Learning (spec 3.3): after a newborn is evaluated, its real effect s is the sign of its
//!   solved count on the current goal minus that of its first parent. One error e = s − p moves
//!   every family it gained by one step and every family it lost by minus that step: the
//!   delta rule, with its rate divided by the number of changed families (`step`). The child's
//!   Workers learn at once; their source Workers learn the same steps at the commit.
//! - Changing goals (spec 3.4): with `goals` on, the tasks that count for lexicase selection
//!   and for s change every `length` generations. Every second goal is all tasks; the others
//!   leave out the tasks whose targets name one family (`goal_cycle`).
//!
//! Pure computation: no files, no network, no clock.
#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::cascade::genome::Worker;
use crate::cascade::variation::Op;
use crate::util::{BrutError, Result};

/// The operators that imagine (spec 3.2): those that can change the families a Worker
/// names. StageAdd can add a family too, when the phrase it writes names one, but it does
/// not imagine; its newborns still learn.
pub const IMAGINING_OPS: [Op; 4] = [
    Op::SectionReplace,
    Op::SectionAdd,
    Op::SectionDrop,
    Op::SectionCrossover,
];

/// The largest learned value, 1 in thousandths; the smallest is its negative (spec 3.1).
pub const VALUE_MAX: i16 = 1000;

/// One in thousandths: the unit of values, predictions, errors and steps (spec 3.3).
pub const ONE: i64 = 1000;

/// The most draws a setting may ask for. A draw calls no model, but every draw is one more
/// operator call for each child that imagines.
pub const MAX_DRAWS: usize = 64;

/// The name of the goal of all tasks, G0 (spec 3.4).
pub const ALL_TASKS: &str = "all";

/// The settings of experience and imagination (arm I, spec 3.1 to 3.3). Both fields are
/// required and no other is allowed. They define the experiment, so a resume cannot change
/// them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImaginationConfig {
    /// How often an imagining operator is drawn on the same parent, the first time from the
    /// generation's stream (spec 3.2): 4 in arm I.
    pub draws: usize,
    /// The learning rate in thousandths (spec 3.3): 250 is the rate 1/4 of arm I.
    pub rate: u16,
}

impl ImaginationConfig {
    /// The setting of arms I and IG: 4 draws and the rate 1/4.
    pub fn arm_i() -> ImaginationConfig {
        ImaginationConfig {
            draws: 4,
            rate: 250,
        }
    }

    /// Reject settings that cannot give a meaningful Run.
    pub fn check(&self) -> Result<()> {
        if !(1..=MAX_DRAWS).contains(&self.draws) {
            return Err(BrutError::Invalid(format!(
                "breed config: imagination: draws must lie in 1..={MAX_DRAWS}, not {}",
                self.draws
            )));
        }

        if i64::from(self.rate) > ONE {
            return Err(BrutError::Invalid(format!(
                "breed config: imagination: rate is in thousandths and must lie in 0..={ONE}, not {}",
                self.rate
            )));
        }

        Ok(())
    }
}

/// The setting of changing goals (arm G, spec 3.4). Its one field is required and no other
/// is allowed. It defines the experiment, so a resume cannot change it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoalsConfig {
    /// The generations each goal lasts: E = 20 in arm G, as in Kashtan, Noor and Alon (2007).
    pub length: usize,
}

impl GoalsConfig {
    /// The setting of arms G and IG: every goal lasts 20 generations.
    pub fn arm_g() -> GoalsConfig {
        GoalsConfig { length: 20 }
    }

    /// Reject settings that cannot give a meaningful Run.
    pub fn check(&self) -> Result<()> {
        if self.length == 0 {
            return Err(BrutError::Invalid(
                "breed config: goals: length must be at least 1 generation".to_string(),
            ));
        }

        Ok(())
    }
}

/// The families that `worker` names: those that the gene pool's rule `families_of` finds in
/// any of its section texts, each once, in index order. An index at or above `families`
/// names no family of the Domain and is left out. A section's name is not read; no section
/// name of a gene pool holds a family's stem, so this is what the simulated designer reads in
/// the Worker's system prompt.
pub fn named(
    worker: &Worker,
    families_of: fn(&str) -> Vec<usize>,
    families: usize,
) -> BTreeSet<usize> {
    worker
        .sections
        .iter()
        .flat_map(|s| families_of(&s.text))
        .filter(|&f| f < families)
        .collect()
}

/// What one child Worker gained and lost against its source Worker (spec 3.2, 3.3): the
/// families it names that its source does not, and those its source names that it does not,
/// each in index order. A Worker without a source has neither.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Change {
    pub gained: Vec<usize>,
    pub lost: Vec<usize>,
}

impl Change {
    /// The change of a Worker that names `child` against a source that names `source`.
    pub fn between(child: &BTreeSet<usize>, source: &BTreeSet<usize>) -> Change {
        Change {
            gained: child.difference(source).copied().collect(),
            lost: source.difference(child).copied().collect(),
        }
    }

    /// The families gained and lost: this Worker's part of n (spec 3.3).
    pub fn families(&self) -> usize {
        self.gained.len() + self.lost.len()
    }

    /// This Worker's part of the prediction, in thousandths (spec 3.2): the values of the
    /// gained families minus those of the lost ones, read in `source`, the values of its
    /// source Worker. A family without a value counts 0.
    pub fn prediction(&self, source: &[i16]) -> i64 {
        let at = |f: &usize| source.get(*f).map_or(0, |&v| i64::from(v));

        self.gained.iter().map(at).sum::<i64>() - self.lost.iter().map(at).sum::<i64>()
    }

    /// Learn one step (spec 3.3): every gained family of `values` moves by +`step` and every
    /// lost family by −`step`, each clamped to [−1000, 1000] on its own. A family without a
    /// value is left out.
    pub fn learn(&self, values: &mut [i16], step: i64) {
        for (families, by) in [(&self.gained, step), (&self.lost, -step)] {
            for &f in families {
                if let Some(v) = values.get_mut(f) {
                    *v = moved(*v, by);
                }
            }
        }
    }
}

/// `v` moved by `by` thousandths and clamped to [−1000, 1000].
fn moved(v: i16, by: i64) -> i16 {
    let to = (i64::from(v) + by).clamp(-i64::from(VALUE_MAX), i64::from(VALUE_MAX));

    // Inside the clamp, so the conversion never fails.
    i16::try_from(to).unwrap_or(v)
}

/// The prediction p of a draw or a newborn, in thousandths (spec 3.2): the sum over its
/// Workers of each one's part, read in the values of its source. `parts` pairs the change of
/// each Worker with its source's values.
pub fn prediction<'v>(parts: impl IntoIterator<Item = (&'v Change, &'v [i16])>) -> i64 {
    parts
        .into_iter()
        .map(|(c, source)| c.prediction(source))
        .sum()
}

/// The candidate that imagination chooses among the draws whose predictions are
/// `predictions`, in draw order (spec 3.2): the one with the highest prediction, and among
/// several, the earliest. So while every prediction is 0 it is the first. None without a
/// candidate.
pub fn choose(predictions: &[i64]) -> Option<usize> {
    let mut best: Option<usize> = None;

    for (k, &p) in predictions.iter().enumerate() {
        if best.is_none_or(|b| p > predictions[b]) {
            best = Some(k);
        }
    }

    best
}

/// The real effect s (spec 3.3): the sign of `child` − `parent`, two solved counts on the same
/// tasks, as −1, 0 or 1.
pub fn effect(child: usize, parent: usize) -> i8 {
    match child.cmp(&parent) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// The error e = s − p in thousandths (spec 3.3), with s in units and p in thousandths.
pub fn error(s: i8, p: i64) -> i64 {
    i64::from(s) * ONE - p
}

/// The step of a newborn with the error `e` and `n` families gained and lost over all its
/// Workers, at the rate `rate`, all in thousandths (spec 3.3): every gained family moves by
/// the step and every lost family by minus the step. It is e · rate / (1000 · n), rounded
/// half away from zero, computed exactly in integers as
///
/// (e · rate + sign(e) · 500 · n) / (1000 · n), with division towards zero.
///
/// With the rate 250 of arm I this is the spec's (e + sign(e) · 2n) / (4n): both sides are the
/// same fraction, the second one scaled by 250, so division towards zero gives the same
/// integer. So the delta rule's step e/4 is divided by n: a change of many families takes no
/// larger step than a change of one. With n = 0 there is no step: the newborn learns nothing.
pub fn step(e: i64, n: usize, rate: u16) -> i64 {
    let Ok(n) = i64::try_from(n) else {
        return 0;
    };

    if n == 0 {
        return 0;
    }

    let numerator = e * i64::from(rate);
    let denominator = ONE * n;

    (numerator + numerator.signum() * (denominator / 2)) / denominator
}

/// One goal (spec 3.4): its name, and the tasks that count for lexicase selection and for
/// the real effect s, by their places on the search panel, in panel order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Goal {
    /// "all" for G0; "-" and the family's name for G−f, such as "-Linear".
    pub name: String,
    pub tasks: Vec<usize>,
}

/// One cycle of goals over a search panel whose tasks have the Family strengths `strengths`,
/// each in the order of `families` (spec 3.4). G0 holds every task. G−f holds the tasks whose
/// strength for family f is not positive: the tasks of the targets whose `fits` do not name
/// f. A G−f that is empty or equals G0 is left out. The cycle is G0, G−f1, G0, G−f2, and so on
/// over the families that remain, in index order, so half of the goals are G0. Without such a
/// family it is G0 alone.
pub fn goal_cycle(families: &[&str], strengths: &[Vec<f64>]) -> Vec<Goal> {
    let all = Goal {
        name: ALL_TASKS.to_string(),
        tasks: (0..strengths.len()).collect(),
    };
    let mut cycle = vec![all.clone()];

    for (f, family) in families.iter().enumerate() {
        let tasks: Vec<usize> = strengths
            .iter()
            .enumerate()
            .filter(|(_, s)| s.get(f).is_none_or(|&x| x <= 0.0 || x.is_nan()))
            .map(|(t, _)| t)
            .collect();

        if tasks.is_empty() || tasks.len() == all.tasks.len() {
            continue;
        }

        if cycle.len() > 1 {
            cycle.push(all.clone());
        }

        cycle.push(Goal {
            name: format!("-{family}"),
            tasks,
        });
    }

    cycle
}

/// The place in a cycle of `cycle` goals of the goal of generation `g`, when each goal lasts
/// `length` generations (spec 3.4): generations 1 to `length` have the first goal, the next
/// `length` the second, and after the last goal the cycle starts again. Generation 0, whose
/// seeds no selection draws, has the first goal too.
pub fn goal_index(g: usize, length: usize, cycle: usize) -> usize {
    if cycle == 0 {
        return 0;
    }

    (g.saturating_sub(1) / length.max(1)) % cycle
}

/// What a newborn's learning read (spec 3.3, 3.5). Its Born event holds it when its Workers
/// gained or lost families (n > 0), whichever operator made it, and whenever imagination chose
/// it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Effect {
    /// The prediction p, in thousandths: the sum over its Workers of the values of the
    /// families each gained minus those it lost, read in the values of its source Worker when
    /// the child was drafted.
    pub p: i64,
    /// The real effect s: the sign of its solved count on the goal of its generation minus
    /// that of its first parent, −1, 0 or 1 (in thousandths −1000, 0 or 1000).
    pub s: i8,
    /// For each of its Workers, in Stage order, the families it gained and lost against its
    /// source Worker, in index order.
    pub gained: Vec<Vec<usize>>,
    pub lost: Vec<Vec<usize>>,
    /// The chosen draw, from 1 (today's draw) to the setting's `draws`, when imagination chose
    /// the newborn; left out otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draw: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cascade::circuit::CircuitDomain;
    use crate::cascade::designer::{FAMILY_STEMS, named_families};
    use crate::cascade::domain::{CircuitConfig, Domain};
    use crate::cascade::genome::{Section, Traits};

    fn set(families: &[usize]) -> BTreeSet<usize> {
        families.iter().copied().collect()
    }

    fn worker(texts: &[&str]) -> Worker {
        Worker {
            model: "sim:small".to_string(),
            sections: texts
                .iter()
                .enumerate()
                .map(|(i, t)| Section {
                    name: format!("part{i}"),
                    text: t.to_string(),
                })
                .collect(),
            traits: Traits::default(),
        }
    }

    #[test]
    fn a_worker_names_the_families_of_its_section_texts() {
        let w = worker(&[
            "Treat each CNOT as a row operation.",
            "Search from both ends at once and meet in the middle.",
            "Answer with the line ANSWER.",
        ]);

        // Linear (0), Search (4) and Meet (5); an index beyond the Domain's families is left
        // out.
        assert_eq!(named(&w, named_families, 8), set(&[0, 4, 5]));
        assert_eq!(named(&w, named_families, 5), set(&[0, 4]));
        assert_eq!(
            named(&worker(&["Be careful."]), named_families, 8),
            set(&[])
        );
        assert_eq!(FAMILY_STEMS.len(), 8);
    }

    #[test]
    fn a_change_is_what_a_worker_gained_and_lost_against_its_source() {
        let c = Change::between(&set(&[0, 2, 5]), &set(&[1, 2]));

        assert_eq!(c.gained, vec![0, 5]);
        assert_eq!(c.lost, vec![1]);
        assert_eq!(c.families(), 3);
        assert_eq!(Change::between(&set(&[3]), &set(&[3])), Change::default());
    }

    #[test]
    fn the_prediction_sums_gains_minus_losses_over_the_changed_workers() {
        let source = [100, -200, 300, 0, 0, 50, 0, 0];
        let c = Change {
            gained: vec![0, 5],
            lost: vec![1],
        };

        // 100 + 50 − (−200).
        assert_eq!(c.prediction(&source), 350);

        // A draw that changes two Workers: one p over both, each read in its own source.
        let other_source = [0, 0, -400, 0, 700, 0, 0, 0];
        let d = Change {
            gained: vec![4],
            lost: vec![2],
        };

        assert_eq!(
            prediction([(&c, &source[..]), (&d, &other_source[..])]),
            350 + 700 + 400
        );

        // Nothing changed, or all values 0: no prediction.
        assert_eq!(Change::default().prediction(&source), 0);
        assert_eq!(c.prediction(&[0; 8]), 0);
        assert_eq!(prediction(std::iter::empty()), 0);
    }

    #[test]
    fn the_highest_prediction_wins_and_a_tie_goes_to_the_earliest_draw() {
        assert_eq!(choose(&[]), None);
        assert_eq!(choose(&[0]), Some(0));
        assert_eq!(choose(&[0, 0, 0, 0]), Some(0));
        assert_eq!(choose(&[-5, 3, 3, 1]), Some(1));
        assert_eq!(choose(&[-5, -3, -3, -4]), Some(1));
        assert_eq!(choose(&[2, 2, 7, 7]), Some(2));
        assert_eq!(choose(&[0, -1, -2, -3]), Some(0));
        assert_eq!(choose(&[-1, 0, 0, 0]), Some(1));
    }

    #[test]
    fn the_effect_is_a_sign_and_the_error_is_in_thousandths() {
        assert_eq!(effect(5, 3), 1);
        assert_eq!(effect(3, 3), 0);
        assert_eq!(effect(0, 3), -1);
        assert_eq!(error(1, 350), 650);
        assert_eq!(error(0, 350), -350);
        assert_eq!(error(-1, -250), -750);
    }

    /// The spec's rounding for the rate 1/4: (e + sign(e)·2n) / (4n), towards zero.
    fn spec_step(e: i64, n: i64) -> i64 {
        (e + e.signum() * 2 * n) / (4 * n)
    }

    #[test]
    fn the_step_is_e_over_4n_rounded_half_away_from_zero() {
        // n = 1: e/4.
        assert_eq!(step(1000, 1, 250), 250);
        assert_eq!(step(-1000, 1, 250), -250);
        assert_eq!(step(0, 1, 250), 0);
        assert_eq!(step(1, 1, 250), 0);
        assert_eq!(step(-1, 1, 250), 0);
        assert_eq!(step(3, 1, 250), 1);
        assert_eq!(step(-3, 1, 250), -1);

        // Steps that end in exactly one half go away from zero, on both sides.
        assert_eq!(step(2, 1, 250), 1);
        assert_eq!(step(-2, 1, 250), -1);
        assert_eq!(step(10, 1, 250), 3);
        assert_eq!(step(-10, 1, 250), -3);
        assert_eq!(step(4, 2, 250), 1);
        assert_eq!(step(-4, 2, 250), -1);
        assert_eq!(step(12, 2, 250), 2);
        assert_eq!(step(-12, 2, 250), -2);
        assert_eq!(step(-1250, 1, 250), -313);

        // n above 1 divides the step: three families take a third of one family's step.
        assert_eq!(step(1200, 3, 250), 100);
        assert_eq!(step(-1200, 3, 250), -100);
        assert_eq!(step(1000, 3, 250), 83);
        assert_eq!(step(-1000, 3, 250), -83);
        assert_eq!(step(1000, 8, 250), 31);
        assert_eq!(step(-1000, 8, 250), -31);

        // No changed family, no step.
        assert_eq!(step(1000, 0, 250), 0);

        // The general formula is the spec's for every e and n tried, positive and negative.
        for n in 1..=24 {
            for e in -5000..=5000 {
                assert_eq!(step(e, n as usize, 250), spec_step(e, n), "e {e}, n {n}");
            }
        }

        // Other rates round the same way: e · rate / (1000 n), half away from zero.
        assert_eq!(step(1000, 1, 1000), 1000);
        assert_eq!(step(1, 2, 1000), 1);
        assert_eq!(step(-1, 2, 1000), -1);
        assert_eq!(step(1000, 1, 0), 0);
    }

    #[test]
    fn learning_moves_gained_families_up_and_lost_ones_down_and_clamps_each() {
        let c = Change {
            gained: vec![0, 2],
            lost: vec![1],
        };
        let mut v = [100, 100, 990, 7, 0, 0, 0, 0];

        c.learn(&mut v, 250);

        // Family 2 stops at 1000; family 3 is neither gained nor lost.
        assert_eq!(v, [350, -150, 1000, 7, 0, 0, 0, 0]);

        // A negative step moves the other way, and −1000 is the floor.
        let mut w = [-900, 900, 0, 0, 0, 0, 0, 0];

        c.learn(&mut w, -250);
        assert_eq!(w, [-1000, 1000, -250, 0, 0, 0, 0, 0]);

        // Each step is clamped on its own, so at a bound the order of the steps matters: from
        // 900, two steps of +250 and then one of −250 end at 750, but −250 first and then two
        // of +250 end at 1000.
        let up = Change {
            gained: vec![0],
            lost: Vec::new(),
        };
        let mut x = [900, 0, 0, 0, 0, 0, 0, 0];

        up.learn(&mut x, 250);
        up.learn(&mut x, 250);
        up.learn(&mut x, -250);
        assert_eq!(x[0], 750);

        let mut y = [900, 0, 0, 0, 0, 0, 0, 0];

        up.learn(&mut y, -250);
        up.learn(&mut y, 250);
        up.learn(&mut y, 250);
        assert_eq!(y[0], 1000);
    }

    #[test]
    fn the_settings_are_checked() {
        assert!(ImaginationConfig::arm_i().check().is_ok());
        assert!(GoalsConfig::arm_g().check().is_ok());

        for bad in [
            ImaginationConfig {
                draws: 0,
                rate: 250,
            },
            ImaginationConfig {
                draws: MAX_DRAWS + 1,
                rate: 250,
            },
            ImaginationConfig {
                draws: 4,
                rate: 1001,
            },
        ] {
            assert!(bad.check().is_err(), "{bad:?}");
        }

        assert!(GoalsConfig { length: 0 }.check().is_err());

        // An unknown or a missing field is refused.
        for bad in [
            r#"{"draws":4,"rate":250,"temperature":1}"#,
            r#"{"draws":4}"#,
        ] {
            assert!(
                serde_json::from_str::<ImaginationConfig>(bad).is_err(),
                "{bad}"
            );
        }

        assert!(serde_json::from_str::<GoalsConfig>(r#"{"length":20,"x":1}"#).is_err());
    }

    /// The names of the targets of `tasks` on `panel`, each once.
    fn targets<T: AsRef<str>>(names: &[T], tasks: &[usize]) -> BTreeSet<String> {
        tasks
            .iter()
            .map(|&t| names[t].as_ref().to_string())
            .collect()
    }

    #[test]
    fn the_goals_of_todays_circuit_library() {
        let domain = CircuitDomain::new(&CircuitConfig::default()).expect("circuits");
        let panel = domain.panel(4242, 33);
        let names: Vec<&str> = panel.iter().map(|t| t.name).collect();
        let strengths: Vec<Vec<f64>> = panel.iter().map(|t| domain.strengths(t)).collect();
        let cycle = goal_cycle(&domain.families(), &strengths);
        let all: BTreeSet<String> = names.iter().map(|n| n.to_string()).collect();
        let but = |left: &[&str]| -> BTreeSet<String> {
            all.iter()
                .filter(|n| !left.contains(&n.as_str()))
                .cloned()
                .collect()
        };
        let only =
            |kept: &[&str]| -> BTreeSet<String> { kept.iter().map(|n| n.to_string()).collect() };

        assert_eq!(all.len(), 11);
        assert_eq!(
            cycle.iter().map(|g| g.name.as_str()).collect::<Vec<_>>(),
            [
                "all",
                "-Linear",
                "all",
                "-Phase",
                "all",
                "-Hadamard",
                "all",
                "-Compose",
                "all",
                "-Search",
                "all",
                "-Meet",
                "all",
                "-Simplify"
            ]
        );

        let expected = [
            all.clone(),
            only(&["cs", "ch"]),
            all.clone(),
            only(&["cz", "swap", "ch"]),
            all.clone(),
            only(&["swap", "cs", "ch", "ccz"]),
            all.clone(),
            but(&["fredkin"]),
            all.clone(),
            but(&["cz", "ch"]),
            all.clone(),
            but(&["ch"]),
            all.clone(),
            but(&["fredkin"]),
        ];

        for (goal, want) in cycle.iter().zip(&expected) {
            assert_eq!(&targets(&names, &goal.tasks), want, "{}", goal.name);

            // A goal holds every task of its targets, in panel order.
            assert!(goal.tasks.windows(2).all(|w| w[0] < w[1]), "{}", goal.name);
            assert_eq!(
                goal.tasks.len(),
                names.iter().filter(|n| want.contains(**n)).count(),
                "{}",
                goal.name
            );
        }

        assert_eq!(cycle[0].tasks, (0..33).collect::<Vec<_>>());

        // A full cycle is 14 goals of 20 generations: 280 generations.
        assert_eq!(goal_index(0, 20, 14), 0);
        assert_eq!(goal_index(1, 20, 14), 0);
        assert_eq!(goal_index(20, 20, 14), 0);
        assert_eq!(goal_index(21, 20, 14), 1);
        assert_eq!(goal_index(40, 20, 14), 1);
        assert_eq!(goal_index(41, 20, 14), 2);
        assert_eq!(goal_index(280, 20, 14), 13);
        assert_eq!(goal_index(281, 20, 14), 0);
    }

    #[test]
    fn goals_that_equal_all_tasks_or_none_are_left_out() {
        // Three tasks: family 0 is named by every task (G−0 is empty), family 1 by none (G−1
        // is all tasks), family 2 by the first task only.
        let strengths = vec![
            vec![1.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
        ];
        let cycle = goal_cycle(&["A", "B", "C"], &strengths);

        assert_eq!(
            cycle,
            vec![
                Goal {
                    name: "all".to_string(),
                    tasks: vec![0, 1, 2],
                },
                Goal {
                    name: "-C".to_string(),
                    tasks: vec![1, 2],
                },
            ]
        );

        // Without a variant the cycle is G0 alone.
        let alone = goal_cycle(&["A"], &strengths);

        assert_eq!(alone.len(), 1);
        assert_eq!(goal_index(57, 20, alone.len()), 0);
    }
}
