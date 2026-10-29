//! The seam between the breeder and a task family (step 1, ADR 0006).
//!
//! A `Domain` is the evaluator side: it makes the panels, renders and parses prompts, and
//! grades answers. It never sees a Cascade, so no Cascade can change how it is judged. A
//! `GenePool` is the breeder side: the seeds, the phrase pool, the Mutator's brief and an
//! admission rule for new Cascades. Grid puzzles (Stage 2c) are one domain; quantum
//! circuits (step 1) are the next.
//!
//! Pure computation: no files, no network, no clock.
#![forbid(unsafe_code)]

use std::fmt::Debug;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::cascade::arena::BreedConfig;
use crate::cascade::genome::{Cascade, seed_cascades};
use crate::cascade::puzzle::{
    Grid, Puzzle, SELF_CHECK_TEXT, generate_panel, parse_answer, render_judge_prompt,
    render_solve_prompt,
};
use crate::cascade::variation::{GRID_BRIEF, MutatorBrief, PHRASES};
use crate::util::{BrutError, Result};

/// How one answer did on one task.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grade {
    /// Grid: equal to the truth. Circuit: equal to the target exactly, up to a global
    /// phase.
    pub correct: bool,
    /// Grid: the same as `correct`. Circuit: correct and no costlier than the reference.
    pub solved: bool,
    /// In [0, 1]; 0 unless correct. Grid: 1 when correct.
    pub value: f64,
}

/// A task family on the evaluator side. Everything a Cascade meets goes through here.
pub trait Domain: Sized + Sync {
    type Task: Clone + Debug + Send + Sync;
    type Answer: Clone + PartialEq + Debug + Send + Sync + Serialize + DeserializeOwned;

    /// This domain for `cfg`; an error when `cfg.domain` names another one.
    fn from_config(cfg: &BreedConfig) -> Result<Self>;

    /// `n` tasks from `seed`. A longer panel extends a shorter one with the same seed.
    fn panel(&self, seed: u64, n: usize) -> Vec<Self::Task>;

    /// A task's id. It seeds model calls and names the task in the log; it is never
    /// rendered into a prompt.
    fn task_id<'t>(&self, task: &'t Self::Task) -> &'t str;

    /// The prompt that asks for an answer, with the answers of earlier stages.
    fn solve_prompt(&self, task: &Self::Task, previous: &[Self::Answer]) -> String;

    /// The prompt that asks which of two answers is better.
    fn judge_prompt(&self, task: &Self::Task, a: &Self::Answer, b: &Self::Answer) -> String;

    /// The answer in a model's reply, if it holds a well-formed one.
    fn parse_answer(&self, task: &Self::Task, text: &str) -> Option<Self::Answer>;

    /// Grade an answer. An error only when the answer beats a proven bound, which means
    /// the checker or the reference is wrong (an integrity alarm).
    fn grade(&self, task: &Self::Task, answer: &Self::Answer) -> Result<Grade>;

    /// Some(true) when `a` is strictly better than `b`, Some(false) when `b` is, None when
    /// neither is: the pairs that score a judge.
    fn prefer(&self, task: &Self::Task, a: &Self::Answer, b: &Self::Answer) -> Option<bool>;

    /// False when fitness uses the solved count only (grids); true when it uses the
    /// graded value.
    fn graded(&self) -> bool;

    /// The answer as text for the log, or None (grids log no answers).
    fn artefact(&self, answer: &Self::Answer) -> Option<String>;

    /// The fixed text a Worker's self-check Trait adds to its system prompt.
    fn self_check_text(&self) -> &'static str;
}

/// The breeder side of a task family: what new Cascades are made from.
#[derive(Clone, Copy, Debug)]
pub struct GenePool {
    /// The hand-written starting Cascades.
    pub seeds: fn() -> Vec<(String, Cascade)>,
    /// The phrase pool the section operators draw from, as (section name, text).
    pub phrases: &'static [(&'static str, &'static str)],
    /// The words that tell the Mutator what it breeds.
    pub brief: &'static MutatorBrief,
    /// A rule every new Cascade must also pass (Ok for grids).
    pub admit: fn(&Cascade) -> std::result::Result<(), String>,
}

impl GenePool {
    /// The grid puzzles of Stage 2c: today's seeds, phrases and brief.
    pub fn grid() -> GenePool {
        GenePool {
            seeds: seed_cascades,
            phrases: PHRASES,
            brief: &GRID_BRIEF,
            admit: |_| Ok(()),
        }
    }
}

/// Which task family a run breeds on. Grid runs leave it out of their settings, so their
/// recorded configs and hashes stay as they were.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DomainConfig {
    #[default]
    Grid,
    Circuits(CircuitConfig),
}

impl DomainConfig {
    pub fn is_grid(&self) -> bool {
        matches!(self, DomainConfig::Grid)
    }
}

/// Settings of the circuit domain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CircuitConfig {
    /// Library targets by name; empty means every verified target.
    #[serde(default)]
    pub targets: Vec<String>,
    /// Draw a qubit relabelling and X flips per task.
    #[serde(default = "dress_default")]
    pub dress: bool,
    /// Gate cap of an answer; a config may lower the code's cap, never raise it.
    #[serde(default = "max_gates_default")]
    pub max_gates: usize,
}

fn dress_default() -> bool {
    true
}

fn max_gates_default() -> usize {
    64
}

impl Default for CircuitConfig {
    fn default() -> Self {
        CircuitConfig {
            targets: Vec::new(),
            dress: dress_default(),
            max_gates: max_gates_default(),
        }
    }
}

/// The grid puzzles of Stage 2c as a domain: it forwards to `puzzle.rs`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GridDomain;

impl Domain for GridDomain {
    type Task = Puzzle;
    type Answer = Grid;

    fn from_config(cfg: &BreedConfig) -> Result<Self> {
        if cfg.domain.is_grid() {
            Ok(GridDomain)
        } else {
            Err(BrutError::Invalid(
                "breed config: this run breeds on grid puzzles, but the config names another domain"
                    .to_string(),
            ))
        }
    }

    fn panel(&self, seed: u64, n: usize) -> Vec<Puzzle> {
        generate_panel(seed, n)
    }

    fn task_id<'t>(&self, task: &'t Puzzle) -> &'t str {
        &task.id
    }

    fn solve_prompt(&self, task: &Puzzle, previous: &[Grid]) -> String {
        render_solve_prompt(task, previous)
    }

    fn judge_prompt(&self, task: &Puzzle, a: &Grid, b: &Grid) -> String {
        render_judge_prompt(task, a, b)
    }

    fn parse_answer(&self, _task: &Puzzle, text: &str) -> Option<Grid> {
        parse_answer(text)
    }

    fn grade(&self, task: &Puzzle, answer: &Grid) -> Result<Grade> {
        let right = answer == &task.test_output;

        Ok(Grade {
            correct: right,
            solved: right,
            value: if right { 1.0 } else { 0.0 },
        })
    }

    fn prefer(&self, task: &Puzzle, a: &Grid, b: &Grid) -> Option<bool> {
        let (a_right, b_right) = (a == &task.test_output, b == &task.test_output);

        (a_right != b_right).then_some(a_right)
    }

    fn graded(&self) -> bool {
        false
    }

    fn artefact(&self, _answer: &Grid) -> Option<String> {
        None
    }

    fn self_check_text(&self) -> &'static str {
        SELF_CHECK_TEXT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_domain_forwards_to_stage_2c() {
        let d = GridDomain;
        let panel = d.panel(9, 6);

        assert_eq!(panel, generate_panel(9, 6));
        assert_eq!(
            d.panel(9, 4)[..],
            panel[..4],
            "a longer panel extends a shorter one"
        );

        let p = &panel[0];
        let right = p.test_output.clone();
        let mut wrong = right.clone();

        wrong[0][0] = (wrong[0][0] + 1) % 10;

        assert_eq!(d.task_id(p), p.id);
        assert_eq!(d.solve_prompt(p, &[]), render_solve_prompt(p, &[]));
        assert_eq!(
            d.judge_prompt(p, &right, &wrong),
            render_judge_prompt(p, &right, &wrong)
        );
        assert_eq!(
            d.parse_answer(p, "ANSWER\n1 2\n3 4"),
            parse_answer("ANSWER\n1 2\n3 4")
        );
        assert_eq!(d.prefer(p, &right, &wrong), Some(true));
        assert_eq!(d.prefer(p, &wrong, &right), Some(false));
        assert_eq!(d.prefer(p, &wrong, &wrong), None);
        assert_eq!(d.prefer(p, &right, &right), None);
        assert!(d.grade(p, &right).expect("grade").solved);
        assert_eq!(d.grade(p, &wrong).expect("grade").value, 0.0);
        assert!(!d.graded());
        assert_eq!(d.artefact(&right), None);
        assert_eq!(d.self_check_text(), SELF_CHECK_TEXT);
    }

    #[test]
    fn from_config_refuses_the_wrong_domain() {
        let mut cfg = BreedConfig::default();

        assert!(GridDomain::from_config(&cfg).is_ok());

        cfg.domain = DomainConfig::Circuits(CircuitConfig::default());
        assert!(GridDomain::from_config(&cfg).is_err());
    }

    #[test]
    fn grid_pool_is_the_old_constants() {
        let pool = GenePool::grid();

        assert_eq!((pool.seeds)(), seed_cascades());
        assert_eq!(pool.phrases, PHRASES);
        assert!(std::ptr::eq(pool.brief, &GRID_BRIEF));

        for (_, c) in seed_cascades() {
            assert_eq!((pool.admit)(&c), Ok(()));
        }
    }

    #[test]
    fn domain_config_keeps_grid_settings_unchanged() {
        // A grid config serialises without the field, so old configs and hashes hold.
        let text = serde_json::to_string(&BreedConfig::default()).expect("json");

        assert!(!text.contains("domain"), "{text}");

        let circuits = DomainConfig::Circuits(CircuitConfig::default());
        let json = serde_json::to_string(&circuits).expect("json");

        assert_eq!(
            json,
            r#"{"circuits":{"targets":[],"dress":true,"max_gates":64}}"#
        );
        assert_eq!(
            serde_json::from_str::<DomainConfig>(&json).expect("back"),
            circuits
        );
        assert_eq!(
            serde_json::from_str::<CircuitConfig>("{}").expect("defaults"),
            CircuitConfig::default()
        );
        assert!(serde_json::from_str::<CircuitConfig>(r#"{"gates":3}"#).is_err());
    }
}
