//! Upbringing at birth (step 2, `.scratch/seele-und-kappa/spec.md`, section 10) [Q73 a, d].
//!
//! Upbringing is the third channel of inheritance, beside genes and Souletics (Q55, Q59).
//! It passes on what the parents' lives gave them, never as a gene:
//!
//! - Imprinting: a child answers the first task of the search panel with its birth Souls.
//!   Then each of its Workers moves toward the current Souls of its source Workers, one
//!   source after another in parent order, by the observed side of the rule of an
//!   Observation (`soul::imprint`). The parents read no answer and do not change, and no
//!   model is called. The child answers its other search tasks with the imprinted Souls,
//!   and they start its current Souls at the commit.
//! - Teaching: with `teach` on, fixed code writes one prompt section, `upbringing`, from the
//!   parents' best solved examples (`teaching`). It ends every system prompt the child
//!   sends. It lives in the member and in its `born` event, never in the genome, so the
//!   child's own children get a section written from the child's examples.
//!
//! The teaching text is data that fixed code writes: the domain renders each example from a
//! task and an answer, and the template around them holds no word that a simulated model
//! reads (its family stems and keywords, and both self-check texts). So in simulated Runs
//! teaching changes no answer, only the length of the prompts.
//!
//! Pure computation: no files, no network, no clock.
#![forbid(unsafe_code)]

use crate::cascade::domain::Domain;
use crate::cascade::genome::check_section_text;
use crate::cascade::soul::imprint;
use crate::util::Result;

/// The name of the teaching section (spec 10.2).
pub const UPBRINGING: &str = "upbringing";

/// The first line of every teaching section.
pub const TEACHING_INTRO: &str = "Solved examples from your parents.";

/// The imprinted Soul of one child Worker (spec 10.1) [Q73 a]: its birth Soul moved toward
/// the current Soul of each of its source Workers in turn, by that source's Coupling
/// strength. `sources` holds the current Soul and κ of each source Worker, in parent order.
/// With `on` false, or without a source, the birth Soul stays.
pub fn imprinted(birth: &[u16], sources: &[(&[u16], u16)], on: bool) -> Vec<u16> {
    let mut soul = birth.to_vec();

    if on {
        for &(now, kappa) in sources {
            soul = imprint(&soul, now, kappa);
        }
    }

    soul
}

/// The text of the teaching section of a child whose parents gave the answers `parents` on
/// the search panel `panel`, one list per parent in parent order (spec 10.2) [Q73 d].
///
/// A parent's example is the first task with the highest graded value among those it
/// solved, rendered by the domain with the parent's answer. The text starts with
/// `TEACHING_INTRO` and takes the examples in parent order, each numbered, as long as the
/// text keeps the rules of a genome's section text, `MAX_SECTION_CHARS` included. An example
/// that would break them is left out, and so is one equal to an example already taken. None
/// when no example is taken. An error is the domain's integrity alarm from grading.
pub fn teaching<D: Domain>(
    domain: &D,
    panel: &[D::Task],
    parents: &[&[Option<D::Answer>]],
) -> Result<Option<String>> {
    let mut text = TEACHING_INTRO.to_string();
    let mut taken: Vec<String> = Vec::new();

    for answers in parents {
        let Some((task, answer)) = best_solved(domain, panel, answers)? else {
            continue;
        };
        let Some(example) = domain.example(task, answer) else {
            continue;
        };

        if taken.contains(&example) {
            continue;
        }

        let longer = format!("{text}\nEXAMPLE {}\n{example}", taken.len() + 1);

        if check_section_text(&longer).is_ok() {
            text = longer;
            taken.push(example);
        }
    }

    Ok((!taken.is_empty()).then_some(text))
}

/// The first task of `panel`, with its answer, that has the highest graded value among the
/// tasks that `answers` solve; None when they solve none.
fn best_solved<'a, D: Domain>(
    domain: &D,
    panel: &'a [D::Task],
    answers: &'a [Option<D::Answer>],
) -> Result<Option<(&'a D::Task, &'a D::Answer)>> {
    let mut best: Option<(f64, &D::Task, &D::Answer)> = None;

    for (task, answer) in panel.iter().zip(answers) {
        let Some(a) = answer else {
            continue;
        };
        let grade = domain.grade(task, a)?;

        if grade.solved && best.is_none_or(|(value, _, _)| grade.value > value) {
            best = Some((grade.value, task, a));
        }
    }

    Ok(best.map(|(_, task, a)| (task, a)))
}

/// The teaching section as it ends a system prompt: its name line, its text and a blank
/// line, as `Worker::system_prompt` writes a section.
pub fn section_block(text: &str) -> String {
    format!("## {UPBRINGING}\n{text}\n\n")
}

/// Every word a simulated model reads in a system prompt: the family stems of the circuit
/// designer, the keywords of the grid model, and the self-check texts of both (ticket 07).
/// Text that fixed code writes into a prompt holds none of them.
#[cfg(test)]
pub(crate) fn simulated_model_words() -> Vec<&'static str> {
    use crate::cascade::circuit_text::CIRCUIT_SELF_CHECK_TEXT;
    use crate::cascade::designer::FAMILY_STEMS;
    use crate::cascade::puzzle::SELF_CHECK_TEXT;
    use crate::cascade::sim::family_keywords;

    FAMILY_STEMS
        .iter()
        .flat_map(|&(_, stems)| stems.iter().copied())
        .chain(family_keywords())
        .chain([SELF_CHECK_TEXT, CIRCUIT_SELF_CHECK_TEXT])
        .collect()
}

/// The first word of `simulated_model_words` that `text` holds, in any case.
#[cfg(test)]
pub(crate) fn model_word_in(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();

    simulated_model_words()
        .into_iter()
        .find(|w| text.contains(w) || lower.contains(&w.to_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cascade::arena::BreedConfig;
    use crate::cascade::circuit::CircuitDomain;
    use crate::cascade::circuit_text::render_circuit_example;
    use crate::cascade::domain::{CircuitConfig, Grade, GridDomain};
    use crate::cascade::genome::MAX_SECTION_CHARS;
    use crate::cascade::puzzle::{Grid, generate_panel, render_example};
    use crate::cascade::qubit::{Circuit, Gate};
    use crate::util::{BrutError, derive_seed};

    #[test]
    fn the_word_list_holds_both_models() {
        let words = simulated_model_words();

        for w in [
            "xor",
            "meet in the middle",
            "previous circuit",
            "mirror",
            "fall",
            "farb",
        ] {
            assert!(words.contains(&w), "{w}");
        }

        assert!(
            words
                .iter()
                .any(|w| w.starts_with("Before you answer, test your rule"))
        );
        assert!(
            words
                .iter()
                .any(|w| w.starts_with("Before you answer, run your circuit"))
        );
        assert_eq!(model_word_in("Use GRAVITY."), Some("gravity"));
        assert_eq!(model_word_in("Solved examples."), None);
    }

    #[test]
    fn the_template_holds_no_model_word() {
        let circuit = Circuit::new(vec![Gate::H(0)]).expect("circuit");
        let circuits = CircuitDomain::new(&CircuitConfig::default()).expect("domain");
        let task = circuits.panel(5, 1).remove(0);
        let puzzle = generate_panel(5, 1).remove(0);

        for text in [
            TEACHING_INTRO.to_string(),
            section_block(TEACHING_INTRO),
            format!("{TEACHING_INTRO}\nEXAMPLE 1\nEXAMPLE 2"),
            render_example(&puzzle.test_input, &puzzle.test_output),
            render_circuit_example(&task.target, &circuit),
        ] {
            assert_eq!(model_word_in(&text), None, "{text}");
            assert!(check_section_text(&text).is_ok(), "{text}");
        }

        // The block reads as a section of its own, after whatever comes before it.
        assert_eq!(section_block("x"), "## upbringing\nx\n\n");
    }

    #[test]
    fn teaching_takes_each_parents_first_best_solved_task() {
        let grid = GridDomain;
        let panel = generate_panel(derive_seed(5, "panel/search"), 6);
        let right = |i: usize| Some(panel[i].test_output.clone());
        let wrong = |i: usize| {
            let mut g = panel[i].test_output.clone();

            g[0][0] = (g[0][0] + 1) % 10;
            Some(g)
        };
        let example = |i: usize| render_example(&panel[i].test_input, &panel[i].test_output);

        // Parent A solves tasks 2 and 4, parent B tasks 1 and 2, parent C none.
        let a: Vec<Option<Grid>> = vec![wrong(0), None, right(2), wrong(3), right(4), None];
        let b: Vec<Option<Grid>> = vec![None, right(1), right(2), None, None, None];
        let c: Vec<Option<Grid>> = vec![wrong(0), None, None, None, None, wrong(5)];

        assert_eq!(
            teaching(&grid, &panel, &[&a, &b]).expect("teach"),
            Some(format!(
                "{TEACHING_INTRO}\nEXAMPLE 1\n{}\nEXAMPLE 2\n{}",
                example(2),
                example(1)
            ))
        );

        // A parent without a solved task gives no example; without any, no section.
        assert_eq!(
            teaching(&grid, &panel, &[&c, &a]).expect("teach"),
            Some(format!("{TEACHING_INTRO}\nEXAMPLE 1\n{}", example(2)))
        );
        assert_eq!(teaching(&grid, &panel, &[&c]).expect("teach"), None);
        assert_eq!(teaching(&grid, &panel, &[]).expect("teach"), None);

        // The same example twice is taken once.
        assert_eq!(
            teaching(&grid, &panel, &[&a, &a]).expect("teach"),
            teaching(&grid, &panel, &[&a]).expect("teach")
        );
    }

    /// Jobs whose solved answers are worth the job's own value: any answer but 0 solves its
    /// job, and 999 trips the integrity alarm.
    struct Valued;

    #[derive(Clone, Debug)]
    struct Job {
        id: String,
        value: f64,
    }

    impl Domain for Valued {
        type Task = Job;
        type Answer = u32;

        fn from_config(_cfg: &BreedConfig) -> Result<Valued> {
            Ok(Valued)
        }

        fn panel(&self, _seed: u64, n: usize) -> Vec<Job> {
            [0.5, 0.9, 0.9, 0.7]
                .iter()
                .cycle()
                .take(n)
                .enumerate()
                .map(|(i, &value)| Job {
                    id: format!("job{i}"),
                    value,
                })
                .collect()
        }

        fn task_id<'t>(&self, task: &'t Job) -> &'t str {
            &task.id
        }

        fn solve_prompt(&self, task: &Job, _previous: &[u32]) -> String {
            task.id.clone()
        }

        fn judge_prompt(&self, task: &Job, _a: &u32, _b: &u32) -> String {
            task.id.clone()
        }

        fn parse_answer(&self, _task: &Job, text: &str) -> Option<u32> {
            text.parse().ok()
        }

        fn grade(&self, task: &Job, answer: &u32) -> Result<Grade> {
            if *answer == 999 {
                return Err(BrutError::Integrity("999 beats a bound".to_string()));
            }

            let solved = *answer > 0;

            Ok(Grade {
                correct: solved,
                solved,
                value: if solved { task.value } else { 0.0 },
            })
        }

        fn prefer(&self, _task: &Job, _a: &u32, _b: &u32) -> Option<bool> {
            None
        }

        fn graded(&self) -> bool {
            true
        }

        fn artefact(&self, answer: &u32) -> Option<String> {
            Some(answer.to_string())
        }

        fn self_check_text(&self) -> &'static str {
            "Check once."
        }

        fn example(&self, task: &Job, answer: &u32) -> Option<String> {
            Some(format!("{} gives {answer}", task.id))
        }
    }

    #[test]
    fn teaching_takes_the_first_task_of_the_highest_value() {
        let panel = Valued.panel(0, 4);
        let section = |answers: &[Option<u32>]| teaching(&Valued, &panel, &[answers]);
        let one = |example: &str| Some(format!("{TEACHING_INTRO}\nEXAMPLE 1\n{example}"));

        // Values 0.5, 0.9, 0.9 and 0.7: the first of the two worth 0.9.
        assert_eq!(
            section(&[Some(1), Some(2), Some(3), Some(4)]).expect("teach"),
            one("job1 gives 2")
        );
        assert_eq!(
            section(&[Some(1), None, Some(3), Some(4)]).expect("teach"),
            one("job2 gives 3")
        );
        // An unsolved task counts nothing, whatever its value.
        assert_eq!(
            section(&[Some(1), Some(0), None, Some(4)]).expect("teach"),
            one("job3 gives 4")
        );
        assert_eq!(section(&[Some(0), None, None, None]).expect("teach"), None);

        // The domain's integrity alarm is not swallowed.
        assert!(section(&[Some(999), None, None, None]).is_err());
    }

    #[test]
    fn teaching_keeps_the_section_limit() {
        let circuits = CircuitDomain::new(&CircuitConfig::default()).expect("domain");
        let panel = circuits.panel(derive_seed(9, "panel/search"), 22);
        let swaps: Vec<usize> = (0..panel.len())
            .filter(|&i| panel[i].name == "swap")
            .collect();
        let cnot = |control: u8, target: u8| Gate::Cnot { control, target };
        // The swap at its reference, 3 CNOTs and no T, with `pad` H gates that change
        // nothing, dressed for task i.
        let exact = |i: usize, pad: usize| {
            let mut gates = vec![cnot(0, 1), cnot(1, 0), cnot(0, 1)];

            gates.extend(std::iter::repeat_n(Gate::H(0), pad));

            let plain = Circuit::new(gates).expect("circuit");
            let answer = panel[i].dressing.dress_circuit(&plain).expect("dressed");

            assert!(circuits.grade(&panel[i], &answer).expect("grade").solved);

            answer
        };
        let answers = |i: usize, c: Circuit| {
            let mut all: Vec<Option<Circuit>> = vec![None; panel.len()];

            all[i] = Some(c);
            all
        };
        let example = |i: usize, c: &Circuit| render_circuit_example(&panel[i].target, c);

        assert_eq!(swaps.len(), 2, "two dressings of swap");

        let (long_a, long_b, short) =
            (exact(swaps[0], 50), exact(swaps[1], 50), exact(swaps[1], 0));
        let (a, b, c) = (
            answers(swaps[0], long_a.clone()),
            answers(swaps[1], long_b.clone()),
            answers(swaps[1], short.clone()),
        );
        let first = format!(
            "{TEACHING_INTRO}\nEXAMPLE 1\n{}",
            example(swaps[0], &long_a)
        );

        // Each long example fits alone, but not both together: the second is left out.
        assert_eq!(
            teaching(&circuits, &panel, &[&a, &b]).expect("teach"),
            Some(first.clone())
        );
        assert_eq!(
            teaching(&circuits, &panel, &[&b]).expect("teach"),
            Some(format!(
                "{TEACHING_INTRO}\nEXAMPLE 1\n{}",
                example(swaps[1], &long_b)
            ))
        );
        assert!(
            format!("{first}\nEXAMPLE 2\n{}", example(swaps[1], &long_b))
                .chars()
                .count()
                > MAX_SECTION_CHARS
        );

        // A short one still fits after it.
        let both = teaching(&circuits, &panel, &[&a, &c])
            .expect("teach")
            .expect("a section");

        assert_eq!(
            both,
            format!("{first}\nEXAMPLE 2\n{}", example(swaps[1], &short))
        );
        assert!(both.chars().count() <= MAX_SECTION_CHARS);
        assert_eq!(model_word_in(&both), None);
    }

    #[test]
    fn imprinting_follows_the_sources_in_parent_order() {
        let birth = [0u16, 30000];
        let first = [16384u16, 30000];
        let second = [32768u16, 0];

        // κ = 1000: the first source's phases, then the second's.
        assert_eq!(imprinted(&birth, &[(&first, 1000)], true), first);
        assert_eq!(
            imprinted(&birth, &[(&first, 1000), (&second, 1000)], true),
            second
        );
        // Each step starts where the one before ended.
        assert_eq!(
            imprinted(&birth, &[(&first, 500), (&second, 200)], true),
            imprint(&imprint(&birth, &first, 500), &second, 200)
        );
        // Off, without a source, or at κ = 0, the birth Soul stays.
        assert_eq!(imprinted(&birth, &[(&first, 1000)], false), birth);
        assert_eq!(imprinted(&birth, &[], true), birth);
        assert_eq!(imprinted(&birth, &[(&first, 0), (&second, 0)], true), birth);
    }
}
