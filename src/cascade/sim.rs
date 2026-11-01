//! A simulated Worker model, the positive control of Cascade breeding (Stage 2c, ADR 0005).
//!
//! `SimModel` is not a language model. It reads the task from the prompt text alone (it
//! never sees a Puzzle, its rule or its ground truth) and answers by testing the rule
//! hypotheses its system prompt names, within its thinking budget. Its skill therefore
//! depends only on the system prompt and the Traits. We know which prompts and Traits
//! make it strong, so a search that cannot find them is broken.
//!
//! How it answers, deterministically from the request:
//! 1. Knowledge: keyword families in the system prompt (mirror, rotate, transpose,
//!    gravity, colour, compose, previous answer) give the hypotheses, in the order the
//!    prompt first names them. A prompt that names none knows only Identity.
//! 2. Thinking: testing one hypothesis on one example costs `test_cost` thinking tokens,
//!    and the first test of a call is free. A hypothesis is tested on the first example
//!    only, or on every example when the system prompt carries `SELF_CHECK_TEXT`.
//! 3. Temperature: with probability 0.5 * temperature / TEMPERATURE_MAX the hypotheses
//!    are tried in shuffled order.
//! 4. Solve: keep a previous answer that a fitting hypothesis explains (if the prompt
//!    names that check), else apply the first fitting hypothesis, else pass the latest
//!    previous answer through, else guess with the first hypothesis.
//! 5. Judge: pick the one candidate a fitting hypothesis explains, else flip a coin.
#![forbid(unsafe_code)]

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::backend::{GenRequest, GenResponse, ModelBackend, StopReason, approx_tokens};
use crate::cascade::genome::TEMPERATURE_MAX;
use crate::cascade::puzzle::{
    Geo, Grid, Pair, ParsedPrompt, Rule, SELF_CHECK_TEXT, Verdict, apply, apply_geo,
    fit_colour_map, parse_prompt, render_answer, render_verdict,
};
use crate::util::{Result, derive_seed};

/// The reply to a prompt that does not follow the puzzle protocol.
const CANNOT_READ: &str = "I cannot read this task.";

/// A family of rules a system prompt can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Mirror,
    Rotate,
    Transpose,
    Gravity,
    /// The colour map alone (Identity, then a fitted map).
    Colour,
    /// Together with Colour: every known geometric step followed by a fitted map.
    Compose,
    /// Verify the answers of earlier stages.
    Previous,
}

/// Each family with the keywords that name it (matched in the lower-cased system prompt).
const FAMILIES: [(Family, &[&str]); 7] = [
    (Family::Mirror, &["mirror", "flip", "spiegel"]),
    (Family::Rotate, &["rotat", "dreh"]),
    (Family::Transpose, &["transpos", "diagonal"]),
    (Family::Gravity, &["gravity", "fall", "schwerkraft"]),
    (Family::Colour, &["colour", "color", "farb"]),
    (
        Family::Compose,
        &["combine", "compose", "two steps", "kombin"],
    ),
    (
        Family::Previous,
        &["previous answer", "earlier answer", "vorige antwort"],
    ),
];

impl Family {
    /// The geometric steps this family knows.
    fn geos(self) -> &'static [Geo] {
        match self {
            Family::Mirror => &[Geo::FlipH, Geo::FlipV],
            Family::Rotate => &[Geo::Rot90, Geo::Rot180, Geo::Rot270],
            Family::Transpose => &[Geo::Transpose],
            Family::Gravity => &[Geo::GravityDown],
            Family::Colour | Family::Compose | Family::Previous => &[],
        }
    }

    /// The family's name, as `family_names` lists it.
    fn name(self) -> &'static str {
        match self {
            Family::Mirror => "Mirror",
            Family::Rotate => "Rotate",
            Family::Transpose => "Transpose",
            Family::Gravity => "Gravity",
            Family::Colour => "Colour",
            Family::Compose => "Compose",
            Family::Previous => "Previous",
        }
    }
}

/// The names of the families a system prompt can name, in the order of `FAMILIES`: the
/// hypothesis families of a Soul on grid puzzles (step 2, spec 5.3).
pub fn family_names() -> Vec<&'static str> {
    FAMILIES.iter().map(|&(family, _)| family.name()).collect()
}

/// The Family strengths of a puzzle with the hidden rule `rule`, in the order of
/// `family_names` (step 2, spec 5.3): 1 for the family that knows the rule's geometric step
/// (none for the identity), 1 for Colour when the rule carries a colour map, 1 for Compose
/// when it carries both, and 0 for every other family, Previous included.
pub fn rule_strengths(rule: &Rule) -> Vec<f64> {
    let moves = rule.geo != Geo::Identity;
    let colours = rule.colours.is_some();

    FAMILIES
        .iter()
        .map(|&(family, _)| {
            let fits = match family {
                Family::Colour => colours,
                Family::Compose => moves && colours,
                Family::Previous => false,
                Family::Mirror | Family::Rotate | Family::Transpose | Family::Gravity => {
                    family.geos().contains(&rule.geo)
                }
            };

            if fits { 1.0 } else { 0.0 }
        })
        .collect()
}

/// The families a system prompt names, in the order of their first keyword.
fn knowledge(system: &str) -> Vec<Family> {
    let lower = system.to_lowercase();
    let mut named: Vec<(usize, Family)> = FAMILIES
        .iter()
        .filter_map(|&(family, words)| {
            let first = words.iter().filter_map(|w| lower.find(w)).min();

            first.map(|at| (at, family))
        })
        .collect();

    named.sort_by_key(|&(at, _)| at);
    named.into_iter().map(|(_, family)| family).collect()
}

/// One rule hypothesis: a geometric step, then (if `colour`) a colour map fitted to the
/// checked examples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Hypothesis {
    geo: Geo,
    colour: bool,
}

/// The hypotheses a Worker with this knowledge tries, in order.
fn hypotheses(known: &[Family]) -> Vec<Hypothesis> {
    let mut out = Vec::new();

    for &family in known {
        if family == Family::Colour {
            out.push(Hypothesis {
                geo: Geo::Identity,
                colour: true,
            });
        }

        out.extend(
            family
                .geos()
                .iter()
                .map(|&geo| Hypothesis { geo, colour: false }),
        );
    }

    if known.contains(&Family::Compose) && known.contains(&Family::Colour) {
        let geos = known.iter().flat_map(|family| family.geos());

        out.extend(geos.map(|&geo| Hypothesis { geo, colour: true }));
    }

    if out.is_empty() {
        out.push(Hypothesis {
            geo: Geo::Identity,
            colour: false,
        });
    }

    out
}

/// What testing one hypothesis on the checked examples showed.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Outcome {
    /// It reproduces every checked example; the rule carries any fitted colour map.
    Fits(Rule),
    /// It fails a checked example.
    Fails,
    /// The thinking budget ran out before it was tested on every checked example.
    Unfinished,
}

/// The thinking of one call: the examples a hypothesis must pass, the budget, and the
/// outcome of every hypothesis tested so far, so that none is paid for twice.
struct Thinker<'a> {
    checked: &'a [Pair],
    budget: u32,
    test_cost: u32,
    used: u32,
    free_test: bool,
    exhausted: bool,
    tested: Vec<(Hypothesis, Outcome)>,
}

impl<'a> Thinker<'a> {
    fn new(checked: &'a [Pair], budget: u32, test_cost: u32) -> Self {
        Thinker {
            checked,
            budget,
            test_cost,
            used: 0,
            free_test: true,
            exhausted: false,
            tested: Vec::new(),
        }
    }

    /// Pay for testing one hypothesis on one example. The first test of a call is free;
    /// each later one costs `test_cost`, and once the budget cannot cover one, thinking
    /// stops for the rest of the call.
    fn pay(&mut self) -> bool {
        if self.exhausted {
            return false;
        }

        if self.free_test {
            self.free_test = false;
            return true;
        }

        match self.used.checked_add(self.test_cost) {
            Some(total) if total <= self.budget => {
                self.used = total;
                true
            }
            _ => {
                self.exhausted = true;
                false
            }
        }
    }

    /// The outcome of `h`, tested on first use.
    fn outcome(&mut self, h: Hypothesis) -> Outcome {
        if let Some((_, known)) = self.tested.iter().find(|(t, _)| *t == h) {
            return known.clone();
        }

        let fresh = self.test(h);

        self.tested.push((h, fresh.clone()));
        fresh
    }

    /// Test `h` example by example, stopping at the first failure. A colour hypothesis
    /// refits its map over every example tested so far.
    fn test(&mut self, h: Hypothesis) -> Outcome {
        let checked = self.checked;
        let mut moved: Vec<Grid> = Vec::new();
        let mut colours = None;

        for example in checked {
            if !self.pay() {
                return Outcome::Unfinished;
            }

            let step = apply_geo(h.geo, &example.input);

            if h.colour {
                moved.push(step);

                let pairs: Vec<(&Grid, &Grid)> = moved
                    .iter()
                    .zip(checked)
                    .map(|(m, e)| (m, &e.output))
                    .collect();

                match fit_colour_map(&pairs) {
                    Some(map) => colours = Some(map),
                    None => return Outcome::Fails,
                }
            } else if step != example.output {
                return Outcome::Fails;
            }
        }

        Outcome::Fits(Rule {
            geo: h.geo,
            colours,
        })
    }
}

/// The solve answer: a previous answer that a fitting hypothesis explains (latest first,
/// only when `verify_previous`), else the first fitting hypothesis applied to the test
/// input, else the latest previous answer passed through, else an unchecked guess with
/// the first hypothesis (which has no fitted colour map, so only its step applies).
fn solve(
    task: &ParsedPrompt,
    verify_previous: bool,
    tries: &[Hypothesis],
    thinker: &mut Thinker<'_>,
) -> Grid {
    let test = &task.test_input;

    if verify_previous {
        for previous in task.previous.iter().rev() {
            for &h in tries {
                if let Outcome::Fits(rule) = thinker.outcome(h)
                    && apply(&rule, test) == *previous
                {
                    return previous.clone();
                }
            }
        }
    }

    let fitted = tries.iter().find_map(|&h| match thinker.outcome(h) {
        Outcome::Fits(rule) => Some(rule),
        Outcome::Fails | Outcome::Unfinished => None,
    });

    if let Some(rule) = fitted {
        return apply(&rule, test);
    }

    if let Some(previous) = task.previous.last() {
        return previous.clone();
    }

    apply_geo(tries.first().map_or(Geo::Identity, |h| h.geo), test)
}

/// The judge's verdict: the one candidate some fitting hypothesis explains, else a coin
/// flip (when both or neither are explained).
fn judge(
    test: &Grid,
    (a, b): (&Grid, &Grid),
    tries: &[Hypothesis],
    thinker: &mut Thinker<'_>,
    rng: &mut ChaCha8Rng,
) -> Verdict {
    let (mut explains_a, mut explains_b) = (false, false);

    for &h in tries {
        if let Outcome::Fits(rule) = thinker.outcome(h) {
            let predicted = apply(&rule, test);

            explains_a |= predicted == *a;
            explains_b |= predicted == *b;
        }

        if explains_a && explains_b {
            break;
        }
    }

    match (explains_a, explains_b) {
        (true, false) => Verdict::A,
        (false, true) => Verdict::B,
        _ if rng.random_bool(0.5) => Verdict::A,
        _ => Verdict::B,
    }
}

/// Chance that temperature shuffles the hypotheses: none at 0, one half at the ceiling.
fn shuffle_chance(temperature: f32) -> f64 {
    let t = f64::from(temperature) / f64::from(TEMPERATURE_MAX);

    if t.is_nan() {
        0.0
    } else {
        0.5 * t.clamp(0.0, 1.0)
    }
}

/// The solve reply cut to `max_tokens` by dropping trailing rows; the ANSWER line stays
/// as long as it fits at all.
fn answer_reply(answer: &Grid, max_tokens: u32) -> (String, StopReason) {
    for keep in (0..=answer.len()).rev() {
        let rows: Grid = answer.iter().take(keep).cloned().collect();
        let text = render_answer(&rows);

        if approx_tokens(&text) <= max_tokens {
            let stop = if keep == answer.len() {
                StopReason::Stop
            } else {
                StopReason::Length
            };

            return (text, stop);
        }
    }

    (String::new(), StopReason::Length)
}

/// Any other reply cut to `max_tokens` words.
fn word_reply(text: &str, max_tokens: u32) -> (String, StopReason) {
    if approx_tokens(text) <= max_tokens {
        return (text.to_string(), StopReason::Stop);
    }

    let words = usize::try_from(max_tokens).unwrap_or(usize::MAX);
    let kept: Vec<&str> = text.split_whitespace().take(words).collect();

    (kept.join(" "), StopReason::Length)
}

/// A simulated Worker model. NOT a language model: a positive control whose skill depends
/// on the system prompt and Traits only.
#[derive(Clone, Debug)]
pub struct SimModel {
    tag: String,
    /// Thinking tokens that testing one hypothesis on one example costs.
    test_cost: u32,
}

impl SimModel {
    /// The small model "sim:small": 64 thinking tokens per test.
    pub fn small() -> SimModel {
        SimModel {
            tag: "sim:small".to_string(),
            test_cost: 64,
        }
    }

    /// The large model "sim:large": 32 thinking tokens per test, so the same budget
    /// tests twice as much.
    pub fn large() -> SimModel {
        SimModel {
            tag: "sim:large".to_string(),
            test_cost: 32,
        }
    }

    /// The simulated model with this tag, if there is one.
    pub fn from_tag(tag: &str) -> Option<SimModel> {
        [SimModel::small(), SimModel::large()]
            .into_iter()
            .find(|m| m.tag == tag)
    }

    /// The model tag, as on the allow-list of `CascadeLimits`.
    pub fn tag(&self) -> &str {
        &self.tag
    }

    fn response(
        &self,
        text: String,
        stop: StopReason,
        prompt_tokens: u32,
        thinking: u32,
    ) -> GenResponse {
        GenResponse {
            completion_tokens: thinking.saturating_add(approx_tokens(&text)),
            prompt_tokens,
            text,
            stop_reason: stop,
            backend: self.tag.clone(),
        }
    }
}

impl ModelBackend for SimModel {
    fn name(&self) -> &str {
        &self.tag
    }

    fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
        let prompt_tokens = approx_tokens(&req.system).saturating_add(approx_tokens(&req.prompt));

        let Some(task) = parse_prompt(&req.prompt) else {
            let (text, stop) = word_reply(CANNOT_READ, req.max_tokens);

            return Ok(self.response(text, stop, prompt_tokens, 0));
        };

        let known = knowledge(&req.system);
        let mut tries = hypotheses(&known);
        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(req.seed, "sim"));

        if rng.random::<f64>() < shuffle_chance(req.temperature) {
            tries.shuffle(&mut rng);
        }

        let checked = if req.system.contains(SELF_CHECK_TEXT) {
            task.examples.as_slice()
        } else {
            task.examples.get(..1).unwrap_or_default()
        };
        let mut thinker = Thinker::new(checked, req.think_tokens, self.test_cost);

        let (text, stop) = match &task.candidates {
            Some((a, b)) => {
                let verdict = judge(&task.test_input, (a, b), &tries, &mut thinker, &mut rng);

                word_reply(render_verdict(verdict), req.max_tokens)
            }
            None => {
                let verify = known.contains(&Family::Previous);
                let answer = solve(&task, verify, &tries, &mut thinker);

                answer_reply(&answer, req.max_tokens)
            }
        };

        Ok(self.response(text, stop, prompt_tokens, thinker.used))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::cascade::genome::{CascadeLimits, seed_cascades};
    use crate::cascade::puzzle::{
        Puzzle, generate_panel, parse_answer, parse_verdict, render_judge_prompt,
        render_solve_prompt,
    };

    /// Names every family but Previous, and composition.
    const STRONG: &str = "Look for mirror images, rotations, transposition, gravity and colour changes. Combine two steps when one is not enough.";

    const MIRROR: &str = "Look for mirror images.";

    fn grid(rows: &[&[u8]]) -> Grid {
        rows.iter().map(|row| row.to_vec()).collect()
    }

    /// A hand-made puzzle from three example inputs and a test input; every output comes
    /// from the rule.
    fn puzzle(geo: Geo, colours: Option<[u8; 10]>, grids: [&[&[u8]]; 4]) -> Puzzle {
        let rule = Rule { geo, colours };
        let [a, b, c, test_input] = grids.map(grid);
        let train = [a, b, c]
            .into_iter()
            .map(|input| Pair {
                output: apply(&rule, &input),
                input,
            })
            .collect();
        let test_output = apply(&rule, &test_input);
        let id = "hand-made".to_string();

        Puzzle {
            id,
            rule,
            train,
            test_input,
            test_output,
        }
    }

    /// Gravity; every example and the test input change.
    fn gravity_puzzle() -> Puzzle {
        let grids: [&[&[u8]]; 4] = [
            &[&[1, 0, 2], &[0, 0, 0], &[0, 3, 0]],
            &[&[2, 2, 0], &[0, 1, 0], &[0, 0, 0]],
            &[&[0, 3, 0], &[1, 0, 0], &[0, 0, 1]],
            &[&[3, 0, 1], &[0, 2, 0], &[0, 0, 0]],
        ];

        puzzle(Geo::GravityDown, None, grids)
    }

    /// A quarter turn, then the colour cycle 1 -> 2 -> 3 -> 1. The first example shows
    /// every colour, so a map fitted on it alone is complete.
    fn turn_and_recolour_puzzle() -> Puzzle {
        let grids: [&[&[u8]]; 4] = [
            &[&[1, 2, 3], &[0, 0, 1]],
            &[&[3, 0], &[2, 1], &[0, 0]],
            &[&[2, 2, 0], &[0, 3, 0], &[1, 0, 0]],
            &[&[1, 0], &[3, 2]],
        ];

        puzzle(Geo::Rot90, Some([0, 2, 3, 1, 4, 5, 6, 7, 8, 9]), grids)
    }

    /// Mirror top-bottom. The first input is symmetric under a half turn, so on the first
    /// example alone mirroring left-right fits just as well; the other examples tell.
    fn ambiguous_mirror_puzzle() -> Puzzle {
        let grids: [&[&[u8]]; 4] = [
            &[&[1, 2, 0], &[0, 3, 0], &[0, 2, 1]],
            &[&[1, 0, 0], &[2, 0, 0], &[0, 0, 3]],
            &[&[0, 1, 1], &[0, 0, 2], &[3, 0, 0]],
            &[&[1, 1, 2], &[0, 0, 0], &[3, 0, 0]],
        ];

        puzzle(Geo::FlipV, None, grids)
    }

    fn request(system: &str, prompt: String, think_tokens: u32) -> GenRequest {
        GenRequest {
            system: system.to_string(),
            prompt,
            max_tokens: 512,
            temperature: 0.0,
            seed: 1,
            think_tokens,
        }
    }

    /// The grid the small model answers to `p`, if its reply parses.
    fn answer(system: &str, p: &Puzzle, previous: &[Grid], think: u32) -> Option<Grid> {
        let req = request(system, render_solve_prompt(p, previous), think);

        parse_answer(&SimModel::small().generate(&req).expect("generate").text)
    }

    /// The verdict the large model gives on candidates `a` and `b`, if its reply parses.
    fn verdict(system: &str, p: &Puzzle, (a, b): (&Grid, &Grid), seed: u64) -> Option<Verdict> {
        let req = GenRequest {
            seed,
            ..request(system, render_judge_prompt(p, a, b), 1024)
        };

        parse_verdict(&SimModel::large().generate(&req).expect("generate").text)
    }

    #[test]
    fn knowledge_follows_the_order_of_first_mention() {
        use Family::{Colour, Compose, Gravity, Mirror, Previous, Rotate, Transpose};

        let german = "Achte auf Spiegelung, Drehung, Farbe; kombiniere; prüfe die vorige Antwort.";

        assert_eq!(
            knowledge("Try COLOURS, then flip it, or let it fall."),
            [Colour, Mirror, Gravity]
        );
        assert_eq!(
            knowledge(german),
            [Mirror, Rotate, Colour, Compose, Previous]
        );
        assert!(knowledge(SELF_CHECK_TEXT).is_empty(), "names no family");

        // The seed Cascades name the family of their strategy; the gravity seed's
        // "coloured cells" also names colour.
        let seeds: Vec<Vec<Family>> = seed_cascades()
            .iter()
            .map(|(_, c)| knowledge(&c.stages[0].workers[0].system_prompt()))
            .collect();

        assert_eq!(
            seeds,
            [
                vec![Mirror],
                vec![Rotate, Transpose],
                vec![Colour],
                vec![Gravity, Colour]
            ]
        );
    }

    #[test]
    fn the_family_names_follow_the_keyword_table() {
        let named: Vec<String> = FAMILIES.iter().map(|(f, _)| format!("{f:?}")).collect();

        assert_eq!(family_names(), named);
    }

    #[test]
    fn knowledge_gives_the_hypotheses() {
        let identity = Hypothesis {
            geo: Geo::Identity,
            colour: false,
        };
        let tried = |system: &str| hypotheses(&knowledge(system));
        let composed = tried("Rotate, recolour, combine.");
        let coloured: Vec<Geo> = composed
            .iter()
            .filter(|h| h.colour)
            .map(|h| h.geo)
            .collect();

        assert_eq!(tried("Solve it."), [identity]);
        assert_eq!(tried("Check the previous answer."), [identity]);
        assert_eq!(tried("Rotate; combine.").len(), 3, "compose needs colour");
        assert_eq!(composed.len(), 7);
        assert_eq!(
            coloured,
            [Geo::Identity, Geo::Rot90, Geo::Rot180, Geo::Rot270]
        );
    }

    #[test]
    fn naming_the_right_family_solves_a_single_step_puzzle() {
        let p = gravity_puzzle();
        let right = Some(p.test_output.clone());

        for think in [0, 256] {
            let unrelated = answer("Look for rotations.", &p, &[], think);

            assert_eq!(answer("Look for gravity.", &p, &[], think), right);
            assert!(unrelated.is_some() && unrelated != right, "think {think}");
        }

        // A prompt that names nothing knows only Identity and echoes the test input.
        assert_eq!(answer("Solve it.", &p, &[], 256), Some(p.test_input));
    }

    #[test]
    fn a_two_step_rule_needs_colour_compose_and_budget() {
        let p = turn_and_recolour_puzzle();
        let full = "Look for rotations and colour changes; combine two steps.";
        let solves = |system: &str, think: u32| {
            answer(system, &p, &[], think) == Some(p.test_output.clone())
        };

        assert!(solves(full, 4096));
        assert!(solves(full, 256), "five tests: one free, four at 64 tokens");
        assert!(!solves(full, 255), "one token short of the fifth test");
        assert!(!solves(full, 0));
        assert!(!solves("Rotate; recolour.", 4096), "no compose");
        assert!(!solves("Rotate; combine.", 4096), "no colour");
    }

    #[test]
    fn self_check_resolves_an_ambiguous_first_example() {
        let p = ambiguous_mirror_puzzle();
        let checking = format!("{MIRROR}\n{SELF_CHECK_TEXT}");
        let left_right = Some(apply_geo(Geo::FlipH, &p.test_input));

        assert_ne!(left_right, Some(p.test_output.clone()));
        assert_eq!(answer(MIRROR, &p, &[], 4096), left_right, "fooled");
        assert_eq!(
            answer(&checking, &p, &[], 4096),
            Some(p.test_output.clone())
        );

        // Checking every example costs thinking: with no budget the check cannot finish,
        // and the Worker falls back to an unchecked guess with its first hypothesis.
        assert_eq!(answer(&checking, &p, &[], 0), left_right);

        // FlipH passes example 1 (free) and fails example 2; FlipV passes all three.
        let req = request(&checking, render_solve_prompt(&p, &[]), 4096);
        let text = approx_tokens(&render_answer(&p.test_output));
        let thinking = |m: SimModel| m.generate(&req).expect("generate").completion_tokens - text;

        assert_eq!(thinking(SimModel::small()), 4 * 64);
        assert_eq!(thinking(SimModel::large()), 4 * 32);
    }

    #[test]
    fn previous_answers_are_verified_or_passed_through() {
        let p = ambiguous_mirror_puzzle();
        let right = Some(p.test_output.clone());
        let verifying = "Look for mirror images. Check the previous answer.";
        let latest_right = [p.test_output.clone()];
        let older_right = [p.test_output.clone(), grid(&[&[5]])];

        // A Worker that verifies keeps a right previous answer its own first fit would
        // have missed, an older one too; a Worker that does not verify overrides it.
        assert_eq!(answer(verifying, &p, &latest_right, 4096), right);
        assert_eq!(answer(verifying, &p, &older_right, 4096), right);
        assert_ne!(answer(MIRROR, &p, &latest_right, 4096), right);

        // When no hypothesis fits, the latest previous answer is passed through.
        let g = gravity_puzzle();
        let earlier = [grid(&[&[1]]), grid(&[&[4, 4]])];
        let passed = answer("Look for rotations.", &g, &earlier, 4096);

        assert_eq!(passed, Some(grid(&[&[4, 4]])));
    }

    #[test]
    fn judge_prefers_the_candidate_its_knowledge_explains() {
        let p = gravity_puzzle();
        let (right, wrong) = (&p.test_output, &p.test_input);

        for seed in 0..8 {
            assert_eq!(
                verdict("Gravity.", &p, (right, wrong), seed),
                Some(Verdict::A)
            );
            assert_eq!(
                verdict("Gravity.", &p, (wrong, right), seed),
                Some(Verdict::B)
            );
        }

        // A judge that explains neither candidate flips a coin: both verdicts occur.
        let flips: Vec<Option<Verdict>> = (0..32)
            .map(|seed| verdict("Rotations.", &p, (right, wrong), seed))
            .collect();

        assert!(flips.contains(&Some(Verdict::A)) && flips.contains(&Some(Verdict::B)));
        assert!(flips.iter().all(Option::is_some));
    }

    #[test]
    fn temperature_shuffles_the_order_of_hypotheses() {
        let prompt = render_solve_prompt(&ambiguous_mirror_puzzle(), &[]);
        let answers = |temperature: f32| -> HashSet<Option<Grid>> {
            (0..64)
                .map(|seed| {
                    let req = GenRequest {
                        temperature,
                        seed,
                        ..request(MIRROR, prompt.clone(), 4096)
                    };

                    parse_answer(&SimModel::small().generate(&req).expect("generate").text)
                })
                .collect()
        };

        assert_eq!(answers(0.0).len(), 1, "cold: always the first hypothesis");
        assert_eq!(answers(f32::NAN).len(), 1, "NaN counts as cold");
        assert_eq!(
            answers(TEMPERATURE_MAX).len(),
            2,
            "hot: sometimes FlipV first"
        );
    }

    #[test]
    fn responses_depend_on_the_request_only() {
        let p = generate_panel(3, 1).remove(0);
        let mut truth_changed = p.clone();
        let solve = GenRequest {
            temperature: 1.0,
            ..request(STRONG, render_solve_prompt(&p, &[]), 512)
        };
        let judge = request(
            STRONG,
            render_judge_prompt(&p, &p.test_input, &p.test_output),
            512,
        );
        let model = SimModel::small();

        for req in [&solve, &judge] {
            assert_eq!(
                model.generate(req).expect("first"),
                model.generate(req).expect("again")
            );
        }

        truth_changed.test_output = vec![vec![9; 4]; 4];
        assert_eq!(solve.prompt, render_solve_prompt(&truth_changed, &[]));
    }

    #[test]
    fn token_counts_stay_within_the_budgets() {
        let model = SimModel::small();

        for p in generate_panel(8, 20) {
            for think in [0, 64, 300, 4096] {
                for system in [MIRROR, STRONG] {
                    let req = request(system, render_solve_prompt(&p, &[]), think);
                    let reply = model.generate(&req).expect("generate");
                    let text = approx_tokens(&reply.text);
                    let prompt = approx_tokens(&req.system) + approx_tokens(&req.prompt);

                    assert!((text..=text + think).contains(&reply.completion_tokens));
                    assert_eq!(reply.prompt_tokens, prompt);
                    assert_eq!(reply.stop_reason, StopReason::Stop);
                    assert_eq!(reply.backend, "sim:small");
                }
            }
        }
    }

    #[test]
    fn a_tiny_answer_budget_cuts_rows_and_reports_length() {
        let p = gravity_puzzle();
        let reply = |prompt: String, max_tokens: u32| {
            let req = GenRequest {
                max_tokens,
                ..request("Look for gravity.", prompt, 0)
            };
            let r = SimModel::small().generate(&req).expect("generate");

            assert!(
                r.completion_tokens <= max_tokens,
                "no thinking, so all of it is text"
            );
            (r.text, r.stop_reason)
        };
        let solve = |max_tokens| reply(render_solve_prompt(&p, &[]), max_tokens);
        let two_rows = render_answer(&p.test_output[..2].to_vec());
        let judge = render_judge_prompt(&p, &p.test_output, &p.test_input);

        assert_eq!(solve(10), (render_answer(&p.test_output), StopReason::Stop));
        assert_eq!(solve(9), (two_rows, StopReason::Length));
        assert_eq!(solve(0), (String::new(), StopReason::Length));
        assert_eq!(reply(judge, 1), ("VERDICT".to_string(), StopReason::Length));
    }

    #[test]
    fn an_unreadable_prompt_gets_a_plain_reply() {
        let ragged = "TASK\nEXAMPLE 1\nINPUT\n1 2\n3\nOUTPUT\n1\nTEST INPUT\n1\nEND";

        for prompt in ["", "Hello", ragged, "ANSWER\n1"] {
            let req = request(STRONG, prompt.to_string(), 4096);
            let reply = SimModel::large().generate(&req).expect("generate");

            assert_eq!(
                (reply.text.as_str(), reply.stop_reason),
                (CANNOT_READ, StopReason::Stop)
            );
            assert_eq!(
                reply.completion_tokens,
                approx_tokens(CANNOT_READ),
                "no thinking"
            );
        }
    }

    #[test]
    fn every_default_model_tag_has_a_simulated_model() {
        for spec in CascadeLimits::default().models {
            let model = SimModel::from_tag(&spec.tag).expect("simulated model");

            assert_eq!(
                (model.tag(), model.name()),
                (spec.tag.as_str(), spec.tag.as_str())
            );
        }

        assert!(SimModel::from_tag("gpt-unknown").is_none());
    }

    /// Share of `panel` a Worker solves, at the default temperature and one seed per puzzle.
    fn solve_rate(model: &SimModel, system: &str, think: u32, panel: &[Puzzle]) -> f64 {
        let solved = (0u64..)
            .zip(panel)
            .filter(|&(seed, p)| {
                let req = GenRequest {
                    temperature: 0.2,
                    seed,
                    ..request(system, render_solve_prompt(p, &[]), think)
                };
                let reply = model.generate(&req).expect("generate");

                parse_answer(&reply.text).as_ref() == Some(&p.test_output)
            })
            .count();

        solved as f64 / panel.len() as f64
    }

    #[test]
    fn a_strong_worker_solves_clearly_more_than_a_weak_one() {
        let panel = generate_panel(1, 60);
        let strong_prompt = format!("{STRONG}\n{SELF_CHECK_TEXT}");
        let strong = solve_rate(&SimModel::large(), &strong_prompt, 4096, &panel);
        let weak = solve_rate(&SimModel::small(), MIRROR, 0, &panel);

        assert!(strong >= 0.9, "strong Worker solved {strong}");
        assert!(weak <= 0.2, "weak Worker solved {weak}");
    }
}
