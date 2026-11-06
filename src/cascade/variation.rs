//! Variation of Cascades (ADR 0005): random mutation, stage and section crossover, and the
//! requests to and answers from the LLM mutator.
//!
//! Every operator here is pure and deterministic: the same parents and the same rng state
//! give the same child. After an operator the child is clamped to the run's limits
//! (`clamp_to_limits`); it is returned only if it passes `Cascade::check` and its id differs
//! from every parent's, otherwise the operator gives None and the caller decides what to do.
//! The LLM mutator (Claude) is reached only through the `GenRequest`s built here; its answers
//! are recorded and replayed (ADR 0002), and `parse_llm_cascade` turns one answer back into a
//! checked Cascade without repairing it.
//!
//! The operators work on genes (step 2, spec 9.1): each removes its parents' Souls and
//! Coupling strengths first, so a child never carries either, it must differ from its
//! parents' genes, and the draws are those of a Run without Souls. The `_sourced` variants
//! also name the source of every child Worker: the parent Worker it was built from, or one
//! Worker of each parent where the operator pairs Workers. Souletics in the breeder writes
//! the child's Souls from these sources.
//!
//! With the setting `variation` on (arm W, `.scratch/seelen-modell/spec.md`, section 4.1),
//! the `_steered` variants of SectionReplace, SectionAdd and section crossover weight their
//! choices. Beside the genes they get a preference per Worker of the parent they change,
//! and the gene pool's rule that names the families of a text. A weight multiplies the
//! chance that today's operator gives an option, and the result is normalised again. Each
//! choice takes one draw. With every preference 1 the weights are equal, and the operators
//! make exactly today's draws.
#![forbid(unsafe_code)]

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::backend::GenRequest;
use crate::cascade::genome::{
    ANSWER_TOKENS_CEILING, ANSWER_TOKENS_MIN, Cascade, CascadeLimits, MAX_SECTION_CHARS,
    MAX_SECTION_NAME_CHARS, MAX_SECTIONS, Section, Stage, TEMPERATURE_MAX, TEMPERATURE_MIN, Traits,
    Worker,
};
use crate::util::{BrutError, Result};

/// What produced a child; recorded in the lineage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    /// A hand-written starting Cascade; it has no parent.
    Seed,
    /// One section's text replaced by a phrase from the gene pool.
    SectionReplace,
    /// A phrase appended as a new section, or written over a section of the same name.
    SectionAdd,
    /// One section removed; sections named "role" or "format" are never removed.
    SectionDrop,
    /// Two sections of one Worker swapped in order.
    SectionSwap,
    /// One Worker's thinking budget doubled (0 becomes 64), up to its stage's ceiling.
    ThinkUp,
    /// One Worker's thinking budget halved; a budget below 64 becomes 0.
    ThinkDown,
    /// One Worker's temperature moved up or down by 0.2.
    TemperatureShift,
    /// One Worker's self-check switched on or off.
    SelfCheckFlip,
    /// One Worker's answer budget doubled or halved.
    AnswerTokensShift,
    /// One Worker moved to another model on the allow-list.
    ModelSwitch,
    /// A Worker cloned into its own stage with a temperature 0.3 higher.
    WorkerAdd,
    /// One Worker removed from a stage that has more than one.
    WorkerDrop,
    /// A single-Worker stage, inserted anywhere after the first, that checks the previous
    /// answer.
    StageAdd,
    /// One stage removed from a Cascade that has more than one.
    StageDrop,
    /// One-point crossover over stages: a prefix of one parent, a suffix of the other.
    StageCrossover,
    /// Uniform crossover inside Workers: sections, Traits and model from either parent.
    SectionCrossover,
    /// A child proposed by the LLM mutator from one parent.
    LlmMutate,
    /// A child proposed by the LLM mutator by merging two parents.
    LlmCrossover,
}

/// The source of a child Worker (step 2, spec 9.1): the Worker of a parent that the child
/// Worker was built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Source {
    /// The parent, in the order the operator takes them: 0 for the first or only parent,
    /// 1 for the second.
    pub parent: usize,
    /// The Worker's index in that parent, counted over its Stages in order.
    pub worker: usize,
}

/// The sources of a child's Workers: one entry per Worker, in Stage order, with one source,
/// or two in parent order where the operator pairs Workers.
pub type Sources = Vec<Vec<Source>>;

/// The sources of every Worker, stage by stage, while an operator edits the child.
type SourceMap = Vec<Vec<Vec<Source>>>;

/// The floor ε of every weight of arm W (`.scratch/seelen-modell/spec.md`, section 4.1), so
/// that no choice becomes impossible. The spec marks it as a placeholder.
pub const WEIGHT_FLOOR: f64 = 0.05;

/// What steers the section operators in arm W: the gene pool's rule that names the families
/// of a text, in the Domain's family order, and a preference per Worker of the parent that
/// the operator changes, by the Worker's index over the parent's Stages. A preference holds
/// one number w_f in [0, 2] per family; 1 neither favours nor disfavours the family. For
/// section crossover the preferences are those of the first parent.
#[derive(Clone, Copy, Debug)]
pub struct Steer<'a> {
    pub families_of: fn(&str) -> Vec<usize>,
    pub preferences: &'a [Vec<f64>],
}

impl Steer<'_> {
    /// The lens of the parent's Worker `k`; a Worker without a preference leans nowhere.
    fn lens(&self, k: usize) -> Lens<'_> {
        Lens {
            families_of: self.families_of,
            pref: self.preferences.get(k).map_or(&[], Vec::as_slice),
        }
    }
}

/// The choices that the weights of arm W decided in one operator call, and those they
/// changed. Only a choice among options of unequal weight counts; one of equal weights
/// makes today's uniform draw.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Steered {
    /// Choices among options whose weights differ.
    pub choices: u64,
    /// Those of them whose pick differs from the uniform pick of the same draw.
    pub changed: u64,
}

impl Steered {
    /// Add the counts of `other`.
    pub fn add(&mut self, other: Steered) {
        self.choices += other.choices;
        self.changed += other.changed;
    }
}

/// One Worker's preference with the rule that names families: what weighs the choices of an
/// operator that changes this Worker.
#[derive(Clone, Copy, Debug)]
struct Lens<'a> {
    families_of: fn(&str) -> Vec<usize>,
    pref: &'a [f64],
}

impl Lens<'_> {
    /// The lean L of `text`: w_f − 1 summed over the families it names, each once, in
    /// family order. A text that names no family has L = 0.
    fn lean(&self, text: &str) -> f64 {
        (self.families_of)(text)
            .into_iter()
            .filter_map(|f| self.pref.get(f))
            .fold(0.0, |l, w| l + (w - 1.0))
    }

    /// 1 + L, at least ε: the weight of a Worker by its whole prompt, of a phrase, and of
    /// each side of the coin of section crossover.
    fn favour(&self, text: &str) -> f64 {
        (1.0 + self.lean(text)).max(WEIGHT_FLOOR)
    }

    /// 1 − L, at least ε: the weight of a section that SectionReplace may overwrite, so a
    /// Worker protects the texts it prefers.
    fn protect(&self, text: &str) -> f64 {
        (1.0 - self.lean(text)).max(WEIGHT_FLOOR)
    }
}

/// Prompt-text gene pool: (section name, text). Mutation draws section texts from here.
///
/// The pool holds strategies that name one kind of grid rule, strategies that combine two
/// or three kinds, German variants, strategies and checks that point nowhere in particular,
/// residual instructions that review the previous answer, style lines, roles and answer
/// formats. Every text passes the genome's text rules.
pub const PHRASES: &[(&str, &str)] = &[
    // Roles.
    (
        "role",
        "You solve grid puzzles. Each puzzle shows input and output grids that follow one hidden rule.",
    ),
    (
        "role",
        "You are a careful solver of grid puzzles. Study every worked example before you answer.",
    ),
    (
        "role",
        "Du löst Gitterrätsel. Jedes Rätsel zeigt Eingabe- und Ausgabegitter, die einer gemeinsamen Regel folgen.",
    ),
    // Strategies that name one kind of rule.
    (
        "strategy",
        "Look for mirror images: flip the grid left-right or top-bottom.",
    ),
    (
        "strategy",
        "Mirror the grid along its vertical axis and compare the result with the output.",
    ),
    (
        "strategy",
        "Check whether the output is the input rotated by 90, 180 or 270 degrees.",
    ),
    (
        "strategy",
        "Turn the grid by a quarter turn and compare each example with its rotation.",
    ),
    (
        "strategy",
        "Rows may have become columns: compare the output with the input transposed along its main diagonal.",
    ),
    (
        "strategy",
        "Look for gravity: non-zero cells fall down until they rest on the floor or on another cell.",
    ),
    (
        "strategy",
        "Objects drop to the bottom row like stones under gravity.",
    ),
    (
        "strategy",
        "Look for a colour map: every colour turns into one fixed other colour while the layout stays the same.",
    ),
    (
        "strategy",
        "Each color may be swapped for another color; build the color table from the examples.",
    ),
    (
        "strategy",
        "The rule may combine two steps: find one change that explains part of every example, then look for the second.",
    ),
    // Strategies that combine two or three kinds of rule.
    (
        "strategy",
        "Try a mirror flip first, then a rotation; one of them or both together may explain every example.",
    ),
    (
        "strategy",
        "Mirror the grid or transpose it along the diagonal; keep whichever matches every example.",
    ),
    (
        "strategy",
        "Consider mirror, rotation and transposition in turn and keep the one that fits all examples.",
    ),
    (
        "strategy",
        "Cells may fall down under gravity and change colour on the way; look for both effects.",
    ),
    (
        "strategy",
        "Compose two steps: first a rotation, then a colour map.",
    ),
    (
        "strategy",
        "The rule may combine gravity with a mirror image: let the cells fall, then flip the grid.",
    ),
    // German strategies.
    (
        "strategy",
        "Suche nach Spiegelungen: das Gitter wird links-rechts oder oben-unten gespiegelt.",
    ),
    (
        "strategy",
        "Prüfe, ob das Ausgabegitter das um 90 oder 180 Grad gedrehte Eingabegitter ist.",
    ),
    (
        "strategy",
        "Prüfe, ob Zeilen zu Spalten werden: eine Transposition an der Hauptdiagonale.",
    ),
    (
        "strategy",
        "Achte auf Schwerkraft: Zellen sinken nach unten, bis sie aufliegen.",
    ),
    (
        "strategy",
        "Jede Farbe wird zu einer anderen Farbe; die Anordnung der Zellen bleibt gleich.",
    ),
    (
        "strategy",
        "Die Regel kann zwei Schritte kombinieren, etwa eine Drehung und danach einen Farbwechsel.",
    ),
    // Strategies that point nowhere in particular.
    (
        "strategy",
        "Look at shapes and counts first; the rule may move, grow or copy whole objects.",
    ),
    (
        "strategy",
        "Look at the border cells first; the rule may only touch the edge of the grid.",
    ),
    // Checks of the Worker's own answer.
    (
        "check",
        "Before you answer, apply your rule to every example input and compare the result with its output.",
    ),
    (
        "check",
        "If your rule fails on even one example, drop it and test the next idea.",
    ),
    (
        "check",
        "Compare the grid size of input and output: if height and width swap, think of a rotation or a transposition.",
    ),
    (
        "check",
        "Count the cells of each colour in input and output; equal counts hint at a move rather than a recolouring.",
    ),
    (
        "check",
        "Prüfe deine Regel an jedem Beispiel, bevor du antwortest.",
    ),
    // Residual checks of the answer an earlier stage chose.
    (
        "residual",
        "Read the previous answer first. If it follows the rule of every example, repeat it; otherwise fix it.",
    ),
    (
        "residual",
        "Start from the earlier answer and correct only the cells that break the rule.",
    ),
    (
        "residual",
        "Compare the previous answer with each example; keep it if it fits, otherwise solve the puzzle again.",
    ),
    (
        "residual",
        "Check the previous answer cell by cell; if one cell breaks the rule, write a corrected grid.",
    ),
    (
        "residual",
        "Lies zuerst die vorige Antwort. Stimmt sie mit allen Beispielen überein, gib sie unverändert zurück.",
    ),
    // Style.
    ("style", "Be concise."),
    (
        "style",
        "Think step by step, but keep the final answer short.",
    ),
    (
        "style",
        "Work calmly and precisely; one wrong cell makes the whole answer wrong.",
    ),
    ("style", "Write as little as possible outside the grid."),
    (
        "style",
        "Name the rule in one short sentence before you write the grid.",
    ),
    ("style", "Arbeite sorgfältig und knapp."),
    // Answer formats.
    (
        "format",
        "Answer with the line ANSWER followed by the output grid, one row per line, digits separated by spaces.",
    ),
    (
        "format",
        "End with the line ANSWER and then the output grid: one row per line, single digits separated by spaces.",
    ),
];

/// Attempts `mutate` makes before it gives up.
const MUTATE_ATTEMPTS: usize = 32;

/// Draws (cut points or coin flips) each crossover makes before it gives up.
const CROSSOVER_ATTEMPTS: usize = 8;

/// Smallest non-zero thinking budget the operators produce: ThinkUp lifts 0 to it, and
/// ThinkDown turns anything below it into 0.
const THINK_STEP_MIN: u32 = 64;

/// Step of TemperatureShift.
const TEMPERATURE_STEP: f32 = 0.2;

/// Temperature added to the clone in WorkerAdd, so that the copy explores more.
const WORKER_ADD_WARMTH: f32 = 0.3;

/// Probability that SectionReplace draws only from phrases named like the section.
const SAME_NAME_P: f64 = 0.7;

/// Sections SectionDrop never removes: without them a Worker loses its task or its format.
const PROTECTED_SECTIONS: [&str; 2] = ["role", "format"];

/// Name of the section StageAdd gives its new stage.
const RESIDUAL: &str = "residual";

/// Relative weights of the mutation operators, in percent: Trait ops 40, section ops 40,
/// Worker and stage ops 20.
const MUTATION_WEIGHTS: [(Op, u32); 14] = [
    (Op::ThinkUp, 7),
    (Op::ThinkDown, 7),
    (Op::TemperatureShift, 7),
    (Op::SelfCheckFlip, 6),
    (Op::AnswerTokensShift, 6),
    (Op::ModelSwitch, 7),
    (Op::SectionReplace, 16),
    (Op::SectionAdd, 10),
    (Op::SectionDrop, 7),
    (Op::SectionSwap, 7),
    (Op::WorkerAdd, 5),
    (Op::WorkerDrop, 5),
    (Op::StageAdd, 5),
    (Op::StageDrop, 5),
];

/// Upper bound on the tokens of one LLM mutator answer.
const LLM_MAX_TOKENS: u32 = 4096;

/// Sampling temperature of the LLM mutator; the run stays reproducible through recorded
/// answers (ADR 0002), so this only widens the model's choices.
const LLM_TEMPERATURE: f32 = 0.7;

/// System prompt of every LLM mutator request.
const LLM_SYSTEM: &str = "You are the mutation operator of a research breeding system. \
You edit Cascades: JSON data that configures model Workers, which solve grid puzzles. \
Return exactly one JSON object for the new Cascade and nothing else.";

/// What the puzzles are; the first paragraph of every request.
const LLM_CONTEXT: &str = "A Cascade solves grid puzzles. Each puzzle shows a few worked \
examples, each an input grid with its output grid, that follow one rule, and asks for the \
output grid of a new input.\n\n";

/// The Cascade JSON format in words, mirroring `genome.rs`.
const LLM_FORMAT: &str = "Format: a Cascade is a JSON object. Every field below is required \
and no other field is allowed.\n\
- Cascade: {\"stages\": [Stage, ...]}. The stages run in order; every stage sees the puzzle \
and the answers chosen by the earlier stages.\n\
- Stage: {\"workers\": [Worker, ...]}. The Workers of one stage run in parallel on the same \
input.\n\
- Worker: {\"model\": string, \"sections\": [Section, ...], \"traits\": Traits}. The system \
prompt of a Worker is its sections in order, each under its name.\n\
- Section: {\"name\": string, \"text\": string}. Typical names are \"role\", \"strategy\", \
\"check\", \"residual\", \"style\" and \"format\"; texts are plain text.\n\
- Traits: {\"think_tokens\": integer, \"temperature\": number, \"answer_tokens\": integer, \
\"self_check\": boolean}. think_tokens is the budget for reasoning before the answer (0 \
answers at once); answer_tokens bounds the visible answer; self_check makes the Worker \
check its answer against the worked examples once.\n\n";

/// The task of a mutation request.
const LLM_MUTATE_TASK: &str = "Task: change the parent Cascade so that it solves more grid \
puzzles at lower cost. You may rewrite, add or remove sections, change Traits and models, \
and add or remove Workers and stages. ";

/// The task of a crossover request.
const LLM_CROSSOVER_TASK: &str = "Task: merge Parent A and Parent B into one child Cascade \
that keeps what works in each, so that it solves more grid puzzles at lower cost. You may \
take stages, Workers, sections and Traits from either parent and change them. ";

/// The words that tell the Mutator what it breeds, one set per task family (grid puzzles;
/// quantum circuits in step 1). The Cascade format's limits and `LLM_RULES` stay shared.
#[derive(Debug)]
pub struct MutatorBrief {
    pub system: &'static str,
    pub context: &'static str,
    pub format: &'static str,
    pub mutate_task: &'static str,
    pub crossover_task: &'static str,
}

/// The brief of the grid puzzles: the Stage 2c texts, byte for byte, so recorded answer
/// keys stay valid.
pub static GRID_BRIEF: MutatorBrief = MutatorBrief {
    system: LLM_SYSTEM,
    context: LLM_CONTEXT,
    format: LLM_FORMAT,
    mutate_task: LLM_MUTATE_TASK,
    crossover_task: LLM_CROSSOVER_TASK,
};

/// The rules every request ends with.
const LLM_RULES: &str = "Every model call and every thinking token costs fitness, so spend \
them only where they pay off. Keep the result valid under the format and the limits above. \
Section texts are plain-text instructions for a model; never put credentials or anything \
that looks like one into them.\n\n\
Reply with exactly one JSON object for the new Cascade and nothing else.\n";

/// Clamp every Trait into range and each stage's `think_tokens` to
/// `limits.think_ceiling(stage)`, and drop stages and Workers beyond the limits. Used after
/// every operator.
///
/// A NaN temperature becomes the default temperature. A valid Cascade is left unchanged.
/// Models and prompt text are not touched, so a Cascade with an unknown model or a bad
/// section still fails `check`.
pub fn clamp_to_limits(c: &mut Cascade, limits: &CascadeLimits) {
    c.stages.truncate(limits.stages());

    for (k, stage) in c.stages.iter_mut().enumerate() {
        stage.workers.truncate(limits.width());

        let think_cap = limits.think_ceiling(k);

        for worker in &mut stage.workers {
            let t = &mut worker.traits;

            t.think_tokens = t.think_tokens.min(think_cap);
            t.temperature = clamp_temperature(t.temperature);
            t.answer_tokens = t
                .answer_tokens
                .clamp(ANSWER_TOKENS_MIN, ANSWER_TOKENS_CEILING);
        }
    }
}

/// One random structural or Trait mutation. Returns a child that passes `check(limits)` and
/// differs from the parent (by id), with the operator that made it, or None after 32 failed
/// attempts.
///
/// Each attempt draws an operator by weight (Trait ops 40 %, section ops 40 %, Worker and
/// stage ops 20 %); an operator that cannot change this parent is skipped and another is
/// drawn.
pub fn mutate(
    parent: &Cascade,
    limits: &CascadeLimits,
    rng: &mut ChaCha8Rng,
) -> Option<(Cascade, Op)> {
    mutate_with(parent, limits, PHRASES, rng)
}

/// `mutate` for the task family whose phrase pool is `phrases`: the section operators
/// draw their texts from there. The grid puzzles use `PHRASES`.
pub fn mutate_with(
    parent: &Cascade,
    limits: &CascadeLimits,
    phrases: &[(&str, &str)],
    rng: &mut ChaCha8Rng,
) -> Option<(Cascade, Op)> {
    mutate_sourced(parent, limits, phrases, rng).map(|(child, op, _)| (child, op))
}

/// `mutate_with` that also names the source of every child Worker (spec 9.1): each Worker's
/// source is the parent Worker at its position. The clone of WorkerAdd has the cloned
/// Worker as its source, the new Worker of StageAdd the Worker it was copied from, and
/// WorkerDrop and StageDrop shift the positions.
pub fn mutate_sourced(
    parent: &Cascade,
    limits: &CascadeLimits,
    phrases: &[(&str, &str)],
    rng: &mut ChaCha8Rng,
) -> Option<(Cascade, Op, Sources)> {
    mutate_inner(parent, limits, phrases, None, rng)
        .map(|(child, op, sources, _)| (child, op, sources))
}

/// `mutate_sourced` for arm W: SectionReplace and SectionAdd weight their choices by the
/// preferences of `steer`, one per Worker of `parent`. Also returns the choices the
/// weights decided in the attempt that made the child.
pub fn mutate_steered(
    parent: &Cascade,
    limits: &CascadeLimits,
    phrases: &[(&str, &str)],
    steer: &Steer<'_>,
    rng: &mut ChaCha8Rng,
) -> Option<(Cascade, Op, Sources, Steered)> {
    mutate_inner(parent, limits, phrases, Some(steer), rng)
}

/// `mutate_sourced`, weighted by `steer` when it is given.
fn mutate_inner(
    parent: &Cascade,
    limits: &CascadeLimits,
    phrases: &[(&str, &str)],
    steer: Option<&Steer<'_>>,
    rng: &mut ChaCha8Rng,
) -> Option<(Cascade, Op, Sources, Steered)> {
    let parent = parent.genes();
    let parents = [parent.id()];

    for _ in 0..MUTATE_ATTEMPTS {
        let op = draw_op(rng);

        if let Some((child, sources, steered)) =
            attempt(&parent, &parents, op, limits, phrases, steer, rng)
        {
            return Some((child, op, sources, steered));
        }
    }

    None
}

/// One draw of the mutation operator `op` on `parent` (imagination,
/// `.scratch/seelen-modell/spec-imagination.md`, section 3.2): what one attempt of
/// `mutate_sourced` or `mutate_steered` makes once it has drawn `op`, with the same draws, but
/// without drawing the operator and without a second attempt. The operator sees genes only,
/// as every operator does. None when `op` cannot change this parent, or when the child fails
/// `check(limits)` or has the parent's id.
pub fn mutate_once(
    parent: &Cascade,
    op: Op,
    limits: &CascadeLimits,
    phrases: &[(&str, &str)],
    steer: Option<&Steer<'_>>,
    rng: &mut ChaCha8Rng,
) -> Option<(Cascade, Sources, Steered)> {
    let parent = parent.genes();
    let parents = [parent.id()];

    attempt(&parent, &parents, op, limits, phrases, steer, rng)
}

/// One attempt of a mutation with the operator `op` on the genes `parent`, whose id is in
/// `parents`: apply, clamp, and keep the child when it passes `check(limits)` and differs
/// from the parent, with the source of every Worker and the choices the weights decided.
fn attempt(
    parent: &Cascade,
    parents: &[String],
    op: Op,
    limits: &CascadeLimits,
    phrases: &[(&str, &str)],
    steer: Option<&Steer<'_>>,
    rng: &mut ChaCha8Rng,
) -> Option<(Cascade, Sources, Steered)> {
    let mut steered = Steered::default();
    let (mut child, map) = apply_steered(parent, op, limits, phrases, steer, &mut steered, rng)?;

    clamp_to_limits(&mut child, limits);

    if !acceptable(&child, limits, parents) {
        return None;
    }

    let sources = flatten_sources(map, &child);

    Some((child, sources, steered))
}

/// One-point crossover over stages: a prefix of `a` (1..=len stages), then a suffix of `b`,
/// clamped. The child must pass `check(limits)` and differ from both parents; a few cut
/// points are tried before the result is None.
///
/// The cuts leave room under the depth limit for at least one stage of `b`, so the child ends
/// with `b`'s last stage, the one that gives the final answer. A depth limit of 1 leaves no
/// room, and two valid parents then give None.
pub fn stage_crossover(
    a: &Cascade,
    b: &Cascade,
    limits: &CascadeLimits,
    rng: &mut ChaCha8Rng,
) -> Option<Cascade> {
    stage_crossover_sourced(a, b, limits, rng).map(|(child, _)| child)
}

/// `stage_crossover` that also names the source of every child Worker (spec 9.1): the
/// Workers of the prefix come from `a`, those of the suffix from `b`, each from its own
/// position.
pub fn stage_crossover_sourced(
    a: &Cascade,
    b: &Cascade,
    limits: &CascadeLimits,
    rng: &mut ChaCha8Rng,
) -> Option<(Cascade, Sources)> {
    let (a, b) = (&a.genes(), &b.genes());

    if a.stages.is_empty() || b.stages.is_empty() {
        return None;
    }

    let parents = [a.id(), b.id()];
    let depth = limits.stages();
    let prefix_max = a.stages.len().min(depth.saturating_sub(1)).max(1);

    for _ in 0..CROSSOVER_ATTEMPTS {
        let prefix = rng.random_range(1..=prefix_max);
        let room = depth.saturating_sub(prefix).max(1);
        let suffix = rng.random_range(1..=b.stages.len().min(room));

        let head = a.stages.get(..prefix)?;
        let tail = b.stages.get(b.stages.len() - suffix..)?;

        let mut child = Cascade {
            stages: head.iter().chain(tail).cloned().collect(),
        };
        let map: SourceMap = (0..prefix)
            .map(|s| stage_sources(a, 0, s))
            .chain((b.stages.len() - suffix..b.stages.len()).map(|s| stage_sources(b, 1, s)))
            .collect();

        clamp_to_limits(&mut child, limits);

        if acceptable(&child, limits, &parents) {
            let sources = flatten_sources(map, &child);

            return Some((child, sources));
        }
    }

    None
}

/// Uniform crossover inside Workers: the child has `a`'s shape; Worker (s, w) pairs with
/// `b`'s Worker at (min(s, len - 1), min(w, len - 1)); each section of `a`'s Worker whose
/// name also exists in the partner is taken from either with p = 0.5; each Trait field and
/// the model come from either with p = 0.5. Must pass `check(limits)` and differ from both
/// parents, else None.
///
/// The coin flips are repeated a few times before the result is None, since parents that
/// differ in only a few genes often yield a copy of one of them.
pub fn section_crossover(
    a: &Cascade,
    b: &Cascade,
    limits: &CascadeLimits,
    rng: &mut ChaCha8Rng,
) -> Option<Cascade> {
    section_crossover_sourced(a, b, limits, rng).map(|(child, _)| child)
}

/// `section_crossover` that also names the sources of every child Worker (spec 9.1): Worker
/// (s, w) has two, `a`'s Worker (s, w) and the partner Worker of `b`.
pub fn section_crossover_sourced(
    a: &Cascade,
    b: &Cascade,
    limits: &CascadeLimits,
    rng: &mut ChaCha8Rng,
) -> Option<(Cascade, Sources)> {
    section_crossover_inner(a, b, limits, None, rng).map(|(child, sources, _)| (child, sources))
}

/// `section_crossover_sourced` for arm W: the coin of each section weighs the two texts by
/// the preferences of `steer`, one per Worker of the first parent `a`. Also returns the
/// choices the weights decided in the attempt that made the child.
pub fn section_crossover_steered(
    a: &Cascade,
    b: &Cascade,
    limits: &CascadeLimits,
    steer: &Steer<'_>,
    rng: &mut ChaCha8Rng,
) -> Option<(Cascade, Sources, Steered)> {
    section_crossover_inner(a, b, limits, Some(steer), rng)
}

/// `section_crossover_sourced`, weighted by `steer` when it is given.
fn section_crossover_inner(
    a: &Cascade,
    b: &Cascade,
    limits: &CascadeLimits,
    steer: Option<&Steer<'_>>,
    rng: &mut ChaCha8Rng,
) -> Option<(Cascade, Sources, Steered)> {
    let (a, b) = (&a.genes(), &b.genes());

    if a.stages.is_empty() || b.stages.is_empty() {
        return None;
    }

    let parents = [a.id(), b.id()];

    for _ in 0..CROSSOVER_ATTEMPTS {
        let mut child = a.clone();
        let mut steered = Steered::default();

        for (s, stage) in child.stages.iter_mut().enumerate() {
            for (w, worker) in stage.workers.iter_mut().enumerate() {
                if let Some(other) = partner(b, s, w) {
                    let lens = steer.map(|st| st.lens(worker_index(a, s, w)));

                    cross_worker(worker, other, lens, &mut steered, rng);
                }
            }
        }

        let map: SourceMap = a
            .stages
            .iter()
            .enumerate()
            .map(|(s, stage)| {
                (0..stage.workers.len())
                    .map(|w| {
                        let own = Source {
                            parent: 0,
                            worker: worker_index(a, s, w),
                        };

                        std::iter::once(own)
                            .chain(partner_source(b, 1, s, w))
                            .collect()
                    })
                    .collect()
            })
            .collect();

        clamp_to_limits(&mut child, limits);

        if acceptable(&child, limits, &parents) {
            let sources = flatten_sources(map, &child);

            return Some((child, sources, steered));
        }
    }

    None
}

/// The sources of a child the Mutator wrote from `parents` (spec 9.1): Worker (s, w) has the
/// source `partner(parent, s, w)` of every parent, in parent order. The Mutator sees no
/// Souls and writes none, so only the shapes count.
pub fn partner_sources(child: &Cascade, parents: &[&Cascade]) -> Sources {
    child
        .stages
        .iter()
        .enumerate()
        .flat_map(|(s, stage)| {
            (0..stage.workers.len()).map(move |w| {
                parents
                    .iter()
                    .enumerate()
                    .filter_map(|(p, parent)| partner_source(parent, p, s, w))
                    .collect()
            })
        })
        .collect()
}

/// The index of Worker (stage `s`, position `w`) of `c`, counted over its Stages in order.
pub fn worker_index(c: &Cascade, s: usize, w: usize) -> usize {
    c.stages
        .iter()
        .take(s)
        .map(|st| st.workers.len())
        .sum::<usize>()
        + w
}

/// Every Worker of Stage `s` of `c`, each its own source as Worker of parent `parent`.
fn stage_sources(c: &Cascade, parent: usize, s: usize) -> Vec<Vec<Source>> {
    let first = worker_index(c, s, 0);
    let width = c.stages.get(s).map_or(0, |st| st.workers.len());

    (first..first + width)
        .map(|worker| vec![Source { parent, worker }])
        .collect()
}

/// The source that `partner(c, s, w)` picks, as Worker of parent `parent`.
fn partner_source(c: &Cascade, parent: usize, s: usize, w: usize) -> Option<Source> {
    let (ps, pw) = partner_at(c, s, w)?;

    Some(Source {
        parent,
        worker: worker_index(c, ps, pw),
    })
}

/// The sources in Stage order, cut to the shape `child` kept after `clamp_to_limits`, which
/// drops trailing Stages and Workers only.
fn flatten_sources(mut map: SourceMap, child: &Cascade) -> Sources {
    map.truncate(child.stages.len());

    for (stage, sources) in child.stages.iter().zip(&mut map) {
        sources.truncate(stage.workers.len());
    }

    debug_assert_eq!(
        map.iter().map(Vec::len).collect::<Vec<_>>(),
        child
            .stages
            .iter()
            .map(|s| s.workers.len())
            .collect::<Vec<_>>(),
        "the source map follows the child's shape"
    );

    map.into_iter().flatten().collect()
}

/// Request to an LLM mutator (Claude) to improve one parent, given its public search
/// fitness. think_tokens 0, temperature 0.7, max_tokens 4096, seed = `seed`.
///
/// The prompt states the Cascade format, the run's limits, the parent as pretty JSON with
/// its fitness, and the task. Equal inputs give an equal request.
pub fn llm_mutate_request(
    parent: &Cascade,
    fitness: f64,
    limits: &CascadeLimits,
    seed: u64,
) -> GenRequest {
    llm_mutate_request_with(&GRID_BRIEF, parent, fitness, limits, seed)
}

/// `llm_mutate_request` for the task family that `brief` describes.
pub fn llm_mutate_request_with(
    brief: &MutatorBrief,
    parent: &Cascade,
    fitness: f64,
    limits: &CascadeLimits,
    seed: u64,
) -> GenRequest {
    let prompt = llm_prompt(
        brief,
        &[("Parent Cascade", parent, fitness)],
        limits,
        brief.mutate_task,
    );

    llm_request(brief, prompt, seed)
}

/// Request to merge two parents into one child (semantic crossover), with both fitness
/// values. Same generation settings and the same determinism as `llm_mutate_request`.
pub fn llm_crossover_request(
    a: &Cascade,
    fa: f64,
    b: &Cascade,
    fb: f64,
    limits: &CascadeLimits,
    seed: u64,
) -> GenRequest {
    llm_crossover_request_with(&GRID_BRIEF, (a, fa), (b, fb), limits, seed)
}

/// `llm_crossover_request` for the task family that `brief` describes; each parent comes
/// with its fitness.
pub fn llm_crossover_request_with(
    brief: &MutatorBrief,
    (a, fa): (&Cascade, f64),
    (b, fb): (&Cascade, f64),
    limits: &CascadeLimits,
    seed: u64,
) -> GenRequest {
    let prompt = llm_prompt(
        brief,
        &[("Parent A", a, fa), ("Parent B", b, fb)],
        limits,
        brief.crossover_task,
    );

    llm_request(brief, prompt, seed)
}

/// Extract the JSON object from an LLM answer (from the first `{` to the `}` that closes
/// it; braces inside JSON strings do not count, and code fences and prose around the object
/// are ignored), deserialize a Cascade (unknown fields are rejected by the struct), clamp
/// nothing, and run `check(limits)`. Errors are `BrutError::Invalid` with a short reason.
pub fn parse_llm_cascade(text: &str, limits: &CascadeLimits) -> Result<Cascade> {
    let cascade = read_llm_cascade(text)?;

    cascade.check(limits).map_err(|e| match e {
        BrutError::Invalid(reason) => llm_invalid(&reason),
        other => other,
    })?;

    Ok(cascade)
}

/// The Cascade in an LLM answer as `parse_llm_cascade` finds it, but unchecked. The breeder
/// reads it to refuse an answer that writes a Soul before any other rule (step 2, spec 4.5).
pub fn read_llm_cascade(text: &str) -> Result<Cascade> {
    let start = text
        .find('{')
        .ok_or_else(|| llm_invalid("no JSON object"))?;
    let end =
        closing_brace(text, start).ok_or_else(|| llm_invalid("the JSON object is not closed"))?;
    let json = text
        .get(start..=end)
        .ok_or_else(|| llm_invalid("the JSON object is not closed"))?;

    serde_json::from_str(json).map_err(|e| llm_invalid(&format!("not a Cascade: {e}")))
}

/// An error about an LLM answer.
fn llm_invalid(reason: &str) -> BrutError {
    BrutError::Invalid(format!("llm answer: {reason}"))
}

/// Draw one mutation operator by `MUTATION_WEIGHTS`.
fn draw_op(rng: &mut ChaCha8Rng) -> Op {
    let total: u32 = MUTATION_WEIGHTS.iter().map(|(_, w)| w).sum();
    let mut x = rng.random_range(0..total.max(1));

    for (op, w) in MUTATION_WEIGHTS {
        if x < w {
            return op;
        }

        x -= w;
    }

    Op::SectionReplace
}

/// Apply one mutation operator to a copy of `parent`, or None when the operator cannot
/// change this parent. Section texts come from `phrases`. The result is not clamped yet.
#[cfg(test)]
fn apply_with(
    parent: &Cascade,
    op: Op,
    limits: &CascadeLimits,
    phrases: &[(&str, &str)],
    rng: &mut ChaCha8Rng,
) -> Option<Cascade> {
    apply_sourced(parent, op, limits, phrases, rng).map(|(child, _)| child)
}

/// `apply_with` with the source of every child Worker, stage by stage.
#[cfg(test)]
fn apply_sourced(
    parent: &Cascade,
    op: Op,
    limits: &CascadeLimits,
    phrases: &[(&str, &str)],
    rng: &mut ChaCha8Rng,
) -> Option<(Cascade, SourceMap)> {
    apply_steered(
        parent,
        op,
        limits,
        phrases,
        None,
        &mut Steered::default(),
        rng,
    )
}

/// Apply one mutation operator to a copy of `parent`, with the source of every child
/// Worker, stage by stage. A Worker that keeps its position keeps its source; the operators
/// that add, remove or move Workers or Stages edit the map as they edit the child.
///
/// With `steer` (arm W), SectionReplace and SectionAdd weight their choices by its
/// preferences, and `steered` counts the choices the weights decided. Without it every
/// operator draws as before.
fn apply_steered(
    parent: &Cascade,
    op: Op,
    limits: &CascadeLimits,
    phrases: &[(&str, &str)],
    steer: Option<&Steer<'_>>,
    steered: &mut Steered,
    rng: &mut ChaCha8Rng,
) -> Option<(Cascade, SourceMap)> {
    let mut child = parent.clone();
    let mut map: SourceMap = (0..parent.stages.len())
        .map(|s| stage_sources(parent, 0, s))
        .collect();

    match op {
        Op::ThinkUp => {
            let slots = worker_slots(parent, |s, w| {
                w.traits.think_tokens < limits.think_ceiling(s)
            });
            let (s, w) = pick(rng, &slots)?;
            let t = &mut worker_mut(&mut child, s, w)?.traits;

            t.think_tokens = think_up(t.think_tokens, limits.think_ceiling(s));
        }

        Op::ThinkDown => {
            let slots = worker_slots(parent, |_, w| w.traits.think_tokens > 0);
            let (s, w) = pick(rng, &slots)?;
            let t = &mut worker_mut(&mut child, s, w)?.traits;

            t.think_tokens = think_down(t.think_tokens);
        }

        Op::TemperatureShift => {
            let (s, w) = pick(rng, &worker_slots(parent, |_, _| true))?;
            let t = &mut worker_mut(&mut child, s, w)?.traits;
            let up = rng.random_bool(0.5);

            t.temperature = either_way(t.temperature, up, |x, up| {
                let step = if up {
                    TEMPERATURE_STEP
                } else {
                    -TEMPERATURE_STEP
                };

                shifted_temperature(x, step)
            })?;
        }

        Op::SelfCheckFlip => {
            let (s, w) = pick(rng, &worker_slots(parent, |_, _| true))?;
            let t = &mut worker_mut(&mut child, s, w)?.traits;

            t.self_check = !t.self_check;
        }

        Op::AnswerTokensShift => {
            let (s, w) = pick(rng, &worker_slots(parent, |_, _| true))?;
            let t = &mut worker_mut(&mut child, s, w)?.traits;
            let up = rng.random_bool(0.5);

            t.answer_tokens = either_way(t.answer_tokens, up, scaled_answer_tokens)?;
        }

        Op::ModelSwitch => {
            let tags = model_tags(limits);
            let slots = worker_slots(parent, |_, w| tags.iter().any(|tag| *tag != w.model));
            let (s, w) = pick(rng, &slots)?;
            let worker = worker_mut(&mut child, s, w)?;
            let others: Vec<&str> = tags
                .iter()
                .copied()
                .filter(|tag| *tag != worker.model)
                .collect();

            worker.model = pick(rng, &others)?.to_string();
        }

        Op::SectionReplace => {
            let slots = worker_slots(parent, |_, w| !w.sections.is_empty());
            let (s, w) = pick_worker(rng, parent, &slots, steer, steered)?;
            let lens = steer.map(|st| st.lens(worker_index(parent, s, w)));
            let worker = worker_mut(&mut child, s, w)?;
            let i = match lens {
                None => rng.random_range(0..worker.sections.len()),
                Some(lens) => {
                    let weights: Vec<f64> = worker
                        .sections
                        .iter()
                        .map(|x| lens.protect(&x.text))
                        .collect();

                    weighted_index(rng, &weights, steered)?
                }
            };
            let section = worker.sections.get_mut(i)?;

            section.text =
                replacement_text(rng, phrases, &section.name, &section.text, lens, steered)?
                    .to_string();
        }

        Op::SectionAdd => {
            let slots = worker_slots(parent, |_, _| true);
            let (s, w) = pick_worker(rng, parent, &slots, steer, steered)?;
            let lens = steer.map(|st| st.lens(worker_index(parent, s, w)));
            let worker = worker_mut(&mut child, s, w)?;
            let options = add_options(worker, phrases);
            let (name, text) = match lens {
                None => pick(rng, &options)?,
                Some(lens) => {
                    let weights = add_weights(worker, &options, lens);

                    *options.get(weighted_index(rng, &weights, steered)?)?
                }
            };

            set_section(&mut worker.sections, name, text);
        }

        Op::SectionDrop => {
            let slots = worker_slots(parent, |_, w| {
                w.sections.len() > 1 && w.sections.iter().any(droppable)
            });
            let (s, w) = pick(rng, &slots)?;
            let worker = worker_mut(&mut child, s, w)?;
            let candidates: Vec<usize> = worker
                .sections
                .iter()
                .enumerate()
                .filter(|(_, x)| droppable(x))
                .map(|(i, _)| i)
                .collect();
            let i = pick(rng, &candidates)?;

            worker.sections.remove(i);
        }

        Op::SectionSwap => {
            let (s, w) = pick(rng, &worker_slots(parent, |_, w| w.sections.len() > 1))?;
            let worker = worker_mut(&mut child, s, w)?;
            let n = worker.sections.len();
            let i = rng.random_range(0..n);
            let j = (i + rng.random_range(1..n)) % n;

            worker.sections.swap(i, j);
        }

        Op::WorkerAdd => {
            let width = limits.width();
            let open = stage_slots(parent, |st| {
                !st.workers.is_empty() && st.workers.len() < width
            });
            let k = pick(rng, &open)?;
            let stage = child.stages.get_mut(k)?;
            let w = rng.random_range(0..stage.workers.len());
            let mut clone = stage.workers.get(w)?.clone();

            clone.traits.temperature =
                shifted_temperature(clone.traits.temperature, WORKER_ADD_WARMTH);
            stage.workers.push(clone);

            let cloned = map.get(k)?.get(w)?.clone();

            map.get_mut(k)?.push(cloned);
        }

        Op::WorkerDrop => {
            let k = pick(rng, &stage_slots(parent, |st| st.workers.len() > 1))?;
            let stage = child.stages.get_mut(k)?;
            let w = rng.random_range(0..stage.workers.len());

            stage.workers.remove(w);
            map.get_mut(k)?.remove(w);
        }

        Op::StageAdd => {
            if parent.stages.len() >= limits.stages() {
                return None;
            }

            let (s, w) = pick(rng, &worker_slots(parent, |_, _| true))?;
            let mut worker = parent.stages.get(s)?.workers.get(w)?.clone();
            let residual: Vec<&str> = phrases
                .iter()
                .filter(|(name, _)| *name == RESIDUAL)
                .map(|(_, text)| *text)
                .collect();
            let text = pick(rng, &residual)?;
            let has_residual = worker.sections.iter().any(|x| x.name == RESIDUAL);

            if !has_residual && worker.sections.len() >= MAX_SECTIONS {
                return None;
            }

            set_section(&mut worker.sections, RESIDUAL, text);

            let at = rng.random_range(1..=parent.stages.len());
            let copied = Source {
                parent: 0,
                worker: worker_index(parent, s, w),
            };

            child.stages.insert(
                at,
                Stage {
                    workers: vec![worker],
                },
            );
            map.insert(at, vec![vec![copied]]);
        }

        Op::StageDrop => {
            if parent.stages.len() < 2 {
                return None;
            }

            let k = rng.random_range(0..parent.stages.len());

            child.stages.remove(k);
            map.remove(k);
        }

        Op::Seed | Op::StageCrossover | Op::SectionCrossover | Op::LlmMutate | Op::LlmCrossover => {
            return None;
        }
    }

    Some((child, map))
}

/// True when `child` passes `check(limits)` and its id differs from every id in `parents`.
fn acceptable(child: &Cascade, limits: &CascadeLimits, parents: &[String]) -> bool {
    if child.check(limits).is_err() {
        return false;
    }

    !parents.contains(&child.id())
}

/// A uniformly random element of `items`, or None when it is empty.
fn pick<T: Copy>(rng: &mut ChaCha8Rng, items: &[T]) -> Option<T> {
    if items.is_empty() {
        return None;
    }

    items.get(rng.random_range(0..items.len())).copied()
}

/// The index of one of the options whose weights are `weights`, by one draw (arm W), or
/// None without an option. When every weight is the same, the draw is today's uniform one
/// and nothing is counted. Otherwise u is uniform in [0, 1), and the pick is the first
/// option whose summed weight passes u times the total. `steered` counts the choice, and
/// counts it as changed when the pick differs from ⌊u·n⌋, the pick of equal weights.
fn weighted_index(rng: &mut ChaCha8Rng, weights: &[f64], steered: &mut Steered) -> Option<usize> {
    let n = weights.len();
    let first = *weights.first()?;

    if weights.iter().all(|&w| w == first) {
        return Some(rng.random_range(0..n));
    }

    let u: f64 = rng.random();
    let total = weights.iter().fold(0.0, |sum, &w| sum + w);
    let target = u * total;
    let mut summed = 0.0;
    // Rounding may leave the target at the total; then the last option, whose weight is at
    // least ε, takes it.
    let mut chosen = n - 1;

    for (i, &w) in weights.iter().enumerate() {
        summed += w;

        if target < summed {
            chosen = i;
            break;
        }
    }

    let uniform = ((u * n as f64) as usize).min(n - 1);

    steered.choices += 1;

    if chosen != uniform {
        steered.changed += 1;
    }

    Some(chosen)
}

/// The coin of section crossover in arm W: true takes the partner's section, whose weight
/// is `theirs` against `mine`. With equal weights it is today's coin of p = 1/2 and counts
/// nothing. Otherwise u is uniform in [0, 1), and the coin is u < theirs / (mine + theirs);
/// `steered` counts it, and counts it as changed when it differs from u < 1/2.
fn weighted_coin(rng: &mut ChaCha8Rng, mine: f64, theirs: f64, steered: &mut Steered) -> bool {
    if mine == theirs {
        return rng.random_bool(0.5);
    }

    let u: f64 = rng.random();
    let take = u < theirs / (mine + theirs);

    steered.choices += 1;

    if take != (u < 0.5) {
        steered.changed += 1;
    }

    take
}

/// The Worker that SectionReplace or SectionAdd changes, among `slots` of `parent`:
/// uniform without `steer`, otherwise with the weight 1 + L of each Worker's whole prompt
/// under its own preference (choice 1 of spec 4.1).
fn pick_worker(
    rng: &mut ChaCha8Rng,
    parent: &Cascade,
    slots: &[(usize, usize)],
    steer: Option<&Steer<'_>>,
    steered: &mut Steered,
) -> Option<(usize, usize)> {
    let Some(steer) = steer else {
        return pick(rng, slots);
    };
    let weights: Vec<f64> = slots
        .iter()
        .map(|&(s, w)| {
            let prompt = parent
                .stages
                .get(s)
                .and_then(|stage| stage.workers.get(w))
                .map(Worker::system_prompt)
                .unwrap_or_default();

            steer.lens(worker_index(parent, s, w)).favour(&prompt)
        })
        .collect();

    slots.get(weighted_index(rng, &weights, steered)?).copied()
}

/// The phrases SectionAdd may write into `worker`: one whose name a section has, if its
/// text differs, or one of a new name while the Worker has room for another section.
fn add_options<'p>(worker: &Worker, phrases: &[(&'p str, &'p str)]) -> Vec<(&'p str, &'p str)> {
    phrases
        .iter()
        .copied()
        .filter(
            |(name, text)| match worker.sections.iter().find(|x| x.name == *name) {
                Some(existing) => existing.text != *text,
                None => worker.sections.len() < MAX_SECTIONS,
            },
        )
        .collect()
}

/// The weights of the `options` of SectionAdd for `worker` (choice 3 of spec 4.1): 1 + L of
/// the phrase, times the protection 1 − L of the section it would write over, if any; each
/// factor and the product at least ε.
fn add_weights(worker: &Worker, options: &[(&str, &str)], lens: Lens<'_>) -> Vec<f64> {
    options
        .iter()
        .map(|(name, text)| {
            let over = worker
                .sections
                .iter()
                .find(|x| x.name == *name)
                .map_or(1.0, |x| lens.protect(&x.text));

            (lens.favour(text) * over).max(WEIGHT_FLOOR)
        })
        .collect()
}

/// Positions (stage, worker) of the Workers of `c` that satisfy `ok(stage, worker)`.
fn worker_slots(c: &Cascade, ok: impl Fn(usize, &Worker) -> bool) -> Vec<(usize, usize)> {
    let mut out = Vec::new();

    for (s, stage) in c.stages.iter().enumerate() {
        for (w, worker) in stage.workers.iter().enumerate() {
            if ok(s, worker) {
                out.push((s, w));
            }
        }
    }

    out
}

/// Indices of the stages of `c` that satisfy `ok`.
fn stage_slots(c: &Cascade, ok: impl Fn(&Stage) -> bool) -> Vec<usize> {
    c.stages
        .iter()
        .enumerate()
        .filter(|(_, stage)| ok(stage))
        .map(|(k, _)| k)
        .collect()
}

/// The Worker at (stage `s`, index `w`) of `c`.
fn worker_mut(c: &mut Cascade, s: usize, w: usize) -> Option<&mut Worker> {
    c.stages.get_mut(s)?.workers.get_mut(w)
}

/// ThinkUp: double the budget (0 becomes 64), never above `cap`.
fn think_up(think: u32, cap: u32) -> u32 {
    let up = if think == 0 {
        THINK_STEP_MIN
    } else {
        think.saturating_mul(2)
    };

    up.min(cap)
}

/// ThinkDown: halve the budget; a result below 64 becomes 0.
fn think_down(think: u32) -> u32 {
    let half = think / 2;

    if half < THINK_STEP_MIN { 0 } else { half }
}

/// AnswerTokensShift: double (`up`) or halve the budget, inside the allowed range.
fn scaled_answer_tokens(answer: u32, up: bool) -> u32 {
    let scaled = if up {
        answer.saturating_mul(2)
    } else {
        answer / 2
    };

    scaled.clamp(ANSWER_TOKENS_MIN, ANSWER_TOKENS_CEILING)
}

/// `step(current, up)` if it differs from `current`, else `step(current, !up)` if that
/// does, else None: a value at the end of its range moves the other way.
fn either_way<T: PartialEq + Copy>(current: T, up: bool, step: impl Fn(T, bool) -> T) -> Option<T> {
    [up, !up]
        .into_iter()
        .map(|direction| step(current, direction))
        .find(|next| *next != current)
}

/// `t + delta`, clamped into the temperature range and rounded to two decimals, so that the
/// JSON of the child, and with it its id, stays stable.
fn shifted_temperature(t: f32, delta: f32) -> f32 {
    let clamped = clamp_temperature(t + delta);

    (clamped * 100.0).round() / 100.0
}

/// `t` inside the temperature range; NaN becomes the default temperature.
fn clamp_temperature(t: f32) -> f32 {
    if t.is_nan() {
        Traits::default().temperature
    } else {
        t.clamp(TEMPERATURE_MIN, TEMPERATURE_MAX)
    }
}

/// The distinct model tags of the allow-list, in order.
fn model_tags(limits: &CascadeLimits) -> Vec<&str> {
    let mut tags: Vec<&str> = Vec::new();

    for spec in &limits.models {
        if !tags.contains(&spec.tag.as_str()) {
            tags.push(&spec.tag);
        }
    }

    tags
}

/// A text of `phrases` other than `current`: with p = 0.7 one of the phrases named `name`
/// (when there is such a phrase), otherwise one of the whole pool. Within the pool that is
/// drawn, each text weighs alike, or with `lens` 1 + L of the text (arm W, choice 3).
fn replacement_text<'p>(
    rng: &mut ChaCha8Rng,
    phrases: &[(&'p str, &'p str)],
    name: &str,
    current: &str,
    lens: Option<Lens<'_>>,
    steered: &mut Steered,
) -> Option<&'p str> {
    let same_name: Vec<&'p str> = phrases
        .iter()
        .filter(|(n, t)| *n == name && *t != current)
        .map(|(_, t)| *t)
        .collect();

    if !same_name.is_empty() && rng.random_bool(SAME_NAME_P) {
        return pick_text(rng, &same_name, lens, steered);
    }

    let any: Vec<&'p str> = phrases
        .iter()
        .filter(|(_, t)| *t != current)
        .map(|(_, t)| *t)
        .collect();

    pick_text(rng, &any, lens, steered)
}

/// One of `texts`: uniform without `lens`, otherwise with the weight 1 + L of each text.
fn pick_text<'p>(
    rng: &mut ChaCha8Rng,
    texts: &[&'p str],
    lens: Option<Lens<'_>>,
    steered: &mut Steered,
) -> Option<&'p str> {
    let Some(lens) = lens else {
        return pick(rng, texts);
    };
    let weights: Vec<f64> = texts.iter().map(|t| lens.favour(t)).collect();

    texts.get(weighted_index(rng, &weights, steered)?).copied()
}

/// Write `text` over the first section named `name`, or append a new section.
fn set_section(sections: &mut Vec<Section>, name: &str, text: &str) {
    match sections.iter_mut().find(|s| s.name == name) {
        Some(existing) => existing.text = text.to_string(),
        None => sections.push(Section {
            name: name.to_string(),
            text: text.to_string(),
        }),
    }
}

/// True when SectionDrop may remove `s`: every section except "role" and "format".
fn droppable(s: &Section) -> bool {
    let name = s.name.trim();

    !PROTECTED_SECTIONS
        .iter()
        .any(|protected| name.eq_ignore_ascii_case(protected))
}

/// `b`'s Worker paired with Worker (s, w) of the other parent: the same position, or the
/// last stage or Worker that `b` has.
fn partner(b: &Cascade, s: usize, w: usize) -> Option<&Worker> {
    let (ps, pw) = partner_at(b, s, w)?;

    b.stages.get(ps)?.workers.get(pw)
}

/// The position (stage, Worker) of `partner(b, s, w)`, or None when `b` has no Worker there.
fn partner_at(b: &Cascade, s: usize, w: usize) -> Option<(usize, usize)> {
    let ps = s.min(b.stages.len().checked_sub(1)?);
    let pw = w.min(b.stages.get(ps)?.workers.len().checked_sub(1)?);

    Some((ps, pw))
}

/// Uniform crossover of one Worker with its partner, in place: every section whose name the
/// partner also has, the model and each Trait field come from the partner with p = 0.5.
/// The Soul and the Coupling strength are no genes and are never crossed (spec 9.1): the
/// operators see genes only, and Souletics gives the child its Souls.
///
/// With `lens` (arm W), the coin of each section takes the partner's text with the
/// probability (1 + L(theirs)) / ((1 + L(mine)) + (1 + L(theirs))), each weight at least ε;
/// `steered` counts the coins the weights decided. The other coins stay as they are.
fn cross_worker(
    child: &mut Worker,
    other: &Worker,
    lens: Option<Lens<'_>>,
    steered: &mut Steered,
    rng: &mut ChaCha8Rng,
) {
    for section in &mut child.sections {
        if let Some(theirs) = other.sections.iter().find(|s| s.name == section.name)
            && match lens {
                None => rng.random_bool(0.5),
                Some(lens) => weighted_coin(
                    rng,
                    lens.favour(&section.text),
                    lens.favour(&theirs.text),
                    steered,
                ),
            }
        {
            section.text = theirs.text.clone();
        }
    }

    if rng.random_bool(0.5) {
        child.model = other.model.clone();
    }

    let (mine, theirs) = (&mut child.traits, &other.traits);

    if rng.random_bool(0.5) {
        mine.think_tokens = theirs.think_tokens;
    }

    if rng.random_bool(0.5) {
        mine.temperature = theirs.temperature;
    }

    if rng.random_bool(0.5) {
        mine.answer_tokens = theirs.answer_tokens;
    }

    if rng.random_bool(0.5) {
        mine.self_check = theirs.self_check;
    }
}

/// A request to the LLM mutator with the fixed generation settings.
fn llm_request(brief: &MutatorBrief, prompt: String, seed: u64) -> GenRequest {
    GenRequest {
        system: brief.system.to_string(),
        prompt,
        max_tokens: LLM_MAX_TOKENS,
        temperature: LLM_TEMPERATURE,
        seed,
        think_tokens: 0,
        phases: None,
    }
}

/// The user prompt: context, format, limits, each parent as pretty JSON with its public
/// search fitness, then the task and the rules. A parent shows its genes only: the Mutator
/// never sees a Soul or a Coupling strength (step 2, spec 4.5), so the format keeps its
/// four Traits, and a parent with a Soul gives the request of the same parent without.
fn llm_prompt(
    brief: &MutatorBrief,
    parents: &[(&str, &Cascade, f64)],
    limits: &CascadeLimits,
    task: &str,
) -> String {
    let mut p = String::new();

    p.push_str(brief.context);
    p.push_str(brief.format);
    p.push_str(&describe_limits(limits));
    p.push('\n');

    for (label, cascade, fitness) in parents {
        let json = serde_json::to_string_pretty(&cascade.genes()).unwrap_or_default();

        p.push_str(&format!("{label}:\n{json}\n{}\n\n", fitness_line(*fitness)));
    }

    p.push_str(task);
    p.push_str(LLM_RULES);

    p
}

/// The run's limits in words: models with their cost per call, depth, width, the thinking
/// ceiling of every stage, and the ranges of the other Traits and of the prompt sections.
fn describe_limits(limits: &CascadeLimits) -> String {
    let models: Vec<String> = limits
        .models
        .iter()
        .map(|m| format!("\"{}\" (costs {} fitness per call)", m.tag, m.cost_per_call))
        .collect();
    let ceilings: Vec<String> = (0..limits.stages())
        .map(|k| format!("stage {k}: {}", limits.think_ceiling(k)))
        .collect();

    let mut s = String::from("Limits of this run:\n");

    s.push_str(&format!("- model: one of {}\n", models.join(", ")));
    s.push_str(&format!(
        "- stages: 1 to {}; Workers per stage: 1 to {}\n",
        limits.stages(),
        limits.width()
    ));
    s.push_str(&format!(
        "- think_tokens: 0 up to the ceiling of the Worker's stage ({})\n",
        ceilings.join(", ")
    ));
    s.push_str(&format!(
        "- answer_tokens: {ANSWER_TOKENS_MIN} to {ANSWER_TOKENS_CEILING}\n"
    ));
    s.push_str(&format!(
        "- temperature: {TEMPERATURE_MIN} to {TEMPERATURE_MAX}\n"
    ));
    s.push_str(&format!(
        "- sections per Worker: 1 to {MAX_SECTIONS}; section names: 1 to \
         {MAX_SECTION_NAME_CHARS} characters; section texts: 1 to {MAX_SECTION_CHARS} \
         characters\n"
    ));

    s
}

/// "public search fitness: 0.1234", or "not measured" for a value that is not finite.
fn fitness_line(fitness: f64) -> String {
    if fitness.is_finite() {
        format!("public search fitness: {fitness:.4}")
    } else {
        "public search fitness: not measured".to_string()
    }
}

/// Byte index of the `}` that closes the `{` at `start`, ignoring braces inside JSON
/// strings, or None when the object is not closed. Only ASCII bytes are compared, and those
/// never occur inside a multi-byte UTF-8 character, so the index is a char boundary.
fn closing_brace(text: &str, start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (i, &b) in text.as_bytes().iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }

            continue;
        }

        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth = depth.saturating_sub(1);

                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use rand::SeedableRng;

    use super::*;
    use crate::cascade::circuit_genes::{CIRCUIT_PHRASES, circuit_seed_cascades};
    use crate::cascade::designer::named_families;
    use crate::cascade::genome::{
        COUPLING_MAX, ModelSpec, PhaseState, THINK_TOKENS_CEILING, seed_cascades,
    };
    use crate::cascade::soul::{QUARTER_TURN, Readout, preference};

    /// Every Op, in declaration order.
    const ALL_OPS: [Op; 19] = [
        Op::Seed,
        Op::SectionReplace,
        Op::SectionAdd,
        Op::SectionDrop,
        Op::SectionSwap,
        Op::ThinkUp,
        Op::ThinkDown,
        Op::TemperatureShift,
        Op::SelfCheckFlip,
        Op::AnswerTokensShift,
        Op::ModelSwitch,
        Op::WorkerAdd,
        Op::WorkerDrop,
        Op::StageAdd,
        Op::StageDrop,
        Op::StageCrossover,
        Op::SectionCrossover,
        Op::LlmMutate,
        Op::LlmCrossover,
    ];

    /// Ops only the lineage records; `mutate` never returns them.
    const NOT_MUTATIONS: [Op; 5] = [
        Op::Seed,
        Op::StageCrossover,
        Op::SectionCrossover,
        Op::LlmMutate,
        Op::LlmCrossover,
    ];

    /// The kinds of grid rule the phrases talk about, as lower-case word stems. The last
    /// family is the residual review of an earlier stage's answer.
    const FAMILIES: [&[&str]; 7] = [
        &["mirror", "flip", "spiegel"],
        &["rotat", "dreh"],
        &["transpos", "diagonal"],
        &["gravity", "fall", "schwerkraft"],
        &["colour", "color", "farb"],
        &["combine", "compose", "two steps", "kombin"],
        &["previous answer", "earlier answer", "vorige antwort"],
    ];

    /// Limits whose late stages may think less: stage 1 a little, stage 2 not at all.
    fn staged_limits() -> CascadeLimits {
        CascadeLimits {
            think_ceiling_per_stage: vec![4096, 512, 0],
            ..CascadeLimits::default()
        }
    }

    /// The four seed Cascades, without their names.
    fn seeds() -> Vec<Cascade> {
        seed_cascades().into_iter().map(|(_, c)| c).collect()
    }

    /// A fresh rng for `seed`.
    fn rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }

    /// One operator with the grid phrase pool, as the operator tests below call it.
    fn apply(
        parent: &Cascade,
        op: Op,
        limits: &CascadeLimits,
        rng: &mut ChaCha8Rng,
    ) -> Option<Cascade> {
        apply_with(parent, op, limits, PHRASES, rng)
    }

    /// The first seed with its one Worker's Traits replaced.
    fn single(traits: Traits) -> Cascade {
        let mut c = seeds().remove(0);

        c.stages[0].workers[0].traits = traits;

        c
    }

    /// Seed `k % 4` after `steps` mutations, so that parents differ in more than one gene.
    fn evolved(k: usize, steps: usize, limits: &CascadeLimits, rng: &mut ChaCha8Rng) -> Cascade {
        let seeds = seeds();
        let mut c = seeds[k % seeds.len()].clone();

        for _ in 0..steps {
            if let Some((child, _)) = mutate(&c, limits, rng) {
                c = child;
            }
        }

        c
    }

    /// Panics unless every Worker of `c` stays within its stage's thinking ceiling.
    fn assert_within_ceilings(c: &Cascade, limits: &CascadeLimits) {
        for (k, stage) in c.stages.iter().enumerate() {
            for worker in &stage.workers {
                assert!(
                    worker.traits.think_tokens <= limits.think_ceiling(k),
                    "stage {k} thinks {} tokens, ceiling {}",
                    worker.traits.think_tokens,
                    limits.think_ceiling(k)
                );
            }
        }
    }

    /// Panics unless `child` is valid and differs from both parents.
    fn assert_valid_and_new(child: &Cascade, a: &Cascade, b: &Cascade, limits: &CascadeLimits) {
        assert!(child.check(limits).is_ok(), "{:?}", child.check(limits));
        assert_ne!(child.id(), a.id(), "child equals parent a");
        assert_ne!(child.id(), b.id(), "child equals parent b");
        assert_within_ceilings(child, limits);
    }

    /// The section names of every Worker, stage by stage.
    fn shape(c: &Cascade) -> Vec<Vec<Vec<String>>> {
        c.stages
            .iter()
            .map(|stage| {
                stage
                    .workers
                    .iter()
                    .map(|w| w.sections.iter().map(|s| s.name.clone()).collect())
                    .collect()
            })
            .collect()
    }

    /// Indices of the FAMILIES that `text` uses.
    fn families_of(text: &str) -> Vec<usize> {
        let lower = text.to_lowercase();

        FAMILIES
            .iter()
            .enumerate()
            .filter(|(_, stems)| stems.iter().any(|stem| lower.contains(stem)))
            .map(|(k, _)| k)
            .collect()
    }

    #[test]
    fn mutation_walks_stay_valid_new_and_use_every_operator() {
        let limits = staged_limits();
        let seeds = seeds();
        let mut counts: HashMap<Op, usize> = HashMap::new();
        let mut none = 0;

        for seed in 0..300u64 {
            let mut rng = rng(seed);
            let mut parent = seeds[seed as usize % seeds.len()].clone();

            for _ in 0..12 {
                let Some((child, op)) = mutate(&parent, &limits, &mut rng) else {
                    none += 1;
                    continue;
                };

                assert!(
                    child.check(&limits).is_ok(),
                    "{op:?}: {:?}",
                    child.check(&limits)
                );
                assert_ne!(child.id(), parent.id(), "{op:?} returned the parent");
                assert_within_ceilings(&child, &limits);

                *counts.entry(op).or_default() += 1;
                parent = child;
            }
        }

        assert_eq!(none, 0, "a valid parent always has some mutation");

        for op in ALL_OPS {
            let n = counts.get(&op).copied().unwrap_or(0);

            if NOT_MUTATIONS.contains(&op) {
                assert_eq!(n, 0, "{op:?} is not a mutation");
            } else {
                assert!(n > 0, "{op:?} never occurred: {counts:?}");
            }
        }
    }

    #[test]
    fn mutate_is_deterministic_and_seed_sensitive() {
        let limits = staged_limits();
        let seeds = seeds();
        let mut ids = BTreeSet::new();

        for seed in 0..300u64 {
            let parent = &seeds[seed as usize % seeds.len()];
            let first = mutate(parent, &limits, &mut rng(seed));
            let again = mutate(parent, &limits, &mut rng(seed));

            assert_eq!(first, again, "seed {seed}");

            if let Some((child, _)) = first {
                ids.insert(child.id());
            }
        }

        // Different seeds explore: 300 draws from four parents give many distinct children.
        assert!(ids.len() > 100, "only {} distinct children", ids.len());
    }

    #[test]
    fn operator_weights_split_forty_forty_twenty() {
        let trait_ops = [
            Op::ThinkUp,
            Op::ThinkDown,
            Op::TemperatureShift,
            Op::SelfCheckFlip,
            Op::AnswerTokensShift,
            Op::ModelSwitch,
        ];
        let section_ops = [
            Op::SectionReplace,
            Op::SectionAdd,
            Op::SectionDrop,
            Op::SectionSwap,
        ];
        let weight = |ops: &[Op]| -> u32 {
            MUTATION_WEIGHTS
                .iter()
                .filter(|(op, _)| ops.contains(op))
                .map(|(_, w)| w)
                .sum()
        };
        let total: u32 = MUTATION_WEIGHTS.iter().map(|(_, w)| w).sum();

        assert_eq!(total, 100);
        assert_eq!(weight(&trait_ops), 40);
        assert_eq!(weight(&section_ops), 40);

        // The draw follows the weights (binomial sd about 0.0015 at n = 100 000).
        let mut rng = rng(11);
        let n = 100_000;
        let mut trait_draws = 0;
        let mut section_draws = 0;

        for _ in 0..n {
            let op = draw_op(&mut rng);

            if trait_ops.contains(&op) {
                trait_draws += 1;
            } else if section_ops.contains(&op) {
                section_draws += 1;
            }
        }

        let share = |k: usize| k as f64 / n as f64;

        assert!(
            (share(trait_draws) - 0.4).abs() < 0.01,
            "{}",
            share(trait_draws)
        );
        assert!(
            (share(section_draws) - 0.4).abs() < 0.01,
            "{}",
            share(section_draws)
        );
    }

    #[test]
    fn think_operators_double_halve_and_respect_the_stage_ceiling() {
        assert_eq!(think_up(0, 4096), 64);
        assert_eq!(think_up(256, 4096), 512);
        assert_eq!(think_up(3000, 4096), 4096);
        assert_eq!(think_up(0, 32), 32);
        assert_eq!(think_down(64), 0);
        assert_eq!(think_down(100), 0);
        assert_eq!(think_down(128), 64);
        assert_eq!(think_down(4096), 2048);

        let limits = CascadeLimits::default();
        let at_ceiling = single(Traits {
            think_tokens: THINK_TOKENS_CEILING,
            ..Traits::default()
        });
        let silent = single(Traits {
            think_tokens: 0,
            ..Traits::default()
        });

        assert!(apply(&at_ceiling, Op::ThinkUp, &limits, &mut rng(1)).is_none());
        assert!(apply(&silent, Op::ThinkDown, &limits, &mut rng(1)).is_none());

        let up = apply(&silent, Op::ThinkUp, &limits, &mut rng(1)).expect("0 can grow");

        assert_eq!(up.stages[0].workers[0].traits.think_tokens, 64);

        let down = apply(&at_ceiling, Op::ThinkDown, &limits, &mut rng(1)).expect("can shrink");

        assert_eq!(down.stages[0].workers[0].traits.think_tokens, 2048);

        // A stage that may not think at all offers nothing to ThinkUp.
        let mute = CascadeLimits {
            think_ceiling_per_stage: vec![0],
            ..CascadeLimits::default()
        };

        assert!(apply(&silent, Op::ThinkUp, &mute, &mut rng(1)).is_none());
    }

    #[test]
    fn temperature_and_answer_shifts_stay_in_range() {
        let limits = CascadeLimits::default();
        let temperature = |t: f32, seed: u64| {
            let c = single(Traits {
                temperature: t,
                ..Traits::default()
            });
            let child = apply(&c, Op::TemperatureShift, &limits, &mut rng(seed)).expect("shift");

            child.stages[0].workers[0].traits.temperature
        };
        let answer = |a: u32, seed: u64| {
            let c = single(Traits {
                answer_tokens: a,
                ..Traits::default()
            });
            let child = apply(&c, Op::AnswerTokensShift, &limits, &mut rng(seed)).expect("shift");

            child.stages[0].workers[0].traits.answer_tokens
        };
        let mut seen = BTreeSet::new();

        for seed in 0..40 {
            assert_eq!(
                temperature(TEMPERATURE_MIN, seed),
                0.2,
                "the floor only moves up"
            );
            assert_eq!(
                temperature(TEMPERATURE_MAX, seed),
                1.3,
                "the top only moves down"
            );

            // Rounded to two decimals, so the JSON (and the id) carries no float noise.
            let t = temperature(0.33, seed);
            let json = serde_json::to_string(&t).expect("encode");

            assert!(json == "0.53" || json == "0.13", "{json}");
            seen.insert(json);

            assert_eq!(answer(ANSWER_TOKENS_CEILING, seed), 1024);
            assert_eq!(answer(ANSWER_TOKENS_MIN, seed), 128);
            assert!([256, 1024].contains(&answer(512, seed)));
        }

        assert_eq!(seen.len(), 2, "both directions occur");
    }

    #[test]
    fn model_switch_picks_another_allowed_model() {
        let limits = CascadeLimits::default();
        let parent = seeds().remove(0);
        let child = apply(&parent, Op::ModelSwitch, &limits, &mut rng(3)).expect("two models");

        assert_eq!(child.stages[0].workers[0].model, "sim:large");

        let one = CascadeLimits {
            models: vec![ModelSpec {
                tag: "sim:small".to_string(),
                cost_per_call: 0.001,
            }],
            ..CascadeLimits::default()
        };

        assert!(apply(&parent, Op::ModelSwitch, &one, &mut rng(3)).is_none());
    }

    #[test]
    fn section_operators_follow_their_rules() {
        let limits = CascadeLimits::default();
        let parent = seeds().remove(0);
        let names = |c: &Cascade| -> Vec<String> {
            c.stages[0].workers[0]
                .sections
                .iter()
                .map(|s| s.name.clone())
                .collect()
        };

        // SectionDrop takes the strategy of a seed and never its role or format.
        for seed in 0..50 {
            let child = apply(&parent, Op::SectionDrop, &limits, &mut rng(seed)).expect("drop");

            assert_eq!(names(&child), ["role", "format"]);
        }

        let mut bare = parent.clone();

        bare.stages[0].workers[0]
            .sections
            .retain(|s| s.name != "strategy");

        assert!(apply(&bare, Op::SectionDrop, &limits, &mut rng(1)).is_none());

        // SectionSwap keeps the sections and changes their order.
        for seed in 0..50 {
            let child = apply(&parent, Op::SectionSwap, &limits, &mut rng(seed)).expect("swap");
            let mut after = names(&child);
            let mut before = names(&parent);

            assert_ne!(after, before);

            after.sort();
            before.sort();

            assert_eq!(after, before);
        }

        // SectionReplace keeps the name and prefers phrases of that name (p = 0.7, plus their
        // share of the whole pool).
        let mut r = rng(5);
        let n = 2000;
        let mut same_name = 0;

        for _ in 0..n {
            let child = apply(&parent, Op::SectionReplace, &limits, &mut r).expect("replace");
            let before = &parent.stages[0].workers[0].sections;
            let after = &child.stages[0].workers[0].sections;
            let (old, new) = before
                .iter()
                .zip(after)
                .find(|(x, y)| x != y)
                .expect("one section changed");

            assert_eq!(old.name, new.name);
            assert!(PHRASES.iter().any(|(_, t)| *t == new.text));

            if PHRASES
                .iter()
                .any(|(m, t)| *m == new.name && *t == new.text)
            {
                same_name += 1;
            }
        }

        let share = same_name as f64 / n as f64;

        assert!(share > 0.7, "same-name share {share}");

        // SectionAdd appends a new section or rewrites the one of the same name.
        for seed in 0..50 {
            let child = apply(&parent, Op::SectionAdd, &limits, &mut rng(seed)).expect("add");
            let before = &parent.stages[0].workers[0].sections;
            let after = &child.stages[0].workers[0].sections;

            if after.len() == before.len() + 1 {
                assert_eq!(&after[..before.len()], &before[..]);
            } else {
                assert_eq!(names(&child), names(&parent));
            }

            assert!(
                after
                    .iter()
                    .any(|s| PHRASES.iter().any(|(m, t)| *m == s.name && *t == s.text))
            );
        }

        // A Worker with MAX_SECTIONS sections of other names takes no more.
        let mut full = parent.clone();

        full.stages[0].workers[0].sections = (0..MAX_SECTIONS)
            .map(|i| Section {
                name: format!("part{i}"),
                text: "Keep going.".to_string(),
            })
            .collect();

        assert!(apply(&full, Op::SectionAdd, &limits, &mut rng(1)).is_none());
    }

    #[test]
    fn worker_and_stage_operators_follow_their_rules() {
        let limits = staged_limits();
        let parent = seeds().remove(0);

        // WorkerAdd clones a Worker into its own stage, 0.3 warmer.
        let child = apply(&parent, Op::WorkerAdd, &limits, &mut rng(1)).expect("room");
        let workers = &child.stages[0].workers;

        assert_eq!(workers.len(), 2);
        assert_eq!(workers[1].traits.temperature, 0.5);
        assert_eq!(workers[1].sections, workers[0].sections);

        // A full stage takes no more Workers, and a lone Worker is never dropped.
        let mut wide = parent.clone();

        wide.stages[0].workers = vec![parent.stages[0].workers[0].clone(); limits.width()];

        assert!(apply(&wide, Op::WorkerAdd, &limits, &mut rng(1)).is_none());
        assert!(apply(&parent, Op::WorkerDrop, &limits, &mut rng(1)).is_none());

        let narrower = apply(&wide, Op::WorkerDrop, &limits, &mut rng(1)).expect("drop");

        assert_eq!(narrower.stages[0].workers.len(), limits.width() - 1);

        // StageAdd puts a residual checker after the first stage.
        for seed in 0..50 {
            let mut child = apply(&parent, Op::StageAdd, &limits, &mut rng(seed)).expect("room");

            clamp_to_limits(&mut child, &limits);

            assert_eq!(child.stages.len(), 2);
            assert_eq!(child.stages[0], parent.stages[0]);
            assert_eq!(child.stages[1].workers.len(), 1);

            let residual = child.stages[1].workers[0]
                .sections
                .iter()
                .find(|s| s.name == RESIDUAL)
                .expect("a residual section");

            assert!(
                PHRASES
                    .iter()
                    .any(|(m, t)| *m == RESIDUAL && *t == residual.text)
            );
            assert!(child.check(&limits).is_ok());
        }

        // A full-depth Cascade takes no more stages, and a lone stage is never dropped.
        let mut deep = parent.clone();

        deep.stages = vec![parent.stages[0].clone(); limits.stages()];

        assert!(apply(&deep, Op::StageAdd, &limits, &mut rng(1)).is_none());
        assert!(apply(&parent, Op::StageDrop, &limits, &mut rng(1)).is_none());

        let shallower = apply(&deep, Op::StageDrop, &limits, &mut rng(1)).expect("drop");

        assert_eq!(shallower.stages.len(), limits.stages() - 1);
    }

    #[test]
    fn clamp_to_limits_repairs_an_over_limit_cascade() {
        let limits = staged_limits();
        let seed = seeds().remove(0);
        let mut hot = seed.stages[0].workers[0].clone();

        hot.traits = Traits {
            think_tokens: 1_000_000,
            temperature: 9.0,
            answer_tokens: 1,
            self_check: true,
            ..Traits::default()
        };

        let mut odd = hot.clone();

        odd.traits.temperature = f32::NAN;
        odd.traits.answer_tokens = 1_000_000;

        let mut cold = hot.clone();

        cold.traits.temperature = -3.0;

        let stage = Stage {
            workers: vec![hot.clone(), odd, cold, hot.clone(), hot],
        };
        let mut c = Cascade {
            stages: vec![stage; 7],
        };

        assert!(c.check(&limits).is_err());

        clamp_to_limits(&mut c, &limits);

        assert!(c.check(&limits).is_ok(), "{:?}", c.check(&limits));
        assert_eq!(c.stages.len(), limits.stages());
        assert!(c.stages.iter().all(|s| s.workers.len() == limits.width()));
        assert!(
            c.stages[2]
                .workers
                .iter()
                .all(|w| w.traits.think_tokens == 0)
        );
        assert_within_ceilings(&c, &limits);

        // A valid Cascade stays as it is.
        let mut same = seed.clone();

        clamp_to_limits(&mut same, &limits);

        assert_eq!(same, seed);
    }

    #[test]
    fn degenerate_parents_never_panic() {
        let limits = staged_limits();
        let empty = Cascade { stages: Vec::new() };
        let hollow = Cascade {
            stages: vec![Stage {
                workers: Vec::new(),
            }],
        };
        let mut odd = seeds().remove(1);

        odd.stages[0].workers[0].sections.clear();
        odd.stages[0].workers[0].traits = Traits {
            think_tokens: u32::MAX,
            temperature: f32::NAN,
            answer_tokens: u32::MAX,
            self_check: false,
            ..Traits::default()
        };

        for seed in 0..100 {
            let mut r = rng(seed);

            assert!(mutate(&empty, &limits, &mut r).is_none());
            assert!(mutate(&hollow, &limits, &mut r).is_none());

            if let Some((child, op)) = mutate(&odd, &limits, &mut r) {
                assert!(child.check(&limits).is_ok(), "{op:?}");
            }

            let pairs = [
                (&empty, &odd),
                (&odd, &empty),
                (&hollow, &odd),
                (&odd, &hollow),
                (&hollow, &hollow),
            ];

            for (a, b) in pairs {
                let children = [
                    stage_crossover(a, b, &limits, &mut r),
                    section_crossover(a, b, &limits, &mut r),
                ];

                for child in children.into_iter().flatten() {
                    assert!(child.check(&limits).is_ok());
                }
            }
        }

        // Depth and width 1 still mutate; a stage crossover can only copy a parent.
        let tiny = CascadeLimits {
            max_stages: 1,
            max_width: 1,
            ..CascadeLimits::default()
        };
        let seeds = seeds();

        for seed in 0..50 {
            let mut r = rng(seed);
            let (child, op) = mutate(&seeds[0], &tiny, &mut r).expect("Traits and sections");

            assert!(child.check(&tiny).is_ok(), "{op:?}");
            assert!(stage_crossover(&seeds[0], &seeds[1], &tiny, &mut r).is_none());
        }
    }

    #[test]
    fn crossovers_give_valid_children_that_differ_from_both_parents() {
        let limits = staged_limits();
        let runs = 300u64;
        let mut stage_some = 0usize;
        let mut section_some = 0usize;

        for seed in 0..runs {
            let mut r = rng(seed);
            let a = evolved(seed as usize, 3, &limits, &mut r);
            let b = evolved(seed as usize / 4, 3, &limits, &mut r);

            if let Some(child) = stage_crossover(&a, &b, &limits, &mut r) {
                assert_valid_and_new(&child, &a, &b, &limits);
                assert_eq!(child.stages[0], a.stages[0], "starts with a's first stage");

                // It ends with b's last stage (its thinking may be clamped).
                let last = child.stages.last().expect("non-empty");
                let b_last = b.stages.last().expect("non-empty");

                assert_eq!(last.workers.len(), b_last.workers.len());

                for (x, y) in last.workers.iter().zip(&b_last.workers) {
                    assert_eq!((&x.model, &x.sections), (&y.model, &y.sections));
                }

                stage_some += 1;
            }

            if let Some(child) = section_crossover(&a, &b, &limits, &mut r) {
                assert_valid_and_new(&child, &a, &b, &limits);
                assert_eq!(shape(&child), shape(&a), "keeps a's shape");

                section_some += 1;
            }
        }

        // The plain seeds: one Worker each, differing in one or two genes only.
        let seeds = seeds();
        let mut raw_runs = 0usize;
        let mut raw_stage = 0usize;
        let mut raw_section = 0usize;

        for (i, a) in seeds.iter().enumerate() {
            for (j, b) in seeds.iter().enumerate() {
                for s in 0..10u64 {
                    let mut r = rng(1000 + 100 * (i * 4 + j) as u64 + s);

                    raw_runs += 1;

                    if let Some(child) = stage_crossover(a, b, &limits, &mut r) {
                        assert_valid_and_new(&child, a, b, &limits);
                        raw_stage += 1;
                    }

                    if let Some(child) = section_crossover(a, b, &limits, &mut r) {
                        assert_valid_and_new(&child, a, b, &limits);
                        raw_section += 1;
                    }
                }
            }
        }

        let frac = |k: usize, n: usize| k as f64 / n as f64;
        let stage_frac = frac(stage_some, runs as usize);
        let section_frac = frac(section_some, runs as usize);

        eprintln!(
            "stage crossover: {stage_frac:.3} of evolved pairs, {:.3} of seed pairs",
            frac(raw_stage, raw_runs)
        );
        eprintln!(
            "section crossover: {section_frac:.3} of evolved pairs, {:.3} of seed pairs",
            frac(raw_section, raw_runs)
        );

        // Measured at 1.000 and 0.993 for evolved pairs.
        assert!(stage_frac >= 0.95, "stage crossover {stage_frac}");
        assert!(section_frac >= 0.9, "section crossover {section_frac}");

        // Two seeds always stack into two stages. A section crossover of two seeds can only
        // copy a parent unless they differ in two genes: the strategy and the self-check of
        // the colour seed, so 6 of the 16 ordered pairs (measured 0.375).
        assert_eq!(raw_stage, raw_runs);
        assert!(
            frac(raw_section, raw_runs) >= 0.3,
            "{raw_section} of {raw_runs}"
        );
    }

    #[test]
    fn operators_hold_under_other_limits() {
        let single_model = CascadeLimits {
            models: vec![ModelSpec {
                tag: "sim:small".to_string(),
                cost_per_call: 0.001,
            }],
            ..CascadeLimits::default()
        };
        let variants = [
            CascadeLimits::default(),
            CascadeLimits {
                max_stages: 2,
                max_width: 1,
                ..CascadeLimits::default()
            },
            CascadeLimits {
                think_ceiling_per_stage: vec![64, 32, 16, 8],
                ..CascadeLimits::default()
            },
            // The seeds think 256 tokens, so here every parent starts over a ceiling.
            CascadeLimits {
                think_ceiling_per_stage: vec![0, 0, 0, 0],
                ..CascadeLimits::default()
            },
            single_model,
        ];

        for (v, limits) in variants.iter().enumerate() {
            for seed in 0..60u64 {
                let mut r = rng(seed);
                let mut parent = seeds()[seed as usize % 4].clone();
                let other = evolved(seed as usize + 1, 2, limits, &mut r);

                for _ in 0..8 {
                    let (child, op) = mutate(&parent, limits, &mut r).expect("a mutation");

                    assert!(child.check(limits).is_ok(), "limits {v}, {op:?}");
                    assert_ne!(child.id(), parent.id(), "limits {v}, {op:?}");

                    for cross in [
                        stage_crossover(&child, &other, limits, &mut r),
                        section_crossover(&child, &other, limits, &mut r),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        assert_valid_and_new(&cross, &child, &other, limits);
                    }

                    parent = child;
                }
            }
        }
    }

    #[test]
    fn crossovers_are_deterministic() {
        let limits = staged_limits();

        for seed in 0..50u64 {
            let mut r = rng(seed);
            let a = evolved(seed as usize, 2, &limits, &mut r);
            let b = evolved(seed as usize + 1, 2, &limits, &mut r);

            assert_eq!(
                stage_crossover(&a, &b, &limits, &mut rng(seed)),
                stage_crossover(&a, &b, &limits, &mut rng(seed))
            );
            assert_eq!(
                section_crossover(&a, &b, &limits, &mut rng(seed)),
                section_crossover(&a, &b, &limits, &mut rng(seed))
            );
        }
    }

    #[test]
    fn parse_accepts_fenced_json_with_prose_around_it() {
        let limits = CascadeLimits::default();
        let c = seeds().remove(2);
        let json = serde_json::to_string_pretty(&c).expect("encode");
        let fenced = format!(
            "Here is the new Cascade.\n```json\n{json}\n```\nI kept the {{format}} section."
        );

        assert_eq!(parse_llm_cascade(&fenced, &limits).expect("fenced"), c);
        assert_eq!(parse_llm_cascade(&json, &limits).expect("bare"), c);

        // A brace inside a JSON string does not end the object early.
        let mut tricky = c.clone();

        tricky.stages[0].workers[0].sections[1].text =
            "Mark the {corner} cells \"first\".".to_string();

        let answer = format!("Sure:\n{}", serde_json::to_string(&tricky).expect("encode"));

        assert_eq!(
            parse_llm_cascade(&answer, &limits).expect("braces in a string"),
            tricky
        );
    }

    #[test]
    fn parse_rejects_bad_answers() {
        let limits = CascadeLimits::default();
        let c = seeds().remove(0);
        let json = |c: &Cascade| serde_json::to_string_pretty(c).expect("encode");
        let rejects = |text: &str, why: &str| match parse_llm_cascade(text, &limits) {
            Err(BrutError::Invalid(reason)) => assert!(!reason.is_empty(), "{why}"),
            other => panic!("{why}: expected Invalid, got {other:?}"),
        };

        rejects("I would rather not.", "no JSON at all");
        rejects(
            "```json\n{\"stages\": [\n```",
            "an object that is not closed",
        );
        rejects("{\"stages\": [}", "invalid JSON");
        rejects("{\"stages\": []}", "no stages");

        let mut unknown = c.clone();

        unknown.stages[0].workers[0].model = "gpt-unknown".to_string();
        rejects(&json(&unknown), "unknown model");

        let mut leaky = c.clone();

        leaky.stages[0].workers[0].sections[1].text = "Use the key sk-abc123 here.".to_string();
        rejects(&json(&leaky), "credential look-alike");

        let mut greedy = c.clone();

        greedy.stages[0].workers[0].traits.think_tokens = THINK_TOKENS_CEILING + 1;
        rejects(&json(&greedy), "over the ceiling: nothing is clamped");

        let extra = json(&c).replacen("\"stages\"", "\"notes\": \"try harder\",\n  \"stages\"", 1);

        rejects(&extra, "unknown field");

        // Every reason says where it came from.
        match parse_llm_cascade(&json(&unknown), &limits) {
            Err(BrutError::Invalid(reason)) => {
                assert!(reason.starts_with("llm answer: "), "{reason}")
            }
            other => panic!("expected Invalid, got {other:?}"),
        }

        // An answer cut off before its object closes is an error at every length, never a
        // panic; once the object is closed, the missing end of the fence does not matter.
        let full = format!("```json\n{}\n```", json(&c));
        let close = full.rfind('}').expect("a closing brace");

        for (end, _) in full.char_indices() {
            let cut = parse_llm_cascade(&full[..end], &limits);

            if end <= close {
                assert!(cut.is_err(), "prefix of {end} bytes");
            } else {
                assert_eq!(cut.expect("closed object"), c);
            }
        }
    }

    #[test]
    fn llm_requests_are_deterministic_and_carry_parents_and_limits() {
        let limits = staged_limits();
        let seeds = seeds();
        let pretty = |c: &Cascade| serde_json::to_string_pretty(c).expect("encode");

        let m = llm_mutate_request(&seeds[0], 0.123456, &limits, 7);

        assert_eq!(m, llm_mutate_request(&seeds[0], 0.123456, &limits, 7));
        assert_eq!((m.think_tokens, m.max_tokens, m.seed), (0, 4096, 7));
        assert_eq!(m.temperature, 0.7);
        assert!(m.system.contains("exactly one JSON object"));
        assert!(m.prompt.contains(&pretty(&seeds[0])));
        assert!(m.prompt.contains("public search fitness: 0.1235"));
        assert!(
            m.prompt
                .contains("stage 0: 4096, stage 1: 512, stage 2: 0, stage 3: 4096")
        );

        for spec in &limits.models {
            assert!(
                m.prompt.contains(&format!("\"{}\"", spec.tag)),
                "{}",
                spec.tag
            );
        }

        // The seed is passed through and changes nothing else.
        let reseeded = llm_mutate_request(&seeds[0], 0.123456, &limits, 8);

        assert_eq!((&reseeded.system, &reseeded.prompt), (&m.system, &m.prompt));
        assert_eq!(reseeded.seed, 8);

        let x = llm_crossover_request(&seeds[0], 0.25, &seeds[3], 0.5, &limits, 9);

        assert_eq!(
            x,
            llm_crossover_request(&seeds[0], 0.25, &seeds[3], 0.5, &limits, 9)
        );
        assert_eq!((x.think_tokens, x.max_tokens, x.seed), (0, 4096, 9));
        assert!(x.prompt.contains(&pretty(&seeds[0])));
        assert!(x.prompt.contains(&pretty(&seeds[3])));
        assert!(x.prompt.contains("public search fitness: 0.2500"));
        assert!(x.prompt.contains("public search fitness: 0.5000"));
        assert_ne!(x.prompt, m.prompt);

        // A fitness that is not a number is said so, not printed as NaN.
        let unmeasured = llm_mutate_request(&seeds[0], f64::NAN, &limits, 7);

        assert!(
            unmeasured
                .prompt
                .contains("public search fitness: not measured")
        );

        // What the request asks for, the parser takes back.
        let back = parse_llm_cascade(&pretty(&seeds[0]), &limits).expect("round trip");

        assert_eq!(back, seeds[0]);
    }

    #[test]
    fn llm_requests_never_mention_held_out_evaluation() {
        let limits = CascadeLimits::default();
        let mut parent = seeds().remove(0);

        // The seed's role mentions its puzzle rule as hidden; that is the parent's text, not ours.
        parent.stages[0].workers[0].sections[0].text = "You solve grid puzzles.".to_string();

        let requests = [
            llm_mutate_request(&parent, 0.5, &limits, 1),
            llm_crossover_request(&parent, 0.5, &parent, 0.4, &limits, 2),
        ];

        for req in requests {
            let text = format!("{}\n{}", req.system, req.prompt).to_lowercase();

            for word in [
                "hidden", "held-out", "held out", "holdout", "secret", "test set",
            ] {
                assert!(!text.contains(word), "{word:?} in a request");
            }
        }
    }

    #[test]
    fn phrases_pass_the_genome_text_rules() {
        let limits = CascadeLimits::default();
        let names = ["strategy", "check", "residual", "style", "role", "format"];

        assert!(PHRASES.len() >= 30, "{} phrases", PHRASES.len());

        for (k, (name, text)) in PHRASES.iter().enumerate() {
            let worker = Worker {
                model: "sim:small".to_string(),
                sections: vec![Section {
                    name: name.to_string(),
                    text: text.to_string(),
                }],
                traits: Traits::default(),
            };
            let c = Cascade {
                stages: vec![Stage {
                    workers: vec![worker],
                }],
            };

            assert!(
                c.check(&limits).is_ok(),
                "{name}: {text}: {:?}",
                c.check(&limits)
            );
            assert!(names.contains(name), "{name}");
            assert!(text.chars().count() <= 200, "{text}");
            assert!(
                text.chars()
                    .all(|ch| ch.is_ascii() || "äöüÄÖÜß".contains(ch)),
                "{text}"
            );
            assert!(
                PHRASES[..k].iter().all(|(_, t)| t != text),
                "duplicate: {text}"
            );
        }
    }

    #[test]
    fn phrases_mix_single_combined_german_and_neutral_texts() {
        for (k, stems) in FAMILIES.iter().enumerate() {
            assert!(
                PHRASES.iter().any(|(_, t)| families_of(t) == [k]),
                "no phrase uses {stems:?} alone"
            );
        }

        let combined = PHRASES
            .iter()
            .filter(|(_, t)| families_of(t).len() >= 2)
            .count();
        let triple = PHRASES
            .iter()
            .filter(|(_, t)| families_of(t).len() >= 3)
            .count();
        let neutral = PHRASES
            .iter()
            .filter(|(_, t)| families_of(t).is_empty())
            .count();

        assert!(combined >= 5, "{combined} combined phrases");
        assert!(triple >= 2, "{triple} phrases with three kinds");
        assert!(neutral >= 8, "{neutral} neutral phrases");

        for stem in [
            "spiegel",
            "dreh",
            "schwerkraft",
            "farb",
            "kombin",
            "vorige antwort",
        ] {
            assert!(
                PHRASES.iter().any(|(_, t)| t.to_lowercase().contains(stem)),
                "no German phrase with {stem:?}"
            );
        }

        // Every residual phrase reviews the answer of an earlier stage.
        let residual: Vec<&str> = PHRASES
            .iter()
            .filter(|(name, _)| *name == RESIDUAL)
            .map(|(_, t)| *t)
            .collect();

        assert!(residual.len() >= 3);
        assert!(
            residual.iter().all(|t| families_of(t).contains(&6)),
            "{residual:?}"
        );
    }

    #[test]
    fn op_serialises_in_snake_case() {
        let encode = |op: Op| serde_json::to_string(&op).expect("encode");

        assert_eq!(encode(Op::SectionReplace), "\"section_replace\"");
        assert_eq!(encode(Op::ThinkUp), "\"think_up\"");
        assert_eq!(encode(Op::LlmCrossover), "\"llm_crossover\"");

        for op in ALL_OPS {
            let back: Op = serde_json::from_str(&encode(op)).expect("decode");

            assert_eq!(back, op);
        }
    }

    /// A Cascade with Stages of these widths whose Workers differ in every gene: a note
    /// section that names `tag` and the Worker, the model by parity, and each Trait.
    fn distinct(tag: &str, widths: &[usize]) -> Cascade {
        let base = seeds().remove(0).stages[0].workers[0].clone();
        let mut k = 0u32;
        let mut stages = Vec::new();

        for &width in widths {
            let mut workers = Vec::new();

            for _ in 0..width {
                let mut w = base.clone();

                w.sections.push(Section {
                    name: "note".to_string(),
                    text: format!("Worker {tag}{k}."),
                });
                w.model = if k.is_multiple_of(2) {
                    "sim:small"
                } else {
                    "sim:large"
                }
                .to_string();
                w.traits = Traits {
                    think_tokens: 128 * (k + 1),
                    temperature: 0.1 * (k as f32 + 1.0),
                    answer_tokens: 128 + 64 * k,
                    self_check: !k.is_multiple_of(2),
                    ..Traits::default()
                };
                workers.push(w);
                k += 1;
            }

            stages.push(Stage { workers });
        }

        Cascade { stages }
    }

    /// Every Worker of `c`, in Stage order.
    fn workers_of(c: &Cascade) -> Vec<&Worker> {
        c.stages.iter().flat_map(|s| &s.workers).collect()
    }

    /// The genes of `w` that the mutation `op` leaves as they are: a child Worker shares
    /// them with its source Worker.
    fn kept(op: Op, w: &Worker) -> (Option<String>, Option<Vec<Section>>, Option<Traits>) {
        let model = Some(w.model.clone());
        let sections = Some(w.sections.clone());
        let traits = Some(w.traits.clone());

        match op {
            Op::ThinkUp
            | Op::ThinkDown
            | Op::TemperatureShift
            | Op::SelfCheckFlip
            | Op::AnswerTokensShift => (model, sections, None),
            Op::ModelSwitch => (None, sections, traits),
            Op::SectionReplace
            | Op::SectionAdd
            | Op::SectionDrop
            | Op::SectionSwap
            | Op::StageAdd => (model, None, traits),
            Op::WorkerAdd => {
                let warm = Traits {
                    temperature: 0.0,
                    ..w.traits.clone()
                };

                (model, sections, Some(warm))
            }
            _ => (model, sections, traits),
        }
    }

    /// The flat index of the Worker that `partner` picks in a parent of these widths,
    /// computed by the rule itself: the same position, or the last Stage or Worker there is.
    fn partner_by_rule(widths: &[usize], s: usize, w: usize) -> usize {
        let ps = s.min(widths.len() - 1);
        let pw = w.min(widths[ps] - 1);

        widths[..ps].iter().sum::<usize>() + pw
    }

    #[test]
    fn every_operator_names_its_sources() {
        let limits = CascadeLimits::default();
        let parent = distinct("P", &[2, 1, 2]);
        let mutations: Vec<Op> = MUTATION_WEIGHTS.iter().map(|&(op, _)| op).collect();

        assert!(parent.check(&limits).is_ok(), "{:?}", parent.check(&limits));

        // Each mutation operator: every child Worker has one source, the parent Worker it
        // was built from, which shares every gene the operator leaves alone.
        for &op in &mutations {
            let mut applied = 0;

            for seed in 0..60 {
                let Some((mut child, map)) =
                    apply_sourced(&parent, op, &limits, PHRASES, &mut rng(seed))
                else {
                    continue;
                };

                clamp_to_limits(&mut child, &limits);

                let sources = flatten_sources(map, &child);
                let from = workers_of(&parent);

                assert_eq!(sources.len(), child.worker_count(), "{op:?}");

                for (w, src) in workers_of(&child).into_iter().zip(&sources) {
                    assert_eq!(src.len(), 1, "{op:?}");
                    assert_eq!(src[0].parent, 0, "{op:?}");
                    assert_eq!(
                        kept(op, w),
                        kept(op, from[src[0].worker]),
                        "{op:?} seed {seed}"
                    );
                }

                applied += 1;
            }

            assert!(applied > 0, "{op:?} never applied");
        }

        // The structural operators move the map as they move the Workers.
        let structural = |op: Op, seed: u64| {
            let (child, map) =
                apply_sourced(&parent, op, &limits, PHRASES, &mut rng(seed)).expect("applies");

            flatten_sources(map, &child)
                .iter()
                .map(|s| s[0].worker)
                .collect::<Vec<usize>>()
        };
        let shapes: BTreeSet<Vec<usize>> = (0..40)
            .flat_map(|seed| {
                [Op::WorkerAdd, Op::WorkerDrop, Op::StageAdd, Op::StageDrop]
                    .map(|op| structural(op, seed))
            })
            .collect();

        assert!(
            shapes.contains(&vec![0, 1, 2, 3, 4, 0]),
            "a clone of Worker 0"
        );
        assert!(shapes.contains(&vec![0, 2, 3, 4]), "Worker 1 dropped");
        assert!(
            shapes.contains(&vec![0, 1, 2, 3, 4, 2]),
            "a new last Stage copied from 2"
        );
        assert!(
            shapes.contains(&vec![0, 1, 3, 4]),
            "the middle Stage dropped"
        );

        // The public operator names the same sources as the operator it drew.
        for seed in 0..200 {
            if let Some((child, op, sources)) =
                mutate_sourced(&parent, &limits, PHRASES, &mut rng(seed))
            {
                let from = workers_of(&parent);

                for (w, src) in workers_of(&child).into_iter().zip(&sources) {
                    assert_eq!(kept(op, w), kept(op, from[src[0].worker]), "{op:?}");
                }
            }
        }

        // Both crossovers.
        let (a, b) = (distinct("A", &[2, 1, 2]), distinct("B", &[1, 3]));
        let (from_a, from_b) = (workers_of(&a), workers_of(&b));
        let mut by_stage = 0;
        let mut by_section = 0;

        for seed in 0..100 {
            // StageCrossover: the prefix from A, the suffix from B, each Worker as it was.
            if let Some((child, sources)) = stage_crossover_sourced(&a, &b, &limits, &mut rng(seed))
            {
                for (w, src) in workers_of(&child).into_iter().zip(&sources) {
                    let from = if src[0].parent == 0 { &from_a } else { &from_b };

                    assert_eq!(src.len(), 1);
                    assert_eq!(w, from[src[0].worker], "seed {seed}");
                }

                by_stage += 1;
            }

            // SectionCrossover: A's Worker at the same position and B's partner Worker;
            // every gene of the child comes from one of them.
            if let Some((child, sources)) =
                section_crossover_sourced(&a, &b, &limits, &mut rng(seed))
            {
                let mut k = 0;

                for (s, stage) in child.stages.iter().enumerate() {
                    for (w, worker) in stage.workers.iter().enumerate() {
                        let expected = [
                            Source {
                                parent: 0,
                                worker: worker_index(&a, s, w),
                            },
                            Source {
                                parent: 1,
                                worker: partner_by_rule(&[1, 3], s, w),
                            },
                        ];

                        assert_eq!(sources[k], expected, "seed {seed}");

                        let (x, y) = (from_a[expected[0].worker], from_b[expected[1].worker]);

                        assert!([&x.model, &y.model].contains(&&worker.model));
                        assert!(
                            [x.traits.think_tokens, y.traits.think_tokens]
                                .contains(&worker.traits.think_tokens)
                        );
                        assert!(
                            [x.traits.answer_tokens, y.traits.answer_tokens]
                                .contains(&worker.traits.answer_tokens)
                        );

                        for (mine, own) in worker.sections.iter().zip(&x.sections) {
                            let theirs = y.sections.iter().find(|t| t.name == own.name);

                            assert_eq!(mine.name, own.name);
                            assert!(
                                mine.text == own.text
                                    || theirs.is_some_and(|t| t.text == mine.text)
                            );
                        }

                        k += 1;
                    }
                }

                assert_eq!(k, sources.len());
                by_section += 1;
            }
        }

        assert!(by_stage > 0 && by_section > 0, "{by_stage} {by_section}");

        // Mutator children: the partner rule on each parent, whatever shape the Mutator
        // wrote.
        let written = distinct("C", &[3, 1, 2, 2]);
        let mutated = partner_sources(&written, &[&b]);
        let crossed = partner_sources(&written, &[&a, &b]);
        let mut k = 0;

        for (s, stage) in written.stages.iter().enumerate() {
            for w in 0..stage.workers.len() {
                let of_a = Source {
                    parent: 0,
                    worker: partner_by_rule(&[2, 1, 2], s, w),
                };
                let of_b = Source {
                    parent: 0,
                    worker: partner_by_rule(&[1, 3], s, w),
                };

                assert_eq!(mutated[k], vec![of_b]);
                assert_eq!(crossed[k], vec![of_a, Source { parent: 1, ..of_b }]);

                // The source is the Worker that `partner` pairs with this position.
                assert!(std::ptr::eq(
                    from_b[of_b.worker],
                    partner(&b, s, w).expect("a partner")
                ));

                k += 1;
            }
        }

        assert_eq!((mutated.len(), crossed.len()), (8, 8));
    }

    /// `c` with a random Soul of `n` phases and a random Coupling strength on every Worker.
    fn with_random_souls(c: &Cascade, n: usize, rng: &mut ChaCha8Rng) -> Cascade {
        let mut c = c.clone();

        for w in c.stages.iter_mut().flat_map(|s| &mut s.workers) {
            w.traits.phase_state = Some(PhaseState((0..n).map(|_| rng.random()).collect()));
            w.traits.coupling = rng.random_range(0..=COUPLING_MAX);
        }

        c
    }

    /// What one operator gives, and where the stream stands afterwards.
    type Outcome = (Option<(Cascade, Option<Op>, Sources)>, u128);

    /// Each operator on `(a, b)` from the stream `seed`, as the breeder calls it.
    fn outcomes(a: &Cascade, b: &Cascade, limits: &CascadeLimits, seed: u64) -> [Outcome; 3] {
        let mut r = rng(seed);
        let mutated = mutate_sourced(a, limits, PHRASES, &mut r).map(|(c, op, s)| (c, Some(op), s));
        let after_mutation = r.get_word_pos();
        let mut r = rng(seed);
        let by_stage = stage_crossover_sourced(a, b, limits, &mut r).map(|(c, s)| (c, None, s));
        let after_stage = r.get_word_pos();
        let mut r = rng(seed);
        let by_section = section_crossover_sourced(a, b, limits, &mut r).map(|(c, s)| (c, None, s));

        [
            (mutated, after_mutation),
            (by_stage, after_stage),
            (by_section, r.get_word_pos()),
        ]
    }

    #[test]
    fn operators_see_no_souls() {
        let limits = staged_limits();
        let mut walk = rng(5);
        let mut souls = rng(6);

        for k in 0..1000 {
            let a = evolved(k, 1 + k % 5, &limits, &mut walk);
            let b = evolved(k + 1 + k % 3, 1 + k % 4, &limits, &mut walk);
            let (sa, sb) = (
                with_random_souls(&a, 7, &mut souls),
                with_random_souls(&b, 8, &mut souls),
            );

            for (plain, souled) in outcomes(&a, &b, &limits, k as u64)
                .into_iter()
                .zip(outcomes(&sa, &sb, &limits, k as u64))
            {
                assert_eq!(plain, souled, "pair {k}");

                if let Some((child, _, _)) = &souled.0 {
                    assert!(!child.carries_soul(), "pair {k}");
                }
            }
        }

        // A SectionCrossover child with the genes of one parent is refused and drawn again,
        // as without Souls. Parents that differ in one gene give only such children, so the
        // operator gives up; with two genes some draws are refused and some accepted.
        let a = seeds().remove(1);
        let mut one = a.clone();

        one.stages[0].workers[0].traits.self_check ^= true;

        let mut two = one.clone();

        two.stages[0].workers[0].traits.answer_tokens *= 2;

        // Every coin of one attempt draws one u64, two words of the stream.
        let coins = a.stages[0].workers[0].sections.len() + 5;
        let words = 2 * coins as u128 * CROSSOVER_ATTEMPTS as u128;
        let mut redrawn = 0;

        for seed in 0..64 {
            let (sa, s1, s2) = (
                with_random_souls(&a, 7, &mut souls),
                with_random_souls(&one, 7, &mut souls),
                with_random_souls(&two, 7, &mut souls),
            );
            let crossed = |x: &Cascade, y: &Cascade| {
                let mut r = rng(seed);
                let child = section_crossover_sourced(x, y, &limits, &mut r);

                (child, r.get_word_pos())
            };

            let (plain, pos) = crossed(&a, &one);

            assert_eq!(plain, None, "a copy of a parent every time");
            assert_eq!(pos, words, "all attempts drawn");
            assert_eq!(crossed(&sa, &s1), (None, pos), "seed {seed}");

            let (plain, pos) = crossed(&a, &two);
            let attempts = pos * CROSSOVER_ATTEMPTS as u128 / words;

            assert_eq!(crossed(&sa, &s2), (plain.clone(), pos), "seed {seed}");

            if plain.is_some() && attempts > 1 {
                redrawn += 1;
            }
        }

        assert!(redrawn > 0, "some children were drawn again");
    }

    #[test]
    fn cross_worker_never_crosses_a_soul() {
        let (mut mine, mut theirs) = (
            seeds()[0].stages[0].workers[0].clone(),
            seeds()[1].stages[0].workers[0].clone(),
        );

        mine.traits.phase_state = Some(PhaseState(vec![1, 2, 3]));
        mine.traits.coupling = 100;
        theirs.traits.phase_state = Some(PhaseState(vec![4, 5, 6]));
        theirs.traits.coupling = 900;

        for seed in 0..50 {
            let mut child = mine.clone();

            cross_worker(
                &mut child,
                &theirs,
                None,
                &mut Steered::default(),
                &mut rng(seed),
            );

            assert_eq!(child.traits.phase_state, mine.traits.phase_state);
            assert_eq!(child.traits.coupling, 100);
        }
    }

    /// The places of three circuit families in `designer::FAMILY_STEMS`.
    const LINEAR: usize = 0;
    const PHASE: usize = 1;
    const SEARCH: usize = 4;

    /// Draws of the statistical tests of arm W.
    const DRAWS: u32 = 100_000;

    /// The circuit seed Cascade `k` (linear, phase, hadamard, search), without its name.
    fn circuit_seed(k: usize) -> Cascade {
        circuit_seed_cascades().remove(k % 4).1
    }

    /// Circuit seed `k` after `steps` mutations with the circuit phrases.
    fn evolved_circuit(
        k: usize,
        steps: usize,
        limits: &CascadeLimits,
        rng: &mut ChaCha8Rng,
    ) -> Cascade {
        let mut c = circuit_seed(k);

        for _ in 0..steps {
            if let Some((child, _)) = mutate_with(&c, limits, CIRCUIT_PHRASES, rng) {
                c = child;
            }
        }

        c
    }

    /// The preference of a circuit Soul with every phase at a quarter turn, which gives 1,
    /// except the phases `given` as (family, phase).
    fn prefer(given: &[(usize, u16)]) -> Vec<f64> {
        let mut soul = [QUARTER_TURN; 8];

        for &(f, p) in given {
            soul[f] = p;
        }

        preference(&soul, Readout::Complex)
    }

    /// True when `got` lies within 3 standard errors of the share `p` over `n` draws.
    fn within_3_se(got: f64, p: f64, n: u32) -> bool {
        (got - p).abs() < 3.0 * (p * (1.0 - p) / f64::from(n)).sqrt()
    }

    /// A small pool for the protection test: strategies that name Search or no family, and
    /// two texts that a new section takes.
    const PROTECTION_PHRASES: &[(&str, &str)] = &[
        (
            "strategy",
            "Search through all short gate sequences and stop at the first one that matches every row.",
        ),
        (
            "strategy",
            "Count the T and TDG gates first, then the CNOT gates; of two exact circuits, the one with fewer T gates wins.",
        ),
        (
            "check",
            "If your circuit fails on even one input, drop it and test the next idea.",
        ),
        ("style", "Be concise."),
    ];

    #[test]
    fn no_weight_falls_below_the_floor() {
        let limits = CascadeLimits::default();
        let mut souls = rng(31);
        let mut walk = rng(32);
        // The extremes, every family at 0 and at 2, and the preferences of random Souls in
        // both readouts.
        let mut prefs = vec![vec![0.0; 8], vec![2.0; 8], prefer(&[])];

        for _ in 0..200 {
            let soul: Vec<u16> = (0..8).map(|_| souls.random()).collect();

            prefs.push(preference(&soul, Readout::Complex));
            prefs.push(preference(&soul, Readout::Real));
        }

        let workers: Vec<Worker> = (0..60)
            .flat_map(|k| evolved_circuit(k, k % 9, &limits, &mut walk).stages)
            .flat_map(|s| s.workers)
            .collect();
        let mut lowest = f64::INFINITY;

        for pref in &prefs {
            let lens = Lens {
                families_of: named_families,
                pref,
            };

            for w in &workers {
                let options = add_options(w, CIRCUIT_PHRASES);
                let weights = [lens.favour(&w.system_prompt())]
                    .into_iter()
                    .chain(w.sections.iter().map(|s| lens.protect(&s.text)))
                    .chain(CIRCUIT_PHRASES.iter().map(|(_, t)| lens.favour(t)))
                    .chain(add_weights(w, &options, lens));

                for x in weights {
                    assert!(x >= WEIGHT_FLOOR, "{x} below the floor");
                    lowest = lowest.min(x);
                }
            }
        }

        assert_eq!(lowest, WEIGHT_FLOOR, "the floor binds");
    }

    #[test]
    fn a_preferring_worker_protects_its_family_by_the_expected_factor() {
        let limits = CascadeLimits::default();
        // The linear seed: a role and a format that name no family, and a strategy that
        // names Linear.
        let parent = circuit_seed(0);
        let strategy = parent.stages[0].workers[0].sections[1].text.clone();

        assert_eq!(named_families(&strategy), [LINEAR]);

        // Its Soul prefers Linear at 2 (no turn), where the floor ε protects the strategy,
        // and at about 1.5 (a sixth of a turn), where 1 − L does.
        for phase in [0, 10923] {
            let prefs = vec![prefer(&[(LINEAR, phase)])];
            let steer = Steer {
                families_of: named_families,
                preferences: &prefs,
            };
            let protect = (2.0 - prefs[0][LINEAR]).max(WEIGHT_FLOOR);

            assert!(prefs[0][LINEAR] > 1.49);

            // The share of draws that overwrite the strategy, today and with the Soul.
            // SectionReplace draws one of three sections with the weights 1, 1 − L and 1;
            // SectionAdd writes one of two strategies over it, each with the weight
            // 1 · (1 − L), or one of two new sections with the weight 1.
            let cases = [
                (Op::SectionReplace, 1.0 / 3.0, protect / (2.0 + protect)),
                (Op::SectionAdd, 0.5, 2.0 * protect / (2.0 + 2.0 * protect)),
            ];

            for (op, today, steered) in cases {
                let share = |steer: Option<&Steer<'_>>| {
                    let mut r = rng(33);
                    let hits = (0..DRAWS)
                        .filter(|_| {
                            let (child, _) = apply_steered(
                                &parent,
                                op,
                                &limits,
                                PROTECTION_PHRASES,
                                steer,
                                &mut Steered::default(),
                                &mut r,
                            )
                            .expect("applies");

                            child.stages[0].workers[0].sections[1].text != strategy
                        })
                        .count();

                    hits as f64 / f64::from(DRAWS)
                };
                let (n, w) = (share(None), share(Some(&steer)));
                let factor = steered / today;
                // The standard error of a ratio of two independent shares, by the delta
                // method.
                let se = (w / n)
                    * ((1.0 - w) / (w * f64::from(DRAWS)) + (1.0 - n) / (n * f64::from(DRAWS)))
                        .sqrt();

                eprintln!(
                    "{op:?}, phase {phase}: overwritten {n:.4} today, {w:.4} with the Soul; factor {:.4} for {factor:.4}, 3 standard errors {:.4}",
                    w / n,
                    3.0 * se
                );
                assert!(w < n, "{op:?}: {w} against {n}");
                assert!(
                    (w / n - factor).abs() < 3.0 * se,
                    "{op:?}, phase {phase}: factor {} for {factor}, standard error {se}",
                    w / n
                );
                assert!(within_3_se(n, today, DRAWS), "{op:?} today: {n}");
                assert!(within_3_se(w, steered, DRAWS), "{op:?} with the Soul: {w}");
            }
        }
    }

    #[test]
    fn each_choice_follows_its_weights() {
        let limits = CascadeLimits::default();
        let n = f64::from(DRAWS);

        // Choice 1: SectionReplace changes the Worker whose prompt names Linear, with
        // preference 2, by the weight 1 + 1 against 1 for the Worker that names no family
        // the Soul leans to (Search, at preference 1).
        let mut two = circuit_seed(0);
        let other = circuit_seed(3).stages[0].workers[0].clone();

        assert_eq!(
            named_families(&two.stages[0].workers[0].system_prompt()),
            [LINEAR]
        );
        assert_eq!(named_families(&other.system_prompt()), [SEARCH]);
        two.stages[0].workers.push(other);

        let prefs = vec![prefer(&[(LINEAR, 0)]); 2];
        let steer = Steer {
            families_of: named_families,
            preferences: &prefs,
        };
        let mut r = rng(34);
        let first = (0..DRAWS)
            .filter(|_| {
                let (child, _) = apply_steered(
                    &two,
                    Op::SectionReplace,
                    &limits,
                    CIRCUIT_PHRASES,
                    Some(&steer),
                    &mut Steered::default(),
                    &mut r,
                )
                .expect("applies");

                child.stages[0].workers[0] != two.stages[0].workers[0]
            })
            .count() as f64;

        assert!(within_3_se(first / n, 2.0 / 3.0, DRAWS), "{}", first / n);

        // Choice 3: the phrase, still from its section's name with probability 0.7 and from
        // the whole pool otherwise, and within either by 1 + L. With Phase at 2, a strategy
        // text that names Phase weighs 2 and every other 1.
        let pref = prefer(&[(PHASE, 0)]);
        let lens = Lens {
            families_of: named_families,
            pref: &pref,
        };
        let current = "Count the T and TDG gates first, then the CNOT gates; of two exact circuits, the one with fewer T gates wins.";
        let weight = |t: &str| {
            if named_families(t).contains(&PHASE) {
                2.0
            } else {
                1.0
            }
        };
        let same: Vec<&str> = CIRCUIT_PHRASES
            .iter()
            .filter(|(m, t)| *m == "strategy" && *t != current)
            .map(|(_, t)| *t)
            .collect();
        let any: Vec<&str> = CIRCUIT_PHRASES
            .iter()
            .filter(|(_, t)| *t != current)
            .map(|(_, t)| *t)
            .collect();
        let total = |texts: &[&str]| texts.iter().map(|t| weight(t)).sum::<f64>();
        let (phase_same, phase_any): (Vec<&str>, Vec<&str>) = (
            same.iter().copied().filter(|t| weight(t) > 1.0).collect(),
            any.iter().copied().filter(|t| weight(t) > 1.0).collect(),
        );
        let p_same = SAME_NAME_P + (1.0 - SAME_NAME_P) * total(&same) / total(&any);
        let p_phase = SAME_NAME_P * total(&phase_same) / total(&same)
            + (1.0 - SAME_NAME_P) * total(&phase_any) / total(&any);
        let mut r = rng(35);
        let (mut by_name, mut by_phase) = (0u32, 0u32);

        assert!(!phase_same.is_empty() && phase_any.len() > phase_same.len());

        for _ in 0..DRAWS {
            let t = replacement_text(
                &mut r,
                CIRCUIT_PHRASES,
                "strategy",
                current,
                Some(lens),
                &mut Steered::default(),
            )
            .expect("a text");

            by_name += u32::from(same.contains(&t));
            by_phase += u32::from(weight(t) > 1.0);
        }

        assert!(within_3_se(f64::from(by_name) / n, p_same, DRAWS));
        assert!(within_3_se(f64::from(by_phase) / n, p_phase, DRAWS));

        // The coin of section crossover: the partner's text, which names no family, against
        // a text that names Linear at 2, so with probability 1 / (1 + 2).
        let pref = prefer(&[(LINEAR, 0)]);
        let lens = Lens {
            families_of: named_families,
            pref: &pref,
        };
        let mine = circuit_seed(0).stages[0].workers[0].clone();
        let mut theirs = mine.clone();

        theirs.sections[1].text = current.to_string();

        let mut r = rng(36);
        let taken = (0..DRAWS)
            .filter(|_| {
                let mut child = mine.clone();

                cross_worker(
                    &mut child,
                    &theirs,
                    Some(lens),
                    &mut Steered::default(),
                    &mut r,
                );

                child.sections[1].text == current
            })
            .count() as f64;

        assert!(within_3_se(taken / n, 1.0 / 3.0, DRAWS), "{}", taken / n);
    }

    #[test]
    fn one_draw_of_an_operator_is_the_attempt_that_drew_it() {
        // `mutate_once` makes what an attempt of today's mutation makes once it has drawn its
        // operator: from the same place in the stream, the same child, the same sources and
        // the same choices of arm W, and the stream stops at the same word. It draws no
        // operator and tries once.
        let limits = staged_limits();
        let mut walk = rng(41);
        let mut compared = BTreeSet::new();
        let prefs = prefer(&[(0, 0), (2, 32768)]);

        for k in 0..300 {
            let parent = evolved_circuit(k, k % 7, &limits, &mut walk);
            let prefs = vec![prefs.clone(); parent.worker_count()];
            let steer = Steer {
                families_of: named_families,
                preferences: &prefs,
            };
            let seed = 4100 + k as u64;

            for weighted in [None, Some(&steer)] {
                let (mut today, mut once) = (rng(seed), rng(seed));
                let op = draw_op(&mut once);
                let Some((child, sources, steered)) =
                    mutate_once(&parent, op, &limits, CIRCUIT_PHRASES, weighted, &mut once)
                else {
                    continue;
                };
                let full = match weighted {
                    None => mutate_sourced(&parent, &limits, CIRCUIT_PHRASES, &mut today)
                        .map(|(c, op, s)| (c, op, s, Steered::default())),
                    Some(steer) => {
                        mutate_steered(&parent, &limits, CIRCUIT_PHRASES, steer, &mut today)
                    }
                };

                assert_eq!(full, Some((child, op, sources, steered)), "{op:?} {k}");
                assert_eq!(today.get_word_pos(), once.get_word_pos(), "{op:?} {k}");
                compared.insert(format!("{op:?}"));
            }
        }

        for op in ["SectionReplace", "SectionAdd", "SectionDrop"] {
            assert!(compared.contains(op), "{compared:?}");
        }

        // An operator that cannot change the parent gives no draw: a single Stage cannot
        // drop a Stage.
        let seed_one = circuit_seed(0);

        assert_eq!(
            mutate_once(
                &seed_one,
                Op::StageDrop,
                &limits,
                CIRCUIT_PHRASES,
                None,
                &mut rng(7)
            ),
            None
        );
    }

    #[test]
    fn with_every_preference_one_the_operators_are_todays() {
        // Complex phases all at a quarter turn give every family the preference 1.
        let neutral = preference(&[QUARTER_TURN; 8], Readout::Complex);

        assert_eq!(neutral, vec![1.0; 8]);

        let limits = staged_limits();
        let mut walk = rng(37);
        let none = Steered::default();

        for k in 0..400 {
            let a = evolved_circuit(k, k % 9, &limits, &mut walk);
            let b = evolved_circuit(k + 1 + k % 3, k % 6, &limits, &mut walk);
            let prefs = vec![neutral.clone(); a.worker_count()];
            let steer = Steer {
                families_of: named_families,
                preferences: &prefs,
            };
            let seed = k as u64;

            // Each section operator alone gives the same child, sources and place in the
            // stream, and counts no weighted choice.
            for op in [Op::SectionReplace, Op::SectionAdd] {
                let (mut today, mut steered) = (rng(seed), rng(seed));
                let mut tally = Steered::default();

                assert_eq!(
                    apply_sourced(&a, op, &limits, CIRCUIT_PHRASES, &mut today),
                    apply_steered(
                        &a,
                        op,
                        &limits,
                        CIRCUIT_PHRASES,
                        Some(&steer),
                        &mut tally,
                        &mut steered
                    ),
                    "{op:?} {k}"
                );
                assert_eq!(today.get_word_pos(), steered.get_word_pos(), "{op:?} {k}");
                assert_eq!(tally, none);
            }

            // So do the operators as the breeder calls them.
            let (mut today, mut steered) = (rng(seed), rng(seed));
            let plain = mutate_sourced(&a, &limits, CIRCUIT_PHRASES, &mut today);
            let weighted = mutate_steered(&a, &limits, CIRCUIT_PHRASES, &steer, &mut steered);

            assert_eq!(
                plain,
                weighted.map(|(c, op, s, tally)| {
                    assert_eq!(tally, none);
                    (c, op, s)
                })
            );
            assert_eq!(today.get_word_pos(), steered.get_word_pos());

            let (mut today, mut steered) = (rng(seed), rng(seed));
            let plain = section_crossover_sourced(&a, &b, &limits, &mut today);
            let weighted = section_crossover_steered(&a, &b, &limits, &steer, &mut steered);

            assert_eq!(
                plain,
                weighted.map(|(c, s, tally)| {
                    assert_eq!(tally, none);
                    (c, s)
                })
            );
            assert_eq!(today.get_word_pos(), steered.get_word_pos());
        }

        // So the choices match today's in distribution: SectionReplace takes a phrase of the
        // section's name with probability 0.7 plus that name's share of the whole pool, and
        // each phrase within the pool that is drawn alike.
        let lens = Lens {
            families_of: named_families,
            pref: &neutral,
        };
        let current = "Count the T and TDG gates first, then the CNOT gates; of two exact circuits, the one with fewer T gates wins.";
        let same: Vec<&str> = CIRCUIT_PHRASES
            .iter()
            .filter(|(m, t)| *m == "strategy" && *t != current)
            .map(|(_, t)| *t)
            .collect();
        let any: Vec<&str> = CIRCUIT_PHRASES
            .iter()
            .filter(|(_, t)| *t != current)
            .map(|(_, t)| *t)
            .collect();
        let mut r = rng(38);
        let mut counts: HashMap<&str, u32> = HashMap::new();

        for _ in 0..DRAWS {
            let t = replacement_text(
                &mut r,
                CIRCUIT_PHRASES,
                "strategy",
                current,
                Some(lens),
                &mut Steered::default(),
            )
            .expect("a text");

            *counts.entry(t).or_default() += 1;
        }

        let p_same = SAME_NAME_P + (1.0 - SAME_NAME_P) * same.len() as f64 / any.len() as f64;
        let by_name: u32 = same
            .iter()
            .map(|t| counts.get(t).copied().unwrap_or(0))
            .sum();

        assert!(within_3_se(
            f64::from(by_name) / f64::from(DRAWS),
            p_same,
            DRAWS
        ));

        // Each phrase: a strategy has 0.7 / |same| + 0.3 / |any|, any other text 0.3 / |any|.
        // The bound is 4 standard errors, since 60 phrases are tested at once.
        for t in &any {
            let p = (1.0 - SAME_NAME_P) / any.len() as f64
                + if same.contains(t) {
                    SAME_NAME_P / same.len() as f64
                } else {
                    0.0
                };
            let got = f64::from(counts.get(t).copied().unwrap_or(0)) / f64::from(DRAWS);
            let se = (p * (1.0 - p) / f64::from(DRAWS)).sqrt();

            assert!((got - p).abs() < 4.0 * se, "{t}: {got} for {p}");
        }
    }

    #[test]
    fn a_choice_changes_when_the_uniform_pick_of_its_draw_differs() {
        // Equal weights make today's draw and count nothing.
        for seed in 0..200 {
            let (mut steered, mut today) = (rng(seed), rng(seed));
            let mut tally = Steered::default();

            assert_eq!(
                weighted_index(&mut steered, &[0.4; 5], &mut tally),
                Some(today.random_range(0..5))
            );
            assert_eq!(
                weighted_coin(&mut steered, 3.0, 3.0, &mut tally),
                today.random_bool(0.5)
            );
            assert_eq!(steered.get_word_pos(), today.get_word_pos());
            assert_eq!(tally, Steered::default());
        }

        // Unequal weights draw one u in [0, 1): the pick is the first option whose summed
        // weight passes u times the total, and the choice changed when ⌊u·n⌋ differs.
        let weights = [1.0, 2.0, 0.25, 0.75];
        let mut changed = 0;

        for seed in 0..2000 {
            let (mut steered, mut today) = (rng(seed), rng(seed));
            let mut tally = Steered::default();
            let pick = weighted_index(&mut steered, &weights, &mut tally).expect("a pick");
            let u: f64 = today.random();
            let x = 4.0 * u;
            let expected = match x {
                x if x < 1.0 => 0,
                x if x < 3.0 => 1,
                x if x < 3.25 => 2,
                _ => 3,
            };
            let moved = expected != (4.0 * u) as usize;

            assert_eq!(pick, expected);
            assert_eq!(
                tally,
                Steered {
                    choices: 1,
                    changed: u64::from(moved)
                }
            );
            assert_eq!(steered.get_word_pos(), today.get_word_pos());
            changed += usize::from(moved);
        }

        // The weighted picks take u in [0, 0.25), [0.25, 0.75), [0.75, 0.8125) and [0.8125,
        // 1), the uniform ones in quarters. They differ for u in [0.5, 0.8125): 5 / 16 of the
        // draws change.
        assert!(within_3_se(
            f64::from(changed as u32) / 2000.0,
            0.3125,
            2000
        ));

        // The coin takes the partner's text when u lies below theirs / (mine + theirs), and
        // it changed when that differs from u < 1/2.
        for seed in 0..2000 {
            let (mut steered, mut today) = (rng(seed), rng(seed));
            let mut tally = Steered::default();
            let take = weighted_coin(&mut steered, 2.0, 1.0, &mut tally);
            let u: f64 = today.random();

            assert_eq!(take, u < 1.0 / 3.0);
            assert_eq!(
                tally,
                Steered {
                    choices: 1,
                    changed: u64::from((1.0 / 3.0..0.5).contains(&u))
                }
            );
        }

        // Without an option nothing is drawn.
        let mut r = rng(1);
        let mut tally = Steered::default();

        assert_eq!(weighted_index(&mut r, &[], &mut tally), None);
        assert_eq!((r.get_word_pos(), tally), (0, Steered::default()));
    }

    #[test]
    fn steered_operators_see_genes_only() {
        let limits = staged_limits();
        let mut walk = rng(39);
        let mut souls = rng(40);
        let mut steered_some = 0;

        for k in 0..300 {
            let a = evolved_circuit(k, 1 + k % 6, &limits, &mut walk);
            let b = evolved_circuit(k + 2, 1 + k % 4, &limits, &mut walk);
            let prefs: Vec<Vec<f64>> = (0..a.worker_count())
                .map(|_| {
                    let soul: Vec<u16> = (0..8).map(|_| souls.random()).collect();

                    preference(&soul, Readout::Complex)
                })
                .collect();
            let steer = Steer {
                families_of: named_families,
                preferences: &prefs,
            };
            let (sa, sb) = (
                with_random_souls(&a, 8, &mut souls),
                with_random_souls(&b, 8, &mut souls),
            );
            let seed = k as u64;
            let mutated =
                |x: &Cascade| mutate_steered(x, &limits, CIRCUIT_PHRASES, &steer, &mut rng(seed));
            let crossed = |x: &Cascade, y: &Cascade| {
                section_crossover_steered(x, y, &limits, &steer, &mut rng(seed))
            };
            let (m, c) = (mutated(&a), crossed(&a, &b));

            // Parents with Souls give the same children as without, and no child carries a
            // Soul.
            assert_eq!(m, mutated(&sa), "{k}");
            assert_eq!(c, crossed(&sa, &sb), "{k}");

            for child in [m.as_ref().map(|x| &x.0), c.as_ref().map(|x| &x.0)]
                .into_iter()
                .flatten()
            {
                assert!(!child.carries_soul(), "{k}");
            }

            steered_some += [m.map(|x| x.3), c.map(|x| x.2)]
                .into_iter()
                .flatten()
                .filter(|t| t.choices > 0)
                .count();
        }

        assert!(steered_some > 100, "{steered_some}");
    }
}
