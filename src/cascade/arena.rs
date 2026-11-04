//! Arena: runs Cascades on puzzles, lets them judge each other blind, rates them, and
//! breeds new ones with Darwin-Gödel-Machine parent selection (ADR 0005).
//!
//! Pure: no files, no network, no clock. Workers are reached only through `ModelPool`,
//! the LLM that writes new Cascades only through `Mutator` (normally Claude through
//! recorded answers, ADR 0002). The binary `brut-breed` supplies both, decides when to
//! stop, and writes the events this module returns.
//!
//! Every Cascade is also a judge. Inside a stage, its parallel Workers vote blind on each
//! other's answers. Across the archive, a Cascade judges pairs of answers from two
//! Cascades it is not related to. Judging skill is measured against the known answers of
//! the search panel and enters fitness with a fixed weight (decisions Q2, Q12, Q13). The
//! puzzles, their answers and the scoring never depend on a Cascade's content.
//!
//! The breeder works on any task family (`Domain`, step 1); the grid puzzles of Stage 2c
//! are the default. Every committed generation ends with a `state` event that holds the
//! universe observables (`observe.rs`).
//!
//! In a Run with Souls (step 2, `.scratch/seele-und-kappa/spec.md`), Souletics gives every
//! newborn Worker its birth Soul and Coupling strength from the Workers it was built from.
//! The genetic operators see genes only, and novelty and the seeds of calls go by gene ids.
//! Upbringing then shapes the newborn (`upbringing.rs`): it answers its first search task
//! with its birth Souls and the others with Souls imprinted toward its source Workers'
//! current Souls, which also start its current Souls; with teaching on, a section written
//! from its parents' best solved examples ends every system prompt it sends.
//! A Stage vote then tallies amplitudes: the verdicts of each judge weigh with its vote
//! phasor, and the answer with the largest squared magnitude goes on. The solve and vote
//! requests of a Cascade's own evaluation carry its Workers' phases to a backend that reads
//! them, a simulated model; a peer request never does. Judging is an Observation: the
//! judgments of a newborn's Stage votes and of the peer round are recorded, and at the
//! commit they pull the current Souls of both sides together, one after another in log
//! order, as strongly as the observer's Coupling strength says. A `souls` event after the
//! generation event lists the Workers whose current Soul changed. The select phase observes
//! nothing. With `variation` on (arm W, `.scratch/seelen-modell/spec.md`, section 4.1), the
//! section operators also get the preference of each parent Worker's birth Soul, which
//! weights their choices; they still see genes only.
//!
//! With `imagination` on (arm I, `.scratch/seelen-modell/spec-imagination.md`, sections 3.1 to
//! 3.3), every Worker carries learned values in its member, never in its genome
//! (`mind.rs`). An operator that can change the families a Worker names is drawn several
//! times on the same parents, and the draw whose prediction from these values is highest
//! becomes the child; the draws after the first take their own streams, so the generation's
//! stream advances as without imagination. Each newborn's real effect on the current goal then
//! moves the values of its Workers before its Born event, and those of their source Workers
//! at the commit, which a `learned` event after the generation event lists. With `goals` on
//! (arm G, section 3.4), the tasks that count for lexicase selection and for that effect
//! change every few generations; every newborn is still evaluated on the whole search panel.
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, HashMap, HashSet};

use rand::seq::{IndexedRandom, SliceRandom};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::backend::{GenRequest, GenResponse, ModelBackend, StopReason, approx_tokens};
use crate::cascade::domain::{Domain, DomainConfig, GenePool, Grade, GridDomain};
use crate::cascade::genome::{
    COUPLING_MAX, Cascade, CascadeLimits, PhaseState, Worker, adapt_seeds, check_section_text,
};
use crate::cascade::mind::{
    ALL_TASKS, Change, Effect, Goal, GoalsConfig, IMAGINING_OPS, ImaginationConfig, VALUE_MAX,
    choose, error, goal_cycle, goal_index, named, prediction, step,
};
use crate::cascade::observe::{
    ImaginationCounts, MindInput, ObserveInput, SoulInput, UniverseState, observe, strength_counts,
};
use crate::cascade::puzzle::{Grid, Puzzle, SELF_CHECK_TEXT, Verdict, parse_verdict};
use crate::cascade::selection::{
    SelectionParams, lexicase_pick, lexicase_weights, tournament_pick, tournament_weights,
};
use crate::cascade::soul::{
    ArrayVote, FIXED_PREFERRED, Inheritance, Phasor, Readout, SoulConfig, SoulDraws, Variation,
    array_vote, count_winner, fixed_preference, observation, preference, readout_phases, souletics,
    vote_phasor,
};
use crate::cascade::upbringing::{imprinted, section_block, teaching};
use crate::cascade::variation::{
    Op, Sources, Steer, Steered, llm_crossover_request_with, llm_mutate_request_with, mutate_once,
    mutate_sourced, mutate_steered, parse_llm_cascade, partner_sources, read_llm_cascade,
    section_crossover_sourced, section_crossover_steered, stage_crossover_sourced,
};
use crate::util::{BrutError, Result, derive_seed, sha256_hex};

/// Draws a parent or a partner gets before it gives up (each draw is cheap).
const DRAW_ATTEMPTS: usize = 16;

/// Partners drawn per crossover; the one with the best peer rating mates (Q2).
const PARTNER_CANDIDATES: usize = 3;

/// Why an LLM answer that writes a Soul or a Coupling strength is refused (step 2, spec
/// 4.5): only Souletics gives a Soul.
pub const SOUL_NOT_ADMITTED: &str = "not admitted: the Mutator does not write Souls";

/// The models a run may call, by tag. Every allow-listed tag must resolve here.
#[derive(Default)]
pub struct ModelPool {
    models: BTreeMap<String, Box<dyn ModelBackend>>,
}

impl ModelPool {
    pub fn new() -> ModelPool {
        ModelPool::default()
    }

    /// Register `backend` under `tag` (the tag Workers name in their `model` field).
    pub fn insert(&mut self, tag: &str, backend: Box<dyn ModelBackend>) {
        self.models.insert(tag.to_string(), backend);
    }

    /// The backend for `tag`.
    pub fn get(&self, tag: &str) -> Result<&dyn ModelBackend> {
        self.models
            .get(tag)
            .map(|b| b.as_ref())
            .ok_or_else(|| BrutError::Invalid(format!("no backend for model {tag:?}")))
    }

    /// Error unless every allow-listed model has a backend.
    pub fn covers(&self, limits: &CascadeLimits) -> Result<()> {
        for m in &limits.models {
            self.get(&m.tag)?;
        }

        Ok(())
    }
}

/// The models and the limits every call runs under, and the teaching section of the
/// Cascade that calls.
#[derive(Clone, Copy)]
struct Ctx<'a> {
    pool: &'a ModelPool,
    limits: &'a CascadeLimits,
    /// The calling Cascade's teaching section (step 2, spec 10.2): it ends every system
    /// prompt the Cascade sends, to solve and to judge. None without one.
    upbringing: Option<&'a str>,
}

/// Model usage of one evaluation, summed over calls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub calls: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Sum of the allow-list's fixed per-call costs (fitness units).
    pub model_cost: f64,
}

impl Usage {
    fn add(&mut self, other: &Usage) {
        self.calls += other.calls;
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.model_cost += other.model_cost;
    }
}

/// The request for one Worker call: its system prompt (plus the fixed self-check text
/// when that Trait is on) and its Traits as generation limits.
pub fn worker_request(w: &Worker, prompt: String, seed: u64) -> GenRequest {
    worker_request_with(w, prompt, seed, SELF_CHECK_TEXT)
}

/// `worker_request` for a domain whose self-check text is `check_text`.
pub fn worker_request_with(w: &Worker, prompt: String, seed: u64, check_text: &str) -> GenRequest {
    let mut system = w.system_prompt();

    if w.traits.self_check {
        system.push_str("## check\n");
        system.push_str(check_text);
        system.push('\n');
    }

    GenRequest {
        system,
        prompt,
        max_tokens: w.traits.answer_tokens,
        temperature: w.traits.temperature,
        seed,
        think_tokens: w.traits.think_tokens,
        phases: None,
    }
}

/// One Worker call with usage accounting. Every attempted call counts against the budget
/// and pays its model's fixed price, also when the backend fails, since a failed request
/// may already have used compute; tokens count only for an answer. A model without a
/// backend is not attempted and gives None. `check_text` is the domain's self-check text.
///
/// `phases` are the Worker's phases with the readout applied, in a Run with Souls whose
/// simulator setting is on (step 2, spec 11.1). The request carries them only when the
/// backend reads phases, so a real model never gets them. A teaching section in `ctx` ends
/// the system prompt (spec 10.2).
fn call_worker(
    ctx: Ctx,
    w: &Worker,
    prompt: String,
    seed: u64,
    check_text: &str,
    phases: Option<&[u16]>,
    usage: &mut Usage,
) -> Option<GenResponse> {
    let backend = ctx.pool.get(&w.model).ok()?;
    let cost = ctx
        .limits
        .model(&w.model)
        .map(|m| m.cost_per_call)
        .unwrap_or(0.0);

    usage.calls += 1;
    usage.model_cost += cost;

    let mut req = worker_request_with(w, prompt, seed, check_text);

    if let Some(text) = ctx.upbringing {
        req.system.push_str(&section_block(text));
    }

    if backend.reads_phases() {
        req.phases = phases.map(<[u16]>::to_vec);
    }

    let resp = backend.generate(&req).ok()?;

    usage.prompt_tokens += u64::from(resp.prompt_tokens);
    usage.completion_tokens += u64::from(resp.completion_tokens);

    Some(resp)
}

/// What one judge call gave.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Judgment {
    /// No reply at all: the backend failed or has no such model. Not the judge's doing.
    NoReply,
    /// A reply without a usable verdict: no exact VERDICT line, or cut off.
    NoVerdict,
    /// The verdict; true when `a` won.
    Verdict(bool),
}

/// Ask Worker `w` which answer of the pair `(a, b)` is the better one on `task`. The
/// order shown is drawn from `seed`, so a judge cannot learn a position bias.
fn judge_in<D: Domain>(
    ctx: Ctx,
    domain: &D,
    w: &Worker,
    task: &D::Task,
    (a, b): (&D::Answer, &D::Answer),
    seed: u64,
    usage: &mut Usage,
) -> Judgment {
    judge_phased(ctx, domain, w, task, (a, b), seed, None, usage).0
}

/// `judge_in` with the judge's phases on the request (step 2, spec 11.1), as a Stage vote
/// sends it. Also returns whether the phases decided the reply (`soul_decided`).
#[allow(clippy::too_many_arguments)]
fn judge_phased<D: Domain>(
    ctx: Ctx,
    domain: &D,
    w: &Worker,
    task: &D::Task,
    (a, b): (&D::Answer, &D::Answer),
    seed: u64,
    phases: Option<&[u16]>,
    usage: &mut Usage,
) -> (Judgment, bool) {
    let swap = ChaCha8Rng::seed_from_u64(derive_seed(seed, "order")).random_bool(0.5);
    let (first, second) = if swap { (b, a) } else { (a, b) };
    let prompt = domain.judge_prompt(task, first, second);
    let check = domain.self_check_text();
    let Some(resp) = call_worker(ctx, w, prompt, seed, check, phases, usage) else {
        return (Judgment::NoReply, false);
    };
    let decided = resp.soul_decided;

    if resp.stop_reason != StopReason::Stop {
        return (Judgment::NoVerdict, decided);
    }

    let judgment = match parse_verdict(&resp.text) {
        Some(Verdict::A) => Judgment::Verdict(!swap),
        Some(Verdict::B) => Judgment::Verdict(swap),
        None => Judgment::NoVerdict,
    };

    (judgment, decided)
}

/// Upper bound on calls one stage of `width` Workers makes on one puzzle: every Worker
/// answers, then every Worker judges every pair of distinct answers.
fn stage_call_bound(width: usize) -> u64 {
    let w = width as u64;

    w + w * (w * w.saturating_sub(1) / 2)
}

/// Upper bound on calls of one Cascade on one puzzle.
pub fn cascade_call_bound(c: &Cascade) -> u64 {
    c.stages
        .iter()
        .map(|s| stage_call_bound(s.workers.len()))
        .sum()
}

/// The result of one Cascade on one task.
#[derive(Clone, Debug, PartialEq)]
pub struct CascadeRun<A = Grid> {
    /// The last answer the Cascade chose (the residual path keeps an earlier one when a
    /// later stage produces nothing usable).
    pub answer: Option<A>,
    pub usage: Usage,
    /// The authors of the answer (step 2, spec 7.7): the Workers of the Stage that chose
    /// it whose own answer it was, by their index over the Stages. Empty without an answer.
    pub authors: Vec<usize>,
    /// The Stage votes tallied by amplitudes, in a Run with Souls (spec 7.7), and the
    /// replies whose phases decided a simulated model's choice.
    pub votes: VoteTally,
    /// The Observations of the Stage votes (step 2, spec 8.2), in log order: (observer,
    /// observed), each a Worker index over the Stages. Empty without Souls, and when the
    /// votes observe nothing.
    pub observed: Vec<(usize, usize)>,
}

/// What the Souls give one run of a Cascade on one task, frozen for the generation (step 2,
/// spec 7.2 and 8.4): the current Soul of every Worker in Stage order, the task's Family
/// strengths, and the settings that read them.
#[derive(Clone, Copy, Debug)]
pub struct RunSouls<'a> {
    pub souls: &'a [PhaseState],
    pub strengths: &'a [f64],
    pub readout: Readout,
    /// The complex Stage vote (spec 12.2); off, a vote counts its verdicts as today.
    pub vote: bool,
    /// The phases reach the simulated models (spec 11.1, 12.2): every solve and Stage-vote
    /// request carries its Worker's phases, readout applied, to a backend that reads them.
    pub simulator: bool,
    /// The judgments of the Stage votes are Observations (spec 8.2): on the search panel
    /// with `observe_vote` on, never on the select panel.
    pub observe: bool,
    /// The Cascade's teaching section (spec 10.2): it ends every system prompt of the run.
    /// None without one.
    pub upbringing: Option<&'a str>,
}

/// The Stage votes a Cascade tallied by amplitudes (step 2, spec 7.7): its votes, those
/// whose winner differs from the count's, those that the tie rule decided, and the sum of
/// their coherence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VoteTally {
    pub complex: u64,
    pub flipped: u64,
    pub tied: u64,
    pub coherence_sum: f64,
    /// Replies in which the phases decided a simulated model's choice (spec 7.7, 11.1).
    #[serde(default)]
    pub sim_decided: u64,
}

impl VoteTally {
    /// Count one vote tallied by amplitudes.
    fn count(&mut self, vote: &ArrayVote) {
        self.complex += 1;
        self.flipped += u64::from(vote.winner != vote.count_winner);
        self.tied += u64::from(vote.tied);
        self.coherence_sum += vote.coherence;
    }

    /// Add the votes of `other`.
    fn add(&mut self, other: &VoteTally) {
        self.complex += other.complex;
        self.flipped += other.flipped;
        self.tied += other.tied;
        self.coherence_sum += other.coherence_sum;
        self.sim_decided += other.sim_decided;
    }
}

/// Run Cascade `c` on puzzle `p`. Stage k sees the task and the answers chosen by stages
/// 0..k. When a stage's Workers disagree, they judge each other's answers blind, pair by
/// pair, and the answer with the most wins goes on (ties: the earliest Worker's answer).
pub fn run_cascade(
    pool: &ModelPool,
    limits: &CascadeLimits,
    c: &Cascade,
    p: &Puzzle,
    seed: u64,
) -> CascadeRun {
    run_cascade_in(&GridDomain, pool, limits, c, p, seed)
}

/// `run_cascade` on a task of `domain`, which renders the prompts and reads the answers.
pub fn run_cascade_in<D: Domain>(
    domain: &D,
    pool: &ModelPool,
    limits: &CascadeLimits,
    c: &Cascade,
    task: &D::Task,
    seed: u64,
) -> CascadeRun<D::Answer> {
    run_cascade_with(domain, pool, limits, c, task, seed, None)
}

/// `run_cascade_in` with the Souls of a Run with Souls (step 2, spec 7). With the complex
/// vote on, a Stage vote tallies amplitudes: the verdicts of each judge weigh with its vote
/// phasor, and the answer with the largest Π goes on (`soul::array_vote`). The run names the
/// authors of its answer and tallies its complex votes. None runs as without Souls.
///
/// With the simulator setting on, every solve and Stage-vote request carries its Worker's
/// phases, readout applied, to a backend that reads them (spec 11.1); the tally counts the
/// replies whose phases decided. With `observe` on, the run records the Observations of its
/// Stage votes (spec 8.2): a judgment that got a reply is an Observation by the judge of
/// every author of both answers other than itself. A teaching section ends every system
/// prompt of the run (spec 10.2).
pub fn run_cascade_with<D: Domain>(
    domain: &D,
    pool: &ModelPool,
    limits: &CascadeLimits,
    c: &Cascade,
    task: &D::Task,
    seed: u64,
    souls: Option<&RunSouls<'_>>,
) -> CascadeRun<D::Answer> {
    let ctx = Ctx {
        pool,
        limits,
        upbringing: souls.and_then(|r| r.upbringing),
    };
    let check = domain.self_check_text();
    let mut usage = Usage::default();
    let mut chosen: Vec<D::Answer> = Vec::new();
    let mut authors: Vec<usize> = Vec::new();
    let mut votes = VoteTally::default();
    let mut observed = Vec::new();
    // The phases each Worker's requests carry, readout applied (spec 11.1).
    let phases: Vec<Option<Vec<u16>>> = (0..c.worker_count())
        .map(|w| {
            souls
                .filter(|r| r.simulator)
                .and_then(|r| Some(readout_phases(r.souls.get(w)?, r.readout)))
        })
        .collect();
    // The index of the Stage's first Worker over all Stages.
    let mut first = 0;

    for (k, stage) in c.stages.iter().enumerate() {
        let prompt = domain.solve_prompt(task, &chosen);
        let mut distinct: Vec<D::Answer> = Vec::new();
        // The Workers whose own answer each distinct answer is (spec 7.7).
        let mut by: Vec<Vec<usize>> = Vec::new();

        for (j, w) in stage.workers.iter().enumerate() {
            let s = derive_seed(seed, &format!("stage{k}/worker{j}"));
            let own = phases[first + j].as_deref();
            let resp = call_worker(ctx, w, prompt.clone(), s, check, own, &mut usage);

            votes.sim_decided += u64::from(resp.as_ref().is_some_and(|r| r.soul_decided));

            let answer = resp
                .filter(|r| r.stop_reason == StopReason::Stop)
                .and_then(|r| domain.parse_answer(task, &r.text));

            if let Some(a) = answer {
                match distinct.iter().position(|d| d == &a) {
                    Some(x) => by[x].push(first + j),
                    None => {
                        distinct.push(a);
                        by.push(vec![first + j]);
                    }
                }
            }
        }

        // With Souls and the complex vote on, each judge votes with its phasor v_j =
        // Z(P_j, t), from its current Soul and the task's strengths (spec 7.2).
        let phasors: Option<Vec<Phasor>> = souls.filter(|r| r.vote).map(|r| {
            (first..first + stage.workers.len())
                .map(|w| {
                    r.souls.get(w).map_or(Phasor::default(), |p| {
                        vote_phasor(p, r.strengths, r.readout)
                    })
                })
                .collect()
        });
        let with = StageSouls {
            phasors: phasors.as_deref(),
            phases: &phases[first..first + stage.workers.len()],
            observe: souls.filter(|r| r.observe).map(|_| (by.as_slice(), first)),
        };
        let winner = match distinct.len() {
            0 => None,
            1 => Some(0),
            _ => {
                let vote = stage_vote(
                    ctx,
                    domain,
                    &stage.workers,
                    task,
                    &distinct,
                    derive_seed(seed, &format!("vote{k}")),
                    &mut usage,
                    &with,
                );

                if let Some(array) = &vote.array {
                    votes.count(array);
                }

                votes.sim_decided += vote.decided;
                observed.extend(vote.observed);

                Some(vote.winner)
            }
        };

        if let Some(i) = winner
            && let Some(a) = distinct.get(i)
        {
            chosen.push(a.clone());
            authors = by[i].clone();
        }

        first += stage.workers.len();
    }

    CascadeRun {
        answer: chosen.last().cloned(),
        usage,
        authors,
        votes,
        observed,
    }
}

/// What the Souls give one Stage vote (step 2): the judges' vote phasors (spec 7.2), the
/// phases each judge's requests carry (spec 11.1), and, when its judgments are
/// Observations, the authors of each answer and the index of the Stage's first Worker,
/// both over the Stages (spec 8.2). Without Souls, none of them.
struct StageSouls<'s> {
    phasors: Option<&'s [Phasor]>,
    phases: &'s [Option<Vec<u16>>],
    observe: Option<(&'s [Vec<usize>], usize)>,
}

/// The Souls a Cascade's Workers answer a panel with in a Run with Souls, frozen for the
/// generation (step 2, spec 8.4 and 10.1): those of the panel's first task and those of
/// every later task, and the teaching section that ends each of its system prompts (spec
/// 10.2). A newborn answers its first search task with its birth Souls and the others with
/// its imprinted Souls; the select phase uses the current Souls throughout.
#[derive(Clone, Copy, Debug)]
struct PanelSouls<'s> {
    first: &'s [PhaseState],
    rest: &'s [PhaseState],
    upbringing: Option<&'s str>,
}

/// What a Stage vote gave: the winner, the vote by amplitudes if it was one, the replies
/// whose phases decided, and the Observations of its judgments in log order.
struct VoteOutcome {
    winner: usize,
    array: Option<ArrayVote>,
    decided: u64,
    observed: Vec<(usize, usize)>,
}

/// Blind vote among a stage's distinct answers: every Worker judges every pair. Returns
/// the index of the answer with the most pairwise wins, ties to the lowest index.
///
/// With phasors, one per Worker, the same judgments are tallied by amplitudes instead
/// (step 2, spec 7.3 and 7.4): the answer with the largest Π goes on, and the vote comes
/// back with its Array pattern. The prompts, seeds and calls stay as they are. A judgment
/// that got a reply, with or without a verdict, is an Observation by its judge of every
/// author of x and then of y other than itself, when the vote observes (spec 8.2); the
/// pairs (x, y) and the judges go in order.
#[allow(clippy::too_many_arguments)]
fn stage_vote<D: Domain>(
    ctx: Ctx,
    domain: &D,
    workers: &[Worker],
    task: &D::Task,
    answers: &[D::Answer],
    seed: u64,
    usage: &mut Usage,
    with: &StageSouls<'_>,
) -> VoteOutcome {
    // The verdicts of each judge for each answer.
    let mut verdicts = vec![vec![0u32; answers.len()]; workers.len()];
    let mut decided = 0;
    let mut observed = Vec::new();

    for (x, ax) in answers.iter().enumerate() {
        for (y, ay) in answers.iter().enumerate().skip(x + 1) {
            for (j, w) in workers.iter().enumerate() {
                let s = derive_seed(seed, &format!("{x}/{y}/{j}"));
                let phases = with.phases.get(j).and_then(Option::as_deref);
                let (judgment, by_phase) =
                    judge_phased(ctx, domain, w, task, (ax, ay), s, phases, usage);

                decided += u64::from(by_phase);

                match judgment {
                    Judgment::Verdict(true) => verdicts[j][x] += 1,
                    Judgment::Verdict(false) => verdicts[j][y] += 1,
                    Judgment::NoReply | Judgment::NoVerdict => {}
                }

                if judgment != Judgment::NoReply
                    && let Some((by, first)) = with.observe
                {
                    let judge = first + j;
                    let authors = by.get(x).into_iter().chain(by.get(y)).flatten();

                    observed.extend(authors.filter(|&&t| t != judge).map(|&t| (judge, t)));
                }
            }
        }
    }

    let (winner, array) = match with.phasors {
        Some(p) => {
            let vote = array_vote(&verdicts, p);

            (vote.winner, Some(vote))
        }
        None => {
            let wins: Vec<u32> = (0..answers.len())
                .map(|x| verdicts.iter().map(|v| v[x]).sum())
                .collect();

            (count_winner(&wins), None)
        }
    };

    VoteOutcome {
        winner,
        array,
        decided,
        observed,
    }
}

/// Weighted Bradley-Terry strengths by the MM algorithm of Hunter (2004, Annals of
/// Statistics 32(1), 384-406). `games` holds (winner, loser, weight). Every player also
/// plays one virtual win and one virtual loss against a fixed opponent of strength 1, so
/// the estimate exists for every player and shrinks towards 1 with little evidence.
/// Returns ln(strength) per player.
pub fn bradley_terry(players: usize, games: &[(usize, usize, f64)]) -> Vec<f64> {
    let mut wins = vec![1.0f64; players];
    // Ordered map: the float sums below must run in the same order on every run.
    let mut pair: BTreeMap<(usize, usize), f64> = BTreeMap::new();

    for &(w, l, weight) in games {
        if w >= players || l >= players || w == l || weight <= 0.0 || !weight.is_finite() {
            continue;
        }

        wins[w] += weight;
        *pair.entry((w.min(l), w.max(l))).or_insert(0.0) += weight;
    }

    let mut p = vec![1.0f64; players];

    for _ in 0..1000 {
        let mut denom: Vec<f64> = p.iter().map(|&pi| 2.0 / (pi + 1.0)).collect();

        for (&(i, j), &n) in &pair {
            let d = n / (p[i] + p[j]);

            denom[i] += d;
            denom[j] += d;
        }

        let next: Vec<f64> = wins.iter().zip(&denom).map(|(w, d)| w / d).collect();
        let change = next
            .iter()
            .zip(&p)
            .map(|(a, b)| (a.ln() - b.ln()).abs())
            .fold(0.0, f64::max);

        p = next;

        if change < 1e-10 {
            break;
        }
    }

    p.iter().map(|x| x.ln()).collect()
}

/// Darwin-Gödel-Machine selection weights (Zhang et al., ICLR 2026, appendix C.2):
/// w_i = sigmoid(lambda (alpha_i - alpha_0)) / (1 + n_i), where alpha_i is the score and
/// n_i the number of archived children. Members with a perfect score (alpha_i >= 1) are
/// not eligible and get weight 0.
pub fn dgm_weights(scores: &[f64], children: &[usize], lambda: f64, alpha0: f64) -> Vec<f64> {
    scores
        .iter()
        .zip(children)
        .map(|(&a, &n)| {
            if a >= 1.0 || !a.is_finite() {
                0.0
            } else {
                let s = 1.0 / (1.0 + (-lambda * (a - alpha0)).exp());

                s / (1.0 + n as f64)
            }
        })
        .collect()
}

/// One categorical draw from `weights`; uniform when they are all zero. None when empty.
fn draw(weights: &[f64], rng: &mut ChaCha8Rng) -> Option<usize> {
    if weights.is_empty() {
        return None;
    }

    let total: f64 = weights.iter().sum();

    if total <= 0.0 || !total.is_finite() {
        return Some(rng.random_range(0..weights.len()));
    }

    let mut x = rng.random::<f64>() * total;

    for (i, &w) in weights.iter().enumerate() {
        if x < w {
            return Some(i);
        }

        x -= w;
    }

    Some(weights.len() - 1)
}

/// A random one of the `candidates` with the fewest assignments so far, or None when
/// there is no candidate.
fn least_used(candidates: &[usize], assigned: &[usize], rng: &mut ChaCha8Rng) -> Option<usize> {
    let fewest = candidates.iter().map(|&j| assigned[j]).min()?;
    let tied: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|&j| assigned[j] == fewest)
        .collect();

    tied.choose(rng).copied()
}

/// Source of new Cascades written by a language model (normally Claude).
pub trait Mutator: Sync {
    /// The answer to `req`, or None when there is none yet (for example a request still
    /// waiting for an answer file).
    fn answer(&self, req: &GenRequest) -> Option<GenResponse>;
}

/// Stable key of a request: the first 16 hex of the SHA-256 of its JSON. Recorded
/// answers are filed under this key (ADR 0002).
pub fn request_key(req: &GenRequest) -> String {
    let json = serde_json::to_string(req).unwrap_or_default();

    sha256_hex(json.as_bytes())[..16].to_string()
}

/// Recorded answers keyed by `request_key`. A run replays them instead of calling a
/// model, so it stays reproducible (ADR 0002), and a missing answer pauses the run until
/// someone (a Claude session, a human) writes it.
#[derive(Clone, Debug, Default)]
pub struct AnswerBook {
    answers: BTreeMap<String, String>,
}

impl AnswerBook {
    pub fn new() -> AnswerBook {
        AnswerBook::default()
    }

    /// File `text` as the answer to the request with key `key`.
    pub fn insert(&mut self, key: &str, text: String) {
        self.answers.insert(key.to_string(), text);
    }

    pub fn len(&self) -> usize {
        self.answers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.answers.is_empty()
    }
}

impl Mutator for AnswerBook {
    fn answer(&self, req: &GenRequest) -> Option<GenResponse> {
        let text = self.answers.get(&request_key(req))?.clone();

        Some(GenResponse {
            prompt_tokens: approx_tokens(&req.system) + approx_tokens(&req.prompt),
            completion_tokens: approx_tokens(&text),
            text,
            stop_reason: StopReason::Stop,
            backend: "answer-book".to_string(),
            soul_decided: false,
        })
    }
}

/// Any model backend as a live Mutator (for the Broker of ADR 0001, once it exists).
pub struct LiveMutator<'a>(pub &'a dyn ModelBackend);

impl Mutator for LiveMutator<'_> {
    fn answer(&self, req: &GenRequest) -> Option<GenResponse> {
        self.0.generate(req).ok()
    }
}

/// Settings of one breeding run. Recorded at the start; on resume only `generations` and
/// `max_calls` may change, everything else defines the experiment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BreedConfig {
    pub seed: u64,
    pub generations: usize,
    pub children_per_generation: usize,
    /// Size of the public search panel; fitness is measured here.
    pub search_puzzles: usize,
    /// Size of the select panel, unseen during search; only finalists and seeds meet it.
    pub select_puzzles: usize,
    /// Share of random children made by crossover rather than mutation.
    pub crossover_rate: f64,
    /// Opponents each new Cascade is compared with, and puzzles per comparison.
    pub peer_opponents: usize,
    pub peer_puzzles_per_pair: usize,
    /// Weight of judging skill in fitness; the rest is the solved rate (Q13).
    pub judge_weight: f64,
    /// Fitness charged per 1000 completion tokens per puzzle, thinking included (Q19).
    pub token_cost_per_1k: f64,
    /// DGM selection sharpness and midpoint (paper values 10 and 0.5).
    pub dgm_lambda: f64,
    pub dgm_alpha0: f64,
    /// Every `llm_every`-th generation is an LLM round (0: never), with up to
    /// `llm_children` children written by the Mutator.
    pub llm_every: usize,
    pub llm_children: usize,
    /// Pause the run when an LLM answer is missing, instead of falling back to random.
    pub pause_for_llm: bool,
    /// Hard cap on Worker calls over the whole run, select panel included (Q16).
    pub max_calls: u64,
    /// Best bred (non-seed) members re-scored on the select panel at the end.
    pub finalists: usize,
    pub limits: CascadeLimits,
    /// The task family (step 1). Grid runs leave it out, so their configs and fixed
    /// hashes are unchanged.
    #[serde(default, skip_serializing_if = "DomainConfig::is_grid")]
    pub domain: DomainConfig,
    /// How parents are drawn (step 1). DGM selection is left out, so configs and fixed
    /// hashes of earlier runs are unchanged.
    #[serde(default, skip_serializing_if = "ParentSelection::is_dgm")]
    pub selection: ParentSelection,
    /// Settings of the lexicase and tournament strategies; left out at their defaults.
    #[serde(default, skip_serializing_if = "SelectionParams::is_default")]
    pub selection_params: SelectionParams,
    /// The Soul and its settings (step 2, spec 12.2); None, the default, is a Run without
    /// Souls. It is left out of the JSON then, so configs and fixed hashes of earlier runs
    /// are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soul: Option<SoulConfig>,
    /// Experience and imagination (arm I, `.scratch/seelen-modell/spec-imagination.md`,
    /// sections 3.1 to 3.3): learned values per Worker and family, imagined draws of the
    /// section operators, and learning from each newborn's real effect. It needs no Soul.
    /// None, the default, is a Run without them; it is left out of the JSON then, so configs
    /// and fixed hashes of earlier runs are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imagination: Option<ImaginationConfig>,
    /// Changing goals (arm G, spec 3.4): the tasks that count for lexicase selection and for
    /// the real effect of imagination change every `length` generations. None, the default,
    /// is a Run whose goal is always every task; it is left out of the JSON then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goals: Option<GoalsConfig>,
}

/// How a generation draws its parents and crossover partners.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParentSelection {
    /// Darwin-Gödel-Machine weights; a partner is the best rated of a few draws (Q2).
    #[default]
    Dgm,
    /// The control arm of step 1: every member weighs the same, and a partner is the
    /// first valid draw, whatever its peer rating.
    Uniform,
    /// Epsilon-lexicase selection over the search tasks (see `selection.rs`): the
    /// candidates are filtered task by task in a fresh random order, so the aggregate
    /// fitness plays no part. A partner is the first valid draw.
    Lexicase,
    /// Tournament selection: the best-ranked of `selection_params.tournament_size` uniform
    /// draws. A partner is the first valid draw.
    Tournament,
}

impl ParentSelection {
    pub fn is_dgm(&self) -> bool {
        matches!(self, ParentSelection::Dgm)
    }

    /// The name used on command lines and in reports.
    pub fn name(&self) -> &'static str {
        match self {
            ParentSelection::Dgm => "dgm",
            ParentSelection::Uniform => "uniform",
            ParentSelection::Lexicase => "lexicase",
            ParentSelection::Tournament => "tournament",
        }
    }

    pub fn parse(name: &str) -> Option<ParentSelection> {
        [
            ParentSelection::Dgm,
            ParentSelection::Uniform,
            ParentSelection::Lexicase,
            ParentSelection::Tournament,
        ]
        .into_iter()
        .find(|s| s.name() == name)
    }
}

/// How one generation draws parents: the strategy with what it needs, computed once from
/// the archive.
pub(crate) struct Selector {
    kind: ParentSelection,
    params: SelectionParams,
    /// Dgm and Uniform: the weight of each member.
    weights: Vec<f64>,
    /// Lexicase: the value of each member on each search task (0 where it gave no answer).
    scores: Vec<Vec<f64>>,
    /// Tournament: each member's position in the ranking, 0 for the best.
    rank_of: Vec<usize>,
    members: usize,
    /// Seed for the estimate of the lexicase parent distribution that the logged
    /// observables use; the generation's own `rng` is never touched by it.
    observe_seed: u64,
}

/// Selections used to estimate the lexicase parent distribution for the logged observables.
const LEXICASE_OBSERVE_DRAWS: usize = 256;

impl Selector {
    /// One parent. Dgm and Uniform consume the `rng` exactly as before this type existed.
    pub(crate) fn pick(&self, rng: &mut ChaCha8Rng) -> Option<usize> {
        match self.kind {
            ParentSelection::Dgm | ParentSelection::Uniform => draw(&self.weights, rng),
            ParentSelection::Lexicase => {
                let pool: Vec<usize> = (0..self.members).collect();

                lexicase_pick(&self.scores, &pool, self.params.lexicase_epsilon, rng)
            }
            ParentSelection::Tournament => {
                tournament_pick(&self.rank_of, self.params.tournament_size, rng)
            }
        }
    }

    /// The chance of each member to be drawn as a parent, as the observables log it:
    /// exact for Dgm, Uniform and Tournament, an estimate from a fixed seed for Lexicase.
    pub(crate) fn observed_weights(&self) -> Vec<f64> {
        match self.kind {
            ParentSelection::Dgm | ParentSelection::Uniform => self.weights.clone(),
            ParentSelection::Tournament => tournament_weights(&self.rank_of, self.params.tournament_size),
            ParentSelection::Lexicase => {
                let mut rng = ChaCha8Rng::seed_from_u64(self.observe_seed);

                lexicase_weights(
                    &self.scores,
                    self.params.lexicase_epsilon,
                    LEXICASE_OBSERVE_DRAWS,
                    &mut rng,
                )
            }
        }
    }
}

impl Default for BreedConfig {
    fn default() -> Self {
        BreedConfig {
            seed: 1,
            generations: 10,
            children_per_generation: 8,
            search_puzzles: 40,
            select_puzzles: 40,
            crossover_rate: 0.5,
            peer_opponents: 3,
            peer_puzzles_per_pair: 2,
            judge_weight: 0.2,
            token_cost_per_1k: 0.01,
            dgm_lambda: 10.0,
            dgm_alpha0: 0.5,
            llm_every: 0,
            llm_children: 2,
            pause_for_llm: true,
            max_calls: 20_000,
            finalists: 3,
            limits: CascadeLimits::default(),
            domain: DomainConfig::Grid,
            selection: ParentSelection::Dgm,
            selection_params: SelectionParams::default(),
            soul: None,
            imagination: None,
            goals: None,
        }
    }
}

impl BreedConfig {
    /// Reject settings that cannot give a meaningful run.
    pub fn check(&self) -> Result<()> {
        let bad = |m: &str| Err(BrutError::Invalid(format!("breed config: {m}")));

        if self.children_per_generation == 0 || self.search_puzzles == 0 {
            return bad("children_per_generation and search_puzzles must be at least 1");
        }

        if !(0.0..=1.0).contains(&self.crossover_rate) || !(0.0..=1.0).contains(&self.judge_weight)
        {
            return bad("crossover_rate and judge_weight must lie in [0, 1]");
        }

        if !(self.token_cost_per_1k >= 0.0 && self.token_cost_per_1k.is_finite()) {
            return bad("token_cost_per_1k must be finite and not negative");
        }

        if !(self.dgm_lambda > 0.0 && self.dgm_lambda.is_finite() && self.dgm_alpha0.is_finite()) {
            return bad("dgm_lambda must be positive and dgm_alpha0 finite");
        }

        if let Err(e) = self.selection_params.check() {
            return bad(&e);
        }

        if self.limits.models.is_empty() {
            return bad("the model allow-list is empty");
        }

        // A negative price would reward calls; a huge one would overflow the cost sums.
        if self
            .limits
            .models
            .iter()
            .any(|m| !(0.0..=1.0).contains(&m.cost_per_call))
        {
            return bad("every model's cost_per_call must lie in [0, 1]");
        }

        if let Some(soul) = &self.soul {
            soul.check()?;
        }

        if let Some(imagination) = &self.imagination {
            imagination.check()?;
        }

        if let Some(goals) = &self.goals {
            goals.check()?;
        }

        Ok(())
    }

    /// Hash of the settings that define the experiment (all but `generations` and
    /// `max_calls`), so a resume with changed settings is refused.
    pub fn fixed_hash(&self) -> String {
        let mut fixed = self.clone();

        fixed.generations = 0;
        fixed.max_calls = 0;

        sha256_hex(serde_json::to_string(&fixed).unwrap_or_default().as_bytes())
    }
}

/// One Cascade in the archive with everything measured about it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Member<A = Grid> {
    pub id: String,
    pub cascade: Cascade,
    pub parents: Vec<String>,
    pub op: Op,
    pub generation: usize,
    /// Answer per search-panel task (None: nothing usable).
    pub answers: Vec<Option<A>>,
    pub solved: usize,
    /// Summed graded value over the search panel. None for a domain that is not graded
    /// (grids): there fitness uses the solved count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<f64>,
    pub usage: Usage,
    /// Archived children (the n_i of DGM selection).
    pub children: usize,
    pub judge_correct: u32,
    pub judge_wrong: u32,
    pub fitness: f64,
    pub rating: f64,
    /// Where each Worker's birth Soul came from, in Stage order (step 2, spec 9 and 13.2).
    /// None in a Run without Souls, and then left out of the JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soul_sources: Option<Vec<SoulSource>>,
    /// The imprinted Soul of each Worker, in Stage order (spec 4.3, 10.1): its birth Soul
    /// moved toward the current Souls of its source Workers after the first search task.
    /// A seed's is its birth Soul. None in a Run without Souls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imprinted: Option<Vec<PhaseState>>,
    /// The current Soul of each Worker, in Stage order (spec 4.3). It starts as the
    /// imprinted Soul; the birth Soul of the genome never changes. None in a Run without
    /// Souls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soul_now: Option<Vec<PhaseState>>,
    /// The teaching section written at birth from the parents' best solved examples (spec
    /// 10.2). It ends every system prompt the member sends, and it is never part of the
    /// genome. None without teaching, and when no example was taken.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teaching: Option<String>,
    /// The authors of the answer on each search task (spec 7.7). None without Souls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authors: Option<Vec<Vec<usize>>>,
    /// The tally of the Stage votes at birth, on the search panel (spec 7.7). None without
    /// Souls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub votes: Option<VoteTally>,
    /// The learned values of each Worker, in Stage order, one per family of the Domain in
    /// thousandths, after the member's own learning step at birth: what its Born event logs
    /// (`.scratch/seelen-modell/spec-imagination.md`, sections 3.1, 3.3 and 3.5). A seed's are
    /// 0. They are no gene. None without imagination.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<Vec<i16>>>,
    /// The current learned values of each Worker, in Stage order (spec 3.3). They start as
    /// `values`; at each commit the member learns the steps of its children whose Workers
    /// have their sources in it, which the `learned` event of that commit logs. None without
    /// imagination.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values_now: Option<Vec<Vec<i16>>>,
    /// What the member's learning step at birth read, as its Born event logs it (spec 3.5):
    /// when its Workers gained or lost families, and whenever imagination chose it. None
    /// otherwise, and without imagination.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect: Option<Effect>,
}

/// One Worker of one member: the member's id and the Worker's index over its Stages in
/// order.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WorkerRef {
    pub member: String,
    pub worker: usize,
}

/// A Worker whose current Soul changed at a commit (step 2, spec 8.5): its member's id,
/// its index over the member's Stages, and its new phases.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoulChange {
    pub member: String,
    pub worker: usize,
    pub phases: PhaseState,
}

/// A Worker whose learned values changed at a commit, as a source Worker of the
/// generation's newborns (`.scratch/seelen-modell/spec-imagination.md`, section 3.3): its
/// member's id, its index over the member's Stages, and its new values in thousandths.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueChange {
    pub member: String,
    pub worker: usize,
    pub values: Vec<i16>,
}

/// Where a Worker's birth Soul came from (step 2, spec 9 and 13.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoulSource {
    /// The Workers it was built from, in parent order (spec 9.1); none for a seed.
    pub sources: Vec<WorkerRef>,
    /// True when the Worker got an entirely new Soul (spec 9.4).
    pub new: bool,
    /// The origin of the Soul, Hoppe's colour: the Worker where it was new. A copied or
    /// mixed Soul keeps the origin of its first source.
    pub origin: WorkerRef,
}

impl<A> Member<A> {
    /// Judging skill with a Laplace prior: (correct + 1) / (judged + 2).
    pub fn judge_skill(&self) -> f64 {
        f64::from(self.judge_correct + 1) / f64::from(self.judge_correct + self.judge_wrong + 2)
    }
}

/// One judged pair of answers from two Cascades on one search task.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Comparison {
    pub generation: usize,
    pub puzzle: String,
    pub first: String,
    pub second: String,
    pub judge: String,
    /// None when the judge replied without a usable verdict; such a record exists only
    /// for a scored pair, and counts as a wrong judgment.
    pub winner: Option<String>,
    /// Some for a scored pair, one whose better answer the domain knows (for grids:
    /// exactly one of the two is right): whether the judge chose that answer.
    pub judge_correct: Option<bool>,
}

/// Everything the run records, in order; the binary writes these to a hash-chained log.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    RunStart {
        config: BreedConfig,
        config_hash: String,
        search: Vec<String>,
        select: Vec<String>,
    },
    Born {
        generation: usize,
        id: String,
        parents: Vec<String>,
        op: Op,
        cascade: Cascade,
        solved: usize,
        puzzles: usize,
        usage: Usage,
        /// The summed graded value, for a graded domain.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        quality: Option<f64>,
        /// The answers as text, one per search task, for a domain that logs them (grids do
        /// not). Left out when no answer has a text.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        answers: Option<Vec<Option<String>>>,
        /// The imprinted Soul of each Worker (step 2, spec 10.1, 13.2); left out in a Run
        /// without Souls.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        imprinted: Option<Vec<PhaseState>>,
        /// Where each Worker's birth Soul came from (step 2, spec 13.2); left out in a Run
        /// without Souls. The genome holds the birth Souls themselves.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        soul_sources: Option<Vec<SoulSource>>,
        /// The teaching section (spec 10.2, 13.2); left out when there is none.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        teaching: Option<String>,
        /// The authors of the answer on each search task (spec 7.7, 13.2); left out
        /// without Souls.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        authors: Option<Vec<Vec<usize>>>,
        /// The tally of the Stage votes at birth (spec 7.7, 13.2); left out without Souls.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        votes: Option<VoteTally>,
        /// The learned values of each Worker after the newborn's own learning step, in
        /// thousandths (`.scratch/seelen-modell/spec-imagination.md`, section 3.5); left out
        /// without imagination.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        values: Option<Vec<Vec<i16>>>,
        /// What that learning step read: p, s and the families each Worker gained and lost,
        /// for a newborn whose Workers gained or lost families or that imagination chose, and
        /// the chosen draw for the latter (spec 3.5). Left out otherwise, and without
        /// imagination.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        effect: Option<Effect>,
    },
    LlmExchange {
        generation: usize,
        key: String,
        request: GenRequest,
        response: Option<GenResponse>,
        accepted: bool,
        reason: String,
    },
    Compared(Comparison),
    Generation {
        generation: usize,
        best_id: String,
        best_fitness: f64,
        best_solved: usize,
        archive_size: usize,
        calls_used: u64,
    },
    Finalist(Finalist),
    /// The Workers whose current Soul changed at the commit of `generation`, after its
    /// `Generation` event, in a Run with Souls (step 2, spec 8.5, 13.3).
    Souls {
        generation: usize,
        changed: Vec<SoulChange>,
    },
    /// The Workers whose learned values changed at the commit of `generation`, in archive
    /// order, each once with its new values: the source Workers of the generation's newborns,
    /// which learn their steps in birth order (`.scratch/seelen-modell/spec-imagination.md`,
    /// section 3.3). In a Run with imagination, one after each `Generation` event and its
    /// `Souls` event.
    Learned {
        generation: usize,
        changed: Vec<ValueChange>,
    },
    /// The universe observables of a committed generation: the last event of its step.
    State(UniverseState),
}

/// A member's standing, for reports.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Ranked {
    pub id: String,
    pub generation: usize,
    pub op: Op,
    pub fitness: f64,
    pub rating: f64,
    pub solved: usize,
    pub puzzles: usize,
    pub judge_skill: f64,
    pub stages: usize,
    pub workers: usize,
    pub completion_tokens_per_puzzle: f64,
}

/// A finalist or seed scored on the select panel at the end.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Finalist {
    pub id: String,
    pub is_seed: bool,
    pub search_solved_rate: f64,
    pub select_solved: usize,
    pub select_puzzles: usize,
    /// The summed graded value on the select panel, for a graded domain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub select_quality: Option<f64>,
}

/// The end-of-run summary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub generations_completed: usize,
    pub archive_size: usize,
    pub calls_used: u64,
    pub mutator_answers_used: u64,
    pub comparisons: usize,
    pub stop_reason: String,
    pub top: Vec<Ranked>,
    pub finalists: Vec<Finalist>,
    /// Best select-panel solved rate of a finalist and of a seed Cascade. Descriptive
    /// only: the pre-registered endpoint runs on ARC in the sealed evaluator.
    pub best_finalist_select_rate: f64,
    pub best_seed_select_rate: f64,
}

/// Select-panel answers already obtained: member id, then task id.
pub type SelectAnswers<A = Grid> = BTreeMap<String, BTreeMap<String, Option<A>>>;

/// Resumable state, written after every completed generation, and before the first call
/// of a generation or select phase (then with `reserved` set).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Snapshot<A = Grid> {
    pub config_hash: String,
    pub next_generation: usize,
    pub archive: Vec<Member<A>>,
    pub comparisons: Vec<Comparison>,
    pub calls: u64,
    pub mutator_answers: u64,
    /// Worst-case calls of the generation or select phase in flight when this snapshot
    /// was written. A resume counts them as spent: their real number died with the
    /// process, so `max_calls` holds across crashes too.
    #[serde(default)]
    pub reserved: u64,
    /// Answers on the select panel so far, so a stopped select phase never pays twice. In
    /// a Run with Souls a commit drops those of every member whose current Soul it changed
    /// (spec 8.6). Empty when absent. The empty map is named directly: a plain `default`
    /// would make serde require `A: Default`, which a domain's answer type need not have.
    #[serde(default = "BTreeMap::new")]
    pub select_answers: SelectAnswers<A>,
    /// True when the select phase finished on this archive and returned its events.
    #[serde(default)]
    pub finished: bool,
}

/// What one generation step ended with.
#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    /// The generation was committed; call `step` again.
    Continue,
    /// No more generations: the reason (generations done, budget, stopped).
    Done(String),
    /// An LLM round needs answers that do not exist yet; nothing was committed.
    Pending(Vec<GenRequest>),
}

/// A child before evaluation.
struct Draft {
    cascade: Cascade,
    parents: Vec<String>,
    op: Op,
    /// The source Workers of each Worker, by the parent's place in `parents` (spec 9.1);
    /// none for a seed.
    sources: Sources,
    /// Where each Worker's birth Soul came from, once Souletics gave it (Runs with Souls).
    soul_sources: Option<Vec<SoulSource>>,
    /// The imprinted Souls and the teaching section, once Upbringing gave them (Runs with
    /// Souls; spec 10).
    imprinted: Option<Vec<PhaseState>>,
    teaching: Option<String>,
    /// The choices the preferences decided in the operator that made the child (arm W);
    /// none for a seed, a Mutator's child, or with `variation` off.
    steered: Steered,
    /// How imagination chose the child (`.scratch/seelen-modell/spec-imagination.md`,
    /// section 3.2); None when no imagining operator made it, or with `imagination` off.
    imagined: Option<Imagined>,
    /// What the child brings to its learning, once `experiences` gave it (Runs with
    /// imagination; spec 3.1 to 3.3).
    experience: Option<Experience>,
}

/// How imagination chose a child among the draws of its operator (spec 3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Imagined {
    /// The chosen draw: 1 for today's draw, up to the setting's `draws`.
    draw: usize,
    /// True when some draw that was a candidate had a prediction other than 0: an
    /// imagination choice of the `mind` group (spec 3.5).
    choice: bool,
}

/// What a child brings to its learning (spec 3.1 to 3.3): for each of its Workers, in Stage
/// order, the values it copies at birth, its source Worker, and what it gained and lost
/// against that source; and the prediction p over all of them.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Experience {
    /// The values of each Worker's source Worker when the child was drafted, or 0 for a
    /// Worker without a source.
    start: Vec<Vec<i16>>,
    /// The source Worker of each Worker that has one, as its member's place in the archive
    /// and its index over that member's Stages: the first of its sources, the Worker that
    /// passes on the birth Soul too.
    sources: Vec<Option<(usize, usize)>>,
    changes: Vec<Change>,
    p: i64,
    /// The first parent's place in the archive: the parent the real effect compares the
    /// child with. None for a seed.
    first_parent: Option<usize>,
}

impl Experience {
    /// The families gained and lost over all Workers: n of spec 3.3.
    fn families(&self) -> usize {
        self.changes.iter().map(Change::families).sum()
    }

    /// The child's learning step for the real effect `s` at the rate `rate` in thousandths
    /// (spec 3.3): one error e = s − p for the whole child, one step for its n families
    /// (`mind::step`), and every gained family of every Worker moved by the step and every
    /// lost one by minus the step, each clamped on its own. Returns the values of its Workers
    /// after the step, and the step.
    fn learn(&self, s: i8, rate: u16) -> (Vec<Vec<i16>>, i64) {
        let step = step(error(s, self.p), self.families(), rate);
        let mut values = self.start.clone();

        for (v, change) in values.iter_mut().zip(&self.changes) {
            change.learn(v, step);
        }

        (values, step)
    }

    /// What the child teaches its source Workers at the commit with the step `step` (spec
    /// 3.3): the change of each of its Workers that gained or lost families, with the
    /// source Worker that learns it.
    fn lesson(&self, step: i64) -> Lesson {
        Lesson {
            step,
            changes: self
                .sources
                .iter()
                .zip(&self.changes)
                .filter(|(_, change)| change.families() > 0)
                .filter_map(|(source, change)| Some(((*source)?, change.clone())))
                .collect(),
        }
    }
}

/// One candidate draw of an operator: the child, the sources of its Workers, and the
/// choices the preferences decided (arm W).
struct Drawn {
    cascade: Cascade,
    sources: Sources,
    steered: Steered,
}

/// What a newborn's learning step at birth gave (spec 3.3, 3.5): the values of its Workers
/// after the step, what its Born event logs of it, and what it teaches its source Workers at
/// the commit; a seed teaches nothing.
struct LearnedAtBirth {
    values: Vec<Vec<i16>>,
    effect: Option<Effect>,
    lesson: Option<Lesson>,
}

/// What one newborn teaches its source Workers at the commit (spec 3.3): its step, and for
/// each of its Workers that changed families, the source Worker (its member's place in the
/// archive and its Worker index) with the change.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Lesson {
    step: i64,
    changes: Vec<((usize, usize), Change)>,
}

/// Receives a snapshot to store before calls are made (see `Breeder::step_with`).
pub type Persist<'p, A = Grid> = dyn FnMut(&Snapshot<A>) -> Result<()> + 'p;

/// The breeding run: archive, panels, budget and the generation loop, on the tasks of
/// domain `D` with Cascades made from a gene pool.
pub struct Breeder<'a, D: Domain = GridDomain> {
    cfg: BreedConfig,
    domain: D,
    genes: GenePool,
    pool: &'a ModelPool,
    mutator: Option<&'a dyn Mutator>,
    search: Vec<D::Task>,
    select: Vec<D::Task>,
    archive: Vec<Member<D::Answer>>,
    comparisons: Vec<Comparison>,
    calls: u64,
    mutator_answers: u64,
    next_generation: usize,
    select_answers: SelectAnswers<D::Answer>,
    finished: bool,
    /// The gene id of every archived member, in archive order (spec 4.4): novelty, call
    /// seeds and the last tie-break of the ranking go by genes. Without Souls these are the
    /// ids.
    gene_ids: Vec<String>,
    /// One cycle of the goals on the search panel (`.scratch/seelen-modell/spec-imagination.md`,
    /// section 3.4); empty with `goals` off.
    goals: Vec<Goal>,
}

impl<'a> Breeder<'a, GridDomain> {
    /// A fresh run on grid puzzles. Generation 0 (the seed Cascades) is evaluated by the
    /// first `step`.
    pub fn new(
        cfg: BreedConfig,
        pool: &'a ModelPool,
        mutator: Option<&'a dyn Mutator>,
    ) -> Result<Breeder<'a>> {
        Breeder::start(cfg, GenePool::grid(), pool, mutator)
    }

    /// Continue a run on grid puzzles from `snap` (see `resume_from`).
    pub fn resume(
        cfg: BreedConfig,
        pool: &'a ModelPool,
        mutator: Option<&'a dyn Mutator>,
        snap: Snapshot,
    ) -> Result<Breeder<'a>> {
        Breeder::resume_from(cfg, GenePool::grid(), pool, mutator, snap)
    }
}

impl<'a, D: Domain> Breeder<'a, D> {
    /// A fresh run on the domain `cfg` names, with Cascades made from `genes`.
    /// Generation 0 (the seed Cascades) is evaluated by the first `step`.
    pub fn start(
        cfg: BreedConfig,
        genes: GenePool,
        pool: &'a ModelPool,
        mutator: Option<&'a dyn Mutator>,
    ) -> Result<Breeder<'a, D>> {
        cfg.check()?;

        let domain = D::from_config(&cfg)?;

        pool.covers(&cfg.limits)?;

        let seeds = adapt_seeds((genes.seeds)(), &cfg.limits)?;

        // A Soul has one phase per hypothesis family of the domain (spec 5.1).
        if cfg.soul.is_some() && domain.families().is_empty() {
            return Err(BrutError::Invalid(
                "breed config: a Run with Souls needs a domain that names its hypothesis families"
                    .to_string(),
            ));
        }

        // A Soul that steers variation needs a gene pool that names the families of a text
        // (arm W), and the fixed preference of arm WF needs the families it favours.
        if let Some(soul) = &cfg.soul
            && !soul.variation.is_off()
        {
            if genes.families_of.is_none() {
                return Err(BrutError::Invalid(
                    "breed config: soul: variation needs a gene pool that names the families of a text, and this Domain's gene pool does not"
                        .to_string(),
                ));
            }

            let names = domain.families();

            if soul.variation == Variation::Fixed
                && let Some(missing) = FIXED_PREFERRED.iter().find(|f| !names.contains(f))
            {
                return Err(BrutError::Invalid(format!(
                    "breed config: soul: variation fixed favours {}, but this Domain has no family {missing}",
                    FIXED_PREFERRED.join(", ")
                )));
            }
        }

        // Experience and imagination read the families a Worker names (spec-imagination 3.1,
        // 3.2), so they need a gene pool that names the families of a text.
        if cfg.imagination.is_some()
            && (genes.families_of.is_none() || domain.families().is_empty())
        {
            return Err(BrutError::Invalid(
                "breed config: imagination needs a gene pool that names the families of a text, and this Domain's gene pool does not"
                    .to_string(),
            ));
        }

        // Changing goals leave out the targets that name a family, and only lexicase reads
        // single tasks (spec-imagination 3.4).
        if cfg.goals.is_some()
            && (cfg.selection != ParentSelection::Lexicase
                || !matches!(cfg.domain, DomainConfig::Circuits(_)))
        {
            return Err(BrutError::Invalid(
                "breed config: goals need lexicase selection and the circuits Domain".to_string(),
            ));
        }

        // A seed brings no Soul and no Coupling strength (spec 4.4): a Run without Souls has
        // none, and in a Run with Souls Souletics gives every seed Worker its own.
        if let Some((name, _)) = seeds.iter().find(|(_, c)| c.carries_soul()) {
            let why = match cfg.soul {
                None => "this Run has no Souls",
                Some(_) => "Souletics gives the seeds their Souls",
            };

            return Err(BrutError::Invalid(format!(
                "seed {name} carries a Soul or a Coupling strength, but {why}"
            )));
        }

        let search = domain.panel(derive_seed(cfg.seed, "panel/search"), cfg.search_puzzles);
        let select = domain.panel(derive_seed(cfg.seed, "panel/select"), cfg.select_puzzles);
        let goals = match cfg.goals {
            Some(_) => {
                let strengths: Vec<Vec<f64>> = search.iter().map(|t| domain.strengths(t)).collect();

                goal_cycle(&domain.families(), &strengths)
            }
            None => Vec::new(),
        };

        Ok(Breeder {
            cfg,
            domain,
            genes,
            pool,
            mutator,
            search,
            select,
            archive: Vec::new(),
            comparisons: Vec::new(),
            calls: 0,
            mutator_answers: 0,
            next_generation: 0,
            select_answers: BTreeMap::new(),
            finished: false,
            gene_ids: Vec::new(),
            goals,
        })
    }

    /// Continue a run from `snap`. Refused when the experiment-defining settings changed.
    /// Calls the snapshot reserved for work in flight count as spent.
    pub fn resume_from(
        cfg: BreedConfig,
        genes: GenePool,
        pool: &'a ModelPool,
        mutator: Option<&'a dyn Mutator>,
        snap: Snapshot<D::Answer>,
    ) -> Result<Breeder<'a, D>> {
        if snap.config_hash != cfg.fixed_hash() {
            return Err(BrutError::Invalid(
                "resume refused: settings other than generations and max_calls changed".to_string(),
            ));
        }

        let mut b: Breeder<'a, D> = Breeder::start(cfg, genes, pool, mutator)?;

        match &b.cfg.soul {
            // A Run without Souls has no Worker with a Soul or a Coupling strength, and no
            // member with a record of Souls (spec 4.4).
            None => {
                let souled = |m: &Member<D::Answer>| {
                    m.cascade.carries_soul()
                        || m.soul_sources.is_some()
                        || m.imprinted.is_some()
                        || m.soul_now.is_some()
                        || m.teaching.is_some()
                        || m.authors.is_some()
                        || m.votes.is_some()
                };

                if let Some(m) = snap.archive.iter().find(|m| souled(m)) {
                    return Err(BrutError::Invalid(format!(
                        "resume refused: member {} carries a Soul, a Coupling strength or a record of Souls, but this Run has no Souls",
                        m.id
                    )));
                }
            }
            // In a Run with Souls every Worker has a Soul of a phase per family (spec 4.4).
            Some(soul) => {
                let n = b.domain.families().len();
                let tasks = b.search.len();

                for m in &snap.archive {
                    check_souls(m, n, tasks, soul).map_err(|why| {
                        BrutError::Invalid(format!("resume refused: member {}: {why}", m.id))
                    })?;
                }
            }
        }

        // A Run without imagination has no learned values; in a Run with it every Worker has
        // a value per family in range (spec-imagination 3.1).
        match &b.cfg.imagination {
            None => {
                let learned = |m: &Member<D::Answer>| {
                    m.values.is_some() || m.values_now.is_some() || m.effect.is_some()
                };

                if let Some(m) = snap.archive.iter().find(|m| learned(m)) {
                    return Err(BrutError::Invalid(format!(
                        "resume refused: member {} carries learned values, but this Run has no imagination",
                        m.id
                    )));
                }
            }
            Some(imagination) => {
                let n = b.domain.families().len();

                for m in &snap.archive {
                    check_values(m, n, imagination).map_err(|why| {
                        BrutError::Invalid(format!("resume refused: member {}: {why}", m.id))
                    })?;
                }
            }
        }

        b.gene_ids = snap
            .archive
            .iter()
            .map(|m| b.gene_key(&m.id, &m.cascade))
            .collect();
        b.archive = snap.archive;
        b.comparisons = snap.comparisons;
        b.calls = snap.calls.saturating_add(snap.reserved);
        b.mutator_answers = snap.mutator_answers;
        b.next_generation = snap.next_generation;
        b.select_answers = snap.select_answers;
        b.finished = snap.finished;

        Ok(b)
    }

    /// The domain this run breeds on.
    pub fn domain(&self) -> &D {
        &self.domain
    }

    /// The key a member with this id and Cascade has for novelty, call seeds and the last
    /// tie-break of the ranking: its gene id in a Run with Souls, its id otherwise. The two
    /// agree without Souls, and the id keeps every recorded Run as it was.
    fn gene_key(&self, id: &str, cascade: &Cascade) -> String {
        match self.cfg.soul {
            Some(_) => cascade.gene_id(),
            None => id.to_string(),
        }
    }

    /// Souletics for the drafts of generation `g`, in birth order (spec 9). Every Worker
    /// gets its birth Soul and Coupling strength from its sources, by the draws of its
    /// draft's own stream (spec 9.7), and the draft records where each came from. Hoppe's
    /// urn holds the archive's Workers and the Workers drawn before (spec 9.4).
    fn souletics(&self, g: usize, drafts: &mut [Draft], soul: &SoulConfig) -> Result<()> {
        let n = self.domain.families().len();
        let by_id: HashMap<&str, &Member<D::Answer>> =
            self.archive.iter().map(|m| (m.id.as_str(), m)).collect();
        let mut urn: usize = self.archive.iter().map(|m| m.cascade.worker_count()).sum();
        let broken = |what: String| BrutError::Integrity(format!("Souletics: {what}"));

        for (i, d) in drafts.iter_mut().enumerate() {
            let tag = match g {
                0 => format!("soul/seed/{i}"),
                _ => format!("soul/{g}/{i}"),
            };
            let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(self.cfg.seed, &tag));
            let parents = d
                .parents
                .iter()
                .map(|p| by_id.get(p.as_str()).copied())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| broken(format!("a parent of child {i} is not archived")))?;

            if d.sources.len() != d.cascade.worker_count() {
                return Err(broken(format!(
                    "child {i} has {} source lists for {} Workers",
                    d.sources.len(),
                    d.cascade.worker_count()
                )));
            }

            let mut births = Vec::with_capacity(d.sources.len());

            for sources in &d.sources {
                let draws = SoulDraws::draw(&mut rng, n);
                let from = sources
                    .iter()
                    .map(|s| {
                        let parent = parents.get(s.parent)?;

                        source_soul(parent, s.worker)
                    })
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| broken(format!("a source of child {i} has no Soul")))?;
                let inherited: Vec<Inheritance<'_>> = from.iter().map(|(x, _, _)| *x).collect();
                let birth = souletics(soul, &inherited, urn, &draws);

                urn += 1;
                births.push((birth, from));
            }

            for (w, (birth, _)) in d
                .cascade
                .stages
                .iter_mut()
                .flat_map(|s| &mut s.workers)
                .zip(&births)
            {
                w.traits.phase_state = Some(PhaseState(birth.phases.clone()));
                w.traits.coupling = birth.coupling;
            }

            // A new Soul's origin is its own Worker, so the record needs the id with the
            // Souls in it; a copied or mixed Soul keeps the origin of its first source.
            let id = d.cascade.id();
            let records = births
                .into_iter()
                .enumerate()
                .map(|(k, (birth, from))| {
                    let origin = match from.first() {
                        Some((_, _, origin)) if !birth.new => (*origin).clone(),
                        _ => WorkerRef {
                            member: id.clone(),
                            worker: k,
                        },
                    };

                    SoulSource {
                        sources: from.into_iter().map(|(_, at, _)| at).collect(),
                        new: birth.new,
                        origin,
                    }
                })
                .collect();

            d.soul_sources = Some(records);
        }

        Ok(())
    }

    /// Upbringing for the drafts of generation `g`, after Souletics (spec 10) [Q73 a, d].
    ///
    /// Imprinting: each Worker's imprinted Soul is its birth Soul moved toward the current
    /// Soul of each of its source Workers in turn, in parent order, by that source Worker's
    /// Coupling strength (`soul::imprint`); with `imprint` off, and for a seed, it is the
    /// birth Soul. The newborn answers its first search task with its birth Souls and the
    /// others with its imprinted Souls. Imprinting reads no answer, so it is computed here,
    /// before the evaluation, which then still runs its tasks in parallel.
    ///
    /// Teaching: with `teach` on, each draft gets the section its parents' best solved
    /// examples on the search panel give (`upbringing::teaching`).
    ///
    /// Both read the parents as committed after the previous generation and change nothing
    /// in the archive.
    fn upbringing(&self, g: usize, drafts: &mut [Draft], soul: &SoulConfig) -> Result<()> {
        let by_id: HashMap<&str, &Member<D::Answer>> =
            self.archive.iter().map(|m| (m.id.as_str(), m)).collect();
        let broken = |what: String| BrutError::Integrity(format!("Upbringing: {what}"));

        for (i, d) in drafts.iter_mut().enumerate() {
            let records = d.soul_sources.as_ref().ok_or_else(|| {
                broken(format!("child {i} of generation {g} has no Soul records"))
            })?;
            let mut imprinted_souls = Vec::with_capacity(records.len());

            for (birth, rec) in birth_souls(&d.cascade).iter().zip(records) {
                let sources = rec
                    .sources
                    .iter()
                    .map(|s| current_soul(by_id.get(s.member.as_str())?, s.worker))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        broken(format!(
                            "a source of child {i} of generation {g} has no current Soul"
                        ))
                    })?;

                imprinted_souls.push(PhaseState(imprinted(birth, &sources, soul.imprint)));
            }

            let section = if soul.teach {
                let parents = d
                    .parents
                    .iter()
                    .map(|p| Some(by_id.get(p.as_str())?.answers.as_slice()))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        broken(format!(
                            "a parent of child {i} of generation {g} is not archived"
                        ))
                    })?;

                teaching(&self.domain, &self.search, &parents)?
            } else {
                None
            };

            d.imprinted = Some(imprinted_souls);
            d.teaching = section;
        }

        Ok(())
    }

    /// The tasks of the search panel.
    pub fn search_panel(&self) -> &[D::Task] {
        &self.search
    }

    /// The tasks of the select panel (held out: no selection, mutation or recombination
    /// reads its results; only finalists, seeds and outside observers are scored on it).
    pub fn select_panel(&self) -> &[D::Task] {
        &self.select
    }

    /// The settings of this run.
    pub fn config(&self) -> &BreedConfig {
        &self.cfg
    }

    /// Archive indices best first by fitness, then peer rating, then gene id (the id in a
    /// Run without Souls): the order that finalists and reports use.
    pub fn ranking(&self) -> Vec<usize> {
        rank(&self.archive, &self.gene_ids)
    }

    /// The state after the last committed generation.
    pub fn snapshot(&self) -> Snapshot<D::Answer> {
        self.snapshot_with(0)
    }

    /// The state with `reserved` calls set aside for work about to start.
    fn snapshot_with(&self, reserved: u64) -> Snapshot<D::Answer> {
        Snapshot {
            config_hash: self.cfg.fixed_hash(),
            next_generation: self.next_generation,
            archive: self.archive.clone(),
            comparisons: self.comparisons.clone(),
            calls: self.calls,
            mutator_answers: self.mutator_answers,
            reserved,
            select_answers: self.select_answers.clone(),
            finished: self.finished,
        }
    }

    /// True when `need` more calls stay within `max_calls`.
    fn fits(&self, need: u64) -> bool {
        self.calls.saturating_add(need) <= self.cfg.max_calls
    }

    pub fn archive(&self) -> &[Member<D::Answer>] {
        &self.archive
    }

    pub fn comparisons(&self) -> &[Comparison] {
        &self.comparisons
    }

    pub fn calls_used(&self) -> u64 {
        self.calls
    }

    pub fn next_generation(&self) -> usize {
        self.next_generation
    }

    /// Calls kept back for scoring finalists and seeds on the select panel at the end:
    /// every seed, plus `finalists` Cascades of the widest shape in the archive or among
    /// `drafts` (finalists can only come from there).
    fn select_reserve(&self, drafts: &[Draft]) -> u64 {
        let seeds: u64 = adapt_seeds((self.genes.seeds)(), &self.cfg.limits)
            .unwrap_or_default()
            .iter()
            .map(|(_, c)| cascade_call_bound(c))
            .sum();
        let widest = self
            .archive
            .iter()
            .map(|m| cascade_call_bound(&m.cascade))
            .chain(drafts.iter().map(|d| cascade_call_bound(&d.cascade)))
            .max()
            .unwrap_or(0);
        let per_puzzle = seeds.saturating_add((self.cfg.finalists as u64).saturating_mul(widest));

        per_puzzle.saturating_mul(self.cfg.select_puzzles as u64)
    }

    /// `step_with` without storing the reservation (for in-memory runs and tests).
    pub fn step(&mut self, stop: &(dyn Fn() -> bool + Sync)) -> Result<(Status, Vec<Event>)> {
        self.step_with(stop, &mut |_| Ok(()))
    }

    /// Run one generation: draft children, evaluate them on the search panel, let them
    /// judge and be judged, recompute fitness, and commit. When the step is stopped, out
    /// of budget, or waiting for LLM answers, nothing is committed except the calls that
    /// were already made (they stay spent, so stop-and-resume cycles cannot exceed
    /// `max_calls`); a resume repeats exactly this generation. `stop` is polled before
    /// every puzzle and every judgment.
    ///
    /// After the budget check and before the first Worker call, `persist` receives a
    /// snapshot that reserves the generation's worst-case calls. Stored as the
    /// checkpoint, it makes a resume after a crash or kill count those calls as spent.
    ///
    /// An integrity alarm from the domain (an answer graded beyond a proven bound) ends
    /// the step with that error. As on a stop, only the calls made stay spent.
    ///
    /// A committed step's events end with one `State` event: the observables of the
    /// committed state (`observe.rs`).
    pub fn step_with(
        &mut self,
        stop: &(dyn Fn() -> bool + Sync),
        persist: &mut Persist<'_, D::Answer>,
    ) -> Result<(Status, Vec<Event>)> {
        let g = self.next_generation;

        if g > self.cfg.generations {
            return Ok((Status::Done("all generations done".to_string()), Vec::new()));
        }

        let mut events = Vec::new();
        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(self.cfg.seed, &format!("gen/{g}")));

        if g == 0 {
            let ids = |panel: &[D::Task]| -> Vec<String> {
                panel
                    .iter()
                    .map(|t| self.domain.task_id(t).to_string())
                    .collect()
            };

            events.push(Event::RunStart {
                config: self.cfg.clone(),
                config_hash: self.cfg.fixed_hash(),
                search: ids(&self.search),
                select: ids(&self.select),
            });
        }

        let (drafts, llm_events, answers_used) = if g == 0 {
            let seeds = adapt_seeds((self.genes.seeds)(), &self.cfg.limits)?
                .into_iter()
                .map(|(_, c)| Draft {
                    sources: vec![Vec::new(); c.worker_count()],
                    cascade: c,
                    parents: Vec::new(),
                    op: Op::Seed,
                    soul_sources: None,
                    imprinted: None,
                    teaching: None,
                    steered: Steered::default(),
                    imagined: None,
                    experience: None,
                })
                .collect();

            (seeds, Vec::new(), 0)
        } else {
            match self.draft_children(g, &mut rng)? {
                Drafted::Ready(d, e, n) => (d, e, n),
                Drafted::Pending(reqs, llm, random) => {
                    // Ask for answers only if the cheapest generation they can give still
                    // fits: every answer accepted as one Worker (one call per puzzle), next
                    // to the random children drafted with them. A rejected answer is
                    // replaced by a further random child, which costs no less. Otherwise
                    // the answers could not be used.
                    let puzzles = self.search.len() as u64;
                    let smallest = random
                        .iter()
                        .map(|d| cascade_call_bound(&d.cascade).saturating_mul(puzzles))
                        .fold((llm as u64).saturating_mul(puzzles), u64::saturating_add)
                        .saturating_add(self.peer_call_bound(llm + random.len()))
                        .saturating_add(self.select_reserve(&random));

                    if !self.fits(smallest) {
                        return Ok((Status::Done("call budget reached".to_string()), Vec::new()));
                    }

                    return Ok((Status::Pending(reqs), Vec::new()));
                }
            }
        };

        events.extend(llm_events);

        let mut drafts = drafts;

        if let Some(soul) = &self.cfg.soul {
            self.souletics(g, &mut drafts, soul)?;
            self.upbringing(g, &mut drafts, soul)?;
        }

        // Experience (`.scratch/seelen-modell/spec-imagination.md`, section 3.1): what each
        // child copies from its source Workers, what it gained and lost against them, and its
        // prediction. Then what the `mind` group of this generation's state line reads from
        // the children (spec 3.5): how imagination chose them, and the learned values of their
        // parents' Workers when the parents were chosen, before this commit's learning.
        if self.cfg.imagination.is_some() {
            self.experiences(&mut drafts)?;
        }

        let imagination_counts = self
            .cfg
            .imagination
            .as_ref()
            .map(|_| self.imagination_counts(&drafts));

        // The choices the preferences decided in this generation's children (arm W).
        let steered = drafts.iter().fold(Steered::default(), |mut sum, d| {
            sum.add(d.steered);
            sum
        });

        let need = drafts
            .iter()
            .map(|d| cascade_call_bound(&d.cascade).saturating_mul(self.search.len() as u64))
            .fold(0u64, u64::saturating_add)
            .saturating_add(self.peer_call_bound(drafts.len()));

        if !self.fits(need.saturating_add(self.select_reserve(&drafts))) {
            return Ok((Status::Done("call budget reached".to_string()), Vec::new()));
        }

        persist(&self.snapshot_with(need))?;

        // Evaluate the children on the search panel.
        let mut archive = self.archive.clone();
        let mut calls = self.calls;
        let mut generation_usage = Usage::default();
        let first_new = archive.len();
        let mut gene_ids = self.gene_ids.clone();

        // The Observations of the newborns' Stage votes, in log order (spec 8.4).
        let mut vote_observations: Vec<Observation> = Vec::new();
        // What each newborn teaches its source Workers at the commit, in birth order
        // (spec-imagination 3.3).
        let mut lessons: Vec<Lesson> = Vec::new();

        for d in drafts {
            let id = d.cascade.id();
            let index = archive.len();
            // Calls take their seeds from the genes (spec 15): members with the same genes
            // and other Souls make the same calls.
            let gene_id = self.gene_key(&id, &d.cascade);
            let tag = format!("search/{gene_id}");
            // A newborn answers its first search task with its birth Souls and every other
            // one with its imprinted Souls, and its teaching section ends each of its system
            // prompts (spec 8.4, 10).
            let birth = d.soul_sources.as_ref().map(|_| birth_souls(&d.cascade));
            let souls = birth.as_deref().map(|first| PanelSouls {
                first,
                rest: d.imprinted.as_deref().unwrap_or(first),
                upbringing: d.teaching.as_deref(),
            });
            let (runs, usage) = self.evaluate(&d.cascade, souls, &self.search, &tag, stop);

            calls += usage.calls;

            let Some(runs) = runs else {
                self.calls = calls;

                return Ok((Status::Done("stopped".to_string()), Vec::new()));
            };
            let mut answers = Vec::with_capacity(runs.len());
            let mut authors = Vec::with_capacity(runs.len());
            let mut votes = VoteTally::default();

            // In panel order, so the sums and the Observations do not depend on threads.
            for r in runs {
                votes.add(&r.votes);
                vote_observations.extend(r.observed.iter().map(|&(o, t)| Observation {
                    observer: (index, o),
                    observed: (index, t),
                }));
                authors.push(r.authors);
                answers.push(r.answer);
            }

            let souled = birth.is_some();
            let authors = souled.then_some(authors);
            let votes = souled.then_some(votes);
            let (solved, quality) = match tally(&self.domain, &answers, &self.search) {
                Ok(t) => t,
                Err(e) => {
                    self.calls = calls;

                    return Err(e);
                }
            };
            let artefacts: Vec<Option<String>> = answers
                .iter()
                .map(|a| a.as_ref().and_then(|a| self.domain.artefact(a)))
                .collect();
            // Its Workers learn at once from the real effect on this generation's goal, before
            // its Born event; their source Workers learn the same steps at the commit
            // (spec-imagination 3.3).
            let (values, effect) = match &d.experience {
                Some(exp) => match self.learn_at_birth(g, &d, exp, &answers) {
                    Ok(birth) => {
                        lessons.extend(birth.lesson);

                        (Some(birth.values), birth.effect)
                    }
                    Err(e) => {
                        self.calls = calls;

                        return Err(e);
                    }
                },
                None => (None, None),
            };

            generation_usage.add(&usage);

            for p in &d.parents {
                if let Some(m) = archive.iter_mut().find(|m| &m.id == p) {
                    m.children += 1;
                }
            }

            events.push(Event::Born {
                generation: g,
                id: id.clone(),
                parents: d.parents.clone(),
                op: d.op,
                cascade: d.cascade.clone(),
                solved,
                puzzles: self.search.len(),
                usage,
                quality,
                answers: artefacts.iter().any(Option::is_some).then_some(artefacts),
                imprinted: d.imprinted.clone(),
                soul_sources: d.soul_sources.clone(),
                teaching: d.teaching.clone(),
                authors: authors.clone(),
                votes,
                values: values.clone(),
                effect: effect.clone(),
            });

            // The imprinted Souls start the current Souls (spec 4.3, 10.1), and the values
            // after the newborn's own learning step its current values (spec-imagination 3.3).
            let soul_now = d.imprinted.clone();
            let values_now = values.clone();

            gene_ids.push(gene_id);

            archive.push(Member {
                id,
                cascade: d.cascade,
                parents: d.parents,
                op: d.op,
                generation: g,
                answers,
                solved,
                quality,
                usage,
                children: 0,
                judge_correct: 0,
                judge_wrong: 0,
                fitness: 0.0,
                rating: 0.0,
                soul_sources: d.soul_sources,
                imprinted: d.imprinted,
                soul_now,
                teaching: d.teaching,
                authors,
                votes,
                values,
                values_now,
                effect,
            });
        }

        // Each new Cascade meets up to peer_opponents unrelated members on puzzles where
        // their answers differ (it may meet none), and new members are the preferred
        // judges (Q12).
        let new: Vec<usize> = (first_new..archive.len()).collect();
        let (peer, peer_usage) = self.peer_round(&archive, &new, g, &mut rng, stop);

        calls += peer_usage.calls;
        generation_usage.add(&peer_usage);

        let Some((new_comparisons, peer_observations)) = peer else {
            self.calls = calls;

            return Ok((Status::Done("stopped".to_string()), Vec::new()));
        };

        let mut comparisons = self.comparisons.clone();

        for c in &new_comparisons {
            events.push(Event::Compared(c.clone()));
        }

        comparisons.extend(new_comparisons);
        refresh(&mut archive, &comparisons, &self.cfg, self.search.len());

        let best = rank(&archive, &gene_ids)
            .first()
            .map(|&i| archive[i].clone());

        if let Some(b) = best {
            events.push(Event::Generation {
                generation: g,
                best_id: b.id,
                best_fitness: b.fitness,
                best_solved: b.solved,
                archive_size: archive.len(),
                calls_used: calls,
            });
        }

        // The Observations act at the commit, one after another in log order: the newborns'
        // Stage votes, then the peer judgments in the order of the plan (spec 8.4). The
        // `souls` event lists the Workers whose current Soul changed (spec 8.5).
        let applied = self.cfg.soul.as_ref().map(|_| {
            let applied = apply_observations(&mut archive, &vote_observations, &peer_observations);

            events.push(Event::Souls {
                generation: g,
                changed: applied.changed.clone(),
            });

            applied
        });

        // The source Workers of the newborns learn their steps, from each newborn in birth
        // order (spec-imagination 3.3). The `learned` event lists the Workers whose values
        // changed. Values reach no answer, so the select answers stay.
        if self.cfg.imagination.is_some() {
            let changed = match learn_at_commit(&mut archive, &lessons) {
                Ok(changed) => changed,
                Err(e) => {
                    self.calls = calls;

                    return Err(e);
                }
            };

            events.push(Event::Learned {
                generation: g,
                changed,
            });
        }

        // A member votes on the select panel with its current Souls (spec 8.6). So the
        // select answers kept from an earlier select phase are dropped for every member
        // whose current Soul changed, and the next select phase scores it again. Only a
        // member's own Souls shape its answers, so the answers of the others stay exact. The
        // calls that `select_reserve` keeps back cover a whole select phase.
        if let Some(applied) = &applied {
            for c in &applied.changed {
                self.select_answers.remove(&c.member);
            }
        }

        // Commit. The archive changed, so its select phase is still to come.
        self.archive = archive;
        self.gene_ids = gene_ids;
        self.comparisons = comparisons;
        self.calls = calls;
        self.mutator_answers += answers_used;
        self.next_generation = g + 1;
        self.finished = false;

        // The observables of the committed state, with the weights the next generation
        // draws its parents by, and in a Run with Souls this commit's Observations and the
        // search panel's strengths (spec 13.1).
        let weights = self.selector().observed_weights();
        let strengths: Vec<(Vec<f64>, usize)> = match &self.cfg.soul {
            Some(soul) => strength_counts(
                self.search
                    .iter()
                    .map(|t| soul.strengths.of(&self.domain, t)),
            ),
            None => Vec::new(),
        };
        let moved: Vec<(PhaseState, PhaseState)> = applied
            .as_ref()
            .map(|a| {
                a.before
                    .iter()
                    .cloned()
                    .zip(a.changed.iter().map(|c| c.phases.clone()))
                    .collect()
            })
            .unwrap_or_default();
        let soul = self
            .cfg
            .soul
            .as_ref()
            .zip(applied.as_ref())
            .map(|(soul, applied)| SoulInput {
                readout: soul.readout,
                families: self.domain.families().len(),
                strengths: &strengths,
                observations_vote: applied.vote,
                observations_peer: applied.peer,
                moved: &moved,
                variation: (!soul.variation.is_off()).then_some(steered),
            });

        // With imagination or goals on, the mind of the Run (spec-imagination 3.5): the goal of
        // this generation, and how imagination chose its children.
        let mind =
            (self.cfg.imagination.is_some() || self.cfg.goals.is_some()).then(|| MindInput {
                goal: self.goal(g).map_or(ALL_TASKS, |goal| goal.name.as_str()),
                imagination: imagination_counts,
            });

        events.push(Event::State(observe(&ObserveInput {
            generation: g,
            archive: &self.archive,
            first_new,
            comparisons: &self.comparisons,
            selection_weights: &weights,
            generation_usage,
            tasks: self.search.len(),
            soul,
            mind,
        })));

        Ok((Status::Continue, events))
    }

    /// The goal of generation `g` (`.scratch/seelen-modell/spec-imagination.md`, section
    /// 3.4); None with `goals` off, when every task counts.
    fn goal(&self, g: usize) -> Option<&Goal> {
        let goals = self.cfg.goals?;

        self.goals
            .get(goal_index(g, goals.length, self.goals.len()))
    }

    /// The tasks of the search panel, by their places, that `answers` solve among `tasks`,
    /// or among all tasks when `tasks` is None. An error is the domain's integrity alarm.
    fn solved_on(&self, answers: &[Option<D::Answer>], tasks: Option<&[usize]>) -> Result<usize> {
        let mut solved = 0;

        for (t, (answer, task)) in answers.iter().zip(&self.search).enumerate() {
            if tasks.is_some_and(|tasks| tasks.binary_search(&t).is_err()) {
                continue;
            }

            if let Some(a) = answer
                && checked_grade(&self.domain, task, a)?.solved
            {
                solved += 1;
            }
        }

        Ok(solved)
    }

    /// The learning step of the newborn `d` of generation `g` with the search-panel answers
    /// `answers` (`.scratch/seelen-modell/spec-imagination.md`, section 3.3): its real effect
    /// s on the goal of the generation against its first parent, the one error e = s − p, and
    /// the step of `mind::step` for its n changed families. Returns the values of its Workers
    /// after they learned the step, what its Born event logs of it (when n > 0 or imagination
    /// chose it), and what its source Workers learn at the commit. A seed has no parent: it
    /// keeps the values it starts with and learns nothing.
    fn learn_at_birth(
        &self,
        g: usize,
        d: &Draft,
        exp: &Experience,
        answers: &[Option<D::Answer>],
    ) -> Result<LearnedAtBirth> {
        let Some(first) = exp.first_parent else {
            return Ok(LearnedAtBirth {
                values: exp.start.clone(),
                effect: None,
                lesson: None,
            });
        };
        let parent = self.archive.get(first).ok_or_else(|| {
            BrutError::Integrity(format!("learning: parent {first} is not archived"))
        })?;
        let rate = self.cfg.imagination.map_or(0, |i| i.rate);
        let tasks = self.goal(g).map(|goal| goal.tasks.as_slice());
        let s = crate::cascade::mind::effect(
            self.solved_on(answers, tasks)?,
            self.solved_on(&parent.answers, tasks)?,
        );
        let n = exp.families();
        let (values, step) = exp.learn(s, rate);
        let effect = (n > 0 || d.imagined.is_some()).then(|| Effect {
            p: exp.p,
            s,
            gained: exp.changes.iter().map(|c| c.gained.clone()).collect(),
            lost: exp.changes.iter().map(|c| c.lost.clone()).collect(),
            draw: d.imagined.map(|i| i.draw),
        });

        Ok(LearnedAtBirth {
            values,
            effect,
            lesson: Some(exp.lesson(step)),
        })
    }

    /// What the `mind` group reads from the children `drafts` of a generation with
    /// imagination (spec-imagination 3.5): the children imagination chose with some
    /// prediction other than 0, those whose chosen draw is not the first, and the current
    /// learned values of their parents' Workers, for Linear, Phase and Hadamard and for the
    /// other families. Each parent counts once per child, and both parents of a crossover
    /// count.
    fn imagination_counts(&self, drafts: &[Draft]) -> ImaginationCounts {
        let families = self.domain.families();
        let preferred: Vec<bool> = families
            .iter()
            .map(|f| FIXED_PREFERRED.contains(f))
            .collect();
        let at: HashMap<&str, &Member<D::Answer>> =
            self.archive.iter().map(|m| (m.id.as_str(), m)).collect();
        let mut counts = ImaginationCounts::default();

        for d in drafts {
            if let Some(imagined) = d.imagined {
                counts.choices += u64::from(imagined.choice);
                counts.changed += u64::from(imagined.draw != 1);
            }

            for parent in d.parents.iter().filter_map(|p| at.get(p.as_str())) {
                for values in parent.values_now.iter().flatten() {
                    for (f, &v) in values.iter().enumerate() {
                        if preferred.get(f).copied().unwrap_or(false) {
                            counts.linear_phase_hadamard.add(v);
                        } else {
                            counts.other_families.add(v);
                        }
                    }
                }
            }
        }

        counts
    }

    /// `finish_with` without storing the reservation (for in-memory runs and tests).
    pub fn finish(
        &mut self,
        stop: &(dyn Fn() -> bool + Sync),
        stop_reason: &str,
    ) -> Result<Option<(Report, Vec<Event>)>> {
        self.finish_with(stop, stop_reason, &mut |_| Ok(()))
    }

    /// Score the best bred members (finalists) and, for reference, every seed Cascade on
    /// the unseen select panel, and summarise. Refused with a budget error before any
    /// call when the remaining calls cannot cover the worst case (a resume may have
    /// lowered `max_calls`). Returns None when `stop` fires; the calls made so far stay
    /// spent, and so do the answers already obtained: a later `finish` scores only the
    /// puzzles still missing, so a stop never costs a second select pass.
    ///
    /// Before each member it scores, `persist` receives a snapshot that keeps the answers
    /// so far and reserves that member's worst case. The events come back only once per
    /// archive: a run finished before gets its report again with no events, since its log
    /// already holds them. An integrity alarm from the domain ends the phase with that
    /// error.
    pub fn finish_with(
        &mut self,
        stop: &(dyn Fn() -> bool + Sync),
        stop_reason: &str,
        persist: &mut Persist<'_, D::Answer>,
    ) -> Result<Option<(Report, Vec<Event>)>> {
        let order = rank(&self.archive, &self.gene_ids);
        let mut events = Vec::new();
        let mut finalists = Vec::new();
        let picked = self.select_picks(&order);
        let need = picked
            .iter()
            .map(|&i| self.member_need(i))
            .fold(0u64, u64::saturating_add);

        // A select phase that needs no call (every answer is kept) cannot exceed the cap.
        if need > 0 && !self.fits(need) {
            return Err(BrutError::Budget(format!(
                "the select phase may need {need} calls, but only {} of max_calls {} remain; resume with a larger max_calls",
                self.cfg.max_calls.saturating_sub(self.calls),
                self.cfg.max_calls
            )));
        }

        for &i in &picked {
            let id = self.archive[i].id.clone();
            let missing = self.select_missing(&id);

            if !missing.is_empty() {
                // Members run one after another, and the snapshot stored before each one
                // keeps the answers of those before it, so the reservation covers only the
                // member in flight.
                persist(&self.snapshot_with(self.member_need(i)))?;

                let tag = format!("select/{}", self.gene_ids[i]);
                let tasks: Vec<&D::Task> = missing.iter().map(|&j| &self.select[j]).collect();
                // A member votes with its current Souls (spec 8.4, 8.6), and its teaching
                // section ends its system prompts (spec 10.2). The select phase records no
                // Observation, so it changes no Soul (spec 8.2).
                let m = &self.archive[i];
                let souls = m.soul_now.as_deref().map(|now| PanelSouls {
                    first: now,
                    rest: now,
                    upbringing: m.teaching.as_deref(),
                });
                let (runs, usage) = self.run_panel(&m.cascade, souls, &tasks, &tag, false, stop);
                let mut complete = true;
                let got = self.select_answers.entry(id.clone()).or_default();

                for (task, run) in tasks.iter().zip(runs) {
                    match run {
                        Some(r) => {
                            got.insert(self.domain.task_id(task).to_string(), r.answer);
                        }
                        None => complete = false,
                    }
                }

                self.calls += usage.calls;

                if !complete {
                    return Ok(None);
                }
            }

            let got = self.select_answers.get(&id);
            let answers: Vec<Option<D::Answer>> = self
                .select
                .iter()
                .map(|t| {
                    got.and_then(|g| g.get(self.domain.task_id(t)))
                        .cloned()
                        .flatten()
                })
                .collect();
            let (select_solved, select_quality) = tally(&self.domain, &answers, &self.select)?;
            let m = &self.archive[i];
            let f = Finalist {
                id,
                is_seed: m.op == Op::Seed,
                search_solved_rate: m.solved as f64 / self.search.len().max(1) as f64,
                select_solved,
                select_puzzles: self.select.len(),
                select_quality,
            };

            events.push(Event::Finalist(f.clone()));
            finalists.push(f);
        }

        let rate = |f: &Finalist| f.select_solved as f64 / f.select_puzzles.max(1) as f64;
        let best_of = |seed: bool| {
            finalists
                .iter()
                .filter(|f| f.is_seed == seed)
                .map(rate)
                .fold(0.0, f64::max)
        };
        let top = order
            .iter()
            .take(10)
            .map(|&i| ranked(&self.archive[i], self.search.len()))
            .collect();

        let report = Report {
            generations_completed: self.next_generation,
            archive_size: self.archive.len(),
            calls_used: self.calls,
            mutator_answers_used: self.mutator_answers,
            comparisons: self.comparisons.len(),
            stop_reason: stop_reason.to_string(),
            top,
            best_finalist_select_rate: best_of(false),
            best_seed_select_rate: best_of(true),
            finalists,
        };

        if self.finished {
            events.clear();
        }

        self.finished = true;

        Ok(Some((report, events)))
    }

    /// The members the select phase scores, in `order` (best first): the best
    /// `finalists` bred members, then every seed.
    fn select_picks(&self, order: &[usize]) -> Vec<usize> {
        let mut picked: Vec<usize> = order
            .iter()
            .copied()
            .filter(|&i| self.archive[i].op != Op::Seed)
            .take(self.cfg.finalists)
            .collect();

        for (i, m) in self.archive.iter().enumerate() {
            if m.op == Op::Seed && !picked.contains(&i) {
                picked.push(i);
            }
        }

        picked
    }

    /// Worst-case calls member `i` still needs on the select panel.
    fn member_need(&self, i: usize) -> u64 {
        let m = &self.archive[i];
        let missing = self.select_missing(&m.id).len() as u64;

        cascade_call_bound(&m.cascade).saturating_mul(missing)
    }

    /// Worst-case calls the select phase still needs: 0 when every answer is kept, as
    /// for a finished run.
    pub fn select_need(&self) -> u64 {
        self.select_picks(&rank(&self.archive, &self.gene_ids))
            .iter()
            .map(|&i| self.member_need(i))
            .fold(0u64, u64::saturating_add)
    }

    /// Indices of the select-panel tasks member `id` has no answer for yet.
    fn select_missing(&self, id: &str) -> Vec<usize> {
        let got = self.select_answers.get(id);

        (0..self.select.len())
            .filter(|&j| {
                let task = self.domain.task_id(&self.select[j]);

                !got.is_some_and(|g| g.contains_key(task))
            })
            .collect()
    }

    /// The runs of `c` on `panel`, with `souls` as its Workers' Souls in a Run with Souls,
    /// and their usage. The runs are None when `stop` fired before some task; the usage
    /// still counts every call that was made. A newborn's evaluation on the search panel,
    /// so its Stage votes observe when `observe_vote` is on (spec 8.2).
    fn evaluate(
        &self,
        c: &Cascade,
        souls: Option<PanelSouls<'_>>,
        panel: &[D::Task],
        tag: &str,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> (Option<Vec<CascadeRun<D::Answer>>>, Usage) {
        let tasks: Vec<&D::Task> = panel.iter().collect();
        let observe = self.cfg.soul.as_ref().is_some_and(|s| s.observe_vote);
        let (runs, usage) = self.run_panel(c, souls, &tasks, tag, observe, stop);

        (runs.into_iter().collect(), usage)
    }

    /// Run `c` on every task of `panel`, in parallel. An entry is None when `stop` fired
    /// before that task started; a started task always finishes, and the usage counts
    /// every call made. Every call's seed comes from `tag` (which names the Cascade) and
    /// the task id, so the result depends neither on thread count nor on when, or with
    /// which other tasks, the Cascade runs. In a Run with Souls, `souls` holds the Souls
    /// its Workers vote with, frozen for the generation: those of the panel's first task and
    /// those of every later one (spec 7.2, 8.4, 10.1), and its teaching section; each task
    /// gives its strengths. With `observe`, the runs record the Observations of their Stage
    /// votes (spec 8.2); the select phase records none.
    fn run_panel(
        &self,
        c: &Cascade,
        souls: Option<PanelSouls<'_>>,
        panel: &[&D::Task],
        tag: &str,
        observe: bool,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> (Vec<Option<CascadeRun<D::Answer>>>, Usage) {
        let runs: Vec<Option<CascadeRun<D::Answer>>> = panel
            .par_iter()
            .enumerate()
            .map(|(i, task)| {
                if stop() {
                    return None;
                }

                let id = self.domain.task_id(task);
                let seed = derive_seed(self.cfg.seed, &format!("{tag}/{id}"));
                let run = match (&self.cfg.soul, souls) {
                    (Some(soul), Some(panel_souls)) => {
                        let strengths = soul.strengths.of(&self.domain, task);
                        let with = RunSouls {
                            souls: match i {
                                0 => panel_souls.first,
                                _ => panel_souls.rest,
                            },
                            strengths: &strengths,
                            readout: soul.readout,
                            vote: soul.vote,
                            simulator: soul.simulator,
                            observe,
                            upbringing: panel_souls.upbringing,
                        };

                        run_cascade_with(
                            &self.domain,
                            self.pool,
                            &self.cfg.limits,
                            c,
                            task,
                            seed,
                            Some(&with),
                        )
                    }
                    _ => run_cascade_in(&self.domain, self.pool, &self.cfg.limits, c, task, seed),
                };

                Some(run)
            })
            .collect();

        let mut usage = Usage::default();

        for r in runs.iter().flatten() {
            usage.add(&r.usage);
        }

        (runs, usage)
    }

    /// Upper bound on judge calls of one peer round with `n` new members: each meets at
    /// most `peer_opponents` of the other members on at most `peer_puzzles_per_pair`
    /// puzzles. Saturating, so extreme settings cannot wrap below the real count.
    fn peer_call_bound(&self, n: usize) -> u64 {
        let others = (self.archive.len() + n).saturating_sub(1);
        let opponents = self.cfg.peer_opponents.min(others) as u64;
        let puzzles = self.cfg.peer_puzzles_per_pair.min(self.search.len()) as u64;

        (n as u64).saturating_mul(opponents).saturating_mul(puzzles)
    }

    /// Plan and run the cross-Cascade judgments of one generation. Each new member meets
    /// up to `peer_opponents` unrelated members (no parent, child or sibling) on puzzles
    /// where their answers differ; a third member, related to neither, judges blind. New members are preferred as
    /// judges, the least used first, so the judgments spread over them and their judging
    /// skill gets measured; a member whose judgments are never scored keeps the prior
    /// skill of 1/2. The comparisons are None when `stop` fired; the usage still counts
    /// every call that was made.
    ///
    /// In a Run with Souls and `observe_peer` on, a judgment that got a reply is an
    /// Observation by the judge Worker of every author of the first answer and then of the
    /// second's (step 2, spec 8.2). They come back with the comparisons, in the order of the
    /// plan. A peer request never carries phases (spec 7.8, 11.1).
    #[allow(clippy::type_complexity)]
    fn peer_round(
        &self,
        archive: &[Member<D::Answer>],
        new: &[usize],
        g: usize,
        rng: &mut ChaCha8Rng,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> (Option<(Vec<Comparison>, Vec<Observation>)>, Usage) {
        let kin = Kinship::new(archive);
        let mut plan: Vec<(usize, usize, usize, usize, u64)> = Vec::new();
        let mut assigned = vec![0usize; archive.len()];

        for &c in new {
            let mut opponents: Vec<usize> = (0..archive.len())
                .filter(|&o| o != c && !kin.related(c, o))
                .collect();

            opponents.shuffle(rng);

            for &o in opponents.iter().take(self.cfg.peer_opponents) {
                let mut differ: Vec<usize> = (0..self.search.len())
                    .filter(
                        |&i| match (&archive[c].answers.get(i), &archive[o].answers.get(i)) {
                            (Some(Some(a)), Some(Some(b))) => a != b,
                            _ => false,
                        },
                    )
                    .collect();

                differ.shuffle(rng);

                for &i in differ.iter().take(self.cfg.peer_puzzles_per_pair) {
                    let fair = |j: &usize| {
                        *j != c && *j != o && !kin.related(*j, c) && !kin.related(*j, o)
                    };
                    let mut judges: Vec<usize> = new.iter().copied().filter(fair).collect();

                    if judges.is_empty() {
                        judges = (0..archive.len()).filter(fair).collect();
                    }

                    let Some(j) = least_used(&judges, &assigned, rng) else {
                        continue;
                    };

                    assigned[j] += 1;
                    plan.push((c, o, i, j, rng.random::<u64>()));
                }
            }
        }

        let observe = self.cfg.soul.as_ref().is_some_and(|s| s.observe_peer);
        let outcomes: Vec<Option<PeerOutcome>> = plan
            .par_iter()
            .map(|&(c, o, i, j, seed)| {
                if stop() {
                    return None;
                }

                let task = &self.search[i];
                let (Some(Some(a)), Some(Some(b))) =
                    (archive[c].answers.get(i), archive[o].answers.get(i))
                else {
                    return Some((None, Usage::default(), Vec::new()));
                };
                let judge = &archive[j].cascade;
                let mut usage = Usage::default();
                let Some(w) = judge.stages.last().and_then(|s| s.workers.first()) else {
                    return Some((None, usage, Vec::new()));
                };
                // The judge Worker's index over the Stages.
                let judge_worker =
                    judge.worker_count() - judge.stages.last().map_or(0, |s| s.workers.len());
                // The judge's teaching section ends its request, as it ends every system
                // prompt the judge sends (spec 10.2).
                let ctx = Ctx {
                    pool: self.pool,
                    limits: &self.cfg.limits,
                    upbringing: archive[j].teaching.as_deref(),
                };
                // Scored: the domain strictly prefers one of the two answers (for grids,
                // exactly one of them is right). Some(true) when it prefers `a`.
                let preferred = self.domain.prefer(task, a, b);
                let record = |winner: Option<usize>, judge_correct: Option<bool>| Comparison {
                    generation: g,
                    puzzle: self.domain.task_id(task).to_string(),
                    first: archive[c].id.clone(),
                    second: archive[o].id.clone(),
                    judge: archive[j].id.clone(),
                    winner: winner.map(|x| archive[x].id.clone()),
                    judge_correct,
                };
                let verdict = judge_in(ctx, &self.domain, w, task, (a, b), seed, &mut usage);
                // The authors of both answers on this task, first then second (spec 7.7).
                let observed: Vec<Observation> = match verdict {
                    Judgment::NoReply => Vec::new(),
                    _ if !observe => Vec::new(),
                    _ => [c, o]
                        .into_iter()
                        .flat_map(|m| {
                            let authors = archive[m].authors.as_ref().and_then(|a| a.get(i));

                            authors.into_iter().flatten().map(move |&t| Observation {
                                observer: (j, judge_worker),
                                observed: (m, t),
                            })
                        })
                        .collect(),
                };
                let comparison = match verdict {
                    Judgment::Verdict(a_won) => {
                        let winner = if a_won { c } else { o };
                        let judge_correct = preferred.map(|a_better| a_better == a_won);

                        Some(record(Some(winner), judge_correct))
                    }
                    // A reply without a usable verdict on a scored pair is a wrong
                    // judgment, so a Cascade cannot keep the prior skill of 1/2 by
                    // never answering in the judge format.
                    Judgment::NoVerdict if preferred.is_some() => Some(record(None, Some(false))),
                    Judgment::NoVerdict | Judgment::NoReply => None,
                };

                Some((comparison, usage, observed))
            })
            .collect();

        let mut comparisons = Vec::new();
        let mut observations = Vec::new();
        let mut usage = Usage::default();
        let mut complete = true;

        for out in outcomes {
            match out {
                Some((c, u, seen)) => {
                    usage.add(&u);
                    comparisons.extend(c);
                    observations.extend(seen);
                }
                None => complete = false,
            }
        }

        (complete.then_some((comparisons, observations)), usage)
    }

    /// Draft the children of generation `g` (g >= 1): first the LLM round's children if
    /// this is one, then random mutation and crossover, all distinct and new.
    fn draft_children(&self, g: usize, rng: &mut ChaCha8Rng) -> Result<Drafted> {
        let mut drafts: Vec<Draft> = Vec::new();
        let mut events = Vec::new();
        let mut answers_used = 0u64;
        // Novelty goes by genes (spec 9.6): a child with the genes of a member and another
        // Soul is no new child. Without Souls the gene ids are the ids.
        let mut seen: HashSet<String> = self.gene_ids.iter().cloned().collect();
        let sel = self.selector();

        let llm_round = self.cfg.llm_every > 0 && g.is_multiple_of(self.cfg.llm_every);

        if llm_round && let Some(mutator) = self.mutator {
            let n = self.cfg.llm_children.min(self.cfg.children_per_generation);
            let mut planned: Vec<(GenRequest, Vec<usize>, Op)> = Vec::new();

            for i in 0..n {
                let Some(a) = sel.pick(rng) else {
                    break;
                };
                let seed = derive_seed(self.cfg.seed, &format!("llm/{g}/{i}"));
                let partner = if i % 2 == 1 {
                    self.partner(a, &sel, rng)
                } else {
                    None
                };
                let pa = &self.archive[a];

                match partner {
                    Some(b) => {
                        let pb = &self.archive[b];
                        let req = llm_crossover_request_with(
                            self.genes.brief,
                            (&pa.cascade, pa.fitness),
                            (&pb.cascade, pb.fitness),
                            &self.cfg.limits,
                            seed,
                        );

                        planned.push((req, vec![a, b], Op::LlmCrossover));
                    }
                    None => {
                        let req = llm_mutate_request_with(
                            self.genes.brief,
                            &pa.cascade,
                            pa.fitness,
                            &self.cfg.limits,
                            seed,
                        );

                        planned.push((req, vec![a], Op::LlmMutate));
                    }
                }
            }

            let answered: Vec<Option<GenResponse>> = planned
                .iter()
                .map(|(req, _, _)| mutator.answer(req))
                .collect();

            if self.cfg.pause_for_llm && answered.iter().any(Option::is_none) {
                let missing = planned
                    .iter()
                    .zip(&answered)
                    .filter(|(_, a)| a.is_none())
                    .map(|((req, _, _), _)| req.clone())
                    .collect();

                // The random children this generation gets if every answer is accepted:
                // the cheapest generation the answers can give (see step_with).
                let mut random = Vec::new();

                self.fill_random(g, &mut random, planned.len(), &mut seen, &sel, rng)?;

                return Ok(Drafted::Pending(missing, planned.len(), random));
            }

            for ((req, parents, op), resp) in planned.into_iter().zip(answered) {
                let key = request_key(&req);

                if resp.is_some() {
                    answers_used += 1;
                }

                let parsed = match &resp {
                    None => Err("no answer".to_string()),
                    Some(r) if !r.is_usable() => {
                        Err(format!("unusable answer ({:?})", r.stop_reason))
                    }
                    // The Mutator never writes a Soul (spec 4.5). This rule comes first, so
                    // it names the reason also when the answer breaks other rules.
                    Some(r) if read_llm_cascade(&r.text).is_ok_and(|c| c.carries_soul()) => {
                        Err(SOUL_NOT_ADMITTED.to_string())
                    }
                    Some(r) => {
                        parse_llm_cascade(&r.text, &self.cfg.limits).map_err(|e| e.to_string())
                    }
                };
                // The gene pool's admission rule holds for every new Cascade.
                let parsed = parsed.and_then(|c| match (self.genes.admit)(&c) {
                    Ok(()) => Ok(c),
                    Err(e) => Err(format!("not admitted: {e}")),
                });
                let (accepted, reason) = match parsed {
                    Ok(c) if seen.contains(&c.gene_id()) => {
                        (false, "duplicate of an archived Cascade".to_string())
                    }
                    Ok(c) => {
                        // The Mutator writes any shape: each Worker's sources follow the
                        // partner rule on every parent (spec 9.1).
                        let from: Vec<&Cascade> =
                            parents.iter().map(|&p| &self.archive[p].cascade).collect();

                        seen.insert(c.gene_id());
                        drafts.push(Draft {
                            sources: partner_sources(&c, &from),
                            cascade: c,
                            parents: parents
                                .iter()
                                .map(|&p| self.archive[p].id.clone())
                                .collect(),
                            op,
                            soul_sources: None,
                            imprinted: None,
                            teaching: None,
                            steered: Steered::default(),
                            imagined: None,
                            experience: None,
                        });

                        (true, "ok".to_string())
                    }
                    Err(e) => (false, e),
                };

                events.push(Event::LlmExchange {
                    generation: g,
                    key,
                    request: req,
                    response: resp,
                    accepted,
                    reason,
                });
            }
        }

        self.fill_random(g, &mut drafts, 0, &mut seen, &sel, rng)?;

        Ok(Drafted::Ready(drafts, events, answers_used))
    }

    /// Add random children (mutation and crossover, all distinct, new and admitted by the
    /// gene pool) to `drafts` of generation `g` until they and `others` other children reach
    /// `children_per_generation`, or the draws run out.
    ///
    /// With `imagination` on (`.scratch/seelen-modell/spec-imagination.md`, section 3.2), a
    /// child that SectionReplace, SectionAdd, SectionDrop or section crossover made, and that
    /// passed every check, is draw 1 of its operator. Draws 2 and on apply the same operator
    /// to the same parents, each with its own stream, and the draw with the highest
    /// prediction becomes the child (`imagine`). They never touch `rng`, so the generation's
    /// stream advances exactly as without imagination.
    fn fill_random(
        &self,
        g: usize,
        drafts: &mut Vec<Draft>,
        others: usize,
        seen: &mut HashSet<String>,
        sel: &Selector,
        rng: &mut ChaCha8Rng,
    ) -> Result<()> {
        let mut attempts = 0;

        while others + drafts.len() < self.cfg.children_per_generation
            && attempts < self.cfg.children_per_generation * DRAW_ATTEMPTS
        {
            attempts += 1;

            let Some(a) = sel.pick(rng) else {
                break;
            };
            let parent = &self.archive[a];
            let crossover = rng.random_bool(self.cfg.crossover_rate);
            // The child with its parents' places in the archive.
            let mut child: Option<(Cascade, Vec<usize>, Op, Sources, Steered)> = None;
            // Arm W: the preference of each of the parent's Workers, which weights the
            // choices of the section operators.
            let preferences = self.preferences(parent);
            let steer = preferences.as_deref().zip(self.genes.families_of).map(
                |(preferences, families_of)| Steer {
                    families_of,
                    preferences,
                },
            );

            // The operators see genes only: they remove the parents' Souls and Coupling
            // strengths first, so their draws are those of a Run without Souls (spec 9.1).
            // In arm W the section operators also get the parent's preferences, and their
            // draws follow them.
            if crossover && let Some(b) = self.partner(a, sel, rng) {
                let other = &self.archive[b];
                let by_stage = rng.random_bool(0.5);
                let limits = &self.cfg.limits;
                let made = if by_stage {
                    stage_crossover_sourced(&parent.cascade, &other.cascade, limits, rng)
                        .map(|(c, s)| (c, Op::StageCrossover, s, Steered::default()))
                } else {
                    match &steer {
                        Some(steer) => section_crossover_steered(
                            &parent.cascade,
                            &other.cascade,
                            limits,
                            steer,
                            rng,
                        ),
                        None => {
                            section_crossover_sourced(&parent.cascade, &other.cascade, limits, rng)
                                .map(|(c, s)| (c, s, Steered::default()))
                        }
                    }
                    .map(|(c, s, steered)| (c, Op::SectionCrossover, s, steered))
                };

                child = made.map(|(c, op, s, steered)| (c, vec![a, b], op, s, steered));
            }

            if child.is_none() {
                let (limits, phrases) = (&self.cfg.limits, self.genes.phrases);
                let made = match &steer {
                    Some(steer) => mutate_steered(&parent.cascade, limits, phrases, steer, rng),
                    None => mutate_sourced(&parent.cascade, limits, phrases, rng)
                        .map(|(c, op, s)| (c, op, s, Steered::default())),
                };

                child = made.map(|(c, op, s, steered)| (c, vec![a], op, s, steered));
            }

            if let Some((c, parents, op, sources, steered)) = child
                && self.admissible(&c, seen)
            {
                let first = Drawn {
                    cascade: c,
                    sources,
                    steered,
                };
                let (drawn, imagined) =
                    if self.cfg.imagination.is_some() && IMAGINING_OPS.contains(&op) {
                        let i = others + drafts.len();

                        self.imagine((g, i), &parents, op, first, steer.as_ref(), seen)?
                    } else {
                        (first, None)
                    };

                seen.insert(drawn.cascade.id());
                drafts.push(Draft {
                    cascade: drawn.cascade,
                    parents: parents
                        .iter()
                        .map(|&p| self.archive[p].id.clone())
                        .collect(),
                    op,
                    sources: drawn.sources,
                    soul_sources: None,
                    imprinted: None,
                    teaching: None,
                    steered: drawn.steered,
                    imagined,
                    experience: None,
                });
            }
        }

        Ok(())
    }

    /// True when `c` may become a child: it passes `check(limits)` and the gene pool's
    /// admission rule, and its id is not in `seen`, the gene ids of the archive and of the
    /// children drafted so far.
    fn admissible(&self, c: &Cascade, seen: &HashSet<String>) -> bool {
        c.check(&self.cfg.limits).is_ok()
            && (self.genes.admit)(c).is_ok()
            && !seen.contains(&c.id())
    }

    /// Imagination (`.scratch/seelen-modell/spec-imagination.md`, section 3.2) for the child
    /// `first` that today's attempt made with the operator `op` on `parents` (their places
    /// in the archive), as child `i` in birth order of generation `g`. Draws 2 to `draws`
    /// apply `op` once more to the same parents (`redraw`), each with its own stream from the
    /// tag `imagine/{g}/{i}/{j}`, where j is the draw's number; a draw that fails a check of
    /// `admissible` is no candidate. The candidate with the highest prediction wins, and a
    /// tie goes to the earliest draw. So with every value 0 the first draw wins, and the Run
    /// is today's. No model is called.
    fn imagine(
        &self,
        (g, i): (usize, usize),
        parents: &[usize],
        op: Op,
        first: Drawn,
        steer: Option<&Steer<'_>>,
        seen: &HashSet<String>,
    ) -> Result<(Drawn, Option<Imagined>)> {
        let draws = self.cfg.imagination.map_or(1, |i| i.draws);
        let mut candidates = vec![(1, first)];

        for j in 2..=draws {
            let tag = format!("imagine/{g}/{i}/{j}");
            let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(self.cfg.seed, &tag));

            if let Some(drawn) = self.redraw(parents, op, steer, &mut rng)
                && self.admissible(&drawn.cascade, seen)
            {
                candidates.push((j, drawn));
            }
        }

        let predictions = candidates
            .iter()
            .map(|(_, d)| Ok(self.experience(parents, &d.cascade, &d.sources)?.p))
            .collect::<Result<Vec<i64>>>()?;
        let k = choose(&predictions).unwrap_or(0);
        let choice = predictions.iter().any(|&p| p != 0);
        let (draw, chosen) = candidates.swap_remove(k);

        Ok((chosen, Some(Imagined { draw, choice })))
    }

    /// One more draw of the imagining operator `op` on `parents` (spec-imagination 3.2): one
    /// application of the same mutation operator to the same parent (`mutate_once`), or one
    /// full section crossover of the same two parents, with the preferences of arm W when
    /// `steer` is given. None when the draw gives no child.
    fn redraw(
        &self,
        parents: &[usize],
        op: Op,
        steer: Option<&Steer<'_>>,
        rng: &mut ChaCha8Rng,
    ) -> Option<Drawn> {
        let limits = &self.cfg.limits;
        let a = &self.archive.get(*parents.first()?)?.cascade;
        let (cascade, sources, steered) = match op {
            Op::SectionCrossover => {
                let b = &self.archive.get(*parents.get(1)?)?.cascade;

                match steer {
                    Some(steer) => section_crossover_steered(a, b, limits, steer, rng)?,
                    None => {
                        let (c, s) = section_crossover_sourced(a, b, limits, rng)?;

                        (c, s, Steered::default())
                    }
                }
            }
            _ => mutate_once(a, op, limits, self.genes.phrases, steer, rng)?,
        };

        Some(Drawn {
            cascade,
            sources,
            steered,
        })
    }

    /// What the child `child` with the Worker sources `sources` brings to its learning
    /// (`.scratch/seelen-modell/spec-imagination.md`, sections 3.1 to 3.3), when its parents
    /// are the members at `parents` in the archive: for each Worker, the current values of its
    /// source Worker (the first of its sources), or 0 without a source; what it gained and
    /// lost against that source; and the prediction p over all Workers. It reads the archive
    /// as committed after the previous generation.
    fn experience(
        &self,
        parents: &[usize],
        child: &Cascade,
        sources: &Sources,
    ) -> Result<Experience> {
        let broken = |what: String| BrutError::Integrity(format!("experience: {what}"));
        let families_of = self
            .genes
            .families_of
            .ok_or_else(|| broken("the gene pool names no families".to_string()))?;
        let n = self.domain.families().len();
        let workers = child.worker_count();

        if sources.len() != workers {
            return Err(broken(format!(
                "a child has {} source lists for {workers} Workers",
                sources.len()
            )));
        }

        let mut exp = Experience {
            start: Vec::with_capacity(workers),
            sources: Vec::with_capacity(workers),
            changes: Vec::with_capacity(workers),
            p: 0,
            first_parent: parents.first().copied(),
        };

        for (w, from) in child.stages.iter().flat_map(|s| &s.workers).zip(sources) {
            let Some(src) = from.first() else {
                exp.start.push(vec![0; n]);
                exp.sources.push(None);
                exp.changes.push(Change::default());
                continue;
            };
            let m = *parents
                .get(src.parent)
                .ok_or_else(|| broken(format!("a source names parent {}", src.parent)))?;
            let member = self
                .archive
                .get(m)
                .ok_or_else(|| broken(format!("parent {m} is not archived")))?;
            let source = member
                .cascade
                .stages
                .iter()
                .flat_map(|s| &s.workers)
                .nth(src.worker)
                .ok_or_else(|| {
                    broken(format!("member {} has no Worker {}", member.id, src.worker))
                })?;
            let values = member
                .values_now
                .as_ref()
                .and_then(|v| v.get(src.worker))
                .ok_or_else(|| {
                    broken(format!(
                        "Worker {} of member {} has no learned values",
                        src.worker, member.id
                    ))
                })?;

            exp.start.push(values.clone());
            exp.sources.push(Some((m, src.worker)));
            exp.changes.push(Change::between(
                &named(w, families_of, n),
                &named(source, families_of, n),
            ));
        }

        exp.p = prediction(
            exp.changes
                .iter()
                .zip(&exp.start)
                .map(|(c, v)| (c, v.as_slice())),
        );

        Ok(exp)
    }

    /// The experience of every draft (spec-imagination 3.1 to 3.3), with its parents found
    /// in the archive by id.
    fn experiences(&self, drafts: &mut [Draft]) -> Result<()> {
        let at: HashMap<&str, usize> = self
            .archive
            .iter()
            .enumerate()
            .map(|(i, m)| (m.id.as_str(), i))
            .collect();

        for d in drafts.iter_mut() {
            let parents = d
                .parents
                .iter()
                .map(|p| at.get(p.as_str()).copied())
                .collect::<Option<Vec<usize>>>()
                .ok_or_else(|| {
                    BrutError::Integrity(
                        "experience: a parent of a child is not archived".to_string(),
                    )
                })?;

            d.experience = Some(self.experience(&parents, &d.cascade, &d.sources)?);
        }

        Ok(())
    }

    /// The preference of every Worker of `m`, in Stage order, that weights the choices of the
    /// section operators (arm W, `.scratch/seelen-modell/spec.md`, section 4.1): with
    /// `variation: prefer` that of the Worker's birth Soul, never of its current Soul; with
    /// `fixed` the preference of arm WF. None without Souls or with `variation` off.
    fn preferences(&self, m: &Member<D::Answer>) -> Option<Vec<Vec<f64>>> {
        let soul = self.cfg.soul.as_ref()?;
        let workers = m.cascade.stages.iter().flat_map(|s| &s.workers);

        match soul.variation {
            Variation::Off => None,
            Variation::Prefer => Some(
                workers
                    .map(|w| {
                        preference(
                            w.traits.phase_state.as_deref().unwrap_or_default(),
                            soul.readout,
                        )
                    })
                    .collect(),
            ),
            Variation::Fixed => {
                let fixed = fixed_preference(&self.domain.families());

                Some(workers.map(|_| fixed.clone()).collect())
            }
        }
    }

    /// Parent selection weights over the archive: DGM weights from fitness and children
    /// counts, or the same weight for every member under uniform selection. The lexicase
    /// and tournament strategies do not use weights (see `selector`).
    fn selection_weights(&self) -> Vec<f64> {
        if self.cfg.selection != ParentSelection::Dgm {
            return vec![1.0; self.archive.len()];
        }

        let scores: Vec<f64> = self.archive.iter().map(|m| m.fitness).collect();
        let children: Vec<usize> = self.archive.iter().map(|m| m.children).collect();

        dgm_weights(&scores, &children, self.cfg.dgm_lambda, self.cfg.dgm_alpha0)
    }

    /// The parent selector of the next generation, from the archive as it stands.
    fn selector(&self) -> Selector {
        let scores = if self.cfg.selection == ParentSelection::Lexicase {
            self.case_scores()
        } else {
            Vec::new()
        };
        let mut rank_of = vec![0; self.archive.len()];

        for (pos, i) in rank(&self.archive, &self.gene_ids).into_iter().enumerate() {
            rank_of[i] = pos;
        }

        Selector {
            kind: self.cfg.selection,
            params: self.cfg.selection_params,
            weights: self.selection_weights(),
            scores,
            rank_of,
            members: self.archive.len(),
            observe_seed: derive_seed(self.cfg.seed, &format!("observe/{}", self.next_generation)),
        }
    }

    /// The graded value of each member on each search task, 0 where it gave no answer.
    /// A grade error cannot occur here: it would have stopped the member's birth.
    ///
    /// With `goals` on, only the tasks of the goal of the next generation are cases, in panel
    /// order (`.scratch/seelen-modell/spec-imagination.md`, section 3.4).
    fn case_scores(&self) -> Vec<Vec<f64>> {
        let scores = self
            .archive
            .iter()
            .map(|m| {
                self.search
                    .iter()
                    .zip(&m.answers)
                    .map(|(task, answer)| {
                        answer
                            .as_ref()
                            .and_then(|a| self.domain.grade(task, a).ok())
                            .map_or(0.0, |g| g.value)
                    })
                    .collect()
            })
            .collect();

        match self.goal(self.next_generation) {
            Some(goal) => goal_cases(scores, &goal.tasks),
            None => scores,
        }
    }

    /// A crossover partner for member `a`: draw a few by selection weight and take the
    /// one the peers rate highest (Q2: peer judgment decides who mates with whom). Under
    /// the other strategies, the first draw other than `a`, so ratings play no part.
    fn partner(&self, a: usize, sel: &Selector, rng: &mut ChaCha8Rng) -> Option<usize> {
        let candidates = match self.cfg.selection {
            ParentSelection::Dgm => PARTNER_CANDIDATES,
            _ => 1,
        };
        let mut best: Option<usize> = None;

        for _ in 0..candidates {
            for _ in 0..DRAW_ATTEMPTS {
                let b = sel.pick(rng)?;

                if b != a {
                    let better =
                        best.is_none_or(|x| self.archive[b].rating > self.archive[x].rating);

                    if better {
                        best = Some(b);
                    }

                    break;
                }
            }
        }

        best
    }
}

/// One peer judgment's comparison, if it gives one, its usage, and its Observations.
type PeerOutcome = (Option<Comparison>, Usage, Vec<Observation>);

/// One Observation (step 2, spec 8.2): the Worker `observer` observes the Worker
/// `observed`, each named by its member's place in the archive and its index over the
/// member's Stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Observation {
    observer: (usize, usize),
    observed: (usize, usize),
}

/// What the Observations of one commit did (spec 8.4, 8.5, 13.1).
struct Applied {
    /// Every Worker whose current Soul changed, in archive order, with its new Soul.
    changed: Vec<SoulChange>,
    /// The Soul each of them had when the commit began, in the same order.
    before: Vec<PhaseState>,
    /// The Observations applied, of Stage votes and of the peer round.
    vote: u64,
    peer: u64,
}

/// Apply the Observations of the Stage votes `vote` and then those of the peer round `peer`,
/// one after another in their order, to the current Souls of `archive` (spec 8.3, 8.4) [Q69
/// a]: each moves the observed Worker toward the observer by the observer's Coupling
/// strength, and the observer back by half of that, both from the Souls before it. Returns
/// every Worker whose current Soul changed, in archive order, with its new Soul (spec 8.5)
/// and its Soul before, and how many Observations of each kind were applied. An Observation
/// that names a Worker without a current Soul changes nothing and is not counted.
fn apply_observations<A>(
    archive: &mut [Member<A>],
    vote: &[Observation],
    peer: &[Observation],
) -> Applied {
    let current = |archive: &[Member<A>], (m, w): (usize, usize)| -> Option<PhaseState> {
        archive.get(m)?.soul_now.as_ref()?.get(w).cloned()
    };
    let mut before: BTreeMap<(usize, usize), PhaseState> = BTreeMap::new();
    let mut applied = [0u64; 2];
    let observations = vote
        .iter()
        .map(|o| (0, o))
        .chain(peer.iter().map(|o| (1, o)));

    for (kind, o) in observations {
        // A Worker never observes itself (spec 8.2); this guards the rule.
        if o.observer == o.observed {
            continue;
        }

        let kappa = archive
            .get(o.observer.0)
            .and_then(|m| {
                m.cascade
                    .stages
                    .iter()
                    .flat_map(|s| &s.workers)
                    .nth(o.observer.1)
            })
            .map(|w| w.traits.coupling);
        let (Some(kappa), Some(p), Some(q)) = (
            kappa,
            current(archive, o.observer),
            current(archive, o.observed),
        ) else {
            continue;
        };
        let (moved_p, moved_q) = observation(&p, &q, kappa);

        applied[kind] += 1;
        before.entry(o.observer).or_insert(p);
        before.entry(o.observed).or_insert(q);

        for ((m, w), phases) in [(o.observer, moved_p), (o.observed, moved_q)] {
            if let Some(now) = archive[m].soul_now.as_mut().and_then(|n| n.get_mut(w)) {
                *now = PhaseState(phases);
            }
        }
    }

    let (changed, before) = before
        .into_iter()
        .filter_map(|((m, w), old)| {
            let now = current(archive, (m, w))?;
            let change = SoulChange {
                member: archive[m].id.clone(),
                worker: w,
                phases: now,
            };

            (change.phases != old).then_some((change, old))
        })
        .unzip();

    Applied {
        changed,
        before,
        vote: applied[0],
        peer: applied[1],
    }
}

/// The cases of lexicase under a goal (`.scratch/seelen-modell/spec-imagination.md`, section
/// 3.4): of each member's row of `scores`, only the columns of the tasks `tasks`, in their
/// order.
fn goal_cases(scores: Vec<Vec<f64>>, tasks: &[usize]) -> Vec<Vec<f64>> {
    scores
        .into_iter()
        .map(|row| tasks.iter().filter_map(|&t| row.get(t).copied()).collect())
        .collect()
}

/// The source Workers in `archive` learn the steps of `lessons`, one newborn after another
/// in birth order, each step clamped on its own (`.scratch/seelen-modell/spec-imagination.md`,
/// section 3.3). Returns every Worker whose current values changed, in archive order, each
/// once with its new values. A source Worker without values is an integrity error: a Run
/// with imagination gives every Worker values at its birth.
fn learn_at_commit<A>(archive: &mut [Member<A>], lessons: &[Lesson]) -> Result<Vec<ValueChange>> {
    let mut before: BTreeMap<(usize, usize), Vec<i16>> = BTreeMap::new();

    for lesson in lessons {
        for ((m, w), change) in &lesson.changes {
            let (m, w) = (*m, *w);
            let now = archive
                .get_mut(m)
                .and_then(|member| member.values_now.as_mut())
                .and_then(|values| values.get_mut(w))
                .ok_or_else(|| {
                    BrutError::Integrity(format!(
                        "learning: Worker {w} of archive member {m} has no learned values"
                    ))
                })?;

            before.entry((m, w)).or_insert_with(|| now.clone());
            change.learn(now, lesson.step);
        }
    }

    Ok(before
        .into_iter()
        .filter_map(|((m, w), old)| {
            let member = archive.get(m)?;
            let now = member.values_now.as_ref()?.get(w)?;

            (*now != old).then(|| ValueChange {
                member: member.id.clone(),
                worker: w,
                values: now.clone(),
            })
        })
        .collect())
}

/// Children drafted for one generation, or the requests still waiting for answers.
enum Drafted {
    Ready(Vec<Draft>, Vec<Event>, u64),
    /// The requests still waiting, the number of LLM children planned, and the random
    /// children drafted next to them when every answer is accepted.
    Pending(Vec<GenRequest>, usize, Vec<Draft>),
}

/// Who is related to whom: parents, children and siblings (sharing a parent).
struct Kinship {
    kin: Vec<HashSet<usize>>,
}

impl Kinship {
    fn new<A>(archive: &[Member<A>]) -> Kinship {
        let index: HashMap<&str, usize> = archive
            .iter()
            .enumerate()
            .map(|(i, m)| (m.id.as_str(), i))
            .collect();
        let mut kin: Vec<HashSet<usize>> = vec![HashSet::new(); archive.len()];
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); archive.len()];

        for (i, m) in archive.iter().enumerate() {
            for p in &m.parents {
                if let Some(&pi) = index.get(p.as_str()) {
                    kin[i].insert(pi);
                    kin[pi].insert(i);
                    children[pi].push(i);
                }
            }
        }

        for siblings in &children {
            for &x in siblings {
                for &y in siblings {
                    if x != y {
                        kin[x].insert(y);
                    }
                }
            }
        }

        Kinship { kin }
    }

    fn related(&self, a: usize, b: usize) -> bool {
        a == b || self.kin.get(a).is_some_and(|k| k.contains(&b))
    }
}

/// The grade of `answer` on `task`. An error is the domain's integrity alarm, or a value
/// outside [0, 1]: a broken grader must not reach fitness, the log or a measurement.
pub(crate) fn checked_grade<D: Domain>(
    domain: &D,
    task: &D::Task,
    answer: &D::Answer,
) -> Result<Grade> {
    let grade = domain.grade(task, answer)?;

    if !(0.0..=1.0).contains(&grade.value) {
        return Err(BrutError::Integrity(format!(
            "task {} was graded {}, outside [0, 1]",
            domain.task_id(task),
            grade.value
        )));
    }

    Ok(grade)
}

/// The number of tasks of `panel` that `answers` solve and, for a graded domain, their
/// summed value. An error is the domain's integrity alarm, or a value outside [0, 1]: a
/// broken grader must not reach fitness or the log.
fn tally<D: Domain>(
    domain: &D,
    answers: &[Option<D::Answer>],
    panel: &[D::Task],
) -> Result<(usize, Option<f64>)> {
    let mut solved = 0;
    let mut value = 0.0;

    for (answer, task) in answers.iter().zip(panel) {
        let Some(a) = answer else {
            continue;
        };
        let grade = checked_grade(domain, task, a)?;

        if grade.solved {
            solved += 1;
        }

        value += grade.value;
    }

    Ok((solved, domain.graded().then_some(value)))
}

/// Recompute judging skill, peer ratings and fitness of every member from all
/// comparisons so far. A judge's vote weighs ln(u / (1 - u)) for skill u > 0.5, else 0:
/// the optimal weight of an independent voter of known accuracy (Nitzan and Paroush
/// 1982, International Economic Review 23(2), 289-297). Fitness counts the graded value
/// of a graded domain where the solved count stands otherwise.
fn refresh<A>(
    archive: &mut [Member<A>],
    comparisons: &[Comparison],
    cfg: &BreedConfig,
    puzzles: usize,
) {
    let index: HashMap<String, usize> = archive
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id.clone(), i))
        .collect();

    for m in archive.iter_mut() {
        m.judge_correct = 0;
        m.judge_wrong = 0;
    }

    for c in comparisons {
        if let (Some(&j), Some(ok)) = (index.get(&c.judge), c.judge_correct) {
            if ok {
                archive[j].judge_correct += 1;
            } else {
                archive[j].judge_wrong += 1;
            }
        }
    }

    let weight: Vec<f64> = archive
        .iter()
        .map(|m| {
            let u = m.judge_skill();

            (u / (1.0 - u)).ln().max(0.0)
        })
        .collect();

    let games: Vec<(usize, usize, f64)> = comparisons
        .iter()
        .filter_map(|c| {
            let winner = c.winner.as_ref()?;
            let w = *index.get(winner)?;
            let loser = if winner == &c.first {
                &c.second
            } else {
                &c.first
            };
            let l = *index.get(loser)?;
            let j = *index.get(&c.judge)?;

            Some((w, l, weight[j]))
        })
        .collect();

    let ratings = bradley_terry(archive.len(), &games);
    let n = puzzles.max(1) as f64;

    for (m, r) in archive.iter_mut().zip(ratings) {
        let solved_rate = m.quality.unwrap_or(m.solved as f64) / n;
        let tokens = m.usage.completion_tokens as f64 / n;
        let cost = cfg.token_cost_per_1k * tokens / 1000.0 + m.usage.model_cost / n;

        m.rating = r;
        m.fitness =
            (1.0 - cfg.judge_weight) * solved_rate + cfg.judge_weight * m.judge_skill() - cost;
    }
}

/// Archive indices best first: fitness, then peer rating (tie-break, Q13), then `keys`, the
/// members' gene ids, so that no Soul breaks a tie (spec 15). Without Souls the gene ids are
/// the ids.
fn rank<A>(archive: &[Member<A>], keys: &[String]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..archive.len()).collect();

    debug_assert_eq!(archive.len(), keys.len());

    order.sort_by(|&a, &b| {
        let (x, y) = (&archive[a], &archive[b]);

        y.fitness
            .total_cmp(&x.fitness)
            .then(y.rating.total_cmp(&x.rating))
            .then(keys[a].cmp(&keys[b]))
    });

    order
}

/// The birth Soul of every Worker of `c`, in Stage order.
fn birth_souls(c: &Cascade) -> Vec<PhaseState> {
    c.stages
        .iter()
        .flat_map(|s| &s.workers)
        .map(|w| {
            w.traits
                .phase_state
                .clone()
                .unwrap_or(PhaseState(Vec::new()))
        })
        .collect()
}

/// The current Soul of Worker `worker` of `m`, and its Coupling strength: what imprinting
/// moves a child Worker toward, and how far (spec 10.1).
fn current_soul<A>(m: &Member<A>, worker: usize) -> Option<(&[u16], u16)> {
    let now = m.soul_now.as_ref()?.get(worker)?;
    let w = m
        .cascade
        .stages
        .iter()
        .flat_map(|s| &s.workers)
        .nth(worker)?;

    Some((now.0.as_slice(), w.traits.coupling))
}

/// What source Worker `worker` of `parent` passes on: its birth Soul from the genome, never
/// its current Soul (Q65), and its Coupling strength; where it is; and its Soul's origin.
fn source_soul<A>(
    parent: &Member<A>,
    worker: usize,
) -> Option<(Inheritance<'_>, WorkerRef, &WorkerRef)> {
    let w = parent
        .cascade
        .stages
        .iter()
        .flat_map(|s| &s.workers)
        .nth(worker)?;
    let origin = &parent.soul_sources.as_ref()?.get(worker)?.origin;
    let inherited = Inheritance {
        phases: w.traits.phase_state.as_deref()?,
        coupling: w.traits.coupling,
    };
    let at = WorkerRef {
        member: parent.id.clone(),
        worker,
    };

    Some((inherited, at, origin))
}

/// Why member `m` breaks the rule of a Run with Souls of `n` phases and the settings `soul`
/// (spec 4.4), if it does: every Worker has a Soul of exactly `n` phases and a Coupling
/// strength of at most 1000, and the member records where each Soul came from, each
/// imprinted and current Soul, the authors on each of the `tasks` search tasks and its
/// tally of votes. A teaching section exists only when the Run teaches, and it keeps the
/// rules of a genome's section text (spec 10.2).
fn check_souls<A>(
    m: &Member<A>,
    n: usize,
    tasks: usize,
    soul: &SoulConfig,
) -> std::result::Result<(), String> {
    let workers = m.cascade.worker_count();

    for (k, w) in m.cascade.stages.iter().flat_map(|s| &s.workers).enumerate() {
        match &w.traits.phase_state {
            None => {
                return Err(format!(
                    "Worker {k} has no Soul; a Soul of this Run has {n} phases"
                ));
            }
            Some(p) if p.len() != n => {
                return Err(format!(
                    "Worker {k} has a Soul of {} phases; a Soul of this Run has {n}",
                    p.len()
                ));
            }
            Some(_) => {}
        }

        if w.traits.coupling > COUPLING_MAX {
            return Err(format!(
                "Worker {k} has a Coupling strength above {COUPLING_MAX}"
            ));
        }
    }

    if m.soul_sources.as_ref().map(Vec::len) != Some(workers) {
        return Err(format!(
            "it does not record where each of its {workers} Souls came from"
        ));
    }

    for (what, souls) in [("an imprinted", &m.imprinted), ("a current", &m.soul_now)] {
        match souls {
            Some(s) if s.len() == workers && s.iter().all(|p| p.len() == n) => {}
            _ => {
                return Err(format!(
                    "it does not hold {what} Soul of {n} phases for each of its {workers} Workers"
                ));
            }
        }
    }

    match (&m.teaching, soul.teach) {
        (None, _) => {}
        (Some(_), false) => {
            return Err("it holds a teaching section, but this Run does not teach".to_string());
        }
        (Some(text), true) => check_section_text(text)
            .map_err(|e| format!("its teaching section breaks the rules of a section: {e}"))?,
    }

    if m.authors.as_ref().map(Vec::len) != Some(tasks) || m.votes.is_none() {
        return Err(format!(
            "it does not record the authors on each of the {tasks} search tasks and its votes"
        ));
    }

    Ok(())
}

/// Why member `m` breaks the rule of a Run with imagination over `n` families
/// (`.scratch/seelen-modell/spec-imagination.md`, sections 3.1 and 3.5), if it does: its
/// values at birth and its current values hold, for each of its Workers, one value per family
/// in [−1000, 1000], and what its learning read names, for each Worker, families of the
/// Domain only, an effect s of −1, 0 or 1, and a draw among those the setting makes.
fn check_values<A>(
    m: &Member<A>,
    n: usize,
    imagination: &ImaginationConfig,
) -> std::result::Result<(), String> {
    let workers = m.cascade.worker_count();
    let in_range =
        |v: &Vec<i16>| v.len() == n && v.iter().all(|x| (-VALUE_MAX..=VALUE_MAX).contains(x));

    for (what, values) in [
        ("values at birth", &m.values),
        ("current values", &m.values_now),
    ] {
        match values {
            Some(v) if v.len() == workers && v.iter().all(in_range) => {}
            _ => {
                return Err(format!(
                    "it does not hold {what} of {n} families in [-{VALUE_MAX}, {VALUE_MAX}] for each of its {workers} Workers"
                ));
            }
        }
    }

    if let Some(e) = &m.effect {
        let named = |lists: &Vec<Vec<usize>>| {
            lists.len() == workers && lists.iter().flatten().all(|&f| f < n)
        };

        if !named(&e.gained) || !named(&e.lost) {
            return Err(format!(
                "its effect does not name the families each of its {workers} Workers gained and lost"
            ));
        }

        if !(-1..=1).contains(&e.s)
            || e.draw
                .is_some_and(|d| !(1..=imagination.draws).contains(&d))
        {
            return Err("its effect holds an s or a draw this Run cannot give".to_string());
        }
    }

    Ok(())
}

fn ranked<A>(m: &Member<A>, puzzles: usize) -> Ranked {
    Ranked {
        id: m.id.clone(),
        generation: m.generation,
        op: m.op,
        fitness: m.fitness,
        rating: m.rating,
        solved: m.solved,
        puzzles,
        judge_skill: m.judge_skill(),
        stages: m.cascade.stages.len(),
        workers: m.cascade.worker_count(),
        completion_tokens_per_puzzle: m.usage.completion_tokens as f64 / puzzles.max(1) as f64,
    }
}

/// Run a whole breeding run on grid puzzles to the end in memory: steps until done, then
/// `finish`. Used by tests and by callers that need no checkpoints. A pending LLM round
/// ends the run with reason "waiting for LLM answers".
pub fn breed_to_end(
    cfg: BreedConfig,
    pool: &ModelPool,
    mutator: Option<&dyn Mutator>,
) -> Result<(Report, Vec<Event>)> {
    let (report, events, _) = breed_to_end_in::<GridDomain>(cfg, GenePool::grid(), pool, mutator)?;

    Ok((report, events))
}

/// `breed_to_end` on the domain `cfg` names, with Cascades made from `genes`. Also
/// returns the final state.
pub fn breed_to_end_in<D: Domain>(
    cfg: BreedConfig,
    genes: GenePool,
    pool: &ModelPool,
    mutator: Option<&dyn Mutator>,
) -> Result<(Report, Vec<Event>, Snapshot<D::Answer>)> {
    let mut b: Breeder<'_, D> = Breeder::start(cfg, genes, pool, mutator)?;
    let never = || false;
    let mut events = Vec::new();

    let reason = loop {
        let (status, ev) = b.step(&never)?;

        events.extend(ev);

        match status {
            Status::Continue => {}
            Status::Done(r) => break r,
            Status::Pending(_) => break "waiting for LLM answers".to_string(),
        }
    };

    let (report, ev) = b
        .finish(&never, &reason)?
        .ok_or_else(|| BrutError::Stopped("select phase stopped".to_string()))?;

    events.extend(ev);

    Ok((report, events, b.snapshot()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::StubModel;
    use crate::cascade::circuit::CircuitDomain;
    use crate::cascade::circuit_genes::CIRCUIT_PHRASES;
    use crate::cascade::circuit_genes::{CIRCUIT_BRIEF, circuit_pool};
    use crate::cascade::circuit_text::parse_circuit_prompt;
    use crate::cascade::designer::SimDesigner;
    use crate::cascade::domain::{CircuitConfig, Grade};
    use crate::cascade::genome::{
        MAX_SECTION_CHARS, Section, Stage, Traits, seed_cascades, seeds_for,
    };
    use crate::cascade::observe::SoulState;
    use crate::cascade::observe::{MindState, ValueSums};
    use crate::cascade::puzzle::{generate_panel, parse_prompt};
    use crate::cascade::qubit::Circuit;
    use crate::cascade::sim::SimModel;
    use crate::cascade::soul::{
        Inheritance, QUARTER_TURN, SoulConfig, SoulDraws, Strengths, imprint, new_soul_chance,
        souletics,
    };
    use crate::cascade::upbringing::{TEACHING_INTRO, UPBRINGING, model_word_in, teaching};
    use crate::cascade::variation::{GRID_BRIEF, MutatorBrief, Source};
    use std::collections::BTreeSet;

    fn sim_pool() -> ModelPool {
        let mut pool = ModelPool::new();

        pool.insert("sim:small", Box::new(SimModel::small()));
        pool.insert("sim:large", Box::new(SimModel::large()));

        pool
    }

    fn small_config(seed: u64) -> BreedConfig {
        BreedConfig {
            seed,
            generations: 3,
            children_per_generation: 4,
            search_puzzles: 12,
            select_puzzles: 12,
            finalists: 2,
            ..BreedConfig::default()
        }
    }

    /// `judge_in` on a grid puzzle, as the Stage 2c tests below call it.
    fn judge_with(
        ctx: Ctx,
        w: &Worker,
        p: &Puzzle,
        a: &Grid,
        b: &Grid,
        seed: u64,
        usage: &mut Usage,
    ) -> Judgment {
        judge_in(ctx, &GridDomain, w, p, (a, b), seed, usage)
    }

    #[test]
    fn bradley_terry_orders_by_wins_and_is_symmetric() {
        let r = bradley_terry(3, &[(0, 1, 1.0), (0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);

        assert!(r[0] > r[1] && r[1] > r[2], "{r:?}");

        let even = bradley_terry(2, &[(0, 1, 1.0), (1, 0, 1.0)]);

        assert!((even[0] - even[1]).abs() < 1e-9);

        let none = bradley_terry(2, &[(0, 1, 0.0)]);

        assert!(
            none.iter().all(|x| x.abs() < 1e-9),
            "zero weight changes nothing"
        );
    }

    #[test]
    fn dgm_weights_follow_the_paper() {
        let w = dgm_weights(&[0.5, 0.5, 0.9, 1.0], &[0, 1, 0, 0], 10.0, 0.5);

        assert!((w[0] - 0.5).abs() < 1e-12, "sigmoid(0) / 1");
        assert!((w[1] - 0.25).abs() < 1e-12, "sigmoid(0) / 2");
        assert!((w[2] - 1.0 / (1.0 + (-4.0f64).exp())).abs() < 1e-12);
        assert_eq!(w[3], 0.0, "a perfect score is not eligible");
    }

    #[test]
    fn run_cascade_is_deterministic_and_counts_calls() {
        let pool = sim_pool();
        let limits = CascadeLimits::default();
        let panel = generate_panel(5, 4);
        let (_, c) = seed_cascades().remove(0);

        for p in &panel {
            let a = run_cascade(&pool, &limits, &c, p, 9);
            let b = run_cascade(&pool, &limits, &c, p, 9);

            assert_eq!(a, b);
            assert_eq!(a.usage.calls, 1, "one Worker, one stage, no vote");
        }
    }

    #[test]
    fn wide_stage_votes_within_its_call_bound() {
        let pool = sim_pool();
        let limits = CascadeLimits::default();
        let panel = generate_panel(6, 6);
        let seeds = seed_cascades();
        let mut c = seeds[0].1.clone();

        c.stages[0]
            .workers
            .push(seeds[1].1.stages[0].workers[0].clone());
        c.stages[0]
            .workers
            .push(seeds[2].1.stages[0].workers[0].clone());

        for p in &panel {
            let run = run_cascade(&pool, &limits, &c, p, 3);

            assert!(run.usage.calls >= 3);
            assert!(run.usage.calls <= cascade_call_bound(&c));
        }
    }

    #[test]
    fn breeding_is_deterministic_and_keeps_its_budget() {
        let pool = sim_pool();
        let (r1, e1) = breed_to_end(small_config(7), &pool, None).expect("run 1");
        let (r2, e2) = breed_to_end(small_config(7), &pool, None).expect("run 2");

        assert_eq!(r1, r2);
        assert_eq!(e1, e2);
        assert!(r1.calls_used <= small_config(7).max_calls);
        assert!(r1.archive_size > seed_cascades().len());
        assert!(!r1.finalists.is_empty());
    }

    #[test]
    fn judges_are_never_related_to_the_judged() {
        let pool = sim_pool();
        let cfg = small_config(11);
        let mut b = Breeder::new(cfg, &pool, None).expect("breeder");
        let never = || false;

        while let (Status::Continue, _) = b.step(&never).expect("step") {}

        let kin = Kinship::new(b.archive());
        let index: HashMap<&str, usize> = b
            .archive()
            .iter()
            .enumerate()
            .map(|(i, m)| (m.id.as_str(), i))
            .collect();

        assert!(!b.comparisons().is_empty(), "peer round ran");

        for c in b.comparisons() {
            let j = index[c.judge.as_str()];
            let x = index[c.first.as_str()];
            let y = index[c.second.as_str()];

            assert!(!kin.related(j, x) && !kin.related(j, y) && !kin.related(x, y));
        }
    }

    #[test]
    fn resume_from_a_snapshot_equals_an_uninterrupted_run() {
        let pool = sim_pool();
        let never = || false;
        let mut straight = Breeder::new(small_config(13), &pool, None).expect("straight");
        let mut events_straight = Vec::new();

        while let (Status::Continue, ev) = straight.step(&never).expect("step") {
            events_straight.extend(ev);
        }

        let mut first = Breeder::new(small_config(13), &pool, None).expect("first");
        let mut events_split = Vec::new();

        for _ in 0..2 {
            let (status, ev) = first.step(&never).expect("step");

            assert_eq!(status, Status::Continue);
            events_split.extend(ev);
        }

        let snap = first.snapshot();
        let mut second = Breeder::resume(small_config(13), &pool, None, snap).expect("resume");

        while let (Status::Continue, ev) = second.step(&never).expect("step") {
            events_split.extend(ev);
        }

        assert_eq!(events_straight, events_split);
        assert_eq!(straight.snapshot(), second.snapshot());
    }

    #[test]
    fn resume_refuses_changed_settings_but_allows_more_generations() {
        let pool = sim_pool();
        let b = Breeder::new(small_config(2), &pool, None).expect("breeder");
        let snap = b.snapshot();
        let mut more = small_config(2);

        more.generations = 9;
        more.max_calls = 50_000;
        assert!(Breeder::resume(more, &pool, None, snap.clone()).is_ok());

        let mut other = small_config(2);

        other.judge_weight = 0.5;
        assert!(Breeder::resume(other, &pool, None, snap).is_err());
    }

    #[test]
    fn stop_aborts_without_committing() {
        let pool = sim_pool();
        let mut b = Breeder::new(small_config(3), &pool, None).expect("breeder");
        let always = || true;
        let (status, events) = b.step(&always).expect("step");

        assert_eq!(status, Status::Done("stopped".to_string()));
        assert!(events.is_empty());
        assert_eq!(b.next_generation(), 0);
        assert!(b.archive().is_empty());
        assert_eq!(b.calls_used(), 0, "stopped before the first puzzle");
    }

    /// A backend that always fails, as a crashed model server would.
    struct Broken;

    impl ModelBackend for Broken {
        fn name(&self) -> &str {
            "broken"
        }

        fn generate(&self, _req: &GenRequest) -> Result<GenResponse> {
            Err(BrutError::Eval("model server down".to_string()))
        }
    }

    #[test]
    fn failed_backend_calls_still_count_against_the_budget() {
        let mut pool = ModelPool::new();

        pool.insert("sim:small", Box::new(Broken));
        pool.insert("sim:large", Box::new(Broken));

        let limits = CascadeLimits::default();
        let panel = generate_panel(8, 2);
        let (_, c) = seed_cascades().remove(0);
        let run = run_cascade(&pool, &limits, &c, &panel[0], 1);

        assert_eq!(run.answer, None);
        assert_eq!(run.usage.calls, 1);
        assert!(run.usage.model_cost > 0.0);
        assert_eq!(run.usage.completion_tokens, 0);
    }

    #[test]
    fn finish_refuses_a_budget_lowered_on_resume() {
        let pool = sim_pool();
        let never = || false;
        let mut b = Breeder::new(small_config(17), &pool, None).expect("breeder");

        while let (Status::Continue, _) = b.step(&never).expect("step") {}

        let snap = b.snapshot();
        let tight = BreedConfig {
            max_calls: snap.calls + 1,
            ..small_config(17)
        };
        let mut resumed = Breeder::resume(tight, &pool, None, snap).expect("resume");
        let calls_before = resumed.calls_used();
        let refused = resumed.finish(&never, "all generations done");

        assert!(matches!(refused, Err(BrutError::Budget(_))), "{refused:?}");
        assert_eq!(resumed.calls_used(), calls_before, "no call was made");
    }

    #[test]
    fn seeds_adapt_to_a_host_allow_list() {
        let limits = CascadeLimits {
            models: vec![crate::cascade::genome::ModelSpec {
                tag: "sim:large".to_string(),
                cost_per_call: 0.001,
            }],
            think_ceiling_per_stage: vec![0],
            ..CascadeLimits::default()
        };

        let seeds = seeds_for(&limits).expect("adapted seeds");

        for (_, c) in &seeds {
            assert!(c.check(&limits).is_ok());
            assert!(
                c.stages
                    .iter()
                    .flat_map(|s| &s.workers)
                    .all(|w| w.model == "sim:large")
            );
            assert!(
                c.stages
                    .iter()
                    .flat_map(|s| &s.workers)
                    .all(|w| w.traits.think_tokens == 0)
            );
        }
    }

    #[test]
    fn tiny_budget_stops_before_spending() {
        let pool = sim_pool();
        let cfg = BreedConfig {
            max_calls: 10,
            ..small_config(4)
        };
        let (report, _) = breed_to_end(cfg, &pool, None).expect("run");

        assert_eq!(report.stop_reason, "call budget reached");
        assert_eq!(report.calls_used, 0);
    }

    #[test]
    fn llm_round_pauses_then_replays_recorded_answers() {
        let pool = sim_pool();
        let cfg = BreedConfig {
            llm_every: 1,
            llm_children: 2,
            ..small_config(21)
        };
        let empty = AnswerBook::new();
        let mut b = Breeder::new(cfg.clone(), &pool, Some(&empty)).expect("breeder");
        let never = || false;

        assert_eq!(b.step(&never).expect("gen 0").0, Status::Continue);

        let (status, events) = b.step(&never).expect("gen 1");
        let Status::Pending(requests) = status else {
            panic!("expected a pause for LLM answers, got {status:?}");
        };

        assert!(events.is_empty());
        assert_eq!(requests.len(), 2);
        assert_eq!(b.next_generation(), 1, "nothing committed");

        // Answer one request with a valid Cascade and one with junk.
        let mut book = AnswerBook::new();
        let mut child = seed_cascades()[3].1.clone();

        child.stages[0].workers[0].traits.think_tokens = 1024;

        let json = serde_json::to_string_pretty(&child).expect("json");

        book.insert(
            &request_key(&requests[0]),
            format!("Here you go:\n```json\n{json}\n```"),
        );
        book.insert(&request_key(&requests[1]), "no idea".to_string());

        let snap = b.snapshot();
        let mut resumed = Breeder::resume(cfg, &pool, Some(&book), snap).expect("resume");
        let (status, events) = resumed.step(&never).expect("gen 1 again");

        assert_eq!(status, Status::Continue);

        let exchanges: Vec<&Event> = events
            .iter()
            .filter(|e| matches!(e, Event::LlmExchange { .. }))
            .collect();

        assert_eq!(exchanges.len(), 2);
        assert!(
            resumed
                .archive()
                .iter()
                .any(|m| m.id == child.id() && m.op == Op::LlmMutate)
        );
    }

    #[test]
    fn stub_mutator_answers_are_rejected_and_replaced_by_random_children() {
        let pool = sim_pool();
        let stub = StubModel::new("stub:mutator");
        let live = LiveMutator(&stub);
        let cfg = BreedConfig {
            llm_every: 1,
            pause_for_llm: false,
            ..small_config(5)
        };
        let (report, events) = breed_to_end(cfg, &pool, Some(&live)).expect("run");
        let rejected = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Event::LlmExchange {
                        accepted: false,
                        ..
                    }
                )
            })
            .count();

        assert!(rejected > 0);
        assert!(report.archive_size > seed_cascades().len());
    }

    #[test]
    fn a_generation_reserves_its_worst_case_before_the_first_call() {
        let pool = sim_pool();
        let never = || false;
        let mut straight = Breeder::new(small_config(23), &pool, None).expect("straight");

        while let (Status::Continue, _) = straight.step(&never).expect("step") {}

        // Store what `persist` receives, as the binary does, then "crash": the breeder is
        // dropped and only the stored snapshot survives.
        let mut b = Breeder::new(small_config(23), &pool, None).expect("breeder");
        let mut stored: Vec<Snapshot> = Vec::new();
        let (status, _) = b
            .step_with(&never, &mut |s| {
                stored.push(s.clone());

                Ok(())
            })
            .expect("gen 0");

        assert_eq!(status, Status::Continue);
        assert_eq!(stored.len(), 1, "one reservation per generation");

        let reserved = stored[0].clone();

        assert_eq!(reserved.calls, 0, "stored before the first call");
        assert!(
            reserved.reserved >= b.calls_used(),
            "the worst case covers what was spent"
        );

        let mut resumed =
            Breeder::resume(small_config(23), &pool, None, reserved.clone()).expect("resume");

        assert_eq!(
            resumed.calls_used(),
            reserved.reserved,
            "a crash counts the reservation"
        );

        while let (Status::Continue, _) = resumed.step(&never).expect("step") {}

        assert_eq!(
            resumed.archive(),
            straight.archive(),
            "the repeated generation is the same"
        );
        assert_eq!(
            resumed.calls_used(),
            straight.calls_used() + reserved.reserved
        );
    }

    #[test]
    fn a_stopped_select_phase_never_pays_twice() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let pool = sim_pool();
        let never = || false;
        let mut straight = Breeder::new(small_config(29), &pool, None).expect("straight");

        while let (Status::Continue, _) = straight.step(&never).expect("step") {}

        let before = straight.calls_used();
        let (report, events) = straight
            .finish(&never, "all generations done")
            .expect("finish")
            .expect("not stopped");
        let select_calls = straight.calls_used() - before;

        // Same run, but the select phase is stopped twice before it completes.
        let mut b = Breeder::new(small_config(29), &pool, None).expect("breeder");

        while let (Status::Continue, _) = b.step(&never).expect("step") {}

        for limit in [5, 20] {
            let polls = AtomicUsize::new(0);
            let stop_later = || polls.fetch_add(1, Ordering::SeqCst) >= limit;

            assert!(
                b.finish(&stop_later, "all generations done")
                    .expect("finish")
                    .is_none()
            );
        }

        let (again, again_events) = b
            .finish(&never, "all generations done")
            .expect("finish")
            .expect("not stopped");

        assert_eq!(
            b.calls_used() - before,
            select_calls,
            "every puzzle was scored once"
        );
        assert_eq!(again, report);
        assert_eq!(again_events, events);

        // A finished run returns its report again, without events and without calls.
        let (third, third_events) = b
            .finish(&never, "all generations done")
            .expect("finish")
            .expect("not stopped");

        assert_eq!(third, report);
        assert!(third_events.is_empty(), "the log already holds them");
        assert_eq!(b.calls_used() - before, select_calls);
    }

    #[test]
    fn an_llm_round_does_not_pause_when_no_child_could_run() {
        let pool = sim_pool();
        let empty = AnswerBook::new();
        let cfg = BreedConfig {
            llm_every: 1,
            max_calls: 160,
            ..small_config(5)
        };
        let mut b = Breeder::new(cfg, &pool, Some(&empty)).expect("breeder");
        let never = || false;

        assert_eq!(b.step(&never).expect("gen 0").0, Status::Continue);

        let (status, _) = b.step(&never).expect("gen 1");

        assert_eq!(status, Status::Done("call budget reached".to_string()));
    }

    #[test]
    fn extreme_peer_settings_cannot_overrun_the_budget() {
        let pool = sim_pool();

        for opponents in [usize::MAX, 1 << 32] {
            let cfg = BreedConfig {
                peer_opponents: opponents,
                peer_puzzles_per_pair: 1 << 32,
                max_calls: 600,
                ..small_config(31)
            };
            let (report, _) = breed_to_end(cfg, &pool, None).expect("run");

            assert!(report.calls_used <= 600, "{}", report.calls_used);
        }
    }

    #[test]
    fn model_prices_must_lie_in_the_unit_range() {
        for price in [-0.05, 1.5, f64::NAN, f64::INFINITY] {
            let mut cfg = small_config(1);

            cfg.limits.models[0].cost_per_call = price;
            assert!(cfg.check().is_err(), "price {price}");
        }

        let mut cfg = small_config(1);

        cfg.limits.models[0].cost_per_call = 0.0;
        assert!(cfg.check().is_ok());
    }

    #[test]
    fn a_grid_breeder_refuses_another_domain() {
        let pool = sim_pool();
        let mut cfg = small_config(1);

        cfg.domain = DomainConfig::Circuits(Default::default());

        let refused = Breeder::new(cfg, &pool, None).err().map(|e| e.to_string());

        assert!(
            refused
                .as_deref()
                .is_some_and(|e| e.contains("another domain")),
            "{refused:?}"
        );
    }

    /// Solves like the small simulated model but never gives a verdict when judging.
    struct NoVerdicts(SimModel);

    impl ModelBackend for NoVerdicts {
        fn name(&self) -> &str {
            "no-verdicts"
        }

        fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
            if !req.prompt.starts_with("JUDGE") {
                return self.0.generate(req);
            }

            Ok(GenResponse {
                text: "Both look plausible to me.".to_string(),
                prompt_tokens: 10,
                completion_tokens: 6,
                stop_reason: StopReason::Stop,
                backend: "no-verdicts".to_string(),
                soul_decided: false,
            })
        }
    }

    #[test]
    fn a_judge_without_verdicts_is_scored_wrong_on_scored_pairs() {
        let mut pool = ModelPool::new();

        pool.insert("sim:small", Box::new(NoVerdicts(SimModel::small())));
        pool.insert("sim:large", Box::new(NoVerdicts(SimModel::large())));

        let mut b = Breeder::new(small_config(37), &pool, None).expect("breeder");
        let never = || false;

        while let (Status::Continue, _) = b.step(&never).expect("step") {}

        assert!(!b.comparisons().is_empty(), "scored pairs were judged");

        for c in b.comparisons() {
            assert_eq!(c.winner, None);
            assert_eq!(c.judge_correct, Some(false));

            // Only a scored pair (exactly one right answer) gives such a record.
            let i = b
                .search
                .iter()
                .position(|p| p.id == c.puzzle)
                .expect("search puzzle");
            let right = |id: &str| {
                let m = b.archive().iter().find(|m| m.id == id).expect("member");

                m.answers[i].as_ref() == Some(&b.search[i].test_output)
            };

            assert_ne!(right(&c.first), right(&c.second), "{c:?}");
        }

        assert!(
            b.archive()
                .iter()
                .any(|m| m.judge_wrong > 0 && m.judge_skill() < 0.5)
        );
    }

    #[test]
    fn judges_are_drawn_least_used_first() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let assigned = [0, 5, 0, 1];

        for _ in 0..50 {
            let j = least_used(&[0, 1, 2, 3], &assigned, &mut rng).expect("a judge");

            assert!(j == 0 || j == 2, "{j}");
        }

        assert_eq!(least_used(&[1, 3], &assigned, &mut rng), Some(3));
        assert_eq!(least_used(&[], &assigned, &mut rng), None);
    }

    #[test]
    fn every_judge_was_the_least_used_fair_choice() {
        let pool = sim_pool();
        let never = || false;

        for seed in [5, 6] {
            let mut b = Breeder::new(small_config(seed), &pool, None).expect("breeder");

            while let (Status::Continue, _) = b.step(&never).expect("step") {}

            let archive = b.archive();
            let index: HashMap<&str, usize> = archive
                .iter()
                .enumerate()
                .map(|(i, m)| (m.id.as_str(), i))
                .collect();
            let last = archive.iter().map(|m| m.generation).max().unwrap_or(0);

            // Replay each generation's plan: members are archived in generation order,
            // and simulated judges always give a verdict, so the comparisons are the plan.
            for g in 0..=last {
                let prefix = archive.iter().take_while(|m| m.generation <= g).count();
                let kin = Kinship::new(&archive[..prefix]);
                let new: Vec<usize> = (0..prefix)
                    .filter(|&i| archive[i].generation == g)
                    .collect();
                let mut assigned = vec![0usize; prefix];

                for c in b.comparisons().iter().filter(|c| c.generation == g) {
                    let (x, y) = (index[c.first.as_str()], index[c.second.as_str()]);
                    let j = index[c.judge.as_str()];
                    let fair =
                        |k: usize| k != x && k != y && !kin.related(k, x) && !kin.related(k, y);
                    let mut candidates: Vec<usize> =
                        new.iter().copied().filter(|&k| fair(k)).collect();

                    if candidates.is_empty() {
                        candidates = (0..prefix).filter(|&k| fair(k)).collect();
                    }

                    let fewest = candidates
                        .iter()
                        .map(|&k| assigned[k])
                        .min()
                        .expect("a fair judge");

                    assert!(fair(j), "seed {seed} gen {g}: judge {j} is related");
                    assert_eq!(
                        assigned[j], fewest,
                        "seed {seed} gen {g}: judge {j} was not least used"
                    );
                    assigned[j] += 1;
                }
            }
        }
    }

    /// Judges like an oracle: votes for the candidate equal to `right`.
    struct Oracle {
        right: Grid,
    }

    impl ModelBackend for Oracle {
        fn name(&self) -> &str {
            "oracle"
        }

        fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
            let first_is_right = crate::cascade::puzzle::parse_prompt(&req.prompt)
                .and_then(|p| p.candidates)
                .is_some_and(|(a, _)| a == self.right);

            Ok(GenResponse {
                text: if first_is_right {
                    "VERDICT A"
                } else {
                    "VERDICT B"
                }
                .to_string(),
                prompt_tokens: 1,
                completion_tokens: 2,
                stop_reason: StopReason::Stop,
                backend: "oracle".to_string(),
                soul_decided: false,
            })
        }
    }

    #[test]
    fn a_verdict_credits_the_answer_the_judge_chose_in_either_order() {
        let p = generate_panel(3, 1).remove(0);
        let right = p.test_output.clone();
        let mut wrong = right.clone();

        wrong[0][0] = (wrong[0][0] + 1) % 10;

        let mut pool = ModelPool::new();

        pool.insert(
            "sim:small",
            Box::new(Oracle {
                right: right.clone(),
            }),
        );

        let limits = CascadeLimits::default();
        let ctx = Ctx {
            pool: &pool,
            limits: &limits,
            upbringing: None,
        };
        let w = seed_cascades()[0].1.stages[0].workers[0].clone();
        let mut orders = std::collections::BTreeSet::new();

        for seed in 0..32u64 {
            let mut usage = Usage::default();

            assert_eq!(
                judge_with(ctx, &w, &p, &right, &wrong, seed, &mut usage),
                Judgment::Verdict(true)
            );
            assert_eq!(
                judge_with(ctx, &w, &p, &wrong, &right, seed, &mut usage),
                Judgment::Verdict(false)
            );
            orders.insert(ChaCha8Rng::seed_from_u64(derive_seed(seed, "order")).random_bool(0.5));
        }

        assert_eq!(orders.len(), 2, "both orders were shown");
    }

    #[test]
    fn a_select_phase_reserves_only_the_cascade_in_flight() {
        let pool = sim_pool();
        let never = || false;
        let mut b = Breeder::new(small_config(41), &pool, None).expect("breeder");

        while let (Status::Continue, _) = b.step(&never).expect("step") {}

        let before = b.calls_used();
        let picked = b.select_picks(&rank(&b.archive, &b.gene_ids));
        let mut stored: Vec<Snapshot> = Vec::new();
        let (report, _) = b
            .finish_with(&never, "all generations done", &mut |s| {
                stored.push(s.clone());

                Ok(())
            })
            .expect("finish")
            .expect("not stopped");

        // One reservation per scored Cascade, each for that Cascade alone.
        assert_eq!(stored.len(), picked.len());
        assert_eq!(stored[0].calls, before);

        for (s, &i) in stored.iter().zip(&picked) {
            assert_eq!(
                s.reserved,
                cascade_call_bound(&b.archive[i].cascade) * b.select.len() as u64
            );
        }

        // A crash inside the first Cascade costs its reservation once more; the resumed
        // select phase still fits and gives the same finalists.
        let first = stored[0].clone();
        let mut resumed =
            Breeder::resume(small_config(41), &pool, None, first.clone()).expect("resume");

        assert_eq!(resumed.calls_used(), first.calls + first.reserved);

        let (again, _) = resumed
            .finish(&never, "all generations done")
            .expect("fits")
            .expect("not stopped");

        assert_eq!(again.finalists, report.finalists);
    }

    #[test]
    fn a_finished_run_reports_again_under_a_lowered_cap() {
        let pool = sim_pool();
        let never = || false;
        let mut b = Breeder::new(small_config(43), &pool, None).expect("breeder");

        while let (Status::Continue, _) = b.step(&never).expect("step") {}

        let (report, _) = b
            .finish(&never, "all generations done")
            .expect("finish")
            .expect("not stopped");
        let snap = b.snapshot();
        let lowered = BreedConfig {
            max_calls: snap.calls - 1,
            ..small_config(43)
        };
        let mut resumed = Breeder::resume(lowered, &pool, None, snap).expect("resume");

        assert_eq!(resumed.select_need(), 0);

        let (again, events) = resumed
            .finish(&never, "all generations done")
            .expect("no call is needed")
            .expect("not stopped");

        assert_eq!(again, report);
        assert!(events.is_empty());
    }

    #[test]
    fn an_llm_round_pauses_only_when_its_answers_can_be_used() {
        let pool = sim_pool();
        let empty = AnswerBook::new();
        let never = || false;
        let cfg = |max_calls: u64| BreedConfig {
            llm_every: 1,
            max_calls,
            ..small_config(5)
        };
        let pauses = |max_calls: u64| {
            let mut b = Breeder::new(cfg(max_calls), &pool, Some(&empty)).expect("breeder");

            assert_eq!(b.step(&never).expect("gen 0").0, Status::Continue);
            matches!(b.step(&never).expect("gen 1").0, Status::Pending(_))
        };

        // Generation 0 needs 144 calls with its select reserve; scan above that.
        let threshold = (150..600)
            .find(|&cap| pauses(cap))
            .expect("a cap that pauses");

        assert!(!pauses(threshold - 1));

        // At the threshold, the cheapest answers (one small Worker each) are all used.
        let mut b = Breeder::new(cfg(threshold), &pool, Some(&empty)).expect("breeder");

        b.step(&never).expect("gen 0");

        let Status::Pending(requests) = b.step(&never).expect("gen 1").0 else {
            panic!("expected a pause");
        };
        let mut book = AnswerBook::new();

        for (k, req) in requests.iter().enumerate() {
            let mut c = seed_cascades()[0].1.clone();

            c.stages[0].workers[0].traits.temperature = 0.3 + 0.1 * k as f32;
            book.insert(&request_key(req), serde_json::to_string(&c).expect("json"));
        }

        let mut resumed =
            Breeder::resume(cfg(threshold), &pool, Some(&book), b.snapshot()).expect("resume");
        let (status, events) = resumed.step(&never).expect("gen 1 again");
        let used = events
            .iter()
            .filter(|e| matches!(e, Event::LlmExchange { accepted: true, .. }))
            .count();

        assert_eq!(status, Status::Continue, "the answers fit");
        assert_eq!(used, requests.len());
    }

    #[test]
    fn one_state_per_committed_generation_last_in_step() {
        let pool = sim_pool();
        let never = || false;
        let mut b = Breeder::new(small_config(47), &pool, None).expect("breeder");
        let mut committed = 0;
        let mut judged = false;

        loop {
            let before = b.calls_used();
            let (status, events) = b.step(&never).expect("step");

            if status != Status::Continue {
                assert!(events.is_empty(), "{status:?}");
                break;
            }

            let g = b.next_generation() - 1;
            let states = events
                .iter()
                .filter(|e| matches!(e, Event::State(_)))
                .count();
            let Some(Event::State(state)) = events.last() else {
                panic!("generation {g} does not end with a state event");
            };

            // The state follows the generation's summary and describes the committed
            // archive.
            assert_eq!(states, 1, "generation {g}");
            assert!(matches!(
                &events[events.len() - 2],
                Event::Generation { generation, .. } if *generation == g
            ));
            assert_eq!(state.generation, g);
            assert_eq!(state.performance.archive_size, b.archive().len());
            assert_eq!(
                state.performance.born,
                b.archive().iter().filter(|m| m.generation == g).count()
            );

            // Its cost is the generation's own: every call of the step, made by the
            // newborns on the search panel and by the peer round.
            let newborn_calls: u64 = events
                .iter()
                .filter_map(|e| match e {
                    Event::Born { usage, .. } => Some(usage.calls),
                    _ => None,
                })
                .sum();

            assert_eq!(state.cost.calls, b.calls_used() - before, "generation {g}");
            judged |= state.cost.calls > newborn_calls;
            committed += 1;
        }

        assert_eq!(committed, small_config(47).generations + 1);
        assert!(judged, "no peer round made a call");

        // A stopped step commits nothing, so it logs no state.
        let mut stopped = Breeder::new(small_config(47), &pool, None).expect("breeder");
        let always = || true;
        let (status, events) = stopped.step(&always).expect("step");

        assert_eq!(status, Status::Done("stopped".to_string()));
        assert!(events.is_empty());
    }

    #[test]
    fn lexicase_and_tournament_breed_deterministically_and_record_their_settings() {
        let pool = sim_pool();

        for sel in [ParentSelection::Lexicase, ParentSelection::Tournament] {
            let cfg = BreedConfig {
                selection: sel,
                selection_params: SelectionParams { lexicase_epsilon: 0.05, tournament_size: 2 },
                ..small_config(23)
            };
            let json = serde_json::to_string(&cfg).expect("json");

            assert!(json.contains(sel.name()) && json.contains("selection_params"), "{json}");
            assert_ne!(cfg.fixed_hash(), small_config(23).fixed_hash());

            let (r1, e1) = breed_to_end(cfg.clone(), &pool, None).expect("run");
            let (r2, e2) = breed_to_end(cfg.clone(), &pool, None).expect("run");

            assert_eq!(e1, e2);
            assert_eq!(r1.calls_used, r2.calls_used);
            assert!(r1.archive_size > seed_cascades().len());
        }

        // Settings that cannot select are refused.
        let bad = BreedConfig {
            selection_params: SelectionParams { tournament_size: 0, ..Default::default() },
            ..small_config(1)
        };

        assert!(bad.check().is_err());
        assert!(ParentSelection::parse("lexicase") == Some(ParentSelection::Lexicase));
        assert!(ParentSelection::parse("nope").is_none());
    }

    #[test]
    fn uniform_selection_is_uniform_and_ignores_ratings() {
        let pool = sim_pool();
        let never = || false;
        let uniform = BreedConfig {
            selection: ParentSelection::Uniform,
            ..small_config(19)
        };
        let mut b = Breeder::new(uniform.clone(), &pool, None).expect("breeder");

        assert_eq!(b.step(&never).expect("gen 0").0, Status::Continue);

        // Fitness and children counts do not matter: every member weighs the same.
        let n = b.archive.len();

        b.archive[0].fitness = 0.9;
        b.archive[1].children = 5;
        assert_eq!(b.selection_weights(), vec![1.0; n]);

        // The partner is the first draw other than the parent, whatever the ratings.
        let weights = b.selection_weights();
        let sel = b.selector();

        for seed in 0..32u64 {
            let mut replay = ChaCha8Rng::seed_from_u64(seed);
            let first = (0..DRAW_ATTEMPTS)
                .filter_map(|_| draw(&weights, &mut replay))
                .find(|&x| x != 0);

            for flip in [1.0, -1.0] {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);

                for (i, m) in b.archive.iter_mut().enumerate() {
                    m.rating = flip * i as f64;
                }

                assert_eq!(b.partner(0, &sel, &mut rng), first);
            }
        }

        // Under DGM selection the ratings decide between the drawn partners.
        let mut dgm = Breeder::new(small_config(19), &pool, None).expect("breeder");

        dgm.step(&never).expect("gen 0");

        let sel = dgm.selector();
        let mut follows_ratings = false;

        for seed in 0..32u64 {
            let mut picks = Vec::new();

            for flip in [1.0, -1.0] {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);

                for (i, m) in dgm.archive.iter_mut().enumerate() {
                    m.rating = flip * i as f64;
                }

                picks.push(dgm.partner(0, &sel, &mut rng));
            }

            follows_ratings |= picks[0] != picks[1];
        }

        assert!(follows_ratings);

        // The choice is part of the settings; DGM runs record and hash as before.
        let json = serde_json::to_string(&uniform).expect("json");
        let dgm_json = serde_json::to_string(&small_config(19)).expect("json");

        assert!(json.contains(r#""selection":"uniform""#), "{json}");
        assert!(!dgm_json.contains("selection"), "{dgm_json}");
        assert_ne!(uniform.fixed_hash(), small_config(19).fixed_hash());

        let (report, _) = breed_to_end(uniform, &pool, None).expect("run");

        assert!(report.archive_size > seed_cascades().len());
    }

    /// Admits no Worker on the large model.
    fn small_only(c: &Cascade) -> std::result::Result<(), String> {
        let mut workers = c.stages.iter().flat_map(|s| &s.workers);

        match workers.find(|w| w.model != "sim:small") {
            Some(w) => Err(format!("uses {}", w.model)),
            None => Ok(()),
        }
    }

    /// Answers every Mutator request with the first seed moved to the large model, with a
    /// temperature drawn from the request's seed.
    struct Large;

    impl Mutator for Large {
        fn answer(&self, req: &GenRequest) -> Option<GenResponse> {
            let mut c = seed_cascades()[0].1.clone();
            let w = &mut c.stages[0].workers[0];

            w.model = "sim:large".to_string();
            w.traits.temperature = (req.seed % 100) as f32 / 100.0;

            Some(GenResponse {
                text: serde_json::to_string(&c).ok()?,
                prompt_tokens: 10,
                completion_tokens: 10,
                stop_reason: StopReason::Stop,
                backend: "large".to_string(),
                soul_decided: false,
            })
        }
    }

    #[test]
    fn admit_rule_rejects_drafts() {
        let pool = sim_pool();
        let cfg = BreedConfig {
            llm_every: 1,
            ..small_config(53)
        };
        let genes = GenePool {
            admit: small_only,
            ..GenePool::grid()
        };
        let (_, events, snap) =
            breed_to_end_in::<GridDomain>(cfg.clone(), genes, &pool, Some(&Large)).expect("run");
        let exchanges: Vec<(bool, &str)> = events
            .iter()
            .filter_map(|e| match e {
                Event::LlmExchange {
                    accepted, reason, ..
                } => Some((*accepted, reason.as_str())),
                _ => None,
            })
            .collect();

        // Every answer from the Mutator is refused, and no random child slips through.
        assert!(!exchanges.is_empty());

        for (accepted, reason) in &exchanges {
            assert!(!accepted, "{reason}");
            assert!(
                reason.starts_with("not admitted: uses sim:large"),
                "{reason}"
            );
        }

        assert!(snap.archive.iter().all(|m| small_only(&m.cascade).is_ok()));
        assert!(snap.archive.iter().any(|m| m.op != Op::Seed));

        // Without the rule, the same answers are bred, and so is a random child on the
        // large model.
        let (_, events, open) =
            breed_to_end_in::<GridDomain>(cfg, GenePool::grid(), &pool, Some(&Large)).expect("run");
        let bred = events
            .iter()
            .any(|e| matches!(e, Event::LlmExchange { accepted: true, .. }));
        let random_large = open.archive.iter().any(|m| {
            !matches!(m.op, Op::Seed | Op::LlmMutate | Op::LlmCrossover)
                && small_only(&m.cascade).is_err()
        });

        assert!(bred && random_large);
    }

    /// A tiny domain for the generic breeder: task n asks for 2n. The exact answer solves
    /// the task and is worth 1; 2n + 2 is right in kind and worth 1/2; anything else is
    /// worth 0. An answer above 1000 beats a "proven bound": an integrity alarm.
    struct Doubling;

    /// One task of `Doubling`.
    #[derive(Clone, Debug, PartialEq)]
    struct Number {
        id: String,
        n: u64,
    }

    /// The value of `answer` on task `n`.
    fn doubled(n: u64, answer: u64) -> f64 {
        if answer == 2 * n {
            1.0
        } else if answer == 2 * n + 2 {
            0.5
        } else {
            0.0
        }
    }

    impl Domain for Doubling {
        type Task = Number;
        type Answer = u64;

        fn from_config(_cfg: &BreedConfig) -> Result<Doubling> {
            Ok(Doubling)
        }

        fn panel(&self, seed: u64, n: usize) -> Vec<Number> {
            (0..n)
                .map(|i| {
                    let n = derive_seed(seed, &format!("number/{i}")) % 40 + 1;

                    Number {
                        id: format!("task{i}-{n}"),
                        n,
                    }
                })
                .collect()
        }

        fn task_id<'t>(&self, task: &'t Number) -> &'t str {
            &task.id
        }

        fn solve_prompt(&self, task: &Number, previous: &[u64]) -> String {
            format!("DOUBLE {} AFTER {previous:?}", task.n)
        }

        fn judge_prompt(&self, task: &Number, a: &u64, b: &u64) -> String {
            format!("WHICH {} {a} {b}", task.n)
        }

        fn parse_answer(&self, _task: &Number, text: &str) -> Option<u64> {
            text.strip_prefix("ANSWER ")?.trim().parse().ok()
        }

        fn grade(&self, task: &Number, answer: &u64) -> Result<Grade> {
            if *answer > 1000 {
                let alarm = format!("{answer} beats a proven bound");

                return Err(BrutError::Integrity(alarm));
            }

            let value = doubled(task.n, *answer);

            Ok(Grade {
                correct: value > 0.0,
                solved: value == 1.0,
                value,
            })
        }

        fn prefer(&self, task: &Number, a: &u64, b: &u64) -> Option<bool> {
            let (x, y) = (doubled(task.n, *a), doubled(task.n, *b));

            (x != y).then_some(x > y)
        }

        fn graded(&self) -> bool {
            true
        }

        fn artefact(&self, answer: &u64) -> Option<String> {
            Some(answer.to_string())
        }

        fn self_check_text(&self) -> &'static str {
            "Check the double once."
        }
    }

    /// Answers `Doubling` prompts by the words of the system prompt: "overflow" gives
    /// 5000, "exact" or the domain's self-check text 2n, "over" 2n + 2, anything else n.
    /// A judge whose prompt says "careful", or carries the self-check text, picks the
    /// better answer; any other picks the first one shown.
    struct Scripted;

    impl ModelBackend for Scripted {
        fn name(&self) -> &str {
            "scripted"
        }

        fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
            let numbers: Vec<u64> = req
                .prompt
                .split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect();
            let n = numbers[0];
            let checked = req.system.contains(Doubling.self_check_text());
            let text = if req.prompt.starts_with("WHICH") {
                let (a, b) = (numbers[1], numbers[2]);
                let careful = checked || req.system.contains("careful");
                let first = !careful || doubled(n, a) >= doubled(n, b);

                if first { "VERDICT A" } else { "VERDICT B" }.to_string()
            } else if req.system.contains("overflow") {
                "ANSWER 5000".to_string()
            } else if checked || req.system.contains("exact") {
                format!("ANSWER {}", 2 * n)
            } else if req.system.contains("over") {
                format!("ANSWER {}", 2 * n + 2)
            } else {
                format!("ANSWER {n}")
            };

            Ok(GenResponse {
                text,
                prompt_tokens: 10,
                completion_tokens: 5,
                stop_reason: StopReason::Stop,
                backend: "scripted".to_string(),
                soul_decided: false,
            })
        }
    }

    fn scripted_pool() -> ModelPool {
        let mut pool = ModelPool::new();

        pool.insert("sim:small", Box::new(Scripted));
        pool.insert("sim:large", Box::new(Scripted));

        pool
    }

    /// A one-Worker Cascade on the small model with this strategy.
    fn doubler(strategy: &str) -> Cascade {
        let section = |name: &str, text: &str| Section {
            name: name.to_string(),
            text: text.to_string(),
        };

        Cascade {
            stages: vec![Stage {
                workers: vec![Worker {
                    model: "sim:small".to_string(),
                    sections: vec![
                        section("role", "You double numbers."),
                        section("strategy", strategy),
                        section("format", "Answer with the line ANSWER and the number."),
                    ],
                    traits: Traits::default(),
                }],
            }],
        }
    }

    fn doubling_seeds() -> Vec<(String, Cascade)> {
        vec![
            ("plain".to_string(), doubler("Repeat the number.")),
            ("exact".to_string(), doubler("Give the exact double.")),
            ("over".to_string(), doubler("Go over the double by two.")),
            ("careful".to_string(), doubler("Repeat it; be careful.")),
        ]
    }

    /// The phrase pool of `Doubling`; its admission rule refuses the last text.
    const DOUBLING_PHRASES: &[(&str, &str)] = &[
        ("strategy", "Give the exact double."),
        ("strategy", "Go over the double by two."),
        ("strategy", "Repeat the number."),
        ("check", "Be careful when you judge."),
        ("style", "Be brief."),
        ("style", "This text is forbidden."),
    ];

    static DOUBLING_BRIEF: MutatorBrief = MutatorBrief {
        system: "You edit Cascades that double numbers.",
        context: "A Cascade doubles numbers.\n\n",
        format: "Format: as for grid puzzles.\n\n",
        mutate_task: "Task: make the parent double better. ",
        crossover_task: "Task: merge Parent A and Parent B. ",
    };

    /// Admits no section that says "forbidden".
    fn no_forbidden(c: &Cascade) -> std::result::Result<(), String> {
        let said = c
            .stages
            .iter()
            .flat_map(|s| &s.workers)
            .flat_map(|w| &w.sections)
            .any(|s| s.text.contains("forbidden"));

        if said {
            Err("a section says forbidden".to_string())
        } else {
            Ok(())
        }
    }

    fn doubling_genes() -> GenePool {
        GenePool {
            seeds: doubling_seeds,
            phrases: DOUBLING_PHRASES,
            brief: &DOUBLING_BRIEF,
            admit: no_forbidden,
            families_of: None,
        }
    }

    /// Answers every Mutator request: a mutation with an "exact" Cascade that has its own
    /// style line, a crossover with one whose style text is forbidden.
    struct Writer;

    impl Mutator for Writer {
        fn answer(&self, req: &GenRequest) -> Option<GenResponse> {
            let style = if req.prompt.contains("Parent A") {
                "This text is forbidden.".to_string()
            } else {
                format!("Variant {}.", req.seed % 997)
            };
            let mut c = doubler("Give the exact double.");

            c.stages[0].workers[0].sections.push(Section {
                name: "style".to_string(),
                text: style,
            });

            Some(GenResponse {
                text: serde_json::to_string(&c).ok()?,
                prompt_tokens: 10,
                completion_tokens: 10,
                stop_reason: StopReason::Stop,
                backend: "writer".to_string(),
                soul_decided: false,
            })
        }
    }

    #[test]
    fn a_generic_domain_breeds_on_graded_values_and_preferences() {
        let pool = scripted_pool();
        let cfg = BreedConfig {
            llm_every: 1,
            ..small_config(59)
        };
        let (report, events, snap) =
            breed_to_end_in::<Doubling>(cfg.clone(), doubling_genes(), &pool, Some(&Writer))
                .expect("run");
        let panel = Doubling.panel(derive_seed(cfg.seed, "panel/search"), cfg.search_puzzles);
        let n = panel.len() as f64;
        let answer = |id: &str, task: &str| -> u64 {
            let m = snap.archive.iter().find(|m| m.id == id).expect("member");
            let i = panel.iter().position(|t| t.id == task).expect("task");

            m.answers[i].expect("an answer")
        };

        // Fitness counts the graded value, so a Cascade can earn without solving.
        for m in &snap.archive {
            let quality: f64 = m
                .answers
                .iter()
                .zip(&panel)
                .filter_map(|(a, t)| a.map(|a| doubled(t.n, a)))
                .sum();
            let cost = cfg.token_cost_per_1k * (m.usage.completion_tokens as f64 / n) / 1000.0
                + m.usage.model_cost / n;
            let fitness =
                (1.0 - cfg.judge_weight) * quality / n + cfg.judge_weight * m.judge_skill() - cost;

            assert_eq!(m.quality, Some(quality), "{}", m.id);
            assert!((m.fitness - fitness).abs() < 1e-12, "{}", m.id);
        }

        assert!(
            snap.archive
                .iter()
                .any(|m| m.solved == 0 && m.quality > Some(0.0))
        );

        // Born events carry the graded value and the answers as text.
        for e in &events {
            if let Event::Born {
                id,
                quality,
                answers,
                ..
            } = e
            {
                let m = snap.archive.iter().find(|m| &m.id == id).expect("archived");
                let texts: Vec<Option<String>> =
                    m.answers.iter().map(|a| a.map(|a| a.to_string())).collect();

                assert_eq!(*quality, m.quality);
                assert_eq!(answers.as_ref(), Some(&texts));
            }
        }

        // A judgment is scored exactly when the domain prefers one answer, and it is
        // right when the judge chose that one. Pairs without an exact answer count too.
        let mut inexact = 0;

        assert!(!snap.comparisons.is_empty());

        for c in &snap.comparisons {
            let task = panel.iter().find(|t| t.id == c.puzzle).expect("task");
            let (a, b) = (answer(&c.first, &c.puzzle), answer(&c.second, &c.puzzle));
            let preferred = Doubling.prefer(task, &a, &b);

            assert_eq!(c.judge_correct.is_some(), preferred.is_some(), "{c:?}");

            if let (Some(winner), Some(a_better)) = (&c.winner, preferred) {
                let better = if a_better { &c.first } else { &c.second };

                assert_eq!(c.judge_correct, Some(winner == better), "{c:?}");
            }

            if preferred.is_some() && a != 2 * task.n && b != 2 * task.n {
                inexact += 1;
            }
        }

        assert!(inexact > 0);

        // The Mutator gets the domain's brief, and its forbidden answers are refused.
        let mut refused = 0;

        for e in &events {
            if let Event::LlmExchange {
                request,
                accepted,
                reason,
                ..
            } = e
            {
                assert_eq!(request.system, DOUBLING_BRIEF.system);

                if !accepted {
                    assert!(reason.starts_with("not admitted: "), "{reason}");
                    refused += 1;
                }
            }
        }

        assert!(refused > 0);
        assert!(snap.archive.iter().any(|m| m.op == Op::LlmMutate));
        assert!(
            snap.archive
                .iter()
                .all(|m| no_forbidden(&m.cascade).is_ok())
        );

        // Random children take their texts from the domain's phrases, never the grid's.
        let random: Vec<&str> = snap
            .archive
            .iter()
            .filter(|m| !matches!(m.op, Op::Seed | Op::LlmMutate | Op::LlmCrossover))
            .flat_map(|m| m.cascade.stages.iter().flat_map(|s| &s.workers))
            .flat_map(|w| w.sections.iter().map(|s| s.text.as_str()))
            .collect();
        let grid_phrases = GenePool::grid().phrases;

        assert!(
            ["Be brief.", "Be careful when you judge."]
                .iter()
                .any(|t| random.contains(t))
        );
        assert!(
            random
                .iter()
                .all(|t| grid_phrases.iter().all(|(_, g)| g != t))
        );

        // A finalist's select quality is its summed value on the select panel.
        let select = Doubling.panel(derive_seed(cfg.seed, "panel/select"), cfg.select_puzzles);

        for f in &report.finalists {
            let got = &snap.select_answers[&f.id];
            let values: Vec<f64> = select
                .iter()
                .filter_map(|t| got.get(&t.id).copied().flatten().map(|a| doubled(t.n, a)))
                .collect();
            let quality: f64 = values.iter().sum();
            let solved = values.iter().filter(|&&v| v == 1.0).count();

            assert_eq!(f.select_quality, Some(quality), "{}", f.id);
            assert_eq!(f.select_solved, solved, "{}", f.id);
        }

        // Every generation logs its state, with quality rates; finalists carry theirs.
        let states: Vec<&UniverseState> = events
            .iter()
            .filter_map(|e| match e {
                Event::State(s) => Some(s),
                _ => None,
            })
            .collect();

        assert_eq!(states.len(), cfg.generations + 1);
        assert!(states.iter().all(|s| s.performance.quality_best.is_some()));
        assert!(report.finalists.iter().all(|f| f.select_quality.is_some()));

        // The state of a generic run reads back as it was written.
        let json = serde_json::to_string(&snap).expect("json");
        let back: Snapshot<u64> = serde_json::from_str(&json).expect("back");

        assert_eq!(back, snap);
    }

    #[test]
    fn a_generic_cascade_checks_and_votes_by_its_domain() {
        let pool = scripted_pool();
        let limits = CascadeLimits::default();
        let ctx = Ctx {
            pool: &pool,
            limits: &limits,
            upbringing: None,
        };
        let task = Number {
            id: "seven".to_string(),
            n: 7,
        };
        let plain = doubler("Repeat the number.");
        let mut checked = plain.clone();

        checked.stages[0].workers[0].traits.self_check = true;

        // The self-check Trait adds the domain's text, not the grid's. Only the domain's
        // text makes this Worker answer the exact double.
        let w = &checked.stages[0].workers[0];
        let grid_request = worker_request(w, Doubling.solve_prompt(&task, &[]), 3);
        let unchecked = run_cascade_in(&Doubling, &pool, &limits, &plain, &task, 3);
        let run = run_cascade_in(&Doubling, &pool, &limits, &checked, &task, 3);
        let grid_reply = Scripted.generate(&grid_request).expect("reply");

        assert_eq!(grid_reply.text, "ANSWER 7");
        assert_eq!(unchecked.answer, Some(7));
        assert_eq!(run.answer, Some(14));
        assert_eq!(run.usage.calls, 1);

        // As a judge, the Worker gets the domain's text too, so it picks the better
        // answer. Asked in both orders, it sees each answer first once. A verdict is true
        // when the first answer of the pair won.
        let (worse, better) = (7, 14);
        let mut usage = Usage::default();
        let verdicts = [(&worse, &better), (&better, &worse)]
            .map(|pair| judge_in(ctx, &Doubling, w, &task, pair, 11, &mut usage));

        assert_eq!(
            verdicts,
            [Judgment::Verdict(false), Judgment::Verdict(true)]
        );

        // Two Workers that disagree vote by the domain's judge prompt. Both judge with
        // care, so the better answer wins, although it is not the first Worker's. That
        // takes two answers and two votes.
        let worker = |strategy: &str| doubler(strategy).stages[0].workers[0].clone();
        let pair = Cascade {
            stages: vec![Stage {
                workers: vec![
                    worker("Repeat it; be careful."),
                    worker("Go over the double by two; be careful."),
                ],
            }],
        };
        let run = run_cascade_in(&Doubling, &pool, &limits, &pair, &task, 5);

        assert_eq!(run.answer, Some(16));
        assert_eq!(run.usage.calls, 4);
        assert_eq!(run.usage.calls, cascade_call_bound(&pair));
    }

    #[test]
    fn a_generic_run_resumes_like_a_straight_one() {
        let pool = scripted_pool();
        let never = || false;
        let cfg = BreedConfig {
            llm_every: 2,
            ..small_config(67)
        };
        let start = || -> Breeder<'_, Doubling> {
            Breeder::start(cfg.clone(), doubling_genes(), &pool, Some(&Writer)).expect("breeder")
        };
        let mut straight = start();
        let mut events_straight = Vec::new();

        while let (Status::Continue, ev) = straight.step(&never).expect("step") {
            events_straight.extend(ev);
        }

        let mut first = start();
        let mut events_split = Vec::new();

        for _ in 0..2 {
            events_split.extend(first.step(&never).expect("step").1);
        }

        let snap = first.snapshot();
        let mut second: Breeder<'_, Doubling> =
            Breeder::resume_from(cfg.clone(), doubling_genes(), &pool, Some(&Writer), snap)
                .expect("resume");

        while let (Status::Continue, ev) = second.step(&never).expect("step") {
            events_split.extend(ev);
        }

        assert_eq!(events_straight, events_split);
        assert_eq!(straight.snapshot(), second.snapshot());

        // The panel comes from the domain.
        let panel = Doubling.panel(derive_seed(cfg.seed, "panel/search"), cfg.search_puzzles);

        assert_eq!(second.search_panel(), &panel[..]);
        assert_eq!(second.domain().task_id(&panel[0]), panel[0].id);
    }

    /// Seeds of which the second answers beyond the proven bound.
    fn overflowing() -> Vec<(String, Cascade)> {
        vec![
            ("exact".to_string(), doubler("Give the exact double.")),
            ("overflow".to_string(), doubler("Let the answer overflow.")),
            ("plain".to_string(), doubler("Repeat the number.")),
        ]
    }

    #[test]
    fn an_integrity_alarm_commits_nothing() {
        let pool = scripted_pool();
        let genes = GenePool {
            seeds: overflowing,
            ..doubling_genes()
        };
        let mut b: Breeder<'_, Doubling> =
            Breeder::start(small_config(61), genes, &pool, None).expect("breeder");
        let never = || false;
        let mut stored = Vec::new();
        let alarm = b.step_with(&never, &mut |s| {
            stored.push(s.clone());

            Ok(())
        });

        assert!(matches!(alarm, Err(BrutError::Integrity(_))), "{alarm:?}");

        // Nothing is committed. The calls made, one per task for each of the first two
        // seeds, stay spent, and the reservation stored before them covers them.
        let snap = b.snapshot();

        assert!(snap.archive.is_empty() && snap.comparisons.is_empty());
        assert_eq!(snap.next_generation, 0);
        assert_eq!(b.calls_used(), 2 * 12);
        assert_eq!(snap.calls, b.calls_used());
        assert_eq!(stored.len(), 1);
        assert!(stored[0].reserved >= b.calls_used());
    }

    /// An answer type without a `Default`, which the `Domain` trait allows.
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct Label(String);

    #[test]
    fn a_snapshot_reads_back_for_answers_without_a_default() {
        let (_, cascade) = seed_cascades().swap_remove(0);
        let member = Member {
            id: cascade.id(),
            cascade,
            parents: Vec::new(),
            op: Op::Seed,
            generation: 0,
            answers: vec![Some(Label("a".to_string())), None],
            solved: 1,
            quality: None,
            usage: Usage::default(),
            children: 0,
            judge_correct: 0,
            judge_wrong: 0,
            fitness: 0.5,
            rating: 0.0,
            soul_sources: None,
            imprinted: None,
            soul_now: None,
            teaching: None,
            authors: None,
            votes: None,
            values: None,
            values_now: None,
            effect: None,
        };
        let mut snap = Snapshot {
            config_hash: "h".to_string(),
            next_generation: 1,
            archive: vec![member],
            comparisons: Vec::new(),
            calls: 2,
            mutator_answers: 0,
            reserved: 0,
            select_answers: BTreeMap::new(),
            finished: false,
        };

        snap.select_answers
            .entry(snap.archive[0].id.clone())
            .or_default()
            .insert("t".to_string(), Some(Label("b".to_string())));

        // This compiles only while reading a snapshot asks nothing more of the answer
        // type than the `Domain` trait promises.
        let json = serde_json::to_string(&snap).expect("json");
        let back: Snapshot<Label> = serde_json::from_str(&json).expect("back");

        assert_eq!(back, snap);

        // A snapshot without select answers, as written before they were kept, has none.
        let mut old: serde_json::Value = serde_json::from_str(&json).expect("value");

        old.as_object_mut()
            .expect("object")
            .remove("select_answers");

        let back: Snapshot<Label> = serde_json::from_value(old).expect("old");

        assert!(back.select_answers.is_empty());
        assert_eq!(back.archive, snap.archive);
    }

    /// Gives a Worker a Soul of seven phases.
    fn give_phases(t: &mut Traits) {
        t.phase_state = Some(PhaseState(vec![1, 2, 3, 4, 5, 6, 7]));
    }

    /// Gives a Worker a Coupling strength of 0.2.
    fn give_coupling(t: &mut Traits) {
        t.coupling = 200;
    }

    /// Gives a Worker a Coupling strength above the ceiling, which `check` refuses too.
    fn give_too_much_coupling(t: &mut Traits) {
        t.coupling = 5000;
    }

    /// Gives a Worker both a Soul and a Coupling strength.
    fn give_both(t: &mut Traits) {
        give_phases(t);
        give_coupling(t);
    }

    /// `c` with `give` applied to every Worker.
    fn souled(c: &Cascade, give: fn(&mut Traits)) -> Cascade {
        let mut c = c.clone();

        for w in c.stages.iter_mut().flat_map(|s| &mut s.workers) {
            give(&mut w.traits);
        }

        c
    }

    /// Answers every Mutator request with the last grid seed and a note of the request's
    /// seed, every Worker changed by `.0`.
    struct SoulWriter(fn(&mut Traits));

    impl Mutator for SoulWriter {
        fn answer(&self, req: &GenRequest) -> Option<GenResponse> {
            let mut c = seed_cascades()[3].1.clone();

            c.stages[0].workers[0].sections.push(Section {
                name: "note".to_string(),
                text: format!("Variant {}.", req.seed),
            });

            Some(GenResponse {
                text: serde_json::to_string(&souled(&c, self.0)).ok()?,
                prompt_tokens: 10,
                completion_tokens: 10,
                stop_reason: StopReason::Stop,
                backend: "soul-writer".to_string(),
                soul_decided: false,
            })
        }
    }

    /// Whether each LLM answer was accepted, and why, in a short grid run with an LLM round
    /// whose Mutator is `m`.
    fn exchanges_with(m: &dyn Mutator) -> Vec<(bool, String)> {
        let cfg = BreedConfig {
            llm_every: 1,
            generations: 1,
            ..small_config(61)
        };
        let (_, events) = breed_to_end(cfg, &sim_pool(), Some(m)).expect("run");

        events
            .iter()
            .filter_map(|e| match e {
                Event::LlmExchange {
                    accepted, reason, ..
                } => Some((*accepted, reason.clone())),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_mutator_never_sees_or_writes_a_soul() {
        let limits = CascadeLimits::default();
        let seeds = seed_cascades();
        let (a, b) = (&seeds[0].1, &seeds[1].1);
        let (sa, sb) = (souled(a, give_both), souled(b, give_phases));

        for brief in [&GRID_BRIEF, &CIRCUIT_BRIEF] {
            let plain = [
                llm_mutate_request_with(brief, a, 0.25, &limits, 42),
                llm_crossover_request_with(brief, (a, 0.25), (b, 0.5), &limits, 43),
            ];
            let with_souls = [
                llm_mutate_request_with(brief, &sa, 0.25, &limits, 42),
                llm_crossover_request_with(brief, (&sa, 0.25), (&sb, 0.5), &limits, 43),
            ];

            for (p, s) in plain.iter().zip(&with_souls) {
                assert_eq!(
                    serde_json::to_string(p).expect("json"),
                    serde_json::to_string(s).expect("json")
                );
                assert_eq!(request_key(p), request_key(s));
                assert!(!s.prompt.contains("phase_state") && !s.prompt.contains("coupling"));
            }
        }

        // An answer that writes a Soul or a Coupling strength is refused, before any other
        // rule: also when the Coupling strength is above the ceiling.
        for give in [
            give_phases,
            give_coupling,
            give_too_much_coupling,
            give_both,
        ] {
            let exchanges = exchanges_with(&SoulWriter(give));

            assert_eq!(exchanges.len(), 2);

            for (accepted, reason) in &exchanges {
                assert!(!accepted);
                assert_eq!(reason, "not admitted: the Mutator does not write Souls");
            }
        }

        // The same answers without either field are bred.
        let exchanges = exchanges_with(&SoulWriter(|_| {}));

        assert_eq!(exchanges.len(), 2);
        assert!(
            exchanges
                .iter()
                .all(|(accepted, reason)| *accepted && reason == "ok"),
            "{exchanges:?}"
        );
    }

    /// The grid seeds with a Soul on the colour seed's Worker.
    fn seeds_with_phases() -> Vec<(String, Cascade)> {
        let mut seeds = seed_cascades();

        give_phases(&mut seeds[2].1.stages[0].workers[0].traits);
        seeds
    }

    /// The grid seeds with a Coupling strength on the colour seed's Worker.
    fn seeds_with_coupling() -> Vec<(String, Cascade)> {
        let mut seeds = seed_cascades();

        give_coupling(&mut seeds[2].1.stages[0].workers[0].traits);
        seeds
    }

    /// The reason of an `Invalid` error, or a panic naming what was expected.
    fn invalid<T>(result: Result<T>, what: &str) -> String {
        match result {
            Err(BrutError::Invalid(reason)) => reason,
            Err(e) => panic!("{what}: expected Invalid, got {e}"),
            Ok(_) => panic!("{what}: expected a refusal"),
        }
    }

    #[test]
    fn a_run_with_souls_needs_a_domain_that_names_its_families() {
        let pool = scripted_pool();
        let task = Doubling.panel(1, 1).remove(0);
        let cfg = BreedConfig {
            soul: Some(SoulConfig::arm_s()),
            ..small_config(71)
        };
        let reason = invalid(
            Breeder::<Doubling>::start(cfg, doubling_genes(), &pool, None),
            "no families",
        );

        assert!(Doubling.families().is_empty() && Doubling.strengths(&task).is_empty());
        assert!(reason.contains("families"), "{reason}");

        // Without Souls the same domain breeds as before.
        assert!(
            Breeder::<Doubling>::start(small_config(71), doubling_genes(), &pool, None).is_ok()
        );
    }

    #[test]
    fn a_run_without_souls_refuses_souled_workers() {
        let pool = sim_pool();
        let never = || false;

        // A seed.
        for seeds in [
            seeds_with_phases as fn() -> Vec<(String, Cascade)>,
            seeds_with_coupling,
        ] {
            let genes = GenePool {
                seeds,
                ..GenePool::grid()
            };
            let reason = invalid(
                Breeder::<GridDomain>::start(small_config(67), genes, &pool, None),
                "seed",
            );

            assert!(
                reason.contains("seed colour") && reason.contains("Soul"),
                "{reason}"
            );
        }

        // A resumed member.
        let mut b = Breeder::new(small_config(67), &pool, None).expect("breeder");

        b.step(&never).expect("gen 0");

        let snap = b.snapshot();

        assert!(Breeder::resume(small_config(67), &pool, None, snap.clone()).is_ok());

        for give in [give_phases, give_coupling] {
            let mut tampered = snap.clone();
            let id = tampered.archive[1].id.clone();

            give(&mut tampered.archive[1].cascade.stages[0].workers[0].traits);

            let reason = invalid(
                Breeder::resume(small_config(67), &pool, None, tampered),
                "resumed member",
            );

            assert!(reason.contains(&id) && reason.contains("Soul"), "{reason}");
        }

        // A record of Upbringing too (ticket 07).
        for which in 0..2 {
            let mut tampered = snap.clone();
            let m = &mut tampered.archive[1];

            if which == 0 {
                m.imprinted = Some(vec![PhaseState(vec![1; 7])]);
            } else {
                m.teaching = Some("Solved examples.".to_string());
            }

            let reason = invalid(
                Breeder::resume(small_config(67), &pool, None, tampered),
                "resumed record",
            );

            assert!(reason.contains("record of Souls"), "{reason}");
        }

        // A Mutator answer.
        for give in [give_phases, give_coupling] {
            for (accepted, reason) in exchanges_with(&SoulWriter(give)) {
                assert!(!accepted && reason == SOUL_NOT_ADMITTED, "{reason}");
            }
        }

        // A Run with Souls starts, since Souletics gives Souls to seeds and children
        // (ticket 03). It refuses to resume the archive of a Run without Souls, whose
        // Workers have none, and seeds that bring their own.
        let souled_cfg = BreedConfig {
            soul: Some(SoulConfig::arm_s()),
            ..small_config(67)
        };
        let resumed = Snapshot {
            config_hash: souled_cfg.fixed_hash(),
            ..snap
        };

        assert!(Breeder::new(souled_cfg.clone(), &pool, None).is_ok());

        let reason = invalid(
            Breeder::resume(souled_cfg.clone(), &pool, None, resumed),
            "resumed",
        );

        assert!(reason.contains("has no Soul"), "{reason}");

        let genes = GenePool {
            seeds: seeds_with_phases,
            ..GenePool::grid()
        };
        let reason = invalid(
            Breeder::<GridDomain>::start(souled_cfg, genes, &pool, None),
            "seed with a Soul",
        );

        assert!(
            reason.contains("Souletics gives the seeds their Souls"),
            "{reason}"
        );
    }

    /// The simulated designer under both tags, for circuit runs.
    fn designer_pool() -> ModelPool {
        let mut pool = ModelPool::new();

        pool.insert("sim:small", Box::new(SimDesigner::small()));
        pool.insert("sim:large", Box::new(SimDesigner::large()));

        pool
    }

    /// A short circuit run: every verified target once on each panel.
    fn circuit_config(seed: u64) -> BreedConfig {
        BreedConfig {
            seed,
            generations: 3,
            children_per_generation: 4,
            search_puzzles: 11,
            select_puzzles: 11,
            finalists: 2,
            domain: DomainConfig::Circuits(CircuitConfig::default()),
            ..BreedConfig::default()
        }
    }

    /// What `draft_children` drafts in generations 1 to 3 of a run: every child's id,
    /// parents and operator, every LLM exchange, and the word position of the generation's
    /// stream afterwards. The generations are committed by `step` in between.
    fn drafting_record<D: Domain>(mut b: Breeder<'_, D>) -> String {
        let never = || false;
        let mut out = String::new();

        assert_eq!(b.step(&never).expect("gen 0").0, Status::Continue);

        for _ in 0..3 {
            let g = b.next_generation();
            let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(b.cfg.seed, &format!("gen/{g}")));
            let Drafted::Ready(drafts, events, used) =
                b.draft_children(g, &mut rng).expect("draft")
            else {
                panic!("generation {g} waits for answers");
            };

            for d in &drafts {
                out.push_str(&format!(
                    "g={g} id={} parents={} op={:?}\n",
                    d.cascade.id(),
                    d.parents.join(","),
                    d.op
                ));
            }

            for e in &events {
                if let Event::LlmExchange {
                    key,
                    accepted,
                    reason,
                    ..
                } = e
                {
                    out.push_str(&format!("g={g} llm={key} {accepted} {reason}\n"));
                }
            }

            out.push_str(&format!("g={g} used={used} pos={}\n", rng.get_word_pos()));
            assert_eq!(b.step(&never).expect("step").0, Status::Continue);
        }

        out
    }

    /// The digests of `drafting_record`, recorded before step 2 gave the operators their
    /// source maps (ticket 03): a grid run, a grid run with an LLM round in every
    /// generation, and a circuit run.
    const DRAFTING_GRID: &str = "30e66fd96682aacb1a9ac062d7ae8bbede74bed3310bd192d53aaca6be2603d6";
    const DRAFTING_GRID_LLM: &str =
        "a11e9a3873d3a1bf8996487467bbe4594e75ec359ff900633879297a42001816";
    const DRAFTING_CIRCUITS: &str =
        "54cf9aae5b2f4e733c8f33a115ed5e0e02869aa9e6188fa63bbddb6229df209b";

    #[test]
    fn drafting_without_souls_is_unchanged() {
        let pool = sim_pool();
        let designers = designer_pool();
        let writer = SoulWriter(|_| {});
        let llm = BreedConfig {
            llm_every: 1,
            ..small_config(43)
        };
        let records = [
            drafting_record(Breeder::new(small_config(41), &pool, None).expect("grid")),
            drafting_record(Breeder::new(llm, &pool, Some(&writer)).expect("grid llm")),
            drafting_record(
                Breeder::<CircuitDomain>::start(
                    circuit_config(47),
                    circuit_pool(),
                    &designers,
                    None,
                )
                .expect("circuits"),
            ),
        ];
        let digests = records.each_ref().map(|r| sha256_hex(r.as_bytes()));

        assert_eq!(
            digests,
            [DRAFTING_GRID, DRAFTING_GRID_LLM, DRAFTING_CIRCUITS],
            "{records:#?}"
        );
    }

    /// `cfg` with the Soul settings `soul`.
    fn with_souls(cfg: BreedConfig, soul: SoulConfig) -> BreedConfig {
        BreedConfig {
            soul: Some(soul),
            ..cfg
        }
    }

    /// The birth Soul of every Worker of `c`, in Stage order; empty without one.
    fn birth_phases(c: &Cascade) -> Vec<Vec<u16>> {
        c.stages
            .iter()
            .flat_map(|s| &s.workers)
            .map(|w| w.traits.phase_state.as_deref().unwrap_or_default().to_vec())
            .collect()
    }

    /// The Coupling strength of every Worker of `c`, in Stage order.
    fn couplings(c: &Cascade) -> Vec<u16> {
        c.stages
            .iter()
            .flat_map(|s| &s.workers)
            .map(|w| w.traits.coupling)
            .collect()
    }

    /// The circular distance of two phases, in phase units.
    fn apart(a: u16, b: u16) -> u16 {
        let d = a.wrapping_sub(b);

        d.min(d.wrapping_neg())
    }

    /// The member with this id.
    fn member<'m, A>(archive: &'m [Member<A>], id: &str) -> &'m Member<A> {
        archive
            .iter()
            .find(|m| m.id == id)
            .expect("an archived member")
    }

    /// The Soul records of `m`.
    fn records<A>(m: &Member<A>) -> &[SoulSource] {
        m.soul_sources.as_deref().expect("Soul records")
    }

    /// A breeder stepped until its generations are done.
    fn bred<D: Domain>(mut b: Breeder<'_, D>) -> Breeder<'_, D> {
        let never = || false;

        while let (Status::Continue, _) = b.step(&never).expect("step") {}

        b
    }

    /// The archive of a grid run with these settings, bred to its last generation.
    fn grid_archive(cfg: BreedConfig) -> Vec<Member> {
        let pool = sim_pool();

        bred(Breeder::new(cfg, &pool, None).expect("breeder"))
            .archive()
            .to_vec()
    }

    /// A grid run of six generations with six children each.
    fn longer_config(seed: u64) -> BreedConfig {
        BreedConfig {
            generations: 6,
            children_per_generation: 6,
            ..small_config(seed)
        }
    }

    /// The gene ids of the members born in generation 1, in birth order, and the lengths of
    /// the Souls of all Workers then.
    fn generation_one<D: Domain>(mut b: Breeder<'_, D>) -> (Vec<String>, BTreeSet<usize>) {
        let never = || false;

        for _ in 0..2 {
            assert_eq!(b.step(&never).expect("step").0, Status::Continue);
        }

        let ids = b
            .archive()
            .iter()
            .filter(|m| m.generation == 1)
            .map(|m| m.cascade.gene_id())
            .collect();
        let lengths = b
            .archive()
            .iter()
            .flat_map(|m| birth_phases(&m.cascade))
            .map(|p| p.len())
            .collect();

        (ids, lengths)
    }

    #[test]
    fn souls_do_not_touch_the_genes() {
        let pool = sim_pool();
        let designers = designer_pool();
        let circuits = |cfg: BreedConfig| {
            Breeder::<CircuitDomain>::start(cfg, circuit_pool(), &designers, None)
                .expect("circuits")
        };
        let plain_grid = generation_one(Breeder::new(small_config(83), &pool, None).expect("grid"));
        let plain_circuits = generation_one(circuits(circuit_config(89)));

        assert_eq!(plain_grid.1, BTreeSet::from([0]));
        assert!(!plain_grid.0.is_empty() && !plain_circuits.0.is_empty());

        for arm in [
            SoulConfig::arm_s(),
            SoulConfig::arm_v(),
            SoulConfig::arm_k(),
        ] {
            let grid = generation_one(
                Breeder::new(with_souls(small_config(83), arm.clone()), &pool, None).expect("grid"),
            );
            let circuit = generation_one(circuits(with_souls(circuit_config(89), arm)));

            assert_eq!(grid.0, plain_grid.0, "the same children, in the same order");
            assert_eq!(circuit.0, plain_circuits.0);
            assert_eq!(grid.1, BTreeSet::from([7]), "a phase per grid family");
            assert_eq!(circuit.1, BTreeSet::from([8]), "a phase per circuit family");
        }
    }

    #[test]
    fn copy_mix_and_drift() {
        for drift in [512u16, 0] {
            let soul = SoulConfig {
                phase_drift: drift,
                ..SoulConfig::arm_s()
            };
            let archive = grid_archive(with_souls(longer_config(97), soul));
            let birth = |r: &WorkerRef| {
                birth_phases(&member(&archive, &r.member).cascade)[r.worker].clone()
            };
            let (mut copies, mut mixes, mut moved, mut from_both) = (0, 0, 0, 0);

            for m in archive.iter().filter(|m| m.op != Op::Seed) {
                for (phases, rec) in birth_phases(&m.cascade).iter().zip(records(m)) {
                    if rec.new {
                        continue;
                    }

                    let sources: Vec<Vec<u16>> = rec.sources.iter().map(birth).collect();

                    match sources.as_slice() {
                        [one] => {
                            for (&p, &q) in phases.iter().zip(one) {
                                assert!(apart(p, q) <= drift, "{p} from {q}, drift {drift}");
                                moved += usize::from(p != q);
                            }

                            copies += 1;
                        }
                        [a, b] => {
                            let mut took = [false; 2];

                            for f in 0..phases.len() {
                                let (p, x, y) = (phases[f], a[f], b[f]);

                                assert!(
                                    apart(p, x) <= drift || apart(p, y) <= drift,
                                    "{p} from {x} or {y}"
                                );
                                took[0] |= p == x && p != y;
                                took[1] |= p == y && p != x;
                            }

                            from_both += usize::from(took == [true, true]);
                            mixes += 1;
                        }
                        other => panic!("{} sources", other.len()),
                    }
                }
            }

            assert!(copies > 0 && mixes > 0, "{copies} copies, {mixes} mixes");

            if drift == 0 {
                assert_eq!(moved, 0, "with phase_drift 0 a copy is exact");
                assert!(from_both > 0, "a mix takes phases of both sources");
            } else {
                assert!(moved > 0, "the phases drift");
            }
        }
    }

    #[test]
    fn new_souls_come_at_hoppes_rate() {
        // Over 100 000 draws the share of new Souls meets θ / (θ + N), within 3 standard
        // errors; θ = 0 gives none.
        let p = [0u16; 7];
        let source = [Inheritance {
            phases: &p,
            coupling: 0,
        }];

        for (theta, urn) in [
            (1.0, 4usize),
            (1.0, 100),
            (1.0, 1000),
            (0.0, 4),
            (0.0, 1000),
        ] {
            let cfg = SoulConfig {
                new_soul_theta: theta,
                ..SoulConfig::arm_s()
            };
            let mut rng =
                ChaCha8Rng::seed_from_u64(derive_seed(5, &format!("hoppe/{theta}/{urn}")));
            let n = 100_000;
            let new = (0..n)
                .filter(|_| souletics(&cfg, &source, urn, &SoulDraws::draw(&mut rng, 7)).new)
                .count();
            let expected = theta / (theta + urn as f64);
            let se = (expected * (1.0 - expected) / n as f64).sqrt();
            let share = new as f64 / n as f64;

            if theta == 0.0 {
                assert_eq!(new, 0, "θ = 0 never gives a new Soul");
            } else {
                assert!(
                    (share - expected).abs() < 3.0 * se,
                    "N {urn}: {share} against {expected}"
                );
            }
        }

        // In a Run, N counts the archive's Workers and the child Workers drawn before: each
        // coin, replayed from its stream, decides as θ / (θ + N) says.
        let theta = 20.0;
        let soul = SoulConfig {
            new_soul_theta: theta,
            ..SoulConfig::arm_s()
        };
        let cfg = with_souls(longer_config(101), soul);
        let archive = grid_archive(cfg.clone());
        let (mut new, mut old) = (0, 0);

        for m in archive.iter().filter(|m| m.generation == 0) {
            assert!(records(m).iter().all(|r| r.new), "a seed's Soul is new");
        }

        for g in 1..=cfg.generations {
            let mut urn: usize = archive
                .iter()
                .filter(|m| m.generation < g)
                .map(|m| m.cascade.worker_count())
                .sum();

            for (i, m) in archive.iter().filter(|m| m.generation == g).enumerate() {
                let mut rng =
                    ChaCha8Rng::seed_from_u64(derive_seed(cfg.seed, &format!("soul/{g}/{i}")));

                for rec in records(m) {
                    let draws = SoulDraws::draw(&mut rng, 7);

                    assert_eq!(
                        rec.new,
                        draws.new_coin < new_soul_chance(theta, urn),
                        "generation {g}, child {i}, {urn} in the urn"
                    );

                    urn += 1;
                    new += usize::from(rec.new);
                    old += usize::from(!rec.new);
                }
            }
        }

        assert!(new > 0 && old > 0, "{new} new, {old} inherited");
    }

    #[test]
    fn a_parent_passes_on_its_birth_soul() {
        // Changing the parents' current Souls changes none of their children's birth Souls
        // (Q65). Only Upbringing reads the current Souls: with imprinting on, the children's
        // imprinted Souls differ; with it off, the current Souls pass on nothing at all.
        let pool = sim_pool();
        let never = || false;

        for imprint in [true, false] {
            let cfg = with_souls(
                BreedConfig {
                    generations: 5,
                    ..small_config(103)
                },
                SoulConfig {
                    imprint,
                    ..SoulConfig::arm_s()
                },
            );
            let mut b = Breeder::new(cfg.clone(), &pool, None).expect("breeder");

            for _ in 0..3 {
                b.step(&never).expect("step");
            }

            let snap = b.snapshot();
            let mut moved = snap.clone();

            for m in &mut moved.archive {
                for soul in m.soul_now.as_mut().expect("current Souls") {
                    for p in &mut soul.0 {
                        *p = p.wrapping_add(12345);
                    }
                }
            }

            // The Born events of each generation after the resume.
            let children = |snap: Snapshot| {
                let mut b = Breeder::resume(cfg.clone(), &pool, None, snap).expect("resume");
                let mut born = Vec::new();

                while let (Status::Continue, events) = b.step(&never).expect("step") {
                    born.push(
                        events
                            .into_iter()
                            .filter(|e| matches!(e, Event::Born { .. }))
                            .collect::<Vec<Event>>(),
                    );
                }

                born
            };
            // What Souletics gave a Born event: the genome with its birth Souls, and where
            // each came from.
            let births = |events: &[Event]| -> Vec<(Cascade, Option<Vec<SoulSource>>)> {
                events
                    .iter()
                    .filter_map(|e| match e {
                        Event::Born {
                            cascade,
                            soul_sources,
                            ..
                        } => Some((cascade.clone(), soul_sources.clone())),
                        _ => None,
                    })
                    .collect()
            };
            let imprints = |events: &[Event]| -> Vec<Option<Vec<PhaseState>>> {
                events
                    .iter()
                    .filter_map(|e| match e {
                        Event::Born { imprinted, .. } => Some(imprinted.clone()),
                        _ => None,
                    })
                    .collect()
            };
            let straight = children(snap.clone());
            let changed = children(moved.clone());

            assert!(!straight.is_empty() && !straight[0].is_empty());
            assert_ne!(snap.archive, moved.archive);
            assert_eq!(
                births(&changed[0]),
                births(&straight[0]),
                "the next generation's birth Souls"
            );

            if imprint {
                assert_ne!(imprints(&changed[0]), imprints(&straight[0]));
            } else {
                assert_eq!(changed, straight, "the current Souls pass on nothing");
            }
        }
    }

    #[test]
    fn kappa_at_birth() {
        let delta = 20;
        let archive = grid_archive(with_souls(longer_config(127), SoulConfig::arm_s()));
        let (mut copied, mut moved) = (0, 0);

        for m in &archive {
            for (k, (rec, &kappa)) in records(m).iter().zip(&couplings(&m.cascade)).enumerate() {
                if m.op == Op::Seed {
                    assert_eq!(kappa, 200, "κ of the seeds");
                    continue;
                }

                // From one of its sources, moved by at most δ.
                let near: Vec<u16> = rec
                    .sources
                    .iter()
                    .map(|r| couplings(&member(&archive, &r.member).cascade)[r.worker])
                    .filter(|&s| s.abs_diff(kappa) <= delta)
                    .collect();

                assert!(!near.is_empty(), "{} Worker {k}: κ {kappa}", m.id);
                copied += usize::from(near.contains(&kappa));
                moved += usize::from(!near.contains(&kappa));
            }
        }

        assert!(copied > 0 && moved > 0, "{copied} {moved}");

        // Clamped at 0 and at 1000.
        for (start, edge) in [(0u16, 0u16), (1000, 1000)] {
            let soul = SoulConfig {
                kappa_seed: start,
                kappa_drift: 1000,
                ..SoulConfig::arm_s()
            };
            let all: Vec<u16> = grid_archive(with_souls(longer_config(131), soul))
                .iter()
                .filter(|m| m.op != Op::Seed)
                .flat_map(|m| couplings(&m.cascade))
                .collect();

            assert!(all.iter().all(|&k| k <= 1000), "{all:?}");
            assert!(all.contains(&edge), "clamped at {edge}: {all:?}");
        }

        // With κ fixed every Worker keeps κ of the seeds; arm K's all have 0.
        let fixed = SoulConfig {
            kappa_seed: 300,
            kappa_fixed: true,
            ..SoulConfig::arm_s()
        };

        for (soul, kappa) in [(fixed, 300), (SoulConfig::arm_k(), 0)] {
            let all: BTreeSet<u16> = grid_archive(with_souls(longer_config(137), soul))
                .iter()
                .flat_map(|m| couplings(&m.cascade))
                .collect();

            assert_eq!(all, BTreeSet::from([kappa]));
        }
    }

    #[test]
    fn the_arms_share_their_soul_draws() {
        let pool = sim_pool();
        let never = || false;
        let arms = [
            SoulConfig::arm_s(),
            SoulConfig::arm_v(),
            SoulConfig::arm_k(),
        ]
        .map(|arm| {
            let mut b =
                Breeder::new(with_souls(longer_config(139), arm), &pool, None).expect("breeder");

            for _ in 0..2 {
                b.step(&never).expect("step");
            }

            b.archive().to_vec()
        });
        let [s, v, k] = &arms;

        assert_eq!(s.len(), v.len());
        assert_eq!(s.len(), k.len());
        assert!(s.iter().any(|m| m.generation == 1));

        // Generations 0 and 1: the same children with the same phases, new Souls and
        // sources, in every arm; κ differs in arm K alone.
        for ((x, y), z) in s.iter().zip(v).zip(k) {
            let sources = |m: &Member| -> Vec<(bool, Vec<usize>)> {
                records(m)
                    .iter()
                    .map(|r| (r.new, r.sources.iter().map(|w| w.worker).collect()))
                    .collect()
            };

            assert_eq!(x.cascade.gene_id(), y.cascade.gene_id());
            assert_eq!(x.cascade.gene_id(), z.cascade.gene_id());
            assert_eq!(birth_phases(&x.cascade), birth_phases(&y.cascade));
            assert_eq!(birth_phases(&x.cascade), birth_phases(&z.cascade));
            assert_eq!(sources(x), sources(y));
            assert_eq!(sources(x), sources(z));
            assert_eq!(x.cascade, y.cascade, "arm V differs only in the readout");
            assert!(couplings(&z.cascade).iter().all(|&c| c == 0));
        }
    }

    /// Answers every Mutator request with the genes of the last grid seed, which the archive
    /// holds from generation 0 on.
    struct Repeater;

    impl Mutator for Repeater {
        fn answer(&self, _req: &GenRequest) -> Option<GenResponse> {
            Some(GenResponse {
                text: serde_json::to_string(&seed_cascades()[3].1).ok()?,
                prompt_tokens: 10,
                completion_tokens: 10,
                stop_reason: StopReason::Stop,
                backend: "repeater".to_string(),
                soul_decided: false,
            })
        }
    }

    #[test]
    fn novelty_is_by_genes() {
        let pool = sim_pool();
        let cfg = with_souls(
            BreedConfig {
                llm_every: 1,
                generations: 2,
                ..small_config(109)
            },
            SoulConfig::arm_s(),
        );
        let (_, events, snap) =
            breed_to_end_in::<GridDomain>(cfg, GenePool::grid(), &pool, Some(&Repeater))
                .expect("run");
        let repeated = seed_cascades()[3].1.clone();
        let seed = snap
            .archive
            .iter()
            .find(|m| m.cascade.gene_id() == repeated.id())
            .expect("the seed is archived");

        // The archived seed has another Soul, so only its genes repeat.
        assert_ne!(seed.id, repeated.id());

        let exchanges: Vec<(bool, String)> = events
            .iter()
            .filter_map(|e| match e {
                Event::LlmExchange {
                    accepted, reason, ..
                } => Some((*accepted, reason.clone())),
                _ => None,
            })
            .collect();

        assert!(!exchanges.is_empty());

        for (accepted, reason) in exchanges {
            assert!(
                !accepted && reason == "duplicate of an archived Cascade",
                "{reason}"
            );
        }

        // No two members share their genes.
        let genes: HashSet<String> = snap.archive.iter().map(|m| m.cascade.gene_id()).collect();

        assert_eq!(genes.len(), snap.archive.len());
    }

    #[test]
    fn a_soul_has_n_phases() {
        let pool = sim_pool();
        let never = || false;
        let cfg = with_souls(small_config(113), SoulConfig::arm_s());
        let mut b = Breeder::new(cfg.clone(), &pool, None).expect("breeder");

        b.step(&never).expect("gen 0");

        let snap = b.snapshot();

        assert!(Breeder::resume(cfg.clone(), &pool, None, snap.clone()).is_ok());

        let id = snap.archive[1].id.clone();

        for phases in [None, Some(6usize), Some(8)] {
            let mut tampered = snap.clone();

            tampered.archive[1].cascade.stages[0].workers[0]
                .traits
                .phase_state = phases.map(|n| PhaseState(vec![7; n]));

            let reason = invalid(
                Breeder::resume(cfg.clone(), &pool, None, tampered),
                "a Soul of the wrong size",
            );

            assert!(reason.contains(&id) && reason.contains('7'), "{reason}");
        }

        // The imprinted and the current Souls must fit as well (ticket 07).
        for imprinted in [true, false] {
            let mut tampered = snap.clone();
            let m = &mut tampered.archive[2];
            let souls = if imprinted {
                &mut m.imprinted
            } else {
                &mut m.soul_now
            };

            souls.as_mut().expect("Souls")[0].0.pop();

            let reason = invalid(
                Breeder::resume(cfg.clone(), &pool, None, tampered),
                "a Soul of the wrong size",
            );
            let what = if imprinted { "imprinted" } else { "current" };

            assert!(
                reason.contains(&snap.archive[2].id) && reason.contains(what),
                "{reason}"
            );
        }

        let mut tampered = snap.clone();

        tampered.archive[3].imprinted = None;
        assert!(Breeder::resume(cfg.clone(), &pool, None, tampered).is_err());

        // A teaching section only in a Run that teaches, and only with the text of a
        // section.
        let mut taught = snap.clone();

        taught.archive[1].teaching = Some("Solved examples.".to_string());

        let reason = invalid(
            Breeder::resume(cfg.clone(), &pool, None, taught.clone()),
            "a section without teaching",
        );

        assert!(reason.contains("does not teach"), "{reason}");

        let teaching_cfg = with_souls(small_config(113), teaching_arm());
        let resumed = |archive: Vec<Member>| {
            Breeder::resume(
                teaching_cfg.clone(),
                &pool,
                None,
                Snapshot {
                    config_hash: teaching_cfg.fixed_hash(),
                    archive,
                    ..snap.clone()
                },
            )
        };

        assert!(resumed(taught.archive.clone()).is_ok());

        for bad in ["Hidden\u{200B}text.", "Line\u{7}bell.", ""] {
            let mut archive = taught.archive.clone();

            archive[1].teaching = Some(bad.to_string());

            let reason = invalid(resumed(archive), "a section that breaks the rules");

            assert!(reason.contains("teaching section"), "{reason}");
        }
    }

    /// A backend that records every request as JSON before its model answers it. It reads
    /// phases when `reads` is true and its model reads them.
    struct Recording {
        model: Box<dyn ModelBackend>,
        log: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        reads: bool,
    }

    impl ModelBackend for Recording {
        fn name(&self) -> &str {
            self.model.name()
        }

        fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
            let json = serde_json::to_string(req).expect("request json");

            self.log.lock().expect("log").push(json);
            self.model.generate(req)
        }

        fn reads_phases(&self) -> bool {
            self.reads && self.model.reads_phases()
        }
    }

    /// A pool of two recording backends on one log: `small` and `large` under the tags of
    /// the simulated models, reading phases when `reads` is true.
    fn recording_pool(
        small: Box<dyn ModelBackend>,
        large: Box<dyn ModelBackend>,
        reads: bool,
    ) -> (ModelPool, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut pool = ModelPool::new();

        for (tag, model) in [("sim:small", small), ("sim:large", large)] {
            pool.insert(
                tag,
                Box::new(Recording {
                    model,
                    log: log.clone(),
                    reads,
                }),
            );
        }

        (pool, log)
    }

    /// Every request a run with these settings sends, sorted, and its archive.
    fn requests<D: Domain>(
        cfg: BreedConfig,
        genes: GenePool,
        small: Box<dyn ModelBackend>,
        large: Box<dyn ModelBackend>,
    ) -> (Vec<String>, Vec<Member<D::Answer>>) {
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut pool = ModelPool::new();

        pool.insert(
            "sim:small",
            Box::new(Recording {
                model: small,
                log: log.clone(),
                reads: false,
            }),
        );
        pool.insert(
            "sim:large",
            Box::new(Recording {
                model: large,
                log: log.clone(),
                reads: false,
            }),
        );

        let (_, _, snap) = breed_to_end_in::<D>(cfg, genes, &pool, None).expect("run");
        let mut seen = log.lock().expect("log").clone();

        seen.sort();

        (seen, snap.archive)
    }

    #[test]
    fn calls_are_seeded_by_genes() {
        // Arms whose members have the same genes and other Souls: κ differs in arm K, the
        // phases with another drift. The count decides every vote, so the outcomes agree.
        let arms = [
            SoulConfig::arm_s(),
            SoulConfig::arm_k(),
            SoulConfig {
                phase_drift: 0,
                ..SoulConfig::arm_s()
            },
        ]
        .map(|arm| SoulConfig { vote: false, ..arm });
        let grid = |soul: Option<SoulConfig>| {
            requests::<GridDomain>(
                BreedConfig {
                    soul,
                    ..small_config(149)
                },
                GenePool::grid(),
                Box::new(SimModel::small()),
                Box::new(SimModel::large()),
            )
        };
        let circuits = |soul: Option<SoulConfig>| {
            requests::<CircuitDomain>(
                BreedConfig {
                    soul,
                    ..circuit_config(151)
                },
                circuit_pool(),
                Box::new(SimDesigner::small()),
                Box::new(SimDesigner::large()),
            )
        };
        let (plain_grid, plain_archive) = grid(None);
        let (plain_circuits, _) = circuits(None);

        assert!(!plain_grid.is_empty() && !plain_circuits.is_empty());

        let mut souls = BTreeSet::new();

        for arm in arms {
            let (sent, archive) = grid(Some(arm.clone()));

            assert_eq!(sent, plain_grid, "the same requests with the same seeds");
            assert_eq!(circuits(Some(arm)).0, plain_circuits);

            // The same genes as without Souls, with other Souls in every arm.
            for (m, plain) in archive.iter().zip(&plain_archive) {
                assert_eq!(m.cascade.gene_id(), plain.id);
                assert_ne!(m.id, plain.id);
            }

            souls.insert(
                archive
                    .iter()
                    .map(|m| m.id.clone())
                    .collect::<Vec<String>>(),
            );
        }

        assert_eq!(souls.len(), 3, "each arm gives its members other Souls");
    }

    #[test]
    fn switches_give_their_degenerate_case() {
        // With `copy` off every child Worker gets a new Soul, whose origin is itself.
        let fresh = SoulConfig {
            copy: false,
            ..SoulConfig::arm_s()
        };

        for m in grid_archive(with_souls(longer_config(157), fresh)) {
            for (k, rec) in records(&m).iter().enumerate() {
                assert!(rec.new, "{} Worker {k}", m.id);
                assert_eq!(
                    rec.origin,
                    WorkerRef {
                        member: m.id.clone(),
                        worker: k
                    }
                );
            }
        }

        // With `mix` off a crossover child copies its first source: exactly, with no drift.
        let unmixed = SoulConfig {
            mix: false,
            phase_drift: 0,
            ..SoulConfig::arm_s()
        };
        let archive = grid_archive(with_souls(longer_config(163), unmixed));
        let mut paired = 0;

        for m in archive.iter().filter(|m| m.op != Op::Seed) {
            for (phases, rec) in birth_phases(&m.cascade).iter().zip(records(m)) {
                if !rec.new && rec.sources.len() == 2 {
                    let first = &rec.sources[0];

                    assert_eq!(
                        phases,
                        &birth_phases(&member(&archive, &first.member).cascade)[first.worker]
                    );
                    assert_eq!(
                        rec.origin,
                        records(member(&archive, &first.member))[first.worker].origin
                    );
                    paired += 1;
                }
            }
        }

        assert!(paired > 0, "a crossover child with two sources");

        // With θ = 0 no child gets a new Soul.
        let none = SoulConfig {
            new_soul_theta: 0.0,
            ..SoulConfig::arm_s()
        };

        for m in grid_archive(with_souls(longer_config(167), none))
            .iter()
            .filter(|m| m.op != Op::Seed)
        {
            assert!(records(m).iter().all(|r| !r.new), "{}", m.id);
        }
    }

    /// The last entry of every Worker in the `souls` events, by member id and Worker index.
    fn last_souls(events: &[Event]) -> BTreeMap<(String, usize), PhaseState> {
        let mut last = BTreeMap::new();

        for e in events {
            if let Event::Souls { changed, .. } = e {
                for c in changed {
                    last.insert((c.member.clone(), c.worker), c.phases.clone());
                }
            }
        }

        last
    }

    #[test]
    fn a_souled_run_records_its_souls_and_resumes_like_a_straight_one() {
        let pool = sim_pool();
        let never = || false;
        let cfg = with_souls(small_config(173), SoulConfig::arm_s());
        let mut straight = Breeder::new(cfg.clone(), &pool, None).expect("straight");
        let mut events_straight = Vec::new();

        while let (Status::Continue, ev) = straight.step(&never).expect("step") {
            events_straight.extend(ev);
        }

        let mut first = Breeder::new(cfg.clone(), &pool, None).expect("first");
        let mut events_split = Vec::new();

        for _ in 0..2 {
            events_split.extend(first.step(&never).expect("step").1);
        }

        let json = serde_json::to_string(&first.snapshot()).expect("json");
        let snap: Snapshot = serde_json::from_str(&json).expect("back");
        let mut second = Breeder::resume(cfg, &pool, None, snap).expect("resume");

        while let (Status::Continue, ev) = second.step(&never).expect("step") {
            events_split.extend(ev);
        }

        assert_eq!(events_straight, events_split);
        assert_eq!(straight.snapshot(), second.snapshot());

        // Every member records its Souls, and its Born event the same; a seed's Workers have
        // new Souls of their own origin; the current Souls start as the imprinted Souls
        // (ticket 07), and each then is its last entry in the `souls` events (ticket 05).
        let archive = straight.archive();
        type Logged<'e> = (&'e Option<Vec<SoulSource>>, &'e Option<Vec<PhaseState>>);
        let born: Vec<Logged<'_>> = events_straight
            .iter()
            .filter_map(|e| match e {
                Event::Born {
                    soul_sources,
                    imprinted,
                    ..
                } => Some((soul_sources, imprinted)),
                _ => None,
            })
            .collect();
        let last = last_souls(&events_straight);

        assert_eq!(born.len(), archive.len());
        assert!(!last.is_empty(), "the Observations moved some Soul");

        for (m, (logged, imprinted)) in archive.iter().zip(born) {
            assert_eq!(logged, &m.soul_sources);
            assert_eq!(imprinted, &m.imprinted);
            assert_eq!(records(m).len(), m.cascade.worker_count());

            let now: Vec<Vec<u16>> = m
                .soul_now
                .as_ref()
                .expect("current Souls")
                .iter()
                .map(|p| p.0.clone())
                .collect();
            let expected: Vec<Vec<u16>> = plain(&m.imprinted)
                .into_iter()
                .enumerate()
                .map(|(k, start)| last.get(&(m.id.clone(), k)).map_or(start, |p| p.0.clone()))
                .collect();

            assert_eq!(now, expected);

            if m.op == Op::Seed {
                for (k, r) in records(m).iter().enumerate() {
                    assert!(r.new && r.sources.is_empty());
                    assert_eq!(
                        r.origin,
                        WorkerRef {
                            member: m.id.clone(),
                            worker: k
                        }
                    );
                }
            }
        }
    }

    #[test]
    fn the_ranking_breaks_ties_by_genes() {
        // Members of equal fitness and rating are ordered by their keys, the gene ids, not
        // by the ids that carry their Souls.
        let (_, c) = seed_cascades().swap_remove(0);
        let tied = |id: &str| Member::<Grid> {
            id: id.to_string(),
            cascade: c.clone(),
            parents: Vec::new(),
            op: Op::Seed,
            generation: 0,
            answers: Vec::new(),
            solved: 0,
            quality: None,
            usage: Usage::default(),
            children: 0,
            judge_correct: 0,
            judge_wrong: 0,
            fitness: 0.5,
            rating: 0.0,
            soul_sources: None,
            imprinted: None,
            soul_now: None,
            teaching: None,
            authors: None,
            votes: None,
            values: None,
            values_now: None,
            effect: None,
        };
        let archive = [tied("a"), tied("b")];
        let keys = |x: &str, y: &str| [x.to_string(), y.to_string()];

        assert_eq!(rank(&archive, &keys("y", "x")), [1, 0]);
        assert_eq!(rank(&archive, &keys("x", "y")), [0, 1]);

        // The breeder keeps the key of every member: its gene id in a Run with Souls, its
        // id otherwise, also after a resume.
        let pool = sim_pool();

        for soul in [None, Some(SoulConfig::arm_s())] {
            let cfg = BreedConfig {
                soul: soul.clone(),
                ..small_config(179)
            };
            let b = bred(Breeder::new(cfg.clone(), &pool, None).expect("breeder"));
            let expected: Vec<String> = b
                .archive()
                .iter()
                .map(|m| match soul {
                    Some(_) => m.cascade.gene_id(),
                    None => m.id.clone(),
                })
                .collect();
            let resumed = Breeder::resume(cfg, &pool, None, b.snapshot()).expect("resume");

            assert_eq!(b.gene_ids, expected);
            assert_eq!(resumed.gene_ids, expected);
            assert_eq!(
                soul.is_some(),
                b.archive().iter().all(|m| m.id != m.cascade.gene_id())
            );
        }
    }

    /// What a run with these settings decides and calls: every request it sends, sorted,
    /// and per member its id, parents, operator, answers, solved count, graded value and
    /// usage, as one digest.
    fn decisions<D: Domain>(
        cfg: BreedConfig,
        genes: GenePool,
        small: Box<dyn ModelBackend>,
        large: Box<dyn ModelBackend>,
    ) -> String {
        let (sent, archive) = requests::<D>(cfg, genes, small, large);
        let mut out = sent.join("\n");

        for m in &archive {
            out.push_str(&format!(
                "\n{} {:?} {:?} {} {} {:?} {:?}",
                m.id,
                m.parents,
                m.op,
                serde_json::to_string(&m.answers).expect("answers json"),
                m.solved,
                m.quality,
                m.usage
            ));
        }

        sha256_hex(out.as_bytes())
    }

    /// The digests of `decisions`, recorded before the complex Stage vote (ticket 04): a
    /// grid run and a circuit run, each without Souls and with the Souls of arm S and
    /// `vote: false`.
    const VOTES_GRID: &str = "615c03e6c95c10c2d3fc741b0c7143425fc4f189882afd34d66e52ed9fd47048";
    const VOTES_GRID_OFF: &str = "cf7f18fddcb7373b02bc2da02b5af2a4d2735a842025cdb7a708b68f07a22231";
    const VOTES_CIRCUITS: &str = "564d8c123739e069cc43545c81c46c0cea45049ab91a06d6e8d90fcfa1cec400";
    const VOTES_CIRCUITS_OFF: &str =
        "598b6afd918233dcf0f9dfd3ebd670abf0027be5c8da6d83ef33bc37d50ea67e";

    #[test]
    fn without_souls_the_vote_is_todays() {
        let off = Some(SoulConfig {
            vote: false,
            ..SoulConfig::arm_s()
        });
        let grid = |soul: Option<SoulConfig>| {
            decisions::<GridDomain>(
                BreedConfig {
                    soul,
                    ..small_config(191)
                },
                GenePool::grid(),
                Box::new(SimModel::small()),
                Box::new(SimModel::large()),
            )
        };
        let circuits = |soul: Option<SoulConfig>| {
            decisions::<CircuitDomain>(
                BreedConfig {
                    soul,
                    ..circuit_config(193)
                },
                circuit_pool(),
                Box::new(SimDesigner::small()),
                Box::new(SimDesigner::large()),
            )
        };
        let digests = [
            grid(None),
            grid(off.clone()),
            circuits(None),
            circuits(off.clone()),
        ];

        assert_eq!(
            digests,
            [
                VOTES_GRID,
                VOTES_GRID_OFF,
                VOTES_CIRCUITS,
                VOTES_CIRCUITS_OFF
            ]
        );

        // Those short runs hold no Stage vote, so longer ones that do are compared with the
        // same runs without Souls, which the goldens of `tests/universe.rs` and the pilot
        // hash hold: with `vote: false` every vote picks the same answer by the same calls.
        fn compare<D: Domain>(
            cfg: BreedConfig,
            genes: GenePool,
            pool: fn() -> [Box<dyn ModelBackend>; 2],
            off: &Option<SoulConfig>,
        ) {
            let run = |soul: Option<SoulConfig>| {
                let [small, large] = pool();

                requests::<D>(
                    BreedConfig {
                        soul,
                        ..cfg.clone()
                    },
                    genes,
                    small,
                    large,
                )
            };
            let (plain, plain_archive) = run(None);
            let (counted, counted_archive) = run(off.clone());
            let (_, voted_archive) = run(Some(SoulConfig::arm_s()));
            let complex: u64 = voted_archive
                .iter()
                .filter_map(|m| m.votes)
                .map(|v| v.complex)
                .sum();

            assert!(complex > 0, "the run holds complex votes");
            assert_eq!(counted, plain, "the same calls");
            assert_eq!(counted_archive.len(), plain_archive.len());

            for (m, p) in counted_archive.iter().zip(&plain_archive) {
                assert_eq!(m.cascade.gene_id(), p.id);
                assert_eq!(m.answers, p.answers, "{}", p.id);
                assert_eq!(m.votes.map(|v| v.complex), Some(0));
            }
        }

        compare::<GridDomain>(
            BreedConfig {
                generations: 8,
                children_per_generation: 6,
                ..small_config(191)
            },
            GenePool::grid(),
            || [Box::new(SimModel::small()), Box::new(SimModel::large())],
            &off,
        );
        compare::<CircuitDomain>(
            BreedConfig {
                generations: 20,
                children_per_generation: 6,
                max_calls: 200_000,
                ..circuit_config(41)
            },
            circuit_pool(),
            || {
                [
                    Box::new(SimDesigner::small()),
                    Box::new(SimDesigner::large()),
                ]
            },
            &off,
        );
    }

    #[test]
    fn strengths_never_reach_a_request() {
        let soul = |strengths: Strengths, vote: bool| {
            Some(SoulConfig {
                vote,
                strengths,
                ..SoulConfig::arm_s()
            })
        };
        let grid = |soul: Option<SoulConfig>| {
            requests::<GridDomain>(
                BreedConfig {
                    soul,
                    ..small_config(197)
                },
                GenePool::grid(),
                Box::new(SimModel::small()),
                Box::new(SimModel::large()),
            )
        };
        let circuits = |soul: Option<SoulConfig>| {
            requests::<CircuitDomain>(
                BreedConfig {
                    soul,
                    ..circuit_config(199)
                },
                circuit_pool(),
                Box::new(SimDesigner::small()),
                Box::new(SimDesigner::large()),
            )
        };
        fn answers<A: Serialize>(archive: &[Member<A>]) -> Vec<String> {
            archive
                .iter()
                .map(|m| {
                    format!(
                        "{} {}",
                        m.id,
                        serde_json::to_string(&m.answers).expect("json")
                    )
                })
                .collect()
        }

        // Without the complex vote, the strengths of the Domain and equal strengths give the
        // same requests, byte for byte, and the same answers.
        let (grid_domain, grid_archive) = grid(soul(Strengths::Domain, false));
        let (grid_equal, grid_equal_archive) = grid(soul(Strengths::Equal, false));
        let (circuit_domain, circuit_archive) = circuits(soul(Strengths::Domain, false));
        let (circuit_equal, circuit_equal_archive) = circuits(soul(Strengths::Equal, false));

        assert_eq!(grid_domain, grid_equal);
        assert_eq!(answers(&grid_archive), answers(&grid_equal_archive));
        assert_eq!(circuit_domain, circuit_equal);
        assert_eq!(answers(&circuit_archive), answers(&circuit_equal_archive));

        // The vote is the path that carries them: with it the tallies differ.
        let tallies = |archive: Vec<Member>| -> Vec<Option<VoteTally>> {
            archive.into_iter().map(|m| m.votes).collect()
        };
        let by_domain = tallies(grid(soul(Strengths::Domain, true)).1);
        let by_equal = tallies(grid(soul(Strengths::Equal, true)).1);

        assert!(by_domain.iter().flatten().any(|t| t.complex > 0));
        assert_ne!(by_domain, by_equal);
    }

    /// Answers `Doubling` prompts by words of the system prompt, the same on every task:
    /// "say K" answers K and "silent" nothing usable; as a judge, "high" picks the larger
    /// number and any other judge the smaller.
    struct Fixed;

    impl ModelBackend for Fixed {
        fn name(&self) -> &str {
            "fixed"
        }

        fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
            let text = if req.prompt.starts_with("WHICH") {
                let numbers: Vec<u64> = req
                    .prompt
                    .split_whitespace()
                    .filter_map(|t| t.parse().ok())
                    .collect();
                let (a, b) = (numbers[1], numbers[2]);
                let first = if req.system.contains("high") {
                    a >= b
                } else {
                    a <= b
                };

                if first { "VERDICT A" } else { "VERDICT B" }.to_string()
            } else if req.system.contains("silent") {
                "I do not know.".to_string()
            } else {
                let said = req
                    .system
                    .split("say ")
                    .nth(1)
                    .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
                    .unwrap_or("0");

                format!("ANSWER {said}")
            };

            Ok(GenResponse {
                text,
                prompt_tokens: 10,
                completion_tokens: 5,
                stop_reason: StopReason::Stop,
                backend: "fixed".to_string(),
                soul_decided: false,
            })
        }
    }

    /// A Worker that says `answer`, or nothing, and judges `high` or `low`.
    fn fixed_worker(answer: Option<u64>, judge: &str) -> Worker {
        let section = |name: &str, text: String| Section {
            name: name.to_string(),
            text,
        };
        let say = match answer {
            Some(k) => format!("say {k}"),
            None => "stay silent".to_string(),
        };

        Worker {
            model: "sim:small".to_string(),
            sections: vec![
                section("role", "You answer with a number.".to_string()),
                section(
                    "strategy",
                    format!("Always {say}; as a judge prefer the {judge} number."),
                ),
                section(
                    "format",
                    "Answer with the line ANSWER and the number.".to_string(),
                ),
            ],
            traits: Traits::default(),
        }
    }

    /// Three Stages with fixed answers and verdicts: one answer of two Workers; a vote both
    /// judges give to 20; a vote that the count gives to 30.
    fn three_votes() -> Cascade {
        let stage = |workers: Vec<Worker>| Stage { workers };

        Cascade {
            stages: vec![
                stage(vec![
                    fixed_worker(Some(50), "high"),
                    fixed_worker(Some(50), "low"),
                ]),
                stage(vec![
                    fixed_worker(Some(10), "high"),
                    fixed_worker(Some(20), "high"),
                ]),
                stage(vec![
                    fixed_worker(Some(10), "high"),
                    fixed_worker(Some(20), "low"),
                    fixed_worker(Some(30), "high"),
                ]),
            ],
        }
    }

    /// A Stage of one answer of two Workers, then a Stage that gives nothing usable.
    fn silent_end() -> Cascade {
        Cascade {
            stages: vec![
                Stage {
                    workers: vec![
                        fixed_worker(Some(50), "high"),
                        fixed_worker(Some(50), "low"),
                    ],
                },
                Stage {
                    workers: vec![fixed_worker(None, "high")],
                },
            ],
        }
    }

    /// `Doubling` with two hypothesis families, of which the first fits every task.
    struct TwoFamilies;

    impl Domain for TwoFamilies {
        type Task = Number;
        type Answer = u64;

        fn from_config(_cfg: &BreedConfig) -> Result<TwoFamilies> {
            Ok(TwoFamilies)
        }

        fn panel(&self, seed: u64, n: usize) -> Vec<Number> {
            Doubling.panel(seed, n)
        }

        fn task_id<'t>(&self, task: &'t Number) -> &'t str {
            &task.id
        }

        fn solve_prompt(&self, task: &Number, previous: &[u64]) -> String {
            Doubling.solve_prompt(task, previous)
        }

        fn judge_prompt(&self, task: &Number, a: &u64, b: &u64) -> String {
            Doubling.judge_prompt(task, a, b)
        }

        fn parse_answer(&self, task: &Number, text: &str) -> Option<u64> {
            Doubling.parse_answer(task, text)
        }

        fn grade(&self, task: &Number, answer: &u64) -> Result<Grade> {
            Doubling.grade(task, answer)
        }

        fn prefer(&self, task: &Number, a: &u64, b: &u64) -> Option<bool> {
            Doubling.prefer(task, a, b)
        }

        fn graded(&self) -> bool {
            true
        }

        fn artefact(&self, answer: &u64) -> Option<String> {
            Doubling.artefact(answer)
        }

        fn self_check_text(&self) -> &'static str {
            Doubling.self_check_text()
        }

        fn families(&self) -> Vec<&'static str> {
            vec!["first", "second"]
        }

        fn strengths(&self, _task: &Number) -> Vec<f64> {
            vec![1.0, 0.0]
        }

        fn example(&self, task: &Number, answer: &u64) -> Option<String> {
            Some(format!("DOUBLE {} GIVES {answer}", task.n))
        }
    }

    #[test]
    fn authors_and_tallies_are_recorded() {
        let mut pool = ModelPool::new();

        pool.insert("sim:small", Box::new(Fixed));
        pool.insert("sim:large", Box::new(Fixed));

        let limits = CascadeLimits::default();
        let task = Doubling.panel(3, 1).remove(0);
        let c = three_votes();
        let strengths = [1.0, 0.0];
        // The phases on the fitting family: Stage 1 cancels, Stage 2 lies at 0, 90 and 180
        // degrees.
        let souls: Vec<PhaseState> = [0u16, 0, 0, 32768, 0, 16384, 32768]
            .map(|p| PhaseState(vec![p, 777]))
            .to_vec();
        let run = |vote: bool, readout: Readout| {
            let with = RunSouls {
                souls: &souls,
                strengths: &strengths,
                readout,
                vote,
                simulator: false,
                observe: false,
                upbringing: None,
            };

            run_cascade_with(&Doubling, &pool, &limits, &c, &task, 9, Some(&with))
        };

        for readout in [Readout::Complex, Readout::Real] {
            // Stage 1 ties at Π = 0 and goes to 20 by the count; in Stage 2 the count says 30,
            // the amplitudes 10, the answer of Worker 4 alone.
            let voted = run(true, readout);

            assert_eq!(voted.answer, Some(10), "{readout:?}");
            assert_eq!(voted.authors, [4]);
            assert_eq!(
                voted.votes,
                VoteTally {
                    complex: 2,
                    flipped: 1,
                    tied: 1,
                    coherence_sum: 1.0 / 3.0,
                    sim_decided: 0,
                }
            );
            // 2 answers in Stage 0, which agrees; 2 and 2 judgments; 3 and 9 judgments.
            assert_eq!(voted.usage.calls, 18);
            assert!(voted.usage.calls < cascade_call_bound(&c));

            // Without the complex vote the count decides, as without Souls.
            let counted = run(false, readout);
            let plain = run_cascade_in(&Doubling, &pool, &limits, &c, &task, 9);

            assert_eq!(counted, plain);
            assert_eq!(
                (counted.answer, counted.authors.as_slice()),
                (Some(30), &[6][..])
            );
            assert_eq!(counted.votes, VoteTally::default());
        }

        // The authors of an answer kept from an earlier Stage are that Stage's.
        let kept = run_cascade_in(&Doubling, &pool, &limits, &silent_end(), &task, 9);

        assert_eq!(
            (kept.answer, kept.authors.as_slice()),
            (Some(50), &[0, 1][..])
        );

        // In a Run, each member keeps the authors per search task and the sum of its
        // tallies, as its Born event does: what its own run gives on each task, with its
        // birth Souls on the first task and its imprinted Souls on the others (ticket 07).
        fn fixed_seeds() -> Vec<(String, Cascade)> {
            vec![
                ("votes".to_string(), three_votes()),
                ("silent".to_string(), silent_end()),
            ]
        }

        let genes = GenePool {
            seeds: fixed_seeds,
            ..doubling_genes()
        };
        let cfg = with_souls(small_config(211), SoulConfig::arm_s());
        let (_, events, snap) =
            breed_to_end_in::<TwoFamilies>(cfg.clone(), genes, &pool, None).expect("run");
        let panel = TwoFamilies.panel(derive_seed(cfg.seed, "panel/search"), cfg.search_puzzles);
        let mut complex = 0;

        for m in &snap.archive {
            let birth = birth_souls(&m.cascade);
            let imprinted = m.imprinted.clone().expect("imprinted Souls");
            let mut tally = VoteTally::default();
            let mut authors = Vec::new();

            for (i, t) in panel.iter().enumerate() {
                let with = RunSouls {
                    souls: if i == 0 { &birth } else { &imprinted },
                    strengths: &strengths,
                    readout: Readout::Complex,
                    vote: true,
                    simulator: true,
                    observe: true,
                    upbringing: None,
                };
                let seed = derive_seed(
                    cfg.seed,
                    &format!("search/{}/{}", m.cascade.gene_id(), t.id),
                );
                let r = run_cascade_with(
                    &TwoFamilies,
                    &pool,
                    &cfg.limits,
                    &m.cascade,
                    t,
                    seed,
                    Some(&with),
                );

                tally.add(&r.votes);
                authors.push(r.authors);
            }

            assert_eq!(m.authors.as_ref(), Some(&authors), "{}", m.id);
            assert_eq!(m.votes, Some(tally), "{}", m.id);
            complex += tally.complex;

            if m.cascade == silent_end().genes() {
                assert_eq!(authors, vec![vec![0, 1]; panel.len()]);
            }
        }

        assert!(complex > 0);

        for e in &events {
            if let Event::Born {
                id, authors, votes, ..
            } = e
            {
                let m = member(&snap.archive, id);

                assert_eq!((authors, votes), (&m.authors, &m.votes));
            }
        }
    }

    #[test]
    fn the_select_phase_votes_with_the_current_souls() {
        let mut pool = ModelPool::new();

        pool.insert("sim:small", Box::new(Fixed));
        pool.insert("sim:large", Box::new(Fixed));

        fn one_seed() -> Vec<(String, Cascade)> {
            vec![("votes".to_string(), three_votes())]
        }

        let genes = GenePool {
            seeds: one_seed,
            ..doubling_genes()
        };
        let cfg = with_souls(
            BreedConfig {
                generations: 0,
                ..small_config(223)
            },
            SoulConfig::arm_s(),
        );
        let never = || false;
        let snap = {
            let mut b: Breeder<'_, TwoFamilies> =
                Breeder::start(cfg.clone(), genes, &pool, None).expect("breeder");

            assert_eq!(b.step(&never).expect("gen 0").0, Status::Continue);

            b.snapshot()
        };
        let id = snap.archive[0].id.clone();

        // The select answers of the seed when its current Souls are `phases` on the fitting
        // family, whatever its birth Souls.
        let select = |phases: [u16; 7]| -> BTreeSet<Option<u64>> {
            let mut s = snap.clone();

            s.archive[0].soul_now = Some(phases.map(|p| PhaseState(vec![p, 777])).to_vec());

            let mut b: Breeder<'_, TwoFamilies> =
                Breeder::resume_from(cfg.clone(), genes, &pool, None, s).expect("resume");

            b.finish(&never, "all generations done")
                .expect("finish")
                .expect("not stopped");
            b.snapshot().select_answers[&id].values().cloned().collect()
        };

        // Stage 2 goes to 10 by the amplitudes, or to 30 when every phasor points alike.
        assert_eq!(
            select([0, 0, 0, 32768, 0, 16384, 32768]),
            BTreeSet::from([Some(10)])
        );
        assert_eq!(select([0; 7]), BTreeSet::from([Some(30)]));
    }

    /// Every event of a run stepped until its generations are done, and the breeder.
    fn bred_with_events<D: Domain>(mut b: Breeder<'_, D>) -> (Breeder<'_, D>, Vec<Event>) {
        let never = || false;
        let mut events = Vec::new();

        while let (Status::Continue, ev) = b.step(&never).expect("step") {
            events.extend(ev);
        }

        (b, events)
    }

    /// The `souls` events among `events`.
    fn souls_events(events: &[Event]) -> Vec<&Event> {
        events
            .iter()
            .filter(|e| matches!(e, Event::Souls { .. }))
            .collect()
    }

    /// The current Soul of every Worker of every member.
    fn current_souls<A>(archive: &[Member<A>]) -> Vec<Vec<Vec<u16>>> {
        archive
            .iter()
            .map(|m| {
                m.soul_now
                    .as_ref()
                    .expect("current Souls")
                    .iter()
                    .map(|p| p.0.clone())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn kappa_zero_changes_nothing() {
        // Arm K: every commit writes a `souls` event, none lists a Worker, and every current
        // Soul stays its birth Soul. Arm S on the same Runs moves Souls, so the Runs hold
        // Observations.
        let pool = sim_pool();
        let designers = designer_pool();

        for arm in [SoulConfig::arm_k(), SoulConfig::arm_s()] {
            let grid = bred_with_events(
                Breeder::new(with_souls(small_config(229), arm.clone()), &pool, None)
                    .expect("grid"),
            );
            let circuits = bred_with_events(
                Breeder::<CircuitDomain>::start(
                    with_souls(circuit_config(233), arm.clone()),
                    circuit_pool(),
                    &designers,
                    None,
                )
                .expect("circuits"),
            );
            let runs = [
                (
                    souls_events(&grid.1),
                    current_souls(grid.0.archive()),
                    grid.0
                        .archive()
                        .iter()
                        .map(|m| birth_phases(&m.cascade))
                        .collect(),
                ),
                (
                    souls_events(&circuits.1),
                    current_souls(circuits.0.archive()),
                    circuits
                        .0
                        .archive()
                        .iter()
                        .map(|m| birth_phases(&m.cascade))
                        .collect::<Vec<_>>(),
                ),
            ];

            for (souls, now, birth) in runs {
                let changed: usize = souls
                    .iter()
                    .map(|e| match e {
                        Event::Souls { changed, .. } => changed.len(),
                        _ => 0,
                    })
                    .sum();

                assert_eq!(souls.len(), 4, "one per committed generation");

                if arm.kappa_fixed {
                    assert_eq!(changed, 0, "arm K changes no Soul");
                    assert_eq!(now, birth);
                } else {
                    assert!(changed > 0, "arm S moves Souls");
                    assert_ne!(now, birth);
                }
            }
        }
    }

    /// Writes the children of the LLM round of generation 1: the i-th request gets the i-th
    /// Cascade, by the request's seed.
    struct Children {
        seed: u64,
        children: Vec<Cascade>,
    }

    impl Mutator for Children {
        fn answer(&self, req: &GenRequest) -> Option<GenResponse> {
            let i = (0..self.children.len())
                .find(|&i| req.seed == derive_seed(self.seed, &format!("llm/1/{i}")))?;

            Some(GenResponse {
                text: serde_json::to_string(&self.children[i]).ok()?,
                prompt_tokens: 10,
                completion_tokens: 10,
                stop_reason: StopReason::Stop,
                backend: "children".to_string(),
                soul_decided: false,
            })
        }
    }

    /// A member of generation 0 of a hand-built archive on one search task: `cascade` with
    /// these birth Souls and Coupling strengths, its answer, the authors of it, and its
    /// fitness. Its current Souls are its birth Souls.
    fn hand_member(
        mut cascade: Cascade,
        souls: &[(&[u16], u16)],
        answer: Option<u64>,
        authors: Vec<usize>,
        fitness: f64,
    ) -> Member<u64> {
        for (w, &(phases, kappa)) in cascade
            .stages
            .iter_mut()
            .flat_map(|s| &mut s.workers)
            .zip(souls)
        {
            w.traits.phase_state = Some(PhaseState(phases.to_vec()));
            w.traits.coupling = kappa;
        }

        let id = cascade.id();
        let sources = (0..cascade.worker_count())
            .map(|k| SoulSource {
                sources: Vec::new(),
                new: true,
                origin: WorkerRef {
                    member: id.clone(),
                    worker: k,
                },
            })
            .collect();

        Member {
            soul_sources: Some(sources),
            imprinted: Some(birth_souls(&cascade)),
            soul_now: Some(birth_souls(&cascade)),
            teaching: None,
            id,
            cascade,
            parents: Vec::new(),
            op: Op::Seed,
            generation: 0,
            answers: vec![answer],
            solved: 0,
            quality: Some(0.0),
            usage: Usage::default(),
            children: 0,
            judge_correct: 0,
            judge_wrong: 0,
            fitness,
            rating: 0.0,
            authors: Some(vec![authors]),
            votes: Some(VoteTally::default()),
            values: None,
            values_now: None,
            effect: None,
        }
    }

    /// A Cascade of one Stage whose Workers say these numbers and judge high.
    fn saying(numbers: &[u64]) -> Cascade {
        Cascade {
            stages: vec![Stage {
                workers: numbers
                    .iter()
                    .map(|&k| fixed_worker(Some(k), "high"))
                    .collect(),
            }],
        }
    }

    /// The Soul settings of the hand-built generation: arm S with copies without drift, no
    /// new Soul and κ copied, so the children's Workers carry their parent's Souls.
    fn exact_copies() -> SoulConfig {
        SoulConfig {
            phase_drift: 0,
            kappa_drift: 0,
            new_soul_theta: 0.0,
            ..SoulConfig::arm_s()
        }
    }

    /// A hand-built Run with the Soul settings `soul` after its generation 0, on one search
    /// task and one select task: its settings, its checkpoint and the Mutator of its LLM
    /// round. The archive: J judges and has no answer; P is the parent of both children; Q is
    /// their opponent. Fitness 1 is not eligible, so only P is drawn as a parent. The LLM
    /// round of generation 1 writes two children of P, each a Stage vote of two Workers.
    fn hand_start(soul: SoulConfig) -> (BreedConfig, Snapshot<u64>, Children) {
        let j = hand_member(saying(&[70]), &[(&[32768, 1000], 200)], None, vec![], 1.0);
        let p = hand_member(
            saying(&[10, 20]),
            &[(&[0, 0], 1000), (&[16384, 40000], 500)],
            Some(20),
            vec![1],
            0.5,
        );
        let q = hand_member(
            saying(&[60]),
            &[(&[8000, 60000], 0)],
            Some(60),
            vec![0],
            1.0,
        );
        let cfg = with_souls(
            BreedConfig {
                seed: 227,
                generations: 1,
                children_per_generation: 2,
                llm_every: 1,
                llm_children: 2,
                search_puzzles: 1,
                select_puzzles: 1,
                peer_opponents: 3,
                peer_puzzles_per_pair: 1,
                ..BreedConfig::default()
            },
            soul,
        );
        let children = Children {
            seed: cfg.seed,
            children: vec![saying(&[11, 21]), saying(&[30, 40])],
        };
        let snap = Snapshot {
            config_hash: cfg.fixed_hash(),
            next_generation: 1,
            archive: vec![j, p, q],
            comparisons: Vec::new(),
            calls: 0,
            mutator_answers: 0,
            reserved: 0,
            select_answers: BTreeMap::new(),
            finished: false,
        };

        (cfg, snap, children)
    }

    /// Generation 1 of the hand-built Run of `hand_start` with the Soul settings `soul`.
    /// Returns the events of the step and the archive: J, P, Q and the two children.
    fn hand_generation(soul: SoulConfig) -> (Vec<Event>, Vec<Member<u64>>) {
        let mut pool = ModelPool::new();

        pool.insert("sim:small", Box::new(Fixed));
        pool.insert("sim:large", Box::new(Fixed));

        let (cfg, snap, children) = hand_start(soul);
        let never = || false;
        let mut b: Breeder<'_, TwoFamilies> =
            Breeder::resume_from(cfg, doubling_genes(), &pool, Some(&children), snap)
                .expect("resume");
        let (status, events) = b.step(&never).expect("generation 1");

        assert_eq!(status, Status::Continue);

        (events, b.archive().to_vec())
    }

    #[test]
    fn observations_act_in_log_order() {
        let (events, archive) = hand_generation(exact_copies());
        let (jid, pid, qid) = (
            archive[0].id.clone(),
            archive[1].id.clone(),
            archive[2].id.clone(),
        );

        // The generation as built: two children of P, each answers with its Worker 1 after
        // a vote, and J judges each child against Q, A first.
        let (a, bb) = (&archive[3], &archive[4]);

        assert_eq!(
            (a.parents.clone(), a.op),
            (vec![pid.clone()], Op::LlmMutate)
        );
        assert_eq!(a.cascade.genes(), saying(&[11, 21]));
        assert_eq!(bb.cascade.genes(), saying(&[30, 40]));
        assert_eq!(birth_phases(&a.cascade), [vec![0, 0], vec![16384, 40000]]);
        assert_eq!(couplings(&bb.cascade), [1000, 500]);
        assert_eq!(
            (a.answers.clone(), a.authors.clone()),
            (vec![Some(21)], Some(vec![vec![1]]))
        );
        assert_eq!(
            (bb.answers.clone(), bb.authors.clone()),
            (vec![Some(40)], Some(vec![vec![1]]))
        );

        let judged: Vec<(String, String, String)> = events
            .iter()
            .filter_map(|e| match e {
                Event::Compared(c) => Some((c.first.clone(), c.second.clone(), c.judge.clone())),
                _ => None,
            })
            .collect();

        assert_eq!(
            judged,
            [
                (a.id.clone(), qid.clone(), jid.clone()),
                (bb.id.clone(), qid.clone(), jid.clone())
            ]
        );

        // By hand, in log order: A's vote (A0 observes A1, then A1 observes A0), B's vote
        // likewise, then the peer judgments: J observes A1 and then Q0, J observes B1 and
        // then Q0. Each step follows spec 8.3 from the Souls before it; for example A0 at
        // κ = 1000 moves A1 by the gap -16384 to 0 and itself by 8192.
        let expected: Vec<(String, usize, Vec<u16>)> = vec![
            (jid.clone(), 0, vec![23714, 64792]),
            (qid.clone(), 0, vec![14867, 62016]),
            (a.id.clone(), 0, vec![4096, 59152]),
            (a.id.clone(), 1, vec![8192, 63182]),
            (bb.id.clone(), 0, vec![4096, 59152]),
            (bb.id.clone(), 1, vec![7144, 62976]),
        ];
        let now = current_souls(&archive);

        assert_eq!(
            now,
            [
                vec![vec![23714, 64792]],
                vec![vec![0, 0], vec![16384, 40000]],
                vec![vec![14867, 62016]],
                vec![vec![4096, 59152], vec![8192, 63182]],
                vec![vec![4096, 59152], vec![7144, 62976]],
            ]
        );

        // The `souls` event follows the generation event and lists the changed Workers in
        // archive order; P did not change. The state stays the last event.
        let kinds: Vec<&str> = events
            .iter()
            .map(|e| match e {
                Event::Generation { .. } => "generation",
                Event::Souls { .. } => "souls",
                Event::State(_) => "state",
                _ => "other",
            })
            .collect();

        assert_eq!(kinds[kinds.len() - 3..], ["generation", "souls", "state"]);

        let Some(Event::Souls {
            generation,
            changed,
        }) = souls_events(&events).first().copied()
        else {
            panic!("no souls event");
        };
        let listed: Vec<(String, usize, Vec<u16>)> = changed
            .iter()
            .map(|c| (c.member.clone(), c.worker, c.phases.0.clone()))
            .collect();

        assert_eq!(*generation, 1);
        assert_eq!(listed, expected);

        // The state line counts the four Observations of each kind, and the mean shift of
        // the six Workers that moved, from their Souls before the commit (ticket 08).
        let Some(Event::State(state)) = events.last() else {
            panic!("no state event");
        };
        let soul = state.soul.as_ref().expect("a soul group");
        let before: BTreeMap<(String, usize), Vec<u16>> = [
            (jid.clone(), 0, vec![32768, 1000]),
            (qid.clone(), 0, vec![8000, 60000]),
            (a.id.clone(), 0, vec![0, 0]),
            (a.id.clone(), 1, vec![16384, 40000]),
            (bb.id.clone(), 0, vec![0, 0]),
            (bb.id.clone(), 1, vec![16384, 40000]),
        ]
        .into_iter()
        .map(|(id, k, p)| ((id, k), p))
        .collect();
        let degrees: f64 = expected
            .iter()
            .flat_map(|(id, k, after)| {
                let from = before[&(id.clone(), *k)].clone();

                from.into_iter().zip(after.clone()).map(|(p, q)| {
                    let d = u32::from(p.abs_diff(q));

                    f64::from(d.min(65536 - d)) * 360.0 / 65536.0
                })
            })
            .sum();

        assert_eq!((soul.observations_vote, soul.observations_peer), (4, 4));
        assert_eq!((soul.souls, soul.souls_born), (8, 4));
        assert!(
            (soul.shift_mean.expect("a shift") - degrees / 12.0).abs() < 1e-9,
            "{:?}",
            soul.shift_mean
        );
    }

    #[test]
    fn the_observation_switches_give_their_degenerate_case() {
        let now = |soul: SoulConfig| current_souls(&hand_generation(soul).1);

        // Only the peer judgments observe: J observes A1 and Q0, then B1 and Q0, while the
        // children keep their birth Souls from their votes. By hand, as in
        // `observations_act_in_log_order`.
        assert_eq!(
            now(SoulConfig {
                observe_vote: false,
                ..exact_copies()
            }),
            [
                vec![vec![26079, 61108]],
                vec![vec![0, 0], vec![16384, 40000]],
                vec![vec![15616, 60850]],
                vec![vec![0, 0], vec![19661, 45307]],
                vec![vec![0, 0], vec![18871, 44699]],
            ]
        );

        // Only the Stage votes observe: J and Q keep their Souls.
        assert_eq!(
            now(SoulConfig {
                observe_peer: false,
                ..exact_copies()
            }),
            [
                vec![vec![32768, 1000]],
                vec![vec![0, 0], vec![16384, 40000]],
                vec![vec![8000, 60000]],
                vec![vec![4096, 59152], vec![2048, 62344]],
                vec![vec![4096, 59152], vec![2048, 62344]],
            ]
        );

        // The state line counts each kind of Observation that its switch lets act: four
        // judgments of the votes and four of the peer round, by hand (ticket 08).
        let counted = |observe_vote: bool, observe_peer: bool| {
            let (events, _) = hand_generation(SoulConfig {
                observe_vote,
                observe_peer,
                ..exact_copies()
            });
            let soul = soul_groups(&events)[0].clone();

            (soul.observations_vote, soul.observations_peer)
        };

        assert_eq!(counted(true, false), (4, 0));
        assert_eq!(counted(false, true), (0, 4));
        assert_eq!(counted(false, false), (0, 0));

        // Neither: the `souls` event lists no Worker, and every Soul stays its birth Soul.
        let (events, archive) = hand_generation(SoulConfig {
            observe_vote: false,
            observe_peer: false,
            ..exact_copies()
        });
        let births: Vec<Vec<Vec<u16>>> = archive.iter().map(|m| birth_phases(&m.cascade)).collect();

        assert!(
            matches!(souls_events(&events)[..], [Event::Souls { changed, .. }] if changed.is_empty())
        );
        assert_eq!(current_souls(&archive), births);

        // With the simulator off no request carries phases, also to a backend that reads
        // them.
        let (steps, select, _) = phased_requests::<GridDomain>(
            with_souls(
                small_config(277),
                SoulConfig {
                    simulator: false,
                    ..SoulConfig::arm_s()
                },
            ),
            GenePool::grid(),
            Box::new(SimModel::small()),
            Box::new(SimModel::large()),
        );
        let sent: Vec<&GenRequest> = steps.iter().flat_map(|step| &step.requests).collect();

        assert!(!sent.is_empty() && sent.iter().all(|r| r.phases.is_none()));
        assert!(!select.is_empty() && select.iter().all(|r| r.phases.is_none()));
    }

    #[test]
    fn the_select_phase_changes_no_soul() {
        let pool = sim_pool();
        let never = || false;
        let cfg = |generations: usize| {
            with_souls(
                BreedConfig {
                    generations,
                    ..small_config(239)
                },
                SoulConfig::arm_s(),
            )
        };
        let (mut first, mut before) =
            bred_with_events(Breeder::new(cfg(2), &pool, None).expect("first"));
        let souls_before = current_souls(first.archive());

        first
            .finish(&never, "all generations done")
            .expect("finish")
            .expect("not stopped");

        // The select phase votes with the current Souls and changes none.
        assert_eq!(current_souls(first.archive()), souls_before);

        // Resumed to more generations after it, the Run logs what a straight Run logs.
        let resumed = Breeder::resume(cfg(4), &pool, None, first.snapshot()).expect("resume");
        let (_, after) = bred_with_events(resumed);
        let (_, straight) = bred_with_events(Breeder::new(cfg(4), &pool, None).expect("straight"));
        let logged = |events: &[Event]| -> Vec<Event> {
            events
                .iter()
                .filter(|e| matches!(e, Event::Souls { .. } | Event::State(_)))
                .cloned()
                .collect()
        };

        before.extend(after);

        assert_eq!(logged(&before), logged(&straight));
        assert!(
            souls_events(&straight)
                .iter()
                .any(|e| matches!(e, Event::Souls { changed, .. } if !changed.is_empty())),
            "the Run moves Souls"
        );
    }

    #[test]
    fn a_resume_after_the_select_phase_reports_like_a_straight_run() {
        // On this grid Run the Observations of generations 3 to 6 move the current Soul of a
        // member that the select phase after generation 2 scored, and its new Soul changes
        // its select answers. A member votes on the select panel with its current Souls
        // (spec 8.6), so the resumed Run scores it again: its finalists, their answers and
        // its report are those of the straight Run, calls aside.
        let pool = sim_pool();
        let never = || false;
        let cfg = |generations: usize| {
            with_souls(
                BreedConfig {
                    generations,
                    children_per_generation: 6,
                    ..small_config(30)
                },
                SoulConfig::arm_s(),
            )
        };
        let scored = |b: &mut Breeder<'_, GridDomain>| {
            let (report, _) = b
                .finish(&never, "all generations done")
                .expect("finish")
                .expect("not stopped");
            let kept = b.snapshot().select_answers;
            let answers: Vec<_> = report
                .finalists
                .iter()
                .map(|f| kept.get(&f.id).cloned())
                .collect();

            (
                Report {
                    calls_used: 0,
                    ..report
                },
                answers,
            )
        };
        let mut first = bred(Breeder::new(cfg(2), &pool, None).expect("first"));

        scored(&mut first);

        let resumed = Breeder::resume(cfg(6), &pool, None, first.snapshot()).expect("resume");
        let (report, answers) = scored(&mut bred(resumed));
        let (straight, straight_answers) = scored(&mut bred(
            Breeder::new(cfg(6), &pool, None).expect("straight"),
        ));

        assert_eq!(report.finalists, straight.finalists);
        assert!(answers == straight_answers, "the finalists' select answers");
        assert_eq!(report, straight);
    }

    #[test]
    fn a_moved_member_is_scored_again_with_its_current_souls() {
        // The hand-built Run finishes a select phase after generation 0, while every current
        // Soul is a birth Soul. Resumed, generation 1 moves J and Q but not P
        // (`observations_act_in_log_order`). So the next select phase asks the two children,
        // and J and Q again, with their new Souls; P keeps the answers its Souls still give.
        let (pool, log) = phase_log_pool();
        let (cfg, snap, children) = hand_start(exact_copies());
        let never = || false;
        let mut first: Breeder<'_, TwoFamilies> =
            Breeder::resume_from(cfg.clone(), doubling_genes(), &pool, Some(&children), snap)
                .expect("resume");

        first
            .finish(&never, "all generations done")
            .expect("finish")
            .expect("not stopped");

        let kept = first.snapshot();

        assert_eq!(
            kept.select_answers.keys().collect::<BTreeSet<_>>(),
            kept.archive.iter().map(|m| &m.id).collect::<BTreeSet<_>>(),
            "the first select phase scores J, P and Q"
        );

        let mut b: Breeder<'_, TwoFamilies> =
            Breeder::resume_from(cfg, doubling_genes(), &pool, Some(&children), kept)
                .expect("resume");

        assert_eq!(b.step(&never).expect("generation 1").0, Status::Continue);
        taken(&log);
        b.finish(&never, "all generations done")
            .expect("finish")
            .expect("not stopped");

        // With the complex readout, a request carries its Worker's current Soul as it is.
        let asked: BTreeSet<Vec<u16>> = taken(&log)
            .iter()
            .map(|r| r.phases.clone().expect("phases"))
            .collect();
        let now = |members: &[usize]| -> BTreeSet<Vec<u16>> {
            members
                .iter()
                .flat_map(|&i| plain(&b.archive()[i].soul_now))
                .collect()
        };

        assert_eq!(asked, now(&[0, 2, 3, 4]), "J, Q and the two children");
        assert!(asked.is_disjoint(&now(&[1])), "P is not asked again");
        assert_eq!(b.snapshot().select_answers.len(), 5);
    }

    #[test]
    fn stop_and_resume_keep_the_souls() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let pool = sim_pool();
        let cfg = with_souls(small_config(241), SoulConfig::arm_s());
        let (straight, straight_events) =
            bred_with_events(Breeder::new(cfg.clone(), &pool, None).expect("straight"));

        assert!(
            souls_events(&straight_events)
                .iter()
                .any(|e| matches!(e, Event::Souls { changed, .. } if !changed.is_empty())),
            "the Run moves Souls"
        );

        // Stop generation 1 while its newborns run, and generation 2 in its peer round: 4
        // newborns poll once per task (12 tasks), then each peer judgment polls.
        for (generation, polls) in [(1usize, 7usize), (2, 50)] {
            let mut b = Breeder::new(cfg.clone(), &pool, None).expect("split");
            let never = || false;
            let mut events = Vec::new();

            for _ in 0..generation {
                events.extend(b.step(&never).expect("step").1);
            }

            let counted = AtomicUsize::new(0);
            let stop = || counted.fetch_add(1, Ordering::SeqCst) >= polls;
            let (status, stopped) = b.step(&stop).expect("stopped step");

            assert_eq!(status, Status::Done("stopped".to_string()));
            assert!(stopped.is_empty(), "a stopped generation logs nothing");

            let json = serde_json::to_string(&b.snapshot()).expect("json");
            let snap: Snapshot = serde_json::from_str(&json).expect("back");
            let (resumed, rest) =
                bred_with_events(Breeder::resume(cfg.clone(), &pool, None, snap).expect("resume"));

            events.extend(rest);

            assert_eq!(
                souls_events(&events),
                souls_events(&straight_events),
                "stopped in generation {generation}"
            );
            assert_eq!(
                current_souls(resumed.archive()),
                current_souls(straight.archive())
            );
        }
    }

    /// A Cascade of one Stage with the first Workers of the first three seeds: it votes on
    /// most tasks.
    fn wide_seed(seeds: &[(String, Cascade)]) -> Cascade {
        Cascade {
            stages: vec![Stage {
                workers: seeds
                    .iter()
                    .take(3)
                    .map(|(_, c)| c.stages[0].workers[0].clone())
                    .collect(),
            }],
        }
    }

    /// `c` with a random Soul of `n` phases and a random Coupling strength on every Worker.
    fn with_random_souls(c: &Cascade, n: usize, rng: &mut ChaCha8Rng) -> Cascade {
        let mut souled = c.clone();

        for w in souled.stages.iter_mut().flat_map(|s| &mut s.workers) {
            w.traits.phase_state = Some(PhaseState((0..n).map(|_| rng.random()).collect()));
            w.traits.coupling = rng.random_range(0..=1000);
        }

        souled
    }

    /// Every request of `log`, emptied.
    fn drained(log: &std::sync::Mutex<Vec<String>>) -> Vec<String> {
        std::mem::take(&mut *log.lock().expect("log"))
    }

    #[test]
    fn requests_without_phases_are_unchanged() {
        // A backend that does not read phases gets, task by task, byte for byte the requests
        // the same Cascade without Souls sends, with the seed from its gene id.
        fn compare<D: Domain>(
            domain: &D,
            tasks: &[D::Task],
            c: &Cascade,
            n: usize,
            pool: &ModelPool,
            log: &std::sync::Mutex<Vec<String>>,
        ) -> usize {
            let limits = CascadeLimits::default();
            let mut rng = ChaCha8Rng::seed_from_u64(251);
            let mut sent = 0;

            for task in tasks {
                let souled = with_random_souls(c, n, &mut rng);
                let souls = birth_souls(&souled);
                let strengths = domain.strengths(task);
                let seed = derive_seed(
                    7,
                    &format!("search/{}/{}", souled.gene_id(), domain.task_id(task)),
                );
                let with = RunSouls {
                    souls: &souls,
                    strengths: &strengths,
                    readout: Readout::Complex,
                    vote: true,
                    simulator: true,
                    observe: true,
                    upbringing: None,
                };

                run_cascade_in(domain, pool, &limits, c, task, seed);

                let plain = drained(log);

                run_cascade_with(domain, pool, &limits, &souled, task, seed, Some(&with));

                let phased = drained(log);

                assert_eq!(phased, plain, "task {}", domain.task_id(task));
                assert!(phased.iter().all(|r| !r.contains("phases")));
                sent += phased.len();
            }

            sent
        }

        let (grid_pool, grid_log) = recording_pool(
            Box::new(SimModel::small()),
            Box::new(SimModel::large()),
            false,
        );
        let (circuit_pool_, circuit_log) = recording_pool(
            Box::new(SimDesigner::small()),
            Box::new(SimDesigner::large()),
            false,
        );
        let circuits = CircuitDomain::new(&CircuitConfig::default()).expect("domain");
        // The circuit seeds' Workers with the most thinking, and the third one also knowing
        // row operations, so that they disagree and vote: on CZ, Phase alone and Hadamard
        // with Linear build two different circuits.
        let mut thinking = wide_seed(&(circuit_pool().seeds)());

        for w in &mut thinking.stages[0].workers {
            w.traits.think_tokens = 4096;
        }

        thinking.stages[0].workers[2].sections[1]
            .text
            .push_str(" Treat each CNOT as a row operation.");

        let grid_sent = compare(
            &GridDomain,
            &generate_panel(23, 12),
            &wide_seed(&seed_cascades()),
            7,
            &grid_pool,
            &grid_log,
        );
        let circuit_sent = compare(
            &circuits,
            &circuits.panel(derive_seed(3, "panel/search"), 11),
            &thinking,
            8,
            &circuit_pool_,
            &circuit_log,
        );

        // Three Workers answer and judge: more than one call per task.
        assert!(
            grid_sent > 3 * 12 && circuit_sent > 3 * 11,
            "{grid_sent} {circuit_sent}"
        );

        // The JSON: no `phases` key without phases, no `soul_decided` key when it is false,
        // and a response written without it reads as false.
        let w = &seed_cascades()[0].1.stages[0].workers[0];
        let mut req = worker_request(w, "task".to_string(), 3);
        let plain = serde_json::to_string(&req).expect("json");

        assert!(!plain.contains("phases"), "{plain}");

        req.phases = Some(vec![1, 2]);

        assert!(
            serde_json::to_string(&req)
                .expect("json")
                .contains(r#""phases":[1,2]"#)
        );
        assert_eq!(
            serde_json::from_str::<GenRequest>(&plain)
                .expect("read")
                .phases,
            None
        );

        let resp = GenResponse {
            text: "ANSWER".to_string(),
            prompt_tokens: 1,
            completion_tokens: 1,
            stop_reason: StopReason::Stop,
            backend: "sim:small".to_string(),
            soul_decided: false,
        };
        let json = serde_json::to_string(&resp).expect("json");

        assert!(!json.contains("soul_decided"), "{json}");
        assert!(
            !serde_json::from_str::<GenResponse>(&json)
                .expect("read")
                .soul_decided
        );

        let decided = GenResponse {
            soul_decided: true,
            ..resp
        };

        assert!(
            serde_json::to_string(&decided)
                .expect("json")
                .ends_with(r#","soul_decided":true}"#)
        );
    }

    /// One committed generation of a Run whose backend reads phases: its requests, the
    /// Cascades born in it, the calls of their own evaluations, and all calls of the step.
    struct PhasedStep {
        requests: Vec<GenRequest>,
        born: Vec<Cascade>,
        own: u64,
        all: u64,
    }

    /// The requests of a Run with Souls to a backend that reads phases, one committed
    /// generation after another, then those of its select phase, and the ids of its search
    /// tasks.
    fn phased_requests<D: Domain>(
        cfg: BreedConfig,
        genes: GenePool,
        small: Box<dyn ModelBackend>,
        large: Box<dyn ModelBackend>,
    ) -> (Vec<PhasedStep>, Vec<GenRequest>, Vec<String>) {
        let (pool, log) = recording_pool(small, large, true);
        let mut b: Breeder<'_, D> = Breeder::start(cfg, genes, &pool, None).expect("breeder");
        let never = || false;
        let parse = |log: &std::sync::Mutex<Vec<String>>| -> Vec<GenRequest> {
            drained(log)
                .iter()
                .map(|r| serde_json::from_str(r).expect("request"))
                .collect()
        };
        let tasks: Vec<String> = b
            .search_panel()
            .iter()
            .map(|t| b.domain().task_id(t).to_string())
            .collect();
        let mut steps = Vec::new();

        loop {
            let before = b.calls_used();
            let (status, events) = b.step(&never).expect("step");

            if status != Status::Continue {
                break;
            }

            let mut own = 0;
            let mut born = Vec::new();

            for e in &events {
                if let Event::Born { usage, cascade, .. } = e {
                    own += usage.calls;
                    born.push(cascade.clone());
                }
            }

            steps.push(PhasedStep {
                requests: parse(&log),
                born,
                own,
                all: b.calls_used() - before,
            });
        }

        b.finish(&never, "all generations done")
            .expect("finish")
            .expect("not stopped");

        (steps, parse(&log), tasks)
    }

    /// Every seed a Stage vote of these newborns can give a judge call on these search
    /// tasks: from the task's seed of the Cascade's genes (spec 15), the Stage, the pair of
    /// answers and the judge, as `run_cascade_with` and `stage_vote` derive them.
    fn vote_seeds(seed: u64, tasks: &[String], born: &[Cascade]) -> HashSet<u64> {
        let mut seeds = HashSet::new();

        for c in born {
            for task in tasks {
                let run = derive_seed(seed, &format!("search/{}/{task}", c.gene_id()));

                for (k, stage) in c.stages.iter().enumerate() {
                    let vote = derive_seed(run, &format!("vote{k}"));
                    let width = stage.workers.len();

                    for x in 0..width {
                        for y in x + 1..width {
                            for j in 0..width {
                                seeds.insert(derive_seed(vote, &format!("{x}/{y}/{j}")));
                            }
                        }
                    }
                }
            }
        }

        seeds
    }

    #[test]
    fn peer_requests_carry_no_phases() {
        // Every request is told apart by its seed: a Stage vote's judge calls take seeds
        // derived from the newborn's genes, the task, the Stage, the pair and the judge; a
        // peer judgment's seed is drawn for the plan.
        fn check<D: Domain>(
            cfg: BreedConfig,
            genes: GenePool,
            models: [Box<dyn ModelBackend>; 2],
            judges: fn(&GenRequest) -> bool,
            n: usize,
        ) {
            let seed = cfg.seed;
            let [small, large] = models;
            let (steps, select, tasks) = phased_requests::<D>(cfg, genes, small, large);
            let (mut solves, mut votes, mut peers) = (0, 0, 0);

            for (g, step) in steps.iter().enumerate() {
                let in_votes = vote_seeds(seed, &tasks, &step.born);
                let mut phased = 0;

                for r in &step.requests {
                    let carries = r.phases.as_ref().is_some_and(|p| p.len() == n);

                    phased += u64::from(r.phases.is_some());

                    if !judges(r) {
                        assert!(carries, "generation {g}: a solve request without phases");
                        solves += 1;
                    } else if in_votes.contains(&r.seed) {
                        assert!(carries, "generation {g}: a Stage vote without phases");
                        votes += 1;
                    } else {
                        assert!(
                            r.phases.is_none(),
                            "generation {g}: a peer request with phases"
                        );
                        peers += 1;
                    }
                }

                // The newborns' own evaluations are exactly the requests with phases.
                assert_eq!(phased, step.own, "generation {g}");
                assert_eq!(step.requests.len() as u64, step.all, "generation {g}");
            }

            assert!(
                solves > 0 && votes > 0 && peers > 0,
                "{solves} {votes} {peers}"
            );
            assert!(!select.is_empty() && select.iter().all(|r| r.phases.is_some()));
        }

        // Runs that hold Stage votes (ticket 04): grid seed 191 over 8 generations, circuit
        // seed 41 over 20.
        check::<GridDomain>(
            with_souls(
                BreedConfig {
                    generations: 8,
                    children_per_generation: 6,
                    ..small_config(191)
                },
                SoulConfig::arm_s(),
            ),
            GenePool::grid(),
            [Box::new(SimModel::small()), Box::new(SimModel::large())],
            |r| parse_prompt(&r.prompt).is_some_and(|t| t.candidates.is_some()),
            7,
        );
        check::<CircuitDomain>(
            with_souls(
                BreedConfig {
                    generations: 20,
                    children_per_generation: 6,
                    max_calls: 200_000,
                    ..circuit_config(41)
                },
                SoulConfig::arm_s(),
            ),
            circuit_pool(),
            [
                Box::new(SimDesigner::small()),
                Box::new(SimDesigner::large()),
            ],
            |r| parse_circuit_prompt(&r.prompt).is_some_and(|t| t.candidates.is_some()),
            8,
        );
    }

    #[test]
    fn real_arm_phases_are_signs() {
        // In arm V every phase a request carries is 0 or half a turn; in arm S, on the same
        // Run, the phases are the Souls themselves.
        let phases_of = |soul: SoulConfig| -> Vec<u16> {
            let (steps, select, _) = phased_requests::<GridDomain>(
                with_souls(small_config(269), soul),
                GenePool::grid(),
                Box::new(SimModel::small()),
                Box::new(SimModel::large()),
            );

            steps
                .into_iter()
                .flat_map(|step| step.requests)
                .chain(select)
                .flat_map(|r| r.phases.unwrap_or_default())
                .collect()
        };
        let signs = phases_of(SoulConfig::arm_v());
        let angles = phases_of(SoulConfig::arm_s());

        assert!(!signs.is_empty());
        assert!(signs.iter().all(|&p| p == 0 || p == 32768), "{signs:?}");
        assert!(angles.iter().any(|&p| p != 0 && p != 32768));

        // Circuits too.
        let (steps, _, _) = phased_requests::<CircuitDomain>(
            with_souls(circuit_config(271), SoulConfig::arm_v()),
            circuit_pool(),
            Box::new(SimDesigner::small()),
            Box::new(SimDesigner::large()),
        );
        let circuit_signs: Vec<u16> = steps
            .into_iter()
            .flat_map(|step| step.requests)
            .flat_map(|r| r.phases.unwrap_or_default())
            .collect();

        assert!(!circuit_signs.is_empty());
        assert!(circuit_signs.iter().all(|&p| p == 0 || p == 32768));
    }

    /// `Fixed` that reads phases and keeps every request it gets.
    struct PhaseLog {
        log: std::sync::Arc<std::sync::Mutex<Vec<GenRequest>>>,
    }

    impl ModelBackend for PhaseLog {
        fn name(&self) -> &str {
            "phase-log"
        }

        fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
            self.log.lock().expect("log").push(req.clone());
            Fixed.generate(req)
        }

        fn reads_phases(&self) -> bool {
            true
        }
    }

    type RequestLog = std::sync::Arc<std::sync::Mutex<Vec<GenRequest>>>;

    /// A pool of `PhaseLog` under both tags, and its log.
    fn phase_log_pool() -> (ModelPool, RequestLog) {
        let log = RequestLog::default();
        let mut pool = ModelPool::new();

        for tag in ["sim:small", "sim:large"] {
            pool.insert(tag, Box::new(PhaseLog { log: log.clone() }));
        }

        (pool, log)
    }

    /// Every request of `log`, emptied.
    fn taken(log: &RequestLog) -> Vec<GenRequest> {
        std::mem::take(&mut *log.lock().expect("log"))
    }

    /// The current Souls of P in `upbringing_generation`: not its birth Souls.
    const P_NOW: [[u16; 2]; 2] = [[12000, 50000], [30000, 1000]];

    /// The birth Souls of P in `upbringing_generation`, which its children copy.
    const P_BIRTH: [[u16; 2]; 2] = [[0, 0], [16384, 40000]];

    /// Generation 1 of a hand-built Run as `hand_generation` builds it, on `tasks` search
    /// tasks and with the models of `pool`: P's Workers have the Coupling strength `kappa`
    /// and the current Souls `P_NOW`, P solves the second task, and J carries the teaching
    /// section `j_teaching`. With `finish`, the select phase follows. Returns the events,
    /// the archive (J, P, Q and the two children of P) and the settings.
    fn upbringing_generation(
        soul: SoulConfig,
        kappa: u16,
        tasks: usize,
        j_teaching: Option<&str>,
        finish: bool,
        pool: &ModelPool,
    ) -> (Vec<Event>, Vec<Member<u64>>, BreedConfig) {
        let cfg = with_souls(
            BreedConfig {
                seed: 227,
                generations: 1,
                children_per_generation: 2,
                llm_every: 1,
                llm_children: 2,
                search_puzzles: tasks,
                select_puzzles: 1,
                peer_opponents: 3,
                peer_puzzles_per_pair: 1,
                ..BreedConfig::default()
            },
            soul,
        );
        let panel = TwoFamilies.panel(derive_seed(cfg.seed, "panel/search"), tasks);
        let on_every = |mut m: Member<u64>| {
            m.answers = vec![m.answers[0]; tasks];
            m.authors = m.authors.map(|a| vec![a[0].clone(); tasks]);
            m
        };
        let [b0, b1] = P_BIRTH;
        let mut j = on_every(hand_member(
            saying(&[70]),
            &[(&[32768, 1000], 200)],
            None,
            vec![],
            1.0,
        ));
        let mut p = on_every(hand_member(
            saying(&[10, 20]),
            &[(&b0, kappa), (&b1, kappa)],
            Some(20),
            vec![1],
            0.5,
        ));
        let q = on_every(hand_member(
            saying(&[60]),
            &[(&[8000, 60000], 0)],
            Some(60),
            vec![0],
            1.0,
        ));

        j.teaching = j_teaching.map(str::to_string);
        p.soul_now = Some(P_NOW.iter().map(|s| PhaseState(s.to_vec())).collect());

        if let Some(task) = panel.get(1) {
            p.answers[1] = Some(2 * task.n);
        }

        let children = Children {
            seed: cfg.seed,
            children: vec![saying(&[11, 21]), saying(&[30, 40])],
        };
        let snap = Snapshot {
            config_hash: cfg.fixed_hash(),
            next_generation: 1,
            archive: vec![j, p, q],
            comparisons: Vec::new(),
            calls: 0,
            mutator_answers: 0,
            reserved: 0,
            select_answers: BTreeMap::new(),
            finished: false,
        };
        let never = || false;
        let mut b: Breeder<'_, TwoFamilies> =
            Breeder::resume_from(cfg.clone(), doubling_genes(), pool, Some(&children), snap)
                .expect("resume");
        let (status, mut events) = b.step(&never).expect("generation 1");

        assert_eq!(status, Status::Continue);
        assert_eq!(b.archive().len(), 5, "two children of P");

        if finish {
            let (_, finalists) = b
                .finish(&never, "all generations done")
                .expect("finish")
                .expect("not stopped");

            events.extend(finalists);
        }

        (events, b.archive().to_vec(), cfg)
    }

    /// The Souls of every Worker of `souls`, as plain phases.
    fn plain(souls: &Option<Vec<PhaseState>>) -> Vec<Vec<u16>> {
        souls
            .as_ref()
            .expect("Souls")
            .iter()
            .map(|p| p.0.clone())
            .collect()
    }

    #[test]
    fn imprinting_off_keeps_the_birth_soul() {
        let (pool, log) = phase_log_pool();
        let births = P_BIRTH.map(|s| s.to_vec());

        // The hand-built generation: with imprinting off, or at κ = 0, each child keeps its
        // birth Souls, and its calls on every task carry them. With imprinting on and κ
        // above 0, the children take on P's current Souls.
        for (soul, kappa, kept) in [
            (
                SoulConfig {
                    imprint: false,
                    ..exact_copies()
                },
                1000,
                true,
            ),
            (exact_copies(), 0, true),
            (exact_copies(), 1000, false),
        ] {
            let (_, archive, _) = upbringing_generation(soul, kappa, 3, None, false, &pool);
            let sent = taken(&log);

            for child in &archive[3..] {
                assert_eq!(birth_phases(&child.cascade), births, "copies of P");
                assert_eq!(plain(&child.imprinted) == births, kept, "κ {kappa}");
            }

            let phased: Vec<&Vec<u16>> = sent.iter().filter_map(|r| r.phases.as_ref()).collect();

            assert!(!phased.is_empty());
            assert_eq!(phased.iter().all(|p| births.contains(p)), kept, "κ {kappa}");
        }

        // Runs on both Domains: arm S with imprinting off and arm K, whose κ is 0, give
        // every member its birth Souls as its imprinted Souls; arm S imprints.
        let pool = sim_pool();
        let designers = designer_pool();
        let off = SoulConfig {
            imprint: false,
            ..SoulConfig::arm_s()
        };

        for (soul, kept) in [
            (off, true),
            (SoulConfig::arm_k(), true),
            (SoulConfig::arm_s(), false),
        ] {
            let grid = bred(
                Breeder::new(with_souls(small_config(283), soul.clone()), &pool, None)
                    .expect("grid"),
            );
            let circuits = bred(
                Breeder::<CircuitDomain>::start(
                    with_souls(circuit_config(293), soul.clone()),
                    circuit_pool(),
                    &designers,
                    None,
                )
                .expect("circuits"),
            );
            let same = |imprinted: &Option<Vec<PhaseState>>, c: &Cascade| {
                imprinted.as_ref() == Some(&birth_souls(c))
            };
            let grid_kept: Vec<bool> = grid
                .archive()
                .iter()
                .map(|m| same(&m.imprinted, &m.cascade))
                .collect();
            let circuit_kept: Vec<bool> = circuits
                .archive()
                .iter()
                .map(|m| same(&m.imprinted, &m.cascade))
                .collect();

            for all in [grid_kept, circuit_kept] {
                assert_eq!(all.iter().all(|&k| k), kept, "{soul:?}");
            }
        }
    }

    #[test]
    fn the_first_task_uses_the_birth_soul() {
        let (pool, log) = phase_log_pool();
        let tasks = 3;

        for kappa in [1000u16, 200] {
            let (events, archive, cfg) =
                upbringing_generation(exact_copies(), kappa, tasks, None, false, &pool);
            let sent = taken(&log);
            let panel = TwoFamilies.panel(derive_seed(cfg.seed, "panel/search"), tasks);
            let mut checked = 0;

            for child in &archive[3..] {
                let birth = birth_phases(&child.cascade);
                let imprinted = plain(&child.imprinted);

                // Each Worker is P's Worker at its place, copied, then moved toward P's
                // current Soul by κ, the observed side of an Observation.
                for k in 0..2 {
                    assert_eq!(birth[k], P_BIRTH[k]);
                    assert_eq!(imprinted[k], imprint(&birth[k], &P_NOW[k], kappa));
                }

                if kappa == 1000 {
                    assert_eq!(
                        imprinted,
                        P_NOW.map(|s| s.to_vec()),
                        "the source Workers' current phases"
                    );
                }

                // Task 0 with the birth Souls, every later task with the imprinted ones: the
                // solve call of each Worker and its judgment in the Stage vote.
                for (t, task) in panel.iter().enumerate() {
                    let run = derive_seed(
                        cfg.seed,
                        &format!("search/{}/{}", child.cascade.gene_id(), task.id),
                    );
                    let vote = derive_seed(run, "vote0");

                    for k in 0..2 {
                        let expected = if t == 0 { &birth[k] } else { &imprinted[k] };
                        let seeds = [
                            derive_seed(run, &format!("stage0/worker{k}")),
                            derive_seed(vote, &format!("0/1/{k}")),
                        ];

                        for seed in seeds {
                            let req = sent
                                .iter()
                                .find(|r| r.seed == seed)
                                .unwrap_or_else(|| panic!("task {t}, Worker {k}: no call"));

                            assert_eq!(req.phases.as_ref(), Some(expected), "task {t}, Worker {k}");
                            checked += 1;
                        }
                    }
                }
            }

            assert_eq!(checked, 2 * tasks * 2 * 2);

            // The Born events log the imprinted Souls the members keep.
            let mut logged = 0;

            for e in &events {
                if let Event::Born { id, imprinted, .. } = e {
                    assert_eq!(imprinted, &member(&archive, id).imprinted);
                    logged += 1;
                }
            }

            assert_eq!(logged, 2);
        }

        // In a Run, every newborn Worker's imprinted Soul is its birth Soul moved toward the
        // current Soul of each source Worker before the generation, in parent order, each by
        // that source's own κ, which drifts at birth, so a child's κ is not its source's.
        let pool = sim_pool();
        let never = || false;
        let mut b = Breeder::new(
            with_souls(longer_config(337), SoulConfig::arm_s()),
            &pool,
            None,
        )
        .expect("breeder");
        let (mut one, mut two, mut other_kappa) = (0, 0, 0);

        loop {
            let before: BTreeMap<String, Member> = b
                .archive()
                .iter()
                .map(|m| (m.id.clone(), m.clone()))
                .collect();
            let g = b.next_generation();

            if b.step(&never).expect("step").0 != Status::Continue {
                break;
            }

            for m in b.archive().iter().filter(|m| m.generation == g && g > 0) {
                let birth = birth_phases(&m.cascade);
                let kappas = couplings(&m.cascade);

                for (k, rec) in records(m).iter().enumerate() {
                    let sources: Vec<(&[u16], u16)> = rec
                        .sources
                        .iter()
                        .map(|r| {
                            let parent = &before[&r.member];

                            (
                                parent.soul_now.as_ref().expect("current")[r.worker]
                                    .0
                                    .as_slice(),
                                couplings(&parent.cascade)[r.worker],
                            )
                        })
                        .collect();

                    assert_eq!(
                        plain(&m.imprinted)[k],
                        imprinted(&birth[k], &sources, true),
                        "{} Worker {k}",
                        m.id
                    );

                    one += usize::from(sources.len() == 1);
                    two += usize::from(sources.len() == 2);
                    other_kappa += usize::from(sources.iter().all(|s| s.1 != kappas[k]));
                }
            }
        }

        assert!(
            one > 0 && two > 0 && other_kappa > 0,
            "{one} {two} {other_kappa}"
        );
    }

    #[test]
    fn imprinting_leaves_the_parents_alone() {
        let (pool, _log) = phase_log_pool();
        let quiet = |soul: SoulConfig| SoulConfig {
            observe_vote: false,
            observe_peer: false,
            ..soul
        };
        let (_, archive, _) =
            upbringing_generation(quiet(exact_copies()), 1000, 3, None, false, &pool);
        let now = current_souls(&archive);

        // P keeps the current Souls it had before its children were imprinted toward them,
        // and J and Q keep theirs; each child starts with its imprinted Souls.
        assert_eq!(now[1], P_NOW.map(|s| s.to_vec()));
        assert_eq!(
            birth_phases(&archive[1].cascade),
            P_BIRTH.map(|s| s.to_vec())
        );
        assert_eq!(now[0], [vec![32768, 1000]]);
        assert_eq!(now[2], [vec![8000, 60000]]);

        for child in &archive[3..] {
            assert_eq!(plain(&child.imprinted), P_NOW.map(|s| s.to_vec()));
            assert_eq!(child.soul_now, child.imprinted);
        }

        // A Run without Observations: every current Soul stays its imprinted Soul, a seed's
        // is its birth Soul, and children are imprinted toward their parents' Souls.
        let archive = grid_archive(with_souls(longer_config(281), quiet(SoulConfig::arm_s())));
        let mut moved = 0;

        for m in &archive {
            assert_eq!(m.soul_now, m.imprinted, "{}", m.id);

            if m.op == Op::Seed {
                assert_eq!(m.imprinted, Some(birth_souls(&m.cascade)), "{}", m.id);
            }

            moved += usize::from(m.imprinted != Some(birth_souls(&m.cascade)));
        }

        assert!(moved > 0, "children are imprinted");
    }

    /// The sections of a Run's members, checked: each is what its parents' answers give,
    /// none is in a genome, and the Born events log them. Returns how many members hold one,
    /// and how many of those are children of a child.
    fn checked_sections<D: Domain>(b: &Breeder<'_, D>, events: &[Event]) -> (usize, usize) {
        let archive = b.archive();
        let (mut taught, mut grand) = (0, 0);

        for m in archive {
            let parents: Vec<&Member<D::Answer>> =
                m.parents.iter().map(|p| member(archive, p)).collect();
            let answers: Vec<&[Option<D::Answer>]> =
                parents.iter().map(|p| p.answers.as_slice()).collect();
            let genome = serde_json::to_string(&m.cascade).expect("genome json");

            assert_eq!(
                m.teaching,
                teaching(&b.domain, &b.search, &answers).expect("teach"),
                "{}",
                m.id
            );
            assert!(!genome.contains(UPBRINGING), "{}", m.id);

            if let Some(text) = &m.teaching {
                assert!(!genome.contains(text.as_str()), "{}", m.id);
                taught += 1;
                grand += usize::from(parents.iter().any(|p| p.op != Op::Seed));
            }
        }

        let mut born = 0;

        for e in events {
            if let Event::Born { id, teaching, .. } = e {
                assert_eq!(teaching, &member(archive, id).teaching);
                born += 1;
            }
        }

        assert_eq!(born, archive.len());

        (taught, grand)
    }

    /// Arm S with teaching on.
    fn teaching_arm() -> SoulConfig {
        SoulConfig {
            teach: true,
            ..SoulConfig::arm_s()
        }
    }

    #[test]
    fn teaching_is_never_a_gene() {
        let pool = sim_pool();
        let designers = designer_pool();
        let (grid, grid_events) = bred_with_events(
            Breeder::new(with_souls(longer_config(307), teaching_arm()), &pool, None)
                .expect("grid"),
        );
        let (circuits, circuit_events) = bred_with_events(
            Breeder::<CircuitDomain>::start(
                with_souls(circuit_config(311), teaching_arm()),
                circuit_pool(),
                &designers,
                None,
            )
            .expect("circuits"),
        );
        let (grid_taught, grid_grand) = checked_sections(&grid, &grid_events);
        let (circuit_taught, circuit_grand) = checked_sections(&circuits, &circuit_events);

        assert!(
            grid_taught > 0 && circuit_taught > 0,
            "{grid_taught} {circuit_taught}"
        );
        assert!(
            grid_grand > 0 && circuit_grand > 0,
            "{grid_grand} {circuit_grand}"
        );

        // A section is never inherited: with every archived section replaced, the next
        // generation's sections are again those its parents' answers give.
        let never = || false;
        let cfg = with_souls(small_config(313), teaching_arm());
        let mut b = Breeder::new(cfg.clone(), &pool, None).expect("breeder");

        for _ in 0..3 {
            assert_eq!(b.step(&never).expect("step").0, Status::Continue);
        }

        let mut snap = b.snapshot();
        let inherited = "Inherited text.";

        for m in &mut snap.archive {
            m.teaching = Some(inherited.to_string());
        }

        let mut resumed = Breeder::resume(cfg, &pool, None, snap).expect("resume");
        let (status, _) = resumed.step(&never).expect("generation 3");

        assert_eq!(status, Status::Continue);

        let archive = resumed.archive();
        let newborn: Vec<&Member> = archive.iter().filter(|m| m.generation == 3).collect();

        assert!(!newborn.is_empty());
        assert!(newborn.iter().any(|m| m.teaching.is_some()));

        for m in newborn {
            let answers: Vec<&[Option<Grid>]> = m
                .parents
                .iter()
                .map(|p| member(archive, p).answers.as_slice())
                .collect();

            assert_eq!(
                m.teaching,
                teaching(&resumed.domain, &resumed.search, &answers).expect("teach")
            );
            assert!(
                !m.teaching
                    .as_deref()
                    .unwrap_or_default()
                    .contains(inherited)
            );
        }
    }

    #[test]
    fn the_teaching_text_holds_no_stem() {
        // The sections the Runs wrote, and the section that every ordered pair of archived
        // members would give a child of theirs. Returns how many the Run wrote.
        fn collect<D: Domain>(b: &Breeder<'_, D>, out: &mut Vec<String>) -> usize {
            let archive = b.archive();
            let written: Vec<String> = archive.iter().filter_map(|m| m.teaching.clone()).collect();
            let n = written.len();

            out.extend(written);

            for x in archive {
                for y in archive {
                    let both = [x.answers.as_slice(), y.answers.as_slice()];

                    out.extend(teaching(&b.domain, &b.search, &both).expect("teach"));
                }
            }

            n
        }

        let pool = sim_pool();
        let designers = designer_pool();
        let (mut grid, mut circuits) = (Vec::new(), Vec::new());
        let (mut grid_written, mut circuits_written) = (0, 0);
        let mut seed = 317;

        while grid.len() < 1000 || circuits.len() < 1000 {
            let g = bred(
                Breeder::new(with_souls(small_config(seed), teaching_arm()), &pool, None)
                    .expect("grid"),
            );
            let c = bred(
                Breeder::<CircuitDomain>::start(
                    with_souls(circuit_config(seed), teaching_arm()),
                    circuit_pool(),
                    &designers,
                    None,
                )
                .expect("circuits"),
            );

            grid_written += collect(&g, &mut grid);
            circuits_written += collect(&c, &mut circuits);
            seed += 1;
        }

        assert!(grid_written > 0 && circuits_written > 0, "the Runs teach");

        for text in grid.iter().chain(&circuits) {
            assert!(check_section_text(text).is_ok(), "{text}");
            assert!(text.chars().count() <= MAX_SECTION_CHARS, "{text}");
            assert!(text.starts_with(TEACHING_INTRO), "{text}");
            assert_eq!(model_word_in(text), None, "{text}");
        }

        // Most sections hold an example of each parent; some had to leave one out.
        let two = |all: &[String]| all.iter().filter(|t| t.contains("EXAMPLE 2")).count();

        assert!(two(&grid) > 0 && two(&circuits) > 0);
    }

    #[test]
    fn teaching_changes_no_simulated_answer() {
        /// What a Run decides: per member its id, parents, operator, answers, solved count,
        /// graded value, authors and vote tally; every comparison; the finalists. And every
        /// request it sends.
        fn outcome<D: Domain>(
            cfg: BreedConfig,
            genes: GenePool,
            small: Box<dyn ModelBackend>,
            large: Box<dyn ModelBackend>,
        ) -> (Vec<String>, Vec<GenRequest>, usize) {
            let (pool, log) = recording_pool(small, large, true);
            let (report, _, snap) = breed_to_end_in::<D>(cfg, genes, &pool, None).expect("run");
            let mut out: Vec<String> = snap
                .archive
                .iter()
                .map(|m| {
                    format!(
                        "{} {:?} {:?} {} {} {:?} {:?} {:?}",
                        m.id,
                        m.parents,
                        m.op,
                        serde_json::to_string(&m.answers).expect("answers json"),
                        m.solved,
                        m.quality,
                        m.authors,
                        m.votes
                    )
                })
                .collect();
            let sent: Vec<GenRequest> = drained(&log)
                .iter()
                .map(|r| serde_json::from_str(r).expect("request"))
                .collect();

            out.extend(snap.comparisons.iter().map(|c| format!("{c:?}")));
            out.extend(report.finalists.iter().map(|f| format!("{f:?}")));

            let taught = snap.archive.iter().filter(|m| m.teaching.is_some()).count();

            (out, sent, taught)
        }

        let teach = |on: bool| SoulConfig {
            teach: on,
            ..SoulConfig::arm_s()
        };
        let runs = [
            [false, true].map(|on| {
                outcome::<GridDomain>(
                    with_souls(longer_config(331), teach(on)),
                    GenePool::grid(),
                    Box::new(SimModel::small()),
                    Box::new(SimModel::large()),
                )
            }),
            [false, true].map(|on| {
                outcome::<CircuitDomain>(
                    with_souls(circuit_config(337), teach(on)),
                    circuit_pool(),
                    Box::new(SimDesigner::small()),
                    Box::new(SimDesigner::large()),
                )
            }),
        ];

        for [(plain, plain_sent, none), (taught, sent, sections)] in runs {
            assert_eq!(taught, plain, "the same answers, counts and verdicts");
            assert_eq!(none, 0);
            assert!(sections > 0, "the Run teaches");
            assert_eq!(sent.len(), plain_sent.len(), "the same calls");
            assert!(
                plain_sent
                    .iter()
                    .all(|r| !r.system.contains("## upbringing"))
            );

            // The section ends each system prompt that carries it, after the self-check text
            // where a Worker checks itself.
            let carrying: Vec<&GenRequest> = sent
                .iter()
                .filter(|r| r.system.contains("## upbringing"))
                .collect();

            assert!(!carrying.is_empty());
            assert!(carrying.iter().any(|r| r.system.contains("## check\n")));

            for r in carrying {
                let last = r.system.rsplit("## ").next().unwrap_or_default();

                assert!(
                    last.starts_with("upbringing\n") && last.ends_with("\n\n"),
                    "{}",
                    r.system
                );
            }
        }
    }

    #[test]
    fn the_teaching_section_ends_every_system_prompt() {
        // In the hand-built generation P solved task 1, so its children are taught that
        // example, and J judges with a section of its own.
        let (pool, log) = phase_log_pool();
        let tasks = 3;
        let judge = "Notes of the judge.";
        let (events, archive, cfg) =
            upbringing_generation(teaching_arm(), 200, tasks, Some(judge), false, &pool);
        let sent = taken(&log);
        let panel = TwoFamilies.panel(derive_seed(cfg.seed, "panel/search"), tasks);
        let lesson = format!(
            "{TEACHING_INTRO}\nEXAMPLE 1\nDOUBLE {} GIVES {}",
            panel[1].n,
            2 * panel[1].n
        );
        let mut counted = 0;

        for child in &archive[3..] {
            assert_eq!(child.teaching.as_deref(), Some(lesson.as_str()));

            for w in child.cascade.stages.iter().flat_map(|s| &s.workers) {
                let own = w.system_prompt();
                let calls: Vec<&GenRequest> =
                    sent.iter().filter(|r| r.system.starts_with(&own)).collect();

                assert_eq!(
                    calls.len(),
                    2 * tasks,
                    "a solve call and a judgment per task"
                );

                for r in calls {
                    assert_eq!(r.system, format!("{own}{}", section_block(&lesson)));
                    counted += 1;
                }
            }
        }

        // The peer judge's requests end with its own section, and P and Q send none.
        let j = &archive[0].cascade.stages[0].workers[0];
        let judged: Vec<&GenRequest> = sent
            .iter()
            .filter(|r| r.system.starts_with(&j.system_prompt()))
            .collect();

        assert!(!judged.is_empty());

        for r in &judged {
            assert_eq!(
                r.system,
                format!("{}{}", j.system_prompt(), section_block(judge))
            );
        }

        assert_eq!(counted + judged.len(), sent.len());

        // The Born events log the sections, which are in no genome.
        for e in &events {
            if let Event::Born {
                id,
                teaching,
                cascade,
                ..
            } = e
            {
                assert_eq!(teaching.as_deref(), Some(lesson.as_str()), "{id}");
                assert!(
                    !serde_json::to_string(cascade)
                        .expect("json")
                        .contains("EXAMPLE")
                );
            }
        }

        // With the select phase: every request of every member is its Worker's system
        // prompt, then the member's section if it has one, and the finalists' select calls
        // carry theirs.
        let (events, archive, _) =
            upbringing_generation(teaching_arm(), 200, tasks, Some(judge), true, &pool);
        let sent = taken(&log);
        let mut taught = 0;

        assert!(events.iter().any(|e| matches!(e, Event::Finalist(_))));

        for r in &sent {
            let (m, w) = archive
                .iter()
                .flat_map(|m| {
                    m.cascade
                        .stages
                        .iter()
                        .flat_map(|s| &s.workers)
                        .map(move |w| (m, w))
                })
                .find(|(_, w)| r.system.starts_with(&w.system_prompt()))
                .expect("a member's Worker sent it");
            let section = m.teaching.as_deref().map(section_block);

            assert_eq!(
                r.system,
                format!("{}{}", w.system_prompt(), section.unwrap_or_default())
            );
            taught += usize::from(m.teaching.is_some() && m.op != Op::Seed);
        }

        // The children's calls: per Worker a solve call and a judgment on each search task
        // and on the select task.
        assert_eq!(taught, 2 * 2 * 2 * (tasks + 1));
    }

    /// The soul group of every state line among `events`, in order.
    fn soul_groups(events: &[Event]) -> Vec<&SoulState> {
        events
            .iter()
            .filter_map(|e| match e {
                Event::State(s) => s.soul.as_deref(),
                _ => None,
            })
            .collect()
    }

    /// The state lines among `events`, in order.
    fn state_lines(events: &[Event]) -> Vec<UniverseState> {
        events
            .iter()
            .filter_map(|e| match e {
                Event::State(s) => Some(s.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn no_soul_group_without_souls() {
        let pool = sim_pool();
        let designers = designer_pool();

        for soul in [None, Some(SoulConfig::arm_s())] {
            let (_, grid) = bred_with_events(
                Breeder::new(
                    BreedConfig {
                        soul: soul.clone(),
                        ..small_config(347)
                    },
                    &pool,
                    None,
                )
                .expect("grid"),
            );
            let (_, circuits) = bred_with_events(
                Breeder::<CircuitDomain>::start(
                    BreedConfig {
                        soul: soul.clone(),
                        ..circuit_config(349)
                    },
                    circuit_pool(),
                    &designers,
                    None,
                )
                .expect("circuits"),
            );

            for events in [grid, circuits] {
                let lines = state_lines(&events);

                assert_eq!(lines.len(), 4, "one per committed generation");

                for state in lines {
                    let v = serde_json::to_value(&state).expect("state json");

                    assert_eq!(v.get("soul").is_some(), soul.is_some(), "{v}");
                }
            }
        }
    }

    #[test]
    fn the_soul_population_never_falls() {
        let pool = sim_pool();
        let cfg = with_souls(
            BreedConfig {
                generations: 50,
                max_calls: 10_000_000,
                ..small_config(353)
            },
            SoulConfig::arm_s(),
        );
        let (b, events) = bred_with_events(Breeder::new(cfg, &pool, None).expect("breeder"));
        let groups = soul_groups(&events);
        let mut last = 0;

        assert_eq!(groups.len(), 51, "50 generations after the seeds");
        assert_eq!(
            groups[0].souls_new, groups[0].souls_born,
            "a seed's Soul is new"
        );

        for (g, s) in groups.iter().enumerate() {
            assert!(s.souls >= last, "generation {g}: {} after {last}", s.souls);
            assert_eq!(s.souls, last + s.souls_born, "generation {g}");
            last = s.souls;

            // The votes are the sum of the tallies the generation's Born events log.
            let mut tally = VoteTally::default();

            for e in &events {
                if let Event::Born {
                    generation,
                    votes: Some(v),
                    ..
                } = e
                    && *generation == g
                {
                    tally.add(v);
                }
            }

            assert_eq!(
                (s.votes, s.votes_flipped, s.votes_tied, s.sim_decided),
                (tally.complex, tally.flipped, tally.tied, tally.sim_decided),
                "generation {g}"
            );
        }

        assert!(groups.iter().any(|s| s.votes > 0), "the Run holds votes");

        let workers: usize = b.archive().iter().map(|m| m.cascade.worker_count()).sum();

        assert_eq!(last, workers);
    }

    #[test]
    fn fresh_souls_meet_the_formula() {
        // Without copies every Soul is entirely new, and at κ fixed 0 none is pulled: the
        // Souls are independent and uniform, as the formula of spec 6.3 assumes.
        let designers = designer_pool();

        for readout in [Readout::Complex, Readout::Real] {
            let soul = SoulConfig {
                readout,
                copy: false,
                kappa_seed: 0,
                kappa_fixed: true,
                ..SoulConfig::arm_s()
            };
            let cfg = with_souls(
                BreedConfig {
                    generations: 50,
                    max_calls: 10_000_000,
                    ..circuit_config(359)
                },
                soul,
            );
            let (_, events) = bred_with_events(
                Breeder::<CircuitDomain>::start(cfg, circuit_pool(), &designers, None)
                    .expect("circuits"),
            );
            let lines = state_lines(&events);
            let late: Vec<&SoulState> = lines
                .iter()
                .filter(|s| (10..=50).contains(&s.generation))
                .map(|s| s.soul.as_deref().expect("a soul group"))
                .collect();
            let formula = late[0].contrast_independent.expect("the formula");
            let mean = late
                .iter()
                .map(|s| s.contrast.expect("a contrast"))
                .sum::<f64>()
                / late.len() as f64;

            assert_eq!(late.len(), 41);
            assert!(late.iter().all(|s| s.contrast_independent == Some(formula)));
            assert!(
                late.iter()
                    .all(|s| s.moved_mean == Some(0.0) && s.shift_mean.is_none()),
                "no Soul moved"
            );
            assert!(
                (mean - formula).abs() < 0.1,
                "{readout:?}: contrast {mean}, formula {formula}"
            );
        }
    }

    #[test]
    fn soul_states_survive_a_resume() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let pool = sim_pool();
        let cfg = with_souls(small_config(241), SoulConfig::arm_s());
        let (_, straight) =
            bred_with_events(Breeder::new(cfg.clone(), &pool, None).expect("straight"));

        assert_eq!(soul_groups(&straight).len(), 4);
        assert!(
            soul_groups(&straight)
                .iter()
                .all(|s| s.souls > 0 && s.contrast.is_some() && s.kappa.is_some())
        );

        // Stopped in generation 1 while its newborns run, and in generation 2 in its peer
        // round, as in `stop_and_resume_keep_the_souls`; resumed from the snapshot's JSON.
        for (generation, polls) in [(1usize, 7usize), (2, 50)] {
            let mut b = Breeder::new(cfg.clone(), &pool, None).expect("split");
            let never = || false;
            let mut events = Vec::new();

            for _ in 0..generation {
                events.extend(b.step(&never).expect("step").1);
            }

            let counted = AtomicUsize::new(0);
            let stop = || counted.fetch_add(1, Ordering::SeqCst) >= polls;
            let (status, stopped) = b.step(&stop).expect("stopped step");

            assert_eq!(status, Status::Done("stopped".to_string()));
            assert!(stopped.is_empty());

            let json = serde_json::to_string(&b.snapshot()).expect("json");
            let snap: Snapshot = serde_json::from_str(&json).expect("back");
            let (_, rest) =
                bred_with_events(Breeder::resume(cfg.clone(), &pool, None, snap).expect("resume"));

            events.extend(rest);

            assert_eq!(
                state_lines(&events),
                state_lines(&straight),
                "stopped in generation {generation}"
            );
        }

        // One and four threads: the same bytes.
        let at = |threads: usize| {
            let workers = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("a thread pool");

            workers.install(|| {
                let (_, events) =
                    bred_with_events(Breeder::new(cfg.clone(), &pool, None).expect("run"));

                serde_json::to_string(&events).expect("json")
            })
        };
        let one = at(1);

        assert!(one.contains(r#""soul":{"souls":"#));
        assert_eq!(one, at(4));
    }

    /// Every number in `v`.
    fn json_numbers(v: &serde_json::Value, out: &mut Vec<f64>) {
        match v {
            serde_json::Value::Number(n) => out.extend(n.as_f64()),
            serde_json::Value::Array(items) => items.iter().for_each(|x| json_numbers(x, out)),
            serde_json::Value::Object(fields) => fields.values().for_each(|x| json_numbers(x, out)),
            _ => {}
        }
    }

    #[test]
    fn no_value_is_nan() {
        let pool = sim_pool();
        let designers = designer_pool();
        let arms = [
            SoulConfig::arm_s(),
            SoulConfig::arm_v(),
            SoulConfig::arm_k(),
            SoulConfig {
                strengths: Strengths::Equal,
                ..SoulConfig::arm_s()
            },
        ];
        let mut checked = 0;

        for arm in arms {
            let (_, grid) = bred_with_events(
                Breeder::new(with_souls(small_config(373), arm.clone()), &pool, None)
                    .expect("grid"),
            );
            let (_, circuits) = bred_with_events(
                Breeder::<CircuitDomain>::start(
                    with_souls(circuit_config(379), arm.clone()),
                    circuit_pool(),
                    &designers,
                    None,
                )
                .expect("circuits"),
            );

            for events in [grid, circuits] {
                for soul in soul_groups(&events) {
                    let json = serde_json::to_string(soul).expect("json");
                    let mut numbers = Vec::new();

                    json_numbers(&serde_json::from_str(&json).expect("value"), &mut numbers);

                    assert!(!json.contains("null"), "{json}");
                    assert!(
                        numbers
                            .iter()
                            .all(|x| x.is_finite() && x.to_bits() != (-0.0f64).to_bits()),
                        "{json}"
                    );
                    assert_eq!(
                        &serde_json::from_str::<SoulState>(&json).expect("back"),
                        soul
                    );
                    // The Character statistics are defined on every committed archive.
                    assert!(soul.contrast.is_some() && soul.order.is_some(), "{json}");
                    checked += 1;
                }
            }
        }

        assert_eq!(
            checked,
            4 * 2 * 4,
            "four arms, two Domains, four state lines each"
        );

        // Arm N, without Souls, writes no soul group.
        let (_, plain) =
            bred_with_events(Breeder::new(small_config(373), &pool, None).expect("plain"));

        assert!(soul_groups(&plain).is_empty());
    }

    /// Arm S with the setting `variation`.
    fn varied(variation: Variation) -> SoulConfig {
        SoulConfig {
            variation,
            ..SoulConfig::arm_s()
        }
    }

    #[test]
    fn variation_needs_a_gene_pool_that_names_families() {
        let pool = sim_pool();
        let designers = designer_pool();

        for variation in [Variation::Prefer, Variation::Fixed] {
            // The grid gene pool cannot name the families of a text.
            let reason = invalid(
                Breeder::new(
                    with_souls(small_config(331), varied(variation)),
                    &pool,
                    None,
                ),
                "grid",
            );

            assert!(reason.contains("variation"), "{reason}");

            // The circuit gene pool can.
            assert!(
                Breeder::<CircuitDomain>::start(
                    with_souls(circuit_config(337), varied(variation)),
                    circuit_pool(),
                    &designers,
                    None,
                )
                .is_ok()
            );
        }

        // The fixed preference needs the families it favours: a grid gene pool that names
        // families still lacks Linear, Phase and Hadamard.
        let named = GenePool {
            families_of: Some(|_| vec![0]),
            ..GenePool::grid()
        };
        let reason = invalid(
            Breeder::<GridDomain>::start(
                with_souls(small_config(331), varied(Variation::Fixed)),
                named,
                &pool,
                None,
            ),
            "fixed on grids",
        );

        assert!(reason.contains("Linear"), "{reason}");
        assert!(
            Breeder::<GridDomain>::start(
                with_souls(small_config(331), varied(Variation::Prefer)),
                named,
                &pool,
                None,
            )
            .is_ok()
        );

        // A resume refuses a change of the setting, as of every soul setting.
        let snap = Breeder::<CircuitDomain>::start(
            with_souls(circuit_config(337), SoulConfig::arm_s()),
            circuit_pool(),
            &designers,
            None,
        )
        .expect("arm S")
        .snapshot();
        let reason = invalid(
            Breeder::<CircuitDomain>::resume_from(
                with_souls(circuit_config(337), varied(Variation::Prefer)),
                circuit_pool(),
                &designers,
                None,
                snap,
            ),
            "resume",
        );

        assert!(reason.contains("resume refused"), "{reason}");
    }

    /// What generation 1 of a circuit run with the Soul settings `soul` drafts, 24 children,
    /// after every Worker of generation 0 got the birth Soul `birth` and the imprinted and
    /// current Soul `now`: each child's gene id, parents and operator, and the place in the
    /// generation's stream afterwards; and the choices the preferences decided.
    fn steered_drafts(soul: SoulConfig, birth: [u16; 8], now: [u16; 8]) -> (String, Steered) {
        let designers = designer_pool();
        let never = || false;
        let cfg = BreedConfig {
            children_per_generation: 24,
            ..circuit_config(347)
        };
        let mut b = Breeder::<CircuitDomain>::start(
            with_souls(cfg, soul),
            circuit_pool(),
            &designers,
            None,
        )
        .expect("circuits");

        assert_eq!(b.step(&never).expect("generation 0").0, Status::Continue);

        for m in &mut b.archive {
            let n = m.cascade.worker_count();

            for w in m.cascade.stages.iter_mut().flat_map(|s| &mut s.workers) {
                w.traits.phase_state = Some(PhaseState(birth.to_vec()));
            }

            m.imprinted = Some(vec![PhaseState(now.to_vec()); n]);
            m.soul_now = Some(vec![PhaseState(now.to_vec()); n]);
        }

        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(b.cfg.seed, "gen/1"));
        let Drafted::Ready(drafts, _, _) = b.draft_children(1, &mut rng).expect("draft") else {
            panic!("generation 1 waits for answers");
        };
        let mut steered = Steered::default();
        let mut record = String::new();

        for d in &drafts {
            steered.add(d.steered);
            record.push_str(&format!(
                "{} {} {:?}\n",
                d.cascade.gene_id(),
                d.parents.join(","),
                d.op
            ));
        }

        record.push_str(&format!("pos={}", rng.get_word_pos()));

        (record, steered)
    }

    #[test]
    fn the_birth_soul_steers_variation() {
        let quarter = [QUARTER_TURN; 8];
        let mut linear = quarter;

        // Linear at no turn: its preference is 2.
        linear[0] = 0;

        let (today, none) = steered_drafts(SoulConfig::arm_s(), quarter, quarter);

        assert_eq!(none, Steered::default());

        // With every birth Soul at a quarter turn every preference is 1, and arm W drafts
        // exactly what arm S drafts, whatever the current Souls.
        for now in [quarter, linear, [0; 8]] {
            assert_eq!(
                steered_drafts(varied(Variation::Prefer), quarter, now),
                (today.clone(), Steered::default())
            );
        }

        // A birth Soul that prefers Linear steers: the preferences decide choices and
        // change some, and the children differ.
        let (by_birth, counts) = steered_drafts(varied(Variation::Prefer), linear, quarter);

        assert!(
            counts.changed > 0 && counts.changed <= counts.choices,
            "{counts:?}"
        );
        assert_ne!(by_birth, today);

        // Arm WF steers by its fixed preference, whatever the Souls.
        let (fixed, counts) = steered_drafts(varied(Variation::Fixed), quarter, quarter);

        assert!(counts.choices > 0, "{counts:?}");
        assert_ne!(fixed, today);
        assert_eq!(
            steered_drafts(varied(Variation::Fixed), linear, [0; 8]),
            (fixed, counts)
        );
    }

    #[test]
    fn a_run_with_variation_logs_its_choices_and_resumes_like_a_straight_one() {
        let designers = designer_pool();
        let never = || false;
        let run = |soul: SoulConfig| {
            let cfg = with_souls(circuit_config(349), soul);
            let start = || {
                Breeder::<CircuitDomain>::start(cfg.clone(), circuit_pool(), &designers, None)
                    .expect("circuits")
            };
            let mut straight = start();
            let mut events = Vec::new();

            while let (Status::Continue, ev) = straight.step(&never).expect("step") {
                events.extend(ev);
            }

            // Stopped after two generations and resumed from the snapshot: the same events.
            let mut first = start();
            let mut split = Vec::new();

            for _ in 0..2 {
                split.extend(first.step(&never).expect("step").1);
            }

            let json = serde_json::to_string(&first.snapshot()).expect("json");
            let mut second = Breeder::<CircuitDomain>::resume_from(
                cfg.clone(),
                circuit_pool(),
                &designers,
                None,
                serde_json::from_str(&json).expect("back"),
            )
            .expect("resume");

            while let (Status::Continue, ev) = second.step(&never).expect("step") {
                split.extend(ev);
            }

            assert_eq!(events, split);
            assert_eq!(straight.snapshot(), second.snapshot());

            events
        };
        let counts = |events: &[Event]| -> Vec<(Option<u64>, Option<u64>)> {
            soul_groups(events)
                .iter()
                .map(|s| (s.variation_choices, s.variation_changed))
                .collect()
        };

        // Arm W counts in every state line: nothing for the seeds, then the choices of the
        // children, of which the weights changed some.
        let steered = counts(&run(varied(Variation::Prefer)));

        assert_eq!(steered.len(), 4);
        assert_eq!(steered[0], (Some(0), Some(0)));

        for &(choices, changed) in &steered {
            assert!(changed.expect("changed") <= choices.expect("choices"));
        }

        assert!(
            steered.iter().any(|&(_, changed)| changed > Some(0)),
            "{steered:?}"
        );

        // Arm S leaves both out.
        assert!(
            counts(&run(SoulConfig::arm_s()))
                .iter()
                .all(|c| *c == (None, None))
        );
    }

    // Experience, imagination and changing goals (`.scratch/seelen-modell/spec-imagination.md`,
    // tickets 04 and 05).

    /// `cfg` with experience and imagination as in arm I.
    fn imagining(cfg: BreedConfig) -> BreedConfig {
        BreedConfig {
            imagination: Some(ImaginationConfig::arm_i()),
            ..cfg
        }
    }

    /// `cfg` with lexicase selection and goals of `length` generations each.
    fn with_goals(cfg: BreedConfig, length: usize) -> BreedConfig {
        BreedConfig {
            selection: ParentSelection::Lexicase,
            goals: Some(GoalsConfig { length }),
            ..cfg
        }
    }

    /// The mind group of every state line among `events`, in order.
    fn mind_groups(events: &[Event]) -> Vec<Option<MindState>> {
        events
            .iter()
            .filter_map(|e| match e {
                Event::State(s) => Some(s.mind.clone()),
                _ => None,
            })
            .collect()
    }

    /// A circuit breeder with the settings `cfg` whose generation 0, the seeds, is committed.
    fn circuits_after_seeds(cfg: BreedConfig, designers: &ModelPool) -> Breeder<'_, CircuitDomain> {
        let never = || false;
        let mut b = Breeder::<CircuitDomain>::start(cfg, circuit_pool(), designers, None)
            .expect("circuits");

        assert_eq!(b.step(&never).expect("generation 0").0, Status::Continue);

        b
    }

    /// What generation `g` of `b` drafts from its archive as it stands: each child's id,
    /// parents, operator, Worker sources and the choices of arm W, then the word position of
    /// the generation's stream afterwards; and the drafts.
    fn draft_record<D: Domain>(b: &Breeder<'_, D>, g: usize) -> (String, Vec<Draft>) {
        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(b.cfg.seed, &format!("gen/{g}")));
        let Drafted::Ready(drafts, _, _) = b.draft_children(g, &mut rng).expect("draft") else {
            panic!("generation {g} waits for answers");
        };
        let mut record = String::new();

        for d in &drafts {
            record.push_str(&format!(
                "{} {} {:?} {:?} {:?}\n",
                d.cascade.id(),
                d.parents.join(","),
                d.op,
                d.sources,
                d.steered
            ));
        }

        record.push_str(&format!("pos={}", rng.get_word_pos()));

        (record, drafts)
    }

    /// A member with the Cascade `cascade` whose values at birth and current values are
    /// `values`, one row per Worker.
    fn valued(cascade: Cascade, values: Vec<Vec<i16>>) -> Member<Circuit> {
        Member {
            id: cascade.id(),
            cascade,
            parents: Vec::new(),
            op: Op::Seed,
            generation: 0,
            answers: Vec::new(),
            solved: 0,
            quality: None,
            usage: Usage::default(),
            children: 0,
            judge_correct: 0,
            judge_wrong: 0,
            fitness: 0.0,
            rating: 0.0,
            soul_sources: None,
            imprinted: None,
            soul_now: None,
            teaching: None,
            authors: None,
            votes: None,
            values: Some(values.clone()),
            values_now: Some(values),
            effect: None,
        }
    }

    /// The single Worker of each circuit seed: linear, phase, hadamard and search.
    fn seed_workers() -> Vec<Worker> {
        (circuit_pool().seeds)()
            .into_iter()
            .map(|(_, c)| c.stages[0].workers[0].clone())
            .collect()
    }

    /// Eight values, all 0 but those of `given` (family, value).
    fn eight(given: &[(usize, i16)]) -> Vec<i16> {
        let mut v = vec![0; 8];

        for &(f, x) in given {
            v[f] = x;
        }

        v
    }

    #[test]
    fn imagination_and_goals_are_refused_where_they_cannot_work() {
        let pool = sim_pool();
        let designers = designer_pool();
        let named = GenePool {
            families_of: Some(|_| vec![0]),
            ..GenePool::grid()
        };

        // Imagination reads the families a Worker names, and the grid gene pool names none.
        let reason = invalid(
            Breeder::new(imagining(small_config(6101)), &pool, None),
            "grid",
        );

        assert!(reason.contains("imagination needs a gene pool"), "{reason}");
        assert!(
            Breeder::<CircuitDomain>::start(
                imagining(circuit_config(6103)),
                circuit_pool(),
                &designers,
                None
            )
            .is_ok()
        );
        assert!(
            Breeder::<GridDomain>::start(imagining(small_config(6101)), named, &pool, None).is_ok()
        );

        // Changing goals need lexicase selection and the circuits Domain.
        for selection in [
            ParentSelection::Dgm,
            ParentSelection::Uniform,
            ParentSelection::Tournament,
        ] {
            let cfg = BreedConfig {
                selection,
                ..with_goals(circuit_config(6107), 20)
            };
            let reason = invalid(
                Breeder::<CircuitDomain>::start(cfg, circuit_pool(), &designers, None),
                "selection",
            );

            assert!(reason.contains("goals need lexicase selection"), "{reason}");
        }

        let reason = invalid(
            Breeder::<GridDomain>::start(with_goals(small_config(6109), 20), named, &pool, None),
            "grid goals",
        );

        assert!(
            reason.contains("goals need lexicase selection and the circuits Domain"),
            "{reason}"
        );
        assert!(
            Breeder::<CircuitDomain>::start(
                with_goals(circuit_config(6107), 20),
                circuit_pool(),
                &designers,
                None
            )
            .is_ok()
        );

        // A setting out of range fails the check of the whole config.
        for cfg in [
            BreedConfig {
                imagination: Some(ImaginationConfig {
                    draws: 0,
                    rate: 250,
                }),
                ..circuit_config(6103)
            },
            BreedConfig {
                imagination: Some(ImaginationConfig {
                    draws: 4,
                    rate: 1001,
                }),
                ..circuit_config(6103)
            },
            with_goals(circuit_config(6107), 0),
        ] {
            assert!(cfg.check().is_err(), "{cfg:?}");
            invalid(
                Breeder::<CircuitDomain>::start(cfg, circuit_pool(), &designers, None),
                "out of range",
            );
        }

        // A resume refuses a change of either setting, in both directions.
        let base = BreedConfig {
            selection: ParentSelection::Lexicase,
            ..circuit_config(6113)
        };
        let fewer = BreedConfig {
            imagination: Some(ImaginationConfig {
                draws: 3,
                rate: 250,
            }),
            ..base.clone()
        };
        let arms = [
            base.clone(),
            imagining(base.clone()),
            with_goals(base.clone(), 20),
            with_goals(imagining(base.clone()), 20),
            with_goals(imagining(base.clone()), 10),
            fewer,
        ];

        for from in &arms {
            let snap = circuits_after_seeds(from.clone(), &designers).snapshot();

            for to in arms.iter().filter(|to| *to != from) {
                let reason = invalid(
                    Breeder::<CircuitDomain>::resume_from(
                        to.clone(),
                        circuit_pool(),
                        &designers,
                        None,
                        snap.clone(),
                    ),
                    "resume",
                );

                assert!(reason.contains("resume refused"), "{reason}");
            }
        }

        // The archive of a resume keeps the rule of its Run: values of the right shape and in
        // range with imagination, none without.
        let cfg = imagining(base.clone());
        let snap = circuits_after_seeds(cfg.clone(), &designers).snapshot();
        let tampers: [fn(&mut Member<Circuit>); 5] = [
            |m| m.values_now.as_mut().expect("values")[0][3] = 1001,
            |m| m.values.as_mut().expect("values")[0][0] = i16::MIN,
            |m| m.values.as_mut().expect("values")[0].push(0),
            |m| m.values_now = None,
            |m| {
                m.effect = Some(Effect {
                    p: 0,
                    s: 2,
                    gained: vec![Vec::new()],
                    lost: vec![Vec::new()],
                    draw: None,
                })
            },
        ];

        assert!(
            Breeder::<CircuitDomain>::resume_from(
                cfg.clone(),
                circuit_pool(),
                &designers,
                None,
                snap.clone()
            )
            .is_ok()
        );

        for tamper in tampers {
            let mut changed = snap.clone();

            tamper(&mut changed.archive[0]);

            let reason = invalid(
                Breeder::<CircuitDomain>::resume_from(
                    cfg.clone(),
                    circuit_pool(),
                    &designers,
                    None,
                    changed,
                ),
                "tampered values",
            );

            assert!(reason.contains("resume refused: member"), "{reason}");
        }

        let mut plain = circuits_after_seeds(base.clone(), &designers).snapshot();

        plain.archive[1].values_now = Some(vec![vec![0; 8]]);

        let reason = invalid(
            Breeder::<CircuitDomain>::resume_from(base, circuit_pool(), &designers, None, plain),
            "values without imagination",
        );

        assert!(reason.contains("no imagination"), "{reason}");
    }

    #[test]
    fn with_every_value_zero_imagination_drafts_todays_children() {
        let designers = designer_pool();
        let never = || false;

        for selection in [ParentSelection::Dgm, ParentSelection::Lexicase] {
            let cfg = BreedConfig {
                selection,
                children_per_generation: 24,
                ..circuit_config(6121)
            };
            let mut today = circuits_after_seeds(cfg.clone(), &designers);
            let mut imagined = circuits_after_seeds(imagining(cfg), &designers);

            // The seeds start at 0.
            assert!(
                imagined
                    .archive
                    .iter()
                    .flat_map(|m| m.values_now.iter().flatten().flatten())
                    .all(|&v| v == 0)
            );

            for g in 1..=3 {
                // Generation 1 drafts from the seeds as they are; later generations after
                // their values are set back to 0.
                for v in imagined
                    .archive
                    .iter_mut()
                    .flat_map(|m| m.values_now.iter_mut().flatten())
                {
                    v.fill(0);
                }

                let (want, _) = draft_record(&today, g);
                let (got, drafts) = draft_record(&imagined, g);

                // The same children, and the generation's stream ends at the same word.
                assert_eq!(got, want, "{selection:?}, generation {g}");

                // Imagination drew for every child of an imagining operator, and with every
                // prediction 0 each chose today's first draw.
                for d in &drafts {
                    assert_eq!(
                        d.imagined.is_some(),
                        IMAGINING_OPS.contains(&d.op),
                        "{:?}",
                        d.op
                    );
                }

                assert!(
                    drafts.iter().any(|d| d.imagined.is_some()),
                    "generation {g}"
                );
                assert!(drafts.iter().filter_map(|d| d.imagined).all(|i| i
                    == Imagined {
                        draw: 1,
                        choice: false
                    }));

                assert_eq!(today.step(&never).expect("step").0, Status::Continue);
                assert_eq!(imagined.step(&never).expect("step").0, Status::Continue);
            }

            // Both Runs bred the same members; only the learned values set them apart, and
            // learning gave some Workers values other than 0.
            let ids = |b: &Breeder<'_, CircuitDomain>| -> Vec<String> {
                b.archive.iter().map(|m| m.id.clone()).collect()
            };

            assert_eq!(ids(&today), ids(&imagined));
            assert!(
                imagined
                    .archive
                    .iter()
                    .flat_map(|m| m.values_now.iter().flatten().flatten())
                    .any(|&v| v != 0)
            );
        }
    }

    #[test]
    fn imagination_chooses_the_draw_with_the_highest_prediction() {
        let designers = designer_pool();
        let cfg = BreedConfig {
            selection: ParentSelection::Lexicase,
            children_per_generation: 24,
            ..circuit_config(6131)
        };
        let today = circuits_after_seeds(cfg.clone(), &designers);
        let mut b = circuits_after_seeds(imagining(cfg), &designers);

        // Every seed Worker values Hadamard at 1 and Linear at −1.
        for v in b
            .archive
            .iter_mut()
            .flat_map(|m| m.values_now.iter_mut().flatten())
        {
            *v = eight(&[(0, -1000), (2, 1000)]);
        }

        let (_, drafts) = draft_record(&b, 1);
        let (_, firsts) = draft_record(&today, 1);
        let at: HashMap<String, usize> = b
            .archive
            .iter()
            .enumerate()
            .map(|(i, m)| (m.id.clone(), i))
            .collect();
        let mut seen: HashSet<String> = b.gene_ids.iter().cloned().collect();
        let mut changed = 0;

        // Draw 1 comes from the generation's stream, so it is the child of today's Run at the
        // same place as long as both Runs made their children by the same attempts.
        assert_eq!(drafts.len(), firsts.len());
        assert!(
            drafts
                .iter()
                .zip(&firsts)
                .all(|(d, f)| d.parents == f.parents && d.op == f.op),
            "the Runs part; choose another seed"
        );

        for (i, (d, first)) in drafts.iter().zip(&firsts).enumerate() {
            let parents: Vec<usize> = d.parents.iter().map(|p| at[p]).collect();
            let p_of =
                |c: &Cascade, s: &Sources| b.experience(&parents, c, s).expect("experience").p;

            if let Some(imagined) = d.imagined {
                let mut predictions = vec![p_of(&first.cascade, &first.sources)];
                let mut ids = vec![first.cascade.id()];

                for j in 2..=4 {
                    let tag = format!("imagine/1/{i}/{j}");
                    let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(b.cfg.seed, &tag));

                    if let Some(drawn) = b.redraw(&parents, d.op, None, &mut rng)
                        && b.admissible(&drawn.cascade, &seen)
                    {
                        predictions.push(p_of(&drawn.cascade, &drawn.sources));
                        ids.push(drawn.cascade.id());
                    } else {
                        predictions.push(i64::MIN);
                        ids.push(String::new());
                    }
                }

                // The chosen draw has the highest prediction, and no earlier draw has it.
                let best = *predictions.iter().max().expect("a prediction");
                let k = imagined.draw - 1;

                assert_eq!(predictions[k], best, "child {i}: {predictions:?}");
                assert!(
                    predictions[..k].iter().all(|&p| p < best),
                    "{predictions:?}"
                );
                assert_eq!(ids[k], d.cascade.id());
                assert_eq!(
                    imagined.choice,
                    predictions.iter().any(|&p| p != 0 && p != i64::MIN)
                );

                changed += usize::from(imagined.draw != 1);
            } else {
                // Without imagination the child is today's.
                assert_eq!(d.cascade.id(), first.cascade.id());
            }

            seen.insert(d.cascade.id());
        }

        // The values changed picks.
        assert!(changed > 0);
    }

    #[test]
    fn a_child_worker_copies_the_values_of_its_source_worker() {
        let designers = designer_pool();
        let mut b = circuits_after_seeds(imagining(circuit_config(6137)), &designers);
        let w = seed_workers();
        let limits = b.cfg.limits.clone();

        // A seed has no source: its Workers start at 0, and nothing changed.
        for m in &b.archive {
            assert_eq!(m.values, Some(vec![vec![0; 8]]));
            assert_eq!(m.values_now, m.values);
            assert_eq!(m.effect, None);
        }

        // Parent A: the linear and the phase Worker in one Stage, then the hadamard Worker.
        // Parent B: the search and the phase Worker. Each Worker's values mark it: Previous
        // holds 100 times its place plus 100, and Simplify its parent.
        let a = Cascade {
            stages: vec![
                Stage {
                    workers: vec![w[0].clone(), w[1].clone()],
                },
                Stage {
                    workers: vec![w[2].clone()],
                },
            ],
        };
        let other = Cascade {
            stages: vec![Stage {
                workers: vec![w[3].clone(), w[1].clone()],
            }],
        };
        let mark = |parent: i16, k: i16| eight(&[(6, parent), (7, 100 * (k + 1))]);

        b.archive
            .push(valued(a.clone(), vec![mark(1, 0), mark(1, 1), mark(1, 2)]));
        b.archive
            .push(valued(other.clone(), vec![mark(2, 0), mark(2, 1)]));

        let (ia, ib) = (b.archive.len() - 2, b.archive.len() - 1);
        let values_of =
            |m: usize, k: usize| b.archive[m].values_now.as_ref().expect("values")[k].clone();
        let check = |parents: &[usize], child: &Cascade, sources: &Sources| {
            let exp = b.experience(parents, child, sources).expect("experience");

            for (k, from) in sources.iter().enumerate() {
                let src = from[0];
                let m = parents[src.parent];

                assert_eq!(exp.start[k], values_of(m, src.worker), "Worker {k}");
                assert_eq!(exp.sources[k], Some((m, src.worker)));
            }

            exp
        };
        let mut grown = BTreeSet::new();

        // A mutation: each Worker copies its source, the parent's Worker it was built from,
        // also a Worker that WorkerAdd cloned or StageAdd copied.
        let mutations = [
            Op::ThinkUp,
            Op::ThinkDown,
            Op::TemperatureShift,
            Op::SelfCheckFlip,
            Op::AnswerTokensShift,
            Op::ModelSwitch,
            Op::SectionReplace,
            Op::SectionAdd,
            Op::SectionDrop,
            Op::SectionSwap,
            Op::WorkerAdd,
            Op::WorkerDrop,
            Op::StageAdd,
            Op::StageDrop,
        ];

        for op in mutations {
            for seed in 0..12 {
                let mut rng = ChaCha8Rng::seed_from_u64(6137 + seed);

                if let Some((child, sources, _)) =
                    mutate_once(&a, op, &limits, CIRCUIT_PHRASES, None, &mut rng)
                {
                    check(&[ia], &child, &sources);

                    if child.worker_count() > a.worker_count() {
                        grown.insert(format!("{op:?}"));
                    }
                }
            }
        }

        assert_eq!(
            grown,
            BTreeSet::from(["StageAdd".to_string(), "WorkerAdd".to_string()])
        );

        // A section crossover: each Worker copies the first parent's Worker at its place,
        // never the partner's.
        for seed in 0..12 {
            let mut rng = ChaCha8Rng::seed_from_u64(6151 + seed);

            if let Some((child, sources)) = section_crossover_sourced(&a, &other, &limits, &mut rng)
            {
                let exp = check(&[ia, ib], &child, &sources);

                assert!(exp.start.iter().all(|v| v[6] == 1), "{:?}", exp.start);
            }
        }

        // A stage crossover: the prefix copies A's Workers and the suffix B's.
        let mut suffixed = 0;

        for seed in 0..24 {
            let mut rng = ChaCha8Rng::seed_from_u64(6163 + seed);

            if let Some((child, sources)) = stage_crossover_sourced(&a, &other, &limits, &mut rng) {
                let exp = check(&[ia, ib], &child, &sources);

                assert_eq!(exp.start.last().map(|v| v[6]), Some(2));
                suffixed += 1;
            }
        }

        assert!(suffixed > 0);

        // A new Worker, one without a source, starts at 0 and gains and loses nothing.
        let exp = b
            .experience(&[ia], &a, &vec![Vec::new(); 3])
            .expect("experience");

        assert_eq!(exp.start, vec![vec![0; 8]; 3]);
        assert_eq!(exp.changes, vec![Change::default(); 3]);
        assert_eq!((exp.p, exp.families()), (0, 0));
    }

    #[test]
    fn a_section_crossover_that_changes_two_workers_learns_one_error() {
        let designers = designer_pool();
        let mut b = circuits_after_seeds(imagining(circuit_config(6173)), &designers);
        let w = seed_workers();
        let limits = b.cfg.limits.clone();

        // A: the linear and the phase Worker; B: the hadamard and the search Worker, each with
        // a style section that A lacks, so a child of A's shape never equals B.
        let a = Cascade {
            stages: vec![Stage {
                workers: vec![w[0].clone(), w[1].clone()],
            }],
        };
        let mut other = Cascade {
            stages: vec![Stage {
                workers: vec![w[2].clone(), w[3].clone()],
            }],
        };

        for worker in &mut other.stages[0].workers {
            worker.sections.push(Section {
                name: "style".to_string(),
                text: "Be concise.".to_string(),
            });
        }

        // The child takes both strategies from B, as the coins of section crossover may.
        let mut child = a.clone();

        for (mine, theirs) in child.stages[0]
            .workers
            .iter_mut()
            .zip(&other.stages[0].workers)
        {
            mine.sections[1] = theirs.sections[1].clone();
        }

        assert!(child.check(&limits).is_ok() && child.id() != a.id() && child.id() != other.id());

        // Its sources are those that section crossover names for this pair: each Worker of A
        // with its partner in B.
        let mut rng = ChaCha8Rng::seed_from_u64(6173);
        let (_, sources) =
            section_crossover_sourced(&a, &other, &limits, &mut rng).expect("a crossover");
        let own = |worker| Source { parent: 0, worker };
        let partner = |worker| Source { parent: 1, worker };

        assert_eq!(
            sources,
            vec![vec![own(0), partner(0)], vec![own(1), partner(1)]]
        );

        // A's first Worker values Linear at 0.4 and Hadamard at 0.1, its second Phase at −0.2
        // and Search at 0.3. B's values play no part.
        let a0 = eight(&[(0, 400), (2, 100)]);
        let a1 = eight(&[(1, -200), (4, 300)]);

        b.archive.push(valued(a, vec![a0.clone(), a1.clone()]));
        b.archive
            .push(valued(other, vec![vec![900; 8], vec![-900; 8]]));

        let (ia, ib) = (b.archive.len() - 2, b.archive.len() - 1);
        let exp = b
            .experience(&[ia, ib], &child, &sources)
            .expect("experience");

        // Worker 0 lost Linear and gained Hadamard; Worker 1 lost Phase and gained Search.
        assert_eq!(
            exp.changes,
            vec![
                Change {
                    gained: vec![2],
                    lost: vec![0]
                },
                Change {
                    gained: vec![4],
                    lost: vec![1]
                },
            ]
        );
        assert_eq!(exp.families(), 4);

        // One p over both Workers, each read in its own source: (0.1 − 0.4) + (0.3 + 0.2).
        assert_eq!(exp.p, 200);

        // One error for the child, one step for its four families: with s = 1, e = 800 and
        // the step 800 / 16 = 50; with s = −1, e = −1200 and the step −75; with s = 0,
        // e = −200 and −200 / 16 = −12.5, which rounds away from zero to −13.
        let moved = |step: i16| {
            vec![
                eight(&[(0, 400 - step), (2, 100 + step)]),
                eight(&[(1, -200 - step), (4, 300 + step)]),
            ]
        };

        for (s, step) in [(1, 50), (-1, -75), (0, -13)] {
            let (values, got) = exp.learn(s, 250);

            assert_eq!(got, i64::from(step), "s {s}");
            assert_eq!(values, moved(step), "s {s}");
        }

        // At the commit A's two Workers learn the same step; B, the partner, does not.
        let mut archive = b.archive.clone();
        let changed = learn_at_commit(&mut archive, &[exp.lesson(50)]).expect("learn");
        let new = moved(50);

        assert_eq!(
            changed,
            vec![
                ValueChange {
                    member: archive[ia].id.clone(),
                    worker: 0,
                    values: new[0].clone(),
                },
                ValueChange {
                    member: archive[ia].id.clone(),
                    worker: 1,
                    values: new[1].clone(),
                },
            ]
        );
        assert_eq!(archive[ib].values_now, b.archive[ib].values_now);
    }

    #[test]
    fn source_workers_learn_the_steps_of_their_children_in_birth_order() {
        let designers = designer_pool();
        let mut b = circuits_after_seeds(imagining(circuit_config(6179)), &designers);

        // The linear seed's Worker values Hadamard at 0.9.
        b.archive[0].values_now = Some(vec![eight(&[(2, 900)])]);

        let gained = |f: usize| Change {
            gained: vec![f],
            lost: Vec::new(),
        };
        let lost = |f: usize| Change {
            gained: Vec::new(),
            lost: vec![f],
        };
        // In birth order: a child whose step raises Hadamard on the linear Worker by 250,
        // then one whose step lowers it by 250 and that lost Phase against the phase
        // Worker, then one whose step is 0.
        let lessons = [
            Lesson {
                step: 250,
                changes: vec![((0, 0), gained(2))],
            },
            Lesson {
                step: -250,
                changes: vec![((0, 0), gained(2)), ((1, 0), lost(1))],
            },
            Lesson {
                step: 0,
                changes: vec![((2, 0), gained(5))],
            },
        ];
        let mut archive = b.archive.clone();
        let changed = learn_at_commit(&mut archive, &lessons).expect("learn");

        // 900 + 250 stops at 1000, and then 1000 − 250 = 750. Lost Phase moves by minus the
        // step: −(−250) = +250.
        assert_eq!(
            changed,
            vec![
                ValueChange {
                    member: archive[0].id.clone(),
                    worker: 0,
                    values: eight(&[(2, 750)]),
                },
                ValueChange {
                    member: archive[1].id.clone(),
                    worker: 0,
                    values: eight(&[(1, 250)]),
                },
            ]
        );

        // The hadamard seed, whose step was 0, did not change and is not listed.
        assert_eq!(archive[2].values_now, b.archive[2].values_now);

        // In the other order the clamp acts last: 900 − 250 = 650, then 900.
        let mut reversed = b.archive.clone();

        learn_at_commit(&mut reversed, &[lessons[1].clone(), lessons[0].clone()]).expect("learn");
        assert_eq!(reversed[0].values_now, Some(vec![eight(&[(2, 900)])]));

        // A source Worker without values is an integrity error.
        let mut none = b.archive.clone();

        none[0].values_now = None;
        assert!(matches!(
            learn_at_commit(&mut none, &lessons),
            Err(BrutError::Integrity(_))
        ));
    }

    /// The operators whose children keep their first parent's shape, each Worker built from
    /// the parent's Worker at its place.
    const POSITIONAL: [Op; 10] = [
        Op::SectionReplace,
        Op::SectionAdd,
        Op::SectionDrop,
        Op::SectionSwap,
        Op::ThinkUp,
        Op::ThinkDown,
        Op::TemperatureShift,
        Op::SelfCheckFlip,
        Op::AnswerTokensShift,
        Op::SectionCrossover,
    ];

    /// Check that the log `events` holds what learning read and did (spec-imagination 3.3,
    /// 3.5), from the log alone: every newborn's values are its source Workers' values after
    /// the previous commit moved by its one step, its p is the sum over its changed Workers,
    /// and each `learned` event holds the steps of the generation's children applied to their
    /// source Workers in birth order. Children of operators that move Workers are checked
    /// where their Workers keep their places, and their parents' learning is not checked when
    /// such a child changed families. Returns the newborns that learned something.
    fn check_learning_in(events: &[Event], rate: u16) -> usize {
        let mut now: HashMap<String, Vec<Vec<i16>>> = HashMap::new();
        let mut taught: BTreeMap<String, Vec<(i64, Vec<Change>)>> = BTreeMap::new();
        let mut unknown: BTreeSet<String> = BTreeSet::new();
        let mut learned = 0;

        for e in events {
            match e {
                Event::Born {
                    generation,
                    parents,
                    op,
                    cascade,
                    values,
                    effect,
                    id,
                    ..
                } => {
                    let values = values.clone().expect("values");

                    assert_eq!(values.len(), cascade.worker_count());

                    if *generation == 0 {
                        assert!(values.iter().flatten().all(|&v| v == 0));
                        assert_eq!(*effect, None);
                    } else if POSITIONAL.contains(op) {
                        let start = now[&parents[0]].clone();
                        let changes: Vec<Change> = match effect {
                            Some(e) => e
                                .gained
                                .iter()
                                .zip(&e.lost)
                                .map(|(gained, lost)| Change {
                                    gained: gained.clone(),
                                    lost: lost.clone(),
                                })
                                .collect(),
                            None => vec![Change::default(); values.len()],
                        };
                        let n: usize = changes.iter().map(Change::families).sum();
                        let p =
                            prediction(changes.iter().zip(&start).map(|(c, v)| (c, v.as_slice())));
                        let step = effect
                            .as_ref()
                            .map_or(0, |e| step(error(e.s, e.p), n, rate));
                        let mut expected = start.clone();

                        for (v, c) in expected.iter_mut().zip(&changes) {
                            c.learn(v, step);
                        }

                        if let Some(e) = effect {
                            assert_eq!(e.p, p, "{id}");
                            assert!((-1..=1).contains(&e.s));
                            assert!(n > 0 || e.draw.is_some(), "{id}");
                        }

                        assert_eq!(
                            effect.is_some(),
                            n > 0 || IMAGINING_OPS.contains(op),
                            "{id}"
                        );
                        assert_eq!(values, expected, "{id}");

                        if n > 0 {
                            learned += 1;
                            taught
                                .entry(parents[0].clone())
                                .or_default()
                                .push((step, changes));
                        }
                    } else if effect
                        .as_ref()
                        .is_some_and(|e| e.gained.iter().chain(&e.lost).any(|f| !f.is_empty()))
                    {
                        learned += 1;
                        unknown.extend(parents.iter().cloned());
                    }

                    now.insert(id.clone(), values);
                }
                Event::Learned { changed, .. } => {
                    let listed: BTreeMap<(String, usize), Vec<i16>> = changed
                        .iter()
                        .map(|c| ((c.member.clone(), c.worker), c.values.clone()))
                        .collect();

                    for (parent, lessons) in std::mem::take(&mut taught) {
                        if unknown.contains(&parent) {
                            continue;
                        }

                        let mut expected = now[&parent].clone();

                        for (step, changes) in &lessons {
                            for (v, c) in expected.iter_mut().zip(changes) {
                                c.learn(v, *step);
                            }
                        }

                        for (k, v) in expected.iter().enumerate() {
                            match listed.get(&(parent.clone(), k)) {
                                Some(got) => assert_eq!(got, v, "{parent} Worker {k}"),
                                None => assert_eq!(&now[&parent][k], v, "{parent} Worker {k}"),
                            }
                        }
                    }

                    for c in changed {
                        now.get_mut(&c.member).expect("a member")[c.worker] = c.values.clone();
                    }

                    unknown.clear();
                }
                _ => {}
            }
        }

        learned
    }

    #[test]
    fn the_log_holds_what_learning_read_and_did() {
        let designers = designer_pool();
        let base = BreedConfig {
            generations: 8,
            children_per_generation: 12,
            selection: ParentSelection::Lexicase,
            ..circuit_config(6181)
        };

        for cfg in [
            imagining(base.clone()),
            with_goals(imagining(base.clone()), 2),
        ] {
            let start =
                Breeder::<CircuitDomain>::start(cfg.clone(), circuit_pool(), &designers, None)
                    .expect("circuits");
            let (b, events) = bred_with_events(start);

            // One `learned` event after each generation event, and the learning the log
            // records adds up.
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, Event::Learned { .. }))
                    .count(),
                cfg.generations + 1
            );

            for (i, e) in events.iter().enumerate() {
                if matches!(e, Event::Learned { .. }) {
                    let before = &events[i - 1];

                    assert!(matches!(before, Event::Generation { .. }), "{before:?}");
                    assert!(matches!(events[i + 1], Event::State(_)));
                }
            }

            assert!(check_learning_in(&events, 250) > 0, "nothing was learned");

            // The real effect s compares the solved counts on the goal of the newborn's
            // generation, its own and its first parent's.
            let by_id: HashMap<&str, &Member<Circuit>> =
                b.archive.iter().map(|m| (m.id.as_str(), m)).collect();
            let mut effects = 0;

            for m in &b.archive {
                let Some(effect) = &m.effect else {
                    continue;
                };
                let parent = by_id[m.parents[0].as_str()];
                let tasks = b.goal(m.generation).map(|goal| goal.tasks.as_slice());
                let s = crate::cascade::mind::effect(
                    b.solved_on(&m.answers, tasks).expect("grade"),
                    b.solved_on(&parent.answers, tasks).expect("grade"),
                );

                assert_eq!(effect.s, s, "{}", m.id);

                if cfg.goals.is_none() {
                    assert_eq!(s, crate::cascade::mind::effect(m.solved, parent.solved));
                }

                effects += 1;
            }

            assert!(effects > 0);
        }
    }

    #[test]
    fn the_mind_group_holds_the_goal_and_how_imagination_chose() {
        let designers = designer_pool();
        let base = BreedConfig {
            generations: 6,
            children_per_generation: 12,
            selection: ParentSelection::Lexicase,
            ..circuit_config(6191)
        };
        let run = |cfg: BreedConfig| {
            let b = Breeder::<CircuitDomain>::start(cfg, circuit_pool(), &designers, None)
                .expect("circuits");

            bred_with_events(b).1
        };

        // Off, there is no mind group, and no state line names one.
        let off = run(base.clone());

        assert!(mind_groups(&off).iter().all(Option::is_none));
        assert!(
            !serde_json::to_string(&off)
                .expect("json")
                .contains("\"mind\"")
        );
        assert!(!off.iter().any(|e| matches!(e, Event::Learned { .. })));

        // Goals of two generations: generations 0 to 2 have all tasks, 3 and 4 the targets
        // without Linear, 5 and 6 all tasks again. Nothing of imagination is logged.
        let goals: Vec<MindState> = mind_groups(&run(with_goals(base.clone(), 2)))
            .into_iter()
            .map(|m| m.expect("a mind group"))
            .collect();

        assert_eq!(
            goals.iter().map(|m| m.goal.as_str()).collect::<Vec<_>>(),
            ["all", "all", "all", "-Linear", "-Linear", "all", "all"]
        );

        for m in &goals {
            assert_eq!(
                (
                    m.imagination_choices,
                    m.imagination_changed,
                    m.linear_phase_hadamard_mean,
                    m.other_families_mean
                ),
                (None, None, None, None)
            );
        }

        // Imagination alone: the goal is all tasks; the seeds have no parents, and from
        // generation 1 on the parents' values have means. In generation 1 every value is 0.
        let events = run(imagining(base.clone()));
        let imagined: Vec<MindState> = mind_groups(&events)
            .into_iter()
            .map(|m| m.expect("a mind group"))
            .collect();

        assert_eq!(imagined.len(), 7);
        assert_eq!(
            (
                imagined[0].imagination_choices,
                imagined[0].imagination_changed,
                imagined[0].linear_phase_hadamard_mean
            ),
            (Some(0), Some(0), None)
        );
        assert_eq!(
            (
                imagined[1].imagination_choices,
                imagined[1].imagination_changed,
                imagined[1].linear_phase_hadamard_mean,
                imagined[1].other_families_mean
            ),
            (Some(0), Some(0), Some(0.0), Some(0.0))
        );

        for m in &imagined {
            assert_eq!(m.goal, "all");
            assert!(m.imagination_changed <= m.imagination_choices);
        }

        for m in &imagined[1..] {
            assert!(m.linear_phase_hadamard_mean.is_some() && m.other_families_mean.is_some());
        }

        assert!(
            imagined.iter().any(|m| m.imagination_choices > Some(0)),
            "{imagined:?}"
        );

        // The log alone gives the changed picks, the Born events whose chosen draw is not the
        // first, and the means: the values of the parents' Workers before the generation's
        // commit, which the Born events and the earlier `learned` events hold.
        let mut now: HashMap<String, Vec<Vec<i16>>> = HashMap::new();
        let mut parents: Vec<String> = Vec::new();
        let mut changed = 0u64;
        let mut sums = (ValueSums::default(), ValueSums::default());
        let mut checked = 0;

        for e in &events {
            match e {
                Event::Born {
                    id,
                    parents: p,
                    values,
                    effect,
                    ..
                } => {
                    parents.extend(p.iter().cloned());
                    changed += u64::from(effect.as_ref().and_then(|e| e.draw) > Some(1));
                    now.insert(id.clone(), values.clone().expect("values"));
                }
                Event::Learned { changed: moved, .. } => {
                    // Linear, Phase and Hadamard are families 0 to 2.
                    for parent in parents.drain(..) {
                        for (f, &x) in now[&parent].iter().flat_map(|v| v.iter().enumerate()) {
                            if f < 3 {
                                sums.0.add(x);
                            } else {
                                sums.1.add(x);
                            }
                        }
                    }

                    for c in moved {
                        now.get_mut(&c.member).expect("a member")[c.worker] = c.values.clone();
                    }
                }
                Event::State(s) => {
                    let m = s.mind.as_ref().expect("a mind group");

                    assert_eq!(m.imagination_changed, Some(changed), "{}", s.generation);
                    assert_eq!(m.linear_phase_hadamard_mean, sums.0.mean());
                    assert_eq!(m.other_families_mean, sums.1.mean());
                    changed = 0;
                    sums = (ValueSums::default(), ValueSums::default());
                    checked += 1;
                }
                _ => {}
            }
        }

        assert_eq!(checked, 7);

        // Both: the goals and the counts.
        let both: Vec<MindState> = mind_groups(&run(with_goals(imagining(base), 2)))
            .into_iter()
            .map(|m| m.expect("a mind group"))
            .collect();

        assert_eq!(both[3].goal, "-Linear");
        assert!(both.iter().all(|m| m.imagination_choices.is_some()));
    }

    #[test]
    fn lexicase_with_a_goal_draws_only_on_the_tasks_of_that_goal() {
        // Member 0 does well only on task 0, members 1 and 2 on tasks 1 and 2.
        let scores = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.5, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];
        let selector = |scores: Vec<Vec<f64>>| Selector {
            kind: ParentSelection::Lexicase,
            params: SelectionParams::default(),
            weights: Vec::new(),
            scores,
            rank_of: vec![0; 3],
            members: 3,
            observe_seed: 0,
        };
        let picks = |sel: &Selector| {
            let mut rng = ChaCha8Rng::seed_from_u64(6197);
            let mut counts = [0usize; 3];

            for _ in 0..3000 {
                counts[sel.pick(&mut rng).expect("a pick")] += 1;
            }

            counts
        };

        // On all tasks, member 0 is drawn whenever task 0 comes first.
        assert!(picks(&selector(scores.clone()))[0] > 0);

        // On the goal of tasks 1 and 2 it scores 0 on every case, so it is never drawn.
        let on_goal = picks(&selector(goal_cases(scores.clone(), &[1, 2])));

        assert_eq!(on_goal[0], 0);
        assert!(on_goal[1] > 0 && on_goal[2] > 0, "{on_goal:?}");
        assert_eq!(
            goal_cases(scores, &[1, 2]),
            vec![vec![0.0, 0.0], vec![0.5, 1.0], vec![1.0, 0.0]]
        );

        // A breeder with goals of one generation draws the parents of each generation on the
        // tasks of its goal, in panel order: generation 2 on cs and ch, generation 4 on cz,
        // swap and ch.
        let designers = designer_pool();
        let mut b = circuits_after_seeds(with_goals(circuit_config(6199), 1), &designers);
        let full: Vec<Vec<f64>> = b
            .archive
            .iter()
            .map(|m| {
                b.search
                    .iter()
                    .zip(&m.answers)
                    .map(|(t, a)| {
                        a.as_ref()
                            .and_then(|a| b.domain.grade(t, a).ok())
                            .map_or(0.0, |g| g.value)
                    })
                    .collect()
            })
            .collect();
        let names =
            |tasks: &[usize]| -> Vec<&str> { tasks.iter().map(|&t| b.search[t].name).collect() };

        for (g, goal, targets) in [
            (1, "all", None),
            (2, "-Linear", Some(vec!["cs", "ch"])),
            (3, "all", None),
            (4, "-Phase", Some(vec!["cz", "swap", "ch"])),
        ] {
            b.next_generation = g;

            let at = b.goal(g).expect("a goal").clone();
            let sel = b.selector();

            assert_eq!(at.name, goal);
            assert_eq!(
                sel.scores,
                goal_cases(full.clone(), &at.tasks),
                "generation {g}"
            );
            assert!(sel.scores.iter().all(|row| row.len() == at.tasks.len()));

            match targets {
                Some(targets) => assert_eq!(names(&at.tasks), targets),
                None => assert_eq!(at.tasks.len(), b.search.len()),
            }
        }
    }

    #[test]
    fn a_run_with_imagination_and_goals_resumes_like_a_straight_one() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let designers = designer_pool();
        let never = || false;
        let cfg = with_goals(
            imagining(BreedConfig {
                generations: 6,
                children_per_generation: 8,
                ..circuit_config(6203)
            }),
            2,
        );
        let start = || {
            Breeder::<CircuitDomain>::start(cfg.clone(), circuit_pool(), &designers, None)
                .expect("circuits")
        };
        let (straight, events) = bred_with_events(start());

        // Stopped after two generations and resumed from the snapshot's JSON, and stopped
        // inside generation 4 while its newborns run and resumed again: the same events and
        // the same final state.
        let mut first = start();
        let mut split = Vec::new();

        for _ in 0..2 {
            split.extend(first.step(&never).expect("step").1);
        }

        let json = serde_json::to_string(&first.snapshot()).expect("json");
        let mut second = Breeder::<CircuitDomain>::resume_from(
            cfg.clone(),
            circuit_pool(),
            &designers,
            None,
            serde_json::from_str(&json).expect("back"),
        )
        .expect("resume");

        for _ in 0..2 {
            split.extend(second.step(&never).expect("step").1);
        }

        let polls = AtomicUsize::new(0);
        let stop_later = || polls.fetch_add(1, Ordering::SeqCst) >= 9;
        let (status, ev) = second
            .step_with(&stop_later, &mut |_| Ok(()))
            .expect("stopped step");

        assert_eq!(status, Status::Done("stopped".to_string()));
        assert!(ev.is_empty());

        let json = serde_json::to_string(&second.snapshot()).expect("json");
        let mut third = Breeder::<CircuitDomain>::resume_from(
            cfg.clone(),
            circuit_pool(),
            &designers,
            None,
            serde_json::from_str(&json).expect("back"),
        )
        .expect("resume");

        while let (Status::Continue, ev) = third.step(&never).expect("step") {
            split.extend(ev);
        }

        // The stopped attempt's calls show in the cumulative calls of the generation events,
        // and nowhere else.
        let without_calls = |events: &[Event]| -> Vec<Event> {
            events
                .iter()
                .cloned()
                .map(|mut e| {
                    if let Event::Generation { calls_used, .. } = &mut e {
                        *calls_used = 0;
                    }

                    e
                })
                .collect()
        };

        assert_eq!(without_calls(&events), without_calls(&split));
        assert_ne!(events, split);
        assert_eq!(straight.archive, third.archive);
        assert_eq!(straight.comparisons, third.comparisons);
        assert!(third.calls_used() > straight.calls_used());
    }
}
