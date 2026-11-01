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
//! A Stage vote then tallies amplitudes: the verdicts of each judge weigh with its vote
//! phasor, and the answer with the largest squared magnitude goes on.
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, HashMap, HashSet};

use rand::seq::{IndexedRandom, SliceRandom};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::backend::{GenRequest, GenResponse, ModelBackend, StopReason, approx_tokens};
use crate::cascade::domain::{Domain, DomainConfig, GenePool, GridDomain};
use crate::cascade::genome::{
    COUPLING_MAX, Cascade, CascadeLimits, PhaseState, Worker, adapt_seeds,
};
use crate::cascade::observe::{ObserveInput, UniverseState, observe};
use crate::cascade::puzzle::{Grid, Puzzle, SELF_CHECK_TEXT, Verdict, parse_verdict};
use crate::cascade::soul::{
    ArrayVote, Inheritance, Phasor, Readout, SoulConfig, SoulDraws, array_vote, count_winner,
    souletics, vote_phasor,
};
use crate::cascade::variation::{
    Op, Sources, llm_crossover_request_with, llm_mutate_request_with, mutate_sourced,
    parse_llm_cascade, partner_sources, read_llm_cascade, section_crossover_sourced,
    stage_crossover_sourced,
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

/// The models and the limits every call runs under.
#[derive(Clone, Copy)]
struct Ctx<'a> {
    pool: &'a ModelPool,
    limits: &'a CascadeLimits,
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
    }
}

/// One Worker call with usage accounting. Every attempted call counts against the budget
/// and pays its model's fixed price, also when the backend fails, since a failed request
/// may already have used compute; tokens count only for an answer. A model without a
/// backend is not attempted and gives None. `check_text` is the domain's self-check text.
fn call_worker(
    ctx: Ctx,
    w: &Worker,
    prompt: String,
    seed: u64,
    check_text: &str,
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

    let req = worker_request_with(w, prompt, seed, check_text);
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
    let swap = ChaCha8Rng::seed_from_u64(derive_seed(seed, "order")).random_bool(0.5);
    let (first, second) = if swap { (b, a) } else { (a, b) };
    let prompt = domain.judge_prompt(task, first, second);
    let Some(resp) = call_worker(ctx, w, prompt, seed, domain.self_check_text(), usage) else {
        return Judgment::NoReply;
    };

    if resp.stop_reason != StopReason::Stop {
        return Judgment::NoVerdict;
    }

    match parse_verdict(&resp.text) {
        Some(Verdict::A) => Judgment::Verdict(!swap),
        Some(Verdict::B) => Judgment::Verdict(swap),
        None => Judgment::NoVerdict,
    }
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
    /// The Stage votes tallied by amplitudes, in a Run with Souls (spec 7.7).
    pub votes: VoteTally,
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
pub fn run_cascade_with<D: Domain>(
    domain: &D,
    pool: &ModelPool,
    limits: &CascadeLimits,
    c: &Cascade,
    task: &D::Task,
    seed: u64,
    souls: Option<&RunSouls<'_>>,
) -> CascadeRun<D::Answer> {
    let ctx = Ctx { pool, limits };
    let check = domain.self_check_text();
    let mut usage = Usage::default();
    let mut chosen: Vec<D::Answer> = Vec::new();
    let mut authors: Vec<usize> = Vec::new();
    let mut votes = VoteTally::default();
    // The index of the Stage's first Worker over all Stages.
    let mut first = 0;

    for (k, stage) in c.stages.iter().enumerate() {
        let prompt = domain.solve_prompt(task, &chosen);
        let mut distinct: Vec<D::Answer> = Vec::new();
        // The Workers whose own answer each distinct answer is (spec 7.7).
        let mut by: Vec<Vec<usize>> = Vec::new();

        for (j, w) in stage.workers.iter().enumerate() {
            let s = derive_seed(seed, &format!("stage{k}/worker{j}"));
            let resp = call_worker(ctx, w, prompt.clone(), s, check, &mut usage);
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
        let winner = match distinct.len() {
            0 => None,
            1 => Some(0),
            _ => {
                let (winner, array) = stage_vote(
                    ctx,
                    domain,
                    &stage.workers,
                    task,
                    &distinct,
                    derive_seed(seed, &format!("vote{k}")),
                    &mut usage,
                    phasors.as_deref(),
                );

                if let Some(vote) = &array {
                    votes.count(vote);
                }

                Some(winner)
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
    }
}

/// Blind vote among a stage's distinct answers: every Worker judges every pair. Returns
/// the index of the answer with the most pairwise wins, ties to the lowest index.
///
/// With `phasors`, one per Worker, the same judgments are tallied by amplitudes instead
/// (step 2, spec 7.3 and 7.4): the answer with the largest Π goes on, and the vote comes
/// back with its Array pattern. The prompts, seeds and calls stay as they are.
#[allow(clippy::too_many_arguments)]
fn stage_vote<D: Domain>(
    ctx: Ctx,
    domain: &D,
    workers: &[Worker],
    task: &D::Task,
    answers: &[D::Answer],
    seed: u64,
    usage: &mut Usage,
    phasors: Option<&[Phasor]>,
) -> (usize, Option<ArrayVote>) {
    // The verdicts of each judge for each answer.
    let mut verdicts = vec![vec![0u32; answers.len()]; workers.len()];

    for (x, ax) in answers.iter().enumerate() {
        for (y, ay) in answers.iter().enumerate().skip(x + 1) {
            for (j, w) in workers.iter().enumerate() {
                let s = derive_seed(seed, &format!("{x}/{y}/{j}"));

                match judge_in(ctx, domain, w, task, (ax, ay), s, usage) {
                    Judgment::Verdict(true) => verdicts[j][x] += 1,
                    Judgment::Verdict(false) => verdicts[j][y] += 1,
                    Judgment::NoReply | Judgment::NoVerdict => {}
                }
            }
        }
    }

    match phasors {
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
    /// The Soul and its settings (step 2, spec 12.2); None, the default, is a Run without
    /// Souls. It is left out of the JSON then, so configs and fixed hashes of earlier runs
    /// are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soul: Option<SoulConfig>,
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
}

impl ParentSelection {
    pub fn is_dgm(&self) -> bool {
        matches!(self, ParentSelection::Dgm)
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
            soul: None,
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
    /// The current Soul of each Worker, in Stage order (spec 4.3). It starts as the birth
    /// Soul of the genome, which never changes. None in a Run without Souls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soul_now: Option<Vec<PhaseState>>,
    /// The authors of the answer on each search task (spec 7.7). None without Souls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authors: Option<Vec<Vec<usize>>>,
    /// The tally of the Stage votes at birth, on the search panel (spec 7.7). None without
    /// Souls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub votes: Option<VoteTally>,
}

/// One Worker of one member: the member's id and the Worker's index over its Stages in
/// order.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WorkerRef {
    pub member: String,
    pub worker: usize,
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
        /// Where each Worker's birth Soul came from (step 2, spec 13.2); left out in a Run
        /// without Souls. The genome holds the birth Souls themselves.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        soul_sources: Option<Vec<SoulSource>>,
        /// The authors of the answer on each search task (spec 7.7, 13.2); left out
        /// without Souls.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        authors: Option<Vec<Vec<usize>>>,
        /// The tally of the Stage votes at birth (spec 7.7, 13.2); left out without Souls.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        votes: Option<VoteTally>,
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
    /// Answers on the select panel so far, so a stopped select phase never pays twice.
    /// Empty when absent. The empty map is named directly: a plain `default` would make
    /// serde require `A: Default`, which a domain's answer type need not have.
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
                        || m.soul_now.is_some()
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
            Some(_) => {
                let n = b.domain.families().len();
                let tasks = b.search.len();

                for m in &snap.archive {
                    check_souls(m, n, tasks).map_err(|why| {
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

    /// The tasks of the search panel.
    pub fn search_panel(&self) -> &[D::Task] {
        &self.search
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
        }

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

        for d in drafts {
            let id = d.cascade.id();
            // Calls take their seeds from the genes (spec 15): members with the same genes
            // and other Souls make the same calls.
            let gene_id = self.gene_key(&id, &d.cascade);
            let tag = format!("search/{gene_id}");
            // A newborn answers with its birth Souls (spec 8.4).
            let souls = d.soul_sources.as_ref().map(|_| birth_souls(&d.cascade));
            let (runs, usage) =
                self.evaluate(&d.cascade, souls.as_deref(), &self.search, &tag, stop);

            calls += usage.calls;

            let Some(runs) = runs else {
                self.calls = calls;

                return Ok((Status::Done("stopped".to_string()), Vec::new()));
            };
            let mut answers = Vec::with_capacity(runs.len());
            let mut authors = Vec::with_capacity(runs.len());
            let mut votes = VoteTally::default();

            // In panel order, so the sums do not depend on threads.
            for r in runs {
                votes.add(&r.votes);
                authors.push(r.authors);
                answers.push(r.answer);
            }

            let souled = souls.is_some();
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
                soul_sources: d.soul_sources.clone(),
                authors: authors.clone(),
                votes,
            });

            // The current Souls start as the birth Souls (spec 4.3).
            let soul_now = d.soul_sources.as_ref().map(|_| birth_souls(&d.cascade));

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
                soul_now,
                authors,
                votes,
            });
        }

        // Each new Cascade meets up to peer_opponents unrelated members on puzzles where
        // their answers differ (it may meet none), and new members are the preferred
        // judges (Q12).
        let new: Vec<usize> = (first_new..archive.len()).collect();
        let (new_comparisons, peer_usage) = self.peer_round(&archive, &new, g, &mut rng, stop);

        calls += peer_usage.calls;
        generation_usage.add(&peer_usage);

        let Some(new_comparisons) = new_comparisons else {
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

        // Commit. The archive changed, so its select phase is still to come.
        self.archive = archive;
        self.gene_ids = gene_ids;
        self.comparisons = comparisons;
        self.calls = calls;
        self.mutator_answers += answers_used;
        self.next_generation = g + 1;
        self.finished = false;

        // The observables of the committed state, with the weights the next generation
        // draws its parents by.
        let weights = self.selection_weights();

        events.push(Event::State(observe(&ObserveInput {
            generation: g,
            archive: &self.archive,
            first_new,
            comparisons: &self.comparisons,
            selection_weights: &weights,
            generation_usage,
            tasks: self.search.len(),
        })));

        Ok((Status::Continue, events))
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
                // A member votes with its current Souls (spec 8.4, 8.6).
                let m = &self.archive[i];
                let (runs, usage) =
                    self.run_panel(&m.cascade, m.soul_now.as_deref(), &tasks, &tag, stop);
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
    /// still counts every call that was made.
    fn evaluate(
        &self,
        c: &Cascade,
        souls: Option<&[PhaseState]>,
        panel: &[D::Task],
        tag: &str,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> (Option<Vec<CascadeRun<D::Answer>>>, Usage) {
        let tasks: Vec<&D::Task> = panel.iter().collect();
        let (runs, usage) = self.run_panel(c, souls, &tasks, tag, stop);

        (runs.into_iter().collect(), usage)
    }

    /// Run `c` on every task of `panel`, in parallel. An entry is None when `stop` fired
    /// before that task started; a started task always finishes, and the usage counts
    /// every call made. Every call's seed comes from `tag` (which names the Cascade) and
    /// the task id, so the result depends neither on thread count nor on when, or with
    /// which other tasks, the Cascade runs. In a Run with Souls, `souls` holds the Souls
    /// its Workers vote with, frozen for the generation, and each task gives its strengths
    /// (spec 7.2, 8.4).
    fn run_panel(
        &self,
        c: &Cascade,
        souls: Option<&[PhaseState]>,
        panel: &[&D::Task],
        tag: &str,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> (Vec<Option<CascadeRun<D::Answer>>>, Usage) {
        let runs: Vec<Option<CascadeRun<D::Answer>>> = panel
            .par_iter()
            .map(|task| {
                if stop() {
                    return None;
                }

                let id = self.domain.task_id(task);
                let seed = derive_seed(self.cfg.seed, &format!("{tag}/{id}"));
                let run = match (&self.cfg.soul, souls) {
                    (Some(soul), Some(souls)) => {
                        let strengths = soul.strengths.of(&self.domain, task);
                        let with = RunSouls {
                            souls,
                            strengths: &strengths,
                            readout: soul.readout,
                            vote: soul.vote,
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
    fn peer_round(
        &self,
        archive: &[Member<D::Answer>],
        new: &[usize],
        g: usize,
        rng: &mut ChaCha8Rng,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> (Option<Vec<Comparison>>, Usage) {
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

        let outcomes: Vec<Option<(Option<Comparison>, Usage)>> = plan
            .par_iter()
            .map(|&(c, o, i, j, seed)| {
                if stop() {
                    return None;
                }

                let task = &self.search[i];
                let (Some(Some(a)), Some(Some(b))) =
                    (archive[c].answers.get(i), archive[o].answers.get(i))
                else {
                    return Some((None, Usage::default()));
                };
                let judge = &archive[j].cascade;
                let mut usage = Usage::default();
                let Some(w) = judge.stages.last().and_then(|s| s.workers.first()) else {
                    return Some((None, usage));
                };
                let ctx = Ctx {
                    pool: self.pool,
                    limits: &self.cfg.limits,
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

                Some((comparison, usage))
            })
            .collect();

        let mut comparisons = Vec::new();
        let mut usage = Usage::default();
        let mut complete = true;

        for out in outcomes {
            match out {
                Some((c, u)) => {
                    usage.add(&u);
                    comparisons.extend(c);
                }
                None => complete = false,
            }
        }

        (complete.then_some(comparisons), usage)
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
        let weights = self.selection_weights();

        let llm_round = self.cfg.llm_every > 0 && g.is_multiple_of(self.cfg.llm_every);

        if llm_round && let Some(mutator) = self.mutator {
            let n = self.cfg.llm_children.min(self.cfg.children_per_generation);
            let mut planned: Vec<(GenRequest, Vec<usize>, Op)> = Vec::new();

            for i in 0..n {
                let Some(a) = draw(&weights, rng) else {
                    break;
                };
                let seed = derive_seed(self.cfg.seed, &format!("llm/{g}/{i}"));
                let partner = if i % 2 == 1 {
                    self.partner(a, &weights, rng)
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

                self.fill_random(&mut random, planned.len(), &mut seen, &weights, rng);

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

        self.fill_random(&mut drafts, 0, &mut seen, &weights, rng);

        Ok(Drafted::Ready(drafts, events, answers_used))
    }

    /// Add random children (mutation and crossover, all distinct, new and admitted by the
    /// gene pool) to `drafts` until they and `others` other children reach
    /// `children_per_generation`, or the draws run out.
    fn fill_random(
        &self,
        drafts: &mut Vec<Draft>,
        others: usize,
        seen: &mut HashSet<String>,
        weights: &[f64],
        rng: &mut ChaCha8Rng,
    ) {
        let mut attempts = 0;

        while others + drafts.len() < self.cfg.children_per_generation
            && attempts < self.cfg.children_per_generation * DRAW_ATTEMPTS
        {
            attempts += 1;

            let Some(a) = draw(weights, rng) else {
                break;
            };
            let parent = &self.archive[a];
            let crossover = rng.random_bool(self.cfg.crossover_rate);
            let mut child: Option<(Cascade, Vec<String>, Op, Sources)> = None;

            // The operators see genes only: they remove the parents' Souls and Coupling
            // strengths first, so their draws are those of a Run without Souls (spec 9.1).
            if crossover && let Some(b) = self.partner(a, weights, rng) {
                let other = &self.archive[b];
                let by_stage = rng.random_bool(0.5);
                let made = if by_stage {
                    stage_crossover_sourced(&parent.cascade, &other.cascade, &self.cfg.limits, rng)
                        .map(|(c, s)| (c, Op::StageCrossover, s))
                } else {
                    section_crossover_sourced(
                        &parent.cascade,
                        &other.cascade,
                        &self.cfg.limits,
                        rng,
                    )
                    .map(|(c, s)| (c, Op::SectionCrossover, s))
                };

                child =
                    made.map(|(c, op, s)| (c, vec![parent.id.clone(), other.id.clone()], op, s));
            }

            if child.is_none() {
                child = mutate_sourced(&parent.cascade, &self.cfg.limits, self.genes.phrases, rng)
                    .map(|(c, op, s)| (c, vec![parent.id.clone()], op, s));
            }

            if let Some((c, parents, op, sources)) = child
                && c.check(&self.cfg.limits).is_ok()
                && (self.genes.admit)(&c).is_ok()
                && seen.insert(c.id())
            {
                drafts.push(Draft {
                    cascade: c,
                    parents,
                    op,
                    sources,
                    soul_sources: None,
                });
            }
        }
    }

    /// Parent selection weights over the archive: DGM weights from fitness and children
    /// counts, or the same weight for every member under uniform selection.
    fn selection_weights(&self) -> Vec<f64> {
        if self.cfg.selection == ParentSelection::Uniform {
            return vec![1.0; self.archive.len()];
        }

        let scores: Vec<f64> = self.archive.iter().map(|m| m.fitness).collect();
        let children: Vec<usize> = self.archive.iter().map(|m| m.children).collect();

        dgm_weights(&scores, &children, self.cfg.dgm_lambda, self.cfg.dgm_alpha0)
    }

    /// A crossover partner for member `a`: draw a few by selection weight and take the
    /// one the peers rate highest (Q2: peer judgment decides who mates with whom). Under
    /// uniform selection, the first draw other than `a`, so ratings play no part.
    fn partner(&self, a: usize, weights: &[f64], rng: &mut ChaCha8Rng) -> Option<usize> {
        let candidates = match self.cfg.selection {
            ParentSelection::Dgm => PARTNER_CANDIDATES,
            ParentSelection::Uniform => 1,
        };
        let mut best: Option<usize> = None;

        for _ in 0..candidates {
            for _ in 0..DRAW_ATTEMPTS {
                let b = draw(weights, rng)?;

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
        let grade = domain.grade(task, a)?;

        if !(0.0..=1.0).contains(&grade.value) {
            return Err(BrutError::Integrity(format!(
                "task {} was graded {}, outside [0, 1]",
                domain.task_id(task),
                grade.value
            )));
        }

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

/// Why member `m` breaks the rule of a Run with Souls of `n` phases (spec 4.4), if it does:
/// every Worker has a Soul of exactly `n` phases and a Coupling strength of at most 1000,
/// and the member records where each Soul came from, each current Soul, the authors on
/// each of the `tasks` search tasks and its tally of votes.
fn check_souls<A>(m: &Member<A>, n: usize, tasks: usize) -> std::result::Result<(), String> {
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

    match &m.soul_now {
        Some(now) if now.len() == workers && now.iter().all(|p| p.len() == n) => {}
        _ => {
            return Err(format!(
                "it does not hold a current Soul of {n} phases for each of its {workers} Workers"
            ));
        }
    }

    if m.authors.as_ref().map(Vec::len) != Some(tasks) || m.votes.is_none() {
        return Err(format!(
            "it does not record the authors on each of the {tasks} search tasks and its votes"
        ));
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
    use crate::cascade::circuit_genes::{CIRCUIT_BRIEF, circuit_pool};
    use crate::cascade::designer::SimDesigner;
    use crate::cascade::domain::{CircuitConfig, Grade};
    use crate::cascade::genome::{Section, Stage, Traits, seed_cascades, seeds_for};
    use crate::cascade::puzzle::generate_panel;
    use crate::cascade::sim::SimModel;
    use crate::cascade::soul::{
        Inheritance, SoulConfig, SoulDraws, Strengths, new_soul_chance, souletics,
    };
    use crate::cascade::variation::{GRID_BRIEF, MutatorBrief};
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

                assert_eq!(b.partner(0, &weights, &mut rng), first);
            }
        }

        // Under DGM selection the ratings decide between the drawn partners.
        let mut dgm = Breeder::new(small_config(19), &pool, None).expect("breeder");

        dgm.step(&never).expect("gen 0");

        let weights = dgm.selection_weights();
        let mut follows_ratings = false;

        for seed in 0..32u64 {
            let mut picks = Vec::new();

            for flip in [1.0, -1.0] {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);

                for (i, m) in dgm.archive.iter_mut().enumerate() {
                    m.rating = flip * i as f64;
                }

                picks.push(dgm.partner(0, &weights, &mut rng));
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
            soul_now: None,
            authors: None,
            votes: None,
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
        let pool = sim_pool();
        let never = || false;
        let cfg = with_souls(
            BreedConfig {
                generations: 5,
                ..small_config(103)
            },
            SoulConfig::arm_s(),
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

        // The children born after the resume, as their Born events record them.
        let children = |snap: Snapshot| {
            let mut b = Breeder::resume(cfg.clone(), &pool, None, snap).expect("resume");
            let mut born = Vec::new();

            while let (Status::Continue, events) = b.step(&never).expect("step") {
                born.extend(
                    events
                        .into_iter()
                        .filter(|e| matches!(e, Event::Born { .. })),
                );
            }

            born
        };
        let straight = children(snap.clone());

        assert!(!straight.is_empty());
        assert_ne!(snap.archive, moved.archive);
        assert_eq!(
            children(moved),
            straight,
            "the current Souls pass on nothing"
        );
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

        // The current Souls must fit as well.
        let mut tampered = snap.clone();

        tampered.archive[2]
            .soul_now
            .as_mut()
            .expect("current Souls")[0]
            .0
            .pop();

        let reason = invalid(Breeder::resume(cfg, &pool, None, tampered), "current Soul");

        assert!(reason.contains(&snap.archive[2].id), "{reason}");
    }

    /// A backend that records every request as JSON before its model answers it.
    struct Recording {
        model: Box<dyn ModelBackend>,
        log: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
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
            }),
        );
        pool.insert(
            "sim:large",
            Box::new(Recording {
                model: large,
                log: log.clone(),
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
        // new Souls of their own origin; the current Souls start as the birth Souls.
        let archive = straight.archive();
        let born: Vec<&Option<Vec<SoulSource>>> = events_straight
            .iter()
            .filter_map(|e| match e {
                Event::Born { soul_sources, .. } => Some(soul_sources),
                _ => None,
            })
            .collect();

        assert_eq!(born.len(), archive.len());

        for (m, logged) in archive.iter().zip(born) {
            assert_eq!(logged, &m.soul_sources);
            assert_eq!(records(m).len(), m.cascade.worker_count());

            let now: Vec<Vec<u16>> = m
                .soul_now
                .as_ref()
                .expect("current Souls")
                .iter()
                .map(|p| p.0.clone())
                .collect();

            assert_eq!(now, birth_phases(&m.cascade));

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
            soul_now: None,
            authors: None,
            votes: None,
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
        // tallies, as its Born event does: what its own run gives on each task.
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
            let souls = birth_souls(&m.cascade);
            let mut tally = VoteTally::default();
            let mut authors = Vec::new();

            for t in &panel {
                let with = RunSouls {
                    souls: &souls,
                    strengths: &strengths,
                    readout: Readout::Complex,
                    vote: true,
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
}
