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
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, HashMap, HashSet};

use rand::seq::{IndexedRandom, SliceRandom};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::backend::{GenRequest, GenResponse, ModelBackend, StopReason, approx_tokens};
use crate::cascade::domain::{Domain, DomainConfig, GenePool, GridDomain};
use crate::cascade::genome::{Cascade, CascadeLimits, Worker, adapt_seeds};
use crate::cascade::observe::{ObserveInput, UniverseState, observe};
use crate::cascade::puzzle::{Grid, Puzzle, SELF_CHECK_TEXT, Verdict, parse_verdict};
use crate::cascade::variation::{
    Op, llm_crossover_request_with, llm_mutate_request_with, mutate_with, parse_llm_cascade,
    section_crossover, stage_crossover,
};
use crate::util::{BrutError, Result, derive_seed, sha256_hex};

/// Draws a parent or a partner gets before it gives up (each draw is cheap).
const DRAW_ATTEMPTS: usize = 16;

/// Partners drawn per crossover; the one with the best peer rating mates (Q2).
const PARTNER_CANDIDATES: usize = 3;

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
    let cost = ctx.limits.model(&w.model).map(|m| m.cost_per_call).unwrap_or(0.0);

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
    c.stages.iter().map(|s| stage_call_bound(s.workers.len())).sum()
}

/// The result of one Cascade on one task.
#[derive(Clone, Debug, PartialEq)]
pub struct CascadeRun<A = Grid> {
    /// The last answer the Cascade chose (the residual path keeps an earlier one when a
    /// later stage produces nothing usable).
    pub answer: Option<A>,
    pub usage: Usage,
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
    let ctx = Ctx { pool, limits };
    let check = domain.self_check_text();
    let mut usage = Usage::default();
    let mut chosen: Vec<D::Answer> = Vec::new();

    for (k, stage) in c.stages.iter().enumerate() {
        let prompt = domain.solve_prompt(task, &chosen);
        let mut distinct: Vec<D::Answer> = Vec::new();

        for (j, w) in stage.workers.iter().enumerate() {
            let s = derive_seed(seed, &format!("stage{k}/worker{j}"));
            let resp = call_worker(ctx, w, prompt.clone(), s, check, &mut usage);
            let answer = resp
                .filter(|r| r.stop_reason == StopReason::Stop)
                .and_then(|r| domain.parse_answer(task, &r.text));

            if let Some(a) = answer
                && !distinct.contains(&a)
            {
                distinct.push(a);
            }
        }

        let winner = match distinct.len() {
            0 => None,
            1 => Some(0),
            _ => Some(stage_vote(
                ctx,
                domain,
                &stage.workers,
                task,
                &distinct,
                derive_seed(seed, &format!("vote{k}")),
                &mut usage,
            )),
        };

        if let Some(i) = winner
            && let Some(a) = distinct.get(i)
        {
            chosen.push(a.clone());
        }
    }

    CascadeRun {
        answer: chosen.last().cloned(),
        usage,
    }
}

/// Blind vote among a stage's distinct answers: every Worker judges every pair. Returns
/// the index of the answer with the most pairwise wins, ties to the lowest index.
fn stage_vote<D: Domain>(
    ctx: Ctx,
    domain: &D,
    workers: &[Worker],
    task: &D::Task,
    answers: &[D::Answer],
    seed: u64,
    usage: &mut Usage,
) -> usize {
    let mut wins = vec![0usize; answers.len()];

    for (x, ax) in answers.iter().enumerate() {
        for (y, ay) in answers.iter().enumerate().skip(x + 1) {
            for (j, w) in workers.iter().enumerate() {
                let s = derive_seed(seed, &format!("{x}/{y}/{j}"));

                match judge_in(ctx, domain, w, task, (ax, ay), s, usage) {
                    Judgment::Verdict(true) => wins[x] += 1,
                    Judgment::Verdict(false) => wins[y] += 1,
                    Judgment::NoReply | Judgment::NoVerdict => {}
                }
            }
        }
    }

    let mut best = 0;

    for (i, &n) in wins.iter().enumerate() {
        if n > wins[best] {
            best = i;
        }
    }

    best
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

        if !(0.0..=1.0).contains(&self.crossover_rate) || !(0.0..=1.0).contains(&self.judge_weight) {
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
        if self.limits.models.iter().any(|m| !(0.0..=1.0).contains(&m.cost_per_call)) {
            return bad("every model's cost_per_call must lie in [0, 1]");
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
        adapt_seeds((genes.seeds)(), &cfg.limits)?;

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
                "resume refused: settings other than generations and max_calls changed"
                    .to_string(),
            ));
        }

        let mut b = Breeder::start(cfg, genes, pool, mutator)?;

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
                    cascade: c,
                    parents: Vec::new(),
                    op: Op::Seed,
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

        for d in drafts {
            let id = d.cascade.id();
            let tag = format!("search/{id}");
            let (answers, usage) = self.evaluate(&d.cascade, &self.search, &tag, stop);

            calls += usage.calls;

            let Some(answers) = answers else {
                self.calls = calls;

                return Ok((Status::Done("stopped".to_string()), Vec::new()));
            };
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
            });

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

        let best = rank(&archive).first().map(|&i| archive[i].clone());

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
        let order = rank(&self.archive);
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

                let tag = format!("select/{id}");
                let tasks: Vec<&D::Task> = missing.iter().map(|&j| &self.select[j]).collect();
                let (answers, usage) = self.run_panel(&self.archive[i].cascade, &tasks, &tag, stop);
                let mut complete = true;
                let got = self.select_answers.entry(id.clone()).or_default();

                for (task, answer) in tasks.iter().zip(answers) {
                    match answer {
                        Some(a) => {
                            got.insert(self.domain.task_id(task).to_string(), a);
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
        self.select_picks(&rank(&self.archive))
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

    /// Answers and usage of `c` on `panel`. The answers are None when `stop` fired before
    /// some task; the usage still counts every call that was made.
    fn evaluate(
        &self,
        c: &Cascade,
        panel: &[D::Task],
        tag: &str,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> (Option<Vec<Option<D::Answer>>>, Usage) {
        let tasks: Vec<&D::Task> = panel.iter().collect();
        let (answers, usage) = self.run_panel(c, &tasks, tag, stop);

        (answers.into_iter().collect(), usage)
    }

    /// Run `c` on every task of `panel`, in parallel. An entry is None when `stop` fired
    /// before that task started; a started task always finishes, and the usage counts
    /// every call made. Every call's seed comes from `tag` (which names the Cascade) and
    /// the task id, so the result depends neither on thread count nor on when, or with
    /// which other tasks, the Cascade runs.
    fn run_panel(
        &self,
        c: &Cascade,
        panel: &[&D::Task],
        tag: &str,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> (Vec<Option<Option<D::Answer>>>, Usage) {
        let runs: Vec<Option<CascadeRun<D::Answer>>> = panel
            .par_iter()
            .map(|task| {
                if stop() {
                    return None;
                }

                let id = self.domain.task_id(task);
                let seed = derive_seed(self.cfg.seed, &format!("{tag}/{id}"));
                let run = run_cascade_in(&self.domain, self.pool, &self.cfg.limits, c, task, seed);

                Some(run)
            })
            .collect();

        let mut usage = Usage::default();
        let answers = runs
            .into_iter()
            .map(|r| {
                r.map(|r| {
                    usage.add(&r.usage);

                    r.answer
                })
            })
            .collect();

        (answers, usage)
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
                    .filter(|&i| match (&archive[c].answers.get(i), &archive[o].answers.get(i)) {
                        (Some(Some(a)), Some(Some(b))) => a != b,
                        _ => false,
                    })
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
                let (Some(Some(a)), Some(Some(b))) = (archive[c].answers.get(i), archive[o].answers.get(i))
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
        let mut seen: HashSet<String> = self.archive.iter().map(|m| m.id.clone()).collect();
        let weights = self.selection_weights();

        let llm_round = self.cfg.llm_every > 0 && g.is_multiple_of(self.cfg.llm_every);

        if llm_round && let Some(mutator) = self.mutator {
            let n = self.cfg.llm_children.min(self.cfg.children_per_generation);
            let mut planned: Vec<(GenRequest, Vec<String>, Op)> = Vec::new();

            for i in 0..n {
                let Some(a) = draw(&weights, rng) else {
                    break;
                };
                let seed = derive_seed(self.cfg.seed, &format!("llm/{g}/{i}"));
                let partner = if i % 2 == 1 { self.partner(a, &weights, rng) } else { None };
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

                        planned.push((req, vec![pa.id.clone(), pb.id.clone()], Op::LlmCrossover));
                    }
                    None => {
                        let req = llm_mutate_request_with(
                            self.genes.brief,
                            &pa.cascade,
                            pa.fitness,
                            &self.cfg.limits,
                            seed,
                        );

                        planned.push((req, vec![pa.id.clone()], Op::LlmMutate));
                    }
                }
            }

            let answered: Vec<Option<GenResponse>> =
                planned.iter().map(|(req, _, _)| mutator.answer(req)).collect();

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
                    Some(r) if !r.is_usable() => Err(format!("unusable answer ({:?})", r.stop_reason)),
                    Some(r) => parse_llm_cascade(&r.text, &self.cfg.limits).map_err(|e| e.to_string()),
                };
                // The gene pool's admission rule holds for every new Cascade.
                let parsed = parsed.and_then(|c| match (self.genes.admit)(&c) {
                    Ok(()) => Ok(c),
                    Err(e) => Err(format!("not admitted: {e}")),
                });
                let (accepted, reason) = match parsed {
                    Ok(c) if seen.contains(&c.id()) => (false, "duplicate of an archived Cascade".to_string()),
                    Ok(c) => {
                        seen.insert(c.id());
                        drafts.push(Draft {
                            cascade: c,
                            parents: parents.clone(),
                            op,
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
            let mut child: Option<(Cascade, Vec<String>, Op)> = None;

            if crossover && let Some(b) = self.partner(a, weights, rng) {
                let other = &self.archive[b];
                let by_stage = rng.random_bool(0.5);
                let made = if by_stage {
                    stage_crossover(&parent.cascade, &other.cascade, &self.cfg.limits, rng)
                        .map(|c| (c, Op::StageCrossover))
                } else {
                    section_crossover(&parent.cascade, &other.cascade, &self.cfg.limits, rng)
                        .map(|c| (c, Op::SectionCrossover))
                };

                child = made.map(|(c, op)| (c, vec![parent.id.clone(), other.id.clone()], op));
            }

            if child.is_none() {
                child = mutate_with(&parent.cascade, &self.cfg.limits, self.genes.phrases, rng)
                    .map(|(c, op)| (c, vec![parent.id.clone()], op));
            }

            if let Some((c, parents, op)) = child
                && c.check(&self.cfg.limits).is_ok()
                && (self.genes.admit)(&c).is_ok()
                && seen.insert(c.id())
            {
                drafts.push(Draft {
                    cascade: c,
                    parents,
                    op,
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
                    let better = best.is_none_or(|x| self.archive[b].rating > self.archive[x].rating);

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
            let loser = if winner == &c.first { &c.second } else { &c.first };
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

/// Archive indices best first: fitness, then peer rating (tie-break, Q13), then id.
fn rank<A>(archive: &[Member<A>]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..archive.len()).collect();

    order.sort_by(|&a, &b| {
        let (x, y) = (&archive[a], &archive[b]);

        y.fitness
            .total_cmp(&x.fitness)
            .then(y.rating.total_cmp(&x.rating))
            .then(x.id.cmp(&y.id))
    });

    order
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
    use crate::cascade::domain::Grade;
    use crate::cascade::genome::{Section, Stage, Traits, seed_cascades, seeds_for};
    use crate::cascade::puzzle::generate_panel;
    use crate::cascade::sim::SimModel;
    use crate::cascade::variation::MutatorBrief;

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

        assert!(none.iter().all(|x| x.abs() < 1e-9), "zero weight changes nothing");
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

        c.stages[0].workers.push(seeds[1].1.stages[0].workers[0].clone());
        c.stages[0].workers.push(seeds[2].1.stages[0].workers[0].clone());

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
            assert!(c.stages.iter().flat_map(|s| &s.workers).all(|w| w.model == "sim:large"));
            assert!(c.stages.iter().flat_map(|s| &s.workers).all(|w| w.traits.think_tokens == 0));
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

        book.insert(&request_key(&requests[0]), format!("Here you go:\n```json\n{json}\n```"));
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
        assert!(resumed
            .archive()
            .iter()
            .any(|m| m.id == child.id() && m.op == Op::LlmMutate));
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
            .filter(|e| matches!(e, Event::LlmExchange { accepted: false, .. }))
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
        assert!(reserved.reserved >= b.calls_used(), "the worst case covers what was spent");

        let mut resumed = Breeder::resume(small_config(23), &pool, None, reserved.clone()).expect("resume");

        assert_eq!(resumed.calls_used(), reserved.reserved, "a crash counts the reservation");

        while let (Status::Continue, _) = resumed.step(&never).expect("step") {}

        assert_eq!(resumed.archive(), straight.archive(), "the repeated generation is the same");
        assert_eq!(resumed.calls_used(), straight.calls_used() + reserved.reserved);
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

            assert!(b.finish(&stop_later, "all generations done").expect("finish").is_none());
        }

        let (again, again_events) = b
            .finish(&never, "all generations done")
            .expect("finish")
            .expect("not stopped");

        assert_eq!(b.calls_used() - before, select_calls, "every puzzle was scored once");
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
            refused.as_deref().is_some_and(|e| e.contains("another domain")),
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
            let i = b.search.iter().position(|p| p.id == c.puzzle).expect("search puzzle");
            let right = |id: &str| {
                let m = b.archive().iter().find(|m| m.id == id).expect("member");

                m.answers[i].as_ref() == Some(&b.search[i].test_output)
            };

            assert_ne!(right(&c.first), right(&c.second), "{c:?}");
        }

        assert!(b.archive().iter().any(|m| m.judge_wrong > 0 && m.judge_skill() < 0.5));
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
                let new: Vec<usize> = (0..prefix).filter(|&i| archive[i].generation == g).collect();
                let mut assigned = vec![0usize; prefix];

                for c in b.comparisons().iter().filter(|c| c.generation == g) {
                    let (x, y) = (index[c.first.as_str()], index[c.second.as_str()]);
                    let j = index[c.judge.as_str()];
                    let fair = |k: usize| k != x && k != y && !kin.related(k, x) && !kin.related(k, y);
                    let mut candidates: Vec<usize> = new.iter().copied().filter(|&k| fair(k)).collect();

                    if candidates.is_empty() {
                        candidates = (0..prefix).filter(|&k| fair(k)).collect();
                    }

                    let fewest = candidates.iter().map(|&k| assigned[k]).min().expect("a fair judge");

                    assert!(fair(j), "seed {seed} gen {g}: judge {j} is related");
                    assert_eq!(assigned[j], fewest, "seed {seed} gen {g}: judge {j} was not least used");
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
                text: if first_is_right { "VERDICT A" } else { "VERDICT B" }.to_string(),
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

        pool.insert("sim:small", Box::new(Oracle { right: right.clone() }));

        let limits = CascadeLimits::default();
        let ctx = Ctx {
            pool: &pool,
            limits: &limits,
        };
        let w = seed_cascades()[0].1.stages[0].workers[0].clone();
        let mut orders = std::collections::BTreeSet::new();

        for seed in 0..32u64 {
            let mut usage = Usage::default();

            assert_eq!(judge_with(ctx, &w, &p, &right, &wrong, seed, &mut usage), Judgment::Verdict(true));
            assert_eq!(judge_with(ctx, &w, &p, &wrong, &right, seed, &mut usage), Judgment::Verdict(false));
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
        let picked = b.select_picks(&rank(&b.archive));
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
            assert_eq!(s.reserved, cascade_call_bound(&b.archive[i].cascade) * b.select.len() as u64);
        }

        // A crash inside the first Cascade costs its reservation once more; the resumed
        // select phase still fits and gives the same finalists.
        let first = stored[0].clone();
        let mut resumed = Breeder::resume(small_config(41), &pool, None, first.clone()).expect("resume");

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
        let threshold = (150..600).find(|&cap| pauses(cap)).expect("a cap that pauses");

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

        let mut resumed = Breeder::resume(cfg(threshold), &pool, Some(&book), b.snapshot()).expect("resume");
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

        loop {
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
            committed += 1;
        }

        assert_eq!(committed, small_config(47).generations + 1);

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
    /// 5000, "exact" 2n, "over" 2n + 2, anything else n. A judge whose prompt says
    /// "careful" picks the better answer; any other picks the first one shown.
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
            let text = if req.prompt.starts_with("WHICH") {
                let (a, b) = (numbers[1], numbers[2]);
                let first = !req.system.contains("careful") || doubled(n, a) >= doubled(n, b);

                if first { "VERDICT A" } else { "VERDICT B" }.to_string()
            } else if req.system.contains("overflow") {
                "ANSWER 5000".to_string()
            } else if req.system.contains("exact") {
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
}
