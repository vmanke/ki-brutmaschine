//! The compute-equal majority-vote control of a finished grid breeding run: does breeding
//! beat sampling one seed Cascade several times and keeping its most frequent answer?
//! Self-consistency (Wang et al. 2022) is that sampler; a cost-equal comparison gives it as
//! many model calls as the bred Cascade spends (arXiv 2606.13003).
//!
//! For a run directory it reads `config.json` and `snapshot.json`, regenerates the select
//! panel exactly as the arena does (`panel(derive_seed(seed, "panel/select"), n)`), and runs
//! one Cascade N times on every select puzzle, each time with its own request seed. The
//! Cascade is by default the seed finalist with the best search fitness: chosen by its
//! search-panel standing alone, never by a select result. N is the bred champion's average
//! Worker calls per search puzzle divided by the Cascade's own, rounded, at least 1. The
//! champion is the bred (non-seed) finalist with the best search fitness. The most frequent
//! parsed grid wins, ties to the grid seen first (lowest sample index), and it is scored
//! exactly against the hidden test output, as the arena scores.
//!
//! Every sample goes through the core's `run_cascade`, so prompts, Traits, answer parsing
//! and a stage's blind vote are the arena's own. Real models are wrapped in `Recorded`, so
//! every call is filed under `<run>/vote-control/` and a repeated control replays it. The
//! seed of sample k does not depend on N, so a larger N reuses the calls of a smaller one.
//! A control with a failed call is incomplete (`VoteReport::complete`) and never becomes
//! the result of record.
//!
//! The comparison is fair only when the run's own calls worked, since its select answers
//! are the baseline: `run_health` flags a run that filed failed calls, and a vLLM run from
//! before the 63-bit seed fix (`SEED_FIX`), which lost about half its calls unfiled.
#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use brutmaschine::backend::{GenRequest, GenResponse, ModelBackend};
use brutmaschine::cascade::arena::{
    BreedConfig, CascadeRun, Member, ModelPool, Snapshot, cascade_call_bound, run_cascade,
};
use brutmaschine::cascade::domain::{Domain, GridDomain};
use brutmaschine::cascade::genome::{Cascade, adapt_seeds, seed_cascades};
use brutmaschine::cascade::puzzle::{Grid, Puzzle};
use brutmaschine::cascade::sim::SimModel;
use brutmaschine::cascade::variation::Op;
use brutmaschine::util::{BrutError, Result, derive_seed};

use crate::vllm::{self, VllmBackend};
use crate::{OllamaBackend, Recorded, TAG_PREFIX, wire_seed};

/// The run's settings, as brut-breed writes them.
pub const CONFIG: &str = "config.json";

/// The run's checkpoint: the snapshot plus where the log stood.
pub const SNAPSHOT: &str = "snapshot.json";

/// The run's event log; its first line names the select panel.
pub const EVENTS: &str = "events.jsonl";

/// The result, beside the run's report. Only a complete result goes there.
pub const OUTPUT: &str = "vote-control.json";

/// The result of a control with failed calls, which never replaces `OUTPUT`.
pub const INCOMPLETE: &str = "vote-control/incomplete.json";

/// The record of the control's model calls, apart from the run's own
/// `llm/worker-calls.jsonl`, so the run's call counts stay as they were. Failed calls go to
/// `worker-errors.jsonl` beside it.
pub const RECORD: &str = "vote-control/worker-calls.jsonl";

/// The run's own record of its model calls, as brut-breed-ollama files it.
pub const RUN_RECORD: &str = "llm/worker-calls.jsonl";

/// The run's own failed calls, filed beside its record (brut-breed-ollama from #40 on).
pub const RUN_ERRORS: &str = "llm/worker-errors.jsonl";

/// The commit that masked request seeds to 63 bits (#40). Before it, vLLM refused every
/// call with a seed above i64::MAX, about half of all calls, and nothing filed the failure.
pub const SEED_FIX: &str = "d79c93f";

/// vLLM records a run must hold before `run_health` reads the lack of any seed above
/// i64::MAX as the mark of a run before the fix. The core draws seeds uniformly from u64,
/// so a run after the fix shows none in this many with probability 2^-32.
pub const SEED_PROBE_RECORDS: usize = 32;

/// A finished run as recorded: its settings, its last snapshot, and whether its own calls
/// were healthy.
#[derive(Clone, Debug)]
pub struct RunDir {
    pub cfg: BreedConfig,
    pub snapshot: Snapshot<Grid>,
    pub health: RunHealth,
}

/// Whether the run's own select answers, the baseline of the control, came from calls
/// that worked. The control's samples run with 63-bit seeds and never fail for that
/// reason, so a run whose own calls failed compares a healthy arm with a damaged one.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunHealth {
    /// Failed Worker calls the run filed in `llm/worker-errors.jsonl`.
    pub run_failed_calls: u64,
    /// The run allows a vLLM model and its record holds at least `SEED_PROBE_RECORDS`
    /// vLLM calls, none with a seed above i64::MAX: it predates the seed fix.
    pub predates_seed_fix: bool,
}

impl RunHealth {
    /// Why the run's own answers are no fair baseline, or None when they are.
    pub fn problem(&self) -> Option<String> {
        let mut why = Vec::new();

        if self.run_failed_calls > 0 {
            why.push(format!("{} of its own Worker calls failed ({RUN_ERRORS})", self.run_failed_calls));
        }

        if self.predates_seed_fix {
            why.push(format!(
                "it predates the 63-bit seed fix ({SEED_FIX}): no vLLM seed in {RUN_RECORD} lies above i64::MAX, so about half its calls failed without a record"
            ));
        }

        (!why.is_empty()).then(|| why.join("; "))
    }
}

/// The health of the run in `dir` under its settings `cfg`: (a) the failed calls it filed,
/// one per non-empty line of `llm/worker-errors.jsonl`; (b) whether it predates the seed
/// fix, see `RunHealth::predates_seed_fix`. A missing file counts as none.
pub fn run_health(dir: &Path, cfg: &BreedConfig) -> Result<RunHealth> {
    let errors = dir.join(RUN_ERRORS);
    let run_failed_calls = match File::open(&errors) {
        Ok(f) => {
            let mut n = 0;

            for line in BufReader::new(f).lines() {
                if !line.map_err(|e| invalid(format!("{}: {e}", errors.display())))?.trim().is_empty() {
                    n += 1;
                }
            }

            n
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
        Err(e) => return Err(invalid(format!("{}: {e}", errors.display()))),
    };
    let allows_vllm = cfg.limits.models.iter().any(|m| m.tag.starts_with(vllm::TAG_PREFIX));

    Ok(RunHealth {
        run_failed_calls,
        predates_seed_fix: allows_vllm && no_high_vllm_seed(&dir.join(RUN_RECORD))?,
    })
}

/// True when `record` holds at least `SEED_PROBE_RECORDS` vLLM calls and none of them has
/// a request seed above i64::MAX. It stops at the first such seed. A last line without its
/// newline is a write in progress and is not read.
fn no_high_vllm_seed(record: &Path) -> Result<bool> {
    #[derive(Deserialize)]
    struct Probe {
        tag: String,
        request: ProbeRequest,
    }

    #[derive(Deserialize)]
    struct ProbeRequest {
        seed: u64,
    }

    let file = match File::open(record) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(invalid(format!("{}: {e}", record.display()))),
    };
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let (mut number, mut vllm) = (0usize, 0usize);

    loop {
        line.clear();
        number += 1;

        let read = reader
            .read_line(&mut line)
            .map_err(|e| invalid(format!("{}: {e}", record.display())))?;

        if read == 0 || !line.ends_with('\n') {
            break;
        }

        if line.trim().is_empty() {
            continue;
        }

        let rec: Probe = serde_json::from_str(&line)
            .map_err(|e| integrity(format!("{} line {number}: {e}", record.display())))?;

        if rec.tag.starts_with(vllm::TAG_PREFIX) {
            if rec.request.seed > i64::MAX as u64 {
                return Ok(false);
            }

            vllm += 1;
        }
    }

    Ok(vllm >= SEED_PROBE_RECORDS)
}

fn invalid(m: String) -> BrutError {
    BrutError::Invalid(format!("vote-control: {m}"))
}

fn integrity(m: String) -> BrutError {
    BrutError::Integrity(format!("vote-control: {m}"))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).map_err(|e| invalid(format!("{}: {e}", path.display())))?;

    Ok(serde_json::from_str(&text)?)
}

/// The run in `dir`: `config.json` and `snapshot.json` (a brut-breed checkpoint, or a bare
/// snapshot), and its `run_health`. Refused for a run on another domain, and when the
/// snapshot was taken under other settings than `config.json` holds.
pub fn load_run(dir: &Path) -> Result<RunDir> {
    let cfg: BreedConfig = read_json(&dir.join(CONFIG))?;

    if !cfg.domain.is_grid() {
        return Err(invalid(format!(
            "{} is not a grid run; the majority vote compares grids only",
            dir.display()
        )));
    }

    let stored: serde_json::Value = read_json(&dir.join(SNAPSHOT))?;
    let value = match stored.get("snapshot") {
        Some(s) => s.clone(),
        None => stored,
    };
    let snapshot: Snapshot<Grid> = serde_json::from_value(value)?;

    if snapshot.config_hash != cfg.fixed_hash() {
        return Err(integrity(format!(
            "{} does not match the settings the snapshot was taken under",
            dir.join(CONFIG).display()
        )));
    }

    let health = run_health(dir, &cfg)?;

    Ok(RunDir { cfg, snapshot, health })
}

/// The select panel of `cfg`, exactly as the arena makes it.
pub fn select_panel(cfg: &BreedConfig) -> Result<Vec<Puzzle>> {
    let domain = GridDomain::from_config(cfg)?;

    Ok(domain.panel(derive_seed(cfg.seed, "panel/select"), cfg.select_puzzles))
}

/// Error unless the regenerated `panel` is the one the run scored: every select answer in
/// the snapshot names a puzzle of the panel, and in a finished run every finalist has an
/// answer for every puzzle.
pub fn check_panel(panel: &[Puzzle], run: &RunDir) -> Result<()> {
    let snap = &run.snapshot;
    let ids: BTreeSet<&str> = panel.iter().map(|p| p.id.as_str()).collect();

    for (member, got) in &snap.select_answers {
        if let Some(task) = got.keys().find(|t| !ids.contains(t.as_str())) {
            return Err(integrity(format!(
                "member {member} holds a select answer for puzzle {task}, which the regenerated panel lacks"
            )));
        }
    }

    if !snap.finished {
        return Ok(());
    }

    for i in finalists(&snap.archive, run.cfg.finalists) {
        let id = &snap.archive[i].id;
        let got = snap.select_answers.get(id).map_or(0, |g| g.len());

        if got != ids.len() {
            return Err(integrity(format!(
                "finalist {id} has {got} select answers, the regenerated panel {} puzzles",
                ids.len()
            )));
        }
    }

    Ok(())
}

/// Compare `panel` with the select ids of the log's first event (run_start), when the run
/// has a log. Ok(false) when there is none to compare with.
pub fn check_run_start(dir: &Path, panel: &[Puzzle]) -> Result<bool> {
    let path = dir.join(EVENTS);

    if !path.exists() {
        return Ok(false);
    }

    let file = File::open(&path).map_err(BrutError::Io)?;
    let Some(first) = BufReader::new(file).lines().next() else {
        return Ok(false);
    };
    let line: serde_json::Value = serde_json::from_str(&first.map_err(BrutError::Io)?)?;
    let event = &line["event"];

    if event["kind"] != "run_start" {
        return Err(integrity(format!("{}: the first event is not run_start", path.display())));
    }

    let logged: Vec<String> = serde_json::from_value(event["select"].clone())?;
    let ours: Vec<&str> = panel.iter().map(|p| p.id.as_str()).collect();

    if logged != ours {
        return Err(integrity(format!(
            "{}: the logged select panel differs from the regenerated one",
            path.display()
        )));
    }

    Ok(true)
}

/// The archive in the arena's order: fitness, then peer rating, best first; ties by id.
pub fn rank<A>(archive: &[Member<A>]) -> Vec<usize> {
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

/// The seed finalist with the best search fitness. It reads no select result.
pub fn default_cascade<A>(archive: &[Member<A>]) -> Option<usize> {
    rank(archive).into_iter().find(|&i| archive[i].op == Op::Seed)
}

/// The members the arena's select phase scores, best first: the `n` best bred members by
/// search standing, then every seed.
pub fn finalists<A>(archive: &[Member<A>], n: usize) -> Vec<usize> {
    let order = rank(archive);
    let bred = order.iter().copied().filter(|&i| archive[i].op != Op::Seed).take(n);
    let seeds = order.iter().copied().filter(|&i| archive[i].op == Op::Seed);

    bred.chain(seeds).collect()
}

/// The champion: the bred finalist with the best search fitness. None when the run scores
/// no bred finalist (`finalists` 0, or no bred member).
pub fn champion<A>(archive: &[Member<A>], n: usize) -> Option<usize> {
    finalists(archive, n).into_iter().find(|&i| archive[i].op != Op::Seed)
}

/// Average Worker calls per search puzzle, from the member's usage.
pub fn calls_per_puzzle<A>(m: &Member<A>) -> f64 {
    m.usage.calls as f64 / m.answers.len().max(1) as f64
}

fn tokens_per_puzzle<A>(m: &Member<A>) -> f64 {
    m.usage.completion_tokens as f64 / m.answers.len().max(1) as f64
}

/// Samples per puzzle for a call budget of `budget` calls per puzzle, when one run of the
/// sampled Cascade costs `own` calls: the nearest whole number, halves up, at least 1.
pub fn samples_for(budget: f64, own: f64) -> usize {
    if !(own > 0.0 && budget.is_finite()) {
        return 1;
    }

    ((budget / own).round() as usize).max(1)
}

/// The Cascade seed of sample `k` of the Cascade `cascade_id` on puzzle `task`, masked to
/// the 63 bits a server takes (`wire_seed`). It does not depend on how many samples are
/// drawn, and its tag differs from the arena's ("search/", "select/"), so no sample repeats
/// a call of the run.
pub fn sample_seed(run_seed: u64, cascade_id: &str, k: usize, task: &str) -> u64 {
    wire_seed(derive_seed(run_seed, &format!("vote/{cascade_id}/{k}/{task}")))
}

/// One distinct grid among a puzzle's samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ballot {
    pub grid: Grid,
    pub count: usize,
    /// The sample that gave this grid first.
    pub first: usize,
}

/// The votes of one puzzle: distinct grids in the order first seen, and the samples
/// without a usable grid.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    pub ballots: Vec<Ballot>,
    pub abstained: usize,
}

impl Tally {
    /// Count `answers`, one per sample in sample order; None abstains.
    pub fn of(answers: &[Option<Grid>]) -> Tally {
        let mut t = Tally::default();

        for (k, a) in answers.iter().enumerate() {
            match a {
                None => t.abstained += 1,
                Some(g) => match t.ballots.iter_mut().find(|b| &b.grid == g) {
                    Some(b) => b.count += 1,
                    None => t.ballots.push(Ballot {
                        grid: g.clone(),
                        count: 1,
                        first: k,
                    }),
                },
            }
        }

        t
    }

    /// The majority: the most votes, ties to the grid seen first. None when every sample
    /// abstained.
    pub fn winner(&self) -> Option<&Ballot> {
        let mut best: Option<&Ballot> = None;

        // Ballots are in first-seen order, so only a strictly larger count replaces.
        for b in &self.ballots {
            if best.is_none_or(|w| b.count > w.count) {
                best = Some(b);
            }
        }

        best
    }
}

/// Exactly right: equal to the hidden test output, as the arena grades.
fn right(p: &Puzzle, g: &Grid) -> bool {
    GridDomain.grade(p, g).is_ok_and(|x| x.correct)
}

/// Exact two-sided sign test (McNemar's exact test) on `b` and `c` discordant pairs: the
/// chance of a split at least this uneven when either side wins a discordant pair with
/// probability 1/2. Computed in log space, so it holds for any panel size.
pub fn sign_test(b: usize, c: usize) -> f64 {
    let n = b + c;

    if n == 0 {
        return 1.0;
    }

    let mut ln_term = -(n as f64) * std::f64::consts::LN_2;
    let mut terms = vec![ln_term];

    for i in 0..b.min(c) {
        ln_term += ((n - i) as f64).ln() - ((i + 1) as f64).ln();
        terms.push(ln_term);
    }

    let top = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let tail = top.exp() * terms.iter().map(|t| (t - top).exp()).sum::<f64>();

    (2.0 * tail).min(1.0)
}

/// What the control runs, before any call.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Archive member to sample instead of the default seed finalist.
    pub cascade: Option<String>,
    /// Samples per puzzle instead of the champion's call budget.
    pub samples: Option<usize>,
    /// A temperature for every Worker of the sampled Cascade instead of its own.
    pub temperature: Option<f32>,
    /// Only the first K select puzzles (a smoke test; the result is marked partial).
    pub puzzles: Option<usize>,
}

/// The Cascade that is sampled.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Voted {
    /// The sampled Cascade's id (another than `member` after a temperature override).
    pub id: String,
    /// The archive member it comes from.
    pub member: String,
    /// The hand-written seed's name, when the member is one.
    pub seed_name: Option<String>,
    pub is_seed: bool,
    pub models: Vec<String>,
    pub search_fitness: f64,
    pub search_solved: usize,
    pub search_puzzles: usize,
    pub calls_per_puzzle: f64,
    pub completion_tokens_per_puzzle: f64,
    /// The temperature override, if one was given.
    pub temperature: Option<f32>,
}

/// The bred champion whose call budget the control gets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Champion {
    pub id: String,
    pub generation: usize,
    pub stages: usize,
    pub workers: usize,
    pub search_fitness: f64,
    pub search_solved: usize,
    pub search_puzzles: usize,
    pub calls_per_puzzle: f64,
    pub completion_tokens_per_puzzle: f64,
}

/// The plan of a control: what is sampled, how often, on which puzzles.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub run_seed: u64,
    pub config_hash: String,
    pub generations_completed: usize,
    pub finished: bool,
    pub cascade: Voted,
    pub champion: Option<Champion>,
    pub samples_per_puzzle: usize,
    /// "champion_calls" (the champion's calls per puzzle) or "override" (--samples).
    pub samples_from: String,
    /// Select puzzles scored (all of them unless `partial`).
    pub select_puzzles: usize,
    pub partial: bool,
    /// Upper bound on the calls: puzzles x samples x the Cascade's call bound.
    pub max_calls: u64,
    /// The run's own failed calls (`RunHealth`): the champion's and the sampled member's
    /// select answers may count failures as wrong answers.
    pub run_failed_calls: u64,
    /// The run predates the 63-bit seed fix (`RunHealth`), so about half its calls failed.
    pub predates_seed_fix: bool,
}

/// The plan for `run` under `opts`, and the Cascade to sample.
pub fn plan(run: &RunDir, opts: &Options) -> Result<(Plan, Cascade)> {
    let (cfg, snap) = (&run.cfg, &run.snapshot);
    let idx = match &opts.cascade {
        Some(id) => snap
            .archive
            .iter()
            .position(|m| &m.id == id)
            .ok_or_else(|| invalid(format!("no archive member {id:?}")))?,
        None => default_cascade(&snap.archive)
            .ok_or_else(|| invalid("the archive holds no seed Cascade".to_string()))?,
    };
    let member = &snap.archive[idx];
    let mut voted = member.cascade.clone();

    if let Some(t) = opts.temperature {
        for w in voted.stages.iter_mut().flat_map(|s| s.workers.iter_mut()) {
            w.traits.temperature = t;
        }
    }

    voted.check(&cfg.limits).map_err(|e| invalid(format!("the sampled Cascade: {e}")))?;

    let own = if member.usage.calls > 0 {
        calls_per_puzzle(member)
    } else {
        cascade_call_bound(&voted) as f64
    };
    let champ = champion(&snap.archive, cfg.finalists).map(|i| &snap.archive[i]);
    let (samples, from) = match (opts.samples, champ) {
        (Some(0), _) => return Err(invalid("--samples must be at least 1".to_string())),
        (Some(n), _) => (n, "override"),
        (None, Some(c)) => (samples_for(calls_per_puzzle(c), own), "champion_calls"),
        (None, None) => {
            return Err(invalid(
                "no bred finalist to take the call budget from; give --samples N".to_string(),
            ));
        }
    };
    let puzzles = opts.puzzles.unwrap_or(cfg.select_puzzles);

    if puzzles == 0 || puzzles > cfg.select_puzzles {
        return Err(invalid(format!(
            "--puzzles must lie in 1..={}, got {puzzles}",
            cfg.select_puzzles
        )));
    }

    let seed_name = adapt_seeds(seed_cascades(), &cfg.limits)
        .ok()
        .and_then(|s| s.into_iter().find(|(_, c)| c.id() == member.id).map(|(n, _)| n));
    let models: BTreeSet<String> = voted
        .stages
        .iter()
        .flat_map(|s| &s.workers)
        .map(|w| w.model.clone())
        .collect();
    let plan = Plan {
        run_seed: cfg.seed,
        config_hash: snap.config_hash.clone(),
        generations_completed: snap.next_generation,
        finished: snap.finished,
        cascade: Voted {
            id: voted.id(),
            member: member.id.clone(),
            seed_name,
            is_seed: member.op == Op::Seed,
            models: models.into_iter().collect(),
            search_fitness: member.fitness,
            search_solved: member.solved,
            search_puzzles: member.answers.len(),
            calls_per_puzzle: own,
            completion_tokens_per_puzzle: tokens_per_puzzle(member),
            temperature: opts.temperature,
        },
        champion: champ.map(|c| Champion {
            id: c.id.clone(),
            generation: c.generation,
            stages: c.cascade.stages.len(),
            workers: c.cascade.worker_count(),
            search_fitness: c.fitness,
            search_solved: c.solved,
            search_puzzles: c.answers.len(),
            calls_per_puzzle: calls_per_puzzle(c),
            completion_tokens_per_puzzle: tokens_per_puzzle(c),
        }),
        samples_per_puzzle: samples,
        samples_from: from.to_string(),
        select_puzzles: puzzles,
        partial: puzzles < cfg.select_puzzles,
        max_calls: (puzzles as u64)
            .saturating_mul(samples as u64)
            .saturating_mul(cascade_call_bound(&voted)),
        run_failed_calls: run.health.run_failed_calls,
        predates_seed_fix: run.health.predates_seed_fix,
    };

    Ok((plan, voted))
}

/// One distinct grid of a puzzle's vote, for the result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoteCount {
    pub count: usize,
    pub correct: bool,
    pub first_sample: usize,
}

/// The vote on one select puzzle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PuzzleVote {
    pub id: String,
    /// The majority grid is exactly the test output.
    pub correct: bool,
    pub majority: Option<Grid>,
    /// Distinct grids in the order first seen.
    pub votes: Vec<VoteCount>,
    pub abstained: usize,
    /// Some sample was right (an oracle bound for the vote, pass@N).
    pub any_correct: bool,
    /// The sampled member's own select-phase answer was right (one sample, from the run).
    pub single_correct: Option<bool>,
    /// The champion's select-phase answer was right (from the run).
    pub champion_correct: Option<bool>,
}

/// The vote on puzzle `p` from its samples' `answers`, in sample order.
pub fn judge_puzzle(
    p: &Puzzle,
    answers: &[Option<Grid>],
    single_correct: Option<bool>,
    champion_correct: Option<bool>,
) -> PuzzleVote {
    let tally = Tally::of(answers);
    let winner = tally.winner();

    PuzzleVote {
        id: p.id.clone(),
        correct: winner.is_some_and(|b| right(p, &b.grid)),
        majority: winner.map(|b| b.grid.clone()),
        votes: tally
            .ballots
            .iter()
            .map(|b| VoteCount {
                count: b.count,
                correct: right(p, &b.grid),
                first_sample: b.first,
            })
            .collect(),
        abstained: tally.abstained,
        any_correct: tally.ballots.iter().any(|b| right(p, &b.grid)),
        single_correct,
        champion_correct,
    }
}

/// Counts over the scored puzzles.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Totals {
    pub puzzles: usize,
    pub vote_correct: usize,
    pub any_correct: usize,
    /// The sampled member's own select-phase run (one sample), when it has every answer.
    pub single_correct: Option<usize>,
    /// The champion's select-phase run, when it has every answer.
    pub champion_correct: Option<usize>,
    /// Puzzles only the vote solved, and only the champion solved (the discordant pairs).
    pub vote_only: Option<usize>,
    pub champion_only: Option<usize>,
    /// Exact two-sided sign test on the discordant pairs (`sign_test`).
    pub sign_test_p: Option<f64>,
}

impl Totals {
    pub fn of(puzzles: &[PuzzleVote]) -> Totals {
        let count = |f: &dyn Fn(&PuzzleVote) -> bool| puzzles.iter().filter(|p| f(p)).count();
        let all = |f: &dyn Fn(&PuzzleVote) -> Option<bool>| {
            puzzles
                .iter()
                .map(f)
                .collect::<Option<Vec<bool>>>()
                .map(|v| v.into_iter().filter(|&x| x).count())
        };
        let has_champion = puzzles.iter().all(|p| p.champion_correct.is_some());
        let split = |vote: bool| {
            has_champion.then(|| count(&|p| p.correct == vote && p.champion_correct == Some(!vote)))
        };
        let (vote_only, champion_only) = (split(true), split(false));

        Totals {
            puzzles: puzzles.len(),
            vote_correct: count(&|p| p.correct),
            any_correct: count(&|p| p.any_correct),
            single_correct: all(&|p| p.single_correct),
            champion_correct: all(&|p| p.champion_correct),
            vote_only,
            champion_only,
            sign_test_p: vote_only.zip(champion_only).map(|(b, c)| sign_test(b, c)),
        }
    }
}

/// The result of a control.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VoteReport {
    #[serde(flatten)]
    pub plan: Plan,
    /// Worker calls the samples made (replayed ones included), and their tokens.
    pub calls_used: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub calls_per_puzzle: f64,
    pub completion_tokens_per_puzzle: f64,
    /// Calls the record did not hold yet, so the model answered them in this process.
    /// None when no model is recorded (simulated models).
    pub new_calls: Option<u64>,
    /// Calls that failed in this process; each one counts as an abstention. They are filed
    /// in `worker-errors.jsonl` beside the record and retried by the next control.
    pub failed_calls: u64,
    /// Every select puzzle scored (not `partial`) and no call failed. Only a complete
    /// result goes to `vote-control.json`; `vote` leaves it false until `settle`.
    pub complete: bool,
    pub totals: Totals,
    pub puzzles: Vec<PuzzleVote>,
}

impl VoteReport {
    /// Take the counts of the recorded backends `filed`, which held `before` answers when
    /// the control began: the calls the models answered now, the failed ones, and with
    /// them whether the result is complete.
    pub fn settle(&mut self, filed: &[Arc<dyn Filed>], before: usize) {
        if !filed.is_empty() {
            let after: usize = filed.iter().map(|f| f.filed()).sum();

            self.new_calls = Some(after.saturating_sub(before) as u64);
        }

        self.failed_calls = filed.iter().map(|f| f.failed()).sum();
        self.complete = !self.plan.partial && self.failed_calls == 0;
    }
}

/// Run `c` (id `cascade_id`) `n` times on every puzzle of `panel`, on up to `jobs` threads.
/// The result is in puzzle order, then sample order, whatever the thread count.
/// `progress` gets (runs done, runs in all) after every run.
#[allow(clippy::too_many_arguments)]
pub fn sample_all(
    pool: &ModelPool,
    cfg: &BreedConfig,
    c: &Cascade,
    cascade_id: &str,
    panel: &[Puzzle],
    n: usize,
    jobs: usize,
    progress: &(dyn Fn(usize, usize) + Sync),
) -> Vec<CascadeRun> {
    let total = panel.len() * n;
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let mut out: Vec<Option<CascadeRun>> = vec![None; total];

    std::thread::scope(|s| {
        let workers: Vec<_> = (0..jobs.clamp(1, total.max(1)))
            .map(|_| {
                s.spawn(|| {
                    let mut mine = Vec::new();

                    loop {
                        // Samples of one puzzle are adjacent, so a server's prefix cache
                        // serves the repeated prompt.
                        let j = next.fetch_add(1, Ordering::Relaxed);

                        if j >= total {
                            break;
                        }

                        let (p, k) = (&panel[j / n], j % n);
                        let seed = sample_seed(cfg.seed, cascade_id, k, &p.id);

                        mine.push((j, run_cascade(pool, &cfg.limits, c, p, seed)));
                        progress(done.fetch_add(1, Ordering::Relaxed) + 1, total);
                    }

                    mine
                })
            })
            .collect();

        for w in workers {
            let runs = w.join().unwrap_or_else(|e| std::panic::resume_unwind(e));

            for (j, r) in runs {
                out[j] = Some(r);
            }
        }
    });

    out.into_iter().flatten().collect()
}

/// The control as `plan` says: sample `voted`, vote per puzzle, score, and compare with
/// the run's own select-phase answers. `panel` is the regenerated select panel.
pub fn vote(
    pool: &ModelPool,
    run: &RunDir,
    plan: Plan,
    voted: &Cascade,
    panel: &[Puzzle],
    jobs: usize,
    progress: &(dyn Fn(usize, usize) + Sync),
) -> Result<VoteReport> {
    let panel = panel
        .get(..plan.select_puzzles)
        .ok_or_else(|| invalid("the panel is shorter than the plan".to_string()))?;
    let n = plan.samples_per_puzzle;
    let runs = sample_all(pool, &run.cfg, voted, &plan.cascade.id, panel, n, jobs, progress);

    if runs.len() != panel.len() * n {
        return Err(invalid("a sample did not run".to_string()));
    }

    let stored = |member: Option<&str>, p: &Puzzle| -> Option<bool> {
        let answer = run.snapshot.select_answers.get(member?)?.get(&p.id)?;

        Some(answer.as_ref().is_some_and(|g| right(p, g)))
    };
    let champion = plan.champion.as_ref().map(|c| c.id.as_str());
    let puzzles: Vec<PuzzleVote> = panel
        .iter()
        .zip(runs.chunks(n))
        .map(|(p, samples)| {
            let answers: Vec<Option<Grid>> = samples.iter().map(|r| r.answer.clone()).collect();

            judge_puzzle(
                p,
                &answers,
                stored(Some(&plan.cascade.member), p),
                stored(champion, p),
            )
        })
        .collect();
    let sum = |f: fn(&CascadeRun) -> u64| runs.iter().map(f).sum::<u64>();
    let (calls, prompt, completion) = (
        sum(|r| r.usage.calls),
        sum(|r| r.usage.prompt_tokens),
        sum(|r| r.usage.completion_tokens),
    );
    let per = panel.len().max(1) as f64;

    Ok(VoteReport {
        plan,
        calls_used: calls,
        prompt_tokens: prompt,
        completion_tokens: completion,
        calls_per_puzzle: calls as f64 / per,
        completion_tokens_per_puzzle: completion as f64 / per,
        new_calls: None,
        failed_calls: 0,
        complete: false,
        totals: Totals::of(&puzzles),
        puzzles,
    })
}

/// A recorded backend whose filing can be counted.
pub trait Filed: ModelBackend {
    /// Answers on file.
    fn filed(&self) -> usize;
    /// Failed calls in this process.
    fn failed(&self) -> u64;
}

impl<B: ModelBackend> Filed for Recorded<B> {
    fn filed(&self) -> usize {
        self.len()
    }

    fn failed(&self) -> u64 {
        self.errors()
    }
}

/// A recorded backend shared between the pool and the counts.
pub struct Shared(pub Arc<dyn Filed>);

impl ModelBackend for Shared {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
        self.0.generate(req)
    }
}

/// The servers, as brut-breed-ollama reads them from the environment.
#[derive(Clone, Debug)]
pub struct Endpoints {
    pub ollama: String,
    pub vllm: String,
    pub timeout: Duration,
}

impl Endpoints {
    /// `OLLAMA_URL`, `VLLM_URL` and `OLLAMA_TIMEOUT_SECS` (seconds, for both), with the
    /// defaults of brut-breed-ollama.
    pub fn from_env() -> Endpoints {
        let secs = std::env::var("OLLAMA_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300);

        Endpoints {
            ollama: std::env::var("OLLAMA_URL").unwrap_or_else(|_| crate::DEFAULT_BASE.to_string()),
            vllm: std::env::var("VLLM_URL").unwrap_or_else(|_| vllm::DEFAULT_BASE.to_string()),
            timeout: Duration::from_secs(secs),
        }
    }
}

/// A backend for every model `c` names, as the run had: `ollama:` and `vllm:` tags behind
/// `Recorded` on `record`, simulated models as they are (they are deterministic and the
/// run did not record them either). Also returns the recorded ones, for their counts.
pub fn build_pool(c: &Cascade, record: &Path, ep: &Endpoints) -> Result<(ModelPool, Vec<Arc<dyn Filed>>)> {
    let mut pool = ModelPool::new();
    let mut filed: Vec<Arc<dyn Filed>> = Vec::new();
    let tags: BTreeSet<&str> = c
        .stages
        .iter()
        .flat_map(|s| &s.workers)
        .map(|w| w.model.as_str())
        .collect();

    for tag in tags {
        let recorded: Arc<dyn Filed> = if tag.starts_with(TAG_PREFIX) {
            Arc::new(Recorded::open(OllamaBackend::new(tag, &ep.ollama, ep.timeout)?, record)?)
        } else if tag.starts_with(vllm::TAG_PREFIX) {
            Arc::new(Recorded::open(VllmBackend::new(tag, &ep.vllm, ep.timeout)?, record)?)
        } else if let Some(sim) = SimModel::from_tag(tag) {
            pool.insert(tag, Box::new(sim));
            continue;
        } else {
            return Err(invalid(format!("no backend for model {tag:?}")));
        };

        pool.insert(tag, Box::new(Shared(recorded.clone())));
        filed.push(recorded);
    }

    Ok((pool, filed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashMap};
    use std::path::PathBuf;
    use std::sync::Mutex;

    use brutmaschine::backend::StopReason;
    use brutmaschine::cascade::arena::{Event, breed_to_end_in};
    use brutmaschine::cascade::domain::GenePool;
    use brutmaschine::cascade::genome::CascadeLimits;
    use brutmaschine::cascade::puzzle::{generate_panel, parse_prompt, render_answer, render_grid};

    fn grid(rows: &[&[u8]]) -> Grid {
        rows.iter().map(|r| r.to_vec()).collect()
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("vote-control-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");

        dir
    }

    fn sim_pool() -> ModelPool {
        let mut pool = ModelPool::new();

        for m in [SimModel::small(), SimModel::large()] {
            let tag = m.tag().to_string();

            pool.insert(&tag, Box::new(m));
        }

        pool
    }

    fn tiny_config() -> BreedConfig {
        BreedConfig {
            seed: 5,
            generations: 1,
            children_per_generation: 3,
            search_puzzles: 6,
            select_puzzles: 5,
            finalists: 2,
            max_calls: 5000,
            ..BreedConfig::default()
        }
    }

    /// A finished simulated run written as brut-breed writes it: config.json, the
    /// checkpoint in snapshot.json, and the log's first line.
    fn fixture_run(name: &str) -> (PathBuf, Vec<Event>) {
        let dir = temp_dir(name);
        let cfg = tiny_config();
        let (_, events, snapshot) =
            breed_to_end_in::<GridDomain>(cfg.clone(), GenePool::grid(), &sim_pool(), None).expect("breed");
        let checkpoint = serde_json::json!({"snapshot": snapshot, "log_len": events.len(), "log_head": "00"});
        let first = serde_json::json!({"seq": 0, "prev": "00", "hash": "11", "event": events[0]});

        std::fs::write(dir.join(CONFIG), serde_json::to_string_pretty(&cfg).expect("cfg")).expect("write");
        std::fs::write(dir.join(SNAPSHOT), serde_json::to_string(&checkpoint).expect("cp")).expect("write");
        std::fs::write(dir.join(EVENTS), format!("{first}\n")).expect("write");

        (dir, events)
    }

    #[test]
    fn majority_counts_votes_and_breaks_ties_by_first_seen() {
        let (a, b, c) = (grid(&[&[1]]), grid(&[&[2]]), grid(&[&[3, 4]]));

        let t = Tally::of(&[Some(a.clone()), Some(b.clone()), Some(a.clone())]);
        assert_eq!(t.winner().map(|w| (&w.grid, w.count)), Some((&a, 2)));
        assert_eq!(t.abstained, 0);

        // A tie goes to the grid seen first, in sample order.
        let t = Tally::of(&[Some(a.clone()), Some(b.clone()), Some(b.clone()), Some(a.clone())]);
        assert_eq!(t.winner().map(|w| &w.grid), Some(&a));
        let t = Tally::of(&[Some(b.clone()), Some(a.clone()), Some(a.clone()), Some(b.clone())]);
        assert_eq!(t.winner().map(|w| &w.grid), Some(&b));

        // Abstentions are counted, never voted; a later larger count still wins.
        let t = Tally::of(&[None, Some(c.clone()), Some(a.clone()), None, Some(a.clone())]);
        assert_eq!(t.abstained, 2);
        assert_eq!(t.winner().map(|w| (&w.grid, w.count, w.first)), Some((&a, 2, 2)));
        assert_eq!(
            t.ballots.iter().map(|b| (b.count, b.first)).collect::<Vec<_>>(),
            vec![(1, 1), (2, 2)]
        );

        let t = Tally::of(&[None, None]);
        assert_eq!(t.winner(), None);
        assert_eq!(t.abstained, 2);
        assert_eq!(Tally::of(&[]).winner(), None);
    }

    #[test]
    fn scoring_is_exact_against_the_test_output() {
        let p = generate_panel(3, 1).remove(0);
        let truth = p.test_output.clone();
        let mut near = truth.clone();
        near[0][0] = (near[0][0] + 1) % 10;
        let mut longer = truth.clone();
        longer.push(truth[0].clone());

        let v = judge_puzzle(&p, &[Some(truth.clone())], Some(true), None);
        assert!(v.correct && v.any_correct);
        assert_eq!(v.majority.as_ref(), Some(&truth));

        // A near miss in the majority is wrong, though a minority sample was right.
        let v = judge_puzzle(&p, &[Some(near.clone()), Some(truth.clone()), Some(near.clone())], None, Some(false));
        assert!(!v.correct && v.any_correct);
        assert_eq!(
            v.votes,
            vec![
                VoteCount { count: 2, correct: false, first_sample: 0 },
                VoteCount { count: 1, correct: true, first_sample: 1 },
            ]
        );

        // An extra row is wrong too, and so is no answer at all.
        assert!(!judge_puzzle(&p, &[Some(longer)], None, None).correct);
        let v = judge_puzzle(&p, &[None, None], None, None);
        assert!(!v.correct && !v.any_correct && v.majority.is_none());
        assert_eq!(v.abstained, 2);
    }

    #[test]
    fn sign_test_matches_the_binomial_tail() {
        assert_eq!(sign_test(0, 0), 1.0);
        assert!((sign_test(0, 5) - 0.0625).abs() < 1e-12);
        assert!((sign_test(5, 0) - 0.0625).abs() < 1e-12);
        assert!((sign_test(1, 9) - 22.0 / 1024.0).abs() < 1e-12);
        assert_eq!(sign_test(3, 3), 1.0);
        assert_eq!(sign_test(1000, 1000), 1.0);

        let p = sign_test(400, 600);
        assert!(p > 0.0 && p < 1e-9, "{p}");
    }

    #[test]
    fn totals_count_the_discordant_pairs() {
        let vote = |correct, champion| PuzzleVote {
            id: String::new(),
            correct,
            majority: None,
            votes: Vec::new(),
            abstained: 0,
            any_correct: correct,
            single_correct: Some(false),
            champion_correct: champion,
        };
        let t = Totals::of(&[
            vote(true, Some(false)),
            vote(true, Some(true)),
            vote(false, Some(true)),
            vote(true, Some(false)),
            vote(false, Some(false)),
        ]);

        assert_eq!((t.puzzles, t.vote_correct, t.any_correct), (5, 3, 3));
        assert_eq!((t.single_correct, t.champion_correct), (Some(0), Some(2)));
        assert_eq!((t.vote_only, t.champion_only), (Some(2), Some(1)));
        assert_eq!(t.sign_test_p, Some(sign_test(2, 1)));

        // Without the champion's answers there is no paired count.
        let t = Totals::of(&[vote(true, None), vote(false, Some(true))]);
        assert_eq!((t.champion_correct, t.vote_only, t.sign_test_p), (None, None, None));
    }

    /// Hands out answers and keeps every request seed.
    struct Seeds(Mutex<Vec<u64>>);

    impl ModelBackend for Seeds {
        fn name(&self) -> &str {
            "sim:small"
        }

        fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
            self.0.lock().expect("lock").push(req.seed);

            Ok(GenResponse {
                text: "ANSWER\n1".to_string(),
                prompt_tokens: 1,
                completion_tokens: 1,
                stop_reason: StopReason::Stop,
                backend: "sim:small".to_string(),
            })
        }
    }

    impl Filed for Seeds {
        fn filed(&self) -> usize {
            0
        }

        fn failed(&self) -> u64 {
            0
        }
    }

    #[test]
    fn sample_seeds_are_deterministic_distinct_and_63_bit() {
        let tasks: Vec<String> = generate_panel(8, 6).into_iter().map(|p| p.id).collect();
        let mut seen = std::collections::HashSet::new();

        for task in &tasks {
            for k in 0..32 {
                let s = sample_seed(11, "abc", k, task);

                assert_eq!(s, sample_seed(11, "abc", k, task), "deterministic");
                assert!(s <= i64::MAX as u64);
                assert!(seen.insert(s), "sample {k} on {task} repeats a seed");
                assert_ne!(s, wire_seed(derive_seed(11, &format!("select/abc/{task}"))));
            }
        }

        assert_ne!(sample_seed(11, "abc", 0, &tasks[0]), sample_seed(12, "abc", 0, &tasks[0]));
        assert_ne!(sample_seed(11, "abc", 0, &tasks[0]), sample_seed(11, "abd", 0, &tasks[0]));

        // The seeds the Worker requests carry differ on the wire too, sample by sample.
        let shared = Arc::new(Seeds(Mutex::new(Vec::new())));
        let pool = pool_of(shared.clone());
        let cfg = tiny_config();
        let (_, c) = seed_cascades().remove(0);
        let panel = generate_panel(8, 2);
        let runs = sample_all(&pool, &cfg, &c, &c.id(), &panel, 16, 4, &|_, _| {});
        let wire: BTreeSet<u64> = shared.0.lock().expect("lock").iter().map(|&s| wire_seed(s)).collect();

        assert_eq!(runs.len(), 32);
        assert_eq!(wire.len(), 32, "every sample asks with its own seed");
    }

    #[test]
    fn panel_regeneration_matches_a_stored_snapshot() {
        let (dir, events) = fixture_run("panel");
        let run = load_run(&dir).expect("load");
        let panel = select_panel(&run.cfg).expect("panel");
        let Event::RunStart { select, .. } = &events[0] else {
            panic!("the first event is run_start");
        };

        assert!(run.snapshot.finished);
        assert_eq!(panel.len(), 5);
        assert_eq!(&panel.iter().map(|p| p.id.clone()).collect::<Vec<_>>(), select);
        assert_eq!(panel, generate_panel(derive_seed(5, "panel/select"), 5));
        assert!(!run.snapshot.select_answers.is_empty());
        check_panel(&panel, &run).expect("snapshot matches");
        assert!(check_run_start(&dir, &panel).expect("log matches"));

        // Another seed's panel does not match the snapshot.
        let other = generate_panel(derive_seed(6, "panel/select"), 5);
        assert!(check_panel(&other, &run).is_err());
        assert!(check_run_start(&dir, &other).is_err());

        // A shorter panel misses answers of the finished run.
        assert!(check_panel(&panel[..4], &run).is_err());

        // A finalist without its select answers is refused too.
        let mut gap = run.clone();
        let champ = champion(&gap.snapshot.archive, gap.cfg.finalists).expect("champion");
        let id = gap.snapshot.archive[champ].id.clone();
        gap.snapshot.select_answers.remove(&id);
        assert!(check_panel(&panel, &gap).is_err());

        // Settings changed after the run are refused.
        let mut cfg = run.cfg.clone();
        cfg.select_puzzles = 6;
        std::fs::write(dir.join(CONFIG), serde_json::to_string(&cfg).expect("cfg")).expect("write");
        assert!(matches!(load_run(&dir), Err(BrutError::Integrity(_))));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_champion_budget_sets_the_samples() {
        let (dir, _) = fixture_run("budget");
        let mut run = load_run(&dir).expect("load");
        let seed = default_cascade(&run.snapshot.archive).expect("seed");
        let champ = champion(&run.snapshot.archive, run.cfg.finalists).expect("champion");
        let picks = finalists(&run.snapshot.archive, run.cfg.finalists);
        let scored: BTreeSet<&String> = run.snapshot.select_answers.keys().collect();

        // The finalists are the members the run scored on the select panel, champion first.
        assert_eq!(picks.iter().map(|&i| &run.snapshot.archive[i].id).collect::<BTreeSet<_>>(), scored);
        assert_eq!(picks[0], champ);
        let puzzles = run.snapshot.archive[champ].answers.len() as u64;

        assert_eq!(run.snapshot.archive[seed].op, Op::Seed);
        assert_ne!(run.snapshot.archive[champ].op, Op::Seed);
        // The seed is the best seed by search standing, whatever it did on select.
        assert_eq!(
            Some(seed),
            rank(&run.snapshot.archive).into_iter().find(|&i| run.snapshot.archive[i].op == Op::Seed)
        );

        run.snapshot.archive[champ].usage.calls = 3 * puzzles;
        let (p, c) = plan(&run, &Options::default()).expect("plan");
        assert_eq!((p.samples_per_puzzle, p.samples_from.as_str()), (3, "champion_calls"));
        assert_eq!(p.cascade.member, run.snapshot.archive[seed].id);
        assert_eq!(p.cascade.id, c.id());
        assert!(p.cascade.is_seed && p.cascade.seed_name.is_some());
        assert_eq!(p.champion.as_ref().map(|c| c.calls_per_puzzle), Some(3.0));
        assert_eq!((p.select_puzzles, p.partial, p.max_calls), (5, false, 15));

        // Halves round up; less than half a run is still one sample.
        run.snapshot.archive[champ].usage.calls = 5 * puzzles / 2;
        assert_eq!(plan(&run, &Options::default()).expect("plan").0.samples_per_puzzle, 3);
        run.snapshot.archive[champ].usage.calls = 1;
        assert_eq!(plan(&run, &Options::default()).expect("plan").0.samples_per_puzzle, 1);
        assert_eq!(samples_for(7.4, 2.0), 4);
        assert_eq!(samples_for(3.0, 0.0), 1);

        let opts = Options { samples: Some(7), puzzles: Some(2), ..Options::default() };
        let (p, _) = plan(&run, &opts).expect("plan");
        assert_eq!((p.samples_per_puzzle, p.samples_from.as_str()), (7, "override"));
        assert_eq!((p.select_puzzles, p.partial), (2, true));

        for bad in [
            Options { samples: Some(0), ..Options::default() },
            Options { puzzles: Some(0), ..Options::default() },
            Options { puzzles: Some(6), ..Options::default() },
            Options { temperature: Some(2.0), ..Options::default() },
            Options { cascade: Some("nobody".to_string()), ..Options::default() },
        ] {
            assert!(plan(&run, &bad).is_err(), "{bad:?}");
        }

        // A temperature override samples another Cascade, from the same member.
        let opts = Options { temperature: Some(0.7), ..Options::default() };
        let (p, c) = plan(&run, &opts).expect("plan");
        assert_ne!(p.cascade.id, p.cascade.member);
        assert!(c.stages.iter().flat_map(|s| &s.workers).all(|w| w.traits.temperature == 0.7));

        // Without a bred finalist the budget must be given.
        run.cfg.finalists = 0;
        assert_eq!(champion(&run.snapshot.archive, 0), None);
        assert!(plan(&run, &Options::default()).is_err());
        assert!(plan(&run, &Options { samples: Some(2), ..Options::default() }).is_ok());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Knows the right grid of every puzzle by its test input; answers right, a near miss,
    /// or nothing usable, by the request seed. Counts its calls.
    struct Oracle {
        truth: HashMap<String, Grid>,
        calls: AtomicUsize,
    }

    impl Oracle {
        fn new(panel: &[Puzzle]) -> Oracle {
            Oracle {
                truth: panel.iter().map(|p| (render_grid(&p.test_input), p.test_output.clone())).collect(),
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl ModelBackend for Oracle {
        fn name(&self) -> &str {
            "sim:small"
        }

        fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);

            let task = parse_prompt(&req.prompt).expect("a solve prompt");
            let truth = self.truth[&render_grid(&task.test_input)].clone();
            let mut near = truth.clone();
            near[0][0] = (near[0][0] + 1) % 10;
            let text = match wire_seed(req.seed) % 5 {
                0 | 1 => render_answer(&truth),
                2 | 3 => render_answer(&near),
                _ => "no idea".to_string(),
            };

            Ok(GenResponse {
                text,
                prompt_tokens: 10,
                completion_tokens: 4,
                stop_reason: StopReason::Stop,
                backend: "sim:small".to_string(),
            })
        }
    }

    impl Filed for Oracle {
        fn filed(&self) -> usize {
            0
        }

        fn failed(&self) -> u64 {
            0
        }
    }

    fn pool_of(backend: Arc<dyn Filed>) -> ModelPool {
        let mut pool = ModelPool::new();

        pool.insert("sim:small", Box::new(Shared(backend)));
        pool
    }

    #[test]
    fn the_vote_scores_every_puzzle_as_its_samples_say() {
        let (dir, _) = fixture_run("vote");
        let run = load_run(&dir).expect("load");
        let panel = select_panel(&run.cfg).expect("panel");
        let opts = Options { samples: Some(5), ..Options::default() };
        let (p, voted) = plan(&run, &opts).expect("plan");
        let oracle = Arc::new(Oracle::new(&panel));
        let pool = pool_of(oracle.clone());
        let report = vote(&pool, &run, p.clone(), &voted, &panel, 4, &|_, _| {}).expect("vote");

        // One Worker, one call per sample.
        assert_eq!(report.calls_used, 25);
        assert_eq!(oracle.calls.load(Ordering::SeqCst), 25);
        assert_eq!(report.calls_per_puzzle, 5.0);
        assert_eq!(report.completion_tokens, 100);

        // Each puzzle, recounted from its samples run one by one.
        for (pz, got) in panel.iter().zip(&report.puzzles) {
            let answers: Vec<Option<Grid>> = (0..5)
                .map(|k| {
                    let seed = sample_seed(run.cfg.seed, &p.cascade.id, k, &pz.id);

                    run_cascade(&pool, &run.cfg.limits, &voted, pz, seed).answer
                })
                .collect();
            let mut counts: BTreeMap<(usize, usize), (Grid, usize)> = BTreeMap::new();
            let mut firsts: Vec<Grid> = Vec::new();

            for a in answers.iter().flatten() {
                if !firsts.contains(a) {
                    firsts.push(a.clone());
                }
            }

            for (i, g) in firsts.iter().enumerate() {
                let n = answers.iter().filter(|a| a.as_ref() == Some(g)).count();

                counts.insert((usize::MAX - n, i), (g.clone(), n));
            }

            let best = counts.values().next().map(|(g, _)| g.clone());

            assert_eq!(got.id, pz.id);
            assert_eq!(got.majority, best);
            assert_eq!(got.correct, best.as_ref() == Some(&pz.test_output));
            assert_eq!(got.abstained, answers.iter().filter(|a| a.is_none()).count());
            assert_eq!(got.votes.iter().map(|v| v.count).sum::<usize>() + got.abstained, 5);
            assert_eq!(got.any_correct, answers.contains(&Some(pz.test_output.clone())));

            let single = run.snapshot.select_answers[&p.cascade.member][&pz.id].as_ref();
            assert_eq!(got.single_correct, Some(single == Some(&pz.test_output)));
            assert!(got.champion_correct.is_some());
        }

        assert_eq!(report.totals, Totals::of(&report.puzzles));
        assert_eq!(report.totals.puzzles, 5);

        // The fixture meets every case: right and wrong majorities, split votes and
        // abstentions.
        let some = |f: fn(&PuzzleVote) -> bool| report.puzzles.iter().any(f);
        assert!(some(|v| v.correct) && some(|v| !v.correct));
        assert!(some(|v| v.votes.len() > 1) && some(|v| v.abstained > 0));
        assert!(some(|v| v.any_correct && !v.correct));

        // The same control on one thread gives the same result.
        let again = vote(&pool, &run, p, &voted, &panel, 1, &|_, _| {}).expect("again");
        assert_eq!(again, report);

        // The result reads back as written.
        let json = serde_json::to_string_pretty(&report).expect("json");
        assert_eq!(serde_json::from_str::<VoteReport>(&json).expect("back"), report);
        assert!(json.contains("\"samples_per_puzzle\": 5"), "{json}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn recorded_samples_replay_and_a_larger_n_reuses_them() {
        let (dir, _) = fixture_run("replay");
        let run = load_run(&dir).expect("load");
        let panel = select_panel(&run.cfg).expect("panel");
        let record = dir.join(RECORD);
        let control = |n: usize| {
            let oracle = Oracle::new(&panel);
            let rec = Arc::new(Recorded::open(oracle, &record).expect("open"));
            let pool = pool_of(rec.clone());
            let (p, voted) = plan(&run, &Options { samples: Some(n), ..Options::default() }).expect("plan");
            let report = vote(&pool, &run, p, &voted, &panel, 3, &|_, _| {}).expect("vote");

            (report, rec.inner.calls.load(Ordering::SeqCst), rec.len())
        };

        let (first, asked, filed) = control(2);
        assert_eq!((asked, filed), (10, 10));

        // A repeat asks nothing and gives the same result.
        let (again, asked, filed) = control(2);
        assert_eq!((asked, filed), (0, 10));
        assert_eq!(again, first);

        // Three samples reuse the two on file and ask only for the third.
        let (more, asked, filed) = control(3);
        assert_eq!((asked, filed), (5, 15));
        assert_eq!(more.calls_used, 15);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Fails every call, as a dead server or one serving another model does.
    struct Dead;

    impl ModelBackend for Dead {
        fn name(&self) -> &str {
            "sim:small"
        }

        fn generate(&self, _req: &GenRequest) -> Result<GenResponse> {
            Err(BrutError::Io(std::io::Error::other("vllm: HTTP 404 Not Found")))
        }
    }

    #[test]
    fn a_failed_call_leaves_the_result_incomplete() {
        let (dir, _) = fixture_run("dead");
        let run = load_run(&dir).expect("load");
        let panel = select_panel(&run.cfg).expect("panel");
        let (p, voted) = plan(&run, &Options { samples: Some(2), ..Options::default() }).expect("plan");

        let dead: Arc<dyn Filed> = Arc::new(Recorded::open(Dead, &dir.join(RECORD)).expect("open"));
        let mut report = vote(&pool_of(dead.clone()), &run, p.clone(), &voted, &panel, 2, &|_, _| {}).expect("vote");

        report.settle(&[dead], 0);
        assert_eq!((report.failed_calls, report.new_calls), (10, Some(0)));
        assert!(!report.complete, "a failed call never makes a result of record");
        assert!(report.puzzles.iter().all(|v| v.abstained == 2 && !v.correct));
        assert!(serde_json::to_string(&report).expect("json").contains("\"complete\":false"));

        // The same control with a working model is complete, once settled.
        let ok: Arc<dyn Filed> =
            Arc::new(Recorded::open(Oracle::new(&panel), &dir.join("ok/worker-calls.jsonl")).expect("open"));
        let mut report = vote(&pool_of(ok.clone()), &run, p, &voted, &panel, 2, &|_, _| {}).expect("vote");

        assert!(!report.complete, "not before settle");
        report.settle(std::slice::from_ref(&ok), 0);
        assert!(report.complete && report.failed_calls == 0);
        assert_eq!(report.new_calls, Some(10));

        // A smoke test on part of the panel is never complete either.
        let (p, voted) = plan(&run, &Options { samples: Some(2), puzzles: Some(2), ..Options::default() }).expect("plan");
        let mut report = vote(&pool_of(ok.clone()), &run, p, &voted, &panel, 2, &|_, _| {}).expect("vote");

        report.settle(&[ok], 10);
        assert!(!report.complete && report.failed_calls == 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `n` lines of a run's record for `tag`, with request seeds `seed(i)`.
    fn records(tag: &str, n: usize, seed: impl Fn(usize) -> u64) -> String {
        (0..n)
            .map(|i| {
                let request = GenRequest {
                    system: "sys".to_string(),
                    prompt: format!("puzzle {i}"),
                    max_tokens: 64,
                    temperature: 0.0,
                    seed: seed(i),
                    think_tokens: 0,
                };
                let rec = crate::CallRecord {
                    key: crate::call_key(tag, &request),
                    tag: tag.to_string(),
                    request,
                    response: GenResponse {
                        text: "ANSWER\n1".to_string(),
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        stop_reason: StopReason::Stop,
                        backend: tag.to_string(),
                    },
                };

                format!("{}\n", serde_json::to_string(&rec).expect("json"))
            })
            .collect()
    }

    #[test]
    fn a_run_before_the_seed_fix_or_with_failed_calls_is_flagged() {
        use brutmaschine::cascade::genome::ModelSpec;

        let dir = temp_dir("health");
        let record = dir.join(RUN_RECORD);
        let mut cfg = tiny_config();
        let health = |cfg: &BreedConfig| run_health(&dir, cfg).expect("health");

        // No llm/ directory at all: a simulated run.
        assert_eq!(health(&cfg), RunHealth::default());
        assert_eq!(RunHealth::default().problem(), None);

        // 40 vLLM records, every seed at most i64::MAX: the record of a run before the fix,
        // which lost every call with a higher seed. A run on simulated models only is not
        // read for it.
        std::fs::create_dir_all(record.parent().expect("parent")).expect("llm dir");
        std::fs::write(&record, records("vllm:qwen2.5-32b", 40, |i| (i as u64) << 50)).expect("write");
        assert!(!health(&cfg).predates_seed_fix);

        cfg.limits.models.push(ModelSpec { tag: "vllm:qwen2.5-32b".to_string(), cost_per_call: 0.01 });
        let old = health(&cfg);
        assert_eq!(old, RunHealth { run_failed_calls: 0, predates_seed_fix: true });
        assert!(old.problem().is_some_and(|m| m.contains(SEED_FIX)));

        // One seed above i64::MAX shows a run after the fix, wherever it stands.
        let mut text = records("vllm:qwen2.5-32b", 1, |_| u64::MAX);
        text.push_str(&records("vllm:qwen2.5-32b", 40, |i| i as u64));
        std::fs::write(&record, text).expect("write");
        assert!(!health(&cfg).predates_seed_fix);

        // Fewer than 32 vLLM records say nothing; records of other models do not count, and
        // a last line still being written is not read.
        let mut text = records("ollama:qwen2.5:7b", 40, |i| i as u64);
        text.push_str(&records("vllm:qwen2.5-32b", SEED_PROBE_RECORDS - 1, |i| i as u64));
        text.push_str("{\"key\":\"vllm:qwen2.5-32b#");
        std::fs::write(&record, &text).expect("write");
        assert!(!health(&cfg).predates_seed_fix);

        text.truncate(text.rfind('\n').expect("newline") + 1);
        text.push_str(&records("vllm:qwen2.5-32b", 1, |_| 3));
        std::fs::write(&record, &text).expect("write");
        assert!(health(&cfg).predates_seed_fix, "the 32nd low seed tips it");

        // A broken line in the middle is an error, not a guess.
        std::fs::write(&record, format!("{{broken\n{text}")).expect("write");
        assert!(matches!(run_health(&dir, &cfg), Err(BrutError::Integrity(_))));
        std::fs::remove_file(&record).expect("rm");

        // Every non-empty line of llm/worker-errors.jsonl is a failed call of the run.
        std::fs::write(dir.join(RUN_ERRORS), "{\"key\":\"a\"}\n\n{\"key\":\"b\"}\n{\"key\":\"c\"}\n").expect("write");
        let failed = health(&cfg);
        assert_eq!(failed, RunHealth { run_failed_calls: 3, predates_seed_fix: false });
        assert!(failed.problem().is_some_and(|m| m.contains("3 of its own Worker calls failed")));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_plan_carries_the_run_health() {
        let (dir, _) = fixture_run("plan-health");
        let opts = Options { samples: Some(2), ..Options::default() };

        let (p, _) = plan(&load_run(&dir).expect("load"), &opts).expect("plan");
        assert_eq!((p.run_failed_calls, p.predates_seed_fix), (0, false));

        std::fs::create_dir_all(dir.join("llm")).expect("llm dir");
        std::fs::write(dir.join(RUN_ERRORS), "{\"key\":\"a\"}\n").expect("write");
        let mut run = load_run(&dir).expect("load");
        let (p, _) = plan(&run, &opts).expect("plan");
        assert_eq!((p.run_failed_calls, p.predates_seed_fix), (1, false));

        run.health.predates_seed_fix = true;
        let json = serde_json::to_string(&plan(&run, &opts).expect("plan").0).expect("json");
        assert!(json.contains("\"run_failed_calls\":1") && json.contains("\"predates_seed_fix\":true"), "{json}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_pool_records_real_models_and_keeps_simulated_ones_plain() {
        let dir = temp_dir("pool");
        let record = dir.join(RECORD);
        let ep = Endpoints {
            ollama: crate::DEFAULT_BASE.to_string(),
            vllm: vllm::DEFAULT_BASE.to_string(),
            timeout: Duration::from_secs(1),
        };
        let (_, mut c) = seed_cascades().remove(0);

        let (pool, filed) = build_pool(&c, &record, &ep).expect("sim");
        assert!(pool.get("sim:small").is_ok() && filed.is_empty());

        c.stages[0].workers[0].model = "vllm:qwen2.5-32b".to_string();
        let (pool, filed) = build_pool(&c, &record, &ep).expect("vllm");
        assert_eq!(pool.get("vllm:qwen2.5-32b").map(|b| b.name().to_string()).ok(), Some("vllm:qwen2.5-32b".to_string()));
        assert_eq!(filed.len(), 1);

        for bad in ["vllm:some-model", "gpt-unknown"] {
            c.stages[0].workers[0].model = bad.to_string();
            assert!(build_pool(&c, &record, &ep).is_err(), "{bad}");
        }

        // A non-loopback server is refused before any call.
        c.stages[0].workers[0].model = "vllm:qwen2.5-32b".to_string();
        let far = Endpoints { vllm: "http://10.0.0.1:8000".to_string(), ..ep };
        assert!(build_pool(&c, &record, &far).is_err());

        let limits = CascadeLimits::default();
        assert!(seed_cascades().iter().all(|(_, c)| c.check(&limits).is_ok()));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
