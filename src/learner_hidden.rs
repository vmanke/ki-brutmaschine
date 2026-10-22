//! Hidden evaluation of learner programs: secret tiers, budgets, ARC checks (evaluator side).
//!
//! Only `brut-eval` constructs a `LearnerEvaluator`. The orchestrator only calls
//! `request_learner_eval` / `request_arc_eval`, which start `brut-eval` as a child process.
//!
//! Leakage controls (contract sections 5.15, 6.8 and 9):
//! - One global secret per machine, by default `~/.brut/learner/hidden.env` (lead decision D3;
//!   dir 0700, file 0600). Every hidden task seed derives from it. The secret never leaves
//!   this process: no message, response or file written here contains it.
//! - Every query is charged in the hash-chained ledger next to the secret, under `eval.lock`,
//!   before any hidden task is built. The caps per protocol (total) and per run come from the
//!   secret file; `--budget` can only lower the per-run cap. A new run directory or run id does
//!   not reset the total.
//! - A learner query releases one number, Δ_T rounded down to the step (>= 0.01), plus the
//!   pass/fail bit of the primary test for HIDDEN-GRAMMAR, the only primary endpoint (lead
//!   decision D4). That is about 8 bits per query.
//! - Malformed requests (bad identifiers, not a hidden tier) are rejected without a ledger
//!   entry. Well-formed requests that are refused (invalid program, protocol mismatch, budget,
//!   contamination, overlap, bad ARC data) get a `Refused` entry and are not charged.
//! - Error messages never quote request content, hidden data or the secret.
//!
//! Limit: on one Unix user this is process separation, not a security boundary.
//!
//! Owner: WP4. Contract section 5.15.
#![forbid(unsafe_code)]

use std::collections::{BTreeSet, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::arc_data;
use crate::hidden_eval::{
    AnyEvalRequest, HIDDEN_STEP_KEY, LOCK_FILE, MIN_HIDDEN_SCORE_STEP, check_private,
    create_private_dir, lock_exclusive, quantize, read_limited_bytes, request_eval_json,
    write_private_atomic,
};
use crate::learner::arc_cells::{self, ArcResult};
use crate::learner::baselines::{self, TrainingCost};
use crate::learner::families::{self, InstanceParams};
use crate::learner::grammar::{self, GrammarFamily};
use crate::learner::protocol;
use crate::learner::stats;
use crate::learner::types::{
    ArcPair, ArcTask, BaselineId, Decision, DslLimits, EvalSet, EvalSettings, LearnerGenome,
    MlpRecipe, SUPPORTED_F, SourceId, TaskInstance, TaskResult, Tier, TunedBaselines,
};
use crate::ledger::{Ledger, LedgerEntry, LedgerKind, QueryKind, is_hex64};
use crate::util::{BrutError, Result, derive_seed, now_unix_ms, parse_env_file, sha256_hex};

pub const LEARNER_SEED_KEY: &str = "BRUT_LEARNER_HIDDEN_SEED";
/// Default 48.
pub const BUDGET_TOTAL_KEY: &str = "BRUT_LEARNER_BUDGET_TOTAL";
/// Default 12.
pub const BUDGET_PER_RUN_KEY: &str = "BRUT_LEARNER_BUDGET_PER_RUN";
pub const ARC_ROOT_KEY: &str = "BRUT_ARC_DATA_ROOT";
pub const ARC_MANIFEST_KEY: &str = "BRUT_ARC_MANIFEST";
/// Default 3.
pub const ARC_BUDGET_KEY: &str = "BRUT_ARC_BUDGET_PER_SET";
pub const LEARNER_EVAL_TIMEOUT_SECONDS: f64 = 600.0;
pub const ARC_EVAL_TIMEOUT_SECONDS: f64 = 900.0;

/// Default total learner queries per protocol when the secret file does not set them.
pub const DEFAULT_BUDGET_TOTAL: usize = 48;
/// Default learner queries per run.
pub const DEFAULT_BUDGET_PER_RUN: usize = 12;
/// Default ARC checks per eval set and protocol.
pub const DEFAULT_ARC_BUDGET_PER_SET: usize = 3;

/// File name of the learner secret inside its directory.
pub const LEARNER_SECRET_FILE: &str = "hidden.env";
/// Cached tuned baselines and their hidden-tier scores, next to the secret (never charged).
pub const BASELINE_CACHE_FILE: &str = "baseline_cache.json";

/// Wall-clock limit per task inside brut-eval (contract: `EvalSettings`, brut-eval 30 s).
pub const EVAL_SECONDS_PER_TASK: f64 = 30.0;
/// Validation examples of every hidden task.
pub const HIDDEN_N_VALID: usize = 256;
/// Families of the hidden grammar pool.
pub const HIDDEN_GRAMMAR_FAMILIES: usize = 40;

/// Primary test (contract section 6.8): sign-flip repetitions. One source of truth: the
/// protocol module, whose `PRIMARY_TEST` string is part of the protocol hash.
pub const SIGN_FLIP_REPS: usize = protocol::PRIMARY_REPS;
/// Primary test: one-sided significance level.
pub const PRIMARY_ALPHA: f64 = protocol::PRIMARY_ALPHA;
/// Primary test: smallest Δ that passes.
pub const PRIMARY_MIN_DELTA: f64 = protocol::PRIMARY_MIN_DELTA;
/// Primary test: largest drop of Δ at F = 32 against F ∈ {8, 16}.
pub const COUPLING_MAX_DROP: f64 = protocol::PRIMARY_COUPLING;

/// n_train of hidden instance k is `N_TRAIN_CYCLE[(k / dims.len()) % 3]` (errata E7).
const N_TRAIN_CYCLE: [usize; 3] = [16, 64, 256];

/// Largest secret file read.
const MAX_SECRET_BYTES: u64 = 64 * 1024;
/// Largest baseline cache read.
const MAX_CACHE_BYTES: u64 = 64 << 20;
/// Longest candidate id accepted.
const MAX_CANDIDATE_ID: usize = 64;
/// Most training sources an ARC request may list.
const MAX_TRAIN_SOURCES: usize = 16;

/// Ledger reasons of refused queries (fixed texts, never request content; a failed ARC
/// load adds the loader's message, which stays in the ledger).
const REFUSED_PROGRAM: &str = "program fails check() (structure or static cost cap)";
const REFUSED_PROTOCOL: &str = "protocol hash does not match this brut-eval";
const REFUSED_BUDGET: &str = "query budget exhausted";
const REFUSED_NO_ARC: &str = "no ARC data configured in the secret file";
const REFUSED_CONTAMINATION: &str = "a training source is forbidden for this eval set";
const REFUSED_ARC_DATA: &str = "ARC data failed to load or to verify against the manifest";
const REFUSED_OVERLAP: &str = "a training pair overlaps the eval set (canonical hash)";

/// Capability: only this module can construct it (private field).
pub struct HiddenToken(());

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnerEvalRequest {
    /// Hash of the run's first archive record (MetaRunStarted).
    pub run_id: String,
    pub candidate_id: String,
    pub protocol_hash: String,
    pub tier: Tier,
    /// The program itself (no file reads by brut-eval).
    pub program: LearnerGenome,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnerEvalResponse {
    pub candidate_id: String,
    pub tier: Tier,
    /// Δ_T rounded DOWN to the step (>= 0.01). The only number released.
    pub delta: f64,
    /// Only for HiddenGrammar: the pre-registered decision (section 6.8). One bit.
    pub decision: Option<Decision>,
    pub queries_used_total: usize,
    pub queries_left_total: usize,
    pub queries_left_run: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArcEvalRequest {
    pub run_id: String,
    pub candidate_id: String,
    pub protocol_hash: String,
    pub set: EvalSet,
    pub program: LearnerGenome,
    /// Every source used to train or select this program.
    pub train_sources: Vec<SourceId>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArcEvalResponse {
    pub candidate_id: String,
    pub set: EvalSet,
    pub n_tasks: usize,
    pub n_test_outputs: usize,
    pub solved_outputs: usize,
    pub official: f64,
    pub kaggle: f64,
    /// Rounded down to 0.01.
    pub bca: Option<f64>,
    pub queries_left_set: usize,
}

/// Lives only inside brut-eval. Deliberately not `Debug`: it holds the secret seed.
pub struct LearnerEvaluator {
    secret_seed: u64,
    budget_total: usize,
    budget_run: usize,
    arc_budget_per_set: usize,
    score_step: f64,
    /// Directory of the secret: ledger, lock and baseline cache live here.
    state_dir: PathBuf,
    arc_root: Option<PathBuf>,
    arc_manifest: Option<PathBuf>,
    settings: EvalSettings,
    backend: Box<dyn Backend>,
    token: HiddenToken,
}

impl LearnerEvaluator {
    /// Open the global learner secret (check_private on file and dir), read caps,
    /// open the ledger next to it. `cli_budget` and `cli_step` are the brut-eval flags:
    /// per-run cap = min(cli_budget, BUDGET_PER_RUN), step = max(cli_step, 0.01, BRUT_HIDDEN_STEP).
    ///
    /// The step must stay below 1, so a released Δ always lies in (-2, 1].
    pub fn open(secret_env: &Path, cli_budget: usize, cli_step: f64) -> Result<LearnerEvaluator> {
        LearnerEvaluator::open_with(secret_env, cli_budget, cli_step, Box::new(Library))
    }

    /// `open` with an explicit backend (tests use a fake one).
    fn open_with(
        secret_env: &Path,
        cli_budget: usize,
        cli_step: f64,
        backend: Box<dyn Backend>,
    ) -> Result<LearnerEvaluator> {
        if !cli_step.is_finite() || !(0.0..1.0).contains(&cli_step) {
            return Err(BrutError::Invalid(format!(
                "score step {cli_step} must be finite and in [0, 1)"
            )));
        }

        let state_dir = secret_dir(secret_env);

        check_private_dir(&state_dir)?;
        check_secret_file(secret_env)?;

        let env = parse_env_file(secret_env)?;

        let Some(secret_seed) = env_value::<u64>(&env, LEARNER_SEED_KEY)? else {
            return Err(BrutError::Invalid(format!(
                "secret file has no {LEARNER_SEED_KEY}"
            )));
        };

        let budget_total =
            env_value::<usize>(&env, BUDGET_TOTAL_KEY)?.unwrap_or(DEFAULT_BUDGET_TOTAL);
        let secret_per_run =
            env_value::<usize>(&env, BUDGET_PER_RUN_KEY)?.unwrap_or(DEFAULT_BUDGET_PER_RUN);
        let arc_budget_per_set =
            env_value::<usize>(&env, ARC_BUDGET_KEY)?.unwrap_or(DEFAULT_ARC_BUDGET_PER_SET);
        let secret_step = env_value::<f64>(&env, HIDDEN_STEP_KEY)?.unwrap_or(0.0);

        if !(0.0..1.0).contains(&secret_step) {
            return Err(BrutError::Invalid(format!(
                "{HIDDEN_STEP_KEY} must be in [0, 1)"
            )));
        }

        let score_step = cli_step.max(MIN_HIDDEN_SCORE_STEP).max(secret_step);

        // Verify the ledger once, under the lock: a tampered ledger fails before any work.
        {
            let _lock = lock_exclusive(&state_dir.join(LOCK_FILE))?;

            Ledger::open(&state_dir)?;
        }

        Ok(LearnerEvaluator {
            secret_seed,
            budget_total,
            budget_run: cli_budget.min(secret_per_run),
            arc_budget_per_set: cli_budget.min(arc_budget_per_set),
            score_step,
            arc_root: env.get(ARC_ROOT_KEY).map(PathBuf::from),
            arc_manifest: env.get(ARC_MANIFEST_KEY).map(PathBuf::from),
            state_dir,
            settings: EvalSettings {
                max_seconds_per_task: EVAL_SECONDS_PER_TASK,
            },
            backend,
            token: HiddenToken(()),
        })
    }

    /// The score step this evaluator rounds to (after applying the floors).
    pub fn score_step(&self) -> f64 {
        self.score_step
    }

    /// Order: validate request and program (check()), protocol hash == protocol_hash()
    /// (else Refused, not charged), budget check + Charge entry under eval.lock, build tier
    /// tasks, run candidate and cached tuned baselines, compute Δ (and decision), quantize,
    /// Release entry, respond.
    pub fn evaluate(&mut self, req: &LearnerEvalRequest) -> Result<LearnerEvalResponse> {
        check_identity(&req.run_id, &req.candidate_id, &req.protocol_hash)?;

        if !req.tier.is_hidden() {
            return Err(BrutError::Invalid(
                "learner requests must name a hidden tier".into(),
            ));
        }

        let meta = QueryMeta::new(
            &req.run_id,
            &req.candidate_id,
            &req.protocol_hash,
            &req.program,
            QueryKind::Learner { tier: req.tier },
        )?;

        if self.backend.check_program(&req.program).is_err() {
            return Err(self.refuse(&meta, REFUSED_PROGRAM, invalid(REFUSED_PROGRAM)));
        }

        if req.protocol_hash != self.backend.protocol_hash() {
            return Err(self.refuse(&meta, REFUSED_PROTOCOL, eval_err(REFUSED_PROTOCOL)));
        }

        // Read the cache before the charge: a broken cache file costs no query.
        let cache = read_cache(&self.cache_path())?;

        let (used_total, used_run) = self.charge(&meta, self.budget_total, self.budget_run)?;

        let tasks = self
            .backend
            .tier_tasks(&self.token, self.secret_seed, req.tier)?;
        let stats = self.score_tier(req, &tasks, cache)?;

        let decision = match req.tier {
            Tier::HiddenGrammar => {
                let tag = format!("test/{}/{}", req.candidate_id, tier_name(req.tier));
                let seed = derive_seed(self.secret_seed, &tag);
                let p = self
                    .backend
                    .sign_flip_pvalue(&stats.family_diffs, SIGN_FLIP_REPS, seed);

                Some(primary_decision(
                    p,
                    stats.delta,
                    stats.delta32,
                    stats.delta816,
                ))
            }
            _ => None,
        };

        let resp = LearnerEvalResponse {
            candidate_id: req.candidate_id.clone(),
            tier: req.tier,
            delta: quantize(stats.delta, self.score_step),
            decision,
            queries_used_total: used_total,
            queries_left_total: self.budget_total.saturating_sub(used_total),
            queries_left_run: self.budget_run.saturating_sub(used_run),
        };

        self.release(&meta, serde_json::to_value(&resp)?)?;

        Ok(resp)
    }

    /// Order: validate, protocol hash, check_contamination, verify_manifest,
    /// canonical-hash overlap between training_pair_hashes(train_sources) and the eval set
    /// (any overlap: Refused), budget check + Charge, solve_pass2 on all tasks (rayon),
    /// Release, respond.
    pub fn evaluate_arc(&mut self, req: &ArcEvalRequest) -> Result<ArcEvalResponse> {
        check_identity(&req.run_id, &req.candidate_id, &req.protocol_hash)?;

        if req.train_sources.is_empty() || req.train_sources.len() > MAX_TRAIN_SOURCES {
            return Err(BrutError::Invalid(format!(
                "train_sources must list 1..={MAX_TRAIN_SOURCES} sources"
            )));
        }

        let meta = QueryMeta::new(
            &req.run_id,
            &req.candidate_id,
            &req.protocol_hash,
            &req.program,
            QueryKind::Arc { set: req.set },
        )?;

        if self.backend.check_program(&req.program).is_err() {
            return Err(self.refuse(&meta, REFUSED_PROGRAM, invalid(REFUSED_PROGRAM)));
        }

        if req.protocol_hash != self.backend.protocol_hash() {
            return Err(self.refuse(&meta, REFUSED_PROTOCOL, eval_err(REFUSED_PROTOCOL)));
        }

        let (Some(root), Some(manifest)) = (&self.arc_root, &self.arc_manifest) else {
            return Err(self.refuse(&meta, REFUSED_NO_ARC, eval_err(REFUSED_NO_ARC)));
        };

        if self
            .backend
            .check_contamination(&req.train_sources, req.set)
            .is_err()
        {
            return Err(self.refuse(
                &meta,
                REFUSED_CONTAMINATION,
                eval_err(REFUSED_CONTAMINATION),
            ));
        }

        // SEC-3: loader errors name eval-set files. They go only into the ledger (in the
        // private secret directory); the caller gets the fixed refusal text.
        let inputs = match self
            .backend
            .load_arc(root, manifest, req.set, &req.train_sources)
        {
            Ok(inputs) => inputs,
            Err(e) => {
                let reason = format!("{REFUSED_ARC_DATA}: {e}");

                return Err(self.refuse(&meta, &reason, eval_err(REFUSED_ARC_DATA)));
            }
        };

        let overlap = inputs
            .tasks
            .iter()
            .flat_map(|t| t.train.iter().chain(t.test.iter()))
            .any(|p| inputs.train_hashes.contains(&self.backend.pair_hash(p)));

        if overlap {
            return Err(self.refuse(&meta, REFUSED_OVERLAP, eval_err(REFUSED_OVERLAP)));
        }

        let (used_set, _) = self.charge(&meta, self.arc_budget_per_set, usize::MAX)?;

        let backend = self.backend.as_ref();
        let settings = &self.settings;
        let results: Vec<ArcResult> = inputs
            .tasks
            .par_iter()
            .map(|t| backend.solve(&req.program, t, settings))
            .collect();

        let n_tasks = inputs.tasks.len();
        let n_test_outputs: usize = inputs.tasks.iter().map(|t| t.test.len()).sum();
        let solved_outputs: usize = inputs
            .tasks
            .iter()
            .zip(&results)
            .map(|(t, r)| {
                r.correct
                    .iter()
                    .take(t.test.len())
                    .filter(|&&ok| ok)
                    .count()
            })
            .sum();
        let scores = backend.arc_scores(&results, n_tasks, n_test_outputs);

        let resp = ArcEvalResponse {
            candidate_id: req.candidate_id.clone(),
            set: req.set,
            n_tasks,
            n_test_outputs,
            solved_outputs,
            official: unit_or_zero(scores.official),
            kaggle: unit_or_zero(scores.kaggle),
            bca: scores
                .bca
                .map(|b| quantize(unit_or_zero(b), MIN_HIDDEN_SCORE_STEP)),
            queries_left_set: self.arc_budget_per_set.saturating_sub(used_set),
        };

        self.release(&meta, serde_json::to_value(&resp)?)?;

        Ok(resp)
    }

    /// Candidate and baseline scores on the tier tasks, then Δ and the primary-test inputs.
    ///
    /// Program baselines come from the cache when their task ids match; MlpMatched depends on
    /// the candidate's cost and is recomputed for every query.
    fn score_tier(
        &self,
        req: &LearnerEvalRequest,
        tasks: &[TaskInstance],
        cache: BaselineCache,
    ) -> Result<TierStats> {
        let backend = self.backend.as_ref();
        let settings = &self.settings;
        let g = &req.program;
        let task_ids: Vec<u64> = tasks.iter().map(|t| t.id).collect();
        let mut fresh = BaselineCache::default();

        let tb = match cache.tuned(&req.protocol_hash) {
            Some(tb) => tb.clone(),
            None => {
                let tb = backend.tuned_baselines(settings);

                fresh.tuned.push(CachedTuned {
                    protocol_hash: req.protocol_hash.clone(),
                    baselines: tb.clone(),
                });

                tb
            }
        };

        let set = backend.matched_set(&tb, backend.step_cost_f16(g));

        if set.is_empty() {
            return Err(eval_err("no compute-matched baseline for this candidate"));
        }

        let candidate: Vec<f32> = tasks
            .par_iter()
            .map(|t| unit_aulc(backend.run(g, t, settings)))
            .collect();

        let mut baseline_scores: Vec<Vec<f32>> = Vec::with_capacity(set.len());

        for &id in &set {
            if id == BaselineId::MlpMatched {
                let scores = tasks
                    .par_iter()
                    .map(|t| unit_aulc(backend.mlp_matched(g, t, &tb.mlp)))
                    .collect();

                baseline_scores.push(scores);

                continue;
            }

            if let Some(scores) =
                cache.result(&req.protocol_hash, &tb.hash, req.tier, id, &task_ids)
            {
                baseline_scores.push(scores.to_vec());

                continue;
            }

            let Some((_, program)) = tb.programs.iter().find(|(b, _)| *b == id) else {
                return Err(eval_err("tuned baselines lack a matched program baseline"));
            };

            let scores: Vec<f32> = tasks
                .par_iter()
                .map(|t| unit_aulc(backend.run(program, t, settings)))
                .collect();

            fresh.results.push(CachedResult {
                protocol_hash: req.protocol_hash.clone(),
                tuned_hash: tb.hash.clone(),
                tier: req.tier,
                baseline: id,
                task_ids: task_ids.clone(),
                aulc: scores.clone(),
            });
            baseline_scores.push(scores);
        }

        if !fresh.tuned.is_empty() || !fresh.results.is_empty() {
            self.store_cache(fresh)?;
        }

        let families = family_index(tasks);
        let widths: Vec<usize> = tasks.iter().map(|t| t.f).collect();

        tier_stats(&families, &widths, &candidate, &baseline_scores)
    }

    fn cache_path(&self) -> PathBuf {
        self.state_dir.join(BASELINE_CACHE_FILE)
    }

    /// Merge new entries into the cache on disk, under the lock (other evaluators may have
    /// written meanwhile). Entries with the same key are replaced.
    fn store_cache(&self, fresh: BaselineCache) -> Result<()> {
        let _lock = self.lock()?;
        let mut cache = read_cache(&self.cache_path())?;

        cache.merge(fresh);

        write_private_atomic(&self.cache_path(), &serde_json::to_vec(&cache)?)
    }

    fn lock(&self) -> Result<std::fs::File> {
        lock_exclusive(&self.state_dir.join(LOCK_FILE))
    }

    /// Log a refused query (not charged) and hand back `err` for the caller.
    /// When the ledger cannot be written, that error wins.
    fn refuse(&self, meta: &QueryMeta, reason: &str, err: BrutError) -> BrutError {
        let logged = (|| -> Result<()> {
            let _lock = self.lock()?;
            let mut ledger = Ledger::open(&self.state_dir)?;

            ledger.append(meta.entry(meta.refused(reason)))?;

            Ok(())
        })();

        match logged {
            Ok(()) => err,
            Err(e) => e,
        }
    }

    /// Under eval.lock: when the total cap or the per-run cap is reached, log a refusal;
    /// else append a Charge. Returns (used total, used by this run) after the charge.
    fn charge(&self, meta: &QueryMeta, cap_total: usize, cap_run: usize) -> Result<(usize, usize)> {
        let _lock = self.lock()?;
        let mut ledger = Ledger::open(&self.state_dir)?;
        let used_total = meta.pool_charges(&ledger, None);
        let used_run = meta.pool_charges(&ledger, Some(&meta.run_id));

        if used_total >= cap_total || used_run >= cap_run {
            ledger.append(meta.entry(meta.refused(REFUSED_BUDGET)))?;

            let scope = if used_total >= cap_total {
                format!("{used_total} of {cap_total} used for this protocol")
            } else {
                format!("{used_run} of {cap_run} used by this run")
            };

            return Err(BrutError::Eval(format!("{REFUSED_BUDGET} ({scope})")));
        }

        ledger.append(meta.entry(LedgerKind::Charge {
            query: meta.query.clone(),
        }))?;

        Ok((used_total + 1, used_run + 1))
    }

    /// Log exactly what is released, under the lock. No response leaves without this entry.
    fn release(&self, meta: &QueryMeta, released: serde_json::Value) -> Result<()> {
        let _lock = self.lock()?;
        let mut ledger = Ledger::open(&self.state_dir)?;

        ledger.append(meta.entry(LedgerKind::Release {
            query: meta.query.clone(),
            released,
        }))?;

        Ok(())
    }
}

/// Create <dir> (0700) and <dir>/hidden.env (0600) with a fresh random seed and the
/// default caps. Refuses to overwrite. Returns the file path. `arc_root`/`manifest` are
/// written as ARC_ROOT_KEY/ARC_MANIFEST_KEY when given (made absolute).
pub fn create_learner_secret(
    dir: &Path,
    arc_root: Option<&Path>,
    manifest: Option<&Path>,
) -> Result<PathBuf> {
    let arc_root = arc_root.map(env_path).transpose()?;
    let manifest = manifest.map(env_path).transpose()?;

    create_private_dir(dir)?;
    check_private_dir(dir)?;

    let path = dir.join(LEARNER_SECRET_FILE);
    let seed: u64 = rand::rng().random();
    let mut text = format!(
        "# Secret of the learner hidden evaluator. Only brut-eval reads this file.\n\
         # Do not share, commit or print it.\n\
         {LEARNER_SEED_KEY}={seed}\n\
         {BUDGET_TOTAL_KEY}={DEFAULT_BUDGET_TOTAL}\n\
         {BUDGET_PER_RUN_KEY}={DEFAULT_BUDGET_PER_RUN}\n\
         {HIDDEN_STEP_KEY}={MIN_HIDDEN_SCORE_STEP}\n\
         {ARC_BUDGET_KEY}={DEFAULT_ARC_BUDGET_PER_SET}\n"
    );

    if let Some(root) = &arc_root {
        text.push_str(&format!("{ARC_ROOT_KEY}={root}\n"));
    }

    if let Some(manifest) = &manifest {
        text.push_str(&format!("{ARC_MANIFEST_KEY}={manifest}\n"));
    }

    let mut options = std::fs::OpenOptions::new();

    options.write(true).create_new(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        options.mode(0o600);
    }

    let mut file = options.open(&path)?;

    file.write_all(text.as_bytes())?;
    file.sync_all()?;

    Ok(path)
}

/// Orchestrator side: send one request through hidden_eval::request_eval_json.
/// The answer must be for the same candidate and tier, carry a finite Δ in (-2, 1], and a
/// decision bit exactly when the tier is HiddenGrammar.
pub fn request_learner_eval(
    eval_bin: &Path,
    secret_env: &Path,
    budget: usize,
    step: f64,
    req: &LearnerEvalRequest,
    stop_file: Option<&Path>,
) -> Result<LearnerEvalResponse> {
    let body = AnyEvalRequest::Learner(req.clone()).to_json()?;
    let timeout = Duration::from_secs_f64(LEARNER_EVAL_TIMEOUT_SECONDS);
    let out = request_eval_json(
        eval_bin,
        secret_env,
        budget,
        step,
        body.as_bytes(),
        timeout,
        stop_file,
    )?;

    let resp: LearnerEvalResponse = serde_json::from_slice(&out)
        .map_err(|_| eval_err("bad evaluator output (not a learner response)"))?;

    check_learner_response(req, &resp)?;

    Ok(resp)
}

/// Orchestrator side: send one ARC request through hidden_eval::request_eval_json.
/// The answer must be for the same candidate and set, with counts and scores in range.
pub fn request_arc_eval(
    eval_bin: &Path,
    secret_env: &Path,
    budget: usize,
    step: f64,
    req: &ArcEvalRequest,
    stop_file: Option<&Path>,
) -> Result<ArcEvalResponse> {
    let body = AnyEvalRequest::Arc(req.clone()).to_json()?;
    let timeout = Duration::from_secs_f64(ARC_EVAL_TIMEOUT_SECONDS);
    let out = request_eval_json(
        eval_bin,
        secret_env,
        budget,
        step,
        body.as_bytes(),
        timeout,
        stop_file,
    )?;

    let resp: ArcEvalResponse = serde_json::from_slice(&out)
        .map_err(|_| eval_err("bad evaluator output (not an ARC response)"))?;

    check_arc_response(req, &resp)?;

    Ok(resp)
}

fn check_learner_response(req: &LearnerEvalRequest, resp: &LearnerEvalResponse) -> Result<()> {
    if resp.candidate_id != req.candidate_id || resp.tier != req.tier {
        return Err(eval_err(
            "evaluator answered for a different candidate or tier",
        ));
    }

    if !resp.delta.is_finite() || resp.delta <= -2.0 || resp.delta > 1.0 {
        return Err(eval_err("evaluator returned an impossible delta"));
    }

    if resp.decision.is_some() != (req.tier == Tier::HiddenGrammar) {
        return Err(eval_err(
            "evaluator returned a decision bit for the wrong tier",
        ));
    }

    Ok(())
}

fn check_arc_response(req: &ArcEvalRequest, resp: &ArcEvalResponse) -> Result<()> {
    if resp.candidate_id != req.candidate_id || resp.set != req.set {
        return Err(eval_err(
            "evaluator answered for a different candidate or set",
        ));
    }

    let in_range = resp.solved_outputs <= resp.n_test_outputs
        && is_unit(resp.official)
        && is_unit(resp.kaggle)
        && resp.bca.is_none_or(is_unit);

    if !in_range {
        return Err(eval_err("evaluator returned impossible ARC scores"));
    }

    Ok(())
}

/// Families, instances per family and widths of one hidden tier (contract section 5.15).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TierShape {
    families: usize,
    instances: usize,
    /// "Alternating" tiers use [8, 16], "cycling" tiers [8, 16, 32].
    dims: &'static [usize],
}

fn tier_shape(tier: Tier) -> Option<TierShape> {
    let shape = |families, instances, dims| {
        Some(TierShape {
            families,
            instances,
            dims,
        })
    };

    match tier {
        Tier::HiddenInstance => shape(10, 10, &[8, 16]),
        Tier::HiddenNear => shape(5, 20, &[8, 16, 32]),
        Tier::HiddenFar => shape(4, 25, &[8, 16, 32]),
        Tier::HiddenGrammar => shape(HIDDEN_GRAMMAR_FAMILIES, 5, &[8, 16, 32]),
        Tier::Search | Tier::Select => None,
    }
}

/// Serde (snake_case) name of a tier, as used in seed tags (errata E13).
fn tier_name(tier: Tier) -> &'static str {
    match tier {
        Tier::Search => "search",
        Tier::Select => "select",
        Tier::HiddenInstance => "hidden_instance",
        Tier::HiddenNear => "hidden_near",
        Tier::HiddenFar => "hidden_far",
        Tier::HiddenGrammar => "hidden_grammar",
    }
}

/// One hidden instance before it is built.
#[derive(Clone, Debug, PartialEq, Eq)]
struct InstanceSpec {
    /// Index into the tier's family list.
    family: usize,
    /// Instance number inside the family.
    index: usize,
    seed: u64,
    f: usize,
    n_train: usize,
}

/// Seeds and sizes of every instance of a tier, in output order (errata E7 and E13):
/// k runs over families in order and instances inside them; f = dims[k % len],
/// n_train = [16, 64, 256][(k / len) % 3], seed = derive_seed(secret, "tier/<tier>/<family>/<i>").
fn tier_plan(
    _token: &HiddenToken,
    secret_seed: u64,
    tier: Tier,
    families: &[&str],
) -> Result<Vec<InstanceSpec>> {
    let Some(shape) = tier_shape(tier) else {
        return Err(BrutError::Invalid("not a hidden tier".into()));
    };

    if families.len() != shape.families {
        return Err(BrutError::Invalid(format!(
            "tier {} needs {} families, the library gave {}",
            tier_name(tier),
            shape.families,
            families.len()
        )));
    }

    let len = shape.dims.len();
    let mut plan = Vec::with_capacity(shape.families * shape.instances);

    for (family, name) in families.iter().enumerate() {
        for index in 0..shape.instances {
            let k = plan.len();
            let tag = format!("tier/{}/{name}/{index}", tier_name(tier));

            plan.push(InstanceSpec {
                family,
                index,
                seed: derive_seed(secret_seed, &tag),
                f: shape.dims[k % len],
                n_train: N_TRAIN_CYCLE[(k / len) % N_TRAIN_CYCLE.len()],
            });
        }
    }

    Ok(plan)
}

/// Instance parameters: f and n_train from the plan, level from `sample_params` with
/// single-value choices and the instance seed (errata E7).
fn instance_params(spec: &InstanceSpec) -> InstanceParams {
    let mut rng = ChaCha8Rng::seed_from_u64(spec.seed);
    let drawn = families::sample_params(&mut rng, &[spec.f], &[spec.n_train], HIDDEN_N_VALID);

    InstanceParams {
        f: spec.f,
        n_train: spec.n_train,
        n_valid: HIDDEN_N_VALID,
        level: drawn.level,
    }
}

/// Hidden tier tasks (contract table in section 5.15). Private; requires the capability.
fn hidden_tier_tasks(
    token: &HiddenToken,
    secret_seed: u64,
    tier: Tier,
) -> Result<Vec<TaskInstance>> {
    let tasks: Vec<TaskInstance> = match tier {
        Tier::HiddenGrammar => {
            let pool = hidden_grammar_pool(token, secret_seed)?;
            let names: Vec<&str> = pool.iter().map(|g| g.id.0.as_str()).collect();
            let plan = tier_plan(token, secret_seed, tier, &names)?;

            plan.par_iter()
                .map(|s| grammar::grammar_instance(&pool[s.family], s.seed, &instance_params(s)))
                .collect()
        }
        Tier::HiddenInstance | Tier::HiddenNear | Tier::HiddenFar => {
            let source = match tier {
                Tier::HiddenInstance => Tier::Search,
                other => other,
            };
            let fams = families::hand_families(source);
            let names: Vec<&str> = fams.iter().map(|f| f.0.as_str()).collect();
            let plan = tier_plan(token, secret_seed, tier, &names)?;

            plan.par_iter()
                .map(|s| families::hand_instance(&fams[s.family], s.seed, &instance_params(s)))
                .collect::<Result<Vec<_>>>()?
        }
        Tier::Search | Tier::Select => {
            return Err(BrutError::Invalid("not a hidden tier".into()));
        }
    };

    // Errata E3: every task must fit the interpreter widths.
    let fits = tasks
        .iter()
        .all(|t| SUPPORTED_F.contains(&t.f) && t.kind.k_out() <= t.f);

    if !fits {
        return Err(eval_err(
            "a hidden task does not fit the interpreter widths",
        ));
    }

    Ok(tasks)
}

/// The 40 hidden grammar families. Private; requires the capability.
/// Signatures avoid both public pools; the length histogram matches the SEARCH pool. No
/// family is an exact twin (same kind and semantic key) of a public family or of another
/// hidden family (SCI-R4).
fn hidden_grammar_pool(token: &HiddenToken, secret_seed: u64) -> Result<Vec<GrammarFamily>> {
    let _ = token;

    let search = grammar::public_pool(Tier::Search);
    let select = grammar::public_pool(Tier::Select);
    let exclude: BTreeSet<String> = search
        .iter()
        .chain(select.iter())
        .map(|g| g.signature.clone())
        .collect();
    let lengths: Vec<usize> = search.iter().map(|g| grammar::length(&g.expr)).collect();

    grammar::disjoint_pool_semantic(
        derive_seed(secret_seed, "grammar/hidden"),
        HIDDEN_GRAMMAR_FAMILIES,
        &exclude,
        &grammar::public_semantic_keys(),
        &lengths,
        "grammar/hidden",
    )
}

/// Δ_T and the inputs of the primary test (contract section 6.8).
#[derive(Clone, Debug, PartialEq)]
struct TierStats {
    /// S_T(g) − S_T(b*), with S_T the mean over families of the mean over their tasks.
    delta: f64,
    /// Index of b* in the baseline list (the first one on ties).
    best: usize,
    /// d_f per family: mean over its tasks of A_t(g) − A_t(b*).
    family_diffs: Vec<f64>,
    /// Δ restricted to F = 32 tasks, against the same b* (None without such tasks).
    delta32: Option<f64>,
    /// Δ restricted to F ∈ {8, 16} tasks, against the same b*.
    delta816: Option<f64>,
}

/// Family index of every task, numbered in order of first appearance.
fn family_index(tasks: &[TaskInstance]) -> Vec<usize> {
    let mut ids: HashMap<&str, usize> = HashMap::new();

    tasks
        .iter()
        .map(|t| {
            let next = ids.len();

            *ids.entry(t.family.0.as_str()).or_insert(next)
        })
        .collect()
}

/// Mean over families of the mean over their kept tasks; families without kept tasks are
/// skipped. None when no task is kept.
fn family_mean(
    groups: &[Vec<usize>],
    value: impl Fn(usize) -> f64,
    keep: impl Fn(usize) -> bool,
) -> Option<f64> {
    let means: Vec<f64> = groups
        .iter()
        .filter_map(|members| {
            let kept: Vec<usize> = members.iter().copied().filter(|&i| keep(i)).collect();

            if kept.is_empty() {
                return None;
            }

            Some(kept.iter().map(|&i| value(i)).sum::<f64>() / kept.len() as f64)
        })
        .collect();

    if means.is_empty() {
        return None;
    }

    Some(means.iter().sum::<f64>() / means.len() as f64)
}

/// Δ_T = S_T(g) − max_b S_T(b), b* = argmax (first on ties), d_f, Δ32 and Δ816.
fn tier_stats(
    families: &[usize],
    widths: &[usize],
    candidate: &[f32],
    baselines: &[Vec<f32>],
) -> Result<TierStats> {
    let n = candidate.len();
    let aligned = n > 0
        && families.len() == n
        && widths.len() == n
        && !baselines.is_empty()
        && baselines.iter().all(|b| b.len() == n);

    if !aligned {
        return Err(eval_err("tier scores do not line up with the tier tasks"));
    }

    let n_families = families.iter().copied().max().map_or(0, |m| m + 1);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n_families];

    for (i, &f) in families.iter().enumerate() {
        groups[f].push(i);
    }

    let all = |_: usize| true;
    let s_candidate = family_mean(&groups, |i| f64::from(candidate[i]), all).unwrap_or(0.0);

    let mut best = 0;
    let mut best_score = f64::NEG_INFINITY;

    for (b, scores) in baselines.iter().enumerate() {
        let s = family_mean(&groups, |i| f64::from(scores[i]), all).unwrap_or(0.0);

        if s > best_score {
            best = b;
            best_score = s;
        }
    }

    let diff = |i: usize| f64::from(candidate[i]) - f64::from(baselines[best][i]);

    let family_diffs: Vec<f64> = groups
        .iter()
        .filter(|members| !members.is_empty())
        .map(|members| members.iter().map(|&i| diff(i)).sum::<f64>() / members.len() as f64)
        .collect();

    Ok(TierStats {
        delta: s_candidate - best_score,
        best,
        family_diffs,
        delta32: family_mean(&groups, diff, |i| widths[i] == 32),
        delta816: family_mean(&groups, diff, |i| widths[i] == 8 || widths[i] == 16),
    })
}

/// Pass iff p < 0.05 and Δ_T >= 0.01 and Δ32 >= Δ816 − 0.05 (contract section 6.8).
/// A missing width class or a NaN p fails. The rule itself lives in the protocol module.
fn primary_decision(p: f64, delta: f64, delta32: Option<f64>, delta816: Option<f64>) -> Decision {
    match (delta32, delta816) {
        (Some(d32), Some(d816)) => protocol::primary_decision(p, delta, d32, d816),
        _ => Decision::Fail,
    }
}

/// ARC eval tasks plus the canonical hashes of every training pair of the request's sources.
struct ArcInputs {
    tasks: Vec<ArcTask>,
    train_hashes: BTreeSet<[u8; 32]>,
}

/// Aggregate ARC scores of one program on one set.
struct ArcScores {
    official: f64,
    kaggle: f64,
    bca: Option<f64>,
}

/// What the evaluator needs from the learner library and the ARC loaders.
///
/// `Library` calls the real modules. Unit tests swap in a small fake, so the budget,
/// ledger, cache and release logic runs on hand-made data.
trait Backend: Send + Sync {
    fn protocol_hash(&self) -> String;
    /// `check()` with the default DSL limits (structure and static cost cap).
    fn check_program(&self, g: &LearnerGenome) -> Result<()>;
    fn step_cost_f16(&self, g: &LearnerGenome) -> u64;
    fn tier_tasks(
        &self,
        token: &HiddenToken,
        secret_seed: u64,
        tier: Tier,
    ) -> Result<Vec<TaskInstance>>;
    fn run(&self, g: &LearnerGenome, task: &TaskInstance, settings: &EvalSettings) -> TaskResult;
    fn tuned_baselines(&self, settings: &EvalSettings) -> TunedBaselines;
    fn matched_set(&self, tb: &TunedBaselines, step_cost_f16: u64) -> Vec<BaselineId>;
    /// MlpMatched for candidate g: the budget is g's training cost on the task (COR-1).
    fn mlp_matched(&self, g: &LearnerGenome, task: &TaskInstance, recipe: &MlpRecipe)
    -> TaskResult;
    fn sign_flip_pvalue(&self, d: &[f64], reps: usize, seed: u64) -> f64;
    fn check_contamination(&self, sources: &[SourceId], set: EvalSet) -> Result<()>;
    fn load_arc(
        &self,
        root: &Path,
        manifest: &Path,
        set: EvalSet,
        sources: &[SourceId],
    ) -> Result<ArcInputs>;
    fn pair_hash(&self, p: &ArcPair) -> [u8; 32];
    fn solve(&self, g: &LearnerGenome, t: &ArcTask, settings: &EvalSettings) -> ArcResult;
    fn arc_scores(&self, results: &[ArcResult], n_tasks: usize, n_outputs: usize) -> ArcScores;
}

/// The real learner library and ARC loaders.
struct Library;

impl Backend for Library {
    fn protocol_hash(&self) -> String {
        protocol::protocol_hash()
    }

    fn check_program(&self, g: &LearnerGenome) -> Result<()> {
        g.check(&DslLimits::default())
    }

    fn step_cost_f16(&self, g: &LearnerGenome) -> u64 {
        g.step_cost(16)
    }

    fn tier_tasks(
        &self,
        token: &HiddenToken,
        secret_seed: u64,
        tier: Tier,
    ) -> Result<Vec<TaskInstance>> {
        hidden_tier_tasks(token, secret_seed, tier)
    }

    fn run(&self, g: &LearnerGenome, task: &TaskInstance, settings: &EvalSettings) -> TaskResult {
        protocol::run_learner(g, task, 0, settings, &|| false)
    }

    fn tuned_baselines(&self, settings: &EvalSettings) -> TunedBaselines {
        baselines::tuned_baselines(settings)
    }

    fn matched_set(&self, tb: &TunedBaselines, step_cost_f16: u64) -> Vec<BaselineId> {
        baselines::compute_matched_set(tb, step_cost_f16)
    }

    fn mlp_matched(
        &self,
        g: &LearnerGenome,
        task: &TaskInstance,
        recipe: &MlpRecipe,
    ) -> TaskResult {
        baselines::mlp_matched_training(task, recipe, TrainingCost::of(g, task.f))
    }

    fn sign_flip_pvalue(&self, d: &[f64], reps: usize, seed: u64) -> f64 {
        stats::sign_flip_pvalue(d, reps, seed)
    }

    fn check_contamination(&self, sources: &[SourceId], set: EvalSet) -> Result<()> {
        arc_data::check_contamination(sources, set)
    }

    fn load_arc(
        &self,
        root: &Path,
        manifest: &Path,
        set: EvalSet,
        sources: &[SourceId],
    ) -> Result<ArcInputs> {
        let m = arc_data::load_manifest(manifest)?;

        arc_data::verify_manifest(root, &m)?;

        let train_hashes = arc_data::training_pair_hashes(root, &m, sources)?;
        let tasks = arc_data::load_eval_set(root, &m, set)?;

        Ok(ArcInputs {
            tasks,
            train_hashes,
        })
    }

    fn pair_hash(&self, p: &ArcPair) -> [u8; 32] {
        arc_cells::canonical_pair_hash(p)
    }

    fn solve(&self, g: &LearnerGenome, t: &ArcTask, settings: &EvalSettings) -> ArcResult {
        arc_cells::solve_pass2(g, t, settings, &|| false)
    }

    fn arc_scores(&self, results: &[ArcResult], n_tasks: usize, n_outputs: usize) -> ArcScores {
        ArcScores {
            official: arc_cells::score_official(results, n_tasks),
            kaggle: arc_cells::score_kaggle(results, n_outputs),
            bca: arc_cells::mean_bca(results),
        }
    }
}

/// `<state dir>/baseline_cache.json`: tuned baselines per protocol and their scores per
/// (protocol, tier, baseline). Never charged. Written 0600 under the lock.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BaselineCache {
    tuned: Vec<CachedTuned>,
    results: Vec<CachedResult>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CachedTuned {
    protocol_hash: String,
    baselines: TunedBaselines,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CachedResult {
    protocol_hash: String,
    /// `TunedBaselines::hash` the scores were computed with.
    tuned_hash: String,
    tier: Tier,
    baseline: BaselineId,
    /// Task ids in tier order; a mismatch makes the entry stale (recomputed, replaced).
    task_ids: Vec<u64>,
    aulc: Vec<f32>,
}

impl BaselineCache {
    fn tuned(&self, protocol_hash: &str) -> Option<&TunedBaselines> {
        self.tuned
            .iter()
            .find(|t| t.protocol_hash == protocol_hash)
            .map(|t| &t.baselines)
    }

    fn result(
        &self,
        protocol_hash: &str,
        tuned_hash: &str,
        tier: Tier,
        baseline: BaselineId,
        task_ids: &[u64],
    ) -> Option<&[f32]> {
        self.results
            .iter()
            .find(|r| {
                r.protocol_hash == protocol_hash
                    && r.tuned_hash == tuned_hash
                    && r.tier == tier
                    && r.baseline == baseline
                    && r.task_ids == task_ids
            })
            .map(|r| r.aulc.as_slice())
    }

    /// Add `fresh`; an entry with the same key replaces the old one.
    fn merge(&mut self, fresh: BaselineCache) {
        for t in fresh.tuned {
            self.tuned
                .retain(|old| old.protocol_hash != t.protocol_hash);
            self.tuned.push(t);
        }

        for r in fresh.results {
            self.results.retain(|old| {
                (&old.protocol_hash, old.tier, old.baseline)
                    != (&r.protocol_hash, r.tier, r.baseline)
            });
            self.results.push(r);
        }
    }

    /// Hashes verify, scores are finite and in [0, 1], one score per task id.
    fn is_consistent(&self) -> bool {
        let tuned_ok = self
            .tuned
            .iter()
            .all(|t| tuned_hash(&t.baselines).is_ok_and(|h| h == t.baselines.hash));
        let results_ok = self.results.iter().all(|r| {
            r.aulc.len() == r.task_ids.len() && r.aulc.iter().all(|&a| (0.0..=1.0).contains(&a))
        });

        tuned_ok && results_ok
    }
}

/// SHA-256 of the canonical JSON of `tb` with its hash field set to "" (contract section 4).
fn tuned_hash(tb: &TunedBaselines) -> Result<String> {
    let mut blank = tb.clone();

    blank.hash = String::new();

    Ok(sha256_hex(serde_json::to_string(&blank)?.as_bytes()))
}

/// Read the cache; a missing file is an empty cache. A file that is not a regular file, too
/// large, unparsable or inconsistent is an Integrity error (delete it to rebuild).
fn read_cache(path: &Path) -> Result<BaselineCache> {
    let meta = match std::fs::symlink_metadata(path) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(BaselineCache::default());
        }
        Err(e) => return Err(e.into()),
    };

    let broken = || {
        BrutError::Integrity(format!(
            "{} is damaged; delete it to rebuild the baseline cache",
            path.display()
        ))
    };

    if !meta.is_file() || meta.len() > MAX_CACHE_BYTES {
        return Err(broken());
    }

    let bytes = read_limited_bytes(std::fs::File::open(path)?, MAX_CACHE_BYTES)?;
    let cache: BaselineCache = serde_json::from_slice(&bytes).map_err(|_| broken())?;

    if !cache.is_consistent() {
        return Err(broken());
    }

    Ok(cache)
}

/// Identity of one query as written into the ledger.
struct QueryMeta {
    run_id: String,
    candidate_id: String,
    protocol_hash: String,
    /// SHA-256 of the program's canonical JSON (serde_json::to_string of the genome).
    program_sha256: String,
    query: QueryKind,
}

impl QueryMeta {
    fn new(
        run_id: &str,
        candidate_id: &str,
        protocol_hash: &str,
        program: &LearnerGenome,
        query: QueryKind,
    ) -> Result<QueryMeta> {
        Ok(QueryMeta {
            run_id: run_id.to_string(),
            candidate_id: candidate_id.to_string(),
            protocol_hash: protocol_hash.to_string(),
            program_sha256: sha256_hex(serde_json::to_string(program)?.as_bytes()),
            query,
        })
    }

    fn entry(&self, kind: LedgerKind) -> LedgerEntry {
        LedgerEntry {
            seq: 0,
            unix_ms: now_unix_ms(),
            prev_hash: String::new(),
            protocol_hash: self.protocol_hash.clone(),
            run_id: self.run_id.clone(),
            candidate_id: self.candidate_id.clone(),
            program_sha256: self.program_sha256.clone(),
            kind,
            hash: String::new(),
        }
    }

    fn refused(&self, reason: &str) -> LedgerKind {
        LedgerKind::Refused {
            query: self.query.clone(),
            reason: reason.to_string(),
        }
    }

    /// Charges in this query's budget pool: all learner tiers together, or one ARC set.
    fn pool_charges(&self, ledger: &Ledger, run: Option<&str>) -> usize {
        match self.query {
            QueryKind::Learner { .. } => ledger.learner_charges(&self.protocol_hash, run),
            QueryKind::Arc { .. } => ledger.charges(&self.protocol_hash, run, &self.query),
        }
    }
}

/// Run id and protocol hash: 64 lower-case hex characters. Candidate id: 1..=64 characters
/// of [A-Za-z0-9_.:-]. Checked before anything is logged; errors do not quote the values.
fn check_identity(run_id: &str, candidate_id: &str, protocol_hash: &str) -> Result<()> {
    if !is_hex64(run_id) {
        return Err(invalid("run_id must be 64 lower-case hex characters"));
    }

    if !is_hex64(protocol_hash) {
        return Err(invalid(
            "protocol_hash must be 64 lower-case hex characters",
        ));
    }

    let candidate_ok = !candidate_id.is_empty()
        && candidate_id.len() <= MAX_CANDIDATE_ID
        && candidate_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'-'));

    if !candidate_ok {
        return Err(invalid(
            "candidate_id must be 1..=64 characters of [A-Za-z0-9_.:-]",
        ));
    }

    Ok(())
}

/// Directory that holds the secret (and the ledger, lock and cache).
fn secret_dir(secret_env: &Path) -> PathBuf {
    match secret_env.parent() {
        Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

/// The secret directory must be a directory closed to group and others.
fn check_private_dir(dir: &Path) -> Result<()> {
    let meta = std::fs::metadata(dir)?;

    if !meta.is_dir() {
        return Err(BrutError::Invalid(format!(
            "{} is not a directory",
            dir.display()
        )));
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mode = meta.permissions().mode();

        if mode & 0o077 != 0 {
            return Err(BrutError::Invalid(format!(
                "secret directory {} is open to others (mode {:o}); run chmod 700",
                dir.display(),
                mode & 0o777
            )));
        }
    }

    Ok(())
}

/// The secret must be a private regular file of at most 64 KiB (no device, no FIFO).
fn check_secret_file(path: &Path) -> Result<()> {
    let meta = std::fs::metadata(path)?;

    if !meta.is_file() || meta.len() > MAX_SECRET_BYTES {
        return Err(BrutError::Invalid(format!(
            "secret file {} must be a regular file of at most {MAX_SECRET_BYTES} bytes",
            path.display()
        )));
    }

    check_private(path)
}

/// Parse an optional key of the secret file. The error names the key, never the value.
fn env_value<T: FromStr>(env: &HashMap<String, String>, key: &str) -> Result<Option<T>> {
    match env.get(key) {
        None => Ok(None),
        Some(text) => text
            .parse()
            .map(Some)
            .map_err(|_| BrutError::Invalid(format!("{key} in the secret file is not valid"))),
    }
}

/// An absolute path that survives the `.env` format unchanged: UTF-8, no control
/// characters, no surrounding blanks or quotes. Anything else could change other keys.
fn env_path(path: &Path) -> Result<String> {
    let absolute = std::path::absolute(path)?;

    let Some(text) = absolute.to_str() else {
        return Err(invalid("ARC path is not valid UTF-8"));
    };

    let safe = !text.chars().any(char::is_control)
        && text.trim() == text
        && !text.starts_with(['"', '\''])
        && !text.ends_with(['"', '\'']);

    if !safe {
        return Err(invalid(
            "ARC path contains control characters, surrounding blanks or quotes",
        ));
    }

    Ok(text.to_string())
}

fn is_unit(x: f64) -> bool {
    x.is_finite() && (0.0..=1.0).contains(&x)
}

fn unit_or_zero(x: f64) -> f64 {
    if is_unit(x) { x } else { 0.0 }
}

/// AULC of a task result, forced into [0, 1] (non-finite counts as 0).
fn unit_aulc(r: TaskResult) -> f32 {
    if r.aulc.is_finite() {
        r.aulc.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn invalid(message: &str) -> BrutError {
    BrutError::Invalid(message.to_string())
}

fn eval_err(message: &str) -> BrutError {
    BrutError::Eval(message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use sha2::{Digest, Sha256};

    use crate::genome::Activation;
    use crate::learner::types::{FamilyId, Grid, Group, Instr, Labels, LearnKind, OpCode};

    /// A distinctive secret seed: no response, message or file may contain it.
    const SEED: u64 = 12_345_678_901_234_567_891;

    const CANDIDATE: &str = "3fa1c2d4e5f60718";

    fn ph() -> String {
        "c".repeat(64)
    }

    fn run_id(n: u64) -> String {
        format!("{n:064x}")
    }

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brut-lh-{name}-{}", std::process::id()));

        let _ = std::fs::remove_dir_all(&dir);
        create_private_dir(&dir).expect("temp dir");

        dir
    }

    /// Write a file with mode 0600.
    fn write_private(path: &Path, text: &str) {
        std::fs::write(path, text).expect("write");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
        }
    }

    /// `<dir>/hidden.env` with the test seed plus `extra` lines.
    fn write_secret(dir: &Path, extra: &str) -> PathBuf {
        let path = dir.join(LEARNER_SECRET_FILE);

        write_private(&path, &format!("{LEARNER_SEED_KEY}={SEED}\n{extra}"));

        path
    }

    fn arc_keys(budget: usize) -> String {
        format!(
            "{ARC_ROOT_KEY}=/nonexistent/arc\n\
             {ARC_MANIFEST_KEY}=/nonexistent/arc/manifest.json\n\
             {ARC_BUDGET_KEY}={budget}\n"
        )
    }

    fn instr(op: OpCode, a: u8, b: u8, out: u8, c: f32) -> Instr {
        Instr { op, a, b, out, c }
    }

    /// Fake learners are told apart by the length of their Learn program.
    fn genome(learn_len: usize, setup_len: usize) -> LearnerGenome {
        LearnerGenome {
            setup: vec![instr(OpCode::SConst, 0, 0, 3, 0.5); setup_len],
            predict: vec![instr(OpCode::MatVec, 0, 0, 1, 0.0)],
            learn: vec![instr(OpCode::VSub, 2, 1, 3, 0.0); learn_len],
        }
    }

    fn candidate() -> LearnerGenome {
        genome(3, 1)
    }

    fn learner_req(run: u64, tier: Tier) -> LearnerEvalRequest {
        LearnerEvalRequest {
            run_id: run_id(run),
            candidate_id: CANDIDATE.into(),
            protocol_hash: ph(),
            tier,
            program: candidate(),
        }
    }

    fn arc_req(set: EvalSet, sources: Vec<SourceId>) -> ArcEvalRequest {
        ArcEvalRequest {
            run_id: run_id(1),
            candidate_id: CANDIDATE.into(),
            protocol_hash: ph(),
            set,
            program: candidate(),
            train_sources: sources,
        }
    }

    #[derive(Default)]
    struct Counters {
        runs: AtomicUsize,
        tunes: AtomicUsize,
        mlp: AtomicUsize,
        solves: AtomicUsize,
        last_d_len: AtomicUsize,
    }

    /// Hand-made stand-in for the learner library: fixed scores, plain-data tasks.
    struct Fake {
        protocol: String,
        counters: Arc<Counters>,
        p_value: f64,
        /// The candidate scores less on F = 32 tasks (width coupling).
        drop_at_32: bool,
        fail_tasks: bool,
        overlap: bool,
        /// load_arc fails like a damaged data file (its message names the file).
        broken_data: bool,
    }

    fn fake() -> (Fake, Arc<Counters>) {
        let counters = Arc::new(Counters::default());
        let fake = Fake {
            protocol: ph(),
            counters: counters.clone(),
            p_value: 0.01,
            drop_at_32: false,
            fail_tasks: false,
            overlap: false,
            broken_data: false,
        };

        (fake, counters)
    }

    fn result(aulc: f32) -> TaskResult {
        TaskResult {
            q: [aulc; 3],
            aulc,
            abort: None,
            flops: 0,
            steps_done: 384,
        }
    }

    fn fake_task(name: &str, s: &InstanceSpec) -> TaskInstance {
        TaskInstance {
            id: derive_seed(s.seed, "fake-id"),
            family: FamilyId(name.to_string()),
            group: Group::Grammar,
            f: s.f,
            kind: LearnKind::Regression,
            x_train: Vec::new(),
            y_train: Labels::Values(Vec::new()),
            y_train_late: None,
            drift_at: None,
            x_valid: Vec::new(),
            y_valid: Labels::Values(Vec::new()),
            n_train: s.n_train,
            n_valid: HIDDEN_N_VALID,
            steps: 384,
            seed: s.seed,
            changed_mask: None,
        }
    }

    fn grid(v: u8) -> Grid {
        Grid {
            h: 1,
            w: 2,
            cells: vec![v, v],
        }
    }

    fn pair(a: u8, b: u8) -> ArcPair {
        ArcPair {
            input: grid(a),
            output: grid(b),
        }
    }

    fn arc_task(id: &str, tests: usize) -> ArcTask {
        ArcTask {
            id: id.to_string(),
            train: vec![pair(1, 2)],
            test: (0..tests).map(|j| pair(3 + j as u8, 4)).collect(),
        }
    }

    fn fake_tuned() -> TunedBaselines {
        let mut tb = TunedBaselines {
            programs: vec![
                (BaselineId::LinearSgd, genome(0, 1)),
                (BaselineId::TwoLayerNn, genome(1, 1)),
                (BaselineId::Prototype, genome(0, 2)),
                (BaselineId::Perceptron, genome(0, 3)),
            ],
            mlp: MlpRecipe {
                hidden: 16,
                activation: Activation::Relu,
                lr: 0.01,
                l2: 0.0,
            },
            hash: String::new(),
        };

        tb.hash = tuned_hash(&tb).expect("hash");

        tb
    }

    impl Backend for Fake {
        fn protocol_hash(&self) -> String {
            self.protocol.clone()
        }

        fn check_program(&self, g: &LearnerGenome) -> Result<()> {
            let finite = g
                .setup
                .iter()
                .chain(&g.predict)
                .chain(&g.learn)
                .all(|i| i.c.is_finite());

            if g.setup.len() > 4 || !finite {
                return Err(BrutError::Budget("fake: over cost".into()));
            }

            Ok(())
        }

        fn step_cost_f16(&self, g: &LearnerGenome) -> u64 {
            (g.predict.len() + g.learn.len()) as u64
        }

        fn tier_tasks(
            &self,
            token: &HiddenToken,
            secret_seed: u64,
            tier: Tier,
        ) -> Result<Vec<TaskInstance>> {
            if self.fail_tasks {
                return Err(BrutError::Eval("fake: tasks failed".into()));
            }

            let shape = tier_shape(tier).expect("hidden tier");
            let names: Vec<String> = (0..shape.families).map(|j| format!("fam-{j}")).collect();
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            let plan = tier_plan(token, secret_seed, tier, &refs)?;

            Ok(plan
                .iter()
                .map(|s| fake_task(&names[s.family], s))
                .collect())
        }

        fn run(&self, g: &LearnerGenome, task: &TaskInstance, _s: &EvalSettings) -> TaskResult {
            self.counters.runs.fetch_add(1, Ordering::Relaxed);

            match g.learn.len() {
                3 if self.drop_at_32 && task.f == 32 => result(0.5),
                3 => result(0.625),
                1 => result(0.375),
                _ => result(0.25),
            }
        }

        fn tuned_baselines(&self, _s: &EvalSettings) -> TunedBaselines {
            self.counters.tunes.fetch_add(1, Ordering::Relaxed);

            fake_tuned()
        }

        fn matched_set(&self, tb: &TunedBaselines, step_cost_f16: u64) -> Vec<BaselineId> {
            let mut set: Vec<BaselineId> = tb
                .programs
                .iter()
                .filter(|(_, g)| self.step_cost_f16(g) <= step_cost_f16)
                .map(|(b, _)| *b)
                .collect();

            set.push(BaselineId::MlpMatched);

            set
        }

        fn mlp_matched(
            &self,
            _g: &LearnerGenome,
            _task: &TaskInstance,
            _r: &MlpRecipe,
        ) -> TaskResult {
            self.counters.mlp.fetch_add(1, Ordering::Relaxed);

            result(0.4375)
        }

        fn sign_flip_pvalue(&self, d: &[f64], _reps: usize, _seed: u64) -> f64 {
            self.counters.last_d_len.store(d.len(), Ordering::Relaxed);

            self.p_value
        }

        fn check_contamination(&self, sources: &[SourceId], _set: EvalSet) -> Result<()> {
            if sources.contains(&SourceId::ArcAgi1Eval) || sources.contains(&SourceId::Barc) {
                return Err(BrutError::Invalid("fake: contaminated".into()));
            }

            Ok(())
        }

        fn load_arc(
            &self,
            _root: &Path,
            _manifest: &Path,
            _set: EvalSet,
            _sources: &[SourceId],
        ) -> Result<ArcInputs> {
            if self.broken_data {
                return Err(BrutError::Integrity(format!(
                    "listed file {BROKEN_FILE} does not match its manifest hash"
                )));
            }

            let tasks = vec![arc_task("t1", 1), arc_task("t2", 1), arc_task("t3", 2)];
            let planted = if self.overlap {
                tasks[0].test[0].clone()
            } else {
                pair(7, 7)
            };
            let train_hashes = [self.pair_hash(&planted)].into_iter().collect();

            Ok(ArcInputs {
                tasks,
                train_hashes,
            })
        }

        fn pair_hash(&self, p: &ArcPair) -> [u8; 32] {
            let mut hasher = Sha256::new();

            for g in [&p.input, &p.output] {
                hasher.update([g.h, g.w]);
                hasher.update(&g.cells);
            }

            hasher.finalize().into()
        }

        fn solve(&self, _g: &LearnerGenome, t: &ArcTask, _s: &EvalSettings) -> ArcResult {
            self.counters.solves.fetch_add(1, Ordering::Relaxed);

            let (correct, bca) = match t.id.as_str() {
                "t1" => (vec![true], vec![0.75]),
                "t2" => (vec![false], vec![0.5]),
                _ => (vec![true, false], vec![0.625, 0.25]),
            };

            ArcResult {
                task_id: t.id.clone(),
                correct,
                bca,
            }
        }

        fn arc_scores(&self, results: &[ArcResult], n_tasks: usize, n_outputs: usize) -> ArcScores {
            let per_task: f64 = results
                .iter()
                .map(|r| r.correct.iter().filter(|&&c| c).count() as f64 / r.correct.len() as f64)
                .sum();
            let solved: usize = results
                .iter()
                .map(|r| r.correct.iter().filter(|&&c| c).count())
                .sum();
            let bcas: Vec<f64> = results.iter().flat_map(|r| r.bca.clone()).collect();

            ArcScores {
                official: per_task / n_tasks as f64,
                kaggle: solved as f64 / n_outputs as f64,
                bca: Some(bcas.iter().sum::<f64>() / bcas.len() as f64),
            }
        }
    }

    fn open_fake(secret: &Path, budget: usize, step: f64, fake: Fake) -> LearnerEvaluator {
        LearnerEvaluator::open_with(secret, budget, step, Box::new(fake)).expect("open")
    }

    fn ledger_kinds(dir: &Path) -> Vec<LedgerKind> {
        Ledger::open(dir)
            .expect("ledger")
            .entries()
            .iter()
            .map(|e| e.kind.clone())
            .collect()
    }

    /// (charges, releases, refusals) in the ledger.
    fn tally(dir: &Path) -> (usize, usize, usize) {
        let kinds = ledger_kinds(dir);
        let count = |f: fn(&LedgerKind) -> bool| kinds.iter().filter(|k| f(k)).count();

        (
            count(|k| matches!(k, LedgerKind::Charge { .. })),
            count(|k| matches!(k, LedgerKind::Release { .. })),
            count(|k| matches!(k, LedgerKind::Refused { .. })),
        )
    }

    fn assert_no_secret(text: &str) {
        assert!(!text.contains(&SEED.to_string()), "secret (decimal) leaked");
        assert!(!text.contains(&format!("{SEED:x}")), "secret (hex) leaked");
    }

    fn keys(v: &serde_json::Value) -> BTreeSet<String> {
        v.as_object().expect("object").keys().cloned().collect()
    }

    #[test]
    fn open_applies_caps_and_step_floors() {
        let dir = tmp_dir("caps");
        let secret = write_secret(
            &dir,
            &format!(
                "{BUDGET_TOTAL_KEY}=5\n{BUDGET_PER_RUN_KEY}=3\n{HIDDEN_STEP_KEY}=0.02\n{ARC_BUDGET_KEY}=2\n"
            ),
        );

        // --budget can only lower the caps from the secret file.
        let ev = open_fake(&secret, 2, 0.0, fake().0);

        assert_eq!(
            (ev.budget_total, ev.budget_run, ev.arc_budget_per_set),
            (5, 2, 2)
        );
        assert_eq!(ev.score_step(), 0.02);

        let ev = open_fake(&secret, 100, 0.05, fake().0);

        assert_eq!(
            (ev.budget_total, ev.budget_run, ev.arc_budget_per_set),
            (5, 3, 2)
        );
        assert_eq!(ev.score_step(), 0.05);

        // Defaults when the secret file only holds the seed.
        let other = tmp_dir("caps-default");
        let secret = write_secret(&other, "");
        let ev = open_fake(&secret, 100, 0.0, fake().0);

        assert_eq!(
            (ev.budget_total, ev.budget_run, ev.arc_budget_per_set),
            (
                DEFAULT_BUDGET_TOTAL,
                DEFAULT_BUDGET_PER_RUN,
                DEFAULT_ARC_BUDGET_PER_SET
            )
        );
        assert_eq!(ev.score_step(), MIN_HIDDEN_SCORE_STEP);

        // Opening creates and verifies the ledger next to the secret.
        assert!(other.join(crate::ledger::LEDGER_FILE).is_file());

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&other);
    }

    #[cfg(unix)]
    #[test]
    fn open_refuses_unsafe_or_broken_secrets() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tmp_dir("broken");
        let path = dir.join(LEARNER_SECRET_FILE);
        let open = |budget: usize, step: f64| {
            LearnerEvaluator::open_with(&path, budget, step, Box::new(fake().0)).map(|_| ())
        };
        let refused = |r: Result<()>| {
            let err = r.expect_err("must be refused").to_string();

            assert_no_secret(&err);

            err
        };

        // Readable by others.
        write_secret(&dir, "");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).expect("chmod");
        refused(open(3, 0.01));

        // Directory open to others.
        write_secret(&dir, "");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        refused(open(3, 0.01));
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).expect("chmod");
        assert!(open(3, 0.01).is_ok());

        // Missing seed, malformed values (never quoted), out-of-range steps.
        write_private(&path, &format!("{BUDGET_TOTAL_KEY}=4\n"));
        assert!(refused(open(3, 0.01)).contains(LEARNER_SEED_KEY));

        write_private(&path, &format!("{LEARNER_SEED_KEY}=12x34-private\n"));
        assert!(!refused(open(3, 0.01)).contains("12x34-private"));

        write_secret(&dir, &format!("{BUDGET_PER_RUN_KEY}=lots\n"));
        refused(open(3, 0.01));

        write_secret(&dir, &format!("{HIDDEN_STEP_KEY}=1.5\n"));
        refused(open(3, 0.01));

        write_secret(&dir, "");

        for step in [f64::NAN, f64::INFINITY, -0.1, 1.0] {
            refused(open(3, step));
        }

        // Not a regular file.
        std::fs::remove_file(&path).expect("remove");
        std::os::unix::fs::symlink("/dev/zero", &path).expect("symlink");
        refused(open(3, 0.01));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn create_learner_secret_writes_private_defaults() {
        let base = tmp_dir("create");
        let dir = base.join("brut").join("learner");
        let root = Path::new("/data/external");
        let manifest = Path::new("/data/external/manifest.json");
        let path = create_learner_secret(&dir, Some(root), Some(manifest)).expect("create");

        assert_eq!(path, dir.join(LEARNER_SECRET_FILE));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let mode = |p: &Path| std::fs::metadata(p).expect("meta").permissions().mode() & 0o777;

            assert_eq!(mode(&path), 0o600);
            assert_eq!(mode(&dir), 0o700);
            assert_eq!(mode(&base.join("brut")), 0o700);
        }

        let env = parse_env_file(&path).expect("parse");

        assert!(
            env_value::<u64>(&env, LEARNER_SEED_KEY)
                .expect("seed")
                .is_some()
        );
        assert_eq!(env[BUDGET_TOTAL_KEY], DEFAULT_BUDGET_TOTAL.to_string());
        assert_eq!(env[BUDGET_PER_RUN_KEY], DEFAULT_BUDGET_PER_RUN.to_string());
        assert_eq!(env[ARC_BUDGET_KEY], DEFAULT_ARC_BUDGET_PER_SET.to_string());
        assert_eq!(env[HIDDEN_STEP_KEY], "0.01");
        assert_eq!(env[ARC_ROOT_KEY], "/data/external");
        assert_eq!(env[ARC_MANIFEST_KEY], "/data/external/manifest.json");

        // Never overwritten.
        assert!(create_learner_secret(&dir, None, None).is_err());

        // The real evaluator opens it (opening touches no learner code).
        let ev = LearnerEvaluator::open(&path, 100, 0.0).expect("open");

        assert_eq!(ev.budget_total, DEFAULT_BUDGET_TOTAL);
        assert_eq!(ev.budget_run, DEFAULT_BUDGET_PER_RUN);
        assert_eq!(ev.arc_root.as_deref(), Some(root));

        // Two secrets differ.
        let other = create_learner_secret(&base.join("other"), None, None).expect("other");
        let other_env = parse_env_file(&other).expect("parse");

        assert_ne!(env[LEARNER_SEED_KEY], other_env[LEARNER_SEED_KEY]);

        // A path that would inject another key is refused before anything is created.
        let injected = base.join("injected");
        let evil = PathBuf::from(format!("/x\n{BUDGET_TOTAL_KEY}=999999"));

        assert!(create_learner_secret(&injected, Some(&evil), None).is_err());
        assert!(create_learner_secret(&injected, None, Some(Path::new("/m.json "))).is_err());
        assert!(!injected.exists());

        let _ = std::fs::remove_dir_all(&base);
    }

    /// COR-1: the evaluator gives MlpMatched the candidate's training cost (contract 5.9),
    /// not its task cost with the 3 x 256 validation predictions.
    #[test]
    fn library_matches_the_mlp_to_the_training_cost() {
        let g = crate::learner::seeds::seed_programs()[0].1.clone();
        let p = InstanceParams {
            f: 16,
            n_train: 64,
            n_valid: HIDDEN_N_VALID,
            level: 1,
        };
        let task = families::hand_instance(&FamilyId("lin-cls".into()), 3, &p).expect("family");
        let recipe = MlpRecipe {
            hidden: 16,
            activation: Activation::Tanh,
            lr: 0.03,
            l2: 0.0,
        };
        let want = baselines::mlp_matched_training(&task, &recipe, TrainingCost::of(&g, 16));

        assert_eq!(Library.mlp_matched(&g, &task, &recipe), want);
    }

    #[test]
    fn tier_plan_follows_the_table_and_errata_e7() {
        let token = HiddenToken(());
        let table = [
            (Tier::HiddenInstance, 10, 10, vec![8, 16]),
            (Tier::HiddenNear, 5, 20, vec![8, 16, 32]),
            (Tier::HiddenFar, 4, 25, vec![8, 16, 32]),
            (Tier::HiddenGrammar, 40, 5, vec![8, 16, 32]),
        ];

        for (tier, n_fam, n_inst, dims) in table {
            let names: Vec<String> = (0..n_fam).map(|j| format!("fam-{j}")).collect();
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            let plan = tier_plan(&token, SEED, tier, &refs).expect("plan");

            assert_eq!(plan.len(), n_fam * n_inst, "{tier:?}");

            let mut pairs: HashMap<(usize, usize), usize> = HashMap::new();

            for (k, s) in plan.iter().enumerate() {
                let (family, index) = (k / n_inst, k % n_inst);
                let tag = format!("tier/{}/fam-{family}/{index}", tier_name(tier));

                assert_eq!((s.family, s.index), (family, index));
                assert_eq!(s.f, dims[k % dims.len()]);
                assert_eq!(s.n_train, [16, 64, 256][(k / dims.len()) % 3]);
                assert_eq!(s.seed, derive_seed(SEED, &tag));

                *pairs.entry((s.f, s.n_train)).or_default() += 1;
            }

            // Every (F, n_train) pair appears, and all about equally often.
            let lo = pairs.values().min().copied().unwrap_or(0);
            let hi = pairs.values().max().copied().unwrap_or(0);

            assert_eq!(pairs.len(), dims.len() * 3, "{tier:?}");
            assert!(hi - lo <= 1, "{tier:?}: {pairs:?}");

            // Seeds are distinct and depend on the secret.
            let seeds: BTreeSet<u64> = plan.iter().map(|s| s.seed).collect();
            let other = tier_plan(&token, SEED + 1, tier, &refs).expect("plan");

            assert_eq!(seeds.len(), plan.len());
            assert!(other.iter().all(|s| !seeds.contains(&s.seed)));
        }

        // Every grammar family has both F = 32 and F in {8, 16} tasks (needed by Δ32, Δ816).
        let names: Vec<String> = (0..40).map(|j| format!("fam-{j}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let plan = tier_plan(&token, SEED, Tier::HiddenGrammar, &refs).expect("plan");

        for family in 0..40 {
            let widths: Vec<usize> = plan
                .iter()
                .filter(|s| s.family == family)
                .map(|s| s.f)
                .collect();

            assert!(widths.contains(&32) && widths.iter().any(|&f| f != 32));
        }

        // The literal tag spelling (errata E13).
        assert_eq!(
            plan[7].seed,
            derive_seed(SEED, "tier/hidden_grammar/fam-1/2")
        );

        // Wrong family counts and public tiers are refused.
        assert!(tier_plan(&token, SEED, Tier::HiddenInstance, &["a"; 9]).is_err());
        assert!(tier_plan(&token, SEED, Tier::Search, &[]).is_err());
        assert!(tier_plan(&token, SEED, Tier::Select, &[]).is_err());
    }

    #[test]
    fn tier_names_match_serde() {
        for tier in [
            Tier::Search,
            Tier::Select,
            Tier::HiddenInstance,
            Tier::HiddenNear,
            Tier::HiddenFar,
            Tier::HiddenGrammar,
        ] {
            let serde_name = serde_json::to_value(tier).expect("json");

            assert_eq!(serde_name.as_str(), Some(tier_name(tier)));
            assert_eq!(tier_shape(tier).is_some(), tier.is_hidden());
        }
    }

    #[test]
    fn tier_stats_on_a_hand_fixture() {
        // Two families of two tasks; widths 8, 32 | 16, 32.
        let families = [0, 0, 1, 1];
        let widths = [8, 32, 16, 32];
        let candidate = [0.5, 0.25, 1.0, 0.75];
        let baselines = vec![vec![0.25, 0.25, 0.5, 0.5], vec![0.5, 0.0, 0.25, 0.25]];
        let s = tier_stats(&families, &widths, &candidate, &baselines).expect("stats");

        // S(g) = mean(0.375, 0.875) = 0.625; S(b0) = 0.375; S(b1) = 0.25.
        assert_eq!(s.best, 0);
        assert_eq!(s.delta, 0.25);
        assert_eq!(s.family_diffs, vec![0.125, 0.375]);
        assert_eq!(s.delta32, Some(0.125));
        assert_eq!(s.delta816, Some(0.375));

        // Families weigh equally, whatever their size.
        let s = tier_stats(
            &[0, 0, 0, 1],
            &[8; 4],
            &[1.0, 1.0, 1.0, 0.0],
            &[vec![0.0; 4]],
        )
        .expect("stats");

        assert_eq!(s.delta, 0.5);
        assert_eq!(s.delta32, None);

        // Ties pick the first baseline; misaligned inputs are refused.
        let s = tier_stats(&[0], &[8], &[0.5], &[vec![0.25], vec![0.25]]).expect("stats");

        assert_eq!(s.best, 0);
        assert!(tier_stats(&[0, 0], &[8, 8], &[0.5, 0.5], &[vec![0.1]]).is_err());
        assert!(tier_stats(&[], &[], &[], &[vec![]]).is_err());
        assert!(tier_stats(&[0], &[8], &[0.5], &[]).is_err());
    }

    #[test]
    fn primary_decision_rules() {
        let pass = |p, d, d32, d816| primary_decision(p, d, d32, d816) == Decision::Pass;

        assert!(pass(0.01, 0.05, Some(0.0), Some(0.04)));
        assert!(pass(0.049, 0.01, Some(0.2), Some(0.0)));

        // Each clause fails on its own.
        assert!(!pass(0.05, 0.05, Some(0.05), Some(0.05)));
        assert!(!pass(0.01, 0.009, Some(0.05), Some(0.05)));
        assert!(!pass(0.01, 0.05, Some(0.0), Some(0.1)));
        assert!(!pass(0.01, 0.05, None, Some(0.05)));
        assert!(!pass(0.01, 0.05, Some(0.05), None));
        assert!(!pass(f64::NAN, 0.05, Some(0.05), Some(0.05)));
    }

    /// T-HID-1: per-run and total caps; a new run id does not reset the total.
    #[test]
    fn caps_per_run_and_total_hold_across_runs() {
        let dir = tmp_dir("hid1");
        let secret = write_secret(
            &dir,
            &format!("{BUDGET_TOTAL_KEY}=3\n{BUDGET_PER_RUN_KEY}=2\n"),
        );
        let mut ev = open_fake(&secret, 10, 0.01, fake().0);

        let r = ev
            .evaluate(&learner_req(1, Tier::HiddenInstance))
            .expect("q1");

        assert_eq!(
            (
                r.queries_used_total,
                r.queries_left_total,
                r.queries_left_run
            ),
            (1, 2, 1)
        );

        let r = ev
            .evaluate(&learner_req(1, Tier::HiddenGrammar))
            .expect("q2");

        assert_eq!(
            (
                r.queries_used_total,
                r.queries_left_total,
                r.queries_left_run
            ),
            (2, 1, 0)
        );

        // Per-run cap.
        let err = ev
            .evaluate(&learner_req(1, Tier::HiddenInstance))
            .expect_err("run cap");

        assert!(
            matches!(&err, BrutError::Eval(m) if m.contains("budget")),
            "{err}"
        );

        // A new run id gets its own run cap but shares the total.
        let r = ev
            .evaluate(&learner_req(2, Tier::HiddenInstance))
            .expect("q3");

        assert_eq!(
            (
                r.queries_used_total,
                r.queries_left_total,
                r.queries_left_run
            ),
            (3, 0, 1)
        );
        assert!(ev.evaluate(&learner_req(2, Tier::HiddenInstance)).is_err());

        // Another run directory reaches the same global secret: still exhausted.
        let mut again = open_fake(&secret, 10, 0.01, fake().0);

        assert!(
            again
                .evaluate(&learner_req(3, Tier::HiddenInstance))
                .is_err()
        );

        // Refused queries are logged, not charged.
        assert_eq!(tally(&dir), (3, 3, 3));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two run directories (two `brut run` processes with their own run ids) reach the same
    /// global secret, so they draw on one total budget through the shared ledger.
    #[test]
    fn budget_spans_run_directories_via_the_global_ledger() {
        let dir = tmp_dir("run-dirs");
        let secret = write_secret(
            &dir,
            &format!("{BUDGET_TOTAL_KEY}=4\n{BUDGET_PER_RUN_KEY}=3\n"),
        );

        let mut run_a = open_fake(&secret, 12, 0.01, fake().0);

        for tier in [Tier::HiddenInstance, Tier::HiddenGrammar, Tier::HiddenNear] {
            run_a.evaluate(&learner_req(10, tier)).expect("run A");
        }

        let mut run_b = open_fake(&secret, 12, 0.01, fake().0);
        let r = run_b
            .evaluate(&learner_req(11, Tier::HiddenInstance))
            .expect("run B, first");

        assert_eq!(
            (
                r.queries_used_total,
                r.queries_left_total,
                r.queries_left_run
            ),
            (4, 0, 2)
        );

        let err = run_b
            .evaluate(&learner_req(11, Tier::HiddenGrammar))
            .expect_err("total spent");

        assert!(err.to_string().contains("for this protocol"), "{err}");

        // Both run ids are in the one ledger next to the secret.
        let ledger = Ledger::open(&dir).expect("ledger");

        assert_eq!(ledger.learner_charges(&ph(), Some(&run_id(10))), 3);
        assert_eq!(ledger.learner_charges(&ph(), Some(&run_id(11))), 1);
        assert_eq!(ledger.learner_charges(&ph(), None), 4);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cli_budget_lowers_but_never_raises_the_run_cap() {
        let dir = tmp_dir("cli-budget");
        let secret = write_secret(
            &dir,
            &format!("{BUDGET_TOTAL_KEY}=10\n{BUDGET_PER_RUN_KEY}=2\n"),
        );

        let mut low = open_fake(&secret, 1, 0.01, fake().0);

        low.evaluate(&learner_req(1, Tier::HiddenInstance))
            .expect("first");
        assert!(low.evaluate(&learner_req(1, Tier::HiddenInstance)).is_err());

        let mut high = open_fake(&secret, 1000, 0.01, fake().0);

        high.evaluate(&learner_req(2, Tier::HiddenInstance))
            .expect("first");
        high.evaluate(&learner_req(2, Tier::HiddenInstance))
            .expect("second");
        assert!(
            high.evaluate(&learner_req(2, Tier::HiddenInstance))
                .is_err()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// T-HID-2: protocol mismatch, invalid and over-cost programs are refused before the charge.
    #[test]
    fn refused_queries_are_logged_and_not_charged() {
        let dir = tmp_dir("hid2");
        let secret = write_secret(&dir, "");
        let (fake, counters) = fake();
        let mut ev = open_fake(&secret, 10, 0.01, fake);

        let mut req = learner_req(1, Tier::HiddenGrammar);

        req.protocol_hash = "d".repeat(64);

        let err = ev.evaluate(&req).expect_err("protocol");

        assert!(
            matches!(&err, BrutError::Eval(m) if m.contains("protocol")),
            "{err}"
        );

        let mut req = learner_req(1, Tier::HiddenGrammar);

        req.program = genome(3, 5);
        assert!(matches!(ev.evaluate(&req), Err(BrutError::Invalid(_))));

        req.program = candidate();
        req.program.setup[0].c = f32::NAN;
        assert!(matches!(ev.evaluate(&req), Err(BrutError::Invalid(_))));

        assert_eq!(tally(&dir), (0, 0, 3));

        // Malformed requests are rejected without any ledger entry.
        let mut bad = learner_req(1, Tier::HiddenGrammar);

        bad.run_id = "not-hex".into();
        assert!(matches!(ev.evaluate(&bad), Err(BrutError::Invalid(_))));

        let mut bad = learner_req(1, Tier::HiddenGrammar);

        bad.candidate_id = "a b".into();
        assert!(matches!(ev.evaluate(&bad), Err(BrutError::Invalid(_))));

        let mut bad = learner_req(1, Tier::HiddenGrammar);

        bad.protocol_hash = "C".repeat(64);
        assert!(matches!(ev.evaluate(&bad), Err(BrutError::Invalid(_))));

        assert!(matches!(
            ev.evaluate(&learner_req(1, Tier::Search)),
            Err(BrutError::Invalid(_))
        ));
        assert!(matches!(
            ev.evaluate(&learner_req(1, Tier::Select)),
            Err(BrutError::Invalid(_))
        ));

        assert_eq!(tally(&dir), (0, 0, 3));

        // Nothing ran, and the full budget is still there.
        assert_eq!(counters.runs.load(Ordering::Relaxed), 0);

        let r = ev
            .evaluate(&learner_req(1, Tier::HiddenGrammar))
            .expect("valid");

        assert_eq!(r.queries_used_total, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// T-HID-3: documented fields only; Δ is a multiple of the step and <= the true Δ;
    /// a decision only for HiddenGrammar.
    #[test]
    fn responses_release_one_rounded_number() {
        let dir = tmp_dir("hid3");
        let secret = write_secret(&dir, "");
        let mut ev = open_fake(&secret, 12, 0.01, fake().0);
        let documented: BTreeSet<String> = [
            "candidate_id",
            "tier",
            "delta",
            "decision",
            "queries_used_total",
            "queries_left_total",
            "queries_left_run",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        // Fake scores: candidate 0.625, best matched baseline (MlpMatched) 0.4375.
        let true_delta = 0.1875;

        for tier in [
            Tier::HiddenInstance,
            Tier::HiddenNear,
            Tier::HiddenFar,
            Tier::HiddenGrammar,
        ] {
            let r = ev.evaluate(&learner_req(1, tier)).expect("eval");
            let json = serde_json::to_value(&r).expect("json");

            assert_eq!(keys(&json), documented);
            assert_eq!(r.candidate_id, CANDIDATE);
            assert_eq!(r.tier, tier);
            assert_eq!(r.delta, 0.18);
            assert!(r.delta <= true_delta);
            assert_eq!(r.decision.is_some(), tier == Tier::HiddenGrammar);
        }

        // A coarser step from the caller rounds further down to a multiple of it.
        let mut coarse = open_fake(&secret, 12, 0.05, fake().0);
        let r = coarse
            .evaluate(&learner_req(2, Tier::HiddenInstance))
            .expect("eval");

        assert_eq!(r.delta, 0.15);
        assert!(((r.delta / 0.05).round() * 0.05 - r.delta).abs() < 1e-12);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn grammar_decision_follows_the_primary_test() {
        let dir = tmp_dir("decision");
        let secret = write_secret(&dir, "");

        let (pass, counters) = fake();
        let mut ev = open_fake(&secret, 12, 0.01, pass);
        let r = ev
            .evaluate(&learner_req(1, Tier::HiddenGrammar))
            .expect("eval");

        assert_eq!(r.decision, Some(Decision::Pass));
        assert_eq!(
            counters.last_d_len.load(Ordering::Relaxed),
            HIDDEN_GRAMMAR_FAMILIES
        );

        // Width coupling: Δ32 = 0.0625 < Δ816 - 0.05 = 0.1375.
        let (mut coupled, _) = fake();

        coupled.drop_at_32 = true;

        let mut ev = open_fake(&secret, 12, 0.01, coupled);
        let r = ev
            .evaluate(&learner_req(1, Tier::HiddenGrammar))
            .expect("eval");

        assert_eq!(r.decision, Some(Decision::Fail));
        assert!(r.delta >= 0.01);

        // Not significant.
        let (mut weak, _) = fake();

        weak.p_value = 0.2;

        let mut ev = open_fake(&secret, 12, 0.01, weak);
        let r = ev
            .evaluate(&learner_req(1, Tier::HiddenGrammar))
            .expect("eval");

        assert_eq!(r.decision, Some(Decision::Fail));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ledger_records_exactly_what_was_released_and_never_the_secret() {
        let dir = tmp_dir("release");
        let secret = write_secret(&dir, &arc_keys(3));
        let mut ev = open_fake(&secret, 12, 0.01, fake().0);
        let mut released: Vec<serde_json::Value> = Vec::new();
        let mut outputs: Vec<String> = Vec::new();

        for tier in [Tier::HiddenInstance, Tier::HiddenGrammar] {
            let r = ev.evaluate(&learner_req(1, tier)).expect("eval");

            released.push(serde_json::to_value(&r).expect("json"));
            outputs.push(serde_json::to_string(&r).expect("json"));
        }

        let r = ev
            .evaluate_arc(&arc_req(EvalSet::A1Eval, vec![SourceId::Synthetic]))
            .expect("arc");

        released.push(serde_json::to_value(&r).expect("json"));
        outputs.push(serde_json::to_string(&r).expect("json"));

        let mut bad = learner_req(1, Tier::HiddenInstance);

        bad.protocol_hash = "e".repeat(64);
        outputs.push(ev.evaluate(&bad).expect_err("refused").to_string());

        // Release entries hold exactly the responses, in order; the chain verifies.
        let ledger_path = dir.join(crate::ledger::LEDGER_FILE);
        let in_ledger: Vec<serde_json::Value> = ledger_kinds(&dir)
            .into_iter()
            .filter_map(|k| match k {
                LedgerKind::Release { released, .. } => Some(released),
                _ => None,
            })
            .collect();

        assert_eq!(in_ledger, released);
        assert_eq!(Ledger::verify(&ledger_path).expect("verify"), 7);

        // Neither the secret nor any hidden instance seed appears anywhere.
        let ledger_text = std::fs::read_to_string(&ledger_path).expect("ledger");
        let cache_text = std::fs::read_to_string(dir.join(BASELINE_CACHE_FILE)).expect("cache");
        let token = HiddenToken(());

        for text in outputs.iter().chain([&ledger_text, &cache_text]) {
            assert_no_secret(text);

            for tier in [Tier::HiddenInstance, Tier::HiddenGrammar] {
                let shape = tier_shape(tier).expect("shape");
                let names: Vec<String> = (0..shape.families).map(|j| format!("fam-{j}")).collect();
                let refs: Vec<&str> = names.iter().map(String::as_str).collect();

                for s in tier_plan(&token, SEED, tier, &refs).expect("plan") {
                    assert!(!text.contains(&s.seed.to_string()), "instance seed leaked");
                }
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn baseline_results_are_cached_and_never_charged() {
        let dir = tmp_dir("cache");
        let secret = write_secret(&dir, "");
        let (f, counters) = fake();
        let mut ev = open_fake(&secret, 12, 0.01, f);
        let runs = || counters.runs.load(Ordering::Relaxed);

        // First query: candidate + 4 program baselines on 100 tasks, tuning once.
        let first = ev
            .evaluate(&learner_req(1, Tier::HiddenInstance))
            .expect("q1");

        assert_eq!(runs(), 500);
        assert_eq!(counters.tunes.load(Ordering::Relaxed), 1);
        assert_eq!(counters.mlp.load(Ordering::Relaxed), 100);

        // Same tier again: only the candidate and MlpMatched run.
        let second = ev
            .evaluate(&learner_req(1, Tier::HiddenInstance))
            .expect("q2");

        assert_eq!(runs(), 600);
        assert_eq!(counters.mlp.load(Ordering::Relaxed), 200);
        assert_eq!(first.delta, second.delta);

        // Another tier reuses the tuned baselines.
        ev.evaluate(&learner_req(1, Tier::HiddenGrammar))
            .expect("q3");

        assert_eq!(runs(), 600 + 5 * 200);
        assert_eq!(counters.tunes.load(Ordering::Relaxed), 1);

        // The cache persists for the next evaluator and is private.
        let (f, counters) = fake();
        let mut again = open_fake(&secret, 12, 0.01, f);

        again
            .evaluate(&learner_req(2, Tier::HiddenInstance))
            .expect("q4");

        assert_eq!(counters.runs.load(Ordering::Relaxed), 100);
        assert_eq!(counters.tunes.load(Ordering::Relaxed), 0);
        assert_eq!(tally(&dir).0, 4);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let mode = std::fs::metadata(dir.join(BASELINE_CACHE_FILE))
                .expect("meta")
                .permissions()
                .mode();

            assert_eq!(mode & 0o777, 0o600);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn damaged_cache_is_refused_before_the_charge() {
        let dir = tmp_dir("bad-cache");
        let secret = write_secret(&dir, "");
        let cache = dir.join(BASELINE_CACHE_FILE);
        let mut ev = open_fake(&secret, 12, 0.01, fake().0);

        ev.evaluate(&learner_req(1, Tier::HiddenInstance))
            .expect("fill cache");

        // Tampered tuned baselines no longer match their hash.
        let text = std::fs::read_to_string(&cache).expect("read");
        let mut value: serde_json::Value = serde_json::from_str(&text).expect("json");

        value["tuned"][0]["baselines"]["mlp"]["lr"] = serde_json::json!(0.5);
        write_private(&cache, &value.to_string());

        let err = ev
            .evaluate(&learner_req(1, Tier::HiddenInstance))
            .expect_err("tampered");

        assert!(matches!(err, BrutError::Integrity(_)), "{err}");

        // Not JSON at all.
        write_private(&cache, "{not json");
        assert!(matches!(
            ev.evaluate(&learner_req(1, Tier::HiddenInstance)),
            Err(BrutError::Integrity(_))
        ));

        // Neither attempt was charged.
        assert_eq!(tally(&dir), (1, 1, 0));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn failures_after_the_charge_still_count() {
        let dir = tmp_dir("crash");
        let secret = write_secret(&dir, "");
        let (mut broken, _) = fake();

        broken.fail_tasks = true;

        let mut ev = open_fake(&secret, 12, 0.01, broken);

        assert!(ev.evaluate(&learner_req(1, Tier::HiddenInstance)).is_err());
        assert_eq!(tally(&dir), (1, 0, 0));

        let mut ok = open_fake(&secret, 12, 0.01, fake().0);
        let r = ok
            .evaluate(&learner_req(1, Tier::HiddenInstance))
            .expect("eval");

        assert_eq!(r.queries_used_total, 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_evaluators_share_the_caps_exactly() {
        let dir = tmp_dir("concurrent");
        let secret = write_secret(&dir, &format!("{BUDGET_TOTAL_KEY}=5\n"));

        let results: Vec<Result<LearnerEvalResponse>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8u64)
                .map(|i| {
                    let secret = &secret;

                    scope.spawn(move || {
                        let mut ev = open_fake(secret, 12, 0.01, fake().0);

                        ev.evaluate(&learner_req(i, Tier::HiddenInstance))
                    })
                })
                .collect();

            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(eval_err("thread panicked")))
                })
                .collect()
        });

        let mut used: Vec<usize> = results
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|r| r.queries_used_total)
            .collect();

        used.sort_unstable();

        assert_eq!(used, vec![1, 2, 3, 4, 5]);
        assert_eq!(tally(&dir), (5, 5, 3));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn arc_queries_release_scores_within_their_own_budget() {
        let dir = tmp_dir("arc");
        let secret = write_secret(&dir, &arc_keys(2));
        let (f, counters) = fake();
        let mut ev = open_fake(&secret, 12, 0.01, f);
        let req = arc_req(EvalSet::A1Eval, vec![SourceId::Synthetic]);
        let r = ev.evaluate_arc(&req).expect("arc");

        assert_eq!(r.candidate_id, CANDIDATE);
        assert_eq!(r.set, EvalSet::A1Eval);
        assert_eq!((r.n_tasks, r.n_test_outputs, r.solved_outputs), (3, 4, 2));
        assert_eq!((r.official, r.kaggle), (0.5, 0.5));

        // Mean BCA 0.53125, rounded down to 0.01.
        assert_eq!(r.bca, Some(0.53));
        assert_eq!(r.queries_left_set, 1);
        assert_eq!(counters.solves.load(Ordering::Relaxed), 3);

        let documented: BTreeSet<String> = [
            "candidate_id",
            "set",
            "n_tasks",
            "n_test_outputs",
            "solved_outputs",
            "official",
            "kaggle",
            "bca",
            "queries_left_set",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        assert_eq!(keys(&serde_json::to_value(&r).expect("json")), documented);

        // Budget per set and protocol, shared by all runs.
        assert_eq!(ev.evaluate_arc(&req).expect("second").queries_left_set, 0);

        let mut other_run = req.clone();

        other_run.run_id = run_id(9);
        assert!(ev.evaluate_arc(&other_run).is_err());

        // Another set has its own budget; ARC checks never touch the learner budget.
        let r = ev
            .evaluate_arc(&arc_req(EvalSet::A2Eval, vec![SourceId::Synthetic]))
            .expect("a2");

        assert_eq!(r.queries_left_set, 1);

        let ledger = Ledger::open(&dir).expect("ledger");

        assert_eq!(ledger.learner_charges(&ph(), None), 0);
        assert_eq!(
            ledger.charges(
                &ph(),
                None,
                &QueryKind::Arc {
                    set: EvalSet::A1Eval
                }
            ),
            2
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn arc_contamination_and_overlap_are_refused_without_charge() {
        let dir = tmp_dir("arc-refuse");
        let secret = write_secret(&dir, &arc_keys(3));
        let (f, counters) = fake();
        let mut ev = open_fake(&secret, 12, 0.01, f);

        // A forbidden training source.
        let contaminated = arc_req(
            EvalSet::A1Eval,
            vec![SourceId::Synthetic, SourceId::ArcAgi1Eval],
        );
        let err = ev.evaluate_arc(&contaminated).expect_err("contaminated");

        assert!(err.to_string().contains("forbidden"), "{err}");

        // A planted training pair that equals an eval pair.
        let (mut planted, _) = fake();

        planted.overlap = true;

        let mut ev2 = open_fake(&secret, 12, 0.01, planted);
        let err = ev2
            .evaluate_arc(&arc_req(EvalSet::A1Eval, vec![SourceId::Synthetic]))
            .expect_err("overlap");

        assert!(err.to_string().contains("overlaps"), "{err}");

        // Malformed: no training sources at all (not logged).
        assert!(matches!(
            ev.evaluate_arc(&arc_req(EvalSet::A1Eval, Vec::new())),
            Err(BrutError::Invalid(_))
        ));

        assert_eq!(tally(&dir), (0, 0, 2));
        assert_eq!(counters.solves.load(Ordering::Relaxed), 0);

        // Without ARC data in the secret file, ARC requests are refused.
        let bare = tmp_dir("arc-bare");
        let secret = write_secret(&bare, "");
        let mut ev = open_fake(&secret, 12, 0.01, fake().0);
        let err = ev
            .evaluate_arc(&arc_req(EvalSet::A1Eval, vec![SourceId::Synthetic]))
            .expect_err("no data");

        assert!(err.to_string().contains("no ARC data"), "{err}");
        assert_eq!(tally(&bare), (0, 0, 1));

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&bare);
    }

    /// An eval-set path as the ARC loaders quote it in their errors.
    const BROKEN_FILE: &str = "arc-agi-2/data/evaluation/0934a4d8.json";

    /// SEC-3: a failing ARC load is refused without a charge, and the caller (and so the
    /// orchestrator's archive) only sees the fixed refusal text. The loader's message, which
    /// names eval-set files, stays in the evaluator's ledger.
    #[test]
    fn arc_data_errors_do_not_reach_the_caller() {
        let dir = tmp_dir("arc-broken");
        let secret = write_secret(&dir, &arc_keys(3));
        let (mut f, counters) = fake();

        f.broken_data = true;

        let mut ev = open_fake(&secret, 12, 0.01, f);
        let err = ev
            .evaluate_arc(&arc_req(EvalSet::A2Eval, vec![SourceId::Synthetic]))
            .expect_err("broken data");
        let text = err.to_string();

        assert!(text.contains(REFUSED_ARC_DATA), "{text}");
        assert!(
            !text.contains("data/evaluation") && !text.contains("0934a4d8"),
            "{text}"
        );
        assert_eq!(tally(&dir), (0, 0, 1));
        assert_eq!(counters.solves.load(Ordering::Relaxed), 0);

        let reasons: Vec<String> = ledger_kinds(&dir)
            .into_iter()
            .filter_map(|k| match k {
                LedgerKind::Refused { reason, .. } => Some(reason),
                _ => None,
            })
            .collect();

        assert_eq!(reasons.len(), 1);
        assert!(reasons[0].starts_with(REFUSED_ARC_DATA), "{reasons:?}");
        assert!(reasons[0].contains(BROKEN_FILE), "{reasons:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Write an executable shell script (Unix only).
    #[cfg(unix)]
    fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = dir.join(name);

        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("write script");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");

        path
    }

    /// Retry while a fresh script is still busy ("Text file busy" from a parallel test).
    #[cfg(unix)]
    fn retry_busy<T>(call: impl Fn() -> Result<T>) -> Result<T> {
        for _ in 0..20 {
            match call() {
                Err(BrutError::Eval(m)) if m.contains("busy") => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                other => return other,
            }
        }

        Err(eval_err("script stayed busy"))
    }

    /// A fake brut-eval that stores its stdin in `<dir>/<name>.body` and prints `answer`.
    #[cfg(unix)]
    fn answering(dir: &Path, name: &str, answer: &serde_json::Value) -> (PathBuf, PathBuf) {
        let body = dir.join(format!("{name}.body"));
        let bin = script(
            dir,
            &format!("{name}.sh"),
            &format!("cat > '{}'\nprintf '%s' '{answer}'", body.display()),
        );

        (bin, body)
    }

    #[cfg(unix)]
    #[test]
    fn request_learner_eval_sends_a_tagged_request_and_checks_the_answer() {
        let dir = tmp_dir("request");
        let req = learner_req(1, Tier::HiddenGrammar);
        let good = serde_json::json!({
            "candidate_id": CANDIDATE, "tier": "hidden_grammar", "delta": 0.12,
            "decision": "pass", "queries_used_total": 1, "queries_left_total": 47,
            "queries_left_run": 11
        });
        let call = |bin: &Path| {
            retry_busy(|| request_learner_eval(bin, Path::new("s.env"), 12, 0.01, &req, None))
        };

        let (bin, body) = answering(&dir, "good", &good);
        let resp = call(&bin).expect("good answer");

        assert_eq!(resp.delta, 0.12);
        assert_eq!(resp.decision, Some(Decision::Pass));

        let sent = std::fs::read_to_string(&body).expect("body");

        assert!(
            sent.starts_with("{\"kind\":\"learner\",\"run_id\":"),
            "{sent}"
        );
        assert_eq!(
            AnyEvalRequest::from_json(&sent).expect("parse"),
            AnyEvalRequest::Learner(req.clone())
        );

        // Answers that do not fit the request are refused.
        let mutations: [(&str, serde_json::Value); 5] = [
            ("candidate_id", serde_json::json!("ffffffffffffffff")),
            ("tier", serde_json::json!("hidden_instance")),
            ("decision", serde_json::Value::Null),
            ("delta", serde_json::json!(3.0)),
            ("extra", serde_json::json!(1)),
        ];

        for (i, (key, value)) in mutations.into_iter().enumerate() {
            let mut bad = good.clone();

            bad[key] = value;

            let (bin, _) = answering(&dir, &format!("bad{i}"), &bad);

            assert!(
                matches!(call(&bin), Err(BrutError::Eval(_))),
                "{key} must be refused"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn request_arc_eval_checks_the_answer() {
        let dir = tmp_dir("request-arc");
        let req = arc_req(EvalSet::ConceptArc, vec![SourceId::Synthetic]);
        let good = serde_json::json!({
            "candidate_id": CANDIDATE, "set": "concept_arc", "n_tasks": 160,
            "n_test_outputs": 170, "solved_outputs": 3, "official": 0.0125,
            "kaggle": 0.0176, "bca": 0.51, "queries_left_set": 2
        });
        let call = |bin: &Path| {
            retry_busy(|| request_arc_eval(bin, Path::new("s.env"), 3, 0.01, &req, None))
        };

        let (bin, body) = answering(&dir, "good", &good);

        assert_eq!(call(&bin).expect("good").solved_outputs, 3);
        assert!(
            std::fs::read_to_string(&body)
                .expect("body")
                .starts_with("{\"kind\":\"arc\",")
        );

        for (i, (key, value)) in [
            ("set", serde_json::json!("a1_eval")),
            ("solved_outputs", serde_json::json!(171)),
            ("official", serde_json::json!(1.5)),
            ("bca", serde_json::json!(-0.1)),
        ]
        .into_iter()
        .enumerate()
        {
            let mut bad = good.clone();

            bad[key] = value;

            let (bin, _) = answering(&dir, &format!("bad{i}"), &bad);

            assert!(matches!(call(&bin), Err(BrutError::Eval(_))), "{key}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// SCI-R4: no hidden grammar family is an exact twin (same kind and semantic key) of a
    /// public family or of another hidden family, whatever the secret. Dummy secrets only.
    #[test]
    fn hidden_grammar_families_have_no_semantic_twins() {
        let token = HiddenToken(());
        let public = grammar::public_semantic_keys();

        for secret in 0..100u64 {
            let seed = secret.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let pool = hidden_grammar_pool(&token, seed).expect("pool");
            let keys = grammar::semantic_keys(&pool);

            assert_eq!(pool.len(), HIDDEN_GRAMMAR_FAMILIES);
            assert_eq!(
                keys.len(),
                pool.len(),
                "a twin inside the pool (dummy secret {secret})"
            );
            assert!(
                keys.is_disjoint(&public),
                "a twin of a public family (dummy secret {secret})"
            );
        }
    }

    // Tests against the real learner library; they run after integration.

    /// T-HID-4: hidden grammar signatures avoid both public pools; tiers sized per table.
    #[test]
    #[ignore = "needs the WP2 grammar and family modules; run after integration"]
    fn hidden_tiers_match_the_table_and_avoid_public_signatures() {
        let token = HiddenToken(());
        let pool = hidden_grammar_pool(&token, SEED).expect("pool");
        let public: BTreeSet<String> = grammar::public_pool(Tier::Search)
            .iter()
            .chain(grammar::public_pool(Tier::Select).iter())
            .map(|g| g.signature.clone())
            .collect();

        assert_eq!(pool.len(), HIDDEN_GRAMMAR_FAMILIES);
        assert!(pool.iter().all(|g| !public.contains(&g.signature)));

        for (tier, n) in [
            (Tier::HiddenInstance, 100),
            (Tier::HiddenNear, 100),
            (Tier::HiddenFar, 100),
            (Tier::HiddenGrammar, 200),
        ] {
            let tasks = hidden_tier_tasks(&token, SEED, tier).expect("tasks");

            assert_eq!(tasks.len(), n, "{tier:?}");
            assert!(tasks.iter().all(|t| t.n_valid == HIDDEN_N_VALID));
            assert_eq!(tasks, hidden_tier_tasks(&token, SEED, tier).expect("again"));
            assert_ne!(
                tasks,
                hidden_tier_tasks(&token, SEED + 1, tier).expect("other")
            );
        }
    }

    #[test]
    #[ignore = "needs the WP2 protocol descriptor; run after integration"]
    fn protocol_descriptor_matches_the_evaluator() {
        let d = protocol::protocol_descriptor();
        let sizes: Vec<(Tier, usize, usize)> = [
            Tier::HiddenInstance,
            Tier::HiddenNear,
            Tier::HiddenFar,
            Tier::HiddenGrammar,
        ]
        .iter()
        .map(|&t| {
            let s = tier_shape(t).expect("shape");

            (t, s.families, s.instances)
        })
        .collect();

        assert_eq!(d.hidden_tier_sizes, sizes);
        assert_eq!(d.n_valid_hidden, HIDDEN_N_VALID);
        assert_eq!(
            d.primary_test,
            format!(
                "sign-flip;reps={SIGN_FLIP_REPS};alpha={PRIMARY_ALPHA};\
                 min_delta={PRIMARY_MIN_DELTA};coupling={COUPLING_MAX_DROP}"
            )
        );
    }

    #[test]
    #[ignore = "needs every learner module; slow in debug builds (baseline tuning)"]
    fn live_evaluation_of_a_seed_program() {
        let dir = tmp_dir("live");
        let path = create_learner_secret(&dir.join("learner"), None, None).expect("secret");
        let mut ev = LearnerEvaluator::open(&path, 12, 0.01).expect("open");
        let (_, program) = crate::learner::seeds::seed_programs()
            .into_iter()
            .next()
            .expect("seed");
        let req = LearnerEvalRequest {
            run_id: run_id(1),
            candidate_id: program.id(),
            protocol_hash: protocol::protocol_hash(),
            tier: Tier::HiddenGrammar,
            program,
        };
        let r = ev.evaluate(&req).expect("live evaluation");

        assert!(r.delta.is_finite() && r.delta > -2.0 && r.delta <= 1.0);
        assert!(r.decision.is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
