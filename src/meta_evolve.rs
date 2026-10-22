//! Learner-mode orchestrator: search loop, final phase, hidden phase.
//!
//! Owner: WP5. Contract sections 5.19, 6.5 - 6.7. Does file I/O.
//! Never reads secrets: hidden results come only through `brut-eval`.
//!
//! Run directory layout:
//! - `meta.json` run config, `protocol.json` human copy of the registered protocol
//! - `archive.jsonl` + `archive.head` hash-chained archive (`MetaRunStarted` is record 0)
//! - `programs/<id>.json` finalists, `programs/baseline-<id>.json` tuned program baselines,
//!   `programs/tuned-baselines.json` all tuned baselines incl. the MLP recipe
//! - `lineage/chunk-<first batch:06>.jsonl` lineage chunks
//! - `snapshots/pop-<batches done:09>.json` snapshots, `snapshots/cache-latest.json` score cache
//! - `report.json`, `STOP` (kill switch)
//!
//! Determinism: every random draw comes from `derive_seed` tags (contract section 6.1);
//! parallel work only reads shared state and is merged in slot or member order. A batch,
//! a rotation or a Stage C round that sees the kill switch is discarded as a whole and
//! redone after a resume, so a resumed run continues exactly like an uninterrupted one.
#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::archive::{
    Archive, BatchDigestEntry, CostPhase, LearnerEntry, LearnerStatus, RecordKind,
};
use crate::config::{
    self, HARD_MAX_EPOCHS, HARD_MAX_HIDDEN_LAYERS, HARD_MAX_PARAMS, HARD_MAX_POPULATION,
    HARD_MAX_WIDTH, Limits, MAX_THREADS,
};
use crate::evolve::ARCHIVE_FILE;
use crate::hidden_eval::{MIN_HIDDEN_SCORE_STEP, read_limited};
use crate::learner::baselines::{self, TrainingCost};
use crate::learner::families::{self, Panel};
use crate::learner::fec::ScoreCache;
use crate::learner::mutate::{self, ProgramProposer, RandomProgramMutator};
use crate::learner::protocol::{self, TaskPrep};
use crate::learner::seeds;
use crate::learner::stats::Hurdle;
use crate::learner::types::{
    Abort, BaselineId, DslLimits, EvalSet, EvalSettings, HurdleConfig, LearnerGenome,
    LearnerHiddenResult, MetaConfig, MetaHiddenConfig, MetaWeights, Mutation, PROTOCOL_VERSION,
    PanelConfig, SUPPORTED_F, SelectConfig, SourceId, TaskInstance, TaskResult, Tier,
    TunedBaselines, VariationMode,
};
use crate::learner_hidden::{
    ArcEvalRequest, ArcEvalResponse, LearnerEvalRequest, request_arc_eval, request_learner_eval,
};
use crate::lineage_store::{
    ChunkStats, LineageRecord, LineageWriter, MAX_STORE_FILE_BYTES, PopMember, Snapshot,
    promote_cache, read_run_cache, read_snapshot, write_cache_pending, write_snapshot,
};
use crate::meta_report;
use crate::safety::{self, Guard, StopSignal};
use crate::util::{BrutError, Result, derive_seed, now_unix_ms, sha256_hex, write_atomic};

pub const META_CONFIG_FILE: &str = "meta.json";
pub const PROGRAMS_DIR: &str = "programs";
pub const LINEAGE_DIR: &str = "lineage";
pub const SNAPSHOT_DIR: &str = "snapshots";

/// Human copy of the registered protocol (the archive record is binding).
pub const PROTOCOL_FILE: &str = "protocol.json";

/// All tuned baselines of the run (programs and MLP recipe), inside `PROGRAMS_DIR`.
pub const TUNED_BASELINES_FILE: &str = "tuned-baselines.json";

/// Stage C rounds from this number on are final rounds: `FINAL_ROUND_BASE + batches done`.
/// Periodic rounds count 0, 1, 2, ..; the gap keeps their SELECT tasks apart.
pub const FINAL_ROUND_BASE: u64 = 1 << 63;

/// `select_tasks` round of the finalist retune set (contract section 6.7.2).
pub const RETUNE_ROUND: u64 = u64::MAX;

/// Generator of the initial members that were mutated from a seed.
pub const INIT_GENERATOR: &str = "init";

/// Generator of random-search children (errata E10).
pub const RANDOM_SEARCH_GENERATOR: &str = "random-search";

/// Generator of retuned finalists.
pub const RETUNE_GENERATOR: &str = "retune";

/// Hard caps that guard against typos in `meta.json`.
pub const HARD_MAX_BATCH: usize = 4096;
pub const HARD_MAX_N_TRAIN: usize = 4096;
pub const HARD_MAX_N_VALID: usize = 4096;
pub const HARD_MAX_FEC_CAPACITY: usize = 4_000_000;
pub const HARD_MAX_PROGRAM_LEN: usize = 1024;
pub const HARD_MAX_CONST_ABS: f32 = 1.0e6;
pub const HARD_MAX_TRIALS: usize = 100_000;
pub const HARD_MAX_HOF: usize = 1024;
/// Errata K3: hard cap of the permanent all-time-best memory. Kept small so the memory
/// never reintroduces the AC-11 hall-of-fame blow-up (it stores at most this many genomes).
pub const HARD_MAX_ALL_TIME_BEST: usize = 64;
pub const HARD_MAX_HAND_PER_FAMILY: usize = 256;
pub const HARD_MAX_QUERY_BUDGET: usize = 1000;
pub const HARD_MAX_HURDLE_WINDOW: usize = 1_000_000;

/// Size of the SELECT grammar pool (contract section 5.8).
pub const SELECT_POOL_SIZE: usize = 32;

/// Size of the SEARCH grammar pool (contract section 5.8, errata E12).
pub const SEARCH_POOL_SIZE: usize = 64;

/// Largest `meta.json` that is read.
pub const MAX_META_CONFIG_BYTES: u64 = 1 << 20;

/// Fitness given when the formula is not finite (never expected; keeps I8).
pub const WORST_META_FITNESS: f64 = -1.0e9;

/// A progress line is printed every this many batches.
const PROGRESS_EVERY: u64 = 10;

impl Default for MetaConfig {
    /// Values in contract section 4.
    fn default() -> Self {
        MetaConfig {
            protocol_version: PROTOCOL_VERSION.to_string(),
            master_seed: 1,
            variation: VariationMode::Full,
            population: 256,
            tournament: 10,
            batch: 64,
            max_batches: 10_000_000,
            dims_search: vec![8, 16],
            n_train_choices: vec![16, 64, 256],
            n_valid_search: 64,
            panel: PanelConfig::default(),
            hurdle: HurdleConfig::default(),
            fec_capacity: 1_000_000,
            dsl: DslLimits::default(),
            weights: MetaWeights::default(),
            select: SelectConfig::default(),
            arc_cells: false,
            snapshot_every: 1_000,
            limits: Limits {
                max_total_flops: 1_500_000_000_000,
                max_total_seconds: 14_400.0,
                max_candidate_seconds: 30.0,
                ..Limits::default()
            },
            threads: 4,
            hidden: MetaHiddenConfig {
                secret_file: default_secret_file(),
                tiers: vec![Tier::HiddenInstance, Tier::HiddenGrammar],
                gap_threshold: 0.05,
                score_step: 0.01,
                run_query_budget: 12,
            },
        }
    }
}

/// Default global learner secret directory: `$HOME/.brut/learner`.
/// Relative to the working directory when HOME is not set.
pub fn default_secret_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();

    home.join(".brut").join("learner")
}

/// Default global learner secret file: `$HOME/.brut/learner/hidden.env`.
pub fn default_secret_file() -> PathBuf {
    default_secret_dir().join("hidden.env")
}

impl MetaConfig {
    /// Read and validate a `meta.json` file (unknown fields are rejected).
    pub fn load(path: &Path) -> Result<MetaConfig> {
        let file = std::fs::File::open(path)?;
        let text = read_limited(file, MAX_META_CONFIG_BYTES)?;
        let cfg: MetaConfig = serde_json::from_str(&text)?;

        cfg.validate()?;

        Ok(cfg)
    }

    /// Write this config as pretty JSON (atomic write).
    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;

        let mut bytes = serde_json::to_vec_pretty(self)?;

        bytes.push(b'\n');

        write_atomic(path, &bytes)
    }

    /// Ranges, protocol_version == PROTOCOL_VERSION, dims ⊆ SUPPORTED_F, panel sizes
    /// (hand_tasks == 10), score_step <= gap_threshold / 2, limits within HARD caps,
    /// hidden.tiers contains HiddenInstance and HiddenGrammar.
    ///
    /// Also refuses `arc_cells = true` (experiment E2 is off, errata D2). Whether the seeds
    /// fit the DSL limits is checked at `init_meta_run` and `meta_run` (it runs DSL code).
    pub fn validate(&self) -> Result<()> {
        let fail = |m: String| Err(BrutError::Invalid(format!("meta config: {m}")));

        if self.protocol_version != PROTOCOL_VERSION {
            return fail(format!(
                "protocol_version {:?} is not {PROTOCOL_VERSION:?}",
                self.protocol_version
            ));
        }

        if !(2..=HARD_MAX_POPULATION).contains(&self.population) {
            return fail(format!("population must be in 2..={HARD_MAX_POPULATION}"));
        }

        if self.tournament == 0 || self.tournament > self.population {
            return fail("tournament must be in 1..=population".into());
        }

        if !(1..=HARD_MAX_BATCH).contains(&self.batch) {
            return fail(format!("batch must be in 1..={HARD_MAX_BATCH}"));
        }

        if self.max_batches == 0 {
            return fail("max_batches must be >= 1".into());
        }

        let dims_ok = !self.dims_search.is_empty()
            && self.dims_search.iter().all(|f| SUPPORTED_F.contains(f));

        if !dims_ok {
            return fail(format!(
                "dims_search must be a non-empty subset of {SUPPORTED_F:?}"
            ));
        }

        let n_train_ok = !self.n_train_choices.is_empty()
            && self
                .n_train_choices
                .iter()
                .all(|n| (1..=HARD_MAX_N_TRAIN).contains(n));

        if !n_train_ok {
            return fail(format!(
                "n_train_choices must be non-empty, each in 1..={HARD_MAX_N_TRAIN}"
            ));
        }

        if !(1..=HARD_MAX_N_VALID).contains(&self.n_valid_search) {
            return fail(format!("n_valid_search must be in 1..={HARD_MAX_N_VALID}"));
        }

        self.validate_panel()?;

        if self.arc_cells {
            return fail("arc_cells (experiment E2) is disabled in Stage 1".into());
        }

        self.validate_search()?;
        self.validate_limits()?;
        self.validate_hidden()
    }

    /// Panel and rotation (errata E12).
    fn validate_panel(&self) -> Result<()> {
        let fail = |m: String| Err(BrutError::Invalid(format!("meta config: panel: {m}")));
        let p = &self.panel;
        let len = p.hand_tasks + p.grammar_tasks;

        if p.hand_tasks != 10 {
            return fail("hand_tasks must be 10".into());
        }

        if !(1..=SEARCH_POOL_SIZE).contains(&p.grammar_tasks) {
            return fail(format!("grammar_tasks must be in 1..={SEARCH_POOL_SIZE}"));
        }

        if p.rotate_every > 0 && !(1..=len).contains(&p.rotate_swap) {
            return fail(format!("rotate_swap must be in 1..={len}"));
        }

        if p.stage_a_positions.is_empty() {
            return fail("stage_a_positions must not be empty".into());
        }

        let mut seen = HashSet::new();

        for &pos in &p.stage_a_positions {
            if pos >= len || !seen.insert(pos) {
                return fail(format!("stage_a_positions must be distinct and < {len}"));
            }
        }

        Ok(())
    }

    /// Hurdle, cache, DSL limits, fitness weights, Stage C and snapshots.
    fn validate_search(&self) -> Result<()> {
        let fail = |m: String| Err(BrutError::Invalid(format!("meta config: {m}")));
        let len = self.panel.hand_tasks + self.panel.grammar_tasks;
        let h = &self.hurdle;

        if !(0.0..=1.0).contains(&h.quantile) {
            return fail("hurdle.quantile must be in [0, 1]".into());
        }

        if !(1..=HARD_MAX_HURDLE_WINDOW).contains(&h.window) || h.warmup > h.window {
            return fail(format!(
                "hurdle.window must be in 1..={HARD_MAX_HURDLE_WINDOW} and warmup <= window"
            ));
        }

        if !(1..=HARD_MAX_FEC_CAPACITY).contains(&self.fec_capacity) {
            return fail(format!(
                "fec_capacity must be in 1..={HARD_MAX_FEC_CAPACITY}"
            ));
        }

        let d = &self.dsl;
        let lens = [d.max_setup, d.max_predict, d.max_learn];

        if lens.iter().any(|n| !(1..=HARD_MAX_PROGRAM_LEN).contains(n)) {
            return fail(format!(
                "dsl max_setup, max_predict and max_learn must be in 1..={HARD_MAX_PROGRAM_LEN}"
            ));
        }

        let const_ok = d.const_abs_max.is_finite()
            && d.const_abs_max > 0.0
            && d.const_abs_max <= HARD_MAX_CONST_ABS;

        if !const_ok {
            return fail(format!(
                "dsl.const_abs_max must be finite, > 0 and <= {HARD_MAX_CONST_ABS}"
            ));
        }

        let w = &self.weights;
        let weight_ok = |v: f64| v.is_finite() && v >= 0.0;

        if !(0.0..=1.0).contains(&w.breadth) {
            return fail("weights.breadth must be in [0, 1]".into());
        }

        if !(1..=len).contains(&w.breadth_count) {
            return fail(format!("weights.breadth_count must be in 1..={len}"));
        }

        if !weight_ok(w.len_penalty) || !weight_ok(w.cost_penalty) {
            return fail("weights.len_penalty and cost_penalty must be finite and >= 0".into());
        }

        let s = &self.select;

        if s.every_batches == 0 {
            return fail("select.every_batches must be >= 1".into());
        }

        if !(1..=HARD_MAX_HOF).contains(&s.hof_size) {
            return fail(format!("select.hof_size must be in 1..={HARD_MAX_HOF}"));
        }

        if !(1..=HARD_MAX_HAND_PER_FAMILY).contains(&s.hand_per_family) {
            return fail(format!(
                "select.hand_per_family must be in 1..={HARD_MAX_HAND_PER_FAMILY}"
            ));
        }

        if !(1..=SELECT_POOL_SIZE).contains(&s.grammar_families) {
            return fail(format!(
                "select.grammar_families must be in 1..={SELECT_POOL_SIZE}"
            ));
        }

        if s.finalists == 0 || s.finalists > s.hof_size {
            return fail("select.finalists must be in 1..=hof_size".into());
        }

        // Errata K3: 0 turns the memory off; the cap is kept small (AC-11).
        if s.all_time_best > HARD_MAX_ALL_TIME_BEST {
            return fail(format!(
                "select.all_time_best must be in 0..={HARD_MAX_ALL_TIME_BEST}"
            ));
        }

        if s.retune_trials > HARD_MAX_TRIALS || s.retune_tasks == 0 {
            return fail(format!(
                "select.retune_trials must be <= {HARD_MAX_TRIALS} and retune_tasks >= 1"
            ));
        }

        if self.snapshot_every == 0 {
            return fail("snapshot_every must be >= 1".into());
        }

        Ok(())
    }

    /// Limits (hard caps as in `Config::validate`) and threads.
    fn validate_limits(&self) -> Result<()> {
        let fail = |m: String| Err(BrutError::Invalid(format!("meta config: {m}")));
        let l = &self.limits;
        let seconds_ok = |v: f64| v.is_finite() && v > 0.0;

        if !seconds_ok(l.max_candidate_seconds) || !seconds_ok(l.max_total_seconds) {
            return fail("time limits must be finite and > 0".into());
        }

        if l.max_total_flops == 0 {
            return fail("limits.max_total_flops must be >= 1".into());
        }

        let caps = [
            (
                "max_hidden_layers",
                l.max_hidden_layers,
                HARD_MAX_HIDDEN_LAYERS,
            ),
            ("max_width", l.max_width, HARD_MAX_WIDTH),
            ("max_params", l.max_params, HARD_MAX_PARAMS),
            ("max_epochs", l.max_epochs, HARD_MAX_EPOCHS),
        ];

        for (name, value, cap) in caps {
            if value > cap {
                return fail(format!("limits.{name} {value} > hard cap {cap}"));
            }
        }

        if self.threads > MAX_THREADS {
            return fail(format!("threads {} > {MAX_THREADS}", self.threads));
        }

        Ok(())
    }

    /// Hidden phase settings.
    fn validate_hidden(&self) -> Result<()> {
        let fail = |m: String| Err(BrutError::Invalid(format!("meta config: hidden: {m}")));
        let h = &self.hidden;

        if h.secret_file.as_os_str().is_empty() {
            return fail("secret_file must not be empty".into());
        }

        let required = [Tier::HiddenInstance, Tier::HiddenGrammar];

        if !required.iter().all(|t| h.tiers.contains(t)) {
            return fail("tiers must contain hidden_instance and hidden_grammar".into());
        }

        let mut seen = HashSet::new();

        if !h.tiers.iter().all(|t| t.is_hidden() && seen.insert(*t)) {
            return fail("tiers must be distinct hidden tiers".into());
        }

        if !h.gap_threshold.is_finite() || h.gap_threshold < 0.0 {
            return fail("gap_threshold must be finite and >= 0".into());
        }

        if !(MIN_HIDDEN_SCORE_STEP..1.0).contains(&h.score_step) {
            return fail(format!(
                "score_step must be in [{MIN_HIDDEN_SCORE_STEP}, 1)"
            ));
        }

        // Rounding down may cost at most half the threshold (as in Config::validate).
        if 2.0 * h.score_step > h.gap_threshold {
            return fail(format!(
                "score_step {} must be at most half of gap_threshold {}",
                h.score_step, h.gap_threshold
            ));
        }

        if !(1..=HARD_MAX_QUERY_BUDGET).contains(&h.run_query_budget) {
            return fail(format!(
                "run_query_budget must be in 1..={HARD_MAX_QUERY_BUDGET}"
            ));
        }

        Ok(())
    }

    /// Only max_batches, threads, limits.max_candidate_seconds, limits.max_total_seconds,
    /// limits.max_total_flops may change on resume (uses config::diff_fields).
    /// Returns dotted field names; empty means compatible.
    pub fn resume_conflicts(&self, earlier: &MetaConfig) -> Vec<String> {
        let neutral = |c: &MetaConfig| MetaConfig {
            max_batches: 0,
            threads: 0,
            limits: Limits {
                max_candidate_seconds: 0.0,
                max_total_seconds: 0.0,
                max_total_flops: 0,
                ..c.limits
            },
            ..c.clone()
        };

        let (Ok(now), Ok(before)) = (
            serde_json::to_value(neutral(self)),
            serde_json::to_value(neutral(earlier)),
        ) else {
            return vec!["config".to_string()];
        };

        let mut out = Vec::new();

        config::diff_fields(&now, &before, "", &mut out);

        out
    }
}

/// Where the evaluator binary and the global learner secret file are.
/// `LearnerHiddenResult` is defined in learner/types.rs.
#[derive(Clone, Debug)]
pub struct LearnerEvalSetup {
    pub eval_bin: PathBuf,
    pub secret_env: PathBuf,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetaRunSummary {
    pub batches_done: u64,
    pub children: u64,
    /// Task runs really executed (not served from the score cache).
    pub unique_evals: u64,
    pub fec_task_hit_rate: f64,
    pub hurdle_pass_rate: f64,
    /// Search-phase FLOPs (errata E6: one-time costs are not included).
    pub flops_used: u64,
    pub wall_seconds: f64,
    pub stop_reason: String,
    pub finalists: Vec<String>,
    pub hidden: Vec<LearnerHiddenResult>,
}

/// True when <run_dir>/meta.json exists.
pub fn is_meta_run(run_dir: &Path) -> bool {
    run_dir.join(META_CONFIG_FILE).is_file()
}

/// Create run_dir, write meta.json, create archive with MetaRunStarted and
/// ProtocolRegistered, write protocol.json. Refuses an existing archive.
///
/// Also refuses a directory that holds an MLP run (`config.json`) and a config whose DSL
/// limits do not admit the four seed programs.
pub fn init_meta_run(run_dir: &Path, cfg: &MetaConfig) -> Result<()> {
    cfg.validate()?;
    check_seeds_fit(&cfg.dsl)?;

    let archive_path = run_dir.join(ARCHIVE_FILE);

    if archive_path.exists() {
        return Err(BrutError::Invalid(format!(
            "{} already exists; refusing to overwrite",
            archive_path.display()
        )));
    }

    if run_dir.join(crate::evolve::CONFIG_FILE).exists() {
        return Err(BrutError::Invalid(format!(
            "{} holds an MLP run; use a new directory",
            run_dir.display()
        )));
    }

    std::fs::create_dir_all(run_dir)?;
    cfg.save(&run_dir.join(META_CONFIG_FILE))?;

    let protocol = protocol::protocol_descriptor();
    let protocol_hash = protocol::protocol_hash();
    let mut bytes = serde_json::to_vec_pretty(&protocol)?;

    bytes.push(b'\n');
    write_atomic(&run_dir.join(PROTOCOL_FILE), &bytes)?;

    let mut archive = Archive::create(&archive_path)?;

    let started = RecordKind::MetaRunStarted {
        config: Box::new(cfg.clone()),
        protocol_hash: protocol_hash.clone(),
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
    };

    archive.append(started, now_unix_ms())?;
    archive.append(
        RecordKind::ProtocolRegistered {
            protocol_hash,
            protocol,
        },
        now_unix_ms(),
    )?;

    Ok(())
}

/// Every seed program must pass `check(lim)`, else initial members could be invalid.
fn check_seeds_fit(lim: &DslLimits) -> Result<()> {
    for (name, g) in seeds::seed_programs() {
        g.check(lim).map_err(|e| {
            BrutError::Invalid(format!(
                "meta config: dsl limits do not admit the seed {name}: {e}"
            ))
        })?;
    }

    Ok(())
}

/// Full run (section 6.5–6.7). Resumes from the last BatchDigest. `eval = None` or
/// `no_hidden = true` skips the hidden phase (run stays open for more batches).
///
/// Steps: check meta.json against the archive (resume rules, protocol hash, finality);
/// append `MetaRunStarted` (one per invocation); load or compute the tuned baselines;
/// restore the last snapshot or build the initial population; batch loop with rotation,
/// periodic Stage C and digests; final phase (unless the kill switch stopped the run);
/// hidden phase; `RunFinished`; report.
///
/// Errors before `MetaRunStarted` append nothing. Later errors append a `SafetyEvent` and
/// `RunFinished` (as in MLP mode) and are returned.
pub fn meta_run(
    run_dir: &Path,
    eval: Option<&LearnerEvalSetup>,
    no_hidden: bool,
) -> Result<MetaRunSummary> {
    let cfg = MetaConfig::load(&run_dir.join(META_CONFIG_FILE))?;
    let archive_path = run_dir.join(ARCHIVE_FILE);

    if !archive_path.exists() {
        return Err(BrutError::Invalid(format!(
            "{} not found; create the run with brut init --mode learner",
            archive_path.display()
        )));
    }

    let mut archive = Archive::open(&archive_path)?;

    check_resume(&cfg, &archive)?;
    check_protocol(&archive)?;
    check_not_final(&cfg, &archive)?;
    check_seeds_fit(&cfg.dsl)?;

    let protocol_hash = archive.protocol_hash().unwrap_or_default().to_string();

    let started = RecordKind::MetaRunStarted {
        config: Box::new(cfg.clone()),
        protocol_hash,
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
    };

    archive.append(started, now_unix_ms())?;

    let eval = if no_hidden { None } else { eval };
    let result = run_inner(run_dir, &cfg, eval, &mut archive);

    // Records buffered by the search are written on every path out of the run.
    let flushed = archive.set_buffered(false);

    match result {
        Ok(summary) => flushed.map(|()| summary),
        Err(e) => {
            // Best effort: the original error matters more than a failed log write.
            let _ = archive.append(
                RecordKind::SafetyEvent {
                    what: format!("learner run aborted: {e}"),
                },
                now_unix_ms(),
            );
            let _ = archive.append(
                RecordKind::RunFinished {
                    reason: format!("error: {e}"),
                },
                now_unix_ms(),
            );

            Err(e)
        }
    }
}

/// Refuse a meta.json that defines another experiment than the archive (nothing is written).
/// Before the search started, meta.json may still change (the new config is recorded).
fn check_resume(cfg: &MetaConfig, archive: &Archive) -> Result<()> {
    let Some(experiment) = archive.meta_config() else {
        return Err(BrutError::Invalid(
            "the archive has no MetaRunStarted record; not a learner run".into(),
        ));
    };

    if !archive.learner_search_started() {
        return Ok(());
    }

    let conflicts = cfg.resume_conflicts(experiment);

    if conflicts.is_empty() {
        return Ok(());
    }

    Err(BrutError::Invalid(format!(
        "{META_CONFIG_FILE} changes {} compared with the experiment in this directory; \
         on resume only max_batches, threads, max_candidate_seconds, max_total_seconds and \
         max_total_flops may change (start a new run directory with brut init)",
        conflicts.join(", ")
    )))
}

/// Refuse to continue when this build measures differently than the registered protocol.
fn check_protocol(archive: &Archive) -> Result<()> {
    let registered = archive.protocol_hash().unwrap_or_default();
    let current = protocol::protocol_hash();

    if registered == current {
        return Ok(());
    }

    Err(BrutError::Invalid(format!(
        "protocol changed: the run registered {registered}, this build has {current}; \
         start a new run directory with brut init"
    )))
}

/// Batches done according to the last digest (0 without digest).
fn batches_done(archive: &Archive) -> u64 {
    archive
        .last_batch_digest()
        .map_or(0, |d| d.last_batch.saturating_add(1))
}

/// Refuse new batches once the archive holds a hidden result (contract section 6.7.6).
///
/// Hidden results must not steer the search. A rerun with max_batches <= batches done
/// is allowed: it only fills missing (finalist, tier) pairs of the same fixed list.
fn check_not_final(cfg: &MetaConfig, archive: &Archive) -> Result<()> {
    if archive.learner_hidden_evals().is_empty() {
        return Ok(());
    }

    let done = batches_done(archive);

    if cfg.max_batches <= done {
        return Ok(());
    }

    Err(BrutError::Invalid(format!(
        "this learner run already has a hidden evaluation, so it is finished after {done} \
         batches; more batches would let hidden results steer the search (set max_batches \
         to {done} to finish the hidden evaluation, or start a new run directory)"
    )))
}

/// Fixed inputs of one `meta_run` invocation.
struct Ctx<'a> {
    run_dir: &'a Path,
    cfg: &'a MetaConfig,
    settings: EvalSettings,
    stop: &'a StopSignal,
    probes: [TaskInstance; 2],
    proposer: RandomProgramMutator,
    seeds: Vec<(&'static str, LearnerGenome)>,
    seed_ids: Vec<String>,
    reference: LearnerGenome,
    pool: rayon::ThreadPool,
}

impl<'a> Ctx<'a> {
    fn new(run_dir: &'a Path, cfg: &'a MetaConfig, stop: &'a StopSignal) -> Result<Ctx<'a>> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.threads)
            .build()
            .map_err(|e| BrutError::Invalid(format!("thread pool: {e}")))?;
        let seeds = seeds::seed_programs();
        let seed_ids = seeds.iter().map(|(_, g)| g.id()).collect();

        Ok(Ctx {
            run_dir,
            cfg,
            settings: EvalSettings {
                max_seconds_per_task: cfg.limits.max_candidate_seconds,
            },
            stop,
            probes: protocol::probe_tasks(),
            proposer: RandomProgramMutator {
                mode: cfg.variation,
            },
            seeds,
            seed_ids,
            reference: seeds::reference_nn(),
            pool,
        })
    }

    /// One search-protocol run (attempt 0) that polls the kill switch.
    fn run(&self, g: &LearnerGenome, task: &TaskInstance, prep: &TaskPrep) -> TaskResult {
        let stop = || self.stop.requested();

        protocol::run_learner_prepared(g, task, prep, 0, &self.settings, &stop)
    }

    /// The reference program on every given task, in parallel. None when stopped.
    fn run_reference(&self, tasks: &[(&TaskInstance, &TaskPrep)]) -> Option<(Vec<f32>, u64)> {
        let results: Vec<TaskResult> = self.pool.install(|| {
            tasks
                .par_iter()
                .map(|(t, prep)| self.run(&self.reference, t, prep))
                .collect()
        });

        if results.iter().any(|r| r.abort == Some(Abort::Stopped)) {
            return None;
        }

        let flops = results
            .iter()
            .fold(0u64, |acc, r| acc.saturating_add(r.flops));
        let scores = results.iter().map(aulc_of).collect();

        Some((scores, flops))
    }
}

/// Preparations of every task, in order (PERF-3: computed once per task set, not per run).
fn prepare_all(tasks: &[TaskInstance]) -> Vec<TaskPrep> {
    tasks.iter().map(protocol::prepare).collect()
}

/// AULC of a result as a finite value in [0, 1].
fn aulc_of(r: &TaskResult) -> f32 {
    if r.aulc.is_finite() {
        r.aulc.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Errata K3: one member of the permanent all-time-best memory. Compact: the genome plus
/// its id, fingerprint and best observed SELECT Δ. The Δ is rotation-invariant (it is
/// measured on the fixed SELECT tasks of a Stage C round, not on the current search panel),
/// so a program the panel later disfavours is not forgotten. Not stored in the snapshot:
/// it is rebuilt on resume from the archive's SelectEval and LearnerCandidate records
/// (`rebuild_all_time_best`), so it never reintroduces the AC-11 hall-of-fame blow-up.
#[derive(Clone, Debug)]
struct AtbMember {
    id: String,
    genome: LearnerGenome,
    /// Behavioural fingerprint (16-hex value), used to skip members already in the population.
    fp: u64,
    /// Best Δ_select ever recorded for this program over the completed periodic Stage C rounds.
    select_delta: f64,
}

/// Mutable search state: what a snapshot holds, plus the panel and its reference scores.
struct State {
    /// Batches done (the next batch has this index).
    batch: u64,
    rotation: u64,
    next_age: u64,
    /// Oldest member first.
    population: VecDeque<PopMember>,
    /// Best member first (see `hof_order`).
    hof: Vec<PopMember>,
    hurdle: Hurdle,
    cache: ScoreCache,
    /// Periodic Stage C rounds done.
    select_round: u64,
    panel: Panel,
    /// Stream orders and interpreter seeds of the panel tasks (same order).
    prep: Vec<TaskPrep>,
    /// AULC of the untuned reference NN per panel position (ref_t in section 6.6).
    refs: Vec<f32>,
    /// Errata K3: permanent all-time-best memory, best Δ_select first (`atb_offer` order).
    /// Rebuilt on resume, so it is not part of `snapshot`.
    all_time_best: Vec<AtbMember>,
}

impl State {
    fn snapshot(&self, flops_used: u64) -> Snapshot {
        Snapshot {
            batch: self.batch,
            rotation: self.rotation,
            next_age: self.next_age,
            population: self.population.iter().cloned().collect(),
            hall_of_fame: self.hof.clone(),
            hurdle_values: self.hurdle.values(),
            flops_used,
            select_round: self.select_round,
        }
    }

    /// Best fitness in the hall of fame, else in the population (0 when both are empty).
    fn best_fitness(&self) -> f64 {
        let best_pop = self
            .population
            .iter()
            .map(|m| m.fitness)
            .max_by(f64::total_cmp);

        self.hof
            .first()
            .map(|m| m.fitness)
            .or(best_pop)
            .unwrap_or(0.0)
    }
}

/// How the search loop ended.
#[derive(Clone, Debug, PartialEq)]
enum SearchEnd {
    /// max_batches reached.
    Completed,
    /// Run time or FLOP budget used up: the final phase still runs (section 6.7.3).
    Limit(String),
    /// Kill switch: the final phase does not run.
    Killed(String),
}

impl SearchEnd {
    fn reason(&self) -> String {
        match self {
            SearchEnd::Completed => "completed".to_string(),
            SearchEnd::Limit(m) => m.clone(),
            SearchEnd::Killed(m) => format!("stopped: {m}"),
        }
    }
}

/// Result of a step that may be cut short by the kill switch (then it changed nothing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Done,
    Stopped,
}

/// The lineage chunk being written and its counters.
struct Chunk {
    writer: LineageWriter,
    /// First batch of the chunk (also its file number).
    first_batch: u64,
    stats: ChunkStats,
}

impl Chunk {
    fn new(run_dir: &Path, first_batch: u64) -> Chunk {
        Chunk {
            writer: LineageWriter::new(run_dir, first_batch),
            first_batch,
            stats: ChunkStats::default(),
        }
    }
}

fn run_inner(
    run_dir: &Path,
    cfg: &MetaConfig,
    eval: Option<&LearnerEvalSetup>,
    archive: &mut Archive,
) -> Result<MetaRunSummary> {
    let stop = StopSignal::new(run_dir);
    let mut guard = Guard::new(run_dir, &cfg.limits);
    let ctx = Ctx::new(run_dir, cfg, &stop)?;

    for dir in [PROGRAMS_DIR, LINEAGE_DIR, SNAPSHOT_DIR] {
        std::fs::create_dir_all(run_dir.join(dir))?;
    }

    if guard.stop_requested() {
        return finish_killed(archive, &guard, "kill switch present at start");
    }

    let Some(tb) = load_or_tune_baselines(&ctx, archive)? else {
        return finish_killed(archive, &guard, "kill switch during baseline tuning");
    };

    // The search writes its records in groups: at each Stage C round and each digest
    // (PERF-1). A crash loses at most the records since then; a resume from the last digest
    // recreates them.
    archive.set_buffered(true)?;

    let (mut st, mut chunk) = match archive.last_batch_digest().cloned() {
        Some(d) => {
            prune_unreferenced(run_dir, archive)?;

            let Some((st, flops)) = restore(&ctx, archive, &d)? else {
                return finish_killed(archive, &guard, "kill switch during resume");
            };

            guard.charge(flops);

            (st, Chunk::new(run_dir, d.last_batch + 1))
        }
        None => {
            prune_unreferenced(run_dir, archive)?;

            let mut chunk = Chunk::new(run_dir, 0);

            let Some(st) = initialise(&ctx, archive, &mut chunk)? else {
                return finish_killed(archive, &guard, "kill switch during initialisation");
            };

            guard.charge(chunk.stats.flops);

            (st, chunk)
        }
    };

    ensure_baseline_records(&ctx, archive, &tb, &st)?;

    let end = search_loop(&ctx, archive, &mut guard, &tb, &mut st, &mut chunk)?;

    if st.batch > chunk.first_batch {
        write_digest(&ctx, archive, &st, chunk, &guard)?;
    }

    // From here on every record is written at once (hidden results must be durable).
    archive.set_buffered(false)?;

    if let SearchEnd::Killed(m) = &end {
        log_event(archive, format!("learner run stopped: {m}"))?;

        return finish(archive, &guard, end.reason(), Vec::new());
    }

    if let SearchEnd::Limit(m) = &end {
        log_event(
            archive,
            format!("search stopped after {} batches: {m}", st.batch),
        )?;
    }

    let Some(finalists) = final_phase(&ctx, archive, &tb, &st)? else {
        log_event(archive, "final phase stopped: kill switch".to_string())?;

        return finish(
            archive,
            &guard,
            "stopped: kill switch during the final phase".into(),
            Vec::new(),
        );
    };

    if let Some(setup) = eval {
        hidden_phase(&ctx, archive, setup, &finalists)?;
    }

    let entries: Vec<LearnerEntry> = finalists.iter().map(|f| f.entry.clone()).collect();
    let metrics: Vec<FinalistMetrics> = finalists.iter().map(|f| f.metrics.clone()).collect();
    let ids = entries.iter().map(|e| e.id.clone()).collect();

    // RunFinished first (COR-6): the report's wall time ends there, so this report and one
    // regenerated later from the archive agree.
    let summary = finish(archive, &guard, end.reason(), ids)?;

    meta_report::write_report_with(run_dir, archive, cfg, &tb, &entries, &metrics)?;

    Ok(summary)
}

/// Log a kill-switch stop that happened before the search loop, then finish.
fn finish_killed(archive: &mut Archive, guard: &Guard, what: &str) -> Result<MetaRunSummary> {
    log_event(archive, format!("learner run stopped: {what}"))?;

    finish(archive, guard, format!("stopped: {what}"), Vec::new())
}

/// Append RunFinished and build the summary (counters over all digests of the run).
fn finish(
    archive: &mut Archive,
    guard: &Guard,
    reason: String,
    finalists: Vec<String>,
) -> Result<MetaRunSummary> {
    archive.append(
        RecordKind::RunFinished {
            reason: reason.clone(),
        },
        now_unix_ms(),
    )?;

    let mut total = ChunkStats::default();

    for d in archive.batch_digests() {
        total.add(&d.stats);
    }

    let lookups = total.unique_evals.saturating_add(total.fec_task_hits);
    let passed = total.children.saturating_sub(total.hurdled);

    let summary = MetaRunSummary {
        batches_done: batches_done(archive),
        children: total.children,
        unique_evals: total.unique_evals,
        fec_task_hit_rate: ratio(total.fec_task_hits, lookups),
        hurdle_pass_rate: ratio(passed, total.children),
        flops_used: guard.flops_used(),
        wall_seconds: guard.elapsed_seconds(),
        stop_reason: reason,
        finalists,
        hidden: archive.learner_hidden_evals(),
    };

    Ok(summary)
}

/// a / b, 0 when b = 0.
pub(crate) fn ratio(a: u64, b: u64) -> f64 {
    if b == 0 { 0.0 } else { a as f64 / b as f64 }
}

fn log_event(archive: &mut Archive, what: String) -> Result<()> {
    eprintln!("safety  {what}");
    archive.append(RecordKind::SafetyEvent { what }, now_unix_ms())?;

    Ok(())
}

/// Fingerprint as 16 hex chars.
fn fp_hex(fp: u64) -> String {
    format!("{fp:016x}")
}

/// Fingerprint from its 16-hex form (0 if malformed; snapshots are hash-checked).
fn fp_value(hex: &str) -> u64 {
    u64::from_str_radix(hex, 16).unwrap_or(0)
}

/// Serde name of a mutation kind (as in lineage records).
pub(crate) fn mutation_name(m: Mutation) -> &'static str {
    match m {
        Mutation::Insert => "insert",
        Mutation::Remove => "remove",
        Mutation::Randomize => "randomize",
        Mutation::Arg => "arg",
        Mutation::Const => "const",
    }
}

/// Serde name of a baseline id.
pub(crate) fn baseline_name(b: BaselineId) -> &'static str {
    match b {
        BaselineId::LinearSgd => "linear_sgd",
        BaselineId::TwoLayerNn => "two_layer_nn",
        BaselineId::Prototype => "prototype",
        BaselineId::Perceptron => "perceptron",
        BaselineId::MlpMatched => "mlp_matched",
    }
}

// ---------------------------------------------------------------------------------------
// Pure helpers (unit tested)
// ---------------------------------------------------------------------------------------

/// Hall-of-fame order: higher fitness first, ties by id.
fn hof_order(a: &PopMember, b: &PopMember) -> Ordering {
    b.fitness
        .total_cmp(&a.fitness)
        .then_with(|| a.id.cmp(&b.id))
}

/// Offer `cand` to the hall of fame (top `size` by fitness, one member per fingerprint).
/// Returns true when it entered.
fn hof_offer(hof: &mut Vec<PopMember>, cand: &PopMember, size: usize) -> bool {
    if size == 0 {
        return false;
    }

    if let Some(i) = hof.iter().position(|m| m.fp == cand.fp) {
        if hof_order(cand, &hof[i]) != Ordering::Less {
            return false;
        }

        hof.remove(i);
    } else if hof.len() >= size {
        let worse_than_last = hof
            .last()
            .is_some_and(|last| hof_order(cand, last) != Ordering::Less);

        if worse_than_last {
            return false;
        }

        hof.truncate(size - 1);
    }

    let at = hof.partition_point(|m| hof_order(m, cand) == Ordering::Less);

    hof.insert(at, cand.clone());

    true
}

/// Errata K3: offer a program to the all-time-best memory. Keeps at most `cap` members,
/// ranked by best observed Δ_select (ties by id), one member per id (the higher Δ wins).
/// The merge order is fixed, so the memory is a deterministic function of the offers and
/// matches `rebuild_all_time_best` over the same Stage C rounds.
fn atb_offer(atb: &mut Vec<AtbMember>, cand: AtbMember, cap: usize) {
    if cap == 0 {
        return;
    }

    if let Some(m) = atb.iter_mut().find(|m| m.id == cand.id) {
        if cand.select_delta > m.select_delta {
            m.select_delta = cand.select_delta;
        }
    } else {
        atb.push(cand);
    }

    atb.sort_by(|a, b| {
        b.select_delta
            .total_cmp(&a.select_delta)
            .then_with(|| a.id.cmp(&b.id))
    });
    atb.truncate(cap);
}

/// Errata K3: rebuild the all-time-best memory from the archive after a resume. Scans the
/// SelectEval records of the completed periodic Stage C rounds (`round < select_round`,
/// which excludes the final-phase rounds), keeps the best Δ_select per candidate and pairs
/// each with its genome from the LearnerCandidate records. Equivalent to the incremental
/// `atb_offer` over those rounds, so a resumed run keeps the same memory as an unbroken one.
fn rebuild_all_time_best(archive: &Archive, select_round: u64, cap: usize) -> Vec<AtbMember> {
    if cap == 0 {
        return Vec::new();
    }

    let mut best: HashMap<String, f64> = HashMap::new();

    for r in archive.records() {
        if let RecordKind::SelectEval {
            candidate_id,
            round,
            delta_select,
            ..
        } = &r.kind
            && *round < select_round
        {
            best.entry(candidate_id.clone())
                .and_modify(|d| {
                    if *delta_select > *d {
                        *d = *delta_select;
                    }
                })
                .or_insert(*delta_select);
        }
    }

    let mut atb: Vec<AtbMember> = best
        .into_iter()
        .filter_map(|(id, select_delta)| {
            archive.learner_get(&id).map(|e| AtbMember {
                id,
                fp: fp_value(&e.fp),
                genome: e.genome,
                select_delta,
            })
        })
        .collect();

    // A total order (Δ then id), so the HashMap's iteration order does not matter.
    atb.sort_by(|a, b| {
        b.select_delta
            .total_cmp(&a.select_delta)
            .then_with(|| a.id.cmp(&b.id))
    });
    atb.truncate(cap);
    atb
}

/// Errata K3: the finalist candidate pool — the hall of fame plus all-time-best members
/// that have left it, so a good program dropped from the rotating hall of fame can still
/// become a finalist. Deterministic order: hall of fame first, then all-time-best order.
/// The all-time-best members carry no panel scores (their Δ is re-measured fresh).
fn finalist_candidates(hof: &[PopMember], all_time_best: &[AtbMember]) -> Vec<PopMember> {
    let mut candidates: Vec<PopMember> = hof.to_vec();
    let hof_ids: HashSet<&str> = hof.iter().map(|m| m.id.as_str()).collect();

    for atb in all_time_best {
        if !hof_ids.contains(atb.id.as_str()) {
            candidates.push(PopMember {
                id: atb.id.clone(),
                genome: atb.genome.clone(),
                fp: fp_hex(atb.fp),
                age: 0,
                fitness: 0.0,
                task_scores: Vec::new(),
            });
        }
    }

    candidates
}

/// Tournament (aging evolution): `t` distinct members drawn uniformly; the best fitness
/// wins, ties go to the younger (higher age). `pop` must not be empty.
fn tournament(pop: &VecDeque<PopMember>, t: usize, rng: &mut ChaCha8Rng) -> usize {
    let n = pop.len();
    let t = t.clamp(1, n.max(1));
    let mut picked: Vec<usize> = Vec::with_capacity(t);

    while picked.len() < t {
        let i = rng.random_range(0..n);

        if !picked.contains(&i) {
            picked.push(i);
        }
    }

    let better = |a: usize, b: usize| {
        pop[a]
            .fitness
            .total_cmp(&pop[b].fitness)
            .then_with(|| pop[a].age.cmp(&pop[b].age))
    };

    picked
        .iter()
        .copied()
        .max_by(|&a, &b| better(a, b))
        .unwrap_or(0)
}

/// Mean of the panel scores M(g) (0 for an empty panel).
fn panel_mean(scores: &[f32]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }

    scores.iter().map(|&x| f64::from(x)).sum::<f64>() / scores.len() as f64
}

/// Fitness of section 6.6:
/// (1 - breadth) M + breadth B - len_penalty log10(1 + len) - cost_penalty log10(max(cost, 1)),
/// with B the mean of the `breadth_count` lowest advantages over the reference.
fn fitness(
    scores: &[f32],
    refs: &[f32],
    effective_len: usize,
    step_cost_f16: u64,
    w: &MetaWeights,
) -> f64 {
    let m = panel_mean(scores);

    let mut adv: Vec<f64> = scores
        .iter()
        .zip(refs)
        .map(|(&a, &r)| f64::from(a) - f64::from(r))
        .collect();

    adv.sort_by(f64::total_cmp);

    let k = w.breadth_count.min(adv.len());

    let breadth = if k == 0 {
        0.0
    } else {
        adv[..k].iter().sum::<f64>() / k as f64
    };

    let len_term = (1.0 + effective_len as f64).log10();
    let cost_term = (step_cost_f16.max(1) as f64).log10();

    let value = (1.0 - w.breadth) * m + w.breadth * breadth
        - w.len_penalty * len_term
        - w.cost_penalty * cost_term;

    if value.is_finite() {
        value
    } else {
        WORST_META_FITNESS
    }
}

/// For every slot, the first slot with the same fingerprint (its representative).
fn representatives(fps: &[u64]) -> Vec<usize> {
    let mut first: HashMap<u64, usize> = HashMap::with_capacity(fps.len());

    fps.iter()
        .enumerate()
        .map(|(s, fp)| *first.entry(*fp).or_insert(s))
        .collect()
}

/// Group index per item: items with equal keys share an index (order of first appearance).
fn group_indices<K: std::hash::Hash + Eq>(keys: impl Iterator<Item = K>) -> Vec<usize> {
    let mut index: HashMap<K, usize> = HashMap::new();

    keys.map(|k| {
        let next = index.len();

        *index.entry(k).or_insert(next)
    })
    .collect()
}

/// Family-weighted mean: mean over groups of the mean over their kept items.
/// Groups without kept items are skipped; 0 when nothing is kept.
fn grouped_mean(groups: &[usize], values: &[f64], keep: impl Fn(usize) -> bool) -> f64 {
    let n_groups = groups.iter().copied().max().map_or(0, |g| g + 1);
    let mut sums = vec![0.0f64; n_groups];
    let mut counts = vec![0usize; n_groups];

    for (i, (&g, &v)) in groups.iter().zip(values).enumerate() {
        if keep(i) {
            sums[g] += v;
            counts[g] += 1;
        }
    }

    let means: Vec<f64> = sums
        .iter()
        .zip(&counts)
        .filter(|(_, c)| **c > 0)
        .map(|(s, c)| s / *c as f64)
        .collect();

    if means.is_empty() {
        return 0.0;
    }

    means.iter().sum::<f64>() / means.len() as f64
}

/// Mean of `values` (0 when empty).
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.iter().sum::<f64>() / values.len() as f64
}

// ---------------------------------------------------------------------------------------
// Panel evaluation
// ---------------------------------------------------------------------------------------

/// One score lookup of a program on a panel position.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Lookup {
    pos: usize,
    aulc: f32,
    /// True when the task was run, false when the score came from the cache.
    ran: bool,
    /// False for Timeout results: they depend on the wall clock and are never cached.
    cacheable: bool,
}

/// Abort counts of task runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Aborts {
    nonfinite: u64,
    error: u64,
    timeout: u64,
    overcost: u64,
}

impl Aborts {
    fn count(&mut self, abort: Option<Abort>) {
        match abort {
            Some(Abort::NonFinite) => self.nonfinite += 1,
            Some(Abort::ErrorTooLarge) => self.error += 1,
            Some(Abort::Timeout) => self.timeout += 1,
            Some(Abort::OverCost) => self.overcost += 1,
            Some(Abort::Stopped) | None => {}
        }
    }
}

/// Scores of one program on the panel, with what merging needs.
#[derive(Clone, Debug, Default)]
struct PanelEval {
    /// AULC per panel position (0 for positions not evaluated).
    scores: Vec<f32>,
    /// Lookups in evaluation order (stage A first, then the other positions).
    lookups: Vec<Lookup>,
    mean_a: f64,
    /// True when at least one stage-A task was run (then mean_a enters the hurdle).
    stage_a_ran: bool,
    passed: bool,
    flops: u64,
    aborts: Aborts,
    stopped: bool,
}

impl PanelEval {
    /// Look up or run position `p`. False when the kill switch stopped the task.
    fn look(&mut self, p: usize, g: &LearnerGenome, fp: u64, st: &State, ctx: &Ctx<'_>) -> bool {
        let task = &st.panel.tasks[p];
        let prep = &st.prep[p];

        if let Some(a) = st.cache.get(fp, task.id) {
            let aulc = if a.is_finite() { a } else { 0.0 };

            self.scores[p] = aulc;
            self.lookups.push(Lookup {
                pos: p,
                aulc,
                ran: false,
                cacheable: true,
            });

            return true;
        }

        let r = ctx.run(g, task, prep);
        let aulc = aulc_of(&r);

        self.scores[p] = aulc;
        self.flops = self.flops.saturating_add(r.flops);
        self.aborts.count(r.abort);
        self.lookups.push(Lookup {
            pos: p,
            aulc,
            ran: true,
            cacheable: r.abort != Some(Abort::Timeout),
        });

        if r.abort == Some(Abort::Stopped) {
            self.stopped = true;

            return false;
        }

        true
    }

    /// Lookups served without a run.
    fn cached(&self) -> usize {
        self.lookups.iter().filter(|l| !l.ran).count()
    }

    fn runs(&self) -> usize {
        self.lookups.iter().filter(|l| l.ran).count()
    }
}

/// Evaluate a program on the panel (cache-aware; the cache is only read).
/// Stage A first; with a threshold, a stage-A mean below it ends the evaluation (hurdled).
fn eval_on_panel(
    g: &LearnerGenome,
    fp: u64,
    st: &State,
    ctx: &Ctx<'_>,
    threshold: Option<f64>,
) -> PanelEval {
    let n = st.panel.tasks.len();
    let stage_a = &st.panel.stage_a;
    let mut ev = PanelEval {
        scores: vec![0.0; n],
        lookups: Vec::with_capacity(n),
        ..PanelEval::default()
    };

    for &p in stage_a {
        if !ev.look(p, g, fp, st, ctx) {
            return ev;
        }
    }

    let a: Vec<f64> = stage_a.iter().map(|&p| f64::from(ev.scores[p])).collect();

    ev.mean_a = mean(&a);
    ev.stage_a_ran = ev.lookups.iter().any(|l| l.ran);

    if threshold.is_some_and(|t| ev.mean_a < t) {
        return ev;
    }

    for p in 0..n {
        if stage_a.contains(&p) {
            continue;
        }

        if !ev.look(p, g, fp, st, ctx) {
            return ev;
        }
    }

    ev.passed = true;

    ev
}

/// Stage the cache changes of one program's lookups (in lookup order).
fn stage_lookups(cache: &mut ScoreCache, panel: &Panel, fp: u64, lookups: &[Lookup]) {
    for l in lookups {
        let task = panel.tasks[l.pos].id;

        if !l.ran {
            cache.stage_touch(fp, task);
        } else if l.cacheable {
            cache.stage_insert(fp, task, l.aulc);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Initialisation, restore, digests
// ---------------------------------------------------------------------------------------

/// One proposed child (or initial member) with its fingerprint and static costs.
struct Proposal {
    genome: LearnerGenome,
    parent: Option<String>,
    mutation: Option<Mutation>,
    generator: String,
    fp: u64,
    effective_len: usize,
    step_cost_f16: u64,
}

impl Proposal {
    fn new(
        genome: LearnerGenome,
        parent: Option<String>,
        mutation: Option<Mutation>,
        generator: String,
        ctx: &Ctx<'_>,
    ) -> Proposal {
        let fp = protocol::fingerprint(&genome, &ctx.probes);
        let effective_len = genome.effective_len();
        let step_cost_f16 = genome.step_cost(16);

        Proposal {
            genome,
            parent,
            mutation,
            generator,
            fp,
            effective_len,
            step_cost_f16,
        }
    }

    fn member(&self, age: u64, scores: Vec<f32>, refs: &[f32], w: &MetaWeights) -> PopMember {
        PopMember {
            id: self.genome.id(),
            genome: self.genome.clone(),
            fp: fp_hex(self.fp),
            age,
            fitness: fitness(&scores, refs, self.effective_len, self.step_cost_f16, w),
            task_scores: scores,
        }
    }

    fn entry(&self, m: &PopMember, batch: u64, status: LearnerStatus) -> LearnerEntry {
        LearnerEntry {
            id: m.id.clone(),
            parents: self.parent.iter().cloned().collect(),
            generator: self.generator.clone(),
            batch,
            genome: m.genome.clone(),
            effective_len: self.effective_len,
            step_cost_f16: self.step_cost_f16,
            fp: m.fp.clone(),
            panel_mean: panel_mean(&m.task_scores),
            fitness: m.fitness,
            program_file: None,
            status,
        }
    }
}

/// Fingerprint fps, evaluate one program per distinct fingerprint (in parallel), and
/// return (evals, index of each item's eval). Items with the same fingerprint share the
/// first item's evaluation: the same scores the score cache would serve them later.
fn eval_distinct(
    items: &[Proposal],
    st: &State,
    ctx: &Ctx<'_>,
    threshold: Option<f64>,
) -> (Vec<PanelEval>, Vec<usize>, Vec<bool>) {
    let fps: Vec<u64> = items.iter().map(|p| p.fp).collect();
    let reps = representatives(&fps);
    let unique: Vec<usize> = (0..items.len()).filter(|&s| reps[s] == s).collect();

    let evals: Vec<PanelEval> = ctx.pool.install(|| {
        unique
            .par_iter()
            .map(|&s| eval_on_panel(&items[s].genome, fps[s], st, ctx, threshold))
            .collect()
    });

    let mut eval_index = vec![0usize; items.len()];

    for (k, &s) in unique.iter().enumerate() {
        eval_index[s] = k;
    }

    let of: Vec<usize> = reps.iter().map(|&r| eval_index[r]).collect();
    let is_rep: Vec<bool> = reps.iter().enumerate().map(|(s, &r)| r == s).collect();

    (evals, of, is_rep)
}

/// Count one child's lookups into the chunk counters (duplicates ran nothing themselves).
fn count_lookups(stats: &mut ChunkStats, ev: &PanelEval, is_rep: bool) {
    if is_rep {
        stats.unique_evals += ev.runs() as u64;
        stats.fec_task_hits += ev.cached() as u64;
        stats.aborts_nonfinite += ev.aborts.nonfinite;
        stats.aborts_error += ev.aborts.error;
        stats.aborts_timeout += ev.aborts.timeout;
    } else {
        stats.fec_task_hits += ev.lookups.len() as u64;
    }

    if ev.aborts.overcost > 0 {
        stats.cost_rejected += 1;
    }
}

/// Build and fully evaluate the initial population (no hurdle). None when stopped.
fn initialise(ctx: &Ctx<'_>, archive: &mut Archive, chunk: &mut Chunk) -> Result<Option<State>> {
    let cfg = ctx.cfg;
    let panel = families::search_panel(
        cfg.master_seed,
        0,
        &cfg.panel,
        &cfg.dims_search,
        &cfg.n_train_choices,
        cfg.n_valid_search,
    );

    let prep = prepare_all(&panel.tasks);
    let tasks: Vec<(&TaskInstance, &TaskPrep)> = panel.tasks.iter().zip(&prep).collect();

    let Some((refs, ref_flops)) = ctx.run_reference(&tasks) else {
        return Ok(None);
    };

    let mut st = State {
        batch: 0,
        rotation: 0,
        next_age: 0,
        population: VecDeque::with_capacity(cfg.population + 1),
        hof: Vec::with_capacity(cfg.select.hof_size + 1),
        hurdle: Hurdle::new(&cfg.hurdle),
        cache: ScoreCache::new(cfg.fec_capacity),
        select_round: 0,
        panel,
        prep,
        refs,
        all_time_best: Vec::new(),
    };

    let members: Vec<Proposal> = ctx.pool.install(|| {
        (0..cfg.population)
            .into_par_iter()
            .map(|i| initial_proposal(i, ctx))
            .collect()
    });

    let (evals, of, is_rep) = eval_distinct(&members, &st, ctx, None);

    if evals.iter().any(|e| e.stopped) {
        return Ok(None);
    }

    let mut flops = ref_flops;

    for (i, prop) in members.iter().enumerate() {
        let ev = &evals[of[i]];

        stage_lookups(&mut st.cache, &st.panel, prop.fp, &ev.lookups);
        count_lookups(&mut chunk.stats, ev, is_rep[i]);

        if is_rep[i] {
            flops = flops.saturating_add(ev.flops);
        }

        let member = prop.member(st.next_age, ev.scores.clone(), &st.refs, &cfg.weights);

        st.next_age += 1;

        chunk.writer.push(LineageRecord {
            id: member.id.clone(),
            parent: prop.parent.clone(),
            batch: 0,
            slot: u16::try_from(i).unwrap_or(u16::MAX),
            mutation: None,
            fp: member.fp.clone(),
            fitness: member.fitness,
            cached_tasks: cached_count(ev, is_rep[i]),
        });

        let entered = hof_offer(&mut st.hof, &member, cfg.select.hof_size);

        if i < ctx.seeds.len() {
            let entry = prop.entry(&member, 0, LearnerStatus::Seed);

            archive.append(RecordKind::LearnerCandidate(entry), now_unix_ms())?;
        } else if entered {
            let entry = prop.entry(&member, 0, LearnerStatus::HallOfFame);

            archive.append(RecordKind::LearnerCandidate(entry), now_unix_ms())?;
        }

        st.population.push_back(member);
    }

    st.cache.commit();
    chunk.stats.children += members.len() as u64;
    chunk.stats.flops = chunk.stats.flops.saturating_add(flops);

    eprintln!(
        "init    | {} members | best {:.4} | flops {:.3e}",
        members.len(),
        st.best_fitness(),
        flops as f64
    );

    Ok(Some(st))
}

/// Initial member i: seed i % 4 mutated min(i / 4, 3) times (contract section 5.4).
fn initial_proposal(i: usize, ctx: &Ctx<'_>) -> Proposal {
    let cfg = ctx.cfg;
    let genome = mutate::initial_member(i, cfg.master_seed, cfg.variation, &cfg.dsl);
    let n_seeds = ctx.seeds.len().max(1);

    let (parent, generator) = if i < ctx.seeds.len() {
        (None, format!("seed:{}", ctx.seeds[i].0))
    } else {
        let parent = ctx.seed_ids.get(i % n_seeds).cloned();

        (parent, INIT_GENERATOR.to_string())
    };

    Proposal::new(genome, parent, None, generator, ctx)
}

/// Lookups served without a run for this child (u8, saturating).
fn cached_count(ev: &PanelEval, is_rep: bool) -> u8 {
    let n = if is_rep {
        ev.cached()
    } else {
        ev.lookups.len()
    };

    u8::try_from(n).unwrap_or(u8::MAX)
}

/// Load the snapshot and cache of digest `d`. Returns the state and the search FLOPs
/// recorded so far. Rebuilding the panel references is not charged (it was charged once).
/// None when the kill switch stopped the reference runs.
fn restore(ctx: &Ctx<'_>, archive: &Archive, d: &BatchDigestEntry) -> Result<Option<(State, u64)>> {
    let cfg = ctx.cfg;
    let run_dir = ctx.run_dir;
    let snap = read_snapshot(&run_dir.join(&d.snapshot_file), &d.snapshot_sha256)?;
    let cache = read_run_cache(run_dir, &d.cache_sha256)?;

    if snap.batch != d.last_batch + 1 {
        return Err(BrutError::Integrity(format!(
            "snapshot {} holds batch {}, the digest says {}",
            d.snapshot_file,
            snap.batch,
            d.last_batch + 1
        )));
    }

    let panel = families::search_panel(
        cfg.master_seed,
        snap.rotation,
        &cfg.panel,
        &cfg.dims_search,
        &cfg.n_train_choices,
        cfg.n_valid_search,
    );

    let len = panel.tasks.len();
    let members_ok = snap
        .population
        .iter()
        .chain(&snap.hall_of_fame)
        .all(|m| m.task_scores.len() == len);

    if snap.population.len() != cfg.population || !members_ok {
        return Err(BrutError::Integrity(format!(
            "snapshot {} does not match the run config",
            d.snapshot_file
        )));
    }

    let prep = prepare_all(&panel.tasks);
    let tasks: Vec<(&TaskInstance, &TaskPrep)> = panel.tasks.iter().zip(&prep).collect();

    let Some((refs, _)) = ctx.run_reference(&tasks) else {
        return Ok(None);
    };

    // Errata K3: the all-time-best memory is not in the snapshot; rebuild it from the
    // archive so a resume keeps the same memory as an unbroken run (see the function doc).
    let all_time_best = rebuild_all_time_best(archive, snap.select_round, cfg.select.all_time_best);

    let st = State {
        batch: snap.batch,
        rotation: snap.rotation,
        next_age: snap.next_age,
        population: snap.population.into_iter().collect(),
        hof: snap.hall_of_fame,
        hurdle: Hurdle::from_values(&cfg.hurdle, snap.hurdle_values),
        cache: ScoreCache::from_snapshot(cache),
        select_round: snap.select_round,
        panel,
        prep,
        refs,
        all_time_best,
    };

    eprintln!(
        "resume  | batch {} | rotation {} | best {:.4}",
        st.batch,
        st.rotation,
        st.best_fitness()
    );

    Ok(Some((st, snap.flops_used)))
}

/// Delete lineage chunks and snapshots that no digest references (written after the last
/// digest by an interrupted run). Only files with our own name patterns are touched.
fn prune_unreferenced(run_dir: &Path, archive: &Archive) -> Result<()> {
    let mut keep: HashSet<String> = HashSet::new();

    for d in archive.batch_digests() {
        keep.insert(d.chunk_file.clone());
        keep.insert(d.snapshot_file.clone());
    }

    let patterns = [
        (LINEAGE_DIR, "chunk-", ".jsonl"),
        (SNAPSHOT_DIR, "pop-", ".json"),
    ];

    for (dir, prefix, suffix) in patterns {
        let Ok(entries) = std::fs::read_dir(run_dir.join(dir)) else {
            continue;
        };

        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let ours = name.starts_with(prefix) && name.ends_with(suffix);
            let rel = format!("{dir}/{name}");

            if ours && !keep.contains(&rel) {
                std::fs::remove_file(entry.path())?;
            }
        }
    }

    Ok(())
}

/// Finish the lineage chunk, write snapshot and cache, append the BatchDigest.
fn write_digest(
    ctx: &Ctx<'_>,
    archive: &mut Archive,
    st: &State,
    chunk: Chunk,
    guard: &Guard,
) -> Result<()> {
    let run_dir = ctx.run_dir;
    let Chunk {
        writer,
        first_batch,
        stats,
    } = chunk;
    let (chunk_file, chunk_sha256) = writer.finish()?;
    let (snapshot_file, snapshot_sha256) =
        write_snapshot(run_dir, &st.snapshot(guard.flops_used()))?;
    let cache_sha256 = write_cache_pending(run_dir, &st.cache.snapshot())?;

    let digest = BatchDigestEntry {
        first_batch,
        last_batch: st.batch - 1,
        chunk_file,
        chunk_sha256,
        snapshot_file,
        snapshot_sha256,
        cache_sha256,
        stats,
        best_fitness: st.best_fitness(),
    };

    let children = digest.stats.children;

    archive.append(RecordKind::BatchDigest(digest), now_unix_ms())?;

    // The digest must be on disk before the cache it names replaces the previous one.
    archive.flush()?;
    promote_cache(run_dir)?;

    eprintln!(
        "digest  | batches {first_batch}..={} | children {children} | flops {:.3e}",
        st.batch - 1,
        guard.flops_used() as f64
    );

    Ok(())
}

// ---------------------------------------------------------------------------------------
// Tuned baselines
// ---------------------------------------------------------------------------------------

/// Relative path of a program baseline's file: `programs/baseline-<id>.json`.
fn baseline_rel_path(b: BaselineId) -> String {
    format!("{PROGRAMS_DIR}/baseline-{}.json", baseline_name(b))
}

/// SHA-256 of the canonical JSON of `tb` with `hash` set to "" (contract section 4).
fn tuned_hash(tb: &TunedBaselines) -> Result<String> {
    let mut copy = tb.clone();

    copy.hash = String::new();

    Ok(sha256_hex(serde_json::to_string(&copy)?.as_bytes()))
}

/// Read `programs/tuned-baselines.json`, check its hash and that its programs equal the
/// archived Baseline entries. None when the file does not exist.
pub(crate) fn load_baselines(run_dir: &Path, archive: &Archive) -> Result<Option<TunedBaselines>> {
    let path = run_dir.join(PROGRAMS_DIR).join(TUNED_BASELINES_FILE);

    if !path.exists() {
        return Ok(None);
    }

    let meta = std::fs::metadata(&path)?;

    if !meta.is_file() || meta.len() > MAX_STORE_FILE_BYTES {
        return Err(BrutError::Integrity(format!(
            "{}: not a regular file of sane size",
            path.display()
        )));
    }

    let text = std::fs::read_to_string(&path)?;
    let tb: TunedBaselines = serde_json::from_str(&text)
        .map_err(|e| BrutError::Integrity(format!("{}: bad JSON: {e}", path.display())))?;

    if tuned_hash(&tb)? != tb.hash {
        return Err(BrutError::Integrity(format!(
            "{}: hash mismatch (file was changed)",
            path.display()
        )));
    }

    let recorded: Vec<LearnerEntry> = archive
        .learner_candidates()
        .into_iter()
        .filter(|e| e.status == LearnerStatus::Baseline)
        .collect();

    for (b, g) in &tb.programs {
        let generator = baseline_generator(*b);
        let archived = recorded.iter().find(|e| e.generator == generator);

        if let Some(e) = archived
            && &e.genome != g
        {
            return Err(BrutError::Integrity(format!(
                "{}: baseline {} differs from the archive",
                path.display(),
                baseline_name(*b)
            )));
        }
    }

    Ok(Some(tb))
}

fn baseline_generator(b: BaselineId) -> String {
    format!("baseline:{}", baseline_name(b))
}

/// Tuned baselines of this run: from the run directory, or computed once (≈ 3e10 FLOPs,
/// not charged to the search budget, errata E6; metered in a `LearnerCost` record) and
/// written to `programs/`. None when the kill switch stopped the tuning (checked between
/// tuning trials).
fn load_or_tune_baselines(ctx: &Ctx<'_>, archive: &mut Archive) -> Result<Option<TunedBaselines>> {
    if let Some(tb) = load_baselines(ctx.run_dir, archive)? {
        return Ok(Some(tb));
    }

    eprintln!("tuning  | equal-effort baseline tuning (one-time, not charged to the budget)");

    let stop = || ctx.stop.requested();

    let (tb, flops) = match baselines::tuned_baselines_with_stop(&ctx.settings, &stop) {
        Ok(tuned) => tuned,
        Err(BrutError::Stopped(_)) => return Ok(None),
        Err(e) => return Err(e),
    };

    if tuned_hash(&tb)? != tb.hash {
        return Err(BrutError::Integrity(
            "tuned baselines carry a wrong hash".into(),
        ));
    }

    for (b, g) in &tb.programs {
        write_program(&ctx.run_dir.join(baseline_rel_path(*b)), g)?;
    }

    let mut bytes = serde_json::to_vec_pretty(&tb)?;

    bytes.push(b'\n');
    write_atomic(
        &ctx.run_dir.join(PROGRAMS_DIR).join(TUNED_BASELINES_FILE),
        &bytes,
    )?;

    eprintln!(
        "tuning  | done: {flops:.3e} FLOPs (one-time, errata E6)",
        flops = flops as f64
    );

    let cost = RecordKind::LearnerCost {
        phase: CostPhase::BaselineTuning,
        flops,
    };

    archive.append(cost, now_unix_ms())?;

    Ok(Some(tb))
}

/// Append a Baseline LearnerCandidate per program baseline unless the archive has one.
/// Panel numbers are measured on the current panel (not charged, errata E6).
fn ensure_baseline_records(
    ctx: &Ctx<'_>,
    archive: &mut Archive,
    tb: &TunedBaselines,
    st: &State,
) -> Result<()> {
    let recorded: HashSet<String> = archive
        .learner_candidates()
        .into_iter()
        .filter(|e| e.status == LearnerStatus::Baseline)
        .map(|e| e.generator)
        .collect();

    for (b, g) in &tb.programs {
        let generator = baseline_generator(*b);

        if recorded.contains(&generator) {
            continue;
        }

        let fp = protocol::fingerprint(g, &ctx.probes);
        let ev = eval_on_panel(g, fp, st, ctx, None);

        if ev.stopped {
            return Ok(());
        }

        let effective_len = g.effective_len();
        let step_cost_f16 = g.step_cost(16);

        let entry = LearnerEntry {
            id: g.id(),
            parents: Vec::new(),
            generator,
            batch: st.batch,
            genome: g.clone(),
            effective_len,
            step_cost_f16,
            fp: fp_hex(fp),
            panel_mean: panel_mean(&ev.scores),
            fitness: fitness(
                &ev.scores,
                &st.refs,
                effective_len,
                step_cost_f16,
                &ctx.cfg.weights,
            ),
            program_file: Some(baseline_rel_path(*b)),
            status: LearnerStatus::Baseline,
        };

        archive.append(RecordKind::LearnerCandidate(entry), now_unix_ms())?;
    }

    Ok(())
}

/// Write a program as canonical JSON plus newline (atomic).
fn write_program(path: &Path, g: &LearnerGenome) -> Result<()> {
    let mut bytes = g.canonical_json().into_bytes();

    bytes.push(b'\n');

    write_atomic(path, &bytes)
}

// ---------------------------------------------------------------------------------------
// Search loop
// ---------------------------------------------------------------------------------------

/// Batch loop of section 6.5, from `st.batch` to `max_batches`.
fn search_loop(
    ctx: &Ctx<'_>,
    archive: &mut Archive,
    guard: &mut Guard,
    tb: &TunedBaselines,
    st: &mut State,
    chunk: &mut Chunk,
) -> Result<SearchEnd> {
    let cfg = ctx.cfg;
    let every = cfg.select.every_batches;

    // A Stage C round that the kill switch interrupted after its batch is redone first.
    while st.select_round < st.batch / every {
        if periodic_stage_c(ctx, archive, guard, tb, st, &mut chunk.stats)? == Step::Stopped {
            return Ok(SearchEnd::Killed("kill switch during Stage C".to_string()));
        }
    }

    for b in st.batch..cfg.max_batches {
        if let Err(e) = guard.check() {
            return match e {
                BrutError::Stopped(_) | BrutError::Budget(_) => Ok(classify_stop(&e, guard, ctx)),
                other => Err(other),
            };
        }

        // Rotations due before batch b: b / rotate_every. Counting from the state (not
        // from b alone) keeps a rotation that ran before an interrupted batch from
        // running twice after a resume.
        let due = b.checked_div(cfg.panel.rotate_every).unwrap_or(0);

        if st.rotation < due
            && rotate_panel(ctx, archive, guard, st, &mut chunk.stats)? == Step::Stopped
        {
            return Ok(SearchEnd::Killed(format!(
                "kill switch during the rotation before batch {b}"
            )));
        }

        if run_batch(b, ctx, archive, guard, st, chunk)? == Step::Stopped {
            return Ok(SearchEnd::Killed(format!("kill switch during batch {b}")));
        }

        st.batch = b + 1;

        if st.batch.is_multiple_of(every)
            && periodic_stage_c(ctx, archive, guard, tb, st, &mut chunk.stats)? == Step::Stopped
        {
            return Ok(SearchEnd::Killed(format!(
                "kill switch during Stage C after batch {b}"
            )));
        }

        if st.batch.is_multiple_of(PROGRESS_EVERY) {
            progress(st, &chunk.stats, guard);
        }

        if st.batch.is_multiple_of(cfg.snapshot_every) {
            let done = std::mem::replace(chunk, Chunk::new(ctx.run_dir, st.batch));

            write_digest(ctx, archive, st, done, guard)?;
        }
    }

    Ok(SearchEnd::Completed)
}

/// Kill switch (no final phase) versus time or FLOP limit (final phase runs).
fn classify_stop(e: &BrutError, guard: &Guard, ctx: &Ctx<'_>) -> SearchEnd {
    let killed = guard.stop_requested() || ctx.stop.requested();

    match e {
        BrutError::Stopped(m) if killed => SearchEnd::Killed(m.clone()),
        _ => SearchEnd::Limit(e.to_string()),
    }
}

/// One progress line on stderr.
fn progress(st: &State, stats: &ChunkStats, guard: &Guard) {
    let mut fit: Vec<f64> = st.population.iter().map(|m| m.fitness).collect();

    fit.sort_by(f64::total_cmp);

    let median = fit.get(fit.len() / 2).copied().unwrap_or(0.0);
    let lookups = stats.unique_evals.saturating_add(stats.fec_task_hits);

    eprintln!(
        "batch {:>7} | best {:.4} | pop median {:.4} | hurdle {} | cache hits {:.2} | flops {:.3e} | {:.1} s",
        st.batch,
        st.best_fitness(),
        median,
        st.hurdle
            .threshold()
            .map_or("off".to_string(), |t| format!("{t:.4}")),
        ratio(stats.fec_task_hits, lookups),
        guard.flops_used() as f64,
        guard.elapsed_seconds()
    );
}

/// Propose the child of slot `s` in batch `b` (tournament + proposer, or random search).
fn propose_child(b: u64, s: usize, st: &State, ctx: &Ctx<'_>) -> Proposal {
    let cfg = ctx.cfg;
    let tag = format!("batch/{b}/slot/{s}");
    let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(cfg.master_seed, &tag));

    if cfg.variation == VariationMode::RandomSearch && !ctx.seeds.is_empty() {
        // Errata E10: seed[r % 4] with exactly n mutations, n uniform in 1..=8.
        let r = rng.random_range(0..ctx.seeds.len());
        let n = rng.random_range(1..=8usize);
        let mut genome = ctx.seeds[r].1.clone();
        let mut last = Mutation::Const;

        for _ in 0..n {
            let (child, m) = mutate::mutate(&genome, cfg.variation, &mut rng, &cfg.dsl);

            genome = child;
            last = m;
        }

        let parent = ctx.seed_ids.get(r).cloned();

        return Proposal::new(
            genome,
            parent,
            Some(last),
            RANDOM_SEARCH_GENERATOR.to_string(),
            ctx,
        );
    }

    let winner = tournament(&st.population, cfg.tournament, &mut rng);
    let parent = &st.population[winner];
    let (child, m) = ctx.proposer.propose(&[&parent.genome], &mut rng, &cfg.dsl);
    let generator = format!("aging/{}", mutation_name(m));

    Proposal::new(child, Some(parent.id.clone()), Some(m), generator, ctx)
}

/// One batch: propose in parallel, evaluate distinct fingerprints in parallel, merge in
/// slot order. A batch that saw the kill switch changes nothing.
fn run_batch(
    b: u64,
    ctx: &Ctx<'_>,
    archive: &mut Archive,
    guard: &mut Guard,
    st: &mut State,
    chunk: &mut Chunk,
) -> Result<Step> {
    let cfg = ctx.cfg;
    let threshold = st.hurdle.threshold();
    let shared: &State = st;

    let proposals: Vec<Proposal> = ctx.pool.install(|| {
        (0..cfg.batch)
            .into_par_iter()
            .map(|s| propose_child(b, s, shared, ctx))
            .collect()
    });

    let (evals, of, is_rep) = eval_distinct(&proposals, shared, ctx, threshold);

    if evals.iter().any(|e| e.stopped) {
        return Ok(Step::Stopped);
    }

    let stats = &mut chunk.stats;
    let mut flops = 0u64;

    for (s, prop) in proposals.iter().enumerate() {
        let ev = &evals[of[s]];

        stage_lookups(&mut st.cache, &st.panel, prop.fp, &ev.lookups);
        count_lookups(stats, ev, is_rep[s]);
        stats.children += 1;

        if is_rep[s] {
            flops = flops.saturating_add(ev.flops);
        }

        if ev.stage_a_ran {
            st.hurdle.push(ev.mean_a);
        }

        if !ev.passed {
            stats.hurdled += 1;

            continue;
        }

        let member = prop.member(st.next_age, ev.scores.clone(), &st.refs, &cfg.weights);

        st.next_age += 1;

        chunk.writer.push(LineageRecord {
            id: member.id.clone(),
            parent: prop.parent.clone(),
            batch: b,
            slot: u16::try_from(s).unwrap_or(u16::MAX),
            mutation: prop.mutation,
            fp: member.fp.clone(),
            fitness: member.fitness,
            cached_tasks: cached_count(ev, is_rep[s]),
        });

        if hof_offer(&mut st.hof, &member, cfg.select.hof_size) {
            let entry = prop.entry(&member, b, LearnerStatus::HallOfFame);

            archive.append(RecordKind::LearnerCandidate(entry), now_unix_ms())?;
        }

        st.population.push_back(member);

        while st.population.len() > cfg.population {
            st.population.pop_front();
        }
    }

    st.cache.commit();
    stats.flops = stats.flops.saturating_add(flops);
    guard.charge(flops);

    Ok(Step::Done)
}

/// Rotate the panel (section 6.4) and re-score every population and hall-of-fame member on
/// the positions whose task changed. A rotation that saw the kill switch changes nothing.
fn rotate_panel(
    ctx: &Ctx<'_>,
    archive: &mut Archive,
    guard: &mut Guard,
    st: &mut State,
    stats: &mut ChunkStats,
) -> Result<Step> {
    let cfg = ctx.cfg;
    let rotation = st.rotation + 1;
    let panel = families::search_panel(
        cfg.master_seed,
        rotation,
        &cfg.panel,
        &cfg.dims_search,
        &cfg.n_train_choices,
        cfg.n_valid_search,
    );

    let len = panel.tasks.len().min(st.panel.tasks.len());
    let changed: Vec<usize> = (0..len)
        .filter(|&p| panel.tasks[p].id != st.panel.tasks[p].id)
        .collect();

    let genomes: Vec<(&LearnerGenome, u64)> = st
        .population
        .iter()
        .chain(&st.hof)
        .map(|m| (&m.genome, fp_value(&m.fp)))
        .collect();

    let fps: Vec<u64> = genomes.iter().map(|g| g.1).collect();
    let reps = representatives(&fps);
    let unique: Vec<usize> = (0..fps.len()).filter(|&k| reps[k] == k).collect();
    let prep = prepare_all(&panel.tasks);
    let target = (&panel, prep.as_slice());

    let rescored: Vec<PanelEval> = ctx.pool.install(|| {
        unique
            .par_iter()
            .map(|&k| rescore(genomes[k].0, fps[k], target, &changed, &st.cache, ctx))
            .collect()
    });

    // Errata K3: re-inject all-time-best members that are not in the population, scored on
    // the full new panel (all positions, no hurdle), so a good program the panel had
    // dropped can breed again. Scored here before any state change; the result is applied
    // in the commit below, so the re-injection is atomic with the rotation (a kill switch
    // during it changes nothing, which keeps a resume deterministic).
    let in_pop: HashSet<u64> = st.population.iter().map(|m| fp_value(&m.fp)).collect();
    let to_inject: Vec<usize> = (0..st.all_time_best.len())
        .filter(|&i| !in_pop.contains(&st.all_time_best[i].fp))
        .collect();
    let all_positions: Vec<usize> = (0..panel.tasks.len()).collect();
    let injected: Vec<PanelEval> = ctx.pool.install(|| {
        to_inject
            .par_iter()
            .map(|&i| {
                rescore(
                    &st.all_time_best[i].genome,
                    st.all_time_best[i].fp,
                    target,
                    &all_positions,
                    &st.cache,
                    ctx,
                )
            })
            .collect()
    });

    let new_tasks: Vec<(&TaskInstance, &TaskPrep)> = changed
        .iter()
        .map(|&p| (&panel.tasks[p], &prep[p]))
        .collect();

    let Some((new_refs, ref_flops)) = ctx.run_reference(&new_tasks) else {
        return Ok(Step::Stopped);
    };

    if rescored.iter().any(|e| e.stopped) || injected.iter().any(|e| e.stopped) {
        return Ok(Step::Stopped);
    }

    let mut eval_index = vec![0usize; fps.len()];

    for (k, &m) in unique.iter().enumerate() {
        eval_index[m] = k;
    }

    let mut flops = ref_flops;

    for (k, &fp) in fps.iter().enumerate() {
        let ev = &rescored[eval_index[reps[k]]];

        stage_lookups(&mut st.cache, &panel, fp, &ev.lookups);

        if reps[k] == k {
            flops = flops.saturating_add(ev.flops);
            stats.unique_evals += ev.runs() as u64;
            stats.fec_task_hits += ev.cached() as u64;
        } else {
            stats.fec_task_hits += ev.lookups.len() as u64;
        }
    }

    let n_pop = st.population.len();

    for k in 0..fps.len() {
        let ev = &rescored[eval_index[reps[k]]];

        let member = if k < n_pop {
            &mut st.population[k]
        } else {
            &mut st.hof[k - n_pop]
        };

        for &p in &changed {
            member.task_scores[p] = ev.scores[p];
        }
    }

    for (&p, &r) in changed.iter().zip(&new_refs) {
        st.refs[p] = r;
    }

    let refs = st.refs.clone();
    let w = &cfg.weights;

    for m in st.population.iter_mut().chain(st.hof.iter_mut()) {
        m.fitness = fitness(
            &m.task_scores,
            &refs,
            m.genome.effective_len(),
            m.genome.step_cost(16),
            w,
        );
    }

    // Errata K3: apply the re-injection scored above (atomic with the rotation). Each
    // injected member joins the population as a fresh young member, so the tournament can
    // draw it as a parent on the new panel.
    for (j, &i) in to_inject.iter().enumerate() {
        let ev = &injected[j];
        let fp = st.all_time_best[i].fp;
        let id = st.all_time_best[i].id.clone();
        let genome = st.all_time_best[i].genome.clone();

        stage_lookups(&mut st.cache, &panel, fp, &ev.lookups);
        flops = flops.saturating_add(ev.flops);
        stats.unique_evals += ev.runs() as u64;
        stats.fec_task_hits += ev.cached() as u64;

        let member = PopMember {
            fitness: fitness(
                &ev.scores,
                &refs,
                genome.effective_len(),
                genome.step_cost(16),
                w,
            ),
            id,
            fp: fp_hex(fp),
            age: st.next_age,
            task_scores: ev.scores.clone(),
            genome,
        };

        st.next_age += 1;
        st.population.push_back(member);

        while st.population.len() > cfg.population {
            st.population.pop_front();
        }
    }

    st.hof.sort_by(hof_order);
    st.cache.commit();
    st.panel = panel;
    st.prep = prep;
    st.rotation = rotation;
    stats.flops = stats.flops.saturating_add(flops);
    guard.charge(flops);

    let record = RecordKind::PanelRotated {
        rotation,
        panel_id: fp_hex(st.panel.id),
    };

    archive.append(record, now_unix_ms())?;

    Ok(Step::Done)
}

/// Scores of one program on the `changed` positions of a new panel (cache only read).
/// `target` is the new panel with the preparations of its tasks.
fn rescore(
    g: &LearnerGenome,
    fp: u64,
    target: (&Panel, &[TaskPrep]),
    changed: &[usize],
    cache: &ScoreCache,
    ctx: &Ctx<'_>,
) -> PanelEval {
    let (panel, prep) = target;
    let mut ev = PanelEval {
        scores: vec![0.0; panel.tasks.len()],
        ..PanelEval::default()
    };

    for &p in changed {
        let task = &panel.tasks[p];

        if let Some(a) = cache.get(fp, task.id) {
            let aulc = if a.is_finite() { a } else { 0.0 };

            ev.scores[p] = aulc;
            ev.lookups.push(Lookup {
                pos: p,
                aulc,
                ran: false,
                cacheable: true,
            });

            continue;
        }

        let r = ctx.run(g, task, &prep[p]);
        let aulc = aulc_of(&r);

        ev.scores[p] = aulc;
        ev.flops = ev.flops.saturating_add(r.flops);
        ev.lookups.push(Lookup {
            pos: p,
            aulc,
            ran: true,
            cacheable: r.abort != Some(Abort::Timeout),
        });

        if r.abort == Some(Abort::Stopped) {
            ev.stopped = true;

            return ev;
        }
    }

    ev.passed = true;

    ev
}

// ---------------------------------------------------------------------------------------
// Stage C and finalist metrics
// ---------------------------------------------------------------------------------------

/// A task set with the program baselines' results on it (they do not depend on g).
struct TaskSet {
    tasks: Vec<TaskInstance>,
    /// Stream orders and interpreter seeds of the tasks (same order).
    prep: Vec<TaskPrep>,
    /// Family index per task (families weighted equally).
    groups: Vec<usize>,
    /// Program baselines in `tb.programs` order: AULC per task.
    base: Vec<(BaselineId, Vec<f64>)>,
}

/// Run the program baselines on `tasks`. None when stopped. Adds the work to `flops`.
fn task_set(
    tasks: Vec<TaskInstance>,
    tb: &TunedBaselines,
    ctx: &Ctx<'_>,
    flops: &mut u64,
) -> Option<TaskSet> {
    let groups = group_indices(tasks.iter().map(|t| t.family.clone()));
    let prep = prepare_all(&tasks);
    let jobs: Vec<(usize, usize)> = (0..tb.programs.len())
        .flat_map(|b| (0..tasks.len()).map(move |t| (b, t)))
        .collect();

    let results: Vec<TaskResult> = ctx.pool.install(|| {
        jobs.par_iter()
            .map(|&(b, t)| ctx.run(&tb.programs[b].1, &tasks[t], &prep[t]))
            .collect()
    });

    if results.iter().any(|r| r.abort == Some(Abort::Stopped)) {
        return None;
    }

    for r in &results {
        *flops = flops.saturating_add(r.flops);
    }

    let n = tasks.len();
    let base = tb
        .programs
        .iter()
        .enumerate()
        .map(|(b, (id, _))| {
            let scores = results[b * n..(b + 1) * n]
                .iter()
                .map(|r| f64::from(aulc_of(r)))
                .collect();

            (*id, scores)
        })
        .collect();

    Some(TaskSet {
        tasks,
        prep,
        groups,
        base,
    })
}

/// A candidate against its compute-matched baselines B(g) on one task set.
struct Comparison {
    /// Candidate results per task.
    cand: Vec<TaskResult>,
    /// AULC of b* per task.
    best: Vec<f64>,
    /// S(g): family-weighted mean AULC.
    s_g: f64,
    /// S(g) - S(b*), b* = argmax over B(g) of S(b) (first on ties).
    delta: f64,
    /// b* (None when B(g) was empty).
    best_id: Option<BaselineId>,
}

/// Run g and the compute-matched MLP on every task of `set`; compare with B(g).
/// None when stopped. Adds the work to `flops`.
fn compare(
    g: &LearnerGenome,
    set: &TaskSet,
    tb: &TunedBaselines,
    ctx: &Ctx<'_>,
    flops: &mut u64,
) -> Option<Comparison> {
    compare_all(&[g], set, tb, ctx, flops)?.pop()
}

/// What one MlpMatched fit depends on besides the recipe: the task (index into the set)
/// and the candidate's training cost on it.
type MlpKey = (usize, TrainingCost);

/// The MlpMatched budget input of candidate g on task t (COR-1: training cost only).
fn mlp_key(g: &LearnerGenome, t: &TaskInstance, index: usize) -> MlpKey {
    (index, TrainingCost::of(g, t.f))
}

/// `compare` for several candidates on one task set, in input order (same results).
///
/// PERF-2: an MlpMatched fit is a pure function of (task, recipe, candidate cost), and
/// hall-of-fame members often share their cost (same structure, other constants). Every
/// distinct fit runs once and is charged once. All runs share one parallel pass.
fn compare_all(
    genomes: &[&LearnerGenome],
    set: &TaskSet,
    tb: &TunedBaselines,
    ctx: &Ctx<'_>,
    flops: &mut u64,
) -> Option<Vec<Comparison>> {
    let n = set.tasks.len();
    let matched: Vec<Vec<BaselineId>> = genomes
        .iter()
        .map(|g| baselines::compute_matched_set(tb, g.step_cost(16)))
        .collect();

    let mut fits: Vec<MlpKey> = Vec::new();
    let mut fit_index: HashMap<MlpKey, usize> = HashMap::new();
    let mut fit_of: Vec<Option<Vec<usize>>> = Vec::with_capacity(genomes.len());

    for (g, m) in genomes.iter().zip(&matched) {
        if !m.contains(&BaselineId::MlpMatched) {
            fit_of.push(None);

            continue;
        }

        let per_task = set
            .tasks
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let key = mlp_key(g, t, i);

                *fit_index.entry(key).or_insert_with(|| {
                    fits.push(key);

                    fits.len() - 1
                })
            })
            .collect();

        fit_of.push(Some(per_task));
    }

    let jobs: Vec<(usize, usize)> = (0..genomes.len())
        .flat_map(|g| (0..n).map(move |t| (g, t)))
        .collect();

    let (runs, mlp): (Vec<TaskResult>, Vec<TaskResult>) = ctx.pool.install(|| {
        let runs = jobs
            .par_iter()
            .map(|&(g, t)| ctx.run(genomes[g], &set.tasks[t], &set.prep[t]))
            .collect();
        let mlp = fits
            .par_iter()
            .map(|&(t, cost)| baselines::mlp_matched_training(&set.tasks[t], &tb.mlp, cost))
            .collect();

        (runs, mlp)
    });

    if runs.iter().any(|r| r.abort == Some(Abort::Stopped)) {
        return None;
    }

    for r in runs.iter().chain(&mlp) {
        *flops = flops.saturating_add(r.flops);
    }

    let mlp_scores: Vec<f64> = mlp.iter().map(|m| f64::from(aulc_of(m))).collect();
    let mut out = Vec::with_capacity(genomes.len());

    for ((cand, matched), fit) in runs.chunks(n.max(1)).zip(&matched).zip(&fit_of) {
        let cand = cand.to_vec();
        let cand_scores: Vec<f64> = cand.iter().map(|r| f64::from(aulc_of(r))).collect();
        let s_g = grouped_mean(&set.groups, &cand_scores, |_| true);

        let mut best: Option<(f64, Vec<f64>, BaselineId)> = None;

        for b in matched {
            let scores = if *b == BaselineId::MlpMatched {
                match fit {
                    Some(idx) => idx.iter().map(|&k| mlp_scores[k]).collect(),
                    None => continue,
                }
            } else {
                match set.base.iter().find(|(id, _)| id == b) {
                    Some((_, s)) => s.clone(),
                    None => continue,
                }
            };

            let s_b = grouped_mean(&set.groups, &scores, |_| true);

            if best.as_ref().is_none_or(|(s, _, _)| s_b > *s) {
                best = Some((s_b, scores, *b));
            }
        }

        let best_id = best.as_ref().map(|(_, _, id)| *id);
        let (s_best, best) = match best {
            Some((s, scores, _)) => (s, scores),
            None => (0.0, vec![0.0; cand.len()]),
        };

        out.push(Comparison {
            cand,
            best,
            s_g,
            delta: s_g - s_best,
            best_id,
        });
    }

    Some(out)
}

/// One periodic Stage C round (section 6.6): SelectEval per hall-of-fame member. Charged
/// to the search budget. Does not feed back into fitness. Errata K3: each member's Δ_select
/// is offered to the all-time-best memory, which later feeds parent selection (rotation
/// re-injection) and finalist selection.
fn periodic_stage_c(
    ctx: &Ctx<'_>,
    archive: &mut Archive,
    guard: &mut Guard,
    tb: &TunedBaselines,
    st: &mut State,
    stats: &mut ChunkStats,
) -> Result<Step> {
    let round = st.select_round;
    let tasks = families::select_tasks(ctx.cfg.master_seed, round, &ctx.cfg.select);
    let mut flops = 0u64;

    let Some(set) = task_set(tasks, tb, ctx, &mut flops) else {
        return Ok(Step::Stopped);
    };

    let genomes: Vec<&LearnerGenome> = st.hof.iter().map(|m| &m.genome).collect();

    let Some(comparisons) = compare_all(&genomes, &set, tb, ctx, &mut flops) else {
        return Ok(Step::Stopped);
    };

    for (m, c) in st.hof.iter().zip(&comparisons) {
        let record = RecordKind::SelectEval {
            candidate_id: m.id.clone(),
            round,
            select_mean: c.s_g,
            delta_select: c.delta,
        };

        archive.append(record, now_unix_ms())?;
    }

    archive.flush()?;

    // Errata K3: offer every scored member to the all-time-best memory (in hof order, a
    // fixed order). Collected first so the mutable borrow of `all_time_best` is disjoint.
    let cap = ctx.cfg.select.all_time_best;
    let offers: Vec<AtbMember> = st
        .hof
        .iter()
        .zip(&comparisons)
        .map(|(m, c)| AtbMember {
            id: m.id.clone(),
            genome: m.genome.clone(),
            fp: fp_value(&m.fp),
            select_delta: c.delta,
        })
        .collect();

    for cand in offers {
        atb_offer(&mut st.all_time_best, cand, cap);
    }

    st.select_round += 1;
    stats.flops = stats.flops.saturating_add(flops);
    stats.stage_c_flops = stats.stage_c_flops.saturating_add(flops);
    guard.charge(flops);

    Ok(Step::Done)
}

/// Report numbers of one finalist (section 5.20), computed by the orchestrator only.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FinalistMetrics {
    /// S_select(g) on the final round.
    pub select_mean: f64,
    pub delta_select: f64,
    /// Δ restricted to F = 32 tasks (b* fixed from the full set; 0 without such tasks).
    pub delta_select_f32: f64,
    /// Δ restricted to F ∈ {8, 16} tasks.
    pub delta_select_f8_16: f64,
    /// Mean q per checkpoint over the SELECT tasks.
    pub q_by_checkpoint: [f64; 3],
    /// Mean final-checkpoint q per n_train over the SELECT tasks.
    pub q_by_n_train: Vec<(usize, f64)>,
    /// Δ_public on public_reference_tasks (no breadth term; quarantine input).
    pub delta_public: f64,
    /// Mean AULC on public_reference_tasks (for the winner's curse).
    pub public_mean: f64,
    /// b* on the final SELECT round and on the public reference tasks.
    pub best_select: Option<BaselineId>,
    pub best_public: Option<BaselineId>,
}

/// Finalist metrics from the two comparisons (pure).
fn metrics_from(sel: &TaskSet, c_sel: &Comparison, c_pub: &Comparison) -> FinalistMetrics {
    let diff: Vec<f64> = c_sel
        .cand
        .iter()
        .zip(&c_sel.best)
        .map(|(r, b)| f64::from(aulc_of(r)) - b)
        .collect();

    let f = |i: usize| sel.tasks[i].f;

    let mut q_by_checkpoint = [0.0; 3];

    for (c, q) in q_by_checkpoint.iter_mut().enumerate() {
        let values: Vec<f64> = c_sel.cand.iter().map(|r| f64::from(r.q[c])).collect();

        *q = mean(&values);
    }

    let mut n_trains: Vec<usize> = sel.tasks.iter().map(|t| t.n_train).collect();

    n_trains.sort_unstable();
    n_trains.dedup();

    let q_by_n_train = n_trains
        .iter()
        .map(|&n| {
            let values: Vec<f64> = sel
                .tasks
                .iter()
                .zip(&c_sel.cand)
                .filter(|(t, _)| t.n_train == n)
                .map(|(_, r)| f64::from(r.q[2]))
                .collect();

            (n, mean(&values))
        })
        .collect();

    let pub_scores: Vec<f64> = c_pub.cand.iter().map(|r| f64::from(aulc_of(r))).collect();

    FinalistMetrics {
        select_mean: c_sel.s_g,
        delta_select: c_sel.delta,
        delta_select_f32: grouped_mean(&sel.groups, &diff, |i| f(i) == 32),
        delta_select_f8_16: grouped_mean(&sel.groups, &diff, |i| f(i) != 32),
        q_by_checkpoint,
        q_by_n_train,
        delta_public: c_pub.delta,
        public_mean: mean(&pub_scores),
        best_select: c_sel.best_id,
        best_public: c_pub.best_id,
    }
}

/// One finalist of the final phase.
struct Finalist {
    entry: LearnerEntry,
    metrics: FinalistMetrics,
}

/// The finalists of the latest final phase, in the order the final phase recorded them
/// (its Δ_select ranking); empty before a final phase ran. Quarantine is applied.
pub fn current_finalists(archive: &Archive) -> Vec<LearnerEntry> {
    let latest = archive
        .records()
        .iter()
        .filter_map(|r| match &r.kind {
            RecordKind::LearnerCandidate(e) if e.generator == RETUNE_GENERATOR => Some(e.batch),
            _ => None,
        })
        .max();

    match latest {
        Some(batch) => retuned_entries(archive, batch),
        None => Vec::new(),
    }
}

/// Retuned finalist records of the final phase after `batch` batches, in record order
/// (first record per id), with Quarantined applied.
fn retuned_entries(archive: &Archive, batch: u64) -> Vec<LearnerEntry> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();

    for r in archive.records() {
        let RecordKind::LearnerCandidate(e) = &r.kind else {
            continue;
        };

        if e.generator != RETUNE_GENERATOR || e.batch != batch || !seen.insert(e.id.clone()) {
            continue;
        }

        let mut entry = e.clone();
        let quarantined = archive
            .learner_get(&e.id)
            .is_some_and(|x| x.status == LearnerStatus::Quarantined);

        if quarantined {
            entry.status = LearnerStatus::Quarantined;
        }

        out.push(entry);
    }

    out
}

/// SelectEval values of one round: candidate id -> (select_mean, delta_select).
fn recorded_select(archive: &Archive, round: u64) -> HashMap<String, (f64, f64)> {
    let mut out = HashMap::new();

    for r in archive.records() {
        if let RecordKind::SelectEval {
            candidate_id,
            round: rr,
            select_mean,
            delta_select,
        } = &r.kind
            && *rr == round
        {
            out.insert(candidate_id.clone(), (*select_mean, *delta_select));
        }
    }

    out
}

/// Final phase (section 6.7 steps 1-4). Idempotent: recorded Stage C values and retuned
/// finalists of this batch count are reused. Its cost is metered apart from the search
/// budget (errata E6). None when the kill switch stopped it.
fn final_phase(
    ctx: &Ctx<'_>,
    archive: &mut Archive,
    tb: &TunedBaselines,
    st: &State,
) -> Result<Option<Vec<Finalist>>> {
    let cfg = ctx.cfg;
    let round = FINAL_ROUND_BASE + st.batch;
    let mut flops = 0u64;
    let mut sel_set: Option<TaskSet> = None;

    // Errata K3: finalists are chosen from the hall of fame AND the all-time-best memory,
    // so a good program the panel dropped (and that left the hall of fame) can still be a
    // finalist. Their SELECT Δ is re-measured fresh on the final round below.
    let candidates = finalist_candidates(&st.hof, &st.all_time_best);

    // 1. Final Stage C round on the finalist candidates.
    let mut recorded = recorded_select(archive, round);
    let missing: Vec<&PopMember> = candidates
        .iter()
        .filter(|m| !recorded.contains_key(&m.id))
        .collect();

    if !missing.is_empty() {
        let tasks = families::select_tasks(cfg.master_seed, round, &cfg.select);

        let Some(set) = task_set(tasks, tb, ctx, &mut flops) else {
            return Ok(None);
        };

        let genomes: Vec<&LearnerGenome> = missing.iter().map(|m| &m.genome).collect();

        let Some(comparisons) = compare_all(&genomes, &set, tb, ctx, &mut flops) else {
            return Ok(None);
        };

        for (m, c) in missing.iter().zip(&comparisons) {
            let record = RecordKind::SelectEval {
                candidate_id: m.id.clone(),
                round,
                select_mean: c.s_g,
                delta_select: c.delta,
            };

            archive.append(record, now_unix_ms())?;
            recorded.insert(m.id.clone(), (c.s_g, c.delta));
        }

        sel_set = Some(set);
    }

    // Finalists: top `finalists` candidates by Δ_select, ties by id.
    let mut ranked: Vec<(&PopMember, f64)> = candidates
        .iter()
        .map(|m| (m, recorded.get(&m.id).map_or(f64::MIN, |v| v.1)))
        .collect();

    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.id.cmp(&b.0.id)));
    ranked.truncate(cfg.select.finalists);

    // 2. Retune (reusing finalists already recorded for this batch count).
    let existing = retuned_entries(archive, st.batch);

    let mut retune_tasks: Option<Vec<TaskInstance>> = None;
    let mut entries = Vec::with_capacity(ranked.len());

    for (pre, _) in &ranked {
        if let Some(e) = existing.iter().find(|e| e.parents.first() == Some(&pre.id)) {
            entries.push(e.clone());

            continue;
        }

        if ctx.stop.requested() {
            return Ok(None);
        }

        let tasks = retune_tasks.get_or_insert_with(|| retune_set(cfg));

        let Some(entry) = retune(ctx, archive, st, pre, tasks, &mut flops)? else {
            return Ok(None);
        };

        entries.push(entry);
    }

    // 3.-4. Metrics on the final round and on the public reference tasks.
    let sel = match sel_set {
        Some(set) => set,
        None => {
            let tasks = families::select_tasks(cfg.master_seed, round, &cfg.select);

            let Some(set) = task_set(tasks, tb, ctx, &mut flops) else {
                return Ok(None);
            };

            set
        }
    };

    let pub_tasks = families::public_reference_tasks(cfg.master_seed);

    let Some(pubref) = task_set(pub_tasks, tb, ctx, &mut flops) else {
        return Ok(None);
    };

    let genomes: Vec<&LearnerGenome> = entries.iter().map(|e| &e.genome).collect();

    let Some(c_sel) = compare_all(&genomes, &sel, tb, ctx, &mut flops) else {
        return Ok(None);
    };

    let Some(c_pub) = compare_all(&genomes, &pubref, tb, ctx, &mut flops) else {
        return Ok(None);
    };

    let mut finalists = Vec::with_capacity(entries.len());

    for ((entry, c_sel), c_pub) in entries.into_iter().zip(&c_sel).zip(&c_pub) {
        let metrics = metrics_from(&sel, c_sel, c_pub);

        if !recorded.contains_key(&entry.id) {
            let record = RecordKind::SelectEval {
                candidate_id: entry.id.clone(),
                round,
                select_mean: metrics.select_mean,
                delta_select: metrics.delta_select,
            };

            archive.append(record, now_unix_ms())?;
            recorded.insert(
                entry.id.clone(),
                (metrics.select_mean, metrics.delta_select),
            );
        }

        finalists.push(Finalist { entry, metrics });
    }

    log_event(
        archive,
        format!(
            "final phase after {} batches: {} finalists, {flops} FLOPs \
             (one-time cost, not charged to max_total_flops, errata E6)",
            st.batch,
            finalists.len()
        ),
    )?;

    let cost = RecordKind::LearnerCost {
        phase: CostPhase::FinalPhase,
        flops,
    };

    archive.append(cost, now_unix_ms())?;

    Ok(Some(finalists))
}

/// Retune set: the even indices of select_tasks(master, u64::MAX), at most retune_tasks.
fn retune_set(cfg: &MetaConfig) -> Vec<TaskInstance> {
    families::select_tasks(cfg.master_seed, RETUNE_ROUND, &cfg.select)
        .into_iter()
        .step_by(2)
        .take(cfg.select.retune_tasks)
        .collect()
}

/// Retune one finalist's constants, save it and record it (status Finalist).
/// None when the kill switch stopped its panel evaluation.
fn retune(
    ctx: &Ctx<'_>,
    archive: &mut Archive,
    st: &State,
    pre: &PopMember,
    tasks: &[TaskInstance],
    flops: &mut u64,
) -> Result<Option<LearnerEntry>> {
    let cfg = ctx.cfg;
    let tag = format!("retune/{}", pre.id);
    let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(cfg.master_seed, &tag));
    let trials = cfg.select.retune_trials;

    let (tuned, _) = baselines::tune_constants(&pre.genome, tasks, trials, &mut rng, &ctx.settings);

    // Static estimate of the tuning work: one evaluation of every task per trial.
    let per_trial = tasks.iter().fold(0u64, |acc, t| {
        acc.saturating_add(tuned.task_cost(t.f, t.steps, t.n_valid))
    });

    *flops = flops.saturating_add(per_trial.saturating_mul(trials as u64 + 1));

    let genome = match tuned.check(&cfg.dsl) {
        Ok(()) => tuned,
        Err(e) => {
            log_event(
                archive,
                format!(
                    "retune of {} gave an invalid program ({e}); kept the original",
                    pre.id
                ),
            )?;

            pre.genome.clone()
        }
    };

    let fp = protocol::fingerprint(&genome, &ctx.probes);
    let ev = eval_on_panel(&genome, fp, st, ctx, None);

    if ev.stopped {
        return Ok(None);
    }

    *flops = flops.saturating_add(ev.flops);

    let id = genome.id();
    let rel = format!("{PROGRAMS_DIR}/{id}.json");
    let effective_len = genome.effective_len();
    let step_cost_f16 = genome.step_cost(16);

    write_program(&ctx.run_dir.join(&rel), &genome)?;

    let entry = LearnerEntry {
        id,
        parents: vec![pre.id.clone()],
        generator: RETUNE_GENERATOR.to_string(),
        batch: st.batch,
        effective_len,
        step_cost_f16,
        fp: fp_hex(fp),
        panel_mean: panel_mean(&ev.scores),
        fitness: fitness(
            &ev.scores,
            &st.refs,
            effective_len,
            step_cost_f16,
            &cfg.weights,
        ),
        program_file: Some(rel),
        status: LearnerStatus::Finalist,
        genome,
    };

    archive.append(RecordKind::LearnerCandidate(entry.clone()), now_unix_ms())?;

    eprintln!(
        "final   | {} -> {} | panel mean {:.4}",
        pre.id, entry.id, entry.panel_mean
    );

    Ok(Some(entry))
}

/// Recompute the finalist metrics outside a run (for `write_report`). Deterministic; the
/// kill switch of `run_dir` aborts it (Stopped error).
pub(crate) fn recompute_metrics(
    run_dir: &Path,
    cfg: &MetaConfig,
    tb: &TunedBaselines,
    finalists: &[LearnerEntry],
) -> Result<Vec<FinalistMetrics>> {
    if finalists.is_empty() {
        return Ok(Vec::new());
    }

    let stop = StopSignal::new(run_dir);
    let ctx = Ctx::new(run_dir, cfg, &stop)?;
    let stopped = || BrutError::Stopped("kill switch while computing the report".into());
    let mut flops = 0u64;
    let mut sets: HashMap<u64, TaskSet> = HashMap::new();
    let pub_tasks = families::public_reference_tasks(cfg.master_seed);
    let pubref = task_set(pub_tasks, tb, &ctx, &mut flops).ok_or_else(stopped)?;
    let mut out = Vec::with_capacity(finalists.len());

    for e in finalists {
        let round = FINAL_ROUND_BASE + e.batch;

        let sel = match sets.entry(round) {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => {
                let tasks = families::select_tasks(cfg.master_seed, round, &cfg.select);

                v.insert(task_set(tasks, tb, &ctx, &mut flops).ok_or_else(stopped)?)
            }
        };

        let c_sel = compare(&e.genome, sel, tb, &ctx, &mut flops).ok_or_else(stopped)?;
        let c_pub = compare(&e.genome, &pubref, tb, &ctx, &mut flops).ok_or_else(stopped)?;

        out.push(metrics_from(sel, &c_sel, &c_pub));
    }

    Ok(out)
}

// ---------------------------------------------------------------------------------------
// Hidden phase and ARC check
// ---------------------------------------------------------------------------------------

/// Hidden phase (section 6.7.5): every finalist × every configured tier, skipping pairs
/// that already have a result. Errors and budget exhaustion are logged, not fatal.
fn hidden_phase(
    ctx: &Ctx<'_>,
    archive: &mut Archive,
    setup: &LearnerEvalSetup,
    finalists: &[Finalist],
) -> Result<()> {
    let cfg = ctx.cfg;
    let stop_file = ctx.run_dir.join(safety::STOP_FILE);
    let run_id = archive.run_id().unwrap_or_default().to_string();
    let protocol_hash = archive.protocol_hash().unwrap_or_default().to_string();

    for f in finalists {
        let id = &f.entry.id;
        let delta_public = f.metrics.delta_public;

        for &tier in &cfg.hidden.tiers {
            let done = archive.learner_hidden_evals();

            if done.iter().any(|h| &h.candidate_id == id && h.tier == tier) {
                continue;
            }

            if stop_file.exists() {
                return log_event(archive, "hidden evaluation stopped: kill switch".into());
            }

            // Quarantine is final: a later tier keeps an earlier HiddenInstance verdict.
            let quarantined_before = done.iter().any(|h| &h.candidate_id == id && h.quarantined);

            let req = LearnerEvalRequest {
                run_id: run_id.clone(),
                candidate_id: id.clone(),
                protocol_hash: protocol_hash.clone(),
                tier,
                program: f.entry.genome.clone(),
            };

            let resp = request_learner_eval(
                &setup.eval_bin,
                &setup.secret_env,
                cfg.hidden.run_query_budget,
                cfg.hidden.score_step,
                &req,
                Some(&stop_file),
            );

            let resp = match resp {
                Ok(r) => r,
                Err(BrutError::Stopped(m)) => {
                    return log_event(archive, format!("hidden evaluation stopped at {id}: {m}"));
                }
                Err(e) => {
                    log_event(
                        archive,
                        format!("hidden evaluation failed for {id} ({tier:?}): {e}"),
                    )?;

                    continue;
                }
            };

            let consistent = resp.candidate_id == *id
                && resp.tier == tier
                && resp.delta.is_finite()
                && (tier == Tier::HiddenGrammar || resp.decision.is_none());

            if !consistent {
                log_event(
                    archive,
                    format!("hidden evaluation for {id} ({tier:?}): inconsistent response"),
                )?;

                continue;
            }

            let quarantined = quarantined_before
                || (tier == Tier::HiddenInstance
                    && safety::gap_exceeded(delta_public, resp.delta, cfg.hidden.gap_threshold));

            eprintln!(
                "hidden  {id} | {tier:?} | delta {:.2} | public {delta_public:.4}{}{}",
                resp.delta,
                resp.decision
                    .map_or(String::new(), |d| format!(" | decision {d:?}")),
                if quarantined { " | QUARANTINED" } else { "" }
            );

            let record = RecordKind::LearnerHiddenEval {
                candidate_id: id.clone(),
                tier,
                delta: resp.delta,
                decision: resp.decision,
                delta_public: Some(delta_public),
                quarantined,
            };

            archive.append(record, now_unix_ms())?;
        }
    }

    Ok(())
}

/// External ARC check for one archived program (finalist or baseline).
///
/// The program travels inline to `brut-eval`, which owns the data paths, the budget and
/// the contamination checks. Appends `ExternalEval`; a failure is logged and returned.
/// Then report.json is regenerated (COR-6), so it lists the check; when that fails (no
/// tuned baselines yet, kill switch), a warning goes to stderr and the result stands.
pub fn arc_check(
    run_dir: &Path,
    candidate_id: &str,
    set: EvalSet,
    eval: &LearnerEvalSetup,
) -> Result<ArcEvalResponse> {
    let cfg = MetaConfig::load(&run_dir.join(META_CONFIG_FILE))?;
    let mut archive = Archive::open(&run_dir.join(ARCHIVE_FILE))?;

    let Some(entry) = archive.learner_get(candidate_id) else {
        return Err(BrutError::Invalid(format!(
            "unknown learner candidate {candidate_id}"
        )));
    };

    let eligible = matches!(
        entry.status,
        LearnerStatus::Finalist | LearnerStatus::Baseline | LearnerStatus::Quarantined
    );

    if !eligible {
        return Err(BrutError::Invalid(format!(
            "refused: {candidate_id} has status {:?}; only finalists and baselines are checked",
            entry.status
        )));
    }

    let req = ArcEvalRequest {
        run_id: archive.run_id().unwrap_or_default().to_string(),
        candidate_id: candidate_id.to_string(),
        protocol_hash: archive.protocol_hash().unwrap_or_default().to_string(),
        set,
        program: entry.genome,
        train_sources: vec![SourceId::Synthetic],
    };

    let stop_file = run_dir.join(safety::STOP_FILE);

    let resp = request_arc_eval(
        &eval.eval_bin,
        &eval.secret_env,
        cfg.hidden.run_query_budget,
        cfg.hidden.score_step,
        &req,
        Some(&stop_file),
    );

    let resp = match resp {
        Ok(r) => r,
        Err(e) => {
            let _ = log_event(
                &mut archive,
                format!("ARC check failed for {candidate_id} ({set:?}): {e}"),
            );

            return Err(e);
        }
    };

    let consistent = resp.candidate_id == candidate_id
        && resp.set == set
        && resp.official.is_finite()
        && resp.kaggle.is_finite()
        && resp.bca.is_none_or(f64::is_finite);

    if !consistent {
        return Err(BrutError::Eval(
            "brut-eval returned an inconsistent ARC response".into(),
        ));
    }

    let record = RecordKind::ExternalEval {
        candidate_id: candidate_id.to_string(),
        set,
        official: resp.official,
        kaggle: resp.kaggle,
        bca: resp.bca,
    };

    archive.append(record, now_unix_ms())?;
    drop(archive);

    if let Err(e) = meta_report::write_report(run_dir) {
        eprintln!("warning: report.json not refreshed after the ARC check: {e}");
    }

    Ok(resp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::types::{FamilyId, Group, Instr, Labels, LearnKind, OpCode};

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brut-meta-{name}-{}", std::process::id()));

        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");

        dir
    }

    fn genome(c: f32) -> LearnerGenome {
        LearnerGenome {
            setup: vec![Instr {
                op: OpCode::SConst,
                a: 0,
                b: 0,
                out: 3,
                c,
            }],
            predict: Vec::new(),
            learn: Vec::new(),
        }
    }

    fn member(id: &str, fp: u64, age: u64, fitness: f64) -> PopMember {
        PopMember {
            id: id.into(),
            genome: genome(0.5),
            fp: fp_hex(fp),
            age,
            fitness,
            task_scores: vec![0.5; 3],
        }
    }

    // ----- MetaConfig -----

    #[test]
    fn default_matches_the_contract() {
        let c = MetaConfig::default();

        assert_eq!(c.protocol_version, "learner-v3");
        assert_eq!(c.master_seed, 1);
        assert_eq!(c.variation, VariationMode::Full);
        assert_eq!((c.population, c.tournament, c.batch), (256, 10, 64));
        assert_eq!(c.max_batches, 10_000_000);
        assert_eq!(c.dims_search, vec![8, 16]);
        assert_eq!(c.n_train_choices, vec![16, 64, 256]);
        assert_eq!(c.n_valid_search, 64);
        assert_eq!(c.fec_capacity, 1_000_000);
        assert!(!c.arc_cells);
        assert_eq!(c.snapshot_every, 1_000);
        assert_eq!(c.limits.max_total_flops, 1_500_000_000_000);
        assert_eq!(c.limits.max_total_seconds, 14_400.0);
        assert_eq!(c.limits.max_candidate_seconds, 30.0);
        assert_eq!(c.limits.max_width, Limits::default().max_width);
        assert_eq!(c.threads, 4);
        assert_eq!(
            c.hidden.tiers,
            vec![Tier::HiddenInstance, Tier::HiddenGrammar]
        );
        assert_eq!(c.hidden.gap_threshold, 0.05);
        assert_eq!(c.hidden.score_step, 0.01);
        assert_eq!(c.hidden.run_query_budget, 12);
        assert!(c.hidden.secret_file.ends_with(".brut/learner/hidden.env"));

        // The JSON field names and order of the meta.json excerpt (contract section 7).
        let json = serde_json::to_string(&c).expect("json");

        assert!(json.starts_with(
            "{\"protocol_version\":\"learner-v3\",\"master_seed\":1,\"variation\":\"full\",\
             \"population\":256,\"tournament\":10,\"batch\":64,\"max_batches\":10000000,\
             \"dims_search\":[8,16],\"n_train_choices\":[16,64,256],\"n_valid_search\":64,"
        ));
        assert!(json.contains(
            "\"tiers\":[\"hidden_instance\",\"hidden_grammar\"],\"gap_threshold\":0.05,\
             \"score_step\":0.01,\"run_query_budget\":12}}"
        ));
    }

    #[test]
    fn default_is_valid() {
        assert!(MetaConfig::default().validate().is_ok());

        let tight = MetaConfig {
            hidden: MetaHiddenConfig {
                score_step: 0.025,
                gap_threshold: 0.05,
                ..MetaConfig::default().hidden
            },
            ..MetaConfig::default()
        };

        assert!(tight.validate().is_ok());
    }

    #[test]
    fn validate_rejects_nonsense() {
        let base = MetaConfig::default();
        let with = |f: &dyn Fn(&mut MetaConfig)| {
            let mut c = base.clone();

            f(&mut c);

            c
        };

        let bad = [
            with(&|c| c.protocol_version = "learner-v0".into()),
            with(&|c| c.population = 1),
            with(&|c| c.population = HARD_MAX_POPULATION + 1),
            with(&|c| c.tournament = 0),
            with(&|c| c.tournament = c.population + 1),
            with(&|c| c.batch = 0),
            with(&|c| c.batch = HARD_MAX_BATCH + 1),
            with(&|c| c.max_batches = 0),
            with(&|c| c.dims_search = Vec::new()),
            with(&|c| c.dims_search = vec![8, 12]),
            with(&|c| c.n_train_choices = vec![0]),
            with(&|c| c.n_train_choices = Vec::new()),
            with(&|c| c.n_valid_search = 0),
            with(&|c| c.panel.hand_tasks = 9),
            with(&|c| c.panel.grammar_tasks = 0),
            with(&|c| c.panel.grammar_tasks = 65),
            with(&|c| c.panel.rotate_swap = 0),
            with(&|c| c.panel.rotate_swap = 21),
            with(&|c| c.panel.stage_a_positions = Vec::new()),
            with(&|c| c.panel.stage_a_positions = vec![0, 20]),
            with(&|c| c.panel.stage_a_positions = vec![3, 3]),
            with(&|c| c.arc_cells = true),
            with(&|c| c.hurdle.quantile = f64::NAN),
            with(&|c| c.hurdle.quantile = 1.5),
            with(&|c| c.hurdle.window = 0),
            with(&|c| c.hurdle.warmup = c.hurdle.window + 1),
            with(&|c| c.fec_capacity = 0),
            with(&|c| c.fec_capacity = HARD_MAX_FEC_CAPACITY + 1),
            with(&|c| c.dsl.max_learn = 0),
            with(&|c| c.dsl.max_setup = HARD_MAX_PROGRAM_LEN + 1),
            with(&|c| c.dsl.const_abs_max = f32::NAN),
            with(&|c| c.dsl.const_abs_max = 0.0),
            with(&|c| c.weights.breadth = -0.1),
            with(&|c| c.weights.breadth_count = 0),
            with(&|c| c.weights.breadth_count = 21),
            with(&|c| c.weights.len_penalty = f64::INFINITY),
            with(&|c| c.weights.cost_penalty = -1.0),
            with(&|c| c.select.every_batches = 0),
            with(&|c| c.select.hof_size = 0),
            with(&|c| c.select.hand_per_family = 0),
            with(&|c| c.select.grammar_families = 33),
            with(&|c| c.select.finalists = 0),
            with(&|c| c.select.finalists = c.select.hof_size + 1),
            with(&|c| c.select.all_time_best = HARD_MAX_ALL_TIME_BEST + 1),
            with(&|c| c.select.retune_tasks = 0),
            with(&|c| c.select.retune_trials = HARD_MAX_TRIALS + 1),
            with(&|c| c.snapshot_every = 0),
            with(&|c| c.limits.max_total_seconds = 0.0),
            with(&|c| c.limits.max_candidate_seconds = f64::NAN),
            with(&|c| c.limits.max_total_flops = 0),
            with(&|c| c.limits.max_width = HARD_MAX_WIDTH + 1),
            with(&|c| c.threads = MAX_THREADS + 1),
            with(&|c| c.hidden.secret_file = PathBuf::new()),
            with(&|c| c.hidden.tiers = vec![Tier::HiddenInstance]),
            with(&|c| {
                c.hidden.tiers = vec![Tier::Search, Tier::HiddenInstance, Tier::HiddenGrammar]
            }),
            with(&|c| {
                c.hidden.tiers = vec![
                    Tier::HiddenInstance,
                    Tier::HiddenGrammar,
                    Tier::HiddenGrammar,
                ]
            }),
            with(&|c| c.hidden.gap_threshold = f64::NAN),
            with(&|c| c.hidden.score_step = 0.001),
            with(&|c| c.hidden.score_step = 1.0),
            with(&|c| c.hidden.score_step = 0.03),
            with(&|c| c.hidden.run_query_budget = 0),
        ];

        for (i, c) in bad.iter().enumerate() {
            assert!(c.validate().is_err(), "case {i} was accepted");
        }

        // A fixed panel (rotate_every = 0) does not need a swap count.
        let fixed = with(&|c| {
            c.panel.rotate_every = 0;
            c.panel.rotate_swap = 0;
        });

        assert!(fixed.validate().is_ok());

        // Errata K3: the all-time-best cap is valid from 0 (off) to the hard cap.
        assert!(with(&|c| c.select.all_time_best = 0).validate().is_ok());
        assert!(
            with(&|c| c.select.all_time_best = HARD_MAX_ALL_TIME_BEST)
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn resume_conflicts_name_experiment_fields_only() {
        let base = MetaConfig::default();
        let mut allowed = base.clone();

        allowed.max_batches = 5;
        allowed.threads = 1;
        allowed.limits.max_candidate_seconds = 1.0;
        allowed.limits.max_total_seconds = 2.0;
        allowed.limits.max_total_flops = 3;

        assert!(allowed.resume_conflicts(&base).is_empty());

        let mut changed = base.clone();

        changed.master_seed = 2;
        changed.variation = VariationMode::ConstantsOnly;
        changed.panel.rotate_every = 7;
        changed.limits.max_width = 8;
        changed.hidden.secret_file = PathBuf::from("/elsewhere/hidden.env");

        assert_eq!(
            changed.resume_conflicts(&base),
            vec![
                "hidden.secret_file",
                "limits.max_width",
                "master_seed",
                "panel.rotate_every",
                "variation"
            ]
        );
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir = tmp_dir("config");
        let path = dir.join(META_CONFIG_FILE);
        let c = MetaConfig {
            master_seed: 77,
            variation: VariationMode::RandomSearch,
            ..MetaConfig::default()
        };

        c.save(&path).expect("save");

        assert_eq!(MetaConfig::load(&path).expect("load"), c);
        assert!(is_meta_run(&dir));
        assert!(!is_meta_run(&dir.join("nope")));

        let mut value = serde_json::to_value(&c).expect("value");

        value["surprise"] = serde_json::json!(1);
        std::fs::write(&path, value.to_string()).expect("write");
        assert!(MetaConfig::load(&path).is_err());

        let mut value = serde_json::to_value(&c).expect("value");

        value["panel"]["hand_tasks"] = serde_json::json!(11);
        std::fs::write(&path, value.to_string()).expect("write");
        assert!(MetaConfig::load(&path).is_err());

        let bad = MetaConfig { batch: 0, ..c };

        assert!(bad.save(&dir.join("bad.json")).is_err());
        assert!(!dir.join("bad.json").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ----- pure helpers -----

    #[test]
    fn tournament_picks_the_best_and_breaks_ties_by_age() {
        let pop: VecDeque<PopMember> = vec![
            member("a", 1, 0, 0.1),
            member("b", 2, 1, 0.9),
            member("c", 3, 2, 0.5),
            member("d", 4, 3, 0.9),
        ]
        .into();

        // The whole population: the best wins; of the two at 0.9 the younger (d).
        let mut rng = ChaCha8Rng::seed_from_u64(1);

        for _ in 0..20 {
            assert_eq!(tournament(&pop, 4, &mut rng), 3);
        }

        // Size 1 is uniform; t larger than the population is clamped.
        let mut seen = [0usize; 4];

        for _ in 0..4000 {
            seen[tournament(&pop, 1, &mut rng)] += 1;
        }

        assert!(seen.iter().all(|&n| n > 800), "{seen:?}");
        assert_eq!(tournament(&pop, 99, &mut rng), 3);

        // Deterministic for a given rng state.
        let mut r1 = ChaCha8Rng::seed_from_u64(9);
        let mut r2 = ChaCha8Rng::seed_from_u64(9);
        let a: Vec<usize> = (0..50).map(|_| tournament(&pop, 2, &mut r1)).collect();
        let b: Vec<usize> = (0..50).map(|_| tournament(&pop, 2, &mut r2)).collect();

        assert_eq!(a, b);
    }

    #[test]
    fn tournament_draws_distinct_members() {
        // With t = n - 1 distinct draws, the worst member can only win if it is the
        // sole survivor, which never happens; duplicates would let it win.
        let pop: VecDeque<PopMember> = (0..5)
            .map(|i| member(&i.to_string(), i, i, i as f64))
            .collect();
        let mut rng = ChaCha8Rng::seed_from_u64(3);

        for _ in 0..500 {
            assert!(tournament(&pop, 4, &mut rng) >= 3);
        }
    }

    #[test]
    fn hall_of_fame_keeps_the_best_distinct_fingerprints() {
        let mut hof = Vec::new();

        assert!(hof_offer(&mut hof, &member("a", 1, 0, 0.5), 3));
        assert!(hof_offer(&mut hof, &member("b", 2, 1, 0.7), 3));
        assert!(hof_offer(&mut hof, &member("c", 3, 2, 0.6), 3));

        let ids = |h: &Vec<PopMember>| h.iter().map(|m| m.id.clone()).collect::<Vec<_>>();

        assert_eq!(ids(&hof), vec!["b", "c", "a"]);

        // Full: a worse one stays out, a better one replaces the worst.
        assert!(!hof_offer(&mut hof, &member("d", 4, 3, 0.4), 3));
        assert!(hof_offer(&mut hof, &member("e", 5, 4, 0.65), 3));
        assert_eq!(ids(&hof), vec!["b", "e", "c"]);

        // Same fingerprint: only a strictly better program replaces it.
        assert!(!hof_offer(&mut hof, &member("f", 2, 5, 0.7), 3));
        assert!(!hof_offer(&mut hof, &member("g", 3, 6, 0.55), 3));
        assert!(hof_offer(&mut hof, &member("h", 3, 7, 0.8), 3));
        assert_eq!(ids(&hof), vec!["h", "b", "e"]);

        // Ties in fitness are ordered by id; a tie with the last does not enter.
        assert!(!hof_offer(&mut hof, &member("z", 9, 8, 0.65), 3));
        assert!(hof_offer(&mut hof, &member("a2", 10, 9, 0.65), 3));
        assert_eq!(ids(&hof), vec!["h", "b", "a2"]);

        assert!(!hof_offer(&mut hof, &member("q", 11, 10, 1.0), 0));
    }

    fn atb(id: &str, fp: u64, delta: f64) -> AtbMember {
        AtbMember {
            id: id.into(),
            genome: genome(0.5),
            fp,
            select_delta: delta,
        }
    }

    /// Errata K3: the all-time-best memory keeps the best Δ per id, up to the cap.
    #[test]
    fn all_time_best_keeps_the_best_delta_up_to_the_cap() {
        let ids = |s: &Vec<AtbMember>| s.iter().map(|m| m.id.clone()).collect::<Vec<_>>();
        let mut set = Vec::new();

        atb_offer(&mut set, atb("a", 1, 0.1), 3);
        atb_offer(&mut set, atb("b", 2, 0.3), 3);
        atb_offer(&mut set, atb("c", 3, 0.2), 3);
        assert_eq!(ids(&set), vec!["b", "c", "a"]);

        // Full: a worse program stays out, a better one replaces the worst.
        atb_offer(&mut set, atb("d", 4, 0.05), 3);
        assert_eq!(ids(&set), vec!["b", "c", "a"]);
        atb_offer(&mut set, atb("e", 5, 0.25), 3);
        assert_eq!(ids(&set), vec!["b", "e", "c"]);

        // Same id: only a strictly higher Δ raises it; a lower Δ is ignored.
        atb_offer(&mut set, atb("b", 2, 0.2), 3);
        assert_eq!(set.iter().find(|m| m.id == "b").unwrap().select_delta, 0.3);
        atb_offer(&mut set, atb("c", 3, 0.9), 3);
        assert_eq!(ids(&set), vec!["c", "b", "e"]);

        // Cap 0 turns the memory off.
        let mut off = Vec::new();
        atb_offer(&mut off, atb("x", 9, 1.0), 0);
        assert!(off.is_empty());
    }

    /// Errata K3: the memory rebuilt from the archive on resume equals the memory built
    /// incrementally during the run, for any order of the offers (determinism), and it
    /// uses only the completed periodic Stage C rounds.
    #[test]
    fn all_time_best_rebuild_matches_incremental() {
        let (dir, mut a) = learner_archive("atb-rebuild");

        // Four distinct candidate genomes with distinct ids.
        let gs: Vec<LearnerGenome> = (0..4).map(|i| genome(0.1 * (i as f32 + 1.0))).collect();
        let cids: Vec<String> = gs.iter().map(|g| g.id()).collect();

        for (i, g) in gs.iter().enumerate() {
            let e = LearnerEntry {
                id: g.id(),
                parents: Vec::new(),
                generator: "aging/const".into(),
                batch: 0,
                genome: g.clone(),
                effective_len: g.effective_len(),
                step_cost_f16: g.step_cost(16),
                fp: fp_hex(100 + i as u64),
                panel_mean: 0.0,
                fitness: 0.0,
                program_file: None,
                status: LearnerStatus::HallOfFame,
            };

            a.append(RecordKind::LearnerCandidate(e), 2 + i as u64)
                .expect("append");
        }

        // (candidate index, round, delta). Rounds 0 and 1 are periodic; one final-phase
        // round and one future periodic round must be ignored by a rebuild at round 2.
        let offers = [
            (0usize, 0u64, 0.1f64),
            (1, 0, 0.5),
            (2, 0, 0.3),
            (1, 1, 0.2), // below c1's earlier 0.5, so ignored
            (2, 1, 0.9), // raises c2
            (3, 1, 0.4),
            (3, FINAL_ROUND_BASE + 7, 1.0), // final phase: excluded
            (0, 5, 2.0),                    // future periodic round: excluded at round 2
        ];

        for (k, (i, round, delta)) in offers.iter().enumerate() {
            a.append(
                RecordKind::SelectEval {
                    candidate_id: cids[*i].clone(),
                    round: *round,
                    select_mean: 0.0,
                    delta_select: *delta,
                },
                10 + k as u64,
            )
            .expect("append");
        }

        // Best over rounds 0 and 1: c2 = 0.9, c1 = 0.5, c3 = 0.4, c0 = 0.1; cap 3.
        let rebuilt = rebuild_all_time_best(&a, 2, 3);
        let rows: Vec<(String, f64)> = rebuilt
            .iter()
            .map(|m| (m.id.clone(), m.select_delta))
            .collect();

        assert_eq!(
            rows,
            vec![
                (cids[2].clone(), 0.9),
                (cids[1].clone(), 0.5),
                (cids[3].clone(), 0.4),
            ]
        );

        // The rebuilt members carry the real genomes (not just ids).
        assert_eq!(rebuilt[0].genome, gs[2]);

        // The incremental memory equals the rebuilt one, for either offer order.
        let replay = |order: &[usize]| {
            let mut set = Vec::new();

            for &k in order {
                let (i, round, delta) = offers[k];

                if round < 2 {
                    atb_offer(&mut set, atb(&cids[i], 100 + i as u64, delta), 3);
                }
            }

            set.iter()
                .map(|m| (m.id.clone(), m.select_delta))
                .collect::<Vec<_>>()
        };

        let forward: Vec<usize> = (0..offers.len()).collect();
        let backward: Vec<usize> = (0..offers.len()).rev().collect();

        assert_eq!(replay(&forward), rows);
        assert_eq!(replay(&backward), rows);

        // A resume at an earlier round sees only round 0: c1 = 0.5, c2 = 0.3, c0 = 0.1.
        let earlier = rebuild_all_time_best(&a, 1, 3);
        let early_ids: Vec<String> = earlier.iter().map(|m| m.id.clone()).collect();

        assert_eq!(
            early_ids,
            vec![cids[1].clone(), cids[2].clone(), cids[0].clone()]
        );

        // Cap 0 rebuilds an empty memory.
        assert!(rebuild_all_time_best(&a, 2, 0).is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Errata K3: a good program dropped from the rotating hall of fame is still a finalist
    /// candidate while it lives in the all-time-best memory (and is not duplicated).
    #[test]
    fn dropped_hof_member_recovered_from_all_time_best() {
        let hof = vec![member("h1", 1, 0, 0.9), member("h2", 2, 1, 0.8)];
        let mem = vec![
            atb("h1", 1, 0.4), // already in the hall of fame
            atb("good", 7, 0.5),
        ];

        let candidates = finalist_candidates(&hof, &mem);
        let ids: Vec<String> = candidates.iter().map(|m| m.id.clone()).collect();

        assert_eq!(ids, vec!["h1", "h2", "good"]);

        // The recovered member can rank first on Δ even though it left the hall of fame.
        let delta = |id: &str| -> f64 { if id == "good" { 0.5 } else { 0.1 } };
        let mut ranked: Vec<&PopMember> = candidates.iter().collect();

        ranked.sort_by(|a, b| {
            delta(&b.id)
                .total_cmp(&delta(&a.id))
                .then_with(|| a.id.cmp(&b.id))
        });

        assert_eq!(ranked[0].id, "good");

        // With an empty memory the pool is exactly the hall of fame.
        assert_eq!(finalist_candidates(&hof, &[]).len(), 2);
    }

    #[test]
    fn fitness_follows_section_6_6() {
        let w = MetaWeights {
            breadth: 0.25,
            breadth_count: 2,
            len_penalty: 0.005,
            cost_penalty: 0.002,
        };
        let scores = [0.8f32, 0.5, 0.25, 1.0];
        let refs = [0.5f32, 0.75, 0.25, 0.5];

        // M = 0.6375; advantages 0.3, -0.25, 0, 0.5 -> two lowest -0.25, 0 -> B = -0.125.
        let m = (0.8f64 as f32 as f64 + 0.5 + 0.25 + 1.0) / 4.0;
        let b = (-0.25 + 0.0) / 2.0;
        let expected =
            0.75 * m + 0.25 * b - 0.005 * (1.0f64 + 9.0).log10() - 0.002 * 1000f64.log10();

        assert!((fitness(&scores, &refs, 9, 1000, &w) - expected).abs() < 1e-6);

        // Zero cost counts as 1 (no log of 0); breadth_count beyond the panel is clamped.
        let wide = MetaWeights {
            breadth_count: 50,
            ..w.clone()
        };

        assert!(fitness(&scores, &refs, 0, 0, &wide).is_finite());
        assert_eq!(fitness(&[], &[], 0, 0, &w), 0.0);
        assert_eq!(panel_mean(&[0.5, 1.0]), 0.75);
    }

    #[test]
    fn representatives_and_groups() {
        assert_eq!(representatives(&[7, 3, 7, 7, 3, 9]), vec![0, 1, 0, 0, 1, 5]);
        assert!(representatives(&[]).is_empty());
        assert_eq!(
            group_indices(["a", "b", "a", "c"].into_iter()),
            vec![0, 1, 0, 2]
        );

        // Two families: (1 + 3) / 2 = 2 and 10 -> 6; the filter keeps one item per family.
        let groups = [0, 0, 1];
        let values = [1.0, 3.0, 10.0];

        assert_eq!(grouped_mean(&groups, &values, |_| true), 6.0);
        assert_eq!(grouped_mean(&groups, &values, |i| i != 1), 5.5);
        assert_eq!(grouped_mean(&groups, &values, |i| i == 2), 10.0);
        assert_eq!(grouped_mean(&groups, &values, |_| false), 0.0);
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn names_match_serde() {
        for m in [
            Mutation::Insert,
            Mutation::Remove,
            Mutation::Randomize,
            Mutation::Arg,
            Mutation::Const,
        ] {
            let json = serde_json::to_string(&m).expect("json");

            assert_eq!(json, format!("\"{}\"", mutation_name(m)));
        }

        for b in [
            BaselineId::LinearSgd,
            BaselineId::TwoLayerNn,
            BaselineId::Prototype,
            BaselineId::Perceptron,
            BaselineId::MlpMatched,
        ] {
            let json = serde_json::to_string(&b).expect("json");

            assert_eq!(json, format!("\"{}\"", baseline_name(b)));
        }

        assert_eq!(fp_hex(255), "00000000000000ff");
        assert_eq!(fp_value(&fp_hex(u64::MAX)), u64::MAX);
        assert_eq!(fp_value("zz"), 0);
        assert_eq!(ratio(1, 4), 0.25);
        assert_eq!(ratio(1, 0), 0.0);
    }

    fn task(family: &str, f: usize, n_train: usize) -> TaskInstance {
        TaskInstance {
            id: 0,
            family: FamilyId(family.into()),
            group: Group::Linear,
            f,
            kind: LearnKind::Regression,
            x_train: Vec::new(),
            y_train: Labels::Values(Vec::new()),
            y_train_late: None,
            drift_at: None,
            x_valid: Vec::new(),
            y_valid: Labels::Values(Vec::new()),
            n_train,
            n_valid: 0,
            steps: 384,
            seed: 0,
            changed_mask: None,
        }
    }

    fn result(q: [f32; 3]) -> TaskResult {
        TaskResult {
            q,
            aulc: 0.25 * q[0] + 0.25 * q[1] + 0.5 * q[2],
            abort: None,
            flops: 1,
            steps_done: 384,
        }
    }

    /// COR-1: Stage C and the final phase match the MLP to the training cost.
    #[test]
    fn mlp_key_is_the_training_cost() {
        let g = seeds::reference_nn();
        let t = task("lin-reg", 32, 64);
        let (index, cost) = mlp_key(&g, &t, 5);

        assert_eq!(index, 5);
        assert_eq!(cost, TrainingCost::of(&g, 32));
        assert_eq!(cost.until(t.steps), g.task_cost(32, t.steps, 0));
    }

    #[test]
    fn finalist_metrics_split_by_width_and_n_train() {
        let tasks = vec![
            task("lin-reg", 8, 16),
            task("lin-reg", 32, 64),
            task("quad-reg", 16, 16),
            task("quad-reg", 32, 64),
        ];
        let set = TaskSet {
            groups: group_indices(tasks.iter().map(|t| t.family.clone())),
            prep: prepare_all(&tasks),
            tasks,
            base: Vec::new(),
        };

        let cand = vec![
            result([0.2, 0.4, 0.6]),
            result([0.0, 0.0, 1.0]),
            result([0.4, 0.4, 0.4]),
            result([0.0, 0.0, 0.0]),
        ];
        let aulc: Vec<f64> = cand.iter().map(|r| f64::from(r.aulc)).collect();
        let c_sel = Comparison {
            best: vec![0.1, 0.1, 0.1, 0.1],
            s_g: 0.3,
            delta: 0.2,
            cand,
            best_id: Some(BaselineId::LinearSgd),
        };
        let c_pub = Comparison {
            cand: vec![result([1.0, 1.0, 1.0]), result([0.0, 0.0, 0.0])],
            best: vec![0.0, 0.0],
            s_g: 0.5,
            delta: 0.05,
            best_id: Some(BaselineId::MlpMatched),
        };

        let m = metrics_from(&set, &c_sel, &c_pub);

        assert_eq!(
            (m.best_select, m.best_public),
            (Some(BaselineId::LinearSgd), Some(BaselineId::MlpMatched))
        );
        assert_eq!(m.select_mean, 0.3);
        assert_eq!(m.delta_select, 0.2);
        assert_eq!(m.delta_public, 0.05);
        assert_eq!(m.public_mean, 0.5);

        // F = 32: one task per family -> mean of (aulc - 0.1) over the two families.
        let f32_expected = ((aulc[1] - 0.1) + (aulc[3] - 0.1)) / 2.0;
        let f816_expected = ((aulc[0] - 0.1) + (aulc[2] - 0.1)) / 2.0;

        assert!((m.delta_select_f32 - f32_expected).abs() < 1e-9);
        assert!((m.delta_select_f8_16 - f816_expected).abs() < 1e-9);

        let q0 = (0.2f32 as f64 + 0.0 + 0.4f32 as f64 + 0.0) / 4.0;

        assert!((m.q_by_checkpoint[0] - q0).abs() < 1e-9);
        assert_eq!(m.q_by_n_train.len(), 2);
        assert_eq!(m.q_by_n_train[0].0, 16);
        assert!((m.q_by_n_train[0].1 - (0.6f32 as f64 + 0.4f32 as f64) / 2.0).abs() < 1e-9);
        assert_eq!(m.q_by_n_train[1], (64, 0.5));
    }

    // ----- archive-based rules -----

    fn learner_archive(name: &str) -> (PathBuf, Archive) {
        let dir = tmp_dir(name);
        let path = dir.join(ARCHIVE_FILE);
        let mut a = Archive::create(&path).expect("create");

        a.append(
            RecordKind::MetaRunStarted {
                config: Box::new(MetaConfig::default()),
                protocol_hash: "e".repeat(64),
                crate_version: "test".into(),
            },
            1,
        )
        .expect("append");

        (dir, a)
    }

    fn entry(id: &str, parent: &str, generator: &str, batch: u64) -> LearnerEntry {
        LearnerEntry {
            id: id.into(),
            parents: vec![parent.into()],
            generator: generator.into(),
            batch,
            genome: genome(0.5),
            effective_len: 1,
            step_cost_f16: 1,
            fp: fp_hex(1),
            panel_mean: 0.5,
            fitness: 0.4,
            program_file: None,
            status: LearnerStatus::Finalist,
        }
    }

    fn digest(last: u64) -> BatchDigestEntry {
        BatchDigestEntry {
            first_batch: 0,
            last_batch: last,
            chunk_file: "lineage/chunk-000000.jsonl".into(),
            chunk_sha256: String::new(),
            snapshot_file: format!("snapshots/pop-{:09}.json", last + 1),
            snapshot_sha256: String::new(),
            cache_sha256: String::new(),
            stats: ChunkStats::default(),
            best_fitness: 0.0,
        }
    }

    #[test]
    fn resume_and_finality_rules() {
        let (dir, mut a) = learner_archive("rules");
        let mut cfg = MetaConfig::default();
        let mut other = cfg.clone();

        other.master_seed = 5;

        // Before the search started, any valid meta.json is accepted.
        assert!(check_resume(&cfg, &a).is_ok());
        assert!(check_resume(&other, &a).is_ok());

        a.append(
            RecordKind::LearnerCandidate(entry("s", "x", "seed:nn2-sgd", 0)),
            2,
        )
        .expect("append");

        cfg.max_batches = 3;
        assert!(check_resume(&cfg, &a).is_ok());

        let Err(BrutError::Invalid(m)) = check_resume(&other, &a) else {
            panic!("a changed seed must be refused");
        };

        assert!(m.contains("master_seed"), "{m}");

        // Without hidden results more batches are fine; with one, only <= batches done.
        a.append(RecordKind::BatchDigest(digest(9)), 2)
            .expect("append");
        assert_eq!(batches_done(&a), 10);
        assert!(check_not_final(&cfg, &a).is_ok());

        a.append(
            RecordKind::LearnerHiddenEval {
                candidate_id: "f".into(),
                tier: Tier::HiddenInstance,
                delta: 0.01,
                decision: None,
                delta_public: Some(0.02),
                quarantined: false,
            },
            3,
        )
        .expect("append");

        cfg.max_batches = 10;
        assert!(check_not_final(&cfg, &a).is_ok());

        cfg.max_batches = 11;
        assert!(matches!(
            check_not_final(&cfg, &a),
            Err(BrutError::Invalid(_))
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn finalists_and_select_records() {
        let (dir, mut a) = learner_archive("finalists");

        assert!(current_finalists(&a).is_empty());

        let records = [
            RecordKind::LearnerCandidate(entry("f1", "p1", RETUNE_GENERATOR, 10)),
            RecordKind::LearnerCandidate(entry("f2", "p2", RETUNE_GENERATOR, 10)),
            RecordKind::LearnerCandidate(entry("h1", "x", "aging/const", 12)),
            RecordKind::LearnerCandidate(entry("f3", "p3", RETUNE_GENERATOR, 20)),
            RecordKind::SelectEval {
                candidate_id: "p1".into(),
                round: FINAL_ROUND_BASE + 20,
                select_mean: 0.4,
                delta_select: 0.1,
            },
            RecordKind::SelectEval {
                candidate_id: "p1".into(),
                round: 3,
                select_mean: 0.3,
                delta_select: 0.0,
            },
        ];

        for (i, r) in records.into_iter().enumerate() {
            a.append(r, 2 + i as u64).expect("append");
        }

        // Only the latest final phase counts.
        let ids: Vec<String> = current_finalists(&a).into_iter().map(|e| e.id).collect();

        assert_eq!(ids, vec!["f3"]);

        // Record order of the final phase, even when an id appeared earlier (a retune
        // that changed nothing keeps the id), and quarantine is applied.
        let more = [
            RecordKind::LearnerCandidate(entry("f4", "p4", RETUNE_GENERATOR, 30)),
            RecordKind::LearnerCandidate(entry("h1", "h1", RETUNE_GENERATOR, 30)),
            RecordKind::LearnerHiddenEval {
                candidate_id: "f4".into(),
                tier: Tier::HiddenInstance,
                delta: 0.0,
                decision: None,
                delta_public: Some(0.2),
                quarantined: true,
            },
        ];

        for (i, r) in more.into_iter().enumerate() {
            a.append(r, 20 + i as u64).expect("append");
        }

        let latest = current_finalists(&a);
        let rows: Vec<(String, LearnerStatus)> =
            latest.iter().map(|e| (e.id.clone(), e.status)).collect();

        assert_eq!(
            rows,
            vec![
                ("f4".to_string(), LearnerStatus::Quarantined),
                ("h1".to_string(), LearnerStatus::Finalist)
            ]
        );
        assert_eq!(latest[1].batch, 30);

        let rec = recorded_select(&a, FINAL_ROUND_BASE + 20);

        assert_eq!(rec.len(), 1);
        assert_eq!(rec.get("p1"), Some(&(0.4, 0.1)));
        assert!(recorded_select(&a, 4).is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pruning_keeps_referenced_files_only() {
        let (dir, mut a) = learner_archive("prune");

        for sub in [LINEAGE_DIR, SNAPSHOT_DIR] {
            std::fs::create_dir_all(dir.join(sub)).expect("dir");
        }

        let files = [
            "lineage/chunk-000000.jsonl",
            "lineage/chunk-000010.jsonl",
            "lineage/notes.txt",
            "snapshots/pop-000000010.json",
            "snapshots/pop-000000013.json",
            "snapshots/cache-latest.json",
        ];

        for f in files {
            std::fs::write(dir.join(f), b"x").expect("write");
        }

        a.append(RecordKind::BatchDigest(digest(9)), 2)
            .expect("append");
        prune_unreferenced(&dir, &a).expect("prune");

        let exists: Vec<bool> = files.iter().map(|f| dir.join(f).exists()).collect();

        assert_eq!(exists, vec![true, false, true, true, false, true]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stage_c_rounds_do_not_collide() {
        // Periodic rounds count up from 0; final rounds sit above FINAL_ROUND_BASE and
        // below the retune round for any realistic batch count.
        let max_batches = 10_000_000u64;

        assert!(FINAL_ROUND_BASE > max_batches);
        assert!(FINAL_ROUND_BASE + max_batches < RETUNE_ROUND);
    }
}
