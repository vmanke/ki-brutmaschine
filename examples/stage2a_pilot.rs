//! Stage-2a pilot: wire the LlmProposer into the learner-mode search and run it here.
//!
//! The search gained a proposer seam (`meta_evolve::meta_run_with_proposer`); the default
//! `meta_run` still uses the random mutator, so existing runs are unchanged. This pilot
//! drives a short, in-process search with an `LlmProposer` over two backends, with no
//! hidden evaluation, no secret and no network:
//!
//! - **StubModel.** Deterministic placeholder that never returns a valid program, so the
//!   proposer retries twice and then falls back to the random mutator. This exercises the
//!   plumbing: integration, attempt logging, and run determinism.
//! - **OracleModel.** Returns one known passing learner (nlms-glm-4 from the positive
//!   control) as valid JSON, so the proposer ACCEPTS it. This shows the accept path and
//!   whether an accepted good learner reaches the finalists.
//!
//! The real question — can a real open model propose good learners — needs a real backend
//! on a host (ADR 0003). Here we only prove the machinery.
//!
//! Usage: cargo run --release --example stage2a_pilot -- [scratch_dir]
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use brutmaschine::backend::{
    GenRequest, GenResponse, ModelBackend, StopReason, StubModel, approx_tokens,
};
use brutmaschine::config::Limits;
use brutmaschine::learner::llm_proposer::{AttemptOutcome, LlmAttempt, LlmProposer};
use brutmaschine::learner::mutate::ProgramProposer;
use brutmaschine::learner::{
    HurdleConfig, LearnerGenome, MetaConfig, MetaHiddenConfig, PanelConfig, SelectConfig,
    VariationMode,
};
use brutmaschine::meta_evolve::{MetaRunSummary, init_meta_run, meta_run_with_proposer};
use brutmaschine::util::Result;

/// The known passing learner the oracle backend returns, verbatim, as its answer.
const ORACLE_GENOME_JSON: &str = include_str!("data/oracle_genome.json");

/// A backend that always answers with one fixed, valid learner program. It lets the pilot
/// exercise the proposer's ACCEPT path without a real model or a network.
struct OracleModel {
    name: String,
}

impl OracleModel {
    fn new() -> OracleModel {
        OracleModel {
            name: "oracle:nlms-glm-4".to_string(),
        }
    }
}

impl ModelBackend for OracleModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
        let text = ORACLE_GENOME_JSON.to_string();

        Ok(GenResponse {
            prompt_tokens: approx_tokens(&req.system) + approx_tokens(&req.prompt),
            completion_tokens: approx_tokens(&text),
            text,
            stop_reason: StopReason::Stop,
            backend: self.name.clone(),
        })
    }
}

fn main() {
    let scratch = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("stage2a_pilot"));

    println!("stage-2a pilot: LlmProposer in the learner-mode search");
    println!("no hidden evaluation, no secret, no network");
    println!("scratch: {}\n", scratch.display());

    let oracle: LearnerGenome =
        serde_json::from_str(ORACLE_GENOME_JSON).expect("oracle genome json parses");

    println!("oracle learner id: {}\n", oracle.id());

    // 1) StubModel: plumbing and determinism. Two runs with the same seed must match.
    let (d1, s1, a1, inv1) = run_pilot(
        "stub-a",
        &scratch,
        7,
        Box::new(StubModel::new("stub:pilot")),
    );
    let (_d2, s2, _a2, _) = run_pilot(
        "stub-b",
        &scratch,
        7,
        Box::new(StubModel::new("stub:pilot")),
    );

    report("StubModel", &d1, &s1, &a1, inv1, &oracle);

    let deterministic = s1.batches_done == s2.batches_done
        && s1.children == s2.children
        && s1.unique_evals == s2.unique_evals
        && s1.finalists == s2.finalists;

    println!(
        "  determinism (two runs, same seed): {}\n",
        if deterministic {
            "identical"
        } else {
            "DIFFERENT"
        }
    );

    // 2) OracleModel: the accept path. A valid model answer should be accepted and reach
    // the finalists (as a constants-tuned descendant, so we match by structure).
    let (d3, s3, a3, inv3) = run_pilot("oracle", &scratch, 7, Box::new(OracleModel::new()));

    report("OracleModel", &d3, &s3, &a3, inv3, &oracle);
}

/// Run one short no-hidden search with an `LlmProposer` over `backend`; return the run
/// summary, the drained proposer attempts and the invalid-answer count.
fn run_pilot(
    label: &str,
    scratch: &Path,
    seed: u64,
    backend: Box<dyn ModelBackend>,
) -> (PathBuf, MetaRunSummary, Vec<LlmAttempt>, usize) {
    let dir = scratch.join(label);

    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create run dir");

    // A non-empty placeholder path. With no_hidden = true it is never read, and the file
    // need not exist.
    let placeholder_secret = dir.join("unused-hidden.env");
    let cfg = tiny_config(seed, &placeholder_secret);

    init_meta_run(&dir, &cfg).expect("init meta run");

    // Keep a typed handle to read the proposer's own counters after the run; pass a shared
    // reference of it into the search as the trait object.
    let handle: Arc<LlmProposer> = Arc::new(LlmProposer::new(backend, VariationMode::Full));
    let proposer: Arc<dyn ProgramProposer> = handle.clone();

    let summary = meta_run_with_proposer(&dir, None, true, proposer).expect("meta run");
    let attempts = handle.drain_attempts();
    let invalid = handle.invalid_count();

    // Persist the per-attempt records to the run directory so the pilot's attempt
    // logging is auditable from its output, not only from the printed aggregates
    // (ADR 0002). This keeps the available metadata; it is not full request/response
    // replay, which belongs to a real backend on a host.
    if let Ok(text) = serde_json::to_string_pretty(&attempts) {
        let _ = std::fs::write(dir.join("attempts.json"), text);
    }

    (dir, summary, attempts, invalid)
}

/// True when any finalist program has the same structure as the oracle learner (the final
/// phase tunes only constants, so an accepted oracle descendant keeps its structure but
/// gets a new id). Finalist genomes are stored at `<run_dir>/programs/<id>.json`.
fn oracle_reached_finalists(dir: &Path, finalists: &[String], oracle: &LearnerGenome) -> bool {
    finalists.iter().any(|id| {
        let path = dir.join("programs").join(format!("{id}.json"));

        match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<LearnerGenome>(&text) {
                Ok(g) => g.same_structure(oracle),
                Err(_) => false,
            },
            Err(_) => false,
        }
    })
}

/// Print a short report of one pilot run.
fn report(
    label: &str,
    dir: &Path,
    s: &MetaRunSummary,
    attempts: &[LlmAttempt],
    invalid: usize,
    oracle: &LearnerGenome,
) {
    let mut accepted = 0;
    let mut retry = 0;
    let mut fallback = 0;
    let mut refusal = 0;

    for a in attempts {
        match a.outcome {
            AttemptOutcome::Accepted => accepted += 1,
            AttemptOutcome::Retry => retry += 1,
            AttemptOutcome::Fallback => fallback += 1,
            AttemptOutcome::Refusal => refusal += 1,
        }
    }

    let total = attempts.len();
    let accept_rate = if total > 0 {
        100.0 * accepted as f64 / total as f64
    } else {
        0.0
    };
    let oracle_in_finalists = oracle_reached_finalists(dir, &s.finalists, oracle);

    println!("{label}:");
    println!(
        "  batches {}, children {}, unique_evals {}, stop '{}'",
        s.batches_done, s.children, s.unique_evals, s.stop_reason
    );
    println!(
        "  proposer attempts {total}: accepted {accepted} ({accept_rate:.0} %), retry {retry}, fallback {fallback}, refusal {refusal}; invalid {invalid}"
    );
    println!("  finalists: {:?}", s.finalists);
    println!(
        "  oracle learner reached finalists (same structure): {}",
        if oracle_in_finalists { "yes" } else { "no" }
    );
}

/// A small, fast search configuration for the pilot. Mirrors the e2e `tiny_meta` shape.
fn tiny_config(seed: u64, placeholder_secret: &Path) -> MetaConfig {
    let base = MetaConfig::default();

    MetaConfig {
        master_seed: seed,
        variation: VariationMode::Full,
        population: 24,
        tournament: 4,
        batch: 8,
        max_batches: 3,
        dims_search: vec![8],
        panel: PanelConfig {
            rotate_every: 3,
            ..PanelConfig::default()
        },
        hurdle: HurdleConfig {
            quantile: 0.75,
            window: 64,
            warmup: 16,
        },
        fec_capacity: 100_000,
        select: SelectConfig {
            every_batches: 1_000,
            hof_size: 4,
            finalists: 2,
            retune_trials: 4,
            ..SelectConfig::default()
        },
        snapshot_every: 2,
        limits: Limits {
            max_total_flops: 1_000_000_000,
            max_total_seconds: 600.0,
            max_candidate_seconds: 30.0,
            ..base.limits
        },
        threads: 2,
        hidden: MetaHiddenConfig {
            secret_file: placeholder_secret.to_path_buf(),
            ..base.hidden.clone()
        },
        ..base
    }
}
