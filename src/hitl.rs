//! Human-in-the-loop pull-request mutation of candidates (ADR 0004).
//!
//! The AI drafts mutations as pull requests; a human reviews and merges before a
//! mutated candidate enters a run. This module holds the genome adapter and the pure
//! helpers both binaries use (`brut-propose`, `brut-ingest`). It is pure: no I/O, no
//! network, no git. The binaries do the file and stdin I/O; the git and pull-request
//! steps live in `scripts/`, so the core crate keeps no process or network dependency
//! (ADR 0001).
//!
//! Only the genome adapter (DSL program, JSON) is implemented here. The config-and-prompt
//! and Rust adapters are future work (ADR 0004) and are deliberately absent, not stubbed.
#![forbid(unsafe_code)]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::learner::dsl::reference_step_cost;
use crate::learner::mutate::mutate;
use crate::learner::seeds::seed_programs;
use crate::learner::{
    COST_CAP_RATIO, DslLimits, LearnerGenome, Mutation, PROTOCOL_VERSION, SUPPORTED_F,
    VariationMode,
};
use crate::util::{BrutError, Result, derive_seed};

/// The only directory a mutation pull request may change. The scope fence (ADR 0004)
/// rejects any changed path outside it, so a mutant can never touch the evaluator, the
/// cost model, the hidden grammar, the acceptance harness, or a secret.
pub const CANDIDATE_DIR: &str = "candidates/";

/// True when `path` (a repo-relative, forward-slash path) may be changed by a mutation
/// pull request: it must live strictly under `candidates/`. A leading `./` is ignored and
/// any parent escape (`..`) is rejected.
pub fn path_in_scope(path: &str) -> bool {
    let p = path.strip_prefix("./").unwrap_or(path);

    if p.split('/').any(|seg| seg == "..") {
        return false;
    }

    p.starts_with(CANDIDATE_DIR) && p.len() > CANDIDATE_DIR.len()
}

/// Adapter for the genome representation: a DSL program stored as JSON at
/// `candidates/<content-id>.json`.
#[derive(Default)]
pub struct GenomeAdapter {
    pub limits: DslLimits,
}

impl GenomeAdapter {
    pub fn new() -> GenomeAdapter {
        GenomeAdapter::default()
    }

    pub fn kind(&self) -> &'static str {
        "genome"
    }

    /// Parse candidate JSON into a genome; unknown fields are rejected by the struct.
    pub fn parse(&self, text: &str) -> Result<LearnerGenome> {
        serde_json::from_str(text).map_err(|e| BrutError::Invalid(format!("genome json: {e}")))
    }

    /// Structural plus cost-cap validity.
    pub fn validate(&self, g: &LearnerGenome) -> Result<()> {
        g.check(&self.limits)
    }

    /// Content fingerprint and canonical id of a candidate (SHA-256 of canonical JSON).
    pub fn fingerprint(&self, g: &LearnerGenome) -> String {
        g.id()
    }

    /// Pretty JSON for a candidate file.
    pub fn to_json(&self, g: &LearnerGenome) -> Result<String> {
        serde_json::to_string_pretty(g).map_err(|e| BrutError::Invalid(format!("encode: {e}")))
    }

    /// A short, review-friendly summary: id, part lengths, step cost per width, listing.
    pub fn summary(&self, g: &LearnerGenome) -> String {
        let costs: Vec<String> = SUPPORTED_F
            .iter()
            .map(|&f| format!("F{f}={}", g.step_cost(f)))
            .collect();

        format!(
            "id {}  setup {} predict {} learn {}  step_cost[{}]\n{}",
            g.id(),
            g.setup.len(),
            g.predict.len(),
            g.learn.len(),
            costs.join(" "),
            g.render()
        )
    }
}

/// step_cost and its cap at one width.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CostAtWidth {
    pub f: usize,
    pub step_cost: u64,
    pub cap: u64,
}

/// One mutated candidate's provenance, recorded in the pull request so the mutation
/// replays bit for bit (ADR 0002, ADR 0004).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateProvenance {
    pub adapter: String,
    pub parent_id: String,
    pub child_id: String,
    pub operator: Mutation,
    pub seed: u64,
    pub step_cost: Vec<CostAtWidth>,
}

/// The manifest of one proposal: exactly two mutated candidates (ADR 0004). A binary
/// writes it next to the candidate files for the drafting script to put in the pull
/// request body; it is not itself committed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProposalManifest {
    pub tool: String,
    pub protocol: String,
    pub adapter: String,
    pub mode: VariationMode,
    pub master_seed: u64,
    pub candidates: Vec<CandidateProvenance>,
}

/// A proposed child plus its provenance, kept together so a binary can write both the
/// candidate file and the manifest from one result.
pub struct Proposed {
    pub genome: LearnerGenome,
    pub provenance: CandidateProvenance,
}

fn cost_rows(g: &LearnerGenome) -> Vec<CostAtWidth> {
    SUPPORTED_F
        .iter()
        .map(|&f| CostAtWidth {
            f,
            step_cost: g.step_cost(f),
            cap: COST_CAP_RATIO.saturating_mul(reference_step_cost(f)),
        })
        .collect()
}

/// Mutate one parent into a valid, recorded child under a derived seed. The mutator only
/// returns valid children; we re-check and error if that ever fails, and we reject an
/// unchanged result, so a binary never writes an invalid or no-op candidate.
pub fn propose_one(
    adapter: &GenomeAdapter,
    parent: &LearnerGenome,
    mode: VariationMode,
    seed: u64,
) -> Result<Proposed> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let (child, operator) = mutate(parent, mode, &mut rng, &adapter.limits);

    adapter.validate(&child)?;

    if &child == parent {
        return Err(BrutError::Invalid(
            "mutation produced no change; try another seed or parent".to_string(),
        ));
    }

    let provenance = CandidateProvenance {
        adapter: adapter.kind().to_string(),
        parent_id: adapter.fingerprint(parent),
        child_id: adapter.fingerprint(&child),
        operator,
        seed,
        step_cost: cost_rows(&child),
    };

    Ok(Proposed {
        genome: child,
        provenance,
    })
}

/// Choose two parents from the pool deterministically from the master seed; fall back to
/// the seed programs when the pool is empty (as the random mutator falls back to the
/// reference NN). With a single-element pool, both parents are that element.
fn pick_parents(pool: &[LearnerGenome], master_seed: u64) -> Vec<LearnerGenome> {
    let source: Vec<LearnerGenome> = if pool.is_empty() {
        seed_programs().into_iter().map(|(_, g)| g).collect()
    } else {
        pool.to_vec()
    };

    let n = source.len();

    if n == 1 {
        return vec![source[0].clone(), source[0].clone()];
    }

    let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(master_seed, "parents"));
    let i = rng.random_range(0..n);
    let mut j = rng.random_range(0..n - 1);

    if j >= i {
        j += 1;
    }

    vec![source[i].clone(), source[j].clone()]
}

/// Build a two-candidate proposal from a parent pool (or the seed programs when the pool
/// is empty), under one master seed. Each candidate gets an independent derived seed, so
/// the pair is reproducible and the two are drawn independently.
pub fn propose_pair(
    adapter: &GenomeAdapter,
    pool: &[LearnerGenome],
    mode: VariationMode,
    master_seed: u64,
) -> Result<(Vec<Proposed>, ProposalManifest)> {
    let parents = pick_parents(pool, master_seed);
    let mut proposed = Vec::with_capacity(parents.len());

    for (k, parent) in parents.iter().enumerate() {
        let seed = derive_seed(master_seed, &format!("cand/{k}"));

        proposed.push(propose_one(adapter, parent, mode, seed)?);
    }

    let manifest = ProposalManifest {
        tool: "brut-propose".to_string(),
        protocol: PROTOCOL_VERSION.to_string(),
        adapter: adapter.kind().to_string(),
        mode,
        master_seed,
        candidates: proposed.iter().map(|p| p.provenance.clone()).collect(),
    };

    Ok((proposed, manifest))
}

/// Outcome of validating one candidate file during ingest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IngestRecord {
    pub path: String,
    pub accepted: bool,
    pub id: Option<String>,
    pub reason: String,
}

/// Validate one candidate's JSON against the adapter and the id taken from its file name.
/// Accepts only when it parses, passes `check`, and its content id equals `expected_stem`
/// (the file name without `.json`), so a candidate cannot be renamed to hide its content.
pub fn ingest_one(
    adapter: &GenomeAdapter,
    path: &str,
    text: &str,
    expected_stem: &str,
) -> IngestRecord {
    let reject = |id: Option<String>, reason: String| IngestRecord {
        path: path.to_string(),
        accepted: false,
        id,
        reason,
    };

    let g = match adapter.parse(text) {
        Ok(g) => g,
        Err(e) => return reject(None, e.to_string()),
    };

    if let Err(e) = adapter.validate(&g) {
        return reject(Some(adapter.fingerprint(&g)), e.to_string());
    }

    let id = adapter.fingerprint(&g);

    if id != expected_stem {
        return reject(
            Some(id.clone()),
            format!("file name {expected_stem} does not match content id {id}"),
        );
    }

    IngestRecord {
        path: path.to_string(),
        accepted: true,
        id: Some(id),
        reason: "ok".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_fence_allows_only_candidate_files() {
        assert!(path_in_scope("candidates/abc123.json"));
        assert!(path_in_scope("./candidates/abc123.json"));
        assert!(!path_in_scope("candidates/")); // the directory itself, no file
        assert!(!path_in_scope("src/main.rs"));
        assert!(!path_in_scope("src/learner/dsl.rs"));
        assert!(!path_in_scope("docs/adr/0004.md"));
        assert!(!path_in_scope(".github/workflows/scope-fence.yml"));
        assert!(!path_in_scope("candidates/../src/main.rs"));
        assert!(!path_in_scope("Cargo.toml"));
    }

    #[test]
    fn propose_pair_is_deterministic_valid_and_distinct() {
        let adapter = GenomeAdapter::new();
        let (a, ma) = propose_pair(&adapter, &[], VariationMode::Full, 42).expect("pair a");
        let (_b, mb) = propose_pair(&adapter, &[], VariationMode::Full, 42).expect("pair b");

        assert_eq!(a.len(), 2);
        assert_eq!(ma, mb, "same seed, same manifest");

        for p in &a {
            assert!(adapter.validate(&p.genome).is_ok());
            assert_ne!(p.provenance.child_id, p.provenance.parent_id);
            assert_eq!(adapter.fingerprint(&p.genome), p.provenance.child_id);
        }

        // Different master seed gives a different proposal.
        let (_c, mc) = propose_pair(&adapter, &[], VariationMode::Full, 43).expect("pair c");
        assert_ne!(ma, mc);
    }

    #[test]
    fn ingest_accepts_matching_and_rejects_tampered() {
        let adapter = GenomeAdapter::new();
        let (proposed, _m) = propose_pair(&adapter, &[], VariationMode::Full, 7).expect("pair");
        let g = &proposed[0].genome;
        let id = adapter.fingerprint(g);
        let json = adapter.to_json(g).expect("json");

        // File named by its content id: accepted.
        let ok = ingest_one(&adapter, &format!("candidates/{id}.json"), &json, &id);
        assert!(ok.accepted, "{}", ok.reason);

        // Wrong file name for the same content: rejected.
        let bad = ingest_one(&adapter, "candidates/deadbeef.json", &json, "deadbeef");
        assert!(!bad.accepted);

        // Not JSON: rejected.
        let junk = ingest_one(&adapter, "candidates/x.json", "not json", "x");
        assert!(!junk.accepted);
    }
}
