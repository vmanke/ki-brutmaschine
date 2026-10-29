//! Human-in-the-loop pull-request mutation of candidates (ADR 0004).
//!
//! The AI drafts mutations as pull requests; a human reviews and merges before a
//! mutated candidate enters a run. This module holds the genome adapter and the pure
//! helpers both binaries use (`brut-propose`, `brut-ingest`). It is pure: no I/O, no
//! network, no git. The binaries do the file and stdin I/O; the git and pull-request
//! steps live in `scripts/`, so the core crate keeps no process or network dependency
//! (ADR 0001).
//!
//! Two adapters are implemented: the genome adapter (DSL program, JSON) and the
//! config-and-prompt adapter for Cascades (ADR 0005), both behind `CandidateAdapter`. The
//! Rust adapter is future work (ADR 0004) and is deliberately absent, not stubbed.
#![forbid(unsafe_code)]

use std::collections::HashSet;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::cascade::genome::{Cascade, CascadeLimits, pool_limits, seed_cascades};
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

/// Where promoted Cascades live inside the candidate pool (ADR 0005). It is under
/// `candidates/`, so the same scope fence covers it.
pub const CASCADE_DIR: &str = "candidates/cascades/";

/// The one interface every candidate representation implements (ADR 0004): parse,
/// statically validate, and fingerprint by content.
pub trait CandidateAdapter {
    type Candidate;

    /// Short name recorded in provenance.
    fn kind(&self) -> &'static str;

    /// Parse candidate JSON; unknown fields are rejected.
    fn parse(&self, text: &str) -> Result<Self::Candidate>;

    /// Static validity: structure, limits, and (for prompts) no credential look-alikes.
    fn validate(&self, c: &Self::Candidate) -> Result<()>;

    /// Content id, which is also the candidate's file name.
    fn fingerprint(&self, c: &Self::Candidate) -> String;
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

impl CandidateAdapter for GenomeAdapter {
    type Candidate = LearnerGenome;

    fn kind(&self) -> &'static str {
        GenomeAdapter::kind(self)
    }

    fn parse(&self, text: &str) -> Result<LearnerGenome> {
        GenomeAdapter::parse(self, text)
    }

    fn validate(&self, g: &LearnerGenome) -> Result<()> {
        GenomeAdapter::validate(self, g)
    }

    fn fingerprint(&self, g: &LearnerGenome) -> String {
        GenomeAdapter::fingerprint(self, g)
    }
}

/// Adapter for the config-and-prompt representation (ADR 0004, second adapter): a
/// Cascade stored as JSON at `candidates/cascades/<content-id>.json`. Validity is the
/// Cascade check against the pool allow-list, which also rejects credential look-alikes
/// in prompt text. No model runs during review.
pub struct CascadeAdapter {
    pub limits: CascadeLimits,
}

impl Default for CascadeAdapter {
    fn default() -> Self {
        CascadeAdapter {
            limits: pool_limits(),
        }
    }
}

impl CascadeAdapter {
    pub fn new() -> CascadeAdapter {
        CascadeAdapter::default()
    }

    /// Pretty JSON for a candidate file.
    pub fn to_json(&self, c: &Cascade) -> Result<String> {
        serde_json::to_string_pretty(c).map_err(|e| BrutError::Invalid(format!("encode: {e}")))
    }
}

impl CandidateAdapter for CascadeAdapter {
    type Candidate = Cascade;

    fn kind(&self) -> &'static str {
        "cascade"
    }

    fn parse(&self, text: &str) -> Result<Cascade> {
        serde_json::from_str(text).map_err(|e| BrutError::Invalid(format!("cascade json: {e}")))
    }

    fn validate(&self, c: &Cascade) -> Result<()> {
        c.check(&self.limits)
    }

    fn fingerprint(&self, c: &Cascade) -> String {
        c.id()
    }
}

/// Content ids of the seed Cascades; a promoted Cascade equal to one adds nothing.
pub fn cascade_seed_ids() -> HashSet<String> {
    seed_cascades().into_iter().map(|(_, c)| c.id()).collect()
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

/// Bounded retries so a pair never collides on the same content id. Independent seeds can
/// still produce the same child (e.g. two `Remove`s on a one-instruction parent), which
/// would write the content-addressed file twice and leave one file for a two-entry
/// manifest.
const PROPOSE_ATTEMPTS: usize = 16;

/// Build a two-candidate proposal from a parent pool (or the seed programs when the pool
/// is empty), under one master seed. Each candidate takes its own derived seed; if a draw
/// repeats an earlier child's content id, or is a no-op, the next derived seed is tried.
/// The result is always two distinct, reproducible candidates, and the seed actually used
/// is recorded in each candidate's provenance.
pub fn propose_pair(
    adapter: &GenomeAdapter,
    pool: &[LearnerGenome],
    mode: VariationMode,
    master_seed: u64,
) -> Result<(Vec<Proposed>, ProposalManifest)> {
    let parents = pick_parents(pool, master_seed);
    let mut proposed: Vec<Proposed> = Vec::with_capacity(parents.len());

    for (k, parent) in parents.iter().enumerate() {
        let mut chosen = None;

        for attempt in 0..PROPOSE_ATTEMPTS {
            let seed = derive_seed(master_seed, &format!("cand/{k}/{attempt}"));

            let cand = match propose_one(adapter, parent, mode, seed) {
                Ok(c) => c,
                Err(_) => continue, // no-op or invalid draw: try the next seed
            };

            let clash = proposed
                .iter()
                .any(|p| p.provenance.child_id == cand.provenance.child_id);

            if !clash {
                chosen = Some(cand);
                break;
            }
        }

        match chosen {
            Some(c) => proposed.push(c),
            None => {
                return Err(BrutError::Invalid(format!(
                    "could not draw a distinct candidate {k} after {PROPOSE_ATTEMPTS} seeds; try another master seed"
                )));
            }
        }
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

/// Content ids of the seed programs. A candidate whose content id equals a seed's is a
/// trivial re-submission of a starting program and adds no search progress, so ingest
/// rejects it (fix (e) of the generalization note: dedupe against the evolution seeds).
pub fn seed_ids() -> HashSet<String> {
    seed_programs().into_iter().map(|(_, g)| g.id()).collect()
}

/// Validate one candidate's JSON against the adapter and the id taken from its file name.
/// Accepts only when it parses, passes `check`, its content id equals `expected_stem` (the
/// file name without `.json`, so a candidate cannot be renamed to hide its content), and
/// its content id is not already in `known` — the seed ids plus any existing pool, so a
/// run is never fed a duplicate of a seed or of a candidate it already holds.
pub fn ingest_one<A: CandidateAdapter>(
    adapter: &A,
    path: &str,
    text: &str,
    expected_stem: &str,
    known: &HashSet<String>,
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

    if known.contains(&id) {
        return reject(
            Some(id.clone()),
            format!("duplicate: content id {id} is already a seed or an existing candidate"),
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

        // The two candidates must never share a content id (else one file is written
        // twice for a two-entry manifest).
        assert_ne!(a[0].provenance.child_id, a[1].provenance.child_id);

        // Different master seed gives a different proposal.
        let (_c, mc) = propose_pair(&adapter, &[], VariationMode::Full, 43).expect("pair c");
        assert_ne!(ma, mc);
    }

    #[test]
    fn ingest_accepts_matching_rejects_tampered_and_dedupes() {
        let adapter = GenomeAdapter::new();
        let (proposed, _m) = propose_pair(&adapter, &[], VariationMode::Full, 7).expect("pair");
        let g = &proposed[0].genome;
        let id = adapter.fingerprint(g);
        let json = adapter.to_json(g).expect("json");
        let none = HashSet::new();

        // File named by its content id, not previously known: accepted.
        let ok = ingest_one(
            &adapter,
            &format!("candidates/{id}.json"),
            &json,
            &id,
            &none,
        );
        assert!(ok.accepted, "{}", ok.reason);

        // Wrong file name for the same content: rejected.
        let bad = ingest_one(
            &adapter,
            "candidates/deadbeef.json",
            &json,
            "deadbeef",
            &none,
        );
        assert!(!bad.accepted);

        // Not JSON: rejected.
        let junk = ingest_one(&adapter, "candidates/x.json", "not json", "x", &none);
        assert!(!junk.accepted);

        // Already known (a seed or an existing candidate with this id): rejected as a duplicate.
        let known: HashSet<String> = [id.clone()].into_iter().collect();
        let dup = ingest_one(
            &adapter,
            &format!("candidates/{id}.json"),
            &json,
            &id,
            &known,
        );
        assert!(!dup.accepted);
        assert!(dup.reason.contains("duplicate"), "{}", dup.reason);
    }

    #[test]
    fn cascade_adapter_ingests_new_and_rejects_seeds_and_secrets() {
        let adapter = CascadeAdapter::new();
        let mut c = seed_cascades()[0].1.clone();

        c.stages[0].workers[0].traits.think_tokens = 1024;
        c.stages[0].workers[0].model = "gpt-oss:20b".to_string();

        let id = adapter.fingerprint(&c);
        let json = adapter.to_json(&c).expect("json");
        let path = format!("{CASCADE_DIR}{id}.json");

        assert!(path_in_scope(&path));

        let ok = ingest_one(&adapter, &path, &json, &id, &cascade_seed_ids());

        assert!(ok.accepted, "{}", ok.reason);

        let seed = &seed_cascades()[1].1;
        let seed_id = adapter.fingerprint(seed);
        let dup = ingest_one(
            &adapter,
            "candidates/cascades/s.json",
            &adapter.to_json(seed).expect("json"),
            &seed_id,
            &cascade_seed_ids(),
        );

        assert!(!dup.accepted);
        assert!(dup.reason.contains("duplicate"), "{}", dup.reason);

        let mut leaky = c.clone();

        leaky.stages[0].workers[0].sections[0].text = "token ghp_abcdef".to_string();

        let leaky_id = adapter.fingerprint(&leaky);
        let rec = ingest_one(
            &adapter,
            "candidates/cascades/x.json",
            &adapter.to_json(&leaky).expect("json"),
            &leaky_id,
            &cascade_seed_ids(),
        );

        assert!(!rec.accepted);
        assert!(rec.reason.contains("credential"), "{}", rec.reason);
    }

    #[test]
    fn ingest_rejects_a_seed_resubmission() {
        let adapter = GenomeAdapter::new();
        let (_name, seed) = seed_programs().into_iter().next().expect("a seed program");
        let id = adapter.fingerprint(&seed);
        let json = adapter.to_json(&seed).expect("json");

        let rec = ingest_one(
            &adapter,
            &format!("candidates/{id}.json"),
            &json,
            &id,
            &seed_ids(),
        );

        assert!(!rec.accepted);
        assert!(rec.reason.contains("duplicate"), "{}", rec.reason);
    }
}
