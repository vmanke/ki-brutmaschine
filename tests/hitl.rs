//! Integration tests for the human-in-the-loop PR mutation path (ADR 0004): the genome
//! adapter round-trips a real candidate, proposals are valid and reproducible, and the
//! scope fence and ingest gate behave on the public API.
#![forbid(unsafe_code)]

use std::collections::HashSet;

use brutmaschine::hitl::{GenomeAdapter, ingest_one, path_in_scope, propose_pair, seed_ids};
use brutmaschine::learner::VariationMode;

/// The adapter parses, validates and fingerprints a real stored candidate, and the id is
/// stable across a JSON round-trip.
#[test]
fn adapter_round_trips_a_real_candidate() {
    let text = std::fs::read_to_string("examples/data/oracle_genome.json")
        .expect("read oracle genome fixture");
    let adapter = GenomeAdapter::new();

    let g = adapter.parse(&text).expect("parse");
    adapter.validate(&g).expect("valid");

    let id = adapter.fingerprint(&g);
    let reencoded = adapter.to_json(&g).expect("encode");
    let g2 = adapter.parse(&reencoded).expect("reparse");

    assert_eq!(id, adapter.fingerprint(&g2), "id stable across round-trip");
}

/// A proposal yields two valid, distinct candidates, and the same seed reproduces it; an
/// ingest of the written files accepts them when named by their content id.
#[test]
fn proposal_is_valid_reproducible_and_ingests() {
    let adapter = GenomeAdapter::new();

    let (first, m1) = propose_pair(&adapter, &[], VariationMode::Full, 2024).expect("propose");
    let (_second, m2) = propose_pair(&adapter, &[], VariationMode::Full, 2024).expect("propose");

    assert_eq!(m1, m2, "same seed reproduces the proposal");
    assert_eq!(first.len(), 2);

    for p in &first {
        assert!(adapter.validate(&p.genome).is_ok());

        let id = adapter.fingerprint(&p.genome);
        let json = adapter.to_json(&p.genome).expect("encode");
        let rec = ingest_one(
            &adapter,
            &format!("candidates/{id}.json"),
            &json,
            &id,
            &HashSet::new(),
        );

        assert!(
            rec.accepted,
            "ingest rejected a freshly proposed candidate: {}",
            rec.reason
        );

        // The same candidate is rejected once it is already known (dedupe against seeds
        // or an existing pool).
        let known: HashSet<String> = [id.clone()].into_iter().collect();
        let dup = ingest_one(
            &adapter,
            &format!("candidates/{id}.json"),
            &json,
            &id,
            &known,
        );
        assert!(!dup.accepted, "duplicate candidate should be rejected");
    }

    // A proposed child is never one of the seed programs (a mutation changes the genome).
    let seeds = seed_ids();
    for p in &first {
        assert!(!seeds.contains(&p.provenance.child_id));
    }
}

/// The scope fence passes only candidate files and rejects everything else, including a
/// parent-escape path.
#[test]
fn scope_fence_rejects_non_candidate_paths() {
    assert!(path_in_scope("candidates/0123abcd4567ef89.json"));
    assert!(!path_in_scope("src/learner/dsl.rs"));
    assert!(!path_in_scope("src/hidden_eval.rs"));
    assert!(!path_in_scope(".env"));
    assert!(!path_in_scope("candidates/../.env"));
}
