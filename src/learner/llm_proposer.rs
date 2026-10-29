//! Program proposer backed by a language-model stub (ADR 0001, 0002).
//!
//! Owner: WP1. Pure: no network and no secret. The LLM is reached only through the
//! `ModelBackend` seam (`backend.rs`); here we drive that seam, parse and repair the
//! answer, and fall back to the random mutator when the model is not useful. Every
//! request and its disposition are recorded as an `LlmAttempt` so the orchestrator can
//! write them to the archive and replay them on a re-run (ADR 0002).
#![forbid(unsafe_code)]

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::backend::{GenRequest, ModelBackend, StopReason};
use crate::learner::dsl::ALL_OPS;
use crate::learner::mutate::{ProgramProposer, RandomProgramMutator};
use crate::learner::types::{DslLimits, LearnerGenome, Mutation, VariationMode};
use crate::util::sha256_hex;

/// Upper bound on generated tokens for one request.
const MAX_TOKENS: u32 = 2048;

/// Sampling temperature. The backend stays reproducible through its seed (ADR 0002),
/// so a non-zero value only widens the model's choices on a real backend.
const TEMPERATURE: f32 = 0.7;

/// What became of one attempt (one model call, or the random-mutator fallback).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptOutcome {
    /// The answer parsed, validated (after an optional repair) and was returned.
    Accepted,
    /// The answer was rejected (unusable, unparseable or invalid) and a retry followed.
    Retry,
    /// The model was exhausted; the random mutator produced the returned genome.
    Fallback,
    /// The model declined to answer (`StopReason::Refusal`).
    Refusal,
}

/// One recorded attempt. Plain data for the archive (ADR 0002); no floats, so every
/// number serialised here is finite.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmAttempt {
    /// Name of the backend that produced the answer, or of the fallback mutator.
    pub backend: String,
    /// SHA-256 (hex) of the request's system and user prompt; empty for the fallback.
    pub prompt_sha256: String,
    /// Completion tokens the backend reported; 0 for the fallback or a backend error.
    pub response_tokens: u32,
    pub outcome: AttemptOutcome,
    /// Id of the genome this attempt yielded; empty when the attempt produced none.
    pub genome_id: String,
}

/// A proposer that asks a `ModelBackend` for a new program and repairs or replaces a
/// bad answer. The `ProgramProposer` impl is infallible and, given a deterministic
/// backend, deterministic: the returned genome always passes `check(lim)`.
pub struct LlmProposer {
    backend: Box<dyn ModelBackend>,
    fallback: RandomProgramMutator,
    name: String,
    log: Mutex<Vec<LlmAttempt>>,
    invalid: AtomicUsize,
}

/// Outcome of parsing and validating one model answer.
enum Parsed {
    /// A valid genome (possibly repaired by dead-code elimination).
    Good(LearnerGenome),
    /// The model declined to answer.
    Refused,
    /// The answer was unusable, unparseable or invalid; the string is the reason,
    /// fed back into the retry prompt.
    Bad(String),
}

impl LlmProposer {
    /// A proposer over `backend`, falling back to a `RandomProgramMutator` in `mode`.
    pub fn new(backend: Box<dyn ModelBackend>, mode: VariationMode) -> LlmProposer {
        let name = format!("llm-proposer-v1/{}", backend.name());

        LlmProposer {
            backend,
            fallback: RandomProgramMutator { mode },
            name,
            log: Mutex::new(Vec::new()),
            invalid: AtomicUsize::new(0),
        }
    }

    /// Take and clear the attempts collected so far, for the orchestrator to archive.
    pub fn drain_attempts(&self) -> Vec<LlmAttempt> {
        std::mem::take(&mut *self.lock_log())
    }

    /// Running count of rejected model answers (retries and refusals, not fallbacks).
    pub fn invalid_count(&self) -> usize {
        self.invalid.load(Ordering::Relaxed)
    }

    /// The log guard, recovered from a poisoned lock so a caller never panics.
    fn lock_log(&self) -> std::sync::MutexGuard<'_, Vec<LlmAttempt>> {
        self.log
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Append one call's records and add its invalid count under a single lock.
    fn commit(&self, records: Vec<LlmAttempt>, invalid: usize) {
        if invalid > 0 {
            self.invalid.fetch_add(invalid, Ordering::Relaxed);
        }

        self.lock_log().extend(records);
    }

    /// The fixed system prompt: a brief description of the DSL and its ops.
    fn system_prompt() -> String {
        let ops: Vec<String> = ALL_OPS.iter().map(|op| format!("{op:?}")).collect();

        let mut s = String::new();

        s.push_str(
            "You design tiny learning algorithms in a closed, interpreted register machine.\n",
        );
        s.push_str("A program is a JSON object with three straight-line instruction lists: ");
        s.push_str(
            "\"setup\" runs once on zeroed memory, \"predict\" runs before every prediction, ",
        );
        s.push_str("and \"learn\" runs after each training prediction with the label available.\n");
        s.push_str(
            "Registers: scalars s0..s9, vectors v0..v7 (each F wide), matrices m0..m3 (F by F). ",
        );
        s.push_str(
            "The input row is in v0, the prediction must be written to v1, and the label is ",
        );
        s.push_str("placed in v2 (and s0) before \"learn\".\n");
        s.push_str("Each instruction is {\"op\":<name>,\"a\":<reg>,\"b\":<reg>,\"out\":<reg>,\"c\":<number>}. ");
        s.push_str("Unused operand fields must be 0 and \"c\" must be 0.0 when the op takes no constant.\n");
        s.push_str("Available ops: ");
        s.push_str(&ops.join(", "));
        s.push_str(".\n");
        s.push_str("Answer with only the JSON program and no other text.\n");

        s
    }

    /// The user prompt: the limits, the parent programs (rendered DSL text) with their
    /// public scores when supplied, and an optional rejection reason for a retry.
    fn user_prompt(
        parents: &[&LearnerGenome],
        scores: &[f32],
        lim: &DslLimits,
        retry: Option<&str>,
    ) -> String {
        let mut p = String::new();

        p.push_str("Design a new learner program.\n\n");
        p.push_str(&format!(
            "Limits: setup <= {}, predict <= {}, learn <= {}, |constant| <= {}.\n\n",
            lim.max_setup, lim.max_predict, lim.max_learn, lim.const_abs_max
        ));

        if parents.is_empty() {
            p.push_str("There are no parent programs; propose a fresh learner.\n\n");
        } else {
            p.push_str("Parent program(s) to improve on:\n\n");

            for (i, parent) in parents.iter().enumerate() {
                p.push_str(&format!("Parent {i}:\n"));
                p.push_str(&parent.render());

                if let Some(score) = scores.get(i).filter(|s| s.is_finite()) {
                    p.push_str(&format!("public score: {score}\n"));
                }

                p.push('\n');
            }
        }

        if let Some(reason) = retry {
            p.push_str(&format!(
                "The previous answer was rejected: {reason}\nReturn a corrected program.\n\n"
            ));
        }

        p.push_str(
            "Reply with only a JSON object with keys \"setup\", \"predict\" and \"learn\", \
             each a list of instructions {\"op\",\"a\",\"b\",\"out\",\"c\"}.\n",
        );

        p
    }

    /// Build one request. `seed` comes from the slot rng, so the request is deterministic.
    fn build_request(
        &self,
        parents: &[&LearnerGenome],
        scores: &[f32],
        lim: &DslLimits,
        retry: Option<&str>,
        seed: u64,
    ) -> GenRequest {
        GenRequest {
            system: Self::system_prompt(),
            prompt: Self::user_prompt(parents, scores, lim, retry),
            max_tokens: MAX_TOKENS,
            temperature: TEMPERATURE,
            seed,
            think_tokens: 0,
        }
    }

    /// Call the backend and classify the answer. Returns the disposition and the
    /// reported completion-token count (0 on a backend error).
    fn run(&self, req: &GenRequest, lim: &DslLimits) -> (Parsed, u32) {
        let resp = match self.backend.generate(req) {
            Ok(resp) => resp,
            Err(_) => return (Parsed::Bad("the backend returned an error".to_string()), 0),
        };

        let tokens = resp.completion_tokens;

        if resp.stop_reason == StopReason::Refusal {
            return (Parsed::Refused, tokens);
        }

        if !resp.is_usable() {
            return (
                Parsed::Bad("the answer was truncated or empty".to_string()),
                tokens,
            );
        }

        match extract_genome(&resp.text) {
            None => (
                Parsed::Bad("the answer was not valid LearnerGenome JSON".to_string()),
                tokens,
            ),
            Some(g) => match validate_or_repair(g, lim) {
                Ok(good) => (Parsed::Good(good), tokens),
                Err(why) => (Parsed::Bad(why), tokens),
            },
        }
    }

    /// Record a failed model attempt and return the reason for the retry prompt.
    fn note_failure(
        &self,
        records: &mut Vec<LlmAttempt>,
        invalid: &mut usize,
        sha: String,
        tokens: u32,
        parsed: Parsed,
    ) -> String {
        *invalid += 1;

        let (outcome, reason) = match parsed {
            Parsed::Refused => (
                AttemptOutcome::Refusal,
                "the previous answer declined to produce a program".to_string(),
            ),
            Parsed::Bad(why) => (AttemptOutcome::Retry, why),
            // A Good answer is handled by the caller and never reaches here.
            Parsed::Good(_) => (AttemptOutcome::Retry, String::new()),
        };

        records.push(LlmAttempt {
            backend: self.backend.name().to_string(),
            prompt_sha256: sha,
            response_tokens: tokens,
            outcome,
            genome_id: String::new(),
        });

        reason
    }
}

impl ProgramProposer for LlmProposer {
    fn name(&self) -> &str {
        &self.name
    }

    /// One proposal: ask the model, repair or retry once, then fall back to the random
    /// mutator. The returned genome always passes `check(lim)`. Public scores are not
    /// threaded through this trait yet, so none are sent here.
    fn propose(
        &self,
        parents: &[&LearnerGenome],
        rng: &mut ChaCha8Rng,
        lim: &DslLimits,
    ) -> (LearnerGenome, Mutation) {
        let mut records: Vec<LlmAttempt> = Vec::new();
        let mut invalid = 0usize;
        let scores: &[f32] = &[];

        // First attempt.
        let req = self.build_request(parents, scores, lim, None, rng.random());
        let sha = request_sha(&req);
        let (parsed, tokens) = self.run(&req, lim);

        if let Parsed::Good(g) = parsed {
            let id = g.id();

            records.push(accepted(self.backend.name(), sha, tokens, id));
            self.commit(records, invalid);

            return (g, Mutation::Randomize);
        }

        let reason = self.note_failure(&mut records, &mut invalid, sha, tokens, parsed);

        // One retry, told why the first answer was rejected.
        let req = self.build_request(parents, scores, lim, Some(&reason), rng.random());
        let sha = request_sha(&req);
        let (parsed, tokens) = self.run(&req, lim);

        if let Parsed::Good(g) = parsed {
            let id = g.id();

            records.push(accepted(self.backend.name(), sha, tokens, id));
            self.commit(records, invalid);

            return (g, Mutation::Randomize);
        }

        self.note_failure(&mut records, &mut invalid, sha, tokens, parsed);

        // Fallback: the random mutator on the same parents (always a valid, different child).
        let (child, m) = self.fallback.propose(parents, rng, lim);

        records.push(LlmAttempt {
            backend: self.fallback.name().to_string(),
            prompt_sha256: String::new(),
            response_tokens: 0,
            outcome: AttemptOutcome::Fallback,
            genome_id: child.id(),
        });
        self.commit(records, invalid);

        (child, m)
    }
}

/// An accepted-attempt record.
fn accepted(backend: &str, sha: String, tokens: u32, id: String) -> LlmAttempt {
    LlmAttempt {
        backend: backend.to_string(),
        prompt_sha256: sha,
        response_tokens: tokens,
        outcome: AttemptOutcome::Accepted,
        genome_id: id,
    }
}

/// SHA-256 (hex) of a request's system and user prompt. The `\u{1}` separator matches
/// the stub backend, so the same text always hashes the same way.
fn request_sha(req: &GenRequest) -> String {
    let mut material = String::new();

    material.push_str(&req.system);
    material.push('\u{1}');
    material.push_str(&req.prompt);

    sha256_hex(material.as_bytes())
}

/// Validate `g`; if it fails, try dead-code elimination once as a repair. Err carries
/// the original validation message for the retry prompt.
fn validate_or_repair(g: LearnerGenome, lim: &DslLimits) -> Result<LearnerGenome, String> {
    match g.check(lim) {
        Ok(()) => Ok(g),
        Err(e) => {
            let repaired = g.simplify();

            if repaired.check(lim).is_ok() {
                Ok(repaired)
            } else {
                Err(e.to_string())
            }
        }
    }
}

/// Parse `text` as a `LearnerGenome`: first directly, then each balanced `{...}` object
/// in turn, returning the first that parses, so prose, code fences or an earlier
/// JSON-like object around the program do not defeat it.
fn extract_genome(text: &str) -> Option<LearnerGenome> {
    if let Ok(g) = serde_json::from_str::<LearnerGenome>(text.trim()) {
        return Some(g);
    }

    let bytes = text.as_bytes();
    let mut pos = 0;

    while let Some(rel) = text[pos..].find('{') {
        let start = pos + rel;

        match balanced_end(bytes, start) {
            Some(end) => {
                if let Ok(g) = serde_json::from_str::<LearnerGenome>(&text[start..=end]) {
                    return Some(g);
                }

                pos = end + 1;
            }
            None => break,
        }
    }

    None
}

/// Index of the `}` that closes the `{` at `start`, respecting nesting and strings, or
/// None if it is unbalanced. Only ASCII markers matter, and ASCII bytes never occur
/// inside a UTF-8 multibyte sequence, so the returned index is always a char boundary.
fn balanced_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }

            continue;
        }

        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;

                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::backend::{GenResponse, StubModel, approx_tokens};
    use crate::learner::types::{Instr, OpCode};
    use crate::util::Result;

    /// A backend that always returns one fixed text with a clean stop.
    struct FixedModel {
        name: String,
        text: String,
    }

    impl FixedModel {
        fn new(name: &str, text: &str) -> FixedModel {
            FixedModel {
                name: name.to_string(),
                text: text.to_string(),
            }
        }
    }

    impl ModelBackend for FixedModel {
        fn name(&self) -> &str {
            &self.name
        }

        fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
            Ok(GenResponse {
                prompt_tokens: approx_tokens(&req.system) + approx_tokens(&req.prompt),
                completion_tokens: approx_tokens(&self.text),
                text: self.text.clone(),
                stop_reason: StopReason::Stop,
                backend: self.name.clone(),
            })
        }
    }

    /// A small, definitely valid genome, and the JSON a good model would return.
    fn valid_genome() -> LearnerGenome {
        let g = LearnerGenome {
            predict: vec![Instr {
                op: OpCode::MatVec,
                a: 0,
                b: 0,
                out: 1,
                c: 0.0,
            }],
            ..LearnerGenome::default()
        };

        assert!(g.check(&DslLimits::default()).is_ok());

        g
    }

    fn parent() -> LearnerGenome {
        LearnerGenome {
            predict: vec![Instr {
                op: OpCode::MatVec,
                a: 0,
                b: 0,
                out: 1,
                c: 0.0,
            }],
            learn: vec![Instr {
                op: OpCode::VSub,
                a: 2,
                b: 1,
                out: 3,
                c: 0.0,
            }],
            ..LearnerGenome::default()
        }
    }

    /// (a) A known valid answer is accepted and logged as accepted.
    #[test]
    fn valid_answer_is_accepted() {
        let lim = DslLimits::default();
        let want = valid_genome();
        let backend = FixedModel::new("fixed:good", &want.canonical_json());
        let proposer = LlmProposer::new(Box::new(backend), VariationMode::Full);
        let p = parent();

        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (child, mutation) = proposer.propose(&[&p], &mut rng, &lim);

        assert_eq!(child, want);
        assert_eq!(mutation, Mutation::Randomize);
        assert!(child.check(&lim).is_ok());
        assert_eq!(proposer.invalid_count(), 0);

        let log = proposer.drain_attempts();

        assert_eq!(log.len(), 1);
        assert_eq!(log[0].outcome, AttemptOutcome::Accepted);
        assert_eq!(log[0].genome_id, want.id());
        assert_eq!(log[0].backend, "fixed:good");
        assert!(!log[0].prompt_sha256.is_empty());

        // Draining leaves the log empty.
        assert!(proposer.drain_attempts().is_empty());
    }

    /// (b) The non-program stub retries once, then falls back to the random mutator.
    #[test]
    fn stub_falls_back_after_one_retry() {
        let lim = DslLimits::default();
        let proposer = LlmProposer::new(Box::new(StubModel::new("stub:test")), VariationMode::Full);
        let p = parent();

        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let (child, _) = proposer.propose(&[&p], &mut rng, &lim);

        // The fallback child is valid and differs from the parent.
        assert!(child.check(&lim).is_ok());
        assert_ne!(child, p);

        let log = proposer.drain_attempts();

        assert_eq!(log.len(), 3, "{log:?}");
        assert_eq!(log[0].outcome, AttemptOutcome::Retry);
        assert_eq!(log[1].outcome, AttemptOutcome::Retry);
        assert_eq!(log[2].outcome, AttemptOutcome::Fallback);
        assert_eq!(log[2].backend, proposer.fallback.name());
        assert_eq!(log[2].genome_id, child.id());
        assert!(log[2].prompt_sha256.is_empty());

        // Both retries are the two model answers rejected before the fallback.
        assert_eq!(proposer.invalid_count(), 2);

        // The two prompts differ: the retry carries the rejection reason.
        assert_ne!(log[0].prompt_sha256, log[1].prompt_sha256);
    }

    /// A refusal is recorded as such before the retry.
    #[test]
    fn refusal_is_logged() {
        struct Refuser;

        impl ModelBackend for Refuser {
            fn name(&self) -> &str {
                "refuser"
            }

            fn generate(&self, _req: &GenRequest) -> Result<GenResponse> {
                Ok(GenResponse {
                    text: "I will not".to_string(),
                    prompt_tokens: 1,
                    completion_tokens: 3,
                    stop_reason: StopReason::Refusal,
                    backend: "refuser".to_string(),
                })
            }
        }

        let lim = DslLimits::default();
        let proposer = LlmProposer::new(Box::new(Refuser), VariationMode::Full);
        let p = parent();

        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let (child, _) = proposer.propose(&[&p], &mut rng, &lim);

        assert!(child.check(&lim).is_ok());

        let log = proposer.drain_attempts();

        assert_eq!(log.len(), 3);
        assert_eq!(log[0].outcome, AttemptOutcome::Refusal);
        assert_eq!(log[1].outcome, AttemptOutcome::Refusal);
        assert_eq!(log[2].outcome, AttemptOutcome::Fallback);
        assert_eq!(proposer.invalid_count(), 2);
    }

    /// (c) The same parents and the same backend give the same proposal and log.
    #[test]
    fn proposal_is_deterministic() {
        let lim = DslLimits::default();
        let p = parent();

        let run = || {
            let proposer =
                LlmProposer::new(Box::new(StubModel::new("stub:test")), VariationMode::Full);
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let out = proposer.propose(&[&p], &mut rng, &lim);
            let log = proposer.drain_attempts();

            (out, log, proposer.invalid_count())
        };

        let (out1, log1, inv1) = run();
        let (out2, log2, inv2) = run();

        assert_eq!(out1, out2);
        assert_eq!(log1, log2);
        assert_eq!(inv1, inv2);

        // And the accepted path is deterministic too.
        let valid = valid_genome();
        let det = || {
            let backend = FixedModel::new("fixed:good", &valid.canonical_json());
            let proposer = LlmProposer::new(Box::new(backend), VariationMode::Full);
            let mut rng = ChaCha8Rng::seed_from_u64(9);
            let out = proposer.propose(&[&p], &mut rng, &lim);

            (out, proposer.drain_attempts())
        };

        assert_eq!(det(), det());
    }

    /// (d) The drained log is non-empty and its genome ids match the returned genomes.
    #[test]
    fn drained_ids_match_returned_genomes() {
        let lim = DslLimits::default();
        let valid = valid_genome();
        let backend = FixedModel::new("fixed:good", &valid.canonical_json());
        let proposer = LlmProposer::new(Box::new(backend), VariationMode::Full);
        let p = parent();

        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut returned = Vec::new();

        for _ in 0..3 {
            let (child, _) = proposer.propose(&[&p], &mut rng, &lim);

            returned.push(child.id());
        }

        let log = proposer.drain_attempts();

        assert!(!log.is_empty());

        let ids: Vec<&String> = log
            .iter()
            .filter(|a| !a.genome_id.is_empty())
            .map(|a| &a.genome_id)
            .collect();

        assert_eq!(ids, returned.iter().collect::<Vec<_>>());

        // The fallback path also lands its returned id in the log.
        let stub = LlmProposer::new(Box::new(StubModel::new("stub:test")), VariationMode::Full);
        let (child, _) = stub.propose(&[&p], &mut rng, &lim);
        let log = stub.drain_attempts();
        let last = log.last().expect("one fallback record");

        assert_eq!(last.outcome, AttemptOutcome::Fallback);
        assert_eq!(last.genome_id, child.id());
    }

    /// JSON embedded in prose or a code fence is still parsed.
    #[test]
    fn extract_handles_wrapped_json() {
        let valid = valid_genome();
        let json = valid.canonical_json();
        let wrapped = format!("Sure, here it is:\n```json\n{json}\n```\nHope that helps.");

        assert_eq!(extract_genome(&wrapped), Some(valid.clone()));

        // A brace inside a string does not end the object early.
        let tricky = format!("noise {{\"note\":\"}}\"}} then {json} end");

        assert_eq!(extract_genome(&tricky), Some(valid));

        assert_eq!(extract_genome("no json here"), None);
    }

    /// An over-long but otherwise sound program is repaired by dead-code elimination.
    #[test]
    fn repair_shrinks_an_over_long_program() {
        let lim = DslLimits::default();

        // A dead scalar write repeated past the predict limit, plus the real prediction.
        let dead = Instr {
            op: OpCode::SConst,
            a: 0,
            b: 0,
            out: 5,
            c: 0.5,
        };
        let live = Instr {
            op: OpCode::MatVec,
            a: 0,
            b: 0,
            out: 1,
            c: 0.0,
        };
        let mut predict = vec![dead; lim.max_predict + 4];

        predict.push(live);

        let g = LearnerGenome {
            predict,
            ..LearnerGenome::default()
        };

        assert!(g.check(&lim).is_err(), "should be over the predict limit");

        let repaired = validate_or_repair(g, &lim).expect("repairable");

        assert!(repaired.check(&lim).is_ok());
        assert_eq!(repaired.predict, vec![live]);
    }
}
