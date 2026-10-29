//! Model backend: the seam between the search and a large language model.
//!
//! The core crate never talks to a network. This module defines the interface
//! (`ModelBackend`) and a deterministic `StubModel` that stands in for a real
//! model, so the LLM-driven search can be developed and tested here with no
//! model and no network. The real backend talks to a local Ollama server and
//! lives in a separate crate that runs only on a host with Docker and Ollama;
//! its contract is `docs/STAGE2_BACKEND_CONTRACT.md`.
#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

use crate::util::{Result, sha256_hex};

/// A request to the model: a system prompt, a user prompt and generation limits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenRequest {
    pub system: String,
    pub prompt: String,
    /// Upper bound on generated tokens. A backend stops at this many.
    pub max_tokens: u32,
    /// 0.0 means greedy and as deterministic as the backend allows; higher is more random.
    pub temperature: f32,
    /// Seed for reproducibility, where the backend supports it.
    pub seed: u64,
    /// Budget for hidden reasoning before the answer, in tokens; 0 means answer at once.
    /// A Worker Trait (ADR 0005). A backend whose model cannot reason maps it to nothing;
    /// the Ollama adapter maps it to the model's `think` setting. Defaults to 0 so
    /// requests recorded before this field existed still replay.
    #[serde(default)]
    pub think_tokens: u32,
}

/// Why generation stopped. Mirrors the fields the orchestrator must check before
/// it uses an answer (a refusal or a length cut-off is not a usable program).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The model ended on its own.
    Stop,
    /// The model hit `max_tokens`.
    Length,
    /// The model declined to answer.
    Refusal,
    /// The backend reported an error for this request.
    Error,
}

/// The model's answer and its token counts. The counts feed the usage records
/// in the lineage archive and the run's compute accounting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenResponse {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub stop_reason: StopReason,
    /// Backend name and model tag, recorded so a run says which model produced a proposal.
    pub backend: String,
}

impl GenResponse {
    /// True when the answer is safe to parse as a proposal: it ended cleanly and
    /// carries some text. A refusal, an error or an empty answer is not usable.
    pub fn is_usable(&self) -> bool {
        self.stop_reason == StopReason::Stop && !self.text.is_empty()
    }
}

/// A source of model completions. Implementors must be deterministic for equal
/// requests where they can be, and must never panic on a valid request.
pub trait ModelBackend: Send + Sync {
    /// Short name of the backend and its model, for the archive.
    fn name(&self) -> &str;

    /// Produce one completion for `req`.
    fn generate(&self, req: &GenRequest) -> Result<GenResponse>;
}

/// A rough token count: one token per whitespace-separated word, at least one for
/// non-empty text. The real backend reports exact counts; this is only for the stub.
pub fn approx_tokens(text: &str) -> u32 {
    let words = text.split_whitespace().count() as u32;

    if text.is_empty() { 0 } else { words.max(1) }
}

/// A deterministic stand-in for a real model.
///
/// The same request always gives the same answer, on every platform and every
/// run, so the search stays reproducible while we develop it here. The answer is
/// derived from a hash of the request; it is not a good program, only a stable
/// one, which is all the plumbing needs.
#[derive(Clone, Debug)]
pub struct StubModel {
    name: String,
    /// When true, the answer deliberately fails program validation, so tests can
    /// exercise the one-retry path of the LLM proposer.
    emit_invalid: bool,
}

impl StubModel {
    /// A stub that returns stable, parseable placeholder answers.
    pub fn new(name: &str) -> StubModel {
        StubModel {
            name: name.to_string(),
            emit_invalid: false,
        }
    }

    /// A stub whose answers fail validation, for the retry path.
    pub fn invalid(name: &str) -> StubModel {
        StubModel {
            name: name.to_string(),
            emit_invalid: true,
        }
    }

    /// The deterministic answer text for a request, without the token bookkeeping.
    fn answer(&self, req: &GenRequest) -> String {
        let mut material = String::new();

        material.push_str(&req.system);
        material.push('\u{1}');
        material.push_str(&req.prompt);
        material.push('\u{1}');
        material.push_str(&req.seed.to_string());

        let tag = sha256_hex(material.as_bytes());

        if self.emit_invalid {
            format!("not a program {}", &tag[..8])
        } else {
            format!("{{\"stub\": \"{}\"}}", &tag[..16])
        }
    }
}

impl ModelBackend for StubModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
        let full = self.answer(req);
        let wanted = approx_tokens(&full);
        let allowed = wanted.min(req.max_tokens);

        // Truncate to the token budget by whitespace words, so the counts and the
        // text agree and `max_tokens` has a visible effect.
        let (text, stop_reason) = if allowed < wanted {
            let kept: Vec<&str> = full.split_whitespace().take(allowed as usize).collect();

            (kept.join(" "), StopReason::Length)
        } else {
            (full, StopReason::Stop)
        };

        Ok(GenResponse {
            prompt_tokens: approx_tokens(&req.system) + approx_tokens(&req.prompt),
            completion_tokens: approx_tokens(&text),
            text,
            stop_reason,
            backend: self.name.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(seed: u64) -> GenRequest {
        GenRequest {
            system: "you write programs".to_string(),
            prompt: "make a learner".to_string(),
            max_tokens: 64,
            temperature: 0.0,
            seed,
            think_tokens: 0,
        }
    }

    #[test]
    fn stub_is_deterministic() {
        let m = StubModel::new("stub:test");
        let a = m.generate(&req(7)).expect("ok");
        let b = m.generate(&req(7)).expect("ok");

        assert_eq!(a, b);
    }

    #[test]
    fn different_seed_changes_text() {
        let m = StubModel::new("stub:test");
        let a = m.generate(&req(1)).expect("ok");
        let b = m.generate(&req(2)).expect("ok");

        assert_ne!(a.text, b.text);
    }

    #[test]
    fn usable_only_when_clean_and_nonempty() {
        let m = StubModel::new("stub:test");
        let r = m.generate(&req(3)).expect("ok");

        assert_eq!(r.stop_reason, StopReason::Stop);
        assert!(r.is_usable());
        assert!(r.completion_tokens >= 1);
        assert!(r.prompt_tokens >= 1);
        assert_eq!(r.backend, "stub:test");
    }

    #[test]
    fn invalid_stub_differs_from_valid() {
        let good = StubModel::new("stub:test").generate(&req(5)).expect("ok");
        let bad = StubModel::invalid("stub:test").generate(&req(5)).expect("ok");

        assert_ne!(good.text, bad.text);
        assert!(good.is_usable());
        assert!(bad.is_usable()); // still a clean stop; only its content is invalid
    }

    #[test]
    fn max_tokens_truncates_and_flags_length() {
        let m = StubModel::new("stub:test");
        let mut r = req(9);

        r.max_tokens = 1;

        let out = m.generate(&r).expect("ok");

        assert_eq!(out.stop_reason, StopReason::Length);
        assert!(out.completion_tokens <= 1);
    }

    #[test]
    fn approx_tokens_counts_words() {
        assert_eq!(approx_tokens(""), 0);
        assert_eq!(approx_tokens("one"), 1);
        assert_eq!(approx_tokens("one two three"), 3);
    }
}
