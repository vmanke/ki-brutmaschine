//! A backend for a batching model server with an OpenAI-compatible API (vLLM), with the
//! guarantees of the Ollama adapter (docs/STAGE2_BACKEND_CONTRACT.md): loopback only and no
//! redirects (I3), an allow-list, a timeout and a byte cap (I6), and an honest Length
//! (I5). Wrap it in `Recorded` for I7.
//!
//! Why a second server: Ollama answers the parallel requests of a panel one after another,
//! while vLLM decodes them together in one batch (docs/ERGEBNISSE_STUFE2C_ERSTE_LAEUFE.md).
//!
//! The server must serve the model under the name in the tag: `vllm:qwen2.5-32b` calls
//! `/v1/chat/completions` with `"model": "qwen2.5-32b"`, so start it with
//! `--served-model-name qwen2.5-32b`.

use std::time::Duration;

use serde::Deserialize;

use brutmaschine::backend::{GenRequest, GenResponse, ModelBackend, StopReason, approx_tokens};
use brutmaschine::util::{BrutError, Result};

use crate::{MAX_RESPONSE_BYTES, check_loopback, err, wire_seed};

/// The prefix of every vLLM model tag in a run's allow-list: `vllm:qwen2.5-32b`.
pub const TAG_PREFIX: &str = "vllm:";

/// Served model names the adapter may call (I6), all without a thinking mode.
pub const ALLOWED_MODELS: [&str; 3] = ["qwen2.5-7b", "qwen2.5-32b", "qwen2.5-72b"];

/// The default vLLM server (I3).
pub const DEFAULT_BASE: &str = "http://127.0.0.1:8000";

/// One model on a vLLM server.
pub struct VllmBackend {
    tag: String,
    model: String,
    url: String,
    agent: ureq::Agent,
}

#[derive(Deserialize)]
struct Reply {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct Message {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

impl VllmBackend {
    /// The backend for `tag` (`vllm:<served model name>`), on the server at `base`. Refuses
    /// a model off the allow-list and a non-loopback server before any call.
    pub fn new(tag: &str, base: &str, timeout: Duration) -> Result<VllmBackend> {
        let model = tag
            .strip_prefix(TAG_PREFIX)
            .ok_or_else(|| BrutError::Invalid(format!("vllm: {tag:?} lacks {TAG_PREFIX}")))?;

        if !ALLOWED_MODELS.contains(&model) {
            return Err(BrutError::Invalid(format!(
                "vllm: model {model:?} is not on the allow-list {ALLOWED_MODELS:?}"
            )));
        }

        check_loopback(base)?;

        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .max_redirects(0)
            .http_status_as_error(false)
            .build()
            .into();

        Ok(VllmBackend {
            tag: tag.to_string(),
            model: model.to_string(),
            url: format!("{}/v1/chat/completions", base.trim_end_matches('/')),
            agent,
        })
    }

    /// The JSON body of `req`: the system prompt and the prompt as two chat messages,
    /// max_tokens, temperature and the seed masked to 63 bits (`wire_seed`; vLLM refuses
    /// larger ones with HTTP 400). `think_tokens` is ignored, as for every model
    /// without a thinking mode (I5).
    pub fn body(&self, req: &GenRequest) -> serde_json::Value {
        serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.prompt},
            ],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "seed": wire_seed(req.seed),
            "stream": false,
        })
    }
}

impl ModelBackend for VllmBackend {
    fn name(&self) -> &str {
        &self.tag
    }

    fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
        let body = serde_json::to_vec(&self.body(req)).map_err(BrutError::Json)?;
        let mut resp = self
            .agent
            .post(&self.url)
            .header("Content-Type", "application/json")
            .send(&body[..])
            .map_err(|e| err(format!("vllm: {e}")))?;
        let status = resp.status();
        let text = resp
            .body_mut()
            .with_config()
            .limit(MAX_RESPONSE_BYTES)
            .read_to_string()
            .map_err(|e| err(format!("vllm: reading the answer: {e}")))?;

        if !status.is_success() {
            let head: String = text.chars().take(300).collect();

            return Err(err(format!("vllm: HTTP {status}: {head}")));
        }

        let reply: Reply = serde_json::from_str(&text).map_err(BrutError::Json)?;
        let choice = reply
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| err("vllm: the answer has no choice".to_string()))?;
        let answer = choice.message.content.unwrap_or_default();
        let (prompt, completion) = match reply.usage {
            Some(u) => (u.prompt_tokens, u.completion_tokens),
            None => (0, approx_tokens(&answer)),
        };
        let cut = choice.finish_reason.as_deref() == Some("length") || completion > req.max_tokens;

        Ok(GenResponse {
            prompt_tokens: prompt,
            completion_tokens: completion,
            text: answer,
            stop_reason: if cut { StopReason::Length } else { StopReason::Stop },
            backend: self.tag.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_http::serve;

    fn req(max_tokens: u32) -> GenRequest {
        GenRequest {
            system: "sys".to_string(),
            prompt: "hello".to_string(),
            max_tokens,
            temperature: 0.2,
            seed: 9,
            think_tokens: 128,
        }
    }

    #[test]
    fn the_allow_list_the_prefix_and_loopback_hold_before_any_call() {
        let t = Duration::from_secs(1);

        assert!(VllmBackend::new("vllm:qwen2.5-32b", DEFAULT_BASE, t).is_ok());
        assert!(VllmBackend::new("vllm:gpt-unknown", DEFAULT_BASE, t).is_err());
        assert!(VllmBackend::new("ollama:qwen2.5:32b", DEFAULT_BASE, t).is_err());
        assert!(VllmBackend::new("vllm:qwen2.5-32b", "http://192.168.1.5:8000", t).is_err());
        assert!(VllmBackend::new("vllm:qwen2.5-32b", "https://127.0.0.1:8000", t).is_err());
    }

    #[test]
    fn a_request_is_a_chat_completion_and_the_answer_maps_back() {
        let (base, server) = serve(
            r#"{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ANSWER\n1 2"},"finish_reason":"stop"}],"usage":{"prompt_tokens":17,"completion_tokens":5,"total_tokens":22}}"#,
        );
        let b = VllmBackend::new("vllm:qwen2.5-32b", &base, Duration::from_secs(10)).expect("new");
        let out = b.generate(&req(64)).expect("generate");
        let seen = server.join().expect("server");
        let body: serde_json::Value =
            serde_json::from_str(seen.split("\r\n\r\n").nth(1).expect("body")).expect("json");

        assert!(seen.starts_with("POST /v1/chat/completions "));
        assert_eq!(body["model"], "qwen2.5-32b");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "sys");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"], "hello");
        assert_eq!(body["max_tokens"], 64);
        assert_eq!(body["seed"], 9);
        assert_eq!(body["stream"], false);
        assert_eq!(out.text, "ANSWER\n1 2");
        assert_eq!(out.prompt_tokens, 17);
        assert_eq!(out.completion_tokens, 5);
        assert_eq!(out.stop_reason, StopReason::Stop);
        assert_eq!(out.backend, "vllm:qwen2.5-32b");
    }

    #[test]
    fn a_seed_above_i64_max_reaches_vllm_in_range() {
        let b = VllmBackend::new("vllm:qwen2.5-32b", DEFAULT_BASE, Duration::from_secs(1)).expect("new");
        let mut r = req(8);

        r.seed = u64::MAX - 5;
        assert_eq!(b.body(&r)["seed"], (u64::MAX - 5) & (i64::MAX as u64));
        assert!(b.body(&r)["seed"].as_u64().expect("seed") <= i64::MAX as u64);
    }

    #[test]
    fn a_cut_off_answer_is_length() {
        let (base, server) = serve(
            r#"{"choices":[{"message":{"content":"a b c"},"finish_reason":"length"}],"usage":{"prompt_tokens":3,"completion_tokens":3}}"#,
        );
        let b = VllmBackend::new("vllm:qwen2.5-32b", &base, Duration::from_secs(10)).expect("new");
        let out = b.generate(&req(3)).expect("generate");

        server.join().expect("server");
        assert_eq!(out.stop_reason, StopReason::Length);
        assert!(!out.is_usable());
    }

    #[test]
    fn an_answer_without_a_choice_is_an_error() {
        let (base, server) = serve(r#"{"choices":[]}"#);
        let b = VllmBackend::new("vllm:qwen2.5-32b", &base, Duration::from_secs(10)).expect("new");

        assert!(b.generate(&req(8)).is_err());
        server.join().expect("server");
    }

    /// Talks to the real local server. Run with:
    /// `VLLM_LIVE=1 cargo test --release -- --ignored live_vllm --nocapture`
    #[test]
    #[ignore]
    fn live_vllm_answers() {
        if std::env::var("VLLM_LIVE").is_err() {
            return;
        }

        let b = VllmBackend::new("vllm:qwen2.5-32b", DEFAULT_BASE, Duration::from_secs(300)).expect("new");
        // A seed above i64::MAX, as the core draws them: the server must accept it.
        let out = b
            .generate(&GenRequest {
                system: "Answer with the number only.".to_string(),
                prompt: "What is 17 times 23?".to_string(),
                max_tokens: 16,
                temperature: 0.0,
                seed: u64::MAX - 1,
                think_tokens: 0,
            })
            .expect("generate");

        eprintln!("LIVE: {out:?}");
        assert!(out.text.contains("391"), "{out:?}");
        assert_eq!(out.stop_reason, StopReason::Stop);
    }
}
