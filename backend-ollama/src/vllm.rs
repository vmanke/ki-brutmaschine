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
//!
//! # Thinking models (I5)
//!
//! `qwen3.8-27b` (Qwen3.8-27B, served with `--served-model-name qwen3.8-27b
//! --reasoning-parser qwen3`) is the one thinking model on the allow-list. The parser hands
//! the reasoning back in `message.reasoning` (vLLM 0.32; older versions and the Qwen
//! documentation call it `reasoning_content`, and both are read) apart from the visible
//! answer in `message.content`. Only the visible answer reaches `GenResponse::text`.
//!
//! - `think_tokens == 0`: `chat_template_kwargs: {"enable_thinking": false}` and
//!   `max_tokens = max_tokens`, with the model card's sampling for non-thinking mode
//!   besides the request's temperature: top_p 0.8, top_k 20, presence_penalty 1.5.
//! - `think_tokens > 0`: `enable_thinking: true`, `max_tokens = max_tokens + think_tokens`
//!   (that caps the sum only), the model card's sampling for thinking mode besides the
//!   request's temperature (top_p 0.95, top_k 20), and a `reasoning_effort` by these
//!   thresholds, recorded here and nowhere else (`EFFORT_LOW_MAX`, `EFFORT_MEDIUM_MAX`):
//!
//!   | think_tokens  | reasoning_effort | ThinkUp/ThinkDown steps in the band |
//!   |---------------|------------------|-------------------------------------|
//!   | 1 to 2048     | low              | 64 to 2048 (256 is the seed)        |
//!   | 2049 to 3072  | medium           | none (budgets the LLM Mutator sets) |
//!   | 3073 to 4096  | xhigh            | 4096 (THINK_TOKENS_CEILING)         |
//!
//!   Why these bounds: the level only asks the model how long to think (Qwen3.8's chat
//!   template asks at `low` for brief thinking, adds nothing at `medium`, and asks at
//!   `xhigh` to validate assumptions and weigh alternatives); the numeric cap is
//!   `max_tokens` and the two checks below. A probe on 2026-10-31 (one 3x5 grid puzzle
//!   of the cascade domain, two Workers' prompts, temperature 0.2, `max_tokens` 8192, so
//!   nothing was cut) measured these reasoning lengths in server tokens:
//!
//!   | reasoning_effort | reasoning tokens | visible answer tokens |
//!   |------------------|------------------|-----------------------|
//!   | low              | 1921, 2457       | 510, 613              |
//!   | medium           | 2079, 3391       | 491, 348              |
//!   | xhigh            | 2901, 4990       | 35, 35                |
//!
//!   No level fits a budget below about 1900 tokens, so `low`, the shortest, takes every
//!   budget up to 2048. Above that each level gets the band whose top it fitted in one of
//!   the two calls: `medium` up to 3072, `xhigh` up to the ceiling. All three levels lie
//!   inside the 4096-token ceiling, which the build checks. On the powers of two that
//!   ThinkUp and ThinkDown walk, the step 2048 to 4096 goes from `low` straight to `xhigh`;
//!   `medium` is reached only by budgets in between. Six calls on one puzzle are a first
//!   setting, not a measurement: the record keeps the level and the exact reasoning tokens
//!   of every call (`Thinking`), so the bounds can be fitted to the first runs.
//!
//! Each part is then enforced on its own: a visible answer above `max_tokens`, a reasoning
//! part above `think_tokens` (also any reasoning at all with `think_tokens == 0`), or a
//! `finish_reason` of `length` give `StopReason::Length`. The counts come from the server
//! where it reports them (`usage.completion_tokens_details.reasoning_tokens`, which vLLM
//! 0.32 with the reasoning parser does; the visible answer is then `completion_tokens`
//! minus that). Otherwise both parts are whitespace-word estimates of their texts
//! (`approx_tokens`), and the record says so (`Thinking::estimated`). The estimate counts
//! low: the probe above had 0.36 to 0.48 words per reasoning token. A server started
//! without the reasoning parser leaves the reasoning in the content, closed by `</think>`;
//! the part before that marker is then taken as the reasoning, counted as an estimate, and
//! kept out of the answer.
//!
//! The models without a thinking mode (`qwen2.5-*`) send the same body as before thinking
//! was built, and `think_tokens` is ignored for them (I5).

use std::time::Duration;

use serde::Deserialize;

use brutmaschine::backend::{GenRequest, GenResponse, ModelBackend, StopReason, approx_tokens};
use brutmaschine::cascade::genome::THINK_TOKENS_CEILING;
use brutmaschine::util::{BrutError, Result};

use crate::{Detailed, MAX_RESPONSE_BYTES, Thinking, check_loopback, err, wire_seed};

/// The prefix of every vLLM model tag in a run's allow-list: `vllm:qwen2.5-32b`.
pub const TAG_PREFIX: &str = "vllm:";

/// Served model names the adapter may call (I6), all without a thinking mode.
pub const ALLOWED_MODELS: [&str; 3] = ["qwen2.5-7b", "qwen2.5-32b", "qwen2.5-72b"];

/// Served model names with a thinking mode the adapter may call (I5, I6). The server must
/// run them with a reasoning parser (`--reasoning-parser qwen3`).
pub const THINKING_MODELS: [&str; 1] = ["qwen3.8-27b"];

/// Thinking budgets from 1 up to this many tokens ask for reasoning effort `low` (module
/// docs: why).
pub const EFFORT_LOW_MAX: u32 = 2048;

/// Thinking budgets above `EFFORT_LOW_MAX` up to this many tokens ask for `medium`; larger
/// ones ask for `xhigh`.
pub const EFFORT_MEDIUM_MAX: u32 = 3072;

// All three levels must lie inside the core's ceiling on thinking tokens; the build fails
// if a change to either side breaks that.
const _: () = assert!(EFFORT_LOW_MAX < EFFORT_MEDIUM_MAX && EFFORT_MEDIUM_MAX < THINK_TOKENS_CEILING);

/// Sampling of the Qwen3.8 model card for thinking mode, besides the request's temperature.
pub const THINK_TOP_P: f64 = 0.95;
pub const THINK_TOP_K: u32 = 20;

/// Sampling of the Qwen3.8 model card for non-thinking mode, besides the temperature.
pub const PLAIN_TOP_P: f64 = 0.8;
pub const PLAIN_TOP_K: u32 = 20;
pub const PLAIN_PRESENCE_PENALTY: f64 = 1.5;

/// The marker that ends the reasoning in Qwen3.8's output, and the one that opens it.
const THINK_END: &str = "</think>";
const THINK_START: &str = "<think>";

/// The default vLLM server (I3).
pub const DEFAULT_BASE: &str = "http://127.0.0.1:8000";

/// The reasoning effort a thinking budget asks for (module docs); None for 0, which turns
/// thinking off.
pub fn reasoning_effort(think_tokens: u32) -> Option<&'static str> {
    if think_tokens == 0 {
        None
    } else if think_tokens <= EFFORT_LOW_MAX {
        Some("low")
    } else if think_tokens <= EFFORT_MEDIUM_MAX {
        Some("medium")
    } else {
        Some("xhigh")
    }
}

/// One model on a vLLM server.
pub struct VllmBackend {
    tag: String,
    model: String,
    url: String,
    agent: ureq::Agent,
    /// True for a model on `THINKING_MODELS`.
    thinking: bool,
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
    /// The reasoning, as vLLM 0.32 names it.
    #[serde(default)]
    reasoning: Option<String>,
    /// The reasoning, as older vLLM and the Qwen documentation name it.
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    completion_tokens_details: Option<CompletionDetails>,
}

#[derive(Deserialize)]
struct CompletionDetails {
    #[serde(default)]
    reasoning_tokens: Option<u32>,
}

/// The token counts of the two parts of a thinking model's reply.
#[derive(Debug, PartialEq)]
struct Counts {
    reasoning: u32,
    answer: u32,
    estimated: bool,
}

impl VllmBackend {
    /// The backend for `tag` (`vllm:<served model name>`), on the server at `base`. Refuses
    /// a model off the allow-list and a non-loopback server before any call.
    pub fn new(tag: &str, base: &str, timeout: Duration) -> Result<VllmBackend> {
        let model = tag
            .strip_prefix(TAG_PREFIX)
            .ok_or_else(|| BrutError::Invalid(format!("vllm: {tag:?} lacks {TAG_PREFIX}")))?;
        let thinking = THINKING_MODELS.contains(&model);

        if !thinking && !ALLOWED_MODELS.contains(&model) {
            return Err(BrutError::Invalid(format!(
                "vllm: model {model:?} is not on the allow-list {ALLOWED_MODELS:?} or {THINKING_MODELS:?}"
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
            thinking,
        })
    }

    /// True for a model with a thinking mode.
    pub fn is_thinking(&self) -> bool {
        self.thinking
    }

    /// The JSON body of `req`: the system prompt and the prompt as two chat messages,
    /// max_tokens, temperature and the seed masked to 63 bits (`wire_seed`; vLLM refuses
    /// larger ones with HTTP 400). A model without a thinking mode ignores `think_tokens`
    /// (I5); a thinking model gets the switches and caps of the module docs.
    pub fn body(&self, req: &GenRequest) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.prompt},
            ],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "seed": wire_seed(req.seed),
            "stream": false,
        });

        if !self.thinking {
            return body;
        }

        if req.think_tokens == 0 {
            body["top_p"] = serde_json::json!(PLAIN_TOP_P);
            body["top_k"] = serde_json::json!(PLAIN_TOP_K);
            body["presence_penalty"] = serde_json::json!(PLAIN_PRESENCE_PENALTY);
            body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": false});
        } else {
            body["max_tokens"] = serde_json::json!(req.max_tokens.saturating_add(req.think_tokens));
            body["top_p"] = serde_json::json!(THINK_TOP_P);
            body["top_k"] = serde_json::json!(THINK_TOP_K);
            body["reasoning_effort"] = serde_json::json!(reasoning_effort(req.think_tokens));
            body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": true});
        }

        body
    }

    /// POST `req` and read the answer's first choice and the usage.
    fn post(&self, req: &GenRequest) -> Result<(Choice, Option<Usage>)> {
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

        parse(&text)
    }

    /// The answer of a model without a thinking mode: the content, and Length when the
    /// server cut it or it is longer than `max_tokens`.
    fn plain(&self, req: &GenRequest, choice: Choice, usage: Option<Usage>) -> GenResponse {
        let answer = choice.message.content.unwrap_or_default();
        let (prompt, completion) = match usage {
            Some(u) => (u.prompt_tokens, u.completion_tokens),
            None => (0, approx_tokens(&answer)),
        };
        let cut = choice.finish_reason.as_deref() == Some("length") || completion > req.max_tokens;

        GenResponse {
            prompt_tokens: prompt,
            completion_tokens: completion,
            text: answer,
            stop_reason: if cut { StopReason::Length } else { StopReason::Stop },
            backend: self.tag.clone(),
        }
    }

    /// The answer of a thinking model: the visible answer as the text, the reasoning
    /// apart, and Length when either part is above its own budget or the server cut the
    /// reply. `completion_tokens` counts both parts, as the compute they took.
    fn thought(&self, req: &GenRequest, choice: Choice, usage: Option<Usage>) -> (GenResponse, Thinking) {
        let cut_by_server = choice.finish_reason.as_deref() == Some("length");
        let (reasoning, answer) = split(choice.message);
        let counts = measure(usage.as_ref(), &reasoning, &answer);
        let cut = cut_by_server || counts.answer > req.max_tokens || counts.reasoning > req.think_tokens;
        let (prompt, completion) = match usage {
            Some(u) => (u.prompt_tokens, u.completion_tokens),
            None => (0, counts.reasoning.saturating_add(counts.answer)),
        };
        let response = GenResponse {
            prompt_tokens: prompt,
            completion_tokens: completion,
            text: answer,
            stop_reason: if cut { StopReason::Length } else { StopReason::Stop },
            backend: self.tag.clone(),
        };
        let thinking = Thinking {
            enabled: req.think_tokens > 0,
            effort: reasoning_effort(req.think_tokens).map(str::to_string),
            reasoning_tokens: counts.reasoning,
            answer_tokens: counts.answer,
            estimated: counts.estimated,
            reasoning,
        };

        (response, thinking)
    }
}

/// The first choice and the usage of a chat-completion answer.
fn parse(text: &str) -> Result<(Choice, Option<Usage>)> {
    let reply: Reply = serde_json::from_str(text).map_err(BrutError::Json)?;
    let choice = reply
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| err("vllm: the answer has no choice".to_string()))?;

    Ok((choice, reply.usage))
}

/// The reasoning and the visible answer of a thinking model's reply, apart. The reasoning
/// parser hands them over in two fields. Without it, the reasoning stays in the content up
/// to `</think>` (the chat template opened `<think>` in the prompt already), and the part
/// before that marker is the reasoning, so it never reaches the answer. Either way the
/// answer loses the blank lines that follow `</think>` (vLLM 0.32 hands back
/// `"\n\n391"`); its token count stays the server's.
fn split(m: Message) -> (String, String) {
    let content = m.content.unwrap_or_default();
    let parsed = [m.reasoning, m.reasoning_content]
        .into_iter()
        .flatten()
        .find(|r| !r.trim().is_empty());

    if let Some(reasoning) = parsed {
        return (reasoning, content.trim_start().to_string());
    }

    match content.split_once(THINK_END) {
        Some((before, after)) => {
            let before = before.trim_start();
            let reasoning = before.strip_prefix(THINK_START).unwrap_or(before);

            (reasoning.to_string(), after.trim_start().to_string())
        }
        None => (String::new(), content),
    }
}

/// The token counts of the reasoning and the visible answer (I5): exact where the server
/// reports the reasoning tokens, else whitespace-word estimates of both texts.
fn measure(usage: Option<&Usage>, reasoning: &str, answer: &str) -> Counts {
    let thought = !reasoning.trim().is_empty();
    let reported = usage
        .and_then(|u| u.completion_tokens_details.as_ref())
        .and_then(|d| d.reasoning_tokens);

    // A server count of 0 next to reasoning text it returned is not believed: a parser
    // that missed the opening marker (which sits in the prompt) counts nothing.
    if let (Some(u), Some(r)) = (usage, reported)
        && (r > 0 || !thought)
    {
        return Counts {
            reasoning: r,
            answer: u.completion_tokens.saturating_sub(r),
            estimated: false,
        };
    }

    // No reasoning at all: every generated token belongs to the answer.
    if let Some(u) = usage
        && !thought
    {
        return Counts {
            reasoning: 0,
            answer: u.completion_tokens,
            estimated: false,
        };
    }

    Counts {
        reasoning: approx_tokens(reasoning.trim()),
        answer: approx_tokens(answer),
        estimated: true,
    }
}

impl ModelBackend for VllmBackend {
    fn name(&self) -> &str {
        &self.tag
    }

    fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
        self.generate_detailed(req).map(|(r, _)| r)
    }
}

impl Detailed for VllmBackend {
    fn generate_detailed(&self, req: &GenRequest) -> Result<(GenResponse, Option<Thinking>)> {
        let (choice, usage) = self.post(req)?;

        if self.thinking {
            let (response, thinking) = self.thought(req, choice, usage);

            Ok((response, Some(thinking)))
        } else {
            Ok((self.plain(req, choice, usage), None))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_http::serve;
    use crate::{CallRecord, Recorded};

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

    fn think_req(max_tokens: u32, think_tokens: u32) -> GenRequest {
        GenRequest {
            think_tokens,
            ..req(max_tokens)
        }
    }

    fn thinker() -> VllmBackend {
        VllmBackend::new("vllm:qwen3.8-27b", DEFAULT_BASE, Duration::from_secs(1)).expect("new")
    }

    /// The thinking model's reading of a reply, without a server.
    fn read(req: &GenRequest, reply: &str) -> (GenResponse, Thinking) {
        let (choice, usage) = parse(reply).expect("parse");

        thinker().thought(req, choice, usage)
    }

    /// The body as `body` built it before thinking models existed, frozen here.
    fn body_before_thinking(model: &str, req: &GenRequest) -> serde_json::Value {
        serde_json::json!({
            "model": model,
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
    fn the_thinking_model_is_allow_listed_and_only_it_thinks() {
        let t = Duration::from_secs(1);

        assert!(thinker().is_thinking());
        assert!(VllmBackend::new("vllm:qwen3.8-27b", "http://10.0.0.1:8000", t).is_err());
        assert!(VllmBackend::new("vllm:qwen3.8-27b-fp8", DEFAULT_BASE, t).is_err());
        assert!(VllmBackend::new("vllm:qwen3-32b", DEFAULT_BASE, t).is_err());

        for m in ALLOWED_MODELS {
            assert!(!VllmBackend::new(&format!("vllm:{m}"), DEFAULT_BASE, t).expect("new").is_thinking());
        }
    }

    #[test]
    fn a_model_without_thinking_sends_the_body_of_before() {
        for m in ALLOWED_MODELS {
            let b = VllmBackend::new(&format!("vllm:{m}"), DEFAULT_BASE, Duration::from_secs(1)).expect("new");

            for think in [0, 1, 64, 512, 2048, 2049, 3072, 3073, THINK_TOKENS_CEILING] {
                for max_tokens in [64, 512, 2048] {
                    let r = think_req(max_tokens, think);

                    assert_eq!(b.body(&r), body_before_thinking(m, &r), "{m} {think} {max_tokens}");
                    assert_eq!(
                        serde_json::to_string(&b.body(&r)).expect("json"),
                        serde_json::to_string(&body_before_thinking(m, &r)).expect("json"),
                    );
                }
            }
        }

        // The bytes on the wire, pinned once.
        let b = VllmBackend::new("vllm:qwen2.5-32b", DEFAULT_BASE, Duration::from_secs(1)).expect("new");
        let r = GenRequest {
            temperature: 0.5,
            ..think_req(64, 1024)
        };

        assert_eq!(
            serde_json::to_string(&b.body(&r)).expect("json"),
            r#"{"max_tokens":64,"messages":[{"content":"sys","role":"system"},{"content":"hello","role":"user"}],"model":"qwen2.5-32b","seed":9,"stream":false,"temperature":0.5}"#
        );
    }

    #[test]
    fn think_0_turns_thinking_off_with_the_plain_sampling() {
        let body = thinker().body(&think_req(256, 0));

        assert_eq!(body["model"], "qwen3.8-27b");
        assert_eq!(body["chat_template_kwargs"], serde_json::json!({"enable_thinking": false}));
        assert_eq!(body["max_tokens"], 256);
        assert_eq!(body["top_p"], PLAIN_TOP_P);
        assert_eq!(body["top_k"], 20);
        assert_eq!(body["presence_penalty"], 1.5);
        assert_eq!(body["temperature"], serde_json::json!(0.2f32));
        assert_eq!(body["seed"], 9);
        assert_eq!(body["stream"], false);
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn a_small_budget_thinks_at_low_effort_and_caps_only_the_sum() {
        for think in [1, 64, 256, 1024, EFFORT_LOW_MAX] {
            let body = thinker().body(&think_req(512, think));

            assert_eq!(body["chat_template_kwargs"], serde_json::json!({"enable_thinking": true}));
            assert_eq!(body["reasoning_effort"], "low", "{think}");
            assert_eq!(body["max_tokens"], 512 + think);
            assert_eq!(body["top_p"], THINK_TOP_P);
            assert_eq!(body["top_k"], 20);
            assert_eq!(body["temperature"], serde_json::json!(0.2f32));
            assert!(body.get("presence_penalty").is_none());
        }
    }

    #[test]
    fn larger_budgets_think_at_medium_then_xhigh() {
        let effort = |think| thinker().body(&think_req(2048, think))["reasoning_effort"].clone();

        assert_eq!(effort(EFFORT_LOW_MAX + 1), "medium");
        assert_eq!(effort(3000), "medium");
        assert_eq!(effort(EFFORT_MEDIUM_MAX), "medium");
        assert_eq!(effort(EFFORT_MEDIUM_MAX + 1), "xhigh");
        assert_eq!(effort(THINK_TOKENS_CEILING), "xhigh");
        assert_eq!(thinker().body(&think_req(2048, THINK_TOKENS_CEILING))["max_tokens"], 2048 + 4096);
    }

    #[test]
    fn all_three_levels_lie_inside_the_ceiling() {
        assert_eq!(reasoning_effort(0), None);
        assert_eq!(reasoning_effort(1), Some("low"));
        assert_eq!(reasoning_effort(THINK_TOKENS_CEILING), Some("xhigh"));

        // Every step of ThinkUp and ThinkDown (64 doubling to the ceiling) gets a level.
        let steps: Vec<_> = (6..=12).map(|p| reasoning_effort(1 << p).expect("level")).collect();

        assert_eq!(steps, ["low", "low", "low", "low", "low", "low", "xhigh"]);
        assert_eq!(reasoning_effort(2049), Some("medium"));
        assert_eq!(reasoning_effort(3072), Some("medium"));
        assert_eq!(reasoning_effort(3073), Some("xhigh"));
    }

    #[test]
    fn reasoning_and_answer_come_back_apart_with_exact_counts() {
        let (out, t) = read(
            &think_req(64, 512),
            r#"{"choices":[{"message":{"role":"assistant","content":"ANSWER\n3 4","reasoning":"First, 1 + 2 = 3.\nSo the grid is 3 4."},"finish_reason":"stop"}],"usage":{"prompt_tokens":40,"completion_tokens":300,"completion_tokens_details":{"reasoning_tokens":290}}}"#,
        );

        assert_eq!(out.text, "ANSWER\n3 4");
        assert!(!out.text.contains("First"));
        assert_eq!(out.stop_reason, StopReason::Stop);
        assert_eq!(out.prompt_tokens, 40);
        assert_eq!(out.completion_tokens, 300);
        assert_eq!(out.backend, "vllm:qwen3.8-27b");
        assert_eq!(t.reasoning, "First, 1 + 2 = 3.\nSo the grid is 3 4.");
        assert_eq!((t.reasoning_tokens, t.answer_tokens, t.estimated), (290, 10, false));
        assert!(t.enabled);
        assert_eq!(t.effort.as_deref(), Some("low"));
    }

    #[test]
    fn a_reply_as_vllm_0_32_sends_it_reads_apart() {
        // Recorded from the live server (Qwen3.8-27B, --reasoning-parser qwen3), shortened.
        let (out, t) = read(
            &think_req(1024, 32),
            r#"{"id":"chatcmpl-b21f43b06b0db925","object":"chat.completion","model":"qwen3.8-27b","choices":[{"index":0,"message":{"role":"assistant","content":"\n\n391","refusal":null,"annotations":null,"audio":null,"function_call":null,"reasoning":"The user is asking me to multiply 17 by 23.\n\n17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = 391\n"},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":57,"total_tokens":115,"completion_tokens":58,"prompt_tokens_details":null,"completion_tokens_details":{"reasoning_tokens":52}}}"#,
        );

        assert_eq!(out.text, "391");
        assert_eq!((out.prompt_tokens, out.completion_tokens), (57, 58));
        assert_eq!((t.reasoning_tokens, t.answer_tokens, t.estimated), (52, 6, false));
        assert!(t.reasoning.ends_with("= 391\n"));
        // 52 reasoning tokens are above a budget of 32, though the sum fits.
        assert_eq!(out.stop_reason, StopReason::Length);

        let (out, _) = read(
            &think_req(1024, 64),
            r#"{"choices":[{"message":{"content":"\n\n391","reasoning":"x"},"finish_reason":"stop"}],"usage":{"prompt_tokens":57,"completion_tokens":58,"completion_tokens_details":{"reasoning_tokens":52}}}"#,
        );

        assert_eq!(out.stop_reason, StopReason::Stop);
    }

    #[test]
    fn the_older_field_name_reasoning_content_is_read_too() {
        let (out, t) = read(
            &think_req(64, 512),
            r#"{"choices":[{"message":{"content":"ANSWER\n7","reasoning_content":"seven"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":6,"completion_tokens_details":{"reasoning_tokens":2}}}"#,
        );

        assert_eq!(out.text, "ANSWER\n7");
        assert_eq!(t.reasoning, "seven");
        assert_eq!((t.reasoning_tokens, t.answer_tokens), (2, 4));
    }

    #[test]
    fn reasoning_above_think_tokens_is_length() {
        let (out, t) = read(
            &think_req(64, 256),
            r#"{"choices":[{"message":{"content":"ANSWER\n1","reasoning":"long"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":262,"completion_tokens_details":{"reasoning_tokens":257}}}"#,
        );

        assert_eq!((t.reasoning_tokens, t.answer_tokens), (257, 5));
        assert_eq!(out.stop_reason, StopReason::Length);
        assert!(!out.is_usable());

        // At the budget exactly, it stands.
        let (out, _) = read(
            &think_req(64, 256),
            r#"{"choices":[{"message":{"content":"ANSWER\n1","reasoning":"long"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":261,"completion_tokens_details":{"reasoning_tokens":256}}}"#,
        );

        assert_eq!(out.stop_reason, StopReason::Stop);
    }

    #[test]
    fn an_answer_above_max_tokens_is_length_although_the_sum_fits() {
        // 10 reasoning + 70 answer tokens fit the sum cap 64 + 256, but the answer is above 64.
        let (out, t) = read(
            &think_req(64, 256),
            r#"{"choices":[{"message":{"content":"ANSWER\n1 2 3","reasoning":"short"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":80,"completion_tokens_details":{"reasoning_tokens":10}}}"#,
        );

        assert_eq!((t.reasoning_tokens, t.answer_tokens), (10, 70));
        assert_eq!(out.stop_reason, StopReason::Length);
    }

    #[test]
    fn a_reply_the_server_cut_is_length() {
        // Cut inside the reasoning: no visible answer yet.
        let (out, t) = read(
            &think_req(64, 4096),
            r#"{"choices":[{"message":{"content":null,"reasoning":"and then"},"finish_reason":"length"}],"usage":{"prompt_tokens":5,"completion_tokens":30,"completion_tokens_details":{"reasoning_tokens":30}}}"#,
        );

        assert_eq!(out.text, "");
        assert_eq!(t.reasoning, "and then");
        assert_eq!(out.stop_reason, StopReason::Length);
    }

    #[test]
    fn reasoning_with_thinking_off_is_length() {
        let (out, t) = read(
            &think_req(64, 0),
            r#"{"choices":[{"message":{"content":"ANSWER\n1","reasoning":"I should not think"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":12,"completion_tokens_details":{"reasoning_tokens":8}}}"#,
        );

        assert!(!t.enabled);
        assert_eq!(t.effort, None);
        assert_eq!(out.stop_reason, StopReason::Length);

        // The usual reply with thinking off: no reasoning, every token is answer.
        let (out, t) = read(
            &think_req(64, 0),
            r#"{"choices":[{"message":{"content":"ANSWER\n1","reasoning":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":4,"completion_tokens_details":{"reasoning_tokens":0}}}"#,
        );

        assert_eq!((t.reasoning_tokens, t.answer_tokens, t.estimated), (0, 4, false));
        assert_eq!(out.stop_reason, StopReason::Stop);
        assert_eq!(out.text, "ANSWER\n1");
    }

    #[test]
    fn without_a_server_count_both_parts_are_estimates_and_say_so() {
        let reply = r#"{"choices":[{"message":{"content":"ANSWER\n1 2","reasoning":"one two three four five"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":20}}"#;
        let (out, t) = read(&think_req(64, 8), reply);

        assert_eq!((t.reasoning_tokens, t.answer_tokens, t.estimated), (5, 3, true));
        assert_eq!(out.completion_tokens, 20, "the total stays the server's count");
        assert_eq!(out.stop_reason, StopReason::Stop);

        // The estimate is enforced like a count: 5 words of reasoning are above a budget of 4.
        let (out, _) = read(&think_req(64, 4), reply);

        assert_eq!(out.stop_reason, StopReason::Length);

        // A server count of 0 next to reasoning text is not believed.
        let (_, t) = read(
            &think_req(64, 8),
            r#"{"choices":[{"message":{"content":"ANSWER\n1","reasoning":"a b c"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":9,"completion_tokens_details":{"reasoning_tokens":0}}}"#,
        );

        assert_eq!((t.reasoning_tokens, t.answer_tokens, t.estimated), (3, 2, true));

        // No usage at all: both estimated, the total is their sum.
        let (out, t) = read(
            &think_req(64, 8),
            r#"{"choices":[{"message":{"content":"ANSWER\n1","reasoning":"a b c"},"finish_reason":"stop"}]}"#,
        );

        assert!(t.estimated);
        assert_eq!((out.prompt_tokens, out.completion_tokens), (0, 5));
    }

    #[test]
    fn without_the_reasoning_parser_the_reasoning_is_cut_out_of_the_content() {
        let (out, t) = read(
            &think_req(64, 512),
            r#"{"choices":[{"message":{"content":"The grid doubles.\nSo 2 4.\n</think>\n\nANSWER\n2 4"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":20}}"#,
        );

        assert_eq!(out.text, "ANSWER\n2 4");
        assert_eq!(t.reasoning, "The grid doubles.\nSo 2 4.\n");
        assert_eq!((t.reasoning_tokens, t.answer_tokens, t.estimated), (6, 3, true));
        assert_eq!(out.stop_reason, StopReason::Stop);

        // An opening marker in the output is dropped too.
        let (out, t) = read(
            &think_req(64, 512),
            r#"{"choices":[{"message":{"content":"<think>\nhm\n</think>\n\nANSWER\n1"},"finish_reason":"stop"}]}"#,
        );

        assert_eq!(out.text, "ANSWER\n1");
        assert_eq!(t.reasoning, "\nhm\n");
    }

    #[test]
    fn a_thinking_call_goes_over_the_wire_and_its_reasoning_is_filed_not_answered() {
        let (base, server) = serve(
            r#"{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ANSWER\n391","reasoning":"17 * 23 = 391"},"finish_reason":"stop"}],"usage":{"prompt_tokens":17,"completion_tokens":40,"total_tokens":57,"completion_tokens_details":{"reasoning_tokens":36}}}"#,
        );
        let dir = std::env::temp_dir().join(format!("backend-vllm-think-{}", std::process::id()));
        let path = dir.join("worker-calls.jsonl");
        let _ = std::fs::remove_dir_all(&dir);

        let b = VllmBackend::new("vllm:qwen3.8-27b", &base, Duration::from_secs(10)).expect("new");
        let rec = Recorded::open(b, &path).expect("open");
        let r = think_req(64, 3000);
        let out = rec.generate(&r).expect("generate");
        let seen = server.join().expect("server");
        let body: serde_json::Value =
            serde_json::from_str(seen.split("\r\n\r\n").nth(1).expect("body")).expect("json");

        assert!(seen.starts_with("POST /v1/chat/completions "));
        assert_eq!(body["reasoning_effort"], "medium");
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(body["max_tokens"], 64 + 3000);
        assert_eq!(out.text, "ANSWER\n391");
        assert_eq!(out.stop_reason, StopReason::Stop);

        let line = std::fs::read_to_string(&path).expect("record");
        let filed: CallRecord = serde_json::from_str(line.trim()).expect("json");
        let t = filed.thinking.expect("thinking filed");

        assert_eq!(t.reasoning, "17 * 23 = 391");
        assert_eq!((t.reasoning_tokens, t.answer_tokens, t.estimated), (36, 4, false));
        assert_eq!(t.effort.as_deref(), Some("medium"));
        assert!(!filed.response.text.contains("17 * 23"));

        // A new process replays the answer; the server is gone, so no call is made.
        let again = Recorded::open(
            VllmBackend::new("vllm:qwen3.8-27b", &base, Duration::from_secs(1)).expect("new"),
            &path,
        )
        .expect("reopen");

        assert_eq!(again.generate(&r).expect("replay"), out);

        let _ = std::fs::remove_dir_all(&dir);
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
    fn a_model_without_thinking_files_no_thinking_part() {
        let (base, server) =
            serve(r#"{"choices":[{"message":{"content":"ANSWER\n1"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}"#);
        let b = VllmBackend::new("vllm:qwen2.5-32b", &base, Duration::from_secs(10)).expect("new");
        let (out, thinking) = b.generate_detailed(&req(64)).expect("generate");

        server.join().expect("server");
        assert_eq!(out.text, "ANSWER\n1");
        assert_eq!(thinking, None);
    }

    #[test]
    fn a_seed_above_i64_max_reaches_vllm_in_range() {
        let b = VllmBackend::new("vllm:qwen2.5-32b", DEFAULT_BASE, Duration::from_secs(1)).expect("new");
        let mut r = req(8);

        r.seed = u64::MAX - 5;
        assert_eq!(b.body(&r)["seed"], (u64::MAX - 5) & (i64::MAX as u64));
        assert!(b.body(&r)["seed"].as_u64().expect("seed") <= i64::MAX as u64);
        assert!(thinker().body(&r)["seed"].as_u64().expect("seed") <= i64::MAX as u64);
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

    /// Talks to a real server with Qwen3.8 behind the reasoning parser. Run with:
    /// `VLLM_THINK_LIVE=1 cargo test --release -- --ignored live_think --nocapture`
    #[test]
    #[ignore]
    fn live_think_answers_apart_from_its_reasoning() {
        if std::env::var("VLLM_THINK_LIVE").is_err() {
            return;
        }

        let b = VllmBackend::new("vllm:qwen3.8-27b", DEFAULT_BASE, Duration::from_secs(300)).expect("new");

        for think in [0, 512, 4096] {
            let (out, t) = b
                .generate_detailed(&GenRequest {
                    system: "Answer with the number only.".to_string(),
                    prompt: "What is 17 times 23?".to_string(),
                    max_tokens: 32,
                    temperature: 0.6,
                    seed: u64::MAX - 1,
                    think_tokens: think,
                })
                .expect("generate");
            let t = t.expect("thinking");

            eprintln!("LIVE think {think}: {out:?}\n  {t:?}");
            assert!(!out.text.contains("</think>"), "{out:?}");
            assert!(!t.estimated, "the server reports the reasoning tokens");
            assert_eq!(t.enabled, think > 0);
            assert_eq!(t.reasoning_tokens + t.answer_tokens, out.completion_tokens);

            if out.stop_reason == StopReason::Stop {
                assert!(out.text.contains("391"), "{out:?}");
            }
        }
    }
}
