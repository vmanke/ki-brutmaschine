//! Cascade genome: Workers, their Traits, and the Cascade that chains them (ADR 0005).
//!
//! A Cascade is plain data. Nothing in it is ever executed as code: its prompts are text
//! sent to a model backend, and its Traits are numbers the orchestrator enforces. The
//! hard ceilings below are constants in code, never genes, so evolution cannot raise them
//! (decision Q19); only a code-owner-reviewed pull request can.
#![forbid(unsafe_code)]

use std::ops::RangeInclusive;

use serde::{Deserialize, Serialize};

use crate::util::{BrutError, Result, sha256_hex};

/// Hard ceiling on thinking tokens per Worker call. Not inheritable (Q19).
pub const THINK_TOKENS_CEILING: u32 = 4096;

/// Hard ceiling on answer tokens per Worker call (Q16).
pub const ANSWER_TOKENS_CEILING: u32 = 2048;

/// Hard ceiling on Cascade depth (stages) and width (Workers per stage) (Q11).
pub const MAX_STAGES_CEILING: usize = 4;
pub const MAX_WIDTH_CEILING: usize = 3;

/// Bounds on the system prompt: number of named sections and characters per section.
pub const MAX_SECTIONS: usize = 8;
pub const MAX_SECTION_CHARS: usize = 600;
pub const MAX_SECTION_NAME_CHARS: usize = 32;

/// Temperature range a Worker may carry.
pub const TEMPERATURE_MIN: f32 = 0.0;
pub const TEMPERATURE_MAX: f32 = 1.5;

/// Smallest answer budget a Worker may carry. In whitespace-separated tokens
/// (`approx_tokens`) it holds the ANSWER line and a grid of up to 63 cells: a 7x7 grid is
/// 50 tokens. That is enough for every generated puzzle (at most 6x6, 37 tokens). A 10x10
/// grid needs 101 such tokens, and exact tokenizer counts are higher.
pub const ANSWER_TOKENS_MIN: u32 = 64;

/// Characters that render invisibly and can hide a secret or an instruction in a prompt:
/// the Unicode format characters (category Cf) and default-ignorable code points
/// (DerivedCoreProperties), including variation selectors, tag characters and fillers.
const INVISIBLE: [RangeInclusive<char>; 25] = [
    '\u{00AD}'..='\u{00AD}',
    '\u{034F}'..='\u{034F}',
    '\u{0600}'..='\u{0605}',
    '\u{061C}'..='\u{061C}',
    '\u{06DD}'..='\u{06DD}',
    '\u{070F}'..='\u{070F}',
    '\u{0890}'..='\u{0891}',
    '\u{08E2}'..='\u{08E2}',
    '\u{115F}'..='\u{1160}',
    '\u{17B4}'..='\u{17B5}',
    '\u{180B}'..='\u{180F}',
    '\u{200B}'..='\u{200F}',
    '\u{202A}'..='\u{202E}',
    '\u{2060}'..='\u{206F}',
    '\u{3164}'..='\u{3164}',
    '\u{FE00}'..='\u{FE0F}',
    '\u{FEFF}'..='\u{FEFF}',
    '\u{FFA0}'..='\u{FFA0}',
    '\u{FFF0}'..='\u{FFFB}',
    '\u{110BD}'..='\u{110BD}',
    '\u{110CD}'..='\u{110CD}',
    '\u{13430}'..='\u{1343F}',
    '\u{1BCA0}'..='\u{1BCA3}',
    '\u{1D173}'..='\u{1D17A}',
    '\u{E0000}'..='\u{E0FFF}',
];

/// Credential fragments that count anywhere in the lower-cased text.
const SECRET_FRAGMENTS: [&str; 3] = ["-----begin", "api_key", "apikey"];

/// Prefixes of API tokens in the lower-cased text. A prefix counts only at a token start,
/// and only when at least `TOKEN_TAIL_MIN` characters from [a-z0-9_-] follow it.
const TOKEN_PREFIXES: [&str; 19] = [
    // OpenAI, Anthropic and Stripe.
    "sk-",
    "sk_",
    "rk_",
    // GitHub and GitLab.
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "ghr_",
    "github_pat_",
    "glpat-",
    // Slack.
    "xoxa-",
    "xoxb-",
    "xoxp-",
    "xoxr-",
    "xoxs-",
    "xapp-",
    // Hugging Face, npm and PyPI.
    "hf_",
    "npm_",
    "pypi-",
];

/// Shortest tail after a token prefix that makes it look like a key.
const TOKEN_TAIL_MIN: usize = 6;

/// A set of characters, given as a test on one character.
type CharClass = fn(char) -> bool;

/// Keys matched case-sensitively in the original text: a prefix, then this many characters
/// of the class, with no ASCII letter or digit on either side. AKIA and ASIA start AWS
/// access key ids; AIza starts Google API keys.
const CASED_KEYS: [(&str, usize, CharClass); 3] = [
    ("AKIA", 16, is_upper_or_digit),
    ("ASIA", 16, is_upper_or_digit),
    ("AIza", 35, is_key_char),
];

/// Keywords of key-value pairs in the lower-cased text, each with the separators that may
/// follow it, and whether it counts anywhere (true) or only at a token start. Spaces or
/// tabs may stand between the keyword and its separator. "password", "passwd" and
/// "secret" count anywhere, so "PGPASSWORD=" and "dbsecret=" are caught as the old plain
/// markers caught them. "secret" needs '=', because "The secret: ..." is plausible puzzle
/// prose.
const SECRET_KEYS: [(&str, &str, bool); 15] = [
    ("password", ":=", true),
    ("passwd", ":=", true),
    ("api-key", ":=", false),
    ("api key", ":=", false),
    ("access-key", ":=", false),
    ("access_key", ":=", false),
    ("access key", ":=", false),
    ("private-key", ":=", false),
    ("private_key", ":=", false),
    ("private key", ":=", false),
    ("client-secret", ":=", false),
    ("client_secret", ":=", false),
    ("client secret", ":=", false),
    ("authorization", ":=", false),
    ("secret", "=", true),
];

/// Shortest token after "bearer" that makes it look like a credential.
const BEARER_TOKEN_MIN: usize = 16;

/// One model a Worker may use, with its fixed fitness cost per call (Q21). Costs are set
/// before a run and recorded with it; they are not genes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub tag: String,
    /// Fitness subtracted per call to this model, in solved-rate units.
    pub cost_per_call: f64,
}

/// Run-level limits on Cascades. Every field is clamped to the hard ceilings above, so a
/// configuration can tighten them but never loosen them. Unknown keys are rejected, so a
/// setting the code does not know fails loudly instead of being dropped.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CascadeLimits {
    /// Allow-list of models (Q21). A Worker naming any other model is invalid.
    pub models: Vec<ModelSpec>,
    pub max_stages: usize,
    pub max_width: usize,
    /// Thinking-token ceiling per stage index; a missing entry means the global ceiling.
    /// This is the per-sub-cascade cap of Q19 (for example: late stages only check).
    pub think_ceiling_per_stage: Vec<u32>,
}

impl Default for CascadeLimits {
    fn default() -> Self {
        CascadeLimits {
            models: vec![
                ModelSpec {
                    tag: "sim:small".to_string(),
                    cost_per_call: 0.0005,
                },
                ModelSpec {
                    tag: "sim:large".to_string(),
                    cost_per_call: 0.002,
                },
            ],
            max_stages: MAX_STAGES_CEILING,
            max_width: MAX_WIDTH_CEILING,
            think_ceiling_per_stage: Vec::new(),
        }
    }
}

impl CascadeLimits {
    /// Effective depth limit, never above the hard ceiling.
    pub fn stages(&self) -> usize {
        self.max_stages.clamp(1, MAX_STAGES_CEILING)
    }

    /// Effective width limit, never above the hard ceiling.
    pub fn width(&self) -> usize {
        self.max_width.clamp(1, MAX_WIDTH_CEILING)
    }

    /// Effective thinking ceiling for stage `k`, never above the hard ceiling.
    pub fn think_ceiling(&self, k: usize) -> u32 {
        let stage_cap = self
            .think_ceiling_per_stage
            .get(k)
            .copied()
            .unwrap_or(THINK_TOKENS_CEILING);

        stage_cap.min(THINK_TOKENS_CEILING)
    }

    /// The model spec for `tag`, if it is on the allow-list.
    pub fn model(&self, tag: &str) -> Option<&ModelSpec> {
        self.models.iter().find(|m| m.tag == tag)
    }
}

/// The allow-list of the shared candidate pool (`candidates/cascades/`): the simulated
/// models used in this container and the open models planned for host runs (decision
/// Q20; tags as Ollama names them, to be confirmed on the host). Per-call costs here are
/// placeholders; a run's own configuration sets the real ones. Pool checks use the tags.
pub fn pool_limits() -> CascadeLimits {
    let spec = |tag: &str, cost: f64| ModelSpec {
        tag: tag.to_string(),
        cost_per_call: cost,
    };

    CascadeLimits {
        models: vec![
            spec("sim:small", 0.0005),
            spec("sim:large", 0.002),
            spec("gpt-oss:20b", 0.001),
            spec("qwen3:8b", 0.0005),
            spec("qwen3:14b", 0.001),
            spec("gpt-oss:120b", 0.004),
        ],
        ..CascadeLimits::default()
    }
}

/// Traits: inheritable settings that shape *how* a Worker thinks, not *what* it thinks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Traits {
    /// Tokens the Worker may spend reasoning before it answers. 0 means answer at once.
    pub think_tokens: u32,
    pub temperature: f32,
    /// Upper bound on the visible answer.
    pub answer_tokens: u32,
    /// When true, the Worker checks its answer against the worked examples once.
    pub self_check: bool,
}

impl Default for Traits {
    fn default() -> Self {
        Traits {
            think_tokens: 256,
            temperature: 0.2,
            answer_tokens: 512,
            self_check: false,
        }
    }
}

/// One named part of a system prompt, for example "role", "strategy" or "format".
/// Naming the parts lets crossover inherit each part from either parent (Q14).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Section {
    pub name: String,
    pub text: String,
}

/// A Worker: one sub-AI, defined by a model, a system prompt and Traits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Worker {
    pub model: String,
    pub sections: Vec<Section>,
    pub traits: Traits,
}

impl Worker {
    /// The system prompt sent to the backend: sections in order, each under its name.
    pub fn system_prompt(&self) -> String {
        let mut out = String::new();

        for s in &self.sections {
            out.push_str("## ");
            out.push_str(&s.name);
            out.push('\n');
            out.push_str(&s.text);
            out.push_str("\n\n");
        }

        out
    }
}

/// One stage of a Cascade: Workers that run in parallel on the same input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub workers: Vec<Worker>,
}

/// A Cascade: stages run in order; every stage sees the original task plus every earlier
/// stage's chosen answer (the residual path). This is the unit that competes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cascade {
    pub stages: Vec<Stage>,
}

impl Cascade {
    /// Content id: first 16 hex of SHA-256 over the canonical JSON, the same scheme as
    /// `LearnerGenome::id`.
    pub fn id(&self) -> String {
        let json = serde_json::to_string(self).unwrap_or_default();

        sha256_hex(json.as_bytes())[..16].to_string()
    }

    /// Number of Workers over all stages.
    pub fn worker_count(&self) -> usize {
        self.stages.iter().map(|s| s.workers.len()).sum()
    }

    /// Structural validity against `limits` and the hard ceilings. Rejects unknown models,
    /// out-of-range Traits, oversize prompts, control and invisible characters, and common
    /// credential formats.
    pub fn check(&self, limits: &CascadeLimits) -> Result<()> {
        let bad = |m: String| Err(BrutError::Invalid(format!("cascade: {m}")));

        if self.stages.is_empty() || self.stages.len() > limits.stages() {
            return bad(format!(
                "{} stages, allowed 1..={}",
                self.stages.len(),
                limits.stages()
            ));
        }

        for (k, stage) in self.stages.iter().enumerate() {
            if stage.workers.is_empty() || stage.workers.len() > limits.width() {
                return bad(format!(
                    "stage {k} has {} workers, allowed 1..={}",
                    stage.workers.len(),
                    limits.width()
                ));
            }

            for (w, worker) in stage.workers.iter().enumerate() {
                check_worker(worker, limits, k).map_err(|e| {
                    BrutError::Invalid(format!("cascade: stage {k} worker {w}: {e}"))
                })?;
            }
        }

        Ok(())
    }
}

fn check_worker(
    w: &Worker,
    limits: &CascadeLimits,
    stage: usize,
) -> std::result::Result<(), String> {
    if limits.model(&w.model).is_none() {
        return Err(format!("model {:?} is not on the allow-list", w.model));
    }

    if w.sections.is_empty() || w.sections.len() > MAX_SECTIONS {
        return Err(format!(
            "{} prompt sections, allowed 1..={MAX_SECTIONS}",
            w.sections.len()
        ));
    }

    for s in &w.sections {
        check_text("section name", &s.name, MAX_SECTION_NAME_CHARS)?;
        check_text("section text", &s.text, MAX_SECTION_CHARS)?;
    }

    let t = &w.traits;
    let think_cap = limits.think_ceiling(stage);

    if t.think_tokens > think_cap {
        return Err(format!(
            "think_tokens {} above the ceiling {think_cap}",
            t.think_tokens
        ));
    }

    if !(TEMPERATURE_MIN..=TEMPERATURE_MAX).contains(&t.temperature) || !t.temperature.is_finite() {
        return Err(format!("temperature {} out of range", t.temperature));
    }

    if t.answer_tokens < ANSWER_TOKENS_MIN || t.answer_tokens > ANSWER_TOKENS_CEILING {
        return Err(format!(
            "answer_tokens {} outside {ANSWER_TOKENS_MIN}..={ANSWER_TOKENS_CEILING}",
            t.answer_tokens
        ));
    }

    Ok(())
}

fn check_text(what: &str, text: &str, max_chars: usize) -> std::result::Result<(), String> {
    if text.trim().is_empty() {
        return Err(format!("{what} is empty"));
    }

    if text.chars().count() > max_chars {
        return Err(format!("{what} longer than {max_chars} characters"));
    }

    if text.chars().any(|c| c.is_control() && c != '\n') {
        return Err(format!("{what} contains a control character"));
    }

    if text.chars().any(is_invisible) {
        return Err(format!("{what} contains an invisible formatting character"));
    }

    if let Some(m) = credential_marker(text) {
        return Err(format!("{what} looks like it carries a credential ({m:?})"));
    }

    Ok(())
}

/// The first credential marker in `text`, if any. It finds common credential formats:
/// fragments such as a PEM header, token prefixes with a key-like tail, AWS and Google
/// keys, key-value pairs such as "password: ...", and bearer tokens. The check is best
/// effort. It is not a guarantee that no secret can pass.
fn credential_marker(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();

    let fragment = || SECRET_FRAGMENTS.into_iter().find(|m| lower.contains(m));

    let prefix = || {
        TOKEN_PREFIXES.into_iter().find(|p| {
            after_token_start(&lower, p).any(|rest| run_len(rest, is_key_char) >= TOKEN_TAIL_MIN)
        })
    };

    let cased = || {
        CASED_KEYS
            .into_iter()
            .find(|&(p, len, class)| {
                after_token_start(text, p).any(|rest| {
                    let next = rest.chars().nth(len);

                    run_len(rest, class) >= len && !next.is_some_and(|c| c.is_ascii_alphanumeric())
                })
            })
            .map(|(p, _, _)| p)
    };

    let pair = || {
        SECRET_KEYS
            .into_iter()
            .find(|&(k, separators, anywhere)| {
                let separated = |rest: &str| {
                    let unquoted = rest.trim_start_matches(['"', '\'']);
                    // A quoted key ("password": ...) is a key in JSON, YAML or a dict,
                    // so ':' counts for every keyword then.
                    let seps = if unquoted.len() < rest.len() { ":=" } else { separators };

                    unquoted
                        .trim_start_matches([' ', '\t'])
                        .starts_with(|c: char| seps.contains(c))
                };

                if anywhere {
                    lower.match_indices(k).any(|(i, _)| separated(&lower[i + k.len()..]))
                } else {
                    after_token_start(&lower, k).any(separated)
                }
            })
            .map(|(k, _, _)| k)
    };

    let bearer = || {
        after_token_start(&lower, "bearer")
            .any(|rest| {
                rest.starts_with(' ')
                    && run_len(rest.trim_start_matches(' '), is_bearer_char) >= BEARER_TOKEN_MIN
            })
            .then_some("bearer")
    };

    fragment()
        .or_else(prefix)
        .or_else(cased)
        .or_else(pair)
        .or_else(bearer)
}

/// The rest of `text` after each `marker` at a token start: at the start of `text`, or
/// after a character that is not an ASCII letter or digit.
fn after_token_start<'a>(text: &'a str, marker: &str) -> impl Iterator<Item = &'a str> {
    text.match_indices(marker)
        .filter(move |&(i, _)| !text[..i].ends_with(|c: char| c.is_ascii_alphanumeric()))
        .map(move |(i, m)| &text[i + m.len()..])
}

/// Number of leading characters of `s` that are in `class`.
fn run_len(s: &str, class: CharClass) -> usize {
    s.chars().take_while(|&c| class(c)).count()
}

/// A character from the `INVISIBLE` table.
fn is_invisible(c: char) -> bool {
    INVISIBLE.iter().any(|r| r.contains(&c))
}

/// An ASCII letter or digit, '_' or '-': the characters of most API keys.
fn is_key_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// An upper-case ASCII letter or a digit: the characters of an AWS access key id.
fn is_upper_or_digit(c: char) -> bool {
    c.is_ascii_uppercase() || c.is_ascii_digit()
}

/// An ASCII letter or digit, or one of "._~+/=-": the characters of a bearer token.
fn is_bearer_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || "._~+/=-".contains(c)
}

/// Hand-written starting Cascades. They are deliberately plain: one Worker each, with
/// different strategies, so that search has something to recombine.
pub fn seed_cascades() -> Vec<(String, Cascade)> {
    let worker = |model: &str, strategy: &str, think: u32, check: bool| {
        Worker {
            model: model.to_string(),
            sections: vec![
                Section {
                    name: "role".to_string(),
                    text: "You solve grid puzzles. Each puzzle shows input and output grids that follow one hidden rule.".to_string(),
                },
                Section {
                    name: "strategy".to_string(),
                    text: strategy.to_string(),
                },
                Section {
                    name: "format".to_string(),
                    text: "Answer with the line ANSWER followed by the output grid, one row per line, digits separated by spaces.".to_string(),
                },
            ],
            traits: Traits {
                think_tokens: think,
                temperature: 0.2,
                answer_tokens: 512,
                self_check: check,
            },
        }
    };

    let single = |w: Worker| Cascade {
        stages: vec![Stage { workers: vec![w] }],
    };

    vec![
        (
            "mirror".to_string(),
            single(worker(
                "sim:small",
                "Look for mirror images: flip left-right or top-bottom.",
                256,
                false,
            )),
        ),
        (
            "rotate".to_string(),
            single(worker(
                "sim:small",
                "Look for rotations by 90 or 180 degrees, and for transposition.",
                256,
                false,
            )),
        ),
        (
            "colour".to_string(),
            single(worker(
                "sim:small",
                "Look for colour changes: each colour maps to another colour.",
                256,
                true,
            )),
        ),
        (
            "gravity".to_string(),
            single(worker(
                "sim:small",
                "Look for gravity: coloured cells fall down until they rest.",
                256,
                false,
            )),
        ),
    ]
}

/// The seed Cascades adapted to `limits`, so every run starts from valid members: a
/// Worker whose model is off the allow-list takes the first allowed model, and thinking
/// budgets are clamped to each stage's ceiling. Errors when an adapted seed still fails
/// the check (for example when the limits cannot hold even one Worker).
pub fn seeds_for(limits: &CascadeLimits) -> Result<Vec<(String, Cascade)>> {
    adapt_seeds(seed_cascades(), limits)
}

/// `seeds` adapted to `limits` as `seeds_for` adapts the grid seeds: off-list models take
/// the first allowed model, thinking budgets are clamped per stage, and every adapted seed
/// must pass the check.
pub fn adapt_seeds(
    seeds: Vec<(String, Cascade)>,
    limits: &CascadeLimits,
) -> Result<Vec<(String, Cascade)>> {
    let first = limits
        .models
        .first()
        .map(|m| m.tag.clone())
        .ok_or_else(|| BrutError::Invalid("cascade: the model allow-list is empty".to_string()))?;
    let mut out = Vec::new();

    for (name, mut c) in seeds {
        for (k, stage) in c.stages.iter_mut().enumerate() {
            for w in &mut stage.workers {
                if limits.model(&w.model).is_none() {
                    w.model = first.clone();
                }

                w.traits.think_tokens = w.traits.think_tokens.min(limits.think_ceiling(k));
            }
        }

        c.check(limits)
            .map_err(|e| BrutError::Invalid(format!("seed {name}: {e}")))?;
        out.push((name, c));
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_are_valid_and_distinct() {
        let limits = CascadeLimits::default();
        let seeds = seed_cascades();
        let mut ids = std::collections::HashSet::new();

        for (name, c) in &seeds {
            assert!(c.check(&limits).is_ok(), "{name}: {:?}", c.check(&limits));
            assert!(ids.insert(c.id()), "{name} duplicates an id");
        }
    }

    #[test]
    fn ceilings_cannot_be_loosened_by_limits() {
        let limits = CascadeLimits {
            max_stages: 99,
            max_width: 99,
            think_ceiling_per_stage: vec![1_000_000],
            ..CascadeLimits::default()
        };

        assert_eq!(limits.stages(), MAX_STAGES_CEILING);
        assert_eq!(limits.width(), MAX_WIDTH_CEILING);
        assert_eq!(limits.think_ceiling(0), THINK_TOKENS_CEILING);
    }

    #[test]
    fn check_rejects_bad_cascades() {
        let limits = CascadeLimits::default();
        let (_, good) = seed_cascades().remove(0);

        let mut c = good.clone();
        c.stages[0].workers[0].model = "gpt-unknown".to_string();
        assert!(c.check(&limits).is_err(), "unknown model");

        let mut c = good.clone();
        c.stages[0].workers[0].traits.think_tokens = THINK_TOKENS_CEILING + 1;
        assert!(c.check(&limits).is_err(), "thinking above ceiling");

        let mut c = good.clone();
        c.stages[0].workers[0].sections[0].text = "use key sk-abc123".to_string();
        assert!(c.check(&limits).is_err(), "credential look-alike");

        let mut c = good.clone();
        c.stages[0].workers[0].traits.temperature = f32::NAN;
        assert!(c.check(&limits).is_err(), "nan temperature");

        let c = Cascade { stages: vec![] };
        assert!(c.check(&limits).is_err(), "empty cascade");

        let mut c = good.clone();
        let w = c.stages[0].workers[0].clone();
        c.stages[0].workers = vec![w.clone(), w.clone(), w.clone(), w];
        assert!(c.check(&limits).is_err(), "too wide");
    }

    #[test]
    fn text_check_keeps_prose_and_rejects_secrets() {
        let prose = [
            "Look for task-specific rules.",
            "Build a mask-based map.",
            "Desk-check every cell.",
            "A risk-free check of the examples.",
            "Asia-Pacific colours",
            "Slovakia",
            "The secret: count colours first.",
            "Use at most 512 tokens: be brief.",
            "Never ask for a \"password\" or any login.",
            "The word 'secret' is not a colour.",
            "The password's length does not matter here.",
            "Mirror the grid; 한국어, नमस्ते, שלום and مرحبا are fine.",
            "Keep\u{00A0}spacing and smile \u{1F642} without a selector.",
        ];

        // Key-shaped strings are built at run time, so the source holds nothing a secret
        // scanner would take for a real key.
        let google = format!("key AIza{}", "x".repeat(35));
        let ghs = format!("{}{}", "ghs_", "Ab12".repeat(9));
        let gho = format!("{}{}", "gho_", "Ab12".repeat(9));
        let hugging = format!("{}{}", "hf_", "abcd".repeat(9));
        let aws = format!("{}{}", "AKIA", "Z9".repeat(8));
        let leaks = [
            ("ghp_abcdef123", "ghp_"),
            (ghs.as_str(), "ghs_"),
            (gho.as_str(), "gho_"),
            (hugging.as_str(), "hf_"),
            (aws.as_str(), "AKIA"),
            (google.as_str(), "AIza"),
            ("xoxb-123456-abcdef", "xoxb-"),
            ("x-api-key: abc", "api-key"),
            ("password: hunter2", "password"),
            ("PGPASSWORD=abc", "password"),
            ("dbpasswd = abc", "passwd"),
            ("secret=abc", "secret"),
            ("dbsecret=abc", "secret"),
            (
                "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.e30.abc",
                "authorization",
            ),
            ("Send bearer eyJhbGciOiJIUzI1NiJ9.e30.abc", "bearer"),
            ("-----BEGIN RSA PRIVATE KEY-----", "-----begin"),
            ("s\u{200B}k-ant-api03-abcdef", "invisible"),
            ("Be concise.\u{E0041}", "invisible"),
            ("{\"password\": \"hunter2\"}", "password"),
            ("{'password': 'hunter2'}", "password"),
            ("\"client_secret\": \"abc\"", "client_secret"),
            ("\"Authorization\": \"Basic dXNlcg==\"", "authorization"),
            ("{\"secret\": \"abc\"}", "secret"),
        ];

        for text in prose {
            assert_eq!(
                check_text("text", text, MAX_SECTION_CHARS),
                Ok(()),
                "{text}"
            );
        }

        for (text, why) in leaks {
            let err = check_text("text", text, MAX_SECTION_CHARS).expect_err(text);

            assert!(err.contains(why), "{text}: {err}");
        }

        // One character from each kind of invisible range, splitting a key prefix.
        for c in [
            '\u{034F}', '\u{0890}', '\u{115F}', '\u{17B4}', '\u{180B}', '\u{2065}', '\u{3164}',
            '\u{FFA0}', '\u{FFF0}', '\u{110BD}', '\u{13430}', '\u{1BCA0}', '\u{1D173}', '\u{E0FFF}',
        ] {
            let text = format!("s{c}k-ant-api03-abcdef");
            let err = check_text("text", &text, MAX_SECTION_CHARS).expect_err(&text);

            assert!(err.contains("invisible"), "{c:?}: {err}");
        }
    }

    #[test]
    fn limits_reject_unknown_keys() {
        let limits = CascadeLimits::default();
        let json = serde_json::to_string(&limits).expect("encode");
        let back: CascadeLimits = serde_json::from_str(&json).expect("decode");

        assert_eq!(back, limits);

        let extra_limit = json.replacen('{', "{\"answer_tokens_ceiling\": 256, ", 1);
        let extra_model = json.replacen("\"tag\"", "\"max_calls\": 10, \"tag\"", 1);

        for bad in [extra_limit, extra_model] {
            let err = serde_json::from_str::<CascadeLimits>(&bad).expect_err(&bad);

            assert!(err.to_string().contains("unknown field"), "{err}");
        }
    }

    #[test]
    fn per_stage_think_ceiling_applies() {
        let limits = CascadeLimits {
            think_ceiling_per_stage: vec![4096, 0],
            ..CascadeLimits::default()
        };
        let (_, first) = seed_cascades().remove(0);
        let mut c = first.clone();

        c.stages.push(first.stages[0].clone());
        assert!(c.check(&limits).is_err(), "stage 1 may not think");

        c.stages[1].workers[0].traits.think_tokens = 0;
        assert!(c.check(&limits).is_ok());
    }

    #[test]
    fn seeds_for_default_limits_are_the_plain_seeds() {
        let adapted = seeds_for(&CascadeLimits::default()).expect("seeds");

        assert_eq!(adapted, seed_cascades());
    }

    #[test]
    fn seeds_for_an_empty_allow_list_is_an_error() {
        let limits = CascadeLimits {
            models: Vec::new(),
            ..CascadeLimits::default()
        };

        assert!(seeds_for(&limits).is_err());
    }

    #[test]
    fn json_roundtrip_keeps_id() {
        let (_, c) = seed_cascades().remove(1);
        let json = serde_json::to_string_pretty(&c).expect("encode");
        let back: Cascade = serde_json::from_str(&json).expect("decode");

        assert_eq!(c.id(), back.id());
    }
}
