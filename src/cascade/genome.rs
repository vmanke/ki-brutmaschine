//! Cascade genome: Workers, their Traits, and the Cascade that chains them (ADR 0005).
//!
//! A Cascade is plain data. Nothing in it is ever executed as code: its prompts are text
//! sent to a model backend, and its Traits are numbers the orchestrator enforces. The
//! hard ceilings below are constants in code, never genes, so evolution cannot raise them
//! (decision Q19); only a code-owner-reviewed pull request can.
#![forbid(unsafe_code)]

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

/// Smallest answer budget that can still hold a 10x10 grid.
pub const ANSWER_TOKENS_MIN: u32 = 64;

/// Text fragments that look like credentials. A Cascade carrying one is rejected, so a
/// prompt can never smuggle a secret into a pull request or a model call.
const SECRET_MARKERS: [&str; 8] = [
    "sk-",
    "ghp_",
    "github_pat_",
    "-----begin",
    "api_key",
    "apikey",
    "password=",
    "secret=",
];

/// One model a Worker may use, with its fixed fitness cost per call (Q21). Costs are set
/// before a run and recorded with it; they are not genes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub tag: String,
    /// Fitness subtracted per call to this model, in solved-rate units.
    pub cost_per_call: f64,
}

/// Run-level limits on Cascades. Every field is clamped to the hard ceilings above, so a
/// configuration can tighten them but never loosen them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    /// out-of-range Traits, oversize prompts, control characters and credential look-alikes.
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

fn check_worker(w: &Worker, limits: &CascadeLimits, stage: usize) -> std::result::Result<(), String> {
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

    let lower = text.to_lowercase();

    if let Some(m) = SECRET_MARKERS.iter().find(|m| lower.contains(*m)) {
        return Err(format!("{what} looks like it carries a credential ({m:?})"));
    }

    Ok(())
}

/// Hand-written starting Cascades. They are deliberately plain: one Worker each, with
/// different strategies, so that search has something to recombine.
pub fn seed_cascades() -> Vec<(String, Cascade)> {
    let worker = |model: &str, strategy: &str, think: u32, check: bool| Worker {
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
    let first = limits
        .models
        .first()
        .map(|m| m.tag.clone())
        .ok_or_else(|| BrutError::Invalid("cascade: the model allow-list is empty".to_string()))?;
    let mut out = Vec::new();

    for (name, mut c) in seed_cascades() {
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
