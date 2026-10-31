//! brut-breed-ollama: `brut-breed` with real local models through Ollama.
//!
//! It is the same program as `brut-breed` (the file is included below, not copied), with one
//! addition: every allow-listed tag `ollama:<model>` gets an `OllamaBackend`, wrapped in
//! `Recorded`, so every Worker call and its full answer land in
//! `<out>/llm/worker-calls.jsonl` and a resume replays them (contract I7).
//!
//! Put the tag into the run's allow-list with `--config`, for example
//! `{"limits": {"models": [{"tag": "ollama:qwen2.5:7b", "cost_per_call": 0.01}], ...}}`.
//! The seeds then take the first allowed model (`adapt_seeds`).
//!
//! Environment: `OLLAMA_URL` (default http://127.0.0.1:11434, loopback only) and
//! `OLLAMA_TIMEOUT_SECS` (default 300).
#![forbid(unsafe_code)]

#[path = "../../src/bin/brut-breed.rs"]
mod breed;

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Duration;

use backend_ollama::{DEFAULT_BASE, OllamaBackend, Recorded, TAG_PREFIX};
use brutmaschine::backend::ModelBackend;

struct Settings {
    base: String,
    timeout: Duration,
    record: Option<PathBuf>,
}

static SETTINGS: OnceLock<Settings> = OnceLock::new();

/// The backend for an `ollama:` tag; None for any other tag, or when it cannot be built
/// (then `ModelPool::covers` fails the run with the tag named).
fn ollama(tag: &str) -> Option<Box<dyn ModelBackend>> {
    if !tag.starts_with(TAG_PREFIX) {
        return None;
    }

    let s = SETTINGS.get()?;
    let backend = match OllamaBackend::new(tag, &s.base, s.timeout) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("brut-breed-ollama: {e}");
            return None;
        }
    };

    match &s.record {
        Some(path) => match Recorded::open(backend, path) {
            Ok(r) => {
                eprintln!("brut-breed-ollama: {tag}: {} recorded call(s) loaded", r.len());
                Some(Box::new(r))
            }
            Err(e) => {
                eprintln!("brut-breed-ollama: {}: {e}", path.display());
                None
            }
        },
        None => Some(Box::new(backend)),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);
    let timeout = std::env::var("OLLAMA_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);

    let _ = SETTINGS.set(Settings {
        base: std::env::var("OLLAMA_URL").unwrap_or_else(|_| DEFAULT_BASE.to_string()),
        timeout: Duration::from_secs(timeout),
        record: out.map(|o| o.join("llm").join("worker-calls.jsonl")),
    });
    let _ = breed::EXTRA_BACKENDS.set(ollama);

    breed::main();
}
