//! Step 1 golden tests (ADR 0006): the domain seam must not change a grid run. The
//! fixtures under `tests/fixtures/universe/` were recorded on the grid code before the
//! seam existed. A run with the simulated models, an LLM round, the Mutator's requests and
//! the recorded settings must stay the same, line by line.
#![forbid(unsafe_code)]

use std::path::PathBuf;

use brutmaschine::backend::{GenRequest, GenResponse, StopReason};
use brutmaschine::cascade::arena::{
    BreedConfig, Event, ModelPool, Mutator, breed_to_end, request_key,
};
use brutmaschine::cascade::genome::{CascadeLimits, Section, seed_cascades};
use brutmaschine::cascade::sim::SimModel;
use brutmaschine::cascade::variation::{llm_crossover_request, llm_mutate_request};
use brutmaschine::util::sha256_hex;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/universe")
        .join(format!("{name}.txt"))
}

/// Compare `actual` with the recorded fixture; on a difference, print the whole new text
/// and name the first line that differs.
fn golden(name: &str, actual: &str) {
    let expected = std::fs::read_to_string(fixture(name)).unwrap_or_default();

    if expected == actual {
        return;
    }

    eprintln!("---- actual {name} ----\n{actual}---- end {name} ----");

    let first = expected
        .lines()
        .zip(actual.lines())
        .position(|(e, a)| e != a)
        .unwrap_or_else(|| expected.lines().count().min(actual.lines().count()));

    panic!(
        "{name} differs from its fixture at line {}: expected {:?}, got {:?} ({} vs {} lines)",
        first + 1,
        expected.lines().nth(first),
        actual.lines().nth(first),
        expected.lines().count(),
        actual.lines().count()
    );
}

fn sim_pool() -> ModelPool {
    let mut pool = ModelPool::new();

    pool.insert("sim:small", Box::new(SimModel::small()));
    pool.insert("sim:large", Box::new(SimModel::large()));

    pool
}

fn small_config() -> BreedConfig {
    BreedConfig {
        seed: 7,
        generations: 3,
        children_per_generation: 4,
        search_puzzles: 12,
        select_puzzles: 12,
        finalists: 2,
        ..BreedConfig::default()
    }
}

/// A Mutator that always answers: the first seed Cascade with one extra section whose
/// text depends on the request's seed.
struct Echo;

impl Mutator for Echo {
    fn answer(&self, req: &GenRequest) -> Option<GenResponse> {
        let (_, mut cascade) = seed_cascades().swap_remove(0);

        cascade.stages[0].workers[0].sections.push(Section {
            name: "note".to_string(),
            text: format!("Variant {}.", req.seed % 997),
        });

        Some(GenResponse {
            text: serde_json::to_string(&cascade).expect("cascade json"),
            prompt_tokens: 100,
            completion_tokens: 50,
            stop_reason: StopReason::Stop,
            backend: "echo".to_string(),
        })
    }
}

/// Everything a run decides, without floating-point numbers: who was born from whom and
/// how it did, every comparison, every generation's best, the finalists, the Mutator's
/// request keys and verdicts.
fn digest(events: &[Event]) -> String {
    let mut out = String::new();

    for e in events {
        let line = match e {
            Event::RunStart {
                config_hash,
                search,
                select,
                ..
            } => format!(
                "run_start hash={config_hash} search={} select={}",
                search.join(","),
                select.join(",")
            ),
            Event::Born {
                generation,
                id,
                parents,
                op,
                solved,
                puzzles,
                usage,
                ..
            } => format!(
                "born g={generation} id={id} parents={} op={op:?} solved={solved}/{puzzles} \
                 calls={} prompt={} completion={}",
                parents.join(","),
                usage.calls,
                usage.prompt_tokens,
                usage.completion_tokens
            ),
            Event::LlmExchange {
                generation,
                key,
                accepted,
                reason,
                ..
            } => format!("llm g={generation} key={key} accepted={accepted} reason={reason}"),
            Event::Compared(c) => format!(
                "compared {}",
                serde_json::to_string(c).expect("comparison json")
            ),
            Event::Generation {
                generation,
                best_id,
                best_solved,
                archive_size,
                calls_used,
                ..
            } => format!(
                "generation g={generation} best={best_id} solved={best_solved} \
                 size={archive_size} calls={calls_used}"
            ),
            Event::Finalist(f) => format!(
                "finalist id={} seed={} select={}/{}",
                f.id, f.is_seed, f.select_solved, f.select_puzzles
            ),
        };

        out.push_str(&line);
        out.push('\n');
    }

    out
}

#[test]
fn a_grid_run_is_unchanged_by_the_domain_seam() {
    let (_, events) = breed_to_end(small_config(), &sim_pool(), None).expect("run");

    golden("grid_run", &digest(&events));
}

#[test]
fn a_grid_run_with_llm_rounds_is_unchanged_by_the_domain_seam() {
    let cfg = BreedConfig {
        llm_every: 1,
        ..small_config()
    };
    let (_, events) = breed_to_end(cfg, &sim_pool(), Some(&Echo)).expect("run");

    assert!(
        events.iter().any(|e| matches!(e, Event::LlmExchange { accepted: true, .. })),
        "the LLM rounds must admit children, or the fixture tests nothing"
    );

    golden("grid_llm_run", &digest(&events));
}

#[test]
fn mutator_requests_keep_their_keys() {
    let seeds = seed_cascades();
    let limits = CascadeLimits::default();
    let (a, b) = (&seeds[0].1, &seeds[1].1);
    let mutate = llm_mutate_request(a, 0.5, &limits, 42);
    let crossover = llm_crossover_request(a, 0.5, b, 0.25, &limits, 43);
    let text = format!(
        "mutate key={} sha={}\ncrossover key={} sha={}\n",
        request_key(&mutate),
        sha256_hex(serde_json::to_string(&mutate).expect("json").as_bytes()),
        request_key(&crossover),
        sha256_hex(serde_json::to_string(&crossover).expect("json").as_bytes())
    );

    golden("requests", &text);
}

#[test]
fn grid_settings_keep_their_hashes() {
    let cfg = BreedConfig::default();
    let pretty = serde_json::to_string_pretty(&cfg).expect("json");
    let text = format!(
        "fixed_hash={}\npretty_sha={}\n",
        cfg.fixed_hash(),
        sha256_hex(pretty.as_bytes())
    );

    golden("settings", &text);
}
