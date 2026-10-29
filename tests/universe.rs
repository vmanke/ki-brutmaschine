//! Step 1 golden tests (ADR 0006): the domain seam must not change a grid run. The
//! fixtures under `tests/fixtures/universe/` were recorded on the grid code before the
//! seam existed. A run with the simulated models, an LLM round, the Mutator's requests and
//! the recorded settings must stay the same, line by line.
//!
//! The only new line in a grid run's log is the `state` event (ticket 01). Each one
//! follows its generation's summary and describes committed state only, so a run that is
//! stopped and resumed logs the same states as a straight one. State events stay out of
//! the golden digest: their entropies use log2 and ln, whose last bit may differ between
//! platforms.
//!
//! The last tests run the `brut-breed` binary. They check the state lines of its log
//! across a resume and across thread counts, `timings.jsonl` (wall time, outside the
//! record), and the options `--domain` and `--selection`.
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

use brutmaschine::backend::{GenRequest, GenResponse, StopReason};
use brutmaschine::cascade::arena::{
    BreedConfig, Breeder, Event, ModelPool, Mutator, Status, breed_to_end, request_key,
};
use brutmaschine::cascade::domain::{CircuitConfig, DomainConfig};
use brutmaschine::cascade::genome::{CascadeLimits, Section, seed_cascades};
use brutmaschine::cascade::observe::UniverseState;
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
            Event::State(_) => continue,
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
        events
            .iter()
            .any(|e| matches!(e, Event::LlmExchange { accepted: true, .. })),
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

/// The state events of `events`, in order.
fn states(events: &[Event]) -> Vec<&UniverseState> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::State(s) => Some(s),
            _ => None,
        })
        .collect()
}

/// The cumulative call count of every generation summary in `events`.
fn calls_used(events: &[Event]) -> Vec<u64> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::Generation { calls_used, .. } => Some(*calls_used),
            _ => None,
        })
        .collect()
}

#[test]
fn every_generation_is_followed_by_exactly_one_state() {
    let llm = BreedConfig {
        llm_every: 1,
        ..small_config()
    };
    let (_, plain) = breed_to_end(small_config(), &sim_pool(), None).expect("run");
    let (_, with_llm) = breed_to_end(llm, &sim_pool(), Some(&Echo)).expect("run");

    for events in [plain, with_llm] {
        let mut generations = 0;

        for (i, e) in events.iter().enumerate() {
            match e {
                Event::Generation { generation, .. } => {
                    assert!(
                        matches!(events.get(i + 1), Some(Event::State(s)) if s.generation == *generation),
                        "generation {generation} is not followed by its state"
                    );
                    generations += 1;
                }
                Event::State(s) => {
                    assert!(
                        i > 0 && matches!(events[i - 1], Event::Generation { .. }),
                        "the state of generation {} does not follow its summary",
                        s.generation
                    );
                }
                _ => {}
            }
        }

        assert_eq!(generations, small_config().generations + 1);
        assert_eq!(states(&events).len(), generations);
    }
}

#[test]
fn a_stopped_and_resumed_run_logs_the_states_of_a_straight_one() {
    let pool = sim_pool();
    let never = || false;
    let mut straight = Breeder::new(small_config(), &pool, None).expect("straight");
    let mut straight_events = Vec::new();

    while let (Status::Continue, ev) = straight.step(&never).expect("step") {
        straight_events.extend(ev);
    }

    // The same run, stopped inside generation 2: the calls made so far stay spent and
    // nothing else is committed, so the resume repeats that generation.
    let mut first = Breeder::new(small_config(), &pool, None).expect("first");
    let mut split_events = Vec::new();

    for _ in 0..2 {
        let (status, ev) = first.step(&never).expect("step");

        assert_eq!(status, Status::Continue);
        split_events.extend(ev);
    }

    let polls = AtomicUsize::new(0);
    let stop_later = || polls.fetch_add(1, Ordering::SeqCst) >= 3;
    let (status, ev) = first
        .step_with(&stop_later, &mut |_| Ok(()))
        .expect("stopped step");

    assert_eq!(status, Status::Done("stopped".to_string()));
    assert!(ev.is_empty());

    let snap = first.snapshot();
    let mut second = Breeder::resume(small_config(), &pool, None, snap).expect("resume");

    while let (Status::Continue, ev) = second.step(&never).expect("step") {
        split_events.extend(ev);
    }

    // The stopped attempt's calls show in the cumulative counts, never in the states.
    assert!(second.calls_used() > straight.calls_used());
    assert_ne!(calls_used(&straight_events), calls_used(&split_events));
    assert_eq!(states(&straight_events), states(&split_events));
    assert_eq!(states(&split_events).len(), small_config().generations + 1);
}

#[test]
fn the_thread_count_changes_no_byte_of_a_run() {
    let run = |threads: usize| {
        let workers = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        let cfg = BreedConfig {
            llm_every: 2,
            ..small_config()
        };
        let (report, events) = workers
            .install(|| breed_to_end(cfg, &sim_pool(), Some(&Echo)))
            .expect("run");

        (
            serde_json::to_string(&report).expect("report json"),
            serde_json::to_string(&events).expect("events json"),
        )
    };

    assert_eq!(run(1), run(4));
}

const BREED: &str = env!("CARGO_BIN_EXE_brut-breed");

/// A fresh, empty directory under the test target directory.
fn fresh(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("universe")
        .join(name);

    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("clean");
    }

    std::fs::create_dir_all(dir.parent().expect("parent")).expect("mkdir");

    dir
}

fn breed(args: &[&str]) -> Output {
    Command::new(BREED)
        .args(args)
        .output()
        .expect("run brut-breed")
}

fn ok(out: &Output) {
    assert!(
        out.status.success(),
        "exit {:?}\nstdout:\n{}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        stderr(out)
    );
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

/// The settings of a small run, except its number of generations.
const SMALL: [&str; 8] = [
    "--seed",
    "5",
    "--children",
    "4",
    "--search",
    "12",
    "--select",
    "12",
];

/// A small run of `generations` generations into `dir`, with the options `extra`.
fn small_run(dir: &Path, generations: &str, extra: &[&str]) -> Output {
    let out = dir.to_str().expect("utf-8 path");
    let mut args = vec!["--out", out, "--generations", generations];

    args.extend_from_slice(&SMALL);
    args.extend_from_slice(extra);

    breed(&args)
}

/// The events of a run's log, as JSON values.
fn logged(dir: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(dir.join("events.jsonl"))
        .expect("log")
        .lines()
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).expect("json");

            v["event"].clone()
        })
        .collect()
}

/// The events of one kind, in order.
fn of_kind(events: &[serde_json::Value], kind: &str) -> Vec<serde_json::Value> {
    events
        .iter()
        .filter(|e| e["kind"] == kind)
        .cloned()
        .collect()
}

/// Every `generation` line is followed by the `state` line of the same generation, and
/// every `state` line follows a `generation` line.
fn assert_states_follow_generations(events: &[serde_json::Value]) {
    for (i, e) in events.iter().enumerate() {
        if e["kind"] == "generation" {
            let next = events.get(i + 1);

            assert!(
                next.is_some_and(|n| n["kind"] == "state" && n["generation"] == e["generation"]),
                "generation {} is not followed by its state",
                e["generation"]
            );
        }

        if e["kind"] == "state" {
            assert!(
                i > 0 && events[i - 1]["kind"] == "generation",
                "the state of generation {} does not follow a summary",
                e["generation"]
            );
        }
    }
}

#[test]
fn the_log_holds_one_state_line_per_generation_line() {
    let dir = fresh("states");

    ok(&small_run(&dir, "3", &["--llm-every", "2", "--no-pause"]));

    let events = logged(&dir);

    assert_states_follow_generations(&events);
    assert_eq!(of_kind(&events, "generation").len(), 4);
    assert_eq!(of_kind(&events, "state").len(), 4);

    // The state lines are part of the hash chain and in canonical form.
    ok(&breed(&["--out", dir.to_str().expect("path"), "--verify"]));
}

#[test]
fn a_run_resumed_to_more_generations_logs_the_states_of_a_straight_run() {
    let straight = fresh("states-straight");
    let split = fresh("states-split");
    let path = split.to_str().expect("path");

    ok(&small_run(&straight, "3", &[]));
    ok(&small_run(&split, "1", &[]));
    ok(&breed(&["--out", path, "--resume", "--generations", "3"]));

    // The split log also holds the first select phase's finalists; the states are the
    // same, line by line.
    let (a, b) = (logged(&straight), logged(&split));

    assert_states_follow_generations(&b);
    assert_ne!(a.len(), b.len());
    assert_eq!(of_kind(&a, "state").len(), 4);
    assert_eq!(of_kind(&a, "state"), of_kind(&b, "state"));
}

#[test]
fn the_thread_count_changes_no_byte_the_binary_writes() {
    let run = |threads: &str| {
        let dir = fresh(&format!("threads-{threads}"));
        let out = Command::new(BREED)
            .args(["--out", dir.to_str().expect("path"), "--generations", "3"])
            .args(SMALL)
            .args(["--llm-every", "2", "--no-pause"])
            .env("RAYON_NUM_THREADS", threads)
            .output()
            .expect("run brut-breed");

        ok(&out);

        ["events.jsonl", "report.json", "snapshot.json"]
            .map(|name| std::fs::read(dir.join(name)).expect(name))
    };

    assert_eq!(run("1"), run("4"));
}

#[test]
fn timings_get_one_line_per_committed_generation_outside_the_record() {
    let dir = fresh("timings");
    let path = dir.to_str().expect("path");

    // A run stopped before generation 0 is committed has no timing yet.
    std::fs::create_dir_all(&dir).expect("mkdir");
    std::fs::write(dir.join("STOP"), "").expect("stop");
    ok(&small_run(&dir, "1", &[]));
    assert!(!dir.join("timings.jsonl").exists());
    std::fs::remove_file(dir.join("STOP")).expect("unstop");

    // Generations 0 and 1 in one process, 2 and 3 in the next. Resuming the finished run
    // once more commits nothing.
    ok(&breed(&["--out", path, "--resume"]));
    ok(&breed(&["--out", path, "--resume", "--generations", "3"]));
    ok(&breed(&["--out", path, "--resume"]));

    let text = std::fs::read_to_string(dir.join("timings.jsonl")).expect("timings");
    let lines: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).expect("json"))
        .collect();

    assert_eq!(lines.len(), 4, "{text}");

    for (g, line) in lines.iter().enumerate() {
        let seconds = line["seconds"].as_f64().expect("seconds");

        assert_eq!(line.as_object().map(|o| o.len()), Some(2), "{line}");
        assert_eq!(line["generation"], g, "{line}");
        assert!(seconds.is_finite() && seconds >= 0.0, "{line}");
    }

    // No hash covers the file and --verify does not read it: an edit, even a line that is
    // not JSON, and its removal leave the run verifiable.
    std::fs::write(dir.join("timings.jsonl"), format!("{text}not json\n")).expect("edit");
    ok(&breed(&["--out", path, "--verify"]));
    std::fs::remove_file(dir.join("timings.jsonl")).expect("remove");
    ok(&breed(&["--out", path, "--verify"]));
}

/// The bytes of the files that record the finished run in `dir`.
fn record(dir: &Path) -> [Vec<u8>; 4] {
    [
        "config.json",
        "snapshot.json",
        "events.jsonl",
        "report.json",
    ]
    .map(|name| std::fs::read(dir.join(name)).expect(name))
}

#[test]
fn domain_and_selection_are_refused_on_resume() {
    let dir = fresh("resume-options");
    let path = dir.to_str().expect("path");

    ok(&small_run(&dir, "1", &[]));

    let before = record(&dir);
    let options = [
        ("--domain", "grid"),
        ("--domain", "circuits"),
        ("--selection", "dgm"),
        ("--selection", "uniform"),
    ];

    for (key, value) in options {
        let refused = breed(&["--out", path, "--resume", key, value]);

        assert_eq!(refused.status.code(), Some(2), "{key} {value}");
        assert!(
            stderr(&refused).contains("cannot be given here"),
            "{}",
            stderr(&refused)
        );
    }

    assert_eq!(record(&dir), before);
}

/// The parent-draw entropy and the archive size of each state line in `dir`'s log.
fn parent_draws(dir: &Path) -> Vec<(f64, f64)> {
    of_kind(&logged(dir), "state")
        .iter()
        .map(|s| {
            (
                s["diversity"]["parent_entropy_bits"]
                    .as_f64()
                    .expect("entropy"),
                s["performance"]["archive_size"].as_f64().expect("size"),
            )
        })
        .collect()
}

#[test]
fn uniform_selection_runs_and_is_recorded() {
    let uniform = fresh("uniform");
    let dgm = fresh("uniform-dgm");
    let path = uniform.to_str().expect("path");

    ok(&small_run(&uniform, "3", &["--selection", "uniform"]));
    ok(&small_run(&dgm, "3", &[]));
    ok(&breed(&["--out", path, "--verify"]));

    // config.json and the run start record it, so it is part of the settings hash.
    let text = std::fs::read_to_string(uniform.join("config.json")).expect("config");
    let config: serde_json::Value = serde_json::from_str(&text).expect("json");
    let start = &logged(&uniform)[0];

    assert_eq!(config["selection"], "uniform");
    assert_eq!(start["kind"], "run_start");
    assert_eq!(start["config"]["selection"], "uniform");
    assert_ne!(start["config_hash"], logged(&dgm)[0]["config_hash"]);

    // Every member weighs the same, so the next parent draw has log2(archive size) bits.
    // DGM selection weighs by fitness and draws from fewer effective parents.
    for (bits, size) in parent_draws(&uniform) {
        assert!(
            (bits - size.log2()).abs() < 1e-9,
            "{bits} bits, {size} members"
        );
    }

    assert!(
        parent_draws(&dgm)
            .iter()
            .any(|&(bits, size)| bits < size.log2() - 0.1)
    );
}

#[test]
fn naming_the_default_domain_and_selection_changes_no_byte() {
    let plain = fresh("defaults-plain");
    let named = fresh("defaults-named");

    ok(&small_run(&plain, "3", &[]));
    ok(&small_run(
        &named,
        "3",
        &["--domain", "grid", "--selection", "dgm"],
    ));

    for name in [
        "config.json",
        "events.jsonl",
        "report.json",
        "snapshot.json",
    ] {
        let read = |dir: &Path| std::fs::read(dir.join(name)).expect(name);

        assert_eq!(read(&plain), read(&named), "{name}");
    }
}

/// A config file in `dir` with the default settings on the circuit domain.
fn circuit_config_file(dir: &Path) -> PathBuf {
    let file = dir.join("circuits.json");
    let cfg = BreedConfig {
        domain: DomainConfig::Circuits(CircuitConfig::default()),
        ..BreedConfig::default()
    };

    std::fs::create_dir_all(dir).expect("mkdir");
    std::fs::write(&file, serde_json::to_string_pretty(&cfg).expect("json")).expect("write");

    file
}

/// A finished grid run whose recorded settings were then edited to name the circuit
/// domain.
fn run_recorded_as_circuits(name: &str) -> PathBuf {
    let dir = fresh(name);

    ok(&small_run(&dir, "1", &[]));

    let text = std::fs::read_to_string(dir.join("config.json")).expect("config");
    let mut cfg: serde_json::Value = serde_json::from_str(&text).expect("json");

    cfg["domain"] = serde_json::json!({ "circuits": {} });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&cfg).expect("json"),
    )
    .expect("edit");

    dir
}

/// Until the integration step of step 1 wires the circuit domain in, every way to a
/// circuit run is wrong usage. That step replaces this test with real circuit runs.
#[test]
fn the_circuit_domain_is_refused_until_the_integration_step() {
    // A fresh run, asked for by the option or by a config file: nothing of a run is
    // written.
    let flag = fresh("circuits-flag");
    let files = fresh("circuits-file");
    let file = circuit_config_file(&files);
    let from_file = files.join("run");
    let refused = [
        small_run(&flag, "1", &["--domain", "circuits"]),
        breed(&[
            "--out",
            from_file.to_str().expect("path"),
            "--config",
            file.to_str().expect("path"),
        ]),
    ];

    for out in &refused {
        assert_eq!(out.status.code(), Some(2));
        assert!(stderr(out).contains("integration step"), "{}", stderr(out));
    }

    for dir in [&flag, &from_file] {
        for name in ["config.json", "snapshot.json", "events.jsonl"] {
            assert!(!dir.join(name).exists(), "{name} was written");
        }
    }

    // A recorded run: --resume and --verify read its settings first and change nothing.
    let dir = run_recorded_as_circuits("circuits-recorded");
    let path = dir.to_str().expect("path");
    let before = record(&dir);

    for mode in ["--resume", "--verify"] {
        let refused = breed(&["--out", path, mode]);

        assert_eq!(refused.status.code(), Some(2), "{mode}");
        assert!(
            stderr(&refused).contains("integration step"),
            "{mode}: {}",
            stderr(&refused)
        );
    }

    assert_eq!(record(&dir), before);
}

#[test]
fn circuit_settings_are_never_dropped_and_circuit_runs_never_promoted() {
    // --domain grid next to circuit settings in a config file would drop them.
    let files = fresh("circuits-kept");
    let file = circuit_config_file(&files);
    let out = files.join("run");
    let contradicted = breed(&[
        "--out",
        out.to_str().expect("path"),
        "--config",
        file.to_str().expect("path"),
        "--domain",
        "grid",
    ]);

    assert_eq!(contradicted.status.code(), Some(2));
    assert!(
        stderr(&contradicted).contains("contradicts"),
        "{}",
        stderr(&contradicted)
    );
    assert!(!out.join("config.json").exists());

    // Promotion of circuit Cascades is not part of step 1. The refusal comes before the
    // run is read, and it writes nothing.
    let dir = run_recorded_as_circuits("circuits-promote");
    let before = record(&dir);
    let promote = breed(&["--out", dir.to_str().expect("path"), "--promote"]);

    assert_eq!(promote.status.code(), Some(1));
    assert!(
        stderr(&promote).contains("promotion of circuit Cascades is not part of step 1 (ADR 0006)"),
        "{}",
        stderr(&promote)
    );
    assert_eq!(record(&dir), before);
    assert!(!dir.join("promote").exists());
}

#[test]
fn unknown_domains_and_selections_are_wrong_usage() {
    for (key, value) in [("--domain", "qubits"), ("--selection", "random")] {
        let refused = small_run(&fresh("unknown-values"), "1", &[key, value]);

        assert_eq!(refused.status.code(), Some(2), "{key} {value}");
        assert!(stderr(&refused).contains("must be"), "{}", stderr(&refused));
    }
}
