//! Learner-mode end-to-end tests (contract section 10: T-E2E-1 .. T-E2E-9).
//!
//! Every test makes its own tiny run in its own directory, with its own learner secret.
//! No test touches `$HOME`: CLI calls get a private `HOME`, library calls get explicit paths.
//!
//! Cost: each learner run tunes the baselines once (about 3e10 counted FLOPs, contract
//! section 6.6), and so does `brut-eval` for each new secret directory. In an unoptimised
//! build these tests are therefore slow (errata R1).
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use brutmaschine::arc_data::{DataRole, Manifest, ManifestEntry};
use brutmaschine::archive::{Archive, BatchDigestEntry, LearnerEntry, LearnerStatus, RecordKind};
use brutmaschine::config::Limits;
use brutmaschine::evolve::{ARCHIVE_FILE, CONFIG_FILE};
use brutmaschine::learner::protocol::{ProtocolDescriptor, protocol_descriptor, protocol_hash};
use brutmaschine::learner::{
    DslLimits, EvalSet, HurdleConfig, LearnerGenome, MetaConfig, MetaHiddenConfig, MetaWeights,
    PanelConfig, SelectConfig, SourceId, Tier, VariationMode,
};
use brutmaschine::learner_hidden::{ArcEvalRequest, create_learner_secret, request_arc_eval};
use brutmaschine::ledger::{Ledger, LedgerEntry, LedgerKind, QueryKind};
use brutmaschine::lineage_store::{read_chunk, read_snapshot};
use brutmaschine::meta_evolve::{
    LINEAGE_DIR, LearnerEvalSetup, META_CONFIG_FILE, MetaRunSummary, PROGRAMS_DIR, SNAPSHOT_DIR,
    arc_check, init_meta_run, is_meta_run, meta_run,
};
use brutmaschine::meta_report::{Report, write_report};
use brutmaschine::safety;
use brutmaschine::util::{BrutError, sha256_hex};

const EVAL_BIN: &str = env!("CARGO_BIN_EXE_brut-eval");
const BRUT_BIN: &str = env!("CARGO_BIN_EXE_brut");

/// File names next to the learner secret (contract section 7).
const LEDGER_FILE: &str = "ledger.jsonl";
const PROTOCOL_FILE: &str = "protocol.json";
const REPORT_FILE: &str = "report.json";

/// An id no run produces.
const UNKNOWN_ID: &str = "0000000000000000";

/// Hand-made ARC fixture set (tests/fixtures/arc): five A1Eval tasks with six test outputs,
/// plus one training task that plants a rotated and a mirrored, recoloured copy of eval pairs.
const ARC_FIXTURES: [(&str, SourceId, DataRole, Option<EvalSet>); 6] = [
    (
        "arc-agi-1/data/evaluation/e0000001.json",
        SourceId::ArcAgi1Eval,
        DataRole::External,
        Some(EvalSet::A1Eval),
    ),
    (
        "arc-agi-1/data/evaluation/e0000002.json",
        SourceId::ArcAgi1Eval,
        DataRole::External,
        Some(EvalSet::A1Eval),
    ),
    (
        "arc-agi-1/data/evaluation/e0000003.json",
        SourceId::ArcAgi1Eval,
        DataRole::External,
        Some(EvalSet::A1Eval),
    ),
    (
        "arc-agi-1/data/evaluation/e0000004.json",
        SourceId::ArcAgi1Eval,
        DataRole::External,
        Some(EvalSet::A1Eval),
    ),
    (
        "arc-agi-1/data/evaluation/e0000005.json",
        SourceId::ArcAgi1Eval,
        DataRole::External,
        Some(EvalSet::A1Eval),
    ),
    (
        "arc-agi-1/data/training/d0000001.json",
        SourceId::ArcAgi1Training,
        DataRole::Train,
        None,
    ),
];

const ARC_FIXTURE_TASKS: usize = 5;
const ARC_FIXTURE_TEST_OUTPUTS: usize = 6;

/// Default ARC budget per set and protocol (contract section 5.15).
const ARC_BUDGET_PER_SET: usize = 3;

// ---------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------

/// A fresh, empty directory for one test.
fn fresh_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("learner-e2e")
        .join(name);

    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test dir");

    dir
}

/// A learner secret of its own for one test (never under `$HOME`).
fn secret_for(dir: &Path) -> PathBuf {
    create_learner_secret(&dir.join("learner-secret"), None, None).expect("learner secret")
}

/// Directory of a secret file (where the ledger lives).
fn secret_dir(secret: &Path) -> &Path {
    secret.parent().expect("secret dir")
}

fn eval_setup(secret: &Path) -> LearnerEvalSetup {
    LearnerEvalSetup {
        eval_bin: PathBuf::from(EVAL_BIN),
        secret_env: secret.to_path_buf(),
    }
}

/// Tiny learner config: T-E2E-1 sizes (search FLOPs 1e9, population 32, batch 8), small
/// hall of fame, two finalists, few retune trials, rotation and digests inside the run.
/// SELECT task sets keep their defaults, because the final phase and the baseline tuning
/// are defined on them (contract section 6.7, errata E8).
fn tiny_meta(seed: u64, secret: &Path) -> MetaConfig {
    let base = MetaConfig::default();

    MetaConfig {
        master_seed: seed,
        population: 32,
        tournament: 4,
        batch: 8,
        panel: PanelConfig {
            rotate_every: 3,
            ..PanelConfig::default()
        },
        hurdle: HurdleConfig {
            quantile: 0.75,
            window: 64,
            warmup: 16,
        },
        fec_capacity: 100_000,
        select: SelectConfig {
            every_batches: 1_000,
            hof_size: 4,
            finalists: 2,
            retune_trials: 4,
            ..SelectConfig::default()
        },
        snapshot_every: 2,
        limits: Limits {
            max_total_flops: 1_000_000_000,
            max_total_seconds: 3_600.0,
            max_candidate_seconds: 30.0,
            ..base.limits
        },
        threads: 2,
        hidden: MetaHiddenConfig {
            secret_file: secret.to_path_buf(),
            ..base.hidden.clone()
        },
        ..base
    }
}

/// init_meta_run + meta_run in one step.
fn init_and_run(
    run: &Path,
    cfg: &MetaConfig,
    eval: Option<&LearnerEvalSetup>,
    no_hidden: bool,
) -> MetaRunSummary {
    init_meta_run(run, cfg).expect("init_meta_run");

    meta_run(run, eval, no_hidden).expect("meta_run")
}

fn open_archive(run: &Path) -> Archive {
    Archive::open(&run.join(ARCHIVE_FILE)).expect("open archive")
}

fn record_count(run: &Path) -> usize {
    open_archive(run).records().len()
}

fn digests(archive: &Archive) -> Vec<BatchDigestEntry> {
    archive
        .records()
        .iter()
        .filter_map(|r| match &r.kind {
            RecordKind::BatchDigest(d) => Some(d.clone()),
            _ => None,
        })
        .collect()
}

/// A digest with its chunk file name blanked (hashes and everything else kept).
fn without_chunk_name(d: BatchDigestEntry) -> BatchDigestEntry {
    BatchDigestEntry {
        chunk_file: String::new(),
        ..d
    }
}

fn count_kind(archive: &Archive, pred: impl Fn(&RecordKind) -> bool) -> usize {
    archive.records().iter().filter(|r| pred(&r.kind)).count()
}

/// Run `brut` with a private HOME, so the default secret path never points at the real one.
fn brut(args: &[&str], home: &Path) -> Output {
    Command::new(BRUT_BIN)
        .args(args)
        .env("HOME", home)
        .output()
        .expect("run brut")
}

fn code(out: &Output) -> Option<i32> {
    out.status.code()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn path_str(p: &Path) -> &str {
    p.to_str().expect("utf8 path")
}

/// `x` is finite and a whole multiple of `step` (released hidden numbers).
fn assert_step_multiple(x: f64, step: f64, what: &str) {
    assert!(x.is_finite(), "{what}: {x} is not finite");

    let k = x / step;

    assert!(
        (k - k.round()).abs() < 1e-6,
        "{what}: {x} is not a multiple of {step}"
    );
}

/// The program file of a Finalist or Baseline entry exists, parses and matches the entry.
fn assert_program_file(run: &Path, entry: &LearnerEntry) {
    let file = entry
        .program_file
        .as_ref()
        .unwrap_or_else(|| panic!("{} has no program file", entry.id));

    // "Relative path under programs/": accept it relative to the run or to programs/.
    let direct = run.join(file);
    let path = if direct.is_file() {
        direct
    } else {
        run.join(PROGRAMS_DIR).join(file)
    };
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let genome: LearnerGenome = serde_json::from_slice(&bytes).expect("program json");

    assert_eq!(
        genome,
        entry.genome,
        "{} differs from the archive",
        path.display()
    );
    assert_eq!(genome.id(), entry.id, "{} has another id", path.display());
    assert!(
        genome.check(&DslLimits::default()).is_ok(),
        "{} fails check()",
        entry.id
    );
}

/// All ledger entries next to a learner secret (empty when there is no ledger yet).
fn ledger_entries(secret: &Path) -> Vec<LedgerEntry> {
    let path = secret_dir(secret).join(LEDGER_FILE);
    let Ok(body) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };

    body.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("ledger entry"))
        .collect()
}

/// Learner charges of one run, read through the Ledger API after verifying the chain.
fn learner_charges(secret: &Path, protocol: &str, run_id: &str) -> usize {
    Ledger::verify(&secret_dir(secret).join(LEDGER_FILE)).expect("ledger verifies");

    let ledger = Ledger::open(secret_dir(secret)).expect("open ledger");

    ledger.learner_charges(protocol, Some(run_id))
}

fn run_identity(run: &Path) -> (String, String) {
    let archive = open_archive(run);
    let protocol = archive.protocol_hash().expect("protocol hash").to_string();
    let run_id = archive.run_id().expect("run id").to_string();

    (protocol, run_id)
}

/// Archive record kinds with the one field that may differ between thread counts zeroed.
fn normalized_kinds(run: &Path) -> Vec<RecordKind> {
    open_archive(run)
        .records()
        .iter()
        .map(|r| {
            let mut kind = r.kind.clone();

            if let RecordKind::MetaRunStarted { config, .. } = &mut kind {
                config.threads = 0;
            }

            kind
        })
        .collect()
}

fn has_lineage_chunk(run: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(run.join(LINEAGE_DIR)) else {
        return false;
    };

    entries.flatten().any(|e| {
        let name = e.file_name().to_string_lossy().into_owned();

        name.starts_with("chunk-") && name.ends_with(".jsonl")
    })
}

/// Value of one key in a `.env` style file.
fn env_value(path: &Path, key: &str) -> Option<String> {
    let body = std::fs::read_to_string(path).ok()?;

    body.lines().find_map(|l| {
        l.trim()
            .strip_prefix(key)
            .and_then(|rest| rest.strip_prefix('='))
            .map(|v| v.trim().to_string())
    })
}

/// Change one ASCII digit of a file (a content edit that keeps it readable).
/// A file without digits gets one extra byte instead.
fn flip_a_digit(bytes: &[u8]) -> Vec<u8> {
    let mut out = bytes.to_vec();

    match out.iter().position(|b| b.is_ascii_digit()) {
        Some(pos) => out[pos] = if out[pos] == b'1' { b'2' } else { b'1' },
        None => out.push(b'\n'),
    }

    out
}

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// Write a manifest for the ARC fixtures with freshly computed hashes.
fn write_fixture_manifest(root: &Path, out: &Path) {
    let entries = ARC_FIXTURES
        .iter()
        .map(|&(rel, source, role, set)| {
            let bytes = std::fs::read(root.join(rel)).expect("read fixture");

            ManifestEntry {
                rel_path: rel.to_string(),
                sha256: sha256_hex(&bytes),
                // Hand-made fixtures have no upstream repository. The manifest rules
                // still require a licence name and a hex commit, so use neutral values.
                license: "CC0-1.0".to_string(),
                source,
                commit: "0000000".to_string(),
                role,
                set,
            }
        })
        .collect();

    let manifest = Manifest {
        version: 1,
        entries,
    };

    std::fs::write(out, serde_json::to_vec_pretty(&manifest).expect("json")).expect("write");
}

// ---------------------------------------------------------------------------------------
// T-E2E-1
// ---------------------------------------------------------------------------------------

#[test]
fn t_e2e_1_tiny_run_with_hidden_phase() {
    let dir = fresh_dir("tiny");
    let secret = secret_for(&dir);
    let run = dir.join("run");
    let cfg = tiny_meta(1, &secret);
    let summary = init_and_run(&run, &cfg, Some(&eval_setup(&secret)), false);

    // Search summary is sane.
    assert!(summary.batches_done >= 1, "{summary:?}");
    assert!(summary.children > 0);
    assert!(summary.flops_used > 0);
    assert!((0.0..=1.0).contains(&summary.fec_task_hit_rate));
    assert!((0.0..=1.0).contains(&summary.hurdle_pass_rate));
    assert!(summary.wall_seconds.is_finite() && summary.wall_seconds >= 0.0);
    assert!(!summary.stop_reason.is_empty());
    assert_eq!(summary.finalists.len(), cfg.select.finalists);

    // Hidden phase: one result per finalist and tier; one rounded delta, one bit at most.
    assert_eq!(
        summary.hidden.len(),
        cfg.select.finalists * cfg.hidden.tiers.len()
    );

    for id in &summary.finalists {
        for tier in &cfg.hidden.tiers {
            assert!(
                summary
                    .hidden
                    .iter()
                    .any(|h| &h.candidate_id == id && h.tier == *tier),
                "{id} lacks {tier:?}"
            );
        }
    }

    for h in &summary.hidden {
        assert_step_multiple(h.delta, cfg.hidden.score_step, "hidden delta");
        assert_eq!(h.decision.is_some(), h.tier == Tier::HiddenGrammar);
    }

    // Archive verifies; it starts with the pre-registration and ends with RunFinished.
    let path = run.join(ARCHIVE_FILE);
    let n = Archive::verify(&path).expect("verify");
    let archive = open_archive(&run);
    let records = archive.records();

    assert_eq!(n, records.len());

    let RecordKind::MetaRunStarted {
        config,
        protocol_hash: registered,
        ..
    } = &records[0].kind
    else {
        panic!("record 0 is not MetaRunStarted");
    };

    assert_eq!(config.as_ref(), &cfg);
    assert_eq!(registered, &protocol_hash());
    assert!(matches!(
        &records[1].kind,
        RecordKind::ProtocolRegistered { protocol_hash: h, protocol }
            if h == registered && protocol == &protocol_descriptor()
    ));
    assert!(matches!(
        records.last().map(|r| &r.kind),
        Some(RecordKind::RunFinished { .. })
    ));
    assert_eq!(archive.run_id(), Some(records[0].hash.as_str()));
    assert_eq!(archive.protocol_hash(), Some(registered.as_str()));
    assert_eq!(archive.meta_config(), Some(&cfg));
    assert_eq!(archive.learner_hidden_evals(), summary.hidden);
    assert!(is_meta_run(&run));

    // protocol.json is the human copy of the registered protocol.
    let descriptor: ProtocolDescriptor =
        serde_json::from_slice(&std::fs::read(run.join(PROTOCOL_FILE)).expect("protocol.json"))
            .expect("protocol json");

    assert_eq!(descriptor, protocol_descriptor());

    // Programs are saved: every finalist (retuned) and the four program baselines.
    for id in &summary.finalists {
        let entry = archive.learner_get(id).expect("finalist in archive");

        assert!(
            matches!(
                entry.status,
                LearnerStatus::Finalist | LearnerStatus::Quarantined
            ),
            "{id}: {:?}",
            entry.status
        );
        assert_eq!(entry.generator, "retune");
        assert_program_file(&run, &entry);
    }

    let baselines: Vec<LearnerEntry> = archive
        .learner_candidates()
        .into_iter()
        .filter(|c| c.status == LearnerStatus::Baseline)
        .collect();

    assert_eq!(
        baselines.len(),
        4,
        "LinearSgd, TwoLayerNn, Prototype, Perceptron"
    );

    for b in &baselines {
        assert!(b.generator.starts_with("baseline:"), "{}", b.generator);
        assert_program_file(&run, b);
    }

    // report.json has every finalist x tier.
    let report: Report =
        serde_json::from_slice(&std::fs::read(run.join(REPORT_FILE)).expect("report.json"))
            .expect("report json");

    assert_eq!(&report.protocol_hash, registered);
    assert_eq!(report.config, cfg);
    assert_eq!(report.baselines.programs.len(), 4);

    for id in &summary.finalists {
        let f = report
            .finalists
            .iter()
            .find(|f| &f.id == id)
            .unwrap_or_else(|| panic!("{id} missing in report.json"));

        for tier in &cfg.hidden.tiers {
            assert!(
                f.hidden.iter().any(|(t, d, _)| t == tier && d.is_finite()),
                "report: {id} lacks {tier:?}"
            );
        }
    }

    // COR-6: the report written by the run equals one regenerated from the archive, wall
    // time included (both end at RunFinished).
    let regenerated = write_report(&run).expect("regenerate report");

    assert_eq!(regenerated, report);

    // Lineage chunks and the last snapshot are readable with their recorded hashes.
    let all = digests(&archive);
    let mut lineage_records = 0;

    assert!(!all.is_empty(), "no BatchDigest");

    for d in &all {
        let chunk = read_chunk(&run.join(&d.chunk_file), &d.chunk_sha256).expect("chunk");

        lineage_records += chunk.len();
    }

    assert!(lineage_records > 0);

    let last = archive.last_batch_digest().expect("last digest");
    let snapshot =
        read_snapshot(&run.join(&last.snapshot_file), &last.snapshot_sha256).expect("snapshot");

    assert_eq!(snapshot.population.len(), cfg.population);

    // The global ledger verifies and holds exactly one charge per released result.
    let (protocol, run_id) = run_identity(&run);

    assert_eq!(
        learner_charges(&secret, &protocol, &run_id),
        summary.hidden.len()
    );
}

// ---------------------------------------------------------------------------------------
// T-E2E-2
// ---------------------------------------------------------------------------------------

#[test]
fn t_e2e_2_thread_count_does_not_change_the_archive() {
    let dir = fresh_dir("threads");
    let secret = secret_for(&dir);
    let mut runs = Vec::new();

    for threads in [1, 4] {
        let run = dir.join(format!("run-{threads}"));
        let cfg = MetaConfig {
            threads,
            ..tiny_meta(2, &secret)
        };

        init_and_run(&run, &cfg, None, false);
        runs.push(run);
    }

    let a = normalized_kinds(&runs[0]);
    let b = normalized_kinds(&runs[1]);

    assert!(a.len() > 3, "only {} records", a.len());
    assert_eq!(a.len(), b.len(), "record counts differ");

    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x, y, "record {i} differs between 1 and 4 threads");
    }

    // Same chunk hashes, and the chunk files are byte-identical.
    let da = digests(&open_archive(&runs[0]));
    let db = digests(&open_archive(&runs[1]));

    assert!(!da.is_empty());
    assert_eq!(da, db);

    for d in &da {
        let fa = std::fs::read(runs[0].join(&d.chunk_file)).expect("chunk a");
        let fb = std::fs::read(runs[1].join(&d.chunk_file)).expect("chunk b");

        assert_eq!(fa, fb, "{} differs", d.chunk_file);
    }
}

// ---------------------------------------------------------------------------------------
// T-E2E-3
// ---------------------------------------------------------------------------------------

#[test]
fn t_e2e_3_stop_mid_run_and_resume_match_an_uninterrupted_run() {
    let dir = fresh_dir("stop-resume");
    let secret = secret_for(&dir);
    let base = tiny_meta(3, &secret);

    // Bounded by batches, not FLOPs; long enough that a STOP lands inside the search.
    let cfg = MetaConfig {
        max_batches: 40,
        limits: Limits {
            max_total_flops: 1_000_000_000_000,
            ..base.limits
        },
        ..base
    };

    // Reference: uninterrupted.
    let run_a = dir.join("uninterrupted");
    let full = init_and_run(&run_a, &cfg, None, true);

    // Same experiment, stopped once the first lineage chunk exists.
    let run_b = dir.join("stopped");

    init_meta_run(&run_b, &cfg).expect("init");

    let done = AtomicBool::new(false);

    let (first, stopped_at, ended_at) = std::thread::scope(|scope| {
        let watcher = scope.spawn(|| {
            while !done.load(Ordering::Relaxed) {
                if has_lineage_chunk(&run_b) {
                    safety::request_stop(&run_b).expect("stop");

                    return Some(Instant::now());
                }

                std::thread::sleep(Duration::from_millis(5));
            }

            None
        });

        let result = meta_run(&run_b, None, true);
        let ended = Instant::now();

        done.store(true, Ordering::Relaxed);

        (result, watcher.join().expect("watcher"), ended)
    });

    let first = first.expect("a stop is not an error");
    let stopped_at = stopped_at.expect("the run ended before it wrote a lineage chunk");
    let seconds = ended_at.duration_since(stopped_at).as_secs_f64();

    assert!(
        first.batches_done < cfg.max_batches,
        "the run finished before the STOP took effect"
    );
    assert!(seconds <= 2.0, "run ended {seconds:.2} s after STOP");

    // Resume without STOP.
    std::fs::remove_file(run_b.join(safety::STOP_FILE)).expect("remove STOP");

    let resumed = meta_run(&run_b, None, true).expect("resume");

    assert_eq!(resumed.batches_done, full.batches_done);
    assert_eq!(resumed.finalists, full.finalists);

    // Every digest over the same batch range is identical, the final one included.
    // Chunk file names may be numbered differently after a stop; their hashes may not.
    let da: Vec<BatchDigestEntry> = digests(&open_archive(&run_a))
        .into_iter()
        .map(without_chunk_name)
        .collect();
    let db: Vec<BatchDigestEntry> = digests(&open_archive(&run_b))
        .into_iter()
        .map(without_chunk_name)
        .collect();
    let mut matched = 0;

    for d in &db {
        let same_range = da
            .iter()
            .find(|a| a.first_batch == d.first_batch && a.last_batch == d.last_batch);

        if let Some(a) = same_range {
            assert_eq!(
                a, d,
                "digest of batches {}..={} differs",
                d.first_batch, d.last_batch
            );

            matched += 1;
        }
    }

    assert!(matched >= 2, "only {matched} comparable digests");
    assert_eq!(da.last(), db.last(), "final state differs after resume");
}

// ---------------------------------------------------------------------------------------
// T-E2E-4
// ---------------------------------------------------------------------------------------

#[test]
fn t_e2e_4_resume_refuses_a_changed_experiment() {
    let dir = fresh_dir("resume-changed");
    let secret = secret_for(&dir);
    let run = dir.join("run");
    let cfg = MetaConfig {
        max_batches: 2,
        ..tiny_meta(4, &secret)
    };
    let first = init_and_run(&run, &cfg, None, true);
    let before = record_count(&run);
    let meta_path = run.join(META_CONFIG_FILE);

    // Each change alters what archived candidates mean.
    let changed = [
        (
            "master_seed",
            MetaConfig {
                master_seed: 99,
                ..cfg.clone()
            },
        ),
        (
            "variation",
            MetaConfig {
                variation: VariationMode::ConstantsOnly,
                ..cfg.clone()
            },
        ),
        (
            "population",
            MetaConfig {
                population: 48,
                ..cfg.clone()
            },
        ),
        (
            "panel.rotate_every",
            MetaConfig {
                panel: PanelConfig {
                    rotate_every: 7,
                    ..cfg.panel.clone()
                },
                ..cfg.clone()
            },
        ),
        (
            "weights.breadth",
            MetaConfig {
                weights: MetaWeights {
                    breadth: 0.5,
                    ..cfg.weights.clone()
                },
                ..cfg.clone()
            },
        ),
        (
            "dsl.max_learn",
            MetaConfig {
                dsl: DslLimits {
                    max_learn: 16,
                    ..cfg.dsl
                },
                ..cfg.clone()
            },
        ),
        (
            "hidden.gap_threshold",
            MetaConfig {
                hidden: MetaHiddenConfig {
                    gap_threshold: 0.1,
                    ..cfg.hidden.clone()
                },
                ..cfg.clone()
            },
        ),
    ];

    for (field, c) in changed {
        let conflicts = c.resume_conflicts(&cfg);

        assert!(
            conflicts.iter().any(|f| f == field),
            "{field}: conflicts {conflicts:?}"
        );

        // Also ask for more batches, so an accepted resume would have work to do.
        let c = MetaConfig {
            max_batches: 4,
            ..c
        };

        c.save(&meta_path).expect("save");

        let result = meta_run(&run, None, true);

        assert!(
            matches!(&result, Err(BrutError::Invalid(_))),
            "{field}: {:?}",
            result.err()
        );
        assert_eq!(record_count(&run), before, "{field}: archive changed");
    }

    // More batches, other threads and other run-time limits are fine.
    let allowed = MetaConfig {
        max_batches: 4,
        threads: 1,
        limits: Limits {
            max_candidate_seconds: 20.0,
            max_total_seconds: 1_800.0,
            max_total_flops: cfg.limits.max_total_flops * 2,
            ..cfg.limits
        },
        ..cfg.clone()
    };

    assert!(allowed.resume_conflicts(&cfg).is_empty());
    allowed.save(&meta_path).expect("save");

    let second = meta_run(&run, None, true).expect("resume with allowed changes");

    assert!(second.batches_done > first.batches_done);
    assert!(record_count(&run) > before);
}

// ---------------------------------------------------------------------------------------
// T-E2E-5
// ---------------------------------------------------------------------------------------

#[test]
fn t_e2e_5_hidden_budget_exhaustion_is_logged_not_fatal() {
    let dir = fresh_dir("budget");
    let secret = secret_for(&dir);
    let run = dir.join("run");
    let base = tiny_meta(5, &secret);

    // Two finalists x two tiers would need four queries; the run may make one.
    let cfg = MetaConfig {
        hidden: MetaHiddenConfig {
            run_query_budget: 1,
            ..base.hidden.clone()
        },
        ..base
    };
    let summary = init_and_run(&run, &cfg, Some(&eval_setup(&secret)), false);

    assert_eq!(summary.hidden.len(), 1);

    let archive = open_archive(&run);
    let safety_events = count_kind(&archive, |k| matches!(k, RecordKind::SafetyEvent { .. }));
    let hidden_evals = count_kind(&archive, |k| {
        matches!(k, RecordKind::LearnerHiddenEval { .. })
    });

    assert!(safety_events >= 1, "no SafetyEvent for the refused queries");
    assert_eq!(hidden_evals, 1);
    assert!(matches!(
        archive.records().last().map(|r| &r.kind),
        Some(RecordKind::RunFinished { .. })
    ));

    // Refused queries are not charged.
    let (protocol, run_id) = run_identity(&run);

    assert_eq!(learner_charges(&secret, &protocol, &run_id), 1);
}

// ---------------------------------------------------------------------------------------
// T-E2E-6
// ---------------------------------------------------------------------------------------

#[test]
fn t_e2e_6_quarantine_is_final_and_the_fixed_list_does_not_shift() {
    let dir = fresh_dir("quarantine");
    let secret = secret_for(&dir);
    let run = dir.join("run");
    let base = tiny_meta(6, &secret);

    // The smallest allowed threshold (2 x score_step) makes a quarantine more likely.
    // Whether one happens depends on the random secret; the checks below hold either way.
    // Bounded by batches, so "more batches" below is an unambiguous request.
    let cfg = MetaConfig {
        max_batches: 3,
        limits: Limits {
            max_total_flops: 1_000_000_000_000,
            ..base.limits
        },
        hidden: MetaHiddenConfig {
            gap_threshold: 0.02,
            ..base.hidden.clone()
        },
        ..base
    };
    let eval = eval_setup(&secret);
    let first = init_and_run(&run, &cfg, Some(&eval), false);
    let archive = open_archive(&run);

    // The quarantine rule of section 6.7.5, on every HiddenInstance result.
    for r in archive.records() {
        let RecordKind::LearnerHiddenEval {
            candidate_id,
            tier: Tier::HiddenInstance,
            delta,
            delta_public,
            quarantined,
            ..
        } = &r.kind
        else {
            continue;
        };

        let public = delta_public.expect("HiddenInstance carries delta_public");

        assert_eq!(
            *quarantined,
            public - delta > cfg.hidden.gap_threshold,
            "{candidate_id}: public {public}, hidden {delta}"
        );

        if *quarantined {
            let entry = archive.learner_get(candidate_id).expect("entry");

            assert_eq!(entry.status, LearnerStatus::Quarantined);
        }
    }

    // Quarantined finalists keep their place in the fixed list.
    for h in first.hidden.iter().filter(|h| h.quarantined) {
        assert!(first.finalists.contains(&h.candidate_id));
    }

    // Running again sends no query and changes no result.
    let (protocol, run_id) = run_identity(&run);
    let charges = learner_charges(&secret, &protocol, &run_id);
    let again = meta_run(&run, Some(&eval), false).expect("run again");

    assert_eq!(again.finalists, first.finalists);
    assert_eq!(again.hidden, first.hidden);
    assert_eq!(learner_charges(&secret, &protocol, &run_id), charges);
    assert_eq!(open_archive(&run).learner_hidden_evals(), first.hidden);

    // More batches after a hidden evaluation are refused, with or without the evaluator
    // (as in MLP mode, where more generations give Err(Invalid)).
    let more = MetaConfig {
        max_batches: cfg.max_batches + 2,
        ..cfg.clone()
    };

    more.save(&run.join(META_CONFIG_FILE)).expect("save");

    let before = record_count(&run);

    for setup in [Some(&eval), None] {
        let result = meta_run(&run, setup, false);

        assert!(
            matches!(&result, Err(BrutError::Invalid(_))),
            "{:?}",
            result.err()
        );
        assert_eq!(record_count(&run), before);
    }
}

// ---------------------------------------------------------------------------------------
// T-E2E-7
// ---------------------------------------------------------------------------------------

#[test]
fn t_e2e_7_tampered_chunk_or_snapshot_fails_verify() {
    let dir = fresh_dir("tamper");
    let home = dir.join("home");
    let secret = secret_for(&dir);
    let run = dir.join("run");
    let cfg = MetaConfig {
        max_batches: 4,
        ..tiny_meta(7, &secret)
    };

    std::fs::create_dir_all(&home).expect("home");
    init_and_run(&run, &cfg, None, true);

    let run_arg = path_str(&run);
    let ok = brut(&["verify", run_arg], &home);

    assert_eq!(code(&ok), Some(0), "{}", text(&ok.stderr));
    assert!(text(&ok.stdout).starts_with("OK "));

    // Every digest's chunk and snapshot are checked; the cache only for the latest.
    // The first chunk covers the hurdle warm-up, so it is never empty.
    let all = digests(&open_archive(&run));
    let first = all.first().expect("digest");
    let files = [
        first.chunk_file.clone(),
        first.snapshot_file.clone(),
        format!("{SNAPSHOT_DIR}/cache-latest.json"),
    ];

    for file in files {
        let path = run.join(&file);
        let original = std::fs::read(&path).unwrap_or_else(|e| panic!("{file}: {e}"));

        // An edited file is detected.
        std::fs::write(&path, flip_a_digit(&original)).expect("edit");

        let edited = brut(&["verify", run_arg], &home);

        assert_eq!(code(&edited), Some(1), "edited {file} not detected");
        assert!(
            matches!(
                Archive::open(&run.join(ARCHIVE_FILE)),
                Err(BrutError::Integrity(_))
            ),
            "Archive::open accepts an edited {file}"
        );

        // A missing file is detected.
        std::fs::remove_file(&path).expect("remove");

        assert_eq!(
            code(&brut(&["verify", run_arg], &home)),
            Some(1),
            "missing {file} not detected"
        );
        assert!(Archive::open(&run.join(ARCHIVE_FILE)).is_err());

        // Restored, all is well again.
        std::fs::write(&path, &original).expect("restore");

        assert_eq!(code(&brut(&["verify", run_arg], &home)), Some(0));
    }
}

// ---------------------------------------------------------------------------------------
// T-E2E-8
// ---------------------------------------------------------------------------------------

/// CLI cycle. `brut init --mode learner` is checked on its own directory: it fixes the
/// experiment in the archive with the default sizes, whose final phase alone costs about
/// 4e10 FLOPs. The rest of the cycle runs on a tiny run made with `init_meta_run`.
#[test]
fn t_e2e_8_cli_cycle() {
    let dir = fresh_dir("cli");
    let home = dir.join("home");
    let secret_dir = dir.join("learner-secret");

    std::fs::create_dir_all(&home).expect("home");

    // secret-init prints the path, never the seed, and refuses to overwrite.
    let out = brut(
        &["learner", "secret-init", "--dir", path_str(&secret_dir)],
        &home,
    );

    assert_eq!(code(&out), Some(0), "{}", text(&out.stderr));

    let secret = secret_dir.join("hidden.env");
    let seed = env_value(&secret, "BRUT_LEARNER_HIDDEN_SEED").expect("seed in secret");
    let printed = format!("{}{}", text(&out.stdout), text(&out.stderr));

    assert!(seed.parse::<u64>().is_ok(), "seed is not a u64");
    assert!(printed.contains(path_str(&secret)), "path not printed");

    // A random u64 has about 19 digits; a short one could appear in the path by chance.
    if seed.len() >= 6 {
        assert!(!printed.contains(&seed), "secret-init printed the seed");
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let file_mode = std::fs::metadata(&secret)
            .expect("meta")
            .permissions()
            .mode();
        let dir_mode = std::fs::metadata(&secret_dir)
            .expect("meta")
            .permissions()
            .mode();

        assert_eq!(file_mode & 0o777, 0o600);
        assert_eq!(dir_mode & 0o777, 0o700);
    }

    let again = brut(
        &["learner", "secret-init", "--dir", path_str(&secret_dir)],
        &home,
    );

    assert_eq!(code(&again), Some(1));
    assert_eq!(
        env_value(&secret, "BRUT_LEARNER_HIDDEN_SEED").as_deref(),
        Some(seed.as_str())
    );

    // init --mode learner: the flags reach meta.json and the archive.
    let cli_run = dir.join("cli-run");
    let init = brut(
        &[
            "init",
            path_str(&cli_run),
            "--mode",
            "learner",
            "--seed",
            "5",
            "--flops",
            "1e9",
            "--variation",
            "constants-only",
            "--secret",
            path_str(&secret),
        ],
        &home,
    );

    assert_eq!(code(&init), Some(0), "{}", text(&init.stderr));
    assert!(is_meta_run(&cli_run));
    assert!(cli_run.join(PROTOCOL_FILE).is_file());

    let loaded = MetaConfig::load(&cli_run.join(META_CONFIG_FILE)).expect("meta.json");

    assert_eq!(loaded.master_seed, 5);
    assert_eq!(loaded.limits.max_total_flops, 1_000_000_000);
    assert_eq!(loaded.variation, VariationMode::ConstantsOnly);
    assert_eq!(loaded.hidden.secret_file, secret);
    assert!(matches!(
        &open_archive(&cli_run).records()[0].kind,
        RecordKind::MetaRunStarted { .. }
    ));

    let twice = brut(
        &[
            "init",
            path_str(&cli_run),
            "--mode",
            "learner",
            "--secret",
            path_str(&secret),
        ],
        &home,
    );

    assert_eq!(code(&twice), Some(1));

    // Usage errors give 2; a missing secret gives 1 and writes nothing.
    let bad_variation = dir.join("bad-variation");
    let bad_mode = dir.join("bad-mode");
    let no_secret = dir.join("no-secret");
    let default_secret = dir.join("default-secret");

    assert_eq!(
        code(&brut(
            &[
                "init",
                path_str(&bad_variation),
                "--mode",
                "learner",
                "--variation",
                "nope",
                "--secret",
                path_str(&secret),
            ],
            &home
        )),
        Some(2)
    );
    assert_eq!(
        code(&brut(
            &["init", path_str(&bad_mode), "--mode", "nope"],
            &home
        )),
        Some(2)
    );

    let missing = dir.join("missing").join("hidden.env");

    assert_eq!(
        code(&brut(
            &[
                "init",
                path_str(&no_secret),
                "--mode",
                "learner",
                "--secret",
                path_str(&missing),
            ],
            &home
        )),
        Some(1)
    );
    assert!(!no_secret.join(ARCHIVE_FILE).exists());

    // Without --secret the default lives under HOME, which has none here.
    assert_eq!(
        code(&brut(
            &["init", path_str(&default_secret), "--mode", "learner"],
            &home
        )),
        Some(1)
    );
    assert!(!default_secret.join(ARCHIVE_FILE).exists());

    // Without --mode, init is the unchanged MLP init.
    let mlp_run = dir.join("mlp-run");

    assert_eq!(code(&brut(&["init", path_str(&mlp_run)], &home)), Some(0));
    assert!(mlp_run.join(CONFIG_FILE).is_file());
    assert!(!is_meta_run(&mlp_run));

    // The rest of the cycle on a tiny run.
    let run = dir.join("run");
    let run_arg = path_str(&run);

    init_meta_run(&run, &tiny_meta(8, &secret)).expect("init_meta_run");

    let ran = brut(&["run", run_arg], &home);

    assert_eq!(code(&ran), Some(0), "{}", text(&ran.stderr));

    let archive = open_archive(&run);
    let evaluated: Vec<String> = archive
        .learner_hidden_evals()
        .into_iter()
        .filter(|h| h.tier == Tier::HiddenGrammar)
        .map(|h| h.candidate_id)
        .collect();

    assert!(!evaluated.is_empty(), "no HiddenGrammar evaluation");

    let status = brut(&["status", run_arg], &home);

    assert_eq!(code(&status), Some(0), "{}", text(&status.stderr));

    for id in &evaluated {
        assert!(
            text(&status.stdout).contains(id.as_str()),
            "status lacks {id}"
        );
    }

    let id = evaluated[0].as_str();
    let lineage = brut(&["lineage", run_arg, id], &home);

    assert_eq!(code(&lineage), Some(0), "{}", text(&lineage.stderr));
    assert!(!lineage.stdout.is_empty());
    assert_eq!(
        code(&brut(&["lineage", run_arg, UNKNOWN_ID], &home)),
        Some(1)
    );

    let show = brut(&["learner", "show", run_arg, id], &home);

    assert_eq!(code(&show), Some(0), "{}", text(&show.stderr));
    assert!(!show.stdout.is_empty());
    assert_eq!(
        code(&brut(&["learner", "show", run_arg, UNKNOWN_ID], &home)),
        Some(1)
    );

    let report = brut(&["learner", "report", run_arg], &home);

    assert_eq!(code(&report), Some(0), "{}", text(&report.stderr));
    assert!(run.join(REPORT_FILE).is_file());

    let verify = brut(&["verify", run_arg], &home);

    assert_eq!(code(&verify), Some(0), "{}", text(&verify.stderr));
    assert!(text(&verify.stdout).starts_with("OK "));

    // approve: needs --by; refuses a program without HiddenGrammar evaluation.
    assert_eq!(code(&brut(&["approve", run_arg, id], &home)), Some(2));

    let baseline = archive
        .learner_candidates()
        .into_iter()
        .find(|c| c.status == LearnerStatus::Baseline)
        .expect("baseline entry");

    assert_eq!(
        code(&brut(
            &["approve", run_arg, &baseline.id, "--by", "tester"],
            &home
        )),
        Some(1)
    );

    let quarantined = archive
        .learner_get(id)
        .is_some_and(|e| e.status == LearnerStatus::Quarantined);
    let approve = brut(
        &[
            "approve",
            run_arg,
            id,
            "--by",
            "tester",
            "--note",
            "looks fine",
        ],
        &home,
    );

    if quarantined {
        assert_eq!(code(&approve), Some(1));
    } else {
        assert_eq!(code(&approve), Some(0), "{}", text(&approve.stderr));

        let approved = count_kind(
            &open_archive(&run),
            |k| matches!(k, RecordKind::Approved { candidate_id, .. } if candidate_id == id),
        );

        assert_eq!(approved, 1);
    }

    // stop sets the kill switch; help lists the learner commands.
    assert_eq!(code(&brut(&["stop", run_arg], &home)), Some(0));
    assert!(run.join(safety::STOP_FILE).exists());

    let help = brut(&["help"], &home);
    let usage = text(&help.stdout);

    assert_eq!(code(&help), Some(0));

    for command in [
        "--mode learner",
        "learner secret-init",
        "learner show",
        "learner report",
        "learner arc-check",
    ] {
        assert!(usage.contains(command), "help lacks {command}");
    }

    // Nothing was written below the private HOME.
    assert!(!home.join(".brut").exists());
}

// ---------------------------------------------------------------------------------------
// T-E2E-9
// ---------------------------------------------------------------------------------------

#[test]
fn t_e2e_9_arc_check_on_fixture_set_with_ledger_entries() {
    let dir = fresh_dir("arc");
    let root = crate_path("tests/fixtures/arc");
    let manifest = dir.join("manifest.json");

    write_fixture_manifest(&root, &manifest);

    let secret = create_learner_secret(&dir.join("learner-secret"), Some(&root), Some(&manifest))
        .expect("learner secret with ARC paths");
    let eval = eval_setup(&secret);
    let run = dir.join("run");
    let cfg = MetaConfig {
        max_batches: 2,
        ..tiny_meta(9, &secret)
    };

    init_and_run(&run, &cfg, None, true);

    // A finalist if the final phase made one, else a baseline.
    let candidates = open_archive(&run).learner_candidates();
    let candidate = candidates
        .iter()
        .find(|c| c.status == LearnerStatus::Finalist)
        .or_else(|| {
            candidates
                .iter()
                .find(|c| c.status == LearnerStatus::Baseline)
        })
        .cloned()
        .expect("an archived program");

    let resp = arc_check(&run, &candidate.id, EvalSet::A1Eval, &eval).expect("arc check");

    assert_eq!(resp.candidate_id, candidate.id);
    assert_eq!(resp.set, EvalSet::A1Eval);
    assert_eq!(resp.n_tasks, ARC_FIXTURE_TASKS);
    assert_eq!(resp.n_test_outputs, ARC_FIXTURE_TEST_OUTPUTS);
    assert!(resp.solved_outputs <= resp.n_test_outputs);
    assert!((0.0..=1.0).contains(&resp.official));
    assert!(
        (resp.kaggle - resp.solved_outputs as f64 / ARC_FIXTURE_TEST_OUTPUTS as f64).abs() < 1e-9
    );

    if resp.solved_outputs == 0 {
        assert_eq!(resp.official, 0.0);
    }

    if let Some(bca) = resp.bca {
        assert!((0.0..=1.0).contains(&bca));
        assert_step_multiple(bca, 0.01, "bca");
    }

    assert_eq!(resp.queries_left_set, ARC_BUDGET_PER_SET - 1);

    // The archive records the external check.
    let archive = open_archive(&run);
    let external = count_kind(&archive, |k| {
        matches!(
            k,
            RecordKind::ExternalEval { candidate_id, set: EvalSet::A1Eval, official, kaggle, bca }
                if candidate_id == &candidate.id
                    && *official == resp.official
                    && *kaggle == resp.kaggle
                    && *bca == resp.bca
        )
    });

    assert_eq!(external, 1);

    // COR-6: report.json is refreshed with the external check.
    let report: Report =
        serde_json::from_slice(&std::fs::read(run.join(REPORT_FILE)).expect("report.json"))
            .expect("report json");

    assert!(
        report
            .external_arc
            .iter()
            .any(|row| row.id == candidate.id && row.set == EvalSet::A1Eval),
        "{:?}",
        report.external_arc
    );

    // The ledger verifies and holds one Charge and one Release for this query.
    let (protocol, run_id) = run_identity(&run);
    let arc = QueryKind::Arc {
        set: EvalSet::A1Eval,
    };

    Ledger::verify(&secret_dir(&secret).join(LEDGER_FILE)).expect("ledger verifies");

    let ledger = Ledger::open(secret_dir(&secret)).expect("open ledger");

    assert_eq!(ledger.charges(&protocol, Some(&run_id), &arc), 1);

    let entries = ledger_entries(&secret);
    let charged = entries
        .iter()
        .filter(|e| matches!(&e.kind, LedgerKind::Charge { query } if query == &arc))
        .count();
    let released = entries
        .iter()
        .filter(|e| matches!(&e.kind, LedgerKind::Release { query, .. } if query == &arc))
        .count();

    assert_eq!((charged, released), (1, 1));

    // Contamination (AC-10): refused before scoring and without a charge.
    let refused_before = entries
        .iter()
        .filter(|e| matches!(e.kind, LedgerKind::Refused { .. }))
        .count();
    let forbidden = [
        (
            vec![SourceId::Synthetic, SourceId::ArcAgi2Training],
            "arc_agi2_training",
        ),
        (vec![SourceId::ArcGen], "arc_gen"),
        (vec![SourceId::Barc], "barc"),
        (
            vec![SourceId::ArcAgi1Training],
            "planted rotated + recoloured duplicate",
        ),
    ];

    for (sources, why) in forbidden {
        let req = ArcEvalRequest {
            run_id: run_id.clone(),
            candidate_id: candidate.id.clone(),
            protocol_hash: protocol.clone(),
            set: EvalSet::A1Eval,
            program: candidate.genome.clone(),
            train_sources: sources,
        };
        let result = request_arc_eval(
            &eval.eval_bin,
            &eval.secret_env,
            ARC_BUDGET_PER_SET,
            0.01,
            &req,
            None,
        );

        assert!(result.is_err(), "{why}: not refused");

        let ledger = Ledger::open(secret_dir(&secret)).expect("open ledger");

        assert_eq!(
            ledger.charges(&protocol, Some(&run_id), &arc),
            1,
            "{why}: charged"
        );
    }

    let refused_after = ledger_entries(&secret)
        .iter()
        .filter(|e| matches!(e.kind, LedgerKind::Refused { .. }))
        .count();

    // At least the canonical-hash overlap is logged as Refused (contract S5, S7).
    assert!(refused_after > refused_before, "no Refused entry logged");
}

// ---------------------------------------------------------------------------------------
// SEC-3: ARC loader errors stay on the evaluator side
// ---------------------------------------------------------------------------------------

#[test]
fn arc_data_errors_do_not_quote_eval_files_to_the_orchestrator() {
    let dir = fresh_dir("arc-damaged");
    let root = dir.join("arc");
    let manifest = dir.join("manifest.json");

    for (rel, ..) in ARC_FIXTURES {
        let target = root.join(rel);

        std::fs::create_dir_all(target.parent().expect("parent")).expect("mkdir");
        std::fs::copy(crate_path("tests/fixtures/arc").join(rel), &target).expect("copy");
    }

    write_fixture_manifest(&root, &manifest);

    // One byte appended after the manifest was written: the hash check fails.
    let damaged = root.join(ARC_FIXTURES[0].0);
    let mut bytes = std::fs::read(&damaged).expect("read");

    bytes.push(b' ');
    std::fs::write(&damaged, bytes).expect("damage");

    let secret = create_learner_secret(&dir.join("learner-secret"), Some(&root), Some(&manifest))
        .expect("learner secret with ARC paths");
    let req = ArcEvalRequest {
        run_id: "a".repeat(64),
        candidate_id: "0123456789abcdef".into(),
        protocol_hash: protocol_hash(),
        set: EvalSet::A1Eval,
        program: brutmaschine::learner::seeds::reference_nn(),
        train_sources: vec![SourceId::Synthetic],
    };
    let err = request_arc_eval(
        Path::new(EVAL_BIN),
        &secret,
        ARC_BUDGET_PER_SET,
        0.01,
        &req,
        None,
    )
    .expect_err("damaged data is refused");
    let text = err.to_string();

    assert!(!text.contains("data/evaluation"), "{text}");
    assert!(!text.contains("e0000001"), "{text}");
    assert!(text.contains("ARC data failed"), "{text}");

    // Refused, not charged; the ledger keeps the loader's message.
    let refused: Vec<String> = ledger_entries(&secret)
        .into_iter()
        .filter_map(|e| match e.kind {
            LedgerKind::Refused { reason, .. } => Some(reason),
            _ => None,
        })
        .collect();

    assert_eq!(refused.len(), 1);
    assert!(refused[0].contains("e0000001"), "{refused:?}");
    assert!(
        !ledger_entries(&secret)
            .iter()
            .any(|e| matches!(e.kind, LedgerKind::Charge { .. }))
    );
}
