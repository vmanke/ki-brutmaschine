//! End-to-end tests of the ARC grid tasks (ticket 06): the re-arc converter
//! `tools/rearc_to_jsonl.py`, `brut arc-grid` on hand-made fixture tasks, and two tests on
//! the real data that run on request.
//!
//! The fixtures are made here, from seeded random grids; no ARC file is copied. The
//! converter test needs `python3` and skips itself, with a note, when it is missing. It
//! runs the converter outside the sandbox, since the fixture zip is our own data.
#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use brutmaschine::arc_data::{self, DataRole, Manifest, ManifestEntry, REARC_GRIDS};
use brutmaschine::arc_grid::{self, ARC_SIDE, Report};
use brutmaschine::config::Limits;
use brutmaschine::learner::arc_cells::canonical_pair_hash;
use brutmaschine::learner::types::{EvalSet, SourceId};
use brutmaschine::util::sha256_hex;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

const BRUT_BIN: &str = env!("CARGO_BIN_EXE_brut");

/// A commit for the hand-made fixtures, which have no upstream repository.
const FIXTURE_COMMIT: &str = "0000000";

fn fresh_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("arc-grid")
        .join(name);

    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test dir");

    dir
}

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// One input/output pair as the task files and the re-arc lines store it.
#[derive(Clone, Debug, PartialEq, Serialize)]
struct Pair {
    input: Vec<Vec<u8>>,
    output: Vec<Vec<u8>>,
}

/// A task file of ARC-AGI-1.
#[derive(Serialize)]
struct TaskFile {
    train: Vec<Pair>,
    test: Vec<Pair>,
}

/// One line of the converted re-arc file.
#[derive(Serialize)]
struct Line {
    id: String,
    pairs: Vec<Pair>,
}

/// A random grid of `h` x `w` cells with the colours 0 to 4, and its output: the colours
/// `a` and `b` swapped.
fn swap(rng: &mut ChaCha8Rng, h: usize, w: usize, a: u8, b: u8) -> Pair {
    let input: Vec<Vec<u8>> = (0..h)
        .map(|_| (0..w).map(|_| rng.random_range(0..5u8)).collect())
        .collect();
    let output = input
        .iter()
        .map(|row| {
            row.iter()
                .map(|&c| match c {
                    c if c == a => b,
                    c if c == b => a,
                    c => c,
                })
                .collect()
        })
        .collect();

    Pair { input, output }
}

/// `n` swap pairs of sizes from 2 x 2 to 5 x 5.
fn swaps(seed: u64, n: usize, a: u8, b: u8) -> Vec<Pair> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    (0..n)
        .map(|i| swap(&mut rng, 2 + i % 4, 2 + (i / 4) % 4, a, b))
        .collect()
}

/// Write `value` as JSON to `root/rel` and return its manifest entry.
fn put<T: Serialize>(root: &Path, rel: &str, value: &T, source: SourceId) -> ManifestEntry {
    let text = serde_json::to_string(value).expect("json");

    put_text(root, rel, &text, source)
}

fn put_text(root: &Path, rel: &str, text: &str, source: SourceId) -> ManifestEntry {
    let path = root.join(rel);

    std::fs::create_dir_all(path.parent().expect("parent")).expect("dirs");
    std::fs::write(&path, text).expect("write");

    ManifestEntry {
        rel_path: rel.to_string(),
        sha256: sha256_hex(text.as_bytes()),
        license: "CC0-1.0".to_string(),
        source,
        commit: FIXTURE_COMMIT.to_string(),
        role: DataRole::Train,
        set: None,
    }
}

fn write_manifest(path: &Path, entries: Vec<ManifestEntry>) {
    let m = Manifest {
        version: 1,
        entries,
    };

    std::fs::write(path, serde_json::to_vec(&m).expect("json")).expect("manifest");
}

/// A data root with four ARC-AGI-1 training tasks and their re-arc pairs, and its manifest:
/// two swap tasks that become ARC grid tasks, one whose task file changes the grid size, and
/// one of the two tasks that repeat evaluation pairs. The manifest also lists an evaluation
/// file that does not exist, which the grid tasks must never read.
fn fixture_root(name: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(name);
    let resize = TaskFile {
        train: vec![Pair {
            input: vec![vec![1]],
            output: vec![vec![1, 1]],
        }],
        test: vec![Pair {
            input: vec![vec![2]],
            output: vec![vec![2]],
        }],
    };
    let task = |seed, a, b| {
        let pairs = swaps(seed, 4, a, b);

        TaskFile {
            train: pairs[..3].to_vec(),
            test: pairs[3..].to_vec(),
        }
    };
    let mut entries = vec![
        put(
            &root,
            "arc-agi-1/data/training/a0000001.json",
            &task(1, 1, 2),
            SourceId::ArcAgi1Training,
        ),
        put(
            &root,
            "arc-agi-1/data/training/a0000002.json",
            &task(2, 3, 4),
            SourceId::ArcAgi1Training,
        ),
        put(
            &root,
            "arc-agi-1/data/training/a0000003.json",
            &resize,
            SourceId::ArcAgi1Training,
        ),
        put(
            &root,
            "arc-agi-1/data/training/40853293.json",
            &task(3, 1, 3),
            SourceId::ArcAgi1Training,
        ),
    ];
    let lines: Vec<String> = [
        ("a0000001", 11, 1, 2),
        ("a0000002", 12, 3, 4),
        ("a0000003", 13, 1, 2),
        ("40853293", 14, 1, 3),
    ]
    .iter()
    .map(|&(id, seed, a, b)| {
        serde_json::to_string(&Line {
            id: id.to_string(),
            pairs: swaps(seed, 30, a, b),
        })
        .expect("json")
    })
    .collect();

    entries.push(put_text(
        &root,
        REARC_GRIDS,
        &(lines.join("\n") + "\n"),
        SourceId::ReArc,
    ));
    entries.push(ManifestEntry {
        rel_path: "arc-agi-1/data/evaluation/e0000001.json".to_string(),
        sha256: "0".repeat(64),
        license: "CC0-1.0".to_string(),
        source: SourceId::ArcAgi1Eval,
        commit: FIXTURE_COMMIT.to_string(),
        role: DataRole::External,
        set: Some(EvalSet::A1Eval),
    });

    let manifest = root.join("manifest.json");

    write_manifest(&manifest, entries);

    (root, manifest)
}

fn run_brut(args: &[&str]) -> Output {
    Command::new(BRUT_BIN)
        .args(args)
        .output()
        .expect("run brut")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The report with every wall-clock time set to 0, the one field that differs between runs.
fn timeless(report: &Report) -> String {
    let mut report = report.clone();

    for result in &mut report.results {
        for task in &mut result.tasks {
            if let Some(train) = &mut task.train {
                train.wall_seconds = 0.0;
            }
        }
    }

    serde_json::to_string(&report).expect("json")
}

fn read_report(path: &Path) -> Report {
    serde_json::from_slice(&std::fs::read(path).expect("report")).expect("parse report")
}

#[test]
fn brut_arc_grid_trains_the_base_models_and_scores_them_on_held_out_pairs() {
    let (root, manifest) = fixture_root("base-models");
    let out = root.join("report.json");
    let args = [
        "arc-grid",
        "--arc-root",
        root.to_str().expect("utf-8"),
        "--manifest",
        manifest.to_str().expect("utf-8"),
        "--canvas",
        "6",
        "--threads",
        "2",
        "--out",
        out.to_str().expect("utf-8"),
    ];
    let run = run_brut(&args);

    assert!(run.status.success(), "{}", text(&run.stderr));

    let stdout = text(&run.stdout);

    assert!(
        stdout.contains("arc-grid: 2 of 2 ARC grid tasks"),
        "{stdout}"
    );
    assert!(
        stdout.contains("(2 training tasks left out by the rules)"),
        "{stdout}"
    );

    let report = read_report(&out);
    let names: Vec<&str> = report.results.iter().map(|r| r.name.as_str()).collect();
    let left_out: Vec<(&str, &str)> = report
        .left_out
        .iter()
        .map(|l| (l.id.as_str(), l.reason.as_str()))
        .collect();

    assert_eq!(report.canvas, 6);
    assert_eq!(report.tasks, vec!["a0000001", "a0000002"]);
    assert_eq!(names, vec!["copy", "u-net", "fcn"]);
    assert_eq!(
        left_out,
        vec![
            ("40853293", "repeats pairs of an ARC-AGI-1 evaluation task"),
            ("a0000003", "a pair of the task file changes the grid size"),
        ]
    );

    for result in &report.results {
        let s = &result.summary;

        assert_eq!((s.tasks, s.scored, s.refused), (2, 2, 0), "{}", result.name);
        assert_eq!(s.diverged, 0, "{}", result.name);

        for task in &result.tasks {
            let valid = task.valid.expect("valid");
            let test = task.test.expect("test");

            // 30 re-arc pairs: 24 train after the 3 of the task file, 6 validate.
            assert!((0.0..=1.0).contains(&valid.exact) && (0.0..=1.0).contains(&valid.cells));
            assert!((0.0..=1.0).contains(&test.exact) && (0.0..=1.0).contains(&test.cells));

            // Each base model trains for its own epochs, two steps of at most 16 pairs each.
            if let Some(genome) = &result.genome {
                let train = task.train.as_ref().expect("trained");

                assert_eq!(task.epochs, Some(genome.epochs), "{}", result.name);
                assert_eq!(train.epochs_done, genome.epochs);
                assert_eq!(train.steps, genome.epochs as u64 * 27_u64.div_ceil(16));
                assert!(train.final_train_loss.is_finite());
            }
        }

        assert!(stdout.contains(&format!("\n{}", result.name)), "{stdout}");
    }

    // The copy of the input gets a swap grid right only when it lacks both colours; its
    // cells keep 3 of 5 colours.
    let copy = &report.results[0].summary;

    assert!(
        copy.exact_valid <= 0.2 && copy.exact_test <= 0.5,
        "{copy:?}"
    );
    assert!((0.4..0.8).contains(&copy.cells_valid), "{copy:?}");
}

#[test]
fn brut_arc_grid_repeats_its_report_at_1_and_2_threads_and_checks_its_files() {
    let (root, manifest) = fixture_root("repeat");
    let genome_file = root.join("small.json");
    let (_, fcn) = arc_grid::base_models().remove(1);
    let small = brutmaschine::genome::Genome {
        epochs: 20,
        learning_rate: 0.03,
        ..fcn
    };

    std::fs::write(&genome_file, serde_json::to_vec(&small).expect("json")).expect("genome");

    let run = |threads: &str, out: &Path| {
        run_brut(&[
            "arc-grid",
            "--arc-root",
            root.to_str().expect("utf-8"),
            "--manifest",
            manifest.to_str().expect("utf-8"),
            "--genome",
            genome_file.to_str().expect("utf-8"),
            "--canvas",
            "6",
            "--tasks",
            "1",
            "--threads",
            threads,
            "--out",
            out.to_str().expect("utf-8"),
        ])
    };
    let one = root.join("one.json");
    let two = root.join("two.json");

    for (threads, out) in [("1", &one), ("2", &two)] {
        let output = run(threads, out);

        assert!(output.status.success(), "{}", text(&output.stderr));
    }

    let first = read_report(&one);

    assert_eq!(timeless(&first), timeless(&read_report(&two)));
    assert_eq!(first.limits, Limits::default());
    assert_eq!(first.tasks, vec!["a0000001"]);
    assert_eq!(first.results[1].name, "small");
    assert_eq!(first.results[1].id.as_deref(), Some(small.id().as_str()));

    // A changed training file is caught by its hash, and nothing is trained.
    std::fs::write(root.join("arc-agi-1/data/training/a0000002.json"), "{}").expect("write");

    let output = run("1", &root.join("three.json"));

    assert_eq!(output.status.code(), Some(1));
    assert!(text(&output.stderr).contains("does not match its manifest hash"));
    assert!(!root.join("three.json").exists());

    // A file that is not one genome is refused without quoting it.
    std::fs::write(&genome_file, r#"{"zz-private": 1}"#).expect("write");

    let output = run("1", &root.join("four.json"));
    let stderr = text(&output.stderr);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr.contains("not the JSON of one genome"), "{stderr}");
    assert!(!stderr.contains("zz-private"), "{stderr}");
}

#[test]
fn brut_arc_grid_takes_the_limits_of_a_file_and_refuses_bad_ones() {
    let (root, manifest) = fixture_root("limits");
    let limits_file = root.join("limits.json");
    let run = |out: &Path| {
        run_brut(&[
            "arc-grid",
            "--arc-root",
            root.to_str().expect("utf-8"),
            "--manifest",
            manifest.to_str().expect("utf-8"),
            "--canvas",
            "6",
            "--tasks",
            "1",
            "--limits",
            limits_file.to_str().expect("utf-8"),
            "--out",
            out.to_str().expect("utf-8"),
        ])
    };
    let larger = Limits {
        max_train_flops: 40_000_000_000,
        max_candidate_seconds: 60.0,
        ..Limits::default()
    };

    std::fs::write(&limits_file, serde_json::to_vec(&larger).expect("json")).expect("write");

    let out = root.join("larger.json");
    let output = run(&out);

    assert!(output.status.success(), "{}", text(&output.stderr));
    assert_eq!(read_report(&out).limits, larger);

    // An unknown field, and limits that a config would refuse, are refused before training.
    let mut unknown = serde_json::to_value(larger).expect("json");

    unknown["zz-extra"] = serde_json::json!(1);

    let tight = Limits {
        max_epochs: 1,
        ..Limits::default()
    };

    for (what, bytes) in [
        ("unknown field", serde_json::to_vec(&unknown).expect("json")),
        ("max_epochs 1", serde_json::to_vec(&tight).expect("json")),
    ] {
        std::fs::write(&limits_file, bytes).expect("write");

        let out = root.join("refused.json");
        let output = run(&out);

        assert_eq!(output.status.code(), Some(1), "{what}");
        assert!(text(&output.stderr).contains("limits"), "{what}");
        assert!(!out.exists(), "{what}");
    }
}

/// The Python script that writes a zip of the files in a directory: each file's path below
/// the directory is its member name, with a "^" for "/" so that the test can name members
/// with ".." or a leading "/". Our own code, not downloaded.
const MAKE_ZIP: &str = r#"
import os, sys, zipfile
src, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for name in sorted(os.listdir(src)):
        info = zipfile.ZipInfo(name.replace("^", "/"), date_time=(2024, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, open(os.path.join(src, name), "rb").read())
"#;

/// Write a zip of `members` (name, content) to `out` with python3; false when python3 is
/// missing.
fn make_zip(dir: &Path, members: &[(&str, String)], out: &Path) -> bool {
    let staging = dir.join("members");

    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).expect("staging");

    for (name, content) in members {
        std::fs::write(staging.join(name.replace('/', "^")), content).expect("member");
    }

    match Command::new("python3")
        .args(["-I", "-c", MAKE_ZIP])
        .arg(&staging)
        .arg(out)
        .status()
    {
        Ok(status) => {
            assert!(status.success(), "python3 could not write the zip");
            true
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(e) => panic!("python3: {e}"),
    }
}

/// Run the converter on `zip` with these options; returns its output.
fn convert(zip: &Path, options: &[&str]) -> Output {
    let stdin = std::fs::File::open(zip).expect("zip");

    Command::new("python3")
        .arg("-I")
        .arg(crate_path("tools/rearc_to_jsonl.py"))
        .args(options)
        .stdin(stdin)
        .output()
        .expect("run the converter")
}

#[test]
fn the_rearc_converter_writes_the_lines_that_the_loader_reads() {
    let dir = fresh_dir("converter");
    let good = swaps(21, 3, 1, 2);
    let big = Pair {
        input: vec![vec![1; 30]; 30],
        output: vec![vec![2; 30]; 30],
    };
    let json = |value: &serde_json::Value| value.to_string();
    let pair = |p: &Pair| serde_json::to_value(p).expect("json");
    // The examples of task 00000001: two good ones, then one of each kind that is skipped,
    // then the large one, then a third good one. 00000002 keeps nothing.
    let first = serde_json::Value::Array(vec![
        pair(&good[0]),
        pair(&good[1]),
        serde_json::json!({"input": vec![vec![1; 31]], "output": vec![vec![1; 31]]}),
        serde_json::json!({"input": [[10]], "output": [[1]]}),
        serde_json::json!({"input": [[1, 2]], "output": [[1], [2]]}),
        serde_json::json!({"input": [[true]], "output": [[1]]}),
        serde_json::json!({"input": [[1]], "output": [[1]], "note": 1}),
        serde_json::json!({"input": [[1, 2], [3]], "output": [[1, 2], [3]]}),
        pair(&big),
        pair(&good[2]),
    ]);
    let second = serde_json::json!([{"input": [[1]], "output": [[1, 1]]}]);
    let members = [
        ("re_arc/tasks/00000001.json", json(&first)),
        ("re_arc/tasks/00000002.json", json(&second)),
        ("re_arc/metadata.json", "{}".to_string()),
        ("re_arc/tasks/notahex1.json", "[]".to_string()),
        ("__MACOSX/re_arc/._x", "binary".to_string()),
    ];
    let zip = dir.join("re_arc.zip");

    if !make_zip(&dir, &members, &zip) {
        eprintln!("skipped: python3 is missing");
        return;
    }

    let sha = sha256_hex(&std::fs::read(&zip).expect("zip"));
    let output = convert(&zip, &["--zip-sha256", &sha]);
    let stderr = text(&output.stderr);

    assert!(output.status.success(), "{stderr}");

    // One line for 00000001: the good pairs and the large one, in their order.
    let expected = serde_json::to_string(&Line {
        id: "00000001".to_string(),
        pairs: vec![
            good[0].clone(),
            good[1].clone(),
            big.clone(),
            good[2].clone(),
        ],
    })
    .expect("json")
        + "\n";

    assert_eq!(text(&output.stdout), expected);
    assert!(stderr.contains("read 2 task files"), "{stderr}");
    assert!(stderr.contains("wrote 1 tasks with 4 pairs"), "{stderr}");
    assert!(
        stderr.contains("skipped 4 examples: a grid breaks the ARC limits"),
        "{stderr}"
    );
    assert!(
        stderr.contains("skipped 2 examples: changes the grid size"),
        "{stderr}"
    );
    assert!(
        stderr.contains("skipped 1 examples: not a pair of input and output"),
        "{stderr}"
    );
    assert!(
        stderr.contains(&format!(
            "output sha256 {}",
            sha256_hex(expected.as_bytes())
        )),
        "{stderr}"
    );

    // The loader reads the output as the re-arc file of the grid tasks.
    let root = dir.join("root");
    let entry = put_text(&root, REARC_GRIDS, &expected, SourceId::ReArc);
    let m = Manifest {
        version: 1,
        entries: vec![entry],
    };
    let lines = arc_data::load_rearc_grids(&root, &m).expect("load");

    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].1.len(), 4);

    // The cell families of E2: grids of at most 225 cells, here also only 2 pairs.
    let output = convert(
        &zip,
        &[
            "--zip-sha256",
            &sha,
            "--max-area",
            "225",
            "--max-pairs",
            "2",
        ],
    );
    let e2 = serde_json::to_string(&Line {
        id: "00000001".to_string(),
        pairs: good[..2].to_vec(),
    })
    .expect("json")
        + "\n";

    assert!(output.status.success());
    assert_eq!(text(&output.stdout), e2);

    // Refused: another zip than the pinned one, a pipe, bad options, and a member path
    // with a ".." part or a leading "/".
    let pinned = convert(&zip, &[]);

    assert_eq!(pinned.status.code(), Some(1));
    assert!(text(&pinned.stderr).contains("expected SHA-256"));

    let mut piped = Command::new("python3")
        .arg("-I")
        .arg(crate_path("tools/rearc_to_jsonl.py"))
        .args(["--zip-sha256", &sha])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");

    if let Some(mut stdin) = piped.stdin.take() {
        let _ = stdin.write_all(&std::fs::read(&zip).expect("zip"));
    }

    let piped = piped.wait_with_output().expect("wait");

    assert_eq!(piped.status.code(), Some(1));
    assert!(text(&piped.stderr).contains("not a pipe"));

    for bad in [["--max-area", "901"], ["--max-pairs", "61"]] {
        assert_eq!(convert(&zip, &bad).status.code(), Some(2), "{bad:?}");
    }

    for name in ["re_arc/../evil.json", "/re_arc/tasks/00000003.json"] {
        let evil = dir.join("evil.zip");

        assert!(make_zip(&dir, &[(name, "[]".to_string())], &evil));

        let sha = sha256_hex(&std::fs::read(&evil).expect("zip"));
        let output = convert(&evil, &["--zip-sha256", &sha]);

        assert_eq!(output.status.code(), Some(1), "{name}");
        assert!(text(&output.stderr).contains("'..' part"), "{name}");
        assert!(output.stdout.is_empty(), "{name}");
    }
}

/// The real data: BRUT_ARC_TEST_ROOT, with the manifest at `<root>/manifest.json`.
fn real_data() -> Option<(PathBuf, PathBuf)> {
    let root = PathBuf::from(std::env::var("BRUT_ARC_TEST_ROOT").ok()?);
    let manifest = root.join("manifest.json");

    (root.is_dir() && manifest.is_file()).then_some((root, manifest))
}

/// The pairs that the ARC grid tasks train, validate and test on share no canonical hash
/// (all 8 rotations and reflections, colours renumbered) with any pair of the three
/// evaluation sets. Without the two tasks that repeat evaluation pairs, nothing touches ARC-
/// AGI-1 evaluation; those two touch it. Run with
/// `BRUT_ARC_TEST_ROOT=<data root> cargo test --test arc_grid -- --ignored real_`.
#[test]
#[ignore = "needs the real ARC data (BRUT_ARC_TEST_ROOT, manifest at <root>/manifest.json)"]
fn real_arc_grid_tasks_touch_no_evaluation_pair() {
    let Some((root, manifest)) = real_data() else {
        println!("skipped: no ARC data root with manifest.json found");

        return;
    };

    let (tasks, left_out) = arc_grid::load(&root, &manifest, ARC_SIDE).expect("tasks");
    let pairs: Vec<_> = tasks
        .iter()
        .flat_map(|t| t.train.iter().chain(&t.valid).chain(&t.test))
        .collect();
    let used: BTreeSet<[u8; 32]> = pairs.iter().map(|p| canonical_pair_hash(p)).collect();
    let m = arc_data::load_manifest(&manifest).expect("manifest");
    let training = arc_data::load_training_set(&root, &m).expect("training");
    let dropped: BTreeSet<[u8; 32]> = training
        .iter()
        .filter(|t| arc_data::A1_TRAIN_REPEATS_EVAL.contains(&t.id.as_str()))
        .flat_map(|t| t.train.iter().chain(&t.test))
        .map(canonical_pair_hash)
        .collect();
    let (train, valid, test) = tasks.iter().fold((0, 0, 0), |(a, b, c), t| {
        (a + t.train.len(), b + t.valid.len(), c + t.test.len())
    });

    println!(
        "{} ARC grid tasks, {} left out; pairs: {train} train, {valid} valid, {test} test",
        tasks.len(),
        left_out.len()
    );

    assert!(tasks.len() > 200, "{}", tasks.len());
    assert!(
        left_out
            .iter()
            .any(|l| l.id == arc_data::A1_TRAIN_REPEATS_EVAL[0])
    );

    for set in [EvalSet::A1Eval, EvalSet::A2Eval, EvalSet::ConceptArc] {
        let eval = arc_data::load_eval_set(&root, &m, set).expect("eval set");
        let touched = |hashes: &BTreeSet<[u8; 32]>| -> Vec<String> {
            eval.iter()
                .filter(|t| {
                    t.train
                        .iter()
                        .chain(&t.test)
                        .any(|p| hashes.contains(&canonical_pair_hash(p)))
                })
                .map(|t| t.id.clone())
                .collect()
        };
        let by_used = touched(&used);
        let by_dropped = touched(&dropped);

        println!(
            "{set:?}: touched by the grid tasks {by_used:?}, by the two left out {by_dropped:?}"
        );

        assert!(by_used.is_empty(), "{set:?}: {by_used:?}");

        if set == EvalSet::A1Eval {
            assert_eq!(by_dropped, vec!["070dd51e", "50a16a69"]);
        }
    }
}

/// The two base models train on the first four ARC grid tasks of the real data and are
/// scored on their held-out pairs, without refusal, divergence or a stop by the clock. Run
/// as `real_arc_grid_tasks_touch_no_evaluation_pair`; it takes about two minutes.
#[test]
#[ignore = "needs the real ARC data (BRUT_ARC_TEST_ROOT, manifest at <root>/manifest.json)"]
fn real_arc_grid_base_models_train_and_score_on_four_tasks() {
    let Some((root, manifest)) = real_data() else {
        println!("skipped: no ARC data root with manifest.json found");

        return;
    };

    let (mut tasks, left_out) = arc_grid::load(&root, &manifest, ARC_SIDE).expect("tasks");

    tasks.truncate(4);

    let genomes: Vec<(String, brutmaschine::genome::Genome)> = arc_grid::base_models()
        .into_iter()
        .map(|(name, genome)| (name.to_string(), genome))
        .collect();
    let report = arc_grid::report(&genomes, &tasks, left_out, ARC_SIDE, &Limits::default());

    for result in &report.results {
        println!("{}: {:?}", result.name, result.summary);

        let s = &result.summary;

        assert_eq!((s.tasks, s.scored), (4, 4), "{}", result.name);
        assert_eq!(s.diverged + s.stopped_by_budget, 0, "{}", result.name);

        for value in [s.exact_valid, s.exact_test, s.cells_valid, s.cells_test] {
            assert!((0.0..=1.0).contains(&value), "{}", result.name);
        }
    }
}
