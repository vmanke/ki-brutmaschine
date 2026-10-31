//! Security and hygiene tests of learner mode (contract sections 1, 9 and 10: T-SEC-1 .. T-SEC-4).
//!
//! The grep tests read the sources at run time, so new files under `src/learner/` are
//! covered without editing this file. They are hygiene checks, not proofs: a determined
//! author can always hide a name from a text search.
#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use brutmaschine::genome::Activation;
use brutmaschine::hidden_eval::{EvalRequest, EvalState, STATE_FILE, create_secret};
use brutmaschine::ledger::{LedgerEntry, LedgerKind};
use brutmaschine::nn::Mlp;
use brutmaschine::task::TaskKind;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const EVAL_BIN: &str = env!("CARGO_BIN_EXE_brut-eval");

/// Orchestrator files (contract S1). They never touch secrets or hidden data.
const ORCHESTRATOR_FILES: [&str; 5] = [
    "src/meta_evolve.rs",
    "src/meta_report.rs",
    "src/lineage_store.rs",
    "src/main.rs",
    "src/evolve.rs",
];

/// Names the orchestrator files must never mention (contract S1).
const SECRET_NAMES: [&str; 13] = [
    "HiddenEvaluator",
    "LearnerEvaluator",
    "parse_env_file",
    "HIDDEN_SEED_KEY",
    "LEARNER_SEED_KEY",
    "HiddenToken",
    "hidden_tier_tasks",
    "hidden_grammar_pool",
    "BRUT_ARC_DATA_ROOT",
    "BRUT_ARC_MANIFEST",
    "data/evaluation",
    "arc-agi-2",
    "ConceptARC/corpus",
];

/// Strings that `src/learner/**` must never mention (contract I3), checked verbatim.
const IO_MARKERS: [&str; 12] = [
    "std::fs",
    "std::net",
    "std::process",
    "File::",
    "Command",
    "TcpStream",
    "env::var",
    "include_bytes",
    "include_str",
    "unsafe {",
    "unsafe fn",
    "unsafe impl",
];

/// Markers searched with all whitespace removed, plus `std::env` (extension of I3).
/// Catches spellings like `std :: fs` or `unsafe{`; a hit may repeat a verbatim one.
const IO_MARKERS_COMPACT: [&str; 7] = [
    "std::fs",
    "std::net",
    "std::process",
    "std::env",
    "unsafe{",
    "unsafefn",
    "unsafeimpl",
];

/// Modules that `use std::{..}` groups in `src/learner/**` must not import.
const IO_MODULES: [&str; 4] = ["fs", "net", "process", "env"];

/// Files created for learner mode (contract section 3). Each must start with the forbid line.
const NEW_TOP_LEVEL_FILES: [&str; 8] = [
    "src/arc_data.rs",
    "src/ledger.rs",
    "src/learner_hidden.rs",
    "src/lineage_store.rs",
    "src/meta_evolve.rs",
    "src/meta_report.rs",
    "tests/learner_e2e.rs",
    "tests/learner_security.rs",
];

/// Markers that must be gone from library code before acceptance (contract I5, errata E14).
const STUB_MARKERS: [&str; 4] = ["todo!(", "unimplemented!(", ".unwrap()", "WP0 stub"];

/// The only allowed `[dependencies]` (contract I1; ADR 0007 added candle's two crates).
const ALLOWED_DEPENDENCIES: [&str; 9] = [
    "candle-core",
    "candle-nn",
    "hex",
    "rand",
    "rand_chacha",
    "rayon",
    "serde",
    "serde_json",
    "sha2",
];

/// candle's crates, admitted for the CNN models at one reviewed commit (ADR 0007).
const CANDLE_CRATES: [&str; 2] = ["candle-core", "candle-nn"];

/// The source each candle crate must name in `Cargo.toml`, with default features only.
const CANDLE_SOURCE: &str = r#"git = "https://github.com/huggingface/candle", rev = "9515d410e3af803fd45efebfef8137f47d85a63d""#;

const FORBID_LINE: &str = "#![forbid(unsafe_code)]";

/// Absolute path of a file or directory in the crate.
fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// Text of a crate file. Panics with the path when it cannot be read.
fn read_text(rel: &str) -> String {
    let path = crate_path(rel);

    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// All `.rs` files below `dir` (relative to the crate root), as sorted relative paths.
fn rust_files(dir: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut pending = vec![crate_path(dir)];
    let root = crate_path("");

    while let Some(d) = pending.pop() {
        let entries =
            std::fs::read_dir(&d).unwrap_or_else(|e| panic!("read dir {}: {e}", d.display()));

        for entry in entries {
            let path = entry.expect("dir entry").path();

            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let rel = path.strip_prefix(&root).expect("inside crate");

                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }

    out.sort();

    out
}

/// The first line that is neither blank nor a `//` comment (doc comments included).
fn first_code_line(text: &str) -> Option<&str> {
    text.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with("//"))
}

/// The part of a source file before its first `#[cfg(test)]` line (the library code).
fn library_part(text: &str) -> String {
    text.lines()
        .take_while(|l| !l.trim_start().starts_with("#[cfg(test)]"))
        .map(|l| format!("{l}\n"))
        .collect()
}

/// First path segment of one item in a `use` group, e.g. "fs" for "fs::File".
fn first_segment(item: &str) -> String {
    item.split("::").next().unwrap_or("").to_string()
}

/// Top-level modules named inside `use std::{ .. }` groups (whitespace already removed).
/// Nested groups such as `std::{collections::{A, B}, fs}` are handled.
fn grouped_std_imports(compact: &str) -> Vec<String> {
    let mut out = Vec::new();

    for (start, marker) in compact.match_indices("std::{") {
        let body = &compact[start + marker.len()..];
        let mut depth = 0usize;
        let mut item = String::new();

        for c in body.chars() {
            match c {
                '{' => {
                    depth += 1;
                    item.push(c);
                }
                '}' if depth == 0 => break,
                '}' => {
                    depth -= 1;
                    item.push(c);
                }
                ',' if depth == 0 => {
                    out.push(first_segment(&item));
                    item.clear();
                }
                _ => item.push(c),
            }
        }

        out.push(first_segment(&item));
    }

    out
}

/// Fail with every violation listed, not only the first.
fn assert_no_violations(what: &str, violations: &[String]) {
    assert!(
        violations.is_empty(),
        "{what}: {} violation(s):\n  {}",
        violations.len(),
        violations.join("\n  ")
    );
}

#[test]
fn t_sec_1_orchestrator_files_never_name_secret_identifiers() {
    let mut violations = Vec::new();

    for file in ORCHESTRATOR_FILES {
        let text = read_text(file);

        for name in SECRET_NAMES {
            if text.contains(name) {
                violations.push(format!("{file} mentions {name}"));
            }
        }
    }

    assert_no_violations("S1", &violations);
}

#[test]
fn t_sec_2_learner_modules_do_no_io_and_forbid_unsafe() {
    let files = rust_files("src/learner");

    // The contract lists 14 modules; more are fine, fewer means the walk is broken.
    assert!(files.len() >= 14, "only {} files found", files.len());

    let mut violations = Vec::new();

    for file in &files {
        let text = read_text(file);
        let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();

        if !text.contains(FORBID_LINE) {
            violations.push(format!("{file} lacks {FORBID_LINE}"));
        }

        for marker in IO_MARKERS {
            if text.contains(marker) {
                violations.push(format!("{file} mentions {marker:?}"));
            }
        }

        for marker in IO_MARKERS_COMPACT {
            if compact.contains(marker) {
                violations.push(format!("{file} mentions {marker:?} (whitespace ignored)"));
            }
        }

        for module in grouped_std_imports(&compact) {
            if IO_MODULES.contains(&module.as_str()) {
                violations.push(format!("{file} imports std::{{{module}}}"));
            }
        }
    }

    assert_no_violations("S2 / I3", &violations);
}

#[test]
fn t_sec_3_dependencies_unchanged() {
    let toml = read_text("Cargo.toml");
    let mut section = String::new();
    let mut deps = BTreeSet::new();
    let mut violations = Vec::new();

    for raw in toml.lines() {
        let line = raw.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if line.starts_with('[') && line.ends_with(']') {
            section = line
                .trim_matches(|c| c == '[' || c == ']')
                .trim()
                .to_string();

            if let Some(name) = section.strip_prefix("dependencies.") {
                deps.insert(name.trim_matches('"').to_string());
            } else if section != "dependencies" && section.contains("dependencies") {
                violations.push(format!("extra dependency table [{section}]"));
            }

            if section.starts_with("patch") || section == "replace" {
                violations.push(format!("source override table [{section}]"));
            }

            continue;
        }

        if section != "dependencies" {
            continue;
        }

        let Some((key, _)) = line.split_once('=') else {
            continue;
        };

        let key = key.trim().trim_matches('"');
        let name = key.split('.').next().unwrap_or(key);
        let is_name = !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');

        if is_name {
            deps.insert(name.to_string());
        }
    }

    let allowed: BTreeSet<String> = ALLOWED_DEPENDENCIES.iter().map(|s| s.to_string()).collect();

    if deps != allowed {
        violations.push(format!("[dependencies] are {deps:?}, allowed {allowed:?}"));
    }

    // A candle crate comes from the reviewed commit only, with no branch, tag, path or features.
    for name in CANDLE_CRATES {
        let prefix = format!("{name} =");
        let entry = toml
            .lines()
            .map(str::trim)
            .find(|line| line.starts_with(&prefix));
        let pinned = entry.is_some_and(|line| {
            line.contains(CANDLE_SOURCE)
                && !["branch", "tag", "path", "features"]
                    .iter()
                    .any(|key| line.contains(key))
        });

        if !pinned {
            violations.push(format!("{name} must name exactly {CANDLE_SOURCE}"));
        }
    }

    assert_no_violations("I1", &violations);
}

#[test]
fn i2_new_files_start_with_forbid_unsafe_code() {
    let mut files: Vec<String> = NEW_TOP_LEVEL_FILES.iter().map(|s| s.to_string()).collect();

    files.extend(rust_files("src/learner"));

    let mut violations = Vec::new();

    for file in &files {
        let text = read_text(file);

        if first_code_line(&text) != Some(FORBID_LINE) {
            violations.push(format!("{file}: first code line is not {FORBID_LINE}"));
        }
    }

    assert_no_violations("I2", &violations);
}

/// Acceptance gate of errata E14: fails while WP0 stubs remain (expected before integration).
#[test]
fn i5_library_code_has_no_stubs_todo_or_unwrap() {
    let mut violations = Vec::new();

    for file in rust_files("src") {
        let library = library_part(&read_text(&file));

        for (n, line) in library.lines().enumerate() {
            for marker in STUB_MARKERS {
                if line.contains(marker) {
                    violations.push(format!("{file}:{}: {}", n + 1, line.trim()));
                }
            }
        }
    }

    assert_no_violations("I5 / E14", &violations);
}

#[test]
fn hygiene_gitignore_keeps_data_and_secrets_out_of_git() {
    let text = read_text(".gitignore");
    let lines: BTreeSet<&str> = text.lines().map(str::trim).collect();

    // `/data/*` keeps ARC data out; the negation keeps WP3's example manifest tracked.
    let patterns = [
        "/data/*",
        "!/data/manifest.example.json",
        "secret/",
        "runs/",
        ".env",
        "*.env",
        "target/",
    ];

    for pattern in patterns {
        assert!(lines.contains(pattern), ".gitignore lacks {pattern}");
    }
}

#[test]
fn hygiene_env_example_documents_learner_keys_with_placeholders_only() {
    let text = read_text(".env.example");
    let keys = [
        "BRUT_LEARNER_HIDDEN_SEED",
        "BRUT_LEARNER_BUDGET_TOTAL",
        "BRUT_LEARNER_BUDGET_PER_RUN",
        "BRUT_ARC_DATA_ROOT",
        "BRUT_ARC_MANIFEST",
        "BRUT_ARC_BUDGET_PER_SET",
    ];

    for key in keys {
        assert!(
            text.contains(&format!("{key}=")),
            ".env.example lacks {key}"
        );
    }

    // Every assignment, active or commented out, carries a <placeholder>, never a value.
    for raw in text.lines() {
        let line = raw.trim().trim_start_matches('#').trim();

        let Some((key, value)) = line.split_once('=') else {
            continue;
        };

        let is_key = key.starts_with("BRUT_")
            && key
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');

        if is_key {
            assert!(
                value.starts_with('<') && value.ends_with('>'),
                ".env.example: {key} has a value instead of a <placeholder>"
            );
        }
    }
}

/// A fresh, empty directory for one test.
fn fresh_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("learner-security")
        .join(name);

    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test dir");

    dir
}

/// Write a secret file by hand: directory 0700, file 0600 (format of contract section 7).
fn write_private_secret(dir: &Path, text: &str) -> PathBuf {
    std::fs::create_dir_all(dir).expect("secret dir");

    let path = dir.join("hidden.env");
    let mut options = std::fs::OpenOptions::new();

    options.write(true).create_new(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).expect("chmod");
        options.mode(0o600);
    }

    let mut file = options.open(&path).expect("create secret");

    file.write_all(text.as_bytes()).expect("write secret");

    path
}

/// Run brut-eval once with an empty environment and `body` on stdin.
fn run_eval(secret: &Path, body: Vec<u8>) -> Output {
    let mut child = Command::new(EVAL_BIN)
        .arg("--secret")
        .arg(secret)
        .args(["--budget", "12", "--step", "0.01"])
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn brut-eval");

    let mut stdin = child.stdin.take().expect("stdin");

    // brut-eval stops reading at its cap, so a broken pipe here is expected.
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&body);
    });

    let out = child.wait_with_output().expect("wait for brut-eval");

    let _ = writer.join();

    out
}

/// A syntactically valid JSON request of about `bytes` bytes, padded in a string field.
fn padded_request(prefix: &str, suffix: &str, bytes: usize) -> Vec<u8> {
    let pad = "x".repeat(bytes);

    format!("{prefix}{pad}{suffix}").into_bytes()
}

fn queries_used(secret_dir: &Path) -> usize {
    match std::fs::read_to_string(secret_dir.join(STATE_FILE)) {
        Ok(text) => {
            let state: EvalState = serde_json::from_str(&text).expect("state json");

            state.queries_used
        }
        Err(_) => 0,
    }
}

#[test]
fn t_sec_4_oversize_legacy_request_is_refused_without_charge() {
    let dir = fresh_dir("legacy-oversize");
    let secret = dir.join("secret").join("hidden.env");

    create_secret(&secret).expect("create secret");

    // One valid query first, so a missing charge is visible as an unchanged count.
    let spec = TaskKind::TwoSpirals.spec();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let model = Mlp::new(&[spec.n_in, 4, spec.n_out], Activation::Relu, 1.0, &mut rng);
    let models = dir.join("models");

    std::fs::create_dir_all(&models).expect("models dir");

    let model_path = models.join("m.json");

    std::fs::write(&model_path, serde_json::to_vec(&model).expect("json")).expect("write model");

    let req = EvalRequest {
        candidate_id: "m".into(),
        model_path: model_path.clone(),
        task: TaskKind::TwoSpirals,
    };
    let ok = run_eval(&secret, serde_json::to_vec(&req).expect("json"));

    assert_eq!(
        ok.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&ok.stderr)
    );
    assert_eq!(queries_used(secret.parent().expect("dir")), 1);

    // Now 2 MiB: refused before parsing, no charge, no echo of the content.
    let path_json = serde_json::to_string(&model_path).expect("json");
    let prefix = r#"{"candidate_id":""#;
    let suffix = format!(r#"","model_path":{path_json},"task":"TwoSpirals"}}"#);
    let big = run_eval(&secret, padded_request(prefix, &suffix, 2 << 20));
    let stderr = String::from_utf8_lossy(&big.stderr);

    assert_eq!(big.status.code(), Some(3), "{stderr}");
    assert!(big.stdout.is_empty());
    assert!(!stderr.contains("xxxxxxxx"), "stderr quotes the request");
    assert_eq!(queries_used(secret.parent().expect("dir")), 1);
}

#[test]
fn t_sec_4_oversize_learner_request_is_refused_without_charge() {
    let dir = fresh_dir("learner-oversize");
    let secret_dir = dir.join("learner-secret");

    // Placeholder values for a throwaway test secret; never a real one.
    let secret = write_private_secret(
        &secret_dir,
        "BRUT_LEARNER_HIDDEN_SEED=12345\n\
         BRUT_LEARNER_BUDGET_TOTAL=48\n\
         BRUT_LEARNER_BUDGET_PER_RUN=12\n",
    );

    let prefix = r#"{"kind":"learner","run_id":"r","candidate_id":""#;
    let suffix = r#"","protocol_hash":"p","tier":"hidden_grammar","program":{"setup":[],"predict":[],"learn":[]}}"#;
    let out = run_eval(&secret, padded_request(prefix, suffix, 2 << 20));
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert_eq!(out.status.code(), Some(3), "{stderr}");
    assert!(out.stdout.is_empty());
    assert!(!stderr.contains("xxxxxxxx"), "stderr quotes the request");

    // No charge: either no ledger at all, or one without Charge entries.
    let ledger = secret_dir.join("ledger.jsonl");

    if let Ok(text) = std::fs::read_to_string(&ledger) {
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let entry: LedgerEntry = serde_json::from_str(line).expect("ledger entry");

            assert!(
                !matches!(entry.kind, LedgerKind::Charge { .. }),
                "oversize request was charged"
            );
        }
    }
}
