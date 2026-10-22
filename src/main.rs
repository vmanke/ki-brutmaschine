//! `brut`: command-line front end of the Brutmaschine.
//!
//! Commands:
//!   brut init <run_dir> [--task spirals|checkerboard|parity|sine] [--seed N]
//!   brut init <run_dir> --mode learner [--seed N] [--flops X] [--variation V] [--secret F] [--arc-cells]
//!   brut run <run_dir> [--no-hidden]
//!   brut status <run_dir>
//!   brut lineage <run_dir> <candidate_id>
//!   brut verify <run_dir>
//!   brut approve <run_dir> <candidate_id> --by <name> [--note <text>]
//!   brut stop <run_dir>
//!   brut learner secret-init [--dir D] [--arc-root R] [--manifest M]
//!   brut learner show <run_dir> <id>
//!   brut learner report <run_dir>
//!   brut learner arc-check <run_dir> <id> --set a1-eval|a2-eval|concept-arc
//!   brut help
//!
//! `run`, `status`, `lineage`, `verify` and `approve` work on both run kinds: a directory
//! with `meta.json` is a learner run, one with `config.json` an MLP run.
//!
//! Exit codes: 0 ok, 1 runtime error, 2 usage error.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use brutmaschine::archive::{Archive, CandidateStatus, LearnerStatus, RecordKind};
use brutmaschine::config::Config;
use brutmaschine::evolve::{self, ARCHIVE_FILE, CONFIG_FILE, EvalSetup};
use brutmaschine::hidden_eval::create_secret;
use brutmaschine::learner::types::{EvalSet, MetaConfig, SUPPORTED_F, Tier, VariationMode};
use brutmaschine::learner_hidden::create_learner_secret;
use brutmaschine::lineage_store::read_snapshot;
use brutmaschine::meta_evolve::{
    self, LearnerEvalSetup, META_CONFIG_FILE, MetaRunSummary, default_secret_dir,
};
use brutmaschine::meta_report;
use brutmaschine::safety;
use brutmaschine::task::TaskKind;
use brutmaschine::util::now_unix_ms;

const USAGE: &str = "\
brut - Brutmaschine, a safe prototype of an AI model incubator

usage:
  brut init <run_dir> [--task spirals|checkerboard|parity|sine] [--seed N]
  brut init <run_dir> --mode learner [--seed N] [--flops X]
            [--variation full|constants-only|random-search] [--secret <file>] [--arc-cells]
  brut run <run_dir> [--no-hidden]
  brut status <run_dir>
  brut lineage <run_dir> <candidate_id>
  brut verify <run_dir>
  brut approve <run_dir> <candidate_id> --by <name> [--note <text>]
  brut stop <run_dir>
  brut learner secret-init [--dir <dir>] [--arc-root <dir>] [--manifest <file>]
  brut learner show <run_dir> <id>
  brut learner report <run_dir>
  brut learner arc-check <run_dir> <id> --set a1-eval|a2-eval|concept-arc
  brut help

learner mode: --secret defaults to $HOME/.brut/learner/hidden.env (create it once with
brut learner secret-init); run, status, lineage, verify and approve detect the run kind

exit codes: 0 ok, 1 runtime error, 2 usage error";

/// Relative path of the secret file inside a run directory.
const SECRET_FILE: &str = "secret/hidden.env";

enum CliError {
    Usage(String),
    Runtime(String),
}

impl From<brutmaschine::util::BrutError> for CliError {
    fn from(e: brutmaschine::util::BrutError) -> Self {
        CliError::Runtime(e.to_string())
    }
}

type CliResult = Result<(), CliError>;

/// Split arguments into positionals and `--flag [value]` options. Unknown flags are usage errors.
struct Parsed {
    positional: Vec<String>,
    options: BTreeMap<String, Option<String>>,
}

/// `allowed`: (flag, takes_value).
fn parse(args: &[String], allowed: &[(&str, bool)]) -> Result<Parsed, CliError> {
    let mut positional = Vec::new();
    let mut options = BTreeMap::new();
    let mut i = 0;

    while i < args.len() {
        let arg = &args[i];

        if !arg.starts_with("--") {
            positional.push(arg.clone());
            i += 1;

            continue;
        }

        let Some(&(_, takes_value)) = allowed.iter().find(|(name, _)| *name == arg) else {
            return Err(CliError::Usage(format!("unknown option {arg}")));
        };

        if options.contains_key(arg) {
            return Err(CliError::Usage(format!("option {arg} given twice")));
        }

        if takes_value {
            let Some(value) = args.get(i + 1) else {
                return Err(CliError::Usage(format!("option {arg} needs a value")));
            };

            options.insert(arg.clone(), Some(value.clone()));
            i += 2;
        } else {
            options.insert(arg.clone(), None);
            i += 1;
        }
    }

    Ok(Parsed {
        positional,
        options,
    })
}

impl Parsed {
    fn exactly(&self, n: usize, what: &str) -> Result<(), CliError> {
        if self.positional.len() == n {
            Ok(())
        } else {
            Err(CliError::Usage(format!("expected {what}")))
        }
    }

    fn value(&self, flag: &str) -> Option<&str> {
        self.options.get(flag).and_then(|v| v.as_deref())
    }

    fn flag(&self, flag: &str) -> bool {
        self.options.contains_key(flag)
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    match dispatch(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Usage(m)) => {
            eprintln!("brut: {m}\n\n{USAGE}");

            ExitCode::from(2)
        }
        Err(CliError::Runtime(m)) => {
            eprintln!("brut: error: {m}");

            ExitCode::from(1)
        }
    }
}

fn dispatch(args: &[String]) -> CliResult {
    let Some((command, rest)) = args.split_first() else {
        return Err(CliError::Usage("missing command".into()));
    };

    match command.as_str() {
        "init" => cmd_init(rest),
        "run" => cmd_run(rest),
        "status" => cmd_status(rest),
        "lineage" => cmd_lineage(rest),
        "verify" => cmd_verify(rest),
        "approve" => cmd_approve(rest),
        "stop" => cmd_stop(rest),
        "learner" => cmd_learner(rest),
        "help" | "--help" | "-h" => {
            println!("{USAGE}");

            Ok(())
        }
        other => Err(CliError::Usage(format!("unknown command {other}"))),
    }
}

fn cmd_init(args: &[String]) -> CliResult {
    let p = parse(
        args,
        &[
            ("--task", true),
            ("--seed", true),
            ("--mode", true),
            ("--flops", true),
            ("--variation", true),
            ("--secret", true),
            ("--arc-cells", false),
        ],
    )?;

    p.exactly(1, "init <run_dir>")?;

    match p.value("--mode") {
        Some("learner") => return init_learner(&p),
        Some("mlp") | None => {}
        Some(other) => return Err(CliError::Usage(format!("unknown mode {other}"))),
    }

    let learner_only = ["--flops", "--variation", "--secret", "--arc-cells"];

    if let Some(flag) = learner_only.iter().find(|f| p.flag(f)) {
        return Err(CliError::Usage(format!("{flag} needs --mode learner")));
    }

    let run_dir = PathBuf::from(&p.positional[0]);
    let mut config = Config::default();

    if let Some(name) = p.value("--task") {
        config.task = TaskKind::from_name(name)
            .ok_or_else(|| CliError::Usage(format!("unknown task {name}")))?;
    }

    if let Some(seed) = p.value("--seed") {
        config.master_seed = seed
            .parse()
            .map_err(|_| CliError::Usage(format!("bad seed {seed}")))?;
    }

    let config_path = run_dir.join(CONFIG_FILE);

    if config_path.exists() {
        return Err(CliError::Runtime(format!(
            "{} already exists; refusing to overwrite",
            config_path.display()
        )));
    }

    std::fs::create_dir_all(&run_dir).map_err(|e| CliError::Runtime(e.to_string()))?;
    config.save(&config_path)?;
    create_secret(&run_dir.join(SECRET_FILE))?;

    let dir = run_dir.display();

    println!("created run directory {dir}");
    println!(
        "  config:  {dir}/{CONFIG_FILE}  (task {}, seed {})",
        config.task.name(),
        config.master_seed
    );
    println!("  secret:  {dir}/{SECRET_FILE}  (mode 0600, read only by brut-eval)");
    println!();
    println!("next steps:");
    println!("  1. optional: edit {dir}/{CONFIG_FILE}");
    println!("  2. brut run {dir}");
    println!("  3. brut status {dir}");
    println!("  4. brut approve {dir} <candidate_id> --by <your name>");

    Ok(())
}

/// `brut-eval` next to this executable.
fn eval_bin() -> Result<PathBuf, CliError> {
    let exe =
        std::env::current_exe().map_err(|e| CliError::Runtime(format!("current_exe: {e}")))?;
    let dir = exe.parent().unwrap_or(Path::new("."));
    let eval_bin = dir.join(format!("brut-eval{}", std::env::consts::EXE_SUFFIX));

    if !eval_bin.is_file() {
        return Err(CliError::Runtime(format!(
            "{} not found (build both binaries, or use --no-hidden)",
            eval_bin.display()
        )));
    }

    Ok(eval_bin)
}

fn cmd_run(args: &[String]) -> CliResult {
    let p = parse(args, &[("--no-hidden", false)])?;

    p.exactly(1, "run <run_dir> [--no-hidden]")?;

    let run_dir = PathBuf::from(&p.positional[0]);

    if meta_evolve::is_meta_run(&run_dir) {
        return run_learner(&run_dir, p.flag("--no-hidden"));
    }

    let setup = if p.flag("--no-hidden") {
        None
    } else {
        Some(EvalSetup {
            eval_bin: eval_bin()?,
            secret_env: run_dir.join(SECRET_FILE),
        })
    };

    let summary = evolve::run(&run_dir, setup.as_ref())?;
    let json =
        serde_json::to_string_pretty(&summary).map_err(|e| CliError::Runtime(e.to_string()))?;

    println!("{json}");

    Ok(())
}

fn open_archive(run_dir: &Path) -> Result<Archive, CliError> {
    Ok(Archive::open(&run_dir.join(ARCHIVE_FILE))?)
}

fn cmd_status(args: &[String]) -> CliResult {
    let p = parse(args, &[])?;

    p.exactly(1, "status <run_dir>")?;

    let run_dir = PathBuf::from(&p.positional[0]);

    if meta_evolve::is_meta_run(&run_dir) {
        return status_learner(&run_dir);
    }

    let archive = open_archive(&run_dir)?;
    let mut candidates = archive.candidates();

    candidates.sort_by(|a, b| {
        b.fitness
            .total_cmp(&a.fitness)
            .then_with(|| a.id.cmp(&b.id))
    });

    println!(
        "{:<16}  {:>3}  {:>8}  {:>7}  {:>6}  status",
        "id", "gen", "fitness", "valid", "params"
    );

    for c in candidates.iter().take(10) {
        println!(
            "{:<16}  {:>3}  {:>8.4}  {:>7.4}  {:>6}  {:?}",
            c.id, c.generation, c.fitness, c.valid.primary, c.params, c.status
        );
    }

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();

    for c in &candidates {
        *counts.entry(format!("{:?}", c.status)).or_insert(0) += 1;
    }

    let counts_text: Vec<String> = counts.iter().map(|(k, v)| format!("{k} {v}")).collect();

    println!();
    println!(
        "candidates: {} ({})",
        candidates.len(),
        counts_text.join(", ")
    );

    let hidden = archive.hidden_evals();

    println!("hidden evaluations: {}", hidden.len());

    for h in &hidden {
        println!(
            "  {}  valid {:.4}  hidden {:.4}  gap {:+.4}{}",
            h.candidate_id,
            h.valid_primary,
            h.hidden_primary,
            h.gap,
            if h.quarantined { "  QUARANTINED" } else { "" }
        );
    }

    let approvals: Vec<String> = archive
        .records()
        .iter()
        .filter_map(|r| match &r.kind {
            RecordKind::Approved {
                candidate_id,
                by,
                note,
            } => Some(format!("  {candidate_id}  by {by}  {note}")),
            _ => None,
        })
        .collect();

    println!("approvals: {}", approvals.len());

    for a in approvals {
        println!("{a}");
    }

    let events = archive
        .records()
        .iter()
        .filter(|r| matches!(r.kind, RecordKind::SafetyEvent { .. }))
        .count();

    println!("safety events: {events}");

    if run_dir.join(safety::STOP_FILE).exists() {
        println!("kill switch: STOP file present");
    }

    Ok(())
}

fn cmd_lineage(args: &[String]) -> CliResult {
    let p = parse(args, &[])?;

    p.exactly(2, "lineage <run_dir> <candidate_id>")?;

    let run_dir = Path::new(&p.positional[0]);

    if meta_evolve::is_meta_run(run_dir) {
        return lineage_learner(run_dir, &p.positional[1]);
    }

    let archive = open_archive(run_dir)?;
    let id = &p.positional[1];
    let lineage = archive.lineage(id);

    if lineage.is_empty() {
        return Err(CliError::Runtime(format!("unknown candidate {id}")));
    }

    for c in lineage {
        let parents = if c.parents.is_empty() {
            "-".to_string()
        } else {
            c.parents.join("+")
        };

        println!(
            "{}  gen {:>3}  fit {:.4}  valid {:.4}  {:?}  via {} from {}  {}",
            c.id,
            c.generation,
            c.fitness,
            c.valid.primary,
            c.status,
            c.generator,
            parents,
            c.genome.summary()
        );
    }

    Ok(())
}

fn cmd_verify(args: &[String]) -> CliResult {
    let p = parse(args, &[])?;

    p.exactly(1, "verify <run_dir>")?;

    let archive = open_archive(Path::new(&p.positional[0]))?;
    let records = archive.records();
    let digests = archive.batch_digests().len();

    if digests > 0 {
        println!("OK {digests} batch digests: lineage chunks, snapshots and cache match");
    }

    match records.last() {
        Some(last) => println!(
            "OK {} records, last hash {} (keep it elsewhere as an external anchor)",
            records.len(),
            last.hash
        ),
        None => println!("OK 0 records"),
    }

    Ok(())
}

fn cmd_approve(args: &[String]) -> CliResult {
    let p = parse(args, &[("--by", true), ("--note", true)])?;

    p.exactly(
        2,
        "approve <run_dir> <candidate_id> --by <name> [--note <text>]",
    )?;

    let Some(by) = p.value("--by").map(str::trim).filter(|s| !s.is_empty()) else {
        return Err(CliError::Usage("approve needs --by <name>".into()));
    };

    let note = p.value("--note").unwrap_or("").to_string();
    let run_dir = Path::new(&p.positional[0]);
    let mut archive = open_archive(run_dir)?;
    let id = &p.positional[1];

    if meta_evolve::is_meta_run(run_dir) {
        return approve_learner(&mut archive, id, by, note);
    }

    let Some(candidate) = archive.get(id) else {
        return Err(CliError::Runtime(format!("unknown candidate {id}")));
    };

    if candidate.status != CandidateStatus::Ok {
        return Err(CliError::Runtime(format!(
            "refused: candidate {id} has status {:?}",
            candidate.status
        )));
    }

    let Some(hidden) = archive
        .hidden_evals()
        .into_iter()
        .find(|h| &h.candidate_id == id)
    else {
        return Err(CliError::Runtime(format!(
            "refused: candidate {id} has no hidden evaluation"
        )));
    };

    if hidden.quarantined {
        return Err(CliError::Runtime(format!(
            "refused: candidate {id} is quarantined"
        )));
    }

    let already = archive.records().iter().any(
        |r| matches!(&r.kind, RecordKind::Approved { candidate_id, .. } if candidate_id == id),
    );

    if already {
        return Err(CliError::Runtime(format!(
            "candidate {id} is already approved"
        )));
    }

    let record = RecordKind::Approved {
        candidate_id: id.clone(),
        by: by.to_string(),
        note,
    };

    archive.append(record, now_unix_ms())?;

    println!(
        "approved {id} by {by} (valid {:.4}, hidden {:.4}, gap {:+.4})",
        hidden.valid_primary, hidden.hidden_primary, hidden.gap
    );

    Ok(())
}

fn cmd_stop(args: &[String]) -> CliResult {
    let p = parse(args, &[])?;

    p.exactly(1, "stop <run_dir>")?;

    let run_dir = Path::new(&p.positional[0]);

    safety::request_stop(run_dir)?;

    println!(
        "kill switch set: {}",
        run_dir.join(safety::STOP_FILE).display()
    );
    println!("a running brut run stops within about a second; delete the file to allow new runs");

    Ok(())
}

// ---------------------------------------------------------------------------------------
// Learner mode
// ---------------------------------------------------------------------------------------

fn init_learner(p: &Parsed) -> CliResult {
    if p.flag("--task") {
        return Err(CliError::Usage(
            "--task is for MLP runs, not --mode learner".into(),
        ));
    }

    let run_dir = PathBuf::from(&p.positional[0]);
    let mut config = MetaConfig::default();

    if let Some(seed) = p.value("--seed") {
        config.master_seed = seed
            .parse()
            .map_err(|_| CliError::Usage(format!("bad seed {seed}")))?;
    }

    if let Some(text) = p.value("--flops") {
        config.limits.max_total_flops = parse_flops(text)?;
    }

    if let Some(name) = p.value("--variation") {
        config.variation = parse_variation(name)?;
    }

    if let Some(secret) = p.value("--secret") {
        config.hidden.secret_file = std::path::absolute(secret)
            .map_err(|e| CliError::Usage(format!("bad --secret {secret}: {e}")))?;
    }

    config.arc_cells = p.flag("--arc-cells");

    if !config.hidden.secret_file.is_file() {
        return Err(CliError::Runtime(format!(
            "learner secret {} not found; create it once with: brut learner secret-init \
             (or pass --secret <file>)",
            config.hidden.secret_file.display()
        )));
    }

    meta_evolve::init_meta_run(&run_dir, &config)?;

    let archive = open_archive(&run_dir)?;
    let dir = run_dir.display();

    println!("created learner run directory {dir}");
    println!(
        "  config:   {dir}/{META_CONFIG_FILE}  (seed {}, variation {:?}, flops {:.3e})",
        config.master_seed, config.variation, config.limits.max_total_flops as f64
    );
    println!(
        "  protocol: {}  (registered in the archive)",
        archive.protocol_hash().unwrap_or_default()
    );
    println!(
        "  secret:   {}  (read only by brut-eval)",
        config.hidden.secret_file.display()
    );
    println!();
    println!("next steps:");
    println!("  1. optional: edit {dir}/{META_CONFIG_FILE} (only before the first run)");
    println!("  2. brut run {dir}");
    println!("  3. brut learner report {dir}");
    println!("  4. brut approve {dir} <finalist_id> --by <your name>");

    Ok(())
}

/// `--flops 1e11` and friends: a finite number >= 1 that fits in u64.
fn parse_flops(text: &str) -> Result<u64, CliError> {
    let value: f64 = text
        .parse()
        .map_err(|_| CliError::Usage(format!("bad --flops {text}")))?;

    if !value.is_finite() || value < 1.0 || value >= u64::MAX as f64 {
        return Err(CliError::Usage(format!(
            "--flops {text} must be between 1 and 1.8e19"
        )));
    }

    Ok(value as u64)
}

fn parse_variation(name: &str) -> Result<VariationMode, CliError> {
    match name {
        "full" => Ok(VariationMode::Full),
        "constants-only" => Ok(VariationMode::ConstantsOnly),
        "random-search" => Ok(VariationMode::RandomSearch),
        other => Err(CliError::Usage(format!("unknown variation {other}"))),
    }
}

fn parse_set(name: &str) -> Result<EvalSet, CliError> {
    match name {
        "a1-eval" => Ok(EvalSet::A1Eval),
        "a2-eval" => Ok(EvalSet::A2Eval),
        "concept-arc" => Ok(EvalSet::ConceptArc),
        other => Err(CliError::Usage(format!("unknown ARC set {other}"))),
    }
}

fn load_meta(run_dir: &Path) -> Result<MetaConfig, CliError> {
    Ok(MetaConfig::load(&run_dir.join(META_CONFIG_FILE))?)
}

fn run_learner(run_dir: &Path, no_hidden: bool) -> CliResult {
    let config = load_meta(run_dir)?;

    let setup = if no_hidden {
        None
    } else {
        Some(LearnerEvalSetup {
            eval_bin: eval_bin()?,
            secret_env: config.hidden.secret_file.clone(),
        })
    };

    let summary = meta_evolve::meta_run(run_dir, setup.as_ref(), no_hidden)?;
    let json = serde_json::to_string_pretty(&summary_json(&summary))
        .map_err(|e| CliError::Runtime(e.to_string()))?;

    println!("{json}");

    Ok(())
}

fn summary_json(s: &MetaRunSummary) -> serde_json::Value {
    serde_json::json!({
        "batches_done": s.batches_done,
        "children": s.children,
        "unique_evals": s.unique_evals,
        "fec_task_hit_rate": s.fec_task_hit_rate,
        "hurdle_pass_rate": s.hurdle_pass_rate,
        "flops_used": s.flops_used,
        "wall_seconds": s.wall_seconds,
        "stop_reason": s.stop_reason,
        "finalists": s.finalists,
        "hidden": s.hidden,
    })
}

fn status_learner(run_dir: &Path) -> CliResult {
    let config = load_meta(run_dir)?;
    let archive = open_archive(run_dir)?;
    let digest = archive.last_batch_digest();
    let batches = digest.map_or(0, |d| d.last_batch + 1);

    println!("learner run {}", run_dir.display());
    println!("protocol {}", archive.protocol_hash().unwrap_or_default());
    println!(
        "batches {batches} of {} ({} digests) | variation {:?} | seed {}",
        config.max_batches,
        archive.batch_digests().len(),
        config.variation,
        config.master_seed
    );

    if let Some(d) = digest {
        let snap = read_snapshot(&run_dir.join(&d.snapshot_file), &d.snapshot_sha256)?;

        println!(
            "search flops {:.3e} of {:.3e} | best fitness {:.4} | panel rotation {}",
            snap.flops_used as f64,
            config.limits.max_total_flops as f64,
            d.best_fitness,
            snap.rotation
        );
        println!();
        println!("{:<16}  {:>8}  {:<16}  hall of fame", "id", "fitness", "fp");

        for m in snap.hall_of_fame.iter().take(10) {
            println!("{:<16}  {:>8.4}  {:<16}", m.id, m.fitness, m.fp);
        }
    }

    let finalists = meta_evolve::current_finalists(&archive);

    println!();
    println!("finalists: {}", finalists.len());

    for e in &finalists {
        println!(
            "  {}  from {}  panel mean {:.4}  fitness {:.4}  {:?}",
            e.id,
            e.parents.first().map_or("-", String::as_str),
            e.panel_mean,
            e.fitness,
            e.status
        );
    }

    let hidden = archive.learner_hidden_evals();

    println!("hidden evaluations: {}", hidden.len());

    for h in &hidden {
        println!(
            "  {}  {:?}  delta {:.2}{}{}",
            h.candidate_id,
            h.tier,
            h.delta,
            h.decision
                .map_or(String::new(), |d| format!("  decision {d:?}")),
            if h.quarantined { "  QUARANTINED" } else { "" }
        );
    }

    let mut approvals = 0;
    let mut events = 0;
    let mut arc = Vec::new();

    for r in archive.records() {
        match &r.kind {
            RecordKind::Approved { .. } => approvals += 1,
            RecordKind::SafetyEvent { .. } => events += 1,
            RecordKind::ExternalEval {
                candidate_id,
                set,
                official,
                kaggle,
                ..
            } => arc.push(format!(
                "  {candidate_id}  {set:?}  official {official:.4}  kaggle {kaggle:.4}"
            )),
            _ => {}
        }
    }

    println!("external ARC checks: {}", arc.len());

    for line in arc {
        println!("{line}");
    }

    println!("approvals: {approvals}");
    println!("safety events: {events}");

    if run_dir.join(safety::STOP_FILE).exists() {
        println!("kill switch: STOP file present");
    }

    Ok(())
}

fn lineage_learner(run_dir: &Path, id: &str) -> CliResult {
    let archive = open_archive(run_dir)?;
    let index = meta_report::lineage_index(run_dir, &archive)?;
    let steps = meta_report::ancestors(id, &archive, &index);

    if steps.is_empty() {
        return Err(CliError::Runtime(format!("unknown learner candidate {id}")));
    }

    for a in steps {
        println!(
            "{}  batch {:>7}  fit {:.4}  via {} from {}",
            a.id,
            a.batch,
            a.fitness,
            a.how,
            a.parent.as_deref().unwrap_or("-")
        );
    }

    Ok(())
}

fn approve_learner(archive: &mut Archive, id: &str, by: &str, note: String) -> CliResult {
    let Some(entry) = archive.learner_get(id) else {
        return Err(CliError::Runtime(format!("unknown learner candidate {id}")));
    };

    let hidden: Vec<_> = archive
        .learner_hidden_evals()
        .into_iter()
        .filter(|h| h.candidate_id == id)
        .collect();

    let Some(grammar) = hidden.iter().find(|h| h.tier == Tier::HiddenGrammar) else {
        return Err(CliError::Runtime(format!(
            "refused: {id} has no hidden_grammar evaluation"
        )));
    };

    let quarantined =
        entry.status == LearnerStatus::Quarantined || hidden.iter().any(|h| h.quarantined);

    if quarantined {
        return Err(CliError::Runtime(format!("refused: {id} is quarantined")));
    }

    let already = archive.records().iter().any(
        |r| matches!(&r.kind, RecordKind::Approved { candidate_id, .. } if candidate_id == id),
    );

    if already {
        return Err(CliError::Runtime(format!("{id} is already approved")));
    }

    let delta = grammar.delta;
    let decision = grammar.decision;

    let record = RecordKind::Approved {
        candidate_id: id.to_string(),
        by: by.to_string(),
        note,
    };

    archive.append(record, now_unix_ms())?;

    println!(
        "approved {id} by {by} (hidden_grammar delta {delta:.2}, decision {})",
        decision.map_or("-".to_string(), |d| format!("{d:?}"))
    );

    Ok(())
}

fn cmd_learner(args: &[String]) -> CliResult {
    let Some((sub, rest)) = args.split_first() else {
        return Err(CliError::Usage("learner needs a subcommand".into()));
    };

    match sub.as_str() {
        "secret-init" => learner_secret_init(rest),
        "show" => learner_show(rest),
        "report" => learner_report(rest),
        "arc-check" => learner_arc_check(rest),
        other => Err(CliError::Usage(format!("unknown learner command {other}"))),
    }
}

fn learner_secret_init(args: &[String]) -> CliResult {
    let p = parse(
        args,
        &[("--dir", true), ("--arc-root", true), ("--manifest", true)],
    )?;

    p.exactly(
        0,
        "learner secret-init [--dir <dir>] [--arc-root <dir>] [--manifest <file>]",
    )?;

    let dir = p
        .value("--dir")
        .map_or_else(default_secret_dir, PathBuf::from);
    let arc_root = p.value("--arc-root").map(PathBuf::from);
    let manifest = p.value("--manifest").map(PathBuf::from);
    let path = create_learner_secret(&dir, arc_root.as_deref(), manifest.as_deref())?;

    // Only the path: the secret itself is never printed.
    println!("{}", path.display());

    Ok(())
}

fn learner_show(args: &[String]) -> CliResult {
    let p = parse(args, &[])?;

    p.exactly(2, "learner show <run_dir> <id>")?;

    let run_dir = Path::new(&p.positional[0]);
    let id = &p.positional[1];
    let archive = open_archive(run_dir)?;

    let found = match archive.learner_get(id) {
        Some(e) => Some((e.genome, format!("{:?} via {}", e.status, e.generator))),
        None => match archive.last_batch_digest() {
            Some(d) => {
                let snap = read_snapshot(&run_dir.join(&d.snapshot_file), &d.snapshot_sha256)?;

                snap.population
                    .iter()
                    .chain(&snap.hall_of_fame)
                    .find(|m| &m.id == id)
                    .map(|m| {
                        (
                            m.genome.clone(),
                            "member of the latest snapshot".to_string(),
                        )
                    })
            }
            None => None,
        },
    };

    let Some((genome, what)) = found else {
        return Err(CliError::Runtime(format!("unknown learner program {id}")));
    };

    let simple = genome.simplify();
    let total = genome.setup.len() + genome.predict.len() + genome.learn.len();

    println!("program {id} ({what})");
    println!(
        "instructions {total}, effective {} (setup {}, predict {}, learn {} after simplify)",
        genome.effective_len(),
        simple.setup.len(),
        simple.predict.len(),
        simple.learn.len()
    );

    for f in SUPPORTED_F {
        println!(
            "cost at F = {f:>2}: setup {}, predict {}, learn {}, per step {}",
            genome.setup_cost(f),
            genome.predict_cost(f),
            genome.learn_cost(f),
            genome.step_cost(f)
        );
    }

    let listing = simple.render();

    println!();
    print!("{listing}");

    if !listing.ends_with('\n') {
        println!();
    }

    Ok(())
}

fn learner_report(args: &[String]) -> CliResult {
    let p = parse(args, &[])?;

    p.exactly(1, "learner report <run_dir>")?;

    let run_dir = Path::new(&p.positional[0]);
    let report = meta_report::write_report(run_dir)?;

    print!("{}", meta_report::render_table(&report));
    println!(
        "written to {}",
        run_dir.join(meta_report::REPORT_FILE).display()
    );

    Ok(())
}

fn learner_arc_check(args: &[String]) -> CliResult {
    let p = parse(args, &[("--set", true)])?;

    p.exactly(
        2,
        "learner arc-check <run_dir> <id> --set a1-eval|a2-eval|concept-arc",
    )?;

    let Some(set_name) = p.value("--set") else {
        return Err(CliError::Usage(
            "arc-check needs --set a1-eval|a2-eval|concept-arc".into(),
        ));
    };

    let set = parse_set(set_name)?;
    let run_dir = Path::new(&p.positional[0]);
    let config = load_meta(run_dir)?;

    let setup = LearnerEvalSetup {
        eval_bin: eval_bin()?,
        secret_env: config.hidden.secret_file.clone(),
    };

    let resp = meta_evolve::arc_check(run_dir, &p.positional[1], set, &setup)?;
    let json = serde_json::to_string_pretty(&resp).map_err(|e| CliError::Runtime(e.to_string()))?;

    println!("{json}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parse_splits_positionals_and_options() {
        let p = parse(
            &strings(&["dir", "--by", "ann", "id", "--no-hidden"]),
            &[("--by", true), ("--no-hidden", false)],
        )
        .ok()
        .expect("valid");

        assert_eq!(p.positional, vec!["dir", "id"]);
        assert_eq!(p.value("--by"), Some("ann"));
        assert!(p.flag("--no-hidden"));
        assert!(!p.flag("--note"));
    }

    #[test]
    fn parse_rejects_unknown_repeated_and_missing_values() {
        let allowed = [("--by", true)];

        assert!(matches!(
            parse(&strings(&["--what"]), &allowed),
            Err(CliError::Usage(_))
        ));
        assert!(matches!(
            parse(&strings(&["--by", "a", "--by", "b"]), &allowed),
            Err(CliError::Usage(_))
        ));
        assert!(matches!(
            parse(&strings(&["--by"]), &allowed),
            Err(CliError::Usage(_))
        ));
    }

    #[test]
    fn dispatch_usage_errors() {
        assert!(matches!(dispatch(&[]), Err(CliError::Usage(_))));
        assert!(matches!(
            dispatch(&strings(&["nope"])),
            Err(CliError::Usage(_))
        ));
        assert!(matches!(
            dispatch(&strings(&["verify"])),
            Err(CliError::Usage(_))
        ));
        assert!(matches!(
            dispatch(&strings(&["init", "x", "--seed", "abc"])),
            Err(CliError::Usage(_))
        ));
    }

    #[test]
    fn learner_usage_errors() {
        let usage = |args: &[&str]| matches!(dispatch(&strings(args)), Err(CliError::Usage(_)));

        assert!(usage(&["learner"]));
        assert!(usage(&["learner", "nope"]));
        assert!(usage(&["learner", "show", "dir"]));
        assert!(usage(&["learner", "report"]));
        assert!(usage(&["learner", "arc-check", "dir", "id"]));
        assert!(usage(&[
            "learner",
            "arc-check",
            "dir",
            "id",
            "--set",
            "a3-eval"
        ]));
        assert!(usage(&["learner", "secret-init", "extra"]));
        assert!(usage(&["init", "x", "--mode", "robot"]));
        assert!(usage(&["init", "x", "--flops", "1e9"]));
        assert!(usage(&["init", "x", "--arc-cells"]));
        assert!(usage(&[
            "init", "x", "--mode", "learner", "--task", "parity"
        ]));
        assert!(usage(&["init", "x", "--mode", "learner", "--seed", "-1"]));
        assert!(usage(&["init", "x", "--mode", "learner", "--flops", "abc"]));
        assert!(usage(&["init", "x", "--mode", "learner", "--flops", "0.5"]));
        assert!(usage(&["init", "x", "--mode", "learner", "--flops", "inf"]));
        assert!(usage(&[
            "init",
            "x",
            "--mode",
            "learner",
            "--variation",
            "greedy"
        ]));
    }

    #[test]
    fn learner_option_values() {
        assert_eq!(parse_flops("1e11").ok(), Some(100_000_000_000));
        assert_eq!(parse_flops("1500000000000").ok(), Some(1_500_000_000_000));
        assert_eq!(
            parse_variation("constants-only").ok(),
            Some(VariationMode::ConstantsOnly)
        );
        assert_eq!(
            parse_variation("random-search").ok(),
            Some(VariationMode::RandomSearch)
        );
        assert_eq!(parse_set("a1-eval").ok(), Some(EvalSet::A1Eval));
        assert_eq!(parse_set("a2-eval").ok(), Some(EvalSet::A2Eval));
        assert_eq!(parse_set("concept-arc").ok(), Some(EvalSet::ConceptArc));
    }

    #[test]
    fn learner_init_needs_an_existing_secret() {
        let dir = std::env::temp_dir().join(format!("brut-main-init-{}", std::process::id()));
        let missing = dir.join("no-such-secret.env");
        let args = strings(&[
            "init",
            dir.join("run").to_str().unwrap_or("run"),
            "--mode",
            "learner",
            "--secret",
            missing.to_str().unwrap_or("missing"),
        ]);

        let Err(CliError::Runtime(m)) = dispatch(&args) else {
            panic!("expected a runtime error");
        };

        assert!(m.contains("secret-init"), "{m}");
        assert!(!dir.join("run").exists());
    }
}
