//! brut-ingest: the gate that admits merged candidates into a run, and the scope-fence
//! checker for mutation pull requests (ADR 0004).
//!
//! Four modes. In the first three, each `*.json` must be a regular file (no symlink, FIFO or
//! device) of at most `hitl::MAX_CANDIDATE_FILE_BYTES`, and every printed line escapes the
//! control characters of file names and messages, which come from the pull request.
//!   brut-ingest --candidates <dir> [--against <pool-dir>]
//!     Validate every `*.json` in the candidate directory: it must parse, pass `check`
//!     (structure plus cost cap), have a file name equal to its content id, and not
//!     duplicate a seed program or (with --against) a candidate already in <pool-dir>.
//!     Prints one line per file and exits non-zero if any candidate is rejected.
//!   brut-ingest --cascades <dir> [--against <pool-dir>]
//!     The same gate for promoted Cascades (ADR 0005): parse, check against the pool
//!     allow-list (credential look-alikes in prompts included), file name equals content
//!     id, no duplicate of a seed Cascade or (with --against) of a pooled Cascade.
//!   brut-ingest --models <dir> [--against <pool-dir>]
//!     The same gate for promoted models of the MLP mode (ADR 0008): parse with unknown
//!     fields refused, the shape check and the current default Limits on the task the
//!     provenance names, a consistent provenance, file name equals content id, no duplicate
//!     of a base model of a CNN family or (with --against) of a pooled model.
//!   brut-ingest --scope-check
//!     Read newline-separated changed paths from stdin and fail if any path is outside
//!     `candidates/`. The one scope rule lives in the library (`hitl::path_in_scope`);
//!     git stays in `scripts/scope-fence.sh`, which feeds this checker.
#![forbid(unsafe_code)]

use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process;

use brutmaschine::hitl::{
    CandidateAdapter, CascadeAdapter, GenomeAdapter, MAX_CANDIDATE_FILE_BYTES, ModelAdapter,
    cascade_seed_ids, ingest_one, model_seed_ids, path_in_scope, seed_ids,
};
use brutmaschine::util::{BrutError, Result, read_regular_file};

fn main() {
    if let Err(e) = run() {
        eprintln!("brut-ingest: {e}");
        process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut scope_check = false;
    let mut candidates: Option<String> = None;
    let mut cascades: Option<String> = None;
    let mut models: Option<String> = None;
    let mut against: Option<String> = None;
    let mut it = std::env::args().skip(1);

    while let Some(tok) = it.next() {
        match tok.as_str() {
            "--scope-check" => scope_check = true,
            "--candidates" => {
                candidates = Some(
                    it.next()
                        .ok_or_else(|| BrutError::Invalid("--candidates needs a value".into()))?,
                );
            }
            "--cascades" => {
                cascades = Some(
                    it.next()
                        .ok_or_else(|| BrutError::Invalid("--cascades needs a value".into()))?,
                );
            }
            "--models" => {
                models = Some(
                    it.next()
                        .ok_or_else(|| BrutError::Invalid("--models needs a value".into()))?,
                );
            }
            "--against" => {
                against = Some(
                    it.next()
                        .ok_or_else(|| BrutError::Invalid("--against needs a value".into()))?,
                );
            }
            other => return Err(BrutError::Invalid(format!("unknown argument {other}"))),
        }
    }

    if scope_check {
        return scope_check_stdin();
    }

    let against = against.as_deref().map(Path::new);

    match (candidates, cascades, models) {
        (Some(dir), None, None) => {
            ingest_dir(&GenomeAdapter::new(), seed_ids(), Path::new(&dir), against)
        }
        (None, Some(dir), None) => ingest_dir(
            &CascadeAdapter::new(),
            cascade_seed_ids(),
            Path::new(&dir),
            against,
        ),
        (None, None, Some(dir)) => ingest_dir(
            &ModelAdapter::new(),
            model_seed_ids(),
            Path::new(&dir),
            against,
        ),
        _ => Err(BrutError::Invalid(
            "give exactly one of --candidates <dir>, --cascades <dir>, --models <dir>, or \
             --scope-check"
                .into(),
        )),
    }
}

/// Read changed paths from stdin; exit non-zero if any is outside `candidates/`.
fn scope_check_stdin() -> Result<()> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;

    let mut offenders = Vec::new();

    for line in input.lines() {
        let path = line.trim();

        if path.is_empty() {
            continue;
        }

        if path_in_scope(path) {
            println!("ok    {path}");
        } else {
            println!("OUT   {path}");
            offenders.push(path.to_string());
        }
    }

    if offenders.is_empty() {
        println!("scope-fence: all changed paths are under candidates/");
        Ok(())
    } else {
        eprintln!(
            "scope-fence: {} path(s) outside candidates/ in a mutation pull request",
            offenders.len()
        );
        process::exit(1);
    }
}

/// Validate every candidate file in `dir`, deduping against the seeds (`known`) and, when
/// given, an existing pool; exit non-zero if any is rejected.
fn ingest_dir<A: CandidateAdapter>(
    adapter: &A,
    mut known: HashSet<String>,
    dir: &Path,
    against: Option<&Path>,
) -> Result<()> {
    if !dir.exists() {
        return Err(BrutError::Invalid(format!(
            "candidate directory {} does not exist",
            dir.display()
        )));
    }

    if let Some(pool) = against {
        known.extend(load_pool_ids(adapter, pool)?);
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();

    if files.is_empty() {
        println!("brut-ingest: no candidate files in {}", dir.display());
        return Ok(());
    }

    let mut rejected = 0;

    for path in &files {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let text = match read_regular_file(path, MAX_CANDIDATE_FILE_BYTES) {
            Ok(text) => text,
            Err(e) => {
                rejected += 1;
                println!(
                    "REJECT {}  ({})",
                    one_line(&path.display().to_string()),
                    one_line(&e.to_string())
                );
                continue;
            }
        };
        let rec = ingest_one(adapter, &path.display().to_string(), &text, stem, &known);

        if rec.accepted {
            if let Some(id) = &rec.id {
                known.insert(id.clone()); // dedupe later files in this batch too
            }
            println!(
                "ok    {}  {}",
                rec.id.unwrap_or_default(),
                one_line(&path.display().to_string())
            );
        } else {
            rejected += 1;
            println!(
                "REJECT {}  ({})",
                one_line(&path.display().to_string()),
                one_line(&rec.reason)
            );
        }
    }

    println!(
        "brut-ingest: {} accepted, {} rejected",
        files.len() - rejected,
        rejected
    );

    if rejected > 0 {
        process::exit(1);
    }

    Ok(())
}

/// Content ids of an existing candidate pool, to dedupe incoming candidates against.
/// Unparseable pool files are skipped with a warning (only ids are needed here).
fn load_pool_ids<A: CandidateAdapter>(adapter: &A, dir: &Path) -> Result<HashSet<String>> {
    if !dir.exists() {
        return Err(BrutError::Invalid(format!(
            "pool directory {} does not exist",
            dir.display()
        )));
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();

    let mut ids = HashSet::new();

    for path in files {
        let text = match read_regular_file(&path, MAX_CANDIDATE_FILE_BYTES) {
            Ok(text) => text,
            Err(e) => {
                eprintln!(
                    "brut-ingest: skipping unreadable pool file {}: {}",
                    one_line(&path.display().to_string()),
                    one_line(&e.to_string())
                );
                continue;
            }
        };

        match adapter.parse(&text) {
            Ok(g) => {
                ids.insert(adapter.fingerprint(&g));
            }
            Err(e) => eprintln!(
                "brut-ingest: skipping unparseable pool file {}: {}",
                one_line(&path.display().to_string()),
                one_line(&e.to_string())
            ),
        }
    }

    Ok(ids)
}

/// `text` on one line: line breaks, other control characters and backslashes are escaped
/// as in a Rust string literal; quotes stay as they are. File names and parser messages come
/// from the pull request, so no line of the gate's log can be one that a file wrote.
fn one_line(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '"' | '\'' => c.to_string(),
            _ => c.escape_debug().to_string(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::one_line;

    #[test]
    fn one_line_keeps_a_log_line_to_one_line() {
        assert_eq!(
            one_line("candidates/models/abc.json"),
            "candidates/models/abc.json"
        );
        assert_eq!(
            one_line("x\nok    4ea1fec7699a4627  forged\r::error::y"),
            "x\\nok    4ea1fec7699a4627  forged\\r::error::y"
        );
        assert_eq!(one_line("not \"a\" ('b')"), "not \"a\" ('b')");
        assert!(!one_line("a\u{1b}[2Kb").contains('\u{1b}'));
    }
}
