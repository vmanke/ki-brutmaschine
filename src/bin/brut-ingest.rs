//! brut-ingest: the gate that admits merged candidates into a run, and the scope-fence
//! checker for mutation pull requests (ADR 0004).
//!
//! Two modes:
//!   brut-ingest --candidates <dir>
//!     Validate every `*.json` in the candidate directory: it must parse, pass `check`
//!     (structure plus cost cap), and its content id must equal its file name. Prints one
//!     line per file and exits non-zero if any candidate is rejected.
//!   brut-ingest --scope-check
//!     Read newline-separated changed paths from stdin and fail if any path is outside
//!     `candidates/`. The one scope rule lives in the library (`hitl::path_in_scope`);
//!     git stays in `scripts/scope-fence.sh`, which feeds this checker.
#![forbid(unsafe_code)]

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process;

use brutmaschine::hitl::{GenomeAdapter, ingest_one, path_in_scope};
use brutmaschine::util::{BrutError, Result};

fn main() {
    if let Err(e) = run() {
        eprintln!("brut-ingest: {e}");
        process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut scope_check = false;
    let mut candidates: Option<String> = None;
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
            other => return Err(BrutError::Invalid(format!("unknown argument {other}"))),
        }
    }

    if scope_check {
        return scope_check_stdin();
    }

    match candidates {
        Some(dir) => ingest_dir(Path::new(&dir)),
        None => Err(BrutError::Invalid(
            "give --candidates <dir> or --scope-check".into(),
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

/// Validate every candidate file in `dir`; exit non-zero if any is rejected.
fn ingest_dir(dir: &Path) -> Result<()> {
    if !dir.exists() {
        return Err(BrutError::Invalid(format!(
            "candidate directory {} does not exist",
            dir.display()
        )));
    }

    let adapter = GenomeAdapter::new();

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
        let text = std::fs::read_to_string(path)?;
        let rec = ingest_one(&adapter, &path.display().to_string(), &text, stem);

        if rec.accepted {
            println!("ok    {}  {}", rec.id.unwrap_or_default(), path.display());
        } else {
            rejected += 1;
            println!("REJECT {}  ({})", path.display(), rec.reason);
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
