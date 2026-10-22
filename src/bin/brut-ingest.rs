//! brut-ingest: the gate that admits merged candidates into a run, and the scope-fence
//! checker for mutation pull requests (ADR 0004).
//!
//! Two modes:
//!   brut-ingest --candidates <dir> [--against <pool-dir>]
//!     Validate every `*.json` in the candidate directory: it must parse, pass `check`
//!     (structure plus cost cap), have a file name equal to its content id, and not
//!     duplicate a seed program or (with --against) a candidate already in <pool-dir>.
//!     Prints one line per file and exits non-zero if any candidate is rejected.
//!   brut-ingest --scope-check
//!     Read newline-separated changed paths from stdin and fail if any path is outside
//!     `candidates/`. The one scope rule lives in the library (`hitl::path_in_scope`);
//!     git stays in `scripts/scope-fence.sh`, which feeds this checker.
#![forbid(unsafe_code)]

use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process;

use brutmaschine::hitl::{GenomeAdapter, ingest_one, path_in_scope, seed_ids};
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

    match candidates {
        Some(dir) => ingest_dir(Path::new(&dir), against.as_deref().map(Path::new)),
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

/// Validate every candidate file in `dir`, deduping against the seed programs and, when
/// given, an existing pool; exit non-zero if any is rejected.
fn ingest_dir(dir: &Path, against: Option<&Path>) -> Result<()> {
    if !dir.exists() {
        return Err(BrutError::Invalid(format!(
            "candidate directory {} does not exist",
            dir.display()
        )));
    }

    let adapter = GenomeAdapter::new();

    let mut known = seed_ids();
    if let Some(pool) = against {
        known.extend(load_pool_ids(&adapter, pool)?);
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
        let text = std::fs::read_to_string(path)?;
        let rec = ingest_one(&adapter, &path.display().to_string(), &text, stem, &known);

        if rec.accepted {
            if let Some(id) = &rec.id {
                known.insert(id.clone()); // dedupe later files in this batch too
            }
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

/// Content ids of an existing candidate pool, to dedupe incoming candidates against.
/// Unparseable pool files are skipped with a warning (only ids are needed here).
fn load_pool_ids(adapter: &GenomeAdapter, dir: &Path) -> Result<HashSet<String>> {
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
        let text = std::fs::read_to_string(&path)?;

        match adapter.parse(&text) {
            Ok(g) => {
                ids.insert(adapter.fingerprint(&g));
            }
            Err(e) => eprintln!(
                "brut-ingest: skipping unparseable pool file {}: {e}",
                path.display()
            ),
        }
    }

    Ok(ids)
}
