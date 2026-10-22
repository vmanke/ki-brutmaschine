//! brut-propose: draft a two-candidate genome mutation for a human-reviewed pull request
//! (ADR 0004).
//!
//! Pure compute plus file writes under one output directory. It reads a parent pool of
//! candidate genomes (or falls back to the seed programs), mutates two of them under a
//! master seed, validates each, and writes the two candidate files plus a provenance
//! manifest. It never calls git or the network; the drafting script in `scripts/` turns
//! the output into a branch and a draft pull request.
//!
//! Usage:
//!   brut-propose --out <dir> [--pool <dir>] --seed <u64> [--mode full|constants]
//!
//! Output:
//!   <out>/candidates/<child-id>.json   the two files to commit under `candidates/`
//!   <out>/proposal.json                provenance for the pull-request body (not committed)
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process;

use brutmaschine::hitl::{GenomeAdapter, propose_pair};
use brutmaschine::learner::{LearnerGenome, VariationMode};
use brutmaschine::util::{BrutError, Result};

fn main() {
    if let Err(e) = run() {
        eprintln!("brut-propose: {e}");
        process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = parse_args()?;

    let out = PathBuf::from(require(&args, "out")?);
    let seed: u64 = require(&args, "seed")?
        .parse()
        .map_err(|_| BrutError::Invalid("--seed must be a u64".to_string()))?;
    let mode = match args.get("mode").map(String::as_str) {
        None | Some("full") => VariationMode::Full,
        Some("constants") => VariationMode::ConstantsOnly,
        Some(other) => {
            return Err(BrutError::Invalid(format!(
                "--mode must be full or constants, got {other}"
            )));
        }
    };

    let adapter = GenomeAdapter::new();

    let pool = match args.get("pool") {
        Some(dir) => load_pool(&adapter, Path::new(dir))?,
        None => Vec::new(),
    };

    if pool.is_empty() {
        println!("brut-propose: empty pool, using the seed programs as parents");
    } else {
        println!(
            "brut-propose: {} parent candidate(s) in the pool",
            pool.len()
        );
    }

    let (proposed, manifest) = propose_pair(&adapter, &pool, mode, seed)?;

    let cand_dir = out.join("candidates");

    // Reject a non-empty output so a stale candidate from an earlier run cannot ride along
    // into the pull request; each proposal must carry exactly its two candidates.
    if cand_dir.exists() {
        let has_json = std::fs::read_dir(&cand_dir)?
            .filter_map(|e| e.ok())
            .any(|e| e.path().extension().is_some_and(|x| x == "json"));

        if has_json {
            return Err(BrutError::Invalid(format!(
                "{} already contains candidate files; use a fresh --out so the pull request carries exactly the two proposed candidates",
                cand_dir.display()
            )));
        }
    }

    std::fs::create_dir_all(&cand_dir)?;

    for p in &proposed {
        let path = cand_dir.join(format!("{}.json", p.provenance.child_id));
        std::fs::write(&path, adapter.to_json(&p.genome)?)?;

        println!("\n--- candidate {} ---", p.provenance.child_id);
        println!(
            "parent {}  operator {:?}  seed {}",
            p.provenance.parent_id, p.provenance.operator, p.provenance.seed
        );
        print!("{}", adapter.summary(&p.genome));
        println!("wrote {}", path.display());
    }

    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| BrutError::Invalid(format!("manifest encode: {e}")))?;
    let manifest_path = out.join("proposal.json");
    std::fs::write(&manifest_path, manifest_json)?;

    println!("\nwrote {}", manifest_path.display());
    println!(
        "next: commit the two files under `candidates/` on a `mutate/` branch and open a draft PR\n(scripts/propose-pr.sh does this and puts proposal.json in the PR body)"
    );

    Ok(())
}

/// Read every `*.json` in `dir` as a candidate genome, validating each; error on any
/// invalid file so a run never starts from a broken pool.
fn load_pool(adapter: &GenomeAdapter, dir: &Path) -> Result<Vec<LearnerGenome>> {
    if !dir.exists() {
        return Err(BrutError::Invalid(format!(
            "pool directory {} does not exist",
            dir.display()
        )));
    }

    let mut names: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    names.sort();

    let mut pool = Vec::new();

    for path in names {
        let text = std::fs::read_to_string(&path)?;
        let g = adapter.parse(&text)?;

        adapter.validate(&g).map_err(|e| {
            BrutError::Invalid(format!("pool file {} invalid: {e}", path.display()))
        })?;

        pool.push(g);
    }

    Ok(pool)
}

/// Minimal `--key value` parser; flags without a value are not used here.
fn parse_args() -> Result<BTreeMap<String, String>> {
    let mut map = BTreeMap::new();
    let mut it = std::env::args().skip(1);

    while let Some(tok) = it.next() {
        let key = tok
            .strip_prefix("--")
            .ok_or_else(|| BrutError::Invalid(format!("expected --key, got {tok}")))?;
        let val = it
            .next()
            .ok_or_else(|| BrutError::Invalid(format!("--{key} needs a value")))?;

        map.insert(key.to_string(), val);
    }

    Ok(map)
}

fn require(args: &BTreeMap<String, String>, key: &str) -> Result<String> {
    args.get(key)
        .cloned()
        .ok_or_else(|| BrutError::Invalid(format!("--{key} is required")))
}
