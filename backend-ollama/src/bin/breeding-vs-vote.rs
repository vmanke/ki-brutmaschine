//! breeding-vs-vote: the analysis of docs/PREREG_BREEDING_VS_VOTE.md (module
//! `breeding_vs_vote`, which says what it reads, checks and computes).
//!
//! Usage:
//!   breeding-vs-vote --runs <dir> [--pilot] [--frozen <dir>] [--json <file>] [--table <file>]
//!   breeding-vs-vote --match <run B-s> [--frozen <dir>]
//!   breeding-vs-vote --panels <run B-s> [--frozen <dir>]
//!
//! --runs analyses the runs B-1, L-1, B-2, L-2, ... in <dir>, in the order of the counted
//! list 1 to 60. With --pilot it reads the pilot seeds 901 to 905 instead and makes no
//! claim. It prints the tables and the verdict lines. It writes the result as JSON to
//! <dir>/analysis.json and the tables to <dir>/analysis.txt, or to --json and --table.
//!
//! --match is the call matching for run-seed.sh. It prints one line: `samples N` when the
//! vote record must hold N samples first (run vote-control with --samples N, then ask
//! again), or `matched N_up N_down`, with `-` for an N_down that does not exist. Kept vote
//! results that exist already are checked.
//!
//! --panels writes <run B-s>/panels.json for the recount, or checks the one there.
//!
//! --frozen names the directory of config-B.json and config-L.json (default
//! docs/prereg-breeding-vs-vote, seen from the repository root).
//!
//! Exit codes: 0 done, 1 a refusal or an error, 2 wrong usage.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process;

use backend_ollama::breeding_vs_vote::{
    FROZEN_DIR, Frozen, List, Matching, analyse, match_for_driver, render, write_panels,
};
use brutmaschine::util::{BrutError, Result, write_atomic};

fn main() {
    let code = match run() {
        Ok(()) => 0,
        Err(BrutError::Invalid(m)) if m.starts_with("usage:") => {
            eprintln!("breeding-vs-vote: {m}");
            2
        }
        Err(e) => {
            eprintln!("breeding-vs-vote: {e}");
            1
        }
    };

    process::exit(code);
}

fn usage(m: &str) -> BrutError {
    BrutError::Invalid(format!("usage: {m}"))
}

fn parse_args() -> Result<BTreeMap<String, String>> {
    const FLAGS: [&str; 1] = ["pilot"];
    const KEYS: [&str; 6] = ["runs", "match", "panels", "frozen", "json", "table"];

    let mut map = BTreeMap::new();
    let mut it = std::env::args().skip(1);

    while let Some(tok) = it.next() {
        let key = tok
            .strip_prefix("--")
            .ok_or_else(|| usage(&format!("expected --key, got {tok}")))?;

        let val = if FLAGS.contains(&key) {
            String::new()
        } else if KEYS.contains(&key) {
            it.next()
                .ok_or_else(|| usage(&format!("--{key} needs a value")))?
        } else {
            return Err(usage(&format!("unknown option --{key}")));
        };

        if map.insert(key.to_string(), val).is_some() {
            return Err(usage(&format!("--{key} given twice")));
        }
    }

    Ok(map)
}

fn run() -> Result<()> {
    let args = parse_args()?;
    let modes: Vec<&str> = ["runs", "match", "panels"]
        .into_iter()
        .filter(|m| args.contains_key(*m))
        .collect();
    let [mode] = modes[..] else {
        return Err(usage("give exactly one of --runs, --match and --panels"));
    };

    if mode != "runs"
        && ["pilot", "json", "table"]
            .iter()
            .any(|k| args.contains_key(*k))
    {
        return Err(usage("--pilot, --json and --table go with --runs only"));
    }

    let frozen = Frozen::read(Path::new(
        args.get("frozen").map_or(FROZEN_DIR, String::as_str),
    ))?;
    let dir = PathBuf::from(&args[mode]);

    match mode {
        "match" => {
            match match_for_driver(&dir, &frozen)? {
                Matching::Need(n) => println!("samples {n}"),
                Matching::Matched(m) => println!(
                    "matched {} {}",
                    m.n_up,
                    m.n_down.map_or("-".to_string(), |n| n.to_string())
                ),
            }

            Ok(())
        }
        "panels" => {
            let wrote = write_panels(&dir, &frozen)?;

            eprintln!(
                "breeding-vs-vote: {} {}",
                dir.join(backend_ollama::breeding_vs_vote::PANELS).display(),
                if wrote {
                    "written"
                } else {
                    "checked, unchanged"
                }
            );

            Ok(())
        }
        _ => {
            let list = if args.contains_key("pilot") {
                List::Pilot
            } else {
                List::Counted
            };
            let analysis = analyse(&dir, &frozen, list)?;
            let table = render(&analysis);
            let json_path = args
                .get("json")
                .map_or_else(|| dir.join("analysis.json"), PathBuf::from);
            let table_path = args
                .get("table")
                .map_or_else(|| dir.join("analysis.txt"), PathBuf::from);

            print!("{table}");
            write_atomic(
                &json_path,
                format!("{}\n", serde_json::to_string_pretty(&analysis)?).as_bytes(),
            )?;
            write_atomic(&table_path, table.as_bytes())?;
            eprintln!(
                "breeding-vs-vote: wrote {} and {}",
                json_path.display(),
                table_path.display()
            );

            Ok(())
        }
    }
}
