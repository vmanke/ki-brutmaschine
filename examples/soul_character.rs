//! The Character of every Worker of a Run with Souls, recomputed from the Run's record (step
//! 2, ticket 09 of `.scratch/seele-und-kappa/`; Q51, Q55).
//!
//! No genome, member, snapshot or event stores a Character (Q55). This program shows that any
//! Character can still be recomputed from the record. It reads a run directory of
//! `brut-breed`:
//!
//! - the settings in `config.json`: the Domain, the seed, the search panel's size, and the
//!   readout and strength setting of the Souls;
//! - the current Souls of the checkpoint `snapshot.json`;
//! - the run start in `events.jsonl`, to check that the search panel rebuilt from the settings
//!   is the logged one.
//!
//! For every member of the archive (in archive order), every Worker (in Stage order) and every
//! distinct target (circuits) or rule (grids) of the search panel (in the order of its first
//! task), it prints two numbers of the Worker's current Soul with the task's Family strengths,
//! as the Run reads them (spec 6.1):
//!
//! - the Character χ = |Σ_f s_f · u(p_f)|²;
//! - the normalised Character χ / Σ_f s_f², which the logs use.
//!
//! The dressed tasks of a circuit target share its strengths. So do the grid puzzles of one
//! kind of rule: its geometric step, named as in `Geo`, with "+colours" when the rule also
//! carries a colour map. With `strengths: equal` every family has the strength 1 on every
//! task.
//!
//! The output is a table with tabs. First come comment lines that start with "#": the run,
//! its families, and each target or rule with its number of search tasks and its strengths.
//! Then the header `member worker target character character_norm` follows (`rule` instead
//! of `target` for grids), then one row per member, Worker and target or rule. Each number is
//! printed in the shortest form that reads back to the same f64.
//!
//! It makes no model call and changes no file. It reads the checkpoint as it is, so run
//! `brut-breed --out <dir> --verify` first.
//!
//! Usage:
//!   cargo run --example soul_character -- <run dir>
#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::path::Path;

use serde::Deserialize;
use serde::de::DeserializeOwned;

use brutmaschine::cascade::arena::{BreedConfig, Event, Snapshot};
use brutmaschine::cascade::circuit::CircuitDomain;
use brutmaschine::cascade::domain::{Domain, DomainConfig, GridDomain};
use brutmaschine::cascade::puzzle::Rule;
use brutmaschine::cascade::soul::{Readout, Strengths, character, character_norm};
use brutmaschine::util::derive_seed;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir] = args.as_slice() else {
        eprintln!("usage: cargo run --example soul_character -- <run dir>");
        std::process::exit(2);
    };

    match run(Path::new(dir)) {
        // A reader that stops early, such as `head`, closes the pipe; that is no error.
        Ok(text) => {
            if let Err(e) = std::io::stdout().write_all(text.as_bytes())
                && e.kind() != ErrorKind::BrokenPipe
            {
                eprintln!("soul_character: {e}");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("soul_character: {e}");
            std::process::exit(1);
        }
    }
}

/// The checkpoint that `brut-breed` writes. Only its snapshot is read here.
#[derive(Deserialize)]
struct Checkpoint<A> {
    snapshot: Snapshot<A>,
}

/// The table of the Run in `dir`, as the program prints it.
pub fn run(dir: &Path) -> Result<String, String> {
    let cfg: BreedConfig = read_json(&dir.join("config.json"))?;

    match cfg.domain {
        DomainConfig::Grid => table::<GridDomain>(dir, &cfg, "rule", |p| rule_label(&p.rule)),
        DomainConfig::Circuits(_) => {
            table::<CircuitDomain>(dir, &cfg, "target", |t| t.name.to_string())
        }
    }
}

/// The kind of a grid rule: its geometric step, and "+colours" when it carries a colour map.
/// All puzzles of one kind have the same strengths (spec 5.3).
fn rule_label(rule: &Rule) -> String {
    match rule.colours {
        Some(_) => format!("{:?}+colours", rule.geo),
        None => format!("{:?}", rule.geo),
    }
}

/// The table of the Run in `dir` with the settings `cfg` on the Domain `D`. `label` names the
/// target or rule of a task, and `column` is the header of these names.
fn table<D: Domain>(
    dir: &Path,
    cfg: &BreedConfig,
    column: &str,
    label: impl Fn(&D::Task) -> String,
) -> Result<String, String> {
    let soul = cfg.soul.as_ref().ok_or_else(|| {
        format!(
            "{}: the Run has no Souls, so its Workers have no Character",
            dir.join("config.json").display()
        )
    })?;
    let domain = D::from_config(cfg).map_err(|e| e.to_string())?;
    let checkpoint: Checkpoint<D::Answer> = read_json(&dir.join("snapshot.json"))?;
    let snapshot = checkpoint.snapshot;

    if snapshot.config_hash != cfg.fixed_hash() {
        return Err(format!(
            "{} does not match the checkpoint {}",
            dir.join("config.json").display(),
            dir.join("snapshot.json").display()
        ));
    }

    let panel = domain.panel(derive_seed(cfg.seed, "panel/search"), cfg.search_puzzles);

    check_panel(dir, &domain, &panel, !snapshot.archive.is_empty())?;

    // Each target or rule once, in the order of its first task, with its strengths and the
    // number of its tasks. The tasks of one name must share their strengths.
    let mut groups: Vec<(String, Vec<f64>, usize)> = Vec::new();

    for task in &panel {
        let name = label(task);
        let s = soul.strengths.of(&domain, task);

        match groups.iter_mut().find(|(n, _, _)| *n == name) {
            Some((_, strengths, count)) if *strengths == s => *count += 1,
            Some(_) => {
                return Err(format!(
                    "the search tasks of {column} {name} have different strengths"
                ));
            }
            None => groups.push((name, s, 1)),
        }
    }

    let families = domain.families();
    let workers: usize = snapshot
        .archive
        .iter()
        .map(|m| m.cascade.worker_count())
        .sum();
    let mut lines = vec![
        format!(
            "# {}: {} members, {workers} Workers; readout {}, strengths {}; current Souls of the checkpoint",
            dir.display(),
            snapshot.archive.len(),
            match soul.readout {
                Readout::Complex => "complex",
                Readout::Real => "real",
            },
            match soul.strengths {
                Strengths::Domain => "domain",
                Strengths::Equal => "equal",
            }
        ),
        format!("# families: {}", families.join(" ")),
    ];

    for (name, s, count) in &groups {
        let strengths: Vec<String> = s.iter().map(f64::to_string).collect();

        lines.push(format!(
            "# {column} {name}: {count} search task(s), strengths {}",
            strengths.join(" ")
        ));
    }

    lines.push(format!(
        "member\tworker\t{column}\tcharacter\tcharacter_norm"
    ));

    for m in &snapshot.archive {
        let now = m
            .soul_now
            .as_ref()
            .filter(|now| now.len() == m.cascade.worker_count())
            .ok_or_else(|| {
                format!(
                    "member {} does not hold a current Soul for each of its Workers",
                    m.id
                )
            })?;

        for (k, phases) in now.iter().enumerate() {
            if phases.len() != families.len() {
                return Err(format!(
                    "Worker {k} of member {} has a Soul of {} phases; a Soul of this Run has {}",
                    m.id,
                    phases.len(),
                    families.len()
                ));
            }

            for (name, s, _) in &groups {
                lines.push(format!(
                    "{}\t{k}\t{name}\t{}\t{}",
                    m.id,
                    character(phases, s, soul.readout),
                    character_norm(phases, s, soul.readout)
                ));
            }
        }
    }

    Ok(lines.join("\n") + "\n")
}

/// Error unless the search panel rebuilt from the settings is the one that the run start in
/// `events.jsonl` logged. A Run whose archive holds members (`started`) has logged its start.
fn check_panel<D: Domain>(
    dir: &Path,
    domain: &D,
    panel: &[D::Task],
    started: bool,
) -> Result<(), String> {
    let path = dir.join("events.jsonl");
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(_) if !started => return Ok(()),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    // Only the first line: a long Run's log is large.
    let mut first = String::new();

    BufReader::new(file)
        .read_line(&mut first)
        .map_err(|e| format!("{}: {e}", path.display()))?;

    let start = serde_json::from_str::<serde_json::Value>(&first)
        .ok()
        .and_then(|entry| serde_json::from_value::<Event>(entry["event"].clone()).ok());

    match start {
        Some(Event::RunStart { search, .. }) => {
            let ids: Vec<&str> = panel.iter().map(|t| domain.task_id(t)).collect();

            if ids == search {
                Ok(())
            } else {
                Err(format!(
                    "the search panel rebuilt from the settings is not the one {} logged",
                    path.display()
                ))
            }
        }
        _ if !started => Ok(()),
        _ => Err(format!(
            "{} does not start with the run start",
            path.display()
        )),
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;

    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}
