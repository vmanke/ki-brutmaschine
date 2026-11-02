//! The comparative experiment of docs/SELECTION_STUDY.md: does lexicase selection, or a
//! tournament, breed better Cascades than the aggregate Darwin Gödel selection?
//!
//! Every arm breeds the same grid-puzzle task with the same simulated models, the same seed
//! Cascades (generation 0 does not depend on selection), the same operators and rates, the
//! same panels per seed and the same call budget; only `selection` differs. Seed s uses
//! the same panels and the same generation 0 in every arm. The simulated models are a
//! positive control, not language models.
//!
//! Part A compares the arms. Part B asks what a smaller search panel costs in ranking quality
//! (offline, on the archives of part A, and end to end with the finalists scored again on the
//! full panel).
//!
//! Usage:
//!   cargo run --release --example selection_study -- [--seeds N] [--first-seed S]
//!       [--generations G] [--children C] [--search P] [--select S] [--max-calls M]
//!       [--epsilon E] [--tournament-size K] [--out <dir>]
//! Writes `<dir>/report.md` (the tables) and `<dir>/runs.json`: `runs` (the run records of
//! part A), `quality` (ranking quality on subsets), `cheap` (the run records of the subset
//! runs) and `reeval` (their finalists scored again on the full panel).
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;

use brutmaschine::cascade::arena::Breeder;
use brutmaschine::cascade::arena::{BreedConfig, ModelPool, ParentSelection, Snapshot};
use brutmaschine::cascade::domain::{GenePool, GridDomain};
use brutmaschine::cascade::selection::SelectionParams;
use brutmaschine::cascade::sim::SimModel;
use brutmaschine::cascade::study::{
    Dist, RankQuality, Reevaluation, RunRecord, by_arm, dist, reevaluate_finalists, run_study,
    search_score_matrix, subset_rank_quality,
};

const ARMS: [ParentSelection; 4] = [
    ParentSelection::Dgm,
    ParentSelection::Uniform,
    ParentSelection::Lexicase,
    ParentSelection::Tournament,
];

#[derive(Clone, Debug)]
struct Args {
    seeds: u64,
    first_seed: u64,
    base: BreedConfig,
    out: PathBuf,
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut a = Args {
        seeds: 20,
        first_seed: 1,
        base: BreedConfig {
            generations: 1000,
            children_per_generation: 8,
            search_puzzles: 40,
            select_puzzles: 40,
            max_calls: 40_000,
            ..BreedConfig::default()
        },
        out: PathBuf::from("selection-study"),
    };
    let mut it = argv.iter();

    while let Some(k) = it.next() {
        let v = it.next().ok_or_else(|| format!("{k} needs a value"))?;
        let num = |v: &String| {
            v.parse::<f64>()
                .map_err(|_| format!("{k}: not a number: {v}"))
        };
        let int = |v: &String| {
            v.parse::<u64>()
                .map_err(|_| format!("{k}: not a whole number: {v}"))
        };

        match k.as_str() {
            "--seeds" => a.seeds = int(v)?,
            "--first-seed" => a.first_seed = int(v)?,
            "--generations" => a.base.generations = int(v)? as usize,
            "--children" => a.base.children_per_generation = int(v)? as usize,
            "--search" => a.base.search_puzzles = int(v)? as usize,
            "--select" => a.base.select_puzzles = int(v)? as usize,
            "--max-calls" => a.base.max_calls = int(v)?,
            "--epsilon" => a.base.selection_params.lexicase_epsilon = num(v)?,
            "--tournament-size" => a.base.selection_params.tournament_size = int(v)? as usize,
            "--out" => a.out = PathBuf::from(v),
            _ => return Err(format!("unknown option {k}")),
        }
    }

    if a.seeds == 0 {
        return Err("--seeds must be at least 1".to_string());
    }

    a.base.check().map_err(|e| e.to_string())?;

    Ok(a)
}

fn pool() -> ModelPool {
    let mut p = ModelPool::new();

    p.insert("sim:small", Box::new(SimModel::small()));
    p.insert("sim:large", Box::new(SimModel::large()));

    p
}

fn cfg_for(base: &BreedConfig, arm: ParentSelection, seed: u64) -> BreedConfig {
    BreedConfig {
        seed,
        selection: arm,
        selection_params: if arm == ParentSelection::Dgm || arm == ParentSelection::Uniform {
            SelectionParams::default()
        } else {
            base.selection_params
        },
        ..base.clone()
    }
}

fn run_one(
    base: &BreedConfig,
    arm: &str,
    selection: ParentSelection,
    seed: u64,
    pool: &ModelPool,
) -> (RunRecord, Snapshot) {
    let t = Instant::now();
    let (mut rec, _, snap) = run_study::<GridDomain>(
        arm,
        cfg_for(base, selection, seed),
        GenePool::grid(),
        pool,
        true,
        0.0,
    )
    .expect("run");

    rec.wall_seconds = t.elapsed().as_secs_f64();

    (rec, snap)
}

fn cell(d: &Option<Dist>) -> String {
    match d {
        Some(d) => format!(
            "{:.3} [{:.3}, {:.3}] ({:.3}..{:.3})",
            d.median, d.p25, d.p75, d.min, d.max
        ),
        None => "-".to_string(),
    }
}

/// One table: a row per arm, median [p25, p75] (min..max) of `f` over the seeds.
fn table(out: &mut String, title: &str, runs: &[RunRecord], f: impl Fn(&RunRecord) -> f64) {
    let _ = writeln!(
        out,
        "**{title}**\n\n| arm | median [p25, p75] (min..max) |\n|---|---|"
    );

    for (arm, d) in by_arm(runs, f) {
        let _ = writeln!(out, "| {arm} | {} |", cell(&d));
    }

    out.push('\n');
}

/// Per-seed paired differences of `f` against the dgm arm, as "median diff; seeds better /
/// equal / worse". Better means larger, so read `lower_is_better` metrics accordingly.
fn paired(out: &mut String, title: &str, runs: &[RunRecord], f: impl Fn(&RunRecord) -> f64) {
    let mut by: BTreeMap<(String, u64), f64> = BTreeMap::new();

    for r in runs {
        by.insert((r.arm.clone(), r.seed), f(r));
    }

    let _ = writeln!(
        out,
        "**{title}** (arm minus dgm, per seed)\n\n| arm | median diff | larger / equal / smaller |\n|---|---|---|"
    );

    for arm in ARMS.iter().skip(1).map(ParentSelection::name) {
        let mut diffs = Vec::new();

        for ((a, s), v) in &by {
            if a == arm
                && let Some(base) = by.get(&("dgm".to_string(), *s))
            {
                diffs.push(v - base);
            }
        }

        let (up, eq, down) = diffs.iter().fold((0, 0, 0), |(u, e, d), x| {
            if x.abs() <= 1e-12 {
                (u, e + 1, d)
            } else if *x > 0.0 {
                (u + 1, e, d)
            } else {
                (u, e, d + 1)
            }
        });
        let med = dist(&diffs).map_or("-".to_string(), |d| format!("{:+.3}", d.median));

        let _ = writeln!(out, "| {arm} | {med} | {up} / {eq} / {down} |");
    }

    out.push('\n');
}

/// The ranking quality on subsets of one seed's final dgm archive (part B, offline).
#[derive(Debug, Serialize)]
struct SubsetQuality {
    seed: u64,
    #[serde(flatten)]
    quality: RankQuality,
}

/// The finalists of one subset run scored again on the full panel (part B, end to end).
#[derive(Debug, Serialize)]
struct SubsetReevaluation {
    arm: String,
    seed: u64,
    #[serde(flatten)]
    reevaluation: Reevaluation,
}

/// Everything the experiment measured.
#[derive(Debug, Serialize)]
struct Study {
    /// Part A: every arm on every seed.
    runs: Vec<RunRecord>,
    /// Part B, offline.
    quality: Vec<SubsetQuality>,
    /// Part B, end to end: the finalists of each subset run on the full panel.
    reeval: Vec<SubsetReevaluation>,
    /// Part B, end to end: the subset runs.
    cheap: Vec<RunRecord>,
}

fn study(args: &Args) -> Study {
    let pool = pool();
    let seeds: Vec<u64> = (args.first_seed..args.first_seed + args.seeds).collect();
    let mut runs = Vec::new();
    let mut quality = Vec::new();

    for &seed in &seeds {
        let mut dgm_snap = None;

        for arm in ARMS {
            let (rec, snap) = run_one(&args.base, arm.name(), arm, seed, &pool);

            if arm == ParentSelection::Dgm {
                dgm_snap = Some(snap);
            }

            runs.push(rec);
        }

        // Offline: rank the final dgm archive on subsets of its own search panel.
        let cfg = cfg_for(&args.base, ParentSelection::Dgm, seed);
        let snap = dgm_snap.expect("dgm arm ran");
        let b: Breeder<'_, GridDomain> =
            Breeder::resume_from(cfg, GenePool::grid(), &pool, None, snap).expect("resume");
        let scores = search_score_matrix(&b).expect("scores");
        let p = args.base.search_puzzles;

        for k in [p / 8, p / 4, p / 2].into_iter().filter(|&k| k >= 2) {
            if let Some(q) = subset_rank_quality(&scores, k, 200, 3, seed) {
                quality.push(SubsetQuality { seed, quality: q });
            }
        }
    }

    // End to end: the dgm arm on a quarter and a half of the search panel at the same call
    // budget, finalists scored again on the full panel.
    let mut cheap = Vec::new();
    let mut reeval = Vec::new();

    for &seed in &seeds {
        for k in [args.base.search_puzzles / 4, args.base.search_puzzles / 2] {
            if k < 2 {
                continue;
            }

            let arm = format!("dgm-subset-{k}");
            let small = BreedConfig {
                search_puzzles: k,
                ..cfg_for(&args.base, ParentSelection::Dgm, seed)
            };
            let t = Instant::now();
            let (mut rec, report, snap) =
                run_study::<GridDomain>(&arm, small, GenePool::grid(), &pool, true, 0.0)
                    .expect("run");

            rec.wall_seconds = t.elapsed().as_secs_f64();

            let full = cfg_for(&args.base, ParentSelection::Dgm, seed);

            if let Some(re) =
                reevaluate_finalists::<GridDomain>(full, GenePool::grid(), &pool, &snap, &report)
                    .expect("reeval")
            {
                reeval.push(SubsetReevaluation {
                    arm: arm.clone(),
                    seed,
                    reevaluation: re,
                });
            }

            cheap.push(rec);
        }
    }

    Study {
        runs,
        quality,
        reeval,
        cheap,
    }
}

fn report(args: &Args, s: &Study) -> String {
    let mut out = String::new();
    let c = &args.base;
    let _ = writeln!(
        out,
        "# Selection study: results\n\nSeeds {}..{} ({} per arm). Per run: up to {} generations, {} children per generation, \
         {} search and {} held-out tasks, finalists {}, call budget {} (the budget ends every run). \
         Lexicase epsilon {}, tournament size {}. Simulated models (positive control).\n",
        args.first_seed,
        args.first_seed + args.seeds - 1,
        args.seeds,
        c.generations,
        c.children_per_generation,
        c.search_puzzles,
        c.select_puzzles,
        c.finalists,
        c.max_calls,
        c.selection_params.lexicase_epsilon,
        c.selection_params.tournament_size,
    );
    let _ = writeln!(out, "## Part A: arms at one call budget\n");

    table(
        &mut out,
        "Held-out rate of the best bred finalist (select phase)",
        &s.runs,
        |r| r.best_finalist_heldout,
    );
    table(
        &mut out,
        "Held-out rate of the best seed Cascade (same in every arm)",
        &s.runs,
        |r| r.best_seed_heldout,
    );
    table(
        &mut out,
        "Best training rate in the archive",
        &s.runs,
        |r| r.best_train_rate,
    );
    table(
        &mut out,
        "Training rate of the final incumbent",
        &s.runs,
        |r| r.final_train_rate,
    );
    table(
        &mut out,
        "Held-out rate of the final incumbent",
        &s.runs,
        |r| r.final_heldout_rate,
    );
    table(
        &mut out,
        "Generalisation gap of the final incumbent (train - held-out)",
        &s.runs,
        |r| r.final_gap,
    );
    table(&mut out, "Slope of the gap per generation", &s.runs, |r| {
        r.gap_slope
    });
    table(
        &mut out,
        "Distinct behaviours in the final archive",
        &s.runs,
        |r| r.final_behaviours as f64,
    );
    table(&mut out, "Behaviours per archive member", &s.runs, |r| {
        r.behaviour_share
    });
    table(
        &mut out,
        "Mean entropy of the parent draw (bits)",
        &s.runs,
        |r| r.mean_parent_entropy_bits,
    );
    table(&mut out, "Generations completed", &s.runs, |r| {
        r.generations_completed as f64
    });
    table(
        &mut out,
        "Cascades evaluated on the search panel",
        &s.runs,
        |r| r.cascade_evaluations as f64,
    );
    table(&mut out, "Worker calls (budgeted)", &s.runs, |r| {
        r.worker_calls as f64
    });
    table(&mut out, "Completion tokens", &s.runs, |r| {
        r.completion_tokens as f64
    });
    table(&mut out, "Model cost", &s.runs, |r| r.model_cost);
    table(
        &mut out,
        "Wall seconds (including the held-out observer)",
        &s.runs,
        |r| r.wall_seconds,
    );
    table(&mut out, "Champions frozen", &s.runs, |r| {
        r.champions_frozen as f64
    });
    table(
        &mut out,
        "Share of champions in the lineage of the last generation",
        &s.runs,
        |r| r.champion_lineage_alive,
    );
    table(
        &mut out,
        "Share of earlier champions surpassed by the final one",
        &s.runs,
        |r| r.champions_surpassed,
    );
    table(
        &mut out,
        "Rediscoveries (newborn behaves like an earlier member)",
        &s.runs,
        |r| r.rediscoveries as f64,
    );
    table(&mut out, "Champion cycles", &s.runs, |r| {
        r.champion_cycles as f64
    });
    paired(
        &mut out,
        "Held-out rate of the best finalist (picked on the held-out panel, so optimistic)",
        &s.runs,
        |r| r.best_finalist_heldout,
    );
    paired(
        &mut out,
        "Held-out rate of the final incumbent (picked on training fitness alone)",
        &s.runs,
        |r| r.final_heldout_rate,
    );
    paired(&mut out, "Generalisation gap", &s.runs, |r| r.final_gap);
    paired(&mut out, "Distinct behaviours", &s.runs, |r| {
        r.final_behaviours as f64
    });

    let _ = writeln!(
        out,
        "## Part B: evaluation subsets\n\n**Offline: ranking on a random subset of the search tasks against all of them** (final dgm archives, 200 random subsets per archive; top-3 overlap)\n"
    );
    let _ = writeln!(
        out,
        "| subset | cost share | Kendall tau (median over seeds) | top-1 hit rate | top-1 regret | top-3 overlap |\n|---|---|---|---|---|---|"
    );

    let mut ks: Vec<usize> = s.quality.iter().map(|q| q.quality.subset).collect();

    ks.sort_unstable();
    ks.dedup();

    for k in ks {
        let qs: Vec<&RankQuality> = s
            .quality
            .iter()
            .map(|q| &q.quality)
            .filter(|q| q.subset == k)
            .collect();
        let med = |f: &dyn Fn(&RankQuality) -> f64| {
            dist(&qs.iter().map(|q| f(q)).collect::<Vec<_>>()).map_or(f64::NAN, |d| d.median)
        };

        let _ = writeln!(
            out,
            "| {k} of {} | {:.2} | {:.3} | {:.3} | {:.4} | {:.3} |",
            c.search_puzzles,
            qs[0].cost_fraction,
            med(&|q| q.tau_mean),
            med(&|q| q.top1_hit_rate),
            med(&|q| q.top1_regret_mean),
            med(&|q| q.top_n_overlap_mean),
        );
    }

    out.push('\n');

    let all: Vec<RunRecord> = s
        .runs
        .iter()
        .filter(|r| r.arm == "dgm")
        .cloned()
        .chain(s.cheap.iter().cloned())
        .collect();

    let _ = writeln!(
        out,
        "**End to end: dgm on a subset of the search panel at the same call budget** (reference: the full-panel dgm arm)\n"
    );

    table(
        &mut out,
        "Held-out rate of the best bred finalist (select phase: best of the finalists on the held-out panel)",
        &all,
        |r| r.best_finalist_heldout,
    );
    table(&mut out, "Generations completed", &all, |r| {
        r.generations_completed as f64
    });
    table(
        &mut out,
        "Cascades evaluated on the search panel",
        &all,
        |r| r.cascade_evaluations as f64,
    );

    let _ = writeln!(
        out,
        "| arm | seeds | finalist pick changed by full re-evaluation | held-out of subset pick (median) | held-out of full-panel pick (median) | tau subset vs full (median) | re-evaluation calls (median, min..max) |\n|---|---|---|---|---|---|---|"
    );

    let mut arms: Vec<&str> = s.reeval.iter().map(|r| r.arm.as_str()).collect();

    arms.sort_unstable();
    arms.dedup();

    for arm in arms {
        let rs: Vec<&Reevaluation> = s
            .reeval
            .iter()
            .filter(|r| r.arm == arm)
            .map(|r| &r.reevaluation)
            .collect();
        let med = |f: &dyn Fn(&Reevaluation) -> f64| {
            dist(&rs.iter().map(|r| f(r)).collect::<Vec<_>>()).map_or(f64::NAN, |d| d.median)
        };
        // Outside the run's budget, so reported here and not among the worker calls.
        let calls = dist(&rs.iter().map(|r| r.calls as f64).collect::<Vec<_>>())
            .map_or("-".to_string(), |d| {
                format!("{:.0} ({:.0}..{:.0})", d.median, d.min, d.max)
            });

        let _ = writeln!(
            out,
            "| {arm} | {} | {} | {:.3} | {:.3} | {:.3} | {calls} |",
            rs.len(),
            rs.iter().filter(|r| r.pick_changed).count(),
            med(&|r| r.approx_pick_heldout),
            med(&|r| r.full_pick_heldout),
            med(&|r| r.tau),
        );
    }

    out
}

/// The machine-readable output (`runs.json`): everything the study measured.
fn runs_json(s: &Study) -> String {
    serde_json::to_string_pretty(s).expect("json")
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("selection_study: {e}");
            std::process::exit(2);
        }
    };
    let s = study(&args);
    let text = report(&args, &s);

    std::fs::create_dir_all(&args.out).expect("out dir");

    std::fs::write(args.out.join("runs.json"), runs_json(&s)).expect("write runs.json");
    std::fs::write(args.out.join("report.md"), &text).expect("write report.md");
    print!("{text}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Args {
        let argv: Vec<String> = [
            "--seeds",
            "2",
            "--generations",
            "3",
            "--children",
            "3",
            "--search",
            "8",
            "--select",
            "6",
            "--max-calls",
            "100000",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        parse(&argv).expect("args")
    }

    #[test]
    fn arguments_are_checked() {
        assert!(parse(&["--seeds".into(), "0".into()]).is_err());
        assert!(parse(&["--bogus".into(), "1".into()]).is_err());
        assert!(parse(&["--tournament-size".into(), "0".into()]).is_err());
        assert_eq!(tiny().seeds, 2);
    }

    #[test]
    fn the_experiment_is_reproducible_apart_from_wall_time() {
        let a = tiny();
        let strip = |mut s: Study| {
            for r in s.runs.iter_mut().chain(s.cheap.iter_mut()) {
                r.wall_seconds = 0.0;
            }

            report(
                &a,
                &Study {
                    runs: s.runs.clone(),
                    ..s
                },
            )
        };
        let first = strip(study(&a));
        let second = strip(study(&a));

        assert_eq!(first, second);
        assert!(
            first.contains("| lexicase |")
                && first.contains("| tournament |")
                && first.contains("| dgm |")
        );
    }

    #[test]
    fn runs_json_holds_the_down_sampling_results() {
        let s = study(&tiny());
        let json: serde_json::Value = serde_json::from_str(&runs_json(&s)).expect("json");
        let len = |key: &str| json[key].as_array().map(Vec::len);

        assert!(!s.quality.is_empty() && !s.reeval.is_empty());
        assert_eq!(len("runs"), Some(s.runs.len()));
        assert_eq!(len("cheap"), Some(s.cheap.len()));
        assert_eq!(len("quality"), Some(s.quality.len()));
        assert_eq!(len("reeval"), Some(s.reeval.len()));

        let first = &s.reeval[0];

        assert_eq!(json["reeval"][0]["arm"], first.arm.as_str());
        assert_eq!(json["reeval"][0]["seed"], first.seed);
        assert_eq!(json["reeval"][0]["calls"], first.reevaluation.calls);
        assert_eq!(json["quality"][0]["subset"], s.quality[0].quality.subset);
    }

    #[test]
    fn the_report_shows_the_re_evaluation_calls() {
        let a = tiny();
        let s = study(&a);
        let text = report(&a, &s);

        assert!(text.contains("| re-evaluation calls (median, min..max) |"));

        for arm in ["dgm-subset-2", "dgm-subset-4"] {
            let calls: Vec<f64> = s
                .reeval
                .iter()
                .filter(|r| r.arm == arm)
                .map(|r| r.reevaluation.calls as f64)
                .collect();
            let d = dist(&calls).expect("re-evaluated seeds");
            let cell = format!("| {:.0} ({:.0}..{:.0}) |", d.median, d.min, d.max);
            let row = text
                .lines()
                .find(|l| l.starts_with(&format!("| {arm} | {} |", calls.len())))
                .expect("a row per subset arm");

            assert!(d.max > 0.0 && row.ends_with(&cell), "{row}");
        }
    }

    #[test]
    fn every_arm_starts_from_the_same_generation_zero() {
        let a = tiny();
        let s = study(&a);

        for seed in a.first_seed..a.first_seed + a.seeds {
            let g0: Vec<_> = s
                .runs
                .iter()
                .filter(|r| r.seed == seed)
                .map(|r| {
                    (
                        r.generations[0].incumbent.clone(),
                        r.generations[0].train_rate.to_bits(),
                    )
                })
                .collect();

            assert_eq!(g0.len(), ARMS.len());
            assert!(g0.windows(2).all(|w| w[0] == w[1]));
        }
    }
}
