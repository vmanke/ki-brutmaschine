//! Learner-run report: report.json and the console table (orchestrator side).
//!
//! Owner: WP5. Contract section 5.20. Does file I/O.
//!
//! Everything comes from the archive, the files its digests anchor (lineage chunks, the
//! latest snapshot) and deterministic recomputation; nothing comes from the hidden side
//! except the released numbers recorded in `LearnerHiddenEval`.
#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::archive::{Archive, CostPhase, LearnerEntry, LearnerStatus, Record, RecordKind};
use crate::evolve::ARCHIVE_FILE;
use crate::learner::seeds;
use crate::learner::types::{
    Abort, BaselineId, Decision, EvalSet, LearnerHiddenResult, MetaConfig, Tier, TunedBaselines,
};
use crate::lineage_store::{ChunkStats, LineageRecord, read_chunk, read_snapshot};
use crate::meta_evolve::{self, FinalistMetrics, baseline_name, mutation_name, ratio};
use crate::util::{BrutError, Result, write_atomic};

/// File name of the report inside the run directory.
pub const REPORT_FILE: &str = "report.json";

/// Δ_HI must exceed this before a transfer ratio Δ_GH / Δ_HI is reported.
pub const TRANSFER_MIN_DELTA: f64 = 0.01;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub protocol_hash: String,
    pub config: MetaConfig,
    pub search: SearchHealth,
    /// Costs outside the search budget (errata E6), from the archive's LearnerCost records.
    pub one_time: OneTimeCosts,
    pub finalists: Vec<FinalistReport>,
    pub baselines: TunedBaselines,
    pub external_arc: Vec<ExternalArcRow>,
    pub reference_lines: ReferenceLines,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchHealth {
    pub batches: u64,
    pub children: u64,
    /// Task runs really executed (not served from the score cache).
    pub unique_evals: u64,
    pub fec_task_hit_rate: f64,
    pub hurdle_pass_rate: f64,
    pub cost_rejected: u64,
    pub aborts: Vec<(Abort, u64)>,
    pub population_unique_fps: usize,
    /// Search-phase FLOPs, Stage C included (errata E6; one-time costs: `Report::one_time`).
    pub flops_used: u64,
    /// The part of `flops_used` spent in periodic Stage C rounds.
    pub stage_c_flops: u64,
    /// Sum over `brut run` invocations of the time between their first and last record.
    pub wall_seconds: f64,
}

/// One-time FLOPs of a run, summed over its invocations (static counts, like the search).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OneTimeCosts {
    pub baseline_tuning_flops: u64,
    pub final_phase_flops: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FinalistReport {
    pub id: String,
    pub effective_len: usize,
    pub step_cost_f16: u64,
    /// render() of simplify().
    pub program: String,
    pub panel_fitness: f64,
    pub delta_select: f64,
    /// Coupling, public.
    pub delta_select_f32: f64,
    pub delta_select_f8_16: f64,
    /// Δ_public (quarantine input).
    pub delta_public_instance: f64,
    /// Panel M(g) - mean AULC on public_reference_tasks.
    pub winners_curse: f64,
    /// On SELECT.
    pub q_by_checkpoint: [f64; 3],
    /// On SELECT.
    pub q_by_n_train: Vec<(usize, f64)>,
    pub hidden: Vec<(Tier, f64, Option<Decision>)>,
    /// Δ_GH / Δ_HI when Δ_HI > 0.01.
    pub transfer_ratio: Option<f64>,
    /// The strongest compute-matched baseline b* on SELECT and on the public reference
    /// tasks (the hidden b* is never released).
    pub best_baseline_select: Option<BaselineId>,
    pub best_baseline_public: Option<BaselineId>,
    pub quarantined: bool,
    /// Seed name it descends from.
    pub lineage_seed: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExternalArcRow {
    pub id: String,
    pub set: EvalSet,
    pub official: f64,
    pub kaggle: f64,
    pub bca: Option<f64>,
}

/// Fixed lines measured on the dev machine (plan section 6).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceLines {
    /// (A1Eval 9/400), (A2Eval 0/120), (ConceptArc 17/160).
    pub phi32_ceiling: [(EvalSet, usize, usize); 3],
    /// 0.289.
    pub icecuber_a1_pass2_outputs: f64,
    /// (1, 120).
    pub icecuber_a2_tasks: (usize, usize),
}

impl Default for ReferenceLines {
    fn default() -> Self {
        ReferenceLines {
            phi32_ceiling: [
                (EvalSet::A1Eval, 9, 400),
                (EvalSet::A2Eval, 0, 120),
                (EvalSet::ConceptArc, 17, 160),
            ],
            icecuber_a1_pass2_outputs: 0.289,
            icecuber_a2_tasks: (1, 120),
        }
    }
}

/// Build from archive + lineage; write <run_dir>/report.json atomically; return it.
///
/// The finalist numbers on SELECT and on the public reference tasks are recomputed
/// (deterministic); the hidden numbers are the released ones from the archive.
pub fn write_report(run_dir: &Path) -> Result<Report> {
    let archive = Archive::open(&run_dir.join(ARCHIVE_FILE))?;

    let Some(cfg) = archive.latest_meta_config().cloned() else {
        return Err(BrutError::Invalid(format!(
            "{} is not a learner run",
            run_dir.display()
        )));
    };

    let Some(tb) = meta_evolve::load_baselines(run_dir, &archive)? else {
        return Err(BrutError::Invalid(
            "no tuned baselines yet: start the run with brut run first".into(),
        ));
    };

    let finalists = meta_evolve::current_finalists(&archive);
    let metrics = meta_evolve::recompute_metrics(run_dir, &cfg, &tb, &finalists)?;

    write_report_with(run_dir, &archive, &cfg, &tb, &finalists, &metrics)
}

/// Build the report from known finalist metrics and write it (used at the end of a run).
pub(crate) fn write_report_with(
    run_dir: &Path,
    archive: &Archive,
    cfg: &MetaConfig,
    tb: &TunedBaselines,
    finalists: &[LearnerEntry],
    metrics: &[FinalistMetrics],
) -> Result<Report> {
    let seed_names: HashMap<String, String> = seeds::seed_programs()
        .into_iter()
        .map(|(name, g)| (g.id(), name.to_string()))
        .collect();

    let lineage = lineage_index(run_dir, archive)?;
    let search = search_health(run_dir, archive)?;
    let hidden = archive.learner_hidden_evals();

    let finalist_rows = finalists
        .iter()
        .zip(metrics)
        .map(|(e, m)| {
            let seed = lineage_seed(&e.id, archive, &lineage, &seed_names);
            let program = e.genome.simplify().render();

            finalist_report(e, m, &hidden, seed, program)
        })
        .collect();

    let report = Report {
        protocol_hash: archive.protocol_hash().unwrap_or_default().to_string(),
        config: cfg.clone(),
        search,
        one_time: one_time_costs(archive.records()),
        finalists: finalist_rows,
        baselines: tb.clone(),
        external_arc: external_rows(archive.records()),
        reference_lines: ReferenceLines::default(),
    };

    // Invariant I8: the JSON must parse back to the same report (no NaN or infinity).
    let bytes = serde_json::to_vec_pretty(&report)?;
    let back: std::result::Result<Report, _> = serde_json::from_slice(&bytes);

    if back.ok().as_ref() != Some(&report) {
        return Err(BrutError::Invalid("report: a number is not finite".into()));
    }

    let mut bytes = bytes;

    bytes.push(b'\n');
    write_atomic(&run_dir.join(REPORT_FILE), &bytes)?;

    Ok(report)
}

/// Search counters over all digests, plus the latest snapshot's population.
fn search_health(run_dir: &Path, archive: &Archive) -> Result<SearchHealth> {
    let digests = archive.batch_digests();
    let mut total = ChunkStats::default();

    for d in &digests {
        total.add(&d.stats);
    }

    let (batches, unique_fps, flops_used) = match digests.last() {
        Some(d) => {
            let snap = read_snapshot(&run_dir.join(&d.snapshot_file), &d.snapshot_sha256)?;
            let fps: HashSet<&str> = snap.population.iter().map(|m| m.fp.as_str()).collect();

            (d.last_batch + 1, fps.len(), snap.flops_used)
        }
        None => (0, 0, 0),
    };

    Ok(health_from(
        &total,
        batches,
        unique_fps,
        flops_used,
        wall_seconds(archive.records()),
    ))
}

/// SearchHealth from summed counters (pure).
fn health_from(
    total: &ChunkStats,
    batches: u64,
    unique_fps: usize,
    flops_used: u64,
    wall_seconds: f64,
) -> SearchHealth {
    let lookups = total.unique_evals.saturating_add(total.fec_task_hits);
    let passed = total.children.saturating_sub(total.hurdled);

    SearchHealth {
        batches,
        children: total.children,
        unique_evals: total.unique_evals,
        fec_task_hit_rate: ratio(total.fec_task_hits, lookups),
        hurdle_pass_rate: ratio(passed, total.children),
        cost_rejected: total.cost_rejected,
        aborts: vec![
            (Abort::NonFinite, total.aborts_nonfinite),
            (Abort::ErrorTooLarge, total.aborts_error),
            (Abort::Timeout, total.aborts_timeout),
        ],
        population_unique_fps: unique_fps,
        flops_used,
        stage_c_flops: total.stage_c_flops,
        wall_seconds,
    }
}

/// Sum of the `LearnerCost` records per phase.
fn one_time_costs(records: &[Record]) -> OneTimeCosts {
    let mut costs = OneTimeCosts::default();

    for r in records {
        if let RecordKind::LearnerCost { phase, flops } = r.kind {
            let slot = match phase {
                CostPhase::BaselineTuning => &mut costs.baseline_tuning_flops,
                CostPhase::FinalPhase => &mut costs.final_phase_flops,
            };

            *slot = slot.saturating_add(flops);
        }
    }

    costs
}

/// Sum over invocations of last minus first unix_ms. An invocation starts with
/// `MetaRunStarted` and ends at `RunFinished`; records after that (for example a later
/// ARC check) do not count.
fn wall_seconds(records: &[Record]) -> f64 {
    let mut total_ms = 0u64;
    let mut segment: Option<(u64, u64)> = None;

    for r in records {
        let t = r.unix_ms;

        match r.kind {
            RecordKind::MetaRunStarted { .. } => {
                if let Some((first, last)) = segment {
                    total_ms = total_ms.saturating_add(last.saturating_sub(first));
                }

                segment = Some((t, t));
            }
            RecordKind::RunFinished { .. } => {
                if let Some((first, last)) = segment {
                    total_ms = total_ms.saturating_add(last.max(t).saturating_sub(first));
                }

                segment = None;
            }
            _ => {
                segment = segment.map(|(first, last)| (first, last.max(t)));
            }
        }
    }

    if let Some((first, last)) = segment {
        total_ms = total_ms.saturating_add(last.saturating_sub(first));
    }

    total_ms as f64 / 1000.0
}

/// One finalist row (pure); `program` is the rendered simplified program.
fn finalist_report(
    e: &LearnerEntry,
    m: &FinalistMetrics,
    hidden: &[LearnerHiddenResult],
    lineage_seed: String,
    program: String,
) -> FinalistReport {
    let mine: Vec<&LearnerHiddenResult> =
        hidden.iter().filter(|h| h.candidate_id == e.id).collect();

    let delta_of = |tier: Tier| mine.iter().find(|h| h.tier == tier).map(|h| h.delta);

    let transfer_ratio = match (
        delta_of(Tier::HiddenGrammar),
        delta_of(Tier::HiddenInstance),
    ) {
        (Some(gh), Some(hi)) if hi > TRANSFER_MIN_DELTA => Some(gh / hi),
        _ => None,
    };

    let quarantined = e.status == LearnerStatus::Quarantined || mine.iter().any(|h| h.quarantined);

    FinalistReport {
        id: e.id.clone(),
        effective_len: e.effective_len,
        step_cost_f16: e.step_cost_f16,
        program,
        panel_fitness: e.fitness,
        delta_select: m.delta_select,
        delta_select_f32: m.delta_select_f32,
        delta_select_f8_16: m.delta_select_f8_16,
        delta_public_instance: m.delta_public,
        winners_curse: e.panel_mean - m.public_mean,
        q_by_checkpoint: m.q_by_checkpoint,
        q_by_n_train: m.q_by_n_train.clone(),
        hidden: mine.iter().map(|h| (h.tier, h.delta, h.decision)).collect(),
        transfer_ratio,
        best_baseline_select: m.best_select,
        best_baseline_public: m.best_public,
        quarantined,
        lineage_seed,
    }
}

/// Every `ExternalEval` record, in archive order.
fn external_rows(records: &[Record]) -> Vec<ExternalArcRow> {
    records
        .iter()
        .filter_map(|r| match &r.kind {
            RecordKind::ExternalEval {
                candidate_id,
                set,
                official,
                kaggle,
                bca,
            } => Some(ExternalArcRow {
                id: candidate_id.clone(),
                set: *set,
                official: *official,
                kaggle: *kaggle,
                bca: *bca,
            }),
            _ => None,
        })
        .collect()
}

/// Lineage records of all digests by id (the first record of an id wins).
pub fn lineage_index(run_dir: &Path, archive: &Archive) -> Result<HashMap<String, LineageRecord>> {
    let mut index = HashMap::new();

    for d in archive.batch_digests() {
        for r in read_chunk(&run_dir.join(&d.chunk_file), &d.chunk_sha256)? {
            index.entry(r.id.clone()).or_insert(r);
        }
    }

    Ok(index)
}

/// One step of a learner lineage.
#[derive(Clone, Debug, PartialEq)]
pub struct Ancestor {
    pub id: String,
    pub parent: Option<String>,
    pub batch: u64,
    /// The mutation (lineage chunk) or the generator (archive entry).
    pub how: String,
    pub fitness: f64,
}

/// `id` and its ancestors, newest first, from lineage chunks and archive entries.
/// Stops at a program without parent or at a repeat.
pub fn ancestors(
    id: &str,
    archive: &Archive,
    lineage: &HashMap<String, LineageRecord>,
) -> Vec<Ancestor> {
    let mut out = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut next = Some(id.to_string());

    while let Some(current) = next.take() {
        if !seen.insert(current.clone()) {
            break;
        }

        let step = match (lineage.get(&current), archive.learner_get(&current)) {
            (Some(r), entry) => Ancestor {
                id: current.clone(),
                parent: r.parent.clone(),
                batch: r.batch,
                how: match (r.mutation, entry) {
                    (Some(m), _) => mutation_name(m).to_string(),
                    (None, Some(e)) => e.generator,
                    (None, None) => "init".to_string(),
                },
                fitness: r.fitness,
            },
            (None, Some(e)) => Ancestor {
                id: current.clone(),
                parent: e.parents.first().cloned(),
                batch: e.batch,
                how: e.generator.clone(),
                fitness: e.fitness,
            },
            (None, None) => break,
        };

        next = step.parent.clone();
        out.push(step);
    }

    out
}

/// Name of the seed a program descends from ("unknown" when the chain does not reach one).
fn lineage_seed(
    id: &str,
    archive: &Archive,
    lineage: &HashMap<String, LineageRecord>,
    seed_names: &HashMap<String, String>,
) -> String {
    let mut chain = vec![id.to_string()];

    chain.extend(
        ancestors(id, archive, lineage)
            .into_iter()
            .filter_map(|a| a.parent),
    );

    chain
        .iter()
        .find_map(|i| seed_names.get(i).cloned())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Plain-text table for `brut learner report`.
pub fn render_table(r: &Report) -> String {
    let mut out = String::new();
    let s = &r.search;

    let _ = writeln!(out, "learner report  protocol {}", r.protocol_hash);
    let _ = writeln!(
        out,
        "search    batches {} | children {} | task runs {} | cache hit rate {:.3} | hurdle pass {:.3} | cost rejected {}",
        s.batches,
        s.children,
        s.unique_evals,
        s.fec_task_hit_rate,
        s.hurdle_pass_rate,
        s.cost_rejected
    );

    let aborts: Vec<String> = s.aborts.iter().map(|(a, n)| format!("{a:?} {n}")).collect();

    let _ = writeln!(
        out,
        "          aborts {} | distinct fingerprints in population {} | search flops {:.3e} (Stage C {:.3e}) | wall {:.1} s",
        aborts.join(", "),
        s.population_unique_fps,
        s.flops_used as f64,
        s.stage_c_flops as f64,
        s.wall_seconds
    );
    let _ = writeln!(
        out,
        "one-time  baseline tuning {:.3e} flops | final phase {:.3e} flops (not in the search budget)",
        r.one_time.baseline_tuning_flops as f64, r.one_time.final_phase_flops as f64
    );

    let b = &r.baselines;
    let programs: Vec<String> = b
        .programs
        .iter()
        .map(|(id, g)| format!("{} {}", baseline_name(*id), g.id()))
        .collect();

    let _ = writeln!(
        out,
        "baselines {} | mlp hidden {} {:?} lr {} l2 {}",
        programs.join(", "),
        b.mlp.hidden,
        b.mlp.activation,
        b.mlp.lr,
        b.mlp.l2
    );

    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "{:<16}  {:>4}  {:>7}  {:>8}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  seed",
        "finalist", "len", "cost16", "fitness", "dsel", "dsel32", "dsel816", "dpub", "curse"
    );

    if r.finalists.is_empty() {
        let _ = writeln!(out, "(no finalists yet)");
    }

    for f in &r.finalists {
        let _ = writeln!(
            out,
            "{:<16}  {:>4}  {:>7}  {:>8.4}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}  {}",
            f.id,
            f.effective_len,
            f.step_cost_f16,
            f.panel_fitness,
            f.delta_select,
            f.delta_select_f32,
            f.delta_select_f8_16,
            f.delta_public_instance,
            f.winners_curse,
            f.lineage_seed
        );

        let q = f.q_by_checkpoint;
        let by_n: Vec<String> = f
            .q_by_n_train
            .iter()
            .map(|(n, v)| format!("n={n} {v:.3}"))
            .collect();

        let _ = writeln!(
            out,
            "    q by checkpoint {:.3} / {:.3} / {:.3} | q by n_train {}",
            q[0],
            q[1],
            q[2],
            by_n.join(", ")
        );

        let hidden: Vec<String> = f
            .hidden
            .iter()
            .map(|(t, d, dec)| match dec {
                Some(dec) => format!("{t:?} {d:.2} ({dec:?})"),
                None => format!("{t:?} {d:.2}"),
            })
            .collect();

        let hidden_text = if hidden.is_empty() {
            "none yet".to_string()
        } else {
            hidden.join(", ")
        };

        let transfer = f
            .transfer_ratio
            .map_or(String::new(), |t| format!(" | transfer {t:.2}"));

        let quarantine = if f.quarantined { " | QUARANTINED" } else { "" };

        let _ = writeln!(out, "    hidden {hidden_text}{transfer}{quarantine}");

        for line in f.program.lines() {
            let _ = writeln!(out, "      {line}");
        }
    }

    if !r.external_arc.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "external ARC checks (pass@2)");

        for a in &r.external_arc {
            let bca = a.bca.map_or("-".to_string(), |v| format!("{v:.2}"));

            let _ = writeln!(
                out,
                "  {:<16}  {:?}  official {:.4}  kaggle {:.4}  balanced cell accuracy {bca}",
                a.id, a.set, a.official, a.kaggle
            );
        }
    }

    let l = &r.reference_lines;
    let ceiling: Vec<String> = l
        .phi32_ceiling
        .iter()
        .map(|(set, k, n)| format!("{set:?} {k}/{n}"))
        .collect();

    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "reference lines: phi32 cell-learner ceiling {} | icecuber pass@2 A1 {:.1} % of test outputs, A2 {}/{} tasks",
        ceiling.join(", "),
        100.0 * l.icecuber_a1_pass2_outputs,
        l.icecuber_a2_tasks.0,
        l.icecuber_a2_tasks.1
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archive::GENESIS_HASH;
    use crate::genome::Activation;
    use crate::learner::types::{LearnerGenome, MlpRecipe, Mutation};

    fn record(seq: u64, unix_ms: u64, kind: RecordKind) -> Record {
        Record {
            seq,
            unix_ms,
            prev_hash: GENESIS_HASH.into(),
            kind,
            hash: String::new(),
        }
    }

    fn started() -> RecordKind {
        RecordKind::MetaRunStarted {
            config: Box::new(MetaConfig::default()),
            protocol_hash: String::new(),
            crate_version: "test".into(),
        }
    }

    fn event() -> RecordKind {
        RecordKind::SafetyEvent { what: "x".into() }
    }

    #[test]
    fn wall_time_sums_invocations() {
        // init (1 s), first run 10..25 s, second run 100..103 s.
        let records = vec![
            record(0, 0, started()),
            record(1, 1_000, event()),
            record(2, 10_000, started()),
            record(3, 25_000, event()),
            record(4, 100_000, started()),
            record(5, 101_000, event()),
            record(6, 103_000, event()),
        ];

        assert_eq!(wall_seconds(&records), 19.0);

        // A record after RunFinished (a later ARC check) adds nothing.
        let finished = RecordKind::RunFinished {
            reason: "done".into(),
        };
        let later = vec![
            record(0, 0, started()),
            record(1, 4_000, finished),
            record(2, 60_000, event()),
        ];

        assert_eq!(wall_seconds(&later), 4.0);
        assert_eq!(wall_seconds(&[]), 0.0);
        assert_eq!(wall_seconds(&[record(0, 5, event())]), 0.0);
    }

    /// SCI-R5: the one-time costs reach the report, summed per phase over invocations.
    #[test]
    fn one_time_costs_are_summed_per_phase() {
        let cost = |phase, flops| RecordKind::LearnerCost { phase, flops };
        let records = vec![
            record(0, 0, started()),
            record(1, 1, cost(CostPhase::BaselineTuning, 30)),
            record(2, 2, event()),
            record(3, 3, cost(CostPhase::FinalPhase, 7)),
            record(4, 4, cost(CostPhase::FinalPhase, 5)),
        ];

        assert_eq!(
            one_time_costs(&records),
            OneTimeCosts {
                baseline_tuning_flops: 30,
                final_phase_flops: 12,
            }
        );
        assert_eq!(one_time_costs(&[]), OneTimeCosts::default());
    }

    #[test]
    fn health_rates() {
        let total = ChunkStats {
            children: 10,
            unique_evals: 30,
            fec_task_hits: 10,
            hurdled: 4,
            cost_rejected: 1,
            aborts_nonfinite: 2,
            aborts_error: 3,
            aborts_timeout: 0,
            flops: 99,
            stage_c_flops: 40,
        };

        let h = health_from(&total, 7, 5, 1234, 2.5);

        assert_eq!(h.batches, 7);
        assert_eq!(h.fec_task_hit_rate, 0.25);
        assert_eq!(h.hurdle_pass_rate, 0.6);
        assert_eq!(h.cost_rejected, 1);
        assert_eq!(
            h.aborts,
            vec![
                (Abort::NonFinite, 2),
                (Abort::ErrorTooLarge, 3),
                (Abort::Timeout, 0)
            ]
        );
        assert_eq!(h.population_unique_fps, 5);
        assert_eq!(h.flops_used, 1234);
        assert_eq!(h.stage_c_flops, 40);

        let empty = health_from(&ChunkStats::default(), 0, 0, 0, 0.0);

        assert_eq!(empty.fec_task_hit_rate, 0.0);
        assert_eq!(empty.hurdle_pass_rate, 0.0);
    }

    fn entry(id: &str, parents: &[&str], generator: &str, status: LearnerStatus) -> LearnerEntry {
        LearnerEntry {
            id: id.into(),
            parents: parents.iter().map(|p| p.to_string()).collect(),
            generator: generator.into(),
            batch: 50,
            genome: LearnerGenome::default(),
            effective_len: 12,
            step_cost_f16: 2_640,
            fp: "0000000000000001".into(),
            panel_mean: 0.6,
            fitness: 0.55,
            program_file: Some(format!("programs/{id}.json")),
            status,
        }
    }

    fn metrics() -> FinalistMetrics {
        FinalistMetrics {
            select_mean: 0.5,
            delta_select: 0.04,
            delta_select_f32: 0.03,
            delta_select_f8_16: 0.05,
            q_by_checkpoint: [0.2, 0.4, 0.6],
            q_by_n_train: vec![(16, 0.3), (64, 0.5), (256, 0.7)],
            delta_public: 0.06,
            public_mean: 0.45,
            best_select: Some(BaselineId::LinearSgd),
            best_public: Some(BaselineId::MlpMatched),
        }
    }

    fn hidden(id: &str, tier: Tier, delta: f64, quarantined: bool) -> LearnerHiddenResult {
        LearnerHiddenResult {
            candidate_id: id.into(),
            tier,
            delta,
            decision: (tier == Tier::HiddenGrammar).then_some(Decision::Pass),
            quarantined,
        }
    }

    #[test]
    fn finalist_rows() {
        let e = entry("f", &["p"], "retune", LearnerStatus::Finalist);
        let all = vec![
            hidden("f", Tier::HiddenInstance, 0.04, false),
            hidden("other", Tier::HiddenInstance, 0.5, true),
            hidden("f", Tier::HiddenGrammar, 0.02, false),
        ];

        let row = finalist_report(&e, &metrics(), &all, "nn2-sgd".into(), "prog".into());

        assert_eq!(row.id, "f");
        assert_eq!(row.program, "prog");
        assert_eq!(row.panel_fitness, 0.55);
        assert!((row.winners_curse - 0.15).abs() < 1e-12);
        assert_eq!(row.delta_public_instance, 0.06);
        assert_eq!(row.transfer_ratio, Some(0.5));
        assert!(!row.quarantined);
        assert_eq!(
            row.hidden,
            vec![
                (Tier::HiddenInstance, 0.04, None),
                (Tier::HiddenGrammar, 0.02, Some(Decision::Pass))
            ]
        );

        // No ratio for a tiny Δ_HI; quarantine from the record or the status.
        let small = vec![
            hidden("f", Tier::HiddenInstance, 0.01, true),
            hidden("f", Tier::HiddenGrammar, 0.02, true),
        ];
        let row = finalist_report(&e, &metrics(), &small, "x".into(), "p".into());

        assert_eq!(row.transfer_ratio, None);
        assert!(row.quarantined);

        let q = entry("f", &["p"], "retune", LearnerStatus::Quarantined);

        assert!(finalist_report(&q, &metrics(), &[], "x".into(), "p".into()).quarantined);
    }

    fn lineage_record(
        id: &str,
        parent: Option<&str>,
        batch: u64,
        m: Option<Mutation>,
    ) -> LineageRecord {
        LineageRecord {
            id: id.into(),
            parent: parent.map(str::to_string),
            batch,
            slot: 0,
            mutation: m,
            fp: "0000000000000001".into(),
            fitness: 0.1 * batch as f64,
            cached_tasks: 0,
        }
    }

    #[test]
    fn ancestry_reaches_the_seed() {
        let dir = std::env::temp_dir().join(format!("brut-report-lineage-{}", std::process::id()));

        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");

        let mut archive = Archive::create(&dir.join(ARCHIVE_FILE)).expect("create");

        archive
            .append(
                RecordKind::LearnerCandidate(entry(
                    "fin",
                    &["c2"],
                    "retune",
                    LearnerStatus::Finalist,
                )),
                1,
            )
            .expect("append");

        let mut index = HashMap::new();

        for r in [
            lineage_record("seed1", None, 0, None),
            lineage_record("init5", Some("seed1"), 0, None),
            lineage_record("c1", Some("init5"), 3, Some(Mutation::Arg)),
            lineage_record("c2", Some("c1"), 7, Some(Mutation::Const)),
        ] {
            index.insert(r.id.clone(), r);
        }

        let steps = ancestors("fin", &archive, &index);
        let ids: Vec<&str> = steps.iter().map(|a| a.id.as_str()).collect();
        let how: Vec<&str> = steps.iter().map(|a| a.how.as_str()).collect();

        assert_eq!(ids, vec!["fin", "c2", "c1", "init5", "seed1"]);
        assert_eq!(how, vec!["retune", "const", "arg", "init", "init"]);
        assert_eq!(steps[1].batch, 7);

        let seeds: HashMap<String, String> =
            [("seed1".to_string(), "prototype".to_string())].into();

        assert_eq!(lineage_seed("fin", &archive, &index, &seeds), "prototype");
        assert_eq!(lineage_seed("c1", &archive, &index, &seeds), "prototype");
        assert_eq!(lineage_seed("seed1", &archive, &index, &seeds), "prototype");
        assert_eq!(
            lineage_seed("nope", &archive, &index, &HashMap::new()),
            "unknown"
        );
        assert!(ancestors("nope", &archive, &index).is_empty());

        // A loop (a program that re-created an ancestor) ends the walk.
        index.insert(
            "x".into(),
            lineage_record("x", Some("y"), 1, Some(Mutation::Insert)),
        );
        index.insert(
            "y".into(),
            lineage_record("y", Some("x"), 2, Some(Mutation::Remove)),
        );

        assert_eq!(ancestors("x", &archive, &index).len(), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn report() -> Report {
        let e = entry("f", &["p"], "retune", LearnerStatus::Finalist);
        let hidden = vec![
            hidden("f", Tier::HiddenInstance, 0.04, false),
            hidden("f", Tier::HiddenGrammar, 0.02, false),
        ];

        Report {
            protocol_hash: "e".repeat(64),
            config: MetaConfig::default(),
            search: health_from(&ChunkStats::default(), 3, 2, 10, 1.0),
            one_time: OneTimeCosts::default(),
            finalists: vec![finalist_report(
                &e,
                &metrics(),
                &hidden,
                "nn2-sgd".into(),
                "MatVec v1 <- m0 . v0\n".into(),
            )],
            baselines: TunedBaselines {
                programs: Vec::new(),
                mlp: MlpRecipe {
                    hidden: 16,
                    activation: Activation::Relu,
                    lr: 0.01,
                    l2: 0.0,
                },
                hash: "h".repeat(64),
            },
            external_arc: vec![ExternalArcRow {
                id: "f".into(),
                set: EvalSet::A1Eval,
                official: 0.0025,
                kaggle: 0.002,
                bca: Some(0.51),
            }],
            reference_lines: ReferenceLines::default(),
        }
    }

    #[test]
    fn report_json_and_table() {
        let r = report();
        let json = serde_json::to_string(&r).expect("json");
        let back: Report = serde_json::from_str(&json).expect("parse");

        assert_eq!(back, r);
        assert!(json.contains(
            "\"phi32_ceiling\":[[\"a1_eval\",9,400],[\"a2_eval\",0,120],[\"concept_arc\",17,160]]"
        ));
        assert!(json.contains("\"icecuber_a2_tasks\":[1,120]"));

        let table = render_table(&r);

        assert!(table.contains("learner report  protocol eeee"));
        assert!(table.contains("HiddenGrammar 0.02 (Pass)"));
        assert!(table.contains("transfer 0.50"));
        assert!(table.contains("MatVec v1 <- m0 . v0"));
        assert!(table.contains("A1Eval  official 0.0025"));
        assert!(table.contains("A1Eval 9/400"));
        assert!(table.contains("28.9 %"));
        assert!(!table.contains("QUARANTINED"));

        let mut empty = report();

        empty.finalists.clear();
        empty.external_arc.clear();

        let table = render_table(&empty);

        assert!(table.contains("(no finalists yet)"));
        assert!(!table.contains("external ARC"));
    }

    #[test]
    fn external_rows_in_archive_order() {
        let kind = |id: &str, set: EvalSet| RecordKind::ExternalEval {
            candidate_id: id.into(),
            set,
            official: 0.1,
            kaggle: 0.2,
            bca: None,
        };
        let records = vec![
            record(0, 0, started()),
            record(1, 1, kind("a", EvalSet::A2Eval)),
            record(2, 2, event()),
            record(3, 3, kind("b", EvalSet::ConceptArc)),
        ];

        let rows = external_rows(&records);

        assert_eq!(rows.len(), 2);
        assert_eq!((rows[0].id.as_str(), rows[0].set), ("a", EvalSet::A2Eval));
        assert_eq!(
            (rows[1].id.as_str(), rows[1].set),
            ("b", EvalSet::ConceptArc)
        );
    }
}
