//! Measurement of parent-selection strategies (docs/SELECTION_STUDY.md).
//!
//! An *observer* around the breeder: it steps a `Breeder`, reads its committed archive and
//! never writes to it. That is what keeps the held-out panel out of selection, mutation and
//! recombination: the observer scores the incumbent of every generation on the select
//! panel and records the result here, in structures the breeder cannot see. A test breeds
//! the same run with and without that scoring and compares the archives byte for byte.
//!
//! * **Generalisation gap**: training rate minus held-out rate of the incumbent (the best
//!   member by fitness, the one a user would take), for every generation.
//! * **Champion vault**: every incumbent that replaces the previous one is frozen with its
//!   Cascade, its answers and its held-out values. Later candidates are compared with the
//!   frozen values task by task. The vault has no way to change a champion.
//! * **Rediscovery and cycling**: a behaviour signature (hash of the answers on the search
//!   panel) shows when a new Cascade behaves exactly like an earlier one.
//! * **Down-sampling**: how well the ranking on a random subset of the tasks agrees with the
//!   ranking on all of them.
//!
//! Pure computation, like the rest of `cascade`: wall time is measured by the caller.
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap};

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::cascade::arena::{
    BreedConfig, Breeder, Event, Finalist, Member, ModelPool, Report, Snapshot, Status,
    checked_grade, run_cascade_in,
};
use crate::cascade::domain::{Domain, GenePool};
use crate::cascade::genome::Cascade;
use crate::cascade::selection::TIE_SLACK;
use crate::util::{BrutError, Result, derive_seed, sha256_hex};

/// Five quantiles, the mean and the size of a set of numbers. Quantiles are
/// sorted[(n - 1) k / 4], as in `observe::Quantiles`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dist {
    pub n: usize,
    pub min: f64,
    pub p25: f64,
    pub median: f64,
    pub p75: f64,
    pub max: f64,
    pub mean: f64,
}

/// The distribution of `xs`, None when empty. Non-finite values are refused (None), since
/// they would reach a report as null.
pub fn dist(xs: &[f64]) -> Option<Dist> {
    if xs.is_empty() || xs.iter().any(|x| !x.is_finite()) {
        return None;
    }

    let mut v = xs.to_vec();

    v.sort_by(f64::total_cmp);

    let q = |k: usize| v[(v.len() - 1) * k / 4];

    Some(Dist {
        n: v.len(),
        min: v[0],
        p25: q(1),
        median: q(2),
        p75: q(3),
        max: v[v.len() - 1],
        mean: v.iter().sum::<f64>() / v.len() as f64,
    })
}

/// Least-squares slope of `ys` against 0, 1, 2, ...; 0 for fewer than two points.
pub fn slope(ys: &[f64]) -> f64 {
    let n = ys.len() as f64;

    if ys.len() < 2 {
        return 0.0;
    }

    let mx = (n - 1.0) / 2.0;
    let my = ys.iter().sum::<f64>() / n;
    let num: f64 = ys
        .iter()
        .enumerate()
        .map(|(i, y)| (i as f64 - mx) * (y - my))
        .sum();
    let den: f64 = (0..ys.len()).map(|i| (i as f64 - mx).powi(2)).sum();

    num / den
}

/// Kendall's tau-b between two score lists of equal length (ties handled); 0 when either
/// has no spread.
pub fn kendall_tau(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (mut conc, mut disc, mut ta, mut tb) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

    for i in 0..n {
        for j in i + 1..n {
            let da = a[i] - a[j];
            let db = b[i] - b[j];
            let (sa, sb) = (tie_sign(da), tie_sign(db));

            match (sa, sb) {
                (0, 0) => {}
                (0, _) => ta += 1.0,
                (_, 0) => tb += 1.0,
                _ if sa == sb => conc += 1.0,
                _ => disc += 1.0,
            }
        }
    }

    let denom = ((conc + disc + ta) * (conc + disc + tb)).sqrt();

    if denom == 0.0 {
        0.0
    } else {
        (conc - disc) / denom
    }
}

fn tie_sign(d: f64) -> i32 {
    if d.abs() <= TIE_SLACK {
        0
    } else if d > 0.0 {
        1
    } else {
        -1
    }
}

/// How a ranking made on a random subset of the tasks agrees with the ranking on all
/// tasks, averaged over repetitions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankQuality {
    /// Tasks in the subset, and their share of all tasks (the share of the evaluation
    /// cost, as every task costs about the same number of calls).
    pub subset: usize,
    pub cost_fraction: f64,
    pub repetitions: usize,
    pub tau_mean: f64,
    /// Share of repetitions in which the subset's best member is a best member on all tasks.
    pub top1_hit_rate: f64,
    /// Mean of (best full mean) - (full mean of the subset's pick).
    pub top1_regret_mean: f64,
    /// Mean overlap of the best `top_n` by subset with the best `top_n` by all tasks.
    pub top_n: usize,
    pub top_n_overlap_mean: f64,
}

fn best_indices(means: &[f64], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..means.len()).collect();

    idx.sort_by(|&a, &b| means[b].total_cmp(&means[a]).then(a.cmp(&b)));
    idx.truncate(n);

    idx
}

/// Measure rank agreement on `repetitions` random subsets of `subset` tasks. `scores` has
/// one row per candidate and one column per task. The subsets come from `seed`.
pub fn subset_rank_quality(
    scores: &[Vec<f64>],
    subset: usize,
    repetitions: usize,
    top_n: usize,
    seed: u64,
) -> Option<RankQuality> {
    let tasks = scores.first()?.len();

    if tasks == 0 || subset == 0 || subset > tasks || scores.len() < 2 || repetitions == 0 {
        return None;
    }

    let mean_of = |row: &Vec<f64>, cols: &[usize]| {
        cols.iter().map(|&c| row[c]).sum::<f64>() / cols.len() as f64
    };
    let all: Vec<usize> = (0..tasks).collect();
    let full: Vec<f64> = scores.iter().map(|r| mean_of(r, &all)).collect();
    let full_best = full.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let top_n = top_n.clamp(1, scores.len());
    let full_top: BTreeSet<usize> = best_indices(&full, top_n).into_iter().collect();
    let (mut tau, mut hit, mut regret, mut overlap) = (0.0, 0.0, 0.0, 0.0);

    for rep in 0..repetitions {
        let mut rng =
            ChaCha8Rng::seed_from_u64(derive_seed(seed, &format!("subset/{subset}/{rep}")));
        let mut cols = all.clone();

        cols.shuffle(&mut rng);
        cols.truncate(subset);

        let part: Vec<f64> = scores.iter().map(|r| mean_of(r, &cols)).collect();
        let pick = best_indices(&part, 1)[0];

        tau += kendall_tau(&part, &full);
        hit += f64::from((full[pick] - full_best).abs() <= TIE_SLACK);
        regret += full_best - full[pick];
        overlap += best_indices(&part, top_n)
            .iter()
            .filter(|i| full_top.contains(i))
            .count() as f64
            / top_n as f64;
    }

    let r = repetitions as f64;

    Some(RankQuality {
        subset,
        cost_fraction: subset as f64 / tasks as f64,
        repetitions,
        tau_mean: tau / r,
        top1_hit_rate: hit / r,
        top1_regret_mean: regret / r,
        top_n,
        top_n_overlap_mean: overlap / r,
    })
}

/// How a new candidate compares with a champion on the same tasks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Outperform,
    Match,
    Regress,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Versus {
    pub wins: usize,
    pub ties: usize,
    pub losses: usize,
    pub mean_diff: f64,
    pub verdict: Verdict,
}

/// Compare per-task values of a new candidate with those of a champion, task by task. The
/// verdict follows the mean difference: beyond `epsilon` (and the float slack) either way
/// is an outperform or a regress, else a match. Lists of different length are an error.
pub fn compare(new: &[f64], old: &[f64], epsilon: f64) -> Result<Versus> {
    if new.len() != old.len() || new.is_empty() {
        return Err(BrutError::Invalid(
            "compare: the value lists must have the same, non-zero length".to_string(),
        ));
    }

    let (mut wins, mut ties, mut losses) = (0, 0, 0);

    for (a, b) in new.iter().zip(old) {
        match tie_sign(a - b) {
            1 => wins += 1,
            -1 => losses += 1,
            _ => ties += 1,
        }
    }

    let mean_diff = (new.iter().sum::<f64>() - old.iter().sum::<f64>()) / new.len() as f64;
    let cut = epsilon.max(0.0) + TIE_SLACK;
    let verdict = if mean_diff > cut {
        Verdict::Outperform
    } else if mean_diff < -cut {
        Verdict::Regress
    } else {
        Verdict::Match
    };

    Ok(Versus {
        wins,
        ties,
        losses,
        mean_diff,
        verdict,
    })
}

/// A hash of an answer list: two Cascades with the same signature behaved identically on
/// the panel.
pub fn behaviour_signature<A: Serialize>(answers: &[Option<A>]) -> String {
    sha256_hex(
        serde_json::to_string(answers)
            .unwrap_or_default()
            .as_bytes(),
    )[..16]
        .to_string()
}

/// A frozen historical champion. Its fields are private and it has no mutating method; the
/// vault hands out shared references only.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Champion {
    id: String,
    generation: usize,
    cascade: Cascade,
    search_values: Vec<f64>,
    heldout_values: Vec<f64>,
    signature: String,
}

impl Champion {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn generation(&self) -> usize {
        self.generation
    }

    pub fn cascade(&self) -> &Cascade {
        &self.cascade
    }

    pub fn search_values(&self) -> &[f64] {
        &self.search_values
    }

    pub fn heldout_values(&self) -> &[f64] {
        &self.heldout_values
    }

    pub fn signature(&self) -> &str {
        &self.signature
    }

    pub fn search_rate(&self) -> f64 {
        mean(&self.search_values)
    }

    pub fn heldout_rate(&self) -> f64 {
        mean(&self.heldout_values)
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Append-only store of champions.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChampionVault {
    champions: Vec<Champion>,
}

impl ChampionVault {
    pub fn new() -> ChampionVault {
        ChampionVault::default()
    }

    /// Freeze a champion: copies of the Cascade and the values, so later changes to the
    /// archive cannot reach it.
    pub fn freeze(
        &mut self,
        member_id: &str,
        generation: usize,
        cascade: &Cascade,
        signature: String,
        search_values: &[f64],
        heldout_values: &[f64],
    ) {
        self.champions.push(Champion {
            id: member_id.to_string(),
            generation,
            cascade: cascade.clone(),
            search_values: search_values.to_vec(),
            heldout_values: heldout_values.to_vec(),
            signature,
        });
    }

    pub fn champions(&self) -> &[Champion] {
        &self.champions
    }

    pub fn len(&self) -> usize {
        self.champions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.champions.is_empty()
    }
}

/// Counts of verdicts of new candidates against frozen champions.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerdictCounts {
    pub outperform: usize,
    pub matched: usize,
    pub regress: usize,
}

impl VerdictCounts {
    fn add(&mut self, v: Verdict) {
        match v {
            Verdict::Outperform => self.outperform += 1,
            Verdict::Match => self.matched += 1,
            Verdict::Regress => self.regress += 1,
        }
    }
}

/// What was measured at one committed generation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationRecord {
    pub generation: usize,
    pub born: usize,
    /// Worker calls of the run so far (the breeder's own count).
    pub calls: u64,
    pub incumbent: String,
    /// Training rate of the incumbent (best by fitness) and of the best member by rate.
    pub train_rate: f64,
    pub best_train_rate: f64,
    pub heldout_rate: f64,
    /// train_rate - heldout_rate of the incumbent.
    pub gap: f64,
    /// Distinct behaviours over the archive, and the share of members that are new ones.
    pub behaviours: usize,
    pub archive_size: usize,
    pub parent_entropy_bits: f64,
    pub shape_entropy_bits: f64,
    pub block_entropy_bits: f64,
    /// This generation's newborns against every champion frozen before it, on the
    /// training tasks (frozen answers, identical seeds).
    pub born_vs_champions: VerdictCounts,
    /// Newborns that outperform every earlier champion.
    pub born_beating_all: usize,
    /// Newborns whose behaviour equals that of an earlier member with another Cascade.
    pub rediscoveries: usize,
    /// True when the incumbent replaced the champion before it, and was frozen.
    pub new_champion: bool,
    /// The new champion against the champions before it, on the held-out tasks.
    pub champion_heldout_vs_history: VerdictCounts,
}

/// One breeding run with everything measured around it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    pub arm: String,
    pub seed: u64,
    pub generations_completed: usize,
    pub stop_reason: String,
    pub search_tasks: usize,
    pub select_tasks: usize,
    /// Worker calls of the run, select phase included (the budgeted count).
    pub worker_calls: u64,
    /// Calls the observer spent on the held-out incumbents; outside the budget.
    pub heldout_observer_calls: u64,
    /// Cascades evaluated on the search panel (each on every search task).
    pub cascade_evaluations: usize,
    /// Sums over generations of the search-panel and peer usage.
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub model_cost: f64,
    /// Set by the caller; the library has no clock.
    pub wall_seconds: f64,
    pub best_train_rate: f64,
    pub final_train_rate: f64,
    pub final_heldout_rate: f64,
    pub final_gap: f64,
    /// Slope of the gap per generation over the run; positive means a growing gap.
    pub gap_slope: f64,
    /// Best held-out rate of a bred finalist and of a seed Cascade, from the select phase.
    pub best_finalist_heldout: f64,
    pub best_seed_heldout: f64,
    pub final_behaviours: usize,
    pub final_archive_size: usize,
    pub behaviour_share: f64,
    pub mean_parent_entropy_bits: f64,
    pub champions_frozen: usize,
    /// Share of champions that are, or are ancestors of, a member of the last generation.
    pub champion_lineage_alive: f64,
    /// Share of earlier champions that the final incumbent outperforms on training tasks.
    pub champions_surpassed: f64,
    pub rediscoveries: usize,
    /// Times a new champion behaves like an earlier champion other than its predecessor.
    pub champion_cycles: usize,
    pub generations: Vec<GenerationRecord>,
    pub finalists: Vec<Finalist>,
}

/// The graded value of each answer on its task, 0 where there is no answer. An integrity
/// alarm of the domain, or a value outside [0, 1], is an error, as in the breeder: a fresh
/// answer that only the study grades must not turn a broken grader into a 0.
fn values<D: Domain>(
    domain: &D,
    tasks: &[D::Task],
    answers: &[Option<D::Answer>],
) -> Result<Vec<f64>> {
    tasks
        .iter()
        .zip(answers)
        .map(|(t, a)| match a {
            Some(a) => checked_grade(domain, t, a).map(|g| g.value),
            None => Ok(0.0),
        })
        .collect()
}

/// Score `cascade` on the select panel exactly as the select phase does (same seeds, so a
/// finalist's answers equal these), returning per-task values and the calls made. A
/// grading error is returned (see `values`).
fn heldout_values<D: Domain>(
    b: &Breeder<'_, D>,
    pool: &ModelPool,
    id: &str,
    cascade: &Cascade,
) -> Result<(Vec<f64>, u64)> {
    let mut calls = 0;
    let answers: Vec<Option<D::Answer>> = b
        .select_panel()
        .iter()
        .map(|task| {
            let tid = b.domain().task_id(task);
            let seed = derive_seed(b.config().seed, &format!("select/{id}/{tid}"));
            let run = run_cascade_in(b.domain(), pool, &b.config().limits, cascade, task, seed);

            calls += run.usage.calls;

            run.answer
        })
        .collect();

    Ok((values(b.domain(), b.select_panel(), &answers)?, calls))
}

/// Re-run `cascade` on the search panel under the conditions of its birth: the same
/// seeds. A frozen champion must reproduce its stored answers.
pub fn replay_on_search<D: Domain>(
    b: &Breeder<'_, D>,
    pool: &ModelPool,
    id: &str,
    cascade: &Cascade,
) -> Vec<Option<D::Answer>> {
    b.search_panel()
        .iter()
        .map(|task| {
            let tid = b.domain().task_id(task);
            let seed = derive_seed(b.config().seed, &format!("search/{id}/{tid}"));

            run_cascade_in(b.domain(), pool, &b.config().limits, cascade, task, seed).answer
        })
        .collect()
}

/// Breed to the end under `arm`'s settings and measure. With `track_heldout` false the
/// observer scores nothing on the select panel and the held-out fields are 0; the breeding
/// itself is the same either way (tested).
pub fn run_study<D: Domain>(
    arm: &str,
    cfg: BreedConfig,
    genes: GenePool,
    pool: &ModelPool,
    track_heldout: bool,
    epsilon: f64,
) -> Result<(RunRecord, Report, Snapshot<D::Answer>)> {
    let seed = cfg.seed;
    let mut b: Breeder<'_, D> = Breeder::start(cfg, genes, pool, None)?;
    let never = || false;
    let mut vault = ChampionVault::new();
    let mut held: HashMap<String, Vec<f64>> = HashMap::new();
    let mut observer_calls = 0u64;
    let mut seen_sig: HashMap<String, String> = HashMap::new();
    let mut records: Vec<GenerationRecord> = Vec::new();
    let (mut prompt, mut completion, mut model_cost) = (0u64, 0u64, 0.0f64);
    let mut evaluations = 0usize;
    let mut champion_cycles = 0usize;
    let mut rediscoveries = 0usize;
    let mut last_state: Option<crate::cascade::observe::UniverseState> = None;

    let reason = loop {
        let first_new = b.archive().len();
        let (status, events) = b.step(&never)?;

        for e in &events {
            if let Event::State(s) = e {
                prompt += s.cost.prompt_tokens;
                completion += s.cost.completion_tokens;
                model_cost += s.cost.model_cost;
                last_state = Some(s.clone());
            }
        }

        match status {
            Status::Continue => {}
            Status::Done(r) => break r,
            Status::Pending(_) => break "waiting for LLM answers".to_string(),
        }

        let g = b.next_generation() - 1;
        let archive = b.archive();
        let train: Vec<Vec<f64>> = archive
            .iter()
            .map(|m| values(b.domain(), b.search_panel(), &m.answers))
            .collect::<Result<_>>()?;
        let rate = |i: usize| mean(&train[i]);
        let sigs: Vec<String> = archive
            .iter()
            .map(|m| behaviour_signature(&m.answers))
            .collect();
        let inc = b.ranking().first().copied().ok_or_else(|| {
            BrutError::Invalid(
                "study: the archive is empty after a committed generation".to_string(),
            )
        })?;
        let inc_id = archive[inc].id.clone();

        // Every newborn was evaluated once, on every search task.
        evaluations += archive.len() - first_new;

        // Rediscovery: a newborn that behaves like an earlier member with another Cascade.
        let mut redisc = 0;

        for i in first_new..archive.len() {
            if let Some(prev) = seen_sig.get(&sigs[i])
                && prev != &archive[i].id
            {
                redisc += 1;
            }
        }

        rediscoveries += redisc;

        for i in 0..archive.len() {
            seen_sig
                .entry(sigs[i].clone())
                .or_insert_with(|| archive[i].id.clone());
        }

        // Newborns against the champions frozen before this generation.
        let mut counts = VerdictCounts::default();
        let mut beating_all = 0;

        for row in &train[first_new..] {
            let mut all = !vault.is_empty();

            for c in vault.champions() {
                let v = compare(row, c.search_values(), epsilon)?.verdict;

                counts.add(v);
                all &= v == Verdict::Outperform;
            }

            beating_all += usize::from(all);
        }

        // The incumbent becomes a champion when it replaces the last one.
        let mut hv = Vec::new();
        let mut new_champion = false;
        let mut history = VerdictCounts::default();

        if track_heldout {
            if !held.contains_key(&inc_id) {
                let (v, calls) = heldout_values(&b, pool, &inc_id, &archive[inc].cascade)?;

                observer_calls += calls;
                held.insert(inc_id.clone(), v);
            }

            hv = held[&inc_id].clone();
        }

        if vault.champions().last().is_none_or(|c| c.id() != inc_id) {
            new_champion = true;

            // Back to the behaviour of a champion before the last one: a cycle.
            if vault
                .champions()
                .iter()
                .rev()
                .skip(1)
                .any(|c| c.signature() == sigs[inc])
            {
                champion_cycles += 1;
            }

            if track_heldout {
                for c in vault.champions() {
                    history.add(compare(&hv, c.heldout_values(), epsilon)?.verdict);
                }
            }

            vault.freeze(
                &inc_id,
                g,
                &archive[inc].cascade,
                sigs[inc].clone(),
                &train[inc],
                &hv,
            );
        }

        let st = last_state.as_ref();
        let train_rate = rate(inc);
        let heldout_rate = mean(&hv);
        let behaviours = sigs.iter().collect::<BTreeSet<_>>().len();

        records.push(GenerationRecord {
            generation: g,
            born: archive.len() - first_new,
            calls: b.calls_used(),
            incumbent: inc_id,
            train_rate,
            best_train_rate: (0..archive.len()).map(rate).fold(0.0, f64::max),
            heldout_rate,
            gap: if track_heldout {
                train_rate - heldout_rate
            } else {
                0.0
            },
            behaviours,
            archive_size: archive.len(),
            parent_entropy_bits: st.map_or(0.0, |s| s.diversity.parent_entropy_bits),
            shape_entropy_bits: st.map_or(0.0, |s| s.diversity.shape_entropy_bits),
            block_entropy_bits: st.map_or(0.0, |s| s.diversity.block_entropy_bits),
            born_vs_champions: counts,
            born_beating_all: beating_all,
            rediscoveries: redisc,
            new_champion,
            champion_heldout_vs_history: history,
        });
    };

    let (report, _) = b
        .finish(&never, &reason)?
        .ok_or_else(|| BrutError::Stopped("select phase stopped".to_string()))?;
    let last = records.last().cloned();
    let gaps: Vec<f64> = records.iter().map(|r| r.gap).collect();
    let archive = b.archive();

    // Lineage of the last generation, for champion retention.
    let by_id: HashMap<&str, &Member<D::Answer>> =
        archive.iter().map(|m| (m.id.as_str(), m)).collect();
    let newest = archive.iter().map(|m| m.generation).max().unwrap_or(0);
    let mut ancestors: BTreeSet<String> = BTreeSet::new();
    let mut todo: Vec<&str> = archive
        .iter()
        .filter(|m| m.generation == newest)
        .map(|m| m.id.as_str())
        .collect();

    while let Some(id) = todo.pop() {
        if ancestors.insert(id.to_string())
            && let Some(m) = by_id.get(id)
        {
            todo.extend(m.parents.iter().map(String::as_str));
        }
    }

    let champs = vault.champions();
    let alive = champs.iter().filter(|c| ancestors.contains(c.id())).count();
    let surpassed = match (champs.last(), champs.len()) {
        (Some(fin), n) if n > 1 => {
            let earlier = &champs[..n - 1];
            let beaten = earlier
                .iter()
                .filter(|c| {
                    compare(fin.search_values(), c.search_values(), epsilon)
                        .is_ok_and(|v| v.verdict == Verdict::Outperform)
                })
                .count();

            beaten as f64 / earlier.len() as f64
        }
        _ => 0.0,
    };
    let behaviours_final = last.as_ref().map_or(0, |r| r.behaviours);
    let archive_size = archive.len();
    let entropy: Vec<f64> = records.iter().map(|r| r.parent_entropy_bits).collect();

    let record = RunRecord {
        arm: arm.to_string(),
        seed,
        generations_completed: report.generations_completed,
        stop_reason: reason,
        search_tasks: b.search_panel().len(),
        select_tasks: b.select_panel().len(),
        worker_calls: report.calls_used,
        heldout_observer_calls: observer_calls,
        cascade_evaluations: evaluations,
        prompt_tokens: prompt,
        completion_tokens: completion,
        model_cost,
        wall_seconds: 0.0,
        best_train_rate: records
            .iter()
            .map(|r| r.best_train_rate)
            .fold(0.0, f64::max),
        final_train_rate: last.as_ref().map_or(0.0, |r| r.train_rate),
        final_heldout_rate: last.as_ref().map_or(0.0, |r| r.heldout_rate),
        final_gap: last.as_ref().map_or(0.0, |r| r.gap),
        gap_slope: slope(&gaps),
        best_finalist_heldout: report.best_finalist_select_rate,
        best_seed_heldout: report.best_seed_select_rate,
        final_behaviours: behaviours_final,
        final_archive_size: archive_size,
        behaviour_share: behaviours_final as f64 / archive_size.max(1) as f64,
        mean_parent_entropy_bits: mean(&entropy),
        champions_frozen: champs.len(),
        champion_lineage_alive: alive as f64 / champs.len().max(1) as f64,
        champions_surpassed: surpassed,
        rediscoveries,
        champion_cycles,
        generations: records,
        finalists: report.finalists.clone(),
    };

    Ok((record, report, b.snapshot()))
}

/// A finalist of a run on a down-sampled search panel, scored again on the full panel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reevaluated {
    pub id: String,
    /// Training rate on the evaluation subset the run used, and on the full panel.
    pub subset_rate: f64,
    pub full_rate: f64,
    pub heldout_rate: f64,
}

/// What the full re-evaluation of a down-sampled run's finalists shows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reevaluation {
    pub finalists: Vec<Reevaluated>,
    /// The finalist with the best rate on the evaluation subset: what the shortcut would
    /// have reported. Ties go to the earlier finalist (the better one by the run's
    /// fitness).
    pub approx_pick: String,
    /// The finalist with the best full-panel rate, ties as for `approx_pick`.
    pub full_pick: String,
    /// True when the full panel picks another finalist than the subset.
    pub pick_changed: bool,
    pub approx_pick_heldout: f64,
    pub full_pick_heldout: f64,
    /// Kendall tau between subset and full rates over the finalists.
    pub tau: f64,
    /// Worker calls spent on the re-evaluation (outside the run's budget).
    pub calls: u64,
}

/// Score the bred finalists of a run made with a smaller search panel on the full panel
/// that `full_cfg` (the same settings with the full `search_puzzles`) describes. The
/// smaller panel is a prefix of the full one, so a finalist's first answers reproduce.
/// An approximate evaluation must not decide the final result alone: `full_pick` is the
/// result, `approx_pick` only what the shortcut would have said.
pub fn reevaluate_finalists<D: Domain>(
    full_cfg: BreedConfig,
    genes: GenePool,
    pool: &ModelPool,
    snap: &Snapshot<D::Answer>,
    report: &Report,
) -> Result<Option<Reevaluation>> {
    let full: Breeder<'_, D> = Breeder::start(full_cfg, genes, pool, None)?;
    let by_id: HashMap<&str, &Member<D::Answer>> =
        snap.archive.iter().map(|m| (m.id.as_str(), m)).collect();
    let mut out = Vec::new();
    let mut calls = 0u64;

    for f in report.finalists.iter().filter(|f| !f.is_seed) {
        let Some(m) = by_id.get(f.id.as_str()) else {
            continue;
        };
        let answers: Vec<Option<D::Answer>> = full
            .search_panel()
            .iter()
            .map(|task| {
                let tid = full.domain().task_id(task);
                let seed = derive_seed(full.config().seed, &format!("search/{}/{tid}", m.id));
                let run = run_cascade_in(
                    full.domain(),
                    pool,
                    &full.config().limits,
                    &m.cascade,
                    task,
                    seed,
                );

                calls += run.usage.calls;

                run.answer
            })
            .collect();

        out.push(Reevaluated {
            id: m.id.clone(),
            subset_rate: mean(&values(
                full.domain(),
                &full.search_panel()[..m.answers.len()],
                &m.answers,
            )?),
            full_rate: mean(&values(full.domain(), full.search_panel(), &answers)?),
            heldout_rate: f.select_solved as f64 / f.select_puzzles.max(1) as f64,
        });
    }

    // Both picks follow one rule, each on its own panel, so a full panel that reproduces
    // every subset rate cannot change the pick.
    let (Some(approx), Some(best)) = (
        first_best(&out, |r| r.subset_rate),
        first_best(&out, |r| r.full_rate),
    ) else {
        return Ok(None);
    };
    let tau = kendall_tau(
        &out.iter().map(|r| r.subset_rate).collect::<Vec<_>>(),
        &out.iter().map(|r| r.full_rate).collect::<Vec<_>>(),
    );

    Ok(Some(Reevaluation {
        pick_changed: approx.id != best.id,
        approx_pick: approx.id,
        full_pick: best.id,
        approx_pick_heldout: approx.heldout_rate,
        full_pick_heldout: best.heldout_rate,
        finalists: out,
        tau,
        calls,
    }))
}

/// The finalist with the highest `rate`, None when there is none. The finalists come best
/// first by the run's fitness; ties (within the float slack) go to the earlier one.
fn first_best(out: &[Reevaluated], rate: impl Fn(&Reevaluated) -> f64) -> Option<Reevaluated> {
    let top = out.iter().map(&rate).fold(f64::NEG_INFINITY, f64::max);

    out.iter().find(|r| rate(r) >= top - TIE_SLACK).cloned()
}

/// Per-task values of every archive member on the search panel, rows in archive order: the
/// input of `subset_rank_quality`. A grading error is returned (see `values`).
pub fn search_score_matrix<D: Domain>(b: &Breeder<'_, D>) -> Result<Vec<Vec<f64>>> {
    b.archive()
        .iter()
        .map(|m| values(b.domain(), b.search_panel(), &m.answers))
        .collect()
}

/// Mean over `runs` grouped by arm: the distribution of `f` per arm, sorted by arm name.
pub fn by_arm<F: Fn(&RunRecord) -> f64>(
    runs: &[RunRecord],
    f: F,
) -> BTreeMap<String, Option<Dist>> {
    let mut groups: BTreeMap<String, Vec<f64>> = BTreeMap::new();

    for r in runs {
        groups.entry(r.arm.clone()).or_default().push(f(r));
    }

    groups.into_iter().map(|(k, v)| (k, dist(&v))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cascade::arena::ParentSelection;
    use crate::cascade::domain::{Grade, GridDomain};
    use crate::cascade::puzzle::{Grid, Puzzle};
    use crate::cascade::sim::SimModel;

    fn sim_pool() -> ModelPool {
        let mut pool = ModelPool::new();

        pool.insert("sim:small", Box::new(SimModel::small()));
        pool.insert("sim:large", Box::new(SimModel::large()));

        pool
    }

    fn cfg(seed: u64, selection: ParentSelection) -> BreedConfig {
        BreedConfig {
            seed,
            generations: 4,
            children_per_generation: 4,
            search_puzzles: 10,
            select_puzzles: 10,
            finalists: 2,
            selection,
            ..BreedConfig::default()
        }
    }

    fn study(c: BreedConfig, heldout: bool) -> RunRecord {
        let pool = sim_pool();

        run_study::<GridDomain>("t", c, GenePool::grid(), &pool, heldout, 0.0)
            .expect("study")
            .0
    }

    #[test]
    fn dist_and_slope_and_tau() {
        let d = dist(&[4.0, 1.0, 3.0, 2.0, 5.0]).unwrap();

        assert_eq!(
            (d.min, d.p25, d.median, d.p75, d.max, d.mean),
            (1.0, 2.0, 3.0, 4.0, 5.0, 3.0)
        );
        assert!(dist(&[]).is_none() && dist(&[f64::NAN]).is_none());
        assert!((slope(&[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-12);
        assert_eq!(slope(&[1.0]), 0.0);
        assert!((kendall_tau(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]) - 1.0).abs() < 1e-12);
        assert!((kendall_tau(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]) + 1.0).abs() < 1e-12);
        assert_eq!(kendall_tau(&[1.0, 1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn a_full_subset_ranks_perfectly_and_a_small_one_is_measured_not_assumed() {
        // Four candidates, 8 tasks; candidate i solves tasks 0..=2i.
        let scores: Vec<Vec<f64>> = (0..4)
            .map(|i| (0..8).map(|t| f64::from(t <= 2 * i)).collect())
            .collect();
        let full = subset_rank_quality(&scores, 8, 5, 2, 1).unwrap();

        assert_eq!(
            (full.tau_mean, full.top1_hit_rate, full.top1_regret_mean),
            (1.0, 1.0, 0.0)
        );
        assert_eq!(full.cost_fraction, 1.0);

        let small = subset_rank_quality(&scores, 2, 50, 2, 1).unwrap();

        assert!(small.tau_mean < 1.0, "two tasks cannot rank all four");
        assert_eq!(
            small,
            subset_rank_quality(&scores, 2, 50, 2, 1).unwrap(),
            "same seed, same subsets"
        );
        assert!(subset_rank_quality(&scores, 9, 5, 2, 1).is_none());
    }

    #[test]
    fn compare_classifies_with_epsilon() {
        let v = compare(&[1.0, 1.0, 0.0], &[1.0, 0.0, 0.0], 0.0).unwrap();

        assert_eq!(
            (v.wins, v.ties, v.losses, v.verdict),
            (1, 2, 0, Verdict::Outperform)
        );
        assert_eq!(
            compare(&[1.0, 0.0], &[0.0, 1.0], 0.0).unwrap().verdict,
            Verdict::Match
        );
        assert_eq!(
            compare(&[0.0], &[1.0], 0.0).unwrap().verdict,
            Verdict::Regress
        );
        assert_eq!(
            compare(&[0.9], &[1.0], 0.2).unwrap().verdict,
            Verdict::Match
        );
        assert!(compare(&[1.0], &[1.0, 2.0], 0.0).is_err());
    }

    #[test]
    fn behaviour_signature_tells_equal_answers_apart_from_different_ones() {
        let a: Vec<Option<u8>> = vec![Some(1), None];

        assert_eq!(behaviour_signature(&a), behaviour_signature(&a.clone()));
        assert_ne!(
            behaviour_signature(&a),
            behaviour_signature(&[Some(1u8), Some(2)])
        );
    }

    #[test]
    fn the_observer_does_not_change_the_breeding() {
        // The held-out observer reads the archive and nothing flows back: the run that
        // scores the select panel every generation breeds exactly the run that does not.
        let pool = sim_pool();

        for sel in [ParentSelection::Dgm, ParentSelection::Lexicase] {
            let run = |heldout: bool| {
                let mut b: Breeder<'_, GridDomain> =
                    Breeder::start(cfg(5, sel), GenePool::grid(), &pool, None).unwrap();
                let never = || false;
                let mut states = Vec::new();

                loop {
                    let (st, _) = b.step(&never).unwrap();

                    if heldout && let Some(m) = b.archive().first() {
                        heldout_values(&b, &pool, &m.id, &m.cascade).unwrap();
                    }

                    states.push(serde_json::to_string(&b.snapshot()).unwrap());

                    if st != Status::Continue {
                        break;
                    }
                }

                states
            };

            assert_eq!(run(false), run(true), "{sel:?}");
        }

        let with = study(cfg(5, ParentSelection::Dgm), true);
        let without = study(cfg(5, ParentSelection::Dgm), false);
        let ids = |r: &RunRecord| {
            r.generations
                .iter()
                .map(|g| (g.incumbent.clone(), g.train_rate.to_bits()))
                .collect::<Vec<_>>()
        };

        assert_eq!(ids(&with), ids(&without));
        assert_eq!(with.worker_calls, without.worker_calls);
        assert!(with.heldout_observer_calls > 0 && without.heldout_observer_calls == 0);
    }

    #[test]
    fn a_study_is_reproducible_and_records_every_generation() {
        for sel in [
            ParentSelection::Dgm,
            ParentSelection::Lexicase,
            ParentSelection::Tournament,
        ] {
            let a = study(cfg(3, sel), true);
            let b = study(cfg(3, sel), true);

            assert_eq!(a, b, "{sel:?}");
            assert_eq!(a.generations.len(), a.generations_completed.min(5));
            assert!(a.worker_calls > 0 && a.cascade_evaluations > 0);
            assert!((a.final_gap - (a.final_train_rate - a.final_heldout_rate)).abs() < 1e-12);
        }

        let other = study(cfg(4, ParentSelection::Lexicase), true);

        assert_ne!(other, study(cfg(3, ParentSelection::Lexicase), true));
    }

    #[test]
    fn lexicase_and_dgm_share_generation_zero_and_differ_afterwards() {
        let dgm = study(cfg(2, ParentSelection::Dgm), false);
        let lex = study(cfg(2, ParentSelection::Lexicase), false);

        assert_eq!(dgm.generations[0].incumbent, lex.generations[0].incumbent);
        assert_eq!(dgm.generations[0].train_rate, lex.generations[0].train_rate);
        assert_ne!(
            dgm.generations
                .iter()
                .map(|g| g.incumbent.clone())
                .collect::<Vec<_>>(),
            lex.generations
                .iter()
                .map(|g| g.incumbent.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn champions_stay_frozen_and_replay_reproduces_them() {
        let pool = sim_pool();
        let mut b: Breeder<'_, GridDomain> =
            Breeder::start(cfg(6, ParentSelection::Dgm), GenePool::grid(), &pool, None).unwrap();
        let never = || false;

        b.step(&never).unwrap();

        let inc = b.ranking()[0];
        let m = b.archive()[inc].clone();
        let train = search_score_matrix(&b).unwrap()[inc].clone();
        let mut vault = ChampionVault::new();

        vault.freeze(
            &m.id,
            0,
            &m.cascade,
            behaviour_signature(&m.answers),
            &train,
            &[1.0],
        );

        let frozen = vault.clone();

        // The run goes on and its archive changes; the vault does not.
        while b.step(&never).unwrap().0 == Status::Continue {}

        assert!(b.archive().len() > 1);
        assert_eq!(vault, frozen);
        assert_eq!(vault.champions()[0].search_values(), train.as_slice());

        // Same seeds, same conditions: the champion's answers reproduce.
        let again = replay_on_search(&b, &pool, &m.id, vault.champions()[0].cascade());

        assert_eq!(again, m.answers);

        // A tampered copy of the archive member would not touch the vault either.
        let mut member = m.clone();

        member.answers.clear();
        assert_eq!(vault.champions()[0].search_values().len(), train.len());
    }

    #[test]
    fn the_select_phase_reuses_the_observer_seeds() {
        let pool = sim_pool();
        let (rec, _, _) = run_study::<GridDomain>(
            "t",
            cfg(8, ParentSelection::Dgm),
            GenePool::grid(),
            &pool,
            true,
            0.0,
        )
        .unwrap();
        // The incumbent's held-out rate equals its finalist result when it is a finalist.
        let last = rec.generations.last().unwrap();

        if let Some(f) = rec.finalists.iter().find(|f| f.id == last.incumbent) {
            assert!(
                (f.select_solved as f64 / f.select_puzzles as f64 - last.heldout_rate).abs()
                    < 1e-12
            );
        }
    }

    #[test]
    fn replayed_champions_are_compared_with_newborns() {
        let rec = study(cfg(9, ParentSelection::Dgm), true);

        assert!(rec.champions_frozen >= 1);
        assert!(
            rec.generations[0].born_vs_champions == VerdictCounts::default(),
            "nothing to compare with in generation 0"
        );

        let pairs: usize = rec
            .generations
            .iter()
            .map(|g| {
                let c = &g.born_vs_champions;

                c.outperform + c.matched + c.regress
            })
            .sum();

        assert!(pairs > 0);
    }

    #[test]
    fn cascade_evaluations_count_cascades_not_task_runs() {
        // A run on a smaller panel evaluates each Cascade on fewer tasks; the count of
        // Cascades must not depend on the panel size.
        for search_puzzles in [4, 10] {
            let c = BreedConfig {
                search_puzzles,
                ..cfg(3, ParentSelection::Dgm)
            };
            let rec = study(c, false);
            let born: usize = rec.generations.iter().map(|g| g.born).sum();

            assert_eq!(rec.cascade_evaluations, born, "{search_puzzles} tasks");
            assert_eq!(rec.cascade_evaluations, rec.final_archive_size);
        }
    }

    #[test]
    fn a_subset_that_is_the_whole_panel_never_changes_the_pick() {
        // The full panel reproduces every subset rate, so the two picks must agree. Judging
        // skill weighs much in fitness here, so the best finalist by fitness is often not
        // the one with the best rate.
        let pool = sim_pool();

        for seed in 1..=4 {
            let c = BreedConfig {
                finalists: 3,
                judge_weight: 0.6,
                ..cfg(seed, ParentSelection::Dgm)
            };
            let (_, report, snap) =
                run_study::<GridDomain>("t", c.clone(), GenePool::grid(), &pool, false, 0.0)
                    .unwrap();
            let re = reevaluate_finalists::<GridDomain>(c, GenePool::grid(), &pool, &snap, &report)
                .unwrap()
                .expect("bred finalists");

            assert!(
                re.finalists.iter().all(|f| f.subset_rate == f.full_rate),
                "seed {seed}"
            );
            assert!(!re.pick_changed, "seed {seed}: {re:?}");
            assert_eq!(re.approx_pick, re.full_pick, "seed {seed}");
            assert_eq!(re.approx_pick_heldout, re.full_pick_heldout, "seed {seed}");
        }
    }

    /// The search tasks of a full panel, and how many of them a down-sampled run uses.
    const FULL: usize = 10;
    const SMALL: usize = 4;

    /// Grid puzzles with a broken grader on the full search panel's tasks from `SMALL` on:
    /// with `ALARM` an integrity alarm, else a value of 2, outside [0, 1]. A run on the
    /// first `SMALL` tasks never grades them; only the re-evaluation on the full panel
    /// does.
    struct Broken<const ALARM: bool> {
        bad: BTreeSet<String>,
    }

    impl<const ALARM: bool> Domain for Broken<ALARM> {
        type Task = Puzzle;
        type Answer = Grid;

        fn from_config(cfg: &BreedConfig) -> Result<Self> {
            let full = GridDomain.panel(derive_seed(cfg.seed, "panel/search"), FULL);

            Ok(Broken {
                bad: full[SMALL..].iter().map(|t| t.id.clone()).collect(),
            })
        }

        fn panel(&self, seed: u64, n: usize) -> Vec<Puzzle> {
            GridDomain.panel(seed, n)
        }

        fn task_id<'t>(&self, task: &'t Puzzle) -> &'t str {
            GridDomain.task_id(task)
        }

        fn solve_prompt(&self, task: &Puzzle, previous: &[Grid]) -> String {
            GridDomain.solve_prompt(task, previous)
        }

        fn judge_prompt(&self, task: &Puzzle, a: &Grid, b: &Grid) -> String {
            GridDomain.judge_prompt(task, a, b)
        }

        fn parse_answer(&self, task: &Puzzle, text: &str) -> Option<Grid> {
            GridDomain.parse_answer(task, text)
        }

        fn grade(&self, task: &Puzzle, answer: &Grid) -> Result<Grade> {
            if !self.bad.contains(&task.id) {
                return GridDomain.grade(task, answer);
            }

            if ALARM {
                return Err(BrutError::Integrity(format!(
                    "task {} beats a proven bound",
                    task.id
                )));
            }

            Ok(Grade {
                correct: true,
                solved: true,
                value: 2.0,
            })
        }

        fn prefer(&self, task: &Puzzle, a: &Grid, b: &Grid) -> Option<bool> {
            GridDomain.prefer(task, a, b)
        }

        fn graded(&self) -> bool {
            GridDomain.graded()
        }

        fn artefact(&self, answer: &Grid) -> Option<String> {
            GridDomain.artefact(answer)
        }

        fn self_check_text(&self) -> &'static str {
            GridDomain.self_check_text()
        }
    }

    #[test]
    fn values_grade_like_the_breeder() {
        let c = cfg(12, ParentSelection::Dgm);
        let tasks = GridDomain.panel(derive_seed(c.seed, "panel/search"), FULL);
        let right: Vec<Option<Grid>> = tasks.iter().map(|t| Some(t.test_output.clone())).collect();
        let mut gaps = right.clone();

        gaps[0] = None;

        // A missing answer is worth 0, a right one 1.
        assert_eq!(values(&GridDomain, &tasks, &gaps).unwrap()[..2], [0.0, 1.0]);

        // A broken grader is an error, not a 0: an alarm, or a value outside [0, 1].
        let alarm = Broken::<true>::from_config(&c).unwrap();
        let outside = Broken::<false>::from_config(&c).unwrap();

        assert_eq!(
            values(&alarm, &tasks[..SMALL], &right).unwrap(),
            [1.0; SMALL]
        );
        assert!(matches!(
            values(&alarm, &tasks, &right),
            Err(BrutError::Integrity(e)) if e.contains("proven bound")
        ));
        assert!(matches!(
            values(&outside, &tasks, &right),
            Err(BrutError::Integrity(e)) if e.contains("outside [0, 1]")
        ));

        // Only answers are graded: no answer on a broken task is still a 0.
        for a in &mut gaps[SMALL..] {
            *a = None;
        }

        assert_eq!(
            values(&alarm, &tasks, &gaps).unwrap()[SMALL..],
            [0.0; FULL - SMALL]
        );
    }

    /// Breed on the first `SMALL` tasks, then score the finalists again on all `FULL`.
    fn reevaluate_with<D: Domain>() -> Result<Option<Reevaluation>> {
        let pool = sim_pool();
        let small = BreedConfig {
            search_puzzles: SMALL,
            finalists: 3,
            ..cfg(12, ParentSelection::Dgm)
        };
        let full = BreedConfig {
            search_puzzles: FULL,
            ..small.clone()
        };
        let (_, report, snap) = run_study::<D>("t", small, GenePool::grid(), &pool, true, 0.0)?;

        reevaluate_finalists::<D>(full, GenePool::grid(), &pool, &snap, &report)
    }

    #[test]
    fn a_broken_grader_on_fresh_answers_stops_the_re_evaluation() {
        // The run itself never grades the broken tasks, so it ends normally; the answers
        // the re-evaluation obtains are graded for the first time and must not become 0.
        let alarm = reevaluate_with::<Broken<true>>();

        assert!(
            matches!(&alarm, Err(BrutError::Integrity(e)) if e.contains("proven bound")),
            "{alarm:?}"
        );

        let outside = reevaluate_with::<Broken<false>>();

        assert!(
            matches!(&outside, Err(BrutError::Integrity(e)) if e.contains("outside [0, 1]")),
            "{outside:?}"
        );
    }

    #[test]
    fn finalists_of_a_down_sampled_run_are_scored_again_on_the_full_panel() {
        let pool = sim_pool();
        let small = BreedConfig {
            search_puzzles: 4,
            finalists: 3,
            ..cfg(12, ParentSelection::Dgm)
        };
        let full = BreedConfig {
            search_puzzles: 10,
            ..small.clone()
        };
        let (_, report, snap) =
            run_study::<GridDomain>("t", small.clone(), GenePool::grid(), &pool, false, 0.0)
                .unwrap();
        let re = reevaluate_finalists::<GridDomain>(
            full.clone(),
            GenePool::grid(),
            &pool,
            &snap,
            &report,
        )
        .unwrap()
        .expect("bred finalists");

        assert!(re.calls > 0 && !re.finalists.is_empty());
        assert_eq!(re.pick_changed, re.approx_pick != re.full_pick);

        // The subset is a prefix of the full panel: its answers reproduce under the same seeds.
        let breeder: Breeder<'_, GridDomain> =
            Breeder::start(full, GenePool::grid(), &pool, None).unwrap();
        let m = snap
            .archive
            .iter()
            .find(|m| m.id == re.finalists[0].id)
            .unwrap();
        let again = replay_on_search(&breeder, &pool, &m.id, &m.cascade);

        assert_eq!(&again[..4], m.answers.as_slice());
    }
}
