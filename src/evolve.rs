//! Orchestrator: the evolution loop that ties all components together.
//!
//! This module never reads the hidden secret. It only calls `brut-eval`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::archive::{Archive, CandidateEntry, CandidateStatus, RecordKind};
use crate::cnn::families::{self, BaseModels, Family};
use crate::cnn::variation as cnn_variation;
use crate::config::{Config, ModelKinds};
use crate::genome::Genome;
use crate::hidden_eval::{EVAL_TIMEOUT_SECONDS, EvalRequest, request_hidden_eval};
use crate::model::Model;
use crate::safety::{self, Guard, StopSignal};
use crate::score::{self, Metrics, WORST_FITNESS};
use crate::task::{self, Dataset, Split, TaskSpec};
use crate::train::{self, TrainStats};
use crate::util::{BrutError, Result, derive_seed, now_unix_ms, write_atomic};
use crate::variation::{RandomMutator, VariantGenerator};

/// File names inside the run directory.
pub const CONFIG_FILE: &str = "config.json";
pub const ARCHIVE_FILE: &str = "archive.jsonl";
pub const MODELS_DIR: &str = "models";

/// Generator name of the random initial population.
pub const INIT_GENERATOR: &str = "init";

/// Generator name of the base genomes of the config (`Config::base_genomes`).
pub const BASE_GENERATOR: &str = "base";

/// Start of the generator name of the base models of the CNN families in a run with CNNs;
/// then come "/" and the family's name, e.g. "base-model/resnet" (`Family::name`).
pub const BASE_MODEL_GENERATOR: &str = "base-model";

/// Generator name of the promoted models of the config (`Config::promoted`). The candidate id
/// of one is its content id, which names its file in `candidates/models/`.
pub const PROMOTED_GENERATOR: &str = "promoted";

/// Generator name of random children made when no parent is eligible.
pub const RESTART_GENERATOR: &str = "random-restart";

/// True for the generator name of a base model of a run: a base genome of the config, the
/// base model of a CNN family, or a promoted model. These enter generation 0 as known designs,
/// not as random draws. A model is promoted only if it beats the best of them on the hidden
/// split (`best_base_model`, `crate::promote`).
pub fn is_base_model(generator: &str) -> bool {
    generator == BASE_GENERATOR
        || generator == PROMOTED_GENERATOR
        || generator
            .strip_prefix(BASE_MODEL_GENERATOR)
            .is_some_and(|rest| rest.starts_with('/'))
}

/// The best base model among `candidates` (`is_base_model`): status Ok or Quarantined, the
/// highest fitness, ties by id. In a run that can promote (`Config::can_promote`) it joins the
/// hidden set (`hidden_set`), so that a promotion can compare the winners with it on the
/// hidden split.
pub fn best_base_model(candidates: &[CandidateEntry]) -> Option<&CandidateEntry> {
    candidates
        .iter()
        .filter(|c| is_base_model(&c.generator))
        .filter(|c| matches!(c.status, CandidateStatus::Ok | CandidateStatus::Quarantined))
        .min_by(|a, b| {
            b.fitness
                .total_cmp(&a.fitness)
                .then_with(|| a.id.cmp(&b.id))
        })
}

/// Proposal attempts per wanted child (duplicates are retried).
const ATTEMPTS_PER_CHILD: usize = 4;

/// Where the hidden evaluator lives. `None` in `run` skips hidden evaluation.
#[derive(Clone, Debug)]
pub struct EvalSetup {
    pub eval_bin: PathBuf,
    pub secret_env: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HiddenResult {
    pub candidate_id: String,
    pub valid_primary: f64,
    pub hidden_primary: f64,
    pub quarantined: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunSummary {
    pub generations_done: usize,
    pub candidates: usize,
    pub best_id: Option<String>,
    pub best_fitness: f64,
    pub best_valid_primary: f64,
    pub flops_used: u64,
    pub wall_seconds: f64,
    pub stop_reason: String,
    pub hidden: Vec<HiddenResult>,
}

/// One child waiting for training.
struct Plan {
    genome: Genome,
    parents: Vec<String>,
    generator: String,
}

/// One trained (or failed) child.
struct Outcome {
    entry: CandidateEntry,
    model: Option<Model>,
    error: Option<String>,
}

/// Fixed inputs shared by all children of a run.
struct Context<'a> {
    config: &'a Config,
    spec: &'a TaskSpec,
    train: &'a Dataset,
    valid: &'a Dataset,
    /// Kill switch, checked by every training job between steps.
    stop: &'a StopSignal,
    /// The base models of the CNN families that join generation 0 (`first_base_models`).
    base_models: &'a [(Family, Genome)],
    /// The promoted models of the config that join generation 0 (`first_promoted`).
    promoted: &'a [Genome],
}

/// Run directory layout:
/// - `config.json`      run config
/// - `archive.jsonl`    lineage archive (hash chain)
/// - `models/<id>.json` trained models of status Ok
/// - `STOP`             kill switch (optional)
/// - `secret/hidden.env`, `secret/eval_state.json` (evaluator only)
///
/// Steps:
/// 1. Load config, create or reopen the archive (resume: continue after the last generation).
/// 2. Initial population, generation 0: the base genomes of the config, if any, then in a
///    run with CNNs the base models of the CNN families that fit the task and the limits
///    (a log line names each one left out), then the promoted models of the config that fit
///    the run's kinds, the task and the limits (again with a log line for each one left
///    out), then `population_init` random genomes of the run's kinds (`Config::models`).
/// 3. Per generation: guard check; select parents; propose children (mutation or crossover by
///    `crossover_rate`); skip duplicates by genome id; skip children that do not fit the FLOP budget;
///    train children in parallel, largest estimated cost first (rayon, `threads`); score on Valid; save Ok models; append records
///    in a deterministic order (sorted by id); charge FLOPs.
/// 4. On guard stop: log a SafetyEvent and finish cleanly (not an error).
/// 5. Hidden eval of the fixed hidden set (`hidden_set`: top `hidden_eval_top_k` trained
///    candidates by fitness, quarantined ones included, and the best base model) if `eval`
///    is Some; quarantine when `safety::gap_exceeded`. After this the run is final
///    (`check_not_final`).
/// 6. Append RunFinished, return the summary.
///
/// Resume details: `config.json` must define the same experiment as the archive's first run
/// (see `Config::resume_conflicts`), else the run is refused (Invalid) and nothing is written.
/// The FLOP budget counts all candidates already in the archive; the time limit
/// counts only this invocation. Candidates that already have a hidden evaluation are not sent again.
/// Once the archive has a hidden evaluation, new generations are refused (Invalid).
/// A STOP file also skips the hidden evaluation.
pub fn run(run_dir: &Path, eval: Option<&EvalSetup>) -> Result<RunSummary> {
    let config = Config::load(&run_dir.join(CONFIG_FILE))?;
    let archive_path = run_dir.join(ARCHIVE_FILE);

    let mut archive = if archive_path.exists() {
        Archive::open(&archive_path)?
    } else {
        Archive::create(&archive_path)?
    };

    check_resume(&config, &archive)?;
    check_not_final(&config, &archive)?;

    let started = RecordKind::RunStarted {
        config: config.clone(),
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
    };

    archive.append(started, now_unix_ms())?;

    match run_inner(run_dir, eval, &config, &mut archive) {
        Ok(summary) => Ok(summary),
        Err(e) => {
            // Best effort: the original error matters more than a failed log write.
            let _ = archive.append(
                RecordKind::SafetyEvent {
                    what: format!("run aborted: {e}"),
                },
                now_unix_ms(),
            );
            let _ = archive.append(
                RecordKind::RunFinished {
                    reason: format!("error: {e}"),
                },
                now_unix_ms(),
            );

            Err(e)
        }
    }
}

/// Refuse to resume when `config` defines a different experiment than the archive's first run.
/// Nothing is written to the archive in that case.
fn check_resume(config: &Config, archive: &Archive) -> Result<()> {
    let Some(first) = archive.first_config() else {
        return Ok(());
    };

    let conflicts = config.resume_conflicts(first);

    if conflicts.is_empty() {
        return Ok(());
    }

    Err(BrutError::Invalid(format!(
        "{CONFIG_FILE} changes {} compared with the first run in this directory; \
         on resume only generations, threads, max_candidate_seconds, max_total_seconds \
         and max_total_flops may change (start a new run directory with brut init)",
        conflicts.join(", ")
    )))
}

/// Refuse new generations once the archive holds a hidden evaluation.
///
/// The hidden data must not steer the search: after it, parents would be picked among
/// candidates that passed the hidden check. Re-running without new generations is allowed,
/// so an interrupted hidden phase can finish the same fixed hidden set.
fn check_not_final(config: &Config, archive: &Archive) -> Result<()> {
    if archive.hidden_evals().is_empty() {
        return Ok(());
    }

    let last_generation = archive
        .candidates()
        .iter()
        .map(|c| c.generation)
        .max()
        .unwrap_or(0);

    if config.generations <= last_generation {
        return Ok(());
    }

    Err(BrutError::Invalid(format!(
        "this run already has a hidden evaluation, so it is finished after generation \
         {last_generation}; more generations would let hidden results steer the search \
         (set generations to {last_generation} to finish the hidden evaluation, or start \
         a new run directory with brut init)"
    )))
}

fn run_inner(
    run_dir: &Path,
    eval: Option<&EvalSetup>,
    config: &Config,
    archive: &mut Archive,
) -> Result<RunSummary> {
    let spec = config.task.spec();
    let data_seed = derive_seed(config.master_seed, "data");
    let train_data = task::generate(&spec, Split::Train, data_seed);
    let valid_data = task::generate(&spec, Split::Valid, data_seed);
    let stop = StopSignal::new(run_dir);
    let (base_models, left_out) = first_base_models(config, &spec);
    let (promoted, promoted_left_out) = first_promoted(config, &spec, &base_models);
    let ctx = Context {
        config,
        spec: &spec,
        train: &train_data,
        valid: &valid_data,
        stop: &stop,
        base_models: &base_models,
        promoted: &promoted,
    };

    std::fs::create_dir_all(run_dir.join(MODELS_DIR))?;

    let mut guard = Guard::new(run_dir, &config.limits);
    let existing = archive.candidates();
    let previous_flops = existing
        .iter()
        .fold(0u64, |acc, c| acc.saturating_add(c.train.flops));

    guard.charge(previous_flops);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .build()
        .map_err(|e| BrutError::Invalid(format!("thread pool: {e}")))?;

    let mut seen: HashSet<String> = existing.iter().map(|c| c.id.clone()).collect();
    let first_gen = existing
        .iter()
        .map(|c| c.generation)
        .max()
        .map_or(0, |g| g + 1);
    let mut stop_reason = "completed".to_string();

    if first_gen == 0 {
        for (family, reason) in &left_out {
            log_event(
                archive,
                format!(
                    "generation 0: base model {} left out: {reason}",
                    family.name()
                ),
            )?;
        }

        for (id, reason) in &promoted_left_out {
            log_event(
                archive,
                format!("generation 0: promoted model {id} left out: {reason}"),
            )?;
        }
    }

    for generation in first_gen..=config.generations {
        if let Err(e) = guard.check() {
            match e {
                BrutError::Stopped(_) | BrutError::Budget(_) => {
                    stop_reason = e.to_string();
                    log_event(
                        archive,
                        format!("run stopped before generation {generation}: {e}"),
                    )?;

                    break;
                }
                other => return Err(other),
            }
        }

        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(
            config.master_seed,
            &format!("gen/{generation}"),
        ));
        let proposals = propose(generation, &ctx, archive, &mut rng);
        let wanted = wanted_children(config, base_models.len() + promoted.len(), generation);
        let plans = filter_plans(proposals, wanted, &spec, &guard, &mut seen);

        if plans.skipped_budget > 0 {
            log_event(
                archive,
                format!(
                    "generation {generation}: {} children skipped, FLOP budget too small",
                    plans.skipped_budget
                ),
            )?;
        }

        if plans.kept.is_empty() {
            stop_reason = format!("budget: no new child fits in generation {generation}");
            log_event(archive, stop_reason.clone())?;

            break;
        }

        let mut jobs = plans.kept;

        sort_largest_first(&mut jobs, &spec);

        let mut outcomes = train_all(&pool, &jobs, generation, &ctx);

        // Archive order must not depend on thread timing.
        outcomes.sort_by(|a, b| a.entry.id.cmp(&b.entry.id));

        let interrupted = stop.requested();

        if interrupted {
            let cut = outcomes
                .iter()
                .filter(|o| o.entry.train.stopped_by_kill_switch)
                .count();

            log_event(
                archive,
                format!(
                    "generation {generation}: kill switch during training; {} children not \
                     started, {cut} stopped early",
                    jobs.len() - outcomes.len()
                ),
            )?;
        }

        let median = median_valid(&outcomes);

        for outcome in outcomes {
            record_outcome(run_dir, archive, &mut guard, outcome)?;
        }

        progress(generation, median, archive, &guard);

        if interrupted {
            stop_reason = format!(
                "stopped: kill switch {} present during generation {generation}",
                run_dir.join(safety::STOP_FILE).display()
            );

            break;
        }
    }

    if let Some(setup) = eval {
        if stop.requested() || guard.stop_requested() {
            log_event(
                archive,
                "hidden evaluation skipped: kill switch present".to_string(),
            )?;
        } else {
            hidden_phase(run_dir, setup, config, archive)?;
        }
    }

    archive.append(
        RecordKind::RunFinished {
            reason: stop_reason.clone(),
        },
        now_unix_ms(),
    )?;

    Ok(summarize(archive, &guard, stop_reason))
}

/// The base models of the CNN families that join generation 0 of this run, and the families
/// left out, each with the reason (`families::base_models_for`). None for an MLP run.
fn first_base_models(config: &Config, spec: &TaskSpec) -> BaseModels {
    if config.models.is_mlp() {
        return (Vec::new(), Vec::new());
    }

    families::base_models_for(spec, &config.limits)
}

/// The promoted models of the config (`Config::promoted`) that join generation 0 of this
/// run, in their order, and the content ids of those left out, each with the reason: a model
/// of a kind the run does not breed, one that fails `Genome::validate` for the limits and the
/// task, or one already in generation 0 as a base genome, a base model or an earlier promoted
/// model.
fn first_promoted(
    config: &Config,
    spec: &TaskSpec,
    base_models: &[(Family, Genome)],
) -> (Vec<Genome>, Vec<(String, String)>) {
    let mut taken: HashSet<String> = config
        .base_genomes
        .iter()
        .chain(base_models.iter().map(|(_, genome)| genome))
        .map(Genome::id)
        .collect();
    let mut fit = Vec::new();
    let mut left_out = Vec::new();

    for genome in &config.promoted {
        let id = genome.id();
        let reason = match (config.models, genome.arch.is_mlp()) {
            (ModelKinds::Cnn, true) => Some("an MLP, and this run breeds CNNs only".to_string()),
            (ModelKinds::Mlp, false) => Some("a CNN, and this run breeds MLPs only".to_string()),
            _ => match genome.validate(&config.limits, spec) {
                Err(e) => Some(e.to_string()),
                Ok(()) if taken.contains(&id) => Some("already in generation 0".to_string()),
                Ok(()) => None,
            },
        };

        match reason {
            Some(reason) => left_out.push((id, reason)),
            None => {
                taken.insert(id);
                fit.push(genome.clone());
            }
        }
    }

    (fit, left_out)
}

/// How many new children generation `generation` wants: the base genomes, the
/// `base_models` (of the CNN families, and the promoted models that fit) and
/// `population_init` random ones in generation 0, `children_per_generation` later.
fn wanted_children(config: &Config, base_models: usize, generation: usize) -> usize {
    if generation == 0 {
        config.base_genomes.len() + base_models + config.population_init
    } else {
        config.children_per_generation
    }
}

/// The random genome number `index` of generation 0 for a run of these kinds: an MLP, a
/// CNN, or in a run of both kinds an MLP for an even and a CNN for an odd index. `None`
/// when no CNN fits, which `Config::validate` rules out.
fn first_random(
    models: ModelKinds,
    index: usize,
    rng: &mut ChaCha8Rng,
    ctx: &Context<'_>,
) -> Option<Genome> {
    let limits = &ctx.config.limits;
    let cnn = match models {
        ModelKinds::Mlp => false,
        ModelKinds::Cnn => true,
        ModelKinds::Both => index % 2 == 1,
    };

    if cnn {
        cnn_variation::random(rng, limits, ctx.spec)
    } else {
        Some(Genome::random(rng, limits, ctx.spec))
    }
}

/// Make the raw proposals of one generation (before dedup and budget filtering).
///
/// Generation 0 starts with the base genomes of the config, the base models of the CNN
/// families and the promoted models. They draw nothing from `rng`, so an MLP run without
/// base genomes and promoted models draws the same random genomes as before they existed.
fn propose(
    generation: usize,
    ctx: &Context<'_>,
    archive: &Archive,
    rng: &mut ChaCha8Rng,
) -> Vec<Plan> {
    let config = ctx.config;
    let limits = &config.limits;

    if generation == 0 {
        let count = config.population_init * ATTEMPTS_PER_CHILD;
        let base = config.base_genomes.iter().map(|genome| Plan {
            genome: genome.clone(),
            parents: Vec::new(),
            generator: BASE_GENERATOR.to_string(),
        });
        let base_models = ctx.base_models.iter().map(|(family, genome)| Plan {
            genome: genome.clone(),
            parents: Vec::new(),
            generator: format!("{BASE_MODEL_GENERATOR}/{}", family.name()),
        });
        let promoted = ctx.promoted.iter().map(|genome| Plan {
            genome: genome.clone(),
            parents: Vec::new(),
            generator: PROMOTED_GENERATOR.to_string(),
        });
        let random: Vec<Plan> = (0..count)
            .filter_map(|index| first_random(config.models, index, rng, ctx))
            .map(|genome| Plan {
                genome,
                parents: Vec::new(),
                generator: INIT_GENERATOR.to_string(),
            })
            .collect();

        return base
            .chain(base_models)
            .chain(promoted)
            .chain(random)
            .collect();
    }

    let mutator = RandomMutator {
        models: config.models,
        ..RandomMutator::default()
    };
    let count = config.children_per_generation * ATTEMPTS_PER_CHILD;
    let mut plans = Vec::with_capacity(count);

    for _ in 0..count {
        let want = if rng.random::<f64>() < config.crossover_rate {
            2
        } else {
            1
        };
        let mut parent_ids = archive.select_parents(want, rng);

        parent_ids.dedup();

        let parents: Vec<CandidateEntry> =
            parent_ids.iter().filter_map(|id| archive.get(id)).collect();

        if parents.is_empty() {
            plans.push(Plan {
                genome: mutator.random(rng, limits, ctx.spec),
                parents: Vec::new(),
                generator: RESTART_GENERATOR.to_string(),
            });

            continue;
        }

        let refs: Vec<&Genome> = parents.iter().map(|p| &p.genome).collect();
        let (genome, generator) = mutator.propose_named(&refs, rng, limits, ctx.spec);

        plans.push(Plan {
            genome,
            parents: parents.iter().map(|p| p.id.clone()).collect(),
            generator,
        });
    }

    plans
}

struct Filtered {
    kept: Vec<Plan>,
    skipped_budget: usize,
}

/// Keep the first `wanted` new, affordable plans. Updates `seen`.
fn filter_plans(
    proposals: Vec<Plan>,
    wanted: usize,
    spec: &TaskSpec,
    guard: &Guard,
    seen: &mut HashSet<String>,
) -> Filtered {
    let mut kept = Vec::with_capacity(wanted);
    let mut planned_flops = 0u64;
    let mut skipped_budget = 0;

    for plan in proposals {
        if kept.len() == wanted {
            break;
        }

        let id = plan.genome.id();

        if seen.contains(&id) {
            continue;
        }

        let flops = plan.genome.train_flops(spec);

        if !guard.can_afford(planned_flops.saturating_add(flops)) {
            skipped_budget += 1;

            continue;
        }

        planned_flops = planned_flops.saturating_add(flops);
        seen.insert(id);
        kept.push(plan);
    }

    Filtered {
        kept,
        skipped_budget,
    }
}

/// Sort plans by estimated training cost, largest first (stable, so ties keep their order).
///
/// Costs differ by more than 100x within a generation, and the generation waits for its
/// slowest child. Starting the big ones first keeps the threads busy until the end.
fn sort_largest_first(plans: &mut [Plan], spec: &TaskSpec) {
    plans.sort_by_key(|p| std::cmp::Reverse(p.genome.train_flops(spec)));
}

/// Train all plans on the pool and return their outcomes (in no particular order).
///
/// Jobs start in the order given: each free worker takes the oldest job not yet started
/// (`scope_fifo`), a greedy longest-first schedule when `jobs` is sorted by cost.
/// Once the kill switch is seen, jobs not yet started are skipped (no outcome).
fn train_all(
    pool: &rayon::ThreadPool,
    jobs: &[Plan],
    generation: usize,
    ctx: &Context<'_>,
) -> Vec<Outcome> {
    let mut slots: Vec<Option<Outcome>> = jobs.iter().map(|_| None).collect();

    pool.scope_fifo(|scope| {
        for (plan, slot) in jobs.iter().zip(slots.iter_mut()) {
            scope.spawn_fifo(move |_| {
                if !ctx.stop.requested() {
                    *slot = Some(train_child(plan, generation, ctx));
                }
            });
        }
    });

    slots.into_iter().flatten().collect()
}

/// Train and score one child. Runs inside the thread pool.
fn train_child(plan: &Plan, generation: usize, ctx: &Context<'_>) -> Outcome {
    let spec = ctx.spec;
    let genome = &plan.genome;
    let params = genome.param_count(spec);

    let base = CandidateEntry {
        id: genome.id(),
        generation,
        parents: plan.parents.clone(),
        generator: plan.generator.clone(),
        genome: genome.clone(),
        params,
        train: TrainStats {
            epochs_done: 0,
            steps: 0,
            flops: 0,
            wall_seconds: 0.0,
            final_train_loss: f32::MAX,
            diverged: false,
            stopped_by_budget: false,
            stopped_by_kill_switch: false,
        },
        valid: Metrics::failed(spec.objective),
        fitness: WORST_FITNESS,
        model_file: None,
        status: CandidateStatus::Invalid,
    };

    let stop = || ctx.stop.requested();

    match train::train_with_stop(genome, spec, ctx.train, &ctx.config.limits, &stop) {
        Err(e) => Outcome {
            entry: base,
            model: None,
            error: Some(e.to_string()),
        },
        Ok((model, stats)) => {
            let valid = score::evaluate(&model, spec, ctx.valid);

            let status = if stats.diverged {
                CandidateStatus::Diverged
            } else if stats.stopped_by_kill_switch {
                CandidateStatus::Stopped
            } else if stats.stopped_by_budget {
                CandidateStatus::OverBudget
            } else {
                CandidateStatus::Ok
            };

            let fitness = match status {
                CandidateStatus::Ok | CandidateStatus::OverBudget | CandidateStatus::Stopped => {
                    score::fitness(&valid, params, stats.flops, &ctx.config.score)
                }
                _ => WORST_FITNESS,
            };

            let entry = CandidateEntry {
                train: stats,
                valid,
                fitness,
                status,
                ..base
            };
            let model = (status == CandidateStatus::Ok).then_some(model);

            Outcome {
                entry,
                model,
                error: None,
            }
        }
    }
}

/// Save the model (status Ok only), append the records, charge FLOPs.
fn record_outcome(
    run_dir: &Path,
    archive: &mut Archive,
    guard: &mut Guard,
    outcome: Outcome,
) -> Result<()> {
    let Outcome {
        mut entry,
        model,
        error,
    } = outcome;

    if let Some(message) = error {
        log_event(
            archive,
            format!("candidate {} invalid: {message}", entry.id),
        )?;
    }

    if let Some(model) = model {
        let rel = format!("{MODELS_DIR}/{}.json", entry.id);

        write_atomic(&run_dir.join(&rel), &serde_json::to_vec(&model)?)?;
        entry.model_file = Some(rel);
    }

    guard.charge(entry.train.flops);
    archive.append(RecordKind::Candidate(entry), now_unix_ms())?;

    Ok(())
}

/// The fixed set of candidates for the hidden evaluation: the `top_k` best by validation
/// fitness among trained candidates (status Ok or Quarantined), ties by id, and then, with
/// `with_base`, the best base model (`best_base_model`) if it is not among them. Empty for a
/// `top_k` of 0, which turns the hidden evaluation off.
///
/// The best base model is in the set so that a promotion (`crate::promote`) can compare the
/// winners with it on the hidden split. Callers pass `Config::can_promote` as `with_base`.
/// So a run whose config has the model kinds `Mlp` and no promoted models, as every run
/// before CNNs, has the same set as before, even with base genomes in its config.
///
/// Quarantined candidates keep their place. Otherwise each new run would move down the
/// ranking and send the next candidates, so hidden results would steer which candidate
/// passes in the end. As no generations follow a hidden evaluation, this set never changes.
pub(crate) fn hidden_set(archive: &Archive, top_k: usize, with_base: bool) -> Vec<CandidateEntry> {
    if top_k == 0 {
        return Vec::new();
    }

    let mut trained: Vec<CandidateEntry> = archive
        .candidates()
        .into_iter()
        .filter(|c| matches!(c.status, CandidateStatus::Ok | CandidateStatus::Quarantined))
        .collect();

    trained.sort_by(|a, b| {
        b.fitness
            .total_cmp(&a.fitness)
            .then_with(|| a.id.cmp(&b.id))
    });

    let base = if with_base {
        best_base_model(&trained).cloned()
    } else {
        None
    };

    trained.truncate(top_k);

    if let Some(base) = base
        && !trained.iter().any(|c| c.id == base.id)
    {
        trained.push(base);
    }

    trained
}

/// Send the fixed hidden set (see `hidden_set`) to the hidden evaluator, skipping candidates
/// that already have a result. Errors are logged, not fatal.
fn hidden_phase(
    run_dir: &Path,
    setup: &EvalSetup,
    config: &Config,
    archive: &mut Archive,
) -> Result<()> {
    if config.hidden_eval_top_k == 0 {
        return Ok(());
    }

    let done: HashSet<String> = archive
        .hidden_evals()
        .into_iter()
        .map(|h| h.candidate_id)
        .collect();

    let stop_file = run_dir.join(safety::STOP_FILE);

    for cand in hidden_set(archive, config.hidden_eval_top_k, config.can_promote()) {
        if done.contains(&cand.id) {
            continue;
        }

        if stop_file.exists() {
            return log_event(
                archive,
                "hidden evaluation stopped: kill switch present".to_string(),
            );
        }

        let Some(model_file) = &cand.model_file else {
            log_event(
                archive,
                format!("hidden evaluation skipped for {}: no model file", cand.id),
            )?;

            continue;
        };

        let req = EvalRequest {
            candidate_id: cand.id.clone(),
            model_path: run_dir.join(model_file),
            task: config.task,
        };

        let resp = request_hidden_eval(
            &setup.eval_bin,
            &setup.secret_env,
            config.hidden_query_budget,
            config.hidden_score_step,
            &req,
            Duration::from_secs_f64(EVAL_TIMEOUT_SECONDS),
            Some(&stop_file),
        );

        match resp {
            Ok(resp) => {
                let valid_primary = cand.valid.primary;
                let hidden_primary = resp.hidden_primary;
                let gap = valid_primary - hidden_primary;
                let quarantined = safety::gap_exceeded(
                    valid_primary,
                    hidden_primary,
                    config.hidden_gap_threshold,
                );

                eprintln!(
                    "hidden  {} | valid {valid_primary:.4} | hidden {hidden_primary:.4} | gap {gap:+.4}{}",
                    cand.id,
                    if quarantined { " | QUARANTINED" } else { "" }
                );

                let record = RecordKind::HiddenEval {
                    candidate_id: cand.id.clone(),
                    valid_primary,
                    hidden_primary,
                    gap,
                    quarantined,
                };

                archive.append(record, now_unix_ms())?;
            }
            Err(BrutError::Stopped(m)) => {
                return log_event(
                    archive,
                    format!("hidden evaluation stopped at {}: {m}", cand.id),
                );
            }
            Err(e) => log_event(
                archive,
                format!("hidden evaluation failed for {}: {e}", cand.id),
            )?,
        }
    }

    Ok(())
}

fn log_event(archive: &mut Archive, what: String) -> Result<()> {
    eprintln!("safety  {what}");
    archive.append(RecordKind::SafetyEvent { what }, now_unix_ms())?;

    Ok(())
}

/// Best eligible candidate: status Ok, highest fitness, ties by id.
fn best(archive: &Archive) -> Option<CandidateEntry> {
    archive
        .candidates()
        .into_iter()
        .filter(|c| c.status == CandidateStatus::Ok)
        .max_by(|a, b| {
            a.fitness
                .total_cmp(&b.fitness)
                .then_with(|| b.id.cmp(&a.id))
        })
}

/// Median validation score of the Ok children of one generation (0 if none).
fn median_valid(outcomes: &[Outcome]) -> f64 {
    let mut values: Vec<f64> = outcomes
        .iter()
        .filter(|o| o.entry.status == CandidateStatus::Ok)
        .map(|o| o.entry.valid.primary)
        .collect();

    if values.is_empty() {
        return 0.0;
    }

    values.sort_by(f64::total_cmp);

    let mid = values.len() / 2;

    if values.len() % 2 == 1 {
        values[mid]
    } else {
        0.5 * (values[mid - 1] + values[mid])
    }
}

/// One progress line on stderr.
fn progress(generation: usize, median: f64, archive: &Archive, guard: &Guard) {
    let (fitness, valid) = match best(archive) {
        Some(b) => (b.fitness, b.valid.primary),
        None => (WORST_FITNESS, 0.0),
    };

    eprintln!(
        "gen {generation:>3} | best fitness {fitness:.4} | best valid {valid:.4} | gen median valid {median:.4} | flops {:.3e} | {:.1} s",
        guard.flops_used() as f64,
        guard.elapsed_seconds()
    );
}

fn summarize(archive: &Archive, guard: &Guard, stop_reason: String) -> RunSummary {
    let candidates = archive.candidates();
    let best = best(archive);

    let hidden = archive
        .hidden_evals()
        .into_iter()
        .map(|h| HiddenResult {
            candidate_id: h.candidate_id,
            valid_primary: h.valid_primary,
            hidden_primary: h.hidden_primary,
            quarantined: h.quarantined,
        })
        .collect();

    RunSummary {
        generations_done: candidates.iter().map(|c| c.generation).max().unwrap_or(0),
        candidates: candidates.len(),
        best_id: best.as_ref().map(|b| b.id.clone()),
        best_fitness: best.as_ref().map_or(WORST_FITNESS, |b| b.fitness),
        best_valid_primary: best.as_ref().map_or(0.0, |b| b.valid.primary),
        flops_used: guard.flops_used(),
        wall_seconds: guard.elapsed_seconds(),
        stop_reason,
        hidden,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnn::families::FAMILIES;
    use crate::config::Limits;
    use crate::genome::{Activation, Arch, Optimizer};
    use crate::task::TaskKind;

    /// The proposals of generation `generation` of a run with `config` and an empty archive,
    /// drawn as `run_inner` draws them, and the families whose base models stay out.
    fn proposals(config: &Config, generation: usize) -> (Vec<Plan>, Vec<(Family, String)>) {
        // Tests run in parallel, and two of them may ask for the same proposals.
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

        let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let spec = config.task.spec();
        let train = task::generate(&spec, Split::Train, 1);
        let valid = task::generate(&spec, Split::Valid, 1);
        let dir = std::env::temp_dir().join(format!(
            "brut-evolve-proposals-{}-{}-{}-{generation}-{call}",
            std::process::id(),
            config.models.name(),
            config.task.name()
        ));

        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");

        let archive = Archive::create(&dir.join(ARCHIVE_FILE)).expect("archive");
        let stop = StopSignal::new(&dir);
        let (base_models, left_out) = first_base_models(config, &spec);
        let (promoted, _) = first_promoted(config, &spec, &base_models);
        let ctx = Context {
            config,
            spec: &spec,
            train: &train,
            valid: &valid,
            stop: &stop,
            base_models: &base_models,
            promoted: &promoted,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(
            config.master_seed,
            &format!("gen/{generation}"),
        ));
        let plans = propose(generation, &ctx, &archive, &mut rng);

        let _ = std::fs::remove_dir_all(&dir);

        (plans, left_out)
    }

    #[test]
    fn generation_0_of_a_cnn_run_holds_every_base_model_that_fits() {
        let limits = Limits::default();
        let on_grid = [Family::ResNet, Family::SqueezeNet, Family::UNet];

        for (task, fit) in [
            (
                TaskKind::Shapes,
                FAMILIES
                    .into_iter()
                    .filter(|f| !matches!(f, Family::UNet | Family::Tcn))
                    .collect::<Vec<_>>(),
            ),
            (
                TaskKind::Motif1d,
                FAMILIES
                    .into_iter()
                    .filter(|f| *f != Family::UNet)
                    .collect(),
            ),
            (TaskKind::MirrorGrid, on_grid.to_vec()),
        ] {
            let config = Config {
                task,
                ..Config::for_models(ModelKinds::Cnn)
            };
            let spec = task.spec();
            let (plans, left_out) = proposals(&config, 0);
            let left: Vec<Family> = left_out.iter().map(|(family, _)| *family).collect();

            assert_eq!(
                left,
                FAMILIES
                    .into_iter()
                    .filter(|f| !fit.contains(f))
                    .collect::<Vec<_>>()
            );
            assert!(
                left_out
                    .iter()
                    .all(|(_, reason)| reason.starts_with("invalid: genome: "))
            );

            // First the base models, in the order of the families; on the motif task the
            // 2-D ones as their 1-D twins.
            for (plan, family) in plans.iter().zip(&fit) {
                assert_eq!(plan.generator, format!("base-model/{}", family.name()));
                assert_eq!(plan.genome, family.base_model_for(&spec));
                assert!(plan.parents.is_empty());
                assert!(plan.genome.validate(&limits, &spec).is_ok());
            }

            // Then the random CNNs.
            let random = &plans[fit.len()..];

            assert_eq!(random.len(), config.population_init * ATTEMPTS_PER_CHILD);

            for plan in random {
                assert_eq!(plan.generator, INIT_GENERATOR);
                assert!(!plan.genome.arch.is_mlp());
                assert!(plan.genome.validate(&limits, &spec).is_ok());
            }

            assert_eq!(
                wanted_children(&config, fit.len(), 0),
                fit.len() + config.population_init
            );
        }
    }

    #[test]
    fn a_run_of_both_kinds_starts_with_mlps_and_cnns_in_turn() {
        let config = Config::for_models(ModelKinds::Both);
        let spec = config.task.spec();
        let (plans, _) = proposals(&config, 0);
        let random: Vec<&Plan> = plans
            .iter()
            .filter(|plan| plan.generator == INIT_GENERATOR)
            .collect();

        assert_eq!(plans.len() - random.len(), 14);
        assert_eq!(random.len(), config.population_init * ATTEMPTS_PER_CHILD);

        for (i, plan) in random.iter().enumerate() {
            assert_eq!(plan.genome.arch.is_mlp(), i % 2 == 0);
            assert!(plan.genome.validate(&config.limits, &spec).is_ok());
        }
    }

    #[test]
    fn an_mlp_run_draws_its_random_genomes_as_before() {
        // Before CNN runs, generation 0 was `Genome::random` in a row, and a restart drew
        // the crossover choice, then `Genome::random`.
        let config = Config {
            task: TaskKind::Shapes,
            ..Config::default()
        };
        let spec = config.task.spec();
        let limits = &config.limits;
        let (first, left_out) = proposals(&config, 0);
        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(config.master_seed, "gen/0"));

        assert!(left_out.is_empty());
        assert_eq!(first.len(), config.population_init * ATTEMPTS_PER_CHILD);

        for plan in &first {
            assert_eq!(plan.genome, Genome::random(&mut rng, limits, &spec));
            assert_eq!(plan.generator, INIT_GENERATOR);
        }

        let (restarts, _) = proposals(&config, 1);
        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(config.master_seed, "gen/1"));

        for plan in &restarts {
            let _: f64 = rng.random();

            assert_eq!(plan.genome, Genome::random(&mut rng, limits, &spec));
            assert_eq!(plan.generator, RESTART_GENERATOR);
        }
    }

    #[test]
    fn promoted_models_join_generation_0_after_the_base_models() {
        let config = Config::for_models(ModelKinds::Cnn);
        let spec = config.task.spec();
        let (base_models, _) = first_base_models(&config, &spec);
        let mut bred = Family::ResNet.base_model();
        let mut too_long = Family::Vgg.base_model();
        let mlp = Genome::random(&mut ChaCha8Rng::seed_from_u64(3), &config.limits, &spec);

        bred.seed += 1;
        too_long.seed += 1;
        too_long.epochs = config.limits.max_epochs + 1;

        let config = Config {
            promoted: vec![
                bred.clone(),
                mlp.clone(),
                too_long.clone(),
                Family::Vgg.base_model(),
                bred.clone(),
            ],
            ..config
        };
        let (fit, left_out) = first_promoted(&config, &spec, &base_models);
        let reasons: Vec<(String, &str)> = left_out
            .iter()
            .map(|(id, reason)| (id.clone(), reason.as_str()))
            .collect();

        assert_eq!(fit, vec![bred.clone()]);
        assert_eq!(reasons.len(), 4);
        assert_eq!(
            reasons[0],
            (mlp.id(), "an MLP, and this run breeds CNNs only")
        );
        assert_eq!(reasons[1].0, too_long.id());
        assert!(
            reasons[1].1.starts_with("invalid: genome: epochs "),
            "{}",
            reasons[1].1
        );
        assert_eq!(
            reasons[2],
            (Family::Vgg.base_model().id(), "already in generation 0")
        );
        assert_eq!(reasons[3], (bred.id(), "already in generation 0"));

        // In generation 0 the promoted model comes right after the base models of the
        // families, and the random genomes follow as without it.
        let (plans, _) = proposals(&config, 0);
        let (without, _) = proposals(
            &Config {
                promoted: Vec::new(),
                ..config.clone()
            },
            0,
        );
        let at = base_models.len();

        assert_eq!(plans[at].genome, bred);
        assert_eq!(plans[at].generator, PROMOTED_GENERATOR);
        assert!(plans[at].parents.is_empty());
        assert!(is_base_model(&plans[at].generator));
        assert_eq!(plans.len(), without.len() + 1);

        for (plan, before) in plans[at + 1..].iter().zip(&without[at..]) {
            assert_eq!(plan.genome, before.genome);
            assert_eq!(plan.generator, before.generator);
        }

        assert_eq!(
            wanted_children(&config, at + fit.len(), 0),
            at + 1 + config.population_init
        );

        // An MLP run takes a promoted MLP and leaves a promoted CNN out; a run of both kinds
        // takes both.
        let mlp_run = Config {
            task: TaskKind::Shapes,
            promoted: vec![bred.clone(), mlp.clone()],
            ..Config::default()
        };
        let (fit, left_out) = first_promoted(&mlp_run, &spec, &[]);

        assert_eq!(fit, vec![mlp.clone()]);
        assert_eq!(
            left_out,
            vec![(
                bred.id(),
                "a CNN, and this run breeds MLPs only".to_string()
            )]
        );

        let both = Config {
            models: ModelKinds::Both,
            ..mlp_run
        };
        let (fit, left_out) = first_promoted(&both, &spec, &[]);

        assert_eq!(fit, vec![bred, mlp]);
        assert!(left_out.is_empty());
    }

    #[test]
    fn an_mlp_run_with_a_promoted_mlp_draws_its_random_genomes_as_before() {
        let spec = TaskKind::TwoSpirals.spec();
        let limits = Limits::default();
        let promoted = Genome::random(&mut ChaCha8Rng::seed_from_u64(99), &limits, &spec);
        let config = Config {
            promoted: vec![promoted.clone()],
            ..Config::default()
        };
        let (plans, _) = proposals(&config, 0);
        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(config.master_seed, "gen/0"));

        assert_eq!(plans[0].genome, promoted);
        assert_eq!(plans[0].generator, PROMOTED_GENERATOR);

        for plan in &plans[1..] {
            assert_eq!(plan.genome, Genome::random(&mut rng, &limits, &spec));
            assert_eq!(plan.generator, INIT_GENERATOR);
        }
    }

    /// A trained candidate with this seed, generator, fitness and status.
    fn entry(seed: u64, generator: &str, fitness: f64, status: CandidateStatus) -> CandidateEntry {
        let mut entry = outcome(fitness, status).entry;

        entry.genome.seed = seed;
        entry.id = entry.genome.id();
        entry.generator = generator.to_string();

        entry
    }

    #[test]
    fn the_hidden_set_adds_the_best_base_model() {
        let dir = std::env::temp_dir().join(format!("brut-evolve-hidden-{}", std::process::id()));

        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");

        let mut archive = Archive::create(&dir.join(ARCHIVE_FILE)).expect("archive");
        let entries = [
            entry(1, "random-mutator-v1", 0.9, CandidateStatus::Ok),
            entry(2, INIT_GENERATOR, 0.8, CandidateStatus::Ok),
            entry(3, "base-model/resnet", 0.95, CandidateStatus::Diverged),
            entry(4, "base-model/vgg", 0.7, CandidateStatus::Ok),
            entry(5, PROMOTED_GENERATOR, 0.75, CandidateStatus::Ok),
            entry(6, BASE_GENERATOR, 0.6, CandidateStatus::Ok),
        ];

        for e in &entries {
            archive
                .append(RecordKind::Candidate(e.clone()), 0)
                .expect("append");
        }

        let ids =
            |set: Vec<CandidateEntry>| -> Vec<String> { set.into_iter().map(|c| c.id).collect() };

        // The two best by fitness, then the best trained base model: the promoted one.
        assert_eq!(
            ids(hidden_set(&archive, 2, true)),
            vec![
                entries[0].id.clone(),
                entries[1].id.clone(),
                entries[4].id.clone()
            ]
        );

        // Already among the best: nothing is added.
        assert_eq!(ids(hidden_set(&archive, 3, true)).len(), 3);
        assert!(hidden_set(&archive, 0, true).is_empty());

        // A run that cannot promote keeps the set of the best by fitness alone.
        assert_eq!(
            ids(hidden_set(&archive, 2, false)),
            vec![entries[0].id.clone(), entries[1].id.clone()]
        );

        let candidates = archive.candidates();

        assert_eq!(
            best_base_model(&candidates).map(|c| c.id.as_str()),
            Some(entries[4].id.as_str())
        );

        for (generator, base) in [
            ("base-model/densenet", true),
            ("base-model", false),
            ("base-modelx/vgg", false),
            (BASE_GENERATOR, true),
            (PROMOTED_GENERATOR, true),
            ("promoted/abc", false),
            (INIT_GENERATOR, false),
            (RESTART_GENERATOR, false),
            ("cnn-mutator-v1/crossover", false),
        ] {
            assert_eq!(is_base_model(generator), base, "{generator}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_cnn_run_restarts_with_random_cnns() {
        let config = Config::for_models(ModelKinds::Cnn);
        let spec = config.task.spec();
        let (plans, _) = proposals(&config, 1);

        assert_eq!(
            plans.len(),
            config.children_per_generation * ATTEMPTS_PER_CHILD
        );

        for plan in plans {
            assert_eq!(plan.generator, RESTART_GENERATOR);
            assert!(!plan.genome.arch.is_mlp());
            assert!(plan.genome.validate(&config.limits, &spec).is_ok());
        }
    }

    fn plan(seed: u64, width: usize) -> Plan {
        Plan {
            genome: Genome {
                arch: Arch::Mlp,
                hidden: vec![width],
                activation: Activation::Relu,
                optimizer: Optimizer::Adam,
                learning_rate: 0.01,
                l2: 0.0,
                batch_size: 32,
                epochs: 10,
                init_scale: 1.0,
                seed,
            },
            parents: Vec::new(),
            generator: "test".into(),
        }
    }

    fn outcome(primary: f64, status: CandidateStatus) -> Outcome {
        let p = plan(1, 4);
        let entry = CandidateEntry {
            id: p.genome.id(),
            generation: 0,
            parents: Vec::new(),
            generator: p.generator,
            genome: p.genome,
            params: 1,
            train: TrainStats {
                epochs_done: 1,
                steps: 1,
                flops: 1,
                wall_seconds: 0.0,
                final_train_loss: 0.0,
                diverged: false,
                stopped_by_budget: false,
                stopped_by_kill_switch: false,
            },
            valid: Metrics {
                loss: 0.0,
                accuracy: Some(primary),
                r2: None,
                primary,
            },
            fitness: primary,
            model_file: None,
            status,
        };

        Outcome {
            entry,
            model: None,
            error: None,
        }
    }

    #[test]
    fn filter_skips_duplicates_and_unaffordable() {
        let spec = TaskKind::TwoSpirals.spec();
        let dir = std::env::temp_dir();
        let small = plan(1, 4);
        let flops = small.genome.train_flops(&spec);

        // Budget fits exactly two small plans.
        let limits = Limits {
            max_total_flops: 2 * flops,
            ..Limits::default()
        };
        let guard = Guard::new(&dir, &limits);
        let mut seen = HashSet::new();

        seen.insert(plan(9, 4).genome.id());

        // 8 proposals, 2 wanted (as many as ATTEMPTS_PER_CHILD = 4 makes for 2 children).
        let proposals = vec![
            plan(9, 4),
            plan(1, 4),
            plan(1, 4),
            plan(2, 64),
            plan(3, 4),
            plan(4, 4),
            plan(5, 4),
            plan(6, 4),
        ];
        let out = filter_plans(proposals, 2, &spec, &guard, &mut seen);
        let seeds: Vec<u64> = out.kept.iter().map(|p| p.genome.seed).collect();

        assert_eq!(seeds, vec![1, 3]);
        assert_eq!(out.skipped_budget, 1);
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn plans_are_scheduled_largest_first() {
        let spec = TaskKind::TwoSpirals.spec();
        let mut plans = vec![
            plan(1, 4),
            plan(2, 64),
            plan(3, 16),
            plan(4, 64),
            plan(5, 8),
        ];

        sort_largest_first(&mut plans, &spec);

        let flops: Vec<u64> = plans.iter().map(|p| p.genome.train_flops(&spec)).collect();
        let seeds: Vec<u64> = plans.iter().map(|p| p.genome.seed).collect();

        assert!(flops.windows(2).all(|w| w[0] >= w[1]), "{flops:?}");

        // Equal cost keeps the proposal order.
        assert_eq!(seeds, vec![2, 4, 3, 5, 1]);
    }

    #[test]
    fn median_uses_ok_children_only() {
        let outcomes = vec![
            outcome(0.2, CandidateStatus::Ok),
            outcome(0.9, CandidateStatus::Diverged),
            outcome(0.6, CandidateStatus::Ok),
            outcome(0.4, CandidateStatus::Ok),
        ];

        assert_eq!(median_valid(&outcomes), 0.4);
        assert_eq!(median_valid(&outcomes[..1]), 0.2);
        assert_eq!(median_valid(&[]), 0.0);
    }
}
