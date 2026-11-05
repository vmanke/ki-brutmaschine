//! Promotion of the winners of a finished MLP-mode run (ticket 07, ADR 0008).
//!
//! A run promotes at most two models: the two best by their score on the hidden split, which
//! the run never saw while it bred. Each of them is promoted only if it beats the best base
//! model of the same run on that split (`evolve::best_base_model`; the hidden set holds it,
//! `evolve::hidden_set`). A model equal by content to one in the pool, or to a base model of a
//! CNN family, adds nothing. Each promoted model is a `hitl::PromotedModel`: its genome with
//! its provenance (run, task, seed, commit, scores, and the lineage back to generation 0).
//!
//! `brut promote` writes the promoted models as files for a draft pull request into
//! `candidates/models/`, which a human reviews and merges (ADR 0004). The run never merges
//! anything. A later run takes the merged models into its first generation (`brut init
//! --promoted`, `Config::promoted`), next to the base models of the CNN families.
//!
//! This module is pure: it reads an archive that the caller opened and returns what to
//! write. The binary does the file I/O; git and the pull request live in `scripts/`.
#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::archive::{Archive, CandidateEntry, CandidateStatus, HiddenEvalEntry};
use crate::evolve::{best_base_model, hidden_set, is_base_model};
use crate::hitl::{
    BaseModelScore, CandidateAdapter, LineageStep, ModelAdapter, ModelProvenance, PromotedModel,
    is_commit,
};
use crate::task::TaskKind;
use crate::util::{BrutError, Result};

/// How many models one promotion takes at most: the two best by hidden score. Cascades are
/// promoted two at a time as well (ADR 0005).
pub const PROMOTE_COUNT: usize = 2;

/// The directory inside a run directory that `brut promote` writes, as `brut-breed --promote`
/// writes one for Cascades. It holds `candidates/models/<content-id>.json` and the manifest
/// `proposal.json` for `scripts/propose-pr.sh`.
pub const PROMOTE_DIR: &str = "promote";

/// One of the two best models of a run by hidden score, and what became of it.
#[derive(Clone, Debug, PartialEq)]
pub struct Considered {
    /// Its candidate id, which is also its content id.
    pub id: String,
    pub hidden_primary: f64,
    /// The promoted model, or the reason why it is not promoted.
    pub outcome: std::result::Result<PromotedModel, String>,
}

/// The promotion of one finished run.
#[derive(Clone, Debug, PartialEq)]
pub struct Promotion {
    /// `Archive::run_id` of the run.
    pub run_id: String,
    pub task: TaskKind,
    pub master_seed: u64,
    pub commit: String,
    /// The best base model of the run, the one to beat.
    pub best_base_model: BaseModelScore,
    /// The two best models by hidden score, best first. Fewer when fewer have a hidden score.
    pub considered: Vec<Considered>,
}

impl Promotion {
    /// The promoted models, best first.
    pub fn promoted(&self) -> impl Iterator<Item = &PromotedModel> {
        self.considered
            .iter()
            .filter_map(|c| c.outcome.as_ref().ok())
    }

    /// The manifest for the body of the pull request. It holds no path and no secret.
    pub fn manifest(&self) -> PromotionManifest {
        let candidates = self
            .promoted()
            .map(|m| {
                let p = &m.provenance;
                let (generation, generator) = p
                    .lineage
                    .first()
                    .map(|s| (s.generation, s.generator.clone()))
                    .unwrap_or_default();

                PromotedEntry {
                    id: m.genome.id(),
                    generation,
                    generator,
                    valid_primary: p.valid_primary,
                    hidden_primary: p.hidden_primary,
                    roots: p.roots().into_iter().map(str::to_string).collect(),
                }
            })
            .collect();
        let not_promoted = self
            .considered
            .iter()
            .filter_map(|c| {
                c.outcome.as_ref().err().map(|reason| NotPromoted {
                    id: c.id.clone(),
                    hidden_primary: c.hidden_primary,
                    reason: reason.clone(),
                })
            })
            .collect();

        PromotionManifest {
            tool: "brut promote".to_string(),
            adapter: ModelAdapter::new().kind().to_string(),
            master_seed: self.master_seed,
            run_id: self.run_id.clone(),
            task: self.task,
            commit: self.commit.clone(),
            best_base_model: self.best_base_model.clone(),
            candidates,
            not_promoted,
        }
    }
}

/// A promoted model as the manifest lists it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromotedEntry {
    pub id: String,
    pub generation: usize,
    pub generator: String,
    pub valid_primary: f64,
    pub hidden_primary: f64,
    /// `ModelProvenance::roots`: the base models and random genomes it descends from.
    pub roots: Vec<String>,
}

/// One of the two best that is not promoted, with the reason.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NotPromoted {
    pub id: String,
    pub hidden_primary: f64,
    pub reason: String,
}

/// The manifest of a promotion: `brut promote` writes it as `proposal.json` next to the
/// candidate files, and `scripts/propose-pr.sh` puts it into the body of the draft pull
/// request. It is not itself committed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromotionManifest {
    pub tool: String,
    pub adapter: String,
    pub master_seed: u64,
    pub run_id: String,
    pub task: TaskKind,
    pub commit: String,
    pub best_base_model: BaseModelScore,
    pub candidates: Vec<PromotedEntry>,
    pub not_promoted: Vec<NotPromoted>,
}

/// The promotion of the finished run in `archive`.
///
/// The two best models by hidden score are those of status Ok that are no base model and
/// have a hidden score, ranked by it, then by validation fitness, then by id. Each is
/// promoted if its hidden score is above that of the best base model, its content id is not
/// in `known` (the pool and `hitl::model_seed_ids`), and it passes `adapter`. A model that
/// fails one of these is not replaced by the next one, so a second promotion of the same run
/// promotes nothing new.
///
/// An error when `commit` is no commit, when the archive holds no MLP-mode run, when its
/// config cannot promote (`Config::can_promote`: the model kinds `Mlp` and no promoted
/// models), when the run has no hidden evaluation or an incomplete one, or when it has no
/// base model.
pub fn choose(
    archive: &Archive,
    commit: &str,
    known: &HashSet<String>,
    adapter: &ModelAdapter,
) -> Result<Promotion> {
    let fail = |message: String| Err(BrutError::Invalid(format!("promote: {message}")));

    if !is_commit(commit) {
        return fail(format!(
            "commit {commit:?} must be 7 to 64 lower-case hex digits: the git commit of the \
             code that ran the run"
        ));
    }

    let (Some(config), Some(run_id)) = (archive.first_config(), archive.run_id()) else {
        return fail("the archive holds no run of the MLP mode".to_string());
    };

    if !config.can_promote() {
        return fail(
            "the config of the run has the model kinds mlp and no promoted models, as every \
             config before CNNs, so its hidden set holds no base model to beat \
             (Config::can_promote); a run can promote when it breeds CNNs (--models cnn or \
             both) or starts from promoted models (--promoted)"
                .to_string(),
        );
    }
    let hidden: HashMap<String, HiddenEvalEntry> = archive
        .hidden_evals()
        .into_iter()
        .map(|h| (h.candidate_id.clone(), h))
        .collect();

    if hidden.is_empty() {
        return fail(
            "the run has no hidden evaluation, and a promotion compares models on the hidden \
             split (run brut run without --no-hidden)"
                .to_string(),
        );
    }

    let set = hidden_set(archive, config.hidden_eval_top_k, true);
    let missing = set.iter().filter(|c| !hidden.contains_key(&c.id)).count();

    if missing > 0 {
        return fail(format!(
            "{missing} of the {} candidates of the hidden set have no hidden score yet; run \
             brut run on the run directory again to finish the hidden evaluation",
            set.len()
        ));
    }

    let candidates = archive.candidates();
    let Some(base) = best_base_model(&candidates) else {
        return fail(
            "the run has no base model, so no model of it can beat one (a run has base models \
             when a CNN family fits its task and limits, or when its config holds base \
             genomes or promoted models)"
                .to_string(),
        );
    };
    let Some(base_hidden) = hidden.get(&base.id) else {
        return fail(format!(
            "the best base model {} has no hidden score",
            base.id
        ));
    };
    let best = BaseModelScore {
        id: base.id.clone(),
        generator: base.generator.clone(),
        valid_primary: base.valid.primary,
        hidden_primary: base_hidden.hidden_primary,
    };

    let mut ranked: Vec<(&CandidateEntry, &HiddenEvalEntry)> = candidates
        .iter()
        .filter(|c| c.status == CandidateStatus::Ok && !is_base_model(&c.generator))
        .filter_map(|c| hidden.get(&c.id).filter(|h| !h.quarantined).map(|h| (c, h)))
        .collect();

    ranked.sort_by(|(a, ha), (b, hb)| {
        hb.hidden_primary
            .total_cmp(&ha.hidden_primary)
            .then_with(|| b.fitness.total_cmp(&a.fitness))
            .then_with(|| a.id.cmp(&b.id))
    });

    let provenance = |c: &CandidateEntry, h: &HiddenEvalEntry| ModelProvenance {
        run_id: run_id.to_string(),
        task: config.task,
        master_seed: config.master_seed,
        commit: commit.to_string(),
        valid_primary: c.valid.primary,
        hidden_primary: h.hidden_primary,
        best_base_model: best.clone(),
        lineage: archive
            .lineage(&c.id)
            .into_iter()
            .map(|step| LineageStep {
                id: step.id,
                generation: step.generation,
                generator: step.generator,
                parents: step.parents,
            })
            .collect(),
    };
    let considered = ranked
        .into_iter()
        .take(PROMOTE_COUNT)
        .map(|(c, h)| {
            let outcome = if h.hidden_primary <= best.hidden_primary {
                Err(format!(
                    "its hidden score {:.4} does not beat the {:.4} of the best base model",
                    h.hidden_primary, best.hidden_primary
                ))
            } else if known.contains(&c.genome.id()) {
                Err("a duplicate: the pool or a CNN family holds it already".to_string())
            } else {
                let model = PromotedModel {
                    genome: c.genome.clone(),
                    provenance: provenance(c, h),
                };

                adapter
                    .validate(&model)
                    .map(|()| model)
                    .map_err(|e| format!("it fails the check of the pool: {e}"))
            };

            Considered {
                id: c.id.clone(),
                hidden_primary: h.hidden_primary,
                outcome,
            }
        })
        .collect();

    Ok(Promotion {
        run_id: run_id.to_string(),
        task: config.task,
        master_seed: config.master_seed,
        commit: commit.to_string(),
        best_base_model: best,
        considered,
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::archive::RecordKind;
    use crate::cnn::families::Family;
    use crate::config::{Config, Limits, ModelKinds};
    use crate::evolve::{BASE_MODEL_GENERATOR, INIT_GENERATOR};
    use crate::genome::{Activation, Arch, Genome, Optimizer};
    use crate::hitl::{model_seed_ids, path_in_scope};
    use crate::score::Metrics;
    use crate::train::TrainStats;

    const COMMIT: &str = "4613e2a";

    /// An MLP genome for the shapes task, which fits the default limits.
    fn mlp(seed: u64) -> Genome {
        Genome {
            arch: Arch::Mlp,
            hidden: vec![8],
            activation: Activation::Relu,
            optimizer: Optimizer::Adam,
            learning_rate: 0.01,
            l2: 0.0,
            batch_size: 32,
            epochs: 10,
            init_scale: 1.0,
            seed,
        }
    }

    /// A trained candidate of the shapes task with this genome and record.
    fn entry(
        genome: Genome,
        generation: usize,
        generator: &str,
        parents: &[&CandidateEntry],
        fitness: f64,
    ) -> CandidateEntry {
        CandidateEntry {
            id: genome.id(),
            generation,
            parents: parents.iter().map(|p| p.id.clone()).collect(),
            generator: generator.to_string(),
            params: 1,
            train: TrainStats {
                epochs_done: genome.epochs,
                steps: 1,
                flops: 1,
                wall_seconds: 0.0,
                final_train_loss: 0.5,
                diverged: false,
                stopped_by_budget: false,
                stopped_by_kill_switch: false,
            },
            genome,
            valid: Metrics {
                loss: 0.5,
                accuracy: Some(fitness + 0.01),
                r2: None,
                primary: fitness + 0.01,
            },
            fitness,
            model_file: Some("models/x.json".to_string()),
            status: CandidateStatus::Ok,
        }
    }

    /// A finished run of the shapes task, written to a fresh archive: the ResNet base model,
    /// a random genome, and four children with the hidden scores `hidden` (two of the base
    /// model, two of the random genome, the later ones fitter). The base model scores
    /// `base_hidden` on the hidden split. `skip` names candidates left without a hidden
    /// score.
    struct Run {
        dir: PathBuf,
        archive: Archive,
        base: CandidateEntry,
        random: CandidateEntry,
        children: Vec<CandidateEntry>,
    }

    impl Drop for Run {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn run(name: &str, base_hidden: f64, hidden: [f64; 4], skip: &[usize]) -> Run {
        let dir = std::env::temp_dir().join(format!("brut-promote-{name}-{}", std::process::id()));

        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");

        let config = Config {
            hidden_eval_top_k: 4,
            ..Config::for_models(ModelKinds::Cnn)
        };
        let base = entry(
            Family::ResNet.base_model(),
            0,
            &format!("{BASE_MODEL_GENERATOR}/resnet"),
            &[],
            0.5,
        );
        let random = entry(mlp(1), 0, INIT_GENERATOR, &[], 0.4);
        let children: Vec<CandidateEntry> = (0..4)
            .map(|i| {
                let parent = if i < 2 { &base } else { &random };

                entry(
                    mlp(100 + i as u64),
                    1,
                    "random-mutator-v1",
                    &[parent],
                    0.6 + 0.01 * i as f64,
                )
            })
            .collect();
        let mut archive = Archive::create(&dir.join("archive.jsonl")).expect("archive");
        let mut append = |kind: RecordKind| archive.append(kind, 0).map(|_| ()).expect("append");

        append(RecordKind::RunStarted {
            config,
            crate_version: "test".to_string(),
        });

        for c in [&base, &random].into_iter().chain(&children) {
            append(RecordKind::Candidate(c.clone()));
        }

        let scores = [(&base, base_hidden)]
            .into_iter()
            .chain(children.iter().zip(hidden));

        for (i, (c, score)) in scores.enumerate() {
            if !skip.contains(&i) {
                append(RecordKind::HiddenEval {
                    candidate_id: c.id.clone(),
                    valid_primary: c.valid.primary,
                    hidden_primary: score,
                    gap: c.valid.primary - score,
                    quarantined: false,
                });
            }
        }

        append(RecordKind::RunFinished {
            reason: "completed".to_string(),
        });

        Run {
            dir,
            archive,
            base,
            random,
            children,
        }
    }

    fn choose_in(run: &Run, known: &HashSet<String>) -> Promotion {
        choose(&run.archive, COMMIT, known, &ModelAdapter::new()).expect("promotion")
    }

    fn ids(models: Vec<&PromotedModel>) -> Vec<String> {
        models.into_iter().map(|m| m.genome.id()).collect()
    }

    #[test]
    fn the_two_best_by_hidden_score_are_chosen() {
        // By fitness the order is child 3, 2, 1, 0; by hidden score 1, 2, 0, 3.
        let run = run("two-best", 0.65, [0.7, 0.9, 0.8, 0.6], &[]);
        let promotion = choose_in(&run, &model_seed_ids());
        let promoted: Vec<&PromotedModel> = promotion.promoted().collect();

        assert_eq!(
            ids(promoted.clone()),
            vec![run.children[1].id.clone(), run.children[2].id.clone()]
        );
        assert_eq!(promotion.considered.len(), PROMOTE_COUNT);
        assert_eq!(promotion.run_id, run.archive.run_id().expect("run id"));
        assert_eq!(promotion.task, TaskKind::Shapes);
        assert_eq!(promotion.commit, COMMIT);
        assert_eq!(
            promotion.best_base_model,
            BaseModelScore {
                id: run.base.id.clone(),
                generator: "base-model/resnet".to_string(),
                valid_primary: run.base.valid.primary,
                hidden_primary: 0.65,
            }
        );

        // The provenance names the run, the scores and the lineage back to generation 0.
        let first = promoted[0];
        let p = &first.provenance;

        assert_eq!(first.genome, run.children[1].genome);
        assert_eq!(p.run_id, promotion.run_id);
        assert_eq!(p.master_seed, 1);
        assert_eq!(p.valid_primary, run.children[1].valid.primary);
        assert_eq!(p.hidden_primary, 0.9);
        assert_eq!(
            p.lineage,
            vec![
                LineageStep {
                    id: run.children[1].id.clone(),
                    generation: 1,
                    generator: "random-mutator-v1".to_string(),
                    parents: vec![run.base.id.clone()],
                },
                LineageStep {
                    id: run.base.id.clone(),
                    generation: 0,
                    generator: "base-model/resnet".to_string(),
                    parents: Vec::new(),
                },
            ]
        );
        assert_eq!(p.roots(), vec!["base-model/resnet"]);
        assert_eq!(promoted[1].provenance.roots(), vec![INIT_GENERATOR]);
        assert!(path_in_scope(&format!(
            "{}{}.json",
            crate::hitl::MODEL_DIR,
            first.genome.id()
        )));

        // Each passes the gate of the pool.
        for m in &promoted {
            assert!(ModelAdapter::new().validate(m).is_ok());
        }

        // The manifest lists both and names no path.
        let manifest = promotion.manifest();
        let text = serde_json::to_string(&manifest).expect("json");

        assert_eq!(manifest.adapter, "model");
        assert_eq!(manifest.master_seed, 1);
        assert_eq!(manifest.candidates.len(), 2);
        assert_eq!(manifest.candidates[0].generator, "random-mutator-v1");
        assert_eq!(manifest.candidates[0].roots, vec!["base-model/resnet"]);
        assert!(manifest.not_promoted.is_empty());
        assert!(!text.contains('\\') && !text.contains(".json"), "{text}");
        assert!(!text.contains(&run.dir.display().to_string()));
    }

    #[test]
    fn a_model_that_does_not_beat_the_best_base_model_is_not_promoted() {
        let run_one = run("beat-one", 0.85, [0.7, 0.9, 0.8, 0.6], &[]);
        let promotion = choose_in(&run_one, &HashSet::new());

        assert_eq!(
            ids(promotion.promoted().collect()),
            vec![run_one.children[1].id.clone()]
        );

        let refused = promotion.considered[1]
            .outcome
            .as_ref()
            .expect_err("refused");

        assert!(refused.contains("does not beat"), "{refused}");
        assert_eq!(promotion.manifest().not_promoted.len(), 1);

        // A tie does not beat the base model, and the next model does not move up.
        let tie = run("beat-tie", 0.9, [0.7, 0.9, 0.8, 0.6], &[]);
        let promotion = choose_in(&tie, &HashSet::new());

        assert_eq!(promotion.promoted().count(), 0);
        assert_eq!(promotion.considered.len(), 2);
    }

    #[test]
    fn a_duplicate_adds_nothing() {
        let run = run("duplicate", 0.65, [0.7, 0.9, 0.8, 0.6], &[]);
        let pool: HashSet<String> = [run.children[1].id.clone()].into_iter().collect();
        let promotion = choose_in(&run, &pool);

        // The best is already in the pool; the second is promoted, and the third does not
        // take the free place, so a second promotion of one run adds nothing new.
        assert_eq!(
            ids(promotion.promoted().collect()),
            vec![run.children[2].id.clone()]
        );

        let reason = promotion.considered[0]
            .outcome
            .as_ref()
            .expect_err("refused");

        assert!(reason.contains("duplicate"), "{reason}");

        let both: HashSet<String> = [run.children[1].id.clone(), run.children[2].id.clone()]
            .into_iter()
            .collect();

        assert_eq!(choose_in(&run, &both).promoted().count(), 0);

        // The base models of the families are known too, as built and as 1-D twins.
        let seeds = model_seed_ids();
        let sequence = TaskKind::Motif1d.spec();

        assert_eq!(seeds.len(), 2 * 16 - 1, "the TCN is 1-D already");
        assert!(seeds.contains(&Family::DenseNet.base_model().id()));
        assert!(seeds.contains(&Family::DenseNet.base_model_for(&sequence).id()));
    }

    #[test]
    fn quarantined_models_and_base_models_are_not_promoted() {
        let mut run = run("quarantine", 0.95, [0.7, 0.9, 0.8, 0.6], &[]);
        let quarantined = run.children[3].clone();

        run.archive
            .append(
                RecordKind::HiddenEval {
                    candidate_id: quarantined.id.clone(),
                    valid_primary: quarantined.valid.primary,
                    hidden_primary: 0.99,
                    gap: -0.3,
                    quarantined: true,
                },
                0,
            )
            .expect("append");

        // The base model has the best hidden score, but it is the bar, not a winner; the
        // quarantined child is out of the ranking.
        let promotion = choose_in(&run, &HashSet::new());
        let considered: Vec<&str> = promotion.considered.iter().map(|c| c.id.as_str()).collect();

        assert_eq!(
            considered,
            vec![run.children[1].id.as_str(), run.children[2].id.as_str()]
        );
        assert_eq!(promotion.promoted().count(), 0);
        assert_eq!(run.random.generator, INIT_GENERATOR);
    }

    #[test]
    fn a_model_that_fails_the_current_limits_is_not_promoted() {
        let run = run("limits", 0.65, [0.7, 0.9, 0.8, 0.6], &[]);
        let tight = ModelAdapter {
            limits: Limits {
                max_epochs: 9,
                ..Limits::default()
            },
        };
        let promotion = choose(&run.archive, COMMIT, &HashSet::new(), &tight).expect("promotion");

        assert_eq!(promotion.promoted().count(), 0);

        for c in &promotion.considered {
            let reason = c.outcome.as_ref().expect_err("refused");

            assert!(reason.contains("fails the check of the pool"), "{reason}");
            assert!(reason.contains("epochs 10 outside 5..=9"), "{reason}");
        }
    }

    #[test]
    fn promotion_needs_a_commit_a_finished_hidden_evaluation_and_a_base_model() {
        let message = |run: &Run, commit: &str| {
            choose(&run.archive, commit, &HashSet::new(), &ModelAdapter::new())
                .expect_err("refused")
                .to_string()
        };
        let complete = run("needs", 0.65, [0.7, 0.9, 0.8, 0.6], &[]);

        for commit in ["", "main", "4613E2A", "4613e2", "4613e2a-dirty"] {
            assert!(
                message(&complete, commit).contains("lower-case hex digits"),
                "{commit}"
            );
        }

        // The base model (index 0) has no hidden score yet: the hidden set is incomplete.
        let open = run("needs-open", 0.65, [0.7, 0.9, 0.8, 0.6], &[0]);

        assert!(message(&open, COMMIT).contains("1 of the 5 candidates of the hidden set"));

        let none = run("needs-none", 0.65, [0.7, 0.9, 0.8, 0.6], &[0, 1, 2, 3, 4]);

        assert!(message(&none, COMMIT).contains("no hidden evaluation"));

        // A run with a single candidate, a random genome with a hidden score, under `config`.
        let lone = |name: &str, config: Config| {
            let dir = std::env::temp_dir()
                .join(format!("brut-promote-lone-{name}-{}", std::process::id()));

            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("temp dir");

            let mut archive = Archive::create(&dir.join("archive.jsonl")).expect("archive");
            let child = entry(mlp(7), 0, INIT_GENERATOR, &[], 0.6);

            for kind in [
                RecordKind::RunStarted {
                    config,
                    crate_version: "test".to_string(),
                },
                RecordKind::Candidate(child.clone()),
                RecordKind::HiddenEval {
                    candidate_id: child.id.clone(),
                    valid_primary: 0.61,
                    hidden_primary: 0.6,
                    gap: 0.01,
                    quarantined: false,
                },
            ] {
                archive.append(kind, 0).expect("append");
            }

            let text = choose(&archive, COMMIT, &HashSet::new(), &ModelAdapter::new())
                .expect_err("refused")
                .to_string();

            let _ = std::fs::remove_dir_all(&dir);

            text
        };

        // A run with the model kinds mlp and no promoted models keeps the hidden set of the
        // runs before CNNs, base genomes or not, so it has no base model to compare.
        for config in [
            Config::default(),
            Config {
                base_genomes: vec![mlp(8)],
                ..Config::default()
            },
        ] {
            let text = lone("mlp", config);

            assert!(text.contains("no promoted models"), "{text}");
        }

        // Without a base model there is nothing to beat. Here the run breeds CNNs, but its
        // one candidate is a random genome, so the run has none.
        let text = lone("cnn", Config::for_models(ModelKinds::Cnn));

        assert!(text.contains("no base model"), "{text}");
    }
}
