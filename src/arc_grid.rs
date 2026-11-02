//! ARC grid tasks for the CNN models (ticket 06).
//!
//! Each ARC-AGI-1 training task becomes a grid-to-grid task with one colour class per cell,
//! an ARC grid task (`ArcGridTask`). Its examples are the pairs of its task file and its
//! re-arc generations (Hodel 2024, arXiv:2404.07353, preprint), since a task file alone
//! holds only 3 to 9 pairs, 4 in most. A genome trains a fresh CNN on each task from its own
//! seed, and the CNN is scored on the task's held-out examples (`train_and_score`): the
//! share of whole output grids it gets right (exact match, `Metrics::primary`), and as a
//! descriptive measure the share of right cells (`Metrics::accuracy`).
//!
//! Data. `arc_data` reads both sources and checks every file against its SHA-256 in the
//! manifest: the training tasks (`arc_data::load_training_set`) and the re-arc examples that
//! `tools/rearc_to_jsonl.py` converted (`arc_data::load_rearc_grids`). Both evaluation sets
//! stay held out: this module reads no evaluation file, and its sources (`TRAINING_SOURCES`)
//! pass `arc_data::check_contamination` for every evaluation set. The two training tasks
//! that repeat pairs of evaluation tasks (`arc_data::A1_TRAIN_REPEATS_EVAL`) stay out with
//! their generations.
//!
//! Tasks (`tasks`). A training task becomes an ARC grid task when every pair of its task file
//! keeps its grid size and fits the canvas, and re-arc has a pair for it that fits too. A
//! generated pair equal to a pair of the task file or to an earlier generated pair is
//! dropped, so that every held-out example is new to the training. Then:
//!
//! - train: the train pairs of the task file, then the generated pairs but the last fifth;
//! - valid: the last fifth of the generated pairs, at least one (`VALID_SHARE`);
//! - test: the test pairs of the task file, the pairs that ARC poses.
//!
//! Encoding (`dataset`). Every grid lies at the top left of a square canvas, `ARC_SIDE` (30)
//! cells per side by default, the size of the largest ARC grid. The input has one channel per
//! colour, 1 where the cell has that colour, so a cell outside the grid has no channel on.
//! The targets are one colour class per cell, `NO_CELL` outside the output grid, where a
//! cell carries no loss and no score (`TaskSpec::cell_classes`). One canvas gives every task
//! the same shapes, so a genome that fits one task fits all. A canvas per task would save
//! little: the largest grid of a task has 29 or 30 cells per side in 255 of the 259 tasks
//! that the real data give, and 25 to 28 in the other 4 (2026-11-01).
//!
//! Training and scoring. A genome trains on each task for its own epochs, or for the most that
//! still pass the shape check on that task (`epochs_on`), so that the FLOP limit, not the
//! clock, ends its training. A genome that does not fit at `MIN_EPOCHS` is refused on that
//! task. Tasks train in parallel on the caller's rayon pool; each gives the same bits at any
//! thread count on one machine (ADR 0007). `copy_baseline` scores the copy of the input,
//! which already gets many cells right.
//!
//! Base models (`base_models`): a U-Net (Ronneberger, Fischer and Brox 2015) and a fully
//! convolutional network (Long, Shelhamer and Darrell 2015), both with the per-cell head and
//! sized for the default limits on the largest task.
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::arc_data::{self, A1_TRAIN_REPEATS_EVAL};
use crate::cnn::genome::{
    Block, CnnActivation, CnnGenome, Dims, Downsample, Head, Norm, Stage, Stem,
};
use crate::config::Limits;
use crate::genome::{Activation, Arch, Genome, MIN_EPOCHS, Optimizer};
use crate::learner::types::{ArcPair, ArcTask, EvalSet, SourceId};
use crate::score::{self, Metrics};
use crate::task::{
    Dataset, InputShape, NO_CELL, Objective, OutputShape, Targets, TaskKind, TaskSpec,
};
use crate::train::{self, TrainStats};
use crate::util::{BrutError, Result};

/// The sources of the ARC grid tasks: ARC-AGI-1 training and its re-arc generations.
pub const TRAINING_SOURCES: [SourceId; 2] = [SourceId::ArcAgi1Training, SourceId::ReArc];

/// The evaluation sets that the training sources must not touch.
const EVAL_SETS: [EvalSet; 3] = [EvalSet::A1Eval, EvalSet::A2Eval, EvalSet::ConceptArc];

/// Side of the default canvas: the largest ARC grid that `arc_data` reads.
pub const ARC_SIDE: usize = arc_data::MAX_SIDE;

/// Colours of an ARC grid: one input channel and one output class each.
pub const COLOURS: usize = 10;

/// One generated pair in this many validates: the last fifth, at least one.
pub const VALID_SHARE: usize = 5;

/// One ARC-AGI-1 training task with its re-arc generations, split for training.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArcGridTask {
    /// The id of the ARC-AGI-1 training task, such as "007bbfb7".
    pub id: String,
    /// The train pairs of the task file, then the generated pairs that train.
    pub train: Vec<ArcPair>,
    /// The last fifth of the generated pairs, held out.
    pub valid: Vec<ArcPair>,
    /// The test pairs of the task file, held out.
    pub test: Vec<ArcPair>,
}

impl ArcGridTask {
    /// The spec of this task on a canvas of `canvas` cells per side.
    pub fn spec(&self, canvas: usize) -> TaskSpec {
        spec(canvas, self.train.len(), self.valid.len(), self.test.len())
    }
}

/// A training task that did not become an ARC grid task, and why.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeftOut {
    pub id: String,
    pub reason: String,
}

/// The spec of an ARC grid task on a square canvas of `side` cells per side, with these
/// counts of train, valid and test examples (`n_hidden` holds the test count).
///
/// The input has `COLOURS` channels of `side` x `side` cells, the output a grid of the same
/// size with one channel per colour class (`TaskSpec::cell_classes`). `side` is at most
/// `ARC_SIDE`; `tasks` and `dataset` check it.
pub fn spec(side: usize, n_train: usize, n_valid: usize, n_test: usize) -> TaskSpec {
    let values = COLOURS.saturating_mul(side).saturating_mul(side);

    TaskSpec {
        kind: TaskKind::ArcGrid,
        n_in: values,
        input: InputShape::Image {
            channels: COLOURS,
            height: side,
            width: side,
        },
        n_out: values,
        output: OutputShape::Grid {
            channels: COLOURS,
            height: side,
            width: side,
        },
        objective: Objective::Classification { classes: COLOURS },
        n_train,
        n_valid,
        n_hidden: n_test,
        noise: 0.0,
    }
}

/// The largest ARC grid task (`TaskKind::ArcGrid.spec`): the default canvas, a task file of
/// 10 train and 4 test pairs, and 60 generated pairs, 48 to train and 12 to validate. A
/// genome that passes the shape check on it trains every task for its own epochs.
pub fn largest_spec() -> TaskSpec {
    let valid = arc_data::MAX_REARC_PAIRS / VALID_SHARE;

    spec(
        ARC_SIDE,
        arc_data::MAX_TRAIN_PAIRS + arc_data::MAX_REARC_PAIRS - valid,
        valid,
        arc_data::MAX_TEST_PAIRS,
    )
}

/// An error unless 1 <= `canvas` <= `ARC_SIDE`.
fn check_canvas(canvas: usize) -> Result<()> {
    if !(1..=ARC_SIDE).contains(&canvas) {
        return Err(BrutError::Invalid(format!(
            "arc grid: the canvas has {canvas} cells per side, not 1 to {ARC_SIDE}"
        )));
    }

    Ok(())
}

/// True when the input and the output of `pair` have the same size.
fn keeps_size(pair: &ArcPair) -> bool {
    pair.input.h == pair.output.h && pair.input.w == pair.output.w
}

/// True when both grids of `pair` fit a canvas of `canvas` cells per side.
fn fits(pair: &ArcPair, canvas: usize) -> bool {
    [&pair.input, &pair.output]
        .iter()
        .all(|grid| usize::from(grid.h) <= canvas && usize::from(grid.w) <= canvas)
}

/// The ARC grid tasks of these training tasks and generated pairs on a canvas of `canvas`
/// cells per side, sorted by id, and the training tasks left out, each with the reason (see
/// the module doc). Generated pairs of an id without a training task are left out too.
///
/// An error for a bad canvas, or an id that appears twice in either list.
pub fn tasks(
    training: &[ArcTask],
    generated: &[(String, Vec<ArcPair>)],
    canvas: usize,
) -> Result<(Vec<ArcGridTask>, Vec<LeftOut>)> {
    check_canvas(canvas)?;

    let mut by_id: BTreeMap<&str, &[ArcPair]> = BTreeMap::new();

    for (id, pairs) in generated {
        if by_id.insert(id.as_str(), pairs).is_some() {
            return Err(BrutError::Invalid(format!(
                "arc grid: re-arc task {id} appears twice"
            )));
        }
    }

    let mut sorted: Vec<&ArcTask> = training.iter().collect();

    sorted.sort_by(|a, b| a.id.cmp(&b.id));

    let mut seen = BTreeSet::new();
    let mut kept = Vec::new();
    let mut left_out = Vec::new();

    for task in sorted {
        if !seen.insert(task.id.as_str()) {
            return Err(BrutError::Invalid(format!(
                "arc grid: training task {} appears twice",
                task.id
            )));
        }

        let reason = |text: &str| LeftOut {
            id: task.id.clone(),
            reason: text.to_string(),
        };
        let own: Vec<&ArcPair> = task.train.iter().chain(&task.test).collect();

        if A1_TRAIN_REPEATS_EVAL.contains(&task.id.as_str()) {
            left_out.push(reason("repeats pairs of an ARC-AGI-1 evaluation task"));
            continue;
        }

        if !own.iter().all(|pair| keeps_size(pair)) {
            left_out.push(reason("a pair of the task file changes the grid size"));
            continue;
        }

        if !own.iter().all(|pair| fits(pair, canvas)) {
            left_out.push(reason("a pair of the task file is larger than the canvas"));
            continue;
        }

        let Some(pairs) = by_id.get(task.id.as_str()) else {
            left_out.push(reason("re-arc has no pair of this task"));
            continue;
        };

        let mut fresh: Vec<ArcPair> = Vec::new();

        for pair in pairs.iter() {
            let usable = keeps_size(pair) && fits(pair, canvas);

            if usable && !own.contains(&pair) && !fresh.contains(pair) {
                fresh.push(pair.clone());
            }
        }

        if fresh.is_empty() {
            left_out.push(reason("no re-arc pair of this task fits the canvas"));
            continue;
        }

        let n_valid = (fresh.len() / VALID_SHARE).max(1);
        let valid = fresh.split_off(fresh.len() - n_valid);
        let mut train = task.train.clone();

        train.extend(fresh);

        kept.push(ArcGridTask {
            id: task.id.clone(),
            train,
            valid,
            test: task.test.clone(),
        });
    }

    for id in by_id.keys().filter(|id| !seen.contains(*id)) {
        left_out.push(LeftOut {
            id: id.to_string(),
            reason: "re-arc pairs without an ARC-AGI-1 training task".to_string(),
        });
    }

    Ok((kept, left_out))
}

/// The ARC grid tasks of the data under `root`, whose manifest lies at `manifest`, on a
/// canvas of `canvas` cells per side, and the training tasks left out (`tasks`).
///
/// It reads only the ARC-AGI-1 training files and the re-arc file of the manifest, each
/// checked against its SHA-256, and no evaluation file.
pub fn load(
    root: &Path,
    manifest: &Path,
    canvas: usize,
) -> Result<(Vec<ArcGridTask>, Vec<LeftOut>)> {
    check_canvas(canvas)?;

    for set in EVAL_SETS {
        arc_data::check_contamination(&TRAINING_SOURCES, set)?;
    }

    let m = arc_data::load_manifest(manifest)?;
    let training = arc_data::load_training_set(root, &m)?;
    let generated = arc_data::load_rearc_grids(root, &m)?;

    tasks(&training, &generated, canvas)
}

/// The pairs as a dataset of an ARC grid task on a canvas of `canvas` cells per side (see
/// the module doc, "Encoding"). An error for a bad canvas, or a pair that changes its size or
/// does not fit.
pub fn dataset(pairs: &[ArcPair], canvas: usize) -> Result<Dataset> {
    check_canvas(canvas)?;

    let cells = canvas * canvas;
    let n_in = COLOURS * cells;
    let mut x = vec![0.0f32; pairs.len() * n_in];
    let mut labels = vec![NO_CELL; pairs.len() * cells];

    for (i, pair) in pairs.iter().enumerate() {
        if !keeps_size(pair) || !fits(pair, canvas) {
            return Err(BrutError::Invalid(format!(
                "arc grid: pair {i} changes its size or does not fit the canvas of {canvas}"
            )));
        }

        let (h, w) = (usize::from(pair.input.h), usize::from(pair.input.w));
        let sample = &mut x[i * n_in..(i + 1) * n_in];
        let targets = &mut labels[i * cells..(i + 1) * cells];

        for y in 0..h {
            for col in 0..w {
                let colour = usize::from(pair.input.cells[y * w + col]);
                let at = y * canvas + col;

                sample[colour * cells + at] = 1.0;
                targets[at] = u32::from(pair.output.cells[y * w + col]);
            }
        }
    }

    Ok(Dataset {
        x,
        targets: Targets::Labels(labels),
        n: pairs.len(),
        n_in,
        n_out: n_in,
    })
}

/// The epochs `genome` trains on a task of `spec`: its own when it passes `Genome::validate`
/// there, else the most from `MIN_EPOCHS` on that pass. The cost of training grows with the
/// epochs, so the most is found by bisection. An error with the reason of the shape check
/// when even `MIN_EPOCHS`, or the genome's own epochs below it, do not pass.
pub fn epochs_on(genome: &Genome, spec: &TaskSpec, limits: &Limits) -> Result<usize> {
    if genome.validate(limits, spec).is_ok() {
        return Ok(genome.epochs);
    }

    let with = |epochs| Genome {
        epochs,
        ..genome.clone()
    };

    if genome.epochs <= MIN_EPOCHS {
        genome.validate(limits, spec)?;
    }

    with(MIN_EPOCHS).validate(limits, spec)?;

    let (mut low, mut high) = (MIN_EPOCHS, genome.epochs - 1);

    while low < high {
        let mid = low + (high - low).div_ceil(2);

        if with(mid).validate(limits, spec).is_ok() {
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    Ok(low)
}

/// The scores of a model on one split of an ARC grid task.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GridScore {
    /// Share of examples whose whole output grid is right (exact match).
    pub exact: f64,
    /// Share of right cells inside the output grids (cell accuracy, descriptive).
    pub cells: f64,
    /// Mean cross-entropy over the cells inside the output grids; none for the copy.
    pub loss: Option<f64>,
}

impl GridScore {
    /// The scores in `metrics` of a task with one class per cell.
    fn of(metrics: &Metrics) -> GridScore {
        GridScore {
            exact: metrics.primary,
            cells: metrics.accuracy.unwrap_or(0.0),
            loss: Some(metrics.loss),
        }
    }
}

/// What one genome, or the copy of the input, did on one ARC grid task.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskResult {
    pub id: String,
    /// The epochs it trained for (`epochs_on`); none for the copy and when refused.
    pub epochs: Option<usize>,
    pub train: Option<TrainStats>,
    /// The scores on the held-out generated pairs.
    pub valid: Option<GridScore>,
    /// The scores on the test pairs of the task file.
    pub test: Option<GridScore>,
    /// Why the genome did not train on this task.
    pub refused: Option<String>,
}

impl TaskResult {
    fn refused(id: &str, reason: String) -> TaskResult {
        TaskResult {
            id: id.to_string(),
            epochs: None,
            train: None,
            valid: None,
            test: None,
            refused: Some(reason),
        }
    }
}

/// The means over the tasks of one genome, or of the copy of the input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Summary {
    pub tasks: usize,
    /// Tasks it was scored on; for a genome, the tasks it trained on.
    pub scored: usize,
    pub refused: usize,
    pub diverged: usize,
    pub stopped_by_budget: usize,
    /// Mean exact match over all tasks, on the valid and on the test pairs. A refused task
    /// counts 0, as a missing task does in ARC's official score.
    pub exact_valid: f64,
    pub exact_test: f64,
    /// Mean cell accuracy over the scored tasks, on the valid and on the test pairs.
    pub cells_valid: f64,
    pub cells_test: f64,
}

/// The results of one genome, or of the copy of the input, on every task.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenomeResult {
    /// "copy", the name of a base model, or the name the caller gave.
    pub name: String,
    /// The content id of the genome (`Genome::id`); none for the copy.
    pub id: Option<String>,
    pub genome: Option<Genome>,
    pub summary: Summary,
    pub tasks: Vec<TaskResult>,
}

/// The means of `results` (see `Summary`).
fn summarize(results: &[TaskResult]) -> Summary {
    let n = results.len();
    let scored: Vec<&TaskResult> = results.iter().filter(|r| r.refused.is_none()).collect();
    let mean = |values: Vec<f64>, count: usize| {
        if count == 0 {
            0.0
        } else {
            values.iter().sum::<f64>() / count as f64
        }
    };
    let field = |split: fn(&TaskResult) -> Option<GridScore>, part: fn(GridScore) -> f64| {
        scored
            .iter()
            .map(|r| split(r).map_or(0.0, part))
            .collect::<Vec<f64>>()
    };
    let stats = |flag: fn(&TrainStats) -> bool| {
        results
            .iter()
            .filter(|r| r.train.as_ref().is_some_and(flag))
            .count()
    };

    Summary {
        tasks: n,
        scored: scored.len(),
        refused: n - scored.len(),
        diverged: stats(|t| t.diverged),
        stopped_by_budget: stats(|t| t.stopped_by_budget),
        exact_valid: mean(field(|r| r.valid, |s| s.exact), n),
        exact_test: mean(field(|r| r.test, |s| s.exact), n),
        cells_valid: mean(field(|r| r.valid, |s| s.cells), scored.len()),
        cells_test: mean(field(|r| r.test, |s| s.cells), scored.len()),
    }
}

/// Train a fresh CNN of `genome` on one task and score it on the held-out pairs.
fn train_one(genome: &Genome, task: &ArcGridTask, canvas: usize, limits: &Limits) -> TaskResult {
    let spec = task.spec(canvas);
    let epochs = match epochs_on(genome, &spec, limits) {
        Ok(epochs) => epochs,
        Err(e) => return TaskResult::refused(&task.id, e.to_string()),
    };
    let genome = Genome {
        epochs,
        ..genome.clone()
    };
    let data = (
        dataset(&task.train, canvas),
        dataset(&task.valid, canvas),
        dataset(&task.test, canvas),
    );
    let (Ok(train_data), Ok(valid), Ok(test)) = data else {
        return TaskResult::refused(&task.id, "a pair does not fit the canvas".to_string());
    };

    match train::train(&genome, &spec, &train_data, limits) {
        Ok((model, stats)) => TaskResult {
            id: task.id.clone(),
            epochs: Some(epochs),
            train: Some(stats),
            valid: Some(GridScore::of(&score::evaluate(&model, &spec, &valid))),
            test: Some(GridScore::of(&score::evaluate(&model, &spec, &test))),
            refused: None,
        },
        Err(e) => TaskResult::refused(&task.id, e.to_string()),
    }
}

/// Train `genome` on every task, a fresh CNN per task from its own seed, and score each on
/// its held-out pairs. The tasks train in parallel on the caller's rayon pool; the results
/// keep the order of `tasks`.
pub fn train_and_score(
    name: &str,
    genome: &Genome,
    tasks: &[ArcGridTask],
    canvas: usize,
    limits: &Limits,
) -> GenomeResult {
    let results: Vec<TaskResult> = tasks
        .par_iter()
        .map(|task| train_one(genome, task, canvas, limits))
        .collect();

    GenomeResult {
        name: name.to_string(),
        id: Some(genome.id()),
        genome: Some(genome.clone()),
        summary: summarize(&results),
        tasks: results,
    }
}

/// The scores of copying the input on `pairs`: the exact match and the cell accuracy.
fn copy_score(pairs: &[ArcPair]) -> GridScore {
    let mut exact = 0usize;
    let mut right = 0usize;
    let mut cells = 0usize;

    for pair in pairs {
        let same = pair
            .input
            .cells
            .iter()
            .zip(&pair.output.cells)
            .filter(|(a, b)| a == b)
            .count();

        exact += usize::from(same == pair.output.cells.len());
        right += same;
        cells += pair.output.cells.len();
    }

    GridScore {
        exact: exact as f64 / pairs.len().max(1) as f64,
        cells: right as f64 / cells.max(1) as f64,
        loss: None,
    }
}

/// The copy of the input on every task: what a model scores that changes no cell. Many ARC
/// outputs keep most cells of their input, so the cell accuracy of the copy is high.
pub fn copy_baseline(tasks: &[ArcGridTask]) -> GenomeResult {
    let results: Vec<TaskResult> = tasks
        .iter()
        .map(|task| TaskResult {
            id: task.id.clone(),
            epochs: None,
            train: None,
            valid: Some(copy_score(&task.valid)),
            test: Some(copy_score(&task.test)),
            refused: None,
        })
        .collect();

    GenomeResult {
        name: "copy".to_string(),
        id: None,
        genome: None,
        summary: summarize(&results),
        tasks: results,
    }
}

/// What `brut arc-grid` writes: the tasks, the copy of the input and each genome.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    pub canvas: usize,
    /// The limits every training held to.
    pub limits: Limits,
    /// The ids of the ARC grid tasks scored, sorted.
    pub tasks: Vec<String>,
    /// The training tasks left out by the rules of `tasks`.
    pub left_out: Vec<LeftOut>,
    /// The copy of the input first, then each genome in the given order.
    pub results: Vec<GenomeResult>,
}

/// Train and score each named genome on `tasks` (`train_and_score`), next to the copy of
/// the input.
pub fn report(
    genomes: &[(String, Genome)],
    tasks: &[ArcGridTask],
    left_out: Vec<LeftOut>,
    canvas: usize,
    limits: &Limits,
) -> Report {
    let mut results = vec![copy_baseline(tasks)];

    for (name, genome) in genomes {
        results.push(train_and_score(name, genome, tasks, canvas, limits));
    }

    Report {
        canvas,
        limits: *limits,
        tasks: tasks.iter().map(|t| t.id.clone()).collect(),
        left_out,
        results,
    }
}

/// A unit of a base model: a 3 x 3 convolution with batch norm and ReLU.
fn unit(block: Block, width: usize, stride: usize, residual: bool) -> Stage {
    Stage {
        block,
        width,
        kernel: 3,
        stride,
        down: Downsample::Conv,
        dilation: 1,
        groups: 1,
        se: false,
        residual,
        norm: Norm::Batch,
        activation: CnnActivation::Relu,
    }
}

/// A base model: a stem of a `kernel` x `kernel` convolution to `width` channels with batch
/// norm and ReLU, these stages, the per-cell head, and Adam in batches of 16.
fn base_model(
    kernel: usize,
    width: usize,
    stages: Vec<Stage>,
    learning_rate: f32,
    epochs: usize,
    seed: u64,
) -> Genome {
    Genome {
        arch: Arch::Cnn(CnnGenome {
            dims: Dims::TwoD,
            stem: Stem {
                width,
                kernel,
                stride: 1,
                down: Downsample::Conv,
                norm: Norm::Batch,
                activation: CnnActivation::Relu,
            },
            stages,
            head: Head::PerCell,
        }),
        hidden: Vec::new(),
        activation: Activation::Relu,
        optimizer: Optimizer::Adam,
        learning_rate,
        l2: 0.0,
        batch_size: 16,
        epochs,
        init_scale: 1.0,
        seed,
    }
}

/// The base models of the ARC grid tasks, with their names. Each trains for the most epochs
/// that pass the shape check on the largest task (`largest_spec`) under the default limits,
/// so it trains every task for its own epochs. Neither is tuned beyond a few settings, which
/// the Comments of ticket 06 list with their scores.
///
/// - "u-net" (Ronneberger, Fischer and Brox 2015): a 3 x 3 stem of 8 channels, an encoder
///   that moves to half the size by a strided convolution to 16 channels, two residual basic
///   blocks there, a decoder that upsamples by a transposed convolution and joins the map of
///   the stem, and the per-cell head; a learning rate of 0.03 and 8 epochs. The U-Net base
///   model of the CNN families (`Family::UNet`) reads one channel; its stem of 4 channels
///   would squeeze 10 colours.
/// - "fcn" (Long, Shelhamer and Darrell 2015): a 1 x 1 stem of 16 channels, which keeps the
///   colour of each cell apart, two 3 x 3 convolutions of 16 channels at full size, and the
///   per-cell head; no layer is dense. A learning rate of 0.01 and 9 epochs.
pub fn base_models() -> Vec<(&'static str, Genome)> {
    vec![
        (
            "u-net",
            base_model(
                3,
                8,
                vec![
                    unit(Block::Plain, 16, 2, false),
                    unit(Block::Basic, 16, 1, true),
                    unit(Block::Basic, 16, 1, true),
                    unit(Block::Decoder, 8, 2, false),
                ],
                0.03,
                8,
                17,
            ),
        ),
        (
            "fcn",
            base_model(
                1,
                16,
                vec![
                    unit(Block::Plain, 16, 1, false),
                    unit(Block::Plain, 16, 1, false),
                ],
                0.01,
                9,
                18,
            ),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::types::Grid;

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn grid(rows: &[&[u8]]) -> Grid {
        Grid {
            h: rows.len() as u8,
            w: rows[0].len() as u8,
            cells: rows.iter().flat_map(|r| r.iter().copied()).collect(),
        }
    }

    fn pair(input: &[&[u8]], output: &[&[u8]]) -> ArcPair {
        ArcPair {
            input: grid(input),
            output: grid(output),
        }
    }

    /// A random grid of `h` x `w` cells with colours 0 to 3, and its colours 1 and 2 swapped:
    /// a rule a CNN learns cell by cell.
    fn swap_pair(rng: &mut ChaCha8Rng, h: u8, w: u8) -> ArcPair {
        let cells: Vec<u8> = (0..h as usize * w as usize)
            .map(|_| rng.random_range(0..4u8))
            .collect();
        let swapped = cells
            .iter()
            .map(|&c| match c {
                1 => 2,
                2 => 1,
                other => other,
            })
            .collect();

        ArcPair {
            input: Grid { h, w, cells },
            output: Grid {
                h,
                w,
                cells: swapped,
            },
        }
    }

    fn swap_task(id: &str, seed: u64, train: usize, test: usize) -> ArcTask {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        ArcTask {
            id: id.to_string(),
            train: (0..train).map(|_| swap_pair(&mut rng, 3, 4)).collect(),
            test: (0..test).map(|_| swap_pair(&mut rng, 4, 3)).collect(),
        }
    }

    fn swap_generated(seed: u64, n: usize, side: u8) -> Vec<ArcPair> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        (0..n)
            .map(|i| {
                let h = 2 + (i as u8 % (side - 1));
                let w = side + 1 - h;

                swap_pair(&mut rng, h, w)
            })
            .collect()
    }

    #[test]
    fn tasks_split_their_pairs_and_leave_out_what_breaks_a_rule() {
        let keep = swap_task("aaaaaaaa", 1, 3, 1);
        let mut generated = swap_generated(2, 12, 4);

        // Two copies that must go: a pair of the task file, and a repeated generated pair.
        generated.insert(3, keep.train[1].clone());
        generated.insert(7, generated[0].clone());

        let resize = ArcTask {
            id: "bbbbbbbb".into(),
            train: vec![pair(&[&[1]], &[&[1, 1]])],
            test: vec![pair(&[&[2]], &[&[2]])],
        };
        let big = ArcTask {
            id: "cccccccc".into(),
            train: vec![pair(&[&[1; 7]], &[&[2; 7]])],
            test: vec![pair(&[&[2]], &[&[1]])],
        };
        let bare = swap_task("dddddddd", 3, 2, 1);
        let repeats = swap_task(A1_TRAIN_REPEATS_EVAL[0], 4, 2, 1);
        let only_big = swap_task("eeeeeeee", 5, 2, 1);
        let training = [
            only_big.clone(),
            repeats.clone(),
            bare,
            big,
            resize,
            keep.clone(),
        ];
        let generated = vec![
            ("aaaaaaaa".to_string(), generated),
            ("bbbbbbbb".to_string(), swap_generated(6, 3, 3)),
            ("cccccccc".to_string(), swap_generated(7, 3, 3)),
            (repeats.id.clone(), swap_generated(8, 3, 3)),
            (
                "eeeeeeee".to_string(),
                vec![
                    pair(&[&[1; 7]], &[&[2; 7]]),
                    pair(&[&[3][..]; 7], &[&[3][..]; 7]),
                ],
            ),
            ("ffffffff".to_string(), swap_generated(10, 3, 3)),
        ];
        let (kept, left_out) = tasks(&training, &generated, 6).expect("tasks");

        assert_eq!(kept.len(), 1);

        let task = &kept[0];
        let fresh = swap_generated(2, 12, 4);

        // 12 fresh generated pairs: the first 10 train after the 3 of the file, 2 validate.
        assert_eq!(task.id, "aaaaaaaa");
        assert_eq!(task.train[..3], keep.train[..]);
        assert_eq!(task.train[3..], fresh[..10]);
        assert_eq!(task.valid, fresh[10..]);
        assert_eq!(task.test, keep.test);
        assert_eq!(task.spec(6), spec(6, 13, 2, 1));

        let reasons: Vec<(&str, &str)> = left_out
            .iter()
            .map(|l| (l.id.as_str(), l.reason.as_str()))
            .collect();

        assert_eq!(
            reasons,
            vec![
                (
                    A1_TRAIN_REPEATS_EVAL[0],
                    "repeats pairs of an ARC-AGI-1 evaluation task"
                ),
                ("bbbbbbbb", "a pair of the task file changes the grid size"),
                (
                    "cccccccc",
                    "a pair of the task file is larger than the canvas"
                ),
                ("dddddddd", "re-arc has no pair of this task"),
                ("eeeeeeee", "no re-arc pair of this task fits the canvas"),
                (
                    "ffffffff",
                    "re-arc pairs without an ARC-AGI-1 training task"
                ),
            ]
        );

        // The same canvas of 30 keeps "eeeeeeee" and "cccccccc", whose grids fit there.
        let (kept, _) = tasks(&training, &generated, ARC_SIDE).expect("tasks");
        let ids: Vec<&str> = kept.iter().map(|t| t.id.as_str()).collect();

        assert_eq!(ids, vec!["aaaaaaaa", "cccccccc", "eeeeeeee"]);

        // A bad canvas, and an id twice, are errors.
        assert!(tasks(&training, &generated, 0).is_err());
        assert!(tasks(&training, &generated, ARC_SIDE + 1).is_err());

        let twice = [keep.clone(), keep];

        assert!(tasks(&twice, &generated, 6).is_err());
    }

    #[test]
    fn a_dataset_puts_each_grid_at_the_top_left_of_the_canvas() {
        let p = pair(&[&[3, 0, 9], &[1, 1, 2]], &[&[4, 4, 4], &[0, 9, 2]]);
        let data = dataset(std::slice::from_ref(&p), 4).expect("dataset");
        let cells = 16;

        assert_eq!((data.n, data.n_in, data.n_out), (1, 10 * cells, 10 * cells));

        // One channel on at each cell inside the input grid, none outside.
        for y in 0..4 {
            for col in 0..4 {
                let on: Vec<usize> = (0..COLOURS)
                    .filter(|&c| data.x[c * cells + y * 4 + col] == 1.0)
                    .collect();
                let expected = if y < 2 && col < 3 {
                    vec![usize::from(p.input.cells[y * 3 + col])]
                } else {
                    Vec::new()
                };

                assert_eq!(on, expected, "({y}, {col})");
            }
        }

        assert_eq!(data.x.iter().filter(|&&v| v != 0.0).count(), 6);

        let Targets::Labels(labels) = &data.targets else {
            panic!("labels");
        };
        let n = NO_CELL;

        assert_eq!(
            labels,
            &vec![4, 4, 4, n, 0, 9, 2, n, n, n, n, n, n, n, n, n]
        );

        // A pair that changes its size, or does not fit, is refused.
        assert!(dataset(&[pair(&[&[1]], &[&[1, 1]])], 4).is_err());
        assert!(dataset(&[pair(&[&[1; 5]], &[&[1; 5]])], 4).is_err());
        assert!(dataset(&[p], 0).is_err());
    }

    #[test]
    fn the_largest_task_bounds_every_task() {
        let largest = largest_spec();

        assert_eq!(largest, TaskKind::ArcGrid.spec());
        assert_eq!(largest, spec(ARC_SIDE, 58, 12, 4));
        assert_eq!(largest.cell_classes(), Some((10, 900)));
        assert!(largest.input_fits() && largest.output_fits());

        // The base models fit it at their own epochs, so every task, and they are distinct.
        let limits = Limits::default();

        for (name, genome) in base_models() {
            genome
                .validate(&limits, &largest)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                epochs_on(&genome, &largest, &limits).expect("fits"),
                genome.epochs
            );
        }

        let ids: BTreeSet<String> = base_models().iter().map(|(_, g)| g.id()).collect();

        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn epochs_shrink_only_where_a_task_is_too_large() {
        let limits = Limits::default();
        let (_, fcn) = base_models().remove(1);
        let long = Genome {
            epochs: 400,
            ..fcn.clone()
        };
        let small = spec(6, 13, 2, 1);
        let largest = largest_spec();
        let most = epochs_on(&long, &largest, &limits).expect("fits at fewer epochs");

        // On a small task the genome keeps its epochs; on the largest it gets the most that
        // pass, and one more would not.
        assert_eq!(epochs_on(&long, &small, &limits).expect("fits"), 400);
        assert!((MIN_EPOCHS..400).contains(&most), "{most}");
        assert!(
            Genome {
                epochs: most,
                ..fcn.clone()
            }
            .validate(&limits, &largest)
            .is_ok()
        );
        assert!(
            Genome {
                epochs: most + 1,
                ..fcn.clone()
            }
            .validate(&limits, &largest)
            .is_err()
        );

        // A genome that does not fit at MIN_EPOCHS, or has too few epochs, is refused.
        let tight = Limits {
            max_train_flops: 1_000_000,
            ..limits
        };

        assert!(epochs_on(&fcn, &largest, &tight).is_err());
        assert!(epochs_on(&Genome { epochs: 2, ..fcn }, &small, &limits).is_err());
    }

    #[test]
    fn the_copy_scores_the_cells_that_keep_their_colour() {
        let task = ArcGridTask {
            id: "aaaaaaaa".into(),
            train: Vec::new(),
            valid: vec![
                pair(&[&[1, 2]], &[&[1, 2]]),
                pair(&[&[1, 2], &[3, 3]], &[&[1, 0], &[3, 0]]),
            ],
            test: vec![pair(&[&[5]], &[&[6]])],
        };
        let copy = copy_baseline(std::slice::from_ref(&task));
        let result = &copy.tasks[0];

        assert_eq!(copy.name, "copy");
        assert_eq!(result.valid.expect("valid").exact, 0.5);
        assert_eq!(result.valid.expect("valid").cells, 4.0 / 6.0);
        assert_eq!(result.test.expect("test").exact, 0.0);
        assert_eq!(copy.summary.exact_valid, 0.5);
        assert_eq!(copy.summary.cells_test, 0.0);
        assert_eq!((copy.summary.tasks, copy.summary.scored), (1, 1));
    }

    /// A small fully convolutional network for a canvas of 6 x 6: a stem and one 3 x 3
    /// convolution of 8 channels. A colour swap needs only each cell's own colour.
    fn small_fcn() -> Genome {
        base_model(3, 8, vec![unit(Block::Plain, 8, 1, false)], 0.03, 30, 5)
    }

    /// Two swap tasks and one whose outputs are random: no rule to learn.
    fn small_tasks() -> Vec<ArcGridTask> {
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let noise: Vec<ArcPair> = (0..20)
            .map(|_| {
                let mut p = swap_pair(&mut rng, 4, 4);

                p.output.cells = (0..16).map(|_| rng.random_range(0..4u8)).collect();
                p
            })
            .collect();
        let training = [
            swap_task("aaaaaaaa", 11, 3, 1),
            swap_task("bbbbbbbb", 12, 3, 1),
        ];
        let generated = vec![
            ("aaaaaaaa".to_string(), swap_generated(13, 40, 6)),
            ("bbbbbbbb".to_string(), swap_generated(14, 40, 6)),
        ];
        let (mut kept, left_out) = tasks(&training, &generated, 6).expect("tasks");

        assert!(left_out.is_empty());

        kept.push(ArcGridTask {
            id: "cccccccc".into(),
            train: noise[..16].to_vec(),
            valid: noise[16..].to_vec(),
            test: noise[16..17].to_vec(),
        });

        kept
    }

    #[test]
    fn a_cnn_learns_a_colour_swap_and_is_scored_on_held_out_pairs() {
        let tasks = small_tasks();
        let result = train_and_score("small", &small_fcn(), &tasks, 6, &Limits::default());
        let copy = copy_baseline(&tasks);

        eprintln!("copy: {:?}", copy.summary);

        for r in &result.tasks {
            eprintln!(
                "{}: epochs {:?}, train loss {:?}, valid {:?}, test {:?}",
                r.id,
                r.epochs,
                r.train.as_ref().map(|t| t.final_train_loss),
                r.valid,
                r.test
            );
        }

        assert_eq!(result.summary.tasks, 3);
        assert_eq!(result.summary.scored, 3);
        assert_eq!(
            result.summary.diverged + result.summary.stopped_by_budget,
            0
        );

        // The swap tasks: here 8 and 7 of 8 held-out grids right and the test grid right,
        // where the copy of the input gets none right (and 0.40 of the cells). The bounds leave
        // room for the other bits of another CPU. The random outputs: no grid right, and about
        // a quarter of the cells, as chance gives with 4 colours.
        for (r, c) in result.tasks[..2].iter().zip(&copy.tasks) {
            let valid = r.valid.expect("valid");

            assert!(valid.exact >= 0.75, "{}: {valid:?}", r.id);
            assert!(valid.cells >= 0.95, "{}: {valid:?}", r.id);
            assert!(
                valid.exact >= c.valid.expect("copy").exact + 0.5,
                "{}",
                r.id
            );
        }

        assert!(
            result.summary.exact_test >= 2.0 / 3.0,
            "{:?}",
            result.summary
        );

        let noise = result.tasks[2].valid.expect("valid");

        assert_eq!(noise.exact, 0.0);
        assert!(noise.cells < 0.6, "{}", noise.cells);
    }

    #[test]
    fn training_and_scoring_repeat_their_bits_in_pools_of_1_and_2_threads() {
        let tasks = small_tasks();
        let genome = Genome {
            epochs: 6,
            ..small_fcn()
        };
        let run = |threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool");
            let mut result =
                pool.install(|| train_and_score("small", &genome, &tasks, 6, &Limits::default()));

            for task in &mut result.tasks {
                if let Some(train) = &mut task.train {
                    train.wall_seconds = 0.0;
                }
            }

            serde_json::to_string(&result).expect("json")
        };
        let first = run(1);

        assert_eq!(first, run(1));
        assert_eq!(first, run(2));
    }

    #[test]
    fn a_genome_that_does_not_fit_is_refused_per_task() {
        let tasks = small_tasks();
        let mlp = Genome {
            arch: Arch::Mlp,
            ..small_fcn()
        };
        let result = train_and_score("mlp", &mlp, &tasks, 6, &Limits::default());

        assert_eq!(result.summary.refused, 3);
        assert_eq!(result.summary.exact_valid, 0.0);
        assert!(
            result.tasks[0]
                .refused
                .as_deref()
                .is_some_and(|r| r.contains("one class per cell")),
            "{:?}",
            result.tasks[0].refused
        );
    }
}
