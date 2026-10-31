//! Training of a CNN through candle: the learner that the training loop of `crate::train`
//! drives.
//!
//! The loop is the MLP's: the same shuffling, learning-rate decay, budgets, kill switch and
//! divergence rules. Only the forward and backward pass differ. A step builds the batch
//! as a tensor, runs the network in training mode, takes the loss (mean softmax
//! cross-entropy, or the mean squared error for regression) and its gradients from candle,
//! and then updates the values with the MLP's optimizer (`nn::OptimizerState`).
//!
//! Reproducibility. Training is deterministic: the same genome and data give the same bits
//! at any thread count of the same machine, also inside rayon pools of other sizes
//! (ticket 01 checked 19 thread settings). Another CPU can give other bits: gemm, the
//! matrix library under candle, picks its kernel (with or without fused multiply-add) and
//! its cache blocking at run time from the CPU. Tests therefore compare results within one
//! machine only.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::cnn::genome::CnnGenome;
use crate::cnn::model::CnnModel;
use crate::cnn::net::{self, Net, ParamSpec, Values};
use crate::cnn::ops::{self, Grads, Param, Tensor};
use crate::genome::{Activation, Genome};
use crate::nn::OptimizerState;
use crate::task::{Dataset, Objective, Targets, TaskSpec};
use crate::train::Learner;
use crate::util::{BrutError, Result, derive_seed};

/// A CNN under training: its network, its values and the MLP's optimizer.
pub(crate) struct CnnLearner<'a> {
    arch: CnnGenome,
    hidden: Vec<usize>,
    activation: Activation,
    specs: Vec<ParamSpec>,
    net: Net,
    /// The values the optimizer updates; the network gets a copy after each step.
    values: Values,
    opt: OptimizerState,
    /// The gradients of the last batch, in the shapes of `values`.
    grad_weights: Vec<Vec<f32>>,
    grad_biases: Vec<Vec<f32>>,
    flops_per_sample: u64,
    data: &'a Dataset,
    objective: Objective,
}

impl<'a> CnnLearner<'a> {
    /// The network of `cnn` (the `arch` of `genome`) on `spec`, with its initial values
    /// drawn from the stream `derive_seed(genome.seed, "init")`, as for an MLP.
    pub(crate) fn new(
        genome: &Genome,
        cnn: &CnnGenome,
        spec: &TaskSpec,
        data: &'a Dataset,
    ) -> Result<CnnLearner<'a>> {
        let plan = cnn.plan(&genome.hidden, genome.activation, spec)?;
        let specs = net::param_specs(&plan, genome.activation);
        let channels = net::batch_norm_channels(&plan);
        let mut init_rng = ChaCha8Rng::seed_from_u64(derive_seed(genome.seed, "init"));
        let values = Values::initial(&specs, &channels, genome.init_scale, &mut init_rng);
        let net = Net::new(&plan, &specs, &values)?;
        let opt = OptimizerState::for_lists(genome.optimizer, &values.weights, &values.biases);
        let zeros = |lists: &[Vec<f32>]| -> Vec<Vec<f32>> {
            lists.iter().map(|list| vec![0.0; list.len()]).collect()
        };

        Ok(CnnLearner {
            arch: cnn.clone(),
            hidden: genome.hidden.clone(),
            activation: genome.activation,
            grad_weights: zeros(&values.weights),
            grad_biases: zeros(&values.biases),
            specs,
            net,
            values,
            opt,
            flops_per_sample: plan.train_flops_per_sample(),
            data,
            objective: spec.objective,
        })
    }

    /// The trained model, with the running statistics of the last training step.
    pub(crate) fn into_model(mut self) -> Result<CnnModel> {
        self.values.running = self.net.running()?;

        Ok(CnnModel::from_values(
            self.arch,
            self.hidden,
            self.activation,
            &self.specs,
            &self.values,
        ))
    }

    /// The rows `rows` of the train data as one tensor: rows, then the axes of a sample.
    fn inputs(&self, rows: &[usize]) -> Result<Tensor> {
        let n_in = self.data.n_in;
        let mut values = Vec::with_capacity(rows.len() * n_in);

        for &i in rows {
            values.extend_from_slice(&self.data.x[i * n_in..(i + 1) * n_in]);
        }

        let mut shape = vec![rows.len()];

        shape.extend_from_slice(self.net.sample());

        Tensor::from_f32(values, &shape)
    }

    /// The mean loss of the outputs `out` of the rows `rows` against their targets.
    fn loss(&self, out: &Tensor, rows: &[usize]) -> Result<Tensor> {
        match (&self.data.targets, self.objective) {
            (Targets::Labels(labels), Objective::Classification { .. }) => {
                let batch: Vec<u32> = rows.iter().map(|&i| labels[i]).collect();

                ops::cross_entropy(out, &batch)
            }
            (Targets::Values(values), Objective::Regression) => {
                let n_out = self.data.n_out;
                let mut batch = Vec::with_capacity(rows.len() * n_out);

                for &i in rows {
                    batch.extend_from_slice(&values[i * n_out..(i + 1) * n_out]);
                }

                ops::mse(out, &Tensor::from_f32(batch, &[rows.len(), n_out])?)
            }
            _ => Err(BrutError::Invalid(
                "train data: targets do not match the objective".into(),
            )),
        }
    }
}

/// Copy the gradient of each of `params` into `into`; 0 where the loss does not use one.
fn gather(grads: &Grads, params: &[Param], into: &mut [Vec<f32>]) -> Result<()> {
    for (param, grad) in params.iter().zip(into.iter_mut()) {
        match grads.get(param) {
            Some(tensor) => {
                let values = tensor.to_vec_f32()?;

                if values.len() != grad.len() {
                    return Err(BrutError::Invalid(format!(
                        "cnn: a gradient has {} values for a tensor of {}",
                        values.len(),
                        grad.len()
                    )));
                }

                *grad = values;
            }
            None => grad.fill(0.0),
        }
    }

    Ok(())
}

impl Learner for CnnLearner<'_> {
    fn flops_per_sample(&self) -> u64 {
        self.flops_per_sample
    }

    fn loss_and_grad(&mut self, rows: &[usize]) -> Result<f32> {
        let x = self.inputs(rows)?;
        let out = self.net.forward_train(&x)?;
        let loss = self.loss(&out, rows)?;
        let value = loss.to_vec_f32()?.first().copied().unwrap_or(f32::NAN);

        // A diverged step ends the training; its gradients are not needed.
        if !value.is_finite() {
            return Ok(value);
        }

        let grads = ops::backward(&loss)?;

        gather(&grads, self.net.weights(), &mut self.grad_weights)?;
        gather(&grads, self.net.biases(), &mut self.grad_biases)?;

        Ok(value)
    }

    fn update(&mut self, lr: f32, l2: f32) -> Result<()> {
        self.opt.step_lists(
            &mut self.values.weights,
            &mut self.values.biases,
            &self.grad_weights,
            &self.grad_biases,
            lr,
            l2,
        );

        self.net.set(&self.values.weights, &self.values.biases)
    }

    fn is_finite(&self) -> Result<bool> {
        let running = self.net.running()?;
        let stats = running
            .iter()
            .all(|(mean, var)| mean.iter().chain(var).all(|v| v.is_finite()));

        Ok(self.values.is_finite() && stats)
    }
}

/// Helpers for the tests of the crate.
#[cfg(test)]
pub(crate) mod testing {
    use crate::task::{Dataset, Targets};

    /// The first `rows` samples of `data`.
    pub(crate) fn head(data: &Dataset, rows: usize) -> Dataset {
        let rows = rows.min(data.n);
        let targets = match &data.targets {
            Targets::Labels(labels) => Targets::Labels(labels[..rows].to_vec()),
            Targets::Values(values) => Targets::Values(values[..rows * data.n_out].to_vec()),
        };

        Dataset {
            x: data.x[..rows * data.n_in].to_vec(),
            targets,
            n: rows,
            n_in: data.n_in,
            n_out: data.n_out,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::testing::head;
    use super::*;
    use crate::cnn::genome::testing as genomes;
    use crate::cnn::genome::{CnnActivation, CnnGenome, Dims, Downsample, Head, Norm, Stage, Stem};
    use crate::config::Limits;
    use crate::genome::Optimizer;
    use crate::model::Model;
    use crate::score;
    use crate::task::{Split, TaskKind, generate};
    use crate::train::{TrainStats, train, train_logged, train_with_stop};

    /// A plain 2-D CNN for a 1 x 16 x 16 image: two 3x3 convolutions of 8 and 16 channels,
    /// each followed by ReLU and 2x2 max pooling, then flatten and dense layers of 32 and
    /// the classes. 9604 parameters for 4 classes. Adam, learning rate 0.003, batch 64,
    /// 8 epochs.
    fn plain_2d() -> Genome {
        let cnn = CnnGenome {
            dims: Dims::TwoD,
            stem: Stem {
                width: 8,
                kernel: 3,
                stride: 2,
                down: Downsample::MaxPool,
                norm: Norm::None,
                activation: CnnActivation::Relu,
            },
            stages: vec![Stage {
                stride: 2,
                down: Downsample::MaxPool,
                ..genomes::plain(16, 3, CnnActivation::Relu)
            }],
            head: Head::Flatten,
        };

        Genome {
            learning_rate: 0.003,
            batch_size: 64,
            epochs: 8,
            ..genomes::genome(cnn, vec![32], Activation::Relu)
        }
    }

    /// A plain 1-D CNN for a 4 x 64 one-hot sequence: two convolutions of 5 with 16
    /// channels and ReLU, then global average pooling and dense layers of 32 and the
    /// classes. 2242 parameters. Training genes as in `plain_2d`.
    fn plain_1d() -> Genome {
        let cnn = CnnGenome {
            dims: Dims::OneD,
            stem: Stem {
                width: 16,
                kernel: 5,
                stride: 1,
                down: Downsample::Conv,
                norm: Norm::None,
                activation: CnnActivation::Relu,
            },
            stages: vec![genomes::plain(16, 5, CnnActivation::Relu)],
            head: Head::GlobalAvgPool,
        };

        Genome {
            learning_rate: 0.003,
            batch_size: 64,
            epochs: 8,
            ..genomes::genome(cnn, vec![32], Activation::Relu)
        }
    }

    /// A 1-D CNN with batch norm, average pooling, group norm and a strided convolution,
    /// for a 4 x 64 sequence.
    fn normed_1d() -> Genome {
        let cnn = CnnGenome {
            dims: Dims::OneD,
            stem: Stem {
                width: 8,
                kernel: 3,
                stride: 2,
                down: Downsample::AvgPool,
                norm: Norm::Batch,
                activation: CnnActivation::Silu,
            },
            stages: vec![
                Stage {
                    stride: 2,
                    down: Downsample::MaxPool,
                    dilation: 2,
                    norm: Norm::Group,
                    ..genomes::plain(8, 3, CnnActivation::Gelu)
                },
                Stage {
                    stride: 3,
                    down: Downsample::Conv,
                    groups: 2,
                    norm: Norm::Layer,
                    ..genomes::plain(12, 5, CnnActivation::Tanh)
                },
            ],
            head: Head::Flatten,
        };

        Genome {
            epochs: 5,
            ..genomes::genome(cnn, vec![8], Activation::Sine)
        }
    }

    /// Train `genome` on the Train split of `task` (data seed 1), and return its accuracy
    /// on the Valid split.
    fn learns(task: TaskKind, genome: &Genome) -> f64 {
        let spec = task.spec();
        let data = generate(&spec, Split::Train, 1);
        let valid = generate(&spec, Split::Valid, 1);
        let started = Instant::now();
        let (model, stats) = train(genome, &spec, &data, &Limits::default()).expect("train");
        let accuracy = score::evaluate(&model, &spec, &valid).primary;

        eprintln!(
            "{}: {} parameters, {} epochs in {:.1} s, train loss {:.3}, valid accuracy {accuracy:.3}",
            task.name(),
            model.param_count(),
            stats.epochs_done,
            started.elapsed().as_secs_f64(),
            stats.final_train_loss
        );

        assert_eq!(stats.epochs_done, genome.epochs, "{task:?}");
        assert!(!stats.diverged && !stats.stopped_by_budget, "{task:?}");

        accuracy
    }

    // Chance is 1 / classes, since every split holds each class equally often. The bounds
    // lie 0.10 to 0.23 below what the genomes reached on the machine that wrote these tests
    // (shapes 0.678, bars and stripes 1.000, blob count 0.677, mirror symmetry 0.771, motif
    // 0.803), as room for the other bits of another CPU.

    #[test]
    fn a_plain_cnn_learns_shapes() {
        assert!(learns(TaskKind::Shapes, &plain_2d()) > 0.5, "chance 0.25");
    }

    #[test]
    fn a_plain_cnn_learns_bars_and_stripes() {
        assert!(
            learns(TaskKind::BarsAndStripes, &plain_2d()) > 0.9,
            "chance 0.5"
        );
    }

    #[test]
    fn a_plain_cnn_learns_blob_count() {
        assert!(
            learns(TaskKind::BlobCount, &plain_2d()) > 0.45,
            "chance 0.25"
        );
    }

    #[test]
    fn a_plain_cnn_learns_mirror_symmetry() {
        assert!(
            learns(TaskKind::MirrorSymmetry, &plain_2d()) > 0.65,
            "chance 0.5"
        );
    }

    #[test]
    fn a_plain_cnn_learns_the_motif() {
        assert!(learns(TaskKind::Motif1d, &plain_1d()) > 0.65, "chance 0.5");
    }

    /// `stats` without the wall-clock time, the one field that differs between runs.
    fn timeless(stats: &TrainStats) -> TrainStats {
        TrainStats {
            wall_seconds: 0.0,
            ..stats.clone()
        }
    }

    /// The bits of every step's loss, and the model file.
    fn fingerprint(task: TaskKind, genome: &Genome) -> (Vec<u32>, String, TrainStats) {
        let spec = task.spec();
        let data = head(&generate(&spec, Split::Train, 1), 160);
        let (model, stats, losses) =
            train_logged(genome, &spec, &data, &Limits::default()).expect("train");
        let file = serde_json::to_string(&model).expect("json");

        (
            losses.iter().map(|l| l.to_bits()).collect(),
            file,
            timeless(&stats),
        )
    }

    #[test]
    fn training_is_byte_equal_twice_and_in_pools_of_1_and_4_threads() {
        // VGG-like with batch norm and max pooling, and a 1-D CNN with every norm and pool.
        for (task, genome) in [
            (TaskKind::Shapes, genomes::vgg()),
            (TaskKind::Motif1d, normed_1d()),
        ] {
            let first = fingerprint(task, &genome);
            let second = fingerprint(task, &genome);
            let pool = |threads: usize| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("pool")
                    .install(|| fingerprint(task, &genome))
            };

            assert_eq!(first.0.len() as u64, first.2.steps, "{task:?}");
            assert_eq!(first, second, "{task:?}: two runs in a row");
            assert_eq!(first, pool(1), "{task:?}: a pool of 1 thread");
            assert_eq!(first, pool(4), "{task:?}: a pool of 4 threads");

            // The values moved, so the test compares trained models, not initial ones.
            let mut untrained = genome.clone();

            untrained.learning_rate = crate::genome::LR_MIN;
            assert_ne!(first.1, fingerprint(task, &untrained).1, "{task:?}");
        }
    }

    #[test]
    fn every_optimizer_reduces_the_loss() {
        let spec = TaskKind::BarsAndStripes.spec();
        let data = head(&generate(&spec, Split::Train, 1), 128);

        for optimizer in [Optimizer::Sgd, Optimizer::Momentum, Optimizer::Adam] {
            let genome = Genome {
                optimizer,
                learning_rate: if optimizer == Optimizer::Adam {
                    0.003
                } else {
                    0.05
                },
                l2: 0.001,
                batch_size: 16,
                ..plain_2d()
            };
            let (_, stats, losses) =
                train_logged(&genome, &spec, &data, &Limits::default()).expect("train");
            let first: f32 = losses[..8].iter().sum();
            let last: f32 = losses[losses.len() - 8..].iter().sum();

            assert!(!stats.diverged, "{optimizer:?}");
            assert!(last < 0.5 * first, "{optimizer:?}: {first} -> {last}");
        }
    }

    #[test]
    fn flops_follow_the_plan_and_match_the_estimate() {
        let spec = TaskKind::Motif1d.spec();
        let data = generate(&spec, Split::Train, 1);
        let genome = Genome {
            epochs: 5,
            batch_size: 128,
            ..normed_1d()
        };
        let plan = genome.cnn_plan(&spec).expect("plan");
        let (_, stats) = train(&genome, &spec, &data, &Limits::default()).expect("train");

        assert_eq!(stats.flops, genome.train_flops(&spec));
        assert_eq!(
            stats.flops,
            3 * plan.forward_flops * spec.n_train as u64 * 5
        );
        assert_eq!(stats.steps, (5 * spec.n_train.div_ceil(128)) as u64);
    }

    #[test]
    fn budgets_and_the_kill_switch_stop_cnn_training_cleanly() {
        let spec = TaskKind::Shapes.spec();
        let data = head(&generate(&spec, Split::Train, 1), 64);
        let genome = Genome {
            batch_size: 16,
            ..genomes::lenet()
        };

        let no_time = Limits {
            max_candidate_seconds: 0.0,
            ..Limits::default()
        };
        let (model, stats) = train(&genome, &spec, &data, &no_time).expect("train");

        assert!(stats.stopped_by_budget && stats.epochs_done == 0);
        assert!(model.is_finite());

        // Answers "stop" from the second check on: the start of epoch 2.
        let calls = std::cell::Cell::new(0u32);
        let stop = || {
            calls.set(calls.get() + 1);
            calls.get() >= 2
        };
        let (model, stats) =
            train_with_stop(&genome, &spec, &data, &Limits::default(), &stop).expect("train");

        assert!(stats.stopped_by_kill_switch && !stats.stopped_by_budget);
        assert_eq!(stats.epochs_done, 1);
        assert_eq!(stats.steps, 4);
        assert!(stats.final_train_loss.is_finite());
        assert!(matches!(model, Model::Cnn(_)));
    }

    #[test]
    fn divergence_is_reported_with_finite_loss() {
        let spec = TaskKind::Shapes.spec();
        let mut data = head(&generate(&spec, Split::Train, 1), 64);

        // Finite inputs whose sums overflow f32 in the first layers.
        for v in data.x.iter_mut() {
            *v *= 2.0e38;
        }

        let genome = Genome {
            epochs: 5,
            ..plain_2d()
        };
        let (model, stats) = train(&genome, &spec, &data, &Limits::default()).expect("train");

        assert!(stats.diverged);
        assert_eq!(stats.final_train_loss, f32::MAX);
        assert_eq!(stats.epochs_done, 0);
        assert!(
            model.is_finite(),
            "no step was taken with a non-finite loss"
        );
    }

    #[test]
    fn a_regression_target_trains_with_the_mean_squared_error() {
        // No task gives an image with regression targets yet; this one is made up from the
        // shapes task: the target is the mean of the image.
        let mut spec = TaskKind::Shapes.spec();
        let classes = head(&generate(&spec, Split::Train, 1), 128);
        let means: Vec<f32> = classes
            .x
            .chunks_exact(classes.n_in)
            .map(|row| row.iter().sum::<f32>() / row.len() as f32)
            .collect();
        let data = Dataset {
            targets: Targets::Values(means),
            n_out: 1,
            ..classes
        };

        spec.n_out = 1;
        spec.objective = Objective::Regression;

        let genome = Genome {
            epochs: 10,
            batch_size: 16,
            learning_rate: 0.003,
            ..plain_2d()
        };
        let (model, stats, losses) =
            train_logged(&genome, &spec, &data, &Limits::default()).expect("train");
        let metrics = score::evaluate(&model, &spec, &data);

        assert!(!stats.diverged);
        assert!(losses[losses.len() - 1] < 0.1 * losses[0], "{losses:?}");
        assert!(metrics.r2.is_some_and(|r2| r2 > 0.5), "{metrics:?}");
    }
}
