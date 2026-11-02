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
//! (ticket 01 checked 19 thread settings). This holds while the clock does not stop the
//! training; the time model of the shape check keeps a valid CNN within its FLOP budget's
//! time (`cnn::shape`, "Time"). Another CPU can give other bits: gemm, the matrix library
//! under candle, picks its kernel (with or without fused multiply-add) and its cache
//! blocking at run time from the CPU. Tests therefore compare results within one machine
//! only.

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
    use super::CnnLearner;
    use crate::genome::{Arch, Genome};
    use crate::task::{Dataset, Targets, TaskSpec};
    use crate::train::Learner;
    use crate::util::Result;

    /// One step of the CNN `genome` on all rows of `data` gives a finite loss and an update,
    /// and its model then gives finite outputs. A failure names the genome.
    pub(crate) fn trains_one_step(genome: &Genome, spec: &TaskSpec, data: &Dataset) {
        fn ok<T>(name: &str, result: Result<T>) -> T {
            result.unwrap_or_else(|e| panic!("{name}: {e}"))
        }

        let name = genome.summary();
        let Arch::Cnn(cnn) = &genome.arch else {
            panic!("{name}: not a CNN");
        };
        let rows: Vec<usize> = (0..data.n).collect();
        let mut learner = ok(&name, CnnLearner::new(genome, cnn, spec, data));
        let loss = ok(&name, learner.loss_and_grad(&rows));

        assert!(loss.is_finite(), "{name}: loss {loss}");

        ok(&name, learner.update(genome.learning_rate, 0.0));

        let outputs = ok(
            &name,
            learner
                .into_model()
                .and_then(|model| model.evaluator(spec))
                .and_then(|eval| eval.outputs(&data.x, data.n)),
        );

        assert!(outputs.iter().all(|v| v.is_finite()), "{name}");
    }

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
    use crate::util::sha256_hex;

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
        let cnn = learns(TaskKind::Motif1d, &plain_1d());

        assert!(cnn > 0.65, "chance 0.5");

        // An MLP with the same head and training genes on the flat sequence stays near
        // chance, so the convolutions do the work: they find the motif at every place, a
        // dense layer would have to learn each place on its own. On this machine the MLP
        // reached 0.52 (review of ticket 02).
        let mlp = Genome {
            arch: crate::genome::Arch::Mlp,
            ..plain_1d()
        };
        let mlp = learns(TaskKind::Motif1d, &mlp);

        assert!(cnn > mlp + 0.15, "CNN {cnn}, MLP {mlp}");
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

    /// A 2-D CNN of every block of ticket 03, with both options: a basic block with
    /// squeeze-and-excitation and the input as shortcut, a strided bottleneck with a
    /// projection, a separable block with squeeze-and-excitation and a projection, and a
    /// strided inverted residual. Batch norm, ReLU and ReLU6.
    fn mixed_blocks() -> Genome {
        use crate::cnn::genome::Block;

        let relu = CnnActivation::Relu;
        let unit = |block: Block, width: usize| Stage {
            norm: Norm::Batch,
            ..genomes::stage(block, width, 3, relu)
        };
        let cnn = CnnGenome {
            dims: Dims::TwoD,
            stem: Stem {
                width: 8,
                kernel: 3,
                stride: 2,
                down: Downsample::MaxPool,
                norm: Norm::Batch,
                activation: relu,
            },
            stages: vec![
                Stage {
                    se: true,
                    residual: true,
                    ..unit(Block::Basic, 8)
                },
                Stage {
                    stride: 2,
                    residual: true,
                    ..unit(Block::Bottleneck, 16)
                },
                Stage {
                    se: true,
                    residual: true,
                    ..unit(Block::Separable, 8)
                },
                Stage {
                    stride: 2,
                    activation: CnnActivation::Relu6,
                    ..unit(Block::InvertedResidual, 16)
                },
            ],
            head: Head::GlobalAvgPool,
        };

        Genome {
            epochs: 5,
            ..genomes::genome(cnn, vec![8], Activation::Relu)
        }
    }

    /// A 2-D CNN of the 2-D blocks of ticket 04: a dense layer, an Inception block with
    /// squeeze-and-excitation, a Fire module with the input as shortcut, a ShuffleNet unit
    /// with 2 groups and the input as shortcut, a transition that pools by average, a
    /// ConvNeXt block with layer norm and GELU, and a decoder that upsamples by a transposed
    /// convolution and joins the map before the transition's pooling. Batch norm and ReLU
    /// elsewhere. With a stem of 8 channels and blocks of 12 and 16 the time model gave
    /// 28.8 s for 5 epochs, above the budget of 20 s, so the blocks are narrow.
    fn mixed_blocks_b() -> Genome {
        use crate::cnn::genome::Block;

        let relu = CnnActivation::Relu;
        let unit = |block: Block, width: usize| Stage {
            norm: Norm::Batch,
            ..genomes::stage(block, width, 3, relu)
        };
        let cnn = CnnGenome {
            dims: Dims::TwoD,
            stem: Stem {
                width: 4,
                kernel: 3,
                stride: 2,
                down: Downsample::MaxPool,
                norm: Norm::Batch,
                activation: relu,
            },
            stages: vec![
                unit(Block::Dense, 4),
                Stage {
                    se: true,
                    ..unit(Block::Inception, 8)
                },
                Stage {
                    residual: true,
                    ..unit(Block::Fire, 8)
                },
                Stage {
                    groups: 2,
                    residual: true,
                    ..unit(Block::Shuffle, 8)
                },
                Stage {
                    stride: 2,
                    down: Downsample::AvgPool,
                    ..unit(Block::Transition, 8)
                },
                Stage {
                    residual: true,
                    norm: Norm::Layer,
                    activation: CnnActivation::Gelu,
                    ..unit(Block::ConvNext, 8)
                },
                Stage {
                    stride: 2,
                    ..unit(Block::Decoder, 4)
                },
            ],
            head: Head::GlobalAvgPool,
        };

        Genome {
            epochs: 5,
            ..genomes::genome(cnn, vec![8], Activation::Relu)
        }
    }

    /// A 1-D CNN of TCN blocks and the 1-D blocks of ticket 04: a causal block with dilation
    /// 2, an Inception block, a ShuffleNet unit that moves by 2, a decoder that upsamples by a
    /// transposed convolution along the sequence and joins the unit's input, and a ConvNeXt
    /// block with a depthwise kernel of 7.
    fn mixed_sequence_b() -> Genome {
        use crate::cnn::genome::Block;

        let relu = CnnActivation::Relu;
        let unit = |block: Block, width: usize| Stage {
            norm: Norm::Batch,
            ..genomes::stage(block, width, 3, relu)
        };
        let cnn = CnnGenome {
            dims: Dims::OneD,
            stem: Stem {
                width: 8,
                kernel: 5,
                stride: 2,
                down: Downsample::MaxPool,
                norm: Norm::Batch,
                activation: relu,
            },
            stages: vec![
                Stage {
                    dilation: 2,
                    residual: true,
                    ..unit(Block::Tcn, 8)
                },
                unit(Block::Inception, 8),
                Stage {
                    stride: 2,
                    groups: 2,
                    ..unit(Block::Shuffle, 8)
                },
                Stage {
                    stride: 2,
                    ..unit(Block::Decoder, 8)
                },
                Stage {
                    kernel: 7,
                    residual: true,
                    norm: Norm::Layer,
                    activation: CnnActivation::Gelu,
                    ..unit(Block::ConvNext, 8)
                },
            ],
            head: Head::GlobalAvgPool,
        };

        Genome {
            epochs: 5,
            ..genomes::genome(cnn, Vec::new(), Activation::Relu)
        }
    }

    /// VGG-like with batch norm and max pooling, a 1-D CNN with every norm and pool, a 2-D
    /// CNN of the blocks of ticket 03, a 2-D and a 1-D CNN of the blocks of ticket 04, and a
    /// U-Net with both upsamplings and the per-cell head on the mirror grid.
    fn thread_test_genomes() -> [(TaskKind, Genome); 6] {
        [
            (TaskKind::Shapes, genomes::vgg()),
            (TaskKind::Motif1d, normed_1d()),
            (TaskKind::Shapes, mixed_blocks()),
            (TaskKind::Shapes, mixed_blocks_b()),
            (TaskKind::Motif1d, mixed_sequence_b()),
            (
                TaskKind::MirrorGrid,
                block_genome(
                    "a u-net with a transposed convolution, nearest upsampling and a per-cell \
                     head",
                )
                .1,
            ),
        ]
    }

    /// The training runs in rayon pools of 1 and 4 threads, as candidates do in a run.
    ///
    /// This does not change the threads of candle's matrix products: gemm takes their
    /// number from `RAYON_NUM_THREADS` when a product starts, not from the pool around it.
    /// `training_is_byte_equal_at_1_and_3_matrix_threads` varies that number.
    #[test]
    fn training_is_byte_equal_twice_and_in_rayon_pools_of_1_and_4_threads() {
        for (task, genome) in thread_test_genomes() {
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

    /// Prints the hash of each fingerprint of `thread_test_genomes`, one line each, for
    /// `training_is_byte_equal_at_1_and_3_matrix_threads`, which runs it in child
    /// processes. Alone it checks nothing, so it is ignored.
    #[test]
    #[ignore]
    fn print_the_fingerprints_for_the_thread_test() {
        for (task, genome) in thread_test_genomes() {
            let (losses, file, stats) = fingerprint(task, &genome);
            let text = format!("{losses:?} {file} {stats:?}");

            println!("FINGERPRINT {task:?} {}", sha256_hex(text.as_bytes()));
        }
    }

    #[test]
    fn training_is_byte_equal_at_1_and_3_matrix_threads() {
        // candle splits a matrix product into `RAYON_NUM_THREADS` parts, read from the
        // environment when the product starts (candle-core/src/utils.rs:403-422 and
        // cpu_backend/mod.rs:1482-1487 at the pinned commit): 1 means no split at all. A
        // test cannot change the environment of its own process, so this one runs the
        // test above in child processes, with 1 and with 3 threads.
        let run = |threads: &str| -> Vec<String> {
            let out = std::process::Command::new(std::env::current_exe().expect("test binary"))
                .args([
                    "--exact",
                    "cnn::train::tests::print_the_fingerprints_for_the_thread_test",
                    "--ignored",
                    "--nocapture",
                    "--test-threads",
                    "1",
                ])
                .env("RAYON_NUM_THREADS", threads)
                .output()
                .expect("run the test binary");

            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );

            // The harness may print the name of the test on the same line, before it.
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter_map(|line| line.find("FINGERPRINT ").map(|at| line[at..].to_string()))
                .collect()
        };
        let one = run("1");

        assert_eq!(one.len(), 6, "{one:?}");
        assert_eq!(one, run("3"));
    }

    /// The genome of `cnn::genome::testing::block_genomes` with this name, and its task.
    fn block_genome(name: &str) -> (TaskKind, Genome) {
        genomes::block_genomes()
            .into_iter()
            .find(|(n, _, _)| *n == name)
            .map(|(_, task, genome)| (task, genome))
            .unwrap_or_else(|| panic!("no block genome {name}"))
    }

    /// On a toy batch of 32 samples of its task, the genome `name` of `block_genomes` gets a
    /// finite gradient for every value, and every tensor a gradient that is not zero; then
    /// 50 steps of its optimizer (Adam, learning rate 0.01) on that batch bring the loss
    /// below 0.7 of its first value.
    fn learns_a_toy_batch(name: &str) {
        let (task, genome) = block_genome(name);
        let spec = task.spec();
        let data = head(&generate(&spec, Split::Train, 1), 32);
        let crate::genome::Arch::Cnn(cnn) = &genome.arch else {
            panic!("expected a CNN genome");
        };
        let mut learner = CnnLearner::new(&genome, cnn, &spec, &data).expect("learner");
        let rows: Vec<usize> = (0..32).collect();
        let first = learner.loss_and_grad(&rows).expect("loss");
        let grads = learner.grad_weights.iter().chain(&learner.grad_biases);

        for (i, grad) in grads.enumerate() {
            assert!(grad.iter().all(|g| g.is_finite()), "{name}: tensor {i}");
            assert!(
                grad.iter().any(|&g| g != 0.0),
                "{name}: tensor {i} got no gradient"
            );
        }

        let mut last = first;

        for _ in 0..50 {
            learner.update(genome.learning_rate, 0.0).expect("update");
            last = learner.loss_and_grad(&rows).expect("loss");

            assert!(last.is_finite(), "{name}");
        }

        eprintln!("{name}: loss {first:.3} -> {last:.3} in 50 steps");

        assert!(last < 0.7 * first, "{name}: {first} -> {last}");
    }

    #[test]
    fn a_plain_stage_with_both_options_learns_a_toy_batch() {
        learns_a_toy_batch("plain with squeeze-and-excitation and an identity shortcut");
    }

    #[test]
    fn dilated_plain_stages_learn_a_toy_batch() {
        learns_a_toy_batch("plain, dilated");
    }

    #[test]
    fn a_basic_block_with_a_strided_projection_learns_a_toy_batch() {
        learns_a_toy_batch("basic, strided, with a projection");
    }

    #[test]
    fn a_basic_block_without_norm_learns_a_toy_batch() {
        learns_a_toy_batch("basic without norm, with a projection that has a bias");
    }

    #[test]
    fn a_bottleneck_learns_a_toy_batch() {
        learns_a_toy_batch("bottleneck with group norm, groups and a projection");
    }

    #[test]
    fn a_separable_block_learns_a_toy_batch() {
        learns_a_toy_batch(
            "separable with layer norm, squeeze-and-excitation, a projection and pooling",
        );
    }

    #[test]
    fn inverted_residuals_learn_a_toy_batch() {
        learns_a_toy_batch("inverted residuals, one with an identity shortcut, one strided");
    }

    #[test]
    fn blocks_along_a_sequence_learn_a_toy_batch() {
        learns_a_toy_batch("1-D basic and separable blocks with squeeze-and-excitation");
    }

    #[test]
    fn dense_layers_and_a_transition_learn_a_toy_batch() {
        learns_a_toy_batch("dense layers and a transition with group norm");
    }

    #[test]
    fn inception_blocks_learn_a_toy_batch() {
        learns_a_toy_batch("inception blocks, one strided with groups and a projection");
    }

    #[test]
    fn fire_modules_learn_a_toy_batch() {
        learns_a_toy_batch(
            "fire modules, one with squeeze-and-excitation and an identity shortcut",
        );
    }

    #[test]
    fn shufflenet_units_learn_a_toy_batch() {
        learns_a_toy_batch("shufflenet units with 2 and 4 groups and projections");
    }

    #[test]
    fn convnext_blocks_learn_a_toy_batch() {
        learns_a_toy_batch("convnext blocks with layer norm and gelu, one strided");
    }

    #[test]
    fn tcn_blocks_learn_a_toy_batch() {
        learns_a_toy_batch("tcn blocks with dilations 1, 2 and 4, the last strided");
    }

    #[test]
    fn a_u_net_learns_a_toy_batch() {
        learns_a_toy_batch(
            "a u-net with a transposed convolution, nearest upsampling and a per-cell head",
        );
    }

    #[test]
    fn the_blocks_of_ticket_04_learn_a_toy_batch_along_a_sequence() {
        learns_a_toy_batch("1-D inception, shufflenet, dense, decoder and convnext blocks");
    }

    #[test]
    fn an_inverted_residual_that_pools_its_batch_norm_learns_a_toy_batch() {
        learns_a_toy_batch("an inverted residual that downsamples by max pooling, without options");
    }

    #[test]
    fn inverted_residuals_that_pool_their_batch_norm_learn_a_toy_batch_along_a_sequence() {
        learns_a_toy_batch("1-D inverted residuals that downsample by max and by average pooling");
    }

    /// Every block kind on `task`, as a stage that moves by 2 with every downsampling,
    /// every norm and every pair of options, at a width that keeps the 8 channels of the stem
    /// and at one that changes them, after a stem that moves by 2 and by 4 (some blocks fit
    /// the time budget only on the smaller map): each such genome that passes the shape check
    /// trains (`trains_one_step`). Returns how many genomes passed the shape check.
    ///
    /// The shape check promises that a valid genome trains. Two layouts once broke that
    /// promise: an inverted residual that pooled the output of its batch norm, and the group
    /// norm of the projection of a ConvNeXt block.
    fn every_valid_stage_trains_one_step(dims: Dims, task: TaskKind) -> usize {
        use crate::cnn::genome::{BLOCKS, DOWNSAMPLES, NORMS};

        let spec = task.spec();
        let data = head(&generate(&spec, Split::Train, 1), 4);
        let limits = Limits::default();
        let mut layouts = Vec::new();

        for block in BLOCKS {
            for down in DOWNSAMPLES {
                for norm in NORMS {
                    for (se, residual) in
                        [(false, false), (true, false), (false, true), (true, true)]
                    {
                        for (stem_stride, width) in [(2, 6), (2, 8), (4, 6), (4, 8)] {
                            let stage = Stage {
                                block,
                                width,
                                kernel: 3,
                                stride: 2,
                                down,
                                dilation: 1,
                                groups: 1,
                                se,
                                residual,
                                norm,
                                activation: CnnActivation::Relu,
                            };

                            layouts.push((stem_stride, stage));
                        }
                    }
                }
            }
        }

        let mut valid = 0;

        for (stem_stride, stage) in layouts {
            let cnn = CnnGenome {
                dims,
                stem: Stem {
                    width: 8,
                    kernel: 3,
                    stride: stem_stride,
                    down: Downsample::MaxPool,
                    norm: Norm::Batch,
                    activation: CnnActivation::Relu,
                },
                stages: vec![stage],
                head: Head::GlobalAvgPool,
            };
            let mut genome = genomes::genome(cnn, Vec::new(), crate::genome::Activation::Relu);

            genome.epochs = 5;

            if genome.validate(&limits, &spec).is_ok() {
                super::testing::trains_one_step(&genome, &spec, &data);
                valid += 1;
            }
        }

        eprintln!(
            "{dims:?} on {}: {valid} valid stages trained one step",
            task.name()
        );

        valid
    }

    // The bounds keep the tests from passing on a shape check that refuses almost all
    // layouts; 1300 of the 2-D and 1672 of the 1-D layouts passed it when they were written.
    #[test]
    fn every_valid_2d_stage_trains_one_step() {
        let valid = every_valid_stage_trains_one_step(Dims::TwoD, TaskKind::Shapes);

        assert!(valid > 650, "only {valid} valid stages");
    }

    #[test]
    fn every_valid_1d_stage_trains_one_step() {
        let valid = every_valid_stage_trains_one_step(Dims::OneD, TaskKind::Motif1d);

        assert!(valid > 830, "only {valid} valid stages");
    }

    #[test]
    fn every_block_genome_has_a_toy_batch_test() {
        // The names that the tests above pass to `learns_a_toy_batch`.
        let tested = [
            "plain with squeeze-and-excitation and an identity shortcut",
            "plain, dilated",
            "basic, strided, with a projection",
            "basic without norm, with a projection that has a bias",
            "bottleneck with group norm, groups and a projection",
            "separable with layer norm, squeeze-and-excitation, a projection and pooling",
            "inverted residuals, one with an identity shortcut, one strided",
            "1-D basic and separable blocks with squeeze-and-excitation",
            "dense layers and a transition with group norm",
            "inception blocks, one strided with groups and a projection",
            "fire modules, one with squeeze-and-excitation and an identity shortcut",
            "shufflenet units with 2 and 4 groups and projections",
            "convnext blocks with layer norm and gelu, one strided",
            "tcn blocks with dilations 1, 2 and 4, the last strided",
            "a u-net with a transposed convolution, nearest upsampling and a per-cell head",
            "1-D inception, shufflenet, dense, decoder and convnext blocks",
            "an inverted residual that downsamples by max pooling, without options",
            "1-D inverted residuals that downsample by max and by average pooling",
        ];
        let names: Vec<&str> = genomes::block_genomes()
            .into_iter()
            .map(|(name, _, _)| name)
            .collect();

        assert_eq!(names, tested);

        // Together they hold every block kind, both options and the per-cell head.
        let cnns: Vec<CnnGenome> = genomes::block_genomes()
            .into_iter()
            .filter_map(|(_, _, g)| match g.arch {
                crate::genome::Arch::Cnn(cnn) => Some(cnn),
                crate::genome::Arch::Mlp => None,
            })
            .collect();
        let any = |f: &dyn Fn(&Stage) -> bool| cnns.iter().any(|cnn| cnn.stages.iter().any(f));

        for block in crate::cnn::genome::BLOCKS {
            assert!(any(&|s| s.block == block), "{block:?}");
        }

        assert!(any(&|s| s.se) && any(&|s| s.residual));
        assert!(cnns.iter().any(|cnn| cnn.head == Head::PerCell));

        // Each fits the default limits, so a run could train it as it is.
        for (name, task, genome) in genomes::block_genomes() {
            let checked = genome.validate(&Limits::default(), &task.spec());

            assert!(checked.is_ok(), "{name}: {checked:?}");
        }
    }

    #[test]
    fn every_tensor_gets_a_gradient_and_moves() {
        for (task, genome) in thread_test_genomes() {
            let spec = task.spec();
            let data = head(&generate(&spec, Split::Train, 1), 64);
            let crate::genome::Arch::Cnn(cnn) = &genome.arch else {
                panic!("expected a CNN genome");
            };
            let mut learner = CnnLearner::new(&genome, cnn, &spec, &data).expect("learner");
            let initial = learner.values.clone();
            let rows: Vec<usize> = (0..32).collect();

            learner.loss_and_grad(&rows).expect("loss");

            // Every kernel, matrix, bias, scale and shift gets a gradient from one batch.
            let grads = learner.grad_weights.iter().chain(&learner.grad_biases);

            for (i, grad) in grads.enumerate() {
                assert!(
                    grad.iter().any(|&g| g != 0.0),
                    "{task:?}: tensor {i} got no gradient"
                );
            }

            for _ in 0..3 {
                learner.loss_and_grad(&rows).expect("loss");
                learner.update(genome.learning_rate, 0.0).expect("update");
            }

            // After three steps every tensor differs from its initial value, also inside
            // candle: the network reads the same values back.
            let now = learner.values.weights.iter().chain(&learner.values.biases);
            let before = initial.weights.iter().chain(&initial.biases);

            for (i, (now, before)) in now.zip(before).enumerate() {
                assert_ne!(now, before, "{task:?}: tensor {i} did not move");
            }

            for (param, values) in learner.net.weights().iter().zip(&learner.values.weights) {
                assert_eq!(&param.tensor().to_vec_f32().expect("values"), values);
            }
        }
    }

    /// A small CNN of smooth parts only, so that finite differences apply: batch norm,
    /// group norm and layer norm, tanh, SiLU and GELU, average pooling, a strided and a
    /// grouped dilated convolution, and a dense head with tanh.
    fn smooth(dims: Dims) -> Genome {
        let cnn = CnnGenome {
            dims,
            stem: Stem {
                width: 4,
                kernel: 3,
                stride: 2,
                down: Downsample::AvgPool,
                norm: Norm::Batch,
                activation: CnnActivation::Tanh,
            },
            stages: vec![
                Stage {
                    dilation: 2,
                    groups: 2,
                    norm: Norm::Group,
                    ..genomes::plain(4, 3, CnnActivation::Silu)
                },
                Stage {
                    stride: 2,
                    down: Downsample::Conv,
                    norm: Norm::Layer,
                    ..genomes::plain(6, 3, CnnActivation::Gelu)
                },
            ],
            head: Head::GlobalAvgPool,
        };

        genomes::genome(cnn, vec![5], Activation::Tanh)
    }

    /// The blocks of ticket 03 with smooth activations and average pooling, so that finite
    /// differences apply: a strided basic block with group norm, squeeze-and-excitation and
    /// a projection, a bottleneck with batch norm and a projection, a separable block with
    /// layer norm, squeeze-and-excitation and the input as shortcut, and an inverted
    /// residual with group norm and a projection. The ReLU inside squeeze-and-excitation is
    /// its only kink. No layer norm sees fewer than 16 channels: over the 2 inner channels
    /// of a bottleneck of width 8 it gives about +1 or -1 and jumps between them.
    fn smooth_blocks() -> Genome {
        use crate::cnn::genome::Block;

        let unit = |block: Block, width: usize, norm: Norm, activation: CnnActivation| Stage {
            norm,
            residual: true,
            ..genomes::stage(block, width, 3, activation)
        };
        let cnn = CnnGenome {
            dims: Dims::TwoD,
            stem: Stem {
                width: 4,
                kernel: 3,
                stride: 2,
                down: Downsample::AvgPool,
                norm: Norm::Batch,
                activation: CnnActivation::Tanh,
            },
            stages: vec![
                Stage {
                    stride: 2,
                    se: true,
                    ..unit(Block::Basic, 8, Norm::Group, CnnActivation::Silu)
                },
                unit(Block::Bottleneck, 16, Norm::Batch, CnnActivation::Gelu),
                Stage {
                    se: true,
                    ..unit(Block::Separable, 16, Norm::Layer, CnnActivation::Tanh)
                },
                unit(Block::InvertedResidual, 8, Norm::Group, CnnActivation::Silu),
            ],
            head: Head::GlobalAvgPool,
        };

        genomes::genome(cnn, vec![5], Activation::Tanh)
    }

    /// The 2-D blocks of ticket 04 with smooth activations and average pooling, so that
    /// finite differences apply: a dense layer, an Inception block with GELU, a Fire module,
    /// a ShuffleNet unit with 2 groups and a projection, a transition with group norm that
    /// pools by average, a ConvNeXt block with layer norm and the input as shortcut, and a
    /// decoder that upsamples by a transposed convolution and joins the map before the
    /// transition's pooling. The max pooling branch of the Inception block is its only kink.
    fn smooth_blocks_b() -> Genome {
        use crate::cnn::genome::Block;

        let unit = |block: Block, width: usize, norm: Norm, activation: CnnActivation| Stage {
            norm,
            ..genomes::stage(block, width, 3, activation)
        };
        let cnn = CnnGenome {
            dims: Dims::TwoD,
            stem: Stem {
                width: 4,
                kernel: 3,
                stride: 2,
                down: Downsample::AvgPool,
                norm: Norm::Batch,
                activation: CnnActivation::Tanh,
            },
            stages: vec![
                unit(Block::Dense, 4, Norm::Batch, CnnActivation::Silu),
                unit(Block::Inception, 8, Norm::Batch, CnnActivation::Gelu),
                unit(Block::Fire, 8, Norm::Batch, CnnActivation::Tanh),
                Stage {
                    groups: 2,
                    residual: true,
                    ..unit(Block::Shuffle, 16, Norm::Batch, CnnActivation::Silu)
                },
                Stage {
                    stride: 2,
                    down: Downsample::AvgPool,
                    ..unit(Block::Transition, 8, Norm::Group, CnnActivation::Tanh)
                },
                Stage {
                    residual: true,
                    ..unit(Block::ConvNext, 8, Norm::Layer, CnnActivation::Gelu)
                },
                Stage {
                    stride: 2,
                    ..unit(Block::Decoder, 8, Norm::Batch, CnnActivation::Tanh)
                },
            ],
            head: Head::GlobalAvgPool,
        };

        genomes::genome(cnn, vec![5], Activation::Tanh)
    }

    /// TCN blocks and a 1-D decoder with smooth activations: a causal block with batch norm
    /// and SiLU and the input as shortcut, a causal block with group norm and tanh that moves
    /// by 2 with dilation 2 and a projection, and a decoder that upsamples by a transposed
    /// convolution along the sequence and joins that block's input.
    fn smooth_sequence_b() -> Genome {
        use crate::cnn::genome::Block;

        let unit = |block: Block, width: usize, norm: Norm, activation: CnnActivation| Stage {
            norm,
            ..genomes::stage(block, width, 3, activation)
        };
        let cnn = CnnGenome {
            dims: Dims::OneD,
            stem: Stem {
                width: 4,
                kernel: 3,
                stride: 2,
                down: Downsample::AvgPool,
                norm: Norm::Batch,
                activation: CnnActivation::Tanh,
            },
            stages: vec![
                Stage {
                    residual: true,
                    ..unit(Block::Tcn, 4, Norm::Batch, CnnActivation::Silu)
                },
                Stage {
                    stride: 2,
                    dilation: 2,
                    residual: true,
                    ..unit(Block::Tcn, 8, Norm::Group, CnnActivation::Tanh)
                },
                Stage {
                    stride: 2,
                    ..unit(Block::Decoder, 4, Norm::Batch, CnnActivation::Gelu)
                },
            ],
            head: Head::GlobalAvgPool,
        };

        genomes::genome(cnn, vec![5], Activation::Tanh)
    }

    /// A U-Net with smooth activations on the mirror grid, so the loss is the mean squared
    /// error: a stage with group norm and SiLU that pools 8x8 to 4x4 by average, a decoder
    /// that upsamples by repeating cells and joins the map before that pooling, and the
    /// per-cell head with a dense layer of 3 per cell.
    fn smooth_unet() -> Genome {
        use crate::cnn::genome::Block;

        let cnn = CnnGenome {
            dims: Dims::TwoD,
            stem: Stem {
                width: 4,
                kernel: 3,
                stride: 1,
                down: Downsample::Conv,
                norm: Norm::Batch,
                activation: CnnActivation::Tanh,
            },
            stages: vec![
                Stage {
                    stride: 2,
                    down: Downsample::AvgPool,
                    norm: Norm::Group,
                    ..genomes::plain(8, 3, CnnActivation::Silu)
                },
                Stage {
                    stride: 2,
                    down: Downsample::MaxPool,
                    norm: Norm::Batch,
                    ..genomes::stage(Block::Decoder, 4, 3, CnnActivation::Tanh)
                },
            ],
            head: Head::PerCell,
        };

        genomes::genome(cnn, vec![3], Activation::Tanh)
    }

    #[test]
    fn loss_and_grad_matches_finite_differences() {
        // f32 throughout, so the step is large and the bound loose; the gradient check of
        // `ops` runs every operation in f64 with a bound of 1e-6. The deeper genome of the
        // blocks bends within 1e-2 (at 1e-2, 1 of its 162 values missed the bound, while 1e-3
        // matched), so it takes the smaller step, and so do the genomes of ticket 04.
        for (task, genome, step) in [
            (TaskKind::Shapes, smooth(Dims::TwoD), 1e-2),
            (TaskKind::Motif1d, smooth(Dims::OneD), 1e-2),
            (TaskKind::Shapes, smooth_blocks(), 1e-3),
            (TaskKind::Shapes, smooth_blocks_b(), 1e-3),
            (TaskKind::Motif1d, smooth_sequence_b(), 1e-3),
            (TaskKind::MirrorGrid, smooth_unet(), 1e-3),
        ] {
            let spec = task.spec();
            let data = head(&generate(&spec, Split::Train, 1), 16);
            let crate::genome::Arch::Cnn(cnn) = &genome.arch else {
                panic!("expected a CNN genome");
            };
            let mut learner = CnnLearner::new(&genome, cnn, &spec, &data).expect("learner");
            let rows: Vec<usize> = (0..16).collect();

            learner.loss_and_grad(&rows).expect("loss");

            let analytic = (learner.grad_weights.clone(), learner.grad_biases.clone());
            let mut checked = 0;
            let mut worst = 0.0f64;
            let mut largest = 0.0f64;

            for (decay, lists) in [(true, &analytic.0), (false, &analytic.1)] {
                for (t, grads) in lists.iter().enumerate() {
                    for j in [0, grads.len() / 2, grads.len() - 1] {
                        let mut loss_at = |delta: f32| {
                            let list = if decay {
                                &mut learner.values.weights[t]
                            } else {
                                &mut learner.values.biases[t]
                            };

                            list[j] += delta;
                            learner
                                .net
                                .set(&learner.values.weights, &learner.values.biases)
                                .expect("set");

                            let loss = learner.loss_and_grad(&rows).expect("loss");
                            let list = if decay {
                                &mut learner.values.weights[t]
                            } else {
                                &mut learner.values.biases[t]
                            };

                            list[j] -= delta;

                            loss as f64
                        };
                        let numeric = (loss_at(step) - loss_at(-step)) / (2.0 * step as f64);
                        let analytic = grads[j] as f64;
                        let bound = 1e-3 + 0.03 * analytic.abs();

                        worst = worst.max((numeric - analytic).abs() / bound);
                        largest = largest.max(analytic.abs());

                        assert!(
                            (numeric - analytic).abs() <= bound,
                            "{task:?}: tensor {t} ({decay}) value {j}: analytic {analytic}, \
                             numeric {numeric}"
                        );

                        checked += 1;
                    }
                }
            }

            eprintln!(
                "{}: {checked} values, largest gradient {largest:.3e}, worst error {worst:.2} \
                 of the bound",
                task.name()
            );

            assert!(checked >= 30, "{task:?}: {checked} values checked");

            // The gradients are large against the bound, so a wrong one would show.
            assert!(largest > 0.02, "{task:?}: {largest}");
        }
    }

    /// Valid CNNs that come as close to the time budget as `validate` allows, one per kind
    /// of layout that runs slowly per FLOP: narrow layers at full size, batch norm on 4
    /// channels, layer norm with GELU, depthwise convolutions, many tiny layers, 1-D layers
    /// of one channel, and group norm with pooling. Each takes the most stages that are
    /// valid at 5 epochs, then the most epochs that are valid.
    fn slow_valid_cnns() -> Vec<(TaskKind, Genome)> {
        let relu = CnnActivation::Relu;
        let unit = |width: usize, kernel: usize, norm: Norm, activation: CnnActivation| Stage {
            norm,
            ..genomes::plain(width, kernel, activation)
        };
        let stem = |stage: &Stage, stride: usize, down: Downsample| Stem {
            width: stage.width,
            kernel: stage.kernel,
            stride,
            down,
            norm: stage.norm,
            activation: stage.activation,
        };
        let grouped = Stage {
            groups: 32,
            ..unit(32, 3, Norm::Batch, relu)
        };
        let sequence = Stage {
            dilation: 2,
            ..unit(1, 11, Norm::None, relu)
        };
        let pooled = unit(8, 3, Norm::Group, CnnActivation::Silu);
        let full = |stage: Stage| (stem(&stage, 1, Downsample::Conv), stage);

        // The task, the stem and the stage that repeats, and the head of each layout.
        let layouts = [
            (
                TaskKind::Shapes,
                full(unit(1, 9, Norm::None, relu)),
                Head::GlobalAvgPool,
            ),
            (
                TaskKind::Shapes,
                full(unit(4, 3, Norm::Batch, relu)),
                Head::GlobalAvgPool,
            ),
            (
                TaskKind::Shapes,
                full(unit(8, 1, Norm::Layer, CnnActivation::Gelu)),
                Head::GlobalAvgPool,
            ),
            (
                TaskKind::Shapes,
                (stem(&grouped, 2, Downsample::Conv), grouped.clone()),
                Head::GlobalAvgPool,
            ),
            (
                TaskKind::Shapes,
                full(unit(1, 1, Norm::None, relu)),
                Head::GlobalAvgPool,
            ),
            (
                TaskKind::Motif1d,
                (stem(&sequence, 1, Downsample::Conv), sequence.clone()),
                Head::GlobalAvgPool,
            ),
            (
                TaskKind::Shapes,
                (stem(&pooled, 2, Downsample::MaxPool), pooled.clone()),
                Head::Flatten,
            ),
        ];
        let limits = Limits::default();

        layouts
            .into_iter()
            .filter_map(|(task, (first, stage), head)| {
                let spec = task.spec();
                let dims = match task {
                    TaskKind::Motif1d => Dims::OneD,
                    _ => Dims::TwoD,
                };
                let layout = |n: usize| Genome {
                    batch_size: 16,
                    epochs: 5,
                    ..genomes::genome(
                        CnnGenome {
                            dims,
                            stem: first.clone(),
                            stages: vec![stage.clone(); n],
                            head,
                        },
                        Vec::new(),
                        Activation::Relu,
                    )
                };
                let stages = (1..=crate::cnn::genome::MAX_STAGES)
                    .filter(|&n| layout(n).validate(&limits, &spec).is_ok())
                    .max()?;
                let mut genome = layout(stages);

                while genome.epochs < limits.max_epochs {
                    genome.epochs += 1;

                    if genome.validate(&limits, &spec).is_err() {
                        genome.epochs -= 1;
                        break;
                    }
                }

                Some((task, genome))
            })
            .collect()
    }

    /// The time model keeps slow CNNs from the clock: each of `slow_valid_cnns` trains on
    /// the whole Train split within its time budget, so the FLOP limit ends it. This takes
    /// about 1.5 minutes and measures time, so it runs only on request, alone:
    /// `cargo test --lib slow_valid -- --ignored --test-threads 1`.
    #[test]
    #[ignore]
    fn slow_valid_cnns_train_within_the_time_budget() {
        let limits = Limits::default();
        let budget =
            limits.max_train_flops as f64 / crate::cnn::shape::TIME_BUDGET_FLOPS_PER_SECOND;
        let cnns = slow_valid_cnns();

        assert_eq!(cnns.len(), 7);

        for (task, genome) in cnns {
            let spec = task.spec();
            let data = generate(&spec, Split::Train, 1);
            let plan = genome.cnn_plan(&spec).expect("plan");
            let estimate =
                plan.train_nanos_per_sample() * (spec.n_train * genome.epochs) as f64 / 1e9;
            let started = Instant::now();
            let (_, stats) = train(&genome, &spec, &data, &limits).expect("train");
            let seconds = started.elapsed().as_secs_f64();

            eprintln!(
                "{}: {} epochs, estimate {estimate:.1} s, trained in {seconds:.1} s: {}",
                task.name(),
                genome.epochs,
                genome.summary()
            );

            assert!(estimate <= budget);
            assert!(!stats.stopped_by_budget && stats.epochs_done == genome.epochs);
            assert!(seconds < budget, "{seconds} s");
        }
    }

    /// Valid CNNs of the blocks of ticket 03 that come as close to the time budget as
    /// `validate` allows, one per layout that may run slowly per FLOP: narrow basic blocks
    /// at full size, bottlenecks with group norm, separable blocks at full size, inverted
    /// residuals, many small squeeze-and-excitation parts, projections on every stage, wide
    /// gates on few positions, and 1-D blocks. Each takes the most stages that are valid at
    /// 5 epochs, then the most epochs that are valid.
    fn slow_valid_block_cnns() -> Vec<(&'static str, TaskKind, Genome)> {
        use crate::cnn::genome::Block;

        let relu = CnnActivation::Relu;
        let unit = move |block: Block, width: usize, kernel: usize, norm: Norm| Stage {
            norm,
            residual: true,
            ..genomes::stage(block, width, kernel, relu)
        };
        let stem = move |width: usize, stride: usize, norm: Norm| Stem {
            width,
            kernel: 3,
            stride,
            down: Downsample::MaxPool,
            norm,
            activation: relu,
        };
        let gated = |stage: Stage| Stage { se: true, ..stage };
        let widths = |stage: Stage, other: usize| {
            // Stages alternate between the stage's width and `other`, so every stage
            // changes the channels and needs a projection.
            move |n: usize| {
                (0..n)
                    .map(|i| Stage {
                        width: if i % 2 == 0 { stage.width } else { other },
                        ..stage.clone()
                    })
                    .collect::<Vec<_>>()
            }
        };
        let same = |stage: Stage| move |n: usize| vec![stage.clone(); n];

        // The name, task, stem, stages for a count and head of each layout.
        type Stages = Box<dyn Fn(usize) -> Vec<Stage>>;
        let layouts: Vec<(&'static str, TaskKind, Stem, Stages)> = vec![
            (
                "narrow basic blocks at 16x16",
                TaskKind::Shapes,
                stem(4, 1, Norm::Batch),
                Box::new(same(unit(Block::Basic, 4, 3, Norm::Batch))),
            ),
            (
                "basic blocks with squeeze-and-excitation at 8x8",
                TaskKind::Shapes,
                stem(8, 2, Norm::Batch),
                Box::new(same(gated(unit(Block::Basic, 8, 3, Norm::Batch)))),
            ),
            (
                "bottlenecks with group norm at 8x8",
                TaskKind::Shapes,
                stem(16, 2, Norm::Group),
                Box::new(same(unit(Block::Bottleneck, 16, 3, Norm::Group))),
            ),
            (
                "separable blocks at 16x16",
                TaskKind::Shapes,
                stem(8, 1, Norm::Batch),
                Box::new(same(unit(Block::Separable, 8, 3, Norm::Batch))),
            ),
            (
                "wide separable blocks with layer norm at 4x4",
                TaskKind::Shapes,
                stem(16, 4, Norm::Layer),
                Box::new(same(unit(Block::Separable, 64, 3, Norm::Layer))),
            ),
            (
                "inverted residuals at 8x8",
                TaskKind::Shapes,
                stem(4, 2, Norm::Batch),
                Box::new(same(unit(Block::InvertedResidual, 4, 3, Norm::Batch))),
            ),
            (
                "inverted residuals at 4x4",
                TaskKind::Shapes,
                stem(16, 4, Norm::Batch),
                Box::new(same(unit(Block::InvertedResidual, 16, 3, Norm::Batch))),
            ),
            (
                "tiny plain stages with both options at 1x1",
                TaskKind::Shapes,
                stem(4, 4, Norm::None),
                Box::new(move |n: usize| {
                    let mut stages = vec![
                        Stage {
                            stride: 4,
                            down: Downsample::MaxPool,
                            ..gated(unit(Block::Plain, 4, 1, Norm::None))
                        };
                        1
                    ];

                    stages.extend(vec![gated(unit(Block::Plain, 4, 1, Norm::None)); n - 1]);

                    stages
                }),
            ),
            (
                "projections on every stage at 8x8",
                TaskKind::Shapes,
                stem(8, 2, Norm::Batch),
                Box::new(widths(unit(Block::Plain, 12, 1, Norm::Batch), 8)),
            ),
            (
                "wide gates on 2x2",
                TaskKind::Shapes,
                stem(16, 4, Norm::Batch),
                Box::new(move |n: usize| {
                    let mut stages = vec![Stage {
                        stride: 2,
                        down: Downsample::MaxPool,
                        ..gated(unit(Block::Plain, 64, 1, Norm::Batch))
                    }];

                    stages.extend(vec![gated(unit(Block::Plain, 64, 1, Norm::Batch)); n - 1]);

                    stages
                }),
            ),
            (
                "1-D basic blocks with squeeze-and-excitation",
                TaskKind::Motif1d,
                Stem {
                    kernel: 5,
                    ..stem(8, 1, Norm::Batch)
                },
                Box::new(same(gated(unit(Block::Basic, 8, 3, Norm::Batch)))),
            ),
            (
                "1-D inverted residuals",
                TaskKind::Motif1d,
                stem(4, 1, Norm::Batch),
                Box::new(same(unit(Block::InvertedResidual, 4, 3, Norm::Batch))),
            ),
        ];
        let limits = Limits::default();

        layouts
            .into_iter()
            .filter_map(|(name, task, first, stages)| {
                let spec = task.spec();
                let dims = match task {
                    TaskKind::Motif1d => Dims::OneD,
                    _ => Dims::TwoD,
                };
                let layout = |n: usize| Genome {
                    batch_size: 16,
                    epochs: 5,
                    ..genomes::genome(
                        CnnGenome {
                            dims,
                            stem: first.clone(),
                            stages: stages(n),
                            head: Head::GlobalAvgPool,
                        },
                        Vec::new(),
                        Activation::Relu,
                    )
                };
                let n = (1..=crate::cnn::genome::MAX_STAGES)
                    .filter(|&n| layout(n).validate(&limits, &spec).is_ok())
                    .max()?;
                let mut genome = layout(n);

                while genome.epochs < limits.max_epochs {
                    genome.epochs += 1;

                    if genome.validate(&limits, &spec).is_err() {
                        genome.epochs -= 1;
                        break;
                    }
                }

                Some((name, task, genome))
            })
            .collect()
    }

    /// The time model keeps the blocks of ticket 03 from the clock: each of
    /// `slow_valid_block_cnns` trains on the whole Train split within its time budget, and
    /// its scoring of the Hidden split stays within the scoring estimate. It prints each
    /// measured time as a share of its estimate. This takes about four minutes and measures
    /// time, so it runs only on request, alone, at one thread:
    /// `RAYON_NUM_THREADS=1 cargo test --lib slow_valid_block -- --ignored --test-threads 1`.
    #[test]
    #[ignore]
    fn slow_valid_block_cnns_train_within_the_time_budget() {
        let cnns = slow_valid_block_cnns();

        for (name, task, genome) in &cnns {
            eprintln!("LAYOUT {name} on {}: {}", task.name(), genome.summary());
        }

        assert_eq!(cnns.len(), 12);

        time_layouts(cnns);
    }

    /// Train each CNN on the whole Train split of its task and score the Hidden split, and
    /// require both within the estimates of the time model; print each time as a share of
    /// its estimate.
    fn time_layouts(cnns: Vec<(&'static str, TaskKind, Genome)>) {
        let limits = Limits::default();
        let budget =
            limits.max_train_flops as f64 / crate::cnn::shape::TIME_BUDGET_FLOPS_PER_SECOND;

        for (name, task, genome) in cnns {
            let spec = task.spec();
            let data = generate(&spec, Split::Train, 1);
            let hidden = generate(&spec, Split::Hidden, 1);
            let plan = genome.cnn_plan(&spec).expect("plan");
            let estimate =
                plan.train_nanos_per_sample() * (spec.n_train * genome.epochs) as f64 / 1e9;
            let started = Instant::now();
            let (model, stats) = train(&genome, &spec, &data, &limits).expect("train");
            let seconds = started.elapsed().as_secs_f64();
            let scoring = plan.eval_nanos(spec.n_hidden, score::cnn_chunk_rows(&plan)) / 1e9;
            let started = Instant::now();
            let primary = score::evaluate(&model, &spec, &hidden).primary;
            let scored = started.elapsed().as_secs_f64();

            eprintln!(
                "TIMED {name}: {} layers, {} epochs, train {seconds:.1} s of {estimate:.1} s \
                 ({:.2}), score {scored:.2} s of {scoring:.2} s ({:.2}), primary score \
                 {primary:.2}",
                plan.steps.len(),
                genome.epochs,
                seconds / estimate,
                scored / scoring
            );

            assert!(estimate <= budget);
            assert!(!stats.stopped_by_budget && stats.epochs_done == genome.epochs);
            assert!(seconds < budget, "{name}: {seconds} s");
            assert!(scored < scoring, "{name}: {scored} s");
        }
    }

    /// Valid CNNs of the blocks of ticket 04 that come as close to the time budget as
    /// `validate` allows, one per layout that may run slowly per FLOP: dense layers at full
    /// size and with a large growth rate, Inception blocks of 1 and 2 channels per branch,
    /// Fire modules, ShuffleNet units with 2 and 4 groups, ConvNeXt blocks at 8x8 and wide
    /// ones at 4x4, TCN blocks with growing dilations, U-Nets with transposed convolutions
    /// and with nearest upsampling, wide transposed convolutions, the per-cell head on the
    /// mirror grid, and 1-D decoders. Each takes the most stages that are valid at 5 epochs,
    /// then the most epochs that are valid.
    fn slow_valid_family_b_cnns() -> Vec<(&'static str, TaskKind, Genome)> {
        use crate::cnn::genome::Block;

        let relu = CnnActivation::Relu;
        let unit = move |block: Block, width: usize, kernel: usize, norm: Norm| Stage {
            norm,
            ..genomes::stage(block, width, kernel, relu)
        };
        let stem = move |width: usize, stride: usize, norm: Norm| Stem {
            width,
            kernel: 3,
            stride,
            down: Downsample::MaxPool,
            norm,
            activation: relu,
        };
        let residual = |stage: Stage| Stage {
            residual: true,
            ..stage
        };
        let same = |stage: Stage| move |n: usize| vec![stage.clone(); n];
        // A U-Net of n stages: n / 2 encoder stages that pool by 2, the rest at the bottom,
        // then n / 2 decoders. `bottom` is the stage at the lowest resolution.
        let unet = move |encoder: Stage, bottom: Stage, decoder: Stage| {
            move |n: usize| {
                let levels = n / 2;
                let mut stages = vec![
                    Stage {
                        stride: 2,
                        down: Downsample::MaxPool,
                        ..encoder.clone()
                    };
                    levels
                ];

                stages.extend(vec![bottom.clone(); n - 2 * levels]);
                stages.extend(vec![
                    Stage {
                        stride: 2,
                        ..decoder.clone()
                    };
                    levels
                ]);

                stages
            }
        };
        let nearest = |stage: Stage| Stage {
            down: Downsample::MaxPool,
            ..stage
        };

        // The name, task, stem, stages for a count, and head of each layout.
        type Stages = Box<dyn Fn(usize) -> Vec<Stage>>;
        let layouts: Vec<(&'static str, TaskKind, Stem, Stages, Head)> = vec![
            (
                "dense layers of growth 4 at 16x16",
                TaskKind::Shapes,
                stem(4, 1, Norm::Batch),
                Box::new(same(unit(Block::Dense, 4, 3, Norm::Batch))),
                Head::GlobalAvgPool,
            ),
            (
                "dense layers of growth 16 at 8x8",
                TaskKind::Shapes,
                stem(8, 2, Norm::Batch),
                Box::new(same(unit(Block::Dense, 16, 3, Norm::Batch))),
                Head::GlobalAvgPool,
            ),
            (
                "inception blocks of 1 channel per branch at 16x16",
                TaskKind::Shapes,
                stem(4, 1, Norm::Batch),
                Box::new(same(residual(unit(Block::Inception, 4, 3, Norm::Batch)))),
                Head::GlobalAvgPool,
            ),
            (
                "inception blocks with groups at 8x8",
                TaskKind::Shapes,
                stem(8, 2, Norm::Batch),
                Box::new(same(Stage {
                    groups: 2,
                    ..residual(unit(Block::Inception, 8, 3, Norm::Batch))
                })),
                Head::GlobalAvgPool,
            ),
            (
                "fire modules at 8x8",
                TaskKind::Shapes,
                stem(8, 2, Norm::Batch),
                Box::new(same(residual(unit(Block::Fire, 8, 3, Norm::Batch)))),
                Head::GlobalAvgPool,
            ),
            (
                "shufflenet units with 2 groups at 16x16",
                TaskKind::Shapes,
                stem(8, 1, Norm::Batch),
                Box::new(same(Stage {
                    groups: 2,
                    ..residual(unit(Block::Shuffle, 8, 3, Norm::Batch))
                })),
                Head::GlobalAvgPool,
            ),
            (
                "shufflenet units with 4 groups at 8x8",
                TaskKind::Shapes,
                stem(16, 2, Norm::Batch),
                Box::new(same(Stage {
                    groups: 4,
                    ..residual(unit(Block::Shuffle, 16, 3, Norm::Batch))
                })),
                Head::GlobalAvgPool,
            ),
            (
                "convnext blocks of 7x7 at 8x8",
                TaskKind::Shapes,
                stem(8, 2, Norm::Layer),
                Box::new(same(Stage {
                    activation: CnnActivation::Gelu,
                    ..residual(unit(Block::ConvNext, 8, 7, Norm::Layer))
                })),
                Head::GlobalAvgPool,
            ),
            (
                "wide convnext blocks at 4x4",
                TaskKind::Shapes,
                stem(16, 4, Norm::Layer),
                Box::new(same(Stage {
                    activation: CnnActivation::Gelu,
                    ..residual(unit(Block::ConvNext, 64, 3, Norm::Layer))
                })),
                Head::GlobalAvgPool,
            ),
            (
                "tcn blocks with dilations 1 to 16",
                TaskKind::Motif1d,
                Stem {
                    kernel: 5,
                    ..stem(8, 1, Norm::Batch)
                },
                Box::new(move |n: usize| {
                    (0..n)
                        .map(|i| Stage {
                            dilation: 1 << (i % 5),
                            ..residual(unit(Block::Tcn, 8, 3, Norm::Batch))
                        })
                        .collect()
                }),
                Head::GlobalAvgPool,
            ),
            (
                "u-nets with transposed convolutions from 8x8",
                TaskKind::Shapes,
                stem(8, 2, Norm::Batch),
                Box::new(unet(
                    unit(Block::Plain, 8, 3, Norm::Batch),
                    unit(Block::Plain, 8, 1, Norm::Batch),
                    unit(Block::Decoder, 8, 3, Norm::Batch),
                )),
                Head::GlobalAvgPool,
            ),
            (
                "u-nets with nearest upsampling from 8x8",
                TaskKind::Shapes,
                stem(8, 2, Norm::Batch),
                Box::new(unet(
                    unit(Block::Plain, 8, 3, Norm::Batch),
                    unit(Block::Plain, 8, 1, Norm::Batch),
                    nearest(unit(Block::Decoder, 8, 3, Norm::Batch)),
                )),
                Head::GlobalAvgPool,
            ),
            (
                "wide transposed convolutions from 4x4",
                TaskKind::Shapes,
                stem(16, 4, Norm::Batch),
                Box::new(unet(
                    unit(Block::Plain, 64, 1, Norm::Batch),
                    unit(Block::Plain, 64, 1, Norm::Batch),
                    unit(Block::Decoder, 64, 1, Norm::Batch),
                )),
                Head::GlobalAvgPool,
            ),
            (
                "u-nets with the per-cell head on the mirror grid",
                TaskKind::MirrorGrid,
                stem(8, 1, Norm::Batch),
                Box::new(unet(
                    unit(Block::Plain, 8, 3, Norm::Batch),
                    unit(Block::Plain, 8, 1, Norm::Batch),
                    unit(Block::Decoder, 8, 3, Norm::Batch),
                )),
                Head::PerCell,
            ),
            (
                "1-D u-nets with transposed convolutions",
                TaskKind::Motif1d,
                stem(8, 1, Norm::Batch),
                Box::new(unet(
                    unit(Block::Plain, 8, 3, Norm::Batch),
                    unit(Block::Plain, 8, 3, Norm::Batch),
                    unit(Block::Decoder, 8, 3, Norm::Batch),
                )),
                Head::GlobalAvgPool,
            ),
        ];
        let limits = Limits::default();

        layouts
            .into_iter()
            .filter_map(|(name, task, first, stages, head)| {
                let spec = task.spec();
                let dims = match task {
                    TaskKind::Motif1d => Dims::OneD,
                    _ => Dims::TwoD,
                };
                let layout = |n: usize| Genome {
                    batch_size: 16,
                    epochs: 5,
                    ..genomes::genome(
                        CnnGenome {
                            dims,
                            stem: first.clone(),
                            stages: stages(n),
                            head,
                        },
                        Vec::new(),
                        Activation::Relu,
                    )
                };
                let n = (1..=crate::cnn::genome::MAX_STAGES)
                    .filter(|&n| layout(n).validate(&limits, &spec).is_ok())
                    .max()?;
                let mut genome = layout(n);

                while genome.epochs < limits.max_epochs {
                    genome.epochs += 1;

                    if genome.validate(&limits, &spec).is_err() {
                        genome.epochs -= 1;
                        break;
                    }
                }

                Some((name, task, genome))
            })
            .collect()
    }

    /// The time model keeps the blocks of ticket 04 from the clock: each of
    /// `slow_valid_family_b_cnns` trains on the whole Train split within its time budget,
    /// and its scoring of the Hidden split stays within the scoring estimate. It prints each
    /// measured time as a share of its estimate. This takes about four minutes and measures
    /// time, so it runs only on request, alone, at one thread:
    /// `RAYON_NUM_THREADS=1 cargo test --lib slow_valid_family_b -- --ignored --test-threads 1`.
    #[test]
    #[ignore]
    fn slow_valid_family_b_cnns_train_within_the_time_budget() {
        let cnns = slow_valid_family_b_cnns();

        for (name, task, genome) in &cnns {
            eprintln!("LAYOUT {name} on {}: {}", task.name(), genome.summary());
        }

        assert_eq!(cnns.len(), 15);

        time_layouts(cnns);
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
