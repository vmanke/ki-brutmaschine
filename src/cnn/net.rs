//! The network of a CNN genome, built from its plan through `ops`.
//!
//! Each layer of the plan (`shape::Plan`) becomes one unit, in the same order, so the
//! network has exactly the parameters and shapes that the shape check counted.
//!
//! The trainable values sit in two lists, as an MLP's do (`nn::Mlp`). `weights` holds the
//! kernels of the convolutions and the matrices of the dense layers; they get weight decay.
//! `biases` holds the biases, and the scales and shifts of the norms; they do not. Both
//! lists keep the plan order, so the optimizer of the MLP (`nn::OptimizerState`) trains a
//! CNN with the same formulas.

use rand_chacha::ChaCha8Rng;

use crate::cnn::genome::{CnnActivation, GROUP_NORM_GROUPS, Norm};
use crate::cnn::ops::{self, BatchNorm, Param, Tensor};
use crate::cnn::shape::{Layer, Plan, Pool};
use crate::genome::Activation;
use crate::util::{BrutError, Result, normal_f32};

/// The epsilon of every norm: PyTorch's default for batch, layer and group norm.
pub const NORM_EPS: f64 = 1e-5;

/// How a tensor starts before training.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Init {
    /// Normal draws with this standard deviation, before `Genome::init_scale`.
    Normal(f32),
    /// Every value 0: biases and shifts.
    Zeros,
    /// Every value 1: the scales of norms.
    Ones,
}

/// One trainable tensor of a network.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ParamSpec {
    /// The length of each axis; the values are stored in row-major order.
    pub(crate) shape: Vec<usize>,
    /// True for a kernel or matrix (in `weights`), false for a bias, scale or shift.
    pub(crate) decay: bool,
    pub(crate) init: Init,
}

impl ParamSpec {
    /// The number of values.
    pub(crate) fn len(&self) -> usize {
        self.shape.iter().product()
    }
}

/// The trainable tensors of `plan`, in plan order.
///
/// A convolution has its kernel of shape (outputs, inputs / groups, k, k), or (outputs,
/// inputs / groups, k) along a sequence, then its bias if it has one. A norm has its scale,
/// then its shift. A dense layer has its matrix (outputs, inputs), then its bias.
///
/// A kernel or matrix starts as in `nn::Mlp::new`, by the activation that follows it in the
/// plan: He init, std sqrt(2 / fan_in), for ReLU, GELU, SiLU and ReLU6; Xavier init, std
/// sqrt(2 / (fan_in + fan_out)), for tanh and sine. For a convolution the fans count one
/// group: fan_in is the inputs per group times the taps, fan_out the outputs per group
/// times the taps. The output layer has no activation after it; like the output layer of
/// an MLP, it follows `activation`, the activation of the head. Biases and shifts start at
/// 0, scales at 1.
pub(crate) fn param_specs(plan: &Plan, activation: Activation) -> Vec<ParamSpec> {
    let axes = plan.input.size.len();
    let mut specs = Vec::new();

    for (i, step) in plan.steps.iter().enumerate() {
        let next = next_activation(plan, i).unwrap_or(activation.into());

        match step.layer {
            Layer::Conv {
                inputs,
                outputs,
                kernel,
                groups,
                bias,
                ..
            } => {
                let taps = kernel.pow(axes as u32);
                let mut shape = vec![outputs, inputs / groups];

                shape.extend(std::iter::repeat_n(kernel, axes));

                let fan_in = inputs / groups * taps;
                let fan_out = outputs / groups * taps;

                specs.push(weight(shape, next, fan_in, fan_out));

                if bias {
                    specs.push(constant(outputs, Init::Zeros));
                }
            }
            Layer::Norm { channels, .. } => {
                specs.push(constant(channels, Init::Ones));
                specs.push(constant(channels, Init::Zeros));
            }
            Layer::Dense { inputs, outputs } => {
                specs.push(weight(vec![outputs, inputs], next, inputs, outputs));
                specs.push(constant(outputs, Init::Zeros));
            }
            Layer::Activation(_) | Layer::Pool { .. } | Layer::GlobalAvgPool | Layer::Flatten => {}
        }
    }

    specs
}

/// The channels of each batch norm of `plan`, in plan order. A batch norm keeps a running
/// mean and a running variance of that length.
pub(crate) fn batch_norm_channels(plan: &Plan) -> Vec<usize> {
    plan.steps
        .iter()
        .filter_map(|step| match step.layer {
            Layer::Norm {
                norm: Norm::Batch,
                channels,
            } => Some(channels),
            _ => None,
        })
        .collect()
}

/// The first activation after step `i` of the plan, if any.
fn next_activation(plan: &Plan, i: usize) -> Option<CnnActivation> {
    plan.steps[i + 1..]
        .iter()
        .find_map(|step| match step.layer {
            Layer::Activation(activation) => Some(activation),
            _ => None,
        })
}

/// A kernel or matrix with He or Xavier init for the activation that follows it.
fn weight(
    shape: Vec<usize>,
    activation: CnnActivation,
    fan_in: usize,
    fan_out: usize,
) -> ParamSpec {
    let std = match activation {
        CnnActivation::Relu | CnnActivation::Gelu | CnnActivation::Silu | CnnActivation::Relu6 => {
            (2.0 / fan_in.max(1) as f32).sqrt()
        }
        CnnActivation::Tanh | CnnActivation::Sine => {
            (2.0 / (fan_in + fan_out).max(1) as f32).sqrt()
        }
    };

    ParamSpec {
        shape,
        decay: true,
        init: Init::Normal(std),
    }
}

/// A vector of `len` values that starts at a constant.
fn constant(len: usize, init: Init) -> ParamSpec {
    ParamSpec {
        shape: vec![len],
        decay: false,
        init,
    }
}

/// The values of a network: the two lists of the module doc, and the running statistics.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Values {
    pub(crate) weights: Vec<Vec<f32>>,
    pub(crate) biases: Vec<Vec<f32>>,
    /// The running mean and variance of each batch norm, in plan order.
    pub(crate) running: Vec<(Vec<f32>, Vec<f32>)>,
}

impl Values {
    /// The values before training. Kernels and matrices are drawn from `rng` in plan order,
    /// each value as `std * init_scale` times a standard normal draw, as in `nn::Mlp::new`
    /// (W8: no value comes from candle's generator). A running mean starts at 0, a running
    /// variance at 1.
    pub(crate) fn initial(
        specs: &[ParamSpec],
        batch_norm_channels: &[usize],
        init_scale: f32,
        rng: &mut ChaCha8Rng,
    ) -> Values {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for spec in specs {
            let values: Vec<f32> = match spec.init {
                Init::Normal(std) => {
                    let scale = std * init_scale;

                    (0..spec.len()).map(|_| scale * normal_f32(rng)).collect()
                }
                Init::Zeros => vec![0.0; spec.len()],
                Init::Ones => vec![1.0; spec.len()],
            };

            if spec.decay {
                weights.push(values);
            } else {
                biases.push(values);
            }
        }

        let running = batch_norm_channels
            .iter()
            .map(|&channels| (vec![0.0; channels], vec![1.0; channels]))
            .collect();

        Values {
            weights,
            biases,
            running,
        }
    }

    /// The values in plan order, as `specs` lists them (the inverse of `from_plan_order`).
    pub(crate) fn plan_order(&self, specs: &[ParamSpec]) -> Vec<Vec<f32>> {
        let mut weights = self.weights.iter();
        let mut biases = self.biases.iter();

        specs
            .iter()
            .filter_map(|spec| {
                if spec.decay {
                    weights.next()
                } else {
                    biases.next()
                }
            })
            .cloned()
            .collect()
    }

    /// Split values given in plan order into the two lists, by `specs`.
    pub(crate) fn from_plan_order(
        specs: &[ParamSpec],
        params: &[Vec<f32>],
        running: Vec<(Vec<f32>, Vec<f32>)>,
    ) -> Values {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for (spec, values) in specs.iter().zip(params) {
            if spec.decay {
                weights.push(values.clone());
            } else {
                biases.push(values.clone());
            }
        }

        Values {
            weights,
            biases,
            running,
        }
    }

    /// True if every value is finite.
    pub(crate) fn is_finite(&self) -> bool {
        let finite = |list: &[Vec<f32>]| list.iter().flatten().all(|v| v.is_finite());
        let running = self
            .running
            .iter()
            .all(|(mean, var)| mean.iter().chain(var).all(|v| v.is_finite()));

        finite(&self.weights) && finite(&self.biases) && running
    }
}

/// One layer of the network: a step of the plan, with the places of its tensors.
#[derive(Clone, Debug)]
enum Unit {
    Conv {
        /// Index into `weights`.
        kernel: usize,
        /// Index into `biases`.
        bias: Option<usize>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    },
    Norm {
        norm: Norm,
        /// Indices into `biases`.
        scale: usize,
        shift: usize,
        /// Index into the running statistics, for a batch norm.
        batch: Option<usize>,
    },
    Activation(CnnActivation),
    Pool {
        pool: Pool,
        window: usize,
    },
    GlobalAvgPool,
    Flatten,
    Dense {
        /// Index into `weights`.
        matrix: usize,
        /// Index into `biases`.
        bias: usize,
    },
}

/// A CNN as candle tensors: the units of its plan and their values.
pub(crate) struct Net {
    /// The axes of one input sample: the channels, then the length of a sequence or the
    /// height and width of an image.
    sample: Vec<usize>,
    units: Vec<Unit>,
    weights: Vec<Param>,
    biases: Vec<Param>,
    /// The running mean and variance of each batch norm, in plan order.
    running: Vec<(Tensor, Tensor)>,
}

impl Net {
    /// The network of `plan` with `values`, whose tensors have the shapes of `specs`.
    ///
    /// Errors if a tensor does not fit its shape, or if the lists are too short or too long.
    pub(crate) fn new(plan: &Plan, specs: &[ParamSpec], values: &Values) -> Result<Net> {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for spec in specs {
            let (list, into) = if spec.decay {
                (&values.weights, &mut weights)
            } else {
                (&values.biases, &mut biases)
            };
            let Some(given) = list.get(into.len()) else {
                return Err(mismatch("there are fewer tensors than the plan has"));
            };

            into.push(Param::new(&Tensor::from_f32(given.clone(), &spec.shape)?)?);
        }

        if weights.len() != values.weights.len() || biases.len() != values.biases.len() {
            return Err(mismatch("there are more tensors than the plan has"));
        }

        if values.running.len() != batch_norm_channels(plan).len() {
            return Err(mismatch(
                "the running statistics do not match the batch norms",
            ));
        }

        let mut sample = vec![plan.input.channels];

        sample.extend_from_slice(&plan.input.size);

        let mut net = Net {
            sample,
            units: Vec::with_capacity(plan.steps.len()),
            weights,
            biases,
            running: Vec::with_capacity(values.running.len()),
        };
        let mut next_weight = 0;
        let mut next_bias = 0;
        let mut take = |decay: bool| {
            let counter = if decay {
                &mut next_weight
            } else {
                &mut next_bias
            };

            *counter += 1;

            *counter - 1
        };

        for step in &plan.steps {
            let unit = match step.layer {
                Layer::Conv {
                    stride,
                    padding,
                    dilation,
                    groups,
                    bias,
                    ..
                } => Unit::Conv {
                    kernel: take(true),
                    bias: bias.then(|| take(false)),
                    stride,
                    padding,
                    dilation,
                    groups,
                },
                Layer::Norm { norm, channels } => {
                    let batch = if norm == Norm::Batch {
                        let (mean, var) = &values.running[net.running.len()];

                        net.running.push((
                            Tensor::from_f32(mean.clone(), &[channels])?,
                            Tensor::from_f32(var.clone(), &[channels])?,
                        ));

                        Some(net.running.len() - 1)
                    } else {
                        None
                    };

                    Unit::Norm {
                        norm,
                        scale: take(false),
                        shift: take(false),
                        batch,
                    }
                }
                Layer::Activation(activation) => Unit::Activation(activation),
                Layer::Pool { pool, window } => Unit::Pool { pool, window },
                Layer::GlobalAvgPool => Unit::GlobalAvgPool,
                Layer::Flatten => Unit::Flatten,
                Layer::Dense { .. } => Unit::Dense {
                    matrix: take(true),
                    bias: take(false),
                },
            };

            net.units.push(unit);
        }

        // Every tensor belongs to one layer, so the indices above stay inside the lists.
        if next_weight != net.weights.len() || next_bias != net.biases.len() {
            return Err(mismatch("the tensors do not match the layers of the plan"));
        }

        Ok(net)
    }

    /// The axes of one input sample (see the field).
    pub(crate) fn sample(&self) -> &[usize] {
        &self.sample
    }

    /// The parameters of the kernels and matrices, in plan order.
    pub(crate) fn weights(&self) -> &[Param] {
        &self.weights
    }

    /// The parameters of the biases, scales and shifts, in plan order.
    pub(crate) fn biases(&self) -> &[Param] {
        &self.biases
    }

    /// Overwrite every parameter with the matching list of `values`.
    pub(crate) fn set(&self, weights: &[Vec<f32>], biases: &[Vec<f32>]) -> Result<()> {
        if weights.len() != self.weights.len() || biases.len() != self.biases.len() {
            return Err(mismatch("a new value list has the wrong length"));
        }

        for (params, lists) in [(&self.weights, weights), (&self.biases, biases)] {
            for (param, values) in params.iter().zip(lists) {
                param.set(&Tensor::from_f32(values.clone(), param.tensor().dims())?)?;
            }
        }

        Ok(())
    }

    /// The running mean and variance of each batch norm, in plan order.
    pub(crate) fn running(&self) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        self.running
            .iter()
            .map(|(mean, var)| Ok((mean.to_vec_f32()?, var.to_vec_f32()?)))
            .collect()
    }

    /// The forward pass in training mode: each batch norm normalizes with the statistics of
    /// the batch and moves its running statistics towards them.
    pub(crate) fn forward_train(&mut self, x: &Tensor) -> Result<Tensor> {
        let mut running = std::mem::take(&mut self.running);
        let out = self.forward(x, Some(&mut running));

        self.running = running;

        out
    }

    /// The forward pass in evaluation mode: each batch norm uses its running statistics.
    pub(crate) fn forward_eval(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x, None)
    }

    /// The forward pass of `x` (rows, then the axes of `sample`) to the outputs (rows,
    /// n_out). With `train`, the running statistics there are updated.
    fn forward(&self, x: &Tensor, mut train: Option<&mut Vec<(Tensor, Tensor)>>) -> Result<Tensor> {
        let one_d = self.sample.len() == 2;
        let mut h = x.clone();

        for unit in &self.units {
            h = match unit {
                Unit::Conv {
                    kernel,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let kernel = self.weights[*kernel].tensor();
                    let out = if one_d {
                        ops::conv1d(&h, &kernel, *padding, *stride, *dilation, *groups)?
                    } else {
                        ops::conv2d(&h, &kernel, *padding, *stride, *dilation, *groups)?
                    };

                    match bias {
                        Some(bias) => {
                            let bias = self.biases[*bias].tensor();
                            let mut shape = vec![1; out.dims().len()];

                            shape[1] = bias.dims()[0];

                            ops::broadcast_add(&out, &ops::reshape(&bias, &shape)?)?
                        }
                        None => out,
                    }
                }
                Unit::Norm {
                    norm,
                    scale,
                    shift,
                    batch,
                } => {
                    let gamma = self.biases[*scale].tensor();
                    let beta = self.biases[*shift].tensor();

                    match (norm, batch) {
                        (Norm::Batch, Some(index)) => match train.as_deref_mut() {
                            Some(running) => {
                                let (mean, var) = &running[*index];
                                let norm = BatchNorm::new(&gamma, &beta, mean, var, NORM_EPS)?;
                                let out = norm.forward_train(&h)?;

                                running[*index] = (norm.running_mean(), norm.running_var());

                                out
                            }
                            None => {
                                let (mean, var) = &self.running[*index];

                                BatchNorm::new(&gamma, &beta, mean, var, NORM_EPS)?
                                    .forward_eval(&h)?
                            }
                        },
                        (Norm::Layer, _) => ops::layer_norm_channels(&h, &gamma, &beta, NORM_EPS)?,
                        (Norm::Group, _) => {
                            ops::group_norm(&h, &gamma, &beta, GROUP_NORM_GROUPS, NORM_EPS)?
                        }
                        (Norm::None | Norm::Batch, _) => {
                            return Err(mismatch("a norm unit has no kind or no statistics"));
                        }
                    }
                }
                Unit::Activation(activation) => activate(*activation, &h)?,
                Unit::Pool { pool, window } => match (one_d, pool) {
                    (true, Pool::Max) => ops::max_pool1d(&h, *window, *window)?,
                    (true, Pool::Avg) => ops::avg_pool1d(&h, *window, *window)?,
                    (false, Pool::Max) => ops::max_pool2d(&h, *window, *window, 0)?,
                    (false, Pool::Avg) => ops::avg_pool2d(&h, *window, *window, 0)?,
                },
                Unit::GlobalAvgPool => ops::global_avg_pool(&h)?,
                Unit::Flatten => ops::flatten(&h)?,
                Unit::Dense { matrix, bias } => ops::linear(
                    &h,
                    &self.weights[*matrix].tensor(),
                    &self.biases[*bias].tensor(),
                )?,
            };
        }

        Ok(h)
    }
}

/// An activation of the stem, a stage or the head.
fn activate(activation: CnnActivation, x: &Tensor) -> Result<Tensor> {
    match activation {
        CnnActivation::Relu => ops::relu(x),
        CnnActivation::Tanh => ops::tanh(x),
        // The tanh form, as in the MLP (W7).
        CnnActivation::Gelu => ops::gelu_tanh(x),
        CnnActivation::Sine => ops::sin(x),
        CnnActivation::Silu => ops::silu(x),
        CnnActivation::Relu6 => ops::relu6(x),
    }
}

/// An error for values that do not fit the plan.
fn mismatch(what: &str) -> BrutError {
    BrutError::Invalid(format!("cnn model: {what}"))
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::cnn::genome::testing;
    use crate::genome::Genome;
    use crate::task::{TaskKind, TaskSpec};

    /// The hand-made genomes of `cnn::genome::testing` with a task they fit.
    fn genomes() -> Vec<(Genome, TaskSpec)> {
        vec![
            (testing::lenet(), TaskKind::Shapes.spec()),
            (testing::vgg(), TaskKind::BlobCount.spec()),
            (testing::grouped_dilated(), TaskKind::MirrorSymmetry.spec()),
            (testing::sequence(), TaskKind::Motif1d.spec()),
        ]
    }

    fn initial(genome: &Genome, plan: &Plan, seed: u64) -> (Vec<ParamSpec>, Values) {
        let specs = param_specs(plan, genome.activation);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let values = Values::initial(
            &specs,
            &batch_norm_channels(plan),
            genome.init_scale,
            &mut rng,
        );

        (specs, values)
    }

    #[test]
    fn the_tensors_hold_exactly_the_parameters_of_the_plan() {
        for (genome, spec) in genomes() {
            let plan = genome.cnn_plan(&spec).expect("plan");
            let specs = param_specs(&plan, genome.activation);
            let mut next = specs.iter();

            // Step by step: a convolution has a kernel and maybe a bias, a norm a scale and
            // a shift, a dense layer a matrix and a bias.
            for step in &plan.steps {
                let tensors = match step.layer {
                    Layer::Conv { bias, .. } => 1 + usize::from(bias),
                    Layer::Norm { .. } | Layer::Dense { .. } => 2,
                    _ => 0,
                };
                let values: usize = next.by_ref().take(tensors).map(ParamSpec::len).sum();

                assert_eq!(
                    values,
                    step.params,
                    "{}: {:?}",
                    genome.summary(),
                    step.layer
                );
            }

            assert!(next.next().is_none());
            assert_eq!(
                specs.iter().map(ParamSpec::len).sum::<usize>(),
                genome.param_count(&spec)
            );
        }
    }

    #[test]
    fn initial_values_come_from_the_seed_with_he_or_xavier_scale() {
        let genome = testing::grouped_dilated();
        let spec = TaskKind::MirrorSymmetry.spec();
        let plan = genome.cnn_plan(&spec).expect("plan");
        let (specs, a) = initial(&genome, &plan, 1);
        let (_, b) = initial(&genome, &plan, 1);
        let (_, c) = initial(&genome, &plan, 2);

        assert_eq!(a, b);
        assert_ne!(a.weights, c.weights);
        assert_eq!(
            a.biases, c.biases,
            "biases, scales and shifts are constants"
        );

        // The stem (8 x 1 x 3 x 3, then ReLU) is He-initialized: std sqrt(2 / 9).
        assert_eq!(specs[0].shape, vec![8, 1, 3, 3]);
        assert_eq!(specs[0].init, Init::Normal((2.0f32 / 9.0).sqrt()));

        // Group convolution 16 x 2 x 3 x 3 before SiLU (He): fan_in 2 * 9.
        assert_eq!(specs[2].shape, vec![16, 2, 3, 3]);
        assert_eq!(specs[2].init, Init::Normal((2.0f32 / 18.0).sqrt()));

        // The output layer has no activation; it follows the head's GELU (He): fan_in 8.
        let last = specs.iter().rposition(|s| s.decay).expect("a weight");

        assert_eq!(specs[last].shape, vec![2, 8]);
        assert_eq!(specs[last].init, Init::Normal((2.0f32 / 8.0).sqrt()));

        // Xavier for tanh: the LeNet stem 6 x 1 x 5 x 5, fan_in 25 and fan_out 150.
        let lenet = testing::lenet();
        let plan = lenet.cnn_plan(&TaskKind::Shapes.spec()).expect("plan");
        let specs = param_specs(&plan, lenet.activation);

        assert_eq!(specs[0].init, Init::Normal((2.0f32 / 175.0).sqrt()));

        // The drawn values have about that spread, times init_scale.
        let mut scaled = lenet.clone();

        scaled.init_scale = 2.0;

        let (_, values) = initial(&scaled, &plan, 3);
        let stem = &values.weights[1];
        let rms = (stem.iter().map(|v| v * v).sum::<f32>() / stem.len() as f32).sqrt();
        let Init::Normal(std) = specs[2].init else {
            panic!("the second weight is drawn");
        };

        assert_eq!(specs[2].shape, vec![16, 6, 5, 5]);
        assert!(
            (rms / (2.0 * std) - 1.0).abs() < 0.1,
            "{rms} vs {}",
            2.0 * std
        );
    }

    #[test]
    fn the_network_gives_one_row_of_outputs_per_sample() {
        for (genome, spec) in genomes() {
            let plan = genome.cnn_plan(&spec).expect("plan");
            let (specs, values) = initial(&genome, &plan, 4);
            let mut net = Net::new(&plan, &specs, &values).expect("net");
            let rows = 3;
            let x = Tensor::from_f32(vec![0.5; rows * spec.n_in], &{
                let mut shape = vec![rows];
                shape.extend_from_slice(net.sample());
                shape
            })
            .expect("input");

            assert_eq!(
                net.forward_eval(&x).expect("eval").dims(),
                [rows, spec.n_out]
            );
            assert_eq!(
                net.forward_train(&x).expect("train").dims(),
                [rows, spec.n_out]
            );
        }
    }

    #[test]
    fn values_that_do_not_fit_the_plan_are_refused() {
        let genome = testing::vgg();
        let spec = TaskKind::Shapes.spec();
        let plan = genome.cnn_plan(&spec).expect("plan");
        let (specs, values) = initial(&genome, &plan, 5);

        let mut short = values.clone();

        short.weights.pop();
        assert!(Net::new(&plan, &specs, &short).is_err());

        let mut long = values.clone();

        long.biases.push(vec![0.0]);
        assert!(Net::new(&plan, &specs, &long).is_err());

        let mut wrong = values.clone();

        wrong.weights[1].push(0.0);
        assert!(Net::new(&plan, &specs, &wrong).is_err());

        let mut no_stats = values;

        no_stats.running.clear();
        assert!(Net::new(&plan, &specs, &no_stats).is_err());
    }

    #[test]
    fn plan_order_and_the_two_lists_convert_both_ways() {
        let genome = testing::vgg();
        let plan = genome.cnn_plan(&TaskKind::Shapes.spec()).expect("plan");
        let (specs, values) = initial(&genome, &plan, 6);
        let ordered = values.plan_order(&specs);

        assert_eq!(ordered.len(), specs.len());
        assert_eq!(
            Values::from_plan_order(&specs, &ordered, values.running.clone()),
            values
        );
    }
}
