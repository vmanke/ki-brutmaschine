//! Shape inference of a CNN genome: the layers it describes, with the shape after each
//! layer, its parameters and its forward FLOPs.
//!
//! `Genome::validate` runs `CnnGenome::plan` as its shape check. Training must build the
//! layers of the plan in its order, so that the counts here are those of the trained
//! network.
//!
//! The stem and every plain stage build, in this order:
//!
//! 1. A convolution from the channels before to `width` channels. Zero padding of
//!    `dilation * (kernel - 1) / 2` cells on each side keeps the size at stride 1. The
//!    convolution moves by `stride` when `down` is `Downsample::Conv`, else by 1. It has a
//!    bias only when the unit has no normalization; the normalization's shift takes the
//!    place of the bias.
//! 2. The normalization, unless it is `Norm::None`.
//! 3. The activation.
//! 4. With a stride above 1 and `down` max or average pooling: a pooling window of `stride`
//!    cells per axis, which moves by `stride`. Cells behind the last window are dropped.
//!
//! A 1-D genome runs its kernels and windows along the sequence only (1 x k), a 2-D genome
//! over both axes of the image (k x k). The head pools or flattens. Then come the dense
//! layers of `Genome::hidden`, each followed by `Genome::activation`, and a dense output
//! layer of `n_out` units without activation.
//!
//! Counting. Every trainable value costs 2 FLOPs, one multiply-add, each time the forward
//! pass uses it. A convolution and a normalization use each of their values once per
//! output position, a dense layer once per sample. Activations, pooling, the statistics of
//! a normalization, sums and concatenations cost nothing, as the activations cost nothing
//! in the MLP rule. Training costs 3 times the forward pass, since the backward pass costs
//! twice the forward pass. For a dense layer this gives the MLP's 6 FLOPs per parameter
//! and sample.
//!
//! Time. A FLOP of a CNN takes longer than a FLOP of an MLP, and how much longer depends on
//! the layout: candle copies the kernel cells of every output position before it
//! multiplies, runs each norm and activation as passes over all values, computes a grouped
//! convolution group by group, and its backward pass of a 2-D convolution loops over every
//! kernel cell. A plan of narrow layers at full size runs at 0.15 GFLOP/s by the count
//! above, one of wide layers at 6 GFLOP/s (one thread, ticket 02). So the count alone
//! cannot keep the training of a CNN within `max_candidate_seconds`; when the clock stops
//! it, its result depends on the load of the machine. The time model therefore prices what
//! the time goes into, the `Counts` of the plan, with the `TRAIN_NANOS_*` and `EVAL_NANOS_*`
//! constants. They come from timing 47 CNNs on the machine that wrote this check, at one
//! thread and with the opt-level 2 of `cargo test`, and they hold a safety factor: every
//! measured time stayed below 0.86 of the training estimate and 0.94 of the scoring
//! estimate. More threads made most plans faster; grouped convolutions got up to 15 %
//! slower at four threads. `Genome::validate` holds a CNN's estimated training time to
//! `max_train_flops` divided by `TIME_BUDGET_FLOPS_PER_SECOND`, 20 s at the default limits,
//! and `brut-eval` holds the estimated time of scoring to `MAX_EVAL_SECONDS`. Another or a
//! busier machine can be slower than the model; then the clock can still stop a candidate.

use crate::cnn::genome::{
    CnnActivation, CnnGenome, ConvGenes, Dims, Downsample, GROUP_NORM_GROUPS, Head, MAX_CHANNELS,
    MAX_DILATION, MAX_KERNEL, MAX_STAGES, MAX_STRIDE, Norm,
};
use crate::genome::Activation;
use crate::task::{InputShape, TaskSpec};
use crate::util::{BrutError, Result};

/// The values of one sample between two layers: `channels` values at each position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    pub channels: usize,
    /// The length of a sequence, or the height and width of an image. Empty after the
    /// head pools or flattens; `channels` then counts the features.
    pub size: Vec<usize>,
}

impl Shape {
    /// A plain vector of `features` values.
    pub fn vector(features: usize) -> Shape {
        Shape {
            channels: features,
            size: Vec::new(),
        }
    }

    /// Positions per channel: the product of `size`, 1 for a vector. Saturates at
    /// `usize::MAX`; the shapes of a plan never get there.
    pub fn positions(&self) -> usize {
        self.size
            .iter()
            .fold(1usize, |acc, &len| acc.saturating_mul(len))
    }
}

/// Max or average pooling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pool {
    Max,
    Avg,
}

/// One layer of the network a CNN genome describes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layer {
    /// A convolution over every axis of the shape, from `inputs` to `outputs` channels.
    /// `padding` zero cells sit on both sides of each axis.
    Conv {
        inputs: usize,
        outputs: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    },
    /// A normalization of `channels` channels, with a scale and a shift per channel.
    /// It is never `Norm::None`.
    Norm {
        norm: Norm,
        channels: usize,
    },
    Activation(CnnActivation),
    /// A pooling window of `window` cells per axis that moves by `window` cells.
    Pool {
        pool: Pool,
        window: usize,
    },
    /// The mean of each channel over all positions.
    GlobalAvgPool,
    /// All values of the sample as one vector: channel by channel, each in row-major order.
    Flatten,
    /// A dense layer with a bias.
    Dense {
        inputs: usize,
        outputs: usize,
    },
}

/// One layer of a plan, with the shape after it and its costs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    pub layer: Layer,
    pub output: Shape,
    /// Trainable values of the layer.
    pub params: usize,
    /// Forward FLOPs of the layer for one sample.
    pub flops: u64,
}

/// The layers a CNN genome describes for one task, in the order the forward pass runs them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// The shape of one input sample.
    pub input: Shape,
    pub steps: Vec<Step>,
    /// Trainable values of the network: the sum over the steps.
    pub params: usize,
    /// Forward FLOPs for one sample: the sum over the steps.
    pub forward_flops: u64,
}

impl Plan {
    /// Training FLOPs for one sample: 3 times the forward FLOPs. Saturates at `u64::MAX`.
    pub fn train_flops_per_sample(&self) -> u64 {
        self.forward_flops.saturating_mul(3)
    }

    /// Values of one sample in the input (all channels) and after every layer. Saturates.
    pub fn values_per_row(&self) -> usize {
        let input = self.input.channels.saturating_mul(self.input.positions());

        self.steps.iter().fold(input, |acc, step| {
            acc.saturating_add(step.output.channels.saturating_mul(step.output.positions()))
        })
    }

    /// The counts of the time model for one sample (see "Time" in the module doc).
    pub fn counts(&self) -> Counts {
        let mut counts = Counts::default();
        let mut before = &self.input;

        for step in &self.steps {
            let out = step.output.positions() as u64;
            let values = (step.output.channels as u64).saturating_mul(out);
            let axes = before.size.len() as u32;

            counts.values = counts.values.saturating_add(values);

            match step.layer {
                Layer::Conv {
                    inputs,
                    kernel,
                    groups,
                    ..
                } => {
                    let cells = (kernel as u64).saturating_pow(axes);
                    let taps = cells.saturating_mul(inputs as u64).saturating_mul(out);
                    let tiles = (groups as u64).saturating_mul(out.div_ceil(TILE_POSITIONS));

                    counts.taps = counts.taps.saturating_add(taps);
                    counts.tiles = counts.tiles.saturating_add(tiles);
                    counts.calls = counts.calls.saturating_add(groups as u64);

                    if groups > 1 {
                        let inputs = (inputs as u64).saturating_mul(before.positions() as u64);

                        counts.group_inputs = counts
                            .group_inputs
                            .saturating_add(inputs.saturating_mul(groups as u64));
                    }
                }
                Layer::Pool { window, .. } => {
                    let cells = (window as u64).saturating_pow(axes);

                    counts.pool_taps = counts.pool_taps.saturating_add(cells.saturating_mul(values));
                    counts.calls = counts.calls.saturating_add(1);
                }
                Layer::Norm { .. }
                | Layer::Activation(_)
                | Layer::GlobalAvgPool
                | Layer::Flatten
                | Layer::Dense { .. } => counts.calls = counts.calls.saturating_add(1),
            }

            before = &step.output;
        }

        counts
    }

    /// Estimated training time of one sample (forward pass, backward pass and update), in
    /// nanoseconds at one thread on the reference machine, safety factor included.
    pub fn train_nanos_per_sample(&self) -> f64 {
        let c = self.counts();

        TRAIN_NANOS_PER_FLOP * self.train_flops_per_sample() as f64
            + TRAIN_NANOS_PER_TAP * c.taps as f64
            + TRAIN_NANOS_PER_VALUE * c.values as f64
            + TRAIN_NANOS_PER_POOL_TAP * c.pool_taps as f64
            + TRAIN_NANOS_PER_GROUP_INPUT * c.group_inputs as f64
            + TRAIN_NANOS_PER_TILE * c.tiles as f64
    }

    /// Estimated time to score `rows` samples in chunks of at most `rows_per_chunk` rows,
    /// in nanoseconds at one thread on the reference machine, safety factor included.
    pub fn eval_nanos(&self, rows: usize, rows_per_chunk: usize) -> f64 {
        let c = self.counts();
        let chunks = rows.div_ceil(rows_per_chunk.max(1)) as f64;
        let per_row = EVAL_NANOS_PER_FLOP * self.forward_flops as f64
            + EVAL_NANOS_PER_TAP * c.taps as f64
            + EVAL_NANOS_PER_VALUE * c.values as f64
            + EVAL_NANOS_PER_POOL_TAP * c.pool_taps as f64
            + EVAL_NANOS_PER_TILE * c.tiles as f64;

        rows as f64 * per_row + chunks * EVAL_NANOS_PER_CALL * c.calls as f64
    }
}

/// The counts of a plan that the time model prices, for one sample. Each saturates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// Kernel cells that each convolution reads: kernel cells times input channels times
    /// output positions. candle copies them into a matrix before it multiplies (im2col).
    pub taps: u64,
    /// Values after each layer.
    pub values: u64,
    /// Window cells of each pooling layer: window cells times output values.
    pub pool_taps: u64,
    /// The input values of each grouped convolution times its groups. candle splits the
    /// input into one part per group, and the backward pass of each part fills a tensor of
    /// the whole input.
    pub group_inputs: u64,
    /// Matrix products per sample: for each convolution, its groups times its tiles of
    /// `TILE_POSITIONS` output positions.
    pub tiles: u64,
    /// Calls into candle per chunk of rows: one per group of each convolution, one per
    /// other layer.
    pub calls: u64,
}

/// Output positions per matrix product of candle's convolution (its `TILE_SIZE`).
const TILE_POSITIONS: u64 = 512;

/// Training nanoseconds per training FLOP of the plan.
pub const TRAIN_NANOS_PER_FLOP: f64 = 0.05;

/// Training nanoseconds per kernel cell read (`Counts::taps`).
pub const TRAIN_NANOS_PER_TAP: f64 = 80.0;

/// Training nanoseconds per value after a layer (`Counts::values`).
pub const TRAIN_NANOS_PER_VALUE: f64 = 72.0;

/// Training nanoseconds per window cell of a pooling layer (`Counts::pool_taps`).
pub const TRAIN_NANOS_PER_POOL_TAP: f64 = 30.0;

/// Training nanoseconds per input value and group of a grouped convolution
/// (`Counts::group_inputs`).
pub const TRAIN_NANOS_PER_GROUP_INPUT: f64 = 1.8;

/// Training nanoseconds per matrix product of a convolution (`Counts::tiles`).
pub const TRAIN_NANOS_PER_TILE: f64 = 5000.0;

/// Scoring nanoseconds per forward FLOP of the plan.
pub const EVAL_NANOS_PER_FLOP: f64 = 0.035;

/// Scoring nanoseconds per kernel cell read (`Counts::taps`).
pub const EVAL_NANOS_PER_TAP: f64 = 5.0;

/// Scoring nanoseconds per value after a layer (`Counts::values`).
pub const EVAL_NANOS_PER_VALUE: f64 = 8.5;

/// Scoring nanoseconds per window cell of a pooling layer (`Counts::pool_taps`).
pub const EVAL_NANOS_PER_POOL_TAP: f64 = 8.0;

/// Scoring nanoseconds per matrix product of a convolution (`Counts::tiles`).
pub const EVAL_NANOS_PER_TILE: f64 = 5000.0;

/// Scoring nanoseconds per call into candle and chunk of rows (`Counts::calls`).
pub const EVAL_NANOS_PER_CALL: f64 = 7500.0;

/// FLOPs per second at which `max_train_flops` becomes a time budget for a CNN: 1 GFLOP/s.
///
/// The estimated training time of a valid CNN stays within `max_train_flops` divided by
/// this rate, 20 s at the default limits (`Genome::validate`).
pub const TIME_BUDGET_FLOPS_PER_SECOND: f64 = 1e9;

impl CnnGenome {
    /// Shape inference: the layers this genome describes on the task `spec`.
    ///
    /// `hidden` and `activation` are the dense layers of the head and their activation
    /// (`Genome::hidden` and `Genome::activation`). The error names the first broken rule:
    ///
    /// - the task gives an image to a 2-D genome or a sequence to a 1-D genome, the size of
    ///   its input shape is `n_in`, and it has at least one output;
    /// - there are 1 to `MAX_STAGES` stages;
    /// - every block kind, option and head is built;
    /// - widths lie in 1..=`MAX_CHANNELS`, kernels are odd and at most `MAX_KERNEL`, strides
    ///   lie in 1..=`MAX_STRIDE`, and dilations in 1..=`MAX_DILATION`;
    /// - the groups divide the input channels and the width;
    /// - a group norm's width is a multiple of `GROUP_NORM_GROUPS`;
    /// - the span of a kernel fits into its input along each axis;
    /// - a pooling window fits into its input along each axis, so no size becomes zero;
    /// - every dense layer has a width of at least 1;
    /// - the counts fit into `usize` and `u64`.
    ///
    /// The limits `max_params` and `max_train_flops` are checked by `Genome::validate`.
    pub fn plan(&self, hidden: &[usize], activation: Activation, spec: &TaskSpec) -> Result<Plan> {
        let input = input_shape(self.dims, spec)?;

        if self.stages.is_empty() {
            return Err(invalid(
                "stages",
                "no stages; a CNN needs at least one".into(),
            ));
        }

        if self.stages.len() > MAX_STAGES {
            return Err(invalid(
                "stages",
                format!("{} stages > max {MAX_STAGES}", self.stages.len()),
            ));
        }

        let mut builder = Builder {
            shape: input.clone(),
            steps: Vec::new(),
            params: 0,
            flops: 0,
        };

        builder.conv_unit("stem", &self.stem.conv_genes())?;

        for (i, stage) in self.stages.iter().enumerate() {
            let at = format!("stage {}", i + 1);

            if !stage.block.is_built() {
                return Err(invalid(
                    &at,
                    format!("block {} is not yet built", stage.block.name()),
                ));
            }

            if stage.se {
                return Err(invalid(
                    &at,
                    "squeeze-and-excitation is not yet built".into(),
                ));
            }

            if stage.residual {
                return Err(invalid(
                    &at,
                    "a residual connection is not yet built".into(),
                ));
            }

            builder.conv_unit(&at, &stage.conv_genes())?;
        }

        builder.head(self.head, hidden, activation, spec.n_out)?;

        Ok(Plan {
            input,
            steps: builder.steps,
            params: builder.params,
            forward_flops: builder.flops,
        })
    }
}

/// An error of the shape check at a place of the genome ("stem", "stage 2", "head").
fn invalid(at: &str, message: String) -> BrutError {
    BrutError::Invalid(format!("cnn: {at}: {message}"))
}

/// The shape of one input sample, if the task suits a genome of these dims.
fn input_shape(dims: Dims, spec: &TaskSpec) -> Result<Shape> {
    let task = spec.kind.name();

    if spec.n_out == 0 {
        return Err(invalid("task", format!("task {task} has no outputs")));
    }

    if !spec.input_fits() {
        return Err(invalid(
            "task",
            format!(
                "the input shape {:?} of task {task} does not hold its {} inputs",
                spec.input, spec.n_in
            ),
        ));
    }

    match (dims, spec.input) {
        (_, InputShape::Flat) => Err(invalid(
            "task",
            format!("a CNN needs an image or a sequence, and task {task} gives a flat vector"),
        )),
        (
            Dims::TwoD,
            InputShape::Image {
                channels,
                height,
                width,
            },
        ) => Ok(Shape {
            channels,
            size: vec![height, width],
        }),
        (Dims::OneD, InputShape::Sequence { channels, length }) => Ok(Shape {
            channels,
            size: vec![length],
        }),
        (Dims::TwoD, InputShape::Sequence { .. }) => Err(invalid(
            "task",
            format!("a 2-D CNN needs an image, and task {task} gives a sequence"),
        )),
        (Dims::OneD, InputShape::Image { .. }) => Err(invalid(
            "task",
            format!("a 1-D CNN needs a sequence, and task {task} gives an image"),
        )),
    }
}

/// The plan under construction.
struct Builder {
    /// The shape after the last step.
    shape: Shape,
    steps: Vec<Step>,
    params: usize,
    flops: u64,
}

impl Builder {
    /// Append a layer, and add its costs to the totals.
    fn push(
        &mut self,
        at: &str,
        layer: Layer,
        output: Shape,
        params: usize,
        flops: u64,
    ) -> Result<()> {
        self.params = self
            .params
            .checked_add(params)
            .ok_or_else(|| invalid(at, "the parameter count overflows".into()))?;
        self.flops = self
            .flops
            .checked_add(flops)
            .ok_or_else(|| invalid(at, "the FLOP count overflows".into()))?;
        self.shape = output.clone();
        self.steps.push(Step {
            layer,
            output,
            params,
            flops,
        });

        Ok(())
    }

    /// The convolution, normalization, activation and pooling of the stem or a plain stage.
    fn conv_unit(&mut self, at: &str, genes: &ConvGenes) -> Result<()> {
        let ConvGenes {
            width,
            kernel,
            stride,
            down,
            dilation,
            groups,
            norm,
            activation,
        } = *genes;
        let inputs = self.shape.channels;

        if width == 0 || width > MAX_CHANNELS {
            return Err(invalid(
                at,
                format!("width {width} outside 1..={MAX_CHANNELS}"),
            ));
        }

        if kernel == 0 || kernel > MAX_KERNEL {
            return Err(invalid(
                at,
                format!("kernel {kernel} outside 1..={MAX_KERNEL}"),
            ));
        }

        if kernel.is_multiple_of(2) {
            return Err(invalid(
                at,
                format!("kernel {kernel} is even; only an odd kernel keeps the size at stride 1"),
            ));
        }

        if stride == 0 || stride > MAX_STRIDE {
            return Err(invalid(
                at,
                format!("stride {stride} outside 1..={MAX_STRIDE}"),
            ));
        }

        if dilation == 0 || dilation > MAX_DILATION {
            return Err(invalid(
                at,
                format!("dilation {dilation} outside 1..={MAX_DILATION}"),
            ));
        }

        if groups == 0 {
            return Err(invalid(
                at,
                "0 groups; a convolution needs at least one".into(),
            ));
        }

        if !inputs.is_multiple_of(groups) {
            return Err(invalid(
                at,
                format!("{groups} groups do not divide the {inputs} input channels"),
            ));
        }

        if !width.is_multiple_of(groups) {
            return Err(invalid(
                at,
                format!("{groups} groups do not divide the width {width}"),
            ));
        }

        if norm == Norm::Group && !width.is_multiple_of(GROUP_NORM_GROUPS) {
            return Err(invalid(
                at,
                format!("group norm needs a width divisible by {GROUP_NORM_GROUPS}, not {width}"),
            ));
        }

        // Kernel, dilation and stride are bounded above, so nothing here overflows.
        let span = dilation * (kernel - 1) + 1;

        if let Some(&len) = self.shape.size.iter().find(|&&len| len < span) {
            return Err(invalid(
                at,
                format!(
                    "kernel {kernel} with dilation {dilation} spans {span} cells, \
                     larger than the input of {len}"
                ),
            ));
        }

        let conv_stride = if down == Downsample::Conv { stride } else { 1 };
        let size: Vec<usize> = self
            .shape
            .size
            .iter()
            .map(|&len| (len - 1) / conv_stride + 1)
            .collect();
        let taps = kernel.pow(size.len() as u32);
        let bias = norm == Norm::None;
        let params = taps
            .checked_mul(inputs / groups)
            .and_then(|p| p.checked_mul(width))
            .and_then(|p| p.checked_add(if bias { width } else { 0 }))
            .ok_or_else(|| invalid(at, "the parameter count overflows".into()))?;
        let output = Shape {
            channels: width,
            size,
        };
        let flops = flops_at(at, params, output.positions())?;

        self.push(
            at,
            Layer::Conv {
                inputs,
                outputs: width,
                kernel,
                stride: conv_stride,
                padding: (span - 1) / 2,
                dilation,
                groups,
                bias,
            },
            output.clone(),
            params,
            flops,
        )?;

        if norm != Norm::None {
            let params = 2 * width;
            let flops = flops_at(at, params, output.positions())?;

            self.push(
                at,
                Layer::Norm {
                    norm,
                    channels: width,
                },
                output.clone(),
                params,
                flops,
            )?;
        }

        self.push(at, Layer::Activation(activation), output, 0, 0)?;

        let pool = match down {
            Downsample::Conv => None,
            Downsample::MaxPool => Some(Pool::Max),
            Downsample::AvgPool => Some(Pool::Avg),
        };

        if let Some(pool) = pool
            && stride > 1
        {
            if let Some(&len) = self.shape.size.iter().find(|&&len| len < stride) {
                return Err(invalid(
                    at,
                    format!(
                        "stride {stride} makes the size zero: a pooling window of {stride} \
                         does not fit into {len} cells"
                    ),
                ));
            }

            let output = Shape {
                channels: width,
                size: self.shape.size.iter().map(|&len| len / stride).collect(),
            };

            self.push(
                at,
                Layer::Pool {
                    pool,
                    window: stride,
                },
                output,
                0,
                0,
            )?;
        }

        Ok(())
    }

    /// The head: pooling or flattening, the dense layers of `hidden`, and the output layer.
    fn head(
        &mut self,
        head: Head,
        hidden: &[usize],
        activation: Activation,
        n_out: usize,
    ) -> Result<()> {
        let at = "head";
        let features = match head {
            Head::GlobalAvgPool => {
                let features = self.shape.channels;

                self.push(at, Layer::GlobalAvgPool, Shape::vector(features), 0, 0)?;

                features
            }
            Head::Flatten => {
                let features = self
                    .shape
                    .channels
                    .checked_mul(self.shape.positions())
                    .ok_or_else(|| invalid(at, "the feature count overflows".into()))?;

                self.push(at, Layer::Flatten, Shape::vector(features), 0, 0)?;

                features
            }
            Head::PerCell => {
                return Err(invalid(
                    at,
                    "the per-cell head (U-Net) is not yet built".into(),
                ));
            }
        };
        let mut inputs = features;

        for (i, &outputs) in hidden.iter().enumerate() {
            if outputs == 0 {
                return Err(invalid(at, format!("dense layer {} has width 0", i + 1)));
            }

            self.dense(at, inputs, outputs)?;
            self.push(
                at,
                Layer::Activation(activation.into()),
                Shape::vector(outputs),
                0,
                0,
            )?;
            inputs = outputs;
        }

        self.dense(at, inputs, n_out)
    }

    /// A dense layer with a bias.
    fn dense(&mut self, at: &str, inputs: usize, outputs: usize) -> Result<()> {
        let params = inputs
            .checked_mul(outputs)
            .and_then(|p| p.checked_add(outputs))
            .ok_or_else(|| invalid(at, "the parameter count overflows".into()))?;
        let flops = flops_at(at, params, 1)?;

        self.push(
            at,
            Layer::Dense { inputs, outputs },
            Shape::vector(outputs),
            params,
            flops,
        )
    }
}

/// Forward FLOPs of `params` values used at `positions` positions: 2 per use.
fn flops_at(at: &str, params: usize, positions: usize) -> Result<u64> {
    2u64.checked_mul(params as u64)
        .and_then(|f| f.checked_mul(positions as u64))
        .ok_or_else(|| invalid(at, "the FLOP count overflows".into()))
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::cnn::genome::testing::{self, plain};
    use crate::cnn::genome::{BLOCKS, Block, Stage};
    use crate::config::Limits;
    use crate::genome::{Arch, Genome};
    use crate::task::TaskKind;
    use crate::variation::{self, RandomMutator, VariantGenerator};

    fn cnn_mut(g: &mut Genome) -> &mut CnnGenome {
        match &mut g.arch {
            Arch::Cnn(cnn) => cnn,
            Arch::Mlp => panic!("expected a CNN genome"),
        }
    }

    /// The plan of a genome that must be valid on `kind` under the default limits.
    fn valid_plan(g: &Genome, kind: TaskKind) -> Plan {
        let spec = kind.spec();

        g.validate(&Limits::default(), &spec)
            .unwrap_or_else(|e| panic!("{}: {e}", g.summary()));

        g.cnn_plan(&spec).expect("a plan")
    }

    /// Channels and size after each step.
    fn shapes(plan: &Plan) -> Vec<(usize, Vec<usize>)> {
        plan.steps
            .iter()
            .map(|step| (step.output.channels, step.output.size.clone()))
            .collect()
    }

    /// `validate` with these limits must fail with a message that contains `phrase`.
    fn assert_rejected_with(g: &Genome, kind: TaskKind, limits: &Limits, phrase: &str) {
        match g.validate(limits, &kind.spec()) {
            Ok(()) => panic!("accepted: {}", g.summary()),
            Err(e) => {
                let message = e.to_string();

                assert!(message.contains(phrase), "{message:?} lacks {phrase:?}");
            }
        }
    }

    /// `validate` with the default limits must fail with a message that contains `phrase`.
    fn assert_rejected(g: &Genome, kind: TaskKind, phrase: &str) {
        assert_rejected_with(g, kind, &Limits::default(), phrase);
    }

    #[test]
    fn lenet_counts_match_a_hand_count() {
        let g = testing::lenet();
        let spec = TaskKind::Shapes.spec();
        let plan = valid_plan(&g, TaskKind::Shapes);

        // Stem: 5x5 convolution 1 -> 6 with bias at 16x16, tanh, average pooling to 8x8.
        // Stage: 5x5 convolution 6 -> 16 with bias at 8x8, tanh, average pooling to 4x4.
        // Head: flatten 16 * 4 * 4 = 256 values, dense 256 -> 32, tanh, dense 32 -> 4.
        let stem = 5 * 5 * 6 + 6;
        let stage = 5 * 5 * 6 * 16 + 16;
        let dense = 256 * 32 + 32;
        let out = 32 * 4 + 4;

        assert_eq!((stem, stage, dense, out), (156, 2416, 8224, 132));
        assert_eq!(plan.params, 10_928);
        assert_eq!(plan.params, stem + stage + dense + out);

        // 2 FLOPs per value and output position.
        let forward = 2 * stem * 16 * 16 + 2 * stage * 8 * 8 + 2 * dense + 2 * out;

        assert_eq!(forward, 405_832);
        assert_eq!(plan.forward_flops, forward as u64);

        assert_eq!(
            shapes(&plan),
            vec![
                (6, vec![16, 16]),
                (6, vec![16, 16]),
                (6, vec![8, 8]),
                (16, vec![8, 8]),
                (16, vec![8, 8]),
                (16, vec![4, 4]),
                (256, vec![]),
                (32, vec![]),
                (32, vec![]),
                (4, vec![]),
            ]
        );
        assert_eq!(
            plan.steps[0].layer,
            Layer::Conv {
                inputs: 1,
                outputs: 6,
                kernel: 5,
                stride: 1,
                padding: 2,
                dilation: 1,
                groups: 1,
                bias: true,
            }
        );
        assert_eq!(
            plan.steps[2].layer,
            Layer::Pool {
                pool: Pool::Avg,
                window: 2,
            }
        );
        assert_eq!(plan.steps[6].layer, Layer::Flatten);
        assert_eq!(plan.steps[8].layer, Layer::Activation(CnnActivation::Tanh));

        // Training: 3 times the forward FLOPs per sample, for 1000 samples and 10 epochs.
        assert_eq!(g.param_count(&spec), 10_928);
        assert_eq!(g.train_flops(&spec), 3 * 405_832 * 1000 * 10);
    }

    #[test]
    fn vgg_counts_match_a_hand_count() {
        let g = testing::vgg();
        let spec = TaskKind::Shapes.spec();
        let plan = valid_plan(&g, TaskKind::Shapes);

        // Every convolution has batch norm, so no bias; each batch norm has 2 values per
        // channel. Stem 1 -> 8 and stage 1 8 -> 8 at 16x16; stage 2 8 -> 16 at 16x16, then
        // max pooling to 8x8; stage 3 16 -> 16 at 8x8, then max pooling to 4x4.
        let stem = 3 * 3 * 8;
        let stage1 = 3 * 3 * 8 * 8;
        let stage2 = 3 * 3 * 8 * 16;
        let stage3 = 3 * 3 * 16 * 16;
        let norms = 2 * (8 + 8 + 16 + 16);
        let out = 256 * 4 + 4;

        assert_eq!(
            (stem, stage1, stage2, stage3, norms, out),
            (72, 576, 1152, 2304, 96, 1028)
        );
        assert_eq!(plan.params, 5228);
        assert_eq!(plan.params, stem + stage1 + stage2 + stage3 + norms + out);

        let forward = 2 * (stem + 2 * 8) * 256
            + 2 * (stage1 + 2 * 8) * 256
            + 2 * (stage2 + 2 * 16) * 256
            + 2 * (stage3 + 2 * 16) * 64
            + 2 * out;

        assert_eq!(forward, 1_255_432);
        assert_eq!(plan.forward_flops, forward as u64);
        assert_eq!(
            plan.steps[1].layer,
            Layer::Norm {
                norm: Norm::Batch,
                channels: 8,
            }
        );
        assert_eq!(
            plan.steps.last().map(|s| s.output.clone()),
            Some(Shape::vector(4))
        );
        assert_eq!(g.train_flops(&spec), 3 * 1_255_432 * 1000 * 5);
    }

    #[test]
    fn groups_and_dilation_count_by_hand() {
        let g = testing::grouped_dilated();
        let plan = valid_plan(&g, TaskKind::Shapes);

        // Stem: 3x3 convolution 1 -> 8 with bias at 16x16.
        let stem = 3 * 3 * 8 + 8;
        // Stage 1: 3x3, dilation 2 (span 5, padding 2), 4 groups of 2 -> 4 channels, stride 2:
        // (16 - 1) / 2 + 1 = 8, so 8x8. Group norm, no bias.
        let stage1 = 3 * 3 * 2 * 16;
        // Stage 2: 3x3, dilation 3 (span 7, fits into 8), depthwise: 16 groups of 1 -> 1.
        let stage2 = 3 * 3 * 16;
        // Head: global average pooling to 16, dense 16 -> 8, GELU, dense 8 -> 4.
        let dense = 16 * 8 + 8;
        let out = 8 * 4 + 4;

        assert_eq!((stem, stage1, stage2, dense, out), (80, 288, 144, 136, 36));
        assert_eq!(plan.params, 748);
        assert_eq!(plan.params, stem + stage1 + 32 + stage2 + 32 + dense + out);

        let forward =
            2 * stem * 256 + 2 * (stage1 + 32) * 64 + 2 * (stage2 + 32) * 64 + 2 * dense + 2 * out;

        assert_eq!(forward, 104_792);
        assert_eq!(plan.forward_flops, forward as u64);
        assert_eq!(
            plan.steps[2].layer,
            Layer::Conv {
                inputs: 8,
                outputs: 16,
                kernel: 3,
                stride: 2,
                padding: 2,
                dilation: 2,
                groups: 4,
                bias: false,
            }
        );
        assert_eq!(
            plan.steps[3].layer,
            Layer::Norm {
                norm: Norm::Group,
                channels: 16,
            }
        );
        assert_eq!(plan.steps[4].layer, Layer::Activation(CnnActivation::Silu));
        assert_eq!(
            plan.steps[5].layer,
            Layer::Conv {
                inputs: 16,
                outputs: 16,
                kernel: 3,
                stride: 1,
                padding: 3,
                dilation: 3,
                groups: 16,
                bias: false,
            }
        );
        assert_eq!(plan.steps[7].layer, Layer::Activation(CnnActivation::Relu6));
        assert_eq!(plan.steps[8].layer, Layer::GlobalAvgPool);
        assert_eq!(plan.steps[10].layer, Layer::Activation(CnnActivation::Gelu));
    }

    #[test]
    fn a_sequence_uses_1_by_k_kernels() {
        let g = testing::sequence();
        let plan = valid_plan(&g, TaskKind::Motif1d);

        // Stem: kernel 5 along the 64 cells, 4 -> 8 channels with bias.
        let stem = 5 * 4 * 8 + 8;
        // Stage: kernel 3, dilation 4 (span 9), 8 -> 8, batch norm, max pooling to 32.
        let stage = 3 * 8 * 8;
        let norm = 2 * 8;
        // Head: global average pooling to 8, dense 8 -> 2.
        let out = 8 * 2 + 2;

        assert_eq!(plan.params, 394);
        assert_eq!(plan.params, stem + stage + norm + out);
        assert_eq!(
            plan.forward_flops,
            (2 * stem * 64 + 2 * (stage + norm) * 64 + 2 * out) as u64
        );
        assert_eq!(
            plan.input,
            Shape {
                channels: 4,
                size: vec![64],
            }
        );
        assert_eq!(
            shapes(&plan)[..6].to_vec(),
            vec![
                (8, vec![64]),
                (8, vec![64]),
                (8, vec![64]),
                (8, vec![64]),
                (8, vec![64]),
                (8, vec![32]),
            ]
        );
    }

    #[test]
    fn rejects_a_flat_task() {
        assert_rejected(&testing::lenet(), TaskKind::TwoSpirals, "flat vector");
        assert_rejected(&testing::sequence(), TaskKind::Parity, "flat vector");
    }

    #[test]
    fn rejects_an_input_shape_that_does_not_hold_the_inputs() {
        let g = testing::lenet();
        let mut spec = TaskKind::Shapes.spec();

        spec.n_in = 255;

        let message = g.validate(&Limits::default(), &spec).expect_err("rejected");

        assert!(message.to_string().contains("does not hold"), "{message}");
    }

    #[test]
    fn rejects_a_task_without_outputs() {
        let g = testing::lenet();
        let mut spec = TaskKind::Shapes.spec();

        spec.n_out = 0;

        let cnn = match &g.arch {
            Arch::Cnn(cnn) => cnn,
            Arch::Mlp => panic!("expected a CNN genome"),
        };
        let message = cnn
            .plan(&g.hidden, g.activation, &spec)
            .expect_err("rejected");

        assert!(message.to_string().contains("no outputs"), "{message}");
        assert!(g.validate(&Limits::default(), &spec).is_err());
    }

    #[test]
    fn rejects_a_2d_genome_on_a_sequence_task() {
        assert_rejected(
            &testing::lenet(),
            TaskKind::Motif1d,
            "a 2-D CNN needs an image",
        );
    }

    #[test]
    fn rejects_a_1d_genome_on_an_image_task() {
        for kind in [TaskKind::Shapes, TaskKind::MirrorSymmetry] {
            assert_rejected(&testing::sequence(), kind, "a 1-D CNN needs a sequence");
        }
    }

    #[test]
    fn rejects_a_genome_without_stages() {
        let mut g = testing::lenet();

        cnn_mut(&mut g).stages.clear();
        assert_rejected(&g, TaskKind::Shapes, "no stages");
    }

    #[test]
    fn rejects_too_many_stages() {
        let mut g = testing::grouped_dilated();

        cnn_mut(&mut g).stages = vec![plain(8, 1, CnnActivation::Relu); MAX_STAGES];
        assert!(
            g.validate(&Limits::default(), &TaskKind::Shapes.spec())
                .is_ok()
        );

        cnn_mut(&mut g)
            .stages
            .push(plain(8, 1, CnnActivation::Relu));
        assert_rejected(&g, TaskKind::Shapes, "13 stages > max 12");
    }

    #[test]
    fn rejects_a_width_of_zero() {
        let mut g = testing::lenet();

        cnn_mut(&mut g).stem.width = 0;
        assert_rejected(&g, TaskKind::Shapes, "stem: width 0 outside");

        let mut g = testing::lenet();

        cnn_mut(&mut g).stages[0].width = 0;
        assert_rejected(&g, TaskKind::Shapes, "stage 1: width 0 outside");
    }

    #[test]
    fn rejects_a_width_above_the_bound() {
        let mut g = testing::lenet();

        cnn_mut(&mut g).stages[0].width = MAX_CHANNELS + 1;
        assert_rejected(&g, TaskKind::Shapes, "width 257 outside");
    }

    #[test]
    fn rejects_a_kernel_out_of_range() {
        for kernel in [0, MAX_KERNEL + 2] {
            let mut g = testing::lenet();

            cnn_mut(&mut g).stem.kernel = kernel;
            assert_rejected(&g, TaskKind::Shapes, &format!("kernel {kernel} outside"));
        }
    }

    #[test]
    fn rejects_an_even_kernel() {
        let mut g = testing::lenet();

        cnn_mut(&mut g).stages[0].kernel = 4;
        assert_rejected(&g, TaskKind::Shapes, "kernel 4 is even");
    }

    #[test]
    fn rejects_a_kernel_larger_than_the_input() {
        // The stage sees 8x8 after the stem's pooling: a kernel of 9 does not fit.
        let mut g = testing::lenet();

        cnn_mut(&mut g).stages[0].kernel = 9;
        assert_rejected(
            &g,
            TaskKind::Shapes,
            "spans 9 cells, larger than the input of 8",
        );

        // Dilation widens the span: kernel 3 with dilation 4 spans 9 cells.
        let mut g = testing::lenet();

        cnn_mut(&mut g).stages[0].kernel = 3;
        cnn_mut(&mut g).stages[0].dilation = 4;
        assert_rejected(&g, TaskKind::Shapes, "spans 9 cells");

        // A span equal to the input fits.
        cnn_mut(&mut g).stages[0].dilation = 3;
        assert!(
            g.validate(&Limits::default(), &TaskKind::Shapes.spec())
                .is_ok()
        );
    }

    #[test]
    fn rejects_a_stride_out_of_range() {
        for stride in [0, MAX_STRIDE + 1] {
            let mut g = testing::lenet();

            cnn_mut(&mut g).stages[0].stride = stride;
            assert_rejected(&g, TaskKind::Shapes, &format!("stride {stride} outside"));
        }
    }

    #[test]
    fn rejects_a_stride_that_makes_a_size_zero() {
        // After the LeNet stage the map is 4x4. Pooling by 4 leaves 1x1, and pooling that
        // by 2 would leave nothing.
        let mut g = testing::lenet();
        let pool = |stride| Stage {
            stride,
            down: Downsample::MaxPool,
            ..plain(16, 1, CnnActivation::Relu)
        };

        cnn_mut(&mut g).stages.push(pool(4));
        assert!(
            g.validate(&Limits::default(), &TaskKind::Shapes.spec())
                .is_ok()
        );

        cnn_mut(&mut g).stages.push(pool(2));
        assert_rejected(
            &g,
            TaskKind::Shapes,
            "stage 3: stride 2 makes the size zero",
        );

        // A strided convolution keeps at least one cell: (1 - 1) / 2 + 1 = 1.
        cnn_mut(&mut g).stages[2].down = Downsample::Conv;
        assert!(
            g.validate(&Limits::default(), &TaskKind::Shapes.spec())
                .is_ok()
        );
    }

    #[test]
    fn rejects_a_dilation_out_of_range() {
        for dilation in [0, MAX_DILATION + 1] {
            let mut g = testing::lenet();

            cnn_mut(&mut g).stages[0].dilation = dilation;
            assert_rejected(
                &g,
                TaskKind::Shapes,
                &format!("dilation {dilation} outside"),
            );
        }
    }

    #[test]
    fn rejects_groups_that_do_not_divide_the_channels() {
        // The LeNet stage maps 6 to 16 channels.
        let cases = [
            (0, "0 groups"),
            (4, "4 groups do not divide the 6 input channels"),
            (3, "3 groups do not divide the width 16"),
        ];

        for (groups, phrase) in cases {
            let mut g = testing::lenet();

            cnn_mut(&mut g).stages[0].groups = groups;
            assert_rejected(&g, TaskKind::Shapes, phrase);
        }

        let mut g = testing::lenet();

        cnn_mut(&mut g).stages[0].groups = 2;
        assert!(
            g.validate(&Limits::default(), &TaskKind::Shapes.spec())
                .is_ok()
        );
    }

    #[test]
    fn rejects_group_norm_on_a_width_it_cannot_split() {
        let mut g = testing::lenet();

        cnn_mut(&mut g).stem.norm = Norm::Group;
        assert_rejected(
            &g,
            TaskKind::Shapes,
            "group norm needs a width divisible by 4, not 6",
        );

        cnn_mut(&mut g).stem.norm = Norm::Layer;
        assert!(
            g.validate(&Limits::default(), &TaskKind::Shapes.spec())
                .is_ok()
        );
    }

    #[test]
    fn rejects_every_block_kind_not_yet_built() {
        assert_eq!(BLOCKS.len(), 13);

        for block in BLOCKS {
            let mut g = testing::lenet();

            cnn_mut(&mut g).stages[0].block = block;

            if block == Block::Plain {
                assert!(block.is_built());
                assert!(
                    g.validate(&Limits::default(), &TaskKind::Shapes.spec())
                        .is_ok()
                );
            } else {
                assert!(!block.is_built());
                assert_rejected(
                    &g,
                    TaskKind::Shapes,
                    &format!("stage 1: block {} is not yet built", block.name()),
                );
            }
        }
    }

    #[test]
    fn rejects_squeeze_and_excitation_until_built() {
        let mut g = testing::lenet();

        cnn_mut(&mut g).stages[0].se = true;
        assert_rejected(
            &g,
            TaskKind::Shapes,
            "squeeze-and-excitation is not yet built",
        );
    }

    #[test]
    fn rejects_a_residual_connection_until_built() {
        let mut g = testing::lenet();

        cnn_mut(&mut g).stages[0].residual = true;
        assert_rejected(&g, TaskKind::Shapes, "residual connection is not yet built");
    }

    #[test]
    fn rejects_the_per_cell_head_until_built() {
        let mut g = testing::lenet();

        cnn_mut(&mut g).head = Head::PerCell;
        assert_rejected(
            &g,
            TaskKind::Shapes,
            "per-cell head (U-Net) is not yet built",
        );
    }

    #[test]
    fn rejects_too_many_parameters() {
        let g = testing::lenet();
        let limits = Limits {
            max_params: 10_927,
            ..Limits::default()
        };

        assert_rejected_with(&g, TaskKind::Shapes, &limits, "10928 params > max 10927");

        let limits = Limits {
            max_params: 10_928,
            ..Limits::default()
        };

        assert!(g.validate(&limits, &TaskKind::Shapes.spec()).is_ok());
    }

    #[test]
    fn rejects_too_many_flops() {
        let g = testing::lenet();
        let flops = g.train_flops(&TaskKind::Shapes.spec());
        let limits = Limits {
            max_train_flops: flops - 1,
            ..Limits::default()
        };

        assert_rejected_with(&g, TaskKind::Shapes, &limits, "train FLOPs > max");

        let limits = Limits {
            max_train_flops: flops,
            ..Limits::default()
        };

        assert!(g.validate(&limits, &TaskKind::Shapes.spec()).is_ok());
    }

    #[test]
    fn rejects_bad_head_layers() {
        // `validate` holds the dense layers of a CNN head to the MLP rules.
        let mut g = testing::lenet();

        g.hidden = vec![0];
        assert_rejected(&g, TaskKind::Shapes, "layer width 0 outside");

        g.hidden = vec![65];
        assert_rejected(&g, TaskKind::Shapes, "layer width 65 outside 1..=64");

        g.hidden = vec![4; 5];
        assert_rejected(&g, TaskKind::Shapes, "5 hidden layers > max 4");

        // The plan alone rejects a dense layer of width 0 too.
        let cnn = match &g.arch {
            Arch::Cnn(cnn) => cnn,
            Arch::Mlp => panic!("expected a CNN genome"),
        };
        let message = cnn
            .plan(&[8, 0], g.activation, &TaskKind::Shapes.spec())
            .expect_err("rejected");

        assert!(
            message.to_string().contains("dense layer 2 has width 0"),
            "{message}"
        );
    }

    #[test]
    fn rejects_bad_training_genes_on_a_cnn() {
        let mut bad: Vec<Genome> = Vec::new();
        let mut g = testing::lenet();

        g.learning_rate = f32::NAN;
        bad.push(g);

        let mut g = testing::lenet();
        g.l2 = 1.0;
        bad.push(g);

        let mut g = testing::lenet();
        g.init_scale = 0.0;
        bad.push(g);

        let mut g = testing::lenet();
        g.batch_size = 20;
        bad.push(g);

        let mut g = testing::lenet();
        g.epochs = 1;
        bad.push(g);

        for g in bad {
            assert!(
                g.validate(&Limits::default(), &TaskKind::Shapes.spec())
                    .is_err()
            );
        }
    }

    #[test]
    fn counts_that_overflow_are_rejected() {
        // A hand-made task with 2^62 inputs: the FLOP count of the stem and the feature
        // count of the flattening head overflow and give an error, not a wrapped number.
        let mut g = testing::lenet();
        let mut spec = TaskKind::Shapes.spec();
        let side = 1usize << 31;

        spec.n_in = side * side;
        spec.input = InputShape::Image {
            channels: 1,
            height: side,
            width: side,
        };
        cnn_mut(&mut g).stem.kernel = 1;
        cnn_mut(&mut g).stem.stride = 1;
        cnn_mut(&mut g).stages = vec![plain(16, 1, CnnActivation::Relu)];

        let message = g.cnn_plan(&spec).expect_err("overflow");

        assert!(message.to_string().contains("overflows"), "{message}");
        assert_eq!(g.param_count(&spec), usize::MAX);
        assert_eq!(g.train_flops(&spec), u64::MAX);
        assert!(g.validate(&Limits::default(), &spec).is_err());
    }

    #[test]
    fn no_genome_panics() {
        // Arbitrary CNN genomes, their mutations and their crossovers, on every task. Each
        // call returns Ok or Err; none panics. `propose` always returns a valid genome.
        let limits = Limits::default();
        let mutator = RandomMutator::default();
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut valid = 0;
        let mut invalid = 0;

        for i in 0..18_000 {
            let kind = TaskKind::ALL[i % TaskKind::ALL.len()];
            let spec = kind.spec();
            let dims = match spec.input {
                InputShape::Sequence { .. } => Dims::OneD,
                _ => Dims::TwoD,
            };
            let g = if i % 3 == 0 {
                testing::plausible(&mut rng, dims)
            } else {
                testing::arbitrary(&mut rng)
            };

            match g.validate(&limits, &spec) {
                Ok(()) => {
                    valid += 1;

                    let plan = g.cnn_plan(&spec).expect("a valid genome has a plan");

                    assert_eq!(g.param_count(&spec), plan.params);
                    assert!(plan.params <= limits.max_params);
                    assert!(g.train_flops(&spec) <= limits.max_train_flops);
                }
                Err(_) => invalid += 1,
            }

            let _ = (
                g.param_count(&spec),
                g.train_flops(&spec),
                g.summary(),
                g.id(),
            );

            let child = variation::mutate(&g, 0.3, &mut rng, &limits, &spec);
            let _ = child.validate(&limits, &spec);
            let _ = variation::repair(g.clone(), &limits, &spec).validate(&limits, &spec);

            let other = testing::arbitrary(&mut rng);
            let mixed = variation::crossover(&g, &other, &mut rng);
            let _ = mixed.validate(&limits, &spec);

            for parents in [vec![&g], vec![&g, &other], vec![&other, &g]] {
                let child = mutator.propose(&parents, &mut rng, &limits, &spec);

                assert!(
                    child.validate(&limits, &spec).is_ok(),
                    "{}",
                    child.summary()
                );
            }
        }

        // Both outcomes occur often, so the test covers the valid and the invalid paths.
        assert!(
            valid > 800 && invalid > 8000,
            "{valid} valid, {invalid} invalid"
        );
    }
}
