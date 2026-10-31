//! Shape inference of a CNN genome: the layers it describes, with the shape after each
//! layer, its parameters and its forward FLOPs.
//!
//! `Genome::validate` runs `CnnGenome::plan` as its shape check, and training builds the
//! layers of the plan in its order. So the counts here are those of the trained network.
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
//! Counting (design point D7). Every trainable value costs 2 FLOPs, one multiply-add, each
//! time the forward pass uses it. A convolution and a normalization use each of their values
//! once per output position, a dense layer once per sample. Activations, pooling, the
//! statistics of a normalization, sums and concatenations cost nothing, as the activations
//! cost nothing in the MLP rule. Training costs 3 times the forward pass, since the backward
//! pass costs twice the forward pass. For a dense layer this gives the MLP's 6 FLOPs per
//! parameter and sample.

use crate::cnn::genome::{
    CnnActivation, CnnGenome, ConvGenes, Dims, Downsample, GROUP_NORM_GROUPS, Head,
    MAX_CHANNELS, MAX_DILATION, MAX_KERNEL, MAX_STAGES, MAX_STRIDE, Norm,
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
    Norm { norm: Norm, channels: usize },
    Activation(CnnActivation),
    /// A pooling window of `window` cells per axis that moves by `window` cells.
    Pool { pool: Pool, window: usize },
    /// The mean of each channel over all positions.
    GlobalAvgPool,
    /// All values of the sample as one vector: channel by channel, each in row-major order.
    Flatten,
    /// A dense layer with a bias.
    Dense { inputs: usize, outputs: usize },
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
    /// Training FLOPs for one sample: 3 times the forward FLOPs (D7). Saturates at
    /// `u64::MAX`.
    pub fn train_flops_per_sample(&self) -> u64 {
        self.forward_flops.saturating_mul(3)
    }
}

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
            return Err(invalid("stages", "no stages; a CNN needs at least one".into()));
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
                return Err(invalid(&at, "a residual connection is not yet built".into()));
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
            return Err(invalid(at, "0 groups; a convolution needs at least one".into()));
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
