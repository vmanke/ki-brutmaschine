//! Shape inference of a CNN genome: the layers it describes, with the shape after each
//! layer, its parameters and its forward FLOPs.
//!
//! `Genome::validate` runs `CnnGenome::plan` as its shape check. Training must build the
//! layers of the plan in its order, so that the counts here are those of the trained
//! network.
//!
//! The stem and every plain stage without options build, in this order:
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
//! Blocks. A stage of another block kind replaces step 1 by the convolutions of its block.
//! Below, `c` is the channels before the stage, `w` its width and `s` the stride of its
//! k x k convolution: `stride` with `Downsample::Conv`, else 1. Every convolution is
//! followed by the normalization as in step 2, and "act" is the stage's activation. The
//! stride, the dilation and the padding of step 1 apply to the k x k convolution; a 1x1
//! convolution has neither. Groups apply where the table names them.
//!
//! | Block | Convolutions |
//! |---|---|
//! | `Plain` | k x k, c to w, groups |
//! | `Basic` | k x k, c to w, groups; act; k x k at stride 1, w to w, groups |
//! | `Bottleneck` | 1x1, c to m; act; k x k, m to m, groups; act; 1x1, m to w |
//! | `Separable` | depthwise k x k, c to c; act; 1x1, c to w, groups |
//! | `InvertedResidual` | 1x1, c to e, groups; act; depthwise k x k, e to e; act; 1x1, e to w, groups |
//! | `Inception` | four branches: 1x1, c to b; 1x1, c to b, act, k x k, b to b, groups; 1x1, c to b, act, (k + 2) x (k + 2), b to b, groups; max pooling of k x k at stride 1, then 1x1, c to b |
//! | `Dense` | k x k at stride 1, c to w, groups; then the join with its input |
//! | `Transition` | 1x1, c to w, groups |
//! | `Fire` | 1x1, c to q; act; then two branches: 1x1, q to w / 2; k x k, q to w / 2, groups |
//! | `Shuffle` | 1x1, c to m, groups; act; the channel shuffle; depthwise k x k, m to m; 1x1, m to w, groups |
//! | `ConvNext` | depthwise k x k, c to c; 1x1, c to x, groups, with a bias; act; 1x1, x to w, groups, with a bias |
//! | `Tcn` | causal k, c to w, groups; act; causal k at stride 1, w to w, groups |
//! | `Decoder` | upsampling and the join (see "U-Net" below); k x k at stride 1, to w, groups |
//!
//! Here m = w / `BOTTLENECK_RATIO`, e = `EXPANSION` * c, b = w / `INCEPTION_BRANCHES`, q =
//! w / `FIRE_SQUEEZE_RATIO` and x = `CONVNEXT_EXPANSION` * c. A depthwise convolution has
//! one group per channel. In an Inception block the last convolution of each branch moves
//! by s, and so do both convolutions of a Fire module after its squeeze. The max pooling of
//! the Inception block has `(k - 1) / 2` cells on each side, which never win, so it keeps
//! the size (`Layer::MaxPoolSame`). A transition is always 1x1 without dilation, as in
//! DenseNet; its kernel gene has no effect. The channel shuffle (`Layer::Shuffle`) runs
//! only with more than one group. The depthwise convolution of a ShuffleNet unit has no
//! activation, as in ShuffleNet. The 1x1 convolutions of ConvNeXt's inverted MLP have no
//! normalization; the block normalizes after its depthwise convolution only. A causal
//! convolution (`Layer::CausalConv`) runs along a sequence with all its padding on the
//! left, so an output reads no later input; a TCN block needs a 1-D genome. A dense layer
//! does not move by a stride, so it downsamples by pooling only, and it keeps at most
//! `MAX_CHANNELS` channels after its join.
//!
//! Branches. A block with parallel branches marks them as a skip connection does:
//! `Layer::Fork` keeps the values, `Layer::Branch` ends a branch and starts the next from
//! those values, and `Layer::Concat` joins the outputs of all branches along the channels,
//! in their order. The last convolution of each branch has its normalization but no
//! activation. The activation of step 7 follows the join; since it acts on each value
//! alone, this equals an activation at the end of each branch. A dense layer is one branch
//! that ends after step 7: its new w channels come first in the join, its input after them.
//!
//! After the convolutions of the block come, in this order:
//!
//! 5. With `Stage::se`: squeeze-and-excitation of the w channels (`Layer::SqueezeExcite`).
//! 6. With `Stage::residual`: the skip connection. A `Layer::Fork` before the first
//!    convolution keeps the stage's input. After step 5, `Layer::Branch` starts the
//!    shortcut from that input, and `Layer::Add` sums the block and the shortcut. The
//!    shortcut is the input itself when the block keeps the channels and the size.
//!    Otherwise it is a 1x1 convolution from c to w that moves by s, with the
//!    normalization (ResNet's projection shortcut).
//! 7. The activation, as in step 3. An inverted residual and a ConvNeXt block have none:
//!    their output stays linear, as in MobileNetV2 and ConvNeXt.
//! 8. The pooling of step 4. A decoder has none: its stride upsamples.
//!
//! A plain stage with options follows the same rule, so its activation comes after the
//! join. A 1-D genome runs its kernels and windows along the sequence only (1 x k), a 2-D
//! genome over both axes of the image (k x k). The head pools or flattens. Then come the
//! dense layers of `Genome::hidden`, each followed by `Genome::activation`, and a dense
//! output layer of `n_out` units without activation.
//!
//! U-Net. A decoder joins the map of an earlier unit, its skip point. Every unit with a
//! stride above 1, except a decoder, makes a skip point: its input when its convolution
//! moves by the stride, else its map before the pooling of step 8. Each decoder joins the
//! latest skip point that no decoder has joined yet, as a closing bracket matches the
//! latest open one. A skip point that no decoder joins keeps nothing, so a genome without
//! decoders has the plan it had before ticket 04. A joined skip point is a `Layer::Fork`
//! that stays open until its decoder. The decoder then:
//!
//! 1. upsamples by its stride: with `Downsample::Conv` a transposed convolution from c to w
//!    channels whose kernel side and stride are both the stride (`Layer::ConvTranspose`,
//!    U-Net's up-convolution), with the normalization; with a pooling kind it repeats each
//!    cell (`Layer::Upsample`) and keeps c;
//! 2. joins the map of its skip point along the channels, after the upsampled values; both
//!    must have the same size (`Layer::Branch`, then `Layer::Concat`);
//! 3. runs the k x k convolution of the table on the joined channels, then steps 5 to 7.
//!    Its skip connection (`Stage::residual`) starts after the join.
//!
//! The per-cell head (`Head::PerCell`) is the head of U-Net. It needs a task with a grid of
//! outputs (`OutputShape::Grid`), and the last map must have the size of that grid. The
//! dense layers of `Genome::hidden`, each followed by `Genome::activation`, and the output
//! layer are 1x1 convolutions with a bias, so every cell gets the same small network. The
//! output layer has the channels of the grid. Then `Layer::Flatten` lays the outputs out as
//! the task stores them, channel by channel and row by row.
//!
//! Counting. Every trainable value costs 2 FLOPs, one multiply-add, each time the forward
//! pass uses it. A convolution and a normalization use each of their values once per output
//! position, a dense layer and a squeeze-and-excitation once per sample. A transposed
//! convolution uses each kernel value once per input position and each bias once per output
//! position. Activations, pooling, the statistics of a normalization, the scaling by the
//! gate of a squeeze-and-excitation, sums, concatenations, upsampling and the channel
//! shuffle cost nothing, as the activations cost nothing in the MLP rule. Training costs 3
//! times the forward pass, since the backward pass costs twice the forward pass. For a
//! dense layer this gives the MLP's 6 FLOPs per parameter and sample.
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
//! constants. They come from timing the training of 77 CNNs (134 runs, 102 of them at one
//! thread) and the scoring of 81 (at one thread) on the machine that wrote this check, with
//! the opt-level 2 of `cargo test`, and they hold a safety factor: every measured time
//! stayed below 0.89 of the training estimate and 0.91 of the scoring estimate. More threads made most plans
//! faster; grouped convolutions got up to 15 % slower at four threads, within that factor. `Genome::validate` holds a CNN's estimated training time to
//! `max_train_flops` divided by `TIME_BUDGET_FLOPS_PER_SECOND`, 20 s at the default limits,
//! and `brut-eval` holds the estimated time of scoring to `MAX_EVAL_SECONDS`. Another or a
//! busier machine can be slower than the model; then the clock can still stop a candidate.
//!
//! The blocks and options of ticket 03 use the same prices. A fork and a branch only hand
//! on a tensor, so they count nothing. A sum counts its values and one call. A
//! squeeze-and-excitation counts its scaled output, its small vectors (3 values per
//! channel, 2 per hidden unit) and `SQUEEZE_EXCITE_CALLS` calls; its dense layers count as
//! FLOPs. The slowest valid CNN of each of 12 layouts of these blocks trained in 0.29 to
//! 0.64 of its estimate and scored in 0.19 to 0.34, at one thread, on a machine where the
//! seven slow layouts of ticket 02 trained in 0.36 to 0.66 of theirs (ticket 03,
//! `slow_valid_block_cnns_train_within_the_time_budget`).
//!
//! The blocks of ticket 04 use the same prices as well. A causal convolution counts as a
//! convolution. A transposed convolution counts as the convolution that its backward pass
//! runs, from its outputs back to its input positions: its kernel cells times its output
//! channels times its input positions as taps, a tile per `TILE_POSITIONS` input positions,
//! and one call. Max pooling that keeps the size counts as pooling, its window cells times
//! its values. An upsampling, a channel shuffle and a join count their values and one call.
//! No price changed, so every plan without these layers keeps its estimate to the bit. The
//! slowest valid CNN of each of 15 layouts of these blocks trained in 0.29 to 0.62 of its
//! estimate and scored in 0.22 to 0.50, at one thread, on a machine where the seven slow
//! layouts of ticket 02 trained in 0.37 to 0.66 of theirs (ticket 04,
//! `slow_valid_family_b_cnns_train_within_the_time_budget`).

use crate::cnn::genome::{
    BOTTLENECK_RATIO, Block, CONVNEXT_EXPANSION, CnnActivation, CnnGenome, ConvGenes, Dims,
    Downsample, EXPANSION, FIRE_SQUEEZE_RATIO, GROUP_NORM_GROUPS, Head, INCEPTION_BRANCHES,
    MAX_CHANNELS, MAX_DILATION, MAX_KERNEL, MAX_STAGES, MAX_STRIDE, Norm, SE_REDUCTION,
};
use crate::genome::Activation;
use crate::task::{InputShape, OutputShape, TaskSpec};
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
    /// A causal convolution along a sequence (TCN), from `inputs` to `outputs` channels:
    /// `dilation * (kernel - 1)` zero cells sit on the left only, so output t reads inputs
    /// up to `t * stride`. Its output is as long as that of a `Conv` with the same stride.
    CausalConv {
        inputs: usize,
        outputs: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    },
    /// A transposed convolution from `inputs` to `outputs` channels whose kernel side and
    /// stride are both `factor`: each cell becomes a block of `factor` cells per axis, each
    /// cell of the block with its own kernel values (U-Net's up-convolution).
    ConvTranspose {
        inputs: usize,
        outputs: usize,
        factor: usize,
        bias: bool,
    },
    /// Nearest upsampling: each cell becomes a block of `factor` cells per axis.
    Upsample {
        factor: usize,
    },
    /// The channel shuffle of ShuffleNet with `groups` groups: channel `g * (c / groups) +
    /// i` moves to `i * groups + g`. The output has the shape of the input.
    Shuffle {
        groups: usize,
    },
    /// Max pooling with a window of `window` cells per axis that moves by 1. `(window - 1)
    /// / 2` cells on each side, which never win, keep the size (Inception).
    MaxPoolSame {
        window: usize,
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
    /// Squeeze-and-excitation (Hu, Shen and Sun 2018) of `channels` channels: the mean of
    /// each channel over all positions, a dense layer to `hidden` units with ReLU, a dense
    /// layer back to `channels` units with a sigmoid, and each channel scaled by its unit.
    /// Both dense layers have a bias. The output has the shape of the input.
    SqueezeExcite {
        channels: usize,
        hidden: usize,
    },
    /// The start of a skip connection: the values here feed every branch until the next
    /// `Add`. The output is the input.
    Fork,
    /// The end of a branch: its output waits for `Add`, and the next branch starts from the
    /// values of the last `Fork`. The output is the shape at that `Fork`.
    Branch,
    /// The end of the last branch: the sum of the outputs of the branches since the last
    /// `Fork`. They all have the same shape, which is the output.
    Add,
    /// The end of the last branch: the outputs of the branches since the last `Fork`, in
    /// their order, and then the values of the last branch, joined along the channels. They
    /// all have the same size.
    Concat,
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
    /// The shape after the stem and after each stage, in order: `units[0]` is the output of
    /// the stem, `units[i]` that of stage i. Crossover reads the resolutions here
    /// (`cnn::variation`).
    pub units: Vec<Shape>,
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

            // A fork and a branch only hand on a tensor that exists already.
            if !matches!(step.layer, Layer::Fork | Layer::Branch) {
                counts.values = counts.values.saturating_add(values);
            }

            match step.layer {
                Layer::Conv {
                    inputs,
                    kernel,
                    groups,
                    ..
                }
                | Layer::CausalConv {
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
                // As the convolution that its backward pass runs: from `outputs` channels
                // at the output positions back to the input positions.
                Layer::ConvTranspose {
                    outputs, factor, ..
                } => {
                    let cells = (factor as u64).saturating_pow(axes);
                    let inputs = before.positions() as u64;
                    let taps = cells.saturating_mul(outputs as u64).saturating_mul(inputs);

                    counts.taps = counts.taps.saturating_add(taps);
                    counts.tiles = counts.tiles.saturating_add(inputs.div_ceil(TILE_POSITIONS));
                    counts.calls = counts.calls.saturating_add(1);
                }
                Layer::Pool { window, .. } | Layer::MaxPoolSame { window } => {
                    let cells = (window as u64).saturating_pow(axes);

                    counts.pool_taps = counts
                        .pool_taps
                        .saturating_add(cells.saturating_mul(values));
                    counts.calls = counts.calls.saturating_add(1);
                }
                Layer::SqueezeExcite { channels, hidden } => {
                    // Besides the scaled output: the means, the two dense layers, the ReLU
                    // and the gate, each a call of its own.
                    let small = 3 * channels as u64 + 2 * hidden as u64;

                    counts.values = counts.values.saturating_add(small);
                    counts.calls = counts.calls.saturating_add(SQUEEZE_EXCITE_CALLS);
                }
                Layer::Fork | Layer::Branch => {}
                Layer::Norm { .. }
                | Layer::Activation(_)
                | Layer::GlobalAvgPool
                | Layer::Flatten
                | Layer::Dense { .. }
                | Layer::Add
                | Layer::Concat
                | Layer::Upsample { .. }
                | Layer::Shuffle { .. } => counts.calls = counts.calls.saturating_add(1),
            }

            before = &step.output;
        }

        counts
    }

    /// Estimated seconds to train on `samples` samples: `train_nanos_per_sample` times them.
    /// `Genome::validate` and `variation::repair` both use it, so they agree to the bit.
    pub fn train_seconds(&self, samples: f64) -> f64 {
        self.train_nanos_per_sample() * samples / 1e9
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

/// Calls into candle of one squeeze-and-excitation: the means, two dense layers, the ReLU,
/// the sigmoid and the scaling.
const SQUEEZE_EXCITE_CALLS: u64 = 6;

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
pub const EVAL_NANOS_PER_TAP: f64 = 7.5;

/// Scoring nanoseconds per value after a layer (`Counts::values`).
pub const EVAL_NANOS_PER_VALUE: f64 = 10.0;

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
    /// - widths lie in 1..=`MAX_CHANNELS`, kernels are odd and at most `MAX_KERNEL`, strides
    ///   lie in 1..=`MAX_STRIDE`, and dilations in 1..=`MAX_DILATION`;
    /// - the width of a bottleneck, an Inception block, a Fire module and a ShuffleNet unit
    ///   is a multiple of 4 (`BOTTLENECK_RATIO`, `INCEPTION_BRANCHES`, `FIRE_SQUEEZE_RATIO`);
    /// - an Inception block's second kernel, its kernel plus 2, is at most `MAX_KERNEL`;
    /// - a dense layer moves by 1 and keeps at most `MAX_CHANNELS` channels after its join;
    /// - a TCN block runs along a sequence;
    /// - a decoder has a stride of at least 2, an earlier stage whose map it joins, and
    ///   upsamples to the size of that map;
    /// - the groups divide the channels before and after each convolution they apply to;
    /// - the width of each group norm is a multiple of `GROUP_NORM_GROUPS`;
    /// - the span of a kernel fits into its input along each axis;
    /// - a pooling window fits into its input along each axis, so no size becomes zero;
    /// - the per-cell head has a task with a grid of outputs, and the last map has the size
    ///   of that grid;
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

        let skips = skip_points(self);
        let mut builder = Builder {
            shape: input.clone(),
            steps: Vec::new(),
            params: 0,
            flops: 0,
            forks: Vec::new(),
        };

        builder.unit(
            "stem",
            Block::Plain,
            &self.stem.conv_genes(),
            false,
            false,
            skips[0],
        )?;

        let mut units = Vec::with_capacity(self.stages.len() + 1);

        units.push(builder.shape.clone());

        for (i, stage) in self.stages.iter().enumerate() {
            builder.unit(
                &format!("stage {}", i + 1),
                stage.block,
                &stage.conv_genes(),
                stage.se,
                stage.residual,
                skips[i + 1],
            )?;
            units.push(builder.shape.clone());
        }

        builder.head(self.head, hidden, activation, spec)?;

        // Only a skip point that a decoder joins opens a fork, so none is left open.
        if !builder.forks.is_empty() {
            return Err(invalid("head", "a skip connection is not joined".into()));
        }

        Ok(Plan {
            input,
            steps: builder.steps,
            params: builder.params,
            forward_flops: builder.flops,
            units,
        })
    }
}

/// For the stem and each stage: true when a decoder joins the map at its downsampling (see
/// the module doc, "U-Net").
///
/// Every unit with a stride above 1, decoders aside, makes a skip point. Each decoder joins
/// the latest skip point that no decoder has joined yet, as a closing bracket matches the
/// latest open one. This reads the genes only; the shape check then tests the sizes.
fn skip_points(cnn: &CnnGenome) -> Vec<bool> {
    let units = std::iter::once((Block::Plain, cnn.stem.stride))
        .chain(cnn.stages.iter().map(|stage| (stage.block, stage.stride)));
    let mut joined = vec![false; cnn.stages.len() + 1];
    let mut open = Vec::new();

    for (unit, (block, stride)) in units.enumerate() {
        if block == Block::Decoder {
            if let Some(point) = open.pop() {
                joined[point] = true;
            }
        } else if stride > 1 {
            open.push(unit);
        }
    }

    joined
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
    /// The open forks, the innermost last: the shape at each fork, and the output shapes of
    /// its finished branches.
    forks: Vec<(Shape, Vec<Shape>)>,
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

    /// The layers of the stem or of a stage of kind `block` with these genes and options.
    /// With `skip`, a decoder joins the map at the unit's downsampling (see the module doc).
    fn unit(
        &mut self,
        at: &str,
        block: Block,
        genes: &ConvGenes,
        se: bool,
        residual: bool,
        skip: bool,
    ) -> Result<()> {
        check_genes(at, genes)?;

        // A decoder's block reads the channels of its join.
        let channels = if block == Block::Decoder {
            self.joined_channels(at, genes)?
        } else {
            self.shape.channels
        };
        let axes = self.shape.size.len();
        let parts = block_parts(at, block, channels, axes, genes)?;

        check_parts(at, channels, &parts, genes.norm)?;

        if block == Block::Decoder {
            self.upsample(at, genes)?;
            self.branch(at)?;
            self.concat(at, "the decoder's upsampled map")?;
        }

        let stride = conv_stride(block, genes);
        let pooled = stride == 1 && genes.stride > 1 && block != Block::Decoder;

        // A strided convolution shrinks the map at once, so its skip point is the input.
        if skip && !pooled {
            self.fork(at)?;
        }

        if block == Block::Dense {
            self.fork(at)?;
        }

        if residual {
            self.fork(at)?;
        }

        self.parts(at, &parts, genes)?;

        if se {
            self.squeeze_excite(at)?;
        }

        if residual {
            self.shortcut(at, stride, genes.norm)?;
        }

        // The linear bottleneck of MobileNetV2, and ConvNeXt: no activation at the end.
        if !matches!(block, Block::InvertedResidual | Block::ConvNext) {
            self.activation(at, genes.activation)?;
        }

        if block == Block::Dense {
            self.branch(at)?;
            self.concat(at, "the dense layer")?;
        }

        if skip && pooled {
            self.fork(at)?;
        }

        // A decoder's stride is its upsampling.
        if block == Block::Decoder {
            return Ok(());
        }

        self.pool(at, genes)
    }

    /// The layers of the parts of a block, in order (see `block_parts`).
    fn parts(&mut self, at: &str, parts: &[Part], genes: &ConvGenes) -> Result<()> {
        let convs = parts
            .iter()
            .filter(|part| matches!(part, Part::Conv(_)))
            .count();
        let mut i = 0;

        for part in parts {
            match *part {
                Part::Conv(conv) => {
                    let which = conv_name(convs, i);

                    i += 1;
                    self.conv(at, &which, &conv, genes.norm == Norm::None || !conv.norm)?;

                    if conv.norm {
                        self.norm(at, genes.norm, conv.outputs)?;
                    }

                    if conv.activation {
                        self.activation(at, genes.activation)?;
                    }
                }
                Part::Shuffle(groups) => {
                    let output = self.shape.clone();

                    self.push(at, Layer::Shuffle { groups }, output, 0, 0)?;
                }
                Part::MaxPoolSame(window) => {
                    let output = self.shape.clone();

                    self.push(at, Layer::MaxPoolSame { window }, output, 0, 0)?;
                }
                Part::Fork => self.fork(at)?,
                Part::Branch => self.branch(at)?,
                Part::Concat => self.concat(at, "a branch")?,
            }
        }

        Ok(())
    }

    /// A convolution of the current values to `conv.outputs` channels, with zero padding
    /// that keeps the size at stride 1, and a bias if `bias`. A causal convolution has all
    /// its padding on the left.
    fn conv(&mut self, at: &str, which: &str, conv: &Conv, bias: bool) -> Result<()> {
        let Conv {
            outputs,
            kernel,
            stride,
            dilation,
            groups,
            causal,
            ..
        } = *conv;
        let inputs = self.shape.channels;

        // Kernel, dilation and stride are bounded above, so nothing here overflows.
        let span = dilation * (kernel - 1) + 1;

        if let Some(&len) = self.shape.size.iter().find(|&&len| len < span) {
            return Err(invalid(
                at,
                format!(
                    "{which}kernel {kernel} with dilation {dilation} spans {span} cells, \
                     larger than the input of {len}"
                ),
            ));
        }

        let size: Vec<usize> = self
            .shape
            .size
            .iter()
            .map(|&len| (len - 1) / stride + 1)
            .collect();
        let taps = kernel.pow(size.len() as u32);
        let params = taps
            .checked_mul(inputs / groups)
            .and_then(|p| p.checked_mul(outputs))
            .and_then(|p| p.checked_add(if bias { outputs } else { 0 }))
            .ok_or_else(|| invalid(at, "the parameter count overflows".into()))?;
        let output = Shape {
            channels: outputs,
            size,
        };
        let flops = flops_at(at, params, output.positions())?;
        let layer = if causal {
            Layer::CausalConv {
                inputs,
                outputs,
                kernel,
                stride,
                dilation,
                groups,
                bias,
            }
        } else {
            Layer::Conv {
                inputs,
                outputs,
                kernel,
                stride,
                padding: (span - 1) / 2,
                dilation,
                groups,
                bias,
            }
        };

        self.push(at, layer, output, params, flops)
    }

    /// The normalization of `channels` channels, unless it is `Norm::None`.
    ///
    /// `check_parts` checks the group norms of a block's convolutions first, with the name of
    /// the convolution. This check also covers the norm of a projection shortcut, which no
    /// part lists: in a ConvNeXt block only the depthwise convolution has a norm.
    fn norm(&mut self, at: &str, norm: Norm, channels: usize) -> Result<()> {
        if norm == Norm::None {
            return Ok(());
        }

        if norm == Norm::Group && !channels.is_multiple_of(GROUP_NORM_GROUPS) {
            return Err(invalid(
                at,
                format!(
                    "group norm needs a width divisible by {GROUP_NORM_GROUPS}, not {channels}"
                ),
            ));
        }

        let params = 2 * channels;
        let output = self.shape.clone();
        let flops = flops_at(at, params, output.positions())?;

        self.push(at, Layer::Norm { norm, channels }, output, params, flops)
    }

    /// The activation of the current values.
    fn activation(&mut self, at: &str, activation: CnnActivation) -> Result<()> {
        let output = self.shape.clone();

        self.push(at, Layer::Activation(activation), output, 0, 0)
    }

    /// Squeeze-and-excitation of the current channels, with `SE_REDUCTION`.
    fn squeeze_excite(&mut self, at: &str) -> Result<()> {
        let channels = self.shape.channels;
        let hidden = (channels / SE_REDUCTION).max(1);

        // The channels are at most MAX_CHANNELS here, so nothing overflows.
        let params = 2 * channels * hidden + hidden + channels;
        let flops = flops_at(at, params, 1)?;
        let output = self.shape.clone();

        self.push(
            at,
            Layer::SqueezeExcite { channels, hidden },
            output,
            params,
            flops,
        )
    }

    /// Open a fork: the current values feed every branch until the join.
    fn fork(&mut self, at: &str) -> Result<()> {
        let shape = self.shape.clone();

        self.forks.push((shape.clone(), Vec::new()));
        self.push(at, Layer::Fork, shape, 0, 0)
    }

    /// End a branch of the innermost fork. The next branch starts from the values there.
    fn branch(&mut self, at: &str) -> Result<()> {
        let output = self.shape.clone();
        let Some((input, outputs)) = self.forks.last_mut() else {
            return Err(invalid(at, "a branch has no fork".into()));
        };

        outputs.push(output);

        let input = input.clone();

        self.push(at, Layer::Branch, input, 0, 0)
    }

    /// Join the branches of the innermost fork along the channels. `what` names the branch
    /// that ended first, for the error when the sizes differ.
    fn concat(&mut self, at: &str, what: &str) -> Result<()> {
        let Some((_, outputs)) = self.forks.pop() else {
            return Err(invalid(at, "a join has no fork".into()));
        };
        let mut channels = self.shape.channels;

        for output in &outputs {
            if output.size != self.shape.size {
                return Err(invalid(
                    at,
                    format!(
                        "{what} has {:?} cells, the map it joins {:?}",
                        output.size, self.shape.size
                    ),
                ));
            }

            // At most a few times MAX_CHANNELS, so nothing overflows.
            channels += output.channels;
        }

        let output = Shape {
            channels,
            size: self.shape.size.clone(),
        };

        self.push(at, Layer::Concat, output, 0, 0)
    }

    /// The shortcut of a skip connection from the values at the innermost fork, and the sum
    /// with the block's output. A 1x1 convolution that moves by `stride`, with the
    /// normalization, projects the input when the block changed its shape.
    fn shortcut(&mut self, at: &str, stride: usize, norm: Norm) -> Result<()> {
        let block = self.shape.clone();

        self.branch(at)?;

        if block != self.shape {
            let projection = Conv {
                outputs: block.channels,
                kernel: 1,
                stride,
                dilation: 1,
                groups: 1,
                norm: true,
                activation: false,
                causal: false,
            };

            self.conv(at, "shortcut: ", &projection, norm == Norm::None)?;
            self.norm(at, norm, block.channels)?;
        }

        // Both paths move by the same stride, so their sizes agree; this only guards it.
        if self.shape != block {
            return Err(invalid(
                at,
                format!("the shortcut gives {:?}, the block {:?}", self.shape, block),
            ));
        }

        self.forks.pop();
        self.push(at, Layer::Add, block, 0, 0)
    }

    /// The channels of a decoder after its join: those of its upsampling (its width after a
    /// transposed convolution, else the current channels), then those of the map it joins.
    fn joined_channels(&self, at: &str, genes: &ConvGenes) -> Result<usize> {
        if genes.stride < 2 {
            return Err(invalid(
                at,
                format!(
                    "a decoder upsamples by its stride, so the stride must be at least 2, not {}",
                    genes.stride
                ),
            ));
        }

        let Some((skip, _)) = self.forks.last() else {
            return Err(invalid(
                at,
                "a decoder joins the map of an earlier stage that downsampled, and none is \
                 left"
                    .into(),
            ));
        };
        let upsampled = if genes.down == Downsample::Conv {
            genes.width
        } else {
            self.shape.channels
        };

        Ok(upsampled + skip.channels)
    }

    /// The upsampling of a decoder by its stride: a transposed convolution to its width with
    /// the normalization, when `down` is `Downsample::Conv`; else nearest upsampling.
    fn upsample(&mut self, at: &str, genes: &ConvGenes) -> Result<()> {
        let factor = genes.stride;
        let size = self
            .shape
            .size
            .iter()
            .map(|&len| len.checked_mul(factor))
            .collect::<Option<Vec<usize>>>()
            .ok_or_else(|| invalid(at, "the size overflows".into()))?;

        if genes.down != Downsample::Conv {
            let output = Shape {
                channels: self.shape.channels,
                size,
            };

            return self.push(at, Layer::Upsample { factor }, output, 0, 0);
        }

        let inputs = self.shape.channels;
        let outputs = genes.width;
        let bias = genes.norm == Norm::None;
        let cells = factor.pow(size.len() as u32);
        let weights = cells
            .checked_mul(inputs)
            .and_then(|w| w.checked_mul(outputs))
            .ok_or_else(|| invalid(at, "the parameter count overflows".into()))?;
        let output = Shape {
            channels: outputs,
            size,
        };
        let biases = if bias { outputs } else { 0 };

        // Each kernel value is used once per input position, each bias once per output
        // position.
        let flops = flops_at(at, weights, self.shape.positions())?
            .checked_add(flops_at(at, biases, output.positions())?)
            .ok_or_else(|| invalid(at, "the FLOP count overflows".into()))?;

        self.push(
            at,
            Layer::ConvTranspose {
                inputs,
                outputs,
                factor,
                bias,
            },
            output,
            weights + biases,
            flops,
        )?;
        self.norm(at, genes.norm, outputs)
    }

    /// With a stride above 1 and a pooling `down`: a pooling window of `stride` cells per
    /// axis, which moves by `stride`.
    fn pool(&mut self, at: &str, genes: &ConvGenes) -> Result<()> {
        let stride = genes.stride;
        let pool = match genes.down {
            Downsample::Conv => None,
            Downsample::MaxPool => Some(Pool::Max),
            Downsample::AvgPool => Some(Pool::Avg),
        };

        let Some(pool) = pool else {
            return Ok(());
        };

        if stride == 1 {
            return Ok(());
        }

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
            channels: self.shape.channels,
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
        )
    }

    /// The head: pooling or flattening, the dense layers of `hidden`, and the output layer;
    /// or the per-cell head of a grid-to-grid task.
    fn head(
        &mut self,
        head: Head,
        hidden: &[usize],
        activation: Activation,
        spec: &TaskSpec,
    ) -> Result<()> {
        let at = "head";
        let features = match head {
            Head::GlobalAvgPool => {
                let features = self.shape.channels;

                self.push(at, Layer::GlobalAvgPool, Shape::vector(features), 0, 0)?;

                features
            }
            Head::Flatten => self.flatten(at)?,
            Head::PerCell => return self.per_cell(hidden, activation, spec),
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

        self.dense(at, inputs, spec.n_out)
    }

    /// Every value of every channel at every position as one vector; returns its length.
    fn flatten(&mut self, at: &str) -> Result<usize> {
        let features = self
            .shape
            .channels
            .checked_mul(self.shape.positions())
            .ok_or_else(|| invalid(at, "the feature count overflows".into()))?;

        self.push(at, Layer::Flatten, Shape::vector(features), 0, 0)?;

        Ok(features)
    }

    /// The per-cell head of U-Net for a task with a grid of outputs: the dense layers of
    /// `hidden`, each followed by `activation`, and the output layer, all as 1x1
    /// convolutions with a bias, so every cell gets the same small network. Then all
    /// outputs of the sample as one vector, channel by channel.
    fn per_cell(
        &mut self,
        hidden: &[usize],
        activation: Activation,
        spec: &TaskSpec,
    ) -> Result<()> {
        let at = "head";
        let task = spec.kind.name();
        let OutputShape::Grid {
            channels,
            height,
            width,
        } = spec.output
        else {
            return Err(invalid(
                at,
                format!(
                    "the per-cell head needs a task with a grid of outputs, and task {task} \
                     gives a vector of {}",
                    spec.n_out
                ),
            ));
        };

        if !spec.output_fits() {
            return Err(invalid(
                "task",
                format!(
                    "the output shape {:?} of task {task} does not hold its {} outputs",
                    spec.output, spec.n_out
                ),
            ));
        }

        if self.shape.size != [height, width] {
            return Err(invalid(
                at,
                format!(
                    "the per-cell head needs the last map at the {height}x{width} cells of \
                     the output grid, not at {:?}",
                    self.shape.size
                ),
            ));
        }

        let cell = |outputs| Conv {
            outputs,
            kernel: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
            norm: false,
            activation: false,
            causal: false,
        };

        for (i, &outputs) in hidden.iter().enumerate() {
            if outputs == 0 {
                return Err(invalid(at, format!("dense layer {} has width 0", i + 1)));
            }

            self.conv(at, "", &cell(outputs), true)?;
            self.activation(at, activation.into())?;
        }

        self.conv(at, "", &cell(channels), true)?;
        self.flatten(at).map(|_| ())
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

/// One convolution of a block, as the table of the module doc lists it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Conv {
    outputs: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    /// True when the normalization follows it. A convolution without one has a bias.
    norm: bool,
    /// True when the stage's activation follows it right away, after its normalization.
    /// The last convolution of a block has its activation after the options instead.
    activation: bool,
    /// True for a causal convolution along a sequence (TCN).
    causal: bool,
}

/// One part of a block, in the order of the forward pass (see `block_parts`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Part {
    Conv(Conv),
    /// The channel shuffle of ShuffleNet with this many groups.
    Shuffle(usize),
    /// Max pooling with a window of this many cells that keeps the size.
    MaxPoolSame(usize),
    /// The start of parallel branches: the values here feed each of them.
    Fork,
    /// The end of one branch.
    Branch,
    /// The end of the last branch, and the join of all of them along the channels.
    Concat,
}

/// The stride of the k x k convolutions of a unit: `stride` when the convolution
/// downsamples, else 1. A decoder's stride upsamples, so its convolution moves by 1.
fn conv_stride(block: Block, genes: &ConvGenes) -> usize {
    if genes.down == Downsample::Conv && block != Block::Decoder {
        genes.stride
    } else {
        1
    }
}

/// Every gene of a unit in its range: width, kernel, stride, dilation and groups.
fn check_genes(at: &str, genes: &ConvGenes) -> Result<()> {
    let ConvGenes {
        width,
        kernel,
        stride,
        dilation,
        groups,
        ..
    } = *genes;

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

    Ok(())
}

/// The parts of a block of kind `block` on `inputs` channels with `axes` spatial axes, in
/// order (see the table of the module doc), after the rules of that kind.
fn block_parts(
    at: &str,
    block: Block,
    inputs: usize,
    axes: usize,
    genes: &ConvGenes,
) -> Result<Vec<Part>> {
    let ConvGenes {
        width,
        kernel,
        dilation,
        groups,
        ..
    } = *genes;
    let stride = conv_stride(block, genes);
    let conv = |outputs, kernel, stride, dilation, groups, activation| {
        Part::Conv(Conv {
            outputs,
            kernel,
            stride,
            dilation,
            groups,
            norm: true,
            activation,
            causal: false,
        })
    };
    let divides = |ratio: usize, what: &str| {
        if width.is_multiple_of(ratio) {
            Ok(width / ratio)
        } else {
            Err(invalid(
                at,
                format!("{what} needs a width divisible by {ratio}, not {width}"),
            ))
        }
    };

    let parts = match block {
        Block::Plain => vec![conv(width, kernel, stride, dilation, groups, false)],
        Block::Basic => vec![
            conv(width, kernel, stride, dilation, groups, true),
            conv(width, kernel, 1, dilation, groups, false),
        ],
        Block::Bottleneck => {
            let inner = divides(BOTTLENECK_RATIO, "a bottleneck")?;

            vec![
                conv(inner, 1, 1, 1, 1, true),
                conv(inner, kernel, stride, dilation, groups, true),
                conv(width, 1, 1, 1, 1, false),
            ]
        }
        Block::Separable => vec![
            conv(inputs, kernel, stride, dilation, inputs, true),
            conv(width, 1, 1, 1, groups, false),
        ],
        Block::InvertedResidual => {
            // The inputs come from the stem or a stage, so they are at most MAX_CHANNELS.
            let expanded = EXPANSION * inputs;

            vec![
                conv(expanded, 1, 1, 1, groups, true),
                conv(expanded, kernel, stride, dilation, expanded, true),
                conv(width, 1, 1, 1, groups, false),
            ]
        }
        Block::Dense => {
            if stride > 1 {
                return Err(invalid(
                    at,
                    "a dense layer joins its input, so it keeps the size and downsamples by \
                     pooling only"
                        .into(),
                ));
            }

            if inputs + width > MAX_CHANNELS {
                return Err(invalid(
                    at,
                    format!(
                        "a dense layer adds its width {width} to its {inputs} input channels: \
                         {} > max {MAX_CHANNELS}",
                        inputs + width
                    ),
                ));
            }

            vec![conv(width, kernel, 1, dilation, groups, false)]
        }
        Block::Transition => vec![conv(width, 1, stride, 1, groups, false)],
        Block::Inception => {
            let branch = divides(INCEPTION_BRANCHES, "an inception block")?;

            if kernel + 2 > MAX_KERNEL {
                return Err(invalid(
                    at,
                    format!(
                        "an inception block has a second kernel of {}, its kernel plus 2, \
                         > max {MAX_KERNEL}",
                        kernel + 2
                    ),
                ));
            }

            vec![
                Part::Fork,
                conv(branch, 1, stride, 1, 1, false),
                Part::Branch,
                conv(branch, 1, 1, 1, 1, true),
                conv(branch, kernel, stride, dilation, groups, false),
                Part::Branch,
                conv(branch, 1, 1, 1, 1, true),
                conv(branch, kernel + 2, stride, dilation, groups, false),
                Part::Branch,
                Part::MaxPoolSame(kernel),
                conv(branch, 1, stride, 1, 1, false),
                Part::Concat,
            ]
        }
        Block::Fire => {
            let squeeze = divides(FIRE_SQUEEZE_RATIO, "a fire module")?;
            let half = width / 2;

            vec![
                conv(squeeze, 1, 1, 1, 1, true),
                Part::Fork,
                conv(half, 1, stride, 1, 1, false),
                Part::Branch,
                conv(half, kernel, stride, dilation, groups, false),
                Part::Concat,
            ]
        }
        Block::Shuffle => {
            let inner = divides(BOTTLENECK_RATIO, "a shufflenet unit")?;
            let mut parts = vec![conv(inner, 1, 1, 1, groups, true)];

            if groups > 1 {
                parts.push(Part::Shuffle(groups));
            }

            // ShuffleNet has no activation after its depthwise convolution.
            parts.push(conv(inner, kernel, stride, dilation, inner, false));
            parts.push(conv(width, 1, 1, 1, groups, false));

            parts
        }
        Block::ConvNext => {
            // The inputs come from the stem or a stage, so they are at most MAX_CHANNELS.
            let expanded = CONVNEXT_EXPANSION * inputs;
            let pointwise = |outputs, activation| {
                Part::Conv(Conv {
                    outputs,
                    kernel: 1,
                    stride: 1,
                    dilation: 1,
                    groups,
                    norm: false,
                    activation,
                    causal: false,
                })
            };

            vec![
                conv(inputs, kernel, stride, dilation, inputs, false),
                pointwise(expanded, true),
                pointwise(width, false),
            ]
        }
        Block::Tcn => {
            if axes != 1 {
                return Err(invalid(
                    at,
                    "a TCN block convolves causally along a sequence, so it needs a 1-D \
                     genome"
                        .into(),
                ));
            }

            let causal = |stride, activation| {
                Part::Conv(Conv {
                    outputs: width,
                    kernel,
                    stride,
                    dilation,
                    groups,
                    norm: true,
                    activation,
                    causal: true,
                })
            };

            vec![causal(stride, true), causal(1, false)]
        }
        Block::Decoder => vec![conv(width, kernel, 1, dilation, groups, false)],
    };

    Ok(parts)
}

/// "convolution 2: " for the second of several convolutions; nothing for a single one.
fn conv_name(convs: usize, i: usize) -> String {
    if convs > 1 {
        format!("convolution {}: ", i + 1)
    } else {
        String::new()
    }
}

/// The groups of each convolution divide its channels before and after, and each group
/// norm can split its width. The channels follow the branches of the parts.
fn check_parts(at: &str, inputs: usize, parts: &[Part], norm: Norm) -> Result<()> {
    let mut before = inputs;
    let mut forks: Vec<(usize, usize)> = Vec::new();
    let mut convs = Vec::new();

    for part in parts {
        match *part {
            Part::Conv(conv) => {
                convs.push((before, conv));
                before = conv.outputs;
            }
            Part::Shuffle(_) | Part::MaxPoolSame(_) => {}
            Part::Fork => forks.push((before, 0)),
            Part::Branch => {
                if let Some((input, ended)) = forks.last_mut() {
                    *ended += before;
                    before = *input;
                }
            }
            Part::Concat => {
                if let Some((_, ended)) = forks.pop() {
                    before += ended;
                }
            }
        }
    }

    for (i, (before, conv)) in convs.iter().enumerate() {
        let which = conv_name(convs.len(), i);
        let groups = conv.groups;

        if !before.is_multiple_of(groups) {
            return Err(invalid(
                at,
                format!("{which}{groups} groups do not divide the {before} input channels"),
            ));
        }

        if !conv.outputs.is_multiple_of(groups) {
            return Err(invalid(
                at,
                format!(
                    "{which}{groups} groups do not divide the width {}",
                    conv.outputs
                ),
            ));
        }
    }

    if norm == Norm::Group {
        for (i, (_, conv)) in convs.iter().enumerate() {
            if conv.norm && !conv.outputs.is_multiple_of(GROUP_NORM_GROUPS) {
                return Err(invalid(
                    at,
                    format!(
                        "{}group norm needs a width divisible by {GROUP_NORM_GROUPS}, not {}",
                        conv_name(convs.len(), i),
                        conv.outputs
                    ),
                ));
            }
        }
    }

    Ok(())
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
    use crate::cnn::genome::{BLOCKS, Block, Stage, Stem};
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
        // channel. Stem 1 -> 8 at 16x16, then max pooling to 8x8; stage 1 8 -> 8 at 8x8;
        // stage 2 8 -> 16 at 8x8, then max pooling to 4x4; stage 3 16 -> 16 at 4x4, then
        // max pooling to 2x2. Flatten: 16 * 2 * 2 = 64 values.
        let stem = 3 * 3 * 8;
        let stage1 = 3 * 3 * 8 * 8;
        let stage2 = 3 * 3 * 8 * 16;
        let stage3 = 3 * 3 * 16 * 16;
        let norms = 2 * (8 + 8 + 16 + 16);
        let out = 64 * 4 + 4;

        assert_eq!(
            (stem, stage1, stage2, stage3, norms, out),
            (72, 576, 1152, 2304, 96, 260)
        );
        assert_eq!(plan.params, 4460);
        assert_eq!(plan.params, stem + stage1 + stage2 + stage3 + norms + out);

        let forward = 2 * (stem + 2 * 8) * 256
            + 2 * (stage1 + 2 * 8) * 64
            + 2 * (stage2 + 2 * 16) * 64
            + 2 * (stage3 + 2 * 16) * 16
            + 2 * out;

        assert_eq!(forward, 347_656);
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
        assert_eq!(g.train_flops(&spec), 3 * 347_656 * 1000 * 5);
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

    /// The genome of `testing::block_genomes` with this name.
    fn block_genome(name: &str) -> Genome {
        testing::block_genomes()
            .into_iter()
            .find(|(n, _, _)| *n == name)
            .map(|(_, _, g)| g)
            .unwrap_or_else(|| panic!("no block genome {name}"))
    }

    /// The steps of a plan from `first` on: each layer with its channels, size and
    /// parameters.
    fn steps_from(plan: &Plan, first: usize) -> Vec<(Layer, usize, Vec<usize>, usize)> {
        plan.steps[first..]
            .iter()
            .map(|s| {
                (
                    s.layer.clone(),
                    s.output.channels,
                    s.output.size.clone(),
                    s.params,
                )
            })
            .collect()
    }

    /// A convolution layer without dilation.
    fn conv(inputs: usize, outputs: usize, kernel: usize, stride: usize, groups: usize) -> Layer {
        Layer::Conv {
            inputs,
            outputs,
            kernel,
            stride,
            padding: (kernel - 1) / 2,
            dilation: 1,
            groups,
            bias: false,
        }
    }

    /// A batch norm of `channels` channels.
    fn batch_norm(channels: usize) -> Layer {
        Layer::Norm {
            norm: Norm::Batch,
            channels,
        }
    }

    /// The common stem of `testing::block_genomes`: a 3x3 convolution from 1 to 8 channels
    /// with batch norm (72 + 16 values), ReLU and max pooling to 8x8. Four steps.
    const BLOCK_STEM: usize = 4;

    #[test]
    fn the_block_stem_counts_by_hand() {
        let plan = valid_plan(&block_genome("plain, dilated"), TaskKind::Shapes);
        let relu = Layer::Activation(CnnActivation::Relu);

        assert_eq!(
            plan.steps[..BLOCK_STEM]
                .iter()
                .map(|s| (
                    s.layer.clone(),
                    s.output.channels,
                    s.output.size.clone(),
                    s.params
                ))
                .collect::<Vec<_>>(),
            vec![
                (conv(1, 8, 3, 1, 1), 8, vec![16, 16], 72),
                (batch_norm(8), 8, vec![16, 16], 16),
                (relu, 8, vec![16, 16], 0),
                (
                    Layer::Pool {
                        pool: Pool::Max,
                        window: 2,
                    },
                    8,
                    vec![8, 8],
                    0,
                ),
            ]
        );
    }

    #[test]
    fn a_plain_stage_with_both_options_counts_by_hand() {
        let plan = valid_plan(
            &block_genome("plain with squeeze-and-excitation and an identity shortcut"),
            TaskKind::Shapes,
        );
        let relu = Layer::Activation(CnnActivation::Relu);
        // Squeeze-and-excitation of 8 channels: 8 / 4 = 2 units, so 8 * 2 + 2 and 2 * 8 + 8.
        let se = 8 * 2 + 2 + 2 * 8 + 8;

        assert_eq!(se, 42);
        assert_eq!(
            steps_from(&plan, BLOCK_STEM),
            vec![
                (Layer::Fork, 8, vec![8, 8], 0),
                (conv(8, 8, 3, 1, 1), 8, vec![8, 8], 3 * 3 * 8 * 8),
                (batch_norm(8), 8, vec![8, 8], 16),
                (
                    Layer::SqueezeExcite {
                        channels: 8,
                        hidden: 2,
                    },
                    8,
                    vec![8, 8],
                    se,
                ),
                // The block keeps the shape, so the shortcut is the input itself.
                (Layer::Branch, 8, vec![8, 8], 0),
                (Layer::Add, 8, vec![8, 8], 0),
                (relu, 8, vec![8, 8], 0),
                (Layer::GlobalAvgPool, 8, vec![], 0),
                (
                    Layer::Dense {
                        inputs: 8,
                        outputs: 4,
                    },
                    4,
                    vec![],
                    36,
                ),
            ]
        );
        assert_eq!(plan.params, 88 + 576 + 16 + se + 36);

        // The gate's dense layers use their values once per sample, the rest per position.
        let forward = 2 * 88 * 256 + 2 * (576 + 16) * 64 + 2 * se + 2 * 36;

        assert_eq!(plan.forward_flops, forward as u64);
    }

    #[test]
    fn dilated_plain_stages_count_by_hand() {
        let plan = valid_plan(&block_genome("plain, dilated"), TaskKind::Shapes);
        let relu = Layer::Activation(CnnActivation::Relu);
        let dilated = |dilation: usize| Layer::Conv {
            inputs: 8,
            outputs: 8,
            kernel: 3,
            stride: 1,
            padding: dilation,
            dilation,
            groups: 1,
            bias: false,
        };

        // Dilation 2 spans 5 cells and dilation 3 spans 7, both within 8: the padding of
        // (span - 1) / 2 keeps 8x8. Dilation does not change the parameters.
        assert_eq!(
            steps_from(&plan, BLOCK_STEM)[..6].to_vec(),
            vec![
                (dilated(2), 8, vec![8, 8], 576),
                (batch_norm(8), 8, vec![8, 8], 16),
                (relu.clone(), 8, vec![8, 8], 0),
                (dilated(3), 8, vec![8, 8], 576),
                (batch_norm(8), 8, vec![8, 8], 16),
                (relu, 8, vec![8, 8], 0),
            ]
        );
        assert_eq!(plan.params, 88 + 2 * (576 + 16) + 36);
    }

    #[test]
    fn a_basic_block_with_a_strided_projection_counts_by_hand() {
        let plan = valid_plan(
            &block_genome("basic, strided, with a projection"),
            TaskKind::Shapes,
        );
        let relu = Layer::Activation(CnnActivation::Relu);

        // The first 3x3 convolution moves by 2: (8 - 1) / 2 + 1 = 4. The shortcut is a 1x1
        // convolution from 8 to 16 channels that moves by 2 as well, with its batch norm.
        assert_eq!(
            steps_from(&plan, BLOCK_STEM),
            vec![
                (Layer::Fork, 8, vec![8, 8], 0),
                (conv(8, 16, 3, 2, 1), 16, vec![4, 4], 3 * 3 * 8 * 16),
                (batch_norm(16), 16, vec![4, 4], 32),
                (relu.clone(), 16, vec![4, 4], 0),
                (conv(16, 16, 3, 1, 1), 16, vec![4, 4], 3 * 3 * 16 * 16),
                (batch_norm(16), 16, vec![4, 4], 32),
                (Layer::Branch, 8, vec![8, 8], 0),
                (conv(8, 16, 1, 2, 1), 16, vec![4, 4], 8 * 16),
                (batch_norm(16), 16, vec![4, 4], 32),
                (Layer::Add, 16, vec![4, 4], 0),
                (relu, 16, vec![4, 4], 0),
                (Layer::GlobalAvgPool, 16, vec![], 0),
                (
                    Layer::Dense {
                        inputs: 16,
                        outputs: 4,
                    },
                    4,
                    vec![],
                    68,
                ),
            ]
        );

        let block = 1152 + 32 + 2304 + 32 + 128 + 32;

        assert_eq!(block, 3680);
        assert_eq!(plan.params, 88 + block + 68);

        let forward = 2 * 88 * 256 + 2 * block * 16 + 2 * 68;

        assert_eq!(plan.forward_flops, forward as u64);
    }

    #[test]
    fn a_basic_block_without_norm_has_biases_also_in_its_projection() {
        let plan = valid_plan(
            &block_genome("basic without norm, with a projection that has a bias"),
            TaskKind::Shapes,
        );
        let biased = |inputs: usize, outputs: usize, kernel: usize| Layer::Conv {
            inputs,
            outputs,
            kernel,
            stride: 1,
            padding: (kernel - 1) / 2,
            dilation: 1,
            groups: 1,
            bias: true,
        };
        let silu = Layer::Activation(CnnActivation::Silu);

        assert_eq!(
            steps_from(&plan, BLOCK_STEM)[..8].to_vec(),
            vec![
                (Layer::Fork, 8, vec![8, 8], 0),
                (biased(8, 12, 3), 12, vec![8, 8], 9 * 8 * 12 + 12),
                (silu.clone(), 12, vec![8, 8], 0),
                (biased(12, 12, 3), 12, vec![8, 8], 9 * 12 * 12 + 12),
                (Layer::Branch, 8, vec![8, 8], 0),
                (biased(8, 12, 1), 12, vec![8, 8], 8 * 12 + 12),
                (Layer::Add, 12, vec![8, 8], 0),
                (silu, 12, vec![8, 8], 0),
            ]
        );
        assert_eq!(plan.params, 88 + 876 + 1308 + 108 + 52);
    }

    #[test]
    fn a_bottleneck_counts_by_hand() {
        let plan = valid_plan(
            &block_genome("bottleneck with group norm, groups and a projection"),
            TaskKind::Shapes,
        );
        let relu = Layer::Activation(CnnActivation::Relu);
        let group_norm = |channels| Layer::Norm {
            norm: Norm::Group,
            channels,
        };

        // Width 16, so 16 / 4 = 4 inner channels. The 3x3 convolution has 2 groups of 2
        // channels. The projection maps the 8 input channels to 16 at stride 1.
        assert_eq!(
            steps_from(&plan, BLOCK_STEM)[..13].to_vec(),
            vec![
                (Layer::Fork, 8, vec![8, 8], 0),
                (conv(8, 4, 1, 1, 1), 4, vec![8, 8], 8 * 4),
                (group_norm(4), 4, vec![8, 8], 8),
                (relu.clone(), 4, vec![8, 8], 0),
                (conv(4, 4, 3, 1, 2), 4, vec![8, 8], 3 * 3 * 2 * 4),
                (group_norm(4), 4, vec![8, 8], 8),
                (relu.clone(), 4, vec![8, 8], 0),
                (conv(4, 16, 1, 1, 1), 16, vec![8, 8], 4 * 16),
                (group_norm(16), 16, vec![8, 8], 32),
                (Layer::Branch, 8, vec![8, 8], 0),
                (conv(8, 16, 1, 1, 1), 16, vec![8, 8], 8 * 16),
                (group_norm(16), 16, vec![8, 8], 32),
                (Layer::Add, 16, vec![8, 8], 0),
            ]
        );
        assert_eq!(
            plan.params,
            88 + (32 + 8 + 72 + 8 + 64 + 32 + 128 + 32) + 68
        );
    }

    #[test]
    fn a_separable_block_counts_by_hand() {
        let plan = valid_plan(
            &block_genome(
                "separable with layer norm, squeeze-and-excitation, a projection and pooling",
            ),
            TaskKind::Shapes,
        );
        let relu = Layer::Activation(CnnActivation::Relu);
        let layer_norm = |channels| Layer::Norm {
            norm: Norm::Layer,
            channels,
        };
        let se = 16 * 4 + 4 + 4 * 16 + 16;

        // The depthwise 3x3 convolution has one group per channel: 9 values each. The
        // stage pools, so its convolutions and its projection keep 8x8, and the max pooling
        // after the activation halves the size.
        assert_eq!(
            steps_from(&plan, BLOCK_STEM)[..13].to_vec(),
            vec![
                (Layer::Fork, 8, vec![8, 8], 0),
                (conv(8, 8, 3, 1, 8), 8, vec![8, 8], 3 * 3 * 8),
                (layer_norm(8), 8, vec![8, 8], 16),
                (relu.clone(), 8, vec![8, 8], 0),
                (conv(8, 16, 1, 1, 1), 16, vec![8, 8], 8 * 16),
                (layer_norm(16), 16, vec![8, 8], 32),
                (
                    Layer::SqueezeExcite {
                        channels: 16,
                        hidden: 4,
                    },
                    16,
                    vec![8, 8],
                    se,
                ),
                (Layer::Branch, 8, vec![8, 8], 0),
                (conv(8, 16, 1, 1, 1), 16, vec![8, 8], 8 * 16),
                (layer_norm(16), 16, vec![8, 8], 32),
                (Layer::Add, 16, vec![8, 8], 0),
                (relu, 16, vec![8, 8], 0),
                (
                    Layer::Pool {
                        pool: Pool::Max,
                        window: 2,
                    },
                    16,
                    vec![4, 4],
                    0,
                ),
            ]
        );
        assert_eq!(se, 148);
        assert_eq!(plan.params, 88 + (72 + 16 + 128 + 32 + se + 128 + 32) + 68);
    }

    #[test]
    fn inverted_residuals_count_by_hand() {
        let plan = valid_plan(
            &block_genome("inverted residuals, one with an identity shortcut, one strided"),
            TaskKind::Shapes,
        );
        let relu6 = Layer::Activation(CnnActivation::Relu6);

        // The stem pools by 4, to 4x4. 8 input channels expand to 6 * 8 = 48. The first
        // block keeps the shape, so its shortcut is the input. Neither block ends with an
        // activation: the projection stays linear. The second block's depthwise
        // convolution moves by 2: (4 - 1) / 2 + 1 = 2.
        assert_eq!(plan.steps[BLOCK_STEM - 1].output.size, vec![4, 4]);
        assert_eq!(
            steps_from(&plan, BLOCK_STEM)[..19].to_vec(),
            vec![
                (Layer::Fork, 8, vec![4, 4], 0),
                (conv(8, 48, 1, 1, 1), 48, vec![4, 4], 8 * 48),
                (batch_norm(48), 48, vec![4, 4], 96),
                (relu6.clone(), 48, vec![4, 4], 0),
                (conv(48, 48, 3, 1, 48), 48, vec![4, 4], 3 * 3 * 48),
                (batch_norm(48), 48, vec![4, 4], 96),
                (relu6.clone(), 48, vec![4, 4], 0),
                (conv(48, 8, 1, 1, 1), 8, vec![4, 4], 48 * 8),
                (batch_norm(8), 8, vec![4, 4], 16),
                (Layer::Branch, 8, vec![4, 4], 0),
                (Layer::Add, 8, vec![4, 4], 0),
                (conv(8, 48, 1, 1, 1), 48, vec![4, 4], 8 * 48),
                (batch_norm(48), 48, vec![4, 4], 96),
                (relu6.clone(), 48, vec![4, 4], 0),
                (conv(48, 48, 3, 2, 48), 48, vec![2, 2], 3 * 3 * 48),
                (batch_norm(48), 48, vec![2, 2], 96),
                (relu6, 48, vec![2, 2], 0),
                (conv(48, 16, 1, 1, 1), 16, vec![2, 2], 48 * 16),
                (batch_norm(16), 16, vec![2, 2], 32),
            ]
        );
        assert_eq!(plan.steps[BLOCK_STEM + 19].layer, Layer::GlobalAvgPool);

        let first = 384 + 96 + 432 + 96 + 384 + 16;
        let second = 384 + 96 + 432 + 96 + 768 + 32;

        assert_eq!(plan.params, 88 + first + second + 68);

        // The second block's expansion runs at 4x4, the rest of it at 2x2.
        let forward = 2 * 88 * 256
            + 2 * first * 16
            + 2 * (384 + 96) * 16
            + 2 * (432 + 96 + 768 + 32) * 4
            + 2 * 68;

        assert_eq!(plan.forward_flops, forward as u64);
    }

    #[test]
    fn a_1d_genome_runs_its_blocks_along_the_sequence() {
        let (_, task, g) = testing::block_genomes()
            .into_iter()
            .find(|(_, task, _)| *task == TaskKind::Motif1d)
            .expect("a 1-D block genome");
        let plan = valid_plan(&g, task);

        // Stem: kernel 5 along 64 cells, 4 -> 8 channels with batch norm, pooled to 32.
        // Basic: two kernels of 3 at dilation 2 on 8 channels, with squeeze-and-excitation
        // and the input as shortcut. Separable: a depthwise kernel of 3 that moves by 2,
        // then 8 -> 16, squeeze-and-excitation, and a projection that moves by 2.
        let stem = 5 * 4 * 8 + 16;
        let basic = 2 * (3 * 8 * 8 + 16) + (8 * 2 + 2 + 2 * 8 + 8);
        let separable = 3 * 8 + 16 + 8 * 16 + 32 + (16 * 4 + 4 + 4 * 16 + 16) + 8 * 16 + 32;
        let out = 16 * 2 + 2;

        assert_eq!(plan.params, stem + basic + separable + out);
        assert_eq!(
            plan.steps.last().map(|s| s.output.clone()),
            Some(Shape::vector(2))
        );

        let sizes: Vec<Vec<usize>> = plan.steps.iter().map(|s| s.output.size.clone()).collect();

        assert!(sizes.contains(&vec![32]) && sizes.contains(&vec![16]));
        assert!(sizes.iter().all(|size| size.len() <= 1));
    }

    /// A normalization of this kind of `channels` channels.
    fn norm_of(norm: Norm, channels: usize) -> Layer {
        Layer::Norm { norm, channels }
    }

    /// A 1x1 convolution with a bias, as the inverted MLP of ConvNeXt and the per-cell head
    /// have them.
    fn pointwise(inputs: usize, outputs: usize) -> Layer {
        Layer::Conv {
            inputs,
            outputs,
            kernel: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true,
        }
    }

    #[test]
    fn dense_layers_and_a_transition_count_by_hand() {
        let plan = valid_plan(
            &block_genome("dense layers and a transition with group norm"),
            TaskKind::Shapes,
        );
        let relu = Layer::Activation(CnnActivation::Relu);

        // Each dense layer adds 4 new channels to its input: 8, then 12, then 16. The new
        // ones come first in the join. The transition maps 16 to 8 by a 1x1 convolution with
        // group norm, then pools by 2.
        assert_eq!(
            steps_from(&plan, BLOCK_STEM),
            vec![
                (Layer::Fork, 8, vec![8, 8], 0),
                (conv(8, 4, 3, 1, 1), 4, vec![8, 8], 3 * 3 * 8 * 4),
                (batch_norm(4), 4, vec![8, 8], 8),
                (relu.clone(), 4, vec![8, 8], 0),
                (Layer::Branch, 8, vec![8, 8], 0),
                (Layer::Concat, 12, vec![8, 8], 0),
                (Layer::Fork, 12, vec![8, 8], 0),
                (
                    Layer::Conv {
                        inputs: 12,
                        outputs: 4,
                        kernel: 3,
                        stride: 1,
                        padding: 2,
                        dilation: 2,
                        groups: 1,
                        bias: false,
                    },
                    4,
                    vec![8, 8],
                    3 * 3 * 12 * 4
                ),
                (batch_norm(4), 4, vec![8, 8], 8),
                (relu.clone(), 4, vec![8, 8], 0),
                (Layer::Branch, 12, vec![8, 8], 0),
                (Layer::Concat, 16, vec![8, 8], 0),
                (conv(16, 8, 1, 1, 1), 8, vec![8, 8], 16 * 8),
                (norm_of(Norm::Group, 8), 8, vec![8, 8], 16),
                (relu, 8, vec![8, 8], 0),
                (
                    Layer::Pool {
                        pool: Pool::Avg,
                        window: 2,
                    },
                    8,
                    vec![4, 4],
                    0,
                ),
                (Layer::GlobalAvgPool, 8, vec![], 0),
                (
                    Layer::Dense {
                        inputs: 8,
                        outputs: 4,
                    },
                    4,
                    vec![],
                    36,
                ),
            ]
        );

        let (dense1, dense2, transition) = (288 + 8, 432 + 8, 128 + 16);

        assert_eq!(plan.params, 88 + dense1 + dense2 + transition + 36);
        assert_eq!(plan.params, 1004);

        // All three run at 8x8; the join and the pooling cost nothing.
        let forward = 2 * 88 * 256 + 2 * (dense1 + dense2 + transition) * 64 + 2 * 36;

        assert_eq!(plan.forward_flops, forward as u64);
    }

    #[test]
    fn inception_blocks_count_by_hand() {
        let plan = valid_plan(
            &block_genome("inception blocks, one strided with groups and a projection"),
            TaskKind::Shapes,
        );
        let relu = Layer::Activation(CnnActivation::Relu);
        let same = Layer::MaxPoolSame { window: 3 };

        // Width 16, so four branches of 4 channels: a 1x1; a 1x1 reduction and a 3x3; a 1x1
        // reduction and a 5x5; a 3x3 max pooling that keeps the size and a 1x1. Each branch
        // ends with its norm; the ReLU follows the join.
        let first = vec![
            (Layer::Fork, 8, vec![8, 8], 0),
            (conv(8, 4, 1, 1, 1), 4, vec![8, 8], 32),
            (batch_norm(4), 4, vec![8, 8], 8),
            (Layer::Branch, 8, vec![8, 8], 0),
            (conv(8, 4, 1, 1, 1), 4, vec![8, 8], 32),
            (batch_norm(4), 4, vec![8, 8], 8),
            (relu.clone(), 4, vec![8, 8], 0),
            (conv(4, 4, 3, 1, 1), 4, vec![8, 8], 3 * 3 * 4 * 4),
            (batch_norm(4), 4, vec![8, 8], 8),
            (Layer::Branch, 8, vec![8, 8], 0),
            (conv(8, 4, 1, 1, 1), 4, vec![8, 8], 32),
            (batch_norm(4), 4, vec![8, 8], 8),
            (relu.clone(), 4, vec![8, 8], 0),
            (conv(4, 4, 5, 1, 1), 4, vec![8, 8], 5 * 5 * 4 * 4),
            (batch_norm(4), 4, vec![8, 8], 8),
            (Layer::Branch, 8, vec![8, 8], 0),
            (same.clone(), 8, vec![8, 8], 0),
            (conv(8, 4, 1, 1, 1), 4, vec![8, 8], 32),
            (batch_norm(4), 4, vec![8, 8], 8),
            (Layer::Concat, 16, vec![8, 8], 0),
            (relu.clone(), 16, vec![8, 8], 0),
        ];

        // The second moves by 2: the last convolution of every branch does, and the 3x3 and
        // 5x5 have 2 groups of 2 channels. Its projection maps 16 to 16 channels at 4x4.
        let second = vec![
            (Layer::Fork, 16, vec![8, 8], 0),
            (Layer::Fork, 16, vec![8, 8], 0),
            (conv(16, 4, 1, 2, 1), 4, vec![4, 4], 64),
            (batch_norm(4), 4, vec![4, 4], 8),
            (Layer::Branch, 16, vec![8, 8], 0),
            (conv(16, 4, 1, 1, 1), 4, vec![8, 8], 64),
            (batch_norm(4), 4, vec![8, 8], 8),
            (relu.clone(), 4, vec![8, 8], 0),
            (conv(4, 4, 3, 2, 2), 4, vec![4, 4], 3 * 3 * 2 * 4),
            (batch_norm(4), 4, vec![4, 4], 8),
            (Layer::Branch, 16, vec![8, 8], 0),
            (conv(16, 4, 1, 1, 1), 4, vec![8, 8], 64),
            (batch_norm(4), 4, vec![8, 8], 8),
            (relu.clone(), 4, vec![8, 8], 0),
            (conv(4, 4, 5, 2, 2), 4, vec![4, 4], 5 * 5 * 2 * 4),
            (batch_norm(4), 4, vec![4, 4], 8),
            (Layer::Branch, 16, vec![8, 8], 0),
            (same, 16, vec![8, 8], 0),
            (conv(16, 4, 1, 2, 1), 4, vec![4, 4], 64),
            (batch_norm(4), 4, vec![4, 4], 8),
            (Layer::Concat, 16, vec![4, 4], 0),
            (Layer::Branch, 16, vec![8, 8], 0),
            (conv(16, 16, 1, 2, 1), 16, vec![4, 4], 256),
            (batch_norm(16), 16, vec![4, 4], 32),
            (Layer::Add, 16, vec![4, 4], 0),
            (relu, 16, vec![4, 4], 0),
        ];
        let mut expected = first;

        expected.extend(second);
        assert_eq!(
            steps_from(&plan, BLOCK_STEM)[..expected.len()].to_vec(),
            expected
        );

        let first = 40 + (40 + 152) + (40 + 408) + 40;
        let second = 72 + (72 + 80) + (72 + 208) + 72 + 288;

        assert_eq!((first, second), (720, 864));
        assert_eq!(plan.params, 88 + first + second + 68);

        // The reductions of the second block run at 8x8, the rest of it at 4x4.
        let forward =
            2 * 88 * 256 + 2 * first * 64 + 2 * 144 * 64 + 2 * (second - 144) * 16 + 2 * 68;

        assert_eq!(plan.forward_flops, forward as u64);
    }

    #[test]
    fn fire_modules_count_by_hand() {
        let plan = valid_plan(
            &block_genome("fire modules, one with squeeze-and-excitation and an identity shortcut"),
            TaskKind::Shapes,
        );
        let relu = Layer::Activation(CnnActivation::Relu);

        // Width 16: a squeeze to 16 / 4 = 4 channels, then a 1x1 and a 3x3 expansion to 8
        // each, joined to 16.
        let fire = |inputs: usize| {
            vec![
                (conv(inputs, 4, 1, 1, 1), 4, vec![8, 8], inputs * 4),
                (batch_norm(4), 4, vec![8, 8], 8),
                (relu.clone(), 4, vec![8, 8], 0),
                (Layer::Fork, 4, vec![8, 8], 0),
                (conv(4, 8, 1, 1, 1), 8, vec![8, 8], 32),
                (batch_norm(8), 8, vec![8, 8], 16),
                (Layer::Branch, 4, vec![8, 8], 0),
                (conv(4, 8, 3, 1, 1), 8, vec![8, 8], 3 * 3 * 4 * 8),
                (batch_norm(8), 8, vec![8, 8], 16),
                (Layer::Concat, 16, vec![8, 8], 0),
            ]
        };
        let se = 16 * 4 + 4 + 4 * 16 + 16;
        let mut expected = fire(8);

        expected.push((relu.clone(), 16, vec![8, 8], 0));
        expected.push((Layer::Fork, 16, vec![8, 8], 0));
        expected.extend(fire(16));
        expected.extend([
            (
                Layer::SqueezeExcite {
                    channels: 16,
                    hidden: 4,
                },
                16,
                vec![8, 8],
                se,
            ),
            (Layer::Branch, 16, vec![8, 8], 0),
            (Layer::Add, 16, vec![8, 8], 0),
            (relu.clone(), 16, vec![8, 8], 0),
        ]);
        assert_eq!(
            steps_from(&plan, BLOCK_STEM)[..expected.len()].to_vec(),
            expected
        );

        let (first, second) = (32 + 8 + 48 + 304, 64 + 8 + 48 + 304 + se);

        assert_eq!((first, second), (392, 572));
        assert_eq!(plan.params, 88 + first + second + 68);

        let forward = 2 * 88 * 256 + 2 * (first + second - se) * 64 + 2 * se + 2 * 68;

        assert_eq!(plan.forward_flops, forward as u64);
    }

    #[test]
    fn shufflenet_units_count_by_hand() {
        let plan = valid_plan(
            &block_genome("shufflenet units with 2 and 4 groups and projections"),
            TaskKind::Shapes,
        );
        let relu = Layer::Activation(CnnActivation::Relu);

        // Width 16, so 16 / 4 = 4 inner channels. A 1x1 group convolution, the shuffle, a
        // depthwise 3x3 without activation, and a 1x1 group convolution to 16. The second
        // unit moves by 2 in its depthwise convolution and in its projection.
        assert_eq!(
            steps_from(&plan, BLOCK_STEM),
            vec![
                (Layer::Fork, 8, vec![8, 8], 0),
                (conv(8, 4, 1, 1, 2), 4, vec![8, 8], 4 * 4),
                (batch_norm(4), 4, vec![8, 8], 8),
                (relu.clone(), 4, vec![8, 8], 0),
                (Layer::Shuffle { groups: 2 }, 4, vec![8, 8], 0),
                (conv(4, 4, 3, 1, 4), 4, vec![8, 8], 3 * 3 * 4),
                (batch_norm(4), 4, vec![8, 8], 8),
                (conv(4, 16, 1, 1, 2), 16, vec![8, 8], 2 * 16),
                (batch_norm(16), 16, vec![8, 8], 32),
                (Layer::Branch, 8, vec![8, 8], 0),
                (conv(8, 16, 1, 1, 1), 16, vec![8, 8], 128),
                (batch_norm(16), 16, vec![8, 8], 32),
                (Layer::Add, 16, vec![8, 8], 0),
                (relu.clone(), 16, vec![8, 8], 0),
                (Layer::Fork, 16, vec![8, 8], 0),
                (conv(16, 4, 1, 1, 4), 4, vec![8, 8], 4 * 4),
                (batch_norm(4), 4, vec![8, 8], 8),
                (relu.clone(), 4, vec![8, 8], 0),
                (Layer::Shuffle { groups: 4 }, 4, vec![8, 8], 0),
                (conv(4, 4, 3, 2, 4), 4, vec![4, 4], 3 * 3 * 4),
                (batch_norm(4), 4, vec![4, 4], 8),
                (conv(4, 16, 1, 1, 4), 16, vec![4, 4], 16),
                (batch_norm(16), 16, vec![4, 4], 32),
                (Layer::Branch, 16, vec![8, 8], 0),
                (conv(16, 16, 1, 2, 1), 16, vec![4, 4], 256),
                (batch_norm(16), 16, vec![4, 4], 32),
                (Layer::Add, 16, vec![4, 4], 0),
                (relu, 16, vec![4, 4], 0),
                (Layer::GlobalAvgPool, 16, vec![], 0),
                (
                    Layer::Dense {
                        inputs: 16,
                        outputs: 4,
                    },
                    4,
                    vec![],
                    68,
                ),
            ]
        );

        let first = 16 + 8 + 36 + 8 + 32 + 32 + 128 + 32;
        let second = 16 + 8 + 36 + 8 + 16 + 32 + 256 + 32;

        assert_eq!((first, second), (292, 404));
        assert_eq!(plan.params, 88 + first + second + 68);

        // The first 1x1 of the second unit runs at 8x8, the rest at 4x4.
        let forward = 2 * 88 * 256 + 2 * first * 64 + 2 * 24 * 64 + 2 * (second - 24) * 16 + 2 * 68;

        assert_eq!(plan.forward_flops, forward as u64);
    }

    #[test]
    fn convnext_blocks_count_by_hand() {
        let plan = valid_plan(
            &block_genome("convnext blocks with layer norm and gelu, one strided"),
            TaskKind::Shapes,
        );
        let gelu = Layer::Activation(CnnActivation::Gelu);
        let layer_norm = |channels| norm_of(Norm::Layer, channels);

        // A depthwise 7x7 with layer norm and no bias; an inverted MLP of 1x1 convolutions
        // with biases, to 4 x 8 = 32 channels and back, GELU between them; no activation at
        // the end. The second moves by 2 in its depthwise 3x3, and projects 8 to 16 channels.
        assert_eq!(
            steps_from(&plan, BLOCK_STEM),
            vec![
                (Layer::Fork, 8, vec![8, 8], 0),
                (conv(8, 8, 7, 1, 8), 8, vec![8, 8], 7 * 7 * 8),
                (layer_norm(8), 8, vec![8, 8], 16),
                (pointwise(8, 32), 32, vec![8, 8], 8 * 32 + 32),
                (gelu.clone(), 32, vec![8, 8], 0),
                (pointwise(32, 8), 8, vec![8, 8], 32 * 8 + 8),
                (Layer::Branch, 8, vec![8, 8], 0),
                (Layer::Add, 8, vec![8, 8], 0),
                (Layer::Fork, 8, vec![8, 8], 0),
                (conv(8, 8, 3, 2, 8), 8, vec![4, 4], 3 * 3 * 8),
                (layer_norm(8), 8, vec![4, 4], 16),
                (pointwise(8, 32), 32, vec![4, 4], 288),
                (gelu, 32, vec![4, 4], 0),
                (pointwise(32, 16), 16, vec![4, 4], 32 * 16 + 16),
                (Layer::Branch, 8, vec![8, 8], 0),
                (conv(8, 16, 1, 2, 1), 16, vec![4, 4], 128),
                (layer_norm(16), 16, vec![4, 4], 32),
                (Layer::Add, 16, vec![4, 4], 0),
                (Layer::GlobalAvgPool, 16, vec![], 0),
                (
                    Layer::Dense {
                        inputs: 16,
                        outputs: 4,
                    },
                    4,
                    vec![],
                    68,
                ),
            ]
        );

        let (first, second) = (392 + 16 + 288 + 264, 72 + 16 + 288 + 528 + 128 + 32);

        assert_eq!((first, second), (960, 1064));
        assert_eq!(plan.params, 88 + first + second + 68);

        let forward = 2 * 88 * 256 + 2 * first * 64 + 2 * second * 16 + 2 * 68;

        assert_eq!(plan.forward_flops, forward as u64);
    }

    /// A causal convolution without a bias.
    fn causal(inputs: usize, outputs: usize, stride: usize, dilation: usize) -> Layer {
        Layer::CausalConv {
            inputs,
            outputs,
            kernel: 3,
            stride,
            dilation,
            groups: 1,
            bias: false,
        }
    }

    #[test]
    fn tcn_blocks_count_by_hand() {
        let plan = valid_plan(
            &block_genome("tcn blocks with dilations 1, 2 and 4, the last strided"),
            TaskKind::Motif1d,
        );
        let relu = Layer::Activation(CnnActivation::Relu);

        // The stem keeps the 64 cells. Each block has two causal convolutions of 3 taps with
        // its dilation; the first of the last block moves by 2, and so does its projection.
        assert_eq!(
            steps_from(&plan, 3),
            vec![
                (Layer::Fork, 8, vec![64], 0),
                (causal(8, 8, 1, 1), 8, vec![64], 3 * 8 * 8),
                (batch_norm(8), 8, vec![64], 16),
                (relu.clone(), 8, vec![64], 0),
                (causal(8, 8, 1, 1), 8, vec![64], 3 * 8 * 8),
                (batch_norm(8), 8, vec![64], 16),
                (Layer::Branch, 8, vec![64], 0),
                (Layer::Add, 8, vec![64], 0),
                (relu.clone(), 8, vec![64], 0),
                (Layer::Fork, 8, vec![64], 0),
                (causal(8, 16, 1, 2), 16, vec![64], 3 * 8 * 16),
                (batch_norm(16), 16, vec![64], 32),
                (relu.clone(), 16, vec![64], 0),
                (causal(16, 16, 1, 2), 16, vec![64], 3 * 16 * 16),
                (batch_norm(16), 16, vec![64], 32),
                (Layer::Branch, 8, vec![64], 0),
                (conv(8, 16, 1, 1, 1), 16, vec![64], 128),
                (batch_norm(16), 16, vec![64], 32),
                (Layer::Add, 16, vec![64], 0),
                (relu.clone(), 16, vec![64], 0),
                (Layer::Fork, 16, vec![64], 0),
                (causal(16, 16, 2, 4), 16, vec![32], 768),
                (batch_norm(16), 16, vec![32], 32),
                (relu.clone(), 16, vec![32], 0),
                (causal(16, 16, 1, 4), 16, vec![32], 768),
                (batch_norm(16), 16, vec![32], 32),
                (Layer::Branch, 16, vec![64], 0),
                (conv(16, 16, 1, 2, 1), 16, vec![32], 256),
                (batch_norm(16), 16, vec![32], 32),
                (Layer::Add, 16, vec![32], 0),
                (relu, 16, vec![32], 0),
                (Layer::GlobalAvgPool, 16, vec![], 0),
                (
                    Layer::Dense {
                        inputs: 16,
                        outputs: 2,
                    },
                    2,
                    vec![],
                    34,
                ),
            ]
        );

        let stem = 5 * 4 * 8 + 16;
        let blocks = [
            416,
            384 + 32 + 768 + 32 + 128 + 32,
            768 + 32 + 768 + 32 + 256 + 32,
        ];

        assert_eq!(blocks, [416, 1376, 1888]);
        assert_eq!(plan.params, stem + blocks.iter().sum::<usize>() + 34);

        // The last block's first convolution and its projection already run at 32 cells.
        let forward = 2 * stem * 64 + 2 * (416 + 1376) * 64 + 2 * 1888 * 32 + 2 * 34;

        assert_eq!(plan.forward_flops, forward as u64);
    }

    #[test]
    fn a_u_net_counts_by_hand() {
        let plan = valid_plan(
            &block_genome(
                "a u-net with a transposed convolution, nearest upsampling and a per-cell head",
            ),
            TaskKind::MirrorGrid,
        );
        let relu = Layer::Activation(CnnActivation::Relu);

        // Stage 1 moves by 2 in its first convolution, so its skip point is its input: 8
        // channels at 8x8. Stage 2 pools, so its skip point is its map before the pooling:
        // 16 channels at 4x4. The first decoder upsamples 2x2 to 4x4 by a transposed
        // convolution to 16 channels and joins the map of stage 2; the second upsamples to
        // 8x8 by repeating cells and joins the map of stage 1. The per-cell head maps the 8
        // channels of every cell to 1 value, and the grid becomes a vector of 64.
        assert_eq!(
            steps_from(&plan, 0),
            vec![
                (conv(1, 8, 3, 1, 1), 8, vec![8, 8], 72),
                (batch_norm(8), 8, vec![8, 8], 16),
                (relu.clone(), 8, vec![8, 8], 0),
                (Layer::Fork, 8, vec![8, 8], 0),
                (conv(8, 16, 3, 2, 1), 16, vec![4, 4], 1152),
                (batch_norm(16), 16, vec![4, 4], 32),
                (relu.clone(), 16, vec![4, 4], 0),
                (conv(16, 16, 3, 1, 1), 16, vec![4, 4], 2304),
                (batch_norm(16), 16, vec![4, 4], 32),
                (relu.clone(), 16, vec![4, 4], 0),
                (conv(16, 16, 3, 1, 1), 16, vec![4, 4], 2304),
                (batch_norm(16), 16, vec![4, 4], 32),
                (relu.clone(), 16, vec![4, 4], 0),
                (Layer::Fork, 16, vec![4, 4], 0),
                (
                    Layer::Pool {
                        pool: Pool::Max,
                        window: 2,
                    },
                    16,
                    vec![2, 2],
                    0,
                ),
                (
                    Layer::ConvTranspose {
                        inputs: 16,
                        outputs: 16,
                        factor: 2,
                        bias: false,
                    },
                    16,
                    vec![4, 4],
                    16 * 16 * 4,
                ),
                (batch_norm(16), 16, vec![4, 4], 32),
                (Layer::Branch, 16, vec![4, 4], 0),
                (Layer::Concat, 32, vec![4, 4], 0),
                (conv(32, 16, 3, 1, 1), 16, vec![4, 4], 4608),
                (batch_norm(16), 16, vec![4, 4], 32),
                (relu.clone(), 16, vec![4, 4], 0),
                (Layer::Upsample { factor: 2 }, 16, vec![8, 8], 0),
                (Layer::Branch, 8, vec![8, 8], 0),
                (Layer::Concat, 24, vec![8, 8], 0),
                (Layer::Fork, 24, vec![8, 8], 0),
                (conv(24, 8, 3, 1, 1), 8, vec![8, 8], 1728),
                (batch_norm(8), 8, vec![8, 8], 16),
                (Layer::Branch, 24, vec![8, 8], 0),
                (conv(24, 8, 1, 1, 1), 8, vec![8, 8], 192),
                (batch_norm(8), 8, vec![8, 8], 16),
                (Layer::Add, 8, vec![8, 8], 0),
                (relu, 8, vec![8, 8], 0),
                (pointwise(8, 1), 1, vec![8, 8], 9),
                (Layer::Flatten, 64, vec![], 0),
            ]
        );

        let stages = [88, 3520, 2336, 5696, 1952];

        assert_eq!(plan.params, stages.iter().sum::<usize>() + 9);
        assert_eq!(plan.params, 13_601);

        // The transposed convolution uses each kernel value once per input cell: 2x2.
        let forward = 2 * 88 * 64
            + 2 * (3520 + 2336) * 16
            + 2 * 1024 * 4
            + 2 * (32 + 4640) * 16
            + 2 * 1952 * 64
            + 2 * 9 * 64;

        assert_eq!(plan.forward_flops, forward as u64);

        // The time model counts the transposed convolution as the convolution that its
        // backward pass runs: 2x2 cells of 16 channels at each of 2x2 input positions, one
        // matrix product. The upsampling and the joins count their values and a call each.
        let c = plan.counts();
        let taps = 9 * 64
            + 9 * 8 * 16
            + 2 * 9 * 16 * 16
            + 4 * 16 * 4
            + 9 * 32 * 16
            + 9 * 24 * 64
            + 24 * 64
            + 8 * 64;

        assert_eq!(c.taps, taps as u64);
        assert_eq!(c.tiles, 9);
        assert_eq!(c.pool_taps, 4 * 16 * 4);
        assert_eq!(plan.steps.len(), 35);

        // 35 layers, of which 3 forks and 3 branches cost no call.
        assert_eq!(c.calls, 35 - 6);
    }

    #[test]
    fn a_1d_genome_runs_the_blocks_of_ticket_04_along_the_sequence() {
        let plan = valid_plan(
            &block_genome("1-D inception, shufflenet, dense, decoder and convnext blocks"),
            TaskKind::Motif1d,
        );

        // Stem 176. Inception of width 8 at 32 cells: four branches of 2 channels, 120. A
        // ShuffleNet unit with 2 groups that moves by 2: 46. A dense layer of 4 on 8
        // channels: 104. A decoder with a transposed convolution from 12 to 8 channels,
        // joined to the 8 channels of the shuffle unit's input: 608. A decoder by nearest
        // upsampling, joined to the 8 channels of the stem before its pooling: 400. A
        // ConvNeXt block of 8 channels with a depthwise kernel of 7: 624. Head 18.
        let stages = [176, 120, 46, 104, 608, 400, 624];

        assert_eq!(plan.params, stages.iter().sum::<usize>() + 18);
        assert_eq!(plan.params, 2096);
        assert!(plan.steps.iter().all(|s| s.output.size.len() <= 1));

        let lengths = |f: &dyn Fn(&Layer) -> bool| -> Vec<Vec<usize>> {
            plan.steps
                .iter()
                .filter(|s| f(&s.layer))
                .map(|s| s.output.size.clone())
                .collect()
        };

        assert_eq!(
            lengths(&|l| matches!(l, Layer::MaxPoolSame { .. })),
            vec![vec![32]]
        );
        assert_eq!(
            lengths(&|l| matches!(l, Layer::Shuffle { .. })),
            vec![vec![32]]
        );
        assert_eq!(
            lengths(&|l| matches!(l, Layer::ConvTranspose { .. })),
            vec![vec![32]]
        );
        assert_eq!(
            lengths(&|l| matches!(l, Layer::Upsample { .. })),
            vec![vec![64]]
        );
        assert_eq!(
            lengths(&|l| *l == Layer::Concat),
            vec![vec![32], vec![16], vec![32], vec![64]]
        );
    }

    #[test]
    fn rejects_a_bottleneck_width_the_ratio_does_not_divide() {
        let mut g = testing::vgg();

        cnn_mut(&mut g).stages[1].block = Block::Bottleneck;
        cnn_mut(&mut g).stages[1].width = 10;
        assert_rejected(
            &g,
            TaskKind::Shapes,
            "stage 2: a bottleneck needs a width divisible by 4, not 10",
        );

        cnn_mut(&mut g).stages[1].width = 12;
        assert!(g.cnn_plan(&TaskKind::Shapes.spec()).is_ok());
    }

    #[test]
    fn rejects_groups_that_do_not_divide_a_convolution_of_a_block() {
        // The second VGG stage maps 8 to 16 channels.
        let cases = [
            // The 3x3 convolution of a bottleneck of width 16 has 4 channels.
            (
                Block::Bottleneck,
                3,
                "convolution 2: 3 groups do not divide the 4 input channels",
            ),
            (
                Block::Basic,
                3,
                "convolution 1: 3 groups do not divide the 8 input channels",
            ),
            // Groups apply to the pointwise convolutions of these two blocks.
            (
                Block::Separable,
                3,
                "convolution 2: 3 groups do not divide the 8 input channels",
            ),
            (
                Block::InvertedResidual,
                5,
                "convolution 1: 5 groups do not divide the 8 input channels",
            ),
        ];

        for (block, groups, phrase) in cases {
            let mut g = testing::vgg();

            cnn_mut(&mut g).stages[1].block = block;
            cnn_mut(&mut g).stages[1].groups = groups;
            assert_rejected(&g, TaskKind::Shapes, phrase);
        }

        // Groups that divide both sides fit; the depthwise convolutions keep their own.
        for block in [Block::Separable, Block::InvertedResidual, Block::Basic] {
            let mut g = testing::vgg();

            cnn_mut(&mut g).stages[1].block = block;
            cnn_mut(&mut g).stages[1].groups = 4;
            assert!(g.cnn_plan(&TaskKind::Shapes.spec()).is_ok(), "{block:?}");
        }
    }

    #[test]
    fn rejects_group_norm_on_an_inner_width_it_cannot_split() {
        // A bottleneck of width 8 has 2 inner channels.
        let mut g = testing::vgg();

        cnn_mut(&mut g).stages[1].block = Block::Bottleneck;
        cnn_mut(&mut g).stages[1].width = 8;
        cnn_mut(&mut g).stages[1].norm = Norm::Group;
        assert_rejected(
            &g,
            TaskKind::Shapes,
            "convolution 1: group norm needs a width divisible by 4, not 2",
        );

        // The depthwise convolution of a separable block keeps its 6 input channels.
        let mut g = testing::lenet();

        cnn_mut(&mut g).stages[0].block = Block::Separable;
        cnn_mut(&mut g).stages[0].norm = Norm::Group;
        assert_rejected(
            &g,
            TaskKind::Shapes,
            "convolution 1: group norm needs a width divisible by 4, not 6",
        );

        // An inverted residual expands the 6 channels to 36, which 4 divides.
        cnn_mut(&mut g).stages[0].block = Block::InvertedResidual;
        assert!(g.cnn_plan(&TaskKind::Shapes.spec()).is_ok());
    }

    #[test]
    fn rejects_a_kernel_that_does_not_fit_inside_a_block() {
        // The first convolution of a basic block moves by 4 on 8x8 and leaves 2x2, where
        // the second 5x5 convolution does not fit.
        let mut g = testing::vgg();

        cnn_mut(&mut g).stages[1] = Stage {
            stride: 4,
            down: Downsample::Conv,
            ..plain(16, 5, CnnActivation::Relu)
        };
        assert!(g.cnn_plan(&TaskKind::Shapes.spec()).is_err());

        cnn_mut(&mut g).stages[1].kernel = 3;
        cnn_mut(&mut g).stages.truncate(2);
        assert!(g.cnn_plan(&TaskKind::Shapes.spec()).is_ok());

        cnn_mut(&mut g).stages[1].block = Block::Basic;
        assert_rejected(
            &g,
            TaskKind::Shapes,
            "stage 2: convolution 2: kernel 3 with dilation 1 spans 3 cells, larger than the \
             input of 2",
        );
    }

    #[test]
    fn the_time_model_counts_forks_sums_and_squeeze_and_excitation() {
        let plan = valid_plan(
            &block_genome("plain with squeeze-and-excitation and an identity shortcut"),
            TaskKind::Shapes,
        );

        // Stem: 3x3 cells of 1 channel at 16x16; values 8 x 256 three times (convolution,
        // norm, ReLU), then 8 x 64 after pooling. Stage: 3x3 cells of 8 channels at 8x8;
        // the fork and the branch add no values, the convolution, the norm, the gate, the
        // sum and the ReLU 8 x 64 each, and the gate also its small vectors: 3 x 8 + 2 x 2.
        // Head: 8 means and 4 outputs.
        assert_eq!(
            plan.counts(),
            Counts {
                taps: 9 * 256 + 9 * 8 * 64,
                values: 3 * 8 * 256 + 8 * 64 + 5 * 8 * 64 + (3 * 8 + 2 * 2) + 8 + 4,
                pool_taps: 4 * 8 * 64,
                group_inputs: 0,
                tiles: 2,
                calls: 4 + 2 + SQUEEZE_EXCITE_CALLS + 2 + 2,
            }
        );

        // The projection of a shortcut reads the values of the fork, not those of the
        // block before it: 8 channels at 8x8, which a 1x1 kernel reads once at 4x4.
        let plan = valid_plan(
            &block_genome("basic, strided, with a projection"),
            TaskKind::Shapes,
        );

        assert_eq!(
            plan.counts().taps,
            9 * 256 + 9 * 8 * 16 + 9 * 16 * 16 + 8 * 16
        );
    }

    #[test]
    fn the_time_model_counts_the_layers_of_ticket_04_by_hand() {
        let step = |layer: Layer, channels: usize, size: Vec<usize>| Step {
            layer,
            output: Shape { channels, size },
            params: 0,
            flops: 0,
        };
        let plan = |input: Shape, steps: Vec<Step>| Plan {
            input,
            steps,
            params: 0,
            forward_flops: 0,
            units: Vec::new(),
        };

        // On 4 channels at 8x8: max pooling of 3x3 that keeps the size reads 9 cells for
        // each of its 256 values. A shuffle hands on 256 values. A transposed convolution by
        // 2 to 2 channels counts as the convolution of its backward pass: 2x2 cells of 2
        // channels at each of the 64 input positions, one tile. Then an upsampling by 2 to
        // 32x32 and a join of 6 channels there. Each layer is one call.
        let image = plan(
            Shape {
                channels: 4,
                size: vec![8, 8],
            },
            vec![
                step(Layer::MaxPoolSame { window: 3 }, 4, vec![8, 8]),
                step(Layer::Shuffle { groups: 2 }, 4, vec![8, 8]),
                step(
                    Layer::ConvTranspose {
                        inputs: 4,
                        outputs: 2,
                        factor: 2,
                        bias: false,
                    },
                    2,
                    vec![16, 16],
                ),
                step(Layer::Upsample { factor: 2 }, 2, vec![32, 32]),
                step(Layer::Concat, 6, vec![32, 32]),
            ],
        );

        assert_eq!(
            image.counts(),
            Counts {
                taps: 4 * 2 * 64,
                values: 256 + 256 + 2 * 256 + 2 * 1024 + 6 * 1024,
                pool_taps: 9 * 256,
                group_inputs: 0,
                tiles: 1,
                calls: 5,
            }
        );

        // A causal convolution counts as a convolution with the same genes: 3 cells of 4
        // channels at each of its 8 outputs, 2 groups of one tile each, and 2 copies of the
        // 4 x 16 input values for the groups.
        let causal = Layer::CausalConv {
            inputs: 4,
            outputs: 8,
            kernel: 3,
            stride: 2,
            dilation: 2,
            groups: 2,
            bias: false,
        };
        let conv = Layer::Conv {
            inputs: 4,
            outputs: 8,
            kernel: 3,
            stride: 2,
            padding: 2,
            dilation: 2,
            groups: 2,
            bias: false,
        };
        let sequence = |layer: Layer| {
            plan(
                Shape {
                    channels: 4,
                    size: vec![16],
                },
                vec![step(layer, 8, vec![8])],
            )
            .counts()
        };
        let counts = Counts {
            taps: 3 * 4 * 8,
            values: 8 * 8,
            pool_taps: 0,
            group_inputs: 4 * 16 * 2,
            tiles: 2,
            calls: 2,
        };

        assert_eq!(sequence(causal), counts);
        assert_eq!(sequence(conv), counts);
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

        // Values per row count the input with all 4 channels: 4 x 64, then the layers:
        // 8 x 64 five times, 8 x 32, then 8, 2.
        assert_eq!(plan.values_per_row(), 4 * 64 + 5 * 8 * 64 + 8 * 32 + 8 + 2);
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

        // A stride of 4 in the stem keeps 12 stages within the time budget.
        cnn_mut(&mut g).stem.stride = 4;
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

    /// A genome, its task and the index of a stage that can take `block`: stage 2 of the
    /// VGG-like genome (8 to 16 channels at 8x8) for most kinds, the stage of the 1-D genome
    /// for a TCN block, the last decoder of the U-Net block genome for a decoder.
    fn host(block: Block) -> (Genome, TaskKind, usize) {
        match block {
            Block::Tcn => (testing::sequence(), TaskKind::Motif1d, 0),
            Block::Decoder => (
                block_genome(
                    "a u-net with a transposed convolution, nearest upsampling and a per-cell head",
                ),
                TaskKind::MirrorGrid,
                3,
            ),
            _ => (testing::vgg(), TaskKind::Shapes, 1),
        }
    }

    #[test]
    fn every_block_kind_and_head_is_built() {
        assert_eq!(BLOCKS.len(), 13);

        for block in BLOCKS {
            let (mut g, task, i) = host(block);

            cnn_mut(&mut g).stages[i].block = block;
            g.cnn_plan(&task.spec())
                .unwrap_or_else(|e| panic!("{}: {e}", block.name()));
        }

        for head in [Head::GlobalAvgPool, Head::Flatten, Head::PerCell] {
            let (mut g, task, _) = host(Block::Decoder);

            cnn_mut(&mut g).head = head;
            assert!(g.cnn_plan(&task.spec()).is_ok(), "{head:?}");
        }
    }

    #[test]
    fn every_block_takes_both_options() {
        for block in BLOCKS {
            let count = |se: bool, residual: bool| {
                let (mut g, task, i) = host(block);
                let stage = &mut cnn_mut(&mut g).stages[i];

                stage.block = block;
                stage.se = se;
                stage.residual = residual;

                let plan = g
                    .cnn_plan(&task.spec())
                    .unwrap_or_else(|e| panic!("{block:?}: {e}"));
                let number =
                    |f: &dyn Fn(&Layer) -> bool| plan.steps.iter().filter(|s| f(&s.layer)).count();

                (
                    number(&|l| matches!(l, Layer::SqueezeExcite { .. })),
                    number(&|l| *l == Layer::Add),
                    number(&|l| *l == Layer::Fork),
                )
            };
            // The host of a decoder has the forks of its skip connections already.
            let (gates, sums, forks) = count(false, false);

            for (se, residual) in [(true, false), (false, true), (true, true)] {
                let (g, a, f) = count(se, residual);

                assert_eq!(g, gates + usize::from(se), "{block:?}");
                assert_eq!(a, sums + usize::from(residual), "{block:?}");
                assert_eq!(f, forks + usize::from(residual), "{block:?}");
            }
        }
    }

    #[test]
    fn rejects_the_rules_of_the_blocks_of_ticket_04() {
        // Stage 2 of the VGG-like genome maps 8 to 16 channels at 8x8, then max-pools by 2.
        let set = |block: Block, change: &dyn Fn(&mut Stage)| {
            let mut g = testing::vgg();
            let stage = &mut cnn_mut(&mut g).stages[1];

            stage.block = block;
            change(stage);

            g
        };
        let cases: Vec<(Genome, TaskKind, &str)> = vec![
            (
                set(Block::Inception, &|s| s.width = 10),
                TaskKind::Shapes,
                "stage 2: an inception block needs a width divisible by 4, not 10",
            ),
            (
                set(Block::Inception, &|s| s.kernel = 11),
                TaskKind::Shapes,
                "stage 2: an inception block has a second kernel of 13, its kernel plus 2, \
                 > max 11",
            ),
            (
                set(Block::Fire, &|s| s.width = 6),
                TaskKind::Shapes,
                "stage 2: a fire module needs a width divisible by 4, not 6",
            ),
            (
                set(Block::Shuffle, &|s| s.width = 10),
                TaskKind::Shapes,
                "stage 2: a shufflenet unit needs a width divisible by 4, not 10",
            ),
            // A ShuffleNet unit of width 16 has 4 inner channels.
            (
                set(Block::Shuffle, &|s| s.groups = 8),
                TaskKind::Shapes,
                "stage 2: convolution 1: 8 groups do not divide the width 4",
            ),
            (
                set(Block::Dense, &|s| s.down = Downsample::Conv),
                TaskKind::Shapes,
                "stage 2: a dense layer joins its input, so it keeps the size and downsamples \
                 by pooling only",
            ),
            (
                set(Block::Dense, &|s| s.width = 250),
                TaskKind::Shapes,
                "stage 2: a dense layer adds its width 250 to its 8 input channels: 258 > max 256",
            ),
            // ConvNeXt normalizes after its depthwise convolution only, on 8 channels here.
            (
                set(Block::ConvNext, &|s| {
                    s.norm = Norm::Group;
                    s.width = 6;
                }),
                TaskKind::Shapes,
                "",
            ),
            // With a skip connection, the projection from 8 to 6 channels has a norm as well.
            (
                set(Block::ConvNext, &|s| {
                    s.norm = Norm::Group;
                    s.width = 6;
                    s.residual = true;
                }),
                TaskKind::Shapes,
                "stage 2: group norm needs a width divisible by 4, not 6",
            ),
            (
                set(Block::Tcn, &|_| {}),
                TaskKind::Shapes,
                "stage 2: a TCN block convolves causally along a sequence, so it needs a 1-D \
                 genome",
            ),
            (
                set(Block::Decoder, &|s| s.stride = 1),
                TaskKind::Shapes,
                "stage 2: a decoder upsamples by its stride, so the stride must be at least 2, \
                 not 1",
            ),
        ];

        for (g, task, phrase) in cases {
            if phrase.is_empty() {
                assert!(g.cnn_plan(&task.spec()).is_ok(), "{}", g.summary());
            } else {
                assert_rejected(&g, task, phrase);
            }
        }

        // The stem pools 16x16 to 8x8, and stage 1 keeps 8x8: only the stem has a skip
        // point. A first decoder joins it; a second finds none.
        let mut g = testing::vgg();

        cnn_mut(&mut g).stages.truncate(1);
        cnn_mut(&mut g).stages.push(Stage {
            stride: 2,
            ..plain(8, 3, CnnActivation::Relu)
        });
        cnn_mut(&mut g).stages[1].block = Block::Decoder;
        assert!(g.cnn_plan(&TaskKind::Shapes.spec()).is_ok());

        let decoder = cnn_mut(&mut g).stages[1].clone();

        cnn_mut(&mut g).stages.push(decoder);
        assert_rejected(
            &g,
            TaskKind::Shapes,
            "stage 3: a decoder joins the map of an earlier stage that downsampled, and none \
             is left",
        );

        // A decoder that upsamples by 4 does not land on the 16x16 map of the stem's skip
        // point: 8 x 4 = 32.
        cnn_mut(&mut g).stages.truncate(2);
        cnn_mut(&mut g).stages[1].stride = 4;
        assert_rejected(
            &g,
            TaskKind::Shapes,
            "stage 2: the decoder's upsampled map has [32, 32] cells, the map it joins [16, 16]",
        );
    }

    #[test]
    fn the_per_cell_head_needs_a_grid_task_and_a_map_of_its_size() {
        let g = block_genome(
            "a u-net with a transposed convolution, nearest upsampling and a per-cell head",
        );

        assert!(g.cnn_plan(&TaskKind::MirrorGrid.spec()).is_ok());
        assert_rejected(
            &g,
            TaskKind::Shapes,
            "head: the per-cell head needs a task with a grid of outputs, and task shapes \
             gives a vector of 4",
        );

        // Without its decoders the map stays at the 2x2 of the bottom.
        let mut short = g.clone();

        cnn_mut(&mut short).stages.truncate(2);
        assert_rejected(
            &short,
            TaskKind::MirrorGrid,
            "head: the per-cell head needs the last map at the 8x8 cells of the output grid, \
             not at [2, 2]",
        );

        // A dense head writes the 64 values of a grid as well.
        let mut dense = g;

        cnn_mut(&mut dense).head = Head::GlobalAvgPool;

        let plan = valid_plan(&dense, TaskKind::MirrorGrid);

        assert_eq!(
            plan.steps.last().map(|s| s.output.clone()),
            Some(Shape::vector(64))
        );

        // A grid that does not hold the outputs of its task.
        let mut spec = TaskKind::MirrorGrid.spec();

        spec.n_out = 63;

        let message = block_genome(
            "a u-net with a transposed convolution, nearest upsampling and a per-cell head",
        )
        .cnn_plan(&spec)
        .expect_err("refused")
        .to_string();

        assert!(
            message.contains("does not hold its 63 outputs"),
            "{message}"
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

    /// A CNN of wide 3x3 convolutions, so that its FLOPs, not its time, reach the budget
    /// first: 32 channels, three stages with batch norm, at 16x16.
    fn wide() -> Genome {
        let cnn = CnnGenome {
            dims: Dims::TwoD,
            stem: Stem {
                width: 32,
                kernel: 3,
                stride: 1,
                down: Downsample::Conv,
                norm: Norm::Batch,
                activation: CnnActivation::Relu,
            },
            stages: vec![
                Stage {
                    norm: Norm::Batch,
                    ..plain(32, 3, CnnActivation::Relu)
                };
                3
            ],
            head: Head::GlobalAvgPool,
        };
        let mut g = testing::genome(cnn, Vec::new(), Activation::Relu);

        g.epochs = 5;

        g
    }

    #[test]
    fn rejects_too_many_flops() {
        let g = wide();
        let spec = TaskKind::Shapes.spec();
        let flops = g.train_flops(&spec);
        let plan = g.cnn_plan(&spec).expect("plan");
        let seconds = plan.train_seconds(5000.0);

        // The FLOP rule binds first for this genome: its time fits the FLOP budget.
        assert!(
            seconds < flops as f64 / TIME_BUDGET_FLOPS_PER_SECOND,
            "{seconds} s"
        );

        let limits = Limits {
            max_params: 100_000,
            max_train_flops: flops - 1,
            ..Limits::default()
        };

        assert_rejected_with(&g, TaskKind::Shapes, &limits, "train FLOPs > max");

        let limits = Limits {
            max_params: 100_000,
            max_train_flops: flops,
            ..Limits::default()
        };

        assert!(g.validate(&limits, &spec).is_ok());
    }

    #[test]
    fn the_time_model_counts_by_hand() {
        // LeNet, as in `lenet_counts_match_a_hand_count`.
        let lenet = valid_plan(&testing::lenet(), TaskKind::Shapes);

        // Taps: 5x5 cells of 1 channel at 16x16, and 5x5 cells of 6 channels at 8x8.
        // Values: 6 x 256 twice (convolution, tanh), 6 x 64, 16 x 64 twice, 16 x 16,
        // 256 (flatten), 32 twice, 4. Pooling: 2x2 cells per output value, 6 x 64 and
        // 16 x 16 of them. One matrix product per convolution, 10 layers, no groups.
        assert_eq!(
            lenet.counts(),
            Counts {
                taps: 25 * 256 + 25 * 6 * 64,
                values: 2 * 6 * 256 + 6 * 64 + 2 * 16 * 64 + 16 * 16 + 256 + 2 * 32 + 4,
                pool_taps: 4 * 6 * 64 + 4 * 16 * 16,
                group_inputs: 0,
                tiles: 2,
                calls: 10,
            }
        );

        // Groups: 8 -> 16 in 4 groups at 16x16 (output 8x8), then 16 -> 16 in 16 groups at
        // 8x8. Each group reads the whole input once in the backward pass and is its own
        // matrix product; each group is a call.
        let grouped = valid_plan(&testing::grouped_dilated(), TaskKind::Shapes).counts();

        assert_eq!(grouped.taps, 9 * 256 + 9 * 8 * 64 + 9 * 16 * 64);
        assert_eq!(grouped.group_inputs, 4 * 8 * 256 + 16 * 16 * 64);
        assert_eq!(grouped.tiles, 1 + 4 + 16);
        assert_eq!(grouped.calls, 1 + 1 + 4 + 2 + 16 + 2 + 1 + 3);

        // The estimates are these counts at their prices.
        let c = lenet.counts();
        let train = TRAIN_NANOS_PER_FLOP * lenet.train_flops_per_sample() as f64
            + TRAIN_NANOS_PER_TAP * c.taps as f64
            + TRAIN_NANOS_PER_VALUE * c.values as f64
            + TRAIN_NANOS_PER_POOL_TAP * c.pool_taps as f64
            + TRAIN_NANOS_PER_TILE * c.tiles as f64;

        assert_eq!(lenet.train_nanos_per_sample(), train);

        // 2000 rows in chunks of 512: 4 chunks.
        let per_row = EVAL_NANOS_PER_FLOP * lenet.forward_flops as f64
            + EVAL_NANOS_PER_TAP * c.taps as f64
            + EVAL_NANOS_PER_VALUE * c.values as f64
            + EVAL_NANOS_PER_POOL_TAP * c.pool_taps as f64
            + EVAL_NANOS_PER_TILE * c.tiles as f64;

        assert_eq!(
            lenet.eval_nanos(2000, 512),
            2000.0 * per_row + 4.0 * EVAL_NANOS_PER_CALL * c.calls as f64
        );

        // Values per row: the input, then every layer.
        assert_eq!(lenet.values_per_row() as u64, 256 + c.values);
    }

    /// The two slow CNNs of the review of ticket 02, with as many epochs as
    /// `max_train_flops` allows: before the time rule they passed `validate` and then took
    /// 30 s at one thread, so the clock stopped them.
    fn slow_cnns_of_the_review() -> Vec<Genome> {
        let narrow = |width: usize, kernel: usize, norm: Norm| {
            let stage = Stage {
                norm,
                ..plain(width, kernel, CnnActivation::Relu)
            };
            let cnn = CnnGenome {
                dims: Dims::TwoD,
                stem: Stem {
                    width,
                    kernel,
                    stride: 1,
                    down: Downsample::Conv,
                    norm,
                    activation: CnnActivation::Relu,
                },
                stages: vec![stage; 12],
                head: Head::GlobalAvgPool,
            };
            let mut g = testing::genome(cnn, Vec::new(), Activation::Relu);
            let spec = TaskKind::Shapes.spec();
            let per_epoch =
                g.cnn_plan(&spec).expect("plan").train_flops_per_sample() * spec.n_train as u64;

            g.batch_size = 16;
            g.epochs = (Limits::default().max_train_flops / per_epoch) as usize;

            g
        };

        vec![narrow(1, 11, Norm::None), narrow(4, 3, Norm::Batch)]
    }

    #[test]
    fn rejects_a_cnn_too_slow_for_its_flop_budget() {
        let spec = TaskKind::Shapes.spec();

        for g in slow_cnns_of_the_review() {
            assert!(g.epochs >= 6, "{}", g.summary());
            assert!(g.train_flops(&spec) <= Limits::default().max_train_flops);
            assert_rejected_with(
                &g,
                TaskKind::Shapes,
                &Limits::default(),
                "estimated training time",
            );
        }

        // LeNet with 10 epochs: its FLOPs fit 1.3e10, but its time does not.
        let lenet = testing::lenet();
        let flops = lenet.train_flops(&spec);
        let plan = valid_plan(&lenet, TaskKind::Shapes);
        let seconds = plan.train_seconds(10_000.0);
        let budget = |seconds: f64| Limits {
            max_train_flops: (seconds * TIME_BUDGET_FLOPS_PER_SECOND) as u64,
            ..Limits::default()
        };

        assert!(flops < budget(seconds * 0.99).max_train_flops);
        assert_rejected_with(
            &lenet,
            TaskKind::Shapes,
            &budget(seconds * 0.99),
            "estimated training time",
        );
        assert!(lenet.validate(&budget(seconds * 1.01), &spec).is_ok());
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
        // Valid genomes with a block of another kind than plain, or with an option.
        let mut valid_blocks = 0;
        let uses_blocks = |g: &Genome| match &g.arch {
            Arch::Cnn(cnn) => cnn
                .stages
                .iter()
                .any(|s| s.block != Block::Plain || s.se || s.residual),
            Arch::Mlp => false,
        };

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
                    valid_blocks += usize::from(uses_blocks(&g));

                    let plan = g.cnn_plan(&spec).expect("a valid genome has a plan");
                    let samples = spec.n_train as f64 * g.epochs as f64;

                    assert_eq!(g.param_count(&spec), plan.params);
                    assert!(plan.params <= limits.max_params);
                    assert!(g.train_flops(&spec) <= limits.max_train_flops);
                    assert!(
                        plan.train_seconds(samples)
                            <= limits.max_train_flops as f64 / TIME_BUDGET_FLOPS_PER_SECOND
                    );
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

        // Both outcomes occur often, so the test covers the valid and the invalid paths,
        // also for the blocks and options of ticket 03.
        eprintln!("{valid} valid ({valid_blocks} with blocks or options), {invalid} invalid");
        assert!(
            valid > 300 && valid_blocks > 100 && invalid > 8000,
            "{valid} valid, {valid_blocks} with blocks or options, {invalid} invalid"
        );
    }
}
