//! The CNN genome: an input stem, a list of stages and a head.
//!
//! A CNN genome is plain data, like the MLP genome. It sits in `Genome::arch`. The shared
//! training genes of `Genome` (optimizer, learning rate, l2, batch size, epochs, init scale
//! and seed) train it. `Genome::hidden` gives the dense layers of its head, and
//! `Genome::activation` their activation.
//!
//! The genome declares the parts of all 16 families of the spec:
//!
//! | Family | Part |
//! |---|---|
//! | LeNet-5, AlexNet, VGG | `Block::Plain` with downsampling, `Head::Flatten` |
//! | Dilated CNN | `Block::Plain` with a dilation above 1 |
//! | ResNet | `Block::Basic`, `Block::Bottleneck`, `Stage::residual` |
//! | Xception, MobileNet | `Block::Separable` |
//! | MobileNetV2 | `Block::InvertedResidual` |
//! | SENet | `Stage::se` |
//! | GoogLeNet | `Block::Inception` |
//! | DenseNet | `Block::Dense`, `Block::Transition` |
//! | SqueezeNet | `Block::Fire` |
//! | ShuffleNet | `Block::Shuffle` |
//! | ConvNeXt | `Block::ConvNext` |
//! | TCN | `Block::Tcn` |
//! | U-Net | `Block::Decoder`, `Head::PerCell` |
//!
//! Ticket 02 builds the plain stage, every downsampling, every normalization, the six
//! activations and the heads `GlobalAvgPool` and `Flatten`. Tickets 03 and 04 build the
//! other parts. Until then the shape check (`CnnGenome::plan` in `shape`) rejects them as
//! not yet built.

use serde::{Deserialize, Serialize};

use crate::genome::Activation;

/// Most stages after the stem.
pub const MAX_STAGES: usize = 12;

/// Most channels of the stem or of one stage.
pub const MAX_CHANNELS: usize = 256;

/// Largest kernel side. A kernel is odd, so its padding keeps the size at stride 1.
pub const MAX_KERNEL: usize = 11;

/// Largest stride, the downsampling factor of the stem or of one stage.
pub const MAX_STRIDE: usize = 4;

/// Largest dilation of a convolution.
pub const MAX_DILATION: usize = 16;

/// Groups of a group norm. The width it normalizes must be a multiple of it.
pub const GROUP_NORM_GROUPS: usize = 4;

/// Every block kind, in declaration order.
pub const BLOCKS: [Block; 13] = [
    Block::Plain,
    Block::Basic,
    Block::Bottleneck,
    Block::Separable,
    Block::InvertedResidual,
    Block::Inception,
    Block::Dense,
    Block::Transition,
    Block::Fire,
    Block::Shuffle,
    Block::ConvNext,
    Block::Tcn,
    Block::Decoder,
];

/// Every activation of the stem and the stages: the four of the MLP, then SiLU and ReLU6.
pub const CNN_ACTIVATIONS: [CnnActivation; 6] = [
    CnnActivation::Relu,
    CnnActivation::Tanh,
    CnnActivation::Gelu,
    CnnActivation::Sine,
    CnnActivation::Silu,
    CnnActivation::Relu6,
];

/// Every normalization.
pub const NORMS: [Norm; 4] = [Norm::None, Norm::Batch, Norm::Layer, Norm::Group];

/// Every way to downsample.
pub const DOWNSAMPLES: [Downsample; 3] =
    [Downsample::Conv, Downsample::MaxPool, Downsample::AvgPool];

/// A CNN: an input stem, a list of stages and a head.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CnnGenome {
    /// 1-D convolutions for a sequence task, 2-D convolutions for an image task.
    pub dims: Dims,
    /// The first convolution. It maps the input channels to its width.
    pub stem: Stem,
    /// The stages after the stem, input to output order. There is at least one.
    pub stages: Vec<Stage>,
    /// How the last stage's output becomes the input of the dense layers.
    pub head: Head,
}

/// Whether the convolutions run along a sequence or over an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dims {
    /// A kernel of size k is a 1 x k window along the sequence.
    OneD,
    /// A kernel of size k is a k x k window.
    TwoD,
}

/// The stem: one convolution, then the normalization and the activation.
///
/// It is built like a plain stage without dilation and groups (see `shape`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stem {
    /// Output channels.
    pub width: usize,
    /// Kernel side, odd.
    pub kernel: usize,
    /// Downsampling factor; 1 keeps the size.
    pub stride: usize,
    /// How a stride above 1 downsamples. It has no effect at stride 1.
    pub down: Downsample,
    /// Normalization after the convolution.
    pub norm: Norm,
    /// Activation after the normalization.
    pub activation: CnnActivation,
}

/// One stage: a block of one family with its genes.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub block: Block,
    /// Output channels.
    pub width: usize,
    /// Kernel side, odd.
    pub kernel: usize,
    /// Downsampling factor; 1 keeps the size.
    pub stride: usize,
    /// How a stride above 1 downsamples. It has no effect at stride 1.
    pub down: Downsample,
    /// Gap between the cells a kernel reads; 1 reads neighbours.
    pub dilation: usize,
    /// Groups of the convolution. They must divide the input channels and the width.
    pub groups: usize,
    /// Squeeze-and-excitation on the block's output (SENet). Ticket 03 builds it.
    pub se: bool,
    /// A skip connection around the block (ResNet and others). Ticket 03 builds it.
    pub residual: bool,
    /// Normalization after the convolution.
    pub norm: Norm,
    /// Activation after the normalization.
    pub activation: CnnActivation,
}

/// The block kind of a stage. Each kind names the family it comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Block {
    /// One convolution, then the normalization and the activation: LeNet-5, AlexNet, VGG,
    /// and with a dilation above 1 the dilated CNN. Built.
    Plain,
    /// ResNet basic block: two convolutions and a skip connection. Ticket 03.
    Basic,
    /// ResNet bottleneck: a 1x1 reduction, a k x k convolution, a 1x1 expansion and a skip
    /// connection. Ticket 03.
    Bottleneck,
    /// Depthwise separable convolution: a depthwise k x k, then a pointwise 1x1 convolution
    /// (Xception, MobileNet). Ticket 03.
    Separable,
    /// MobileNetV2 inverted residual with a linear bottleneck: a 1x1 expansion, a depthwise
    /// convolution and a 1x1 projection without activation. Ticket 03.
    InvertedResidual,
    /// GoogLeNet Inception block: parallel 1x1, 3x3, 5x5 and pooling branches, concatenated.
    /// Ticket 04.
    Inception,
    /// DenseNet dense block: every layer's output is concatenated to its input. Ticket 04.
    Dense,
    /// DenseNet transition: a 1x1 convolution and average pooling between dense blocks.
    /// Ticket 04.
    Transition,
    /// SqueezeNet Fire module: a 1x1 squeeze, then 1x1 and 3x3 expansions, concatenated.
    /// Ticket 04.
    Fire,
    /// ShuffleNet unit: group convolutions, a channel shuffle and a depthwise convolution.
    /// Ticket 04.
    Shuffle,
    /// ConvNeXt block: a depthwise 7x7, layer norm, an inverted MLP with GELU and a skip
    /// connection. Ticket 04.
    ConvNext,
    /// TCN block: causal dilated 1-D convolutions with a skip connection. Ticket 04.
    Tcn,
    /// U-Net decoder stage: upsampling, concatenation with the encoder output of the same
    /// size, and a convolution. Ticket 04.
    Decoder,
}

impl Block {
    /// True when this block kind is built, so the shape check accepts it. Tickets 03 and
    /// 04 build the others.
    pub fn is_built(self) -> bool {
        matches!(self, Block::Plain)
    }

    /// Short name for summaries and error messages.
    pub fn name(self) -> &'static str {
        match self {
            Block::Plain => "plain",
            Block::Basic => "basic",
            Block::Bottleneck => "bottleneck",
            Block::Separable => "separable",
            Block::InvertedResidual => "inverted-residual",
            Block::Inception => "inception",
            Block::Dense => "dense",
            Block::Transition => "transition",
            Block::Fire => "fire",
            Block::Shuffle => "shuffle",
            Block::ConvNext => "convnext",
            Block::Tcn => "tcn",
            Block::Decoder => "decoder",
        }
    }
}

/// How a stride above 1 shrinks the output of the stem or of a stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Downsample {
    /// The convolution itself moves by the stride.
    Conv,
    /// Max pooling after the activation, with a window and a step of `stride` cells.
    MaxPool,
    /// Average pooling after the activation, with a window and a step of `stride` cells.
    AvgPool,
}

/// The normalization after a convolution. Each kind has a scale and a shift per channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Norm {
    /// No normalization; the convolution has a bias instead.
    None,
    /// Batch norm (Ioffe and Szegedy 2015): each channel over the batch and the positions.
    Batch,
    /// Layer norm over the channels at each position, as in ConvNeXt (Ba, Kiros and Hinton
    /// 2016, preprint).
    Layer,
    /// Group norm (Wu and He 2018): `GROUP_NORM_GROUPS` groups of channels, each over its
    /// channels and the positions.
    Group,
}

/// Activation of the stem and of the stages.
///
/// The first four are the activations of the MLP, with the same formulas. The dense
/// layers of the head use `Genome::activation`, one of those four.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CnnActivation {
    Relu,
    Tanh,
    /// GELU with the tanh approximation, as in the MLP.
    Gelu,
    Sine,
    /// x * sigmoid(x), also called swish.
    Silu,
    /// min(max(x, 0), 6), as in MobileNetV2.
    Relu6,
}

impl From<Activation> for CnnActivation {
    fn from(activation: Activation) -> Self {
        match activation {
            Activation::Relu => CnnActivation::Relu,
            Activation::Tanh => CnnActivation::Tanh,
            Activation::Gelu => CnnActivation::Gelu,
            Activation::Sine => CnnActivation::Sine,
        }
    }
}

/// How the last stage's output becomes the input of the dense layers of the head.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Head {
    /// The mean of each channel over all positions.
    GlobalAvgPool,
    /// Every value of every channel at every position, as in LeNet-5 and VGG.
    Flatten,
    /// A 1x1 convolution per cell for grid-to-grid tasks (U-Net). Ticket 04.
    PerCell,
}

/// The genes of one convolution: of the stem, or of a plain stage.
///
/// The shape check and the summary read the stem and the stages through it. The stem
/// has a dilation of 1 and one group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ConvGenes {
    pub(crate) width: usize,
    pub(crate) kernel: usize,
    pub(crate) stride: usize,
    pub(crate) down: Downsample,
    pub(crate) dilation: usize,
    pub(crate) groups: usize,
    pub(crate) norm: Norm,
    pub(crate) activation: CnnActivation,
}

impl Stem {
    /// The genes of its convolution.
    pub(crate) fn conv_genes(&self) -> ConvGenes {
        ConvGenes {
            width: self.width,
            kernel: self.kernel,
            stride: self.stride,
            down: self.down,
            dilation: 1,
            groups: 1,
            norm: self.norm,
            activation: self.activation,
        }
    }
}

impl Stage {
    /// The genes of its convolution.
    pub(crate) fn conv_genes(&self) -> ConvGenes {
        ConvGenes {
            width: self.width,
            kernel: self.kernel,
            stride: self.stride,
            down: self.down,
            dilation: self.dilation,
            groups: self.groups,
            norm: self.norm,
            activation: self.activation,
        }
    }
}

impl CnnGenome {
    /// One-line summary, e.g.
    /// `cnn2d stem(w6 k5 s2avg tanh) plain(w16 k5 s2avg tanh) head=flatten`.
    ///
    /// A stride, dilation or group count of 1, no normalization, and the switched-off
    /// options are left out.
    pub fn summary(&self) -> String {
        let dims = match self.dims {
            Dims::OneD => "cnn1d",
            Dims::TwoD => "cnn2d",
        };
        let mut parts = vec![
            dims.to_string(),
            format!("stem({})", self.stem.conv_genes().summary()),
        ];

        for stage in &self.stages {
            let mut text = stage.conv_genes().summary();

            if stage.se {
                text.push_str(" se");
            }

            if stage.residual {
                text.push_str(" res");
            }

            parts.push(format!("{}({text})", stage.block.name()));
        }

        let head = match self.head {
            Head::GlobalAvgPool => "gap",
            Head::Flatten => "flatten",
            Head::PerCell => "per-cell",
        };

        parts.push(format!("head={head}"));

        parts.join(" ")
    }
}

impl ConvGenes {
    /// The genes as text, e.g. `w16 k3 s2max d2 g4 bn relu`.
    fn summary(&self) -> String {
        let mut text = format!("w{} k{}", self.width, self.kernel);

        if self.stride != 1 {
            let how = match self.down {
                Downsample::Conv => "conv",
                Downsample::MaxPool => "max",
                Downsample::AvgPool => "avg",
            };

            text.push_str(&format!(" s{}{how}", self.stride));
        }

        if self.dilation != 1 {
            text.push_str(&format!(" d{}", self.dilation));
        }

        if self.groups != 1 {
            text.push_str(&format!(" g{}", self.groups));
        }

        let norm = match self.norm {
            Norm::None => "",
            Norm::Batch => " bn",
            Norm::Layer => " ln",
            Norm::Group => " gn",
        };
        let activation = match self.activation {
            CnnActivation::Relu => "relu",
            CnnActivation::Tanh => "tanh",
            CnnActivation::Gelu => "gelu",
            CnnActivation::Sine => "sine",
            CnnActivation::Silu => "silu",
            CnnActivation::Relu6 => "relu6",
        };

        format!("{text}{norm} {activation}")
    }
}

/// Genomes for the tests of the crate: hand-made CNNs, and arbitrary ones for robustness tests.
#[cfg(test)]
pub(crate) mod testing {
    use rand::Rng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::genome::{Arch, Genome, Optimizer};

    /// A genome with these parts and fixed training genes (Adam, 10 epochs).
    pub(crate) fn genome(cnn: CnnGenome, hidden: Vec<usize>, activation: Activation) -> Genome {
        Genome {
            arch: Arch::Cnn(cnn),
            hidden,
            activation,
            optimizer: Optimizer::Adam,
            learning_rate: 0.01,
            l2: 0.0,
            batch_size: 32,
            epochs: 10,
            init_scale: 1.0,
            seed: 7,
        }
    }

    /// A plain stage with stride 1, dilation 1, one group, and no normalization.
    pub(crate) fn plain(width: usize, kernel: usize, activation: CnnActivation) -> Stage {
        Stage {
            block: Block::Plain,
            width,
            kernel,
            stride: 1,
            down: Downsample::Conv,
            dilation: 1,
            groups: 1,
            se: false,
            residual: false,
            norm: Norm::None,
            activation,
        }
    }

    /// LeNet-like, for a 1 x 16 x 16 image: two 5x5 convolutions, each followed by 2x2
    /// average pooling, then flatten and dense layers of 32 and the classes, with tanh.
    pub(crate) fn lenet() -> Genome {
        let cnn = CnnGenome {
            dims: Dims::TwoD,
            stem: Stem {
                width: 6,
                kernel: 5,
                stride: 2,
                down: Downsample::AvgPool,
                norm: Norm::None,
                activation: CnnActivation::Tanh,
            },
            stages: vec![Stage {
                stride: 2,
                down: Downsample::AvgPool,
                ..plain(16, 5, CnnActivation::Tanh)
            }],
            head: Head::Flatten,
        };

        genome(cnn, vec![32], Activation::Tanh)
    }

    /// VGG-like, for a 1 x 16 x 16 image: 3x3 convolutions with batch norm and ReLU, two of
    /// them followed by 2x2 max pooling, then flatten and one dense layer to the classes.
    pub(crate) fn vgg() -> Genome {
        let pooled = |width| Stage {
            stride: 2,
            down: Downsample::MaxPool,
            norm: Norm::Batch,
            ..plain(width, 3, CnnActivation::Relu)
        };
        let cnn = CnnGenome {
            dims: Dims::TwoD,
            stem: Stem {
                width: 8,
                kernel: 3,
                stride: 1,
                down: Downsample::MaxPool,
                norm: Norm::Batch,
                activation: CnnActivation::Relu,
            },
            stages: vec![
                Stage {
                    norm: Norm::Batch,
                    ..plain(8, 3, CnnActivation::Relu)
                },
                pooled(16),
                pooled(16),
            ],
            head: Head::Flatten,
        };
        let mut g = genome(cnn, Vec::new(), Activation::Relu);

        g.epochs = 5;

        g
    }

    /// Groups and dilation, for a 1 x 16 x 16 image: a strided, dilated group convolution
    /// with group norm and SiLU, then a dilated depthwise convolution with layer norm and
    /// ReLU6, then global average pooling and dense layers of 8 and the classes, with GELU.
    pub(crate) fn grouped_dilated() -> Genome {
        let cnn = CnnGenome {
            dims: Dims::TwoD,
            stem: Stem {
                width: 8,
                kernel: 3,
                stride: 1,
                down: Downsample::Conv,
                norm: Norm::None,
                activation: CnnActivation::Relu,
            },
            stages: vec![
                Stage {
                    stride: 2,
                    down: Downsample::Conv,
                    dilation: 2,
                    groups: 4,
                    norm: Norm::Group,
                    ..plain(16, 3, CnnActivation::Silu)
                },
                Stage {
                    dilation: 3,
                    groups: 16,
                    norm: Norm::Layer,
                    ..plain(16, 3, CnnActivation::Relu6)
                },
            ],
            head: Head::GlobalAvgPool,
        };

        genome(cnn, vec![8], Activation::Gelu)
    }

    /// 1-D, for a 4 x 64 one-hot sequence: a convolution of 5, then a dilated convolution
    /// of 3 with batch norm and max pooling by 2, then global average pooling and a dense
    /// layer to the classes.
    pub(crate) fn sequence() -> Genome {
        let cnn = CnnGenome {
            dims: Dims::OneD,
            stem: Stem {
                width: 8,
                kernel: 5,
                stride: 1,
                down: Downsample::Conv,
                norm: Norm::None,
                activation: CnnActivation::Relu,
            },
            stages: vec![Stage {
                stride: 2,
                down: Downsample::MaxPool,
                dilation: 4,
                norm: Norm::Batch,
                ..plain(8, 3, CnnActivation::Relu)
            }],
            head: Head::GlobalAvgPool,
        };

        genome(cnn, Vec::new(), Activation::Relu)
    }

    /// A CNN genome of plain stages with every gene in its range.
    ///
    /// Widths are multiples of 4, so 1, 2 or 4 groups and group norm always fit. Many are
    /// valid on the tasks of their dims; some pool too often, read too far, or cost too
    /// much.
    pub(crate) fn plausible(rng: &mut ChaCha8Rng, dims: Dims) -> Genome {
        let unit = |rng: &mut ChaCha8Rng| Stage {
            block: Block::Plain,
            width: one_of(rng, &[4, 8, 12, 16]),
            kernel: one_of(rng, &[1, 3, 3, 5]),
            stride: one_of(rng, &[1, 1, 2]),
            down: one_of(rng, &DOWNSAMPLES),
            dilation: one_of(rng, &[1, 1, 2]),
            groups: one_of(rng, &[1, 1, 2, 4]),
            se: false,
            residual: false,
            norm: one_of(rng, &NORMS),
            activation: one_of(rng, &CNN_ACTIVATIONS),
        };
        let stem = as_stem(&unit(rng));
        let n_stages = rng.random_range(1..=4);
        let cnn = CnnGenome {
            dims,
            stem,
            stages: (0..n_stages).map(|_| unit(rng)).collect(),
            head: one_of(rng, &[Head::GlobalAvgPool, Head::Flatten]),
        };
        let n_hidden = rng.random_range(0..=2);
        let hidden = (0..n_hidden)
            .map(|_| one_of(rng, &[4, 8, 16, 32]))
            .collect();
        let activation = one_of(rng, &crate::genome::ACTIVATIONS);
        let mut g = genome(cnn, hidden, activation);

        g.optimizer = one_of(rng, &crate::genome::OPTIMIZERS);
        g.learning_rate = one_of(rng, &[0.001, 0.01, 0.1]);
        g.l2 = one_of(rng, &[0.0, 1e-4]);
        g.batch_size = one_of(rng, &crate::genome::BATCH_SIZES);
        g.epochs = one_of(rng, &[5, 8, 12]);
        g.init_scale = one_of(rng, &[0.5, 1.0, 1.5]);
        g.seed = rng.random();

        g
    }

    /// The stem genes of a stage: all but block, dilation, groups and the options.
    fn as_stem(stage: &Stage) -> Stem {
        Stem {
            width: stage.width,
            kernel: stage.kernel,
            stride: stage.stride,
            down: stage.down,
            norm: stage.norm,
            activation: stage.activation,
        }
    }

    /// One of the values, at random.
    fn one_of<T: Copy>(rng: &mut ChaCha8Rng, values: &[T]) -> T {
        values[rng.random_range(0..values.len())]
    }

    /// A count that is mostly small, sometimes 0, large, or `usize::MAX`.
    fn count(rng: &mut ChaCha8Rng, small: usize) -> usize {
        match rng.random_range(0..20) {
            0 => 0,
            1 => rng.random_range(small..=small * 40),
            2 => usize::MAX,
            3 => usize::MAX / 2 + 1,
            _ => rng.random_range(1..=small),
        }
    }

    /// A CNN genome with arbitrary genes: mostly in range, sometimes far out of it.
    ///
    /// Many of them break a shape rule or a limit; the point is that none of them panics.
    pub(crate) fn arbitrary(rng: &mut ChaCha8Rng) -> Genome {
        let unit = |rng: &mut ChaCha8Rng| Stage {
            block: if rng.random_bool(0.8) {
                Block::Plain
            } else {
                one_of(rng, &BLOCKS)
            },
            width: count(rng, 24),
            kernel: count(rng, 7),
            stride: count(rng, 3),
            down: one_of(rng, &DOWNSAMPLES),
            dilation: count(rng, 3),
            groups: if rng.random_bool(0.6) {
                1
            } else {
                count(rng, 8)
            },
            se: rng.random_bool(0.05),
            residual: rng.random_bool(0.05),
            norm: one_of(rng, &NORMS),
            activation: one_of(rng, &CNN_ACTIVATIONS),
        };
        let stem = as_stem(&unit(rng));
        let n_stages = rng.random_range(0..=MAX_STAGES + 2);
        let stages = (0..n_stages).map(|_| unit(rng)).collect();
        let cnn = CnnGenome {
            dims: if rng.random_bool(0.5) {
                Dims::OneD
            } else {
                Dims::TwoD
            },
            stem,
            stages,
            head: if rng.random_bool(0.9) {
                one_of(rng, &[Head::GlobalAvgPool, Head::Flatten])
            } else {
                Head::PerCell
            },
        };
        let n_hidden = rng.random_range(0..=5);
        let hidden = (0..n_hidden).map(|_| count(rng, 48)).collect();
        let activation = one_of(rng, &crate::genome::ACTIVATIONS);
        let mut g = genome(cnn, hidden, activation);

        g.optimizer = one_of(rng, &crate::genome::OPTIMIZERS);
        g.learning_rate = one_of(rng, &[0.01, 1e-5, 2.0, f32::NAN, f32::INFINITY]);
        g.l2 = one_of(rng, &[0.0, 1e-4, -1.0, f32::NAN]);
        g.batch_size = one_of(rng, &[16, 32, 64, 0, 1000, usize::MAX]);
        g.epochs = one_of(rng, &[5, 10, 50, 0, 1_000_000, usize::MAX]);
        g.init_scale = one_of(rng, &[1.0, 0.1, f32::NAN]);
        g.seed = rng.random();

        g
    }
}
