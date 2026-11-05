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
//! activations and the heads `GlobalAvgPool` and `Flatten`. Ticket 03 builds the blocks
//! `Basic`, `Bottleneck`, `Separable` and `InvertedResidual`, and the options `Stage::se`
//! and `Stage::residual` on every block. Ticket 04 builds the other blocks and the head
//! `PerCell`. `shape` lists the layers of each block, and `families` holds a scaled-down
//! base model per family.

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

/// The inner layers of a bottleneck have its width divided by this many channels, as in
/// ResNet (He et al. 2016). The width of a bottleneck must be a multiple of it.
pub const BOTTLENECK_RATIO: usize = 4;

/// The inner layers of an inverted residual have this many times its input channels: the
/// expansion factor t = 6 of MobileNetV2 (Sandler et al. 2018).
pub const EXPANSION: usize = 6;

/// Squeeze-and-excitation reduces a width of c channels to c / this units, at least 1.
/// SENet uses 16 (Hu, Shen and Sun 2018); our stages have 4 to 64 channels, where 16
/// would leave 1 to 4 units.
pub const SE_REDUCTION: usize = 4;

/// The branches of an Inception block, as in GoogLeNet (Szegedy et al. 2015). Each has a
/// quarter of the block's width, so the width must be a multiple of it.
pub const INCEPTION_BRANCHES: usize = 4;

/// The squeeze layer of a Fire module has its width divided by this many channels, and
/// each of its two expansions half its width (Iandola et al. 2016, preprint). SqueezeNet
/// squeezes by 8; at our widths of 8 to 64 that would leave 1 to 8 channels.
pub const FIRE_SQUEEZE_RATIO: usize = 4;

/// The inverted MLP of a ConvNeXt block has this many times its input channels, as in
/// ConvNeXt (Liu et al. 2022).
pub const CONVNEXT_EXPANSION: usize = 4;

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
    /// Groups of the convolutions that `shape` names for the block: the k x k ones of a
    /// plain, basic, bottleneck, dense, Inception, Fire, TCN or decoder block, the 1x1 ones
    /// of a separable, inverted residual, transition, ShuffleNet or ConvNeXt block. They
    /// must divide the channels before and after each of them.
    pub groups: usize,
    /// Squeeze-and-excitation (SENet) on the output of the block's last norm, before the
    /// skip connection joins.
    pub se: bool,
    /// A skip connection around the block, as in ResNet: the block's input is added to the
    /// output of its last norm. A 1x1 convolution with the stride and the norm projects
    /// the input when the block changes its channels or its size. A decoder's skip
    /// connection starts after its join.
    pub residual: bool,
    /// Normalization after each convolution of the block; in a ConvNeXt block after its
    /// depthwise convolution only.
    pub norm: Norm,
    /// Activation after the normalizations that `shape` names for the block.
    pub activation: CnnActivation,
}

/// The block kind of a stage. Each kind names the family it comes from.
///
/// `shape` lists the layers of each built kind. Only the k x k convolutions of a block read
/// neighbours: the dilation applies to each of them, the stride to the first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Block {
    /// One convolution, then the normalization and the activation: LeNet-5, AlexNet, VGG,
    /// and with a dilation above 1 the dilated CNN. Built.
    Plain,
    /// ResNet basic block (He et al. 2016): two k x k convolutions to the width, each with
    /// the normalization. With `Stage::residual` it has the skip connection of ResNet;
    /// without, it is the "plain" network of the same paper. Built.
    Basic,
    /// ResNet bottleneck: a 1x1 reduction to width / `BOTTLENECK_RATIO` channels, a k x k
    /// convolution and a 1x1 expansion to the width, each with the normalization. ResNet
    /// adds the skip connection (`Stage::residual`). Built.
    Bottleneck,
    /// Depthwise separable convolution: a depthwise k x k, then a pointwise 1x1 convolution
    /// to the width, each with the normalization and the activation (MobileNet). Xception
    /// stacks it with skip connections (`Stage::residual`). Built.
    Separable,
    /// MobileNetV2 inverted residual with a linear bottleneck: a 1x1 expansion to
    /// `EXPANSION` times the input channels, a depthwise k x k convolution, and a 1x1
    /// projection to the width without activation. MobileNetV2 adds the skip connection
    /// where the input and the output have the same shape. Built.
    InvertedResidual,
    /// GoogLeNet Inception block (Szegedy et al. 2015): four parallel branches, a 1x1, a
    /// k x k and a (k + 2) x (k + 2) convolution, the larger two after a 1x1 reduction, and a
    /// k x k max pooling followed by a 1x1 convolution; joined along the channels. With
    /// k = 3 these are the 1x1, 3x3, 5x5 and pooling branches. Built.
    Inception,
    /// One layer of a DenseNet dense block (Huang et al. 2017): a k x k convolution to
    /// `width` new channels, the growth rate, joined to its input along the channels. So a
    /// stage of this kind has its input channels plus `width`, and a row of them is a dense
    /// block. It keeps its size. Built.
    Dense,
    /// DenseNet transition between dense blocks: a 1x1 convolution to `width` channels and
    /// the downsampling of the stage, average pooling in DenseNet. Built.
    Transition,
    /// SqueezeNet Fire module (Iandola et al. 2016, preprint): a 1x1 squeeze, then a 1x1 and
    /// a k x k expansion side by side, joined along the channels. Built.
    Fire,
    /// ShuffleNet unit (Zhang et al. 2018): a 1x1 group convolution, the channel shuffle, a
    /// depthwise k x k convolution and a second 1x1 group convolution. ShuffleNet adds the
    /// skip connection (`Stage::residual`). Built.
    Shuffle,
    /// ConvNeXt block (Liu et al. 2022): a depthwise k x k convolution (7x7 in ConvNeXt), the
    /// normalization (layer norm in ConvNeXt), and an inverted MLP of two 1x1 convolutions
    /// with the activation between them (GELU in ConvNeXt). It ends without activation.
    /// ConvNeXt adds the skip connection (`Stage::residual`). Built.
    ConvNext,
    /// TCN block (Bai, Kolter and Koltun 2018, preprint): two causal dilated convolutions
    /// along a sequence, so an output reads no later input. The TCN adds the skip connection
    /// (`Stage::residual`). 1-D genomes only. Built.
    Tcn,
    /// U-Net decoder stage (Ronneberger, Fischer and Brox 2015): upsampling by `stride`, the
    /// join with the map of an earlier stage at that size along the channels, and a k x k
    /// convolution. `down` picks the upsampling: `Downsample::Conv` a transposed convolution
    /// (U-Net's up-convolution), a pooling kind nearest upsampling. Built.
    Decoder,
}

impl Block {
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

/// How a stride above 1 shrinks the output of the stem or of a stage. A decoder upsamples
/// instead (see `Block::Decoder`).
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
    /// For a task with a grid of outputs (U-Net): the dense layers of `Genome::hidden` and
    /// the output layer as 1x1 convolutions, the same at every cell. The last map must have
    /// the size of the output grid.
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
    use crate::task::TaskKind;

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
        stage(Block::Plain, width, kernel, activation)
    }

    /// A stage of this block kind with stride 1, dilation 1, one group, no options and no
    /// normalization.
    pub(crate) fn stage(
        block: Block,
        width: usize,
        kernel: usize,
        activation: CnnActivation,
    ) -> Stage {
        Stage {
            block,
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

    /// VGG-like, for a 1 x 16 x 16 image: 3x3 convolutions with batch norm and ReLU, three
    /// of them followed by 2x2 max pooling, then flatten and one dense layer to the classes.
    ///
    /// The stem pools as well: at 16 x 16 throughout its first two units this CNN took 21 s
    /// for 5 epochs at one thread, above the time budget of the default limits.
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
                stride: 2,
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

        let mut g = genome(cnn, vec![8], Activation::Gelu);

        // 5 epochs keep it well within the time budget of the default limits.
        g.epochs = 5;

        g
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

    /// Small genomes for every block kind and both options, each with the task it fits and
    /// a name: first those of ticket 03, then those of ticket 04 (`family_b_genomes`), then
    /// two that pool the output of a batch norm directly (`pooled_norm_genomes`). The
    /// 2-D ones on images start with a stem of 8 channels and batch norm that pools a 1 x 16
    /// x 16 image to 8x8, and end with global average pooling and a dense layer to the
    /// classes. Together they cover the identity and both projection shortcuts (by a strided
    /// convolution, and by a change of channels before pooling), every norm inside a block,
    /// a block without norm, dilation, groups, and 1-D blocks.
    pub(crate) fn block_genomes() -> Vec<(&'static str, TaskKind, Genome)> {
        let relu = CnnActivation::Relu;
        let with = |block: Block, width: usize, norm: Norm, activation: CnnActivation| Stage {
            norm,
            ..stage(block, width, 3, activation)
        };
        let pooled = |stride: usize, stages: Vec<Stage>| {
            let cnn = CnnGenome {
                dims: Dims::TwoD,
                stem: Stem {
                    width: 8,
                    kernel: 3,
                    stride,
                    down: Downsample::MaxPool,
                    norm: Norm::Batch,
                    activation: CnnActivation::Relu,
                },
                stages,
                head: Head::GlobalAvgPool,
            };
            let mut g = genome(cnn, Vec::new(), Activation::Relu);

            g.epochs = 5;

            g
        };
        let image = |stages: Vec<Stage>| pooled(2, stages);

        let mut genomes = vec![
            (
                "plain with squeeze-and-excitation and an identity shortcut",
                TaskKind::Shapes,
                image(vec![Stage {
                    se: true,
                    residual: true,
                    ..with(Block::Plain, 8, Norm::Batch, relu)
                }]),
            ),
            (
                "plain, dilated",
                TaskKind::Shapes,
                image(vec![
                    Stage {
                        dilation: 2,
                        ..with(Block::Plain, 8, Norm::Batch, relu)
                    },
                    Stage {
                        dilation: 3,
                        ..with(Block::Plain, 8, Norm::Batch, relu)
                    },
                ]),
            ),
            (
                "basic, strided, with a projection",
                TaskKind::Shapes,
                image(vec![Stage {
                    stride: 2,
                    residual: true,
                    ..with(Block::Basic, 16, Norm::Batch, relu)
                }]),
            ),
            (
                "basic without norm, with a projection that has a bias",
                TaskKind::Shapes,
                image(vec![Stage {
                    residual: true,
                    ..with(Block::Basic, 12, Norm::None, CnnActivation::Silu)
                }]),
            ),
            (
                "bottleneck with group norm, groups and a projection",
                TaskKind::Shapes,
                image(vec![Stage {
                    groups: 2,
                    residual: true,
                    ..with(Block::Bottleneck, 16, Norm::Group, relu)
                }]),
            ),
            (
                "separable with layer norm, squeeze-and-excitation, a projection and pooling",
                TaskKind::Shapes,
                image(vec![Stage {
                    stride: 2,
                    down: Downsample::MaxPool,
                    se: true,
                    residual: true,
                    ..with(Block::Separable, 16, Norm::Layer, relu)
                }]),
            ),
            // At 8x8 these two were estimated at 36.3 s for 5 epochs, above the time
            // budget of 20 s, so they run at 4x4.
            (
                "inverted residuals, one with an identity shortcut, one strided",
                TaskKind::Shapes,
                pooled(
                    4,
                    vec![
                        Stage {
                            residual: true,
                            ..with(
                                Block::InvertedResidual,
                                8,
                                Norm::Batch,
                                CnnActivation::Relu6,
                            )
                        },
                        Stage {
                            stride: 2,
                            ..with(
                                Block::InvertedResidual,
                                16,
                                Norm::Batch,
                                CnnActivation::Relu6,
                            )
                        },
                    ],
                ),
            ),
            (
                "1-D basic and separable blocks with squeeze-and-excitation",
                TaskKind::Motif1d,
                {
                    let cnn = CnnGenome {
                        dims: Dims::OneD,
                        stem: Stem {
                            width: 8,
                            kernel: 5,
                            stride: 2,
                            down: Downsample::MaxPool,
                            norm: Norm::Batch,
                            activation: CnnActivation::Relu,
                        },
                        stages: vec![
                            Stage {
                                se: true,
                                residual: true,
                                dilation: 2,
                                ..with(Block::Basic, 8, Norm::Batch, relu)
                            },
                            Stage {
                                stride: 2,
                                se: true,
                                residual: true,
                                ..with(Block::Separable, 16, Norm::Batch, relu)
                            },
                        ],
                        head: Head::GlobalAvgPool,
                    };
                    let mut g = genome(cnn, Vec::new(), Activation::Relu);

                    g.epochs = 5;

                    g
                },
            ),
        ];

        genomes.extend(family_b_genomes());
        genomes.extend(pooled_norm_genomes());

        genomes
    }

    /// Inverted residuals that downsample by pooling and have no option. Such a block ends
    /// with a norm and no activation, so its pooling reads the output of the norm directly.
    /// In training mode a batch norm gives a transposed view there, which the strided
    /// pooling of `cnn::ops` must accept.
    fn pooled_norm_genomes() -> Vec<(&'static str, TaskKind, Genome)> {
        let relu6 = CnnActivation::Relu6;
        let pooled = |down: Downsample| Stage {
            stride: 2,
            down,
            norm: Norm::Batch,
            ..stage(Block::InvertedResidual, 8, 3, relu6)
        };
        let stem = |kernel: usize, stride: usize| Stem {
            width: 8,
            kernel,
            stride,
            down: Downsample::MaxPool,
            norm: Norm::Batch,
            activation: CnnActivation::Relu,
        };
        let with_epochs = |cnn: CnnGenome| {
            let mut g = genome(cnn, Vec::new(), Activation::Relu);

            g.epochs = 5;

            g
        };

        // At 8x8 the first was estimated at 26 s for 5 epochs, above the time budget of 20 s,
        // so its stem pools to 4x4.
        vec![
            (
                "an inverted residual that downsamples by max pooling, without options",
                TaskKind::Shapes,
                with_epochs(CnnGenome {
                    dims: Dims::TwoD,
                    stem: stem(3, 4),
                    stages: vec![pooled(Downsample::MaxPool)],
                    head: Head::GlobalAvgPool,
                }),
            ),
            (
                "1-D inverted residuals that downsample by max and by average pooling",
                TaskKind::Motif1d,
                with_epochs(CnnGenome {
                    dims: Dims::OneD,
                    stem: stem(5, 2),
                    stages: vec![pooled(Downsample::MaxPool), pooled(Downsample::AvgPool)],
                    head: Head::GlobalAvgPool,
                }),
            ),
        ]
    }

    /// The block genomes of ticket 04, one or more per block kind of that ticket. The 2-D
    /// ones on images start with the stem of `block_genomes`. Together they cover every join
    /// (a dense layer, Inception's four branches, a Fire module's two, both upsamplings of a
    /// decoder), the channel shuffle, a TCN with every dilation and stride kind, the 1-D path
    /// of every new operation, and the per-cell head.
    fn family_b_genomes() -> Vec<(&'static str, TaskKind, Genome)> {
        let relu = CnnActivation::Relu;
        let with = |block: Block, width: usize, kernel: usize, norm: Norm| Stage {
            norm,
            ..stage(block, width, kernel, relu)
        };
        let image = |stages: Vec<Stage>| {
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
                stages,
                head: Head::GlobalAvgPool,
            };
            let mut g = genome(cnn, Vec::new(), Activation::Relu);

            g.epochs = 5;

            g
        };
        let sequence = |stem: Stem, stages: Vec<Stage>| {
            let cnn = CnnGenome {
                dims: Dims::OneD,
                stem,
                stages,
                head: Head::GlobalAvgPool,
            };
            let mut g = genome(cnn, Vec::new(), Activation::Relu);

            g.epochs = 5;

            g
        };
        let one_d_stem = |stride: usize| Stem {
            width: 8,
            kernel: 5,
            stride,
            down: Downsample::MaxPool,
            norm: Norm::Batch,
            activation: relu,
        };
        let gelu = CnnActivation::Gelu;

        vec![
            (
                "dense layers and a transition with group norm",
                TaskKind::Shapes,
                image(vec![
                    with(Block::Dense, 4, 3, Norm::Batch),
                    Stage {
                        dilation: 2,
                        ..with(Block::Dense, 4, 3, Norm::Batch)
                    },
                    Stage {
                        stride: 2,
                        down: Downsample::AvgPool,
                        ..with(Block::Transition, 8, 1, Norm::Group)
                    },
                ]),
            ),
            (
                "inception blocks, one strided with groups and a projection",
                TaskKind::Shapes,
                image(vec![
                    with(Block::Inception, 16, 3, Norm::Batch),
                    Stage {
                        stride: 2,
                        groups: 2,
                        residual: true,
                        ..with(Block::Inception, 16, 3, Norm::Batch)
                    },
                ]),
            ),
            (
                "fire modules, one with squeeze-and-excitation and an identity shortcut",
                TaskKind::Shapes,
                image(vec![
                    with(Block::Fire, 16, 3, Norm::Batch),
                    Stage {
                        se: true,
                        residual: true,
                        ..with(Block::Fire, 16, 3, Norm::Batch)
                    },
                ]),
            ),
            (
                "shufflenet units with 2 and 4 groups and projections",
                TaskKind::Shapes,
                image(vec![
                    Stage {
                        groups: 2,
                        residual: true,
                        ..with(Block::Shuffle, 16, 3, Norm::Batch)
                    },
                    Stage {
                        stride: 2,
                        groups: 4,
                        residual: true,
                        ..with(Block::Shuffle, 16, 3, Norm::Batch)
                    },
                ]),
            ),
            (
                "convnext blocks with layer norm and gelu, one strided",
                TaskKind::Shapes,
                image(vec![
                    Stage {
                        residual: true,
                        activation: gelu,
                        ..with(Block::ConvNext, 8, 7, Norm::Layer)
                    },
                    Stage {
                        stride: 2,
                        residual: true,
                        activation: gelu,
                        ..with(Block::ConvNext, 16, 3, Norm::Layer)
                    },
                ]),
            ),
            (
                "tcn blocks with dilations 1, 2 and 4, the last strided",
                TaskKind::Motif1d,
                sequence(
                    one_d_stem(1),
                    vec![
                        Stage {
                            residual: true,
                            ..with(Block::Tcn, 8, 3, Norm::Batch)
                        },
                        Stage {
                            dilation: 2,
                            residual: true,
                            ..with(Block::Tcn, 16, 3, Norm::Batch)
                        },
                        Stage {
                            stride: 2,
                            dilation: 4,
                            residual: true,
                            ..with(Block::Tcn, 16, 3, Norm::Batch)
                        },
                    ],
                ),
            ),
            (
                "a u-net with a transposed convolution, nearest upsampling and a per-cell head",
                TaskKind::MirrorGrid,
                {
                    // A 3x3 kernel does not fit a map of 2x2, so the convolutions run at 4x4
                    // and above, and the pooling of stage 2 leads to the bottom of 2x2.
                    let cnn = CnnGenome {
                        dims: Dims::TwoD,
                        stem: Stem {
                            width: 8,
                            kernel: 3,
                            stride: 1,
                            down: Downsample::Conv,
                            norm: Norm::Batch,
                            activation: relu,
                        },
                        stages: vec![
                            Stage {
                                stride: 2,
                                ..with(Block::Basic, 16, 3, Norm::Batch)
                            },
                            Stage {
                                stride: 2,
                                down: Downsample::MaxPool,
                                ..with(Block::Plain, 16, 3, Norm::Batch)
                            },
                            Stage {
                                stride: 2,
                                ..with(Block::Decoder, 16, 3, Norm::Batch)
                            },
                            Stage {
                                stride: 2,
                                down: Downsample::MaxPool,
                                residual: true,
                                ..with(Block::Decoder, 8, 3, Norm::Batch)
                            },
                        ],
                        head: Head::PerCell,
                    };
                    let mut g = genome(cnn, Vec::new(), Activation::Relu);

                    g.epochs = 5;

                    g
                },
            ),
            (
                "1-D inception, shufflenet, dense, decoder and convnext blocks",
                TaskKind::Motif1d,
                sequence(
                    one_d_stem(2),
                    vec![
                        with(Block::Inception, 8, 3, Norm::Batch),
                        Stage {
                            stride: 2,
                            groups: 2,
                            ..with(Block::Shuffle, 8, 3, Norm::Batch)
                        },
                        with(Block::Dense, 4, 3, Norm::Batch),
                        Stage {
                            stride: 2,
                            ..with(Block::Decoder, 8, 3, Norm::Batch)
                        },
                        Stage {
                            stride: 2,
                            down: Downsample::AvgPool,
                            ..with(Block::Decoder, 8, 3, Norm::Batch)
                        },
                        Stage {
                            residual: true,
                            activation: gelu,
                            ..with(Block::ConvNext, 8, 7, Norm::Layer)
                        },
                    ],
                ),
            ),
        ]
    }

    /// A CNN genome with every gene in its range: half of the stages plain, the others of
    /// any block kind, some with squeeze-and-excitation or a skip connection. A quarter of
    /// them end as a U-Net: one decoder for each unit that downsamples, in reverse order and
    /// with its stride, then the per-cell head.
    ///
    /// Widths are multiples of 4, so 1, 2 or 4 groups and group norm often fit. Many are
    /// valid on the tasks of their dims; some pool too often, read too far, split a
    /// bottleneck that group norm cannot split, or cost too much.
    pub(crate) fn plausible(rng: &mut ChaCha8Rng, dims: Dims) -> Genome {
        let unit = |rng: &mut ChaCha8Rng| Stage {
            block: if rng.random_bool(0.5) {
                Block::Plain
            } else {
                one_of(rng, &BLOCKS)
            },
            width: one_of(rng, &[4, 8, 12, 16]),
            kernel: one_of(rng, &[1, 3, 3, 5]),
            stride: one_of(rng, &[1, 1, 2]),
            down: one_of(rng, &DOWNSAMPLES),
            dilation: one_of(rng, &[1, 1, 2]),
            groups: one_of(rng, &[1, 1, 2, 4]),
            se: rng.random_bool(0.2),
            residual: rng.random_bool(0.3),
            norm: one_of(rng, &NORMS),
            activation: one_of(rng, &CNN_ACTIVATIONS),
        };
        let stem = as_stem(&unit(rng));
        let n_stages = rng.random_range(1..=4);
        let mut stages: Vec<Stage> = (0..n_stages).map(|_| unit(rng)).collect();
        let unet = rng.random_bool(0.25);

        if unet {
            let drops: Vec<usize> = std::iter::once(stem.stride)
                .chain(
                    stages
                        .iter()
                        .filter(|s| s.block != Block::Decoder)
                        .map(|s| s.stride),
                )
                .filter(|&stride| stride > 1)
                .collect();

            for &stride in drops.iter().rev() {
                stages.push(Stage {
                    block: Block::Decoder,
                    stride,
                    ..unit(rng)
                });
            }
        }

        let cnn = CnnGenome {
            dims,
            stem,
            stages,
            head: if unet {
                Head::PerCell
            } else {
                one_of(rng, &[Head::GlobalAvgPool, Head::Flatten])
            },
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
