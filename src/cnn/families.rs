//! The 16 CNN families of the spec, and a scaled-down base model of each.
//!
//! A base model is a fixed genome of its family. It enters the first generation of a run
//! with CNNs next to random genomes (ticket 05, `base_models_for`), so the search starts
//! from the known designs. It is scaled down to fit the default `Limits`, so it keeps the
//! building block of its family, not the size of the published network.
//!
//! A run on a task with a sequence input takes the 1-D twin of each 2-D base model
//! (`Family::base_model_for`): the same genes, with each kernel and window along the
//! sequence. On the motif task all 14 twins pass the shape check next to the TCN; they are
//! not tuned, and no test asks them to learn. A base model that does not fit the run's task
//! or limits stays out, with a log line: on the image tasks the TCN and the U-Net, on the
//! mirror grid all but the U-Net, ResNet and SqueezeNet, under the default limits.
//!
//! Each base model is built for one kind of task (`Family::task`), and it passes the shape
//! check of every generated task of that kind under the default `Limits`. Fourteen are 2-D
//! classifiers for the image tasks (one channel of 16 x 16 pixels, see `task`); each learns
//! the shapes task above chance. The TCN reads the one-hot sequences of the motif task and
//! learns it above chance. The U-Net maps the 8 x 8 grids of the mirror grid task to their
//! mirror images, above chance.
//!
//! | Family | Base model |
//! |---|---|
//! | LeNet-5 | two 5x5 convolutions with tanh, each with average pooling; flatten, two dense layers |
//! | AlexNet | a strided 7x7 convolution, then 5x5 and 3x3 convolutions with ReLU and max pooling; flatten, a dense layer |
//! | VGG | 3x3 convolutions with batch norm and ReLU, three of them followed by max pooling; flatten, a dense layer |
//! | GoogLeNet | two Inception blocks with 1x1, 3x3, 5x5 and pooling branches, the second followed by max pooling |
//! | ResNet | a basic block with the input as shortcut, then a strided bottleneck with a projection |
//! | DenseNet | two dense blocks of two layers with a growth rate of 8, and a transition with average pooling |
//! | Xception | separable blocks with skip connections, the first with max pooling and a projection |
//! | MobileNet | a strided convolution, then separable blocks without skip connections |
//! | MobileNetV2 | inverted residuals with ReLU6, the shortcut only where the shape stays |
//! | ShuffleNet | units with 2 groups, the channel shuffle and skip connections, the first strided |
//! | SqueezeNet | Fire modules, one with a simple bypass, the last followed by max pooling |
//! | SENet | the ResNet design with basic blocks, each with squeeze-and-excitation |
//! | ConvNeXt | a block with a depthwise 7x7, layer norm, the inverted MLP with GELU and a skip connection, then a strided one |
//! | U-Net | an encoder from 8x8 to 4x4, residual blocks, a decoder with a transposed convolution and a concatenated skip, the per-cell head |
//! | Dilated CNN | 3x3 convolutions with dilations 1, 2 and 3 at one resolution |
//! | TCN | residual blocks of causal convolutions with dilations 1, 2 and 4 along the sequence |

use crate::cnn::genome::{
    Block, CnnActivation, CnnGenome, Dims, Downsample, Head, Norm, Stage, Stem,
};
use crate::config::Limits;
use crate::genome::{Activation, Arch, Genome, Optimizer};
use crate::task::{InputShape, TaskKind, TaskSpec};

/// The families of the spec, in the order of its table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    LeNet5,
    AlexNet,
    Vgg,
    GoogLeNet,
    ResNet,
    DenseNet,
    Xception,
    MobileNet,
    MobileNetV2,
    ShuffleNet,
    SqueezeNet,
    SeNet,
    ConvNeXt,
    UNet,
    Dilated,
    Tcn,
}

/// Every family, in the order of the spec's table.
pub const FAMILIES: [Family; 16] = [
    Family::LeNet5,
    Family::AlexNet,
    Family::Vgg,
    Family::GoogLeNet,
    Family::ResNet,
    Family::DenseNet,
    Family::Xception,
    Family::MobileNet,
    Family::MobileNetV2,
    Family::ShuffleNet,
    Family::SqueezeNet,
    Family::SeNet,
    Family::ConvNeXt,
    Family::UNet,
    Family::Dilated,
    Family::Tcn,
];

impl Family {
    /// Short name for logs and summaries, e.g. "mobilenet-v2".
    pub fn name(self) -> &'static str {
        match self {
            Family::LeNet5 => "lenet-5",
            Family::AlexNet => "alexnet",
            Family::Vgg => "vgg",
            Family::GoogLeNet => "googlenet",
            Family::ResNet => "resnet",
            Family::DenseNet => "densenet",
            Family::Xception => "xception",
            Family::MobileNet => "mobilenet",
            Family::MobileNetV2 => "mobilenet-v2",
            Family::ShuffleNet => "shufflenet",
            Family::SqueezeNet => "squeezenet",
            Family::SeNet => "senet",
            Family::ConvNeXt => "convnext",
            Family::UNet => "u-net",
            Family::Dilated => "dilated",
            Family::Tcn => "tcn",
        }
    }

    /// The base model of this family (see the module doc).
    pub fn base_model(self) -> Genome {
        match self {
            Family::LeNet5 => lenet5(),
            Family::AlexNet => alexnet(),
            Family::Vgg => vgg(),
            Family::GoogLeNet => googlenet(),
            Family::ResNet => resnet(),
            Family::DenseNet => densenet(),
            Family::Xception => xception(),
            Family::MobileNet => mobilenet(),
            Family::MobileNetV2 => mobilenet_v2(),
            Family::ShuffleNet => shufflenet(),
            Family::SqueezeNet => squeezenet(),
            Family::SeNet => senet(),
            Family::ConvNeXt => convnext(),
            Family::UNet => unet(),
            Family::Dilated => dilated(),
            Family::Tcn => tcn(),
        }
    }

    /// The task the base model is built for: the motif task for TCN, the mirror grid for
    /// U-Net, and the shapes task for the others. Each fits all tasks of its kind of input
    /// and output (`TaskKind::spec`).
    pub fn task(self) -> TaskKind {
        match self {
            Family::Tcn => TaskKind::Motif1d,
            Family::UNet => TaskKind::MirrorGrid,
            _ => TaskKind::Shapes,
        }
    }

    /// The base model of this family for a run on the task `spec`: `base_model`, but for a
    /// task whose input is a sequence the 1-D twin of a 2-D base model, with the same genes
    /// and `Dims::OneD`, so that each k x k kernel or window becomes one of k cells along the
    /// sequence. It may still fail the shape check of the task (`base_models_for`).
    pub fn base_model_for(self, spec: &TaskSpec) -> Genome {
        let mut genome = self.base_model();

        if matches!(spec.input, InputShape::Sequence { .. })
            && let Arch::Cnn(cnn) = &mut genome.arch
        {
            cnn.dims = Dims::OneD;
        }

        genome
    }
}

/// The base models of all families, in the order of `FAMILIES`.
pub fn base_models() -> Vec<(Family, Genome)> {
    FAMILIES
        .into_iter()
        .map(|family| (family, family.base_model()))
        .collect()
}

/// The base models that join a run, and the families left out, each with the reason
/// (`base_models_for`).
pub type BaseModels = (Vec<(Family, Genome)>, Vec<(Family, String)>);

/// The base models that join generation 0 of a run with CNNs on the task `spec`: in the
/// order of `FAMILIES`, `Family::base_model_for` of each family whose genome passes
/// `Genome::validate` for `limits` and the task. The second list names the families left
/// out, each with the reason, for the log of the run.
pub fn base_models_for(spec: &TaskSpec, limits: &Limits) -> BaseModels {
    let mut fit = Vec::new();
    let mut left_out = Vec::new();

    for family in FAMILIES {
        let genome = family.base_model_for(spec);

        match genome.validate(limits, spec) {
            Ok(()) => fit.push((family, genome)),
            Err(e) => left_out.push((family, e.to_string())),
        }
    }

    (fit, left_out)
}

/// A 2-D genome with these parts. It trains with Adam at a learning rate of 0.003, without
/// weight decay, in batches of 32, at the standard init scale; `epochs` and `seed` are its
/// own. Two base models change the learning rate.
fn image_genome(
    stem: Stem,
    stages: Vec<Stage>,
    head: Head,
    hidden: Vec<usize>,
    activation: Activation,
    epochs: usize,
    seed: u64,
) -> Genome {
    Genome {
        arch: Arch::Cnn(CnnGenome {
            dims: Dims::TwoD,
            stem,
            stages,
            head,
        }),
        hidden,
        activation,
        optimizer: Optimizer::Adam,
        learning_rate: 0.003,
        l2: 0.0,
        batch_size: 32,
        epochs,
        init_scale: 1.0,
        seed,
    }
}

/// A 1-D genome with these parts, a global average pooling head without dense layers, and
/// the training genes of `image_genome`.
fn sequence_genome(stem: Stem, stages: Vec<Stage>, epochs: usize, seed: u64) -> Genome {
    let mut genome = image_genome(
        stem,
        stages,
        Head::GlobalAvgPool,
        Vec::new(),
        Activation::Relu,
        epochs,
        seed,
    );

    if let Arch::Cnn(cnn) = &mut genome.arch {
        cnn.dims = Dims::OneD;
    }

    genome
}

/// A stage of this block kind: stride 1, dilation 1, one group, no options.
fn stage(
    block: Block,
    width: usize,
    kernel: usize,
    norm: Norm,
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
        norm,
        activation,
    }
}

/// The stem: a convolution of the image to `width` channels.
fn stem(
    width: usize,
    kernel: usize,
    stride: usize,
    down: Downsample,
    norm: Norm,
    activation: CnnActivation,
) -> Stem {
    Stem {
        width,
        kernel,
        stride,
        down,
        norm,
        activation,
    }
}

/// LeNet-5 (LeCun et al. 1998): 5x5 convolutions of 6 and 16 channels with tanh, each
/// followed by 2x2 average pooling (the subsampling layers), then flatten and dense layers
/// of 32 and 16 with tanh, scaled down from 120 and 84. It trains at a learning rate of
/// 0.01: on shapes it reached 0.563 there, 0.472 at 0.003 (8 epochs), and chance at 0.03.
fn lenet5() -> Genome {
    let tanh = CnnActivation::Tanh;
    let genome = image_genome(
        stem(6, 5, 2, Downsample::AvgPool, Norm::None, tanh),
        vec![Stage {
            stride: 2,
            down: Downsample::AvgPool,
            ..stage(Block::Plain, 16, 5, Norm::None, tanh)
        }],
        Head::Flatten,
        vec![32, 16],
        Activation::Tanh,
        10,
        1,
    );

    Genome {
        learning_rate: 0.01,
        ..genome
    }
}

/// AlexNet style (Krizhevsky et al. 2012): a large first convolution, 7x7 with a stride of
/// 2, then a 5x5 convolution with 2x2 max pooling and a 3x3 convolution, all with ReLU, then
/// flatten and a dense layer of 16 with ReLU.
fn alexnet() -> Genome {
    let relu = CnnActivation::Relu;

    image_genome(
        stem(12, 7, 2, Downsample::Conv, Norm::None, relu),
        vec![
            Stage {
                stride: 2,
                down: Downsample::MaxPool,
                ..stage(Block::Plain, 24, 5, Norm::None, relu)
            },
            stage(Block::Plain, 16, 3, Norm::None, relu),
        ],
        Head::Flatten,
        vec![16],
        Activation::Relu,
        5,
        2,
    )
}

/// VGG (Simonyan and Zisserman 2015): only 3x3 convolutions, here with batch norm (Ioffe and
/// Szegedy 2015) and ReLU, and 2x2 max pooling after the stem and after the second and
/// third stage; then flatten and a dense layer of 32.
fn vgg() -> Genome {
    let relu = CnnActivation::Relu;
    let pooled = |width| Stage {
        stride: 2,
        down: Downsample::MaxPool,
        ..stage(Block::Plain, width, 3, Norm::Batch, relu)
    };

    image_genome(
        stem(8, 3, 2, Downsample::MaxPool, Norm::Batch, relu),
        vec![
            stage(Block::Plain, 16, 3, Norm::Batch, relu),
            pooled(16),
            pooled(32),
        ],
        Head::Flatten,
        vec![32],
        Activation::Relu,
        6,
        3,
    )
}

/// ResNet (He et al. 2016): a 3x3 stem with max pooling, a basic block with the input as
/// shortcut, then a bottleneck that moves by 2 with a projection shortcut; batch norm,
/// ReLU, global average pooling.
fn resnet() -> Genome {
    let relu = CnnActivation::Relu;

    image_genome(
        stem(8, 3, 2, Downsample::MaxPool, Norm::Batch, relu),
        vec![
            Stage {
                residual: true,
                ..stage(Block::Basic, 8, 3, Norm::Batch, relu)
            },
            Stage {
                stride: 2,
                residual: true,
                ..stage(Block::Bottleneck, 32, 3, Norm::Batch, relu)
            },
        ],
        Head::GlobalAvgPool,
        Vec::new(),
        Activation::Relu,
        6,
        5,
    )
}

/// Xception (Chollet 2017): separable blocks with skip connections. The first moves to 16
/// channels and max-pools, with a projection shortcut, as in the entry flow; the second
/// keeps its shape and adds its input, as in the middle flow.
fn xception() -> Genome {
    let relu = CnnActivation::Relu;

    image_genome(
        stem(8, 3, 2, Downsample::MaxPool, Norm::Batch, relu),
        vec![
            Stage {
                stride: 2,
                down: Downsample::MaxPool,
                residual: true,
                ..stage(Block::Separable, 16, 3, Norm::Batch, relu)
            },
            Stage {
                residual: true,
                ..stage(Block::Separable, 16, 3, Norm::Batch, relu)
            },
        ],
        Head::GlobalAvgPool,
        Vec::new(),
        Activation::Relu,
        6,
        7,
    )
}

/// MobileNet (Howard et al. 2017, preprint): a full 3x3 convolution that moves by 2, then
/// separable blocks with batch norm and ReLU, one of them strided, and no skip
/// connections. It trains at a learning rate of 0.01: on shapes it reached 0.854 there,
/// 0.536 at 0.003 (6 epochs).
fn mobilenet() -> Genome {
    let relu = CnnActivation::Relu;
    let genome = image_genome(
        stem(8, 3, 2, Downsample::Conv, Norm::Batch, relu),
        vec![
            stage(Block::Separable, 16, 3, Norm::Batch, relu),
            Stage {
                stride: 2,
                ..stage(Block::Separable, 32, 3, Norm::Batch, relu)
            },
            stage(Block::Separable, 32, 3, Norm::Batch, relu),
        ],
        Head::GlobalAvgPool,
        Vec::new(),
        Activation::Relu,
        8,
        8,
    );

    Genome {
        learning_rate: 0.01,
        ..genome
    }
}

/// MobileNetV2 (Sandler et al. 2018): a 3x3 stem that pools to 4x4, then inverted
/// residuals with ReLU6 and batch norm. The first keeps its shape and adds its input; the
/// second moves by 2 to 16 channels and has no shortcut, as in the paper.
fn mobilenet_v2() -> Genome {
    let relu6 = CnnActivation::Relu6;

    image_genome(
        stem(8, 3, 4, Downsample::MaxPool, Norm::Batch, relu6),
        vec![
            Stage {
                residual: true,
                ..stage(Block::InvertedResidual, 8, 3, Norm::Batch, relu6)
            },
            Stage {
                stride: 2,
                ..stage(Block::InvertedResidual, 16, 3, Norm::Batch, relu6)
            },
        ],
        Head::GlobalAvgPool,
        Vec::new(),
        Activation::Relu,
        5,
        9,
    )
}

/// SENet (Hu, Shen and Sun 2018): SE-ResNet, basic blocks with squeeze-and-excitation on
/// each residual branch. The second block moves by 2 with a projection shortcut.
fn senet() -> Genome {
    let relu = CnnActivation::Relu;

    image_genome(
        stem(8, 3, 2, Downsample::MaxPool, Norm::Batch, relu),
        vec![
            Stage {
                se: true,
                residual: true,
                ..stage(Block::Basic, 8, 3, Norm::Batch, relu)
            },
            Stage {
                stride: 2,
                se: true,
                residual: true,
                ..stage(Block::Basic, 16, 3, Norm::Batch, relu)
            },
        ],
        Head::GlobalAvgPool,
        Vec::new(),
        Activation::Relu,
        6,
        12,
    )
}

/// Dilated CNN (Yu and Koltun 2016): after a stem that pools to 8x8, 3x3 convolutions with
/// dilations 1, 2 and 3 at that one resolution, so the receptive field grows without
/// pooling; batch norm, ReLU, global average pooling.
fn dilated() -> Genome {
    let relu = CnnActivation::Relu;
    let dilated = |width, dilation| Stage {
        dilation,
        ..stage(Block::Plain, width, 3, Norm::Batch, relu)
    };

    image_genome(
        stem(8, 3, 2, Downsample::MaxPool, Norm::Batch, relu),
        vec![dilated(16, 1), dilated(16, 2), dilated(16, 3)],
        Head::GlobalAvgPool,
        Vec::new(),
        Activation::Relu,
        5,
        15,
    )
}

/// GoogLeNet (Szegedy et al. 2015): after a stem that pools to 8x8, two Inception blocks
/// of width 12, so each of the 1x1, 3x3, 5x5 and pooling branches has 3 channels. The
/// second is followed by max pooling, as GoogLeNet pools between its groups of Inception
/// blocks; batch norm, ReLU, global average pooling. At 6 epochs the time model gave 20.1
/// s, above the budget of 20 s, so it trains for 5.
fn googlenet() -> Genome {
    let relu = CnnActivation::Relu;

    image_genome(
        stem(8, 3, 2, Downsample::MaxPool, Norm::Batch, relu),
        vec![
            stage(Block::Inception, 12, 3, Norm::Batch, relu),
            Stage {
                stride: 2,
                down: Downsample::MaxPool,
                ..stage(Block::Inception, 12, 3, Norm::Batch, relu)
            },
        ],
        Head::GlobalAvgPool,
        Vec::new(),
        Activation::Relu,
        5,
        4,
    )
}

/// DenseNet (Huang et al. 2017): two dense blocks of two layers with a growth rate of 8, and
/// a transition between them, a 1x1 convolution to 16 channels and average pooling by 2;
/// batch norm, ReLU, global average pooling.
fn densenet() -> Genome {
    let relu = CnnActivation::Relu;
    let layer = || stage(Block::Dense, 8, 3, Norm::Batch, relu);

    image_genome(
        stem(8, 3, 2, Downsample::MaxPool, Norm::Batch, relu),
        vec![
            layer(),
            layer(),
            Stage {
                stride: 2,
                down: Downsample::AvgPool,
                ..stage(Block::Transition, 16, 1, Norm::Batch, relu)
            },
            layer(),
            layer(),
        ],
        Head::GlobalAvgPool,
        Vec::new(),
        Activation::Relu,
        6,
        6,
    )
}

/// ShuffleNet (Zhang et al. 2018): a stem that pools to 8x8, then three units of width 16
/// with 2 groups, so each has 4 inner channels in 2 groups and the channel shuffle between
/// them. Each unit has its skip connection; the first moves by 2 with a projection. It
/// trains at a learning rate of 0.01 for 12 epochs. A unit of width 32 with 4 groups after a
/// stem of 16 channels cost twice as much and reached 0.723 there (6 epochs), and 0.564 at
/// a learning rate of 0.003.
fn shufflenet() -> Genome {
    let relu = CnnActivation::Relu;
    let unit = |stride| Stage {
        stride,
        groups: 2,
        residual: true,
        ..stage(Block::Shuffle, 16, 3, Norm::Batch, relu)
    };
    let genome = image_genome(
        stem(8, 3, 2, Downsample::MaxPool, Norm::Batch, relu),
        vec![unit(2), unit(1), unit(1)],
        Head::GlobalAvgPool,
        Vec::new(),
        Activation::Relu,
        12,
        10,
    );

    Genome {
        learning_rate: 0.01,
        ..genome
    }
}

/// SqueezeNet (Iandola et al. 2016, preprint): a stem that pools to 8x8, then Fire modules,
/// the second with its simple bypass (a skip connection), the third followed by max
/// pooling; batch norm, ReLU, global average pooling.
fn squeezenet() -> Genome {
    let relu = CnnActivation::Relu;

    image_genome(
        stem(8, 3, 2, Downsample::MaxPool, Norm::Batch, relu),
        vec![
            stage(Block::Fire, 16, 3, Norm::Batch, relu),
            Stage {
                residual: true,
                ..stage(Block::Fire, 16, 3, Norm::Batch, relu)
            },
            Stage {
                stride: 2,
                down: Downsample::MaxPool,
                ..stage(Block::Fire, 32, 3, Norm::Batch, relu)
            },
        ],
        Head::GlobalAvgPool,
        Vec::new(),
        Activation::Relu,
        5,
        11,
    )
}

/// ConvNeXt (Liu et al. 2022): a stem that pools to 8x8, then a block with its depthwise
/// 7x7 convolution, layer norm, the inverted MLP with GELU and the skip connection, then a
/// block that moves by 2 with a projection; global average pooling. It trains at a learning
/// rate of 0.01. The stem has batch norm: with layer norm there, as in ConvNeXt, the base
/// model reached 0.459 on shapes, and 0.382 at a learning rate of 0.003.
fn convnext() -> Genome {
    let gelu = CnnActivation::Gelu;
    let block = |width, kernel, stride| Stage {
        stride,
        residual: true,
        ..stage(Block::ConvNext, width, kernel, Norm::Layer, gelu)
    };
    let genome = image_genome(
        stem(8, 3, 2, Downsample::MaxPool, Norm::Batch, gelu),
        vec![block(8, 7, 1), block(16, 3, 2)],
        Head::GlobalAvgPool,
        Vec::new(),
        Activation::Gelu,
        5,
        13,
    );

    Genome {
        learning_rate: 0.01,
        ..genome
    }
}

/// TCN (Bai, Kolter and Koltun 2018, preprint), for the 1-D motif task: a stem along the
/// sequence, then three residual blocks of two causal convolutions of 3 taps each, with
/// dilations 1, 2 and 4; batch norm, ReLU, global average pooling. An output of the last
/// block reads 29 cells of the stem's output: itself and the 28 before it. The stem is an
/// ordinary convolution, and the head averages over the whole sequence, so only the blocks
/// are causal; the motif task asks about the whole sequence anyway.
fn tcn() -> Genome {
    let relu = CnnActivation::Relu;
    let block = |dilation| Stage {
        dilation,
        residual: true,
        ..stage(Block::Tcn, 16, 3, Norm::Batch, relu)
    };

    sequence_genome(
        stem(16, 3, 1, Downsample::Conv, Norm::Batch, relu),
        vec![block(1), block(2), block(4)],
        5,
        16,
    )
}

/// U-Net (Ronneberger, Fischer and Brox 2015), for the grid-to-grid mirror task: an
/// encoder that moves 8x8 to 4x4 by a strided convolution, two residual basic blocks at
/// 4x4, a decoder that upsamples back to 8x8 by a transposed convolution and joins the map
/// of the stem, and the per-cell head. An 8x8 grid leaves room for one level: a 3x3 kernel
/// does not fit a map of 2x2. It trains at a learning rate of 0.03; on the mirror grid it
/// reached an R^2 of 0.733 there, 0.435 at 0.01 and 0.172 at 0.003.
fn unet() -> Genome {
    let relu = CnnActivation::Relu;
    let genome = image_genome(
        stem(4, 3, 1, Downsample::Conv, Norm::Batch, relu),
        vec![
            Stage {
                stride: 2,
                ..stage(Block::Plain, 16, 3, Norm::Batch, relu)
            },
            Stage {
                residual: true,
                ..stage(Block::Basic, 16, 3, Norm::Batch, relu)
            },
            Stage {
                residual: true,
                ..stage(Block::Basic, 16, 3, Norm::Batch, relu)
            },
            Stage {
                stride: 2,
                ..stage(Block::Decoder, 4, 3, Norm::Batch, relu)
            },
        ],
        Head::PerCell,
        Vec::new(),
        Activation::Relu,
        10,
        14,
    );

    Genome {
        learning_rate: 0.03,
        ..genome
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::time::Instant;

    use super::*;
    use crate::config::Limits;
    use crate::score;
    use crate::task::{InputShape, OutputShape, Split, TaskKind, generate};
    use crate::train::train;

    #[test]
    fn the_16_families_have_names_and_distinct_base_models() {
        let names: BTreeSet<&str> = FAMILIES.iter().map(|f| f.name()).collect();
        let built: Vec<Family> = base_models().into_iter().map(|(f, _)| f).collect();

        assert_eq!(names.len(), 16);
        assert_eq!(built, FAMILIES);

        // No two base models are the same genome.
        let ids: BTreeSet<String> = base_models().iter().map(|(_, g)| g.id()).collect();

        assert_eq!(ids.len(), 16);
    }

    /// The kind of a task's data: image, sequence or flat input, and a flat or a grid output.
    fn data_kind(kind: TaskKind) -> (u8, bool) {
        let spec = kind.spec();
        let input = match spec.input {
            InputShape::Flat => 0,
            InputShape::Image { .. } => 1,
            InputShape::Sequence { .. } => 2,
        };

        (input, matches!(spec.output, OutputShape::Grid { .. }))
    }

    #[test]
    fn every_base_model_passes_the_shape_check_on_every_task_of_its_kind() {
        let limits = Limits::default();

        for (family, genome) in base_models() {
            let kind = data_kind(family.task());
            let (same, other): (Vec<TaskKind>, Vec<TaskKind>) = TaskKind::ALL
                .into_iter()
                .partition(|&task| data_kind(task) == kind);

            // Four image tasks with classes, one sequence task, one grid task.
            let expected = match family {
                Family::Tcn | Family::UNet => 1,
                _ => 4,
            };

            assert_eq!(same.len(), expected, "{}", family.name());

            for task in same {
                genome
                    .validate(&limits, &task.spec())
                    .unwrap_or_else(|e| panic!("{} on {}: {e}", family.name(), task.name()));
            }

            // It fits no task with another kind of input, and the per-cell head of the U-Net
            // fits no task with a vector of outputs. A 2-D classifier may fit the mirror
            // grid: its dense output layer writes the 64 values of the grid.
            for task in other {
                if data_kind(task).0 != kind.0 || family == Family::UNet {
                    assert!(
                        genome.validate(&limits, &task.spec()).is_err(),
                        "{} on {}",
                        family.name(),
                        task.name()
                    );
                }
            }
        }
    }

    #[test]
    fn each_base_model_is_built_from_the_blocks_of_its_family() {
        for (family, genome) in base_models() {
            let Arch::Cnn(cnn) = &genome.arch else {
                panic!("{} is not a CNN", family.name());
            };
            let stages = &cnn.stages;
            let all = |f: &dyn Fn(&Stage) -> bool| stages.iter().all(f);
            let any = |f: &dyn Fn(&Stage) -> bool| stages.iter().any(f);
            let only = |block: Block| all(&|s| s.block == block);
            let two_d = cnn.dims == Dims::TwoD;
            let fits = match family {
                // 5x5 convolutions with average pooling and tanh, and a dense head.
                Family::LeNet5 => {
                    only(Block::Plain)
                        && cnn.stem.kernel == 5
                        && all(&|s| s.kernel == 5 && s.down == Downsample::AvgPool)
                        && cnn.stem.activation == CnnActivation::Tanh
                        && cnn.head == Head::Flatten
                        && genome.hidden.len() == 2
                }
                // A large first convolution that moves by its stride.
                Family::AlexNet => {
                    only(Block::Plain)
                        && cnn.stem.kernel >= 7
                        && cnn.stem.stride > 1
                        && cnn.stem.down == Downsample::Conv
                        && cnn.head == Head::Flatten
                }
                // Only 3x3 convolutions, and max pooling where the size shrinks.
                Family::Vgg => {
                    only(Block::Plain)
                        && cnn.stem.kernel == 3
                        && all(&|s| {
                            s.kernel == 3 && (s.stride == 1 || s.down == Downsample::MaxPool)
                        })
                        && any(&|s| s.stride > 1)
                        && cnn.head == Head::Flatten
                }
                // Inception blocks with 3x3 and 5x5 branches, and max pooling between them.
                Family::GoogLeNet => {
                    only(Block::Inception)
                        && all(&|s| s.kernel == 3)
                        && any(&|s| s.stride > 1 && s.down == Downsample::MaxPool)
                }
                // Both residual blocks, each with its skip connection.
                Family::ResNet => {
                    any(&|s| s.block == Block::Basic)
                        && any(&|s| s.block == Block::Bottleneck)
                        && all(&|s| s.residual && !s.se)
                }
                // Dense layers in rows of two, and a transition that pools by average.
                Family::DenseNet => {
                    stages
                        .windows(2)
                        .any(|pair| pair.iter().all(|s| s.block == Block::Dense))
                        && any(&|s| s.block == Block::Transition && s.down == Downsample::AvgPool)
                        && all(&|s| matches!(s.block, Block::Dense | Block::Transition))
                }
                Family::Xception => only(Block::Separable) && all(&|s| s.residual),
                Family::MobileNet => only(Block::Separable) && all(&|s| !s.residual),
                // ReLU6, and a skip connection only where the shape stays.
                Family::MobileNetV2 => {
                    only(Block::InvertedResidual)
                        && all(&|s| s.activation == CnnActivation::Relu6)
                        && all(&|s| s.residual == (s.stride == 1))
                }
                // Group convolutions, so the channel shuffle runs, and skip connections.
                Family::ShuffleNet => only(Block::Shuffle) && all(&|s| s.groups > 1 && s.residual),
                // Fire modules, at least one with the simple bypass.
                Family::SqueezeNet => only(Block::Fire) && any(&|s| s.residual),
                Family::SeNet => all(&|s| s.se && s.residual),
                // A depthwise 7x7, layer norm and GELU in every block, and skip connections.
                Family::ConvNeXt => {
                    only(Block::ConvNext)
                        && any(&|s| s.kernel == 7)
                        && all(&|s| {
                            s.norm == Norm::Layer
                                && s.activation == CnnActivation::Gelu
                                && s.residual
                        })
                }
                // A decoder with a transposed convolution, and the per-cell head.
                Family::UNet => {
                    two_d
                        && any(&|s| s.block == Block::Decoder && s.down == Downsample::Conv)
                        && cnn.head == Head::PerCell
                }
                // Dilations above 1 and no downsampling after the stem.
                Family::Dilated => {
                    only(Block::Plain) && any(&|s| s.dilation > 1) && all(&|s| s.stride == 1)
                }
                // Causal blocks along a sequence, with growing dilations and skip connections.
                Family::Tcn => {
                    !two_d
                        && only(Block::Tcn)
                        && all(&|s| s.residual)
                        && stages
                            .windows(2)
                            .all(|pair| pair[0].dilation < pair[1].dilation)
                }
            };

            assert!(fits, "{}: {}", family.name(), genome.summary());
            assert_eq!(two_d, family != Family::Tcn, "{}", family.name());
        }
    }

    /// Train the base model of `family` on the Train split of its task (data seed 1), and
    /// return its primary score on the Valid split: the accuracy, or the R^2 of the U-Net.
    ///
    /// The default limits hold, except the clock. These tests run side by side, and on a
    /// busy machine the ShuffleNet took 34 s at one thread, above the 30 s of
    /// `max_candidate_seconds`; alone it takes 6.4 s.
    /// `base_models_train_within_their_estimates` checks the time instead.
    fn valid_score(family: Family) -> f64 {
        let genome = family.base_model();
        let spec = family.task().spec();
        let data = generate(&spec, Split::Train, 1);
        let valid = generate(&spec, Split::Valid, 1);
        let limits = Limits {
            max_candidate_seconds: 600.0,
            ..Limits::default()
        };
        let started = Instant::now();
        let (model, stats) = train(&genome, &spec, &data, &limits).expect("train");
        let primary = score::evaluate(&model, &spec, &valid).primary;

        eprintln!(
            "{}: {} parameters, {} epochs in {:.1} s, train loss {:.3}, valid score \
             {primary:.3}",
            family.name(),
            model.param_count(),
            stats.epochs_done,
            started.elapsed().as_secs_f64(),
            stats.final_train_loss
        );

        assert_eq!(stats.epochs_done, genome.epochs, "{}", family.name());
        assert!(
            !stats.diverged && !stats.stopped_by_budget,
            "{}",
            family.name()
        );

        primary
    }

    /// Each base model trains on the whole Train split of its task within the time that the
    /// time model gives it, and it prints that time as a share of the estimate. This takes
    /// about three minutes and measures time, so it runs only on request, alone, at one
    /// thread: `RAYON_NUM_THREADS=1 cargo test --lib base_models_train -- --ignored
    /// --test-threads 1`.
    #[test]
    #[ignore]
    fn base_models_train_within_their_estimates() {
        let limits = Limits::default();

        for (family, genome) in base_models() {
            let spec = family.task().spec();
            let data = generate(&spec, Split::Train, 1);
            let plan = genome.cnn_plan(&spec).expect("plan");
            let estimate = plan.train_seconds((spec.n_train * genome.epochs) as f64);
            let started = Instant::now();
            let (_, stats) = train(&genome, &spec, &data, &limits).expect("train");
            let seconds = started.elapsed().as_secs_f64();

            eprintln!(
                "BASE {}: {} parameters, {:.2e} training FLOPs, {} epochs, trained in \
                 {seconds:.1} s of {estimate:.1} s ({:.2})",
                family.name(),
                plan.params,
                genome.train_flops(&spec) as f64,
                genome.epochs,
                seconds / estimate
            );

            assert!(!stats.stopped_by_budget && stats.epochs_done == genome.epochs);
            assert!(seconds < estimate, "{}: {seconds} s", family.name());
        }
    }

    // Chance on the shapes task is 0.25: every split holds the 4 classes equally often.
    // Each bound lies 0.15 to 0.21 below the score reached on the machine that wrote these
    // tests (in the message), and at least 0.15 above chance, as room for the other bits of
    // another CPU. The scores are the same at 1 and 2 threads.

    #[test]
    fn lenet5_learns_shapes() {
        assert!(valid_score(Family::LeNet5) > 0.40, "0.563 here");
    }

    #[test]
    fn alexnet_learns_shapes() {
        assert!(valid_score(Family::AlexNet) > 0.55, "0.740 here");
    }

    #[test]
    fn vgg_learns_shapes() {
        assert!(valid_score(Family::Vgg) > 0.75, "0.916 here");
    }

    #[test]
    fn googlenet_learns_shapes() {
        assert!(valid_score(Family::GoogLeNet) > 0.55, "0.744 here");
    }

    #[test]
    fn resnet_learns_shapes() {
        assert!(valid_score(Family::ResNet) > 0.60, "0.782 here");
    }

    #[test]
    fn densenet_learns_shapes() {
        assert!(valid_score(Family::DenseNet) > 0.80, "0.961 here");
    }

    #[test]
    fn xception_learns_shapes() {
        assert!(valid_score(Family::Xception) > 0.60, "0.773 here");
    }

    #[test]
    fn mobilenet_learns_shapes() {
        assert!(valid_score(Family::MobileNet) > 0.65, "0.854 here");
    }

    #[test]
    fn mobilenet_v2_learns_shapes() {
        assert!(valid_score(Family::MobileNetV2) > 0.60, "0.760 here");
    }

    #[test]
    fn shufflenet_learns_shapes() {
        assert!(valid_score(Family::ShuffleNet) > 0.65, "0.819 here");
    }

    #[test]
    fn squeezenet_learns_shapes() {
        assert!(valid_score(Family::SqueezeNet) > 0.65, "0.817 here");
    }

    #[test]
    fn senet_learns_shapes() {
        assert!(valid_score(Family::SeNet) > 0.60, "0.788 here");
    }

    #[test]
    fn convnext_learns_shapes() {
        assert!(valid_score(Family::ConvNeXt) > 0.45, "0.655 here");
    }

    #[test]
    fn dilated_cnn_learns_shapes() {
        assert!(valid_score(Family::Dilated) > 0.70, "0.872 here");
    }

    // Chance on the motif task is 0.5: half of the sequences hold the motif.
    #[test]
    fn tcn_learns_the_motif() {
        assert!(valid_score(Family::Tcn) > 0.80, "0.951 here");
    }

    // The mirror grid scores R^2: a model that ignores its input reaches 0 at best. Over the
    // data seeds 1, 2 and 3 the U-Net reached 0.733, 0.672 and 0.596 here, so its bound lies
    // 0.23 below the score of seed 1.
    #[test]
    fn unet_learns_the_mirror_grid() {
        assert!(valid_score(Family::UNet) > 0.50, "R^2 0.733 here");
    }
}
