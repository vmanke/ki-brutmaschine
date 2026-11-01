//! Variation of CNN genomes (ticket 05 of the CNN spec, ADR 0008): the operators of a
//! mutation, the crossover of whole stages, and random CNN genomes.
//!
//! Mutation. A child of one CNN parent gets one operator, drawn uniformly at random from
//! `OPERATORS`, as each child gets one mutation in Real et al. 2017 (ICML). A unit is the stem
//! or a stage. The first 13 operators are the mutations of the spec; the last three let the
//! other genes of a CNN change too.
//!
//! | Operator | Change |
//! |---|---|
//! | `Family` | the block kind of one stage; the width is rounded to a multiple of what the new kind needs, and a dense layer pools by average where it would move by its stride |
//! | `InsertStage` | a new stage of stride 1 that keeps the channels at its place, of the kind of its neighbour or of a random kind |
//! | `RemoveStage` | one stage less |
//! | `Width` | the width of one unit, by a factor of exp(0.35 N(0, 1)), at least one step |
//! | `Kernel` | the kernel of one unit, by 2 |
//! | `Stride` | the stride of one unit, by 1 |
//! | `Dilation` | the dilation of one stage, doubled or halved |
//! | `Groups` | the groups of one stage, to another power of 2 that divides its width and its input channels |
//! | `SqueezeExcite` | squeeze-and-excitation of one stage, on or off |
//! | `Residual` | the skip connection of one stage, on or off |
//! | `Normalization` | the normalization of one unit, to another kind |
//! | `Downsampling` | how a unit with a stride above 1 downsamples; a decoder switches between a transposed convolution and nearest upsampling |
//! | `Activation` | the activation of one unit, or of the dense layers of the head |
//! | `Head` | the head, to another kind |
//! | `HeadLayers` | one dense layer of the head added, removed or resized, as for an MLP |
//! | `Training` | the training genes, each with the gene rate and at least one, as for an MLP |
//!
//! An operator draws no change that would do nothing: not the kernel of a transition, not
//! the dilation of a 1x1 convolution, not the downsampling of a unit with stride 1, not the
//! activation of a head without dense layers.
//!
//! Redraw. Every child passes `Genome::validate`, the shape check and the limits, after
//! `variation::repair` has cut its epochs to fit the budget. A child that fails it, or that
//! keeps the architecture of its parent, is drawn again with fresh random choices, up to
//! `REDRAWS` times. Then the next operator is drawn from those not yet tried. `Training`
//! keeps the architecture, so a valid parent always gets a valid child. The operators keep a
//! valid genome valid; they do not mend a broken one, so a parent that fails the check gets
//! no child (`RandomMutator::propose` then draws a random genome).
//!
//! Crossover. Two CNN parents exchange whole stages where their resolutions match. A boundary
//! k of a parent lies after its unit k: boundary 0 after the stem, boundary k after stage k.
//! The first parent, drawn at random from the two, gives the stem, its stages up to a
//! boundary i, and its stages from a boundary i2 >= i on. The second gives the run of its
//! stages between two boundaries j < j2, which goes in between. The maps at boundary i of the
//! first parent and at boundary j of the second have the same size, and so have those at i2
//! and j2, unless i2 is the end of the first parent, so that the child ends with the run.
//! A child that ends with the run and takes the second parent's stages to their end takes its
//! head as well: one-point crossover is that case. Otherwise the head is the first parent's.
//! The training genes come from either parent, gene by gene. The cut is drawn uniformly from
//! all cuts at matching sizes; a child that fails the check, or has the architecture of a
//! parent, gives way to another cut, up to `REDRAWS` times.
//!
//! Random CNNs. `random` draws a CNN for the task's kind of input: a stem, then one to four
//! stages, each a plain block with probability 1/3 and else a block of any kind that fits the
//! dims, with the options sometimes on; then global average pooling or flattening, and up to
//! two dense layers. For a task with a grid of outputs it takes the U-Net form three times
//! in four: one decoder for each unit that downsamples, in reverse order and with its stride,
//! then the per-cell head. The training genes come from ranges that suit a CNN, narrower than
//! an MLP's. Each draw is repaired and checked; the first valid one is kept. After
//! `RANDOM_TRIES` failures `smallest` takes its place, the smallest CNN of the task, which
//! `Config::validate` requires to fit the limits of a run with CNNs.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::cnn::genome::{
    BLOCKS, BOTTLENECK_RATIO, Block, CNN_ACTIVATIONS, CnnActivation, CnnGenome, DOWNSAMPLES, Dims,
    Downsample, FIRE_SQUEEZE_RATIO, GROUP_NORM_GROUPS, Head, INCEPTION_BRANCHES, MAX_CHANNELS,
    MAX_DILATION, MAX_STAGES, MAX_STRIDE, NORMS, Norm, Stage, Stem,
};
use crate::cnn::shape::Shape;
use crate::config::Limits;
use crate::genome::{
    ACTIVATIONS, Activation, Arch, BATCH_SIZES, Genome, MIN_EPOCHS, OPTIMIZERS, Optimizer,
};
use crate::task::{InputShape, OutputShape, TaskSpec};
use crate::util::{log_uniform_f32, normal_f32};
use crate::variation::{mutate_head_layers, mutate_training_genes, repair};

/// Draws of one operator, or cuts of one crossover, before the next is tried.
pub const REDRAWS: usize = 16;

/// Draws of a random CNN before `smallest` takes its place.
pub const RANDOM_TRIES: usize = 256;

/// Every head.
const HEADS: [Head; 3] = [Head::GlobalAvgPool, Head::Flatten, Head::PerCell];

/// One kind of change of a CNN genome (see the module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operator {
    Family,
    InsertStage,
    RemoveStage,
    Width,
    Kernel,
    Stride,
    Dilation,
    Groups,
    SqueezeExcite,
    Residual,
    Normalization,
    Downsampling,
    Activation,
    Head,
    HeadLayers,
    Training,
}

/// Every operator, in the order of the module doc.
pub const OPERATORS: [Operator; 16] = [
    Operator::Family,
    Operator::InsertStage,
    Operator::RemoveStage,
    Operator::Width,
    Operator::Kernel,
    Operator::Stride,
    Operator::Dilation,
    Operator::Groups,
    Operator::SqueezeExcite,
    Operator::Residual,
    Operator::Normalization,
    Operator::Downsampling,
    Operator::Activation,
    Operator::Head,
    Operator::HeadLayers,
    Operator::Training,
];

impl Operator {
    /// Short name for the archive, e.g. "insert-stage".
    pub fn name(self) -> &'static str {
        match self {
            Operator::Family => "family",
            Operator::InsertStage => "insert-stage",
            Operator::RemoveStage => "remove-stage",
            Operator::Width => "width",
            Operator::Kernel => "kernel",
            Operator::Stride => "stride",
            Operator::Dilation => "dilation",
            Operator::Groups => "groups",
            Operator::SqueezeExcite => "se",
            Operator::Residual => "residual",
            Operator::Normalization => "norm",
            Operator::Downsampling => "downsampling",
            Operator::Activation => "activation",
            Operator::Head => "head",
            Operator::HeadLayers => "head-layers",
            Operator::Training => "training",
        }
    }

    /// A child of the CNN `parent` by this operator, with a new seed: the first draw that
    /// passes `Genome::validate` for `limits` and `spec` and, except for `Training`, has
    /// another architecture than the parent. `None` after `REDRAWS` draws without one, when
    /// the operator does not apply (no stage to remove, no unit that downsamples), for an
    /// MLP, and for a parent that fails the check itself.
    pub fn mutate(
        self,
        parent: &Genome,
        gene_rate: f64,
        rng: &mut ChaCha8Rng,
        limits: &Limits,
        spec: &TaskSpec,
    ) -> Option<Genome> {
        if parent.arch.is_mlp() || parent.validate(limits, spec).is_err() {
            return None;
        }

        // The channels before each stage, for `InsertStage` and `Groups`.
        let units = parent.cnn_plan(spec).ok().map(|plan| plan.units);

        for _ in 0..REDRAWS {
            let mut child = parent.clone();

            if self
                .change(&mut child, units.as_deref(), gene_rate, rng, limits)
                .is_none()
            {
                continue;
            }

            child.seed = rng.random();

            let child = repair(child, limits, spec);
            let new = self == Operator::Training || architecture(&child) != architecture(parent);

            if new && child.validate(limits, spec).is_ok() {
                return Some(child);
            }
        }

        None
    }

    /// One draw of this operator on `g`. `None` when this draw found nothing to change.
    fn change(
        self,
        g: &mut Genome,
        units: Option<&[Shape]>,
        gene_rate: f64,
        rng: &mut ChaCha8Rng,
        limits: &Limits,
    ) -> Option<()> {
        match self {
            Operator::Training => {
                mutate_training_genes(g, gene_rate, rng, limits);

                return Some(());
            }
            Operator::HeadLayers => {
                mutate_head_layers(g, rng, limits);

                return Some(());
            }
            Operator::Activation => return change_activation(g, rng),
            _ => {}
        }

        let Arch::Cnn(cnn) = &mut g.arch else {
            return None;
        };

        match self {
            Operator::Family => switch_family(cnn, rng),
            Operator::InsertStage => insert_stage(cnn, units, rng),
            Operator::RemoveStage => remove_stage(cnn, rng),
            Operator::Width => change_width(cnn, rng),
            Operator::Kernel => change_kernel(cnn, rng),
            Operator::Stride => change_stride(cnn, rng),
            Operator::Dilation => change_dilation(cnn, rng),
            Operator::Groups => change_groups(cnn, units, rng),
            Operator::SqueezeExcite => {
                let stage = pick_mut(rng, &mut cnn.stages)?;

                stage.se = !stage.se;

                Some(())
            }
            Operator::Residual => {
                let stage = pick_mut(rng, &mut cnn.stages)?;

                stage.residual = !stage.residual;

                Some(())
            }
            Operator::Normalization => {
                let unit = rng.random_range(0..=cnn.stages.len());
                let norm = unit_gene(cnn, unit, |s| &mut s.norm, |s| &mut s.norm)?;

                *norm = pick(rng, &others(&NORMS, *norm))?;

                Some(())
            }
            Operator::Downsampling => change_downsampling(cnn, rng),
            Operator::Head => {
                cnn.head = pick(rng, &others(&HEADS, cnn.head))?;

                Some(())
            }
            Operator::Training | Operator::HeadLayers | Operator::Activation => None,
        }
    }
}

/// A child of the CNN `parent` by one operator, drawn uniformly at random from those not yet
/// tried (see the module doc), and that operator. `None` for an MLP and for a parent that
/// fails the shape check or the limits itself; a valid parent always gets a child, at the
/// latest from `Training`.
pub fn mutate(
    parent: &Genome,
    gene_rate: f64,
    rng: &mut ChaCha8Rng,
    limits: &Limits,
    spec: &TaskSpec,
) -> Option<(Genome, Operator)> {
    if parent.arch.is_mlp() || parent.validate(limits, spec).is_err() {
        return None;
    }

    let mut left = OPERATORS.to_vec();

    while !left.is_empty() {
        let operator = left.remove(rng.random_range(0..left.len()));

        if let Some(child) = operator.mutate(parent, gene_rate, rng, limits, spec) {
            return Some((child, operator));
        }
    }

    None
}

/// The parts of a genome that shape its network: its arch, and the dense layers of its head
/// with their activation.
fn architecture(g: &Genome) -> (&Arch, &[usize], Activation) {
    (&g.arch, &g.hidden, g.activation)
}

/// A random element. `None` for an empty slice.
fn pick<T: Copy>(rng: &mut ChaCha8Rng, values: &[T]) -> Option<T> {
    if values.is_empty() {
        return None;
    }

    Some(values[rng.random_range(0..values.len())])
}

/// A random element to change. `None` for an empty slice.
fn pick_mut<'a, T>(rng: &mut ChaCha8Rng, values: &'a mut [T]) -> Option<&'a mut T> {
    if values.is_empty() {
        return None;
    }

    let i = rng.random_range(0..values.len());

    values.get_mut(i)
}

/// The values other than `current`.
fn others<T: Copy + PartialEq>(values: &[T], current: T) -> Vec<T> {
    values.iter().copied().filter(|&v| v != current).collect()
}

/// A gene of unit `unit`: 0 is the stem, i is stage i. `None` past the last stage.
fn unit_gene<'a, T>(
    cnn: &'a mut CnnGenome,
    unit: usize,
    stem: impl FnOnce(&'a mut Stem) -> &'a mut T,
    stage: impl FnOnce(&'a mut Stage) -> &'a mut T,
) -> Option<&'a mut T> {
    match unit.checked_sub(1) {
        None => Some(stem(&mut cnn.stem)),
        Some(i) => cnn.stages.get_mut(i).map(stage),
    }
}

/// The block kinds a random or an inserted stage may have: every kind but the decoder, and
/// the TCN block only along a sequence.
fn stage_kinds(dims: Dims) -> Vec<Block> {
    BLOCKS
        .into_iter()
        .filter(|&block| block != Block::Decoder && (block != Block::Tcn || dims == Dims::OneD))
        .collect()
}

/// The step that the width of a block of this kind, with these groups and this
/// normalization, must be a multiple of: the divisor of the kinds that split their width,
/// the groups, and the groups of a group norm. At most `MAX_CHANNELS`.
fn width_step(block: Block, groups: usize, norm: Norm) -> usize {
    let mut step = match block {
        Block::Bottleneck | Block::Shuffle => BOTTLENECK_RATIO,
        Block::Inception => INCEPTION_BRANCHES,
        Block::Fire => FIRE_SQUEEZE_RATIO,
        _ => 1,
    };

    if (2..=MAX_CHANNELS).contains(&groups) {
        step = lcm(step, groups);
    }

    if norm == Norm::Group {
        step = lcm(step, GROUP_NORM_GROUPS);
    }

    step.min(MAX_CHANNELS)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Least common multiple of two numbers from 1 to `MAX_CHANNELS`.
fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b).max(1) * b
}

/// `width` rounded to the nearest multiple of `step`, within `step..=MAX_CHANNELS`.
fn round_to(width: usize, step: usize) -> usize {
    let step = step.clamp(1, MAX_CHANNELS);
    let top = MAX_CHANNELS / step * step;

    (width.saturating_add(step / 2) / step)
        .saturating_mul(step)
        .clamp(step, top)
}

/// `Family`: another block kind for one stage (see the module doc).
fn switch_family(cnn: &mut CnnGenome, rng: &mut ChaCha8Rng) -> Option<()> {
    let dims = cnn.dims;
    let stage = pick_mut(rng, &mut cnn.stages)?;
    let kinds: Vec<Block> = BLOCKS
        .into_iter()
        .filter(|&block| {
            block != stage.block
                && (block != Block::Tcn || dims == Dims::OneD)
                && (block != Block::Decoder || stage.stride > 1)
        })
        .collect();

    stage.block = pick(rng, &kinds)?;
    stage.width = round_to(
        stage.width,
        width_step(stage.block, stage.groups, stage.norm),
    );

    // A dense layer joins its input, so it moves by pooling only; DenseNet pools by average.
    if stage.block == Block::Dense && stage.down == Downsample::Conv && stage.stride > 1 {
        stage.down = Downsample::AvgPool;
    }

    Some(())
}

/// `InsertStage`: a new stage of stride 1 that keeps the channels at its place (see the
/// module doc). `units` holds the channels after each unit of the parent.
fn insert_stage(cnn: &mut CnnGenome, units: Option<&[Shape]>, rng: &mut ChaCha8Rng) -> Option<()> {
    let count = cnn.stages.len();

    if count == 0 || count >= MAX_STAGES {
        return None;
    }

    // The new stage follows unit `at`: the stem for 0, stage `at` otherwise. It is built like
    // the stage before it, or like the first stage when it comes first.
    let at = rng.random_range(0..=count);
    let template = cnn.stages.get(at.saturating_sub(1))?.clone();
    let channels = units
        .and_then(|units| units.get(at))
        .map_or(template.width, |shape| shape.channels);
    let block = if template.block != Block::Decoder && rng.random_bool(0.5) {
        template.block
    } else {
        pick(rng, &stage_kinds(cnn.dims))?
    };
    // A dense layer adds its width to the channels; it starts with half of them.
    let width = if block == Block::Dense {
        (channels / 2).max(1)
    } else {
        channels
    };
    let stage = Stage {
        block,
        width: round_to(width, width_step(block, 1, template.norm)),
        kernel: template.kernel,
        stride: 1,
        down: Downsample::Conv,
        dilation: 1,
        groups: 1,
        se: rng.random_bool(0.25),
        residual: rng.random_bool(0.5),
        norm: template.norm,
        activation: template.activation,
    };

    cnn.stages.insert(at, stage);

    Some(())
}

/// `RemoveStage`: one stage less, if at least two are left.
fn remove_stage(cnn: &mut CnnGenome, rng: &mut ChaCha8Rng) -> Option<()> {
    let count = cnn.stages.len();

    if count < 2 {
        return None;
    }

    cnn.stages.remove(rng.random_range(0..count));

    Some(())
}

/// `Width`: the width of one unit, by a factor of exp(0.35 N(0, 1)), rounded to the step
/// its kind needs, and by at least one step.
fn change_width(cnn: &mut CnnGenome, rng: &mut ChaCha8Rng) -> Option<()> {
    let unit = rng.random_range(0..=cnn.stages.len());
    let (step, width) = match unit.checked_sub(1) {
        None => (
            width_step(Block::Plain, 1, cnn.stem.norm),
            &mut cnn.stem.width,
        ),
        Some(i) => {
            let stage = cnn.stages.get_mut(i)?;

            (
                width_step(stage.block, stage.groups, stage.norm),
                &mut stage.width,
            )
        }
    };
    let factor = (0.35 * normal_f32(rng)).exp();
    let scaled = round_to((*width as f32 * factor).round() as usize, step);

    *width = if scaled != *width {
        scaled
    } else if rng.random_bool(0.5) {
        round_to(width.saturating_add(step), step)
    } else {
        round_to(width.saturating_sub(step), step)
    };

    Some(())
}

/// `Kernel`: the kernel of one unit by 2, up from 1. A transition is always 1x1, so its
/// kernel is never drawn.
fn change_kernel(cnn: &mut CnnGenome, rng: &mut ChaCha8Rng) -> Option<()> {
    let units: Vec<usize> = (0..=cnn.stages.len())
        .filter(|&u| u == 0 || cnn.stages[u - 1].block != Block::Transition)
        .collect();
    let unit = pick(rng, &units)?;
    let kernel = unit_gene(cnn, unit, |s| &mut s.kernel, |s| &mut s.kernel)?;

    *kernel = if *kernel <= 1 || rng.random_bool(0.5) {
        kernel.saturating_add(2)
    } else {
        *kernel - 2
    };

    Some(())
}

/// `Stride`: the stride of one unit by 1, within `1..=MAX_STRIDE`.
fn change_stride(cnn: &mut CnnGenome, rng: &mut ChaCha8Rng) -> Option<()> {
    let unit = rng.random_range(0..=cnn.stages.len());
    let stride = unit_gene(cnn, unit, |s| &mut s.stride, |s| &mut s.stride)?;

    *stride = if *stride <= 1 || (*stride < MAX_STRIDE && rng.random_bool(0.5)) {
        stride.saturating_add(1)
    } else {
        *stride - 1
    };

    Some(())
}

/// `Dilation`: the dilation of one stage with a k x k convolution, doubled or halved within
/// `1..=MAX_DILATION`.
fn change_dilation(cnn: &mut CnnGenome, rng: &mut ChaCha8Rng) -> Option<()> {
    let stages: Vec<usize> = (0..cnn.stages.len())
        .filter(|&i| cnn.stages[i].kernel > 1 && cnn.stages[i].block != Block::Transition)
        .collect();
    let stage = cnn.stages.get_mut(pick(rng, &stages)?)?;
    let dilation = stage.dilation;

    stage.dilation = if dilation <= 1 {
        2
    } else if dilation < MAX_DILATION && rng.random_bool(0.5) {
        dilation.saturating_mul(2).min(MAX_DILATION)
    } else {
        dilation / 2
    };

    Some(())
}

/// `Groups`: the groups of one stage, to another power of 2 up to `MAX_CHANNELS` that
/// divides its width and its input channels. `units` holds the channels after each unit.
fn change_groups(cnn: &mut CnnGenome, units: Option<&[Shape]>, rng: &mut ChaCha8Rng) -> Option<()> {
    let count = cnn.stages.len();

    if count == 0 {
        return None;
    }

    let i = rng.random_range(0..count);
    let stage = cnn.stages.get_mut(i)?;
    let inputs = units
        .and_then(|units| units.get(i))
        .map_or(stage.width, |shape| shape.channels);
    let options: Vec<usize> = (0..=MAX_CHANNELS.ilog2())
        .map(|e| 1usize << e)
        .filter(|&groups| {
            groups != stage.groups
                && stage.width.is_multiple_of(groups)
                && inputs.is_multiple_of(groups)
        })
        .collect();

    stage.groups = pick(rng, &options)?;

    Some(())
}

/// `Downsampling`: another way to downsample for a unit with a stride above 1. A decoder
/// upsamples by a transposed convolution (`Downsample::Conv`) or, with either pooling kind,
/// by repeating cells, so it switches between the two.
fn change_downsampling(cnn: &mut CnnGenome, rng: &mut ChaCha8Rng) -> Option<()> {
    let units: Vec<usize> = std::iter::once(cnn.stem.stride)
        .chain(cnn.stages.iter().map(|stage| stage.stride))
        .enumerate()
        .filter(|&(_, stride)| stride > 1)
        .map(|(unit, _)| unit)
        .collect();
    let unit = pick(rng, &units)?;
    let decoder = unit
        .checked_sub(1)
        .and_then(|i| cnn.stages.get(i))
        .is_some_and(|stage| stage.block == Block::Decoder);
    let down = unit_gene(cnn, unit, |s| &mut s.down, |s| &mut s.down)?;

    *down = if !decoder {
        pick(rng, &others(&DOWNSAMPLES, *down))?
    } else if *down == Downsample::Conv {
        pick(rng, &[Downsample::MaxPool, Downsample::AvgPool])?
    } else {
        Downsample::Conv
    };

    Some(())
}

/// `Activation`: another activation for one unit, or for the dense layers of the head when
/// it has some.
fn change_activation(g: &mut Genome, rng: &mut ChaCha8Rng) -> Option<()> {
    let head = !g.hidden.is_empty();
    let Arch::Cnn(cnn) = &mut g.arch else {
        return None;
    };
    let slot = rng.random_range(0..cnn.stages.len() + 1 + usize::from(head));

    if slot > cnn.stages.len() {
        g.activation = pick(rng, &others(&ACTIVATIONS, g.activation))?;

        return Some(());
    }

    let activation = unit_gene(cnn, slot, |s| &mut s.activation, |s| &mut s.activation)?;

    *activation = pick(rng, &others(&CNN_ACTIVATIONS, *activation))?;

    Some(())
}

/// A cut of a crossover (see the module doc), in boundaries of the units: the first parent's
/// stages before boundary `i` and from boundary `i2` on, around the second parent's stages
/// between boundaries `j` and `j2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cut {
    i: usize,
    i2: usize,
    j: usize,
    j2: usize,
}

/// A child of the CNNs `a` and `b` that exchanges whole stages at matching resolutions (see
/// the module doc), with a new seed. `None` when one of them is no CNN or fails the shape
/// check, or when no cut of `REDRAWS` gives a valid child whose architecture differs from
/// both parents.
pub fn crossover(
    a: &Genome,
    b: &Genome,
    rng: &mut ChaCha8Rng,
    limits: &Limits,
    spec: &TaskSpec,
) -> Option<Genome> {
    let (first, second) = if rng.random_bool(0.5) { (a, b) } else { (b, a) };
    let (Arch::Cnn(f), Arch::Cnn(s)) = (&first.arch, &second.arch) else {
        return None;
    };
    let (Ok(first_plan), Ok(second_plan)) = (first.cnn_plan(spec), second.cnn_plan(spec)) else {
        return None;
    };
    let mut cuts = cuts(&first_plan.units, &second_plan.units);

    for _ in 0..REDRAWS {
        if cuts.is_empty() {
            break;
        }

        let cut = cuts.remove(rng.random_range(0..cuts.len()));
        let mut child = join(first, f, second, s, cut);

        mix_training_genes(&mut child, a, b, rng);
        child.seed = rng.random();

        let child = repair(child, limits, spec);
        let new =
            architecture(&child) != architecture(a) && architecture(&child) != architecture(b);

        if new && child.validate(limits, spec).is_ok() {
            return Some(child);
        }
    }

    None
}

/// Every cut of a first parent with these shapes after its units and a second parent with
/// those, at matching sizes (see the module doc), in a fixed order.
fn cuts(first: &[Shape], second: &[Shape]) -> Vec<Cut> {
    let last_first = first.len().saturating_sub(1);
    let mut out = Vec::new();

    for (i, start) in first.iter().enumerate() {
        for (j, other_start) in second.iter().enumerate() {
            if start.size != other_start.size {
                continue;
            }

            for (i2, end) in first.iter().enumerate().skip(i) {
                for (j2, other_end) in second.iter().enumerate().skip(j + 1) {
                    if i2 == last_first || end.size == other_end.size {
                        out.push(Cut { i, i2, j, j2 });
                    }
                }
            }
        }
    }

    out
}

/// The child of a cut: the stem and the outer stages of `first`, the inner run of `second`,
/// and the head of the parent whose stages end the child, as the module doc says.
fn join(first: &Genome, f: &CnnGenome, second: &Genome, s: &CnnGenome, cut: Cut) -> Genome {
    let stages: Vec<Stage> = f.stages[..cut.i]
        .iter()
        .chain(&s.stages[cut.j..cut.j2])
        .chain(&f.stages[cut.i2..])
        .cloned()
        .collect();
    let tail = cut.i2 == f.stages.len() && cut.j2 == s.stages.len();
    let (head, hidden, activation) = if tail {
        (s.head, second.hidden.clone(), second.activation)
    } else {
        (f.head, first.hidden.clone(), first.activation)
    };

    Genome {
        arch: Arch::Cnn(CnnGenome {
            dims: f.dims,
            stem: f.stem.clone(),
            stages,
            head,
        }),
        hidden,
        activation,
        ..first.clone()
    }
}

/// The training genes of `child`, each from `a` or `b` at random.
fn mix_training_genes(child: &mut Genome, a: &Genome, b: &Genome, rng: &mut ChaCha8Rng) {
    let mut from_a = || rng.random_bool(0.5);

    child.optimizer = if from_a() { a.optimizer } else { b.optimizer };
    child.learning_rate = if from_a() {
        a.learning_rate
    } else {
        b.learning_rate
    };
    child.l2 = if from_a() { a.l2 } else { b.l2 };
    child.batch_size = if from_a() { a.batch_size } else { b.batch_size };
    child.epochs = if from_a() { a.epochs } else { b.epochs };
    child.init_scale = if from_a() { a.init_scale } else { b.init_scale };
}

/// The dims of a CNN for the task's input: 2-D for an image, 1-D for a sequence, `None` for
/// a flat vector.
pub fn dims_for(spec: &TaskSpec) -> Option<Dims> {
    match spec.input {
        InputShape::Image { .. } => Some(Dims::TwoD),
        InputShape::Sequence { .. } => Some(Dims::OneD),
        InputShape::Flat => None,
    }
}

/// The smallest CNN for the task `spec`: a stem and one plain stage of one channel each, with
/// 1x1 kernels and no normalization, global average pooling, and the output layer; SGD for
/// `MIN_EPOCHS` epochs. `None` for a task with flat inputs.
pub fn smallest(spec: &TaskSpec) -> Option<Genome> {
    let dims = dims_for(spec)?;
    let relu = CnnActivation::Relu;

    Some(Genome {
        arch: Arch::Cnn(CnnGenome {
            dims,
            stem: Stem {
                width: 1,
                kernel: 1,
                stride: 1,
                down: Downsample::Conv,
                norm: Norm::None,
                activation: relu,
            },
            stages: vec![Stage {
                block: Block::Plain,
                width: 1,
                kernel: 1,
                stride: 1,
                down: Downsample::Conv,
                dilation: 1,
                groups: 1,
                se: false,
                residual: false,
                norm: Norm::None,
                activation: relu,
            }],
            head: Head::GlobalAvgPool,
        }),
        hidden: Vec::new(),
        activation: Activation::Relu,
        optimizer: Optimizer::Sgd,
        learning_rate: 0.01,
        l2: 0.0,
        batch_size: 32,
        epochs: MIN_EPOCHS,
        init_scale: 1.0,
        seed: 0,
    })
}

/// A random CNN for the task `spec` that passes `Genome::validate` for `limits` (see the
/// module doc). `None` for a task with flat inputs, and when not even `smallest` fits the
/// limits.
pub fn random(rng: &mut ChaCha8Rng, limits: &Limits, spec: &TaskSpec) -> Option<Genome> {
    let dims = dims_for(spec)?;

    for _ in 0..RANDOM_TRIES {
        let genome = repair(draw(rng, dims, spec, limits), limits, spec);

        if genome.validate(limits, spec).is_ok() {
            return Some(genome);
        }
    }

    let genome = Genome {
        seed: rng.random(),
        ..smallest(spec)?
    };

    genome.validate(limits, spec).is_ok().then_some(genome)
}

/// One of the values, at random. `values` is not empty.
fn one_of<T: Copy>(rng: &mut ChaCha8Rng, values: &[T]) -> T {
    values[rng.random_range(0..values.len())]
}

/// One random CNN of these dims for the task `spec`, before repair and check.
fn draw(rng: &mut ChaCha8Rng, dims: Dims, spec: &TaskSpec, limits: &Limits) -> Genome {
    let kinds = stage_kinds(dims);
    let stage = |rng: &mut ChaCha8Rng| Stage {
        block: if rng.random_bool(1.0 / 3.0) {
            Block::Plain
        } else {
            one_of(rng, &kinds)
        },
        width: one_of(rng, &[4, 8, 12, 16, 24, 32]),
        kernel: one_of(rng, &[1, 3, 3, 3, 5]),
        stride: one_of(rng, &[1, 1, 2]),
        down: one_of(rng, &DOWNSAMPLES),
        dilation: one_of(rng, &[1, 1, 1, 2]),
        groups: one_of(rng, &[1, 1, 1, 2, 4]),
        se: rng.random_bool(0.2),
        residual: rng.random_bool(0.4),
        norm: one_of(rng, &NORMS),
        activation: one_of(rng, &CNN_ACTIVATIONS),
    };
    let stem = Stem {
        width: one_of(rng, &[4, 8, 12, 16]),
        kernel: one_of(rng, &[3, 3, 5]),
        stride: one_of(rng, &[1, 2, 2]),
        down: one_of(rng, &DOWNSAMPLES),
        norm: one_of(rng, &NORMS),
        activation: one_of(rng, &CNN_ACTIVATIONS),
    };
    let count = rng.random_range(1..=4);
    let mut stages: Vec<Stage> = (0..count).map(|_| stage(rng)).collect();
    let grid = matches!(spec.output, OutputShape::Grid { .. });
    let unet = grid && dims == Dims::TwoD && rng.random_bool(0.75);

    if unet {
        let strides: Vec<usize> = std::iter::once(stem.stride)
            .chain(stages.iter().map(|stage| stage.stride))
            .filter(|&stride| stride > 1)
            .collect();

        for &stride in strides.iter().rev() {
            stages.push(Stage {
                block: Block::Decoder,
                stride,
                ..stage(rng)
            });
        }
    }

    let head = if unet {
        Head::PerCell
    } else {
        one_of(
            rng,
            &[Head::GlobalAvgPool, Head::GlobalAvgPool, Head::Flatten],
        )
    };
    let widths: Vec<usize> = [8, 16, 32]
        .into_iter()
        .filter(|&width| width <= limits.max_width)
        .collect();
    let layers = if widths.is_empty() {
        0
    } else {
        rng.random_range(0..=limits.max_hidden_layers.min(2))
    };
    let hidden = (0..layers).map(|_| one_of(rng, &widths)).collect();

    Genome {
        arch: Arch::Cnn(CnnGenome {
            dims,
            stem,
            stages,
            head,
        }),
        hidden,
        activation: one_of(rng, &ACTIVATIONS),
        optimizer: one_of(rng, &OPTIMIZERS),
        learning_rate: log_uniform_f32(rng, 1e-3, 0.1),
        l2: if rng.random_bool(0.5) {
            0.0
        } else {
            log_uniform_f32(rng, 1e-6, 1e-3)
        },
        batch_size: one_of(rng, &BATCH_SIZES[..4]),
        epochs: log_uniform_f32(rng, MIN_EPOCHS as f32, 20.0).round() as usize,
        init_scale: log_uniform_f32(rng, 0.5, 1.5),
        seed: rng.random(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rand::SeedableRng;

    use super::*;
    use crate::cnn::families::base_models_for;
    use crate::cnn::genome::testing;
    use crate::config::ModelKinds;
    use crate::task::TaskKind;
    use crate::variation::{CNN_GENERATOR, RandomMutator, VariantGenerator};

    /// A task of each kind of CNN input and output: images with classes, sequences with
    /// classes, and a grid to a grid.
    const TASKS: [TaskKind; 3] = [TaskKind::Shapes, TaskKind::Motif1d, TaskKind::MirrorGrid];

    /// Valid parents for `task`: the base models that fit it, 20 random CNNs, and 300
    /// children of these by random operators, so that deeper, wider and other forms occur.
    fn parents(task: TaskKind, seed: u64) -> Vec<Genome> {
        let limits = Limits::default();
        let spec = task.spec();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let (base, _) = base_models_for(&spec, &limits);
        let mut pool: Vec<Genome> = base.into_iter().map(|(_, genome)| genome).collect();

        for _ in 0..20 {
            pool.push(random(&mut rng, &limits, &spec).expect("a random CNN"));
        }

        for _ in 0..300 {
            let parent = &pool[rng.random_range(0..pool.len())];
            let (child, _) = mutate(parent, 0.3, &mut rng, &limits, &spec).expect("a child");

            pool.push(child);
        }

        for genome in &pool {
            assert!(
                genome.validate(&limits, &spec).is_ok(),
                "{}",
                genome.summary()
            );
        }

        pool
    }

    /// The shape check promises that a valid genome trains. The stage layouts of
    /// `cnn::train` hold each block to that one by one; this test holds the forms that the
    /// operators make to it. Every genome of `parents`, on each task, trains one step on 4
    /// samples: the base models, the random CNNs and their children.
    #[test]
    fn the_parents_and_children_of_the_operators_train_one_step() {
        use crate::cnn::train::testing::{head, trains_one_step};
        use crate::task::{Split, generate};

        for (&task, seed) in TASKS.iter().zip(40..) {
            let spec = task.spec();
            let data = head(&generate(&spec, Split::Train, 1), 4);

            for genome in parents(task, seed) {
                trains_one_step(&genome, &spec, &data);
            }
        }
    }

    #[test]
    fn every_operator_keeps_the_genome_valid_on_10_000_draws() {
        let limits = Limits::default();
        let pools: Vec<(TaskSpec, Vec<Genome>)> = TASKS
            .iter()
            .zip(40..)
            .map(|(&task, seed)| (task.spec(), parents(task, seed)))
            .collect();
        let mut rng = ChaCha8Rng::seed_from_u64(43);

        for operator in OPERATORS {
            let mut children = 0;

            for draw in 0..10_000 {
                let (spec, pool) = &pools[draw % pools.len()];
                let parent = &pool[rng.random_range(0..pool.len())];
                let Some(child) = operator.mutate(parent, 0.3, &mut rng, &limits, spec) else {
                    continue;
                };

                assert!(
                    child.validate(&limits, spec).is_ok(),
                    "{}: {}",
                    operator.name(),
                    child.summary()
                );
                assert_ne!(child.seed, parent.seed);
                assert_eq!(
                    operator == Operator::Training,
                    architecture(&child) == architecture(parent),
                    "{}",
                    operator.name()
                );
                children += 1;
            }

            // Every draw that gives a child gives a valid one. Most draws give one: the
            // fewest children, 74 % of the draws, came from `RemoveStage` and `Dilation` on
            // the machine that wrote this test.
            eprintln!("{}: {children} children in 10 000 draws", operator.name());
            assert!(children > 4000, "{}: {children}", operator.name());
        }
    }

    /// The genes in which `child` differs from `parent`, by name: "stem width", "stage 2
    /// block", "stages" for another count of stages, "head", "hidden", "activation", and the
    /// names of the training genes.
    fn diff(parent: &Genome, child: &Genome) -> Vec<String> {
        let (Arch::Cnn(p), Arch::Cnn(c)) = (&parent.arch, &child.arch) else {
            return vec!["arch".to_string()];
        };
        let mut out = Vec::new();
        let mut differ = |name: String, differs: bool| {
            if differs {
                out.push(name);
            }
        };
        let stem = [
            ("width", p.stem.width != c.stem.width),
            ("kernel", p.stem.kernel != c.stem.kernel),
            ("stride", p.stem.stride != c.stem.stride),
            ("down", p.stem.down != c.stem.down),
            ("norm", p.stem.norm != c.stem.norm),
            ("activation", p.stem.activation != c.stem.activation),
        ];

        differ("dims".to_string(), p.dims != c.dims);

        for (gene, differs) in stem {
            differ(format!("stem {gene}"), differs);
        }

        if p.stages.len() == c.stages.len() {
            for (i, (a, b)) in p.stages.iter().zip(&c.stages).enumerate() {
                let genes = [
                    ("block", a.block != b.block),
                    ("width", a.width != b.width),
                    ("kernel", a.kernel != b.kernel),
                    ("stride", a.stride != b.stride),
                    ("down", a.down != b.down),
                    ("dilation", a.dilation != b.dilation),
                    ("groups", a.groups != b.groups),
                    ("se", a.se != b.se),
                    ("residual", a.residual != b.residual),
                    ("norm", a.norm != b.norm),
                    ("activation", a.activation != b.activation),
                ];

                for (gene, differs) in genes {
                    differ(format!("stage {} {gene}", i + 1), differs);
                }
            }
        } else {
            differ("stages".to_string(), true);
        }

        differ("head".to_string(), p.head != c.head);
        differ("hidden".to_string(), parent.hidden != child.hidden);
        differ(
            "activation".to_string(),
            parent.activation != child.activation,
        );
        differ("optimizer".to_string(), parent.optimizer != child.optimizer);
        differ(
            "learning_rate".to_string(),
            parent.learning_rate != child.learning_rate,
        );
        differ("l2".to_string(), parent.l2 != child.l2);
        differ(
            "batch_size".to_string(),
            parent.batch_size != child.batch_size,
        );
        differ("epochs".to_string(), parent.epochs != child.epochs);
        differ(
            "init_scale".to_string(),
            parent.init_scale != child.init_scale,
        );

        out
    }

    /// The stages of `longer` without the one at `at`.
    fn without(longer: &[Stage], at: usize) -> Vec<Stage> {
        longer
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != at)
            .map(|(_, stage)| stage.clone())
            .collect()
    }

    /// True when `changes` names one gene of one unit and nothing else but fewer epochs:
    /// one of `genes` of the stem or of a stage.
    fn one_unit_gene(changes: &[String], genes: &[&str]) -> bool {
        let named: Vec<&String> = changes.iter().filter(|c| *c != "epochs").collect();

        named.len() == 1
            && genes.iter().any(|gene| {
                named[0] == &format!("stem {gene}")
                    || (named[0].starts_with("stage ") && named[0].ends_with(&format!(" {gene}")))
            })
    }

    #[test]
    fn each_operator_changes_only_its_genes() {
        let limits = Limits::default();
        let mut rng = ChaCha8Rng::seed_from_u64(44);
        let mut seen: BTreeMap<&str, usize> = BTreeMap::new();

        for (&task, seed) in TASKS.iter().zip(50..) {
            let spec = task.spec();
            let pool = parents(task, seed);

            for operator in OPERATORS {
                for _ in 0..300 {
                    let parent = &pool[rng.random_range(0..pool.len())];
                    let Some(child) = operator.mutate(parent, 0.3, &mut rng, &limits, &spec) else {
                        continue;
                    };
                    let changes = diff(parent, &child);
                    let (Arch::Cnn(p), Arch::Cnn(c)) = (&parent.arch, &child.arch) else {
                        panic!("a CNN gives a CNN");
                    };
                    // An architecture operator keeps the training genes; repair may cut the
                    // epochs of a child that costs more.
                    let training = [
                        "optimizer",
                        "learning_rate",
                        "l2",
                        "batch_size",
                        "init_scale",
                    ];
                    let kept_training = !changes.iter().any(|c| training.contains(&c.as_str()))
                        && child.epochs <= parent.epochs;
                    let only = |names: &[&str]| {
                        changes
                            .iter()
                            .all(|c| c == "epochs" || names.contains(&c.as_str()))
                    };
                    let fits = match operator {
                        Operator::Family => {
                            let blocks = changes.iter().filter(|c| c.ends_with(" block")).count();
                            let stage = changes.first().and_then(|c| c.split(' ').nth(1));

                            blocks == 1
                                && changes.iter().all(|c| {
                                    c == "epochs"
                                        || (stage
                                            .is_some_and(|s| c.starts_with(&format!("stage {s} ")))
                                            && ["block", "width", "down"]
                                                .iter()
                                                .any(|g| c.ends_with(&format!(" {g}"))))
                                })
                        }
                        Operator::InsertStage => {
                            c.stages.len() == p.stages.len() + 1
                                && (0..c.stages.len()).any(|at| {
                                    c.stages[at].stride == 1 && without(&c.stages, at) == p.stages
                                })
                                && only(&["stages"])
                        }
                        Operator::RemoveStage => {
                            p.stages.len() == c.stages.len() + 1
                                && (0..p.stages.len()).any(|at| without(&p.stages, at) == c.stages)
                                && only(&["stages"])
                        }
                        Operator::Width => one_unit_gene(&changes, &["width"]),
                        Operator::Kernel => {
                            let kernels = std::iter::once((p.stem.kernel, c.stem.kernel)).chain(
                                p.stages
                                    .iter()
                                    .zip(&c.stages)
                                    .map(|(a, b)| (a.kernel, b.kernel)),
                            );

                            one_unit_gene(&changes, &["kernel"])
                                && kernels
                                    .into_iter()
                                    .all(|(a, b)| a == b || a.abs_diff(b) == 2)
                        }
                        Operator::Stride => {
                            let strides = std::iter::once((p.stem.stride, c.stem.stride)).chain(
                                p.stages
                                    .iter()
                                    .zip(&c.stages)
                                    .map(|(a, b)| (a.stride, b.stride)),
                            );

                            one_unit_gene(&changes, &["stride"])
                                && strides.into_iter().all(|(a, b)| a.abs_diff(b) <= 1)
                        }
                        Operator::Dilation => {
                            one_unit_gene(&changes, &["dilation"])
                                && !changes.iter().any(|c| c.starts_with("stem"))
                                && p.stages.iter().zip(&c.stages).all(|(a, b)| {
                                    a.dilation == b.dilation
                                        || (a.dilation == 1 && b.dilation == 2)
                                        || b.dilation == (2 * a.dilation).min(MAX_DILATION)
                                        || b.dilation == a.dilation / 2
                                })
                        }
                        Operator::Groups => {
                            one_unit_gene(&changes, &["groups"])
                                && c.stages
                                    .iter()
                                    .all(|s| s.groups.is_power_of_two() && s.width % s.groups == 0)
                        }
                        Operator::SqueezeExcite => one_unit_gene(&changes, &["se"]),
                        Operator::Residual => one_unit_gene(&changes, &["residual"]),
                        Operator::Normalization => one_unit_gene(&changes, &["norm"]),
                        Operator::Downsampling => {
                            let units = std::iter::once((p.stem.stride, p.stem.down, c.stem.down))
                                .chain(
                                    p.stages
                                        .iter()
                                        .zip(&c.stages)
                                        .map(|(a, b)| (a.stride, a.down, b.down)),
                                );

                            one_unit_gene(&changes, &["down"])
                                && units.into_iter().all(|(stride, a, b)| a == b || stride > 1)
                        }
                        Operator::Activation => {
                            one_unit_gene(&changes, &["activation"])
                                || (only(&["activation"]) && !child.hidden.is_empty())
                        }
                        Operator::Head => only(&["head"]) && changes.iter().any(|c| c == "head"),
                        Operator::HeadLayers => {
                            only(&["hidden"]) && changes.iter().any(|c| c == "hidden")
                        }
                        Operator::Training => {
                            only(&training) || only(&["epochs"]) || changes.is_empty()
                        }
                    };

                    assert!(
                        fits && (operator == Operator::Training || kept_training),
                        "{}: {changes:?}\n{}\n{}",
                        operator.name(),
                        parent.summary(),
                        child.summary()
                    );
                    *seen.entry(operator.name()).or_insert(0) += 1;
                }
            }
        }

        assert_eq!(seen.len(), OPERATORS.len(), "{seen:?}");
    }

    #[test]
    fn mutate_draws_every_operator_and_gives_valid_children() {
        let limits = Limits::default();
        let spec = TaskKind::Shapes.spec();
        let pool = parents(TaskKind::Shapes, 60);
        let mut rng = ChaCha8Rng::seed_from_u64(61);
        let mut chosen: BTreeMap<&str, usize> = BTreeMap::new();
        let mut again = ChaCha8Rng::seed_from_u64(61);

        for _ in 0..4000 {
            let parent = &pool[rng.random_range(0..pool.len())];
            let (child, operator) = mutate(parent, 0.3, &mut rng, &limits, &spec).expect("a child");
            let replay = &pool[again.random_range(0..pool.len())];

            assert!(
                child.validate(&limits, &spec).is_ok(),
                "{}",
                child.summary()
            );
            // The same draws give the same child.
            assert_eq!(
                mutate(replay, 0.3, &mut again, &limits, &spec),
                Some((child, operator))
            );
            *chosen.entry(operator.name()).or_insert(0) += 1;
        }

        // Uniform choice gives each operator 250 of the 4000 on average. An operator that
        // finds no child gives way to another, so the ones that fail more often come less.
        eprintln!("{chosen:?}");
        assert_eq!(chosen.len(), OPERATORS.len());
        assert!(chosen.values().all(|&n| n > 120), "{chosen:?}");
    }

    #[test]
    fn an_mlp_or_a_broken_cnn_gets_no_child() {
        let limits = Limits::default();
        let spec = TaskKind::Shapes.spec();
        let mut rng = ChaCha8Rng::seed_from_u64(62);
        let mlp = Genome {
            arch: Arch::Mlp,
            ..testing::lenet()
        };
        let mut broken = testing::lenet();

        // A 9x9 kernel does not fit the 8x8 input of the LeNet stage.
        if let Arch::Cnn(cnn) = &mut broken.arch {
            cnn.stages[0].kernel = 9;
        }

        for parent in [&mlp, &broken] {
            assert!(mutate(parent, 0.3, &mut rng, &limits, &spec).is_none());

            for operator in OPERATORS {
                assert!(
                    operator
                        .mutate(parent, 0.3, &mut rng, &limits, &spec)
                        .is_none()
                );
            }

            assert!(crossover(parent, &mlp, &mut rng, &limits, &spec).is_none());
        }
    }

    #[test]
    fn crossover_of_two_base_models_of_different_families_gives_valid_children() {
        let limits = Limits::default();
        let mutator = RandomMutator {
            models: ModelKinds::Cnn,
            ..RandomMutator::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(45);
        let stages = format!("{CNN_GENERATOR}/crossover");

        for task in TASKS {
            let spec = task.spec();
            let (base, _) = base_models_for(&spec, &limits);
            let mut pairs = 0;
            let mut whole = Vec::new();

            for (fa, a) in &base {
                for (fb, b) in &base {
                    if fa == fb {
                        continue;
                    }

                    let mut exchanged = false;

                    for _ in 0..10 {
                        let (child, how) = mutator.propose_named(&[a, b], &mut rng, &limits, &spec);

                        assert!(
                            child.validate(&limits, &spec).is_ok(),
                            "{} x {}: {}",
                            fa.name(),
                            fb.name(),
                            child.summary()
                        );
                        assert!(!child.arch.is_mlp());
                        exchanged |= how == stages;
                    }

                    pairs += 1;

                    if !exchanged {
                        whole.push(format!("{} x {}", fa.name(), fb.name()));
                    }
                }
            }

            // Some pairs exchange no stages, and their children take the architecture of
            // one parent. The MobileNetV2 base model pools to 4x4 in its stem. The dilated
            // CNN never leaves 8x8, so the two share no resolution. With GoogLeNet, ResNet,
            // SqueezeNet and ConvNeXt every cut at 4x4 puts the inverted residuals, which
            // expand their input 6 times, behind wider stages: each such child was
            // estimated at 22 to 54 s, above the budget of 20 s. The TCN keeps the length
            // of 64 along its stages, and every 1-D twin halves it in its stem.
            eprintln!(
                "{}: {} of {pairs} pairs exchange stages; not: {whole:?}",
                task.name(),
                pairs - whole.len()
            );
            assert!(
                whole
                    .iter()
                    .all(|pair| pair.contains("mobilenet-v2") || pair.contains("tcn")),
                "{whole:?}"
            );
            assert!(5 * whole.len() <= pairs, "{whole:?}");
        }
    }

    #[test]
    fn cuts_meet_at_matching_resolutions_and_children_take_whole_stages() {
        let shape = |side: usize| Shape {
            channels: 8,
            size: vec![side, side],
        };

        // The first parent: a stem at 8x8 and one stage to 4x4. The second: a stem at 8x8, a
        // stage that keeps 8x8 and one to 4x4.
        let first = [shape(8), shape(4)];
        let second = [shape(8), shape(8), shape(4)];
        let cut = |i, i2, j, j2| Cut { i, i2, j, j2 };

        assert_eq!(
            cuts(&first, &second),
            vec![
                cut(0, 0, 0, 1),
                cut(0, 1, 0, 1),
                cut(0, 1, 0, 2),
                cut(0, 1, 1, 2)
            ]
        );

        let base = testing::lenet();
        let parent = |stages: Vec<Stage>, head: Head, hidden: Vec<usize>| Genome {
            arch: Arch::Cnn(CnnGenome {
                dims: Dims::TwoD,
                stem: Stem {
                    width: hidden.len() + 4,
                    kernel: 3,
                    stride: 2,
                    down: Downsample::MaxPool,
                    norm: Norm::None,
                    activation: CnnActivation::Relu,
                },
                stages,
                head,
            }),
            hidden,
            ..base.clone()
        };
        let stage = |width| testing::plain(width, 3, CnnActivation::Relu);
        let a = parent(vec![stage(10)], Head::Flatten, vec![32]);
        let b = parent(vec![stage(20), stage(21)], Head::GlobalAvgPool, Vec::new());
        let (Arch::Cnn(fa), Arch::Cnn(fb)) = (&a.arch, &b.arch) else {
            panic!("CNNs");
        };
        let child = |cut| match join(&a, fa, &b, fb, cut).arch {
            Arch::Cnn(cnn) => (
                cnn.stem.width,
                cnn.stages.iter().map(|s| s.width).collect::<Vec<_>>(),
                cnn.head,
            ),
            Arch::Mlp => panic!("a CNN"),
        };

        // An inner run of the second parent between the stem and the stage of the first.
        assert_eq!(child(cut(0, 0, 0, 1)), (5, vec![20, 10], Head::Flatten));
        // The child ends with a run that stops before the end of the second parent, so the
        // head stays the first parent's.
        assert_eq!(child(cut(0, 1, 0, 1)), (5, vec![20], Head::Flatten));
        // One-point crossover: the tail of the second parent, with its head and dense layers.
        assert_eq!(
            child(cut(0, 1, 0, 2)),
            (5, vec![20, 21], Head::GlobalAvgPool)
        );
        assert_eq!(child(cut(0, 1, 1, 2)), (5, vec![21], Head::GlobalAvgPool));
        assert!(join(&a, fa, &b, fb, cut(0, 1, 1, 2)).hidden.is_empty());
        assert_eq!(join(&a, fa, &b, fb, cut(0, 1, 0, 1)).hidden, vec![32]);
    }

    #[test]
    fn random_cnns_are_valid_deterministic_and_fit_their_task() {
        let limits = Limits::default();

        for task in TaskKind::ALL {
            let spec = task.spec();
            let mut rng = ChaCha8Rng::seed_from_u64(46);
            let mut again = ChaCha8Rng::seed_from_u64(46);

            let Some(dims) = dims_for(&spec) else {
                assert!(random(&mut rng, &limits, &spec).is_none());
                assert!(smallest(&spec).is_none());

                continue;
            };

            let mut per_cell = 0;

            for _ in 0..200 {
                let genome = random(&mut rng, &limits, &spec).expect("a random CNN");
                let Arch::Cnn(cnn) = &genome.arch else {
                    panic!("a CNN");
                };

                assert!(
                    genome.validate(&limits, &spec).is_ok(),
                    "{}",
                    genome.summary()
                );
                assert_eq!(cnn.dims, dims);
                assert_eq!(random(&mut again, &limits, &spec), Some(genome.clone()));
                per_cell += usize::from(cnn.head == Head::PerCell);
            }

            // The U-Net form only where the task has a grid of outputs, and there often.
            if matches!(spec.output, OutputShape::Grid { .. }) {
                assert!(per_cell > 100, "{per_cell}");
            } else {
                assert_eq!(per_cell, 0);
            }

            let small = smallest(&spec).expect("the smallest CNN");
            let channels = match spec.input {
                InputShape::Image { channels, .. } | InputShape::Sequence { channels, .. } => {
                    channels
                }
                InputShape::Flat => 0,
            };

            // A weight per input channel and a bias, a weight and a bias, then the output
            // layer of one weight and one bias per output.
            assert!(small.validate(&limits, &spec).is_ok(), "{}", task.name());
            assert_eq!(small.param_count(&spec), channels + 1 + 2 + 2 * spec.n_out);
        }
    }

    #[test]
    fn widths_round_to_the_step_of_their_kind() {
        assert_eq!(width_step(Block::Plain, 1, Norm::Batch), 1);
        assert_eq!(width_step(Block::Bottleneck, 1, Norm::Batch), 4);
        assert_eq!(width_step(Block::Plain, 3, Norm::Group), 12);
        assert_eq!(width_step(Block::Fire, 8, Norm::Group), 8);
        assert_eq!(width_step(Block::Plain, usize::MAX, Norm::None), 1);
        assert_eq!(round_to(10, 4), 12);
        assert_eq!(round_to(9, 4), 8);
        assert_eq!(round_to(0, 4), 4);
        assert_eq!(round_to(usize::MAX, 3), 255);
        assert_eq!(round_to(300, 1), MAX_CHANNELS);
    }

    #[test]
    fn the_operators_have_distinct_names() {
        let names: std::collections::BTreeSet<&str> = OPERATORS.iter().map(|o| o.name()).collect();

        assert_eq!(names.len(), OPERATORS.len());
    }
}
