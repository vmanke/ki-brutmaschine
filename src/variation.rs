//! Variant generator: makes new genomes from parent genomes.
//!
//! The trait allows other generators later (for example an LLM-based one).
//! Every generator must return a genome that passes `Genome::validate`.
//!
//! An MLP child is made as before CNNs: random changes of its genes, or uniform crossover,
//! with repair to valid ranges. A child of a CNN parent comes from `cnn::variation`: one
//! operator per mutation, and the exchange of whole stages in a crossover of two CNNs
//! (ticket 05 of the CNN spec, ADR 0008).

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::cnn::shape::TIME_BUDGET_FLOPS_PER_SECOND;
use crate::cnn::variation as cnn_variation;
use crate::config::{Limits, ModelKinds};
use crate::genome::{
    ACTIVATIONS, Arch, BATCH_SIZES, Genome, INIT_SCALE_MAX, INIT_SCALE_MIN, L2_MAX, L2_MIN, LR_MAX,
    LR_MIN, MIN_EPOCHS, OPTIMIZERS,
};
use crate::task::TaskSpec;
use crate::util::{log_uniform_f32, normal_f32};

/// Number of mutable genes (the seed is always renewed and not counted).
const GENE_COUNT: usize = 8;

/// The first training gene. Genes 0 and 1 (hidden layers and activation) shape the
/// network; the genes from here on train it.
const FIRST_TRAINING_GENE: usize = 2;

/// Crossovers tried before a child of a CNN parent gives way to a random genome.
const CNN_CROSSOVER_TRIES: usize = 4;

/// The name of `RandomMutator` for the children of MLPs.
pub const MLP_GENERATOR: &str = "random-mutator-v1";

/// The start of the name of `RandomMutator` for a child with a CNN parent. Then come "/" and
/// how it was made: an operator name of `cnn::variation`, "crossover" (stages exchanged),
/// "crossover-genes" (the architecture of one parent and the training genes of both), or
/// "random" (a random genome, when the parents give no valid child).
pub const CNN_GENERATOR: &str = "cnn-mutator-v1";

pub trait VariantGenerator: Send + Sync {
    /// Short name for the audit log.
    fn name(&self) -> &str;

    /// Make one child. `parents` has length 1 (mutation) or 2 (crossover).
    /// The result must be valid for `limits` and `spec`.
    fn propose(
        &self,
        parents: &[&Genome],
        rng: &mut ChaCha8Rng,
        limits: &Limits,
        spec: &TaskSpec,
    ) -> Genome;

    /// Make one child as `propose` does, and name how it was made, for the archive. The
    /// default names the generator.
    fn propose_named(
        &self,
        parents: &[&Genome],
        rng: &mut ChaCha8Rng,
        limits: &Limits,
        spec: &TaskSpec,
    ) -> (Genome, String) {
        let child = self.propose(parents, rng, limits, spec);

        (child, self.name().to_string())
    }
}

/// Classic random mutation and uniform crossover, with repair to valid ranges, and the CNN
/// operators of `cnn::variation` for the children of CNNs.
#[derive(Clone, Copy, Debug)]
pub struct RandomMutator {
    /// Probability that each gene is mutated.
    pub gene_rate: f64,
    /// The kinds of model of the run. A random genome that takes the place of a child the
    /// parents cannot give is of these kinds.
    pub models: ModelKinds,
}

impl Default for RandomMutator {
    fn default() -> Self {
        RandomMutator {
            gene_rate: 0.3,
            models: ModelKinds::Mlp,
        }
    }
}

impl RandomMutator {
    /// A random genome of the kinds of the run, for a child that the parents cannot give and
    /// for a restart: an MLP, a CNN, or either with equal chance. An MLP when no CNN fits the
    /// task. For an MLP run this is `Genome::random`, with the same draws.
    pub fn random(&self, rng: &mut ChaCha8Rng, limits: &Limits, spec: &TaskSpec) -> Genome {
        let cnn = match self.models {
            ModelKinds::Mlp => false,
            ModelKinds::Cnn => true,
            ModelKinds::Both => rng.random_bool(0.5),
        };

        if cnn && let Some(genome) = cnn_variation::random(rng, limits, spec) {
            return genome;
        }

        Genome::random(rng, limits, spec)
    }
}

/// The name of a child with a CNN parent: `CNN_GENERATOR`, "/" and how it was made.
fn cnn_name(how: &str) -> String {
    format!("{CNN_GENERATOR}/{how}")
}

impl VariantGenerator for RandomMutator {
    fn name(&self) -> &str {
        MLP_GENERATOR
    }

    fn propose(
        &self,
        parents: &[&Genome],
        rng: &mut ChaCha8Rng,
        limits: &Limits,
        spec: &TaskSpec,
    ) -> Genome {
        self.propose_named(parents, rng, limits, spec).0
    }

    fn propose_named(
        &self,
        parents: &[&Genome],
        rng: &mut ChaCha8Rng,
        limits: &Limits,
        spec: &TaskSpec,
    ) -> (Genome, String) {
        match parents {
            [] => (self.random(rng, limits, spec), self.name().to_string()),
            [one] if one.arch.is_mlp() => (
                mutate(one, self.gene_rate, rng, limits, spec),
                self.name().to_string(),
            ),
            // A valid CNN parent always gives a valid child (see `cnn::variation`). A parent
            // that breaks a rule itself may give none; then a random genome takes its place.
            [one] => match cnn_variation::mutate(one, self.gene_rate, rng, limits, spec) {
                Some((child, operator)) => (child, cnn_name(operator.name())),
                None => (self.random(rng, limits, spec), cnn_name("random")),
            },
            [a, b, ..] if a.arch.is_mlp() && b.arch.is_mlp() => {
                let child = crossover(a, b, rng);

                (repair(child, limits, spec), self.name().to_string())
            }
            [a, b, ..] => {
                // Two CNNs exchange whole stages where their resolutions match.
                if !a.arch.is_mlp()
                    && !b.arch.is_mlp()
                    && let Some(child) = cnn_variation::crossover(a, b, rng, limits, spec)
                {
                    return (child, cnn_name("crossover"));
                }

                // An MLP and a CNN, or two CNNs without a valid cut: the child takes the
                // architecture of one parent, which may break the limits if that parent
                // does. Then it is drawn again, and after `CNN_CROSSOVER_TRIES` failures a
                // random genome takes its place.
                for _ in 0..CNN_CROSSOVER_TRIES {
                    let child = repair(crossover(a, b, rng), limits, spec);

                    if child.validate(limits, spec).is_ok() {
                        return (child, cnn_name("crossover-genes"));
                    }
                }

                (self.random(rng, limits, spec), cnn_name("random"))
            }
        }
    }
}

/// Mutate one genome. Always gets a fresh training seed.
///
/// An MLP may change every gene. A CNN gets one operator of `cnn::variation`, which keeps it
/// valid when it was valid. A CNN that breaks a rule itself may find no valid child; then it
/// keeps its architecture, changes its training genes and stays invalid, and
/// `RandomMutator::propose` draws a random genome instead.
pub fn mutate(
    parent: &Genome,
    gene_rate: f64,
    rng: &mut ChaCha8Rng,
    limits: &Limits,
    spec: &TaskSpec,
) -> Genome {
    if !parent.arch.is_mlp() {
        if let Some((child, _)) = cnn_variation::mutate(parent, gene_rate, rng, limits, spec) {
            return child;
        }

        let mut g = parent.clone();

        mutate_training_genes(&mut g, gene_rate, rng, limits);
        g.seed = rng.random();

        return repair(g, limits, spec);
    }

    let rate = clamp_rate(gene_rate);
    let mut g = parent.clone();
    let mut changed = false;

    for gene in 0..GENE_COUNT {
        if rng.random_bool(rate) {
            mutate_gene(&mut g, gene, rng, limits);
            changed = true;
        }
    }

    // A child that only differs by its seed would waste a training run.
    if !changed {
        let gene = rng.random_range(0..GENE_COUNT);

        mutate_gene(&mut g, gene, rng, limits);
    }

    g.seed = rng.random();

    repair(g, limits, spec)
}

/// The gene rate as a probability: clamped to [0, 1], and 0 when it is not finite.
fn clamp_rate(gene_rate: f64) -> f64 {
    if gene_rate.is_finite() {
        gene_rate.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Change the training genes of `g` (optimizer, learning rate, l2, batch size, epochs and
/// init scale), each with probability `gene_rate`, and at least one. The seed stays. This is
/// the CNN operator `Training`; before ticket 05 it was the whole mutation of a CNN.
pub(crate) fn mutate_training_genes(
    g: &mut Genome,
    gene_rate: f64,
    rng: &mut ChaCha8Rng,
    limits: &Limits,
) {
    let rate = clamp_rate(gene_rate);
    let mut changed = false;

    for gene in FIRST_TRAINING_GENE..GENE_COUNT {
        if rng.random_bool(rate) {
            mutate_gene(g, gene, rng, limits);
            changed = true;
        }
    }

    if !changed {
        let gene = rng.random_range(FIRST_TRAINING_GENE..GENE_COUNT);

        mutate_gene(g, gene, rng, limits);
    }
}

/// Add, remove or resize one dense layer of `g.hidden`, as the structural mutation of an
/// MLP does. For a CNN these are the dense layers of its head: the operator `HeadLayers`.
pub(crate) fn mutate_head_layers(g: &mut Genome, rng: &mut ChaCha8Rng, limits: &Limits) {
    mutate_layers(g, rng, limits);
}

/// Change one gene by a small random step.
fn mutate_gene(g: &mut Genome, gene: usize, rng: &mut ChaCha8Rng, limits: &Limits) {
    match gene {
        0 => mutate_layers(g, rng, limits),
        1 => g.activation = ACTIVATIONS[rng.random_range(0..ACTIVATIONS.len())],
        2 => g.optimizer = OPTIMIZERS[rng.random_range(0..OPTIMIZERS.len())],
        3 => g.learning_rate *= (0.5 * normal_f32(rng)).exp(),
        4 => {
            if g.l2 <= 0.0 || rng.random_bool(0.2) {
                g.l2 = if g.l2 <= 0.0 {
                    log_uniform_f32(rng, 1e-6, L2_MAX)
                } else {
                    0.0
                };
            } else {
                g.l2 *= normal_f32(rng).exp();
            }
        }
        5 => {
            let pos = BATCH_SIZES
                .iter()
                .position(|&b| b == g.batch_size)
                .unwrap_or(0);
            let next = if rng.random_bool(0.5) {
                pos.saturating_add(1)
            } else {
                pos.saturating_sub(1)
            };

            g.batch_size = BATCH_SIZES[next.min(BATCH_SIZES.len() - 1)];
        }
        6 => {
            let factor = (0.4 * normal_f32(rng)).exp();

            g.epochs = (g.epochs as f32 * factor).round() as usize;
        }
        _ => g.init_scale *= (0.2 * normal_f32(rng)).exp(),
    }
}

/// Structural mutation: add, remove or resize one hidden layer.
fn mutate_layers(g: &mut Genome, rng: &mut ChaCha8Rng, limits: &Limits) {
    let can_add = g.hidden.len() < limits.max_hidden_layers && limits.max_width > 0;
    let can_remove = !g.hidden.is_empty();
    let choice = rng.random_range(0..4);

    if choice == 0 && can_add {
        let pos = rng.random_range(0..=g.hidden.len());
        let width = match g.hidden.get(pos.saturating_sub(1)) {
            Some(&w) => w,
            None => 8.min(limits.max_width),
        };

        g.hidden.insert(pos, width.max(1));
    } else if choice == 1 && can_remove {
        let pos = rng.random_range(0..g.hidden.len());

        g.hidden.remove(pos);
    } else if can_remove {
        let pos = rng.random_range(0..g.hidden.len());
        let factor = (0.35 * normal_f32(rng)).exp();
        let width = (g.hidden[pos] as f32 * factor).round() as usize;

        // Always move by at least one neuron.
        g.hidden[pos] = if width == g.hidden[pos] {
            if rng.random_bool(0.5) {
                width + 1
            } else {
                width.saturating_sub(1)
            }
        } else {
            width
        };
    } else if can_add {
        g.hidden.push(8.min(limits.max_width).max(1));
    }
}

/// Uniform crossover of two genomes. Hidden layers are taken whole from one parent.
///
/// Once a CNN takes part, the architecture comes whole from one parent: its `arch`, and
/// its `hidden` and `activation`, which for a CNN form the head. The training genes are
/// mixed as for two MLPs.
pub fn crossover(a: &Genome, b: &Genome, rng: &mut ChaCha8Rng) -> Genome {
    let mut pick = || rng.random_bool(0.5);

    let (arch, hidden, activation) = if a.arch.is_mlp() && b.arch.is_mlp() {
        let hidden = if pick() {
            a.hidden.clone()
        } else {
            b.hidden.clone()
        };
        let activation = if pick() { a.activation } else { b.activation };

        (Arch::Mlp, hidden, activation)
    } else {
        let shape = if pick() { a } else { b };

        (shape.arch.clone(), shape.hidden.clone(), shape.activation)
    };
    let optimizer = if pick() { a.optimizer } else { b.optimizer };
    let learning_rate = if pick() {
        a.learning_rate
    } else {
        b.learning_rate
    };
    let l2 = if pick() { a.l2 } else { b.l2 };
    let batch_size = if pick() { a.batch_size } else { b.batch_size };
    let epochs = if pick() { a.epochs } else { b.epochs };
    let init_scale = if pick() { a.init_scale } else { b.init_scale };
    let seed = rng.random();

    Genome {
        arch,
        hidden,
        activation,
        optimizer,
        learning_rate,
        l2,
        batch_size,
        epochs,
        init_scale,
        seed,
    }
}

/// Clamp and fix a genome so it passes `validate`.
/// Shrinks layers until `max_params` and `max_train_flops` hold.
///
/// Order: first cut epochs (down to `MIN_EPOCHS`) to meet the FLOP limit,
/// then halve the widest layer, then drop layers. Feasible limits always give a valid genome.
///
/// A CNN keeps its architecture: only its training genes are clamped and its epochs cut,
/// to fit `max_train_flops` and the time budget of `Genome::validate`. If the architecture
/// itself breaks a shape rule or a limit, the result stays invalid, and
/// `RandomMutator::propose` draws another child.
pub fn repair(g: Genome, limits: &Limits, spec: &TaskSpec) -> Genome {
    let mut g = g;

    g.learning_rate = clamp_or(g.learning_rate, LR_MIN, LR_MAX, 0.01);
    g.l2 = clamp_or(g.l2, L2_MIN, L2_MAX, 0.0);
    g.init_scale = clamp_or(g.init_scale, INIT_SCALE_MIN, INIT_SCALE_MAX, 1.0);
    g.batch_size = nearest_batch(g.batch_size);

    let max_epochs = limits.max_epochs.max(MIN_EPOCHS);

    g.epochs = g.epochs.clamp(MIN_EPOCHS, max_epochs);

    if !g.arch.is_mlp() {
        return fit_cnn_epochs(g, limits, spec);
    }

    g.hidden.truncate(limits.max_hidden_layers);

    if limits.max_width == 0 {
        g.hidden.clear();
    }

    for w in g.hidden.iter_mut() {
        *w = (*w).clamp(1, limits.max_width.max(1));
    }

    loop {
        let params = g.param_count(spec);

        if params <= limits.max_params {
            let per_epoch = 6u64
                .saturating_mul(params as u64)
                .saturating_mul(spec.n_train as u64);
            let affordable = limits
                .max_train_flops
                .checked_div(per_epoch)
                .unwrap_or(u64::MAX);

            if affordable >= MIN_EPOCHS as u64 {
                g.epochs = g.epochs.min(affordable.min(usize::MAX as u64) as usize);

                return g;
            }
        }

        if !shrink(&mut g) {
            // Infeasible limits: nothing left to shrink. Config::validate prevents this.
            g.epochs = MIN_EPOCHS;

            return g;
        }
    }
}

/// Cut the epochs of a CNN genome to meet `max_train_flops` and the time budget of
/// `Genome::validate`, down to `MIN_EPOCHS`.
///
/// Leaves the genome as it is when its shapes do not fit, when it has too many
/// parameters, or when even `MIN_EPOCHS` epochs cost too much.
fn fit_cnn_epochs(g: Genome, limits: &Limits, spec: &TaskSpec) -> Genome {
    let mut g = g;

    let Ok(plan) = g.cnn_plan(spec) else {
        return g;
    };

    if plan.params > limits.max_params {
        return g;
    }

    let per_epoch = plan
        .train_flops_per_sample()
        .saturating_mul(spec.n_train as u64);
    let affordable = limits
        .max_train_flops
        .checked_div(per_epoch)
        .unwrap_or(u64::MAX);

    if affordable < MIN_EPOCHS as u64 {
        return g;
    }

    // The most epochs whose estimated time fits the budget, by the rule of `validate`.
    let budget = limits.max_train_flops as f64 / TIME_BUDGET_FLOPS_PER_SECOND;
    let mut epochs = g.epochs.min(affordable.min(usize::MAX as u64) as usize);
    let seconds = |epochs: usize| plan.train_seconds(spec.n_train as f64 * epochs as f64);

    if seconds(epochs) > budget {
        let per_epoch = seconds(1);

        if per_epoch > 0.0 {
            epochs = epochs.min((budget / per_epoch).floor() as usize);
        }

        while epochs > MIN_EPOCHS && seconds(epochs) > budget {
            epochs -= 1;
        }
    }

    if epochs >= MIN_EPOCHS && seconds(epochs) <= budget {
        g.epochs = epochs;
    }

    g
}

/// Halve the widest layer, or drop the last layer when all widths are 1.
/// Returns false if the genome is already a linear model.
fn shrink(g: &mut Genome) -> bool {
    let widest = g
        .hidden
        .iter()
        .enumerate()
        .max_by_key(|&(i, &w)| (w, usize::MAX - i));

    match widest {
        Some((i, &w)) if w > 1 => {
            g.hidden[i] = w / 2;

            true
        }
        Some(_) => {
            g.hidden.pop();

            true
        }
        None => false,
    }
}

fn clamp_or(value: f32, lo: f32, hi: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(lo, hi)
    } else {
        fallback
    }
}

fn nearest_batch(size: usize) -> usize {
    let mut best = BATCH_SIZES[0];

    for &b in BATCH_SIZES.iter() {
        if b.abs_diff(size) < best.abs_diff(size) {
            best = b;
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnn::genome::{Dims, testing};
    use crate::genome::{Activation, Optimizer};
    use crate::task::TaskKind;
    use rand::SeedableRng;

    fn sample() -> Genome {
        Genome {
            arch: Arch::Mlp,
            hidden: vec![16, 8],
            activation: Activation::Relu,
            optimizer: Optimizer::Momentum,
            learning_rate: 0.05,
            l2: 1e-4,
            batch_size: 64,
            epochs: 50,
            init_scale: 1.0,
            seed: 1,
        }
    }

    #[test]
    fn mutate_gives_valid_new_genome() {
        let spec = TaskKind::TwoSpirals.spec();
        let limits = Limits::default();
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let parent = sample();

        for _ in 0..500 {
            let child = mutate(&parent, 0.3, &mut rng, &limits, &spec);

            assert!(child.validate(&limits, &spec).is_ok(), "{child:?}");
            assert_ne!(child.id(), parent.id());
        }
    }

    #[test]
    fn mutate_with_zero_rate_still_changes_a_gene() {
        let spec = TaskKind::TwoSpirals.spec();
        let limits = Limits::default();
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let parent = sample();
        let mut differs = 0;

        for _ in 0..100 {
            let mut child = mutate(&parent, 0.0, &mut rng, &limits, &spec);

            child.seed = parent.seed;

            if child != parent {
                differs += 1;
            }
        }

        // Some steps (e.g. batch size at the edge) can land on the old value; most must not.
        assert!(differs > 80, "{differs}");
    }

    #[test]
    fn crossover_takes_genes_from_parents() {
        let mut rng = ChaCha8Rng::seed_from_u64(6);
        let a = sample();
        let mut b = sample();

        b.hidden = vec![4];
        b.activation = Activation::Sine;
        b.learning_rate = 0.001;

        for _ in 0..50 {
            let c = crossover(&a, &b, &mut rng);

            assert!(c.hidden == a.hidden || c.hidden == b.hidden);
            assert!(c.activation == a.activation || c.activation == b.activation);
            assert!(c.learning_rate == a.learning_rate || c.learning_rate == b.learning_rate);
        }
    }

    #[test]
    fn repair_fixes_garbage() {
        let spec = TaskKind::TwoSpirals.spec();
        let limits = Limits::default();
        let g = Genome {
            arch: Arch::Mlp,
            hidden: vec![0, 10_000, 3, 7, 9, 11],
            activation: Activation::Gelu,
            optimizer: Optimizer::Adam,
            learning_rate: f32::NAN,
            l2: f32::INFINITY,
            batch_size: 1000,
            epochs: 1_000_000,
            init_scale: -3.0,
            seed: 0,
        };
        let fixed = repair(g, &limits, &spec);

        assert!(fixed.validate(&limits, &spec).is_ok(), "{fixed:?}");
        assert_eq!(fixed.batch_size, 256);
    }

    #[test]
    fn repair_meets_tight_flop_limit() {
        let spec = TaskKind::TwoSpirals.spec();
        let limits = Limits {
            max_train_flops: 6 * 200 * spec.n_train as u64 * 10,
            ..Limits::default()
        };
        let mut g = sample();

        g.hidden = vec![64, 64];
        g.epochs = 400;

        let fixed = repair(g, &limits, &spec);

        assert!(fixed.validate(&limits, &spec).is_ok(), "{fixed:?}");
    }

    /// The architecture of a genome: its arch, hidden layers and activation.
    fn architecture(g: &Genome) -> (Arch, Vec<usize>, Activation) {
        (g.arch.clone(), g.hidden.clone(), g.activation)
    }

    /// Valid CNN genomes for the shapes task.
    fn valid_cnns(count: usize, seed: u64) -> Vec<Genome> {
        let spec = TaskKind::Shapes.spec();
        let limits = Limits::default();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut found = vec![testing::lenet(), testing::vgg(), testing::grouped_dilated()];

        while found.len() < count {
            let g = testing::plausible(&mut rng, Dims::TwoD);

            if g.validate(&limits, &spec).is_ok() {
                found.push(g);
            }
        }

        found
    }

    #[test]
    fn a_cnn_child_gets_one_operator_and_stays_valid() {
        let spec = TaskKind::Shapes.spec();
        let limits = Limits::default();
        let m = RandomMutator {
            models: ModelKinds::Cnn,
            ..RandomMutator::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(21);
        let mut operators = std::collections::BTreeSet::new();
        let mut changed = [false; 6];

        for parent in valid_cnns(40, 22) {
            for _ in 0..25 {
                let (child, how) = m.propose_named(&[&parent], &mut rng, &limits, &spec);
                let operator = how
                    .strip_prefix(&format!("{CNN_GENERATOR}/"))
                    .expect("the name of a CNN child");

                assert!(
                    child.validate(&limits, &spec).is_ok(),
                    "{}",
                    child.summary()
                );
                assert_ne!(child.seed, parent.seed);

                // Only the operator `training` keeps the architecture. It changes the training
                // genes, as the whole mutation of a CNN did before ticket 05.
                assert_eq!(
                    operator == "training",
                    architecture(&child) == architecture(&parent),
                    "{how}"
                );
                operators.insert(operator.to_string());

                if operator == "training" {
                    let genes = [
                        child.optimizer != parent.optimizer,
                        child.learning_rate != parent.learning_rate,
                        child.l2 != parent.l2,
                        child.batch_size != parent.batch_size,
                        child.epochs != parent.epochs,
                        child.init_scale != parent.init_scale,
                    ];

                    for (seen, now) in changed.iter_mut().zip(genes) {
                        *seen |= now;
                    }
                }

                // `mutate` makes the children of CNNs by the same operators.
                let other = mutate(&parent, 0.3, &mut rng, &limits, &spec);

                assert!(other.validate(&limits, &spec).is_ok());
            }
        }

        assert_eq!(
            operators.len(),
            cnn_variation::OPERATORS.len(),
            "{operators:?}"
        );
        assert_eq!(changed, [true; 6]);
    }

    #[test]
    fn crossover_with_a_cnn_takes_the_architecture_whole_or_exchanges_stages() {
        let spec = TaskKind::Shapes.spec();
        let limits = Limits::default();
        let m = RandomMutator::default();
        let mut rng = ChaCha8Rng::seed_from_u64(23);
        let mlp = sample();
        let cnns = valid_cnns(6, 24);
        let mut kinds = [0usize; 2];
        let mut exchanged = 0;

        assert!(mlp.validate(&limits, &spec).is_ok());

        for (i, cnn) in cnns.iter().enumerate() {
            let other = &cnns[(i + 1) % cnns.len()];

            for parents in [[&mlp, cnn], [cnn, &mlp], [cnn, other]] {
                for _ in 0..40 {
                    let (child, how) = m.propose_named(&parents, &mut rng, &limits, &spec);

                    assert!(
                        child.validate(&limits, &spec).is_ok(),
                        "{}",
                        child.summary()
                    );
                    kinds[usize::from(child.arch.is_mlp())] += 1;

                    // Two CNNs exchange whole stages: the child has the stem of one parent,
                    // every stage of one parent, and the architecture of neither.
                    if how == format!("{CNN_GENERATOR}/crossover") {
                        let (Arch::Cnn(c), [Arch::Cnn(a), Arch::Cnn(b)]) =
                            (&child.arch, [&parents[0].arch, &parents[1].arch])
                        else {
                            panic!("a stage crossover of two CNNs");
                        };

                        assert!(c.stem == a.stem || c.stem == b.stem);
                        assert!(
                            c.stages
                                .iter()
                                .all(|s| a.stages.contains(s) || b.stages.contains(s))
                        );
                        assert!(
                            parents
                                .iter()
                                .all(|p| architecture(p) != architecture(&child))
                        );
                        exchanged += 1;
                    } else {
                        assert_eq!(how, format!("{CNN_GENERATOR}/crossover-genes"));
                    }

                    let from = parents
                        .iter()
                        .find(|p| architecture(p) == architecture(&child))
                        .copied()
                        .unwrap_or(parents[0]);

                    // Training genes come from either parent; epochs may be cut to fit.
                    let either =
                        |f: &dyn Fn(&Genome) -> f32| parents.iter().any(|p| f(p) == f(&child));

                    assert!(either(&|g| g.learning_rate));
                    assert!(either(&|g| g.l2));
                    assert!(either(&|g| g.init_scale));
                    assert!(parents.iter().any(|p| p.optimizer == child.optimizer));
                    assert!(parents.iter().any(|p| p.batch_size == child.batch_size));
                    assert!(child.epochs <= parents.iter().map(|p| p.epochs).max().unwrap_or(0));
                    assert!(child.epochs >= MIN_EPOCHS && from.validate(&limits, &spec).is_ok());
                }
            }
        }

        // Both kinds of parents pass on their architecture, and two CNNs mostly exchange
        // stages.
        assert!(kinds[0] > 100 && kinds[1] > 50, "{kinds:?}");
        assert!(exchanged > 120, "{exchanged} of 240");
    }

    #[test]
    fn crossover_of_two_mlps_draws_as_before() {
        // The crossover of two MLPs before CNNs, gene by gene, as a reference.
        fn before(a: &Genome, b: &Genome, rng: &mut ChaCha8Rng) -> Genome {
            let mut pick = || rng.random_bool(0.5);
            let hidden = if pick() {
                a.hidden.clone()
            } else {
                b.hidden.clone()
            };
            let activation = if pick() { a.activation } else { b.activation };
            let optimizer = if pick() { a.optimizer } else { b.optimizer };
            let learning_rate = if pick() {
                a.learning_rate
            } else {
                b.learning_rate
            };
            let l2 = if pick() { a.l2 } else { b.l2 };
            let batch_size = if pick() { a.batch_size } else { b.batch_size };
            let epochs = if pick() { a.epochs } else { b.epochs };
            let init_scale = if pick() { a.init_scale } else { b.init_scale };

            Genome {
                arch: Arch::Mlp,
                hidden,
                activation,
                optimizer,
                learning_rate,
                l2,
                batch_size,
                epochs,
                init_scale,
                seed: rng.random(),
            }
        }

        let a = sample();
        let b = Genome {
            hidden: vec![4],
            activation: Activation::Sine,
            optimizer: Optimizer::Sgd,
            learning_rate: 0.001,
            l2: 0.0,
            batch_size: 16,
            epochs: 20,
            init_scale: 0.5,
            ..sample()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(25);
        let mut reference = ChaCha8Rng::seed_from_u64(25);

        for _ in 0..1000 {
            assert_eq!(crossover(&a, &b, &mut rng), before(&a, &b, &mut reference));
        }
    }

    #[test]
    fn children_of_parents_that_break_the_limits_are_redrawn() {
        let spec = TaskKind::Shapes.spec();
        let limits = Limits::default();
        let m = RandomMutator::default();
        let mut rng = ChaCha8Rng::seed_from_u64(26);
        let mut bad = testing::lenet();

        // A 9x9 kernel does not fit the 8x8 input of the LeNet stage.
        if let Arch::Cnn(cnn) = &mut bad.arch {
            cnn.stages[0].kernel = 9;
        }

        assert!(bad.validate(&limits, &spec).is_err());

        let good = testing::lenet();

        for parents in [
            vec![&bad],
            vec![&bad, &bad],
            vec![&bad, &good],
            vec![&good, &bad],
        ] {
            for _ in 0..20 {
                let child = m.propose(&parents, &mut rng, &limits, &spec);

                assert!(
                    child.validate(&limits, &spec).is_ok(),
                    "{}",
                    child.summary()
                );
                assert_ne!(architecture(&child), architecture(&bad));
            }
        }
    }

    #[test]
    fn repair_keeps_a_cnn_architecture_and_cuts_its_epochs() {
        let spec = TaskKind::Shapes.spec();
        let limits = Limits::default();
        let mut g = testing::lenet();

        g.epochs = 400;
        g.learning_rate = f32::NAN;
        g.batch_size = 1000;

        let fixed = repair(g.clone(), &limits, &spec);

        // 2e10 FLOPs / (3 * 405 832 FLOPs * 1000 samples) = 16.4 epochs, but the time
        // budget of 20 s allows 10.7: 1.87 ms per sample by the time model of `cnn::shape`.
        assert_eq!(fixed.epochs, 10);
        assert_eq!(fixed.learning_rate, 0.01);
        assert_eq!(fixed.batch_size, 256);
        assert_eq!(architecture(&fixed), architecture(&g));
        assert!(fixed.validate(&limits, &spec).is_ok());

        // An architecture that breaks a shape rule stays as it is; `propose` redraws it.
        if let Arch::Cnn(cnn) = &mut g.arch {
            cnn.stages.clear();
        }

        let unfit = repair(g.clone(), &limits, &spec);

        assert_eq!(architecture(&unfit), architecture(&g));
        assert!(unfit.validate(&limits, &spec).is_err());
    }

    #[test]
    fn propose_handles_all_parent_counts() {
        let spec = TaskKind::Parity.spec();
        let limits = Limits::default();
        let m = RandomMutator::default();
        let mut rng = ChaCha8Rng::seed_from_u64(8);
        let a = sample();
        let b = sample();

        for parents in [vec![], vec![&a], vec![&a, &b]] {
            let child = m.propose(&parents, &mut rng, &limits, &spec);

            assert!(child.validate(&limits, &spec).is_ok());
        }

        assert_eq!(m.name(), "random-mutator-v1");
    }
}
