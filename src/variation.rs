//! Variant generator: makes new genomes from parent genomes.
//!
//! The trait allows other generators later (for example an LLM-based one).
//! Every generator must return a genome that passes `Genome::validate`.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::config::Limits;
use crate::genome::{
    ACTIVATIONS, BATCH_SIZES, Genome, INIT_SCALE_MAX, INIT_SCALE_MIN, L2_MAX, L2_MIN, LR_MAX,
    LR_MIN, MIN_EPOCHS, OPTIMIZERS,
};
use crate::task::TaskSpec;
use crate::util::{log_uniform_f32, normal_f32};

/// Number of mutable genes (the seed is always renewed and not counted).
const GENE_COUNT: usize = 8;

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
}

/// Classic random mutation and uniform crossover, with repair to valid ranges.
#[derive(Clone, Copy, Debug)]
pub struct RandomMutator {
    /// Probability that each gene is mutated.
    pub gene_rate: f64,
}

impl Default for RandomMutator {
    fn default() -> Self {
        RandomMutator { gene_rate: 0.3 }
    }
}

impl VariantGenerator for RandomMutator {
    fn name(&self) -> &str {
        "random-mutator-v1"
    }

    fn propose(
        &self,
        parents: &[&Genome],
        rng: &mut ChaCha8Rng,
        limits: &Limits,
        spec: &TaskSpec,
    ) -> Genome {
        match parents {
            [] => Genome::random(rng, limits, spec),
            [one] => mutate(one, self.gene_rate, rng, limits, spec),
            [a, b, ..] => {
                let child = crossover(a, b, rng);

                repair(child, limits, spec)
            }
        }
    }
}

/// Mutate one genome. Always gets a fresh training seed.
pub fn mutate(
    parent: &Genome,
    gene_rate: f64,
    rng: &mut ChaCha8Rng,
    limits: &Limits,
    spec: &TaskSpec,
) -> Genome {
    let rate = if gene_rate.is_finite() {
        gene_rate.clamp(0.0, 1.0)
    } else {
        0.0
    };
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
pub fn crossover(a: &Genome, b: &Genome, rng: &mut ChaCha8Rng) -> Genome {
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
    let seed = rng.random();

    Genome {
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
pub fn repair(g: Genome, limits: &Limits, spec: &TaskSpec) -> Genome {
    let mut g = g;

    g.learning_rate = clamp_or(g.learning_rate, LR_MIN, LR_MAX, 0.01);
    g.l2 = clamp_or(g.l2, L2_MIN, L2_MAX, 0.0);
    g.init_scale = clamp_or(g.init_scale, INIT_SCALE_MIN, INIT_SCALE_MAX, 1.0);
    g.batch_size = nearest_batch(g.batch_size);

    let max_epochs = limits.max_epochs.max(MIN_EPOCHS);

    g.epochs = g.epochs.clamp(MIN_EPOCHS, max_epochs);
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
    use crate::genome::{Activation, Optimizer};
    use crate::task::TaskKind;
    use rand::SeedableRng;

    fn sample() -> Genome {
        Genome {
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
