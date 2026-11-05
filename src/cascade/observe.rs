//! Universe observables (step 1, ticket 01): what one committed generation of a Run looks
//! like, logged as one `state` event.
//!
//! Six groups: performance, diversity, the lineage tree, honesty, judging and cost. Each is
//! a pure function of committed state: the archive, all comparisons, the weights the next
//! generation draws its parents by, and this generation's usage. No cumulative call count
//! enters, since that also counts the calls a stopped or killed attempt reserved. So a Run
//! that is stopped and resumed logs the same states as a straight one. Wall time belongs to
//! the binary.
//!
//! Every number is finite, since a NaN or an infinity would reach the log as null and break
//! it. A value that is undefined because its set is empty is None, or 0 where the field is
//! a count, a share or a best value. Sums run in a fixed order, so the bytes do not depend
//! on thread count.
//!
//! A Run with Souls (step 2, `.scratch/seele-und-kappa/spec.md`, section 13.1) adds a
//! seventh group, `soul`: the Soul population, its Character on the search panel, how alike
//! the Souls are, what experience did to them, and this generation's Observations and
//! votes. It is left out of the JSON in a Run without Souls, so those state lines stay as
//! they were. It too is a pure function of committed state, with this commit's Observations
//! as the generation's usage is for the cost group, and in arm W the choices that the
//! preferences decided in this generation's children. A statistic of the Character is computed
//! once per distinct strength vector of the search panel and weighted by the number of
//! tasks that have it, so every task weighs alike.
//!
//! A Run with imagination or changing goals (`.scratch/seelen-modell/spec-imagination.md`,
//! section 3.5) adds the group `mind`: the goal of the generation and, with imagination, how
//! imagination chose this generation's children and the learned values of their parents.
//! It is left out of the JSON otherwise.
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::cascade::arena::{Comparison, Member, SoulSource, Usage, VoteTally, WorkerRef};
use crate::cascade::genome::{COUPLING_MAX, Cascade, PhaseState};
use crate::cascade::soul::{
    Phasor, Readout, character_norm, independent_contrast, phase_gap, readout,
};
use crate::cascade::variation::{Op, Steered};
use crate::util::sha256_hex;

/// The observables of one committed generation: the payload of a `state` event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UniverseState {
    pub generation: usize,
    pub performance: Performance,
    pub diversity: Diversity,
    pub lineage: Lineage,
    pub honesty: Honesty,
    pub judging: Judging,
    pub cost: Costs,
    /// The Souls (step 2); left out of the JSON in a Run without Souls. Boxed, so that a
    /// state event stays near the size of the other events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soul: Option<Box<SoulState>>,
    /// The mind of a Run with imagination or changing goals; left out of the JSON otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mind: Option<MindState>,
}

/// How well the archive does on the search panel. A solved rate is solved / tasks, a
/// quality rate quality / tasks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Performance {
    pub archive_size: usize,
    /// Members born in this generation.
    pub born: usize,
    /// Tasks on the search panel.
    pub tasks: usize,
    /// Best and lower-median solved rate over the archive (0 when it is empty).
    pub solved_best: f64,
    pub solved_median: f64,
    /// Best solved rate of the members born in this generation (0 when none was born).
    pub solved_best_born: f64,
    /// Best and lower-median fitness over the archive (0 when it is empty).
    pub fitness_best: f64,
    pub fitness_median: f64,
    /// Best and lower-median quality rate; None unless the domain is graded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_best: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_median: Option<f64>,
}

/// How varied the archive is. A shape is the Stage widths of a Cascade, such as "1-2-1". A
/// block is a (section name, text) pair, counted once for every Worker that carries it.
/// Each entropy is the Shannon entropy in bits of the frequencies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Diversity {
    /// Distinct shapes, and the entropy of their frequencies over the archive.
    pub shapes: usize,
    pub shape_entropy_bits: f64,
    /// Distinct blocks, and the entropy of their frequencies.
    pub blocks: usize,
    pub block_entropy_bits: f64,
    /// Entropy of the model tags over all Workers.
    pub model_entropy_bits: f64,
    /// Distinct answer lists on the search panel.
    pub behaviours: usize,
    /// Entropy of the next generation's parent draw. Two to this power is the effective
    /// number of parents.
    pub parent_entropy_bits: f64,
}

/// The family tree of the archive. A seed has depth 0, and a child one more than the
/// deepest of its parents. Branching counts the children of members that have any.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lineage {
    pub depth_max: usize,
    pub depth_mean: f64,
    pub branching_mean: f64,
    pub branching_max: usize,
    /// Share of members without children.
    pub leaf_share: f64,
    /// The seed Cascades.
    pub seed_lineages: usize,
    /// Seeds that are, or are ancestors of, a member born in this generation.
    pub seed_lineages_alive: usize,
    /// Living seed lineages as a share of all (0 without seeds).
    pub surviving_share: f64,
}

/// What overseers caught (ticket 05). Until overseers exist, every field is false or 0.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Honesty {
    pub overseers: bool,
    pub fakes_planted: u64,
    pub fakes_caught: u64,
    pub false_alarms: u64,
    pub confirmed_collusion: u64,
}

/// Five quantiles of a set: sorted[(n - 1) k / 4] for k = 0 to 4. The median is the
/// lower median.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quantiles {
    pub min: f64,
    pub p25: f64,
    pub median: f64,
    pub p75: f64,
    pub max: f64,
}

/// How the Cascades judge. The new comparisons are those of this generation. A scored
/// comparison is one whose better answer is known.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Judging {
    pub comparisons_new: usize,
    pub scored_new: usize,
    /// Share of the scored new comparisons that the judge got right; None when none was
    /// scored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accuracy_new: Option<f64>,
    /// Members with at least one scored judgment, and the quantiles of their judging
    /// skill (None when there is none).
    pub judges_scored: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skill: Option<Quantiles>,
    /// Among those members: the share trusted (skill above 1/2), and the mean vote weight
    /// max(0, ln(u / (1 - u))) of skill u. Both are 0 when there is no such member.
    pub trusted_share: f64,
    pub vote_weight_mean: f64,
}

/// What this generation cost, and what the archive's Cascades cost per task. The usage
/// is that of this generation's newborns on the search panel plus its peer round.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Costs {
    pub calls: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub model_cost: f64,
    /// Means over the archive (0 when it is empty): the thinking budget per task, summed
    /// over a Cascade's Workers, and the Workers and Stages of a Cascade.
    pub think_tokens_mean: f64,
    pub workers_mean: f64,
    pub stages_mean: f64,
}

/// The Souls of the archive after a commit (step 2, spec 13.1). Means over tasks run over
/// the search panel, each task once; a value without its set is absent, and a count of an
/// empty set is 0 (spec 13.4).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoulState {
    /// Workers in the archive: the Soul population (Q58). It never falls.
    pub souls: usize,
    /// Workers born in this generation, and those of them with an entirely new Soul (Q64).
    pub souls_born: usize,
    pub souls_new: usize,
    /// Distinct origins of the archive's Souls, Hoppe's colours, and the entropy of their
    /// frequencies over the Workers.
    pub origins: usize,
    pub origin_entropy_bits: f64,
    /// The mean normalised Character over the archive's Workers and the search tasks, from
    /// the current Souls (Q51).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub character_norm_mean: Option<f64>,
    /// The mean over the search tasks of the Character's contrast over the archive's
    /// Workers, its standard deviation divided by its mean, from the current Souls (Q57). A
    /// task on which every Character is 0 has no spread and counts 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contrast: Option<f64>,
    /// The same mean of the formula of spec 6.3 for this readout: the contrast of
    /// independent, uniform Souls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contrast_independent: Option<f64>,
    /// Per family, the length of the mean readout phasor of the current Souls over all
    /// Workers (1: all alike; near 0: spread out), and the mean over the families.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order_mean: Option<f64>,
    /// The mean circular distance between current and birth Soul over the Workers and their
    /// families, in degrees: what experience did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moved_mean: Option<f64>,
    /// The quantiles of κ over the archive's Workers, as fractions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kappa: Option<Quantiles>,
    /// The Observations applied at this commit, of Stage votes and of the peer round (Q62).
    pub observations_vote: u64,
    pub observations_peer: u64,
    /// The mean phase change of the Workers whose current Soul changed at this commit, over
    /// them and their families, in degrees (Q56).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shift_mean: Option<f64>,
    /// The complex Stage votes of this generation's newborns on the search panel; those
    /// whose winner differs from the count's; those in which two or more answers shared the
    /// largest Π, so the tie rule decided.
    pub votes: u64,
    pub votes_flipped: u64,
    pub votes_tied: u64,
    /// The mean coherence of these votes (spec 7.7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coherence_mean: Option<f64>,
    /// Replies to this generation's newborns on the search panel in which the phases
    /// decided a simulated model's choice (spec 11).
    pub sim_decided: u64,
    /// With `variation` on (arm W, `.scratch/seelen-modell/spec.md`, section 4.1): the
    /// choices that the preferences decided in the operators that made this generation's
    /// newborns, among options of unequal weight, and those of them whose pick differs from
    /// the uniform pick of the same draw. Both are left out when `variation` is off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variation_choices: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variation_changed: Option<u64>,
}

/// The mind of a Run after a commit (`.scratch/seelen-modell/spec-imagination.md`, section
/// 3.5), with imagination or changing goals on. Learned values are in thousandths, as the
/// Born events log them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MindState {
    /// The goal of this generation: "all", or "-" and the family whose targets it leaves out,
    /// such as "-Linear". Without changing goals it is "all".
    pub goal: String,
    /// With imagination on: this generation's children that imagination chose with some
    /// candidate draw of a prediction other than 0, the imagination choices; and those whose
    /// chosen draw is not the first, the changed picks. Left out without imagination.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imagination_choices: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imagination_changed: Option<u64>,
    /// With imagination on: the mean learned value of Linear, Phase and Hadamard, and that of
    /// the other families, over the Workers of this generation's parents, read when the
    /// parents were chosen, before this commit's learning. A parent counts once per child,
    /// and both parents of a crossover count. Left out without a value to average, as in
    /// generation 0, and without imagination.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_phase_hadamard_mean: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub other_families_mean: Option<f64>,
}

/// A sum of learned values and their count, for a mean.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ValueSums {
    pub sum: i64,
    pub count: u64,
}

impl ValueSums {
    /// Count the value `v`.
    pub fn add(&mut self, v: i16) {
        self.sum += i64::from(v);
        self.count += 1;
    }

    /// The mean, or None without a value. The sum is exact in integers, so the one division
    /// gives the same float on every platform.
    pub fn mean(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum as f64 / self.count as f64)
    }
}

/// What the mind group reads about imagination in this generation's children (spec-imagination
/// 3.5): the imagination choices, the changed picks, and the sums of their parents' learned
/// values, for Linear, Phase and Hadamard and for the other families.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImaginationCounts {
    pub choices: u64,
    pub changed: u64,
    pub linear_phase_hadamard: ValueSums,
    pub other_families: ValueSums,
}

/// What the mind group reads (spec-imagination 3.5).
#[derive(Clone, Copy, Debug)]
pub struct MindInput<'a> {
    /// The name of this generation's goal.
    pub goal: &'a str,
    /// With imagination on, how it chose this generation's children; None without it.
    pub imagination: Option<ImaginationCounts>,
}

/// What the soul group reads besides the archive (step 2, spec 13.1).
#[derive(Clone, Copy, Debug)]
pub struct SoulInput<'a> {
    /// How the Run reads its Souls.
    pub readout: Readout,
    /// The phases of a Soul: the Domain's families.
    pub families: usize,
    /// The Family strengths of the search panel's tasks under the Run's setting: each
    /// distinct vector once, in the order of its first task, with the number of tasks that
    /// have it (`strength_counts`).
    pub strengths: &'a [(Vec<f64>, usize)],
    /// The Observations applied at this commit, of Stage votes and of the peer round.
    pub observations_vote: u64,
    pub observations_peer: u64,
    /// Each Worker whose current Soul changed at this commit: the Soul it had when the
    /// commit began, and the Soul it has after it.
    pub moved: &'a [(PhaseState, PhaseState)],
    /// With `variation` on, the choices the preferences decided in the operators that made
    /// this generation's newborns; None with `variation` off.
    pub variation: Option<Steered>,
}

/// The committed state that one generation's observables are computed from.
pub struct ObserveInput<'a, A> {
    /// The generation just committed.
    pub generation: usize,
    /// The archive after the commit. The members from `first_new` on were born in this
    /// generation.
    pub archive: &'a [Member<A>],
    pub first_new: usize,
    /// Every comparison so far.
    pub comparisons: &'a [Comparison],
    /// The weights the next generation draws its parents by, one per member.
    pub selection_weights: &'a [f64],
    /// This generation's usage: its newborns on the search panel plus its peer round.
    pub generation_usage: Usage,
    /// Tasks on the search panel.
    pub tasks: usize,
    /// In a Run with Souls, what the soul group reads besides the archive (step 2).
    pub soul: Option<SoulInput<'a>>,
    /// In a Run with imagination or changing goals, what the mind group reads.
    pub mind: Option<MindInput<'a>>,
}

/// The observables of one committed generation.
pub fn observe<A: Serialize>(input: &ObserveInput<'_, A>) -> UniverseState {
    UniverseState {
        generation: input.generation,
        performance: performance(input),
        diversity: diversity(input),
        lineage: lineage(input.archive, input.first_new),
        honesty: Honesty::default(),
        judging: judging(input.archive, input.comparisons, input.generation),
        cost: costs(input.archive, &input.generation_usage),
        soul: input
            .soul
            .as_ref()
            .map(|s| Box::new(soul_state(input.archive, input.first_new, s))),
        mind: input.mind.as_ref().map(mind_state),
    }
}

/// The mind group of a commit (spec-imagination 3.5).
fn mind_state(m: &MindInput<'_>) -> MindState {
    let counts = m.imagination;

    MindState {
        goal: m.goal.to_string(),
        imagination_choices: counts.map(|c| c.choices),
        imagination_changed: counts.map(|c| c.changed),
        linear_phase_hadamard_mean: counts.and_then(|c| c.linear_phase_hadamard.mean()),
        other_families_mean: counts.and_then(|c| c.other_families.mean()),
    }
}

/// The distinct strength vectors of the tasks whose strengths are `tasks`, each once in the
/// order of its first task, with the number of tasks that have it (spec 13.1).
pub fn strength_counts(tasks: impl IntoIterator<Item = Vec<f64>>) -> Vec<(Vec<f64>, usize)> {
    let mut counts: Vec<(Vec<f64>, usize)> = Vec::new();

    for s in tasks {
        match counts.iter_mut().find(|(v, _)| *v == s) {
            Some((_, n)) => *n += 1,
            None => counts.push((s, 1)),
        }
    }

    counts
}

/// Degrees per phase unit: a phase unit is 1/65536 of a turn. Exact in binary.
const DEGREES_PER_UNIT: f64 = 360.0 / 65536.0;

/// The circular distance of two Souls, in degrees, summed over their families, and the
/// number of families compared.
fn circular_distance(a: &[u16], b: &[u16]) -> (f64, usize) {
    a.iter().zip(b).fold((0.0, 0), |(sum, n), (&p, &q)| {
        (
            sum + f64::from(phase_gap(p, q).unsigned_abs()) * DEGREES_PER_UNIT,
            n + 1,
        )
    })
}

/// The mean and the contrast (standard deviation over the mean) of the normalised
/// Characters of `souls` on a task with the strengths `s`, or None without a Soul. The
/// contrast is 0 when every Character is 0: there is no spread.
fn character_statistics(souls: &[&[u16]], s: &[f64], r: Readout) -> Option<(f64, f64)> {
    if souls.is_empty() {
        return None;
    }

    let n = souls.len() as f64;
    let values: Vec<f64> = souls.iter().map(|p| character_norm(p, s, r)).collect();
    let mean = values.iter().fold(0.0, |sum, &x| sum + x) / n;
    let variance = values
        .iter()
        .fold(0.0, |sum, &x| sum + (x - mean) * (x - mean))
        / n;
    let contrast = if mean > 0.0 {
        variance.sqrt() / mean
    } else {
        0.0
    };

    Some((mean, contrast))
}

/// The soul group of the archive after a commit (spec 13.1). The members from `first_new`
/// on were born in this generation. Character statistics read the current Souls, once per
/// distinct strength vector of `s.strengths`, weighted by its number of tasks.
fn soul_state<A>(archive: &[Member<A>], first_new: usize, s: &SoulInput<'_>) -> SoulState {
    let born = &archive[first_new.min(archive.len())..];
    let workers = |m: &Member<A>| m.cascade.worker_count();
    fn records<A>(members: &[Member<A>]) -> Vec<&SoulSource> {
        members
            .iter()
            .flat_map(|m| m.soul_sources.iter().flatten())
            .collect()
    }

    let mut origins: BTreeMap<&WorkerRef, usize> = BTreeMap::new();

    for r in records(archive) {
        *origins.entry(&r.origin).or_default() += 1;
    }

    // The current Soul of every Worker that has one, in archive and Stage order.
    let now: Vec<&[u16]> = archive
        .iter()
        .flat_map(|m| m.soul_now.iter().flatten().map(|p| p.0.as_slice()))
        .collect();
    // How far each current Soul lies from its birth Soul.
    let mut moved = (0.0, 0);

    for m in archive {
        let births = m.cascade.stages.iter().flat_map(|st| &st.workers);

        for (w, current) in births.zip(m.soul_now.iter().flatten()) {
            if let Some(birth) = &w.traits.phase_state {
                let (sum, n) = circular_distance(birth, current);

                moved = (moved.0 + sum, moved.1 + n);
            }
        }
    }

    // The Character, once per distinct strength vector, weighted by its tasks.
    let tasks: usize = s.strengths.iter().map(|(_, n)| n).sum();
    let mut sums = (0.0, 0.0, 0.0);
    let mut defined = true;

    for (strengths, count) in s.strengths {
        let weight = *count as f64;

        match character_statistics(&now, strengths, s.readout) {
            Some((mean, contrast)) => {
                sums.0 += weight * mean;
                sums.1 += weight * contrast;
            }
            None => defined = false,
        }

        sums.2 += weight * independent_contrast(strengths, s.readout);
    }

    let per_task = |sum: f64| (tasks > 0).then(|| sum / tasks as f64);
    let character = |sum: f64| per_task(sum).filter(|_| defined && !now.is_empty());

    // How alike the Souls are, family by family: the length of the mean readout phasor.
    let order: Option<Vec<f64>> = (!now.is_empty()).then(|| {
        (0..s.families)
            .map(|f| {
                let (sum, n) = now
                    .iter()
                    .filter_map(|p| p.get(f))
                    .fold((Phasor::default(), 0usize), |(z, n), &p| {
                        (z + readout(p, s.readout), n + 1)
                    });

                if n > 0 { sum.norm() / n as f64 } else { 0.0 }
            })
            .collect()
    });
    let order_mean = order
        .as_ref()
        .filter(|o| !o.is_empty())
        .map(|o| o.iter().fold(0.0, |sum, &x| sum + x) / o.len() as f64);

    // What this commit's Observations moved.
    let shift = s.moved.iter().fold((0.0, 0), |(sum, n), (before, after)| {
        let (d, k) = circular_distance(before, after);

        (sum + d, n + k)
    });

    let kappas: Vec<f64> = archive
        .iter()
        .flat_map(|m| m.cascade.stages.iter().flat_map(|st| &st.workers))
        .map(|w| f64::from(w.traits.coupling) / f64::from(COUPLING_MAX))
        .collect();
    let mut votes = VoteTally::default();

    for m in born {
        if let Some(t) = &m.votes {
            votes.complex += t.complex;
            votes.flipped += t.flipped;
            votes.tied += t.tied;
            votes.coherence_sum += t.coherence_sum;
            votes.sim_decided += t.sim_decided;
        }
    }

    let mean_of = |(sum, n): (f64, usize)| (n > 0).then(|| sum / n as f64);

    SoulState {
        souls: archive.iter().map(workers).sum(),
        souls_born: born.iter().map(workers).sum(),
        souls_new: records(born).iter().filter(|r| r.new).count(),
        origins: origins.len(),
        origin_entropy_bits: entropy_bits(&origins),
        character_norm_mean: character(sums.0),
        contrast: character(sums.1),
        contrast_independent: per_task(sums.2),
        order,
        order_mean,
        moved_mean: mean_of(moved),
        kappa: quantiles(&kappas),
        observations_vote: s.observations_vote,
        observations_peer: s.observations_peer,
        shift_mean: mean_of(shift),
        votes: votes.complex,
        votes_flipped: votes.flipped,
        votes_tied: votes.tied,
        coherence_mean: (votes.complex > 0).then(|| votes.coherence_sum / votes.complex as f64),
        sim_decided: votes.sim_decided,
        variation_choices: s.variation.map(|v| v.choices),
        variation_changed: s.variation.map(|v| v.changed),
    }
}

/// Shannon entropy in bits of the frequencies `counts`: the sum of (c/N) log2(N/c), in key
/// order. It is 0 when there is no count. Every term is at least +0, so the result is
/// never -0.
pub fn entropy_bits<K: Ord>(counts: &BTreeMap<K, usize>) -> f64 {
    let parts: Vec<f64> = counts.values().map(|&c| c as f64).collect();

    entropy(&parts)
}

/// Entropy in bits of the distribution proportional to the non-negative `parts`. It is 0
/// when they do not sum to a positive finite number.
///
/// Each term is computed from the share q = p / total as -q log2(q). A share lies in
/// (0, 1], so a term stays finite even for a part far below the smallest normal number,
/// where total / p would overflow.
fn entropy(parts: &[f64]) -> f64 {
    let total = parts.iter().fold(0.0, |sum, &p| sum + p);

    if !(total > 0.0 && total.is_finite()) {
        return 0.0;
    }

    parts
        .iter()
        .map(|&p| p / total)
        .filter(|&q| q > 0.0)
        .fold(0.0, |sum, q| sum - q * q.log2())
}

/// Entropy in bits of a parent draw by `weights`. Weights that do not sum to a positive
/// finite number give a uniform draw, as the breeder then draws uniformly.
fn parent_entropy(weights: &[f64]) -> f64 {
    let total = weights.iter().fold(0.0, |sum, &w| sum + w);

    if total > 0.0 && total.is_finite() {
        entropy(weights)
    } else if weights.is_empty() {
        0.0
    } else {
        (weights.len() as f64).log2()
    }
}

/// The quantiles of `values` under `total_cmp`, or None when there is no value.
fn quantiles(values: &[f64]) -> Option<Quantiles> {
    let mut sorted = values.to_vec();

    sorted.sort_by(f64::total_cmp);

    let last = sorted.len().checked_sub(1)?;
    let at = |k: usize| sorted[last * k / 4];

    Some(Quantiles {
        min: at(0),
        p25: at(1),
        median: at(2),
        p75: at(3),
        max: at(4),
    })
}

/// `total / n`, or 0 when `n` is 0.
fn ratio(total: f64, n: usize) -> f64 {
    if n == 0 { 0.0 } else { total / n as f64 }
}

/// The Stage widths of `c`, joined by "-".
fn shape(c: &Cascade) -> String {
    let widths: Vec<String> = c
        .stages
        .iter()
        .map(|s| s.workers.len().to_string())
        .collect();

    widths.join("-")
}

/// The thinking budget of `c` per task: the sum over its Workers.
fn thinking(c: &Cascade) -> u64 {
    c.stages
        .iter()
        .flat_map(|s| &s.workers)
        .map(|w| u64::from(w.traits.think_tokens))
        .sum()
}

fn performance<A>(input: &ObserveInput<'_, A>) -> Performance {
    let archive = input.archive;
    let born = &archive[input.first_new.min(archive.len())..];
    let n = input.tasks.max(1) as f64;
    let solved = |members: &[Member<A>]| -> Vec<f64> {
        members.iter().map(|m| m.solved as f64 / n).collect()
    };
    let fitness: Vec<f64> = archive.iter().map(|m| m.fitness).collect();
    let quality: Vec<f64> = archive
        .iter()
        .filter_map(|m| m.quality)
        .map(|q| q / n)
        .collect();

    let all = quantiles(&solved(archive));
    let fitness = quantiles(&fitness);
    let quality = quantiles(&quality);

    Performance {
        archive_size: archive.len(),
        born: born.len(),
        tasks: input.tasks,
        solved_best: all.map_or(0.0, |q| q.max),
        solved_median: all.map_or(0.0, |q| q.median),
        solved_best_born: quantiles(&solved(born)).map_or(0.0, |q| q.max),
        fitness_best: fitness.map_or(0.0, |q| q.max),
        fitness_median: fitness.map_or(0.0, |q| q.median),
        quality_best: quality.map(|q| q.max),
        quality_median: quality.map(|q| q.median),
    }
}

fn diversity<A: Serialize>(input: &ObserveInput<'_, A>) -> Diversity {
    let mut shapes: BTreeMap<String, usize> = BTreeMap::new();
    let mut blocks: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    let mut models: BTreeMap<&str, usize> = BTreeMap::new();
    let mut behaviours: BTreeSet<String> = BTreeSet::new();

    for m in input.archive {
        *shapes.entry(shape(&m.cascade)).or_default() += 1;

        for w in m.cascade.stages.iter().flat_map(|s| &s.workers) {
            // Each Worker counts a block once. A Cascade from the Mutator may hold the
            // same section twice in one Worker.
            let carried: BTreeSet<(&str, &str)> = w
                .sections
                .iter()
                .map(|s| (s.name.as_str(), s.text.as_str()))
                .collect();

            *models.entry(w.model.as_str()).or_default() += 1;

            for block in carried {
                *blocks.entry(block).or_default() += 1;
            }
        }

        let answers = serde_json::to_string(&m.answers).unwrap_or_default();

        behaviours.insert(sha256_hex(answers.as_bytes()));
    }

    Diversity {
        shapes: shapes.len(),
        shape_entropy_bits: entropy_bits(&shapes),
        blocks: blocks.len(),
        block_entropy_bits: entropy_bits(&blocks),
        model_entropy_bits: entropy_bits(&models),
        behaviours: behaviours.len(),
        parent_entropy_bits: parent_entropy(input.selection_weights),
    }
}

fn lineage<A>(archive: &[Member<A>], first_new: usize) -> Lineage {
    let index: BTreeMap<&str, usize> = archive
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id.as_str(), i))
        .collect();
    let mut depth: Vec<usize> = Vec::with_capacity(archive.len());
    let mut children = vec![0usize; archive.len()];
    // The seeds each member descends from; a seed descends from itself.
    let mut roots: Vec<BTreeSet<usize>> = Vec::with_capacity(archive.len());

    for (i, m) in archive.iter().enumerate() {
        // Parents are archived before their children; any other link is ignored.
        let parents: Vec<usize> = m
            .parents
            .iter()
            .filter_map(|p| index.get(p.as_str()).copied())
            .filter(|&j| j < i)
            .collect();
        let mut from: BTreeSet<usize> = parents
            .iter()
            .flat_map(|&j| roots[j].iter().copied())
            .collect();

        if m.op == Op::Seed {
            from.insert(i);
        }

        for &j in &parents {
            children[j] += 1;
        }

        depth.push(parents.iter().map(|&j| depth[j] + 1).max().unwrap_or(0));
        roots.push(from);
    }

    let branching: Vec<usize> = children.iter().copied().filter(|&c| c > 0).collect();
    let leaves = children.iter().filter(|&&c| c == 0).count();
    let seeds = archive.iter().filter(|m| m.op == Op::Seed).count();
    let alive: BTreeSet<usize> = roots[first_new.min(roots.len())..]
        .iter()
        .flatten()
        .copied()
        .collect();

    Lineage {
        depth_max: depth.iter().copied().max().unwrap_or(0),
        depth_mean: ratio(depth.iter().sum::<usize>() as f64, depth.len()),
        branching_mean: ratio(branching.iter().sum::<usize>() as f64, branching.len()),
        branching_max: branching.iter().copied().max().unwrap_or(0),
        leaf_share: ratio(leaves as f64, archive.len()),
        seed_lineages: seeds,
        seed_lineages_alive: alive.len(),
        surviving_share: ratio(alive.len() as f64, seeds),
    }
}

fn judging<A>(archive: &[Member<A>], comparisons: &[Comparison], generation: usize) -> Judging {
    let new: Vec<&Comparison> = comparisons
        .iter()
        .filter(|c| c.generation == generation)
        .collect();
    let scored: Vec<bool> = new.iter().filter_map(|c| c.judge_correct).collect();
    let right = scored.iter().filter(|&&ok| ok).count();
    let skills: Vec<f64> = archive
        .iter()
        .filter(|m| m.judge_correct > 0 || m.judge_wrong > 0)
        .map(Member::judge_skill)
        .collect();
    let trusted = skills.iter().filter(|&&u| u > 0.5).count();
    let weight = skills
        .iter()
        .fold(0.0, |sum, &u| sum + (u / (1.0 - u)).ln().max(0.0));

    Judging {
        comparisons_new: new.len(),
        scored_new: scored.len(),
        accuracy_new: (!scored.is_empty()).then(|| right as f64 / scored.len() as f64),
        judges_scored: skills.len(),
        skill: quantiles(&skills),
        trusted_share: ratio(trusted as f64, skills.len()),
        vote_weight_mean: ratio(weight, skills.len()),
    }
}

fn costs<A>(archive: &[Member<A>], usage: &Usage) -> Costs {
    let think: u64 = archive.iter().map(|m| thinking(&m.cascade)).sum();
    let workers: usize = archive.iter().map(|m| m.cascade.worker_count()).sum();
    let stages: usize = archive.iter().map(|m| m.cascade.stages.len()).sum();

    Costs {
        calls: usage.calls,
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        model_cost: usage.model_cost,
        think_tokens_mean: ratio(think as f64, archive.len()),
        workers_mean: ratio(workers as f64, archive.len()),
        stages_mean: ratio(stages as f64, archive.len()),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;
    use crate::cascade::arena::{BreedConfig, Event, dgm_weights};
    use crate::cascade::circuit::CircuitDomain;
    use crate::cascade::domain::{CircuitConfig, Domain};
    use crate::cascade::genome::{Section, seed_cascades};
    use crate::util::derive_seed;

    /// A member on the first seed's Cascade with one answer, its solved count, and
    /// parents by id; a member without parents is a seed.
    fn member(id: &str, parents: &[&str], solved: usize) -> Member<u32> {
        Member {
            id: id.to_string(),
            cascade: seed_cascades()[0].1.clone(),
            parents: parents.iter().map(|p| p.to_string()).collect(),
            op: if parents.is_empty() {
                Op::Seed
            } else {
                Op::SectionAdd
            },
            generation: 0,
            answers: vec![Some(solved as u32)],
            solved,
            quality: None,
            usage: Usage::default(),
            children: 0,
            judge_correct: 0,
            judge_wrong: 0,
            fitness: 0.0,
            rating: 0.0,
            soul_sources: None,
            imprinted: None,
            soul_now: None,
            teaching: None,
            authors: None,
            votes: None,
            values: None,
            values_now: None,
            effect: None,
        }
    }

    /// Seeds s1, s2 and s3; a is a child of s1, b of s1 and s2; c (child of a) and d
    /// (child of b) are born in the generation observed.
    fn family() -> Vec<Member<u32>> {
        vec![
            member("s1", &[], 0),
            member("s2", &[], 1),
            member("s3", &[], 2),
            member("a", &["s1"], 3),
            member("b", &["s1", "s2"], 4),
            member("c", &["a"], 1),
            member("d", &["b"], 2),
        ]
    }

    /// Generation 2 of `family()` on 4 tasks, with c and d new.
    fn input<'a>(
        archive: &'a [Member<u32>],
        comparisons: &'a [Comparison],
        weights: &'a [f64],
    ) -> ObserveInput<'a, u32> {
        ObserveInput {
            generation: 2,
            archive,
            first_new: 5,
            comparisons,
            selection_weights: weights,
            generation_usage: Usage::default(),
            tasks: 4,
            soul: None,
            mind: None,
        }
    }

    /// A comparison of c and d in `generation`, judged by s1.
    fn comparison(generation: usize, judge_correct: Option<bool>) -> Comparison {
        Comparison {
            generation,
            puzzle: "task".to_string(),
            first: "c".to_string(),
            second: "d".to_string(),
            judge: "s1".to_string(),
            winner: Some("c".to_string()),
            judge_correct,
        }
    }

    /// The entropy of `parts` by the definition: the sum of p log2(1 / p).
    fn by_definition(parts: &[f64]) -> f64 {
        let n: f64 = parts.iter().sum();

        parts.iter().map(|&x| x / n * (n / x).log2()).sum()
    }

    /// Every number in `v`.
    fn numbers(v: &Value, out: &mut Vec<f64>) {
        match v {
            Value::Number(n) => out.extend(n.as_f64()),
            Value::Array(items) => items.iter().for_each(|x| numbers(x, out)),
            Value::Object(fields) => fields.values().for_each(|x| numbers(x, out)),
            _ => {}
        }
    }

    #[test]
    fn entropy_is_log2_of_equal_counts_and_never_negative_zero() {
        for k in 1..=9usize {
            let counts: BTreeMap<usize, usize> = (0..k).map(|i| (i, 3)).collect();

            assert!(
                (entropy_bits(&counts) - (k as f64).log2()).abs() < 1e-12,
                "{k} categories"
            );
        }

        let single = BTreeMap::from([("only", 7usize)]);
        let empty: BTreeMap<&str, usize> = BTreeMap::new();
        let with_zero = BTreeMap::from([("a", 2usize), ("b", 0), ("c", 2)]);
        let skewed = BTreeMap::from([("a", 1usize), ("b", 3)]);

        assert_eq!(
            entropy_bits(&single).to_bits(),
            0.0f64.to_bits(),
            "+0, not -0"
        );
        assert_eq!(entropy_bits(&empty).to_bits(), 0.0f64.to_bits());
        assert!(
            (entropy_bits(&with_zero) - 1.0).abs() < 1e-12,
            "a zero count adds nothing"
        );
        assert!((entropy_bits(&skewed) - by_definition(&[1.0, 3.0])).abs() < 1e-12);
    }

    #[test]
    fn parent_entropy_follows_the_draw() {
        assert!(
            (parent_entropy(&[1.0; 4]) - 2.0).abs() < 1e-12,
            "uniform over 4"
        );
        assert!((parent_entropy(&[1.0, 1.0, 2.0]) - 1.5).abs() < 1e-12);
        assert_eq!(
            parent_entropy(&[0.0, 2.5, 0.0]).to_bits(),
            0.0f64.to_bits(),
            "one parent"
        );

        // Weights without a positive finite sum make the draw uniform.
        assert!((parent_entropy(&[0.0; 8]) - 3.0).abs() < 1e-12);
        assert!((parent_entropy(&[f64::NAN, 1.0]) - 1.0).abs() < 1e-12);
        assert_eq!(parent_entropy(&[]).to_bits(), 0.0f64.to_bits());

        // A weight below the smallest normal number adds next to nothing; its term must
        // not overflow to infinity.
        let tiny = parent_entropy(&[0.5, 1.9e-309]);

        assert!((parent_entropy(&[1.0, 1.0, 5e-309]) - 1.0).abs() < 1e-12);
        assert!(tiny.is_finite() && tiny < 1e-300, "{tiny}");
    }

    #[test]
    fn a_subnormal_parent_weight_keeps_the_state_line_readable() {
        // Accepted settings can give a parent a weight below the smallest normal number:
        // the sigmoid bottoms out just before exp overflows.
        let cfg = BreedConfig {
            dgm_lambda: 1000.0,
            ..BreedConfig::default()
        };
        let fitness = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, -0.2095];
        let weights = dgm_weights(&fitness, &[0; 7], cfg.dgm_lambda, cfg.dgm_alpha0);

        assert!(cfg.check().is_ok());
        assert!(
            weights[6] > 0.0 && weights[6] < f64::MIN_POSITIVE,
            "{weights:?}"
        );

        // Six equal weights and one next to nothing give log2 6 bits, and the state reads
        // back from its log line in canonical form (a non-finite number would be written
        // as null).
        let state = observe(&input(&family(), &[], &weights));
        let event = Event::State(state.clone());
        let json = serde_json::to_string(&event).expect("json");
        let back: Event = serde_json::from_str(&json).expect("the line reads back");

        assert!((state.diversity.parent_entropy_bits - 6f64.log2()).abs() < 1e-12);
        assert_eq!(back, event);
        assert_eq!(serde_json::to_string(&back).expect("json"), json);
    }

    #[test]
    fn lineage_and_quantiles_on_a_hand_built_archive() {
        let archive = family();
        let state = observe(&input(&archive, &[], &[]));
        let l = &state.lineage;

        // Depths 0, 0, 0, 1, 1, 2, 2. Children: s1 two, s2, a and b one each.
        assert_eq!(l.depth_max, 2);
        assert!((l.depth_mean - 6.0 / 7.0).abs() < 1e-12);
        assert_eq!(l.branching_max, 2);
        assert!((l.branching_mean - 1.25).abs() < 1e-12);
        assert!((l.leaf_share - 3.0 / 7.0).abs() < 1e-12);

        // c descends from s1, d from s1 and s2; the lineage of s3 died out.
        assert_eq!((l.seed_lineages, l.seed_lineages_alive), (3, 2));
        assert!((l.surviving_share - 2.0 / 3.0).abs() < 1e-12);

        // Solved rates, sorted: 0, 1/4, 1/4, 1/2, 1/2, 3/4, 1. The new members have 1/4
        // and 1/2.
        let p = &state.performance;

        assert_eq!((p.archive_size, p.born, p.tasks), (7, 2, 4));
        assert_eq!(
            (p.solved_best, p.solved_median, p.solved_best_born),
            (1.0, 0.5, 0.5)
        );
        assert_eq!((p.quality_best, p.quality_median), (None, None));

        let q = quantiles(&[1.0, 0.25, 0.5, 0.0, 0.75, 0.5, 0.25]).expect("quantiles");

        assert_eq!(
            (q.min, q.p25, q.median, q.p75, q.max),
            (0.0, 0.25, 0.5, 0.5, 1.0)
        );

        // An even count takes the lower median.
        let even = quantiles(&[4.0, 1.0, 3.0, 2.0]).expect("quantiles");

        assert_eq!(
            (even.min, even.p25, even.median, even.p75, even.max),
            (1.0, 1.0, 2.0, 3.0, 4.0)
        );
        assert_eq!(quantiles(&[]), None);

        // A graded archive reports quality rates too.
        let mut graded = family();

        for (m, q) in graded.iter_mut().zip([0.5, 1.0, 1.5, 2.0, 4.0, 3.0, 2.5]) {
            m.quality = Some(q);
        }

        let p = observe(&input(&graded, &[], &[])).performance;

        assert_eq!((p.quality_best, p.quality_median), (Some(1.0), Some(0.5)));
    }

    #[test]
    fn diversity_counts_shapes_blocks_models_and_behaviours() {
        let mut archive = family();
        let stage = archive[5].cascade.stages[0].clone();
        let mut large = archive[6].cascade.stages[0].workers[0].clone();

        // c gets a second Stage, d a second Worker on the large model.
        large.model = "sim:large".to_string();
        archive[5].cascade.stages.push(stage);
        archive[6].cascade.stages[0].workers.push(large);

        let d = observe(&input(&archive, &[], &[1.0; 7])).diversity;

        // Shapes "1" five times, "1-1" and "2" once each.
        assert_eq!(d.shapes, 3);
        assert!((d.shape_entropy_bits - by_definition(&[5.0, 1.0, 1.0])).abs() < 1e-12);

        // Nine Workers carry the same three sections.
        assert_eq!(d.blocks, 3);
        assert!((d.block_entropy_bits - 3f64.log2()).abs() < 1e-12);
        assert!((d.model_entropy_bits - by_definition(&[8.0, 1.0])).abs() < 1e-12);

        // The answers 0, 1, 2, 3, 4, 1, 2 hold five behaviours.
        assert_eq!(d.behaviours, 5);
        assert!((d.parent_entropy_bits - 7f64.log2()).abs() < 1e-12);
    }

    #[test]
    fn a_block_counts_once_for_each_worker_that_carries_it() {
        // A Worker the Mutator wrote may carry the same section twice; it still carries
        // that block once.
        let mut archive = vec![member("s1", &[], 0), member("s2", &[], 1)];
        let brief = Section {
            name: "style".to_string(),
            text: "Be brief.".to_string(),
        };

        archive[0].cascade.stages[0].workers[0]
            .sections
            .extend([brief.clone(), brief]);

        let d = observe(&input(&archive, &[], &[1.0; 2])).diversity;

        // Both Workers carry the three seed sections, the first one the style as well.
        assert_eq!(d.blocks, 4);
        assert!((d.block_entropy_bits - by_definition(&[2.0, 2.0, 2.0, 1.0])).abs() < 1e-12);
    }

    #[test]
    fn judging_counts_new_scored_and_trusted_judges() {
        let mut archive = family();

        archive[0].judge_correct = 3;
        archive[1].judge_wrong = 2;
        archive[3].judge_correct = 1;
        archive[3].judge_wrong = 1;

        let comparisons = [
            comparison(1, Some(false)),
            comparison(2, Some(true)),
            comparison(2, Some(false)),
            comparison(2, None),
            comparison(2, Some(true)),
        ];
        let j = observe(&input(&archive, &comparisons, &[])).judging;

        // Generation 2 has four comparisons, three of them scored and two right.
        assert_eq!((j.comparisons_new, j.scored_new), (4, 3));
        assert_eq!(j.accuracy_new, Some(2.0 / 3.0));

        // Skills 4/5, 1/4 and 1/2; only the first is trusted, with vote weight ln 4.
        let skill = j.skill.expect("skill");

        assert_eq!(j.judges_scored, 3);
        assert_eq!((skill.min, skill.median, skill.max), (0.25, 0.5, 0.8));
        assert!((j.trusted_share - 1.0 / 3.0).abs() < 1e-12);
        assert!((j.vote_weight_mean - 4f64.ln() / 3.0).abs() < 1e-12);
    }

    #[test]
    fn costs_are_this_generations_usage_and_archive_means() {
        let mut archive = family();
        let usage = Usage {
            calls: 9,
            prompt_tokens: 90,
            completion_tokens: 45,
            model_cost: 0.25,
        };

        archive[0].cascade.stages[0].workers[0].traits.think_tokens = 1024;

        let c = observe(&ObserveInput {
            generation_usage: usage,
            ..input(&archive, &[], &[])
        })
        .cost;

        assert_eq!(
            (c.calls, c.prompt_tokens, c.completion_tokens, c.model_cost),
            (9, 90, 45, 0.25)
        );
        assert!((c.think_tokens_mean - (1024.0 + 6.0 * 256.0) / 7.0).abs() < 1e-9);
        assert_eq!((c.workers_mean, c.stages_mean), (1.0, 1.0));
    }

    #[test]
    fn edge_cases_give_finite_values() {
        let empty: Vec<Member<u32>> = Vec::new();
        let archive = family();
        let mut bred_only = family();

        // No seed in the archive, and members without known parents.
        for m in &mut bred_only {
            m.op = Op::SectionAdd;
            m.parents = vec!["gone".to_string()];
        }

        let states = [
            observe(&ObserveInput {
                generation: 0,
                archive: &empty,
                first_new: 0,
                comparisons: &[],
                selection_weights: &[],
                generation_usage: Usage::default(),
                tasks: 0,
                soul: None,
                mind: None,
            }),
            observe(&ObserveInput {
                first_new: 99,
                tasks: 0,
                ..input(&archive, &[], &[0.0; 7])
            }),
            observe(&input(
                &bred_only,
                &[comparison(2, None)],
                &[f64::INFINITY, 1.0],
            )),
        ];

        for state in states {
            let json = serde_json::to_string(&state).expect("json");
            let mut all = Vec::new();

            numbers(&serde_json::from_str(&json).expect("value"), &mut all);

            assert!(!json.contains("null"), "{json}");
            assert!(
                all.iter()
                    .all(|x| x.is_finite() && x.to_bits() != (-0.0f64).to_bits()),
                "{json}"
            );
        }

        let none = observe(&input(&bred_only, &[], &[]));

        assert_eq!(
            (none.lineage.seed_lineages, none.lineage.surviving_share),
            (0, 0.0)
        );
        assert_eq!((none.lineage.depth_max, none.lineage.branching_max), (0, 0));
        assert_eq!((none.judging.judges_scored, none.judging.skill), (0, None));
    }

    #[test]
    fn state_serialises_canonically() {
        let mut archive = family();

        archive[3].quality = Some(2.5);
        archive[0].judge_correct = 2;
        archive[2].judge_wrong = 1;

        let comparisons = [comparison(2, Some(true)), comparison(2, Some(false))];
        let weights = [0.5, 0.25, 0.0, 0.125, 0.3, 0.2, 0.1];
        let state = observe(&input(&archive, &comparisons, &weights));

        assert!(state.performance.quality_best.is_some());
        assert!(state.judging.accuracy_new.is_some() && state.judging.skill.is_some());

        // A log line must read back to the same bytes (the log checks its canonical form).
        let event = Event::State(state);
        let json = serde_json::to_string(&event).expect("json");
        let back: Event = serde_json::from_str(&json).expect("back");

        assert_eq!(back, event);
        assert_eq!(serde_json::to_string(&back).expect("json"), json);
        assert!(
            json.starts_with(r#"{"kind":"state","generation":2,"performance":{"#),
            "{json}"
        );

        // Undefined values are left out, never written as null.
        let plain = Event::State(observe(&input(&family(), &[], &[])));
        let json = serde_json::to_string(&plain).expect("json");
        let back: Event = serde_json::from_str(&json).expect("back");

        for absent in ["quality", "accuracy", "skill", "null"] {
            assert!(!json.contains(absent), "{absent} in {json}");
        }

        assert_eq!(serde_json::to_string(&back).expect("json"), json);
    }

    #[test]
    fn the_mind_group_averages_values_and_leaves_out_what_it_lacks() {
        let mut lph = ValueSums::default();
        let mut other = ValueSums::default();

        for v in [1000, -500, 0] {
            lph.add(v);
        }

        for v in [250, 250, -1000, 0, 1] {
            other.add(v);
        }

        let counts = ImaginationCounts {
            choices: 3,
            changed: 1,
            linear_phase_hadamard: lph,
            other_families: other,
        };
        let state = observe(&ObserveInput {
            mind: Some(MindInput {
                goal: "-Linear",
                imagination: Some(counts),
            }),
            ..input(&family(), &[], &[])
        });
        let event = Event::State(state.clone());
        let json = serde_json::to_string(&event).expect("json");
        let back: Event = serde_json::from_str(&json).expect("back");

        // Means of the sums: 500 / 3 and −499 / 5, in thousandths, after the other groups.
        assert_eq!(
            state.mind,
            Some(MindState {
                goal: "-Linear".to_string(),
                imagination_choices: Some(3),
                imagination_changed: Some(1),
                linear_phase_hadamard_mean: Some(500.0 / 3.0),
                other_families_mean: Some(-99.8),
            })
        );
        assert!(json.ends_with(r#""mind":{"goal":"-Linear","imagination_choices":3,"imagination_changed":1,"linear_phase_hadamard_mean":166.66666666666666,"other_families_mean":-99.8}}"#), "{json}");
        assert_eq!(back, event);
        assert_eq!(serde_json::to_string(&back).expect("json"), json);

        // Without imagination only the goal is logged; without values to average, no mean;
        // without the settings, no group.
        let goal_only = observe(&ObserveInput {
            mind: Some(MindInput {
                goal: "all",
                imagination: None,
            }),
            ..input(&family(), &[], &[])
        });
        let no_parents = observe(&ObserveInput {
            mind: Some(MindInput {
                goal: "all",
                imagination: Some(ImaginationCounts::default()),
            }),
            ..input(&family(), &[], &[])
        });
        let off = observe(&input(&family(), &[], &[]));

        assert_eq!(
            serde_json::to_string(&goal_only.mind).expect("json"),
            r#"{"goal":"all"}"#
        );
        assert_eq!(
            serde_json::to_string(&no_parents.mind).expect("json"),
            r#"{"goal":"all","imagination_choices":0,"imagination_changed":0}"#
        );
        assert_eq!(off.mind, None);
        assert!(!serde_json::to_string(&off).expect("json").contains("mind"));
    }

    #[test]
    fn honesty_is_zero_until_overseers() {
        let state = observe(&input(&family(), &[comparison(2, Some(true))], &[1.0; 7]));
        let json = serde_json::to_string(&state.honesty).expect("json");

        assert_eq!(state.honesty, Honesty::default());
        assert!(!state.honesty.overseers);
        assert_eq!(
            json,
            r#"{"overseers":false,"fakes_planted":0,"fakes_caught":0,"false_alarms":0,"confirmed_collusion":0}"#
        );
    }

    /// A Worker of a hand-built archive with Souls of two phases: its birth Soul and κ,
    /// whether its Soul is new, the origin of its Soul, its imprinted and its current Soul.
    struct Souled {
        birth: [u16; 2],
        kappa: u16,
        new: bool,
        origin: (&'static str, usize),
        imprinted: [u16; 2],
        now: [u16; 2],
    }

    /// The member `id` of `generation` whose one Stage holds `workers`, with the tally
    /// `votes` of its birth votes.
    fn souled_member(
        id: &str,
        generation: usize,
        workers: &[Souled],
        votes: VoteTally,
    ) -> Member<u32> {
        let worker = seed_cascades()[0].1.stages[0].workers[0].clone();
        let phases = |p: &[u16; 2]| PhaseState(p.to_vec());
        let mut m = member(id, &[], 0);

        m.generation = generation;
        m.cascade.stages[0].workers = workers
            .iter()
            .map(|s| {
                let mut w = worker.clone();

                w.traits.phase_state = Some(phases(&s.birth));
                w.traits.coupling = s.kappa;
                w
            })
            .collect();
        m.soul_sources = Some(
            workers
                .iter()
                .map(|s| SoulSource {
                    sources: Vec::new(),
                    new: s.new,
                    origin: WorkerRef {
                        member: s.origin.0.to_string(),
                        worker: s.origin.1,
                    },
                })
                .collect(),
        );
        m.imprinted = Some(workers.iter().map(|s| phases(&s.imprinted)).collect());
        m.soul_now = Some(workers.iter().map(|s| phases(&s.now)).collect());
        m.votes = Some(votes);
        m
    }

    /// Seeds A (two Workers) and B (one), and C, a child of A born in generation 1: five
    /// Workers. A's second Worker and C's first moved at this commit by a quarter turn in
    /// their second family; C's first Worker was imprinted by a quarter turn before that.
    fn soul_archive() -> Vec<Member<u32>> {
        let tally = |complex, flipped, tied, coherence_sum, sim_decided| VoteTally {
            complex,
            flipped,
            tied,
            coherence_sum,
            sim_decided,
        };
        let w = |birth, kappa, new, origin, imprinted, now| Souled {
            birth,
            kappa,
            new,
            origin,
            imprinted,
            now,
        };

        vec![
            souled_member(
                "A",
                0,
                &[
                    w([0, 0], 200, true, ("A", 0), [0, 0], [0, 0]),
                    w(
                        [16384, 16384],
                        400,
                        true,
                        ("A", 1),
                        [16384, 16384],
                        [16384, 32768],
                    ),
                ],
                tally(7, 7, 7, 7.0, 7),
            ),
            souled_member(
                "B",
                0,
                &[w([32768, 0], 0, true, ("B", 0), [32768, 0], [32768, 0])],
                VoteTally::default(),
            ),
            souled_member(
                "C",
                1,
                &[
                    w([0, 16384], 200, false, ("A", 0), [0, 32768], [0, 49152]),
                    w([49152, 0], 1000, true, ("C", 1), [49152, 0], [49152, 0]),
                ],
                tally(4, 1, 2, 2.5, 3),
            ),
        ]
    }

    #[test]
    fn the_soul_group_counts_by_hand() {
        let archive = soul_archive();
        // Three tasks: two with both families fitting, one with the first alone.
        let strengths = [(vec![1.0, 1.0], 2), (vec![1.0, 0.0], 1)];
        let moved = [
            (
                PhaseState(vec![16384, 16384]),
                PhaseState(vec![16384, 32768]),
            ),
            (PhaseState(vec![0, 32768]), PhaseState(vec![0, 49152])),
        ];
        let group = |readout: Readout| {
            observe(&ObserveInput {
                generation: 1,
                archive: &archive,
                first_new: 2,
                comparisons: &[],
                selection_weights: &[1.0; 3],
                generation_usage: Usage::default(),
                tasks: 3,
                soul: Some(SoulInput {
                    readout,
                    families: 2,
                    strengths: &strengths,
                    observations_vote: 5,
                    observations_peer: 3,
                    moved: &moved,
                    variation: None,
                }),
                mind: None,
            })
            .soul
            .expect("a soul group")
        };
        let near = |x: Option<f64>, y: f64| x.is_some_and(|x| (x - y).abs() < 1e-12);

        for readout in [Readout::Complex, Readout::Real] {
            let s = group(readout);

            // Five Workers, two born now, one of them with a new Soul. Origins: A's first
            // Worker twice (C copied it), A's second, B's, and C's new one.
            assert_eq!((s.souls, s.souls_born, s.souls_new), (5, 2, 1));
            assert_eq!(s.origins, 4);
            assert!((s.origin_entropy_bits - by_definition(&[2.0, 1.0, 1.0, 1.0])).abs() < 1e-12);

            // The circular distance from the birth Soul: A's second Worker 90 degrees in one
            // family, C's first 180 (imprinted, then observed), over 10 phases.
            assert_eq!(s.moved_mean, Some(27.0));

            // κ 0.2, 0.4, 0, 0.2 and 1.
            let k = s.kappa.expect("κ quantiles");

            assert_eq!(
                (k.min, k.p25, k.median, k.p75, k.max),
                (0.0, 0.2, 0.2, 0.4, 1.0)
            );

            // This commit: 5 and 3 Observations; two Workers moved, by 90 degrees in one of
            // their two families.
            assert_eq!((s.observations_vote, s.observations_peer), (5, 3));
            assert_eq!(s.shift_mean, Some(45.0));

            // The votes of the newborn C alone.
            assert_eq!(
                (s.votes, s.votes_flipped, s.votes_tied, s.sim_decided),
                (4, 1, 2, 3)
            );
            assert_eq!(s.coherence_mean, Some(0.625));
        }

        // The current phases read 1, i, -1, -i at the quarter turns. With both families,
        // the normalised Characters are 2, 1, 0, 1, 1 (mean 1, contrast √0.4); with the
        // first alone every one is 1. Order: |1 + i - 1 + 1 - i| / 5 and |2 - i| / 5.
        let complex = group(Readout::Complex);
        let order = complex.order.clone().expect("order");

        assert!(near(complex.character_norm_mean, 1.0));
        assert!(near(complex.contrast, 2.0 * 0.4f64.sqrt() / 3.0));
        assert!(near(
            complex.contrast_independent,
            2.0 * 0.5f64.sqrt() / 3.0
        ));
        assert!((order[0] - 0.2).abs() < 1e-12 && (order[1] - 5f64.sqrt() / 5.0).abs() < 1e-12);
        assert!(near(complex.order_mean, (0.2 + 5f64.sqrt() / 5.0) / 2.0));

        // The signs: +1, -1, -1, +1, +1 and +1, -1, +1, +1, +1. The normalised Characters
        // with both families are 2, 2, 0, 2, 2 (mean 1.6, contrast 0.5), and the formula is
        // √2 times the complex one, 1 on these tasks.
        let real = group(Readout::Real);

        assert!(near(real.character_norm_mean, 1.4));
        assert!(near(real.contrast, 1.0 / 3.0));
        assert!(near(real.contrast_independent, 2.0 / 3.0));
        assert_eq!(real.order, Some(vec![0.2, 0.6]));
        assert!(near(real.order_mean, 0.4));

        // Undefined values are absent and empty counts 0: no Worker, no task, no move, no
        // vote.
        let empty = soul_state::<u32>(
            &[],
            0,
            &SoulInput {
                readout: Readout::Complex,
                families: 2,
                strengths: &[],
                observations_vote: 0,
                observations_peer: 0,
                moved: &[],
                variation: None,
            },
        );
        let json = serde_json::to_string(&empty).expect("json");

        assert_eq!((empty.souls, empty.origins, empty.votes), (0, 0, 0));
        assert_eq!(empty.origin_entropy_bits.to_bits(), 0.0f64.to_bits());
        assert_eq!(
            json,
            r#"{"souls":0,"souls_born":0,"souls_new":0,"origins":0,"origin_entropy_bits":0.0,"observations_vote":0,"observations_peer":0,"votes":0,"votes_flipped":0,"votes_tied":0,"sim_decided":0}"#
        );

        // With `variation` on, the choices the preferences decided end the group, also when
        // they are 0.
        for (steered, tail) in [
            (
                Steered {
                    choices: 7,
                    changed: 2,
                },
                r#","sim_decided":0,"variation_choices":7,"variation_changed":2}"#,
            ),
            (
                Steered::default(),
                r#","sim_decided":0,"variation_choices":0,"variation_changed":0}"#,
            ),
        ] {
            let on = soul_state::<u32>(
                &[],
                0,
                &SoulInput {
                    readout: Readout::Complex,
                    families: 2,
                    strengths: &[],
                    observations_vote: 0,
                    observations_peer: 0,
                    moved: &[],
                    variation: Some(steered),
                },
            );
            let json = serde_json::to_string(&on).expect("json");

            assert!(json.ends_with(tail), "{json}");
            assert_eq!(serde_json::from_str::<SoulState>(&json).expect("read"), on);
        }
    }

    #[test]
    fn contrast_weights_tasks() {
        let circuits = CircuitDomain::new(&CircuitConfig::default()).expect("circuit domain");
        // The search panel of the step 1 pilot: three dressings of each of the 11 targets.
        let panel = circuits.panel(derive_seed(101, "panel/search"), 33);
        let mut per_target: BTreeMap<&str, usize> = BTreeMap::new();

        for t in &panel {
            *per_target.entry(t.name).or_default() += 1;
        }

        assert_eq!(per_target.len(), 11);
        assert!(per_target.values().all(|&n| n == 3), "{per_target:?}");

        let counts = strength_counts(panel.iter().map(|t| circuits.strengths(t)));

        assert_eq!(counts.len(), 7, "seven distinct strength vectors");
        assert_eq!(counts.iter().map(|(_, n)| n).sum::<usize>(), 33);

        for (readout, by_target, by_vector) in [
            (Readout::Complex, 0.655, 0.563),
            (Readout::Real, 0.927, 0.796),
        ] {
            let input = SoulInput {
                readout,
                families: 8,
                strengths: &counts,
                observations_vote: 0,
                observations_peer: 0,
                moved: &[],
                variation: None,
            };
            let k = soul_state(&soul_archive(), 2, &input)
                .contrast_independent
                .expect("a formula value");
            let alike = counts
                .iter()
                .map(|(s, _)| independent_contrast(s, readout))
                .sum::<f64>()
                / counts.len() as f64;

            assert!((k - by_target).abs() < 5e-4, "{readout:?}: {k}");
            assert!((alike - by_vector).abs() < 5e-4, "{readout:?}: {alike}");
            assert!((k - alike).abs() > 0.05);
        }
    }
}
