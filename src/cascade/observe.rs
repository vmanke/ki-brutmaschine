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
//! Every number is finite, since a NaN would reach the log as null and break it. A value
//! that is undefined because its set is empty is None, or 0 where the field is a count, a
//! share or a best value. Sums run in a fixed order, so the bytes do not depend on thread
//! count.
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::cascade::arena::{Comparison, Member, Usage};
use crate::cascade::genome::Cascade;
use crate::cascade::variation::Op;
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
fn entropy(parts: &[f64]) -> f64 {
    let total = parts.iter().fold(0.0, |sum, &p| sum + p);

    if !(total > 0.0 && total.is_finite()) {
        return 0.0;
    }

    parts
        .iter()
        .filter(|&&p| p > 0.0)
        .fold(0.0, |sum, &p| sum + p / total * (total / p).log2())
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
            *models.entry(w.model.as_str()).or_default() += 1;

            for s in &w.sections {
                *blocks
                    .entry((s.name.as_str(), s.text.as_str()))
                    .or_default() += 1;
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
    use crate::cascade::arena::Event;
    use crate::cascade::genome::seed_cascades;

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
}
