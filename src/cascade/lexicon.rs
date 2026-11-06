//! A Lexicon of coined words and a brevity case for lexicase (arms L and R,
//! `.scratch/eigene-sprache/spec.md`, sections 2 and 3). Both settings are off by default; a
//! Run without them writes every byte as before.
//!
//! - The Lexicon (spec 3.2): with `lexicon` on, the gene pool's phrases grow during a Run. A
//!   coining step follows the commit of every generation g > 0 with g mod `every` = 0. It
//!   reads the Workers of the `best` best members, the first of the archive's ranking, retires
//!   the words they no longer hold, and coins at most one word: a readable phrase built from
//!   one fixed term per family, which names a set of two or three families, its meaning
//!   (`phrase`). Arm L coins the set whose word would shorten those Workers' texts most,
//!   counted in words: a description length in the sense of Rissanen (1978). Arm R, the
//!   control, lets L's rule decide whether to coin and how many families the word names, and
//!   draws the set at random among those of that size. A coined phrase joins the pool after
//!   the fixed phrases, under a section name, and the section operators draw it like any other
//!   phrase (`Lexicon::pool`). A coining step changes no member.
//! - Brevity (spec 3.3): with `brevity` on, lexicase gets one more case. A member passes it
//!   when the words of its Workers' system prompts are at most the lower median of the archive
//!   (`brevity_case`). Like every case it goes through lexicase's filter: when it is drawn and
//!   some remaining member passes, the longer ones drop out; when none passes, all stay.
//!
//! Pure computation: no files, no network, no clock. Arm R draws from a stream of its own,
//! never from the generation's.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::backend::approx_tokens;
use crate::cascade::genome::{Cascade, Worker};
use crate::util::{BrutError, Result};

/// The fewest families a coined word names (spec 3.2).
pub const MEANING_MIN: usize = 2;

/// The most families a coined word names (spec 3.2).
pub const MEANING_MAX: usize = 3;

/// How a coining step chooses the meaning of its word (spec 3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Meaning {
    /// Arm L: the set of families whose word saves the most words.
    Compression,
    /// Arm R, the control: L's rule decides whether to coin and the size of the set, and the
    /// set is drawn at random among the sets of that size without a word.
    Random,
}

/// The settings of the Lexicon (arms L and R, spec 3.2). Every field is required and no other
/// is allowed. They define the experiment, so a resume cannot change them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexiconConfig {
    /// A coining step follows the commit of every generation g with g > 0 and g mod `every` =
    /// 0: 20 in arms L and R.
    pub every: usize,
    /// The members a coining step reads: the first `best` of the archive's ranking (fitness,
    /// then peer rating, then gene id), 22 in arms L and R. A smaller archive gives all.
    pub best: usize,
    /// The most words in the pool at once: 8 in arms L and R.
    pub max_words: usize,
    /// How the meaning of a word is chosen: "compression" in arm L, "random" in arm R.
    pub meaning: Meaning,
}

impl LexiconConfig {
    /// The setting of arm L.
    pub fn arm_l() -> LexiconConfig {
        LexiconConfig {
            every: 20,
            best: 22,
            max_words: 8,
            meaning: Meaning::Compression,
        }
    }

    /// The setting of arm R: arm L's with meanings drawn at random.
    pub fn arm_r() -> LexiconConfig {
        LexiconConfig {
            meaning: Meaning::Random,
            ..LexiconConfig::arm_l()
        }
    }

    /// Reject settings that cannot give a meaningful Run.
    pub fn check(&self) -> Result<()> {
        for (name, value) in [
            ("every", self.every),
            ("best", self.best),
            ("max_words", self.max_words),
        ] {
            if value == 0 {
                return Err(BrutError::Invalid(format!(
                    "breed config: lexicon: {name} must be at least 1"
                )));
            }
        }

        Ok(())
    }

    /// True when a coining step follows the commit of generation `g`: g > 0 and g mod
    /// `every` = 0.
    pub fn coins_after(&self, g: usize) -> bool {
        g > 0 && g.is_multiple_of(self.every)
    }
}

/// What a gene pool gives the Lexicon (spec 3.2): one readable term per family of the
/// Domain, in its family order, and the families a text names in the order of their first
/// mention, by their places, as the simulated designer reads them.
#[derive(Clone, Copy, Debug)]
pub struct LexiconGenes {
    pub terms: &'static [&'static str],
    pub mentions: fn(&str) -> Vec<usize>,
}

/// The phrase of a coined word whose families are `order`, by their places and in this
/// order (spec 3.2): "Use {t1}, then {t2}." for two families and "Use {t1}, then {t2}, then
/// {t3}." for three, with each family's term from `terms`. A family without a term is left
/// out. Its words do not depend on the order.
pub fn phrase(terms: &[&str], order: &[usize]) -> String {
    let named: Vec<&str> = order
        .iter()
        .filter_map(|&f| terms.get(f).copied())
        .collect();

    format!("Use {}.", named.join(", then "))
}

/// The words of `text`, as `approx_tokens` counts them and the simulated models count prompt
/// tokens: the pieces between whitespace.
pub fn words(text: &str) -> i64 {
    i64::from(approx_tokens(text))
}

/// The brevity case of lexicase (spec 3.3) for members whose prompt words are `words`, one
/// value per member in the same order: 1 when its words are at most the lower median of all
/// of them, the value at place (n − 1) / 2 of their sorted list, else 0. Empty without a
/// member. At least half of the members pass it.
pub fn brevity_case(words: &[u64]) -> Vec<f64> {
    let mut sorted = words.to_vec();

    sorted.sort_unstable();

    let Some(&median) = sorted.get(sorted.len().saturating_sub(1) / 2) else {
        return Vec::new();
    };

    words
        .iter()
        .map(|&w| if w <= median { 1.0 } else { 0.0 })
        .collect()
}

/// One coined word (spec 3.2, 3.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Word {
    /// Its readable phrase, as `phrase` writes it.
    pub phrase: String,
    /// The families it names, by their places, in index order: two or three.
    pub meaning: Vec<usize>,
    /// The section name it joins the pool under.
    pub section: String,
    /// Its saving in words on the Workers of the step that coined it (`saving`): above 0 in
    /// arm L, any number in arm R.
    pub saving: i64,
    /// The generation whose commit the step that coined it followed.
    pub generation: usize,
    /// The generation of the step that retired it; None while it is in the pool.
    pub retired: Option<usize>,
    /// While it is in the pool: true when no Worker of the best members held it at the last
    /// coining step.
    pub unheld: bool,
}

/// What one coining step did: the words it retired and the word it coined, each by its
/// place in the Lexicon.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Coining {
    pub retired: Vec<usize>,
    pub coined: Option<usize>,
}

/// What one coining step reads (spec 3.2).
#[derive(Clone, Copy, Debug)]
pub struct Step<'a> {
    /// The generation whose commit the step follows.
    pub generation: usize,
    /// The Cascades of the best members, the first `best` of the archive's ranking, in that
    /// order.
    pub best: &'a [&'a Cascade],
    /// The number of families of the Domain.
    pub families: usize,
    /// The families a text names, by their places, in index order: the gene pool's rule.
    pub families_of: fn(&str) -> Vec<usize>,
    /// The gene pool's fixed phrases, whose numbers per section name break a tie of section
    /// names (`section`).
    pub phrases: &'a [(&'a str, &'a str)],
    pub genes: LexiconGenes,
    pub config: LexiconConfig,
    /// The seed of arm R's draw: `derive_seed` of the Run's seed and `lexicon/{g}`. Arm L
    /// draws nothing.
    pub draw_seed: u64,
}

/// A Run's Lexicon (spec 3.2): every word it coined, in coining order. The words not retired
/// are in the pool.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Lexicon {
    pub words: Vec<Word>,
}

impl Lexicon {
    /// The words in the pool, in coining order.
    pub fn active(&self) -> impl Iterator<Item = &Word> {
        self.words.iter().filter(|w| w.retired.is_none())
    }

    /// The words retired so far.
    pub fn retired(&self) -> usize {
        self.words.iter().filter(|w| w.retired.is_some()).count()
    }

    /// The phrase pool the section operators draw from (spec 3.2): `base`, the gene pool's
    /// fixed phrases, then the phrase of every word in the pool, in coining order, under its
    /// section name. Without a word in the pool it is `base`, phrase for phrase, so the
    /// operators draw exactly as without a Lexicon.
    pub fn pool<'p>(&'p self, base: &[(&'p str, &'p str)]) -> Vec<(&'p str, &'p str)> {
        base.iter()
            .copied()
            .chain(
                self.active()
                    .map(|w| (w.section.as_str(), w.phrase.as_str())),
            )
            .collect()
    }

    /// How many of `best` hold a coined word: a Worker with a section whose text is the
    /// phrase of a word of this Lexicon, in the pool or retired (spec 3.4).
    pub fn holding(&self, best: &[&Cascade]) -> usize {
        best.iter()
            .filter(|c| workers(c).any(|w| self.words.iter().any(|word| holds(w, &word.phrase))))
            .count()
    }

    /// The coining step `s` (spec 3.2). The words of the step are `approx_tokens` of a text,
    /// and the Workers it reads are those of `s.best`, member after member, each in Stage
    /// order.
    ///
    /// 1. Retirement comes first. A word in the pool that no Worker holds, as the text of one
    ///    of its sections, at this step and at the step before leaves the pool. A word
    ///    coined at the step before was first read then, so it lives at least two steps.
    /// 2. Then, unless the pool holds `max_words` words, L's rule. A Worker counts for a set
    ///    M of two or three families if, for each family of M, it holds a section whose text
    ///    names exactly that one family; of several, the first in the Worker's order. The
    ///    saving of M is the sum over the Workers that count of the words of those sections
    ///    minus the words of M's phrase, minus the words of M's phrase once more, for the
    ///    Lexicon (`saving`). Every set with a word in the pool is no candidate; a retired
    ///    word's set is one again. The set with the largest saving wins, and a tie goes to the
    ///    lexicographically smaller sorted list of places, so {0, 1} before {0, 1, 2} before
    ///    {0, 2}. If its saving is above 0, one word is coined, else none.
    /// 3. Arm L coins the winning set. Its phrase orders the families as the most Workers that
    ///    count name them by first mention in their system prompts (`order`), and its section
    ///    name is the name the replaced sections carry most often. A tie goes to the name with
    ///    the most fixed phrases in the gene pool, then to the alphabetically first
    ///    (`section`).
    /// 4. Arm R coins a set of the winner's size instead: one uniform draw (`random_range`)
    ///    among the sets of that size without a word in the pool, in lexicographic order,
    ///    from a ChaCha8 stream of `s.draw_seed` that nothing else reads. Its phrase names the
    ///    families in index order. It takes the winner's section name, so that the two arms
    ///    differ only in how the meaning is chosen, and it logs its own saving, which may be 0
    ///    or below.
    pub fn step(&mut self, s: &Step<'_>) -> Coining {
        let g = s.generation;
        let workers: Vec<&Worker> = s.best.iter().flat_map(|c| workers(c)).collect();
        let mut done = Coining::default();

        for (k, word) in self.words.iter_mut().enumerate() {
            if word.retired.is_some() {
                continue;
            }

            if workers.iter().any(|w| holds(w, &word.phrase)) {
                word.unheld = false;
            } else if word.unheld {
                word.retired = Some(g);
                word.unheld = false;
                done.retired.push(k);
            } else {
                word.unheld = true;
            }
        }

        if self.active().count() >= s.config.max_words {
            return done;
        }

        let readings: Vec<Reading<'_>> = workers.iter().map(|w| Reading::of(w, s)).collect();
        let taken: Vec<Vec<usize>> = self.active().map(|w| w.meaning.clone()).collect();
        let free: Vec<Vec<usize>> = meanings(s.families)
            .into_iter()
            .filter(|m| !taken.contains(m))
            .collect();
        let saving_of = |m: &[usize]| saving(&readings, m, words(&phrase(s.genes.terms, m)));
        let mut winner: Option<(&Vec<usize>, i64)> = None;

        for m in &free {
            let saved = saving_of(m);

            if winner.is_none_or(|(_, best)| saved > best) {
                winner = Some((m, saved));
            }
        }

        let Some((chosen, saved)) = winner.filter(|(_, saved)| *saved > 0) else {
            return done;
        };
        let name = section(&readings, chosen, s.phrases).unwrap_or_default();
        let (meaning, order, saved) = match s.config.meaning {
            Meaning::Compression => (chosen.clone(), order(&readings, chosen), saved),
            Meaning::Random => {
                // `free` holds the winner, so a set of its size is always there.
                let sized: Vec<&Vec<usize>> =
                    free.iter().filter(|m| m.len() == chosen.len()).collect();
                let mut rng = ChaCha8Rng::seed_from_u64(s.draw_seed);
                let drawn = sized[rng.random_range(0..sized.len())].clone();
                let saved = saving_of(&drawn);

                (drawn.clone(), drawn, saved)
            }
        };

        self.words.push(Word {
            phrase: phrase(s.genes.terms, &order),
            meaning,
            section: name,
            saving: saved,
            generation: g,
            retired: None,
            unheld: false,
        });
        done.coined = Some(self.words.len() - 1);

        done
    }
}

/// The Workers of `c`, in Stage order.
fn workers(c: &Cascade) -> impl Iterator<Item = &Worker> {
    c.stages.iter().flat_map(|s| &s.workers)
}

/// True when `w` holds `phrase`: the text of one of its sections is the phrase.
fn holds(w: &Worker, phrase: &str) -> bool {
    w.sections.iter().any(|s| s.text == phrase)
}

/// Every set of two or three of `n` families, as sorted lists of places, in lexicographic
/// order: {0, 1}, {0, 1, 2}, …, {0, 1, n − 1}, {0, 2}, {0, 2, 3}, and so on.
pub fn meanings(n: usize) -> Vec<Vec<usize>> {
    let mut all = Vec::new();

    for a in 0..n {
        for b in a + 1..n {
            all.push(vec![a, b]);

            for c in b + 1..n {
                all.push(vec![a, b, c]);
            }
        }
    }

    debug_assert!(
        all.iter()
            .all(|m| (MEANING_MIN..=MEANING_MAX).contains(&m.len()))
    );
    debug_assert!(all.windows(2).all(|w| w[0] < w[1]));

    all
}

/// What a coining step reads of one Worker: for each family, the words and the name of the
/// first section whose text names that family and no other; and the families its system
/// prompt names, in the order of their first mention.
struct Reading<'w> {
    alone: Vec<Option<(i64, &'w str)>>,
    mentions: Vec<usize>,
}

impl<'w> Reading<'w> {
    fn of(w: &'w Worker, s: &Step<'_>) -> Reading<'w> {
        let mut alone = vec![None; s.families];

        for section in &w.sections {
            let named = (s.families_of)(&section.text);

            if let &[f] = named.as_slice()
                && let Some(slot) = alone.get_mut(f)
                && slot.is_none()
            {
                *slot = Some((words(&section.text), section.name.as_str()));
            }
        }

        Reading {
            alone,
            mentions: (s.genes.mentions)(&w.system_prompt()),
        }
    }

    /// The words and the names of the sections a word meaning `m` replaces in this Worker,
    /// one per family of `m`, or None when the Worker does not count for `m`.
    fn replaced(&self, m: &[usize]) -> Option<(i64, Vec<&'w str>)> {
        let mut total = 0;
        let mut names = Vec::with_capacity(m.len());

        for &f in m {
            let (n, name) = (*self.alone.get(f)?)?;

            total += n;
            names.push(name);
        }

        Some((total, names))
    }
}

/// The saving of a word meaning `m` whose phrase has `phrase_words` words (spec 3.2): over
/// the Workers that count for `m`, the words of the sections it replaces minus the words of
/// the phrase, summed, minus the words of the phrase once more, for the Lexicon.
fn saving(readings: &[Reading<'_>], m: &[usize], phrase_words: i64) -> i64 {
    readings
        .iter()
        .filter_map(|r| r.replaced(m))
        .map(|(n, _)| n - phrase_words)
        .sum::<i64>()
        - phrase_words
}

/// The section name of a word meaning `m` (spec 3.2): the name that the sections it replaces
/// in the Workers that count carry most often. None when no Worker counts.
///
/// A tie is common: a Worker that counts often holds one family under "strategy" and the
/// other under a name that SectionReplace wrote it into. It goes to the name with the most of
/// the gene pool's fixed phrases `phrases`, then to the alphabetically first. The section
/// operators draw a text of the section's own name with p = 0.7, each text alike, so under
/// the name with the most texts the word gets the smallest share of those draws: it spreads
/// by selection, not because few texts share its name.
fn section(readings: &[Reading<'_>], m: &[usize], phrases: &[(&str, &str)]) -> Option<String> {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();

    for (_, names) in readings.iter().filter_map(|r| r.replaced(m)) {
        for name in names {
            *counts.entry(name).or_default() += 1;
        }
    }

    let fixed = |name: &str| phrases.iter().filter(|(n, _)| *n == name).count();

    // The names are distinct, so the last comparison decides every full tie.
    counts
        .into_iter()
        .max_by(|(a, x), (b, y)| {
            x.cmp(y)
                .then_with(|| fixed(a).cmp(&fixed(b)))
                .then_with(|| b.cmp(a))
        })
        .map(|(name, _)| name.to_string())
}

/// The order of the families of `m` in arm L's phrase (spec 3.2): the order in which the
/// most Workers that count for `m` name them, by first mention in their system prompts. A tie
/// goes to the order first in index order, the lexicographically smallest list of places, so
/// of two families the lower comes first. Without a Worker that counts, index order.
fn order(readings: &[Reading<'_>], m: &[usize]) -> Vec<usize> {
    let mut counts: BTreeMap<Vec<usize>, usize> = BTreeMap::new();

    for r in readings.iter().filter(|r| r.replaced(m).is_some()) {
        let at = |f: &usize| r.mentions.iter().position(|x| x == f).unwrap_or(usize::MAX);
        let mut seen = m.to_vec();

        seen.sort_by_key(|f| (at(f), *f));
        *counts.entry(seen).or_default() += 1;
    }

    most(counts).unwrap_or_else(|| m.to_vec())
}

/// The key of `counts` with the largest count; of several, the first in key order.
fn most<K: Ord>(counts: BTreeMap<K, usize>) -> Option<K> {
    let mut best: Option<(K, usize)> = None;

    for (key, n) in counts {
        if best.as_ref().is_none_or(|(_, b)| n > *b) {
            best = Some((key, n));
        }
    }

    best.map(|(key, _)| key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cascade::genome::{Section, Stage, Traits};
    use crate::cascade::selection::lexicase_pick;

    /// A toy gene pool of three families, each named by its word: alpha, beta, gamma.
    const TERMS: [&str; 3] = ["alpha", "beta", "gamma"];

    /// The toy gene pool's fixed phrases: two strategies and a check, and none of another
    /// name.
    const FIXED: [(&str, &str); 3] = [
        ("strategy", "make a plan"),
        ("strategy", "think first"),
        ("check", "check the answer"),
    ];

    /// The families a toy text names, in index order.
    fn toy_families(text: &str) -> Vec<usize> {
        (0..TERMS.len())
            .filter(|&f| text.contains(TERMS[f]))
            .collect()
    }

    /// The families a toy text names, in the order of their first mention.
    fn toy_mentions(text: &str) -> Vec<usize> {
        let mut at: Vec<(usize, usize)> = (0..TERMS.len())
            .filter_map(|f| text.find(TERMS[f]).map(|i| (i, f)))
            .collect();

        at.sort();
        at.into_iter().map(|(_, f)| f).collect()
    }

    fn worker(sections: &[(&str, &str)]) -> Worker {
        Worker {
            model: "sim:small".to_string(),
            sections: sections
                .iter()
                .map(|(name, text)| Section {
                    name: name.to_string(),
                    text: text.to_string(),
                })
                .collect(),
            traits: Traits::default(),
        }
    }

    fn cascade(workers: Vec<Worker>) -> Cascade {
        Cascade {
            stages: vec![Stage { workers }],
        }
    }

    /// The step after generation `g` on `best` with the setting `config`.
    fn step<'a>(g: usize, best: &'a [&'a Cascade], config: LexiconConfig) -> Step<'a> {
        Step {
            generation: g,
            best,
            families: TERMS.len(),
            families_of: toy_families,
            phrases: &FIXED,
            genes: LexiconGenes {
                terms: &TERMS,
                mentions: toy_mentions,
            },
            config,
            draw_seed: 7,
        }
    }

    /// What a coining step reads of each of `workers`.
    fn readings(workers: &[Worker]) -> Vec<Reading<'_>> {
        let s = step(20, &[], LexiconConfig::arm_l());

        workers.iter().map(|w| Reading::of(w, &s)).collect()
    }

    /// The word `phrase` of the set {0, 1}, coined after generation 20, retired after
    /// `retired`.
    fn word(phrase: &str, section: &str, retired: Option<usize>) -> Word {
        Word {
            phrase: phrase.to_string(),
            meaning: vec![0, 1],
            section: section.to_string(),
            saving: 1,
            generation: 20,
            retired,
            unheld: false,
        }
    }

    #[test]
    fn the_settings_are_checked() {
        assert!(LexiconConfig::arm_l().check().is_ok());
        assert!(LexiconConfig::arm_r().check().is_ok());
        assert_eq!(LexiconConfig::arm_r().meaning, Meaning::Random);
        assert_eq!(
            serde_json::to_string(&LexiconConfig::arm_l()).expect("json"),
            r#"{"every":20,"best":22,"max_words":8,"meaning":"compression"}"#
        );
        assert_eq!(
            serde_json::to_string(&LexiconConfig::arm_r()).expect("json"),
            r#"{"every":20,"best":22,"max_words":8,"meaning":"random"}"#
        );

        for bad in [
            LexiconConfig {
                every: 0,
                ..LexiconConfig::arm_l()
            },
            LexiconConfig {
                best: 0,
                ..LexiconConfig::arm_l()
            },
            LexiconConfig {
                max_words: 0,
                ..LexiconConfig::arm_r()
            },
        ] {
            assert!(bad.check().is_err(), "{bad:?}");
        }

        // An unknown or a missing field, or an unknown meaning, is refused.
        for bad in [
            r#"{"every":20,"best":22,"max_words":8,"meaning":"compression","x":1}"#,
            r#"{"every":20,"best":22,"meaning":"compression"}"#,
            r#"{"every":20,"best":22,"max_words":8,"meaning":"zipf"}"#,
        ] {
            assert!(serde_json::from_str::<LexiconConfig>(bad).is_err(), "{bad}");
        }

        // The steps follow the commits of generations 20, 40 and 60, never generation 0.
        let l = LexiconConfig::arm_l();

        assert_eq!(
            (0..=61).filter(|&g| l.coins_after(g)).collect::<Vec<_>>(),
            [20, 40, 60]
        );
    }

    #[test]
    fn a_phrase_names_its_terms_in_its_order_and_its_words_do_not_depend_on_it() {
        assert_eq!(phrase(&TERMS, &[1, 0]), "Use beta, then alpha.");
        assert_eq!(
            phrase(&TERMS, &[0, 2, 1]),
            "Use alpha, then gamma, then beta."
        );
        assert_eq!(words(&phrase(&TERMS, &[0, 1])), 4);
        assert_eq!(words(&phrase(&TERMS, &[2, 0, 1])), 6);
        assert_eq!(words(""), 0);

        let terms = ["linear row operations", "the phase polynomial"];

        assert_eq!(
            words(&phrase(&terms, &[0, 1])),
            words(&phrase(&terms, &[1, 0]))
        );
    }

    #[test]
    fn the_sets_are_in_lexicographic_order() {
        assert_eq!(
            meanings(4),
            vec![
                vec![0, 1],
                vec![0, 1, 2],
                vec![0, 1, 3],
                vec![0, 2],
                vec![0, 2, 3],
                vec![0, 3],
                vec![1, 2],
                vec![1, 2, 3],
                vec![1, 3],
                vec![2, 3],
            ]
        );
        // On the eight families of circuits: 28 pairs and 56 triples.
        assert_eq!(meanings(8).len(), 28 + 56);
        assert!(meanings(1).is_empty());
    }

    #[test]
    fn the_saving_counts_the_workers_that_hold_each_family_alone() {
        let workers = vec![
            // Counts for {alpha, beta}: 7 + 3 words.
            worker(&[
                ("role", "one two three"),
                ("strategy", "alpha a b c d e f"),
                ("check", "beta x y"),
            ]),
            // Names alpha and beta only together, so it does not count for them.
            worker(&[("strategy", "alpha and beta together")]),
            // Counts for {alpha, beta}: the first section that names alpha alone, 2 words,
            // and the one that names beta alone, 5 words. It names beta first.
            worker(&[
                ("strategy", "beta b1 b2 b3 b4"),
                ("strategy", "alpha a1"),
                ("residual", "alpha again and more words"),
            ]),
            // Names gamma alone, alpha and beta only together.
            worker(&[("strategy", "gamma g1 g2"), ("style", "alpha beta")]),
        ];
        let r = readings(&workers);
        let pair = words(&phrase(&TERMS, &[0, 1]));

        assert_eq!(pair, 4);
        assert_eq!(
            r[0].replaced(&[0, 1]),
            Some((10, vec!["strategy", "check"]))
        );
        assert_eq!(r[1].replaced(&[0, 1]), None);
        assert_eq!(
            r[2].replaced(&[0, 1]),
            Some((7, vec!["strategy", "strategy"]))
        );
        assert_eq!(r[3].replaced(&[0, 1]), None);
        assert_eq!(r[3].replaced(&[2]), Some((3, vec!["strategy"])));

        // (10 − 4) + (7 − 4) − 4.
        assert_eq!(saving(&r, &[0, 1], pair), 5);
        // No Worker counts for a set with gamma and another family: only the phrase's cost.
        assert_eq!(saving(&r, &[0, 2], pair), -4);
        assert_eq!(saving(&r, &[0, 1, 2], 6), -6);

        // The replaced sections are three strategies and a check. The two Workers that count
        // name the families in the orders [0, 1] and [1, 0]: a tie, so index order.
        assert_eq!(section(&r, &[0, 1], &FIXED).as_deref(), Some("strategy"));
        assert_eq!(order(&r, &[0, 1]), [0, 1]);
        assert_eq!(section(&r, &[0, 2], &FIXED), None);
        assert_eq!(order(&r, &[2, 0]), [2, 0]);

        // One more Worker that names beta first makes [1, 0] the order of the most Workers;
        // the sections are now four strategies and two checks.
        let mut more = workers.clone();

        more.push(worker(&[
            ("strategy", "beta first"),
            ("check", "alpha second"),
        ]));

        let r = readings(&more);

        assert_eq!(order(&r, &[0, 1]), [1, 0]);
        assert_eq!(section(&r, &[0, 1], &FIXED).as_deref(), Some("strategy"));

        // A tie of names goes to the name with the most fixed phrases, whatever the alphabet
        // says, and then to the alphabetically first.
        let tied = |a: &str, b: &str| {
            let even = vec![
                worker(&[(a, "alpha a"), (b, "beta b")]),
                worker(&[(b, "alpha a"), (a, "beta b")]),
            ];

            section(&readings(&even), &[0, 1], &FIXED)
        };

        assert_eq!(tied("check", "strategy").as_deref(), Some("strategy"));
        assert_eq!(tied("strategy", "check").as_deref(), Some("strategy"));
        assert_eq!(tied("style", "check").as_deref(), Some("check"));
        assert_eq!(tied("role", "format").as_deref(), Some("format"));
        assert_eq!(tied("format", "role").as_deref(), Some("format"));

        // The most often still comes first: two checks and a strategy give "check".
        let more_checks = vec![
            worker(&[("check", "alpha a"), ("check", "beta b")]),
            worker(&[("strategy", "alpha a"), ("style", "beta b")]),
        ];

        assert_eq!(
            section(&readings(&more_checks), &[0, 1], &FIXED).as_deref(),
            Some("check")
        );
    }

    #[test]
    fn arm_l_coins_the_largest_saving_and_a_tie_goes_to_the_smaller_set() {
        // A Worker with alpha (5 words), beta (5) and gamma (4) alone: {0, 1} saves
        // 5 + 5 − 4 − 4 = 2, {0, 1, 2} saves 5 + 5 + 4 − 6 − 6 = 2, {0, 2} and {1, 2} save 1.
        // A second best member holds the words, so none retires.
        let c = cascade(vec![worker(&[
            ("strategy", "alpha a b c d"),
            ("strategy", "beta a b c d"),
            ("check", "gamma a b c"),
        ])]);
        let holder = cascade(vec![worker(&[
            ("style", "Use alpha, then beta."),
            ("check", "Use alpha, then beta, then gamma."),
        ])]);
        let best = [&c, &holder];
        let mut lexicon = Lexicon::default();
        let first = lexicon.step(&step(20, &best, LexiconConfig::arm_l()));

        assert_eq!(first.coined, Some(0));
        assert!(first.retired.is_empty());
        assert_eq!(
            lexicon.words[0],
            Word {
                phrase: "Use alpha, then beta.".to_string(),
                meaning: vec![0, 1],
                section: "strategy".to_string(),
                saving: 2,
                generation: 20,
                retired: None,
                unheld: false,
            }
        );

        // A second step on the unchanged archive coins no second word for {0, 1}: the next
        // best set is {0, 1, 2}, its sections two strategies and a check.
        let second = lexicon.step(&step(40, &best, LexiconConfig::arm_l()));

        assert_eq!(second.coined, Some(1));
        assert_eq!(lexicon.words[1].meaning, [0, 1, 2]);
        assert_eq!(lexicon.words[1].phrase, "Use alpha, then beta, then gamma.");
        assert_eq!(lexicon.words[1].saving, 2);
        assert_eq!(lexicon.words[1].section, "strategy");

        // Then {0, 2}, which ties with {1, 2} and is the smaller list. Its sections are a
        // strategy and a check, a tie, which goes to the name with more fixed phrases.
        lexicon.step(&step(60, &best, LexiconConfig::arm_l()));
        assert_eq!(lexicon.words[2].meaning, [0, 2]);
        assert_eq!(lexicon.words[2].phrase, "Use alpha, then gamma.");
        assert_eq!(lexicon.words[2].section, "strategy");
        assert_eq!(lexicon.active().count(), 3);
        assert_eq!(lexicon.holding(&best), 1);

        // A Worker without a family alone coins nothing.
        let plain = cascade(vec![worker(&[("strategy", "alpha beta gamma")])]);
        let mut none = Lexicon::default();

        assert_eq!(
            none.step(&step(20, &[&plain], LexiconConfig::arm_l())),
            Coining::default()
        );
        assert!(none.words.is_empty());
    }

    #[test]
    fn the_cap_holds_and_a_word_retires_after_two_steps_unheld() {
        let c = cascade(vec![worker(&[
            ("strategy", "alpha a b c d"),
            ("strategy", "beta a b c d"),
        ])]);
        let holder = cascade(vec![worker(&[("strategy", "Use alpha, then beta.")])]);
        let with_holder = [&c, &holder];
        let one = LexiconConfig {
            max_words: 1,
            ..LexiconConfig::arm_l()
        };
        let mut lexicon = Lexicon::default();

        assert_eq!(lexicon.step(&step(20, &[&c], one)).coined, Some(0));

        // The cap: a full pool coins nothing, although {0, 1, 2} and {0, 2} are free; and a
        // word that a best member holds stays.
        for g in [40, 60, 80] {
            assert_eq!(
                lexicon.step(&step(g, &with_holder, one)),
                Coining::default()
            );
        }

        assert_eq!(lexicon.holding(&with_holder), 1);
        assert_eq!(lexicon.holding(&[&c]), 0);

        // Unheld at one step, held at the next, unheld again: still in the pool.
        assert_eq!(lexicon.step(&step(100, &[&c], one)), Coining::default());
        assert!(lexicon.words[0].unheld);
        assert_eq!(
            lexicon.step(&step(120, &with_holder, one)),
            Coining::default()
        );
        assert!(!lexicon.words[0].unheld);
        assert_eq!(lexicon.step(&step(140, &[&c], one)), Coining::default());

        // Unheld at the second step in a row: it retires. Retirement comes first, so the
        // pool has room at the same step, and {0, 1} is a candidate again.
        let retired = lexicon.step(&step(160, &[&c], one));

        assert_eq!(retired.retired, [0]);
        assert_eq!(retired.coined, Some(1));
        assert_eq!(lexicon.words[0].retired, Some(160));
        assert_eq!(lexicon.words[1].meaning, [0, 1]);
        assert_eq!(lexicon.words[1].generation, 160);
        assert_eq!(lexicon.retired(), 1);
        assert_eq!(lexicon.active().count(), 1);

        // A word coined at a step is first read at the next one: it lives two steps unheld.
        assert_eq!(lexicon.step(&step(180, &[&c], one)), Coining::default());
        assert_eq!(lexicon.step(&step(200, &[&c], one)).retired, [1]);

        // A retired word still counts for the members that hold one in the state line.
        let gone = Lexicon {
            words: vec![word("Use alpha, then beta.", "strategy", Some(40))],
        };

        assert_eq!(gone.holding(&with_holder), 1);
        assert_eq!(gone.active().count(), 0);
    }

    #[test]
    fn arm_r_coins_when_l_would_and_draws_its_set_from_its_own_stream() {
        // Each pair saves 9 + 9 − 4 − 4 = 10 and the triple 9 + 9 + 9 − 6 − 6 = 15: L's rule
        // picks the triple while it is free and a pair while the triple has a word, and the
        // words retire two steps after their coining, as no Worker holds one.
        let c = cascade(vec![worker(&[
            ("strategy", "alpha a b c d e f g h"),
            ("residual", "beta a b c d e f g h"),
            ("check", "gamma a b c d e f g h"),
        ])]);
        let best = [&c];
        let mut l = Lexicon::default();
        let mut r = Lexicon::default();
        let mut sizes = Vec::new();

        for g in [20, 40, 60, 80, 100, 120] {
            let s = Step {
                draw_seed: 1000 + g as u64,
                ..step(g, &best, LexiconConfig::arm_r())
            };
            // L's rule on R's own Lexicon, as R applies it: R coins exactly when it coins,
            // and a set of the same size under the same section name.
            let mut shadow = r.clone();
            let rule = shadow.step(&Step {
                config: LexiconConfig::arm_l(),
                ..s
            });
            // The sets with a word in the pool once the step has retired its words.
            let taken: Vec<Vec<usize>> = shadow
                .words
                .iter()
                .enumerate()
                .filter(|(i, w)| w.retired.is_none() && Some(*i) != rule.coined)
                .map(|(_, w)| w.meaning.clone())
                .collect();
            let did = r.step(&s);

            assert_eq!(did.retired, rule.retired, "generation {g}");
            assert_eq!(
                did.coined.is_some(),
                rule.coined.is_some(),
                "generation {g}"
            );

            if let (Some(i), Some(j)) = (did.coined, rule.coined) {
                let (word, want) = (&r.words[i], &shadow.words[j]);

                assert_eq!(word.meaning.len(), want.meaning.len(), "generation {g}");
                assert_eq!(word.section, want.section, "generation {g}");
                assert_eq!(word.generation, g);
                sizes.push(word.meaning.len());

                // The set is the first draw of its own stream among the free sets of that
                // size, in lexicographic order; the phrase names them in index order, and the
                // saving is that of the set.
                let sized: Vec<Vec<usize>> = meanings(3)
                    .into_iter()
                    .filter(|m| m.len() == want.meaning.len() && !taken.contains(m))
                    .collect();
                let mut stream = ChaCha8Rng::seed_from_u64(1000 + g as u64);

                assert_eq!(word.meaning, sized[stream.random_range(0..sized.len())]);
                assert_eq!(word.phrase, phrase(&TERMS, &word.meaning));
                assert_eq!(
                    word.saving,
                    saving(
                        &readings(&c.stages[0].workers),
                        &word.meaning,
                        words(&word.phrase)
                    )
                );
            }

            l.step(&step(g, &best, LexiconConfig::arm_l()));
        }

        // R coined at every step, a triple and a pair in turn, and so did L.
        assert_eq!(sizes, [3, 2, 3, 2, 3, 2], "{:?}", r.words);
        assert_eq!(
            l.words.iter().map(|w| w.meaning.len()).collect::<Vec<_>>(),
            sizes
        );

        // Where L's rule picks a pair, another seed draws another pair; L draws nothing,
        // whatever the seed. {0, 1} saves 10 here, {0, 1, 2} 9 and the other pairs 4.
        let pairs = cascade(vec![worker(&[
            ("strategy", "alpha a b c d e f g h"),
            ("residual", "beta a b c d e f g h"),
            ("check", "gamma a b"),
        ])]);
        let pairs = [&pairs];
        let mut differ = false;

        for seed in 0..32 {
            let s = Step {
                draw_seed: seed,
                ..step(20, &pairs, LexiconConfig::arm_r())
            };
            let (mut a, mut b, mut plain) =
                (Lexicon::default(), Lexicon::default(), Lexicon::default());

            a.step(&s);
            b.step(&Step { draw_seed: 0, ..s });
            plain.step(&Step {
                config: LexiconConfig::arm_l(),
                ..s
            });
            differ |= a.words[0].meaning != b.words[0].meaning;
            assert_eq!(plain.words[0].meaning, [0, 1]);
        }

        assert!(differ);
    }

    #[test]
    fn the_pool_appends_the_words_in_the_pool_in_coining_order() {
        let base = [("role", "r"), ("strategy", "s")];

        assert_eq!(Lexicon::default().pool(&base), base);

        let lexicon = Lexicon {
            words: vec![
                word("first", "strategy", None),
                word("gone", "check", Some(40)),
                word("second", "residual", None),
            ],
        };

        assert_eq!(
            lexicon.pool(&base),
            [
                ("role", "r"),
                ("strategy", "s"),
                ("strategy", "first"),
                ("residual", "second")
            ]
        );
        assert_eq!(lexicon.retired(), 1);
    }

    #[test]
    fn the_brevity_case_holds_the_median_rule() {
        // The lower median of [3, 5, 8, 9] is 5; of [2, 4, 4, 6, 7] it is 4.
        assert_eq!(brevity_case(&[9, 3, 8, 5]), [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(brevity_case(&[4, 7, 2, 6, 4]), [1.0, 0.0, 1.0, 0.0, 1.0]);
        // Ties at the median all pass; one member always passes.
        assert_eq!(brevity_case(&[5, 5, 5]), [1.0, 1.0, 1.0]);
        assert_eq!(brevity_case(&[12]), [1.0]);
        assert!(brevity_case(&[]).is_empty());

        // At least half of the members pass.
        for n in 1..40u64 {
            let words: Vec<u64> = (0..n).map(|i| (i * 7919) % 23).collect();
            let passed = brevity_case(&words).iter().filter(|&&x| x == 1.0).count() as u64;

            assert!(2 * passed >= n, "{words:?}");
        }
    }

    #[test]
    fn a_drawn_brevity_case_drops_the_longer_members_unless_none_passes() {
        // The case as the only case, so it is always drawn: members 0 and 2 are at most the
        // lower median of [10, 30, 20, 40], which is 20.
        let column = brevity_case(&[10, 30, 20, 40]);
        let scores: Vec<Vec<f64>> = column.iter().map(|&x| vec![x]).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(3);

        assert_eq!(column, [1.0, 0.0, 1.0, 0.0]);

        for epsilon in [0.0, 0.5, 0.99] {
            // Some remaining member passes: every member above the median drops out.
            for pool in [vec![0, 1, 2, 3], vec![1, 2], vec![0, 3]] {
                for _ in 0..50 {
                    let pick = lexicase_pick(&scores, &pool, epsilon, &mut rng).expect("pick");

                    assert_eq!(column[pick], 1.0, "{pool:?}, epsilon {epsilon}");
                }
            }

            // No remaining member passes: they all stay, so each can be drawn.
            let mut seen = [false; 4];

            for _ in 0..200 {
                seen[lexicase_pick(&scores, &[1, 3], epsilon, &mut rng).expect("pick")] = true;
            }

            assert_eq!(seen, [false, true, false, true], "epsilon {epsilon}");
        }

        // With an epsilon of 1 the case would remove no one, which is why such a Run is
        // refused.
        let mut seen = [false; 4];

        for _ in 0..400 {
            seen[lexicase_pick(&scores, &[0, 1, 2, 3], 1.0, &mut rng).expect("pick")] = true;
        }

        assert_eq!(seen, [true; 4]);
    }
}
