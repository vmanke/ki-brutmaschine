//! The analysis of the step 2 comparison (ticket 10 of `.scratch/seele-und-kappa/`, the draft
//! pre-registration docs/PREREG_STEP2_SOUL.md, spec section 17): does a Soul of complex phases
//! with an inherited Coupling strength make the breeder of step 1 pass sooner than (claim V)
//! the same Soul read as signs, and (claim K) the same Soul with κ fixed at 0?
//!
//! It reads finished circuit runs of `brut-breed`, three per seed: arm S (`--soul s`), arm V
//! (`--soul v`) and arm K (`--soul k`), each with the settings of arm A of step 1
//! (`--domain circuits --search 33 --select 22 --children 8`, every other setting at its
//! default). The pilot adds two arms for description only: arm E, which is arm S with
//! `strengths: equal`, and arm N, which is arm A of step 1 without a Soul. Each run must be
//! finished and must have the settings of its arm in everything but the seed, the generations
//! and the call cap. All runs must share their generations and their call cap.
//!
//! From `rediscovery.json` it reads the pass time T of step 1 (docs/PREREG_STEP1_CIRCUITS.md,
//! section 7): the first generation g in which the bred Cascades of generations 1 to g solve
//! more than half of the targets the seed Cascades miss. T is never when no generation up to G
//! qualifies. Per seed, Y_V is "S first" when T_S < T_V, "V first" when T_V < T_S, and a tie
//! otherwise, also when neither run passes. Y_K is the same with T_K.
//!
//! The rule [Q74 a] reads the triples in seed order. The evidence values E_V and E_K both
//! start at the value 1. E_V is multiplied by a = 2p₁ when Y_V is "S first" and by
//! b = 2(1 − p₁) when it is "V first"; a tie leaves it unchanged. E_K is updated in the same
//! way by Y_K. A claim is proven at the first triple with E ≥ 200 and fails at the first triple
//! with E ≤ 1/20. Each verdict is final at its first crossing. The rule stops after the first
//! triple at which both claims are decided, and at the latest after N_max triples. A claim
//! still open then is undecided. Every decision is made in integers.
//!
//! It makes a claim only for the counted seeds 1002, 1003, ... in this order and without gaps,
//! at most N_max of them, all run with G generations and the call cap M, none ended at the cap.
//! On the pilot seeds 901 to 923 it gives every number and makes no claim. It refuses a seed
//! that is neither, a gap in the list, a mixture of pilot and counted seeds, a seed without a
//! run in each of the arms S, V and K, and more than N_max triples. Gate G1 asks that the seed
//! Cascades solve exactly swap and cs in every counted run; a counted triple up to the stop
//! that fails it voids the experiment.
//!
//! The descriptive lines carry no verdict: the pass rates per arm, D at the last generation,
//! the first generation per target, the divergence generation of each pair of arms, the vote
//! tallies (complex, flipped and tied votes, their coherence, `sim_decided`), the trajectories
//! of the `soul` group of the `state` lines, and the calls and wall time per run. The
//! divergence generation of two runs of a seed is the first generation whose newborns differ,
//! member by member in birth order, in gene id, operator, parents (by gene id), answers, solved
//! count or graded value. The `born` events differ earlier, since they carry κ and the vote
//! tallies.
//!
//! G, M, N_max and p₁ are the pilot's proposals (ticket 11) until the owner approves them.
//!
//! It reads each file of a run once, from the start to the end, so a file may also be a named
//! pipe. Run `brut-breed --out <dir> --verify` on every directory first: this program recounts
//! each run from its own files, but it does not check the hash chain.
//!
//! Usage:
//!   cargo run --release --example soul_comparison -- --s <dir>... --v <dir>... --k <dir>...
//!     [--e <dir>...] [--n <dir>...] [--preset <file>]
//!
//! --preset reads the `soul` settings of arm S from a JSON file, for pilot batches whose
//! placeholders differ from the presets of `--soul`. Arms V, K and E follow from it as the
//! presets follow from arm S. With --preset no claim is made.
#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde::de::DeserializeOwned;

use brutmaschine::cascade::arena::{BreedConfig, Event, Report, VoteTally};
use brutmaschine::cascade::circuit::{Rediscovery, verified};
use brutmaschine::cascade::domain::{CircuitConfig, DomainConfig};
use brutmaschine::cascade::observe::SoulState;
use brutmaschine::cascade::soul::{Readout, SoulConfig, Strengths};
use brutmaschine::cascade::variation::Op;
use brutmaschine::util::sha256_hex;

/// G: the generations of every counted run, and the ceiling of the pass time. The pilot's
/// proposal (ticket 11); step 1 used 1000.
const GENERATIONS: usize = 1000;

/// M: the call cap of every counted run, which no counted run may reach. The pilot's proposal;
/// step 1 used 13 000 000.
const MAX_CALLS: u64 = 13_000_000;

/// N_max: the most triples the experiment counts, one per counted seed. The pilot's proposal;
/// at most 120.
const MAX_TRIPLES: usize = 120;

/// p₁ as numerator and denominator: the rate of "S first" among untied triples that the
/// evidence weighs against 1/2. The pilot's proposal; it lies strictly between 1/2 and 1.
const P1: (u32, u32) = (3, 4);

/// A claim is proven when its evidence first reaches PROOF [Q74 a].
const PROOF: u32 = 200;

/// A claim fails when its evidence first falls to 1 / FAILURE.
const FAILURE: u32 = 20;

/// The first counted seed. The counted seeds are FIRST_SEED, FIRST_SEED + 1, ... in this
/// order, MAX_TRIPLES of them. Seed 1001 is not on the list: a check of the tools in counted
/// mode ran three short Runs on it before the freeze (ticket 10).
const FIRST_SEED: u64 = 1002;

/// The pilot seeds of step 2 (ticket 11), which never count.
const PILOT_SEEDS: RangeInclusive<u64> = 901..=923;

/// Gate G1: the seed Cascades of every counted run solve exactly these targets.
const SEED_TARGETS: [&str; 2] = ["cs", "swap"];

/// How a run ends when it completes every generation.
const ALL_GENERATIONS: &str = "all generations done";

/// How a run ends when the call budget stops it first.
const CALL_BUDGET: &str = "call budget reached";

/// The generations between two rows of the trajectories.
const TRAJECTORY_STEP: usize = 100;

/// The counted seeds in their fixed order. Triple i runs the i-th.
fn counted_seeds() -> impl Iterator<Item = u64> {
    (FIRST_SEED..).take(MAX_TRIPLES)
}

fn main() {
    match run(std::env::args().skip(1).collect()) {
        Ok(text) => print!("{text}"),
        Err(e) => {
            eprintln!("soul_comparison: {e}");
            std::process::exit(1);
        }
    }
}

/// The arms of the comparison (spec 3 and 12.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Arm {
    /// The Soul with complex phases.
    S,
    /// The same Soul read as signs (German Vorzeichen).
    V,
    /// The same Soul with κ fixed at 0.
    K,
    /// Arm S with equal Family strengths; the pilot only, for description.
    E,
    /// No Soul: arm A of step 1; the pilot only, for description.
    N,
}

impl Arm {
    /// Every arm, in the order of the tables.
    const ALL: [Arm; 5] = [Arm::S, Arm::V, Arm::K, Arm::E, Arm::N];

    /// The arms of a triple, which the rule compares.
    const TRIPLE: [Arm; 3] = [Arm::S, Arm::V, Arm::K];

    /// The option that names the arm's run directories.
    fn flag(self) -> &'static str {
        match self {
            Arm::S => "--s",
            Arm::V => "--v",
            Arm::K => "--k",
            Arm::E => "--e",
            Arm::N => "--n",
        }
    }
}

impl fmt::Display for Arm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Arm::S => "S",
            Arm::V => "V",
            Arm::K => "K",
            Arm::E => "E",
            Arm::N => "N",
        })
    }
}

/// The `soul` settings of `arm`, derived from those of arm S as the presets of `--soul` derive
/// them: arm V reads signs, arm K fixes κ at 0 from the seeds on, arm E gives every family the
/// strength 1, and arm N has no Soul.
fn arm_soul(base: &SoulConfig, arm: Arm) -> Option<SoulConfig> {
    let base = base.clone();

    match arm {
        Arm::S => Some(base),
        Arm::V => Some(SoulConfig {
            readout: Readout::Real,
            ..base
        }),
        Arm::K => Some(SoulConfig {
            kappa_seed: 0,
            kappa_fixed: true,
            ..base
        }),
        Arm::E => Some(SoulConfig {
            strengths: Strengths::Equal,
            ..base
        }),
        Arm::N => None,
    }
}

/// The settings a run of `arm` must have, given its seed, generations and call cap: the
/// circuit domain on every verified target, 33 search and 22 select tasks, 8 children per
/// generation, the `soul` settings of the arm, and every other setting at its default.
fn arm_config(
    base: &SoulConfig,
    arm: Arm,
    seed: u64,
    generations: usize,
    max_calls: u64,
) -> BreedConfig {
    BreedConfig {
        seed,
        generations,
        children_per_generation: 8,
        search_puzzles: 33,
        select_puzzles: 22,
        max_calls,
        domain: DomainConfig::Circuits(CircuitConfig::default()),
        soul: arm_soul(base, arm),
        ..BreedConfig::default()
    }
}

/// An exact fraction num / den with den > 0, in lowest terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Ratio {
    num: u128,
    den: u128,
}

fn gcd(a: u128, b: u128) -> u128 {
    if b == 0 { a } else { gcd(b, a % b) }
}

impl Ratio {
    fn new(num: u128, den: u128) -> Ratio {
        let g = gcd(num, den).max(1);

        Ratio {
            num: num / g,
            den: den / g,
        }
    }

    fn value(self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

impl fmt::Display for Ratio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{} = {:.4}", self.num, self.den, self.value())
    }
}

/// An exact natural number of any size, for the evidence values: 3^120 has 191 bits, more than
/// a u128 holds. Its digits are in base 10^9, the lowest first, and the top digit is not 0.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Natural(Vec<u32>);

/// The base of the digits of a Natural.
const DIGIT_BASE: u64 = 1_000_000_000;

impl Natural {
    /// The product of base^exp over the pairs, each base at least 1; 1 for no pairs.
    fn powers(pairs: &[(u32, u32)]) -> Natural {
        pairs
            .iter()
            .fold(Natural(vec![1]), |product, &(base, exp)| {
                (0..exp).fold(product, |p, _| p.times(base))
            })
    }

    /// The product with a factor of at least 1, so that the top digit stays above 0.
    fn times(self, factor: u32) -> Natural {
        let mut digits = self.0;
        let mut carry = 0;

        for digit in &mut digits {
            let x = u64::from(*digit) * u64::from(factor) + carry;

            *digit = (x % DIGIT_BASE) as u32;
            carry = x / DIGIT_BASE;
        }

        while carry > 0 {
            digits.push((carry % DIGIT_BASE) as u32);
            carry /= DIGIT_BASE;
        }

        Natural(digits)
    }
}

impl Ord for Natural {
    /// With no 0 on top, more digits make a larger number; equal lengths compare from the top
    /// digit down.
    fn cmp(&self, other: &Natural) -> Ordering {
        self.0
            .len()
            .cmp(&other.0.len())
            .then_with(|| self.0.iter().rev().cmp(other.0.iter().rev()))
    }
}

impl PartialOrd for Natural {
    fn partial_cmp(&self, other: &Natural) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Natural {
    /// In decimal: the top digit as it is, every lower one with all nine places.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut digits = self.0.iter().rev();

        if let Some(top) = digits.next() {
            write!(f, "{top}")?;
        }

        digits.try_for_each(|digit| write!(f, "{digit:09}"))
    }
}

/// The factors of the evidence, each in lowest terms: a = 2p₁ when arm S passes first and
/// b = 2(1 − p₁) when the control does. They are the likelihood ratio of P(S first | untied)
/// = p₁ against 1/2. For p₁ = n / d in lowest terms, 2n and 2(d − n) share no odd factor
/// with d, so every product of the factors is again in lowest terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Factors {
    /// p₁ as numerator and denominator, as given.
    p1: (u32, u32),
    /// a as numerator and denominator.
    up: (u32, u32),
    /// b as numerator and denominator.
    down: (u32, u32),
}

/// The fraction num / den in lowest terms; both fit in a u32.
fn lowest(num: u64, den: u64) -> Result<(u32, u32), String> {
    let g = gcd(u128::from(num), u128::from(den)).max(1) as u64;
    let fit = |x: u64| u32::try_from(x).map_err(|_| format!("{x} does not fit a u32"));

    Ok((fit(num / g)?, fit(den / g)?))
}

impl Factors {
    /// The factors for p₁ = num / den, which must lie strictly between 1/2 and 1.
    fn new((num, den): (u32, u32)) -> Result<Factors, String> {
        let (n, d) = (u64::from(num), u64::from(den));

        if d == 0 || 2 * n <= d || n >= d {
            return Err(format!(
                "p1 = {num}/{den} must lie strictly between 1/2 and 1"
            ));
        }

        Ok(Factors {
            p1: (num, den),
            up: lowest(2 * n, d)?,
            down: lowest(2 * (d - n), d)?,
        })
    }
}

/// The outcome of a seed between arm S and one control, by the pass time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum First {
    /// Arm S passes first.
    S,
    /// The control passes first.
    Control,
    /// Both runs pass in the same generation, or neither passes.
    Tie,
}

/// The arm with the smaller pass time, where never comes after every generation.
fn first(t_s: Option<usize>, t_control: Option<usize>) -> First {
    match (t_s, t_control) {
        (Some(s), Some(c)) => match s.cmp(&c) {
            Ordering::Less => First::S,
            Ordering::Greater => First::Control,
            Ordering::Equal => First::Tie,
        },
        (Some(_), None) => First::S,
        (None, Some(_)) => First::Control,
        (None, None) => First::Tie,
    }
}

/// An outcome as text, with the control's letter.
fn show_first(y: First, control: Arm) -> String {
    match y {
        First::S => "S first".to_string(),
        First::Control => format!("{control} first"),
        First::Tie => "tie".to_string(),
    }
}

/// An evidence value after `wins` triples in which arm S passed first and `losses` in which
/// the control did: E = a^wins · b^losses. The default is the start, E = 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Evidence {
    wins: u32,
    losses: u32,
}

impl Evidence {
    /// E after one more triple; a tie leaves it unchanged.
    fn step(self, y: First) -> Evidence {
        match y {
            First::S => Evidence {
                wins: self.wins + 1,
                ..self
            },
            First::Control => Evidence {
                losses: self.losses + 1,
                ..self
            },
            First::Tie => self,
        }
    }

    /// The numerator of E in lowest terms.
    fn numerator(self, f: Factors) -> Natural {
        Natural::powers(&[(f.up.0, self.wins), (f.down.0, self.losses)])
    }

    /// The denominator of E in lowest terms.
    fn denominator(self, f: Factors) -> Natural {
        Natural::powers(&[(f.up.1, self.wins), (f.down.1, self.losses)])
    }

    /// E >= PROOF, in integers.
    fn proven(self, f: Factors) -> bool {
        self.numerator(f) >= self.denominator(f).times(PROOF)
    }

    /// E <= 1 / FAILURE, in integers.
    fn failed(self, f: Factors) -> bool {
        self.numerator(f).times(FAILURE) <= self.denominator(f)
    }

    /// E as a float, only to be read; no decision uses it.
    fn value(self, f: Factors) -> f64 {
        let ratio = |(num, den): (u32, u32)| f64::from(num) / f64::from(den);

        ratio(f.up).powi(self.wins as i32) * ratio(f.down).powi(self.losses as i32)
    }

    /// E as an exact fraction in lowest terms, and its value.
    fn show(self, f: Factors) -> String {
        format!(
            "{}/{} = {:.4}",
            self.numerator(f),
            self.denominator(f),
            self.value(f)
        )
    }
}

/// D of one run: of the `open` targets the seed Cascades miss, a bred Cascade solves `hit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Share {
    hit: usize,
    open: usize,
}

impl Share {
    /// D as a fraction; None when the seed Cascades solve every target.
    fn ratio(self) -> Option<Ratio> {
        (self.open > 0).then(|| Ratio::new(self.hit as u128, self.open as u128))
    }

    /// D > 1/2; false when D is undefined.
    fn above_half(self) -> bool {
        self.open > 0 && 2 * self.hit > self.open
    }
}

/// D as text.
fn show_share(share: Share) -> String {
    match share.ratio() {
        Some(_) => format!(
            "{}/{} = {:.3}",
            share.hit,
            share.open,
            share.hit as f64 / share.open as f64
        ),
        None => "undefined".to_string(),
    }
}

/// One run counted under the pre-registered rule: the targets its seed Cascades solve, |L|
/// and D.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Counted {
    seeds: BTreeSet<String>,
    library: usize,
    share: Share,
}

fn count(found: &Rediscovery) -> Counted {
    let mut seeds = BTreeSet::new();
    let mut hit = 0;

    for t in &found.targets {
        if t.reached_by_seed {
            seeds.insert(t.target.clone());
        } else if t.reached_by_bred {
            hit += 1;
        }
    }

    let library = found.targets.len();

    Counted {
        share: Share {
            hit,
            open: library - seeds.len(),
        },
        seeds,
        library,
    }
}

/// Error unless the counts of `found` agree with its own target list, and its targets are the
/// verified library targets in library order: L of step 1. A target must also have a first
/// generation exactly when a bred Cascade solved it, since the pass time is read from these.
fn check_counts(found: &Rediscovery) -> Result<(), String> {
    let names: Vec<&str> = found.targets.iter().map(|t| t.target.as_str()).collect();
    let library: Vec<&str> = verified().iter().map(|t| t.name).collect();
    let counted = count(found);

    if names != library {
        return Err(format!(
            "its targets are {names:?}, not the verified library {library:?}"
        ));
    }

    // D as a float, the same number whether or not the fraction is in lowest terms: one
    // correctly rounded division of the same rational.
    let agrees = found.library == counted.library
        && found.reached_by_seeds == counted.seeds.len()
        && found.reached_by_bred_only == counted.share.hit
        && found.share == counted.share.ratio().map(Ratio::value)
        && found.calibrated == (3 * counted.seeds.len() <= counted.library)
        && found.passes
            == (3 * counted.seeds.len() <= counted.library && counted.share.above_half())
        && found
            .targets
            .iter()
            .all(|t| t.first_generation.is_some() == t.reached_by_bred);

    if !agrees {
        return Err("its counts disagree with its own target list".to_string());
    }

    Ok(())
}

/// The pass time T of a run: the first generation g with 2 |R_bred(<= g) \ R_seed| >
/// |L \ R_seed|, that is D > 1/2. It is the k-th smallest first generation of the targets that
/// only bred Cascades solve, with k = |L \ R_seed| / 2 + 1, rounded down. None when fewer than
/// k such targets exist, as when the seed Cascades solve every target.
fn pass_time(found: &Rediscovery) -> Option<usize> {
    let open = found.targets.iter().filter(|t| !t.reached_by_seed).count();
    let mut firsts: Vec<usize> = found
        .targets
        .iter()
        .filter(|t| t.reached_by_bred && !t.reached_by_seed)
        .filter_map(|t| t.first_generation)
        .collect();

    firsts.sort_unstable();

    // The k-th smallest has the index k - 1 = open / 2.
    firsts.get(open / 2).copied()
}

/// A pass time within the ceiling G: a run that passes only later has not passed.
fn within(t: Option<usize>) -> Option<usize> {
    t.filter(|&g| g <= GENERATIONS)
}

/// A pass time as text: the generation, or never.
fn show_time(t: Option<usize>) -> String {
    t.map_or("never".to_string(), |g| g.to_string())
}

/// The three runs of one seed that the rule compares.
#[derive(Clone, Debug)]
struct Triple {
    seed: u64,
    /// The runs of arms S, V and K, in this order.
    runs: [Rediscovery; 3],
}

/// One seed as the rule reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Row {
    seed: u64,
    library: usize,
    /// R_seed, the same in the three arms, since generation 0 does not depend on the Soul.
    seeds: BTreeSet<String>,
    /// D of arms S, V and K.
    shares: [Share; 3],
    /// The pass times of arms S, V and K within the ceiling, None for never.
    times: [Option<usize>; 3],
}

impl Row {
    /// Gate G1: the seed Cascades solve exactly swap and cs.
    fn g1(&self) -> bool {
        self.seeds.iter().map(String::as_str).eq(SEED_TARGETS)
    }

    /// Y_V: arm S against arm V.
    fn y_v(&self) -> First {
        first(self.times[0], self.times[1])
    }

    /// Y_K: arm S against arm K.
    fn y_k(&self) -> First {
        first(self.times[0], self.times[2])
    }
}

/// The rows of the triples, in their order. A triple whose arms disagree on the targets the
/// seed Cascades solve is refused.
fn rows(triples: &[Triple]) -> Result<Vec<Row>, String> {
    triples
        .iter()
        .map(|t| {
            let counted = t.runs.clone().map(|found| count(&found));
            let [s, v, k] = &counted;

            if s.seeds != v.seeds || s.seeds != k.seeds || s.library != v.library || s.library != k.library {
                return Err(format!(
                    "seed {}: the arms disagree on the targets the seed Cascades solve, so the runs are not a triple",
                    t.seed
                ));
            }

            Ok(Row {
                seed: t.seed,
                library: s.library,
                seeds: s.seeds.clone(),
                shares: counted.clone().map(|c| c.share),
                times: t.runs.clone().map(|found| within(pass_time(&found))),
            })
        })
        .collect()
}

/// The verdict on one claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    /// Its E reached PROOF at this triple.
    Proven(usize),
    /// Its E fell to 1 / FAILURE at this triple.
    Failed(usize),
    /// The rule stopped after MAX_TRIPLES triples before its E crossed: no claim.
    Undecided,
    /// The rule goes on, and its E has not crossed.
    Open,
}

impl Verdict {
    /// The verdict after a triple at which its E is `e`. An open claim is decided at its first
    /// crossing, and that decision is final.
    fn update(self, e: Evidence, f: Factors, triple: usize) -> Verdict {
        match self {
            Verdict::Open if e.proven(f) => Verdict::Proven(triple),
            Verdict::Open if e.failed(f) => Verdict::Failed(triple),
            decided => decided,
        }
    }

    fn decided(self) -> bool {
        matches!(self, Verdict::Proven(_) | Verdict::Failed(_))
    }

    /// The verdict at the stop: a claim still open is undecided.
    fn at_stop(self) -> Verdict {
        match self {
            Verdict::Open => Verdict::Undecided,
            decided => decided,
        }
    }
}

/// The rule over the triples in seed order.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Sequential {
    /// E_V and E_K after each counted triple.
    evidence: Vec<(Evidence, Evidence)>,
    /// The triple after which the rule stops; None while it goes on.
    stop: Option<usize>,
    /// Claim V, decided by E_V.
    v: Verdict,
    /// Claim K, decided by E_K.
    k: Verdict,
}

impl Sequential {
    /// The triples that count: those up to the stop, or all while the rule goes on.
    fn counted(&self) -> usize {
        self.evidence.len()
    }
}

/// The rule over `rows` in seed order (section 8). E_V and E_K start at 1 and take each triple
/// in turn, also after their own claim is decided. The rule stops after the first triple at
/// which both claims are decided, and at the latest after MAX_TRIPLES triples. The rows after
/// the stop do not count.
fn sequential(rows: &[Row], f: Factors) -> Sequential {
    let (mut e_v, mut e_k) = (Evidence::default(), Evidence::default());
    let (mut v, mut k) = (Verdict::Open, Verdict::Open);
    let mut evidence = Vec::new();

    for (triple, row) in (1..).zip(rows) {
        e_v = e_v.step(row.y_v());
        e_k = e_k.step(row.y_k());
        evidence.push((e_v, e_k));
        v = v.update(e_v, f, triple);
        k = k.update(e_k, f, triple);

        if (v.decided() && k.decided()) || triple == MAX_TRIPLES {
            return Sequential {
                evidence,
                stop: Some(triple),
                v: v.at_stop(),
                k: k.at_stop(),
            };
        }
    }

    Sequential {
        evidence,
        stop: None,
        v,
        k,
    }
}

/// Which seeds the runs hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SeedSet {
    /// Pilot seeds of step 2, which never count.
    Pilot,
    /// The first counted seeds of the list.
    Counted,
}

/// The kind of the seeds, given in increasing order. Pilot seeds are accepted, with every claim
/// withheld. Otherwise the seeds must be the first seeds of the counted list, at most
/// MAX_TRIPLES of them. A seed off the list, a gap, more than MAX_TRIPLES seeds and a mixture of
/// pilot and counted seeds are refused.
fn check_seeds(seeds: &[u64]) -> Result<SeedSet, String> {
    if seeds.is_empty() {
        return Err("no seed triples".to_string());
    }

    let pilot: Vec<u64> = seeds
        .iter()
        .copied()
        .filter(|s| PILOT_SEEDS.contains(s))
        .collect();

    if pilot.len() == seeds.len() {
        return Ok(SeedSet::Pilot);
    }

    if !pilot.is_empty() {
        return Err(format!(
            "the pilot seeds {pilot:?} never enter the analysis of the counted seeds"
        ));
    }

    if seeds.len() > MAX_TRIPLES {
        return Err(format!(
            "{} seed triples, more than the {MAX_TRIPLES} the pre-registration allows",
            seeds.len()
        ));
    }

    let last = FIRST_SEED + MAX_TRIPLES as u64 - 1;

    for (triple, (&seed, expected)) in (1..).zip(seeds.iter().zip(counted_seeds())) {
        if seed == expected {
            continue;
        }

        if !(FIRST_SEED..=last).contains(&seed) {
            return Err(format!(
                "seed {seed} is not on the list of the counted seeds, {FIRST_SEED} to {last}, and not a pilot seed"
            ));
        }

        return Err(format!(
            "the counted seeds run {FIRST_SEED}, {}, ... in this order without gaps; triple {triple} has seed {seed}, not {expected}",
            FIRST_SEED + 1
        ));
    }

    Ok(SeedSet::Counted)
}

/// What the runs share besides their results.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Context {
    generations: usize,
    max_calls: u64,
    /// Runs of arms S, V and K that ended at the call cap: seed, arm and generations
    /// completed.
    capped: Vec<(u64, Arm, usize)>,
    /// True when the `soul` settings come from --preset, not from the presets of `--soul`.
    preset: bool,
}

impl Default for Context {
    /// The context of the pre-registered runs: their generations and call cap, no run at the
    /// cap, and the presets of `--soul`.
    fn default() -> Self {
        Context {
            generations: GENERATIONS,
            max_calls: MAX_CALLS,
            capped: Vec::new(),
            preset: false,
        }
    }
}

/// Whether the runs may support a claim.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Claim {
    /// The counted seeds, through gate G1 up to the stop: the rule decides.
    Made,
    /// The counted seeds, but these seeds up to the stop fail gate G1: void.
    Void(Vec<u64>),
    /// No claim, and why.
    Withheld(String),
}

/// The claim the rows support under the rule `rule`. Pilot seeds, settings from --preset,
/// another length or cap than the pre-registered ones, and a run at the cap withhold it. A
/// counted triple up to the stop that fails gate G1 voids the experiment.
fn claim(context: &Context, set: SeedSet, rows: &[Row], rule: &Sequential) -> Claim {
    if set == SeedSet::Pilot {
        return Claim::Withheld(format!(
            "the seeds are pilot seeds of step 2 ({} to {}), which never count",
            PILOT_SEEDS.start(),
            PILOT_SEEDS.end()
        ));
    }

    if context.preset {
        return Claim::Withheld(
            "the soul settings come from --preset, not from the presets of --soul".to_string(),
        );
    }

    if (context.generations, context.max_calls) != (GENERATIONS, MAX_CALLS) {
        return Claim::Withheld(format!(
            "the runs have --generations {} and --max-calls {}, not the pre-registered {GENERATIONS} and {MAX_CALLS}",
            context.generations, context.max_calls
        ));
    }

    if !context.capped.is_empty() {
        return Claim::Withheld(format!(
            "{} run(s) of arms S, V and K ended at the call cap, which the pre-registered cap rules out",
            context.capped.len()
        ));
    }

    // Only the triples up to the stop count, so only they can void the experiment.
    let void: Vec<u64> = rows
        .iter()
        .take(rule.counted())
        .filter(|r| !r.g1())
        .map(|r| r.seed)
        .collect();

    if void.is_empty() {
        Claim::Made
    } else {
        Claim::Void(void)
    }
}

/// The verdict on one claim at the stop, with what it means when proven or failed.
fn verdict_line(control: Arm, verdict: Verdict, f: Factors) -> String {
    let name = format!("Claim {control}");

    match verdict {
        Verdict::Proven(t) => format!(
            "{name}: PROVEN at triple {t}. Among untied triples, arm S passes first more often than arm {control}.\n"
        ),
        Verdict::Failed(t) => format!(
            "{name}: FAILED at triple {t}. The data speak against arm S passing first in a share of p1 = {}/{} or more of the untied triples against arm {control}; they do not prove a share of 1/2 or less.\n",
            f.p1.0, f.p1.1
        ),
        Verdict::Undecided => {
            format!("{name}: UNDECIDED after {MAX_TRIPLES} triples, no claim.\n")
        }
        Verdict::Open => format!("{name}: open.\n"),
    }
}

/// The verdicts at the stop, or the next seed while the rule goes on; only the reason when the
/// runs support no claim.
fn verdict(claim: &Claim, rule: &Sequential, f: Factors) -> String {
    // While the rule goes on, fewer than MAX_TRIPLES triples count, so the list has a next seed.
    let next = counted_seeds().nth(rule.counted());

    match (claim, rule.stop, next) {
        (Claim::Void(_), _, _) => "No claim: the experiment is void.\n".to_string(),
        (Claim::Withheld(why), _, _) => format!("No claim: {why}.\n"),
        (Claim::Made, None, Some(seed)) => format!("OPEN: run seed {seed} next.\n"),
        (Claim::Made, _, _) => format!(
            "{}{}",
            verdict_line(Arm::V, rule.v, f),
            verdict_line(Arm::K, rule.k, f)
        ),
    }
}

/// Where the E of one claim stands after `n` counted triples, and where it first crossed.
fn evidence_line(name: &str, e: Evidence, verdict: Verdict, n: usize, f: Factors) -> String {
    let crossed = match verdict {
        Verdict::Proven(t) => format!("; it reached {PROOF} at triple {t}"),
        Verdict::Failed(t) => format!("; it fell to 1/{FAILURE} at triple {t}"),
        Verdict::Open => format!("; open, it has reached neither {PROOF} nor 1/{FAILURE}"),
        Verdict::Undecided => format!("; it reached neither {PROOF} nor 1/{FAILURE}"),
    };

    format!("{name} = {} after {n} triple(s){crossed}.\n", e.show(f))
}

/// The rule part of the analysis as text: gate G1 and D per seed, the rule triple by triple,
/// where E_V and E_K stand, and the verdicts.
fn render_rule(
    context: &Context,
    set: SeedSet,
    triples: &[Triple],
    f: Factors,
) -> Result<String, String> {
    if triples.is_empty() {
        return Err("no seed triples".to_string());
    }

    if triples.len() > MAX_TRIPLES {
        return Err(format!("more than {MAX_TRIPLES} seed triples"));
    }

    let rows = rows(triples)?;
    let rule = sequential(&rows, f);
    let claim = claim(context, set, &rows, &rule);
    let mut out = String::new();

    out.push_str(
        "Step 2 comparison: the analysis of docs/PREREG_STEP2_SOUL.md (claims V and K, Q74 a)\n",
    );
    out.push_str(&format!(
        "{} seed triple(s); every run: --generations {}, --max-calls {}.\n",
        triples.len(),
        context.generations,
        context.max_calls
    ));

    match &claim {
        Claim::Made => out.push_str(
            "Void conditions checked here: gate G1, the seed Cascades of every counted run solve exactly swap and cs.\n",
        ),
        Claim::Void(seeds) => out.push_str(&format!(
            "VOID: the seed Cascades of the triple(s) of seed(s) {seeds:?} do not solve exactly swap and cs (gate G1).\n"
        )),
        Claim::Withheld(why) => out.push_str(&format!("No claim: {why}.\n")),
    }

    out.push_str(
        "Checked elsewhere: integrity alarms (a run with one has no rediscovery.json and is refused here), \
         brut-breed --verify on every run, gates G2 to G4, and the recount by reference_rule.py.\n\n",
    );
    out.push_str(
        "Gate G1 and D at the last generation: a target counts when an exact answer reaches its T and CNOT reference.\n",
    );
    out.push_str(&format!(
        "{:>5}  {:>3}  {:<12}  {:<3}  {:<15}  {:<15}  {}\n",
        "seed", "|L|", "R_seed", "G1", "D_S", "D_V", "D_K"
    ));

    for row in &rows {
        let seeds = row.seeds.iter().cloned().collect::<Vec<_>>().join(", ");

        out.push_str(&format!(
            "{:>5}  {:>3}  {:<12}  {:<3}  {:<15}  {:<15}  {}\n",
            row.seed,
            row.library,
            seeds,
            if row.g1() { "yes" } else { "NO" },
            show_share(row.shares[0]),
            show_share(row.shares[1]),
            show_share(row.shares[2])
        ));
    }

    out.push_str(&format!(
        "\nThe rule, in seed order: T is the first generation with D > 1/2, never when there is none by generation {GENERATIONS}; \
         Y_V and Y_K compare T_S with T_V and T_K.\n"
    ));
    out.push_str(&format!(
        "E_V and E_K start at 1 and are multiplied by a = 2p1 = {}/{} when arm S passes first and by b = 2(1 - p1) = {}/{} when the control does, \
         with p1 = {}/{}; a tie leaves them unchanged. A claim is proven when its E first reaches {PROOF} and fails when it first falls to 1/{FAILURE}.\n",
        f.up.0, f.up.1, f.down.0, f.down.1, f.p1.0, f.p1.1
    ));
    out.push_str(&format!(
        "{:>5}  {:>5}  {:>5}  {:>5}  {:<7}  {:<7}  {:<24}  {}\n",
        "seed", "T_S", "T_V", "T_K", "Y_V", "Y_K", "E_V", "E_K"
    ));

    for (row, (e_v, e_k)) in rows.iter().zip(&rule.evidence) {
        out.push_str(&format!(
            "{:>5}  {:>5}  {:>5}  {:>5}  {:<7}  {:<7}  {:<24}  {}\n",
            row.seed,
            show_time(row.times[0]),
            show_time(row.times[1]),
            show_time(row.times[2]),
            show_first(row.y_v(), Arm::V),
            show_first(row.y_k(), Arm::K),
            e_v.show(f),
            e_k.show(f)
        ));
    }

    for row in &rows[rule.counted()..] {
        out.push_str(&format!("{:>5}  after the stop, not counted\n", row.seed));
    }

    let n = rule.counted();

    if let Some(&(e_v, e_k)) = rule.evidence.last() {
        out.push_str(&evidence_line("E_V", e_v, rule.v, n, f));
        out.push_str(&evidence_line("E_K", e_k, rule.k, n, f));
    }

    match rule.stop {
        Some(stop) => out.push_str(&format!("The rule stops after triple {stop}.\n")),
        None => out.push_str(&format!("The rule goes on after triple {n}.\n")),
    }

    out.push_str(&verdict(&claim, &rule, f));

    Ok(out)
}

/// What a newborn did, as section 12 compares two runs: what its `born` event says, but κ, the
/// Souls and the vote tallies. The parents are named by gene id, the answers by the SHA-256 of
/// their JSON, the graded value by its bits.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Newborn {
    gene_id: String,
    op: Op,
    parents: Vec<String>,
    answers: String,
    solved: usize,
    quality: Option<u64>,
}

/// What a run's event log gives the descriptive results.
#[derive(Clone, Debug, Default)]
struct LogSummary {
    /// The newborns of each generation, in birth order; the index is the generation.
    newborns: Vec<Vec<Newborn>>,
    /// The `soul` group of the `state` line of each generation, None in a run without Souls.
    souls: Vec<Option<SoulState>>,
    /// The vote tallies of all newborns, summed.
    votes: VoteTally,
    /// The first generation with a newborn whose complex vote flipped a winner.
    first_flip: Option<usize>,
    /// The first generation with a newborn for which the phases decided a simulated model's
    /// choice.
    first_sim: Option<usize>,
}

/// Reads a run's events in log order into a LogSummary.
#[derive(Default)]
struct Summariser {
    /// The gene id of every member born so far, by member id.
    gene_ids: HashMap<String, String>,
    summary: LogSummary,
}

impl Summariser {
    /// Take one event of the log.
    fn add(&mut self, event: Event) -> Result<(), String> {
        match event {
            Event::Born {
                generation,
                id,
                parents,
                op,
                cascade,
                solved,
                quality,
                answers,
                votes,
                ..
            } => {
                let gene_id = cascade.gene_id();
                let parents = parents
                    .iter()
                    .map(|p| {
                        self.gene_ids.get(p).cloned().ok_or_else(|| {
                            format!(
                                "member {id} has the parent {p}, which the log has not born before"
                            )
                        })
                    })
                    .collect::<Result<Vec<String>, String>>()?;
                let answers = serde_json::to_string(&answers).map_err(|e| e.to_string())?;
                let s = &mut self.summary;

                if s.newborns.len() <= generation {
                    s.newborns.resize(generation + 1, Vec::new());
                }

                s.newborns[generation].push(Newborn {
                    gene_id: gene_id.clone(),
                    op,
                    parents,
                    answers: sha256_hex(answers.as_bytes()),
                    solved,
                    quality: quality.map(f64::to_bits),
                });

                if let Some(v) = votes {
                    s.votes.complex += v.complex;
                    s.votes.flipped += v.flipped;
                    s.votes.tied += v.tied;
                    s.votes.coherence_sum += v.coherence_sum;
                    s.votes.sim_decided += v.sim_decided;

                    if v.flipped > 0 && s.first_flip.is_none() {
                        s.first_flip = Some(generation);
                    }

                    if v.sim_decided > 0 && s.first_sim.is_none() {
                        s.first_sim = Some(generation);
                    }
                }

                self.gene_ids.insert(id, gene_id);
            }
            Event::State(state) => {
                let s = &mut self.summary;
                let g = state.generation;

                if s.souls.len() <= g {
                    s.souls.resize(g + 1, None);
                }

                s.souls[g] = state.soul.map(|soul| *soul);
            }
            _ => {}
        }

        Ok(())
    }
}

/// The divergence generation of two runs: the first generation whose newborns differ, member
/// by member in birth order. None when every generation of both runs agrees.
fn divergence(a: &[Vec<Newborn>], b: &[Vec<Newborn>]) -> Option<usize> {
    (0..a.len().max(b.len())).find(|&g| a.get(g) != b.get(g))
}

/// One line of `events.jsonl`; only the event is read.
#[derive(Deserialize)]
struct Entry {
    event: Event,
}

/// The summary of the event log at `path`, read once, line by line.
fn read_log(path: &Path) -> Result<LogSummary, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut s = Summariser::default();

    for (i, line) in BufReader::with_capacity(1 << 20, file).lines().enumerate() {
        let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
        let entry: Entry = serde_json::from_str(&line)
            .map_err(|e| format!("{}: line {}: {e}", path.display(), i + 1))?;

        s.add(entry.event)
            .map_err(|e| format!("{}: {e}", path.display()))?;
    }

    Ok(s.summary)
}

/// The checkpoint `snapshot.json`; only whether the select phase finished is read.
#[derive(Deserialize)]
struct Checkpoint {
    snapshot: Finished,
}

#[derive(Deserialize)]
struct Finished {
    #[serde(default)]
    finished: bool,
}

/// True when the checkpoint at `path` says that the run finished; false when it is missing or
/// unreadable. It is read once, from the start to the end.
fn read_finished(path: &Path) -> bool {
    File::open(path)
        .ok()
        .and_then(|file| {
            serde_json::from_reader::<_, Checkpoint>(BufReader::with_capacity(1 << 20, file)).ok()
        })
        .is_some_and(|c| c.snapshot.finished)
}

/// One line of `timings.jsonl`.
#[derive(Deserialize)]
struct Timing {
    seconds: f64,
}

/// The wall time of a run in minutes, summed over the lines of its `timings.jsonl`; None when
/// the file is missing or unreadable. It is not part of the record.
fn read_minutes(path: &Path) -> Option<f64> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut seconds = 0.0;

    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        seconds += serde_json::from_str::<Timing>(line).ok()?.seconds;
    }

    Some(seconds / 60.0)
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;

    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Ok when a run bred exactly the generations its config plans (generation 0 included), or
/// fewer when the call budget ended it. A run that stopped for any other reason is refused.
fn bred_as_planned(stop_reason: &str, completed: usize, generations: usize) -> Result<(), String> {
    let planned = generations + 1;
    let fits = match stop_reason {
        ALL_GENERATIONS => completed == planned,
        CALL_BUDGET => completed < planned,
        _ => {
            return Err(format!(
                "it stopped with {stop_reason:?}, neither {ALL_GENERATIONS:?} nor {CALL_BUDGET:?}"
            ));
        }
    };

    if fits {
        Ok(())
    } else {
        Err(format!(
            "it bred {completed} generations (generation 0 included) and stopped with \
             {stop_reason:?}, but its config plans {planned}"
        ))
    }
}

/// One finished run, read from its directory.
struct Loaded {
    dir: PathBuf,
    cfg: BreedConfig,
    report: Report,
    found: Rediscovery,
    log: LogSummary,
    minutes: Option<f64>,
}

/// The finished run in `dir`, which must have the settings of `arm` with the base settings
/// `base` of arm S.
fn load(dir: &Path, arm: Arm, base: &SoulConfig) -> Result<Loaded, String> {
    let bad = |m: String| format!("{}: {m}", dir.display());
    let cfg: BreedConfig = read_json(&dir.join("config.json"))?;

    if cfg != arm_config(base, arm, cfg.seed, cfg.generations, cfg.max_calls) {
        return Err(bad(format!("its settings are not those of arm {arm}")));
    }

    if !read_finished(&dir.join("snapshot.json"))
        || !dir.join("report.json").exists()
        || !dir.join("rediscovery.json").exists()
    {
        return Err(bad(
            "the run is not finished; wait until it ends, or, if it was stopped or interrupted, \
             run it again from scratch on the same seed into a fresh directory (section 7, step \
             3 of docs/PREREG_STEP2_SOUL.md: a resumed run would differ from a straight one); \
             then run brut-breed --verify on it"
                .to_string(),
        ));
    }

    let report: Report = read_json(&dir.join("report.json"))?;
    let found: Rediscovery = read_json(&dir.join("rediscovery.json"))?;

    bred_as_planned(
        &report.stop_reason,
        report.generations_completed,
        cfg.generations,
    )
    .map_err(bad)?;
    check_counts(&found).map_err(|m| bad(format!("rediscovery.json: {m}")))?;

    let log = read_log(&dir.join("events.jsonl"))?;

    if log.newborns.len() != report.generations_completed {
        return Err(bad(format!(
            "its event log has newborns in {} generations, but its report completed {}",
            log.newborns.len(),
            report.generations_completed
        )));
    }

    Ok(Loaded {
        dir: dir.to_path_buf(),
        cfg,
        report,
        found,
        log,
        minutes: read_minutes(&dir.join("timings.jsonl")),
    })
}

/// The runs of one arm by seed; a seed given twice is refused.
fn load_arm(
    dirs: &[PathBuf],
    arm: Arm,
    base: &SoulConfig,
) -> Result<BTreeMap<u64, Loaded>, String> {
    let mut runs = BTreeMap::new();

    for dir in dirs {
        let run = load(dir, arm, base)?;
        let seed = run.cfg.seed;

        if let Some(earlier) = runs.insert(seed, run) {
            return Err(format!(
                "seed {seed} has two runs in arm {arm}: {} and {}",
                earlier.dir.display(),
                dir.display()
            ));
        }
    }

    Ok(runs)
}

/// The command line: the run directories of each arm, and the file of --preset.
#[derive(Debug, PartialEq, Eq)]
struct Args {
    dirs: BTreeMap<Arm, Vec<PathBuf>>,
    preset: Option<PathBuf>,
}

fn parse_args(args: Vec<String>) -> Result<Args, String> {
    let usage = "usage: soul_comparison --s <dir>... --v <dir>... --k <dir>... \
                 [--e <dir>...] [--n <dir>...] [--preset <file>]";
    let mut dirs: BTreeMap<Arm, Vec<PathBuf>> = BTreeMap::new();
    let mut preset = None;
    let mut current = None;
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        if arg == "--preset" {
            let file = args
                .next()
                .ok_or_else(|| format!("--preset needs a file; {usage}"))?;

            preset = Some(PathBuf::from(file));
            current = None;
        } else if let Some(arm) = Arm::ALL.into_iter().find(|a| a.flag() == arg) {
            current = Some(arm);
        } else if arg.starts_with("--") {
            return Err(format!("unknown option {arg}; {usage}"));
        } else {
            match current {
                Some(arm) => dirs.entry(arm).or_default().push(PathBuf::from(arg)),
                None => return Err(format!("{arg} belongs to no arm; {usage}")),
            }
        }
    }

    for arm in Arm::TRIPLE {
        if !dirs.contains_key(&arm) {
            return Err(format!("arm {arm} needs run directories; {usage}"));
        }
    }

    Ok(Args { dirs, preset })
}

/// The runs by arm and seed.
type Runs = BTreeMap<Arm, BTreeMap<u64, Loaded>>;

fn run(args: Vec<String>) -> Result<String, String> {
    let args = parse_args(args)?;
    let base = match &args.preset {
        Some(file) => read_json::<SoulConfig>(file)?,
        None => SoulConfig::arm_s(),
    };

    base.check().map_err(|e| e.to_string())?;

    let f = Factors::new(P1)?;
    let mut runs: Runs = BTreeMap::new();

    for (&arm, dirs) in &args.dirs {
        runs.insert(arm, load_arm(dirs, arm, &base)?);
    }

    // The seeds of the triples are those of arm S. Arms V and K need a run on each, and no arm
    // may hold a seed without a triple.
    let seeds: Vec<u64> = runs[&Arm::S].keys().copied().collect();

    for (arm, by_seed) in &runs {
        if let Some(seed) = by_seed.keys().find(|s| !runs[&Arm::S].contains_key(s)) {
            return Err(format!(
                "seed {seed} has no run in arm S, but one in arm {arm}"
            ));
        }
    }

    for arm in [Arm::V, Arm::K] {
        if let Some(seed) = seeds.iter().find(|s| !runs[&arm].contains_key(s)) {
            return Err(format!("seed {seed} has no run in arm {arm}"));
        }
    }

    let set = check_seeds(&seeds)?;
    let shared: BTreeSet<(usize, u64)> = runs
        .values()
        .flat_map(BTreeMap::values)
        .map(|r| (r.cfg.generations, r.cfg.max_calls))
        .collect();
    let Some(&(generations, max_calls)) = shared.first() else {
        return Err("no seed triples".to_string());
    };

    if shared.len() > 1 {
        return Err(format!(
            "the runs differ in --generations or --max-calls: {shared:?}"
        ));
    }

    let capped = Arm::TRIPLE
        .iter()
        .flat_map(|&arm| {
            runs[&arm]
                .iter()
                .filter(|(_, r)| r.report.stop_reason != ALL_GENERATIONS)
                .map(move |(&seed, r)| (seed, arm, r.report.generations_completed))
        })
        .collect();
    let context = Context {
        generations,
        max_calls,
        capped,
        preset: args.preset.is_some(),
    };
    let triples: Vec<Triple> = seeds
        .iter()
        .map(|seed| Triple {
            seed: *seed,
            runs: Arm::TRIPLE.map(|arm| runs[&arm][seed].found.clone()),
        })
        .collect();
    let mut out = render_rule(&context, set, &triples, f)?;

    out.push_str(&describe(&context, &runs, &seeds));

    Ok(out)
}

/// A float with three decimals, or "-" when it is absent.
fn show_float(x: Option<f64>) -> String {
    x.map_or("-".to_string(), |x| format!("{x:.3}"))
}

/// The mean of the values present, or None when none is.
fn mean(values: impl IntoIterator<Item = Option<f64>>) -> Option<f64> {
    let present: Vec<f64> = values.into_iter().flatten().collect();

    (!present.is_empty()).then(|| present.iter().sum::<f64>() / present.len() as f64)
}

/// The descriptive results (section 12): no verdict depends on them.
fn describe(context: &Context, runs: &Runs, seeds: &[u64]) -> String {
    let arms: Vec<Arm> = Arm::ALL
        .into_iter()
        .filter(|a| runs.contains_key(a))
        .collect();
    let time = |r: &Loaded| within(pass_time(&r.found));
    let mut out = String::from("\nDescriptive, with no verdict.\n");

    // Pass rates per arm.
    let rates: Vec<String> = arms
        .iter()
        .map(|arm| {
            let by_seed = &runs[arm];
            let passed = by_seed.values().filter(|r| time(r).is_some()).count();

            format!("{arm} {passed}/{}", by_seed.len())
        })
        .collect();

    out.push_str(&format!(
        "Pass rates within generation {GENERATIONS}: {}.\n",
        rates.join(", ")
    ));

    // Each other arm against arm S by the pass time.
    let against: Vec<String> = arms
        .iter()
        .filter(|&&arm| arm != Arm::S)
        .map(|&arm| {
            let mut tally = [0usize; 3];

            for (seed, r) in &runs[&arm] {
                let y = first(time(&runs[&Arm::S][seed]), time(r));

                tally[match y {
                    First::S => 0,
                    First::Control => 1,
                    First::Tie => 2,
                }] += 1;
            }

            format!("{arm} {}/{}/{}", tally[0], tally[1], tally[2])
        })
        .collect();

    out.push_str(&format!(
        "Against arm S by the pass time (S first / the other arm first / tie): {}.\n",
        against.join(", ")
    ));

    // The divergence generation of each pair of arms.
    let pairs: Vec<(Arm, Arm)> = [
        (Arm::S, Arm::V),
        (Arm::S, Arm::K),
        (Arm::V, Arm::K),
        (Arm::S, Arm::E),
        (Arm::S, Arm::N),
    ]
    .into_iter()
    .filter(|(a, b)| runs.contains_key(a) && runs.contains_key(b))
    .collect();

    out.push_str(
        "\nDivergence generation of two runs of a seed: the first generation whose newborns differ, member by member in birth order, \
         in gene id, operator, parents (by gene id), answers, solved count or graded value; none when no generation differs, - when a run is missing.\n",
    );
    out.push_str(&format!("{:>5}", "seed"));

    for (a, b) in &pairs {
        out.push_str(&format!("  {:>5}", format!("{a}-{b}")));
    }

    out.push('\n');

    for seed in seeds {
        out.push_str(&format!("{seed:>5}"));

        for (a, b) in &pairs {
            let cell = match (runs[a].get(seed), runs[b].get(seed)) {
                (Some(x), Some(y)) => divergence(&x.log.newborns, &y.log.newborns)
                    .map_or("none".to_string(), |g| g.to_string()),
                _ => "-".to_string(),
            };

            out.push_str(&format!("  {cell:>5}"));
        }

        out.push('\n');
    }

    // One line per run.
    out.push_str(
        "\nRuns: T, D at the last generation, calls, minutes (the sum of timings.jsonl, not part of the record), \
         complex votes of the newborns on the search panel, flipped and tied votes, their mean coherence, sim_decided, \
         and the first generation with a flipped vote and with sim_decided.\n",
    );
    out.push_str(&format!(
        "{:<7}  {:>5}  {:<15}  {:>9}  {:>7}  {:>6}  {:>7}  {:>6}  {:>9}  {:>5}  {:>10}  {:>9}\n",
        "run",
        "T",
        "D",
        "calls",
        "minutes",
        "votes",
        "flipped",
        "tied",
        "coherence",
        "sim",
        "first flip",
        "first sim"
    ));

    for arm in &arms {
        for (seed, r) in &runs[arm] {
            let v = &r.log.votes;
            let coherence = (v.complex > 0).then(|| v.coherence_sum / v.complex as f64);

            out.push_str(&format!(
                "{:<7}  {:>5}  {:<15}  {:>9}  {:>7}  {:>6}  {:>7}  {:>6}  {:>9}  {:>5}  {:>10}  {:>9}\n",
                format!("{arm}-{seed}"),
                show_time(time(r)),
                show_share(count(&r.found).share),
                r.report.calls_used,
                r.minutes.map_or("-".to_string(), |m| format!("{m:.2}")),
                v.complex,
                v.flipped,
                v.tied,
                show_float(coherence),
                v.sim_decided,
                r.log.first_flip.map_or("-".to_string(), |g| g.to_string()),
                r.log.first_sim.map_or("-".to_string(), |g| g.to_string())
            ));
        }
    }

    // The votes per arm.
    out.push_str(
        "\nComplex votes of the newborns per arm: flipped (the winner differs from the count's) and tied (the tie rule decided), \
         with their shares of the votes, the mean coherence, and sim_decided.\n",
    );
    out.push_str(&format!(
        "{:<3}  {:>4}  {:>7}  {:>16}  {:>16}  {:>9}  {:>11}\n",
        "arm", "runs", "votes", "flipped", "tied", "coherence", "sim_decided"
    ));

    for arm in &arms {
        let mut v = VoteTally::default();

        for r in runs[arm].values() {
            v.complex += r.log.votes.complex;
            v.flipped += r.log.votes.flipped;
            v.tied += r.log.votes.tied;
            v.coherence_sum += r.log.votes.coherence_sum;
            v.sim_decided += r.log.votes.sim_decided;
        }

        let share = |x: u64| {
            if v.complex > 0 {
                format!("{x} ({:.4})", x as f64 / v.complex as f64)
            } else {
                format!("{x}")
            }
        };
        let coherence = (v.complex > 0).then(|| v.coherence_sum / v.complex as f64);

        out.push_str(&format!(
            "{:<3}  {:>4}  {:>7}  {:>16}  {:>16}  {:>9}  {:>11}\n",
            arm,
            runs[arm].len(),
            v.complex,
            share(v.flipped),
            share(v.tied),
            show_float(coherence),
            v.sim_decided
        ));
    }

    // The trajectories of the soul group.
    out.push_str(&format!(
        "\nTrajectories of the soul group per arm with Souls: at every {TRAJECTORY_STEP}th generation the mean over the arm's runs of \
         contrast, contrast_independent, order_mean, the median of kappa, moved_mean (degrees) and character_norm_mean; \
         then, over its runs and the {TRAJECTORY_STEP} generations up to it, the complex votes, the shares of flipped and of tied votes, \
         their mean coherence, and sim_decided.\n"
    ));
    out.push_str(&format!(
        "{:<3}  {:>5}  {:>8}  {:>11}  {:>10}  {:>12}  {:>10}  {:>14}  {:>6}  {:>7}  {:>6}  {:>9}  {:>5}\n",
        "arm",
        "gen",
        "contrast",
        "independent",
        "order_mean",
        "kappa_median",
        "moved_mean",
        "character_norm",
        "votes",
        "flipped",
        "tied",
        "coherence",
        "sim"
    ));

    let mut marks: Vec<usize> = (0..=context.generations).step_by(TRAJECTORY_STEP).collect();

    if marks.last() != Some(&context.generations) {
        marks.push(context.generations);
    }

    for arm in &arms {
        let souled: Vec<&Loaded> = runs[arm]
            .values()
            .filter(|r| r.log.souls.iter().any(Option::is_some))
            .collect();

        if souled.is_empty() {
            continue;
        }

        let mut from = 0;

        for &g in &marks {
            let at = |r: &&Loaded| r.log.souls.get(g).cloned().flatten();
            let states: Vec<SoulState> = souled.iter().filter_map(at).collect();
            let in_window: Vec<&SoulState> = souled
                .iter()
                .flat_map(|r| r.log.souls.iter().skip(from).take(g + 1 - from))
                .flatten()
                .collect();
            let window =
                |field: fn(&SoulState) -> u64| -> u64 { in_window.iter().map(|s| field(s)).sum() };
            let votes = window(|s| s.votes);
            let share = |x: u64| (votes > 0).then(|| x as f64 / votes as f64);
            // The coherence of each generation is a mean over its votes; weigh it by them.
            let coherence = (votes > 0).then(|| {
                in_window
                    .iter()
                    .map(|s| s.coherence_mean.unwrap_or(0.0) * s.votes as f64)
                    .sum::<f64>()
                    / votes as f64
            });

            out.push_str(&format!(
                "{:<3}  {:>5}  {:>8}  {:>11}  {:>10}  {:>12}  {:>10}  {:>14}  {:>6}  {:>7}  {:>6}  {:>9}  {:>5}\n",
                arm,
                g,
                show_float(mean(states.iter().map(|s| s.contrast))),
                show_float(mean(states.iter().map(|s| s.contrast_independent))),
                show_float(mean(states.iter().map(|s| s.order_mean))),
                show_float(mean(states.iter().map(|s| s.kappa.as_ref().map(|q| q.median)))),
                show_float(mean(states.iter().map(|s| s.moved_mean))),
                show_float(mean(states.iter().map(|s| s.character_norm_mean))),
                votes,
                show_float(share(window(|s| s.votes_flipped))),
                show_float(share(window(|s| s.votes_tied))),
                show_float(coherence),
                window(|s| s.sim_decided)
            ));
            from = g + 1;
        }
    }

    // The first generation per target.
    let targets: Vec<&str> = verified().iter().map(|t| t.name).collect();

    out.push_str(
        "\nThe first generation in which a bred Cascade solved each target (- never; * a target the seed Cascades solve).\n",
    );
    out.push_str(&format!("{:>5}  {:<3}", "seed", "arm"));

    for t in &targets {
        out.push_str(&format!("  {t:>7}"));
    }

    out.push('\n');

    for seed in seeds {
        for arm in &arms {
            let Some(r) = runs[arm].get(seed) else {
                continue;
            };

            out.push_str(&format!("{seed:>5}  {arm:<3}"));

            for t in &r.found.targets {
                let cell = format!(
                    "{}{}",
                    t.first_generation
                        .map_or("-".to_string(), |g| g.to_string()),
                    if t.reached_by_seed { "*" } else { "" }
                );

                out.push_str(&format!("  {cell:>7}"));
            }

            out.push('\n');
        }
    }

    // Runs at the call cap and invention candidates, in every arm.
    let mut capped = Vec::new();
    let mut candidates = Vec::new();

    for arm in &arms {
        for (seed, r) in &runs[arm] {
            if r.report.stop_reason != ALL_GENERATIONS {
                capped.push(format!(
                    "Run of seed {seed}, arm {arm} ended at the call cap after {} generation(s).\n",
                    r.report.generations_completed
                ));
            }

            for t in &r.found.targets {
                if let Some(c) = &t.candidate {
                    candidates.push(format!(
                        "  seed {seed}, arm {arm}, {}: T {}, CNOT {} by {} (generation {}, task {}); literature checked: {}\n",
                        t.target,
                        c.cost.t,
                        c.cost.cnot,
                        c.member,
                        c.generation,
                        c.task,
                        c.literature_checked
                    ));
                }
            }
        }
    }

    out.push('\n');

    if capped.is_empty() {
        out.push_str("Runs that ended at the call cap: none.\n");
    } else {
        capped.iter().for_each(|line| out.push_str(line));
    }

    if candidates.is_empty() {
        out.push_str("Invention candidates (below a BestKnown count): none.\n");
    } else {
        out.push_str(
            "Invention candidates (below a BestKnown count; a human checks the literature before any claim, Q38):\n",
        );
        candidates.iter().for_each(|line| out.push_str(line));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use brutmaschine::cascade::arena::{ModelPool, Usage, breed_to_end_in};
    use brutmaschine::cascade::circuit::{CircuitDomain, Reference, TargetOutcome};
    use brutmaschine::cascade::circuit_genes::circuit_pool;
    use brutmaschine::cascade::designer::SimDesigner;
    use brutmaschine::cascade::qubit::Cost;

    /// A cost with these T and CNOT counts.
    fn cost(t: u32, cnot: u32) -> Cost {
        Cost {
            t,
            cnot,
            ..Cost::default()
        }
    }

    /// One target's outcome: the cheapest exact seed and bred answers, if any. A target counts
    /// as reached when that answer is at or below both reference counts.
    fn outcome(name: &str, best_seed: Option<Cost>, best_bred: Option<Cost>) -> TargetOutcome {
        let entry = verified()
            .into_iter()
            .find(|t| t.name == name)
            .expect("a verified target");
        let reference = entry.reference.expect("a reference");
        let reached = |best: Option<Cost>| best.is_some_and(|c| reference.reached(&c));

        TargetOutcome {
            target: name.to_string(),
            reference,
            tasks: 3,
            best_seed,
            best_bred,
            reached_by_seed: reached(best_seed),
            reached_by_bred: reached(best_bred),
            first_generation: reached(best_bred).then_some(4),
            circuit: None,
            below_best_known: false,
            candidate: None,
        }
    }

    /// The reference cost of a target.
    fn at_reference(name: &str) -> Option<Cost> {
        let r: Reference = verified()
            .into_iter()
            .find(|t| t.name == name)
            .and_then(|t| t.reference)
            .expect("a reference");

        Some(cost(r.t, r.cnot))
    }

    /// The summary fields of `rediscovery` for these outcomes.
    fn summarise_found(targets: Vec<TargetOutcome>) -> Rediscovery {
        let library = targets.len();
        let seeds = targets.iter().filter(|t| t.reached_by_seed).count();
        let bred_only = targets
            .iter()
            .filter(|t| t.reached_by_bred && !t.reached_by_seed)
            .count();
        let open = library - seeds;
        let calibrated = 3 * seeds <= library;

        Rediscovery {
            targets,
            library,
            reached_by_seeds: seeds,
            reached_by_bred_only: bred_only,
            share: (open > 0).then(|| bred_only as f64 / open as f64),
            calibrated,
            passes: calibrated && 2 * bred_only > open,
        }
    }

    /// A run in which the seed Cascades solve `seeds` and bred Cascades solve `bred`, each at
    /// the reference; every other target only exactly, above its reference.
    fn found(seeds: &[&str], bred: &[&str]) -> Rediscovery {
        let targets: Vec<TargetOutcome> = verified()
            .iter()
            .map(|t| {
                let r = t.reference.expect("a reference");
                let above = Some(cost(r.t + 1, r.cnot + 1));
                let seed = if seeds.contains(&t.name) {
                    at_reference(t.name)
                } else {
                    above
                };
                let child = if bred.contains(&t.name) {
                    at_reference(t.name)
                } else {
                    above
                };

                outcome(t.name, seed, child)
            })
            .collect();

        summarise_found(targets)
    }

    const SEEDS_HIT: [&str; 2] = ["swap", "cs"];

    /// A run in which the seed Cascades solve `seeds` and bred Cascades solve the first targets
    /// the seeds miss, in library order, each first in the given generation.
    fn timed(seeds: &[&str], firsts: &[usize]) -> Rediscovery {
        let open: Vec<&str> = verified()
            .iter()
            .map(|t| t.name)
            .filter(|n| !seeds.contains(n))
            .collect();
        let mut run = found(seeds, &open[..firsts.len()]);
        let solved = run.targets.iter_mut().filter(|t| t.reached_by_bred);

        for (t, &g) in solved.zip(firsts) {
            t.first_generation = Some(g);
        }

        run
    }

    /// A run that passes in generation t (five of the nine open targets fall then), or never.
    fn passing(t: Option<usize>) -> Rediscovery {
        match t {
            Some(g) => timed(&SEEDS_HIT, &[g; 5]),
            None => timed(&SEEDS_HIT, &[]),
        }
    }

    /// Triples for these seeds with these pass times of arms S, V and K, None for never. The
    /// seed Cascades solve swap and cs.
    fn triples(seeds: &[u64], times: &[[Option<usize>; 3]]) -> Vec<Triple> {
        assert_eq!(seeds.len(), times.len());

        seeds
            .iter()
            .zip(times)
            .map(|(&seed, t)| Triple {
                seed,
                runs: [passing(t[0]), passing(t[1]), passing(t[2])],
            })
            .collect()
    }

    /// The first n counted seeds.
    fn counted(n: usize) -> Vec<u64> {
        counted_seeds().take(n).collect()
    }

    /// Pass times that give these outcomes: Y_V and Y_K as 's' (S first), 'c' (the control
    /// first) or 't' (a tie).
    fn outcomes(ys: &[(char, char)]) -> Vec<[Option<usize>; 3]> {
        let control = |y: char| match y {
            's' => Some(20),
            'c' => Some(5),
            't' => Some(10),
            _ => panic!("no outcome {y}"),
        };

        ys.iter()
            .map(|&(v, k)| [Some(10), control(v), control(k)])
            .collect()
    }

    /// The rows of `triples`.
    fn rows_of(triples: &[Triple]) -> Vec<Row> {
        rows(triples).expect("rows")
    }

    fn p34() -> Factors {
        Factors::new((3, 4)).expect("p1 = 3/4")
    }

    fn chosen() -> Factors {
        Factors::new(P1).expect("the pre-registered p1")
    }

    /// The evidence after these outcomes, from the start.
    fn evidence_after(ys: &[First]) -> Evidence {
        ys.iter().fold(Evidence::default(), |e, &y| e.step(y))
    }

    #[test]
    fn exact_fractions() {
        assert_eq!(Ratio::new(6, 8), Ratio { num: 3, den: 4 });
        assert_eq!(Ratio::new(0, 9), Ratio { num: 0, den: 1 });
        assert_eq!(Ratio::new(7, 9).to_string(), "7/9 = 0.7778");
    }

    #[test]
    fn exact_naturals() {
        assert_eq!(Natural::powers(&[]).to_string(), "1");
        assert_eq!(Natural::powers(&[(7, 0)]).to_string(), "1");
        assert_eq!(Natural::powers(&[(3, 20)]).to_string(), "3486784401");
        // A lower digit keeps its leading zeros: 2^30 = 1 073741824.
        assert_eq!(Natural::powers(&[(2, 30)]).to_string(), "1073741824");
        assert_eq!(
            Natural::powers(&[(10, 18)]).to_string(),
            format!("1{}", "0".repeat(18))
        );
        assert_eq!(
            Natural::powers(&[(3, 120)]).to_string(),
            "1797010299914431210413179829509605039731475627537851106401"
        );
        // 7^60 * 3^60 = 21^60, written out by Python's integers.
        assert_eq!(
            Natural::powers(&[(7, 60), (3, 60)]),
            Natural::powers(&[(21, 60)])
        );
        assert_eq!(
            Natural::powers(&[(5, 120)]).to_string(),
            "752316384526264005099991383822237233803945956334136013765601092018187046051025390625"
        );
        assert_eq!(
            Natural::powers(&[(2, 30)]).times(100),
            Natural::powers(&[(10, 2)]).times(1 << 30)
        );
        // A carry of more than one digit.
        assert_eq!(
            Natural(vec![999_999_999]).times(u32::MAX).to_string(),
            "4294967290705032705"
        );

        // Order: more digits are larger, equal counts compare from the top digit down.
        assert_eq!(
            Natural::powers(&[(10, 9)]).cmp(&Natural(vec![0, 1])),
            Ordering::Equal
        );
        assert!(Natural::powers(&[(10, 9)]) > Natural(vec![999_999_999]));
        assert!(Natural(vec![5, 2]) > Natural(vec![999_999_999, 1]));
        assert!(Natural(vec![4, 2]) < Natural(vec![5, 2]));
    }

    #[test]
    fn the_factors_are_the_likelihood_ratio_of_p1() {
        // a = 2p1 and b = 2(1 - p1), each in lowest terms.
        assert_eq!(
            p34(),
            Factors {
                p1: (3, 4),
                up: (3, 2),
                down: (1, 2)
            }
        );
        assert_eq!(
            Factors::new((7, 10)),
            Ok(Factors {
                p1: (7, 10),
                up: (7, 5),
                down: (3, 5)
            })
        );
        assert_eq!(
            Factors::new((3, 5)),
            Ok(Factors {
                p1: (3, 5),
                up: (6, 5),
                down: (4, 5)
            })
        );
        assert_eq!(
            Factors::new((2, 3)),
            Ok(Factors {
                p1: (2, 3),
                up: (4, 3),
                down: (2, 3)
            })
        );

        // p1 must lie strictly between 1/2 and 1.
        for bad in [(1, 2), (2, 4), (1, 3), (1, 1), (5, 4), (0, 1), (1, 0)] {
            assert!(Factors::new(bad).is_err(), "{bad:?}");
        }

        // The pre-registered p1 is one of the good ones.
        assert!(Factors::new(P1).is_ok());
    }

    #[test]
    fn the_evidence_of_hand_made_triples_is_exact() {
        use First::{Control as C, S, Tie as T};

        // p1 = 3/4: E = 3^wins / 2^(wins + losses).
        assert_eq!(Evidence::default().show(p34()), "1/1 = 1.0000");
        assert_eq!(
            evidence_after(&[S, S, C, T, S]).show(p34()),
            "27/16 = 1.6875"
        );
        assert_eq!(evidence_after(&[C, C, T, T]).show(p34()), "1/4 = 0.2500");
        assert_eq!(
            evidence_after(&[S; 14]).show(p34()),
            "4782969/16384 = 291.9293"
        );

        // p1 = 7/10: E = 7^wins 3^losses / 5^(wins + losses).
        let p7 = Factors::new((7, 10)).expect("p1 = 7/10");

        assert_eq!(evidence_after(&[S, C, S, T]).show(p7), "147/125 = 1.1760");
        assert_eq!(evidence_after(&[C, C, C]).show(p7), "27/125 = 0.2160");

        // A tie leaves E as it is.
        let e = evidence_after(&[S, C, S]);

        assert_eq!(e.step(T), e);
        assert_eq!(e.step(S), Evidence { wins: 3, losses: 1 });
        assert_eq!(e.step(C), Evidence { wins: 2, losses: 2 });
    }

    #[test]
    fn evidence_boundaries_in_count_form() {
        let e = |wins: u32, losses: u32| Evidence { wins, losses };
        // p1 = 3/4. With 0 to 7 losses, the wins that prove a claim (E >= 200); with 0 to 7
        // wins, the losses that fail it (E <= 1/20).
        let prove = [14, 15, 17, 19, 20, 22, 24, 26];
        let fail = [5, 5, 6, 7, 7, 8, 8, 9];

        for (other, (&wins, &losses)) in (0..).zip(prove.iter().zip(&fail)) {
            assert!(e(wins, other).proven(p34()), "{wins} wins, {other} losses");
            assert!(
                !e(wins - 1, other).proven(p34()),
                "{wins} - 1 wins, {other} losses"
            );
            assert!(
                e(other, losses).failed(p34()),
                "{other} wins, {losses} losses"
            );
            assert!(
                !e(other, losses - 1).failed(p34()),
                "{other} wins, {losses} - 1 losses"
            );
        }
    }

    #[test]
    fn the_integer_tests_agree_with_u128_where_it_fits() {
        // For p1 = 3/4, E = 3^w / 2^(w + l). 200 * 2^n < 2^127 fits for n <= 119, and a 3^w
        // that does not fit exceeds 200 * 2^n.
        let f = p34();

        for n in 0..=119u32 {
            for w in 0..=n {
                let e = Evidence {
                    wins: w,
                    losses: n - w,
                };
                let three = 3u128.checked_pow(w);
                let proven = three.is_none_or(|p| p >= 200 * (1u128 << n));
                let failed = three
                    .and_then(|p| p.checked_mul(20))
                    .is_some_and(|x| x <= 1u128 << n);

                assert_eq!(e.proven(f), proven, "{w} wins in {n}");
                assert_eq!(e.failed(f), failed, "{w} wins in {n}");
            }
        }

        // For p1 = 7/10, E = 7^w 3^l / 5^(w + l). Up to n = 43, 20 * 7^n and 200 * 5^n fit in
        // a u128.
        let f = Factors::new((7, 10)).expect("p1");

        for n in 0..=43u32 {
            for w in 0..=n {
                let e = Evidence {
                    wins: w,
                    losses: n - w,
                };
                let num = 7u128.pow(w) * 3u128.pow(n - w);
                let den = 5u128.pow(n);

                assert_eq!(e.proven(f), num >= 200 * den, "{w} wins in {n}");
                assert_eq!(e.failed(f), 20 * num <= den, "{w} wins in {n}");
            }
        }
    }

    #[test]
    fn pass_time_is_the_kth_bred_only_first_generation() {
        // Nine open targets: D > 1/2 needs five, so T is the fifth smallest first generation.
        assert_eq!(
            pass_time(&timed(&SEEDS_HIT, &[97, 33, 75, 7, 33, 10])),
            Some(75)
        );
        assert_eq!(pass_time(&timed(&SEEDS_HIT, &[7, 10, 33, 33])), None);

        // With |R_seed| = 3, eight targets are open: four are only half, five pass.
        let three = ["swap", "cs", "cz"];

        assert_eq!(pass_time(&timed(&three, &[2, 4, 6, 8])), None);
        assert_eq!(pass_time(&timed(&three, &[12, 10, 8, 6, 4, 2])), Some(10));

        // No open target, no pass time.
        let every: Vec<&str> = verified().iter().map(|t| t.name).collect();

        assert_eq!(pass_time(&timed(&every, &[])), None);

        // A pass after the ceiling is never.
        assert_eq!(within(Some(GENERATIONS)), Some(GENERATIONS));
        assert_eq!(within(Some(GENERATIONS + 1)), None);
    }

    #[test]
    fn the_pass_time_of_a_step1_run_is_the_one_step1_counted() {
        // The rediscovery.json of run A-310 of the step 1 pilot, saved at generation 922. The
        // operating characteristics of step 1 (sequential-rule/oc.txt) read its pass time as
        // 336.
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".scratch/universum-erfinden/sequential-rule/a310-rediscovery-at-922.json");
        let a310: Rediscovery = read_json(&path).expect("A-310");

        assert_eq!(check_counts(&a310), Ok(()));
        assert_eq!(pass_time(&a310), Some(336));
        assert_eq!(
            count(&a310).seeds,
            BTreeSet::from(SEED_TARGETS.map(String::from))
        );
    }

    #[test]
    fn first_to_pass_decides_each_claim() {
        assert_eq!(first(Some(7), Some(380)), First::S);
        assert_eq!(first(Some(97), None), First::S);
        assert_eq!(first(None, Some(720)), First::Control);
        assert_eq!(first(Some(105), Some(48)), First::Control);
        assert_eq!(first(Some(50), Some(50)), First::Tie);
        assert_eq!(first(None, None), First::Tie);
        assert_eq!(show_first(First::Control, Arm::K), "K first");

        let row = &rows_of(&triples(&[FIRST_SEED], &[[Some(30), None, Some(12)]]))[0];

        assert_eq!((row.y_v(), row.y_k()), (First::S, First::Control));
    }

    #[test]
    fn counts_must_agree_with_the_target_list() {
        let run = found(&SEEDS_HIT, &["cz", "qft2"]);

        assert_eq!(check_counts(&run), Ok(()));

        let mut forged = run.clone();

        forged.reached_by_bred_only += 1;
        assert!(check_counts(&forged).is_err());

        let mut forged = run.clone();

        forged.passes = !forged.passes;
        assert!(check_counts(&forged).is_err());

        // A target a bred Cascade solved, without its first generation.
        let mut forged = run.clone();

        forged
            .targets
            .iter_mut()
            .find(|t| t.reached_by_bred)
            .expect("a bred target")
            .first_generation = None;
        assert!(check_counts(&forged).is_err());

        let mut forged = run;

        forged.targets.pop();
        assert!(check_counts(&forged).is_err());
    }

    #[test]
    fn a_triple_whose_arms_disagree_on_r_seed_is_refused() {
        let mut all = triples(&counted(2), &outcomes(&[('s', 's'), ('s', 's')]));

        all[1].runs[2] = found(&["swap"], &[]);

        let refused = rows(&all).expect_err("R_seed differs");

        assert!(
            refused.contains(&format!("seed {}", FIRST_SEED + 1)),
            "{refused}"
        );
    }

    #[test]
    fn ties_leave_the_evidence_unchanged() {
        let all = triples(
            &counted(3),
            &outcomes(&[('t', 's'), ('s', 't'), ('t', 't')]),
        );
        let rule = sequential(&rows_of(&all), p34());

        assert_eq!(
            rule.evidence,
            vec![
                (Evidence::default(), Evidence { wins: 1, losses: 0 }),
                (
                    Evidence { wins: 1, losses: 0 },
                    Evidence { wins: 1, losses: 0 }
                ),
                (
                    Evidence { wins: 1, losses: 0 },
                    Evidence { wins: 1, losses: 0 }
                ),
            ]
        );
    }

    #[test]
    fn each_claim_is_decided_alone_and_the_rule_stops_when_both_are() {
        // Claim V: S first in each triple, proven at triple 14 (p1 = 3/4). Claim K: the
        // control first five times in a row, failed at triple 5.
        let mut ys = vec![('s', 'c'); 5];

        ys.extend(vec![('s', 't'); 15]);

        let all = triples(&counted(ys.len()), &outcomes(&ys));
        let rule = sequential(&rows_of(&all), p34());

        assert_eq!(rule.k, Verdict::Failed(5));
        assert_eq!(rule.v, Verdict::Proven(14));
        assert_eq!(rule.stop, Some(14));
        assert_eq!(rule.counted(), 14);

        // While claim V is open, the failed claim K does not stop the rule.
        let open = sequential(&rows_of(&all[..10]), p34());

        assert_eq!(open.stop, None);
        assert_eq!((open.v, open.k), (Verdict::Open, Verdict::Failed(5)));
    }

    #[test]
    fn the_first_crossing_is_final() {
        // Claim V is proven at triple 14, then arm V passes first in every triple: E_V falls
        // below 1/20, but the verdict stays. Claim K stays open by ties until triple 30, then
        // fails at triple 34.
        let mut ys = vec![('s', 't'); 14];

        ys.extend(vec![('c', 't'); 16]);
        ys.extend(vec![('c', 'c'); 6]);

        let all = triples(&counted(ys.len()), &outcomes(&ys));
        let rule = sequential(&rows_of(&all), p34());

        assert_eq!(rule.v, Verdict::Proven(14));
        assert_eq!(rule.k, Verdict::Failed(35));
        assert_eq!(rule.stop, Some(35));

        let (e_v, _) = rule.evidence[rule.counted() - 1];

        assert!(e_v.failed(p34()), "{e_v:?}");
    }

    /// Outcomes for `n` triples on which neither E crosses: S first exactly when E < 1 before
    /// the triple, so E stays in [1/2, 3/2) for p1 = 3/4.
    fn never_crossing(n: usize) -> Vec<(char, char)> {
        let mut e = Evidence::default();
        let mut ys = Vec::new();

        for _ in 0..n {
            let below = e.numerator(p34()) < e.denominator(p34());
            let y = if below { 's' } else { 'c' };

            ys.push((y, y));
            e = e.step(if below { First::S } else { First::Control });
        }

        ys
    }

    #[test]
    fn a_claim_still_open_at_the_cap_is_undecided() {
        let ys = never_crossing(MAX_TRIPLES);
        let all = triples(&counted(MAX_TRIPLES), &outcomes(&ys));
        let rows = rows_of(&all);
        let rule = sequential(&rows, p34());

        assert_eq!(rule.stop, Some(MAX_TRIPLES));
        assert_eq!((rule.v, rule.k), (Verdict::Undecided, Verdict::Undecided));

        let made = claim(&Context::default(), SeedSet::Counted, &rows, &rule);

        assert_eq!(made, Claim::Made);
        assert_eq!(
            verdict(&made, &rule, p34()),
            format!(
                "Claim V: UNDECIDED after {MAX_TRIPLES} triples, no claim.\n\
                 Claim K: UNDECIDED after {MAX_TRIPLES} triples, no claim.\n"
            )
        );
    }

    #[test]
    fn triples_after_the_stop_are_not_counted() {
        // Both claims are proven at triple 14; the triples after it are listed only.
        let all = triples(&counted(16), &outcomes(&[('s', 's'); 16]));
        let text = render_rule(&Context::default(), SeedSet::Counted, &all, p34()).expect("render");

        assert!(
            text.contains(&format!(
                " {}  after the stop, not counted\n",
                FIRST_SEED + 14
            )),
            "{text}"
        );
        assert!(
            text.contains(&format!(
                " {}  after the stop, not counted\n",
                FIRST_SEED + 15
            )),
            "{text}"
        );
        assert!(text.contains("The rule stops after triple 14.\n"), "{text}");
        assert!(text.contains("Claim V: PROVEN at triple 14."), "{text}");
        assert!(text.contains("Claim K: PROVEN at triple 14."), "{text}");
        assert!(
            text.contains(
                "E_V = 4782969/16384 = 291.9293 after 14 triple(s); it reached 200 at triple 14.\n"
            ),
            "{text}"
        );
    }

    #[test]
    fn an_open_rule_names_the_next_seed() {
        let all = triples(
            &counted(3),
            &outcomes(&[('s', 'c'), ('t', 's'), ('c', 't')]),
        );
        let text = render_rule(&Context::default(), SeedSet::Counted, &all, p34()).expect("render");

        assert!(
            text.contains(&format!("OPEN: run seed {} next.\n", FIRST_SEED + 3)),
            "{text}"
        );
        assert!(
            text.contains("The rule goes on after triple 3.\n"),
            "{text}"
        );
        assert!(
            !text.contains("PROVEN") && !text.contains("FAILED"),
            "{text}"
        );
    }

    #[test]
    fn the_counted_seeds_run_from_1002_without_gaps() {
        assert_eq!(check_seeds(&[1002]), Ok(SeedSet::Counted));
        assert_eq!(check_seeds(&counted(MAX_TRIPLES)), Ok(SeedSet::Counted));

        // Pilot seeds alone are accepted, with gaps and in any part of their range.
        assert_eq!(check_seeds(&[901, 902, 912]), Ok(SeedSet::Pilot));
        assert_eq!(check_seeds(&[921, 922, 923]), Ok(SeedSet::Pilot));

        // Refused: a gap, a list that does not start at 1002, a seed off the list (1001 among
        // them, the seed of the tool check), a mixture of pilot and counted seeds, more than
        // N_max seeds, and no seed.
        let more: Vec<u64> = (FIRST_SEED..).take(MAX_TRIPLES + 1).collect();

        for (seeds, why) in [
            (vec![1002, 1004], "without gaps"),
            (vec![1003], "without gaps"),
            (vec![1001], "not on the list"),
            (vec![77, 1002], "not on the list"),
            (vec![1002, 5000], "not on the list"),
            (vec![924], "not on the list"),
            (vec![901, 1002], "pilot seeds"),
            (more, "more than"),
            (vec![], "no seed"),
        ] {
            let refused = check_seeds(&seeds).expect_err(why);

            assert!(refused.contains(why), "{seeds:?}: {refused}");
        }
    }

    #[test]
    fn pilot_seeds_never_make_a_claim() {
        // Twelve pilot triples in which arm S passes first: the evidence would prove both
        // claims later, but no claim is made, and no seed is named next.
        let seeds: Vec<u64> = (901..=916).collect();
        let all = triples(&seeds, &outcomes(&[('s', 's'); 16]));
        let text = render_rule(&Context::default(), SeedSet::Pilot, &all, p34()).expect("render");

        assert!(
            text.contains(
                "No claim: the seeds are pilot seeds of step 2 (901 to 923), which never count.\n"
            ),
            "{text}"
        );
        assert!(
            !text.contains("PROVEN") && !text.contains("OPEN") && !text.contains("VOID"),
            "{text}"
        );

        let rows = rows_of(&all);
        let rule = sequential(&rows, p34());

        assert!(matches!(
            claim(&Context::default(), SeedSet::Pilot, &rows, &rule),
            Claim::Withheld(_)
        ));
    }

    #[test]
    fn a_seed_that_fails_gate_g1_voids_the_experiment() {
        let ys = vec![('s', 's'); 14];
        let mut all = triples(&counted(14), &outcomes(&ys));

        // The third seed: the seed Cascades also solve cz (too many); the fifth: only swap.
        for run in &mut all[2].runs {
            *run = found(
                &["swap", "cs", "cz"],
                &["ccz", "toffoli", "peres", "rtof", "ccix"],
            );
        }

        for run in &mut all[4].runs {
            *run = found(&["swap"], &["cz", "ccz", "toffoli", "peres", "rtof"]);
        }

        let rows = rows_of(&all);
        let rule = sequential(&rows, p34());

        assert!(!rows[2].g1() && !rows[4].g1() && rows[0].g1());
        assert_eq!(
            claim(&Context::default(), SeedSet::Counted, &rows, &rule),
            Claim::Void(vec![FIRST_SEED + 2, FIRST_SEED + 4])
        );

        let text = render_rule(&Context::default(), SeedSet::Counted, &all, p34()).expect("render");

        assert!(
            text.contains("No claim: the experiment is void.\n"),
            "{text}"
        );
        assert!(text.contains("VOID"), "{text}");
    }

    #[test]
    fn only_the_pre_registered_length_cap_and_presets_make_a_claim() {
        let all = triples(&counted(14), &outcomes(&[('s', 's'); 14]));
        let rows = rows_of(&all);
        let rule = sequential(&rows, chosen());

        assert_eq!(
            claim(&Context::default(), SeedSet::Counted, &rows, &rule),
            Claim::Made
        );

        let other = [
            Context {
                generations: 200,
                ..Context::default()
            },
            Context {
                max_calls: 2_600_000,
                ..Context::default()
            },
            Context {
                capped: vec![(FIRST_SEED, Arm::K, 500)],
                ..Context::default()
            },
            Context {
                preset: true,
                ..Context::default()
            },
        ];

        for context in &other {
            let made = claim(context, SeedSet::Counted, &rows, &rule);
            let text = render_rule(context, SeedSet::Counted, &all, p34()).expect("render");

            assert!(matches!(made, Claim::Withheld(_)), "{made:?}");
            assert!(text.contains("No claim"), "{text}");
            assert!(!text.contains("PROVEN") && !text.contains("OPEN"), "{text}");
        }
    }

    #[test]
    fn the_arms_follow_the_presets_of_soul() {
        let base = SoulConfig::arm_s();

        assert_eq!(arm_soul(&base, Arm::S), Some(SoulConfig::arm_s()));
        assert_eq!(arm_soul(&base, Arm::V), Some(SoulConfig::arm_v()));
        assert_eq!(arm_soul(&base, Arm::K), Some(SoulConfig::arm_k()));
        assert_eq!(arm_soul(&base, Arm::N), None);

        let e = arm_soul(&base, Arm::E).expect("arm E has Souls");

        assert_eq!(e.strengths, Strengths::Equal);
        assert_eq!(
            SoulConfig {
                strengths: Strengths::Domain,
                ..e
            },
            base
        );

        // Arm N is arm A of step 1: the circuit settings and every default.
        let n = arm_config(&base, Arm::N, 7, 1000, 13_000_000);

        assert_eq!(
            n,
            BreedConfig {
                seed: 7,
                generations: 1000,
                children_per_generation: 8,
                search_puzzles: 33,
                select_puzzles: 22,
                max_calls: 13_000_000,
                domain: DomainConfig::Circuits(CircuitConfig::default()),
                ..BreedConfig::default()
            }
        );

        // Each arm differs from arm N only in its soul settings.
        for arm in Arm::ALL {
            let cfg = arm_config(&base, arm, 7, 1000, 13_000_000);

            assert_eq!(
                BreedConfig {
                    soul: None,
                    ..cfg.clone()
                },
                n,
                "{arm}"
            );
            assert_eq!(cfg.soul, arm_soul(&base, arm), "{arm}");
            assert!(cfg.check().is_ok(), "{arm}");
        }
    }

    #[test]
    fn the_divergence_generation_is_the_first_generation_whose_newborns_differ() {
        let newborn = |gene: &str| Newborn {
            gene_id: gene.to_string(),
            op: Op::Seed,
            parents: Vec::new(),
            answers: "a".to_string(),
            solved: 1,
            quality: Some(1.0f64.to_bits()),
        };
        let log: Vec<Vec<Newborn>> = (0..5)
            .map(|g| vec![newborn(&format!("g{g}a")), newborn(&format!("g{g}b"))])
            .collect();

        assert_eq!(divergence(&log, &log), None);

        // Each field of a newborn counts.
        let changes: [fn(&mut Newborn); 6] = [
            |n| n.gene_id.push('x'),
            |n| n.op = Op::WorkerAdd,
            |n| n.parents.push("p".to_string()),
            |n| n.answers.push('x'),
            |n| n.solved += 1,
            |n| n.quality = Some(1.5f64.to_bits()),
        ];

        for change in changes {
            let mut other = log.clone();

            change(&mut other[3][1]);
            assert_eq!(divergence(&log, &other), Some(3));
            assert_eq!(divergence(&other, &log), Some(3));
        }

        // Another order of the same newborns, an extra newborn, and a longer run.
        let mut swapped = log.clone();

        swapped[2].swap(0, 1);
        assert_eq!(divergence(&log, &swapped), Some(2));

        let mut extra = log.clone();

        extra[4].push(newborn("x"));
        assert_eq!(divergence(&log, &extra), Some(4));

        let mut longer = log.clone();

        longer.push(vec![newborn("g5")]);
        assert_eq!(divergence(&log, &longer), Some(5));
    }

    /// The summary of the events in log order, as `read_log` makes it from a file.
    fn summarise(events: impl IntoIterator<Item = Event>) -> Result<LogSummary, String> {
        let mut s = Summariser::default();

        for event in events {
            s.add(event)?;
        }

        Ok(s.summary)
    }

    /// The simulated designers, as `brut-breed` builds them for circuits.
    fn designers() -> ModelPool {
        let mut pool = ModelPool::new();

        for m in [SimDesigner::small(), SimDesigner::large()] {
            let tag = m.tag().to_string();

            pool.insert(&tag, Box::new(m));
        }

        pool
    }

    /// A short circuit run in memory with these `soul` settings and its events.
    fn short_run(soul: Option<SoulConfig>, crossover_rate: f64) -> Vec<Event> {
        let cfg = BreedConfig {
            seed: 41,
            generations: 12,
            children_per_generation: 6,
            search_puzzles: 11,
            select_puzzles: 11,
            max_calls: 200_000,
            crossover_rate,
            domain: DomainConfig::Circuits(CircuitConfig::default()),
            soul,
            ..BreedConfig::default()
        };
        let pool = designers();
        let (_, events, _) =
            breed_to_end_in::<CircuitDomain>(cfg, circuit_pool(), &pool, None).expect("run");

        events
    }

    /// The ids of the members in the `born` events of a log.
    fn born_ids(events: &[Event]) -> Vec<String> {
        events
            .iter()
            .filter_map(|e| match e {
                Event::Born { id, .. } => Some(id.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn two_short_runs_built_to_agree_never_diverge() {
        // An S run and the same run with another κ at the seeds, while Observations and
        // imprinting are off: κ acts nowhere else.
        let quiet = SoulConfig {
            observe_vote: false,
            observe_peer: false,
            imprint: false,
            ..SoulConfig::arm_s()
        };
        let a = short_run(Some(quiet.clone()), 0.5);
        let b = short_run(
            Some(SoulConfig {
                kappa_seed: 700,
                ..quiet
            }),
            0.5,
        );
        let (sa, sb) = (
            summarise(a.clone()).expect("a"),
            summarise(b.clone()).expect("b"),
        );

        assert_eq!(sa.newborns.len(), 13);
        assert_eq!(divergence(&sa.newborns, &sb.newborns), None);
        // The born events differ from generation 0 on: their genomes carry κ.
        assert_ne!(born_ids(&a)[0], born_ids(&b)[0]);

        // An S and a V run with the complex vote and the simulator off: the readout acts
        // nowhere else, since Observations and imprinting move the same hidden angles.
        let mute = |soul: SoulConfig| SoulConfig {
            vote: false,
            simulator: false,
            ..soul
        };
        let s = summarise(short_run(Some(mute(SoulConfig::arm_s())), 0.5)).expect("s");
        let v = summarise(short_run(Some(mute(SoulConfig::arm_v())), 0.5)).expect("v");

        assert_eq!(divergence(&s.newborns, &v.newborns), None);
        // The state lines still differ: the contrast of independent Souls is K_C in one arm
        // and K_R in the other.
        let independent = |l: &LogSummary| {
            l.souls[0]
                .as_ref()
                .and_then(|g| g.contrast_independent)
                .expect("a contrast")
        };

        assert_ne!(independent(&s), independent(&v));

        // A positive control: another crossover rate changes the children of generation 1.
        let other = summarise(short_run(Some(SoulConfig::arm_s()), 0.2)).expect("other");
        let plain = summarise(short_run(Some(SoulConfig::arm_s()), 0.5)).expect("plain");

        assert_eq!(divergence(&plain.newborns, &other.newborns), Some(1));
    }

    #[test]
    fn the_summary_reads_newborns_states_and_votes() {
        let events = short_run(Some(SoulConfig::arm_s()), 0.5);
        let summary = summarise(events.clone()).expect("summary");
        let mut votes = VoteTally::default();
        let mut born = 0;

        for e in &events {
            if let Event::Born {
                generation,
                cascade,
                votes: Some(tally),
                ..
            } = e
            {
                born += 1;
                votes.complex += tally.complex;
                votes.flipped += tally.flipped;
                votes.tied += tally.tied;
                votes.sim_decided += tally.sim_decided;
                votes.coherence_sum += tally.coherence_sum;

                assert!(
                    summary.newborns[*generation]
                        .iter()
                        .any(|n| n.gene_id == cascade.gene_id())
                );
            }
        }

        assert_eq!(summary.newborns.iter().map(Vec::len).sum::<usize>(), born);
        assert_eq!(summary.newborns[0].len(), 4);
        assert_eq!(summary.votes, votes);
        assert_eq!(summary.souls.len(), 13);
        assert!(summary.souls.iter().all(Option::is_some));

        // Without Souls there is no soul group and no vote tally.
        let plain = summarise(short_run(None, 0.5)).expect("plain");

        assert!(plain.souls.iter().all(Option::is_none));
        assert_eq!(plain.votes, VoteTally::default());

        // A parent that the log has not born is refused.
        let orphan = events
            .iter()
            .find(|e| matches!(e, Event::Born { parents, .. } if !parents.is_empty()))
            .cloned()
            .expect("a bred member");

        assert!(summarise([orphan]).is_err());
    }

    #[test]
    fn a_run_that_bred_more_than_its_config_plans_is_refused() {
        assert_eq!(bred_as_planned(ALL_GENERATIONS, 1001, 1000), Ok(()));
        assert_eq!(bred_as_planned(CALL_BUDGET, 150, 1000), Ok(()));
        assert!(bred_as_planned(ALL_GENERATIONS, 1002, 1000).is_err());
        assert!(bred_as_planned(ALL_GENERATIONS, 150, 1000).is_err());
        assert!(bred_as_planned(CALL_BUDGET, 1001, 1000).is_err());
        assert!(bred_as_planned("interrupted", 150, 1000).is_err());
    }

    /// A fresh, empty directory for one test.
    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brut-soulcmp-{name}-{}", std::process::id()));

        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");

        dir
    }

    fn write_json<T: serde::Serialize>(path: &Path, value: &T) {
        std::fs::write(path, serde_json::to_string(value).expect("json")).expect("write");
    }

    /// A finished run in `dir` with the files this program reads: `cfg`, a report that
    /// completed every generation, a finished checkpoint, `found`, an event log with one seed
    /// Cascade born in each generation, and two timings of 30 s.
    fn write_run(dir: &Path, cfg: &BreedConfig, found: &Rediscovery) {
        let report = Report {
            generations_completed: cfg.generations + 1,
            archive_size: cfg.generations + 1,
            calls_used: 1234,
            mutator_answers_used: 0,
            comparisons: 0,
            stop_reason: ALL_GENERATIONS.to_string(),
            top: Vec::new(),
            finalists: Vec::new(),
            best_finalist_select_rate: 0.0,
            best_seed_select_rate: 0.0,
        };
        let seeds = (circuit_pool().seeds)();
        let mut log = String::new();

        for g in 0..=cfg.generations {
            let (_, cascade) = &seeds[g % seeds.len()];
            let event = Event::Born {
                generation: g,
                id: format!("member{g}"),
                parents: Vec::new(),
                op: Op::Seed,
                cascade: cascade.clone(),
                solved: 0,
                puzzles: 33,
                usage: Usage::default(),
                quality: Some(0.0),
                answers: None,
                imprinted: None,
                soul_sources: None,
                teaching: None,
                authors: None,
                votes: None,
            };

            log.push_str(&serde_json::json!({ "event": event }).to_string());
            log.push('\n');
        }

        std::fs::create_dir_all(dir).expect("run dir");
        write_json(&dir.join("config.json"), cfg);
        write_json(&dir.join("report.json"), &report);
        write_json(
            &dir.join("snapshot.json"),
            &serde_json::json!({"snapshot": {"finished": true}, "log_len": 0}),
        );
        write_json(&dir.join("rediscovery.json"), found);
        std::fs::write(dir.join("events.jsonl"), log).expect("log");
        std::fs::write(
            dir.join("timings.jsonl"),
            "{\"generation\":0,\"seconds\":30.0}\n{\"generation\":1,\"seconds\":30.0}\n",
        )
        .expect("timings");
    }

    /// The arguments for these arms and directories.
    fn arm_args(arms: &[(Arm, &[&PathBuf])]) -> Vec<String> {
        let mut args = Vec::new();

        for (arm, dirs) in arms {
            args.push(arm.flag().to_string());
            args.extend(dirs.iter().map(|d| d.display().to_string()));
        }

        args
    }

    #[test]
    fn run_directories_must_have_the_settings_of_their_arm() {
        let root = tmp_dir("runs");
        let base = SoulConfig::arm_s();
        // A finished run of `arm` on `seed` with 2 generations; arm S passes in generation 1.
        let run_dir = |name: &str, arm: Arm, seed: u64, cfg: Option<BreedConfig>| {
            let dir = root.join(name);
            let cfg = cfg.unwrap_or_else(|| arm_config(&base, arm, seed, 2, 13_000_000));
            let t = if arm == Arm::S { Some(1) } else { None };

            write_run(&dir, &cfg, &passing(t));
            dir
        };
        let s = run_dir("S-901", Arm::S, 901, None);
        let v = run_dir("V-901", Arm::V, 901, None);
        let k = run_dir("K-901", Arm::K, 901, None);
        let e = run_dir("E-901", Arm::E, 901, None);
        let n = run_dir("N-901", Arm::N, 901, None);

        // The settings of each arm: the numbers come, but no claim.
        let text = run(arm_args(&[
            (Arm::S, &[&s]),
            (Arm::V, &[&v]),
            (Arm::K, &[&k]),
            (Arm::E, &[&e]),
            (Arm::N, &[&n]),
        ]))
        .expect("run");

        assert!(
            text.contains("1 seed triple(s); every run: --generations 2, --max-calls 13000000.\n"),
            "{text}"
        );
        assert!(
            text.contains("No claim: the seeds are pilot seeds"),
            "{text}"
        );
        assert!(
            text.contains("  901      1  never  never  S first  S first"),
            "{text}"
        );
        assert!(text.contains("brut-breed --verify on every run"), "{text}");

        // A run of another arm, or with any other setting, is refused.
        let wider = BreedConfig {
            search_puzzles: 9,
            ..arm_config(&base, Arm::V, 901, 2, 13_000_000)
        };
        let kappa = BreedConfig {
            soul: Some(SoulConfig {
                kappa_seed: 300,
                ..SoulConfig::arm_s()
            }),
            ..arm_config(&base, Arm::S, 901, 2, 13_000_000)
        };
        let v_wider = run_dir("V-901-wider", Arm::V, 901, Some(wider));
        let s_kappa = run_dir("S-901-kappa", Arm::S, 901, Some(kappa));

        for (args, why) in [
            (
                arm_args(&[(Arm::S, &[&v]), (Arm::V, &[&v]), (Arm::K, &[&k])]),
                "not those of arm S",
            ),
            (
                arm_args(&[(Arm::S, &[&s]), (Arm::V, &[&v_wider]), (Arm::K, &[&k])]),
                "not those of arm V",
            ),
            (
                arm_args(&[(Arm::S, &[&s_kappa]), (Arm::V, &[&v]), (Arm::K, &[&k])]),
                "not those of arm S",
            ),
            (
                arm_args(&[(Arm::S, &[&s]), (Arm::V, &[&v]), (Arm::K, &[&n])]),
                "not those of arm K",
            ),
        ] {
            let refused = run(args).expect_err(why);

            assert!(refused.contains(why), "{refused}");
        }

        // With --preset, the settings of arm S come from the file, and no claim is made.
        let preset = root.join("preset.json");

        write_json(
            &preset,
            &SoulConfig {
                kappa_seed: 300,
                ..SoulConfig::arm_s()
            },
        );

        let mut args = arm_args(&[(Arm::S, &[&s_kappa]), (Arm::V, &[&v]), (Arm::K, &[&k])]);

        args.extend(["--preset".to_string(), preset.display().to_string()]);

        let refused = run(args).expect_err("arm V has kappa_seed 200");

        assert!(refused.contains("not those of arm V"), "{refused}");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_seed_needs_a_run_in_each_arm_of_the_triple() {
        let root = tmp_dir("triples");
        let base = SoulConfig::arm_s();
        let run_dir = |arm: Arm, seed: u64| {
            let dir = root.join(format!("{arm}-{seed}"));

            write_run(
                &dir,
                &arm_config(&base, arm, seed, 2, 13_000_000),
                &passing(None),
            );
            dir
        };
        let (s1, v1, k1) = (
            run_dir(Arm::S, 1002),
            run_dir(Arm::V, 1002),
            run_dir(Arm::K, 1002),
        );
        let (s2, v2) = (run_dir(Arm::S, 1003), run_dir(Arm::V, 1003));
        let e3 = run_dir(Arm::E, 1004);

        // Arm K misses seed 1003.
        let refused = run(arm_args(&[
            (Arm::S, &[&s1, &s2]),
            (Arm::V, &[&v1, &v2]),
            (Arm::K, &[&k1]),
        ]))
        .expect_err("a missing arm");

        assert!(
            refused.contains("seed 1003 has no run in arm K"),
            "{refused}"
        );

        // An arm E run on a seed without a triple.
        let refused = run(arm_args(&[
            (Arm::S, &[&s1]),
            (Arm::V, &[&v1]),
            (Arm::K, &[&k1]),
            (Arm::E, &[&e3]),
        ]))
        .expect_err("an extra seed");

        assert!(
            refused.contains("seed 1004 has no run in arm S"),
            "{refused}"
        );

        // A seed twice in one arm.
        let refused = run(arm_args(&[
            (Arm::S, &[&s1, &s1]),
            (Arm::V, &[&v1]),
            (Arm::K, &[&k1]),
        ]))
        .expect_err("two runs");

        assert!(
            refused.contains("seed 1002 has two runs in arm S"),
            "{refused}"
        );

        // The counted seed 1002 alone: the rule goes on, at 2 generations no claim.
        let text = run(arm_args(&[
            (Arm::S, &[&s1]),
            (Arm::V, &[&v1]),
            (Arm::K, &[&k1]),
        ]))
        .expect("one triple");

        assert!(
            text.contains("No claim: the runs have --generations 2"),
            "{text}"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn more_triples_than_n_max_are_refused() {
        let seeds: Vec<u64> = (FIRST_SEED..).take(MAX_TRIPLES + 1).collect();
        let all = triples(&seeds, &outcomes(&vec![('t', 't'); MAX_TRIPLES + 1]));

        assert!(check_seeds(&seeds).is_err());
        assert!(render_rule(&Context::default(), SeedSet::Counted, &all, p34()).is_err());
    }

    #[test]
    fn arguments_name_the_arms() {
        let args = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let parsed = parse_args(args(&[
            "--s", "a", "b", "--v", "c", "--k", "d", "--n", "e", "--preset", "p.json",
        ]))
        .expect("arguments");

        assert_eq!(
            parsed.dirs[&Arm::S],
            vec![PathBuf::from("a"), PathBuf::from("b")]
        );
        assert_eq!(parsed.dirs[&Arm::N], vec![PathBuf::from("e")]);
        assert!(!parsed.dirs.contains_key(&Arm::E));
        assert_eq!(parsed.preset, Some(PathBuf::from("p.json")));

        assert!(parse_args(args(&["x", "--s", "a", "--v", "b", "--k", "c"])).is_err());
        assert!(parse_args(args(&["--s", "a", "--v", "b"])).is_err());
        assert!(parse_args(args(&["--s", "a", "--v", "b", "--k", "c", "--r", "d"])).is_err());
        assert!(parse_args(args(&["--s", "a", "--v", "b", "--k", "c", "--preset"])).is_err());
    }
}
