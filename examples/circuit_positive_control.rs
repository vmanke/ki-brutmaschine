//! The frozen analysis of the step 1 experiment (plan (g) "Pre-registration", ADR 0006,
//! docs/PREREG_STEP1_CIRCUITS.md): does circuit breeding solve the targets the seed
//! Cascades miss, and does Darwin Gödel selection beat a uniform choice of parents?
//!
//! It reads finished circuit runs of `brut-breed` from two arms, paired by seed: arm A
//! (`--domain circuits --search 33 --select 22 --children 8`, every other setting at its
//! default) and arm R (the same plus `--selection uniform`). Each run must be finished and
//! must have the settings of its arm, and all runs must share their generations and their
//! call cap. From each run it reads `rediscovery.json`.
//!
//! D(s) = |R_bred(s) \ R_seed(s)| / |L \ R_seed(s)|: the share of the targets the seed
//! Cascades miss that a bred Cascade solves, exactly and at the reference cost, on at
//! least one of the target's dressed search tasks. The pass time T of a run is the first
//! generation in which D > 1/2: the k-th smallest first generation of the targets that
//! only bred Cascades solve, with k = |L \ R_seed| / 2 + 1, rounded down. T is never when
//! the run has no such generation up to generation 1000. Every decision is made in
//! integers.
//!
//! The sequential rule (section 8 of the pre-registration, ADR 0006 point 5) reads the
//! pairs in seed order. In each pair X = 1 when arm A passes, and the arm with the smaller
//! T passes first; equal T, also never in both arms, is a tie. Two evidence values start
//! at 1. E1 is multiplied by 3/2 when X = 1 and by 1/2 when X = 0. E2 is multiplied by 3/2
//! when arm A passes first and by 1/2 when arm R does; a tie leaves it unchanged. A claim
//! is proven when its E first reaches 20 and fails when it first falls to 1/20, and this
//! first crossing is final. The primary claim (ticket 03: more than half of all seeds
//! pass) rests on E1. The secondary claim (selection: among untied pairs, arm A passes
//! first more often than arm R) rests on E2 and counts only when the primary is proven.
//! The rule stops after the first pair at which the primary fails, or at which the primary
//! is proven and E2 has crossed, and at the latest after 60 pairs. A claim still open then
//! is undecided.
//!
//! It prints, per seed s, R_seed(s), D_A(s), D_R(s) and the calibration gate
//! |R_seed(s)| <= |L| / 3, and then T_A, T_R, X, the first arm, E1 and E2. Pairs after the
//! stop are listed as not counted. Then it prints E1 and E2 after the counted pairs, and
//! either the verdicts on both claims at the stop or the next seed to run. The void
//! condition it can check is a seed up to the stop that fails the calibration gate. The
//! descriptive lines carry no verdict. Over the counted pairs they give the median of D_A;
//! the T-only variant, where a target counts when an exact answer has at most the
//! reference's T count, and also at most its CNOT count for a Clifford target whose
//! reference has no T, with D_A, D_R and the number of seeds with D_A > 1/2 under it; the
//! first generation in which a bred Cascade solved each target; and the cheapest
//! invention candidate per target and run. They also name the runs that ended at the call
//! cap.
//!
//! It makes a claim only for the counted seeds 1, 2, 3, ... without gaps, at most 60, in
//! both arms, run with the pre-registered `--generations 1000` and `--max-calls 13000000`,
//! with no run ended at the call cap. For other seeds, such as the pilots, or for other
//! generations or caps, it prints the numbers and says that it makes no claim. Run
//! `brut-breed --out <dir> --verify` on every directory first: this program checks each
//! count against its own target list, not the hash chain.
//!
//! Usage:
//!   cargo run --example circuit_positive_control -- --a <dir>... --r <dir>...
#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;

use brutmaschine::cascade::arena::{BreedConfig, ParentSelection, Report};
use brutmaschine::cascade::circuit::{Rediscovery, TargetOutcome, verified};
use brutmaschine::cascade::domain::{CircuitConfig, DomainConfig};
use brutmaschine::cascade::qubit::Cost;

/// The generations of every counted Run (docs/PREREG_STEP1_CIRCUITS.md, section 6). They
/// are also the ceiling of the pass time: a run that passes only later has not passed.
const GENERATIONS: usize = 1000;

/// The call cap of every counted Run, which no counted Run can reach (section 6).
const MAX_CALLS: u64 = 13_000_000;

/// The most seed pairs the experiment counts, its pre-registered cap. Every evidence test
/// fits a u128, since 20 * 3^60 < 2^128.
const MAX_PAIRS: usize = 60;

/// A claim is proven when its evidence reaches EVIDENCE, and it fails when its evidence
/// falls to 1 / EVIDENCE.
const EVIDENCE: u128 = 20;

/// How a run ends when it completes every generation.
const ALL_GENERATIONS: &str = "all generations done";

fn main() {
    match run(std::env::args().skip(1).collect()) {
        Ok(text) => print!("{text}"),
        Err(e) => {
            eprintln!("circuit_positive_control: {e}");
            std::process::exit(1);
        }
    }
}

/// The two arms of the experiment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Arm {
    /// Darwin Gödel selection.
    A,
    /// Uniform parent choice, the control.
    R,
}

impl fmt::Display for Arm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Arm::A => write!(f, "A"),
            Arm::R => write!(f, "R"),
        }
    }
}

/// The settings a run of `arm` must have, given its seed, generations and call cap: the
/// circuit domain on every verified target (L), 33 search and 22 select tasks, 8 children
/// per generation, and every other setting at its default.
fn arm_config(arm: Arm, seed: u64, generations: usize, max_calls: u64) -> BreedConfig {
    BreedConfig {
        seed,
        generations,
        children_per_generation: 8,
        search_puzzles: 33,
        select_puzzles: 22,
        max_calls,
        domain: DomainConfig::Circuits(CircuitConfig::default()),
        selection: match arm {
            Arm::A => ParentSelection::Dgm,
            Arm::R => ParentSelection::Uniform,
        },
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

    fn compare(self, other: Ratio) -> Ordering {
        (self.num * other.den).cmp(&(other.num * self.den))
    }

    /// The mean of two fractions.
    fn midpoint(self, other: Ratio) -> Ratio {
        Ratio::new(
            self.num * other.den + other.num * self.den,
            2 * self.den * other.den,
        )
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

/// D of one run under one rule: of the `open` targets the seed Cascades miss, a bred
/// Cascade solves `hit`.
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

/// When a target counts as solved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    /// The pre-registered rule: exact, at most the reference's T and CNOT counts.
    Full,
    /// Exact, at most the reference's T count; the CNOT count decides only where the
    /// reference has no T gate.
    TOnly,
}

/// True when the cheapest exact answer `best` solves `t` under the T-only rule. `best` is
/// cheapest by T first and then CNOT, so it decides both counts.
fn t_only(t: &TargetOutcome, best: Option<Cost>) -> bool {
    let r = &t.reference;

    best.is_some_and(|c| c.t <= r.t && (r.t > 0 || c.cnot <= r.cnot))
}

/// One run under one rule: the targets its seed Cascades solve, |L| and D.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Counted {
    seeds: BTreeSet<String>,
    library: usize,
    share: Share,
}

impl Counted {
    /// The calibration gate |R_seed| <= |L| / 3.
    fn calibrated(&self) -> bool {
        3 * self.seeds.len() <= self.library
    }
}

fn count(found: &Rediscovery, rule: Rule) -> Counted {
    let mut seeds = BTreeSet::new();
    let mut hit = 0;

    for t in &found.targets {
        let (by_seed, by_bred) = match rule {
            Rule::Full => (t.reached_by_seed, t.reached_by_bred),
            Rule::TOnly => (t_only(t, t.best_seed), t_only(t, t.best_bred)),
        };

        if by_seed {
            seeds.insert(t.target.clone());
        } else if by_bred {
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

/// Error unless the counts of `found` agree with its own target list, and its targets are
/// the verified library targets in library order: L of the pre-registration. A target must
/// also have a first generation exactly when a bred Cascade solved it, since the pass time
/// is read from these.
fn check_counts(found: &Rediscovery) -> Result<(), String> {
    let names: Vec<&str> = found.targets.iter().map(|t| t.target.as_str()).collect();
    let library: Vec<&str> = verified().iter().map(|t| t.name).collect();
    let counted = count(found, Rule::Full);

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
        && found.calibrated == counted.calibrated()
        && found.passes == (counted.calibrated() && counted.share.above_half())
        && found
            .targets
            .iter()
            .all(|t| t.first_generation.is_some() == t.reached_by_bred);

    if !agrees {
        return Err("its counts disagree with its own target list".to_string());
    }

    Ok(())
}

/// The pass time T of a run: the first generation g with
/// 2 |R_bred(<= g) \ R_seed| > |L \ R_seed|, that is D > 1/2. It is the k-th smallest first
/// generation of the targets that only bred Cascades solve, with k = |L \ R_seed| / 2 + 1,
/// rounded down. None when fewer than k such targets exist, as when the seed Cascades
/// solve every target.
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

/// The two runs of one seed.
#[derive(Clone, Debug)]
struct Pair {
    seed: u64,
    a: Rediscovery,
    r: Rediscovery,
}

/// One seed under one rule.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Row {
    seed: u64,
    library: usize,
    /// R_seed: the same in both arms, since generation 0 does not depend on selection.
    seeds: BTreeSet<String>,
    a: Share,
    r: Share,
    /// The pass time of arm A within the ceiling, None for never. The rows of both rules
    /// carry the pass time of the pre-registered rule, since `rediscovery.json` holds first
    /// generations only for that rule.
    t_a: Option<usize>,
    /// The pass time of arm R, in the same way.
    t_r: Option<usize>,
}

impl Row {
    fn calibrated(&self) -> bool {
        3 * self.seeds.len() <= self.library
    }
}

/// The statistics of one rule over the given pairs.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Statistic {
    rows: Vec<Row>,
    /// Seeds with D_A > 1/2.
    above_half: usize,
    /// The median of the defined D_A values: the middle one, or the mean of the two
    /// middle ones.
    median: Option<Ratio>,
}

fn statistic(pairs: &[Pair], rule: Rule) -> Result<Statistic, String> {
    let mut rows = Vec::new();
    // A run that passes only after the ceiling has not passed.
    let within = |t: Option<usize>| t.filter(|&g| g <= GENERATIONS);

    for p in pairs {
        let (a, r) = (count(&p.a, rule), count(&p.r, rule));

        if a.seeds != r.seeds || a.library != r.library {
            return Err(format!(
                "seed {}: the arms disagree on the targets the seed Cascades solve, so the runs are not a pair",
                p.seed
            ));
        }

        rows.push(Row {
            seed: p.seed,
            library: a.library,
            seeds: a.seeds,
            a: a.share,
            r: r.share,
            t_a: within(pass_time(&p.a)),
            t_r: within(pass_time(&p.r)),
        });
    }

    let above_half = rows.iter().filter(|r| r.a.above_half()).count();
    let shares: Vec<Ratio> = rows.iter().filter_map(|r| r.a.ratio()).collect();

    Ok(Statistic {
        median: median(&shares),
        rows,
        above_half,
    })
}

/// The middle value, or the mean of the two middle values; None for no values.
fn median(values: &[Ratio]) -> Option<Ratio> {
    let mut sorted = values.to_vec();

    sorted.sort_by(|a, b| a.compare(*b));

    let n = sorted.len();

    match n {
        0 => None,
        _ if n % 2 == 1 => Some(sorted[n / 2]),
        _ => Some(sorted[n / 2 - 1].midpoint(sorted[n / 2])),
    }
}

/// The arm whose run passes first in a pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum First {
    A,
    R,
    /// Both runs pass in the same generation, or neither passes.
    Tie,
}

impl fmt::Display for First {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            First::A => "A",
            First::R => "R",
            First::Tie => "tie",
        })
    }
}

/// The arm with the smaller pass time, where never comes after every generation.
fn first(t_a: Option<usize>, t_r: Option<usize>) -> First {
    match (t_a, t_r) {
        (Some(a), Some(r)) => match a.cmp(&r) {
            Ordering::Less => First::A,
            Ordering::Greater => First::R,
            Ordering::Equal => First::Tie,
        },
        (Some(_), None) => First::A,
        (None, Some(_)) => First::R,
        (None, None) => First::Tie,
    }
}

/// An evidence value E = 3^up / 2^n after n steps, `up` of them by 3/2 and the rest by
/// 1/2: the likelihood ratio of a rate of 3/4 against 1/2. The default is the start, E = 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Evidence {
    up: u32,
    n: u32,
}

impl Evidence {
    /// E after one more step: by 3/2 when `up`, else by 1/2.
    fn step(self, up: bool) -> Evidence {
        Evidence {
            up: self.up + u32::from(up),
            n: self.n + 1,
        }
    }

    /// E >= EVIDENCE, in integers: 3^up >= EVIDENCE * 2^n.
    fn proven(self) -> bool {
        3u128.pow(self.up) >= EVIDENCE * (1u128 << self.n)
    }

    /// E <= 1 / EVIDENCE, in integers: EVIDENCE * 3^up <= 2^n.
    fn failed(self) -> bool {
        EVIDENCE * 3u128.pow(self.up) <= 1u128 << self.n
    }

    /// E as an exact fraction.
    fn ratio(self) -> Ratio {
        Ratio::new(3u128.pow(self.up), 1u128 << self.n)
    }
}

/// The verdict on one claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    /// Its E reached EVIDENCE at this pair.
    Proven(usize),
    /// Its E fell to 1 / EVIDENCE at this pair.
    Failed(usize),
    /// The rule stopped after MAX_PAIRS pairs before its E crossed: no claim.
    Undecided,
    /// The secondary claim when the rule stopped without the primary proven.
    NotTested,
    /// The rule goes on, and its E has not crossed.
    Open,
}

impl Verdict {
    /// The verdict after a pair at which its E is `e`. An open claim is decided at the
    /// first crossing, and that decision is final.
    fn update(self, e: Evidence, pair: usize) -> Verdict {
        match self {
            Verdict::Open if e.proven() => Verdict::Proven(pair),
            Verdict::Open if e.failed() => Verdict::Failed(pair),
            decided => decided,
        }
    }

    /// The verdict at the stop: a claim still open is undecided.
    fn at_stop(self) -> Verdict {
        match self {
            Verdict::Open => Verdict::Undecided,
            decided => decided,
        }
    }
}

/// The sequential rule over the pairs in seed order.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Sequential {
    /// E1 and E2 after each counted pair.
    evidence: Vec<(Evidence, Evidence)>,
    /// The pair after which the rule stops; None while it goes on.
    stop: Option<usize>,
    /// The primary claim, decided by E1.
    primary: Verdict,
    /// The secondary claim, decided by E2. While the rule goes on it shows where E2
    /// crossed; at the stop it is NotTested unless the primary is proven.
    secondary: Verdict,
}

impl Sequential {
    /// The pairs that count: those up to the stop, or all while the rule goes on.
    fn counted(&self) -> usize {
        self.evidence.len()
    }
}

/// The sequential rule over `rows` in seed order. E1 and E2 start at 1. E1 is multiplied by
/// 3/2 when arm A passes and by 1/2 when it does not. E2 is multiplied by 3/2 when arm A
/// passes first and by 1/2 when arm R does; a tie leaves it unchanged. The rule stops after
/// the first pair at which the primary fails, or at which the primary is proven and E2 has
/// crossed, and at the latest after MAX_PAIRS pairs. The rows after the stop do not count.
fn sequential(rows: &[Row]) -> Sequential {
    let (mut e1, mut e2) = (Evidence::default(), Evidence::default());
    let (mut primary, mut secondary) = (Verdict::Open, Verdict::Open);
    let mut evidence = Vec::new();

    for (pair, row) in (1..).zip(rows) {
        e1 = e1.step(row.t_a.is_some());

        match first(row.t_a, row.t_r) {
            First::A => e2 = e2.step(true),
            First::R => e2 = e2.step(false),
            First::Tie => {}
        }

        evidence.push((e1, e2));
        primary = primary.update(e1, pair);
        secondary = secondary.update(e2, pair);

        let stops = matches!(primary, Verdict::Failed(_))
            || (matches!(primary, Verdict::Proven(_)) && secondary != Verdict::Open)
            || pair == MAX_PAIRS;

        if stops {
            let primary = primary.at_stop();
            // A fixed sequence: the secondary is tested only when the primary is proven.
            let secondary = match primary {
                Verdict::Proven(_) => secondary.at_stop(),
                _ => Verdict::NotTested,
            };

            return Sequential {
                evidence,
                stop: Some(pair),
                primary,
                secondary,
            };
        }
    }

    Sequential {
        evidence,
        stop: None,
        primary,
        secondary,
    }
}

/// Whether the runs may support a claim.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Claim {
    /// The counted seeds, calibrated up to the stop: the rule decides.
    Made,
    /// The counted seeds, but these seeds up to the stop fail the calibration gate: void.
    Void(Vec<u64>),
    /// Not the counted seeds: why no claim is made.
    Withheld(String),
}

/// The claim the rows support. They must be the counted seeds 1, 2, 3, ... without gaps,
/// at most MAX_PAIRS of them; a seed up to the stop that fails the calibration gate voids
/// the experiment.
fn claim(rows: &[Row]) -> Claim {
    let seeds: Vec<u64> = rows.iter().map(|r| r.seed).collect();

    if seeds.len() > MAX_PAIRS {
        return Claim::Withheld(format!(
            "{} seed pairs, more than the {MAX_PAIRS} the pre-registration allows",
            seeds.len()
        ));
    }

    if !seeds.iter().copied().eq(1..=seeds.len() as u64) {
        return Claim::Withheld(format!(
            "the seeds are {seeds:?}, not the counted seeds 1, 2, 3, ... without gaps"
        ));
    }

    // Only the pairs up to the stop count, so only they can void the experiment.
    let void: Vec<u64> = rows
        .iter()
        .take(sequential(rows).counted())
        .filter(|r| !r.calibrated())
        .map(|r| r.seed)
        .collect();

    if void.is_empty() {
        Claim::Made
    } else {
        Claim::Void(void)
    }
}

/// The claim the runs support, given what they share: `claim` of their rows, withheld
/// unless every run has the pre-registered generations and call cap and none ended at the
/// cap. A rule decided on another length or cap is not the pre-registered experiment.
fn claim_in(context: &Context, rows: &[Row]) -> Claim {
    let made = claim(rows);

    if matches!(made, Claim::Withheld(_)) {
        return made;
    }

    if (context.generations, context.max_calls) != (GENERATIONS, MAX_CALLS) {
        return Claim::Withheld(format!(
            "the runs have --generations {} and --max-calls {}, not the pre-registered {GENERATIONS} and {MAX_CALLS}",
            context.generations, context.max_calls
        ));
    }

    if !context.capped.is_empty() {
        return Claim::Withheld(format!(
            "{} run(s) ended at the call cap, which the pre-registered cap rules out",
            context.capped.len()
        ));
    }

    made
}

/// The verdicts on both claims at the stop, or the next seed while the rule goes on; only
/// the reason when the runs support no claim.
fn verdict(claim: &Claim, rule: &Sequential) -> String {
    match (claim, rule.stop) {
        (Claim::Void(_), _) => "No claim: the experiment is void.\n".to_string(),
        (Claim::Withheld(why), _) => format!("No claim: {why}.\n"),
        (Claim::Made, None) => format!("OPEN: run seed {} next.\n", rule.counted() + 1),
        (Claim::Made, Some(_)) => {
            let primary = verdict_line(
                "Primary (ticket 03)",
                rule.primary,
                &format!("More than half of all seeds pass within {GENERATIONS} generations."),
                "The data speak against a pass rate of 3/4 or more; they do not prove a rate of 1/2 or less.",
            );
            let secondary = verdict_line(
                "Secondary (selection)",
                rule.secondary,
                "Among untied pairs, arm A passes first more often than arm R.",
                "The data speak against arm A passing first in 3/4 or more of the untied pairs.",
            );

            format!("{primary}{secondary}")
        }
    }
}

/// The verdict on one claim at the stop, with what it means when proven or failed.
fn verdict_line(name: &str, verdict: Verdict, proven: &str, failed: &str) -> String {
    match verdict {
        Verdict::Proven(pair) => format!("{name}: PROVEN at pair {pair}. {proven}\n"),
        Verdict::Failed(pair) => format!("{name}: FAILED at pair {pair}. {failed}\n"),
        Verdict::Undecided => format!("{name}: UNDECIDED after {MAX_PAIRS} pairs, no claim.\n"),
        Verdict::NotTested => format!("{name}: NOT TESTED, since the primary is not proven.\n"),
        Verdict::Open => format!("{name}: open.\n"),
    }
}

/// What the runs share besides their results.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Context {
    generations: usize,
    max_calls: u64,
    /// Runs that ended at the call cap: seed, arm and generations completed.
    capped: Vec<(u64, Arm, usize)>,
}

impl Default for Context {
    /// The context of the pre-registered Runs: their generations and call cap, and no run
    /// at the cap.
    fn default() -> Self {
        Context {
            generations: GENERATIONS,
            max_calls: MAX_CALLS,
            capped: Vec::new(),
        }
    }
}

fn show(share: Share) -> String {
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

/// A pass time as text: the generation, or never.
fn show_time(t: Option<usize>) -> String {
    t.map_or("never".to_string(), |g| g.to_string())
}

/// The per-seed table of one statistic: R_seed, D_A, D_R and the calibration gate.
fn render_table(out: &mut String, stat: &Statistic) {
    out.push_str(&format!(
        "{:>5}  {:>3}  {:<34}  {:<16}  {:<16}  calibrated\n",
        "seed", "|L|", "R_seed", "D_A", "D_R"
    ));

    for row in &stat.rows {
        let seeds = format!(
            "{}: {}",
            row.seeds.len(),
            row.seeds.iter().cloned().collect::<Vec<_>>().join(", ")
        );

        out.push_str(&format!(
            "{:>5}  {:>3}  {:<34}  {:<16}  {:<16}  {}\n",
            row.seed,
            row.library,
            seeds,
            show(row.a),
            show(row.r),
            if row.calibrated() { "yes" } else { "NO" }
        ));
    }
}

/// Where the E of one claim stands after `pairs` counted pairs, and where it first crossed.
fn evidence_line(name: &str, e: Evidence, verdict: Verdict, pairs: usize) -> String {
    let crossed = match verdict {
        Verdict::Proven(pair) => format!("; it reached {EVIDENCE} at pair {pair}"),
        Verdict::Failed(pair) => format!("; it fell to 1/{EVIDENCE} at pair {pair}"),
        Verdict::Open => format!("; open, it has reached neither {EVIDENCE} nor 1/{EVIDENCE}"),
        Verdict::Undecided => format!("; it reached neither {EVIDENCE} nor 1/{EVIDENCE}"),
        Verdict::NotTested => String::new(),
    };

    format!("{name} = {} after {pairs} pair(s){crossed}.\n", e.ratio())
}

/// The sequential rule pair by pair, the pairs after its stop, and where E1 and E2 stand.
/// `rows` are the counted rows, `after` the pairs after the stop.
fn render_rule(out: &mut String, rows: &[Row], rule: &Sequential, after: &[Pair]) {
    out.push_str(&format!(
        "\nSequential rule, in seed order: T is the first generation with D > 1/2, never when there is none by generation {GENERATIONS}; \
         X = 1 when arm A passes; first is the arm with the smaller T.\n"
    ));
    out.push_str(&format!(
        "E1 = 3^passes / 2^pairs and E2 = 3^(A first) / 2^(untied pairs). A claim is proven when its E first reaches {EVIDENCE} \
         and fails when it first falls to 1/{EVIDENCE}; the secondary counts only if the primary is proven.\n"
    ));
    out.push_str(&format!(
        "{:>5}  {:>5}  {:>5}  {}  {:<5}  {:<22}  {}\n",
        "seed", "T_A", "T_R", "X", "first", "E1", "E2"
    ));

    for (row, (e1, e2)) in rows.iter().zip(&rule.evidence) {
        out.push_str(&format!(
            "{:>5}  {:>5}  {:>5}  {}  {:<5}  {:<22}  {}\n",
            row.seed,
            show_time(row.t_a),
            show_time(row.t_r),
            u8::from(row.t_a.is_some()),
            first(row.t_a, row.t_r),
            e1.ratio().to_string(),
            e2.ratio()
        ));
    }

    for p in after {
        out.push_str(&format!("{:>5}  after the stop, not counted\n", p.seed));
    }

    if let Some(&(e1, e2)) = rule.evidence.last() {
        let n = rule.counted();

        out.push_str(&evidence_line("Primary (ticket 03): E1", e1, rule.primary, n));
        out.push_str(&evidence_line("Secondary (selection): E2", e2, rule.secondary, n));
    }

    match rule.stop {
        Some(stop) => out.push_str(&format!("The rule stops after pair {stop}.\n")),
        None => out.push_str(&format!("The rule goes on after pair {}.\n", rule.counted())),
    }
}

/// The whole analysis as text.
fn render(context: &Context, pairs: &[Pair]) -> Result<String, String> {
    if pairs.is_empty() {
        return Err("no seed pairs".to_string());
    }

    if pairs.len() > MAX_PAIRS {
        return Err(format!("more than {MAX_PAIRS} seed pairs"));
    }

    // The rule reads every pair in seed order and counts those up to its stop; the pairs
    // after the stop are only listed.
    let rows = statistic(pairs, Rule::Full)?.rows;
    let rule = sequential(&rows);
    let claim = claim_in(context, &rows);
    let (counted, after) = pairs.split_at(rule.counted());
    let full = statistic(counted, Rule::Full)?;
    let t_only = statistic(counted, Rule::TOnly)?;
    let mut out = String::new();

    out.push_str(
        "Step 1 circuit positive control: the frozen analysis of docs/PREREG_STEP1_CIRCUITS.md\n",
    );
    out.push_str(&format!(
        "{} seed pair(s); every run: --generations {}, --max-calls {}.\n\n",
        pairs.len(),
        context.generations,
        context.max_calls
    ));

    match &claim {
        Claim::Made => out.push_str(
            "Void conditions checked here: every counted seed passes the calibration gate |R_seed| <= |L| / 3.\n",
        ),
        Claim::Void(seeds) => out.push_str(&format!(
            "VOID: the seed Cascades of seed(s) {seeds:?} solve more than |L| / 3 targets; recalibrate the control.\n"
        )),
        Claim::Withheld(why) => out.push_str(&format!("No claim: {why}.\n")),
    }

    out.push_str(
        "Checked elsewhere: integrity alarms (a run with one has no rediscovery.json and is refused here), \
         equal logs at 1 and 4 threads (seed 1, both arms), and the calibration test at the frozen commit.\n\n",
    );
    out.push_str(
        "Pre-registered rule: a target counts when an exact answer reaches its T and CNOT reference.\n",
    );
    render_table(&mut out, &full);

    let median = full
        .median
        .map_or("undefined".to_string(), |m| m.to_string());

    out.push_str(&format!(
        "Median D_A over the {} counted pair(s), descriptive: {median}.\n",
        counted.len()
    ));
    render_rule(&mut out, &full.rows, &rule, after);
    out.push_str(&verdict(&claim, &rule));
    out.push_str(
        "\nT-only variant, descriptive with no verdict: a target counts when an exact answer reaches its T reference, and its CNOT reference where that T count is 0.\n",
    );
    render_table(&mut out, &t_only);
    out.push_str(&format!(
        "T-only variant: D_A > 1/2 in {} of {} counted seed(s).\n",
        t_only.above_half,
        counted.len()
    ));
    out.push_str(&descriptive(context, counted));

    Ok(out)
}

/// The descriptive lines: first generations, call-cap stops and the cheapest invention
/// candidate per target and run.
fn descriptive(context: &Context, pairs: &[Pair]) -> String {
    let mut out = String::from(
        "\nDescriptive: the first generation in which a bred Cascade solved each target, arm A|arm R (- never; * a target the seed Cascades solve).\n",
    );
    let first = |found: &Rediscovery, name: &str| {
        found
            .targets
            .iter()
            .find(|t| t.target == name)
            .map_or("?".to_string(), |t| {
                let generation = t
                    .first_generation
                    .map_or("-".to_string(), |g| g.to_string());

                format!("{generation}{}", if t.reached_by_seed { "*" } else { "" })
            })
    };

    out.push_str(&format!("{:<10}", "target"));

    for p in pairs {
        out.push_str(&format!("{:>10}", format!("s{}", p.seed)));
    }

    out.push('\n');

    for t in &pairs[0].a.targets {
        out.push_str(&format!("{:<10}", t.target));

        for p in pairs {
            let cell = format!("{}|{}", first(&p.a, &t.target), first(&p.r, &t.target));

            out.push_str(&format!("{cell:>10}"));
        }

        out.push('\n');
    }

    if context.capped.is_empty() {
        out.push_str("Runs that ended at the call cap: none.\n");
    } else {
        for (seed, arm, done) in &context.capped {
            out.push_str(&format!(
                "Run of seed {seed}, arm {arm} ended at the call cap after {done} generation(s).\n"
            ));
        }
    }

    let mut candidates = Vec::new();

    for p in pairs {
        for (arm, found) in [(Arm::A, &p.a), (Arm::R, &p.r)] {
            for t in &found.targets {
                if let Some(c) = &t.candidate {
                    candidates.push(format!(
                        "  seed {}, arm {arm}, {}: T {}, CNOT {} by {} (generation {}, task {}); literature checked: {}\n",
                        p.seed,
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

    if candidates.is_empty() {
        out.push_str("Invention candidates (below a BestKnown count): none.\n");
    } else {
        out.push_str("Invention candidates (below a BestKnown count; a human checks the literature before any claim, Q38):\n");

        for c in candidates {
            out.push_str(&c);
        }
    }

    out
}

/// One finished run, read from its directory.
struct Loaded {
    dir: PathBuf,
    cfg: BreedConfig,
    report: Report,
    found: Rediscovery,
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;

    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Ok when a run bred exactly the generations its config plans (generation 0 included),
/// or fewer when the call budget ended it. A resume may lower --generations below what a
/// run had already bred; such a run still verifies, but it is not a run of its config.
fn bred_as_planned(stop_reason: &str, completed: usize, generations: usize) -> Result<(), String> {
    let planned = generations + 1;
    let fits = match stop_reason {
        "all generations done" => completed == planned,
        _ => completed < planned,
    };

    if fits {
        Ok(())
    } else {
        Err(format!(
            "it bred {completed} generations (generation 0 included) and stopped with \
             {stop_reason:?}, but its config plans {planned}; a resume lowered --generations \
             below what it had bred"
        ))
    }
}

/// The finished run in `dir`, which must have the settings of `arm`.
fn load(dir: &Path, arm: Arm) -> Result<Loaded, String> {
    let bad = |m: String| format!("{}: {m}", dir.display());
    let cfg: BreedConfig = read_json(&dir.join("config.json"))?;

    if cfg != arm_config(arm, cfg.seed, cfg.generations, cfg.max_calls) {
        return Err(bad(format!(
            "its settings are not the pre-registered ones of arm {arm}"
        )));
    }

    let snapshot: serde_json::Value = read_json(&dir.join("snapshot.json"))?;
    let finished = snapshot["snapshot"]["finished"] == true;

    if !finished || !dir.join("report.json").exists() || !dir.join("rediscovery.json").exists() {
        return Err(bad(
            "the run is not finished; resume it, then run brut-breed --verify on it".to_string(),
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

    Ok(Loaded {
        dir: dir.to_path_buf(),
        cfg,
        report,
        found,
    })
}

/// The directories after --a and after --r.
fn parse_args(args: Vec<String>) -> Result<(Vec<PathBuf>, Vec<PathBuf>), String> {
    let usage = "usage: circuit_positive_control --a <dir>... --r <dir>...";
    let mut arms: BTreeMap<Arm, Vec<PathBuf>> = BTreeMap::new();
    let mut current = None;

    for arg in args {
        match arg.as_str() {
            "--a" => current = Some(Arm::A),
            "--r" => current = Some(Arm::R),
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg}; {usage}")),
            _ => match current {
                Some(arm) => arms.entry(arm).or_default().push(PathBuf::from(arg)),
                None => return Err(format!("{arg} belongs to no arm; {usage}")),
            },
        }
    }

    match (arms.remove(&Arm::A), arms.remove(&Arm::R)) {
        (Some(a), Some(r)) => Ok((a, r)),
        _ => Err(format!("both arms need run directories; {usage}")),
    }
}

/// The runs of one arm by seed; a seed given twice is refused.
fn load_arm(dirs: &[PathBuf], arm: Arm) -> Result<BTreeMap<u64, Loaded>, String> {
    let mut runs = BTreeMap::new();

    for dir in dirs {
        let run = load(dir, arm)?;
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

fn run(args: Vec<String>) -> Result<String, String> {
    let (a_dirs, r_dirs) = parse_args(args)?;
    let a = load_arm(&a_dirs, Arm::A)?;
    let mut r = load_arm(&r_dirs, Arm::R)?;

    if a.keys().ne(r.keys()) {
        return Err(format!(
            "the arms have different seeds: A {:?}, R {:?}",
            a.keys().collect::<Vec<_>>(),
            r.keys().collect::<Vec<_>>()
        ));
    }

    let mut context = Context::default();
    let mut shared = BTreeSet::new();
    let mut pairs = Vec::new();

    for (seed, run_a) in a {
        let Some(run_r) = r.remove(&seed) else {
            return Err(format!("seed {seed} has no run in arm R"));
        };

        for (arm, run) in [(Arm::A, &run_a), (Arm::R, &run_r)] {
            shared.insert((run.cfg.generations, run.cfg.max_calls));

            if run.report.stop_reason != ALL_GENERATIONS {
                context
                    .capped
                    .push((seed, arm, run.report.generations_completed));
            }
        }

        pairs.push(Pair {
            seed,
            a: run_a.found,
            r: run_r.found,
        });
    }

    let Some(&(generations, max_calls)) = shared.first() else {
        return Err("no seed pairs".to_string());
    };

    if shared.len() > 1 {
        return Err(format!(
            "the runs differ in --generations or --max-calls: {shared:?}"
        ));
    }

    context.generations = generations;
    context.max_calls = max_calls;

    render(&context, &pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use brutmaschine::cascade::circuit::Reference;

    /// A cost with these T and CNOT counts.
    fn cost(t: u32, cnot: u32) -> Cost {
        Cost {
            t,
            cnot,
            ..Cost::default()
        }
    }

    /// One target's outcome: the cheapest exact seed and bred answers, if any. A target
    /// counts as reached (pre-registered rule) when that answer is at or below both
    /// reference counts.
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

    /// A run in which the seed Cascades solve `seeds` and bred Cascades solve `bred`, each
    /// at the reference; every other target only exactly, above its reference.
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

        summarise(targets)
    }

    /// The summary fields of `rediscovery` for these outcomes.
    fn summarise(targets: Vec<TargetOutcome>) -> Rediscovery {
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

    const SEEDS_HIT: [&str; 2] = ["swap", "cs"];

    /// The first `k` targets the seeds miss.
    fn bred(k: usize) -> Vec<&'static str> {
        verified()
            .iter()
            .map(|t| t.name)
            .filter(|n| !SEEDS_HIT.contains(n))
            .take(k)
            .collect()
    }

    /// Pairs for these seeds, where arm A's bred Cascades solve `a` and arm R's `r` of the
    /// nine targets the seeds miss.
    fn pairs(seeds: &[u64], a: &[usize], r: &[usize]) -> Vec<Pair> {
        seeds
            .iter()
            .zip(a.iter().zip(r))
            .map(|(&seed, (&ka, &kr))| Pair {
                seed,
                a: found(&SEEDS_HIT, &bred(ka)),
                r: found(&SEEDS_HIT, &bred(kr)),
            })
            .collect()
    }

    /// The counted seeds 1 to n.
    fn counted(n: u64) -> Vec<u64> {
        (1..=n).collect()
    }

    /// A run in which the seed Cascades solve `seeds` and bred Cascades solve the first
    /// targets the seeds miss, in library order, each first in the given generation.
    fn timed(seeds: &[&str], firsts: &[usize]) -> Rediscovery {
        let open: Vec<&str> = verified()
            .iter()
            .map(|t| t.name)
            .filter(|n| !seeds.contains(n))
            .collect();
        let mut run = found(seeds, &open[..firsts.len()]);

        for (t, &g) in run.targets.iter_mut().filter(|t| t.reached_by_bred).zip(firsts) {
            t.first_generation = Some(g);
        }

        run
    }

    /// Pairs for the seeds 1, 2, 3, ... with these pass times of arm A and arm R, None for
    /// never. The seed Cascades solve swap and cs.
    fn timed_pairs(times: &[(Option<usize>, Option<usize>)]) -> Vec<Pair> {
        // A run passes when five of the nine open targets fall.
        let passing = |t: Option<usize>| match t {
            Some(g) => timed(&SEEDS_HIT, &[g; 5]),
            None => timed(&SEEDS_HIT, &[]),
        };

        (1..)
            .zip(times)
            .map(|(seed, &(a, r))| Pair {
                seed,
                a: passing(a),
                r: passing(r),
            })
            .collect()
    }

    /// The rows of `pairs` under the pre-registered rule.
    fn rows(pairs: &[Pair]) -> Vec<Row> {
        statistic(pairs, Rule::Full).expect("pairs").rows
    }

    #[test]
    fn exact_fractions() {
        assert_eq!(Ratio::new(6, 8), Ratio { num: 3, den: 4 });
        assert_eq!(Ratio::new(0, 9), Ratio { num: 0, den: 1 });
        assert_eq!(Ratio::new(56, 1024).to_string(), "7/128 = 0.0547");
        assert_eq!(Ratio::new(11, 1024).to_string(), "11/1024 = 0.0107");
    }

    #[test]
    fn half_is_not_above_half() {
        let half = Share { hit: 4, open: 8 };
        let more = Share { hit: 5, open: 8 };

        assert!(!half.above_half());
        assert!(more.above_half());

        // D is undefined when the seed Cascades solve every target: never above 1/2.
        let undefined = Share { hit: 0, open: 0 };

        assert!(!undefined.above_half());
        assert_eq!(undefined.ratio(), None);
    }

    #[test]
    fn the_median_is_the_middle_or_the_mean_of_the_two_middles() {
        let r = |n, d| Ratio::new(n, d);

        assert_eq!(median(&[]), None);
        assert_eq!(median(&[r(2, 9), r(7, 9), r(5, 9)]), Some(r(5, 9)));
        assert_eq!(
            median(&[r(1, 9), r(8, 9), r(5, 9), r(6, 9)]),
            Some(r(11, 18))
        );
    }

    #[test]
    fn evidence_boundaries_in_count_form() {
        let e = |passes: u32, failures: u32| Evidence {
            up: passes,
            n: passes + failures,
        };
        // With 0 to 7 failures, the passes that prove a claim; with 0 to 7 passes, the
        // failures that fail it.
        let prove = [8, 10, 11, 13, 15, 16, 18, 20];
        let fail = [5, 5, 6, 7, 7, 8, 8, 9];

        for (other, (&passes, &failures)) in (0..).zip(prove.iter().zip(&fail)) {
            assert!(e(passes, other).proven(), "{passes} passes, {other} failures");
            assert!(
                !e(passes - 1, other).proven(),
                "{passes} - 1 passes, {other} failures"
            );
            assert!(e(other, failures).failed(), "{other} passes, {failures} failures");
            assert!(
                !e(other, failures - 1).failed(),
                "{other} passes, {failures} - 1 failures"
            );
        }

        // E = 3^passes / 2^pairs as an exact fraction, starting at 1.
        assert_eq!(Evidence::default().ratio(), Ratio::new(1, 1));
        assert_eq!(e(8, 1).ratio().to_string(), "6561/512 = 12.8145");
        assert_eq!(e(1, 5).ratio(), Ratio::new(3, 64));

        // At the cap of 60 pairs both tests still fit a u128.
        assert!(e(60, 0).proven() && !e(60, 0).failed());
        assert!(e(0, 60).failed() && !e(0, 60).proven());
    }

    #[test]
    fn pass_time_is_the_kth_bred_only_first_generation() {
        // Nine open targets: D > 1/2 needs five of them, so T is the fifth smallest first
        // generation, whatever the order of the targets.
        assert_eq!(
            pass_time(&timed(&SEEDS_HIT, &[97, 33, 75, 7, 33, 10])),
            Some(75)
        );
        assert_eq!(pass_time(&timed(&SEEDS_HIT, &[7, 10, 33, 33])), None);

        // A target the seed Cascades solve never counts, also when a bred Cascade solved it
        // early: here swap, in generation 4.
        let with_swap = |firsts: &[usize]| {
            let mut run = timed(&SEEDS_HIT, firsts);

            for t in &mut run.targets {
                if t.target == "swap" {
                    *t = outcome("swap", t.best_seed, at_reference("swap"));
                }
            }

            run
        };

        assert_eq!(pass_time(&with_swap(&[7, 10, 33, 33, 75, 97])), Some(75));
        assert_eq!(pass_time(&with_swap(&[7, 10, 33, 33])), None);

        // With |R_seed| = 3, eight targets are open: four are only half, five pass.
        let three = ["swap", "cs", "cz"];

        assert_eq!(pass_time(&timed(&three, &[2, 4, 6, 8])), None);
        assert_eq!(pass_time(&timed(&three, &[2, 4, 6, 8, 10])), Some(10));
        assert_eq!(pass_time(&timed(&three, &[12, 10, 8, 6, 4, 2])), Some(10));

        // No open target, no pass time.
        let every: Vec<&str> = verified().iter().map(|t| t.name).collect();

        assert_eq!(pass_time(&timed(&every, &[])), None);
    }

    #[test]
    fn a_pass_after_the_ceiling_is_never() {
        // Runs longer than the ceiling: arm A passes in generation 1000, arm R in 1001.
        let all = timed_pairs(&[(Some(1000), Some(1001))]);
        let row = &rows(&all)[0];

        assert_eq!(pass_time(&all[0].r), Some(1001));
        assert_eq!((row.t_a, row.t_r), (Some(1000), None));
        assert_eq!(first(row.t_a, row.t_r), First::A);
    }

    #[test]
    fn first_to_pass_decides_the_pair() {
        assert_eq!(first(Some(7), Some(380)), First::A);
        assert_eq!(first(Some(97), None), First::A);
        assert_eq!(first(None, Some(720)), First::R);
        assert_eq!(first(Some(105), Some(48)), First::R);
        assert_eq!(first(Some(50), Some(50)), First::Tie);
        assert_eq!(first(None, None), First::Tie);
    }

    #[test]
    fn eight_pairs_that_arm_a_passes_first_prove_both_claims() {
        // Arm A solves seven of the nine open targets, all first in generation 4: T_A = 4
        // and D_A = 7/9. Arm R solves four, which is not more than half: T_R is never.
        let all = pairs(&counted(8), &[7; 8], &[4; 8]);
        let rule = sequential(&rows(&all));

        assert_eq!(
            (rule.stop, rule.primary, rule.secondary),
            (Some(8), Verdict::Proven(8), Verdict::Proven(8))
        );
        assert_eq!(claim(&rows(&all)), Claim::Made);

        let text = render(&Context::default(), &all).expect("render");

        assert!(text.contains("    1   11  2: cs, swap"), "{text}");
        assert!(
            text.contains("Median D_A over the 8 counted pair(s), descriptive: 7/9 = 0.7778."),
            "{text}"
        );
        assert!(
            text.contains("    8      4  never  1  A      6561/256 = 25.6289      6561/256 = 25.6289\n"),
            "{text}"
        );
        assert!(
            text.contains(
                "Primary (ticket 03): PROVEN at pair 8. More than half of all seeds pass within 1000 generations."
            ),
            "{text}"
        );
        assert!(
            text.contains(
                "Secondary (selection): PROVEN at pair 8. Among untied pairs, arm A passes first more often than arm R."
            ),
            "{text}"
        );
        assert!(
            !text.contains("No claim") && !text.contains("not counted"),
            "{text}"
        );
    }

    #[test]
    fn the_pilot_order_fails_at_the_sixth_pair() {
        // The order of the 200-generation pilot: arm A passes in the first pair only, arm R
        // in none.
        let mut times = vec![(Some(32), None)];

        times.extend([(None, None); 5]);

        let all = timed_pairs(&times);
        let rule = sequential(&rows(&all));

        assert_eq!(
            (rule.stop, rule.primary, rule.secondary),
            (Some(6), Verdict::Failed(6), Verdict::NotTested)
        );
        assert_eq!(rule.evidence[4].0.ratio(), Ratio::new(3, 32));
        assert_eq!(rule.evidence[5].0.ratio(), Ratio::new(3, 64));
        assert_eq!(rule.evidence[5].1.ratio(), Ratio::new(3, 2));

        let text = render(&Context::default(), &all).expect("render");

        assert!(
            text.contains("E1 = 3/64 = 0.0469 after 6 pair(s); it fell to 1/20 at pair 6."),
            "{text}"
        );
        assert!(text.contains("E2 = 3/2 = 1.5000 after 6 pair(s).\n"), "{text}");
        assert!(text.contains("Primary (ticket 03): FAILED at pair 6."), "{text}");
        assert!(
            text.contains("Secondary (selection): NOT TESTED, since the primary is not proven."),
            "{text}"
        );
    }

    #[test]
    fn ties_leave_e2_unchanged() {
        // Arm A first, then a tie in the same generation, then a tie where neither passes.
        let all = timed_pairs(&[(Some(7), Some(380)), (Some(50), Some(50)), (None, None)]);
        let rule = sequential(&rows(&all));
        let e1: Vec<Ratio> = rule.evidence.iter().map(|(e1, _)| e1.ratio()).collect();
        let e2: Vec<Evidence> = rule.evidence.iter().map(|&(_, e2)| e2).collect();

        assert_eq!(e2, [Evidence { up: 1, n: 1 }; 3]);
        // E1 follows X in every pair: X = 1, 1, 0.
        assert_eq!(e1, [Ratio::new(3, 2), Ratio::new(9, 4), Ratio::new(9, 8)]);
    }

    #[test]
    fn the_secondary_is_claimed_only_after_the_primary() {
        let tie = (None, None);
        let a_first = (Some(100), None);

        // As in the 1000-generation pilot: a tie, then pairs that arm A passes first. E2
        // reaches 20 at pair 9, E1 only at pair 11, and then both claims hold.
        let mut times = vec![tie];

        times.extend([a_first; 8]);

        let early = sequential(&rows(&timed_pairs(&times)));

        assert_eq!(
            (early.stop, early.primary, early.secondary),
            (None, Verdict::Open, Verdict::Proven(9))
        );

        times.extend([a_first; 2]);

        let all = timed_pairs(&times);
        let rule = sequential(&rows(&all));

        assert_eq!(rule.evidence[9].0.ratio(), Ratio::new(19683, 1024));
        assert_eq!(
            (rule.stop, rule.primary, rule.secondary),
            (Some(11), Verdict::Proven(11), Verdict::Proven(9))
        );

        let text = render(&Context::default(), &all).expect("render");

        assert!(text.contains("Primary (ticket 03): PROVEN at pair 11."), "{text}");
        assert!(text.contains("Secondary (selection): PROVEN at pair 9."), "{text}");

        // Four ties, eight pairs that arm A passes first and six ties: E2 reaches 20 at
        // pair 12, but E1 falls to 1/20 at pair 18, so there is no claim on selection.
        let mut times = vec![tie; 4];

        times.extend([a_first; 8]);
        times.extend([tie; 6]);

        let all = timed_pairs(&times);
        let rule = sequential(&rows(&all));

        assert_eq!(rule.evidence[11].1.ratio(), Ratio::new(6561, 256));
        assert_eq!(
            (rule.stop, rule.primary, rule.secondary),
            (Some(18), Verdict::Failed(18), Verdict::NotTested)
        );

        let text = render(&Context::default(), &all).expect("render");

        assert!(!text.contains("PROVEN"), "{text}");
    }

    #[test]
    fn an_open_run_names_the_next_seed() {
        let all = timed_pairs(&[(Some(7), Some(380)), (None, None), (Some(75), None)]);
        let rule = sequential(&rows(&all));

        assert_eq!(
            (rule.stop, rule.primary, rule.secondary),
            (None, Verdict::Open, Verdict::Open)
        );

        let text = render(&Context::default(), &all).expect("render");

        assert!(
            text.contains("The rule goes on after pair 3.\nOPEN: run seed 4 next.\n"),
            "{text}"
        );
        assert!(
            !text.contains("PROVEN") && !text.contains("FAILED"),
            "{text}"
        );
    }

    #[test]
    fn pairs_after_the_stop_are_not_counted() {
        // The pilot order fails at pair 6, and pairs 7 to 10 ran after the stop. In pair 8
        // the seed Cascades solve four targets, which would void a counted pair.
        let mut times = vec![(Some(32), None)];

        times.extend([(None, None); 5]);
        times.extend([(Some(40), None); 4]);

        let mut all = timed_pairs(&times);
        let wide = ["swap", "cs", "cz", "ch"];

        all[7].a = found(&wide, &[]);
        all[7].r = found(&wide, &[]);

        let rule = sequential(&rows(&all));

        assert_eq!((rule.stop, rule.counted()), (Some(6), 6));
        assert_eq!(claim(&rows(&all)), Claim::Made);

        let text = render(&Context::default(), &all).expect("render");

        for seed in 7..=10 {
            assert!(
                text.contains(&format!("{seed:>5}  after the stop, not counted\n")),
                "{text}"
            );
        }

        assert!(text.contains("10 seed pair(s)"), "{text}");
        assert!(text.contains("Median D_A over the 6 counted pair(s)"), "{text}");
        assert!(text.contains("D_A > 1/2 in 1 of 6 counted seed(s)"), "{text}");
        assert!(text.contains("Primary (ticket 03): FAILED at pair 6."), "{text}");
        assert!(
            !text.contains("VOID") && !text.contains("  NO\n"),
            "{text}"
        );
    }

    #[test]
    fn sixty_pairs_that_never_cross_are_undecided() {
        // X = 1 exactly when E1 < 1 before the pair, so E1 stays in [1/2, 3/2). Arm R
        // passes first in every second pair, as in reference_rule.py.
        let mut e1 = Evidence::default();
        let mut times = Vec::new();

        for pair in 1..=60 {
            let x = e1.ratio().compare(Ratio::new(1, 1)) == Ordering::Less;
            let t_r = match (x, pair % 2 == 0) {
                (true, true) => Some(4),
                (false, false) => Some(6),
                _ => None,
            };

            times.push((x.then_some(5), t_r));
            e1 = e1.step(x);
        }

        let all = timed_pairs(&times);
        let rule = sequential(&rows(&all));

        for (e1, _) in &rule.evidence {
            assert_ne!(e1.ratio().compare(Ratio::new(1, 2)), Ordering::Less);
            assert_eq!(e1.ratio().compare(Ratio::new(3, 2)), Ordering::Less);
        }

        assert_eq!(
            (rule.stop, rule.primary, rule.secondary),
            (Some(60), Verdict::Undecided, Verdict::NotTested)
        );
        // The end of reference_rule.py: 38 passes in 60 pairs, 20 of 48 untied pairs to A.
        assert_eq!(
            rule.evidence[59],
            (Evidence { up: 38, n: 60 }, Evidence { up: 20, n: 48 })
        );

        let text = render(&Context::default(), &all).expect("render");

        assert!(
            text.contains("Primary (ticket 03): UNDECIDED after 60 pairs, no claim."),
            "{text}"
        );
    }

    #[test]
    fn an_uncalibrated_seed_voids_the_experiment() {
        let mut all = pairs(&counted(10), &[7; 10], &[0; 10]);

        // In seed 4 the seed Cascades solve four targets: more than 11 / 3.
        let wide = ["swap", "cs", "cz", "ch"];

        all[3].a = found(&wide, &bred(7));
        all[3].r = found(&wide, &[]);

        let made = claim(&rows(&all));

        assert_eq!(made, Claim::Void(vec![4]));
        assert_eq!(
            verdict(&made, &sequential(&rows(&all))),
            "No claim: the experiment is void.\n"
        );

        let text = render(&Context::default(), &all).expect("render");

        assert!(
            text.contains("VOID: the seed Cascades of seed(s) [4]"),
            "{text}"
        );
        assert!(text.contains("  NO\n"), "{text}");
        assert!(!text.contains("PROVEN"), "{text}");
    }

    #[test]
    fn seeds_must_start_at_one_without_gaps() {
        // Also the pilots are not counted, whatever they show.
        for seeds in [&[1, 2, 4][..], &[2, 3], &[101, 102]] {
            let all = pairs(seeds, &[7; 3], &[0; 3]);
            let made = claim(&rows(&all));
            let text = render(&Context::default(), &all).expect("render");

            assert!(matches!(made, Claim::Withheld(_)), "{seeds:?}: {made:?}");
            assert!(
                text.contains(&format!("No claim: the seeds are {seeds:?}")),
                "{text}"
            );
            assert!(!text.contains("OPEN"), "{text}");
        }

        assert_eq!(
            claim(&rows(&pairs(&[1, 2, 3], &[7; 3], &[0; 3]))),
            Claim::Made
        );
    }

    #[test]
    fn more_than_sixty_pairs_are_refused() {
        let sixty = pairs(&counted(60), &[7; 60], &[0; 60]);

        assert!(render(&Context::default(), &sixty).is_ok());

        let sixty_one = pairs(&counted(61), &[7; 61], &[0; 61]);
        let refused = render(&Context::default(), &sixty_one).expect_err("61 pairs");

        assert!(refused.contains("more than 60 seed pairs"), "{refused}");
        assert!(matches!(claim(&rows(&sixty_one)), Claim::Withheld(_)));
    }

    #[test]
    fn a_run_that_bred_more_than_its_config_plans_is_refused() {
        // 200 generations plus generation 0.
        assert_eq!(bred_as_planned("all generations done", 201, 200), Ok(()));
        assert_eq!(bred_as_planned("call budget reached", 150, 200), Ok(()));

        // A run that bred 201 generations and was then resumed with --generations 200.
        let longer = bred_as_planned("all generations done", 202, 200).expect_err("longer");

        assert!(longer.contains("bred 202 generations"), "{longer}");
        assert!(bred_as_planned("all generations done", 150, 200).is_err());
        assert!(bred_as_planned("call budget reached", 201, 200).is_err());
    }

    #[test]
    fn only_the_pre_registered_length_and_cap_make_a_claim() {
        let all = pairs(&counted(10), &[7; 10], &[4; 10]);
        let frozen = Context::default();

        assert_eq!((frozen.generations, frozen.max_calls), (1000, 13_000_000));
        assert!(frozen.capped.is_empty());

        let text = render(&frozen, &all).expect("render");

        assert!(
            text.contains("every run: --generations 1000, --max-calls 13000000"),
            "{text}"
        );
        assert!(text.contains("PROVEN"), "{text}");

        // A longer or shorter run, another cap, or a run at the cap: the same numbers, but
        // no rule decides, also not when a seed would void the experiment. The length and
        // the cap of the old fixed rule are no exception.
        let mut voided = all.clone();

        voided[3].a = found(&["swap", "cs", "cz", "ch"], &bred(7));
        voided[3].r = found(&["swap", "cs", "cz", "ch"], &[]);

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
                capped: vec![(1, Arm::R, 2)],
                ..Context::default()
            },
        ];

        for context in &other {
            for runs in [&all, &voided] {
                let made = claim_in(context, &rows(runs));
                let text = render(context, runs).expect("render");

                assert!(matches!(made, Claim::Withheld(_)), "{made:?}");
                assert!(text.contains("No claim"), "{text}");
                assert!(
                    text.contains("E1 = 6561/256 = 25.6289 after 8 pair(s)"),
                    "{text}"
                );
                assert!(
                    !text.contains("PROVEN")
                        && !text.contains("FAILED")
                        && !text.contains("VOID")
                        && !text.contains("OPEN"),
                    "{text}"
                );
            }
        }

        // Seeds with a gap keep their own reason.
        let gap = rows(&pairs(&[1, 2, 4], &[7; 3], &[0; 3]));

        assert_eq!(claim_in(&other[0], &gap), claim(&gap));
    }

    #[test]
    fn the_t_only_variant_ignores_cnot_except_without_t() {
        // Bred answers: toffoli with the reference T count but one CNOT too many, and cz,
        // a Clifford target, with no T but one CNOT too many.
        let mut targets = found(&SEEDS_HIT, &[]).targets;

        for t in &mut targets {
            let r = t.reference;

            if t.target == "toffoli" || t.target == "cz" {
                *t = outcome(&t.target, t.best_seed, Some(cost(r.t, r.cnot + 1)));
            }
        }

        let run = summarise(targets);

        assert_eq!(count(&run, Rule::Full).share, Share { hit: 0, open: 9 });
        assert_eq!(count(&run, Rule::TOnly).share, Share { hit: 1, open: 9 });

        // A seed answer at the reference T count makes a target a seed target of the
        // T-only variant, so it leaves the open targets there.
        let mut targets = found(&SEEDS_HIT, &[]).targets;

        for t in &mut targets {
            let r = t.reference;

            if t.target == "ccz" {
                *t = outcome(&t.target, Some(cost(r.t, r.cnot + 4)), t.best_bred);
            }
        }

        let run = summarise(targets);
        let counted = count(&run, Rule::TOnly);

        assert_eq!(
            counted.seeds,
            BTreeSet::from(["ccz", "cs", "swap"].map(String::from))
        );
        assert_eq!(counted.share, Share { hit: 0, open: 8 });
    }

    #[test]
    fn runs_that_disagree_on_r_seed_are_not_a_pair() {
        let mut all = pairs(&counted(10), &[7; 10], &[0; 10]);

        all[0].r = found(&["swap"], &[]);

        assert!(statistic(&all, Rule::Full).is_err());
    }

    #[test]
    fn counts_must_agree_with_the_target_list() {
        let run = found(&SEEDS_HIT, &bred(5));

        assert_eq!(check_counts(&run), Ok(()));

        let mut forged = run.clone();

        forged.reached_by_bred_only += 1;
        assert!(check_counts(&forged).is_err());

        let mut forged = run.clone();

        forged.passes = !forged.passes;
        assert!(check_counts(&forged).is_err());

        let mut forged = run.clone();

        forged.share = Some(0.5);
        assert!(check_counts(&forged).is_err());

        // A target a bred Cascade solved, without its first generation: its pass time
        // would be wrong.
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
    fn only_the_settings_of_the_arm_are_accepted() {
        let a = arm_config(Arm::A, 3, 60, 20_000);
        let r = arm_config(Arm::R, 3, 60, 20_000);

        assert_eq!(a.selection, ParentSelection::Dgm);
        assert_eq!(r.selection, ParentSelection::Uniform);
        assert_ne!(a, r);
        assert_eq!(
            BreedConfig {
                selection: ParentSelection::Uniform,
                ..a.clone()
            },
            r
        );
        assert!(a.check().is_ok());
        assert_eq!(
            (
                a.search_puzzles,
                a.select_puzzles,
                a.children_per_generation,
                a.llm_every
            ),
            (33, 22, 8, 0)
        );
        assert_eq!(a.domain, DomainConfig::Circuits(CircuitConfig::default()));
    }

    #[test]
    fn arguments_name_the_two_arms() {
        let args = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();

        assert_eq!(
            parse_args(args(&["--a", "x", "y", "--r", "z"])),
            Ok((
                vec![PathBuf::from("x"), PathBuf::from("y")],
                vec![PathBuf::from("z")]
            ))
        );
        assert!(parse_args(args(&["x", "--a", "y", "--r", "z"])).is_err());
        assert!(parse_args(args(&["--a", "x"])).is_err());
        assert!(parse_args(args(&["--a", "x", "--b", "y"])).is_err());
    }
}
