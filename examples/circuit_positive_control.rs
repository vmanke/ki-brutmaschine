//! The frozen analysis of the step 1 experiment (plan (g) "Pre-registration", ADR 0006,
//! docs/PREREG_STEP1_CIRCUITS.md): does circuit breeding solve the targets the seed
//! Cascades miss, and does Darwin Gödel selection beat a uniform choice of parents?
//!
//! It reads finished circuit runs of `brut-breed` from two arms, paired by seed: arm A
//! (`--domain circuits --search 33 --select 22 --children 8`, every other setting at its
//! default) and arm R (the same plus `--selection uniform`). Each run must be finished and
//! must have the settings of its arm, and all runs must share their generations and their
//! call cap. From each run it reads `rediscovery.json` and prints, per seed s, R_seed(s),
//! D_A(s), D_R(s) and the calibration gate |R_seed(s)| <= |L| / 3. Then it prints:
//! - the void condition it can check: a seed that fails the calibration gate;
//! - the primary result (ticket 03): D_A(s) > 1/2 in at least 9 of the 10 seeds, with the
//!   median of D_A;
//! - the secondary result: D_A(s) > D_R(s) in at least 9 of the 10 pairs, ties counting
//!   against A, an exact one-sided sign test of size 11/1024;
//! - both again for the T-only variant, where a target counts when an exact answer has at
//!   most the reference's T count, and also at most its CNOT count for a Clifford target
//!   whose reference has no T;
//! - descriptive lines: the first generation in which a bred Cascade solved each target,
//!   runs that ended at the call cap, and the cheapest invention candidate per target and
//!   run.
//!
//! D(s) = |R_bred(s) \ R_seed(s)| / |L \ R_seed(s)|: the share of the targets the seed
//! Cascades miss that a bred Cascade solves, exactly and at the reference cost, on at
//! least one of the target's dressed search tasks. Every decision is made in integers.
//!
//! It makes a claim only for the pre-registered seeds 1 to 10 in both arms, run with the
//! pre-registered `--generations 200` and `--max-calls 2600000`, with no run ended at the
//! call cap. For other seeds, such as the pilots 101 and 102, for fewer seeds, or for
//! other generations or caps, it prints the numbers and says that it makes no claim. Run
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

/// The seeds the pre-registration counts.
const COUNTED: [u64; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

/// The generations of every counted Run (docs/PREREG_STEP1_CIRCUITS.md, section 6).
const GENERATIONS: usize = 200;

/// The call cap of every counted Run, which no counted Run can reach (section 6).
const MAX_CALLS: u64 = 2_600_000;

/// A rule holds when it holds in at least this many of the 10 pairs.
const NEEDED: usize = 9;

/// The most seed pairs the exact sign test handles: 2^100 still fits a u128.
const MAX_PAIRS: usize = 100;

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

    /// D of `self` > D of `other`; false when either is undefined, so a tie or an
    /// undefined pair counts against `self`.
    fn beats(self, other: Share) -> bool {
        match (self.ratio(), other.ratio()) {
            (Some(a), Some(b)) => a.compare(b) == Ordering::Greater,
            _ => false,
        }
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
/// the verified library targets in library order: L of the pre-registration.
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
        && found.passes == (counted.calibrated() && counted.share.above_half());

    if !agrees {
        return Err("its counts disagree with its own target list".to_string());
    }

    Ok(())
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
}

impl Row {
    fn calibrated(&self) -> bool {
        3 * self.seeds.len() <= self.library
    }
}

/// The statistics of one rule over all pairs.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Statistic {
    rows: Vec<Row>,
    /// Seeds with D_A > 1/2.
    above_half: usize,
    /// Pairs with D_A > D_R.
    beats: usize,
    /// The median of the defined D_A values: the middle one, or the mean of the two
    /// middle ones.
    median: Option<Ratio>,
    /// The chance of `beats` or more wins in as many pairs under the null of a fair coin:
    /// the exact one-sided p of the sign test.
    p: Ratio,
}

fn statistic(pairs: &[Pair], rule: Rule) -> Result<Statistic, String> {
    let mut rows = Vec::new();

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
        });
    }

    let above_half = rows.iter().filter(|r| r.a.above_half()).count();
    let beats = rows.iter().filter(|r| r.a.beats(r.r)).count();
    let shares: Vec<Ratio> = rows.iter().filter_map(|r| r.a.ratio()).collect();

    Ok(Statistic {
        median: median(&shares),
        p: upper_tail(rows.len(), beats),
        rows,
        above_half,
        beats,
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

/// n choose k, exactly.
fn binomial(n: usize, k: usize) -> u128 {
    (0..k).fold(1u128, |acc, i| acc * (n - i) as u128 / (i + 1) as u128)
}

/// P(X >= k) for X binomial with n trials of probability 1/2, as an exact fraction.
fn upper_tail(n: usize, k: usize) -> Ratio {
    let num: u128 = (k..=n).map(|j| binomial(n, j)).sum();

    Ratio::new(num, 1u128 << n)
}

/// Whether the runs may support a claim.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Claim {
    /// The pre-registered seeds, all calibrated: the rules decide.
    Made,
    /// The pre-registered seeds, but these fail the calibration gate: void.
    Void(Vec<u64>),
    /// Not the pre-registered seeds: why no claim is made.
    Withheld(String),
}

fn claim(rows: &[Row]) -> Claim {
    let seeds: Vec<u64> = rows.iter().map(|r| r.seed).collect();

    if seeds.len() < COUNTED.len() {
        return Claim::Withheld(format!(
            "only {} seed pair(s), fewer than the {} pre-registered ones",
            seeds.len(),
            COUNTED.len()
        ));
    }

    if seeds != COUNTED {
        return Claim::Withheld(format!(
            "the seeds are {seeds:?}, not the pre-registered 1 to 10"
        ));
    }

    let void: Vec<u64> = rows
        .iter()
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

/// The verdict of a rule that holds in `count` pairs.
fn verdict(claim: &Claim, count: usize) -> String {
    match claim {
        Claim::Made if count >= NEEDED => "PASSES".to_string(),
        Claim::Made => "FAILS".to_string(),
        Claim::Void(_) => "No claim: the experiment is void".to_string(),
        Claim::Withheld(why) => format!("No claim: {why}"),
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

/// The table and the two rules of one statistic.
fn render_rule(out: &mut String, stat: &Statistic, claim: &Claim, name: &str) {
    let n = stat.rows.len();

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

    let median = stat
        .median
        .map_or("undefined".to_string(), |m| m.to_string());

    out.push_str(&format!(
        "{name}, primary (ticket 03): D_A > 1/2 in {} of {n} seeds (needed: at least {NEEDED} of 10). Median D_A: {median}. {}\n",
        stat.above_half,
        verdict(claim, stat.above_half)
    ));
    out.push_str(&format!(
        "{name}, secondary (selection): D_A > D_R in {} of {n} pairs, ties counting against A (needed: at least {NEEDED} of 10, an exact one-sided sign test of size {}). p = P(at least {} of {n} by chance) = {}. {}\n",
        stat.beats,
        upper_tail(COUNTED.len(), NEEDED),
        stat.beats,
        stat.p,
        verdict(claim, stat.beats)
    ));
}

/// The whole analysis as text.
fn render(context: &Context, pairs: &[Pair]) -> Result<String, String> {
    if pairs.is_empty() {
        return Err("no seed pairs".to_string());
    }

    if pairs.len() > MAX_PAIRS {
        return Err(format!("more than {MAX_PAIRS} seed pairs"));
    }

    let full = statistic(pairs, Rule::Full)?;
    let t_only = statistic(pairs, Rule::TOnly)?;
    let claim = claim_in(context, &full.rows);
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
            "Void conditions checked here: every seed passes the calibration gate |R_seed| <= |L| / 3.\n",
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
    render_rule(&mut out, &full, &claim, "Pre-registered rule");
    out.push_str(
        "\nT-only variant: a target counts when an exact answer reaches its T reference, and its CNOT reference where that T count is 0.\n",
    );
    render_rule(&mut out, &t_only, &claim, "T-only variant");
    out.push_str(&descriptive(context, pairs));

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

    #[test]
    fn exact_fractions_and_the_sign_test() {
        assert_eq!(binomial(10, 9), 10);
        assert_eq!(binomial(10, 5), 252);
        assert_eq!(upper_tail(10, 9), Ratio::new(11, 1024));
        assert_eq!(upper_tail(10, 10), Ratio::new(1, 1024));
        assert_eq!(upper_tail(10, 8), Ratio::new(56, 1024));
        assert_eq!(upper_tail(10, 0), Ratio::new(1, 1));
        assert_eq!(upper_tail(10, 8).to_string(), "7/128 = 0.0547");
        assert_eq!(upper_tail(10, 9).to_string(), "11/1024 = 0.0107");
        assert_eq!(Ratio::new(6, 8), Ratio { num: 3, den: 4 });
        assert_eq!(Ratio::new(0, 9), Ratio { num: 0, den: 1 });
    }

    #[test]
    fn half_is_not_above_half_and_ties_go_against_a() {
        let half = Share { hit: 4, open: 8 };
        let more = Share { hit: 5, open: 8 };

        assert!(!half.above_half());
        assert!(more.above_half());
        assert!(more.beats(half));
        assert!(!half.beats(half), "a tie counts against A");
        assert!(!half.beats(more));

        // D is undefined when the seed Cascades solve every target: never above 1/2, and it
        // never beats or is beaten.
        let undefined = Share { hit: 0, open: 0 };

        assert!(!undefined.above_half());
        assert!(!undefined.beats(half) && !half.beats(undefined));
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
    fn ten_calibrated_seeds_that_rediscover_pass_both_rules() {
        let all = pairs(&COUNTED, &[7; 10], &[4, 4, 4, 4, 4, 4, 4, 4, 4, 9]);
        let full = statistic(&all, Rule::Full).expect("pairs");

        // D_A = 7/9 everywhere; D_R = 4/9 in nine seeds and 9/9 in the last.
        assert_eq!(full.above_half, 10);
        assert_eq!(full.beats, 9);
        assert_eq!(full.median, Some(Ratio::new(7, 9)));
        assert_eq!(full.p, Ratio::new(11, 1024));
        assert_eq!(claim(&full.rows), Claim::Made);

        let text = render(&Context::default(), &all).expect("render");

        assert!(
            text.contains("Pre-registered rule, primary (ticket 03): D_A > 1/2 in 10 of 10 seeds"),
            "{text}"
        );
        assert!(text.contains("Median D_A: 7/9 = 0.7778. PASSES"), "{text}");
        assert!(
            text.contains(
                "D_A > D_R in 9 of 10 pairs, ties counting against A (needed: at least 9 of 10, an exact one-sided sign test of size 11/1024 = 0.0107). p = P(at least 9 of 10 by chance) = 11/1024 = 0.0107. PASSES"
            ),
            "{text}"
        );
        assert!(text.contains("    1   11  2: cs, swap"), "{text}");
        assert!(!text.contains("No claim"), "{text}");
    }

    #[test]
    fn ties_count_against_a_in_the_sign_test() {
        // Two pairs tie at 5/9, one goes to R: A wins 7 of 10 pairs.
        let a = [5, 5, 6, 6, 6, 6, 6, 6, 6, 5];
        let r = [5, 5, 2, 2, 2, 2, 2, 2, 2, 6];
        let full = statistic(&pairs(&COUNTED, &a, &r), Rule::Full).expect("pairs");

        assert_eq!(full.above_half, 10);
        assert_eq!(full.beats, 7);
        assert_eq!(full.p, upper_tail(10, 7));
        assert_eq!(verdict(&claim(&full.rows), full.beats), "FAILS");
        assert_eq!(verdict(&claim(&full.rows), full.above_half), "PASSES");
    }

    #[test]
    fn eight_seeds_above_half_fail_the_primary_rule() {
        let a = [7, 7, 7, 7, 7, 7, 7, 7, 4, 3];
        let full = statistic(&pairs(&COUNTED, &a, &[0; 10]), Rule::Full).expect("pairs");

        assert_eq!(full.above_half, 8);
        assert_eq!(verdict(&claim(&full.rows), full.above_half), "FAILS");
        // The lower values sort first: 3/9, 4/9, then 7/9 eight times.
        assert_eq!(full.median, Some(Ratio::new(7, 9)));
    }

    #[test]
    fn an_uncalibrated_seed_voids_the_experiment() {
        let mut all = pairs(&COUNTED, &[7; 10], &[0; 10]);

        // In seed 4 the seed Cascades solve four targets: more than 11 / 3.
        let wide = ["swap", "cs", "cz", "ch"];

        all[3].a = found(&wide, &bred(7));
        all[3].r = found(&wide, &[]);

        let full = statistic(&all, Rule::Full).expect("pairs");
        let made = claim(&full.rows);

        assert_eq!(made, Claim::Void(vec![4]));
        assert_eq!(
            verdict(&made, full.above_half),
            "No claim: the experiment is void"
        );

        let text = render(&Context::default(), &all).expect("render");

        assert!(
            text.contains("VOID: the seed Cascades of seed(s) [4]"),
            "{text}"
        );
        assert!(text.contains("  NO\n"), "{text}");
        assert!(!text.contains("PASSES"), "{text}");
    }

    #[test]
    fn fewer_than_ten_seeds_or_other_seeds_make_no_claim() {
        let nine = pairs(&COUNTED[..9], &[9; 9], &[0; 9]);
        let full = statistic(&nine, Rule::Full).expect("pairs");

        assert_eq!(full.above_half, 9);
        assert_eq!(
            claim(&full.rows),
            Claim::Withheld(
                "only 9 seed pair(s), fewer than the 10 pre-registered ones".to_string()
            )
        );

        let text = render(&Context::default(), &nine).expect("render");

        assert!(text.contains("No claim: only 9 seed pair(s)"), "{text}");
        assert!(
            !text.contains("PASSES") && !text.contains("FAILS"),
            "{text}"
        );

        // The pilots are not counted, whatever they show.
        let pilots = pairs(&[101, 102], &[9; 2], &[0; 2]);
        let full = statistic(&pilots, Rule::Full).expect("pairs");

        assert!(matches!(claim(&full.rows), Claim::Withheld(_)));

        let eleven: Vec<u64> = (1..=11).collect();
        let full = statistic(&pairs(&eleven, &[9; 11], &[0; 11]), Rule::Full).expect("pairs");

        assert!(matches!(claim(&full.rows), Claim::Withheld(_)));
    }

    #[test]
    fn only_the_pre_registered_length_and_cap_make_a_claim() {
        let all = pairs(&COUNTED, &[7; 10], &[4; 10]);
        let frozen = Context::default();

        assert_eq!((frozen.generations, frozen.max_calls), (200, 2_600_000));
        assert!(frozen.capped.is_empty());

        let text = render(&frozen, &all).expect("render");

        assert!(
            text.contains("every run: --generations 200, --max-calls 2600000"),
            "{text}"
        );
        assert!(text.contains("PASSES"), "{text}");

        // A longer or shorter run, another cap, or a run at the cap: the same numbers, but
        // no rule decides, also not when a seed would void the experiment.
        let mut voided = all.clone();

        voided[3].a = found(&["swap", "cs", "cz", "ch"], &bred(7));
        voided[3].r = found(&["swap", "cs", "cz", "ch"], &[]);

        let other = [
            Context {
                generations: 60,
                ..Context::default()
            },
            Context {
                max_calls: 200_000,
                ..Context::default()
            },
            Context {
                capped: vec![(1, Arm::R, 2)],
                ..Context::default()
            },
        ];

        for context in &other {
            for runs in [&all, &voided] {
                let made = claim_in(context, &statistic(runs, Rule::Full).expect("pairs").rows);
                let text = render(context, runs).expect("render");

                assert!(matches!(made, Claim::Withheld(_)), "{made:?}");
                assert!(text.contains("No claim"), "{text}");
                assert!(text.contains("D_A > 1/2 in 10 of 10 seeds"), "{text}");
                assert!(
                    !text.contains("PASSES") && !text.contains("FAILS") && !text.contains("VOID"),
                    "{text}"
                );
            }
        }

        // Fewer seeds keep their own reason.
        let nine = pairs(&COUNTED[..9], &[9; 9], &[0; 9]);
        let rows = statistic(&nine, Rule::Full).expect("pairs").rows;

        assert_eq!(claim_in(&other[0], &rows), claim(&rows));
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
        let mut all = pairs(&COUNTED, &[7; 10], &[0; 10]);

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
