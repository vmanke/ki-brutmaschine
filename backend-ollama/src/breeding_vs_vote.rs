//! The analysis of the pre-registered experiment `docs/PREREG_BREEDING_VS_VOTE.md`: does a
//! Cascade bred with a real model solve more unseen puzzles than a majority vote that
//! spends at least as many Worker calls on them?
//!
//! # What it reads
//!
//! For each seed s of the list it reads the finished `brut-breed-ollama` runs `B-s` (arm B,
//! Darwin Gödel selection) and `L-s` (arm L, lexicase selection) in one runs directory.
//! Beside run `B-s`, `docs/prereg-breeding-vs-vote/run-seed.sh` keeps what arm V needs:
//!
//! - `vote-control/worker-calls.jsonl`: the record of the vote's Worker calls, as
//!   `vote-control` files it.
//! - `vote-up.json` and `vote-down.json`: copies of `vote-control.json` right after
//!   `vote-control --temperature 0.7 --samples N` ran with N = N_up(s) and with
//!   N = N_down(s). A later control overwrites `vote-control.json`, so the driver keeps
//!   both copies. No `vote-down.json` exists when no N_down exists.
//! - `panels.json`: both panels of the seed with their answers. This program regenerates
//!   the panels and refuses a file that differs. The independent recount `recount.py`
//!   cannot regenerate them; it checks every puzzle against its logged content id.
//!
//! # What it checks
//!
//! A run is refused when its `config.json` differs from the frozen `config-B.json` or
//! `config-L.json` in anything but the seed, when its name and its seed disagree, and when
//! any of its Worker calls failed. A run without `report.json` is unfinished and not
//! counted. A finished run must have bred every generation; its event log must be intact
//! (canonical lines, an unbroken hash chain, the checkpoint at its end), its logged panels
//! must be the regenerated ones, and its logged select results must be its report's. Both
//! arms of a seed must share their panels and the search answers of their seed Cascades
//! (generation 0, section 4). The champion's select phase is replayed from the run's own
//! record: every call must be on file, and the replayed answers must be the snapshot's.
//! The vote is replayed from its record, and each kept result must be, puzzle by puzzle,
//! what the replayed samples give. Seeds must come from the list, in its order, without
//! gaps; the counted list holds at most 60 pairs (D6).
//!
//! # What it computes
//!
//! Per seed s, on the select puzzles whose id is no id of the seed's search panel (the
//! others overlap and are left out of every count): the champion c(s) and the voted seed
//! as `vote.rs` picks them, C(s), K_c(s), N_up(s) and N_down(s) with K_v(s, N), V(s), the
//! outcome with N_down(s), Y(s), Y1(s) for S1 and Y2(s) for S2. The sequential rule of
//! section 6 then runs in list order with integer comparisons only. It reports E_V without
//! a claim, the secondary values with their threshold of 200, and either the next seed to
//! run or the verdicts. The result is a JSON document and a table.
//!
//! The champion, the voted seed, the select panel and the vote's sample seeds come from
//! `vote.rs`, so these definitions are those of `vote-control`. Replays run through the
//! core's own `run_cascade` against a backend that answers from a record only.
#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use brutmaschine::backend::{GenRequest, GenResponse, ModelBackend};
use brutmaschine::cascade::arena::{
    BreedConfig, Event, Finalist, Member, ModelPool, ParentSelection, Report, cascade_call_bound,
    run_cascade,
};
use brutmaschine::cascade::domain::{Domain, GridDomain};
use brutmaschine::cascade::genome::{Cascade, Stage, seeds_for};
use brutmaschine::cascade::puzzle::{Grid, Puzzle};
use brutmaschine::cascade::variation::Op;
use brutmaschine::util::{BrutError, Result, derive_seed, sha256_hex, write_atomic};

use crate::vote::{self, Options, Plan, PuzzleVote, RunDir, VoteReport};
use crate::{CallRecord, call_key};

/// A claim of section 6 is proven when E_B reaches this.
pub const PROOF: u32 = 100;

/// The rule stops for futility when E_B falls to 1 / FUTILITY.
pub const FUTILITY: u32 = 20;

/// A secondary claim (section 7) needs its E to reach this.
pub const SECONDARY: u32 = 200;

/// An exact natural number of any size. 3^120 has 191 bits, more than a u128 holds, and
/// the cap of D6 may become 120. The digits are in base 10^9, the lowest first; the top
/// digit is not 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Natural(Vec<u32>);

/// The base of the digits of a Natural.
const DIGIT_BASE: u64 = 1_000_000_000;

impl Natural {
    /// base^exp, for a base of at least 1.
    pub fn power(base: u32, exp: u32) -> Natural {
        (0..exp).fold(Natural(vec![1]), |product, _| product.times(base))
    }

    /// The product with a factor of at least 1.
    pub fn times(self, factor: u32) -> Natural {
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
    /// With no 0 on top, more digits make a larger number; equal lengths compare from the
    /// top digit down.
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

/// How one comparison of two counts ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The first count is larger.
    First,
    /// The second count is larger.
    Second,
    Tie,
}

impl Outcome {
    /// The outcome of `a` against `b`.
    pub fn of(a: usize, b: usize) -> Outcome {
        match a.cmp(&b) {
            Ordering::Greater => Outcome::First,
            Ordering::Less => Outcome::Second,
            Ordering::Equal => Outcome::Tie,
        }
    }
}

/// An evidence value E = 3^up / 2^(up + down): `up` factors of 3/2 and `down` of 1/2. The
/// default is the start, E = 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Evidence {
    pub up: u32,
    pub down: u32,
}

impl Evidence {
    /// E after one more pair: by 3/2 on First, by 1/2 on Second, the same on a tie.
    pub fn step(self, o: Outcome) -> Evidence {
        match o {
            Outcome::First => Evidence {
                up: self.up + 1,
                ..self
            },
            Outcome::Second => Evidence {
                down: self.down + 1,
                ..self
            },
            Outcome::Tie => self,
        }
    }

    /// The mirrored value: 3/2 where this one has 1/2, and the other way round.
    pub fn mirrored(self) -> Evidence {
        Evidence {
            up: self.down,
            down: self.up,
        }
    }

    /// 3^up.
    fn numerator(self) -> Natural {
        Natural::power(3, self.up)
    }

    /// 2^(up + down).
    fn denominator(self) -> Natural {
        Natural::power(2, self.up + self.down)
    }

    /// E >= k, in integers: 3^up >= k * 2^(up + down).
    pub fn at_least(self, k: u32) -> bool {
        self.numerator() >= self.denominator().times(k)
    }

    /// E <= 1 / k, in integers: k * 3^up <= 2^(up + down).
    pub fn at_most_one_in(self, k: u32) -> bool {
        self.numerator().times(k) <= self.denominator()
    }

    /// E as an exact fraction "3^up/2^(up + down)", in lowest terms.
    /// A power of 3 and a power of 2 share no factor, so it is in lowest terms already.
    pub fn fraction(self) -> String {
        format!("{}/{}", self.numerator(), self.denominator())
    }

    /// E as a float, only to be read; no decision uses it.
    pub fn value(self) -> f64 {
        1.5f64.powi(self.up as i32) * 0.5f64.powi(self.down as i32)
    }
}

/// The state of the primary rule after a pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Primary {
    /// No stop yet.
    Running,
    /// E_B >= 100: breeding beats the vote.
    Proof,
    /// E_B <= 1/20: no evidence that breeding wins in three of four untied seeds.
    Futility,
    /// The cap is reached without a claim.
    Open,
}

/// The evidence values after one counted pair.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Step {
    pub e_b: Evidence,
    pub e_s1: Evidence,
    pub e_s2: Evidence,
}

impl Step {
    /// E_V, the mirror of E_B.
    pub fn e_v(&self) -> Evidence {
        self.e_b.mirrored()
    }
}

/// The sequential rule of section 6 over the counted pairs in list order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rule {
    /// The values after each pair up to the stop; pairs after it are not counted.
    pub steps: Vec<Step>,
    /// The stop pair, from 1.
    pub stop: Option<usize>,
    pub primary: Primary,
}

/// Run the rule on `outcomes`, one [Y, Y1, Y2] per pair in list order, with the cap `cap`.
///
/// E_B, E_S1 and E_S2 start at 1 and take the pairs strictly in the order given. The rule
/// stops at the first pair with E_B >= PROOF (a proof), else at the first with
/// E_B <= 1 / FUTILITY (futility), else at pair `cap` (open). Pairs after the stop are not
/// counted. The secondary values never stop it.
pub fn sequential(outcomes: &[[Outcome; 3]], cap: usize) -> Rule {
    let mut steps = Vec::new();
    let mut now = Step::default();

    for (i, [y, y1, y2]) in outcomes.iter().enumerate() {
        now = Step {
            e_b: now.e_b.step(*y),
            e_s1: now.e_s1.step(*y1),
            e_s2: now.e_s2.step(*y2),
        };
        steps.push(now);

        let primary = if now.e_b.at_least(PROOF) {
            Primary::Proof
        } else if now.e_b.at_most_one_in(FUTILITY) {
            Primary::Futility
        } else if i + 1 >= cap {
            Primary::Open
        } else {
            continue;
        };

        return Rule {
            steps,
            stop: Some(i + 1),
            primary,
        };
    }

    Rule {
        steps,
        stop: None,
        primary: Primary::Running,
    }
}

/// What the call matching of section 4 finds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Matched {
    pub k_c: u64,
    pub n_up: usize,
    pub k_v_up: u64,
    pub n_down: Option<usize>,
    pub k_v_down: Option<u64>,
}

/// N_up and N_down, or the samples the vote record must hold first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matching {
    Matched(Matched),
    /// The samples on file do not reach K_c: run the vote with this many samples.
    Need(usize),
}

/// The call matching of section 4 from K_c and the Worker calls of each sample on file,
/// in sample order. Every sample costs at least `floor` calls (one per Worker and
/// counted puzzle).
///
/// K_v(N) is the sum of the first N samples. N_up is the smallest N >= 1 with
/// K_v(N) >= K_c, N_down the largest N >= 1 with K_v(N) <= K_c. K_v grows by at least
/// `floor` >= 1 with every sample, so N_down is N_up when K_v(N_up) = K_c, else N_up - 1
/// (none when that is 0); no sample beyond N_up is needed to know it. When the samples on
/// file do not reach K_c, Need names the fewest samples that can: those on file plus
/// enough samples of `floor` calls to cover the rest.
pub fn match_calls(k_c: u64, samples: &[u64], floor: u64) -> std::result::Result<Matching, String> {
    if k_c == 0 {
        return Err("the champion made no Worker call on the counted select puzzles".to_string());
    }

    if floor == 0 {
        return Err("a sample of the vote costs no call: no counted select puzzle".to_string());
    }

    if let Some(k) = samples.iter().position(|&c| c < floor) {
        return Err(format!(
            "sample {k} of the vote made {} counted calls, fewer than its {floor} Workers on the counted puzzles",
            samples[k]
        ));
    }

    let mut k_v = 0u64;

    for (i, &calls) in samples.iter().enumerate() {
        let before = k_v;

        k_v += calls;

        if k_v >= k_c {
            let n_up = i + 1;
            let (n_down, k_v_down) = if k_v == k_c {
                (Some(n_up), Some(k_v))
            } else if n_up >= 2 {
                (Some(n_up - 1), Some(before))
            } else {
                (None, None)
            };

            return Ok(Matching::Matched(Matched {
                k_c,
                n_up,
                k_v_up: k_v,
                n_down,
                k_v_down,
            }));
        }
    }

    Ok(Matching::Need(
        samples.len() + (k_c - k_v).div_ceil(floor) as usize,
    ))
}

/// The ids of `select` that are also ids of `search`, in the order of `select`.
pub fn overlap(search: &[String], select: &[String]) -> Vec<String> {
    let seen: std::collections::BTreeSet<&String> = search.iter().collect();

    select
        .iter()
        .filter(|id| seen.contains(id))
        .cloned()
        .collect()
}

/// Error unless `cfg` equals `frozen` in everything but the seed. The error names the
/// settings that differ.
pub fn check_settings(cfg: &BreedConfig, frozen: &BreedConfig) -> std::result::Result<(), String> {
    let same_seed = BreedConfig {
        seed: frozen.seed,
        ..cfg.clone()
    };

    if &same_seed == frozen {
        return Ok(());
    }

    let (ours, theirs) = match (
        serde_json::to_value(&same_seed),
        serde_json::to_value(frozen),
    ) {
        (Ok(serde_json::Value::Object(a)), Ok(serde_json::Value::Object(b))) => (a, b),
        _ => return Err("the settings differ".to_string()),
    };
    let keys: std::collections::BTreeSet<&String> = ours.keys().chain(theirs.keys()).collect();
    let differ: Vec<&str> = keys
        .into_iter()
        .filter(|k| ours.get(*k) != theirs.get(*k))
        .map(String::as_str)
        .collect();

    Err(format!(
        "the settings differ from the frozen ones in: {}",
        differ.join(", ")
    ))
}

/// Every request seed a run of `c` from the Cascade seed `base` can carry.
///
/// They are those `run_cascade` draws: Worker j of stage k answers with
/// `derive_seed(base, "stage{k}/worker{j}")`, and when the stage's Workers disagree, Worker
/// j judges the pair of distinct answers (x, y) with `derive_seed(v, "{x}/{y}/{j}")`, where
/// `v = derive_seed(base, "vote{k}")`. A stage of w Workers has at most w distinct answers,
/// so there are `cascade_call_bound(c)` seeds, and every recorded call of such a run
/// carries one of them.
pub fn call_seeds(c: &Cascade, base: u64) -> Vec<u64> {
    let mut seeds = Vec::new();

    for (k, stage) in c.stages.iter().enumerate() {
        let w = stage.workers.len();
        let vote = derive_seed(base, &format!("vote{k}"));

        seeds.extend((0..w).map(|j| derive_seed(base, &format!("stage{k}/worker{j}"))));

        for x in 0..w {
            for y in x + 1..w {
                seeds.extend((0..w).map(|j| derive_seed(vote, &format!("{x}/{y}/{j}"))));
            }
        }
    }

    seeds
}

/// The frozen settings of arm B, in the directory of the frozen files.
pub const FROZEN_B: &str = "config-B.json";

/// The frozen settings of arm L.
pub const FROZEN_L: &str = "config-L.json";

/// Where the frozen files lie, seen from the repository root.
pub const FROZEN_DIR: &str = "docs/prereg-breeding-vs-vote";

/// The vote result with N_up samples, kept beside run B-s.
pub const VOTE_UP: &str = "vote-up.json";

/// The vote result with N_down samples, kept beside run B-s.
pub const VOTE_DOWN: &str = "vote-down.json";

/// Both panels of a seed with their answers, written beside run B-s.
pub const PANELS: &str = "panels.json";

/// The report a finished run writes last.
pub const REPORT: &str = "report.json";

/// The wall time of each committed generation of a run.
pub const TIMINGS: &str = "timings.jsonl";

/// The record of a run's Mutator calls.
pub const MUTATOR_RECORD: &str = "llm/mutator-calls.jsonl";

/// The failed Mutator calls of a run; `brut-mutator` asks them again in its next round.
pub const MUTATOR_ERRORS: &str = "llm/mutator-errors.jsonl";

/// The failed calls of the vote; the next `vote-control` asks them again.
pub const VOTE_ERRORS: &str = "vote-control/worker-errors.jsonl";

/// The journal `run-seed.sh` keeps in each run directory.
pub const JOURNAL: &str = "driver.log";

/// How a run ends that bred every generation.
pub const ALL_GENERATIONS: &str = "all generations done";

/// The temperature of every Worker of the vote (D5).
pub const TEMPERATURE: f32 = 0.7;

/// The cap on counted pairs (D6). The counted list is 1, 2, ..., CAP.
pub const CAP: usize = 60;

/// The pilot seeds (section 9). They are not on the list and never count.
pub const PILOT: [u64; 5] = [901, 902, 903, 904, 905];

/// The names of Y(s): breeding first, vote first, tie.
pub const Y_NAMES: [&str; 3] = ["breeding first", "vote first", "tie"];

/// The names of Y1(s) (S1): the champion against one plain run of the voted seed.
pub const Y1_NAMES: [&str; 3] = ["champion first", "seed first", "tie"];

/// The names of Y2(s) (S2): the champion of run L-s against that of run B-s.
pub const Y2_NAMES: [&str; 3] = ["lexicase first", "dgm first", "tie"];

/// The hash before the first entry of an event log.
const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

impl Outcome {
    /// The outcome's name from `names` (first, second, tie).
    pub fn name(self, names: [&'static str; 3]) -> &'static str {
        match self {
            Outcome::First => names[0],
            Outcome::Second => names[1],
            Outcome::Tie => names[2],
        }
    }
}

/// Which seeds the analysis reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum List {
    /// The counted seeds 1, 2, ..., CAP, in this order.
    Counted,
    /// The pilot seeds. The rule runs on them for illustration only.
    Pilot,
}

impl List {
    /// The seeds of the list, in its order.
    pub fn seeds(self) -> Vec<u64> {
        match self {
            List::Counted => (1..=CAP as u64).collect(),
            List::Pilot => PILOT.to_vec(),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            List::Counted => "counted",
            List::Pilot => "pilot",
        }
    }
}

/// The two breeding arms. Arm V has no run of its own: it votes on run B-s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    /// Darwin Gödel selection.
    B,
    /// Lexicase selection (D3).
    L,
}

impl Arm {
    pub fn letter(self) -> &'static str {
        match self {
            Arm::B => "B",
            Arm::L => "L",
        }
    }

    /// The run directory of this arm on `seed` in `runs`.
    pub fn dir(self, runs: &Path, seed: u64) -> PathBuf {
        runs.join(format!("{}-{seed}", self.letter()))
    }
}

/// The arm and seed a run directory's name gives: `B-<seed>` or `L-<seed>`, the seed in
/// plain decimal. None for any other name.
pub fn parse_run_name(name: &str) -> Option<(Arm, u64)> {
    let (arm, rest) = match name.split_once('-')? {
        ("B", rest) => (Arm::B, rest),
        ("L", rest) => (Arm::L, rest),
        _ => return None,
    };
    let seed: u64 = rest.parse().ok()?;

    (seed.to_string() == rest).then_some((arm, seed))
}

fn refused(m: impl fmt::Display) -> BrutError {
    BrutError::Invalid(format!("breeding-vs-vote: {m}"))
}

fn broken(m: impl fmt::Display) -> BrutError {
    BrutError::Integrity(format!("breeding-vs-vote: {m}"))
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let text =
        std::fs::read_to_string(path).map_err(|e| refused(format!("{}: {e}", path.display())))?;

    serde_json::from_str(&text).map_err(|e| refused(format!("{}: {e}", path.display())))
}

/// The non-empty lines of `path`; 0 when the file does not exist.
fn count_lines(path: &Path) -> Result<u64> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(text.lines().filter(|l| !l.trim().is_empty()).count() as u64),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(e) => Err(refused(format!("{}: {e}", path.display()))),
    }
}

/// The frozen settings of both breeding arms.
#[derive(Clone, Debug, PartialEq)]
pub struct Frozen {
    pub b: BreedConfig,
    pub l: BreedConfig,
}

impl Frozen {
    /// `config-B.json` and `config-L.json` in `dir`. Refused unless arm B uses Darwin Gödel
    /// selection, arm L lexicase selection, and the two agree in every other setting.
    pub fn read(dir: &Path) -> Result<Frozen> {
        let frozen = Frozen {
            b: read_json(&dir.join(FROZEN_B))?,
            l: read_json(&dir.join(FROZEN_L))?,
        };
        let l_as_b = BreedConfig {
            selection: ParentSelection::Dgm,
            selection_params: frozen.b.selection_params,
            ..frozen.l.clone()
        };

        if frozen.b.selection != ParentSelection::Dgm
            || frozen.l.selection != ParentSelection::Lexicase
        {
            return Err(refused(format!(
                "{}: arm B must select by dgm and arm L by lexicase",
                dir.display()
            )));
        }

        check_settings(&l_as_b, &frozen.b).map_err(|e| {
            refused(format!(
                "{}: arms B and L differ beyond the selection: {e}",
                dir.display()
            ))
        })?;

        Ok(frozen)
    }

    pub fn of(&self, arm: Arm) -> &BreedConfig {
        match arm {
            Arm::B => &self.b,
            Arm::L => &self.l,
        }
    }
}

/// One line of an event log, as brut-breed writes it.
#[derive(Serialize, Deserialize)]
struct LogLine {
    seq: u64,
    prev: String,
    hash: String,
    event: Event,
}

/// Where a checkpoint says the log stood.
#[derive(Deserialize)]
struct Anchor {
    log_len: u64,
    log_head: String,
}

/// The events of the log at `path` and the hash of its last entry, checked as
/// `brut-breed --verify` checks them: every line canonical and complete, and the hash
/// chain unbroken.
fn read_log(path: &Path) -> Result<(Vec<Event>, String)> {
    let text =
        std::fs::read_to_string(path).map_err(|e| refused(format!("{}: {e}", path.display())))?;

    if !text.is_empty() && !text.ends_with('\n') {
        return Err(broken(format!(
            "{}: the last line is incomplete",
            path.display()
        )));
    }

    let mut head = GENESIS.to_string();
    let mut events = Vec::new();

    for (i, line) in text.split_terminator('\n').enumerate() {
        let bad = |what: &str| broken(format!("{}: entry {i} {what}", path.display()));
        let entry: LogLine = serde_json::from_str(line).map_err(|_| bad("is not a log entry"))?;

        if serde_json::to_string(&entry)? != line {
            return Err(bad("is not in canonical form"));
        }

        let event_json = serde_json::to_string(&entry.event)?;

        if entry.seq != i as u64
            || entry.prev != head
            || entry.hash != sha256_hex(format!("{head}\n{event_json}").as_bytes())
        {
            return Err(bad("breaks the hash chain"));
        }

        head = entry.hash;
        events.push(entry.event);
    }

    Ok((events, head))
}

/// The Finalist events after the last Generation event: the select phase of the archive.
fn logged_finalists(events: &[Event]) -> Vec<Finalist> {
    let start = events
        .iter()
        .rposition(|e| matches!(e, Event::Generation { .. }))
        .map_or(0, |i| i + 1);

    events[start..]
        .iter()
        .filter_map(|e| match e {
            Event::Finalist(f) => Some(f.clone()),
            _ => None,
        })
        .collect()
}

/// One line of `timings.jsonl`.
#[derive(Deserialize)]
struct Timing {
    seconds: f64,
}

/// The summed wall time of the committed generations; 0 without a file.
fn generation_seconds(dir: &Path) -> Result<f64> {
    let path = dir.join(TIMINGS);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0.0),
        Err(e) => return Err(refused(format!("{}: {e}", path.display()))),
    };
    let mut sum = 0.0;

    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let t: Timing =
            serde_json::from_str(line).map_err(|e| refused(format!("{}: {e}", path.display())))?;

        sum += t.seconds;
    }

    Ok(sum)
}

/// The ids of a panel, in its order.
fn ids(panel: &[Puzzle]) -> Vec<String> {
    panel.iter().map(|p| p.id.clone()).collect()
}

/// Exactly right, as the arena grades a grid.
fn right(p: &Puzzle, g: &Grid) -> bool {
    GridDomain.grade(p, g).is_ok_and(|x| x.correct)
}

/// A finished run of one arm, read and checked.
#[derive(Clone, Debug)]
pub struct Run {
    pub dir: PathBuf,
    pub arm: Arm,
    /// Settings, snapshot and call health, as vote-control reads them.
    pub run: RunDir,
    pub report: Report,
    /// Both panels, regenerated from the seed and equal to the logged ones.
    pub search: Vec<Puzzle>,
    pub select: Vec<Puzzle>,
    /// The logged select results, equal to the report's.
    pub finalists: Vec<Finalist>,
    pub generation_seconds: f64,
}

impl Run {
    pub fn cfg(&self) -> &BreedConfig {
        &self.run.cfg
    }

    pub fn seed(&self) -> u64 {
        self.run.cfg.seed
    }

    pub fn archive(&self) -> &[Member<Grid>] {
        &self.run.snapshot.archive
    }

    /// The select-phase answer of member `id` on `p`, as the snapshot keeps it.
    pub fn select_answer(&self, id: &str, p: &Puzzle) -> Option<&Option<Grid>> {
        self.run.snapshot.select_answers.get(id)?.get(&p.id)
    }

    /// Whether member `id` solved `p` exactly in the select phase. None when the run did
    /// not score it there.
    pub fn solved(&self, id: &str, p: &Puzzle) -> Option<bool> {
        self.select_answer(id, p)
            .map(|a| a.as_ref().is_some_and(|g| right(p, g)))
    }

    /// The select puzzles member `id` solved among those `counted` marks.
    pub fn solved_on(&self, id: &str, counted: &[bool]) -> usize {
        self.select
            .iter()
            .zip(counted)
            .filter(|&(p, &c)| c && self.solved(id, p) == Some(true))
            .count()
    }

    /// Error unless member `id` solved on the whole select panel what the log says.
    fn check_logged(&self, id: &str) -> Result<()> {
        let all = self.solved_on(id, &vec![true; self.select.len()]);
        let logged = self
            .finalists
            .iter()
            .find(|f| f.id == id)
            .map(|f| f.select_solved);

        if logged != Some(all) {
            return Err(broken(format!(
                "{}: member {id} solves {all} select puzzles by its answers, the log says {logged:?}",
                self.dir.display()
            )));
        }

        Ok(())
    }
}

/// The checks a run passes from its start on: its settings are the frozen ones of its arm
/// but for the seed, its seed is the one its name gives, and none of its Worker calls
/// failed.
fn check_started(dir: &Path, arm: Arm, seed: u64, frozen: &Frozen) -> Result<()> {
    let config = dir.join(vote::CONFIG);

    if config.exists() {
        let cfg: BreedConfig = read_json(&config)?;

        if cfg.seed != seed {
            return Err(refused(format!(
                "{}: the run has seed {}, its name says {seed}",
                dir.display(),
                cfg.seed
            )));
        }

        check_settings(&cfg, frozen.of(arm))
            .map_err(|e| refused(format!("{}: {e}", config.display())))?;
    }

    let failed = count_lines(&dir.join(vote::RUN_ERRORS))?;

    if failed > 0 {
        return Err(refused(format!(
            "{}: {failed} Worker call(s) failed ({}); such a run is not counted (section 8): delete it and run it again from scratch on the same seed",
            dir.display(),
            vote::RUN_ERRORS
        )));
    }

    Ok(())
}

/// The finished run of `arm` on `seed` in `dir`, with every check of the module docs.
pub fn load_finished(dir: &Path, arm: Arm, seed: u64, frozen: &Frozen) -> Result<Run> {
    check_started(dir, arm, seed, frozen)?;

    let run = vote::load_run(dir)?;

    if let Some(why) = run.health.problem() {
        return Err(refused(format!("{}: {why}", dir.display())));
    }

    let report: Report = read_json(&dir.join(REPORT))?;
    let (cfg, snap) = (&run.cfg, &run.snapshot);
    let planned = cfg.generations + 1;

    if !snap.finished
        || snap.next_generation != planned
        || report.stop_reason != ALL_GENERATIONS
        || report.generations_completed != planned
    {
        return Err(refused(format!(
            "{}: the run ended with {:?} after {} of {planned} generations (generation 0 included); a counted run breeds them all",
            dir.display(),
            report.stop_reason,
            report.generations_completed
        )));
    }

    let (events, head) = read_log(&dir.join(vote::EVENTS))?;
    let anchor: Anchor = read_json(&dir.join(vote::SNAPSHOT))?;

    if anchor.log_len != events.len() as u64 || anchor.log_head != head {
        return Err(broken(format!(
            "{}: the checkpoint does not stand at the end of the event log",
            dir.display()
        )));
    }

    let Some(Event::RunStart {
        config_hash,
        search,
        select,
        ..
    }) = events.first()
    else {
        return Err(broken(format!(
            "{}: the first event is not run_start",
            dir.display()
        )));
    };

    if *config_hash != cfg.fixed_hash() {
        return Err(broken(format!(
            "{}: run_start was logged under other settings",
            dir.display()
        )));
    }

    let search_panel = GridDomain.panel(derive_seed(cfg.seed, "panel/search"), cfg.search_puzzles);
    let select_panel = vote::select_panel(cfg)?;

    if ids(&search_panel) != *search || ids(&select_panel) != *select {
        return Err(broken(format!(
            "{}: the logged panels differ from the regenerated ones",
            dir.display()
        )));
    }

    vote::check_panel(&select_panel, &run)?;

    let finalists = logged_finalists(&events);

    if finalists != report.finalists {
        return Err(broken(format!(
            "{}: the report's select results differ from the logged ones",
            dir.display()
        )));
    }

    Ok(Run {
        dir: dir.to_path_buf(),
        arm,
        generation_seconds: generation_seconds(dir)?,
        run,
        report,
        search: search_panel,
        select: select_panel,
        finalists,
    })
}

/// How far the run of one arm on one seed got.
#[derive(Clone, Debug)]
pub enum ArmState {
    /// No run directory.
    Missing,
    /// A run directory without `report.json`.
    Unfinished,
    Finished(Box<Run>),
}

impl ArmState {
    fn word(&self) -> &'static str {
        match self {
            ArmState::Missing => "not started",
            ArmState::Unfinished => "not finished",
            ArmState::Finished(_) => "finished",
        }
    }
}

/// The run of `arm` on `seed` in `runs`, checked as far as it got. An unfinished run is
/// checked for its settings, its seed and failed calls already.
pub fn arm_state(runs: &Path, arm: Arm, seed: u64, frozen: &Frozen) -> Result<ArmState> {
    let dir = arm.dir(runs, seed);

    if !dir.exists() {
        return Ok(ArmState::Missing);
    }

    if !dir.join(REPORT).exists() {
        check_started(&dir, arm, seed, frozen)?;

        return Ok(ArmState::Unfinished);
    }

    Ok(ArmState::Finished(Box::new(load_finished(
        &dir, arm, seed, frozen,
    )?)))
}

/// The finished run B-s in `dir`, its seed taken from its name.
pub fn load_b(dir: &Path, frozen: &Frozen) -> Result<Run> {
    let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or_default();

    match parse_run_name(name) {
        Some((Arm::B, seed)) => load_finished(dir, Arm::B, seed, frozen),
        _ => Err(refused(format!(
            "{}: the name of run B-s must be B-<seed>",
            dir.display()
        ))),
    }
}

/// How a call record may end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tail {
    /// The record of a finished run: every line complete.
    Complete,
    /// The record of a vote that may still be running: a last line without its newline
    /// is a write in progress and is not read.
    InProgress,
}

/// What one pass over a call record finds.
#[derive(Default)]
struct Scan {
    /// The full records whose request seed was asked for.
    kept: Vec<CallRecord>,
    /// Every record: lines, those from the shared call cache, and their tokens.
    lines: u64,
    cached: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
}

/// The parts of a record line that every line is read for.
#[derive(Deserialize)]
struct Light {
    request: LightRequest,
    response: LightResponse,
    #[serde(default)]
    cached: bool,
}

#[derive(Deserialize)]
struct LightRequest {
    seed: u64,
}

#[derive(Deserialize)]
struct LightResponse {
    prompt_tokens: u32,
    completion_tokens: u32,
}

/// One pass over the call record `path`, keeping the full records whose request seed
/// `keep` accepts. A missing file holds no record. An incomplete or broken line, a key
/// that does not belong to its request, and a key filed twice are integrity errors.
fn scan_record(path: &Path, keep: &dyn Fn(u64) -> bool, tail: Tail) -> Result<Scan> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Scan::default()),
        Err(e) => return Err(refused(format!("{}: {e}", path.display()))),
    };
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut scan = Scan::default();
    let mut keys = HashSet::new();
    let mut number = 0usize;

    loop {
        line.clear();

        if reader
            .read_line(&mut line)
            .map_err(|e| refused(format!("{}: {e}", path.display())))?
            == 0
        {
            break;
        }

        number += 1;

        let bad = |what: &str| broken(format!("{} line {number}: {what}", path.display()));

        if !line.ends_with('\n') {
            if tail == Tail::InProgress {
                break;
            }

            return Err(bad("incomplete"));
        }

        if line.trim().is_empty() {
            continue;
        }

        let light: Light = serde_json::from_str(&line).map_err(|e| bad(&e.to_string()))?;

        scan.lines += 1;
        scan.cached += u64::from(light.cached);
        scan.prompt_tokens += u64::from(light.response.prompt_tokens);
        scan.completion_tokens += u64::from(light.response.completion_tokens);

        if keep(light.request.seed) {
            let rec: CallRecord = serde_json::from_str(&line).map_err(|e| bad(&e.to_string()))?;

            if rec.key != call_key(&rec.tag, &rec.request) {
                return Err(bad("the key does not belong to its request"));
            }

            if !keys.insert(rec.key.clone()) {
                return Err(bad("the key is filed twice"));
            }

            scan.kept.push(rec);
        }
    }

    Ok(scan)
}

/// Answers from a call record, never from a model. A call the record lacks fails, and
/// the shared counter counts it.
struct Replay {
    tag: String,
    book: HashMap<String, GenResponse>,
    misses: Arc<AtomicU64>,
}

impl ModelBackend for Replay {
    fn name(&self) -> &str {
        &self.tag
    }

    fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
        match self.book.get(&call_key(&self.tag, req)) {
            Some(r) => Ok(r.clone()),
            None => {
                self.misses.fetch_add(1, AtomicOrdering::SeqCst);

                Err(refused(format!(
                    "{}: the record holds no answer to this call",
                    self.tag
                )))
            }
        }
    }
}

/// The models a Cascade's Workers name.
fn models(c: &Cascade) -> BTreeSet<String> {
    c.stages
        .iter()
        .flat_map(|s| &s.workers)
        .map(|w| w.model.clone())
        .collect()
}

/// A pool that answers every tag of `tags` from `records` alone, and its miss counter. A
/// tag without records still gets a backend, so each of its calls counts as a miss.
fn replay_pool(records: &[CallRecord], tags: &BTreeSet<String>) -> (ModelPool, Arc<AtomicU64>) {
    let misses = Arc::new(AtomicU64::new(0));
    let mut pool = ModelPool::new();

    for tag in tags {
        let book = records
            .iter()
            .filter(|r| &r.tag == tag)
            .map(|r| (r.key.clone(), r.response.clone()))
            .collect();

        pool.insert(
            tag,
            Box::new(Replay {
                tag: tag.clone(),
                book,
                misses: misses.clone(),
            }),
        );
    }

    (pool, misses)
}

/// How many records carry each request seed.
fn per_seed(records: &[CallRecord]) -> HashMap<u64, u64> {
    let mut n = HashMap::new();

    for r in records {
        *n.entry(r.request.seed).or_insert(0) += 1;
    }

    n
}

/// One run of a Cascade on one select puzzle, replayed from a record.
#[derive(Clone, Debug, PartialEq)]
pub struct Replayed {
    pub calls: u64,
    pub answer: Option<Grid>,
}

/// Replay `c` on `p` from the Cascade seed `base` against `pool`. None when a call is not
/// on file. Error unless the calls it made are exactly the records that carry its seeds.
#[allow(clippy::too_many_arguments)]
fn replay_one(
    pool: &ModelPool,
    misses: &AtomicU64,
    run: &Run,
    c: &Cascade,
    p: &Puzzle,
    base: u64,
    filed: &HashMap<u64, u64>,
    record: &Path,
) -> Result<Option<Replayed>> {
    let before = misses.load(AtomicOrdering::SeqCst);
    let r = run_cascade(pool, &run.cfg().limits, c, p, base);

    if misses.load(AtomicOrdering::SeqCst) > before {
        return Ok(None);
    }

    let on_file: u64 = call_seeds(c, base)
        .iter()
        .map(|s| filed.get(s).copied().unwrap_or(0))
        .sum();

    if on_file != r.usage.calls {
        return Err(broken(format!(
            "{}: a run on puzzle {} made {} calls, but {on_file} records carry its seeds",
            record.display(),
            p.id,
            r.usage.calls
        )));
    }

    Ok(Some(Replayed {
        calls: r.usage.calls,
        answer: r.answer,
    }))
}

/// The record totals of a run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Totals {
    pub recorded_calls: u64,
    pub cached_calls: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

impl Totals {
    fn of(scan: &Scan) -> Totals {
        Totals {
            recorded_calls: scan.lines,
            cached_calls: scan.cached,
            prompt_tokens: scan.prompt_tokens,
            completion_tokens: scan.completion_tokens,
        }
    }
}

/// The select phase of archive member `member` in `run`, replayed from the run's own
/// record (`llm/worker-calls.jsonl`): per select puzzle, in panel order, its Worker calls
/// and its answer, and the totals of the record. Error unless the record holds every call,
/// the calls on each puzzle are exactly the records with that run's seeds, and each answer
/// is the one the snapshot keeps.
pub fn replay_select(run: &Run, member: usize) -> Result<(Vec<Replayed>, Totals)> {
    let m = &run.archive()[member];
    let record = run.dir.join(vote::RUN_RECORD);
    let bases: Vec<u64> = run
        .select
        .iter()
        .map(|p| derive_seed(run.seed(), &format!("select/{}/{}", m.id, p.id)))
        .collect();
    let seeds: HashSet<u64> = bases
        .iter()
        .flat_map(|&b| call_seeds(&m.cascade, b))
        .collect();
    let scan = scan_record(&record, &|s| seeds.contains(&s), Tail::Complete)?;
    let filed = per_seed(&scan.kept);
    let (pool, misses) = replay_pool(&scan.kept, &models(&m.cascade));
    let mut out = Vec::new();

    for (p, &base) in run.select.iter().zip(&bases) {
        let r = replay_one(&pool, &misses, run, &m.cascade, p, base, &filed, &record)?.ok_or_else(
            || {
                broken(format!(
                    "{}: the record lacks a select-phase call of {} on puzzle {}",
                    record.display(),
                    m.id,
                    p.id
                ))
            },
        )?;

        if run.select_answer(&m.id, p) != Some(&r.answer) {
            return Err(broken(format!(
                "{}: the replayed select answer of {} on puzzle {} differs from the snapshot's",
                record.display(),
                m.id,
                p.id
            )));
        }

        out.push(r);
    }

    Ok((out, Totals::of(&scan)))
}

/// The vote samples of `voted` (its id `voted_id`) on file beside run B-s
/// (`vote-control/worker-calls.jsonl`), in sample order, each replayed on every select
/// puzzle with the seeds of `vote::sample_seed`. It stops at the first sample that the
/// record does not hold in full.
pub fn replay_votes(run: &Run, voted: &Cascade, voted_id: &str) -> Result<Vec<Vec<Replayed>>> {
    let record = run.dir.join(vote::RECORD);
    // A vote may still be running: a last line without its newline is a call being
    // filed, and the sample it belongs to is not on file yet.
    let scan = scan_record(&record, &|_| true, Tail::InProgress)?;
    let filed = per_seed(&scan.kept);
    let (pool, misses) = replay_pool(&scan.kept, &models(voted));
    // A full sample files at least one call per puzzle.
    let most = scan.kept.len() / run.select.len().max(1);
    let mut samples = Vec::new();

    for k in 0..=most {
        let mut sample = Vec::new();

        for p in &run.select {
            let base = vote::sample_seed(run.seed(), voted_id, k, &p.id);

            match replay_one(&pool, &misses, run, voted, p, base, &filed, &record)? {
                Some(r) => sample.push(r),
                None => return Ok(samples),
            }
        }

        samples.push(sample);
    }

    Ok(samples)
}

/// The calls of `runs` on the select puzzles that `counted` marks.
fn counted_calls(runs: &[Replayed], counted: &[bool]) -> u64 {
    runs.iter()
        .zip(counted)
        .filter(|&(_, &c)| c)
        .map(|(r, _)| r.calls)
        .sum()
}

/// The overlap of section 5 and the select puzzles that count: those whose id is no id of
/// the seed's search panel.
pub fn counted_puzzles(run: &Run) -> (Vec<String>, Vec<bool>) {
    let shared: BTreeSet<String> = overlap(&ids(&run.search), &ids(&run.select))
        .into_iter()
        .collect();
    let counted = run.select.iter().map(|p| !shared.contains(&p.id)).collect();

    (
        run.select
            .iter()
            .filter(|p| shared.contains(&p.id))
            .map(|p| p.id.clone())
            .collect(),
        counted,
    )
}

/// The call matching of run B-s, as section 4 defines it.
#[derive(Clone, Debug)]
pub struct VoteMatch {
    /// The select puzzles that overlap the search panel, and the mask of those that count.
    pub overlap: Vec<String>,
    pub counted: Vec<bool>,
    /// The champion's archive index, and its select phase replayed from the run's record.
    pub champion: usize,
    pub champion_runs: Vec<Replayed>,
    pub k_c: u64,
    /// The plan of the vote: the voted seed, and its Cascade at the vote's temperature.
    pub plan: Plan,
    pub voted: Cascade,
    /// The vote samples on file, replayed.
    pub samples: Vec<Vec<Replayed>>,
    pub matching: Matching,
    /// The totals of the run's own record.
    pub totals: Totals,
}

/// The call matching of run B-s on the select puzzles that count.
pub fn match_run(run: &Run) -> Result<VoteMatch> {
    let (overlap, counted) = counted_puzzles(run);

    match_masked(run, overlap, counted)
}

/// The call matching of run B-s on the select puzzles that `counted` marks.
///
/// K_c is the champion's calls there, replayed from the run's record. The voted Cascade
/// is the one vote-control samples with `--temperature 0.7`: the seed finalist with the
/// best search fitness, every Worker at the vote's temperature. K_v(N) sums the counted
/// calls of its first N samples on file.
pub fn match_masked(run: &Run, overlap: Vec<String>, counted: Vec<bool>) -> Result<VoteMatch> {
    let champion = vote::champion(run.archive(), run.cfg().finalists).ok_or_else(|| {
        refused(format!(
            "{}: the run has no bred finalist, so no champion",
            run.dir.display()
        ))
    })?;
    let (champion_runs, totals) = replay_select(run, champion)?;
    let k_c = counted_calls(&champion_runs, &counted);
    let opts = Options {
        temperature: Some(TEMPERATURE),
        samples: Some(1),
        ..Options::default()
    };
    let (plan, voted) = vote::plan(&run.run, &opts)?;
    let samples = replay_votes(run, &voted, &plan.cascade.id)?;
    let per_sample: Vec<u64> = samples.iter().map(|s| counted_calls(s, &counted)).collect();
    let floor = voted.worker_count() as u64 * counted.iter().filter(|&&c| c).count() as u64;
    let matching = match_calls(k_c, &per_sample, floor)
        .map_err(|e| refused(format!("{}: {e}", run.dir.display())))?;

    Ok(VoteMatch {
        overlap,
        counted,
        champion,
        champion_runs,
        k_c,
        plan,
        voted,
        samples,
        matching,
        totals,
    })
}

/// The kept vote result in `path`, checked: a complete vote of `n` samples per puzzle of
/// the voted seed at TEMPERATURE, set against the champion, on this run, and puzzle by
/// puzzle what the first `n` replayed samples give.
pub fn read_kept(path: &Path, n: usize, run: &Run, m: &VoteMatch) -> Result<VoteReport> {
    let kept: VoteReport = read_json(path)?;
    let bad = |what: String| refused(format!("{}: {what}", path.display()));
    let p = &kept.plan;
    let champion = &run.archive()[m.champion].id;

    if !kept.complete || p.partial || kept.failed_calls > 0 {
        return Err(bad("not a complete vote".to_string()));
    }

    if p.samples_per_puzzle != n {
        return Err(bad(format!(
            "it holds {} samples per puzzle, the call matching gives {n}",
            p.samples_per_puzzle
        )));
    }

    if p.cascade.member != m.plan.cascade.member
        || p.cascade.id != m.plan.cascade.id
        || p.cascade.temperature != Some(TEMPERATURE)
    {
        return Err(bad(format!(
            "it samples {} ({}), not the voted seed {} at temperature {TEMPERATURE}",
            p.cascade.id, p.cascade.member, m.plan.cascade.member
        )));
    }

    if p.champion.as_ref().map(|c| &c.id) != Some(champion) {
        return Err(bad(format!("it names another champion than {champion}")));
    }

    if p.run_seed != run.seed()
        || p.config_hash != run.run.snapshot.config_hash
        || !p.finished
        || p.select_puzzles != run.select.len()
    {
        return Err(bad("it belongs to another run".to_string()));
    }

    let samples = m.samples.get(..n).ok_or_else(|| {
        bad(format!(
            "the vote record holds {} samples, fewer than {n}",
            m.samples.len()
        ))
    })?;
    let expected: Vec<PuzzleVote> = run
        .select
        .iter()
        .enumerate()
        .map(|(i, puzzle)| {
            let answers: Vec<Option<Grid>> = samples.iter().map(|s| s[i].answer.clone()).collect();

            vote::judge_puzzle(
                puzzle,
                &answers,
                run.solved(&m.plan.cascade.member, puzzle),
                run.solved(champion, puzzle),
            )
        })
        .collect();

    if kept.puzzles != expected || kept.totals != vote::Totals::of(&kept.puzzles) {
        return Err(bad(
            "its puzzles differ from the vote the record gives".to_string()
        ));
    }

    let calls: u64 = samples.iter().flatten().map(|r| r.calls).sum();

    if kept.calls_used != calls {
        return Err(bad(format!(
            "it counts {} calls, the replayed samples {calls}",
            kept.calls_used
        )));
    }

    Ok(kept)
}

/// Both panels of a seed with their answers: what `panels.json` holds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Panels {
    pub seed: u64,
    pub search: Vec<Puzzle>,
    pub select: Vec<Puzzle>,
}

impl Panels {
    fn of(run: &Run) -> Panels {
        Panels {
            seed: run.seed(),
            search: run.search.clone(),
            select: run.select.clone(),
        }
    }
}

/// Error unless the run's `panels.json` holds its regenerated panels.
fn check_panels(run: &Run) -> Result<()> {
    let path = run.dir.join(PANELS);
    let file: Panels = read_json(&path)?;

    if file != Panels::of(run) {
        return Err(broken(format!(
            "{} differs from the regenerated panels",
            path.display()
        )));
    }

    Ok(())
}

/// Write `panels.json` beside the finished run B-s in `dir`. A file that exists already is
/// checked, not written; returns whether the file was written.
pub fn write_panels(dir: &Path, frozen: &Frozen) -> Result<bool> {
    let run = load_b(dir, frozen)?;
    let path = dir.join(PANELS);

    if path.exists() {
        check_panels(&run)?;

        return Ok(false);
    }

    write_atomic(
        &path,
        format!("{}\n", serde_json::to_string_pretty(&Panels::of(&run))?).as_bytes(),
    )?;

    Ok(true)
}

/// The call matching for the driver: N_up and N_down, or how many samples the vote record
/// must hold first. The kept results that exist already are checked.
pub fn match_for_driver(dir: &Path, frozen: &Frozen) -> Result<Matching> {
    let run = load_b(dir, frozen)?;
    let m = match_run(&run)?;

    match m.matching {
        Matching::Matched(x) => {
            if dir.join(VOTE_UP).exists() {
                read_kept(&dir.join(VOTE_UP), x.n_up, &run, &m)?;
            }

            match x.n_down {
                Some(n) if dir.join(VOTE_DOWN).exists() => {
                    read_kept(&dir.join(VOTE_DOWN), n, &run, &m)?;
                }
                None if dir.join(VOTE_DOWN).exists() => {
                    return Err(refused(format!(
                        "{}: {VOTE_DOWN} exists, but no N_down exists",
                        dir.display()
                    )));
                }
                _ => {}
            }
        }
        Matching::Need(_) => {
            if dir.join(VOTE_UP).exists() || dir.join(VOTE_DOWN).exists() {
                return Err(refused(format!(
                    "{}: kept vote results exist, but the vote record holds fewer samples than the call matching needs",
                    dir.display()
                )));
            }
        }
    }

    Ok(m.matching)
}

/// What one breeding run cost, for the report.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Cost {
    /// Worker calls the run counted, as its report says.
    pub worker_calls: u64,
    /// The run's own record: its calls (some taken from the shared call cache) and tokens.
    #[serde(flatten)]
    pub record: Totals,
    pub mutator_answers_used: u64,
    /// Lines of `llm/mutator-calls.jsonl`, and of `llm/mutator-errors.jsonl` (asked again).
    pub mutator_calls: u64,
    pub mutator_failed_calls: u64,
    /// The summed wall time of the committed generations: no select phase, no Mutator
    /// pause.
    pub generation_seconds: f64,
    /// The wall time of the whole breeding from the driver's journal: from the first
    /// start of brut-breed-ollama to its exit with 0, Mutator rounds included. None
    /// without a journal.
    pub wall_seconds: Option<u64>,
}

/// The driver's journal of the run in `dir`, `driver.log`: per line the time in seconds
/// and what happened. Empty without a journal; a line that does not read is skipped.
fn journal(dir: &Path) -> Vec<(u64, String)> {
    std::fs::read_to_string(dir.join(JOURNAL))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            let (t, what) = l.split_once(' ')?;

            Some((t.parse().ok()?, what.to_string()))
        })
        .collect()
}

/// Seconds from the first journal line that starts with `from` to the last that equals
/// `to` after it.
fn span(journal: &[(u64, String)], from: &str, to: &str) -> Option<u64> {
    let start = journal.iter().position(|(_, w)| w.starts_with(from))?;
    let end = journal
        .iter()
        .rposition(|(_, w)| w == to)
        .filter(|&e| e > start)?;

    Some(journal[end].0.saturating_sub(journal[start].0))
}

/// The cost of `run`, with the totals of its record.
fn cost(run: &Run, record: Totals) -> Result<Cost> {
    Ok(Cost {
        worker_calls: run.report.calls_used,
        record,
        mutator_answers_used: run.report.mutator_answers_used,
        mutator_calls: count_lines(&run.dir.join(MUTATOR_RECORD))?,
        mutator_failed_calls: count_lines(&run.dir.join(MUTATOR_ERRORS))?,
        generation_seconds: run.generation_seconds,
        wall_seconds: span(&journal(&run.dir), "breed start", "breed exit 0"),
    })
}

/// The counts of one pair on the select puzzles that count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counts {
    /// C(s), V(s), the vote's count with N_down(s), the plain run of the voted seed, C_L(s).
    pub c: usize,
    pub v: usize,
    pub v_down: Option<usize>,
    pub seed_plain: usize,
    pub c_l: usize,
    /// The McNemar counts of vote-control: puzzles only the vote solved, only the champion.
    pub vote_only: usize,
    pub champion_only: usize,
}

/// The counts of section 5 and 7 for the pair (`b`, `l`) on the puzzles `m.counted` marks.
/// `up` and `down` are the kept votes with N_up and N_down samples.
pub fn counts(
    b: &Run,
    l: &Run,
    m: &VoteMatch,
    up: &VoteReport,
    down: Option<&VoteReport>,
) -> Result<Counts> {
    let champion = &b.archive()[m.champion].id;
    let champion_l = l_champion(l)?;
    let on = |r: &VoteReport, f: &dyn Fn(&PuzzleVote) -> bool| {
        r.puzzles
            .iter()
            .zip(&m.counted)
            .filter(|&(p, &c)| c && f(p))
            .count()
    };

    Ok(Counts {
        c: b.solved_on(champion, &m.counted),
        v: on(up, &|p| p.correct),
        v_down: down.map(|d| on(d, &|p| p.correct)),
        seed_plain: b.solved_on(&m.plan.cascade.member, &m.counted),
        c_l: l.solved_on(&l.archive()[champion_l].id, &m.counted),
        vote_only: on(up, &|p| p.correct && p.champion_correct == Some(false)),
        champion_only: on(up, &|p| !p.correct && p.champion_correct == Some(true)),
    })
}

/// The champion of run L-s.
fn l_champion(l: &Run) -> Result<usize> {
    vote::champion(l.archive(), l.cfg().finalists).ok_or_else(|| {
        refused(format!(
            "{}: the run has no bred finalist, so no champion",
            l.dir.display()
        ))
    })
}

/// One complete pair: everything section 7 reports for it.
#[derive(Clone, Debug, Serialize)]
pub struct Pair {
    pub seed: u64,
    /// Select puzzles whose id is an id of the search panel; they count nowhere.
    pub overlap: Vec<String>,
    pub counted_puzzles: usize,
    pub champion: String,
    pub champion_generation: usize,
    pub champion_workers: usize,
    /// The voted seed member, and the id of its Cascade at the vote's temperature.
    pub voted_member: String,
    pub voted_cascade: String,
    /// The search puzzles the voted seed, the best seed Cascade, solves (the pilot gate of
    /// section 9), and the search panel's size.
    pub seed_search_solved: usize,
    pub search_puzzles: usize,
    pub c: usize,
    pub k_c: u64,
    pub n_up: usize,
    pub k_v_up: u64,
    pub v: usize,
    pub y: &'static str,
    pub n_down: Option<usize>,
    pub k_v_down: Option<u64>,
    pub v_down: Option<usize>,
    pub y_down: Option<&'static str>,
    pub seed_plain: usize,
    pub y1: &'static str,
    pub champion_l: String,
    pub c_l: usize,
    pub y2: &'static str,
    pub vote_only: usize,
    pub champion_only: usize,
    /// The exact two-sided sign test of vote-control on the two counts above.
    pub sign_test_p: f64,
    /// K_c(s) per counted puzzle.
    pub champion_calls_per_puzzle: f64,
    /// The vote with N_up samples on every select puzzle: calls and tokens.
    pub vote_calls: u64,
    pub vote_prompt_tokens: u64,
    pub vote_completion_tokens: u64,
    /// Vote calls that failed once and were asked again (`vote-control/worker-errors.jsonl`).
    pub vote_failed_calls_retried: u64,
    /// The wall time of the vote from the driver's journal: from the first vote-control
    /// to the last that ended with 0. None without a journal.
    pub vote_wall_seconds: Option<u64>,
    pub cost_b: Cost,
    pub cost_l: Cost,
    /// Y, Y1 and Y2, for the rule.
    #[serde(skip)]
    pub outcomes: [Outcome; 3],
}

/// How far a seed of the list got.
#[derive(Clone, Debug)]
pub enum SeedState {
    NotStarted,
    /// Started, not complete yet, and why.
    Running(String),
    Complete(Box<Pair>),
}

/// The state of `seed` in `runs`, with every check of the module docs for what exists.
pub fn seed_state(runs: &Path, seed: u64, frozen: &Frozen) -> Result<SeedState> {
    let b = arm_state(runs, Arm::B, seed, frozen)?;
    let l = arm_state(runs, Arm::L, seed, frozen)?;

    match (b, l) {
        (ArmState::Missing, ArmState::Missing) => Ok(SeedState::NotStarted),
        (ArmState::Finished(b), ArmState::Finished(l)) => pair_state(&b, &l),
        (b, l) => Ok(SeedState::Running(format!(
            "arm B {}, arm L {}",
            b.word(),
            l.word()
        ))),
    }
}

/// The state of a seed whose two runs are finished.
fn pair_state(b: &Run, l: &Run) -> Result<SeedState> {
    if ids(&l.search) != ids(&b.search) || ids(&l.select) != ids(&b.select) {
        return Err(broken(format!(
            "{} and {} have other panels",
            b.dir.display(),
            l.dir.display()
        )));
    }

    let seeds = |r: &Run| -> Vec<(String, Vec<Option<Grid>>)> {
        r.archive()
            .iter()
            .filter(|m| m.op == Op::Seed)
            .map(|m| (m.id.clone(), m.answers.clone()))
            .collect()
    };

    if seeds(b) != seeds(l) {
        return Err(refused(format!(
            "seed {}: the seed Cascades' search answers differ between arms B and L (section 4)",
            b.seed()
        )));
    }

    let m = match_run(b)?;
    let up_path = b.dir.join(VOTE_UP);
    let down_path = b.dir.join(VOTE_DOWN);
    let matched = match m.matching {
        Matching::Matched(x) => x,
        Matching::Need(n) => {
            if up_path.exists() || down_path.exists() {
                return Err(refused(format!(
                    "{}: kept vote results exist, but the vote record holds {} samples and the call matching needs at least {n}",
                    b.dir.display(),
                    m.samples.len()
                )));
            }

            return Ok(SeedState::Running(format!(
                "the vote record holds {} of at least {n} samples",
                m.samples.len()
            )));
        }
    };

    if !up_path.exists() {
        return Ok(SeedState::Running(format!("no {VOTE_UP} yet")));
    }

    let up = read_kept(&up_path, matched.n_up, b, &m)?;
    let down = match matched.n_down {
        Some(n) if down_path.exists() => Some(read_kept(&down_path, n, b, &m)?),
        Some(_) => return Ok(SeedState::Running(format!("no {VOTE_DOWN} yet"))),
        None if down_path.exists() => {
            return Err(refused(format!(
                "{}: {VOTE_DOWN} exists, but no N_down exists",
                b.dir.display()
            )));
        }
        None => None,
    };

    if !b.dir.join(PANELS).exists() {
        return Ok(SeedState::Running(format!("no {PANELS} yet")));
    }

    check_panels(b)?;

    let k = counts(b, l, &m, &up, down.as_ref())?;
    let champion = &b.archive()[m.champion];
    let champion_l = &l.archive()[l_champion(l)?];

    for (run, id) in [
        (b, &champion.id),
        (b, &m.plan.cascade.member),
        (l, &champion_l.id),
    ] {
        run.check_logged(id)?;
    }

    let y = Outcome::of(k.c, k.v);
    let y1 = Outcome::of(k.c, k.seed_plain);
    let y2 = Outcome::of(k.c_l, k.c);
    let counted_puzzles = m.counted.iter().filter(|&&c| c).count();
    let (_, l_totals) = replay_select(l, l_champion(l)?)?;

    Ok(SeedState::Complete(Box::new(Pair {
        seed: b.seed(),
        overlap: m.overlap.clone(),
        counted_puzzles,
        champion: champion.id.clone(),
        champion_generation: champion.generation,
        champion_workers: champion.cascade.worker_count(),
        voted_member: m.plan.cascade.member.clone(),
        voted_cascade: m.plan.cascade.id.clone(),
        seed_search_solved: m.plan.cascade.search_solved,
        search_puzzles: b.search.len(),
        c: k.c,
        k_c: matched.k_c,
        n_up: matched.n_up,
        k_v_up: matched.k_v_up,
        v: k.v,
        y: y.name(Y_NAMES),
        n_down: matched.n_down,
        k_v_down: matched.k_v_down,
        v_down: k.v_down,
        y_down: k.v_down.map(|v| Outcome::of(k.c, v).name(Y_NAMES)),
        seed_plain: k.seed_plain,
        y1: y1.name(Y1_NAMES),
        champion_l: champion_l.id.clone(),
        c_l: k.c_l,
        y2: y2.name(Y2_NAMES),
        vote_only: k.vote_only,
        champion_only: k.champion_only,
        sign_test_p: vote::sign_test(k.vote_only, k.champion_only),
        champion_calls_per_puzzle: matched.k_c as f64 / counted_puzzles.max(1) as f64,
        vote_calls: up.calls_used,
        vote_prompt_tokens: up.prompt_tokens,
        vote_completion_tokens: up.completion_tokens,
        vote_failed_calls_retried: count_lines(&b.dir.join(VOTE_ERRORS))?,
        vote_wall_seconds: span(&journal(&b.dir), "vote start", "vote exit 0"),
        cost_b: cost(b, m.totals)?,
        cost_l: cost(l, l_totals)?,
        outcomes: [y, y1, y2],
    })))
}

/// The seeds of the runs in `runs`: every directory named `B-<seed>` or `L-<seed>`. Other
/// names are not runs and are skipped; a name like a run's that is not plain decimal is
/// refused.
fn started_seeds(runs: &Path) -> Result<BTreeSet<u64>> {
    let mut seeds = BTreeSet::new();
    let entries =
        std::fs::read_dir(runs).map_err(|e| refused(format!("{}: {e}", runs.display())))?;

    for entry in entries {
        let entry = entry.map_err(|e| refused(format!("{}: {e}", runs.display())))?;

        if !entry.path().is_dir() {
            continue;
        }

        let name = entry.file_name().to_string_lossy().to_string();

        match parse_run_name(&name) {
            Some((_, seed)) => {
                seeds.insert(seed);
            }
            None if (name.starts_with("B-") || name.starts_with("L-"))
                && name.len() > 2
                && name[2..].bytes().all(|b| b.is_ascii_digit()) =>
            {
                return Err(refused(format!(
                    "{}: write the seed in plain decimal",
                    entry.path().display()
                )));
            }
            None => {}
        }
    }

    Ok(seeds)
}

/// Error unless the seeds `found` lie on `list`, form its first seeds without a gap, and
/// stay within its cap.
fn check_list(found: &BTreeSet<u64>, list: List) -> Result<()> {
    let seeds = list.seeds();
    let on: BTreeSet<u64> = seeds.iter().copied().collect();
    let off: Vec<u64> = found.difference(&on).copied().collect();

    if list == List::Pilot && !off.is_empty() {
        return Err(refused(format!(
            "seeds {off:?} are not pilot seeds {PILOT:?}"
        )));
    }

    let pilot: Vec<u64> = off.iter().copied().filter(|s| PILOT.contains(s)).collect();
    let beyond: Vec<u64> = off
        .iter()
        .copied()
        .filter(|&s| s > CAP as u64 && !PILOT.contains(&s))
        .collect();
    let other: Vec<u64> = off.iter().copied().filter(|&s| s == 0).collect();

    if !pilot.is_empty() {
        return Err(refused(format!(
            "seeds {pilot:?} are pilot seeds; pilot seeds never count. Analyse them with --pilot in their own runs directory"
        )));
    }

    if !beyond.is_empty() {
        return Err(refused(format!(
            "seeds {beyond:?} lie beyond the list: the experiment counts at most {CAP} pairs (D6)"
        )));
    }

    if !other.is_empty() {
        return Err(refused(format!(
            "seeds {other:?} are not on the list 1 to {CAP}"
        )));
    }

    if let Some(gap) = seeds
        .iter()
        .enumerate()
        .find(|&(i, s)| !found.contains(s) && seeds[i + 1..].iter().any(|t| found.contains(t)))
        .map(|(_, s)| *s)
    {
        return Err(refused(format!(
            "seed {gap} has no run, but later seeds of the list do: pairs enter in the order of the list, without gaps"
        )));
    }

    Ok(())
}

/// An evidence value as the result shows it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Shown {
    pub up: u32,
    pub down: u32,
    /// The exact fraction 3^up / 2^(up + down).
    pub fraction: String,
    pub value: f64,
}

impl From<Evidence> for Shown {
    fn from(e: Evidence) -> Shown {
        Shown {
            up: e.up,
            down: e.down,
            fraction: e.fraction(),
            value: e.value(),
        }
    }
}

/// One pair of the result, in list order.
#[derive(Clone, Debug, Serialize)]
pub struct Row {
    /// Its place in the list, from 1.
    pub pair: usize,
    /// False for a pair after the stop.
    pub counted: bool,
    #[serde(flatten)]
    pub data: Pair,
    /// The evidence values after this pair; none after the stop.
    pub e_b: Option<Shown>,
    pub e_v: Option<Shown>,
    pub e_s1: Option<Shown>,
    pub e_s2: Option<Shown>,
}

/// A started seed that is not counted yet.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Waiting {
    pub seed: u64,
    pub why: String,
}

/// Where the rule stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Stop {
    pub pair: usize,
    pub seed: u64,
}

/// The result of the analysis.
#[derive(Clone, Debug, Serialize)]
pub struct Analysis {
    /// "counted" or "pilot".
    pub list: &'static str,
    pub cap: usize,
    pub proof: u32,
    pub futility: u32,
    pub secondary: u32,
    pub temperature: f32,
    /// The complete pairs at the head of the list.
    pub pairs: Vec<Row>,
    /// Started seeds after them.
    pub waiting: Vec<Waiting>,
    pub counted_pairs: usize,
    pub stop: Option<Stop>,
    pub primary: Primary,
    /// The values after the last counted pair: at the stop, or now.
    pub e_b: Shown,
    pub e_v: Shown,
    pub e_s1: Shown,
    pub e_s2: Shown,
    /// The secondary claims, read at the stop (E >= 200); none before a stop or in the
    /// pilot.
    pub s1_claim: Option<bool>,
    pub s2_claim: Option<bool>,
    /// The next seed of the list to start, while the rule runs on.
    pub next_seed: Option<u64>,
    pub verdict: Vec<String>,
}

/// The rows of the complete pairs at the head of the list: those up to the stop with
/// their evidence values, the later ones listed as not counted.
fn rows(complete: &[&Pair], rule: &Rule) -> Vec<Row> {
    complete
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let step = rule.steps.get(i);

            Row {
                pair: i + 1,
                counted: step.is_some(),
                data: (*p).clone(),
                e_b: step.map(|s| s.e_b.into()),
                e_v: step.map(|s| s.e_v().into()),
                e_s1: step.map(|s| s.e_s1.into()),
                e_s2: step.map(|s| s.e_s2.into()),
            }
        })
        .collect()
}

/// The analysis of the runs in `runs` on `list`, against the `frozen` settings.
pub fn analyse(runs: &Path, frozen: &Frozen, list: List) -> Result<Analysis> {
    let seeds = list.seeds();
    let found = started_seeds(runs)?;

    check_list(&found, list)?;

    let mut states = Vec::new();

    for &seed in seeds.iter().filter(|s| found.contains(s)) {
        states.push((seed, seed_state(runs, seed, frozen)?));
    }

    let complete: Vec<&Pair> = states
        .iter()
        .map_while(|(_, st)| match st {
            SeedState::Complete(p) => Some(p.as_ref()),
            _ => None,
        })
        .collect();
    let outcomes: Vec<[Outcome; 3]> = complete.iter().map(|p| p.outcomes).collect();
    let rule = sequential(&outcomes, seeds.len());
    let pairs = rows(&complete, &rule);
    let waiting: Vec<Waiting> = states[complete.len()..]
        .iter()
        .map(|(seed, st)| Waiting {
            seed: *seed,
            why: match st {
                SeedState::Running(why) => why.clone(),
                _ => format!("complete; it enters after seed {}", seeds[complete.len()]),
            },
        })
        .collect();
    let last = rule.steps.last().copied().unwrap_or_default();
    let stop = rule.stop.map(|k| Stop {
        pair: k,
        seed: seeds[k - 1],
    });
    let claims = (list == List::Counted && stop.is_some())
        .then(|| (last.e_s1.at_least(SECONDARY), last.e_s2.at_least(SECONDARY)));
    let next_seed = match stop {
        None => seeds.get(states.len()).copied(),
        Some(_) => None,
    };
    let mut a = Analysis {
        list: list.name(),
        cap: seeds.len(),
        proof: PROOF,
        futility: FUTILITY,
        secondary: SECONDARY,
        temperature: TEMPERATURE,
        counted_pairs: rule.steps.len(),
        pairs,
        waiting,
        stop,
        primary: rule.primary,
        e_b: last.e_b.into(),
        e_v: last.e_v().into(),
        e_s1: last.e_s1.into(),
        e_s2: last.e_s2.into(),
        s1_claim: claims.map(|c| c.0),
        s2_claim: claims.map(|c| c.1),
        next_seed,
        verdict: Vec::new(),
    };

    a.verdict = verdict(&a, list);

    Ok(a)
}

/// The verdict lines of `a`.
fn verdict(a: &Analysis, list: List) -> Vec<String> {
    let mut lines = Vec::new();
    let e = |s: &Shown| format!("{} = {:.4}", s.fraction, s.value);

    if list == List::Pilot {
        lines.push("Pilot seeds: shown for illustration only. Pilot seeds never count, and no claim is made.".to_string());

        if let Some(gate) = pilot_gate(&a.pairs) {
            lines.push(gate);
        }
    }

    lines.push(match (a.primary, a.stop) {
        (Primary::Proof, Some(s)) => format!(
            "Primary: proof at pair {} (seed {}): breeding beats the vote. E_B = {} >= {PROOF}.",
            s.pair,
            s.seed,
            e(&a.e_b)
        ),
        (Primary::Futility, Some(s)) => format!(
            "Primary: futility at pair {} (seed {}): no evidence that breeding wins in three of four untied seeds. E_B = {} <= 1/{FUTILITY}.",
            s.pair,
            s.seed,
            e(&a.e_b)
        ),
        (Primary::Open, Some(s)) => format!(
            "Primary: open. The cap of {} pairs is reached at seed {} without a claim. E_B = {}.",
            a.cap,
            s.seed,
            e(&a.e_b)
        ),
        _ => format!("Primary: no stop after {} counted pair(s). E_B = {}.", a.counted_pairs, e(&a.e_b)),
    });
    lines.push(format!("E_V = {}, reported without a claim.", e(&a.e_v)));

    for (name, what, value, claim) in [
        (
            "S1",
            "the champion beats one plain run of the voted seed Cascade",
            &a.e_s1,
            a.s1_claim,
        ),
        (
            "S2",
            "lexicase selection beats Darwin Gödel selection",
            &a.e_s2,
            a.s2_claim,
        ),
    ] {
        lines.push(match claim {
            Some(true) => format!("{name}: E = {} >= {SECONDARY}: {what}.", e(value)),
            Some(false) => format!(
                "{name}: E = {}: no claim (a claim needs E >= {SECONDARY}).",
                e(value)
            ),
            None if list == List::Pilot => {
                format!("{name}: E = {}: pilot seeds make no claim.", e(value))
            }
            None => format!(
                "{name}: E = {} after {} pair(s); it is read at the stop.",
                e(value),
                a.counted_pairs
            ),
        });
    }

    if a.stop.is_none() {
        let waiting: Vec<u64> = a.waiting.iter().map(|w| w.seed).collect();

        lines.push(match (a.next_seed, waiting.is_empty()) {
            (Some(s), true) => format!("Next: run seed {s}."),
            (Some(s), false) => {
                format!("Next: run seed {s}; seeds {waiting:?} are started and not counted yet.")
            }
            (None, _) => format!("Next: wait for seeds {waiting:?}."),
        });
    }

    lines
}

/// The pilot gate of section 9: the share of the search puzzles that the best seed
/// Cascade solves in the median pilot seed must be at least 15 %. In integers: the
/// median count m of n puzzles passes when 100 m >= 15 n; the median of an even number of
/// seeds is the mean of the middle two. None without pairs.
fn pilot_gate(pairs: &[Row]) -> Option<String> {
    let mut solved: Vec<usize> = pairs.iter().map(|r| r.data.seed_search_solved).collect();
    let n = pairs.first()?.data.search_puzzles;

    solved.sort_unstable();

    let k = solved.len();
    // Twice the median, so that it stays whole.
    let twice = solved[(k - 1) / 2] + solved[k / 2];
    let passes = 100 * twice >= 15 * 2 * n;

    Some(format!(
        "Pilot gate (section 9): the best seed Cascade solves {:.1} of {n} search puzzles in the median pilot seed ({:.1} %); {}.",
        twice as f64 / 2.0,
        50.0 * twice as f64 / n as f64,
        if passes {
            "the gate of 15 % passes"
        } else {
            "below 15 %: the model has no selection signal, and D1 changes before the freeze"
        }
    ))
}

/// `a` as tables and its verdict lines.
pub fn render(a: &Analysis) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let opt = |v: Option<usize>| v.map_or("-".to_string(), |n| n.to_string());
    let shown = |s: &Option<Shown>| {
        s.as_ref()
            .map_or("-".to_string(), |s| format!("{:.4}", s.value))
    };

    let _ = writeln!(
        out,
        "Breeding against the vote (docs/PREREG_BREEDING_VS_VOTE.md), {} seeds, {} complete pair(s), {} counted.\n",
        a.list,
        a.pairs.len(),
        a.counted_pairs
    );
    let _ = writeln!(
        out,
        "{:>4} {:>5} {:>7} {:>4} {:>4} {:>4} {:>7} {:>7} {:<14} {:>10} {:>10}",
        "pair", "seed", "overlap", "C", "V", "N_up", "K_c", "K_v_up", "Y", "E_B", "E_V"
    );

    for r in &a.pairs {
        let d = &r.data;
        let _ = writeln!(
            out,
            "{:>4} {:>5} {:>7} {:>4} {:>4} {:>4} {:>7} {:>7} {:<14} {:>10} {:>10}{}",
            r.pair,
            d.seed,
            d.overlap.len(),
            d.c,
            d.v,
            d.n_up,
            d.k_c,
            d.k_v_up,
            d.y,
            shown(&r.e_b),
            shown(&r.e_v),
            if r.counted {
                ""
            } else {
                "  after the stop, not counted"
            }
        );
    }

    let _ = writeln!(
        out,
        "\n{:>4} {:>5} {:>6} {:>9} {:>6} {:<14} {:>5} {:<14} {:>4} {:<14} {:>10} {:>10}",
        "pair",
        "seed",
        "N_down",
        "K_v_down",
        "V_down",
        "Y with N_down",
        "plain",
        "Y1",
        "C_L",
        "Y2",
        "E_S1",
        "E_S2"
    );

    for r in &a.pairs {
        let d = &r.data;
        let _ = writeln!(
            out,
            "{:>4} {:>5} {:>6} {:>9} {:>6} {:<14} {:>5} {:<14} {:>4} {:<14} {:>10} {:>10}",
            r.pair,
            d.seed,
            opt(d.n_down),
            d.k_v_down.map_or("-".to_string(), |k| k.to_string()),
            opt(d.v_down),
            d.y_down.unwrap_or("-"),
            d.seed_plain,
            d.y1,
            d.c_l,
            d.y2,
            shown(&r.e_s1),
            shown(&r.e_s2)
        );
    }

    let _ = writeln!(
        out,
        "\n{:>4} {:>5} {:>8} {:>6} {:>6} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8}",
        "pair",
        "seed",
        "calls/pz",
        "v_only",
        "c_only",
        "sign p",
        "calls B",
        "calls L",
        "mut B",
        "mut L",
        "vote",
        "min B",
        "min L"
    );

    let minutes = |c: &Cost| match c.wall_seconds {
        Some(s) => format!("{:.1}", s as f64 / 60.0),
        None => "-".to_string(),
    };

    for r in &a.pairs {
        let d = &r.data;
        let _ = writeln!(
            out,
            "{:>4} {:>5} {:>8.2} {:>6} {:>6} {:>8.4} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8}",
            r.pair,
            d.seed,
            d.champion_calls_per_puzzle,
            d.vote_only,
            d.champion_only,
            d.sign_test_p,
            d.cost_b.worker_calls,
            d.cost_l.worker_calls,
            d.cost_b.mutator_calls,
            d.cost_l.mutator_calls,
            d.vote_calls,
            minutes(&d.cost_b),
            minutes(&d.cost_l)
        );
    }

    if !a.waiting.is_empty() {
        let _ = writeln!(out);

        for w in &a.waiting {
            let _ = writeln!(out, "seed {}: not counted yet: {}", w.seed, w.why);
        }
    }

    let _ = writeln!(out);

    for line in &a.verdict {
        let _ = writeln!(out, "{line}");
    }

    out
}

/// The most Worker calls a run of `cfg` can make, by the call bounds of
/// `src/cascade/arena.rs` (D4, written out in the pre-registration's section 4).
///
/// A Cascade of w Workers per stage makes at most w + w * w(w - 1)/2 calls per stage and
/// puzzle (`stage_call_bound`): every Worker answers, then every Worker judges every pair
/// of distinct answers. Generation 0 runs the seed Cascades on the search panel. Each bred
/// generation runs at most `children_per_generation` Cascades, each at most of the widest
/// shape the limits allow, and its peer round makes at most one judge call per new member,
/// opponent and puzzle (`peer_call_bound`). The select phase runs every seed Cascade and
/// at most `finalists` Cascades of the widest shape on the select panel. Mutator calls
/// are not Worker calls and do not count.
pub fn worst_case_calls(cfg: &BreedConfig) -> Result<u64> {
    let seeds = seeds_for(&cfg.limits)?;
    let seed_calls: u64 = seeds.iter().map(|(_, c)| cascade_call_bound(c)).sum();
    let worker = seeds
        .first()
        .and_then(|(_, c)| c.stages.first())
        .and_then(|s| s.workers.first())
        .cloned()
        .ok_or_else(|| refused("no seed Cascade"))?;
    let widest = cascade_call_bound(&Cascade {
        stages: vec![
            Stage {
                workers: vec![worker; cfg.limits.width()],
            };
            cfg.limits.stages()
        ],
    });
    let search = cfg.search_puzzles as u64;
    let per_pair = cfg.peer_puzzles_per_pair.min(cfg.search_puzzles) as u64;
    let peer = |n: u64, archive: u64| {
        n * (cfg.peer_opponents as u64).min((archive + n).saturating_sub(1)) * per_pair
    };
    let children = cfg.children_per_generation as u64;
    let mut archive = seeds.len() as u64;
    let mut total = seed_calls * search + peer(archive, 0);

    for _ in 1..=cfg.generations {
        total += children * widest * search + peer(children, archive);
        archive += children;
    }

    Ok(total + (seed_calls + cfg.finalists as u64 * widest) * cfg.select_puzzles as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    use brutmaschine::cascade::arena::{ParentSelection, cascade_call_bound};
    use brutmaschine::cascade::genome::{Stage, seed_cascades};

    use Outcome::{First as W, Second as L, Tie as T};

    #[test]
    fn naturals_are_exact_beyond_u128() {
        assert_eq!(Natural::power(3, 0).to_string(), "1");
        assert_eq!(Natural::power(2, 10).to_string(), "1024");
        assert_eq!(Natural::power(3, 20).to_string(), "3486784401");
        assert_eq!(Natural::power(10, 9).to_string(), "1000000000");
        assert_eq!(
            Natural::power(3, 120).to_string(),
            "1797010299914431210413179829509605039731475627537851106401"
        );
        assert!(Natural::power(3, 120) > Natural::power(2, 190));
        assert!(Natural::power(3, 120) < Natural::power(2, 191));
        assert_eq!(
            Natural::power(2, 3).times(25),
            Natural::power(10, 2).times(2)
        );
        assert_eq!(
            Natural::power(3, 4).cmp(&Natural::power(3, 4)),
            Ordering::Equal
        );
    }

    #[test]
    fn evidence_is_three_halves_up_and_one_half_down() {
        let e = Evidence::default();

        assert_eq!(e.fraction(), "1/1");
        assert_eq!(e.step(W).fraction(), "3/2");
        assert_eq!(e.step(L).fraction(), "1/2");
        assert_eq!(e.step(T), e);
        assert_eq!(e.step(W).step(W).step(L).fraction(), "9/8");
        assert_eq!(
            Evidence { up: 2, down: 1 }.mirrored(),
            Evidence { up: 1, down: 2 }
        );

        // 3^12 = 531441 >= 100 * 2^12 = 409600, but 3^11 = 177147 < 100 * 2^11 = 204800.
        assert!(Evidence { up: 12, down: 0 }.at_least(PROOF));
        assert!(!Evidence { up: 11, down: 0 }.at_least(PROOF));
        // Five straight losses: 20 * 1 <= 32; four: 20 > 16.
        assert!(Evidence { up: 0, down: 5 }.at_most_one_in(FUTILITY));
        assert!(!Evidence { up: 0, down: 4 }.at_most_one_in(FUTILITY));
        // The boundary itself counts: E = 1/20 exactly is impossible, but E = 100 * 2^n / 2^n
        // is checked with >=, so the integers decide, never a float.
        assert!(Evidence { up: 0, down: 0 }.at_least(1));
        assert!(Evidence { up: 0, down: 0 }.at_most_one_in(1));
    }

    fn rule(outcomes: &[[Outcome; 3]]) -> Rule {
        sequential(outcomes, 60)
    }

    #[test]
    fn the_rule_proves_at_the_first_pair_with_e_b_at_least_100() {
        // Twelve straight wins: 3^12 / 2^12 = 129.7 >= 100 after pair 12, not before.
        let wins = vec![[W, T, T]; 15];
        let r = rule(&wins);

        assert_eq!((r.stop, r.primary), (Some(12), Primary::Proof));
        assert_eq!(r.steps.len(), 12, "pairs after the stop are not counted");
        assert_eq!(r.steps[11].e_b, Evidence { up: 12, down: 0 });
        assert_eq!(r.steps[11].e_v(), Evidence { up: 0, down: 12 });

        // Ties leave E_B as it is, so they only delay the proof.
        let mut tied = vec![[T, T, T]; 7];
        tied.extend(vec![[W, T, T]; 12]);
        let r = rule(&tied);

        assert_eq!((r.stop, r.primary), (Some(19), Primary::Proof));
        assert_eq!(r.steps[18].e_b, Evidence { up: 12, down: 0 });

        // With one loss, 3^14 = 4782969 >= 100 * 2^15 = 3276800 at 14 wins, not at 13.
        let mut one_loss = vec![[L, T, T]];
        one_loss.extend(vec![[W, T, T]; 20]);
        let r = rule(&one_loss);

        assert_eq!((r.stop, r.primary), (Some(15), Primary::Proof));
    }

    #[test]
    fn the_rule_stops_for_futility_at_e_b_at_most_one_in_20() {
        let r = rule(&[[L, W, W]; 10]);

        assert_eq!((r.stop, r.primary), (Some(5), Primary::Futility));
        assert_eq!(r.steps[4].e_b, Evidence { up: 0, down: 5 });

        // A win first: 20 * 3 <= 2^n needs n >= 6, so five more losses.
        let mut late = vec![[W, T, T]];
        late.extend(vec![[L, T, T]; 10]);
        let r = rule(&late);

        assert_eq!((r.stop, r.primary), (Some(6), Primary::Futility));
        assert_eq!(r.steps[5].e_b, Evidence { up: 1, down: 5 });
    }

    #[test]
    fn the_rule_ends_open_at_the_cap_and_runs_on_before_it() {
        // E_B = 1 forever on ties: no claim either way, open at the cap.
        let r = sequential(&vec![[T, T, T]; 70], 60);

        assert_eq!((r.stop, r.primary), (Some(60), Primary::Open));
        assert_eq!(r.steps.len(), 60);

        // Fewer pairs than the cap, no threshold crossed: the rule runs on.
        let r = sequential(&[[W, L, T], [L, T, W], [T, W, W]], 60);

        assert_eq!((r.stop, r.primary), (None, Primary::Running));
        assert_eq!(r.steps.len(), 3);
        assert_eq!(r.steps[2].e_b, Evidence { up: 1, down: 1 });
        assert_eq!(r.steps[2].e_s1, Evidence { up: 1, down: 1 });
        assert_eq!(r.steps[2].e_s2, Evidence { up: 2, down: 0 });

        // A proof on the cap pair is a proof, not an open end.
        let mut at_cap = vec![[T, T, T]; 48];
        at_cap.extend(vec![[W, T, T]; 12]);
        let r = sequential(&at_cap, 60);

        assert_eq!((r.stop, r.primary), (Some(60), Primary::Proof));

        // No pairs at all.
        assert_eq!(
            sequential(&[], 60),
            Rule {
                steps: Vec::new(),
                stop: None,
                primary: Primary::Running
            }
        );
    }

    #[test]
    fn alternating_wins_and_losses_drift_down_to_futility() {
        // Each win-loss couple multiplies E_B by 3/4; 20 * 3^w <= 2^(2w) from w = 11 on.
        let pairs: Vec<[Outcome; 3]> = (0..60)
            .map(|i| if i % 2 == 0 { [W, T, T] } else { [L, T, T] })
            .collect();
        let r = rule(&pairs);

        assert_eq!(r.primary, Primary::Futility);
        assert_eq!(
            r.steps.last().map(|s| s.e_b),
            Some(Evidence { up: 11, down: 11 })
        );
        assert_eq!(r.stop, Some(22));
    }

    #[test]
    fn the_secondary_values_follow_their_own_outcomes_and_need_200() {
        // S1 wins 14 times (3^14 / 2^14 = 291.9 >= 200), S2 loses all; the primary is open.
        let pairs = vec![[T, W, L]; 14];
        let r = sequential(&pairs, 14);

        assert_eq!((r.stop, r.primary), (Some(14), Primary::Open));

        let last = r.steps[13];

        assert_eq!(last.e_s1, Evidence { up: 14, down: 0 });
        assert!(last.e_s1.at_least(SECONDARY));
        assert_eq!(last.e_s2, Evidence { up: 0, down: 14 });
        assert!(!last.e_s2.at_least(SECONDARY));

        // 13 wins give 3^13 / 2^13 = 194.6: above the primary threshold, below 200.
        assert!(Evidence { up: 13, down: 0 }.at_least(PROOF));
        assert!(!Evidence { up: 13, down: 0 }.at_least(SECONDARY));
    }

    #[test]
    fn outcomes_compare_counts() {
        assert_eq!(Outcome::of(3, 2), W);
        assert_eq!(Outcome::of(2, 3), L);
        assert_eq!(Outcome::of(0, 0), T);
    }

    #[test]
    fn n_up_is_the_smallest_and_n_down_the_largest_n_the_definitions_allow() {
        // K_c = 250 on 100 counted puzzles, one call per sample and puzzle.
        let m = match_calls(250, &[100, 100, 100, 100], 100).expect("match");

        assert_eq!(
            m,
            Matching::Matched(Matched {
                k_c: 250,
                n_up: 3,
                k_v_up: 300,
                n_down: Some(2),
                k_v_down: Some(200)
            })
        );

        // K_v(N) = K_c exactly: N_up = N_down, since one more sample adds at least `floor`.
        let m = match_calls(300, &[100, 100, 100], 100).expect("match");

        assert_eq!(
            m,
            Matching::Matched(Matched {
                k_c: 300,
                n_up: 3,
                k_v_up: 300,
                n_down: Some(3),
                k_v_down: Some(300)
            })
        );

        // The champion spends less than one sample: N_up = 1 and no N_down.
        let m = match_calls(150, &[200, 200], 200).expect("match");

        assert_eq!(
            m,
            Matching::Matched(Matched {
                k_c: 150,
                n_up: 1,
                k_v_up: 200,
                n_down: None,
                k_v_down: None
            })
        );

        // Samples of unequal cost (a Cascade with a stage vote) are summed as they are.
        let m = match_calls(530, &[120, 180, 100, 140, 300], 100).expect("match");

        assert_eq!(
            m,
            Matching::Matched(Matched {
                k_c: 530,
                n_up: 4,
                k_v_up: 540,
                n_down: Some(3),
                k_v_down: Some(400)
            })
        );

        // Samples beyond N_up on file do not change anything.
        let m = match_calls(250, &[100; 9], 100).expect("match");

        assert!(matches!(
            m,
            Matching::Matched(Matched {
                n_up: 3,
                n_down: Some(2),
                ..
            })
        ));
    }

    #[test]
    fn too_few_samples_on_file_ask_for_the_fewest_that_can_reach_k_c() {
        // Nothing on file: K_c / floor rounded up.
        assert_eq!(match_calls(250, &[], 100), Ok(Matching::Need(3)));
        assert_eq!(match_calls(300, &[], 100), Ok(Matching::Need(3)));
        // Two samples hold 260 calls; 40 more need one more sample of at least 100.
        assert_eq!(match_calls(300, &[130, 130], 100), Ok(Matching::Need(3)));
        assert_eq!(match_calls(1000, &[100], 100), Ok(Matching::Need(10)));

        // Impossible inputs are errors, not guesses.
        assert!(match_calls(0, &[100], 100).is_err(), "no champion calls");
        assert!(match_calls(100, &[100], 0).is_err(), "no counted puzzle");
        assert!(
            match_calls(500, &[100, 99], 100).is_err(),
            "a sample below its floor"
        );
    }

    #[test]
    fn overlap_keeps_the_select_order() {
        let ids = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();

        assert_eq!(
            overlap(&ids(&["a", "b", "c"]), &ids(&["d", "e"])),
            Vec::<String>::new()
        );
        assert_eq!(
            overlap(&ids(&["a", "b", "c"]), &ids(&["c", "x", "a"])),
            ids(&["c", "a"])
        );
    }

    #[test]
    fn settings_may_differ_in_the_seed_only() {
        let frozen = BreedConfig {
            generations: 30,
            ..BreedConfig::default()
        };
        let mut cfg = BreedConfig {
            seed: 17,
            ..frozen.clone()
        };

        assert_eq!(check_settings(&cfg, &frozen), Ok(()));

        cfg.max_calls += 1;
        cfg.selection = ParentSelection::Lexicase;

        let e = check_settings(&cfg, &frozen).expect_err("differs");

        assert!(e.contains("max_calls") && e.contains("selection"), "{e}");
        assert!(!e.contains("seed"), "{e}");

        let mut cfg = BreedConfig {
            seed: 17,
            ..frozen.clone()
        };
        cfg.limits.models[0].cost_per_call = 0.5;

        assert!(
            check_settings(&cfg, &frozen)
                .expect_err("differs")
                .contains("limits")
        );
    }

    #[test]
    fn call_seeds_cover_every_call_a_cascade_can_make() {
        let (_, seed) = seed_cascades().remove(0);
        let worker = seed.stages[0].workers[0].clone();
        let wide = Cascade {
            stages: vec![
                Stage {
                    workers: vec![worker.clone(); 3],
                },
                Stage {
                    workers: vec![worker.clone()],
                },
                Stage {
                    workers: vec![worker.clone(); 2],
                },
            ],
        };

        for c in [&seed, &wide] {
            let seeds = call_seeds(c, 42);
            let distinct: std::collections::BTreeSet<u64> = seeds.iter().copied().collect();

            assert_eq!(seeds.len() as u64, cascade_call_bound(c));
            assert_eq!(distinct.len(), seeds.len(), "every call has its own seed");
        }

        // The Worker seeds and the judge seeds of a stage vote, as run_cascade draws them.
        let s = call_seeds(&wide, 42);
        let vote0 = derive_seed(42, "vote0");

        assert_eq!(s[0], derive_seed(42, "stage0/worker0"));
        assert!(s.contains(&derive_seed(vote0, "1/2/0")));
        assert!(s.contains(&derive_seed(derive_seed(42, "vote2"), "0/1/1")));
        assert!(
            !s.contains(&derive_seed(derive_seed(42, "vote1"), "0/1/0")),
            "a one-Worker stage never votes"
        );
        assert_ne!(call_seeds(&wide, 43), s);
    }

    /// Runs on disk as `brut-breed-ollama` and `run-seed.sh` leave them, with a stand-in
    /// for the model of D1, built once for all tests.
    pub(crate) mod fixture {
        use super::super::*;

        use std::sync::OnceLock;

        use brutmaschine::backend::StopReason;
        use brutmaschine::cascade::arena::breed_to_end_in;
        use brutmaschine::cascade::domain::GenePool;
        use brutmaschine::cascade::genome::{CascadeLimits, ModelSpec};
        use brutmaschine::cascade::puzzle::{parse_prompt, render_answer, render_grid};

        use crate::Recorded;
        use crate::vote::{Filed, Shared};

        /// The model tag of D1.
        pub const TAG: &str = "vllm:qwen2.5-32b";

        /// The seeds the fixture holds complete; seed 5 has run B-5 only.
        pub const COMPLETE: [u64; 4] = [1, 2, 3, 4];

        /// A stand-in for the model of D1. It knows the answer of every puzzle of the
        /// seed's two panels, and answers right when a hash of the request falls below a
        /// skill that depends on the system prompt; otherwise a near miss that differs
        /// from call to call, or nothing usable. A judge picks the right candidate four
        /// times in five. So Cascades differ in skill, stage votes pay, and the vote, the
        /// champion and the seeds win and lose.
        #[derive(Clone)]
        pub struct Fake {
            truth: Arc<HashMap<String, Grid>>,
        }

        impl Fake {
            pub fn for_seed(cfg: &BreedConfig) -> Fake {
                let search =
                    GridDomain.panel(derive_seed(cfg.seed, "panel/search"), cfg.search_puzzles);
                let select = vote::select_panel(cfg).expect("panel");
                let truth = search
                    .iter()
                    .chain(&select)
                    .map(|p| (render_grid(&p.test_input), p.test_output.clone()))
                    .collect();

                Fake {
                    truth: Arc::new(truth),
                }
            }
        }

        impl crate::Detailed for Fake {}

        impl ModelBackend for Fake {
            fn name(&self) -> &str {
                TAG
            }

            fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
                let h = derive_seed(req.seed, &req.system);
                let skill = 25 + derive_seed(7, &req.system) % 40;
                let text = match parse_prompt(&req.prompt) {
                    Some(p) if p.judge => {
                        let truth = self.truth.get(&render_grid(&p.test_input));
                        let a_right = p.candidates.as_ref().map(|(a, _)| Some(a) == truth);
                        // An informed judge, right four times in five: stage votes pay.
                        let a = match a_right {
                            Some(right) if !h.is_multiple_of(5) => right,
                            _ => h.is_multiple_of(2),
                        };

                        if a { "VERDICT A" } else { "VERDICT B" }.to_string()
                    }
                    Some(p) => match self.truth.get(&render_grid(&p.test_input)) {
                        Some(t) if h % 100 < skill => render_answer(t),
                        Some(t) if h % 100 < 90 => {
                            let mut near = t.clone();
                            let r = (h / 100) as usize % near.len();
                            let c = (h / 1000) as usize % near[r].len();

                            near[r][c] = (near[r][c] + 1 + (h / 10_000 % 8) as u8) % 10;
                            render_answer(&near)
                        }
                        _ => "no idea".to_string(),
                    },
                    None => "no idea".to_string(),
                };

                Ok(GenResponse {
                    text,
                    prompt_tokens: 50,
                    completion_tokens: 9,
                    stop_reason: StopReason::Stop,
                    backend: TAG.to_string(),
                })
            }
        }

        /// The Mutator of the fixture. Every child it writes has two Workers in one stage,
        /// a mirror and a colour strategy, so the fixture's Cascades vote inside a stage
        /// as those of real runs can. The text depends on the request's seed, so no two
        /// children are equal.
        pub struct Widen;

        impl brutmaschine::cascade::arena::Mutator for Widen {
            fn answer(&self, req: &GenRequest) -> Option<GenResponse> {
                let (_, seed) = brutmaschine::cascade::genome::seed_cascades().remove(0);
                let mut first = seed.stages[0].workers[0].clone();

                first.model = TAG.to_string();
                first.sections[1].text = format!(
                    "Look for mirror images and colour changes, variant {}.",
                    req.seed % 997
                );

                let mut second = first.clone();

                second.traits.temperature = 0.6;
                second.sections[1].text =
                    format!("Look for colour changes first, variant {}.", req.seed % 991);

                let child = Cascade {
                    stages: vec![Stage {
                        workers: vec![first, second],
                    }],
                };

                Some(GenResponse {
                    text: serde_json::to_string(&child).ok()?,
                    prompt_tokens: 100,
                    completion_tokens: 100,
                    stop_reason: StopReason::Stop,
                    backend: "answer-book".to_string(),
                })
            }
        }

        /// The tiny settings of the fixture's arms, with Mutator rounds as D2 has them.
        pub fn config(seed: u64, selection: ParentSelection) -> BreedConfig {
            BreedConfig {
                seed,
                generations: 4,
                children_per_generation: 4,
                llm_every: 2,
                llm_children: 2,
                search_puzzles: 6,
                select_puzzles: 9,
                finalists: 2,
                max_calls: 20_000,
                selection,
                limits: CascadeLimits {
                    models: vec![ModelSpec {
                        tag: TAG.to_string(),
                        cost_per_call: 0.002,
                    }],
                    ..CascadeLimits::default()
                },
                ..BreedConfig::default()
            }
        }

        #[derive(Serialize)]
        struct Checkpoint<'a> {
            snapshot: &'a brutmaschine::cascade::arena::Snapshot<Grid>,
            log_len: u64,
            log_head: String,
        }

        /// Breed `cfg` into `dir` and write it as brut-breed-ollama writes a finished run:
        /// the settings, the hash-chained log, the checkpoint at its end, the report, the
        /// timings, and the record of every Worker call.
        pub fn breed(dir: &Path, cfg: &BreedConfig) {
            std::fs::create_dir_all(dir).expect("run dir");

            let record =
                Recorded::open(Fake::for_seed(cfg), &dir.join(vote::RUN_RECORD)).expect("record");
            let mut pool = ModelPool::new();

            pool.insert(TAG, Box::new(record));

            let (report, events, snapshot) =
                breed_to_end_in::<GridDomain>(cfg.clone(), GenePool::grid(), &pool, Some(&Widen))
                    .expect("breed");
            let mut head = GENESIS.to_string();
            let mut log = String::new();

            for (seq, event) in events.iter().enumerate() {
                let event_json = serde_json::to_string(event).expect("event");
                let hash = sha256_hex(format!("{head}\n{event_json}").as_bytes());
                let line = LogLine {
                    seq: seq as u64,
                    prev: head.clone(),
                    hash: hash.clone(),
                    event: event.clone(),
                };

                log.push_str(&serde_json::to_string(&line).expect("line"));
                log.push('\n');
                head = hash;
            }

            let checkpoint = Checkpoint {
                snapshot: &snapshot,
                log_len: events.len() as u64,
                log_head: head,
            };
            let timings: String = (0..=cfg.generations)
                .map(|g| format!("{{\"generation\":{g},\"seconds\":0.5}}\n"))
                .collect();

            write(
                &dir.join(vote::CONFIG),
                &serde_json::to_string_pretty(cfg).expect("cfg"),
            );
            write(&dir.join(vote::EVENTS), &log);
            write(
                &dir.join(vote::SNAPSHOT),
                &serde_json::to_string(&checkpoint).expect("checkpoint"),
            );
            write(&dir.join(TIMINGS), &timings);
            write(
                &dir.join(REPORT),
                &serde_json::to_string_pretty(&report).expect("report"),
            );
        }

        pub fn write(path: &Path, text: &str) {
            std::fs::write(path, text).expect("write");
        }

        /// `vote-control --run <dir> --temperature 0.7 --samples n`, in this process and with
        /// the stand-in model: the result of record goes to `vote-control.json`.
        pub fn control(dir: &Path, n: usize) {
            let run = vote::load_run(dir).expect("load");
            let panel = vote::select_panel(&run.cfg).expect("panel");
            let opts = Options {
                samples: Some(n),
                temperature: Some(TEMPERATURE),
                ..Options::default()
            };
            let (plan, voted) = vote::plan(&run, &opts).expect("plan");
            let filed: Arc<dyn Filed> = Arc::new(
                Recorded::open(Fake::for_seed(&run.cfg), &dir.join(vote::RECORD)).expect("record"),
            );
            let mut pool = ModelPool::new();

            pool.insert(TAG, Box::new(Shared(filed.clone())));

            let before = filed.filed();
            let mut report =
                vote::vote(&pool, &run, plan, &voted, &panel, 2, &|_, _| {}).expect("vote");

            report.settle(std::slice::from_ref(&filed), before);
            assert!(report.complete);
            write(
                &dir.join(vote::OUTPUT),
                &format!("{}\n", serde_json::to_string_pretty(&report).expect("json")),
            );
        }

        /// The vote of run B-s as run-seed.sh runs it: sample until the call matching is
        /// decided, then keep the results with N_down and with N_up.
        pub fn vote_as_the_driver_does(dir: &Path, frozen: &Frozen) -> Matched {
            let mut asked = 0;
            let matched = loop {
                match match_for_driver(dir, frozen).expect("match") {
                    Matching::Matched(m) => break m,
                    Matching::Need(n) => {
                        assert!(n > asked, "the matching asks for more samples each time");
                        asked = n;
                        control(dir, n);
                    }
                }
            };

            for (n, file) in [(matched.n_down, VOTE_DOWN), (Some(matched.n_up), VOTE_UP)] {
                if let Some(n) = n {
                    control(dir, n);
                    std::fs::copy(dir.join(vote::OUTPUT), dir.join(file)).expect("keep");
                }
            }

            assert_eq!(
                match_for_driver(dir, frozen).expect("again"),
                Matching::Matched(matched)
            );
            matched
        }

        /// The fixture: frozen settings, a runs directory with the pairs of COMPLETE and
        /// run B-5 alone, and one with the five pilot pairs.
        pub struct Fixture {
            pub root: PathBuf,
            pub frozen_dir: PathBuf,
            pub runs: PathBuf,
            pub pilot: PathBuf,
            pub frozen: Frozen,
        }

        /// Both arms and the vote of `seed` in `runs`, as run-seed.sh makes them.
        pub fn pair(runs: &Path, seed: u64, frozen: &Frozen) {
            let b = Arm::B.dir(runs, seed);

            breed(&b, &config(seed, ParentSelection::Dgm));
            vote_as_the_driver_does(&b, frozen);
            breed(
                &Arm::L.dir(runs, seed),
                &config(seed, ParentSelection::Lexicase),
            );
            assert!(write_panels(&b, frozen).expect("panels"));
        }

        /// A fresh directory for one test, under this process's fixture directory.
        pub fn scratch(name: &str) -> PathBuf {
            let dir = get().root.join("scratch").join(name);
            let _ = std::fs::remove_dir_all(&dir);

            std::fs::create_dir_all(&dir).expect("scratch");
            dir
        }

        pub fn copy_dir(from: &Path, to: &Path) {
            std::fs::create_dir_all(to).expect("mkdir");

            for entry in std::fs::read_dir(from).expect("read dir") {
                let entry = entry.expect("entry");
                let target = to.join(entry.file_name());

                if entry.path().is_dir() {
                    copy_dir(&entry.path(), &target);
                } else {
                    std::fs::copy(entry.path(), &target).expect("copy");
                }
            }
        }

        /// A runs directory named `name` with copies of the fixture's runs `dirs`
        /// (for example "B-1").
        pub fn runs_with(name: &str, dirs: &[&str]) -> PathBuf {
            let runs = scratch(name).join("runs");

            for d in dirs {
                copy_dir(&get().runs.join(d), &runs.join(d));
            }

            runs
        }

        /// The fixture, built once per test process.
        pub fn get() -> &'static Fixture {
            static FIXTURE: OnceLock<Fixture> = OnceLock::new();

            FIXTURE.get_or_init(|| {
                let base = std::env::temp_dir().join("breeding-vs-vote-tests");

                // Fixtures of test processes that have ended are removed.
                if let Ok(entries) = std::fs::read_dir(&base) {
                    for e in entries.flatten() {
                        let pid = e.file_name().to_string_lossy().to_string();

                        if Path::new("/proc").exists() && !Path::new("/proc").join(&pid).exists() {
                            let _ = std::fs::remove_dir_all(e.path());
                        }
                    }
                }

                let root = base.join(std::process::id().to_string());
                let _ = std::fs::remove_dir_all(&root);
                let frozen_dir = root.join("frozen");
                let runs = root.join("runs");

                std::fs::create_dir_all(&frozen_dir).expect("frozen dir");

                for (file, selection) in [
                    (FROZEN_B, ParentSelection::Dgm),
                    (FROZEN_L, ParentSelection::Lexicase),
                ] {
                    let text = serde_json::to_string_pretty(&config(1, selection)).expect("cfg");

                    write(&frozen_dir.join(file), &format!("{text}\n"));
                }

                let frozen = Frozen::read(&frozen_dir).expect("frozen");

                for seed in COMPLETE {
                    pair(&runs, seed, &frozen);
                }

                breed(&Arm::B.dir(&runs, 5), &config(5, ParentSelection::Dgm));

                let pilot = root.join("pilot");

                for seed in PILOT {
                    pair(&pilot, seed, &frozen);
                }

                Fixture {
                    root,
                    frozen_dir,
                    runs,
                    pilot,
                    frozen,
                }
            })
        }
    }

    use fixture::{COMPLETE, copy_dir, get, runs_with, write};

    fn load(runs: &Path, arm: Arm, seed: u64) -> Run {
        load_finished(&arm.dir(runs, seed), arm, seed, &get().frozen).expect("load")
    }

    fn message(r: Result<Analysis>) -> String {
        match r {
            Ok(_) => panic!("the analysis should refuse"),
            Err(e) => e.to_string(),
        }
    }

    fn analysed(runs: &Path) -> Analysis {
        analyse(runs, &get().frozen, List::Counted).expect("analysis")
    }

    fn edit_json(path: &Path, f: impl FnOnce(&mut serde_json::Value)) {
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("read")).expect("json");

        f(&mut v);
        write(path, &serde_json::to_string_pretty(&v).expect("json"));
    }

    #[test]
    fn a_finished_fixture_run_passes_every_check() {
        let f = get();

        for seed in COMPLETE {
            for arm in [Arm::B, Arm::L] {
                let run = load(&f.runs, arm, seed);

                assert_eq!(run.seed(), seed);
                assert_eq!((run.search.len(), run.select.len()), (6, 9));
                assert!(run.run.snapshot.finished);
                assert_eq!(run.generation_seconds, 2.5);
            }
        }

        assert!(matches!(
            arm_state(&f.runs, Arm::L, 5, &f.frozen),
            Ok(ArmState::Missing)
        ));
        assert!(matches!(
            arm_state(&f.runs, Arm::B, 5, &f.frozen),
            Ok(ArmState::Finished(_))
        ));
        assert_eq!(parse_run_name("B-12"), Some((Arm::B, 12)));
        assert_eq!(parse_run_name("L-3"), Some((Arm::L, 3)));

        for bad in ["B-012", "B-", "V-3", "B-3x", "B3", "b-3"] {
            assert_eq!(parse_run_name(bad), None, "{bad}");
        }
    }

    #[test]
    fn the_champion_calls_come_from_the_run_record() {
        let f = get();

        for seed in COMPLETE {
            let b = load(&f.runs, Arm::B, seed);
            let m = match_run(&b).expect("match");
            let champion = &b.archive()[m.champion];

            assert_eq!(
                Some(m.champion),
                vote::champion(b.archive(), b.cfg().finalists)
            );
            assert_ne!(champion.op, Op::Seed);
            assert!(m.overlap.is_empty() && m.counted.iter().all(|&c| c));

            // Recounted by hand: the records that carry a seed of the champion's select runs.
            let text = std::fs::read_to_string(b.dir.join(vote::RUN_RECORD)).expect("record");
            let seeds: HashSet<u64> = b
                .select
                .iter()
                .flat_map(|p| {
                    call_seeds(
                        &champion.cascade,
                        derive_seed(seed, &format!("select/{}/{}", champion.id, p.id)),
                    )
                })
                .collect();
            let by_hand = text
                .lines()
                .map(|l| serde_json::from_str::<CallRecord>(l).expect("record"))
                .filter(|r| seeds.contains(&r.request.seed))
                .count() as u64;

            assert_eq!(m.k_c, by_hand, "seed {seed}");
            assert!(m.k_c >= champion.cascade.worker_count() as u64 * b.select.len() as u64);

            // The replayed answers are the snapshot's.
            for (p, r) in b.select.iter().zip(&m.champion_runs) {
                assert_eq!(b.select_answer(&champion.id, p), Some(&r.answer));
            }
        }
    }

    #[test]
    fn the_kept_votes_hold_n_up_and_n_down_samples() {
        let f = get();

        for seed in COMPLETE {
            let b = load(&f.runs, Arm::B, seed);
            let m = match_run(&b).expect("match");
            let Matching::Matched(x) = m.matching else {
                panic!("seed {seed}: the fixture's vote is decided");
            };
            let k_v = |n: usize| -> u64 {
                m.samples[..n]
                    .iter()
                    .map(|s| counted_calls(s, &m.counted))
                    .sum()
            };

            // N_up is the smallest N with K_v >= K_c, N_down the largest with K_v <= K_c.
            assert!(k_v(x.n_up) >= x.k_c && x.k_v_up == k_v(x.n_up));
            assert!(x.n_up == 1 || k_v(x.n_up - 1) < x.k_c);

            match x.n_down {
                Some(n) => {
                    assert!(k_v(n) <= x.k_c);
                    assert!(n == x.n_up || k_v(n + 1) > x.k_c);
                }
                None => assert!(k_v(1) > x.k_c),
            }

            let up = read_kept(&b.dir.join(VOTE_UP), x.n_up, &b, &m).expect("up");

            assert_eq!(up.plan.samples_per_puzzle, x.n_up);
            assert_eq!(up.plan.cascade.temperature, Some(TEMPERATURE));

            if let Some(n) = x.n_down {
                let down = read_kept(&b.dir.join(VOTE_DOWN), n, &b, &m).expect("down");

                assert_eq!(down.plan.samples_per_puzzle, n);
            }

            // The other N is refused for each file.
            assert!(read_kept(&b.dir.join(VOTE_UP), x.n_up + 1, &b, &m).is_err());
        }
    }

    #[test]
    fn the_analysis_counts_the_complete_pairs_in_list_order() {
        let f = get();
        let a = analysed(&f.runs);

        assert_eq!(
            a.pairs.iter().map(|r| r.data.seed).collect::<Vec<_>>(),
            COMPLETE.to_vec()
        );
        assert_eq!(a.counted_pairs, 4);
        assert_eq!(
            a.waiting,
            vec![Waiting {
                seed: 5,
                why: "arm B finished, arm L not started".to_string()
            }]
        );
        assert_eq!(
            (a.stop, a.primary, a.next_seed),
            (None, Primary::Running, Some(6))
        );
        assert_eq!(
            (a.s1_claim, a.s2_claim),
            (None, None),
            "read at the stop only"
        );

        // E_B is the product of the factors of the outcomes, in list order.
        let mut e = Evidence::default();

        for r in &a.pairs {
            let d = &r.data;
            let y = Outcome::of(d.c, d.v);

            assert_eq!(d.y, y.name(Y_NAMES));
            assert_eq!(d.y1, Outcome::of(d.c, d.seed_plain).name(Y1_NAMES));
            assert_eq!(d.y2, Outcome::of(d.c_l, d.c).name(Y2_NAMES));
            assert_eq!(d.counted_puzzles, 9);
            e = e.step(y);
            assert_eq!(
                r.e_b.as_ref().map(|s| s.fraction.clone()),
                Some(e.fraction())
            );
        }

        assert_eq!(a.e_b.fraction, e.fraction());
        assert_eq!(a.e_v.fraction, e.mirrored().fraction());

        // The result reads back as JSON, and the table names every pair.
        let json = serde_json::to_value(&a).expect("json");

        assert_eq!(json["pairs"][0]["seed"], 1);
        assert_eq!(json["primary"], "running");

        let table = render(&a);

        assert!(
            table.contains("Next: run seed 6; seeds [5] are started"),
            "{table}"
        );
        assert!(
            table
                .lines()
                .filter(|l| l.trim_start().starts_with('1'))
                .count()
                >= 3,
            "{table}"
        );
    }

    #[test]
    fn pairs_after_the_stop_are_listed_but_not_counted() {
        let a = analysed(&get().runs);
        let complete: Vec<&Pair> = a.pairs.iter().map(|r| &r.data).collect();
        let outcomes: Vec<[Outcome; 3]> = complete.iter().map(|p| p.outcomes).collect();
        // A cap of 2 stops the rule at pair 2, open or earlier.
        let rule = sequential(&outcomes, 2);
        let rows = rows(&complete, &rule);

        assert_eq!(rule.stop, Some(2));
        assert_eq!(
            rows.iter().map(|r| r.counted).collect::<Vec<_>>(),
            vec![true, true, false, false]
        );
        assert!(
            rows[..2]
                .iter()
                .all(|r| r.e_b.is_some() && r.e_s2.is_some())
        );
        assert!(
            rows[2..]
                .iter()
                .all(|r| r.e_b.is_none() && r.e_v.is_none() && r.e_s1.is_none())
        );

        let mut shown = a.clone();

        shown.pairs = rows;
        assert!(render(&shown).contains("after the stop, not counted"));
    }

    #[test]
    fn a_later_pair_waits_for_an_earlier_unfinished_one() {
        let runs = runs_with("order", &["B-1", "L-1", "B-2", "B-3", "L-3"]);
        let a = analysed(&runs);

        assert_eq!(a.pairs.len(), 1);
        assert_eq!(
            a.waiting,
            vec![
                Waiting {
                    seed: 2,
                    why: "arm B finished, arm L not started".to_string()
                },
                Waiting {
                    seed: 3,
                    why: "complete; it enters after seed 2".to_string()
                },
            ]
        );
        assert_eq!(a.next_seed, Some(4));
    }

    #[test]
    fn an_unfinished_run_is_not_counted() {
        let runs = runs_with("unfinished", &["B-1", "L-1", "B-2", "L-2"]);

        std::fs::remove_file(runs.join("L-2").join(REPORT)).expect("rm");

        let a = analysed(&runs);

        assert_eq!(a.pairs.len(), 1);
        assert_eq!(
            a.waiting[0],
            Waiting {
                seed: 2,
                why: "arm B finished, arm L not finished".to_string()
            }
        );
    }

    #[test]
    fn a_run_with_other_settings_or_another_seed_is_refused() {
        let runs = runs_with("settings", &["B-1", "L-1"]);

        edit_json(&runs.join("L-1").join(vote::CONFIG), |v| {
            v["judge_weight"] = serde_json::json!(0.3)
        });
        assert!(message(analyse(&runs, &get().frozen, List::Counted)).contains("judge_weight"));

        // Arm B's run under arm L's settings.
        let runs = runs_with("settings-arm", &["B-1", "L-1"]);

        std::fs::rename(runs.join("B-1"), runs.join("B-x")).expect("mv");
        std::fs::rename(runs.join("L-1"), runs.join("B-1")).expect("mv");
        std::fs::rename(runs.join("B-x"), runs.join("L-1")).expect("mv");
        assert!(message(analyse(&runs, &get().frozen, List::Counted)).contains("selection"));

        // A run whose name names another seed.
        let runs = runs_with("settings-seed", &["B-1", "L-1"]);

        copy_dir(&runs.join("B-1"), &runs.join("B-2"));
        assert!(message(analyse(&runs, &get().frozen, List::Counted)).contains("its name says 2"));
    }

    #[test]
    fn a_run_with_failed_calls_is_refused_finished_or_not() {
        let runs = runs_with("failed", &["B-1", "L-1"]);

        write(
            &runs.join("L-1").join(vote::RUN_ERRORS),
            "{\"key\":\"a\"}\n",
        );
        assert!(
            message(analyse(&runs, &get().frozen, List::Counted))
                .contains("1 Worker call(s) failed")
        );

        std::fs::remove_file(runs.join("L-1").join(REPORT)).expect("rm");
        assert!(message(analyse(&runs, &get().frozen, List::Counted)).contains("failed"));
    }

    #[test]
    fn a_run_that_did_not_breed_every_generation_is_refused() {
        let runs = runs_with("short", &["B-1", "L-1"]);

        edit_json(&runs.join("B-1").join(REPORT), |v| {
            v["stop_reason"] = serde_json::json!("call budget reached")
        });
        assert!(
            message(analyse(&runs, &get().frozen, List::Counted))
                .contains("a counted run breeds them all")
        );
    }

    #[test]
    fn seeds_off_the_list_gaps_and_more_than_60_pairs_are_refused() {
        for (dirs, extra, why) in [
            (vec!["B-1", "L-1"], "B-0", "not on the list"),
            (vec!["B-1", "L-1"], "B-901", "pilot seeds never count"),
            (vec!["B-1", "L-1"], "L-61", "at most 60 pairs"),
            (vec!["B-1", "L-1"], "B-3", "seed 2 has no run"),
            (vec!["B-2", "L-2"], "B-3", "seed 1 has no run"),
            (vec!["B-1", "L-1"], "B-01", "plain decimal"),
        ] {
            let runs = runs_with(&format!("list-{extra}-{}", dirs[0]), &dirs);

            std::fs::create_dir_all(runs.join(extra)).expect("mkdir");
            assert!(
                message(analyse(&runs, &get().frozen, List::Counted)).contains(why),
                "{extra}: {why}"
            );
        }

        // Other names are not runs.
        let runs = runs_with("list-other", &["B-1", "L-1"]);

        std::fs::create_dir_all(runs.join("notes")).expect("mkdir");
        write(&runs.join("analysis.json"), "{}");
        assert_eq!(analysed(&runs).pairs.len(), 1);

        // The pilot list takes pilot seeds only.
        assert!(message(analyse(&runs, &get().frozen, List::Pilot)).contains("not pilot seeds"));
    }

    #[test]
    fn generation_0_must_be_the_same_in_both_arms() {
        let runs = runs_with("gen0", &["B-1", "L-1"]);
        let path = runs.join("L-1").join(vote::SNAPSHOT);

        edit_json(&path, |v| {
            let archive = v["snapshot"]["archive"].as_array_mut().expect("archive");
            let seed = archive
                .iter_mut()
                .find(|m| m["op"] == "seed")
                .expect("a seed");
            let answers = seed["answers"].as_array_mut().expect("answers");

            answers[0] = match &answers[0] {
                serde_json::Value::Null => serde_json::json!([[1, 2], [3, 4]]),
                _ => serde_json::Value::Null,
            };
        });
        // The tampered checkpoint is written pretty; the log's anchor still holds.
        assert!(
            message(analyse(&runs, &get().frozen, List::Counted))
                .contains("differ between arms B and L")
        );
    }

    #[test]
    fn a_kept_vote_must_be_the_vote_of_record_with_its_n() {
        let f = get();
        let one = runs_with("kept-n", &["B-1", "L-1"]);
        let b1 = one.join("B-1");
        let Matching::Matched(x) = match_run(&load(&one, Arm::B, 1)).expect("match").matching
        else {
            panic!("decided")
        };

        // The files of N_up and N_down swapped, or the result of N_up edited to another N.
        match x.n_down {
            Some(d) if d != x.n_up => {
                std::fs::rename(b1.join(VOTE_UP), b1.join("x")).expect("mv");
                std::fs::rename(b1.join(VOTE_DOWN), b1.join(VOTE_UP)).expect("mv");
                std::fs::rename(b1.join("x"), b1.join(VOTE_DOWN)).expect("mv");
            }
            _ => edit_json(&b1.join(VOTE_UP), |v| {
                v["samples_per_puzzle"] = serde_json::json!(x.n_up + 1)
            }),
        }

        assert!(
            message(analyse(&one, &f.frozen, List::Counted)).contains("the call matching gives")
        );

        // A vote at another temperature.
        let one = runs_with("kept-temperature", &["B-1", "L-1"]);

        edit_json(&one.join("B-1").join(VOTE_UP), |v| {
            v["cascade"]["temperature"] = serde_json::json!(0.2)
        });
        assert!(message(analyse(&one, &f.frozen, List::Counted)).contains("not the voted seed"));

        // One puzzle's result changed.
        let one = runs_with("kept-flip", &["B-1", "L-1"]);

        edit_json(&one.join("B-1").join(VOTE_UP), |v| {
            let c = v["puzzles"][0]["correct"].as_bool().expect("bool");

            v["puzzles"][0]["correct"] = serde_json::json!(!c);
        });
        assert!(
            message(analyse(&one, &f.frozen, List::Counted))
                .contains("differ from the vote the record gives")
        );
    }

    #[test]
    fn a_vote_record_with_too_few_samples_waits_or_is_refused() {
        let one = runs_with("few", &["B-1", "L-1"]);
        let b = one.join("B-1");

        // No sample on file: the call matching needs at least one.
        write(&b.join(vote::RECORD), "");

        // With kept results the files contradict the record.
        assert!(
            message(analyse(&one, &get().frozen, List::Counted))
                .contains("kept vote results exist")
        );

        // Without them the pair waits for the vote.
        for f in [VOTE_UP, VOTE_DOWN] {
            let _ = std::fs::remove_file(b.join(f));
        }

        let a = analysed(&one);

        assert!(a.pairs.is_empty());
        assert!(
            a.waiting[0]
                .why
                .starts_with("the vote record holds 0 of at least"),
            "{:?}",
            a.waiting
        );
        assert!(matches!(match_for_driver(&b, &get().frozen), Ok(Matching::Need(n)) if n >= 1));
    }

    #[test]
    fn a_vote_still_being_written_has_fewer_samples_on_file() {
        let one = runs_with("in-progress", &["B-1", "L-1"]);
        let b = one.join("B-1");
        let record = b.join(vote::RECORD);
        let text = std::fs::read_to_string(&record).expect("record");
        let full = match_run(&load(&one, Arm::B, 1))
            .expect("match")
            .samples
            .len();

        // The vote's last call is being filed: half a line, no newline.
        write(&record, &text[..text.len() - 40]);

        let m = match_run(&load(&one, Arm::B, 1)).expect("a write in progress is no error");

        // Calls are filed in the order they end, so the cut call may be of any sample.
        assert!(m.samples.len() < full);

        // The run's own record must be complete.
        let own = b.join(vote::RUN_RECORD);
        let text = std::fs::read_to_string(&own).expect("record");

        write(&own, &text[..text.len() - 1]);
        assert!(match_run(&load(&one, Arm::B, 1)).is_err());
    }

    #[test]
    fn a_record_without_a_champion_call_is_refused() {
        let one = runs_with("no-call", &["B-1", "L-1"]);
        let run = load(&one, Arm::B, 1);
        let m = match_run(&run).expect("match");
        let champion = &run.archive()[m.champion];
        let p = &run.select[0];
        let gone = call_seeds(
            &champion.cascade,
            derive_seed(1, &format!("select/{}/{}", champion.id, p.id)),
        )[0];
        let record = one.join("B-1").join(vote::RUN_RECORD);
        let text = std::fs::read_to_string(&record).expect("record");
        let kept: String = text
            .lines()
            .filter(|l| {
                serde_json::from_str::<CallRecord>(l)
                    .expect("rec")
                    .request
                    .seed
                    != gone
            })
            .map(|l| format!("{l}\n"))
            .collect();

        assert!(kept.len() < text.len());
        write(&record, &kept);
        assert!(
            message(analyse(&one, &get().frozen, List::Counted))
                .contains("lacks a select-phase call")
        );
    }

    #[test]
    fn panels_json_must_hold_the_regenerated_panels() {
        let one = runs_with("panels", &["B-1", "L-1"]);
        let path = one.join("B-1").join(PANELS);

        edit_json(&path, |v| {
            v["select"][0]["test_output"] = serde_json::json!([[0]])
        });
        assert!(
            message(analyse(&one, &get().frozen, List::Counted))
                .contains("differs from the regenerated panels")
        );

        // Without the file the pair waits; write_panels writes it once and then checks it.
        std::fs::remove_file(&path).expect("rm");
        assert_eq!(analysed(&one).waiting[0].why, format!("no {PANELS} yet"));
        assert!(write_panels(&one.join("B-1"), &get().frozen).expect("write"));
        assert!(!write_panels(&one.join("B-1"), &get().frozen).expect("check"));
        assert_eq!(analysed(&one).pairs.len(), 1);
    }

    #[test]
    fn a_broken_log_or_checkpoint_is_refused() {
        let one = runs_with("log", &["B-1", "L-1"]);
        let log = one.join("L-1").join(vote::EVENTS);
        let text = std::fs::read_to_string(&log).expect("log");

        write(
            &log,
            &text.replacen("\"generation\":1", "\"generation\":7", 1),
        );
        assert!(
            message(analyse(&one, &get().frozen, List::Counted)).contains("breaks the hash chain")
        );

        write(&log, &text[..text.len() - 1]);
        assert!(message(analyse(&one, &get().frozen, List::Counted)).contains("incomplete"));

        let cut: String = text
            .lines()
            .take(text.lines().count() - 1)
            .map(|l| format!("{l}\n"))
            .collect();

        write(&log, &cut);
        assert!(
            message(analyse(&one, &get().frozen, List::Counted))
                .contains("checkpoint does not stand")
        );
    }

    #[test]
    fn overlapping_puzzles_are_left_out_of_every_count() {
        let f = get();
        let b = load(&f.runs, Arm::B, 1);
        let l = load(&f.runs, Arm::L, 1);
        let full = match_run(&b).expect("match");
        let Matching::Matched(x) = full.matching else {
            panic!("decided")
        };
        let up = read_kept(&b.dir.join(VOTE_UP), x.n_up, &b, &full).expect("up");
        let down = x
            .n_down
            .map(|n| read_kept(&b.dir.join(VOTE_DOWN), n, &b, &full).expect("down"));
        let all = counts(&b, &l, &full, &up, down.as_ref()).expect("counts");

        // As if the first three select puzzles were also in the search panel.
        let overlap: Vec<String> = b.select[..3].iter().map(|p| p.id.clone()).collect();
        let counted: Vec<bool> = (0..b.select.len()).map(|i| i >= 3).collect();
        let part = match_masked(&b, overlap, counted.clone()).expect("masked");
        let some = counts(&b, &l, &part, &up, down.as_ref()).expect("counts");
        let solved = |run: &Run, id: &str, from: usize| {
            run.select[from..]
                .iter()
                .filter(|p| run.solved(id, p) == Some(true))
                .count()
        };
        let champion = b.archive()[full.champion].id.clone();
        let champion_l = l.archive()[l_champion(&l).expect("champion")].id.clone();

        assert_eq!(
            part.k_c,
            full.champion_runs[3..].iter().map(|r| r.calls).sum::<u64>()
        );
        assert!(part.k_c < full.k_c);
        assert_eq!(some.c, solved(&b, &champion, 3));
        assert_eq!(some.seed_plain, solved(&b, &full.plan.cascade.member, 3));
        assert_eq!(some.c_l, solved(&l, &champion_l, 3));
        assert_eq!(some.v, up.puzzles[3..].iter().filter(|p| p.correct).count());
        assert_eq!(all.v, up.puzzles.iter().filter(|p| p.correct).count());
        assert_eq!(all.c, solved(&b, &champion, 0));

        if let Matching::Matched(y) = part.matching {
            let k_v: u64 = part.samples[..y.n_up]
                .iter()
                .map(|s| s[3..].iter().map(|r| r.calls).sum::<u64>())
                .sum();

            assert_eq!(y.k_v_up, k_v);
        }
    }

    #[test]
    fn the_pilot_gate_takes_the_median_seed_in_integers() {
        let a = analyse(&get().pilot, &get().frozen, List::Pilot).expect("pilot");
        let with = |solved: &[usize], n: usize| -> String {
            let rows: Vec<Row> = solved
                .iter()
                .map(|&s| {
                    let mut r = a.pairs[0].clone();

                    r.data.seed_search_solved = s;
                    r.data.search_puzzles = n;
                    r
                })
                .collect();

            pilot_gate(&rows).expect("gate")
        };

        // 6 of 40 is exactly 15 %: it passes; 5 of 40 does not. The median is the middle.
        assert!(with(&[1, 6, 30, 2, 9], 40).contains("6.0 of 40"));
        assert!(with(&[1, 6, 30, 2, 9], 40).contains("passes"));
        assert!(with(&[1, 5, 30, 2, 9], 40).contains("below 15 %"));
        // Four seeds: the mean of the middle two, 5.5 of 40 = 13.75 %.
        assert!(with(&[1, 5, 6, 9], 40).contains("5.5 of 40"));
        assert!(with(&[1, 5, 6, 9], 40).contains("below"));
        assert_eq!(pilot_gate(&[]), None);
        assert!(
            a.verdict
                .iter()
                .any(|l| l.starts_with("Pilot gate (section 9)"))
        );
    }

    #[test]
    fn wall_time_comes_from_the_drivers_journal() {
        let j = |lines: &[&str]| -> Vec<(u64, String)> {
            lines
                .iter()
                .filter_map(|l| {
                    l.split_once(' ')
                        .map(|(t, w)| (t.parse().expect("t"), w.to_string()))
                })
                .collect()
        };
        let log = j(&[
            "100 breed start --out x --config c --seed 1",
            "160 breed exit 3",
            "161 mutator start",
            "190 mutator exit 0",
            "191 breed start --out x --resume",
            "400 breed exit 0",
            "401 verify exit 0",
            "402 vote start --samples 3",
            "430 vote exit 0",
            "431 vote start --samples 2",
            "432 vote exit 0",
        ]);

        assert_eq!(span(&log, "breed start", "breed exit 0"), Some(300));
        assert_eq!(span(&log, "vote start", "vote exit 0"), Some(30));
        assert_eq!(span(&log, "breed start", "breed exit 9"), None);
        assert_eq!(span(&[], "breed start", "breed exit 0"), None);
        assert!(journal(Path::new("/nonexistent")).is_empty());
    }

    #[test]
    fn the_frozen_files_are_the_decided_settings() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(FROZEN_DIR);
        let frozen = Frozen::read(&dir).expect("frozen");
        let b = BreedConfig {
            generations: 30,
            children_per_generation: 8,
            search_puzzles: 40,
            select_puzzles: 100,
            llm_every: 2,
            llm_children: 8,
            max_calls: 480_000,
            limits: brutmaschine::cascade::genome::CascadeLimits {
                models: vec![brutmaschine::cascade::genome::ModelSpec {
                    tag: "vllm:qwen2.5-32b".to_string(),
                    cost_per_call: 0.002,
                }],
                ..Default::default()
            },
            ..BreedConfig::default()
        };
        let l = BreedConfig {
            selection: ParentSelection::Lexicase,
            ..b.clone()
        };

        assert_eq!(
            frozen,
            Frozen {
                b: b.clone(),
                l: l.clone()
            }
        );

        // Byte for byte as BreedConfig serialises them, so a run's config.json can equal them.
        for (file, cfg) in [(FROZEN_B, &b), (FROZEN_L, &l)] {
            let text = std::fs::read_to_string(dir.join(file)).expect("read");

            assert_eq!(
                text,
                format!("{}\n", serde_json::to_string_pretty(cfg).expect("json")),
                "{file}"
            );
            assert!(cfg.check().is_ok());
        }

        // The worst case of section 4 (D4) stays below the cap.
        assert_eq!(worst_case_calls(&b).expect("bound"), 477_224);
        assert!(worst_case_calls(&b).expect("bound") <= b.max_calls);
    }

    /// `python3 -I recount.py <args> --json <file>`: its result, or its refusal. None when
    /// python3 is missing; the test then says so and passes.
    fn recount(args: &[&str]) -> Option<std::result::Result<serde_json::Value, String>> {
        use std::process::Command;

        if Command::new("python3").arg("--version").output().is_err() {
            eprintln!("skipped: python3 is missing, so the recount cannot run");
            return None;
        }

        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(FROZEN_DIR)
            .join("recount.py");
        let out = get()
            .root
            .join(format!("recount-{}.json", derive_seed(0, &args.join(" "))));
        let _ = std::fs::remove_file(&out);
        let o = Command::new("python3")
            .arg("-I")
            .arg(&script)
            .args(args)
            .arg("--json")
            .arg(&out)
            .output()
            .expect("python3");

        if !o.status.success() {
            return Some(Err(String::from_utf8_lossy(&o.stderr).to_string()));
        }

        let text = std::fs::read_to_string(&out).expect("the recount's json");

        Some(Ok(serde_json::from_str(&text).expect("json")))
    }

    fn path(p: &Path) -> &str {
        p.to_str().expect("utf-8 path")
    }

    /// The parts of `a` that the recount computes too, in its form.
    fn comparable(a: &Analysis) -> serde_json::Value {
        let e = |s: &Option<Shown>| s.as_ref().map(|s| s.fraction.clone());
        let pairs: Vec<serde_json::Value> = a
            .pairs
            .iter()
            .map(|r| {
                let d = &r.data;

                serde_json::json!({
                    "pair": r.pair, "counted": r.counted, "seed": d.seed, "overlap": d.overlap,
                    "counted_puzzles": d.counted_puzzles, "champion": d.champion,
                    "voted_member": d.voted_member, "voted_cascade": d.voted_cascade,
                    "c": d.c, "k_c": d.k_c, "n_up": d.n_up, "k_v_up": d.k_v_up, "v": d.v, "y": d.y,
                    "n_down": d.n_down, "k_v_down": d.k_v_down, "v_down": d.v_down, "y_down": d.y_down,
                    "seed_plain": d.seed_plain, "y1": d.y1, "champion_l": d.champion_l, "c_l": d.c_l,
                    "y2": d.y2, "e_b": e(&r.e_b), "e_v": e(&r.e_v), "e_s1": e(&r.e_s1), "e_s2": e(&r.e_s2),
                })
            })
            .collect();

        serde_json::json!({
            "list": a.list, "cap": a.cap, "pairs": pairs, "waiting": a.waiting,
            "counted_pairs": a.counted_pairs, "stop": a.stop, "primary": a.primary,
            "e_b": a.e_b.fraction, "e_v": a.e_v.fraction, "e_s1": a.e_s1.fraction, "e_s2": a.e_s2.fraction,
            "s1_claim": a.s1_claim, "s2_claim": a.s2_claim, "next_seed": a.next_seed,
        })
    }

    #[test]
    fn the_recount_gives_what_the_analysis_gives() {
        let f = get();
        let Some(py) = recount(&["--runs", path(&f.runs), "--frozen", path(&f.frozen_dir)]) else {
            return;
        };
        let a = analysed(&f.runs);

        assert_eq!(py.expect("recount"), comparable(&a));

        // The fixture meets both cases of N_down and two outcomes of each comparison.
        assert!(a.pairs.iter().any(|r| r.data.n_down == Some(r.data.n_up)));
        assert!(
            a.pairs
                .iter()
                .any(|r| r.data.n_down == Some(r.data.n_up - 1))
        );
        assert!(
            a.pairs.iter().any(|r| r.data.y == Y_NAMES[0])
                && a.pairs.iter().any(|r| r.data.y == Y_NAMES[1])
        );

        // The pilot list: five pairs, so the rule stops; no claim and no next seed.
        let pilot = analyse(&f.pilot, &f.frozen, List::Pilot).expect("pilot");
        let py = recount(&[
            "--runs",
            path(&f.pilot),
            "--frozen",
            path(&f.frozen_dir),
            "--pilot",
        ])
        .expect("python3");

        assert!(pilot.stop.is_some() && pilot.next_seed.is_none() && pilot.s1_claim.is_none());
        assert_ne!(pilot.primary, Primary::Running);
        assert_eq!(py.expect("recount"), comparable(&pilot));

        // Two pairs of the counted list, with no run after them: the next seed is 3.
        let late = runs_with("recount-late", &["B-1", "L-1", "B-2", "L-2"]);
        let a = analysed(&late);
        let py =
            recount(&["--runs", path(&late), "--frozen", path(&f.frozen_dir)]).expect("python3");

        assert_eq!(py.expect("recount"), comparable(&a));
        assert_eq!(a.next_seed, Some(3));
    }

    /// A change to the files of a runs directory.
    type Tamper = Box<dyn Fn(&Path)>;

    #[test]
    fn the_recount_refuses_what_the_analysis_refuses() {
        let f = get();

        if recount(&["--rule", "/nonexistent"]).is_none() {
            return;
        }

        let cases: Vec<(&str, Tamper)> = vec![
            (
                "settings",
                Box::new(|r: &Path| {
                    edit_json(&r.join("L-1").join(vote::CONFIG), |v| {
                        v["judge_weight"] = serde_json::json!(0.3)
                    })
                }),
            ),
            (
                "failed",
                Box::new(|r: &Path| {
                    write(&r.join("B-1").join(vote::RUN_ERRORS), "{\"key\":\"a\"}\n")
                }),
            ),
            (
                "short",
                Box::new(|r: &Path| {
                    edit_json(&r.join("B-1").join(REPORT), |v| {
                        v["stop_reason"] = serde_json::json!("call budget reached")
                    })
                }),
            ),
            (
                "gap",
                Box::new(|r: &Path| std::fs::create_dir_all(r.join("B-3")).expect("mkdir")),
            ),
            (
                "beyond",
                Box::new(|r: &Path| std::fs::create_dir_all(r.join("L-61")).expect("mkdir")),
            ),
            (
                "pilot-seed",
                Box::new(|r: &Path| std::fs::create_dir_all(r.join("B-902")).expect("mkdir")),
            ),
            (
                "other-n",
                Box::new(|r: &Path| {
                    edit_json(&r.join("B-1").join(VOTE_UP), |v| {
                        let n = v["samples_per_puzzle"].as_u64().expect("n");

                        v["samples_per_puzzle"] = serde_json::json!(n + 1);
                    })
                }),
            ),
            (
                "panels",
                Box::new(|r: &Path| {
                    edit_json(&r.join("B-1").join(PANELS), |v| {
                        v["select"][0]["test_output"] = serde_json::json!([[0]])
                    })
                }),
            ),
            (
                "log",
                Box::new(|r: &Path| {
                    let log = r.join("L-1").join(vote::EVENTS);
                    let text = std::fs::read_to_string(&log).expect("log");

                    write(
                        &log,
                        &text.replacen("\"generation\":1", "\"generation\":7", 1),
                    );
                }),
            ),
        ];

        for (name, tamper) in cases {
            let runs = runs_with(&format!("refuse-{name}"), &["B-1", "L-1"]);

            tamper(&runs);
            assert!(
                analyse(&runs, &f.frozen, List::Counted).is_err(),
                "{name}: the analysis refuses"
            );

            let py = recount(&["--runs", path(&runs), "--frozen", path(&f.frozen_dir)])
                .expect("python3");

            assert!(py.is_err(), "{name}: the recount refuses too");
        }
    }

    #[test]
    fn the_recount_runs_the_rule_as_the_analysis_does() {
        let names =
            |o: [Outcome; 3]| [o[0].name(Y_NAMES), o[1].name(Y1_NAMES), o[2].name(Y2_NAMES)];
        let mut cases: Vec<(usize, Vec<[Outcome; 3]>)> = vec![
            (60, vec![[W, T, T]; 15]),
            (60, vec![[L, W, W]; 10]),
            (
                60,
                (0..60)
                    .map(|i| if i % 2 == 0 { [W, T, L] } else { [L, W, T] })
                    .collect(),
            ),
            (60, vec![[T, W, W]; 70]),
            (60, [vec![[T, T, T]; 48], vec![[W, T, T]; 12]].concat()),
            (60, [vec![[L, T, T]], vec![[W, T, T]; 20]].concat()),
            (5, vec![[T, L, W]; 9]),
            (
                120,
                (0..130)
                    .map(|i| {
                        if i % 4 == 0 {
                            [L, W, T]
                        } else if i % 4 == 2 {
                            [W, L, W]
                        } else {
                            [T, T, L]
                        }
                    })
                    .collect(),
            ),
            (60, Vec::new()),
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;

        for i in 0..300 {
            let len = 1 + (derive_seed(i, "len") % 130) as usize;
            let cap = [5, 60, 120][(i % 3) as usize];
            let win = 30 + derive_seed(i, "win") % 50;
            let pairs = (0..len)
                .map(|_| {
                    let mut pick = || {
                        state = derive_seed(state, "next");

                        match state % 100 {
                            x if x < win => W,
                            x if x < win + (100 - win) * 3 / 4 => L,
                            _ => T,
                        }
                    };

                    [pick(), pick(), pick()]
                })
                .collect();

            cases.push((cap, pairs));
        }

        let spec = serde_json::json!({
            "cases": cases.iter().map(|(cap, o)| serde_json::json!({
                "cap": cap,
                "outcomes": o.iter().map(|x| names(*x)).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        });
        let file = get().root.join("rule-cases.json");

        write(&file, &spec.to_string());

        let Some(py) = recount(&["--rule", path(&file)]) else {
            return;
        };
        let py = py.expect("recount");
        let mut stops = BTreeSet::new();

        for (k, (cap, o)) in cases.iter().enumerate() {
            let r = sequential(o, *cap);
            let theirs = &py["results"][k];
            let steps: Vec<serde_json::Value> = r
                .steps
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "e_b": s.e_b.fraction(), "e_v": s.e_v().fraction(),
                        "e_s1": s.e_s1.fraction(), "e_s2": s.e_s2.fraction(),
                    })
                })
                .collect();

            assert_eq!(theirs["steps"], serde_json::json!(steps), "case {k}");
            assert_eq!(theirs["stop"], serde_json::json!(r.stop), "case {k}");
            assert_eq!(
                theirs["primary"],
                serde_json::to_value(r.primary).expect("json"),
                "case {k}"
            );
            stops.insert(format!("{:?}", r.primary));
        }

        // Every ending occurs among the cases.
        assert_eq!(stops.len(), 4, "{stops:?}");
    }
}
