//! The Soul of a Worker, its Character, and the settings of a Run with Souls (step 2,
//! `.scratch/seele-und-kappa/spec.md`).
//!
//! A Soul is a Worker's hidden, inherited phases, one per hypothesis family of the Run's
//! Domain (Q42, Q52). It is stored in the genome as `Traits::phase_state`, and the
//! Coupling strength κ as `Traits::coupling` (genome.rs). The Soul is off by default: a Run
//! without the `soul` setting has no Soul, and everything it writes stays as before step 2.
//!
//! The Soul is never seen. It shows itself through its Character on each task (Q51, Q57):
//! the squared magnitude of the sum of its phasors, each weighted by the task's Family
//! strength. `character` computes it from the Soul each time; nothing stores a Character
//! (Q55). The phasor of a phase comes from `phasor` alone, so the whole crate uses one table
//! of values (spec 15).
//!
//! Pure computation: no files, no network, no clock.
#![forbid(unsafe_code)]

use std::ops::{Add, AddAssign};

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::cascade::domain::Domain;
use crate::cascade::genome::COUPLING_MAX;
use crate::util::{BrutError, Result};

/// A quarter turn in phase units. A phase p stands for the angle 2πp / 65536.
pub const QUARTER_TURN: u16 = 16384;

/// Half a turn in phase units: the largest phase step a birth may draw (spec 9.3).
pub const HALF_TURN: u16 = 32768;

/// A complex number in f64: a phasor, or a sum of weighted phasors.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Phasor {
    pub re: f64,
    pub im: f64,
}

impl Phasor {
    pub const fn new(re: f64, im: f64) -> Phasor {
        Phasor { re, im }
    }

    /// The squared magnitude |z|^2.
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// The magnitude |z|. It uses the square root, which IEEE 754 rounds exactly, so it is
    /// the same on every platform.
    pub fn norm(self) -> f64 {
        self.norm_sqr().sqrt()
    }

    /// The phasor times the real number `k`.
    pub fn scale(self, k: f64) -> Phasor {
        Phasor::new(self.re * k, self.im * k)
    }
}

impl Add for Phasor {
    type Output = Phasor;

    fn add(self, o: Phasor) -> Phasor {
        Phasor::new(self.re + o.re, self.im + o.im)
    }
}

impl AddAssign for Phasor {
    fn add_assign(&mut self, o: Phasor) {
        *self = *self + o;
    }
}

/// The unit phasor e^(2πip/65536) of the phase `p`, from `cos` and `sin` of the standard
/// library. The angle is reduced to its quarter turn first, so a whole number of quarter
/// turns gives 1, i, -1 or -i exactly, and a quarter turn more multiplies by i exactly.
pub fn phasor(p: u16) -> Phasor {
    let rest = std::f64::consts::TAU * f64::from(p % QUARTER_TURN) / 65536.0;
    let (c, s) = (rest.cos(), rest.sin());

    match p / QUARTER_TURN {
        0 => Phasor::new(c, s),
        1 => Phasor::new(-s, c),
        2 => Phasor::new(-c, -s),
        _ => Phasor::new(s, -c),
    }
}

/// The readout u(p) of spec 3: the unit phasor of `p`, or with the real readout +1 when `p`
/// lies in 0..16384 or 49152..65536 and -1 otherwise, the nearer of 0 and half a turn [Q70
/// a]. Both have length 1, so a Soul carries no strength in either arm (Q52).
pub fn readout(p: u16, r: Readout) -> Phasor {
    match r {
        Readout::Complex => phasor(p),
        Readout::Real if !(QUARTER_TURN..3 * QUARTER_TURN).contains(&p) => Phasor::new(1.0, 0.0),
        Readout::Real => Phasor::new(-1.0, 0.0),
    }
}

/// Z(P, t) = Σ_f s_f(t) · u(p_f): the Soul's phasors, each weighted by its family's
/// strength, summed over the families in order (spec 6.1). As a judge's vote phasor, its
/// direction comes from the Soul and its squared length is the judge's Character [Q68 a].
/// `soul` and `strengths` hold one entry per family; a family missing from either counts
/// nothing, and so does a family of strength 0.
pub fn vote_phasor(soul: &[u16], strengths: &[f64], r: Readout) -> Phasor {
    let mut z = Phasor::default();

    for (&p, &s) in soul.iter().zip(strengths) {
        if s != 0.0 {
            z += readout(p, r).scale(s);
        }
    }

    z
}

/// The Character χ(P, t) = |Z(P, t)|² of the Soul `soul` on a task with these strengths
/// (spec 6.1). It is computed from the Soul each time and never stored (Q55).
pub fn character(soul: &[u16], strengths: &[f64], r: Readout) -> f64 {
    vote_phasor(soul, strengths, r).norm_sqr()
}

/// The mean Character over Souls whose phases are independent and uniform: Σ_f s_f², in
/// both readouts (spec 6.1).
pub fn mean_character(strengths: &[f64]) -> f64 {
    strengths.iter().map(|s| s * s).sum()
}

/// The normalised Character χ / Σ s², whose mean over such Souls is 1 for any strengths,
/// so that tasks with different strengths can be averaged (spec 6.1). 0 when no family has
/// a strength.
pub fn character_norm(soul: &[u16], strengths: &[f64], r: Readout) -> f64 {
    let mean = mean_character(strengths);

    if mean > 0.0 {
        character(soul, strengths, r) / mean
    } else {
        0.0
    }
}

/// K_C = √(1 − Σ s⁴ / (Σ s²)²): the contrast of the Character on a task, its standard
/// deviation over independent uniform Souls divided by its mean, with complex phases (spec
/// 6.3, Q57). 0 for a single fitting family, and 0 when no family has a strength.
pub fn contrast_complex(strengths: &[f64]) -> f64 {
    let squares = mean_character(strengths);
    let fourth: f64 = strengths.iter().map(|s| s * s * s * s).sum();

    if squares > 0.0 {
        (1.0 - fourth / (squares * squares)).max(0.0).sqrt()
    } else {
        0.0
    }
}

/// K_R = √2 · K_C: the same contrast with the real readout (spec 6.3).
pub fn contrast_real(strengths: &[f64]) -> f64 {
    std::f64::consts::SQRT_2 * contrast_complex(strengths)
}

/// The contrast of the Character over independent Souls for the readout `r`: K_C or K_R.
pub fn independent_contrast(strengths: &[f64], r: Readout) -> f64 {
    match r {
        Readout::Complex => contrast_complex(strengths),
        Readout::Real => contrast_real(strengths),
    }
}

/// How the hidden angles are read (Q42, Q70; spec 3 and 12.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Readout {
    /// The unit phasor e^(2πip/65536): the Soul of arm S.
    Complex,
    /// The sign of the nearer of 0 and half a turn: the real-valued control, arm V.
    Real,
}

/// Where the Family strengths of a task come from (Q63; spec 5.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strengths {
    /// As the Domain states them for the task.
    Domain,
    /// Strength 1 for every family on every task: the control.
    Equal,
}

impl Strengths {
    /// The strengths a Run with this setting uses on `task`: those the domain states, or
    /// 1 for each of its families. With equal strengths a Worker's Character is the same on
    /// every task (spec 5.4).
    pub fn of<D: Domain>(self, domain: &D, task: &D::Task) -> Vec<f64> {
        match self {
            Strengths::Domain => domain.strengths(task),
            Strengths::Equal => vec![1.0; domain.families().len()],
        }
    }
}

/// The settings of a Run with Souls (spec 12.2). Every field is required and no other
/// field is allowed, since they define the experiment; a resume cannot change them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SoulConfig {
    /// `complex` or `real` (Q42, Q70).
    pub readout: Readout,
    /// `domain` or `equal` (Q63).
    pub strengths: Strengths,
    /// The complex Stage vote (Q46); off: the count of today.
    pub vote: bool,
    /// The phases reach the simulated models (Q46, Q72).
    pub simulator: bool,
    /// The judgments of a Stage vote are Observations (Q62).
    pub observe_vote: bool,
    /// The judgments of the peer round are Observations (Q62).
    pub observe_peer: bool,
    /// κ of the seed Workers, in thousandths (Q71).
    pub kappa_seed: u16,
    /// κ never moves (Q44).
    pub kappa_fixed: bool,
    /// The largest κ step at a birth, in thousandths.
    pub kappa_drift: u16,
    /// Children copy Souls; off: every child Worker gets a new Soul (Q64).
    pub copy: bool,
    /// Two sources mix; off: the first source is copied (Q64).
    pub mix: bool,
    /// The largest phase step at a birth, in phase units (Q64).
    pub phase_drift: u16,
    /// Hoppe's θ; 0: never a new Soul (Q64).
    pub new_soul_theta: f64,
    /// Imprinting (Q59).
    pub imprint: bool,
    /// Teaching (Q59).
    pub teach: bool,
}

impl SoulConfig {
    /// Arm S, the Soul (spec 12.2, 12.3). κ at the seeds, its drift, the phase drift and θ
    /// are placeholders that the pilot sets (ticket 11).
    pub fn arm_s() -> SoulConfig {
        SoulConfig {
            readout: Readout::Complex,
            strengths: Strengths::Domain,
            vote: true,
            simulator: true,
            observe_vote: true,
            observe_peer: true,
            kappa_seed: 200,
            kappa_fixed: false,
            kappa_drift: 20,
            copy: true,
            mix: true,
            phase_drift: 512,
            new_soul_theta: 1.0,
            imprint: true,
            teach: false,
        }
    }

    /// Arm V: arm S read as real numbers, the sign of each angle (Q42, Q70).
    pub fn arm_v() -> SoulConfig {
        SoulConfig {
            readout: Readout::Real,
            ..SoulConfig::arm_s()
        }
    }

    /// Arm K: arm S with κ fixed at 0 (Q44).
    pub fn arm_k() -> SoulConfig {
        SoulConfig {
            kappa_seed: 0,
            kappa_fixed: true,
            ..SoulConfig::arm_s()
        }
    }

    /// Refuse settings outside their ranges: κ and its step in 0..=1000 thousandths, the
    /// phase step at most half a turn, and θ finite and not negative.
    pub fn check(&self) -> Result<()> {
        let bad = |m: String| Err(BrutError::Invalid(format!("breed config: soul: {m}")));

        if self.kappa_seed > COUPLING_MAX {
            return bad(format!("kappa_seed must lie in 0..={COUPLING_MAX}"));
        }

        if self.kappa_drift > COUPLING_MAX {
            return bad(format!("kappa_drift must lie in 0..={COUPLING_MAX}"));
        }

        if self.phase_drift > HALF_TURN {
            return bad(format!("phase_drift must lie in 0..={HALF_TURN}"));
        }

        if !(self.new_soul_theta >= 0.0 && self.new_soul_theta.is_finite()) {
            return bad("new_soul_theta must be finite and not negative".to_string());
        }

        Ok(())
    }
}

/// A Stage vote tallied by amplitudes (spec 7.3, 7.4).
#[derive(Clone, Debug, PartialEq)]
pub struct ArrayVote {
    /// The Array pattern: Π(x) = |A(x)|² for every answer x, with A(x) = Σ_j w_j(x) · v_j.
    pub pattern: Vec<f64>,
    /// The answer that goes on: the largest Π, then the most verdicts, then the lowest
    /// index.
    pub winner: usize,
    /// The winner of today's count: the most verdicts, then the lowest index.
    pub count_winner: usize,
    /// True when two or more answers share the largest Π, so the tie rule decided.
    pub tied: bool,
    /// The coherence |Σ_j v_j| / Σ_j |v_j| of the judges' phasors (spec 7.7): 1 when they
    /// all point alike, near 0 when they cancel; 0 when every phasor is 0.
    pub coherence: f64,
}

/// The answer with the most verdicts, a tie to the lowest index: the count of the Stage
/// vote since Stage 2c (spec 7.1).
pub fn count_winner(wins: &[u32]) -> usize {
    let mut best = 0;

    for (i, &n) in wins.iter().enumerate() {
        if n > wins[best] {
            best = i;
        }
    }

    best
}

/// The Stage vote by amplitudes (spec 7.2 to 7.4, Q46) [Q68 a]. `verdicts[j][x]` counts
/// the verdicts of judge j for answer x, and `phasors[j]` is judge j's vote phasor v_j.
///
/// A(x) adds w_j(x) · v_j over the judges, j ascending, and Π(x) = |A(x)|². The answer with
/// the largest Π wins, compared with `total_cmp`. A tie in Π, also when every Π is 0, goes
/// to the answer with more verdicts, and a tie there to the lowest index, as today's count
/// decides. So equal phasors, or judges who all give the same verdicts, never lose the
/// count's winner (spec 7.5): with at most three judges and three answers, the sums of
/// equal phasors are exact wherever two answers can tie in the count.
pub fn array_vote(verdicts: &[Vec<u32>], phasors: &[Phasor]) -> ArrayVote {
    let answers = verdicts.iter().map(Vec::len).max().unwrap_or(0);
    let of = |w: &Vec<u32>, x: usize| w.get(x).copied().unwrap_or(0);
    let wins: Vec<u32> = (0..answers)
        .map(|x| verdicts.iter().map(|w| of(w, x)).sum())
        .collect();
    let pattern: Vec<f64> = (0..answers)
        .map(|x| {
            let mut a = Phasor::default();

            for (w, &v) in verdicts.iter().zip(phasors) {
                let n = of(w, x);

                if n > 0 {
                    a += v.scale(f64::from(n));
                }
            }

            a.norm_sqr()
        })
        .collect();
    let mut winner = 0;

    for x in 1..answers {
        let order = pattern[x]
            .total_cmp(&pattern[winner])
            .then(wins[x].cmp(&wins[winner]));

        if order == std::cmp::Ordering::Greater {
            winner = x;
        }
    }

    let top = pattern.get(winner).copied().unwrap_or(0.0);
    let tied = pattern
        .iter()
        .filter(|p| p.total_cmp(&top) == std::cmp::Ordering::Equal)
        .count()
        >= 2;
    let sum = phasors.iter().fold(Phasor::default(), |s, &v| s + v);
    let lengths: f64 = phasors.iter().map(|v| v.norm()).sum();
    // The triangle inequality bounds it by 1; the minimum removes a rounding excess.
    let coherence = if lengths > 0.0 {
        (sum.norm() / lengths).min(1.0)
    } else {
        0.0
    };

    ArrayVote {
        pattern,
        winner,
        count_winner: count_winner(&wins),
        tied,
        coherence,
    }
}

/// The draws Souletics makes for one Worker (spec 9.7), in this order: the new-Soul coin,
/// n uniform phases, n mix coins, n drift steps, the κ coin and the κ step. They are all
/// drawn whatever the switches, and each draw takes the same words of the stream whatever
/// its value, so the arms of the comparison share every draw. A part that is switched off
/// ignores its draws.
#[derive(Clone, Debug, PartialEq)]
pub struct SoulDraws {
    /// Uniform in [0, 1): below Hoppe's chance the Worker gets an entirely new Soul.
    pub new_coin: f64,
    /// The phases of a new Soul.
    pub phases: Vec<u16>,
    /// One coin per family, uniform in [0, 1): below 1/2 a mix takes the second source.
    pub mix: Vec<f64>,
    /// One raw draw per family for the phase step (`uniform_step`).
    pub drift: Vec<u64>,
    /// Uniform in [0, 1): below 1/2 κ comes from the second of two sources.
    pub kappa_coin: f64,
    /// The raw draw for the κ step.
    pub kappa_step: u64,
}

impl SoulDraws {
    /// The draws for one Worker of a Soul of `n` phases, from `rng`.
    pub fn draw(rng: &mut ChaCha8Rng, n: usize) -> SoulDraws {
        let new_coin = rng.random::<f64>();
        let phases = (0..n).map(|_| rng.random::<u16>()).collect();
        let mix = (0..n).map(|_| rng.random::<f64>()).collect();
        let drift = (0..n).map(|_| rng.random::<u64>()).collect();
        let kappa_coin = rng.random::<f64>();
        let kappa_step = rng.random::<u64>();

        SoulDraws {
            new_coin,
            phases,
            mix,
            drift,
            kappa_coin,
            kappa_step,
        }
    }
}

/// A uniform integer in -d..=d from one raw draw, by the widening product of Lemire
/// (2019), so a step takes one draw whatever `d` is. The bias is below (2d + 1) / 2^64.
pub fn uniform_step(raw: u64, d: u16) -> i32 {
    let span = 2 * u128::from(d) + 1;
    let k = (u128::from(raw) * span) >> 64;

    // k lies in 0..=2d, so it fits.
    k as i32 - i32::from(d)
}

/// The phase `p` moved by `step` phase units, modulo a whole turn.
pub fn turn(p: u16, step: i32) -> u16 {
    (i32::from(p) + step).rem_euclid(65536) as u16
}

/// Hoppe's chance θ / (θ + N) that a Worker gets an entirely new Soul, when `urn` Souls are
/// in the urn (spec 9.4). θ = 0 never gives one.
pub fn new_soul_chance(theta: f64, urn: usize) -> f64 {
    if theta > 0.0 {
        theta / (theta + urn as f64)
    } else {
        0.0
    }
}

/// What a source Worker passes on by Souletics: its birth Soul, never its current one
/// (Q65), and its Coupling strength.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Inheritance<'a> {
    pub phases: &'a [u16],
    pub coupling: u16,
}

/// A Worker's birth Soul and Coupling strength, as Souletics gives them.
#[derive(Clone, Debug, PartialEq)]
pub struct Birth {
    pub phases: Vec<u16>,
    pub coupling: u16,
    /// True for an entirely new Soul, Hoppe's new colour.
    pub new: bool,
}

/// Souletics for one Worker (spec 9.2 to 9.5) from its sources, in parent order, when
/// `urn` Souls are already in the urn. A Worker without a source, a seed, gets a new Soul
/// and κ of the seeds. Otherwise the coin decides first whether the Soul is entirely new
/// (always, with `copy` off). If not, one source's birth Soul is copied, or two are mixed
/// by one coin per family (with `mix` off the first is copied). Every phase then drifts.
/// κ is copied from the one source or taken from one of two by a coin, also for a new
/// Soul, moved and clamped to 0..=1000; with `kappa_fixed` it is κ of the seeds.
pub fn souletics(
    cfg: &SoulConfig,
    sources: &[Inheritance<'_>],
    urn: usize,
    draws: &SoulDraws,
) -> Birth {
    let new = match sources {
        [] => true,
        _ => !cfg.copy || draws.new_coin < new_soul_chance(cfg.new_soul_theta, urn),
    };
    let phase = |f: usize| -> u16 {
        let of = |s: &Inheritance<'_>| s.phases.get(f).copied().unwrap_or_default();

        match sources {
            _ if new => draws.phases[f],
            [_, second, ..] if cfg.mix && draws.mix[f] < 0.5 => of(second),
            [first, ..] => of(first),
            [] => draws.phases[f],
        }
    };
    let phases = (0..draws.phases.len())
        .map(|f| turn(phase(f), uniform_step(draws.drift[f], cfg.phase_drift)))
        .collect();
    let coupling = match sources {
        _ if cfg.kappa_fixed => cfg.kappa_seed,
        [] => cfg.kappa_seed,
        [first, rest @ ..] => {
            let from = match rest {
                [second, ..] if draws.kappa_coin < 0.5 => second,
                _ => first,
            };
            let moved = i32::from(from.coupling) + uniform_step(draws.kappa_step, cfg.kappa_drift);

            moved.clamp(0, i32::from(COUPLING_MAX)) as u16
        }
    };

    Birth {
        phases,
        coupling,
        new,
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{SQRT_2, TAU};

    use rand::SeedableRng;

    use super::*;
    use crate::cascade::circuit::{CircuitDomain, library_target, verified};
    use crate::cascade::domain::{CircuitConfig, Domain, GridDomain};
    use crate::cascade::puzzle::{Geo, Rule, generate_panel};
    use crate::cascade::sim::rule_strengths;
    use crate::util::derive_seed;

    const READOUTS: [Readout; 2] = [Readout::Complex, Readout::Real];

    /// The JSON of arm S, field by field as the table of spec 12.2 lists them.
    const ARM_S: &str = r#"{"readout":"complex","strengths":"domain","vote":true,"simulator":true,"observe_vote":true,"observe_peer":true,"kappa_seed":200,"kappa_fixed":false,"kappa_drift":20,"copy":true,"mix":true,"phase_drift":512,"new_soul_theta":1.0,"imprint":true,"teach":false}"#;

    #[test]
    fn the_arms_are_the_table_of_the_spec() {
        let s = SoulConfig::arm_s();

        assert_eq!(serde_json::to_string(&s).expect("json"), ARM_S);
        assert_eq!(serde_json::from_str::<SoulConfig>(ARM_S).expect("read"), s);

        // Arm V differs from S only in the readout, arm K only in κ, fixed at 0.
        let v = SoulConfig {
            readout: Readout::Real,
            ..SoulConfig::arm_s()
        };
        let k = SoulConfig {
            kappa_seed: 0,
            kappa_fixed: true,
            ..SoulConfig::arm_s()
        };

        assert_eq!(SoulConfig::arm_v(), v);
        assert_eq!(SoulConfig::arm_k(), k);
        assert!(
            serde_json::to_string(&v)
                .expect("json")
                .contains(r#""readout":"real""#)
        );

        for arm in [s, v, k] {
            assert!(arm.check().is_ok(), "{arm:?}");
        }

        let equal = SoulConfig {
            strengths: Strengths::Equal,
            ..SoulConfig::arm_s()
        };

        assert!(
            serde_json::to_string(&equal)
                .expect("json")
                .contains(r#""strengths":"equal""#)
        );
    }

    #[test]
    fn every_soul_setting_is_required_and_no_other_is_allowed() {
        let full: serde_json::Value = serde_json::from_str(ARM_S).expect("json");
        let fields: Vec<String> = full.as_object().expect("object").keys().cloned().collect();

        assert_eq!(fields.len(), 15);

        for field in &fields {
            let mut missing = full.clone();

            missing.as_object_mut().expect("object").remove(field);

            let err = serde_json::from_value::<SoulConfig>(missing).expect_err(field);

            assert!(err.to_string().contains("missing field"), "{field}: {err}");
        }

        let mut extra = full.clone();

        extra
            .as_object_mut()
            .expect("object")
            .insert("strength".to_string(), serde_json::json!("domain"));

        let err = serde_json::from_value::<SoulConfig>(extra).expect_err("unknown field");

        assert!(err.to_string().contains("unknown field"), "{err}");

        for (field, bad) in [("readout", "imaginary"), ("strengths", "learned")] {
            let mut wrong = full.clone();

            wrong[field] = serde_json::json!(bad);
            assert!(
                serde_json::from_value::<SoulConfig>(wrong).is_err(),
                "{field}"
            );
        }
    }

    #[test]
    fn the_real_readout_splits_at_quarter_turns() {
        let cases = [
            (16383, 1.0),
            (16384, -1.0),
            (49151, -1.0),
            (49152, 1.0),
            (0, 1.0),
            (32768, -1.0),
            (65535, 1.0),
        ];

        for (p, sign) in cases {
            assert_eq!(readout(p, Readout::Real), Phasor::new(sign, 0.0), "{p}");
        }

        // Half of all phases read +1, half -1.
        let plus = (0..=u16::MAX)
            .filter(|&p| readout(p, Readout::Real).re > 0.0)
            .count();

        assert_eq!(plus, 32768);
    }

    #[test]
    fn the_unit_phasor_is_exact_at_quarter_turns() {
        assert_eq!(phasor(0), Phasor::new(1.0, 0.0));
        assert_eq!(phasor(16384), Phasor::new(0.0, 1.0));
        assert_eq!(phasor(32768), Phasor::new(-1.0, 0.0));
        assert_eq!(phasor(49152), Phasor::new(0.0, -1.0));

        for p in 0..=u16::MAX {
            let u = phasor(p);
            let angle = TAU * f64::from(p) / 65536.0;

            assert!((u.norm_sqr() - 1.0).abs() < 1e-15, "{p}");
            assert!((u.re - angle.cos()).abs() < 1e-15, "{p}");
            assert!((u.im - angle.sin()).abs() < 1e-15, "{p}");

            // A quarter turn multiplies by i, exactly.
            assert_eq!(
                phasor(p.wrapping_add(16384)),
                Phasor::new(-u.im, u.re),
                "{p}"
            );
            assert_eq!(readout(p, Readout::Complex), u);
        }
    }

    #[test]
    fn the_character_is_the_squared_weighted_phase_sum() {
        // Two families half a turn apart cancel; a quarter turn apart they add to |1 + i|^2.
        let s = [1.0, 1.0, 0.0];

        assert_eq!(character(&[0, 32768, 7], &s, Readout::Complex), 0.0);
        assert_eq!(character(&[0, 16384, 7], &s, Readout::Complex), 2.0);
        assert_eq!(character(&[0, 16384, 7], &s, Readout::Real), 0.0);
        assert_eq!(
            vote_phasor(&[0, 16384, 7], &s, Readout::Complex),
            Phasor::new(1.0, 1.0)
        );

        // Strengths weight the phasors: 2 + 1 i, then the square of its length.
        let weighted = [2.0, 1.0, 0.5];

        assert_eq!(
            vote_phasor(&[0, 16384, 32768], &weighted, Readout::Complex),
            Phasor::new(1.5, 1.0)
        );
        assert_eq!(
            character(&[0, 16384, 32768], &weighted, Readout::Complex),
            3.25
        );
        assert_eq!(mean_character(&weighted), 5.25);
        assert_eq!(
            character_norm(&[0, 16384, 32768], &weighted, Readout::Complex),
            3.25 / 5.25
        );

        // No strength at all gives no Character, and the normalised one is 0, not NaN.
        assert_eq!(character(&[1, 2], &[0.0, 0.0], Readout::Complex), 0.0);
        assert_eq!(character_norm(&[1, 2], &[0.0, 0.0], Readout::Real), 0.0);
        assert_eq!(independent_contrast(&[0.0, 0.0], Readout::Real), 0.0);
    }

    #[test]
    fn the_contrast_formulas_give_the_table_of_the_spec() {
        let equal = |k: usize| vec![1.0; k];
        let rows: [(Vec<f64>, f64, f64); 6] = [
            (equal(1), 0.0, 0.0),
            (equal(2), 0.707, 1.000),
            (equal(3), 0.816, 1.155),
            (equal(5), 0.894, 1.265),
            (equal(8), 0.935, 1.323),
            (equal(7), 0.926, 1.309),
        ];

        for (s, kc, kr) in rows {
            assert!((contrast_complex(&s) - kc).abs() < 5e-4, "{s:?}");
            assert!((contrast_real(&s) - kr).abs() < 5e-4, "{s:?}");
            assert_eq!(
                independent_contrast(&s, Readout::Complex),
                contrast_complex(&s)
            );
            assert_eq!(independent_contrast(&s, Readout::Real), contrast_real(&s));
        }

        // Zero strengths do not count, and only their ratios matter.
        assert_eq!(
            contrast_complex(&[0.0, 2.0, 0.0, 2.0]),
            contrast_complex(&equal(2))
        );

        // The mean over the 11 circuit targets, each target once.
        let targets = verified();
        let mean = |r: Readout| {
            targets
                .iter()
                .map(|t| independent_contrast(&t.strengths(), r))
                .sum::<f64>()
                / targets.len() as f64
        };

        assert!((mean(Readout::Complex) - 0.655).abs() < 5e-4);
        assert!((mean(Readout::Real) - 0.927).abs() < 5e-4);
    }

    /// For each strength vector of `cases`: the mean normalised Character and its contrast,
    /// with the complex and with the real readout, over `n` Souls of `families` phases drawn
    /// from `seed`. Both readouts read the same Souls.
    fn fresh_soul_statistics(
        families: usize,
        cases: &[Vec<f64>],
        seed: u64,
        n: usize,
    ) -> Vec<[(f64, f64); 2]> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut sums = vec![[(0.0f64, 0.0f64); 2]; cases.len()];
        let mut soul = vec![0u16; families];

        for _ in 0..n {
            for p in &mut soul {
                *p = rng.random();
            }

            for (case, s) in sums.iter_mut().zip(cases) {
                for (sum, r) in case.iter_mut().zip(READOUTS) {
                    let x = character_norm(&soul, s, r);

                    sum.0 += x;
                    sum.1 += x * x;
                }
            }
        }

        sums.iter()
            .map(|case| {
                case.map(|(sum, squares)| {
                    let mean = sum / n as f64;
                    let variance = (squares / n as f64 - mean * mean).max(0.0);

                    (mean, variance.sqrt() / mean)
                })
            })
            .collect()
    }

    #[test]
    fn the_character_of_fresh_souls_meets_the_formulas() {
        const SOULS: usize = 1 << 20;

        // The strengths of every circuit target, and of equal strengths.
        let circuits = CircuitDomain::new(&CircuitConfig::default()).expect("circuit domain");
        let n_circuit = circuits.families().len();
        let mut circuit_cases: Vec<Vec<f64>> = verified().iter().map(|t| t.strengths()).collect();

        circuit_cases.push(vec![1.0; n_circuit]);

        // The strengths of every kind of grid rule: each geometric family alone, the colour
        // map alone, each geometric family with a colour map; and equal strengths.
        let map = Some([0, 2, 3, 1, 4, 5, 6, 7, 8, 9]);
        let geos = [Geo::FlipH, Geo::Rot90, Geo::Transpose, Geo::GravityDown];
        let n_grid = GridDomain.families().len();
        let mut grid_cases: Vec<Vec<f64>> = geos
            .iter()
            .map(|&geo| rule_strengths(&Rule { geo, colours: None }))
            .chain([rule_strengths(&Rule {
                geo: Geo::Identity,
                colours: map,
            })])
            .chain(
                geos.iter()
                    .map(|&geo| rule_strengths(&Rule { geo, colours: map })),
            )
            .collect();

        grid_cases.push(vec![1.0; n_grid]);

        let domains = [
            ("circuits", n_circuit, circuit_cases, "fresh/circuits"),
            ("grids", n_grid, grid_cases, "fresh/grids"),
        ];

        for (domain, families, cases, tag) in domains {
            let stats = fresh_soul_statistics(families, &cases, derive_seed(2, tag), SOULS);

            for (s, [(mean_c, contrast_c), (mean_r, contrast_r)]) in cases.iter().zip(stats) {
                let what = format!("{domain} {s:?}");

                assert!((mean_c - 1.0).abs() < 0.01, "{what}: complex mean {mean_c}");
                assert!((mean_r - 1.0).abs() < 0.01, "{what}: real mean {mean_r}");
                assert!(
                    (contrast_c - contrast_complex(s)).abs() < 0.01,
                    "{what}: complex contrast {contrast_c}, formula {}",
                    contrast_complex(s)
                );
                assert!(
                    (contrast_r - contrast_real(s)).abs() < 0.01,
                    "{what}: real contrast {contrast_r}, formula {}",
                    contrast_real(s)
                );

                if s.iter().filter(|&&x| x > 0.0).count() >= 2 {
                    let ratio = contrast_r / contrast_c;

                    assert!((ratio - SQRT_2).abs() < 0.01, "{what}: ratio {ratio}");
                }
            }
        }
    }

    #[test]
    fn one_family_gives_no_pattern() {
        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(3, "one family"));

        for name in ["swap", "cs"] {
            let s = library_target(name).expect(name).strengths();

            assert_eq!(s.iter().filter(|&&x| x > 0.0).count(), 1, "{name}");
            assert_eq!(
                (contrast_complex(&s), contrast_real(&s)),
                (0.0, 0.0),
                "{name}"
            );

            for _ in 0..10_000 {
                let soul: Vec<u16> = (0..s.len()).map(|_| rng.random()).collect();

                for r in READOUTS {
                    assert!(
                        (character(&soul, &s, r) - 1.0).abs() < 1e-12,
                        "{name} {r:?}"
                    );
                    assert!(
                        (character_norm(&soul, &s, r) - 1.0).abs() < 1e-12,
                        "{name} {r:?}"
                    );
                }
            }
        }

        // The same holds for a grid puzzle whose rule is one step or one colour map alone.
        let alone = generate_panel(5, 50)
            .into_iter()
            .map(|p| rule_strengths(&p.rule))
            .find(|s| s.iter().filter(|&&x| x > 0.0).count() == 1)
            .expect("a one-family puzzle");

        for p in [0, 9999, 40000] {
            assert_eq!(character(&[p; 7], &alone, Readout::Real), 1.0);
        }
    }

    #[test]
    fn a_coherent_soul_is_k_times_brighter() {
        for k in 1..=8usize {
            let s: Vec<f64> = (0..8).map(|f| if f < k { 1.0 } else { 0.0 }).collect();
            let k = k as f64;

            assert_eq!(mean_character(&s), k);

            for p in [0, 1, 12345, 16384, 32768, 50000, 65535] {
                let soul = [p; 8];

                for r in READOUTS {
                    let chi = character(&soul, &s, r);

                    assert!((chi - k * k).abs() < 1e-9, "{k} {p} {r:?}: {chi}");
                    assert!((chi / mean_character(&s) - k).abs() < 1e-9, "{k} {p} {r:?}");
                    assert!(
                        (character_norm(&soul, &s, r) - k).abs() < 1e-9,
                        "{k} {p} {r:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn equal_strengths_give_every_family_one() {
        let grid = GridDomain;
        let puzzles = generate_panel(4, 20);
        let circuits = CircuitDomain::new(&CircuitConfig::default()).expect("circuit domain");
        let tasks = circuits.panel(4, 11);

        for p in &puzzles {
            assert_eq!(Strengths::Domain.of(&grid, p), grid.strengths(p));
            assert_eq!(Strengths::Equal.of(&grid, p), vec![1.0; 7]);
        }

        for t in &tasks {
            assert_eq!(Strengths::Domain.of(&circuits, t), circuits.strengths(t));
            assert_eq!(Strengths::Equal.of(&circuits, t), vec![1.0; 8]);
        }

        // With equal strengths a Soul has the same Character on every task.
        let soul = [1, 5000, 20000, 33000, 40000, 51000, 60000, 65000];
        let characters: Vec<f64> = tasks
            .iter()
            .map(|t| character(&soul, &Strengths::Equal.of(&circuits, t), Readout::Complex))
            .collect();

        assert!(
            characters.windows(2).all(|w| w[0] == w[1]),
            "{characters:?}"
        );
    }

    /// Draws whose every coin and step is given: the new-Soul coin `new`, the mix coins
    /// `mix`, raw drift steps `drift` and a raw κ step.
    fn draws(new: f64, phases: &[u16], mix: &[f64], drift: &[u64], kappa: (f64, u64)) -> SoulDraws {
        SoulDraws {
            new_coin: new,
            phases: phases.to_vec(),
            mix: mix.to_vec(),
            drift: drift.to_vec(),
            kappa_coin: kappa.0,
            kappa_step: kappa.1,
        }
    }

    /// The raw draw that `uniform_step` maps to `step` in -d..=d.
    fn raw_for(step: i32, d: u16) -> u64 {
        let span = 2 * u128::from(d) + 1;
        let k = u128::try_from(i64::from(step) + i64::from(d)).expect("in range");

        u64::try_from(((k << 64) + (1 << 63)) / span).expect("fits")
    }

    #[test]
    fn uniform_steps_cover_their_range_evenly() {
        for d in [0u16, 1, 2, 20, 512, 32768] {
            for step in [-i32::from(d), -1, 0, 1, i32::from(d)] {
                if step.unsigned_abs() <= u32::from(d) {
                    assert_eq!(uniform_step(raw_for(step, d), d), step, "{step} of {d}");
                }
            }

            assert_eq!(uniform_step(0, d), -i32::from(d));
            assert_eq!(uniform_step(u64::MAX, d), i32::from(d));
        }

        // Every value of -2..=2 comes up a fifth of the time.
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let mut counts = [0usize; 5];

        for _ in 0..100_000 {
            counts[(uniform_step(rng.random(), 2) + 2) as usize] += 1;
        }

        assert!(
            counts.iter().all(|&c| c.abs_diff(20_000) < 600),
            "{counts:?}"
        );

        // A step wraps around the turn.
        assert_eq!(turn(65535, 1), 0);
        assert_eq!(turn(0, -1), 65535);
        assert_eq!(turn(100, -32768), 100 + 32768);
    }

    #[test]
    fn souletics_copies_mixes_and_drifts_by_its_draws() {
        let cfg = SoulConfig::arm_s();
        let a = [10u16, 20, 30];
        let b = [65000u16, 40000, 7];
        let one = [Inheritance {
            phases: &a,
            coupling: 300,
        }];
        let two = [
            one[0],
            Inheritance {
                phases: &b,
                coupling: 700,
            },
        ];
        let still = [raw_for(0, 512); 3];
        let moved = [raw_for(-512, 512), raw_for(1, 512), raw_for(512, 512)];
        let no_new = 0.99;

        // One source: a copy of its birth Soul, then the drift.
        let copy = souletics(
            &cfg,
            &one,
            4,
            &draws(no_new, &[1, 2, 3], &[0.0; 3], &still, (0.0, 0)),
        );

        assert_eq!((copy.phases.as_slice(), copy.new), (&a[..], false));

        let drifted = souletics(
            &cfg,
            &one,
            4,
            &draws(no_new, &[1, 2, 3], &[0.0; 3], &moved, (0.0, 0)),
        );

        assert_eq!(drifted.phases, [65034, 21, 542]);

        // Two sources: one coin per family, below 1/2 the second source.
        let mixed = souletics(
            &cfg,
            &two,
            4,
            &draws(no_new, &[1, 2, 3], &[0.1, 0.9, 0.4], &still, (0.0, 0)),
        );

        assert_eq!(mixed.phases, [65000, 20, 7]);

        // Without mixing the first source is copied.
        let unmixed = SoulConfig {
            mix: false,
            ..cfg.clone()
        };
        let first = souletics(
            &unmixed,
            &two,
            4,
            &draws(no_new, &[1, 2, 3], &[0.1, 0.9, 0.4], &still, (0.0, 0)),
        );

        assert_eq!(first.phases, a);

        // A new Soul takes the drawn phases, and the drift follows as for a copy.
        let fresh = souletics(
            &cfg,
            &two,
            4,
            &draws(0.0, &[1, 2, 3], &[0.1; 3], &moved, (0.0, 0)),
        );

        assert_eq!(
            (fresh.phases.as_slice(), fresh.new),
            (&[65025, 3, 515][..], true)
        );

        // A seed has no source: a new Soul and κ of the seeds.
        let seed = souletics(
            &cfg,
            &[],
            0,
            &draws(0.99, &[5, 6, 7], &[0.0; 3], &still, (0.0, 0)),
        );

        assert_eq!(
            (seed.phases.as_slice(), seed.new, seed.coupling),
            (&[5, 6, 7][..], true, 200)
        );
    }

    #[test]
    fn hoppes_chance_follows_the_urn() {
        assert_eq!(new_soul_chance(1.0, 4), 0.2);
        assert_eq!(new_soul_chance(1.0, 0), 1.0);
        assert_eq!(new_soul_chance(3.0, 1), 0.75);
        assert_eq!(new_soul_chance(0.0, 0), 0.0);
        assert_eq!(new_soul_chance(0.0, 1000), 0.0);

        // The coin compares with the chance: below it a new Soul.
        let cfg = SoulConfig::arm_s();
        let a = [1u16, 2];
        let one = [Inheritance {
            phases: &a,
            coupling: 0,
        }];
        let at = |coin: f64, urn: usize| {
            souletics(
                &cfg,
                &one,
                urn,
                &draws(coin, &[9, 9], &[0.0; 2], &[raw_for(0, 512); 2], (0.0, 0)),
            )
            .new
        };

        assert!(at(0.199, 4) && !at(0.2, 4));
        assert!(at(0.0, 1_000_000) && !at(0.01, 1000));

        // Without copies every Worker with a source gets a new Soul.
        let fresh = SoulConfig {
            copy: false,
            ..cfg.clone()
        };

        assert!(
            souletics(
                &fresh,
                &one,
                1000,
                &draws(0.99, &[9, 9], &[0.0; 2], &[raw_for(0, 512); 2], (0.0, 0))
            )
            .new
        );
    }

    #[test]
    fn souletics_gives_kappa_by_its_sources() {
        let cfg = SoulConfig::arm_s();
        let p = [0u16; 2];
        let still = [raw_for(0, 512); 2];
        let from = |k: u16| Inheritance {
            phases: &p,
            coupling: k,
        };
        let kappa = |cfg: &SoulConfig, sources: &[Inheritance<'_>], coin: f64, step: i32| {
            souletics(
                cfg,
                sources,
                9,
                &draws(
                    0.99,
                    &[0, 0],
                    &[0.0; 2],
                    &still,
                    (coin, raw_for(step, cfg.kappa_drift)),
                ),
            )
            .coupling
        };

        // Copied from one source, taken from one of two by the coin, then moved.
        assert_eq!(kappa(&cfg, &[from(300)], 0.1, 0), 300);
        assert_eq!(kappa(&cfg, &[from(300), from(700)], 0.9, 0), 300);
        assert_eq!(kappa(&cfg, &[from(300), from(700)], 0.1, 0), 700);
        assert_eq!(kappa(&cfg, &[from(300)], 0.5, -20), 280);
        assert_eq!(kappa(&cfg, &[from(300)], 0.5, 20), 320);

        // Clamped into 0..=1000.
        assert_eq!(kappa(&cfg, &[from(5)], 0.5, -20), 0);
        assert_eq!(kappa(&cfg, &[from(995)], 0.5, 20), 1000);

        // A new Soul keeps the κ of its source, since κ is no part of the Soul.
        let new = souletics(
            &cfg,
            &[from(450)],
            9,
            &draws(0.0, &[3, 4], &[0.0; 2], &still, (0.5, raw_for(0, 20))),
        );

        assert_eq!((new.new, new.coupling), (true, 450));

        // With κ fixed every Worker has the κ of the seeds; arm K's is 0.
        let fixed = SoulConfig {
            kappa_fixed: true,
            ..cfg.clone()
        };

        assert_eq!(kappa(&fixed, &[from(300), from(700)], 0.1, 20), 200);
        assert_eq!(kappa(&SoulConfig::arm_k(), &[from(300)], 0.1, 20), 0);
        assert_eq!(kappa(&SoulConfig::arm_k(), &[], 0.1, 20), 0);
    }

    #[test]
    fn soul_draws_take_a_fixed_part_of_the_stream() {
        // Every draw takes the same words whatever its value, so a part switched off and a
        // different step size leave the later draws where they are (spec 9.7).
        for n in [1usize, 7, 8] {
            let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(11, "soul/3/0"));
            let first = SoulDraws::draw(&mut rng, n);
            let after_one = rng.get_word_pos();
            let second = SoulDraws::draw(&mut rng, n);

            assert_eq!(after_one as usize, 2 + n + 2 * n + 2 * n + 2 + 2, "{n}");
            assert_eq!(rng.get_word_pos(), 2 * after_one);
            assert_eq!(
                (first.phases.len(), first.mix.len(), first.drift.len()),
                (n, n, n)
            );
            assert_ne!(first, second);
        }
    }

    /// The verdicts of a vote of `judges` judges on `answers` answers: every judge judges
    /// every pair (x, y), x < y, and `pick(j, x, y)` gives its verdict, Some(x), Some(y) or
    /// None for a call without one.
    fn verdicts(
        answers: usize,
        judges: usize,
        mut pick: impl FnMut(usize, usize, usize) -> Option<usize>,
    ) -> Vec<Vec<u32>> {
        let mut w = vec![vec![0u32; answers]; judges];

        for x in 0..answers {
            for y in x + 1..answers {
                for (j, row) in w.iter_mut().enumerate() {
                    if let Some(chosen) = pick(j, x, y) {
                        row[chosen] += 1;
                    }
                }
            }
        }

        w
    }

    /// The verdicts of each answer over all judges.
    fn totals(verdicts: &[Vec<u32>]) -> Vec<u32> {
        (0..verdicts[0].len())
            .map(|x| verdicts.iter().map(|w| w[x]).sum())
            .collect()
    }

    /// A random verdict on the pair (x, y): x, y or none.
    fn random_pick(rng: &mut ChaCha8Rng, x: usize, y: usize) -> Option<usize> {
        match rng.random_range(0..5) {
            0 | 1 => Some(x),
            2 | 3 => Some(y),
            _ => None,
        }
    }

    /// Random Family strengths of `n` families: 0 or 1 with at least one 1, as the Domains
    /// state them, or random positive numbers.
    fn random_strengths(n: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
        if rng.random_bool(0.5) {
            let mut s: Vec<f64> = (0..n)
                .map(|_| f64::from(rng.random_range(0..2u8)))
                .collect();

            s[rng.random_range(0..n)] = 1.0;

            s
        } else {
            (0..n).map(|_| rng.random_range(0.01..3.0)).collect()
        }
    }

    /// A random Soul of `n` phases.
    fn random_soul(n: usize, rng: &mut ChaCha8Rng) -> Vec<u16> {
        (0..n).map(|_| rng.random()).collect()
    }

    #[test]
    fn equal_phasors_give_the_count() {
        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(13, "equal phasors"));
        let mut cases = 0;

        while cases < 10_000 {
            let answers = rng.random_range(2..=3);
            let judges = rng.random_range(1..=3);
            let table = verdicts(answers, judges, |_, x, y| random_pick(&mut rng, x, y));

            // One phasor for every judge: a Soul's vote phasor in either readout, or any
            // other non-zero number.
            let v = match cases % 3 {
                0 => Phasor::new(rng.random_range(-4.0..4.0), rng.random_range(-4.0..4.0)),
                k => {
                    let s = random_strengths(8, &mut rng);

                    vote_phasor(&random_soul(8, &mut rng), &s, READOUTS[k - 1])
                }
            };

            if v.norm_sqr() == 0.0 {
                continue;
            }

            let vote = array_vote(&table, &vec![v; judges]);
            let count = count_winner(&totals(&table));

            assert_eq!(vote.winner, count, "{table:?} with {v:?}");
            assert_eq!(vote.count_winner, count);
            cases += 1;
        }
    }

    #[test]
    fn a_unanimous_vote_is_never_lost() {
        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(17, "unanimous"));
        let mut opposite_on_one = 0;

        for case in 0..10_000 {
            let answers = rng.random_range(2..=3);
            let judges = rng.random_range(1..=3);
            let mut picks = [[None; 3]; 3];

            for (x, row) in picks.iter_mut().enumerate().take(answers) {
                for (y, pick) in row.iter_mut().enumerate().take(answers).skip(x + 1) {
                    *pick = random_pick(&mut rng, x, y);
                }
            }

            // Every judge gives the same verdicts.
            let table = verdicts(answers, judges, |_, x, y| picks[x][y]);
            let readout = READOUTS[case % 2];
            let s = random_strengths(7, &mut rng);
            let phasors: Vec<Phasor> = (0..judges)
                .map(|_| vote_phasor(&random_soul(7, &mut rng), &s, readout))
                .collect();

            assert!(table.windows(2).all(|w| w[0] == w[1]));

            let vote = array_vote(&table, &phasors);

            assert_eq!(
                vote.winner,
                count_winner(&totals(&table)),
                "{table:?} {phasors:?}"
            );

            // Count the cases of two equally loud real judges of opposite signs that both
            // choose the answer of index 1.
            let one = s.iter().filter(|&&x| x > 0.0).count() == 1;

            if readout == Readout::Real
                && one
                && judges == 2
                && phasors[0] == phasors[1].scale(-1.0)
                && table[0] == [0, 1]
            {
                opposite_on_one += 1;
            }
        }

        // The case itself: both choose index 1, their phasors cancel exactly, and the
        // count keeps their answer.
        let opposite = [0u16, 32768].map(|p| readout(p, Readout::Real));
        let vote = array_vote(&[vec![0, 1], vec![0, 1]], &opposite);

        assert_eq!(vote.pattern, [0.0, 0.0]);
        assert_eq!((vote.winner, vote.count_winner, vote.tied), (1, 1, true));
        assert!(opposite_on_one > 0, "the random cases hold it too");
    }

    #[test]
    fn interference_cases() {
        // Three judges of equal Character at 0, 120 and 240 degrees, the nearest integer
        // phases on one fitting family.
        let s = [1.0, 0.0];
        let at = [0u16, 21845, 43691].map(|p| vote_phasor(&[p, 0], &s, Readout::Complex));

        for x in 0..3 {
            let (y, z) = match x {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };

            // All prefer x to both others and split two to one on (y, z): x loses, although
            // it has the most verdicts.
            let split = verdicts(3, 3, |j, a, b| match (a, b) {
                _ if a == x || b == x => Some(x),
                _ => Some(if j < 2 { y } else { z }),
            });
            let vote = array_vote(&split, &at);

            assert_eq!(vote.count_winner, x);
            assert_ne!(vote.winner, x, "x at index {x}: {:?}", vote.pattern);
            assert!(vote.pattern[x] < 1e-6 && vote.pattern[y] > 0.5 && vote.pattern[z] > 0.5);

            // If they agree on (y, z) too, x wins, as in the count.
            let agreed = verdicts(3, 3, |_, a, b| match (a, b) {
                _ if a == x || b == x => Some(x),
                _ => Some(y),
            });

            assert_eq!(array_vote(&agreed, &at).winner, x);
        }

        // In a split vote the louder judge wins: three coherent fitting families against
        // one, whatever the index.
        let loud = vote_phasor(&[0, 0, 0], &[1.0, 1.0, 1.0], Readout::Complex);
        let quiet = vote_phasor(&[9000, 0, 0], &[1.0, 0.0, 0.0], Readout::Complex);

        assert_eq!((loud.norm_sqr(), quiet.norm_sqr()), (9.0, 1.0));

        for chosen in [0usize, 1] {
            // Judge 0 is loud and chooses `chosen`, judge 1 quiet and chooses the other.
            let table = verdicts(2, 2, |j, _, _| {
                Some(if j == 0 { chosen } else { 1 - chosen })
            });
            let vote = array_vote(&table, &[loud, quiet]);

            assert_eq!(vote.count_winner, 0, "a tie in the count goes to index 0");
            assert_eq!(vote.winner, chosen, "the louder judge decides");
            assert!(!vote.tied);
        }

        // Two equally loud judges of the real readout with opposite signs: when both choose
        // x, x wins by the count; when they split, the index decides.
        let opposite = [0u16, 32768].map(|p| vote_phasor(&[p, 0], &s, Readout::Real));

        for x in 0..2 {
            let both = verdicts(2, 2, |_, _, _| Some(x));
            let vote = array_vote(&both, &opposite);

            assert_eq!((vote.winner, vote.tied), (x, true));

            let split = verdicts(2, 2, |j, _, _| Some(if j == 0 { x } else { 1 - x }));
            let vote = array_vote(&split, &opposite);

            assert_eq!(vote.pattern, [1.0, 1.0]);
            assert_eq!((vote.winner, vote.tied), (0, true));
        }

        // The coherence of the judges' phasors: 1 when they point alike, 0 when they cancel.
        assert_eq!(
            array_vote(&verdicts(2, 2, |_, _, _| Some(0)), &[loud, loud]).coherence,
            1.0
        );
        assert_eq!(
            array_vote(&verdicts(2, 2, |_, _, _| Some(0)), &opposite).coherence,
            0.0
        );
        assert_eq!(
            array_vote(&verdicts(2, 1, |_, _, _| Some(0)), &[Phasor::default()]).coherence,
            0.0
        );
    }

    #[test]
    fn soul_settings_out_of_range_are_refused() {
        let s = SoulConfig::arm_s();
        let edges = [
            SoulConfig {
                kappa_seed: 1000,
                kappa_drift: 1000,
                phase_drift: 32768,
                new_soul_theta: 0.0,
                ..s.clone()
            },
            SoulConfig {
                kappa_seed: 0,
                kappa_drift: 0,
                phase_drift: 0,
                new_soul_theta: 1e6,
                ..s.clone()
            },
        ];

        for ok in edges {
            assert!(ok.check().is_ok(), "{ok:?}");
        }

        let bad = [
            (
                "kappa_seed",
                SoulConfig {
                    kappa_seed: 1001,
                    ..s.clone()
                },
            ),
            (
                "kappa_drift",
                SoulConfig {
                    kappa_drift: 1001,
                    ..s.clone()
                },
            ),
            (
                "phase_drift",
                SoulConfig {
                    phase_drift: 32769,
                    ..s.clone()
                },
            ),
            (
                "new_soul_theta",
                SoulConfig {
                    new_soul_theta: -0.5,
                    ..s.clone()
                },
            ),
            (
                "new_soul_theta",
                SoulConfig {
                    new_soul_theta: f64::NAN,
                    ..s.clone()
                },
            ),
            (
                "new_soul_theta",
                SoulConfig {
                    new_soul_theta: f64::INFINITY,
                    ..s.clone()
                },
            ),
        ];

        for (field, cfg) in bad {
            let err = cfg.check().expect_err(field);

            assert!(err.to_string().contains(field), "{field}: {err}");
        }
    }
}
