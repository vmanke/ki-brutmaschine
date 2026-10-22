//! Positive control: does the HIDDEN-GRAMMAR primary endpoint have headroom?
//!
//! We run the REAL, unmodified primary-endpoint pipeline (`LearnerEvaluator`) on a
//! DEVELOPER-seeded grammar panel. The pipeline is bit-for-bit the one the hidden
//! evaluator uses; only the seed differs, and it is a seed we choose, not the secret.
//!
//! The question: is there a learner program the endpoint certifies as `Pass`? If a
//! hand-written learner passes here, the Stage-1 result of 0 out of 3 is a search
//! failure, not an impossible target: the yardstick admits a winner. If even the
//! strongest hand-written learner fails, the instrument has no headroom against its
//! own compute-matched baseline, and it needs a redesign before Stage 2.
//!
//! Secret safety. The developer secret file lives in a scratch directory the caller
//! passes on the command line, and it holds a developer seed we set in this file.
//! This program never reads `~/.brut/learner/hidden.env`, never writes there, and
//! never touches the real ledger. The developer seed below is plainly not a secret.
//!
//! Usage:
//!   cargo run --release --example positive_control -- <scratch_dir> [dev_seed ...]
//!
//! Extra candidate learners can be supplied as JSON through the environment variable
//! POSCTRL_CANDIDATES: a path to a file holding `[{"name": "...", "genome": {...}}, ...]`
//! where each genome is a `LearnerGenome` (fields `setup`, `predict`, `learn`). These
//! are checked and, when valid, evaluated alongside the four built-in seed learners.
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Deserialize;

use brutmaschine::learner::protocol::protocol_hash;
use brutmaschine::learner::seeds::seed_programs;
use brutmaschine::learner::{Decision, DslLimits, LearnerGenome, Tier};
use brutmaschine::learner_hidden::{
    BUDGET_PER_RUN_KEY, BUDGET_TOTAL_KEY, LEARNER_SEED_KEY, LearnerEvalRequest, LearnerEvaluator,
};
use brutmaschine::util::sha256_hex;

/// Developer seeds. Plainly not a secret: they are written in the source and in the
/// developer `hidden.env` the program creates. They only pick which 40 grammar
/// families the panel draws. The first is used when the caller names none.
const DEV_SEEDS: [u64; 3] = [
    0x00C0_FFEE_0000_0001,
    0x00C0_FFEE_0000_0002,
    0x00C0_FFEE_0000_0003,
];

/// A query budget large enough that no budget limit can shape the result.
const BIG_BUDGET: usize = 1_000_000;

/// The released score step: Δ is rounded down to this. 0.01 is the protocol value.
const SCORE_STEP: f64 = 0.01;

/// One extra candidate from the JSON file named by POSCTRL_CANDIDATES.
#[derive(Deserialize)]
struct NamedGenome {
    name: String,
    genome: LearnerGenome,
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();

    let Some(scratch) = raw.first().map(PathBuf::from) else {
        eprintln!("usage: positive_control <scratch_dir> [dev_seed ...]");
        std::process::exit(2);
    };

    let seeds: Vec<u64> = if raw.len() > 1 {
        raw[1..]
            .iter()
            .map(|s| s.parse().expect("dev seed must be a u64"))
            .collect()
    } else {
        vec![DEV_SEEDS[0]]
    };

    let mut candidates: Vec<(String, LearnerGenome)> = seed_programs()
        .into_iter()
        .map(|(n, g)| (n.to_string(), g))
        .collect();

    candidates.extend(load_extra_candidates());

    let lim = DslLimits::default();
    let ph = protocol_hash();

    println!("positive control: HIDDEN-GRAMMAR headroom check");
    println!("protocol hash: {ph}");
    println!("candidates:    {}", candidates.len());
    println!("dev seeds:     {seeds:?}");
    println!(
        "pass rule:     p < 0.05 and delta >= 0.01 and delta32 >= delta816 - 0.05 (released: delta down to {SCORE_STEP}, decision bit)\n"
    );

    let start = Instant::now();
    let mut any_pass = false;

    for &seed in &seeds {
        let dir = scratch.join(format!("dev-{seed:016x}"));
        write_dev_secret(&dir, seed);
        let secret = dir.join("hidden.env");

        let mut evaluator = match LearnerEvaluator::open(&secret, BIG_BUDGET, SCORE_STEP) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("open failed for dev seed {seed:#x}: {e}");
                std::process::exit(3);
            }
        };

        // One run id per dev directory; one candidate id per program.
        let run_id = sha256_hex(format!("positive-control/run/{seed}").as_bytes());

        println!("dev seed {seed:#018x}  (panel directory {})", dir.display());

        for (name, genome) in &candidates {
            // A program that fails the static check is reported, not scored: the real
            // evaluator would refuse it, and we want to see which hand-written designs
            // are structurally or cost-wise illegal rather than abort the whole run.
            if let Err(e) = genome.check(&lim) {
                println!("  {name:<16}  INVALID: {e}");
                continue;
            }

            let t0 = Instant::now();
            let Some(resp) = evaluate(&mut evaluator, &run_id, name, &ph, genome) else {
                continue;
            };
            let secs = t0.elapsed().as_secs_f64();

            let decision = match resp.decision {
                Some(Decision::Pass) => "PASS",
                Some(Decision::Fail) => "fail",
                None => "(no decision bit)",
            };

            if matches!(resp.decision, Some(Decision::Pass)) {
                any_pass = true;
            }

            println!(
                "  {name:<22}  cost16 {:>6}  delta {:>6.2}  {decision:<6}  ({secs:.1} s)",
                genome.step_cost(16),
                resp.delta,
            );
        }

        println!();
    }

    println!("total wall-clock: {:.1} s", start.elapsed().as_secs_f64());
    println!(
        "\nverdict: {}",
        if any_pass {
            "HEADROOM. At least one hand-written learner passes the endpoint on a developer panel."
        } else {
            "NO PASS. No hand-written learner cleared the endpoint on the developer panel(s); see deltas above."
        }
    );
}

/// Build and send one HIDDEN-GRAMMAR request, returning the released response, or None
/// when the evaluator refused it (the reason is printed).
fn evaluate(
    evaluator: &mut LearnerEvaluator,
    run_id: &str,
    name: &str,
    protocol_hash: &str,
    program: &LearnerGenome,
) -> Option<brutmaschine::learner_hidden::LearnerEvalResponse> {
    // A stable 16-hex candidate id per program name.
    let candidate_id = sha256_hex(name.as_bytes())[..16].to_string();

    let req = LearnerEvalRequest {
        run_id: run_id.to_string(),
        candidate_id,
        protocol_hash: protocol_hash.to_string(),
        tier: Tier::HiddenGrammar,
        program: program.clone(),
    };

    match evaluator.evaluate(&req) {
        Ok(resp) => Some(resp),
        Err(e) => {
            println!("  {name:<16}  REFUSED: {e}");
            None
        }
    }
}

/// Read extra candidates from the file named by POSCTRL_CANDIDATES, if set.
fn load_extra_candidates() -> Vec<(String, LearnerGenome)> {
    let Ok(path) = std::env::var("POSCTRL_CANDIDATES") else {
        return Vec::new();
    };

    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read candidate file {path}: {e}"));
    let extra: Vec<NamedGenome> =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse candidate file {path}: {e}"));

    extra.into_iter().map(|c| (c.name, c.genome)).collect()
}

/// Create the developer secret directory (0700) and `hidden.env` (0600) with a chosen
/// developer seed and generous budgets. Never the real secret path.
fn write_dev_secret(dir: &Path, seed: u64) {
    std::fs::create_dir_all(dir).expect("create dev secret dir");
    set_mode(dir, 0o700);

    let body = format!(
        "{LEARNER_SEED_KEY}={seed}\n{BUDGET_TOTAL_KEY}={BIG_BUDGET}\n{BUDGET_PER_RUN_KEY}={BIG_BUDGET}\n"
    );
    let file = dir.join("hidden.env");

    std::fs::write(&file, body).expect("write dev hidden.env");
    set_mode(&file, 0o600);
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;

    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).expect("set mode");
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _mode: u32) {}
