//! vote-control on a run that brut-breed-ollama itself wrote, with the simulated models, so
//! no server is needed: the plan, the result file, the partial smoke test, a repeat, and the
//! run's own record left as it was.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use backend_ollama::vote::{OUTPUT, Plan, VoteReport};

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vote-control-it-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    dir
}

fn run(bin: &str, args: &[&str]) -> Output {
    Command::new(bin).args(args).output().expect("start")
}

fn vote(dir: &Path, extra: &[&str]) -> Output {
    let mut args = vec!["--run", dir.to_str().expect("utf-8 path")];

    args.extend_from_slice(extra);
    run(env!("CARGO_BIN_EXE_vote-control"), &args)
}

fn stdout(o: &Output) -> &str {
    std::str::from_utf8(&o.stdout).expect("utf-8")
}

#[test]
fn vote_control_on_a_simulated_brut_breed_run() {
    let dir = temp_dir("sim");
    let path = dir.to_str().expect("utf-8 path");
    let bred = run(
        env!("CARGO_BIN_EXE_brut-breed-ollama"),
        &["--out", path, "--seed", "4", "--generations", "1", "--children", "3", "--search", "6", "--select", "5"],
    );

    assert!(bred.status.success(), "{}", String::from_utf8_lossy(&bred.stderr));

    // The plan makes no call and writes nothing.
    let dry = vote(&dir, &["--dry-run"]);
    assert!(dry.status.success(), "{}", String::from_utf8_lossy(&dry.stderr));
    let plan: Plan = serde_json::from_str(stdout(&dry)).expect("plan json");
    assert!(plan.cascade.is_seed && plan.champion.is_some() && plan.finished);
    assert_eq!((plan.select_puzzles, plan.partial, plan.samples_from.as_str()), (5, false, "champion_calls"));
    assert!(!dir.join(OUTPUT).exists());

    // The control writes what it prints.
    let full = vote(&dir, &["--samples", "3", "--jobs", "2"]);
    assert!(full.status.success(), "{}", String::from_utf8_lossy(&full.stderr));
    let report: VoteReport = serde_json::from_str(stdout(&full)).expect("report json");
    let file = std::fs::read_to_string(dir.join(OUTPUT)).expect("result file");
    assert_eq!(serde_json::from_str::<VoteReport>(&file).expect("file json"), report);
    assert_eq!((report.plan.samples_per_puzzle, report.totals.puzzles), (3, 5));
    assert_eq!(report.calls_used, 15, "a seed is one Worker: one call per sample");
    assert_eq!((report.new_calls, report.failed_calls), (None, 0), "simulated models are not recorded");
    assert!(report.totals.champion_correct.is_some() && report.totals.single_correct.is_some());
    assert!(report.puzzles.iter().all(|p| p.votes.iter().map(|v| v.count).sum::<usize>() + p.abstained == 3));

    // A repeat gives the same result.
    let again = vote(&dir, &["--samples", "3", "--jobs", "1"]);
    assert!(again.status.success());
    assert_eq!(serde_json::from_str::<VoteReport>(stdout(&again)).expect("json"), report);

    // A smoke test on two puzzles prints a partial result and leaves the file alone.
    let smoke = vote(&dir, &["--samples", "2", "--puzzles", "2"]);
    assert!(smoke.status.success());
    let partial: VoteReport = serde_json::from_str(stdout(&smoke)).expect("json");
    assert!(partial.plan.partial && partial.puzzles.len() == 2);
    assert_eq!(std::fs::read_to_string(dir.join(OUTPUT)).expect("file"), file);

    // The run itself still verifies: the control wrote nothing into its record.
    let verify = run(env!("CARGO_BIN_EXE_brut-breed-ollama"), &["--out", path, "--verify"]);
    assert!(verify.status.success(), "{}", String::from_utf8_lossy(&verify.stderr));

    // Wrong usage exits with 2: unknown options, zero samples, an unfinished run.
    assert_eq!(vote(&dir, &["--bogus"]).status.code(), Some(2));
    assert_eq!(vote(&dir, &["--jobs", "0"]).status.code(), Some(2));
    assert_eq!(vote(&dir, &["--samples", "0"]).status.code(), Some(2));
    assert_eq!(vote(&dir, &["--cascade", "nobody"]).status.code(), Some(2));

    let snap = std::fs::read_to_string(dir.join("snapshot.json")).expect("snapshot");
    let mut cp: serde_json::Value = serde_json::from_str(&snap).expect("json");
    cp["snapshot"]["finished"] = serde_json::Value::Bool(false);
    std::fs::write(dir.join("snapshot.json"), cp.to_string()).expect("write");
    assert_eq!(vote(&dir, &[]).status.code(), Some(2));
    assert!(vote(&dir, &["--dry-run"]).status.success(), "a plan works on an unfinished run");

    let _ = std::fs::remove_dir_all(&dir);
}
