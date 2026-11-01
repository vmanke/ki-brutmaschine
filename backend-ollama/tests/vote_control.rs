//! vote-control on a run that brut-breed-ollama itself wrote, with the simulated models, so
//! no server is needed: the plan, the result file, the partial smoke test, a repeat, and the
//! run's own record left as it was. Then a vLLM run against a dead loopback port: a run
//! whose own calls failed is refused, and a control whose calls fail never writes the
//! result of record.

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use backend_ollama::vote::{INCOMPLETE, OUTPUT, Plan, RUN_ERRORS, RUN_RECORD, VoteReport};

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vote-control-it-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    dir
}

fn run(bin: &str, args: &[&str]) -> Output {
    Command::new(bin).args(args).output().expect("start")
}

/// `run` with the vLLM server at `vllm`.
fn run_on(vllm: &str, bin: &str, args: &[&str]) -> Output {
    Command::new(bin).args(args).env("VLLM_URL", vllm).output().expect("start")
}

fn vote(dir: &Path, extra: &[&str]) -> Output {
    let mut args = vec!["--run", dir.to_str().expect("utf-8 path")];

    args.extend_from_slice(extra);
    run(env!("CARGO_BIN_EXE_vote-control"), &args)
}

fn vote_on(vllm: &str, dir: &Path, extra: &[&str]) -> Output {
    let mut args = vec!["--run", dir.to_str().expect("utf-8 path")];

    args.extend_from_slice(extra);
    run_on(vllm, env!("CARGO_BIN_EXE_vote-control"), &args)
}

/// A loopback URL nothing listens on: every call is refused at once.
fn dead_server() -> String {
    let port = TcpListener::bind("127.0.0.1:0").expect("bind").local_addr().expect("addr").port();

    format!("http://127.0.0.1:{port}")
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
    assert_eq!((plan.run_failed_calls, plan.predates_seed_fix), (0, false), "a healthy run");
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
    assert!(report.complete);
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
    assert!(partial.plan.partial && partial.puzzles.len() == 2 && !partial.complete);
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

#[test]
fn failed_calls_never_make_the_result_of_record() {
    let dir = temp_dir("dead");
    let path = dir.to_str().expect("utf-8 path");
    let dead = dead_server();
    let cfg = dir.with_extension("config.json");

    // A vLLM run whose every call failed, as against a dead server or a model that is not
    // served: brut-breed-ollama files each failure in llm/worker-errors.jsonl.
    std::fs::write(
        &cfg,
        r#"{"seed": 4, "generations": 1, "children_per_generation": 3, "search_puzzles": 6,
            "select_puzzles": 5, "crossover_rate": 0.5, "peer_opponents": 3,
            "peer_puzzles_per_pair": 2, "judge_weight": 0.2, "token_cost_per_1k": 0.01,
            "dgm_lambda": 10.0, "dgm_alpha0": 0.5, "llm_every": 0, "llm_children": 2,
            "pause_for_llm": true, "max_calls": 5000, "finalists": 2,
            "limits": {"models": [{"tag": "vllm:qwen2.5-32b", "cost_per_call": 0.002}],
                       "max_stages": 4, "max_width": 3, "think_ceiling_per_stage": []}}"#,
    )
    .expect("config");

    let bred = run_on(
        &dead,
        env!("CARGO_BIN_EXE_brut-breed-ollama"),
        &["--out", path, "--config", cfg.to_str().expect("utf-8 path")],
    );
    assert!(bred.status.success(), "{}", String::from_utf8_lossy(&bred.stderr));
    let failed = std::fs::read_to_string(dir.join(RUN_ERRORS)).expect("errors").lines().count() as u64;
    assert!(failed > 0);

    // Its own select answers are no fair baseline: refused, the plan too, unless allowed.
    let refused = vote_on(&dead, &dir, &["--dry-run"]);
    assert_eq!(refused.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&refused.stderr).contains("--allow-degraded-run"));
    assert_eq!(vote_on(&dead, &dir, &["--samples", "2"]).status.code(), Some(2));

    let allowed = vote_on(&dead, &dir, &["--dry-run", "--allow-degraded-run"]);
    assert!(allowed.status.success(), "{}", String::from_utf8_lossy(&allowed.stderr));
    let plan: Plan = serde_json::from_str(stdout(&allowed)).expect("plan json");
    assert_eq!((plan.run_failed_calls, plan.predates_seed_fix), (failed, false));

    // A control whose calls fail exits 1 and leaves an earlier result of record alone; its
    // incomplete result goes to vote-control/incomplete.json.
    let earlier = "{\"earlier\": \"complete result\"}\n";
    std::fs::write(dir.join(OUTPUT), earlier).expect("write");

    let out = vote_on(&dead, &dir, &["--samples", "2", "--allow-degraded-run"]);
    assert_eq!(out.status.code(), Some(1), "{}", String::from_utf8_lossy(&out.stderr));
    let report: VoteReport = serde_json::from_str(stdout(&out)).expect("report json");
    assert_eq!((report.failed_calls, report.complete), (10, false));
    assert_eq!(report.plan.run_failed_calls, failed);
    assert_eq!(std::fs::read_to_string(dir.join(OUTPUT)).expect("file"), earlier);
    let kept: VoteReport =
        serde_json::from_str(&std::fs::read_to_string(dir.join(INCOMPLETE)).expect("incomplete")).expect("json");
    assert_eq!(kept, report);

    // Without a result of record, none appears either.
    std::fs::remove_file(dir.join(OUTPUT)).expect("rm");
    assert_eq!(vote_on(&dead, &dir, &["--samples", "3", "--allow-degraded-run"]).status.code(), Some(1));
    assert!(!dir.join(OUTPUT).exists());

    // A vLLM record of 40 calls without any seed above i64::MAX marks a run from before the
    // 63-bit seed fix, which lost every call with a higher seed and filed none of them.
    std::fs::remove_file(dir.join(RUN_ERRORS)).expect("rm");
    let low: String = (0..40)
        .map(|i| format!("{{\"key\":\"k{i}\",\"tag\":\"vllm:qwen2.5-32b\",\"request\":{{\"seed\":{i}}}}}\n"))
        .collect();
    std::fs::write(dir.join(RUN_RECORD), low).expect("write");
    let old = vote_on(&dead, &dir, &["--dry-run"]);
    assert_eq!(old.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&old.stderr).contains("d79c93f"));
    let plan: Plan = serde_json::from_str(stdout(&vote_on(&dead, &dir, &["--dry-run", "--allow-degraded-run"])))
        .expect("plan json");
    assert_eq!((plan.run_failed_calls, plan.predates_seed_fix), (0, true));

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&cfg);
}
