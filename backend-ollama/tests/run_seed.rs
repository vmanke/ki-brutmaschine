//! The driver of one seed, docs/prereg-breeding-vs-vote/run-seed.sh, on stand-in programs:
//! the order of its steps, the Mutator loop, the call matching and the two kept votes, a
//! second call after a clean finish, and every stop of section 8 (a killed or stopped
//! run, failed calls, a failed Mutator round, a failed vote, wrong usage). It never
//! deletes anything. A last test checks that brut-breed-ollama takes both frozen settings
//! files and writes them back as its config.json, which the analysis compares with them.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn prereg() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("docs")
        .join("prereg-breeding-vs-vote")
}

fn fresh(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("run-seed")
        .join(name);

    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("clean");
    }

    std::fs::create_dir_all(&dir).expect("mkdir");
    dir
}

/// Stand-ins for the four programs. Each logs its arguments. brut-breed-ollama ends each
/// run of arm X with the next code of FAKE_CODES_X ("0" writes report.json, "fail" files a
/// failed call and pauses, "stopped" ends without a report); its --verify passes.
/// brut-mutator and vote-control end with the next code of FAKE_MUTATOR and FAKE_VOTE;
/// vote-control writes vote-control.json naming its samples. breeding-vs-vote answers
/// --match with the next line of FAKE_MATCH (lines split by "|").
fn stand_ins(dir: &Path) -> PathBuf {
    let bin = dir.join("bin");
    let next = r#"next() { # next <counter> <list> <separator>
  n=$(cat "$FAKE_STATE/$1" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "$FAKE_STATE/$1"
  v=$(printf '%s' "$2" | awk -F"$3" -v n="$n" '{ if (n <= NF) print $n; else print $NF }')
}
"#;
    let breed = format!(
        r#"#!/bin/sh
{next}echo "breed $*" >> "$FAKE_LOG"
out=""; verify=0; prev=""
for a in "$@"; do [ "$prev" = "--out" ] && out="$a"; [ "$a" = "--verify" ] && verify=1; prev="$a"; done
if [ "$verify" = 1 ]; then echo "brut-breed: 9 events, hash chain intact; settings match; report recomputed and equal"; exit 0; fi
arm=$(basename "$out" | cut -c1)
eval codes=\$FAKE_CODES_$arm
next "breed-$arm" "$codes" " "
case "$v" in
  0) echo "{{}}" > "$out/report.json"; exit 0 ;;
  fail) mkdir -p "$out/llm"; echo '{{"key":"a"}}' > "$out/llm/worker-errors.jsonl"; exit 3 ;;
  stopped) exit 0 ;;
  *) exit "$v" ;;
esac
"#
    );
    let mutator = format!(
        r#"#!/bin/sh
{next}echo "mutator $*" >> "$FAKE_LOG"
next mutator "${{FAKE_MUTATOR:-0}}" " "
exit "$v"
"#
    );
    let vote = format!(
        r#"#!/bin/sh
{next}echo "vote $*" >> "$FAKE_LOG"
run=""; samples=""; prev=""
for a in "$@"; do [ "$prev" = "--run" ] && run="$a"; [ "$prev" = "--samples" ] && samples="$a"; prev="$a"; done
next vote "${{FAKE_VOTE:-0}}" " "
[ "$v" = 0 ] && echo "{{\"samples_per_puzzle\": $samples}}" > "$run/vote-control.json"
echo "{{}}"
exit "$v"
"#
    );
    let analysis = format!(
        r#"#!/bin/sh
{next}echo "analysis $*" >> "$FAKE_LOG"
case "$1" in
  --match) next match "$FAKE_MATCH" "|"; echo "$v" ;;
  --panels) touch "$2/panels.json" ;;
  --runs) echo "the analysis of $2" ;;
esac
"#
    );

    std::fs::create_dir_all(&bin).expect("mkdir");

    for (name, body) in [
        ("brut-breed-ollama", breed),
        ("brut-mutator", mutator),
        ("vote-control", vote),
        ("breeding-vs-vote", analysis),
    ] {
        let path = bin.join(name);

        std::fs::write(&path, body).expect("write");
        Command::new("chmod")
            .arg("+x")
            .arg(&path)
            .status()
            .expect("chmod");
    }

    bin
}

/// A test directory: stand-ins, their state, and the runs directory.
struct Bench {
    dir: PathBuf,
    runs: PathBuf,
}

impl Bench {
    fn new(name: &str) -> Bench {
        let dir = fresh(name);

        stand_ins(&dir);
        std::fs::create_dir_all(dir.join("state")).expect("state");

        Bench {
            runs: dir.join("runs"),
            dir,
        }
    }

    /// Run the driver with `args` and `env`; its exit code, its output, and the calls it
    /// made since the last call, with the paths written as RUNS and FROZEN.
    fn drive(&self, args: &[&str], env: &[(&str, &str)]) -> (Option<i32>, String, Vec<String>) {
        let log = self.dir.join("log");
        let before = std::fs::read_to_string(&log)
            .unwrap_or_default()
            .lines()
            .count();
        let mut cmd = Command::new("bash");

        cmd.arg(prereg().join("run-seed.sh"))
            .args(args)
            .args(["--runs", self.runs.to_str().expect("path")])
            .env("BRUT_BIN_DIR", self.dir.join("bin"))
            .env("PREREG_RETRY_SECONDS", "0")
            .env("FAKE_LOG", &log)
            .env("FAKE_STATE", self.dir.join("state"))
            .env("FAKE_CODES_B", "0")
            .env("FAKE_CODES_L", "0")
            .env("FAKE_MATCH", "matched 2 1")
            .env_remove("BRUT_CALL_CACHE")
            .env_remove("PREREG_FROZEN");

        for (k, v) in env {
            cmd.env(k, v);
        }

        let o: Output = cmd.output().expect("run the driver");
        let frozen = prereg().canonicalize().expect("frozen dir");
        let calls = std::fs::read_to_string(&log)
            .unwrap_or_default()
            .lines()
            .skip(before)
            .map(|l| {
                l.replace(self.runs.to_str().expect("path"), "RUNS")
                    .replace(frozen.to_str().expect("path"), "FROZEN")
            })
            .collect();
        let text = format!(
            "exit {:?}\nstdout:\n{}\nstderr:\n{}",
            o.status.code(),
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        );

        (o.status.code(), text, calls)
    }

    fn read(&self, rel: &str) -> String {
        std::fs::read_to_string(self.runs.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
    }
}

#[test]
fn a_seed_breeds_both_arms_through_mutator_rounds_and_keeps_both_votes() {
    let b = Bench::new("full");
    let env = [
        ("FAKE_CODES_B", "3 3 0"),
        ("FAKE_CODES_L", "3 0"),
        ("FAKE_MATCH", "samples 4|matched 4 3"),
    ];
    let (code, out, calls) = b.drive(&["--seed", "7"], &env);

    assert_eq!(code, Some(0), "{out}");
    assert_eq!(
        calls,
        vec![
            "breed --out RUNS/B-7 --config FROZEN/config-B.json --seed 7",
            "mutator --out RUNS/B-7 --tag vllm:qwen2.5-32b --parallel 8",
            "breed --out RUNS/B-7 --resume",
            "mutator --out RUNS/B-7 --tag vllm:qwen2.5-32b --parallel 8",
            "breed --out RUNS/B-7 --resume",
            "breed --out RUNS/B-7 --verify",
            "analysis --match RUNS/B-7 --frozen FROZEN",
            "vote --run RUNS/B-7 --temperature 0.7 --samples 4",
            "analysis --match RUNS/B-7 --frozen FROZEN",
            "vote --run RUNS/B-7 --temperature 0.7 --samples 3",
            "vote --run RUNS/B-7 --temperature 0.7 --samples 4",
            "analysis --match RUNS/B-7 --frozen FROZEN",
            "breed --out RUNS/L-7 --config FROZEN/config-L.json --seed 7 --selection lexicase",
            "mutator --out RUNS/L-7 --tag vllm:qwen2.5-32b --parallel 8",
            "breed --out RUNS/L-7 --resume",
            "breed --out RUNS/L-7 --verify",
            "analysis --panels RUNS/B-7 --frozen FROZEN",
            "analysis --runs RUNS --frozen FROZEN",
        ],
        "{out}"
    );

    // Each kept vote is the vote-control.json of its own N.
    assert_eq!(
        b.read("B-7/vote-down.json"),
        "{\"samples_per_puzzle\": 3}\n"
    );
    assert_eq!(b.read("B-7/vote-up.json"), "{\"samples_per_puzzle\": 4}\n");
    assert!(b.read("B-7/driver.log").contains("breed exit 3"));
    assert!(
        b.read("B-7/driver.log").contains(
            "kept RUNS/B-7/vote-up.json with 4 samples"
                .replace("RUNS", b.runs.to_str().expect("path"))
                .as_str()
        )
    );

    // A second call after the clean finish breeds and votes nothing again: it checks.
    let (code, out, calls) = b.drive(&["--seed", "7"], &[("FAKE_MATCH", "matched 4 3")]);

    assert_eq!(code, Some(0), "{out}");
    assert_eq!(
        calls,
        vec![
            "breed --out RUNS/B-7 --verify",
            "analysis --match RUNS/B-7 --frozen FROZEN",
            "analysis --match RUNS/B-7 --frozen FROZEN",
            "breed --out RUNS/L-7 --verify",
            "analysis --panels RUNS/B-7 --frozen FROZEN",
            "analysis --runs RUNS --frozen FROZEN",
        ],
        "{out}"
    );
    assert_eq!(b.read("B-7/vote-up.json"), "{\"samples_per_puzzle\": 4}\n");
}

#[test]
fn without_n_down_no_down_vote_is_kept_and_pilot_seeds_get_the_pilot_analysis() {
    let b = Bench::new("pilot");
    let (code, out, calls) = b.drive(&["--seed", "903"], &[("FAKE_MATCH", "matched 1 -")]);

    assert_eq!(code, Some(0), "{out}");
    assert_eq!(
        calls.iter().filter(|c| c.starts_with("vote ")).count(),
        1,
        "{out}"
    );
    assert!(!b.runs.join("B-903/vote-down.json").exists());
    assert_eq!(
        calls.last().map(String::as_str),
        Some("analysis --runs RUNS --frozen FROZEN --pilot")
    );
}

#[test]
fn an_interrupted_run_is_never_resumed_and_never_deleted() {
    let b = Bench::new("interrupted");
    let dir = b.runs.join("B-3");

    // A process that was killed: its last journal line is its start.
    std::fs::create_dir_all(&dir).expect("mkdir");
    std::fs::write(
        dir.join("driver.log"),
        "1 breed start --out x --config y --seed 3\n",
    )
    .expect("write");

    let (code, out, calls) = b.drive(&["--seed", "3"], &[]);

    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("is not resumed. Delete it (rm -r"), "{out}");
    assert!(calls.is_empty(), "{calls:?}");
    assert!(dir.join("driver.log").exists());

    // A run that ended without a report, as a STOP file or a time limit ends it.
    let b = Bench::new("stopped");
    let (code, out, calls) = b.drive(&["--seed", "3"], &[("FAKE_CODES_B", "stopped")]);

    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("stopped without finishing"), "{out}");
    assert_eq!(calls.len(), 1);

    // A second call sees the same and stops again, before any program runs.
    let (code, out, calls) = b.drive(&["--seed", "3"], &[]);

    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("did not pause for Mutator answers"), "{out}");
    assert!(calls.is_empty());

    // An error of the run.
    let b = Bench::new("error");
    let (code, out, _) = b.drive(&["--seed", "3"], &[("FAKE_CODES_B", "1")]);

    assert_eq!(code, Some(1), "{out}");
    assert!(
        out.contains("ended") && out.contains("with exit 1"),
        "{out}"
    );
}

#[test]
fn a_run_with_failed_calls_stops_the_seed_at_once() {
    let b = Bench::new("failed");
    let (code, out, calls) = b.drive(&["--seed", "4"], &[("FAKE_CODES_B", "fail 0")]);

    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("Worker calls failed"), "{out}");
    assert_eq!(
        calls.len(),
        1,
        "no Mutator round after a failed call: {calls:?}"
    );
}

#[test]
fn a_paused_run_goes_on_in_a_second_call_after_failed_mutator_rounds() {
    let b = Bench::new("paused");
    let env = [("FAKE_CODES_B", "3 0"), ("FAKE_MUTATOR", "1 1 1 0")];
    let (code, out, calls) = b.drive(&["--seed", "5"], &env);

    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("waits cleanly for Mutator answers"), "{out}");
    assert_eq!(calls.iter().filter(|c| c.starts_with("mutator")).count(), 3);

    let (code, out, calls) = b.drive(&["--seed", "5"], &env);

    assert_eq!(code, Some(0), "{out}");
    assert_eq!(
        calls[0],
        "mutator --out RUNS/B-5 --tag vllm:qwen2.5-32b --parallel 8"
    );
    assert_eq!(calls[1], "breed --out RUNS/B-5 --resume");
    assert!(b.runs.join("B-5/report.json").exists());

    // Nothing to answer right after a pause of this call is an error.
    let b = Bench::new("nothing");
    let (code, out, _) = b.drive(
        &["--seed", "5"],
        &[("FAKE_CODES_B", "3 0"), ("FAKE_MUTATOR", "4")],
    );

    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("found nothing to answer"), "{out}");
}

#[test]
fn a_vote_that_keeps_failing_stops_and_a_repeat_goes_on() {
    let b = Bench::new("vote");
    let env = [
        ("FAKE_VOTE", "1 1 1 0"),
        ("FAKE_MATCH", "samples 2|matched 2 1"),
    ];
    let (code, out, calls) = b.drive(&["--seed", "6"], &env);

    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("vote-control failed 3 times"), "{out}");
    assert_eq!(calls.iter().filter(|c| c.starts_with("vote ")).count(), 3);

    let (code, out, _) = b.drive(&["--seed", "6"], &env);

    assert_eq!(code, Some(0), "{out}");
    assert!(b.runs.join("B-6/vote-up.json").exists() && b.runs.join("B-6/vote-down.json").exists());
}

#[test]
fn wrong_usage_and_an_unshared_cache_are_refused_before_any_program() {
    let b = Bench::new("usage");

    for args in [
        vec!["--seed", "61"],
        vec!["--seed", "0"],
        vec!["--seed", "x"],
        vec![],
        vec!["--bogus"],
    ] {
        let (code, out, calls) = b.drive(&args, &[]);

        assert_eq!(code, Some(2), "{args:?}: {out}");
        assert!(calls.is_empty());
    }

    let (code, out, calls) = b.drive(&["--seed", "1"], &[("BRUT_CALL_CACHE", "off")]);

    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("call cache"), "{out}");
    assert!(calls.is_empty());
}

/// brut-breed-ollama takes both frozen settings files, as run-seed.sh gives them, and
/// writes them back unchanged but for the seed. --max-seconds 0 stops the run before its
/// first call, so no model server is needed.
#[test]
fn brut_breed_ollama_takes_both_frozen_settings_files() {
    for (arm, extra) in [("B", vec![]), ("L", vec!["--selection", "lexicase"])] {
        let dir = fresh(&format!("frozen-{arm}"));
        let out = dir.join("run");
        let config = prereg().join(format!("config-{arm}.json"));
        let o = Command::new(env!("CARGO_BIN_EXE_brut-breed-ollama"))
            .args([
                "--out",
                out.to_str().expect("path"),
                "--config",
                config.to_str().expect("path"),
            ])
            .args(["--seed", "17", "--max-seconds", "0"])
            .args(&extra)
            .env("BRUT_CALL_CACHE", "off")
            .output()
            .expect("run");

        assert!(
            o.status.success(),
            "{arm}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        assert!(String::from_utf8_lossy(&o.stdout).contains("stopped at the time limit"));
        assert!(
            !out.join("llm").join("worker-calls.jsonl").exists(),
            "no call was made"
        );

        let written: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(out.join("config.json")).expect("config"),
        )
        .expect("json");
        let mut frozen: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config).expect("frozen")).expect("json");

        frozen["seed"] = serde_json::json!(17);
        assert_eq!(written, frozen, "{arm}");
    }
}

/// A vLLM stand-in on 127.0.0.1 for the end-to-end run. It knows the answers `truth` of
/// the seed's puzzles, by their rendered test input. A Mutator request gets a Cascade of
/// two Workers in one stage. A solve prompt gets the right answer, the test input
/// mirrored, or nothing usable, by the request's seed; a judge prompt gets the right
/// verdict three times in four. Equal requests get equal answers.
fn fake_vllm(truth: std::sync::Arc<std::collections::HashMap<String, Vec<Vec<u8>>>>) -> String {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    use brutmaschine::cascade::puzzle::{parse_prompt, render_answer, render_grid};

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let base = format!("http://{}", listener.local_addr().expect("addr"));

    std::thread::spawn(move || {
        for sock in listener.incoming() {
            let Ok(mut sock) = sock else { continue };
            let truth = truth.clone();

            std::thread::spawn(move || {
                let mut buf = Vec::new();
                let mut chunk = [0u8; 8192];
                let body = loop {
                    let Ok(n) = sock.read(&mut chunk) else { return };

                    buf.extend_from_slice(&chunk[..n]);

                    if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..end]).to_ascii_lowercase();
                        let len = head
                            .lines()
                            .find_map(|l| {
                                l.strip_prefix("content-length:")
                                    .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                            })
                            .unwrap_or(0);

                        if buf.len() >= end + 4 + len {
                            break String::from_utf8_lossy(&buf[end + 4..end + 4 + len])
                                .to_string();
                        }
                    }

                    if n == 0 {
                        return;
                    }
                };
                let req: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
                let seed = req["seed"].as_u64().unwrap_or(0);
                let system = req["messages"][0]["content"].as_str().unwrap_or_default();
                let prompt = req["messages"][1]["content"].as_str().unwrap_or_default();
                let content = if system.contains("mutation operator") {
                    let worker = |strategy: &str, t: f32| {
                        serde_json::json!({
                            "model": "vllm:qwen2.5-32b",
                            "sections": [
                                {"name": "role", "text": "You solve grid puzzles."},
                                {"name": "strategy", "text": strategy},
                                {"name": "format", "text": "Answer with the line ANSWER followed by the output grid."},
                            ],
                            "traits": {"think_tokens": 0, "temperature": t, "answer_tokens": 512, "self_check": false},
                        })
                    };
                    let child = serde_json::json!({"stages": [{"workers": [
                        worker(&format!("Look for mirror images, variant {}.", seed % 997), 0.2),
                        worker(&format!("Look for rotations, variant {}.", seed % 991), 0.5),
                    ]}]});

                    format!("Here is the child:\n{child}\n")
                } else {
                    match parse_prompt(prompt) {
                        Some(p) if p.judge => {
                            let right = truth.get(&render_grid(&p.test_input));
                            let a = match p.candidates.as_ref().map(|(a, _)| Some(a) == right) {
                                Some(a_right) if !seed.is_multiple_of(4) => a_right,
                                _ => seed.is_multiple_of(2),
                            };

                            if a { "VERDICT A" } else { "VERDICT B" }.to_string()
                        }
                        Some(p) => match (seed % 3, truth.get(&render_grid(&p.test_input))) {
                            (0, Some(t)) => render_answer(t),
                            (1, _) => render_answer(
                                &p.test_input
                                    .iter()
                                    .map(|r| r.iter().rev().copied().collect())
                                    .collect(),
                            ),
                            _ => "no idea".to_string(),
                        },
                        None => "no idea".to_string(),
                    }
                };
                let reply = serde_json::json!({
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 300, "completion_tokens": 30},
                })
                .to_string();
                let _ = write!(
                    sock,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                );
            });
        }
    });

    base
}

/// The whole seed with the real programs against the vLLM stand-in: both arms through
/// their Mutator rounds, --verify, the call matching, both votes, panels.json and the
/// analysis. The recount, when python3 exists, gives the same result.
#[test]
fn a_whole_seed_runs_with_the_real_programs() {
    let dir = fresh("end-to-end");
    let frozen = dir.join("frozen");
    let runs = dir.join("runs");
    let bin = Path::new(env!("CARGO_BIN_EXE_breeding-vs-vote"))
        .parent()
        .expect("bin dir");
    let truth = {
        use brutmaschine::cascade::domain::{Domain, GridDomain};
        use brutmaschine::cascade::puzzle::render_grid;
        use brutmaschine::util::derive_seed;

        let search = GridDomain.panel(derive_seed(1, "panel/search"), 5);
        let select = GridDomain.panel(derive_seed(1, "panel/select"), 7);

        search
            .iter()
            .chain(&select)
            .map(|p| (render_grid(&p.test_input), p.test_output.clone()))
            .collect()
    };
    let base = fake_vllm(std::sync::Arc::new(truth));

    std::fs::create_dir_all(&frozen).expect("mkdir");

    for (arm, selection) in [("B", ""), ("L", ",\n  \"selection\": \"lexicase\"")] {
        let cfg = format!(
            r#"{{
  "seed": 1, "generations": 4, "children_per_generation": 3, "search_puzzles": 5,
  "select_puzzles": 7, "crossover_rate": 0.5, "peer_opponents": 3, "peer_puzzles_per_pair": 2,
  "judge_weight": 0.2, "token_cost_per_1k": 0.01, "dgm_lambda": 10.0, "dgm_alpha0": 0.5,
  "llm_every": 2, "llm_children": 2, "pause_for_llm": true, "max_calls": 50000, "finalists": 2,
  "limits": {{"models": [{{"tag": "vllm:qwen2.5-32b", "cost_per_call": 0.002}}],
             "max_stages": 4, "max_width": 3, "think_ceiling_per_stage": []}}{selection}
}}
"#
        );

        std::fs::write(frozen.join(format!("config-{arm}.json")), cfg).expect("config");
    }

    let o = Command::new("bash")
        .arg(prereg().join("run-seed.sh"))
        .args(["--seed", "1", "--runs", runs.to_str().expect("path")])
        .env("BRUT_BIN_DIR", bin)
        .env("PREREG_FROZEN", &frozen)
        .env("VLLM_URL", &base)
        .env("OLLAMA_TIMEOUT_SECS", "30")
        .output()
        .expect("run the driver");
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );

    assert_eq!(o.status.code(), Some(0), "{text}");

    for f in [
        "B-1/report.json",
        "L-1/report.json",
        "B-1/vote-up.json",
        "B-1/panels.json",
        "analysis.json",
    ] {
        assert!(runs.join(f).exists(), "{f}: {text}");
    }

    // Both arms went through Mutator rounds, and arm L took generation 0 from the cache.
    for arm in ["B-1", "L-1"] {
        let journal = std::fs::read_to_string(runs.join(arm).join("driver.log")).expect("journal");

        assert_eq!(
            journal.matches("breed exit 3").count(),
            2,
            "{arm}: {journal}"
        );
    }

    assert!(
        std::fs::read_to_string(runs.join("L-1/llm/worker-calls.jsonl"))
            .expect("record")
            .contains("\"cached\":true")
    );
    assert!(
        runs.join("call-cache/seed-1.jsonl").exists(),
        "the seed's own call cache"
    );

    let analysis: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(runs.join("analysis.json")).expect("analysis"),
    )
    .expect("json");

    assert_eq!(
        analysis["pairs"].as_array().map(Vec::len),
        Some(1),
        "{text}"
    );
    assert_eq!(analysis["next_seed"], 2);

    // A second call changes nothing but the journals and the analysis files.
    let up = std::fs::read_to_string(runs.join("B-1/vote-up.json")).expect("up");
    let again = Command::new("bash")
        .arg(prereg().join("run-seed.sh"))
        .args(["--seed", "1", "--runs", runs.to_str().expect("path")])
        .env("BRUT_BIN_DIR", bin)
        .env("PREREG_FROZEN", &frozen)
        .env("VLLM_URL", &base)
        .output()
        .expect("run the driver");

    assert_eq!(
        again.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&again.stderr)
    );
    assert_eq!(
        std::fs::read_to_string(runs.join("B-1/vote-up.json")).expect("up"),
        up
    );

    if Command::new("python3").arg("--version").output().is_err() {
        eprintln!("skipped the recount: python3 is missing");
        return;
    }

    let out = dir.join("recount.json");
    let py = Command::new("python3")
        .arg("-I")
        .arg(prereg().join("recount.py"))
        .args([
            "--runs",
            runs.to_str().expect("path"),
            "--frozen",
            frozen.to_str().expect("path"),
        ])
        .args(["--json", out.to_str().expect("path")])
        .output()
        .expect("python3");

    assert!(
        py.status.success(),
        "{}",
        String::from_utf8_lossy(&py.stderr)
    );

    let recount: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out).expect("recount")).expect("json");

    for key in [
        "seed",
        "c",
        "v",
        "v_down",
        "k_c",
        "n_up",
        "n_down",
        "k_v_up",
        "k_v_down",
        "seed_plain",
        "c_l",
        "y",
        "y1",
        "y2",
    ] {
        assert_eq!(recount["pairs"][0][key], analysis["pairs"][0][key], "{key}");
    }

    assert_eq!(
        recount["pairs"][0]["e_b"],
        analysis["pairs"][0]["e_b"]["fraction"]
    );
    assert!(
        analysis["pairs"][0]["c"].as_u64() > Some(0),
        "the stand-in solves some puzzles"
    );
    assert_eq!(recount["primary"], analysis["primary"]);
    assert_eq!(recount["next_seed"], analysis["next_seed"]);
}
