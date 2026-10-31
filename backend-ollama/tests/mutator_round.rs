//! End-to-end tests of model-served Mutator rounds: a simulated breeding run pauses for
//! LLM answers, `brut-mutator` answers them through a fake vLLM server on loopback, and
//! the run resumes and uses them; the driver `run-with-mutator.sh` does the same loop on
//! its own. No live model server is needed.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Arc, Mutex};
use std::thread;

use backend_ollama::{CallRecord, wire_seed};
use brutmaschine::backend::GenRequest;
use brutmaschine::cascade::arena::request_key;

const BREED: &str = env!("CARGO_BIN_EXE_brut-breed-ollama");
const MUTATOR: &str = env!("CARGO_BIN_EXE_brut-mutator");
const TAG: &str = "vllm:qwen2.5-32b";

/// A small simulated run in which generations 1 and 2 are LLM rounds of two requests.
const SMALL: [&str; 14] = [
    "--seed", "5", "--generations", "2", "--children", "4", "--search", "12", "--select", "12",
    "--llm-every", "1", "--llm-children", "2",
];

fn driver() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("run-with-mutator.sh")
}

fn fresh(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("mutator").join(name);

    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("clean");
    }

    std::fs::create_dir_all(dir.parent().expect("parent")).expect("mkdir");

    dir
}

fn text(out: &Output) -> String {
    format!(
        "exit {:?}\nstdout:\n{}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// The request bodies a fake vLLM server received.
type Seen = Arc<Mutex<Vec<serde_json::Value>>>;

/// A vLLM stand-in on 127.0.0.1 that answers every chat completion with a valid Cascade in
/// prose. The Cascade depends only on the request's seed, so equal requests get equal
/// answers in every run.
fn fake_vllm() -> (String, Seen) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let base = format!("http://{}", listener.local_addr().expect("addr"));
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();

    thread::spawn(move || {
        for sock in listener.incoming() {
            let Ok(sock) = sock else { continue };
            let log = log.clone();

            thread::spawn(move || serve_one(sock, &log));
        }
    });

    (base, seen)
}

fn serve_one(mut sock: TcpStream, log: &Mutex<Vec<serde_json::Value>>) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];

    let body = loop {
        let Ok(n) = sock.read(&mut chunk) else { return };
        buf.extend_from_slice(&chunk[..n]);

        let head_end = buf.windows(4).position(|w| w == b"\r\n\r\n");

        if let Some(end) = head_end {
            let head = String::from_utf8_lossy(&buf[..end]).to_ascii_lowercase();
            let len = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0)))
                .unwrap_or(0);

            if buf.len() >= end + 4 + len {
                break String::from_utf8_lossy(&buf[end + 4..end + 4 + len]).to_string();
            }
        }

        if n == 0 {
            return;
        }
    };
    let req: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
    let seed = req["seed"].as_u64().unwrap_or(0);
    let think = 64 * (1 + seed % 32);
    let cascade = format!(
        r#"{{"stages":[{{"workers":[{{"model":"sim:large","sections":[
            {{"name":"role","text":"You solve grid puzzles."}},
            {{"name":"strategy","text":"Look for mirror images, rotations and colour changes."}},
            {{"name":"format","text":"Answer with the line ANSWER followed by the output grid."}}],
          "traits":{{"think_tokens":{think},"temperature":0.0,"answer_tokens":512,"self_check":true}}}}]}}]}}"#
    );
    let content = format!("Here is the new Cascade:\n```json\n{cascade}\n```\n");
    let reply = serde_json::json!({
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 120},
    })
    .to_string();

    log.lock().expect("log").push(req);

    let _ = write!(
        sock,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
        reply.len()
    );
}

fn breed(args: &[&str]) -> Output {
    Command::new(BREED).args(args).output().expect("run brut-breed-ollama")
}

fn mutator(base: &str, args: &[&str]) -> Output {
    Command::new(MUTATOR)
        .args(args)
        .env("VLLM_URL", base)
        .env("OLLAMA_TIMEOUT_SECS", "30")
        .output()
        .expect("run brut-mutator")
}

/// The loop by hand: breed, answer, resume, until the run is done. Returns the rounds.
fn manual_loop(dir: &Path, base: &str) -> usize {
    let out = dir.to_str().expect("path");
    let mut args = vec!["--out", out];

    args.extend_from_slice(&SMALL);

    let mut last = breed(&args);
    let mut rounds = 0;

    while last.status.code() == Some(3) {
        assert!(rounds < 5, "too many rounds");

        let m = mutator(base, &["--out", out, "--tag", TAG]);

        assert_eq!(m.status.code(), Some(0), "{}", text(&m));
        rounds += 1;
        last = breed(&["--out", out, "--resume"]);
    }

    assert_eq!(last.status.code(), Some(0), "{}", text(&last));

    rounds
}

fn requests(dir: &Path) -> Vec<(String, GenRequest)> {
    let mut v: Vec<(String, GenRequest)> = std::fs::read_dir(dir.join("llm").join("requests"))
        .expect("requests")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| {
            let key = p.file_stem().and_then(|s| s.to_str()).expect("key").to_string();
            let req = serde_json::from_str(&std::fs::read_to_string(&p).expect("read")).expect("request");

            (key, req)
        })
        .collect();

    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

fn records(dir: &Path) -> Vec<CallRecord> {
    std::fs::read_to_string(dir.join("llm").join("mutator-calls.jsonl"))
        .expect("record")
        .lines()
        .map(|l| serde_json::from_str(l).expect("json"))
        .collect()
}

#[test]
fn a_paused_run_goes_on_with_model_answers_and_a_repeat_replays_them() {
    let (base, seen) = fake_vllm();
    let dir = fresh("manual");
    let out = dir.to_str().expect("path");

    assert_eq!(manual_loop(&dir, &base), 2, "generations 1 and 2 are LLM rounds");

    // Every request went to the model as it stands in its file (the seed in 63 bits).
    let reqs = requests(&dir);
    let bodies = seen.lock().expect("seen").clone();

    assert_eq!(reqs.len(), 4);
    assert_eq!(bodies.len(), 4);

    for (key, req) in &reqs {
        assert_eq!(&request_key(req), key);

        let body = bodies
            .iter()
            .find(|b| b["messages"][1]["content"] == req.prompt.as_str())
            .expect("the request reached the model");

        assert_eq!(body["model"], "qwen2.5-32b");
        assert_eq!(body["messages"][0]["content"], req.system.as_str());
        assert_eq!(body["max_tokens"], req.max_tokens);
        assert_eq!(body["seed"], wire_seed(req.seed));
        assert!((body["temperature"].as_f64().expect("t") - f64::from(req.temperature)).abs() < 1e-6);
    }

    // Every call and its full answer are recorded, and each answer file holds that answer.
    let recs = records(&dir);

    assert_eq!(recs.len(), 4);

    for (key, req) in &reqs {
        let rec = recs.iter().find(|r| &r.request == req).expect("recorded");
        let answer = std::fs::read_to_string(dir.join("llm").join("answers").join(format!("{key}.txt"))).expect("answer");

        assert_eq!(rec.tag, TAG);
        assert_eq!(answer, rec.response.text);
    }

    // The core used the answers: four exchanges, and the Cascades in them were accepted.
    let log = std::fs::read_to_string(dir.join("events.jsonl")).expect("log");
    let exchanges: Vec<&str> = log.lines().filter(|l| l.contains("\"kind\":\"llm_exchange\"")).collect();

    assert_eq!(exchanges.len(), 4);
    assert!(exchanges.iter().any(|l| l.contains("\"accepted\":true")), "{exchanges:?}");
    assert!(dir.join("report.json").is_file());

    // A repeated round finds nothing to answer and asks no model.
    let again = mutator(&base, &["--out", out, "--tag", TAG]);

    assert_eq!(again.status.code(), Some(4), "{}", text(&again));

    // A lost answer comes back from the record, without a new call and unchanged.
    let (key, _) = &reqs[0];
    let path = dir.join("llm").join("answers").join(format!("{key}.txt"));
    let before = std::fs::read_to_string(&path).expect("answer");

    std::fs::remove_file(&path).expect("rm");

    let replay = mutator(&base, &["--out", out, "--tag", TAG]);

    assert_eq!(replay.status.code(), Some(0), "{}", text(&replay));
    assert_eq!(std::fs::read_to_string(&path).expect("answer"), before);
    assert_eq!(seen.lock().expect("seen").len(), 4, "no new call");
    assert_eq!(records(&dir).len(), 4);
    assert!(Command::new(BREED).args(["--out", out, "--verify"]).status().expect("verify").success());
}

#[test]
fn the_driver_breeds_through_every_llm_round_as_the_manual_loop_does() {
    let (base, seen) = fake_vllm();
    let by_hand = fresh("by-hand");
    let driven = fresh("driven");

    manual_loop(&by_hand, &base);

    let calls_by_hand = seen.lock().expect("seen").len();
    let out = driven.to_str().expect("path");
    let bin_dir = Path::new(MUTATOR).parent().expect("bin dir");
    let mut args = vec!["--tag", TAG, "--max-rounds", "5", "--parallel", "2", "--", "--out", out];

    args.extend_from_slice(&SMALL);

    let run = Command::new("bash")
        .arg(driver())
        .args(&args)
        .env("BRUT_BIN_DIR", bin_dir)
        .env("VLLM_URL", &base)
        .output()
        .expect("run the driver");

    assert_eq!(run.status.code(), Some(0), "{}", text(&run));
    assert!(String::from_utf8_lossy(&run.stderr).contains("mutator round 2 of at most 5"), "{}", text(&run));
    assert_eq!(seen.lock().expect("seen").len(), calls_by_hand * 2);

    // The same answers to the same requests give the same run.
    for file in ["events.jsonl", "report.json"] {
        let a = std::fs::read_to_string(by_hand.join(file)).expect("by hand");
        let b = std::fs::read_to_string(driven.join(file)).expect("driven");

        assert_eq!(a, b, "{file}");
    }
}

#[test]
fn brut_mutator_refuses_before_any_call() {
    let (base, seen) = fake_vllm();
    let dir = fresh("refuse");
    let out = dir.to_str().expect("path");
    let req = GenRequest {
        system: "s".to_string(),
        prompt: "p".to_string(),
        max_tokens: 64,
        temperature: 0.7,
        seed: 1,
        think_tokens: 0,
    };
    let key = request_key(&req);
    let requests = dir.join("llm").join("requests");

    std::fs::create_dir_all(&requests).expect("mkdir");

    // Without config.json the directory holds no run.
    let no_run = mutator(&base, &["--out", out, "--tag", TAG]);

    assert_eq!(no_run.status.code(), Some(2), "{}", text(&no_run));

    std::fs::write(dir.join("config.json"), "{}").expect("config");
    std::fs::write(requests.join(format!("{key}.json")), serde_json::to_string_pretty(&req).expect("json"))
        .expect("request");

    let elsewhere = fresh("refuse-elsewhere");
    let elsewhere = elsewhere.to_str().expect("path");

    for (args, why) in [
        (vec!["--out", out, "--tag", TAG, "--llm-answers", elsewhere], "answers outside the run"),
        (vec!["--out", out, "--tag", "vllm:gpt-unknown"], "a model off the allow-list"),
        (vec!["--out", out, "--tag", "sim:large"], "a tag of no local model"),
        (vec!["--out", out], "no tag"),
        (vec!["--out", out, "--tag", TAG, "--parallel", "0"], "zero in flight"),
        (vec!["--out", out, "--tag", TAG, "--resume", "x"], "an unknown option"),
    ] {
        let o = mutator(&base, &args);

        assert_eq!(o.status.code(), Some(2), "{why}: {}", text(&o));
    }

    // A server that is not on loopback (I3).
    let far = Command::new(MUTATOR)
        .args(["--out", out, "--tag", TAG])
        .env("VLLM_URL", "http://10.1.2.3:8000")
        .output()
        .expect("run");

    assert_eq!(far.status.code(), Some(2), "{}", text(&far));

    // Another process holds the run.
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join("LOCK"))
        .expect("lock file");

    lock.try_lock().expect("lock");

    let held = mutator(&base, &["--out", out, "--tag", TAG]);

    assert_eq!(held.status.code(), Some(2), "{}", text(&held));
    drop(lock);

    // The kill switch: no model is asked.
    std::fs::write(dir.join("STOP"), "").expect("stop");

    let stopped = mutator(&base, &["--out", out, "--tag", TAG]);

    assert_eq!(stopped.status.code(), Some(0), "{}", text(&stopped));
    assert!(seen.lock().expect("seen").is_empty(), "no call was made");
    assert!(!dir.join("llm").join("answers").join(format!("{key}.txt")).exists());

    // Without STOP the one request is answered.
    std::fs::remove_file(dir.join("STOP")).expect("rm");

    let ok = mutator(&base, &["--out", out, "--tag", TAG]);

    assert_eq!(ok.status.code(), Some(0), "{}", text(&ok));
    assert_eq!(seen.lock().expect("seen").len(), 1);
}

/// A fake pair of programs for the driver: brut-breed-ollama exits with the codes in
/// FAKE_CODES in turn (the last one repeats), brut-mutator with FAKE_MUTATOR (default 0)
/// and touches FAKE_TOUCH if set; both log their arguments.
fn fake_bins(dir: &Path) -> PathBuf {
    let bin = dir.join("bin");

    std::fs::create_dir_all(&bin).expect("mkdir");

    let breed = r#"#!/bin/sh
echo "breed $*" >> "$FAKE_LOG"
n=$(cat "$FAKE_STATE" 2>/dev/null || echo 0)
n=$((n + 1))
echo "$n" > "$FAKE_STATE"
code=$(echo "$FAKE_CODES" | cut -d' ' -f"$n")
[ -n "$code" ] || code=$(echo "$FAKE_CODES" | awk '{print $NF}')
exit "$code"
"#;
    let mutator = r#"#!/bin/sh
echo "mutator $*" >> "$FAKE_LOG"
[ -z "${FAKE_TOUCH:-}" ] || touch "$FAKE_TOUCH"
exit "${FAKE_MUTATOR:-0}"
"#;

    for (name, body) in [("brut-breed-ollama", breed), ("brut-mutator", mutator)] {
        let path = bin.join(name);

        std::fs::write(&path, body).expect("write");
        Command::new("chmod").arg("+x").arg(&path).status().expect("chmod");
    }

    bin
}

/// Run the driver on fake programs; returns its exit code and the logged calls.
fn drive(name: &str, codes: &str, mutator_code: &str, touch_stop: bool, args: &[&str]) -> (Option<i32>, Vec<String>, String) {
    let dir = fresh(name);
    let run = dir.join("run");

    std::fs::create_dir_all(&run).expect("mkdir");

    let bin = fake_bins(&dir);
    let log = dir.join("log");
    let stop = run.join("STOP");
    let out = run.to_str().expect("path").to_string();
    let args: Vec<String> = args.iter().map(|a| a.replace("RUN", &out)).collect();
    let o = Command::new("bash")
        .arg(driver())
        .args(&args)
        .env("BRUT_BIN_DIR", &bin)
        .env("FAKE_LOG", &log)
        .env("FAKE_STATE", dir.join("state"))
        .env("FAKE_CODES", codes)
        .env("FAKE_MUTATOR", mutator_code)
        .env("FAKE_TOUCH", if touch_stop { stop.to_str().expect("path") } else { "" })
        .output()
        .expect("run the driver");
    let calls = std::fs::read_to_string(&log)
        .unwrap_or_default()
        .lines()
        .map(|l| l.replace(&out, "RUN"))
        .collect();

    (o.status.code(), calls, text(&o))
}

const FRESH: [&str; 16] = [
    "--tag", "vllm:qwen2.5-32b", "--parallel", "3", "--",
    "--out", "RUN", "--config", "c.json", "--no-pause", "--seed", "5", "--max-seconds", "60", "--llm-answers", "RUN/a",
];

#[test]
fn the_driver_resumes_after_each_round_with_only_the_options_a_resume_takes() {
    let (code, calls, out) = drive("drive-loop", "3 3 0", "0", false, &FRESH);

    assert_eq!(code, Some(0), "{out}");
    assert_eq!(
        calls,
        vec![
            "breed --out RUN --config c.json --no-pause --seed 5 --max-seconds 60 --llm-answers RUN/a",
            "mutator --out RUN --tag vllm:qwen2.5-32b --llm-answers RUN/a --parallel 3",
            "breed --out RUN --resume --max-seconds 60 --llm-answers RUN/a",
            "mutator --out RUN --tag vllm:qwen2.5-32b --llm-answers RUN/a --parallel 3",
            "breed --out RUN --resume --max-seconds 60 --llm-answers RUN/a",
        ],
        "{out}"
    );
}

#[test]
fn the_driver_stops_on_errors_the_kill_switch_and_the_round_cap() {
    // An error or wrong usage of the run ends the loop with its code, no round started.
    for c in ["1", "2"] {
        let (code, calls, out) = drive(&format!("drive-err-{c}"), c, "0", false, &FRESH);

        assert_eq!(code, Some(c.parse().expect("n")), "{out}");
        assert_eq!(calls.len(), 1, "{out}");
    }

    // The round cap: the run still waits after it.
    let mut capped = vec!["--max-rounds", "2"];

    capped.extend_from_slice(&FRESH);

    let (code, calls, out) = drive("drive-cap", "3", "0", false, &capped);

    assert_eq!(code, Some(3), "{out}");
    assert_eq!(calls.iter().filter(|c| c.starts_with("mutator")).count(), 2, "{out}");
    assert_eq!(calls.iter().filter(|c| c.starts_with("breed")).count(), 3, "{out}");

    // STOP written during a round: no resume follows.
    let (code, calls, out) = drive("drive-stop", "3 0", "0", true, &FRESH);

    assert_eq!(code, Some(0), "{out}");
    assert_eq!(calls.len(), 2, "{out}");
    assert!(out.contains("STOP exists"), "{out}");

    // A failed round, and a round with nothing to answer, end the loop as errors.
    for (m, want) in [("1", 1), ("4", 1), ("2", 2)] {
        let (code, calls, out) = drive(&format!("drive-mutator-{m}"), "3 0", m, false, &FRESH);

        assert_eq!(code, Some(want), "{out}");
        assert_eq!(calls.len(), 2, "{out}");
    }

    // Wrong usage of the driver itself.
    for bad in [
        vec!["--", "--out", "RUN"],
        vec!["--tag", TAG, "--", "--seed", "5"],
        vec!["--tag", TAG, "--max-rounds", "0", "--", "--out", "RUN"],
        vec!["--tag", TAG],
        vec!["--tag", TAG, "--out", "RUN"],
    ] {
        let (code, calls, out) = drive("drive-usage", "0", "0", false, &bad);

        assert_eq!(code, Some(2), "{bad:?}: {out}");
        assert!(calls.is_empty(), "{out}");
    }
}
