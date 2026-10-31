//! The Ollama adapter of docs/STAGE2_BACKEND_CONTRACT.md (contract-v2): a local open model
//! behind the core's `ModelBackend` trait.
//!
//! - I1: a separate package; the core crate gains no dependency.
//! - I2: ureq 3 without an async runtime, and without TLS.
//! - I3: the base URL must name a loopback host; redirects are refused.
//! - I5: POST /api/generate with model, prompt, system, stream false, temperature, seed and
//!   num_predict; `done_reason` "length", or more tokens than asked, maps to Length.
//! - I6: a per-request timeout, a byte cap on the answer, and an allow-list of models.
//! - I7: `Recorded` files every request and its full answer and replays them.
//!
//! Only models without a thinking mode are on the allow-list for now, so `think_tokens` is
//! ignored as I5 says for such models. Thinking models (Qwen3, gpt-oss) follow once their
//! separate caps are built.
//!
//! The module `vllm` holds a second backend with the same guarantees for a batching
//! server with an OpenAI-compatible API (vLLM). The module `vote` holds the compute-equal
//! majority-vote control of a finished run (binary `vote-control`).
#![forbid(unsafe_code)]

pub mod vllm;
pub mod vote;

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use brutmaschine::backend::{GenRequest, GenResponse, ModelBackend, StopReason};
use brutmaschine::cascade::arena::request_key;
use brutmaschine::util::{BrutError, Result};

/// The prefix of every Ollama model tag in a run's allow-list: `ollama:qwen2.5:7b`.
pub const TAG_PREFIX: &str = "ollama:";

/// Models the adapter may call (I6), all without a thinking mode.
pub const ALLOWED_MODELS: [&str; 4] = ["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "llama3.1:8b"];

/// The default Ollama server (I3).
pub const DEFAULT_BASE: &str = "http://127.0.0.1:11434";

/// Hard cap on the bytes of one answer (I6).
pub const MAX_RESPONSE_BYTES: u64 = 1 << 20;

/// The seed as the servers take it: the low 63 bits. The core draws request seeds as full
/// u64 values; vLLM refuses any seed above i64::MAX with HTTP 400, and Ollama reads it as a
/// signed integer. Masking keeps the seed deterministic and in range for both.
pub fn wire_seed(seed: u64) -> u64 {
    seed & (i64::MAX as u64)
}

pub(crate) fn err(msg: String) -> BrutError {
    BrutError::Io(std::io::Error::other(msg))
}

/// Error unless `base` is `http://` followed by a loopback host and an optional port (I3).
pub fn check_loopback(base: &str) -> Result<()> {
    let rest = base
        .strip_prefix("http://")
        .ok_or_else(|| BrutError::Invalid(format!("ollama: {base:?} is not an http:// URL")))?;
    let authority = rest.split('/').next().unwrap_or_default();

    if authority.contains('@') {
        return Err(BrutError::Invalid(format!("ollama: {base:?} carries user info")));
    }

    let host = match authority.strip_prefix('[') {
        Some(v6) => v6.split(']').next().unwrap_or_default(),
        None => authority.split(':').next().unwrap_or_default(),
    };
    let loopback = host == "localhost"
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback());

    if loopback {
        Ok(())
    } else {
        Err(BrutError::Invalid(format!(
            "ollama: {base:?} is not a loopback address; the adapter calls only the local server"
        )))
    }
}

/// One local model on an Ollama server.
pub struct OllamaBackend {
    tag: String,
    model: String,
    url: String,
    agent: ureq::Agent,
}

#[derive(Deserialize)]
struct Reply {
    #[serde(default)]
    response: String,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

impl OllamaBackend {
    /// The backend for `tag` (`ollama:<model>`), on the server at `base`. Refuses a model
    /// off the allow-list and a non-loopback server before any call.
    pub fn new(tag: &str, base: &str, timeout: Duration) -> Result<OllamaBackend> {
        let model = tag
            .strip_prefix(TAG_PREFIX)
            .ok_or_else(|| BrutError::Invalid(format!("ollama: {tag:?} lacks {TAG_PREFIX}")))?;

        if !ALLOWED_MODELS.contains(&model) {
            return Err(BrutError::Invalid(format!(
                "ollama: model {model:?} is not on the allow-list {ALLOWED_MODELS:?}"
            )));
        }

        check_loopback(base)?;

        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .max_redirects(0)
            .http_status_as_error(false)
            .build()
            .into();

        Ok(OllamaBackend {
            tag: tag.to_string(),
            model: model.to_string(),
            url: format!("{}/api/generate", base.trim_end_matches('/')),
            agent,
        })
    }

    /// The JSON body of `req` (I5). A model without a thinking mode gets
    /// num_predict = max_tokens, and `think_tokens` is ignored.
    pub fn body(&self, req: &GenRequest) -> serde_json::Value {
        serde_json::json!({
            "model": self.model,
            "system": req.system,
            "prompt": req.prompt,
            "stream": false,
            "options": {
                "temperature": req.temperature,
                "seed": wire_seed(req.seed),
                "num_predict": req.max_tokens,
            },
        })
    }
}

impl ModelBackend for OllamaBackend {
    fn name(&self) -> &str {
        &self.tag
    }

    fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
        let body = serde_json::to_vec(&self.body(req)).map_err(BrutError::Json)?;
        let mut resp = self
            .agent
            .post(&self.url)
            .header("Content-Type", "application/json")
            .send(&body[..])
            .map_err(|e| err(format!("ollama: {e}")))?;
        let status = resp.status();
        let text = resp
            .body_mut()
            .with_config()
            .limit(MAX_RESPONSE_BYTES)
            .read_to_string()
            .map_err(|e| err(format!("ollama: reading the answer: {e}")))?;

        if !status.is_success() {
            return Err(err(format!("ollama: HTTP {status}: {}", text.chars().take(300).collect::<String>())));
        }

        let reply: Reply = serde_json::from_str(&text).map_err(BrutError::Json)?;
        let completion = reply
            .eval_count
            .unwrap_or_else(|| brutmaschine::backend::approx_tokens(&reply.response));
        let cut = reply.done_reason.as_deref() == Some("length") || completion > req.max_tokens;

        Ok(GenResponse {
            prompt_tokens: reply.prompt_eval_count.unwrap_or(0),
            completion_tokens: completion,
            text: reply.response,
            stop_reason: if cut { StopReason::Length } else { StopReason::Stop },
            backend: self.tag.clone(),
        })
    }
}

/// One recorded call: the request, which model got it, and its full answer (I7).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CallRecord {
    pub key: String,
    pub tag: String,
    pub request: GenRequest,
    pub response: GenResponse,
}

/// A backend whose answers are filed in a JSONL file and replayed from it. A request it
/// has seen is never sent again, so a resumed or repeated run gets the same answers even
/// though the model is not bit-for-bit deterministic (contract I7, ADR 0002). Failed calls
/// are not filed as answers: each one goes to `worker-errors.jsonl` beside the record and
/// to stderr, because the core counts a failed Worker call as an unsolved puzzle without a
/// word, which hid a fault that failed half of all calls in the first vLLM runs. An answer
/// that cannot be filed (a full disk) is a failed call as well, filed and counted the same.
pub struct Recorded<B: ModelBackend> {
    inner: B,
    path: PathBuf,
    book: Mutex<BTreeMap<String, GenResponse>>,
    errors: std::sync::atomic::AtomicU64,
}

/// One failed call, as filed in `worker-errors.jsonl`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ErrorRecord {
    pub key: String,
    pub tag: String,
    pub seed: u64,
    pub error: String,
}

/// The key of a call: the model tag and the request's own key.
pub fn call_key(tag: &str, req: &GenRequest) -> String {
    format!("{tag}#{}", request_key(req))
}

impl<B: ModelBackend> Recorded<B> {
    /// Wrap `inner`, loading the answers already filed in `path` for its tag.
    pub fn open(inner: B, path: &Path) -> Result<Recorded<B>> {
        let mut book = BTreeMap::new();

        if path.exists() {
            for line in BufReader::new(File::open(path).map_err(BrutError::Io)?).lines() {
                let line = line.map_err(BrutError::Io)?;

                if line.trim().is_empty() {
                    continue;
                }

                let rec: CallRecord = serde_json::from_str(&line).map_err(BrutError::Json)?;

                if rec.tag == inner.name() {
                    book.insert(rec.key, rec.response);
                }
            }
        }

        Ok(Recorded {
            inner,
            path: path.to_path_buf(),
            book: Mutex::new(book),
            errors: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Number of answers on file for this backend.
    pub fn len(&self) -> usize {
        self.book.lock().map(|b| b.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of failed calls in this process.
    pub fn errors(&self) -> u64 {
        self.errors.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// The file that lists the failed calls, beside the record.
    pub fn error_path(&self) -> PathBuf {
        self.path.with_file_name("worker-errors.jsonl")
    }

    /// File a failed call and say so on stderr: the first five, then every thousandth.
    fn file_error(&self, key: &str, req: &GenRequest, e: &BrutError) {
        let n = self.errors.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        let rec = ErrorRecord {
            key: key.to_string(),
            tag: self.inner.name().to_string(),
            seed: req.seed,
            error: e.to_string(),
        };

        if n <= 5 || n.is_multiple_of(1000) {
            eprintln!("brut-breed-ollama: call {n} failed ({}): {}", rec.tag, rec.error);
        }

        if let Ok(mut line) = serde_json::to_string(&rec) {
            line.push('\n');

            let path = self.error_path();
            let written = path
                .parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .and_then(|_| OpenOptions::new().create(true).append(true).open(&path))
                .and_then(|mut f| f.write_all(line.as_bytes()));

            if let Err(io) = written {
                eprintln!("brut-breed-ollama: cannot write {}: {io}", path.display());
            }
        }
    }
}

impl<B: ModelBackend> ModelBackend for Recorded<B> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
        let key = call_key(self.inner.name(), req);

        if let Some(hit) = self.book.lock().ok().and_then(|b| b.get(&key).cloned()) {
            return Ok(hit);
        }

        let response = match self.inner.generate(req) {
            Ok(r) => r,
            Err(e) => {
                self.file_error(&key, req, &e);
                return Err(e);
            }
        };
        let rec = CallRecord {
            key: key.clone(),
            tag: self.inner.name().to_string(),
            request: req.clone(),
            response: response.clone(),
        };
        let mut book = match self.book.lock() {
            Ok(b) => b,
            Err(_) => {
                let e = err("ollama: the record lock is poisoned".to_string());

                self.file_error(&key, req, &e);
                return Err(e);
            }
        };
        // An answer that cannot be filed is a failed call too: the core would count it as
        // an unsolved puzzle without a word, as it does any error (a full disk, a
        // read-only directory). A partly written line is taken back, so the record still
        // reads and a repeat retries the call.
        let filed = (|| -> Result<()> {
            let io = |e: std::io::Error| err(format!("ollama: filing the answer in {}: {e}", self.path.display()));
            let mut line = serde_json::to_string(&rec).map_err(BrutError::Json)?;

            line.push('\n');

            if let Some(dir) = self.path.parent() {
                std::fs::create_dir_all(dir).map_err(io)?;
            }

            let mut f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
                .map_err(io)?;
            let start = f.metadata().map_err(io)?.len();

            if let Err(e) = f.write_all(line.as_bytes()).and_then(|()| f.sync_data()) {
                let _ = f.set_len(start);
                return Err(io(e));
            }

            Ok(())
        })();

        if let Err(e) = filed {
            drop(book);
            self.file_error(&key, req, &e);
            return Err(e);
        }

        book.insert(key, response.clone());

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_http::serve;
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn req(max_tokens: u32) -> GenRequest {
        GenRequest {
            system: "sys".to_string(),
            prompt: "hello".to_string(),
            max_tokens,
            temperature: 0.0,
            seed: 7,
            think_tokens: 0,
        }
    }

    #[test]
    fn seeds_above_i64_max_are_masked_to_63_bits() {
        assert_eq!(wire_seed(7), 7);
        assert_eq!(wire_seed(i64::MAX as u64), i64::MAX as u64);
        assert_eq!(wire_seed(u64::MAX), i64::MAX as u64);
        assert_eq!(wire_seed(1 << 63), 0);

        let b = OllamaBackend::new("ollama:qwen2.5:7b", DEFAULT_BASE, Duration::from_secs(1)).expect("new");
        let mut r = req(8);
        r.seed = u64::MAX;
        assert_eq!(b.body(&r)["options"]["seed"], i64::MAX);
    }

    /// Fails every call.
    struct Broken;

    impl ModelBackend for Broken {
        fn name(&self) -> &str {
            "vllm:qwen2.5-32b"
        }

        fn generate(&self, _req: &GenRequest) -> Result<GenResponse> {
            Err(err("vllm: HTTP 400 Bad Request: seed out of range".to_string()))
        }
    }

    #[test]
    fn failed_calls_are_filed_and_counted_not_recorded() {
        let dir = std::env::temp_dir().join(format!("backend-ollama-errors-{}", std::process::id()));
        let path = dir.join("worker-calls.jsonl");
        let _ = std::fs::remove_dir_all(&dir);

        let b = Recorded::open(Broken, &path).expect("open");

        assert!(b.generate(&req(8)).is_err());
        assert!(b.generate(&req(9)).is_err());
        assert_eq!(b.errors(), 2);
        assert_eq!(b.len(), 0);
        assert!(!path.exists(), "a failure must never be filed as an answer");

        let lines: Vec<ErrorRecord> = std::fs::read_to_string(b.error_path())
            .expect("errors file")
            .lines()
            .map(|l| serde_json::from_str(l).expect("json"))
            .collect();

        assert_eq!(lines.len(), 2);
        assert!(lines[0].error.contains("HTTP 400"));
        assert_eq!(lines[0].tag, "vllm:qwen2.5-32b");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_answer_that_cannot_be_filed_is_a_failed_call() {
        let dir = std::env::temp_dir().join(format!("backend-ollama-unfiled-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");

        // The record's parent is a regular file, so no directory can be made for it.
        let blocker = dir.join("blocker");
        std::fs::write(&blocker, "not a directory").expect("write");

        let b = Recorded::open(Counter(AtomicUsize::new(0)), &blocker.join("worker-calls.jsonl")).expect("open");
        let got = b.generate(&req(8));

        assert!(got.as_ref().is_err_and(|e| e.to_string().contains("filing the answer")), "{got:?}");
        assert_eq!(b.inner.0.load(Ordering::SeqCst), 1, "the model was asked");
        assert_eq!(b.errors(), 1, "an unfiled answer counts as a failed call");
        assert_eq!(b.len(), 0, "and is not served from the book either");

        // A full disk: every write to /dev/full fails with ENOSPC (Linux). Only without
        // root, since the error file would land in /dev beside it.
        let full = Path::new("/dev/full");
        let root = std::fs::read_to_string("/proc/self/status").map_or(true, |s| {
            s.lines().any(|l| l.starts_with("Uid:") && l.split_whitespace().nth(1) == Some("0"))
        });

        if !root && OpenOptions::new().append(true).open(full).is_ok() {
            let mut b = Recorded::open(Counter(AtomicUsize::new(0)), &dir.join("worker-calls.jsonl")).expect("open");
            b.path = full.to_path_buf();
            let got = b.generate(&req(8));

            assert!(got.as_ref().is_err_and(|e| e.to_string().contains("No space left")), "{got:?}");
            assert_eq!(b.errors(), 1);
            assert_eq!(b.len(), 0);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn only_loopback_hosts_pass() {
        for ok in ["http://127.0.0.1:11434", "http://localhost:11434/", "http://[::1]:11434", "http://127.0.0.2"] {
            assert!(check_loopback(ok).is_ok(), "{ok}");
        }

        for bad in [
            "https://127.0.0.1:11434",
            "http://10.0.0.1:11434",
            "http://example.com",
            "http://127.0.0.1.example.com",
            "http://user@127.0.0.1:11434",
            "ftp://127.0.0.1",
        ] {
            assert!(check_loopback(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_allow_list_and_the_prefix_are_enforced_before_any_call() {
        let t = Duration::from_secs(1);

        assert!(OllamaBackend::new("ollama:qwen2.5:7b", DEFAULT_BASE, t).is_ok());
        assert!(OllamaBackend::new("ollama:some-model", DEFAULT_BASE, t).is_err());
        assert!(OllamaBackend::new("qwen2.5:7b", DEFAULT_BASE, t).is_err());
        assert!(OllamaBackend::new("ollama:qwen2.5:7b", "http://10.1.2.3:11434", t).is_err());
    }

    #[test]
    fn a_request_has_the_shape_of_i5_and_the_answer_maps_back() {
        let (base, server) = serve(
            r#"{"response":"42","done":true,"done_reason":"stop","prompt_eval_count":11,"eval_count":2}"#,
        );
        let b = OllamaBackend::new("ollama:qwen2.5:7b", &base, Duration::from_secs(10)).expect("new");
        let out = b.generate(&req(64)).expect("generate");
        let seen = server.join().expect("server");
        let body: serde_json::Value =
            serde_json::from_str(seen.split("\r\n\r\n").nth(1).expect("body")).expect("json");

        assert!(seen.starts_with("POST /api/generate "));
        assert_eq!(body["model"], "qwen2.5:7b");
        assert_eq!(body["stream"], false);
        assert_eq!(body["system"], "sys");
        assert_eq!(body["prompt"], "hello");
        assert_eq!(body["options"]["seed"], 7);
        assert_eq!(body["options"]["num_predict"], 64);
        assert_eq!(out.text, "42");
        assert_eq!(out.prompt_tokens, 11);
        assert_eq!(out.completion_tokens, 2);
        assert_eq!(out.stop_reason, StopReason::Stop);
        assert_eq!(out.backend, "ollama:qwen2.5:7b");
    }

    #[test]
    fn a_cut_off_answer_is_length() {
        let (base, server) =
            serve(r#"{"response":"a b c","done":true,"done_reason":"length","eval_count":3}"#);
        let b = OllamaBackend::new("ollama:qwen2.5:7b", &base, Duration::from_secs(10)).expect("new");
        let out = b.generate(&req(3)).expect("generate");

        server.join().expect("server");
        assert_eq!(out.stop_reason, StopReason::Length);
        assert!(!out.is_usable());
    }

    #[test]
    fn a_dead_server_is_an_error_not_a_panic() {
        let port = TcpListener::bind("127.0.0.1:0").expect("bind").local_addr().expect("addr").port();
        let b = OllamaBackend::new("ollama:qwen2.5:7b", &format!("http://127.0.0.1:{port}"), Duration::from_secs(2))
            .expect("new");

        assert!(b.generate(&req(8)).is_err());
    }

    /// Counts calls and answers with the call number.
    struct Counter(AtomicUsize);

    impl ModelBackend for Counter {
        fn name(&self) -> &str {
            "ollama:qwen2.5:7b"
        }

        fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
            let n = self.0.fetch_add(1, Ordering::SeqCst);

            Ok(GenResponse {
                text: format!("answer {n} to {}", req.prompt),
                prompt_tokens: 1,
                completion_tokens: 3,
                stop_reason: StopReason::Stop,
                backend: self.name().to_string(),
            })
        }
    }

    #[test]
    fn recorded_answers_replay_without_a_new_call() {
        let dir = std::env::temp_dir().join(format!("backend-ollama-test-{}", std::process::id()));
        let path = dir.join("worker-calls.jsonl");
        let _ = std::fs::remove_dir_all(&dir);

        let first = Recorded::open(Counter(AtomicUsize::new(0)), &path).expect("open");
        let a = first.generate(&req(8)).expect("a");
        let again = first.generate(&req(8)).expect("again");

        assert_eq!(a, again);
        assert_eq!(first.inner.0.load(Ordering::SeqCst), 1);

        // A new process reads the file and never calls the model for that request.
        let second = Recorded::open(Counter(AtomicUsize::new(100)), &path).expect("reopen");

        assert_eq!(second.len(), 1);
        assert_eq!(second.generate(&req(8)).expect("replay"), a);
        assert_eq!(second.inner.0.load(Ordering::SeqCst), 100);

        // Another request is a new call, filed too.
        let mut other = req(8);
        other.seed = 8;
        assert_ne!(second.generate(&other).expect("new").text, a.text);
        assert_eq!(second.len(), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Talks to the real local server. Run with:
    /// `OLLAMA_LIVE=1 cargo test --release -- --ignored live --nocapture`
    #[test]
    #[ignore]
    fn live_qwen_answers() {
        if std::env::var("OLLAMA_LIVE").is_err() {
            return;
        }

        let b = OllamaBackend::new("ollama:qwen2.5:7b", DEFAULT_BASE, Duration::from_secs(120)).expect("new");
        let out = b
            .generate(&GenRequest {
                system: "Answer with the number only.".to_string(),
                prompt: "What is 17 times 23?".to_string(),
                max_tokens: 16,
                temperature: 0.0,
                seed: 1,
                think_tokens: 0,
            })
            .expect("generate");

        eprintln!("LIVE: {out:?}");
        assert!(out.text.contains("391"), "{out:?}");
        assert_eq!(out.stop_reason, StopReason::Stop);
    }
}

/// A one-shot HTTP server for the unit tests of both backends.
#[cfg(test)]
pub(crate) mod test_http {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    /// A one-shot HTTP server on 127.0.0.1 that answers `body` and hands back the request.
    pub(crate) fn serve(body: &'static str) -> (String, thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let base = format!("http://{}", listener.local_addr().expect("addr"));
        let handle = thread::spawn(move || {
            let (mut sock, _) = listener.accept().expect("accept");
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];

            // Read the head, then as many body bytes as Content-Length says.
            loop {
                let n = sock.read(&mut chunk).expect("read");
                buf.extend_from_slice(&chunk[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();

                if let Some(end) = text.find("\r\n\r\n") {
                    let len = text
                        .lines()
                        .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0)))
                        .unwrap_or(0);

                    if buf.len() >= end + 4 + len {
                        break;
                    }
                }

                if n == 0 {
                    break;
                }
            }

            let reply = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );

            sock.write_all(reply.as_bytes()).expect("write");
            String::from_utf8_lossy(&buf).to_string()
        });

        (base, handle)
    }

}
