//! Model-served Mutator rounds: answer the requests that a paused run left in
//! `<out>/llm/requests/` with a local model, so that the run can go on without a person.
//!
//! The core pauses an LLM round with exit code 3 and writes each missing request as
//! `<out>/llm/requests/<key>.json`: the `GenRequest` as pretty JSON, `<key>` its
//! `request_key`. On resume it reads `<answers>/<key>.txt` (default `<out>/llm/answers`,
//! else `--llm-answers`) as the model's answer text and parses it itself
//! (`parse_llm_cascade`). This module
//!
//! - finds the pending requests: every request file whose key has no answer file yet;
//! - sends each request unchanged to a backend, normally wrapped in `Recorded`, so that
//!   every request and its full answer land in `<out>/llm/mutator-calls.jsonl` and a
//!   repeated round replays them instead of calling the model again (ADR 0002, I7);
//! - writes the answer text as it came, atomically, and never over an existing answer;
//! - reads and writes only inside the run directory.
//!
//! It checks nothing the core does not check: an answer that is not a valid Cascade is
//! rejected by the core on resume and replaced by a random child. A failed call writes
//! no answer, so its request stays pending. The backends keep I3 (loopback only) and I6
//! (timeout, byte cap, allow-list).

use std::fs::{File, OpenOptions, TryLockError};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use brutmaschine::backend::{GenRequest, ModelBackend, StopReason};
use brutmaschine::cascade::arena::request_key;
use brutmaschine::util::{BrutError, Result};

/// The record of the Mutator calls, in `<out>/llm/` beside `worker-calls.jsonl`.
pub const MUTATOR_CALLS: &str = "mutator-calls.jsonl";

/// The file every `brut-breed` run writes at its start; its presence marks a run directory.
const CONFIG: &str = "config.json";

/// The kill switch of a run, shared with `brut-breed`.
pub const STOP: &str = "STOP";

/// The lock file of a run: one writer at a time, as in `brut-breed`.
const LOCK: &str = "LOCK";

fn invalid(msg: String) -> BrutError {
    BrutError::Invalid(msg)
}

/// True for a key as `request_key` makes it: 16 lowercase hex digits. Only such a name
/// becomes a file name, so no key can leave the answers directory.
pub fn is_key(s: &str) -> bool {
    s.len() == 16 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// `dir` as an absolute path whose existing part has its symlinks resolved, refused
/// unless it lies strictly inside `root` (canonical). Relative paths count from the
/// working directory, as `brut-breed` reads `--llm-answers`; a `..` is refused outright.
fn resolve_inside(root: &Path, dir: &Path) -> Result<PathBuf> {
    let abs = if dir.is_absolute() {
        dir.to_path_buf()
    } else {
        std::env::current_dir().map_err(BrutError::Io)?.join(dir)
    };

    if abs.components().any(|c| c == Component::ParentDir) {
        return Err(invalid(format!("{} contains '..'; give a path inside the run directory", dir.display())));
    }

    // The deepest part that exists is resolved; the rest is appended as written.
    let mut base = abs.as_path();
    let mut rest = Vec::new();

    while std::fs::symlink_metadata(base).is_err() {
        let (Some(name), Some(parent)) = (base.file_name(), base.parent()) else {
            break;
        };

        rest.push(name.to_os_string());
        base = parent;
    }

    let mut full = base.canonicalize().map_err(BrutError::Io)?;

    for name in rest.iter().rev() {
        full.push(name);
    }

    if full.starts_with(root) && full != root {
        Ok(full)
    } else {
        Err(invalid(format!(
            "{} lies outside the run directory {}; the Mutator writes only inside the run",
            dir.display(),
            root.display()
        )))
    }
}

/// Where one run keeps its LLM files, each resolved inside the run directory.
#[derive(Clone, Debug)]
pub struct RunDirs {
    /// The run directory, canonical.
    pub out: PathBuf,
    /// `<out>/llm/requests`, written by the core.
    pub requests: PathBuf,
    /// Where the core reads the answers; created if missing.
    pub answers: PathBuf,
    /// `<out>/llm/mutator-calls.jsonl`.
    pub record: PathBuf,
}

impl RunDirs {
    /// The directories of the run in `out`, with the answers in `answers` (None: the
    /// core's default `<out>/llm/answers`). Refuses a directory that holds no run, and any
    /// of the paths that resolves outside it, also through a symlink.
    pub fn new(out: &Path, answers: Option<&Path>) -> Result<RunDirs> {
        let root = out
            .canonicalize()
            .map_err(|e| invalid(format!("{}: {e}", out.display())))?;

        if !root.join(CONFIG).is_file() {
            return Err(invalid(format!("{} holds no run (no {CONFIG})", out.display())));
        }

        let llm = resolve_inside(&root, &root.join("llm"))?;
        let requests = resolve_inside(&root, &llm.join("requests"))?;
        let answers = match answers {
            Some(dir) => resolve_inside(&root, dir)?,
            None => resolve_inside(&root, &llm.join("answers"))?,
        };

        std::fs::create_dir_all(&answers).map_err(BrutError::Io)?;

        // Again after creating it: a symlink that appeared meanwhile must not lead out.
        let answers = resolve_inside(&root, &answers)?;
        let record = llm.join(MUTATOR_CALLS);

        if let Ok(meta) = std::fs::symlink_metadata(&record)
            && !meta.file_type().is_file()
        {
            return Err(invalid(format!("{} is not a regular file", record.display())));
        }

        Ok(RunDirs {
            out: root,
            requests,
            answers,
            record,
        })
    }

    /// The run's kill switch `<out>/STOP`.
    pub fn stop_file(&self) -> PathBuf {
        self.out.join(STOP)
    }

    /// Where the answer to `key` goes.
    pub fn answer_path(&self, key: &str) -> PathBuf {
        self.answers.join(format!("{key}.txt"))
    }
}

/// Take `<out>/LOCK` exclusively for the life of the returned file, as `brut-breed` does
/// for every writer, so a round never runs beside a breeding process or a second round.
pub fn lock_run(dirs: &RunDirs) -> Result<File> {
    let path = dirs.out.join(LOCK);
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .map_err(|e| invalid(format!("{}: {e}", path.display())))?;

    match lock.try_lock() {
        Ok(()) => Ok(lock),
        Err(TryLockError::WouldBlock) => Err(invalid(format!(
            "usage: another process is using {}",
            dirs.out.display()
        ))),
        Err(TryLockError::Error(e)) => Err(invalid(format!("{}: {e}", path.display()))),
    }
}

/// A request that waits for its answer.
#[derive(Clone, Debug)]
pub struct Pending {
    pub key: String,
    pub request: GenRequest,
}

/// A request file that cannot be answered, and why.
#[derive(Clone, Debug)]
pub struct Refused {
    pub path: PathBuf,
    pub reason: String,
}

/// What the requests directory holds.
#[derive(Debug, Default)]
pub struct Scan {
    /// Requests without an answer, sorted by key.
    pub pending: Vec<Pending>,
    /// Requests that have an answer already.
    pub answered: usize,
    /// `.json` files that are no request of this run, or not under their own key.
    pub refused: Vec<Refused>,
}

/// Read every `<key>.json` in the requests directory. A file counts as pending only if it
/// is a regular file, parses as a `GenRequest`, and `request_key` of that request is its
/// name, which is the key the core looks its answer up by. Files of other kinds (the
/// core's `.tmp` files) are ignored.
pub fn scan(dirs: &RunDirs) -> Result<Scan> {
    let mut scan = Scan::default();

    if std::fs::symlink_metadata(&dirs.requests).is_err() {
        return Ok(scan);
    }

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dirs.requests)
        .map_err(BrutError::Io)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<_>>()
        .map_err(BrutError::Io)?;

    entries.sort();

    for path in entries {
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }

        let refuse = |reason: String| Refused {
            path: path.clone(),
            reason,
        };
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default();

        if !is_key(stem) {
            scan.refused.push(refuse("the name is no request key".to_string()));
            continue;
        }

        match std::fs::symlink_metadata(&path) {
            Ok(meta) if meta.file_type().is_file() => {}
            Ok(_) => {
                scan.refused.push(refuse("not a regular file".to_string()));
                continue;
            }
            Err(e) => {
                scan.refused.push(refuse(e.to_string()));
                continue;
            }
        }

        let request: GenRequest = match std::fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string()))
        {
            Ok(r) => r,
            Err(e) => {
                scan.refused.push(refuse(format!("not a request: {e}")));
                continue;
            }
        };
        let key = request_key(&request);

        if key != stem {
            scan.refused.push(refuse(format!("its request has the key {key}")));
            continue;
        }

        if std::fs::symlink_metadata(dirs.answer_path(&key)).is_ok() {
            scan.answered += 1;
        } else {
            scan.pending.push(Pending { key, request });
        }
    }

    Ok(scan)
}

/// How `write_answer` ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Written {
    /// The answer file is new.
    New,
    /// An answer was there already and stays as it was.
    Kept,
}

/// Write `text` as `<answers>/<key>.txt`, atomically and never over an existing file:
/// the text goes to a temporary file, which is synced and then hard-linked under the
/// final name; the link fails if the name exists, also as a symlink.
pub fn write_answer(answers: &Path, key: &str, text: &str) -> Result<Written> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    if !is_key(key) {
        return Err(invalid(format!("{key:?} is no request key")));
    }

    let path = answers.join(format!("{key}.txt"));
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp = answers.join(format!(".{key}.{}.{n}.tmp", std::process::id()));
    let linked = (|| -> std::io::Result<Written> {
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;

        f.write_all(text.as_bytes())?;
        f.sync_all()?;

        match std::fs::hard_link(&tmp, &path) {
            Ok(()) => {
                File::open(answers)?.sync_all()?;
                Ok(Written::New)
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(Written::Kept),
            Err(e) => Err(e),
        }
    })();
    let _ = std::fs::remove_file(&tmp);

    linked.map_err(|e| invalid(format!("{}: {e}", path.display())))
}

/// What one round did, every list sorted by key.
#[derive(Debug, Default)]
pub struct Round {
    /// Answers written.
    pub written: Vec<String>,
    /// Answers that appeared meanwhile and were left as they were.
    pub kept: Vec<String>,
    /// Written answers the model cut off at `max_tokens` (also in `written`).
    pub cut: Vec<String>,
    /// Requests whose call or write failed, with the error; they stay pending.
    pub failed: Vec<(String, String)>,
    /// Requests not tried because the stop signal fired.
    pub skipped: usize,
}

/// Answer every request in `todo` through `backend`, at most `parallel` at a time, and
/// write each answer as soon as it arrives. `stop` is asked before each call; calls in
/// flight finish and their answers are written.
pub fn answer(
    backend: &dyn ModelBackend,
    answers: &Path,
    todo: &[Pending],
    parallel: usize,
    stop: &(dyn Fn() -> bool + Sync),
) -> Round {
    let next = AtomicUsize::new(0);
    let round = Mutex::new(Round::default());
    let note = |f: &mut dyn FnMut(&mut Round)| {
        let mut r = round.lock().unwrap_or_else(|p| p.into_inner());

        f(&mut r);
    };

    std::thread::scope(|s| {
        for _ in 0..parallel.clamp(1, todo.len().max(1)) {
            s.spawn(|| {
                loop {
                    if stop() {
                        break;
                    }

                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some(p) = todo.get(i) else {
                        break;
                    };
                    let key = p.key.clone();

                    match backend.generate(&p.request) {
                        Err(e) => note(&mut |r| r.failed.push((key.clone(), e.to_string()))),
                        Ok(resp) => match write_answer(answers, &key, &resp.text) {
                            Ok(Written::New) => note(&mut |r| {
                                if resp.stop_reason == StopReason::Length {
                                    r.cut.push(key.clone());
                                }

                                r.written.push(key.clone());
                            }),
                            Ok(Written::Kept) => note(&mut |r| r.kept.push(key.clone())),
                            Err(e) => note(&mut |r| r.failed.push((key.clone(), e.to_string()))),
                        },
                    }
                }
            });
        }
    });

    let mut r = round.into_inner().unwrap_or_else(|p| p.into_inner());
    let tried = r.written.len() + r.kept.len() + r.failed.len();

    r.skipped = todo.len().saturating_sub(tried);
    r.written.sort();
    r.kept.sort();
    r.cut.sort();
    r.failed.sort();

    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Recorded;
    use brutmaschine::backend::GenResponse;
    use brutmaschine::cascade::arena::{AnswerBook, Mutator};

    /// A fresh directory that looks like a run: it has a config.json.
    fn run_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brut-mutator-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join(CONFIG), "{}").expect("config");

        dir
    }

    fn request(seed: u64) -> GenRequest {
        GenRequest {
            system: "You are the mutation operator.".to_string(),
            prompt: format!("Parent Cascade {seed}:\n{{\"stages\": []}}\n\"quoted\" \u{e9}"),
            max_tokens: 4096,
            temperature: 0.7,
            seed,
            think_tokens: 0,
        }
    }

    /// File `req` as the core does in `brut-breed`: pretty JSON under its key.
    fn file_request(out: &Path, req: &GenRequest) -> String {
        let dir = out.join("llm").join("requests");
        let key = request_key(req);

        std::fs::create_dir_all(&dir).expect("requests");
        std::fs::write(
            dir.join(format!("{key}.json")),
            serde_json::to_string_pretty(req).expect("json"),
        )
        .expect("write");

        key
    }

    /// Answers with the call number and the seed, or cut off when the seed is 99.
    struct Counter(AtomicUsize);

    impl ModelBackend for Counter {
        fn name(&self) -> &str {
            "vllm:qwen2.5-32b"
        }

        fn generate(&self, req: &GenRequest) -> Result<GenResponse> {
            let n = self.0.fetch_add(1, Ordering::SeqCst);

            Ok(GenResponse {
                text: format!("answer {n} to seed {}", req.seed),
                prompt_tokens: 10,
                completion_tokens: 5,
                stop_reason: if req.seed == 99 { StopReason::Length } else { StopReason::Stop },
                backend: self.name().to_string(),
            })
        }
    }

    /// Fails every call.
    struct Broken;

    impl ModelBackend for Broken {
        fn name(&self) -> &str {
            "vllm:qwen2.5-32b"
        }

        fn generate(&self, _req: &GenRequest) -> Result<GenResponse> {
            Err(crate::err("vllm: HTTP 503 Service Unavailable".to_string()))
        }
    }

    fn never() -> bool {
        false
    }

    #[test]
    fn keys_are_sixteen_lowercase_hex_digits() {
        assert!(is_key(&request_key(&request(1))));
        assert!(is_key("0123456789abcdef"));

        for bad in ["", "0123456789abcde", "0123456789abcdef0", "0123456789ABCDEF", "../../etc/passw", "0123456789abcdeg"] {
            assert!(!is_key(bad), "{bad}");
        }
    }

    #[test]
    fn pending_requests_are_found_under_the_key_the_core_looks_up() {
        let out = run_dir("scan");
        let a = file_request(&out, &request(1));
        let b = file_request(&out, &request(2));
        let dirs = RunDirs::new(&out, None).expect("dirs");

        assert_eq!(dirs.answers, out.canonicalize().expect("canon").join("llm").join("answers"));
        assert!(dirs.answers.is_dir(), "the answers directory is created");

        // A person answered one request by hand; the other waits.
        std::fs::write(dirs.answer_path(&a), "by hand").expect("answer");
        // A temporary file of the core's write_atomic is no request.
        std::fs::write(dirs.requests.join(format!(".{b}.json.1.0.tmp")), "x").expect("tmp");

        let scan = scan(&dirs).expect("scan");

        assert_eq!(scan.answered, 1);
        assert!(scan.refused.is_empty(), "{:?}", scan.refused);
        assert_eq!(scan.pending.len(), 1);
        assert_eq!(scan.pending[0].key, b);
        assert_eq!(scan.pending[0].request, request(2));

        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn a_run_without_requests_has_nothing_pending() {
        let out = run_dir("empty");
        let dirs = RunDirs::new(&out, None).expect("dirs");
        let scan = scan(&dirs).expect("scan");

        assert!(scan.pending.is_empty() && scan.refused.is_empty() && scan.answered == 0);

        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn request_files_that_are_not_under_their_own_key_are_refused() {
        let out = run_dir("refuse");
        let good = file_request(&out, &request(3));
        let dirs = RunDirs::new(&out, None).expect("dirs");
        let req = &dirs.requests;

        // The request of seed 4 under another key, a name that is no key, broken JSON.
        std::fs::write(req.join("0000000000000000.json"), serde_json::to_string(&request(4)).expect("json"))
            .expect("w");
        std::fs::write(req.join("notes.json"), "{}").expect("w");
        std::fs::write(req.join("1111111111111111.json"), "{not json").expect("w");
        // A symlink to a file outside the run is not read.
        let outside = std::env::temp_dir().join(format!("brut-mutator-outside-{}.json", std::process::id()));
        std::fs::write(&outside, serde_json::to_string(&request(5)).expect("json")).expect("w");
        std::os::unix::fs::symlink(&outside, req.join(format!("{}.json", request_key(&request(5))))).expect("link");
        // Files of other kinds are ignored.
        std::fs::write(req.join("README.txt"), "x").expect("w");

        let scan = scan(&dirs).expect("scan");
        let reasons: Vec<String> = scan.refused.iter().map(|r| r.reason.clone()).collect();

        assert_eq!(scan.pending.len(), 1);
        assert_eq!(scan.pending[0].key, good);
        assert_eq!(scan.refused.len(), 4, "{reasons:?}");
        assert!(reasons.iter().any(|r| r.contains(&request_key(&request(4)))), "{reasons:?}");
        assert!(reasons.iter().any(|r| r == "the name is no request key"), "{reasons:?}");
        assert!(reasons.iter().any(|r| r.starts_with("not a request")), "{reasons:?}");
        assert!(reasons.iter().any(|r| r == "not a regular file"), "{reasons:?}");

        let _ = std::fs::remove_file(&outside);
        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn an_answer_is_written_once_and_never_overwritten() {
        let out = run_dir("write");
        let dirs = RunDirs::new(&out, None).expect("dirs");
        let key = request_key(&request(1));

        assert_eq!(write_answer(&dirs.answers, &key, "first").expect("w"), Written::New);
        assert_eq!(write_answer(&dirs.answers, &key, "second").expect("w"), Written::Kept);
        assert_eq!(std::fs::read_to_string(dirs.answer_path(&key)).expect("r"), "first");

        // A dangling symlink under the name is not followed and not replaced.
        let other = request_key(&request(2));
        let target = std::env::temp_dir().join(format!("brut-mutator-nowhere-{}", std::process::id()));
        std::os::unix::fs::symlink(&target, dirs.answer_path(&other)).expect("link");
        assert_eq!(write_answer(&dirs.answers, &other, "x").expect("w"), Written::Kept);
        assert!(!target.exists());

        // A name that is no key never becomes a path.
        assert!(write_answer(&dirs.answers, "../../escape", "x").is_err());

        // No temporary file is left behind.
        let names: Vec<String> = std::fs::read_dir(&dirs.answers)
            .expect("ls")
            .map(|e| e.expect("e").file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names.len(), 2, "{names:?}");

        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn paths_outside_the_run_directory_are_refused() {
        let out = run_dir("paths");
        let elsewhere = std::env::temp_dir().join(format!("brut-mutator-elsewhere-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&elsewhere);
        std::fs::create_dir_all(&elsewhere).expect("mkdir");

        // An answers directory inside the run is fine, as in the core's own test.
        let inside = RunDirs::new(&out, Some(&out.join("my-answers"))).expect("inside");
        assert!(inside.answers.ends_with("my-answers") && inside.answers.is_dir());

        // Outside, through '..', or the run directory itself.
        assert!(RunDirs::new(&out, Some(&elsewhere)).is_err());
        assert!(RunDirs::new(&out, Some(&out.join("..").join("x"))).is_err());
        assert!(RunDirs::new(&out, Some(&out.join("llm").join("..").join("a"))).is_err());
        assert!(RunDirs::new(&out, Some(&out)).is_err());
        assert!(RunDirs::new(&out, Some(Path::new("/tmp/answers"))).is_err());

        // Through a symlink inside the run that leads out.
        std::os::unix::fs::symlink(&elsewhere, out.join("link")).expect("link");
        assert!(RunDirs::new(&out, Some(&out.join("link"))).is_err());
        assert!(RunDirs::new(&out, Some(&out.join("link").join("deeper"))).is_err());
        assert!(!elsewhere.join("deeper").exists(), "nothing created outside");

        // A run whose llm directory is a symlink out is refused as a whole.
        let linked = run_dir("paths-llm");
        std::os::unix::fs::symlink(&elsewhere, linked.join("llm")).expect("link");
        assert!(RunDirs::new(&linked, None).is_err());

        // A directory without config.json holds no run.
        assert!(RunDirs::new(&elsewhere, None).is_err());

        let _ = std::fs::remove_dir_all(&out);
        let _ = std::fs::remove_dir_all(&linked);
        let _ = std::fs::remove_dir_all(&elsewhere);
    }

    #[test]
    fn a_round_answers_every_pending_request_and_the_core_finds_the_answers() {
        let out = run_dir("round");
        let reqs: Vec<GenRequest> = (10..16).map(request).collect();
        let keys: Vec<String> = reqs.iter().map(|r| file_request(&out, r)).collect();
        let dirs = RunDirs::new(&out, None).expect("dirs");
        let todo = scan(&dirs).expect("scan").pending;
        let backend = Recorded::open(Counter(AtomicUsize::new(0)), &dirs.record).expect("open");
        let round = answer(&backend, &dirs.answers, &todo, 3, &never);

        assert_eq!(round.written.len(), 6);
        assert!(round.failed.is_empty() && round.kept.is_empty() && round.skipped == 0);
        assert_eq!(backend.len(), 6, "every call is recorded");

        // The core's own lookup (AnswerBook, keyed by request_key) finds each answer.
        let mut book = AnswerBook::new();

        for key in &keys {
            book.insert(key, std::fs::read_to_string(dirs.answer_path(key)).expect("answer"));
        }

        for req in &reqs {
            let got = book.answer(req).expect("found");
            assert!(got.text.ends_with(&format!("to seed {}", req.seed)), "{}", got.text);
        }

        // Every request and its full answer are in the record.
        let lines = std::fs::read_to_string(&dirs.record).expect("record");
        let recorded: Vec<crate::CallRecord> =
            lines.lines().map(|l| serde_json::from_str(l).expect("json")).collect();
        assert_eq!(recorded.len(), 6);
        assert!(recorded.iter().all(|c| reqs.contains(&c.request)));

        // Nothing is pending afterwards.
        let again = scan(&dirs).expect("scan");
        assert!(again.pending.is_empty());
        assert_eq!(again.answered, 6);

        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn a_repeated_round_replays_the_record_without_a_new_call() {
        let out = run_dir("replay");
        let key = file_request(&out, &request(20));
        let dirs = RunDirs::new(&out, None).expect("dirs");
        let first = Recorded::open(Counter(AtomicUsize::new(0)), &dirs.record).expect("open");
        let todo = scan(&dirs).expect("scan").pending;

        answer(&first, &dirs.answers, &todo, 1, &never);

        let text = std::fs::read_to_string(dirs.answer_path(&key)).expect("answer");

        // The answer file is lost; a new process answers from the record.
        std::fs::remove_file(dirs.answer_path(&key)).expect("rm");

        let counter = Counter(AtomicUsize::new(100));
        let second = Recorded::open(counter, &dirs.record).expect("reopen");
        let todo = scan(&dirs).expect("scan").pending;

        assert!(second.contains(&todo[0].request));

        let round = answer(&second, &dirs.answers, &todo, 1, &never);

        assert_eq!(round.written, vec![key.clone()]);
        assert_eq!(std::fs::read_to_string(dirs.answer_path(&key)).expect("answer"), text);
        assert_eq!(second.len(), 1);

        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn a_failed_call_writes_no_answer_and_is_filed_as_a_mutator_error() {
        let out = run_dir("broken");
        let key = file_request(&out, &request(30));
        let dirs = RunDirs::new(&out, None).expect("dirs");
        let backend = Recorded::open(Broken, &dirs.record).expect("open");
        let todo = scan(&dirs).expect("scan").pending;
        let round = answer(&backend, &dirs.answers, &todo, 2, &never);

        assert_eq!(round.failed.len(), 1);
        assert_eq!(round.failed[0].0, key);
        assert!(round.failed[0].1.contains("HTTP 503"));
        assert!(round.written.is_empty());
        assert!(!dirs.answer_path(&key).exists());
        assert!(!dirs.record.exists(), "a failure is never filed as an answer");
        assert!(dirs.record.with_file_name("mutator-errors.jsonl").is_file());
        assert_eq!(scan(&dirs).expect("scan").pending.len(), 1, "it stays pending");

        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn a_cut_off_answer_is_written_as_it_came_and_reported() {
        let out = run_dir("cut");
        let key = file_request(&out, &request(99));
        let dirs = RunDirs::new(&out, None).expect("dirs");
        let todo = scan(&dirs).expect("scan").pending;
        let round = answer(&Counter(AtomicUsize::new(0)), &dirs.answers, &todo, 1, &never);

        assert_eq!(round.written, vec![key.clone()]);
        assert_eq!(round.cut, vec![key.clone()]);
        assert_eq!(std::fs::read_to_string(dirs.answer_path(&key)).expect("answer"), "answer 0 to seed 99");

        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn an_answer_that_appears_meanwhile_is_kept() {
        let out = run_dir("race");
        let key = file_request(&out, &request(40));
        let dirs = RunDirs::new(&out, None).expect("dirs");
        let todo = scan(&dirs).expect("scan").pending;

        // A person writes the answer after the scan, before the call returns.
        std::fs::write(dirs.answer_path(&key), "by hand").expect("w");

        let round = answer(&Counter(AtomicUsize::new(0)), &dirs.answers, &todo, 1, &never);

        assert_eq!(round.kept, vec![key.clone()]);
        assert!(round.written.is_empty());
        assert_eq!(std::fs::read_to_string(dirs.answer_path(&key)).expect("r"), "by hand");

        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn the_stop_signal_stops_before_the_next_call() {
        let out = run_dir("stop");

        for s in 50..54 {
            file_request(&out, &request(s));
        }

        let dirs = RunDirs::new(&out, None).expect("dirs");
        let todo = scan(&dirs).expect("scan").pending;
        let backend = Counter(AtomicUsize::new(0));

        // Stop after the first answer is on disk.
        let answers = dirs.answers.clone();
        let stop = move || std::fs::read_dir(&answers).map(|d| d.count() > 0).unwrap_or(false);
        let round = answer(&backend, &dirs.answers, &todo, 1, &stop);

        assert_eq!(round.written.len(), 1);
        assert_eq!(round.skipped, 3);
        assert_eq!(backend.0.load(Ordering::SeqCst), 1);

        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn a_second_writer_is_refused_by_the_run_lock() {
        let out = run_dir("lock");
        let dirs = RunDirs::new(&out, None).expect("dirs");
        let held = lock_run(&dirs).expect("first");
        let second = lock_run(&dirs).expect_err("second");

        assert!(second.to_string().contains("another process"), "{second}");
        drop(held);
        assert!(lock_run(&dirs).is_ok(), "free again once the first is gone");

        let _ = std::fs::remove_dir_all(&out);
    }
}
