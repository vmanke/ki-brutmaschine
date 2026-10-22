//! Global hash-chained query ledger of the learner evaluator (evaluator side).
//!
//! One JSON line per entry in `<secret dir>/ledger.jsonl`. Each entry stores the hash of the
//! previous one, so any edit, deletion or reordering inside the file breaks the chain.
//! Removing entries from the end would leave a valid chain and refill the query budget, so a
//! head anchor (`ledger.head`: entry count and last hash) is written after every entry and
//! checked too. Whoever can rewrite both files (the same Unix user) can still forge a shorter
//! history; this is process separation, not a security boundary.
//!
//! Callers hold the evaluator lock (`eval.lock`) from `open` to the last `append`.
//!
//! Owner: WP4. Contract section 5.14. Does file I/O.
#![forbid(unsafe_code)]

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::hidden_eval::{read_limited_bytes, write_private_atomic};
use crate::learner::types::{EvalSet, Tier};
use crate::util::{BrutError, Result, sha256_hex};

/// File name of the ledger inside the secret directory.
pub const LEDGER_FILE: &str = "ledger.jsonl";

/// prev_hash of the first entry.
pub const LEDGER_GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// Largest ledger the evaluator reads (about 100 000 entries). A larger file is refused,
/// so the evaluator fails closed instead of reading without bound.
pub const MAX_LEDGER_BYTES: u64 = 64 << 20;

/// Largest head anchor read.
const MAX_HEAD_BYTES: u64 = 4096;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LedgerKind {
    /// Written under the lock before scoring; counts against the budget.
    Charge { query: QueryKind },
    /// Written after scoring with exactly what was released.
    Release {
        query: QueryKind,
        released: serde_json::Value,
    },
    /// Written when a query was refused (budget, protocol mismatch); does not count.
    Refused { query: QueryKind, reason: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum QueryKind {
    Learner { tier: Tier },
    Arc { set: EvalSet },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LedgerEntry {
    pub seq: u64,
    pub unix_ms: u64,
    pub prev_hash: String,
    pub protocol_hash: String,
    pub run_id: String,
    pub candidate_id: String,
    pub program_sha256: String,
    pub kind: LedgerKind,
    /// SHA-256 of this entry's JSON with hash = "" (same scheme as archive.rs).
    pub hash: String,
}

/// Head anchor next to the ledger: entry count and last hash.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Head {
    entries: u64,
    last_hash: String,
}

/// The verified ledger as read under the evaluator lock.
pub struct Ledger {
    path: PathBuf,
    entries: Vec<LedgerEntry>,
    /// File length when it was read; `append` refuses when the file changed since.
    bytes: u64,
}

impl Ledger {
    /// Open or create <dir>/ledger.jsonl (0600) and verify the whole chain.
    pub fn open(dir: &Path) -> Result<Ledger> {
        let path = dir.join(LEDGER_FILE);
        let head = head_path(&path);

        match std::fs::symlink_metadata(&path) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if std::fs::symlink_metadata(&head).is_ok() {
                    return Err(BrutError::Integrity(format!(
                        "{}: the ledger is missing but its head anchor exists",
                        path.display()
                    )));
                }

                create_empty(&path)?;
            }
            Err(e) => return Err(e.into()),
        }

        let (entries, bytes) = read_and_verify(&path)?;

        Ok(Ledger {
            path,
            entries,
            bytes,
        })
    }

    /// Append with fsync. Caller holds the eval lock.
    ///
    /// `seq`, `prev_hash` and `hash` of `e` are filled in here. Refuses (Invalid) an entry
    /// whose JSON would not parse back to an equal entry, and (Integrity) a ledger file that
    /// changed since it was opened, which means someone wrote without holding the lock.
    pub fn append(&mut self, mut e: LedgerEntry) -> Result<&LedgerEntry> {
        let on_disk = std::fs::symlink_metadata(&self.path)?.len();

        if on_disk != self.bytes {
            return Err(BrutError::Integrity(format!(
                "{}: the ledger changed since it was opened (writer without eval.lock?)",
                self.path.display()
            )));
        }

        e.seq = self.entries.len() as u64;
        e.prev_hash = match self.entries.last() {
            Some(last) => last.hash.clone(),
            None => LEDGER_GENESIS.to_string(),
        };
        e.hash = String::new();

        let unhashed = serde_json::to_string(&e)?;
        let parsed: std::result::Result<LedgerEntry, _> = serde_json::from_str(&unhashed);

        if parsed.ok().as_ref() != Some(&e) {
            return Err(BrutError::Invalid(
                "ledger: entry does not survive a JSON round trip".into(),
            ));
        }

        e.hash = sha256_hex(unhashed.as_bytes());

        let mut line = serde_json::to_string(&e)?;

        line.push('\n');

        let mut file = std::fs::OpenOptions::new().append(true).open(&self.path)?;

        file.write_all(line.as_bytes())?;
        file.sync_all()?;

        self.bytes += line.len() as u64;
        self.entries.push(e);

        let last = self.entries.len() - 1;

        // If this fails, the anchor stays one entry behind, which `open` still accepts.
        write_head(
            &self.path,
            self.entries.len() as u64,
            &self.entries[last].hash,
        )?;

        Ok(&self.entries[last])
    }

    /// Verify a ledger file and its head anchor. Returns the number of entries.
    pub fn verify(path: &Path) -> Result<usize> {
        read_and_verify(path).map(|(entries, _)| entries.len())
    }

    /// Charges for this protocol (all runs) and for this run, per query kind.
    pub fn charges(&self, protocol_hash: &str, run_id: Option<&str>, q: &QueryKind) -> usize {
        self.count_charges(protocol_hash, run_id, |query| query == q)
    }

    /// Total learner charges for this protocol (all tiers) and for one run.
    pub fn learner_charges(&self, protocol_hash: &str, run_id: Option<&str>) -> usize {
        self.count_charges(protocol_hash, run_id, |query| {
            matches!(query, QueryKind::Learner { .. })
        })
    }

    /// All verified entries, oldest first.
    pub fn entries(&self) -> &[LedgerEntry] {
        &self.entries
    }

    fn count_charges(
        &self,
        protocol_hash: &str,
        run_id: Option<&str>,
        wanted: impl Fn(&QueryKind) -> bool,
    ) -> usize {
        self.entries
            .iter()
            .filter(|e| e.protocol_hash == protocol_hash)
            .filter(|e| run_id.is_none_or(|run| e.run_id == run))
            .filter(|e| matches!(&e.kind, LedgerKind::Charge { query } if wanted(query)))
            .count()
    }
}

/// Path of the head anchor that belongs to a ledger file (`ledger.head`).
fn head_path(ledger: &Path) -> PathBuf {
    ledger.with_extension("head")
}

/// Create an empty ledger (0600) and its anchor. Fails if the ledger already exists.
fn create_empty(path: &Path) -> Result<()> {
    let mut options = std::fs::OpenOptions::new();

    options.write(true).create_new(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        options.mode(0o600);
    }

    let file = options.open(path)?;

    file.sync_all()?;

    write_head(path, 0, LEDGER_GENESIS)
}

fn write_head(ledger: &Path, entries: u64, last_hash: &str) -> Result<()> {
    let head = Head {
        entries,
        last_hash: last_hash.to_string(),
    };
    let mut bytes = serde_json::to_vec(&head)?;

    bytes.push(b'\n');

    write_private_atomic(&head_path(ledger), &bytes)
}

/// Read a regular file (no symlink, no device) of at most `limit` bytes.
fn read_regular(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let meta = std::fs::symlink_metadata(path)?;

    if !meta.is_file() {
        return Err(BrutError::Integrity(format!(
            "{} is not a regular file",
            path.display()
        )));
    }

    if meta.len() > limit {
        return Err(BrutError::Integrity(format!(
            "{} is larger than {limit} bytes",
            path.display()
        )));
    }

    let file = std::fs::File::open(path)?;

    read_limited_bytes(file, limit)
}

/// Read every line; check JSON, sequence numbers, the chain, each hash and the head anchor.
///
/// The hash is checked on the exact bytes of the line (hash value blanked), so the check does
/// not depend on float formatting. Errors name the line, never its content.
fn read_and_verify(path: &Path) -> Result<(Vec<LedgerEntry>, u64)> {
    let bytes = read_regular(path, MAX_LEDGER_BYTES)?;
    let integrity = |line: usize, m: &str| {
        BrutError::Integrity(format!("{} line {}: {m}", path.display(), line + 1))
    };

    let Ok(text) = std::str::from_utf8(&bytes) else {
        return Err(BrutError::Integrity(format!(
            "{}: not valid UTF-8",
            path.display()
        )));
    };

    if !text.is_empty() && !text.ends_with('\n') {
        return Err(BrutError::Integrity(format!(
            "{}: last entry is incomplete",
            path.display()
        )));
    }

    let mut entries: Vec<LedgerEntry> = Vec::new();
    let mut prev_hash = LEDGER_GENESIS.to_string();

    for (i, line) in text.lines().enumerate() {
        let entry: LedgerEntry =
            serde_json::from_str(line).map_err(|_| integrity(i, "not a ledger entry"))?;

        if entry.seq != i as u64 {
            return Err(integrity(i, "sequence number out of order"));
        }

        if entry.prev_hash != prev_hash {
            return Err(integrity(i, "prev_hash does not match the previous entry"));
        }

        if !is_hex64(&entry.hash) {
            return Err(integrity(i, "malformed hash"));
        }

        let suffix = format!("\"hash\":\"{}\"}}", entry.hash);
        let Some(body) = line.strip_suffix(&suffix) else {
            return Err(integrity(i, "hash is not the last field"));
        };

        let unhashed = format!("{body}\"hash\":\"\"}}");

        if sha256_hex(unhashed.as_bytes()) != entry.hash {
            return Err(integrity(i, "hash mismatch (entry was changed)"));
        }

        prev_hash = entry.hash.clone();
        entries.push(entry);
    }

    check_head(path, &entries)?;

    Ok((entries, bytes.len() as u64))
}

/// The anchor may be one entry behind (crash between entry and anchor), never ahead and never
/// further behind. A missing anchor is only accepted for an empty ledger.
fn check_head(path: &Path, entries: &[LedgerEntry]) -> Result<()> {
    let head_file = head_path(path);
    let integrity = |m: &str| BrutError::Integrity(format!("{}: {m}", head_file.display()));

    let bytes = match read_regular(&head_file, MAX_HEAD_BYTES) {
        Ok(bytes) => bytes,
        Err(BrutError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
            if entries.is_empty() {
                return Ok(());
            }

            return Err(integrity(
                "head anchor is missing; entries may have been removed from the end",
            ));
        }
        Err(e) => return Err(e),
    };

    let head: Head = serde_json::from_slice(&bytes).map_err(|_| integrity("bad head anchor"))?;
    let len = entries.len() as u64;

    if head.entries > len {
        return Err(integrity(
            "the anchor expects more entries: entries were removed from the end",
        ));
    }

    if len - head.entries > 1 {
        return Err(integrity(
            "more than one entry was added without updating the anchor",
        ));
    }

    let expected = match head.entries.checked_sub(1) {
        None => Some(LEDGER_GENESIS),
        Some(i) => entries.get(i as usize).map(|e| e.hash.as_str()),
    };

    if expected != Some(head.last_hash.as_str()) {
        return Err(integrity("the anchor hash does not match the ledger"));
    }

    Ok(())
}

/// 64 lower-case hex characters.
pub(crate) fn is_hex64(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hidden_eval::{create_private_dir, lock_exclusive};

    /// Env var that turns the ignored helper test into a child process that charges.
    const CHILD_DIR_ENV: &str = "BRUT_TEST_LEDGER_CHILD_DIR";

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brut-ledger-{name}-{}", std::process::id()));

        let _ = std::fs::remove_dir_all(&dir);
        create_private_dir(&dir).expect("temp dir");

        dir
    }

    fn ph() -> String {
        "a".repeat(64)
    }

    fn query(tier: Tier) -> QueryKind {
        QueryKind::Learner { tier }
    }

    fn entry(run: &str, kind: LedgerKind) -> LedgerEntry {
        LedgerEntry {
            seq: 0,
            unix_ms: 1,
            prev_hash: String::new(),
            protocol_hash: ph(),
            run_id: run.to_string(),
            candidate_id: "c0ffee".into(),
            program_sha256: "0".repeat(64),
            kind,
            hash: String::new(),
        }
    }

    fn charge(run: &str, tier: Tier) -> LedgerEntry {
        entry(run, LedgerKind::Charge { query: query(tier) })
    }

    /// A ledger with 4 entries: charge, release, refused, charge.
    fn sample(name: &str) -> (PathBuf, PathBuf) {
        let dir = tmp_dir(name);
        let mut ledger = Ledger::open(&dir).expect("open");

        ledger
            .append(charge("run-a", Tier::HiddenInstance))
            .expect("charge");
        ledger
            .append(entry(
                "run-a",
                LedgerKind::Release {
                    query: query(Tier::HiddenInstance),
                    released: serde_json::json!({"delta": 0.12, "decision": null}),
                },
            ))
            .expect("release");
        ledger
            .append(entry(
                "run-a",
                LedgerKind::Refused {
                    query: query(Tier::HiddenGrammar),
                    reason: "protocol mismatch".into(),
                },
            ))
            .expect("refused");
        ledger
            .append(charge("run-b", Tier::HiddenGrammar))
            .expect("charge");

        let path = dir.join(LEDGER_FILE);

        (dir, path)
    }

    fn lines(path: &Path) -> Vec<String> {
        std::fs::read_to_string(path)
            .expect("read")
            .lines()
            .map(str::to_string)
            .collect()
    }

    fn write_lines(path: &Path, lines: &[String]) {
        let mut text = lines.join("\n");

        text.push('\n');
        std::fs::write(path, text).expect("write");
    }

    #[test]
    fn chain_verifies_and_counts_charges() {
        let (dir, path) = sample("verify");

        assert_eq!(Ledger::verify(&path).expect("verify"), 4);

        let ledger = Ledger::open(&dir).expect("reopen");

        assert_eq!(ledger.entries().len(), 4);
        assert_eq!(ledger.entries()[0].prev_hash, LEDGER_GENESIS);
        assert_eq!(ledger.entries()[1].prev_hash, ledger.entries()[0].hash);

        // Refused and Release entries never count.
        assert_eq!(ledger.learner_charges(&ph(), None), 2);
        assert_eq!(ledger.learner_charges(&ph(), Some("run-a")), 1);
        assert_eq!(ledger.learner_charges(&ph(), Some("run-c")), 0);
        assert_eq!(ledger.charges(&ph(), None, &query(Tier::HiddenGrammar)), 1);
        assert_eq!(
            ledger.charges(&ph(), Some("run-a"), &query(Tier::HiddenGrammar)),
            0
        );
        assert_eq!(
            ledger.charges(
                &ph(),
                None,
                &QueryKind::Arc {
                    set: EvalSet::A1Eval
                }
            ),
            0
        );

        // Another protocol has its own count.
        assert_eq!(ledger.learner_charges(&"b".repeat(64), None), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn entry_json_has_the_documented_shape() {
        let (dir, path) = sample("shape");
        let first: serde_json::Value = serde_json::from_str(&lines(&path)[0]).expect("json");

        assert_eq!(first["kind"]["type"], "charge");
        assert_eq!(first["kind"]["query"]["kind"], "learner");
        assert_eq!(first["kind"]["query"]["tier"], "hidden_instance");
        assert!(is_hex64(first["hash"].as_str().expect("hash")));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn new_ledger_is_private() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tmp_dir("mode");

        Ledger::open(&dir).expect("open");

        let mode = std::fs::metadata(dir.join(LEDGER_FILE))
            .expect("meta")
            .permissions()
            .mode();

        assert_eq!(mode & 0o777, 0o600);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn edited_entry_is_detected() {
        let (dir, path) = sample("edit");
        let mut l = lines(&path);

        // Turn a charge into a refusal: the budget would be refilled.
        l[0] = l[0].replace(
            "\"type\":\"charge\"",
            "\"type\":\"refused\",\"reason\":\"x\"",
        );
        write_lines(&path, &l);

        assert!(matches!(
            Ledger::verify(&path),
            Err(BrutError::Integrity(_))
        ));
        assert!(matches!(Ledger::open(&dir), Err(BrutError::Integrity(_))));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deleted_entry_is_detected() {
        let (dir, path) = sample("delete");
        let mut l = lines(&path);

        l.remove(0);
        write_lines(&path, &l);

        assert!(matches!(
            Ledger::verify(&path),
            Err(BrutError::Integrity(_))
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reordered_entries_are_detected() {
        let (dir, path) = sample("reorder");
        let mut l = lines(&path);

        l.swap(1, 2);
        write_lines(&path, &l);

        assert!(matches!(
            Ledger::verify(&path),
            Err(BrutError::Integrity(_))
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncated_tail_and_missing_files_are_detected() {
        let (dir, path) = sample("truncate");
        let l = lines(&path);

        // Removing the last charge would refill the budget; the anchor catches it.
        write_lines(&path, &l[..3]);

        let err = Ledger::verify(&path).expect_err("truncated");

        assert!(err.to_string().contains("removed from the end"), "{err}");

        // Deleting the ledger but keeping its anchor is refused too.
        std::fs::remove_file(&path).expect("remove");

        assert!(matches!(Ledger::open(&dir), Err(BrutError::Integrity(_))));

        // Deleting the anchor of a non-empty ledger is refused.
        write_lines(&path, &l);
        std::fs::remove_file(head_path(&path)).expect("remove head");

        assert!(matches!(
            Ledger::verify(&path),
            Err(BrutError::Integrity(_))
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn incomplete_last_line_is_detected() {
        let (dir, path) = sample("partial");
        let mut text = std::fs::read_to_string(&path).expect("read");

        text.pop();
        std::fs::write(&path, text).expect("write");

        assert!(matches!(
            Ledger::verify(&path),
            Err(BrutError::Integrity(_))
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn append_refuses_a_ledger_changed_behind_its_back() {
        let dir = tmp_dir("stale");
        let mut first = Ledger::open(&dir).expect("open");
        let mut second = Ledger::open(&dir).expect("open");

        first
            .append(charge("run", Tier::HiddenInstance))
            .expect("first");

        // `second` was read before `first` appended: appending would fork the chain.
        assert!(matches!(
            second.append(charge("run", Tier::HiddenInstance)),
            Err(BrutError::Integrity(_))
        ));
        assert_eq!(Ledger::verify(&dir.join(LEDGER_FILE)).expect("verify"), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn errors_do_not_quote_entry_content() {
        let (dir, path) = sample("quote");
        let mut l = lines(&path);

        l[1] = "{\"seq\":1,\"private-content\":true}".into();
        write_lines(&path, &l);

        let err = Ledger::verify(&path).expect_err("bad line");

        assert!(!err.to_string().contains("private-content"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_ledger_is_refused() {
        let dir = tmp_dir("symlink");

        std::os::unix::fs::symlink("/dev/zero", dir.join(LEDGER_FILE)).expect("symlink");

        assert!(matches!(Ledger::open(&dir), Err(BrutError::Integrity(_))));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Charge under the lock while fewer than `cap` charges exist. Returns true when charged.
    fn charge_with_cap(dir: &Path, cap: usize) -> Result<bool> {
        let _lock = lock_exclusive(&dir.join("eval.lock"))?;
        let mut ledger = Ledger::open(dir)?;

        if ledger.learner_charges(&ph(), None) >= cap {
            return Ok(false);
        }

        ledger.append(charge("run", Tier::HiddenGrammar))?;

        Ok(true)
    }

    #[test]
    fn concurrent_threads_charge_exactly() {
        let dir = tmp_dir("threads");
        let cap = 7;

        let charged: usize = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..12)
                .map(|_| scope.spawn(|| charge_with_cap(&dir, cap)))
                .collect();

            handles
                .into_iter()
                .map(|h| match h.join() {
                    Ok(Ok(true)) => 1,
                    Ok(Ok(false)) => 0,
                    other => panic!("charge failed: {other:?}"),
                })
                .sum()
        });

        assert_eq!(charged, cap);
        assert_eq!(Ledger::verify(&dir.join(LEDGER_FILE)).expect("verify"), cap);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Child-process half of `concurrent_processes_charge_exactly`: does work only when the
    /// parent test starts this test binary with `CHILD_DIR_ENV` set.
    #[test]
    #[ignore = "helper: started as a child process by concurrent_processes_charge_exactly"]
    fn child_process_charges() {
        let Ok(dir) = std::env::var(CHILD_DIR_ENV) else {
            return;
        };

        for _ in 0..10 {
            charge_with_cap(Path::new(&dir), 25).expect("child charge");
        }
    }

    #[test]
    fn concurrent_processes_charge_exactly() {
        let dir = tmp_dir("processes");
        let exe = std::env::current_exe().expect("test binary");

        let children: Vec<_> = (0..4)
            .map(|_| {
                std::process::Command::new(&exe)
                    .args([
                        "--exact",
                        "ledger::tests::child_process_charges",
                        "--ignored",
                        "--test-threads=1",
                    ])
                    .env(CHILD_DIR_ENV, &dir)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                    .expect("spawn child")
            })
            .collect();

        for mut child in children {
            assert!(child.wait().expect("wait").success());
        }

        // 4 processes x 10 attempts against a cap of 25: exactly 25 charges, chain intact.
        let ledger = Ledger::open(&dir).expect("open");

        assert_eq!(ledger.learner_charges(&ph(), None), 25);
        assert_eq!(Ledger::verify(&dir.join(LEDGER_FILE)).expect("verify"), 25);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
