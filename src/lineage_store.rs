//! Lineage chunks and population snapshots of a learner run (orchestrator side).
//!
//! Owner: WP5. Contract section 5.18. Does file I/O.
//!
//! Every file is written atomically and returned with its SHA-256. A `BatchDigest` archive
//! record anchors those hashes, so `Archive::open` detects any later edit.
#![forbid(unsafe_code)]

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::hidden_eval::read_limited_bytes;
use crate::learner::fec::CacheSnapshot;
use crate::learner::types::{LearnerGenome, Mutation};
use crate::meta_evolve::{LINEAGE_DIR, SNAPSHOT_DIR};
use crate::util::{BrutError, Result, sha256_hex, write_atomic};

/// File name of the score cache inside `SNAPSHOT_DIR` (overwritten at every digest).
pub const CACHE_FILE: &str = "cache-latest.json";

/// The next cache, written before the BatchDigest that anchors it and renamed to
/// `CACHE_FILE` right after. If a crash hits between the two, this file still matches
/// the latest digest and is accepted (and promoted on resume).
pub const CACHE_PENDING_FILE: &str = "cache-pending.json";

/// Largest lineage chunk or snapshot file that is read back.
pub const MAX_STORE_FILE_BYTES: u64 = 256 << 20;

/// Largest cache file that is read back (4 million entries need about 250 MB).
pub const MAX_CACHE_FILE_BYTES: u64 = 512 << 20;

/// One child that entered the population (hurdle passed). ~110 bytes as JSON.
///
/// Initial members are recorded as well: batch 0, slot = member index, `mutation` None,
/// `parent` the seed they were mutated from (None for the unmutated seeds).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineageRecord {
    pub id: String,
    pub parent: Option<String>,
    pub batch: u64,
    pub slot: u16,
    pub mutation: Option<Mutation>,
    /// Fingerprint as 16 hex chars.
    pub fp: String,
    pub fitness: f64,
    /// Panel tasks served from the cache (0..=20).
    pub cached_tasks: u8,
}

/// Per-chunk counters (also copied into BatchDigest).
///
/// `unique_evals` counts task runs that were really executed; `fec_task_hits` counts task
/// scores served without a run (score cache, or an earlier child of the same batch with the
/// same fingerprint). Abort counters count tasks; `cost_rejected` counts children.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChunkStats {
    pub children: u64,
    pub unique_evals: u64,
    pub fec_task_hits: u64,
    pub hurdled: u64,
    pub cost_rejected: u64,
    pub aborts_nonfinite: u64,
    pub aborts_error: u64,
    pub aborts_timeout: u64,
    pub flops: u64,
    /// The part of `flops` spent in periodic Stage C rounds (absent in older archives).
    #[serde(default)]
    pub stage_c_flops: u64,
}

impl ChunkStats {
    /// Add `other` field by field (saturating).
    pub fn add(&mut self, other: &ChunkStats) {
        self.children = self.children.saturating_add(other.children);
        self.unique_evals = self.unique_evals.saturating_add(other.unique_evals);
        self.fec_task_hits = self.fec_task_hits.saturating_add(other.fec_task_hits);
        self.hurdled = self.hurdled.saturating_add(other.hurdled);
        self.cost_rejected = self.cost_rejected.saturating_add(other.cost_rejected);
        self.aborts_nonfinite = self.aborts_nonfinite.saturating_add(other.aborts_nonfinite);
        self.aborts_error = self.aborts_error.saturating_add(other.aborts_error);
        self.aborts_timeout = self.aborts_timeout.saturating_add(other.aborts_timeout);
        self.flops = self.flops.saturating_add(other.flops);
        self.stage_c_flops = self.stage_c_flops.saturating_add(other.stage_c_flops);
    }
}

/// Buffers the lineage records of one chunk until `finish` writes them.
pub struct LineageWriter {
    dir: PathBuf,
    chunk: u64,
    buffer: Vec<LineageRecord>,
}

impl LineageWriter {
    /// `dir` is the run directory; the chunk goes to `<dir>/lineage/chunk-<chunk:06>.jsonl`.
    pub fn new(dir: &Path, chunk: u64) -> Self {
        LineageWriter {
            dir: dir.to_path_buf(),
            chunk,
            buffer: Vec::new(),
        }
    }

    pub fn push(&mut self, r: LineageRecord) {
        self.buffer.push(r);
    }

    /// Records buffered so far.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// True when nothing is buffered.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Atomically write <dir>/lineage/chunk-<chunk:06>.jsonl; return (relative path, sha256).
    ///
    /// Refuses (Invalid) a record with a non-finite fitness (invariant I8).
    pub fn finish(self) -> Result<(String, String)> {
        let mut bytes = Vec::with_capacity(self.buffer.len() * 128);

        for r in &self.buffer {
            if !r.fitness.is_finite() {
                return Err(BrutError::Invalid(format!(
                    "lineage: record {} has a non-finite fitness",
                    r.id
                )));
            }

            serde_json::to_writer(&mut bytes, r)?;
            bytes.push(b'\n');
        }

        let rel = chunk_rel_path(self.chunk);

        std::fs::create_dir_all(self.dir.join(LINEAGE_DIR))?;
        write_atomic(&self.dir.join(&rel), &bytes)?;

        Ok((rel, sha256_hex(&bytes)))
    }
}

/// Relative path of chunk `chunk`: `lineage/chunk-<chunk:06>.jsonl`.
pub fn chunk_rel_path(chunk: u64) -> String {
    format!("{LINEAGE_DIR}/chunk-{chunk:06}.jsonl")
}

/// Relative path of the snapshot after `batch` batches: `snapshots/pop-<batch:09>.json`.
pub fn snapshot_rel_path(batch: u64) -> String {
    format!("{SNAPSHOT_DIR}/pop-{batch:09}.json")
}

/// Relative path of the score cache file: `snapshots/cache-latest.json`.
pub fn cache_rel_path() -> String {
    format!("{SNAPSHOT_DIR}/{CACHE_FILE}")
}

/// Relative path of the pending cache file: `snapshots/cache-pending.json`.
pub fn cache_pending_rel_path() -> String {
    format!("{SNAPSHOT_DIR}/{CACHE_PENDING_FILE}")
}

/// Read a chunk, check its SHA-256 and parse every line (unknown fields are rejected).
pub fn read_chunk(path: &Path, expected_sha256: &str) -> Result<Vec<LineageRecord>> {
    let bytes = read_verified(path, expected_sha256, MAX_STORE_FILE_BYTES)?;
    let text = std::str::from_utf8(&bytes).map_err(|_| integrity(path, "not UTF-8"))?;
    let mut out = Vec::new();

    for (i, line) in text.lines().enumerate() {
        let record: LineageRecord = serde_json::from_str(line)
            .map_err(|e| integrity(path, &format!("line {}: bad JSON: {e}", i + 1)))?;

        out.push(record);
    }

    Ok(out)
}

/// One population or hall-of-fame member as stored in a snapshot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PopMember {
    pub id: String,
    pub genome: LearnerGenome,
    pub fp: String,
    pub age: u64,
    pub fitness: f64,
    /// Per panel position: AULC (needed to re-score only rotated tasks).
    pub task_scores: Vec<f32>,
}

/// Search state after `batch` batches: everything a resume needs except the score cache.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    /// Batches done (the next batch to run has this index).
    pub batch: u64,
    pub rotation: u64,
    pub next_age: u64,
    /// Oldest member first.
    pub population: Vec<PopMember>,
    /// Best member first.
    pub hall_of_fame: Vec<PopMember>,
    pub hurdle_values: Vec<f64>,
    /// Search-phase FLOPs charged so far.
    pub flops_used: u64,
    /// Periodic Stage C rounds done.
    pub select_round: u64,
}

impl Snapshot {
    /// Invariant I8: every number is finite.
    fn check_finite(&self) -> Result<()> {
        let members = self.population.iter().chain(&self.hall_of_fame);

        for m in members {
            let finite = m.fitness.is_finite() && m.task_scores.iter().all(|x| x.is_finite());

            if !finite {
                return Err(BrutError::Invalid(format!(
                    "snapshot: member {} has a non-finite score",
                    m.id
                )));
            }
        }

        if !self.hurdle_values.iter().all(|x| x.is_finite()) {
            return Err(BrutError::Invalid(
                "snapshot: non-finite hurdle value".into(),
            ));
        }

        Ok(())
    }
}

/// Atomically write <dir>/snapshots/pop-<batch:09>.json; return (relative path, sha256).
pub fn write_snapshot(dir: &Path, s: &Snapshot) -> Result<(String, String)> {
    s.check_finite()?;

    let mut bytes = serde_json::to_vec(s)?;

    bytes.push(b'\n');

    let rel = snapshot_rel_path(s.batch);

    std::fs::create_dir_all(dir.join(SNAPSHOT_DIR))?;
    write_atomic(&dir.join(&rel), &bytes)?;

    Ok((rel, sha256_hex(&bytes)))
}

/// Read a snapshot and check its SHA-256.
pub fn read_snapshot(path: &Path, expected_sha256: &str) -> Result<Snapshot> {
    let bytes = read_verified(path, expected_sha256, MAX_STORE_FILE_BYTES)?;

    serde_json::from_slice(&bytes).map_err(|e| integrity(path, &format!("bad JSON: {e}")))
}

/// Overwrite <dir>/snapshots/cache-latest.json atomically; return sha256.
pub fn write_cache(dir: &Path, c: &CacheSnapshot) -> Result<String> {
    write_cache_as(dir, c, &cache_rel_path())
}

/// Write <dir>/snapshots/cache-pending.json atomically; return sha256 (see `promote_cache`).
pub fn write_cache_pending(dir: &Path, c: &CacheSnapshot) -> Result<String> {
    write_cache_as(dir, c, &cache_pending_rel_path())
}

/// Rename cache-pending.json to cache-latest.json (atomic rename). No-op without it.
pub fn promote_cache(dir: &Path) -> Result<()> {
    let pending = dir.join(cache_pending_rel_path());

    if pending.is_file() {
        std::fs::rename(&pending, dir.join(cache_rel_path()))?;
    }

    Ok(())
}

fn write_cache_as(dir: &Path, c: &CacheSnapshot, rel: &str) -> Result<String> {
    if !c.entries.iter().all(|e| e.2.is_finite()) {
        return Err(BrutError::Invalid("cache: non-finite score".into()));
    }

    let mut bytes = Vec::with_capacity(c.entries.len() * 48 + 64);

    serde_json::to_writer(&mut bytes, c)?;
    bytes.push(b'\n');

    std::fs::create_dir_all(dir.join(SNAPSHOT_DIR))?;
    write_atomic(&dir.join(rel), &bytes)?;

    Ok(sha256_hex(&bytes))
}

/// Read the score cache and check its SHA-256.
pub fn read_cache(path: &Path, expected_sha256: &str) -> Result<CacheSnapshot> {
    let bytes = read_verified(path, expected_sha256, MAX_CACHE_FILE_BYTES)?;

    serde_json::from_slice(&bytes).map_err(|e| integrity(path, &format!("bad JSON: {e}")))
}

/// The cache that matches `expected_sha256` in run directory `dir`: cache-latest.json, or
/// cache-pending.json after a crash between digest and rename (then promoted).
pub fn read_run_cache(dir: &Path, expected_sha256: &str) -> Result<CacheSnapshot> {
    let latest = read_cache(&dir.join(cache_rel_path()), expected_sha256);

    if latest.is_ok() {
        return latest;
    }

    let pending = read_cache(&dir.join(cache_pending_rel_path()), expected_sha256);

    match pending {
        Ok(c) => {
            promote_cache(dir)?;

            Ok(c)
        }
        Err(_) => latest,
    }
}

/// Read a regular file of at most `limit` bytes and check its SHA-256 (Integrity error).
pub(crate) fn read_verified(path: &Path, expected_sha256: &str, limit: u64) -> Result<Vec<u8>> {
    let meta =
        std::fs::metadata(path).map_err(|e| integrity(path, &format!("cannot read: {e}")))?;

    if !meta.is_file() {
        return Err(integrity(path, "not a regular file"));
    }

    let file = std::fs::File::open(path)?;
    let bytes = read_limited_bytes(file, limit)?;

    if sha256_hex(&bytes) != expected_sha256 {
        return Err(integrity(path, "SHA-256 mismatch (file was changed)"));
    }

    Ok(bytes)
}

/// True for a non-empty relative path made of plain names only (no `..`, no root).
pub(crate) fn is_safe_relative(rel: &str) -> bool {
    let path = Path::new(rel);

    !rel.is_empty() && path.components().all(|c| matches!(c, Component::Normal(_)))
}

fn integrity(path: &Path, m: &str) -> BrutError {
    BrutError::Integrity(format!("{}: {m}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::types::{Instr, OpCode};

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brut-lineage-{name}-{}", std::process::id()));

        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");

        dir
    }

    fn record(id: &str, parent: Option<&str>, fitness: f64) -> LineageRecord {
        LineageRecord {
            id: id.into(),
            parent: parent.map(str::to_string),
            batch: 1234,
            slot: 17,
            mutation: Some(Mutation::Const),
            fp: "9a8b7c6d5e4f3a2b".into(),
            fitness,
            cached_tasks: 15,
        }
    }

    fn member(id: &str, fitness: f64) -> PopMember {
        let instr = Instr {
            op: OpCode::SConst,
            a: 0,
            b: 0,
            out: 3,
            c: 0.05,
        };

        PopMember {
            id: id.into(),
            genome: LearnerGenome {
                setup: vec![instr],
                predict: Vec::new(),
                learn: Vec::new(),
            },
            fp: "00000000000000ff".into(),
            age: 7,
            fitness,
            task_scores: vec![0.25, 0.5, 0.125],
        }
    }

    fn snapshot(batch: u64) -> Snapshot {
        Snapshot {
            batch,
            rotation: 2,
            next_age: 300,
            population: vec![member("a", 0.3), member("b", -0.1)],
            hall_of_fame: vec![member("b", 0.4)],
            hurdle_values: vec![0.1, 0.2],
            flops_used: 123_456,
            select_round: 1,
        }
    }

    #[test]
    fn chunk_round_trip_and_hash() {
        let dir = tmp_dir("chunk");
        let mut w = LineageWriter::new(&dir, 1);

        assert!(w.is_empty());

        w.push(record("3fa1c2d4e5f60718", Some("0b1c2d3e4f506172"), 0.3412));
        w.push(record("0b1c2d3e4f506172", None, 0.25));

        assert_eq!(w.len(), 2);

        let (rel, sha) = w.finish().expect("finish");

        assert_eq!(rel, "lineage/chunk-000001.jsonl");

        let path = dir.join(&rel);
        let bytes = std::fs::read(&path).expect("read");

        assert_eq!(sha256_hex(&bytes), sha);
        assert_eq!(bytes.iter().filter(|&&b| b == b'\n').count(), 2);

        let back = read_chunk(&path, &sha).expect("read chunk");

        assert_eq!(back.len(), 2);
        assert_eq!(
            back[0],
            record("3fa1c2d4e5f60718", Some("0b1c2d3e4f506172"), 0.3412)
        );
        assert_eq!(back[1].parent, None);

        // A wrong hash, an edited file and unknown fields are refused.
        assert!(matches!(
            read_chunk(&path, &"0".repeat(64)),
            Err(BrutError::Integrity(_))
        ));

        let edited = String::from_utf8(bytes)
            .expect("utf8")
            .replace("0.3412", "0.9");

        std::fs::write(&path, &edited).expect("write");
        assert!(matches!(
            read_chunk(&path, &sha),
            Err(BrutError::Integrity(_))
        ));

        let extra = edited.replacen("\"cached_tasks\"", "\"x\":1,\"cached_tasks\"", 1);

        std::fs::write(&path, &extra).expect("write");
        assert!(read_chunk(&path, &sha256_hex(extra.as_bytes())).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_chunk_and_non_finite_fitness() {
        let dir = tmp_dir("empty");
        let (rel, sha) = LineageWriter::new(&dir, 42).finish().expect("finish");

        assert_eq!(rel, "lineage/chunk-000042.jsonl");
        assert_eq!(sha, sha256_hex(b""));
        assert!(read_chunk(&dir.join(&rel), &sha).expect("read").is_empty());

        let mut w = LineageWriter::new(&dir, 43);

        w.push(record("x", None, f64::NAN));

        assert!(matches!(w.finish(), Err(BrutError::Invalid(_))));
        assert!(!dir.join(chunk_rel_path(43)).exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn snapshot_round_trip() {
        let dir = tmp_dir("snapshot");
        let s = snapshot(1000);
        let (rel, sha) = write_snapshot(&dir, &s).expect("write");

        assert_eq!(rel, "snapshots/pop-000001000.json");
        assert_eq!(read_snapshot(&dir.join(&rel), &sha).expect("read"), s);
        assert!(matches!(
            read_snapshot(&dir.join(&rel), &sha256_hex(b"x")),
            Err(BrutError::Integrity(_))
        ));

        let mut bad = snapshot(1001);

        bad.population[1].task_scores[0] = f32::INFINITY;

        assert!(matches!(
            write_snapshot(&dir, &bad),
            Err(BrutError::Invalid(_))
        ));

        let mut bad = snapshot(1002);

        bad.hurdle_values.push(f64::NAN);

        assert!(write_snapshot(&dir, &bad).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_round_trip_overwrites() {
        let dir = tmp_dir("cache");
        let first = CacheSnapshot {
            capacity: 10,
            entries: vec![(1, 2, 0.5), (u64::MAX, 7, 0.0)],
        };
        let sha1 = write_cache(&dir, &first).expect("write");
        let path = dir.join(cache_rel_path());

        assert_eq!(read_cache(&path, &sha1).expect("read"), first);

        let second = CacheSnapshot {
            capacity: 10,
            entries: vec![(3, 4, 0.75)],
        };
        let sha2 = write_cache(&dir, &second).expect("write");

        assert_ne!(sha1, sha2);
        assert_eq!(read_cache(&path, &sha2).expect("read"), second);
        assert!(read_cache(&path, &sha1).is_err());

        let bad = CacheSnapshot {
            capacity: 1,
            entries: vec![(1, 1, f32::NAN)],
        };

        assert!(write_cache(&dir, &bad).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pending_cache_survives_a_crash_before_the_rename() {
        let dir = tmp_dir("pending");
        let old = CacheSnapshot {
            capacity: 4,
            entries: vec![(1, 1, 0.25)],
        };
        let new = CacheSnapshot {
            capacity: 4,
            entries: vec![(2, 2, 0.5)],
        };
        let sha_old = write_cache(&dir, &old).expect("write");
        let sha_new = write_cache_pending(&dir, &new).expect("pending");

        // Crash before the rename: the latest file is old, the pending one matches.
        assert_eq!(read_run_cache(&dir, &sha_old).expect("old"), old);
        assert_eq!(read_run_cache(&dir, &sha_new).expect("pending"), new);

        // Reading the pending cache promoted it.
        assert!(!dir.join(cache_pending_rel_path()).exists());
        assert_eq!(
            read_cache(&dir.join(cache_rel_path()), &sha_new).expect("latest"),
            new
        );
        assert!(read_run_cache(&dir, &sha_old).is_err());

        promote_cache(&dir).expect("no-op");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verified_reads_need_small_regular_files() {
        let dir = tmp_dir("regular");

        assert!(matches!(
            read_verified(&dir, &sha256_hex(b""), 10),
            Err(BrutError::Integrity(_))
        ));
        assert!(matches!(
            read_verified(&dir.join("missing"), &sha256_hex(b""), 10),
            Err(BrutError::Integrity(_))
        ));

        let path = dir.join("big");

        std::fs::write(&path, [0u8; 32]).expect("write");

        assert!(read_verified(&path, &sha256_hex(&[0u8; 32]), 16).is_err());
        assert!(read_verified(&path, &sha256_hex(&[0u8; 32]), 32).is_ok());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn safe_relative_paths() {
        assert!(is_safe_relative("lineage/chunk-000001.jsonl"));
        assert!(is_safe_relative("snapshots/pop-000001000.json"));
        assert!(!is_safe_relative(""));
        assert!(!is_safe_relative("/etc/passwd"));
        assert!(!is_safe_relative("../x"));
        assert!(!is_safe_relative("lineage/../../x"));
        assert!(!is_safe_relative("./x"));
    }

    #[test]
    fn stats_add_saturates() {
        let mut a = ChunkStats {
            children: 2,
            flops: u64::MAX - 1,
            ..ChunkStats::default()
        };
        let b = ChunkStats {
            children: 3,
            hurdled: 1,
            flops: 5,
            stage_c_flops: 4,
            ..ChunkStats::default()
        };

        a.add(&b);

        assert_eq!(a.children, 5);
        assert_eq!(a.hurdled, 1);
        assert_eq!(a.flops, u64::MAX);
        assert_eq!(a.stage_c_flops, 4);

        // Digests of older archives lack the Stage C counter.
        let old: ChunkStats = serde_json::from_str(
            "{\"children\":1,\"unique_evals\":0,\"fec_task_hits\":0,\"hurdled\":0,\
             \"cost_rejected\":0,\"aborts_nonfinite\":0,\"aborts_error\":0,\
             \"aborts_timeout\":0,\"flops\":7}",
        )
        .expect("old stats");

        assert_eq!((old.flops, old.stage_c_flops), (7, 0));
    }
}
