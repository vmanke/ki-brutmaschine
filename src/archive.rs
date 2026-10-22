//! Lineage archive: append-only JSONL file with a SHA-256 hash chain.
//!
//! Each record stores the hash of the previous record.
//! Any later edit, deletion or reordering inside the file breaks the chain and is detected.
//! Removing records from the end leaves a valid chain, so a separate head anchor
//! (`archive.head`: record count and last hash) is written after every write and checked too.
//! Learner searches may buffer records and write them in groups (`set_buffered`): one write,
//! one fsync and two anchor updates per group instead of per record.
//! Whoever can rewrite both files can still forge a shorter history (no external anchor).
//! This also serves as the automatic event log (cf. EU AI Act Art. 12).

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write;
use std::path::{Path, PathBuf};

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::genome::Genome;
use crate::learner::protocol::ProtocolDescriptor;
use crate::learner::types::{
    Decision, EvalSet, LearnerGenome, LearnerHiddenResult, MetaConfig, Tier,
};
use crate::lineage_store::{
    ChunkStats, MAX_CACHE_FILE_BYTES, MAX_STORE_FILE_BYTES, cache_pending_rel_path, cache_rel_path,
    is_safe_relative, read_verified,
};
use crate::score::Metrics;
use crate::train::TrainStats;
use crate::util::{BrutError, Result, sha256_hex, write_atomic};

/// prev_hash of the first record.
pub const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// Sharpness of the parent-selection sigmoid.
pub const SELECTION_LAMBDA: f64 = 10.0;

/// Number of top candidates whose median fitness is the sigmoid midpoint.
pub const SELECTION_TOP: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandidateStatus {
    Ok,
    Diverged,
    OverBudget,
    Invalid,
    /// Hidden score far below validation score. Not eligible as parent or for approval.
    Quarantined,
    /// Training was cut short by the kill switch. Scored, but not eligible as parent.
    Stopped,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateEntry {
    /// `Genome::id()`.
    pub id: String,
    pub generation: usize,
    /// Parent ids; empty for the initial population.
    pub parents: Vec<String>,
    /// Name of the variant generator that made it ("init" for random genomes).
    pub generator: String,
    pub genome: Genome,
    pub params: usize,
    pub train: TrainStats,
    pub valid: Metrics,
    pub fitness: f64,
    /// Model file path relative to the run directory, if saved.
    pub model_file: Option<String>,
    pub status: CandidateStatus,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RecordKind {
    RunStarted {
        config: Config,
        crate_version: String,
    },
    Candidate(CandidateEntry),
    HiddenEval {
        candidate_id: String,
        valid_primary: f64,
        hidden_primary: f64,
        gap: f64,
        quarantined: bool,
    },
    Approved {
        candidate_id: String,
        by: String,
        note: String,
    },
    SafetyEvent {
        what: String,
    },
    RunFinished {
        reason: String,
    },
    // Learner mode (contract section 5.21). MLP code paths ignore these records.
    MetaRunStarted {
        /// Boxed only to keep RecordKind small (errata E4); the JSON is unchanged.
        config: Box<MetaConfig>,
        protocol_hash: String,
        crate_version: String,
    },
    ProtocolRegistered {
        protocol_hash: String,
        protocol: ProtocolDescriptor,
    },
    LearnerCandidate(LearnerEntry),
    BatchDigest(BatchDigestEntry),
    PanelRotated {
        rotation: u64,
        panel_id: String,
    },
    SelectEval {
        candidate_id: String,
        round: u64,
        select_mean: f64,
        delta_select: f64,
    },
    LearnerHiddenEval {
        candidate_id: String,
        tier: Tier,
        delta: f64,
        decision: Option<Decision>,
        delta_public: Option<f64>,
        quarantined: bool,
    },
    ExternalEval {
        candidate_id: String,
        set: EvalSet,
        official: f64,
        kaggle: f64,
        bca: Option<f64>,
    },
    /// A one-time cost outside the search budget (errata E6), metered in FLOPs.
    LearnerCost {
        phase: CostPhase,
        flops: u64,
    },
}

/// One-time phases of a learner run that `max_total_flops` does not bound (errata E6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostPhase {
    /// The equal-effort tuning of the baselines (once per run directory).
    BaselineTuning,
    /// Final Stage C round, retuning and the finalist metrics (section 6.7).
    FinalPhase,
}

/// Role of a learner program in the archive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearnerStatus {
    Seed,
    HallOfFame,
    Finalist,
    Baseline,
    Quarantined,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearnerEntry {
    pub id: String,
    pub parents: Vec<String>,
    /// "seed:<name>", "aging/<mutation>", "random-search", "retune", "baseline:<id>".
    pub generator: String,
    pub batch: u64,
    pub genome: LearnerGenome,
    pub effective_len: usize,
    pub step_cost_f16: u64,
    pub fp: String,
    pub panel_mean: f64,
    pub fitness: f64,
    /// Relative path under programs/ for Finalist/Baseline; None otherwise.
    pub program_file: Option<String>,
    pub status: LearnerStatus,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchDigestEntry {
    pub first_batch: u64,
    pub last_batch: u64,
    pub chunk_file: String,
    pub chunk_sha256: String,
    pub snapshot_file: String,
    pub snapshot_sha256: String,
    pub cache_sha256: String,
    pub stats: ChunkStats,
    pub best_fitness: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub seq: u64,
    pub unix_ms: u64,
    pub prev_hash: String,
    pub kind: RecordKind,
    /// SHA-256 hex of the canonical JSON of this record with `hash` set to "".
    pub hash: String,
}

/// Latest hidden evaluation of one candidate (from `HiddenEval` records).
#[derive(Clone, Debug, PartialEq)]
pub struct HiddenEvalEntry {
    pub candidate_id: String,
    pub valid_primary: f64,
    pub hidden_primary: f64,
    pub gap: f64,
    pub quarantined: bool,
}

pub struct Archive {
    path: PathBuf,
    records: Vec<Record>,
    /// Candidate id -> index into `records`.
    index: HashMap<String, usize>,
    /// Candidate id -> number of children recorded.
    children: HashMap<String, usize>,
    /// Learner id -> index of its latest `LearnerCandidate` record.
    learner_index: HashMap<String, usize>,
    /// Learner ids in order of first appearance.
    learner_order: Vec<String>,
    /// When true, `append` keeps new records in memory until `flush`.
    buffered: bool,
    /// JSON lines of the records not yet written (the last `pending_count` records).
    pending: String,
    pending_count: u64,
}

impl Archive {
    /// Create a new archive file and its head anchor. Fails if the archive already exists.
    pub fn create(path: &Path) -> Result<Archive> {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;

        write_head(path, 0, GENESIS_HASH, 0)?;

        Ok(Archive::empty(path, 0))
    }

    /// Open an existing archive and verify the whole chain. Integrity error on any mismatch.
    ///
    /// Learner runs: also checks that the lineage chunk and snapshot of every `BatchDigest`
    /// exist and match their SHA-256, and the score cache of the latest digest.
    pub fn open(path: &Path) -> Result<Archive> {
        let records = read_and_verify(path)?;

        verify_digest_files(path, &records)?;

        let mut archive = Archive::empty(path, records.len());

        for record in records {
            archive.push_indexed(record);
        }

        Ok(archive)
    }

    /// Append one record: fill seq, prev_hash, hash; write one JSON line; fsync;
    /// then update the head anchor.
    /// Refuses (Invalid) a record whose JSON would not parse back to an equal record
    /// (for example non-finite floats, which serde_json writes as null).
    pub fn append(&mut self, kind: RecordKind, unix_ms: u64) -> Result<&Record> {
        let prev_hash = match self.records.last() {
            Some(r) => r.hash.clone(),
            None => GENESIS_HASH.to_string(),
        };

        let mut record = Record {
            seq: self.records.len() as u64,
            unix_ms,
            prev_hash,
            kind,
            hash: String::new(),
        };

        let unhashed = serde_json::to_string(&record)?;
        let parsed: std::result::Result<Record, _> = serde_json::from_str(&unhashed);

        if parsed.ok().as_ref() != Some(&record) {
            return Err(BrutError::Invalid(
                "archive: record does not survive a JSON round trip (non-finite number?)".into(),
            ));
        }

        record.hash = sha256_hex(unhashed.as_bytes());

        let mut line = serde_json::to_string(&record)?;

        line.push('\n');

        if self.buffered {
            self.pending.push_str(&line);
            self.pending_count += 1;

            let idx = self.push_indexed(record);

            if self.pending_count >= MAX_PENDING_RECORDS {
                self.flush()?;
            }

            return Ok(&self.records[idx]);
        }

        let mut file = std::fs::OpenOptions::new().append(true).open(&self.path)?;

        file.write_all(line.as_bytes())?;
        file.sync_all()?;

        let idx = self.push_indexed(record);

        // If this fails, the anchor stays one record behind, which `open` still accepts.
        write_head(
            &self.path,
            self.records.len() as u64,
            &self.records[idx].hash,
            0,
        )?;

        Ok(&self.records[idx])
    }

    /// Buffer later appends in memory (true) or write each one at once (false, the default).
    /// Buffered records are visible to every query at once; `flush` writes them. Turning
    /// buffering off flushes first.
    pub fn set_buffered(&mut self, on: bool) -> Result<()> {
        if !on {
            self.flush()?;
        }

        self.buffered = on;

        Ok(())
    }

    /// Write the buffered records as one group: announce the group in the anchor, append all
    /// lines, fsync once, then move the anchor to the last record. A crash in between leaves
    /// an anchor that `open` accepts (up to the announced number of records behind).
    pub fn flush(&mut self) -> Result<()> {
        if self.pending_count == 0 {
            return Ok(());
        }

        let total = self.records.len();
        let committed = total - self.pending_count as usize;
        let committed_hash = match committed.checked_sub(1) {
            Some(i) => self.records[i].hash.clone(),
            None => GENESIS_HASH.to_string(),
        };

        write_head(
            &self.path,
            committed as u64,
            &committed_hash,
            self.pending_count,
        )?;

        // Taken out first: after a failed write nothing is written twice (not even by Drop).
        let lines = std::mem::take(&mut self.pending);

        self.pending_count = 0;

        let mut file = std::fs::OpenOptions::new().append(true).open(&self.path)?;

        file.write_all(lines.as_bytes())?;
        file.sync_all()?;

        write_head(&self.path, total as u64, &self.records[total - 1].hash, 0)
    }

    /// Verify a file, its head anchor and (learner runs) the files its digests anchor,
    /// without keeping it open. Returns the number of records.
    pub fn verify(path: &Path) -> Result<usize> {
        let records = read_and_verify(path)?;

        verify_digest_files(path, &records)?;

        Ok(records.len())
    }

    pub fn records(&self) -> &[Record] {
        &self.records
    }

    /// Path of the archive file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// All candidates in insertion order, with their latest status
    /// (a HiddenEval with quarantined = true turns status into Quarantined).
    pub fn candidates(&self) -> Vec<CandidateEntry> {
        let quarantined = self.quarantined_ids();

        self.records
            .iter()
            .enumerate()
            .filter_map(|(i, r)| match &r.kind {
                RecordKind::Candidate(c) if self.index.get(&c.id) == Some(&i) => {
                    Some(with_status(c, &quarantined))
                }
                _ => None,
            })
            .collect()
    }

    pub fn get(&self, id: &str) -> Option<CandidateEntry> {
        let idx = *self.index.get(id)?;

        match &self.records.get(idx)?.kind {
            RecordKind::Candidate(c) => Some(with_status(c, &self.quarantined_ids())),
            _ => None,
        }
    }

    /// The candidate and all its ancestors, breadth-first, no duplicates.
    pub fn lineage(&self, id: &str) -> Vec<CandidateEntry> {
        let mut out = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();

        queue.push_back(id.to_string());
        seen.insert(id.to_string());

        while let Some(next) = queue.pop_front() {
            let Some(entry) = self.get(&next) else {
                continue;
            };

            for parent in &entry.parents {
                if seen.insert(parent.clone()) {
                    queue.push_back(parent.clone());
                }
            }

            out.push(entry);
        }

        out
    }

    /// Config of the first `RunStarted` record: the experiment this archive belongs to.
    pub fn first_config(&self) -> Option<&Config> {
        self.records.iter().find_map(|r| match &r.kind {
            RecordKind::RunStarted { config, .. } => Some(config),
            _ => None,
        })
    }

    pub fn children_count(&self, id: &str) -> usize {
        self.children.get(id).copied().unwrap_or(0)
    }

    /// Latest hidden evaluation per candidate, in order of first evaluation.
    pub fn hidden_evals(&self) -> Vec<HiddenEvalEntry> {
        let mut out: Vec<HiddenEvalEntry> = Vec::new();
        let mut pos: HashMap<String, usize> = HashMap::new();

        for r in &self.records {
            if let RecordKind::HiddenEval {
                candidate_id,
                valid_primary,
                hidden_primary,
                gap,
                quarantined,
            } = &r.kind
            {
                let entry = HiddenEvalEntry {
                    candidate_id: candidate_id.clone(),
                    valid_primary: *valid_primary,
                    hidden_primary: *hidden_primary,
                    gap: *gap,
                    quarantined: *quarantined,
                };

                match pos.get(candidate_id) {
                    Some(&i) => out[i] = entry,
                    None => {
                        pos.insert(candidate_id.clone(), out.len());
                        out.push(entry);
                    }
                }
            }
        }

        out
    }

    /// Choose `k` parent ids (with replacement) among candidates with status Ok.
    ///
    /// Quarantined candidates are not eligible. In a run this never matters: the orchestrator
    /// refuses new generations once a hidden evaluation exists (`evolve::run`), so hidden
    /// results cannot steer the selection.
    ///
    /// Open-ended selection adapted from the Darwin Gödel Machine
    /// (Zhang et al., ICLR 2026, arXiv:2505.22954):
    /// weight = sigmoid(lambda * (fitness - median_top)) / (1 + children), lambda = 10,
    /// median_top = median fitness of the best min(n, 10) eligible candidates.
    /// Only lambda = 10 and the 1 / (1 + children) term match the DGM. Unlike the DGM, the
    /// sigmoid midpoint is that median (DGM: fixed 0.5), and every child counts, whatever its
    /// status (DGM: only functioning children that entered the archive).
    /// Keeps weak but novel lines alive instead of pure greedy hill-climbing.
    /// Returns an empty vec if no candidate is eligible.
    pub fn select_parents(&self, k: usize, rng: &mut ChaCha8Rng) -> Vec<String> {
        let eligible: Vec<CandidateEntry> = self
            .candidates()
            .into_iter()
            .filter(|c| c.status == CandidateStatus::Ok)
            .collect();

        if eligible.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut sorted: Vec<f64> = eligible.iter().map(|c| c.fitness).collect();

        sorted.sort_by(|a, b| b.total_cmp(a));

        let top = &sorted[..sorted.len().min(SELECTION_TOP)];
        let mid = top.len() / 2;
        let median = if top.len() % 2 == 1 {
            top[mid]
        } else {
            0.5 * (top[mid - 1] + top[mid])
        };

        let weights: Vec<f64> = eligible
            .iter()
            .map(|c| {
                let sig = 1.0 / (1.0 + (-SELECTION_LAMBDA * (c.fitness - median)).exp());
                let w = sig / (1.0 + self.children_count(&c.id) as f64);

                if w.is_finite() && w > 0.0 { w } else { 0.0 }
            })
            .collect();

        let total: f64 = weights.iter().sum();

        (0..k)
            .map(|_| {
                let idx = if total > 0.0 {
                    sample_index(&weights, total, rng)
                } else {
                    rng.random_range(0..eligible.len())
                };

                eligible[idx].id.clone()
            })
            .collect()
    }

    /// An archive object without records.
    fn empty(path: &Path, capacity: usize) -> Archive {
        Archive {
            path: path.to_path_buf(),
            records: Vec::with_capacity(capacity),
            index: HashMap::new(),
            children: HashMap::new(),
            learner_index: HashMap::new(),
            learner_order: Vec::new(),
            buffered: false,
            pending: String::new(),
            pending_count: 0,
        }
    }

    /// Add a record to memory and update the indexes. Returns its position.
    fn push_indexed(&mut self, record: Record) -> usize {
        let idx = self.records.len();

        if let RecordKind::LearnerCandidate(e) = &record.kind
            && self.learner_index.insert(e.id.clone(), idx).is_none()
        {
            self.learner_order.push(e.id.clone());
        }

        if let RecordKind::Candidate(c) = &record.kind {
            let first_time = self.index.insert(c.id.clone(), idx).is_none();

            if first_time {
                let unique: HashSet<&String> = c.parents.iter().collect();

                for parent in unique {
                    *self.children.entry(parent.clone()).or_insert(0) += 1;
                }
            }
        }

        self.records.push(record);

        idx
    }

    /// Ids whose latest hidden evaluation set `quarantined`.
    fn quarantined_ids(&self) -> HashSet<String> {
        self.hidden_evals()
            .into_iter()
            .filter(|e| e.quarantined)
            .map(|e| e.candidate_id)
            .collect()
    }
}

/// Learner-mode queries (contract section 5.21).
impl Archive {
    /// Config of the experiment this archive belongs to: the latest `MetaRunStarted` before
    /// the search started (first `LearnerCandidate` or `BatchDigest`). Before that, meta.json
    /// may still be edited, as config.json before the first MLP run.
    pub fn meta_config(&self) -> Option<&MetaConfig> {
        let mut config = None;

        for r in &self.records {
            match &r.kind {
                RecordKind::MetaRunStarted { config: c, .. } => config = Some(c.as_ref()),
                RecordKind::LearnerCandidate(_) | RecordKind::BatchDigest(_) => break,
                _ => {}
            }
        }

        config
    }

    /// True once a learner search wrote its first result (a candidate or a digest).
    pub fn learner_search_started(&self) -> bool {
        self.records.iter().any(|r| {
            matches!(
                r.kind,
                RecordKind::LearnerCandidate(_) | RecordKind::BatchDigest(_)
            )
        })
    }

    /// Config of the latest `MetaRunStarted` record (one per `brut run` of a learner run).
    pub fn latest_meta_config(&self) -> Option<&MetaConfig> {
        self.records.iter().rev().find_map(|r| match &r.kind {
            RecordKind::MetaRunStarted { config, .. } => Some(config.as_ref()),
            _ => None,
        })
    }

    /// Protocol hash registered by the first `MetaRunStarted` record.
    pub fn protocol_hash(&self) -> Option<&str> {
        self.records.iter().find_map(|r| match &r.kind {
            RecordKind::MetaRunStarted { protocol_hash, .. } => Some(protocol_hash.as_str()),
            _ => None,
        })
    }

    /// Latest entry per learner id, with Quarantined applied from LearnerHiddenEval.
    /// Ids come in order of first appearance.
    pub fn learner_candidates(&self) -> Vec<LearnerEntry> {
        let quarantined = self.learner_quarantined_ids();

        self.learner_order
            .iter()
            .filter_map(|id| self.learner_entry(id, &quarantined))
            .collect()
    }

    /// Latest entry of one learner id, with Quarantined applied.
    pub fn learner_get(&self, id: &str) -> Option<LearnerEntry> {
        self.learner_entry(id, &self.learner_quarantined_ids())
    }

    pub fn last_batch_digest(&self) -> Option<&BatchDigestEntry> {
        self.records.iter().rev().find_map(|r| match &r.kind {
            RecordKind::BatchDigest(d) => Some(d),
            _ => None,
        })
    }

    /// All batch digests in archive order.
    pub fn batch_digests(&self) -> Vec<&BatchDigestEntry> {
        self.records
            .iter()
            .filter_map(|r| match &r.kind {
                RecordKind::BatchDigest(d) => Some(d),
                _ => None,
            })
            .collect()
    }

    /// Latest hidden result per (candidate, tier), in order of first evaluation.
    pub fn learner_hidden_evals(&self) -> Vec<LearnerHiddenResult> {
        let mut out: Vec<LearnerHiddenResult> = Vec::new();
        let mut pos: HashMap<(String, Tier), usize> = HashMap::new();

        for r in &self.records {
            let RecordKind::LearnerHiddenEval {
                candidate_id,
                tier,
                delta,
                decision,
                quarantined,
                ..
            } = &r.kind
            else {
                continue;
            };

            let entry = LearnerHiddenResult {
                candidate_id: candidate_id.clone(),
                tier: *tier,
                delta: *delta,
                decision: *decision,
                quarantined: *quarantined,
            };
            let key = (candidate_id.clone(), *tier);

            match pos.get(&key) {
                Some(&i) => out[i] = entry,
                None => {
                    pos.insert(key, out.len());
                    out.push(entry);
                }
            }
        }

        out
    }

    /// Hash of record 0 (used as run_id).
    pub fn run_id(&self) -> Option<&str> {
        self.records.first().map(|r| r.hash.as_str())
    }

    /// Ids with any `LearnerHiddenEval` that set `quarantined` (quarantine is final).
    fn learner_quarantined_ids(&self) -> HashSet<&str> {
        self.records
            .iter()
            .filter_map(|r| match &r.kind {
                RecordKind::LearnerHiddenEval {
                    candidate_id,
                    quarantined: true,
                    ..
                } => Some(candidate_id.as_str()),
                _ => None,
            })
            .collect()
    }

    fn learner_entry(&self, id: &str, quarantined: &HashSet<&str>) -> Option<LearnerEntry> {
        let idx = *self.learner_index.get(id)?;

        let RecordKind::LearnerCandidate(e) = &self.records.get(idx)?.kind else {
            return None;
        };

        let mut entry = e.clone();

        if quarantined.contains(id) {
            entry.status = LearnerStatus::Quarantined;
        }

        Some(entry)
    }
}

fn with_status(c: &CandidateEntry, quarantined: &HashSet<String>) -> CandidateEntry {
    let mut entry = c.clone();

    if quarantined.contains(&entry.id) {
        entry.status = CandidateStatus::Quarantined;
    }

    entry
}

/// Best effort: records still buffered when the archive goes away (an error path or a panic)
/// are written. Errors cannot be reported here; the next `open` checks the result.
impl Drop for Archive {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// Most records a buffered archive holds before it writes them on its own.
const MAX_PENDING_RECORDS: u64 = 4_096;

/// Head anchor stored next to the archive: how many records it had and the last hash.
/// `pending` > 0 announces a group of that many records being written after them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Head {
    records: u64,
    last_hash: String,
    #[serde(default, skip_serializing_if = "is_zero")]
    pending: u64,
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

/// Path of the head anchor that belongs to an archive file (`archive.head`).
pub fn head_path(archive: &Path) -> PathBuf {
    archive.with_extension("head")
}

/// Write the head anchor atomically.
fn write_head(archive: &Path, records: u64, last_hash: &str, pending: u64) -> Result<()> {
    let head = Head {
        records,
        last_hash: last_hash.to_string(),
        pending,
    };
    let mut bytes = serde_json::to_vec(&head)?;

    bytes.push(b'\n');

    write_atomic(&head_path(archive), &bytes)
}

/// Check the head anchor against the verified records.
///
/// The anchor may be one record behind (a crash between writing a record and the anchor),
/// or as many records as it announces (a crash during a group write), never ahead and never
/// further behind. A missing anchor is only fine for an empty archive.
fn check_head(path: &Path, records: &[Record]) -> Result<()> {
    let head_file = head_path(path);
    let integrity = |m: String| BrutError::Integrity(format!("{}: {m}", head_file.display()));

    let text = match std::fs::read_to_string(&head_file) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && records.is_empty() => {
            return Ok(());
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(integrity(
                "head anchor is missing; records may have been removed from the end".into(),
            ));
        }
        Err(e) => return Err(e.into()),
    };

    let head: Head =
        serde_json::from_str(&text).map_err(|e| integrity(format!("bad head anchor: {e}")))?;
    let len = records.len() as u64;

    if head.records > len {
        return Err(integrity(format!(
            "the anchor expects {} records, the archive has {len}: records were removed from the end",
            head.records
        )));
    }

    if len - head.records > head.pending.max(1) {
        return Err(integrity(format!(
            "the anchor expects {} records, the archive has {len}: more records were added \
             than the anchor announced",
            head.records
        )));
    }

    let expected = match head.records.checked_sub(1) {
        None => Some(GENESIS_HASH),
        Some(i) => records.get(i as usize).map(|r| r.hash.as_str()),
    };

    if expected != Some(head.last_hash.as_str()) {
        return Err(integrity(format!(
            "the anchor hash does not match record {}",
            head.records.saturating_sub(1)
        )));
    }

    Ok(())
}

/// Roulette-wheel sampling by weight.
fn sample_index(weights: &[f64], total: f64, rng: &mut ChaCha8Rng) -> usize {
    let target = rng.random::<f64>() * total;
    let mut acc = 0.0;

    for (i, w) in weights.iter().enumerate() {
        acc += w;

        if target < acc {
            return i;
        }
    }

    // Rounding can leave target == total; take the last positive weight.
    weights.iter().rposition(|&w| w > 0.0).unwrap_or(0)
}

/// Check the files anchored by `BatchDigest` records (learner runs; a no-op for MLP runs).
///
/// Every digest's lineage chunk and snapshot must exist and match its SHA-256. The score
/// cache is overwritten at every digest, so only the latest digest's cache is checked.
/// Paths are relative to the archive's directory and may not leave it.
fn verify_digest_files(path: &Path, records: &[Record]) -> Result<()> {
    let digests: Vec<&BatchDigestEntry> = records
        .iter()
        .filter_map(|r| match &r.kind {
            RecordKind::BatchDigest(d) => Some(d),
            _ => None,
        })
        .collect();

    let Some(latest) = digests.last() else {
        return Ok(());
    };

    let dir = match path.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => Path::new("."),
    };

    let checked = |rel: &str, sha: &str, limit: u64| -> Result<()> {
        if !is_safe_relative(rel) {
            return Err(BrutError::Integrity(format!(
                "{}: digest names an unsafe path",
                path.display()
            )));
        }

        read_verified(&dir.join(rel), sha, limit).map(|_| ())
    };

    for d in &digests {
        checked(&d.chunk_file, &d.chunk_sha256, MAX_STORE_FILE_BYTES)?;
        checked(&d.snapshot_file, &d.snapshot_sha256, MAX_STORE_FILE_BYTES)?;
    }

    // A crash between the digest and the rename leaves the matching cache pending.
    checked(
        &cache_rel_path(),
        &latest.cache_sha256,
        MAX_CACHE_FILE_BYTES,
    )
    .or_else(|e| {
        checked(
            &cache_pending_rel_path(),
            &latest.cache_sha256,
            MAX_CACHE_FILE_BYTES,
        )
        .map_err(|_| e)
    })
}

/// Read every line, check JSON, sequence numbers, the hash chain, each hash and the head anchor.
///
/// The hash is checked on the exact bytes of the line (with the hash value blanked),
/// so the check does not depend on float parsing.
fn read_and_verify(path: &Path) -> Result<Vec<Record>> {
    let text = std::fs::read_to_string(path)?;
    let integrity = |line: usize, m: &str| {
        BrutError::Integrity(format!("{} line {}: {m}", path.display(), line + 1))
    };

    if !text.is_empty() && !text.ends_with('\n') {
        return Err(BrutError::Integrity(format!(
            "{}: last record is incomplete",
            path.display()
        )));
    }

    let mut records: Vec<Record> = Vec::new();
    let mut prev_hash = GENESIS_HASH.to_string();

    for (i, line) in text.lines().enumerate() {
        let record: Record =
            serde_json::from_str(line).map_err(|e| integrity(i, &format!("bad JSON: {e}")))?;

        if record.seq != i as u64 {
            return Err(integrity(i, &format!("seq {} expected {i}", record.seq)));
        }

        if record.prev_hash != prev_hash {
            return Err(integrity(i, "prev_hash does not match the previous record"));
        }

        let well_formed = record.hash.len() == 64
            && record
                .hash
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));

        if !well_formed {
            return Err(integrity(i, "malformed hash"));
        }

        let suffix = format!("\"hash\":\"{}\"}}", record.hash);
        let Some(body) = line.strip_suffix(&suffix) else {
            return Err(integrity(i, "hash is not the last field"));
        };

        let unhashed = format!("{body}\"hash\":\"\"}}");

        if sha256_hex(unhashed.as_bytes()) != record.hash {
            return Err(integrity(i, "hash mismatch (record was changed)"));
        }

        prev_hash = record.hash.clone();
        records.push(record);
    }

    check_head(path, &records)?;

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{Activation, Optimizer};
    use rand::SeedableRng;

    fn tmp_file(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brut-archive-{name}-{}", std::process::id()));

        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");

        dir.join("archive.jsonl")
    }

    fn genome(seed: u64) -> Genome {
        Genome {
            hidden: vec![8],
            activation: Activation::Tanh,
            optimizer: Optimizer::Adam,
            learning_rate: 0.01,
            l2: 0.0,
            batch_size: 32,
            epochs: 10,
            init_scale: 1.0,
            seed,
        }
    }

    fn entry(id: &str, parents: &[&str], fitness: f64, status: CandidateStatus) -> CandidateEntry {
        CandidateEntry {
            id: id.to_string(),
            generation: if parents.is_empty() { 0 } else { 1 },
            parents: parents.iter().map(|p| p.to_string()).collect(),
            generator: "test".into(),
            genome: genome(id.len() as u64),
            params: 42,
            train: TrainStats {
                epochs_done: 10,
                steps: 100,
                flops: 1000,
                wall_seconds: 0.5,
                final_train_loss: 0.3,
                diverged: false,
                stopped_by_budget: false,
                stopped_by_kill_switch: false,
            },
            valid: Metrics {
                loss: 0.3,
                accuracy: Some(fitness),
                r2: None,
                primary: fitness,
            },
            fitness,
            model_file: None,
            status,
        }
    }

    fn sample_archive(name: &str) -> (PathBuf, Archive) {
        let path = tmp_file(name);
        let mut a = Archive::create(&path).expect("create");

        a.append(
            RecordKind::RunStarted {
                config: Config::default(),
                crate_version: "test".into(),
            },
            1,
        )
        .expect("append");
        a.append(
            RecordKind::Candidate(entry("a", &[], 0.6, CandidateStatus::Ok)),
            2,
        )
        .expect("append");
        a.append(
            RecordKind::Candidate(entry("b", &[], 0.7, CandidateStatus::Ok)),
            3,
        )
        .expect("append");
        a.append(
            RecordKind::Candidate(entry("c", &["a", "b"], 0.8, CandidateStatus::Ok)),
            4,
        )
        .expect("append");
        a.append(
            RecordKind::Candidate(entry("d", &["c"], 0.1, CandidateStatus::Diverged)),
            5,
        )
        .expect("append");

        (path, a)
    }

    #[test]
    fn create_append_open_verify() {
        let (path, a) = sample_archive("basic");

        assert_eq!(a.records().len(), 5);
        assert_eq!(a.records()[0].prev_hash, GENESIS_HASH);
        assert_eq!(a.records()[1].prev_hash, a.records()[0].hash);
        assert_eq!(Archive::verify(&path).expect("verify"), 5);

        let b = Archive::open(&path).expect("open");

        assert_eq!(b.records(), a.records());
        assert_eq!(b.candidates(), a.candidates());
        assert!(Archive::create(&path).is_err());
    }

    #[test]
    fn tampering_is_detected() {
        let (path, _) = sample_archive("tamper");
        let text = std::fs::read_to_string(&path).expect("read");

        // Edit a value.
        let edited = text.replacen("\"fitness\":0.7", "\"fitness\":0.9", 1);

        assert_ne!(edited, text);
        std::fs::write(&path, &edited).expect("write");
        assert!(matches!(
            Archive::verify(&path),
            Err(BrutError::Integrity(_))
        ));

        // Delete a line.
        let lines: Vec<&str> = text.lines().collect();
        let deleted = format!("{}\n{}\n", lines[0], lines[2]);

        std::fs::write(&path, deleted).expect("write");
        assert!(matches!(
            Archive::verify(&path),
            Err(BrutError::Integrity(_))
        ));

        // Swap two lines.
        let swapped = format!("{}\n{}\n{}\n", lines[0], lines[2], lines[1]);

        std::fs::write(&path, swapped).expect("write");
        assert!(matches!(Archive::open(&path), Err(BrutError::Integrity(_))));

        // Truncated last line.
        std::fs::write(&path, &text[..text.len() - 5]).expect("write");
        assert!(Archive::verify(&path).is_err());
    }

    #[test]
    fn truncating_the_tail_is_detected() {
        let (path, archive) = sample_archive("truncate");
        let text = std::fs::read_to_string(&path).expect("read");
        let lines: Vec<&str> = text.lines().collect();
        let keep = |n: usize| -> String { lines[..n].iter().map(|l| format!("{l}\n")).collect() };
        let hashes: Vec<String> = archive.records().iter().map(|r| r.hash.clone()).collect();

        // Removing whole records from the end keeps the chain intact, but not the anchor.
        for n in [0, 3, 4] {
            std::fs::write(&path, keep(n)).expect("write");

            assert!(
                matches!(Archive::verify(&path), Err(BrutError::Integrity(_))),
                "{n} records kept"
            );
            assert!(matches!(Archive::open(&path), Err(BrutError::Integrity(_))));
        }

        std::fs::write(&path, &text).expect("write");
        assert_eq!(Archive::verify(&path).expect("intact"), 5);

        // Deleting the anchor as well is detected.
        let head = head_path(&path);

        std::fs::remove_file(&head).expect("remove anchor");
        assert!(matches!(
            Archive::verify(&path),
            Err(BrutError::Integrity(_))
        ));

        // A crash between writing a record and its anchor leaves the anchor one record
        // behind. That is accepted; two records behind is not.
        write_head(&path, 4, &hashes[3], 0).expect("anchor");
        assert_eq!(Archive::verify(&path).expect("one behind"), 5);

        write_head(&path, 3, &hashes[2], 0).expect("anchor");
        assert!(matches!(
            Archive::verify(&path),
            Err(BrutError::Integrity(_))
        ));

        // An anchor that names a different record is detected.
        write_head(&path, 5, &hashes[3], 0).expect("anchor");
        assert!(matches!(
            Archive::verify(&path),
            Err(BrutError::Integrity(_))
        ));

        // The next append brings a lagging anchor up to date.
        write_head(&path, 4, &hashes[3], 0).expect("anchor");

        let mut reopened = Archive::open(&path).expect("open");

        reopened
            .append(RecordKind::SafetyEvent { what: "x".into() }, 9)
            .expect("append");
        assert_eq!(Archive::verify(&path).expect("verify"), 6);

        std::fs::write(&path, keep(5)).expect("write");
        assert!(Archive::verify(&path).is_err());
    }

    /// Buffered appends write exactly the bytes and anchor that one-by-one appends write.
    #[test]
    fn buffered_appends_match_single_appends() {
        let (single, _) = sample_archive("single");
        let path = tmp_file("buffered");
        let mut a = Archive::create(&path).expect("create");
        let reference = Archive::open(&single).expect("open");

        a.set_buffered(true).expect("buffered");

        for r in reference.records() {
            a.append(r.kind.clone(), r.unix_ms).expect("append");
        }

        // Nothing reached the disk yet, but the records are visible in memory.
        assert_eq!(std::fs::read_to_string(&path).expect("read"), "");
        assert_eq!(a.records(), reference.records());
        assert_eq!(a.candidates(), reference.candidates());

        a.flush().expect("flush");

        let read = |p: &Path| std::fs::read(p).expect("read");

        assert_eq!(read(&path), read(&single));
        assert_eq!(read(&head_path(&path)), read(&head_path(&single)));
        assert_eq!(Archive::verify(&path).expect("verify"), 5);

        // Turning buffering off flushes; later appends are written at once.
        a.append(RecordKind::SafetyEvent { what: "x".into() }, 9)
            .expect("append");
        a.set_buffered(false).expect("unbuffered");
        assert_eq!(Archive::verify(&path).expect("verify"), 6);

        a.append(RecordKind::SafetyEvent { what: "y".into() }, 10)
            .expect("append");
        assert_eq!(Archive::verify(&path).expect("verify"), 7);
    }

    /// A crash after a group of records was written but before the anchor caught up leaves
    /// the announced anchor: accepted up to the announced count, never beyond it.
    #[test]
    fn a_group_write_may_leave_the_anchor_behind_by_the_announced_count() {
        let (path, archive) = sample_archive("group-crash");
        let hashes: Vec<String> = archive.records().iter().map(|r| r.hash.clone()).collect();
        let write_pending = |records: u64, hash: &str, pending: u64| {
            let head = Head {
                records,
                last_hash: hash.to_string(),
                pending,
            };
            let mut bytes = serde_json::to_vec(&head).expect("json");

            bytes.push(b'\n');
            write_atomic(&head_path(&path), &bytes).expect("anchor");
        };

        // Three records announced after record 1: the archive holds 5, two more than the
        // anchor, which the announcement covers.
        write_pending(2, &hashes[1], 3);
        assert_eq!(Archive::verify(&path).expect("announced"), 5);

        // Announcing only one record makes two missing records an error again.
        write_pending(3, &hashes[2], 1);
        assert!(matches!(
            Archive::verify(&path),
            Err(BrutError::Integrity(_))
        ));

        // An announcement never covers removed records.
        write_pending(5, &hashes[4], 3);

        let text = std::fs::read_to_string(&path).expect("read");
        let lines: Vec<&str> = text.lines().collect();
        let kept: String = lines[..4].iter().map(|l| format!("{l}\n")).collect();

        std::fs::write(&path, kept).expect("write");
        assert!(matches!(
            Archive::verify(&path),
            Err(BrutError::Integrity(_))
        ));

        // An anchor without the field (as written by older builds) still parses.
        let old = format!("{{\"records\":4,\"last_hash\":\"{}\"}}\n", hashes[3]);

        std::fs::write(head_path(&path), old).expect("old anchor");
        assert_eq!(Archive::verify(&path).expect("old format"), 4);
    }

    /// Dropping a buffered archive writes what it holds (best effort, e.g. on a panic).
    #[test]
    fn dropping_a_buffered_archive_flushes_it() {
        let path = tmp_file("drop-flush");

        {
            let mut a = Archive::create(&path).expect("create");

            a.set_buffered(true).expect("buffered");
            a.append(
                RecordKind::SafetyEvent {
                    what: "kept".into(),
                },
                1,
            )
            .expect("append");
        }

        assert_eq!(Archive::verify(&path).expect("verify"), 1);
    }

    #[test]
    fn non_finite_numbers_are_refused() {
        let (path, mut a) = sample_archive("nan");
        let bad = entry("x", &[], f64::NAN, CandidateStatus::Ok);

        assert!(matches!(
            a.append(RecordKind::Candidate(bad), 9),
            Err(BrutError::Invalid(_))
        ));
        assert_eq!(a.records().len(), 5);
        assert_eq!(Archive::verify(&path).expect("still valid"), 5);
    }

    #[test]
    fn lineage_and_children() {
        let (_, a) = sample_archive("lineage");
        let ids: Vec<String> = a.lineage("d").into_iter().map(|c| c.id).collect();

        assert_eq!(ids, vec!["d", "c", "a", "b"]);
        assert_eq!(a.children_count("a"), 1);
        assert_eq!(a.children_count("c"), 1);
        assert_eq!(a.children_count("d"), 0);
        assert!(a.lineage("nope").is_empty());
        assert!(a.get("nope").is_none());
    }

    #[test]
    fn quarantine_changes_status_and_eligibility() {
        let (_, mut a) = sample_archive("quarantine");
        let hidden = RecordKind::HiddenEval {
            candidate_id: "c".into(),
            valid_primary: 0.8,
            hidden_primary: 0.6,
            gap: 0.2,
            quarantined: true,
        };

        a.append(hidden, 10).expect("append");

        assert_eq!(
            a.get("c").map(|c| c.status),
            Some(CandidateStatus::Quarantined)
        );

        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let picks = a.select_parents(200, &mut rng);

        assert_eq!(picks.len(), 200);
        assert!(picks.iter().all(|id| id == "a" || id == "b"));
        assert_eq!(a.hidden_evals().len(), 1);
    }

    #[test]
    fn selection_prefers_fit_but_keeps_others() {
        let (_, a) = sample_archive("select");
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let picks = a.select_parents(3000, &mut rng);
        let count = |id: &str| picks.iter().filter(|p| p.as_str() == id).count();

        // c is the fittest, but already has a child; a and b keep a fair share.
        assert!(count("d") == 0);
        assert!(count("a") > 100);
        assert!(count("b") > count("a"));
        assert!(count("c") > count("a"));

        let mut r1 = ChaCha8Rng::seed_from_u64(5);
        let mut r2 = ChaCha8Rng::seed_from_u64(5);

        assert_eq!(a.select_parents(10, &mut r1), a.select_parents(10, &mut r2));
    }

    #[test]
    fn selection_empty_cases() {
        let path = tmp_file("empty");
        let a = Archive::create(&path).expect("create");
        let mut rng = ChaCha8Rng::seed_from_u64(1);

        assert!(a.select_parents(3, &mut rng).is_empty());
        assert_eq!(Archive::verify(&path).expect("verify"), 0);

        let (_, b) = sample_archive("k0");

        assert!(b.select_parents(0, &mut rng).is_empty());
    }

    // ----- learner mode (contract section 5.21) -----

    use crate::learner::fec::CacheSnapshot;
    use crate::learner::types::{FamilyId, Group, MetaConfig};
    use crate::lineage_store::{
        LineageRecord, LineageWriter, Snapshot, write_cache, write_snapshot,
    };

    fn protocol() -> ProtocolDescriptor {
        ProtocolDescriptor {
            version: "learner-v1".into(),
            families: vec![(
                FamilyId("lin-cls".into()),
                Group::Linear,
                Tier::Search,
                false,
            )],
            grammar_search_pool_hash: "a".repeat(64),
            grammar_select_pool_hash: "b".repeat(64),
            steps_per_task: 384,
            checkpoint_divisors: [16, 4, 1],
            aulc_weights: [0.25, 0.25, 0.5],
            n_valid_hidden: 256,
            hidden_tier_sizes: vec![(Tier::HiddenGrammar, 40, 5)],
            seeds_hash: "c".repeat(64),
            baseline_tuning: "trials=200".into(),
            primary_test: "sign-flip".into(),
            dsl_ops_hash: "d".repeat(64),
        }
    }

    fn learner(id: &str, parents: &[&str], status: LearnerStatus, batch: u64) -> LearnerEntry {
        LearnerEntry {
            id: id.into(),
            parents: parents.iter().map(|p| p.to_string()).collect(),
            generator: "aging/const".into(),
            batch,
            genome: LearnerGenome::default(),
            effective_len: 3,
            step_cost_f16: 2_640,
            fp: "00000000000000aa".into(),
            panel_mean: 0.4,
            fitness: 0.35,
            program_file: None,
            status,
        }
    }

    fn hidden(id: &str, tier: Tier, delta: f64, quarantined: bool) -> RecordKind {
        RecordKind::LearnerHiddenEval {
            candidate_id: id.into(),
            tier,
            delta,
            decision: (tier == Tier::HiddenGrammar).then_some(Decision::Fail),
            delta_public: Some(0.1),
            quarantined,
        }
    }

    fn learner_archive(name: &str) -> (PathBuf, Archive) {
        let path = tmp_file(name);
        let mut a = Archive::create(&path).expect("create");

        a.append(
            RecordKind::MetaRunStarted {
                config: Box::new(MetaConfig::default()),
                protocol_hash: "e".repeat(64),
                crate_version: "test".into(),
            },
            1,
        )
        .expect("append");
        a.append(
            RecordKind::ProtocolRegistered {
                protocol_hash: "e".repeat(64),
                protocol: protocol(),
            },
            2,
        )
        .expect("append");

        (path, a)
    }

    #[test]
    fn learner_queries() {
        let (path, mut a) = learner_archive("learner");

        assert_eq!(a.run_id(), Some(a.records()[0].hash.as_str()));
        assert_eq!(a.protocol_hash(), Some("e".repeat(64).as_str()));
        assert_eq!(a.meta_config(), Some(&MetaConfig::default()));
        assert!(a.learner_candidates().is_empty());
        assert!(a.last_batch_digest().is_none());

        let later = MetaConfig {
            max_batches: 7,
            ..MetaConfig::default()
        };

        a.append(
            RecordKind::MetaRunStarted {
                config: Box::new(later.clone()),
                protocol_hash: "f".repeat(64),
                crate_version: "test".into(),
            },
            3,
        )
        .expect("append");

        // Before the search starts, the latest config defines the experiment.
        assert_eq!(a.meta_config(), Some(&later));
        assert_eq!(a.latest_meta_config(), Some(&later));
        assert_eq!(a.protocol_hash(), Some("e".repeat(64).as_str()));
        assert!(!a.learner_search_started());

        let records = [
            RecordKind::LearnerCandidate(learner("s", &[], LearnerStatus::Seed, 0)),
            RecordKind::LearnerCandidate(learner("h", &["s"], LearnerStatus::HallOfFame, 4)),
            RecordKind::LearnerCandidate(learner("f", &["h"], LearnerStatus::Finalist, 9)),
            hidden("f", Tier::HiddenInstance, 0.02, false),
            hidden("f", Tier::HiddenGrammar, 0.01, false),
            hidden("f", Tier::HiddenInstance, 0.03, false),
            RecordKind::LearnerCandidate(learner("s", &[], LearnerStatus::HallOfFame, 5)),
            hidden("h", Tier::HiddenInstance, 0.0, true),
        ];

        for (i, r) in records.into_iter().enumerate() {
            a.append(r, 10 + i as u64).expect("append");
        }

        let ids: Vec<String> = a.learner_candidates().into_iter().map(|e| e.id).collect();

        assert_eq!(ids, vec!["s", "h", "f"]);
        assert!(a.learner_search_started());

        // Once the search started, later invocations do not redefine the experiment.
        let resumed = MetaConfig {
            max_batches: 9,
            ..MetaConfig::default()
        };

        a.append(
            RecordKind::MetaRunStarted {
                config: Box::new(resumed.clone()),
                protocol_hash: "e".repeat(64),
                crate_version: "test".into(),
            },
            30,
        )
        .expect("append");

        assert_eq!(a.meta_config(), Some(&later));
        assert_eq!(a.latest_meta_config(), Some(&resumed));

        // Latest entry per id; quarantine applied from the hidden record.
        assert_eq!(
            a.learner_get("s").map(|e| (e.status, e.batch)),
            Some((LearnerStatus::HallOfFame, 5))
        );
        assert_eq!(
            a.learner_get("h").map(|e| e.status),
            Some(LearnerStatus::Quarantined)
        );
        assert_eq!(
            a.learner_get("f").map(|e| e.status),
            Some(LearnerStatus::Finalist)
        );
        assert!(a.learner_get("nope").is_none());

        // Latest result per (candidate, tier), in order of first evaluation.
        let h = a.learner_hidden_evals();
        let rows: Vec<(String, Tier, f64, bool)> = h
            .iter()
            .map(|r| (r.candidate_id.clone(), r.tier, r.delta, r.quarantined))
            .collect();

        assert_eq!(
            rows,
            vec![
                ("f".into(), Tier::HiddenInstance, 0.03, false),
                ("f".into(), Tier::HiddenGrammar, 0.01, false),
                ("h".into(), Tier::HiddenInstance, 0.0, true),
            ]
        );
        assert_eq!(h[1].decision, Some(Decision::Fail));

        // Reopening rebuilds the same indexes; MLP queries ignore learner records.
        let b = Archive::open(&path).expect("open");

        assert_eq!(b.learner_candidates(), a.learner_candidates());
        assert_eq!(b.learner_hidden_evals(), h);
        assert!(b.candidates().is_empty());
        assert!(b.hidden_evals().is_empty());
        assert!(b.first_config().is_none());
    }

    /// Write a chunk, a snapshot and the cache under the archive's directory and return
    /// the digest that anchors them.
    fn digest_files(dir: &Path, first: u64, last: u64) -> BatchDigestEntry {
        let mut w = LineageWriter::new(dir, first);

        w.push(LineageRecord {
            id: format!("{first:016x}"),
            parent: None,
            batch: first,
            slot: 0,
            mutation: None,
            fp: "0000000000000001".into(),
            fitness: 0.5,
            cached_tasks: 0,
        });

        let (chunk_file, chunk_sha256) = w.finish().expect("chunk");

        let snap = Snapshot {
            batch: last + 1,
            rotation: 0,
            next_age: 0,
            population: Vec::new(),
            hall_of_fame: Vec::new(),
            hurdle_values: Vec::new(),
            flops_used: last,
            select_round: 0,
        };

        let (snapshot_file, snapshot_sha256) = write_snapshot(dir, &snap).expect("snapshot");

        let cache = CacheSnapshot {
            capacity: 4,
            entries: vec![(1, last, 0.5)],
        };

        let cache_sha256 = write_cache(dir, &cache).expect("cache");

        BatchDigestEntry {
            first_batch: first,
            last_batch: last,
            chunk_file,
            chunk_sha256,
            snapshot_file,
            snapshot_sha256,
            cache_sha256,
            stats: ChunkStats::default(),
            best_fitness: 0.5,
        }
    }

    #[test]
    fn digests_anchor_their_files() {
        let (path, mut a) = learner_archive("digest");
        let dir = path.parent().expect("dir").to_path_buf();

        let first = digest_files(&dir, 0, 9);

        a.append(RecordKind::BatchDigest(first.clone()), 3)
            .expect("append");

        // The second digest overwrites the cache: only the latest cache hash is checked.
        let second = digest_files(&dir, 10, 19);

        a.append(RecordKind::BatchDigest(second.clone()), 4)
            .expect("append");

        assert_eq!(a.last_batch_digest(), Some(&second));
        assert_eq!(a.batch_digests().len(), 2);
        assert_eq!(Archive::verify(&path).expect("verify"), 4);
        assert!(Archive::open(&path).is_ok());

        // An edited chunk, a missing snapshot and a changed cache are detected.
        let chunk = dir.join(&first.chunk_file);
        let original = std::fs::read(&chunk).expect("read");

        std::fs::write(&chunk, b"{}\n").expect("write");
        assert!(matches!(Archive::open(&path), Err(BrutError::Integrity(_))));
        std::fs::write(&chunk, &original).expect("restore");
        assert!(Archive::open(&path).is_ok());

        let snapshot = dir.join(&second.snapshot_file);
        let original = std::fs::read(&snapshot).expect("read");

        std::fs::remove_file(&snapshot).expect("remove");
        assert!(matches!(
            Archive::verify(&path),
            Err(BrutError::Integrity(_))
        ));
        std::fs::write(&snapshot, &original).expect("restore");

        let cache = dir.join(cache_rel_path());
        let original = std::fs::read(&cache).expect("read");

        std::fs::write(&cache, b"{\"capacity\":4,\"entries\":[]}\n").expect("write");
        assert!(matches!(Archive::open(&path), Err(BrutError::Integrity(_))));

        // A crash between digest and rename: the pending file holds the anchored cache.
        std::fs::write(dir.join(cache_pending_rel_path()), &original).expect("write");
        assert!(Archive::open(&path).is_ok());
    }

    #[test]
    fn digests_may_not_leave_the_run_directory() {
        let (path, mut a) = learner_archive("digest-path");
        let dir = path.parent().expect("dir").to_path_buf();
        let mut d = digest_files(&dir, 0, 0);

        d.chunk_file = format!("../{}", d.chunk_file);

        a.append(RecordKind::BatchDigest(d), 3).expect("append");

        assert!(matches!(Archive::open(&path), Err(BrutError::Integrity(_))));
    }
}
