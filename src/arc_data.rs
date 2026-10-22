//! ARC, ConceptARC and re-arc loading with a hash manifest and contamination rules.
//!
//! Owner: WP3. Contract section 5.13. Does file I/O: regular files only, size-capped reads,
//! strict shape and value checks. Error messages name a task id or a manifest path, never
//! file content.
//!
//! Layout under the data root (manifest paths are relative to it):
//! - `arc-agi-1/data/{training,evaluation}/<id>.json` (fchollet/ARC-AGI)
//! - `arc-agi-2/data/{training,evaluation}/<id>.json` (arcprize/ARC-AGI-2)
//! - `ConceptARC/corpus/<concept>/<id>.json` (victorvikram/ConceptARC)
//!
//! `data/manifest.example.json` pins the commits and the SHA-256 of every evaluation file.
#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::hidden_eval::read_limited_bytes;
use crate::learner::arc_cells::canonical_pair_hash;
use crate::learner::types::{ArcPair, ArcTask, EvalSet, Grid, SourceId};
use crate::util::{BrutError, Result, sha256_hex};

pub const MAX_ARC_FILE_BYTES: u64 = 1 << 20;
pub const MAX_REARC_FILE_BYTES: u64 = 32 << 20;
/// ARC-AGI-2 eval tasks that duplicate ARC-AGI-1 eval tasks.
///
/// Same ids in both sets. The canonical-pair audit finds one more link: A2 eval db695cfb is
/// a rotated, recoloured copy of A1 eval 55783887 (pinned commits, checked 2026-10-02).
pub const A1_A2_SHARED: [&str; 6] = [
    "0934a4d8", "136b0064", "16b78196", "981571dc", "aa4ec2a5", "da515329",
];

/// Manifest format version this loader reads.
const MANIFEST_VERSION: u32 = 1;

/// Size cap of the manifest file itself.
const MAX_MANIFEST_BYTES: u64 = 4 << 20;

/// Largest manifest path in bytes.
const MAX_REL_PATH: usize = 256;

/// Largest task id in bytes.
const MAX_ID: usize = 64;

/// Grid sides are 1..=MAX_SIDE; colours 0..=MAX_COLOUR.
const MAX_SIDE: usize = 30;
const MAX_COLOUR: u8 = 9;

/// Pair counts of one task file.
const MAX_TRAIN_PAIRS: usize = 10;
const MAX_TEST_PAIRS: usize = 4;

/// Limits of one re-arc family line (as written by tools/rearc_to_jsonl.py).
const MAX_REARC_PAIRS: usize = 60;
const MAX_REARC_AREA: usize = 225;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataRole {
    Train,
    External,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestEntry {
    /// Relative to the data root; no "..", no absolute paths.
    pub rel_path: String,
    pub sha256: String,
    pub license: String,
    pub source: SourceId,
    pub commit: String,
    pub role: DataRole,
    /// For External entries: the set this directory/file belongs to.
    pub set: Option<EvalSet>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    pub entries: Vec<ManifestEntry>,
}

/// One task file as stored on disk.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTask {
    train: Vec<RawPair>,
    test: Vec<RawPair>,
    /// Some ARC-AGI-1 files repeat their id here; the value is skipped.
    #[serde(default, rename = "name")]
    _name: Option<serde::de::IgnoredAny>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPair {
    input: Vec<Vec<u8>>,
    output: Vec<Vec<u8>>,
}

/// One line of a converted re-arc file.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFamily {
    id: String,
    pairs: Vec<RawPair>,
}

/// Load and check a manifest: version 1, safe relative paths, 64-hex hashes, hex commits,
/// role and set consistent with the source, no duplicate paths, no synthetic or BARC entry.
pub fn load_manifest(path: &Path) -> Result<Manifest> {
    let bytes = read_regular(path, MAX_MANIFEST_BYTES, "manifest")?;

    // A generic message: the parser's own error would quote the file.
    let m: Manifest = serde_json::from_slice(&bytes)
        .map_err(|_| BrutError::Invalid("manifest is not valid manifest JSON".into()))?;

    check_manifest(&m)?;

    Ok(m)
}

/// Recompute SHA-256 of every listed file; Integrity error on mismatch or missing file.
pub fn verify_manifest(root: &Path, m: &Manifest) -> Result<()> {
    check_manifest(m)?;

    for e in &m.entries {
        read_listed(root, e)?;
    }

    Ok(())
}

/// Regular file only (no symlink to devices), <= MAX_ARC_FILE_BYTES via read_limited_bytes,
/// JSON {"train": [...], "test": [...]}, grids rectangular 1..=30, values 0..=9,
/// 1..=10 train pairs, 1..=4 test pairs.
///
/// The id is the file stem. An optional "name" key (some ARC-AGI-1 files) is ignored;
/// any other key is refused.
pub fn load_task_file(path: &Path) -> Result<ArcTask> {
    let id = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

    check_task_id(id)?;

    let bytes = read_regular(path, MAX_ARC_FILE_BYTES, &format!("ARC task file {id}"))?;

    parse_task(id, &bytes)
}

/// All tasks of a set (ARC-AGI dirs: data/evaluation/*.json; ConceptARC: corpus/*/*.json),
/// sorted by id; only files listed in the manifest.
///
/// Every file is hashed as it is read and must match its manifest entry (Integrity error
/// otherwise), so the checked bytes are the parsed bytes. An empty set is an error.
pub fn load_eval_set(root: &Path, m: &Manifest, set: EvalSet) -> Result<Vec<ArcTask>> {
    check_manifest(m)?;

    let mut tasks = Vec::new();

    for e in &m.entries {
        if e.role != DataRole::External || e.set != Some(set) {
            continue;
        }

        if !in_set_layout(&e.rel_path, set) {
            return Err(BrutError::Invalid(format!(
                "manifest path {} lies outside the {set:?} layout",
                e.rel_path
            )));
        }

        let bytes = read_listed(root, e)?;

        tasks.push(parse_task(stem_of(&e.rel_path), &bytes)?);
    }

    if tasks.is_empty() {
        return Err(BrutError::Invalid(format!(
            "the manifest lists no tasks for {set:?}"
        )));
    }

    tasks.sort_by(|a, b| a.id.cmp(&b.id));

    if let Some(w) = tasks.windows(2).find(|w| w[0].id == w[1].id) {
        return Err(BrutError::Invalid(format!(
            "task id {} appears twice in {set:?}",
            w[0].id
        )));
    }

    Ok(tasks)
}

/// Hard error when a training source touches the eval set:
/// A1Eval forbids ArcAgi1Eval, ArcAgi2Training, ArcGen, ArcTgi, Barc;
/// A2Eval forbids ArcAgi2Eval, ArcAgi1Eval, ArcTgi, Barc;
/// ConceptArc forbids ConceptArc, Barc. Barc is always forbidden (no licence).
///
/// A1Eval also forbids ArcAgi2Eval: it holds the six A1_A2_SHARED copies of A1 eval tasks.
pub fn check_contamination(train_sources: &[SourceId], set: EvalSet) -> Result<()> {
    for &source in train_sources {
        if let Some(reason) = contamination_reason(source, set) {
            return Err(BrutError::Invalid(format!(
                "contamination: training source {source:?} is not allowed when checking \
                 {set:?} ({reason})"
            )));
        }
    }

    Ok(())
}

/// Why `source` may not train a program checked on `set`; None when it may.
fn contamination_reason(source: SourceId, set: EvalSet) -> Option<&'static str> {
    const ITSELF: &str = "it is the evaluation set itself";
    const SHARED: &str = "ARC-AGI-2 eval holds 6 copies of ARC-AGI-1 eval tasks (A1_A2_SHARED)";

    match (set, source) {
        (_, SourceId::Barc) => Some("BARC ships no licence file"),
        (EvalSet::A1Eval, SourceId::ArcAgi1Eval) => Some(ITSELF),
        (EvalSet::A1Eval, SourceId::ArcAgi2Eval) => Some(SHARED),
        (EvalSet::A1Eval, SourceId::ArcAgi2Training) => {
            Some("ARC-AGI-2 training holds 376 of the 400 ARC-AGI-1 eval tasks")
        }
        (EvalSet::A1Eval, SourceId::ArcGen) => {
            Some("ARC-GEN has generators for 328 ARC-AGI-1 eval tasks")
        }
        (EvalSet::A1Eval | EvalSet::A2Eval, SourceId::ArcTgi) => {
            Some("forbidden by the contamination table of contract section 5.13")
        }
        (EvalSet::A2Eval, SourceId::ArcAgi2Eval) => Some(ITSELF),
        (EvalSet::A2Eval, SourceId::ArcAgi1Eval) => Some(SHARED),
        (EvalSet::ConceptArc, SourceId::ConceptArc) => Some(ITSELF),
        _ => None,
    }
}

/// Canonical hashes of all training pairs of the given sources listed in the manifest.
///
/// Every pair of every listed file counts (train and test pairs alike). Synthetic has no
/// files and is skipped. Any other source without manifest entries is an error: its pairs
/// could not be audited.
///
/// At the pinned commits, A1 eval tasks 070dd51e and 50a16a69 repeat pairs of A1 training
/// tasks 40853293 and caa06a1f. Since any overlap refuses (S7), ARC-AGI-1 training cannot
/// serve as a training source for A1Eval without dropping those two tasks.
pub fn training_pair_hashes(
    root: &Path,
    m: &Manifest,
    sources: &[SourceId],
) -> Result<BTreeSet<[u8; 32]>> {
    check_manifest(m)?;

    let wanted: BTreeSet<SourceId> = sources
        .iter()
        .copied()
        .filter(|&s| s != SourceId::Synthetic)
        .collect();

    for &source in &wanted {
        if !m.entries.iter().any(|e| e.source == source) {
            return Err(BrutError::Invalid(format!(
                "training source {source:?} has no files in the manifest, so its pairs \
                 cannot be audited"
            )));
        }
    }

    let mut hashes = BTreeSet::new();

    for e in m.entries.iter().filter(|e| wanted.contains(&e.source)) {
        let bytes = read_listed(root, e)?;

        for pair in entry_pairs(e, &bytes)? {
            hashes.insert(canonical_pair_hash(&pair));
        }
    }

    Ok(hashes)
}

/// E2 only: converted re-arc cell families (JSONL written by tools/rearc_to_jsonl.py).
///
/// Line format: {"id": "<8 hex>", "pairs": [{"input": grid, "output": grid}, ...]} with
/// 1..=60 pairs that keep their shape and have area <= 225. The file must match a ReArc
/// manifest entry: same SHA-256, and its path ends with the entry's rel_path.
pub fn load_rearc_jsonl(path: &Path, m: &Manifest) -> Result<Vec<(String, Vec<ArcPair>)>> {
    check_manifest(m)?;

    let bytes = read_regular(path, MAX_REARC_FILE_BYTES, "re-arc file")?;
    let digest = sha256_hex(&bytes);
    let listed = m
        .entries
        .iter()
        .any(|e| e.source == SourceId::ReArc && e.sha256 == digest && path.ends_with(&e.rel_path));

    if !listed {
        return Err(BrutError::Integrity(
            "re-arc file is not listed in the manifest with this path and hash".into(),
        ));
    }

    parse_rearc(&bytes)
}

/// Read a regular file of at most `cap` bytes; `label` names it in errors.
///
/// Devices are refused before reading (/dev/zero reports length 0 and never ends).
/// The read itself stops after `cap` bytes, so a file that grows after the check cannot
/// exhaust memory either.
fn read_regular(path: &Path, cap: u64, label: &str) -> Result<Vec<u8>> {
    let not_regular = || BrutError::Invalid(format!("{label} is not a regular file"));

    if !std::fs::metadata(path)?.is_file() {
        return Err(not_regular());
    }

    let file = std::fs::File::open(path)?;
    let meta = file.metadata()?;

    // Check the opened handle again: the path may have changed since the first check.
    if !meta.is_file() {
        return Err(not_regular());
    }

    if meta.len() > cap {
        return Err(BrutError::Invalid(format!(
            "{label} has {} bytes, max {cap}",
            meta.len()
        )));
    }

    read_limited_bytes(file, cap)
}

/// Read a manifest-listed file and check its SHA-256. Missing file or mismatch: Integrity.
fn read_listed(root: &Path, e: &ManifestEntry) -> Result<Vec<u8>> {
    let cap = match e.source {
        SourceId::ReArc => MAX_REARC_FILE_BYTES,
        _ => MAX_ARC_FILE_BYTES,
    };
    let label = format!("listed file {}", e.rel_path);

    let bytes = match read_regular(&root.join(&e.rel_path), cap, &label) {
        Ok(bytes) => bytes,
        Err(BrutError::Io(err)) if err.kind() == std::io::ErrorKind::NotFound => {
            return Err(BrutError::Integrity(format!("{label} is missing")));
        }
        Err(other) => return Err(other),
    };

    if sha256_hex(&bytes) != e.sha256 {
        return Err(BrutError::Integrity(format!(
            "{label} does not match its manifest hash"
        )));
    }

    Ok(bytes)
}

/// All pairs of one listed file.
fn entry_pairs(e: &ManifestEntry, bytes: &[u8]) -> Result<Vec<ArcPair>> {
    if e.source == SourceId::ReArc {
        let families = parse_rearc(bytes)?;

        return Ok(families.into_iter().flat_map(|(_, pairs)| pairs).collect());
    }

    let task = parse_task(stem_of(&e.rel_path), bytes)?;

    Ok(task.train.into_iter().chain(task.test).collect())
}

/// Parse and check one task file.
fn parse_task(id: &str, bytes: &[u8]) -> Result<ArcTask> {
    check_task_id(id)?;

    let bad = |what: &str| BrutError::Invalid(format!("ARC task {id}: {what}"));

    // A generic message: the parser's own error would quote the file.
    let raw: RawTask = serde_json::from_slice(bytes)
        .map_err(|_| bad("not a task file (JSON with train and test pairs of 0..=9 grids)"))?;

    if !(1..=MAX_TRAIN_PAIRS).contains(&raw.train.len()) {
        return Err(bad("needs 1 to 10 train pairs"));
    }

    if !(1..=MAX_TEST_PAIRS).contains(&raw.test.len()) {
        return Err(bad("needs 1 to 4 test pairs"));
    }

    let train = raw
        .train
        .iter()
        .map(pair_from_raw)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(bad)?;
    let test = raw
        .test
        .iter()
        .map(pair_from_raw)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(bad)?;

    Ok(ArcTask {
        id: id.to_string(),
        train,
        test,
    })
}

/// Parse and check a converted re-arc file (format in `load_rearc_jsonl`).
fn parse_rearc(bytes: &[u8]) -> Result<Vec<(String, Vec<ArcPair>)>> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| BrutError::Invalid("re-arc file is not UTF-8".into()))?;
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();

    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        let bad = |what: &str| BrutError::Invalid(format!("re-arc line {}: {what}", n + 1));

        let raw: RawFamily =
            serde_json::from_str(line).map_err(|_| bad("not a family (JSON with id and pairs)"))?;

        if raw.id.len() != 8 || !is_lower_hex(&raw.id) {
            return Err(bad("id must be 8 lower-case hex digits"));
        }

        if !seen.insert(raw.id.clone()) {
            return Err(bad("duplicate id"));
        }

        if !(1..=MAX_REARC_PAIRS).contains(&raw.pairs.len()) {
            return Err(bad("needs 1 to 60 pairs"));
        }

        let mut pairs = Vec::with_capacity(raw.pairs.len());

        for p in &raw.pairs {
            let pair = pair_from_raw(p).map_err(bad)?;

            if pair.input.h != pair.output.h || pair.input.w != pair.output.w {
                return Err(bad("every pair must keep its shape"));
            }

            if pair.input.cells.len() > MAX_REARC_AREA {
                return Err(bad("grid area must be at most 225"));
            }

            pairs.push(pair);
        }

        out.push((raw.id, pairs));
    }

    if out.is_empty() {
        return Err(BrutError::Invalid("re-arc file has no families".into()));
    }

    Ok(out)
}

fn pair_from_raw(p: &RawPair) -> std::result::Result<ArcPair, &'static str> {
    Ok(ArcPair {
        input: grid_from_rows(&p.input)?,
        output: grid_from_rows(&p.output)?,
    })
}

/// Rectangular, sides 1..=30, colours 0..=9.
fn grid_from_rows(rows: &[Vec<u8>]) -> std::result::Result<Grid, &'static str> {
    let h = rows.len();
    let w = rows.first().map_or(0, Vec::len);

    if !(1..=MAX_SIDE).contains(&h) || !(1..=MAX_SIDE).contains(&w) {
        return Err("grid sides must be 1 to 30");
    }

    let mut cells = Vec::with_capacity(h * w);

    for row in rows {
        if row.len() != w {
            return Err("grid is not rectangular");
        }

        if row.iter().any(|&v| v > MAX_COLOUR) {
            return Err("grid colours must be 0 to 9");
        }

        cells.extend_from_slice(row);
    }

    Ok(Grid {
        h: h as u8,
        w: w as u8,
        cells,
    })
}

/// 1..=64 characters from [A-Za-z0-9_-]. The message never quotes a bad id.
fn check_task_id(id: &str) -> Result<()> {
    let ok = !id.is_empty()
        && id.len() <= MAX_ID
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');

    if !ok {
        return Err(BrutError::Invalid(
            "ARC task id must be 1 to 64 characters from [A-Za-z0-9_-]".into(),
        ));
    }

    Ok(())
}

/// File name of a manifest path without ".json" / ".jsonl".
fn stem_of(rel: &str) -> &str {
    let name = rel.rsplit('/').next().unwrap_or(rel);

    name.strip_suffix(".jsonl")
        .or_else(|| name.strip_suffix(".json"))
        .unwrap_or(name)
}

/// The source whose files make up `set`.
fn eval_source(set: EvalSet) -> SourceId {
    match set {
        EvalSet::A1Eval => SourceId::ArcAgi1Eval,
        EvalSet::A2Eval => SourceId::ArcAgi2Eval,
        EvalSet::ConceptArc => SourceId::ConceptArc,
    }
}

/// ARC-AGI sets live in `.../data/evaluation/<id>.json`, ConceptARC in
/// `.../corpus/<concept>/<id>.json`.
fn in_set_layout(rel: &str, set: EvalSet) -> bool {
    let parts: Vec<&str> = rel.split('/').collect();
    let n = parts.len();

    if n < 3 || !rel.ends_with(".json") {
        return false;
    }

    match set {
        EvalSet::A1Eval | EvalSet::A2Eval => parts[n - 3] == "data" && parts[n - 2] == "evaluation",
        EvalSet::ConceptArc => parts[n - 3] == "corpus",
    }
}

fn is_lower_hex(s: &str) -> bool {
    s.bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Structural checks of a manifest (also applied to manifests built in memory).
fn check_manifest(m: &Manifest) -> Result<()> {
    if m.version != MANIFEST_VERSION {
        return Err(BrutError::Invalid(format!(
            "manifest version {} is not supported (expected {MANIFEST_VERSION})",
            m.version
        )));
    }

    let mut paths = BTreeSet::new();

    for (i, e) in m.entries.iter().enumerate() {
        check_entry(e).map_err(|what| BrutError::Invalid(format!("manifest entry {i}: {what}")))?;

        if !paths.insert(e.rel_path.as_str()) {
            return Err(BrutError::Invalid(format!(
                "manifest entry {i}: path {} is listed twice",
                e.rel_path
            )));
        }
    }

    Ok(())
}

fn check_entry(e: &ManifestEntry) -> std::result::Result<(), &'static str> {
    check_rel_path(&e.rel_path)?;

    if e.sha256.len() != 64 || !is_lower_hex(&e.sha256) {
        return Err("sha256 must be 64 lower-case hex digits");
    }

    if !(7..=64).contains(&e.commit.len()) || !is_lower_hex(&e.commit) {
        return Err("commit must be 7 to 64 lower-case hex digits");
    }

    if e.license.is_empty()
        || e.license.len() > 64
        || !e.license.bytes().all(|b| b.is_ascii_graphic())
    {
        return Err("license must be a short licence name such as Apache-2.0");
    }

    match e.source {
        SourceId::Synthetic => return Err("synthetic data has no files"),
        SourceId::Barc => return Err("BARC ships no licence file and is never used"),
        _ => {}
    }

    let extension_ok = match e.source {
        SourceId::ReArc => e.rel_path.ends_with(".jsonl"),
        _ => e.rel_path.ends_with(".json"),
    };

    if !extension_ok {
        return Err("re-arc files end in .jsonl, all other files in .json");
    }

    match (e.role, e.set) {
        (DataRole::External, Some(set)) if e.source == eval_source(set) => Ok(()),
        (DataRole::External, _) => Err("an external entry needs the set of its source"),
        (DataRole::Train, None) => Ok(()),
        (DataRole::Train, Some(_)) => Err("a train entry has no set"),
    }
}

/// Relative, '/'-separated, no empty, "." or ".." parts, characters [A-Za-z0-9._-].
/// The message never quotes the path.
fn check_rel_path(rel: &str) -> std::result::Result<(), &'static str> {
    let bad = "path must be relative, at most 256 bytes, without '.' or '..' parts, \
               and made of [A-Za-z0-9._-] parts joined by '/'";

    if rel.is_empty() || rel.len() > MAX_REL_PATH {
        return Err(bad);
    }

    for part in rel.split('/') {
        let chars_ok = part
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-');

        if part.is_empty() || part == "." || part == ".." || !chars_ok {
            return Err(bad);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;
    use std::path::PathBuf;

    use crate::learner::arc_cells::{
        ArcResult, cell_tasks, mean_bca, phi32_ceiling, score_kaggle, score_official, solve_with,
    };
    use crate::learner::types::{Labels, TaskInstance};

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brut-arc-{name}-{}", std::process::id()));

        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");

        dir
    }

    /// Write `text` to `root/rel` (parents created) and return its manifest hash.
    fn put(root: &Path, rel: &str, text: &str) -> String {
        let path = root.join(rel);

        std::fs::create_dir_all(path.parent().expect("parent")).expect("dirs");
        std::fs::write(&path, text).expect("write");

        sha256_hex(text.as_bytes())
    }

    fn entry(rel: &str, sha: &str, source: SourceId, set: Option<EvalSet>) -> ManifestEntry {
        ManifestEntry {
            rel_path: rel.to_string(),
            sha256: sha.to_string(),
            license: "Apache-2.0".into(),
            source,
            commit: "399030444e0ab0cc8b4e199870fb20b863846f34".into(),
            role: if set.is_some() {
                DataRole::External
            } else {
                DataRole::Train
            },
            set,
        }
    }

    fn manifest(entries: Vec<ManifestEntry>) -> Manifest {
        Manifest {
            version: 1,
            entries,
        }
    }

    /// A small valid task: one train pair, one test pair, colours shifted by `shift`.
    fn task_json(shift: u8) -> String {
        let a = shift % 10;
        let b = (shift + 1) % 10;

        format!(
            r#"{{"train":[{{"input":[[{a},0],[0,{a}]],"output":[[{b},0],[0,{b}]]}}],
               "test":[{{"input":[[0,{a}]],"output":[[0,{b}]]}}]}}"#
        )
    }

    fn expect_invalid(r: Result<ArcTask>, what: &str) {
        match r {
            Err(BrutError::Invalid(_)) => {}
            other => panic!("{what}: expected Invalid, got {other:?}"),
        }
    }

    #[test]
    fn loads_a_valid_task_file_and_ignores_name() {
        let dir = tmp_dir("valid");
        let path = dir.join("0934a4d8.json");

        std::fs::write(&path, task_json(1)).expect("write");

        let t = load_task_file(&path).expect("valid task");

        assert_eq!(t.id, "0934a4d8");
        assert_eq!(t.train.len(), 1);
        assert_eq!(t.test.len(), 1);
        assert_eq!(t.train[0].input.cells, vec![1, 0, 0, 1]);
        assert_eq!((t.test[0].output.h, t.test[0].output.w), (1, 2));

        // Some ARC-AGI-1 files carry "name"; it is accepted and ignored.
        let named = task_json(1).replacen('{', r#"{"name":"0934a4d8","#, 1);

        std::fs::write(&path, named).expect("write");

        assert_eq!(load_task_file(&path).expect("named"), t);
    }

    // T-DATA-1: oversize, non-rectangular, value 10, 31 x 31, devices, bad counts.
    #[test]
    fn rejects_bad_task_files() {
        let dir = tmp_dir("bad");
        let path = dir.join("t.json");
        let pair = |g: &str| format!(r#"{{"input":{g},"output":{g}}}"#);
        let file = |train: &[&str], test: &[&str]| {
            format!(
                r#"{{"train":[{}],"test":[{}]}}"#,
                train.join(","),
                test.join(",")
            )
        };
        let ok = pair("[[1]]");
        let ok = ok.as_str();
        let big_row = format!("[{}]", vec!["1"; 31].join(","));
        let big = pair(&format!("[{}]", vec![big_row.as_str(); 31].join(",")));
        let wide = pair(&format!("[[{}]]", vec!["1"; 31].join(",")));
        let ragged = pair("[[1,2],[3]]");
        let ten = pair("[[10]]");
        let negative = pair("[[-1]]");
        let float = pair("[[1.5]]");
        let empty_grid = pair("[]");
        let empty_row = pair("[[]]");
        let pair_key = r#"{"input":[[1]],"output":[[1]],"x":1}"#;
        let no_output = r#"{"input":[[1]]}"#;

        let cases = [
            ("non-rectangular", file(&[&ragged], &[ok])),
            ("value 10", file(&[&ten], &[ok])),
            ("negative", file(&[&negative], &[ok])),
            ("float", file(&[&float], &[ok])),
            ("31 x 31", file(&[&big], &[ok])),
            ("31 wide", file(&[&wide], &[ok])),
            ("empty grid", file(&[&empty_grid], &[ok])),
            ("empty row", file(&[&empty_row], &[ok])),
            ("no train", file(&[], &[ok])),
            ("no test", file(&[ok], &[])),
            ("11 train", file(&[ok; 11], &[ok])),
            ("5 test", file(&[ok], &[ok; 5])),
            (
                "unknown key",
                file(&[ok], &[ok]).replacen('{', r#"{"x":1,"#, 1),
            ),
            ("pair key", file(&[pair_key], &[ok])),
            ("missing output", file(&[ok], &[no_output])),
            ("not json", "train: []".to_string()),
        ];

        for (what, text) in cases {
            std::fs::write(&path, text).expect("write");

            expect_invalid(load_task_file(&path), what);
        }

        // 10 train and 4 test pairs are the largest legal counts.
        std::fs::write(&path, file(&[ok; 10], &[ok; 4])).expect("write");

        assert!(load_task_file(&path).is_ok());

        // Oversize: refused from the metadata, before parsing.
        std::fs::write(&path, vec![b' '; (MAX_ARC_FILE_BYTES + 1) as usize]).expect("write");

        match load_task_file(&path) {
            Err(BrutError::Invalid(m)) => assert!(m.contains("bytes"), "{m}"),
            other => panic!("oversize: {other:?}"),
        }

        // A directory and an id with bad characters.
        expect_invalid(load_task_file(&dir), "directory");

        let odd = dir.join("a b.json");

        std::fs::write(&odd, task_json(1)).expect("write");
        expect_invalid(load_task_file(&odd), "bad id");
    }

    #[cfg(unix)]
    #[test]
    fn refuses_devices_and_symlinks_to_devices() {
        let dir = tmp_dir("dev");
        let link = dir.join("zero.json");

        std::os::unix::fs::symlink("/dev/zero", &link).expect("symlink");

        for path in [
            link.as_path(),
            Path::new("/dev/zero"),
            Path::new("/dev/null"),
        ] {
            match load_task_file(path) {
                Err(BrutError::Invalid(m)) => assert!(m.contains("regular file"), "{m}"),
                other => panic!("{}: {other:?}", path.display()),
            }
        }

        // The same check guards listed files and the manifest itself.
        match load_manifest(&link) {
            Err(BrutError::Invalid(m)) => assert!(m.contains("regular file"), "{m}"),
            other => panic!("manifest: {other:?}"),
        }
    }

    #[test]
    fn manifest_round_trip_and_example_file() {
        let dir = tmp_dir("manifest");
        let m = manifest(vec![
            entry(
                "arc-agi-1/data/evaluation/0934a4d8.json",
                &"a".repeat(64),
                SourceId::ArcAgi1Eval,
                Some(EvalSet::A1Eval),
            ),
            entry(
                "arc-agi-1/data/training/007bbfb7.json",
                &"b".repeat(64),
                SourceId::ArcAgi1Training,
                None,
            ),
        ]);
        let path = dir.join("manifest.json");

        std::fs::write(&path, serde_json::to_string(&m).expect("json")).expect("write");

        assert_eq!(load_manifest(&path).expect("valid"), m);

        // The committed example parses and passes every structural check.
        let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/manifest.example.json");
        let ex = load_manifest(&example).expect("example manifest");
        let count = |set| ex.entries.iter().filter(|e| e.set == Some(set)).count();

        assert_eq!(count(EvalSet::A1Eval), 400);
        assert_eq!(count(EvalSet::A2Eval), 120);
        assert_eq!(count(EvalSet::ConceptArc), 160);
    }

    // T-DATA-1: ".." in a manifest path and other structural errors.
    #[test]
    fn manifest_rejects_bad_entries() {
        let good = || {
            entry(
                "arc-agi-1/data/evaluation/x.json",
                &"a".repeat(64),
                SourceId::ArcAgi1Eval,
                Some(EvalSet::A1Eval),
            )
        };

        assert!(check_manifest(&manifest(vec![good()])).is_ok());

        type Edit = fn(&mut ManifestEntry);

        let edits: [(&str, Edit); 17] = [
            ("dotdot", |e| {
                e.rel_path = "arc-agi-1/../../etc/x.json".into()
            }),
            ("absolute", |e| e.rel_path = "/etc/x.json".into()),
            ("dot", |e| e.rel_path = "./x.json".into()),
            ("double slash", |e| e.rel_path = "a//x.json".into()),
            ("backslash", |e| e.rel_path = "a\\x.json".into()),
            ("empty", |e| e.rel_path = String::new()),
            ("long", |e| e.rel_path = format!("{}.json", "a".repeat(300))),
            ("short hash", |e| e.sha256 = "abc".into()),
            ("upper hash", |e| e.sha256 = "A".repeat(64)),
            ("bad commit", |e| e.commit = "main".into()),
            ("no license", |e| e.license = String::new()),
            ("synthetic", |e| e.source = SourceId::Synthetic),
            ("barc", |e| e.source = SourceId::Barc),
            ("wrong set", |e| e.set = Some(EvalSet::A2Eval)),
            ("no set", |e| e.set = None),
            ("train with set", |e| e.role = DataRole::Train),
            ("rearc json", |e| e.source = SourceId::ReArc),
        ];

        for (what, edit) in edits {
            let mut e = good();

            edit(&mut e);

            assert!(check_manifest(&manifest(vec![e])).is_err(), "{what}");
        }

        assert!(
            check_manifest(&manifest(vec![good(), good()])).is_err(),
            "duplicate"
        );
        assert!(
            check_manifest(&Manifest {
                version: 2,
                entries: vec![]
            })
            .is_err(),
            "version"
        );

        // Unknown fields are refused by the parser.
        let dir = tmp_dir("manifest-unknown");
        let path = dir.join("m.json");

        std::fs::write(&path, r#"{"version":1,"entries":[],"extra":1}"#).expect("write");

        assert!(load_manifest(&path).is_err());
    }

    /// A data root with two A1 eval tasks, one unlisted eval file and one training task.
    fn small_root(name: &str) -> (PathBuf, Manifest) {
        let root = tmp_dir(name);
        let b = put(
            &root,
            "arc-agi-1/data/evaluation/bbbbbbbb.json",
            &task_json(2),
        );
        let a = put(
            &root,
            "arc-agi-1/data/evaluation/aaaaaaaa.json",
            &task_json(1),
        );
        let t = put(
            &root,
            "arc-agi-1/data/training/cccccccc.json",
            &task_json(3),
        );

        put(
            &root,
            "arc-agi-1/data/evaluation/unlisted.json",
            &task_json(4),
        );

        let m = manifest(vec![
            entry(
                "arc-agi-1/data/evaluation/bbbbbbbb.json",
                &b,
                SourceId::ArcAgi1Eval,
                Some(EvalSet::A1Eval),
            ),
            entry(
                "arc-agi-1/data/evaluation/aaaaaaaa.json",
                &a,
                SourceId::ArcAgi1Eval,
                Some(EvalSet::A1Eval),
            ),
            entry(
                "arc-agi-1/data/training/cccccccc.json",
                &t,
                SourceId::ArcAgi1Training,
                None,
            ),
        ]);

        (root, m)
    }

    #[test]
    fn eval_set_is_sorted_and_limited_to_the_manifest() {
        let (root, m) = small_root("evalset");

        verify_manifest(&root, &m).expect("hashes match");

        let tasks = load_eval_set(&root, &m, EvalSet::A1Eval).expect("set");
        let ids: Vec<&str> = tasks.iter().map(|t| t.id.as_str()).collect();

        assert_eq!(ids, vec!["aaaaaaaa", "bbbbbbbb"]);

        // No ConceptARC entries: an empty set is an error, not zero tasks.
        assert!(load_eval_set(&root, &m, EvalSet::ConceptArc).is_err());

        // An eval entry outside data/evaluation is refused.
        let mut moved = m.clone();

        moved.entries[0].rel_path = "arc-agi-1/data/training/cccccccc.json".into();
        moved.entries[0].sha256 = m.entries[2].sha256.clone();
        moved.entries.pop();

        assert!(load_eval_set(&root, &moved, EvalSet::A1Eval).is_err());
    }

    // T-DATA-1: hash mismatch and missing file are integrity errors.
    #[test]
    fn hash_mismatch_and_missing_files_are_integrity_errors() {
        let (root, m) = small_root("integrity");

        put(
            &root,
            "arc-agi-1/data/evaluation/aaaaaaaa.json",
            &task_json(5),
        );

        assert!(matches!(
            verify_manifest(&root, &m),
            Err(BrutError::Integrity(_))
        ));
        assert!(matches!(
            load_eval_set(&root, &m, EvalSet::A1Eval),
            Err(BrutError::Integrity(_))
        ));

        let (root, m) = small_root("missing");

        std::fs::remove_file(root.join("arc-agi-1/data/training/cccccccc.json")).expect("rm");

        match verify_manifest(&root, &m) {
            Err(BrutError::Integrity(msg)) => assert!(msg.contains("missing"), "{msg}"),
            other => panic!("missing: {other:?}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn listed_symlink_to_a_device_is_refused() {
        let (root, mut m) = small_root("listed-dev");
        let rel = "arc-agi-1/data/evaluation/dddddddd.json";

        std::os::unix::fs::symlink("/dev/zero", root.join(rel)).expect("symlink");
        m.entries.push(entry(
            rel,
            &"0".repeat(64),
            SourceId::ArcAgi1Eval,
            Some(EvalSet::A1Eval),
        ));

        match verify_manifest(&root, &m) {
            Err(BrutError::Invalid(msg)) => assert!(msg.contains("regular file"), "{msg}"),
            other => panic!("device: {other:?}"),
        }
    }

    // T-DATA-2: the contamination blocklist, including the A1_A2_SHARED rule.
    #[test]
    fn contamination_blocklist() {
        use SourceId::*;

        let all = [
            Synthetic,
            ArcAgi1Training,
            ArcAgi1Eval,
            ArcAgi2Training,
            ArcAgi2Eval,
            ReArc,
            ConceptArc,
            ArcGen,
            ArcTgi,
            Barc,
        ];
        let forbidden: [(EvalSet, &[SourceId]); 3] = [
            (
                EvalSet::A1Eval,
                &[
                    ArcAgi1Eval,
                    ArcAgi2Training,
                    ArcAgi2Eval,
                    ArcGen,
                    ArcTgi,
                    Barc,
                ],
            ),
            (EvalSet::A2Eval, &[ArcAgi2Eval, ArcAgi1Eval, ArcTgi, Barc]),
            (EvalSet::ConceptArc, &[ConceptArc, Barc]),
        ];

        for (set, banned) in forbidden {
            for source in all {
                let r = check_contamination(&[Synthetic, source], set);

                assert_eq!(r.is_err(), banned.contains(&source), "{set:?} / {source:?}");
            }

            assert!(check_contamination(&[], set).is_ok());
        }

        // The six shared tasks block ARC-AGI-2 eval for A1 and ARC-AGI-1 eval for A2.
        for (source, set) in [
            (ArcAgi2Eval, EvalSet::A1Eval),
            (ArcAgi1Eval, EvalSet::A2Eval),
        ] {
            match check_contamination(&[source], set) {
                Err(BrutError::Invalid(m)) => assert!(m.contains("A1_A2_SHARED"), "{m}"),
                other => panic!("{source:?}: {other:?}"),
            }
        }

        assert_eq!(A1_A2_SHARED.len(), 6);
        assert!(
            A1_A2_SHARED
                .iter()
                .all(|id| id.len() == 8 && is_lower_hex(id))
        );
    }

    // AC-10: a rotated and recoloured copy of an eval pair is found by the audit.
    #[test]
    fn audit_finds_a_planted_rotated_recoloured_pair() {
        let root = tmp_dir("audit");
        let eval_pair = r#"{"input":[[1,2,0],[0,0,3]],"output":[[1,1,0],[0,0,3]]}"#;

        // Rotated 90 degrees clockwise, then colours 1 -> 7, 2 -> 4, 3 -> 9.
        let planted = r#"{"input":[[0,7],[0,4],[9,0]],"output":[[0,7],[0,7],[9,0]]}"#;
        let other = r#"{"input":[[5]],"output":[[6]]}"#;
        let train = format!(r#"{{"train":[{other},{planted}],"test":[{other}]}}"#);
        let eval = format!(r#"{{"train":[{other}],"test":[{eval_pair}]}}"#);
        let t = put(&root, "arc-agi-1/data/training/eeeeeeee.json", &train);
        let e = put(&root, "arc-agi-1/data/evaluation/ffffffff.json", &eval);
        let m = manifest(vec![
            entry(
                "arc-agi-1/data/training/eeeeeeee.json",
                &t,
                SourceId::ArcAgi1Training,
                None,
            ),
            entry(
                "arc-agi-1/data/evaluation/ffffffff.json",
                &e,
                SourceId::ArcAgi1Eval,
                Some(EvalSet::A1Eval),
            ),
        ]);

        let hashes =
            training_pair_hashes(&root, &m, &[SourceId::Synthetic, SourceId::ArcAgi1Training])
                .expect("hashes");
        let eval_task = load_eval_set(&root, &m, EvalSet::A1Eval)
            .expect("eval")
            .remove(0);

        assert_eq!(hashes.len(), 2, "two distinct training pairs");
        assert!(hashes.contains(&canonical_pair_hash(&eval_task.test[0])));
        assert!(hashes.contains(&canonical_pair_hash(&eval_task.train[0])));

        // Synthetic alone has nothing to audit; an unlisted source cannot be audited.
        assert!(
            training_pair_hashes(&root, &m, &[SourceId::Synthetic])
                .expect("empty")
                .is_empty()
        );
        assert!(training_pair_hashes(&root, &m, &[SourceId::ReArc]).is_err());
    }

    fn rearc_line(id: &str, pairs: usize, side: usize) -> String {
        let row = format!("[{}]", vec!["1"; side].join(","));
        let g = format!("[{}]", vec![row.as_str(); side].join(","));
        let p = format!(r#"{{"input":{g},"output":{g}}}"#);

        format!(r#"{{"id":"{id}","pairs":[{}]}}"#, vec![p; pairs].join(","))
    }

    #[test]
    fn rearc_jsonl_is_checked_against_the_manifest() {
        let root = tmp_dir("rearc");
        let rel = "re-arc/families.jsonl";
        let text = format!(
            "{}\n{}\n",
            rearc_line("007bbfb7", 2, 3),
            rearc_line("00d62c1b", 60, 15)
        );
        let sha = put(&root, rel, &text);
        let m = manifest(vec![entry(rel, &sha, SourceId::ReArc, None)]);
        let families = load_rearc_jsonl(&root.join(rel), &m).expect("valid");

        assert_eq!(families.len(), 2);
        assert_eq!(families[0].0, "007bbfb7");
        assert_eq!(families[1].1.len(), 60);

        // Its pairs feed the audit too (two distinct pairs: 3 x 3 and 15 x 15 of ones).
        assert_eq!(
            training_pair_hashes(&root, &m, &[SourceId::ReArc])
                .expect("audit")
                .len(),
            2
        );

        // Not listed with this hash.
        put(&root, rel, &rearc_line("007bbfb7", 1, 3));

        assert!(matches!(
            load_rearc_jsonl(&root.join(rel), &m),
            Err(BrutError::Integrity(_))
        ));

        let bad_lines = [
            ("61 pairs", rearc_line("007bbfb7", 61, 3)),
            ("area 256", rearc_line("007bbfb7", 1, 16)),
            ("bad id", rearc_line("XYZ", 1, 3)),
            (
                "duplicate",
                format!(
                    "{}\n{}",
                    rearc_line("007bbfb7", 1, 3),
                    rearc_line("007bbfb7", 1, 3)
                ),
            ),
            (
                "shape change",
                r#"{"id":"007bbfb7","pairs":[{"input":[[1]],"output":[[1,1]]}]}"#.into(),
            ),
            ("empty", String::new()),
        ];

        for (what, text) in bad_lines {
            let sha = put(&root, rel, &text);
            let m = manifest(vec![entry(rel, &sha, SourceId::ReArc, None)]);

            assert!(
                matches!(
                    load_rearc_jsonl(&root.join(rel), &m),
                    Err(BrutError::Invalid(_))
                ),
                "{what}"
            );
        }
    }

    /// Where the real data lives: BRUT_ARC_TEST_ROOT, else `data/external` in the crate.
    fn real_data_root() -> Option<(PathBuf, PathBuf)> {
        let root = match std::env::var("BRUT_ARC_TEST_ROOT") {
            Ok(r) => PathBuf::from(r),
            Err(_) => Path::new(env!("CARGO_MANIFEST_DIR")).join("data/external"),
        };
        let manifest = match std::env::var("BRUT_ARC_TEST_MANIFEST") {
            Ok(m) => PathBuf::from(m),
            Err(_) => root.join("manifest.json"),
        };

        (root.is_dir() && manifest.is_file()).then_some((root, manifest))
    }

    /// The phi32 key of a feature row (all features are 0 or 1).
    fn row_key(row: &[f32]) -> u32 {
        row.iter()
            .enumerate()
            .fold(0, |k, (i, &x)| k | (((x == 1.0) as u32) << i))
    }

    /// The prototype's 1-NN cell learner (arc_cell_knn.py): majority colour per phi32 key
    /// (ties: colour seen first), unseen keys take the nearest key in Hamming distance with
    /// the centre colour as one symbol (ties: key seen first). Outputs are one-hot.
    fn one_nn(task: &TaskInstance) -> Vec<f32> {
        let Labels::Classes(y) = &task.y_train else {
            return Vec::new();
        };

        let mut index: HashMap<u32, usize> = HashMap::new();
        let mut keys: Vec<u32> = Vec::new();
        let mut votes: Vec<[(u32, u32); 10]> = Vec::new();

        for (seq, (row, &colour)) in task.x_train.chunks_exact(32).zip(y).enumerate() {
            let key = row_key(row);
            let slot = *index.entry(key).or_insert_with(|| {
                keys.push(key);
                votes.push([(0, u32::MAX); 10]);

                keys.len() - 1
            });
            let v = &mut votes[slot][colour as usize];

            v.0 += 1;
            v.1 = v.1.min(seq as u32);
        }

        let label = |slot: usize| {
            (0..10)
                .max_by_key(|&c| (votes[slot][c].0, std::cmp::Reverse(votes[slot][c].1)))
                .unwrap_or(0)
        };
        let distance = |a: u32, b: u32| {
            let x = a ^ b;

            ((x & 0x3ff) != 0) as u32 + (x >> 10).count_ones()
        };

        let mut out = vec![0.0; task.n_valid * 10];

        for (i, row) in task.x_valid.chunks_exact(32).enumerate() {
            let key = row_key(row);
            let slot = match index.get(&key) {
                Some(&s) => s,
                None => {
                    let mut best = 0;

                    for (s, &k) in keys.iter().enumerate() {
                        if distance(key, k) < distance(key, keys[best]) {
                            best = s;
                        }
                    }

                    best
                }
            };

            out[i * 10 + label(slot)] = 1.0;
        }

        out
    }

    /// Every pair, train and test, keeps its shape (the plan's "same-shape" count).
    fn all_same_shape(t: &ArcTask) -> bool {
        t.train
            .iter()
            .chain(&t.test)
            .all(|p| p.input.h == p.output.h && p.input.w == p.output.w)
    }

    /// T-ARC-4 (lives here, not in arc_cells.rs, because src/learner/ may not do I/O).
    /// Run with BRUT_ARC_TEST_ROOT=<data root> cargo test --release -- --ignored real_arc
    #[test]
    #[ignore = "needs the real ARC data (BRUT_ARC_TEST_ROOT, manifest at <root>/manifest.json)"]
    fn real_arc_data_counts() {
        let Some((root, manifest_path)) = real_data_root() else {
            println!("skipped: no ARC data root with manifest.json found");

            return;
        };

        let m = load_manifest(&manifest_path).expect("manifest");

        verify_manifest(&root, &m).expect("manifest hashes");

        // (set, tasks, same-shape, phi32 ceiling, 1-NN solved outputs, 1-NN BCA pairs, BCA).
        let expected = [
            (EvalSet::A1Eval, 400, 270, 9, 2, 273, 0.577801),
            (EvalSet::A2Eval, 120, 81, 0, 0, 113, 0.427503),
            (EvalSet::ConceptArc, 160, 119, 17, 6, 344, 0.573152),
        ];

        for (set, n, same, ceiling, nn_solved, nn_pairs, nn_bca) in expected {
            let tasks = load_eval_set(&root, &m, set).expect("eval set");
            let same_shape = tasks.iter().filter(|t| all_same_shape(t)).count();
            let ceil = tasks.iter().filter(|t| phi32_ceiling(t)).count();
            let with_cells = tasks.iter().filter(|t| cell_tasks(t).is_some()).count();
            let outputs: usize = tasks.iter().map(|t| t.test.len()).sum();
            let results: Vec<ArcResult> = tasks
                .iter()
                .map(|t| solve_with(t, |task, _| Some(one_nn(task))))
                .collect();
            let solved: usize = results
                .iter()
                .map(|r| r.correct.iter().filter(|&&c| c).count())
                .sum();
            let solved_tasks = results
                .iter()
                .filter(|r| r.correct.iter().all(|&c| c))
                .count();
            let pairs: usize = results.iter().map(|r| r.bca.len()).sum();
            let bca = mean_bca(&results).unwrap_or(0.0);

            println!(
                "{set:?}: tasks {} | same-shape {same_shape} | cell tasks {with_cells} | \
                 phi32 ceiling {ceil} | 1-NN solved tasks {solved_tasks}, outputs \
                 {solved}/{outputs}, official {:.6}, kaggle {:.6}, BCA {bca:.6} over {pairs} pairs",
                tasks.len(),
                score_official(&results, tasks.len()),
                score_kaggle(&results, outputs),
            );

            assert_eq!(
                (tasks.len(), same_shape, ceil),
                (n, same, ceiling),
                "{set:?}"
            );
            assert_eq!((solved, pairs), (nn_solved, nn_pairs), "{set:?} 1-NN");
            assert!((bca - nn_bca).abs() < 1e-6, "{set:?} BCA {bca}");
        }

        // Canonical-hash audits (the source rules forbid these pairings in real runs).
        let a1_eval = load_eval_set(&root, &m, EvalSet::A1Eval).expect("a1");
        let a2_eval = load_eval_set(&root, &m, EvalSet::A2Eval).expect("a2");
        let touched = |tasks: &[ArcTask], hashes: &BTreeSet<[u8; 32]>| {
            tasks
                .iter()
                .filter(|t| {
                    t.train
                        .iter()
                        .chain(&t.test)
                        .any(|p| hashes.contains(&canonical_pair_hash(p)))
                })
                .map(|t| t.id.clone())
                .collect::<Vec<_>>()
        };

        let a1_hashes = training_pair_hashes(&root, &m, &[SourceId::ArcAgi1Eval]).expect("a1");
        let shared = touched(&a2_eval, &a1_hashes);

        println!("A2 eval tasks sharing a pair with A1 eval: {shared:?}");

        // The six id copies, plus db695cfb: a rotated, recoloured copy of A1 eval 55783887.
        for id in A1_A2_SHARED.iter().chain(&["db695cfb"]) {
            assert!(shared.iter().any(|s| s == id), "{id} not flagged");
        }

        // Expected: 2 (070dd51e, 50a16a69 repeat pairs of A1 training) and 376 (plan).
        for (source, name, expected) in [
            (SourceId::ArcAgi1Training, "ARC-AGI-1 training", 2),
            (SourceId::ArcAgi2Training, "ARC-AGI-2 training", 376),
        ] {
            match training_pair_hashes(&root, &m, &[source]) {
                Ok(h) => {
                    let n = touched(&a1_eval, &h).len();

                    println!("A1 eval tasks sharing a pair with {name}: {n} of 400");

                    assert_eq!(n, expected, "{name}");
                }
                Err(e) => println!("{name}: not audited ({e})"),
            }
        }
    }
}
