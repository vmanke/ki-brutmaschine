//! Shared helpers: error type, hashing, seeds, time, `.env` parsing.
//!
//! This module is complete. Other modules build on it.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};

/// One error type for the whole crate.
#[derive(Debug)]
pub enum BrutError {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// Bad input or bad configuration.
    Invalid(String),
    /// A resource budget was exceeded.
    Budget(String),
    /// The run was stopped on purpose (kill switch, time limit).
    Stopped(String),
    /// The archive hash chain or a file failed an integrity check.
    Integrity(String),
    /// The hidden evaluator failed or refused.
    Eval(String),
}

impl fmt::Display for BrutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BrutError::Io(e) => write!(f, "io: {e}"),
            BrutError::Json(e) => write!(f, "json: {e}"),
            BrutError::Invalid(m) => write!(f, "invalid: {m}"),
            BrutError::Budget(m) => write!(f, "budget: {m}"),
            BrutError::Stopped(m) => write!(f, "stopped: {m}"),
            BrutError::Integrity(m) => write!(f, "integrity: {m}"),
            BrutError::Eval(m) => write!(f, "eval: {m}"),
        }
    }
}

impl std::error::Error for BrutError {}

impl From<std::io::Error> for BrutError {
    fn from(e: std::io::Error) -> Self {
        BrutError::Io(e)
    }
}

impl From<serde_json::Error> for BrutError {
    fn from(e: serde_json::Error) -> Self {
        BrutError::Json(e)
    }
}

pub type Result<T> = std::result::Result<T, BrutError>;

/// Lower-case hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);

    hex::encode(digest.as_slice())
}

/// Derive an independent 64-bit seed from a master seed and a text tag.
///
/// Same inputs give the same output on every platform.
pub fn derive_seed(master: u64, tag: &str) -> u64 {
    let mut hasher = Sha256::new();

    hasher.update(master.to_le_bytes());
    hasher.update(tag.as_bytes());

    let digest = hasher.finalize();
    let mut first = [0u8; 8];

    first.copy_from_slice(&digest.as_slice()[..8]);

    u64::from_le_bytes(first)
}

/// Milliseconds since the Unix epoch.
pub fn now_unix_ms() -> u64 {
    let now = std::time::SystemTime::now();

    match now.duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => d.as_millis() as u64,
        Err(_) => 0,
    }
}

/// Parse a minimal `.env` file: `KEY=VALUE` per line.
///
/// Blank lines and lines starting with `#` are skipped.
/// Surrounding single or double quotes on the value are removed.
/// No variable expansion, no multi-line values.
pub fn parse_env_file(path: &Path) -> Result<HashMap<String, String>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = HashMap::new();

    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let line = line.strip_prefix("export ").unwrap_or(line);

        let Some((key, value)) = line.split_once('=') else {
            return Err(BrutError::Invalid(format!(
                "{}:{}: expected KEY=VALUE",
                path.display(),
                lineno + 1
            )));
        };

        let key = key.trim();
        let mut value = value.trim();

        if value.len() >= 2 {
            let quoted = (value.starts_with('"') && value.ends_with('"'))
                || (value.starts_with('\'') && value.ends_with('\''));

            if quoted {
                value = &value[1..value.len() - 1];
            }
        }

        if key.is_empty() {
            return Err(BrutError::Invalid(format!(
                "{}:{}: empty key",
                path.display(),
                lineno + 1
            )));
        }

        out.insert(key.to_string(), value.to_string());
    }

    Ok(out)
}

/// Write `bytes` to `path` atomically: write a temp file next to it, fsync, rename,
/// then fsync the directory (Unix, best effort) so the rename survives a crash.
///
/// The temp name is unique per process and call (`.<name>.<pid>.<n>.tmp`), so concurrent
/// writers to the same path never rename each other's half-written file; the last rename wins.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let name = path
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = path.with_file_name(format!(".{name}.{}.{n}.tmp", std::process::id()));

    let written = (|| -> Result<()> {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;

        f.write_all(bytes)?;
        f.sync_all()?;
        std::fs::rename(&tmp, path)?;

        Ok(())
    })();

    if written.is_err() {
        // Best effort: the write error matters more than a leftover temp file.
        let _ = std::fs::remove_file(&tmp);

        return written;
    }

    sync_parent_dir(path);

    Ok(())
}

/// fsync the directory that contains `path` (Unix only; errors are ignored).
fn sync_parent_dir(path: &Path) {
    #[cfg(unix)]
    {
        let dir = match path.parent() {
            Some(d) if !d.as_os_str().is_empty() => d,
            _ => Path::new("."),
        };

        if let Ok(d) = std::fs::File::open(dir) {
            let _ = d.sync_all();
        }
    }

    #[cfg(not(unix))]
    let _ = path;
}

/// One standard normal sample (Box-Muller). Always finite.
pub fn normal_f32(rng: &mut ChaCha8Rng) -> f32 {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();

    // random() is in [0, 1), so 1 - u1 is in (0, 1] and ln() stays finite.
    let radius = (-2.0 * (1.0 - u1).ln()).sqrt();
    let angle = std::f64::consts::TAU * u2;

    (radius * angle.cos()) as f32
}

/// Log-uniform sample in [lo, hi]. Both bounds must be > 0.
pub fn log_uniform_f32(rng: &mut ChaCha8Rng, lo: f32, hi: f32) -> f32 {
    let u: f32 = rng.random();
    let value = (lo.ln() + u * (hi.ln() - lo.ln())).exp();

    value.clamp(lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_seed_is_stable_and_tag_sensitive() {
        let a = derive_seed(42, "train");
        let b = derive_seed(42, "train");
        let c = derive_seed(42, "valid");

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn normal_has_roughly_unit_variance() {
        use rand::SeedableRng;

        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let n = 20_000;
        let samples: Vec<f64> = (0..n).map(|_| normal_f32(&mut rng) as f64).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;

        assert!(mean.abs() < 0.05);
        assert!((var - 1.0).abs() < 0.05);
        assert!(samples.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn log_uniform_stays_in_range() {
        use rand::SeedableRng;

        let mut rng = ChaCha8Rng::seed_from_u64(3);

        for _ in 0..1000 {
            let v = log_uniform_f32(&mut rng, 1e-4, 0.3);

            assert!((1e-4..=0.3).contains(&v));
        }
    }

    #[test]
    fn concurrent_atomic_writes_do_not_collide() {
        let dir = std::env::temp_dir().join(format!("brut-util-atomic-{}", std::process::id()));

        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");

        let path = dir.join("state.json");

        std::thread::scope(|scope| {
            for t in 0..8u8 {
                let path = &path;

                scope.spawn(move || {
                    for _ in 0..50 {
                        write_atomic(path, &[b'0' + t; 64]).expect("atomic write");
                    }
                });
            }
        });

        // The file always holds one complete write, and no temp files are left behind.
        let bytes = std::fs::read(&path).expect("read");

        assert_eq!(bytes.len(), 64);
        assert!(bytes.iter().all(|&b| b == bytes[0]));
        assert_eq!(std::fs::read_dir(&dir).expect("list").count(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sha256_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
