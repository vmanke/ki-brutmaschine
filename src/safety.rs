//! Safety guard: kill switch, run budgets, generalization-gap check.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use crate::config::Limits;
use crate::util::{BrutError, Result, now_unix_ms};

/// File name of the kill switch inside the run directory.
pub const STOP_FILE: &str = "STOP";

/// `StopSignal` looks at the STOP file at most this often (milliseconds).
pub const STOP_POLL_MS: u64 = 100;

pub struct Guard {
    stop_file: PathBuf,
    started: Instant,
    max_total_seconds: f64,
    max_total_flops: u64,
    flops_used: u64,
}

impl Guard {
    pub fn new(run_dir: &Path, limits: &Limits) -> Guard {
        Guard {
            stop_file: run_dir.join(STOP_FILE),
            started: Instant::now(),
            max_total_seconds: limits.max_total_seconds,
            max_total_flops: limits.max_total_flops,
            flops_used: 0,
        }
    }

    /// Err(Stopped) if the STOP file exists or the run time is over.
    /// Err(Budget) if total FLOPs are used up.
    pub fn check(&self) -> Result<()> {
        if self.stop_file.exists() {
            return Err(BrutError::Stopped(format!(
                "kill switch {} present",
                self.stop_file.display()
            )));
        }

        // A NaN limit compares false; treat it as "no time left" to fail safe.
        let time_ok = self.elapsed_seconds() <= self.max_total_seconds;

        if !time_ok {
            return Err(BrutError::Stopped(format!(
                "run time limit of {} s reached",
                self.max_total_seconds
            )));
        }

        if self.flops_used >= self.max_total_flops {
            return Err(BrutError::Budget(format!(
                "run FLOP budget used up ({} of {})",
                self.flops_used, self.max_total_flops
            )));
        }

        Ok(())
    }

    /// True if `extra` more FLOPs still fit in the run budget.
    pub fn can_afford(&self, extra: u64) -> bool {
        self.flops_used.saturating_add(extra) <= self.max_total_flops
    }

    /// Add spent FLOPs (saturating).
    pub fn charge(&mut self, flops: u64) {
        self.flops_used = self.flops_used.saturating_add(flops);
    }

    pub fn flops_used(&self) -> u64 {
        self.flops_used
    }

    pub fn elapsed_seconds(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// True if the STOP file exists right now.
    pub fn stop_requested(&self) -> bool {
        self.stop_file.exists()
    }
}

/// Kill-switch check for hot loops: many threads may call `requested` very often.
///
/// It looks at the STOP file at most every `STOP_POLL_MS` (one thread checks, the others
/// continue) and remembers a positive answer, so a STOP seen once stays seen.
pub struct StopSignal {
    stop_file: PathBuf,
    started: Instant,
    /// Milliseconds since `started` before which nobody needs to look at the file.
    next_check_ms: AtomicU64,
    stopped: AtomicBool,
}

impl StopSignal {
    pub fn new(run_dir: &Path) -> StopSignal {
        StopSignal {
            stop_file: run_dir.join(STOP_FILE),
            started: Instant::now(),
            next_check_ms: AtomicU64::new(0),
            stopped: AtomicBool::new(false),
        }
    }

    /// True once the STOP file has been seen.
    pub fn requested(&self) -> bool {
        if self.stopped.load(Ordering::Relaxed) {
            return true;
        }

        let now = u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX);
        let next = self.next_check_ms.load(Ordering::Relaxed);

        if now < next {
            return false;
        }

        let claimed = self
            .next_check_ms
            .compare_exchange(
                next,
                now.saturating_add(STOP_POLL_MS),
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .is_ok();

        if claimed && self.stop_file.exists() {
            self.stopped.store(true, Ordering::Relaxed);
        }

        self.stopped.load(Ordering::Relaxed)
    }
}

/// Create the STOP file. Idempotent.
pub fn request_stop(run_dir: &Path) -> Result<()> {
    if !run_dir.is_dir() {
        return Err(BrutError::Invalid(format!(
            "run directory {} does not exist",
            run_dir.display()
        )));
    }

    let path = run_dir.join(STOP_FILE);

    if path.exists() {
        return Ok(());
    }

    std::fs::write(
        &path,
        format!("stop requested at unix_ms {}\n", now_unix_ms()),
    )?;

    Ok(())
}

/// True if valid - hidden > threshold (possible overfitting to validation).
/// Non-finite inputs count as exceeded (fail safe).
pub fn gap_exceeded(valid: f64, hidden: f64, threshold: f64) -> bool {
    let gap = valid - hidden;

    if !gap.is_finite() || !threshold.is_finite() {
        return true;
    }

    gap > threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brut-safety-{name}-{}", std::process::id()));

        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");

        dir
    }

    #[test]
    fn kill_switch_stops() {
        let dir = tmp_dir("stop");
        let guard = Guard::new(&dir, &Limits::default());

        assert!(guard.check().is_ok());

        request_stop(&dir).expect("stop");
        request_stop(&dir).expect("idempotent");

        assert!(matches!(guard.check(), Err(BrutError::Stopped(_))));
        assert!(guard.stop_requested());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stop_signal_sees_the_file_and_latches() {
        let dir = tmp_dir("signal");
        let signal = StopSignal::new(&dir);

        assert!(!signal.requested());

        request_stop(&dir).expect("stop");

        let started = Instant::now();

        while !signal.requested() {
            assert!(started.elapsed().as_secs_f64() < 5.0, "STOP not seen");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Once seen, a stop stays requested even if the file goes away.
        std::fs::remove_file(dir.join(STOP_FILE)).expect("remove");
        assert!(signal.requested());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn request_stop_needs_existing_dir() {
        assert!(request_stop(Path::new("/nonexistent/brut/dir")).is_err());
    }

    #[test]
    fn flop_budget() {
        let dir = tmp_dir("flops");
        let limits = Limits {
            max_total_flops: 100,
            ..Limits::default()
        };
        let mut guard = Guard::new(&dir, &limits);

        assert!(guard.can_afford(100));
        assert!(!guard.can_afford(101));

        guard.charge(60);
        assert_eq!(guard.flops_used(), 60);
        assert!(guard.check().is_ok());
        assert!(!guard.can_afford(41));

        guard.charge(u64::MAX);
        assert_eq!(guard.flops_used(), u64::MAX);
        assert!(matches!(guard.check(), Err(BrutError::Budget(_))));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn time_limit() {
        let dir = tmp_dir("time");
        let limits = Limits {
            max_total_seconds: -1.0,
            ..Limits::default()
        };
        let guard = Guard::new(&dir, &limits);

        assert!(matches!(guard.check(), Err(BrutError::Stopped(_))));
        assert!(guard.elapsed_seconds() >= 0.0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gap_check() {
        assert!(!gap_exceeded(0.9, 0.88, 0.05));
        assert!(gap_exceeded(0.9, 0.80, 0.05));
        assert!(!gap_exceeded(0.8, 0.9, 0.05));
        assert!(gap_exceeded(f64::NAN, 0.9, 0.05));
        assert!(gap_exceeded(0.9, 0.9, f64::NAN));
    }
}
