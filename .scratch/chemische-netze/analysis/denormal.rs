//! Time per floating-point operation on normal and on subnormal f64 values, for `research.md`,
//! section 8.4. Our own measurement, not peer reviewed.
//!
//! Run: rustc -O -o /tmp/denormal denormal.rs && /tmp/denormal

use std::hint::black_box;
use std::time::Instant;

const VALUES: usize = 64;
const STEPS: usize = 200_000;
const ROUNDS: usize = 8;

/// Multiplies and divides every value by a factor, so that it stays near its start.
fn nanos_per_operation(start: f64, factor: f64) -> f64 {
    let start = black_box(start);
    let factor = black_box(factor);
    let began = Instant::now();
    let mut total = 0.0;

    for _ in 0..ROUNDS {
        let mut values = [0.0f64; VALUES];

        for (index, value) in values.iter_mut().enumerate() {
            *value = start * (1.0 + index as f64 * 0.001);
        }

        for _ in 0..STEPS {
            for value in values.iter_mut() {
                *value *= factor;
                *value /= factor;
            }
        }

        // Use every value. With only one of them, the compiler drops the work on the others.
        total += values.iter().sum::<f64>();
    }

    let seconds = began.elapsed().as_secs_f64();
    black_box(total);

    let operations = (ROUNDS * STEPS * VALUES * 2) as f64;
    seconds * 1e9 / operations
}

fn main() {
    let normal_start: f64 = 1.0;
    let subnormal_start: f64 = 1e-310;

    assert!(normal_start.is_normal());
    assert!(subnormal_start.is_subnormal());

    for round in 1..=3 {
        let normal = nanos_per_operation(normal_start, 0.75);
        let subnormal = nanos_per_operation(subnormal_start, 0.75);
        println!(
            "run {round}: normal {normal:.2} ns per operation, subnormal {subnormal:.2} ns, factor {:.1}",
            subnormal / normal
        );
    }
}
