# 01 Library spike: candle

Status: done
Start: after the counted Runs of step 1 have finished.
Spec: `../spec.md`

## What to do

- In a throwaway crate outside the repository, add `candle-core` and `candle-nn` 0.11.0
  with default features only (CPU, no MKL, no CUDA).
- Run `cargo audit` on that crate. Record every advisory of candle and of each new
  transitive dependency. Record the licences and the number of new crates.
- Train one small CNN (two 3×3 convolutions, pooling, a dense head) on a fixed toy batch
  for a few steps, from a fixed seed. Compare the weights and losses byte for byte at
  `RAYON_NUM_THREADS=1` and `4`, and between two runs at the same thread count.
- Measure the time per training step for a 16×16 input batch of 64.

## Done when

- A short report in this file (Comments) gives: versions, licences, advisories (none
  open, or the crate is rejected), new crate count, determinism result, step times.
- If training is not byte-equal across thread counts, the report names the thread
  setting that makes it so, or says that only a fixed thread count is reproducible.
- If candle fails a check, the report compares burn on the same points.

Cost: about 8 to 15 dollars.

## Comments

**Report, 2026-10-31.** The library is candle, from GitHub at commit
`9515d410e3af803fd45efebfef8137f47d85a63d` of its main branch, with default features only
(ADR 0007). Two runs back it: the spike, and an independent check that rebuilt the spike and
added its own harness and network. A third run repeated build, audit, gradient check and
determinism test at that commit.

- **Versions.** candle-core and candle-nn 0.11.0 are the newest release; the commit also
  calls itself 0.11.0. The release needs zip 8.6.0, the commit zip 9.1.3. gemm 0.19.0 does
  the matrix products in both.
- **Licences.** candle-core and candle-nn: MIT OR Apache-2.0. The 103 new crates of the
  commit are all permissive: MIT or Apache-2.0 (59, in several spellings), MIT (25),
  Apache-2.0 (4), Unicode-3.0 (4), Zlib or Apache-2.0 or MIT (4), Unlicense or MIT (2),
  Apache-2.0 with LLVM exception or Apache-2.0 or MIT (2), Zlib (1), Apache-2.0 or BSL-1.0
  (1), BSD-2-Clause or Apache-2.0 or MIT (1).
- **Advisories.** cargo-audit 0.22.2, advisory database at commit 2295466c (2026-10-30,
  1395 advisories). The release fails: exit 1, three vulnerabilities in zip 8.6.0
  (RUSTSEC-2026-0367, -0368, -0369, all fixed from zip 9.0.0 on), and candle-core requires
  zip 8.6.0. So the release is rejected. The repository with the commit passes: 164
  packages, exit 0, one warning, RUSTSEC-2024-0436 (paste 1.0.15, unmaintained, a
  compile-time macro crate of gemm).
- **New crates.** The compiled tree for x86_64-unknown-linux-gnu grows from 34 to 137
  crates, so 103 are new; `Cargo.lock` grows from 40 to 164 packages, and the 40 keep their
  versions. `cargo tree -i cc` finds no `cc`, so the build needs no C or C++ compiler (the
  release needed a C compiler for Oniguruma). No new crate declares a Rust version above
  1.88; we built with 1.97.0 only.
- **Gradients.** In f64 against finite differences, 17 of 60 native backward cases are
  wrong or fail, through 8 defects (D1 to D8), at the release and at the commit. Each has a
  replacement (W1 to W8); all 30 replacement cases are correct. `src/cnn/ops.rs` now holds
  them, with the same check on every public operation.
- **Determinism.** Training is byte-equal at every thread setting tested: 19 settings
  (RAYON_NUM_THREADS 1 to 4 and unset, installed pools, `Device::with_context`,
  CANDLE_NUM_THREADS, gemm's threading threshold at 1), one hash of the losses and one of
  the final state per variant, at the release and at the commit. So no thread setting is
  needed; 1 and 4 threads give the same bits. The bits of the commit differ from those of
  the release (final loss 0.238868 against 0.238249). Other CPUs were not tested.
- **Step times.** One step (forward, backward, AdamW) of the 2736-parameter CNN on 64
  images of 16x16 took 42 to 51 ms at 1 and at 2 threads, at the release, with the
  experiment of step 1 running. A second thread changed the time by a factor of 0.91 to
  1.2. The commit was not timed.
- **burn, for comparison.** burn 0.22.1 (std, ndarray, autodiff, optim) declares Rust 1.95
  and compiles 144 crates, 133 of them new. Its lockfile of 608 packages fails cargo audit
  as well: exit 1, 4 vulnerabilities (zip 0.6.6, zstd-safe 5.0.2) and 9 warnings, through
  tch and torch-sys, which these features do not compile.
