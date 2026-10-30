# 01 Library spike: candle

Status: ready-for-agent
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
