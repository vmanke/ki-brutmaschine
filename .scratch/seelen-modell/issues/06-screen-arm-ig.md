# 06 Screen arm IG against N, with arms G and I, on seeds 5001 to 5008

Status: needs-triage
Blocked by: 04, 05
Spec: `../spec-imagination.md`, sections 4 and 5

A screen, not a test: it decides whether a confirmatory experiment on arm IG is worth its cost.
The criteria below are fixed before any part is built, and they never change after it.

## What to do

- Write `config-N.json`: the `config.json` that the binary writes for arm N with the flags below.
  Write `config-G.json`, `config-I.json` and `config-IG.json`: the same with `goals`, with
  `imagination`, and with both. Arm N runs without `--config`, as in the W screen.
- Run the four arms on seeds 5001 to 5008, two Runs at a time, each with `RAYON_NUM_THREADS=2`
  and `nice -n 5`, into a directory outside the repository, from a binary copied out of the
  shared target directory (record its SHA-256):

  ```sh
  B=bin/brut-breed
  for s in $(seq 5001 5008); do
    common="--domain circuits --search 33 --select 22 --children 8 --generations 1000 \
      --max-calls 13000000 --selection lexicase --seed $s"
    $B --out runs/N-$s $common
    $B --out runs/G-$s $common --config config-G.json
    $B --out runs/I-$s $common --config config-I.json
    $B --out runs/IG-$s $common --config config-IG.json
  done
  ```

- After each Run: `brut-breed --out <dir> --verify`, the SHA-256 of the five record files, then
  `xz -6 -T1` on `events.jsonl` and `snapshot.json`.
- Add the analysis to `.scratch/seelen-modell/analysis/` (Python standard library, run with
  `python3 -I`), reading each compressed file once. Fix every open definition in its docstring
  before reading any output of the screen. Reuse the definitions of `analysis/screen_w.py` for the
  pass time, the divergence generation, promising Workers and the draws that change one.
- Write the results, criterion by criterion with their numbers, and the verdict to
  `.scratch/seelen-modell/screen-ig.md`. Report a failed criterion as plainly as a met one. Give
  the numbers of arms I and G beside them, marked as description.

## Criteria (fixed before any part is built)

1. **Early.** In IG, imagination changes a pick (the chosen draw is not the first) in generation 5
   or earlier, on 8 of 8 seeds.
2. **Learning.** In IG, the mean value for Linear, Phase and Hadamard over the Workers of the
   parents chosen in generations 41 to 50 is higher than the mean value of the other five
   families, on at least 6 of 8 seeds. A parent counts once per child, and both parents of a
   crossover count.
3. **Imagination measures the effect.** Pooled over the seeds of IG, among newborns chosen by
   imagination with p ≠ 0 and s ≠ 0, the sign of p equals s in more than half, with a one-sided
   binomial p-value below 0.05. With fewer than 30 such newborns, the criterion fails.
4. **Protection.** Pooled over the seeds, before the first Worker with all three families, at
   most 40 % of the draws that change a promising Worker lose one of its families in IG, and IG's
   share lies at least 10 percentage points below N's. If an arm has fewer than 30 such draws, the
   criterion fails.
5. **Direction, reported.** The seeds on which IG passes before N, I before N, G before N, IG
   before I, and IG before G. With no effect, an arm passes first on at least 5 of 8 seeds with
   probability 0.36, so this is a signal, not a test.

**Go** if 1, 2, 3 and 4 hold and IG passes before N on at least 5 of the 8 seeds. **Stop** if 1, 2
or 3 fails. In every other case the owner decides. A go leads to a pre-registration on fresh
seeds; seeds 5001 to 5008 never count. Arms I and G are reported for description: their values
for criteria 1 to 4 where they apply, and their pass times.

## Done when

- All 32 Runs pass `--verify`, and their hashes, wall times and calls are recorded.
- `.scratch/seelen-modell/screen-ig.md` holds every criterion with its numbers and the verdict,
  and the numbers of arms I and G for description.
- The analysis script and its outputs are in `.scratch/seelen-modell/analysis/`.
