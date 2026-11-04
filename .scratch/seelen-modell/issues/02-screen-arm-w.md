# 02 Screen arm W against W0 and N on seeds 913 to 920

Status: resolved
Blocked by: 01
Spec: `../spec.md`, sections 3, 4.1 and 5

A screen, not a test: it decides whether a confirmatory experiment on arm W is worth its cost.
The criteria below are fixed before the first Run and never change after it.

## What to do

- Write `config-W.json`: the `config.json` of a pilot Run of arm S with `"variation": "prefer"`
  added to its `soul` object, as the pilot wrote `config-E.json`; the flags below override its
  seed and numbers. Write `config-W0.json`: the same with `"copy": false` and
  `"imprint": false`. Arm N has no Soul.
- Write `config-WF.json`: `config-W.json` with `"variation": "fixed"`. Arm WF is the upper
  bound (spec 4.1, Controls): every Worker prefers Linear, Phase and Hadamard at 2 and every
  other family at 1, and no Soul gives it. It shows whether preferences can shorten the wait
  at all, and whether the inherited Soul adds anything beyond a fixed, hand-made rule. Its 8
  Runs are reported for description only; they are no part of the criteria below.
- Run the four arms on seeds 913 to 920 with the settings of the pilot, two Runs at a time,
  each with `RAYON_NUM_THREADS=2` and `nice -n 5`, into a directory outside the repository:

  ```sh
  B=target/release/brut-breed
  for s in $(seq 913 920); do
    common="--domain circuits --search 33 --select 22 --children 8 --generations 1000 \
      --max-calls 13000000 --seed $s"
    $B --out runs/W-$s $common --config config-W.json
    $B --out runs/W0-$s $common --config config-W0.json
    $B --out runs/N-$s $common
    $B --out runs/WF-$s $common --config config-WF.json
  done
  ```

- After each Run: `brut-breed --out <dir> --verify`, the SHA-256 of the five record files, then
  `xz -6` on `events.jsonl` and `snapshot.json`, since 32 Runs would not fit on the disk
  uncompressed.
- Add the analysis to `.scratch/seelen-modell/analysis/` (Python standard library, run with
  `python3 -I`), reading the compressed files once each: the pass time; the divergence
  generation of W and N, and of W and W0, by the rule of `diagnose.py`; the first generation
  whose `variation_changed` is above 0; the mean preference for Linear, Phase and Hadamard over
  the Workers of the parents chosen in generations 41 to 50; and, before the first Worker that
  names all three families, the draws on a parent with a promising Worker: the share that changes
  a promising Worker, the share of those that loses one of its families, and the share that makes
  the Worker with all three. For arm WF, for description: the pass time, the divergence
  generations of WF and W and of WF and N, and the same draws on a parent with a promising
  Worker. Its preference is fixed, so it has no mean preference to measure.
- Write the results, criterion by criterion with their numbers, and the verdict to
  `.scratch/seelen-modell/screen-w.md`. Report a failed criterion as plainly as a met one. Give
  the numbers of arm WF beside them, marked as description: its pass times against those of
  W and N, and its share of draws that lose a family.

## Criteria (spec 4.1, fixed before the first Run)

1. **Early.** W parts from N before either Run passes on at least 7 of 8 seeds, and the weights
   change a choice in generation 1 or 2 on all 8.
2. **Self-adaptation.** The mean preference for Linear, Phase and Hadamard over the Workers of
   the parents chosen in generations 41 to 50 is above 1.2 in W on at least 6 of 8 seeds, and
   below 1.1 in W0 on at least 6 of 8.
3. **Protection.** Pooled over the seeds, before the first Worker with all three families, at
   most 40 % of the draws that change a promising Worker lose one of its families in W, and W's
   share lies at least 10 percentage points below that of W0 and below that of N. If an arm has
   fewer than 30 such draws, the criterion fails. The pilot's share was 57.5 %.
4. **Direction, reported.** The seeds on which W passes before N, and before W0. With no effect,
   W passes first on at least 5 of 8 seeds with probability 0.36, so this is a signal, not a
   test.

**Go** if 1, 2 and 3 hold and N passes before W on at most 4 of the 8 seeds. **Stop** if 1 or 2
fails. In every other case the owner decides. A go leads to a pre-registration on fresh seeds
(spec 8); seeds 913 to 920 never count.

## Done when

- All 32 Runs pass `--verify`, and their hashes, wall times and calls are recorded.
- `.scratch/seelen-modell/screen-w.md` holds every criterion with its numbers and the verdict,
  and the numbers of arm WF for description.
- The analysis scripts and their outputs are in `.scratch/seelen-modell/analysis/`.

## Comments

- 2026-11-03, done: the screen ran as fixed at 65ec9e3. All 32 Runs passed `--verify`. The
  verdict is stop, because criteria 1, 2 and 3 fail. Early: W parted from N before either Run
  passed on 8 of 8 seeds, but its weights changed a choice in generation 1 or 2 on only 7 of 8;
  on seed 914 the first change came in generation 5. Self-adaptation: W's mean preference was
  above 1.2 on 3 of 8 seeds, where 6 were needed; on the other 5 it lay below 1. Protection: W
  lost a family in 43.6 % of the draws that changed a promising Worker, where at most 40 % was
  allowed, against 42.3 % in W0 and 51.3 % in N. N passed before W on 6 of 8 seeds. Arm WF, for
  description: it passed before W on 8 of 8 seeds and before N on 5, and it lost a family in
  6.0 % of such draws. The results are in `../screen-w.md`. No confirmatory experiment on W
  follows; the owner adopts or drops ADR 0011.
