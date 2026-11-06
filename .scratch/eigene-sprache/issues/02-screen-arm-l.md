# 02 Screen arm L against N and R on seeds 6001 to 6008

Status: needs-triage
Blocked by: 01
Spec: `../spec.md`, section 3

A screen, not a test: it decides whether a confirmatory experiment on arm L is worth its cost.
The criteria below are proposals for the owner. Once the owner approves them, they are fixed
before any part is built, and they never change after the first Run.

## What to do

- Arms: N (no Lexicon), R (`"meaning": "random"`) and L (`"meaning": "compression"`). All three use
  lexicase, the standard since the owner's decision "L gewinnt", `"brevity": true`, and the other
  flags of the IG screen. No arm has a Soul, imagination or goals.
- Write `config-N.json` (the `config.json` the binary writes with the flags below and
  `"brevity": true`), and `config-R.json` and `config-L.json`, which add the `lexicon`. Every arm
  runs with `--config`.
- Run the three arms on seeds 6001 to 6008, which have never been used, two Runs at a time, each
  with `RAYON_NUM_THREADS=2` and `nice -n 5`, from a binary copied out of the shared target
  directory (record its SHA-256). Use the slots of the IG screen (`slot.sh`), so that no Run
  meets the session's limit of 7200 s per background task:

  ```sh
  common="--domain circuits --search 33 --select 22 --children 8 --generations 1000 \
    --max-calls 13000000 --selection lexicase --seed $s"
  ```

- After each Run: `brut-breed --out <dir> --verify`, the SHA-256 of the five record files, then
  `xz -6 -T1` on `events.jsonl` and `snapshot.json`.
- Add the analysis to `.scratch/eigene-sprache/analysis/` (Python standard library, run with
  `python3 -I`), reading each compressed file once. Fix every open definition in its docstring
  before reading any output of the screen. Reuse `analysis/screen_w.py` and `screen_ig.py` of
  `.scratch/seelen-modell/` for the pass time.
- Write the results, criterion by criterion with their numbers, and the verdict to
  `.scratch/eigene-sprache/screen-l.md`. Report a failed criterion as plainly as a met one. Give
  the numbers of arm R beside them, marked as description.
- Time: in the IG screen a Run took 23 to 102 minutes, two at a time. 24 Runs take about 8 hours.

## Criteria (proposed; fixed once the owner approves them)

1. **Invented and used.** In L, a newborn whose Workers hold a coined phrase is born in
   generation 100 or earlier, on 8 of 8 seeds.
2. **Efficient.** The mean prompt words of the parents chosen in generations 101 to 200, summed
   over each parent's Workers, are at least 5 % lower in L than in N, on at least 6 of 8 seeds. A
   parent counts once per child, and both parents of a crossover count.
3. **The choice matters.** L passes before R on at least 5 of 8 seeds. A tie counts as not
   before.
4. **Direction.** L passes before N on at least 5 of 8 seeds. A tie counts as not before.

**Go** if 1, 2, 3 and 4 hold. **Stop** if 1 or 2 fails. In every other case the owner decides.
With no effect, one arm passes first on at least 5 of 8 seeds with probability about 0.36, so
criteria 3 and 4 are signals, not tests. A go leads to a pre-registration on fresh seeds; seeds
6001 to 6008 never count. For description: the words each Run coined and retired, with their
meanings and generations, and how often a word joined two of Hadamard, Phase and Linear, in L
and in R.

## Done when

- All 24 Runs pass `--verify`, and their hashes, wall times and calls are recorded.
- `.scratch/eigene-sprache/screen-l.md` holds every criterion with its numbers and the verdict.
- The analysis script and its outputs are in `.scratch/eigene-sprache/analysis/`.

## Comments

- 2026-11-06: not run, by the owner's decision of the same day; ticket 01 is built. Before this
  screen is approved, note the probes in the Comments of ticket 01. With the section-name rule
  now in spec 3.2, a coined word joins "strategy" and spreads about as often as any strategy
  phrase. In 6 probes of 100 generations, at most 5 of the 22 best members held one in any
  generation, and at most 2 at generation 100. Whether 1000 generations give it time to act is
  open.
