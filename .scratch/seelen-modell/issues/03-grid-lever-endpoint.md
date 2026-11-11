# 03 Grid track: a pre-specified screen of the step 2 Soul on grids

Status: resolved
Blocked by: none (it needs no code; the owner chose this track on 2026-11-02)
Spec: `../spec.md`, sections 5, 6 and 7

The first grid screen (spec section 6) showed that the step 2 Soul, as built, acts from
generation 1 on the grid Domain, and that its arms part by generation 39. This ticket has two
phases. Phase 1, the text below, fixes the whole screen before any new Run: the question, the
arms, the settings, the seeds, the endpoint, the ties, the gates, the go/stop rule and the
analysis script. Its commit comes before the first Run of phase 2. Phase 2 runs the screen and
applies the rule unchanged. A go leads to phase 3, a pre-registration of a counted experiment on
other fresh seeds.

## Pre-specification (phase 1, fixed before any Run)

**Question.** On the grid Domain, does breeding with the step 2 Soul (arm S) find Cascades that
solve more search puzzles, sooner, than breeding without a Soul (N), with the same Soul read as
signs (V), and with the Coupling strength fixed at 0 (K)? This is a screen. It decides which of
the three comparisons, if any, earns a counted experiment. It makes no claim.

**Arms.** As built: S with `--soul s`, V with `--soul v`, K with `--soul k`, and N without a
Soul. The presets are those of section 12.2 of `.scratch/seele-und-kappa/spec.md`, with the
pilot's placeholders (`kappa_seed` 200, `kappa_drift` 20, `phase_drift` 512, `new_soul_theta`
1). The analysis checks them in every `config.json`.

**Settings.** `--domain grid --search 33 --select 22 --children 8 --generations 300 --max-calls
13000000 --seed s`. Everything else stays at its default: crossover rate 0.5, 3 peer opponents
with 2 puzzles per pair, judge weight 0.2, token cost 0.01 per 1000 tokens, Darwin Gödel
selection with λ 10 and α₀ 0.5, no LLM rounds, 3 finalists, the models `sim:small` and
`sim:large`, at most 4 Stages of at most 3 Workers. These are the settings of the first grid
screen. The 300 generations come from the cost, about one minute per Run in this container, not
from any outcome.

**Seeds.** 3001 to 3040, 40 seeds, in this order. No Run has used them:

- No document, ticket, script, test or log names a seed between 2000 and 5999. We searched
  `docs/`, `.scratch/`, `src/`, `tests/`, `examples/`, `backend-ollama/` and the READMEs of every
  worktree in this container, and of the main checkout.
- No Run directory in this container has such a seed. Their `config.json` seeds are 0 to 131,
  191, 193, 201 to 210, 301 to 310, 401, 402, 901 to 920 and 1001.
- The seed lists of the earlier experiments end below 2000. Step 1 counted 1 to 100 and 111 to
  130, with pilot seeds 101 to 110, 201 to 210, 301 to 310, 401 and 402. The withdrawn step 2
  pre-registration listed 1002 onward, at most 120 seeds. The selection study used 1 to 30. The
  real-model vote pre-registration counts from 1, with pilot seeds 901 to 905.

Seeds 913 to 920 ran in the first grid screen and never count. A counted grid experiment after a
go takes another fresh block, from 3101 on, named in its own pre-registration.

**Endpoint.** For a Run, let B(g) be the largest `solved` count among the Cascades born in
generations 1 to g, read from the `born` events; the search panel has 33 puzzles. The endpoint is
A = B(1) + B(2) + ... + B(300), an integer from 0 to 9 900. A larger A means that breeding found
Cascades that solve more search puzzles, and found them sooner. The seed Cascades of generation 0
do not count, as only bred Cascades count in step 1.

**Why this endpoint.**

- It keeps the idea of the step 1 pass time: how soon breeding finds Cascades that solve more. It
  drops the threshold, since a threshold time ties whenever both Runs cross it before they part.
  Those ties left step 2 without power on circuits (spec section 1).
- It is measured on the search panel, where fitness is measured, as the targets of step 1 are.
  The select panel stays descriptive, as everywhere in this repository (`CONTEXT.md`, "Select
  panel").
- It uses every generation up to 300. So two Runs tie only if their best counts agree in every
  generation after they part.
- B never falls, since it is a running maximum. A Cascade found earlier counts in every later
  generation: sooner is better, as with a pass time.

**What we know already.** The first grid screen gave the first votes and flips, the divergence
generations, and the best solved count at generation 300 per arm (S above N on 3 seeds, below on
4, equal on 1). The gates below take its ranges with a margin. We have not computed A on its 32
Runs and will not: the analysis script reads only seeds 3001 to 3040.

**Outcome per seed.** For each control X (N, V or K), seed s gives "S higher" when A_S > A_X, "X
higher" when A_X > A_S, and a tie when they are equal. Every comparison is between integers.

**Ties.** A tie counts for neither side. The analysis reports the ties, and among them the seeds
on which the two Runs never parted by generation 300. More than 10 ties in a comparison make the
endpoint too coarse there: that comparison gets no go, whatever its count.

**Gates.**

- **G1, integrity.** Every Run finishes its 300 generations (`stop_reason` "all generations
  done"), passes `--verify`, and its `config.json` holds exactly the settings above. A Run that
  fails voids its seed in all three comparisons. More than 2 void seeds void the screen.
- **G2, the Soul acts early.** Arm S has a flipped vote by generation 50 on at least 9 in 10 of
  the valid seeds (36 of 40).
- **G3, the arms part early.** Arm S parts from each of N, V and K before generation 100 on at
  least 9 in 10 of the valid seeds. Parting follows the rule of `diagnose.py`: the first
  generation whose newborns differ, member by member in birth order, in genes without Souls,
  operator, parents by genes or solved count. Grid `born` events carry no answers and no graded
  value.

If G2 or G3 fails, the premise of the grid track fails, and the screen stops. In the first
screen the first flip came by generation 30 and the arms parted by generation 39.

**Go/stop rule, per comparison.** Let w be the seeds with S higher, l those with X higher, and
u = w + l the untied seeds. Let k(u) be the smallest count with P(W ≥ k(u)) ≤ 1/20 for W
binomial with u trials and probability 1/2 (exact, one-sided).

- **Go** if G1, G2 and G3 hold, the comparison has at most 10 ties, and w ≥ k(u).
- **Stop** if w ≤ u/2.
- Otherwise the comparison is open, and the owner decides.
- If l ≥ k(u), we report that X is ahead of S by the same rule.

| u | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k(u) | 19 | 20 | 20 | 21 | 22 | 22 | 23 | 23 | 24 | 24 | 25 | 26 | 26 |

**Errors and power** (our calculation). With no effect, a comparison goes with probability at most
0.05, and some comparison of the three with probability at most 0.15. At u = 40 a comparison goes
with probability 0.32, 0.57, 0.81 and 0.95 when P(S higher | untied) is 0.60, 0.65, 0.70 and
0.75. A go is no claim. It earns a pre-registered counted experiment on fresh seeds, with p₁ and
N_max set from this screen's estimates and tie rate.

**Descriptive results, no criterion.** Per arm: A, B(300), the first complex and the first
flipped vote, the generations with a flip among 1 to 50, the calls, the seconds per Run and the
best finalist's rate on the select panel. Per comparison: the divergence generations and the
median of A_S − A_X.

**Analysis.** `.scratch/seelen-modell/analysis/grid_screen.py`, written with this
pre-specification, Python standard library only, run with `python3 -I -B`. It was tested only on
the synthetic Runs of its `--self-test`. It runs once, after all 160 Runs have finished and passed
`--verify`.

## Run plan (phase 2)

- Build with `cargo build --release --bin brut-breed` and the existing build cache. `src/` has
  not changed since commit `e315220`, so the binary's SHA-256 must be
  `898b7b4082c93b115c4992ac76f2145e7d57b629a36e34912a6903079961c220`, the pilot's. Another hash
  stops the screen before its first Run.
- Run the seeds 3001 to 3040 in order, and for each seed the arms S, V, K and N. Two Runs go at a
  time, each with `RAYON_NUM_THREADS=2` and `nice -n 5`, into a directory outside the repository:

  ```sh
  B=target/release/brut-breed
  for s in $(seq 3001 3040); do
    common="--domain grid --search 33 --select 22 --children 8 --generations 300 \
      --max-calls 13000000 --seed $s"
    $B --out runs/S-$s $common --soul s
    $B --out runs/V-$s $common --soul v
    $B --out runs/K-$s $common --soul k
    $B --out runs/N-$s $common
  done
  ```

- After each Run: `brut-breed --out <dir> --verify`, with its output and its exit code
  ("verify exit 0") in `<dir>.verify`; the wall seconds in `<dir>.wall`; the SHA-256 of
  `config.json`, `events.jsonl`, `snapshot.json` and `report.json` in `hashes.txt`. Then `xz -6`
  compresses `events.jsonl` and `snapshot.json`.
- Nobody looks at A, B or a comparison before all 160 Runs have finished and passed `--verify`.
  While the Runs go, the driver looks only at exit codes and `--verify`.
- Cost: about one minute per Run (43 to 64 seconds in the first grid screen). With `--verify` and
  compression, the first screen used 31 seconds of wall time per Run at two Runs at a time, so the
  160 Runs take about 1.5 hours. They need about 0.35 GB once compressed (1.3 to 2.6 MB per Run).
- Then run `python3 -I -B .scratch/seelen-modell/analysis/grid_screen.py <run root>` once. Write
  its output, the verdicts, the binary's hash and the Runs' hashes to
  `.scratch/seelen-modell/grid-screen.md`.

## Done when

- Phase 1: this pre-specification and `analysis/grid_screen.py`, with its self-test passing, are
  committed before the first Run of phase 2.
- Phase 2: all 160 Runs are done and checked, and `.scratch/seelen-modell/grid-screen.md` holds
  every gate and every comparison with its numbers and its verdict. Nothing in the
  pre-specification changed; a deviation is reported with its reason.
- Phase 3, only after a go: a pre-registration of the counted grid experiment, approved by the
  owner before its first Run.

## Comments

- 2026-11-02, phase 2 done: the screen ran as pre-specified at dbbfc7d. All 160 Runs passed
  `--verify`, and all three gates hold: the Soul flipped a vote by generation 50 on 40 of 40
  seeds, and S parted from each control before generation 100 on 40 of 40. No comparison
  earned a go. S against N: 21 to 18 with 1 tie (open). S against V: 20 to 18 with 2 ties
  (open). S against K: 18 to 20 with 2 ties (stop). The results are in `../grid-screen.md`.
  So on grids the step 2 Soul acts early and often, but its effect on the endpoint is near a
  coin toss. Phase 3 does not start without a go; the owner decides on the two open comparisons.
- 2026-11-11, the owner's decision. Both open comparisons, S against N and S against V, are
  closed. No counted grid experiment is pre-registered, so phase 3 does not start. Section 7 of
  `../grid-screen.md` holds the decision and the calculations the owner had.
