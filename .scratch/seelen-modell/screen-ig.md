# Screen of arm IG: the result

Status: done on 2026-11-05. Verdict by the fixed rule: **stop**. Criterion 1 fails, and so does
criterion 4. Criteria 2 and 3 hold.

This is the screen of `issues/06-screen-arm-ig.md`, run as that ticket says. Its criteria were
fixed before any part was built (`ccf059e`), and criterion 3 was revised in `da5d9cb`, before
any Run. They are unchanged here. Seeds 5001 to 5008 never count, so no claim follows from
these numbers.

## How it ran

- **Program.** `cargo build --release --bin brut-breed` at commit `00abe5c` (the build of
  tickets 04 and 05, pull request #57), with rustc 1.97.0 (2d8144b78 2026-07-07) on x86_64
  Linux. The binary was copied out of the shared target directory before the first Run. Every
  Run and every `--verify` used that copy. Its SHA-256 is
  `db177f9aa016b3386c05c652ee5a8c9bcfa08fcdab69a70ff6df03dcd4a8c6c5`. The checksum was checked
  again after each restart of the container.
- **Settings.** `--domain circuits --search 33 --select 22 --children 8 --generations 1000
  --max-calls 13000000 --selection lexicase --seed <s>`. Arm N runs without `--config`.
  `config-N.json` is the `config.json` that the binary wrote for arm N with these flags on seed
  5001. `config-G.json` adds `"goals": {"length": 20}`, `config-I.json` adds
  `"imagination": {"draws": 4, "rate": 250}`, and `config-IG.json` adds both. All four are in
  `analysis/driver-ig/`.
- **Order and threads.** The queue of the ticket: for each seed from 5001 to 5008, N, G, I and
  IG. Two Runs ran at a time, each with `RAYON_NUM_THREADS=2` and `nice -n 5`, on 4 cores, from
  2026-11-04T14:45Z to 2026-11-05T04:54Z. No other experiment ran on these cores in that time.
- **Interruptions.** The session ends each background task after 7200 s. `driver.sh` ran the
  queue in one such task and was stopped at 16:45:03Z, with G-5002 and I-5002 unfinished. The
  owner chose to go on here. From 19:28Z, two slots (`slot.sh`) ran the queue, each in its own
  task. A slot starts a further Run only if that Run can end within its task. The container
  restarted at about 20:40Z and stopped both slots, with N-5003 and G-5003 unfinished. At
  01:16Z, while IG-5005 was still running, `slot.sh` changed, so that a new slot always starts
  its first Run. IG-5005 took 5945 s. With the old rule, which asks for 1.25 times the longest
  Run, no slot would have started a Run after it. The driver's log records each step.
- **Reruns.** Each of the four unfinished Runs was moved aside and ran again from generation 0
  in a fresh directory. No file of an unfinished Run is used. The record files of a Run depend
  only on its settings and seed, not on the load or the thread count; the tests of tickets 04
  and 05 check this at 1 and 4 threads.
- **After each Run.** `brut-breed --out <dir> --verify`, then the SHA-256 of the five record
  files, then `xz -6 -T1` on `events.jsonl` and `snapshot.json`. The scripts are in
  `analysis/driver-ig/`, with the driver's log and the hashes. In the copy of the log, the local
  base folder reads `/path/to/ig-screen`.
- **Analysis.** `analysis/screen_ig.py` (Python standard library, `python3 -I`) ran once, after
  all 32 Runs had passed `--verify`. It reads each `events.jsonl.xz` once through `lzma`, and
  each Run's `config.json`, `report.json` and `rediscovery.json`. It takes the pass time and the
  divergence generation from `.scratch/seele-und-kappa/rule/reference_rule.py`, and the
  definitions of promising Workers and of the draws that change one from `analysis/screen_w.py`.
  Its docstring fixes every definition that the ticket leaves open. Its output is
  `analysis/screen_ig.txt` and `analysis/screen_ig.json`. It reported no deviation and no
  anomaly.

## Every Run passed `--verify`

All 32 Runs ran 1000 generations to the end ("all generations done"), with the settings of their
arm. All 32 `--verify` lines say "hash chain intact; checkpoint ... matches; settings match;
report recomputed and equal; rediscovery recomputed and equal". The Runs used 0.97 to 2.73
million calls, far below the cap of 13 million. The full SHA-256 of the five record files of
every Run are in `analysis/driver-ig/hashes.txt`.

| Run | Pass time | Calls | Minutes | Events | Head of the log |
|---|---|---|---|---|---|
| N-5001 | 115 | 2 427 119 | 40.0 | 51 847 | `2f4885a7a374cb19…` |
| G-5001 | 134 | 2 219 984 | 30.7 | 41 394 | `4c7a1cd36fcd0640…` |
| I-5001 | 90 | 1 105 419 | 66.9 | 53 366 | `eb2d12ebc87bdbd9…` |
| IG-5001 | 187 | 2 729 845 | 35.9 | 52 190 | `094e69c9a91cbd77…` |
| N-5002 | 90 | 1 098 459 | 25.8 | 44 379 | `a025dd5b094c2dd4…` |
| G-5002 | 65 | 1 519 379 | 26.3 | 42 840 | `7839357229e718da…` |
| I-5002 | 8 | 1 230 115 | 38.4 | 37 336 | `b05b42f4f75a9cc8…` |
| IG-5002 | 8 | 1 341 574 | 31.6 | 48 143 | `a1fd45d22a6607cc…` |
| N-5003 | 22 | 1 413 146 | 34.7 | 51 419 | `f95396cc753695f7…` |
| G-5003 | 272 | 1 865 439 | 26.2 | 52 659 | `b275fc37a4418f72…` |
| I-5003 | 46 | 1 082 545 | 24.1 | 45 091 | `6ae5ab0df815aa56…` |
| IG-5003 | 92 | 1 344 820 | 27.6 | 50 298 | `87bc548b644e416a…` |
| N-5004 | 28 | 2 189 507 | 65.7 | 45 835 | `358618f1f3ab6eeb…` |
| G-5004 | 43 | 1 021 381 | 23.9 | 38 274 | `f1875fa50e1f6b35…` |
| I-5004 | 500 | 1 642 323 | 26.0 | 41 449 | `c72588920ed75bd6…` |
| IG-5004 | 186 | 1 065 764 | 28.6 | 49 876 | `6b1b712ef27bad56…` |
| N-5005 | 9 | 1 181 556 | 29.8 | 44 475 | `a76adb3004cbcf6f…` |
| G-5005 | 9 | 1 450 416 | 34.3 | 53 106 | `0d8b375bfbb11bd1…` |
| I-5005 | 9 | 1 494 697 | 34.4 | 38 872 | `e46245d20d17ec7a…` |
| IG-5005 | 9 | 1 093 140 | 99.1 | 50 792 | `9f8a63380f18c896…` |
| N-5006 | 14 | 1 704 266 | 37.6 | 42 632 | `e72ebd916bade40d…` |
| G-5006 | 14 | 1 019 547 | 22.5 | 29 570 | `68305fbd125e6b1a…` |
| I-5006 | 14 | 1 096 890 | 29.3 | 25 070 | `3b0ecabd8be5fd20…` |
| IG-5006 | 14 | 970 767 | 24.2 | 25 926 | `989882deca326444…` |
| N-5007 | 142 | 1 222 200 | 39.2 | 50 768 | `c5120fc03ce0bbb7…` |
| G-5007 | 59 | 1 139 357 | 93.0 | 45 458 | `0663b220c71fe55d…` |
| I-5007 | 109 | 1 685 886 | 37.5 | 51 929 | `a85ce5ca48f6d29a…` |
| IG-5007 | 68 | 1 164 305 | 33.0 | 50 813 | `b16c69162ea79032…` |
| N-5008 | 68 | 2 016 024 | 102.3 | 53 591 | `345e94dbfb739150…` |
| G-5008 | 154 | 2 459 082 | 32.7 | 50 780 | `259449c013d8c859…` |
| I-5008 | 73 | 1 081 020 | 23.0 | 35 637 | `8687ffb027766550…` |
| IG-5008 | 198 | 1 471 556 | 36.7 | 46 454 | `80ddf6316e9069d1…` |

## Per seed

T is the pass time of step 1; every Run passed by generation 1000. The divergence generation of
two arms is the first generation whose newborns differ. "First change" is the first generation
in which imagination changed a pick, so that the chosen draw is not the first. The mean values
are those of the Workers of the parents chosen in generations 41 to 50, in thousandths, once for
Linear, Phase and Hadamard and once for the other five families.

| Seed | T_N | T_G | T_I | T_IG | IG-N | I-N | G-N | First change I | IG | I: L, P, H | I: others | IG: L, P, H | IG: others |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5001 | 115 | 134 | 90 | 187 | 5 | 5 | 21 | 5 | 5 | 137.14 | −5.48 | 59.62 | −4.12 |
| 5002 | 90 | 65 | 8 | 8 | 3 | 3 | 21 | 3 | 3 | 97.81 | 12.75 | 99.54 | 12.64 |
| 5003 | 22 | 272 | 46 | 92 | 4 | 4 | 21 | 4 | 4 | 19.56 | 0.32 | 0.98 | −0.06 |
| 5004 | 28 | 43 | 500 | 186 | 5 | 5 | 21 | 5 | 5 | 0.08 | 0.92 | 0.57 | 0.22 |
| 5005 | 9 | 9 | 9 | 9 | 10 | 10 | 21 | 10 | 10 | 107.66 | 17.12 | 107.64 | 16.80 |
| 5006 | 14 | 14 | 14 | 14 | 6 | 6 | 21 | 6 | 6 | 28.31 | −0.32 | 30.28 | −0.40 |
| 5007 | 142 | 59 | 109 | 68 | 13 | 13 | 21 | 13 | 13 | 28.08 | 0.00 | 24.53 | 0.55 |
| 5008 | 68 | 154 | 73 | 198 | 8 | 8 | 21 | 8 | 8 | 12.36 | 6.53 | 2.66 | −0.48 |

Arms G and N part in generation 21 on every seed, when the first goal without a family begins.
Arms I and IG have the same first change on every seed, because both work on the goal of all
tasks until generation 20.

## The criteria

1. **Early: failed.** In IG, imagination changed a pick in generation 5 or earlier on 4 of 8
   seeds (5001, 5002, 5003 and 5004); all 8 were needed. On the other four seeds the first change
   came later: in generation 10 on seed 5005, 6 on 5006, 13 on 5007 and 8 on 5008.
2. **Learning: met.** In IG, the parents' mean value for Linear, Phase and Hadamard in
   generations 41 to 50 was above that of the other five families on 8 of 8 seeds; 6 were
   needed. On seeds 5003 and 5004 the gap is small: 0.98 against −0.06 thousandths, and 0.57
   against 0.22 thousandths.
3. **Imagination measures the effect: met.** On each seed of IG, D is the mean s of the newborns
   with p > 0 minus the mean s of those with p < 0. D was above 0 on 7 of 8 seeds; 7 were
   needed. It was below 0 on seed 5003 (−0.085). No seed had fewer than 5 newborns in a group;
   the smallest group held 23. The per-seed values of D were 0.24, 0.18, −0.08, 0.17, 0.38, 0.24,
   0.26 and 0.03. Pooled over the seeds, 2783 newborns had p > 0, with mean s −0.17, and 2907
   had p < 0, with mean s −0.36; D = 0.19.
4. **Protection: failed.** Pooled before the first Worker with all three families, IG lost a
   family in 150 of the 346 draws that changed a promising Worker, 43.4 %; at most 40 % was
   allowed. N lost one in 142 of 260, 54.6 %. So IG lies 11.2 points below N, and that part of
   the criterion holds. Both arms had more than 30 such draws.
5. **Direction, reported.** IG passed before N on 2 seeds (5002 and 5007), N before IG on 4
   (5001, 5003, 5004 and 5008), and both in the same generation on 2 (5005 and 5006). I passed
   before N on 3 seeds and after N on 3, with 2 ties. G passed before N on 2 seeds and after N
   on 4, with 2 ties. IG passed before I on 2 seeds and after I on 3, with 3 ties. IG passed
   before G on 2 seeds and after G on 4, with 2 ties.

**Verdict.** Criterion 1 fails, so the fixed rule says stop. Criterion 4 fails too, and IG
passes before N on 2 seeds, where a go needed 5.

On seeds 5005 and 5006 all four arms passed in the same generation (9 and 14). On seed 5005 the
pass came before the arms parted in generation 10. On seed 5006 every arm made its first Worker
with all three families in generation 14, each by a section replacement.

## Arms I and G, for description only

These numbers are no part of the criteria.

- **Criterion 1 in I.** Imagination changed a pick by generation 5 on 4 of 8 seeds, in the
  same generations as in IG.
- **Criterion 2 in I.** The mean for Linear, Phase and Hadamard was above that of the others on
  7 of 8 seeds; on seed 5004 it was 0.08 against 0.92 thousandths.
- **Criterion 3 in I.** D was above 0 on 8 of 8 seeds. Pooled, 3621 newborns had p > 0, with
  mean s −0.16, and 3593 had p < 0, with mean s −0.28; D = 0.11.
- **Protection.** I lost a family in 174 of the 395 draws that changed a promising Worker,
  44.1 %. G lost one in 209 of 391, 53.5 %.
- **Imagination over the Run.** In IG, imagination chose between 818 and 2538 children per Run,
  and it changed between 319 and 1487 picks. In I, the ranges were 522 to 3012 and 202 to 1851.
  Every seed's counts are in `analysis/screen_ig.txt`.
- **Goals.** In G and IG the goals ran as fixed on every seed: 50 blocks of 20 generations, the
  goal of all tasks every other block, then −Linear, −Phase, −Hadamard, −Compose, −Search,
  −Meet and −Simplify in turn. They were the same in both arms.

A reading, not a test: the learned values carry information. On 15 of the 16 Runs of I and IG,
a newborn with p > 0 fared better than one with p < 0, and the parents valued Linear, Phase and
Hadamard more than the other families on 15 of 16. Still, in both groups more children lost
than gained: the mean s was below 0 on every seed. And imagination starts late. Its values begin at
0, and they move only after children whose Workers gained or lost families are evaluated. So on
half the seeds the first changed pick came after generation 5. Protection improved by about 11
points against N, but not below 40 %, and the pass did not come earlier: IG passed before N on 2
seeds and after N on 4.

## Deviations

- **Interruptions.** The background limit and a restart of the container stopped four Runs:
  G-5002 and I-5002 at the limit, N-5003 and G-5003 at the restart. Each ran again from the
  start, as described above. My estimate of the screen's time was wrong by about a factor of
  four, because these Runs took 23 to 102 minutes each, against 5 to 12 in the screen of W.
- **`slot.sh` changed during the screen.** At 01:16Z a new slot was set to always start its
  first Run. The version in `analysis/driver-ig/` is the one after the change. The log notes the
  change. It changes only when a Run starts, not how it runs.
- **The analysis was committed 12 s before the first Run.** Commit `8e3527f` holds
  `screen_ig.py`, the configurations and the driver; it was made at 14:44:53Z. The first Run
  started at 14:45:05Z. The push reached GitHub at about 14:45:06Z, when the check of that
  commit began. No output of a Run was read before the analysis ran.
- **Smoke Runs were seen before.** For tickets 04 and 05, four smoke Runs of 60 generations ran
  on seed 4242, which is no screen seed, with the screen's flags. In I and IG their first
  changed pick came in generation 11, and D was −0.13 in both. Both facts were disclosed in pull
  request #57 before the screen. The criteria stayed as fixed. These Runs are no part of the
  screen.
- **Tool test.** Before the screen, the script ran on those smoke Runs, with no anomaly.
- **Checks after the analysis.** After its single run, separate code recounted criteria 1 and 3
  from the event logs alone, for I and IG on all 8 seeds: the first changed pick from the Born
  events and from the state lines, and the counts, mean s and D. All agree with the script.
