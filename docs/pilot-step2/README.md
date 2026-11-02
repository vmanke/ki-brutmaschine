# Step 2 pilot runs

These are the pilot Runs of the step 2 comparison (ticket 11 of `.scratch/seele-und-kappa/`), not
counted Runs. Pilot seeds never count: 901 to 923. The pilot ran seeds 901 to 912 in five arms:
S (the Soul), V (the same Soul read as signs), K (κ fixed at 0), E (arm S with equal Family
strengths) and N (no Soul, arm A of step 1). Seeds 913 to 920 were kept for a second batch with
changed placeholders, which the pilot did not need (section "Degenerate? No second batch"); 921
to 923 are kept for acceptance triples in case a preset changes (PREREG section 7). The Run
directories are not in the repository. The draft pre-registration that these Runs inform is
[docs/PREREG_STEP2_SOUL.md](../PREREG_STEP2_SOUL.md).

**In short.** All 60 Runs passed `--verify`, and seed 901, run again in all five arms, gave
byte-equal files. The pilot is not degenerate in the sense of ticket 11: complex votes flipped
winners in 45 of the 48 Runs with Souls, the phases decided simulated replies in all 48, and the
arms S, V and K parted on every seed. But the Soul changes no outcome before its first flipped vote
or its first phase-decided reply. Every tie of either claim came about because arm S and the
control passed before their two Runs first differed, and in 4 of the 12 triples all three Runs had
passed by then. Among the untied triples arm S passed first 4 of 8 times against arm V and 3 of 6
times against arm K, so both estimates of P(S first | untied) are 1/2. At these estimates the
proposed rule (p₁ = 3/4, N_max = 120) proves claim V with probability 0.0038 and claim K with
probability 0.0034, and averaged over the uncertainty of the estimates with 0.1485 and 0.1528.
**With N_max = 120 the power of each claim at the pilot's estimate is far below 0.5. We therefore
propose not to run the counted experiment of ticket 10 as designed**, or, which comes to the same,
to drop both claims and report step 2 descriptively. The owner decides before ticket 10 is frozen.
In the pilot, arm S never passed before arm N: it passed later on 5 seeds and at the same
generation on 7. This describes 12 seeds; it is no test.

This folder holds:

| File | What it is |
|---|---|
| `pilot.json` | Per Run: the pass time, D, calls, wall time, the vote tallies, the `soul` group of every 10th generation, the `--verify` result and the SHA-256 of the five record files. Per seed: Y_V, Y_K (and the outcomes against E and N) and the divergence generation of each pair of arms. Per arm: pass rates, the shares of flipped and of tied votes, `sim_decided`, minutes and calls. The reruns of seed 901. Its `notes` explain the fields. |
| `analysis.txt` | The output of `examples/soul_comparison.rs` on the 60 Runs. |
| `recount.txt` | The output of `.scratch/seele-und-kappa/rule/reference_rule.py --pilot 901-912`. |
| `driver/` | The scripts that ran the pilot, and `config-E.json`, the settings of arm E. Only the line `ROOT=` differs from the scripts that ran. |
| `driver.log` | The driver's log: the start, end, `--verify` line and compression of every Run, and the reruns. |
| `rerun.log` | The comparison of the five record files of each rerun with the pilot's hashes. |

The scripts of the analysis of the pilot are in `.scratch/seele-und-kappa/rule/`: `pilot.py`
builds `pilot.json` (its summary is `pilot.txt`), `diagnose.py` finds why and when two arms part
(`diagnose.txt`), `oc.py` computes the operating characteristics of the proposed rule (`oc.txt`),
and `own_check.py` recounts them with other methods (`own_check.txt`).

## How the pilot ran

- **Program.** `cargo build --release --bin brut-breed` at commit `e315220` (pull request #48,
  the build of step 2), with rustc 1.97.0 (2d8144b78 2026-07-07) on x86_64 Linux (Ubuntu
  24.04.5 LTS). The binary has the SHA-256
  `898b7b4082c93b115c4992ac76f2145e7d57b629a36e34912a6903079961c220`. Every Run used it.
- **Settings.** Those of the counted Runs of step 1 (`--domain circuits --search 33 --select 22
  --children 8 --generations 1000 --max-calls 13000000`) and the placeholders of spec 12.2:
  `kappa_seed` 200, `kappa_drift` 20, `phase_drift` 512 and `new_soul_theta` 1.
- **Threads.** Two Runs at a time on 4 cores, each with `RAYON_NUM_THREADS=2` and `nice -n 5`,
  from 2026-11-02T13:30Z to 19:03Z. Each Run kept about one core busy. The thread count changes
  only the wall time (spec 15). The test suite and the analysis ran beside some Runs at a lower
  priority.
- **Check.** Right after each Run, `brut-breed --out <dir> --verify`.
- **Storage.** A finished Run holds 70 to 115 MB, nearly all of it in `events.jsonl` and
  `snapshot.json`; the 60 Runs would not have fit on the disk of this container. After `--verify`
  the driver recorded the SHA-256 of the five record files (`config.json`, `events.jsonl`,
  `snapshot.json`, `report.json`, `rediscovery.json`) and compressed the two large files with
  `xz -6`, to 3.5 to 8.5 MB per Run. The analysis read the compressed files through named pipes
  (`driver/view.sh`), since it reads each file once from start to end; `reference_rule.py` reads
  `events.jsonl.xz` itself.

```sh
B=target/release/brut-breed
for s in $(seq 901 912); do
  common="--domain circuits --search 33 --select 22 --children 8 --generations 1000 \
    --max-calls 13000000 --seed $s"
  $B --out runs/S-$s $common --soul s
  $B --out runs/V-$s $common --soul v
  $B --out runs/K-$s $common --soul k
  $B --out runs/E-$s $common --config driver/config-E.json
  $B --out runs/N-$s $common
done
```

`config-E.json` is the `config.json` of arm S with `"strengths": "equal"` in its `soul` object.
The flags override its seed and its numbers, so each arm-E Run's `config.json` differs from arm
S's only in that field.

## Results per seed

T is the pass time of step 1 (PREREG_STEP1_CIRCUITS.md, section 7): the first generation in which
the bred Cascades solve more than half of the targets the seed Cascades miss; "–" means no pass by
generation 1000. Y_V and Y_K compare T_S with T_V and T_K. The last five columns give the
divergence generation of two arms: the first generation whose newborns differ, member by member
in birth order, in gene id, operator, parents (by gene id), answers, solved count or graded value
(PREREG section 12).

| Seed | T_S | T_V | T_K | T_E | T_N | Y_V | Y_K | S-V | S-K | V-K | S-E | S-N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 901 | 131 | 131 | 131 | 131 | 131 | tie | tie | 197 | 197 | 242 | 197 | 197 |
| 902 | 799 | 140 | 998 | 140 | 140 | V first | S first | 101 | 564 | 101 | 101 | 101 |
| 903 | 23 | 111 | 23 | 23 | 23 | S first | tie | 4 | 83 | 4 | 79 | 55 |
| 904 | 393 | 274 | 398 | 393 | 270 | V first | S first | 242 | 303 | 242 | 536 | 163 |
| 905 | 42 | 28 | 30 | 42 | 28 | V first | K first | 18 | 27 | 18 | 18 | 18 |
| 906 | 232 | 232 | 232 | 107 | 232 | tie | tie | 246 | 246 | 246 | 40 | 246 |
| 907 | 196 | – | 196 | 196 | 196 | S first | tie | 187 | 268 | 187 | 238 | 229 |
| 908 | 46 | 46 | 46 | 46 | 46 | tie | tie | 83 | 71 | 71 | 79 | 75 |
| 909 | 289 | 92 | 183 | 170 | 285 | V first | K first | 53 | 92 | 53 | 92 | 263 |
| 910 | 933 | – | 405 | 592 | 528 | S first | K first | 73 | 73 | 73 | 73 | 73 |
| 911 | 7 | 7 | 7 | 7 | 7 | tie | tie | 12 | 12 | 12 | 12 | 12 |
| 912 | 320 | – | 390 | 320 | 320 | S first | S first | 39 | 88 | 39 | 21 | 39 |

The seed Cascades solved exactly swap and cs in every Run, so gate G1 of the draft holds
throughout. The evidence values after the twelve triples would be E_V = 81/256 and E_K = 27/64;
they are an illustration only, since pilot seeds never count.

## Results per arm

"Complex votes" counts the Stage votes of the newborns on the search panel that were tallied by
amplitudes; "flipped" those whose winner differs from the count's winner, and "tied" those in
which two or more answers shared the largest Π, so that the tie rule decided. `sim_decided`
counts the replies to newborns in which the phases decided a simulated model's choice. The
coherence is |Σ_j v_j| / Σ_j |v_j| of a vote's phasors, averaged over the votes. Minutes are the
wall time of the `brut-breed` process, two Runs at a time.

| Arm | Passes | Complex votes | Flipped | Tied | Coherence | sim_decided | Minutes (mean, min to max) | Calls (most) |
|---|---|---|---|---|---|---|---|---|
| S | 12 of 12 | 45 795 | 2 371 (5.2 %) | 2 614 (5.7 %) | 0.946 | 2 912 | 11.08 (9.46 to 12.86) | 1 243 954 |
| V | 9 of 12 | 40 297 | 412 (1.0 %) | 15 169 (37.6 %) | 0.736 | 6 049 | 10.81 (9.34 to 12.29) | 1 250 775 |
| K | 12 of 12 | 51 569 | 2 809 (5.4 %) | 3 137 (6.1 %) | 0.886 | 3 423 | 11.28 (9.67 to 14.68) | 1 264 833 |
| E | 12 of 12 | 54 293 | 5 242 (9.7 %) | 0 | 0.910 | 3 824 | 10.99 (9.46 to 14.72) | 1 257 405 |
| N | 12 of 12 | 0 | – | – | – | 0 | 8.38 (7.15 to 10.13) | 1 261 620 |

The `soul` group at generation 1000, the mean over the arm's 12 Runs (`analysis.txt` gives every
100th generation, `pilot.json` every 10th):

| Arm | `contrast` (formula for independent Souls) | `order_mean` | κ, median | `moved_mean` | origins |
|---|---|---|---|---|---|
| S | 0.563 (0.655) | 0.743 | 0.198 | 12.1° | 12.6 (7 to 17) |
| V | 1.036 (0.927) | 0.673 | 0.198 | 12.5° | 12.2 (6 to 17) |
| K | 0.559 (0.655) | 0.647 | 0 | 0° | 12.2 (7 to 16) |
| E | 0.655 (0.935) | 0.746 | 0.198 | 12.2° | 12.8 (7 to 17) |

## What the pilot shows

- **Every Run is sound.** All 60 Runs passed `--verify` ("report recomputed and equal;
  rediscovery recomputed and equal"). Seed 901 ran again in all five arms, one process at a time;
  the five record files of each rerun are byte-equal to the pilot's (`rerun.log`). Every Run of
  arm N has a `config.json` without a `soul` field, and its log has no `souls` event and no
  `soul` group; every Run with Souls has a `souls` event and a `soul` group in each of its 1001
  generations. No Run ended at the call cap: the most calls were 1 264 833 (K-908), below the
  bound of 12 723 412. No Run raised an integrity alarm or found an invention candidate.
- **The pilot is not degenerate** in the sense of ticket 11. Complex votes flipped winners in 45
  of the 48 Runs with Souls (in all but V-902, V-904 and V-909), and the phases decided simulated
  replies in all 48. On every seed the arms S, V and K parted: S and V between generation 4 and
  246, S and K between 12 and 564.
- **Why seed 901 made the same 484 newborns in 60 generations.** Arms that hold the same genes
  share every random number (spec 15), so their newborns can differ only after the Soul has
  changed an outcome: a complex vote that flips a winner, or a reply in which the phases decide a
  simulated model's choice. On seed 901 the first flipped vote came in generation 197 in arms S,
  V, K and E, and the first phase-decided reply in generation 242 to 269. Up to generation 60 each
  arm had 32 complex votes, none flipped, and no phase-decided reply (arm V's tie rule decided 12
  of the votes, each time for the count's winner), so the newborns had to be equal. Over 1000
  generations the arms part, on seed 901 at generation 197 (S-V, S-K) and 242 (V-K).
  `diagnose.py` checks the mechanism on every seed: in all 60 first differences (12 seeds, 5 pairs
  of arms), the first newborn that differs holds a flipped vote or a phase-decided reply in one of
  the two Runs. A flip needs a Stage of two or more Workers whose verdicts split, since a vote in
  which every judge gives the same verdicts is never lost (spec 7.5). The designer's judges split
  only where their checks differ (all columns with self-check, the first and the last without, or
  a judge out of steps) or where the stated costs tie and their seeded coins differ
  (`src/cascade/designer.rs`, `judge`). A phase-decided reply needs a self-checking Worker that
  found equally cheap circuits with two or more families (spec 11.2). These chances come from the
  genes and the tasks. The phases, and so the placeholders, decide only which of the chances turn
  into a flip or another reply: on seed 901 the arms S, V, K and E, whose phasors differ, flipped
  their first vote in the same generation, 197. So no placeholder can make the Soul act before its
  first chance. Whether other values would make it act more often after that, and with which
  effect on the pass time, the pilot does not show. We changed none of them.
- **The Soul often acts after the pass time.** Every tie of either claim is a tie by
  construction: arm S and the control passed in the same generation, before their two Runs first
  differed (4 ties of claim V, 6 of claim K). In 4 of the 12 triples (seeds 901, 906, 908 and 911)
  all three Runs passed before any two of them differed. Where the arms parted before the pass,
  the pass times scattered widely: on seed 902 arm S passed in generation 799, arm V in 140 and
  arm K in 998. Arm V can part early through the designer: its signs let the phases of two
  families cancel, so a circuit that two families found gets the intensity 0 against 1 for a
  circuit of one family. Arm V had 6 049 phase-decided replies against 2 912 in arm S, and on seed
  903 it parted from S in generation 4.
- **The outcomes.** Against arm V, arm S passed first 4 times, arm V 4 times, and 4 triples tied.
  Against arm K, arm S passed first 3 times, arm K 3 times, and 6 tied. So both estimates of
  P(S first | untied) are 1/2. Arm V did not pass by generation 1000 on 3 seeds (907, 910 and 912);
  the other arms passed on all 12.
- **Arms E and N, for description.** Against arm E, arm S passed first 0 times, arm E 4 times,
  and 8 tied; against arm N, 0, 5 and 7. In the pilot arm S never passed before arm N or arm E.
  This describes 12 seeds; it is no test, and these arms are not part of the comparison [Q74 a].
- **Votes.** Flipped votes were rare: 5.2 % of the complex votes in arm S, 1.0 % in V, 5.4 % in K
  and 9.7 % in E. In arm V the tie rule decided 37.6 % of the votes, as exact cancellation of
  signs makes it decide (spec 7.4); with complex phases it decided 5.7 % (arm S). The judges'
  phasors of a vote point nearly alike: the mean coherence is 0.946 in arm S, 0.886 in K, 0.910 in
  E and 0.736 in V. In arm K neither Observations nor imprinting move a Soul, since its κ is 0, so
  its coherence of 0.886 comes from inheritance alone: a birth copies or mixes Souls of the
  parents and moves each phase by at most 512 units, 2.8 degrees.
- **The Souls.** In arms S and K the contrast of the Character lies below the formula for
  independent Souls (0.563 and 0.559 against 0.655 at generation 1000), as shared ancestry makes
  Souls alike. In arm V it lies above its formula (1.036 against 0.927), as on seed 901 after 60
  generations (docs/STEP2_SOUL.md, section 4). `order_mean` stays at 0.65 to 0.75 in every arm
  (1 would mean that all Souls are alike), so the Souls of a population do not align. The current
  Souls moved about 12 degrees from the birth Souls in arms S, V and E, and not at all in arm K.
- **Cost.** A Run took 9.3 to 14.7 minutes with Souls and 7.2 to 10.1 minutes in arm N, two Runs
  at a time. Alone, the reruns of seed 901 took 10.71 to 10.97 minutes with Souls and 7.95 in
  arm N, 1 to 11 % less than in the pilot: each Run keeps about one core busy.

## Degenerate? No second batch

Ticket 11 asks for a second batch, on seeds 913 to 920 with changed placeholders, if the
placeholders give a degenerate pilot: for example no complex vote ever flips a winner, or every
Stage's Souls align within a few generations. Neither holds. Votes flip in 45 of the 48 Runs with
Souls, the phases decide simulated replies in all 48, and S, V and K part on every seed. The
population's Souls stay partly spread (`order_mean` 0.65 to 0.75). The Workers of one Stage do
point nearly alike (coherence 0.886 to 0.946 with complex phases), but this comes mostly from
inheritance, as arm K shows, and it holds from the first votes on (0.955 in arm S over generations
1 to 100); votes still flip. The diagnosis above shows that the placeholders cannot move the Soul's
first chance to act. So the pilot changed no placeholder and ran no second batch; seeds 913 to 920
stay unused. If the owner wants to learn whether larger phase steps give the Soul more effect,
those seeds are free for such a batch; this pilot does not answer that question.

## Proposals for ticket 10

Each number is a proposal for the owner. Ticket 10 is frozen only after the owner's approval.

| Number | Proposal | Reason |
|---|---|---|
| G | 1000 generations | The horizon of step 1, where arm A passed in 24 of 27 counted seeds; in the pilot all 12 Runs of arm N (arm A of step 1) passed by it. Every tie of the pilot came before the Soul first changed an outcome, which a longer horizon cannot untie. Keeping 1000 also lets the pilot triples serve as acceptance triples. |
| M | 13 000 000 calls | Above the bound of 12 723 412 calls for 1000 generations, to which the Soul adds no call; the pilot used at most 1 264 833. The cap of step 1. |
| p₁ | 3/4 | Both estimates of the pilot are 1/2, so the data support no p₁ above 1/2. 3/4 is the alternative of step 1. Among the p₁ that `oc.py` compares (3/5 to 4/5) at N_max = 120, it gives an averaged power of 0.149 and 0.153, within 0.01 of the best (7/10: 0.158 and 0.155), and it needs fewer triples on average (53.01 against 76.54). |
| N_max | 120 triples | The most the outline allows; every smaller cap has less power (at 60: averaged power 0.086 and 0.079). |
| `kappa_seed` | 200 (κ = 0.2) | Unchanged: the pilot is not degenerate, and no placeholder can make the Soul act before its first chance (section "What the pilot shows"). With 0.2 the current Souls moved about 12 degrees on average, and the population stays partly spread. |
| `kappa_drift` | 20 (0.02) | Unchanged, for the same reasons; the median κ stays at 0.198 by generation 1000. |
| `phase_drift` | 512 (2.8 degrees) | Unchanged, for the same reasons. |
| `new_soul_theta` | 1 | Unchanged, for the same reasons; it gave 7 to 17 Soul origins per Run of arm S by generation 1000. |

Since no number of a preset changes, the acceptance check of ticket 10 (PREREG section 7) uses the
pilot triples 901 to 903, run again at the frozen commit: their five record files must be
byte-equal to the pilot's, whose SHA-256 `pilot.json` records. Seeds 921 to 923 stay reserved for
the case that the owner changes a preset. Seeds 901 to 923 are pilot seeds and never count.

## Operating characteristics of the proposed rule

`oc.py` computes them; `own_check.py` recounts them with other methods, written apart from
`oc.py` (the untied walk times a binomial tail for one claim, a Beta factorisation for the
average over the uncertainty, and a chain over the triples that are not a tie in both claims,
mixed with binomials, for the length of the experiment). Both print the same figures to the last
digit (`oc.txt`, `own_check.txt`). The pilot enters as the probabilities of "S first", "control
first" and a tie per triple, 4/12, 4/12 and 4/12 for claim V and 3/12, 3/12 and 6/12 for claim K,
and as the joint distribution of (Y_V, Y_K) over the 12 seeds.

- **False proof.** The exact worst case of a false proof of one claim at the cap of 120 triples,
  at P(S first | untied) = 1/2 with no ties, is
  2617038152666572495921938445760671/664613997892457936451903530140172288 = 0.003938, below 1/200.
  Ties and a share of "S first" below 1/2 only lower it. Some claim is made falsely with
  probability at most 0.007875, below 1/100.
- **Wrong failure.** At P(S first | untied) = p₁ = 3/4 with no ties, a claim fails with
  probability 0.036848, below 1/20.
- **Power at the pilot's estimate.** Claim V 0.0038, claim K 0.0034: at an estimate of 1/2 a
  claim is proven only as often as a false proof would be.
- **Power over the uncertainty of the estimate.** Averaged over a Dirichlet distribution of the
  three probabilities per triple (the posterior of a uniform prior after the pilot's counts):
  claim V 0.1485, claim K 0.1528.
- **What power needs.** At the pilot's mean tie rate of 5/12, one claim is proven with
  probability 0.30 at P(S first | untied) = 0.65, 0.61 at 0.70 and 0.88 at 0.75; so its power
  reaches 0.5 only from about 0.68 on. The pilot found 1/2.
- **Length.** Under the pilot's joint distribution both claims would most likely fail: no claim is
  proven with probability 0.9929. The experiment needs 53.01 triples on average, and 90 % of such
  experiments need at most 102.
- **Hours.** For Runs one at a time at 2 threads, a triple of arms S, V and K took 32.32 minutes on
  average: the pilot's minutes, two Runs at a time, scaled by 0.9744, the ratio of the reruns'
  minutes alone to the same Runs in the pilot. The experiment would then take 28.55 hours on
  average and at most 75.32 hours (120 triples at the slowest triple, 37.66 minutes), plus about
  2.7 hours for the acceptance check and the thread check of gate G4 (5 triples).

| p₁ | N_max | Worst-case false proof | Power V, K at the estimate | Averaged power V, K | Mean triples | 90 % need at most |
|---|---|---|---|---|---|---|
| 3/5 | 120 | 0.000564 | 0.0001, 0.0000 | 0.1026, 0.0842 | 118.54 | 120 |
| 2/3 | 120 | 0.003314 | 0.0021, 0.0013 | 0.1547, 0.1452 | 94.20 | 120 |
| 7/10 | 120 | 0.003874 | 0.0031, 0.0024 | 0.1583, 0.1545 | 76.54 | 120 |
| **3/4** | **120** | **0.003938** | **0.0038, 0.0034** | **0.1485, 0.1528** | **53.01** | **102** |
| 4/5 | 120 | 0.003687 | 0.0037, 0.0036 | 0.1259, 0.1376 | 35.45 | 67 |
| 3/4 | 60 | 0.003371 | 0.0024, 0.0016 | 0.0861, 0.0793 | 43.45 | 60 |

**The power is far below 0.5.** With N_max = 120, the power of claim V at the pilot's estimate is
0.0038 and that of claim K 0.0034; averaged over the uncertainty of the estimates, 0.1485 and
0.1528. This holds for every p₁ and cap that `oc.py` compares. As ticket 11 asks, we say so
plainly and propose either to drop each claim from the pre-registration or not to run the counted
experiment; since both claims lack power, both proposals come to the same: **we propose not to
run the counted experiment of ticket 10 as designed**, and to report step 2 descriptively from the
pilot. A confirmatory test of the Soul would need a design in which the Soul acts before the pass
time decides, for example another endpoint or Domain; that needs a pilot of its own. The owner
decides before ticket 10 is frozen.

## Reproduce

Build commit `e315220` with `cargo build --release --bin brut-breed` and run the commands of
section "How the pilot ran", one Run at a time or two at a time. Then, with the analysis and the
scripts of this pull request:

```sh
for d in runs/*-9*; do target/release/brut-breed --out $d --verify; done
cargo run --release --example soul_comparison -- \
  --s runs/S-9{01..12} --v runs/V-9{01..12} --k runs/K-9{01..12} \
  --e runs/E-9{01..12} --n runs/N-9{01..12} > analysis.txt
python3 -I .scratch/seele-und-kappa/rule/reference_rule.py --pilot 901-912 runs > recount.txt
python3 -I -B .scratch/seele-und-kappa/rule/pilot.py --batch batch1=runs \
  --hashes hashes.txt --analysis batch1=analysis.txt --rerun rerun=rerun.log --out pilot.json
python3 -I -B .scratch/seele-und-kappa/rule/oc.py docs/pilot-step2/pilot.json
python3 -I -B .scratch/seele-und-kappa/rule/own_check.py docs/pilot-step2/pilot.json
python3 -I -B .scratch/seele-und-kappa/rule/diagnose.py 901-912 runs
```

`pilot.py` needs, beside each Run directory, the driver's files `<run>.wall` (the wall seconds)
and `<run>.verify` (the output of `--verify`), and the driver's `hashes.txt`; it checks the rule
and divergence tables of `analysis.txt` against its own recount and stops at any difference. On
rustc 1.97.0 on x86_64 Linux each Run's files must have the SHA-256 that `pilot.json` records; on
another platform a floating-point function may round its last bit differently (spec 15), and
then the hashes can differ.
