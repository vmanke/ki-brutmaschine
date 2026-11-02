# Pre-registration of step 2: does a Soul of complex phases make the breeder pass sooner? (draft)

**Status: draft for the owner; nothing is frozen.** This document writes out the outline of
ticket 10 of `.scratch/seele-und-kappa/`. Every number marked "proposal" comes from the pilot
of ticket 11 ([docs/pilot-step2/](pilot-step2/README.md)) and waits for the owner's decision.
No counted seed has run. The document is frozen only after the owner's approval, by the squash
merge of its pull request, as pull request #19 froze step 1. The frozen commit is the one commit
that this merge creates on the main branch. A commit cannot hold its own hash, so we record the
hash with the results. From that commit on nothing below changes, and nothing changes after seed
1001 has started: a change then makes a new experiment on seeds not used before. Every deviation
is reported with the results, with its reason.

**The pilot's finding comes first.** The pilot of ticket 11 finds that this comparison has no
useful power. The Soul changes no outcome before its first flipped vote or its first
phase-decided reply in the simulated designer, and every tie of the pilot came about because
arm S and the control passed before their two Runs first differed. Among the untied pilot triples
arm S passed first 4 of 8 times against arm V and 3 of 6 times against arm K, so both estimates of
P(S first | untied) are 1/2. At these estimates the rule of section 8 proves claim V with
probability 0.0038 and claim K with probability 0.0034 within 120 triples; averaged over the
uncertainty of the estimates, 0.1485 and 0.1528 (section 9). Each is far below 0.5. So we
propose, as ticket 11 asks, not to run the counted experiment as designed; dropping both claims
comes to the same. The owner decides before this document is frozen. The numbers below are the
design that the pilot proposes in case the owner runs it anyway.

The design is section 17 of the [spec](../.scratch/seele-und-kappa/spec.md) and
[ADR 0009](adr/0009-workers-carry-an-inherited-soul-whose-votes-interfere.md). The report on step
2 is [STEP2_SOUL.md](STEP2_SOUL.md) (German: [SCHRITT2_SEELE.md](SCHRITT2_SEELE.md)). The terms
are those of `CONTEXT.md`. The step 1 experiment that this one builds on is
[PREREG_STEP1_CIRCUITS.md](PREREG_STEP1_CIRCUITS.md).

## 1. Question

Does a Soul of complex phases with an inherited Coupling strength make the breeder of step 1
reach the pass criterion of step 1 sooner than (claim V) the same Soul read as real numbers, and
(claim K) the same complex Soul with κ fixed at 0? We ask it on circuits, with the simulated
designer and the settings of arm A of step 1 (Darwin Gödel selection). This tests the soul
mechanism as a positive control (Q42, Q44). It makes no claim about real language models, about
inventions, or about souls in nature: it is rung 1 of the ladder (Q48 stays open, Q57). The
letter V (German Vorzeichen, the sign readout) avoids arm R of step 1, uniform parent choice.

## 2. What is frozen

| Item | Frozen as |
|---|---|
| Code | The whole tree of the frozen commit. A later edit of this file or of the code is a deviation. |
| Toolchain | rustc 1.97.0 on x86_64 Linux, as in the pilot. On another toolchain or platform a floating-point function may round its last bit differently (spec 15); such Runs are a deviation. |
| Program | `cargo build --release --bin brut-breed` at that commit. |
| Analysis | `examples/soul_comparison.rs` at that commit. |
| Recount | `.scratch/seele-und-kappa/rule/reference_rule.py` at that commit: an independent implementation of sections 5 and 8 and of the divergence generation of section 12, Python standard library only, run with `python3 -I`. |
| Target list L | The 11 verified library targets of step 1, in library order (PREREG_STEP1_CIRCUITS.md, section 3). |
| Designer and genes | The constants of `src/cascade/designer.rs` and the genes of `src/cascade/circuit_genes.rs` (the 4 seed Cascades, the 50 phrases, the Mutator's brief and the admission rule), unchanged since step 1. |
| Family strengths | The table of spec 5.2, in `LIBRARY` of `src/cascade/circuit.rs`. |
| Presets | `SoulConfig::arm_s`, `arm_v` and `arm_k` of `src/cascade/soul.rs`, which `--soul s`, `--soul v` and `--soul k` set. Their numbers are the table below; the analysis compares every Run with them. |
| Seeds | The fixed list of section 6. |
| Definitions and rule | Sections 5, 8, 9 and 10. |

The `soul` settings of the three arms (spec 12.2). The four numbers marked "proposal" are the
placeholders that the pilot set (section 11); arm K's `kappa_drift` has no effect, since its κ
never moves.

| Field | Arm S | Arm V | Arm K |
|---|---|---|---|
| `readout` | complex | real | complex |
| `strengths` | domain | domain | domain |
| `vote`, `simulator`, `observe_vote`, `observe_peer` | on | on | on |
| `kappa_seed` (thousandths) | 200 (proposal) | 200 (proposal) | 0 |
| `kappa_fixed` | off | off | on |
| `kappa_drift` (thousandths) | 20 (proposal) | 20 (proposal) | 20 (proposal) |
| `copy`, `mix` | on | on | on |
| `phase_drift` (phase units) | 512 (proposal) | 512 (proposal) | 512 (proposal) |
| `new_soul_theta` | 1 (proposal) | 1 (proposal) | 1 (proposal) |
| `imprint` | on | on | on |
| `teach` | off | off | off |

## 3. Arms

For each counted seed s of section 6, one Run per arm, one process at a time, each into a fresh
directory:

```sh
brut-breed --out runs/S-s --domain circuits --search 33 --select 22 --children 8 \
  --generations G --max-calls M --seed s --soul s
brut-breed --out runs/V-s --domain circuits --search 33 --select 22 --children 8 \
  --generations G --max-calls M --seed s --soul v
brut-breed --out runs/K-s --domain circuits --search 33 --select 22 --children 8 \
  --generations G --max-calls M --seed s --soul k
```

G = 1000 generations and M = 13 000 000 calls (proposal; the values of step 1). Every other
setting stays at its default, as in section 6 of the step 1 pre-registration: crossover rate 0.5,
3 peer opponents with 2 tasks per pair, judge weight 0.2, token cost 0.01 per 1000 tokens, DGM
lambda 10 and alpha_0 0.5, no LLM rounds, 3 finalists, at most 4 Stages of at most 3 Workers,
the models `sim:small` and `sim:large`, and every verified target, dressed, with at most 64 gates
per answer. The analysis refuses a Run whose settings differ from its arm's in anything but the
seed, the generations and the call cap.

**Calls.** The Soul adds no call beyond the bound of step 1, 12 723 412 calls for 1000
generations: imprinting makes no call [Q73 a], and no Stage votes more often than that bound
counts. The Soul can change whether the Workers of a Stage agree, and so how many judge calls a
Stage makes, so the calls of the arms can differ. The pilot Runs used at most 1 264 833
calls. The analysis reports the calls per Run and withholds every claim when a Run ended at the
cap. Teaching is off in all three arms [Q73 d].

## 4. Gates

- **G1.** In every counted Run the seed Cascades solve exactly swap and cs, so the gate
  3 · |R_seed| ≤ |L| of step 1 holds. The analysis checks it.
- **G2.** At the frozen commit the test of ticket 02 on 2^20 = 1 048 576 fresh Souls per Domain
  passes: the contrast meets K_C and K_R within 0.01, and K_R / K_C lies within 0.01 of √2
  (Q57). At this size a correct implementation does not fail by the luck of the seed (spec 17).
- **G3.** At the frozen commit, Runs without Souls reproduce step 1: the head hash of pilot seed
  101 in `docs/pilot-step1/pilot.json` (the release test `souls_off_reproduces_the_step1_pilot`)
  and the five hashes of A-1 in `docs/step1-counted/hashes.txt`.
- **G4.** After the first triple, seed 1001 runs again in each arm at 1 and at 4 threads, and the
  five files of each Run (`config.json`, `events.jsonl`, `snapshot.json`, `report.json`,
  `rediscovery.json`) are byte-equal to the counted Run's.

## 5. Definitions

Exact, solved(t), R_seed, R_bred, D, the gate and the pass time T(s) are those of section 7 of
the step 1 pre-registration, read from each Run's `rediscovery.json`. T is the k-th smallest
`first_generation` among the targets with `reached_by_bred` true and `reached_by_seed` false,
with k = floor(|L \ R_seed| / 2) + 1; T is never when fewer such targets exist or when T > G.
Per seed s we have T_S, T_V and T_K. The outcome Y_V(s) is "S first" when T_S < T_V, "V first"
when T_V < T_S, and a tie otherwise, also when neither Run passes ("never" counts as later than
every generation). Y_K(s) is the same with T_K. Every comparison is between integers.

## 6. Seeds

The pilot seeds 901 to 923 of ticket 11 never count, because they chose the design. The counted
seeds are the fixed list 1001, 1002, ..., in this order, at most N_max of them, with N_max = 120
(proposal). Triple i runs the i-th seed of the list. None of them has been used before: step 1
used 1 to 130 and its pilot seeds 101 to 110, 201 to 210, 301 to 310, 401 and 402, and step 2
used 901 to 912 in its pilot. A triple counts once its three Runs are finished.

## 7. Procedure

1. Build the program at the frozen commit with the frozen toolchain. Run `cargo test --lib`,
   which holds the test of gate G2 (`the_character_of_fresh_souls_meets_the_formulas`),
   `cargo test --test circuits`, `cargo test --test soul`, the release test of gate G3
   (`cargo test --release --test soul -- --ignored souls_off_reproduces_the_step1_pilot`), and
   `cargo test --example soul_comparison`, which holds the tests of the analysis. All must pass.
   Then run A-1 of step 1 again and check its five hashes (gate G3).
2. Acceptance check, before seed 1001 and not counted, on three triples of the frozen program.
   If no number of a preset changed after the pilot, these are the pilot triples 901 to 903,
   run again at the frozen commit; their five files must be byte-equal to the pilot's, whose
   hashes `docs/pilot-step2/pilot.json` records. If a number of a preset changed, the frozen
   analysis would refuse the pilot's Runs; then three acceptance triples run at the frozen commit
   with the final presets, on the pilot seeds 921 to 923. On these triples the analysis gives
   their numbers and withholds every claim, and `reference_rule.py --pilot` gives the same pass
   times, outcomes, evidence values and divergence generations.
3. Run the counted triples in the order of the list. For seed s, run arms S, V and K of seed s,
   one process at a time, each into a fresh directory. A Run that is stopped, killed or otherwise
   interrupted is not resumed: we delete its directory and run it again from scratch, on the same
   seed, into a fresh one (a resumed Run counts the reserved calls of the interrupted generation
   as spent, so its files would differ from a straight Run's).
4. After each triple, run `brut-breed --out <dir> --verify` on its three Runs. Each must end with
   "report recomputed and equal; rediscovery recomputed and equal". Then run the frozen analysis
   on all finished triples, in the order of the list:

   ```sh
   cargo run --release --example soul_comparison -- \
     --s runs/S-1001 ... runs/S-s --v runs/V-1001 ... runs/V-s --k runs/K-1001 ... runs/K-s
   ```

   It updates the evidence values of section 8 in the order of the list and then names the next
   seed or gives the verdicts.
5. Thread check G4, once, right after triple 1 and before triple 2.
6. The experiment stops at the stop triple of section 8. Triples run after it are listed but not
   counted.
7. Before the result is reported, recount it independently:

   ```sh
   python3 -I .scratch/seele-und-kappa/rule/reference_rule.py --runs runs
   ```

   It reads S-1001, V-1001, K-1001, S-1002, ... in the order of the list, counts a triple once
   its three Runs are finished, checks every setting of each Run against the arm's table, and
   refuses a seed off the list and a gap. Its pass times, outcomes, evidence values, stop triple,
   verdicts and divergence generations must equal those of the analysis; we report the recount
   with the results, and any difference.

## 8. Rule

This is decision Q74 (a): two separate claims, "S before V" and "S before K".

- **Evidence values.** E_V and E_K start at 1. They are updated strictly in the order of the
  seed list: triple i enters only after triples 1 to i - 1, never in the order in which Runs
  finish. E_V is multiplied by a = 2p₁ when Y_V is "S first", by b = 2(1 - p₁) when it is
  "V first", and stays on a tie. E_K likewise with Y_K. The factors are the likelihood ratio of
  P(S first | untied) = p₁ against 1/2, as 3/2 and 1/2 were for p₁ = 3/4 in step 1. We propose
  p₁ = 3/4 (proposal; section 11), so a = 3/2 and b = 1/2, and after w triples "S first" and l
  triples "V first", E_V = 3^w / 2^(w + l).
- **Verdicts.** Claim V is proven at the first triple with E_V ≥ 200 and fails at the first
  triple with E_V ≤ 1/20; claim K the same with E_K. Each verdict is final at its first
  crossing; its E value goes on with the later triples and is reported at the stop. In counts,
  with p₁ = 3/4, a claim is proven at 14, 15, 17, 19, 20, 22, 24 or 26 triples "S first" with
  0 to 7 against it, and it fails at 5, 5, 6, 7, 7, 8, 8 or 9 triples against it with 0 to 7
  "S first".
- **Stop.** The experiment stops after the first triple at which both claims are decided, or
  after N_max triples. A claim still open then is "undecided, no claim" and is reported with its
  E value.
- **Integers.** Every decision is made in integers: E = a^w b^l is compared with 200 and 1/20 as
  products of the factors' numerators and denominators, exact natural numbers of any size, as in
  step 1.
- **Reported per triple.** R_seed, D_S, D_V and D_K at generation G, T_S, T_V, T_K, Y_V, Y_K,
  and E_V and E_K after the triple. Then the stop triple and both verdicts, with E_V and E_K at
  the stop. We never report the largest E along the way, which overstates the evidence (Shafer,
  Shen, Vereshchagin and Vovk 2011).

## 9. Error bounds

Under the null hypothesis of claim V, P(S first | untied, earlier triples) ≤ 1/2 for every
triple, so E_V is a test supermartingale, and by Ville's inequality claim V is proven falsely
with probability at most 1/200 (Shafer, Shen, Vereshchagin and Vovk 2011; Waudby-Smith and
Ramdas 2024, as cited in step 1). The same holds for claim K, so some claim is made falsely with
probability at most 1/100. A "failed" verdict is wrong with probability at most 1/20 when
P(S first | untied) ≥ p₁: under that alternative 1/E is a test supermartingale. The bounds
require a fixed seed list and order, no skipped or replaced seed, updates in list order, and no
change after the first counted seed. They assume that seeds act as independent draws, which we
assume because the random streams of a Run are derived from a hash of its seed.

**Operating characteristics** of the proposed design, p₁ = 3/4 and N_max = 120
(`.scratch/seele-und-kappa/rule/oc.py` and `oc.txt`, recounted with other methods by
`own_check.py` and `own_check.txt`; both print the same figures to the last digit). The pilot
enters as the probabilities of "S first", "control first" and a tie per triple: 4/12, 4/12 and
4/12 for claim V, 3/12, 3/12 and 6/12 for claim K, and their joint distribution over the 12 pilot
seeds.

- **False proof.** The exact worst case of a false proof of one claim at the cap, at
  P(S first | untied) = 1/2 with no ties, is
  2617038152666572495921938445760671/664613997892457936451903530140172288 = 0.003938, below
  1/200. Ties and a share below 1/2 only lower it. Some claim is made falsely with probability at
  most 0.007875, below 1/100.
- **Wrong failure.** At P(S first | untied) = p₁ = 3/4 with no ties, a claim fails with
  probability 0.036848, below 1/20.
- **Power at the pilot's estimate.** Claim V 0.0038, claim K 0.0034. Averaged over the
  uncertainty of the estimates (a Dirichlet distribution of the three probabilities per triple,
  the posterior of a uniform prior): claim V 0.1485, claim K 0.1528.
- **What power needs.** At the pilot's mean tie rate of 5/12, one claim is proven with probability
  0.30 at P(S first | untied) = 0.65, 0.61 at 0.70 and 0.88 at 0.75: its power reaches 0.5 only
  from about 0.68 on.
- **Length and hours.** Under the pilot's joint distribution of (Y_V, Y_K), no claim is proven
  with probability 0.9929. The experiment needs 53.01 triples on average, and 90 % of such
  experiments need at most 102. For Runs one at a time at 2 threads at the pilot's speed (32.32
  minutes per triple on average, 37.66 for the slowest), this is 28.55 hours on average and at
  most 75.32 hours for 120 triples, plus about 2.7 hours for the acceptance check and the thread
  check of gate G4.

## 10. Void conditions

The experiment is void, and makes no claim, when a counted Run raises an integrity alarm, a gate
G1 to G4 fails, a counted Run fails `--verify`, or a counted Run has other settings than section
3, other generations or another call cap, or ended at the cap.

## 11. Pilot

The pilot of ticket 11 ran arms S, V, K, E and N on the pilot seeds 901 to 912, with the
settings of section 3 and the placeholders of spec 12.2; its data and its full account are in
[docs/pilot-step2/](pilot-step2/README.md). It sets p₁, N_max, G, M and the four placeholders,
and it gives the planning figures of section 9. Its seeds never count, because they chose the
design.

- **Soundness.** All 60 Runs pass `--verify`; seed 901 run again gives byte-equal files in all
  five arms; the analysis and the recount agree on all 12 triples.
- **Not degenerate.** Complex votes flip winners in 45 of the 48 Runs with Souls, the phases
  decide simulated replies in all 48, and S, V and K part on every seed (S and V between
  generation 4 and 246, S and K between 12 and 564). In all 60 first differences of two arms (12
  seeds, 5 pairs) the first newborn that differs holds a flipped vote or a phase-decided reply in
  one of the two Runs. So the pilot changed no placeholder and ran no second batch; seeds 913 to
  920 stay unused, and 921 to 923 stay reserved for acceptance triples.
- **The outcomes.** Against arm V: arm S first 4 times, arm V 4 times, 4 ties. Against arm K: 3, 3
  and 6. Every tie is a tie by construction: both Runs passed before they first differed. In the
  pilot arm S never passed before arm N (0 first, 5 later, 7 ties) or arm E (0, 4, 8); this is
  description only.

| Number | Proposal | Reason |
|---|---|---|
| G | 1000 | The horizon of step 1; every tie of the pilot came before the Soul first changed an outcome, which a longer horizon cannot untie. |
| M | 13 000 000 | Above the bound of 12 723 412 calls; the pilot used at most 1 264 833. |
| p₁ | 3/4 | Both estimates are 1/2, so the data support no p₁ above 1/2. 3/4 is the alternative of step 1; its averaged power lies within 0.01 of the best p₁ that `oc.py` compares, and it needs fewer triples. |
| N_max | 120 | The most the outline allows; every smaller cap has less power. |
| `kappa_seed`, `kappa_drift`, `phase_drift`, θ | 200, 20, 512, 1 | Unchanged: the pilot is not degenerate, and no placeholder can make the Soul act before its first chance. |

## 12. Descriptive results

No verdict depends on these; the analysis prints them.

- Pass rates per arm, and D at generation G per Run.
- The first generation in which a bred Cascade solved each target, per Run.
- Per triple and pair of arms, the divergence generation: the first generation whose newborns
  differ, member by member in birth order, in gene id, operator, parents (by gene id), answers,
  solved count or graded value. The `born` events themselves differ earlier: they carry κ, which
  differs between S and K from generation 0, and the vote tallies, whose coherence differs
  between S and V from the first vote. So the divergence generation is defined by outcomes, not
  by `born` events (spec 15).
- The trajectories, per arm, of `contrast` against `contrast_independent`, `order_mean`, κ, the
  shares of flipped and of tied votes, `coherence_mean`, `sim_decided` and `moved_mean`.
- The calls and the wall time of every Run; the Runs that ended at the call cap; invention
  candidates.
- The arms E and N of the pilot, in `docs/pilot-step2/`.

## 13. The claim

If claim V is proven: in this model, a Soul of complex phases lets the breeder reach the pass
criterion sooner than the same Soul read as real numbers; among untied seeds, arm S passes first
more often than arm V. Claim V compares the two readouts under the tie rule of spec 7.4, by which
neither readout loses a unanimous vote; so it is not a claim about exact cancellation alone. If
claim K is proven: the same against the Soul without coupling. Both are claims about a positive
control of the breeder, not about real models, not about inventions, and not about souls in
nature.

If a claim fails, we report that it fails: the data then speak against P(S first | untied) ≥ p₁,
which is no proof of a share of 1/2 or less. A claim that is not decided by triple N_max is
reported as undecided, with its E value. A later change of the presets, the designer, the genes,
the settings or the analysis needs a new pre-registration and seeds not used before.

## Sources

The statistics follow the sources that section 8 of the step 1 pre-registration gives, verified
there against their texts:

- Shafer, Shen, Vereshchagin and Vovk (2011). Test martingales, Bayes factors and p-values.
  Statistical Science 26(1), 84–101. https://doi.org/10.1214/10-STS347
- Waudby-Smith and Ramdas (2024). Estimating means of bounded random variables by betting.
  Journal of the Royal Statistical Society, Series B 86(1), 1–27.
  https://doi.org/10.1093/jrsssb/qkad009
