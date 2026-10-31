# 10 Pre-registration of the step 2 comparison (outline)

Status: needs-info (Q74; the numbers come from the pilot, ticket 11)
Blocked by: 09
Spec: `../spec.md`, section 17

This ticket writes `docs/PREREG_STEP2_SOUL.md` on the model of `docs/PREREG_STEP1_CIRCUITS.md`.
It is frozen only after the pilot (ticket 11) and the owner's approval, by the squash merge of its
pull request, as pull request #19 froze step 1. Until then nothing in it is fixed, and no counted
seed runs. The outline below is what the document must say; a number marked "pilot" is set from
ticket 11.

## The outline

**1. Question.** Does a Soul of complex phases with an inherited Coupling strength make the breeder
of step 1 reach the pass criterion of step 1 sooner than (claim V) the same Soul read as real
numbers, and (claim K) the same complex Soul with κ fixed at 0? On circuits, with the simulated
designer and the settings of arm A of step 1 (Darwin Gödel selection). This tests the soul
mechanism as a positive control (Q42, Q44): no claim about real language models, about inventions,
or about souls in nature (rung 1 of the ladder; Q48 stays open, Q57). The letter V (German
Vorzeichen, the sign readout) avoids arm R of step 1, uniform parent choice.

**2. What is frozen.** The tree of the frozen commit; rustc 1.97.0 on x86_64 Linux;
`cargo build --release --bin brut-breed`; the analysis `examples/soul_comparison.rs`; the recount
`.scratch/seele-und-kappa/rule/reference_rule.py`; the 11 targets of step 1; the designer's
constants and the genes of step 1, unchanged; the Family strengths of spec 5.2; the three `soul`
presets with the pilot's numbers, which the presets of `--soul` in the code equal; the seed list;
the definitions and the rule below.

**3. Arms.** For each counted seed s, one Run per arm, one process at a time:

```sh
brut-breed --out runs/S-s --domain circuits --search 33 --select 22 --children 8 \
  --generations G --max-calls M --seed s --soul s
brut-breed --out runs/V-s ... --soul v
brut-breed --out runs/K-s ... --soul k
```

Every other setting stays at its default, as in section 6 of the step 1 pre-registration. G and M:
pilot (step 1 used 1000 and 13 000 000). The Soul adds no call beyond the bound of step 1,
12 723 412 calls for 1000 generations: imprinting makes no call, and no Stage votes more often than
that bound counts. But the Soul can change whether the Workers of a Stage agree, and so how many
judge calls a Stage makes, so the calls of the arms can differ; the analysis reports calls per arm
and checks that no Run ended at the cap. Teaching is off in all three arms [Q73 d].

**4. Gates.** (G1) In every counted Run the seed Cascades solve exactly swap and cs, so the gate
3 · |R_seed| ≤ |L| of step 1 holds. (G2) At the frozen commit the test of ticket 02 on
2^20 = 1 048 576 fresh Souls per Domain passes: the contrast meets K_C and K_R within 0.01, and
K_R / K_C lies within 0.01 of √2 (Q57); at this size a correct implementation does not fail by the
luck of the seed. (G3) At the frozen commit, Runs without Souls reproduce step 1: the head hash of
pilot seed 101 in `docs/pilot-step1/pilot.json` and the five hashes of A-1 in
`docs/step1-counted/hashes.txt`. (G4) After the first triple, seed 1001 runs again in each arm at 1
and at 4 threads, and the five files of each Run are byte-equal to the counted Run's.

**5. Definitions.** Exact, solved(t), R_seed, R_bred, D, the gate and the pass time T(s) are those of
section 7 of the step 1 pre-registration, read from each Run's `rediscovery.json`. Per seed s:
T_S, T_V and T_K. The outcome Y_V(s) is "S first" when T_S < T_V, "V first" when T_V < T_S, and a
tie otherwise, also when neither passes ("never" counts as later than every generation). Y_K(s) is
the same with T_K.

**6. Seeds.** The pilot seeds of ticket 11 (901 to 923) never count. The counted seeds are the fixed
list 1001, 1002, ..., in this order, at most N_max of them (pilot; at most 120). None of them has
been used before: step 1 used 1 to 130 and its pilot seeds 101 to 110, 201 to 210, 301 to 310, 401
and 402. A triple counts once its three Runs are finished.

**7. Procedure.** As section 6 of step 1: build and test at the frozen commit; an acceptance check,
before seed 1001 and not counted, on three triples of the frozen program. If the pilot kept every
number of the presets, these are the pilot triples 901 to 903, run again at the frozen commit, and
their files must be byte-equal to the pilot's. If the pilot changed any number of a preset, the
frozen analysis would refuse the pilot's Runs, whose settings differ from the presets; then three
acceptance triples run at the frozen commit with the final presets, on the pilot seeds 921 to 923.
On these triples the analysis withholds every claim and gives their numbers, and `reference_rule.py`
agrees. Then the triples in list order, each Run into a fresh directory; an interrupted Run is
deleted and run again from scratch on the same seed; `--verify` on all three Runs after each triple;
the analysis on all finished triples names the next seed or gives the verdicts; the thread check G4
after triple 1; the recount before any report.

**8. Rule** [Q74 a]. Two evidence values E_V and E_K start at 1 and are updated strictly in list
order. E_V is multiplied by a = 2p₁ when Y_V is "S first", by b = 2(1 − p₁) when it is "V first",
and stays on a tie; E_K likewise with Y_K. The factors are the likelihood ratio of P(S first |
untied) = p₁ against 1/2, as 3/2 and 1/2 were for p₁ = 3/4 in step 1; p₁ > 1/2: pilot. Claim V is
proven at the first triple with E_V ≥ 200 and fails at the first with E_V ≤ 1/20; claim K the same
with E_K. Each verdict is final at its first crossing. The experiment stops after the first triple at
which both claims are decided, or after N_max triples; a claim still open is "undecided, no claim"
and is reported with its E value. Decisions are made in integers, as in step 1. With Q74 (d), arm E
runs on every seed too, so a seed has four Runs; a third value E_E for claim E (S against E) is
updated the same way; each claim is proven when its evidence value reaches 300; and the experiment
stops when all three claims are decided.

**9. Error bounds.** Under the null hypothesis of claim V, P(S first | untied, earlier triples) ≤
1/2 for every triple, so E_V is a test supermartingale, and by Ville's inequality claim V is proven
falsely with probability at most 1/200 (Shafer, Shen, Vereshchagin and Vovk 2011; Waudby-Smith and
Ramdas 2024, as cited in step 1). The same holds for claim K, so some claim is made falsely with
probability at most 1/100 (with Q74 (d): 1/300 per claim, 1/100 for all three). A "failed" verdict
is wrong with probability at most 1/20 when P(S first | untied) ≥ p₁. The bounds require a fixed
seed list and order, no skipped or replaced seed, updates in list order, and no change after the
first counted seed; they assume that seeds act as independent draws. The exact worst-case error at
the cap, the power at the pilot's estimate and the expected number of triples are computed before
the freeze, as issue 09 of step 1 computed them, and recounted independently.

**10. Void conditions.** The experiment is void, and makes no claim, when a counted Run raises an
integrity alarm, a gate G1 to G4 fails, a counted Run fails `--verify`, or a counted Run has other
settings than section 3, other generations or another call cap, or ended at the cap.

**11. Pilot.** The pilot of ticket 11 sets p₁, N_max, G, M and the four placeholders of spec 12.2,
and gives the planning figures; its seeds never count, because they chose the design.

**12. Descriptive results** (no verdict). Pass rates per arm; D at generation G; the first
generation per target; per triple and pair of arms the divergence generation, the first generation
whose newborns differ, member by member in birth order, in gene id, operator, parents (by gene id),
answers, solved count or graded value (the `born` events differ earlier, since they carry κ and the
vote tallies); the trajectories of `contrast` against `contrast_independent`, `order_mean`, κ, the
shares of flipped and of tied votes, `coherence_mean`, `sim_decided` and `moved_mean`, per arm;
calls and wall time per Run; the arms E and N of the pilot.

**13. The claim.** If claim V is proven: in this model, a Soul of complex phases lets the breeder reach
the pass criterion sooner than the same Soul read as real numbers; among untied seeds, arm S passes
first more often than arm V. Claim V compares the two readouts under the tie rule of spec 7.4, by
which neither readout loses a unanimous vote; so it is not a claim about exact cancellation alone. If
claim K is proven: the same against the Soul without coupling. Both are claims about a positive
control of the breeder, not about real models, not about inventions, and not about souls in nature.

## What to do

- Write `docs/PREREG_STEP2_SOUL.md` with the 13 sections above.
- Write `examples/soul_comparison.rs` (sections 5 to 8 and the divergence generation of section 12)
  and its tests, and `.scratch/seele-und-kappa/rule/reference_rule.py`, Python standard library
  only, run with `python3 -I`.
- After ticket 11: fill in the pilot's numbers and their operating characteristics, set the presets
  of `--soul` in the code to the approved numbers, and ask the owner for approval.

## Done when

- `docs/PREREG_STEP2_SOUL.md` has the 13 sections; every number is fixed or marked "pilot".
- `examples/soul_comparison.rs` has tests that: recompute the pass times from `rediscovery.json`,
  equal to those of `examples/circuit_positive_control.rs` on the arm-N Runs of the pilot (that
  analysis refuses any Run with a `soul` field) and to those of `reference_rule.py` on the S, V and
  K Runs; give the evidence values of hand-made triple sequences as exact fractions; refuse a Run
  whose settings differ from its arm's preset in anything but the seed, the generations and the call
  cap; refuse a seed off the list, a gap in the list, a missing arm and more than N_max triples;
  withhold every claim on pilot seeds; and compute the divergence generation, giving none for two
  short Runs built to agree (an S Run and the same Run with another `kappa_seed` while Observations
  and imprinting are off; an S and a V Run with `vote` and `simulator` off).
- `reference_rule.py` gives the same pass times, evidence values, stop triple and verdicts as the
  analysis on the acceptance triples of section 7.
- The presets of `--soul` in the code equal the numbers of section 2.
- The operating characteristics of the chosen rule, with an independent recount, are in this
  ticket's Comments; the owner's approval is recorded there; the squash merge of the pull request
  freezes the document, and its commit hash is recorded, before the first counted seed starts.

## Comments
