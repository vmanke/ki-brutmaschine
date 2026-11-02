# 11 Pilot of the step 2 comparison

Status: needs-triage (waits for the owner's approval of the design; Q71 and Q74 are decided)
Blocked by: 10 (its draft and analysis; ticket 10 is resolved only after this pilot)
Spec: `../spec.md`, sections 12.2 and 17

## What to do

- Build at the commit that holds tickets 01 to 10. Run the arms S, V and K on the pilot seeds 901 to
  912, with the settings of the counted Runs of step 1 (1000 generations, call cap 13 000 000) and the
  placeholders of spec 12.2 (κ_seed 200, `kappa_drift` 20, `phase_drift` 512, θ = 1). Run arm E (arm S
  with `strengths: equal`) and arm N (no Soul, which is arm A of step 1) on the same seeds, for
  description only.
- If the placeholders give a degenerate pilot, for example no complex vote ever flips a winner, or
  every Stage's Souls align within a few generations, run one more batch on seeds 913 to 920 with
  changed placeholders. Write the change and its reason into this ticket before that batch starts.
- Write `docs/pilot-step2/README.md` and `docs/pilot-step2/pilot.json`, as `docs/pilot-step1/` does.
- From the pilot, propose p₁, N_max, G, M and the four placeholders for ticket 10, each with its
  reason, and compute the operating characteristics of the proposed rule with a script in
  `.scratch/seele-und-kappa/rule/`; a second script, written separately, recounts them.
- Say in the README which acceptance triples ticket 10 uses: the pilot triples 901 to 903, run
  again at the frozen commit, if no number of a preset changes; else three new triples on seeds 921
  to 923 with the final presets (ticket 10, section 7).
- Mark seeds 901 to 923 as pilot seeds that never count.

## Done when

- Every pilot Run passes `--verify`, and one Run per arm, run again, gives byte-equal files.
- Every Run of arm N has a `config.json` without a `soul` field, and its log has no `souls` event and
  no `soul` group.
- `pilot.json` holds, per Run: the pass time, D, calls, wall time and the `soul` group of every 10th
  generation; per seed: Y_V, Y_K and the divergence generation of each pair of arms, as section 12
  of ticket 10 defines it (not the first difference of the `born` events, which carry κ and the vote
  tallies and so differ from generation 0 or from the first vote); per arm: the shares of complex
  votes that flipped a winner and that were tied, and the replies in which the phases decided a
  simulated model's choice (`sim_decided`).
- The README proposes the numbers with their reasons and gives the operating characteristics of the
  proposed rule: the worst-case probability of a false proof at the cap, the power at the pilot's
  estimate of P(S first | untied) and over its uncertainty, the mean and the 90th percentile of the
  triples needed, and the hours; both scripts agree to the last printed digit.
- If, with N_max = 120, the power of a claim at the pilot's estimate is below 0.5, the README says
  so plainly and proposes either to drop that claim from the pre-registration or not to run the
  counted experiment. The owner decides before ticket 10 is frozen.
- Once the owner has approved the numbers, the presets of `--soul` in the code equal them, before
  ticket 10 is frozen.
- `reference_rule.py` and `examples/soul_comparison.rs` give the same numbers on the pilot triples;
  the acceptance check itself runs at the frozen commit (ticket 10, section 7).

## Comments

- 2026-11-02, pilot. Not committed (this work makes no commits). The status stays: the owner
  decides the numbers.
  - **What ran.** Arms S, V, K, E and N on seeds 901 to 912, 60 Runs, with the release build of
    `e315220` (pull request #48) on rustc 1.97.0, x86_64 Linux, the settings of the counted Runs
    of step 1 and the placeholders of spec 12.2 (`kappa_seed` 200, `kappa_drift` 20,
    `phase_drift` 512, θ 1). Arm E ran through `--config` (arm S with `strengths: equal`). Two
    Runs at a time, `RAYON_NUM_THREADS=2`, `nice -n 5`. Data: `docs/pilot-step2/` (README,
    `pilot.json`, the outputs of the analysis and the recount, the driver and its log).
  - **Checks.** All 60 Runs pass `--verify`. Seed 901 ran again in all five arms, one process at
    a time: the five record files of each are byte-equal to the pilot's. Every Run of arm N has
    no `soul` field, no `souls` event and no `soul` group. `examples/soul_comparison.rs` and
    `reference_rule.py --pilot 901-912` give the same pass times, outcomes, evidence values and
    divergence generations on all 12 triples (`pilot.py` checks every row); the step 1 analysis
    gives the arm-N Runs the same pass times and refuses an arm-S Run.
  - **Numbers.** Pass rates by generation 1000: S, K, E and N 12 of 12, V 9 of 12. Y_V: S first
    4, V first 4, ties 4; Y_K: S first 3, K first 3, ties 6; so both estimates of P(S first |
    untied) are 1/2. Against E and N (description only): S first 0, E first 4, ties 8; S first
    0, N first 5, ties 7. Divergence generations: S-V 4 to 246, S-K 12 to 564, V-K 4 to 246, S-E
    12 to 536, S-N 12 to 263; every tie of either claim is a tie by construction (S and the
    control passed before their Runs first differed), and in 4 triples (901, 906, 908, 911) all
    three Runs passed before any two differed. Shares of complex votes: flipped 5.2 % (S), 1.0 %
    (V), 5.4 % (K), 9.7 % (E); tied 5.7 %, 37.6 %, 6.1 %, 0 %. `sim_decided` over all newborns:
    2 912 (S), 6 049 (V), 3 423 (K), 3 824 (E). At most 1 264 833 calls per Run (K-908); no Run
    at the cap, no integrity alarm, no invention candidate. A Run took 9.3 to 14.7 min with
    Souls, 7.2 to 10.1 min in arm N, two at a time.
  - **Degenerate? No; no second batch.** Votes flip in 45 of the 48 Runs with Souls (not in
    V-902, V-904, V-909), the phases decide simulated replies in all 48, and S, V and K part on
    every seed. The hint (seed 901, 60 generations, the same 484 newborns in S, V and K) has a
    mechanism: arms with the same genes share every random number (spec 15) until a vote flips
    or the phases decide a simulated reply; on seed 901 the first flip came in generation 197 in
    S, V, K and E alike and the first phase-decided reply at 242 to 269, and up to generation 60
    each arm had 32 complex votes, none flipped. `diagnose.py` confirms the mechanism on all 60
    first differences (12 seeds, 5 pairs). Such chances need split verdicts or two-family
    circuits, which come from the genes and the tasks; the placeholders only decide which
    chances turn into a flip. So no placeholder changed, no second batch ran, and seeds 913 to
    920 stay unused. Whether larger phase steps would give the Soul more effect the pilot does
    not answer; those seeds are free if the owner wants to ask.
  - **Proposals** (for ticket 10; the owner decides): G = 1000, M = 13 000 000, p₁ = 3/4, N_max
    = 120, and the four placeholders unchanged (200, 20, 512, 1). Reasons in the README, section
    "Proposals for ticket 10". So no preset changes, and the acceptance triples of ticket 10 are
    the pilot triples 901 to 903, run again at the frozen commit; seeds 921 to 923 stay
    reserved. Seeds 901 to 923 are pilot seeds and never count.
  - **Operating characteristics** of the proposed rule (`oc.py`; `own_check.py`, written apart,
    prints the same figures to the last digit): worst-case false proof of one claim at the cap
    2617038152666572495921938445760671/664613997892457936451903530140172288 = 0.003938 (some
    claim at most 0.007875); a wrong failure at P(S first | untied) = 3/4 without ties 0.036848;
    power at the pilot's estimate 0.0038 (V) and 0.0034 (K); averaged over the uncertainty
    0.1485 and 0.1528; no claim proven with probability 0.9929; 53.01 triples on average, 90 %
    need at most 102; 28.55 hours on average and at most 75.32 hours for Runs one at a time at 2
    threads (32.32 min per triple on average), plus about 2.7 hours for the acceptance and
    thread checks. `oc.py` also compares p₁ = 3/5, 2/3, 7/10 and 4/5 and caps of 60 to 120: no
    design reaches an averaged power of 0.16.
  - **Power.** With N_max = 120 the power of each claim at the pilot's estimate is far below 0.5
    (0.0038 and 0.0034). As this ticket asks, the README says so plainly and proposes not to run
    the counted experiment as designed (dropping both claims comes to the same). One claim
    reaches power 0.5 only if P(S first | untied) is about 0.68 or more at the pilot's tie rate.
    The owner decides before ticket 10 is frozen.
