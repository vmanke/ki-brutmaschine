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
