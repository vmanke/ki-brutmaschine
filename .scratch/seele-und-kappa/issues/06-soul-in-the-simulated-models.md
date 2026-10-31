# 06 The Soul in the simulated models

Status: needs-info (Q72)
Blocked by: 04
Spec: `../spec.md`, section 11

## What to do

- `src/backend.rs`: `GenRequest` gets `phases: Option<Vec<u16>>`, left out of the JSON when `None`,
  so every request key stays as it is. `GenResponse` gets `soul_decided: bool`, left out of the
  JSON when false. `ModelBackend` gets `reads_phases()`, false by default and true for `SimModel`
  and `SimDesigner`.
- `docs/STAGE2_BACKEND_CONTRACT.md`, which is normative for the backend: contract-v3, with the two
  fields and `reads_phases()`. `phases` has the serde default `None`, is left out of the JSON when
  `None` and is never filled for a backend whose `reads_phases()` is false, so the Ollama adapter
  and every recorded request stay as they are.
- The breeder fills `phases` with the Worker's current phases, readout applied (0 or 32768 in arm
  V), only when `simulator` is on, only for a backend that reads phases, and only in the solve and
  Stage-vote requests of a Cascade's own evaluation, on the search and on the select panel; never
  in a peer judgment (spec 7.8, 11.1). Strengths never enter a request (spec 5.1, 11.1). The
  member's tally counts the replies with `soul_decided` (spec 7.7).
- `src/cascade/designer.rs` [Q72 a]: with phases and the self-check text, the pool also keeps a
  later candidate of the best rank that passes the check on every column, without paying for that
  check; each kept candidate records its support, the set of families that produced it; the reply
  is the kept candidate of the best rank with the largest |Σ e^(iφ_f)|² over its support, then the
  first found (spec 11.2). Without the self-check text the reply stays the first found. The judge
  is unchanged. `soul_decided` is true when the reply differs from the first found.
- `src/cascade/sim.rs` [Q72 a]: with phases, when fitting hypotheses explain both answers, the
  judge picks the answer whose families, each counted once among the explaining hypotheses tested
  so far, give the larger |Σ e^(iφ_f)|²; equal values go to the coin, which is drawn wherever it
  is drawn today, also when the phases decide (spec 11.3). `soul_decided` is true when the verdict
  differs from the coin's. The solver is unchanged.

## Done when

- `requests_without_phases_are_unchanged`: for a backend that does not read phases, a Cascade with
  Souls sends, task by task, byte for byte the requests that the same Cascade without Souls sends
  (the seeds come from the gene id, ticket 03); the JSON of such a request has no `phases` key, and
  a response with `soul_decided` false has no `soul_decided` key.
- `peer_requests_carry_no_phases`: in a short Run with Souls on each Domain, no peer-judgment
  request carries `phases`, and every solve and Stage-vote request of a Cascade's own evaluation
  to a phase-reading backend does.
- `docs/STAGE2_BACKEND_CONTRACT.md` says contract-v3 and states the two fields and
  `reads_phases()` as above.
- `the_step1_calibration_holds_with_phases`: the calibration tests of `PREREG_STEP1_CIRCUITS.md`,
  section 5, pass unchanged, and pass again with random phases attached to every request: the seed
  Cascades solve exactly swap and cs on every dressing, and every pinned route keeps its fewest
  thinking tokens and its exact answer cost.
- `designer_costs_do_not_move`: on 200 random requests with and without phases, with and without
  the self-check text, the reply's T, CNOT and gate counts, its exactness (`grade(..).correct`)
  and the reported thinking tokens are equal.
- `the_designer_prefers_constructive_support`: a hand-built target with two equally cheap exact
  circuits, one of them found by two families (for example Linear and Simplify finding the same
  circuit): with the self-check text the reply is the one with the larger intensity, and with equal
  intensities the first found; without it the reply is the first found; between two one-family
  candidates the reply is the first found whatever the phases.
- `the_grid_judge_breaks_its_coin_by_phase`: when both answers are explained, the answer whose
  families give the larger intensity wins; two hypotheses of one family count once; with equal
  intensities the seeded coin decides as today; the random stream after the verdict is the same
  with and without phases.
- `soul_decided_is_reported`: on the cases above, the flag is true exactly when the reply differs
  from the reply to the same request without phases.
- `real_arm_phases_are_signs`: in arm V every `phases` entry is 0 or 32768.

## Comments
