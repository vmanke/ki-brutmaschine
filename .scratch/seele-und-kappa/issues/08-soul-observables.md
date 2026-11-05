# 08 Soul observables in the `state` line

Status: resolved
Blocked by: 07
Spec: `../spec.md`, section 13

## What to do

- `src/cascade/observe.rs`: `UniverseState` gets a seventh group `soul`, left out of the JSON in
  Runs without Souls, with the fields of the table in spec 13.1 (Q33, Q51, Q56, Q57, Q58).
- Every field is a pure function of committed state: the archive with its birth, imprinted and
  current Souls, the members' vote tallies (with their counts of `soul_decided` replies), and the
  counts of this commit's Observations and imprints.
- Means over tasks run over the tasks of the search panel, each task once. A statistic of the
  Character is computed once per distinct strength vector and weighted by the number of tasks that
  have that vector (spec 13.1).
- The rules of spec 13.4: sums in a fixed order, no NaN or infinity, undefined values absent, empty
  counts 0.

## Done when

- `no_soul_group_without_souls`: the state lines of a grid Run and a circuit Run without Souls
  hold no `soul` key, and the goldens and state tests of `tests/universe.rs` pass.
- `the_soul_group_counts_by_hand`: on a hand-built archive of 3 members with 5 Workers, every
  field equals a hand count or a hand computation: souls, born and new Souls, origins and their
  entropy, mean normalised Character, contrast and its formula, order per family, moved mean, κ
  quantiles, Observation counts, shift, votes, flipped, tied, coherence and `sim_decided`.
- `contrast_weights_tasks`: with the strengths of spec 5.2 on a search panel of three dressings per
  target, `contrast_independent` is 0.655 with the complex and 0.927 with the real readout, the
  means over the 11 targets of spec 6.3, and not 0.563 and 0.796, the means over the 7 distinct
  strength vectors.
- `the_soul_population_never_falls`: in a 50-generation Run with Souls, `souls` never decreases.
- `fresh_souls_meet_the_formula`: in a Run with `copy` off and κ fixed at 0, so that every Soul
  is fresh and none is pulled, the mean of `contrast` over generations 10 to 50 lies within 0.1
  of `contrast_independent`, with the complex and with the real readout. The archive is small, so
  at 0.05 a correct implementation would miss in about 2 to 6 % of Runs with the real readout (a
  Monte Carlo check of the review).
- `soul_states_survive_a_resume`: a stopped and resumed Run logs the same `soul` groups as a
  straight one, and 1 and 4 threads give the same bytes.
- `no_value_is_nan`: in Runs of every arm, every field of every `soul` group is finite or absent.

## Comments

- 2026-11-02, build. Not committed: this build makes no commits.
  - `observe.rs`: `UniverseState::soul`, a boxed `SoulState` left out of the JSON
    without Souls, with the fields of spec 13.1. It reads the archive (birth, imprinted
    and current Souls, κ, Soul records, the newborns' vote tallies) and a `SoulInput`:
    the readout, the number of families, the search panel's distinct strength vectors
    with their task counts (`strength_counts`), this commit's Observations by kind and the
    Workers they moved, with their Souls before and after.
  - `arena.rs`: `apply_observations` takes the vote and the peer Observations as two lists
    and returns what it applied: the changed Workers, their Souls before, and the count of
    each kind. The state line gets them.
  - Tests: `the_soul_group_counts_by_hand` and `contrast_weights_tasks` in `observe.rs`;
    `no_soul_group_without_souls`, `the_soul_population_never_falls`,
    `fresh_souls_meet_the_formula`, `soul_states_survive_a_resume` and `no_value_is_nan` in
    `arena.rs`. Added: the Observation counts in
    `the_observation_switches_give_their_degenerate_case`, the counts and the shift in
    `observations_act_in_log_order`, and the vote sums per generation in
    `the_soul_population_never_falls`.
  - Measured in `fresh_souls_meet_the_formula` (circuits, 51 state lines, 743 Souls at the
    end): the mean contrast of generations 10 to 50 was 0.661 against 0.655 (complex) and
    0.946 against 0.927 (real).
  - Eight deliberate faults in the code were each caught. One of them showed that a task
    on which every Character is 0 does occur in short Runs; its contrast counts 0.
