# 02 Family strengths, readouts and the Character

Status: resolved
Blocked by: 01
Spec: `../spec.md`, sections 5, 6 and 12.1

## What to do

- `Domain` gets `families()` and `strengths(task)` (spec 5.1); the default `families()` is
  empty. `GridDomain` and `CircuitDomain` implement both.
- Circuits: the families in the order of `FAMILY_STEMS`; the table of spec 5.2 as a new field of
  each verified `LIBRARY` entry, with a note that names the pinned routes [Q67 a]. A target's
  dressed tasks share its strengths.
- Grids: the families in the order of `sim.rs`; strengths from the puzzle's hidden rule (spec
  5.3).
- The setting `strengths: equal` gives every family 1 on every task (Q63).
- A new module `src/cascade/soul.rs` with pure functions: the readouts u(p) of spec 3 (complex,
  and real by the sign rule [Q70 a]); `character(soul, strengths, readout)`, the Character
  χ = |Z|² of spec 6.1, and `character_norm`, χ divided by Σ s², which the logs use;
  `vote_phasor(soul, strengths, readout)`, the v = Z of spec 7.2; and the formulas K_C and K_R of
  spec 6.3. One function builds the unit phasor of a phase from `cos` and `sin`, so the whole crate
  uses one table of values.
- Nothing stores a Character (Q55).

## Done when

- `the_circuit_families_are_the_designers`: `CircuitDomain::families()` equals the names of
  `FAMILY_STEMS`, in order; `the_grid_families_are_the_grid_models`: the same for `GridDomain` and
  the families of `sim.rs`.
- `every_target_states_its_strengths`: the table equals spec 5.2; each target has at least one
  positive strength; Previous is 0; the dressed tasks of a target have its strengths.
- `grid_strengths_follow_the_rule`: one case per kind of rule (each geometric family, colour only,
  step with colour), each with the strengths of spec 5.3.
- `strengths_never_reach_a_prompt`: for every task of a search panel of each Domain, the solve and
  judge prompts and `worker_request_with` give the same bytes as before this ticket (hashes
  recorded before the change), so adding the strengths to `LIBRARY` and to the puzzles changes no
  prompt. Whole Runs cannot be compared here, since the breeder refuses Runs with Souls until
  ticket 03; that check follows in ticket 04, where the strengths first act.
- `the_real_readout_splits_at_quarter_turns`: u(16383) = +1, u(16384) = −1, u(49151) = −1,
  u(49152) = +1, u(0) = +1, u(32768) = −1.
- `the_character_of_fresh_souls_meets_the_formulas`: over 2^20 = 1 048 576 Souls per Domain drawn
  from a fixed seed, for the strengths of every circuit target, of every kind of grid rule and of
  equal strengths: the mean normalised Character lies within 0.01 of 1; the contrast within 0.01
  of K_C (complex) and of K_R (real); where at least two families fit, K_R / K_C lies within 0.01
  of √2. This checks the code: for independent Souls the formulas are exact, and at this size 0.01
  is about seven standard errors or more, so a correct implementation does not fail by the luck of
  the seed (at 65 536 Souls it would fail at 4 to 5 seeds in 10).
- `one_family_gives_no_pattern`: on swap and cs every Soul has Character 1.
- `a_coherent_soul_is_k_times_brighter`: with all phases equal and k equal strengths the Character
  is k², k times its mean k over independent Souls; the normalised Character is k.

## Comments

- 2026-10-31, build. `Domain` has `families()` and `strengths(task)`, both empty by default,
  and the breeder refuses a Run with Souls on a domain without families. Grids:
  `sim::family_names` and `sim::rule_strengths`. Circuits: each `LIBRARY` entry has `fits`
  (the table of spec 5.2) and `routes` (the pinned routes it comes from), and
  `LibraryTarget::strengths`; a dressed task looks up its target, so it keeps the strengths.
  `soul.rs` has `Phasor`, `phasor` (the one unit phasor of the crate, exact at quarter turns),
  `readout`, `vote_phasor`, `character`, `character_norm`, `mean_character`,
  `contrast_complex`, `contrast_real`, `independent_contrast` and `Strengths::of` (equal
  strengths). Nothing stores a Character. Tests: in `circuit.rs`
  `the_circuit_families_are_the_designers` and `every_target_states_its_strengths`; in
  `domain.rs` `the_grid_families_are_the_grid_models` and `grid_strengths_follow_the_rule`;
  in `tests/soul.rs` `strengths_never_reach_a_prompt` (digests recorded before the change);
  in `soul.rs` `the_real_readout_splits_at_quarter_turns`,
  `the_character_of_fresh_souls_meets_the_formulas` (2^20 Souls per Domain; every value came
  within 0.002 of its formula), `one_family_gives_no_pattern`,
  `a_coherent_soul_is_k_times_brighter` and four more; in `arena.rs`
  `a_run_with_souls_needs_a_domain_that_names_its_families`.
