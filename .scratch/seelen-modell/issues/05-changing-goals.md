# 05 Changing goals behind a switch (arm G)

Status: needs-triage
Blocked by: none
Spec: `../spec-imagination.md`, sections 3.4 and 3.5

## What to do

- Add a setting `goals` to `BreedConfig`, at the top level, with the length of a goal in
  generations (20). It is left out of the JSON when off, so everything stays byte-identical. It
  has no CLI flag; it comes from the config file.
- Refuse a Run with `goals` unless its selection is lexicase and its Domain is circuits, and
  refuse a resume that changes it.
- Build the sequence of goals from the library (spec 3.4): G0 is all tasks of the search panel;
  G−f is the tasks of the targets whose `fits` do not name family f; leave out a G−f that equals
  G0 or is empty; order G0, G−f1, G0, G−f2, and so on over the families in index order, then
  repeat. Generations 1 to 20 have G0.
- Lexicase uses only the tasks of the current goal as its cases. Nothing else changes: every
  newborn is still evaluated on the whole search panel, and the select phase, the pass time and
  `rediscovery.json` stay as they are.
- When ticket 04 is on too, its real effect s uses the solved counts on the current goal.
- Log the current goal in the `mind` group of the state line, for example `all` or `-Linear`.

## Done when

- With `goals` off, everything is byte-identical, as in ticket 04.
- A unit test pins the sequence on today's circuit library: G0, G−Linear, G0, G−Phase, G0,
  G−Hadamard, G0, G−Compose, G0, G−Search, G0, G−Meet, G0, G−Simplify, with the targets of each
  goal as in spec 3.4.
- A unit test shows that lexicase with a goal draws only on the tasks of that goal.
- A Run with `goals` over 60 generations is byte-equal at 1 and at 4 threads, passes `--verify`,
  and a resumed Run equals a straight one. Its `rediscovery.json` is computed as today.
- `cargo test` and `cargo clippy --all-targets -- -D warnings` are clean.

## Comments

- 2026-11-04, built together with ticket 04; committed in #57 for the owner to review.
- The setting: `"goals": {"length": 20}` at the top level of `BreedConfig`
  (`src/cascade/mind.rs`, `GoalsConfig`). Its one field is required and no other is allowed; it
  is left out of the JSON when off. A Run with it is refused unless its selection is lexicase
  and its Domain circuits, and exits with code 1 before it writes a record. A resume refuses a
  change through the fixed hash.
- What was built. At the start the breeder builds one cycle of goals from the search panel
  (`mind::goal_cycle`): G0, then, for each family in index order whose G−f is neither empty nor
  G0, G0 and G−f in turn, with no G0 before the first variant. G−f holds the tasks whose Family
  strength for f, from `LibraryTarget.fits`, is 0. Generation g ≥ 1 has the goal at place
  ((g − 1) div length) mod (the cycle's length); generation 0, which draws no parent, has G0.
  Lexicase reads only the goal's tasks as cases, in panel order: when generation g drafts, and
  for the state line's estimate of the parent draw of generation g + 1. So the state line of
  generation 20 already reads the weights of goal G−Linear. Every newborn is still evaluated on
  the whole panel; fitness, the ranking, the select phase and `rediscovery.json` are as before.
  With ticket 04 on, s compares the solved counts on the goal of the newborn's generation. The
  `mind` group logs the goal as `all` or `-Linear`.
- Tests. `mind.rs`: `the_goals_of_todays_circuit_library` pins G0, G−Linear, G0, G−Phase, G0,
  G−Hadamard, G0, G−Compose, G0, G−Search, G0, G−Meet, G0, G−Simplify with the targets of each
  goal of spec 3.4, and the generations of each goal; `goals_that_equal_all_tasks_or_none_are_left_out`.
  `arena.rs`: `lexicase_with_a_goal_draws_only_on_the_tasks_of_that_goal` (a member that does
  well only outside the goal is never drawn, and the breeder's cases are the goal's columns of
  its scores), and the refusals, the mind group, s on the goal and the resume. `tests/mind.rs`:
  Runs of 60 generations in arms G and IG are byte-equal at 1 and at 4 threads and pass
  `--verify`; their `rediscovery.json` equals the count recomputed from the logged answers, as
  `tests/circuits.rs` recomputes it; the goals of generations 0 to 20, 21 to 40 and 41 to 60;
  Runs resumed at generation 30 equal straight ones.
- Gate and smoke Runs: see ticket 04. In the smoke Run of arm G on seed 4242 the goals were all
  tasks in generations 0 to 20, G−Linear in 21 to 40 and all tasks in 41 to 60; its events equal
  arm N's up to the state line of generation 20.
