# 04 Experience and imagination behind a switch (arm I)

Status: resolved
Blocked by: none
Spec: `../spec-imagination.md`, sections 3.1 to 3.3 and 3.5

## What to do

- Add a setting `imagination` to `BreedConfig`, at the top level, because it does not need the
  phase Soul. It holds the number of draws (4) and the learning rate (1/4). It is left out of the
  JSON when off, so configurations, hashes, logs and fixtures stay byte-identical. It has no CLI
  flag; it comes from the config file, as `variation` does.
- Refuse a Run with `imagination` when its gene pool cannot name families (`families_of` is
  None, grids today), and refuse a resume that changes it.
- Experience (spec 3.1): learned values per Worker and family in the Member, not in the genome.
  Store them as integers (thousandths) so that bytes and hashes cannot depend on float printing.
  Use the exact rounding of spec 3.3: step = (e + sign(e) · 2n) / (4n), integer division towards
  zero, then clamp to [−1000, 1000] after each step.
  A child Worker copies the values of its source Worker; a Worker without a source starts at 0.
- Imagination (spec 3.2): for SectionReplace, SectionAdd, SectionDrop and section crossover, draw
  the operator 4 times on the same parent. Draw 1 uses today's stream; draws 2 to 4 use streams
  derived with the tag `imagine/{g}/{i}/{j}`, so the main stream advances exactly as today. Take
  the draw with the highest prediction; a tie goes to the earliest draw.
- Learning (spec 3.3): after a newborn is evaluated, compute s on the current goal (all tasks
  while ticket 05 is off), the one error e = s − p over all changed Workers, and the normalized
  step of each gained or lost family. The child's Workers learn before the Born event. Their
  source Workers in the parents learn the same steps at commit, from each child in birth order,
  logged as a new event that `--verify` replays, as it replays `Souls`. Draw nothing from the
  generation's random stream.
- Logs (spec 3.5): the `mind` group of the state line, and the new fields of the Born event.
  Every new field is left out when off.

## Done when

- With `imagination` off, everything is byte-identical: the golden fixtures, the drafting
  digests, the prompt digests, and the release runs of seed 101 and of the README's seed 901.
- With `imagination` on and all values 0, generation 1 drafts exactly the children of a Run
  without it, and the generation's random stream ends at the same word.
- Unit tests: the prediction, the update, the clamp, the tie rule, the copy of values at birth
  (mutation, section crossover, stage crossover, new Worker), and the parent's learning in birth
  order. The rounding: negative errors, steps that end in exactly one half, and n above 1. A
  section crossover that changes two Workers: one p over both, one e, and the steps of both.
- A Run with `imagination` over 60 generations is byte-equal at 1 and at 4 threads, passes
  `--verify`, and a resumed Run equals a straight one.
- `cargo test` and `cargo clippy --all-targets -- -D warnings` are clean.

## Comments

- 2026-11-04, built; committed in #57 for the owner to review. Built on ccf059e, with spec
  3.2, 3.3 and 3.5 as revised in da5d9cb and b5c6564 while the ticket was built.
- The setting: `"imagination": {"draws": 4, "rate": 250}` at the top level of `BreedConfig`
  (`src/cascade/mind.rs`, `ImaginationConfig`), with the rate in thousandths. Both fields are
  required and no other is allowed; it is left out of the JSON when off. A Run with it on a
  gene pool that names no families (grids) exits with code 1 before it writes a record, leaving
  the empty run directory and its LOCK file. A resume refuses a change through the fixed hash,
  and refuses an archive whose values break the Run's rule.
- What was built. The member holds `values` (each Worker's values after its own learning step,
  as its Born event logs them) and `values_now` (the current ones), integers in thousandths.
  The Born event gets `values` and `effect` = {p, s, gained, lost, draw}: p in thousandths, s as
  −1, 0 or 1, gained and lost as one list per Worker in Stage order, and draw (from 1) only
  when imagination chose the child. A new event `learned`, after the generation event and its
  `souls` event, lists the Workers whose values changed at the commit, each once with its new
  values, in archive order; it is logged in every generation, empty when nothing changed, as
  `souls` is. `--verify` rebuilds the new Born fields from the archive and replays the `learned`
  events (`check_current_values` in `brut-breed.rs`). The state line gets the group `mind`:
  `goal`, `imagination_choices`, `imagination_changed`, `linear_phase_hadamard_mean` and
  `other_families_mean`.
- Decisions within the spec.
  - Draw 1 is today's child after all of today's checks; if today's attempt fails, no other
    draw is made. Draws 2 to 4 are one application of the same operator on the same parent
    (`variation::mutate_once`) or one full section crossover of the same two parents, each from
    the stream `imagine/{g}/{i}/{j}`, with i the child's place in birth order and j the draw's
    number. A draw that fails validity, admission or novelty is no candidate. Agreed with the
    coordinator and now in spec 3.2.
  - The value source of a child Worker is its first source, the Worker that passes on the
    birth Soul. A Worker names the families of its section texts (`designer::named_families`),
    as `analysis/screen_w.py` reads them. No section name of the gene pool holds a stem, so this
    is what the designer reads in the system prompt.
  - The step is e · rate / (1000 n), rounded half away from zero, computed as
    (e · rate + sign(e) · 500 n) / (1000 n) with division towards zero. At the rate 250 this is
    the spec's (e + sign(e) · 2n) / (4n); a test checks it for e from −5000 to 5000 and n from
    1 to 24.
  - A seed has no parent and learns nothing; its values are 0.
  - The means of the `mind` group read the parents' values when the parents were chosen, before
    the commit's learning. A parent counts once per child, both parents of a crossover count,
    and every Worker of a parent counts. The choices and the changed picks count the children
    of the generation, as `variation_choices` does. A choice is a child that imagination chose
    with some candidate whose p ≠ 0.
- Tests. `mind.rs`: the families a Worker names, gained and lost, the prediction over two
  Workers, the tie rule (`choose`), s and e, the step (negative errors, halves, n above 1, the
  spec's formula), learning and the clamp (also the order at a bound), and the settings.
  `arena.rs`: `imagination_and_goals_are_refused_where_they_cannot_work` (with resumes and
  tampered values); `with_every_value_zero_imagination_drafts_todays_children` (generations 1
  to 3, with dgm and lexicase: the same children, and the stream ends at the same word);
  `imagination_chooses_the_draw_with_the_highest_prediction` (candidates recomputed from their
  streams, the tie rule, changed picks); `a_child_worker_copies_the_values_of_its_source_worker`
  (every mutation operator, WorkerAdd and StageAdd included, section crossover, stage
  crossover, a Worker without a source); `a_section_crossover_that_changes_two_workers_learns_one_error`
  (one p over both Workers, one e, the steps of both, and of their sources at the commit);
  `source_workers_learn_the_steps_of_their_children_in_birth_order`;
  `the_log_holds_what_learning_read_and_did` (values, p, s and every `learned` event
  recomputed from the log alone); `the_mind_group_holds_the_goal_and_how_imagination_chose`;
  `a_run_with_imagination_and_goals_resumes_like_a_straight_one`. `variation.rs`:
  `one_draw_of_an_operator_is_the_attempt_that_drew_it`. `observe.rs`:
  `the_mind_group_averages_values_and_leaves_out_what_it_lacks`. `tests/mind.rs`: the settings
  are left out when off; Runs of 60 generations with the screen's flags in arms I, G and IG
  are byte-equal at 1 and at 4 threads and pass `--verify`; Runs resumed at generation 30
  equal straight ones; `--verify` finds changed values; and the refusals.
- Gate, 2026-11-04. `cargo test` and `cargo clippy --all-targets -- -D warnings` are clean; the
  fixtures are unchanged; the drafting and prompt digests and the golden fixtures pass; the
  release tests `souls_off_reproduces_the_step1_pilot` and `the_readme_run_of_arm_s_is_unchanged`
  pass.
- Smoke Runs on seed 4242 (not a screen seed), 60 generations, the flags of ticket 06, a release
  binary (SHA-256 db177f9a…), configs written as ticket 06 says; all four pass `--verify`. Arm N
  is byte-equal, all five record files, to the same Run of the binary of the W screen, whose
  source equals ccf059e. Arms I and IG first changed a pick in generation 11, and until then
  their events equal arm N's but for the new fields; generations 2 to 10 had no candidate with
  p ≠ 0, because few Workers had values other than 0 and the drawn changes did not touch their
  families. Over 60 generations I made 118 choices and changed 63 picks, IG 96 and 53. These Runs
  are no part of the screen.
- 2026-11-05, screened in ticket 06: the verdict is stop (`../screen-ig.md`).
