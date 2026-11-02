# 05 Observations and the Coupling strength

Status: resolved
Blocked by: 04
Spec: `../spec.md`, section 8

## What to do

- Record the Observations of spec 8.2 (Q62): in Stage votes, judge → authors of both answers other
  than itself, for every judgment with a reply; in the peer round, judge Worker → authors of both
  judged answers. A call without a reply observes nothing. The switches `observe_vote` and
  `observe_peer` turn each kind off.
- Apply them at the commit, one after another in the log order of spec 8.4, by the integer rule of
  spec 8.3 [Q69 a], to the members' current Souls (`Member::soul_now`, left out of the JSON
  without Souls). Uncommitted generations apply nothing.
- The select phase records and applies nothing.
- Write one `souls` event after each `generation` event: the Workers whose current Soul changed,
  with member id, Worker index and new phases (spec 8.5). The `state` event stays the last of the
  step.
- `--verify` checks every member's current Soul against the `souls` events (spec 16). The check
  goes into `check_archive`, which today runs only for a finished circuit Run; ticket 09 runs it for
  grid Runs with Souls too.

## Done when

- `kappa_zero_changes_nothing`: a Run of arm K writes `souls` events without any changed Worker, and
  every current Soul equals the birth Soul.
- `one_observation_by_the_rule`: at κ = 1000, an observed Soul 150 degrees away takes the observer's
  phases and the observer moves 75 degrees, in exact integers; at κ = 200 the hand-computed integers
  of spec 8.3.
- `observations_act_in_log_order`: a hand-built generation with two Stage votes and two peer
  judgments gives the hand-computed current Souls.
- `the_select_phase_changes_no_soul`: the current Souls after `finish` equal those before; a Run
  resumed to more generations after a finished select phase writes the same `souls` events and
  `state` lines as a straight Run.
- `stop_and_resume_keep_the_souls`: a Run stopped mid-generation and resumed writes the same
  `souls` events as a straight Run.
- `souls_do_not_depend_on_threads`: 1 and 4 threads give byte-equal `events.jsonl` and
  `snapshot.json`.
- `verify_finds_a_changed_soul`: on a finished circuit Run, one changed phase in a checkpoint's
  current Soul makes `--verify` exit with code 1 and an "integrity: …" message.

## Comments

- 2026-11-01, build. Not committed: this build makes no commits.
  - `soul.rs`: `phase_gap`, `coupling_step` and `observation`, the integer rule of spec 8.3
    [Q69 a].
  - `arena.rs`, recording: `stage_vote` records an Observation for every judgment with a
    reply, by the judge of each author of x and then of y other than itself;
    `CascadeRun.observed` carries them. The peer round records the judge Worker's
    Observations of the authors of the first answer, then of the second's. `observe_vote`
    and `observe_peer` switch each kind off. The select phase calls `run_panel` with
    `observe` off and records nothing.
  - `arena.rs`, commit: `apply_observations` applies them one after another to
    `Member::soul_now`: the newborns in birth order (tasks, Stages, pairs, judges, authors),
    then the peer judgments in plan order. A stopped or failed generation applies nothing.
    The new `souls` event (`Event::Souls`, `SoulChange`) follows each generation event and
    lists the changed Workers in archive order; the `state` event stays last.
  - `brut-breed`: `check_archive` calls the new `check_current_souls`: each current Soul
    must be its last entry in the `souls` events, or its birth Soul when none moved it, and
    every Worker an event names must be in the archive.
  - Tests: `one_observation_by_the_rule` (soul.rs); `kappa_zero_changes_nothing`,
    `observations_act_in_log_order`, `the_select_phase_changes_no_soul`,
    `stop_and_resume_keep_the_souls` (arena.rs); `souls_do_not_depend_on_threads` and
    `verify_finds_a_changed_soul` (tests/soul.rs). `observations_act_in_log_order` resumes
    a hand-built archive: an LLM round writes two children of one parent, each with a Stage
    vote, and the peer round gives two judgments by the same judge; every resulting phase
    is computed by hand in the test. Added: `the_observation_switches_give_their_degenerate_case`
    runs the same generation with `observe_vote`, `observe_peer` or both off (hand-computed
    too), and a Run with `simulator` off.
  - Changed test: `a_souled_run_records_its_souls_and_resumes_like_a_straight_one` now
    checks each current Soul against the `souls` events instead of the birth Soul.
  - Left for later tickets: until ticket 07, a newborn's current Souls start as its birth
    Souls, and `--verify` compares with the birth Soul where no event moved a Worker. The
    counts of Observations in the `state` line are ticket 08.
