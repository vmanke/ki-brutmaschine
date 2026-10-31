# 05 Observations and the Coupling strength

Status: needs-info (Q69)
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
