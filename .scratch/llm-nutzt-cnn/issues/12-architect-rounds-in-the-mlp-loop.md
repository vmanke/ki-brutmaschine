# 12 Architect rounds in the MLP loop

Status: needs-triage (waits for the owner's approval of the design)
Type: task
Blocked by: 01, 11
Spec: ../spec.md (sections 4.3.1, 4.3.2, 4.3.4, 6; decisions Q101 to Q104)
Pull request: PR 2 (F2)

## What to do

- `Config.architect: Option<ArchitectConfig { every, children, pause }>`, left out of the
  JSON when absent (pattern of `src/config.rs:117-122`). `resume_conflicts` covers it
  (`src/config.rs:175-200`). `Config::validate` refuses it in a run whose model kinds
  have no CNN, `every` = 0, and `children` above `children_per_generation`.
- `src/evolve.rs`: in an Architect round, `propose` (`src/evolve.rs:376-441`) first draws
  up to `children` parents from the generation's stream (`src/evolve.rs:271-275, 413`).
  Every second request merges two parents when a partner exists; the others change one.
  It returns the children, or the requests still waiting. The `RandomMutator` makes the
  rest, as today (`src/evolve.rs:403`).
- An accepted Blueprint is a child with the generator name `architect`. A refused or
  missing one (with `pause` off) is replaced by a random child.
- `RecordKind::ArchitectExchange { generation, key, request, response, accepted, reason,
  child }` in the lineage archive (`src/archive.rs:76`).
- `brut run`:
  - answers from `<run>/llm/answers/<key>.txt` or `--llm-answers <dir>`;
  - an optional `<key>.json` side file names the model that answered, and the record
    stores it; without it the backend stays "answer-book" (`src/cascade/arena.rs:487`).
    It holds at most 4 KiB and exactly `{"model": "<tag>"}`, the tag matching
    `^[a-z0-9][a-z0-9:._-]{0,63}$`. Anything else is refused before the run changes a
    file;
  - a missing answer with `pause` on writes `<run>/llm/requests/<key>.json`, commits
    nothing of that generation, appends one `RunFinished` record with the reason
    "waiting for Architect answers" and exits with 3;
  - the usage text names exit code 3 (`src/main.rs:71`).
- A live Architect (a backend instead of files) is wrapped in the Call record of
  ticket 01.
- README.md (German) and README.en.md describe Architect rounds.

## Tests (write them first)

- `runs_without_an_architect_keep_their_bytes`: `tests/mlp_golden.rs` fixtures.
- `a_missing_answer_pauses_with_exit_3_and_commits_nothing`.
- `a_paused_and_answered_run_equals_a_run_with_answers_from_the_start`: its `Candidate`
  and `ArchitectExchange` records equal those of the straight run, in the same order,
  with timestamps masked as in `tests/mlp_golden.rs`. `RunStarted`, `RunFinished` and
  `SafetyEvent` records are compared on their own: each pause adds one `RunFinished` and
  one `RunStarted` (`src/evolve.rs:134-139, 352-357`).
- `every_architect_exchange_is_an_archive_record`.
- `a_refused_blueprint_is_replaced_by_a_random_child`.
- `pause_off_never_waits`.
- `a_resume_cannot_change_the_architect_settings`.
- `the_side_file_names_the_model_in_the_archive`.
- `an_oversized_or_malformed_side_file_is_refused_before_any_change`.
- `live_architect_answers_are_call_records_and_replay`.
- `architect_answers_cost_no_flops_and_are_counted`.

## Acceptance criteria

- All tests pass. clippy is clean.
- The README in German and English describes Architect rounds.

## Comments
