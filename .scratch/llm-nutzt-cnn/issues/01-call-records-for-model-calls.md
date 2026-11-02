# 01 Call records for model calls

Status: needs-triage (waits for the owner's approval of the design)
Type: task
Blocked by: none (it builds on PR #33, which has merged)
Spec: ../spec.md (sections 4.1, 6, 7; decisions Q76, Q99)
Pull request: PR 0, a small one of its own (open question O4)

## What exists (PR #33)

- `backend-ollama/src/lib.rs:184-290`: `CallRecord { key, tag, request, response }`,
  `call_key` (the tag, `#` and `request_key`) and `Recorded<B>`. `Recorded` appends every
  answered call to a JSONL file and serves a filed answer instead of a new call (contract
  I7).
- `backend-ollama/src/main.rs:50-61, 80`: `brut-breed-ollama` wraps every `ollama:`
  backend in `Recorded`, with the file `<out>/llm/worker-calls.jsonl`. A resume replays it.

This ticket keeps that file, that key and that record form, so the records of #33 stay
readable. It adds what #33 lacks.

## What to do

- Move `CallRecord`, `call_key` and `Recorded` into a new core module `src/call_record.rs`,
  outside `src/cascade/`, because it does file I/O (`tests/cascade.rs:14-35`). The `cnn:`
  backends of ticket 05 live in the core, and the core cannot depend on the adapter
  (contract I1). `backend-ollama` uses the moved types. Its test
  `recorded_answers_replay_without_a_new_call` moves with them.
- What the record gains over #33:
  - The outcome is the full `GenResponse` or, new, the error text. #33 files no failed
    call.
  - Unknown fields are refused. Reads are capped at 8 MiB per line, above the 1 MiB
    answer cap of #33 (contract I6). A record whose key is not the `call_key` of its tag
    and request is refused. A hit also needs the stored tag and request to equal the live
    ones. An error names only the line number, never the record's text. #33 checks only
    the stored tag.
  - A torn last line, which a hard stop may leave, is dropped on open. Any other bad line
    is refused. Under #33 one bad line fails the open, and the resume stops at
    `ModelPool::covers`.
  - A stored error is retried, and its new line replaces it. A second request with a key
    in flight waits for the first; #33 would call the model twice.
  - `RecordsOnly` serves records only. A missing record is a fault.
  - The fault slot. The arena turns every backend error into a lost call
    (`src/cascade/arena.rs:143-150, 183-185`), so the recorder keeps the first fault in a
    slot the binary reads. A fault is a missing record in a replay, an error of a tag that
    is neither `sim:` nor `cnn:`, or a failed record write. Errors of `cnn:` tags repeat
    on every try, so they are no faults. Under #33 a failed Ollama call becomes a lost
    call that its generation commits.
- brut-breed:
  - Wrap every backend whose tag does not start with `sim:` in the core's `Recorded`, with
    the file `<out>/llm/worker-calls.jsonl`. That covers the `cnn:` backends of ticket 05
    and every backend that the slot `EXTRA_BACKENDS` builds
    (`src/bin/brut-breed.rs:920-951`). `brut-breed-ollama` then hands over the bare
    `OllamaBackend` and drops its own record path (`backend-ollama/src/main.rs:50-61,
    66-71, 80`). The silenced tags of a Knock-out are the one exception (ticket 07).
  - Check the fault slot before every log append and every checkpoint write, the
    reservation checkpoints of `step_with` and `finish_with` included
    (`src/bin/brut-breed.rs:559-568, 605-631`). On a fault, exit with 1, name the key,
    and commit nothing of that generation or select member. A resume then repeats it.
  - `--replay <old> --out <new>`: run the old settings as one straight run; recorded tags
    come from the old records only, so no genome file is loaded and no Ollama server is
    needed. Mutator answers come from the old log's `LlmExchange` events
    (`src/cascade/arena.rs:710-717`), and `--llm-answers` is refused. A missing record or
    Mutator answer is a fault: exit code 1, never 3. #33 replays only inside a resume of
    the same Run, and there a miss asks the model.
  - `--verify` also checks every record: size, form, key.
  - `<out>/call_times.jsonl`: one line `{tag, seconds}` per live call. No hash covers
    it, and `--verify` ignores it, as for `timings.jsonl`
    (`src/bin/brut-breed.rs:50-52`). #33 records no times.
- README.md (German, `docs/SCHREIBSTIL.md`) and README.en.md describe the record,
  `--replay` and `call_times.jsonl`, and update their section on `backend-ollama`.

## Tests (write them first)

Library tests drive a `Breeder` with a pool in which a test tag is `Recorded` around a
test model. Its reply depends only on the call key and on how many earlier calls had that
key. Thread timing then cannot change a reply, and a repeated call shows.

- `call_record_round_trip`: a record written and read back is equal; its key is the
  `call_key` of its tag and request.
- `a_record_line_written_by_backend_ollama_stays_readable`: a line in the form of #33
  loads and serves its reply.
- `a_record_whose_request_does_not_match_its_key_is_refused`.
- `a_record_is_served_only_to_its_own_tag_and_request`: two tags with one request get two
  records.
- `an_oversized_record_is_refused_without_quoting_it`: the error names only the line
  number.
- `a_torn_last_line_is_dropped_and_any_other_bad_line_is_refused`.
- `recorded_backend_calls_the_inner_model_once_per_key`, also under parallel requests.
- `a_stored_error_is_retried_in_live_mode_and_replayed_in_replay_mode`.
- `records_only_refuses_a_missing_record_and_makes_no_live_call`: the key lands in the
  fault slot.
- `a_failing_real_model_and_a_failed_write_fill_the_fault_slot`, while a failing `cnn:`
  tag does not.
- `a_stopped_and_resumed_run_repeats_no_finished_call`: the inner calls of a straight
  run equal those of the stopped run plus the resumed run, and the lineage is equal.
- `a_run_replayed_from_its_records_writes_the_same_events`: zero live calls, and the
  events are equal byte for byte for a straight run.
- `a_replay_of_a_resumed_run_gives_the_same_born_compared_and_llm_events`: call counts
  differ only by the reservations, as in `kill_switch_then_resume_equals_a_straight_run`
  (`tests/cascade.rs`).
- `the_records_are_equal_at_one_and_three_threads`: the test binary runs itself as a
  child process with `RAYON_NUM_THREADS` at 1 and 3, as `src/cnn/train.rs:554-591` does.
  The records are equal as a set; only their order in the file may differ.
- `a_replay_of_a_run_with_llm_rounds_needs_no_answer_files`: a grid run with LLM rounds
  replays from its log alone; `--llm-answers` is refused.
- `sim_runs_write_no_record_and_keep_their_files`: a grid run and a circuit run write no
  `llm/worker-calls.jsonl`.
- `verify_refuses_an_edited_record_and_ignores_call_times`.
- In `backend-ollama`: `a_backend_from_extra_backends_is_recorded_once`.

## Acceptance criteria

- All tests pass, in the core and in `backend-ollama`. `cargo clippy --all-targets` is
  clean in both.
- The golden tests of `tests/universe.rs:182-236`, `tests/mlp_golden.rs` and the purity
  scan pass unchanged.
- The README in German and English describes the Call record and replay.

## Comments
