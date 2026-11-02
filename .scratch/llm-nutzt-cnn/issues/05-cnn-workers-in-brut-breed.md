# 05 CNN Workers in brut-breed

Status: needs-triage (waits for the owner's approval of the design)
Type: task
Blocked by: 01, 03, 04, 06 (the grid-mix Domain and its sizes); outside: cnn-basismodelle 07
(two strict entry points: one for a bare genome, one for a promotion file)
Spec: ../spec.md (sections 4.1, 4.2.1, 4.2.5; decisions Q91 to Q93)
Pull request: PR 1 (F1)

## What to do

- `--cnn-models <dir>`, default `candidates/models/`. For each `cnn:` tag on the
  allow-list, load `<dir>/<id>.json` through a strict entry point of cnn-basismodelle 07.
  A promotion file goes through the promotion-file parse, whose genome part goes through
  the bare-genome parse. A bare genome (the test fixtures) goes through the bare-genome
  parse alone. Size cap, unknown fields refused at every level, the genome must be a CNN,
  and its `Genome::id` must equal the tag.
- At load, check the genome on the smallest and on the largest padded task the run's
  Domain can draw (Q92): the shape check on both, and e_task at least `MIN_EPOCHS` on the
  largest. A new pure Domain method, with a default, returns the smallest and largest
  padded size and the most examples. For grid and grid-mix it is 3 × 3 to 6 × 6 with 3
  examples (`src/cascade/puzzle.rs:34, 343-359`).
- Register `PredictorWorker<CnnPredictor>` under the tag, wrapped in the Call record
  (ticket 01). Register the Silent CNN under `cnn:0000000000000000` without a file. Both
  go in where the pool is built, before the slot `EXTRA_BACKENDS` is asked (`sim_pool`,
  `src/bin/brut-breed.rs:929-951`). So `brut-breed` and `brut-breed-ollama` load CNN
  Workers alike, and a host run mixes `ollama:` and `cnn:` tags in one program.
- Every refusal comes before the first call. A refused resume leaves the run directory as
  it was. A refused fresh run leaves only `<out>/LOCK`, which brut-breed creates before it
  reads the settings (`src/bin/brut-breed.rs:417-424`).
- `RESUME_KEYS` gains `cnn-models` and `allow-other-cpu` (`src/bin/brut-breed.rs:92-94`).
  The genome's content is fixed by its tag, so only the directory may change.
- `--verify` and `--promote` take only `--out` (`src/bin/brut-breed.rs:95`). For each
  allow-listed `cnn:` tag they register a backend that fails if called. A finished run's
  rebuild makes no call (`src/bin/brut-breed.rs:827-847`), so they need no genome file.
- `<out>/cpu.jsonl`: one line per process of a run with `cnn:` tags, with the CPU model
  and its FMA and AVX-512 flags. No hash covers it. A resume refuses a CPU that differs
  from the first line, unless `--allow-other-cpu` is given; the new line then says so.
- Pick `GRID_CNN_BRIEF` when a `cnn:` tag is listed (ticket 03).
- After each committed generation, write the number of CNN trainings of that generation
  to `call_times.jsonl`.
- README.md (German) and README.en.md: CNN Workers, `--cnn-models`, a short container
  demo with its measured numbers.

## Tests (write them first)

- `brut_breed_runs_a_cnn_cascade_and_records_its_calls`: 3 generations with the reference
  genome; every CNN call is in the store.
- `a_cnn_run_is_byte_equal_across_processes_and_thread_counts`: `RAYON_NUM_THREADS` at 1
  and 3; the events are equal, and so are the records of `llm/worker-calls.jsonl` as a
  set.
- `a_cnn_run_replays_with_zero_live_calls` (`--replay`).
- `a_replay_with_one_record_deleted_exits_1_and_appends_no_event_of_that_generation`.
- `brut_breed_refuses_a_bad_cnn_tag`: missing file, id mismatch, unknown field, MLP
  genome, genome too large for the Domain. Each exits with 1 before any call, for a fresh
  run and for a resume. A refused resume leaves the run directory as it was; a refused
  fresh run leaves only `<out>/LOCK`.
- `verify_passes_for_a_cnn_run_whose_genome_is_outside_candidates_models`, and
  `--promote` works on that run too.
- `a_resume_refuses_a_cpu_that_differs_from_the_first_line` unless `--allow-other-cpu`
  is given.
- `a_listed_cnn_tag_without_a_backend_is_refused` (`ModelPool::covers`,
  `src/cascade/arena.rs:70-76`).
- `a_resume_may_change_the_models_directory_but_not_the_genome`.
- `llm_round_requests_of_a_cnn_run_use_the_cnn_brief`.
- `the_silent_cnn_needs_no_file`.

## Acceptance criteria

- All tests pass. clippy is clean. The golden tests pass unchanged.
- The README in German and English shows the demo with its measured numbers.

## Comments
