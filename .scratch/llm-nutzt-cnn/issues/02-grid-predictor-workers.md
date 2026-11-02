# 02 Grid-predictor Workers and the Silent CNN

Status: needs-triage (waits for the owner's approval of the design)
Type: task
Blocked by: none
Spec: ../spec.md (sections 4.2.1, 4.2.4, 4.2.6; decisions Q77, Q80 to Q83)
Pull request: PR 1 (F1)

## What to do

- New pure module `src/cascade/predictor.rs` (no files, network, process or clock; the
  purity scan of `tests/cascade.rs:14-35` covers it):
  - `trait GridPredictor { fn predict(&self, examples: &[Pair], test_input: &Grid) ->
    Prediction }`. `Prediction` is a grid or a refusal with a short reason.
  - `PredictorWorker<P: GridPredictor>` implements `ModelBackend`
    (`src/backend.rs:71-79`).
  - `is_cnn_tag(tag)`: `cnn:` plus 16 lowercase hex characters.
  - `SILENT_CNN_TAG = "cnn:0000000000000000"` and a `SilentPredictor` that refuses every
    task.
- The protocol of `PredictorWorker`:
  - A solve prompt: read it with `parse_prompt` (`src/cascade/puzzle.rs:614-671`),
    predict, and reply with `render_answer` (`src/cascade/puzzle.rs:445-452`). Stop
    reason `Stop`, 0 prompt and 0 completion tokens, backend name = the tag.
  - Earlier answers, the system prompt, temperature, `max_tokens` and `think_tokens` are
    ignored.
  - A judge prompt: predict the test output. Count, for each candidate, the cells equal
    to the prediction; a candidate of another shape has none. Reply `VERDICT A` or
    `VERDICT B` for the higher count (`src/cascade/puzzle.rs:454-460`). A coin from
    `derive_seed(request.seed, "tie")` breaks a tie. The coin picks a position, A or B, as
    SimModel's coin does (`src/cascade/sim.rs:319-325`).
  - A refused prediction, or a prompt it cannot read, gives `StopReason::Refusal` with a
    short text ("I cannot read this task." for an unreadable prompt,
    `src/cascade/sim.rs:36-37`).
- A test predictor in `tests/` that fits a colour map with `fit_colour_map`
  (`src/cascade/puzzle.rs:193-227`) and refuses when none fits.

## Tests (write them first)

- `predictor_answers_a_solve_prompt_with_its_grid`: the reply parses with
  `parse_answer` to the predicted grid.
- `predictor_reads_only_the_prompt`: replacing a puzzle's `test_output` changes no byte
  of any reply.
- `predictor_ignores_sections_traits_seed_and_previous_answers`.
- `predictor_judges_by_agreement_with_its_prediction`: the closer candidate wins, in both
  orders A/B.
- `a_tie_is_broken_by_the_request_seed`: the same request gives the same verdict, and
  over many seeds both verdicts occur.
- `a_refused_task_costs_one_call_and_gives_no_answer`: in `run_cascade`, `usage.calls`
  is 1 and the answer is None.
- `a_refused_judgment_is_no_verdict`: on a scored peer pair it counts as wrong
  (`src/cascade/arena.rs:1592-1596`).
- `a_later_stage_sees_the_predictor_answer`: Stage 1's prompt holds `PREVIOUS ANSWER 1`
  with the predicted grid.
- `residual_path_keeps_a_predictor_answer_sim_cannot_explain`: [colour-map predictor;
  `sim:small` with the mirror strategy] solves more colour-only puzzles of
  `generate_panel` than the mirror Worker alone (`src/cascade/sim.rs:289-291`).
- `the_silent_predictor_refuses_every_task`.
- `is_cnn_tag_accepts_only_16_lowercase_hex`.

## Acceptance criteria

- All tests pass. clippy is clean.
- The purity scan covers `predictor.rs` and passes.
- The golden tests pass unchanged.

## Comments
