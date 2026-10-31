# 16 CNN tool in brut-breed, simulated tool use and container evidence

Status: needs-triage (waits for the owner's approval of the design)
Type: task
Blocked by: 05, 07, 15
Spec: ../spec.md (sections 4.4.3, 9.1, 9.3; decisions Q96, Q105, Q112)
Pull request: PR 3 (F3)

## What to do

- brut-breed registers the tool's backend through `--cnn-models`, as for a CNN Worker
  (ticket 05), wrapped in the Call record. The silent tool is the Silent CNN tag.
  `ModelPool::covers` checks the tool's tag too. `--verify` and `--promote` register, for
  the tool's tag, a backend that fails if called, as ticket 05 does for `cnn:` Workers.
- `src/cascade/sim.rs`: when the system prompt holds `TOOL_TEXT` and no hypothesis fits,
  SimModel asks the tool once with the test input. With a `TOOL RESULT` grid it answers
  with that grid; with `NONE` it goes on as today (`src/cascade/sim.rs:256-294`). Without
  `TOOL_TEXT` its replies stay byte for byte.
- `--knock-out` (ticket 07) also silences the tool. The tool's tag is registered as the
  bare `SilentPredictor`, never inside `Recorded`, as for silenced CNN Workers. The
  Worker's unchanged first turn may still come from the run's `llm/worker-calls.jsonl`.
- Run the container controls F3-pos (grid-mix, T vs L), F3-neg (grid-mix, U vs L) and
  F3-vs-C (grid-mix, T vs C) of section 9.1. Every arm sets `llm_every` to 0 (Q86). Each
  runs one pilot seed first. Its pass criteria and its counted seeds are then committed
  and pushed to the draft pull request, before the counted seeds run. The report cites
  that commit.
- Report: section F3 of `docs/LLM_NUTZT_CNN.md` and `docs/LLM_USES_CNN.md`, with M5
  (tool requests per answer, the share that parse, the share of final answers equal to a
  tool result).
- README.md (German) and README.en.md describe the CNN tool and the demo's numbers.

## Tests (write them first)

- `sim_with_the_tool_solves_neighbour_puzzles`.
- `sim_replies_are_unchanged_without_the_tool_text`.
- `the_tool_text_names_no_sim_keyword` (`src/cascade/sim.rs:54-69`).
- `a_tool_run_is_byte_equal_across_processes_and_thread_counts`.
- `a_tool_run_replays_from_its_records`.
- `a_silent_tool_answers_none_and_still_counts`.
- `a_knock_out_silences_the_tool_and_keeps_the_call_seeds`: no silenced tool call is
  answered from the run's records.

## Acceptance criteria

- All tests pass. clippy is clean. The golden tests pass unchanged.
- The report section and the README show the measured numbers.
- If the positive control F3-pos fails, the report says why, and ticket 17 stays blocked.

## Comments
