# 15 CNN tool: protocol, Tool budget and Tool turns in the arena

Status: needs-triage (waits for the owner's approval of the design)
Type: task
Blocked by: the merge of PR 2 (the owner's order); 01, 02
Spec: ../spec.md (sections 4.4.1, 4.4.2, 4.4.4 to 4.4.6; decisions Q105 to Q109)
Pull request: PR 3 (F3)

## What to do

- `src/cascade/genome.rs`:
  - `ToolSpec { model, cost_per_call, max_tool_calls }` and
    `CascadeLimits.tool: Option<ToolSpec>`, left out of the JSON when absent;
  - `TOOL_CALLS_CEILING = 3`;
  - `Traits.tool_calls`, left out of the JSON at 0, so old Cascade ids stay
    (`src/cascade/genome.rs:314-321`);
  - checks: the tool's model is a `cnn:` tag that is not on the allow-list; its cost lies
    in [0, 1]; `max_tool_calls` lies in 1 to the ceiling; `check_worker`
    (`src/cascade/genome.rs:362-405`) refuses `tool_calls` above 0 without a tool, above
    `max_tool_calls`, or on a CNN Worker.
- `src/cascade/puzzle.rs`: `TOOL_TEXT`; the tool request parser (the reply's last
  non-blank lines are `TOOL CNN` and a valid grid); the tool prompt (the task's examples
  and the requested grid as `TEST INPUT`); the transcript (after `END`: `REPLY k`, each
  reply line prefixed with `| `, then `TOOL RESULT k` and a grid or `NONE`);
  `ParsedPrompt.tool_results`.
- `src/cascade/domain.rs`: tool methods with defaults that mean "no tool". Circuits keep
  the defaults and refuse `tool_calls` above 0. Grid and grid-mix implement them.
- `src/cascade/arena.rs`:
  - a new `solve_request_with`: for a Worker with `tool_calls = t > 0` in a run with a
    tool, it appends `## tool`, `TOOL_TEXT` and "You may ask the tool at most t times."
    (as `## check`, `src/cascade/arena.rs:115-119`). Only `run_cascade_in` uses it, for
    solve calls (`src/cascade/arena.rs:253`). `worker_request_with` stays as it is,
    because judge calls go through it too (`src/cascade/arena.rs:149, 183`);
  - the loop of section 4.4.1 for solve calls; the seed of turn k is
    `derive_seed(s, "turn/k")`;
  - the tool request of turn k: an empty system text, the Worker's generation limits and
    the seed `derive_seed(s, "tool/k")`, so it never equals a Worker request;
  - a requested grid of another height or width than the test input gets `NONE` without
    a tool call;
  - a request after the budget is spent ends the answer without a grid;
  - judges never use the tool;
  - the call bounds sum (1 + 2 t) over a Stage's Workers, plus the judging calls; without
    tools they equal today's (`src/cascade/arena.rs:198-209`);
  - `Usage.tool_calls`, left out at 0.
- `src/cascade/variation.rs`: ToolCallsShift (±1) and the `tool_calls` crossover, drawn
  only in runs with a tool; static tool briefs (`GRID_BRIEF` plus the tool paragraph, and
  plus both paragraphs); one more line in `describe_limits` when a tool exists.

## Tests (write them first)

- `a_tool_request_is_read_only_at_the_end_of_a_reply`.
- `a_tool_request_with_a_bad_grid_is_a_final_reply`.
- `a_reply_cannot_forge_a_tool_result`: a reply that writes `TOOL RESULT 1` and a grid
  shows up quoted, and `parse_prompt` does not read it as a result.
- `the_tool_prompt_holds_only_the_examples_and_the_requested_grid`.
- `a_grid_of_another_size_gets_none_without_a_call`.
- `tool_calls_is_left_out_of_the_json_at_zero`.
- `tool_calls_above_the_cap_without_a_tool_or_on_a_cnn_worker_is_invalid`.
- `a_domain_without_tools_refuses_tool_calls`.
- `each_tool_turn_counts_one_tool_call_and_one_model_call_at_their_costs`.
- `actual_calls_never_exceed_the_bound`: random Cascades with scripted backends.
- `the_loop_stops_at_the_budget_and_a_late_request_gives_no_answer`.
- `judges_never_get_the_tool`.
- `judge_requests_of_a_tool_worker_carry_no_tool_text`: in the in-stage vote and in peer
  judging.
- `the_tool_request_has_an_empty_system_text_and_its_own_seed`.
- `every_turn_is_a_call_record_and_replays`.
- `parse_prompt_reads_a_transcript_after_end`.
- `runs_without_a_tool_keep_every_golden_fixture` and
  `describe_limits_names_the_tool_only_when_present`.

## Acceptance criteria

- All tests pass. clippy is clean.
- The golden tests pass unchanged.

## Comments
