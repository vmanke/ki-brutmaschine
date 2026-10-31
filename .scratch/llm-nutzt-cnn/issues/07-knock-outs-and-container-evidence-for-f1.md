# 07 Knock-outs, the Silent CNN and container evidence for F1

Status: needs-triage (waits for the owner's approval of the design)
Type: task
Blocked by: 05, 06
Spec: ../spec.md (sections 4.2.4, 9.1, 9.3; decisions Q95, Q96, Q112)
Pull request: PR 1 (F1)

## What to do

- `brut-breed --out <dir> --knock-out` for a finished run:
  - Score the best finalist, and every other finalist with a CNN Worker, on the select
    panel with its CNN Workers silenced. The best finalist is the first bred `Finalist`
    event, in search rank (spec 9.3). A silenced Worker keeps its tag, so Cascade ids and
    call seeds (`select/<id>`) stay; only its backend refuses.
  - Register each silenced tag as the bare `SilentPredictor`, never inside `Recorded`.
    The Call-record key names the tag, not the backend. So a silenced Stage-0 CNN sends
    its as-bred request under its as-bred key, and a stored reply would undo the silencing.
  - Every other tag keeps `Recorded`. It reads the run's `llm/worker-calls.jsonl`, so an
    unchanged request gets its stored reply, and it writes new records to
    `<dir>/knockout/worker-calls.jsonl`. The run's file stays as it was. On the host this
    holds for the `ollama:` tags of `brut-breed-ollama` too.
  - Use the stored as-bred select answers (`src/cascade/arena.rs:796-800`) for the other
    column.
  - Write `<dir>/knockout.json`: per Cascade, b (solved only as bred), c (solved only when
    silenced) and the exact one-sided sign test.
  - Reserve its worst-case calls before it starts, like the select phase, and resume after
    a stop. Its state lives in `<dir>/knockout/state.json`; the run's checkpoint stays as
    it was.
- Run the container controls of section 9.1 with SimModel: F1-pos (grid-mix, C vs L),
  F1-neg (grid-mix, Z vs L) and F1-grid (grid, C vs L). Arm L is `without_cnn` of arm C
  (ticket 03). Every arm sets `llm_every` to 0 (Q86).
  Each runs one pilot seed first. Its pass criteria and its counted seeds are then
  committed and pushed to the draft pull request, before the counted seeds run. The
  report cites that commit.
- Report: `docs/LLM_NUTZT_CNN.md` in German (`docs/SCHREIBSTIL.md`) and
  `docs/LLM_USES_CNN.md` in English, section F1, with the measured numbers. Every number
  from SimModel is labelled as a positive control that says nothing about real models.

## Tests (write them first)

- `the_silent_cnn_refuses_every_task_and_still_counts_its_calls`.
- `a_knock_out_pairs_each_finalist_with_its_silenced_self`: same seeds; the as-bred
  column equals the stored select answers.
- `a_knock_out_never_serves_a_stored_reply_to_a_silenced_tag`: a one-Worker Cascade whose
  CNN solves every select task. Every silenced call is a refusal, none is answered from
  the run's records, c is 0 and b equals the as-bred solved count.
- `the_best_finalist_is_the_first_bred_finalist_in_search_rank`.
- `a_knock_out_spends_its_own_budget_and_resumes`.
- `the_sign_test_matches_hand_counts`.
- `the_control_arm_is_without_cnn_of_the_cnn_arm`.

## Acceptance criteria

- All tests pass. clippy is clean.
- The report gives the measured numbers of F1-pos, F1-neg and F1-grid.
- If the positive control F1-pos fails, the report says why, and ticket 10 stays blocked.

## Comments
