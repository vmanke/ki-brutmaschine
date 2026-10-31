# 17 Pre-registration and host run: does the CNN tool help?

Status: needs-triage (waits for the owner's approval of the design)
Type: research
Blocked by: 10, 16; outside: the context follow-up of spec 10.3, a context long enough for
the transcript. PR #33 already passes the prompt, and so the transcript, through
`/api/generate` unchanged (`backend-ollama/src/lib.rs:128-140`), but sets no context.
Spec: ../spec.md (sections 4.4, 8, 9; decisions Q106 to Q108, Q112, Q114)
Pull request: P3 (the pre-registration, frozen at the commit its merge creates), then a
report

## What to do

- Pilot: 2 seeds × arms T and L with real models, on grid-mix, then on ARC training
  tasks. It measures M5 and sets the tool's fitness cost per call, the budgets and the
  most pairs. It also answers O15: native tool calling is reconsidered only if fewer than
  half of the real models' tool requests parse.
- `docs/PREREG_LLM_CNN.md`, section F3, frozen at merge:
  - primary comparison: arm T against arm L; secondary: T against C;
  - everything else ticket 10 fixes, with the tool's `cnn:` tag in place of the Worker's:
    the Domain and its panel sizes, `llm_every` 0, the commit, the best finalist, and
    that a proof takes at least 12 pairs;
  - Knock-outs silence the tool, with the same call seeds;
  - the sequential rule with the thresholds 100 and 1/20;
  - the void conditions of ticket 10.
- Counted runs, then the report section F3. Their Call-record tarballs go to the lasting
  store of O13, and each passes the replay on another machine of ticket 10 before the
  report merges.

## Tests (write them first)

- `the_tool_knock_out_keeps_the_call_seeds`.
- `the_analysis_reproduces_its_numbers`.

## Acceptance criteria

- The owner has approved and merged the pre-registration.
- The report and the run manifests exist.
- The VM is stopped in the Nebius console after the runs.

## Comments
