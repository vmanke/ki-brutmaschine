# 14 Pre-registration and host run: does the Architect design better CNNs?

Status: needs-triage (waits for the owner's approval of the design)
Type: research
Blocked by: 08, 13; outside: cnn-basismodelle 06 (ARC grid tasks in the MLP mode); for a
gpt-oss Architect (O12), the thinking caps of spec 10.3. PR #33 has `OllamaBackend` but no
file bridge, so this ticket builds the bridge.
Spec: ../spec.md (sections 4.3, 9; decisions Q103, Q104, Q112, Q114)
Pull request: P2 (the pre-registration, frozen at the commit its merge creates), then a
report

## What to do

- Build the file bridge (Q104) in `backend-ollama`, as a second program next to
  `brut-breed-ollama`. It answers each `<run>/llm/requests/<key>.json` through
  `OllamaBackend` and writes `<key>.txt` and the side file of ticket 12, which names the
  `ollama:` tag. An unusable answer (`GenResponse::is_usable`) becomes an empty answer
  file, which the Blueprint check refuses (Q102). Its tests run against a test server on
  127.0.0.1, as those of #33 do.
- Fix one answering model (O12) and record its digest. Every answer carries the side file
  that names it.
- Pilot: 2 seeds of an Architect Run and its Control run on the ARC grid task of
  cnn-basismodelle 06, or on a generated image task until 06 exists.
- `docs/PREREG_LLM_CNN.md`, section F2, frozen at merge. It fixes:
  - Architect Runs against Control runs with the same settings, secret and seeds
    (CONTEXT.md, "Control run");
  - the task and its budgets, the Architect settings (`every`, `children`, `pause`), the
    answering model with its digest, and the commit the runs are built from;
  - the primary outcome: the rounded hidden score of each run's best genome, from
    `brut-eval`, paired by seed, decided by the sequential rule; a proof takes at least
    12 pairs (ADR 0006:175);
  - secondary outcomes: the acceptance rate of Blueprints and the refusal reasons;
  - a descriptive outcome: the best Blueprint as a CNN Worker in arm C of ticket 10.
- Counted runs, then the report section F2. The run directories, with their answer and
  side files, go to the lasting store of O13 before the report merges. A counted run
  whose files are lost is void.

## Tests (write them first)

- `the_control_run_differs_only_in_the_architect_setting`.
- `the_analysis_reproduces_its_numbers_from_the_archives`.
- In `backend-ollama`: `the_bridge_answers_every_request_and_names_the_model`, and
  `an_unusable_answer_becomes_an_empty_answer_file`.

## Acceptance criteria

- The owner has approved and merged the pre-registration.
- The report and the run manifests exist.
- The VM is stopped in the Nebius console after the runs.

## Comments
