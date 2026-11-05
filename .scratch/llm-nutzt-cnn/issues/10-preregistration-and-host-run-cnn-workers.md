# 10 Pre-registration and host run: do CNN Workers help?

Status: needs-triage (waits for the owner's approval of the design)
Type: research
Blocked by: 07, 08, 09; outside: the adapter follow-ups of spec 10.3, which PR #33 left
open: a context setting for ARC prompts, and thinking caps if the runs use gpt-oss:20b or
qwen3:8b. The adapter and the seam of O5 exist (`brut-breed-ollama`, `EXTRA_BACKENDS`).
For a confirmatory claim on ARC-AGI-1 evaluation, kaskaden-zucht 02 (the sealed endpoint)
Spec: ../spec.md (sections 5, 8, 9; decisions Q93, Q96, Q97, Q112, Q114)
Pull request: P1 (the pre-registration, frozen at the commit its merge creates), then a
report

## What to do

- Pilot first: 2 seeds × arms C and L, 20 generations, on grid-mix and grid, then on ARC
  training tasks (ticket 09), with real models through `brut-breed-ollama`: gpt-oss:20b
  and qwen3:8b once the adapter has their thinking caps, models of its list until then
  (spec 10.3). It measures M1 to M10 and sets:
  - the fitness cost per call of the `cnn:` tag (Q93);
  - the generations, `max_calls` and `--max-seconds`;
  - the most pairs of the sequential rule.
- `docs/PREREG_LLM_CNN.md`, section F1, frozen at merge (pattern of
  `docs/PREREG_STEP1_CIRCUITS.md:3-6`). It fixes:
  - the arms C, L and Z, the seeds and their order;
  - the Domain and its panel sizes; for ARC also the manifest SHA-256 and `select_share`;
  - the `cnn:` tag, and `llm_every` 0, so no LLM round runs (spec 9.3);
  - the budgets and the models with their digests;
  - the commit the runs are built from;
  - the primary outcome (select-panel solved rate of the best finalist per run, the best
    finalist being the first bred `Finalist` in search rank);
  - the sequential rule (thresholds 100 and 1/20) and the Knock-outs; a proof takes at
    least 12 pairs (ADR 0006:175);
  - the void conditions (an integrity error; logs that differ between one and three
    threads; a failed data-hygiene audit; a committed error of a real model; Call records
    that are lost or fail the replay below);
  - the analysis program `examples/llm_cnn_analysis.rs`.
- Counted runs on the host. Their Call-record tarballs go to the lasting store of O13.
  Before the report merges, `--replay` on another machine, from the downloaded tarball,
  must give each run's `Born`, `Compared` and `LlmExchange` events. Comments record this
  check.
- Then a German report with an English version, in `docs/LLM_NUTZT_CNN.md` and
  `docs/LLM_USES_CNN.md`.
- If the counted Domain is ARC, the claim is about select panels of ARC-AGI-1 training
  tasks. Every number about ARC-AGI-1 evaluation stays descriptive until kaskaden-zucht 02
  exists (Q97).

## Tests (write them first)

- `the_analysis_reproduces_its_numbers_from_the_run_files`.
- `the_control_arm_is_without_cnn_of_the_cnn_arm`.
- `knock_outs_keep_the_call_seeds`.

## Acceptance criteria

- The owner has approved and merged the pre-registration.
- The report, the run manifests and the Call-record tarball hashes exist. The tarballs
  are in the lasting store, and each passed the replay on another machine.
- The VM is stopped in the Nebius console after the runs.

## Comments
