# 11 Architect requests and the Blueprint check

Status: needs-triage (waits for the owner's approval of the design)
Type: task
Blocked by: the merge of PR 1 (the owner's order); outside: cnn-basismodelle 05 (CNN
mutations and the model-kind CLI touch the same loop) and 07 (its strict bare-genome
entry point, besides the parse of a promotion file)
Spec: ../spec.md (sections 4.3, 5, 7; decisions Q100 to Q103)
Pull request: PR 2 (F2)

## What to do

- Move `Mutator`, `request_key`, `AnswerBook` and `LiveMutator` from
  `src/cascade/arena.rs:436-499` to `src/backend.rs`. `arena.rs` re-exports them, so
  `src/bin/brut-breed.rs:66-69`, `tests/universe.rs:23-26` and
  `backend-ollama/src/lib.rs:27` keep working.
- New pure module `src/architect.rs`:
  - `architect_mutate_request` and `architect_merge_request`. Settings: 4096 tokens,
    temperature 0.7, no thinking (as `src/cascade/variation.rs:1042-1052`).
  - The prompt: a fixed system text; the task in words (kind, input shape, objective,
    sample counts; `src/task.rs:258-278`); the limits in words (`src/config.rs:32-52`);
    the gene ranges and CNN parts, generated from the code's constants and from what the
    shape check accepts; each parent as pretty JSON with its public validation score; the
    task and the rules.
  - `parse_blueprint`: take the first balanced JSON object, as
    `src/cascade/variation.rs:591-613` does; parse it with the bare-genome entry point of
    cnn-basismodelle 07, never with the promotion-file parse, so unknown fields are
    refused at every level; require a CNN genome; set the seed gene from the run and log
    the model's value; run `Genome::validate`; repair nothing.
- Extend the purity scan of `tests/cascade.rs:14-35` to `src/architect.rs`. Today it
  covers `src/cascade/` only, and this module parses untrusted text.
- Cap answer files at 64 KiB in both loops. The Cascade side reads them whole today
  (`src/bin/brut-breed.rs:1149`). An oversized file stops the run with an error that names
  the file, before any call.

## Tests (write them first)

- `moving_the_mutator_types_keeps_every_key`: `tests/universe.rs` fixtures unchanged.
- `architect_request_keys_are_pinned`: a new fixture.
- `architect_requests_hold_no_data_and_no_hidden_score`.
- `a_blueprint_is_the_first_json_object_and_nothing_else`.
- `unknown_fields_are_refused_at_every_level`, the top level included.
- `an_mlp_genome_is_refused`.
- `the_run_sets_the_blueprint_seed`.
- `an_invalid_blueprint_is_refused_with_its_reason_and_never_repaired`: one case per rule
  of `Genome::validate`.
- `the_brief_lists_only_parts_the_shape_check_accepts`.
- `answer_files_over_64_kib_are_refused_in_both_loops`.
- `the_purity_scan_covers_architect_rs`.

## Acceptance criteria

- All tests pass. clippy is clean.
- The golden tests and `tests/mlp_golden.rs` pass unchanged.

## Comments
