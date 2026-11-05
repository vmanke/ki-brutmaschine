# 03 Breeder rules for CNN Workers

Status: needs-triage (waits for the owner's approval of the design)
Type: task
Blocked by: 02; outside: cnn-basismodelle 07 (`ModelAdapter`) for the gate part only
Spec: ../spec.md (sections 4.2.3, 4.2.5, 4.2.6; decisions Q86 to Q90, Q94)
Pull request: PR 1 (F1)

## What to do

- `src/cascade/variation.rs`, skip rule (Q87):
  - ThinkUp, ThinkDown, TemperatureShift, SelfCheckFlip, AnswerTokensShift and the four
    section operators draw only from Workers whose tag is not a `cnn:` tag.
  - WorkerAdd and StageAdd clone only such Workers.
  - ModelSwitch, WorkerDrop, StageDrop, stage crossover and section crossover stay as
    they are.
  - Without a `cnn:` tag the slot lists, and so the random draws, are unchanged.
- `src/cascade/domain.rs`, admission rule (Q88): no two Workers of one `cnn:` tag in one
  Stage. It joins the admission rule of every gene pool that may meet `cnn:` tags (grid
  and grid-mix); for Cascades without `cnn:` tags it always admits.
- `src/cascade/arena.rs`, `BreedConfig::check` (Q89): refuse a `cnn:` tag as the first
  allow-list entry; refuse `cnn:` tags in circuit runs.
- `src/cascade/variation.rs`, brief (Q90): a static `GRID_CNN_BRIEF`, `GRID_BRIEF` plus
  one fixed paragraph. It says what a `cnn:` Worker does (it trains on the examples and
  answers with a grid; its sections and Traits are ignored but still required), and
  nothing about where to put one. The binary picks it when a `cnn:` tag is listed.
  `GRID_BRIEF` stays byte for byte (`src/cascade/variation.rs:357-365`).
- The pure `without_cnn(&BreedConfig) -> BreedConfig`: it drops every `cnn:` tag from the
  allow-list and the tool, so arm L follows from arm C by rule.
- Gate part (needs cnn-basismodelle 07): `CascadeAdapter` (`src/hitl.rs:150-158`) and
  `brut-ingest --cascades` add the tags of the genomes merged in `candidates/models/` to
  the pool allow-list. They read those tags from the base revision only. CI reads the
  pull request's own files as `pr/` (`.github/workflows/scope-fence.yml:47-54`), so the
  workflow passes `base/candidates/models` to `brut-ingest --cascades`.
  `brut-breed --promote` writes Cascades with CNN Workers; the gate refuses one whose
  genome is not merged (Q94).

## Tests (write them first)

- `edit_and_clone_operators_never_touch_a_cnn_worker`: 10 000 draws.
- `random_draws_are_unchanged_without_cnn_tags`: the golden fixtures
  (`tests/universe.rs:182-236`).
- `model_switch_moves_workers_to_and_from_cnn_tags`.
- `stage_crossover_can_put_a_cnn_stage_first`.
- `admission_refuses_two_workers_of_one_cnn_tag_in_a_stage`, for random children and for
  Mutator answers.
- `check_refuses_a_cnn_tag_first_and_cnn_tags_in_circuit_runs`.
- `cnn_runs_use_the_cnn_brief_and_grid_runs_keep_their_keys`: `requests.txt` is
  unchanged.
- `cnn_brief_request_keys_are_pinned`: a new fixture.
- `without_cnn_keeps_seeds_panels_and_every_other_setting`: only the allow-list changes.
- Gate part: `a_cascade_with_an_unmerged_cnn_genome_is_not_admitted` and
  `a_cascade_with_a_merged_cnn_genome_is_admitted`;
  `a_mutation_pr_that_adds_a_genome_and_a_cascade_using_it_is_refused`;
  `the_scope_fence_still_rejects_paths_outside_candidates`.

## Acceptance criteria

- All tests pass. clippy is clean.
- The golden tests pass unchanged.

## Comments
