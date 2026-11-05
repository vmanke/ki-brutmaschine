# 13 Architect controls and container evidence

Status: needs-triage (waits for the owner's approval of the design)
Type: task
Blocked by: 12
Spec: ../spec.md (sections 4.3, 9.1; decisions Q102, Q104, Q112)
Pull request: PR 2 (F2)

## What to do

- Two scripted Architects in tests:
  - one that returns a known better genome for the run's task;
  - one that returns garbage (prose, broken JSON, unknown fields, an MLP genome, a genome
    over the limits).
- Runs F2-scripted and F2-session of section 9.1, on one generated image task of the MLP mode:
  - F2-scripted: scripted Architect against a Control run (CONTEXT.md, "Control run")
    with the same settings, secret, seeds and FLOP budget;
  - F2-session: Architect runs whose answers a Claude session writes through files, against
    Control runs. Descriptive.
- Each runs one pilot seed first. Its pass criteria and its counted seeds are then
  committed and pushed to the draft pull request, before the counted seeds run. The
  report cites that commit.
- Report: section F2 of `docs/LLM_NUTZT_CNN.md` (German, `docs/SCHREIBSTIL.md`) and
  `docs/LLM_USES_CNN.md`, with the measured numbers, the acceptance rate of Blueprints
  and the refusal reasons (M7).

## Tests (write them first)

- `a_scripted_architect_with_a_known_better_genome_is_taken_up`: its child enters the
  archive and wins on validation.
- `an_architect_that_writes_garbage_leaves_only_records`: every answer is refused with a
  reason; the run's genomes equal those of a run with pause off and no answers.

## Acceptance criteria

- Both tests pass.
- The report section exists with measured numbers.

## Comments
