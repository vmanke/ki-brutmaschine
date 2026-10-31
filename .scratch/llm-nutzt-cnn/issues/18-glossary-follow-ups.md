# 18 Glossary follow-ups

Status: needs-triage (waits for the owner's approval of the design)
Type: task
Blocked by: part A the merge of PR 1, part B the merge of PR 2, part C the merge of PR 3
Spec: ../spec.md (decision Q115)
Pull request: G1, G2 and G3, one small pull request per part (Q53,
`.scratch/universum-erfinden/spec.md:46`)

The planned terms are already in CONTEXT.md, subsection "LLMs using CNNs (planned)". The
design changed no existing entry. Each part updates the existing entries once its feature
is built, in a pull request of its own, never inside a code pull request.

## What to do

- Part A, after PR 1:
  - **Worker**: the model is a language model or a CNN (a CNN
    Worker).
  - **Promotion**: a Cascade with a CNN Worker is promoted only if
    its genome is merged in `candidates/models/`.
  - **Kill switch**: a CNN training in flight can delay the stop up
    to its time ceiling.
  - **Domain**: grid-mix is a Domain.
  - **Arm**: arms L, C and Z of F1. The letters avoid S and N, which step 2 adds to the
    same entry (`.scratch/seele-und-kappa/spec.md:90-92` on main).
  - Move the built terms of F1 (CNN Worker, Per-task training, Call record, Neighbour
    rule, Silent CNN, Knock-out) under the heading "LLMs using CNNs"; the others stay
    under "(planned)".
- Part B, after PR 2:
  - **Control run**: or without the Architect.
  - **Lineage archive**: it also holds the Architect's exchanges.
- Part C, after PR 3:
  - **Trait**: the Tool budget is a Trait.
  - **Arm**: arms T and U of F3.
  - Drop "(planned)" from the heading of the subsection.
- Before each part merges, compare with the terms that cnn-basismodelle 05 and step 2 add,
  so no term repeats or contradicts another.

## Tests (write them first)

None; the check is a review: every term has one definition and an Avoid line, and the
tickets and reports use exactly these words.

## Acceptance criteria

- Each part is its own small pull request that the owner merges.
- No term repeats or contradicts an existing one.

## Comments
