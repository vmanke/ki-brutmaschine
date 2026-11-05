# 04 A library of named macros for the Architect

Status: needs-triage
Blocked by: `.scratch/llm-nutzt-cnn/issues/` 11 and 12 (Blueprint check, Architect rounds)
Spec: `../spec.md`, sections 2 and 5.2

## What to do

- After each Architect round, compress the accepted CNN genomes of the best members into macros:
  named sub-architectures that occur in several of them, chosen by the words they save, as Stitch
  does for programs (Bowers et al. 2023).
- Give each macro a name and a one-line description (Grand et al. 2024). List the macros in the
  next Architect request.
- Expand every macro of an answer into the JSON genome with a fixed expander before the
  Blueprint check of Q102, so that the check stays as strict as today. Log each answer before and
  after expansion.
- Retire a macro that no answer uses for two rounds. Cap the library. The control offers random
  macros of the same number and size.
- Measure the tokens per Architect answer and the validation score per answer, against Architect
  rounds without a library.

## Done when

- With the library off, Architect requests and answers are byte-identical to the design of
  `.scratch/llm-nutzt-cnn/`.
- Tests: the expander's output passes or fails the Blueprint check exactly as the hand-expanded
  genome does; a macro is coined only when it saves words; retirement and the cap hold.
- A recorded Architect Run replays with the library on.

## Comments
