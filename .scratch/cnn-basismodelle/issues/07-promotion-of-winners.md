# 07 Promotion of winners: bred models become base models

Status: ready-for-agent
Blocked by: 05
Spec: `../spec.md`

The owner asked on 2026-10-30 that winners can be fed back upward, that is, promoted.
The project already promotes the two best finalist Cascades of a Run into
`candidates/cascades/` through a draft pull request that a human merges (ADR 0004, ADR
0005; CONTEXT.md, "Promotion"). Models of the MLP mode, MLPs and CNNs, get the same path.

## What to do

- A `ModelAdapter` that implements the existing `CandidateAdapter` interface of
  `src/hitl.rs` for model genomes: parse with unknown fields rejected, validate with the
  shape check and the current Limits, fingerprint by the SHA-256 of canonical JSON.
  Promoted models live in `candidates/models/<content-id>.json`, under `candidates/`, so
  the existing scope fence covers them.
- Promotion of a finished run: the two best models by their score on the hidden split,
  which the run never saw while breeding. A model is promoted only if it beats the best
  base model of the same run on that split. Each file records its provenance: the run,
  the task, the seed, the commit, the scores on validation and hidden splits, and the
  lineage back to its base model. A model equal by content to one in the pool adds
  nothing.
- The promotion is a draft pull request that changes files under `candidates/models/`
  only. A human merges it; the run never merges anything (ADR 0004).
- Later runs: the first generation holds the merged promoted models next to the base
  models of the 16 families and the random genomes. A promoted model that fails the
  current shape check or Limits is skipped, with a log line.
- Docs: the "Promotion" entry of CONTEXT.md covers models too; README (German and
  English) and ADR 0008 describe the path.

## Done when

- Tests: the two best by hidden score are chosen; a model that does not beat the best
  base model is not promoted; a duplicate adds nothing; unknown JSON fields are refused;
  a promoted model loads into the first generation of a new run, and an invalid one is
  skipped; a run with promoted models is reproducible byte for byte from its seed.

Cost: about 10 to 20 dollars.

## Comments
