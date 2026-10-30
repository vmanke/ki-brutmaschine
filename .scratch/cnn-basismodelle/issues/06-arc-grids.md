# 06 ARC grid tasks for the CNN models

Status: ready-for-agent
Blocked by: 05
Spec: `../spec.md`

## What to do

- Data: the sources, commits and licences the project already pins
  (docs/PLAN_GENERALISIERUNG.md, sources; docs/SICHERHEIT_UND_RECHT.md, data licences):
  fchollet/ARC-AGI at 39903044 (Apache-2.0), arcprize/ARC-AGI-2 at f3283f72
  (Apache-2.0), victorvikram/ConceptARC at 0e67da6a (MIT) and michaelhodel/re-arc at
  e5b7f1d0 (MIT; Hodel 2024, arXiv:2404.07353, preprint). BARC is never used (no licence).
  `src/arc_data.rs` loads them with a SHA-256 manifest (`data/manifest.example.json`) and
  enforces the contamination rules; the data are not in the repository.
- Training data come from the ARC-AGI-1 training tasks and their re-arc generations,
  since each ARC task alone has only 3 to 5 example pairs. re-arc covers only ARC-AGI-1
  training tasks, so both evaluation sets stay clean (docs/PLAN_GENERALISIERUNG.md).
  The evaluation sets are held out and never bred on.
- `tools/rearc_to_jsonl.py` is missing. Rewrite it as the plan prescribes: Python
  standard library only, run once, in the tested sandbox recipe; no foreign Python code
  is imported or run. Pin the output's SHA-256 in the manifest.
- The download needs network access to these GitHub repositories at the pinned commits.
  If this session's network policy blocks it, say so, and the owner allows the hosts.
- If data are passed on, the attribution goes into a NOTICE file.
- Grid-to-grid tasks: input and output grids up to 30×30, 10 colours as channels, one
  colour class per cell. Score: exact match of the whole output grid, plus cell accuracy
  as a descriptive measure.
- Base models for this: U-Net and a fully convolutional network.

## Done when

- Every data file used is in the manifest with its source, commit, licence and SHA-256.
- A CNN genome trains on a set of ARC task families and is scored on held-out examples.

Cost: about 10 to 20 dollars.

## Comments
