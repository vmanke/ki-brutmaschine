# 05 Lossless result formats for the CNN tool, chosen by tokens

Status: needs-triage
Blocked by: `.scratch/llm-nutzt-cnn/issues/` 15 and 16 (tool protocol, tool in brut-breed)
Spec: `../spec.md`, sections 2 and 5.3

## What to do

- Define three lossless text formats for the grid after `TOOL RESULT k`: the full grid as today,
  runs of equal cells, and the difference to the input grid. Each one decodes to the same grid.
- Choose the format per Run by measured efficiency: the tokens per result, at equal accuracy of
  the Workers that read it. Log the chosen format and the measurements.
- Keep `NONE` and the rest of the protocol of Q106 as they are. Old readers must still read the
  full grid.
- Later variant, not in this ticket: the Worker names the format in its request (Chen et al.
  2024).

## Done when

- With the full-grid format, every byte of the tool protocol stays as designed.
- Tests: every format decodes to the grid it encodes, for all grids of the ARC tasks in the repo;
  a malformed result is refused, never repaired.
- A recorded Run replays in each format.

## Comments
