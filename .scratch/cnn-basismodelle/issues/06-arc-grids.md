# 06 ARC grid tasks for the CNN models

Status: needs-info
Blocked by: 05
Spec: `../spec.md`

## What to do

- Decide with the owner where the ARC data come from. `src/arc_data.rs` reads converted
  re-arc files (Hodel 2024, arXiv:2404.07353, preprint), but the repository holds no such
  data and `tools/rearc_to_jsonl.py` is missing. Check the licence of the source before
  any download.
- Grid-to-grid tasks: input and output grids up to 30×30, 10 colours as channels, one
  colour class per cell. Score: exact match of the whole output grid, plus cell accuracy
  as a descriptive measure.
- Base models for this: U-Net and a fully convolutional network.

## Done when

- The data source, its licence and its checksum are recorded.
- A CNN genome trains on a set of ARC task families and is scored on held-out examples.

Cost: about 10 to 20 dollars.

## Comments
