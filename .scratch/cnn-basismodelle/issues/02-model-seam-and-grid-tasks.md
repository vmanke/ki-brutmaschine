# 02 Model genome, shape check and grid tasks

Status: ready-for-agent
Blocked by: 01
Spec: `../spec.md`

## What to do

- A model genome that is either the existing MLP genome or a CNN genome. Old archives,
  checkpoints and logs stay readable; a test loads one written before this change.
- The CNN genome of the spec (stem, stages, head), with shape inference, parameter count
  and training FLOPs. Training and evaluation through candle behind the same interface
  the MLP uses (`src/train.rs`), so that the evolution loop, scoring, archive and hidden
  evaluation work unchanged.
- `TaskSpec` gets an input shape (channels, height, width; or channels, length). The
  existing tasks keep a flat shape.
- Generated tasks with train, validation and hidden splits from independent seeds:
  shapes (16×16), bars and stripes, blob count (1 to 4), mirror symmetry, and a 1-D motif
  task. Each generator is tested for determinism and class balance.

## Done when

- `cargo test` passes, clippy is clean, and a CNN genome trains on each new task.
- An MLP run gives byte-equal logs before and after this change (golden test).
- The shape check rejects every genome that breaks a shape or a limit, with a test per
  rule.

Cost: about 10 to 20 dollars.

## Comments
