# 06 ARC grid tasks for the CNN models

Status: resolved
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

### 2026-11-02, Claude: built

What is built (ADR 0010):

- Data. The four repositories at their pinned commits were fetched with git; the network
  allowed it. All 680 entries of the manifest matched the downloaded files. The manifest now
  also lists the 400 ARC-AGI-1 training files (Apache-2.0, commit 39903044) and the converted
  re-arc file `re-arc/grids.jsonl` (MIT, commit e5b7f1d0, SHA-256
  1990605d1027d33da70b4e71b6eecd58c4cf89ec9fe87032baf1885412aa1083): 1081 entries.
- Converter. `tools/rearc_to_jsonl.py` uses the Python standard library only and reads
  `re_arc.zip` as data (SHA-256 4a7c3094..., checked first); no code of re-arc is imported or
  run. `tools/rearc_sandbox.sh` ran it once in the plan's sandbox: setpriv to 65534 with
  no_new_privs, then user, network, PID, mount, IPC and UTS namespaces, a private tmpfs and
  prlimit. The zip goes in on standard input and the script on file descriptor 3, so no
  staging directory is needed. 400 task files read, 266 tasks with 15 616 pairs written
  (the first 60 that keep their size, at most 30 x 30), 140 924 examples skipped for a size
  change and 129 for grids above 30 cells. A second run gave the same bytes.
- Tasks. `src/arc_grid.rs`: one ARC grid task per training task, a fresh CNN per task from
  the genome's seed. Train: the task file's train pairs and the re-arc pairs but the last
  fifth; valid: the last fifth; test: the task file's test pairs. 259 of 400 tasks remain:
  138 change the grid size, 2 repeat evaluation pairs (40853293, caa06a1f), 1 has no re-arc
  pair of the same size. That gives 13 185 train, 3 095 valid and 264 test pairs.
- Grids. A 30 x 30 canvas, the 10 colours as channels, one colour class per cell, `NO_CELL`
  outside the grid. A classification with a grid of one channel per class classifies each
  cell (`TaskSpec::cell_classes`); `Metrics::primary` is the exact match and
  `Metrics::accuracy` the cell accuracy. `ops::cell_cross_entropy` is the masked loss.
- `TaskKind::ArcGrid` is not generated and not in `TaskKind::ALL`; `Config::validate` and
  `brut-eval` refuse it. `brut arc-grid` trains and scores the base models, or a genome of a
  file, next to the copy of the input, and writes a JSON report.
- Base models: a U-Net and a fully convolutional network (`arc_grid::base_models`).

Contamination audit on the real data (`real_arc_grid_tasks_touch_no_evaluation_pair`): no pair
of the 259 tasks shares a canonical hash with a pair of ARC-AGI-1 evaluation, ARC-AGI-2
evaluation or ConceptARC. The two tasks left out touch 070dd51e and 50a16a69 of ARC-AGI-1
evaluation. The existing `real_arc_data_counts` passes and now audits the training files: 2 of
400 evaluation tasks share a pair with ARC-AGI-1 training, as recorded.

Design runs on the first 8 tasks (default limits; dev profile, 2 threads; the copy of the
input: cells 0.767 valid and 0.752 test; no genome and not the copy got a grid right):

| Genome | Epochs | Rate | Cells valid | Cells test |
|---|---|---|---|---|
| U-Net, 3 x 3 stem of 8 | 7 | 0.01 | 0.398 | 0.546 |
| the same | 7 | 0.03 | 0.460 | 0.592 |
| the same | 7 | 0.1 | 0.443 | 0.577 |
| the same, no norm | 7 | 0.03 | 0.369 | 0.521 |
| the same, 1 x 1 stem | 9 | 0.01 | 0.383 | 0.303 |
| the same, 1 x 1 stem | 9 | 0.03 | 0.525 | 0.384 |
| **U-Net base model** | 8 | 0.03 | 0.490 | 0.614 |
| FCN, 3 x 3 stem and 3 stages of 12 | 7 | 0.01 | 0.509 | 0.543 |
| the same | 7 | 0.03 | 0.462 | 0.601 |
| the same | 7 | 0.1 | 0.330 | 0.521 |
| the same, no norm | 7 | 0.03 | 0.311 | 0.427 |
| 3 x 3 stem of 12, 2 residual basic blocks | 5 | 0.01 | 0.535 | 0.604 |
| 1 x 1 stem of 12, 2 residual basic blocks | 5 | 0.01 | 0.561 | 0.680 |
| 1 x 1 stem of 16, 2 stages of 16 | 5 | 0.01 | 0.565 | 0.700 |
| **FCN base model**, the same | 9 | 0.01 | 0.635 | 0.743 |
| the same | 9 | 0.03 | 0.647 | 0.727 |

Each trained 20 to 36 steps per task, in 8.8 to 19.8 s; none was stopped by the clock or
diverged. Ten times the FLOP limit (2e11, 300 s of
clock) on the first 4 tasks (copy 0.721 and 0.731): the 3 x 3 FCN of 12 trained 88 to 93 epochs
(352 to 372 steps, 208 to 227 s on a busy machine) and reached 0.691 and 0.753; the U-Net at a
rate of 0.01 trained 90 to 95 epochs and reached 0.730 and 0.765. No grid was right there either.

Full run, both base models on all 259 tasks (`brut arc-grid`, dev profile, `--threads 3`,
3552 s). The limits are the defaults except `max_candidate_seconds`, 120 s instead of 30 s,
since a step 2 build shared the 4 cores: 22 trainings of the FCN took more than 30 s of wall
clock there, which the default clock would have stopped. The epochs come from the FLOP limit
and the time model, not from the clock.

| Model | Exact valid | Exact test | Cells valid | Cells test | Steps per task |
|---|---|---|---|---|---|
| copy of the input | 0.000 | 0.000 | 0.766 | 0.752 | |
| u-net 811f0b55 | 0.002 | 0.000 | 0.571 | 0.628 | 32 on 257 tasks, 16 and 24 on 2 |
| fcn db327b07 | 0.003 | 0.004 | 0.673 | 0.713 | 36 on 257 tasks, 18 and 27 on 2 |

The FCN got 10 of the 3 095 valid grids and 1 of the 264 test grids right (task 6f8cd79b),
the U-Net 6 and none, the copy none. Valid grids were right in 0d3d703e, 42a50994, 5582e5ca,
d23f8c26 and d511f180 (U-Net) and in 0d3d703e, 42a50994, 67385a82, aedd82e4 and d23f8c26
(FCN). In cell accuracy both stay below the copy on average and beat it on 29 (U-Net, valid)
to 54 (FCN, test) of the 259 tasks. No task was refused, no training diverged, none was
stopped. A training took 4.2 to 27.3 s (U-Net) and 8.0 to 39.5 s (FCN) on the shared machine.

Tests: `tests/arc_grid.rs` (converter with `python3 -I` on a hand-made zip; `brut arc-grid` on
hand-made fixture tasks with both base models; equal reports at 1 and 2 threads; a changed
training file and a broken genome or limits file refused); unit tests in `arc_grid`,
`arc_data`, `task`, `score`, `train`, `genome`, `config`, `hidden_eval`, `cnn::ops` (gradient
cases of `cell_cross_entropy`) and `cnn::train`; two real-data tests that run on request with
`BRUT_ARC_TEST_ROOT`. Not built: breeding on ARC (`brut run` on these tasks), tasks that change
the grid size, a NOTICE file (no data are passed on).

