# 02 Model genome, shape check and grid tasks

Status: done
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

**2026-10-31, Claude.** Built in four parts by subagents of one session: G0 (golden test
of the MLP mode), B2 (input shapes and tasks), B3 (CNN genome and shape check) and B4
(training, evaluation, model files and the run loop). Nothing was committed by them.

### What was built

- **Golden test (G0).** `tests/mlp_golden.rs`, 9 tests, with 10 fixtures written at
  bec9aa1, before this change: the files of a tiny run (archive, models, summary), the
  datasets, and the JSON of a genome, a model, the task specs, two configs, train stats
  and an eval request. The 10 fixture files are unchanged after all parts.
- **Input shapes and tasks (B2).** `TaskSpec::input` (`Flat`, `Image`, `Sequence`), left
  out of the JSON when flat. Five tasks with 1000/1000/2000 samples: `shapes`,
  `bars-and-stripes`, `blob-count`, `mirror-symmetry` (1 x 16 x 16) and `motif-1d`
  (4 x 64), each with exactly balanced classes.
- **CNN genome and shape check (B3).** `Genome::arch` (`Arch::Mlp` by default, left out
  of the JSON) or `Arch::Cnn(CnnGenome)`: stem, stages, head (`src/cnn/genome.rs`).
  `CnnGenome::plan` (`src/cnn/shape.rs`) infers the layers with their shapes, parameters
  and forward FLOPs; `Genome::validate` runs it and holds CNNs to `max_params` and
  `max_train_flops`. The block kinds of all 16 families are declared; only
  `Block::Plain` and the heads `GlobalAvgPool` and `Flatten` are built, the rest is
  refused as "not yet built".
- **The trained model (B4).** `model::Model` is `Mlp(nn::Mlp)` or
  `Cnn(cnn::model::CnnModel)`. `train`, `train_with_stop` and `score::evaluate` use it.
  An MLP file is the bare JSON of `nn::Mlp`, as before. A CNN file is the JSON of
  `CnnModel` and starts with `"kind":"cnn"`; it holds the architecture, the trained
  values in plan order and the running statistics of each batch norm.
- **CNN training (B4).** `src/cnn/net.rs` builds the network of a plan from `ops` only,
  layer by layer, so it has exactly the counted parameters. `src/cnn/train.rs` trains it
  in the loop of the MLP: `train::run_epochs` now drives both kinds through a small
  `Learner` trait, with the same shuffling, cosine decay to `LR_FLOOR`, FLOP and time
  budgets, kill switch and divergence rules. Initial values come from the ChaCha8 stream
  `derive_seed(seed, "init")` with He or Xavier scale times `init_scale` (W8). The
  optimizer is the MLP's own code (`OptimizerState::step_lists`), so SGD, momentum and
  Adam, the decay of weights only and the flushing of tiny moments are the same. Batch
  norm uses the batch statistics in training and the running ones in evaluation.
  `TrainStats::flops` counts 3 times the forward FLOPs of the plan per sample seen (D7).
  Classification uses the mean softmax cross-entropy, regression the mean squared error.
- **New operations (B4).** `ops` gained `broadcast_add` (the bias of a convolution),
  `max_pool1d` and `avg_pool1d` (candle has no 1-D pooling; built from shifted views like
  W3 and W4) and `mse`. Each has gradient cases (worst error 2.3e-10 against the limit
  1e-6), and the 1-D pools match candle's 2-D pools on a height of 1 exactly.
- **Hidden evaluation (B4).** `brut-eval` scores both kinds. A CNN file is read with
  unknown fields refused and at most `HARD_MAX_PARAMS` (2 000 000) values in all. Before
  any tensor is built, `check_cnn` runs the shape check against the task, the parameter
  cap, a new cap of 1e11 forward FLOPs for scoring the hidden split
  (`MAX_EVAL_FLOPS`; a CNN within the default limits needs less than 3 % of it by our
  count), and then checks every value list against its tensor. A refused file costs no
  query. Scoring runs in chunks sized by all layer outputs of a sample.
- **The run loop (B4).** No random genome is a CNN yet, so `Config::base_genomes` lets
  genomes join generation 0 before the random ones (generator `base`). Each must pass
  `Genome::validate`. The field is left out of the JSON when empty, and generation 0
  draws exactly as before then, so MLP configs and runs keep their bytes.

### Tests

- Learning, `src/cnn/train.rs`: `a_plain_cnn_learns_shapes`,
  `a_plain_cnn_learns_bars_and_stripes`, `a_plain_cnn_learns_blob_count`,
  `a_plain_cnn_learns_mirror_symmetry`, `a_plain_cnn_learns_the_motif` (see below).
- Training rules: `training_is_byte_equal_twice_and_in_pools_of_1_and_4_threads` (the bits
  of every step's loss and the model file, for a VGG-like 2-D CNN with batch norm and a
  1-D CNN with all three norms and both pools), `every_optimizer_reduces_the_loss`,
  `flops_follow_the_plan_and_match_the_estimate`,
  `budgets_and_the_kill_switch_stop_cnn_training_cleanly`,
  `divergence_is_reported_with_finite_loss`,
  `a_regression_target_trains_with_the_mean_squared_error`, and in `src/train.rs`
  `a_cnn_genome_trains_as_a_cnn`.
- Network: `the_tensors_hold_exactly_the_parameters_of_the_plan`,
  `initial_values_come_from_the_seed_with_he_or_xavier_scale`,
  `the_network_gives_one_row_of_outputs_per_sample`,
  `values_that_do_not_fit_the_plan_are_refused`,
  `plan_order_and_the_two_lists_convert_both_ways` (`src/cnn/net.rs`).
- Model files: `a_model_file_starts_with_its_kind_and_parses_back_equal`,
  `check_refuses_values_that_do_not_fit_the_plan` (13 kinds of damage),
  `unknown_fields_are_refused_also_inside_the_genome`,
  `the_reader_stops_at_the_value_cap_and_refuses_empty_lists`,
  `evaluation_uses_the_running_statistics_of_batch_norm` (`src/cnn/model.rs`);
  `an_mlp_file_is_the_bare_mlp_json`, `a_cnn_file_starts_with_its_kind_and_parses_back_equal`,
  `broken_files_give_generic_errors` (`src/model.rs`).
- Hostile files: `cnn_files_are_scored_and_bad_ones_cost_no_query` (`src/hidden_eval.rs`):
  a valid CNN is scored; 10 broken or hostile files are refused with `Invalid` and no
  panic, among them a file over the value cap, an architecture too costly to score, one
  with too many parameters, and a 1-D genome on an image; none costs a query.
- Runs, `tests/e2e.rs`: `a_cnn_base_genome_trains_is_saved_and_scored_in_a_run` (the CNN
  trains, its file gives back its archived validation metrics exactly, `brut-eval` scores
  it on the hidden split, and its children are CNNs with CNN files);
  `a_cnn_run_does_not_depend_on_the_thread_count` (1 and 2 worker threads give the same
  candidates and model files); `a_run_written_before_cnns_opens_scores_and_resumes` (a
  run that `brut run --no-hidden` wrote at bec9aa1, in `tests/fixtures/pre_cnn_run`: the
  archive verifies, its 10 model files give back their archived validation metrics, and
  it resumes for one more generation); `tiny_runs_on_the_image_and_sequence_tasks` (B2).
- Config: `base_genomes_are_left_out_when_empty_and_checked_when_given`.
- Shape check (B3): one test per rule in `src/cnn/shape.rs`, among them
  `rejects_too_many_parameters` and `rejects_too_many_flops`, plus hand counts and
  `no_genome_panics`; genome JSON and variation tests in `src/genome.rs` and
  `src/variation.rs`. Tasks (B2): `src/task.rs` has 20 tests, 14 more than at bec9aa1.

### Measurements

The learning tests train a plain CNN on the Train split (data seed 1) and score it on the
Valid split. 2-D: two 3x3 convolutions of 8 and 16 channels, each with ReLU and 2x2 max
pooling, flatten, dense 32. 1-D: two convolutions of 5 with 16 channels, global average
pooling, dense 32. Adam, learning rate 0.003, batch 64, 8 epochs. Times at
RAYON_NUM_THREADS=2 and nice 10, with the experiment of step 1 running; alone, then with
the five tests at once.

| Task | Classes | Chance | Parameters | Valid accuracy | Test bound | Time |
|---|---|---|---|---|---|---|
| shapes | 4 | 0.25 | 9604 | 0.678 | 0.5 | 4.2 s, 8.4 s |
| bars-and-stripes | 2 | 0.5 | 9538 | 1.000 | 0.9 | 4.0 s, 8.3 s |
| blob-count | 4 | 0.25 | 9604 | 0.677 | 0.45 | 4.0 s, 8.4 s |
| mirror-symmetry | 2 | 0.5 | 9538 | 0.771 | 0.65 | 4.1 s, 8.2 s |
| motif-1d | 2 | 0.5 | 2242 | 0.803 | 0.65 | 3.0 s, 3.1 s |

By our count the 2-D network spends 5.0e9 training FLOPs in 4.0 s, about 1.2 GFLOP/s,
close to the 1.6 GFLOP/s of the spike (ticket 01). In the run of
`a_cnn_base_genome_trains_is_saved_and_scored_in_a_run` (master seed 16) the base CNN of
598 parameters reached 1.0 on the Valid and on the hidden split of bars and stripes, its
two children 1.0 and 1.0 on Valid and 1.0 and 0.99 on the hidden split, and the random MLP
next to them 0.515 and 0.51.

### Reproducibility (D8)

CNN training gives the same bits at any thread count on one machine: the test above
compares the bits of every step's loss and of the model file across two runs in a row and
rayon pools of 1 and 4 threads, and a whole run with 1 and 2 worker threads. Other CPUs
may give other bits, because gemm picks its kernel (with or without fused multiply-add)
and its cache blocking at run time (ADR 0007). The tests compare within one machine only;
the bounds of the learning tests lie 0.10 to 0.23 below the accuracies reached here, as
room for other bits.

### Known limits

- Only plain stages and the heads `GlobalAvgPool` and `Flatten` are built; residual
  connections, squeeze-and-excitation, the other blocks and the per-cell head follow in
  tickets 03 and 04.
- CNNs enter a run only as `Config::base_genomes`. Random CNN genomes, the base models of
  the families, mutations of the architecture and the CLI are ticket 05; until then a CNN
  child changes only its training genes (B3).
- A batch norm needs at least 2 values per channel in training. A last batch of 1 sample
  at a 1x1 size fails the training of that candidate with an error, as in PyTorch. With
  1000 training samples and the allowed batch sizes the last batch holds at least 8
  samples, so the current tasks never get there.
- Each optimizer step copies the values between candle and Rust (a few times the
  parameter count), small against the convolutions.
- During scoring candle keeps every layer output of a chunk until its end, since the
  weights are candle variables; the chunks are therefore sized by the sum of all layer
  outputs per sample, not their maximum.
- A JSON file of an MLP with an extra `"kind"` key is now read as a CNN file and refused.
  The crate never writes such a file.
- `brut-eval` refuses a CNN whose scoring would cost more than 1e11 FLOPs, even when its
  run allowed it with larger limits.

### Done when

- `cargo test` passes, clippy is clean, and a CNN genome trains on each new task: yes (the
  five learning tests above; the commands and results are in the report of B4).
- An MLP run gives byte-equal logs before and after this change: yes,
  `a_tiny_run_writes_the_same_files` in `tests/mlp_golden.rs` passes with its fixtures
  unchanged.
- The shape check rejects every genome that breaks a shape or a limit, with a test per
  rule: yes, the `rejects_*` tests of `src/cnn/shape.rs` (B3).
