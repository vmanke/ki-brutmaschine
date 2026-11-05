# 04 Per-task training of a CNN

Status: needs-triage (waits for the owner's approval of the design)
Type: task
Blocked by: 02; outside: the merge of cnn-basismodelle 04 (`Head::PerCell`; its work in
progress on main already builds the head, `src/cnn/shape.rs:1138`), and cnn-basismodelle 06
for the per-cell classification objective only (today the loss takes one label per sample,
`src/cnn/train.rs:115-137`)
Spec: ../spec.md (sections 4.2.2, 4.2.7, 6, 7; decisions Q78, Q79, Q81, Q82, Q84, Q85)
Pull request: PR 1 (F1)

## What to do

- New `src/cnn/worker.rs`, registered in `src/cnn/mod.rs`: `CnnPredictor` implements
  `GridPredictor` (ticket 02) for one genome.
  1. Refuse a task whose examples change the grid size, that has more than 10 examples,
     or whose grids break `is_valid_grid`.
  2. Pad every grid of the task to its largest height and width, over the examples and
     the test input. Build 10 one-hot colour channels and one inside channel. Each inside
     cell carries its output colour as its label.
  3. Build the task spec with the grid output of cnn-basismodelle 04 and the per-cell
     objective of cnn-basismodelle 06; `n_train` is the number of examples.
  4. Compute e_task: the genome's `epochs`, or the largest smaller count that passes
     `Genome::validate` against that spec (`src/genome.rs:229-323`) under
     `Limits::default()`. It comes from the plan, before any tensor exists. A genome that
     fails even at `MIN_EPOCHS` is refused.
  5. Train for e_task epochs with the new crate-private `train::train_counted`.
  6. Predict one row through `CnnModel::evaluator` and `Evaluator::outputs`
     (`src/cnn/model.rs:181-198, 244-263`). Take the best colour per inside cell, ties to
     the lower colour, and crop to the test input.
  7. A diverged training is a refusal. A candle or shape error is a backend error.
- `src/train.rs`: `train_counted` runs `run_epochs` (`src/train.rs:145-241`) with no clock
  and no stop signal. `run_epochs` takes an optional clock; the existing entries pass one,
  so their behaviour and bits stay. On the counted path `TrainStats.wall_seconds` is 0.
- Padded cells carry no loss. If cnn-basismodelle 06 brings no mask for them, add it in
  `src/cnn/` here, after 06 has merged.
- The memo (Q85): per `CnnPredictor`, a map from the SHA-256 of (examples, padded size)
  to the trained CNN. A request that misses trains its own copy and never waits for
  another request. A wait inside a rayon job could deadlock. candle and gemm run nested
  parallel loops on the caller's pool (`candle-core/src/cpu_backend/mod.rs:1482-1487` and
  `cpu_backend/conv2d.rs:91` at the pinned commit;
  `gemm-common-0.19.0/src/gemm.rs:163-170`). A rayon thread that waits for such a loop
  runs other jobs meanwhile, such as a judgment that needs the key it trains. Equal bits
  (ADR 0007) make every copy the same. The first copy finished is stored, and later
  requests read it. The lock around the map is never held during a training. The memo is
  never written to disk. Counters of trainings and of extra copies are readable for the
  binary.
- A reference genome fixture `tests/fixtures/cnn/<id>.json`: a small per-cell network
  that passes the checks on every padded size from 3 × 3 to 6 × 6 with 3 examples.

## Tests (write them first)

- `the_training_set_is_exactly_the_worked_examples`.
- `padded_cells_carry_no_loss`: changing their labels changes neither the loss nor the
  gradients.
- `per_task_training_is_byte_equal_twice_and_in_rayon_pools_of_1_and_4_threads`.
- `per_task_training_is_byte_equal_at_1_and_3_matrix_threads`: child processes with
  `RAYON_NUM_THREADS` at 1 and 3 (ADR 0007:115-121). It lives under `tests/`, so the
  source scan below covers all of `worker.rs`.
- `the_answer_ignores_seed_temperature_and_previous_answers`.
- `the_hidden_truth_cannot_change_the_answer`.
- `a_shape_changing_task_is_refused_before_training`.
- `a_genome_too_big_for_the_padded_task_is_refused_before_any_tensor`.
- `a_diverging_training_is_a_refusal`.
- `a_cached_answer_equals_a_fresh_one`.
- `parallel_judgments_on_one_task_finish`: 16 parallel CNN judgments on one 30 × 30
  task, a 16-channel genome, an empty memo and a 120 s watchdog. Every verdict equals the
  verdict of a single call.
- `a_cold_memo_peer_round_finishes`: a resumed run whose peer judges start their last
  Stage with a CNN, in a rayon pool of 4 threads, with a watchdog.
- `epochs_shrink_only_on_tasks_too_large_for_the_genome`: e_task equals `epochs` on a
  6 × 6 task and is smaller on a 30 × 30 task with 10 examples.
- `the_reference_genome_learns_colour_maps_from_three_examples`: on N colour-only
  generated puzzles it solves at least a bound set 0.1 below the first measured share,
  then pinned.
- `existing_training_keeps_its_bits`: `tests/mlp_golden.rs` and the CNN byte tests pass
  unchanged.
- `worker_rs_names_no_file_network_process_or_clock_api`: a source scan.

## Acceptance criteria

- All tests pass. clippy is clean.
- Comments record the measured training seconds per task of the reference genome on this
  machine, next to the time model's estimate and e_task.

## Comments
