# 04 CNN families, part B

Status: resolved
Blocked by: 02
Spec: `../spec.md`

## What to do

Blocks and one scaled-down base model each, within the limits:

- DenseNet (dense block, transition layer);
- Inception (parallel 1×1, 3×3, 5×5 and pooling branches, concatenated);
- SqueezeNet (Fire module);
- ShuffleNet (group convolution, channel shuffle);
- ConvNeXt (depthwise 7×7, LayerNorm, inverted MLP, GELU);
- TCN (causal dilated 1-D convolution with residuals), for the 1-D task;
- U-Net (encoder, decoder, concatenated skips; transposed convolution or upsampling),
  for grid-to-grid outputs.

## Done when

- The same tests as ticket 03 for each block and base model.
- The TCN learns the 1-D motif task above chance; U-Net maps a generated grid to its
  mirror image above chance (a stand-in until ticket 06).

Cost: about 15 to 30 dollars.

## Comments

**2026-10-31, Claude (subagent build-t04).** Built in the worktree of branch
`claude/cnn-basismodelle-teil2`, on top of ticket 03, nothing committed.

### What was built

- **Blocks** (`src/cnn/shape.rs`, `src/cnn/net.rs`). `Inception`, `Dense`, `Transition`,
  `Fire`, `Shuffle`, `ConvNext`, `Tcn` and `Decoder` now pass the shape check, and so does
  the head `PerCell`. The module doc of `shape` lists the convolutions of each block. An
  Inception block has four branches of width / 4 channels: a 1x1; a 1x1 reduction and a
  k x k; a 1x1 reduction and a (k + 2) x (k + 2); a k x k max pooling that keeps the size,
  then a 1x1. A dense layer is one k x k convolution to its growth rate, joined to its
  input. A transition is a 1x1 convolution with the stage's downsampling. A Fire module
  squeezes to width / 4 and expands by a 1x1 and a k x k of width / 2 each. A ShuffleNet
  unit has a 1x1 group convolution to width / 4, the channel shuffle, a depthwise k x k
  without activation and a 1x1 group convolution. A ConvNeXt block has a depthwise k x k
  with its norm and an inverted MLP of 4 times its input channels, and no activation at the
  end. A TCN block has two causal convolutions along a sequence. Both options of ticket 03
  work on every block.
- **Branches.** The plan gains `Layer::Concat`. `Fork` and `Branch` mark parallel branches
  as for a skip connection, and `Concat` joins them along the channels. The last
  convolution of a branch has its norm; the activation follows the join.
- **U-Net.** Every unit with a stride above 1, except a decoder, makes a skip point: its
  input when its convolution moves by the stride, else its map before the pooling. Each
  decoder joins the latest skip point that no decoder has joined yet, as a closing bracket
  matches the latest open one. It upsamples by its stride, by a transposed convolution with
  `Downsample::Conv` or by repeating cells with a pooling kind. Then it joins the skip map
  along the channels and runs its k x k convolution. No gene was added, and a genome without
  decoders keeps its plan. The per-cell head applies the dense layers and the output layer
  as 1x1 convolutions with a bias, so every cell gets the same small network.
- **New operations** (`src/cnn/ops.rs`), each with gradient cases in f64:
  `conv_transpose1d`, `upsample_nearest1d`, `causal_conv1d` with a stride, `max_pool1d`
  with padding, and `channel_shuffle` along a sequence. New cases also check `cat` where an
  input feeds both parts of a join, as in a dense layer and a U-Net skip. The plan gains the
  layers `CausalConv`, `ConvTranspose`, `Upsample`, `Shuffle` and `MaxPoolSame`.
- **A ninth candle defect, D9.** candle's `conv_transpose1d` has no backward pass, and its
  forward pass is wrong when the batch and the input channels both exceed 1. Replacement W9
  runs `conv_transpose2d` on a height of 1. The tests require candle's own operation to fail
  both checks, and the source scan refuses a call of it. ADR 0007 has a dated addendum. The
  library paragraph of the spec still names D1 to D8.
- **Grid outputs and the mirror grid task** (`src/task.rs`). `OutputShape` is `Flat`, the
  default, or `Grid` with channels, height and width. `TaskSpec::output` has a serde
  default and is left out of the JSON while it is flat, so the specs of all older tasks keep
  their bytes. The task `mirror-grid` gives 8 x 8 grids of random cells with pixel noise.
  Its targets are the clean grid mirrored about its vertical axis, one value per cell. It
  is a regression of 64 values, so its score is R^2; a model that ignores its input reaches
  0 at best. An MLP reads and writes the grids flattened. The task stands in for the ARC
  grids of ticket 06. The CLI knows its name, and `brut help` lists it.
- **FLOPs and time model.** A transposed convolution counts its kernel values once per input
  position. The time model prices the new layers with the existing constants: a causal
  convolution as a convolution, a transposed convolution as the convolution of its backward
  pass, max pooling that keeps the size as pooling, and an upsampling, a shuffle or a join by
  its values and one call. No price changed, so every plan without the new layers keeps its
  estimate to the bit.
- **Running statistics.** `MAX_RUNNING_VALUES` stays at 86 528. Every new block holds at most
  3 times `MAX_CHANNELS` channels of batch norm per stage, the inverted residual 14 times.
- **Formats.** The genome format did not change: it had declared every new block and head
  already. `TaskSpec` gains the one field `output`, with a serde default that is skipped at
  its default. Old archives, specs and model files read as before.
- **Base models** (`src/cnn/families.rs`). `Family::base_model` now gives a genome for all 16
  families, `Family::task` the task each is built for, and `base_models()` lists all 16 for
  ticket 05. The TCN reads the sequences of `motif-1d`, and the U-Net maps the grids of
  `mirror-grid`. The other five are 2-D classifiers for the 16 x 16 image tasks.

### Tests

- **Hand counts** (`src/cnn/shape.rs`): `dense_layers_and_a_transition_count_by_hand`,
  `inception_blocks_count_by_hand`, `fire_modules_count_by_hand`,
  `shufflenet_units_count_by_hand`, `convnext_blocks_count_by_hand`,
  `tcn_blocks_count_by_hand`, `a_u_net_counts_by_hand` and
  `a_1d_genome_runs_the_blocks_of_ticket_04_along_the_sequence` compare each block's layers,
  with their output shapes and parameters, and the plan's parameters and forward FLOPs with
  a count by hand. `a_u_net_counts_by_hand` also counts the time model of a transposed
  convolution, an upsampling and the joins, and
  `the_time_model_counts_the_layers_of_ticket_04_by_hand` counts every new layer kind.
- **Rules**: `rejects_the_rules_of_the_blocks_of_ticket_04` (widths that 4 does not divide,
  a second kernel above 11, groups that do not divide a ShuffleNet unit, a dense layer that
  moves by a stride or grows above 256 channels, a TCN block on an image, a decoder without
  a stride, without a skip point, or with a size that misses it),
  `the_per_cell_head_needs_a_grid_task_and_a_map_of_its_size`,
  `every_block_kind_and_head_is_built` and `every_block_takes_both_options`.
  `no_genome_panics` draws every block and the U-Net form: 823 valid genomes, 633 of them
  with a block or an option, and 17 177 invalid ones; none panics.
- **Gradients and the loss on a toy batch** (`src/cnn/train.rs`): one test per new genome of
  `testing::block_genomes`. Each checks that every gradient is finite and reaches every
  tensor, then requires the loss below 0.7 of its start after 50 Adam steps on 32 samples.
  Measured: dense layers and a transition 1.657 to 0.704, Inception blocks 1.481 to 0.008,
  Fire modules 1.497 to 0.099, ShuffleNet units 1.478 to 0.071, ConvNeXt blocks 2.069 to
  0.006, TCN blocks 0.798 to 0.001, the U-Net 3.617 to 0.204 (mean squared error), and the
  1-D blocks 2.024 to 0.003. `every_block_genome_has_a_toy_batch_test` keeps the list
  complete and checks that it holds every block kind, both options and the per-cell head.
- **Wiring**: `loss_and_grad_matches_finite_differences` has three more genomes, at a step
  of 1e-3. The 2-D blocks with a decoder: 195 values, worst error 0.55 of the bound. TCN
  blocks with a 1-D decoder: 84 values, 0.06. A U-Net with nearest upsampling and the
  per-cell head on the mirror grid: 39 values, 0.05.
- **Reproducibility**: the thread tests have three more genomes: the 2-D blocks of this
  ticket with a decoder, TCN and other 1-D blocks with a transposed convolution, and the
  U-Net block genome on the mirror grid. Their losses and model files are byte-equal in
  rayon pools of 1 and 4 threads and in child processes with `RAYON_NUM_THREADS` at 1 and 3.
- **Operations** (`src/cnn/ops.rs`): 11 new gradient cases, the worst at a relative error of
  4.6e-10 against the bound of 1e-6; all 82 cases pass, the worst at 3.5e-9.
  `replacements_match_candle_forward` compares W9 and the 1-D upsampling with candle where
  candle is right, `candle_defects_fail_the_gradient_check` covers D9, and
  `causal_conv1d_reads_no_later_input` covers the stride.
- **Files and scoring**: `models_of_every_block_parse_back_check_and_give_outputs` covers the
  new genomes, and `no_other_block_holds_more_norms_than_an_inverted_residual`
  (`src/cnn/model.rs`) checks the cap. `cnn_files_are_scored_and_bad_ones_cost_no_query`
  (`src/hidden_eval.rs`) scores the seven new base models through `brut-eval`, each on its
  task for one query.
- **Tasks** (`src/task.rs`): `the_mirror_grid_maps_each_grid_to_its_mirror_image`,
  `output_shapes_fit_n_out`, and the spec tests: only the mirror grid writes an `output`.
- **Base models** (`src/cnn/families.rs`): `the_16_families_have_names_and_distinct_base_models`,
  `every_base_model_passes_the_shape_check_on_every_task_of_its_kind` (default limits),
  `each_base_model_is_built_from_the_blocks_of_its_family`, and one learning test per
  family, `googlenet_learns_shapes` to `unet_learns_the_mirror_grid`. The learning tests,
  those of ticket 03 included, now lift `max_candidate_seconds` to 600 s. Side by side at
  one thread on a busy machine, the ShuffleNet took 34 s, and the default clock of 30 s
  stopped it after 11 of its 12 epochs; alone it takes 6.4 s. The other limits keep their
  defaults, and the ignored timing test checks the time.
- **Timing** (ignored, run alone at one thread):
  `slow_valid_family_b_cnns_train_within_the_time_budget` and
  `base_models_train_within_their_estimates`.

### Measurements

Base models of this ticket: trained on the Train split of their task (data seed 1), scored
on the Valid split. The scores are the same at 1 and 2 threads. Times at one thread and
nice 10, alone, against the estimate of the time model.

| Family | Task (chance) | Parameters | Training FLOPs | Epochs | Time (estimate) | Valid score | Test bound |
|---|---|---|---|---|---|---|---|
| GoogLeNet | shapes (0.25) | 1064 | 2.45e9 | 5 | 7.4 s (16.8 s) | 0.744 | 0.55 |
| DenseNet | shapes (0.25) | 5308 | 7.51e9 | 6 | 10.1 s (19.8 s) | 0.961 | 0.80 |
| ShuffleNet | shapes (0.25) | 744 | 2.39e9 | 12 | 6.4 s (18.2 s) | 0.819 | 0.65 |
| SqueezeNet | shapes (0.25) | 2524 | 5.10e9 | 5 | 7.1 s (17.1 s) | 0.817 | 0.65 |
| ConvNeXt | shapes (0.25) | 2180 | 3.03e9 | 5 | 9.0 s (19.3 s) | 0.655 | 0.45 |
| TCN | motif-1d (0.5) | 5058 | 9.65e9 | 5 | 10.3 s (17.2 s) | 0.951 | 0.80 |
| U-Net | mirror-grid, R^2 (0) | 10 561 | 1.12e10 | 10 | 11.1 s (18.5 s) | 0.733 | 0.50 |

On the data seeds 2 and 3 the scores were 0.742 and 0.751 (GoogLeNet), 0.961 and 0.921
(DenseNet), 0.813 and 0.803 (ShuffleNet), 0.791 and 0.804 (SqueezeNet), 0.615 and 0.631
(ConvNeXt), 0.961 and 0.957 (TCN), and 0.672 and 0.596 (U-Net). Each bound lies 0.15 to
0.21 below the score of seed 1, except the U-Net's: its R^2 varied most, so its bound lies
0.23 below. Each bound lies at least 0.15 above chance.

Tuning. GoogLeNet trains 5 epochs: at 6 the estimate was 20.1 s, above the budget of 20 s.
The first ShuffleNet design, width 32 with 4 groups after a stem of 16, reached 0.564 at a
learning rate of 0.003 and 0.723 at 0.01. Width 16 with 2 groups at 0.01 cost half as much
and reached 0.819 in 12 epochs. ConvNeXt with layer norm also in the stem reached 0.382 at
0.003 and 0.459 at 0.01; with batch norm in the stem it reached 0.655 at 0.01. The U-Net
reached 0.172 at 0.003, 0.435 at 0.01 and 0.733 at 0.03.

Time model. The slowest valid CNN of each of 15 layouts of the new blocks trained in 0.29 to
0.62 of its estimate, and it scored the Hidden split in 0.22 to 0.50 of the scoring
estimate. The highest share, 0.62, came from U-Nets with the per-cell head on the mirror
grid and from 1-D U-Nets with transposed convolutions. The 16 base models trained in 0.35 to
0.78 of their estimates, the seven new ones in 0.35 to 0.60. On this machine the seven slow
layouts of ticket 02 trained in 0.37 to 0.66 of their estimates; ticket 03 measured 0.36
to 0.66.

### Commands

- `cargo test` (nice -n 10, CARGO_BUILD_JOBS=3, RAYON_NUM_THREADS=2): every test binary
  passes, 917 passed, 0 failed, 18 ignored in all. The library: "test result: ok. 783
  passed; 0 failed; 13 ignored". `tests/mlp_golden.rs`: "test result: ok. 9 passed; 0
  failed; 0 ignored", with its fixtures unchanged.
- `cargo clippy --all-targets -- -D warnings`: no warning.
- `rustfmt --edition 2024 --check` on every changed `.rs` file: no change.

### Known limits

- The base models are scaled down and keep only the block of their family. The 2-D ones fit
  the 16 x 16 images, the TCN the sequences of 64, the U-Net the 8 x 8 grids. An 8 x 8 grid
  leaves room for one level of the U-Net: a 3x3 kernel does not fit a map of 2x2.
- The mirror grid is a regression with R^2, one value per cell. Ticket 06 brings ARC grids
  with a class per cell and an exact match. The per-cell head needs a 2-D genome.
- Inception's four branches have equal widths, and its second kernel is k + 2; GoogLeNet
  gives its branches different widths. Its pooling branch uses the kernel gene as window.
- A dense layer is convolution, norm, activation, as everywhere in this genome; DenseNet puts
  norm and activation before the convolution. There is no bottleneck layer (DenseNet-B) and
  no compression gene. The kernel gene of a transition has no effect.
- A Fire module squeezes by 4, SqueezeNet by 8, and its two expansions have equal widths.
- A strided ShuffleNet unit uses a projection and a sum; ShuffleNet uses an average-pooled
  shortcut and a concatenation.
- ConvNeXt has no layer scale and no stochastic depth. A strided ConvNeXt block moves in its
  depthwise convolution and projects its shortcut; ConvNeXt downsamples in separate layers.
  The base model's stem has batch norm.
- A TCN block has the stage's norm in place of weight norm, and no dropout. In the base model
  only the blocks are causal: the stem is an ordinary convolution, and the head averages
  over the whole sequence.
- A decoder has one k x k convolution after its join; U-Net has two. The skip points follow
  from the strides, not from genes, so a mutation of a stride can move them (ticket 05).
- The constants of the blocks (4 Inception branches, a squeeze by 4, a ConvNeXt expansion of
  4) are not genes.
- The time model was checked at one thread on one machine only, as in tickets 02 and 03.

### Done when

- The same tests as ticket 03 for each block and base model: yes (Tests, the first five
  items and the base model tests).
- The TCN learns the 1-D motif task above chance: yes, 0.951 against 0.5
  (`tcn_learns_the_motif`). U-Net maps a generated grid to its mirror image above chance:
  yes, an R^2 of 0.733 against 0 (`unet_learns_the_mirror_grid`).

**2026-10-31, Claude (subagent fix), after the review of tickets 03 to 05 and 07.** Two
layouts passed the shape check but could not train. Both are fixed.

- **Pooling a batch norm.** In training mode candle's batch norm returns a transposed view
  (candle-nn/src/batch_norm.rs:252), and candle's `index_select` refuses a view that is not
  contiguous (cpu_backend/mod.rs:729). An inverted residual ends with a norm and no
  activation. With pooled downsampling and no option, its strided pooling read that view:
  2-D max pooling and 1-D max and average pooling failed with "index-select only supports
  contiguous tensors". The review counted 48 such children in 20 000 on motif-1d. Now
  `take_strided` in `cnn::ops` makes its input contiguous first. candle returns a contiguous
  tensor itself (tensor.rs:2494-2497), so no plan that trained before changes a bit.
- **Group norm of a projection.** A projection shortcut normalizes the block's width, and no
  rule checked that norm for group norm. In a ConvNeXt block only the depthwise convolution
  has a norm. So a ConvNeXt stage with group norm, a skip connection and a width that 4 does
  not divide passed the check, and candle's GroupNorm then failed. `Builder::norm` now refuses
  such a group norm. `check_parts` still names the convolution first for the norms inside a
  block, so only the projection case is new.
- **Tests.** Five gradient cases pool a batch norm in training mode (worst relative error
  4.5e-10). Two block genomes pool a batch norm directly, 2-D at 4x4 and 1-D, each with a
  toy-batch test; `every_block_genome_has_a_toy_batch_test` now also checks that each block
  genome fits the default limits. `rejects_the_rules_of_the_blocks_of_ticket_04` has the
  ConvNeXt case. `every_valid_2d_stage_trains_one_step` and
  `every_valid_1d_stage_trains_one_step` take every block kind with every downsampling, norm
  and pair of options, at widths 6 and 8, after stems that move by 2 and by 4. Each layout
  that passes the shape check trains one step and gives finite outputs: 1300 in 2-D and 1672
  in 1-D. `the_parents_and_children_of_the_operators_train_one_step` (`cnn::variation`) does
  the same for the base models, random CNNs and 300 children per task kind. With the fixes
  undone, the toy-batch tests, the gradient test and both layout tests fail with the two
  errors above.
- **Evidence.** Whole suite (`nice -n 10`, `CARGO_BUILD_JOBS=3`, `RAYON_NUM_THREADS=2`): exit
  0, 962 passed, 0 failed, 18 ignored (952 before this fix). Library "820 passed; 0 failed; 13
  ignored ... finished in 146.78s"; `tests/e2e.rs` "21 passed; 0 failed ... finished in
  143.27s"; `tests/mlp_golden.rs` "9 passed; 0 failed ... finished in 0.28s", with
  `tests/fixtures` unchanged. `cargo clippy --all-targets -- -D warnings`: exit 0. `rustfmt
  --edition 2024 --check` on the 13 changed `.rs` files: exit 0. `cargo audit`: exit 0 over
  164 crates, with the one allowed warning RUSTSEC-2024-0436 (paste); no dependency changed.
- **Limits.** A strided pooling of a view now copies it, once per window row in 2-D and once
  per window position in 1-D; the time model does not count these copies. The layout tests
  cover one stage after one stem. The operator test covers deeper forms, but only a sample.
