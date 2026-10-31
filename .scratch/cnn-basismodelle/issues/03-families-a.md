# 03 CNN families, part A

Status: resolved
Blocked by: 02
Spec: `../spec.md`

## What to do

Blocks and one scaled-down base model each, within the limits:

- LeNet-5 (5×5 convolution, pooling, dense head);
- AlexNet style (large strided first convolution);
- VGG (stacks of 3×3);
- ResNet (basic and bottleneck residual blocks);
- Xception and MobileNet (depthwise separable convolution);
- MobileNetV2 (inverted residual, linear bottleneck, ReLU6);
- SE (squeeze and excitation as an option on any block);
- dilated convolution.

## Done when

- Each block has a test of its output shape and parameter count against a hand count,
  and a test that its gradients are finite and reduce the loss on a toy batch.
- Each base model passes the shape check and learns the shapes task above chance.

Cost: about 15 to 30 dollars.

## Comments

**2026-10-31, Claude (subagent build-t03).** Built in the worktree of branch
`claude/cnn-basismodelle-teil2`, nothing committed.

### What was built

- **Blocks** (`src/cnn/shape.rs`, `src/cnn/net.rs`). `Basic`, `Bottleneck`, `Separable`
  and `InvertedResidual` now pass the shape check next to `Plain`. The module doc of
  `shape` lists the convolutions of each block. A bottleneck has width / 4 inner channels
  (`BOTTLENECK_RATIO`, as in ResNet). An inverted residual expands to 6 times its input
  channels (`EXPANSION`, t = 6 of MobileNetV2) and ends without activation. The dilation
  applies to every k x k convolution of a block, the stride to the first. LeNet-5,
  AlexNet, VGG and the dilated CNN use the plain stage of ticket 02.
- **Options on every block.** `Stage::se` puts squeeze-and-excitation on the output of the
  block's last norm, as in SE-ResNet, with a reduction of 4 (`SE_REDUCTION`; SENet uses
  16, which leaves 1 to 4 units at our widths). `Stage::residual` adds the block's input
  before the last activation. The shortcut is the input itself when the block keeps the
  channels and the size, else a 1x1 convolution with the stride and the norm (ResNet's
  projection). The plan marks a skip connection with three new layers, `Fork`, `Branch`
  and `Add`; squeeze-and-excitation is one layer, `SqueezeExcite`.
- **Shape check.** New rules: a bottleneck's width is a multiple of 4; the groups divide the
  channels before and after each convolution they apply to; every group norm can split its
  width; the span of every kernel fits. An error names the convolution, e.g. "stage 2:
  convolution 2: kernel 3 with dilation 1 spans 3 cells, larger than the input of 2".
- **FLOPs and time model.** The FLOP rule is unchanged; the dense layers of
  squeeze-and-excitation count once per sample. The time model prices the new layers with
  the existing constants: a fork and a branch count nothing, a sum its values and one call,
  squeeze-and-excitation its output, its small vectors and 6 calls. No price changed, so
  every plan without the new layers keeps its estimate to the bit.
- **Model files.** `MAX_RUNNING_VALUES` grows from 6656 to 86 528: an inverted residual with
  a projection holds batch norms of 14 x 256 channels per stage. `docs/SICHERHEIT_UND_RECHT.md`
  names the new cap. No serialized format changed, and no field was added.
- **Families and base models** (`src/cnn/families.rs`, new). `Family` names the 16 families,
  `FAMILIES` lists them in the order of the spec, `Family::base_model` gives the base model
  of the 9 families of this ticket and `None` for the 7 of ticket 04, and `base_models()`
  lists the built ones for ticket 05.
- **No new tensor operation.** The blocks use operations of `ops` that have gradient cases:
  grouped `conv2d`, `add`, `global_avg_pool`, `linear`, `relu`, `sigmoid`, `reshape` and
  `broadcast_mul`. The chain of squeeze-and-excitation has its own gradient case in
  `ops`.

### Tests

- **Hand counts** (`src/cnn/shape.rs`): `the_block_stem_counts_by_hand`,
  `a_plain_stage_with_both_options_counts_by_hand`, `dilated_plain_stages_count_by_hand`,
  `a_basic_block_with_a_strided_projection_counts_by_hand`,
  `a_basic_block_without_norm_has_biases_also_in_its_projection`,
  `a_bottleneck_counts_by_hand`, `a_separable_block_counts_by_hand`,
  `inverted_residuals_count_by_hand` and `a_1d_genome_runs_its_blocks_along_the_sequence`
  compare each block's layers, with their output shapes and parameters, and the plan's
  parameter count with a count by hand; three of them also compare the forward FLOPs.
  `the_time_model_counts_forks_sums_and_squeeze_and_excitation` does the same for the
  counts of the time model.
- **Rules**: `rejects_every_block_kind_not_yet_built` (8 kinds left),
  `every_built_block_takes_both_options`, `rejects_a_bottleneck_width_the_ratio_does_not_divide`,
  `rejects_groups_that_do_not_divide_a_convolution_of_a_block`,
  `rejects_group_norm_on_an_inner_width_it_cannot_split`,
  `rejects_a_kernel_that_does_not_fit_inside_a_block`. `no_genome_panics` now draws the new
  blocks and options too: 369 valid genomes, 272 of them with a new block or option, and
  17 631 invalid ones; none panics.
- **Gradients and the loss on a toy batch** (`src/cnn/train.rs`): one test per genome of
  `testing::block_genomes`. Each checks that every gradient is finite and reaches every
  tensor, then runs 50 Adam steps on 32 samples and requires the loss below 0.7 of its
  start. Measured: plain with both options 2.237 to 0.772, dilated plain 1.475 to 0.241,
  strided basic with a projection 1.481 to 0.004, basic without norm 1.452 to 0.356,
  bottleneck with group norm and groups 1.632 to 0.785, separable with layer norm, gate,
  projection and pooling 1.713 to 0.711, inverted residuals 1.494 to 0.001, 1-D blocks
  0.720 to 0.000. `every_block_genome_has_a_toy_batch_test` keeps the list complete.
- **Wiring**: `loss_and_grad_matches_finite_differences` has a third genome with all four
  blocks, gates and both kinds of shortcut: 162 values, worst error 0.19 of the bound, at
  a step of 1e-3. At a step of 1e-2 one value missed the bound; the step of 1e-3 matched.
- **Reproducibility**: the two thread tests have a third genome of all four blocks with
  both options. Its losses and model file are byte-equal in rayon pools of 1 and 4 threads
  and in child processes with `RAYON_NUM_THREADS` at 1 and 3.
- **Files and scoring**: `models_of_every_block_parse_back_check_and_give_outputs`
  (`src/cnn/model.rs`); `the_running_statistics_have_their_own_small_cap` reaches the new
  cap with 12 projections on a sequence of 4096 cells; `cnn_files_are_scored_and_bad_ones_cost_no_query`
  (`src/hidden_eval.rs`) scores the base models of ResNet, Xception, MobileNetV2 and SENet
  through `brut-eval`, one query each.
- **Base models** (`src/cnn/families.rs`): `the_16_families_have_names_and_ticket_03_built_9_of_them`,
  `every_base_model_passes_the_shape_check_on_every_image_task` (all four image tasks,
  default limits), `each_base_model_is_built_from_the_blocks_of_its_family`, and one
  learning test per family (`lenet5_learns_shapes` to `dilated_cnn_learns_shapes`).
- **Timing** (ignored, run alone at one thread): `slow_valid_block_cnns_train_within_the_time_budget`
  and `base_models_train_within_their_estimates`.

### Measurements

Base models on shapes: trained on the Train split (data seed 1), scored on the Valid
split. Chance is 0.25. The accuracies are the same at 1 and 2 threads. Times at one
thread and nice 10, alone, against the estimate of the time model.

| Family | Parameters | Training FLOPs | Epochs | Time (estimate) | Valid accuracy | Test bound |
|---|---|---|---|---|---|---|
| LeNet-5 | 11 392 | 1.22e10 | 10 | 11.6 s (18.7 s) | 0.563 | 0.40 |
| AlexNet | 15 476 | 1.68e10 | 5 | 10.6 s (13.5 s) | 0.740 | 0.55 |
| VGG | 12 540 | 1.18e10 | 6 | 9.7 s (16.6 s) | 0.916 | 0.75 |
| ResNet | 2716 | 4.44e9 | 6 | 6.8 s (13.8 s) | 0.782 | 0.60 |
| Xception | 1028 | 2.02e9 | 6 | 5.3 s (13.7 s) | 0.773 | 0.60 |
| MobileNet | 2660 | 2.72e9 | 8 | 7.3 s (18.3 s) | 0.854 | 0.65 |
| MobileNetV2 | 3372 | 1.74e9 | 5 | 6.1 s (13.6 s) | 0.760 | 0.60 |
| SENet | 5210 | 5.67e9 | 6 | 7.2 s (13.9 s) | 0.788 | 0.60 |
| Dilated CNN | 6012 | 1.19e10 | 5 | 9.3 s (16.9 s) | 0.872 | 0.70 |

Each bound lies 0.15 to 0.21 below the accuracy reached here and at least 0.15 above
chance. All base models train with Adam in batches of 32. LeNet-5 and MobileNet train at a
learning rate of 0.01; at 0.003 they reached 0.472 (8 epochs) and 0.536 (6 epochs), and
LeNet-5 at 0.03 stayed at chance. The others train at 0.003.

Time model. The slowest valid CNN of each of 12 layouts of the new blocks trained in 0.29
to 0.64 of its estimate and scored the 2000 Hidden rows in 0.19 to 0.34 of the scoring
estimate. The highest share, 0.64, came from 12 tiny plain stages with both options at 1x1
and 40 epochs. The nine base models trained in 0.38 to 0.79 of their estimates; the 0.79
is AlexNet, a plain CNN of ticket 02. On this machine the seven slow layouts of ticket 02
trained in 0.36 to 0.66 of their estimates, while ticket 02 measured up to 0.89 over all
its runs, so this machine is not slower than the one that wrote the model.

### Commands

- `cargo test` (RAYON_NUM_THREADS=2): every test binary passes, 887 passed, 0 failed, 17
  ignored in all. The library: "test result: ok. 753 passed; 0 failed; 12 ignored".
  `tests/mlp_golden.rs`: "test result: ok. 9 passed; 0 failed", with its fixtures
  unchanged.
- `cargo clippy --all-targets -- -D warnings`: no warning.
- `rustfmt --edition 2024 --check` on every changed `.rs` file: no change.

### Known limits

- The base models are 2-D and fit the 16 x 16 image tasks; none fits the 1-D task. They do
  not enter a run yet: ticket 05 puts them into the first generation (`base_models()`;
  `Config::base_genomes` takes them today).
- Xception's separable convolution has no activation between its depthwise and its
  pointwise part; ours has the norm and activation of MobileNet. Pooled downsampling comes
  after the join of a skip connection. For average pooling this equals pooling both paths;
  for max pooling, as in Xception's entry flow, it does not.
- The expansion of 6 and the reduction of 4 are constants, not genes.
- Inverted residuals are dear in the time model: two of them on 8 channels at 8x8 were
  estimated at 36.3 s for 5 epochs, above the 20 s budget, so the MobileNetV2 base model
  and the test genome run at 4x4. They trained in 0.38 to 0.45 of their estimates, so the
  model is cautious there.
- A layer norm over very few channels gives about +1 or -1 and jumps between them: over the
  2 inner channels of a bottleneck of width 8, finite differences at 1e-2 fail. The shape
  check allows such a layout; it trains, but its gradients are steep.
- The time model was checked at one thread on one machine only, as in ticket 02.

### Done when

- Each block has a test of its output shape and parameter count against a hand count, and
  a test that its gradients are finite and reduce the loss on a toy batch: yes (Tests,
  first and third item).
- Each base model passes the shape check and learns the shapes task above chance: yes
  (`every_base_model_passes_the_shape_check_on_every_image_task` and the nine learning
  tests; table above).
