# 05 Mutations, base models in the population, docs

Status: resolved
Blocked by: 03, 04
Spec: `../spec.md`

## What to do

- Mutation operators of the spec: switch a stage's family where shapes allow it; insert
  or remove a stage; change width, kernel, stride, dilation, groups; toggle SE, residual
  and normalization; change downsampling and activation. Crossover of whole stages at
  matching resolutions. A child that fails the shape check is redrawn.
- The first generation holds the base models of all families next to random genomes.
- CLI: choose the model kinds of a run (MLP, CNN or both) and the new tasks.
- README in German and English, ADR 0008 (CNN families and their mutations in the MLP mode;
  ADR 0007 already records the library), new terms
  in CONTEXT.md.

## Done when

- Tests: every operator keeps the genome valid on 10 000 random draws; crossover of two
  base models of different families yields valid children; a short evolution run on the
  shapes task is reproducible byte for byte from its seed.
- A short demo run is described in the README with its numbers.

Cost: about 10 to 20 dollars.

## Comments

**2026-10-31, Claude (subagent build-t05).** Built in the worktree of branch
`claude/cnn-basismodelle-teil2`, on top of tickets 03 and 04, nothing committed.

### What was built

- **Operators** (`src/cnn/variation.rs`, new). A child of one CNN gets exactly one operator,
  drawn uniformly at random, as each child gets one mutation in Real et al. 2017 (ICML).
  `OPERATORS` holds 16: the 13 of the spec (`family`, `insert-stage`, `remove-stage`,
  `width`, `kernel`, `stride`, `dilation`, `groups`, `se`, `residual`, `norm`,
  `downsampling`, `activation`) and three more, so that every gene can change: `head`,
  `head-layers` (the MLP's layer mutation on the dense layers of the head) and `training`
  (the whole CNN mutation of ticket 02). `family` rounds the width to what the new kind needs
  and lets a dense layer pool by average where it would move by its stride. `insert-stage`
  adds a stage of stride 1 that keeps the channels at its place, like its neighbour or of a
  random kind. No operator draws a change that does nothing: not the kernel of a transition,
  not the dilation of a 1x1 convolution, not the downsampling at stride 1, not the activation
  of a head without dense layers.
- **Redraw.** A child is repaired (its epochs cut to fit), then checked by
  `Genome::validate`. A child that fails, or that keeps its parent's architecture, is drawn
  again, up to `REDRAWS` = 16 times; then the next operator not yet tried is drawn. A valid
  parent always gets a valid child, at the latest from `training`. The operators do not mend
  a broken parent: for one, `mutate` gives `None`, and `RandomMutator::propose` draws a random
  genome of the run's kinds.
- **Crossover of whole stages.** The first parent, drawn at random, keeps its stem, its stages
  before a boundary i and from a boundary i2 on; the other gives its stages between
  boundaries j and j2. The maps at i and j have the same size, and so have those at i2 and j2,
  unless the child ends with the run. A child that ends with the other parent's tail takes its
  head too (one-point crossover). Training genes come from either parent, gene by gene. The cut
  is drawn uniformly from all cuts at matching sizes, again with up to 16 redraws. Pairs
  without a valid cut, and an MLP with a CNN, keep the crossover of ticket 02: the architecture
  of one parent, the training genes of both. `Plan::units` (`src/cnn/shape.rs`) now records
  the shape after the stem and after each stage for this.
- **Random CNNs.** `cnn::variation::random` draws a stem, 1 to 4 stages (plain with
  probability 1/3, else any kind that fits the dims), global average pooling or flattening,
  up to two dense layers, and training genes from CNN ranges (learning rate 1e-3 to 0.1,
  batch 16 to 128, epochs 5 to 20 before repair). On a grid task it takes the U-Net form three
  times in four. Each draw is repaired and checked; after 256 failures `smallest`, a CNN of one
  channel, takes its place. 200 draws take about 10 ms at most.
- **Model kinds of a run** (`src/config.rs`). `Config::models` is `ModelKinds::Mlp` (default,
  left out of the JSON), `Cnn` or `Both`. `Config::validate` refuses CNNs on a task with flat
  inputs, and limits in which the smallest CNN does not fit. `Config::for_models` gives the
  config of `brut init --models`: for CNNs the shapes task, 8 random genomes, 8 children, 5
  generations and 3600 s.
- **Generation 0** (`src/evolve.rs`, `src/cnn/families.rs`). After the base genomes of the
  config come the base models that pass the shape check and the limits on the run's task, in
  the order of the spec (`families::base_models_for`), with the generator name
  `base-model/<family>`. On a sequence task a 2-D base model enters as its 1-D twin
  (`Family::base_model_for`: the same genes with `Dims::OneD`). Each one left out gets a log
  line (a `SafetyEvent`, as other run logs). Then come the random genomes: CNNs, or MLPs and
  CNNs in turn for `Both`. Restarts draw from the run's kinds too. An MLP run draws exactly as
  before.
- **Proposer** (`src/variation.rs`). MLP parents take the old code path, with the same draws
  and the name `random-mutator-v1`. CNN parents go to `cnn::variation`.
  `VariantGenerator::propose_named` (a provided method) returns the child and how it was made;
  the archive names CNN children `cnn-mutator-v1/<operator>`, `cnn-mutator-v1/crossover`,
  `cnn-mutator-v1/crossover-genes` or `cnn-mutator-v1/random`.
- **CLI** (`src/main.rs`). `brut init <dir> [--task <task>] [--models mlp|cnn|both]`. An
  unknown kind, CNNs on a flat task, and `--models` with `--mode learner` are usage errors that
  leave nothing behind. The new tasks were selectable since ticket 02; `brut help` now names
  the model kinds.
- **Docs.** ADR 0008 (`docs/adr/0008-cnn-families-enter-a-run-as-base-models-and-each-child-changes-by-one-operator.md`);
  CONTEXT.md gains a section "CNNs in MLP mode" with Model kind, CNN, CNN stage, CNN family,
  Base model, Shape check, Time model and Operator, and Candidate now covers CNNs; README.md
  and README.en.md gain "CNN-Modelle im MLP-Modus" / "CNN models in the MLP mode" with the
  demo run; the module docs of `lib.rs`, `cnn`, `families`, `variation` and `evolve` follow.

### Tests

- `every_operator_keeps_the_genome_valid_on_10_000_draws` (`src/cnn/variation.rs`): for each
  of the 16 operators, 10 000 draws from valid parents of the shapes, motif and mirror grid
  tasks (the base models that fit, 20 random CNNs and 300 random descendants each). Every
  child passes `Genome::validate`, has a new seed, and, except for `training`, another
  architecture. Children per 10 000 draws: `remove-stage` 7452 and `dilation` 7447 (the
  fewest), `stride` 7894, `downsampling` 8241, `groups` 9079, `insert-stage` 9698, `head`
  9727, `family` 9885, all others 9979 to 10 000.
- `each_operator_changes_only_its_genes`: 300 draws per operator and task; each child differs
  from its parent only in the genes its operator names (plus fewer epochs from repair), e.g. a
  kernel by 2, a stride by 1, one inserted stage of stride 1.
- `crossover_of_two_base_models_of_different_families_gives_valid_children`: every ordered
  pair of base models of different families, on all three kinds of task, 10 proposals each
  through `RandomMutator::propose`: all valid, all CNNs. Pairs that exchanged stages: shapes
  172 of 182, motif 176 of 210, mirror grid 6 of 6. Every pair without one contains
  MobileNetV2 or the TCN (asserted). MobileNetV2 pools to 4x4 in its stem; the dilated CNN
  never leaves 8x8, and with GoogLeNet, ResNet, SqueezeNet and ConvNeXt every cut at 4x4 was
  estimated at 22 to 54 s, above the budget of 20 s. The TCN keeps length 64, which no 1-D twin
  has after its stem.
- `cuts_meet_at_matching_resolutions_and_children_take_whole_stages` (cuts and heads by
  hand), `mutate_draws_every_operator_and_gives_valid_children` (4000 draws, each operator
  155 to 283 times, the same child for the same draws),
  `an_mlp_or_a_broken_cnn_gets_no_child`, `random_cnns_are_valid_deterministic_and_fit_their_task`
  (all 10 tasks), `widths_round_to_the_step_of_their_kind`, `the_operators_have_distinct_names`.
- `src/variation.rs`: `a_cnn_child_gets_one_operator_and_stays_valid` (all 16 operator names
  occur; only `training` keeps the architecture) and
  `crossover_with_a_cnn_takes_the_architecture_whole_or_exchanges_stages` replace the two tests
  of ticket 02 that pinned the old CNN behaviour; `crossover_of_two_mlps_draws_as_before` still
  passes.
- `src/evolve.rs`: `generation_0_of_a_cnn_run_holds_every_base_model_that_fits` (14 on shapes,
  15 on motif, 3 on the mirror grid, in order, with their names, then random CNNs),
  `a_run_of_both_kinds_starts_with_mlps_and_cnns_in_turn`,
  `an_mlp_run_draws_its_random_genomes_as_before` (generation 0 and restarts equal
  `Genome::random` in a row), `a_cnn_run_restarts_with_random_cnns`.
- `src/config.rs`: `model_kinds_are_left_out_for_mlp_runs_and_checked_for_cnns`; `src/main.rs`:
  `init_takes_the_model_kinds_of_a_run`.
- `a_short_cnn_run_on_shapes_is_reproducible_byte_for_byte` (`tests/e2e.rs`): a CNN run on
  shapes with `max_params` 1100 (base models of GoogLeNet, Xception and ShuffleNet; 13 log
  lines), 2 random CNNs and 3 children, in two child processes `brut run --no-hidden` at
  `RAYON_NUM_THREADS` 1 and 2 at once. Config, archive (clock fields masked), head and all 8
  model files are equal, and so are the summaries. On this seed the children came from
  `stride`, `width` and a crossover of ShuffleNet and GoogLeNet. About 52 s.
- MLP behaviour: `tests/mlp_golden.rs` passes with its fixtures unchanged
  (`git status tests/fixtures` is clean), and so does the pre-CNN run of `tests/e2e.rs`.

### Demo run

`brut init <dir> --models cnn --seed 1`, then `brut run <dir>`, release build,
`RAYON_NUM_THREADS=2` on 4 cores, with the hidden evaluation: 62 candidates (14 base models,
8 random CNNs, 40 children), all Ok, none stopped by the clock, 68.6 s, 3.8e11 FLOPs. The base
models scored 0.670 (LeNet-5) to 0.939 (DenseNet) on Valid, the random CNNs at most 0.573. The
median per generation was 0.728 in generation 0 and 0.791 to 0.828 later. Top three by fitness
with their hidden scores: DenseNet's base model 0.939 / 0.93; a generation-5 child 0.934 /
0.94, a mixed form of GoogLeNet's Inception blocks and a VGG stage that descends from two
crossovers (GoogLeNet with a descendant of VGG x ShuffleNet); its generation-4 parent 0.929 /
0.92. 16 of the 40 children and 4 of the 8 random CNNs mix families. No child beat the best
base model on Valid; on Hidden the best child lies one rounding step above it, so the run
shows the mechanics, not a gain from breeding. A second run at one thread without the hidden
evaluation took 2 min 9 s and gave the same 62 candidates and model files byte for byte. The
README sections give the full table.

### Commands

- `cargo test` (nice -n 10, CARGO_BUILD_JOBS=3, RAYON_NUM_THREADS=2): all 16 test binaries
  pass, 933 passed, 0 failed, 18 ignored. The library: "test result: ok. 797 passed; 0
  failed; 13 ignored; 0 measured; 0 filtered out; finished in 108.90s". `tests/e2e.rs`: "test
  result: ok. 17 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in
  58.30s". `tests/mlp_golden.rs`: "test result: ok. 9 passed; 0 failed; 0 ignored; 0
  measured; 0 filtered out; finished in 0.19s".
- `cargo clippy --all-targets -- -D warnings`: no warning.
- `rustfmt --edition 2024 --check` on every changed `.rs` file: no change.

### Known limits

- In the demo, breeding did not beat the best base model within 5 generations of 8 children.
  A longer run may; this ticket shows only that the operators, the crossover and the base
  models work together.
- The operator choice is uniform and fixed. Operators that often fail (`remove-stage`,
  `dilation`, `stride`) come a little less often, since a failed operator gives way to another.
- A base model that breaks a run's tighter limits stays out; it is not cut down to fit. On the
  mirror grid only 3 base models fit; the 1-D twins are not tuned and no test asks them to
  learn.
- Pairs with MobileNetV2 or the TCN often exchange no stages (above); their children take one
  parent's architecture.
- A child trains from scratch; Real et al. 2017 inherit weights.
- The skip points of a U-Net follow from the strides, so `stride`, `insert-stage` and
  `remove-stage` can move or break the decoder wiring; such children are redrawn.
- In the demo the release build trained each base model in 1.6 to 2.7 s at 2 threads, and no
  candidate of generation 0 took more than 4.6 s. The time model estimates 13 to 20 s for the
  base models; it comes from the build of `cargo test` at one thread, so it is far from tight
  for a release build.
- `a_short_cnn_run_on_shapes_is_reproducible_byte_for_byte` also asserts that a crossover child
  appears on seed 1. Parent choice depends on the scores, so another CPU with other bits may
  draw other children.
- Not done, as asked: no git writes; ticket 07 (promotion) and ticket 06 (ARC grids) are not
  built. The library paragraph of the spec still names D1 to D8 (noted by ticket 04).

### Done when

- Every operator keeps the genome valid on 10 000 random draws: yes
  (`every_operator_keeps_the_genome_valid_on_10_000_draws`).
- Crossover of two base models of different families yields valid children: yes
  (`crossover_of_two_base_models_of_different_families_gives_valid_children`; 172 of 182 pairs
  on shapes also exchange stages).
- A short evolution run on the shapes task is reproducible byte for byte from its seed: yes
  (`a_short_cnn_run_on_shapes_is_reproducible_byte_for_byte`, at 1 and 2 threads).
- A short demo run is described in the README with its numbers: yes (README.md, "Demolauf mit
  CNNs"; README.en.md, "A demo run with CNNs").

**2026-10-31, Claude (subagent fix), after the review of tickets 03 to 05 and 07.**

- README.md and README.en.md, "Mutation und Kreuzung": the operators `dilation` and `groups`
  change a stage only, since the stem has neither gene. The last two rows of the table, with
  `head`, `head-layers` and `training`, go beyond the spec, not three. The two counts of the
  section now say that we measured them. ADR 0008 describes LeNet-5 as a 5x5 stem and one
  plain stage, as `families::lenet5` builds it.
- The fixes of ticket 04's note of the same date change no candidate of the demo run: a rerun of `brut init --models cnn --seed 1` (dev build, `RAYON_NUM_THREADS=2`, clock limit lifted to 600 s so that load cannot cut a training, `--no-hidden`) gives the same 62 candidate records, clock fields masked, and the same 62 model files, byte for byte, as the release-build demo run of this ticket. All 62 are Ok, and the best is still the DenseNet base model at 0.939.
