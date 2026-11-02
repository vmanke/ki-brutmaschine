# ARC grid tasks train a fresh CNN per ARC-AGI-1 training task and score whole grids

Ticket 06 of the CNN feature (`.scratch/cnn-basismodelle/issues/06-arc-grids.md`) asks for
grid-to-grid tasks from ARC: grids up to 30 x 30, the 10 colours as input channels, one
colour class per cell, scored by the exact match of the whole output grid, with the cell
accuracy as a descriptive measure. The training data are the ARC-AGI-1 training tasks and
their re-arc generations, since each task alone has only a few pairs; both evaluation sets
stay held out. The base models are a U-Net and a fully convolutional network. Done means
that a CNN genome trains on a set of ARC task families and is scored on held-out examples.
The ticket and the spec fix these points. This record decides the rest. The code is in
`src/arc_grid.rs`, `src/arc_data.rs`, `src/task.rs`, `src/score.rs`, `src/cnn/ops.rs`,
`src/cnn/train.rs` and `tools/rearc_to_jsonl.py`; the ticket's Comments hold the
measurements. ADR 0009 is taken by step 2 (`.scratch/seele-und-kappa/issues/12-docs-adr-and-glossary.md`).

**One ARC grid task per training task, and a fresh CNN for each.** A plain CNN maps one
input grid to one output grid. Pooled over many tasks, the same input could need different
outputs, and nothing tells the network which task it faces. So each ARC-AGI-1 training task
is its own ARC grid task (`ArcGridTask`). A genome trains a fresh CNN on each task from its
own seed, as an MLP child trains from scratch (ADR 0008), and the CNN is scored on that
task's held-out pairs only. Its result on a set of tasks is the mean over the tasks. This
is also how a CNN Worker would use a genome (`.scratch/llm-nutzt-cnn/spec.md`, Q78 and O2).

**The splits.** The train pairs of the task file and the re-arc pairs train, except the last
fifth of the re-arc pairs, at least one, which validates (`VALID_SHARE`). The test pairs of
the task file are the test split: the pairs that ARC poses. Before the split, a re-arc pair
equal to a pair of the task file or to an earlier re-arc pair is dropped, so that every
held-out pair is new to the training. No pair is chosen by a seed, so the splits are the
same for every genome and every run.

**What is left out.** A training task whose task file has a pair that changes the grid size
is left out: a per-cell network keeps the size of its input, and predicting the size is
later work (as O10 of the same spec says for CNN Workers). So are the two training tasks
that repeat pairs of ARC-AGI-1 evaluation tasks, 40853293 and caa06a1f
(`arc_data::A1_TRAIN_REPEATS_EVAL`), with their re-arc pairs, and a task without a re-arc
pair that keeps its size. Of the 400 tasks, 259 remain: 138 change the size, 2 repeat
evaluation pairs and 1 has no usable re-arc pair. The training sources (`TRAINING_SOURCES`)
pass `check_contamination` for all three evaluation sets, and the grid path reads no
evaluation file. On the real data no pair of the 259 tasks shares a canonical hash (all 8
rotations and reflections, colours renumbered) with a pair of ARC-AGI-1 evaluation,
ARC-AGI-2 evaluation or ConceptARC; the two left out touch 070dd51e and 50a16a69.

**One canvas of 30 x 30 for every task.** Each grid lies at the top left of a square canvas
of 30 cells per side, the largest ARC grid. The input has one channel per colour, so a cell
outside the grid has no channel on; there is no eleventh channel for the inside, since the
sum of the colour channels already says it. Each output cell gets the class of its colour,
and a cell outside the output grid the label `NO_CELL`, which carries no loss and no score.
One canvas gives every task the same shapes, so the shape check of a genome holds for all
tasks, and `TaskKind::ArcGrid.spec()` can be the largest task (`arc_grid::largest_spec`). A
canvas per task would save little: the largest grid of a task has 29 or 30 cells per side in
255 of the 259 tasks, because re-arc draws large grids. `--canvas` takes a smaller side,
which leaves out the pairs that do not fit; the tests use 6.

**One class per cell is a classification with a grid of classes.** No new objective: a
classification whose outputs form a grid with one channel per class classifies each cell
(`TaskSpec::cell_classes`). So the matches over `Objective` stay as they were, and so does
every task before ticket 06, none of which has such an output. The scores reuse the fields of
`Metrics`: `primary` is the exact match, the share of grids with every cell inside right;
`accuracy` is the cell accuracy; `loss` is the mean cross-entropy over the cells inside. No
field is added, so archives and model files keep their bytes.

**The loss over the cells inside is one operation of `ops`.** `ops::cell_cross_entropy`
gathers the logits of the cells that carry a label with candle's `index_select` and takes
candle_nn's cross-entropy on them. Its gradient passes the f64 check of `ops` with the other
operations, a cell outside gets no gradient, and moving its logits changes the loss by no bit.
The learner passes the cells inside each batch (`cnn::train`).

**Epochs shrink to fit a task.** A genome trains on a task for its own epochs, or for the most
that pass the shape check there (`epochs_on`), so the FLOP limit and the time model, not the
clock, end its training (ADR 0007). A genome that does not fit at `MIN_EPOCHS` is refused on
that task, and the task counts 0, as a missing task counts in ARC's official score. With one
canvas the counts of pairs are the only difference between tasks, so a genome that fits the
largest task trains every task for its own epochs; both base models do.

**An MLP cannot learn one class per cell.** The hand-written loss of the MLP (`nn`) knows one
class per sample. Rather than teach it a second loss, the shape check refuses an MLP on a task
with one class per cell. On the 30 x 30 canvas no MLP fits the default limits anyway: even the
linear one has 81 million parameters.

**No breeding on ARC yet.** `TaskKind::ArcGrid` names these tasks in their specs. It is not in
`TaskKind::ALL`, `generate` gives it no sample, `Config::validate` refuses it, and `brut-eval`
refuses it before it charges a query. `brut arc-grid` trains and scores genomes on the tasks
instead: the base models, or the genome of a file, next to the copy of the input. A run that
breeds on ARC would need a fitness over many tasks per candidate and a held-out split for the
hidden phase; that is its own ticket.

**The copy of the input is the baseline.** Many ARC outputs keep most cells of their input. So
the report of `brut arc-grid` puts the copy first. Cell accuracy is only read next to it.

**The converter writes the grid tasks' file, and E2 can still have its own.** The contract
(`docs/STAGE1_CONTRACT.md`, section 5.13) describes the converter for experiment E2: the first
60 examples of a task that keep their size, with at most 225 cells. The grid tasks need grids
of up to 900 cells. So `tools/rearc_to_jsonl.py` takes `--max-area` (default 900; 225 for E2)
and `--max-pairs` (default 60), and the grid tasks read the file at `re-arc/grids.jsonl`
(`arc_data::REARC_GRIDS`). The loader of E2 keeps refusing grids above 225 cells, so it needs
its own file, made with `--max-area 225`. The audit reads both kinds. The converter checks the
SHA-256 of `re_arc.zip` before it reads a member, uses the Python standard library only, and
imports and runs no code of re-arc.

**The sandbox hands over file descriptors instead of a staging directory.** The contract asks
for the sandbox recipe of the plan, from a staging directory of user 65534, because that user
cannot read `/tmp/claude-0`. `tools/rearc_sandbox.sh` runs the same layers (setpriv to 65534
with no_new_privs first, then user, network, PID, mount, IPC and UTS namespaces, a private
tmpfs and prlimit), but gives the converter the zip on standard input and its own code on
file descriptor 3, and takes the lines from standard output. So the converter opens no path
outside its tmpfs, and no downloaded file is copied anywhere.

**No NOTICE file.** No data of ARC-AGI-1 or re-arc is passed on: the repository holds the
manifest of hashes, the converter and hand-made fixtures. A NOTICE file follows when data are
passed on (docs/SICHERHEIT_UND_RECHT.md).

**The base models.** BASE-MODELS-PLACEHOLDER

Rejected: one CNN pooled over all tasks (it cannot know which task a grid belongs to); a canvas
per task (little saved, and a genome could fit one task and not another); an eleventh input
channel for the inside (the colour channels already say it); a new `Objective` variant (it
would change matches that tests of the MLP rely on); the U-Net base model of the CNN families
(its stem of 4 channels squeezes 10 colours; it was built for one channel); a loss for MLPs
(none fits the limits on this canvas); reading the evaluation sets in the grid path to audit
the pairs (the audit runs as a test on the real data instead, so the grid path never opens an
evaluation file).

Status: accepted on 2026-11-01. The owner approved ARC grids as the second kind of task on
2026-10-30 (spec), and the ticket fixes the data, the score and the two base models. Claude,
the AI assistant, chose the details above while building ticket 06.
