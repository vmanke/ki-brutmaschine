# CNN families enter a run as base models, and each child changes by one operator

The CNN feature (`.scratch/cnn-basismodelle/spec.md`) adds convolutional networks as a second
kind of model in the MLP mode. Tickets 03 and 04 built the blocks of the 16 families of the
spec and a scaled-down base model of each. ADR 0007 records the library. This record decides
how a run breeds CNNs: which kinds of model a run holds, how its first generation starts, how
a child differs from its parents, and what stays as it was for MLPs. Ticket 05 built it, in
`src/cnn/variation.rs`, `src/cnn/families.rs`, `src/variation.rs` and `src/evolve.rs`; its
Comments hold the measurements. It also decides how the winners of a run become base models
of later runs. Ticket 07 built that, in `src/promote.rs`, `src/hitl.rs` and `src/evolve.rs`.

**A genome is a kit, and a family is one way to use it.** "All kinds of CNN" cannot be listed.
So a CNN genome holds a stem, up to 12 stages and a head, and each stage names a block kind
with its genes (`src/cnn/genome.rs`). The 16 families are special cases of this kit. LeNet-5,
for example, is a 5x5 stem and one plain 5x5 stage, each with average pooling, and a flatten
head with two dense layers; ResNet is a basic and a bottleneck block with skip connections. A mixed form is a genome whose stages come from
several families. The shape check (`cnn::shape`) decides for each genome whether its shapes
fit the task. It also holds the parameters, the training FLOPs and the estimated training time
to the limits that MLPs share.

**A run names the kinds of model it breeds.** `Config::models` is `Mlp`, `Cnn` or `Both`, and
`brut init --models` sets it. `Mlp` is the default, and the JSON leaves it out. So configs,
archives and model files of MLP runs keep their bytes: `tests/mlp_golden.rs` passes with its
fixtures from before CNNs. A run with CNNs needs a task with images or sequences, and limits
in which the smallest CNN fits; `Config::validate` refuses anything else. `brut init` writes a
smaller run for CNNs (`Config::for_models`): the shapes task unless `--task` names another, 8
random genomes, 8 children per generation, 5 generations and an hour of run time. A CNN trains
for seconds where an MLP trains for milliseconds.

**The first generation starts from the known designs.** Generation 0 of a run with CNNs holds
the base model of each family that passes the shape check and the limits on the run's task. They
come in the order of the spec, after the base genomes of the config and before the random
genomes. On a task with a sequence input, a 2-D base model enters as its 1-D twin: the same
genes, with each kernel and window along the sequence. The twins are not tuned. A base model
that does not fit stays out, and the archive records why. Under the default limits the image
tasks get 14 base models (all but the TCN and the U-Net), the motif task 15 (the TCN and 14
twins), and the mirror grid 3 (the U-Net, ResNet and SqueezeNet). The random genomes of a run
with CNNs are random CNNs; a run of both kinds draws an MLP and a CNN in turn. The generator name
of a base model names its family, for example `base-model/resnet`.

**A child of one CNN gets exactly one operator.** In Real et al. 2017 (ICML) each child gets one
mutation, drawn uniformly at random from a list. We do the same. Our list holds the 13 mutations
of the spec: switch the family of a stage; insert or remove a stage; change the width, kernel,
stride, dilation or groups; turn squeeze-and-excitation or the skip connection on or off; change
the normalization, the downsampling or the activation. Three more operators let the other genes
change: the head, the dense layers of the head, and the training genes. The last one is the whole
CNN mutation of ticket 02. An operator never draws a change that does nothing, such as the kernel
of a 1x1 transition. The archive names the operator of each child, for example
`cnn-mutator-v1/insert-stage`. MLP children keep the name `random-mutator-v1` and the draws they
had.

**Every child passes the shape check, or it is drawn again.** A child is repaired first, which
may cut its epochs to fit the budget. A child that still fails the check, or keeps its parent's
architecture, is drawn again with fresh random choices, up to 16 times. Then the next operator is
drawn from those not yet tried. The training operator keeps the architecture, so a valid parent
always gets a valid child. In 10 000 draws per operator from base models, random CNNs and their
descendants, every child was valid, and 74 to 100 % of the draws gave one. The operators keep a
valid genome valid; they do not mend a broken one. A parent that fails the check, which a run
never trains, gets a random genome of the run's kinds as its child.

**Two CNN parents exchange whole stages where their resolutions match.** A boundary of a parent
lies after its stem or after one of its stages. The first parent, drawn at random of the two,
keeps its stem, its stages before a boundary i and its stages from a boundary i2 on. The other
parent gives the run of its stages between two boundaries j and j2. The maps at i and j have the
same size, and so have those at i2 and j2, unless the child ends with the run. A child that ends
with the other parent's tail takes that parent's head as well; this is one-point crossover. The
training genes come from either parent, gene by gene. The cut is drawn uniformly from all cuts at
matching sizes, and again a child must pass the check, or another cut is drawn. Two parents
without a valid cut, and an MLP with a CNN, give a child with the architecture of one parent and
the training genes of both, as before. Of the 182 ordered pairs of base models of different
families on the shapes task, 172 exchanged stages. The other 10 pair MobileNetV2, whose stem pools
to 4x4 at once, with five families. The dilated CNN never leaves 8x8, so the two share no
resolution. With GoogLeNet, ResNet, SqueezeNet and ConvNeXt, every child of a cut at 4x4 was
estimated at 22 to 54 s, above the budget of 20 s.

**A child trains from scratch.** Real et al. 2017 let a child inherit its parent's weights. We do
not. A child of another shape would need a rule per operator to carry weights over, and its
training would depend on its parent's model file. So each child starts from the weights that its
own seed draws, as an MLP child does. A run with CNNs then stays reproducible byte for byte from
its seed on one machine, at any thread count, while no candidate hits the clock
(`a_short_cnn_run_on_shapes_is_reproducible_byte_for_byte` in `tests/e2e.rs`).

**The winners of a run are promoted, if they beat its best base model.** The owner asked on
2026-10-30 that winners can be fed back upward. A finished run promotes at most two models, MLPs
or CNNs: the two best by their score on the hidden split, which the run never saw while it bred.
Only the candidates of the hidden set have that score: the `hidden_eval_top_k` best by validation
fitness, 3 by default, and the best base model. So the two come from the first ones by fitness.
Each of the two is promoted only if its hidden score is above that of the run's best base model.
A base model of a run is a known design in its generation 0: a base genome of the config, the base
model of a CNN family, or a promoted model (`evolve::is_base_model`). The best one has the highest
validation fitness (`evolve::best_base_model`). Only a run that breeds CNNs or holds promoted
models can promote (`Config::can_promote`). Its hidden set holds the best base model next to the
`hidden_eval_top_k` best, so that its hidden score exists; that costs at most one hidden query
more. Every other run keeps its hidden set as before, base genomes or not. So an MLP run as
before CNNs spends the same hidden queries and writes the same archive, and it promotes nothing.
A run without base models has nothing to beat, so it promotes nothing either. The first
promoted MLP therefore comes from a run of both kinds. A model whose content id is in the pool or
is that of a base model of a family adds nothing. So does a model that fails the gate of the pool.
The third best does not take its place, so a second promotion of the same run adds nothing new. `brut promote <run> --commit <sha>`
writes the files under `<run>/promote/` (`src/promote.rs`). `scripts/propose-pr.sh` opens a draft
pull request into `candidates/models/` with them. Such a pull request carries one or two models,
not always two as a mutation does (ADR 0004), since only models that beat the bar go. A human
merges it, and the run merges nothing (ADR 0004).

**A promoted model is its genome with its provenance.** The file
`candidates/models/<content-id>.json` holds the genome and where it comes from: the run (the hash
of its first record, `Archive::run_id`), the task, the master seed, the git commit, the scores on
the validation and hidden splits, the best base model with its scores, and the lineage. The
lineage lists the model and all its ancestors, breadth-first. It ends at candidates without
parents: base models and random genomes of generation 0, or random restarts. The content id is the genome's id, the first 16 hex digits
of the SHA-256 of its canonical JSON. So the same genome from another run is the same candidate.
`hitl::ModelAdapter` is the gate, behind the `CandidateAdapter` interface of ADR 0004. It refuses
unknown fields, in the genome too. It runs the shape check under the current default `Limits` on
the task that the provenance names. It checks the provenance: the model beats the base model it
names, and the lineage starts with the model and names every parent it refers to. CI runs the gate
on every model of a pull request (`brut-ingest --models`). The gate reads only regular files of at
most 16 MiB, and its log lines escape what a file wrote. CI also refuses a mutation pull request
that changes or removes a file that the base branch holds. The content id of a model hashes its
genome alone, so a file with a rewritten provenance would keep its name. The script refuses such a
file before it pushes. The weights are not promoted: a later run trains the genome from its own
seed, as it trains every child (above).

**Later runs start from the merged models.** `brut init --promoted candidates/models` copies the
merged models into the config of a new run (`Config::promoted`), so the run directory holds its
whole experiment. Generation 0 holds them after the base models of the families and before the
random genomes, under the generator name `promoted`. A model of a kind the run does not breed, or
one that fails the run's shape check and limits, stays out with a log line, as a base model of a
family does. A promoted model draws nothing from the random stream. So the random genomes stay as
they were, and the run stays reproducible byte for byte from its seed
(`a_run_with_promoted_models_is_reproducible_byte_for_byte` in `tests/e2e.rs`). In a later run a
promoted model is a base model. Its descendants must beat it to be promoted in turn, so the pool
grows only by models that beat what a run started from.

**The hidden score chooses, and a later run judges anew.** The mutation path keeps the reviewer
blind to held-out scores (ADR 0004). A promotion is chosen by the hidden split of its run by
design: the spec and ticket 07 ask for it. The run is final by then (`evolve::check_not_final`),
so its hidden data no longer steer it. The file records the hidden scores, and the reviewer sees
them. A later run draws a new hidden seed (`brut init`), so its hidden scores come from data that
no promotion chose on. A model promoted on one task may enter a run on another task where it
fits; there it trains on that task like any other candidate.

Rejected: operators weighted by how often they helped (no data yet, and a uniform list is easier
to check); several operators per child (a child that changed in two places hides which change
helped); one family gene for the whole genome (a switch of all stages at once loses the mixed
forms that the spec asks for); crossover by innovation numbers as in NEAT (our genomes are lists
of stages, and their resolutions align them); operators that mend broken genomes (a run never
trains one, and mending makes the operators harder to test); inherited weights (above). For the
promotion: a choice by validation score (the run selected on it, so it is optimistic); a refill
of a place that a duplicate leaves free (a second promotion of one run would then add models that
were not among its two best); promoted weights (they would tie a later run to the data and the
candle commit of the earlier one); a run that reads the pool when it starts (it would depend on
files outside its directory, which can change between a run and its rerun); an MLP run with only
base genomes that promotes (its hidden set would then hold one more candidate than before CNNs, so
an MLP run of an old config would spend one more hidden query and change its archive).

Status: accepted on 2026-10-31. The owner approved the mutations, the crossover and the base models
of the spec on 2026-10-30. Claude, the AI assistant, chose the details in ticket 05: one operator
per child, the three added operators, the rule for redraws, the rule for cuts, the 1-D twins and
the model kinds of a run. The owner asked for the promotion of winners on 2026-10-30 (spec,
ticket 07). Claude chose its details in ticket 07: the best base model by validation fitness as the
bar, base genomes of the config and promoted models counted as base models, no refill, the gate
under the current default limits, and the copy of the pool into the config. The review of
2026-10-31 narrowed the promotion to runs that breed CNNs or hold promoted models, so that MLP
runs keep their bytes. It also made CI refuse changed files and the gate refuse files that are
no small regular files, and the gate escapes its log lines.
