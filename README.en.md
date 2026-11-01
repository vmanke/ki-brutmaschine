# Brutmaschine (English)

This page is the English version of the CNN, Stage 2c and step 1 parts of the German
[README.md](README.md), with the status, the roadmap and the setup of a GPU VM. The rest
of the project documentation is in German; the code, its comments, `CONTEXT.md` and the
ADRs are in English.

CNNs (convolutional neural networks) are the second model kind of the MLP mode, which the
German README describes; a run can breed them from 16 known families. Parts 1 and 2 are
built: the CNN genome, the families with their base models, the mutations and the
promotion of winners (see "CNN models in the MLP mode"). Part 3, ARC grids as tasks
(ticket 06), comes next.

**Step 2** will give every Worker a hidden, inherited **Soul**, which shows itself only in
the Worker's Character. A judgment then becomes an Observation: it pulls the Souls of both
sides together, as strongly as the observer's **Coupling strength** κ says. The design is
merged (PR #28 and #29), and we have been building it since 2026-10-31.

An illustrated overview is the [Atlas](docs/atlas/index.html), a page in German and English
with diagrams, a speckle simulator and the result of step 1. Open the file in a browser.

## Status and roadmap

Status on 2026-10-31.

**Built**

- **Up to step 1.** The MLP mode, the Learner mode (Stage 1), Cascade breeding with the
  simulated model (Stage 2c, PR #8, #10 and #11) and step 1 with the result of its counted
  Runs (PR #21). The German README describes the first two, this page the others.
- **CNNs, part 1 (PR #24).** The library candle, the CNN genome and image tasks (tickets 01
  and 02; [spec](.scratch/cnn-basismodelle/spec.md),
  [ADR 0007](docs/adr/0007-cnns-train-through-candle-at-a-pinned-commit.md)).
- **CNNs, part 2 (PR #27, #32 and #43).** Tickets 03, 04, 05 and 07: the 16 families
  with one base model each, the mutation operators and the crossover, the base models in
  generation 0 of a run, the command line and the promotion of winners
  ([ADR 0008](docs/adr/0008-cnn-families-enter-a-run-as-base-models-and-each-child-changes-by-one-operator.md));
  see "CNN models in the MLP mode". PR #27 merged an interim state, and PR #32 brought the
  CNN tests up to ticket 04 and fixed GoogLeNet and ConvNeXt.
- **Design of step 2 (PR #28 and #29).** It describes the Soul and the Coupling strength κ
  ([spec](.scratch/seele-und-kappa/spec.md)). PR #28 was the first draft, PR #29 is the
  final version. The questions Q67 to Q74 were decided by their recommended answers on
  2026-10-31: Q67 a, Q68 a, Q69 a, Q70 a, Q71 a, Q72 a, Q73 a and d, Q74 a
  ([questions](.scratch/seele-und-kappa/owner-questions.md), in German). The owner can
  overrule each of these decisions.
- **Real language models (PR #33).** The separate package `backend-ollama` with the program
  `brut-breed-ollama`; see "Real models through Ollama". The first three runs on the H200
  are summarized in [docs/ERGEBNISSE_STUFE2C_ERSTE_LAEUFE.md](docs/ERGEBNISSE_STUFE2C_ERSTE_LAEUFE.md)
  (PR #34, in German).
- **Own VM (PR #26, #30 and #31).** `tools/vm-setup.sh` sets up an Ubuntu 24.04 VM with one
  command; see "Setting up your own GPU VM (optional)".

**Being built**

- **Step 2.** Tickets 01 to 09 and 12, started on 2026-10-31.

**Planned**

- **CNNs, part 3.** Ticket 06, ARC grids as tasks; it comes next.
- **Step 2, comparison.** Pre-registration (ticket 10), pilot (ticket 11) and counted Runs
  (ticket 13).
- **LLMs use CNNs (design written, pull request follows).** F1: a CNN as a Worker in a
  Cascade. F2: a language model as Architect that proposes CNN blueprints. F3: a CNN as a
  tool that the language model calls, at most 3 calls per answer.
- **Chemical networks (research done, PR #37).** Chemical reaction networks as generalized
  neural networks, a later model kind. The note with 74 sources and four tickets for first
  prototypes are in [.scratch/chemische-netze/](.scratch/chemische-netze/map.md).

**Parked**

- **Quantum neural networks.** The idea waits until real quantum hardware is in reach
  ([idea 12](.scratch/universum-erfinden/issues/12-quantum-neural-networks-later.md)).

## CNN models in the MLP mode

Status: tickets 01 to 05 and 07 of `.scratch/cnn-basismodelle/` are built, the promotion
of winners included. The ARC grids (ticket 06) follow.
[ADR 0007](docs/adr/0007-cnns-train-through-candle-at-a-pinned-commit.md) records the
library, and
[ADR 0008](docs/adr/0008-cnn-families-enter-a-run-as-base-models-and-each-child-changes-by-one-operator.md)
the families, their mutations and the promotion.

The MLP mode breeds convolutional networks (CNNs) next to MLPs. A CNN genome is a kit: a
stem, one to twelve stages and a head. Each stage holds the block of one of 16 published
families, such as the residual block of ResNet or the Inception block of GoogLeNet. The
families are special cases of the kit. A network whose stages come from several families
is a mixed form. Before each training, the shape check tests whether the shapes fit the
task, and whether the parameters, the training FLOPs and the estimated training time fit
the limits that MLPs and CNNs share. CNNs train on the CPU through the library candle.

### Starting a CNN run

```sh
cargo build --release

# A run with CNNs; without --task, brut init takes the shapes task for it
./target/release/brut init runs/cnn --models cnn --seed 1
./target/release/brut run runs/cnn
./target/release/brut status runs/cnn
./target/release/brut lineage runs/cnn <candidate-id>
```

`--models` chooses the kinds of model of a run: `mlp` (the default) breeds MLPs only, as
before; `cnn` CNNs only; `both` the two side by side. CNNs need a task with images or
sequences:

| Task | Input | Output |
|---|---|---|
| `shapes` | 16x16 image of a shape: square, circle, triangle or cross | 4 classes |
| `bars-and-stripes` | 16x16 image of rows or of columns | 2 classes |
| `blob-count` | 16x16 image with 1 to 4 blobs | 4 classes |
| `mirror-symmetry` | 16x16 image, mirror symmetric or not | 2 classes |
| `motif-1d` | sequence of 64 symbols out of 4, one channel per symbol | 2 classes: motif present or not |
| `mirror-grid` | 8x8 grid | its mirror image, 64 values, scored by R² |

An MLP reads the same data as a flat vector. For a run with CNNs, `brut init` writes a
smaller run: 8 random genomes, 8 children per generation, 5 generations and at most one
hour of run time. A CNN trains for seconds where an MLP trains for milliseconds. An MLP
run stays byte for byte what it was before CNNs; `tests/mlp_golden.rs` checks this with
unchanged fixtures.

### Families and base models

The 16 families are LeNet-5, AlexNet, VGG, GoogLeNet, ResNet, DenseNet, Xception,
MobileNet, MobileNetV2, ShuffleNet, SqueezeNet, SENet, ConvNeXt, U-Net, the dilated CNN
and TCN; the spec `.scratch/cnn-basismodelle/spec.md` names their sources. Each family has
a scaled-down base model that fits the default limits (at most 20 000 parameters and
2e10 training FLOPs). It keeps the block of its family, not the size of the published
network.

Generation 0 of a run with CNNs holds each base model that passes the shape check on the
run's task, before the random genomes. On a sequence task, a 2-D base model enters as its
1-D twin: the same genes, with each kernel along the sequence. A base model that does not
fit stays out, with a line in the archive.

| Tasks | Base models in generation 0, default limits |
|---|---|
| the four image tasks | 14: all but TCN (needs a sequence) and U-Net (needs a grid output) |
| `motif-1d` | 15: TCN and the 14 twins of the 2-D base models |
| `mirror-grid` | 3: U-Net, ResNet and SqueezeNet |

### Mutation and crossover

A child of one CNN gets exactly one operator, drawn uniformly at random, as in Real et
al. 2017 (ICML):

| Operator | Change |
|---|---|
| `family` | the family (block kind) of one stage |
| `insert-stage`, `remove-stage` | one stage more or less |
| `width`, `kernel`, `stride`, `dilation`, `groups` | width, kernel or stride of one stage or of the stem; dilation or groups of one stage |
| `se`, `residual` | squeeze-and-excitation or the skip connection of one stage, on or off |
| `norm`, `downsampling`, `activation` | the normalization, the way to downsample, or the activation |
| `head`, `head-layers` | the head and its dense layers |
| `training` | the training genes, such as learning rate and epochs, as for an MLP |

The last two rows, `head`, `head-layers` and `training`, go beyond the list of the spec,
so that every gene can change. A child that fails the shape check, or keeps the
architecture of its parent, is drawn again, up to 16 times; then the next operator comes.
We measure 10 000 draws per operator: every child was valid, and 74 to 100 % of the draws
gave one (`every_operator_keeps_the_genome_valid_on_10_000_draws`).

Two CNN parents exchange whole stages where their resolutions match. The child keeps the
stem and the outer stages of one parent and puts a run of the other parent's stages in
between, a run that starts and ends at the same resolutions. A child that ends with the
other parent's tail takes its head as well. We measure the 182 ordered pairs of base
models of different families on `shapes`: 172 of them exchange stages. The other 10 pair
MobileNetV2, whose stem shrinks to 4x4 at once, with five families: their children exceed
the time budget, or they share no resolution with it. Such a pair gives a child with the
architecture of one parent and the training genes of both.

The archive names how each child came about: `base-model/resnet` for a base model,
`cnn-mutator-v1/insert-stage` for an operator, `cnn-mutator-v1/crossover` for exchanged
stages. A child trains from scratch, with the weights its own seed draws; it inherits no
weights.

### A demo run with CNNs

We measured a run with the values of `brut init runs/cnn --models cnn --seed 1` (the
shapes task, release build, `RAYON_NUM_THREADS=2` on 4 cores, with the hidden evaluation):

| Quantity | Value |
|---|---|
| Candidates | 62: 14 base models, 8 random CNNs, 40 children in 5 generations |
| Status | all 62 trained to their end, none stopped by the clock |
| Validation score of the base models | 0.670 (LeNet-5) to 0.939 (DenseNet); chance 0.25 |
| Best validation score of the random CNNs | 0.573 |
| Median per generation | 0.728 in generation 0, 0.791 to 0.828 in generations 1 to 5 |
| Mixed forms | 16 of the 40 children and 4 of the 8 random CNNs have stages from several families; the shared `plain` stage of LeNet-5, AlexNet, VGG and the dilated CNN counts as one |
| Run time | 68.6 s; at one thread and without the hidden evaluation 2 min 9 s |
| Estimated compute | 3.8e11 FLOPs |

The three best by fitness, which the hidden evaluation scored (rounded down to 0.01):

| Candidate | Origin | Valid | Hidden | Gap |
|---|---|---|---|---|
| `784227392e754f63` | base model of DenseNet | 0.939 | 0.93 | +0.009 |
| `4ea1fec7699a4627` | generation 5, operator `head` | 0.934 | 0.94 | -0.006 |
| `bebb51cdbc9bcf2c` | generation 4, crossover | 0.929 | 0.92 | +0.009 |

The second is a mixed form: the stem and the two Inception blocks of GoogLeNet, then a 3x3
stage of VGG with max pooling, global average pooling and a dense layer of 32. It came from
a crossover of the GoogLeNet base model with a descendant of a crossover of VGG and
ShuffleNet; then the operator `head` replaced flattening by average pooling. Of the 40
children, 10 came from crossovers, 5 from `downsampling`, 4 from `kernel`, 3 each from
`groups`, `head`, `se` and `width`, 2 each from `head-layers`, `norm` and `residual`, and 1
each from `activation`, `family` and `remove-stage`.

No child beat the best base model on the validation data in this short run. On the hidden
data the best child lies 0.01 above it, which is one rounding step. So the run shows that
base models, operators and crossover together breed valid and good networks; it does not
show a gain from breeding. A second run at one thread and without the hidden evaluation gave
the same 62 candidates and the same 62 model files, byte for byte. Your hidden scores may
differ, since `brut init` draws their seed at random.

### Promoting the winners

A finished run feeds its winners upward, as base models of later runs (ticket 07,
ADR 0008). It promotes at most two models, MLPs or CNNs: the two best by their score on the
hidden split, which the run never saw while it bred. Only the candidates of the hidden
evaluation have that score, the `hidden_eval_top_k` best by fitness (3 by default). Each of the
two goes only if it beats the best base model of the same run on that split. The base
models of a run are the known designs of its generation 0: the base models of the families,
the promoted models, and the base genomes of the config. Only a run that breeds CNNs
(`--models cnn` or `both`) or starts from promoted models (`--promoted`) can promote. Such a
run also scores the best of its base models, by fitness, on the hidden split. Any other run,
such as an MLP run as before CNNs, keeps its hidden evaluation as it was, base genomes or not,
and promotes nothing. A run without a base model has nothing to beat and promotes nothing
either.

```sh
# 1. Prepare the winners of a finished run with the hidden evaluation
./target/release/brut promote runs/cnn --commit "$(git rev-parse HEAD)"

# 2. Open the draft pull request; a human reviews and merges it
scripts/propose-pr.sh runs/cnn/promote

# 3. After the merge, a new run starts with the promoted models
./target/release/brut init runs/cnn2 --models cnn --seed 2 --promoted candidates/models
./target/release/brut run runs/cnn2
```

`--commit` names the git commit of the code that ran the run. `brut promote` checks only its
form: 7 to 64 lower-case hex digits. Call it from the root of the repository, since it reads
`candidates/models/` as the pool; `--pool` names another. The files go to
`runs/cnn/promote/candidates/models/<id>.json`. The id is the content: the first 16 hex
digits of the SHA-256 of the genome's canonical JSON. Each file holds the genome and its
provenance: the run (the hash of its first record), the task, the seed, the commit, the
scores on the validation and hidden splits, the base model it beats, and the lineage back to
generation 0.

A model that the pool holds already, or that equals a base model of a family, adds nothing.
The third best does not take its place, so a second promotion of the same run adds nothing
new. The pull request only adds files under `candidates/models/`. CI checks each file with
`brut-ingest --models`: it refuses unknown fields, runs the shape check and the current
default limits on the task of the provenance, checks that the provenance is consistent, and
requires the file name to be the id. It reads only regular files of up to 16 MiB. The
script refuses a file that the base branch holds already. CI also refuses a pull request
that changes or removes a file of the base branch. Only a human merges; the run never merges
anything (ADR 0004).

`brut init --promoted` copies the genomes of the pool into the config of the new run. In
generation 0 they come after the base models of the families and before the random genomes,
under the generator name `promoted`. A model of a kind the run does not breed, or one that
fails the run's shape check or limits, stays out with a line in the archive. A promoted model
draws nothing from the random stream, so the run stays reproducible byte for byte from its
seed (`a_run_with_promoted_models_is_reproducible_byte_for_byte`). In the later run it is a
base model itself: its descendants must beat it to be promoted in turn. Weights are not
promoted; the new run trains the genome from its seed on its own data.

We measure the path on three runs on the shapes task: the demo run above (release build; we
promote from a copy of its directory) and two new runs of seed 2, built with `cargo build`
(profile dev, `opt-level` 2). All run at `RAYON_NUM_THREADS=2` with the hidden evaluation.
Another process kept the machine busy, so we compare no run times. None of the promotions is
in the pool; they served the measurement only.

| Run | Candidates | Best base model (valid, hidden) | Best candidate (valid, hidden) | Promoted |
|---|---|---|---|---|
| demo run above, seed 1 | 62 | DenseNet (0.939, 0.93) | DenseNet (0.939, 0.93) | `4ea1fec7699a4627` (0.934, 0.94) |
| seed 2 with `--promoted` | 63 | the promoted `4ea1fec7699a4627` (0.951, 0.94) | `84d5076ce10aab99` (0.969, 0.97) | two children, 0.97 hidden each |
| seed 2 without `--promoted` | 62 | DenseNet (0.940, 0.95) | `5ffbf573c5b4c630` (0.956, 0.96) | that child |

- **Promotion from the demo run.** The mixed form of GoogLeNet and VGG from above is promoted.
  Its lineage holds 7 candidates back to the base models of GoogLeNet, VGG and ShuffleNet. The
  second best by hidden score, `bebb51cdbc9bcf2c` at 0.92, does not beat DenseNet and stays
  out. `brut-ingest --models` accepts the file.
- **A later run with the promoted model.** All 63 candidates trained to their end. On the data
  of seed 2 the promoted model beats all 14 base models of the families, whose best is DenseNet
  at 0.940. 15 of the 40 children descend from it, the best one among them: it came from the
  operators `downsampling`, a crossover with a descendant of DenseNet and VGG, and `residual` in
  generation 5.
- **The same run without the promoted model.** Its best candidate comes in generation 1 from
  the VGG base model, and no better one follows.

This is a single pair of runs. It shows that the path works: a promoted model enters
generation 0, becomes a parent, and the later run promotes its better descendants. It does not
show whether promoted models improve later runs on average. The hidden scores of the two runs
of seed 2 come from different secret seeds, since `brut init` draws one per run; only the
validation scores compare.

### Limits of the CNN models

- **Small models.** The base models are scaled down; they are measured on 16x16 images, not
  on ImageNet.
- **A time model of one machine.** The estimate of the training time comes from one machine
  at one thread. A slower or busier machine can reach the clock (`max_candidate_seconds`,
  30 s); then the run marks the candidate, and its bits depend on the load.
- **Equal bits on one machine only.** The same seed gives the same files at any thread
  count; another CPU may give other bits (ADR 0007).
- **Untuned twins.** The 1-D twins on `motif-1d` are not tuned, and no test asks them to
  learn.
- **A stand-in for ARC.** `mirror-grid` is a regression scored by R²; the ARC grids with a
  colour class per cell follow in ticket 06.
- **No inherited weights.** Each child learns from scratch. That costs compute, but keeps
  runs simple to reproduce.
- **Promotion after one run.** Whether a model is promoted depends on its hidden score in a
  single run, rounded down to 0.01. A lead of one rounding step can be chance; only later runs
  show whether the model holds up as a base model.
- **A commit by hand.** `brut promote` checks only the form of the commit. A run made from a
  working tree with uncommitted changes has no matching commit.
- **The first promoted MLP.** A run of MLPs alone promotes only when it starts from promoted
  models. So the first promoted MLP comes from a run of both kinds, on an image or sequence
  task.

## Cascade breeding (Stage 2c)

Status: built with the simulated model (PR #8); the findings of two adversarial reviews
are fixed (PR #10 and #11); since PR #33 the package `backend-ollama` connects real
models (below). Described in
[docs/STUFE2C_KASKADEN_ZUCHT.md](docs/STUFE2C_KASKADEN_ZUCHT.md) and
[ADR 0005](docs/adr/0005-cascades-are-bred-by-darwin-goedel-selection-and-every-cascade-is-also-a-judge.md).

Here the Brutmaschine breeds **Cascades** of language-model **Workers**. A Worker is a
model with a system prompt and **Traits** (thinking budget, temperature, answer length,
self-check). A Cascade has one to four Stages, with one to three Workers each. Every
Cascade is also a judge. It judges the answers of other Cascades blind, never those of
a parent, child or sibling. The Brutmaschine chooses parents the way the Darwin Gödel
Machine does. Claude can write new Cascades as the **Mutator**.

The same rule holds here: candidates are data. A Cascade is JSON. Its prompt is sent to a
model and never executed. There is no language model in the container. The Workers use a
simulated model as a positive control. Real Runs need a GPU server with Ollama (ADR 0003).

```sh
cargo build --release

# Breeding Run with the simulated model
./target/release/brut-breed --out runs/k1 --seed 1 --generations 10

# Resume, optionally with more generations or more calls
./target/release/brut-breed --out runs/k1 --resume --generations 20

# Verify the hash-chained event log and prepare the two best Cascades for a PR
./target/release/brut-breed --out runs/k1 --verify
./target/release/brut-breed --out runs/k1 --promote
scripts/propose-pr.sh runs/k1/promote

# Kill switch: the Run stops before the next puzzle
touch runs/k1/STOP

# Going on after the kill switch: delete STOP first, then resume
rm runs/k1/STOP
./target/release/brut-breed --out runs/k1 --resume
```

### Real models through Ollama (`backend-ollama`)

The separate package `backend-ollama` serves a local model through Ollama
(`docs/STAGE2_BACKEND_CONTRACT.md`); the core crate gains no dependency (ADR 0001). Its
program `brut-breed-ollama` is `brut-breed` plus one thing: every allow-listed tag
`ollama:<model>` gets a backend that talks only to a loopback address, with its own model
allow-list, a timeout and a byte cap. Every Worker call and its full answer go to
`<out>/llm/worker-calls.jsonl`, and a resume replays them instead of asking the model
again (contract I7). Through Ollama only models without a thinking mode are allowed
(`qwen2.5:7b`, `qwen2.5:14b`, `qwen2.5:32b`, `llama3.1:8b`); the thinking model
`qwen3.8-27b` runs through vLLM (below).

```sh
ollama pull qwen2.5:7b
cd backend-ollama && cargo build --release && cd ..
# config.json: a full BreedConfig (for example runs/k1/config.json of a simulated run)
# with "limits": {"models": [{"tag": "ollama:qwen2.5:7b", "cost_per_call": 0.002}], ...}
backend-ollama/target/release/brut-breed-ollama --out runs/q1 --config config.json \
  --seed 1 --generations 2 --children 4 --search 10 --select 10 --max-calls 800
```

Tags of the form `vllm:<name>` go to a vLLM server with an OpenAI-compatible API instead
of Ollama; it decodes the parallel requests of a panel in one batch (on the H200, 40
times the throughput). The server must serve the model under that name:

```sh
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen2.5-32B-Instruct --host 127.0.0.1 \
  --port 8000 --served-model-name qwen2.5-32b --max-model-len 8192
# in config.json: "models": [{"tag": "vllm:qwen2.5-32b", "cost_per_call": 0.002}]
```

`VLLM_URL` (default `http://127.0.0.1:8000`), `OLLAMA_URL` (default `http://127.0.0.1:11434`) and `OLLAMA_TIMEOUT_SECS` (default 300, for both servers); `BRUT_CALL_CACHE` names the answer cache that every run on a machine shares (default `~/.cache/brut/calls.jsonl`, `off` turns it off). A request any run already got answered is not sent to the model again; the run files the answer in its own record with `"cached": true`
set the server and the timeout. The live test:
`cd backend-ollama && OLLAMA_LIVE=1 cargo test --release -- --ignored live`.

**Thinking model `vllm:qwen3.8-27b`.** Qwen3.8-27B is the one model with a thinking mode
on the `vllm:` allow-list. The server must run it with the reasoning parser, so that the
reasoning and the visible answer come back apart:

```sh
vllm serve Qwen/Qwen3.8-27B --host 127.0.0.1 --port 8000 --served-model-name qwen3.8-27b \
  --reasoning-parser qwen3 --max-model-len 32768 --limit-mm-per-prompt '{"image": 0, "video": 0}'
# in config.json: "models": [{"tag": "vllm:qwen3.8-27b", "cost_per_call": 0.002}]
```

The Worker Trait `think_tokens` picks the thinking mode as contract I5 says. The
thresholds live only in the adapter (`backend-ollama/src/vllm.rs`, `EFFORT_LOW_MAX` and
`EFFORT_MEDIUM_MAX`):

| think_tokens | thinking | reasoning_effort | max_tokens sent |
|---|---|---|---|
| 0 | off (`enable_thinking: false`) | none | answer_tokens |
| 1 to 2048 | on | low | answer_tokens + think_tokens |
| 2049 to 3072 | on | medium | answer_tokens + think_tokens |
| 3073 to 4096 | on | xhigh | answer_tokens + think_tokens |

The thresholds rest on a probe we ran on 31 October 2026 without any cap: six calls on
one puzzle with a 3x5 grid, two Worker prompts, two calls per level. The model thought
1921 and 2457 tokens at `low`, 2079 and 3391 at `medium`, 2901 and 4990 at `xhigh`. No
level fits below about 1900 tokens, so `low`, the shortest, takes every budget up to
2048. Above that, each level gets the band whose top it kept in one of its two calls. On
the powers of two that ThinkUp and ThinkDown walk, the step from 2048 to 4096 thus goes
from `low` straight to `xhigh`; only budgets in between reach `medium`. Six calls are a
first setting, not a measurement; we will refit the thresholds once the first runs are
in. In the same probe the visible answer took 348 to 613 tokens at `low` and `medium`
and 35 at `xhigh`, which matters with `answer_tokens` 512.

`max_tokens` caps only the sum. The adapter checks each part on its own: a visible answer
above `answer_tokens`, reasoning above `think_tokens`, or a `finish_reason` of `length`
give `StopReason::Length`, so the Run never scores that answer. vLLM 0.32 reports the
token counts itself (`usage.completion_tokens_details.reasoning_tokens`). Without them,
the adapter estimates both parts by whitespace words and marks the estimate; it counts
low, 0.36 to 0.48 words per token in the probe. The Worker gets only the visible answer.
The reasoning text, both counts, the level and the `estimated` mark go beside each call
into the field `thinking` of `<out>/llm/worker-calls.jsonl`. Sampling follows the model
card: with thinking top_p 0.95 and top_k 20, without it top_p 0.8, top_k 20 and
presence_penalty 1.5; the temperature always comes from the Worker. For the `qwen2.5-*`
models the request stays byte for byte as before. The live test:
`cd backend-ollama && VLLM_THINK_LIVE=1 cargo test --release -- --ignored live_think --nocapture`.

The first three runs on an H200 (`qwen2.5:7b` and `qwen2.5:32b`, 2704 Worker
calls) are summarised, in German, in
[docs/ERGEBNISSE_STUFE2C_ERSTE_LAEUFE.md](docs/ERGEBNISSE_STUFE2C_ERSTE_LAEUFE.md). In
short: the machine runs with real models, a gain from breeding is not measurable with
panels of 12 to 24 puzzles, and Ollama serves the parallel requests one after another
(0.35 s per call for 7B, 1.7 s for 32B).

With `--llm-every K`, every K-th generation is an LLM round. If an answer from the
Mutator is missing, the Run pauses and exits with exit code 3. The requests are then in
`runs/k1/llm/requests/`. Write each answer as `runs/k1/llm/answers/<schlüssel>.txt`, or
let a Claude session do it. Here `<schlüssel>` is the key from the request's file name.
Then continue the Run with `--resume`. If `--llm-answers <dir>` puts the answers
elsewhere, give that option on resume too. Exit codes: 0 = done or stopped cleanly,
1 = error, 2 = wrong usage (also: another process holds the run directory),
3 = waiting for LLM answers.

A local model can answer the requests instead of a person. `brut-mutator` (also in
`backend-ollama`) sends every pending request unchanged to the model of the tag and
writes its answer text as `<run>/llm/answers/<key>.txt`. Every call and its full answer
go to `<run>/llm/mutator-calls.jsonl`, a failed call to `mutator-errors.jsonl`. A
repeated round replays the record instead of asking the model again (ADR 0002). The
program never overwrites an existing answer, and it reads and writes only inside the run
directory; it refuses an `--llm-answers` outside it. Whether an answer is a valid
Cascade is still checked by the core on resume. The script `run-with-mutator.sh` repeats
breeding, answering and resuming until the Run is done:

```sh
backend-ollama/run-with-mutator.sh --tag vllm:qwen2.5-32b --max-rounds 20 -- \
  --out runs/m1 --config config.json --seed 1 --generations 8 --llm-every 2
```

The script resumes with `--resume` and those of the options a resume accepts
(`--generations`, `--max-calls`, `--max-seconds`, `--llm-answers`). It stops when
`brut-breed-ollama` exits with any code but 3 (and passes that code on), when
`<run>/STOP` exists (exit code 0), when a Mutator round fails (1), and after
`--max-rounds` rounds (default 20) if the Run still waits then (3). `brut-mutator` on its
own exits with 0 (everything answered, or stopped cleanly), 1 (a call failed; the other
answers stay), 2 (wrong usage, a path outside the run, the lock is held) or 4 (nothing to
answer). `--parallel P` (default 4) sends P requests at once, `--max-requests N`
(default 64) caps one round. An answer that did not end cleanly (for example because it
reached `max_tokens`) is written as an empty file and reported. The core reads every
answer file as a clean stop; it rejects an empty one as unusable and puts a random child
in its place, so it never parses a cut-off answer. `mutator-calls.jsonl` keeps the full
text and the actual stop reason.

`--resume` may change only `--generations`, `--max-calls`, `--max-seconds` and
`--llm-answers`; everything else defines the experiment. Before each generation, the
checkpoint reserves its worst-case number of calls. If the process ends hard (kill,
crash), the resume counts that reservation as spent, so `--max-calls` holds even then.
Details are in [docs/STAGE2C_CASCADE_BREEDING.md](docs/STAGE2C_CASCADE_BREEDING.md),
section 5.

Since step 1, every Run writes one `state` line to its log after each committed
generation. It holds six observables of the "universe": performance, diversity (with the
Shannon entropy of the building blocks), lineage, honesty, judging and cost. The line
depends only on committed state. So it is the same after a kill switch and a resume as
in a straight Run. The wall time per generation goes to `timings.jsonl`; that file is not
part of the verified record.

```sh
# Control arm: choose parents uniformly instead of by Darwin Gödel selection
./target/release/brut-breed --out runs/k2 --seed 1 --generations 10 --selection uniform
```

`--domain` and `--selection` define the experiment; a resume refuses them. `--domain grid`
is the default; `--domain circuits` starts a circuit Run (step 1, below). `--verify`
recomputes the report of a finished Run from the checkpoint, without a model call,
including its stop reason.

### Control arm: compute-equal majority vote (`vote-control`)

A lead of the best bred finalist mixes breeding with compute, since it often makes more
Worker calls per puzzle than the seed Cascades
([docs/STAGE2C_CASCADE_BREEDING.md](docs/STAGE2C_CASCADE_BREEDING.md), "Not
compute-matched"). The program `vote-control` of `backend-ollama` computes the control
arm with as many calls: self-consistency after Wang et al. (2022), that is, asking one
seed Cascade N times and keeping the most frequent grid, cost-equal as in arXiv
2606.13003 (preprint).

- **Seed Cascade.** By default the seed finalist with the best search fitness. The choice
  reads no select-panel result.
- **N.** The champion's (the bred finalist with the best search fitness) average Worker
  calls per search puzzle, divided by the seed Cascade's, rounded, at least 1. A champion
  with 3.18 calls per puzzle gives N = 3.
- **Panel.** The program regenerates the select panel exactly as the arena does and
  checks it against the checkpoint and the first line of the log.
- **Votes.** Every sample runs with its own seed through the core's `run_cascade`, so with
  the prompts and the parser of the Run. The most frequent grid wins, ties to the one seen
  first. It is scored exactly against the hidden test output.

```sh
# The plan, without a model call (works on an unfinished Run too)
backend-ollama/target/release/vote-control --run runs/q1 --dry-run
# The control arm; the result goes to stdout and to runs/q1/vote-control.json
backend-ollama/target/release/vote-control --run runs/q1
```

Per puzzle, the result lists the votes, whether the majority grid is right, and whether
any sample was. Beside them stand the answers of the champion and of the seed Cascade from
the Run's select phase. The totals hold the discordant pairs against the champion with an
exact sign test, and the calls and tokens. `--samples N` fixes N, `--cascade ID` picks
another archive member, `--temperature T` sets the temperature of every Worker (another
Cascade, with its own id), `--puzzles K` scores only the first K puzzles as a smoke test
and writes no file, `--jobs J` (default 16) sets the samples run at a time. Real models go
through `Recorded` into `<run>/vote-control/worker-calls.jsonl`, apart from the Run's own
record. A repeat replays them, and a larger N reuses the samples of a smaller one. Exit
code 2 means wrong usage, also for an unfinished Run.

- **Failed calls.** A failed call counts as an abstention, is filed in
  `worker-errors.jsonl` beside the record and sets exit code 1; running again retries it.
  So does an answer that cannot be filed, for example on a full disk. Such a result is
  incomplete (`complete` is false): it does not go to `vote-control.json` but to stdout
  and to `<run>/vote-control/incomplete.json`, so an earlier complete result stays.
- **Only healthy Runs as a baseline.** The Run's own select answers are the baseline,
  while the samples run with 63-bit seeds and never fail for that reason. Runs from before
  d79c93f (63-bit seed, #40) therefore cannot be used as a baseline: vLLM refused every
  call with a seed above i64::MAX there, about half of them, without a record. In
  `v-dgm-11` we measure 3475 recorded calls of 6966, and none of the 3475 seeds lies above
  i64::MAX. `vote-control` refuses such a Run with exit code 2, `--dry-run` included, and
  likewise a Run with entries in its own `llm/worker-errors.jsonl`. A Run counts as old
  when it allows a `vllm:` model and its `llm/worker-calls.jsonl` holds at least 32 vLLM
  calls, none with a seed above i64::MAX; a new Run looks like that with probability
  2^-32. `--allow-degraded-run` computes anyway, and the plan and the result say so in
  `run_failed_calls` and `predates_seed_fix`.

## Setting up your own GPU VM (optional)

Real Runs with language models need a GPU server with Ollama (ADR 0003). The script
`tools/vm-setup.sh` sets up a fresh Ubuntu 24.04 VM with one command (PR #26, extended in
PR #30 and #31). It does this:

- **Waiting.** It waits until the first-boot setup (cloud-init) and the automatic updates
  are done.
- **Packages.** Build tools, git, curl, python3, tmux and ufw, and the GitHub CLI `gh` from
  GitHub's own package source.
- **Firewall.** Only SSH is reachable from outside (ufw). If Docker is installed, Docker
  then publishes ports on 127.0.0.1 by default, because ufw does not filter the ports that
  Docker publishes. Networks of docker compose and ports given explicitly as `0.0.0.0` are
  not covered.
- **Tools.** Rust 1.97.0 with clippy and rustfmt, cargo-audit and Claude Code.
- **Repository.** A clone in `~/ki-brutmaschine`, and signed commits, because `main`
  accepts only signed commits. The script creates the SSH key for this and registers it
  with GitHub. The GitHub login needs a single code, which you enter at
  github.com/login/device.
- **Service.** The systemd service `brut-remote-control` starts Claude Remote Control in
  tmux after every boot.

The script installs neither Ollama nor a GPU driver. It stores no secret in the
repository. API keys belong in the file `~/ki-brutmaschine/.env`, which is in
`.gitignore`. The script may run again at any time and skips what is already done.

Run it as your normal user, not as root. On a fresh VM, download it at a pinned commit and
check its SHA-256 sum. Take the commit and the sum from the pull request or release that
names them:

```sh
COMMIT=<commit>
SHA256=<sha256>
curl -fsSL -o vm-setup.sh \
  "https://raw.githubusercontent.com/vmanke/ki-brutmaschine/$COMMIT/tools/vm-setup.sh"
echo "$SHA256  vm-setup.sh" | sha256sum -c - && bash vm-setup.sh
```

If the clone is already in `~/ki-brutmaschine`, for example after an earlier run of the
script, update it and run the script from there:

```sh
cd ~/ki-brutmaschine && git pull && bash tools/vm-setup.sh
```

At the end, the script names three steps that you do once by hand:

1. **Log in again.** Log out with `exit` and connect with SSH again. Only then does the
   shell find `claude` and `cargo`.
2. **Log in to Claude.** Use your claude.ai account: `claude auth login`.
3. **Start Remote Control.** Start the service and attach to its tmux session. Answer the
   first questions of Remote Control there with `y`:

   ```sh
   sudo systemctl start brut-remote-control
   tmux -L brut attach -t brut
   ```

   Leave tmux with Ctrl+b, then d. Remote Control keeps running and restarts by itself
   when it ends.

From then on, the service starts Remote Control after every boot. A later run of the
script does not turn a disabled service back on.

```sh
# State of the service
systemctl status brut-remote-control

# Watch it; leave with Ctrl+b, then d
tmux -L brut attach -t brut

# Turn it off; ending it inside tmux is not enough, because systemd starts it again
sudo systemctl disable --now brut-remote-control
```

The owner's VM is a preemptible Nebius VM at the spot price, with one NVIDIA H200. We
measured 143 771 MiB of GPU memory, 16 vCPU and 196 GB of RAM on it.

**Costs.** At Nebius, billing of the GPU stops only when you stop the VM in the console or
through the API. A shutdown from inside the VM does not stop billing (according to the Nebius
documentation). Disks are billed also while the VM is stopped.

## Universes that invent (step 1)

Status: built; the counted experiment has run. The six decisions below were confirmed on
2026-10-30, decision 5 first with the threshold 20 and at most 60 pairs. Its
scientifically load-bearing version (proof threshold 100, at most 120 seed pairs), whose
numbers Claude chose, was confirmed with the merge of pull request #19, which froze the
pre-registration. Then the counted experiment ran, and the rule stopped after 27 seed
pairs. The primary claim is proven at pair 15, the claim about selection at pair 27
([docs/STEP1_CIRCUITS.md](docs/STEP1_CIRCUITS.md), section 7; data in
[docs/step1-counted/](docs/step1-counted/README.md)). The experiment is a positive control
with the simulated designer and says nothing about real language models.
Described in [docs/STEP1_CIRCUITS.md](docs/STEP1_CIRCUITS.md) and
[ADR 0006](docs/adr/0006-cascades-breed-on-domains-and-circuits-are-checked-exactly.md);
the experiment is pre-registered in
[docs/PREREG_STEP1_CIRCUITS.md](docs/PREREG_STEP1_CIRCUITS.md). Specification:
[.scratch/universum-erfinden/spec.md](.scratch/universum-erfinden/spec.md), with the
tickets in `.scratch/universum-erfinden/issues/`. Step 1 covers tickets 01 to 03.

Above all, the breeding should **invent**: produce artefacts that an exact checker
confirms as correct, that beat a pre-registered reference, and that are new. The first
field is quantum circuits over the gates H, S, S†, T, T†, X, Z and CNOT. As a positive
control, step 1 first measures whether breeding rediscovers the best known T and CNOT
counts that its seed Cascades miss.

Built:

- **Universe observables** (ticket 01): the `state` line per generation, see above.
- **One breeder for every kind of task:** the evaluator side of a task kind (`Domain`)
  never sees a Cascade; the breeder side (`GenePool`) supplies the seeds, the phrase
  pool, the Mutator's brief and an admission rule. Grid puzzles and circuits share one
  breeding loop. Golden tests, recorded on the grid code before the change, pin a grid Run
  and a Run with LLM rounds (without floating-point numbers), the Mutator's request keys
  and the settings hashes; the only new line in a grid log is the `state` line.
- **Exact simulator** (ticket 02, `src/cascade/qubit.rs`): it holds amplitudes exactly as
  x/√2^k with x in Z[ω], ω = e^(iπ/4), in 64-bit integers. "Correct" means exactly equal
  up to one global phase; the fidelity is only reported. Caps: 12 qubits for states, 6
  for whole unitaries, 256 gates and 120 H gates.
- **Circuit tasks** (ticket 03, `src/cascade/circuit.rs`): 11 targets with T and CNOT
  counts from the literature: CZ, SWAP, CS, CH, QFT2, CCZ, Toffoli, Peres, Fredkin, RTOF
  and CCiX. Every count carries its status: proven, derived or best known.
  - A task is a disguised target: its qubits are relabelled, and X gates sit on its
    inputs and outputs. A prompt never names a target.
  - Correctness is scored first, then the T count, then the CNOT count.
  - An exact answer below a proven or derived bound stops the Run with an integrity
    alarm.
  - An exact answer below a best-known count is recorded as an invention candidate, with
    `literature_checked = false` until a human has checked the literature.
- **Positive control** (`src/cascade/designer.rs`, `src/cascade/circuit_genes.rs`): a
  simulated circuit designer under the model tags `sim:small` and `sim:large`. The
  methods a system prompt names switch its synthesis methods on; it knows no circuit and
  no target list. The four seed Cascades reach only SWAP and CS. A strong prompt reaches
  all 11 references. Its numbers say nothing about real models.
- **Second simulator** (`src/quantum.rs`, PR #12): a floating-point simulator with a
  fidelity threshold that shares no code with the exact simulator. It is the independent
  cross-check: on 900 pairs of random circuits and the 15 witness circuits of the
  library it decides as the exact simulator does (`tests/circuits.rs`).
- **Circuit Runs** (`src/bin/brut-breed.rs`): `brut-breed --domain circuits`. A finished
  Run writes `rediscovery.json`, the pre-registered count, and `--verify` recomputes it
  from the checkpoint. The frozen analysis is `examples/circuit_positive_control.rs`.

The pilot Runs are in [docs/pilot-step1/](docs/pilot-step1/README.md): the first pilot on
seeds 101 and 102 and four later batches. Pilot Runs never count; the data of the counted
Runs are in [docs/step1-counted/](docs/step1-counted/README.md). The owner confirmed six
decisions on 2026-10-30. Later that day the owner asked for the scientifically
load-bearing version of decision 5. Claude, the AI assistant, chose its numbers after a
review of 16 designs; the owner confirmed them with the merge of pull request #19, which
froze the pre-registration. Decision 5 is shown in that version:

1. Exact equality decides what is correct.
2. Scoring goes by T count first, then by CNOT count.
3. Grid Runs carry the observables line too.
4. Circuit Cascades are not promoted in step 1.
5. The experiment decides by itself how many seeds it needs (sequential rule). A Run
   passes when breeding finds, by generation 1000, more than half of the targets that the
   seed Cascades miss. Success means: more than half of all seeds pass. The counted seeds
   are 1 to 100, then 111 to 130, in this order. After each pair of seeds the analysis
   checks whether the evidence suffices for a verdict; the experiment ends after 120 pairs
   at the latest. However it ends, it reports a false success with probability at most
   1/100; the worst case is about 0.0079. That is slightly stricter than the old rule
   "9 of 10", whose bound was 11/1024 = 0.0107; the sequential rule replaces it.
6. The panels have 33 search and 22 select tasks.

## Building blocks

| Block | Job | Files |
|---|---|---|
| Breeding | Darwin Gödel selection, judging rounds, call budget, checkpoints | `src/cascade/arena.rs` |
| Cascades and variation | genome and limits, mutation and crossover, Mutator requests | `src/cascade/genome.rs`, `src/cascade/variation.rs` |
| Grid puzzles | puzzle generator, prompt protocol, simulated model | `src/cascade/puzzle.rs`, `src/cascade/sim.rs` |
| Seam | task kind (`Domain`) and genes (`GenePool`) | `src/cascade/domain.rs` |
| Observables | the `state` line per generation | `src/cascade/observe.rs` |
| Qubits | exact simulator, text protocol for circuits | `src/cascade/qubit.rs`, `src/cascade/circuit_text.rs` |
| Circuits | target library, tasks, scoring, rediscovery | `src/cascade/circuit.rs` |
| Positive control | simulated designer, seed Cascades, phrase pool | `src/cascade/designer.rs`, `src/cascade/circuit_genes.rs` |
| Program | run directory, hash chain, resume, verify, promotion | `src/bin/brut-breed.rs` |

Everything under `src/cascade/` is pure computation without file, network or clock
access; `tests/cascade.rs` checks that.

## Tests

- `tests/cascade.rs` checks the purity of `src/cascade/` and makes small real Runs with
  `brut-breed`: equal results across processes, `--verify` against tampered logs and
  reports, kill switch and resume, hard kills, the lock and LLM rounds.
- `tests/universe.rs` shows with golden fixtures from before step 1 that the new seam
  leaves grid Runs unchanged. It also checks the `state` lines (exactly one per
  generation, equal after a stop and resume, equal with 1 and 4 threads) and the new
  options.
- The circuit building blocks have unit tests in their own files. Among them: the 11
  witness circuits of the verified targets, most of them from the literature, must be
  exact at exactly their T and CNOT counts, and an independent reference simulator agrees
  on 500 random circuits.
- `tests/circuits.rs` makes real circuit Runs with `brut-breed` (equal results across
  processes and with 1 and 4 threads, stop and resume, `--verify` against a changed
  `rediscovery.json`, refused promotion), pins the calibration of the positive control
  and compares the exact simulator with `src/quantum.rs`.

## Limits

- The simulated models (`sim:small`, `sim:large`) are positive controls. Their numbers
  show that breeding works, not how good real models would be.
- Step 1: the counted experiment is a positive control with the simulated designer. Its
  primary claim says that more than half of all seeds pass by generation 1000; it does not
  estimate how high the pass rate is, and it says nothing about real language models. What
  the experiment and the pilot show and what they do not is in
  [docs/STEP1_CIRCUITS.md](docs/STEP1_CIRCUITS.md), section 8.
