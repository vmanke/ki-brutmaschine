# Brutmaschine (English)

This page is the English version of the Stage 2c and step 1 parts of the German
[README.md](README.md), with the status, the roadmap and the setup of a GPU VM. The rest
of the project documentation is in German; the code, its comments, `CONTEXT.md` and the
ADRs are in English.

CNNs (convolutional neural networks) are coming as a second model kind to the MLP mode,
which the German README describes. Part 1 and an interim state of part 2 are merged. We
are now building the mutations and the promotion of winners.

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
- **CNNs, part 2, interim state (PR #27 and #32).** Ticket 03 is built, ticket 04 is begun.
  PR #32 brought the CNN tests up to ticket 04 and fixed GoogLeNet and ConvNeXt.
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

- **CNNs, part 2.** Ticket 05 (mutations, base models in the population, command line,
  README, ADR 0008) and ticket 07 (promotion of winners).
- **Step 2.** Tickets 01 to 09 and 12, started on 2026-10-31.

**Planned**

- **CNNs, part 3.** Ticket 06, ARC grids as tasks.
- **Step 2, comparison.** Pre-registration (ticket 10), pilot (ticket 11) and counted Runs
  (ticket 13).
- **LLMs use CNNs (design written, pull request follows).** F1: a CNN as a Worker in a
  Cascade. F2: a language model as Architect that proposes CNN blueprints. F3: a CNN as a
  tool that the language model calls, at most 3 calls per answer.
- **Chemical networks (research running).** Chemical reaction networks as generalized
  neural networks, a later model kind.

**Parked**

- **Quantum neural networks.** The idea waits until real quantum hardware is in reach
  ([idea 12](.scratch/universum-erfinden/issues/12-quantum-neural-networks-later.md)).

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
again (contract I7). For now only models without a thinking mode are allowed
(`qwen2.5:7b`, `qwen2.5:14b`, `qwen2.5:32b`, `llama3.1:8b`).

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

`VLLM_URL` (default `http://127.0.0.1:8000`), `OLLAMA_URL` (default `http://127.0.0.1:11434`) and `OLLAMA_TIMEOUT_SECS` (default 300, for both servers)
set the server and the timeout. The live test:
`cd backend-ollama && OLLAMA_LIVE=1 cargo test --release -- --ignored live`.

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
