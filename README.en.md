# Brutmaschine (English)

This page is the English version of the Stage 2c and step 1 parts of the German
[README.md](README.md). The rest of the project documentation is in German; the code,
its comments, `CONTEXT.md` and the ADRs are in English.

## Cascade breeding (Stage 2c)

Status: built with the simulated model (PR #8); the findings of two adversarial reviews
are fixed (PR #10 and #11); real models follow. Described in
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
is the default. Until the integration of step 1, the program refuses `--domain circuits`
as wrong usage. `--verify` recomputes the report of a finished Run from the checkpoint,
without a model call, including its stop reason.

## Universes that invent (step 1)

Status: in progress. Specification:
[.scratch/universum-erfinden/spec.md](.scratch/universum-erfinden/spec.md), with the
tickets in `.scratch/universum-erfinden/issues/`. Step 1 covers tickets 01 to 03.

Above all, the breeding should **invent**: produce artefacts that an exact checker
confirms as correct, that beat a pre-registered reference, and that are new. The first
field is quantum circuits over the gates H, S, S†, T, T†, X, Z and CNOT. As a positive
control, step 1 first measures whether breeding rediscovers the best known T and CNOT
counts that its seed Cascades miss.

Already on `main`:

- **Universe observables** (ticket 01): the `state` line per generation, see above.
- **One breeder for every kind of task:** the evaluator side of a task kind (`Domain`)
  never sees a Cascade; the breeder side (`GenePool`) supplies the seeds, the phrase
  pool, the Mutator's brief and an admission rule. Grid puzzles and circuits share one
  breeding loop. Grid Runs stay byte-identical except for the `state` line; a golden test
  pins that.
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
  fidelity threshold. It serves as an independent cross-check of the exact simulator.

Still open is the integration: circuit Runs with `brut-breed --domain circuits`, the
rediscovery evaluation `rediscovery.json`, ADR 0006, the pre-registration of the
experiment and pilot Runs. Counted Runs start only after the owner has confirmed six
decisions:

1. Exact equality decides what is correct.
2. Scoring goes by T count first, then by CNOT count.
3. Grid Runs carry the observables line too.
4. Circuit Cascades are not promoted in step 1.
5. Success means: in at least 9 of 10 seeds, breeding finds more than half of the
   targets that the seed Cascades miss.
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
  circuits from the literature must be exact at exactly their T and CNOT counts, and an
  independent reference simulator agrees on 500 random circuits.

## Limits

- The simulated models (`sim:small`, `sim:large`) are positive controls. Their numbers
  show that breeding works, not how good real models would be.
- Step 1: circuit Runs are not yet possible from the command line; they come with the
  integration.
