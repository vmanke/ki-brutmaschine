# Step 1: breeding on quantum circuits with an exact checker

German original: [SCHRITT1_SCHALTUNGEN.md](SCHRITT1_SCHALTUNGEN.md).

## 1. Purpose

Step 1 of the second build order (Q47, tickets 01 to 03 in `.scratch/universum-erfinden/`)
opens the Cascade breeding of Stage 2c to a second kind of task: quantum circuits over the
gates H, S, S†, T, T†, X, Z and CNOT, whose correctness an exact simulator decides. Later
the breeding is to invent: circuits that are exactly correct, beat a pre-registered
reference and are new (Q38). First we check, with a positive control, whether it
rediscovers the best known T and CNOT counts that its seed Cascades miss (Q40). Every Run
also writes observables after each generation, so we can study it as a universe (Q32,
Q33). The decisions are in
[ADR 0006](adr/0006-cascades-breed-on-domains-and-circuits-are-checked-exactly.md), the
pre-registered experiment in [PREREG_STEP1_CIRCUITS.md](PREREG_STEP1_CIRCUITS.md), the
pilot Runs in [pilot-step1/](pilot-step1/README.md), and the terms in `CONTEXT.md`.

## 2. What step 1 built

- **One breeding loop for every kind of task.** The evaluator side of a kind of task
  (`Domain`, `src/cascade/domain.rs`) makes the panels, writes the prompts, reads and grades
  the answers, and says which of two answers is better. It never sees a Cascade, so no
  Cascade can change how it is judged. The breeder side (`GenePool`) supplies the seed
  Cascades, the phrases of the section operators, the Mutator's brief and an admission
  rule. Grid puzzles and circuits share one loop. Golden tests, recorded on the grid code
  before the change, pin a grid Run, a Run with LLM rounds, the Mutator's request keys and
  the settings hashes; the only new line in a grid log is the `state` line
  (`tests/universe.rs`).
- **Universe observables** (ticket 01, `src/cascade/observe.rs`). After every committed
  generation, every Run writes one `state` line to its log (section 4).
- **Exact simulator** (ticket 02, `src/cascade/qubit.rs`). It holds every amplitude as
  x/√2^k with x in Z[ω], ω = e^(iπ/4), in four 64-bit integers, and applies every gate in
  integer arithmetic. "Correct" means exactly equal up to one global phase; there is no
  tolerance. We only report the fidelity, because the threshold F ≥ 1 − 1e-9 of Q38 admits
  approximations: by our calculation, a two-qubit circuit whose only error is a relative
  phase of 7e-5 rad on one basis state still reaches it. The caps are code: 12 qubits for
  states, 6 for whole unitaries, 256 gates and 120 H gates. With at most 120 H gates every
  coefficient stays below 2^60, so no 64-bit integer overflows; every operation is checked
  anyway.
- **A second simulator as a cross-check.** `src/quantum.rs` (PR #12) computes in floating
  point and shares no code with `qubit.rs`. Its qubit 0 is the least significant bit, the
  one of `qubit.rs` the most significant, so the tests map qubit q to n − 1 − q. In
  `tests/circuits.rs` both simulators decide 900 pairs of random circuits on 1 to 4 qubits
  alike: 300 pairs are equal by construction, 300 unequal, and 300 drawn independently. For
  all 15 witness circuits of the library both give the same matrix within 1e-12.
- **Target library and tasks** (ticket 03, `src/cascade/circuit.rs`). 11 targets carry T
  and CNOT counts that the research read in the literature (table 1). Four more (W gate,
  √SWAP, quantum OR, 1-bit adder) have no verified reference; they never enter a panel and
  serve only as test vectors of the checker. Every target is built from its definition,
  never from a circuit. A task is a disguised target: its qubits are relabelled, and X
  gates sit on its inputs and outputs. The disguise keeps the optimal T and CNOT counts,
  but a memorised circuit no longer fits. A prompt shows only the action on every basis
  state, never a name, id, source or reference. The search panel has 33 tasks (three
  disguises per target), the select panel 22 (two per target).
- **Scoring.** An inexact circuit gets 0, an exact one 0.5 + 0.5/(1 + ΔT + ΔC/(1 + ΔC)),
  where ΔT and ΔC are the T and CNOT counts above the reference (0 when they do not exceed
  it). So one more T gate always costs more than any number of extra CNOT gates, and the
  value is 1 exactly when the circuit reaches both reference counts: then the task counts
  as solved. Fitness takes the mean value instead of the solved rate, so it is
  0.8 · mean value + 0.2 · judging skill − cost. A judge is scored when the Domain strictly
  prefers one of the two answers.
- **Integrity alarm and invention candidate.** An exact answer below a proven or derived
  bound (Proven, Derived) stops the Run with an integrity error, because then the checker
  or the reference is wrong; the generation is not committed. An exact answer below a best
  known count (BestKnown: 7 CNOT for Fredkin, 4 CNOT for CCiX) is an invention candidate;
  `rediscovery.json` records the cheapest one per target, with
  `literature_checked = false`. We call it new only after a human has checked the
  literature (Q38).
- **Positive control** (`src/cascade/designer.rs`, `src/cascade/circuit_genes.rs`). No
  language model writes circuits in the container. A simulated designer under the model
  tags `sim:small` and `sim:large` answers the circuit prompts (section 5). The genes of
  circuit breeding are four seed Cascades, 50 phrases, the Mutator's brief and the
  admission rule `no_gate_lists`: a section with three or more gate mentions such as
  "CNOT 0 1" is refused, so no single section spells out a circuit. Spread over several
  sections, two gates in each, a circuit still passes (`tests/circuits.rs`); against that
  stands the disguise of each task.
- **Program** (`src/bin/brut-breed.rs`). `--domain circuits` starts a circuit Run. A
  finished Run writes `rediscovery.json`, the pre-registered count, before its report
  (section 4). `--verify` recomputes the report and the count from the checkpoint, without
  a model call. `--promote` refuses circuit Runs with exit code 1, because promoting
  circuit Cascades is not part of step 1.
- **Analysis** (`examples/circuit_positive_control.rs`). It reads the finished Runs of
  both arms, paired by seed, finds the pass time of every Run and applies the
  pre-registered rule across seeds (section 6). After each pair it names the next seed or
  gives the verdict. It makes a claim only for the seeds 1, 2, 3, ... without a gap, at
  most 60 pairs, with 1000 generations and the call cap of 13 000 000.
- **Tests.** `tests/circuits.rs` has 17 tests: circuit Runs are reproducible across
  processes and pass `--verify`; a stop and a resume give the same lineage and `state`
  lines as a straight Run; `rediscovery.json` appears only once a select phase is
  complete; 1 and 4 threads give byte-equal files; `--promote` is refused; the line about
  D holds for an uncalibrated Run too; `rediscovery.json` equals a recount from the logged
  answers; a change to the file fails `--verify`, and so does a member of the
  checkpoint's archive that differs from its `born` event in any field the event
  records, or a member too many or too few; `--verify` says why it skips the file; a
  source scan shows that `designer.rs` does not know the library; the admission rule
  counts gate mentions per section; the calibration is pinned, and so are two routes with
  fewer families at the ceiling of the Traits; and the two cross-checks with
  `quantum.rs`.

Table 1: the 11 targets with a verified reference. The full sources of each entry are in
`LIBRARY` in `src/cascade/circuit.rs`. "Proven" and "derived" are lower bounds; a best
known count is only the best published construction.

| Target | Qubits | Action | T | CNOT | Sources |
|---|---|---|---|---|---|
| CZ | 2 | diag(1, 1, 1, −1) | 0 (proven) | 1 (proven) | Amy, Maslov, Mosca, Roetteler 2013 |
| SWAP | 2 | swaps two qubits | 0 (proven) | 3 (proven) | Vatan, Williams 2004 |
| CS | 2 | diag(1, 1, 1, i) | 3 (proven) | 2 (derived) | Amy et al. 2013; Gheorghiu, Mosca, Mukhopadhyay 2022 |
| CH | 2 | controlled H | 2 (derived) | 1 (proven) | Amy et al. 2013 |
| QFT2 | 2 | quantum Fourier transform on two qubits | 3 (proven) | 3 (proven) | Gheorghiu et al. 2022; Shende, Bullock, Markov 2004 |
| CCZ | 3 | diag(1, …, 1, −1) | 7 (proven) | 6 (proven) | Gosset et al. 2014; Shende, Markov 2009 |
| Toffoli | 3 | flips c when a = b = 1 | 7 (proven) | 6 (proven) | Gosset et al. 2014; Shende, Markov 2009; Maslov 2016 |
| Peres | 3 | Toffoli, then CNOT from a to b | 7 (proven) | 5 (proven) | Shende, Markov 2009; Mosca, Mukhopadhyay 2022; Di Matteo, Mosca 2016 |
| Fredkin | 3 | a = 1 swaps b and c | 7 (proven) | 7 (best known) | Gosset et al. 2014; Niroula, Nam 2021 |
| RTOF | 3 | Toffoli up to relative phases (Maslov) | 4 (derived) | 3 (proven) | Maslov 2016 |
| CCiX | 3 | doubly controlled iX (Selinger) | 4 (derived) | 4 (best known) | Selinger 2013; Maslov 2016 |

## 3. A circuit Run and how to verify it

```sh
cargo build --release

# Circuit Run, arm A: Darwin Gödel parent choice (pilot seed 101, 60 generations)
./target/release/brut-breed --out runs/c101-A --domain circuits --search 33 --select 22 \
  --children 8 --generations 60 --seed 101 --max-calls 200000

# Control arm R: parents drawn uniformly
./target/release/brut-breed --out runs/c101-R --domain circuits --search 33 --select 22 \
  --children 8 --generations 60 --seed 101 --max-calls 200000 --selection uniform

# Verify the log, the report and rediscovery.json, without a model call
./target/release/brut-breed --out runs/c101-A --verify
./target/release/brut-breed --out runs/c101-R --verify

# The frozen analysis; it makes no claim on pilot seeds
cargo run --example circuit_positive_control -- --a runs/c101-A --r runs/c101-R
```

Arm A chooses parents as the Darwin Gödel Machine does (Zhang et al., ICLR 2026,
appendix C.2), arm R uniformly; otherwise both arms are the same. The seeds 1 to 60 are
reserved for the pre-registered experiment, so the examples use the pilot seed
101. A Run prints one line per generation (best Cascade, fitness,
solved search tasks, archive size, calls), then one line about `rediscovery.json`, and at
the end the best Cascades and the comparison on the select panel. For arm A on seed 101
the line about `rediscovery.json` reads:

```
brut-breed: wrote runs/c101-A/rediscovery.json: the seeds solve 2 of 11 targets, bred Cascades 7 of the other 9; D > 1/2: yes; calibrated
```

On this Run, `--verify` prints "brut-breed: 1362 events, hash chain intact, head
e93ed521…; checkpoint at entry 1362 matches; settings match; report recomputed and equal;
rediscovery recomputed and equal". We ran arm A on seed 101 again: `events.jsonl` and
`rediscovery.json` are byte-equal to the pilot Run's, and the Run took 3.2 s (release
build, 2 threads).

**What `--verify` checks.** As in Stage 2c, it checks the hash chain, the checkpoint's
anchor and the settings. For a finished Run with its `report.json` it rebuilds the Run
from the checkpoint, recomputes the report, and derives the stop reason from the
checkpoint instead of trusting the report ("all generations done" or "call budget
reached"). No hash covers the checkpoint, so for such a circuit Run it also requires the
checkpoint's archive to hold one member per `born` event of the log, in their order, each
with every field its event records (id, Cascade, parents, generation, operator, solved
count, graded value, usage and answers), and `rediscovery.json` to equal the
recomputation from that archive byte for byte. What a member gains in later generations
(children, judgments, fitness, peer rating) is not in its `born` event and is not
checked. If the archive or the file differs, or the file is missing, `--verify` ends with
exit code 1 and a message "integrity: …". It does the same when a circuit Run has a report
that its checkpoint cannot recompute, because the checkpoint is unfinished or lacks
select-panel answers: a circuit Run keeps all of them, so this means the checkpoint was
cut or edited. Otherwise it says why it did not check the file:
"rediscovery.json not checked: the run is not finished", or, for a finished Run whose
report is missing, "… there is no report.json; resume the run to write it".
`timings.jsonl` is not part of the verified record.

**Settings.** Without `--config`, `--domain circuits` takes all 11 targets, disguised,
with at most 64 gates per answer. Other circuit settings come with `--config <file>`: a
complete Run configuration in the format of `config.json`, whose field `domain` reads, for
example, `{"circuits": {"targets": ["cz", "ccz"], "dress": true, "max_gates": 32}}`. The
gate cap may go down but never above 256; the Run refuses an unknown or unverified target.
Resume, kill switch, lock and exit codes work as in Stage 2c
([STAGE2C_CASCADE_BREEDING.md](STAGE2C_CASCADE_BREEDING.md), section 5). `--domain` and
`--selection` define the experiment; a resume refuses them. A resume to more generations
removes `report.json` and `rediscovery.json` until the select phase is complete again.

## 4. What the observables mean

**The `state` line.** After every committed generation, its summary is followed by exactly
one `state` line with six groups:

| Group | Fields | Meaning |
|---|---|---|
| Performance | `solved_best`, `solved_median`, `solved_best_born`, `fitness_best`, `fitness_median`, `quality_best`, `quality_median` | best and median solved rate on the search panel, the best of the newborns, fitness; `quality` is the mean task value and appears only for circuits |
| Diversity | `shapes`, `blocks`, `behaviours`, entropies in bits | shapes (Stage widths, such as "1-2-1"), building blocks (pairs of section name and text), distinct answer lists, entropy of the model tags; `parent_entropy_bits` is the entropy of the next generation's parent draw, and 2 to that power is the effective number of parents |
| Lineage | `depth_max`, `depth_mean`, `branching_mean`, `branching_max`, `leaf_share`, `seed_lineages_alive`, `surviving_share` | depth of the family tree (seed Cascades have depth 0), branching, share without children, seed Cascades with descendants in this generation |
| Honesty | `fakes_planted`, `fakes_caught`, `false_alarms`, `confirmed_collusion` | stays 0 until overseers exist (ticket 05) |
| Judging | `comparisons_new`, `scored_new`, `accuracy_new`, `skill`, `trusted_share`, `vote_weight_mean` | comparisons of this generation, the scored ones and the share judged right; quantiles of judging skill; share with judging skill above 1/2 |
| Cost | `calls`, `prompt_tokens`, `completion_tokens`, `model_cost`, `think_tokens_mean`, `workers_mean`, `stages_mean` | usage of this generation (newborns on the search panel and the peer round) and means over the archive |

The line depends only on committed state, never on cumulative call counts, which also hold
the reservations of interrupted attempts. So a stopped and resumed Run writes the same lines
as a straight one. A median is the lower median. Undefined values are absent rather than
NaN. The wall time per generation goes to `timings.jsonl`, outside the hash chain.

An example from the pilot Runs (`pilot.json`, every tenth generation): in arm A on seed
101 the best Cascade solves 3 of the 33 search tasks in generation 0, 9 in generation 20,
24 in generation 40 and 25 in generation 60; the number of distinct answer lists rises
from 4 to 30. `parent_entropy_bits` rises there from 1.66 to 8.43 bits. In arm R it is
always the base-2 logarithm of the archive size, from 2 bits with 4 seed Cascades to
8.92 bits with 484 Cascades, as a uniform parent choice gives.

**`rediscovery.json`.** For every target (`targets`) it holds the reference with its
status, the number of the target's search tasks (`tasks`), the cost of the cheapest exact
answer of the seed Cascades and of the bred ones (`best_seed`, `best_bred`: T, CNOT, H,
gates, depth, T-depth), whether each reached the reference (`reached_by_seed`,
`reached_by_bred`), the earliest generation of a bred Cascade that solved the target
(`first_generation`), the text of the cheapest exact bred answer (`circuit`; it implements
the disguised task, not the target in its plain form), and the cheapest invention
candidate if there is one (`below_best_known`, `candidate`). Below follow |L| (`library`),
|R_seed| (`reached_by_seeds`), |R_bred \ R_seed| (`reached_by_bred_only`), the share D
(`share`), the gate 3 · |R_seed| ≤ |L| (`calibrated`) and `passes` (calibrated and
D > 1/2). The pre-registration defines these terms in section 7.

**`report.json`.** As in Stage 2c: the best Cascades with their solved search tasks, the
finalists with their solved select tasks (for circuits also `select_quality`, the sum of
the task values), and the best rates of the finalists and of the seed Cascades on the
select panel.

## 5. Calibration of the positive control

**The designer.** It reads only the request. In the system prompt, word stems in English
and German switch on eight method families; the order of their first mention is the
order of the methods. Linear builds CNOT networks for affine permutations, Phase builds
phase polynomials (after Amy and Mosca 2019), Hadamard changes the basis on some of the
qubits, Compose frames the target with one CNOT, Search searches gate sequences
breadth-first, Meet searches from both ends (meet in the middle after Amy et al. 2013),
Simplify cancels and merges, and Previous checks and keeps the answers of earlier Stages.
It knows no circuit, no template and no target list; a source scan in
`tests/circuits.rs` checks that `designer.rs` does not name the library. All work costs
steps, paid before it is done: 16 free steps per call and one step per 4 thinking tokens on
`sim:small`, per 2 on `sim:large`, at most 2^13 = 8 192 steps per call. At 4096 thinking
tokens, the ceiling of the Traits, a call has at most 2 064 steps, so the ceiling binds
only on direct requests. With self-check it verifies a candidate on every basis state,
without it only on the first and the last, so a cheap wrong circuit can fool it. It
answers with the cheapest verified candidate by T, then CNOT, then gate count. All
constants are in the pre-registration, section 4.

**Seed Cascades.** Each is one `sim:small` Worker with 256 thinking tokens, temperature
0.2, 512 answer tokens and no self-check; only its strategy names a family (linear, phase,
hadamard, search). On the 33 search tasks that seed 101 draws, the linear seed solves
SWAP, the phase seed CS, and the other two nothing (`designer_calibration_is_pinned` in
`tests/circuits.rs`). So R_seed = {SWAP, CS}, 2 of 11 targets, and 3 · 2 = 6 ≤ 11 meets
the gate. The same holds in each of the 32 disguises of a two-qubit target and the 384 of
a three-qubit target (`circuit_genes.rs`).

**Strong prompt.** One `sim:large` Worker with 4096 thinking tokens, temperature 0 and
self-check, whose strategy names Hadamard, Phase, Linear, Compose, Simplify, Search and
Meet in this order, solves all 33 tasks at the reference. So no target is out of reach,
and L holds all 11 targets. The prompt passes the genome's rules and the admission rule.
It is one hand-written text, which a Run without LLM rounds cannot write; Cascades of
pool phrases reach every target at its reference, plain and in one disguise
(`pool_phrases_reach_every_target_at_its_best_known_cost` in `circuit_genes.rs`).

**Routes to the targets.** The tests in `designer.rs` fix, for the families listed, the
fewest thinking tokens that give the answer; one thinking token less misses it.

| Target | Families named | Self-check | Model | Fewest thinking tokens | Exact answer (T, CNOT) |
|---|---|---|---|---|---|
| SWAP | Linear | off | `sim:small` | 0 | (0, 3), solved |
| CS | Phase | off | `sim:small` | 0 | (3, 2), solved |
| CCZ | Phase | off | `sim:small` | 8 | (7, 10), not solved |
| CCZ | Phase, Linear | off | `sim:small` | 196 | (7, 6), solved |
| CZ | Search | on | `sim:large` | 104 | (0, 1), solved |
| RTOF, CCiX, Peres, Toffoli | Hadamard, Phase, Linear | on | `sim:large` | 78, 92, 124, 158 | reference |
| Fredkin | Hadamard, Phase, Linear, Compose | on | `sim:large` | 654 | (7, 8), not solved |
| Fredkin | Compose, Hadamard, Phase, Linear, Simplify | on | `sim:large` | 1 566 | (7, 7), solved |
| CH | Search, Meet | on | `sim:large` | 3 124 | (2, 1), solved |

At 4096 thinking tokens on `sim:large`, Hadamard and Linear also solve CZ, Hadamard, Phase
and Linear solve QFT2, and Meet alone solves CH; Phase alone gives only (0, 2) for CZ.

**Phrases.** The 50 phrases fall on the sections `strategy` (25), `residual` (9), `style`
(6), `check` (5), `role` (3) and `format` (2); we counted them in the source. No phrase
names two of Hadamard, Phase and Linear. A seed Cascade knows at most one of these three
families (the search seed knows none), and one random step with a phrase teaches it at
most one more. Of the 704 children that such a step makes of the seed Cascades, 40 solve
one target beyond SWAP and CS (20 CZ, 20 CCZ) and none solves two. So QFT2, Toffoli,
Peres, RTOF and CCiX take at least two steps: the 60 children whose last Worker knows two
of the three have 384 children that know all three, and at the seeds' Traits each of these
solves CZ, QFT2, Toffoli, Peres, RTOF and CCiX. The tests in `circuit_genes.rs` fix these
numbers, for each target plain and in three disguises. The three alone miss Fredkin, which
takes Compose besides (`designer.rs`). With the most thinking, the larger model and
self-check, fewer families suffice for two of these targets: at 4096 thinking tokens on
`sim:large` with self-check, Meet alone solves QFT2 in 4 of its 32 disguises, and
Hadamard, Phase and Simplify solve CCiX in 128 of its 384 (`tests/circuits.rs`). Such a
Worker differs from every seed Cascade in model, thinking budget and self-check, so it,
too, lies several steps away.

## 6. Pilot Runs

Pilot seeds never count. The first were seeds 101 and 102; four later batches follow below
(table 4). In the first pilot we ran both arms on both seeds for 60 generations, with
`--search 33 --select 22 --children 8`, a release build, rustc 1.97.0 and 2 threads. The
program of the commit that holds this report repeats each of these pilot Runs byte for
byte (see Reproducibility below). The default cap of 20 000 calls ended arm A on seed 101
after generation 20 (18 451 calls), so all four Runs use `--max-calls 200000`. Then we
repeated each Run with 200 generations. All numbers come from
[pilot.json](pilot-step1/pilot.json); we checked all nine run directories again with
`--verify` and ran the analysis again.

Table 2: pilot Runs with 60 generations.

| Arm | Seed | Calls | Wall time | R_seed | D | First bred solution per target (generation) |
|---|---|---|---|---|---|---|
| A | 101 | 60 140 | 2.96 s | SWAP, CS | 7/9 = 0.778 | CZ 1; QFT2, Toffoli, Peres, RTOF, CCiX 32; CCZ 55 |
| A | 102 | 53 996 | 2.29 s | SWAP, CS | 2/9 = 0.222 | CCZ 6; CZ 29 |
| R | 101 | 49 716 | 5.77 s | SWAP, CS | 1/9 = 0.111 | CZ 35 |
| R | 102 | 40 484 | 2.37 s | SWAP, CS | 2/9 = 0.222 | CZ 26; CCZ 40 |

Table 3: D by generation in the Runs with 200 generations (`--max-calls 1000000`). Each
of these Runs repeats its 60-generation Run exactly through generation 60, since only the
stop check reads `--generations`.

| Arm | Seed | Gen. 30 | Gen. 40 | Gen. 60 | Gen. 100 | Gen. 200 | Calls | Wall time |
|---|---|---|---|---|---|---|---|---|
| A | 101 | 1/9 | 6/9 | 7/9 | 7/9 | 7/9 | 214 049 | 20.9 s |
| A | 102 | 2/9 | 2/9 | 2/9 | 2/9 | 2/9 | 193 928 | 16.8 s |
| R | 101 | 0/9 | 1/9 | 1/9 | 2/9 | 2/9 | 178 684 | 25.9 s |
| R | 102 | 1/9 | 2/9 | 2/9 | 2/9 | 2/9 | 153 624 | 12.6 s |

- **Calibration.** In every Run the seed Cascades solve exactly SWAP and CS; the gate
  holds. Every Run passes `--verify`; none reports an integrity alarm or an invention
  candidate.
- **First solutions.** In these Runs every first solution came by generation 90, none in
  generations 91 to 200. No Run solved CH or Fredkin.
- **Bottleneck.** In arm A on seed 101, five targets fell in generation 32, the first in
  which one Worker named Hadamard, Phase and Linear together. In the other three Runs no
  such Worker arose in 200 generations. We found these generations (`first_worker_naming`
  in `pilot.json`) outside the program, by a search for the word stems in the section
  texts.
- **Analysis.** For the four Runs with 60 generations, the first version of the analysis,
  with the rule "9 of 10", made no claim, since there were only two seed pairs. D_A > 1/2
  holds on one of two seeds (median 1/2), D_A > D_R in one of two pairs. In the T-only
  variant the seed Cascades also solve CCZ (7 T, 10 CNOT); then D_A is 6/8 and 1/8, D_R
  3/8 and 3/8.
- **Select panel.** In arm A on seed 101 the best finalist solves 16 of the 22 select
  tasks, the best seed Cascade 2. In the other three Runs the best finalist solves 4
  (`report.json` of the pilot Runs).
- **Reproducibility.** We ran arm A on seed 101 again; `events.jsonl` and
  `rediscovery.json` are byte-equal to the pilot Run's. After the review fixes, which
  change only `--verify`, one printed line, the analysis and tests, we ran all nine pilot
  Runs again with the program of the commit that holds this report: `config.json`,
  `events.jsonl`, `snapshot.json`, `report.json` and `rediscovery.json` are byte-equal to
  the pilot's, and all nine pass its `--verify`.

**Later pilot batches.** After the first pilot we ran four more batches, both arms with
the settings of the counted Runs except the seed, the generations and the call cap
(table 4). The rule of the counted experiment was made from their data, so their seeds
never count. The Runs are not in the repository:
`.scratch/universum-erfinden/sequential-rule/pilot.json` holds the values read from them,
and [pilot-step1/](pilot-step1/README.md) gives the pass time of every Run. That is the
first generation in which the bred Cascades solve more than half of the targets that the
seed Cascades miss, here 5 of the 9.

Table 4: later pilot batches. "Passes" means here: by the last generation of the Run; the
pass times are in brackets.

| Seeds | Generations | `--max-calls` | Arm A passes | Arm R passes |
|---|---|---|---|---|
| 101 to 110 | 200 | 2 600 000 | 1 of 10 (32) | 1 of 10 (48) |
| 201 to 210 | 200 | 2 600 000 | 5 of 10 (78 to 177) | 2 of 10 (118, 149) |
| 301 to 309 | 1000 | 13 000 000 | 8 of 9 (7 to 810) | 1 of 9 (380) |
| 401, 402 | 2000 | 26 000 000 | 1 of 2 (10) | 1 of 1 (720) |

- **Incomplete.** The Run A-310 was interrupted after generation 922; it is incomplete and
  left out of every figure. R-310 and R-402 were not run.
- **Calibration.** In all 61 complete Runs the seed Cascades solve exactly SWAP and CS, the
  gate holds, and the Run ended with all generations done. None reported an integrity
  alarm or an invention candidate, and none solved CH or Fredkin.
- **Late passes.** By generation 200, arm A passed on 6 of the 20 seeds 101 to 110 and 201
  to 210; by generation 1000, in 9 of the 11 Runs that went that far. Passes still came in
  generation 810 in arm A (seed 309) and in generation 720 in arm R (seed 401). So the
  first pilot's hint that no target is added after generation 90 does not hold.
- **Selection.** In the 10 pairs whose arms both ran to generation 1000, arm A passed
  first 8 times, arm R once (seed 401), and once neither passed (seed 301).
- **Wall time.** At 2 threads, a Run to generation 1000 took 6.01 to 8.57 min in arm A
  (mean 7.38 min, 11 Runs) and 4.54 to 6.12 min in arm R (mean 5.38 min, 10 Runs), a pair
  10.57 to 14.31 min (mean 12.68 min, 10 pairs).

The settings of the counted Runs follow from the pilot Runs: `--generations 1000` and
`--max-calls 13000000`. At a pass rate of 0.30, as by generation 200, the rule below fails
with probability 1.000, after 9.1 pairs on average; so the ceiling could not stay at 200
generations. 1000 is the longest horizon with pilot data in both arms. By our count from
the call bounds in `arena.rs`, a Run of 1000 generations needs at most 12 723 412 calls,
so the cap ends no counted Run early; the pilot Runs used at most 1 161 738 calls by
generation 1000.

**The rule of the counted experiment.** The experiment decides by itself how many seeds
it needs. The rule, its reasons and its sources are in the pre-registration, section 8.
The counted seeds 1, 2, 3, ... run in this order, each in both arms and always to
generation 1000. A Run passes when its pass time T is at most 1000. After each pair two
evidence values change, both starting at 1:

- E1 grows by the factor 3/2 when arm A passes; otherwise it is halved.
- E2 grows by the factor 3/2 when arm A passes first, and is halved when arm R passes
  first; on a tie it stays the same.

When E1 reaches 20, the primary claim is proven: more than half of all seeds pass by
generation 1000. When E1 falls to 1/20, it has failed. The same thresholds hold for E2.
But we claim the second result, that among pairs without a tie arm A passes first more
often than arm R, only if the primary claim is proven. The experiment ends as soon as the
primary claim fails, as soon as it is proven and E2 has reached a threshold, or after 60
pairs. However it ends, it reports a false claim with probability at most 1/20; for the
primary claim the worst case is exactly 0.0385. The old rule "9 of 10" had
11/1024 = 0.0107. So the new bound is weaker; it matches the 5 % bound of the project's
other tests. It holds only if the seed order is fixed, no seed is skipped or replaced,
and nothing changes after seed 1.

After each pair the analysis runs and names the next seed or gives the verdict. Right after
pair 1 we run seed 1 in both arms again with 1 and with 4 threads; all files but
`timings.jsonl` must be byte-equal. Before the report,
`python3 -I .scratch/universum-erfinden/sequential-rule/reference_rule.py --runs <dir>`
recounts the result independently.

If the pilot rates hold, the primary claim is proven with probability 0.82 to 0.99. We
then expect 17 to 29 seed pairs on average, that is 3.6 to 6.1 hours of computing at
12.68 min per pair. In the worst case it takes 60 pairs: 14.3 hours at the pace of the
slowest pilot pair, 14.7 hours if every pair had the slowest Runs of both arms, so about
15 hours. The check with 1 and 4 threads comes on top.

## 7. Limits

- **Positive control.** The designer is not a language model. Its numbers show whether
  breeding finds genes that we know exist; they say nothing about real models.
- **Counted Runs are still to come.** The owner confirmed six points on 2026-10-30: exact
  equality as the criterion (it tightens Q38), scoring by T first and CNOT second, the
  `state` line in grid Runs too, no promotion of circuit Cascades in step 1, the rule
  across seeds of section 6 in place of the 9-of-10 rule, and the panel sizes 33 and 22.
  The pre-registration is frozen with the same sign-off, at the commit that adds its
  signed-off version to the main branch. There are no counted Runs yet; every number in
  this report comes from pilot Runs.
- **What the pilot shows.** The calibration holds in every finished pilot Run. By
  generation 1000, arm A passed in 9 of 11 Runs and arm R in 2 of 10, and in 8 of the 10
  pairs whose arms both went that far, arm A passed first. The nine Runs of the first
  pilot pass `--verify`.
- **What the pilot does not show.** The pilot Runs estimate the pass rate only roughly:
  9 of 11 has a 95 % confidence interval of 0.48 to 0.98, and the Kaplan-Meier estimate
  over all 31 complete Runs of arm A is 0.710. The early batches differ: by generation 200,
  arm A passed on 6 of the 20 seeds 101 to 110 and 201 to 210, in the later batches in 7
  of 11 Runs (Fisher's test, p = 0.128). A pass hinges on one event, a Worker that names
  Hadamard, Phase and Linear together. So the pass times are widely spread, and a pass
  after generation 1000 counts as a failure. At the rate 0.710 the rule ends as failed with
  probability 0.099 and without a verdict with 0.080; a failed primary claim proves no
  rate of 1/2 or less. So with the current genes the counted experiment may well fail.
  The genes stay as they are; any change would need new pilot Runs and a new
  pre-registration.
- **CH and Fredkin.** No pilot Run solved them. The routes the tests pin to them are long:
  CH falls to Search and Meet from 3 124 thinking tokens on `sim:large`; Fredkin to
  Compose besides Hadamard, Phase and Linear, and to 7 CNOT only with Simplify.
- **Derived bounds.** For four counts the bound is derived: CS with 2 CNOT, CH with 2 T,
  RTOF and CCiX with 4 T each. The count is in a source; that it cannot be beaten, the
  research derives from published results. If an exact answer goes below such a count,
  the Run stops with an integrity alarm.
- **Common-mode error.** The designer and the checker both use `qubit.rs`; an error there
  could fool both alike. Against it stand the cross-check with `quantum.rs`, the targets
  built from their definitions, and the witness circuits, most of them from the
  literature, the others built and checked by the research for step 1.
- **Platform.** The hashes of the first pilot hold for rustc 1.97.0 on x86_64 Linux.
  Elsewhere a floating-point function may round its last bit differently; then the Runs
  differ.
- **No Mutator in the pilot.** The pilot Runs have no LLM rounds (`llm_every` 0). What
  Claude changes as the Mutator on circuits we have not measured.
- **Inventions.** No pilot Run found an invention candidate. A candidate is no invention
  until a human has checked the literature. Since `--verify` refuses a changed
  `rediscovery.json`, `literature_checked` stays `false` in the Run; the result of such a
  check belongs in a report.

## 8. Sources

The research for step 1 collected the sources of the target library and the places in
them; `src/cascade/circuit.rs` lists them per target.

- Amy, Maslov, Mosca, Roetteler: A meet-in-the-middle algorithm for fast synthesis of
  depth-optimal quantum circuits. IEEE TCAD 32(6), 2013, pp. 818–830.
  [arXiv:1206.0758](https://arxiv.org/abs/1206.0758)
- Amy, Mosca: T-count optimization and Reed-Muller codes. IEEE Transactions on Information
  Theory 65(8), 2019.
- Vatan, Williams: Optimal quantum circuits for general two-qubit gates. Physical Review A
  69, 032315, 2004. [arXiv:quant-ph/0308006](https://arxiv.org/abs/quant-ph/0308006)
- Shende, Bullock, Markov: Recognizing small-circuit structure in two-qubit operators.
  Physical Review A 70, 012310, 2004.
  [arXiv:quant-ph/0308045](https://arxiv.org/abs/quant-ph/0308045)
- Shende, Markov: On the CNOT-cost of TOFFOLI gates. Quantum Information and Computation
  9(5–6), 2009, pp. 461–486. [arXiv:0803.2316](https://arxiv.org/abs/0803.2316)
- Gosset, Kliuchnikov, Mosca, Russo: An algorithm for the T-count. Quantum Information and
  Computation 14(15–16), 2014, pp. 1261–1276. [arXiv:1308.4134](https://arxiv.org/abs/1308.4134)
- Mosca, Mukhopadhyay: A polynomial time and space heuristic algorithm for T-count.
  Quantum Science and Technology 7, 015003, 2022.
  [arXiv:2006.12440](https://arxiv.org/abs/2006.12440)
- Di Matteo, Mosca: Parallelizing quantum circuit synthesis. Quantum Science and
  Technology 1, 015003, 2016. [arXiv:1606.07413](https://arxiv.org/abs/1606.07413)
- Gheorghiu, Mosca, Mukhopadhyay: T-count and T-depth of any multi-qubit unitary. npj
  Quantum Information 8, 141, 2022.
  [doi:10.1038/s41534-022-00651-y](https://doi.org/10.1038/s41534-022-00651-y)
- Selinger: Quantum circuits of T-depth one. Physical Review A 87, 042302, 2013.
  [arXiv:1210.0974](https://arxiv.org/abs/1210.0974)
- Maslov: Advantages of using relative-phase Toffoli gates with an application to multiple
  control Toffoli optimization. Physical Review A 93, 022311, 2016.
  [arXiv:1508.03273](https://arxiv.org/abs/1508.03273)
- Niroula, Nam: A quantum algorithm for string matching. npj Quantum Information 7, 37,
  2021, Supplementary Note 1.
  [doi:10.1038/s41534-021-00369-3](https://doi.org/10.1038/s41534-021-00369-3)
- Zhang, Hu, Lu, Lange, Clune: Darwin Gödel Machine: Open-Ended Evolution of
  Self-Improving Agents. ICLR 2026. [arXiv:2505.22954](https://arxiv.org/abs/2505.22954)
