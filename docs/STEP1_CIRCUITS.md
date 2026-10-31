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
pilot Runs in [pilot-step1/](pilot-step1/README.md), and the terms in `CONTEXT.md`. The
counted experiment has run and proved both pre-registered claims, the primary claim at pair
15 and the claim about selection at pair 27 (section 7). Both concern the breeder as a
positive control with the simulated designer; they say nothing about real language models.
The data of the experiment are in [step1-counted/](step1-counted/README.md).

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
  pre-registered rule across seeds (section 7). After each pair it names the next seed or
  gives the verdict. It makes a claim only for the first seeds of the fixed list 1 to 100,
  then 111 to 130, without a gap, at most 120 pairs, with 1000 generations and the call
  cap of 13 000 000.
- **Tests.** `tests/circuits.rs` has 18 tests: circuit Runs are reproducible across
  processes and pass `--verify`; a stop and a resume give the same lineage and `state`
  lines as a straight Run; `rediscovery.json` appears only once a select phase is
  complete; 1 and 4 threads give byte-equal files; `--promote` is refused; the line about
  D holds for an uncalibrated Run too; `rediscovery.json` equals a recount from the logged
  answers; a change to the file fails `--verify`, and so does a member of the
  checkpoint's archive that differs from its `born` event in any field the event
  records, or a member too many or too few; `--verify` says why it skips the file, and it
  refuses a report that its checkpoint cannot recompute; a source scan shows that
  `designer.rs` does not know the library; the admission rule counts gate mentions per
  section; the calibration is pinned, and so are two routes with fewer families at the
  ceiling of the Traits; and the two cross-checks with `quantum.rs`.

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

Arm A chooses parents as the Darwin Gödel Machine does (Zhang et al., ICLR 2026, appendix
C.2), arm R uniformly; otherwise both arms are the same. The seeds 1 to 100 and 111 to 130
are reserved for the pre-registered experiment, so the examples use the pilot seed 101. A
Run prints one line per generation (best Cascade, fitness, solved search tasks, archive
size, calls), then one line about `rediscovery.json`, and at the end the best Cascades and
the comparison on the select panel. For arm A on seed 101 the line about
`rediscovery.json` reads:

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

## 7. The counted experiment

**Result.** The counted experiment ran from 2026-10-30 at 22:32 UTC to 2026-10-31 at
12:44 UTC, by the frozen pre-registration. The rule stopped after 27 seed pairs. The primary
claim (ticket 03) is proven at pair 15. It says that more than half of all seeds pass by
generation 1000. The secondary claim is proven at pair 27. It says that, among pairs without
a tie, arm A passes first more often than arm R. Both claims concern the breeder as a
positive control with the simulated designer; they say nothing about real language models.
The data and scripts are in [step1-counted/](step1-counted/README.md).

The settings of the counted Runs follow from the pilot Runs: `--generations 1000` and
`--max-calls 13000000`. At a pass rate of 0.30, as by generation 200, the rule below fails
with probability 1.000, after 9.1 pairs on average; so the ceiling could not stay at 200
generations. 1000 is the longest horizon that most pilot Runs reached in both arms; only
three went on to 2000. By our count from the call bounds in `arena.rs`, a Run of 1000
generations needs at most 12 723 412 calls, so the cap ends no counted Run early; the
pilot Runs used at most 1 161 738 calls by generation 1000.

**The rule.** The experiment decides by itself how many seeds it needs. The rule, its
reasons and its sources are in the pre-registration, section 8. The counted seeds are the
fixed list 1 to 100, then 111 to 130; it skips the pilot seeds 101 to 110. They run in
this order, each in both arms and always to generation 1000. A Run passes when its pass
time T is at most 1000. After each pair two evidence values change, both starting at 1:

- **E1.** It grows by the factor 3/2 when arm A passes; otherwise it is halved.
- **E2.** It grows by the factor 3/2 when arm A passes first, and is halved when arm R
  passes first; on a tie it stays the same.

When E1 reaches 100, the primary claim is proven: more than half of all seeds pass by
generation 1000. That takes at least 12 pairs. When E1 falls to 1/20, it has failed. The
same thresholds hold for E2. But we claim the second result, that among pairs without a
tie arm A passes first more often than arm R, only if the primary claim is proven. The
experiment ends as soon as the primary claim fails, as soon as it is proven and E2 has
reached a threshold, or after 120 pairs. However it ends, it reports a false claim with
probability at most 1/100; for the primary claim the worst case is about 0.0079. The old
rule "9 of 10" had 11/1024 = 0.0107, so the new bound is slightly stricter. It holds only
if the seed list and its order are fixed, no seed of the list is skipped or replaced, and
nothing changes after seed 1.

**Procedure.** We built the program at the frozen commit
`15cb15100a39db9d815674b0778ab6be30932e56`, with rustc 1.97.0 (2d8144b78 2026-07-07) on
x86_64 Linux (Ubuntu 24.04.5 LTS). This commit was created on 2026-10-30 at 21:50 UTC by
the merge of pull request #19. There, `cargo test --lib` passed (610 tests; none of the 7
ignored ones is a calibration test), and so did `cargo test --test circuits` (18) and
`cargo test --example circuit_positive_control` (31). Before seed 1, the acceptance check
on the pilot pairs 301 to 309 gave exactly the pass times and evidence values that the
pre-registration names in section 6, step 2, in the analysis and in `reference_rule.py`;
the analysis withheld the claim there, as required. Then a driver script
([step1-driver.sh](step1-counted/step1-driver.sh)) ran the pairs in the order of the list,
one process at a time. After each pair it checked both Runs with `--verify`, recorded the
hashes of their five files and called the frozen analysis, which named the next seed. Right
after pair 1, seed 1 ran again in both arms with 1 and with 4 threads. For the counted Runs
the driver did not set `RAYON_NUM_THREADS`; rayon then takes one thread per CPU, and the
container reported 4 CPUs after the experiment.

**Verdicts.**

- **Primary claim (ticket 03): proven at pair 15.** After 15 pairs with 14 passes of arm A,
  E1 = 3^14/2^15 = 145.9646 reached 100 for the first time. At the stop after 27 pairs,
  E1 = 3^24/2^27 = 2104.2640; arm A passed on 24 of the 27 seeds.
- **Secondary claim (selection): proven at pair 27.** At the stop,
  E2 = 3^20/2^25 = 103.9143. Among the 25 pairs without a tie, arm A passed first 20 times
  and arm R 5 times (pairs 5, 9, 11, 20 and 25). In pairs 10 and 19 neither Run passed;
  they count as ties. The secondary claim counts because the primary claim is proven, as
  the fixed sequence of section 8 of the pre-registration requires.

So we make the pre-registered claim (pre-registration, section 11): breeding assembles
prompts and Traits that unlock exact, reference-cost syntheses for most library targets the
seed Cascades miss, in more than half of all seeds within 1000 generations. Darwin Gödel
selection gets there sooner than a uniform choice of parents with the same number of
children: among untied pairs, arm A passes first more often than arm R. Both are claims
about the breeder as a positive control, not about real models, and not about inventions.

**Stop.** The rule stopped after pair 27, by the second of its three stop rules: the
primary claim was proven, and E2 had crossed a threshold. After pair 26, E2 stood at
69.2762. In pair 27 arm A passed in generation 126 and arm R never, so E2 rose by the
factor 3/2 to 103.9143. No pair ran after the stop. We report both evidence values at the
stop, not their largest values along the way.

**Per pair.** Table 5 gives what section 8 of the pre-registration requires per pair. Pair
i has seed i. D is the share of the 9 targets the seed Cascades miss that bred Cascades
solve by generation 1000. "never" means no pass time by generation 1000. X is 1 when arm A
passes, else 0. Y names the arm that passed first. E1 and E2 hold after the pair, rounded
to four decimal places; their fractions are in [analysis.txt](step1-counted/analysis.txt).

Table 5: the 27 counted pairs.

| Pair | R_seed | D_A | D_R | T_A | T_R | X | Y | E1 | E2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | SWAP, CS | 7/9 | 2/9 | 574 | never | 1 | A first | 1.5000 | 1.5000 |
| 2 | SWAP, CS | 7/9 | 6/9 | 199 | 376 | 1 | A first | 2.2500 | 2.2500 |
| 3 | SWAP, CS | 7/9 | 7/9 | 278 | 665 | 1 | A first | 3.3750 | 3.3750 |
| 4 | SWAP, CS | 7/9 | 2/9 | 65 | never | 1 | A first | 5.0625 | 5.0625 |
| 5 | SWAP, CS | 7/9 | 6/9 | 227 | 61 | 1 | R first | 7.5938 | 2.5312 |
| 6 | SWAP, CS | 7/9 | 2/9 | 296 | never | 1 | A first | 11.3906 | 3.7969 |
| 7 | SWAP, CS | 8/9 | 3/9 | 557 | never | 1 | A first | 17.0859 | 5.6953 |
| 8 | SWAP, CS | 7/9 | 2/9 | 211 | never | 1 | A first | 25.6289 | 8.5430 |
| 9 | SWAP, CS | 7/9 | 7/9 | 184 | 126 | 1 | R first | 38.4434 | 4.2715 |
| 10 | SWAP, CS | 2/9 | 2/9 | never | never | 0 | tie | 19.2217 | 4.2715 |
| 11 | SWAP, CS | 7/9 | 7/9 | 56 | 18 | 1 | R first | 28.8325 | 2.1357 |
| 12 | SWAP, CS | 7/9 | 7/9 | 107 | 916 | 1 | A first | 43.2488 | 3.2036 |
| 13 | SWAP, CS | 7/9 | 2/9 | 47 | never | 1 | A first | 64.8732 | 4.8054 |
| 14 | SWAP, CS | 7/9 | 7/9 | 20 | 220 | 1 | A first | 97.3098 | 7.2081 |
| 15 | SWAP, CS | 7/9 | 2/9 | 28 | never | 1 | A first | 145.9646 | 10.8122 |
| 16 | SWAP, CS | 7/9 | 3/9 | 610 | never | 1 | A first | 218.9469 | 16.2183 |
| 17 | SWAP, CS | 7/9 | 7/9 | 204 | 618 | 1 | A first | 328.4204 | 24.3274 |
| 18 | SWAP, CS | 7/9 | 2/9 | 62 | never | 1 | A first | 492.6306 | 36.4912 |
| 19 | SWAP, CS | 3/9 | 3/9 | never | never | 0 | tie | 246.3153 | 36.4912 |
| 20 | SWAP, CS | 3/9 | 7/9 | never | 116 | 0 | R first | 123.1577 | 18.2456 |
| 21 | SWAP, CS | 7/9 | 2/9 | 225 | never | 1 | A first | 184.7365 | 27.3684 |
| 22 | SWAP, CS | 7/9 | 2/9 | 304 | never | 1 | A first | 277.1047 | 41.0526 |
| 23 | SWAP, CS | 7/9 | 7/9 | 158 | 182 | 1 | A first | 415.6571 | 61.5788 |
| 24 | SWAP, CS | 7/9 | 7/9 | 23 | 37 | 1 | A first | 623.4856 | 92.3682 |
| 25 | SWAP, CS | 7/9 | 6/9 | 629 | 439 | 1 | R first | 935.2285 | 46.1841 |
| 26 | SWAP, CS | 7/9 | 2/9 | 923 | never | 1 | A first | 1402.8427 | 69.2762 |
| 27 | SWAP, CS | 7/9 | 2/9 | 126 | never | 1 | A first | 2104.2640 | 103.9143 |

**Error bounds.** By section 8 of the pre-registration, however the experiment ends, each
of the two claims is made falsely with probability at most 1/100, and some claim is made
falsely with probability at most 1/100. This follows from Ville's inequality for test
supermartingales and from the fixed sequence of the two claims; for the primary claim the
worst case is 0.0079. The bounds require that the seed list and its order are fixed, that
no seed of the list is skipped or replaced, that the evidence values follow the order of
the list, and that nothing changes after seed 1. We met these conditions. The seeds 1 to 27
ran in this order. All 54 `config.json` hold exactly the settings of section 6 of the
pre-registration, with the seed of their pair and, in arm R, the uniform parent choice. The
driver ran at the frozen commit at both of its starts. A-12 ran again from scratch on the
same seed after an interruption (deviation 1, below). The assumption that the seeds act as
independent draws cannot be checked. We assume it, because the random streams of a Run are
derived from a hash of its seed.

**Recount.** Before this report, `reference_rule.py`, an independent implementation of the
rule, recounted the result from the run directories
([recount.txt](step1-counted/recount.txt)). It gave the same 27 rows (T_A, T_R, X, Y, E1
and E2), the same stop after pair 27 and the same verdicts as the frozen analysis; there is
no difference. A third script, [make_counted.py](step1-counted/make_counted.py),
recomputes these numbers from `rediscovery.json` once more, compares them with both outputs
and with the recorded hashes, and writes [counted.json](step1-counted/counted.json). It
stops at any difference; on the 54 Runs it ran through. All 270 hashes (27 pairs, 2 arms,
5 files) match the files.

**Void conditions.** None of the six conditions of section 9 of the pre-registration
holds:

1. No counted Run raised an integrity alarm. All 54 ended with exit code 0 and wrote
   `rediscovery.json`.
2. The thread check found no differing byte. `config.json`, `events.jsonl`,
   `snapshot.json`, `report.json` and `rediscovery.json` of seed 1 are, in both arms, at 1
   and at 4 threads equal to those of the counted Runs. We compared the 20 files again after
   the experiment.
3. Every calibration test of section 5 of the pre-registration passed at the frozen commit
   (Procedure, above).
4. Every pair meets the gate 3 · |R_seed| ≤ |L|: in all 54 Runs the seed Cascades solve
   exactly SWAP and CS, and 3 · 2 = 6 ≤ 11.
5. All 54 counted Runs passed `--verify`; the last line ended each time in "report
   recomputed and equal; rediscovery recomputed and equal". The same holds for the four
   Runs of the thread check.
6. All 54 Runs have 1000 generations and the call cap 13 000 000, and all ended with "all
   generations done"; none ended at the cap.

**Descriptive results.** The numbers below carry no verdict (pre-registration, section 8).

- **Median of D_A** (descriptive). Over the 27 pairs it is 7/9 = 0.778. D_A was 7/9 in 23
  pairs, 8/9 once (seed 7), 3/9 twice (seeds 19 and 20) and 2/9 once (seed 10).
- **T-only variant** (descriptive). The seed Cascades then also solve CCZ in every Run, with
  7 T and 10 CNOT, so 8 targets are open. D_A is then 7/8 in 15 seeds, 6/8 in 9, 4/8 once
  (seed 10) and 3/8 twice (seeds 19 and 20); D_R is 7/8 in 6 seeds, 6/8 in 5, 5/8 once
  (seed 2), 4/8 once (seed 13) and 3/8 in 14. So D_A > 1/2 holds in 24 of the 27 seeds,
  D_R > 1/2 in 12. The values per seed are in `analysis.txt` and `counted.json`.
- **First solution per target** (descriptive). Table 6 counts on how many seeds a bred
  member of each arm solved the target, and gives the median of the generation in which it
  first fell; the values per seed are in `analysis.txt`. Only one Run solved CH, A-7 in
  generation 594. No Run solved Fredkin at the reference; exact Fredkin circuits with 7 T
  and 8 to 12 CNOT were found by 15 Runs of arm A and 7 of arm R.
- **Calls to the first hit** (descriptive). We take `calls_used` of the generation in which
  a target first fell. Table 6 gives the median, `counted.json` the value per Run and
  target.
- **Call cap and inventions** (descriptive). No Run ended at the call cap, and no Run found
  an invention candidate.
- **Best single-Cascade coverage** (descriptive). The member of the archive with the most
  solved search tasks solves 12 to 27 of the 33 in arm A (median 27), 9 to 27 in arm R
  (median 12). The best member by fitness solves 9 to 27 in arm A (median 27), 3 to 27 in
  arm R (median 9); in 32 of the 54 Runs it solves as many.
- **Select panel** (descriptive). The best finalist solves 6 to 18 of the 22 select tasks in
  arm A (median 18), 4 to 18 in arm R (median 6). The best seed Cascade solves 2 in every
  Run.
- **Trajectories** (descriptive). The `state` lines of every tenth generation are in
  `counted.json`, per Run.

Table 6: the targets in the 27 seeds per arm (descriptive). The medians hold for the Runs
that solved the target; the calls count to the end of the generation of the first solution.
A star marks SWAP and CS, which the seed Cascades already solve; the columns count bred
Cascades here.

| Target | Seeds solved, A | Seeds solved, R | First generation, median A | First generation, median R | Calls by then, median A (thousands) | Calls by then, median R (thousands) |
|---|---|---|---|---|---|---|
| CZ | 27 | 27 | 12 | 5 | 6.8 | 2.9 |
| SWAP* | 27 | 27 | 1 | 1 | 0.5 | 0.5 |
| CS* | 27 | 27 | 1 | 1 | 0.5 | 0.5 |
| CH | 1 | 0 | 594 | – | 814.7 | – |
| QFT2 | 24 | 12 | 201.5 | 201 | 186.4 | 151.2 |
| CCZ | 27 | 27 | 24 | 37 | 16.4 | 22.2 |
| Toffoli | 24 | 10 | 207.5 | 298 | 195.5 | 241.0 |
| Peres | 24 | 12 | 201.5 | 201 | 186.4 | 151.2 |
| Fredkin | 0 | 0 | – | – | – | – |
| RTOF | 24 | 11 | 201.5 | 182 | 186.4 | 131.3 |
| CCiX | 26 | 15 | 107 | 220 | 102.4 | 171.2 |

**Calls and wall time.** Both are in `counted.json` per Run; here is the summary. A Run of
arm A used 963 146 to 1 409 998 calls (mean 1 080 499), one of arm R 780 867 to 958 373
(mean 853 058). By generation 1000 it was at most 1 409 602 (A-18), far below the bound of
12 723 412 of section 6 of the pre-registration. As wall time we measure the time between
the "start" and the "done" line of a Run in the driver's log, to the second. A Run took
4.87 to 11.18 min in arm A (mean 7.36 min) and 4.33 to 5.92 min in arm R (mean 5.01 min),
a pair 10.30 to 15.95 min (mean 12.37 min). The 54 Runs took 5.57 hours together, the four
Runs of the thread check 33.6 min: A-1 at 1 thread 14.42 min, at 4 threads 8.00 min, R-1
6.23 and 4.95 min. Beforehand we had expected 26 to 46 pairs on average, that is 5.8 to
10.1 hours including 0.42 hours for the thread check; for it we had assumed the duration of
two pairs at 2 threads. We measured 5.57 hours for 27 pairs and 0.56 hours for the thread
check, 6.13 hours together. The pilot Runs ran at 2 threads, the counted ones at the
default, so their wall times are not directly comparable.

**Deviation 1.** A container restart between 2026-10-31 at 01:18:49 UTC and about 09:11 UTC
interrupted Run A-12. It had started at 01:18:49 UTC and had no `report.json` yet. Pairs 1
to 11 were finished and verified, and their 110 hashes still matched after the restart. The
driver started again at 09:20:01 UTC (`driver.log`); `deviations.txt` gives 09:20:04 UTC,
the time of the second check of R-1 in `driver.log`. Following section 6, step 3 of the
pre-registration, the driver deleted the unfinished directory at 09:20:09 UTC and ran A-12
again from scratch on the same seed, until 09:27:40 UTC. No finished pair ran again or
changed. At the restart the driver checked pair 1 again with `--verify` and wrote its ten
hashes a second time into [hashes.txt](step1-counted/hashes.txt); both copies are equal
([deviations.txt](step1-counted/deviations.txt), [driver.log](step1-counted/driver.log)).

**Why 100 and 120 pairs.** On 2026-10-30 the owner first approved the threshold 20 with at
most 60 pairs. Later that day the owner asked for the scientifically load-bearing version
and named no numbers. Claude, the AI assistant that works on this repository, then ran a
review of 16 designs: the thresholds 20, 40, 1024/11 and 100, each with a cap of 60, 80,
100 and 120 pairs. The review computed their exact error rates and power, recounted them
independently, tested them by Monte Carlo simulation and checked the sources. The three
independent judges of the review each favoured a different design: 40 with at most 100
pairs, 1024/11 with 120 and 100 with 120. An adversarial critic recomputed the numbers and
upheld 1024/11 with 120 pairs (bound 11/1024 = 0.0107), on the condition of a fixed list
of counted seeds. Claude chose the threshold 100 with 120 pairs and the seed list above.
The owner confirmed this choice with the squash merge of pull request #19, which froze the
pre-registration at commit 15cb151; only then did seed 1 start.

Three of the 16 designs side by side, from
`.scratch/universum-erfinden/sequential-rule/designs.txt`; the last three columns hold the
primary claim at the Kaplan-Meier rate 0.710:

| Proof threshold, cap | False proof, worst case | Proven at 0.710 | Failed at 0.710 | Undecided at 0.710 |
|---|---|---|---|---|
| 100, 120 pairs (chosen) | 0.0079 | 0.8605 | 0.1144 | 0.0252 |
| 1024/11, 120 pairs | 0.0086 | 0.8620 | 0.1142 | 0.0238 |
| 20, 60 pairs (approved first) | 0.0385 | 0.8211 | 0.0988 | 0.0801 |

Statistically, the chosen design and 1024/11 with 120 pairs are almost the same. Of the
two, the threshold 100 is slightly stricter, and only its bound is round. Four further
reasons support the choice:

- **The level.** The bound of 1 % is one-sided, as the claims are. For confirmatory claims
  like these, a one-sided bound of 5 % is lenient: the regulatory guideline ICH E9 (1998,
  section 5.5) holds it preferable, in regulatory settings, to set the type I error of a
  one-sided test at half the conventional two-sided one, that is at 2.5 %. The bound of
  1 % is stricter than that by a factor of 2.5; by the same halving it matches a two-sided
  test at 2 %. The other pass tests of the project keep their pre-registered one-sided
  5 %.
- **The evidence.** On Jeffreys's scale, as Kass and Raftery (1995) and Vovk and Wang
  (2021) give it, an evidence value above 100 counts as "decisive", one of 20 only as
  "strong". Kass and Raftery's own, slightly modified scale calls 20 to 150 "strong"; on
  it, 20 and every evidence value at a proof fall in one class, since such a value lies
  between 100 and 150.
- **The power.** At the most conservative pilot estimate, the Kaplan-Meier rate 0.710, the
  primary claim is proven with probability 0.86, above the conventional power of 0.80
  (Cohen 1992).
- **The balance of errors.** At the Kaplan-Meier rate, this design gives the smallest
  4 · alpha + beta among the 16, where alpha is the worst-case false proof and
  beta = 1 - power. The weight 4 is the ratio beta : alpha of 4 to 1 (0.20 to 0.05) that
  Cohen (1992) derives from his conventions alpha = 0.05 and power 0.80.

The gain has a limit: the new design proves more often than the former one only above a
pass rate of about 0.67. Averaged over the uncertainty of the pilot rate (Beta(10, 3): 9
passes in 11 Runs with a uniform prior), the chance of a true proof is about the same:
0.827 against 0.829. What the change buys is a false success about five times less likely,
at more computing time.

## 8. Limits

- **Positive control.** The designer is not a language model. Its numbers show whether
  breeding finds genes that we know exist; they say nothing about real models. This holds
  for the two claims of the counted experiment too.
- **Approval and freeze.** The owner confirmed six points on 2026-10-30: exact
  equality as the criterion (it tightens Q38), scoring by T first and CNOT second, the
  `state` line in grid Runs too, no promotion of circuit Cascades in step 1, the rule
  across seeds of section 7 in place of the 9-of-10 rule, and the panel sizes 33 and 22.
  Later that day the owner asked for the scientifically load-bearing version. After a
  review of 16 designs, Claude, the AI assistant, chose the proof threshold 100, the cap
  of 120 pairs and the seed list of section 7. The pre-registration implements these
  points. It was frozen at commit 15cb151, which the squash merge of its pull request #19
  created on the main branch. With that merge the owner confirmed Claude's choice. The
  counted Runs ran only after that (section 7). Numbers from counted Runs appear only in
  sections 1, 7 and 8; all other numbers from Runs come from pilot Runs.
- **What the counted experiment shows.** The primary claim rejects the null hypothesis that
  at most half of all seeds pass by generation 1000. The secondary claim rejects the null
  hypothesis that, among pairs without a tie, arm A passes first at most as often as arm R.
  The rule makes each of the two claims falsely with probability at most 1/100, under the
  assumptions of section 8 of the pre-registration. These assumptions hold when the seeds
  act as independent draws, which we assume.
- **What it does not show.** The experiment does not estimate the pass rate. 24 passes in
  27 seeds is a descriptive count. Since the stop depended on the data, a confidence
  interval for a fixed number of seeds does not hold here, and we give none.
  Nor does the secondary claim say by how many generations arm A passes sooner. E2 reached
  the threshold 100 only at pair 27, where the rule stopped.
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
  probability 0.114 and without a verdict with 0.025; a failed primary claim proves no
  rate of 1/2 or less. If the true rate is only 0.6, the primary claim is proven with
  probability 0.18. So with the current genes the counted experiment might well have
  failed; it proved both claims (section 7). The genes stayed as they were; any change
  would have needed new pilot Runs and a new pre-registration.
- **CH and Fredkin.** No pilot Run solved them. Of the 54 counted Runs only A-7 solved CH,
  in generation 594, and none solved Fredkin at the reference. The routes the tests pin to
  them are long: CH falls to Search and Meet from 3 124 thinking tokens on `sim:large`;
  Fredkin to Compose besides Hadamard, Phase and Linear, and to 7 CNOT only with
  Simplify.
- **Derived bounds.** For four counts the bound is derived: CS with 2 CNOT, CH with 2 T,
  RTOF and CCiX with 4 T each. The count is in a source; that it cannot be beaten, the
  research derives from published results. If an exact answer goes below such a count,
  the Run stops with an integrity alarm.
- **Common-mode error.** The designer and the checker both use `qubit.rs`; an error there
  could fool both alike. Against it stand the cross-check with `quantum.rs`, the targets
  built from their definitions, and the witness circuits, most of them from the
  literature, the others built and checked by the research for step 1.
- **Platform.** The hashes of the first pilot and of the counted Runs
  ([hashes.txt](step1-counted/hashes.txt)) hold for rustc 1.97.0 on x86_64 Linux.
  Elsewhere a floating-point function may round its last bit differently; then the Runs
  differ.
- **No Mutator.** The pilot Runs and the counted Runs have no LLM rounds (`llm_every` 0).
  What Claude changes as the Mutator on circuits we have not measured.
- **Inventions.** No pilot Run and no counted Run found an invention candidate. A candidate
  is no invention until a human has checked the literature. Since `--verify` refuses a
  changed `rediscovery.json`, `literature_checked` stays `false` in the Run; the result of
  such a check belongs in a report.

## 9. Sources

The research for step 1 collected the sources of the target library and the places in
them; `src/cascade/circuit.rs` lists them per target. Zhang et al. describe the parent
choice of arm A (section 3). The last four entries are the sources that section 7 cites
for the thresholds; the pre-registration, section 8, gives the other sources of the rule.

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
- International Conference on Harmonisation: ICH Harmonised Tripartite Guideline E9,
  Statistical Principles for Clinical Trials. Step 4 version, 5 February 1998. A
  regulatory guideline, not peer-reviewed.
- Kass, Raftery: Bayes factors. Journal of the American Statistical Association 90(430),
  1995, pp. 773–795.
  [doi:10.1080/01621459.1995.10476572](https://doi.org/10.1080/01621459.1995.10476572)
- Vovk, Wang: E-values: Calibration, combination and applications. Annals of Statistics
  49(3), 2021, pp. 1736–1754. [doi:10.1214/20-AOS2020](https://doi.org/10.1214/20-AOS2020)
- Cohen: A power primer. Psychological Bulletin 112(1), 1992, pp. 155–159.
  [doi:10.1037/0033-2909.112.1.155](https://doi.org/10.1037/0033-2909.112.1.155)
