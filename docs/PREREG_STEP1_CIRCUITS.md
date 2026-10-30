# Pre-registration of step 1: does breeding rediscover circuit references?

Status: proposed; frozen on the owner's sign-off.

This document fixes the step 1 experiment of plan (g) and
[ADR 0006](adr/0006-cascades-breed-on-domains-and-circuits-are-checked-exactly.md) before
any counted Run. Of this experiment, only the pilot seeds 101 and 102 have run
([docs/pilot-step1/](pilot-step1/README.md)); the counted seeds 1 to 10 have not run in
either arm. The owner's sign-off covers this document together with the six open
decisions of ADR 0006, and it names the hash of the commit that adds this file. From then
on nothing below changes; every deviation is reported with the results, with its reason.
The report on step 1 is [SCHRITT1_SCHALTUNGEN.md](SCHRITT1_SCHALTUNGEN.md) (English:
[STEP1_CIRCUITS.md](STEP1_CIRCUITS.md)).

## 1. Question

Does the Stage 2c breeder, working on quantum circuits with the simulated designer as its
model, solve most of the library targets that its seed Cascades miss, exactly and at the
reference T and CNOT counts? And does Darwin Gödel selection (arm A) do better than a
uniform choice of parents (arm R) with the same number of children? This is a positive
control of the breeder. It says nothing about real language models.

## 2. What is frozen

| Item | Frozen as |
|---|---|
| Code | The whole tree of the commit that adds this file. The owner's sign-off names its hash. |
| Toolchain | rustc 1.97.0 on x86_64 Linux, as in the pilot (`docs/pilot-step1/pilot.json`). |
| Program | `cargo build --release --bin brut-breed` at that commit. |
| Analysis | `examples/circuit_positive_control.rs` at that commit. |
| Target list L | The 11 verified library targets (section 3). |
| Designer | The constants of `src/cascade/designer.rs` (section 4). |
| Genes | `src/cascade/circuit_genes.rs`: the 4 seed Cascades, the 50 phrases, the Mutator's brief and the admission rule `no_gate_lists`. |
| Calibration | The results of section 5, pinned by tests. |
| Arms | Section 6. |
| Seeds | 1 to 10, paired between the arms. The pilot seeds 101 and 102 are not counted (section 10). |
| Definitions and rules | Sections 7 to 9. |

The pilot ran commit d4a650e. Between that commit and the commit that adds this file only
documents changed: `git diff d4a650e <frozen commit> -- src tests examples Cargo.toml
Cargo.lock` is empty. On another toolchain or platform a floating-point function may round
its last bit differently, and then the Runs may differ from Runs on the frozen toolchain;
such Runs are a deviation.

## 3. The target list L

L holds the 11 library targets with a verified reference, in library order, which is also
the round-robin order of a panel. The numbers and their sources are in `LIBRARY` in
`src/cascade/circuit.rs`. Proven and Derived numbers are lower bounds; BestKnown numbers
are the best published constructions.

| # | Target | Qubits | T (status) | CNOT (status) |
|---|---|---|---|---|
| 1 | cz | 2 | 0 (Proven) | 1 (Proven) |
| 2 | swap | 2 | 0 (Proven) | 3 (Proven) |
| 3 | cs | 2 | 3 (Proven) | 2 (Derived) |
| 4 | ch | 2 | 2 (Derived) | 1 (Proven) |
| 5 | qft2 | 2 | 3 (Proven) | 3 (Proven) |
| 6 | ccz | 3 | 7 (Proven) | 6 (Proven) |
| 7 | toffoli | 3 | 7 (Proven) | 6 (Proven) |
| 8 | peres | 3 | 7 (Proven) | 5 (Proven) |
| 9 | fredkin | 3 | 7 (Proven) | 7 (BestKnown) |
| 10 | rtof | 3 | 4 (Derived) | 3 (Proven) |
| 11 | ccix | 3 | 4 (Derived) | 4 (BestKnown) |

No target is declared reach-limited, so |L| = 11 in every counted Run. The strong prompt
of section 5 solves all 33 search tasks of the pilot seed 101 panel at the reference
(`designer_calibration_is_pinned` in `tests/circuits.rs`), and every target plain and in
three disguises (`the_strong_prompt_reaches_every_target_cost` in `designer.rs`). The
analysis refuses a Run whose targets are not these 11 in this order.

## 4. The designer's constants

The simulated designer pays for all work in steps before it does it. A call has the free
steps plus one step per `step_tokens` thinking tokens, up to the ceiling. These are the
constants of `src/cascade/designer.rs`:

| Constant | Value | Meaning |
|---|---|---|
| `FREE_STEPS` | 16 | Steps every call has without thinking tokens. |
| `step_tokens` | 4 (`sim:small`), 2 (`sim:large`) | Thinking tokens per step. |
| `MAX_STEPS` | 2^13 = 8 192 | Hard ceiling of steps per call. At the Traits' ceiling of 4096 thinking tokens a call has at most 2 064 steps, so it binds only on direct requests. |
| `STATES_PER_STEP` | 16 | States a network search reaches per step. |
| `MAX_NETWORK_STATES` | 2^18 = 262 144 | Memory cap of one network search. |
| `COLUMNS_PER_STEP` | 16 | Columns of new words a word search computes per step: 4 words on two qubits, 2 on three. |
| `MAX_SEARCH_COLUMNS` | 2^16 = 65 536 | Memory cap of one word search. |
| `SEARCH_QUBITS` | 3 | Most qubits a word search works on. |
| `CLASSES_PER_STEP` | 4 | Classes the table of two-qubit Cliffords gains per step. |
| `TEMPERATURE_CEILING` | 1.5 | The method order is shuffled with probability 0.5 · temperature / 1.5. |

Further step costs, fixed in the same file: verifying a candidate costs one step per
checked column (all 2^n columns with self-check, the first and the last without);
testing whether a target, framed or not, is an affine permutation with phases costs 2^n
steps, and so does the Walsh transform of its phases; a peephole pass, a phase fold and a
scan for Clifford windows cost one step per gate. The reported thinking tokens are the
steps used beyond the free ones, times `step_tokens`. The eight method families and their
stems are `FAMILY_STEMS`.

## 5. The calibration

The tests named here pass at the frozen commit; if one fails there, the experiment is
void (section 9).

**Seeds.** Each seed Cascade is one `sim:small` Worker with 256 thinking tokens,
temperature 0.2, 512 answer tokens and no self-check; its strategy names one family
(linear, phase, hadamard, search). On the 33 search tasks of the pilot seed 101 panel,
with the requests a Run sends, the linear seed solves swap, the phase seed cs, and the
hadamard and search seeds solve nothing (`designer_calibration_is_pinned`). So R_seed =
{swap, cs}, and 3 · 2 = 6 <= 11 passes the gate. The same holds on every target in
every disguise, 32 per two-qubit and 384 per three-qubit target: the linear seed solves
swap in all 32, the phase seed cs in all 32, and no seed solves anything else
(`the_seeds_reach_only_swap_and_cs_on_every_dressing` in `circuit_genes.rs`).

**Strong prompt.** One `sim:large` Worker with 4096 thinking tokens, temperature 0, 512
answer tokens and self-check, whose strategy names Hadamard, Phase, Linear, Compose,
Simplify, Search and Meet in this order, solves all 33 tasks at the reference. It passes
the genome's rules and the admission rule, so breeding could make it.

**Routes.** `one_step_short_misses_and_enough_thinking_hits` in `designer.rs` pins, on
each plain target, the fewest thinking tokens that give the answer below; one token less
does not. The two rows with 0 tokens fit the free steps.

| Target | Families named | Self-check | Model | Fewest thinking tokens | Exact answer (T, CNOT) | Reference |
|---|---|---|---|---|---|---|
| swap | Linear | off | `sim:small` | 0 | (0, 3) | (0, 3) |
| cs | Phase | off | `sim:small` | 0 | (3, 2) | (3, 2) |
| ccz | Phase | off | `sim:small` | 8 | (7, 10), not solved | (7, 6) |
| ccz | Phase, Linear | off | `sim:small` | 196 | (7, 6) | (7, 6) |
| cz | Search | on | `sim:large` | 104 | (0, 1) | (0, 1) |
| rtof | Hadamard, Phase, Linear | on | `sim:large` | 78 | (4, 3) | (4, 3) |
| ccix | Hadamard, Phase, Linear | on | `sim:large` | 92 | (4, 4) | (4, 4) |
| peres | Hadamard, Phase, Linear | on | `sim:large` | 124 | (7, 5) | (7, 5) |
| toffoli | Hadamard, Phase, Linear | on | `sim:large` | 158 | (7, 6) | (7, 6) |
| fredkin | Hadamard, Phase, Linear, Compose | on | `sim:large` | 654 | (7, 8), not solved | (7, 7) |
| fredkin | Compose, Hadamard, Phase, Linear, Simplify | on | `sim:large` | 1 566 | (7, 7) | (7, 7) |
| ch | Search, Meet | on | `sim:large` | 3 124 | (2, 1) | (2, 1) |

`each_route_reaches_its_cost_on_plain_and_dressed_targets` in `designer.rs` pins further
routes at 4096 thinking tokens on `sim:large`, each on the plain target and five
disguises: CZ by Hadamard and Linear (0, 1); CZ by Phase alone (0, 2), exact but not
solved; QFT2 by Hadamard, Phase and Linear (3, 3); CH by Meet alone (2, 1). It also pins
misses: Fredkin without Compose, Toffoli without Hadamard, SWAP by Phase, CZ by Hadamard
alone, CH by Search alone, and CH by Hadamard, Phase, Linear, Compose, Previous and
Simplify.

**Phrase steps.** No phrase names two of Hadamard, Phase and Linear. Of the 704 children
that one phrase step (SectionReplace, SectionAdd or StageAdd) makes of the seeds, 40
solve one target beyond swap and cs (20 cz, 20 ccz) and none solves two
(`one_phrase_step_from_a_seed_gains_at_most_one_target`). Two steps can teach all three:
the 60 one-step children whose last Worker knows two of them have 384 children that know
all three, and each of these solves cz, qft2, toffoli, peres, rtof and ccix
(`two_phrase_steps_from_a_seed_reach_the_six_basis_targets`). Both tests count each
target plain and in three disguises.

## 6. The arms and the procedure

**Arm A** (Darwin Gödel selection), for each seed S from 1 to 10:

```sh
brut-breed --out runs/A-S --domain circuits --search 33 --select 22 --children 8 \
  --generations 200 --max-calls 2600000 --seed S
```

**Arm R** (uniform parent choice): the same command with `--out runs/R-S` and
`--selection uniform`.

Every other setting stays at its default, as `config.json` records it: crossover rate
0.5, 3 peer opponents with 2 tasks per pair, judge weight 0.2, token cost 0.01 per 1000
tokens, DGM lambda 10 and alpha_0 0.5, no LLM rounds (`llm_every` 0), 3 finalists, at
most 4 Stages of at most 3 Workers, the models `sim:small` (0.0005 per call) and
`sim:large` (0.002 per call), and the circuit settings: every verified target, dressed,
at most 64 gates per answer. The analysis refuses a Run whose settings differ from these
in anything but the seed, the generations and the call cap, and it requires all Runs to
share their generations and their call cap.

**Length and cap.** The pilot sets `--generations 200` and `--max-calls 2600000`
(section 10). By our count from the call bounds in `src/cascade/arena.rs`, generation 0
needs at most 156 calls (4 seeds on 33 tasks plus 24 peer calls), each bred generation
at most 12 720 (8 children of at most 48 calls per task on 33 tasks plus 48 peer calls),
and the select phase at most 3 256 ((4 + 3 · 48) · 22). A Run of 200 generations thus
needs at most 2 547 412 calls, so the cap cannot end a counted Run early. The
200-generation pilot Runs used at most 214 049 calls and took at most 25.9 s
(`pilot.json`).

**Procedure.**

1. Build the program at the frozen commit with the frozen toolchain. Run
   `cargo test --lib` and `cargo test --test circuits` there, which hold every test named
   in sections 3 and 5; all must pass.
2. Run the 20 counted Runs, one at a time, each into a fresh directory.
3. Thread check: run seed 1 of each arm twice more, once with `RAYON_NUM_THREADS=1` and
   once with `RAYON_NUM_THREADS=4`, into fresh directories. In each arm, `config.json`,
   `events.jsonl`, `snapshot.json`, `report.json` and `rediscovery.json` must be
   byte-equal across the three Runs of seed 1. `timings.jsonl` holds wall time and is not
   compared.
4. Run `brut-breed --out <dir> --verify` on all 24 directories. Each must end with
   "report recomputed and equal; rediscovery recomputed and equal".
5. Run the frozen analysis on the 20 counted Runs:

```sh
cargo run --example circuit_positive_control -- \
  --a runs/A-1 runs/A-2 runs/A-3 runs/A-4 runs/A-5 runs/A-6 runs/A-7 runs/A-8 runs/A-9 runs/A-10 \
  --r runs/R-1 runs/R-2 runs/R-3 runs/R-4 runs/R-5 runs/R-6 runs/R-7 runs/R-8 runs/R-9 runs/R-10
```

## 7. Definitions

For the Run of one arm on seed s, a member is a Cascade of the archive. The four
hand-written seed Cascades of generation 0 are the seed members; every other member is
bred. The archive keeps every member, so nothing below gets smaller with more
generations.

- **Exact.** An answer is exact when its unitary equals the task's dressed target up to
  one global phase, as the integer checker of `src/cascade/qubit.rs` decides.
- **solved(t).** A target t in L is solved by a set of members when one of them gave an
  exact answer with T <= T_ref(t) and CNOT <= CNOT_ref(t) on at least one of the three
  dressed search tasks of t.
- **R_seed(s)** is the set of targets the seed members solve.
- **R_bred(s)** is the set of targets the bred members solve.
- **D(s)** = |R_bred(s) \ R_seed(s)| / |L \ R_seed(s)|: the share of the targets the seed
  Cascades miss that a bred Cascade solves.
- **The gate.** Seed s is calibrated when 3 · |R_seed(s)| <= |L|.
- **D_A(s) and D_R(s)** are D of the arm A and the arm R Run of seed s. R_seed(s) is the
  same in both arms, since generation 0 does not depend on the parent choice; the
  analysis refuses a pair where it differs.

`rediscovery()` in `src/cascade/circuit.rs` computes these sets and writes them to each
Run's `rediscovery.json`; `--verify` checks the checkpoint's archive against the logged
`born` events and recomputes the file from that archive, and the analysis recounts it
from its own target list. Every decision is made in integers: D > 1/2
means 2 · |R_bred \ R_seed| > |L \ R_seed|, and D_A > D_R compares exact fractions.

## 8. Results

**Primary result (ticket 03).** It passes when D_A(s) > 1/2 for at least 9 of the 10
seeds. We report the median of the 10 values D_A(s), as the analysis computes it: the
mean of the fifth and sixth smallest value.

**Secondary result: selection matters.** It passes when D_A(s) > D_R(s) in at least 9 of
the 10 pairs; a tie counts against arm A. Under the null hypothesis that each pair goes
either way with probability 1/2, P(at least 9 of 10) = (10 + 1) / 2^10 = 11/1024, about
0.0107, so this is an exact one-sided sign test of size 11/1024. The analysis also prints
the exact p for the observed count k, P(at least k of 10).

**T-only variant (secondary).** Both rules once more, where a target counts as solved
when an exact answer has T <= T_ref, and also CNOT <= CNOT_ref where T_ref is 0 (cz and
swap). R_seed, D and both rules are recomputed under this rule. In the pilot the seed
Cascades solve ccz under it, with 7 T and 10 CNOT. The variant neither voids nor rescues
the primary result.

**Descriptive results** (no decision attached). The analysis prints, per target and
seed, the first generation in which a bred member of each arm solved the target; the
Runs that ended at the call cap; and every invention candidate. From each Run's files we
also report: the `state` lines of every tenth generation (as in `pilot.json`), the calls
used and the wall time, the best member's solved tasks on the search panel (`report.json`,
`top`), and how many of the 22 select tasks the finalists and the seed Cascades solve
(`report.json`, `finalists`). The select panel is descriptive, as for grids.

## 9. Void conditions

The experiment is void, and we make no claim, when any of these holds:

1. A counted Run raises an integrity alarm: an exact answer below a Proven or Derived
   reference. Such a Run exits with code 1 and has no `rediscovery.json`.
2. The thread check of section 6 finds a byte that differs.
3. A calibration test of section 5 fails at the frozen commit.
4. A seed fails the gate 3 · |R_seed(s)| <= |L|. Then the positive control is too easy
   and must be recalibrated before a new pre-registration.
5. A counted Run fails `brut-breed --verify`.

## 10. The pilot seeds

Seeds 101 and 102 are pilots. We do not count them, and they never enter the analysis of
the counted seeds. They may set only `--generations` and `--max-calls`. The calibration
test reads the search panel of seed 101 without a Run. `docs/pilot-step1/pilot.json`
holds every number below; all its Runs pass `--verify`.

- **60 generations, `--max-calls 200000`.** R_seed = {swap, cs} in all four Runs. Arm A:
  D = 7/9 on seed 101 and 2/9 on seed 102. Arm R: 1/9 and 2/9. No integrity alarm, no
  invention candidate, no Run at the cap.
- **The default cap.** 20 000 calls stopped arm A on seed 101 after generation 20.
- **200 generations, `--max-calls 1000000`.** Each Run repeats its 60-generation Run
  exactly through generation 60, since only the stop check reads `--generations`. D
  stays 7/9 and 2/9 in arm A; in arm R it is 2/9 on both seeds (ccz at generation 90 on
  seed 101). Every first hit came by generation 90, none in generations 91 to 200. At 60
  generations the last first hit of arm A on seed 101 came at generation 55.
- **Cost.** A 60-generation Run took 2.3 to 5.8 s and a 200-generation Run 12.6 to
  25.9 s, wall time at 2 threads.

So the counted Runs use 200 generations, which lie past every first hit of the pilot at
moderate cost, and a cap that cannot bind. Two pilot seeds cannot estimate the pass
rate: by our calculation, exactly one pass in two seeds has probability 0.5 at a pass
rate of 0.5 per seed and 0.18 at 0.9, while the primary rule then holds with probability
0.0107 and 0.736.

## 11. The claim

If the primary rule passes, we claim: breeding assembles prompts and Traits that unlock
exact, reference-cost syntheses for most library targets the seed Cascades miss. If the
secondary rule passes too: Darwin Gödel selection beats a uniform choice of parents with
the same number of children. Both are claims about the breeder as a positive control,
not about real models, and not about inventions: an answer below a BestKnown count is
only an invention candidate until a human has checked the literature (Q38).

If a rule fails, we report that it fails. A later change of the genes, the designer, the
settings or the analysis needs a new pre-registration and seeds not used before.
