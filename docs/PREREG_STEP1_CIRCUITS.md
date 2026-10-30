# Pre-registration of step 1: does breeding rediscover circuit references?

Status: signed off by the owner on 2026-10-30; frozen at the commit that adds this
version to the main branch.

This document fixes the step 1 experiment of plan (g) and
[ADR 0006](adr/0006-cascades-breed-on-domains-and-circuits-are-checked-exactly.md) before
any counted Run. Of this experiment, only pilot seeds have run
([docs/pilot-step1/](pilot-step1/README.md)); no counted seed has run in either arm. The
owner signed this document off on 2026-10-30, together with the six decisions of ADR
0006. Decision 5 is the sequential rule of section 8, which replaced the fixed rule
"9 of 10" on the same day. The frozen commit is the commit that adds this version of the
file to the main branch. A commit cannot hold its own hash, so we record the hash with
the results and in the pull request that adds this version. From that commit on nothing
below changes; every deviation is reported with the results, with its reason. A change
after seed 1 has started makes a new experiment on seeds not used before. The report on
step 1 is [SCHRITT1_SCHALTUNGEN.md](SCHRITT1_SCHALTUNGEN.md) (English:
[STEP1_CIRCUITS.md](STEP1_CIRCUITS.md)).

## 1. Question

Does the Stage 2c breeder, working on quantum circuits with the simulated designer as its
model, solve most of the library targets that its seed Cascades miss, exactly and at the
reference T and CNOT counts, within 1000 generations? And does Darwin Gödel selection
(arm A) get there sooner than a uniform choice of parents (arm R) with the same number of
children? This is a positive control of the breeder. It says nothing about real language
models.

## 2. What is frozen

| Item | Frozen as |
|---|---|
| Code | The whole tree of the frozen commit: the commit that adds this version of the file to the main branch. Its hash is recorded with the results and in the pull request that adds this version. A later edit of this file or of the code is a deviation. |
| Toolchain | rustc 1.97.0 on x86_64 Linux, as in the pilot (`docs/pilot-step1/pilot.json`). |
| Program | `cargo build --release --bin brut-breed` at that commit. |
| Analysis | `examples/circuit_positive_control.rs` at that commit. |
| Recount | `.scratch/universum-erfinden/sequential-rule/reference_rule.py` at that commit, an independent implementation of the rule of section 8 (section 6, step 7). |
| Target list L | The 11 verified library targets (section 3). |
| Designer | The constants of `src/cascade/designer.rs` (section 4). |
| Genes | `src/cascade/circuit_genes.rs`: the 4 seed Cascades, the 50 phrases, the Mutator's brief and the admission rule `no_gate_lists`. |
| Calibration | The results of section 5, pinned by tests. |
| Arms | Section 6. |
| Seeds | 1, 2, 3, ... in this order, at most 60, paired between the arms. The pilot seeds 101 to 110, 201 to 210, 301 to 310, 401 and 402 are not counted (section 10). |
| Definitions and rules | Sections 7 to 9. |

The Runs of the first pilot (seeds 101 and 102, section 10) were first made with an
earlier state of this work, before the review fixes. Those fixes changed the code only
where no Run's files depend on it: `--verify` also checks a finished circuit Run's archive
against the logged `born` events and says why it skips `rediscovery.json`; one printed line
about D follows D alone; the analysis makes a claim only for the generations and the cap
of section 6; and there are new tests and comments. When the first version of this file
was proposed, we built the program from the same tree and ran all nine Runs of the first
pilot again: their `config.json`, `events.jsonl`, `snapshot.json`, `report.json` and
`rediscovery.json` are byte-equal to the pilot's, and all nine pass its `--verify`. So the
first pilot's numbers are numbers of that program, and the frozen commit can be checked on
its own: at that commit, the commands of `docs/pilot-step1/README.md` must give event logs
that end in the head hashes that `pilot.json` records. On another toolchain or platform a
floating-point function may round its last bit differently, and then the Runs may differ
from Runs on the frozen toolchain; such Runs are a deviation.

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
the genome's rules and the admission rule. It is one hand-written text, which a Run
without LLM rounds cannot write; Cascades of pool phrases reach every target at its
reference, plain and in one disguise
(`pool_phrases_reach_every_target_at_its_best_known_cost` in `circuit_genes.rs`).

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
Simplify. `fewer_families_reach_qft2_and_ccix_at_the_ceiling_of_the_traits` in
`tests/circuits.rs` pins two routes of pool phrases with fewer families, at 4096 thinking
tokens on `sim:large` with self-check: Meet alone solves QFT2 in 4 of its 32 disguises,
and Hadamard, Phase and Simplify solve CCiX in 128 of its 384.

**Phrase steps.** No phrase names two of Hadamard, Phase and Linear. Of the 704 children
that one phrase step (SectionReplace, SectionAdd or StageAdd) makes of the seeds, 40
solve one target beyond swap and cs (20 cz, 20 ccz) and none solves two
(`one_phrase_step_from_a_seed_gains_at_most_one_target`). Two steps can teach all three:
the 60 one-step children whose last Worker knows two of them have 384 children that know
all three, and each of these solves cz, qft2, toffoli, peres, rtof and ccix
(`two_phrase_steps_from_a_seed_reach_the_six_basis_targets`). Both tests count each
target plain and in three disguises.

## 6. The arms and the procedure

**Arm A** (Darwin Gödel selection), for each counted seed S = 1, 2, 3, ... in this order,
at most 60:

```sh
brut-breed --out runs/A-S --domain circuits --search 33 --select 22 --children 8 \
  --generations 1000 --max-calls 13000000 --seed S
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
share their generations and their call cap. It also refuses a Run that bred more
generations than its config plans (a resume may lower `--generations` below what a Run
had bred) and a Run that stopped short of them for any reason but the call budget. It
makes a claim only when these are the 1000 generations and the cap of 13 000 000 calls
below, no Run ended at the cap, and the seeds are exactly 1 to n with n <= 60; otherwise
it prints the numbers and says that it makes no claim.

**Length and cap.** Every Run completes all 1000 generations; there is no early stop. The
pilot batches set `--generations 1000` and `--max-calls 13000000` (section 10). By our
count from the call bounds in `src/cascade/arena.rs`, generation 0 needs at most 156
calls (4 seeds on 33 tasks plus 24 peer calls), each bred generation at most 12 720 (8
children of at most 48 calls per task on 33 tasks plus 48 peer calls), and the select
phase at most 3 256 ((4 + 3 · 48) · 22). A Run of 1000 generations thus needs at most
156 + 1000 · 12 720 + 3 256 = 12 723 412 calls, so the cap cannot end a counted Run
early. The pilot Runs used at most 1 161 738 calls by generation 1000 (A-401), and a
pilot pair of 1000 generations took 10.57 to 14.31 min, wall time at 2 threads.

**Procedure.**

1. Build the program at the frozen commit with the frozen toolchain. Run
   `cargo test --lib` and `cargo test --test circuits` there, which hold every test named
   in sections 3 and 5; all must pass.
2. Acceptance check, before seed 1 and not counted: run the frozen analysis on the pilot
   pairs 301 to 309 of section 10. It must withhold the claim, give the pass times
   T_A = never, 7, 75, 97, 349, 143, 33, 180, 810 and T_R = never, 380, then never for
   the rest, and end open with E1 = 6561/512 and E2 = 6561/256, which crossed 20 at the
   ninth pair. `reference_rule.py --pilots` gives the same values.
3. Run the counted pairs in seed order. For pair S, run arm A and arm R of seed S, one
   process at a time, each into a fresh directory. A Run that is stopped, killed or
   otherwise interrupted before it finishes is not resumed: we delete its directory and
   run it again from scratch, on the same seed, into a fresh one. A resumed Run would
   reach the same Cascades, judgments, `state` lines and `rediscovery.json`, but it counts
   the reserved calls of the interrupted generation as spent, so `calls_used` in its
   `events.jsonl`, `snapshot.json` and `report.json` would differ from a straight Run's.
   Only uninterrupted Runs are counted, and the same holds for the Runs of step 5.
   Finished pairs are kept.
4. After each pair, run `brut-breed --out <dir> --verify` on both of its Runs. Each must
   end with "report recomputed and equal; rediscovery recomputed and equal". Then run the
   frozen analysis on all finished pairs, 1 to n:

   ```sh
   cargo run --example circuit_positive_control -- \
     --a runs/A-1 runs/A-2 ... runs/A-n --r runs/R-1 runs/R-2 ... runs/R-n
   ```

   It updates the evidence values of section 8 in seed order and then either names the
   next seed to run or gives the verdicts. As long as it names a seed, go on with step 3
   for that seed.
5. Thread check, once, right after pair 1 and before pair 2, so that a void shows early:
   run seed 1 of each arm twice more, once with `RAYON_NUM_THREADS=1` and once with
   `RAYON_NUM_THREADS=4`, into fresh directories, one process at a time. In each arm,
   `config.json`, `events.jsonl`, `snapshot.json`, `report.json` and `rediscovery.json`
   must be byte-equal across the three Runs of seed 1. `timings.jsonl` holds wall time and
   is not compared. `--verify` must pass on these four Runs too.
6. The experiment stops at the stop pair of section 8. Pairs run after it are listed but
   not counted.
7. Before the result is reported, recount it independently:

   ```sh
   python3 -I .scratch/universum-erfinden/sequential-rule/reference_rule.py --runs runs
   ```

   This reads A-1, R-1, A-2, R-2, ... in seed order and counts a pair once both of its
   Runs are finished. For every counted Run it checks the seed, the arm, the settings of
   this section, the stop reason and the calibration gate. It refuses gaps in the seeds
   and more than 60 pairs, and it reports the experiment void when a pair up to the stop
   fails the gate. Its pass times, evidence values, stop pair and verdicts must equal
   those of the analysis; we report the recount with the results, and any difference.

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
- **Pass time T(s).** The first generation g with
  2 · |R_bred(s, <= g) \ R_seed(s)| > |L \ R_seed(s)|, where R_bred(s, <= g) is the set of
  targets that the bred members of generations 1 to g solve. T(s) = never when no
  generation of the Run qualifies. T(s) is a generation, not a T count. The Run passes
  when T(s) <= 1000. Since the archive keeps every member and D never falls, this equals
  D(s) > 1/2 at generation 1000.
- **Reading T from `rediscovery.json`.** Take the targets with `reached_by_bred` = true
  and `reached_by_seed` = false. T is the k-th smallest `first_generation` among them,
  where k = floor(|L \ R_seed| / 2) + 1, so k = 5 when R_seed = {swap, cs}. T = never
  when fewer than k such targets exist.
- **T_A(s) and T_R(s)** are T of the arm A and the arm R Run of seed s.
- **X(s)** = 1 when the arm A Run of seed s passes, else 0.
- **Pair outcome Y(s).** "A first" when T_A(s) < T_R(s); "R first" when
  T_R(s) < T_A(s); "tie" when T_A(s) = T_R(s), including the case where neither Run
  passes. Never counts as later than every generation, so "A first" includes a pass of arm
  A when arm R never passes.

`rediscovery()` in `src/cascade/circuit.rs` computes these sets and writes them to each
Run's `rediscovery.json`; `--verify` checks the checkpoint's archive against the logged
`born` events and recomputes the file from that archive, and the analysis recounts it from
its own target list. Every decision is made in integers: D > 1/2 means
2 · |R_bred \ R_seed| > |L \ R_seed|, pass times are compared as generations, and the
evidence values of section 8 are compared as 3^k against 20 · 2^n.

## 8. Results

The experiment decides by itself how many seeds it needs: a sequential rule across seeds.
It replaces the fixed rule "9 of 10" of the first version of this document (ADR 0006,
decision 5). Its design, its sources and its operating characteristics are in
`.scratch/universum-erfinden/issues/09-sequential-rule-for-step-1.md`;
`own_check.py` in `.scratch/universum-erfinden/sequential-rule/` recomputes them from the
rule text alone.

**Evidence values.** Two evidence values, E1 and E2, start at 1. They are updated strictly
in seed order: pair s enters only after pairs 1 to s - 1, never in the order in which Runs
finish.

- E1 is multiplied by 3/2 when X(s) = 1 and by 1/2 when X(s) = 0. After n pairs with k
  passes, E1 = 3^k / 2^n.
- E2 is multiplied by 3/2 when Y(s) is "A first" and by 1/2 when it is "R first"; a tie
  leaves it unchanged. After w pairs "A first" and l pairs "R first", E2 = 3^w / 2^(w + l).
- The factors 3/2 and 1/2 are the likelihood ratio of a pass rate of 3/4 against a pass
  rate of 1/2.

**Primary result (ticket 03).** The primary claim is proven at the first pair with
E1 >= 20, that is 3^k >= 20 · 2^n. It fails at the first pair with E1 <= 1/20, that is
20 · 3^k <= 2^n. In counts, it is proven at 8, 10, 11, 13, 15, 16, 18 or 20 passes with 0
to 7 failures, and it fails at 5, 5, 6, 7, 7, 8, 8 or 9 failures with 0 to 7 passes. So
it takes at least 8 pairs to prove and at least 5 to fail. Proven means that more than
half of all seeds pass within 1000 generations: it rejects the null hypothesis "pass rate
<= 1/2". Failed means that the data speak against a pass rate of 3/4 or more: this verdict
is wrong with probability at most 1/20 when the rate is 3/4 or more (0.0347 at exactly
3/4). It does not prove a rate of 1/2 or less.

**Secondary result: selection matters.** The same two thresholds apply to E2. Proven means
that among untied pairs, arm A passes first more often than arm R: it rejects the null
hypothesis "P(A first | not tied) <= 1/2". Pairs in which neither Run passes say nothing
about selection. The two claims form a fixed sequence: the secondary is claimed only if
the primary is proven, whichever of the two crosses a threshold first. Every verdict is
final at its first crossing.

**Stop.** The ceiling is 1000 generations per Run and 60 pairs per experiment. After each
pair, the experiment stops at the first of:

1. The primary has failed. The secondary is then "not tested", and E2 is reported.
2. The primary is proven and E2 has crossed either threshold.
3. 60 pairs are done. A claim still open is "undecided, no claim" and is reported with its
   E value. Running more pairs later is a deviation and is reported as one.

**Error bounds.** However the experiment stops, each claim is made falsely with
probability at most 1/20, and some claim is made falsely with probability at most 1/20.
This follows from Ville's inequality for test supermartingales (Shafer, Shen, Vereshchagin
and Vovk 2011; Waudby-Smith and Ramdas 2024) and from the fixed sequence of the two
claims. The exact worst case for the primary claim, at a pass rate of 1/2 and the cap of
60 pairs, is a false proof with probability 5555272097573807 / 2^57 = 0.0385; the largest
probability of any false claim over the joint null scenarios of issue 09 is also 0.0385.
The bar is weaker than that of the fixed rule "9 of 10", whose size was 11/1024 = 0.0107;
it is the 5 % bar of the project's other pass tests. A fixed test cannot simply be checked
again after every pair instead: an exact one-sided 5 % binomial test run after every pair
succeeds falsely with probability 0.0987 by 20 pairs.

**Conduct.** The error bounds require that the seed order is fixed, that no seed is skipped
or replaced, that the updates go by seed and never by completion order, and that the
settings, the ceiling, the definitions of T, X and Y and the thresholds do not change after
seed 1. A breach voids the bounds. The bounds also rest on one assumption that cannot be
checked: under the null hypothesis, each new seed passes with probability at most 1/2,
given the earlier seeds. We assume this because the random streams of a Run are derived
from a hash of its seed. In this container we run one process at a time; elsewhere Runs
may run in parallel as long as the updates stay in seed order.

**Reported per pair.** R_seed, D_A and D_R at generation 1000, T_A, T_R, X, Y, and E1 and E2
after the pair. Then the stop pair and both verdicts, with E1 and E2 at the stop. We never
report the largest E along the way, which overstates the evidence.

**T-only variant (descriptive).** A target counts as solved when an exact answer has
T <= T_ref, and also CNOT <= CNOT_ref where T_ref is 0 (cz and swap). R_seed and D are
recomputed under this rule at generation 1000. We report D_A and D_R under it and the
number of seeds with D_A > 1/2 under it, with no verdict: `rediscovery.json` holds no first
generations under this rule, so it gives no pass time. In the pilot the seed Cascades solve
ccz under it, with 7 T and 10 CNOT. The variant neither voids nor rescues a result.

**Descriptive results** (no decision attached). We report the median of D_A over the
counted pairs, as the analysis computes it: the middle value, or the mean of the two middle
values. The analysis prints, per target and seed, the first generation in which a bred
member of each arm solved the target; the Runs that ended at the call cap; and the cheapest
invention candidate per target and Run (any others are in the answers of the `born`
events). From each Run's files we also report the other descriptive results of plan (g),
defined from the logged data:

- **Calls to the first hit** per target: `calls_used` of the `generation` event of the
  target's first generation, that is the calls through that generation
  (`events.jsonl`).
- **Best single-Cascade coverage:** the most search tasks that one member of the archive
  solves, the largest `solved` of the `born` events, next to the solved tasks of the
  best member by fitness (`report.json`, `top`), which need not be the same Cascade.
- **Select-panel hits:** how many of the 22 select tasks the finalists and the seed
  Cascades solve (`report.json`, `finalists`). The select panel is descriptive, as for
  grids.
- **Observable trajectories:** the `state` lines of every tenth generation (as in
  `pilot.json`).

We also report the calls used and the wall time of every Run.

## 9. Void conditions

The experiment is void, and we make no claim, when any of these holds:

1. A counted Run raises an integrity alarm: an exact answer below a Proven or Derived
   reference. Such a Run exits with code 1 and has no `rediscovery.json`.
2. The thread check of section 6 finds a byte that differs.
3. A calibration test of section 5 fails at the frozen commit.
4. A counted pair, up to the stop, fails the gate 3 · |R_seed(s)| <= |L|. Then the
   positive control is too easy and must be recalibrated before a new pre-registration.
5. A counted Run fails `brut-breed --verify`.
6. A counted Run has other generations or another call cap than section 6 (1000
   generations, 13 000 000 calls), or it ended at the call cap. The analysis then makes no
   claim.

## 10. The pilot seeds

Pilot seeds never count: 101 to 110, 201 to 210, 301 to 310, 401 and 402, which include
the first pilot seeds 101 and 102. They never enter the analysis of the counted seeds,
because they chose the design: the later batches set the ceiling of 1000 generations, the
call cap and the planning figures below. The calibration test reads the search panel of
seed 101 without a Run. The pass time of every pilot Run is in
[docs/pilot-step1/](pilot-step1/README.md).

**First pilot, seeds 101 and 102.** `docs/pilot-step1/pilot.json` holds every number of
this list; all its Runs pass `--verify`.

- **60 generations, `--max-calls 200000`.** R_seed = {swap, cs} in all four Runs. Arm A:
  D = 7/9 on seed 101 and 2/9 on seed 102. Arm R: 1/9 and 2/9. No integrity alarm, no
  invention candidate, no Run at the cap.
- **The default cap.** 20 000 calls stopped arm A on seed 101 after generation 20.
- **200 generations, `--max-calls 1000000`.** Each Run repeats its 60-generation Run
  exactly through generation 60, since only the stop check reads `--generations`. D
  stays 7/9 and 2/9 in arm A; in arm R it is 2/9 on both seeds (ccz at generation 90 on
  seed 101). In these four Runs every first hit came by generation 90, none in
  generations 91 to 200. At 60 generations the last first hit of arm A on seed 101 came
  at generation 55.
- **Cost.** A 60-generation Run took 2.3 to 5.8 s and a 200-generation Run 12.6 to
  25.9 s, wall time at 2 threads.

**Later batches.** Both arms ran with the settings of section 6 except the seed, the
generations and the call cap. The Runs are not in the repository;
`.scratch/universum-erfinden/sequential-rule/pilot.json` holds the values read from them,
and `pilot.txt` beside it summarises them.

| Seeds | Generations | `--max-calls` | Arm A passes | Arm R passes |
|---|---|---|---|---|
| 101 to 110 | 200 | 2 600 000 | 1 of 10 | 1 of 10 |
| 201 to 210 | 200 | 2 600 000 | 5 of 10 | 2 of 10 |
| 301 to 309 | 1000 | 13 000 000 | 8 of 9 | 1 of 9 |
| 401 and 402 | 2000 | 26 000 000 | 1 of 2 | 1 of 1 |

The Run A-310 was interrupted after generation 922; it is incomplete and left out of every
figure. R-310 and R-402 were not run. In all 61 complete Runs, R_seed = {swap, cs}, the gate
holds and the Run ended with all generations done. None raised an integrity alarm, found an
invention candidate or solved ch or fredkin.

- **The ceiling.** By generation 200, arm A passed on 6 of the 20 seeds 101 to 110 and 201
  to 210: a rate of 0.30, with a 95 % confidence interval of 0.119 to 0.543. At that rate
  the rule of section 8 fails with probability 1.000, after 9.1 pairs on average, so the
  ceiling cannot stay at 200. Passes came as late as generation 810 in arm A (seed 309)
  and 720 in arm R (seed 401). 1000 is the longest horizon with pilot data in both arms:
  11 Runs of arm A and 10 of arm R reached it.
- **The call cap.** 13 000 000 lies above the bound of 12 723 412 calls of section 6. The
  pilot Runs used at most 1 161 738 calls by generation 1000.
- **Pass rates at 1000 generations.** Arm A passed in 9 of the 11 Runs that reached
  generation 1000 (0.818; 95 % confidence interval 0.482 to 0.977). A Kaplan-Meier
  estimate over all 31 complete Runs of arm A gives 22/31 = 0.710. Arm R passed in 2 of
  10 (Kaplan-Meier: 0.280). In the 10 pairs with both arms run to generation 1000, arm A
  passed first 8 times, arm R once (seed 401), and once neither passed (seed 301).
- **Wall time at 2 threads.** A Run to generation 1000 took 6.01 to 8.57 min in arm A
  (mean 7.38 min, 11 Runs) and 4.54 to 6.12 min in arm R (mean 5.38 min, 10 Runs). A pair
  took 10.57 to 14.31 min (mean 12.68 min, 10 pairs).

**Planning figures** (not part of the rule). At the pilot rates the primary claim is
proven with probability 0.82 to 0.99, and both claims with 0.815 to 0.990: the low end
draws pairs from the Kaplan-Meier curves of both arms, the high end from the 10 long pilot
pairs as observed. The experiment then needs 29.1 to 17.0 pairs on average, that is 6.1
to 3.6 hours at 12.68 min per pair; 90 % of such experiments need at most 57 to 30 pairs.
It needs at most 60 pairs: 14.3 hours at the slowest pilot pair, or 14.7 hours if every
pair had the slowest Run of each arm. The thread check comes on top. These figures assume
that the arms of a pair are independent, that the Kaplan-Meier curves stay flat after
their last pilot pass, and that this container's timings carry over. The fixed rule
"9 of 10" would have passed with probability 0.165 at the Kaplan-Meier rate and 0.433 at
9/11.

## 11. The claim

If the primary claim is proven, we claim: breeding assembles prompts and Traits that
unlock exact, reference-cost syntheses for most library targets the seed Cascades miss, in
more than half of all seeds within 1000 generations. If the secondary claim is proven
too: Darwin Gödel selection gets there sooner than a uniform choice of parents with the
same number of children; among untied pairs, arm A passes first more often than arm R.
Both are claims about the breeder as a positive control, not about real models,
and not about inventions: an answer below a BestKnown count is only an invention
candidate until a human has checked the literature (Q38).

If the primary claim fails, we report that it fails: the data then speak against a pass
rate of 3/4 or more, which is no proof of a rate of 1/2 or less. A claim that is not
decided by pair 60 is reported as undecided, with its E value. A later change of the
genes, the designer, the settings or the analysis needs a new pre-registration and seeds
not used before.
