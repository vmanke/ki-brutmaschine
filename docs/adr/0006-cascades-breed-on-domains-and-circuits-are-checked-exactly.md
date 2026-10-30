# Cascades breed on domains, and circuits are checked exactly

Step 1 of the second build order (Q47, tickets 01 to 03 in `.scratch/universum-erfinden/`)
lets the Stage 2c breeder work on more than grid puzzles, and makes its first new field
quantum circuits whose correctness an exact checker decides. It also logs, per generation,
the observables that let a Run be studied as a universe (Q32, Q33). This record decides
the seam, the checker, the targets, the scoring, the positive control and the experiment.
The report is `docs/SCHRITT1_SCHALTUNGEN.md` (English: `docs/STEP1_CIRCUITS.md`), and the
experiment is pre-registered in `docs/PREREG_STEP1_CIRCUITS.md`.

**One loop, many Domains.** The breeder talks to a kind of task only through two pieces.
A **Domain** is the evaluator side: it makes the search and select panels, renders the
solve and judge prompts, reads answers, grades them and says which of two answers is
better. It never sees a Cascade, so no Cascade can change how it is judged (ADR 0005's
layering). A **Gene pool** is the breeder side: the seed Cascades, the phrase pool of the
section operators, the brief that tells the Mutator what it breeds, and an admission rule
every new Cascade must pass. Grid puzzles are one Domain; circuits are the second. A Run
names its Domain with `--domain grid|circuits` and its parent choice with
`--selection dgm|uniform`; both define the experiment, so a resume refuses them. A judge
is scored on a pair exactly when the Domain strictly prefers one answer; for grids this is
the old rule (exactly one answer right). Fitness uses the graded value of a Domain that
grades (circuits) and the solved count otherwise (grids), so grid fitness is computed bit
for bit as before. A golden test, recorded on the unchanged grid code before the seam was
built, pins a grid run, a run with LLM rounds, the Mutator's request keys and the settings
hashes; it still holds. The only new line in a grid log is the observables line.

**Observables are a pure function of committed state.** After every committed generation
the log gets one `state` event with six groups: performance, diversity (including the
Shannon entropy of the building blocks), lineage, honesty, judging and cost. It is computed
from the archive, all comparisons, the next generation's selection weights and the
generation's own usage, never from cumulative call counts, which include crash
reservations. So a Run that is stopped or killed and resumed writes the same `state`
lines as a straight Run. Undefined values are absent rather than NaN. Honesty stays zero
until overseers exist (ticket 05). Wall time goes to `timings.jsonl`, outside the hash
chain, because the pure core has no clock.

**Correct means exactly equal.** A circuit over H, S, S†, T, T†, X, Z and CNOT is correct
when its unitary equals the target exactly, up to one global phase. The checker
(`src/cascade/qubit.rs`) computes with amplitudes x / sqrt(2)^k, where x lies in Z[omega]
(omega = e^(i pi / 4)) and is held as four 64-bit integers. For a unit vector every
coefficient satisfies |c| <= 2^(k/2), and only H raises k; with at most 120 H gates,
|c| <= 2^60, so 64-bit integers never overflow, and every operation is checked anyway. If
U equals lambda T exactly, lambda must be a power of omega, so the check has no tolerance.
The fidelity F = |tr(T^dagger U)|^2 / d^2 is computed in floating point and reported,
because Q38 named F >= 1 - 1e-9 as the criterion. Exact equality implies that bound and
is stricter: by our calculation, a two-qubit circuit whose only error is a relative phase
of 7e-5 rad on one basis state still reaches F >= 1 - 1e-9. A test-only dense simulator
agrees with the exact engine within 1e-12 on 500 random circuits of up to 5 qubits and 60
gates. Caps are code: 12 qubits for states, 6 for whole unitaries, 256 gates, 120 H
gates; a Run may lower the gate cap (64 by default), never raise it.

**A second simulator checks the first.** `src/quantum.rs` (PR #12) is a floating-point
state-vector simulator written separately; it shares no code with the exact checker and
decides equivalence by fidelity, at F >= 1 - 1e-9. Its qubit 0 is the least significant
bit, the checker's the most significant, so the tests map qubit q to n - 1 - q. In
`tests/circuits.rs` both decide alike on 900 pairs of random circuits on one to four
qubits: 300 pairs equal by construction (identity words, exact rewrites, swapped
commuting neighbours), 300 unequal by construction and 300 drawn independently. For
each of the 15 library witnesses both give the same matrix within 1e-12, and
`quantum.rs` confirms that the witness implements its target, which is built from its
definition, not from a circuit.

**Only sourced references.** The library holds the 11 targets whose T and CNOT counts the
research verified against the literature, each number with a status: Proven (a published
lower bound), Derived (our own argument from published results) or BestKnown (a published
construction without a lower bound). They are CZ, SWAP, CS, CH, QFT2, CCZ, Toffoli,
Peres, Fredkin, RTOF and CCiX; the sources are listed with each entry (among them Amy,
Maslov, Mosca and Roetteler 2013; Vatan and Williams 2004; Shende and Markov 2009;
Gosset et al. 2014; Gheorghiu, Mosca and Mukhopadhyay 2022; Maslov 2016; Selinger 2013).
Four further targets (W gate, sqrt(SWAP), quantum OR, 1-bit adder) have no verified
reference; they never enter a panel and serve only as checker test vectors. An exact answer
below a Proven or Derived number stops the Run with an integrity error, because then the
checker or the reference is wrong; nothing of that generation is committed. An exact
answer below a BestKnown number (Fredkin's 7 CNOTs, CCiX's 4 CNOTs) is an invention
candidate; `rediscovery.json` records the cheapest one per target, with
`literature_checked = false`. A candidate is called new only after a human has checked
the literature (Q38).

**Tasks hide what they are.** A task is a library target in a disguise drawn per task:
a relabelling of the qubits and X gates on inputs and outputs. The disguise keeps the
optimal T and CNOT counts in both directions, and it keeps memorised circuits from
fitting. Prompts show only the target's action on every basis state; they never carry its
name, id, sources or reference counts. The pre-registered panels have 33 search tasks
(three disguises per target) and 22 select tasks (two per target); as for grids, the
select panel is descriptive.

**Correctness first, then T, then CNOT.** A task's value is 0 for an inexact circuit and
0.5 + 0.5 / (1 + dT + dC / (1 + dC)) for an exact one, where dT and dC are the T and CNOT
counts above the reference (zero when at or below it). One T gate more always costs more
than any number of extra CNOT gates, and the value is 1 exactly when the circuit reaches
both reference counts; that is what "solved" means. The order is lexicographic because
T gates dominate the cost of fault-tolerant execution (Amy et al. 2013).

**A simulated designer is the positive control.** In this container no real model writes
circuits, so a simulated designer (`src/cascade/designer.rs`) answers circuit prompts
under the model tags `sim:small` and `sim:large`. It reads only the request: the system
prompt, where the names of known methods switch on eight synthesis families (linear
networks, phase polynomials, basis change, conjugation, word search, meet in the middle,
simplification, reuse of earlier answers), and the task prompt. Each family is a generic
algorithm (meet in the middle after Amy et al. 2013; phase polynomials after Amy and Mosca
2019); the designer holds no circuit, template or target list, and a source scan checks
that it names no library. All work costs steps, paid before it is done: 16 free steps
per call plus one step per 4 thinking tokens on `sim:small` and per 2 on `sim:large`. A
ceiling of 2^13 = 8 192 steps per call is code, not a gene; within the Traits' ceiling of
4096 thinking tokens a call has at most 2 064 steps, so the ceiling binds only on direct
requests. The designer verifies every candidate before it answers: on every basis state
when the Worker's self-check Trait is on, otherwise only on the first and the last, so a
cheap wrong circuit can fool it. Success therefore depends on the genes: which methods a
prompt names and in which order, the budget and model, self-check, stage structure and
temperature. The calibration is pinned by `tests/circuits.rs` on the 33 search tasks
that pilot seed 101 draws. The four seed Cascades, each one `sim:small` Worker with 256
thinking tokens whose strategy names one family, solve only SWAP (the linear seed) and CS
(the phase seed), 2 of 11 targets. A hand-written strong prompt, one `sim:large` Worker
with 4096 thinking tokens and self-check that names Hadamard, Phase, Linear, Compose,
Simplify, Search and Meet, solves all 33 tasks at the reference, so no target is
reach-limited. Its numbers say nothing about real models.

**The genes make the hard targets take steps.** The circuit gene pool
(`src/cascade/circuit_genes.rs`) holds the four seeds, 50 phrases for the section
operators and the Mutator's brief. No phrase names two of Hadamard, Phase and Linear. A
seed knows at most one of the three, so one random phrase step teaches it at most one
more: of the 704 one-step children of the seeds, 40 solve one target beyond SWAP and CS
(CZ or CCZ) and none solves two, on every target plain and in three disguises. So QFT2,
Toffoli, Peres, RTOF and CCiX take at least two steps: the tests pin 60 stepping stones
whose last Worker knows two of the three and their 384 children that know all three,
each of which solves CZ, QFT2, Toffoli, Peres, RTOF and CCiX at the seeds' Traits.
These counts come from the unit tests of `circuit_genes.rs`. The three alone miss
Fredkin, which takes conjugation besides (`designer.rs`). Routes with fewer families
lie further away: at the ceiling of the Traits (`sim:large`, 4096 thinking tokens,
self-check), Meet alone solves QFT2 in 4 of its 32 disguises, and Hadamard, Phase and
Simplify solve CCiX in 128 of its 384 (`tests/circuits.rs`); such a Worker differs from
every seed in model, thinking budget and self-check. A circuit Cascade is refused when
any one section holds three or more gate mentions such as "CNOT 0 1", so no section can
spell out a circuit. Spread over several sections, two gates in each, a circuit still
passes (`tests/circuits.rs`); against such a memorised circuit stands the disguise of
every task.

**A finished circuit Run is recounted.** Before its report, a finished circuit Run writes
`rediscovery.json`: per target, the cheapest exact answer of the seed Cascades and of the
bred ones, whether each reached the reference, the first generation of a bred hit and the
cheapest invention candidate; then |L|, |R_seed|, |R_bred \ R_seed|, D and the
calibration gate. For a finished Run with its `report.json`, `--verify` rebuilds the Run
from its checkpoint without a model call and requires the recomputed report to be equal.
No hash covers the checkpoint, so it also requires the checkpoint's archive to hold one
member per hash-chained `born` event, in their order, each with every field its event
records: id, Cascade, parents, generation, operator, solved count, graded value, usage and
answers. It requires `rediscovery.json` to match its recomputation from that archive byte
for byte. A member that differs from its `born` event, a member too many or too few, and a
missing or edited file are integrity errors. What a member gains in later generations
(children, judgments, fitness, peer rating) is not in its `born` event, and `--verify`
does not check it. It also derives a finished Run's stop reason from the checkpoint ("all
generations done" or "call budget reached") instead of trusting the report. Without
`report.json` it does not check `rediscovery.json` and says why: the Run is not finished,
or its report is missing. A resume removes both files until the select phase is complete
again. Since `--verify` refuses an edited file, `literature_checked` stays false in the
Run, and a human's literature check is recorded outside it.

**The experiment is fixed before it runs.** `docs/PREREG_STEP1_CIRCUITS.md` freezes the
commit, the target list, the designer's constants, the calibration, the configuration,
the seeds, the analysis code (`examples/circuit_positive_control.rs`) and the thresholds
before any counted Run. Arm A uses Darwin Gödel selection; arm R is identical but chooses
parents uniformly, which separates selection from the sheer number of children. The
counted seeds are the fixed list 1 to 100, then 111 to 130; they run in this order,
paired between the arms, for at most 120 pairs. The pilot seeds 101 to 110, 201 to 210,
301 to 310, 401 and 402 are not counted; the list skips 101 to 110, which lie in its
range. For a seed s, D(s) is the share of the targets the seed Cascades miss that some
bred Cascade solves. Every Run goes to generation 1000, and its pass time T is the first
generation with D > 1/2, or never. The experiment is void if the seed Cascades solve more
than a third of the targets in a counted seed, if an integrity error occurs, or if the
logs differ between one and four threads. It follows the sequential rule: after each
pair, in the order of the list, the evidence value E1 is multiplied by 3/2 when arm A
passes and by 1/2 when it does not, and E2 by 3/2 when arm A passes first, by 1/2 when
arm R passes first, and not at all on a tie. The primary claim is proven when E1 reaches
100 and fails when E1 falls to 1/20; the secondary uses the same thresholds on E2 and is
claimed only if the primary is proven. A proof takes at least 12 pairs. The experiment
stops when the primary fails, when the primary is proven and E2 has crossed a threshold,
or after 120 pairs. However it stops, each claim, and any claim at all, is made falsely
with probability at most 1/100 (exact worst case 0.0079); the fixed rule "9 of 10" had
11/1024 = 0.0107. The threshold 100 bounds false claims at a one-sided level of 1 %,
stricter by a factor of 2.5 than the one-sided 2.5 % that the guideline ICH E9 (1998,
section 5.5) prefers in regulatory settings. On Jeffreys's scale an evidence value above
100 is "decisive" (Kass and Raftery 1995; Vovk and Wang 2021). With at most 120 pairs,
the primary claim is proven with probability 0.86 at the most conservative pilot estimate
of the pass rate, the Kaplan-Meier rate 0.710; the conventional power is 0.80 (Cohen
1992). The former threshold 20 with at most 60 pairs proves the primary claim more often
below a pass rate of about 0.67. Averaged over the uncertainty of the pilot rate, both
designs give about the same chance of a true proof. The change buys a false success about
five times less likely (0.0079 instead of 0.0385), at more computing time. If the primary
claim is proven, the claim is only this: breeding assembles prompts and Traits that unlock
exact, reference-cost syntheses for most targets the seeds miss, in more than half of all
seeds within 1000 generations; with the secondary, selection gets there sooner than
uniform search. It is a positive control of the breeder.

**The pilots set the length.** The first pilot seeds 101 and 102 ran in both arms for 60
generations and again for 200 (`docs/pilot-step1/pilot.json`). Every run passed
`--verify`, none raised an integrity alarm or found an invention candidate, and in every
run the seed Cascades solved SWAP and CS only, so the gate held. Arm A reached D = 7/9
on seed 101, where five targets fell in generation 32, the first in which one Worker
named Hadamard, Phase and Linear together, and D = 2/9 on seed 102. Arm R reached 1/9 and
2/9 at 60 generations and 2/9 on both seeds at 200. The default cap of 20 000 calls
stopped arm A on seed 101 after generation 20. Later uncounted batches showed that a pass
often comes late: by generation 200 arm A passed on 6 of the 20 seeds 101 to 110 and 201
to 210, but by generation 1000 in 9 of the 11 Runs that reached it, with passes as late
as generation 810. At a pass rate of 0.30 the sequential rule fails with probability
1.000, so the counted Runs use `--generations 1000 --max-calls 13000000`: by our count from
the call bounds, a Run of 1000 generations needs at most 12 723 412 calls, so the cap
cannot end a Run early, and a pilot pair of 1000 generations took at most 14.31 min. At
the pilot rates the primary claim is proven with probability 0.86 to 0.99, and the
experiment needs 26 to 46 pairs on average, at most 120: 28.6 hours at the pace of the
slowest pilot pair. The fixed rule "9 of 10" would have passed with probability 0.165 at
the Kaplan-Meier rate of 0.710. The pass time of every pilot Run is in
`docs/pilot-step1/README.md`.

**Promotion waits.** Circuit Cascades are not promoted in step 1; `--promote` refuses
them. Promotion needs a reviewed candidate format for circuit Cascades, which step 1 does
not decide.

Rejected: a floating-point fidelity threshold as the criterion (it admits approximations);
references from unsourced targets or from our own generator (they break the rule that
every reference is verified before use); a weighted score such as 10 x T + CNOT (not
lexicographic, and it gives almost no pressure on CNOT); coupling the grid simulator to
circuits (the designer is its own backend); a phrase that names two of Hadamard, Phase and
Linear (one random step could then teach a seed all three, and five targets would fall at
once); a designer budget without a ceiling (a direct request could then run without
bound); a structure normal form, generated targets, Traits-only and reach-control arms,
record and invention events and per-task reports in step 1 (deferred, to keep the step
reviewable); a fixed number of seeds such as the 9-of-10 rule (the owner asked that the
experiment decide by itself how many seeds it needs); a fixed test checked again after
every pair (an exact 5 % binomial test then succeeds falsely with probability 0.0987 by
20 pairs); the threshold 1024/11 with at most 60 pairs, which keeps a bar of about 1 in
100 (0.0079) but proves the primary only with probability 0.696 at the Kaplan-Meier rate;
the threshold 20 with at most 60 pairs, approved first and replaced on 2026-10-30 when the
owner asked for the scientifically load-bearing version (for confirmatory claims like
these a one-sided 5 % level is lenient, and E = 20 is only "strong" evidence); the
threshold 40 with at most 100 pairs (a one-sided 2.5 % level, less strict than 1 %); the
threshold 1024/11 with at most 120 pairs (a statistical twin of the chosen design; 100 is
the round 1 % level and slightly stricter).

Status: accepted. The owner confirmed all six points on 2026-10-30: exact equality as the
criterion (it tightens Q38), the lexicographic score, the `state` line in grid logs, no
promotion of circuit Cascades in step 1, the sequential rule across seeds, which replaced
the 9-of-10 rule as point 5
(`.scratch/universum-erfinden/issues/09-sequential-rule-for-step-1.md`), and the panel
sizes 33 and 22. Later that day the owner asked for the scientifically load-bearing
version of point 5. After a review of 16 designs, Claude, the AI assistant, chose the
proof threshold 100, at most 120 pairs and the counted seeds 1 to 100, then 111 to 130
(issue 09, Comments). The pre-registration implements these points. It is frozen at the
one commit that the squash merge of its pull request creates on the main branch; with
that merge the owner confirms point 5 in this version.
