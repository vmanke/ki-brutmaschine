# Pre-registration draft: does breeding with a real model beat a majority vote?

Status: **draft, not frozen.** Claude, the AI assistant, wrote it on 2026-11-02 at the
owner's request. The owner decided the six decisions of section 10 on 2026-11-02, all as
recommended. A pilot runs next (section 9). After the pilot, this file gets its final
numbers in a pull request of its own, and it is frozen at the squash-merge commit of that
pull request, as in step 1 ([PREREG_STEP1_CIRCUITS.md](PREREG_STEP1_CIRCUITS.md), status
paragraph). Merging this draft, with its tools, freezes nothing; it puts the pilot's tools
on the main branch. No counted run starts before the freezing commit exists.

All runs made before the freeze are pilot data. They never count. This includes the
first real runs of 2026-10-31 and the runs of DGM against uniform selection that a VM
session started before the VM was stopped.

## 1. Question

**Primary question.** A real open model breeds Cascades on grid puzzles. Does the bred
champion solve more unseen puzzles than a majority vote that spends the same number of
model calls per puzzle?

The majority vote samples the best seed Cascade N times per puzzle. It keeps the most
frequent answer. N is the smallest number of samples for which the vote spends at least
as many Worker calls on the unseen puzzles as the champion did (section 4). This is
self-consistency (Wang et al. 2023). It is a strong and simple
baseline: in their study it beat a single greedy answer on nearly every model and task,
by up to 17.9 points on GSM8K. A gain
from breeding only counts if it goes beyond this baseline.

**Secondary questions** (section 7):

- S1. Does the champion beat one plain run of the best seed Cascade?
- S2. Does lexicase selection beat Darwin Gödel selection (arm L against arm B)?

**What this does not test.** The unseen puzzles come from the same generator as the
search puzzles, with another seed. So the experiment tests generalisation within one task
family, not to new kinds of tasks. The calls that breeding itself spends are a one-time
cost. They are reported, but the primary comparison is at equal calls per puzzle at use
time.

## 2. Why now, and what we know

- **First real runs** ([ERGEBNISSE_STUFE2C_ERSTE_LAEUFE.md](ERGEBNISSE_STUFE2C_ERSTE_LAEUFE.md)):
  the machine runs with real models (2704 recorded Worker calls). A breeding gain is not
  shown. The panels had 10 to 24 puzzles, which separates only differences above about
  30 percentage points.
- **Step 1** ([SCHRITT1_SCHALTUNGEN.md](SCHRITT1_SCHALTUNGEN.md)): with the simulated
  designer, breeding found most circuit targets that its seeds missed. That is a positive
  control of the breeder. It says nothing about real models.
- **Selection study, simulated** (PR #45, open, `docs/SELECTION_STUDY.md` there): with
  simulated models, the final incumbent of lexicase selection, chosen on training fitness
  alone, scored higher on the held-out panel than that of DGM in 17 of 30 seeds and lower
  in 5 (p about 0.05 after a correction for three arms); at a smaller call budget, 23 and 3.
  That result is exploratory and simulated. It motivates arm L, nothing more.
- **The control exists.** `vote-control` in `backend-ollama/` (PR #42) measures the
  compute-equal majority vote on a finished run's select panel. It picks the champion and
  the voted seed by their search results alone, never by a select result.

A pre-registration fixes the question, the arms, the outcome and the decision rule before
any counted data exist (Nosek et al. 2018). Then the result cannot be bent by choices made
after seeing it.

## 3. What is frozen at the freeze commit

| Item | Frozen as |
|---|---|
| Code | The whole tree of the freeze commit. Its hash is recorded with the results. |
| Programs | `brut-breed-ollama`, `brut-mutator`, `vote-control` and `breeding-vs-vote` of `backend-ollama/`, built with `cargo build --release` at that commit. |
| Toolchain | rustc 1.97.0 on x86_64 Linux. |
| Model | Decision D1: name, weights revision (Hugging Face commit hash, written into `vllm-start.sh` from the pilot) and served name. |
| Model server | vLLM 0.32.0 on 127.0.0.1, bf16, the start command `docs/prereg-breeding-vs-vote/vllm-start.sh`. `VLLM_USE_FLASHINFER_SAMPLER=0`, as measured on 2026-10-31. |
| Settings | One `config.json` per arm in `docs/prereg-breeding-vs-vote/`, from decisions D2 to D4. Every setting not named there stays at `BreedConfig::default()`. The Soul of step 2 stays off. |
| Seeds | The counted list 1, 2, 3, ... up to the cap of D6, in this order. Pilot seeds are 901 to 905; they are not on the list. |
| Analysis | The analysis program `breeding-vs-vote` and the independent recount `docs/prereg-breeding-vs-vote/recount.py` (section 11). |
| Rule | Sections 5 to 8. |

A change after the freeze is a deviation. It is reported with the results, with its
reason. A change after counted seed 1 has started makes a new experiment on new seeds.

## 4. The arms

For every counted seed s, in the order of the list:

- **Arm B (breeding, DGM selection).** `brut-breed-ollama --out runs/B-s --config
  docs/prereg-breeding-vs-vote/config-B.json --seed s` with the panel sizes and run length
  of D4. The model is the one of D1. Mutator rounds follow D2.
- **Arm V (vote control).** `vote-control --run runs/B-s --temperature T --samples N`
  with T from D5 and N from the call matching below. It samples the seed finalist with the
  best search fitness. Arm V runs only after arm B of the same seed has finished.
- **Call matching.** K_c(s) is the number of Worker calls the champion made in the select
  phase of run B-s, read from the run's call record. K_v(s, N) is the number of Worker
  calls the vote makes on the same puzzles with N samples. N_up(s) is the smallest N >= 1
  with K_v(s, N) >= K_c(s). The primary outcome uses N_up(s), so the vote never has fewer
  calls than the champion; this favours the vote. N_down(s) is the largest N >= 1 with
  K_v(s, N) <= K_c(s); its outcome is reported as a sensitivity check. Only call counts set
  N, never a right or wrong answer, and one N holds for every puzzle of a run.
- **Why not the default N.** The default of `vote-control` divides the champion's average
  calls per search puzzle by the seed's own and rounds. That does not match the calls on
  the unseen puzzles: a Stage spends extra calls on its vote only when its Workers give
  different answers, so a Cascade's calls per puzzle change from panel to panel. A
  sample's seed does not depend on N, so a larger N reuses the calls of a smaller one;
  finding N_up costs no calls beyond its own samples.
- **Arm L (lexicase selection, D3).** Like arm B with `--selection lexicase` and
  `docs/prereg-breeding-vs-vote/config-L.json`. This needs PR #45 merged before the
  freeze. Uniform parent choice does not run (D3).

Generation 0 is the same in every arm of a seed, since parent choice starts in
generation 1. The analysis refuses a seed where the seed Cascades' search answers differ
between arms. With a real model this holds only because arm L takes the generation-0 answers of arm
B from the seed's call cache `runs/call-cache/seed-S.jsonl`: vLLM is not bit-reproducible
across batches (contract I7), so fresh calls could differ.

**Length and call cap (D4).** Every run breeds 30 generations of 8 children on 40 search
puzzles and ends with the select phase on 100 puzzles; there is no early stop. The cap
`max_calls` comes from the call bounds in `src/cascade/arena.rs`. A Stage of w Workers
makes at most w + w · w(w - 1)/2 Worker calls on one puzzle: every Worker answers, then
every Worker judges every pair of distinct answers (`stage_call_bound`). A Cascade has at
most 4 Stages of at most 3 Workers (the hard ceilings), so it makes at most
4 · (3 + 3 · 3) = 48 calls per puzzle. The 4 seed Cascades have one Worker each, so 1 call.
A peer round makes at most one judge call per new member, opponent and puzzle
(`peer_call_bound`), with 3 opponents and 2 puzzles per pair.

- Generation 0: the 4 seeds on 40 puzzles, 160 calls, and at most 4 · 3 · 2 = 24 peer
  calls: 184.
- Each of the 30 bred generations: at most 8 children of 48 calls on 40 puzzles, 15 360
  calls, and at most 8 · 3 · 2 = 48 peer calls: 15 408. Children written by the Mutator
  must fit the same ceilings.
- The select phase: every seed and at most 3 finalists of 48 calls, on 100 puzzles:
  (4 + 3 · 48) · 100 = 14 800.

A run thus makes at most 184 + 30 · 15 408 + 14 800 = 477 224 Worker calls. The cap is
480 000, so it cannot end a run early: before each generation the arena checks that the
calls so far, the generation's worst case and the select phase's worst case fit, and here
they always do. Mutator calls do not count toward `max_calls`. The Mutator's answers reach
the run as answer files, not as Worker calls; `brut-mutator` records its calls apart, in
`llm/mutator-calls.jsonl`, and the report counts the answers used. The same holds for the
vote's calls, which `vote-control` makes after the run. `worst_case_calls` in
`backend-ollama/src/breeding_vs_vote.rs` computes this bound from a run's settings, and a
test pins it to `config-B.json`.

## 5. Definitions

- **Search panel and select panel.** Each run has both. The arena draws them from the run
  seed on two separate streams (`panel/search`, `panel/select`). Breeding reads only the
  search panel. The select panel is the unseen panel.
- **Overlap.** A select puzzle overlaps when its content id equals the id of a puzzle of
  the same seed's search panel. Overlapping select puzzles are left out of every count
  below. Their number is reported. The test misses only a select puzzle that equals a
  search puzzle and also repeats within the select panel, where the panel salts its id;
  with generated puzzles this has not been seen.
- **Champion c(s).** The bred finalist of run B-s with the best search fitness, as
  `vote.rs` (`champion`) defines it. No select result enters this choice.
- **C(s).** The number of select puzzles that c(s) solves exactly in the select phase of
  run B-s.
- **V(s).** The number of select puzzles that the majority vote of arm V solves exactly,
  with N_up(s) samples.
- **Outcome Y(s).** "breeding first" when C(s) > V(s); "vote first" when C(s) < V(s);
  "tie" when they are equal.

## 6. The primary rule

The rule is the sequential rule of step 1 (its section 8), applied to Y(s). Its design and
sources are in `.scratch/universum-erfinden/issues/09-sequential-rule-for-step-1.md`.

- **Evidence for breeding, E_B.** It starts at 1. Pairs enter strictly in the order of the
  seed list, never in the order in which runs finish. E_B is multiplied by 3/2 on
  "breeding first", by 1/2 on "vote first", and stays on a tie. After w wins and l losses,
  E_B = 3^w / 2^(w + l).
- **The factors.** 3/2 and 1/2 are the likelihood ratio of a win rate of 3/4 against 1/2
  among untied pairs.
- **Stop with a claim.** The experiment stops at the first pair where E_B >= 100. The
  claim is: breeding beats the vote.
- **One direction only.** The rule cannot claim that the vote beats breeding. The mirrored
  value E_V (3/2 on "vote first", 1/2 on "breeding first") satisfies
  E_B · E_V = (3/4)^(w + l), so the futility stop of E_B comes first: five straight losses
  end the experiment, while E_V needs twelve to reach 100. E_V is reported at the stop pair,
  without a claim.
- **Stop for futility.** It stops when E_B <= 1/20. Then the claim is: no evidence that
  breeding wins in three of four untied seeds.
- **Cap.** It stops at the cap of D6. Without a claim by then, the result is "open".
- **Integers only.** E_B >= 100 is checked as 3^w >= 100 · 2^(w + l), and E_B <= 1/20 as
  20 · 3^w <= 2^(w + l).

**Error bound.** Under the null hypothesis (a win rate of at most 1/2 among untied pairs),
E_B is a nonnegative supermartingale that starts at 1. By Ville's inequality, the chance
that it ever reaches 100 is at most 1/100 (Shafer 2021; Ramdas et al. 2023). The bound
holds at every stopping time, so looking after each pair costs nothing. Each factor has
mean 1/2 + p <= 1 under the null, where p is the win rate among untied pairs; a tie has the
factor 1.

**Operating characteristics** (`docs/prereg-breeding-vs-vote/oc.py`, exact dynamic
programming; the tie rate 0.2 is an assumption that the pilot measures):

| Cap | Win rate among untied pairs | P(proof) | P(futility) | P(open) | Expected pairs |
|---|---|---|---|---|---|
| 60 | 0.50 | 0.006 | 0.914 | 0.080 | 26.4 |
| 60 | 0.70 | 0.512 | 0.116 | 0.372 | 44.5 |
| 60 | 0.75 | 0.778 | 0.034 | 0.188 | 38.7 |
| 60 | 0.80 | 0.942 | 0.007 | 0.051 | 31.3 |
| 120 | 0.50 | 0.008 | 0.987 | 0.005 | 28.1 |
| 120 | 0.70 | 0.777 | 0.143 | 0.080 | 56.1 |
| 120 | 0.75 | 0.950 | 0.037 | 0.013 | 42.8 |
| 120 | 0.80 | 0.992 | 0.007 | 0.001 | 32.0 |

So the rule finds a strong effect (breeding wins about three of four untied seeds). It
does not find a small one. That is a design choice: a small gain over a cheap vote would
not justify the machinery.

## 7. Secondary analyses

They are read at the pair where the primary rule stops. They never stop the experiment.

- **S1.** Y1(s) compares C(s) with the select result of one plain run of the voted seed
  Cascade, from run B-s.
- **S2.** Y2(s) compares C_L(s), the count of section 5 for the champion of run L-s, with
  C(s).

Each gets one evidence value, built as in section 6, in the direction named above: a win
of the champion of run B-s (S1) or of lexicase (S2) multiplies it by 3/2. A secondary claim
needs E >= 200 in that direction; the other direction gets no claim. By Ville's inequality
each false claim has a chance of at most 1/200, so by the union bound the chance of any
false secondary claim is at most 2/200 = 1/100 (Vovk and Wang 2021 on combining
e-values).

**Reported for every pair, without a claim:** C(s), V(s), N_up(s), N_down(s) and the
outcome with N_down(s), K_c(s) and K_v(s, N_up(s)), the per-puzzle McNemar
counts and the sign-test p-value of `vote-control`, calls and tokens of breeding and of
the vote, the champion's Worker calls per puzzle, the overlap count, failed calls, and
wall time.

## 8. Procedure and validity

1. Build the programs at the freeze commit: `cd backend-ollama && cargo build --release`.
   Run `cargo test` in the root package and in `backend-ollama/`. All must pass.
2. Start vLLM with the frozen command: `docs/prereg-breeding-vs-vote/vllm-start.sh`. It
   prints the vLLM version and `nvidia-smi`; record both.
3. Run the counted seeds in list order, each with
   `docs/prereg-breeding-vs-vote/run-seed.sh --seed S` from the repository root, which
   writes `runs/B-S` and `runs/L-S`. Several seeds may run at once; the evidence values
   still take the pairs in list order. The driver breeds arm B, runs arm V on it (the call
   matching of section 4, then `vote-control --run runs/B-S --temperature 0.7 --samples N`
   for N_up and for N_down), breeds arm L, and writes `runs/B-S/panels.json`.
4. A run with any failed call is not counted. The driver stops as soon as a run has
   filed one. The run is deleted (`rm -r runs/B-S` or `rm -r runs/L-S`) and run again
   from scratch on the same seed with the same command. `vote-control` refuses such a run
   anyway. "From scratch" means a new run directory, not new model answers: the run takes
   every answer already in the seed's call cache, and only its failed calls go to the
   model again. So deleting a run cannot draw new answers for a seed, and no seed can be
   rolled again until it looks better.
5. A run that is killed or interrupted is not resumed. It is deleted and run again from
   scratch. A pause for Mutator answers (exit code 3), answered by `brut-mutator` and
   continued with `--resume`, is part of the design, not an interruption (D2). The driver
   resumes a run only after such a pause; its journal `driver.log` in each run directory
   tells the two apart, and for any other unfinished run it stops and says what to delete.
6. After each run: `brut-breed-ollama --out <dir> --verify` must pass; the driver runs it
   and stops when it fails. The standalone `brut-breed` knows only the simulated models,
   and its `--verify` refuses a run whose allow-list names a `vllm:` model. After each
   pair: run the analysis on all finished pairs,
   `backend-ollama/target/release/breeding-vs-vote --runs runs`; the driver runs it last.
   It names the next seed or gives the verdict, and writes `runs/analysis.json` and
   `runs/analysis.txt`.
7. Before the result is reported, the independent recount
   `python3 -I docs/prereg-breeding-vs-vote/recount.py --runs runs` must give the same
   outcomes, evidence values, stop pair and verdicts.

**The experiment is void** when a frozen item changes during the counted runs (model,
weights, server version, code), or when the recount and the analysis disagree and the
cause is not found.

## 9. The pilot (not counted)

The pilot runs seeds 901, 902 and 903 at the same time, with all arms and the settings
of D1 to D5, in one command: `docs/prereg-breeding-vs-vote/pilot.sh`. It starts vLLM, runs
the three seeds (six breeding runs, the load that filled the measured H200), writes
`runs/pilot/summary.txt` and stops vLLM; then the VM can be stopped. The VM is rented by
the hour, so it runs only for the pilot and, after the freeze, for the counted runs.
Seeds 904 and 905 stay in reserve for a second pilot after a change of model. The pilot
measures:

- calls, tokens and minutes per run on the H200, with several runs at once;
- the best seed Cascade's search rate;
- the tie rate of Y(s), which section 6 assumes to be 0.2;
- that Mutator rounds (if D2 is on) pass `--verify` and replay byte-identically;
- the longest Mutator prompt in tokens, and how many Mutator requests vLLM refused as too
  long (`llm/mutator-errors.jsonl`).

**Pilot gate for the model.** If the best seed Cascade solves fewer than 15 % of the search
puzzles in the median pilot seed, the model has no selection signal. Then D1 changes to a
stronger model before the freeze, and the pilot runs again. 15 % is about the mean unseen
rate of the four seeds with `qwen2.5:32b` on 2026-10-31 (15.6 %).

With the measured minutes per pair, the owner confirms the cap of D6. Nothing here
forecasts money.

## 10. Decisions of the owner

The owner decided all six on 2026-11-02 as recommended ("alles wie empfohlen"). D3 was
first drafted as "B and V; add U". After PR #45 measured a clear lexicase lead on the
clean measure and no difference for uniform choice, the recommendation became B, V and L,
and that version was decided.

| # | Decision | Decided | Reason |
|---|---|---|---|
| D1 | Model | `Qwen2.5-32B-Instruct`, bf16, on vLLM | Measured on this H200: 40 times the throughput of Ollama. Its seeds left the floor (6 of 24 unseen). `qwen2.5:7b` stayed at the floor. Alternatives: `Qwen2.5-72B-Instruct` (stronger, slower) or the thinking model `qwen3.8-27b` (contract I5; more tokens per call). |
| D2 | Mutator rounds | On: `--llm-every 2`, answered by `brut-mutator` with the same model | The machine's idea is that a model rewrites the prompts. Without the Mutator, breeding only recombines phrases. The pilot checks the pause and resume loop. |
| D3 | Arms | B, V and L | The simulation of PR #45 predicts a lexicase lead (S2); uniform choice showed no difference there. Each breeding arm adds one run per seed, so U does not run. L needs PR #45 merged before the freeze. |
| D4 | Panels and length | 40 search, 100 select, 8 children, 30 generations; a call cap that cannot bind | 100 unseen puzzles make each pair's count finer and ties rarer. A fixed number of generations gives every arm the same number of children, as in step 1. The cap is set from the worst case per generation, as step 1 did (its section 6): a run makes at most 477 224 Worker calls, and the cap is 480 000 (section 4). |
| D5 | Vote temperature | T = 0.7 | Wang et al. (2023) used T = 0.7 for PaLM and GPT-3 and found self-consistency robust between 0.3 and 0.7. A seed's own temperature (0.2) would make the samples near copies and weaken the control. |
| D6 | Cap on pairs | 60, confirmed after the pilot | At a win rate of 3/4, the chance of a proof is 0.78 with cap 60 and 0.95 with cap 120 (section 6). |

## 11. Work before the freeze

**Built.** Everything below lies in this repository; the tests run with `cargo test` in
`backend-ollama/`.

- **The analysis program** `breeding-vs-vote`: the module
  `backend-ollama/src/breeding_vs_vote.rs` with its unit tests, and the thin binary
  `backend-ollama/src/bin/breeding-vs-vote.rs`. It takes the champion, the voted seed, the
  select panel and the vote's sample seeds from `backend-ollama/src/vote.rs`, so its
  definitions are those of `vote-control`. `--runs <dir>` reads the pairs of the counted
  list in its order (`--pilot`: the pilot seeds, without a claim), checks them, computes
  the values of sections 4 to 7 and runs the rule of section 6. It prints a table and the
  verdict lines and writes `analysis.json` and `analysis.txt`. `--match <run B-s>` is the
  call matching for the driver; `--panels <run B-s>` writes `panels.json`.
- **What it refuses.** A run whose `config.json` differs from the frozen file of its arm
  in anything but the seed; a run whose name and seed disagree; a run with a failed
  Worker call; a finished run that did not breed all 30 generations; a broken event log,
  a checkpoint not at its end, logged panels or select results that differ from the
  regenerated or reported ones; a seed whose arms differ in generation 0 (section 4); a
  record that lacks a call of the champion's select phase; a kept vote result that is not
  the vote its record gives with N_up or N_down samples; a `panels.json` that differs from
  the regenerated panels; a seed off the list 1 to 60, a pilot seed among counted ones, a
  gap in the list, and a 61st pair. An unfinished run is not counted: its pair and every
  later pair wait, as a pair with no kept vote yet does.
- **The overlap check of section 5** in both programs: a select puzzle whose id is an id
  of the seed's search panel counts nowhere, neither in C, V, C_L and the plain run nor in
  K_c and K_v. The analysis reports the overlapping ids per pair.
- **The call matching of section 4.** K_c(s) replays the champion's select phase on the
  counted puzzles from the run's record `llm/worker-calls.jsonl`; every call must be on
  file, and the replayed answers must be the snapshot's. K_v(s, N) replays the first N
  samples from the vote's record `vote-control/worker-calls.jsonl`. A sample makes at
  least one call per Worker and puzzle, so K_v grows with every sample, and N_down is
  N_up when K_v(s, N_up) = K_c(s) and N_up - 1 otherwise (none when that is 0). The
  analysis checks that N_up is the smallest and N_down the largest N that section 4
  allows.
- **How the vote is kept.** `vote-control` writes its result to
  `runs/B-s/vote-control.json`, and a later control overwrites it. The driver therefore
  copies it right after `vote-control --samples N_down` to `runs/B-s/vote-down.json` and
  right after `--samples N_up` to `runs/B-s/vote-up.json`. A larger N reuses the samples
  of a smaller one, so both controls only replay the record. The analysis reads only these
  two copies and checks each one against the record, puzzle by puzzle. No
  `vote-down.json` exists when no N_down exists.
- **The independent recount** `docs/prereg-breeding-vs-vote/recount.py`, for
  `python3 -I` and the standard library alone. It shares no code with the analysis. It
  reads the same files and computes the same outcomes, evidence values, stop pair and
  verdicts by other means: it counts calls by their request seeds instead of replaying
  Cascades, and it grades every answer itself against the puzzles of `panels.json`, which
  it checks against their logged content ids. The analysis tests run it on the test runs
  and compare its result with their own, and run its rule on 300 random and 9 chosen
  outcome sequences.
- **The settings** `docs/prereg-breeding-vs-vote/config-B.json` and `config-L.json`.
  D1: the one model `vllm:qwen2.5-32b`, at 0.002 fitness units per call, the price of the
  first real runs (ERGEBNISSE_STUFE2C_ERSTE_LAEUFE.md, section 2). D2: `llm_every` 2 and
  `llm_children` 8. D3: `"selection": "lexicase"` in `config-L.json` only. D4: 40 search
  and 100 select puzzles, 8 children, 30 generations and `max_calls` 480 000 (section 4).
  Everything else is `BreedConfig::default()` as it serialises; a test pins both files to
  that, byte for byte, and another runs `brut-breed-ollama` on both.
- **Why `llm_children` 8.** A Mutator round plans `min(llm_children,
  children_per_generation)` children written by the Mutator, alternately a rewrite of one
  parent and a merge of two, and fills the rest of the 8 children by random mutation and
  crossover; an answer that is no valid new Cascade is replaced by a random child. With
  the default of 2, only 2 of the 8 children in every second generation, one child in 8,
  would come from the model, and breeding would still mostly recombine phrases, which is
  what D2 wants to go beyond. With 8, every child of a Mutator round comes from the model,
  so half of all children do, and the generations in between recombine what it wrote.
  Mutator calls are no Worker calls (section 4), so the larger share changes neither the
  cap nor the call matching. A round asks 8 requests, which `brut-mutator` answers in one
  batch.
- **The vLLM start command** `docs/prereg-breeding-vs-vote/vllm-start.sh`, as measured
  on 2026-10-31: vLLM 0.32.0, `Qwen/Qwen2.5-32B-Instruct` in bf16, served as
  `qwen2.5-32b` on 127.0.0.1:8000, `VLLM_USE_FLASHINFER_SAMPLER=0`. It reads the weights
  from the local Hugging Face cache only. The context is 32 768 tokens, the model's native
  length, not the 8 192 of the measurement: a Mutator prompt that merges two large Cascades
  measured about 20 000 characters, roughly 5 000 tokens, and with 4 096 answer tokens it
  would not fit 8 192. vLLM would refuse it every time, and the run would never finish.
- **Too-long Mutator requests.** As a second guard, `brut-mutator` writes an empty answer
  for a request that vLLM refuses as longer than the context (HTTP 400, "maximum context
  length"), and the core fills a random child, as for a cut-off answer. Every other failed
  Mutator call stays pending. The refusal stays on file in `llm/mutator-errors.jsonl`.
- **The pilot** `docs/prereg-breeding-vs-vote/pilot.sh`: section 9 in one command. Its
  tests run it on stand-in programs: three seeds at once, the summary, and a seed that
  stops.
- **The driver** `docs/prereg-breeding-vs-vote/run-seed.sh` for one seed: the procedure of
  section 8, steps 3 to 6. It never deletes anything. Both arms of a seed share the call
  cache `runs/call-cache/seed-S.jsonl`, so arm L takes generation 0 from arm B. Its tests
  run it on stand-in programs and once on the real programs with a stand-in model server.
- **The worst-case call bound of D4**, with its arithmetic: section 4.

**Chosen by Claude, open to the owner.** `llm_children` 8, the price 0.002 per call, the
context of 32 768 tokens and the too-long rule above are not part of D1 to D6. Claude
chose them for the reasons given; the owner can change any of them before the freeze.

**Still open before the freeze.** The weights revision: the pilot reads it from the VM's
Hugging Face cache and writes it into `vllm-start.sh`, which refuses counted use without
it. PR #45 must be merged (arm L). The pilot of section 9 runs, and the owner confirms
the cap of D6.

## 12. Sources

- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E. H., Narang, S., Chowdhery, A., and
  Zhou, D. (2023). Self-consistency improves chain of thought reasoning in language
  models. ICLR 2023. (Sampling settings: section 3.1; robustness to T: section 3.5.)
- Nosek, B. A., Ebersole, C. R., DeHaven, A. C., and Mellor, D. T. (2018). The
  preregistration revolution. PNAS 115(11), 2600–2606.
- Shafer, G. (2021). Testing by betting: a strategy for statistical and scientific
  communication. Journal of the Royal Statistical Society A 184(2), 407–431.
- Ramdas, A., Grünwald, P., Vovk, V., and Shafer, G. (2023). Game-theoretic statistics and
  safe anytime-valid inference. Statistical Science 38(4), 576–601.
- Vovk, V., and Wang, R. (2021). E-values: calibration, combination and applications.
  Annals of Statistics 49(3), 1736–1754.
- Helmuth, T., Spector, L., and Matheson, J. (2015). Solving uncompromising problems with
  lexicase selection. IEEE Transactions on Evolutionary Computation 19(5), 630–643.
- La Cava, W., Spector, L., and Danai, K. (2016). Epsilon-lexicase selection for
  regression. GECCO 2016, 741–748.
- Guo, Q. et al. (2024). Connecting large language models with evolutionary algorithms
  yields powerful prompt optimizers (EvoPrompt). ICLR 2024.
- Fernando, C., Banarse, D., Michalewski, H., Osindero, S., and Rocktäschel, T. (2024).
  Promptbreeder: self-referential self-improvement via prompt evolution. ICML 2024.
