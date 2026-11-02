# Pre-registration draft: does breeding with a real model beat a majority vote?

Status: **draft, not frozen.** Claude, the AI assistant, wrote it on 2026-11-02 at the
owner's request. Six decisions are open (section 10). Each has a recommendation. The
owner decides them in the pull request of this file. A pilot then runs (section 9). After
the pilot, this file gets its final numbers and is frozen at the squash-merge commit of
its pull request, as in step 1 ([PREREG_STEP1_CIRCUITS.md](PREREG_STEP1_CIRCUITS.md),
status paragraph). No counted run starts before that commit exists.

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
- S2. Does Darwin Gödel selection beat uniform parent choice, if arm U runs?
- S3. Does lexicase selection beat Darwin Gödel selection, if arm L runs?

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
  simulated models, the best finalist of lexicase selection scored higher on the held-out
  panel than that of DGM in 18 of 30 seeds, lower in 5. That result is exploratory and simulated. It motivates arm L, nothing more.
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
| Programs | `brut-breed-ollama` and `vote-control` of `backend-ollama/`, built with `cargo build --release` at that commit. |
| Toolchain | rustc 1.97.0 on x86_64 Linux. |
| Model | Decision D1: name, weights revision (Hugging Face commit hash) and served name. |
| Model server | vLLM 0.32.0 on 127.0.0.1, bf16, the start command written into `docs/prereg-breeding-vs-vote/`. `VLLM_USE_FLASHINFER_SAMPLER=0`, as measured on 2026-10-31. |
| Settings | One `config.json` per arm in `docs/prereg-breeding-vs-vote/`, from decisions D2 to D4. Every setting not named there stays at `BreedConfig::default()`. The Soul of step 2 stays off. |
| Seeds | The counted list 1, 2, 3, ... up to the cap of D6, in this order. Pilot seeds are 901 to 905; they are not on the list. |
| Analysis | An analysis program and an independent recount, both to be built before the freeze (section 11). |
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
- **Arm U (uniform selection), if D3 says so.** Like arm B with `--selection uniform`.
- **Arm L (lexicase selection), if D3 says so.** Like arm B with `--selection lexicase`.
  This needs PR #45 merged before the freeze.

Generation 0 is the same in every arm of a seed, since parent choice starts in
generation 1. The analysis refuses a seed where the seed Cascades' search answers differ
between arms.

## 5. Definitions

- **Search panel and select panel.** Each run has both. The arena draws them from the run
  seed on two separate streams (`panel/search`, `panel/select`). Breeding reads only the
  search panel. The select panel is the unseen panel.
- **Overlap.** A select puzzle overlaps when its content id equals the id of a puzzle of
  the same seed's search panel. Overlapping select puzzles are left out of every count
  below. Their number is reported.
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
- **S2.** Y2(s) compares C(s) with C_U(s), the same count for arm U.
- **S3.** Y3(s) compares C_L(s) with C(s).

Each gets one evidence value, built as in section 6, in the direction named above: a win
of the champion (S1, S2) or of lexicase (S3) multiplies it by 3/2. A secondary claim needs
E >= 300 in that direction; the other direction gets no claim. By Ville's inequality each
false claim has a chance of at most 1/300, so by the union bound the chance of any false
secondary claim is at most 3/300 = 1/100 (Vovk and Wang 2021 on combining e-values).

**Reported for every pair, without a claim:** C(s), V(s), N_up(s), N_down(s) and the
outcome with N_down(s), K_c(s) and K_v(s, N_up(s)), the per-puzzle McNemar
counts and the sign-test p-value of `vote-control`, calls and tokens of breeding and of
the vote, the champion's Worker calls per puzzle, the overlap count, failed calls, and
wall time.

## 8. Procedure and validity

1. Build both programs at the freeze commit. Run `cargo test` in the root package and in
   `backend-ollama/`. All must pass.
2. Start vLLM with the frozen command. Record `nvidia-smi` and the vLLM version.
3. Run the counted seeds in list order. Several runs may run at once; the evidence values
   still take the pairs in list order.
4. A run with any failed call is not counted. It is deleted and run again from scratch on
   the same seed. `vote-control` refuses such a run anyway.
5. A run that is killed or interrupted is not resumed. It is deleted and run again from
   scratch. A pause for Mutator answers (exit code 3), answered by `brut-mutator` and
   continued with `--resume`, is part of the design, not an interruption (D2).
6. After each run: `brut-breed-ollama --out <dir> --verify` must pass. The standalone
   `brut-breed` knows only the simulated models, and its `--verify` refuses a run whose
   allow-list names a `vllm:` model. After each pair: run the
   analysis on all finished pairs. It names the next seed or gives the verdict.
7. Before the result is reported, the independent recount must give the same outcomes,
   evidence values, stop pair and verdicts.

**The experiment is void** when a frozen item changes during the counted runs (model,
weights, server version, code), or when the recount and the analysis disagree and the
cause is not found.

## 9. The pilot (not counted)

Pilot seeds 901 to 905 run all chosen arms with the settings of D1 to D5. The pilot
measures:

- calls, tokens and minutes per run on the H200, with several runs at once;
- the best seed Cascade's search rate;
- the tie rate of Y(s), which section 6 assumes to be 0.2;
- that Mutator rounds (if D2 is on) pass `--verify` and replay byte-identically.

**Pilot gate for the model.** If the best seed Cascade solves fewer than 15 % of the search
puzzles in the median pilot seed, the model has no selection signal. Then D1 changes to a
stronger model before the freeze, and the pilot runs again. 15 % is about the mean unseen
rate of the four seeds with `qwen2.5:32b` on 2026-10-31 (15.6 %).

With the measured minutes per pair, the owner confirms the cap of D6. Nothing here
forecasts money.

## 10. Decisions for the owner

| # | Decision | Recommendation | Reason |
|---|---|---|---|
| D1 | Model | `Qwen2.5-32B-Instruct`, bf16, on vLLM | Measured on this H200: 40 times the throughput of Ollama. Its seeds left the floor (6 of 24 unseen). `qwen2.5:7b` stayed at the floor. Alternatives: `Qwen2.5-72B-Instruct` (stronger, slower) or the thinking model `qwen3.8-27b` (contract I5; more tokens per call). |
| D2 | Mutator rounds | On: `--llm-every 2`, answered by `brut-mutator` with the same model | The machine's idea is that a model rewrites the prompts. Without the Mutator, breeding only recombines phrases. The pilot checks the pause and resume loop. |
| D3 | Arms | B and V; add U | U answers whether the selection matters at all (S2), as in step 1. L only if PR #45 is merged; each arm adds one breeding run per seed. |
| D4 | Panels and length | 40 search, 100 select, 8 children, 30 generations; a call cap that cannot bind | 100 unseen puzzles make each pair's count finer and ties rarer. A fixed number of generations gives every arm the same number of children, as in step 1. The cap is set from the worst case per generation, as step 1 did (its section 6). |
| D5 | Vote temperature | T = 0.7 | Wang et al. (2023) used T = 0.7 for PaLM and GPT-3 and found self-consistency robust between 0.3 and 0.7. A seed's own temperature (0.2) would make the samples near copies and weaken the control. |
| D6 | Cap on pairs | 60, to be confirmed after the pilot | At a win rate of 3/4, the chance of a proof is 0.78 with cap 60 and 0.95 with cap 120 (section 6). |

## 11. Work before the freeze

- The analysis program (Rust, like `examples/circuit_positive_control.rs`) and an
  independent recount (Python, run with `python3 -I`, like step 1's `reference_rule.py`).
  Both read the run directories and `vote-control.json`, check the frozen settings,
  compute Y(s), the evidence values and the verdicts.
- The overlap check of section 5, in the analysis.
- The call matching of section 4: read K_c(s) from the run's call record, and find
  N_up(s) and N_down(s) from the vote's call record.
- `config-B.json` (and `config-U.json`, `config-L.json`), the vLLM start command and the
  model's weights revision, in `docs/prereg-breeding-vs-vote/`.
- The worst-case call bound per generation for D4, written down with its arithmetic.

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
