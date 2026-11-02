# Selection study: lexicase, held-out evaluation, champion replay, down-sampling

Research question: does evaluating candidates on single tasks, and keeping historical reference points, breed better Cascades than selecting on one aggregate fitness?

Scope: Stage 2c Cascade breeding (`src/cascade`, grid-puzzle domain, **simulated** models). Learner and MLP/CNN modes are untouched. Numbers below say nothing about real language models.

## What already existed (not rebuilt)

- Parent selection: `ParentSelection::{Dgm, Uniform}` in `arena.rs` (DGM weights from fitness and child counts).
- A separate **select panel** unseen during search, scored only for finalists and seeds at the end (`finish_with`). It is the held-out set; the study reuses it.
- Per-task answers are stored in each `Member`; per-call seeds derive from `(run seed, Cascade id, task id)`, so a Cascade replayed later gets identical conditions.
- Per-generation `state` events (diversity, lineage, cost); call budget `max_calls`; hash-chained logs.

## What was added

| Piece | Where |
|---|---|
| Lexicase (epsilon form) and tournament selection | `src/cascade/selection.rs`, `ParentSelection::{Lexicase, Tournament}` |
| Settings `selection_params` (`lexicase_epsilon`, `tournament_size`) | `BreedConfig`; omitted from config and hash at defaults, so earlier runs hash as before |
| CLI | `brut-breed --selection dgm\|uniform\|lexicase\|tournament --lexicase-epsilon E --tournament-size K` |
| Measurement observer, champion vault, replay, gap, subset ranking, finalist re-evaluation | `src/cascade/study.rs` |
| Comparative experiment | `examples/selection_study.rs` |

### Algorithms

- **Lexicase**: per selection event, shuffle the search tasks with the generation's seeded RNG; keep candidates within `epsilon + 1e-12` of the best remaining value on each task; stop at one survivor; draw uniformly among survivors if tasks run out. Per-task value is the graded value of the stored answer (0 for no answer). No aggregate, judging skill, peer rating or cost enters. Crossover partners are the first valid draw (like `uniform`).
- **Tournament**: best-ranked (fitness, rating, id) of `tournament_size` uniform draws with replacement. Its logged pick chances are exact: ((n − r)/n)^k − ((n − r − 1)/n)^k for rank r of n members and size k, finite for every size.
- **Dgm / uniform**: unchanged; they consume the RNG exactly as before.
- The logged `parent_entropy_bits` for lexicase is estimated from 256 selections with a separate fixed seed; it never touches the run's RNG.

### Held-out isolation

The breeder never reads the select panel during search. `study.rs` is an *observer*: it steps the breeder, reads the archive, scores each generation's incumbent (best by fitness) on the select panel, and stores results in its own structures. Test `the_observer_does_not_change_the_breeding` compares the breeder's snapshot after every step with and without that scoring (byte equal). Gap = incumbent train rate − held-out rate, per generation, with its least-squares slope per run. Observer calls are reported separately and are outside `max_calls`.

The study grades some answers itself: the held-out answers of the incumbents and the answers of the full re-evaluation. It grades them as the breeder does (`arena::checked_grade`): an integrity alarm of the domain, or a value outside [0, 1], stops the study with that error. Only a missing answer counts as 0.

Caveat: the study picks the incumbent per generation on training fitness only, but the final reported "best finalist" is from the existing select phase, which is also what a user would pick; the held-out panel is therefore a final-report panel, not a tuning panel. Do not tune settings on it.

### Champion replay

Each time the incumbent changes, it is frozen into a `ChampionVault` (private fields, copies, append-only, no mutator; tested). Newborns are compared with every earlier champion task by task on the search panel (same seeds, stored answers): outperform / match / regress by mean difference with a tolerance. New champions are also compared with earlier champions on held-out tasks. **Rediscovery** = a newborn whose answer list on the search panel equals that of an earlier member with a different Cascade. **Cycle** = a new champion behaving like a champion older than its predecessor.

### Down-sampled evaluation

Two measurements, no assumption of equivalence:

1. Offline: on final dgm archives, rank members on random subsets of the search tasks vs all tasks (Kendall tau-b, top-1 hit rate, top-1 regret, top-3 overlap).
2. End to end: dgm with a smaller `search_puzzles` at the same call budget; its bred finalists are scored again on the full panel (`reevaluate_finalists`). Both picks use one rule: the finalist with the best training rate, the subset pick on the run's subset and the full-panel pick on the full panel; ties go to the finalist the run ranks higher. So the pick changes only when the full panel ranks the finalists differently. The full-panel pick is the result; the subset pick is only reported for comparison. Re-evaluation calls are outside the run budget; the Part B table reports them (median and range). They include re-running the subset's own tasks, whose answers the run already holds.

## Experiment

`cargo run --release --example selection_study -- --seeds 30 --out docs/selection-study/main`

Per seed, all four arms share: panels, seed Cascades (generation 0 identical, tested), operators and rates, models, call budget (`max_calls` 40 000 for the main run; 12 000 for `low-budget`, `--max-calls 12000`), 40 search + 40 held-out tasks, 8 children per generation, 3 finalists, epsilon 0, tournament size 3. Seeds 1–30; every run ended by the call budget. Elapsed time is host wall clock (noisy). Reproducible: identical output apart from wall seconds (tested). Committed: `report.md` of each run (full tables, medians/quartiles/min–max). `runs.json` (every run record, the subset ranking quality and the full re-evaluations) is not committed (size); regenerate it.

## Results (measured; 30 seeds, simulated models)

Held-out rate of the best bred finalist, median [p25, p75] (picked on the held-out panel, so optimistic; the clean measure follows the table):

| arm | 40k calls | 12k calls |
|---|---|---|
| dgm | 0.675 [0.625, 0.750] | 0.625 [0.575, 0.700] |
| uniform | 0.700 [0.625, 0.725] | 0.650 [0.575, 0.700] |
| lexicase | 0.750 [0.700, 0.850] | 0.700 [0.650, 0.750] |
| tournament | 0.700 [0.650, 0.750] | 0.650 [0.575, 0.700] |

Paired per seed against dgm at 40k (better / equal / worse): lexicase 18/7/5 (two-sided sign test on untied seeds p ≈ 0.011, unadjusted); tournament 12/7/11 (no evidence); uniform 9/6/15 (no evidence). At 12k: lexicase 19/7/4. Best seed Cascade is ≈0.33 in every arm, so all arms improve on the seeds.

**The headline number is optimistic.** It takes the best of the three bred finalists on the held-out panel itself, so the panel helps to pick what it then scores. Part B shows the size: for dgm-subset-10 the best of the finalists scores 0.700, while a pick made without the held-out panel scores 0.625 (subset pick) or 0.675 (full-panel pick). The final incumbent is chosen on training fitness alone, so its held-out rate is the clean measure. Paired against dgm (better / equal / worse): at 40k lexicase 17/8/5 (sign test p ≈ 0.017, unadjusted; about 0.05 after a Holm correction over the three arms), tournament 13/7/10, uniform 7/9/14; at 12k lexicase 23/4/3 (p < 0.001), tournament 11/10/9, uniform 12/6/12. Medians at 40k: dgm 0.675, lexicase 0.750, tournament 0.700, uniform 0.675; at 12k: dgm 0.625, lexicase 0.700. So the lexicase lead holds on the clean measure. It stays exploratory: simulated models, and both budgets use the same seeds.

Other measurements at 40k, median:

- **Generalisation gap** (final incumbent): 0.025 in every arm; per-seed gap differences vs dgm are balanced. The gap slope is ≈0 in every arm. **No overfitting trend and no arm difference was observed** at this budget.
- **Diversity**: distinct behaviours in the final archive: dgm 143, uniform 177, tournament 144, **lexicase 94** (28/30 seeds lower than dgm). Parent-draw entropy: lexicase 4.5 bits vs dgm 6.4. So lexicase did **not** increase archive diversity here; it concentrated search while scoring higher.
- **Cost**: calls are budget-matched (~38.4k). Lexicase completed fewer generations (31 vs 37) and evaluated fewer Cascades (244 vs 292), i.e. chose more expensive (more tokens: 4.7M vs 4.4M) Cascades. Wall time is similar (≈0.75 s).
- **Champions**: champions in the final lineage: dgm 0.78, lexicase 0.67, tournament 0.78, uniform 0.43. Lexicase had slightly more champion cycles (median 1 vs 0) and ≈ equal rediscovery counts. Share of earlier champions surpassed by the final: dgm 0.86, lexicase 0.78.
- **Down-sampling, offline** (40k): 5/40 tasks (12% of cost): tau 0.63, top-1 hit 0.19, regret 0.14; 10/40: tau 0.76, hit 0.53, regret 0.05; 20/40: tau 0.87, hit 0.80, regret 0.01. Subsets do **not** reproduce the full ranking.
- **Down-sampling, end to end** (same call budget): subset runs complete 1.7–3× more generations and evaluate more Cascades (10 tasks: 940, 20 tasks: 516, vs 292), the best of their finalists on the held-out panel scores like dgm (10 tasks: 0.700, 20 tasks: 0.675, vs 0.675; optimistic, as above), but full re-evaluation changed the finalist pick in 8/30 (10 tasks) and 10/30 (20 tasks) seeds at 40k, and the full-panel pick had higher median held-out (0.675 vs 0.625 and 0.650). So the final choice must be re-evaluated on the full set. At 12k the pick changed in 7/30 (10 tasks) and 4/30 (20 tasks) seeds; the median held-out of the pick rose with 10 tasks (0.625 to 0.650) but not with 20 tasks (0.650 both). The re-evaluation took a median of 480 calls per run at 40k (about 1.2% of the budget; outside it).

## Limitations

- One task family (grid puzzles), simulated models whose skill depends on prompt and traits; not evidence for LLM cascades. Real-model results need the host (ADR 0003).
- Search panel is 40 tasks, so rates move in steps of 0.025; held-out also 40 tasks.
- Many arms and metrics, no multiplicity correction; the p-value above is exploratory. The lexicase advantage is one result at two budgets with the same seeds (not independent confirmation).
- Lexicase epsilon and tournament size were not tuned (epsilon 0, size 3); no ablation of why lexicase helps (e.g. specialist retention vs concentration) was run. Hypothesis only: it rewards candidates that solve rare tasks; not tested.
- The reported "tau subset vs full" over finalists is uninformative (3 finalists, many ties; median 0); use the offline table for ranking quality and the "pick changed" count for end-to-end.
- Champion cycle and rediscovery use behaviour equality on the search panel, which is coarse.
- Arms completing different numbers of generations at equal calls is part of the comparison, not a bug.

## Tests

`selection.rs` (specialist retention, seed reproducibility, epsilon/float ties, degenerate inputs, tournament weights, also for tournaments of 200 draws among 100 members and for sizes beyond `i32`; `tournament_size` is at most 1024); `arena.rs` (strategy determinism, config hash/serialisation, uniform unchanged); `study.rs` (observer does not change breeding, reproducibility, champion immutability and replay, newborn-vs-champion comparison, subset quality, re-evaluation; a subset that is the whole panel never changes the pick; each Cascade counts once, whatever the panel size; a broken grader on answers that only the study grades is an error, not a 0); `examples/selection_study.rs` (args, same generation 0 across arms, reproducible report, `runs.json` holds the down-sampling results, the report shows the re-evaluation calls). Re-checked on the final state on 2026-11-02: `cargo test` passed on every target (the library: 844 passed, 13 ignored), `cargo test` in `backend-ollama` passed, `cargo clippy --all-targets -- -D warnings` is clean, and the three new files pass rustfmt. Both reports were regenerated after the review fixes. Only the Cascade counts and the re-evaluation table changed (new pick rule, new calls column); the wall-seconds tables are still those of the release run.
