# 09 Sequential rule for step 1

Status: resolved
Type: task

Note, 2026-10-30: after the approval, the owner asked for the scientifically load-bearing
version, so U and the cap changed the same day. A claim is now proven at E >= 100 instead
of 20, the cap is 120 pairs instead of 60, and the counted seeds are the fixed list 1 to
100, then 111 to 130 (see Comments). The spec below is kept as first approved; the end of
the Comments says how to read it.

The owner rejected fixed counts and asked for a dynamic approach (2026-10-30): the
experiment should decide by itself how many seeds it needs. A design workflow made three
independent designs, checked each with an adversarial verifier and merged them into the
rule below. Nothing here is in force until the owner approves it as point 5 of ADR 0006,
in place of "9 of 10". The owner approved points 1-4 and 6 on 2026-10-30, and point 5 on
the same day (see Comments).

## Summary for the owner (German)

Ja, das geht: Das Experiment entscheidet selbst, wie viele Seeds es braucht. Nach jedem Seed-Paar prüft es, ob die Belege für ein Urteil reichen, und hört dann auf, spätestens nach 60 Paaren, notfalls ohne Urteil. Jeder Lauf zeigt selbst, in welcher Generation er besteht; wer bis Generation 1000 nicht besteht, zählt als nicht bestanden. Besteht die Zucht in Wahrheit höchstens in jedem zweiten Seed, meldet das Experiment höchstens mit einer Chance von 1 zu 20 einen Erfolg, gleich wann es aufhört. Das ist die übliche Grenze der anderen Tests im Projekt, während die alte Regel „9 von 10“ bei etwa 1 zu 100 lag. Stimmen die Vorversuche, gelingt der Hauptnachweis mit 82 bis 99 % Wahrscheinlichkeit. Wir rechnen mit 17 bis 29 Seed-Paaren, also etwa 4 bis 7 Stunden Rechenzeit, im ungünstigsten Fall etwa 15 Stunden. Sie müssen nur eines freigeben: diese Regel anstelle von Punkt 5 („9 von 10“), bevor der erste zählende Seed startet.

Unabhängige Gegenrechnung: `own_check.py` ist ohne den Code des Workflows geschrieben und
ergibt dieselben Zahlen. Alle Skripte in `.scratch/universum-erfinden/sequential-rule/`
reproduzieren ihre gespeicherten Ausgaben Zeichen für Zeichen.

## Design

SEQUENTIAL RULE FOR STEP 1. It replaces the fixed rule: ADR 0006 point 5 and PREREG sections 6 and 8. Everything below is fixed before seed 1 starts and never changes afterwards.

1. Runs. The counted seeds are s = 1, 2, 3, ... in this order, at most 60. Pilot seeds never count: 101-110, 201-210, 301-310 and 401-402, which include the old pilots 101 and 102.
   For each s, run arm A and arm R, one process at a time, each into a fresh directory:
     brut-breed --out runs/A-s --domain circuits --search 33 --select 22 --children 8 --generations 1000 --max-calls 13000000 --seed s
     Arm R: the same command with --out runs/R-s --selection uniform.
   Every other setting stays at its default.
   Every run completes all 1000 generations. There is no early stop.
   An interrupted run is deleted and rerun from scratch on the same seed.
   Run brut-breed --out <dir> --verify on both runs of the pair.
   The call cap cannot bind. By the bounds of PREREG section 6, a 1000-generation run needs at most 156 + 1000 · 12 720 + 3 256 = 12 723 412 calls.

2. Measured per run: the pass time T. T is the first generation g with 2 · |R_bred(<= g) \ R_seed| > |L \ R_seed|.
   - Reading T from rediscovery.json: take the targets with reached_by_bred = true and reached_by_seed = false. T is the k-th smallest first_generation among them, where k = floor(|L \ R_seed| / 2) + 1. So k = 5 when R_seed = {swap, cs}.
   - T = never when fewer than k such targets exist.
   - The run passes when T <= 1000. This equals D > 1/2 at generation 1000, because the archive keeps every member and D never falls.

3. Recorded per pair:
   - X_s = 1 when arm A passes, else 0.
   - Y_s = 'A first' when T_A < T_R (T_R may be never).
   - Y_s = 'R first' when T_R < T_A.
   - Y_s = 'tie' when T_A = T_R, including the case where neither run passes.

4. Rule across seeds. Two evidence values start at 1. They are updated strictly in seed order: pair s enters only after pairs 1 to s-1.
   - E1 is multiplied by 3/2 when X_s = 1 and by 1/2 when X_s = 0. After n pairs with k passes, E1 = 3^k / 2^n.
   - E2 is multiplied by 3/2 on 'A first' and by 1/2 on 'R first'. A tie leaves it unchanged. With w and l such pairs, E2 = 3^w / 2^(w+l).
   - The factors 3/2 and 1/2 are the likelihood ratio of a pass rate of 3/4 against 1/2.
   Primary claim:
   - Proven at the first pair with 3^k >= 20 · 2^n, that is E1 >= 20.
   - Failed at the first pair with 20 · 3^k <= 2^n, that is E1 <= 1/20.
   - Count form, proven: 8, 10, 11, 13, 15, 16, 18 or 20 passes with 0 to 7 failures.
   - Count form, failed: 5, 5, 6, 7, 7, 8, 8 or 9 failures with 0 to 7 passes.
   Secondary claim: the same two thresholds on E2.
   Every verdict is final at its first crossing.

5. How the two claims combine: a fixed sequence. The secondary is claimed only if the primary is proven. It does not matter which of the two crosses first.

6. Caps and stop. The ceiling is 1000 generations per run and 60 pairs per experiment. After each pair, the experiment stops at the first of:
   (i) the primary failed. The secondary is then 'not tested', and E2 is reported.
   (ii) the primary is proven and E2 has crossed either threshold.
   (iii) 60 pairs are done. A claim still open is 'undecided, no claim' and is reported with its E value.
   The analysis runs after each pair and prints either 'run seed n+1' or the verdict. Pairs run after the stop are listed but not counted.

7. Meaning of the verdicts.
   - Primary proven: more than half of all seeds pass within 1000 generations. This rejects the null 'pass rate <= 1/2'.
   - Secondary proven: among untied pairs, arm A passes first more often than arm R. This rejects the null 'P(A first | not tied) <= 1/2'.
   - Error bound: each claim is falsely made with probability <= 1/20, and some claim is falsely made with probability <= 1/20. This holds however the experiment stops (Ville's inequality).
   - Primary failed: the data speak against a pass rate of 3/4 or more. This verdict is wrong with probability <= 1/20 when the rate is >= 3/4. It does not prove a rate <= 1/2.

8. Conduct. The error bounds require all of the following:
   - The seed order is fixed.
   - No seed is skipped or replaced.
   - Updates go by seed, never by completion order.
   - The settings, the ceiling, the definitions of T, X and Y, and the thresholds do not change after seed 1. A change makes a new experiment on unused seeds.
   Elsewhere, runs may execute in parallel if the updates stay in seed order. In this container, one process at a time.

9. Reported.
   - Per seed: R_seed, D_A and D_R at generation 1000, T_A, T_R, X, Y, and E1 and E2 after the pair.
   - The stop pair, and both verdicts with E at the stop. The path maximum is never reported.
   - Descriptive, with no decision attached:
     - the median of D_A over the counted pairs;
     - the T-only variant: D_A and D_R at generation 1000 under the T-only rule, and the number of seeds with D_A > 1/2 under it;
     - the other descriptive results of PREREG section 8;
     - the calls and wall time of every run.

10. Unchanged.
   - The calibration tests and the gate.
   - The void conditions of PREREG section 9. Item 6 now names 1000 generations and 13 000 000 calls.
   - The thread check: seed 1 of both arms again at RAYON_NUM_THREADS 1 and 4. It is best run right after pair 1, so that a void shows early.

## Numbers

How the numbers were made.
- Scripts are in `.scratch/universum-erfinden/sequential-rule/`:
  - pilot.py: pass times, Kaplan-Meier and timings;
  - oc.py: exact DP;
  - extra.py: scenarios, assurance and time;
  - reference_rule.py.
- Outputs are the matching .txt files in the same folder.
- The primary DP runs in exact fractions; the joint DP in doubles.
- Engine check: the engine reproduces every d1/v1 value I compared. These are the type-I fractions at caps 50 and 60, table C, v1's strict-bar rows, all v1 joint rows at caps 50 and 60, and the assurance values 0.8100 and 0.8856 at cap 50.

FALSE SUCCESS (factors 3/2 and 1/2, thresholds 20 and 1/20, cap 60)
- Bound: <= 1/20 per claim, and <= 1/20 for any false claim, under any stopping.
- Exact worst case for the primary (p = 1/2): 5555272097573807/2^57 = 0.0385 (own, equals v1).
- Largest P(any false claim) over the joint null scenarios: 0.0385 (own, equals v1). The scenarios are: both nulls; primary null with arm R much worse; selection irrelevant at pass rates of 0.75 and 0.85.
- The old 9-of-10 rule: 11/1024 = 0.0107.
- P('failed' | p = 3/4) = 0.0347 (own).
- Why a fixed test cannot simply be re-checked: an exact 5 % binomial test run after every pair gives 0.0987 false success by 20 pairs and 0.1579 by 50 (d1, confirmed by v1).

PILOT RATES AT 1000 GENERATIONS
I recounted all 61 rediscovery.json files; the pass times equal the brief exactly. All 61 have R_seed = {swap, cs}. A-310 is incomplete; the completed copy in d2 passes at generation 336.
- p_A:
  - Kaplan-Meier 22/31 = 0.710 (95 % CI 0.417-0.941, from v2);
  - including A-310: 0.7625;
  - exact 9/11 = 0.818 (Clopper-Pearson 0.482-0.977).
- p_R: Kaplan-Meier 0.280 (0.113-0.595); exact 2/10.
- Pair outcomes (Kaplan-Meier, arms independent): A first 0.637, R first 0.154, tie 0.209. P(A first | untied) = 0.806 (own, equals v2).
- Pairs run to 1000 in both arms: 8 A first, 1 R first (seed 401), 1 tie (seed 301).

CHANCE OF SUCCESS, CAP 60
Columns: P(primary proven) | P(both claims) | E[pairs] | 90 % of experiments need at most this many pairs.
- Kaplan-Meier: 0.821 | 0.815 | 29.1 | 57
- Kaplan-Meier with A-310: 0.958 | 0.955 | 23.0 | 40
- The 10 long pairs as observed: 0.990 | 0.990 | 17.0 | 30
- p = 9/11, primary only: 0.995 with 15.5 pairs.
- Averaged over Beta(10, 3) (the 9/11 count with a uniform prior): 0.829.
Sensitivity, primary only:
- At p = 0.6 / 0.7 / 0.75 / 0.8: proven 0.273 / 0.781 / 0.938 / 0.990; undecided 0.150 / 0.096 / 0.027 / 0.003; E[pairs] 31.1 / 28.3 / 22.3 / 17.0.
- At p = 1/2: failed 0.937, undecided 0.025, 20.9 pairs.
- At p = 0.30, the 200-generation pilot rate (6/20, CI 0.119-0.543): failed with probability 1.000 after 9.1 pairs. This is why the ceiling cannot stay at 200.
- The old 9-of-10 rule would pass with probability 0.165 at p = 0.710 and 0.433 at p = 9/11.

PAIRS
- At least 8 pairs to prove, at least 5 to fail.
- At most 60 pairs, i.e. 120 counted runs, plus 4 thread-check runs.

WALL TIME
Own sums of timings.jsonl, at 2 threads:
- A 1000-generation run takes 6.01-8.57 min in arm A (mean 7.38, 11 runs) and 4.54-6.12 min in arm R (mean 5.38, 10 runs).
- A pair takes 10.57-14.31 min (mean 12.68, 10 pairs).
- The select phase and --verify add a few seconds per run (v2).
Totals:
- Expected total = E[pairs] · 12.68 min (Wald's identity): 3.6 h, 4.9 h or 6.1 h in the three scenarios above.
- Worst case: 60 · 14.31 min = 14.3 h, or 14.7 h if every pair had the slowest A run and the slowest R run.
- The thread check comes on top. It is not timed at 1 or 4 threads.

CALLS
- Bound: <= 12 723 412 per run, below the cap of 13 000 000.
- The pilots used at most 1 161 738 calls by generation 1000 (A-401). Own grep of its events.jsonl, equal to v2's corrected value.

WHY 1000 GENERATIONS AND 60 PAIRS
- 1000 is the longest horizon with pilot data in both arms: 11 A runs and 10 R runs reached it (v2).
- At generation 500, p_A is only 0.57-0.73.
- Generations 1000 to 2000 cost about 3 times as much as generations 0 to 1000 (v2: 3.17 for A, 3.07 for R).
- 60 is the smallest multiple of 10 with P(undecided) <= 5 % at p = 1/2 (0.025) and at p = 3/4 (0.027). Cap 50 gives 0.053 and 0.060.
- Cap 80 would raise P(proven) at the Kaplan-Meier rate to 0.864, for 4.8 h more in the worst case.

## Risks

1. The bar is weaker than in the old proposal, and the recommendation says so openly.
   - New: <= 1/20 (exact 0.0385). Old: 11/1024 = 0.0107.
   - The new bar matches the project's other pass tests.
   - On Shafer's (2021) betting scale (preprint wording), a factor of 20 lies between 'merits attention' (5) and 'conclusive' (100 or 1000).

2. Validity rests on conduct.
   - The seed order is fixed, no seed is skipped or replaced, updates go by seed, and nothing changes after seed 1.
   - A breach voids the bound.

3. Modelling assumption: seeds act as independent draws, because the random streams are derived from a hash of the seed.
   - The bound only needs each new seed's pass chance, given the earlier seeds, to be <= 1/2 under the null.
   - This cannot be checked.
   - No run with the counted settings exists on seeds 1-60. I checked 695 config.json files. Test fixtures used seeds 5, 31, 33 and 57, but with 9-12 search tasks.

4. 'Failed' is not proof of a rate <= 1/2.
   - At the Kaplan-Meier rate of 0.71 the rule ends 'failed' with probability 0.099 and 'undecided' with 0.080.
   - At p = 0.6 it is undecided with probability 0.150.

5. The claims change meaning.
   - 'Within 1000 generations' replaces 200.
   - Selection becomes 'passes first among untied pairs' instead of 'more targets at the end'.
   - Pairs where neither run passes say nothing about selection.

6. The planning numbers are uncertain.
   - p_A(1000) has a 95 % interval of 0.42-0.94 (Kaplan-Meier) or 0.48-0.98 (9/11).
   - The early batches differ: arm A passed by generation 200 in 6/20 against 7/11 (Fisher p = 0.128).
   - Success hinges on one event: a Worker naming Hadamard, Phase and Linear together. So pass times are widely spread (pilot passes at 349 and 810), and a pass after generation 1000 counts as a failure.

7. The power and time figures rest on three assumptions:
   - the two arms of a pair are independent;
   - the Kaplan-Meier curves are flat after their last pilot pass;
   - the timings of this container, at 2 threads and one process at a time, carry over.

8. Long runtime: up to about 15 h.
   - The container crashed twice under load, so run one process at a time.
   - An interrupted run is rerun from scratch; finished pairs are kept.
   - The thread check at 4 threads is the heaviest step.

9. The new analysis code could misreport. Mitigation: the tests above, plus an independent recount with reference_rule.py.

10. Misreading.
   - Report E at the stop, never the path maximum.
   - Pilot seeds must never be counted, because they chose the design.

## Open points

1. The single approval.
   - The owner signs this rule off as ADR 0006 point 5, replacing '9 of 10', before seed 1 starts. That sign-off also freezes the pre-registration.
   - The other five points of ADR 0006 are not touched by this change and still need the owner's sign-off, as before.

2. If the owner wants to keep the old bar of about 1 in 100:
   - Use the threshold 1024/11 instead of 20. Everything else stays the same, including cap 60. All values below are my own exact DP.
   - Exact false-success probability: 0.0079.
   - P(primary proven): 0.696 at p = 0.71, 0.919 at 0.76, 0.992 at 0.82.
   - Both claims at the Kaplan-Meier rate: 0.696 and 0.683, with 39.7 pairs (about 8.4 h). The worst case is unchanged.
   - Count form: proven at 12, 13, 15, 17, 19, 20, 22 or 24 passes with 0 to 7 failures.

3. An 'undecided' result after 60 pairs can be extended later. Ville's bound still holds, but the extension is a deviation and must be reported as one.

4. Not measured or not verified:
   - the runtime at 1 and 4 threads (the thread check) and on the host;
   - the texts of Ville 1939 and Wald 1945;
   - Westfall & Krishen beyond title, DOI and keywords.

5. Choices I made here, which the owner need not decide but may question:
   - Cap 60 instead of d1's 50: the smallest multiple of 10 with P(undecided) <= 5 % at p = 1/2 and p = 3/4.
   - Both arms always run all 1000 generations: the simplest option, with full integrity checks.
   - The T-only variant becomes descriptive. In the old rule it was secondary; a sequential decision on it is impossible because rediscovery.json has no T-only first generations.

6. Already resolved from the verifiers:
   - '63 runs' is now 61 complete runs plus the interrupted A-310.
   - The word 'pre-specified' is dropped.
   - The relaxed bar is now stated.
   - The implementation list is complete.
   - The calls by generation 1000 are now 1 161 738.
   - The integrity coverage gap of early stops: full runs, no early stop.
   - The sign test at a random N is replaced by E2.
   - The six citation fixes are applied.
   - d1's comparative claims about other designs are not used.

7. Optional later: an anytime-valid interval for the pass rate, a confidence sequence (Waudby-Smith & Ramdas 2024). A fixed-sample interval is not valid after a stop that depends on the data.

Update 2026-10-30: the owner approved points 1-4 and 6 of ADR 0006.

## Code changes after approval

This is the smallest set of changes. brut-breed needs no change: every run goes to generation 1000, and T is read from rediscovery.json.

1. examples/circuit_positive_control.rs (the frozen analysis)
   Test with: cargo test --example circuit_positive_control. Keep cargo clippy --all-targets -- -D warnings clean.

 a. Constants:
    - Delete COUNTED (line 51) and NEEDED (line 60).
    - GENERATIONS (54): 200 becomes 1000.
    - MAX_CALLS (57): 2_600_000 becomes 13_000_000.
    - MAX_PAIRS (63): 100 becomes 60. It is now the pre-registered cap, and 20 · 3^60 < 2^128.
    - Add EVIDENCE: u128 = 20.

 b. New pure functions:
    - pass_time(&Rediscovery) -> Option<usize>: the k-th smallest bred-only first_generation. Returns None when there are fewer, or when open = 0.
    - first(t_a, t_r) -> First {A, R, Tie}.
    - Evidence {up, n} with two integer tests in u128:
        proven: 3^up >= EVIDENCE * 2^n
        failed: EVIDENCE * 3^up <= 2^n
    - sequential(&[Row]) -> the per-pair (E1, E2), the stop pair, and two verdicts. Each verdict is one of Proven(n), Failed(n), Undecided, NotTested or Open.

 c. Data structures:
    - Row (290-304) gains t_a and t_r.
    - statistic() (321-355) drops beats and the sign-test p.
    - Delete binomial and upper_tail (372-381).

 d. Claim logic:
    - claim() (394-425): require seeds exactly 1..=n with n <= 60, instead of == COUNTED.
    - A counted pair up to the stop that fails the gate makes the experiment void.
    - claim_in() (427-449): only the constants change.
    - run() (795-849) is unchanged. The shared generations/cap check (838) and the stop-reason check (820) stay valid, because every run ends with 'all generations done'.

 e. Output:
    - verdict() (452-459), render_rule() (495-538) and render() (541-591) print a per-seed table: T_A, T_R, X, first, E1, E2.
    - Then two verdict lines, or 'OPEN: run seed n+1 next'.
    - Pairs after the stop are printed as 'not counted'.
    - The T-only variant is printed as a descriptive table with no verdict.
    - Update the module doc (1-35).

 f. Tests.
    Replace:
    - exact_fractions_and_the_sign_test (976): keep the fraction part;
    - ten_calibrated_seeds_that_rediscover_pass_both_rules (1021);
    - ties_count_against_a_in_the_sign_test (1050);
    - eight_seeds_above_half_fail_the_primary_rule (1064);
    - fewer_than_ten_seeds_or_other_seeds_make_no_claim (1104).
    Update only_the_pre_registered_length_and_cap_make_a_claim (1151) to 1000 generations and 13 000 000 calls.
    New tests:
    - evidence_boundaries_in_count_form: proven at 8, 10, 11, 13, 15, 16, 18 and 20 passes with 0 to 7 failures, and not one pass earlier. Failed at 5, 5, 6, 7, 7, 8, 8 and 9 failures with 0 to 7 passes.
    - pass_time_is_the_kth_bred_only_first_generation:
      - first generations [7, 10, 33, 33, 75, 97] with 9 open targets give 75;
      - only four of them give None;
      - a target the seeds solve never counts;
      - with |R_seed| = 3, k = 5 of 8.
    - first_to_pass_decides_the_pair: (7, 380) gives A; (97, never) gives A; (never, 720) gives R; (105, 48) gives R; (50, 50) and (never, never) give a tie.
    - the_pilot_order_fails_at_the_sixth_pair: X = 1, 0, 0, 0, 0, 0 fails at pair 6 with E1 = 3/64. The secondary is not tested.
    - ties_leave_e2_unchanged.
    - the_secondary_is_claimed_only_after_the_primary: E2 crosses 20 first and E1 later gives both claims. If E1 fails, there is no claim.
    - an_open_run_names_the_next_seed: 3 pairs print 'run seed 4'.
    - pairs_after_the_stop_are_not_counted.
    - sixty_pairs_that_never_cross_are_undecided: setting X = 1 exactly when E1 < 1 keeps E1 in [1/2, 3/2).
    - seeds_must_start_at_one_without_gaps: [1, 2, 4] and [2, 3] are withheld.
    - more_than_sixty_pairs_are_refused.

 g. Acceptance check before seed 1. It is manual and uncounted. Run the new analysis on pilot-1000, seeds 301-309. It must:
    - withhold the claim;
    - print T_A = never, 7, 75, 97, 349, 143, 33, 180, 810;
    - print T_R = never, 380, and never for the rest;
    - show E1 = 6561/512 (open) and E2 = 6561/256, which crossed 20 at pair 9.
    The independent script reference_rule.py gives exactly this (`--pilots <scratchpad>`). Before the counted result is reported, recount it with the same script:
      python3 -I .scratch/universum-erfinden/sequential-rule/reference_rule.py --runs runs
    That mode reads A-1, R-1, A-2, R-2, ... in seed order and counts a pair once both runs are finished. For every counted run it checks the seed, the arm, the pre-registered settings, the stop reason and the calibration gate. It refuses gaps in the seeds and more than 60 pairs. A failed gate up to the stop makes the experiment void.

2. docs/PREREG_STEP1_CIRCUITS.md
   - Lines 7-9: no counted seed has run.
   - Line 37: the Seeds row.
   - Lines 172-202:
     - the commands with 1000 and 13 000 000;
     - the bound 12 723 412;
     - the pilot maximum of 1 161 738 calls by generation 1000.
   - Lines 204-229, the procedure:
     - run pair by pair in seed order;
     - run --verify and the analysis after each pair;
     - run the thread check after pair 1.
   - Section 7: add T and the pair outcome.
   - Lines 260-274: the rule; the T-only variant becomes descriptive.
   - Lines 276-294: add T and E per pair.
   - Lines 306-308: void item 6.
   - Lines 310-333: the uncounted pilot batches and what they set.
   - Lines 335-345: the claims become 'within 1000 generations' and 'passes sooner'.

3. Other documents
   - docs/adr/0006-cascades-breed-on-domains-and-circuits-are-checked-exactly.md lines 161-170, 173-186 and 202-205: 'the 9-of-10 rule' becomes 'the sequential rule'.
   - README.md 536-537 and README.en.md 139-140: decision 5.
   - docs/SCHRITT1_SCHALTUNGEN.md 370-375, 385 and 392-395.
   - docs/STEP1_CIRCUITS.md 354-373.
   - Optional: a CONTEXT.md entry 'Pass time'.

NOT RECOMMENDED NOW (optional time savers)
- Run arm R only up to A's pass generation. No breeder change; 9.5-9.8 min per pair instead of about 13 (v2).
- A --stop-at-pass flag. This is a breeder change, specified by d2 with 8 tests; 4.3-5.0 min per pair.
Neither changes a verdict. Both shorten the answers that void condition 1 checks, and both need more changes to the analysis.

## Sources

Statuses come from the v3 verifier unless marked otherwise. Corrections by v3 are applied.

1. Ville, J. (1939). Etude critique de la notion de collectif. Gauthier-Villars (thesis; no DOI; Numdam THESE_1939__218__1_0).
   - Supports: historical source of the maximal inequality.
   - Ville proved it for discrete-time test martingales (Shafer et al. 2011, p. 3). The text was not read; cite it together with entries 2 and 3.

2. Shafer, Shen, Vereshchagin & Vovk (2011). Test martingales, Bayes factors and p-values. Statistical Science 26(1):84-101, doi:10.1214/10-STS347.
   - Supports: Ville's inequality (eq. 2), and reporting the evidence at the stop.
   - Correction: quoting the running maximum exaggerates the evidence, and the exaggeration is 'limited', not bounded.

3. Waudby-Smith & Ramdas (2024). Estimating means of bounded random variables by betting. JRSS-B 86(1):1-27, doi:10.1093/jrsssb/qkad009.
   - Supports: Ville's inequality for nonnegative supermartingales (eq. 9).
   - Supports: a test supermartingale for a whole class of nulls, which covers our null p <= 1/2.

4. Wald, A. (1945). Sequential tests of statistical hypotheses. Ann. Math. Statist. 16(2):117-186, doi:10.1214/aoms/1177731118.
   - Supports: the shape of the rule, a likelihood ratio with an upper and a lower boundary.
   - The paywalled text was not read. Not cited for our error rates.

5. Wald & Wolfowitz (1948). Optimum character of the sequential probability ratio test. Ann. Math. Statist. 19(3):326-339, doi:10.1214/aoms/1177730197.
   - Supports: among tests with no larger error rates, the SPRT needs the fewest observations on average at its two hypotheses, here p = 1/2 and 3/4.
   - Scope: the untruncated test only.

6. Grünwald, de Heide & Koolen (2024). Safe testing. JRSS-B 86(5):1091-1128, doi:10.1093/jrsssb/qkae011.
   - Supports: type-I control under optional continuation.
   - Supports (v3 correction): Prop. 3 / Example 4. For H0: θ <= δ- against θ >= δ+ in a monotone-likelihood-ratio family, the growth-optimal e-variable is p_δ+/p_δ-.
   - With δ- = 1/2 and δ+ = 3/4 that is exactly ×1.5 and ×0.5 (my arithmetic; v3 checked the proposition with δ+ = 0.8). δ+ is the user's choice.

7. Vovk & Wang (2021). E-values: calibration, combination and applications. Ann. Statist. 49(3):1736-1754, doi:10.1214/20-AOS2020.
   - Supports: sequential e-values multiply into a test supermartingale (§4).
   - Supports: p = min(1, 1/E) is the only admissible calibrator (Prop. 2.2), so E >= 20 means p <= 0.05.
   - Supports (added 2026-10-30): Jeffreys's rule of thumb for e-values (§7): e > 100 is 'decisive', e in (31.6, 100) 'very strong', e in (10, 31.6) 'strong'. Read in the arXiv version (arXiv:1912.06116v4).

8. Ramdas, Grünwald, Vovk & Shafer (2023). Game-theoretic statistics and safe anytime-valid inference. Statistical Science 38(4):576-601, doi:10.1214/23-STS894.
   - Supports: valid at all stopping times, with 'optional stopping or continuation for any reason'.

9. Armitage, McPherson & Rowe (1969). Repeated significance tests on accumulating data. JRSS-A 132(2):235-244, doi:10.2307/2343787.
   - Supports: re-testing a fixed-level rule as data come in inflates false positives, shown with exact binomial results.
   - Hence '9 of 10, re-checked after each seed' is not an option.

10. Westfall & Krishen (2001). Optimally weighted, fixed sequence and gatekeeper multiple testing procedures. J. Statist. Plann. Inference 99:25-40/41, doi:10.1016/S0378-3758(01)00077-5.
   - Supports: the fixed-sequence (gatekeeper) procedure for a primary and a secondary claim.
   - Verified in this session: title, authors and DOI via a search result, and the zbMATH keywords 'family-wise error rate' and 'closed testing'. Volume 99 is from v1. The text and the end page were not checked.
   - The bound used in the design rests on its own short proof, which v1 checked.

11. Dixon & Mood (1946). The statistical sign test. JASA 41(236):557-566, doi:10.1080/01621459.1946.10501898 (JSTOR 10.2307/2280577).
   - Supports: the secondary is a sign test on paired outcomes, here in sequential form.
   - Verified by me from the T&F and JSTOR records and the abstract.

12. Derrac, García, Molina & Herrera (2011). Swarm and Evolutionary Computation 1(1):3-18, doi:10.1016/j.swevo.2011.02.002.
   - Contrast: in a fixed-sample sign test, ties should be split evenly. Our e-process instead gives a tie factor 1, which is valid at any tie rate (v1).

13. Mathieu et al. (2024). AdaStop. TMLR 08/2024, OpenReview lXyZr9TLEU.
   - Supports: precedent for letting the data decide the number of seeds, with a maximum budget and a possible 'cannot decide'.
   - It names the SPRT as the fully sequential alternative.

14. Henderson et al. (2018). Deep reinforcement learning that matters. AAAI 32(1):3207-3214, doi:10.1609/aaai.v32i1.11694.
   - Supports: two 5-seed groups of the same algorithm differed significantly.
   - Quote: 'there can be no specific number of trials specified as a recommendation'.

15. Younes & Simmons (2006). Information and Computation 204(9):1368-1409, doi:10.1016/j.ic.2006.05.002.
   - Supports: precedent for the SPRT on yes/no outcomes of independent simulation runs.

16. Kaplan & Meier (1958). JASA 53(282):457-481, doi:10.1080/01621459.1958.10501452.
   - Supports: pooling pilot runs of different lengths, for planning only.

17. Hansen, Auger, Brockhoff & Tušar (2022). IEEE TEVC 26(6):1293-1305, doi:10.1109/TEVC.2022.3210897.
   - Supports: measuring the runtime to a target (our T) instead of the result at one budget.
   - Correction: it has only fixed-sample, Bonferroni-corrected significance marks.

18. Project files (read in this session):
   - The 5 % house standard: PLAN_GENERALISIERUNG.md:192 'p < 0.05 (einseitig)', POSITIVKONTROLLE.md:29 and ARCHITEKTUR.md:653.
   - The old size 11/1024: PREREG_STEP1_CIRCUITS.md, section 8, paragraph "Error bounds".

Added 2026-10-30 for the thresholds of the Comments; each was verified against its text in that review.

19. International Conference on Harmonisation (1998). ICH Harmonised Tripartite Guideline E9: Statistical Principles for Clinical Trials. Step 4 version, 5 February 1998. No DOI.
   - A regulatory guideline, not a peer-reviewed paper.
   - Supports reason (a), §5.5: "The approach of setting type I errors for one-sided tests at half the conventional type I error used in two-sided tests is preferable in regulatory settings." With the conventional 5 %, that is one-sided 2.5 %.

20. Kass & Raftery (1995). Bayes factors. Journal of the American Statistical Association 90(430):773-795, doi:10.1080/01621459.1995.10476572.
   - Supports reason (b), §3.2, p. 777: Jeffreys's categories call a Bayes factor from 10 to 100 'strong' and above 100 'decisive'.
   - Caveat: Kass & Raftery say of Jeffreys's categories that they "are not a calibration of the Bayes factor, but rather a rough descriptive statement about standards of evidence in scientific investigation" (p. 777). Their own slight modification calls 20 to 150 'strong' and above 150 'very strong'; on it, 20 and every E at a proof (100 to 150) fall in one class.

21. Cohen (1992). A power primer. Psychological Bulletin 112(1):155-159, doi:10.1037/0033-2909.112.1.155.
   - Supports reason (c), p. 156: power .80 as "a convention proposed for general use".
   - Supports reason (d), the weight 4 as the β:α ratio of 4:1, p. 156: "Taken with the conventional α = .05, power of .80 results in a β:α ratio of 4:1 (.20 to .05) of the two kinds of risks."
   - Not cited: the reading that a type I error is four times as serious as a type II error. It is Cohen's (1988, p. 56), which the review saw only in secondary quotations.

## Provenance

- Design workflow of 2026-10-30. d1: sequential test across seeds. d2: run length and wall
  time. d3: literature. v1 to v3: one adversarial verifier per design. Then one synthesis.
  7 agents, about 2.37 million tokens.
- Scripts and their outputs: `.scratch/universum-erfinden/sequential-rule/`. The raw pilot
  runs are not in the repo; `pilot.json` holds the values extracted from them.
- `own_check.py` is an independent check written from the rule text alone.

## Comments

2026-10-30, approval. The owner approved the rule of this issue as point 5 of ADR 0006, in
place of "9 of 10", with the evidence threshold 20 and the cap of 60 pairs as designed
above. The strict variant of open point 2, with the threshold 1024/11, was not chosen.
Points 1-4 and 6 were approved the same day, so all six points of ADR 0006 now hold. The
code changes after approval are agent work, so the status is now ready-for-agent.

2026-10-30, thresholds. Later the same day the owner wrote: "Na was denn jetzt,
ordentliche Schwellen oder nicht? ... Wir nehmen die wissenschaftlich tragende Version".
So the owner asked for the scientifically load-bearing version of point 5 and named no
numbers. Claude, the AI assistant that works on this repository, then ran a review. It
computed the exact operating characteristics of 16 designs: the proof thresholds 20, 40,
1024/11 and 100, each with a cap of 60, 80, 100 and 120 pairs. An independent recount and
a Monte Carlo simulation checked those numbers. The sources were checked, and only those
verified against their texts are cited, in the Sources section above (7, 18, 19, 20 and
21). The three independent judges of the review split: they voted for U = 40 with cap 100,
U = 1024/11 with cap 120 and U = 100 with cap 120, one each. An adversarial critic
recomputed the numbers and upheld U = 1024/11 with cap 120 (bound 11/1024 = 0.0107), on
the condition of a fixed list of counted seeds. Claude chose U = 100 with cap 120 and the
seed list below. This design is a statistical twin of U = 1024/11 with cap 120 (type I
0.0079 against 0.0086, Kaplan-Meier power 0.8605 against 0.8620), and 100 is the round 1 %
level; the other reasons are (a) to (d) below. The owner confirms point 5 in this version
with the squash merge of the pull request, which freezes the pre-registration. Until that
merge nothing is frozen and no counted seed runs. This version replaces U = 20, the cap of
60 and the open seed sequence 1, 2, 3, ... (now a fixed list of 120 seeds); everything
else of the rule stays:

- Proof threshold U = 100 for both claims. A claim is proven at the first pair with
  E >= 100, in integers 3^w >= 100 · 2^n.
- Lower threshold unchanged. A claim fails at the first pair with E <= 1/20, in integers
  20 · 3^w <= 2^n.
- Cap: at most 120 pairs.
- Counted seeds: the fixed list 1, 2, ..., 100, then 111, 112, ..., 130, 120 seeds in all.
  The pilot seeds 101 to 110 lie in that range and never count, so the list skips them.
  The other pilot seeds (201-210, 301-310, 401, 402) lie outside it. The seeds run in this
  order without gaps; pair i is the i-th seed of the list.
- The seeds of the list are fresh. A scan of the 729 events.jsonl files on this machine
  found no circuit Run with the counted settings on any seed of the list. The only circuit
  Runs on seeds of the list are test Runs on 5, 31, 33 and 57, with 9 to 12 search tasks.
- Unchanged: 1000 generations, the call cap of 13 000 000, the three stop rules, the fixed
  sequence (the secondary only after the primary is proven), the first crossing as final,
  E2 unchanged on ties, the thread check after pair 1, the void rules and the T-only
  variant.

Reasons:

- (a) The bound of 1 % is one-sided, as the claims are. For confirmatory claims like
  these, a one-sided level of 5 % is lenient: the regulatory guideline ICH E9 (§5.5,
  source 19) holds it preferable, in regulatory settings, to set the type I error of a
  one-sided test at half the conventional two-sided one, that is at 2.5 %. The bound of
  1 % is stricter than that by a factor of 2.5, and at least as strict as the bar of the
  first version ("9 of 10", 11/1024 = 0.0107). By the same halving it matches a two-sided
  test at 2 %. The other pass tests of the project keep their pre-registered one-sided
  5 % (source 18).
- (b) E >= 100 is 'decisive' evidence on Jeffreys's scale (Kass & Raftery 1995, source 20;
  Vovk & Wang 2021, §7, source 7), whereas 20 is only 'strong'. E1 and E2 are powers of 3
  over powers of 2 and never equal 100, so a proof means E above 100. On Kass & Raftery's
  own, slightly modified scale, 20 and every E at a proof (100 to 150) are both 'strong'
  (see the caveat of source 20).
- (c) The power is at least 0.80 at the most conservative pilot estimate, the
  Kaplan-Meier rate 22/31: 0.8605 (Cohen 1992, source 21). P(undecided) is at most 5 %
  there (0.0252), at p = 1/2 (0.0014) and at p = 3/4 (0.0031).
- (d) Among the 16 designs it minimises 4 alpha + beta at the Kaplan-Meier rate (0.1710).
  The weight 4 is the β:α ratio of 4:1 that Cohen (1992, source 21) derives from his
  conventions.

Caveat, stated honestly: the gain in power over U = 20 with cap 60 holds only above a pass
rate of about 0.67 (0.666 by bisection, designs.txt). Averaged over the uncertainty of the
pilot rate, the chance of a true proof is about the same: 0.8272 against 0.8287 under
Beta(10, 3), the 9/11 count with a uniform prior (designs.txt). What the change buys is a
false-success probability about five times smaller, at more compute.

Numbers. They come from the DP of oc.py, which runs in exact fractions for the primary
claim and in floats for the joint rule, and were checked twice. The evidence files are
`.scratch/universum-erfinden/sequential-rule/designs.py` with its output `designs.txt`,
`own_check.py` with `own_check.txt` for the primary claim alone, and `reference_rule.txt`
for the pilot illustration.

- Count form, proven: 12, 14, 15, 17, 19, 20, 22 or 24 passes with 0 to 7 failures, so at
  least 12 pairs to prove. Failed, unchanged: 5, 5, 6, 7, 7, 8, 8 or 9 failures with 0 to
  7 passes, so at least 5 pairs to fail.
- Worst-case type I error of the primary (p = 1/2, cap 120):
  5230783465047250584069050481905459/2^119 = 0.007870 (bound 1/100). The largest
  probability of any false claim over the joint null scenarios is 0.0079. Under both
  nulls, claim 2 is made with probability 0.0010.
- P(failed | p = 3/4) = 0.0367 (bound 1/20).
- Primary power at cap 120: 0.1828 at p = 0.6; 0.8183 at 0.7; 0.8605 at 22/31 (failed
  0.1144, undecided 0.0252); 0.9601 at 3/4; 0.9737 at 61/80; 0.9927 at 0.8; 0.9962 at
  9/11.
- Joint rule, Kaplan-Meier scenario: P(claim 1) 0.8605, P(both) 0.8574, E[N] 45.92 pairs,
  N90 82, expected 10.12 h (E[N] × 12.68 min plus 0.42 h for the thread check). The
  0.42 h assume that the four Runs of the thread check take as long as two pairs at 2
  threads; they have not been timed at 1 or 4 threads.
- Joint rule, the 10 long pilot pairs as observed: 0.9927 and 0.9927, E[N] 25.61, N90 42,
  5.83 h.
- Worst case, 120 pairs: 28.6 h at the slowest pilot pair (14.31 min), 29.4 h if every
  pair had the slowest run of each arm (14.69 min), plus the thread check.
- Comparison: U = 20 with cap 60 had type I 0.0385, Kaplan-Meier power 0.8211,
  P(undecided) 0.0801 at the Kaplan-Meier rate, and 4.0 to 6.6 h expected with the thread
  check. Its worst case is 14.3 h (14.7 h with the slowest run of each arm), plus the
  thread check. U = 1024/11 with cap 120 is a statistical twin (type I 0.0086,
  Kaplan-Meier power 0.8620); 100 is the round 1 % level and slightly stricter.
- Pilot illustration on the pairs 301 to 309: the pass times are unchanged. After 9 pairs
  E1 = 6561/512 = 12.81 and E2 = 6561/256 = 25.63, and both are open: neither reached 100
  nor 1/20. Under U = 20, E2 had crossed 20 at the ninth pair.
- Integers: 3^120 has 191 bits and does not fit u128, while 100 · 2^120 < 2^127 does.
  Every decision must stay exact and in integers, with no new crate.

How to read the sections above: the Summary, Design, Numbers, Risks, Open points and Code
changes describe the design as first approved. That design has U = 20, a cap of 60 pairs,
the seeds 1, 2, 3, ..., and one constant EVIDENCE = 20 for both tests in u128. Every
number and risk in those sections belongs to that design. The current design, its numbers
and its integer plan are those of this comment. The analysis compares 3^w with
PROOF · 2^n (PROOF = 100) and FAILURE · 3^w with 2^n (FAILURE = 20) as exact natural
numbers, since 3^120 needs 191 bits. The code changes are agent work, so the status stays
ready-for-agent.

2026-10-31, result. The code changes are in the frozen commit 15cb151, and the rule has
decided the counted experiment. Before seed 1, the acceptance check on the pilot pairs 301
to 309 gave the pass times and evidence values of the pre-registration. The rule then
stopped after pair 27, by stop rule 2: the primary claim is proven at pair 15 (E1 =
3^24/2^27 = 2104.2640 at the stop), and the secondary claim is proven at pair 27 (E2 =
3^20/2^25 = 103.9143 at the stop; 20 pairs A first, 5 R first, ties at pairs 10 and 19).
Both are results about the breeder as a positive control with the simulated designer, not
about real models. `reference_rule.py --runs runs` gave the same 27 rows, the same stop
pair and the same verdicts as the frozen analysis. One deviation is reported: a container
restart interrupted Run A-12, which was deleted and run again from scratch on the same
seed, as section 6 step 3 of the pre-registration requires. Report:
`docs/STEP1_CIRCUITS.md`, section 7 (German: `docs/SCHRITT1_SCHALTUNGEN.md`, Abschnitt 7);
data: `docs/step1-counted/`. The status is now done.
