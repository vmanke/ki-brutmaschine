# 03 Souletics at birth

Status: resolved
Blocked by: 02
Spec: `../spec.md`, sections 9 and 15

The ticket has two parts, built and committed in this order. Part A changes no behaviour, and its
tests pass before part B starts, so a failure in part B is not one of part A. When part A is done, a
line under Comments records its commit. Part A does not wait for Q71.

## What to do

### Part A: the operators name their sources and see no Souls

- `src/cascade/variation.rs`: every operator also returns the source map of spec 9.1 (one or two
  parent Workers per child Worker). `cross_worker` does not cross `phase_state` or `coupling`.
- The breeder hands every genetic operator its parents with `phase_state` and `coupling` removed
  (spec 9.1), as the Mutator requests do (spec 4.5). So the operators' test that a child differs
  from both parents (`acceptable`) compares genes, and every random draw stays as today.

### Part B: Souletics, κ at birth, novelty and call seeds

- Souletics in the breeder, after a draft is accepted and before its evaluation: seeds get new
  Souls and `kappa_seed`; a child Worker copies its source's birth Soul or mixes two, drifts, and
  now and then gets an entirely new Soul by Hoppe's rate θ / (θ + N), with N counted in Workers
  (spec 9.2 to 9.4, Q58, Q64, Q65). κ is copied or taken by a coin, moved and clamped, or kept
  with `kappa_fixed` [Q71 a].
- Own random streams with the fixed draw order of spec 9.7; switched-off parts ignore their draws.
- Novelty by `gene_id` (spec 9.6): the set of seen ids in `draft_children` and `fill_random` holds
  gene ids.
- The breeder's rule of spec 4.4 for Runs with Souls: every Worker carries exactly n phases, from
  the seeds, a resumed archive or Souletics. From this part on, a Run with Souls starts on both
  Domains.
- Model calls take their seeds from the gene id in a Run with Souls (spec 15): the tags
  `search/{gene_id}` and `select/{gene_id}`.
- Each member records the origin of every Worker's Soul (Hoppe's colour) and the `soul_sources` of
  spec 13.2.

## Done when

### Part A

- `drafting_without_souls_is_unchanged`: from the same seed and archive, `draft_children` gives the
  same children, ids and order as before this ticket, and the generation's stream is in the same
  state afterwards. It still holds after part B.
- `every_operator_names_its_sources`: one case per operator (each mutation operator, both
  crossovers, both Mutator kinds through the `partner` rule), checked against the Worker the child
  Worker was built from.
- `operators_see_no_souls`: for 1000 random parent pairs with random Souls, each operator, called
  as the breeder calls it, gives the same child genes, source map and random-stream state as for
  the same parents without Souls. This includes a SectionCrossover whose child has the genes of
  one parent, which must be refused and drawn again as today.

### Part B

- `souls_do_not_touch_the_genes`: with Souls on, generation 1 has the same gene ids, in the same
  order, as the same Run without Souls. It still holds after tickets 04 to 07 (the seeds hold no
  vote, and ticket 06 keeps phases out of peer requests).
- `copy_mix_and_drift`: a copied Soul equals its source's birth Soul plus steps in −d..=d; each
  mixed phase equals one source's phase plus a step; with `phase_drift` 0 a copy is exact.
- `new_souls_come_at_hoppes_rate`: over 100 000 draws with N = 4, 100 and 1000 Workers in the urn
  and θ = 1, the share of new Souls lies within 3 standard errors of θ / (θ + N); with θ = 0 there
  is none; N counts the archive's Workers and the child Workers drawn before.
- `a_parent_passes_on_its_birth_soul`: changing a parent's current Soul changes none of its
  children's birth Souls (Q65).
- `kappa_at_birth`: copy, coin, step and clamp at 0 and 1000; with `kappa_fixed` every Worker keeps
  `kappa_seed`; arm K's Workers all have κ 0.
- `the_arms_share_their_soul_draws`: for the same child, arms S, V and K draw the same phases, coins
  and steps.
- `novelty_is_by_genes`: a child with the genes of an archived member and another Soul is refused.
- `a_soul_has_n_phases`: a resumed archive with a Worker of too few or too many phases is refused.
- `calls_are_seeded_by_genes`: two members with the same genes and different Souls send the same
  requests with the same seeds; from ticket 06 on, apart from the `phases` field that ticket adds.
- `switches_give_their_degenerate_case`: with `copy` off every child Worker has a new Soul; with
  `mix` off a crossover child copies its first source.

## Comments

- 2026-10-31, part A. Built and tested before part B started. It is not committed: this
  build makes no commits, so no commit is recorded here. `variation.rs`: each operator has a
  `_sourced` variant that also returns the source of every child Worker (`Source`,
  `Sources`), and `partner_sources` gives the sources of a Mutator child by the partner
  rule. The operators work on the `genes()` of their parents, so a child carries no Soul and
  every draw stays as today. The breeder keeps the sources in each draft. Tests:
  `drafting_without_souls_is_unchanged` in `arena.rs` (digests recorded before the change);
  `every_operator_names_its_sources`, `operators_see_no_souls` and
  `cross_worker_never_crosses_a_soul` in `variation.rs`.
- 2026-10-31, part B. `Breeder::souletics` runs after drafting and before the evaluation;
  the maths is in `soul.rs` (`SoulDraws`, `souletics`, `uniform_step`, `turn`,
  `new_soul_chance`). Each draft draws from its own stream, `soul/seed/{i}` or
  `soul/{g}/{i}`, in the order of spec 9.7, and each draw takes a fixed part of the stream.
  A member records `soul_sources` (sources, new, origin) and `soul_now`, which starts as the
  birth Soul; the `born` event records `soul_sources`. Novelty, the call seeds
  (`search/{gene_id}`, `select/{gene_id}`) and the last tie-break of the ranking go by gene
  ids, which `Breeder::gene_ids` keeps. A Run with Souls starts on both Domains. A resumed
  member must hold a Soul of n phases per Worker and its records. Tests in `arena.rs`: the
  ten of part B above, and `a_souled_run_records_its_souls_and_resumes_like_a_straight_one`
  and `the_ranking_breaks_ties_by_genes`; in `soul.rs` six unit tests of the maths; in
  `tests/soul.rs` `a_run_with_souls_starts_on_both_domains` and
  `promote_refuses_a_run_with_souls`, now on a real Run. Without Souls the goldens of
  `tests/universe.rs` and `tests/mlp_golden.rs` and the pilot head `e93ed521…` hold.
