# 04 The complex Stage vote and the Array pattern

Status: resolved
Blocked by: 02, 03
Spec: `../spec.md`, section 7

## What to do

- `run_cascade_in` gets the current Souls of the Cascade's Workers, the task's strengths and the
  readout (all frozen for the generation, spec 8.4). Without Souls it is called as today.
- `stage_vote` keeps its blind pairwise judgments, prompts, seeds and calls. With Souls and `vote`
  on, it tallies amplitudes: v_j = Z(P_j, t), A(x) = Σ_j w_j(x) · v_j, Π(x) = |A(x)|²; the largest
  Π wins; a tie in Π, also when every Π is 0, goes to the answer with more verdicts, and a tie
  there to the lowest index (spec 7.2 to 7.4; Q46, Q54, Q61) [Q68 a].
- `CascadeRun` returns the authors of its final answer and the vote tally of spec 7.7; the member
  keeps the authors per search task and the tally of its birth votes.
- The peer round, `refresh`, judging skill and the Bradley-Terry ratings stay unchanged (spec 7.8).

## Done when

- `equal_phasors_give_the_count`: on 10 000 random win tables of 2 to 3 answers and 1 to 3 judges,
  the complex vote with equal non-zero phasors picks the count's winner.
- `a_unanimous_vote_is_never_lost`: on 10 000 random cases of 2 to 3 answers and 1 to 3 judges in
  which every judge gives the same verdicts, with random Souls, both readouts pick the count's
  winner. The cases include two equally loud judges of the real readout with opposite signs who
  both choose the answer of index 1.
- `without_souls_the_vote_is_todays`: with no `soul` field, and with `vote: false`, every vote of a
  short grid Run and a short circuit Run picks the same answers and makes the same calls as before
  this ticket (the goldens of `tests/universe.rs` hold).
- `interference_cases`: hand-built votes. Three judges with equal Characters at 0, 120 and 240
  degrees who all prefer x to both other answers, and split two to one on the other pair, lose x,
  whatever x's index; if they agree on that pair too, x wins. In a split vote the louder judge
  wins. Two equally loud judges of the real readout with opposite signs: when both choose x, x wins
  by the count; when they split, the index decides.
- `strengths_never_reach_a_request`: with `vote: false`, a short grid Run and a short circuit Run
  with Souls make byte-equal Worker requests and give byte-equal answers under `strengths: domain`
  and `strengths: equal`, so no path but the vote carries the strengths. The test stays in place
  for tickets 05 to 07.
- `authors_and_tallies_are_recorded`: with a scripted backend whose answers and verdicts are fixed,
  the authors per task and the counts of complex, flipped and tied votes equal hand counts.
- `a_souled_run_does_not_depend_on_threads`: a short circuit Run with Souls gives byte-equal files
  at 1 and 4 threads.

## Comments

- 2026-10-31, build. `soul.rs`: `array_vote` gives the Array pattern, the decision of spec
  7.4, whether the tie rule decided, and the coherence; `count_winner` is today's count.
  `arena.rs`: `run_cascade_with` takes `RunSouls` (current Souls, strengths, readout and
  `vote`), and `run_cascade_in` calls it without. `stage_vote` makes the same calls and
  counts the verdicts per judge; with phasors it tallies amplitudes. `CascadeRun` returns
  `authors` and a `VoteTally` (`complex`, `flipped`, `tied`, `coherence_sum`). Members and
  `born` events keep the authors per search task and the tally of the birth votes. A newborn
  votes with its birth Souls, the select phase with `soul_now`. The peer round, `refresh`,
  judging skill and the ratings are unchanged. `check_archive` compares the new `born`
  fields too. Tests: `equal_phasors_give_the_count`, `a_unanimous_vote_is_never_lost` and
  `interference_cases` in `soul.rs`; `without_souls_the_vote_is_todays` (digests recorded on
  the code of ticket 03, and longer Runs that hold votes compared with the same Runs without
  Souls), `strengths_never_reach_a_request`, `authors_and_tallies_are_recorded` and
  `the_select_phase_votes_with_the_current_souls` in `arena.rs`;
  `a_souled_run_does_not_depend_on_threads` in `tests/soul.rs`. Votes are rare in short
  Runs: of 16 circuit seeds run for 20 generations, four hold any, and none flipped one.
