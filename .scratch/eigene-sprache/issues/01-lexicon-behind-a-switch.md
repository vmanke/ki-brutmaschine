# 01 A Lexicon of coined words and a brevity case, behind switches (arms L and R)

Status: resolved
Blocked by: none
Spec: `../spec.md`, sections 2 and 3

## What to do

- Add a setting `lexicon` to `BreedConfig`, at the top level:
  `{"every": 20, "best": 22, "max_words": 8, "meaning": "compression"}`, or `"random"` for arm R.
  It is left out of the JSON when off, so configurations, hashes, logs and fixtures stay
  byte-identical. It has no CLI flag; it comes from the config file, as `imagination` does.
- Refuse a Run with `lexicon` unless its gene pool can name families (`families_of` is Some,
  circuits today), and refuse a resume that changes it.
- Make the circuit gene pool able to grow during a Run: the fixed `CIRCUIT_PHRASES` plus the
  Run's current coined phrases, in coining order. The draw rule of `replacement_text` stays as it
  is; a coined phrase is one more text under its section name.
- The coining step of spec 3.2, after the commit of every generation g with g mod `every` = 0:
  the 22 best members in the archive's ranking, the sets of two or three families, the saving in
  words, no candidate for a set that already has a word, the tie rule, the cap, one word per
  step, and the retirement after two steps unused.
- A fixed readable term per family, so that a coined phrase is "Use {term 1}, then {term 2}."
  (and ", then {term 3}" for three). A test checks that `named_families` of each term names
  exactly its family, and of each phrase exactly its meaning, and that each phrase passes the
  genome's text rules and `no_gate_lists`.
- Arm R: the same step decides whether to coin and the size k of the set, from R's own
  archive. The meaning is then a set of k families drawn from the sets without a word, from the
  stream `lexicon/{g}` (`derive_seed`), so the generation's random stream advances exactly as
  today.
- A second top-level setting `brevity` (`true`, or left out): lexicase gets one more case. A
  member scores 1 on it if the words of all its Workers' system prompts (`approx_tokens`) are at
  most the median of the archive before the draft, else 0. The case uses the same filter as the
  task cases (`src/cascade/selection.rs:107-119`). Refuse it unless the selection is lexicase
  with an epsilon below 1. It works with or without `lexicon`.
- Logs: the events `coined` (word, meaning, section name, saving, generation) and `retired`, and
  the state-line group `lexicon` (words now, coined so far, retired so far, best members that hold
  a coined word). `--verify` recomputes the events from the archive, as it replays `learned`.
- Resume: the Lexicon comes from the checkpoint and the logged events.

## Done when

- With `lexicon` and `brevity` off, everything is byte-identical: the golden fixtures, the
  drafting digests, the prompt digests, and the release runs of seed 101 and of the README's
  seed 901.
- Unit tests of `brevity`: the case's column holds the median rule; when the case is drawn and
  some remaining member passes it, every remaining member above the median drops out; when no
  remaining member passes it, all of them stay.
- With `lexicon` on and no word coined yet, generations 1 to 19 draft exactly the children of a
  Run without it, and the generation's random stream ends at the same word.
- Unit tests: the saving of a set over hand-made Workers; a Worker that names a family only
  together with another does not count; the tie rule; the cap; retirement after two steps; a
  second coining step on an unchanged archive coins no second word for the same set; arm R coins
  at the same steps and in the same sizes as L's rule on its archive, and draws only from
  `lexicon/{g}`; the terms and phrases name exactly their families.
- A Run with `lexicon` over 60 generations, in arms L and R, is byte-equal at 1 and at 4 threads,
  passes `--verify`, and a Run resumed at generation 30 equals a straight one.
- `cargo test` and `cargo clippy --all-targets -- -D warnings` are clean.

## Comments

- 2026-11-06, built; committed on `claude/kind-dijkstra-1bfe2f` for the owner to review. Built
  on 5dfb627. The screen of ticket 02 was not run, by the owner's decision of the same day.
- The settings, at the top level of `BreedConfig`:
  - `"lexicon": {"every": 20, "best": 22, "max_words": 8, "meaning": "compression"}` for arm L,
    or `"meaning": "random"` for arm R (`src/cascade/lexicon.rs`, `LexiconConfig`). All four
    fields are required, no other is allowed, and `every`, `best` and `max_words` must be at
    least 1.
  - `"brevity": true`. `"brevity": false` counts as off and gives the same hash.
  - Both are left out of the JSON when off, and a resume refuses a change of either.
  - A Run with `lexicon` is refused unless the gene pool names the families of a text and has
    a term for each family (circuits today). A Run with `brevity` is refused unless the
    selection is lexicase and 1 − epsilon − 1e-12, lexicase's tie slack, is above 0. It works
    on grids too.
- What was built.
  - `src/cascade/lexicon.rs` (new): the setting, the phrase, the coining step, the grown pool,
    the holding count and the brevity column.
  - `src/cascade/arena.rs`: the settings and refusals, the coining step after the commit, the
    events `retired` and `coined`, the state-line group `lexicon`, the grown pool for drafting,
    the brevity column after the task or goal columns, and the replay on resume.
  - `src/cascade/circuit_genes.rs`: `LEXICON_TERMS`, one readable term per family.
  - `src/cascade/designer.rs`: `families_in_order`, the designer's order of first mention.
  - `src/cascade/observe.rs`: the group `lexicon` of the state line.
  - `src/bin/brut-breed.rs`: `check_lexicon`, used by `--verify` and `--resume`.
  - `src/cascade/variation.rs`: the phrase pool no longer needs `'static`; no logic changed.
  - `tests/lexicon.rs` (new): the tests of the binary.
- Decisions within the spec.
  - The coining step follows the commit of generation g, for g > 0 and g mod `every` = 0. It
    reads the first `best` members of the ranking after that commit: fitness, then peer rating,
    then gene id. A smaller archive gives all its members.
  - A Worker counts for a set M when, for each family of M, one of its sections names that
    family and no other. Of several such sections, the first in the Worker's order is the one
    a word would replace. Words are counted by `approx_tokens`.
  - A tie of sets goes to the lexicographically smaller sorted list of places, so {0, 1} comes
    before {0, 1, 2}, which comes before {0, 2}.
  - The phrase orders the families as the most counting Workers name them, by first mention in
    their system prompts. A tie goes to the lexicographically smallest order.
  - The section name is the name the replaced sections carry most often. A tie goes to the name
    with the most fixed phrases in the gene pool, then to the alphabetically first. Spec 3.2
    now says so; the probes below show why.
  - Retirement comes first in a step. A word that no best member holds at two steps in a row
    leaves the pool. A held word starts its count again, and a new word is first read at the
    next step, so it stays at least two steps.
  - A retired word's set is a candidate again, at the same step too. So the same phrase can be
    retired and coined again in one step, and this happens in practice.
  - Arm R lists the free sets of the winner's size in lexicographic order and draws one with
    `random_range` from a ChaCha8 stream seeded by `derive_seed(seed, "lexicon/{g}")`. Its
    phrase names the families in index order. It takes the section name of L's winning set, so
    the two arms differ only in the meaning. It logs its own saving, which is often below 0.
  - Brevity counts the words of each request's exact system prompt: the sections, the
    self-check text where that Trait is on, and the teaching section. The median is the lower
    one, at place (n − 1)/2 of the sorted list.
  - Events: `retired`, then `coined`, after the generation's other events and before `state`.
    They are logged only when something happens. `meaning` holds family names in index order.
    `best_holding` counts the best members that hold a coined word, in the pool or retired.
  - Resume: the snapshot has no new field. `resume_from` replays every coining step on the
    members born up to its generation, ranked by `refresh` over the comparisons up to it. The
    archive only grows, and a member's Cascade, solved count, graded value and usage never
    change, so this is the ranking the live commit had. The binary refuses a resume whose replay differs
    from the logged events up to the checkpoint, before any file changes.
- "Done when", checked.
  - Off: `tests/fixtures` is untouched, and the golden, drafting and prompt digests pass. The
    release runs of seed 101 and of the README's seed 901 reproduce
    (`souls_off_reproduces_the_step1_pilot`, `the_readme_run_of_arm_s_is_unchanged`).
  - With a Lexicon and no word yet, generations 1 to 20 draft the children of a Run without it,
    and the generation's stream ends at the same word.
  - Arms N (brevity only), R and L over 60 generations are byte-equal at 1 and at 4 threads and
    pass `--verify`. A Run resumed at generation 30 equals a straight one. A log with a changed
    word fails both `--verify` and `--resume`.
  - `cargo test`, after the fixes below: 1223 passed, 0 failed, 22 ignored, over 23 test
    binaries. `cargo clippy --all-targets -- -D warnings`: clean.
- Probes, not the screen: 100 generations with the flags of ticket 02, seeds 6411 to 6413,
  arms L and R, the same configs in both rounds. Only the tie rule of the section name differs.
  - With the tie going to the alphabetically first name, none of the 27 coined words joined
    "strategy": 16 joined "format", 7 "role" and 4 "residual". The gene pool has 2, 3 and 9
    fixed phrases under these names, against 25 under "strategy". So the same-name draw
    (p = 0.7) spread a word fast. At generation 100, 17, 22 and 14 of the 22 best members held
    a word in arm L, and 22, 22 and 0 in arm R. Words of random meaning spread as fast as those
    of arm L. They also replaced answer-format texts: 279 newborns of arm L on seed 6412, and
    425 of arm R on seed 6411, held a coined phrase as their "format" text.
  - With the tie going to the most fixed phrases, all 28 coined words joined "strategy". At
    generation 100, 0, 2 and 1 of the best members held a word in arm L, and 0, 0 and 0 in
    arm R. The most in any generation was 5.
  - So a word now spreads about as often as any strategy phrase, and within 100 generations it
    barely spreads. This is worth weighing before ticket 02 is run.
- 2026-11-06, two findings of the review of #62, both verified and fixed.
  - `--verify` checked the `coined` and `retired` events only for a finished run with a report.
    Every checkpoint holds the archive they replay on, so it now checks them in every run with
    a Lexicon. A test changes a word in a finished run without its report and in a run that
    waits for LLM answers, under a rebuilt hash chain and anchor; `--verify` finds both.
  - The refusal of `brevity` accepted an epsilon just below 1, such as 0.9999999999995, at
    which lexicase's tie slack makes the cut negative, so the case removed no one. The refusal
    now computes the cut as lexicase does, and spec 3.3 says so.
