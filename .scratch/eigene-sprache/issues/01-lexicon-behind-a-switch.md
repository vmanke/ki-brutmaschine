# 01 A Lexicon of coined words and a brevity case, behind switches (arms L and R)

Status: needs-triage
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
