# 04 Experience and imagination behind a switch (arm I)

Status: needs-triage
Blocked by: none
Spec: `../spec-imagination.md`, sections 3.1 to 3.3 and 3.5

## What to do

- Add a setting `imagination` to `BreedConfig`, at the top level, because it does not need the
  phase Soul. It holds the number of draws (4) and the learning rate (1/4). It is left out of the
  JSON when off, so configurations, hashes, logs and fixtures stay byte-identical. It has no CLI
  flag; it comes from the config file, as `variation` does.
- Refuse a Run with `imagination` when its gene pool cannot name families (`families_of` is
  None, grids today), and refuse a resume that changes it.
- Experience (spec 3.1): learned values per Worker and family in the Member, not in the genome.
  Store them as integers (thousandths) so that bytes and hashes cannot depend on float printing.
  A child Worker copies the values of its source Worker; a Worker without a source starts at 0.
- Imagination (spec 3.2): for SectionReplace, SectionAdd, SectionDrop and section crossover, draw
  the operator 4 times on the same parent. Draw 1 uses today's stream; draws 2 to 4 use streams
  derived with the tag `imagine/{g}/{i}/{j}`, so the main stream advances exactly as today. Take
  the draw with the highest prediction; a tie goes to the earliest draw.
- Learning (spec 3.3): after a newborn is evaluated, compute s on the current goal (all tasks
  while ticket 05 is off), the error e = s − p and the update of each gained or lost family,
  clamped to [−1, 1]. The child's Worker learns before its Born event. Each parent's Worker learns
  at commit, from its children in birth order, logged as a new event that `--verify` replays, as
  it replays `Souls`. Draw nothing from the generation's random stream.
- Logs (spec 3.5): the `mind` group of the state line, and the new fields of the Born event.
  Every new field is left out when off.

## Done when

- With `imagination` off, everything is byte-identical: the golden fixtures, the drafting
  digests, the prompt digests, and the release runs of seed 101 and of the README's seed 901.
- With `imagination` on and all values 0, generation 1 drafts exactly the children of a Run
  without it, and the generation's random stream ends at the same word.
- Unit tests: the prediction, the update, the clamp, the tie rule, the copy of values at birth
  (mutation, section crossover, stage crossover, new Worker), and the parent's learning in birth
  order.
- A Run with `imagination` over 60 generations is byte-equal at 1 and at 4 threads, passes
  `--verify`, and a resumed Run equals a straight one.
- `cargo test` and `cargo clippy --all-targets -- -D warnings` are clean.
