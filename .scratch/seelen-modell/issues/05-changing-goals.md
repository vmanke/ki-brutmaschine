# 05 Changing goals behind a switch (arm G)

Status: needs-triage
Blocked by: none
Spec: `../spec-imagination.md`, sections 3.4 and 3.5

## What to do

- Add a setting `goals` to `BreedConfig`, at the top level, with the length of a goal in
  generations (20). It is left out of the JSON when off, so everything stays byte-identical. It
  has no CLI flag; it comes from the config file.
- Refuse a Run with `goals` unless its selection is lexicase and its Domain is circuits, and
  refuse a resume that changes it.
- Build the sequence of goals from the library (spec 3.4): G0 is all tasks of the search panel;
  G−f is the tasks of the targets whose `fits` do not name family f; leave out a G−f that equals
  G0 or is empty; order G0, G−f1, G0, G−f2, and so on over the families in index order, then
  repeat. Generations 1 to 20 have G0.
- Lexicase uses only the tasks of the current goal as its cases. Nothing else changes: every
  newborn is still evaluated on the whole search panel, and the select phase, the pass time and
  `rediscovery.json` stay as they are.
- When ticket 04 is on too, its real effect s uses the solved counts on the current goal.
- Log the current goal in the `mind` group of the state line, for example `all` or `-Linear`.

## Done when

- With `goals` off, everything is byte-identical, as in ticket 04.
- A unit test pins the sequence on today's circuit library: G0, G−Linear, G0, G−Phase, G0,
  G−Hadamard, G0, G−Compose, G0, G−Search, G0, G−Meet, G0, G−Simplify, with the targets of each
  goal as in spec 3.4.
- A unit test shows that lexicase with a goal draws only on the tasks of that goal.
- A Run with `goals` over 60 generations is byte-equal at 1 and at 4 threads, passes `--verify`,
  and a resumed Run equals a straight one. Its `rediscovery.json` is computed as today.
- `cargo test` and `cargo clippy --all-targets -- -D warnings` are clean.
