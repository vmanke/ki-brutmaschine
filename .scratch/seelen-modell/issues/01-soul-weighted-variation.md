# 01 Soul-weighted variation behind a switch (arm W)

Status: needs-triage
Blocked by: none
Spec: `../spec.md`, section 4.1

The owner decides first whether candidate W gets built. This ticket builds it, off by default,
so that ticket 02 can screen it. It changes `src/`, so it waits for the owner's approval.

## What to do

- `src/cascade/soul.rs`: `SoulConfig` gets a field `variation`, `off` or `prefer`, with a serde
  default of `off` and left out of the JSON when off. Every existing config, settings hash, log
  and test then stays as it is, and the pilot's `config.json` files still read. A resume refuses
  a change, as for every other `soul` field.
- The preference of section 4.1: `preference(soul, readout)` gives w_f = 1 + Re u(p_f) for each
  family, so 1 + cos(2π p_f / 65536) with complex phases and 0 or 2 with the sign readout.
- The gene pool gets a hook that names the families of a text, in family order. For circuits it
  uses the stems of `designer::FAMILY_STEMS`, as `designer::knowledge` does. A Run with
  `variation: prefer` on a Domain without the hook is refused.
- `src/cascade/variation.rs`: weighted forms of SectionReplace (which Worker, which section,
  which phrase), SectionAdd (which Worker, which phrase, times the protection of the section it
  would overwrite) and `cross_worker` (the coin of each section). The weights are those of
  section 4.1, each at least ε = 0.05, with one uniform draw per choice. With `variation` off
  the operators run exactly as today.
- `src/cascade/arena.rs`: the operators still get the parents' genes without Souls (spec 9.1),
  plus each parent Worker's preference from its birth Soul. The child's Souls come from
  Souletics as today.
- The `soul` group of the `state` line gets `variation_choices` and `variation_changed` per
  generation: the weighted choices, and those whose pick differs from the uniform pick with the
  same draw. Both are left out when `variation` is off.
- An amendment of ADR 0009: with `variation: prefer` the Soul shapes variation too, not only
  votes and the designer's replies.

## Done when

- `cargo test` passes and clippy is clean; the fixtures are unchanged
  (`git diff --exit-code tests/fixtures`), and the release test of pilot seed 101 still ends in
  its recorded head hash.
- Tests show: the mean preference over 2^20 fresh Souls is 1 within 0.01 per family; no weight
  falls below ε; a Worker with w = 2 on a family overwrites a text naming it less often than
  today's operator does, by the expected factor within 3 standard errors over 100 000 draws; with complex
  phases all at a quarter turn (w = 1) the weighted choices match today's choices in
  distribution, including the draw of `replacement_text` (0.7 from the same section name).
- A 60-generation circuit Run with `variation: prefer` passes `--verify`, and its five record
  files are byte-equal at 1 and at 4 threads.
- `CONTEXT.md` is not changed by this ticket: "preference" stays a word of the design note until
  the owner adopts the model.
