# 03 Grid lever: fix an endpoint, then pilot the step 2 Soul on grids

Status: needs-triage
Blocked by: none (it needs no code; it waits only for the owner's choice of this lever)
Spec: `../spec.md`, sections 5, 6 and 7

The short grid screen of spec section 6 showed that the step 2 Soul, as built, acts from
generation 1 on the grid Domain and that its arms part within a few generations. Grids have no
pre-registered endpoint, though, and every number of that screen has now been seen. This ticket
fixes an endpoint first and only then runs anything new.

## What to do

- Choose a grid endpoint without looking at the screen's outcomes again, and write down why.
  Candidates: (a) the mean of the best search solved count over generations 1 to G; (b) the
  first generation in which the best search solved count reaches a share of the search panel
  fixed in advance; (c) the best finalist's rate on the select panel at generation G. Name G,
  the share and the tie rule before any Run.
- Name a fresh list of grid pilot seeds and a fresh list of counted seeds. Seeds 913 to 920 have
  run on grids in the screen of spec section 6; they never count.
- Pilot arms S, V, K and N on the fresh pilot seeds (`--domain grid --search 33 --select 22
  --children 8 --max-calls 13000000`, `--soul s|v|k`, two Runs at a time with
  `RAYON_NUM_THREADS=2`). Measure the tie rate of the endpoint, the divergence generations, the
  minutes per Run and the calls.
- Draft a pre-registration in the form of step 1: claims S against V, S against K and S against
  N, the sequential rule with p₁ and N_max from the pilot, and a gate that the Soul acts (a
  flipped vote by generation 10 in most Runs).
- Add the grid analysis to `.scratch/seelen-modell/analysis/` (Python standard library, run with
  `python3 -I`): per seed and arm the first complex and flipped vote, the generations with a
  flip, and the divergence generation from arm S by the rule of `diagnose.py`, as in spec
  section 9. The scripts of the screen can serve as a start.

## Done when

- The endpoint and its reasons are written before the first new Run.
- The pilot's numbers and the draft are in `.scratch/seelen-modell/`, and the owner has approved
  or dropped the draft before any counted Run.
