# 02 A thermodynamically consistent genome for reaction networks

Type: research
Status: needs-triage
Map: `../map.md`
Research: `../research.md`, sections 6 and 8.2 to 8.7

## Question

Which genome makes every reaction network it can describe obey mass action, local detailed
balance, conservation and the limits by construction, so that no mutation can leave the
physics?

## What to do

- Write a design note under Comments. Start from the sketch in `research.md` section 8.2.
- Fix the genes:
  - species, each with an integer standard free energy and a composition vector;
  - reversible reactions with at most two species per side, each with a nonnegative integer
    barrier excess over its higher side (`research.md` section 6.3);
  - optional chemostatted fuel and waste species, each with an integer chemical potential;
  - dual-rail input and output pairs;
  - initial concentrations of the other species, stored fuel included (`research.md`
    section 6.6);
  - the integration settings, accepted only under a step rule (`research.md` section 8.2).
- Show from Eq. 50 of source 39 that the derived rate constants obey local detailed balance for
  every gene value. Show that Wegscheider's conditions (source 40) and conservation then hold as
  identities in integer arithmetic.
- Decide how the irreversible reactions of compiled networks (sources 7 and 8) appear. The
  candidate form is a reversible reaction coupled to a chemostatted fuel, within the conditions
  of source 40, Thm. 3, and source 42, Sec. VI A.
- Decide the energy unit and the one reference concentration (source 42, Eq. 25).
- Decide the energy bill. Charge the dissipation of source 39, Eq. 79, or Eq. 55 summed over
  the steps, not the chemostats' work, which depends on the energy reference (`research.md`
  section 8.3). Both forms need `ln`. Eq. 55 needs it inside the loop, against the no-`ln`
  rule of ticket 01; Eq. 79 needs it only at the start and at the read-out. Record the choice
  in the Fog of `../map.md` (item 5).
- List every validation rule with its source. List the limits with proposed defaults: species,
  reactions, molecularity, the ranges of energies, barrier excesses and potentials, peak
  concentration, and steps.
- List the mutation operators and the crossover. For each, show that a child passes the
  validator or is redrawn.
- Map the genome onto the existing seam: `Arch` (`src/genome.rs:58-65`), `Genome::validate`
  (`src/genome.rs:229-323`), `Limits` (`src/config.rs:34-67`), the content id
  (`src/genome.rs:326-338`) and archive compatibility (`.scratch/cnn-basismodelle/spec.md`,
  Design).
- Propose the glossary terms for /domain-modeling.

## Done when

- The note gives the genes, the derived rate constants, every rule with its source, the limits,
  the operators and the mapping onto the seam.
- It gives a property-test plan of tests that can fail:
  - random valid genomes without chemostats relax to the equilibrium of source 39, Eq. 88, for
    the conserved totals of their initial state;
  - with one chemostat, which gives no emergent cycle (source 39, Example 6), the entropy
    production of source 39, Eq. 55, falls to zero;
  - at a nonequilibrium steady state, the entropy production of Eq. 55 equals the exchange term
    of source 39, Eq. 60, and the cycle term of Eq. 61, within a stated tolerance;
  - each emergent-cycle affinity computed from the fluxes (source 39, Eq. 27) equals the value
    the chemostat genes imply (source 39, Eqs. 28 and 49);
  - every mutant of a valid genome is valid or redrawn.
- A nonnegative entropy production (Eq. 55) is no test: each term `(a − b)·ln(a/b)` is
  nonnegative for all positive a and b.
- Open choices go to the Fog of `../map.md`, not into the design.

## Comments
