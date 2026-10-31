# 05 Mutations, base models in the population, docs

Status: ready-for-agent
Blocked by: 03, 04
Spec: `../spec.md`

## What to do

- Mutation operators of the spec: switch a stage's family where shapes allow it; insert
  or remove a stage; change width, kernel, stride, dilation, groups; toggle SE, residual
  and normalization; change downsampling and activation. Crossover of whole stages at
  matching resolutions. A child that fails the shape check is redrawn.
- The first generation holds the base models of all families next to random genomes.
- CLI: choose the model kinds of a run (MLP, CNN or both) and the new tasks.
- README in German and English, ADR 0008 (CNN families and their mutations in the MLP mode;
  ADR 0007 already records the library), new terms
  in CONTEXT.md.

## Done when

- Tests: every operator keeps the genome valid on 10 000 random draws; crossover of two
  base models of different families yields valid children; a short evolution run on the
  shapes task is reproducible byte for byte from its seed.
- A short demo run is described in the README with its numbers.

Cost: about 10 to 20 dollars.

## Comments
