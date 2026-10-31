# 03 Compile a small MLP into a reaction network

Type: prototype
Status: needs-triage
Blocked by: 01
Map: `../map.md`
Research: `../research.md`, sections 3.1, 4.3, 4.5 and 6.4

## Question

Does a small ReLU MLP of the Brutmaschine, compiled into a dual-rail noncompetitive reaction
network, compute the same function under mass-action simulation? How fast does it converge?
How far does it drift once every reaction is reversible with a finite free-energy drop?

## What to do

- Train an MLP with `Activation::Relu` (`src/genome.rs:42-47`) on checkerboard or spirals,
  which have 2 inputs (`src/task.rs:262-283`). Keep it small enough to compile: one or two
  hidden layers.
- Quantize its weights to a few bits, since rational weights cost reactions in proportion to
  their bits (source 8, "Rational Weight ReLU Neural Networks"). The reference for every
  comparison is the quantized network.
- Compile it as source 8 describes: dual-rail inputs, ReLU modules (source 7, Sec. 3.1),
  doubling and halving chains for the weights, biases as initial concentrations, and
  cancellation reactions. The outputs are the logits; the argmax stays in the scorer
  (`research.md` section 4.5).
- Convert it before simulating. The compiled network is not a genome of `research.md`
  section 8.2: its fan-outs have one product per branch (source 7, Sec. 3.2, reactions (3)-(6)),
  above the cap of two per side of ticket 01. Split every fan-out into a binary tree of
  reactions `A → B + C`. Keep every division as halving steps `2A → B` (source 8, "Rational
  Weight ReLU Neural Networks"), not as one reaction with d reactants (source 6, Lemma 5.12).
- Simulate it with the integrator of ticket 01, all rate constants 1, as source 7 did (Sec. 4).
- Measure on the validation inputs, for at least two depths:
  - the largest output difference from the quantized network at the read-out time;
  - the time to 90 % of the output, and the time to stabilization;
  - the numbers of reactions and species;
  - the peak concentration.
- Then make every reaction reversible and couple it to a chemostatted fuel, with a free-energy
  drop ΔG per reaction. Use the form of ticket 02 if it is resolved. Otherwise each reaction
  gains a fuel reactant F and a waste product W, both chemostatted. This gives `M + X− → Y−`
  three reactants and `X+ → M + Y+` three products. State whether chemostatted F and W count
  toward the cap of two per side. If they do not, fold them into pseudo-first-order rate
  constants and say so. Measure output error and accuracy against ΔG.

## Done when

- A report under Comments gives the measurements above in a table, and the curve of error and
  accuracy against ΔG.
- It states whether the compiled network matches the quantized network to a stated tolerance.
  Source 8 found exact matches in all its cases ("Simulations").
- It names the smallest ΔG per reaction at which the accuracy stays within 0.01 of the
  network's, or says that no tested value does.

## Comments
