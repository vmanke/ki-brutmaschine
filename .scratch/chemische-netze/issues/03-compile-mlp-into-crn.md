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

- 2026-11-05, predictions before the prototype. This is our own derivation, not peer reviewed
  (`../research.md`, section 3.7; `../analysis/crn_checks.py`). We checked it on one module and on
  chains without weights only, so this ticket can falsify it for real networks.
  - Read-out at a fixed time: for the irreversible module, the output at `x = 0` should fall like
    `1/(k·T)`, not exponentially. With a fast first reaction, `y = x / (1 − C·exp(−k·T·x))` with
    `C = 1 − x/x+`.
  - Finite reverse rates: the module should follow `y = (x + sqrt(x² + 4h))/2` with `h = y−/K`.
    The largest error should lie at the kink, at about `sqrt(x+/K)`.
  - The question of this ticket: a kink error of 0.01 at unit concentration takes about 9.2 RT per
    reaction in one module. Accuracy within 0.01 of the network is another measure. Please report
    both against ΔG.
  - Exact match: source 8 reads its outputs at convergence. With Euler or RK4 and a fixed number
    of steps, every output is a polynomial of the initial concentrations (`../research.md`,
    section 3.7), so a match can only hold within the tolerance this ticket states.
  - Not tested by us: weights, biases, fan-out, a finite read-out time together with finite
    reverse rates, and whether cancellation reactions sharpen the kink by lowering the overlap of
    the rails, as `h = y−/K` suggests.
  - Line reference: `src/task.rs:262-283` is `src/task.rs:373-396` at commit 3d3b2fe.
