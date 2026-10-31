# 04 Gradients through the integrator

Type: prototype
Status: needs-triage
Blocked by: 01
Map: `../map.md`
Research: `../research.md`, sections 3.2, 7.4, 7.5 and 8.5

## Question

Can a reaction-network Candidate be trained within the limits of the MLP mode, by exact
gradients through the fixed-step integrator and with equal bits? What does that cost compared
with training the equivalent network and only simulating to check it?

## What to do

- Differentiate the discrete integrator of ticket 01, discretize-then-optimize (source 57,
  Sec. IV). Do it once in forward mode with dual numbers, and once in reverse mode over the
  stored trajectory.
- Train only initial concentrations of catalyst and context species; energies stay integer genes
  (`research.md` section 8.5). Weights as catalyst concentrations follow source 9, Table 1.
- Check every gradient against central finite differences. Check equal bits across runs and
  thread settings, as in ticket 01.
- Task: checkerboard (`src/task.rs:273-283`). The clean points lie in [−1, 1) and get Gaussian
  noise of std 0.02 (`src/task.rs:86-88`, `src/task.rs:454-462`). The construction needs
  nonnegative inputs (source 9, Eq. 11). Shift every input by +1, clamp it at 0, and report how
  many inputs were clamped.
- Network: a smoothed-ReLU network of source 9 with one hidden layer of a few units. Its weight
  reactions are trimolecular (source 9, Table 1), above the cap of two per side that
  `research.md` section 6.9 suggests; note whether a bimolecular variant is needed.
- Its production, weight and `qX → X` reactions are irreversible, and they create or destroy X,
  so they do not conserve composition (source 9, Table 1). Route A would therefore train a model
  that the genome of ticket 02 cannot hold.
- Find a reversible, composition-conserving variant that keeps the steady state of source 9,
  Prop. 4.9, or report that none was found.
- Route A: train the CRN by gradients through the integrator.
- Route B: train its equivalent network `φ(W·x + β)` (source 9, Eq. 4) directly. Then simulate
  it to check convergence and to measure its time.
- Count FLOPs per gradient for both modes, as functions of the number of trained values and of
  the steps. Compare them with the default `max_train_flops` of 2e10 (`src/config.rs:61`).

## Done when

- A report under Comments gives:
  - the gradient error against finite differences;
  - the bit-identity result;
  - FLOPs and memory per gradient for both modes;
  - the validation accuracy of both routes after the same FLOP budget;
  - the convergence time of route B's network once it is simulated;
  - whether a reversible, composition-conserving variant of source 9 was found, and how many
    inputs the shift had to clamp.
- It recommends route A, route B, or training outside chemistry, for a first reaction-network
  Candidate.

## Comments
