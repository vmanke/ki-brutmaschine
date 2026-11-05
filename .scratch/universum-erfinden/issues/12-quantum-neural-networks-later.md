# 12 Quantum neural networks as a later model kind

Status: needs-triage
Type: research

On 2026-10-31 the owner asked whether quantum neural networks (QNNs) exist. The owner left the
decision about them to Claude. Claude advised to park them, and this issue keeps the idea and
its sources.

## What is known

Sources checked on 2026-10-31.

- **QNNs exist as a research field.** Most current work means a parameterized quantum circuit.
  Its rotation angles are trained like the weights of a neural network.
- **Quantum convolutional neural networks exist.** Cong, Choi and Lukin 2019, Nature Physics
  15, 1273-1278, doi:10.1038/s41567-019-0648-8.
- **Deep feed-forward QNNs exist.** They are built from a quantum analogue of the perceptron.
  Beer et al. 2020, Nature Communications, doi:10.1038/s41467-020-14454-2.
- **Training can stall.** For random parameterized circuits the gradients vanish
  exponentially with the number of qubits ("barren plateaus"). McClean et al. 2018, Nature
  Communications 9, 4812, doi:10.1038/s41467-018-07090-4.
- **Classical models often win on small data.** A benchmark tested 12 quantum models on 160
  small datasets. Out-of-the-box classical models did better overall. Removing entanglement
  often did not hurt. Bowles, Ahmed and Schuld 2024, arXiv:2403.07059. This is a preprint, not
  peer reviewed.

## Why not now

- On a classical computer, a simulated QNN is a classical model. It cannot show a quantum
  advantage there.
- `src/cascade/qubit.rs` simulates Clifford+T circuits exactly, in integers. Trainable
  rotation angles would need a second simulator that works in floating point.
- The project has no access to quantum hardware yet (issue 08).

## When to look again

- When real quantum hardware is in reach (issue 08).
- Or when the research on chemical networks (branch `claude/chemische-netze`) yields a
  general frame for physical networks as a model kind, which QNNs could also fit.

## Comments
