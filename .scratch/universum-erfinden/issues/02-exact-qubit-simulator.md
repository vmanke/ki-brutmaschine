# 02 Exact qubit simulator (Clifford+T)

Status: ready-for-agent
Type: task

Q39, Q41. A pure Rust state-vector simulator without new dependencies (own complex type):
gates H, S, S-dagger, T, T-dagger, X, Z, CNOT; circuits as gate lists; full unitary for
small n; equivalence up to global phase with fidelity >= 1 - 1e-9 (Q38); T-count, CNOT
count and depth as costs; hard caps on qubits (12 here) and gates. Property tests against
textbook identities (HZH = X, S*S = Z, T^2 = S, CNOT conjugations).

## Comments
