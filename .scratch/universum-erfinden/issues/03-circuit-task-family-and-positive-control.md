# 03 Circuit task family and positive control

Status: resolved
Type: task

Q37 (a), Q38, Q40. Circuits become a task family the arena can breed on: targets from a
small library whose best known Clifford+T costs are sourced from the literature (verify
every number before it is used); a solve prompt that shows the target as a truth table or
unitary and asks for a gate list; an exact checker; scoring by correctness first, then
cost. The arena gains a domain seam so grid puzzles and circuits share one breeding loop.
A simulated circuit designer (positive control) searches gate sequences within its
thinking budget, with prompt strategies that enable gate subsets and identities. Success
criterion: the breeding rediscovers the known optimal costs on most library targets
while the seeds do not.

## Comments

2026-10-31, result. The counted experiment of step 1 ran by its frozen pre-registration
(`docs/PREREG_STEP1_CIRCUITS.md`, frozen at commit 15cb151, the squash merge of PR #19).
The rule stopped after 27 seed pairs, by stop rule 2 of its section 8. The primary claim,
the success criterion of this ticket, is proven at pair 15; at the stop E1 = 3^24/2^27 =
2104.2640, and arm A passed in 24 of the 27 seeds. The secondary claim (selection) is
proven at pair 27, with E2 = 3^20/2^25 = 103.9143 at the stop. The independent recount by
`reference_rule.py` agrees, and no void condition holds. This is a result about the breeder
as a positive control with the simulated designer, not about real models. Report:
`docs/STEP1_CIRCUITS.md`, section 7 (German: `docs/SCHRITT1_SCHALTUNGEN.md`, Abschnitt 7);
data: `docs/step1-counted/`. The success criterion is met, so the status is now done.
