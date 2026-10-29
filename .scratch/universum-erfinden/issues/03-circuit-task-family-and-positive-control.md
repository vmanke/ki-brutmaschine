# 03 Circuit task family and positive control

Status: ready-for-agent
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
