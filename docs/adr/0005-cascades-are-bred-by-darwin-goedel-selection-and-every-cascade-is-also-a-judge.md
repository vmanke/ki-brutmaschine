# Cascades are bred by Darwin-Gödel selection, and every Cascade is also a judge

Stage 2c breeds **Cascades** instead of single programs. A Cascade is one to four stages
run in order; each stage holds one to three **Workers** that answer in parallel; every
stage sees the original task plus the answers chosen by all earlier stages (the residual
path). A Worker is a model tag from an allow-list, a system prompt made of named sections,
and **Traits**: inheritable settings that shape how it thinks (thinking budget,
temperature, answer length, self-check), not what it thinks. Everything is data; nothing
in a Cascade is ever executed as code. This realises the design decided in the grilling
session of 2026-10-29 (questions Q1 to Q23).

**Every Cascade is also a judge.** Inside a stage, Workers that disagree judge each
other's answers blind, pair by pair, and the answer with the most wins goes on. Across the
archive, each new Cascade is compared with up to three unrelated Cascades on puzzles where
their answers differ, and a third Cascade, related to neither (no parent, child or
sibling), judges blind. Where exactly one answer is right, the judgment is scored, which
measures each Cascade's **judging skill** u = (correct + 1) / (judged + 2). Judgments
feed a weighted Bradley-Terry rating (Hunter 2004), each vote weighted ln(u / (1 - u))
when u > 0.5 and 0 otherwise, the optimal weight for independent voters of known accuracy
(Nitzan and Paroush 1982). Fitness is 0.8 x solved rate + 0.2 x judging skill - cost,
where cost charges completion tokens (thinking included) and a fixed per-call price per
model. The peer rating breaks ties and chooses crossover partners. It never replaces the
objective score: peers that only score each other can praise each other without solving
anything, the objective hacking the Darwin Gödel Machine reported when an agent removed
the markers its detector relied on (Zhang et al., ICLR 2026, appendix H). The judging that
matters most is inside the Cascade, where no answer key exists at test time.

**Parent selection is the Darwin Gödel Machine's** (appendix C.2): weight
sigmoid(lambda (fitness - alpha_0)) / (1 + children), lambda = 10, alpha_0 = 0.5, a perfect
score not eligible, drawn from the whole open-ended archive. Children come from random
operators (Trait, section, worker and stage edits), stage crossover, section crossover,
and, in fixed LLM rounds, from the **Mutator**: a language model (normally Claude) that
rewrites one parent or merges two. Mutator answers are recorded and replayed (ADR 0002).
When an answer is missing, the run pauses with the requests written to files and resumes
once someone, typically a Claude session, writes the answers, so the core gains no network
access (ADR 0001).

**Hard ceilings are code, not genes.** Thinking tokens (4096), answer tokens (2048),
depth (4) and width (3) are constants; a run configuration may tighten them, per stage
too (a late stage may be allowed only to check, not to think anew), but never loosen
them. Without a ceiling and a price, selection drives thinking up, because deeper
thinking solves more; then the control Traits are meant to give is lost.

**Two speeds, one gate.** The breeding loop runs sealed and without a human. Only its two
best finalists are promoted, as files under `candidates/cascades/`, through a draft pull
request that a human merges (ADR 0004, the config-and-prompt adapter). Self-improvement
reaches three layers differently: candidates change inside the loop; the breeder itself
(operators, selection, Mutator prompts) changes only through ordinary pull requests that
need the code owner; the evaluator (puzzles, answer keys, scoring, `brut-eval`, the cost
model) never changes from inside the loop. The Darwin Gödel Machine itself kept its
selection fixed; we let it change only through review.

**Models are genes, compute is matched.** The allow-list holds the simulated models used
in this container and the open models planned for the host (gpt-oss 20b and 120b, Qwen3);
real calls need the host adapter (ADR 0003). We run single-model searches, which isolate
what breeding adds, and mixed-model searches with per-call prices, which show whether
evolution mixes well. Any comparison gives the baseline the same models and the same
budget, so it measures breeding and not the model.

**What counts.** Here, puzzles come from our own ARC-style generator: a public search
panel and a select panel unseen during search. The simulated models are a positive
control whose skill depends on prompt and Traits; their numbers say nothing about real
models. The pre-registered endpoint for host runs is ARC-AGI-1 evaluation in the sealed
evaluator, pass@2 and exact grids, best Cascade against a compute-matched single Worker
with the same models, one-sided sign test at p < 0.05; it is not built yet.

Rejected: peer-only scoring (collusion, objective hacking); inheritable ceilings
(unbounded thinking); auto-merging promotions (ADR 0004); Claude as a Worker now (at up to
12 calls per puzzle, 400 puzzles, 20 Cascades and 30 generations, about 2.9 million calls,
which a hosted frontier model prices at five figures; deferred until local runs give
numbers).

Status: accepted. Later, as a pre-registered comparison: Traits modulated at run time by
a signal such as peer disagreement (the "feeling" variant, after Doya 2002), against
fixed Traits under equal seeds and budget.
