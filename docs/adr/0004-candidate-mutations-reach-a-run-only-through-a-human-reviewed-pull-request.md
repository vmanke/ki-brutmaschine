# Candidate mutations reach a run only through a human-reviewed pull request

Stage 2a wired a `ProgramProposer` seam and an in-process `LlmProposer`
(docs/STUFE2A_PILOT.md): proposals are accepted or rejected by the search itself,
with no human in the loop. For the host runs we add a second path. The proposer
drafts mutations as git branches and **draft** pull requests against the public
repository; a human reviews and merges before the mutated candidate enters a run.
Each pull request carries exactly two mutated candidates, reviewed and merged as one
unit. The in-process `LlmProposer` stays for fast, sealed experiments; the
pull-request path is the oversight path, and it is deliberately slow.

A candidate's "model code" has three representations, and we support all three behind
one `CandidateAdapter` interface (locate, statically validate, render a diff, record
provenance):

- **Genome (JSON).** The DSL program at `candidates/<id>.json`. Lowest risk: validity
  is the existing parser plus the cost model plus the FEC fingerprint, and the review
  surface is a small JSON diff. This adapter ships first and is the default.
- **Config and system prompt.** A Stage-2 open-model candidate: its configuration plus
  a pseudonymized system prompt from the CC0 seed corpus. No code runs during review.
  CI re-runs the pseudonymizer and a secret scan; a brand or secret residue fails the
  check.
- **Rust module.** Actual candidate code. Highest risk, shipped last and behind a
  feature flag: a mutation may touch only a confined candidate module,
  `#![forbid(unsafe_code)]` holds, the candidate crate is denied I/O, network, clock
  and process access, and it compiles and runs resource-capped in an isolated
  workspace. We enable it only once the genome and prompt paths have proven the loop.

The pull request may change candidate files only. CI rejects any mutation pull request
that touches the evaluator, the cost model (`dsl.rs` op costs), the hidden-grammar
code, the acceptance harness, or any secret. This scope fence is the direct guard
against the failure the Darwin-Gödel-Machine reported (accepted at ICLR 2026): a
self-modifying agent deleted its detector's logging to fake a perfect score. A mutant
that could edit the evaluator could pass by disabling the test instead of by learning,
which is exactly the loophole the primary endpoint is meant to close.

The proposer and the human reviewer are both blind to the hidden benchmark. Review
judges static validity, scope, safety, provenance and whether the change is a plausible
learning change, never a held-out score, because no held-out score exists at review
time. The hidden evaluation runs only after merge, in the sealed, network-free
evaluator (ADR 0001, ADR 0003). Letting either the model or the reviewer see held-out
families would turn review into a tuning channel and break the developer-aware
separation the generalization endpoint depends on (docs/PLAN_GENERALISIERUNG.md): the
reviewer is a "developer", so anything the reviewer sees is a prior the system was
handed, not evidence of generalization.

Nothing is auto-merged. Branch protection already requires a pull request on the public
repository; the merge is the explicit human approval, and that approval is the only
event that admits a mutation into a run. Each pull request records, in its body, the
parent candidate id(s), the mutation operator, the RNG seed and the adapter, so the
merged mutation replays bit for bit (ADR 0002). The pull request body carries no
credentials, paths or secrets.

We rejected two alternatives. Keeping only the in-process `LlmProposer` is faster and
needs no human, but gives no oversight at the mutation step and no auditable history;
we keep it for sealed experiments, not for the oversight path. Auto-merging on CI gates
alone removes the human, but CI can check validity and scope and cannot judge whether a
change is genuine learning or a cost-starving trick, which is the one judgment this path
exists to get from a person.

Status: proposed. Build order: genome adapter, then config-and-prompt adapter, then the
Rust adapter behind a flag once its sandbox is reviewed.
