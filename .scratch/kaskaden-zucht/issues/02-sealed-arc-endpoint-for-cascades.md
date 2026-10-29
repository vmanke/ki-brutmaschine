# 02 Sealed ARC endpoint for Cascades

Status: needs-triage
Type: task

Pre-registered endpoint (Q15, Q22): the best Cascade against a compute-matched single
Worker (same models, same call and token budget, majority of samples) on ARC-AGI-1
evaluation tasks, pass@2, exact grids, one-sided sign test at p < 0.05. It must run in
`brut-eval`, with the existing manifest and hash checks and its query budget. ARC
training tasks may serve as the search panel on the host. Open question: how a Cascade
file reaches the evaluator without the orchestrator reading evaluation data.

## Comments
