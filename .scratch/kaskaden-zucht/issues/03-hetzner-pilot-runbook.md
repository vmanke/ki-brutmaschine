# 03 Hetzner pilot runbook

Status: ready-for-human
Type: task

Q7 to Q9 and Q16. Rent one GEX44 as a block (the setup fee is charged per order, so do
not order per run). Install Docker and Ollama, pull the two models, run the pilot with
`--max-seconds 7200 --max-calls 20000`. Results go to a `runs/` branch with a deploy
key that may write only there; the key lives in a `.env` on the host and never in git.
Verify current prices and setup fees on hetzner.com before ordering; third-party
sources conflict (see the session notes of 2026-10-29). A runbook script can be drafted
by an agent; renting and keys are the owner's.

## Comments
