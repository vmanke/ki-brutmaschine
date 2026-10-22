# Candidate pool

This directory holds candidate learners, one JSON file per candidate, named by its
content id (`<id>.json`, where the id is the first 16 hex of the SHA-256 over the
canonical genome JSON, the same scheme as `LearnerGenome::id`).

It is the mutation sandbox of the human-in-the-loop path (ADR 0004). A mutation pull
request (branch prefix `mutate/`) may change files **here only**. The `scope-fence` CI
job rejects any such pull request that touches a path outside `candidates/`, and it
revalidates every candidate (parse, structural check, cost cap, and that each file name
equals its content id).

Tools:

- `brut-propose --out <dir> --seed <u64> [--pool candidates] [--mode full|constants]`
  drafts two mutated candidates plus a provenance manifest under `<dir>`. With no pool it
  falls back to the seed programs.
- `scripts/propose-pr.sh <dir>` copies those two files in here, commits on a `mutate/`
  branch, and opens a draft pull request with the provenance in the body. It never merges.
- `brut-ingest --candidates candidates` validates the pool; it is the gate that admits
  merged candidates into a run.
