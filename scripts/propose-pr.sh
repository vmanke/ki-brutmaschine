#!/usr/bin/env bash
# Draft a human-review pull request from a brut-propose output directory (ADR 0004).
#
# Thin git + gh wrapper, kept outside the core crate so the crate keeps no process or
# network dependency (ADR 0001). It copies the two candidate files into candidates/,
# commits on a `mutate/` branch, pushes, and opens a DRAFT pull request with the
# provenance manifest in the body. It never merges; a human reviews and merges.
#
# Usage: scripts/propose-pr.sh <out-dir> [base-branch]
set -euo pipefail

out="${1:?usage: propose-pr.sh <out-dir> [base-branch]}"
base="${2:-main}"

[ -d "${out}/candidates" ] || { echo "no ${out}/candidates"; exit 1; }
[ -f "${out}/proposal.json" ] || { echo "no ${out}/proposal.json"; exit 1; }

seed="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["master_seed"])' "${out}/proposal.json")"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
branch="mutate/${seed}-${stamp}"

git fetch origin "${base}"
git checkout -B "${branch}" "origin/${base}"

mkdir -p candidates
cp "${out}"/candidates/*.json candidates/
git add candidates/
git commit -q -m "mutate: two candidate genomes for review (seed ${seed})"
git push -u origin "${branch}"

# Build the pull-request body: a short note plus the provenance manifest.
body="$(mktemp)"
{
  echo "Two mutated candidate genomes for human review (ADR 0004), seed ${seed}."
  echo
  echo "Scope: candidates/ only. Review judges validity, scope, safety and provenance,"
  echo "never a held-out score. No auto-merge."
  echo
  echo '```json'
  cat "${out}/proposal.json"
  echo '```'
} > "${body}"

# REST create (GraphQL may be blocked by the proxy); draft by default.
gh api "repos/{owner}/{repo}/pulls" \
  -f title="mutate: two candidate genomes for review (seed ${seed})" \
  -f head="${branch}" \
  -f base="${base}" \
  -F body=@"${body}" \
  -F draft=true

rm -f "${body}"
