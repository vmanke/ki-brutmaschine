#!/usr/bin/env bash
# Run examples/soul_comparison on pilot Runs through views of their compressed files.
# Usage: analyse.sh BIN VIEWROOT SEEDS BATCHDIR... [-- EXTRA ARGS]
#   SEEDS: a space-separated list in one argument, e.g. "901 902 903".
#   Each arm takes the runs <ARM>-<seed> found in the batch directories.
set -u
ROOT=/path/to/pilot   # the local base folder; the only line that differs from the script that ran
bin=$1
view=$2
seeds=$3
shift 3
batches=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  batches+=("$1")
  shift
done
[ "${1:-}" = "--" ] && shift
rm -rf "$view"
args=()
for arm in S V K E N; do
  flag="--$(echo "$arm" | tr 'SVKEN' 'svken')"
  dirs=()
  for s in $seeds; do
    for b in "${batches[@]}"; do
      [ -d "$b/$arm-$s" ] && dirs+=("$b/$arm-$s")
    done
  done
  if [ ${#dirs[@]} -gt 0 ]; then
    args+=("$flag")
    while read -r v; do args+=("$v"); done < <("$ROOT/view.sh" "$view" "${dirs[@]}")
  fi
done
"$bin" "${args[@]}" "$@"
rc=$?
# End every writer that still waits for a reader (the analysis stopped before it read the
# pipe). Their command lines are those of view.sh with this view's root.
pkill -f "view.sh $view " 2>/dev/null
rm -rf "$view"
exit $rc
