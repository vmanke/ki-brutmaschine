#!/usr/bin/env bash
# Check that the step 1 analysis (examples/circuit_positive_control.rs) gives the arm-N Runs
# of the pilot the pass times that examples/soul_comparison.rs gives them, and that it
# refuses a Run with a soul field. The step 1 analysis needs an arm R beside arm A, so each
# N Run also stands in as a stand-in arm R: a view with its report and rediscovery and a
# config that adds "selection": "uniform". Only the column T_A is compared.
# Usage: cpc_check.sh CPC_BIN VIEWROOT SEEDS BATCHDIR
set -u
ROOT=/path/to/pilot   # the local base folder; the only line that differs from the script that ran
bin=$1
view=$2
seeds=$3
batch=$4
rm -rf "$view"
mkdir -p "$view"
a=()
r=()
for s in $seeds; do
  while read -r v; do a+=("$v"); done < <("$ROOT/view.sh" "$view/a" "$batch/N-$s")
  d=$view/r/N-$s
  mkdir -p "$d"
  ln -s "$batch/N-$s/report.json" "$d/report.json"
  ln -s "$batch/N-$s/rediscovery.json" "$d/rediscovery.json"
  python3 -I -c "
import json, sys
c = json.load(open(sys.argv[1]))
assert 'soul' not in c and 'selection' not in c
c['selection'] = 'uniform'
json.dump(c, open(sys.argv[2], 'w'))
" "$batch/N-$s/config.json" "$d/config.json"
  echo '{"snapshot":{"finished":true}}' >"$d/snapshot.json"
  r+=("$d")
done
"$bin" --a "${a[@]}" --r "${r[@]}"
rc=$?
pkill -f "view.sh $view/a " 2>/dev/null
# A Run with Souls is refused.
first=${seeds%% *}
while read -r v; do sv=$v; done < <("$ROOT/view.sh" "$view/s" "$batch/S-$first")
echo "--- the same analysis on S-$first as arm A:"
"$bin" --a "$sv" --r "$view/r/N-$first"
echo "exit=$?"
pkill -f "view.sh $view/s " 2>/dev/null
rm -rf "$view"
exit $rc
