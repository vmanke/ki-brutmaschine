#!/usr/bin/env bash
# The final data steps of the pilot: analysis, recount, pilot.json, operating
# characteristics and their recount, diagnosis, the check against the step 1 analysis.
set -u
ROOT=/path/to/pilot   # the local base folder; the only line that differs from the script that ran
WT=$ROOT/../wt/s2pilot
DEV=$ROOT/../s2pilot-dev
RULE=$WT/.scratch/seele-und-kappa/rule
DOCS=$WT/docs/pilot-step2
SEEDS="901 902 903 904 905 906 907 908 909 910 911 912"
cd "$ROOT" || exit 1
./analyse.sh ../target-step2/release/examples/soul_comparison "$DEV/vfinal" "$SEEDS" "$ROOT/batch1" \
  >"$DEV/analysis.txt" 2>&1 || { echo "analysis failed"; exit 1; }
(cd "$ROOT/batch1" && python3 -I "$RULE/reference_rule.py" --pilot 901-912 .) >"$DEV/recount.txt" 2>&1 \
  || { echo "recount failed"; exit 1; }
rerun=()
[ -s "$ROOT/logs/rerun.log" ] && rerun=(--rerun "$ROOT/rerun=$ROOT/logs/rerun.log")
python3 -I -B "$RULE/pilot.py" --batch "batch1=$ROOT/batch1" --hashes "$ROOT/logs/hashes.txt" \
  --analysis "batch1=$DEV/analysis.txt" "${rerun[@]}" --out "$DOCS/pilot.json" >"$RULE/pilot.txt" 2>&1 \
  || { echo "pilot.py failed"; cat "$RULE/pilot.txt"; exit 1; }
python3 -I -B "$RULE/oc.py" "$DOCS/pilot.json" >"$RULE/oc.txt" 2>&1 || { echo "oc failed"; exit 1; }
python3 -I -B "$RULE/own_check.py" "$DOCS/pilot.json" >"$RULE/own_check.txt" 2>&1 \
  || { echo "own_check failed"; exit 1; }
(cd "$ROOT/batch1" && python3 -I -B "$RULE/diagnose.py" 901-912 .) >"$RULE/diagnose.txt" 2>&1 \
  || { echo "diagnose failed"; exit 1; }
./cpc_check.sh ../target-step2/release/examples/circuit_positive_control "$DEV/cpc" "$SEEDS" "$ROOT/batch1" \
  >"$DEV/cpc.txt" 2>&1
cp "$DEV/analysis.txt" "$DOCS/analysis.txt"
cp "$DEV/recount.txt" "$DOCS/recount.txt"
sed "s|$ROOT/||g" "$ROOT/logs/driver.log" >"$DOCS/driver.log"
[ -s "$ROOT/logs/rerun.log" ] && cp "$ROOT/logs/rerun.log" "$DOCS/rerun.log"
echo "final steps done"
