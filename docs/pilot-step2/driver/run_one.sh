#!/usr/bin/env bash
# Run one pilot Run of step 2 and verify it.
# Usage: run_one.sh ARM SEED BATCHDIR [EXTRA...]
# ARM is S, V, K (--soul s|v|k), E (arm S with strengths: equal, through --config) or
# N (no Soul: arm A of step 1). The settings are those of the counted Runs of step 1.
set -u
ROOT=/path/to/pilot   # the local base folder; the only line that differs from the script that ran
B=$ROOT/bin/brut-breed
arm=$1
seed=$2
batch=$3
shift 3
dir=$batch/$arm-$seed
log=$ROOT/logs/driver.log
common=(--domain circuits --search 33 --select 22 --children 8 --generations 1000
  --max-calls 13000000 --seed "$seed")
case $arm in
  S) extra=(--soul s) ;;
  V) extra=(--soul v) ;;
  K) extra=(--soul k) ;;
  E) extra=(--config "$ROOT/config-E.json") ;;
  N) extra=() ;;
  *) echo "bad arm $arm" >&2; exit 2 ;;
esac
if [ -e "$dir" ]; then
  echo "$(date -u +%FT%TZ) SKIP $dir exists" >>"$log"
  exit 0
fi
echo "$(date -u +%FT%TZ) start $dir threads=${RAYON_NUM_THREADS:-unset} $*" >>"$log"
t0=$(date +%s.%N)
RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-2} nice -n 5 "$B" --out "$dir" "${common[@]}" "${extra[@]}" "$@" \
  >"$dir.out" 2>&1
rc=$?
t1=$(date +%s.%N)
wall=$(python3 -I -c "import sys; print(f'{float(sys.argv[2]) - float(sys.argv[1]):.3f}')" "$t0" "$t1")
echo "$(date -u +%FT%TZ) done $dir exit=$rc wall_seconds=$wall" >>"$log"
nice -n 5 "$B" --out "$dir" --verify >"$dir.verify" 2>&1
vrc=$?
echo "$(date -u +%FT%TZ) verify $dir exit=$vrc $(tail -n 1 "$dir.verify")" >>"$log"
echo "$wall" >"$dir.wall"
du -sb "$dir" >>"$ROOT/logs/sizes.txt"
if [ "$rc" = 0 ] && [ "$vrc" = 0 ] && [ "${KEEP_PLAIN:-0}" = 0 ]; then
  "$ROOT/after_run.sh" "$dir"
fi
