#!/usr/bin/env bash
# Run one Run of the screen of ticket 06 (.scratch/seelen-modell/issues/06-screen-arm-ig.md)
# and verify it; after_run.sh then records its hashes and compresses it.
# Usage: run_one.sh ARM SEED   (ARM: G, I or IG through --config, or N without one)
set -u
ROOT=/path/to/ig-screen  # the local base folder; the only line that differs from the script that ran
B=$ROOT/bin/brut-breed
arm=$1
seed=$2
dir=$ROOT/runs/$arm-$seed
log=$ROOT/logs/driver.log
common=(--domain circuits --search 33 --select 22 --children 8 --generations 1000
  --max-calls 13000000 --selection lexicase --seed "$seed")
case $arm in
  G | I | IG) extra=(--config "$ROOT/config-$arm.json") ;;
  N) extra=() ;;
  *) echo "$(date -u +%FT%TZ) FAILED bad arm $arm" >>"$log"; exit 2 ;;
esac
if [ -e "$dir" ]; then
  echo "$(date -u +%FT%TZ) SKIP $dir exists" >>"$log"
  exit 0
fi
echo "$(date -u +%FT%TZ) start $arm-$seed threads=2" >>"$log"
t0=$(date +%s.%N)
RAYON_NUM_THREADS=2 nice -n 5 "$B" --out "$dir" "${common[@]}" "${extra[@]}" >"$dir.out" 2>&1
rc=$?
t1=$(date +%s.%N)
wall=$(python3 -I -c "import sys; print(f'{float(sys.argv[2]) - float(sys.argv[1]):.3f}')" "$t0" "$t1")
echo "$wall" >"$dir.wall"
echo "$(date -u +%FT%TZ) done $arm-$seed exit=$rc wall_seconds=$wall" >>"$log"
nice -n 5 "$B" --out "$dir" --verify >"$dir.verify" 2>&1
vrc=$?
echo "$(date -u +%FT%TZ) verify $arm-$seed exit=$vrc $(tail -n 1 "$dir.verify")" >>"$log"
du -sb "$dir" >>"$ROOT/logs/sizes.txt"
if [ "$rc" = 0 ] && [ "$vrc" = 0 ]; then
  "$ROOT/driver/after_run.sh" "$dir"
else
  echo "$(date -u +%FT%TZ) FAILED $arm-$seed exit=$rc verify=$vrc" >>"$log"
fi
