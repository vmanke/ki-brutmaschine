#!/usr/bin/env bash
# Run one pilot Run per arm again, one process at a time, and compare its five record files
# with the hashes recorded for the pilot Run. Afterwards the large files are removed.
# Usage: rerun.sh SEED
set -u
ROOT=/path/to/pilot   # the local base folder; the only line that differs from the script that ran
seed=$1
out=$ROOT/rerun
log=$ROOT/logs/rerun.log
mkdir -p "$out"
for arm in S V K E N; do
  KEEP_PLAIN=1 "$ROOT/run_one.sh" "$arm" "$seed" "$out"
  dir=$out/$arm-$seed
  ok=yes
  for f in config.json events.jsonl snapshot.json report.json rediscovery.json; do
    want=$(grep "  $arm-$seed/$f\$" "$ROOT/logs/hashes.txt" | head -n 1 | cut -d' ' -f1)
    have=$(sha256sum "$dir/$f" | cut -d' ' -f1)
    if [ "$want" = "$have" ]; then
      echo "$arm-$seed/$f equal $have" >>"$log"
    else
      echo "$arm-$seed/$f DIFFERS pilot $want rerun $have" >>"$log"
      ok=no
    fi
  done
  echo "$(date -u +%FT%TZ) $arm-$seed byte-equal: $ok; wall $(cat "$dir.wall") s" >>"$log"
  rm -f "$dir/events.jsonl" "$dir/snapshot.json"
done
