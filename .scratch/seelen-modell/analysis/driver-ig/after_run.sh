#!/usr/bin/env bash
# After a Run has finished and passed --verify: record the SHA-256 of its five record files,
# then compress events.jsonl and snapshot.json with xz -6, as the pilot of step 2 did.
# Usage: after_run.sh RUNDIR
set -u
ROOT=/path/to/ig-screen  # the local base folder; the only line that differs from the script that ran
dir=$1
log=$ROOT/logs/driver.log
for f in config.json events.jsonl snapshot.json report.json rediscovery.json; do
  if [ ! -f "$dir/$f" ]; then
    echo "$(date -u +%FT%TZ) FAILED hash: $dir/$f missing" >>"$log"
    exit 1
  fi
done
(cd "$dir" && sha256sum config.json events.jsonl snapshot.json report.json rediscovery.json \
  | sed "s|  |  $(basename "$dir")/|") >>"$ROOT/logs/hashes.txt"
if nice -n 5 xz -6 -T1 "$dir/events.jsonl" "$dir/snapshot.json"; then
  echo "$(date -u +%FT%TZ) compressed $(basename "$dir") $(du -sb "$dir" | cut -f1)" >>"$log"
else
  echo "$(date -u +%FT%TZ) FAILED xz $(basename "$dir")" >>"$log"
fi
