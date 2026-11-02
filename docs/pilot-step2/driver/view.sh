#!/usr/bin/env bash
# Make a read-only view of pilot Runs whose events.jsonl and snapshot.json are compressed:
# small files are linked, and each compressed file becomes a named pipe that xz fills once.
# The analysis reads every file of a run once, from the start to the end, so a pipe works.
# Usage: view.sh VIEWROOT RUNDIR...   (prints the view directories, one per line)
set -u
root=$1
shift
mkdir -p "$root"
# Absolute paths, so that the links below also work for relative arguments (a fix after
# review; the pilot passed absolute paths).
root=$(cd "$root" && pwd) || exit 1
for dir in "$@"; do
  dir=$(cd "$dir" && pwd) || exit 1
  name=$(basename "$dir")
  v=$root/$name
  rm -rf "$v"
  mkdir -p "$v"
  for f in config.json report.json rediscovery.json timings.jsonl; do
    [ -e "$dir/$f" ] && ln -s "$dir/$f" "$v/$f"
  done
  for f in events.jsonl snapshot.json; do
    if [ -e "$dir/$f" ]; then
      ln -s "$dir/$f" "$v/$f"
    elif [ -e "$dir/$f.xz" ]; then
      mkfifo "$v/$f"
      # The writer must not hold this script's stdout while it waits for a reader.
      (exec </dev/null >/dev/null 2>&1; xz -dc "$dir/$f.xz" >"$v/$f" &)
    fi
  done
  echo "$v"
done
