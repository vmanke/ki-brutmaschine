#!/usr/bin/env python3
"""Convert re-arc's re_arc.zip into the JSON lines that src/arc_data.rs reads.

re-arc (Hodel 2024, arXiv:2404.07353, preprint; github.com/michaelhodel/re-arc at commit
e5b7f1d06362a76f9d3b8c25154ff1fafca897ce, MIT licence) ships 1000 generated examples for
each of the 400 ARC-AGI-1 training tasks in re_arc.zip. This script reads that zip as data
only: it imports and runs no code of re-arc. It uses the Python standard library only
(argparse, hashlib, json, re, sys, zipfile, zlib).

It writes one line per task, sorted by task id:

    {"id":"<8 hex>","pairs":[{"input":[[...]],"output":[[...]]},...]}

with the first --max-pairs examples of the task (default 60) that keep their grid size,
have rectangular grids of 1 to 30 cells per side with the colours 0 to 9, and have at most
--max-area cells (default 900, for the ARC grid tasks of the CNN models; 225 gives the cell
families of experiment E2). A task without such an example gets no line. The bytes depend
on the zip and the options only.

It refuses a zip whose SHA-256 is not --zip-sha256 (default: the pinned re_arc.zip), a
member with an absolute path, a backslash or a ".." part, an encrypted member, a task file
above 32 MiB, more than 800 MB of task files in all, and a task file that is not a JSON list.
It skips __MACOSX/ and every member but re_arc/tasks/<8 hex>.json.

Run it once, in the sandbox of tools/rearc_sandbox.sh, which hands it the zip on standard
input and takes the lines from standard output:

    sudo tools/rearc_sandbox.sh <re-arc clone>/re_arc.zip <data root>/re-arc/grids.jsonl

The summary goes to standard error, with the SHA-256 of the output for the manifest.
Exit codes: 0 ok, 1 refused input, 2 usage error.
"""

import argparse
import hashlib
import json
import re
import sys
import zipfile
import zlib

# SHA-256 of re_arc.zip at the pinned commit (docs/STAGE1_CONTRACT.md, section 5.13).
PINNED_ZIP_SHA256 = "4a7c309499f450eb47c2117fa06f3c97fe741cf6e7912579c0b5b8cc759e9ac4"

MEMBER_CAP = 32 << 20
TOTAL_CAP = 800_000_000
MAX_SIDE = 30
MOST_PAIRS = 60
TASK_MEMBER = re.compile(r"re_arc/tasks/([0-9a-f]{8})\.json")
HEX64 = re.compile(r"[0-9a-f]{64}")


class Refused(Exception):
    """The input breaks a rule; the message says which, without quoting the input."""


def reject_constant(_name):
    raise ValueError("NaN or Infinity is not a grid value")


def unique_keys(pairs):
    keys = [key for key, _ in pairs]

    if len(set(keys)) != len(keys):
        raise ValueError("a JSON object repeats a key")

    return dict(pairs)


def grid(value):
    """The rows of a valid ARC grid, or None."""
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_SIDE:
        return None

    width = None

    for row in value:
        if not isinstance(row, list) or not 1 <= len(row) <= MAX_SIDE:
            return None

        if width is None:
            width = len(row)
        elif len(row) != width:
            return None

        for cell in row:
            # type(...) is int refuses True and False, which are ints in Python.
            if type(cell) is not int or not 0 <= cell <= 9:
                return None

    return value


def why_not(example, max_area):
    """Why an example cannot be written, or None when it can."""
    if not isinstance(example, dict) or set(example) != {"input", "output"}:
        return "not a pair of input and output"

    rows_in = grid(example["input"])
    rows_out = grid(example["output"])

    if rows_in is None or rows_out is None:
        return "a grid breaks the ARC limits"

    if len(rows_in) != len(rows_out) or len(rows_in[0]) != len(rows_out[0]):
        return "changes the grid size"

    if len(rows_in) * len(rows_in[0]) > max_area:
        return "larger than --max-area"

    return None


def options(argv):
    parser = argparse.ArgumentParser(
        description="Convert re_arc.zip (standard input) into JSON lines (standard output)."
    )
    parser.add_argument("--max-pairs", type=int, default=MOST_PAIRS)
    parser.add_argument("--max-area", type=int, default=MAX_SIDE * MAX_SIDE)
    parser.add_argument("--zip-sha256", default=PINNED_ZIP_SHA256)
    args = parser.parse_args(argv)

    if not 1 <= args.max_pairs <= MOST_PAIRS:
        parser.error(f"--max-pairs must be 1 to {MOST_PAIRS}")

    if not 1 <= args.max_area <= MAX_SIDE * MAX_SIDE:
        parser.error(f"--max-area must be 1 to {MAX_SIDE * MAX_SIDE}")

    if not HEX64.fullmatch(args.zip_sha256):
        parser.error("--zip-sha256 must be 64 lower-case hex digits")

    return args


def read_tasks(source, args, skipped):
    """The kept examples of every task of the zip in `source`, by task id."""
    tasks = {}
    total = 0

    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            name = info.filename

            if name.startswith("/") or "\\" in name or ".." in name.split("/"):
                raise Refused("a member path is absolute or has a backslash or a '..' part")

            if name.startswith("__MACOSX/"):
                continue

            match = TASK_MEMBER.fullmatch(name)

            if not match:
                continue

            task_id = match.group(1)

            if info.flag_bits & 0x1:
                raise Refused(f"task file {task_id} is encrypted")

            if info.file_size > MEMBER_CAP:
                raise Refused(f"task file {task_id} is above {MEMBER_CAP} bytes")

            total += info.file_size

            if total > TOTAL_CAP:
                raise Refused(f"the task files hold more than {TOTAL_CAP} bytes")

            with archive.open(info) as member:
                raw = member.read(MEMBER_CAP + 1)

            if len(raw) > MEMBER_CAP:
                raise Refused(f"task file {task_id} is above {MEMBER_CAP} bytes")

            try:
                examples = json.loads(
                    raw.decode("utf-8"),
                    parse_constant=reject_constant,
                    object_pairs_hook=unique_keys,
                )
            except (UnicodeDecodeError, ValueError) as error:
                raise Refused(f"task file {task_id} is not valid JSON") from error

            if not isinstance(examples, list):
                raise Refused(f"task file {task_id} is not a JSON list")

            if task_id in tasks:
                raise Refused(f"task file {task_id} appears twice")

            kept = []

            for example in examples:
                reason = why_not(example, args.max_area)

                if reason is not None:
                    skipped[reason] = skipped.get(reason, 0) + 1
                elif len(kept) < args.max_pairs:
                    kept.append({"input": example["input"], "output": example["output"]})

            tasks[task_id] = kept

    return tasks


def main(argv):
    args = options(argv)
    source = open(0, "rb", closefd=False)

    if not source.seekable():
        raise Refused("standard input must be the zip file itself, not a pipe")

    digest = hashlib.sha256()

    while chunk := source.read(1 << 20):
        digest.update(chunk)

    if digest.hexdigest() != args.zip_sha256:
        raise Refused("the zip does not have the expected SHA-256")

    source.seek(0)

    skipped = {}
    tasks = read_tasks(source, args, skipped)
    out = sys.stdout.buffer
    out_digest = hashlib.sha256()
    lines = 0
    pairs = 0

    for task_id in sorted(tasks):
        kept = tasks[task_id]

        if not kept:
            continue

        line = json.dumps({"id": task_id, "pairs": kept}, separators=(",", ":")).encode(
            "ascii"
        ) + b"\n"
        out.write(line)
        out_digest.update(line)
        lines += 1
        pairs += len(kept)

    out.flush()

    print(f"rearc_to_jsonl: read {len(tasks)} task files", file=sys.stderr)
    print(f"rearc_to_jsonl: wrote {lines} tasks with {pairs} pairs", file=sys.stderr)

    for reason in sorted(skipped):
        print(f"rearc_to_jsonl: skipped {skipped[reason]} examples: {reason}", file=sys.stderr)

    print(f"rearc_to_jsonl: output sha256 {out_digest.hexdigest()}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Refused as error:
        print(f"rearc_to_jsonl: refused: {error}", file=sys.stderr)
        sys.exit(1)
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError):
        print("rearc_to_jsonl: refused: the zip is broken or cannot be read", file=sys.stderr)
        sys.exit(1)
