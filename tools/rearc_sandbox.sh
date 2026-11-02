#!/usr/bin/env bash
# Run tools/rearc_to_jsonl.py once in the sandbox recipe of docs/PLAN_GENERALISIERUNG.md,
# section 10, as docs/STAGE1_CONTRACT.md, section 5.13, asks for the re-arc conversion:
#
#   1. setpriv to user and group 65534 with no_new_privs and no supplementary groups, first,
#      so that the user namespace maps root to 65534, not to the real root;
#   2. unshare a user, network, PID, mount, IPC and UTS namespace; the network namespace
#      has no interface but loopback, so no connection leaves it;
#   3. a private tmpfs of 64 MB on /tmp (nosuid, nodev);
#   4. prlimit: 1 GiB of address space, 900 s of CPU, 64 processes, 64 MiB per written
#      file, 128 open files, no core dump.
#
# The converter gets the zip on standard input and its own code on file descriptor 3, and
# its output goes to standard output. So it opens no file by a path outside its tmpfs, and
# user 65534 needs no staging directory (it cannot read /tmp/claude-0, mode 0700 root).
# Root writes that output into a private directory beside the target and renames it into
# place at the end. Relative paths are made absolute first.
#
# usage: sudo tools/rearc_sandbox.sh <re_arc.zip> <output.jsonl> [converter options]
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: sudo $0 <re_arc.zip> <output.jsonl> [converter options]" >&2
    exit 2
fi

if [ "$(id -u)" != 0 ]; then
    echo "rearc_sandbox: needs root to switch to user 65534 (setpriv)" >&2
    exit 2
fi

zip=$1
out=$2
shift 2
script="$(cd "$(dirname "$0")" && pwd)/rearc_to_jsonl.py"

# The sandbox below starts in / (cd /), so make both paths absolute first: a relative path
# would otherwise resolve from /.
case $zip in /*) ;; *) zip="$PWD/$zip" ;; esac
case $out in /*) ;; *) out="$PWD/$out" ;; esac

for tool in setpriv unshare prlimit mount python3; do
    if ! command -v "$tool" >/dev/null; then
        echo "rearc_sandbox: $tool is missing" >&2
        exit 2
    fi
done

if [ ! -f "$zip" ]; then
    echo "rearc_sandbox: $zip is not a file" >&2
    exit 2
fi

if [ -e "$out" ] || [ -L "$out" ]; then
    echo "rearc_sandbox: $out exists; remove it first" >&2
    exit 2
fi

# Stage the output in a fresh private directory beside it (mktemp -d: random name, mode
# 0700, owned by root). Nobody else can place a symlink there, so root never writes through
# one, and the final rename stays on the same file system. A rename replaces a symlink at
# the target instead of following it.
stage=$(mktemp -d "$(dirname "$out")/.rearc_sandbox.XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
partial="$stage/output.jsonl"

cd /
setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs -- \
    unshare --user --map-root-user --net --pid --fork --mount --mount-proc --ipc --uts -- \
    sh -c 'mount -t tmpfs -o size=64m,nosuid,nodev tmpfs /tmp && exec prlimit \
        --as=1073741824 --cpu=900 --nproc=64 --fsize=67108864 --nofile=128 --core=0 -- \
        python3 -I /dev/fd/3 "$@"' rearc-sandbox "$@" \
    3<"$script" <"$zip" >"$partial"

mv -T -- "$partial" "$out"
