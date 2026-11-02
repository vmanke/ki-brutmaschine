#!/usr/bin/env bash
# vllm-start.sh: the frozen model server of docs/PREREG_BREEDING_VS_VOTE.md (D1, section 3).
#
# It starts vLLM 0.32.0 with Qwen2.5-32B-Instruct in bf16 and 32768 tokens of context, under
# the served name qwen2.5-32b (the tag vllm:qwen2.5-32b), on 127.0.0.1:8000 only, with
# VLLM_USE_FLASHINFER_SAMPLER=0. This is the server as measured on the H200 on 2026-10-31
# (docs/ERGEBNISSE_STUFE2C_ERSTE_LAEUFE.md, section 5; README, "vllm:" tags). Two flags are
# written out that were defaults then: --dtype bfloat16 (the weights are stored in bf16, so
# "auto" chose it) and the revision of the weights. HF_HUB_OFFLINE=1 makes vLLM read the
# weights from the local cache only, so nothing is fetched or updated while runs count.
#
# Usage, on the VM:
#   docs/prereg-breeding-vs-vote/vllm-start.sh
# The server runs in the foreground until Ctrl-C. Before it starts, the script prints the
# vLLM version and nvidia-smi; record both with the results (section 8, step 2).
#
# The pilot (section 9) starts the server before REVISION below is filled in:
#   PREREG_PILOT=1 docs/prereg-breeding-vs-vote/vllm-start.sh
# It then takes the revision the cache holds, prints it, and starts with it. Write that
# revision into REVISION before the freeze. A counted run never starts without it.
#
# The model is public, so no token is needed. Never write a token into this file.
set -euo pipefail

MODEL="Qwen/Qwen2.5-32B-Instruct"
SERVED="qwen2.5-32b"
VERSION="0.32.0"

# ======================================================================================
# TO BE FILLED IN FROM THE PILOT, BEFORE THE FREEZE: the commit hash (40 hex digits) of
# the weights in the VM's Hugging Face cache. The pilot records it from
#   ${HF_HOME:-~/.cache/huggingface}/hub/models--Qwen--Qwen2.5-32B-Instruct/refs/main
REVISION=""
# ======================================================================================

fail() {
    echo "vllm-start: $*" >&2
    exit 2
}

cache="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}/models--Qwen--Qwen2.5-32B-Instruct"

if [ -z "$REVISION" ]; then
    [ "${PREREG_PILOT:-}" = 1 ] || fail "REVISION is not filled in. Only the pilot starts without it (PREREG_PILOT=1)."
    REVISION="$(cat "$cache/refs/main" 2>/dev/null || true)"
    [ -n "$REVISION" ] || fail "no revision in $cache/refs/main; download the model into the cache first"
    echo "vllm-start: PILOT: the weights in the cache are revision $REVISION." >&2
    echo "vllm-start: PILOT: write it into REVISION in this file before the freeze." >&2
fi

[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || fail "REVISION must be a commit hash of 40 hex digits, got '$REVISION'"
[ -d "$cache/snapshots/$REVISION" ] || fail "the cache holds no snapshot $REVISION in $cache/snapshots"

found="$(vllm --version 2>/dev/null | tail -n 1 || true)"
[ "$found" = "$VERSION" ] || fail "vLLM $VERSION is frozen, but this is '${found:-no vllm}'"

echo "vllm-start: vLLM $found, $MODEL at revision $REVISION, served as $SERVED on 127.0.0.1:8000" >&2
nvidia-smi >&2 || fail "nvidia-smi failed"

export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1

exec vllm serve "$MODEL" \
    --revision "$REVISION" \
    --tokenizer-revision "$REVISION" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --served-model-name "$SERVED" \
    --host 127.0.0.1 \
    --port 8000
