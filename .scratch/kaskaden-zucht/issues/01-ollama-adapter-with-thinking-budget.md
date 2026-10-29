# 01 Ollama adapter with the thinking budget

Status: ready-for-agent
Type: task

Build the separate `backend-ollama` crate of docs/STAGE2_BACKEND_CONTRACT.md on the host
(ADR 0003). New since contract-v1: `GenRequest::think_tokens`. Map it per model family:
gpt-oss takes a reasoning level (low, medium, high), Qwen3 takes thinking on or off. Set
`num_predict = max_tokens + think_tokens`. Record the mapping table in the contract.
Verify the model tags (`gpt-oss:20b`, `qwen3:8b`) and their memory needs on the GEX44
before the pilot. Then add a `--models <json>` option to `brut-breed` that registers the
adapter for real tags.

## Comments
