# Stage 2 model-backend contract

Version: contract-v2, 2026-10-29 (v1: 2026-10-21; v2 adds `think_tokens`). This document is normative for the model backend
of Stage 2. It defines the interface the core already has (`src/backend.rs`) and the
separate adapter that serves a local open model through Ollama in a Docker sandbox.
The adapter runs only on a host the user controls; this managed container cannot run it
(see ADR 0003).

## 0. Scope and non-goals

Scope: a backend that turns a `GenRequest` into a `GenResponse` by calling a local
Ollama server. It is the only component in Stage 2 that opens a network socket, and it
reaches only `127.0.0.1` (the Ollama server on the same host), never the public internet.

Non-goals: no hosted API, no API key, no payment. There are no per-token fees; the only
cost is wall-clock time and the host's own compute. The three-level hierarchy (worker,
prompt mutator, code mutator) is specified elsewhere and is not part of this contract.

## 1. Interface (already in core, `src/backend.rs`)

```rust
pub struct GenRequest { system, prompt, max_tokens: u32, temperature: f32, seed: u64, think_tokens: u32 }
pub enum StopReason { Stop, Length, Refusal, Error }
pub struct GenResponse { text, prompt_tokens: u32, completion_tokens: u32, stop_reason, backend }
pub trait ModelBackend { fn name(&self) -> &str; fn generate(&self, &GenRequest) -> Result<GenResponse>; }
pub struct StubModel;   // deterministic, no network, for tests and local development here
```

The orchestrator depends only on this trait. It uses `StubModel` here and the Ollama
adapter on the host. It checks `GenResponse::is_usable()` (clean stop and non-empty) before
it parses an answer as a proposal.

## 2. The Ollama adapter (separate crate `backend-ollama`, host only)

MUST hold for the adapter:

- I1. **Separate crate.** It lives in its own Cargo package and depends on the core crate
  for the `ModelBackend` trait and its types. The core crate gains no network dependency
  and no new dependency of any kind (ADR 0001).
- I2. **HTTP client.** `ureq` 3.x, which pulls no async runtime and has a small dependency
  tree; the plan found it clean under `cargo audit`. No `reqwest`, no `tokio`.
- I3. **Localhost only.** The adapter reads the Ollama base URL from its own configuration
  (default `http://127.0.0.1:11434`) and refuses any host that is not a loopback address.
- I4. **No secret in the sandbox.** The adapter runs on the host, outside the Docker
  sandbox that runs mutated code. The sandbox never receives the Ollama URL or any
  credential, and child processes start with a cleared environment.
- I5. **Request shape.** `generate` POSTs to `/api/generate` with `model`, `prompt`,
  `system`, `stream: false`, `options.temperature`, and `options.seed`. For a model
  without a thinking mode, `options.num_predict = max_tokens` and `think_tokens` is
  ignored. For a thinking model, `options.num_predict = max_tokens + think_tokens` caps
  the sum only; the adapter enforces each part on its own. It reads the thinking text and
  the visible answer as separate fields, measures each (exact token counts where the
  server reports them, else a whitespace-word estimate that the adapter records as an
  estimate), and maps a visible answer above `max_tokens` or a thinking part above
  `think_tokens` to `StopReason::Length`, which the orchestrator never parses as an
  answer. `think_tokens` (contract-v2, a Worker Trait of ADR 0005; serde default 0, so
  contract-v1 records still replay) also selects the model's thinking control: 0 turns
  thinking off where the model allows it; for gpt-oss the budget picks the reasoning
  level (low, medium, high) by thresholds recorded in the adapter; for Qwen3 a budget
  above 0 turns thinking on. Level and on/off switches only choose the mode; the numeric
  caps come from `num_predict` and the two checks above. It reads `response`, and the token counts from `prompt_eval_count`
  and `eval_count`. A `done_reason` of `length` maps to `StopReason::Length`, otherwise
  `StopReason::Stop`; a transport or HTTP error maps to `StopReason::Error` and a
  `BrutError`.
- I6. **Timeouts and caps.** A per-request wall-clock timeout and a hard byte cap on the
  response. A model name not on an allow-list is refused before any call.
- I7. **Determinism, honestly stated.** With `temperature = 0` and a fixed `seed`, Ollama
  is close to deterministic but not guaranteed bit-identical across builds of the model
  runtime. Every request and its full answer are recorded in the run (ADR 0002), and a
  repeated run replays the recorded answers instead of calling the model again. Thus the
  run is reproducible even though the model is not.

## 3. Host setup (user, once)

Documented for the host, not run here:

- Install Docker (daemon running) and Ollama; pull the chosen open model (for example
  `llama3.1:8b` or `qwen2.5:7b`).
- Run the mutated-code sandbox as a Docker container with no network, a read-only root,
  a memory and pid limit, a dropped capability set and a seccomp profile; mount only a
  scratch work directory. Keep `runc` at 1.5.x or newer.
- The adapter and the orchestrator run on the host, outside that container.

## 4. Acceptance

- The core builds, tests and clippy stay clean with `backend` present (done).
- The adapter crate, once added on a host, exchanges a real `GenRequest`/`GenResponse`
  with a running Ollama server and records both in the archive.
- A request to any non-loopback host is refused with a `BrutError`.
- With the sandbox, a mutated program that tries to open a socket fails, because the
  container has no network.

## 5. What is built here vs on the host

- **Here (this container):** the `ModelBackend` trait, `StubModel`, their tests, and this
  contract. Nothing calls a network.
- **On the host:** the `backend-ollama` crate, the Docker sandbox, and the first real run.
