# 08 Nebius host runbook

Status: needs-triage (waits for the owner's approval of the design). After approval the
scripts and the runbook become ready-for-agent. The console steps, the deploy key and its
ruleset, the GitHub logout of the VM and every stop of the VM become ready-for-human: they
stay with the owner.
Type: task
Blocked by: none. Real Cascade runs use `brut-breed-ollama` of `backend-ollama` (PR #33),
which exists. Runs with gpt-oss:20b or qwen3:8b also need the thinking caps of spec 10.3.
Spec: ../spec.md (sections 8, 9.2; decisions Q110, Q111, Q114)
Pull request: H1, a small one of its own

This ticket replaces kaskaden-zucht issue 03 (a Hetzner GEX44) for the owner's Nebius VM.
Its pull request adds a comment to issue 03 that points here.

## What to do

- `tools/host-llm-setup.sh`, run after `tools/vm-setup.sh`, which installs neither Docker
  nor Ollama (`tools/vm-setup.sh:7-17`):
  - install Ollama natively, at a fixed version (the installer's `OLLAMA_VERSION`), as
    the README section "Echte Modelle über Ollama (`backend-ollama`)" assumes (Q111). It
    uses the GPU through the VM's NVIDIA driver;
  - keep Ollama's default address 127.0.0.1:11434, which `backend-ollama` expects
    (`backend-ollama/src/lib.rs:36-37`);
  - assert that nothing listens on port 11434 on any other address;
  - keep the SSH-only firewall (`tools/vm-setup.sh:79-81`);
  - pull the models of the run and verify their tags. `backend-ollama` serves only the
    models of its list (`backend-ollama/src/lib.rs:33-34`); gpt-oss:20b and qwen3:8b
    follow with its thinking caps (spec 10.3);
  - build `brut-breed-ollama` and run its live test, as that README section says
    (`cd backend-ollama && cargo build --release`, then
    `OLLAMA_LIVE=1 cargo test --release -- --ignored live`);
  - write the Ollama version, the model digests and the `nvidia-smi` output into host
    notes outside git;
  - offer `--dry-run`, which prints the steps and changes nothing.
- Docker is optional: for Ollama in a container, or later for the sandbox of mutated code
  (contract I4). The setup script does not install it. If the owner installs it, run
  `tools/vm-setup.sh` again. It then makes 127.0.0.1 Docker's default published address
  (`tools/vm-setup.sh:82-103`). The default does not cover a port published with an
  explicit address, nor the networks of docker compose (`tools/vm-setup.sh:83-86`). An
  Ollama container is pinned by its image digest, and the host notes record it. The port
  check above applies unchanged.
- `tools/brut-run.service` (systemd): it runs the run, and on stop it writes the run's
  STOP file and waits up to 50 s. Resuming after a restart stays a manual command unless
  the owner enables it. The run lives in this unit, not in the tmux session of
  `brut-remote-control`: when that unit stops or restarts, systemd also ends every process
  started inside it (`tools/vm-setup.sh:185-191`).
- `tools/gpu-sample.sh`: sample `nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu
  --format=csv` and the CPU load every 10 s into a file outside the run directory.
- Credentials (the owner's steps). `tools/vm-setup.sh` installs `gh` from GitHub's own
  package source, logs the owner's GitHub account into the VM and gives git its token
  (`tools/vm-setup.sh:64-77, 134-142`). An agent on the VM could then push any branch,
  open pull requests and merge them as the owner. So:
  - after setup, the owner runs `gh auth logout` on the VM and removes the token from
    git's credential helper;
  - every later run of `tools/vm-setup.sh`, such as the one after a Docker install, asks
    for that login again (`tools/vm-setup.sh:137-138`), so the owner logs out again after
    it;
  - results go to `runs/` only, pushed with the deploy key;
  - a repository ruleset blocks that key on every branch except `runs/`, since a deploy
    key with write access reaches every branch otherwise;
  - the key lives in `~/.ssh` with mode 0600, outside the repository.
- Before each run, check free space on the run disk against the worst case: `max_calls`
  times the largest Call record of the pilot (M8). A full disk stops a run with a fault
  (spec 4.1).
- `docs/NEBIUS_HOST.md` in German (`docs/SCHREIBSTIL.md`) with the steps of section 8:
  - who does what (the owner: console start and stop, the deploy key and its ruleset,
    the GitHub logout after setup, Remote Control approval, billing; Claude: setup, runs,
    measurements, reports, pushes to `runs/` with the deploy key);
  - Remote Control runs in the system service `brut-remote-control`, in `tmux -L brut`
    (`tools/vm-setup.sh:164-220`). The owner starts it once after the Claude login and
    answers its first questions (`sudo systemctl start brut-remote-control`, then
    `tmux -L brut attach -t brut`). From then on it starts after every boot and restarts
    by itself. Check it after a restart with `systemctl status brut-remote-control`;
  - the credential steps above, and where the Call-record tarballs go (spec, O13);
  - what a preemption does and how to resume;
  - every path ends with a stop in the Nebius console or through its API, and a look at
    the console. A shutdown inside the VM does not end billing.
  - No host address, no key and no price in the document. Add the file to the list of
    German reports in `docs/SCHREIBSTIL.md`.

## Tests (write them first)

- The scripts pass shellcheck.
- `--dry-run` changes nothing on the machine.
- The setup script fails when port 11434 is reachable on a non-loopback address.
- A scan of these files finds no IPv4 address other than 127.0.0.1, no key and no price.
- On the host, later: `gh auth status` fails on the VM, and a push of any branch other
  than `runs/` with the deploy key is refused.
- On the host, later: a restart of `brut-remote-control` leaves a run of
  `brut-run.service` running.

## Acceptance criteria

- In the pull request: the tests pass, and the owner has reviewed the scripts and the
  runbook.
- On the host, later: the owner has run the setup once. Comments record the Ollama
  version, the model digests, the result of the live test and M2 (GPU memory per model).
  The VM is then stopped in the console.

## Comments
