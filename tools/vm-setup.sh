#!/usr/bin/env bash
# Set up a fresh Ubuntu 24.04 VM for vmanke/ki-brutmaschine.
#
# Run it as your normal login user, not as root:
#   bash vm-setup.sh
#
# What it does:
#   - waits until the first-boot setup (cloud-init) and the automatic updates are done;
#   - system packages: build tools (Rust needs a linker), git, curl, python3, tmux, ufw, gh;
#   - firewall: only SSH is reachable from outside;
#   - Rust 1.97.0 with clippy and rustfmt (the toolchain of step 1), and cargo-audit;
#   - Claude Code;
#   - a clone of the repository in ~/ki-brutmaschine;
#   - git commit signing with an SSH key, because main only accepts signed commits;
#   - a system service that starts Claude Remote Control in tmux after every boot.
# Two logins stay with you: GitHub (in the browser, started by this script) and Claude.
# The script stores no secret in the repository. Put API keys into ~/ki-brutmaschine/.env
# (it is in .gitignore), never into a commit.
# The script may run again at any time. Every step skips what is already done.
set -euo pipefail

RUST_TOOLCHAIN=1.97.0
REPO_URL=https://github.com/vmanke/ki-brutmaschine.git
REPO_DIR="$HOME/ki-brutmaschine"
SIGNING_KEY="$HOME/.ssh/git_signing_ed25519"
SERVICE=brut-remote-control
# Wait up to 10 minutes for the dpkg lock instead of failing at once.
APT_GET=(sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600)

say() { printf '\n== %s\n' "$*"; }

# Runs a command up to 10 times, 15 s apart. For apt, which fails while another apt runs.
retry() {
  local n
  for n in 1 2 3 4 5 6 7 8 9; do
    "$@" && return 0
    echo "Failed; trying again in 15 s ($n of 9)." >&2
    sleep 15
  done
  "$@"
}

if [ "$(id -u)" -eq 0 ]; then
  echo "Please run this as your normal user, not as root." >&2
  exit 1
fi

say "Waiting for the first-boot setup and the automatic updates"
if command -v cloud-init >/dev/null 2>&1; then
  # Exit code 2 means "done, with recoverable errors"; the steps below work either way.
  sudo cloud-init status --wait >/dev/null || true
fi
# A transient unit ordered after the update services starts only when they have finished.
sudo systemd-run --quiet --wait \
  --property=After=apt-daily.service --property=After=apt-daily-upgrade.service /bin/true || true

say "System packages"
retry "${APT_GET[@]}" update
retry "${APT_GET[@]}" install -y --no-install-recommends \
  build-essential pkg-config git curl ca-certificates python3 tmux ufw
if ! "${APT_GET[@]}" install -y --no-install-recommends gh; then
  echo "The gh package was not found; the GitHub login below is skipped." >&2
fi

say "Firewall: only SSH from outside"
sudo ufw allow OpenSSH
sudo ufw --force enable

say "Rust $RUST_TOOLCHAIN"
if ! command -v rustup >/dev/null 2>&1 && [ ! -x "$HOME/.cargo/bin/rustup" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh
  sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain "$RUST_TOOLCHAIN"
fi
# shellcheck disable=SC1091
. "$HOME/.cargo/env"
rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal --component clippy --component rustfmt
rustup default "$RUST_TOOLCHAIN"
command -v cargo-audit >/dev/null 2>&1 || cargo install cargo-audit --locked

say "Claude Code"
# The installer puts claude into ~/.local/bin. A shell that started before that folder
# existed does not have it on its PATH, so look there first.
if [ ! -x "$HOME/.local/bin/claude" ] && ! command -v claude >/dev/null 2>&1; then
  curl -fsSL https://claude.ai/install.sh -o /tmp/claude-install.sh
  bash /tmp/claude-install.sh
fi
if [ -x "$HOME/.local/bin/claude" ]; then
  CLAUDE_BIN="$HOME/.local/bin/claude"
else
  CLAUDE_BIN=$(command -v claude)
fi

say "Repository"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

if command -v gh >/dev/null 2>&1; then
  say "GitHub login (a code appears; enter it at github.com/login/device)"
  # One login with the scope for signing keys, so that a new VM needs only one code.
  if ! gh auth status --hostname github.com >/dev/null 2>&1; then
    gh auth login --hostname github.com --git-protocol https --web --scopes admin:ssh_signing_key
  elif ! gh api user/ssh_signing_keys >/dev/null 2>&1; then
    gh auth refresh --hostname github.com --scopes admin:ssh_signing_key
  fi
  gh auth setup-git

  say "Signed commits"
  if [ ! -f "$SIGNING_KEY" ]; then
    ssh-keygen -t ed25519 -N "" -C "ki-brutmaschine commit signing" -f "$SIGNING_KEY"
  fi
  # Upload the key unless GitHub knows it already. This also repairs a run that created
  # the key but stopped before the upload.
  key_body=$(awk '{print $2}' "$SIGNING_KEY.pub")
  known_keys=$(gh api user/ssh_signing_keys --jq '.[].key')
  if ! grep -q -F -- "$key_body" <<<"$known_keys"; then
    gh ssh-key add "$SIGNING_KEY.pub" --type signing --title "ki-brutmaschine VM $(hostname)"
  fi
  login=$(gh api user --jq .login)
  id=$(gh api user --jq .id)
  git -C "$REPO_DIR" config user.name "$login"
  git -C "$REPO_DIR" config user.email "$id+$login@users.noreply.github.com"
  git -C "$REPO_DIR" config gpg.format ssh
  git -C "$REPO_DIR" config user.signingkey "$SIGNING_KEY.pub"
  git -C "$REPO_DIR" config commit.gpgsign true
fi

say "Remote Control after every boot"
# The service starts Remote Control in a tmux session named brut, on its own tmux socket
# (-L brut), so it never mixes with tmux sessions you start yourself. It is enabled for
# the next boot but not started now: the first start must happen by hand, because Remote
# Control asks two questions once (see below).
run_user=$(id -un)
sudo tee "/etc/systemd/system/$SERVICE.service" >/dev/null <<EOF
[Unit]
Description=Claude Remote Control for ki-brutmaschine (tmux -L brut, session brut)
Wants=network-online.target
After=network-online.target

[Service]
Type=forking
User=$run_user
WorkingDirectory=$REPO_DIR
Environment=HOME=$HOME
Environment=PATH=$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/usr/bin/tmux -L brut new-session -d -s brut -c $REPO_DIR $CLAUDE_BIN remote-control
ExecStop=/usr/bin/tmux -L brut kill-session -t brut
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE" >/dev/null 2>&1

say "Done. If you have not done so yet, do this once by hand:"
cat <<EOF
  1. Log out and in again, so that the shell finds claude and cargo:
       exit
     Then connect with ssh again.
  2. Log in to Claude with your claude.ai account:
       claude auth login
  3. Start Remote Control once by hand and answer its questions with y:
       tmux new -s brut -c ~/ki-brutmaschine 'claude remote-control'
     You may close the terminal then; tmux keeps it running.
From the next boot on, the service $SERVICE starts Remote Control by itself.
Do not start it by hand as well then, or two copies run.
Look at it with: systemctl status $SERVICE
Watch it with:   tmux -L brut attach -t brut   (leave again with Ctrl+b, then d)
EOF
