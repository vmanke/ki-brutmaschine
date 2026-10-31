#!/usr/bin/env bash
# Set up a fresh Ubuntu 24.04 VM for vmanke/ki-brutmaschine.
#
# Run it as your normal login user, not as root:
#   bash vm-setup.sh
#
# What it does:
#   - system packages: build tools (Rust needs a linker), git, curl, python3, tmux, ufw, gh;
#   - firewall: only SSH is reachable from outside;
#   - Rust 1.97.0 with clippy and rustfmt (the toolchain of step 1), and cargo-audit;
#   - Claude Code;
#   - a clone of the repository in ~/ki-brutmaschine;
#   - git commit signing with an SSH key, because main only accepts signed commits.
# Two logins stay with you: GitHub (in the browser, started by this script) and Claude.
# The script stores no secret in the repository. Put API keys into ~/ki-brutmaschine/.env
# (it is in .gitignore), never into a commit.
set -euo pipefail

RUST_TOOLCHAIN=1.97.0
REPO_URL=https://github.com/vmanke/ki-brutmaschine.git
REPO_DIR="$HOME/ki-brutmaschine"
SIGNING_KEY="$HOME/.ssh/git_signing_ed25519"

say() { printf '\n== %s\n' "$*"; }

if [ "$(id -u)" -eq 0 ]; then
  echo "Please run this as your normal user, not as root." >&2
  exit 1
fi

say "System packages"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential pkg-config git curl ca-certificates python3 tmux ufw
if ! sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gh; then
  echo "The gh package was not found; the GitHub login below is skipped." >&2
fi

say "Firewall: only SSH from outside"
sudo ufw allow OpenSSH
sudo ufw --force enable

say "Rust $RUST_TOOLCHAIN"
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh
  sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain "$RUST_TOOLCHAIN"
fi
# shellcheck disable=SC1091
. "$HOME/.cargo/env"
rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal --component clippy --component rustfmt
rustup default "$RUST_TOOLCHAIN"
command -v cargo-audit >/dev/null 2>&1 || cargo install cargo-audit --locked

say "Claude Code"
if ! command -v claude >/dev/null 2>&1; then
  curl -fsSL https://claude.ai/install.sh -o /tmp/claude-install.sh
  bash /tmp/claude-install.sh
fi

say "Repository"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

if command -v gh >/dev/null 2>&1; then
  say "GitHub login (a browser code appears; needed for push and for the signing key)"
  gh auth status >/dev/null 2>&1 || gh auth login --hostname github.com --git-protocol https --web
  gh auth refresh --hostname github.com --scopes admin:ssh_signing_key
  gh auth setup-git

  say "Signed commits"
  if [ ! -f "$SIGNING_KEY" ]; then
    ssh-keygen -t ed25519 -N "" -C "ki-brutmaschine commit signing" -f "$SIGNING_KEY"
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

say "Done. Now log in to Claude once and start Remote Control:"
cat <<'EOF'
  cd ~/ki-brutmaschine
  claude                                   # log in once, then leave with /exit
  tmux new -s brut 'claude remote-control'  # keeps running after you log out
After that the VM appears in the Claude app. Tell Claude, and it connects to the folder.
After a restart of the VM, run the tmux line again.
EOF
