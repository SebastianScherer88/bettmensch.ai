#!/usr/bin/env bash
# Runs once, after the devcontainer is created (see devcontainer.json's
# postCreateCommand). Installs the two things the base python devcontainer
# image doesn't already have but `make pipelines.*` needs: `make` itself,
# and `uv` (pipelines's own dependency manager - see pipelines.makefile).
set -euo pipefail

echo "Installing make..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends make

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$HOME/.venv-devcontainer/bin:$PATH"
if ! grep -q '.venv-devcontainer/bin' "$HOME/.bashrc" 2>/dev/null; then
  echo 'export PATH="$HOME/.local/bin:$HOME/.venv-devcontainer/bin:$PATH"' >> "$HOME/.bashrc"
fi

# UV_PROJECT_ENVIRONMENT (set in devcontainer.json's containerEnv) points
# this at $HOME/.venv-devcontainer, *outside* the bind-mounted workspace -
# deliberately: `uv sync`'s default `.venv` would otherwise collide with
# whatever `.venv` already exists there from the Windows host (a different,
# incompatible OS's venv). Do not remove that env var or run `uv sync`
# without it set.
echo "Installing pipelines dependencies (uv sync --extra postgres)..."
uv sync --extra postgres

cat <<'EOF'

Devcontainer ready. From this shell, try:

  make pipelines.docker.up
  make pipelines.test SUITE=unit
  make pipelines.test SUITE=integration
  make pipelines.test.docker SUITE=all
  make pipelines.docker.down

See README.md's "Running tests" > pipelines section for the full flow.
EOF
