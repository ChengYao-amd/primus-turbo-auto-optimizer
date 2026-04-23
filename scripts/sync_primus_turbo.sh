#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${PRIMUS_TURBO_URL:-git@github.com:AMD-AGI/Primus-Turbo.git}"
DEST="${PRIMUS_TURBO_DIR:-agent_workspace/Primus-Turbo}"
BRANCH="${PRIMUS_TURBO_BRANCH:-main}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$(dirname "$DEST")"
  git clone --branch "$BRANCH" "$REPO_URL" "$DEST"
  cd "$DEST"
  git submodule init
  git submodule update
  cd "$REPO_ROOT"
else
  git -C "$DEST" fetch origin "$BRANCH"
  git -C "$DEST" checkout "$BRANCH"
  git -C "$DEST" reset --hard "origin/$BRANCH"
fi

echo "Primus-Turbo synced: $(git -C "$DEST" rev-parse --short HEAD) on $BRANCH"
