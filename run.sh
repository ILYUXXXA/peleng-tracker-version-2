#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

CONFIG_PATH="${1:-${PELENG_TRACKER_CONFIG:-config/hackathon.local.yaml}}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: Node.js/npm is not installed: https://nodejs.org/" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  if [[ "$CONFIG_PATH" != "config/hackathon.local.yaml" ]]; then
    echo "Error: config not found: $CONFIG_PATH" >&2
    exit 1
  fi
  cp config/hackathon.example.yaml "$CONFIG_PATH"
  echo "Created $CONFIG_PATH from the example config."
fi

uv sync

if [[ ! -d web/node_modules ]]; then
  (cd web && npm ci)
fi
(cd web && npm run build)

echo "Starting Peleng Tracker with $CONFIG_PATH"
exec uv run peleng-tracker run --config "$CONFIG_PATH"
