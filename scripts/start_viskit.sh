#!/usr/bin/env bash
# Start the Viskit web UI for Ray trial progress.csv files.
# Usage:
#   ./scripts/start_viskit.sh              # all trials under RAY_ROOT
#   ./scripts/start_viskit.sh latest       # newest seed:* trial only
#   ./scripts/start_viskit.sh "$TRIAL"     # one trial directory
#   PORT=6009 ./scripts/start_viskit.sh    # custom port
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAY_ROOT="${RAY_ROOT:-$HOME/ray_mbpo/PVTracking/pv_tracking}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mbpo}"
PORT="${PORT:-6008}"
TARGET="${1:-all}"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Missing conda init script: $CONDA_SH" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

resolve_target() {
  local input="$1"
  if [[ "$input" == "all" ]]; then
    if [[ ! -d "$RAY_ROOT" ]]; then
      echo "Ray trial root not found: $RAY_ROOT" >&2
      echo "Train first, or set RAY_ROOT to your pv_tracking folder." >&2
      exit 1
    fi
    echo "$RAY_ROOT"
    return 0
  fi

  if [[ "$input" == "latest" ]]; then
    local latest
    latest="$(ls -td "$RAY_ROOT"/seed:*/ 2>/dev/null | head -1 || true)"
    if [[ -z "$latest" ]]; then
      echo "No seed:* trials under $RAY_ROOT" >&2
      exit 1
    fi
    echo "${latest%/}"
    return 0
  fi

  echo "${input%/}"
}

DATA_PATH="$(resolve_target "$TARGET")"
if [[ ! -d "$DATA_PATH" ]]; then
  echo "Trial path does not exist: $DATA_PATH" >&2
  exit 1
fi

if [[ ! -f "$DATA_PATH/progress.csv" ]] && [[ "$TARGET" != "all" ]]; then
  echo "Warning: $DATA_PATH/progress.csv not found yet (training may still be starting)." >&2
fi

cd "$ROOT"

if command -v viskit >/dev/null 2>&1; then
  VISKIT_CMD=(viskit)
else
  VISKIT_CMD=(python viskit/viskit/frontend.py)
fi

echo "Viskit data path: $DATA_PATH"
echo "Open in your browser: http://localhost:${PORT}"
echo "Press Ctrl+C to stop."
exec "${VISKIT_CMD[@]}" "$DATA_PATH" --port "$PORT"
