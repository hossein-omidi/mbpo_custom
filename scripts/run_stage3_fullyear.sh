#!/usr/bin/env bash
# Deprecated — use ./train.sh and ./result.sh from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Use: $ROOT/train.sh <run> conf1  and  $ROOT/result.sh <run> --full" >&2
case "${1:-}" in
  train) exec "$ROOT/train.sh" stage3_fullyear conf1 "${@:2}" ;;
  plot)  export RUN_NAME=stage3_fullyear; exec "$ROOT/result.sh" stage3_fullyear --plot-only ;;
  eval|all-eval) export RUN_NAME=stage3_fullyear; exec "$ROOT/result.sh" stage3_fullyear --full ;;
  verify) exec python "$ROOT/scripts/verify_preflight.py" --config examples.config.pv_tracking.conf1 ;;
  status) exec "$ROOT/result.sh" stage3_fullyear --status ;;
  *) echo "Wrapper maps: train→train.sh stage3_fullyear conf1; eval→result.sh --full" >&2; exit 1 ;;
esac
