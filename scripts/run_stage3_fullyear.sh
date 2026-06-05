#!/usr/bin/env bash
# Deprecated — use ./train.sh and ./result.sh from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Use: $ROOT/train.sh <run> stage3_nsrdb  and  $ROOT/result.sh <run> --full" >&2
case "${1:-}" in
  train) exec "$ROOT/train.sh" stage3_nsrdb stage3_nsrdb "${@:2}" ;;
  plot)  exec "$ROOT/result.sh" stage3_nsrdb --plot-only ;;
  eval|all-eval) exec "$ROOT/result.sh" stage3_nsrdb --full ;;
  verify) exec python "$ROOT/scripts/verify_preflight.py" --config examples.config.pv_tracking.stage3_multiyear_nsrdb_scenario ;;
  status) exec "$ROOT/result.sh" stage3_nsrdb --status ;;
  *) echo "Wrapper maps: train→train.sh stage3_nsrdb; eval→result.sh --full" >&2; exit 1 ;;
esac
