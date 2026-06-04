#!/usr/bin/env bash
# Train MBPO-SAC: ./train.sh <run_name> <conf> [--cpus N] [--trial-cpus M] [--verify]
# Example: ./train.sh run1 conf1 --cpus 4 --trial-cpus 2
#          ./train.sh smoke1 conf_smoke --cpus 2 --trial-cpus 1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mbpo}"

usage() {
  echo "Usage: $0 <run_name> <conf> [--cpus N] [--trial-cpus M] [--verify]"
  echo "  conf: conf1 | conf2 | conf3 | conf_smoke  (see: python -c \"from examples.config.pv_tracking.conf_registry import list_configs; list_configs()\")"
  exit 1
}

[[ $# -ge 2 ]] || usage
RUN_NAME="$1"
CONF="$2"
shift 2

CPUS="${CPUS:-4}"
TRIAL_CPUS="${TRIAL_CPUS:-2}"
GPUS="${GPUS:-0}"
TRIAL_GPUS="${TRIAL_GPUS:-0}"
DO_VERIFY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpus) CPUS="$2"; shift 2 ;;
    --trial-cpus) TRIAL_CPUS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --trial-gpus) TRIAL_GPUS="$2"; shift 2 ;;
    --verify) DO_VERIFY=1; shift ;;
    cpu[0-9]*) CPUS="${1#cpu}"; shift ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
cd "$ROOT"

if ! python -c "import softlearning.scripts.console_scripts" 2>/dev/null; then
  pip install -e "$ROOT" -q
fi

export PV_CPU_PROFILE="${PV_CPU_PROFILE:-full}"
# shellcheck source=/dev/null
source "$ROOT/scripts/pv_cpu_env.sh"

RUN_DIR="$ROOT/runs/$RUN_NAME"
export MBPO_LOG_DIR="$RUN_DIR/checkpoints"
export RAY_ROOT="$MBPO_LOG_DIR/PVTracking/pv_tracking"
RAY_TMP="$RUN_DIR/ray_tmp"
mkdir -p "$RUN_DIR" "$RAY_TMP" "$MBPO_LOG_DIR"

MODULE="$(python -c "
import sys
sys.path.insert(0, '$ROOT')
from examples.config.pv_tracking.conf_registry import resolve_conf
print(resolve_conf('$CONF')[0])
")"
CONFIG_PATH="$ROOT/examples/config/pv_tracking/$(python -c "
from examples.config.pv_tracking.conf_registry import resolve_conf
print(resolve_conf('$CONF')[1])
")"

python -c "
import sys, json, os
sys.path.insert(0, '$ROOT')
from scripts.run_registry import write_run_meta
write_run_meta('$RUN_NAME', '$CONF', '$MODULE', $CPUS, $TRIAL_CPUS)
"

if [[ "$DO_VERIFY" -eq 1 ]]; then
  python scripts/verify_preflight.py --config "$MODULE" --config-path "$CONFIG_PATH"
fi

echo "=== train.sh | run=$RUN_NAME conf=$CONF cpus=$CPUS trial-cpus=$TRIAL_CPUS ==="
echo "Checkpoints: $RAY_ROOT"
echo "Ray temp:   $RAY_TMP"

python -m softlearning.scripts.console_scripts run_local examples.development \
  --config="$MODULE" \
  --log-dir="$MBPO_LOG_DIR" \
  --gpus="$GPUS" --trial-gpus="$TRIAL_GPUS" \
  --cpus="$CPUS" --trial-cpus="$TRIAL_CPUS" \
  --temp-dir="$RAY_TMP"

TRIAL="$(python -c "
import sys
sys.path.insert(0, '$ROOT')
from scripts.run_registry import sync_trial_pointer_from_disk
t = sync_trial_pointer_from_disk('$RUN_NAME')
if not t:
    raise SystemExit(1)
print(t)
")" || {
  echo "WARNING: training finished but no seed:* trial found under $RAY_ROOT"
  echo "Check Ray output above; then: python -c \"from scripts.run_registry import sync_trial_pointer_from_disk; print(sync_trial_pointer_from_disk('$RUN_NAME'))\""
  exit 1
}

echo "Trial saved: $TRIAL"
echo "Pointer:     $RUN_DIR/trial_dir.txt"
echo "Next:        ./result.sh $RUN_NAME --plot-only   # while or after training"
echo "             ./result.sh $RUN_NAME --full        # plot + eval"
