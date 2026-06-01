#!/usr/bin/env bash
# Stage 0 MBPO paper + movement_penalty — train / plot / eval / gate (single run).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_MODULE="examples.config.pv_tracking.stage0_single_day_mbpo_paper"
CONFIG_PATH="$ROOT/examples/config/pv_tracking/stage0_single_day_mbpo_paper.py"
ARTIFACT="$ROOT/sequential_stage_artifacts/stage0_mbpo_paper_trial_dir.txt"
PLOT_DIR="$ROOT/training_plots/stage0_mbpo_paper"
EVAL_DIR="$ROOT/evaluation/pv_stage0_mbpo_paper_movement"
RAY_TMP="$ROOT/.ray_tmp/stage0_mbpo_paper"
RAY_ROOT="${RAY_ROOT:-$HOME/ray_mbpo/PVTracking/pv_tracking}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mbpo}"
CMD="${1:-train}"

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
cd "$ROOT"

# mbpo CLI needs: pip install -e .  (from repo root, once per env)
if ! python -c "import softlearning.scripts.console_scripts" 2>/dev/null; then
  echo "Installing mbpo package (pip install -e .) ..."
  pip install -e "$ROOT" -q
fi

export PV_CPU_PROFILE="${PV_CPU_PROFILE:-full}"
# shellcheck source=/dev/null
source "$ROOT/scripts/pv_cpu_env.sh"
mkdir -p "$RAY_TMP" "$(dirname "$ARTIFACT")" "$PLOT_DIR" "$EVAL_DIR"

run_local() {
  python -m softlearning.scripts.console_scripts run_local "$@"
}

resolve_trial() {
  if [[ -n "${TRIAL:-}" ]]; then echo "${TRIAL%/}"; return; fi
  if [[ -f "$ARTIFACT" ]]; then tr -d '\n' < "$ARTIFACT"; return; fi
  python - <<'PY'
import glob, json, os
root = os.path.expanduser(os.environ.get('RAY_ROOT', '~/ray_mbpo/PVTracking/pv_tracking'))
want = 'pv_tracking_stage0_single_day_mbpo_paper'
for t in sorted(glob.glob(os.path.join(root, 'seed:*')), key=os.path.getmtime, reverse=True):
    p = os.path.join(t, 'params.json')
    if not os.path.isfile(p): continue
    if want in json.load(open(p)).get('config_version', ''):
        print(t.rstrip('/')); break
PY
}

case "$CMD" in
  train)
    python scripts/verify_training_config.py --config "$CONFIG_MODULE"
    python scripts/validate_pv_rollouts.py --config-path "$CONFIG_PATH"
    python scripts/verify_movement_cost_fairness.py --config-path "$CONFIG_PATH"
    echo "=== MBPO PAPER+MOVEMENT | cpus=$CPUS trial-cpus=$TRIAL_CPUS ==="
    echo "Ray: $RAY_TMP  Plots: $PLOT_DIR  Eval: $EVAL_DIR"
    run_local examples.development --config="$CONFIG_MODULE" \
      --gpus="$GPUS" --trial-gpus="$TRIAL_GPUS" \
      --cpus="$CPUS" --trial-cpus="$TRIAL_CPUS" --temp-dir="$RAY_TMP"
    ls -1dt "$RAY_ROOT"/seed:* 2>/dev/null | head -1 | tr -d '\n' > "$ARTIFACT"
    echo "Trial: $(cat "$ARTIFACT")"
    ;;
  plot)
    TRIAL="$(resolve_trial)"
    python scripts/plot_training_progress.py "$TRIAL" --outdir "$PLOT_DIR" \
      --metrics evaluation/return-average training/return-average model/val_loss \
      policy/shifts-mean policy/actions-std alpha
    echo "Plots: $PLOT_DIR"
    ;;
  eval)
    TRIAL="$(resolve_trial)"
    CKPT="$TRIAL/best_eval_checkpoint"; [[ -d "$CKPT" ]] || CKPT="$TRIAL/latest_checkpoint"
    python scripts/evaluate_agent.py "$CKPT" --outdir "$EVAL_DIR" \
      --eval-protocol inherit --max-path-length 78 --compare-baselines \
      --eval-weather-source clearsky --fixed-eval-dates 2020-06-21 --num-rollouts 10
    python scripts/verify_movement_cost_fairness.py --eval-dir "$EVAL_DIR"
    echo "Eval: $EVAL_DIR"
    ;;
  gate)
    TRIAL="$(resolve_trial)"
    python scripts/diagnose_tracking.py --eval-dir "$EVAL_DIR" --trial-dir "$TRIAL" \
      --progress-csv "$TRIAL/progress.csv" --verify-env --gate
    ;;
  verify)
    python scripts/verify_training_config.py --config "$CONFIG_MODULE"
    python scripts/validate_pv_rollouts.py --config-path "$CONFIG_PATH"
    python scripts/verify_movement_cost_fairness.py --config-path "$CONFIG_PATH"
    ;;
  *)
    echo "Usage: $0 {train|plot|eval|gate|verify}"; exit 1 ;;
esac
