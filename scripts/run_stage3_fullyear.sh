#!/usr/bin/env bash
# Stage 3 full-year — verify / train / plot / eval / MC eval.
# Trial paths: scripts/pv_trial_paths.py (artifact, $TRIAL, ~/ray_mbpo, repo/mbpo_runs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_MODULE="examples.config.pv_tracking.stage3_fullyear_stable_mbpo"
CONFIG_PATH="$ROOT/examples/config/pv_tracking/stage3_fullyear_stable_mbpo.py"
ARTIFACT="$ROOT/sequential_stage_artifacts/stage3_fullyear_trial_dir.txt"
PLOT_DIR="$ROOT/training_plots/stage3_fullyear"
EVAL_DIR="$ROOT/evaluation/pv_stage3_fullyear"
MC_DIR="$ROOT/evaluation/pv_stage3_fullyear_mc"
RAY_TMP="$ROOT/.ray_tmp/stage3_fullyear"
# Default log root (matches config log_dir ~/ray_mbpo/ + domain PVTracking)
export RAY_ROOT="${RAY_ROOT:-$HOME/ray_mbpo/PVTracking/pv_tracking}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mbpo}"
CMD="${1:-verify}"

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
cd "$ROOT"

if ! python -c "import softlearning.scripts.console_scripts" 2>/dev/null; then
  pip install -e "$ROOT" -q
fi

export PV_CPU_PROFILE="${PV_CPU_PROFILE:-full}"
# shellcheck source=/dev/null
source "$ROOT/scripts/pv_cpu_env.sh"
mkdir -p "$RAY_TMP" "$(dirname "$ARTIFACT")" "$PLOT_DIR" "$EVAL_DIR" "$MC_DIR"

run_local() {
  python -m softlearning.scripts.console_scripts run_local "$@"
}

# Single source of truth for trial directory (never use empty artifact file).
resolve_trial() {
  python - "$ROOT" "$ARTIFACT" <<'PY'
import os, sys
repo = sys.argv[1]
artifact = sys.argv[2] if len(sys.argv) > 2 else None
sys.path.insert(0, os.path.join(repo, 'scripts'))
from pv_trial_paths import find_stage3_trial, STAGE3_RUNS_ROOT

trial, warn = find_stage3_trial(artifact_path=artifact)
if warn:
    print(warn, file=sys.stderr)
if not trial:
    raise SystemExit(
        'No Stage 3 trial found. Run: ./scripts/run_stage3_fullyear.sh train\n'
        'Searched: $TRIAL, artifact, %s, ~/ray_mbpo/...' % STAGE3_RUNS_ROOT)
print(trial)
PY
}

write_trial_artifact() {
  local trial
  trial="$(resolve_trial)"
  printf '%s\n' "$trial" > "$ARTIFACT"
  echo "$trial"
}

resolve_ckpt() {
  local trial="$1"
  if [[ -n "${CKPT:-}" ]]; then echo "${CKPT%/}"; return; fi
  if [[ -d "$trial/best_eval_checkpoint" ]]; then echo "$trial/best_eval_checkpoint"; return; fi
  echo "$trial/latest_checkpoint"
}

case "$CMD" in
  status)
    python - <<'PY'
import sys
sys.path.insert(0, 'scripts')
from pv_trial_paths import trial_status_lines
print('\n'.join(trial_status_lines()))
PY
    ;;
  verify)
    python scripts/verify_stage3_fullyear.py
    python scripts/verify_pv_state_space.py --outdir verification/pv_state_space
    echo "Preflight OK — see verification/stage3_fullyear/preflight_report.txt"
    ;;
  train)
    python scripts/verify_stage3_fullyear.py
    echo "=== STAGE 3 FULL YEAR | cpus=$CPUS trial-cpus=$TRIAL_CPUS ==="
    echo "Logs under: $RAY_ROOT (config log_dir ~/ray_mbpo/)"
    run_local examples.development --config="$CONFIG_MODULE" \
      --gpus="$GPUS" --trial-gpus="$TRIAL_GPUS" \
      --cpus="$CPUS" --trial-cpus="$TRIAL_CPUS" --temp-dir="$RAY_TMP"
    TRIAL="$(write_trial_artifact)"
    echo "Trial: $TRIAL"
    echo "export TRIAL=$TRIAL"
    ;;
  plot)
    TRIAL="$(resolve_trial)"
    export TRIAL
    python scripts/plot_training_progress.py "$TRIAL" --outdir "$PLOT_DIR" \
      --metrics evaluation/return-average training/return-average model/val_loss \
      model_rollout_length real_batch_ratio alpha Q_loss policy/shifts-mean
    echo "Plots: $PLOT_DIR"
    echo "Trial: $TRIAL"
    ;;
  eval)
    TRIAL="$(resolve_trial)"
    export TRIAL
    CKPT="$(resolve_ckpt "$TRIAL")"
    python scripts/evaluate_agent.py "$CKPT" \
      --outdir "$EVAL_DIR" \
      --eval-protocol inherit \
      --max-path-length 78 \
      --compare-baselines \
      --num-rollouts "${NUM_ROLLOUTS:-16}" \
      --max-rollout-plots "${MAX_ROLLOUT_PLOTS:-4}" \
      --eval-seed-base 100000
    python scripts/diagnose_tracking.py \
      --eval-dir "$EVAL_DIR" \
      --outdir "$EVAL_DIR/diagnostics" \
      --trial-dir "$TRIAL" \
      --progress-csv "$TRIAL/progress.csv" \
      --max-paired-plot-days "${MAX_PAIRED_PLOT_DAYS:-6}" \
      --verify-env
    echo "Eval: $EVAL_DIR"
    echo "Trial: $TRIAL  CKPT: $CKPT"
    ;;
  mc-eval)
    TRIAL="$(resolve_trial)"
    export TRIAL
    CKPT="$(resolve_ckpt "$TRIAL")"
    python scripts/evaluate_fullyear_mc.py "$CKPT" \
      --outdir "$MC_DIR" \
      --eval-protocol inherit \
      --date-set annual \
      --num-rollouts "${MC_REPLICATES:-16}" \
      --replicates-per-date 1 \
      --error-bars sem \
      --run-standard-eval \
      ${MC_EXTRA_ARGS:-}
    echo "MC eval: $MC_DIR"
    ;;
  mc-eval-stress)
    TRIAL="$(resolve_trial)"
    export TRIAL
    CKPT="$(resolve_ckpt "$TRIAL")"
    python scripts/evaluate_fullyear_mc.py "$CKPT" \
      --outdir "${MC_DIR}_stress" \
      --eval-protocol inherit \
      --date-set stress_test \
      --replicates-per-date "${MC_REPLICATES:-8}" \
      --error-bars sem
    echo "Stress-test eval (diagnostic only): ${MC_DIR}_stress"
    ;;
  mc-eval-holdout)
    "$0" mc-eval-stress
    ;;
  all-eval)
    "$0" eval
    "$0" mc-eval
    ;;
  gate)
    TRIAL="$(resolve_trial)"
    export TRIAL
    python scripts/diagnose_tracking.py \
      --eval-dir "$EVAL_DIR" \
      --outdir "$EVAL_DIR/diagnostics" \
      --trial-dir "$TRIAL" \
      --progress-csv "$TRIAL/progress.csv" \
      --max-paired-plot-days "${MAX_PAIRED_PLOT_DAYS:-6}" \
      --verify-env --gate
    ;;
  *)
    echo "Usage: $0 {status|verify|train|plot|eval|mc-eval|mc-eval-stress|all-eval|gate}"
    echo "  ./scripts/run_stage3_fullyear.sh status   # show trial paths"
    echo "  export TRIAL=\$(./scripts/run_stage3_fullyear.sh status 2>&1 | grep '^Resolved' | awk '{print \$3}')"
    exit 1
    ;;
esac
