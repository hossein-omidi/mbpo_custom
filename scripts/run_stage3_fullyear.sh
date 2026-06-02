#!/usr/bin/env bash
# Stage 3 full-year historical weather — verify / train / plot / eval / MC eval.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_MODULE="examples.config.pv_tracking.stage3_fullyear_random_clean_split"
CONFIG_PATH="$ROOT/examples/config/pv_tracking/stage3_fullyear_random_clean_split.py"
ARTIFACT="$ROOT/sequential_stage_artifacts/stage3_fullyear_trial_dir.txt"
PLOT_DIR="$ROOT/training_plots/stage3_fullyear"
EVAL_DIR="$ROOT/evaluation/pv_stage3_fullyear"
MC_DIR="$ROOT/evaluation/pv_stage3_fullyear_mc"
RAY_TMP="$ROOT/.ray_tmp/stage3_fullyear"
RAY_ROOT="${RAY_ROOT:-$HOME/ray_mbpo/PVTracking/pv_tracking}"
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

resolve_trial() {
  if [[ -n "${TRIAL:-}" ]]; then echo "${TRIAL%/}"; return; fi
  if [[ -f "$ARTIFACT" ]]; then tr -d '\n\r' < "$ARTIFACT"; return; fi
  python - <<'PY'
import glob, json, os
root = os.path.expanduser(os.environ.get('RAY_ROOT', '~/ray_mbpo/PVTracking/pv_tracking'))
want = 'pv_tracking_stage3_fullyear_rl_annual'
for t in sorted(glob.glob(os.path.join(root, 'seed:*')), key=os.path.getmtime, reverse=True):
    p = os.path.join(t, 'params.json')
    if not os.path.isfile(p): continue
    if want in json.load(open(p)).get('config_version', ''):
        print(t.rstrip('/')); break
PY
}

resolve_ckpt() {
  local trial="$1"
  if [[ -n "${CKPT:-}" ]]; then echo "${CKPT%/}"; return; fi
  if [[ -d "$trial/best_eval_checkpoint" ]]; then echo "$trial/best_eval_checkpoint"; return; fi
  echo "$trial/latest_checkpoint"
}

case "$CMD" in
  verify)
    python scripts/verify_stage3_fullyear.py
    python scripts/verify_pv_state_space.py --outdir verification/pv_state_space
    echo "Preflight OK — see verification/stage3_fullyear/preflight_report.txt"
    ;;
  train)
    python scripts/verify_stage3_fullyear.py
    echo "=== STAGE 3 FULL YEAR | cpus=$CPUS trial-cpus=$TRIAL_CPUS ==="
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
      model_rollout_length real_batch_ratio alpha Q_loss policy/shifts-mean
    echo "Plots: $PLOT_DIR (see also evaluation/*/paper_figures/training_diagnostics/)"
    ;;
  eval)
    TRIAL="$(resolve_trial)"
    CKPT="$(resolve_ckpt "$TRIAL")"
    python scripts/evaluate_agent.py "$CKPT" \
      --outdir "$EVAL_DIR" \
      --eval-protocol inherit \
      --max-path-length 78 \
      --compare-baselines \
      --num-rollouts "${NUM_ROLLOUTS:-40}" \
      --eval-seed-base 100000
    python scripts/diagnose_tracking.py \
      --eval-dir "$EVAL_DIR" \
      --outdir "$EVAL_DIR/diagnostics" \
      --trial-dir "$TRIAL" \
      --progress-csv "$TRIAL/progress.csv" \
      --verify-env
    echo "Eval: $EVAL_DIR"
    ;;
  mc-eval)
    TRIAL="$(resolve_trial)"
    CKPT="$(resolve_ckpt "$TRIAL")"
    python scripts/evaluate_fullyear_mc.py "$CKPT" \
      --outdir "$MC_DIR" \
      --eval-protocol inherit \
      --date-set annual \
      --num-rollouts "${MC_REPLICATES:-40}" \
      --replicates-per-date 1 \
      --error-bars sem \
      --run-standard-eval \
      ${MC_EXTRA_ARGS:-}
    echo "MC eval: $MC_DIR"
    echo "  paper_figures/ (annual performance, decomposition, seasonal, gains, trajectories)"
    echo "  season_energy_yield_mc_sem.png"
    echo "  mc_dashboard.pdf"
    ;;
  mc-eval-stress)
    TRIAL="$(resolve_trial)"
    CKPT="$(resolve_ckpt "$TRIAL")"
    python scripts/evaluate_fullyear_mc.py "$CKPT" \
      --outdir "${MC_DIR}_stress" \
      --eval-protocol inherit \
      --date-set stress_test \
      --replicates-per-date "${MC_REPLICATES:-8}" \
      --error-bars sem
    echo "Stress-test eval: ${MC_DIR}_stress"
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
    python scripts/diagnose_tracking.py \
      --eval-dir "$EVAL_DIR" \
      --outdir "$EVAL_DIR/diagnostics" \
      --trial-dir "$TRIAL" \
      --progress-csv "$TRIAL/progress.csv" \
      --verify-env --gate
    ;;
  *)
    echo "Usage: $0 {verify|train|plot|eval|mc-eval|mc-eval-stress|all-eval|gate}"
    echo "  MC_REPLICATES=12  optional extra MC replicates"
    exit 1
    ;;
esac
