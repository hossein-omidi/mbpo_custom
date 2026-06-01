#!/usr/bin/env bash
# Stage 0 baseline (stage0_single_day.py) — train / plot / eval / gate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_MODULE="examples.config.pv_tracking.stage0_single_day"
CONFIG_PATH="$ROOT/examples/config/pv_tracking/stage0_single_day.py"
ARTIFACT="$ROOT/sequential_stage_artifacts/stage0_baseline_trial_dir.txt"
PLOT_DIR="$ROOT/training_plots/stage0_baseline"
EVAL_DIR="$ROOT/evaluation/pv_stage0_baseline"
RAY_TMP="$ROOT/.ray_tmp/stage0_baseline"
RAY_ROOT="${RAY_ROOT:-$HOME/ray_mbpo/PVTracking/pv_tracking}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mbpo}"
CMD="${1:-train}"

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
source "$ROOT/scripts/pv_cpu_env.sh"
cd "$ROOT"
mkdir -p "$RAY_TMP" "$(dirname "$ARTIFACT")"

resolve_trial() {
  if [[ -n "${TRIAL:-}" ]]; then echo "${TRIAL%/}"; return; fi
  if [[ -f "$ARTIFACT" ]]; then tr -d '\n' < "$ARTIFACT"; return; fi
  python - <<'PY'
import glob, json, os
root = os.path.expanduser(os.environ.get('RAY_ROOT', '~/ray_mbpo/PVTracking/pv_tracking'))
trials = sorted(glob.glob(os.path.join(root, 'seed:*')), key=os.path.getmtime, reverse=True)
for t in trials:
    p = os.path.join(t, 'params.json')
    if not os.path.isfile(p): continue
    v = json.load(open(p))
    cv = v.get('config_version', '')
    if 'stage0_single_day' in cv and 'mbpo_paper' not in cv:
        print(t.rstrip('/')); break
PY
}

case "$CMD" in
  train)
    python scripts/verify_training_config.py --config "$CONFIG_MODULE"
    python scripts/validate_pv_rollouts.py --config-path "$CONFIG_PATH"
    echo "=== BASELINE | cpus=$CPUS trial-cpus=$TRIAL_CPUS | $RAY_TMP ==="
    mbpo run_local examples.development --config="$CONFIG_MODULE" \
      --gpus="$GPUS" --trial-gpus="$TRIAL_GPUS" \
      --cpus="$CPUS" --trial-cpus="$TRIAL_CPUS" --temp-dir="$RAY_TMP"
    ls -1dt "$RAY_ROOT"/seed:* 2>/dev/null | head -1 | tr -d '\n' > "$ARTIFACT"
    echo "Trial: $(cat "$ARTIFACT")"
    ;;
  plot)
    TRIAL="$(resolve_trial)"; mkdir -p "$PLOT_DIR"
    python scripts/plot_training_progress.py "$TRIAL" --outdir "$PLOT_DIR"
    echo "$PLOT_DIR"
    ;;
  eval)
    TRIAL="$(resolve_trial)"; CKPT="$TRIAL/best_eval_checkpoint"
    [[ -d "$CKPT" ]] || CKPT="$TRIAL/latest_checkpoint"
    mkdir -p "$EVAL_DIR"
    python scripts/evaluate_agent.py "$CKPT" --outdir "$EVAL_DIR" \
      --eval-protocol inherit --max-path-length 39 --compare-baselines \
      --eval-weather-source clearsky --fixed-eval-dates 2020-06-21 --num-rollouts 10
    ;;
  gate)
    TRIAL="$(resolve_trial)"
    python scripts/diagnose_tracking.py --eval-dir "$EVAL_DIR" --trial-dir "$TRIAL" \
      --progress-csv "$TRIAL/progress.csv" --verify-env --gate
    ;;
  *) echo "Usage: PV_CPU_PROFILE={single|dual} $0 {train|plot|eval|gate}"; exit 1 ;;
esac
