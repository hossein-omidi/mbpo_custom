#!/usr/bin/env bash
# Simple PV eval workflow for stage0 MBPO paper run.
#
# Usage:
#   ./scripts/run_pv_eval.sh monitor     # training curves only (safe while training)
#   ./scripts/run_pv_eval.sh midterm     # quick eval → evaluation/midterm/latest/
#   ./scripts/run_pv_eval.sh final       # full eval + diagnose + gate → evaluation/pv_stage0_mbpo_paper_movement/
#
# Optional env overrides:
#   TRIAL=/path/to/seed:...   (default: sequential_stage_artifacts/stage0_mbpo_paper_trial_dir.txt)
#   NUM_ROLLOUTS=5            (midterm default 5, final default 10)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
ARTIFACT="$ROOT/sequential_stage_artifacts/stage0_mbpo_paper_trial_dir.txt"
CONFIG_PATH="$ROOT/examples/config/pv_tracking/stage0_single_day_mbpo_paper.py"
EVAL_DATE="2020-06-21"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${CONDA_ENV_NAME:-mbpo}"
cd "$ROOT"

if [[ -z "$MODE" || "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  sed -n '2,8p' "$0"
  exit 0
fi

# --- resolve trial ---
if [[ -n "${TRIAL:-}" ]]; then
  TRIAL="${TRIAL%/}"
elif [[ -f "$ARTIFACT" ]]; then
  TRIAL="$(tr -d '\n\r' < "$ARTIFACT")"
else
  echo "ERROR: set TRIAL=... or train first (missing $ARTIFACT)" >&2
  exit 1
fi
if [[ ! -f "$TRIAL/params.json" ]]; then
  echo "ERROR: not a Ray trial (no params.json): $TRIAL" >&2
  exit 1
fi

# --- resolve checkpoint ---
if [[ -n "${CKPT:-}" ]]; then
  CKPT="${CKPT%/}"
elif [[ -d "$TRIAL/best_eval_checkpoint" ]]; then
  CKPT="$TRIAL/best_eval_checkpoint"
elif [[ -d "$TRIAL/latest_checkpoint" ]]; then
  CKPT="$TRIAL/latest_checkpoint"
else
  CKPT="$(ls -1dt "$TRIAL"/checkpoint_* 2>/dev/null | head -1 || true)"
fi
if [[ -z "$CKPT" || ! -d "$CKPT" ]]; then
  echo "ERROR: no checkpoint in $TRIAL (wait for first save)" >&2
  exit 1
fi

echo "TRIAL=$TRIAL"
echo "CKPT=$CKPT"

case "$MODE" in
  monitor)
    OUT="$ROOT/training_plots/midterm/latest"
    mkdir -p "$OUT"
    python scripts/plot_training_progress.py "$TRIAL" --outdir "$OUT" \
      -m evaluation/return-average policy/shifts-mean model/val_loss real_batch_ratio alpha
    echo ""
    echo "Plots: $OUT"
    echo "Live log: tail -f $TRIAL/progress.csv"
    ;;

  midterm)
    OUT="$ROOT/evaluation/midterm/latest"
    N="${NUM_ROLLOUTS:-5}"
    mkdir -p "$OUT"
    python scripts/evaluate_agent.py "$CKPT" \
      --outdir "$OUT" \
      --eval-protocol inherit \
      --compare-baselines \
      --eval-weather-source clearsky \
      --fixed-eval-dates "$EVAL_DATE" \
      --num-rollouts "$N"
    python scripts/diagnose_tracking.py \
      --eval-dir "$OUT" \
      --outdir "$OUT/diagnostics" \
      --trial-dir "$TRIAL" \
      --progress-csv "$TRIAL/progress.csv" \
      --verify-env
    python scripts/verify_movement_cost_fairness.py \
      --config-path "$CONFIG_PATH" \
      --eval-dir "$OUT"
    python scripts/plot_training_progress.py "$TRIAL" \
      --outdir "$ROOT/training_plots/midterm/latest"
    echo ""
    echo "Eval:  $OUT"
    echo "Diag:  $OUT/diagnostics/tracking_diagnosis.txt"
    echo "Plots: $OUT/evaluation_method_comparison.png"
    ;;

  final)
    OUT="$ROOT/evaluation/pv_stage0_mbpo_paper_movement"
    N="${NUM_ROLLOUTS:-10}"
    mkdir -p "$OUT"
    python scripts/evaluate_agent.py "$CKPT" \
      --outdir "$OUT" \
      --eval-protocol inherit \
      --compare-baselines \
      --eval-weather-source clearsky \
      --fixed-eval-dates "$EVAL_DATE" \
      --num-rollouts "$N"
    python scripts/diagnose_tracking.py \
      --eval-dir "$OUT" \
      --trial-dir "$TRIAL" \
      --progress-csv "$TRIAL/progress.csv" \
      --verify-env \
      --gate
    python scripts/verify_movement_cost_fairness.py \
      --config-path "$CONFIG_PATH" \
      --eval-dir "$OUT"
    python scripts/plot_training_progress.py "$TRIAL" \
      --outdir "$ROOT/training_plots/stage0_mbpo_paper" \
      -m evaluation/return-average policy/shifts-mean model/val_loss alpha
    echo ""
    echo "Eval:  $OUT"
    echo "Diag:  $OUT/diagnostics/tracking_diagnosis.txt"
    echo "Plots: $OUT/evaluation_method_comparison.png"
    echo "Train: $ROOT/training_plots/stage0_mbpo_paper/"
    ;;

  *)
    echo "Unknown mode: $MODE  (use monitor | midterm | final)" >&2
    exit 1
    ;;
esac
