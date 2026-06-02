#!/usr/bin/env bash
# PV monitor / midterm / final eval (auto-detects newest trial).
#
#   ./scripts/run_pv_eval.sh trial          # print active TRIAL + CKPT
#   ./scripts/run_pv_eval.sh monitor        # plots once → training_plots/active_run/
#   ./scripts/run_pv_eval.sh watch-progress # tail progress.csv every 30s
#   ./scripts/run_pv_eval.sh watch-plots    # refresh plots every 60s
#   ./scripts/run_pv_eval.sh midterm        # → evaluation/active_latest/
#   ./scripts/run_pv_eval.sh final          # → evaluation/paper_movement_final/
#
# Override:  TRIAL=/path/to/seed:...  CKPT=...  NUM_ROLLOUTS=5
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
RAY_ROOT="${RAY_ROOT:-$HOME/ray_mbpo/PVTracking/pv_tracking}"
ARTIFACT="$ROOT/sequential_stage_artifacts/stage0_mbpo_paper_trial_dir.txt"
CONFIG_PATH="$ROOT/examples/config/pv_tracking/stage0_single_day_mbpo_paper.py"
# Inherit eval date + weather_source from checkpoint config (historical cloudy day).
PLOT_DIR="$ROOT/training_plots/active_run"
MID_DIR="$ROOT/evaluation/active_latest"
FINAL_DIR="$ROOT/evaluation/paper_movement_final"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${CONDA_ENV_NAME:-mbpo}"
cd "$ROOT"

if [[ -z "$MODE" || "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  sed -n '2,11p' "$0"
  exit 0
fi

resolve_trial() {
  if [[ -n "${TRIAL:-}" ]]; then
    echo "${TRIAL%/}"
    return
  fi
  local newest
  newest="$(ls -1dt "$RAY_ROOT"/seed:* 2>/dev/null | head -1 || true)"
  if [[ -n "$newest" && -f "$newest/params.json" ]]; then
    echo "${newest%/}"
    return
  fi
  if [[ -f "$ARTIFACT" ]]; then
    tr -d '\n\r' < "$ARTIFACT"
    return
  fi
  echo "ERROR: no trial — set TRIAL= or train first" >&2
  exit 1
}

resolve_ckpt() {
  local trial="$1"
  if [[ -n "${CKPT:-}" ]]; then
    echo "${CKPT%/}"
    return
  fi
  local numbered
  numbered="$(ls -1d "$trial"/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1 || true)"
  if [[ -n "$numbered" && -d "$numbered" ]]; then
    echo "$numbered"
    return
  fi
  if [[ -d "$trial/best_eval_checkpoint" ]]; then
    echo "$trial/best_eval_checkpoint"
    return
  fi
  if [[ -d "$trial/latest_checkpoint" ]]; then
    echo "$trial/latest_checkpoint"
    return
  fi
  echo "ERROR: no checkpoint in $trial (wait for save)" >&2
  exit 1
}

TRIAL="$(resolve_trial)"
CKPT="$(resolve_ckpt "$TRIAL")"
echo "Active trial: $TRIAL"
echo "Checkpoint:   $CKPT"

case "$MODE" in
  trial)
    echo "export TRIAL=$TRIAL"
    echo "export CKPT=$CKPT"
    ;;

  monitor)
    mkdir -p "$PLOT_DIR"
    python scripts/plot_training_progress.py "$TRIAL" --outdir "$PLOT_DIR" \
      -m evaluation/return-average policy/shifts-mean model/val_loss real_batch_ratio alpha
    echo "Plots: $PLOT_DIR"
    echo "Live:  tail -f $TRIAL/progress.csv"
    ;;

  watch-progress)
    echo "Refreshing every 30s (Ctrl+C to stop)"
    watch -n 30 "tail -1 $TRIAL/progress.csv | cut -d',' -f1-3"
    ;;

  watch-plots)
    mkdir -p "$PLOT_DIR"
    echo "Refreshing plots every 60s → $PLOT_DIR (Ctrl+C to stop)"
    watch -n 60 "python scripts/plot_training_progress.py \"$TRIAL\" --outdir \"$PLOT_DIR\" >/dev/null 2>&1"
    ;;

  midterm)
    N="${NUM_ROLLOUTS:-5}"
    mkdir -p "$MID_DIR"
    python scripts/evaluate_agent.py "$CKPT" \
      --outdir "$MID_DIR" \
      --eval-protocol inherit \
      --compare-baselines \
      --num-rollouts "$N" \
      --max-path-length 78
    python scripts/diagnose_tracking.py \
      --eval-dir "$MID_DIR" \
      --outdir "$MID_DIR/diagnostics" \
      --trial-dir "$TRIAL" \
      --progress-csv "$TRIAL/progress.csv" \
      --verify-env
    python scripts/verify_movement_cost_fairness.py \
      --config-path "$CONFIG_PATH" \
      --eval-dir "$MID_DIR"
    echo "Eval:  $MID_DIR"
    echo "Diag:  $MID_DIR/diagnostics/tracking_diagnosis.txt"
    ;;

  final)
    N="${NUM_ROLLOUTS:-10}"
    mkdir -p "$FINAL_DIR"
    python scripts/evaluate_agent.py "$CKPT" \
      --outdir "$FINAL_DIR" \
      --eval-protocol inherit \
      --compare-baselines \
      --num-rollouts "$N" \
      --max-path-length 78
    python scripts/diagnose_tracking.py \
      --eval-dir "$FINAL_DIR" \
      --outdir "$FINAL_DIR/diagnostics" \
      --trial-dir "$TRIAL" \
      --progress-csv "$TRIAL/progress.csv" \
      --verify-env \
      --gate
    python scripts/verify_movement_cost_fairness.py \
      --config-path "$CONFIG_PATH" \
      --eval-dir "$FINAL_DIR"
    python scripts/plot_training_progress.py "$TRIAL" --outdir "$PLOT_DIR" \
      -m evaluation/return-average policy/shifts-mean model/val_loss alpha
    echo "Eval:  $FINAL_DIR"
    echo "Diag:  $FINAL_DIR/diagnostics/tracking_diagnosis.txt"
    echo "Plots: $PLOT_DIR"
    ;;

  *)
    echo "Unknown mode: $MODE" >&2
    exit 1
    ;;
esac
