#!/usr/bin/env bash
# Results for a named run: ./result.sh <run_name> [--plot-only | --eval-only | --full | --status]
#
# All NSRDB configs: evaluate_fullyear_mc.py --date-set nsrdb_multiyear — scenario MC
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mbpo}"

usage() {
  echo "Usage: $0 <run_name> [--status | --plot-only | --eval-only | --full]"
  echo "  NSRDB runs (stage3_nsrdb, conf1–conf5): eval uses --date-set nsrdb_multiyear automatically"
  exit 1
}

[[ $# -ge 1 ]] || usage
RUN_NAME="$1"
shift
MODE="full"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-64}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --status) MODE="status"; shift ;;
    --plot-only) MODE="plot"; shift ;;
    --eval-only) MODE="eval"; shift ;;
    --full) MODE="full"; shift ;;
    --num-rollouts) NUM_ROLLOUTS="$2"; shift 2 ;;
    *) echo "Unknown: $1"; usage ;;
  esac
done

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
cd "$ROOT"

RUN_DIR="$ROOT/runs/$RUN_NAME"
RESULTS="$RUN_DIR/results"
TRAIN_PLOTS="$RESULTS/training"
EVAL_OUT="$RESULTS/evaluation"

read_vars() {
  eval "$(python -c "
import sys, json, os
sys.path.insert(0, '$ROOT')
from scripts.run_registry import (
    read_run_meta, resolve_trial_for_run, resolve_checkpoint, ray_trial_root)
meta = read_run_meta('$RUN_NAME') or {}
trial, warn = resolve_trial_for_run('$RUN_NAME', meta.get('conf'))
if not trial:
    raise SystemExit('No trial for run $RUN_NAME. Run: ./train.sh $RUN_NAME conf1')
ckpt = resolve_checkpoint(trial)
conf = meta.get('conf', '')
print('TRIAL=%s' % trial)
print('CKPT=%s' % ckpt)
print('CONF=%s' % conf)
from examples.config.pv_tracking.conf_registry import is_nsrdb_conf
is_nsrdb = 1 if is_nsrdb_conf(conf) else 0
print('IS_NSRDB=%s' % is_nsrdb)
if warn:
    import sys as _s
    print('WARN=%s' % warn, file=_s.stderr)
")"
}

if [[ "$MODE" == "status" ]]; then
  python -c "
import sys, os, json
sys.path.insert(0, '$ROOT')
from scripts.run_registry import read_run_meta, read_trial_pointer, ray_trial_root, list_trial_dirs
meta = read_run_meta('$RUN_NAME')
print('Run:      $RUN_NAME')
print('Dir:      $RUN_DIR')
if meta:
    print('Conf:     %s' % meta.get('conf'))
    print('Module:   %s' % meta.get('config_module'))
ptr = read_trial_pointer('$RUN_NAME')
print('Trial ptr:%s' % (ptr or '(none)'))
root = ray_trial_root('$RUN_NAME')
for t in list_trial_dirs(root, 5):
    print('  ', t)
"
  exit 0
fi

read_vars
export TRIAL CKPT CONF IS_NSRDB
mkdir -p "$TRAIN_PLOTS" "$EVAL_OUT"

do_plot() {
  echo "=== Training plots → $TRAIN_PLOTS ==="
  python scripts/plot_training_progress.py "$TRIAL" --outdir "$TRAIN_PLOTS" \
    --metrics evaluation/return-average training/return-average model/val_loss \
    model_rollout_length real_batch_ratio alpha Q_loss policy/shifts-mean
  cp -f "$TRIAL/progress.csv" "$RESULTS/progress.csv" 2>/dev/null || true
}

do_eval_tmy() {
  echo "=== TMY eval → $EVAL_OUT (checkpoint: $CKPT) ==="
  python scripts/evaluate_agent.py "$CKPT" \
    --outdir "$EVAL_OUT" \
    --eval-protocol inherit \
    --max-path-length 117 \
    --compare-baselines \
    --num-rollouts "$NUM_ROLLOUTS" \
    --max-rollout-plots 4 \
    --eval-seed-base 100000
}

do_eval_nsrdb() {
  echo "=== NSRDB scenario MC eval → $EVAL_OUT (checkpoint: $CKPT) ==="
  echo "    Protocol: e~p(e) per episode; fixed NSRDB trajectory; native 5min control"
  rm -rf "$EVAL_OUT"
  mkdir -p "$EVAL_OUT"
  python scripts/evaluate_fullyear_mc.py "$CKPT" \
    --outdir "$EVAL_OUT" \
    --date-set nsrdb_multiyear \
    --num-rollouts "$NUM_ROLLOUTS" \
    --eval-seed-base 100000 \
    --max-path-length 117 \
    --max-rollout-plots 4 \
    --eval-protocol inherit \
    --policy-mode deterministic
}

do_eval() {
  rm -rf "$EVAL_OUT"
  mkdir -p "$EVAL_OUT"
  if [[ "${IS_NSRDB:-0}" -eq 1 ]]; then
    do_eval_nsrdb
  else
    do_eval_tmy
  fi
  python scripts/diagnose_tracking.py \
    --eval-dir "$EVAL_OUT" \
    --outdir "$EVAL_OUT/diagnostics" \
    --trial-dir "$TRIAL" \
    --progress-csv "$TRIAL/progress.csv" \
    --max-paired-plot-days 6
  if [[ "${IS_NSRDB:-0}" -eq 1 ]]; then
    echo "=== NSRDB state-space / baseline MC check ==="
    python scripts/verify_pv_state_space.py \
      --mode nsrdb \
      --nsrdb-mc-rollouts 8 \
      --outdir "$EVAL_OUT/state_space_check"
  fi
}

case "$MODE" in
  plot) do_plot ;;
  eval) do_eval ;;
  full) do_plot; do_eval ;;
esac

echo ""
echo "Results folder: $RESULTS"
echo "  training/     learning curves (E[R] ± σ from progress.csv)"
if [[ "${IS_NSRDB:-0}" -eq 1 ]]; then
  echo "  evaluation/   NSRDB scenario MC: PAIRED_MC_COMPARISON.txt, paper_figures/, rollout_plots/, mc_records.json"
else
  echo "  evaluation/   TMY annual MC: PAIRED_MC_COMPARISON.txt, paper_figures/, evaluation_summary.txt"
fi
ls -la "$RESULTS" 2>/dev/null || true
