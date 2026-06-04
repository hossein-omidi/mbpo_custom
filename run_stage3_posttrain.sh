#!/usr/bin/env bash
# Stage 3 post-train: frozen-policy RL evaluation on full annual day support.
#
# RL protocol (see docs/RL_EVAL_PROTOCOL.md):
#   - All calendar days remain in environment support (no excluded_dates).
#   - Evaluation independence uses independent stochastic seeds (eval_seed_base + i),
#     not calendar-day hold-out.
#   - Policy is frozen: evaluate_agent / evaluate_fullyear_mc only (no SAC/BNN/replay updates).
#   - Real PVTrackingEnv only (not the learned dynamics model).
#
# Fixed dates are used only in optional stress diagnostics (mode=stress), not as the main RL test set.
#
# Usage:
#   ./run_stage3_posttrain.sh [trial_dir|latest] [mode]
#
# Modes:
#   eval   — default: best_eval_checkpoint + seed-based annual rollouts
#   rank   — re-rank checkpoints with seed-based rollouts, then eval winner
#   mc     — annual Monte Carlo season summary (evaluate_fullyear_mc)
#   stress — optional fixed-date diagnostic panel (STAGE3_STRESS_TEST_DATES)
#   all    — eval + mc (+ stress if RUN_STRESS=1)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT="$ROOT/sequential_stage_artifacts/stage3_fullyear_trial_dir.txt"
RAY_ROOT="${RAY_ROOT:-$HOME/ray_mbpo/PVTracking/pv_tracking}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mbpo}"

# Main RL evaluation: full-year support, independent seeds (not fixed calendar dates).
EVAL_NUM_ROLLOUTS="${EVAL_NUM_ROLLOUTS:-16}"
EVAL_SEED_BASE="${EVAL_SEED_BASE:-100000}"
MAX_PATH_LENGTH="${MAX_PATH_LENGTH:-78}"
RANK_ROLLOUTS="${RANK_ROLLOUTS:-24}"
RUN_MC="${RUN_MC:-0}"
RUN_STRESS="${RUN_STRESS:-0}"
RUN_GATE="${RUN_GATE:-0}"

OUTDIR="$ROOT/evaluation/pv_stage3_posttrain"
MC_OUTDIR="$ROOT/evaluation/pv_stage3_posttrain_mc"
STRESS_OUTDIR="$ROOT/evaluation/pv_stage3_posttrain_stress"

TRIAL_INPUT="${1:-latest}"
MODE="${2:-eval}"

resolve_trial_dir() {
  local trial_input="$1"
  if [[ "$trial_input" == "latest" ]]; then
    python - "$ROOT" "$ARTIFACT" <<'PY'
import os, sys
repo = sys.argv[1]
artifact = sys.argv[2]
sys.path.insert(0, os.path.join(repo, 'scripts'))
from pv_trial_paths import find_stage3_trial
trial, _ = find_stage3_trial(artifact_path=artifact)
if not trial:
    raise SystemExit('No Stage 3 trial found. Run train first.')
print(trial)
PY
    return 0
  fi
  echo "${trial_input%/}"
}

backup_existing_outdir() {
  local outdir="$1"
  if [[ -d "$outdir" ]]; then
    mv "$outdir" "${outdir}_backup_$(date +%Y%m%d_%H%M%S)"
  fi
}

# Preflight: default workflow must not pass calendar hold-out flags to main eval.
preflight_posttrain_protocol() {
  local script="$ROOT/run_stage3_posttrain.sh"
  if grep -qE 'evaluate_agent\.py.*--fixed-eval-dates' "$script" \
     && ! grep -q 'run_stress_diagnostics' "$script"; then
    echo "Internal error: main evaluate_agent must not use --fixed-eval-dates" >&2
    exit 1
  fi
  if ! grep -qE 'EVAL_SEED_BASE|eval-seed-base' "$script"; then
    echo "Internal error: posttrain script must set eval seed base" >&2
    exit 1
  fi
}

verify_stage3_trial_rl_contract() {
  local trial="$1"
  python - "$trial" <<'PY'
import json
import os
import sys

trial = sys.argv[1]
params_path = os.path.join(trial, 'params.json')
if not os.path.exists(params_path):
    raise SystemExit('Missing params.json: %s' % params_path)

variant = json.load(open(params_path))
config_version = str(variant.get('config_version', ''))
env = variant['environment_params']['training']['kwargs']
eval_env = variant['environment_params']['evaluation']['kwargs']

if 'stage3' not in config_version.lower():
    raise SystemExit('Not a Stage 3 trial: %s' % config_version)
if env.get('observation_mode') != 'physical':
    raise SystemExit('observation_mode must be physical: %r' % env.get('observation_mode'))
if env.get('weather_source') != 'historical':
    raise SystemExit('weather_source must be historical: %r' % env.get('weather_source'))
if not env.get('randomize_day'):
    raise SystemExit('training randomize_day must be True for annual RL protocol')
if env.get('start_date') != '2020-01-01' or env.get('end_date') != '2020-12-31':
    raise SystemExit('expected full-year 2020-01-01..2020-12-31, got %s..%s' % (
        env.get('start_date'), env.get('end_date')))
if env.get('excluded_dates'):
    raise SystemExit('RL protocol forbids excluded_dates in training kwargs: %s' % (
        env.get('excluded_dates')))
if eval_env.get('fixed_eval_dates'):
    raise SystemExit('RL protocol forbids fixed_eval_dates in evaluation kwargs: %s' % (
        eval_env.get('fixed_eval_dates')))
if not eval_env.get('randomize_day', True):
    raise SystemExit('evaluation randomize_day should be True for seed-based annual eval')

print('Verified Stage 3 RL trial contract:')
print('  trial:', trial)
print('  config_version:', config_version)
print('  train dates:', env.get('start_date'), '->', env.get('end_date'))
print('  excluded_dates: none')
print('  fixed_eval_dates: none')
print('  irradiance_perturbation_std:', env.get('irradiance_perturbation_std'))
PY
}

resolve_best_checkpoint() {
  local trial="$1"
  if [[ -d "$trial/best_eval_checkpoint" ]]; then
    echo "$trial/best_eval_checkpoint"
    return 0
  fi
  echo "$trial/latest_checkpoint"
}

run_seed_based_eval() {
  local ckpt="$1"
  local outdir="$2"
  python scripts/evaluate_agent.py "$ckpt" \
    --outdir "$outdir" \
    --num-rollouts "$EVAL_NUM_ROLLOUTS" \
    --eval-seed-base "$EVAL_SEED_BASE" \
    --max-path-length "$MAX_PATH_LENGTH" \
    --max-rollout-plots "${MAX_ROLLOUT_PLOTS:-4}" \
    --deterministic \
    --compare-baselines \
    --eval-protocol inherit
}

run_rank_checkpoints() {
  local trial="$1"
  local log="$2"
  python scripts/select_best_checkpoint.py "$trial" \
    --compare-baselines \
    --num-rollouts "$RANK_ROLLOUTS" \
    --eval-seed-base "$EVAL_SEED_BASE" \
    --max-path-length "$MAX_PATH_LENGTH" \
    --eval-protocol inherit | tee "$log"
}

parse_recommended_checkpoint() {
  local log="$1"
  python - "$log" <<'PY'
import re
import sys

text = open(sys.argv[1], 'r', encoding='utf-8').read()
matches = re.findall(r'Recommended for reporting:\s*(.+)', text)
if not matches:
    raise SystemExit('Could not parse recommended checkpoint from ranking log.')
print(matches[-1].strip())
PY
}

run_mc_eval() {
  local ckpt="$1"
  local outdir="$2"
  python scripts/evaluate_fullyear_mc.py "$ckpt" \
    --outdir "$outdir" \
    --eval-protocol inherit \
    --date-set annual \
    --num-rollouts "${MC_NUM_ROLLOUTS:-$EVAL_NUM_ROLLOUTS}" \
    --eval-seed-base "$EVAL_SEED_BASE" \
    --error-bars sem
}

# Optional diagnostic only — fixed calendar stress panel, not main RL evaluation.
run_stress_diagnostics() {
  local ckpt="$1"
  local outdir="$2"
  echo "[posttrain] stress diagnostics: fixed dates only (not main RL test set)"
  python scripts/evaluate_fullyear_mc.py "$ckpt" \
    --outdir "$outdir" \
    --eval-protocol inherit \
    --date-set stress_test \
    --replicates-per-date "${STRESS_REPLICATES:-8}" \
    --eval-seed-base "$EVAL_SEED_BASE" \
    --error-bars sem
}

run_diagnostics() {
  local trial="$1"
  local eval_dir="$2"
  local extra=()
  if [[ "$RUN_GATE" == "1" ]]; then
    extra+=(--gate)
  else
    extra+=(--verify-env)
  fi
  python scripts/diagnose_tracking.py \
    --eval-dir "$eval_dir" \
    --outdir "$eval_dir/diagnostics" \
    --progress-csv "$trial/progress.csv" \
    --trial-dir "$trial" \
    "${extra[@]}"
}

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Missing conda init script: $CONDA_SH" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
cd "$ROOT"
preflight_posttrain_protocol

TRIAL="$(resolve_trial_dir "$TRIAL_INPUT")"
if [[ ! -d "$TRIAL" ]]; then
  echo "Trial directory does not exist: $TRIAL" >&2
  exit 1
fi

verify_stage3_trial_rl_contract "$TRIAL"

case "$MODE" in
  eval|rank|all)
    if [[ "$MODE" == "rank" ]]; then
      backup_existing_outdir "$OUTDIR"
      mkdir -p "$OUTDIR"
      SELECT_LOG="$OUTDIR/checkpoint_ranking.txt"
      echo "[posttrain] ranking checkpoints (seed-based, annual support)"
      run_rank_checkpoints "$TRIAL" "$SELECT_LOG"
      BEST_CKPT="$(parse_recommended_checkpoint "$SELECT_LOG")"
    else
      backup_existing_outdir "$OUTDIR"
      mkdir -p "$OUTDIR"
      BEST_CKPT="$(resolve_best_checkpoint "$TRIAL")"
      if [[ "${RANK_BEFORE_EVAL:-0}" == "1" ]]; then
        SELECT_LOG="$OUTDIR/checkpoint_ranking.txt"
        run_rank_checkpoints "$TRIAL" "$SELECT_LOG"
        BEST_CKPT="$(parse_recommended_checkpoint "$SELECT_LOG")"
      fi
    fi
    if [[ ! -d "$BEST_CKPT" ]]; then
      echo "Checkpoint does not exist: $BEST_CKPT" >&2
      exit 1
    fi
    printf '%s\n' "$TRIAL" > "$OUTDIR/trial_dir.txt"
    printf '%s\n' "$BEST_CKPT" > "$OUTDIR/recommended_checkpoint.txt"
    echo "[posttrain] frozen-policy eval: rollouts=$EVAL_NUM_ROLLOUTS seed_base=$EVAL_SEED_BASE"
    run_seed_based_eval "$BEST_CKPT" "$OUTDIR"
    run_diagnostics "$TRIAL" "$OUTDIR"
    if [[ "$MODE" == "all" ]]; then
      if [[ "$RUN_MC" == "1" ]]; then
        backup_existing_outdir "$MC_OUTDIR"
        mkdir -p "$MC_OUTDIR"
        run_mc_eval "$BEST_CKPT" "$MC_OUTDIR"
      fi
      if [[ "$RUN_STRESS" == "1" ]]; then
        backup_existing_outdir "$STRESS_OUTDIR"
        mkdir -p "$STRESS_OUTDIR"
        run_stress_diagnostics "$BEST_CKPT" "$STRESS_OUTDIR"
      fi
    fi
    ;;
  mc)
    BEST_CKPT="$(resolve_best_checkpoint "$TRIAL")"
    backup_existing_outdir "$MC_OUTDIR"
    mkdir -p "$MC_OUTDIR"
    run_mc_eval "$BEST_CKPT" "$MC_OUTDIR"
    ;;
  stress)
    BEST_CKPT="$(resolve_best_checkpoint "$TRIAL")"
    backup_existing_outdir "$STRESS_OUTDIR"
    mkdir -p "$STRESS_OUTDIR"
    run_stress_diagnostics "$BEST_CKPT" "$STRESS_OUTDIR"
    ;;
  *)
    echo "Usage: $0 [trial_dir|latest] {eval|rank|mc|stress|all}" >&2
    echo "  eval   — seed-based annual evaluation (default)" >&2
    echo "  rank   — re-rank checkpoints with seeds, then eval" >&2
    echo "  mc     — annual MC season plots" >&2
    echo "  stress — optional fixed-date diagnostics only" >&2
    echo "  all    — eval; add RUN_MC=1 and/or RUN_STRESS=1 for extras" >&2
    exit 1
    ;;
esac

echo
echo "Stage 3 post-train RL evaluation completed (mode=$MODE)."
echo "Trial:      $TRIAL"
if [[ "$MODE" == "mc" ]]; then
  echo "Output:     $MC_OUTDIR"
elif [[ "$MODE" == "stress" ]]; then
  echo "Stress diag: $STRESS_OUTDIR (fixed dates — not main RL test set)"
else
  echo "Output:     $OUTDIR"
fi
if [[ "$MODE" == "all" && "$RUN_MC" == "1" ]]; then
  echo "MC output:  $MC_OUTDIR"
fi
if [[ "$MODE" == "all" && "$RUN_STRESS" == "1" ]]; then
  echo "Stress diag: $STRESS_OUTDIR (fixed dates — not main RL test set)"
fi
