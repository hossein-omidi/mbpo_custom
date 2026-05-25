#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ecer/PVRL/mbpo"
RAY_ROOT="${RAY_ROOT:-$HOME/ray_mbpo/PVTracking/pv_tracking}"
CONDA_SH="${CONDA_SH:-/home/ecer/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mbpo}"

STAGE3_FIXED_DATES="2020-01-15,2020-03-20,2020-06-21,2020-08-01,2020-10-15,2020-12-21"
MAX_PATH_LENGTH="${MAX_PATH_LENGTH:-39}"

# Strict, balanced replicate count: 24 = 4 seeds per fixed date.
# With six fixed dates, using a multiple of 6 avoids overweighting any date.
SELECT_ROLLOUTS="${SELECT_ROLLOUTS:-24}"
FINAL_EVAL_ROLLOUTS="${FINAL_EVAL_ROLLOUTS:-24}"

OUTDIR="$ROOT/evaluation/pv_stage3_fullyear_random"
TRIAL_INPUT="${1:-latest}"

resolve_trial_dir() {
  local trial_input="$1"

  if [[ "$trial_input" == "latest" ]]; then
    local latest
    latest="$(ls -td "$RAY_ROOT"/seed:*/ 2>/dev/null | head -1 || true)"
    if [[ -z "$latest" ]]; then
      echo "No Stage 3 Ray trial found under $RAY_ROOT" >&2
      exit 1
    fi
    echo "${latest%/}"
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

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Missing conda init script: $CONDA_SH" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

cd "$ROOT"

TRIAL="$(resolve_trial_dir "$TRIAL_INPUT")"
if [[ ! -d "$TRIAL" ]]; then
  echo "Trial directory does not exist: $TRIAL" >&2
  exit 1
fi

if (( SELECT_ROLLOUTS % 6 != 0 )); then
  echo "SELECT_ROLLOUTS must be a multiple of 6 for balanced Stage 3 fixed-date coverage." >&2
  exit 1
fi
if (( FINAL_EVAL_ROLLOUTS % 6 != 0 )); then
  echo "FINAL_EVAL_ROLLOUTS must be a multiple of 6 for balanced Stage 3 fixed-date coverage." >&2
  exit 1
fi

python - "$TRIAL" <<'PY'
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

assert 'stage3' in config_version.lower(), 'Not a Stage 3 trial: %s' % config_version
assert env.get('observation_mode') == 'physical', env.get('observation_mode')
assert env.get('weather_source') == 'random', env.get('weather_source')
assert env.get('randomize_day') is True, env.get('randomize_day')
assert env.get('start_date') == '2020-01-01', env.get('start_date')
assert env.get('end_date') == '2020-12-31', env.get('end_date')
assert eval_env.get('weather_source') == 'random', eval_env.get('weather_source')
assert eval_env.get('randomize_day') is False, eval_env.get('randomize_day')
assert eval_env.get('fixed_eval_dates') == [
    '2020-01-15',
    '2020-03-20',
    '2020-06-21',
    '2020-08-01',
    '2020-10-15',
    '2020-12-21',
], eval_env.get('fixed_eval_dates')

print('Verified Stage 3 trial contract:')
print('  trial:', trial)
print('  config_version:', config_version)
print('  train date range:', env.get('start_date'), '->', env.get('end_date'))
print('  eval fixed dates:', ','.join(eval_env.get('fixed_eval_dates', [])))
print('  observation_mode:', env.get('observation_mode'))
print('  weather_source:', env.get('weather_source'))
PY

backup_existing_outdir "$OUTDIR"
mkdir -p "$OUTDIR"

SELECT_LOG="$OUTDIR/checkpoint_ranking.txt"
python scripts/select_best_checkpoint.py "$TRIAL" \
  --fixed-eval-dates "$STAGE3_FIXED_DATES" \
  --compare-baselines \
  --num-rollouts "$SELECT_ROLLOUTS" \
  --max-path-length "$MAX_PATH_LENGTH" \
  --eval-protocol inherit | tee "$SELECT_LOG"

BEST_CKPT="$(
python - "$SELECT_LOG" <<'PY'
import re
import sys

text = open(sys.argv[1], 'r', encoding='utf-8').read()
matches = re.findall(r'Recommended for reporting:\s*(.+)', text)
if not matches:
    raise SystemExit('Could not parse recommended checkpoint from ranking log.')
print(matches[-1].strip())
PY
)"

if [[ ! -d "$BEST_CKPT" ]]; then
  echo "Recommended checkpoint does not exist: $BEST_CKPT" >&2
  exit 1
fi

printf '%s\n' "$TRIAL" > "$OUTDIR/trial_dir.txt"
printf '%s\n' "$BEST_CKPT" > "$OUTDIR/recommended_checkpoint.txt"

python scripts/evaluate_agent.py "$BEST_CKPT" \
  --outdir "$OUTDIR" \
  --num-rollouts "$FINAL_EVAL_ROLLOUTS" \
  --max-path-length "$MAX_PATH_LENGTH" \
  --deterministic \
  --compare-baselines \
  --fixed-eval-dates "$STAGE3_FIXED_DATES" \
  --eval-weather-source random \
  --eval-protocol inherit

python scripts/diagnose_tracking.py \
  --eval-dir "$OUTDIR" \
  --outdir "$OUTDIR/diagnostics" \
  --progress-csv "$TRIAL/progress.csv" \
  --trial-dir "$TRIAL" \
  --gate

echo
echo "Stage 3 post-train protocol completed."
echo "Trial:                 $TRIAL"
echo "Recommended checkpoint: $BEST_CKPT"
echo "Evaluation output:      $OUTDIR"
