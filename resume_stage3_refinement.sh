#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ecer/PVRL/mbpo"
ARTIFACT_DIR="$ROOT/sequential_stage_artifacts"
RAY_ROOT="${RAY_ROOT:-$HOME/ray_mbpo/PVTracking/pv_tracking}"
CONDA_SH="${CONDA_SH:-/home/ecer/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mbpo}"

CPUS="${CPUS:-4}"
TRIAL_CPUS="${TRIAL_CPUS:-2}"
GPUS="${GPUS:-0}"
TRIAL_GPUS="${TRIAL_GPUS:-0}"

# Extend the restored Stage 3 run by this many additional epochs.
EXTRA_EPOCHS="${EXTRA_EPOCHS:-100}"
VALIDATION_DATES="2020-02-15,2020-05-15,2020-08-15,2020-11-15"
FINAL_TEST_DATES="2020-01-15,2020-03-20,2020-06-21,2020-09-22,2020-10-15,2020-12-21"

# Optional input:
#   - omitted: prefer sequential_stage_artifacts/stage3_trial_dir.txt, else latest trial
#   - trial dir: /home/.../seed:1234_...
#   - checkpoint dir: .../latest_checkpoint or .../checkpoint_500 or .../best_eval_checkpoint
SOURCE_INPUT="${1:-}"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Missing conda init script: $CONDA_SH" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
cd "$ROOT"

if ! [[ "$EXTRA_EPOCHS" =~ ^[0-9]+$ ]] || [[ "$EXTRA_EPOCHS" -le 0 ]]; then
  echo "EXTRA_EPOCHS must be a positive integer, got: $EXTRA_EPOCHS" >&2
  exit 1
fi

SOURCE_FILE="$ARTIFACT_DIR/stage3_trial_dir.txt"
if [[ -z "$SOURCE_INPUT" && -f "$SOURCE_FILE" ]]; then
  SOURCE_INPUT="$(tr -d '\r\n' < "$SOURCE_FILE")"
fi
if [[ -z "$SOURCE_INPUT" ]]; then
  SOURCE_INPUT="latest"
fi

INSPECT_JSON="$(
python - "$ROOT" "$RAY_ROOT" "$SOURCE_INPUT" "$VALIDATION_DATES" "$FINAL_TEST_DATES" <<'PY'
import csv
import json
import os
import pickle
import re
import sys

repo_root, ray_root, source_input, validation_dates, final_test_dates = sys.argv[1:]
sys.path.insert(0, repo_root)

from scripts.pv_trial_paths import resolve_trial_dir

def _is_checkpoint_like(path):
    name = os.path.basename(path.rstrip('/'))
    return (
        name.startswith('checkpoint_')
        or name in ('latest_checkpoint', 'best_eval_checkpoint')
    )

def _choose_restore_checkpoint(trial_dir, explicit_input=None):
    if explicit_input:
        explicit = os.path.expanduser(explicit_input.rstrip('/'))
        if os.path.isdir(explicit) and _is_checkpoint_like(explicit):
            return explicit

    latest = os.path.join(trial_dir, 'latest_checkpoint')
    if os.path.isdir(latest):
        return latest

    numbered = []
    for name in os.listdir(trial_dir):
        full = os.path.join(trial_dir, name)
        m = re.match(r'checkpoint_(\d+)$', name)
        if m and os.path.isdir(full):
            numbered.append((int(m.group(1)), full))
    if numbered:
        numbered.sort()
        return numbered[-1][1]

    best = os.path.join(trial_dir, 'best_eval_checkpoint')
    if os.path.isdir(best):
        return best

    raise FileNotFoundError('No resume checkpoint found under %s' % trial_dir)

def _load_restore_epoch(checkpoint_dir):
    checkpoint_pkl = os.path.join(checkpoint_dir, 'checkpoint.pkl')
    if not os.path.isfile(checkpoint_pkl):
        raise FileNotFoundError('Missing checkpoint.pkl in %s' % checkpoint_dir)
    with open(checkpoint_pkl, 'rb') as f:
        picklable = pickle.load(f)
    algorithm = picklable['algorithm']
    state = algorithm.__getstate__()
    return int(state.get('_epoch', 0))

def _last_progress_epoch(trial_dir):
    progress_csv = os.path.join(trial_dir, 'progress.csv')
    if not os.path.isfile(progress_csv):
        return None
    last = None
    with open(progress_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get('epoch')
            if raw in (None, ''):
                continue
            try:
                last = int(float(raw))
            except Exception:
                pass
    return last

explicit_checkpoint = None
if source_input not in ('', 'latest'):
    expanded = os.path.expanduser(source_input.rstrip('/'))
    if os.path.isdir(expanded) and _is_checkpoint_like(expanded):
        explicit_checkpoint = expanded
        trial_dir = os.path.dirname(expanded)
    else:
        trial_dir = resolve_trial_dir(source_input, root=ray_root)
else:
    trial_dir = resolve_trial_dir(source_input, root=ray_root)

restore_ckpt = _choose_restore_checkpoint(trial_dir, explicit_checkpoint)
params_path = os.path.join(trial_dir, 'params.json')
if not os.path.isfile(params_path):
    raise FileNotFoundError('Missing params.json in %s' % trial_dir)

with open(params_path, 'r', encoding='utf-8') as f:
    variant = json.load(f)

config_version = str(variant.get('config_version', ''))
env = variant['environment_params']['training']['kwargs']
eval_env = variant['environment_params']['evaluation']['kwargs']
algo = variant['algorithm_params']['kwargs']
expected_excluded = [d for d in (validation_dates.split(',') + final_test_dates.split(',')) if d]
expected_validation = [d for d in validation_dates.split(',') if d]

if 'stage3' not in config_version.lower():
    raise SystemExit('Refinement requires a Stage 3 trial, got config_version=%r' % config_version)
if env.get('start_date') != '2020-01-01' or env.get('end_date') != '2020-12-31':
    raise SystemExit('Refinement requires full-year Stage 3 dates, got %r -> %r'
                     % (env.get('start_date'), env.get('end_date')))
if env.get('weather_source') != 'historical':
    raise SystemExit('Refinement requires weather_source=historical, got %r' % env.get('weather_source'))
if env.get('randomize_day') is not True:
    raise SystemExit('Refinement requires randomize_day=True, got %r' % env.get('randomize_day'))
if env.get('observation_mode') != 'physical':
    raise SystemExit('Refinement requires observation_mode=physical, got %r' % env.get('observation_mode'))
if env.get('movement_penalty') != 0.0:
    raise SystemExit('Refinement requires movement_penalty=0.0, got %r' % env.get('movement_penalty'))
if list(env.get('excluded_dates') or []) != expected_excluded:
    raise SystemExit('Refinement requires clean-split excluded_dates=%r, got %r'
                     % (expected_excluded, env.get('excluded_dates')))
if list(eval_env.get('fixed_eval_dates') or []) != expected_validation:
    raise SystemExit('Refinement requires validation fixed_eval_dates=%r, got %r'
                     % (expected_validation, eval_env.get('fixed_eval_dates')))

restore_epoch = _load_restore_epoch(restore_ckpt)
progress_epoch = _last_progress_epoch(trial_dir)

payload = {
    'trial_dir': trial_dir,
    'restore_checkpoint': restore_ckpt,
    'restore_epoch': restore_epoch,
    'progress_epoch_last': progress_epoch,
    'config_version': config_version,
    'base_n_epochs': int(algo.get('n_epochs', 0)),
    'real_ratio': algo.get('real_ratio'),
}
print(json.dumps(payload))
PY
)"

TRIAL_DIR="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["trial_dir"])' "$INSPECT_JSON")"
RESTORE_CKPT="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["restore_checkpoint"])' "$INSPECT_JSON")"
RESTORE_EPOCH="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["restore_epoch"])' "$INSPECT_JSON")"
PROGRESS_EPOCH_LAST="$(python -c 'import json,sys; v=json.loads(sys.argv[1])["progress_epoch_last"]; print("" if v is None else v)' "$INSPECT_JSON")"
BASE_CONFIG_VERSION="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["config_version"])' "$INSPECT_JSON")"
BASE_N_EPOCHS="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["base_n_epochs"])' "$INSPECT_JSON")"
BASE_REAL_RATIO="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["real_ratio"])' "$INSPECT_JSON")"

TARGET_TOTAL_EPOCHS=$((RESTORE_EPOCH + EXTRA_EPOCHS))
GENERATED_CONFIG="$ARTIFACT_DIR/stage3_refinement_resume_tmp.py"
GENERATED_MODULE="stage3_refinement_resume_tmp"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"

cat > "$GENERATED_CONFIG" <<PY
"""Generated by resume_stage3_refinement.sh.

Same Stage 3 full-year historical-weather training contract, but with a larger
total epoch budget so a restored checkpoint can continue optimizing.
"""

import importlib

_base = importlib.import_module('examples.config.pv_tracking.stage3_fullyear_random_clean_split')

CONFIG_VERSION = 'pv_tracking_stage3_refine_to_${TARGET_TOTAL_EPOCHS}_${STAMP}'
TRAINING_STAGE = 'stage3'

params = dict(_base.params)
params['config_version'] = CONFIG_VERSION
params['kwargs'] = dict(_base.params['kwargs'])
params['kwargs'].update({
    'n_epochs': ${TARGET_TOTAL_EPOCHS},
})
params['environment_kwargs'] = dict(_base.params['environment_kwargs'])
params['evaluation_environment_kwargs'] = dict(_base.params['evaluation_environment_kwargs'])
PY

echo "Stage 3 refinement resume"
echo "  source trial:         $TRIAL_DIR"
echo "  restore checkpoint:   $RESTORE_CKPT"
echo "  restore epoch:        $RESTORE_EPOCH"
echo "  progress.csv last:    ${PROGRESS_EPOCH_LAST:-n/a}"
echo "  previous n_epochs:    $BASE_N_EPOCHS"
echo "  extra epochs:         $EXTRA_EPOCHS"
echo "  new total n_epochs:   $TARGET_TOTAL_EPOCHS"
echo "  real_ratio:           $BASE_REAL_RATIO"
echo "  base config_version:  $BASE_CONFIG_VERSION"
echo
echo "Scientific note:"
echo "  This is reasonable only because it resumes the SAME Stage 3 MDP"
echo "  (full year, historical weather, physical observations, same reward/action)."
echo "  It continues from a real Stage 3 checkpoint instead of trying to inject"
echo "  raw policy weights across mismatched stages."
echo
echo "Operational note:"
echo "  Resume uses latest_checkpoint when available. Because save_every_epochs=5,"
echo "  stopping mid-run may lose up to ~5 epochs since the last saved checkpoint."
echo

PYTHONPATH="$ARTIFACT_DIR:$ROOT:${PYTHONPATH:-}" \
python scripts/verify_training_config.py --config "$GENERATED_MODULE" --file-only

echo
echo "Starting resumed Stage 3 refinement run..."
echo

PYTHONPATH="$ARTIFACT_DIR:$ROOT:${PYTHONPATH:-}" \
mbpo run_local examples.development \
  --config="$GENERATED_MODULE" \
  --restore="$RESTORE_CKPT" \
  --gpus="$GPUS" \
  --trial-gpus="$TRIAL_GPUS" \
  --cpus="$CPUS" \
  --trial-cpus="$TRIAL_CPUS"

echo
echo "Refinement run finished."
echo "To select the scientific/reporting checkpoint afterwards, run:"
echo "  cd $ROOT"
echo "  ./run_stage3_posttrain.sh latest"
