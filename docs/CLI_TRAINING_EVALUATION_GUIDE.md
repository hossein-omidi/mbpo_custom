# PV Tracking CLI Training And Evaluation Guide

This file is the current A-to-Z CLI workflow for the clean Stage 3 experiment.

It is written for the current project state:

- pure RL only
- `movement_penalty=0.0`
- physical observations
- full-year random-weather Stage 3
- programmatic exclusion of validation and final-test dates from training sampling
- checkpoint selection on validation dates only
- final reporting on untouched final-test dates only

## 1. Canonical root directories

Use these paths consistently.

- Repo root: `/home/ecer/PVRL/mbpo`
- Ray training trials: `~/ray_mbpo/PVTracking/pv_tracking/seed:<id>_<timestamp>.../`
- Sequential-script artifacts: `/home/ecer/PVRL/mbpo/sequential_stage_artifacts/`
- Training plots: `/home/ecer/PVRL/mbpo/training_plots/`
- Smoke-test runs: `/home/ecer/PVRL/mbpo/smoke_runs/`
- Stage 3 validation eval output: `/home/ecer/PVRL/mbpo/evaluation/pv_stage3_validation_clean_split/`
- Stage 3 final test output: `/home/ecer/PVRL/mbpo/evaluation/pv_stage3_final_clean_test/`

Inside each Ray trial directory, the most important files are:

- `params.json`
- `progress.csv`
- `best_eval_checkpoint/`
- `latest_checkpoint/`
- `checkpoint_<epoch>/`

Inside the final Stage 3 evaluation directory, the most important files are:

- `evaluation_summary.json`
- `evaluation_summary.txt`
- `eval_scenario_confirmation.txt`
- `reward_time_analysis.txt`
- `recommended_checkpoint.txt`
- `checkpoint_ranking.txt`
- `rollouts/`
- `baseline_rollouts/`
- `advanced/`
- `diagnostics/tracking_diagnosis.txt`

## 2. Clean split used by the current Stage 3 workflow

Training config:

- module: `examples.config.pv_tracking.stage3_fullyear_random_clean_split`
- file: `examples/config/pv_tracking/stage3_fullyear_random_clean_split.py`

Validation dates used for checkpoint selection:

- `2020-02-15`
- `2020-05-15`
- `2020-08-15`
- `2020-11-15`

Untouched final-test dates used only for final reported comparison:

- `2020-01-15`
- `2020-03-20`
- `2020-06-21`
- `2020-09-22`
- `2020-10-15`
- `2020-12-21`

## 3. One-time shell setup

Run this first in every new shell:

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo
```

Optional helper variables:

```bash
export ROOT=/home/ecer/PVRL/mbpo
export RAY_ROOT="$HOME/ray_mbpo/PVTracking/pv_tracking"
```

## 4. Recommended scientific workflow

The best current workflow is:

1. run preflight checks
2. optionally run the short MBPO smoke test
3. train Stage 3 from scratch with the clean-split config
4. monitor `progress.csv` during training
5. use `run_stage3_posttrain.sh` to rank checkpoints on validation dates and then evaluate the selected checkpoint on untouched final-test dates
6. read the diagnosis in `evaluation/pv_stage3_final_clean_test/diagnostics/`

Do not restore an old replay pool for the final clean experiment.

## 5. Preflight commands

Run these before the long training job:

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

python scripts/verify_training_config.py \
  --config examples.config.pv_tracking.stage3_fullyear_random_clean_split

python scripts/validate_pv_rollouts.py \
  --config-path examples/config/pv_tracking/stage3_fullyear_random_clean_split.py

pytest tests/test_pv_tracking_audit.py -q
```

## 6. Optional smoke test before the long run

This short run is useful when you want to confirm MBPO synthetic rollouts and the new rollout-boundary filtering are active before full retraining.

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

mbpo run_local examples.development \
  --config=examples.config.pv_tracking.smoke_model_rollout \
  --gpus=0 \
  --trial-gpus=0 \
  --cpus=4 \
  --trial-cpus=2
```

Smoke-test artifacts are written under `/home/ecer/PVRL/mbpo/smoke_runs/`.

## 7. Stage 3 full training from scratch

This is the main clean experiment.

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

mbpo run_local examples.development \
  --config=examples.config.pv_tracking.stage3_fullyear_random_clean_split \
  --gpus=0 \
  --trial-gpus=0 \
  --cpus=4 \
  --trial-cpus=2
```

After the run starts, point `TRIAL` at the newest Ray folder:

```bash
export TRIAL=$(ls -td ~/ray_mbpo/PVTracking/pv_tracking/seed:*/ | head -1)
export TRIAL="${TRIAL%/}"
export CKPT="$TRIAL/best_eval_checkpoint"
```

## 8. Runtime sanity checks for the active trial

Check that the live trial really uses the clean-split contract:

```bash
python - "$TRIAL/params.json" <<'PY'
import json
import sys

v = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
train = v['environment_params']['training']['kwargs']
eval_env = v['environment_params']['evaluation']['kwargs']
algo = v['algorithm_params']['kwargs']

print('config_version:', v.get('config_version'))
print('n_epochs:', algo.get('n_epochs'))
print('real_ratio:', algo.get('real_ratio'))
print('movement_penalty:', train.get('movement_penalty'))
print('training excluded_dates:', train.get('excluded_dates'))
print('validation fixed_eval_dates:', eval_env.get('fixed_eval_dates'))
print('restore:', v.get('restore'))
PY
```

What you should see:

- `movement_penalty: 0.0`
- `training excluded_dates` containing all validation and final-test dates
- `validation fixed_eval_dates` containing only the 4 validation dates
- no intentional old replay-pool restore for the clean final experiment

## 9. Online training monitoring commands

### 9.1 Save progress plots

```bash
python scripts/plot_training_progress.py "$TRIAL" \
  --outdir training_plots/stage3_clean_split_latest
```

This writes:

- `training_plots/stage3_clean_split_latest/evaluation_return-average.png`
- `training_plots/stage3_clean_split_latest/training_return-average.png`
- `training_plots/stage3_clean_split_latest/model_val_loss.png`
- `training_plots/stage3_clean_split_latest/training_summary.txt`

### 9.2 Refresh plots every 60 seconds

```bash
watch -n 60 "bash -lc 'cd /home/ecer/PVRL/mbpo && source /home/ecer/miniconda3/etc/profile.d/conda.sh && conda activate mbpo && python scripts/plot_training_progress.py \"\$TRIAL\" --outdir training_plots/stage3_clean_split_latest >/dev/null 2>&1'"
```

### 9.3 Print the latest progress rows in the terminal

```bash
python - "$TRIAL/progress.csv" <<'PY'
import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])
cols = [
    'epoch',
    'training_iteration',
    'evaluation/return-average',
    'training/return-average',
    'model/val_loss',
    'model_rollout_length',
    'candidate_start_states_before_filter',
    'candidate_start_states_after_filter',
    'min_accepted_remaining_steps',
    'model_rollout_boundary_reaches_or_crosses',
]
cols = [c for c in cols if c in df.columns]
print(df[cols].tail(10).to_string(index=False))
PY
```

### 9.4 Read the generated summary quickly

```bash
python - "training_plots/stage3_clean_split_latest/training_summary.txt" <<'PY'
import sys
print(open(sys.argv[1], 'r', encoding='utf-8').read())
PY
```

## 10. Final checkpoint selection and untouched final-test evaluation

This is the canonical post-train command now:

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

./run_stage3_posttrain.sh "$TRIAL"
```

If you omit `"$TRIAL"`, the script uses `latest`.

What the script now does:

1. verifies the Stage 3 clean-split trial contract
2. ranks checkpoints on the 4 validation dates
3. picks the reporting checkpoint
4. evaluates that checkpoint on the 6 untouched final-test dates
5. runs advanced evaluation
6. runs `scripts/diagnose_tracking.py --gate`

Final outputs are written to:

- `/home/ecer/PVRL/mbpo/evaluation/pv_stage3_final_clean_test/`
- `/home/ecer/PVRL/mbpo/evaluation/pv_stage3_final_clean_test/advanced/`
- `/home/ecer/PVRL/mbpo/evaluation/pv_stage3_final_clean_test/diagnostics/`

Important generated files:

```bash
python - <<'PY'
paths = [
    '/home/ecer/PVRL/mbpo/evaluation/pv_stage3_final_clean_test/recommended_checkpoint.txt',
    '/home/ecer/PVRL/mbpo/evaluation/pv_stage3_final_clean_test/evaluation_summary.txt',
    '/home/ecer/PVRL/mbpo/evaluation/pv_stage3_final_clean_test/eval_scenario_confirmation.txt',
    '/home/ecer/PVRL/mbpo/evaluation/pv_stage3_final_clean_test/diagnostics/tracking_diagnosis.txt',
]
for p in paths:
    print(p)
PY
```

## 11. Sequential curriculum workflow

Yes, you can still use the three shell scripts as one workflow, but their roles are now:

1. `./run_sequential_stages.sh`
2. optional `./resume_stage3_refinement.sh`
3. `./run_stage3_posttrain.sh`

### 11.1 Start the sequential curriculum

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

./run_sequential_stages.sh
```

What it now does:

- trains Stage 0, Stage 1, Stage 2, and Stage 3 in sequence
- keeps Stage 3 on the clean-split config
- evaluates the Stage 3 training run on validation dates only
- writes Stage 3 validation evaluation to `evaluation/pv_stage3_validation_clean_split/`
- writes logs and discovered trial paths to `sequential_stage_artifacts/`

Important files produced by the sequential script:

- `sequential_stage_artifacts/stage0_trial_dir.txt`
- `sequential_stage_artifacts/stage1_trial_dir.txt`
- `sequential_stage_artifacts/stage2_trial_dir.txt`
- `sequential_stage_artifacts/stage3_trial_dir.txt`
- `sequential_stage_artifacts/stage3_checkpoint_dir.txt`
- `sequential_stage_artifacts/stage3_gate_status.txt`
- `sequential_stage_artifacts/stage*_train_<timestamp>.log`

### 11.2 Resume the Stage 3 run if you want more epochs

The refinement script still works, but now it is intentionally restricted to clean-split Stage 3 trials only.

Resume from the latest saved Stage 3 trial:

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

EXTRA_EPOCHS=100 ./resume_stage3_refinement.sh "$( < sequential_stage_artifacts/stage3_trial_dir.txt )"
```

Resume from a specific checkpoint directory:

```bash
EXTRA_EPOCHS=100 ./resume_stage3_refinement.sh /full/path/to/checkpoint_500
```

What it now checks before resuming:

- Stage 3 full-year random-weather contract
- physical observations
- `movement_penalty=0.0`
- clean-split `excluded_dates`
- validation `fixed_eval_dates`

### 11.3 Run the final untouched test after sequential training or refinement

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

./run_stage3_posttrain.sh "$( < sequential_stage_artifacts/stage3_trial_dir.txt )"
```

If you resumed Stage 3 and want the newest trial automatically:

```bash
./run_stage3_posttrain.sh latest
```

## 12. Minimal direct-command workflow

If you do not want the staged curriculum and only want the final clean experiment:

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

python scripts/verify_training_config.py \
  --config examples.config.pv_tracking.stage3_fullyear_random_clean_split

python scripts/validate_pv_rollouts.py \
  --config-path examples/config/pv_tracking/stage3_fullyear_random_clean_split.py

mbpo run_local examples.development \
  --config=examples.config.pv_tracking.stage3_fullyear_random_clean_split \
  --gpus=0 \
  --trial-gpus=0 \
  --cpus=4 \
  --trial-cpus=2

export TRIAL=$(ls -td ~/ray_mbpo/PVTracking/pv_tracking/seed:*/ | head -1)
export TRIAL="${TRIAL%/}"

./run_stage3_posttrain.sh "$TRIAL"
```

## 13. What changed in the reviewed shell scripts

The shell scripts were updated to match the current project state.

`run_sequential_stages.sh` now:

- uses a clean-split Stage 3 wrapper
- evaluates Stage 3 on validation dates only
- writes Stage 3 validation results to `evaluation/pv_stage3_validation_clean_split/`

`resume_stage3_refinement.sh` now:

- resumes only a clean-split Stage 3 trial
- verifies excluded dates and validation dates before continuing
- builds the resume config from `stage3_fullyear_random_clean_split`

`run_stage3_posttrain.sh` now:

- selects checkpoints using validation dates
- evaluates the selected checkpoint on untouched final-test dates
- writes final outputs to `evaluation/pv_stage3_final_clean_test/`

## 14. Recommended end state

For the final report, the directory that matters most is:

- `/home/ecer/PVRL/mbpo/evaluation/pv_stage3_final_clean_test/`

The three files to read first are:

```bash
python - <<'PY'
files = [
    '/home/ecer/PVRL/mbpo/evaluation/pv_stage3_final_clean_test/evaluation_summary.txt',
    '/home/ecer/PVRL/mbpo/evaluation/pv_stage3_final_clean_test/eval_scenario_confirmation.txt',
    '/home/ecer/PVRL/mbpo/evaluation/pv_stage3_final_clean_test/diagnostics/tracking_diagnosis.txt',
]
for f in files:
    print('\n===== %s =====' % f)
    print(open(f, 'r', encoding='utf-8').read())
PY
```

That is the current best CLI-only training and evaluation workflow for this project.
