# Full-Year Simple Training

Use this exact flow for the clean full-year run.

## Rules

- Use only `examples.config.pv_tracking.stage3_fullyear_random_clean_split`
- Do not use `run_sequential_stages.sh`
- Do not use `--restore`
- Do not reuse old replay pools
- Train fresh, monitor, then run final evaluation

## 1. Shell setup

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo
```

## 2. Clean old generated outputs

```bash
cd /home/ecer/PVRL/mbpo
rm -rf training_plots/stage3_clean_split_latest
rm -rf evaluation/pv_stage3_final_clean_test
rm -rf evaluation/pv_stage3_validation_clean_split
```

## 3. Prepare historical weather file

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

python scripts/prepare_default_historical_weather.py
```

## 4. Preflight

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

python scripts/check_pv_env.py --observation-mode physical --weather-source historical --validate-weather
python scripts/verify_training_config.py --config examples.config.pv_tracking.stage3_fullyear_random_clean_split
python scripts/validate_pv_rollouts.py --config-path examples/config/pv_tracking/stage3_fullyear_random_clean_split.py
pytest tests/test_pv_tracking_audit.py -q
```

## 5. Train

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

## 6. Set the active trial

```bash
export TRIAL=$(ls -td ~/ray_mbpo/PVTracking/pv_tracking/seed:*/ | head -1)
export TRIAL="${TRIAL%/}"
export CKPT="$TRIAL/best_eval_checkpoint"
echo "$TRIAL"
```

## 7. Safe monitoring

Create updated plots:

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

python scripts/plot_training_progress.py "$TRIAL" \
  --outdir training_plots/stage3_clean_split_latest
```

Refresh plots every 60 seconds:

```bash
watch -n 60 "bash -lc 'cd /home/ecer/PVRL/mbpo && source /home/ecer/miniconda3/etc/profile.d/conda.sh && conda activate mbpo && python scripts/plot_training_progress.py \"\$TRIAL\" --outdir training_plots/stage3_clean_split_latest >/dev/null 2>&1'"
```

Print latest training status:

```bash
python - "$TRIAL/progress.csv" <<'PY'
import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])
cols = [
    'epoch',
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
print(df[cols].tail(20).to_string(index=False))
PY
```

## 8. Final evaluation

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

./run_stage3_posttrain.sh "$TRIAL"
```

## 9. Read the final outputs

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

## 10. One-command summary

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo
python scripts/prepare_default_historical_weather.py
python scripts/check_pv_env.py --observation-mode physical --weather-source historical --validate-weather
python scripts/verify_training_config.py --config examples.config.pv_tracking.stage3_fullyear_random_clean_split
python scripts/validate_pv_rollouts.py --config-path examples/config/pv_tracking/stage3_fullyear_random_clean_split.py
pytest tests/test_pv_tracking_audit.py -q
mbpo run_local examples.development --config=examples.config.pv_tracking.stage3_fullyear_random_clean_split --gpus=0 --trial-gpus=0 --cpus=4 --trial-cpus=2
```
