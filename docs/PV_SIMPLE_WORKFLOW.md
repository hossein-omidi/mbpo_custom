# PV tracking — simple workflow

Copy-paste commands per stage. **Each stage is independent:** fresh Ray trial, no `--restore`, no weights from a previous stage. Use the config and eval directory for that stage only.

Full procedure, gates, and directory rules: [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md).

Verified calendar days (Albuquerque PVGIS TMY): `examples/config/pv_tracking/verified_dates.py`.

**Physics:** Power and cumulative energy use **pvlib** in `PVTrackingEnv.step` for the learned policy and for baselines (`sun_tracking`, `fixed_no_motion`). Baselines only choose orientation; they do not use a separate energy formula.

---

## Setup (once per shell)

```bash
cd /home/user01/mbpo_custom
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mbpo
pip install -e .    # if mbpo CLI is missing
```

---

## Stage map (pick one track)

| Track | Config module | Helper script | Training plots | Eval output |
|-------|---------------|---------------|----------------|-------------|
| **Stage 0 clearsky** | `stage0_single_day` | `run_stage0_baseline_trial.sh` | `training_plots/stage0_baseline/` | `evaluation/pv_stage0_baseline/` |
| **Stage 0 cloudy + movement** | `stage0_single_day_mbpo_paper` | `run_stage0_paper_trial.sh` | `training_plots/stage0_mbpo_paper/` | `evaluation/pv_stage0_mbpo_paper_movement/` |
| **Stage 1 summer clearsky** | `0` | manual `mbpo run_local` | your choice | `evaluation/pv_stage1_clearsky_summer/` |
| **Stage 2 summer historical** | `stage2_random_weather` | manual | your choice | `evaluation/pv_stage2_random_weather_summer/` |
| **Stage 3 full year** | `stage3_fullyear_random_clean_split` | manual | your choice | define e.g. `evaluation/pv_stage3_final_test/` |

**Episode contract (all stages):** one calendar day = one episode; **78** env steps; UTC grid `13:30`–`23:15`; `periods=79`, `freq=7min30s`; i.i.d. day draw when `randomize_day=True`.

---

## A. Stage 0 MBPO paper (cloudy historical day) — usual active run

### Preflight

```bash
python scripts/verify_training_config.py --config examples.config.pv_tracking.stage0_single_day_mbpo_paper
python scripts/validate_pv_rollouts.py --config-path examples/config/pv_tracking/stage0_single_day_mbpo_paper.py
python scripts/verify_movement_cost_fairness.py --config-path examples/config/pv_tracking/stage0_single_day_mbpo_paper.py
python scripts/verify_pv_state_space.py
```

### Train

```bash
./scripts/run_stage0_paper_trial.sh train
```

### Training curves (this trial only)

```bash
./scripts/run_stage0_paper_trial.sh plot
```

Plots: `training_plots/stage0_mbpo_paper/` (`evaluation/return-average`, `alpha`, `policy/shifts-mean`, …).

**While training** (any trial — picks newest Ray seed unless `TRIAL` is set):

```bash
./scripts/run_pv_eval.sh monitor
./scripts/run_pv_eval.sh watch-plots
./scripts/run_pv_eval.sh trial
```

`run_pv_eval.sh` plots → `training_plots/active_run/` (handy for monitoring; for gates use the stage script below).

### Eval + fairness check

```bash
./scripts/run_stage0_paper_trial.sh eval
```

Uses `--eval-protocol inherit` (same date `2020-06-21`, `weather_source=historical`, movement penalty as config). **Do not** pass `--eval-weather-source clearsky` for this config.

### Gate

```bash
./scripts/run_stage0_paper_trial.sh gate
```

Report: `evaluation/pv_stage0_mbpo_paper_movement/diagnostics/tracking_diagnosis.txt`  
Rollout plots: `evaluation/pv_stage0_mbpo_paper_movement/diagnostics/plots/`

### Shortcut midterm / final (monitoring only)

`run_pv_eval.sh` uses the newest Ray trial and writes to convenience folders:

```bash
export TRIAL=$(cat sequential_stage_artifacts/stage0_mbpo_paper_trial_dir.txt)
./scripts/run_pv_eval.sh midterm    # → evaluation/active_latest/
./scripts/run_pv_eval.sh final      # → evaluation/paper_movement_final/
```

For official gates, use `./scripts/run_stage0_paper_trial.sh gate` → `evaluation/pv_stage0_mbpo_paper_movement/`.

---

## B. Stage 0 baseline (clearsky single day)

```bash
./scripts/run_stage0_baseline_trial.sh train
./scripts/run_stage0_baseline_trial.sh plot
./scripts/run_stage0_baseline_trial.sh eval
./scripts/run_stage0_baseline_trial.sh gate
```

| Step | Output |
|------|--------|
| Plots | `training_plots/stage0_baseline/` |
| Eval | `evaluation/pv_stage0_baseline/` |

Training day: `2020-06-15` (verified clear reference). Weather: `clearsky`.

---

## C. Stage 1 — summer i.i.d. clearsky (no helper script)

```bash
python scripts/verify_training_config.py --config examples.config.pv_tracking.0
python scripts/validate_pv_rollouts.py --config-path examples/config/pv_tracking/0.py

mbpo run_local examples.development \
  --config=examples.config.pv_tracking.0 \
  --gpus=0 --trial-gpus=0 --cpus=10 --trial-cpus=4 \
  --temp-dir=$PWD/.ray_tmp/stage1

export TRIAL=/path/to/seed:YOUR_TRIAL
export CKPT="$TRIAL/best_eval_checkpoint"

python scripts/plot_training_progress.py "$TRIAL" \
  --outdir training_plots/stage1 \
  --metrics evaluation/return-average policy/shifts-mean model/val_loss alpha

python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_stage1_clearsky_summer \
  --eval-protocol inherit \
  --max-path-length 78 \
  --compare-baselines \
  --num-rollouts 10

python scripts/diagnose_tracking.py \
  --eval-dir evaluation/pv_stage1_clearsky_summer \
  --trial-dir "$TRIAL" \
  --progress-csv "$TRIAL/progress.csv" \
  --verify-env --gate
```

Training: random summer days `2020-06-01`…`2020-08-31`, `weather_source=clearsky`, i.i.d. episodes.  
In-train / post-train hold-outs: `2020-06-07`, `2020-06-15`, `2020-07-15`, `2020-08-01`.

---

## D. Stage 2 — summer i.i.d. historical weather

Same as Stage 1, but:

- Config: `examples.config.pv_tracking.stage2_random_weather`
- Eval dir: `evaluation/pv_stage2_random_weather_summer`
- `weather_source=historical` (PVGIS TMY at 35°N, 106°W)

```bash
python scripts/verify_training_config.py --config examples.config.pv_tracking.stage2_random_weather
mbpo run_local examples.development --config=examples.config.pv_tracking.stage2_random_weather \
  --gpus=0 --trial-gpus=0 --cpus=10 --trial-cpus=4 \
  --temp-dir=$PWD/.ray_tmp/stage2
```

Post-train eval: `--eval-protocol inherit` only (do not override weather).

---

## E. Stage 3 — full-year historical (clean split)

Config: `examples.config.pv_tracking.stage3_fullyear_random_clean_split`

- Training: i.i.d. days in `2020-01-01`…`2020-12-31` excluding validation + final-test dates
- In-train checkpoint eval: `2020-02-15`, `2020-05-15`, `2020-08-15`, `2020-11-15`
- Final test (post-train only): see `STAGE3_FINAL_TEST_DATES` in `verified_dates.py`

```bash
python scripts/verify_training_config.py --config examples.config.pv_tracking.stage3_fullyear_random_clean_split
mbpo run_local examples.development \
  --config=examples.config.pv_tracking.stage3_fullyear_random_clean_split \
  --temp-dir=$PWD/.ray_tmp/stage3
```

Post-train eval example:

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_stage3_final_test \
  --eval-protocol inherit \
  --fixed-eval-dates 2020-01-15,2020-03-20,2020-06-21,2020-09-22,2020-10-15,2020-12-07 \
  --max-path-length 78 \
  --compare-baselines \
  --num-rollouts 10
```

---

## F. Stage 3 — full-year historical (standalone)

Skip Stages 0–2. Full guide: **[STAGE3_FULLYEAR_WORKFLOW.md](STAGE3_FULLYEAR_WORKFLOW.md)**

```bash
./scripts/run_stage3_fullyear.sh verify
./scripts/run_stage3_fullyear.sh train
./scripts/run_stage3_fullyear.sh plot
./scripts/run_stage3_fullyear.sh all-eval
```

Config: `stage3_fullyear_random_clean_split.py` · MC plots: `evaluation/pv_stage3_fullyear_mc/`

---

## Optional cleanup (plots / eval only)

Does **not** delete Ray trials.

```bash
rm -rf training_plots/active_run evaluation/active_latest evaluation/paper_movement_final
```

---

## Where to look

| What | Where |
|------|--------|
| State / weather verification | `verification/pv_state_space/` |
| Paper training curves | `training_plots/stage0_mbpo_paper/` |
| Paper eval + diagnostics | `evaluation/pv_stage0_mbpo_paper_movement/` |
| Ray trial | `~/ray_mbpo/PVTracking/pv_tracking/seed:…/` |
| Trial pointer (paper) | `sequential_stage_artifacts/stage0_mbpo_paper_trial_dir.txt` |

See also: [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md), [STAGE3_FULLYEAR_WORKFLOW.md](STAGE3_FULLYEAR_WORKFLOW.md) (full-year track).
