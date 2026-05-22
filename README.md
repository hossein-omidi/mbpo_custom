# Model-Based Policy Optimization (MBPO)

This repository implements [Model-Based Policy Optimization (MBPO)](https://arxiv.org/abs/1906.08253) on top of [softlearning](https://github.com/rail-berkeley/softlearning). It includes a custom **PV solar tracking** Gym environment built with **pvlib**, wired into the full MBPO training, checkpointing, evaluation, and plotting workflow.

## What this repository can do

- Train a model-based reinforcement learning agent to control a 2-axis solar tracker.
- Use `pvlib` to simulate realistic solar geometry and irradiance.
- Train on randomized days of the year and stochastic weather conditions.
- Evaluate the learned policy against baseline strategies.
- Export rollout data and plots for inspection.

## Project overview

| Layer | Location | Role |
|-------|----------|------|
| Gym environment | `mbpo/env/pv_tracking.py` | PV tracking env with pvlib-based solar geometry, irradiance, seasonal sampling, and stochastic weather |
| Environment registration | `mbpo/env/__init__.py` | Registers `PVTracking-v0` |
| Model termination fn | `mbpo/static/pv_tracking.py` | Marks fake-model transitions done when the predicted next state is invalid |
| MBPO algorithm | `mbpo/algorithms/mbpo.py` | Ensemble dynamics model + SAC policy, plus model rollouts |
| Training entrypoint | `examples/development/main.py` | Ray Tune experiment runner |
| Training config | `examples/config/pv_tracking/0.py` | Default PV tracking hyperparameters and environment settings |
| Variant builder | `examples/development/base.py` | Merges config and env params into Ray Tune variant spec |
| Utility scripts | `scripts/` | Environment check, evaluation, plotting, export utilities |

### Data flow (training → evaluation)

```
examples/config/pv_tracking/0.py
        ↓
examples.development (Ray Tune)
        ↓
GymAdapter → PVTracking-v0 (pvlib)
        ↓
MBPO: collect real data → train ensemble BNN → imaginary rollouts → train SAC
        ↓
checkpoint_*/  (checkpoint.pkl, policy_weights.pkl, TF checkpoint)
        ↓
scripts/evaluate_agent.py  +  scripts/plot_training_progress.py
```

## Installation

### 1. Clone and install

```bash
git clone --recursive https://github.com/jannerm/mbpo.git
cd mbpo
conda env create -f environment/gpu-env.yml
conda activate mbpo
pip install -e viskit
pip install -e .
```

The conda environment installs dependencies from `environment/requirements.txt`, including **pvlib**.

### 2. Optional MuJoCo support

MuJoCo is not required for PV tracking. It is only needed for classic MBPO benchmarks like Hopper and HalfCheetah.

## PV tracking: end-to-end workflow

Use this order every time you train or evaluate. All steps assume **UTC** time (`tz='UTC'`, grid **06:00–21:45 UTC**, **63** steps/day). Details: [docs/PV_TRACKING_TIME_AND_CHECKPOINTS.md](docs/PV_TRACKING_TIME_AND_CHECKPOINTS.md).

| Step | What | Commands (below) |
|------|------|------------------|
| 0 | Pre-flight checks (env, time, config) | [§ Step 0](#step-0-pre-training-checks) |
| 1 | Train MBPO | [§ Step 1](#step-1-training) |
| 2 | Monitor run + pick checkpoint | [§ Step 2](#step-2-during-and-after-training) |
| 3 | Post-training eval + plots | [§ Step 3](#step-3-post-training-evaluation) |
| 4 | Interpret results | [§ Interpretation](#how-to-interpret-results) |

**Paths (adjust if yours differ):**

- Repo: `/home/ecer/PVRL/mbpo`
- Ray trials: `~/ray_mbpo/PVTracking/pv_tracking/seed:<number>_<date>_<id>/` (contains `progress.csv`, `params.json`, `checkpoint_*`, `best_eval_checkpoint/`)
- Eval outputs: `/home/ecer/PVRL/mbpo/evaluation/<run_name>/`

Do not use the literal name `YOUR_SEED_DIR` — set `TRIAL` to a real `seed:...` folder from `ls -lt ~/ray_mbpo/PVTracking/pv_tracking/`.

---

## Step 0: Pre-training checks

Run these **before** starting a long training job. They do not modify code.

```bash
conda activate mbpo
cd /home/ecer/PVRL/mbpo

# A) Environment smoke test (shapes, one reset/step)
python scripts/check_pv_env.py --observation-mode physical --log-obs

# B) Rollout schedule + observation scaling vs config
python scripts/validate_pv_rollouts.py

# C) End-to-end UTC + horizon uniformity (train / eval / baselines / plots)
python scripts/verify_utc_uniformity.py

# D) UTC time / pvlib alignment (December example)
python scripts/solar_time_sanity.py --date 2020-12-21
python scripts/solar_time_sanity.py --date 2020-12-21 --env-rollout

# E) Dry-run: variant wiring, prints tz / start_time / periods at startup
mbpo run_example_dry examples.development \
  --config=examples.config.pv_tracking.0 \
  --gpus=0 --trial-gpus=0 --cpus=2 --trial-cpus=1
```

**Expect from current config** (`examples/config/pv_tracking/0.py`):

- `observation_mode='physical'` → **11-D** observations (legacy **15-D** only if you remove `observation_mode` from config).
- `tz='UTC'`, `start_time='06:00'`, `periods=64`, `epoch_length=63`.
- At **35°N, 106°W** in **December**, ~**33/64** episode timestamps are **night** (solar altitude ≤ 0°); zero power on the plot before ~**14:15 UTC** is normal, not a bug.

Optional: compare with local time **for human reading only** (not used in training):

```bash
python scripts/solar_time_sanity.py --date 2020-12-21 --compare-tz America/Denver
```

---

## Step 1: Training

### Start training

```bash
conda activate mbpo
cd /home/ecer/PVRL/mbpo

mbpo run_local examples.development \
  --config=examples.config.pv_tracking.0 \
  --gpus=0 --trial-gpus=0 \
  --cpus=2 --trial-cpus=1
```

At startup, look for a line like:

```text
PVTracking training env check: tz='UTC' start_time='06:00' periods=64 (63 steps), observation_mode='physical' ...
```

If `tz` is not `UTC` or `periods` ≠ 64, fix `examples/config/pv_tracking/0.py` before relying on results.

### What the training does

1. **Episode sampling** — each reset selects a random calendar day from the configured range.
2. **Weather generation** — each episode can use clear, partly cloudy, or overcast irradiance profiles.
3. **Interaction** — `SimpleSampler` collects `obs, action, reward, next_obs` transitions.
4. **Model training** — MBPO trains an ensemble dynamics model on real transitions.
5. **Imagined rollouts** — the learned model generates synthetic transitions to augment training.
6. **Policy optimization** — SAC updates the policy using mixed real/model batches.
7. **Checkpointing** — `latest_checkpoint/`, `checkpoint_<epoch>/`, and **`best_eval_checkpoint/`** when `evaluation/return-average` improves.

---

## Step 2: During and after training

### Find the trial directory

Ray writes each run under `~/ray_mbpo/PVTracking/pv_tracking/seed:<number>_<timestamp><id>/` (not `YOUR_SEED_DIR` — that was only a README placeholder).

```bash
# List recent seeds (newest first)
ls -lt ~/ray_mbpo/PVTracking/pv_tracking/ | head

# Option A — latest trial automatically (recommended)
export TRIAL=$(ls -td ~/ray_mbpo/PVTracking/pv_tracking/seed:*/ | head -1)
export TRIAL="${TRIAL%/}"   # strip trailing slash

# Option B — pick one run explicitly (copy name from ls)
# export TRIAL=~/ray_mbpo/PVTracking/pv_tracking/seed:6892_2026-05-22_21-55-029kyh15y1

echo "TRIAL=$TRIAL"
ls -la "$TRIAL"
ls "$TRIAL"/checkpoint_* "$TRIAL"/best_eval_checkpoint 2>/dev/null | head
test -f "$TRIAL/progress.csv" && test -f "$TRIAL/params.json" && echo "OK: trial files found"
```

### Monitor training metrics

Scripts accept `latest` if you skip `export TRIAL`:

```bash
# Training curves (return, Q-loss, etc.)
python scripts/plot_training_progress.py latest
# or: python scripts/plot_training_progress.py "$TRIAL"

# Raw log (last lines)
tail -30 "$TRIAL/progress.csv"
```

### Checkpoint selection (scientific)

| Checkpoint | When to use |
|------------|-------------|
| `best_eval_checkpoint/` | Default — best **training-time** `evaluation/return-average` |
| `checkpoint_<N>/` | Compare a specific epoch |
| `latest_checkpoint/` | Resume / debug only |

**Training metric ≠ best energy on hold-out days.** After training, confirm with **total_energy_kwh** vs baselines (Step 3).

```bash
export CKPT="$TRIAL/best_eval_checkpoint"

python scripts/select_best_checkpoint.py latest \
  # or: python scripts/select_best_checkpoint.py "$TRIAL" \
  --deterministic \
  --compare-baselines \
  --fixed-eval-dates 2020-12-07,2020-12-14,2020-12-21,2020-12-28 \
  --num-rollouts 4 \
  --max-path-length 63
```

Use the path printed as “Recommended for reporting” for Step 3, or `best_eval_checkpoint` if it ranks first.

---

## Step 3: Post-training evaluation

Full pipeline: load checkpoint → rollouts (same UTC MDP as training) → baselines → CSV → plots → reports.

**Always pass:** `--max-path-length 63`, `--eval-protocol inherit`, and `--deterministic` for reporting.

### 3a) Standard hold-out (December, fixed dates, with baselines)

```bash
conda activate mbpo
cd /home/ecer/PVRL/mbpo

export CKPT="$TRIAL/best_eval_checkpoint"   # or checkpoint from select_best_checkpoint.py

python scripts/evaluate_agent.py "$CKPT" \
  --outdir /home/ecer/PVRL/mbpo/evaluation/pv_final_utc \
  --eval-protocol inherit \
  --max-path-length 63 \
  --num-rollouts 10 \
  --deterministic \
  --compare-baselines \
  --fixed-eval-dates 2020-12-07,2020-12-14,2020-12-21,2020-12-28
```

### 3b) Random days over the training year (generalization)

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir /home/ecer/PVRL/mbpo/evaluation/pv_tracking_random \
  --eval-protocol inherit \
  --max-path-length 63 \
  --num-rollouts 10 \
  --deterministic \
  --compare-baselines
```

### 3c) Held-out month (random days in December only)

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir /home/ecer/PVRL/mbpo/evaluation/pv_holdout_dec2020 \
  --eval-protocol inherit \
  --max-path-length 63 \
  --num-rollouts 10 \
  --deterministic \
  --compare-baselines \
  --test-start-date 2020-12-01 \
  --test-end-date 2020-12-31
```

### 3d) Seasonal equinox/solstice dates

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir /home/ecer/PVRL/mbpo/evaluation/pv_tracking_seasonal \
  --eval-protocol inherit \
  --max-path-length 63 \
  --num-rollouts 4 \
  --deterministic \
  --compare-baselines \
  --fixed-eval-dates 2020-03-21,2020-06-21,2020-09-22,2020-12-21
```

### 3e) Baselines only (no neural policy)

```bash
python scripts/compare_baselines.py "$CKPT" \
  --outdir /home/ecer/PVRL/mbpo/evaluation/pv_baselines_dec \
  --eval-protocol inherit \
  --max-path-length 63 \
  --num-rollouts 10 \
  --deterministic \
  --fixed-eval-dates 2020-12-07,2020-12-14,2020-12-21,2020-12-28
```

### 3f) Re-plot one rollout CSV

```bash
python scripts/plot_rollout_trajectory.py \
  --csv /home/ecer/PVRL/mbpo/evaluation/pv_final_utc/rollouts/rollout_3.csv \
  --outdir /home/ecer/PVRL/mbpo/evaluation/pv_final_utc/rollout_plots
```

### Post-training outputs (what to open)

| File | Purpose |
|------|---------|
| `evaluation_summary.txt` / `.json` | Mean reward, **total_energy_kwh**, movement, per-season stats |
| `eval_scenario_confirmation.txt` | **Fairness:** same UTC grid, seeds, dates for policy + baselines; **sunrise/sunset UTC** per date |
| `reward_time_analysis.txt` | Peak power/reward **UTC hours**, solar-altitude windows |
| `evaluation_method_comparison.png` | Policy vs baselines (energy, reward, movement) |
| `evaluation_reward_by_solar_altitude.png` | Physics-based windows (preferred for “midday sun”) |
| `evaluation_reward_by_time_window.png` | UTC clock windows (winter “midday” may be night) |
| `rollouts/rollout_<n>.csv` | Per-step `clock_hour_utc`, power, `solar_altitude_deg` |
| `rollout_plots/rollout_<n>_combined.png` | UTC x-axis; **gray = night** (solar alt ≤ 0°) |

### Important environment concepts

- `start_date` and `end_date` define the annual range for sampling training days.
- `randomize_day` ensures episodes are drawn from across the year.
- `weather_source='random'` enables daily weather variation.
- `randomize_initial_orientation` lets the tracker start from different angles.
- One PV day = **64 timestamps** at **15 min** from **06:00** → **21:45** (**63** `env.step()` calls). `max_path_length=63` and `epoch_length=63` must stay aligned.

## Key parameters and guidance

### Primary training settings

| Parameter | Type | Default | Purpose | Guidance |
|-----------|------|---------|---------|----------|
| `n_epochs` | int | 200 | Number of training epochs | Longer for better convergence; use 200–500 for PV tracking |
| `epoch_length` | int | 63 | Real env steps per epoch | Must match one day (63 steps); do not use 64 unless you change `periods` |
| `n_initial_exploration_steps` | int | 630 | Real exploration steps before learning | Use `max_path_length * 10` to ensure enough coverage before training |
| `model_train_freq` | int | 100 | Train model every this many env steps | 100 is reasonable; lower if model needs faster adaptation |
| `rollout_batch_size` | int | 300 | Imagined samples per rollout phase | Use a smaller batch size for more stable PV training |
| `real_ratio` | float | 0.5 | Fraction of real data in SAC minibatch | Increase if the real model is weak or policy is unstable |
| `min_alpha` | float | 0.05 | Lower bound for SAC temperature | Prevents entropy from collapsing too quickly |
| `max_model_rollout_length` | int | 4 | Hard cap on model rollout horizon | Keep imagined trajectories short for PV tracking |
| `rollout_schedule` | list | `[20, 120, 1, 4]` | Imagined rollout length schedule | Start at 1, then ramp to 4 by epoch 120 for conservative use |
| `target_entropy` | float | -2 | SAC exploration tuning | For 2D actions, -2 is a good starting value |

### PV environment settings

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `start_date` | `2020-01-01` | first candidate training day |
| `end_date` | `2020-12-31` | last candidate training day |
| `tz` | `UTC` | pvlib `Location.tz` and env clock (`info['clock_hour_utc']`) |
| `start_time` | `06:00` | episode grid start (UTC morning on the index) |
| `periods` | `64` | timestamps per episode (63 actions) |
| `freq` | `15min` | timestep resolution → 06:00–21:45 UTC |
| `weather_source` | `random` | choose between `clearsky` and `random` weather |
| `temperature` | `23.0` | base ambient temperature |
| `wind_speed` | `2.0` | base wind speed |
| `movement_penalty` | `0.0001` | cost for motion (config default; increase only with care) |

### Choosing parameters

- Use `n_epochs` ≥ 200 for PV problems to allow enough learning.
- Keep `epoch_length` and `max_path_length` aligned with one day if you want daily episodes.
- Use `weather_source='random'` for training to force robustness across weather.
- Use `start_date`/`end_date` to define your climate range and evaluate over hold-out day ranges.
- If training is unstable, reduce `rollout_batch_size` or shorten model rollout length.

## Reward, state space, and evaluation (read before interpreting plots)

### Reward function (`mbpo/env/pv_tracking.py`)

Per step:

```text
energy_kwh     = power_W * (15 min as hours) / 1000
movement_cost  = movement_penalty * (|Δtilt|/max_Δtilt + |Δazimuth|/max_Δazimuth)
reward         = energy_kwh - movement_cost
```

This is **physically meaningful** for maximizing collected energy with an actuator penalty. It is **not** cumulative energy in the reward; SAC sums per-step rewards over the episode.

**Important:** Step reward = energy − movement. Peak **step reward** time often differs from peak **power** time (movement penalty). Use **total_energy_kwh** for tracking quality. This is visible in `reward_time_analysis.txt` and is **not a plotting bug**.

Use **total_energy_kwh** and **peak_power_time** to judge tracking quality, not peak step reward alone.

### Observation vector

**Legacy (default, 15-D):** matches existing checkpoints. Index 10 is **current** power (not previous-step). Time/day sin/cos are largely redundant with solar angles and can encourage calendar overfitting — see [docs/PV_TRACKING_OBSERVATION_REVIEW.md](docs/PV_TRACKING_OBSERVATION_REVIEW.md).

**Recommended for new training (11-D physical):** set `observation_mode='physical'` in `environment_kwargs` — sensor/actuator fields + `cos_aoi` only (no `episode_progress`, clock, or calendar in the policy vector). MBPO horizon uses solar-time decoding in `StaticFns` (not in obs). **Retrain required** after switching.

```bash
python scripts/check_pv_env.py --observation-mode physical --log-obs
```

### Hyperparameters (current defaults — review, do not change blindly)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `movement_penalty` | 0.0001 | Low default; raising it discourages motion (may freeze tracker) |
| `real_ratio` | 0.5 | Standard MBPO; keep if model rollouts are stable |
| `rollout_schedule` | [20,120,1,3] | Conservative imagined horizon |
| `discount` | 0.99 | OK for 63-step days |
| `reward_scale` | 1.0 | OK (do not inflate to hide movement penalty) |
| `target_entropy` / `min_alpha` | -2 / 0.05 | OK for 2D actions |
| `eval_n_episodes` | 5 | Training-time eval only; use ≥10 rollouts in `evaluate_agent.py` |

**Risky if training is unstable:** `real_ratio` too high with poor model terminals (addressed in `mbpo/static/pv_tracking.py`), `movement_penalty` too low (agent jitters), `movement_penalty` too high (agent barely moves).

---

## Time standard (UTC) — read before plots

| Setting | Value |
|---------|--------|
| `tz` | **`UTC`** everywhere (config, env, pvlib, train, eval, CSV, plots) |
| Episode grid | **06:00–21:45 UTC**, 64 timestamps, **63** actions |
| Eval flag | `--eval-protocol inherit` (same as `utc` / `legacy_utc`) |
| Plot x-axis | `clock_hour_utc` — **not** Denver local time |

**Winter at 35°N, 106°W:** sunrise ≈ **14:15 UTC**, GHI peak ≈ **19:00 UTC**. Zero power from **06:00–~14:00 UTC** = **night** (solar altitude &lt; 0°), not a train/test bug. Gray bands on rollout plots mark those steps.

**Do not** change `tz`, `start_time`, or `periods` without **retraining** (different MDP).

Baselines (fair comparison, same seeds/dates):

- `fixed_no_motion` — panel held near tilt=30°, azimuth=180°
- `sun_tracking` — myopic step toward current sun position

More detail: [docs/PV_TRACKING_TIME_AND_CHECKPOINTS.md](docs/PV_TRACKING_TIME_AND_CHECKPOINTS.md).

---

## Command cheat sheet (copy-paste)

```bash
conda activate mbpo
cd /home/ecer/PVRL/mbpo

# Point at your newest Ray trial (required before using $TRIAL / $CKPT)
export TRIAL=$(ls -td ~/ray_mbpo/PVTracking/pv_tracking/seed:*/ | head -1)
export TRIAL="${TRIAL%/}"
export CKPT="$TRIAL/best_eval_checkpoint"
echo "TRIAL=$TRIAL"
```

| Goal | Command |
|------|---------|
| Pre-check env | `python scripts/check_pv_env.py --observation-mode physical` |
| **UTC + horizon audit** | `python scripts/verify_utc_uniformity.py` |
| Pre-check time/pvlib | `python scripts/solar_time_sanity.py --date 2020-12-21 --env-rollout` |
| Train | `mbpo run_local examples.development --config=examples.config.pv_tracking.0 --gpus=0 --trial-gpus=0 --cpus=2 --trial-cpus=1` |
| Plot training | `python scripts/plot_training_progress.py latest` |
| Rank checkpoints | `python scripts/select_best_checkpoint.py latest --deterministic --compare-baselines --fixed-eval-dates 2020-12-07,2020-12-14,2020-12-21,2020-12-28 --num-rollouts 4 --max-path-length 63` |
| Full eval + baselines | `python scripts/evaluate_agent.py "$CKPT" --outdir evaluation/pv_final_utc --eval-protocol inherit --max-path-length 63 --num-rollouts 10 --deterministic --compare-baselines --fixed-eval-dates 2020-12-07,2020-12-14,2020-12-21,2020-12-28` |
| Baselines only | `python scripts/compare_baselines.py "$CKPT" --outdir evaluation/pv_baselines --eval-protocol inherit --max-path-length 63 --deterministic --fixed-eval-dates 2020-12-21` |

Evaluation scripts batch observations correctly via `prepare_policy_observation_batch()` — do not call `policy.actions_np([obs_vector])` on a 1-D vector.

---

## How to interpret results

1. **Total energy_kwh** — primary physical performance metric.  
2. **Rollout plots** — x-axis is **UTC**; winter peaks often **17–20h UTC** at this longitude.  
3. **Solar-altitude windows** — use `evaluation_reward_by_solar_altitude.png` for physics.  
4. **UTC clock windows** — labels like “midday” are UTC, not Denver local.  
5. Compare all methods in `eval_scenario_confirmation.txt` (same dates, seeds, weather, horizon).  
6. **Do not** change `tz` or episode grid without retraining (different MDP).

## Scripts and utilities

| Script | Purpose |
|--------|---------|
| `scripts/check_pv_env.py` | Pre-training: env smoke test, obs/action shapes |
| `scripts/validate_pv_rollouts.py` | Pre-training: schedule + obs scaling vs config |
| `scripts/verify_utc_uniformity.py` | **Pre/post:** end-to-end UTC, 63-step horizon, plot index audit |
| `scripts/solar_time_sanity.py` | Pre/post: UTC grid vs pvlib solar position / GHI |
| `scripts/evaluate_agent.py` | **Post-training:** rollouts, baselines, CSV, plots, reports |
| `scripts/compare_baselines.py` | **Post-training:** baselines only (no policy forward pass) |
| `scripts/select_best_checkpoint.py` | **Post-training:** rank checkpoints by hold-out energy |
| `scripts/pv_trial_paths.py` | Helper: resolve `seed:...` trial dirs (used by other scripts) |
| `scripts/plot_training_progress.py` | **During training:** curves from `progress.csv` (`latest` or trial path) |
| `scripts/plot_rollout_trajectory.py` | Re-plot one `rollout_*.csv` |
| `scripts/plot_ray_results.py` | Ray trial resource/status logs |
| `scripts/evaluate_and_viskit.py` | Evaluation + optional viskit |
| `scripts/export_policy_weights.py` | Export weights from older checkpoints |
| `docs/PV_TRACKING_TIME_AND_CHECKPOINTS.md` | UTC time standard, grid design, checkpoint protocol |
| `docs/PV_TRACKING_OBSERVATION_REVIEW.md` | Legacy vs physical observation notes |

## Extending the project

To add a new environment:

1. Create a new Gym environment under `mbpo/env/`.
2. Register it in `mbpo/env/__init__.py`.
3. Add a termination function in `mbpo/static/` with the lowercase domain name.
4. Add a new config file under `examples/config/<domain>/0.py`.
5. If the episode length is not 1000, add a domain-specific `max_path_length` in `examples/development/base.py`.

## Notes on the PV tracking design

- Episodes are one day (63 × 15 min steps from 06:00).
- Reward = incremental energy minus movement penalty (see **Reward, state space, and evaluation** above).
- Step reward can be maximized late in the day without maximizing power; always report energy and peak power time.
- Evaluation uses `scripts/eval_utils.py` for consistent env settings and timing diagnostics.

## Reference

```bibtex
@inproceedings{janner2019mbpo,
  author = {Michael Janner and Justin Fu and Marvin Zhang and Sergey Levine},
  title = {When to Trust Your Model: Model-Based Policy Optimization},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2019}
}
```

## Acknowledgments

SAC implementation from [softlearning](https://github.com/rail-berkeley/softlearning). Dynamics modeling from [PETS](https://github.com/kchua/handful-of-trials).
