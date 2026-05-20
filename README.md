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

## Quick validation (no training)

```bash
conda activate mbpo
cd mbpo

# 1) Check the PV environment and action/observation shapes
python scripts/check_pv_env.py

# 2) Validate the PV rollout schedule and observation scaling
python scripts/validate_pv_rollouts.py

# 3) Dry-run the training config and verify variant wiring
mbpo run_example_dry examples.development \
  --config=examples.config.pv_tracking.0 \
  --gpus=0 --trial-gpus=0 --cpus=2 --trial-cpus=1
```

Expect output showing the observation and action shapes, rollout schedule, and a dry-run variant summary.

## Training PV tracking

### Basic command

```bash
conda activate mbpo
cd mbpo

mbpo run_local examples.development \
  --config=examples.config.pv_tracking.0 \
  --gpus=0 --trial-gpus=0 \
  --cpus=2 --trial-cpus=1
```

### What the training does

1. **Episode sampling** — each reset selects a random calendar day from the configured range.
2. **Weather generation** — each episode can use clear, partly cloudy, or overcast irradiance profiles.
3. **Interaction** — `SimpleSampler` collects `obs, action, reward, next_obs` transitions.
4. **Model training** — MBPO trains an ensemble dynamics model on real transitions.
5. **Imagined rollouts** — the learned model generates synthetic transitions to augment training.
6. **Policy optimization** — SAC updates the policy using mixed real/model batches.
7. **Checkpointing** — model and policy state are saved periodically.

### Important environment concepts

- `start_date` and `end_date` define the annual range for sampling training days.
- `randomize_day` ensures episodes are drawn from across the year.
- `weather_source='random'` enables daily weather variation.
- `randomize_initial_orientation` lets the tracker start from different angles.
- `max_path_length=63` corresponds to a single day of 15-minute steps.

## Key parameters and guidance

### Primary training settings

| Parameter | Type | Default | Purpose | Guidance |
|-----------|------|---------|---------|----------|
| `n_epochs` | int | 200 | Number of training epochs | Longer for better convergence; use 200–500 for PV tracking |
| `epoch_length` | int | 64 | Real env steps per epoch | Keep at 63 for one full day; increase only if you want multi-day episodes |
| `n_initial_exploration_steps` | int | 630 | Real exploration steps before learning | Use `max_path_length * 10` to ensure enough coverage before training |
| `model_train_freq` | int | 100 | Train model every this many env steps | 100 is reasonable; lower if model needs faster adaptation |
| `rollout_batch_size` | int | 500 | Imagined samples per rollout phase | Use a smaller batch size for more stable PV training |
| `real_ratio` | float | 0.2 | Fraction of real data in SAC minibatch | Increase if the real model is weak or policy is unstable |
| `rollout_schedule` | list | `[0, 100, 1, 10]` | Imagined rollout length schedule | Start at 1, then ramp to 10 by epoch 100 for conservative use |
| `target_entropy` | float | -2 | SAC exploration tuning | For 2D actions, -2 is a good starting value |

### PV environment settings

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `start_date` | `2020-01-01` | first candidate training day |
| `end_date` | `2020-12-31` | last candidate training day |
| `start_time` | `06:00` | first step of each episode |
| `periods` | `64` | number of timesteps per episode |
| `freq` | `15min` | timestep resolution |
| `weather_source` | `random` | choose between `clearsky` and `random` weather |
| `temperature` | `23.0` | base ambient temperature |
| `wind_speed` | `2.0` | base wind speed |
| `movement_penalty` | `0.01` | cost for motion to discourage unnecessary movement |

### Choosing parameters

- Use `n_epochs` ≥ 200 for PV problems to allow enough learning.
- Keep `epoch_length` and `max_path_length` aligned with one day if you want daily episodes.
- Use `weather_source='random'` for training to force robustness across weather.
- Use `start_date`/`end_date` to define your climate range and evaluate over hold-out day ranges.
- If training is unstable, reduce `rollout_batch_size` or shorten model rollout length.

## Evaluation and generalization

### Basic evaluation

```bash
python scripts/evaluate_agent.py \
  "/home/ecer/ray_mbpo/PVTracking/pv_tracking/seed:<seed>_<timestamp>/checkpoint_<N>" \
  --outdir evaluation/pv_tracking \
  --num-rollouts 10 \
  --max-path-length 63 \
  --deterministic
```

### Generalization testing

- Use `--test-start-date` and `--test-end-date` to evaluate on a different date range from training.
- Evaluate separately by season to measure robustness.
- Use `--compare-baselines` to compare against fixed and rule-based strategies.

### Example hold-out evaluation

```bash
python scripts/evaluate_agent.py \
  "/home/ecer/ray_mbpo/PVTracking/pv_tracking/seed:<seed>_<timestamp>/checkpoint_<N>" \
  --outdir evaluation/pv_tracking \
  --num-rollouts 5 \
  --max-path-length 63 \
  --deterministic \
  --compare-baselines \
  --test-start-date 2020-12-01 \
  --test-end-date 2020-12-31
```

### Baseline comparison helper

A lightweight helper script compares the learned policy against simple baselines using the same PV energy and movement-cost metrics.

```bash
python scripts/compare_baselines.py \
  "/home/ecer/ray_mbpo/PVTracking/pv_tracking/seed:<seed>_<timestamp>/checkpoint_<N>" \
  --outdir evaluation/pv_tracking \
  --num-rollouts 10 \
  --max-path-length 63
```

This script evaluates the learned policy and the following baselines:
- `fixed_no_motion` — keep the current tracker orientation unchanged
- `sun_tracking` — incremental action toward the current sun direction

### Evaluation output

- `evaluation_summary.txt` — includes reward, energy, season, and weather for every rollout
- `rollouts/rollout_<n>.csv` — full trajectory logs
- `rollout_plots/rollout_<n>_combined.png` — power/tilt/azimuth/reward plots
- `baseline_rollouts/` — baseline strategy comparisons if requested

## Scripts and utilities

| Script | Purpose |
|--------|---------|
| `scripts/check_pv_env.py` | Smoke-test the environment and action/observation shapes |
| `scripts/evaluate_agent.py` | Evaluate a saved checkpoint and save rollout plots |
| `scripts/plot_rollout_trajectory.py` | Plot a single rollout CSV file |
| `scripts/plot_training_progress.py` | Plot training metrics from `progress.csv` |
| `scripts/plot_ray_results.py` | Plot Ray trial resource/status logs |
| `scripts/evaluate_and_viskit.py` | Run evaluation and optionally launch viskit |
| `scripts/export_policy_weights.py` | Export policy weights from older checkpoints |
| `scripts/validate_pv_rollouts.py` | Validate PV rollout schedule and environment observation scaling |
| `scripts/compare_baselines.py` | Compare a trained PV policy against fixed/sun-tracking baselines |

## Extending the project

To add a new environment:

1. Create a new Gym environment under `mbpo/env/`.
2. Register it in `mbpo/env/__init__.py`.
3. Add a termination function in `mbpo/static/` with the lowercase domain name.
4. Add a new config file under `examples/config/<domain>/0.py`.
5. If the episode length is not 1000, add a domain-specific `max_path_length` in `examples/development/base.py`.

## Notes on the PV tracking design

- Episodes are defined as a single day, which is appropriate for solar tracking.
- The environment is designed to sample different days and weather conditions to avoid overfitting to one season.
- Reward is based on collected energy with a small motion penalty, which matches the optimization objective.
- The model-based pipeline is compatible because it uses the same `obs, action, reward, next_obs` transitions as real rollouts.

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
