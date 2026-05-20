# Model-Based Policy Optimization (MBPO)

This repository implements [Model-Based Policy Optimization (MBPO)](https://arxiv.org/abs/1906.08253) on top of [softlearning](https://github.com/rail-berkeley/softlearning). It includes a custom **PV solar tracking** Gym environment built with **pvlib**, wired into the full MBPO training, checkpointing, evaluation, and plotting workflow.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1siZA55atJi8Tgeefvv28WOqk7pFSynJP" width="80%">
</p>

## Project overview

| Layer | Location | Role |
|-------|----------|------|
| Gym environment | `mbpo/env/pv_tracking.py` | pvlib irradiance + 2D panel control (tilt/azimuth) |
| Environment registration | `mbpo/env/__init__.py` | Registers `PVTracking-v0` |
| Model termination fn | `mbpo/static/pv_tracking.py` | Marks done when predicted next state is non-finite |
| MBPO algorithm | `mbpo/algorithms/mbpo.py` | Ensemble dynamics model + SAC policy |
| Training entrypoint | `examples/development/main.py` | Ray Tune `ExperimentRunner` |
| Training config | `examples/config/pv_tracking/0.py` | Hyperparameters for PV runs |
| Variant builder | `examples/development/base.py` | Merges config into Ray variant spec |
| Utility scripts | `scripts/` | Env check, evaluate, plot, export weights |

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

### 1. MuJoCo (only for classic MBPO benchmarks)

MuJoCo is **not** required for PV tracking. For Hopper/HalfCheetah-style tasks, install [MuJoCo 1.50](https://www.roboti.us/index.html) at `~/.mujoco/mjpro150` and place your license at `~/.mujoco/mjkey.txt`.

### 2. Clone and install

```bash
git clone --recursive https://github.com/jannerm/mbpo.git
cd mbpo
conda env create -f environment/gpu-env.yml
conda activate mbpo
pip install -e viskit
pip install -e .
```

The conda environment installs dependencies from `environment/requirements.txt`, including **pvlib** for the PV environment.

## Quick validation (no training)

```bash
conda activate mbpo
cd mbpo   # repository root

# 1) pvlib + Gym environment smoke test
python scripts/check_pv_env.py

# 2) Verify Ray variant / config wiring (dry run)
mbpo run_example_dry examples.development \
  --config=examples.config.pv_tracking.0 \
  --gpus=0 --trial-gpus=0 --cpus=2 --trial-cpus=1
```

Expected: env checker prints obs/action shapes; dry run reports `max_path_length: 63`, `n_epochs: 50`, and one trial.

## Training (PV tracking)

### Command

```bash
conda activate mbpo
cd mbpo

mbpo run_local examples.development \
  --config=examples.config.pv_tracking.0 \
  --gpus=0 --trial-gpus=0 \
  --cpus=2 --trial-cpus=1
```

### What happens each epoch

1. **Environment interaction** — `SimpleSampler` collects transitions from `PVTracking-v0` (episode length 63 steps).
2. **Initial exploration** — uniform policy until `n_initial_exploration_steps` (630 ≈ 10 episodes) are in the replay pool.
3. **Dynamics model** — ensemble BNN trained every `model_train_freq` steps on real data.
4. **Model rollouts** — short imagined trajectories added to the pool (`rollout_schedule` controls horizon).
5. **Policy training** — SAC updated with mixed real/model batches (`real_ratio`).
6. **Evaluation** — one deterministic episode; metrics logged to `progress.csv`.
7. **Checkpoint** — `checkpoint.pkl`, `policy_weights.pkl`, and TensorFlow weights under `checkpoint_*`.

### Logs and checkpoints

Results are written under:

```
~/ray_mbpo/PVTracking/pv_tracking/seed:<seed>_<timestamp>/
  params.json          # full variant (written by Ray Tune)
  progress.csv         # per-epoch metrics
  result.json
  checkpoint_*/
    checkpoint.pkl     # full picklable state
    policy_weights.pkl # policy only (for fast evaluation)
    checkpoint         # TF checkpoint prefix
```

View runs with viskit:

```bash
viskit ~/ray_mbpo/PVTracking --port 6008
```

### Hyperparameters (`examples/config/pv_tracking/0.py`)

Defaults are tuned for a **balance between wall-clock and accuracy** on CPU:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_epochs` | 50 | Increase to 200–500 for stronger policies |
| `epoch_length` | 64 | Environment steps per training epoch |
| `max_path_length` | 63 | Full PV day episode (set in `base.py`) |
| `n_initial_exploration_steps` | 630 | ~10 episodes before learning |
| `model_train_freq` | 100 | Retrain dynamics every 100 steps |
| `max_model_t` | 120 s | Cap model training time per update |
| `rollout_batch_size` | 1000 | Imagined samples per model rollout phase |
| `num_networks` / `num_elites` | 5 / 3 | Ensemble size |
| `real_ratio` | 0.1 | Fraction of real vs model data in SAC batches |
| `rollout_schedule` | `[1, 20, 1, 1]` | Model rollout length schedule |
| `target_entropy` | -2 | Matches 2D action space |

TensorFlow / NumPy / Gym deprecation messages are safe to ignore.

## Evaluation

After training, evaluate a checkpoint (no retraining):

```bash
CKPT_DIR=~/ray_mbpo/PVTracking/pv_tracking/seed:<seed>_<timestamp>/checkpoint_<N>

python scripts/evaluate_agent.py \
  "${CKPT_DIR}" \
  --outdir evaluation/pv_tracking \
  --num-rollouts 10 \
  --max-path-length 63 \
  --deterministic
```

Outputs:

- `evaluation_summary.txt` — per-rollout returns and lengths
- `evaluation_rewards.png`
- `evaluation_lengths.png`

### Export policy weights (older checkpoints)

If `policy_weights.pkl` is missing (runs before the save hook was added):

```bash
python scripts/export_policy_weights.py "${CKPT_DIR}"
```

### Interactive rollouts (optional)

```bash
python -m examples.development.simulate_policy \
  "${CKPT_DIR}" \
  --num-rollouts 3 \
  --max-path-length 63 \
  --render-mode None \
  --deterministic
```

## Plotting

### Training curves (`progress.csv`)

```bash
TRIAL_DIR=~/ray_mbpo/PVTracking/pv_tracking/seed:<seed>_<timestamp>

python scripts/plot_training_progress.py \
  "${TRIAL_DIR}" \
  --outdir evaluation/pv_tracking/training_plots
```

Default metrics: `evaluation/return-average`, `training/return-average`, `model/val_loss`.

### Ray trial status (terminal summary)

Save the Ray status block to a text file, then:

```bash
python scripts/plot_ray_results.py /path/to/ray_status.txt \
  --outdir evaluation/pv_tracking/ray_plots
```

## Adding other environments

1. Copy [`examples/config/custom/0.py`](examples/config/custom/0.py).
2. Implement a Gym env under `mbpo/env/` and register it in `mbpo/env/__init__.py`.
3. Add a termination function in `mbpo/static/` (filename = lowercase domain, e.g. `pv_tracking.py` for `PVTracking`).
4. Set `max_path_length` in `examples/development/base.py` if episodes are shorter than 1000 steps.

## Classic MBPO benchmarks (MuJoCo)

```bash
mbpo run_local examples.development \
  --config=examples.config.halfcheetah.0 \
  --gpus=1 --trial-gpus=1
```

Rollout schedule format: `[start_epoch, end_epoch, start_length, end_length]` — e.g. `[20, 100, 1, 5]` ramps imagined rollout length from 1 to 5 between epochs 20 and 100.

## Comparing to published MBPO results

Precomputed learning curves: [Google Drive folder](https://drive.google.com/drive/folders/1matvC7hPi5al9-5S2uL4GuXfT5rzO9qU?usp=sharing).

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
