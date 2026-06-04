# PV tracking — full-year MBPO-SAC

Single guide for setup, training, monitoring, and evaluation. Configs are **`conf1`**, **`conf2`**, **`conf3`**, **`conf_smoke`** (not event names).

## Setup (once per machine)

```bash
cd <repo>
conda env create -f environment/pv-env.yml
conda activate mbpo
pip install 'pvlib==0.10.4' 'tables==3.7.0' --no-deps
pip install 'opencv-python-headless==4.2.0.34'
pip install -e viskit
pip install -e .
python scripts/prepare_default_historical_weather.py
```

Optional env overrides (any machine):

| Variable | Default |
|----------|---------|
| `CONDA_SH` | `$HOME/miniconda3/etc/profile.d/conda.sh` |
| `CONDA_ENV_NAME` | `mbpo` |

All paths are **relative to the repo**; checkpoints go under `runs/<run_name>/checkpoints/`.

## Configs

```bash
python -c "from examples.config.pv_tracking.conf_registry import list_configs; list_configs()"
```

| Name | Role |
|------|------|
| **conf1** | Full year 2020, historical weather, annual day sampling (main) |
| **conf2** | Same MDP, conservative MBPO (`real_ratio=0.1`, shorter model rollouts) |
| **conf3** | Summer historical (shorter / debug) |
| **conf_smoke** | Few epochs — pipeline check only |

## Training (`train.sh`)

```bash
chmod +x train.sh result.sh

./train.sh run1 conf1 --cpus 4 --trial-cpus 2
./train.sh run2 conf2 --cpus 4 --trial-cpus 2    # parallel, separate dirs

./train.sh smoke1 conf_smoke --cpus 2 --trial-cpus 1 --verify
```

Shorthand: `./train.sh run1 conf1 cpu4` sets `--cpus 4`.

Creates:

```
runs/run1/
  run.json           # conf, cpus, module
  trial_dir.txt      # auto after train
  checkpoints/       # Ray Tune seed:* trials
  ray_tmp/
```

## Results (`result.sh`)

```bash
./result.sh run1 --status      # trial path + checkpoint
./result.sh run1 --plot-only  # training curves only
./result.sh run1 --eval-only  # frozen-policy MC eval
./result.sh run1 --full       # plot + eval (default)
```

Output (one folder per run):

```
runs/run1/results/
  training/          # evaluation_return-average.png, etc.
  progress.csv
  evaluation/        # vs sun_tracking + fixed_no_motion, summaries, paper_figures/
```

No manual `export TRIAL=...` — resolved from `runs/run1/trial_dir.txt`.

More eval rollouts: `NUM_ROLLOUTS=32 ./result.sh run1 --eval-only`

## Clean outputs before re-run

```bash
rm -rf runs/run1/results
# full reset for run1 (keeps nothing):
rm -rf runs/run1
```

## What is being trained / evaluated

- **Env:** real `PVTrackingEnv`, pvlib power, UTC grid 13:30–23:15, 78 steps.
- **MBPO:** BNN on real replay; short imagined rollouts; SAC on mixed batches (`real_ratio` in config).
- **Eval:** frozen policy, real env only (not the BNN). Matched seeds for learned / sun / fixed.
- **fixed_no_motion:** `action=0` → zero movement cost.
- **Reward:** `energy_kwh - movement_penalty × (|a0|+|a1|)`.

### Training plot σ (`evaluation_return-average.png`)

- **Line:** mean return over `eval_n_episodes` (default 8) each epoch, deterministic policy.
- **Band:** `evaluation/return-std` from the same 8 episodes — **not** SAC entropy; small-sample monitoring only.

### Post-train eval σ

Sample mean and std over `NUM_ROLLOUTS` MC episodes (`eval_seed_base + i`). See `results/evaluation/EVAL_STATISTICS.txt`.

## Preflight

```bash
python scripts/verify_preflight.py --config examples.config.pv_tracking.conf1
pytest tests/test_pv_tracking_audit.py -q
```

## Layout (essential source)

| Path | Role |
|------|------|
| `train.sh` / `result.sh` | Entry points |
| `examples/config/pv_tracking/conf*.py` | Hyperparameters |
| `mbpo/env/pv_tracking.py` | Environment |
| `mbpo/algorithms/mbpo.py` | MBPO + SAC |
| `scripts/evaluate_agent.py` | Post-train eval |
| `scripts/eval_utils.py` | Baselines, env build |
| `scripts/plot_training_progress.py` | Training plots |
| `environment/pv-env.yml` | Conda env |

Legacy `stage3_*.py` configs re-export `conf1`/`conf2` for compatibility.
