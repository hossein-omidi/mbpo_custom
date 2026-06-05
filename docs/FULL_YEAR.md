# PV tracking — NSRDB multi-year MBPO-SAC

Single guide for setup, training, monitoring, and evaluation. All configs use **NSRDB** empirical weather (`nsrdb_multiyear`) on a native **5-minute** control grid (117 transitions per episode).

## Setup (once per machine)

```bash
cd <repo>
conda env create -f environment/pv-env.yml
conda activate mbpo
pip install 'pvlib==0.10.4' 'tables==3.7.0' --no-deps
pip install 'opencv-python-headless==4.2.0.34'
pip install -e viskit
pip install -e .
```

NSRDB CSVs live under `data/pv_weather/nsrdb/` with manifest `data/pv_weather/nsrdb/albuquerque_multiyear_manifest.json`.

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
| **stage3_nsrdb** | Main reference (`stage3_multiyear_nsrdb_scenario.py`) |
| **conf1** | Fast — shorter epochs, light MBPO (parallel run) |
| **conf2** | Slow — long run, conservative MBPO (parallel run) |
| **conf3** | Strong — stable MBPO, recommended production (parallel run) |
| **conf4** | Noisy — irradiance/observation augmentation (parallel run) |

## Training (`train.sh`)

```bash
chmod +x train.sh result.sh

./train.sh run_nsrdb stage3_nsrdb --cpus 4 --trial-cpus 2 --verify
./train.sh run_fast conf1 --cpus 4 --trial-cpus 2    # parallel profiles
./train.sh run_slow conf2 --cpus 4 --trial-cpus 2
./train.sh run_strong conf3 --cpus 4 --trial-cpus 2
./train.sh run_noisy conf4 --cpus 4 --trial-cpus 2
```

Shorthand: `./train.sh run1 conf3 cpu4` sets `--cpus 4`.

Creates:

```
runs/run1/
  run.json           # conf, cpus, module
  trial_dir.txt      # auto after train
  checkpoints/       # Ray Tune seed:* trials
  ray_tmp/
```

## Evaluation (`result.sh`)

All NSRDB configs use `evaluate_fullyear_mc.py` with `--date-set nsrdb_multiyear`:

```bash
./result.sh run_nsrdb --full
./result.sh run_strong --plot-only
./result.sh run_nsrdb --status
```

## Preflight

```bash
python scripts/verify_preflight.py --config examples.config.pv_tracking.stage3_multiyear_nsrdb_scenario
python scripts/verify_preflight.py --config examples.config.pv_tracking.conf3
```
