# Viskit And Ray Localhost Guide

This file explains how to monitor MBPO training from your browser on `localhost` using:

- `viskit` for interactive training curves
- Ray optional web UI / Tune server
- `scripts/plot_training_progress.py` for simple saved PNG updates

This guide is written for the current repo layout and commands.

## 1. What each tool is for

### Viskit

Use Viskit when you want an interactive browser view over one or more Ray trial folders that contain:

- `progress.csv`
- `params.json`

Best use:

- compare multiple trials
- inspect training curves while training is running
- filter runs by parameters

### Ray web UI / Tune server

This repo forwards two Ray/Tune flags during training:

- `--include-webui=True`
- `--with-server=True`

From the code:

- `ray.init(... include_webui=example_args.include_webui ...)`
- `tune.run_experiments(... with_server=example_args.with_server, server_port=4321 ...)`

In practice:

- the Tune server is optional and mainly for Tune client/server features
- the Ray web UI depends on the installed Ray version
- the most reliable live plotting workflow in this repo is still `plot_training_progress.py` plus `viskit`

### `plot_training_progress.py`

Use this when you want quick local PNG files that refresh as training progresses.

It reads `progress.csv` and writes:

- `evaluation_return-average.png`
- `training_return-average.png`
- `model_val_loss.png`
- `training_summary.txt`

## 2. Required paths

Repo root:

- `/home/ecer/PVRL/mbpo`

Default Ray trial root:

- `~/ray_mbpo/PVTracking/pv_tracking`

Typical Ray trial path:

- `~/ray_mbpo/PVTracking/pv_tracking/seed:<id>_<timestamp>.../`

Useful shell variables:

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

export ROOT=/home/ecer/PVRL/mbpo
export RAY_ROOT="$HOME/ray_mbpo/PVTracking/pv_tracking"
export TRIAL=$(ls -td ~/ray_mbpo/PVTracking/pv_tracking/seed:*/ | head -1)
export TRIAL="${TRIAL%/}"
```

## 3. Start training with optional Ray localhost services

For the current clean Stage 3 workflow:

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

mbpo run_local examples.development \
  --config=examples.config.pv_tracking.stage3_fullyear_random_clean_split \
  --gpus=0 \
  --trial-gpus=0 \
  --cpus=4 \
  --trial-cpus=2 \
  --include-webui=True \
  --with-server=True
```

What this does:

- starts the training run
- asks Ray to include its web UI if supported by your installed Ray
- starts the Tune server on port `4321`

Important:

- this repo does not hardcode a Ray dashboard browser URL
- check the training terminal output for the printed Ray UI address
- the Tune server port is fixed in this repo to `4321`

## 4. Use Viskit on localhost

### 4.1 View all PV trials

This is the most common Viskit command:

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

viskit ~/ray_mbpo/PVTracking/pv_tracking --port 6008
```

Then open:

- [http://localhost:6008](http://localhost:6008)

The underlying Viskit code serves on `0.0.0.0` and prints:

- `View http://localhost:<port> in your browser`

### 4.2 View one single trial only

If you only want one run:

```bash
viskit "$TRIAL" --port 6008
```

### 4.3 Alternative direct Python command

If the `viskit` command is not found in `PATH`, use:

```bash
python viskit/viskit/frontend.py ~/ray_mbpo/PVTracking/pv_tracking --port 6008
```

or for one trial:

```bash
python viskit/viskit/frontend.py "$TRIAL" --port 6008
```

### 4.4 Use another port if 6008 is busy

```bash
viskit ~/ray_mbpo/PVTracking/pv_tracking --port 6009
```

Then open:

- [http://localhost:6009](http://localhost:6009)

## 5. Use live PNG plots on localhost

If you prefer simple image files instead of a live web app:

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

python scripts/plot_training_progress.py "$TRIAL" \
  --outdir training_plots/stage3_clean_split_latest
```

This writes:

- `training_plots/stage3_clean_split_latest/evaluation_return-average.png`
- `training_plots/stage3_clean_split_latest/training_return-average.png`
- `training_plots/stage3_clean_split_latest/model_val_loss.png`
- `training_plots/stage3_clean_split_latest/training_summary.txt`

### Auto-refresh the saved plots every 60 seconds

```bash
watch -n 60 "bash -lc 'cd /home/ecer/PVRL/mbpo && source /home/ecer/miniconda3/etc/profile.d/conda.sh && conda activate mbpo && python scripts/plot_training_progress.py \"\$TRIAL\" --outdir training_plots/stage3_clean_split_latest >/dev/null 2>&1'"
```

Then open the generated PNGs from your file browser or IDE preview.

## 6. Recommended localhost workflow

The most useful practical setup is:

### Terminal 1: training

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

mbpo run_local examples.development \
  --config=examples.config.pv_tracking.stage3_fullyear_random_clean_split \
  --gpus=0 \
  --trial-gpus=0 \
  --cpus=4 \
  --trial-cpus=2 \
  --include-webui=True \
  --with-server=True
```

### Terminal 2: pick the active trial

```bash
export TRIAL=$(ls -td ~/ray_mbpo/PVTracking/pv_tracking/seed:*/ | head -1)
export TRIAL="${TRIAL%/}"
echo "$TRIAL"
```

### Terminal 3: Viskit

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

viskit "$TRIAL" --port 6008
```

Open:

- [http://localhost:6008](http://localhost:6008)

### Terminal 4: static plot refresher

```bash
watch -n 60 "bash -lc 'cd /home/ecer/PVRL/mbpo && source /home/ecer/miniconda3/etc/profile.d/conda.sh && conda activate mbpo && python scripts/plot_training_progress.py \"\$TRIAL\" --outdir training_plots/stage3_clean_split_latest >/dev/null 2>&1'"
```

This combination gives you:

- browser-based interactive curves from Viskit
- optional Ray/Tune localhost services
- saved PNG snapshots in `training_plots/`

## 7. Optional combined eval + Viskit helper

This repo also contains:

- `scripts/evaluate_and_viskit.py`

It can:

1. evaluate a checkpoint
2. generate training plots
3. start Viskit

Example:

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

python scripts/evaluate_and_viskit.py \
  --ckpt-dir "$TRIAL/best_eval_checkpoint" \
  --outdir evaluation/pv_stage3_final_clean_test \
  --num-rollouts 10 \
  --max-path-length 39 \
  --deterministic \
  --port 6008
```

But for the main scientific workflow, the repo documentation already prefers direct commands over this wrapper.

## 8. WSL / Linux localhost note

On your current setup, these services are local web servers. Open them from the browser using:

- `http://localhost:6008` for Viskit
- the URL printed by Ray for its web UI, if available
- `http://localhost:4321` for the Tune server if you explicitly enabled `--with-server=True`

If a port is busy, change it with `--port`.

## 9. Troubleshooting

### `viskit: command not found`

Run:

```bash
cd /home/ecer/PVRL/mbpo
source /home/ecer/miniconda3/etc/profile.d/conda.sh
conda activate mbpo

python viskit/viskit/frontend.py ~/ray_mbpo/PVTracking/pv_tracking --port 6008
```

If needed, reinstall the editable package:

```bash
pip install -e viskit
```

### Browser opens but no curves appear

Check that the target folder really contains:

- `progress.csv`
- `params.json`

Quick check:

```bash
python - "$TRIAL" <<'PY'
import os
import sys
p = sys.argv[1]
print('progress.csv:', os.path.isfile(os.path.join(p, 'progress.csv')))
print('params.json :', os.path.isfile(os.path.join(p, 'params.json')))
PY
```

### Port already in use

Use another port:

```bash
viskit "$TRIAL" --port 6010
```

### Ray UI does not appear

That usually means one of these:

- your installed Ray version does not expose the UI the same way
- the UI address was printed in the training log and needs to be copied from there
- you started training without `--include-webui=True`

This does not block training. You can still use:

- `viskit`
- `plot_training_progress.py`
- `progress.csv`

## 10. Short command reference

### Start clean Stage 3 training with optional Ray localhost services

```bash
mbpo run_local examples.development \
  --config=examples.config.pv_tracking.stage3_fullyear_random_clean_split \
  --gpus=0 --trial-gpus=0 --cpus=4 --trial-cpus=2 \
  --include-webui=True \
  --with-server=True
```

### Start Viskit for all PV trials

```bash
viskit ~/ray_mbpo/PVTracking/pv_tracking --port 6008
```

### Start Viskit for one trial

```bash
viskit "$TRIAL" --port 6008
```

### Generate static plots once

```bash
python scripts/plot_training_progress.py "$TRIAL" \
  --outdir training_plots/stage3_clean_split_latest
```

### Refresh static plots continuously

```bash
watch -n 60 "bash -lc 'cd /home/ecer/PVRL/mbpo && source /home/ecer/miniconda3/etc/profile.d/conda.sh && conda activate mbpo && python scripts/plot_training_progress.py \"\$TRIAL\" --outdir training_plots/stage3_clean_split_latest >/dev/null 2>&1'"
```

That is the recommended localhost plotting setup for this project.
