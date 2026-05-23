# Model-Based Policy Optimization (MBPO) — PV solar tracking

This repository trains a **model-based RL agent** to control a **two-axis solar tracker** in a **pvlib** simulator (`PVTracking-v0`). Training uses **MBPO** (ensemble dynamics + SAC). Evaluation compares the learned policy to simple baselines on the **same UTC episode grid** as training.

**Companion docs**

| Document | Contents |
|----------|----------|
| [docs/PV_TRACKING_TIME_AND_CHECKPOINTS.md](docs/PV_TRACKING_TIME_AND_CHECKPOINTS.md) | UTC daylight grid, checkpoint protocol, eval commands |
| [docs/PV_TRACKING_OBSERVATION_REVIEW.md](docs/PV_TRACKING_OBSERVATION_REVIEW.md) | Legacy 15-D vs physical 11-D observations |

---

## 1. What this project does

- **Simulates** irradiance and panel power with pvlib at **35°N, 106°W** (configurable in `mbpo/env/pv_tracking.py`).
- **Trains** on random calendar days in 2020 with stochastic weather (`weather_source='random'`).
- **Acts** with 2-D continuous commands: tilt and azimuth rate limits per 15-minute step.
- **Optimizes** per-step reward: collected energy minus a movement penalty.
- **Evaluates** with deterministic rollouts, optional baselines (`fixed_no_motion`, `sun_tracking`), CSV logs, and summary plots.

**Primary metric for reporting:** `total_energy_kwh` per episode (not peak step reward).

---

## 2. Repository layout

### 2.1 Source code (keep)

| Path | Role |
|------|------|
| `mbpo/env/pv_tracking.py` | Gym environment, pvlib, reward, `info` timestamps |
| `mbpo/static/pv_tracking.py` | MBPO termination + cyclic obs normalization |
| `mbpo/algorithms/mbpo.py` | MBPO training loop |
| `examples/config/pv_tracking/0.py` | Training hyperparameters + env kwargs |
| `examples/development/main.py` | Ray Tune entrypoint |
| `examples/development/base.py` | Variant builder; `MAX_PATH_LENGTH_PER_DOMAIN['PVTracking']` |
| `scripts/` | Health checks, training plots, evaluation (see §7) |
| `docs/` | Time standard, observation notes |

### 2.2 Generated artifacts (not in git; safe to delete and regenerate)

| Path | Produced by |
|------|-------------|
| `~/ray_mbpo/PVTracking/pv_tracking/seed:*/` | `mbpo run_local` — checkpoints, `progress.csv`, `params.json` |
| `evaluation/<run_name>/` | `scripts/evaluate_agent.py`, `compare_baselines.py` |

### 2.3 Training → evaluation data flow

```mermaid
flowchart LR
  config["examples/config/pv_tracking/0.py"]
  train["mbpo run_local examples.development"]
  ray["~/ray_mbpo/.../seed:*/"]
  ckpt["best_eval_checkpoint/"]
  eval["scripts/evaluate_agent.py"]
  out["evaluation/pv_daylight_utc/"]

  config --> train --> ray --> ckpt --> eval --> out
```

---

## 3. Episode MDP (current standard)

All training, evaluation, baselines, and plots share **one** time definition. There is **no** civil-time conversion layer.

| Setting | Value |
|---------|--------|
| `tz` | `UTC` |
| `start_time` | `13:30` |
| `periods` | `40` (timestamps) |
| `freq` | `15min` |
| Env steps per episode | **39** (`periods - 1`) |
| Clock span | **13:30 → 23:15 UTC** on the episode date |
| Site default | 35°N, 106°W, 1600 m |
| Policy observations | **11-D physical** (`observation_mode='physical'`) |

**Why this grid:** At this latitude, `06:00–21:45 UTC` put ~half of winter steps in night (zero power). The daylight window keeps almost all steps in sun; expect **0–2** gray-band steps at the start in December (pre-sunrise). GHI peak on the grid is typically near **17–20 UTC** (solar noon at this longitude).

**Changing** `start_time`, `periods`, or `tz` defines a **new MDP** → retrain and re-evaluate. Old checkpoints are not comparable.

**Align these when changing horizon:**

- `environment_kwargs.periods` → `epoch_length = periods - 1`
- `examples/development/base.py` → `MAX_PATH_LENGTH_PER_DOMAIN['PVTracking']`
- Evaluation → `--max-path-length` = same step count

---

## 4. Installation

```bash
git clone --recursive https://github.com/jannerm/mbpo.git
cd mbpo
conda env create -f environment/gpu-env.yml
conda activate mbpo
pip install -e viskit
pip install -e .
```

MuJoCo is **not** required for PV tracking (only for classic MBPO benchmarks).

---

## 5. Workflow overview

Run phases **in order** for a new experiment.

| Phase | Goal | Section |
|-------|------|---------|
| **A** | Confirm env, UTC grid, config wiring | [§5.1 Health checks](#51-health-checks-dry-runs) |
| **B** | Train MBPO | [§5.2 Training](#52-training) |
| **C** | Monitor run; pick checkpoint | [§5.3 Checkpoints](#53-checkpoints) |
| **D** | Hold-out evaluation + reports | [§5.4 Post-training evaluation](#54-post-training-evaluation) |

From the repository root:

```bash
conda activate mbpo
cd mbpo   # repository root
```

---

### 5.1 Health checks (dry runs)

Run **before** a long training job. None of these modify the codebase. Full script reference: [§7 Scripts directory](#7-scripts-directory).

**Minimum (run every time):**

| Check | Command | Pass criterion |
|-------|---------|----------------|
| UTC + horizon audit | `python scripts/verify_utc_uniformity.py` | `ALL CHECKS PASSED`; grid **13:30**, **40** periods, **39** steps |
| Env smoke test | `python scripts/check_pv_env.py --observation-mode physical` | Reset/step OK; obs dim **11** |
| Training wiring (no learning) | `mbpo run_example_dry examples.development --config=examples.config.pv_tracking.0 --gpus=0 --trial-gpus=0 --cpus=2 --trial-cpus=1` | Log shows `start_time='13:30' periods=40 (39 steps)` |

**Recommended when changing config or debugging pvlib:**

| Check | Command |
|-------|---------|
| MBPO schedule + env rollout | `python scripts/validate_pv_rollouts.py` |
| pvlib vs UTC grid | `python scripts/solar_time_sanity.py --date 2020-12-21 --env-rollout` |

**Optional (redundant with the two above):** `python scripts/audit_pv_workflow.py`

---

### 5.2 Training

```bash
mbpo run_local examples.development \
  --config=examples.config.pv_tracking.0 \
  --gpus=0 --trial-gpus=0 \
  --cpus=2 --trial-cpus=1
```

**What happens each epoch**

1. Sample a random day in `[start_date, end_date]` and build weather for that day.
2. Collect **39** real transitions per episode (`epoch_length`).
3. Train ensemble dynamics; generate short model rollouts; update SAC.
4. Periodically evaluate in-env; update `best_eval_checkpoint/` when `evaluation/return-average` improves.
5. Save `checkpoint_<epoch>/` and `latest_checkpoint/`.

**Config file:** `examples/config/pv_tracking/0.py` (edit `n_epochs`, dates, `movement_penalty`, etc.). Production runs often use `n_epochs` ≥ 150; the file may be set lower for quick tests.

**Ray output directory** (from `log_dir` + `exp_name` in config):

```text
~/ray_mbpo/PVTracking/pv_tracking/seed:<id>_<timestamp><hash>/
├── progress.csv
├── params.json
├── best_eval_checkpoint/
├── checkpoint_*/
└── latest_checkpoint/
```

---

### 5.3 Checkpoints

**Resolve the trial directory**

```bash
export TRIAL=$(ls -td ~/ray_mbpo/PVTracking/pv_tracking/seed:*/ | head -1)
export TRIAL="${TRIAL%/}"
export CKPT="$TRIAL/best_eval_checkpoint"
ls "$TRIAL/progress.csv" "$TRIAL/params.json" "$CKPT"
```

**Monitor training**

```bash
python scripts/plot_training_progress.py "$TRIAL"
# or: python scripts/plot_training_progress.py latest
tail -30 "$TRIAL/progress.csv"
```

| Checkpoint | Use |
|------------|-----|
| `best_eval_checkpoint/` | Default — best training-time eval return |
| `checkpoint_<N>/` | Compare a specific epoch |
| `latest_checkpoint/` | Resume / debug only |

**Hold-out ranking (energy, not training return)**

```bash
python scripts/select_best_checkpoint.py "$TRIAL" \
  --deterministic \
  --compare-baselines \
  --fixed-eval-dates 2020-12-07,2020-12-14,2020-12-21,2020-12-28 \
  --num-rollouts 4 \
  --max-path-length 39
```

Use the printed “Recommended for reporting” path, or `best_eval_checkpoint` if it wins on **total_energy_kwh**.

---

### 5.4 Post-training evaluation

**Required for comparable results**

- `--max-path-length 39` (must match `periods - 1`)
- `--eval-protocol inherit` (forces the same UTC daylight grid as training)
- `--deterministic` for reporting

**Canonical output directory name:** `evaluation/pv_daylight_utc` (daylight grid, current MDP).

#### Run type 1 — Standard hold-out (recommended)

Fixed December dates; learned policy + baselines; full reports and plots.

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_daylight_utc \
  --eval-protocol inherit \
  --max-path-length 39 \
  --num-rollouts 10 \
  --deterministic \
  --compare-baselines \
  --fixed-eval-dates 2020-12-07,2020-12-14,2020-12-21,2020-12-28
```

#### Run type 2 — Random days over the training year

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_random_2020 \
  --eval-protocol inherit \
  --max-path-length 39 \
  --num-rollouts 10 \
  --deterministic \
  --compare-baselines
```

#### Run type 3 — Hold-out date range (e.g. all of December)

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_holdout_dec2020 \
  --eval-protocol inherit \
  --max-path-length 39 \
  --num-rollouts 10 \
  --deterministic \
  --compare-baselines \
  --test-start-date 2020-12-01 \
  --test-end-date 2020-12-31
```

#### Run type 4 — Seasonal solstice / equinox dates

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_seasonal_2020 \
  --eval-protocol inherit \
  --max-path-length 39 \
  --num-rollouts 4 \
  --deterministic \
  --compare-baselines \
  --fixed-eval-dates 2020-03-21,2020-06-21,2020-09-22,2020-12-21
```

#### Run type 5 — Clearsky weather only

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_clearsky_holdout \
  --eval-protocol inherit \
  --max-path-length 39 \
  --num-rollouts 10 \
  --deterministic \
  --compare-baselines \
  --eval-weather-source clearsky \
  --fixed-eval-dates 2020-12-07,2020-12-14,2020-12-21,2020-12-28
```

#### Run type 6 — Baselines only (no neural policy forward pass)

```bash
python scripts/compare_baselines.py "$CKPT" \
  --outdir evaluation/pv_baselines_only \
  --eval-protocol inherit \
  --max-path-length 39 \
  --num-rollouts 10 \
  --deterministic \
  --fixed-eval-dates 2020-12-07,2020-12-14,2020-12-21,2020-12-28
```

#### Run type 7 — Regenerate one rollout figure from CSV

```bash
python scripts/plot_rollout_trajectory.py \
  --csv evaluation/pv_daylight_utc/rollouts/rollout_1.csv \
  --outdir evaluation/pv_daylight_utc/rollout_plots_replot
```

`plot_rollout_trajectory.py` produces a **simple** 4-panel plot. The **rich** 6-panel UTC plot (power, energy, tilt, azimuth, actions, reward + night bands) is written only by `evaluate_agent.py`.

---

## 6. Evaluation outputs (essential vs optional)

Each `evaluate_agent.py` run creates a directory (e.g. `evaluation/pv_daylight_utc/`).

### 6.1 Essential (keep one canonical run for papers / reports)

| Artifact | Role |
|----------|------|
| `evaluation_summary.txt` / `.json` | Mean reward, **total_energy_kwh**, movement, season breakdown |
| `eval_scenario_confirmation.txt` | Fairness: same dates, seeds, UTC grid, baselines |
| `reward_time_analysis.txt` | Peak power/reward UTC hours; solar-altitude aggregates |
| `evaluation_method_comparison.png` | Policy vs baselines (requires `--compare-baselines`) |
| `evaluation_reward_by_solar_altitude.png` | Physics windows (preferred for “midday sun”) |
| `rollouts/rollout_<n>.csv` | Per-step `clock_hour_utc`, `power_w`, `solar_altitude_deg`, actions |

### 6.2 Useful but regenerable

| Artifact | Notes |
|----------|--------|
| `evaluation_reward_by_time_window.png` | UTC clock bins on this grid |
| `evaluation_rewards.png` | Per-rollout reward bars |
| `evaluation_by_season.png` | Skipped with `--no-report-by-season` |
| `rollout_plots/rollout_<n>_combined.png` | Large; regenerable from CSV via `evaluate_agent.py` |
| `baseline_rollouts/**/*.csv` | Regenerable with `--compare-baselines` |

### 6.3 Plot interpretation (UTC only)

- X-axis: **`clock_hour_utc`** (post-step wall clock in UTC).
- Gray bands: **`solar_altitude_deg ≤ 0`** (night / below horizon).
- Peak power often near **17–20 UTC** in winter on this grid = solar noon at 106°W, not “evening” in local civil time.
- Do **not** use peak step reward as the main metric; use **total_energy_kwh**.

---

## 7. Scripts directory

All paths below are from the **repository root** (`cd mbpo`). Scripts are grouped by role in the PV workflow.

### 7.1 Summary table

| File | Role | Verdict |
|------|------|---------|
| `eval_utils.py` | Shared library (env build, CSV, plots, reports) | **Keep** — imported by eval scripts; not run directly |
| `pv_trial_paths.py` | Resolve `latest` Ray trial directory | **Keep** — library for plotting / checkpoint tools |
| `verify_utc_uniformity.py` | Pre-flight + post-change audit | **Essential** |
| `check_pv_env.py` | Env smoke test (obs, weather) | **Essential** |
| `evaluate_agent.py` | Full post-training eval + plots + CSV | **Essential** |
| `evaluate_agent_advanced.py` | Aligned learned vs baseline, mean±std bands, dashboard PDF | **Optional** — use after canonical `evaluate_agent.py` when comparing dynamics on one figure |
| `plot_training_progress.py` | Training curves from `progress.csv` | **Essential** |
| `select_best_checkpoint.py` | Rank checkpoints on hold-out energy | **Recommended** |
| `validate_pv_rollouts.py` | MBPO rollout schedule + env validation | **Recommended** |
| `solar_time_sanity.py` | pvlib / UTC grid diagnostics | **Recommended** |
| `compare_baselines.py` | Baselines only (no policy network) | **Optional** — subset of `evaluate_agent.py` |
| `plot_rollout_trajectory.py` | Re-plot one rollout CSV (4 panels) | **Optional** — `evaluate_agent.py` already writes richer 6-panel plots |
| `export_policy_weights.py` | Extract `policy_weights.pkl` from old `checkpoint.pkl` | **Optional** — only if a checkpoint lacks `policy_weights.pkl` |
| `audit_pv_workflow.py` | Combined physical-obs + FakeEnv smoke test | **Optional** — overlaps `check_pv_env` + `verify_utc_uniformity` |
| `evaluate_and_viskit.py` | Shell wrapper: eval + training plots + viskit | **Can remove** — duplicates README commands; prefer direct calls |
| `plot_ray_results.py` | Parse pasted Ray **status** text into PNG | **Can remove** — not part of PV train/eval; use `plot_training_progress.py` instead |

**Safe to delete from the repo (no impact on train/eval):** `evaluate_and_viskit.py`, `plot_ray_results.py`.  
**Keep but rarely run:** `export_policy_weights.py`, `audit_pv_workflow.py`, `plot_rollout_trajectory.py`.

---

### 7.2 Libraries (do not run directly)

#### `eval_utils.py`

- **Purpose:** Single implementation for evaluation env construction (`apply_pv_utc_schedule`), rollout CSV columns, aggregate plots, `eval_scenario_confirmation.txt`, `reward_time_analysis.txt`, UTC time windows.
- **Used by:** `evaluate_agent.py`, `compare_baselines.py`, `select_best_checkpoint.py`, `verify_utc_uniformity.py`, `solar_time_sanity.py`.
- **Do not delete.**

#### `pv_trial_paths.py`

- **Purpose:** Resolve `latest` or a path to a Ray folder `seed:…/` (must contain `params.json`).
- **Used by:** `plot_training_progress.py`, `select_best_checkpoint.py`.
- **Do not delete.**

---

### 7.3 Essential scripts

#### `verify_utc_uniformity.py` — UTC grid and horizon audit

Confirms `mbpo/env/pv_tracking.py`, config, `examples/development/base.py`, and eval scripts all agree on **13:30–23:15 UTC**, **39** steps, and plot axis = `clock_hour_utc`.

```bash
python scripts/verify_utc_uniformity.py
python scripts/verify_utc_uniformity.py --config examples/config/pv_tracking/0.py
```

Run after any change to `start_time`, `periods`, `epoch_length`, or `MAX_PATH_LENGTH_PER_DOMAIN`.

#### `check_pv_env.py` — environment smoke test

One reset/step, observation size, optional weather-table check.

```bash
python scripts/check_pv_env.py --observation-mode physical
python scripts/check_pv_env.py --observation-mode physical --validate-weather
python scripts/check_pv_env.py --observation-mode physical --log-obs
```

#### `evaluate_agent.py` — post-training evaluation (main)

Loads checkpoint, runs rollouts, writes `evaluation/<outdir>/` (summaries, CSV, PNG, optional baselines). This is the **primary** post-processing entry point.

```bash
export CKPT="$TRIAL/best_eval_checkpoint"

python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_daylight_utc \
  --eval-protocol inherit \
  --max-path-length 39 \
  --num-rollouts 10 \
  --deterministic \
  --compare-baselines \
  --fixed-eval-dates 2020-12-07,2020-12-14,2020-12-21,2020-12-28
```

Important options:

| Option | Purpose |
|--------|---------|
| `--outdir` | Output folder (use `evaluation/pv_daylight_utc` for canonical run) |
| `--max-path-length 39` | Must match `periods - 1` |
| `--eval-protocol inherit` | Same UTC daylight grid as training |
| `--deterministic` | Greedy policy for reporting |
| `--compare-baselines` | Also run `fixed_no_motion` and `sun_tracking` |
| `--fixed-eval-dates` | Comma-separated `YYYY-MM-DD` (fair comparison) |
| `--test-start-date` / `--test-end-date` | Random hold-out days in a range |
| `--eval-weather-source clearsky` | Override weather for eval only |
| `--no-report-by-season` | Skip season breakdown plots |

#### `evaluate_agent_advanced.py` — aligned baselines + ensemble bands + dashboard

Same real-env + policy stack as `evaluate_agent.py` (no MBPO dynamics model). Use when you need **learned vs baseline on one time series**, **mean ± std over replicates** vs UTC hour, or a **single `dashboard.pdf`** instead of many per-rollout PNGs.

```bash
python scripts/evaluate_agent_advanced.py "$CKPT" \
  --outdir evaluation/pv_daylight_utc_advanced \
  --eval-protocol inherit \
  --max-path-length 39 \
  --fixed-eval-dates 2020-12-07,2020-12-21 \
  --replicates-per-date 8 \
  --policy-mode deterministic \
  --compare-weather-sources
```

| Option | Purpose |
|--------|---------|
| `--fixed-eval-dates` | **Required** — one aligned + ensemble figure set per date |
| `--replicates-per-date` | Rollouts per date for mean ± std (weather varies via env seed) |
| `--policy-mode` | `deterministic` (default) or `stochastic` (sampled SAC actions) |
| `--eval-weather-source` | `random` (default) or `clearsky` for main + ensemble runs |
| `--compare-weather-sources` | Overlay random vs clearsky learned policy (same seed/date) |
| `--vary-init-orientation` | Randomize initial panel pose across replicates |
| `--no-dashboard` | Skip multi-page PDF (keep PNGs under `aligned/`, `ensemble/`) |

Outputs: `aligned/aligned_<date>.png`, `ensemble/ensemble_<date>_power.png`, optional `weather_compare/`, `dashboard.pdf`, plus CSVs and `eval_scenario_confirmation.txt` via `eval_utils.py`.

#### `plot_training_progress.py` — training monitor

Plots metrics from Ray `progress.csv` (not from `evaluation/`).

```bash
python scripts/plot_training_progress.py latest
python scripts/plot_training_progress.py "$TRIAL" --outdir "$TRIAL/training_plots"
```

Accepts trial dir, `progress.csv` path, or `latest`.

---

### 7.4 Recommended scripts

#### `select_best_checkpoint.py` — checkpoint ranking

Compares `checkpoint_*` and `best_eval_checkpoint/` on hold-out **energy** (not training return alone).

```bash
python scripts/select_best_checkpoint.py latest \
  --deterministic \
  --compare-baselines \
  --fixed-eval-dates 2020-12-07,2020-12-14,2020-12-21,2020-12-28 \
  --num-rollouts 4 \
  --max-path-length 39
```

#### `validate_pv_rollouts.py` — MBPO + env deep check

Prints imagined rollout length schedule from config, episode timing, observation scaling, and short random-action rollouts.

```bash
python scripts/validate_pv_rollouts.py
python scripts/validate_pv_rollouts.py --config examples/config/pv_tracking/0.py
```

Use when tuning `rollout_schedule`, `max_model_rollout_length`, or observation mode.

#### `solar_time_sanity.py` — pvlib vs episode grid

Tables of solar altitude / GHI on the UTC grid; optional one-day env rollout at zero action.

```bash
python scripts/solar_time_sanity.py --date 2020-12-21
python scripts/solar_time_sanity.py --date 2020-12-21 --env-rollout
```

Use when interpreting peak power times on plots or verifying a new `start_time`.

---

### 7.5 Optional scripts

#### `compare_baselines.py`

Same env and metrics as `evaluate_agent.py`, but **only** baselines (faster if you do not need the learned policy).

```bash
python scripts/compare_baselines.py "$CKPT" \
  --outdir evaluation/pv_baselines_only \
  --eval-protocol inherit \
  --max-path-length 39 \
  --num-rollouts 10 \
  --deterministic \
  --fixed-eval-dates 2020-12-07,2020-12-14,2020-12-21,2020-12-28
```

If you already ran `evaluate_agent.py` with `--compare-baselines`, this is **redundant**.

#### `plot_rollout_trajectory.py`

Reads one `rollouts/rollout_N.csv` and writes a **simple** 4-panel figure (power, tilt, azimuth, reward). The main eval script already produces **6-panel** UTC plots with night shading in `rollout_plots/`.

```bash
python scripts/plot_rollout_trajectory.py \
  --csv evaluation/pv_daylight_utc/rollouts/rollout_1.csv \
  --outdir evaluation/pv_daylight_utc/rollout_plots_replot
```

#### `export_policy_weights.py`

One-time helper: create `policy_weights.pkl` from an older `checkpoint.pkl` that does not already export weights.

```bash
python scripts/export_policy_weights.py "$TRIAL/checkpoint_51"
```

Modern training saves `policy_weights.pkl` automatically; skip unless loading a legacy checkpoint fails.

#### `audit_pv_workflow.py`

Quick check: config uses physical 11-D obs, legacy 15-D still registers, FakeEnv tensor shapes. Overlaps `check_pv_env.py` + `verify_utc_uniformity.py`.

```bash
python scripts/audit_pv_workflow.py
```

---

### 7.6 Scripts you can remove (not required for workflow)

#### `evaluate_and_viskit.py`

Subprocess wrapper around `evaluate_agent.py` + `plot_training_progress.py` + optional viskit server. Everything it does is already documented in §5 with explicit commands. It also writes `training_plots/` **inside** `evaluation/<outdir>/`, which duplicates curves that belong under the Ray trial directory.

**Replacement:** use §5.2–§5.4 commands directly.

#### `plot_ray_results.py`

Parses **pasted Ray trial status text** (not `progress.csv`) into status/metric PNGs. Unrelated to PV physics or `evaluate_agent.py`. For training curves, use `plot_training_progress.py`.

```bash
# Only if you have a Ray status dump file:
python scripts/plot_ray_results.py --input ray_status.txt --outdir /tmp/ray_plots
```

---

### 7.7 Minimal workflow (scripts only)

```text
Phase A:  verify_utc_uniformity.py  →  check_pv_env.py
Phase B:  mbpo run_local …          →  plot_training_progress.py latest
Phase C:  select_best_checkpoint.py (optional)
Phase D:  evaluate_agent.py  →  open evaluation/pv_daylight_utc/
```

---

## 8. Configuration reference

**Training:** `examples/config/pv_tracking/0.py`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_epochs` | (see file) | Training epochs; increase for convergence |
| `epoch_length` | `39` | Real env steps per epoch |
| `n_initial_exploration_steps` | `390` | `≈ 10 × epoch_length` |
| `observation_mode` | `physical` | 11-D obs (legacy 15-D needs old checkpoints) |
| `start_date` / `end_date` | 2020 full year | Training day pool |
| `weather_source` | `random` | Stochastic clouds for training |
| `movement_penalty` | `0.0001` | Motion cost weight |

**Reward** (`mbpo/env/pv_tracking.py`):

```text
energy_kwh     = power_W × (15 min in hours) / 1000
movement_cost  = movement_penalty × (|Δtilt|/max_Δtilt + |Δazimuth|/max_Δazimuth)
reward         = energy_kwh - movement_cost
```

---

## 9. Evaluation directory policy

**On disk (local):** keep one canonical run:

```text
evaluation/
├── .gitkeep
└── pv_daylight_utc/          # full hold-out eval (current UTC daylight MDP)
    ├── evaluation_summary.txt
    ├── eval_scenario_confirmation.txt
    ├── reward_time_analysis.txt
    ├── evaluation_*.png
    ├── rollouts/
    ├── rollout_plots/
    └── baseline_rollouts/
```

New runs use a **new** `--outdir` name (e.g. `evaluation/pv_random_2020`). Delete old folders when finished so only `pv_daylight_utc` remains for day-to-day work.

**In git:** `evaluation/*` is ignored except `evaluation/.gitkeep` (see `.gitignore`). Checkpoints stay under `~/ray_mbpo/…`, not in the repo.

**Optional space savings** inside `pv_daylight_utc` (regenerable):

| Remove | Regenerate with |
|--------|-----------------|
| `rollout_plots/*.png` | `evaluate_agent.py` (same `--outdir`) |
| `baseline_rollouts/` | `--compare-baselines` or full `evaluate_agent.py` |

---

## 10. Quick command reference

```bash
conda activate mbpo
cd mbpo

# --- Phase A ---
python scripts/verify_utc_uniformity.py
python scripts/check_pv_env.py --observation-mode physical

# --- Phase B ---
mbpo run_local examples.development --config=examples.config.pv_tracking.0 \
  --gpus=0 --trial-gpus=0 --cpus=2 --trial-cpus=1

# --- Phase C ---
export TRIAL=$(ls -td ~/ray_mbpo/PVTracking/pv_tracking/seed:*/ | head -1)
export TRIAL="${TRIAL%/}"
export CKPT="$TRIAL/best_eval_checkpoint"
python scripts/plot_training_progress.py "$TRIAL"

# --- Phase D ---
python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_daylight_utc \
  --eval-protocol inherit \
  --max-path-length 39 \
  --num-rollouts 10 \
  --deterministic \
  --compare-baselines \
  --fixed-eval-dates 2020-12-07,2020-12-14,2020-12-21,2020-12-28
```

---

## 11. Extending to other environments

1. Add `mbpo/env/<name>.py` and register in `mbpo/env/__init__.py`.
2. Add `mbpo/static/<domain>.py` termination function.
3. Add `examples/config/<domain>/0.py`.
4. Set `MAX_PATH_LENGTH_PER_DOMAIN` in `examples/development/base.py` if episode length ≠ 1000.

---

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

SAC from [softlearning](https://github.com/rail-berkeley/softlearning). Dynamics modeling from [PETS](https://github.com/kchua/handful-of-trials).
