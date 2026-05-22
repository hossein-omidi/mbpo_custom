# PV tracking: UTC time standard and checkpoint selection

**Step-by-step commands (training → eval → plots):** see the main [README](../README.md#pv-tracking-end-to-end-workflow) sections *Step 0–3* and *Command cheat sheet*.

**Automated audit:** `python scripts/verify_utc_uniformity.py` checks constants, training/eval env, step indexing, plot axes, and that `local_day` is disabled.

## Why UTC (not local Denver time)

- **pvlib** uses `Location.tz` on every `DatetimeIndex` for solar position, weather, and `ModelChain` power. Keeping `tz='UTC'` avoids extra conversions and matches the default env (`mbpo/env/pv_tracking.py`, `PV_TIMEZONE = 'UTC'`).
- **Train / test / plots** must share one MDP: same `tz`, `start_time`, `periods`, `freq`, and horizon (`max_path_length = 63`).
- **Efficiency:** one timezone end-to-end; eval scripts force the UTC grid via `apply_pv_utc_schedule()` regardless of stale variant fields.
- **Reliability:** plot x-axis uses `info['clock_hour_utc']` only — no `America/Denver` conversion that made winter rollouts look like “0–6 h with zero power.”

## Episode grid (project standard)

| Field | Value |
|-------|--------|
| `tz` | `UTC` |
| `start_time` | `06:00` |
| `periods` | `64` (timestamps) |
| `freq` | `15min` |
| Env steps per day | `63` (`periods - 1`) |
| Clock span | **06:00–21:45 UTC** on the episode date |

At **35°N, 106°W** in **December**, the sun is up roughly **14:15–22:00 UTC** on the episode grid. Steps from **06:00–~14:00 UTC** show **zero power** because **solar altitude ≤ 0°** (night at the site), not because of a timezone conversion bug.

Run `python scripts/solar_time_sanity.py --date 2020-12-21` or read `eval_scenario_confirmation.txt` after eval — it now lists **sunrise/sunset UTC** vs the fixed grid.

### Why `start_time='06:00'` and `periods=64`?

The project uses a **fixed wall-clock window** in UTC (06:00–21:45), not “local sunrise to sunset.” That choice is **consistent** across train, test, baselines, and pvlib, but in winter ~**half the steps can be night**. Changing `start_time` / `periods` to trim night hours is a **new MDP** and requires **retraining**.

Peak power near **17–20h UTC** in December is **solar noon** at this longitude (~11:00–13:00 US Mountain), not “evening” in local civil time.

## Physics vs clock labels

- **UTC clock windows** (`morning` / `midday` / … in `reward_time_analysis.txt`): labels are wall-clock UTC; do not read them as Denver local midday.
- **Solar-altitude windows** (`evaluation_reward_by_solar_altitude.png`): use these for sun-up / peak-sun interpretation.

## Evaluation commands

```bash
cd /home/ecer/PVRL/mbpo
conda activate mbpo

python scripts/evaluate_agent.py CHECKPOINT \
  --eval-protocol inherit \
  --max-path-length 63 \
  --deterministic \
  --compare-baselines \
  --fixed-eval-dates 2020-12-07,2020-12-14,2020-12-21,2020-12-28 \
  --outdir evaluation/pv_holdout_utc

python scripts/compare_baselines.py CHECKPOINT \
  --eval-protocol inherit --max-path-length 63 --deterministic \
  --fixed-eval-dates 2020-12-21 --num-rollouts 4

python scripts/solar_time_sanity.py --date 2020-12-21 --env-rollout
```

`--eval-protocol inherit`, `utc`, and `legacy_utc` all apply the same UTC grid. `local_day` is **disabled** (different MDP).

## Checkpoint selection protocol

Training (`examples/development/main.py`) tracks **`monitor_metric`** (default: `evaluation/return-average` from periodic eval rollouts in the **training** env). When the metric improves, it saves **`best_eval_checkpoint/`**.

### Step 1 — Training-time best (automatic)

- **Criterion:** highest `evaluation/return-average` seen during training (same reward definition: energy − movement).
- **Artifact:** `best_eval_checkpoint/` next to `progress.csv` and epoch checkpoints.
- **Use when:** you want the policy that generalized best on the **training evaluator** (random days, same UTC MDP).

### Step 2 — Scientific confirmation (required for papers)

Training return can favor “safe” policies (low movement, late-day reward). Confirm on **hold-out** days:

1. **Primary:** mean **`total_energy_kwh`** vs `sun_tracking` and `fixed_no_motion` on fixed or held-out dates (`--fixed-eval-dates` or `--test-start-date` / `--test-end-date`).
2. **Secondary:** mean **movement cost**, tilt/azimuth traces, and **solar-altitude** window aggregates.
3. **Diagnostics:** `reward_time_analysis.txt` peak-power **UTC hour** vs solar altitude at that step (should align).
4. **Reject** a checkpoint if energy is clearly below baselines on the same seeds/dates (`eval_scenario_confirmation.txt`).

### Step 3 — Compare epoch checkpoints (optional)

If `best_eval_checkpoint` underperforms on hold-out energy:

```bash
python scripts/select_best_checkpoint.py EXPERIMENT_ROOT \
  --fixed-eval-dates 2020-12-07,2020-12-14,2020-12-21,2020-12-28 \
  --compare-baselines
```

This ranks `checkpoint_*` and `best_eval_checkpoint` by hold-out **total energy**, not training return alone.

### What not to use

- Peak **step reward** time (biased by movement penalty).
- UTC clock “midday” window mean power in winter (often night at the site).
- A checkpoint trained with a **different** `tz` / grid without retraining.

## Files that encode the standard

| Component | Location |
|-----------|----------|
| Env + info fields | `mbpo/env/pv_tracking.py` |
| Training config | `examples/config/pv_tracking/0.py` |
| MBPO solar-time decode | `mbpo/static/pv_tracking.py` (`DEFAULT_TZ='UTC'`) |
| Eval / plots / CSV | `scripts/eval_utils.py`, `evaluate_agent.py`, `compare_baselines.py` |
