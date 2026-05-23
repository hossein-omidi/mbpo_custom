# PV tracking: UTC time standard and checkpoint selection

**Step-by-step commands (training → eval → plots):** see the main [README](../README.md#pv-tracking-end-to-end-workflow) sections *Step 0–3* and *Command cheat sheet*.

**Automated audit:** `python scripts/verify_utc_uniformity.py` checks constants, training/eval env, step indexing, plot axes, and that `local_day` is disabled.

## Why UTC only (no civil-time conversion)

- **pvlib** uses `Location.tz` on every `DatetimeIndex` for solar position, weather, and power. The project keeps `tz='UTC'` and reads wall-clock labels directly from those timestamps (`info['clock_hour_utc']`).
- **Train / test / plots** must share one MDP: same `tz`, `start_time`, `periods`, `freq`, and horizon (`max_path_length = periods - 1`).
- **No conversion layer:** eval scripts force the UTC grid via `apply_pv_utc_schedule()`; plots and CSVs never use local civil time.

## Daylight episode grid (current standard)

| Field | Value |
|-------|--------|
| `tz` | `UTC` |
| `start_time` | `13:30` |
| `periods` | `40` (timestamps) |
| `freq` | `15min` |
| Env steps per day | `39` (`periods - 1`) |
| Clock span | **13:30–23:15 UTC** on the episode date |

At **35°N, 106°W** this window was chosen so training and evaluation spend most steps in **daylight through early evening** (pvlib solar altitude &gt; 0°), without any timezone conversion code.

Typical geometry on hold-out December dates:

- Sunrise on grid: ~**14.25 UTC** (1–2 pre-sunrise steps at 13:30–14:15 are expected).
- GHI peak: ~**19.0 UTC** (solar noon at this longitude).
- Episode end: **23:15 UTC** while the sun is still above the horizon in winter.

Run `python scripts/solar_time_sanity.py --date 2020-12-21` or read `eval_scenario_confirmation.txt` after eval for per-date sunrise/sunset UTC on this grid.

### Why not seasonal `start_time` shifting?

Seasonal UTC offsets (different `start_time` per month) would require extra scheduling logic and retraining per season. A **single fixed UTC window** (`13:30`–`23:15`) is simpler, keeps train/test/plots identical, and removes ~50% “night-only” steps from the old `06:00`–`21:45` grid in winter while still covering June–December training days. Tuning is done by changing **UTC numbers only** in `mbpo/env/pv_tracking.py` and `examples/config/pv_tracking/0.py`, then **retraining**.

### Scenario settings vs bugs

| Observation | Type | Action |
|-------------|------|--------|
| Zero power when `solar_altitude_deg ≤ 0` | Expected physics | Gray bands on rollout plots |
| 1–2 zero-power steps at episode start in December | Grid margin before sunrise | Accept or nudge `start_time` to `14:00` (new MDP) |
| Peak power near 17–20h UTC in December | Solar noon at 106°W | Read x-axis as **UTC** |
| Policy below `sun_tracking` energy | Learning / reward design | More training or algorithm tuning |
| UTC clock bins “morning/midday” in reports | Reporting labels on this grid | Prefer `evaluation_reward_by_solar_altitude.png` |

## Physics vs clock labels

- **UTC clock windows** (`morning` / `midday` / … in `reward_time_analysis.txt`): bins on the **13:30–23:15** grid only.
- **Solar-altitude windows** (`evaluation_reward_by_solar_altitude.png`): use these for sun-up / peak-sun interpretation.

## Evaluation commands

```bash
cd /home/ecer/PVRL/mbpo
conda activate mbpo

python scripts/evaluate_agent.py CHECKPOINT \
  --eval-protocol inherit \
  --max-path-length 39 \
  --deterministic \
  --compare-baselines \
  --fixed-eval-dates 2020-12-07,2020-12-14,2020-12-21,2020-12-28 \
  --outdir evaluation/pv_daylight_utc

python scripts/compare_baselines.py CHECKPOINT \
  --eval-protocol inherit --max-path-length 39 --deterministic \
  --fixed-eval-dates 2020-12-21 --num-rollouts 4

python scripts/solar_time_sanity.py --date 2020-12-21 --env-rollout
```

`--eval-protocol inherit`, `utc`, and `legacy_utc` all apply the same UTC daylight grid. `local_day` is **disabled** (different MDP).

## Checkpoint selection protocol

Training tracks **`monitor_metric`** (default: `evaluation/return-average`). When the metric improves, it saves **`best_eval_checkpoint/`**.

### Step 1 — Training-time best (automatic)

- **Criterion:** highest `evaluation/return-average` during training.
- **Artifact:** `best_eval_checkpoint/`.

### Step 2 — Scientific confirmation (hold-out)

1. **Primary:** mean **`total_energy_kwh`** vs `sun_tracking` and `fixed_no_motion` on fixed dates.
2. **Secondary:** movement cost, traces, **solar-altitude** aggregates.
3. **Diagnostics:** `reward_time_analysis.txt` peak-power **UTC hour** vs `solar_altitude_deg` at that step.

### Step 3 — Compare epoch checkpoints (optional)

```bash
python scripts/select_best_checkpoint.py EXPERIMENT_ROOT \
  --fixed-eval-dates 2020-12-07,2020-12-14,2020-12-21,2020-12-28 \
  --compare-baselines \
  --max-path-length 39
```

## Files that encode the standard

| Component | Location |
|-----------|----------|
| Env + info fields | `mbpo/env/pv_tracking.py` |
| Training config | `examples/config/pv_tracking/0.py` |
| MBPO horizon | `mbpo/static/pv_tracking.py`, `examples/development/base.py` |
| Eval / plots / CSV | `scripts/eval_utils.py`, `evaluate_agent.py`, `compare_baselines.py` |
