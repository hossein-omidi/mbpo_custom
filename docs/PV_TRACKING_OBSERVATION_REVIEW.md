# PVTracking observation space — scientific review

**Scope:** `mbpo/env/pv_tracking.py`. Reward, SAC/MBPO, and evaluation metrics are unchanged.

## Final physical observation (11-D, deployable)

| # | Feature | Source on real system | In sim |
|---|---------|----------------------|--------|
| 0 | `solar_zenith_norm` | Computed from GPS+time **or** sun sensor | pvlib `get_solarposition` |
| 1–2 | `solar_azimuth_sin/cos` | Same | pvlib |
| 3–5 | `dni/dhi/ghi_norm` | Pyranometer / weather station | See [Weather](#weather-and-irradiance) |
| 6 | `temperature_norm` | Ambient sensor (°C) | Weather profile |
| 7 | `panel_tilt_norm` | Tilt encoder | Actuator state |
| 8–9 | `panel_azimuth_sin/cos` | Azimuth encoder | Actuator state |
| 10 | `cos_aoi` | Computed from sun + panel pose (pvlib `aoi`) | Same |

**Removed from physical (not deployable / redundant):**

- `episode_progress` — training episode index, not a field sensor
- `power_norm` — redundant with irradiance + alignment (legacy only)
- `time_of_day` / `day_of_year` sin/cos — optional RTC features; omitted so policy uses sun + weather only (see [MBPO horizon](#mbpo-horizon-without-episode_progress))

**Legacy (15-D, default):** unchanged for existing checkpoints.

## MBPO horizon without `episode_progress`

`StaticFns.decoded_time_of_day()`:

- **Legacy:** decodes clock from obs indices 11–12 (validated against `info['time']`).
- **Physical:** policy vector has no clock. `StaticFns.termination_fn` does **not** use solar time (avoids early stop); imagined depth is capped by MBPO `rollout_length` (config max 3). Real episodes end via `env.done`. `is_valid_rollout_start_obs` still uses solar-hour estimate internally to filter near-horizon pool starts (not in policy obs).

Real env episodes still end via `step_index >= len(times)-1`; replay terminals stay correct.

## Weather and irradiance

Built in `_build_weather_profile(times)` aligned to episode `times` (15 min, same index as `step_index`).

### `weather_source='clearsky'`

| Field | Source | Units |
|-------|--------|-------|
| DNI, GHI, DHI | `location.get_clearsky(times)` | W/m² |
| temperature | Constant `self.temperature` | °C |
| wind_speed | Constant `self.wind_speed` | m/s (not in obs) |

### `weather_source='historical'` (training default)

1. Load a **pvlib-compatible weather catalog** for the default site.
2. Use timestamp-aligned **historical/TMY irradiance** (`dni`, `ghi`, `dhi`) instead of synthetic attenuation.
3. Reindex that catalog onto the env episode grid (`13:30`–`23:15` UTC, 15 min).
4. Derive a coarse `condition` label (`clear` / `partly_cloudy` / `overcast`) from historical-vs-clearsky irradiance ratios for diagnostics only.
5. Carry through historical `temperature` and `wind_speed` from the dataset.

**Scaling in obs:** divide by `IRRADIANCE_NORM=2000` W/m², `TEMPERATURE_NORM=50` °C.

**Power:** `get_total_irradiance(..., dni, ghi, dhi, solar_zenith, solar_azimuth, model='isotropic')` × area × efficiency. Historical irradiance therefore changes power directly. Temperature/wind are now dataset-backed and logged, but they do **not** yet affect the current power model.

**Physical validity:** DNI, DHI, GHI ≥ 0; timestamps share index with solar position; historical weather is loaded through pvlib-compatible readers / cached CSV and checked with `scripts/check_pv_env.py --validate-weather`.

## Modes

| Mode | Dim | Use |
|------|-----|-----|
| `legacy` (default) | 15 | Old checkpoints, eval without retrain |
| `physical` | 11 | New training / deployment target |

```python
'environment_kwargs': {
    ...
    'observation_mode': 'physical',
}
```

**Retrain required** when switching to `physical` (11-D ≠ 15-D).

## Verification commands

```bash
/home/ecer/miniconda3/envs/mbpo/bin/python scripts/check_pv_env.py --observation-mode legacy
/home/ecer/miniconda3/envs/mbpo/bin/python scripts/check_pv_env.py --observation-mode physical --log-obs --validate-weather
/home/ecer/miniconda3/envs/mbpo/bin/python scripts/validate_pv_rollouts.py
```
