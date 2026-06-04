# Baseline comparison fairness (PV tracking)

All methods use the same `PVTrackingEnv.step()` path and reward:

```text
energy_kwh     = power_W × interval_hours / 1000
movement_cost  = movement_penalty × (|a0| + |a1|)   # a in [-1,1], normalized deltas
reward         = energy_kwh - movement_cost
```

`movement_penalty` comes from the trial variant (`params.json`). Eval builds the env from that variant, so **learned**, **fixed_no_motion**, and **sun_tracking** share the same penalty formula.

## Baselines (`scripts/eval_utils.py` → `make_baseline_rollout`)

| Name | Behavior | Movement cost |
|------|----------|----------------|
| **sun_tracking** | Each step: slew toward `tilt=solar_zenith`, `azimuth=solar_azimuth` (±5°/±10° limits) | Non-zero when tracking |
| **fixed_no_motion** | Each step: `action = [0, 0]` — panel **frozen** at reset orientation | **Zero** (no slew) |
| **fixed_tilt_south** (optional) | Slew toward fixed 30° tilt / 180° azimuth | Non-zero until pose reached |

`fixed_no_motion` is **not** “slew to 30°/180°”. That older behavior mis-attributed actuator cost to a “fixed” station. Compare **gross `total_energy_kwh`** when the fixed array starts at a random pose (same `randomize_initial_orientation` as the learned policy).

## Sun tracking baseline

- Same incremental limits: ±5° tilt / ±10° azimuth per step.
- Same normalized actions in `[-1, 1]²`, then `env.step(action)`.
- Not an oracle: no pvlib bypass, no infinite slew rate.

## Eval pairing

`scripts/evaluate_agent.py` uses `seed = eval_seed_base + i` for learned and each baseline so **calendar date and weather** match. See `eval_scenario_confirmation.txt` in each `--outdir`.

## Post-train statistics (not SAC)

Over `N` MC rollouts:

- **E[R]** = sample mean of episode returns `R_i = Σ_t r_t`
- **σ** = sample std with `ddof=1` (spread across rollouts, **not** policy entropy or Q-value bounds)

In-training `evaluation/return-std` in `progress.csv` is the std over Ray’s `n_eval` episodes **that epoch** — a different estimator. See `EVAL_STATISTICS.txt` in each eval output dir.

## Season labels

- **`season_calendar`** (reporting): month buckets (Jun–Aug = summer). Used in `evaluation_by_season.png` and summary “By calendar season”.
- **`season`** (env `info`): equinox-based day-of-year buckets in `PVTrackingEnv` (e.g. 2020-06-01 can be env-**spring** before day 172).

## Episode clock (UTC)

- Start **13:30 UTC**, step **7 min 30 s**, **78** transitions (same as training).
- Plots and CSV `clock_hour_utc` are UTC only.

## Metrics

| Report field | Meaning |
|--------------|---------|
| `total_energy_kwh` | Gross daily harvest (sum of step `energy_kwh`) |
| `total_movement_cost` | Sum of step penalties |
| `total_reward` | Net = energy − movement (training objective) |

Compare **gross energy** for production; compare **net reward** for the motion-penalized objective.
