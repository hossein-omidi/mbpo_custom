# Baseline comparison fairness (PV tracking)

All methods use the same `PVTrackingEnv.step()` path and reward:

```text
energy_kwh     = power_W × (15 min) / 1000
movement_cost  = movement_penalty × (|Δtilt|/5° + |Δazimuth|/10°)
reward         = energy_kwh - movement_cost
```

`movement_penalty` comes from `environment_kwargs` in `examples/config/pv_tracking/0.py` and is stored in trial `params.json`. Eval builds the env from that variant, so **learned, fixed, and sun_tracking** share the same penalty.

**Stage 1 training** sets `movement_penalty=0.0` so net reward equals gross energy; compare methods on `total_energy_kwh`.

## Sun tracking baseline

Implemented in `scripts/eval_utils.py` → `make_baseline_rollout()`:

- Targets `tilt = solar_zenith`, `azimuth = solar_azimuth` (from decoded obs).
- Same incremental limits: ±5° tilt / ±10° azimuth per 15-min step.
- Same normalized actions in `[-1, 1]²`, then `env.step(action)`.

Not an oracle: no pvlib bypass, no infinite slew rate.

## Eval pairing

`scripts/evaluate_agent.py` uses `seed=rollout_index` for learned and each baseline so **date and weather** match. See `eval_scenario_confirmation.txt` in each `--outdir`.

## Metrics

| Report field | Meaning |
|--------------|---------|
| `total_energy_kwh` | Gross daily harvest (sum of step `energy_kwh`) |
| `total_movement_cost` | Sum of step penalties |
| `total_reward` | Net = energy − movement (training objective) |

Compare **gross energy** for “maximize production”; compare **net reward** for “optimize under motion cost.”
