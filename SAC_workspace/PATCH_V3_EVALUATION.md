# SAC v3 — Minimal evaluation correction and scientific plots

**Apply inside the existing `mbpo_custom` repository root.**
This is an overlay for the SAC v2 workspace. It does **not** change training, checkpointing,
`PVTrackingEnv`, pvlib/NSRDB ingestion, the reference task configuration, action definitions,
or the original model-based reinforcement-learning code.

## What caused the exception

`conf_compact.py` explicitly sets `fixed_eval_scenarios=None` to enable the full-manifest
Monte Carlo protocol. The previous standalone evaluation tried to use that `None` as its
implicit default, producing an empty list and `ValueError: No evaluation scenarios specified`.

`evaluate(..., scenario_ids=None)` now resolves the original
`NSRDB_VALIDATION_SCENARIO_IDS` directly from `_nsrdb_base.py` when the compact evaluation
configuration provides no fixed list. Explicit `--scenarios` and `--all-scenarios` still override
that default. No configuration hash changes: the already-trained `latest.pt` stays valid.

## Efficiency and mathematical parity

- The fixed-mount rollout already records **physical UTC step timestamps** and weather.
  Its underlying NSRDB/pvlib measurements are now reused to compute the instantaneous
  dual-axis *gross energy* with the existing `dual_axis_tracking_targets` and
  `compute_panel_power_w` functions; this avoids running an extra 117-step solar simulation.
- Three feasible controllers (SAC, slew-limited dual axis, fixed panel) still call the
  same `PVTrackingEnv.step`, with identical day/seed, initial pose, action limits,
  5-minute schedule, and finite-horizon reward definition.
- Instantaneous sun pointing is NOT actuator-feasible and therefore has **no defined
  net energy**. It remains an informative gross-energy reference, not proof of global POA
  optimality.
- With the reference `movement_penalty=0.0`, simulated `net_energy_kwh` equals
  generated energy for the feasible controllers; it does not include real motor-power
  consumption. That follows the original MDP, not a change in this patch.
- All 2018–2024 scenarios sampled for training may overlap with evaluation scenarios;
  report outcomes as matched / in-distribution benchmarks, **not held-out generalization**.

## New outputs

Alongside the existing `summary.json`, `episodes.csv`, `trajectories.csv` and images,
this patch adds:

- `paired_metrics.csv`: one row per day/seed and feasible baseline, containing both
  net energies and paired differences.
- `paired_scenario_net_energy_scatter.png`: SAC net energy against each feasible
  reference for the *same* NSRDB scenario.
- `seasonal_paired_net_gain.png`: SAC minus baseline paired net kWh per season, with
  whole-scenario bootstrap uncertainty where at least two scenarios are available.
- `monthly_paired_net_gain.png`: analogous per-calendar-month comparison.
- `representative_days/<season>_<scenario-id>.png`: up to one recorded trajectory per
  represented season, displaying matched power, gross energy, tilt and azimuth.

`summary.json` adds `pooled_net_gain_pct` and `fraction_pairs_sac_better` for feasible
baselines; it retains all existing metrics.

## Apply

From your repository root (where `SAC_workspace` already exists):

```bash
unzip -o solar_sac_v3_minimal_evaluation_patch.zip -d .
python -m unittest discover -s SAC_workspace/tests -p 'test_*.py' -v
python -m SAC_workspace.workflows.04_evaluate \
    --checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
    --training-csv SAC_workspace/runs/sac_seed0/episodes.csv \
    --outdir SAC_workspace/evaluation/sac_seed0
```

For reproducible limited-scenario testing:

```bash
python -m SAC_workspace.workflows.04_evaluate \
    --checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
    --scenarios 2018-02-15 2019-05-15 2020-08-15 2021-11-15 2024-06-21 \
    --training-csv SAC_workspace/runs/sac_seed0/episodes.csv \
    --outdir SAC_workspace/evaluation/reference_five
```

For the full manifest use `--all-scenarios`. This can involve thousands of NSRDB days
and much more CPU time and memory; it generates files only once evaluation finishes.
A process started before applying the patch continues running **its existing code**;
this package cannot alter the already-running Python process or its outputs.

## Verification boundaries

Local Python unit tests and mocked 117-step training/evaluation/checkpoint integration
were executed for this patch. Physical NSRDB and pvlib parity tests require your
existing repository installation; run them locally to complete physical validation.
Do not treat the test suite alone as proof that SAC outperforms a dual-axis tracker.
