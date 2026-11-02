# MBPO full-day compatibility and scientific-reference repair

Target repository: `hossein-omidi/mbpo_custom`, branch `PVwatt_dev`  
Reviewed source: user-uploaded `src.zip` (12 MBPO source files)

## Included replacements

- `MBPO_workspace/src/dynamics.py`
- `MBPO_workspace/src/trainer.py`
- `MBPO_workspace/src/evaluation.py`
- `MBPO_workspace/src/power_references.py`
- `MBPO_workspace/src/paper_reporting.py`
- `MBPO_workspace/src/plots.py`
- `MBPO_workspace/src/verified_figures.py`
- `MBPO_workspace/src/advanced_suite.py`
- `mbpo/env/baseline_contract.py`
- `tests/test_mbpo_full_day_compatibility.py`

## Corrections

1. MBPO synthetic horizon is derived from the real environment and checked
   against `num_action_steps` and `periods - 1`; no hard-coded 117.
2. Model fitting frequency must be a positive integer multiple of the real
   episode horizon, preserving whole-day model-validation boundaries.
3. Fresh run metadata is written only after successful environment/model
   construction and validation. Failed startup no longer poisons a run folder.
4. The instantaneous dual-axis reference passes recorded per-step NSRDB albedo
   into the shared PV power backend.
5. The MBPO POA/DC oracle delegates to the existing Stage-A albedo-aware
   analytical implementation instead of duplicating a fixed-albedo calculation.
6. The paired baseline contract accepts dynamic episode lengths, checks every
   trajectory series length, and verifies recorded NSRDB albedo equality.
7. Rich paper exports and figures no longer assume 117 steps or the old
   12:05–21:45 UTC window. Albedo values and provenance are retained.
8. Ensemble disagreement is expressed in standardized target coordinates,
   making the optional uncertainty threshold dimensionless across DNI, DHI
   and power channels.
9. The evaluator's POA-only option no longer depends on the optional
   dual-axis-classical branch being enabled.

## Intentionally unchanged

- Six-dimensional compact observation and two-dimensional normalized action.
- POMDP formulation and learned three-channel one-step dynamics target.
- SAC actor/critic architecture, losses, entropy tuning and optimizer settings.
- MBPO ensemble size, elite count, one-step rollout, replay ratio and reward.
- PV power backend, SAPM temperature model, NSRDB source data and checkpoints.

## Install

Extract this ZIP at the repository root, preserving folders. Back up existing
files first if you have uncommitted local changes.

## Verify

```bash
python -m pytest -q tests/test_mbpo_full_day_compatibility.py
python -m pytest -q tests/test_full_day_episode_window.py
python -m pytest -q tests/test_nsrdb_multiyear.py
python -m pytest -q tests/test_stage_a_physics.py
```

Then train with a fresh run directory. Do not resume a 117-step checkpoint or
reuse an old replay buffer.

## Scope limitation

The uploaded archive contained MBPO `src/` modules, not the complete repository.
The `mbpo/env/baseline_contract.py` replacement is based on the `PVwatt_dev`
branch source. The local SAC environment adapter, NSRDB loader, PV backend and
SAC evaluation archive were not included in the upload; their runtime
compatibility still needs the listed integration tests.
