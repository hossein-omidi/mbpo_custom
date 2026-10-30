# Stage 1 — PVWatts + SAPM temperature backend (additive, NOT activated)

## What this ZIP actually does

This package contains a **new**, optional physics helper and diagnostic scripts. It
**does not overwrite** the current simulator, trainer, checkpoint, evaluation,
representation, reward, or config files. Simply copying the ZIP into your repo
**does not make SAC or MBPO use PVWatts**. That switch belongs to a later,
separately audited stage. This is intentional: it is the minimum-safe first step.

Package relative to `mbpo_custom/`:

```
mbpo/env/pv_power_backend.py              # new selectable helper
scripts/pvwatts_stage1_preflight.py        # no NSRDB needed
 tests/test_pv_power_backend.py            # math, physical trends, validation
 tests/pvwatts_daily_sensitivity.py        # seasonal NSRDB scenario trajectories
PVWATTS_STAGE1_README.md                   # this file
PVWATTS_STAGE1_UPGRADE_MAP.md              # downstream integration plan
PVWATTS_STAGE1_FILE_HASHES.sha256          # archive contents digest list
```

No new dependencies need to be installed **if** your existing `simple-worm`
environment already has `pvlib==0.13.1`, NumPy, pandas, pytest, and the repo's
training dependencies. Python 3.12 is compatible with pvlib 0.13.1.

## Deploy safely

1. `cd ~/projects/PVLIB_Paper/mbpo_custom`
2. `git status --short` — inspect any changes you have already made.
3. Copy/extract the ZIP contents into the **repository root**, preserving paths.
   If you installed the previous Stage-1 ZIP, allow replacement of exactly its
   added `mbpo/env/pv_power_backend.py` and the two Stage-1 test scripts.
   Do **not** replace `mbpo/env/pv_tracking.py` or `mbpo/env/pvlib_physics.py`.
4. Ensure Python runs from the existing Conda environment.
5. Run the commands below:

```bash
conda activate simple-worm
python scripts/pvwatts_stage1_preflight.py
python -m pytest -q tests/test_pv_power_backend.py
python tests/pvwatts_daily_sensitivity.py --wind-reference-height-m 2 \
    --json /tmp/pvwatts_stage1_sensitivity.json
```

**`--wind-reference-height-m 2` is a provisional assumption**, not a
verified claim about all files. You must check the NSRDB metadata/dataset
specification. Change the argument if your actual wind data has a different
measurement height. For reproducible research, also record the exponent,
mounting preset, and `gamma_pdc` for the selected physical module.

All scripts fail nonzero on test failure. The seasonal script normally writes
no files; it writes the JSON only when explicitly asked. Neither writes into
training directories or touches checkpoints.

## Physics

The shared `compute_poa_global()` calculation is unchanged. The new model:

1. Computes POA from the existing pvlib isotropic transposition helper.
2. Converts wind from declared reference height to 10 m using a configurable
   power-law exponent (default **illustrative** 0.14).
3. Estimates steady-state cell temperature with `pvlib.temperature.sapm_cell`
   using the **illustrative** `open_rack_glass_polymer` preset.
4. Uses `pvlib.pvsystem.pvwatts_dc` with
   `pdc0 = 1000 W/m2 * area * STC_efficiency` and **illustrative**
   `gamma_pdc = -0.004 1/C`.
5. Returns **actual POA** and power in distinct fields.

No IAM, spectral, soiling, shading, inverter or AC model is introduced.
Passing global POA as *effective irradiance* is an explicit approximation.
The candidate model is **not calibrated to a specific module**.

## Run this gate before further edits

- Python interpreter / pvlib API preflight PASS.
- Mathematical and validation tests PASS.
- Seasonal candidate sensitivity PASS on actual NSRDB data.
- Confirm legacy simulator still produces exactly the old result.
- Record Git commit + the full `git status --short` and dataset metadata.

**Only after this gate:** add the opt-in simulator/config/evaluation wiring
as a separate stage, with new experiment IDs and checkpoint identity. See
`PVWATTS_STAGE1_UPGRADE_MAP.md`.

## Runtime validation status

The package was source-checked, compiled and tested against analytic API stubs
in the packaging environment. An official pvlib wheel and your NSRDB dataset
were not available there, so the definitive real-pvlib and real-data tests must
be executed in your `simple-worm` environment. A packaging test or stub test is
**not** evidence of physics accuracy or training integration.
