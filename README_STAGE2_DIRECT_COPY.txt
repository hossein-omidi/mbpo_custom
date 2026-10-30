PVWATTS STAGE 2 - DIRECT REPLACEMENT FILES
==========================================
Built specifically from the provided pvwatts_stage2_sources.zip.

1. From the repo root (~/projects/PVLIB_Paper/mbpo_custom), unpack this ZIP:

       unzip -o PVWatts_Stage2_Direct_Replacement_Local_Source.zip -d .

   The ZIP contains complete .py files, not a patch or an installer.

2. Activate the existing Python environment:

       conda activate simple-worm

3. Verify the existing Stage 1 and Stage 2 tests, then real-world audit:

       python -m pytest -q tests/test_pv_power_backend.py tests/test_pvwatts_stage2.py
       python scripts/pvwatts_stage2_audit.py

4. Explicitly select the upgraded physical model for NEW SAC/MBPO runs:

       python -m SAC_workspace.workflows.03_train --power-model pvwatts_sapm --wind-reference-height-m 2 --run-dir SAC_workspace/runs/sac_pvwatts_new --epochs 1
       python -m MBPO_workspace.workflows.02_train --power-model pvwatts_sapm --wind-reference-height-m 2 --run-dir MBPO_workspace/runs/mbpo_pvwatts_new --epochs 1

   Add --power-model pvwatts_sapm to both 04_evaluate and 03_evaluate
   (SAC and MBPO respectively) along with identical temperature parameters.
   Do not use old-physics checkpoints in these new-physics evaluations.

NOTES
- Legacy remains the default so existing checkpoint/resume configs are not silently broken.
- No changed neural network shape: observations remain 6D and actions 2D.
- SAPM cell temperature uses ambient temperature and 10-m-equivalent wind speed.
- Wind data height 2m, gamma=-0.004/C, and open_rack_glass_polymer are
  provisional study assumptions, not measured module characteristics.
- The current isotropic POA is used as effective irradiance. No AOI/IAM,
  spectral, soiling, inverter/AC, dynamic cell thermal inertia is modeled.
- The MBPO learned model must be retrained for the upgraded power signal.
- The package was syntax-checked, its SAC/MBPO config parity checked against
  the supplied sources, and its rich CSV old/new parser exercised here.
  The numerical pvlib and NSRDB end-to-end runtime cannot be verified here
  without your local environment/dataset. Run the audit above before training.
