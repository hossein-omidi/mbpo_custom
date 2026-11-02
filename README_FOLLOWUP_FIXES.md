# Full-day episode-window follow-up fixes

Target: `hossein-omidi/mbpo_custom`, branch `PVwatt_dev`  
Reviewed branch head: `9793dec1fcb37d12d8d6cd37318f4064e66d343c`

This is a small follow-up overlay for the previously supplied
`FULL_DAY_EPISODE_WINDOW_REPAIR_OVERLAY.zip`. It contains only test corrections;
it does not alter the training algorithm, physics, configuration, checkpoints,
manifest, dataset, or run directories.

## What it fixes

1. **False daylight-test failure.** The old test began scanning at 00:00 UTC.
   At New York's longitude that can include daylight from the previous local
   evening. The corrected test starts at 08:45 UTC, scans the full following
   24-hour solar day, and verifies that sunrise and sunset both fall within
   the 08:45–01:00 UTC episode window.

2. **TensorFlow import during NSRDB test collection.** The NSRDB tests no longer
   import `scripts.eval_utils` at module load. That legacy utility imports an
   unused TensorFlow helper at module level, and the tests do not need it.
   Do not install TensorFlow just to collect or run these tests.

3. **Invalid legacy test helper references.** The previous test version imported
   `verify_rollout_pvlib_power`, which is not defined in the branch's
   `scripts/eval_utils.py`, and used a baseline name unsupported by
   `make_baseline_rollout`. The revised tests check scenario pairing using
   direct environment rollouts and verify power from an actual environment
   transition.

4. **Optional evaluation archive dependency.** The Stage-A physics integration
   test no longer imports the unextracted `SAC_workspace/src/evaluation`
   package. It checks the actual Stage-A environment, PVWatts power, recorded
   albedo, energy identity, and 195-step horizon directly.

5. **Dataset path robustness.** The NSRDB reader test uses the manifest's
   resolved `_year_path` and `_data_dir`, rather than assuming the manifest
   directory is always the data directory.

## Install

Extract this ZIP directly into the repository root, preserving folders and
allowing the three included test files to overwrite the previous versions.

No installation command, TensorFlow package, dataset regeneration, or model
retraining is required for these corrections.

## Run

```bash
python -m pytest -q tests/test_full_day_episode_window.py
python -m pytest -q tests/test_nsrdb_multiyear.py
python -m pytest -q tests/test_stage_a_physics.py
```

The NSRDB and Stage-A integration tests require the real local NSRDB manifest,
year CSVs, pvlib, and Gym. Missing data or dependencies should be reported as
such; they must not be replaced with synthetic weather.
