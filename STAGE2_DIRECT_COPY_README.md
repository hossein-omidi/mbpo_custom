# PVWatts Stage 2: correct installation for your local SAC / MBPO project

## Diagnosis (not a directory problem)

Your shell was already in the correct repository root:
`~/projects/PVLIB_Paper/mbpo_custom`.

Earlier Stage-2 installers stopped before editing any core files because:
1. The v2 code demanded one exact text fragment in your locally modified
   `MBPO_workspace/src/evaluation.py`.
2. The v3 source transformer generated malformed Python for
   `MBPO_workspace/workflows/01_setup_config.py` (unterminated string).

Therefore the `power_model` configuration and `PVTrackingEnv.power_model` never
existed, and seven post-install tests naturally failed. The missing direct-copy
manifest was a consequence of the builder stopping early, not a separate fault.

## This package and its limitations

Because your **local source differs from the published branch**, this ZIP cannot
safely contain final, pre-built versions of your 13 existing source files.
Overwriting those with public copies would lose unsubmitted local changes,
including parts of your evaluation pipeline. Instead, the package generates
**complete, ready-to-copy replacement files** from the actual existing files
in your checkout. It does not use `--apply`, relax source checks, or bypass tests.

The v4 transformer corrects the broken setup string and uses syntax compilation
for all seven workflow files. Remaining unmatched local source changes cause a
clear FAIL before any core file is modified. Do NOT force an incompatible change.

## Installation: only two commands after extraction

1. Extract this ZIP into the EXISTING repo root, preserving paths. It adds
   scripts/modules/tests but does not overwrite the 13 integrated core targets.
2. Run these commands from that same root, in the `simple-worm` conda env:

```bash
cd ~/projects/PVLIB_Paper/mbpo_custom
conda activate simple-worm
python scripts/build_stage2_direct_replacements.py
```

On PASS, the command creates all of the following:

- `PVWATTS_STAGE2_READY_TO_COPY/` — all **complete updated source files** in
  the original directory hierarchy, plus test/support files and a manifest.
- `PVWatts_Stage2_Direct_Replacements_<timestamp>.zip` — the same full files.
- `PVWatts_Stage2_Local_Originals_Backup_<timestamp>.zip` — the original 13
  source files, preserved for rollback. Keep this backup outside the repo too.

No source has been overwritten yet. Once you have inspected/retained the backup,
**copy the complete replacement files** with:

```bash
cp -a PVWATTS_STAGE2_READY_TO_COPY/. .
```

This is a normal complete-file copy, *not* a patch application or a version of
`apply_pvwatts_stage2.py --apply`. The builder generates complete replacements
in a staging directory before you copy them.

## Required acceptance verification

```bash
python scripts/verify_stage2_direct_copy.py
python -m pytest -q tests/test_pv_power_backend.py tests/test_pvwatts_stage2.py tests/test_pvwatts_stage2_patcher.py tests/test_stage2_all_workflow_source.py
python scripts/pvwatts_stage2_audit.py
```

Only after these pass, run new-physics experiments with identical settings for
SAC and MBPO, e.g. `--power-model pvwatts_sapm --wind-reference-height-m 2`.
Use **new run directories** and do not resume existing legacy checkpoints under
PVWatts. The `2 m` wind-height, mounting, and gamma assumptions are provisional.

No source-packaging check, mock test, or compilation proves your real NSRDB
runtime acceptance; the real audit must execute locally. Do not start expensive
training while that audit is failing.

## If the builder still fails

Do not change directories, bypass checks, or run the old installers. The error
will identify a local file the transform cannot safely reconcile. Supply that
file (at least `MBPO_workspace/src/evaluation.py`, if mentioned) as an actual
source attachment so complete replacements can be prepared for your precise
checkout. No blanket source overwrite is justified without that comparison.
