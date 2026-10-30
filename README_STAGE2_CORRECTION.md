# PVWatts Stage 2 — direct-copy audit correction

## What this fixes

Your Stage 2 physical backend and the 26 physics/configuration tests already passed.
The subsequent audit stopped because `SAC_workspace.src.evaluation.evaluation.run_episode`
stores trajectory power at `path['power'][i]` and puts the canonical diagnostic
in `path['infos'][i]['power_dc_w']`; it **does not** copy the simulator-only
`info['power']` key into the per-step evaluation metadata. The old audit
unconditionally read that missing field.

The corrected audit checks `power_dc_w` and independently **recomputes** each
of the 6 x 117 evaluation power samples from its recorded geometry and weather
using the active PVWatts backend. It checks actual POA, effective irradiance,
cell temperature, wind at 10 m, energy and reward, and rejects inconsistent or
missing metadata. It does not weaken the production MBPO model-skill gate.

If a two-epoch mini-fit does not pass the learned-model skill gate, the audit
reports PARTIALLY VERIFIED and exits with code 3 instead of claiming a full
synthetic-training integration PASS. That is not a physics runtime failure.

## Files in this archive

- `scripts/pvwatts_stage2_audit.py`: full corrected end-to-end audit.
- `tests/test_pvwatts_stage2_audit_contract.py`: 7 fail-closed tests for evaluation metadata.
- `tests/test_pvwatts_report_roundtrip_offline.py`: real 117-step CSV export/import test.

## Install — one direct copy

From the existing repository root (do not change the repo location):

```bash
cd ~/projects/PVLIB_Paper/mbpo_custom
unzip -o /path/to/PVWatts_Stage2_Audit_Correction.zip -d .
```

Copy the ZIP to the repository root first if you prefer, then replace the
`/path/to/...` operand with the filename.

## Verify

```bash
conda activate simple-worm
python -m pytest -q \
  tests/test_pv_power_backend.py \
  tests/test_pvwatts_stage2.py \
  tests/test_pvwatts_stage2_audit_contract.py \
  tests/test_pvwatts_report_roundtrip_offline.py

python scripts/pvwatts_stage2_audit.py
```

- PASS 0: all audited pathways, including actual skill-approved mixed replay,
  were exercised.
- PARTIALLY VERIFIED 3: learned model was not admitted to replay during the
  short test. The guard was respected, but mixed replay is not yet proven.
- FAIL 2: actual compatibility/physics/reporting error; report the traceback.

### Correct CLI usage

`--power-model` is a flag, not a shell command. It must follow the workflow:

```bash
python -m SAC_workspace.workflows.03_train \
  --power-model pvwatts_sapm --wind-reference-height-m 2 \
  --run-dir SAC_workspace/runs/sac_pvwatts_seed0 \
  --seed 0 --epochs 1 --device cpu

python -m MBPO_workspace.workflows.02_train \
  --power-model pvwatts_sapm --wind-reference-height-m 2 \
  --run-dir MBPO_workspace/runs/mbpo_pvwatts_seed0 \
  --seed 0 --epochs 1 --device cpu
```

These are short smoke runs; do not use the example run directories if they
already contain a previous experiment. During the audit, all new checkpoint
artifacts are created in temporary directories.

**Gym deprecation** is a warning, not the cause of this audit failure. The
CUDA-driver warning affects GPU availability; CPU tests remain valid.

## Verification limits

- The audit fix was reviewed against the uploaded repository source ZIP and
  the Stage 2 local-source replacement package.
- All 8 new offline contract and reporting tests passed in the available runtime.
- Python syntax compilation and ZIP integrity passed.
- End-to-end testing against your local NSRDB files, pvlib 0.13.1 and your
  installed PyTorch is possible only in your local `simple-worm` environment.
- The SAPM mounting preset, temperature coefficient and NSRDB wind reference
  height remain model assumptions pending device-specific calibration.
