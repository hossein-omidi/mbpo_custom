# SAC Fairness Diagnostic v3 — Direct-execution import-path correction

This ZIP replaces **only two read-only diagnostic scripts** and adds one import-path test. It does **not** change SAC, MBPO, the simulator, training configurations, trained models, or saved evaluation results.

## Root cause

`python scripts/sac_checkpoint_config_audit.py` initializes Python's import search path with `scripts/`, not the containing repository root. Consequently `from SAC_workspace...` may fail with `ModuleNotFoundError` even from the correct current directory. This also affects `scripts/sac_policy_forensic.py`. Both scripts now prepend `Path(__file__).resolve().parents[1]` to `sys.path` before they perform any workspace imports, which works regardless of current directory and shell `PYTHONPATH`.

## Direct copy

Unzip at the **existing** repository root:

```bash
cd ~/projects/PVLIB_Paper/mbpo_custom
unzip -o PVWatts_SAC_Fairness_Diagnostic_v3_Import_Fix.zip -d .
conda activate simple-worm
```

## Run

```bash
python scripts/sac_checkpoint_config_audit.py \
  --eval-dir MBPO_workspace/evaluation/pvwatts_seed0_heldout1 \
  --sac-checkpoint SAC_workspace/runs/sac_pvwatts_new_seed0/latest.pt \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_pvwatts_new_seed0/latest.pt \
  --sac-training-csv SAC_workspace/runs/sac_pvwatts_new_seed0/episodes.csv \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_pvwatts_new_seed0/episodes.csv \
  --require-checkpoints

python scripts/sac_policy_forensic.py \
  --eval-dir MBPO_workspace/evaluation/pvwatts_seed0_heldout1 \
  --sac-checkpoint SAC_workspace/runs/sac_pvwatts_new_seed0/latest.pt \
  --sac-training-csv SAC_workspace/runs/sac_pvwatts_new_seed0/episodes.csv \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_pvwatts_new_seed0/episodes.csv
```

## Checks

```bash
python -m pytest -q tests/test_direct_launch_imports.py
python -m pytest -q tests/test_sac_policy_forensic_offline.py tests/test_sac_checkpoint_config_offline.py
```

## Immediate alternative, without replacement

Use `PYTHONPATH="$PWD" python scripts/sac_checkpoint_config_audit.py ...` or `python -m scripts.sac_checkpoint_config_audit ...` **from the repository root**. The fixed scripts do not require either workaround.

## Honest verification boundary

The precise `ModuleNotFoundError` has been reproduced and the import path verified in isolated child processes. The actual SAC/MBPO checkpoint and local training-history validations still need to run on your machine, because the checkpoint files are not in the supplied evaluation archive. If subsequent checks reject the training physics fingerprint, checkpoint epoch, or recorded actions, that is **new diagnostic evidence**, not another import issue; do not bypass those checks. Only load `.pt` files produced by your own trusted training environment because `torch.load(weights_only=False)` is used.
