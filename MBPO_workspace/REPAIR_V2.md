# Self-contained workspace recovery patch (v2)

## Why these import errors occurred

The previous `solar_mbpo_remaster_complement_reporting_v1_1.zip` was an **add-on**, not a complete MBPO installation. It contained `src/paper_reporting.py` and `workflows/03_evaluate.py`, but **did not contain** the `src/config.py`, `src/evaluation.py`, `src/trainer.py`, `src/dynamics.py`, `src/replay.py`, or `src/plots.py` it imported. A machine with only the add-on therefore discovers just one MBPO test and fails with `ModuleNotFoundError`.

This recovery ZIP is **self-contained for the MBPO workspace** and contains the complete original MBPO remaster, the advanced-reporting add-on, plus targeted source and regression-test fixes. It requires your existing `SAC_workspace` v2 (already functioning according to your tests) and the repo's existing `mbpo/env` and `examples/config` sources.

## Corrected code behavior

1. `MBPO_workspace/src/*` and its required training/evaluation workflows are present in the installed tree. The complete MBPO module hierarchy imports without Ray or TensorFlow.
2. SAC's evaluation module exports `reference_config` and `reference_scenario_ids()`. If the compact reference configuration sets `fixed_eval_scenarios=None`, the evaluator uses the original `NSRDB_VALIDATION_SCENARIO_IDS` from `_nsrdb_base.py` via the existing safe AST reader; it does not invent scenarios or change training distributions. This fixes **both** standalone SAC and MBPO default evaluation.
3. The instantaneous dual-axis gross-only reference now reads the **already recorded, post-action** irradiance and solar geometry from the paired fixed-mount path. It never reseeds or advances the simulator a second time and therefore does not require `.seed()` on a stand-in object. Its energy is calculated using the original `mbpo.env.pvlib_physics` functions with the evaluator's existing area, efficiency, and interval.
4. No new model-based actions enter evaluation. The matched physical simulator, 6D observation, actions, reward, 117-step horizon, reference config, trained checkpoint schema, dynamics training, and legacy `scripts/` are unchanged.

## Apply from the repository root

```bash
unzip -l solar_mbpo_sac_recovery_patch_v2.zip
unzip -o solar_mbpo_sac_recovery_patch_v2.zip -d .
python -m unittest discover -s MBPO_workspace/tests -p 'test_*.py' -v
python -m unittest discover -s SAC_workspace/tests -p 'test_*.py' -v
python -m MBPO_workspace.workflows.01_setup_config
python -m MBPO_workspace.workflows.03_evaluate --help
python -m MBPO_workspace.workflows.04_advanced_report --help
```

**Apply with your modern Python environment**, not Python 3.6.

The ZIP **overwrites workspace source files in place**, but never deletes any files. If you have custom edits to `MBPO_workspace/` or `SAC_workspace/src/evaluation/evaluation.py`, inspect the ZIP contents or compare your changes first. It does not touch training run folders, existing checkpoints, original TensorFlow/Ray sources, NSRDB CSVs, or legacy plotting scripts.

Evaluation requires both trained checkpoints to exist. The `--help` invocation and the test suites can run before either is trained. Use the actual checkpoint paths on your machine rather than expecting the example names to exist.

## Remaining validation boundary

This patch was tested with the available standalone PyTorch environment and mocked physics, including dynamics training, resume, and offline PDF plotting. Full `PVTrackingEnv` + real NSRDB execution cannot be reproduced in the patch-building container because the project-specific data manifest, pvlib and gym are unavailable there. Your machine's `test_real_pv_parity` and longer ensemble training are the next authoritative environment-integration checks. Passing software tests is not proof of RL convergence or superior net-energy yield.
