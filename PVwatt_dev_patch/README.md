# PVwatt_dev — SAC/MBPO critical and high-priority repair (local ZIP)

**No GitHub commits, pushes, or repository writes occur.** The supplied installer only modifies your *local* checkout after validating every existing modified file against the inspected `PVwatt_dev` source. The patch includes complete replacement Python modules for inspected uploaded files and exact anchored transformations for the shared PV environment. Source base: `33415e5d71f3e9756a8c7adb7e935b7bed54c61b`.

## Install (Windows / PowerShell or bash)

Extract the ZIP into the root of your local `mbpo_custom` checkout. Preserve the inner `PVwatt_dev_patch/` folder. From the repository root:

```bash
python PVwatt_dev_patch/apply_patch.py --root . --dry-run
python PVwatt_dev_patch/apply_patch.py --root .
```

The dry run checks Git blob hashes, performs all source transformations and checks Python syntax **before changing anything**. A mismatch is a hard stop; do not override it. The installer creates `_pvwatt_dev_repair_v3_backups/` and `PVWATT_DEV_REPAIR_MANIFEST.json`, and does not call Git or GitHub. It must be run **only once on the pristine inspected branch**. If you have already installed a previous patch, restore the pristine baseline or reconcile changes before running this installer.

The `replacement_files/` directory contains the complete updated modules that could be rebuilt from the uploaded `src_MBPO.zip` and `SAC_src.zip`. The *shared simulator modules and workflow modules* not included in those ZIPs are updated in place by the verified installer. Do not copy just `replacement_files/` and assume that the full patch is installed. `uploaded_sources_only.diff` is a reviewable diff, **not** a substitute for the installer.

## Preserved architecture and interface

- SAC actor, twin critics, entropy adjustment and replay algorithm are preserved.
- MBPO still uses its SAC learner, separate real/model replay and **one-step** synthetic rollouts.
- The compact observation remains six-dimensional `[DNI_norm, DHI_norm, power_norm, panel_tilt_norm, panel_azimuth_norm, tau]`.
- The action remains two continuous normalized incremental movements in `[-1,1]^2`.
- The physical environment advances one historical NSRDB day chronologically; no date or weather feature is appended to the actor input.

## Implemented repairs

### Physical movement energy (enabled)

For the active compact experiment, **both the training and evaluation environments** use `movement_cost_mode='geometry'`, `movement_penalty=1.0`. Default actuator power is 30 W; tilt and azimuth rates are 1.5 and 2.0 degrees/second. Estimated *movement-only* electricity is

```
E_move [kWh] = 1.0 * 30 W * (|executed tilt degrees| / 1.5 deg/s
                         + |executed azimuth degrees| / 2.0 deg/s) / 3,600,000
reward [kWh] = generated DC energy [kWh] - E_move [kWh]
```

**These values are project modelling assumptions, not calibrated actuator measurements.** The two movement durations are added, corresponding to sequential axis motions or the stated accounting convention; standby, holding torque, and controller electricity are excluded. Sensitivity tests using actual motor specifications are recommended for publication. Actuator cost is based on **executed** rather than requested tilt/azimuth movement; this matters at the 0°/90° tilt bounds and the 360° azimuth wrap. The same formula appears in real and fake environments.

### Separate exogenous weather and electrical response

- First Gaussian ensemble: `(DNI, DHI, remaining time) -> (next DNI, next DHI)`; it **never reads the panel action**.
- Second Gaussian ensemble: `(six-dimensional observed state, action, predicted next DNI/DHI) -> next power`.
- Real training examples use observed next weather for conditional power-model fitting; **model rollouts and end-to-end validation use predicted weather**.
- Existing known kinematics, clock, terminal flag and physical movement reward remain deterministic.
- Both groups retain probabilistic neural networks, bootstrapped ensemble member training, elite selection, and optimizer persistence.
- Synthetic transitions are accepted only when the configured model-skill gate passes; model-generated rollouts never increment *real* environment interactions.
- The original compact observation is partially observable. The separated model should be evaluated, not represented as physically exact. Reported internal holdout values are **model-selection diagnostics**, since those days choose the elites. The external, frozen validation days in `06_model_validation` provide a separate generalization check.

### Replay / validation / checkpoint consistency

- The per-scenario model validation fraction is persisted and checked.
- Training and model-selection scenarios have stable day roles across refits.
- Synthetic rollout start states come **only** from the dynamics training scenarios, not internal holdout days.
- The minibatch mixer uses available synthetic data even if the synthetic pool is smaller than the configured target portion; full batch size is retained.
- MBPO dynamics checkpoints carry a **different network schema** and cannot be loaded from old trajectories. Trainer/evaluation checkpoint schema is 2; old version-1 checkpoints are intentionally rejected.
- SAC and MBPO train on the same frozen month-stratified *policy-training* scenarios. Independent policy-validation and final-test scenarios are isolated.

### Periodic evaluation / sample complexity

Included from the previously supplied experiment-infrastructure patch:

- Deterministic policy evaluation every 50 full episodes and policy snapshots.
- Matched validation days, scenario split/protocol fingerprint, and reproducible seeds.
- Real environment interaction counter and synthetic attempted/accepted/used counts.
- `MBPO_workspace/workflows/05_sample_complexity.py`: multi-seed real-interaction learning curves, threshold attainment, AUC and timing analyses.
- `MBPO_workspace/workflows/06_model_validation.py`: model prediction, counterfactual action, uncertainty and persistence baseline assessment.

### Reporting

- `paper_reporting.py`: headline gross/net/movement means and seasonal/weather means average seeds **within each NSRDB day first**, then average days. Paired gain histograms and performance ratios aggregate by day. UTC time crosses midnight correctly for the project's 08:45–01:00 UTC evaluation window.
- `MBPO_workspace/src/plots.py`: descriptive gross/net and paired/seasonal bars are balanced by scenario; its existing unwrapped UTC plotting is retained.
- `SAC_workspace/src/evaluation/plots.py`: gross/net bars and paired summary titles no longer conflate repeated seeds with independent weather days; bootstrap bands are explicitly labelled *descriptive*, not design-based confidence intervals.
- **Important statistical distinction:** these offline convenience plots are *equal-day descriptive estimates*. For the stratified sampling design, population-representative weighted estimates must come from the existing `verified_figures.py` / `stratified_statistics.py` / `advanced_suite.py` workflow with a valid selection plan; this patch does not pretend that equal-day means are calendar-population estimates.

## Verification

From the repository root, after installing dependencies and preparing NSRDB files:

```bash
python -m unittest discover -s MBPO_workspace/tests -p 'test_*.py' -v
python -m compileall -q MBPO_workspace SAC_workspace mbpo
python -m SAC_workspace.workflows.01_setup_config
python -m MBPO_workspace.workflows.01_setup_config
python -m SAC_workspace.workflows.02_environment_replay
```

Then train **new** SAC and MBPO runs. Never reuse old checkpoints or mix zero-cost historical training/evaluation with the new positive-cost environment. Suggested training commands:

```bash
python -m SAC_workspace.workflows.03_train --run-dir SAC_workspace/runs/sac_repaired_seed0 --seed 0 --epochs 2000
python -m MBPO_workspace.workflows.02_train --run-dir MBPO_workspace/runs/mbpo_repaired_seed0 --seed 0 --epochs 2000
```

The sample-complexity analysis (after paired runs for several seeds):

```bash
python -m MBPO_workspace.workflows.05_sample_complexity \
  --sac-runs SAC_workspace/runs/sac_repaired_seed0 \
  --mbpo-runs MBPO_workspace/runs/mbpo_repaired_seed0 \
  --outdir MBPO_workspace/analysis/sample_complexity

python -m MBPO_workspace.workflows.06_model_validation \
  --run-dir MBPO_workspace/runs/mbpo_repaired_seed0 \
  --checkpoint MBPO_workspace/runs/mbpo_repaired_seed0/latest.pt \
  --outdir MBPO_workspace/runs/mbpo_repaired_seed0/model_validation
```

For meaningful confidence intervals on between-seed differences, run several *independent paired seeds*, not just seed 0. Validation episodes do not count as training interactions. Report wall-clock training cost separately from *simulator-interaction sample complexity*.

## Evidence limits

The code was syntax-checked and 12 available pure-model/replay/statistical unit tests passed in the packaging environment. The packaging runtime has `torch`, `numpy`, and `pytest`, but **not** `pvlib`, `gym`, or the prepared NSRDB catalogue; thus it cannot certify complete NSRDB-backed 195-step episodes, physical simulator parity, or 2,000-episode end-to-end SAC/MBPO training. The tests in your local environment are essential. External model accuracy and actual sample-efficiency improvement are **empirical questions**, not guarantees of the refactor.

The installer checks hashes against the original `PVwatt_dev` branch state. It may refuse a newer development branch with diverging source files; in that case, reconcile the changes instead of forcing the patch.
