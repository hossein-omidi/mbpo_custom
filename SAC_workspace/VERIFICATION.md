# Standalone PV SAC — independent audit, revision 2 (2026-10-15)

This replaces the first patch set. Use only the **v2** ZIPs in order; they are
self-contained replacements for the earlier `SAC_workspace` files.

## Verification performed in the audit container

- Syntax compilation of the complete new workspace.
- **14 unittest cases discovered; 12 passed, 2 deliberately skipped** because
  the original cloned repository and complete NSRDB/pvlib environment are not
  available in this container. A skip is not a successful physical test.
- Configuration extraction tested with a synthetic source fixture that
  independently reconstructs **training and evaluation** overrides.
- Tanh log-density independently checked against PyTorch `TanhTransform`;
  soft Bellman terminal masking, 100x reward scale, exact Polyak updates,
  temperature floor, action bounds, and replay-ring serialization checked.
- End-to-end simulated 117-step environment integration: real-transition-only
  training, 37 gradient updates, checkpoint save/restore, matching scenario and
  timestamp comparison, all four benchmarking methods, metrics CSV and plots.
- Scientific semantics: instantaneous sun-pointing is displayed as gross
  energy only. Feasible controllers are compared on net energy. Multiple seeds
  with an identical fixed evaluation start pose are rejected to avoid false
  independent replication.

## Unverified until executed inside the full repository

- The original `PVTrackingEnv` and the native NSRDB PSM4 year files under the
  site's actual local pvlib/Gym installation. Run `test_real_pv_parity.py` and
  require it to **pass, not skip**.
- Manifest site, dates and grid; `01_setup_config` checks these and refuses
  to fall back to clear-sky data.
- Full 2,000-episode training stability and convergence, empirical energy
  advantage, annual/cross-season uncertainty estimates, and independent
  held-out generalization. No scientific gain is claimed from mock outputs.
- Perfect bit-for-bit GPU determinism across distinct drivers/hardware.

## Original repository libraries reused directly

- `mbpo.env.pv_tracking.PVTrackingEnv`: observation, control bounds, motion,
  episode transition, reward and telemetry.
- `mbpo.env.pvlib_physics.dual_axis_tracking_targets` and
  `compute_panel_power_w`: dual-axis geometry and exactly matched POA/power.
- `mbpo.env.nsrdb_weather.load_scenario_manifest`: scenario ID resolution.
- AST-only reading of `examples/config/pv_tracking/_nsrdb_base.py` and
  `conf_compact.py`: no Ray or TensorFlow imports.

## Execute inside your local repository root

```sh
python -m unittest discover -s SAC_workspace/tests -p "test_*.py" -v
python -m SAC_workspace.workflows.01_setup_config
python -m SAC_workspace.workflows.02_environment_replay
python -m SAC_workspace.workflows.03_train --run-dir SAC_workspace/runs/new_smoke --seed 0 --epochs 2
```

A 2-episode smoke run is below the 8,000-step initial exploration threshold;
use a **new** run folder for full training. The physical parity test may be
slower because it constructs the original NSRDB environment twice.

Full evaluation once a trained checkpoint exists:

```sh
python -m SAC_workspace.workflows.04_evaluate --checkpoint SAC_workspace/runs/sac_seed0/latest.pt --outdir SAC_workspace/evaluation/seed0
python -m SAC_workspace.workflows.04_evaluate --checkpoint SAC_workspace/runs/sac_seed0/latest.pt --all-scenarios --outdir SAC_workspace/evaluation/full_nsrdb
```

Do not interpret the five reference scenarios as unseen holdout cases;
the training NSRDB distribution includes them. For sound manuscript claims,
report actual source versions, evaluate multiple independent trained seeds
in separate runs, use sufficient distinct weather scenarios, and show
paired statistics.
