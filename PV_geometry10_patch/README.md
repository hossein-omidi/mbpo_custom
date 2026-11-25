# Geometry10 observation-mode patch

**Audited base:** `hossein-omidi/mbpo_custom`, `PostProcessing`, commit `92a1b00e5a0cbd5f770514396c0a347f5b9aca6d`.

## 1. Scientific scope and decision

This is a controlled observation-space experiment. It preserves the six-feature experiment and its checkpoint provenance. The installer creates a **separate sibling workspace**, by default `mbpo_custom_geometry10`, with the active managed Torch SAC/MBPO pipeline adapted to the new mode. It does not patch the original working tree, make a Git commit, change the physical plant, or copy previous experiment results. It is not a Git worktree; it is an independent source snapshot. This isolation is deliberate because the original saved checkpoints include source/configuration fingerprints, and changing source files in place would invalidate them.

New configuration: `examples/config/pv_tracking/conf_geometry10.py`. Both SAC and MBPO configuration loaders use that file in the new workspace. The original `conf_compact.py` is retained; use the original checkout for compact training/evaluation. The new workspace's managed training interface is geometry10-only. It is not an arbitrary-state-dimension or legacy TensorFlow/Ray upgrade.

### State contract (exact feature order)

| Index | Name | Range / units |
|---:|---|---|
| 0 | DNI | normalized by 2000 W/m², clipped to [0,1] |
| 1 | DHI | normalized by 2000 W/m², clipped to [0,1] |
| 2 | measured current DC power | normalized by 2000 W, clipped to [0,1] |
| 3 | panel tilt | degrees / 90 |
| 4 | sin(panel azimuth) | [-1,1] |
| 5 | cos(panel azimuth) | [-1,1] |
| 6 | solar zenith | degrees / 180 |
| 7 | sin(solar azimuth) | [-1,1] |
| 8 | cos(solar azimuth) | [-1,1] |
| 9 | remaining episode fraction | [0,1] |

The current power measurement is causal (the reward uses power after the next action); it is a feedback signal for real physical conversion and the existing MBPO delta-power model. We retain it for the first controlled ablation. It may become redundant given richer meteorological variables, but that is not established. Omitting it would change the power model target and confound the experiment. There is no future power observation in the actor. The ten-variable mode is still partially observable: albedo, GHI, temperature, wind and future meteorological evolution are not all observed. No claim of globally optimal/minimal Markov state is made.

## 2. Configuration and data flow

`conf_geometry10.py` → SAC AST reference config loader → shared MBPO config → native PVTrackingEnv geometry10 → validated `SACEnvironment` → actor/critic and replay buffer → real and synthetic updates → periodic validation → saved checkpoints → managed Monte Carlo evaluator → existing numerical reports and representation.

The shared schema lives in `mbpo/geometry10.py` (not `mbpo.env`, avoiding the Gym registration dependency for schema-only operations). Both training algorithms use the exact same features, action limits, initial orientation, physics, movement objective, horizons, scenario split, uncertainty assumptions and optimizer hyperparameters. Hidden layer sizes, reward scale, entropy settings, replay capacity and default 2000 epochs are unchanged. `tau` is index 9, not index 5; current power stays index 2. The compact-only shape assertions, replay allocation, agent inference reshape, metadata and preflight checks are updated.

MBPO's exogenous weather model has 6 inputs: current DNI, DHI, solar zenith, solar azimuth sine and cosine, tau. No action, panel orientation or power enters this weather predictor. Its outputs are the two irradiance deltas. The conditional power model has 14 inputs: current 10D state + 2D action + 2D next weather (predicted during generation; true next weather only during supervised conditional training, as in the original implementation). It predicts normalized power delta. The new power model receives current solar geometry; it does not directly condition on exact next solar geometry. This retains the original one-step model design.

For the structured synthetic environment, panel pose, finite-horizon time, movement energy and solar position are reconstructed. The sampled real replay scenario ID is retained **out of band**, not as an extra policy input. The fake environment uses the original scenario's episode date, location, time grid and pvlib solar-position function to obtain the known next-step ephemeris. It validates current solar geometry against the cached grid and does not read future irradiance or power. Missing/wrong scenario context fails closed. One-step model rollouts remain mandatory. The same scenario-context interface is used in external on-policy and counterfactual model validation. Circular azimuth RMSE and solar-geometry RMSE are reported in the validation summary.

The original physics/reward and paired astronomical baseline remain unaltered. The representation pipeline reads physical quantities and CSV artifacts rather than assuming the network input width. The managed core now requires geometry10 metadata; existing checkpoint and provenance checks are preserved. The static time and cyclic-pair utility is aware of geometry10, but this does not make the old Ray/TensorFlow trainer supported.

## 3. Install (Linux / WSL)

Extract the ZIP into your original repository root and activate the existing `simple-worm` environment. Run these commands **from the original repo**:

```bash
git status --short
git rev-parse HEAD
python PV_geometry10_patch/install.py --root . --dry-run
python PV_geometry10_patch/install.py --root . --apply
```

If the base commit is not exactly the audited one, if tracked local changes exist, if a source hash differs, or if the destination already exists, the installer stops. Do not force it or overwrite existing work. Untracked local files (including this ZIP/patch) are not copied. It checks 16 exact source Git blob hashes and replacement anchors, then parses/compiles the generated Python source. It stages the new workspace before publishing it. No GitHub access is needed at installation time.

The destination is a sibling of the original repository, with `data` and `nsrdb_newyork_5min` linked to the original data directories. Do not run data-generation or data-modification scripts from the geometry workspace; these are shared inputs, not physical read-only mounts. Original outputs and trained policies remain in the original checkout. A different new destination can be provided using `--dest /absolute/path/to/new_workspace`.

## 4. Mandatory local tests before training

```bash
cd ../mbpo_custom_geometry10
python -m pytest -q geometry10_tests
```

If your original directory name differs, use the printed destination. The test suite contains eight standalone numerical schema tests and five integration tests covering configuration parity, actor optimization and replay round trip, ensemble fitting and mixing, managed CLI/manifest, and real PVLib/NSRDB physical parity plus exact synthetic geometry and kinematics. The physical test uses a TRAIN-only historical day and a fixed MC realization; it uses a stub model prediction solely to isolate deterministic transition correctness. It does not estimate predictive skill.

Do not start expensive training if any test fails. Preserve the traceback and command. The package's eight standalone numerical tests were run successfully in the preparation environment. The GitHub source anchors were checked remotely, but the full checkout, Gym and pvlib were unavailable in that environment, so the complete installer dry-run and integration tests have **not** been executed there. They must pass in your local environment. No claim of trained-policy improvement or end-to-end certification is made.

## 5. Controlled validation pilot

The following commands use the existing managed interfaces. Use a fresh experiment name. A 100-epoch paired pilot passes the 8000-interaction warm-up and provides two scheduled validation checkpoints; it is an integration and early-learning diagnostic, not a convergence study. MBPO may generate no synthetic data initially if the model-skill/holdout guard rejects its ensemble. Check the synthetic accounting before drawing conclusions.

```bash
# Optional standalone dataset plot (no training)
python -m PV_experiments.represent --kind dataset --tag geometry10

# Dry-run the experiment plan
python -m PV_experiments.train --name geom10_pilot --seeds 0:1 --epochs 100 --dry-run
# Actual paired pilot
python -m PV_experiments.train --name geom10_pilot --seeds 0:1 --epochs 100

# Verify saved checkpoints and physical MC rollout on two validation days
python -m PV_experiments.evaluate --name geom10_pilot --mode smoke --draws 3
# Full validation partition (small number of draws for the pilot)
python -m PV_experiments.evaluate --name geom10_pilot --mode validation --draws 3
# One-step ensemble validation; run after MBPO has a fitted model
python -m PV_experiments.evaluate --name geom10_pilot --mode model

# Existing representation and advanced sample-efficiency/calendar interface
python -m PV_experiments.represent --name geom10_pilot --kind all
python -m PV_experiments.advanced_report --name geom10_pilot --kind all
python -m PV_experiments.represent --name geom10_pilot --kind collect
python -m PV_experiments.represent --name geom10_pilot --kind trajectories
```

For the actual compact-vs-geometry experiment use fresh runs with identical seeds, equal real-transition budgets, weather assumptions and all other hyperparameters. The pilot budget is frozen in `experiment.json`; use a new experiment name for a longer planned run. Do not mix six-dimensional and ten-dimensional checkpoints, replay pools, reports or run directories. Do not tune on the held-out test set. Compare within each observation mode using its paired learning analysis, then compare the two modes using matched seed/scenario results separately. Do not force the existing same-MDP report to pool different observation modes.

## 6. Audit coverage and limitations

| Component | Treatment |
|---|---|
| Native observation mode and bounds | New geometry10 branch; old modes retained |
| Config resolution | New reference config; both managed algorithms use it |
| SAC actor and critic | Schema dimensions; explicit state-dimension checkpoint guard |
| Replay | 10D storage, encoder validation, checkpoint state dimension |
| MBPO model | Changed input dimensions; 3 learned delta outputs retained |
| MBPO fake transition | Exact known geometry, circular kinematics, time and movement cost |
| Scenario replay and trainer | Context IDs sampled alongside aligned states |
| Periodic validation and model validation | Mode-aware metadata, context, correct angular metrics |
| Managed Monte Carlo | Existing deterministic load-policy and same-physics evaluator via updated shared config and actor |
| Managed reports and plots | Existing numerical inputs; updated managed mode guard |
| Source/experiment provenance | New workspace + config + observation schema hash |
| Legacy standalone Ray / TensorFlow | Not supported for geometry10; original checkout unaffected |
| Legacy standalone SAC evaluation | `SAC_workspace/src/evaluation/evaluation.py` was absent from the inspected GitHub snapshot, so that separate workflow could not be verified or included as a supported path |

If local results reveal excessive chattering or inferior energy collection after training, investigate learning curves, entropy/action saturation, model skill, and action/pose diagnostics; do not assume geometry alone solves the optimization problem. The new state improves access to solar geometry but does not guarantee superior energy harvest relative to a classical analytical controller.
