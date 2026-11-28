# Integrated Geometry10 Instantaneous-Target Action Patch (v2)

This bundle replaces the separate v1.1 patch. It combines the agreed action
mechanism with the minimal nonlinear-function corrections needed to keep the
real environment, MBPO, baselines, counterfactual validation and trajectory
representation consistent.

## 1. Action contract

For every five-minute Geometry10 decision:

\[
a_t=(a_\beta,a_\gamma)\in[-1,1]^2
\]

\[
\beta_{t+1}=45(a_\beta+1)^\circ
\]

\[
\gamma_{t+1}=(\gamma_t+180a_\gamma)\bmod 360^\circ
\]

Tilt is an absolute target. Azimuth is a relative circular target. The target
pose is applied at the next discrete transition; there is no 5°/10° command
increment cap, action smoothing, or extra nonlinear action filter.

A zero vector is **not** a no-motion command. A no-motion command is generated
by `hold_action(current_tilt)`. The fixed baseline and dark-period astronomical
baseline use this hold action.

## 2. Preserved components

The patch does not change:
- Geometry10 state dimension, feature order, or normalized state meaning;
- SAC actor/critic architecture, tanh squashing, log-standard-deviation bounds,
  twin-critic minimum, target update, entropy objective, or configured alpha floor;
- SAC/MBPO optimizers, losses, replay ratios, training schedule, rollout length,
  ensemble architecture, or model fitting algorithm;
- five-minute grid, finite horizon, physical PV power backend, or reward equation;
- geometry-mode actuator movement-energy formula.

The geometry-mode movement cost continues to use the **executed physical angular
displacement** and the existing motor power/slew-rate parameters. Instantaneous
target acquisition is the MDP transition convention; actuator energy remains
a separate reward cost. The maximum possible sequential travel under the
reviewed default rates is 90/1.5 + 180/2 = 150 seconds, within a 300-second
interval.

`max_delta_tilt` and `max_delta_azimuth` remain available for legacy
non-Geometry10 incremental modes and legacy movement-cost normalization. They
are no longer action limits in Geometry10.

## 3. Necessary corrections

- One shared action decoder/encoder is added to `mbpo/geometry10.py`.
- The physical Geometry10 environment and MBPO structured fake environment use
  that same decoder.
- Action values are validated and passed through unchanged; redundant action
  clipping is removed from the actor output adapter, physical adapter,
  Geometry10 pose transition and synthetic environment.
- DNI/DHI values outside the declared 0–2000 W/m² normalization domain are
  rejected rather than silently saturated. The state normalization itself is
  unchanged. If a real source value exceeds 2000 W/m², inspect the dataset and
  make an explicit normalization decision before training.
- The duplicate next-state projection in `StructuredFakeEnv` is removed.
  `DynamicsEnsemble.predict_delta` retains the single physical projection of
  stochastic weather/power predictions and its raw out-of-range diagnostics.
- The fixed and astronomical baselines are updated. The astronomical baseline
  is renamed `sun_tracking` / `dual_axis_instantaneous`; it is no longer
  described as slew-limited.
- Counterfactual validation uses a real `hold` action instead of assuming
  `[0, 0]` means no movement. Azimuth-only counterfactuals hold tilt fixed.
- The MBPO dynamics checkpoint schema is incremented to 5. The resolved MDP
  records the new action semantics, preventing old SAC/MBPO checkpoints from
  being resumed under the new interpretation.

## 4. Nonlinear functions deliberately retained

These are not removed:
- SAC `tanh`: standard bounded squashed-Gaussian policy transformation;
- SAC log-standard-deviation bounds: numerical/variance safeguards;
- SAC and ensemble gradient-norm clipping: retained pending measured clipping
  frequency diagnostics;
- twin-critic `minimum`: standard clipped Double-Q target;
- azimuth modulo/wrap and physical tilt bounds in legacy modes;
- actuator `abs(delta)` terms: movement energy depends on distance, not sign;
- MBPO's single weather/power projection in `predict_delta`: physical state
  bounds, with raw out-of-range counts retained;
- weather-uncertainty lognormal numerical guard and physical albedo/GHI
  projections;
- `np.sign` in counterfactual pairwise agreement: diagnostic only.

The configured SAC alpha floor is deliberately unchanged. Removing it would
change a training hyperparameter and is not required for the action-contract
integration.

## 5. Apply safely

Use the exact reviewed `PostProcessing` source snapshot. Keep the patch bundle
outside the repository.

```bash
git switch PostProcessing
git switch -c integrated-instantaneous-target-v2
git status --short
```

The working tree must be clean. Then run:

```bash
python /path/to/patch_bundle/apply_integrated_action_patch.py --repo .
```

First perform a non-writing verification:

```bash
python /path/to/patch_bundle/apply_integrated_action_patch.py --repo . --dry-run
```

The installer verifies the Git blob hashes of all 12 source files, exact
replacement counts, Python syntax, static cross-pipeline consistency and
mathematical action tests before writing anything. It creates no backup files,
does not commit, and does not push. It rolls back from in-memory originals if
a write or `git diff --check` fails.

## 6. Verify after application

From the repository root:

```bash
python -m unittest discover -s /path/to/patch_bundle/tests -p "test_*.py"
python -m compileall -q mbpo SAC_workspace MBPO_workspace PV_experiments
git diff --check
git status --short
```

Then inspect the complete diff before committing:

```bash
git diff --stat
git diff
```

Do not resume an old SAC or MBPO checkpoint. Start a fresh run because the
action semantics have changed. The installer intentionally does not launch
training or alter configuration/hyperparameters.

## 7. Scope and verification boundary

The patch is pinned to the reviewed `PostProcessing` file contents. A source
hash mismatch is a hard stop, not an invitation to force-apply. The bundle
contains deterministic unit tests and compile/static checks. It does not claim
that a full 2,000-epoch training experiment or local NSRDB data run has been
executed; those must be performed in the user's actual workspace after
application.
