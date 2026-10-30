# Stage 2: PVWatts+SAPM physical model integration and acceptance

**Canonical installation instructions:** read `STAGE2_DIRECT_COPY_README.md`.
Do not run previous `apply_pvwatts_stage2.py --apply` or perform a forced patch.

## Model and interface contract

- Both real training workspaces construct the shared `SACEnvironment`, which
  wraps `mbpo.env.pv_tracking.PVTrackingEnv`.
- `legacy`: original constant-efficiency power is retained and remains default;
  its previous effective-MDP metadata is preserved for historical checkpoints.
- `pvwatts_sapm`: same pvlib isotropic POA, SAPM steady-state cell temperature
  (using ambient air temperature plus 10 m adjusted wind), pvlib PVWatts DC.
  Total POA is used as an effective-irradiance approximation; no IAM, spectral,
  soiling, inverter, or transient thermal dynamics are added in this stage.
- `reset` and `step` share the selected backend. State remains 6D, action 2D;
  power normalization and reward identity are unchanged. New reward values are
  `power_dc_w * interval_hours / 1000 - movement_cost_kwh`.
- The new model is chosen by identical CLI arguments for both workspaces and
  is stored in the train/evaluation resolved MDP and checkpoint metadata.
  The source hashes of the power backend and irradiance helper are recorded.
- MBPO's model predicts DNI/DHI/power deltas, while the fake env reconstructs
  energy and reward from predicted normalized power. The ensemble must be
  retrained and assessed on real new-model transitions, using the existing
  held-out day skill gate. No model-skill gate is bypassed.
- Selected-model instantaneous references recompute DC power; the historical
  method ID `poa_instant_ideal` is retained for compatible charts, but displayed
  meaning is now *instantaneous gross DC reference* under the active backend.
- Detailed evaluation reporting exports source weather, real POA, effective
  irradiance, cell temperature, wind, chosen power model, and actual DC power.
  The legacy result remains available as a separate experimental baseline.

## Verify before new training

```bash
python scripts/verify_stage2_direct_copy.py
python -m pytest -q tests/test_pv_power_backend.py tests/test_pvwatts_stage2.py tests/test_pvwatts_stage2_patcher.py tests/test_stage2_all_workflow_source.py
python scripts/pvwatts_stage2_audit.py
```

## New-run command examples (after full audit PASS)

```bash
python -m SAC_workspace.workflows.03_train \
    --power-model pvwatts_sapm --wind-reference-height-m 2 \
    --run-dir SAC_workspace/runs/sac_pvwatts_seed0 --seed 0 --epochs 1

python -m MBPO_workspace.workflows.02_train \
    --power-model pvwatts_sapm --wind-reference-height-m 2 \
    --run-dir MBPO_workspace/runs/mbpo_pvwatts_seed0 --seed 0 --epochs 1
```

The 1-episode examples are training/plumbing smoke tests, not converged
experimental results. Evaluation should use new checkpoints produced under the
*same model settings* and disjoint held-out NSRDB scenarios for generalization
claims. Save model assumptions and source/checkpoint metadata with reports.

## Limits of verification

- The builder's source-level checks preserve local changes; they are not
  proof of predictive physical accuracy or RL convergence.
- Steady-state SAPM does not make the six-feature observation formally Markov;
  temperature and wind remain latent weather inputs. Evaluate held-out MBPO
  power-prediction skill before using model rollouts scientifically.
- `pvwatts_sapm` uses provisional cell-temperature parameters until actual
  mounting/module data are available.
- The user's own data-backed audit is the final installation acceptance gate.
