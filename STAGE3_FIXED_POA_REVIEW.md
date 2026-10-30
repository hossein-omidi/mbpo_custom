# Stage 3: Fixed mount and instantaneous DC reference — source audit and correction

This package is a **direct-copy replacement** prepared against the user's uploaded
`pvwatts_stage2_sources.zip` plus their subsequent Stage-2 direct-replacement
package. It does **not** replace the shared PVTrackingEnv, the PVWatts power
backend, SAC/MBPO model weights, data loaders or training code.

## Scope and decisions

1. **Fixed mount (`fixed_no_motion`)**: unchanged. It sends `[0,0]` actions
   through `PVTrackingEnv.step()` on the same NSRDB weather, temperature, wind,
   cell-temperature model, and DC power equation as the learned controllers.
   With the configured nonrandom evaluation initial pose it represents a 30°
   tilt / 180° (south) static mount. It is *not* the globally optimal fixed
   panel for a season or year.
2. **Instantaneous reference (`poa_instant_ideal`)**: method ID is kept for
   compatibility. It now uses an **analytic global maximum of isotropic POA**
   over `[0°,90°]` tilt, with azimuth facing the sun. For PVWatts with SAPM,
   it mathematically checks that DC power increases with POA over the feasible
   interval, thus certifying the same orientation maximizes gross DC power.
   Otherwise the evaluator raises an error rather than falsely claiming an
   upper bound. Instantaneous values are based on post-step realized weather
   and have **no slew or actuator-energy constraints**. This is a gross-only
   upper reference, not a deployable tracking policy.
3. **Paired evaluation**: now enforces the same scenario ID, evaluation seed,
   initial pose, UTC timestamps, irradiance, solar position, ambient
   temperature, wind speed, PVWatts model parameters, and reward/power units.
   The instantaneous upper reference must dominate every observed controller
   on **gross DC power at each step**. This is fail-closed: no clipping or
   replacement of a noncompliant result.
4. **Representation**: verified gross-power comparison and monthly curves now
   use *gross DC energy for every method*. Feasible-only net-energy figures
   remain separate. Labels identify the instantaneous *DC* bound.
5. **Dual-axis sun-pointing reference**: unchanged, and retains its distinct
   meaning: unconstrained pointing at the sun, not globally optimized DC and
   not feasible net energy. The optional slew-limited astronomical tracker
   does use the real simulator, but also has access to future astronomical
   sun position; it is a privileged-ephemeris comparison, not an observation-
   matched controller.

The package does not add a bifacial model, IAM, inverter, PV module validation,
or temperature state. `pvlib.irradiance.poa_components()` is a decomposition
helper; replacing `get_total_irradiance()` with it would not, by itself, add
new physics. The pvfactors fixed-tilt example concerns **bifacial rear-side
irradiance** and is not required for this monofacial experiment.

## Direct installation

Extract the ZIP into the `mbpo_custom` repository root, preserving paths:

```bash
cd ~/projects/PVLIB_Paper/mbpo_custom
unzip -o PVWatts_Stage3_Fixed_Oracle_Fairness_Correction.zip -d .
```

No generator, installer, or new training checkpoint is required. The archive
contains only these changed/new paths:

- `mbpo/env/baseline_contract.py` (new; analytic isotropic upper geometry,
  PVWatts monotonicity and strict paired-trajectory contract)
- `MBPO_workspace/src/evaluation.py` (DC reference, paired validation)
- `SAC_workspace/src/evaluation/evaluation.py` (same paired validation)
- `MBPO_workspace/src/verified_figures.py` (consistent gross vs gross charts)
- `MBPO_workspace/src/advanced_suite.py` (correct scientific labels)
- `MBPO_workspace/src/plots.py` (correct scientific label)
- `tests/test_baseline_contract_stage3.py` (independent global-grid and
  fail-closed scientific-regression tests)
- `scripts/pvwatts_baseline_stage3_audit.py` (read-only real-NSRDB and independent
  pvlib physics audit)

## Verification steps

The 14 bundled source-level mathematical and strict-guard tests ran successfully
in the packaging environment, and all replacement Python files compiled.
That environment does *not* contain pvlib or the user's NSRDB dataset, so the
following real-data checks must be run in `simple-worm` before publication.

```bash
conda activate simple-worm
python -m pytest -q tests/test_baseline_contract_stage3.py
python scripts/pvwatts_baseline_stage3_audit.py --scenario 2022-07-15
python scripts/pvwatts_baseline_stage3_audit.py --scenario 2022-01-15
```

If you already have **both** trained PVWatts checkpoints, include them for an
actual four/six-method comparison using the real policies:

```bash
python scripts/pvwatts_baseline_stage3_audit.py \
  --scenario 2022-07-15 \
  --sac-checkpoint SAC_workspace/runs/sac_pvwatts_new_seed0/latest.pt \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_pvwatts_new_seed0/latest.pt
```

If either saved checkpoint was trained under different model parameters, the
existing checkpoint metadata checks **must fail**, as designed.

Finally, run the original paired evaluation workflow with your existing
`--power-model pvwatts_sapm --wind-reference-height-m 2` parameters and your
usual held-out-scenario selection. The `verified/` and `advanced_suite/`
figures will reflect the consistent gross-energy semantics.

## Mathematical note

For an isotropic monofacial surface facing solar azimuth, write zenith `z`,
tilt `b`, ground albedo `rho=0.25`, and `G` the plane-of-array irradiance:

```
G(b) = DNI * max(0, cos(z-b))
     + DHI*(1+cos(b))/2
     + rho*GHI*(1-cos(b))/2.
```

Its global maximum over `b in [0,90 degrees]` is attained at one of: the two
interval endpoints, the beam-active stationary point
`atan2(DNI*sin(z), DNI*cos(z)+(DHI-rho*GHI)/2)`, or the beam-clipping
boundary `b=z-90` when `z>90`. No stochastic optimization is required.

With the chosen SAPM steady-state model, cell temperature is affine in G at
fixed ambient temperature/wind. Standard PVWatts DC is then a concave
quadratic in G, and its minimum derivative on the feasible POA interval is
attained at the maximal POA. The new monotonicity check enforces a
nonnegative derivative at that point before claiming the DC upper bound.

## Boundaries and limitations

- The gross DC upper reference is **not** a feasible controller and must not be
  compared as if its uncharged instantaneous motion incurred zero real motor
  energy. Its `net_energy_kwh` and `movement_cost_kwh` remain `None`.
- Fair source-weather/model parity does not imply that SAC/MBPO and an
  astronomical tracker have the same *available information*. The oracle
  uses realized post-step weather; the ephemeris-based slew controller has
  privileged sun position; keep these distinctions in paper descriptions.
- A 30° south-facing fixed panel is a **reference orientation**, not an
  optimized fixed mount. Add a *separately calibrated training-only optimal
  fixed* baseline only if that stronger question matters to your paper.
- These are modeled **DC** energy figures over 117 five-minute steps, not
  measured PV-system AC output or full-day energy.
