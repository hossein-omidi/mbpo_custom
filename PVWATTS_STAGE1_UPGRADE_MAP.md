# Dependency hierarchy and future activation gates

This is an integration map, **not** a claim that the next-stage modifications
have already been applied.

## Inputs and power calculation

`NSRDB SAM CSV` -> `mbpo/env/nsrdb_iotools.py`
-> `mbpo/env/nsrdb_weather.py` (native 5-min weather and UTC index)
-> `PVTrackingEnv` weather profile including `dni`, `ghi`, `dhi`,
   `temperature` (= AMBIENT, not cell), `wind_speed`.

The PV orientation + solar position are resolved by `PVTrackingEnv`.
`mbpo/env/pvlib_physics.py::compute_poa_global` is authoritative POA.
The new Stage-1 `mbpo/env/pv_power_backend.py::compute_pv_power` exposes
candidate `legacy` and `pvwatts_sapm` results, but is **NOT CALLED BY THE ENV**.

## Minimal future activation — all components to change together

1. `mbpo/env/pv_tracking.py`: opt-in `power_model`, ambient temp + wind ->
   backend at **reset AND step**, set `info['power']` and the same 6D power
   observation. Preserve `info['poa_global']` as physical POA, not
   `power/(area*efficiency)`. Report `cell_temperature_c`, wind-reference
   metadata, model name, effective irradiance and relevant assumptions.
2. Shared config source and AST reader in
   `examples/config/pv_tracking/_nsrdb_base.py`,
   `examples/config/pv_tracking/conf_compact.py`, and
   `SAC_workspace/src/config/config.py`: add an **explicit new config path or
   fully audited mode selector** so both train/eval environments resolve to the
   *same* model. Preserve default legacy run snapshots byte-identical.
3. `MBPO_workspace/src/config.py`: keep inheritance of the shared SAC MDP;
   do not introduce an independent MBPO-only physical configuration.
4. `SAC_workspace/src/environment/environment.py` resolved MDP and saved
   snapshots, `MBPO_workspace/src/evaluation.py::PHYSICS_KEYS`, and local
   SAC evaluation loader: reject cross-model checkpoint usage. In particular,
   `MBPO_workspace/src/evaluation.py::ideal_instantaneous_poa` must call the
   selected physics backend, not hardcoded old `compute_panel_power_w`. Audit
   `SAC_workspace/src/evaluation/*` in the *local* repo: those files are not
   present in the remote branch at the time of this package's review.
5. `SAC_workspace/workflows/03_train.py`, `04_evaluate.py`,
   `MBPO_workspace/workflows/02_train.py`, `03_evaluate.py`: select the model
   in a single auditable way; use NEW run directories and checkpoints. Review
   any alternate workflow configs that instantiate `PVTrackingEnv` directly.
6. `SAC_workspace/src/representation/representation.py` &
   `SAC_workspace/src/agents/sac_agent.py`: **no shape change** needed for
   6D; normalized `power_w` now means selected-backend DC power, other indices
   unchanged. This is still a POMDP; temperature/wind need not be added to
   observation just to use steady-state SAPM. Test that `POWER_NORM=2000` still
   avoids clipping under new physical parameters.
7. `MBPO_workspace/src/dynamics.py`: ensemble learns 3 deltas `[DNI, DHI,
   power]`, NOT reward directly. 6D+2D network interfaces remain valid;
   retrain dynamics on new transition physics. Fake env computes reward from
   predicted power and executes known kinematics. Do NOT introduce duplicate
   pvlib calls into synthetic rollouts. Confirm held-out power prediction skill
   and acceptance masks before allowing large model-rollout fractions.
8. Paired baselines, statistical reports, CSV/figures, and offline reports:
   audit all direct POA->power formulas (`MBPO_workspace/src/advanced_suite.py`,
   `paper_reporting.py`, `verified_figures.py`, `plots.py`, plus LOCAL SAC
   evaluation package), ensure genuine POA is logged independently of DC power,
   and use identical candidate physics for all methods on the same date/seed.
9. V4.1/V5: retain existing legacy regression. New active-mode audit must
   verify SAPM and PVWatts against independent equations and real NSRDB inputs,
   full 117-step rollouts, reward/POA fields, paired checkpoint evaluation,
   and rejection of mismatched experiment physics.
10. Reward stays `power_dc_w * (5/60)/1000 - movement_cost`; with the current
    configured `movement_penalty=0`, it is gross DC energy. Changing watts
    changes numerical TD targets; recheck reward_scale and alpha diagnostics.

## Validation/staging

Stage 1 (this ZIP): **no changes to simulator or SAC/MBPO pipelines**.
Additive backend, exact old-model parity, temperature/wind/math tests,
read-only power sensitivity on legacy trajectories.

Stage 2: Wire opt-in physics at the canonical environment level;
shared configuration and report metadata; ensure old mode unchanged.

Stage 3: Revalidate baseline evaluators/POA oracle, paired episodes,
reporting and checkpoint physical identity.

Stage 4: Run fresh SAC optimizer and MBPO dynamics-fit/model-rollout smoke
experiments; examine held-out dynamics skill and observation clipping.

Stage 5: Fresh matched SAC/MBPO training and independent paired evaluation.

## Modeling assumptions that remain unverified

1. Actual NSRDB wind measurement height for these exact user files.
2. Panel module `gamma_pdc`, glass construction and thermal mounting type.
3. Terrain wind profile exponent for the chosen NYC site.
4. Whether POA-only orientation ranking remains sufficiently faithful under
   module optical losses and self-heating (requires sensitivity experiment).
5. Any claims of field-level accuracy require measured PV output.
