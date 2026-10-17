# V2 audit summary — PVTracking standalone SAC

**Scope:** `hossein-omidi/mbpo_custom`, branch `state-space-reduction`, read-only
reference reviewed 2026-10-15. Changes are limited to `SAC_workspace/`.
The trained policy has **not** been measured on actual NSRDB files in this
container; mocked energy is never presented as a research result.

## Substantive repairs from v1

1. Evaluation config now reconstructs the reference evaluation builder via
   restricted AST interpretation instead of inheriting train-only overrides.
2. Reference entropy-temperature lower bound `min_alpha=0.001` is propagated
   from the original config into the actor/target effective temperature.
3. Environment validates each true PV simulator reward identity,
   tau=(117-t)/117, UTC ISO timestamp, genuine 117-step termination, and
   native 5-minute NSRDB reporting; never reconstructs solar physics.
4. Training/evaluation now record/compare **resolved** simulator parameters
   (including omitted constructor defaults) and pvlib/Gym library versions.
5. Checkpoint schema, source config hash and effective physical model are
   verified before policy evaluation and when resuming training.
6. Replay-buffer storage validates observations, actions and reward; ring
   positions, RNGs, optimizer state, policy and target critics are serialized.
7. Dual-axis evaluation reuses a single simulator/NSRDB cache across methods,
   explicitly resets scenario and seed, validates exact post-action timestamps,
   starting pose and trajectory length; handles azimuth wrapping correctly.
8. Measured orientation movement, weather-paired net energy differences,
   per-scenario percentage gains and confidence intervals are exported.
9. Ideal instantaneous sun pointing is NOT a feasible net-energy comparator;
   it has gross energy only, and no finite-horizon global optimality claim.
10. Output trajectory UTC clock and plots use **real** simulator timestamp
    metadata rather than synthesizing a 12:00 start and five-minute increments.
11. Existing run directories are guarded before writing snapshots; no
    silent run overwrite; resume checks CSV/checkpoint alignment.
12. Added source-level config fixture and actual-repository 117-step parity
    regression test (requires your cloned repo and installed NSRDB data).

## Why PyTorch rather than replacing the agent with Stable-Baselines3?

SB3 uses Gymnasium's five-return `step` protocol and a different default
configuration/entropy convention; the existing PVTrackingEnv implements
legacy Gym's four-return protocol. Direct swapping would create a *second*
training interface, policy/checkpoint implementation, and a potentially
subtle mismatch with the reference 117-step terminal semantics. The current
work uses native PyTorch optimizers and distribution objects, while the
shared PVTrackingEnv remains a Gym environment. An SB3 adapter can be added
as an **independently verified reference algorithm** later, not silently
substituted for the custom SAC under audit.

## Theoretical caveats and protocol

- Actor: reparameterized tanh Gaussian with exact log-density change of
  variables; two learned Q functions; two frozen slowly updated targets.
- Soft target: `100*r + 1*(1-done)*(min(Qtargets)-alpha*log_pi)`, with no
  bootstrap on terminal daily transitions; alpha learned from log-alpha.
- Source config: 117 five-minute actions, 118 UTC timestamps from 12:00
  through 21:45, using **post-action** rewards at 12:05 .. 21:45.
- 6D compact observation is preserved exactly, although full Markov
  sufficiency is not proven because sun position/calendar are absent.
- Fixed no-motion and slew-limited dual-axis and SAC use the same feasible
  simulator step and energy cost; ideal instantaneous dual axis is a
  counterfactual gross-output reference only.
- The original movement_penalty is zero; thus feasible-policy net and gross
  agree **by definition of this simulator**, not because motor power is
  physically zero. Do not assert real-hardware net energy based on it.
- The built-in reference days may be sampled during training; no claim
  of out-of-distribution generalization may be made using them alone.

## Next acceptance gate

Run every test and all four workflows locally with `PVTrackingEnv` and
manifest. The real-parity integration test must PASS rather than skip.
Train at least one full 2,000-episode run, then review SAC reward, α,
critic loss, Q values, and scenario-paired net kWh against dual-axis tracking.
No convergence or energy superiority is guaranteed by unit tests alone.
