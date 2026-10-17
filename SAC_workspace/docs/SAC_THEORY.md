# Independent finite-horizon SAC: theoretical contract

The policy sees the **existing** PVTrackingEnv compact observation
`[DNI/2000, DHI/2000, P/2000, tilt/90, (azimuth % 360)/180-1, (H-t)/H]`.
The environment has `H=117`, 5-min steps, normalized incremental actions
`[-1,1]^2` (physical maximum movements 5° tilt and 10° azimuth), NSRDB
weather, and kWh reward computed by `PVTrackingEnv.step()`.

### Learning objective

We use SAC-v2 (tanh-squashed Gaussian reparameterization):
`z = mu(s) + std(s) * eps`, `a = tanh(z)`, `eps ~ N(0,I)`.
The action log density includes the exact tanh Jacobian:

`log pi(a|s) = log N(z;mu,std) - sum_j log(1 - tanh(z_j)^2)`.

Independent target critics `Qbar1,Qbar2` use a clipped double-Q soft target:

`y = reward_scale*r + gamma*(1-done)*(min(Qbar1(s',a'),Qbar2(s',a')) - alpha*log pi(a'|s'))`

With `gamma=1.0`, `reward_scale=100.0`, and genuine day-end
`done=True`, the last transition does not bootstrap.

Critics minimize `MSE(Q1,y)+MSE(Q2,y)`.
The actor minimizes `mean(alpha*log pi(a|s) - min(Q1(s,a),Q2(s,a)))`.
Temperature minimizes `-mean(log_alpha*(log pi(a|s).detach() + target_entropy))`.
The effective temperature is `max(exp(log_alpha), min_alpha)`, where `min_alpha`
is extracted from the reference configuration (`0.001`).
Target critics update `theta_target <- (1-tau)*theta_target + tau*theta`.

The trained policy outputs normalized commands, not degrees. The environment
handles the physical motion, orientation clipping/wrapping, power and reward.
Reward scaling is restricted to Q targets; reported energy stays kWh.

### Important limitations

1. Reduced 6D observations do not contain full sun geometry/date/weather
   history, so the strict Markov condition is not automatically established.
   This is the **unchanged** observation contract for this experiment.
2. Using exactly the same reward and horizon does not guarantee convergent
   learning or equal model capacities if hyperparameters drift. We fail loudly
   on changed reference config, and record a snapshot.
3. This code uses the native NSRDB/pvlib simulator for training transitions.
   No learned dynamics, simulated imagination/replay, or MBPO BNN is involved.
4. Ideal instantaneous dual-axis pointing is a **gross-energy-only**
   sun-pointing reference, not a globally optimal finite-horizon policy or
   physical slew-limited controller. Net actuator energy is undefined.
5. Reference evaluation dates may be in training support. Results are paired,
   in-distribution, not evidence of generalization to unseen days.

### Validation requirements

- Validate log-density against a PyTorch transformed Normal distribution.
- Verify terminal mask and exact reward scaling on analytic targets.
- Run gradient/finite checks and target network-update tests.
- Resume a full episode checkpoint and compare subsequent trajectories.
- Run a full physical NSRDB smoke test on the original repository and check
  UTC timestamps, sample count, scenario IDs and generated energy.
- Do not claim publication-ready performance until multiple successful
  trained checkpoints and scenario-level evaluation statistics are available.
