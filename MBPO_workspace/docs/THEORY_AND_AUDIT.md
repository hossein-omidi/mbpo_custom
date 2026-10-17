# Mathematical and interface audit — modern structured MBPO

## Inherited physical MDP

`SAC_workspace.src.config.config.load_config` extracts `conf_compact.py`, `_nsrdb_base.py`, and `0.py` without importing their Ray or TensorFlow modules. The modern MBPO **must use this same config object** (`MBPOConfig.sac`) and the same `SACEnvironment` and `SACAgent` classes. The original physical simulator (`mbpo/env/pv_tracking.py`) is not modified. Episode transitions are returned by the same `step` function and have the same post-action UTC observation and net reward.

`H = 117, gamma = 1, a in [-1,1]^2`, `obs in R^6`, `r = electrical_energy_kwh - movement_cost_kwh`. The value-function target is implemented by the exact shared SAC agent:

`y = 100*r + (1-d)*(min(Q_target_1(s', a'), Q_target_2(s', a')) - alpha*log pi(a'|s'))`.

Both policies use the same squashed, Jacobian-corrected Gaussian actions, twin critics, frozen targets and temperature objective. MBPO does **not** use a second, independently implemented SAC optimizer.

## Probabilistic dynamics ensemble

The original TensorFlow BNN predicts `(reward, next_obs - obs)` for all observation channels. The remaster applies an **explicit physics-based structural refinement**:

- Input: `x = [s, a]` (8 features), initially standardized from real replay training data.
- Target: `y = [s'_{DNI}-s_{DNI}, s'_{DHI}-s_{DHI}, s'_{P}-s_{P}]` (3 features), standardized using train-only statistics.
- Five independent probabilistic MLPs output Gaussian `mu` and `log sigma`, fitted by minimizing

  `E[(y-mu)^2/(2 sigma^2) + log(sigma) + log(sqrt(2*pi))]` summed across target features.

- Models are ranked by scenario-held-out Gaussian NLL. The best three constitute the elite set. Each generated step samples an elite and (by default) draws its learned Gaussian residual. The ensemble's inter-model mean standard deviation is logged as an epistemic proxy; it is not calibrated automatically.
- Network weights/Adam moments are retained between model fits and network standardizers are fixed from the first fit for a stable coordinate system. Training data are resampled from the latest *real* replay. A patience-limited best validation state is restored per fit.

**Exact analytical reconstruction:**

`tilt' = clip(tilt + a_tilt * max_delta_tilt, 0, 90)`

`az' = (az + a_az * max_delta_azimuth) mod 360`

`tau' = max(tau - 1/117, 0)`

Synthetic `r'` is reconstructed from predicted **next power** and the **original** project helper `energy_kwh_from_power_w()`, minus the original movement-cost formula. This faithfully matches the reference *reward definition*, but can still be numerically biased because predicted power is learned.

The model cannot guarantee a fully Markov physical state from 6D observations; no algorithm can infer genuinely missing solar/weather state perfectly. Therefore, use short rollouts, held-out model fit metrics, and direct real-environment evaluation.

## Synthetic rollout validity

Real replay contains the NSRDB scenario ID for **every transition**. Model training draws a holdout of complete scenario IDs. Imaginary rollouts start from nonterminal real states, have length **one**, and assign `done = (tau' == 0)`. The model never constructs synthetic time outside a real episode. The imaginary replay buffer is separate and its capacity corresponds to the legacy **four most recent model-generating epochs** (within the defined upper bound). On a failed validation-skill gate, the imaginary pool is emptied, avoiding reuse of a demonstrably weak model's predictions.

MBPO's SAC minibatch contains `floor(real_ratio * batch_size)` real simulator transitions and the remainder model-generated, sampled independently and shuffled. **If the imaginary pool is not sufficiently populated, the update is real-only**. Real replay never receives model-generated transitions.

The presence of model-generated transitions changes the critic's training distribution, *not* the reference physical MDP or evaluation simulator.

## How this implementation differs from legacy MBPO

| Area | Original | Modernized |
|---|---|---|
| Training process | Ray Tune + TF1 graph | Plain Python CLI + PyTorch |
| SAC optimizer | Internal TensorFlow implementation | **Imports tested SAC v2 optimizer** |
| Dynamics model | TF BNN Gaussian reward+full-state residual | PyTorch probabilistic ensemble; solar residuals only, known physics exact |
| Model validation | Transition-level random holdout | **Whole-NSRDB-scenario holdout** |
| Synthetic depth | Reference `max_model_rollout_length=1` | Forced 1 step, terminal mask from tau |
| Replay mixing | 85% real / 15% model | Same ratio parsed from reference; real-only fallback if model unavailable |
| Model data retention | Four model epochs, 10k hard upper cap | Same bounded FIFO intent, separate synthetic replay |
| Model-fit computation | Legacy time/early-stop budgets | Persistent optimizers, bounded dataset and epochs; configurable CPU budget |
| Logging | Ray progress / TF logs | CSV, configuration and MDP snapshots, full trusted checkpoint, validation diagnostics |
| Plotting | Legacy TensorFlow-dependent imports | Reuse SAC v2 plotting + modern 2D/3D/PDF reports; legacy scripts preserved |

## Trustworthiness and acceptance

**Verified locally:** Gaussian loss expression, full-scenario holdout independence, replay source ratios, analytically reconstructed kinematics/horizon/reward, a mocked complete training/model-fit/SAC-update/checkpoint-resume cycle, paired summary and figure production, and Python syntax/import checks. Tests are in `MBPO_workspace/tests`.

**Not yet verified:** real `pvlib`/NSRDB MBPO training, source-config resolution against the complete original clone at runtime, dynamics calibration on out-of-sample real scenarios, long-horizon convergence, implementation performance profiling, multi-seed statistical reliability, field-operation value, and improvements relative to SAC. Actual experiments are required before any scientific performance claim.

**Benchmark design:** evaluate feasible SAC, feasible MBPO, fixed no-motion, and rate-limited astronomical dual-axis on exactly the same complete NSRDB scenario trajectories, UTC timestamps, horizon and initial pose. The two instantaneous ideal references report **gross energy only**. The instantaneous scalar-optimized POA reference uses the **same project power model** (isotropic transposition) and is not the globally optimal finite-horizon controller. Provide scenario-clustered bootstrap confidence intervals of paired energy differences and separate real interactions from computational training time.
