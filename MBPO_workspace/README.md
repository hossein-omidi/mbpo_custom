# Remastered MBPO workspace — PyTorch, no Ray/TensorFlow training

**Status:** locally unit/integration tested with mocked pvlib; **real NSRDB model-based training and final energy performance require validation on the complete repository**. Existing MBPO and plot scripts remain unchanged.

This is an **additive replacement training architecture**, not an in-place rewrite of `mbpo/algorithms/mbpo.py`. The old program remains available for reproducibility. The new MBPO uses the corrected **same** `SACAgent`, `SACEnvironment`, `SACConfig`, observation encoding, and real replay schema from your installed **SAC_workspace v2**, as well as the unmodified project `PVTrackingEnv`, `mbpo.env.pvlib_physics`, and `mbpo.env.nsrdb_weather`.

## Install

Extract the ZIP **at repository root** so you have both `SAC_workspace/` (v2) and `MBPO_workspace/`. Use your separate Python 3.10–3.12 SAC Conda environment, **not the Python 3.6 `mbpo` environment**. The project needs `torch`, `numpy<2`, `pandas`, `scipy`, `matplotlib`, `pvlib`, and `gym` installed by the existing `SAC_workspace/requirements-sac.txt`. Do not install Ray or legacy TensorFlow for this new workspace.

```bash
# from mbpo_custom/ root; using your SAC Python environment
python -m unittest discover -s SAC_workspace/tests -p 'test_*.py' -v
python -m unittest discover -s MBPO_workspace/tests -p 'test_*.py' -v
python -m MBPO_workspace.workflows.01_setup_config
# Only checks environment and captures two episodes; no learned model yet:
python -m MBPO_workspace.workflows.02_train --run-dir MBPO_workspace/runs/mbpo_smoke --seed 0 --epochs 2
# Actual model updates begin after reference warm-up of 8,000 real steps:
python -m MBPO_workspace.workflows.02_train --run-dir MBPO_workspace/runs/mbpo_seed0 --seed 0
```

Training can be resumed **only at the last checkpoint's exact history boundary**:

```bash
python -m MBPO_workspace.workflows.02_train \
  --run-dir MBPO_workspace/runs/mbpo_seed0 --seed 0 \
  --resume MBPO_workspace/runs/mbpo_seed0/latest.pt
```

Run a paired comparison when you have **both** trained policy checkpoints (standalone SAC and remastered MBPO):

```bash
python -m MBPO_workspace.workflows.03_evaluate \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_seed0/latest.pt \
  --sac-checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_seed0/episodes.csv \
  --sac-training-csv SAC_workspace/runs/sac_seed0/episodes.csv \
  --outdir MBPO_workspace/evaluation/five_scenarios
```

Full NSRDB manifest evaluation (can be computationally expensive, especially instantaneous optimized POA):

```bash
python -m MBPO_workspace.workflows.03_evaluate \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_seed0/latest.pt \
  --sac-checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
  --all-scenarios --skip-poa-opt \
  --outdir MBPO_workspace/evaluation/full_manifest
```

Use `--skip-poa-opt` only to omit the expensive *additional* instantaneous POA optimum. The ideal sun-pointing gross reference and feasible slew-limited dual-axis tracker are still evaluated.

## What is exactly shared?

- **MDP:** exact 6-component compact observation, 2-component normalized incremental actions, 117 actions per UTC daily episode, NSRDB native five-minute profiles, identical movement cost and simulator parameters. Both models call `PVTrackingEnv.step()` for real transitions.
- **SAC:** the **same class and same optimizer implementation**, with `gamma=1`, `reward_scale=100`, 64–64 actor/critic, two critics, target networks, automatic entropy tuning, and `min_alpha` from reference configuration.
- **Model-based components:** a probabilistic dynamics ensemble with five members and three elites, one-step **imaginary** rollouts, 0.85 real ratio in SAC minibatches, model fitting every 117 real steps, and retention of recent generated samples. These values are **parsed from the existing reference configuration** rather than duplicated.
- **No model-based operations during evaluation:** both trained actor networks are run directly against `PVTrackingEnv`.

The unmodified legacy MBPO can still be run with its original environment, and its TensorFlow/Ray-bound plotting pipeline is not deleted. The modern `plots.py` reuses the *SAC v2 plotting functions* without importing legacy TensorFlow. It also provides direct paired net/gross energy, seasonal/weather-scenario comparisons, 3D tilt/azimuth trajectories, and a combined PDF. Some specialized historical figures in `scripts/plot_paper_eval.py` are **not ported** to this workspace; that module imports the legacy TensorFlow evaluation stack indirectly and therefore cannot simply be imported safely into the modern environment.

## Model-specific design and safety decisions

1. **Known physics stays exact:** the ensemble predicts only changes in `DNI/2000`, `DHI/2000`, and `power/2000`. The panel tilt/azimuth transitions, horizon, and reward are calculated using the original reference actuation and energy helpers, not a neural approximation.
2. **No rollouts across episode boundaries:** synthetic terminal flags are derived from the exact remaining-time feature, and only real replay states with positive remaining horizon can start imagined steps. Rollout depth is **exactly one**.
3. **Group-level model holdout:** dynamics validation splits complete NSRDB scenarios, so adjacent samples from a scenario cannot be in both its model train and holdout sets. These model holdouts do *not* constitute disjoint final-policy evaluation scenarios.
4. **Model-quality gate:** if the held-out model fails to beat the last-observation persistence forecast on both power and combined dynamic channels, imaginary transitions are disabled until a subsequent satisfactory fit; SAC updates continue on real data. The reason and scores are logged. This is a conservative modification of original MBPO, **not** a theorem that any accepted model is unbiased.
5. **Recent synthetic replay:** capped by the reference four-epoch retention window and a 10,000-transition upper cap. Synthetic transitions never enter the real replay or train the dynamics model.
6. **No silent fallback:** missing NSRDB files, physical MDP changes, malformed actions, missing checkpoints or mismatched scenarios raise errors. Never silently use clear-sky weather.
7. **Checkpoints:** model and SAC networks, optimizer state, scalers, buffer states, environment RNG, configuration and physical model metadata are stored. `episodes.csv` is never silently overwritten.

### Accuracy of the POA and tracking references

- `fixed_no_motion` and `dual_axis_slew_limited` operate on the actual simulator and receive *net* reward using the project movement-cost definition.
- `dual_axis_instant_ideal` is ideal **sun pointing** and has no actuator constraints; gross energy only.
- `poa_instant_ideal` is an **instantaneous power** maximizer for the existing isotropic POA transposition model. Facing the sun maximizes the beam term for a fixed tilt, and a bounded scalar optimizer finds the best tilt while comparing the endpoints. Uses `mbpo.env.pvlib_physics.compute_panel_power_w`. Gross energy only. It is **not** a proof of optimal finite-horizon control and cannot be used as a physically achievable net-energy policy.
- Both learned agents and feasible physical references are evaluated on the **same scenario, seed, starting orientation and post-step UTC timestamps**. Do not call the default five validation days “held out”: the reference training pool may include them.

## Output files

Training: `MBPO_workspace/runs/<run>/config.json`, `resolved_mdp.json`, `episodes.csv`, `latest.pt`.

Paired evaluation: `summary.json`, `episodes.csv`, `trajectories.csv`, `comparison_report.pdf`, `01_gross_energy.png`, `02_net_energy.png`, `03_paired_net_differences.png`, `04_physical_trajectories.png`, `05_orientation_3d.png`, `06_scenario_energy_heatmap.png`, `07_seasonal_net_energy.png`, `08_training_learning_curves.png`, `09_dynamics_training_diagnostics.png`, plus `sac_reference/` plots reused from the standalone workspace.

### Known experimental limitations

- A 6D observation without solar position, calendar context and full weather history may be partially observable. The ensemble cannot infer omitted exogenous variables perfectly. One-step rollouts and quality gating reduce but do not remove model bias.
- The simulator is physics-based; simulated transitions are not field measurements. Field performance, weather forecast performance, convergence and improvement over SAC are **not yet established**.
- Training the probabilistic ensemble incurs extra wall-clock computation. The model minibatch size and maximum fitted real samples are training-efficiency decisions, not part of the shared MDP. Compare wall-clock and real simulator steps separately.
- A strict evaluation claim requires truly disjoint training and testing scenarios; the original reference pipeline samples the full manifest. Use a separately defined, common training split for *both* methods before claiming out-of-sample generalization.
- The original reference configuration has `movement_penalty=0`. As a result, “net energy” equals generated energy *under that exact simulated reward*, even though the tracker moves. A real motor-consumption model would require a carefully controlled change for all compared controllers.

See `docs/THEORY_AND_AUDIT.md` for equations and audit boundaries.
