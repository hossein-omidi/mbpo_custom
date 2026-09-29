# MBPO-PV Tracker — Test / Train / Evaluation Guide

End-to-end instructions for running the MBPO + SAC intelligent PV tracker on
this machine. Everything runs locally; all paths are relative to the repo root
(`mbpo_custom/`). One conda env (`mbpo`) is used throughout.

```bash
source ~/miniconda3/etc/profile.d/conda.sh   # or: export CONDA_SH=<path>
conda activate mbpo                          # or: export CONDA_ENV_NAME=<name>
cd /home/ecer/projects/PVLIB_Paper/mbpo_custom
pip install -e .                             # once (train.sh also self-heals)
```

---

## 1. Project map (post-cleanup)

```
mbpo_custom/
├── train.sh                     # training launcher  -> runs/<run>/
├── result.sh                    # results launcher   -> runs/<run>/results/
├── setup.py                     # `pip install -e .` installs packages + `mbpo` CLI
├── requirements.txt             # curated pip list (PV stack)
├── environment/pv-env.yml       # conda env spec (+ requirements-pv.txt)
├── mbpo/                        # MBPO algorithm + PV environment
│   ├── algorithms/mbpo.py       # MBPO (on softlearning RLAlgorithm)
│   ├── models/                  # BNN ensemble (bnn, fc, constructor, fake_env, utils)
│   ├── env/
│   │   ├── pv_tracking.py       # PVTrackingEnv (gym id PVTracking-v0)
│   │   ├── nsrdb_weather.py     # NSRDB scenario loader / year cache
│   │   ├── nsrdb_iotools.py     # raw NSRDB CSV reader (shared)
│   │   ├── historical_weather.py# bundled PVGIS-TMY catalog (pvgis_tmy mode)
│   │   └── pvlib_physics.py     # POA irradiance / power
│   ├── static/pv_tracking.py    # static fns: termination, cyclic obs, etc.
│   └── utils/                   # filesystem, logging (Progress)
├── softlearning/                # SAC library (trimmed to used modules)
│   ├── algorithms/              # rl_algorithm, sac, utils(registry: SAC, MBPO)
│   ├── policies/  replay_pools/  samplers/  value_functions/  models/
│   ├── environments/            # gym adapter -> registers PVTracking-v0
│   └── scripts/console_scripts.py  # `python -m ... run_local|run_example_dry|run_example_debug`
├── examples/
│   ├── development/             # Ray Tune entry (main.py, base.py variant builder)
│   ├── config/pv_tracking/      # stage3_nsrdb, conf1..conf5, _nsrdb_base.py, conf_registry.py
│   └── instrument.py            # run_example_local/dry/debug
├── scripts/                     # pipeline + verification + plotting (see §5)
├── tests/                       # pytest suite
├── data/pv_weather/             # NSRDB 5-min CSVs + manifest, PVGIS TMY csv
├── nsrdb_newyork_5min/          # raw NSRDB API downloads (data provenance)
├── docs/GUIDE.md                # this file
└── docs/FULL_YEAR.md            # full-year protocol notes
```

Removed during cleanup (recoverable via git): `viskit/` submodule,
`NSRDB_original.py` (superseded bulk downloader), `verify_pendulum_setup.py`,
`demonstration/`, `.cursor_tmp_pkgs/`, dead softlearning modules
(`sql.py`, `kernel.py`, `plotter.py`, `real_nvp_flow.py`, `convnet.py`,
`remote/explore/extra_policy_info/dummy_sampler.py`,
`extra_policy_info/union/trajectory_replay_pool.py`, `value_function.py`,
`models/utils.py`, `utils/numpy.py`, `environments/helpers.py`,
`environments/gym/{mujoco,robotics,multi_goal.py}`, `environments/dm_control/`),
`examples/development/{variants.py,simulate_policy.py}`,
`examples/config/custom/`, and unused scripts
(`plot_ray_results.py`, `export_policy_weights.py`, `run_stage3_fullyear.sh`,
`collect_sun_demonstrations.py`, `evaluate_and_viskit.py`, `start_viskit.sh`,
`verify_stage3_fullyear.py`).

---

## 2. Data

NSRDB New-York-City 5-min data (2018–2024, UTC) is expected at
`data/pv_weather/nsrdb/newyork_{2018..2024}_5min.csv` plus manifest
`newyork_multiyear_manifest.json` (already present in this repo).

To regenerate from the official NSRDB API (needs API key):

```bash
python scripts/download_nsrdb_nyc.py          # 1) download raw SAM CSVs -> nsrdb_newyork_5min/
python scripts/prepare_nsrdb_newyork_catalog.py  # 2) clean + build data/pv_weather/nsrdb/
```

A bundled PVGIS-TMY catalog (`data/pv_weather/albuquerque_pvgis_tmy_utc_15min.csv`)
backs the non-training `historical`/`pvgis_tmy` weather mode.

---

## 3. Test

### 3.1 Unit / regression suite (fast, no training)

```bash
pytest tests/ -q          # 78 tests: env audit, NSRDB multiyear, checkpoints, plotting
```

### 3.2 Preflight wiring gate (run before any training)

```bash
python scripts/verify_preflight.py --config examples.config.pv_tracking.stage3_multiyear_nsrdb_scenario
# or via the launcher:
./train.sh <run_name> <conf> --verify    # runs preflight, then starts training
```

Preflight subprocess-checks: `verify_training_config.py`,
`validate_pv_rollouts.py`, `verify_movement_cost_fairness.py`,
`verify_utc_uniformity.py`, and for NSRDB configs also
`verify_nsrdb_multiyear.py` + `verify_nsrdb_timing_sync.py`.
Reports land in `verification/`.

### 3.3 Standalone verification / diagnosis tools

```bash
python scripts/check_pv_env.py            # env vs pvlib solar-position sanity
python scripts/audit_pv_workflow.py       # end-to-end env+model audit (11-D obs)
python scripts/solar_time_sanity.py       # time labels / solar position alignment
python tests/test1.py                     # script: NSRDB csv -> pvlib POA smoke
```

### 3.4 Dry-run the full training stack (no training steps)

```bash
python -m softlearning.scripts.console_scripts run_example_dry examples.development \
  --config=examples.config.pv_tracking.conf3 --cpus 2 --trial-cpus 1 --gpus=0 --trial-gpus=0
```
Prints the resolved variant spec and constructs a PVTracking env as smoke check.

---

## 4. Training

```bash
./train.sh <run_name> <conf> [--cpus N] [--trial-cpus M] [--gpus G] [--trial-gpus g] [--verify]
```

| conf | description (from `conf_registry.list_configs()`) |
|------|---------------------------------------------------|
| `stage3_nsrdb` | main reference (`stage3_multiyear_nsrdb_scenario.py`) |
| `conf1` | simple-strong — compact nets, fixed 30°S pose, real_ratio=0.9 |
| `conf2` | slow — long run, conservative MBPO |
| `conf3` | strong — deep SAC + wide BNN, high real_ratio (recommended) |
| `conf4` | light-fast — 64-wide nets, short run (smoke/iteration) |
| `conf5` | advanced — deep+wide A/B variant of conf3 |

Example:

```bash
./train.sh run1 conf3 --cpus 4 --trial-cpus 2 --verify
```

Outputs:

```
runs/run1/
├── run.json            # run metadata (conf, module, cpus)
├── trial_dir.txt       # auto-synced pointer to the active trial
├── ray_tmp/            # ray temp files
└── checkpoints/PVTracking/pv_tracking/seed:*/   # Ray Tune trial
    ├── progress.csv    # epoch metrics (training/return-average, evaluation/…)
    ├── checkpoint_*/   # policy + Q + model snapshots
    └── params.json
```

Env overrides: `CONDA_SH`, `CONDA_ENV_NAME`, `PV_CPU_PROFILE` (see
`scripts/pv_cpu_env.sh`), `MBPO_LOG_DIR`, `CPUS`, `TRIAL_CPUS`.

### 4.1 Monitoring during training

```bash
./result.sh run1 --status      # trial dir + latest checkpoint + tail of progress
./result.sh run1 --plot-only   # learning curves -> runs/run1/results/training/
```

### 4.2 Creating a new configuration (the intended extension point)

1. Copy `examples/config/pv_tracking/conf4.py` to e.g. `conf6.py`.
2. Adjust kwargs (see table below).
3. Register in `examples/config/pv_tracking/conf_registry.py`
   (`CONFIGS['conf6'] = ('examples.config.pv_tracking.conf6', 'conf6.py', '…')`;
   add to `NSRDB_CONF_NAMES` in `_nsrdb_base.py` if NSRDB-based).
4. Train: `./train.sh run6 conf6 --verify`.

Key knobs (all flow through `_nsrdb_base.build_nsrdb_params`):

| Group | Key | Effect |
|-------|-----|--------|
| Environment (`environment_kwargs`) | `start_time`, `periods`, `freq` | episode window (default 12:00 UTC, 118×5 min) |
| | `randomize_scenario` / `fixed_eval_scenarios` | scenario sampling vs fixed dates |
| | `movement_penalty`, `movement_cost_mode` | tracker movement cost (`legacy`/`geometry`) |
| | `irradiance_perturbation_std`, `observation_noise_std` | robustness/noise aug (conf4-style) |
| | `randomize_initial_orientation` | random vs fixed 30°S start pose |
| | `observation_mode` | `legacy` (6-D) vs `physical` (11-D) |
| | `weather_scenario_mode` | `nsrdb_multiyear` / `pvgis_tmy` / `clearsky` |
| Algorithm (`algo_kwargs`) | `n_epochs`, `epoch_length` (117), `n_initial_exploration_steps` | schedule |
| | `real_ratio`, `rollout_schedule`, `max_model_rollout_length` | MBPO mixing / model rollouts |
| | `n_train_repeat`, `model_train_freq`, `rollout_batch_size` | compute intensity |
| | `reward_scale`, `discount`, `target_entropy` | SAC |
| Networks | `policy_params_kwargs`, `q_params_kwargs`, `model_params_kwargs` | hidden sizes (`(64,)` … `(400,300)`), ensemble `num_networks`/`num_elites` |

Core physics/env logic: `mbpo/env/pv_tracking.py` (+ `pvlib_physics.py`,
`nsrdb_weather.py`); static termination/obs fns: `mbpo/static/pv_tracking.py`;
MBPO algorithm: `mbpo/algorithms/mbpo.py`; SAC: `softlearning/algorithms/sac.py`.

---

## 5. Evaluation

### 5.1 One-command results (recommended)

```bash
./result.sh <run_name> --full         # plots + MC eval + diagnostics + state-space check
./result.sh <run_name> --eval-only    # eval + diagnostics only
./result.sh <run_name> --plot-only    # training curves only (safe during training)
./result.sh <run_name> --num-rollouts 300 --eval-only   # bigger MC sample
```

Internal call for all NSRDB configs:
`scripts/evaluate_fullyear_mc.py <checkpoint> --date-set nsrdb_multiyear ...`
then `scripts/diagnose_tracking.py` and `scripts/verify_pv_state_space.py --mode nsrdb`.

### 5.2 Direct evaluation (custom conditions)

```bash
CKPT=runs/run1/checkpoints/PVTracking/pv_tracking/seed:*/checkpoint_000500   # pick yours

# NSRDB multi-year scenario MC (same protocol result.sh uses)
python scripts/evaluate_fullyear_mc.py "$CKPT" --outdir evaluation/run1_mc \
  --date-set nsrdb_multiyear --num-rollouts 120 --eval-protocol inherit \
  --policy-mode deterministic --max-path-length 117

# Held-out / stress / custom date sets
python scripts/evaluate_fullyear_mc.py "$CKPT" --date-set holdout      ...
python scripts/evaluate_fullyear_mc.py "$CKPT" --date-set stress_test  ...
python scripts/evaluate_fullyear_mc.py "$CKPT" --date-set custom \
  --fixed-eval-dates 2021-06-15,2022-12-15 --replicates-per-date 8 ...

# TMY-path evaluation vs baselines (historical PVGIS weather)
python scripts/evaluate_agent.py "$CKPT" --outdir evaluation/run1_tmy \
  --compare-baselines --num-rollouts 64 --eval-protocol inherit

# Trajectory figures for single rollouts
python scripts/plot_rollout_trajectory.py <rollout.csv> --outdir evaluation/figs

# Rank checkpoints of a run on hold-out energy
python scripts/select_best_checkpoint.py runs/run1

# Policy vs physical baselines (sun-tracker / fixed / etc.)
python scripts/compare_baselines.py "$CKPT" --num-rollouts 32
```

Outputs per eval: `summary.csv/.txt`, MC band plots, paired vs-oracle figures,
per-rollout PNGs/CSVs under the chosen `--outdir`.

---

## 7. Finite-Horizon Time-Aware MBPO (Theoretical Foundation)

This implementation uses a **finite-horizon MDP** formulation where the episode has a fixed
horizon `H = 117` control steps (8 hours of 5-minute intervals, 12:00–21:45 UTC).

### 7.1 Augmented State with Remaining Time

The agent's state is augmented with `τ` (remaining time fraction):

```
observation = [physical_state_11_dims, τ]   # 12 total (physical) / 16 total (legacy)
τ = remaining_steps / H   ∈ [0, 1]
```

At `t=0` (12:00): `τ = 1.0`; at `t=H` (21:45): `τ = 0` (terminal).

### 7.2 Deterministic Time Evolution

`τ` updates deterministically in both real and model rollouts:

```
τ_next = τ - 1/H
d = 1 if τ == 1/H else 0   # terminal flag
```

The dynamics model **does not predict τ** — it only predicts `(s', r)`.
FakeEnv overrides the model's output to enforce `τ_next = τ - 1/H`.

### 7.3 SAC Bellman Target (Finite-Horizon)

```
y = r + γ (1 - d) [ min_i Q_φ_i'(s', τ-1, a') - α log π_θ(a' | s', τ-1) ]
```

The next-state value uses `τ-1`, not `τ`. Discount `γ = 1` (undiscounted).

### 7.4 Configuration

No new config flags needed — finite-horizon is the native formulation.
All configs (`stage3_nsrdb`, `conf1`–`conf5`) automatically use `τ`-augmented observations.

### 7.5 Key Implementation Files

| Component | File | Change |
|-----------|------|--------|
| Environment obs | `mbpo/env/pv_tracking.py` | Appends `τ` to `_build_observation` |
| Observation bounds | `mbpo/env/pv_tracking.py` | Adds `τ ∈ [0,1]` to bounds |
| Static dims | `mbpo/static/pv_tracking.py` | `PHYSICAL_OBS_DIM = 12`, `LEGACY_OBS_DIM = 16` |
| FakeEnv | `mbpo/models/fake_env.py` | Deterministic `τ` override in `step()` |
| MBPO init | `mbpo/algorithms/mbpo.py` | Passes `horizon` to FakeEnv |
| SAC | `softlearning/algorithms/sac.py` | Uses `next_observations_ph` (includes `τ-1`) |
| Replay buffer | `softlearning/replay_pools/simple_replay_pool.py` | Stores `remaining_steps` (consistent with `τ`) |
| Eval decoding | `scripts/eval_utils.py` | Handles 12-dim physical obs with `τ` |

### 7.6 Verification

```bash
# Dry-run shows obs_space=(12,) for physical mode
python -m softlearning.scripts.console_scripts run_example_dry examples.development \
  --config=examples.config.pv_tracking.conf3 --gpus=0 --trial-gpus=0 --cpus=2 --trial-cpus=1

# Preflight gate confirms finite-horizon alignment
python scripts/verify_preflight.py --config examples.config.pv_tracking.conf3

# State space verification includes τ
python scripts/verify_pv_state_space.py --mode nsrdb
```

### 7.7 Mathematical Justification (Summary)

1. **Markov Property**: Without `τ`, states with identical physics but different remaining time are aliased (violates Markov). See Pardo et al. "Time Limits in RL" (2018).
2. **γ = 1 Valid**: Return `G = Σ r` is bounded by `H * r_max`.
3. **Boundary Condition**: `V(s, 0) = 0` at `τ = 0` (no future reward).
4. **Model Rollout Consistency**: Start states filtered by `remaining_steps > rollout_length` ensures `τ > k` throughout rollout.

---

## 8. Troubleshooting
