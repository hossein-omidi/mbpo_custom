# Stage A follow-up — verified hotfix and training configuration

This follow-up package is extracted over a repository that already has the Stage A replacement package installed.

## Fixes

1. **Stage 4 fingerprint mismatch:** the diagnostic now hashes exactly the same five ordered PV source modules as `SACEnvironment`: `pv_power_backend`, `pvlib_physics`, `pv_tracking`, `stage_a_pv_tracking`, and `stage_a_data_contract`. The check remains strict. A genuine mismatch reports saved/current digests and per-module current hashes.
2. **SAC-only dual-axis baseline:** the sun-pointing dual-axis reference now uses the recorded NSRDB `surface_albedo`, and fails if that value is absent for NSRDB. This avoids a silent fallback to pvlib's default 0.25.
3. **SAC alpha floor:** `log_alpha` is projected onto `log(min_alpha)` after updates and checkpoint loading. Outward Adam first-moment momentum is cleared at the floor so alpha can recover if the entropy gradient later changes direction. Effective alpha was already clamped in the actor/critic objectives; this fixes raw parameter drift and slow recovery.
4. **Config auditor:** `scripts/stage_a_config_audit.py` prints the effective shared SAC/MBPO settings and estimated interaction/update budget.

No observation/action dimensions, actor/critic architecture, PV backend, reward, MBPO rollout length, or real/synthetic replay ratio is otherwise changed.

## Install

```bash
unzip -o STAGE_A_FOLLOWUP_HOTFIX.zip -d .
```

The ZIP replaces `scripts/sac_mbpo_deep_diagnostic.py`, `SAC_workspace/src/evaluation/evaluation.py`, `SAC_workspace/src/agents/sac_agent.py`, and `README_STAGE_A.md`. It adds `scripts/stage_a_config_audit.py` and `tests/test_stage_a_followup.py`.

## Tests

```bash
python -m pytest -q tests/test_stage_a_followup.py
```

Then run the Stage A suite:

```bash
python -m pytest -q \
  tests/test_stage_a_physics.py \
  tests/test_stage_a_replay.py \
  tests/test_stage_a_multiseed.py \
  tests/test_sac_mbpo_deep_diagnostic.py \
  tests/test_stage_a_followup.py
```

The real-pvlib integration test in the Stage A physics suite must execute, not skip, on WSL.

## Why the diagnostic failed

The training environment fingerprinted five ordered files. The diagnostic fingerprinted only three. Different hashes were guaranteed even if the checkpoint and current physical environment were identical. The new diagnostic uses the exact five-file list and concatenation order. Do not disable the check.

## Current configuration review

The standalone configuration derived from `conf_compact` is:

| Setting | Current value | Assessment |
|---|---:|---|
| Horizon | 117 transitions | Preserve |
| Control interval | 5 minutes | Preserve |
| Observation/action | 6 / 2 | Preserve |
| Discount | 1.0 | Correct for finite-horizon undiscounted energy |
| Actor/critic hidden layers | 64, 64 | Reasonable baseline |
| SAC batch | 256 | Reasonable |
| SAC replay | 50,000 transitions | About 427 full episodes |
| Initial random exploration | 8,000 transitions | About 68.4 episodes |
| SAC updates per real step | 2 | Substantial update ratio |
| SAC learning rate | 3e-4 | Standard |
| Reward scale | 100 | Consistent with kWh-scale reward |
| Target tau | 0.005 | Standard conservative Polyak update |
| Alpha init/minimum | 1.0 / 0.001 | Retain initially; floor is now projected |
| Target entropy | -2 | Standard for two continuous action dimensions |
| MBPO real ratio | 0.85 | 85% real, 15% synthetic |
| MBPO rollout length | 1 | Preserve to limit model error |
| Dynamics ensemble | 5 models, 3 elites, hidden 64 | Reasonable compact ensemble |
| Model fit frequency | 117 transitions | About once per episode after warm-up |
| Dynamics fitting | 5 epochs, patience 5, batch 512 | Keep until held-out-day metrics justify change |
| Dynamics train cap | 6,000 samples | About 51 full days |
| Dynamics validation | 20% whole-day holdout | Preserve day-level separation |

Verify the values from your local code:

```bash
python scripts/stage_a_config_audit.py --epochs 500
python scripts/stage_a_config_audit.py --epochs 2000
```

### Is 500 epochs enough?

**500 episodes are a useful pilot, not a defensible convergence budget by themselves.**

One episode is 117 transitions.

| Budget | 500 episodes | 2,000 episodes |
|---|---:|---:|
| Real transitions | 58,500 | 234,000 |
| Random exploration | 8,000 | 8,000 |
| Estimated SAC gradient updates | 101,000 | 452,000 |
| Approx. MBPO model fits after warm-up | 432 | 1,932 |

The repository's `conf_compact` baseline specifies **2,000 epochs**. The earlier 500-epoch command was a shortened pilot. Use 2,000 as the initial reference budget, then verify convergence empirically; 2,000 is not automatically sufficient.

The standalone trainers do not run validation simply because `eval_interval=10` is present in the config. They log training episodes and save `latest.pt` at checkpoint intervals. Evaluate checkpoints separately on fixed development scenarios. Training return alone is not convergence evidence.

### Should the existing 500-episode checkpoints be resumed?

First, keep the current 500-episode checkpoints for the Stage 4 diagnosis. The fingerprint correction does not require retraining.

The alpha-floor correction changes how the *raw entropy-temperature parameter* is constrained. If the old run's `log_alpha` drifted below `log(min_alpha)`, loading it now projects it back to the floor. That is the intended correction, but it means the continuation is not exactly the same optimizer trajectory as the first 500 episodes.

Inspect the trusted local SAC checkpoint:

```bash
python -c "import torch,math; p='SAC_workspace/runs/sac_stage_a_seed1/latest.pt'; c=torch.load(p,map_location='cpu',weights_only=False); x=float(c['agent']['log_alpha'].item()); f=math.log(float(c['agent']['min_alpha'])); print('log_alpha=',x,'floor=',f,'below_floor=',x<f-1e-8)"
```

For the cleanest publication experiment, **start a fresh corrected 2,000-episode run** in a new directory. This avoids mixing the pre-fix and post-fix entropy-temperature update histories.

**SAC — corrected fresh run**

```bash
python -m SAC_workspace.workflows.03_train \
  --run-dir SAC_workspace/runs/sac_stage_a_final_seed1 \
  --seed 1 --epochs 2000 --device cpu \
  --power-model pvwatts_sapm \
  --wind-reference-height-m 2 \
  --gamma-pdc -0.004 \
  --sapm-mount open_rack_glass_polymer \
  --wind-powerlaw-exponent 0.14
```

**MBPO — corrected fresh run**

```bash
python -m MBPO_workspace.workflows.02_train \
  --run-dir MBPO_workspace/runs/mbpo_stage_a_final_seed1 \
  --seed 1 --epochs 2000 --device cpu \
  --power-model pvwatts_sapm \
  --wind-reference-height-m 2 \
  --gamma-pdc -0.004 \
  --sapm-mount open_rack_glass_polymer \
  --wind-powerlaw-exponent 0.14
```

If `below_floor=False`, the old run has not activated the projection. Resuming is technically compatible if the run's `config.json`, `resolved_mdp.json`, checkpoint and `episodes.csv` all match, but fresh runs remain easier to defend and reproduce. Do not overwrite the existing 500-episode runs.

For convergence evidence, compare checkpoints at 500, 1,000, 1,500 and 2,000 episodes on the same fixed development scenarios. The trainer overwrites `latest.pt`; stop at each comparison point and copy it to an epoch-specific filename before continuing. Keep final test scenarios untouched.

Use at least 3 independent training seeds as a minimum; 5 is preferable. Match seeds and interaction budgets between SAC and MBPO. A single seed-1 run is promising, not an algorithm-level conclusion.

## Rerun Stage 4 on the existing 500-episode checkpoints first

Do this before deciding to resume:

```bash
python scripts/sac_mbpo_deep_diagnostic.py \
  --sac-checkpoint SAC_workspace/runs/sac_stage_a_seed1/latest.pt \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_stage_a_seed1/latest.pt \
  --sac-training-csv SAC_workspace/runs/sac_stage_a_seed1/episodes.csv \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_stage_a_seed1/episodes.csv \
  --eval-dir MBPO_workspace/evaluation/stage_a_seed1_heldoutverified \
  --branch-days 4 \
  --outdir MBPO_workspace/evaluation/stage_a_deep_diagnostic
```

The backslash must be the last character on each continued shell line; no trailing spaces.

If the check still fails, do not bypass it. Send the new error including saved/current digests and per-module hashes.

## Gym warning

The `Gym has been unmaintained since 2022` message is a compatibility warning, not the cause of the fingerprint failure. Do not perform a broad Gym-to-Gymnasium migration in this hotfix: the project uses the legacy four-value `step()` contract. A migration needs a coordinated adapter and regression tests.

## Current evaluation interpretation

The reported means are encouraging: SAC is about 14.0% above fixed, MBPO about 6.3% above fixed, SAC about 7.3% above MBPO, and the instantaneous POA/DC reference about 6.1% above SAC.

Because `movement_penalty=0`, `net_energy_kwh` equals gross DC energy. It is not net generation after actuator electricity. Use that wording in the paper. Mean performance alone does not establish convergence or statistical significance.
