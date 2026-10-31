# SAC vs MBPO: checkpoint, configuration, reward, normalization and gain audit (v2)

**Read-only.** This package does not edit SAC, MBPO, their configuration, the PVWatts simulator, replay contents, trained checkpoints, evaluation outputs or scientific figures. It adds two standalone scripts plus regression tests under your existing repository paths.

**Sources used:** the author's existing Stage 2/3 SAC/MBPO code, the submitted `pvwatts_seed0_heldout1.zip` evaluation and its `summary.json`, `episodes.csv`, `scenario_selection.json`, and `trajectories_detailed.csv`. No new sensors or reinforcement-learning hyperparameters are introduced.

## 1. Copy into the same repository

```bash
cd ~/projects/PVLIB_Paper/mbpo_custom
unzip -o PVWatts_SAC_Fairness_Diagnostic_v2.zip -d .
conda activate simple-worm
```

## 2. Run the fast numerical audit (no checkpoint required)

```bash
python scripts/sac_checkpoint_config_audit.py \
  --eval-dir MBPO_workspace/evaluation/pvwatts_seed0_heldout1
```

Recomputes the actual 336-day population-weighted results from the saved **per-day episode records** and the sampling weights. Checks correctly paired dates and initial panel poses, all energy and movement-cost identities, gross-only ideal reference, reported gain numerator and denominator, and shared SAC/MBPO configuration (including PVWatts–SAPM parameters). Exposes the observation power divisor and its discrepancy with the existing docstring, **not** a proven model error. Output: `.../sac_forensic/checkpoint_config_audit.json`.

## 3. Verify real checkpoint provenance, optimization settings and replay reward

The following command is strongest when each `.pt` is the **exact checkpoint used in the original evaluation**:

```bash
python scripts/sac_checkpoint_config_audit.py \
  --eval-dir MBPO_workspace/evaluation/pvwatts_seed0_heldout1 \
  --sac-checkpoint SAC_workspace/runs/sac_pvwatts_new_seed0/latest.pt \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_pvwatts_new_seed0/latest.pt \
  --sac-training-csv SAC_workspace/runs/sac_pvwatts_new_seed0/episodes.csv \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_pvwatts_new_seed0/episodes.csv \
  --require-checkpoints
```

The audit **fails** for an inconsistent saved SAC/MBPO configuration hash, changed physical source file fingerprint, invalid epoch/real-step count, mismatched `config.json` or `resolved_mdp.json` sidecar, checkpoint-vs-training-history discrepancy, mismatched SAC loss parameters, corrupt terminal flags, or real/synthetic replay reward inconsistent with the corresponding **next normalized power state** when movement penalty is zero.

Other outputs: effective alpha and whether at minimum, actual checkpoint epochs and real interactions, SAC/MBPO interaction-budget parity, synthetic experiences generated and used, recorded training-day overlap with selected evaluation days, and physical-model settings.

**Security:** `.pt` files are loaded using `torch.load(weights_only=False)` because the project's trusted training checkpoint contains a non-weight Python state. Run this on **only your own trusted checkpoints**. Never run it on downloaded/untrusted pickle files.

**Historical checkpoint limitation:** the existing evaluation `summary.json` does not record checkpoint SHA256 or epoch. Hashes computed today prove only the identity of today's file. It is therefore impossible to retrospectively prove the exact historical evaluation checkpoint *from `summary.json` alone*. A later `latest.pt` from continued training may not match the original evaluation. For the definitive behavior-level check, run the command below using the original `.pt`.

## 4. Reproduce the recorded SAC actor actions; inspect learned critic preferences

```bash
python scripts/sac_policy_forensic.py \
  --eval-dir MBPO_workspace/evaluation/pvwatts_seed0_heldout1 \
  --sac-checkpoint SAC_workspace/runs/sac_pvwatts_new_seed0/latest.pt \
  --sac-training-csv SAC_workspace/runs/sac_pvwatts_new_seed0/episodes.csv \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_pvwatts_new_seed0/episodes.csv
```

This reconstructs the **pre-action 6D observation from the previous CSV row**, tests every feasible action from step 2 through 117, then reports the deterministic policy's action difference. Step 1 is not reconstructed because the reset-time observation was never exported. The new v2 critic section compares the **trained SAC Q estimate** for its own action against a maximum downward tilt action (the same azimuth command), with actor Gaussian standard deviations and action saturation rates. These Q estimates are *not* physical ground truth or a proof that one action is globally optimal.

The original physical consistency and hindsight one-step tilt counterfactual remain in this command, and run over all paired dates.

## 5. Tests

```bash
python -m pytest -q \
  tests/test_sac_policy_forensic_offline.py \
  tests/test_sac_checkpoint_config_offline.py
```

These are fast offline unit tests; some test an intentionally small artificial replay or checkpoint. Separately run **steps 2–4 on your actual experiment** to assess your real SAC policy and its checkpoint.

## How to interpret the results

- A **PASS** for numeric reported gains does not validate policy convergence.
- `ratio_of_weighted_means_percent = 100 × (weighted SAC energy - weighted fixed energy) / weighted fixed energy` is different from the mean of daily percent changes.
- The existing `POWER_NORM=2000` is applied consistently to training and evaluation with the real/synthetic normalized power state. The docstring's `2000 × area × efficiency` is not the code's current scaling. A scaling ablation would require coordinated retraining of SAC and MBPO; the package does **not** make that change.
- Both methods share the same zero-penalty gross-energy reward in the current experiment, but the SAC TD target multiplies reward by `reward_scale=100` and has a separate entropy term.
- A current checkpoint can match its current CSV but still be **different** from the one used for an old evaluation. Check action replay before inferring training failure.
- The existing evaluator **does not automatically select a best policy**. It uses the `--sac-checkpoint` and `--mbpo-checkpoint` paths supplied for that run.
- Training with randomized initial orientation but evaluation with a fixed 30°/180° pose is identical across SAC and MBPO but is a potential training-to-deployment distribution difference. Do not silently modify it.
- A 336-day result with one RL training seed remains descriptive evidence, not proof of algorithm-level superiority or convergence.

## Verified from the user's supplied evaluation (no checkpoint required)

| Quantity | Verified result |
|---|---:|
| Selected evaluation days | 336 |
| Represented population days | 1,167 |
| Population-weighted SAC mean DC energy | 0.7487482291 kWh |
| Population-weighted fixed mean DC energy | 0.7454190285 kWh |
| SAC − fixed weighted gain | +0.0033292005 kWh (+0.446621%) |
| Population-weighted MBPO mean DC energy | 0.8575851613 kWh |
| MBPO − fixed weighted gain | +0.1121661328 kWh (+15.047393%) |
| SAC winning sampled days | 136/336 |

These numbers come from a complete independent recomputation from the supplied daily `episodes.csv` and actual stratum weights, not from the plotted lines.

**Runtime limitation in the assistant's environment:** the user's `.pt` checkpoints, local training-history CSVs and NSRDB files are not present. Only the offline tests and evaluation-file arithmetic were executed here. No checkpoint-specific result is asserted until you run the diagnostic locally.
