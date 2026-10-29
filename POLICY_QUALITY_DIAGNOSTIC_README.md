# PV SAC / MBPO — frozen-policy evaluation & improvement diagnostic (v1)

**Purpose.** Decide whether the existing trained critics and policies choose scientifically defensible actions under the project's **unchanged six-dimensional compact observation, 117-step NSRDB/pvlib environment, physical action limits and reward**. This is a **read-only test**; not a patch to the algorithms. Run from the `mbpo_custom` repository root in the existing Python environment. Only trusted, locally generated `.pt` checkpoints should be loaded.

## Installation

Unzip the three files into the repository root, alongside `plot_modern_results.sh`. No package installation, training, or changes to other files are needed. The test has the same Python/NumPy/PyTorch and PV environment dependencies already required by the project.

## Fast probe (start here)

```bash
python pv_policy_quality_diagnostic.py \
  --sac-checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_seed0/latest.pt \
  --eval-dir MBPO_workspace/evaluation/scientific_v52_fresh \
  --sac-training-csv SAC_workspace/runs/sac_seed0/episodes.csv \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_seed0/episodes.csv \
  --days 2024-12-23 \
  --steps 35 65 \
  --repeats 3 \
  --model-diagnostic \
  --outdir MBPO_workspace/evaluation/policy_quality_fast
```

Run `python test_policy_quality_diagnostic.py` for seven fast synthetic unit tests (no NSRDB data or trained checkpoints needed).

## Broader policy probe

Run AFTER the fast probe succeeds. More repeats reduce Monte Carlo noise from stochastic future actions, but DO NOT increase independent weather samples:

```bash
python pv_policy_quality_diagnostic.py \
  --sac-checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_seed0/latest.pt \
  --eval-dir MBPO_workspace/evaluation/scientific_v52_fresh \
  --days 2024-12-23 2018-12-27 2021-07-02 \
  --steps 25 50 75 \
  --repeats 5 \
  --model-diagnostic \
  --outdir MBPO_workspace/evaluation/policy_quality_broader
```

This can take much longer than plotting: each candidate requires simulator replays. No shortcuts mutate private simulator state; every candidate is tested by resetting the same date, replaying the same policy-generated prefix, applying ONE feasible alternative action, and continuing in the real simulator.

## Design and units

For each `(policy, weather day, 0-based step)`:

1. Replay the frozen **deterministic** actor from the reference initial pose and date; when `--eval-dir` is given, fail if the replay total differs from the saved matched episode by >`1e-5` kWh.
2. Build a finite test set of feasible *normalized* actions: actor's own action, zero, and actor action ±`candidate-step` on the tilt and azimuth channels, clipped to `[-1,+1]`. De-duplicate identical actions. These are feasible actions, NOT privileged globally optimal actions.
3. For each first-action candidate, follow the **same frozen deterministic actor afterward**. The difference versus the actor baseline is the realized **complete remaining-episode energy** change (kWh) conditional on the selected future weather. This measures local finite-candidate improvement opportunity; it is not a deployable controller's generalization estimate.
4. Also draw `--repeats` independent **stochastic** actor continuations for each first-action candidate, using the same repeat seeds across candidates. With current `gamma=1`, compute the soft-action Q Monte Carlo target

   `Q_MC(s_t,a_t) = sum_{j=t}^{H-1}(reward_scale * reward_j) - alpha * sum_{j=t+1}^{H-1}log pi(a_j|s_j)`.

   The *first* action has **no** entropy term because Q conditions on that action. Report `min(Q1,Q2)` from the trained critic, MC mean and sample SD in **scaled Q units**, and both divided by `reward_scale` as energy-equivalent units (these include entropy and are **not** physical generated kWh).
5. Calculate the *one-step SAC soft Bellman residual* using the target critics, sampled next actions, and terminal mask. Small residual only means self-consistency with the **current target**; it does not prove the learned policy is good.
6. If `--model-diagnostic` is enabled, load the MBPO dynamics ensemble checkpoint **read-only** and calculate one-step RMSE versus persistence for DNI, DHI and power over **real visited state/action transitions**. These dates are NOT assumed to have been permanently excluded from dynamics training; the metric is diagnostic, not an unseen-data validation claim.

Outputs:
- `candidate_scores.csv`: per-candidate deterministic and stochastic scores, Q/MC discrepancies.
- `quality_diagnostic.json`: concise per-probe action-ranking agreement, local deterministic candidate gain, Bellman residual, model RMSE diagnostics and training-history date overlap.

## How to interpret

- **Realized local gain > 0**: one tested intervention achieves higher remaining-episode energy on *that same recorded weather day*. This shows the actor is not locally optimal with hindsight over the finite tested candidates; it does not identify a universal policy improvement.
- **Critic versus MC candidate ranking disagreement**: reason to investigate value-function approximation, training coverage and loss weighting. A single-day MC continuation freezes future weather, whereas the critic has learned to average over weather encountered during training. This disagreement alone is NOT proof of an incorrect Bellman formula.
- **Large Bellman residual**: critic/target-network self-consistency issue worth following up; may reflect high approximation error or undertraining. Small residual does NOT establish useful ranking.
- **Negative ensemble power skill**: on the visited real actions, the ensemble-mean one-step power estimate is worse than a persistence predictor. That may undermine synthetic Q updates. It does not measure unseen-action or held-out-date performance.
- **Convergence:** one `latest.pt` cannot establish convergence. Save independent periodic snapshots in a future experiment and evaluate on an identical validation panel.
- **Generalization:** overlap with dates named in training histories can expose leakage. Lack of observed overlap cannot establish data independence when both training and evaluation draw from the same 2018–2024 scenario catalog. Strict held-out inference needs a pre-designated test split excluded from *all training*.

Avoid attaching confidence intervals to cross-day performance on the basis of repeated stochastic continuation draws: these are **conditional action randomness**, not independently sampled NSRDB dates. The script does not fabricate statistical significance thresholds or apply policy updates.

## Compatibility and limits

- Assumes the current repo state from v5.3 with `MBPO_workspace.src.evaluation.load_mbpo_agent`, `SAC_workspace.src.evaluation.evaluation.load_agent`, 6D observations and 117-step PV episode.
- `--steps` are 0-based action indices from 0 to 116.
- Runs on CPU; the PV physical simulation is the time-consuming part.
- Does not modify the four paper-method comparisons or `plot_modern_results.sh`.
- The supplied synthetic tests exercise numerical identities, replay invariance and intervention logic, but full trained-model behavior must be verified locally using real NSRDB and the user's checkpoints.
