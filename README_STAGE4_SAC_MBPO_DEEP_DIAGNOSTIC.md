# Stage 4 — SAC + MBPO actor/critic, replay, target and rollout diagnosis

**Purpose:** Investigate poor SAC tilt decisions using the **exact checkpoints**, real/synthetic replay buffers, one-step SAC targets, policy/critic gradients, training histories, a non-mutating in-memory **shadow SAC optimizer update** (including actor, twin critics, alpha and Polyak targets), and optional physical finite-horizon continuations. The tool is read-only with respect to original project sources, checkpoints, logs and datasets. It writes outputs only to `--outdir`.

## Verified code contracts from the `state-space-reduction` source review

- SAC and MBPO both instantiate **the same** `SAC_workspace.src.agents.sac_agent.SACAgent` (Gaussian tanh actor, twin critics, entropy temperature, Polyak targets).
- The SAC and MBPO training loops use the same SAC hyperparameters, `start_steps`, `updates_per_step`, 117-step episode horizon and `gamma=1`; MBPO additionally uses separate real and synthetic replay.
- SAC target is `reward_scale * reward + gamma*(1-done)*(min(target_Q1,target_Q2) - alpha*log_pi(next_action))`.
- SAC real replay is a FIFO ring containing 50,000 recent transitions at the provided checkpoints. MBPO real replay is the same with scenario labels; model replay is separate.
- Scenarios are sampled at episode reset and their five-minute weather trajectories then unfold sequentially; the *minibatch sampler* samples transitions with replacement.
- MBPO's synthetic rollout length is 1, and the predictive model's rotating holdout split is not a persistent independent validation set. This package does not modify the safety gate or claim it is independent.
- The compact power observation is `P / 2000 W` in both workspaces. No observation/input change is made.

## Copy

From the repository root:

```bash
unzip -o PVWatts_Stage4_SAC_MBPO_Deep_Diagnostic.zip -d .
conda activate simple-worm
python -m pytest -q tests/test_sac_mbpo_deep_diagnostic.py
```

## Full read-only investigation (recommended)

```bash
cd ~/projects/PVLIB_Paper/mbpo_custom
conda activate simple-worm

python scripts/sac_mbpo_deep_diagnostic.py \
  --sac-checkpoint SAC_workspace/runs/sac_pvwatts_new_seed0/latest.pt \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_pvwatts_new_seed0/latest.pt \
  --sac-training-csv SAC_workspace/runs/sac_pvwatts_new_seed0/episodes.csv \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_pvwatts_new_seed0/episodes.csv \
  --eval-dir MBPO_workspace/evaluation/pvwatts_seed0_heldout1 \
  --outdir MBPO_workspace/evaluation/pvwatts_seed0_heldout1/deep_actor_critic \
  --samples 2048 \
  --branch-days 4 \
  --branch-steps 12,42,78 \
  --soft-mc-repeats 2 \
  --seed 712
```

This uses a random but calendar-balanced selection of four **predefined** diagnostic dates (January, April, July, October) rather than cherry-picking days where SAC loses. At steps 12/42/78, it checks the saved deterministic action, real simulator replay, and then compares the **same frozen policy** after a chosen first action or maximum downward-tilt first action. It also optionally evaluates two Monte Carlo estimates of the *conditional, entropy-regularized* SAC Q objective using the same random seeds for paired branches. `--soft-mc-repeats 2` is only a small exploratory Monte Carlo, not a reliable confidence interval.

### Faster first pass, without physical branches

If the 4-day physical branch run is too costly or fails due to local NSRDB paths, rerun with the same paths and `--branch-days 0 --soft-mc-repeats 0`. This still loads both checkpoints and compares actor/critic/replay, and reproduces deterministic evaluation actions (if `--eval-dir` is supplied). It explicitly records branches as `NOT_RUN`.

### Outputs

The script creates, *only within `--outdir`*:

- `deep_diagnostic.json`: checkpoint/config provenance; expected versus actual SAC updates; power/reward/terminal/FIFO consistency; scenario and month distribution; batch duplication rates; actual SAC Bellman targets, twin-Q disagreement, TD errors by remaining time, alpha and entropy; actor/critic gradient norms; **one cloned SAC optimizer step per checkpoint with Polyak arithmetic verification**; Q derivatives and action preferences; MBPO dynamics replay-prediction diagnostics and synthetic-versus-real comparison; exact SAC **and MBPO** evaluation action replay; optional physical continuation and soft-return evidence.
- `sampled_td_residuals.csv`: a compact diagnostic subset of up to 200 sampled transitions per pool, including target and twin critic outputs (not raw NSRDB weather data).
- `branched_real_continuations.csv`: sampled branch scenarios and steps, Q gaps and **real** continuing energy gaps; optional stochastic soft-return gaps.
- `SEND_ME_SAC_MBPO_DEEP_DIAGNOSTIC.zip`: the three previous files plus an interpretation note, **not** your models/checkpoints or NSRDB data.

**Upload `SEND_ME_SAC_MBPO_DEEP_DIAGNOSTIC.zip` here** for the next analysis.

## Interpretation — what the tests can and cannot establish

- **Target masking:** for terminal transitions, critic target must equal `reward_scale * reward`; elsewhere it includes the expected next-state soft value. A violation is a correctness defect.
- **Twin critics:** persistent disagreement and large TD residuals may indicate unstable Q learning, but a training-replay TD residual is not an independent estimate of policy-value prediction error.
- **Shadow optimizer:** performs ONE actual SAC `.update` on an independently instantiated agent loaded from the checkpoint. It verifies actor/critic weights change and target Polyak arithmetic; no changes are written to any source model. Passing does not mean Q estimates are correct.
- **Actor gradient:** useful for diagnosing near-flat or extreme updates; nonzero single-batch gradient does **not** prove non-convergence.
- **Q ranking versus finite-horizon return:** the deterministic physical continuation compares the same initial state and then follows the same learned policy. Its sign can challenge a claimed Q preference but does not use the exact entropy-regularized objective.
- **Optional soft MC:** subsequent actions are sampled from the Gaussian actor, and `-alpha * log_pi` is included *only after the first conditioned action*. This matches the semantics of `Q(s,a)` under the current fixed-alpha, finite-horizon objective. It is still a finite Monte Carlo estimate and cannot by itself prove a critic bug.
- **MBPO model skill:** replay-data RMSE is in-sample. The rotating holdout previously identified in the source means independent validation remains necessary; do not loosen its safety gate to generate more synthetic experience.
- **Batch size/epochs/day repeats:** the script measures distribution and replay diversity, not whether a hypothetical larger batch or more epochs would improve performance. That requires fresh controlled experiments and fixed development-policy evaluations.
- **Checkpoint hashes:** current checkpoint SHA256 is recorded. Historical evaluation provenance cannot be inferred from a checkpoint hash alone; both checkpoint actors are independently replayed against exported evaluation actions.
- **Dataset:** real branch runs use your local installed NSRDB+pvlib simulator. The offline tests here use mock physical states because the full NSRDB dataset and pvlib installation are not available in this execution environment.

## Safety

Do not load untrusted `.pt` files, because PyTorch's `weights_only=False` checkpoint loading permits Python object deserialization. Only use your own trained checkpoints. This script does *not* change any original source, agent, replay buffer, environment definition or training configuration. Outputs are diagnostic evidence, not an automatic retuning of the agent or a guarantee that SAC must outperform fixed or MBPO.
