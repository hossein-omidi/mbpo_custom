# SAC vs MBPO — scientific investigation and critique (Stage 5)

**Scope:** Audit of the supplied Stage 4 diagnostic JSON/CSV and source code on GitHub branch `state-space-reduction`; no retraining, original final evaluation CSV, private NSRDB weather files, or original policy checkpoints were supplied to this environment. This report distinguishes what is **observed**, what is **inferred**, and what remains **unverified**. The original `SEND_ME_SAC_MBPO_DEEP_DIAGNOSTIC.zip` contains 1 JSON and 2 CSVs and a short README, **not evaluation figures, full paired evaluation episode data, or checkpoint weights**.

**GitHub source:** `hossein-omidi/mbpo_custom`, branch `state-space-reduction`, branch HEAD reported as `b0e27b3ee6cbb03bbf7e84d26386498d02e19c81` when accessed. Important modules inspected include `SAC_workspace/src/agents/sac_agent.py`, `SAC_workspace/src/training/trainer.py`, `SAC_workspace/src/training/replay_buffer.py`, `MBPO_workspace/src/trainer.py`, `MBPO_workspace/src/replay.py`, `MBPO_workspace/src/dynamics.py`, and `MBPO_workspace/src/evaluation.py`.

## A. Scientific conclusion

1. **Fixed tilt can outperform a learned SAC or MBPO policy on some days or even on the measured average.** This is fully compatible with reinforcement-learning theory. Because the fixed `30°/180°` orientation is feasible by choosing zero incremental actions from the same starting pose, and movement cost was zero in these experiments, the *optimal feasible controller* cannot perform worse than fixed on the same deterministic day. But neither trained actor is guaranteed to have found that optimum. Thus, `fixed > learned SAC` does not mean the physics is impossible or that data were manipulated; it is evidence of policy suboptimality **if the evaluations are matched and correct**.
2. **Stage 4 does not prove a SAC code bug or a failure of critic convergence.** Both shadow optimizer updates passed; terminal targets and reward/energy identities passed; checkpoint actions match previously saved trajectories extremely closely. These findings exclude some major implementation errors, but do not prove Q accuracy or convergence.
3. **Actor–critic calibration is the priority.** On four selected calendar days × three timepoints, some critic preferences fail to agree in sign with *deterministic physical* counterfactual returns. This is a useful anomaly but **not a one-to-one critic error test**: SAC Q predicts a frozen-policy **entropy-regularized stochastic continuation**, not a deterministic return. Two paired Monte Carlo continuations per branch are not sufficient to resolve the mismatch reliably.
4. **No evidence supports an immediate hyperparameter fix.** Neither alpha is stuck at its configured lower bound; small alpha is not proof of a broken entropy update. A 50k FIFO replay buffer is not inherently wrong, and a batch of 256 has high within-batch uniqueness.
5. **The reported MBPO +15% is not independently verified here.** That number appears in the submitted reviewer text, but neither uploaded ZIP contains the original `summary.json`, per-day evaluation `episodes.csv`, full rich trajectory file, or the figures needed to reconstruct and statistically test the claim. The diagnostic does reproduce learned **actions** from 336 complete evaluation trajectories, not the original **energy comparison**.
6. **Reproducibility issue in the GitHub commit:** the branch directory listing does not contain `SAC_workspace/src/evaluation/`, although `MBPO_workspace/src/evaluation.py` and `MBPO_workspace/workflows/03_evaluate.py` import it. A checkout of that branch alone may not be able to run the current evaluation pipeline. The generated output appears to have been created on a more complete local tree or a separately restored patch; this must be reconciled before claiming that the archived commit reproduces the figures. Run the included preflight on your actual machine and commit any missing source files intentionally.

## B. Verified algorithm: what SAC and MBPO actually optimize

For a real transition `(s,a,r,s',done)` with 6-dimensional compact `s`, 2-dimensional bounded incremental `a`, and 117 time steps per episode:

\[
y = 100r + (1-done)[\min(Q_{1,targ}(s',a'),Q_{2,targ}(s',a')) - \alpha\log\pi(a'\mid s')], \quad a'\sim\pi(\cdot\mid s').
\]

The critic minimizes two squared Bellman residuals; the actor minimizes `E[alpha*log_pi - min(Q1,Q2)]`; the alpha log-parameter tunes toward target entropy `-2`; the target critics follow Polyak averaging with `tau=0.005`. `gamma=1` is **not itself an error**, because trajectories terminate after 117 actions and terminal bootstrapping is masked. A finite-horizon problem can legitimately optimize undiscounted total daily energy. The actor samples from a properly tanh-squashed Gaussian with a stable analytical log-Jacobian correction in the inspected source; unit tests of the original Stage 4 package passed locally (14/14). Those tests do not establish that the entire PV simulator works on our host.

**SAC:** 234,000 real steps = 2,000 × 117; 452,000 policy/critic updates; warmup 8,000 steps; two updates per new step after warmup; real replay FIFO 50,000; uniformly sampled minibatches **with replacement**; 256 transitions per minibatch. The training days are sampled at reset and then weather progresses sequentially at five-minute intervals. Each SGD minibatch is composed of random transitions, **not one entire day**.

**MBPO:** 58,500 real steps = 500 × 117; 101,000 actor/critic optimizer steps after warmup; same `SACAgent` implementation and base hyperparameters. The real and synthetic buffers are separate; when available, each 256-size update uses 217 real transitions plus 39 synthetic (85%/15%). An ensemble of 5 neural dynamics predictors, 3 elites, predicts one-step deltas in 3 observation channels (normalized DNI, DHI, power). Pose/time kinematics and reward of synthetic transitions are reconstructed by the structured model. The model is refitted on a recent real replay distribution; last model pool has 1,600 synthetic entries (intentional 4 × 400 retention). This **does not** train MBPO's actor with a second algorithm. It changes the critic's training data distribution and bootstrap targets.

## C. Measured checkpoint evidence

| Diagnostic | SAC | MBPO | Interpretation |
|---|---:|---:|---|
| Real interactions | 234,000 | 58,500 | Unequal budgets (4× SAC) |
| Training episodes | 2,000 | 500 | 117 transitions per episode |
| Distinct training dates | 1,390 | 451 | Seasonal diversity already substantial |
| Episodes revisiting a training date | 610 | 49 | Repeat-days already occur naturally |
| Final alpha | 0.015580 | 0.012783 | **Both above** minimum 0.001 |
| Mean real replay absolute Q1 TD residual | 1.197 | 1.202 | Similar in-sample residuals; cannot compare generalization |
| Typical real replay batch unique fraction | 0.9972 | 0.9972 | No material within-batch duplication |
| Shadow optimizer step | Passed | Passed | Wiring works on tested checkpoint/batch |
| Terminal-mask errors | 0 | 0 | Checked replay terminal logic passes |
| Deterministic actions reconstructed | 38,976 | 38,976 | Reproduced checkpoint policy actions |
| Last-100 logged training entropy mean | -1.996 | -1.995 | Near configured target -2; not proof of policy quality |

The measured Q values around 35–40 are plausible in a reward-scaled finite daily horizon: average scaled reward was ~0.67 per step in the sampled real replay, and the remaining horizon is typically dozens of steps. A universal divergence cutoff at `|Q| > 200` is scientifically inappropriate; inspect horizon-conditioned targets and independent conditional returns instead.

**Counterfactual finding:** From the 12 branch states per method (4 dates × 3 steps), the mean predicted critic preference for the actor action instead of max-downward tilt is **+0.2745 scaled-Q units for SAC** and **+0.6227 for MBPO**. The measured **deterministic** continuation energy differences are **−0.000430 kWh SAC** and **−0.000195 kWh MBPO**, respectively. Critic sign agrees with deterministic energy on 7/12 SAC branches and 2/12 MBPO branches. These are small, correlated, nonrepresentative conditional contrasts, not final daily returns; the Q and physical numbers are *not interchangeable*. The added stochastic conditional soft-return diagnostic has only **2 Monte Carlo replications** per branch, so it is inconclusive. Notably this signal exists for **both** algorithms, not only SAC.

**Model skill:** For MBPO the sampled **training replay** one-step power delta RMSE is ~0.007668 normalized, compared with ~0.007887 for persistence, a ~2.8% reduction in RMSE on those samples. The most recent rotating holdout reports ~3.2% power skill. This is a modest edge and not independent evidence of reliable dynamics on new years/dates: the ensemble has been repeatedly trained on the persistent buffer using changing, potentially previously trained holdout scenarios. `uncertainty_max_std=None` means no additional per-transition rejection threshold is set. Avoid claiming the model is validated merely because its safety gate passed.

**Representation caveat:** observed power is divided by 2000 W, with replay mean ~0.036 in at least one real buffer. That may result in a relatively small state feature; it is worth inspecting activation or feature-sensitivity statistics, but is not established to be a bug and is shared between SAC and MBPO. Azimuth is normalized linearly with wraparound, potentially discontinuous near north; a follow-up probe may inspect boundary crossings without changing the 6-sensor constraint.

## D. Reviewer proposals: critical judgment

| Proposal | Judgment | Reason / next action |
|---|---|---|
| First audit fairness of shared physics, initial pose, evaluation scenarios | Agree | Reconstruct from full `summary.json`, `episodes.csv`, `trajectories_detailed.csv`, checkpoint hash and scenario selection plan; no mixed `legacy`/`pvwatts_sapm` |
| Observe alpha, Q, actor/critic loss over training | Agree | Current snapshots insufficient for convergence: evaluate checkpoint trajectories on a **fixed validation set** |
| Alpha reached minimum and is “likely the bug” | Disagree | Actual alpha 0.0128–0.0156, minimum 0.001; logged entropy near target |
| Require `alpha/reward_scale >= 0.1` | Disagree | No general threshold; depends on reward, differential entropy, Q advantages and state/action distribution |
| Scale **target entropy** with reward magnitude | Disagree | Target entropy is an entropy constraint in action space, not an energy-unit quantity |
| Reward scale 100 necessarily broken | Disagree | SAC is reward-scale sensitive; a controlled normalized-reward ablation is reasonable, but empirical evidence is required; full retraining both variants for apples-to-apples comparison. **Do not normalize by the same day's fixed total energy during deployment without documenting the future-information requirement and the change in day weighting** |
| Q exceeding 200 proves divergence | Disagree | Horizon, discount and reward range determine expected scale; compare against physically computed conditional returns and time-to-go |
| `tau=0.005` too slow with gamma=1 | Unsupported | Direction of tau adjustment cannot be deduced from gamma; experiment if target-lag diagnostics indicate cause |
| Q1 and Q2 agreement proves unbiased critic | Disagree | Two similarly biased critics can agree |
| Replay capacity should exceed 200k to avoid forgetting | Disagree as universal rule | Recent 50k FIFO is intentional and contains many days; increasing capacity changes replay distribution and recency, not merely statistical efficiency |
| Increase batch size to 512 | Optional controlled ablation | 256 batch already spans many transitions; no diversity crisis is shown |
| Repeat every training day multiple times | Not supported | SAC has already trained 2,000 episodes on 1,390 distinct dates; learn on day distribution with fair validation; forced repetition may overweight weather days |
| MBPO good under 6D means six sensors are provably Markov-sufficient | Disagree | Shared observation permits useful policies, but not proof of state sufficiency; different algorithms can compensate differently |
| Compare a 72° state with an arbitrary 30° action Q | Reject exact proposed code | One action can only change ±5° tilt; Q and power cannot be contrasted while both state and action vary. Branch from *same state*, execute feasible first action, and compare consistent conditional objectives |
| Match real-environment budget and use multiple training seeds | Strongly agree | SAC@500 episodes = **58,500 steps**; 3 seeds minimum, ideally 5; compare at shared training checkpoints and held-out evaluation protocol |
| Lower SAC loss proves convergence | Disagree | Bootstrapped, nonstationary targets permit low TD loss with poor action ranking; evaluate held-out return across checkpoints and seeds |

## E. Genuine command and repository integrity hazards

1. The proposed example `python -m SAC_workspace.workflows.03_train --config conf_compact --total-steps 58000 --output-dir ...` is **not valid for the committed workflow**. The real CLI takes `--epochs` (an **absolute episode count**), `--run-dir`, `--seed`, `--power-model`. **500 complete episodes = 58,500 steps**, not 58,000. The matching MBPO trainer is `python -m MBPO_workspace.workflows.02_train`.
2. Both training entry points and the paired evaluator use `--power-model` with default **`legacy`**, even though the Stage 4 checkpoints resolve **`pvwatts_sapm`**. This does **not** prove the existing experiments used the wrong backend (checkpoint/evaluation consistency rejects mixed modes); it does create an avoidable reproduction hazard. The pack prints verified commands with PVWatts explicitly specified.
3. The checked GitHub branch tree lacks the complete `SAC_workspace/src/evaluation` package, even though its MBPO evaluation imports `SAC_workspace.src.evaluation.evaluation` and `statistical_inference`, and workflow imports `scenario_sampling`. A fresh checkout may therefore be incomplete. Run `scripts/repo_preflight.py` **before any evaluation**. The preflight is protective; it deliberately does not fabricate or overwrite missing modules.
4. The original 336-day evaluation action reconstruction establishes that actor checkpoints are consistent with the **saved actions**. It cannot independently validate absolute energy totals, exact evaluation scenario independence, or figures not contained in the ZIP.
5. Other reviewer snippets are *pseudocode*, not executable project API: `cfg` is a configuration object in some scripts (not always a dict), `getattr(cfg, "power_model")` on a dict will not read a dict key, `SACAgent.load(...)` and `agent.Q(...)` do not exist in the inspected `SACAgent` class, and `SAC_workspace/workflows/04_evaluate.py` is not the required paired evaluator command. Copying those snippets into a paper reproduction script without correction would introduce additional failures.
6. The PVWatts physics checks in the original paired evaluator already compare checkpoint and live simulator metadata, including power backend and PV source file hashes. This is stronger than simply printing config properties. No need to weaken those checks.

## F. Experiments that discriminate real causes without invalid retuning

**Priority 0 — Freeze evidence and validate evaluation:** Save the exact Git commit and local untracked changes, checkpoint SHA256, config snapshot, model and baseline physics, exact CLI commands, full `summary.json`, per-day `episodes.csv`, `trajectories_detailed.csv`, and the scenario selection/weights. Reproduce all 117 timepoints and paired scenario IDs, verify baseline zero actions, initial 30°/180°, UTC times, power→kWh and reward−movement. Verify the claimed fixed-versus-SAC difference *per day* rather than from unmatched plots. Check for training-day overlap before calling any result held-out. Avoid interpreting the unconstrained instantaneous POA upper bound as a feasible net-energy controller.

**Priority 1 — Conditional critic calibration (no new training):** Choose fixed, precommitted development dates across seasons and random branch states; hold `s` fixed and compare a chosen actor action against `a_tilt=-1`, `a=0`, and a locally plausible incremental tilt change, all feasible. Estimate **the same frozen-policy stochastic soft-Q objective** by 16–64 paired Monte Carlo continuations with common random numbers (including entropy after the conditioned first action). Compare the Q gap and Monte Carlo gap with confidence estimates grouped **by day**. Also show deterministic physical-energy gaps separately. Use 12+ dates, not four, and do not tune on your final test set.

**Priority 2 — Learning progress and checkpoint comparability:** Run existing `latest.pt` and earlier saved checkpoints at 50,100,200,300,400,500,... epochs on one frozen development-day suite. Plot net-energy gain over fixed, percent below fixed, mean tilt and fraction of action saturation, Q prediction-calibration error and alpha against true environment steps. Training loss alone is inadequate. Require branch evidence for degradation or improvement.

**Priority 3 — Controlled budget and seed experiments:** SAC and MBPO at **500 episodes (58,500 real steps)** with seeds 0, 1, 2 (prefer five seeds if compute permits), common distinct train/dev/test partitions and the same evaluator. Compare full SAC@2,000 episodes as a separate asymptotic experiment. Report the random-seed spread, paired day-level differences and experiment CPU/GPU time separately. Vary just one SAC hyperparameter per experiment, only if diagnostic evidence motivates it: (a) fixed reward rescaling plus compatible temperature tuning; (b) replay 50k vs 200k; (c) batch 256 vs 512; (d) tau 0.005 vs an alternative. Do not automatically change all at once.

**Priority 4 — Model validation:** Independently hold out calendar days/years from all **dynamics model fits**, not a rotating holdout that can be reused over time. Measure model error by hour, season, irradiance, and control action; compare against persistence, test uncertainty calibration, and conduct a MBPO ablation **without synthetic samples** under the same real budget. A strong MBPO advantage is not necessarily caused by the one-step model unless this ablation demonstrates it.

## G. Minimal next data package and exact commands

Extract this ZIP **at the repository root**. The scripts do not mutate your original policies, checkpoints, training logs, code, or datasets. They write only to the specified output folder.

```bash
python scripts/repo_preflight.py --repo-root . --output stage5_repo_preflight.json
```

A nonzero exit code indicates a missing source file. Restore the exact missing file(s) from your authoritative local commit/patch **before** proceeding. Do not run evaluation from an incomplete checkout.

Then audit your actual *already generated* files (replace `pvwatts_seed0_heldout1` if your original evaluation directory is different):

```bash
python scripts/stage5_evidence_audit.py \
  --deep MBPO_workspace/evaluation/pvwatts_seed0_heldout1/deep_actor_critic/SEND_ME_SAC_MBPO_DEEP_DIAGNOSTIC.zip \
  --eval-dir MBPO_workspace/evaluation/pvwatts_seed0_heldout1 \
  --sac-training-csv SAC_workspace/runs/sac_pvwatts_new_seed0/episodes.csv \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_pvwatts_new_seed0/episodes.csv \
  --outdir MBPO_workspace/evaluation/pvwatts_seed0_heldout1/stage5_audit \
  --bundle-eval --bundle-plots
```

To generate safe examples for an *optional new* fair matched-budget run:

```bash
python scripts/print_verified_commands.py --seed 0 --sac-episodes 500
```

It **only prints** commands and does not execute training. Remember that `--all-scenarios` in the printed evaluation example is descriptive unless you enforce a previously defined unseen test split; do not use it for a test-set generalization claim.

**Send back:** `stage5_repo_preflight.json`, `SEND_ME_STAGE5_EVIDENCE.zip` produced in your stage5 audit output, the full original figures you wish to publish, any selection/stratification metadata, and the exact shell command used for original evaluation. If raw evaluation cannot be bundled, send `summary.json`, `episodes.csv`, `trajectories_detailed.csv` separately; these are necessary to verify the central claim about fixed vs SAC.

## H. What has (and has not) been changed

**Included protective improvements:** a read-only repository completeness preflight, an evaluation-data/action/energy/physics and pairing audit, a source-aligned command generator requiring explicit PVWatts physics, and offline unit tests. **No algorithm, state representation, replay implementation, reward objective, dynamics model, or hyperparameter in the original repo was changed** because the available evidence does not justify claiming a defect or selecting a repair. This is an intentionally conservative evidence-preserving scientific workflow.
