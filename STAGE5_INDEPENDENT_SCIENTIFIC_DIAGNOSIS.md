# Independent Stage 5 scientific diagnosis — SAC versus MBPO

**Assessment date:** 2026-10-30  
**Evidence:** User-supplied `SEND_ME_STAGE5_EVIDENCE.zip` (including `stage5_audit.json`, `evaluation/episodes.csv`, `evaluation/trajectories_detailed.csv`, `evaluation/scenario_selection.json`, `training/sac_episodes.csv`, `training/mbpo_episodes.csv` and original figures), the Stage 4 deep-diagnostic ZIP, and the `state-space-reduction` GitHub branch at commit `b0e27b3ee6cbb03bbf7e84d26386498d02e19c81`.

## Verdict

1. **Observed behavioral failure:** The evaluated model-free SAC policy over-tilts the panel across seasons. Its mean daylight tilt is approximately 76° overall, including 72.8° in summer when the recorded mean daylight solar zenith is approximately 39.3°. This systematically sacrifices spring and summer PV capture versus fixed orientation.
2. **Confirmed provenance/reporting mismatch:** The SAC *training history bundled with Stage 5* cannot belong to the SAC checkpoint that was characterized in Stage 4: at epoch 2,000, checkpoint alpha is 0.0155801819637 but uploaded training CSV alpha is 0.0224180519581. Last-100-episode reward, Q and loss means also disagree. MBPO history matches its checkpoint's prior diagnostics exactly. Consequently Stage 5 SAC learning curves do not establish properties of the evaluated SAC checkpoint.
3. **Physical-evaluation contract appears consistent:** All 336 days have matched weather, solar geometry, temperature and wind, identical starting pose, 117 steps and the same PVWatts/SAPM physical backend; the supplied audit passed the action/pose/energy checks. We find no evidence of a fixed-panel energy calculation error in these artifacts.
4. **Learning-mechanism root cause remains unidentified:** Shadow optimizer updates, terminal checks and replay sample checks pass. SAC critic miscalibration, incomplete state information, training dynamics and stochastic-vs-deterministic policy differences remain hypotheses, not confirmed implementation bugs.
5. **Scope limits:** One trained seed per method, an evaluation frame conditional on SAC training-day exclusions, zero movement penalty, and a model holdout protocol that rotates on persistent MBPO dynamics networks. Claims about statistically reproducible algorithm superiority and real motor-net energy require additional experiments.

## Recomputed paired evaluation

336 unique evaluation scenarios from 7 years (48/year), seed 42, one fixed initial orientation of tilt 30°, azimuth 180°. The design takes one evaluation day from each within-month temporal bin, then weights observations by the number of eligible days per bin. The total eligible post-exclusion population is 1,167 days (2,557 original frame days, 1,390 SAC training IDs excluded).

| Method | Unweighted sampled-day mean (kWh) | Design-weighted mean (kWh) | Weighted relative-to-fixed |
|---|---:|---:|---:|
| Fixed 30°/180° | 0.737267 | 0.745419 | — |
| SAC | 0.744942 | 0.748748 | +0.4466% |
| MBPO | 0.849442 | 0.857585 | +15.0474% |
| POA instantaneous gross oracle | — | 0.912487 | Gross-only, not actuator feasible |

Paired design-weighted differences: SAC minus fixed **+0.0033292 kWh**, MBPO minus fixed **+0.1121661 kWh**, and MBPO minus SAC **+0.1088369 kWh**. Unweighted daywise frequencies: SAC beats fixed on 136/336 and loses on 200/336; MBPO beats fixed on 278/336; MBPO beats SAC on 312/336. The distribution of SAC minus fixed is highly asymmetric: median **-0.022807 kWh**, mean **+0.007675 kWh** unweighted, range **[-0.185450, +0.321355]**. Thus the fixed baseline wins **most days**, but SAC's larger occasional wins make its mean slightly positive. Do not confuse the unweighted descriptive mean (+0.007675) with the design-weighted estimate (+0.003329).

Day-level paired/seasonal differences below use the original selection weights.

| Season | SAC minus fixed (kWh) | MBPO minus fixed (kWh) | SAC wins (raw selected days) |
|---|---:|---:|---:|
| Winter | +0.063506 | +0.108717 | 49/83 |
| Spring | -0.067605 | +0.109691 | 15/87 |
| Summer | -0.080028 | +0.091786 | 9/81 |
| Autumn | +0.106930 | +0.138287 | 63/85 |

Across each of the 7 separate evaluation years, design-weighted MBPO minus SAC remains positive (between +0.09385 and +0.11980 kWh). This is a repeatable descriptive year-by-year pattern within the one checkpoint pair, **not** a multi-training-seed uncertainty estimate.

## Physical mechanism of the SAC deficit

Computed from the original `trajectories_detailed.csv`, filtering **GHI > 50 W/m²**, to avoid interpreting nighttime panel orientations as productive behavior:

| Season | Mean solar zenith | SAC mean tilt | MBPO mean tilt | SAC fraction tilt >=80° |
|---|---:|---:|---:|---:|
| Winter | 67.7° | 81.6° | 50.7° | 63.5% |
| Spring | 46.3° | 76.0° | 48.4° | 50.8% |
| Summer | 39.3° | 72.8° | 47.2° | 44.0% |
| Autumn | 60.1° | 79.1° | 50.2° | 56.6% |

Using the solar-angle identity `cos(AOI) = cos(z)*cos(beta) + sin(z)*sin(beta)*cos(solar_az - panel_az)`, direct-irradiance-weighted mean cos(AOI) values in summer are approximately **0.660 for SAC, 0.766 for fixed, and 0.925 for MBPO**. The SAC policy can therefore be *energetically worse than fixed* even with its changing azimuth: its tilt is simply too steep for much of the bright summer window. This is direct evidence for the **mechanism of the observed energy deficit**, not yet why its neural actor adopted that policy.

Example, 2022-07-03: fixed=1.042905 kWh; SAC=0.857455 kWh; MBPO=1.139941 kWh. SAC loses 0.185450 kWh to fixed on that date, with mean lit-window tilt 73.44° against mean solar zenith 36.79°. Example, 2018-12-18: SAC=1.287075 kWh; fixed=0.965720 kWh. On that low-sun winter day steep tilt is helpful. The season-to-season reversal is consistent with solar geometry, not a computational contradiction.

## Verification of the physics comparison

- 1,344 episode records: 336 days x 4 methods (the fourth is the gross-only POA oracle).
- 157,248 detailed step rows; 117,936 feasible-controller transitions from 1,008 feasible trajectories (SAC/MBPO/fixed).
- For each `(scenario_id, step)` we independently compared DNI, DHI, GHI, solar zenith, solar azimuth, ambient temperature and wind speed across methods. Their recorded maximum absolute difference is **0** for every comparison.
- Each method starts at 30°/180°; the supplied Stage 5 contract audit reports no violations of movement or power-to-energy consistency.
- `movement_penalty=0.0`, therefore the present reported *net* energy equals generated energy within this simulator. It does **not** establish net production after realistic tracker-actuator electricity use.
- The instantaneous POA optimum is unconstrained gross-only and cannot be treated as a realizable controller.
- Train/evaluation IDs overlap **zero** for both agents. MBPO's 451 distinct training IDs are a subset of SAC's 1,390 training IDs. These are held-out **scenario IDs**, but not a validation of training-seed robustness or of all possible time-correlated meteorological leakage.

## Proven SAC training-history inconsistency

From the original Stage 4 diagnostic of the evaluated checkpoint `SAC_workspace/runs/sac_pvwatts_new_seed0/latest.pt`, SHA-256 `749e28902aede9d64154bfb27431b3a47522f33f43e6e85015f51db1c28f1688`:

| Quantity | Stage 4 matched checkpoint/history | Stage 5 SAC history |
|---|---:|---:|
| Epoch | 2000 | 2000 |
| Real steps | 234000 | 234000 |
| Critic/actor optimizer updates | 452000 | 452000 |
| Last alpha | **0.01558018196** | **0.02241805196** |
| Mean last-100 return (kWh) | 0.743684 | 0.824050 |
| Mean last-100 critic loss | 10.505924 | 15.954891 |
| Mean last-100 Q | 34.171192 | 40.160332 |

This numerical discrepancy rules out exact checkpoint–history correspondence. Same `epoch`, `env_steps` and `updates` is an insufficient identity test: multiple runs can use identical budgets. The Stage 5 audit did not detect this conflict. It is most likely a misselected SAC training CSV, but the exact cause requires comparing both original local checkpoint and CSV files. The MBPO results agree to numerical precision, so this issue is SAC-specific in the supplied bundle. The evaluated SAC policy actions themselves **did** match the Stage 4 checkpoint (38,976 tested actions; max absolute normalized difference < 4e-6), so training-plot provenance does not by itself disprove the paired evaluation.

Run the separate `check_checkpoint_training_provenance.py` from your repository root on the **trusted original** `.pt` and each candidate `episodes.csv`. It checks alpha as well as epochs, steps, updates and fingerprints.

## Actor, critic, replay and bootstrapping

- Both agents instantiate exactly the same `SACAgent` optimizer class, 64x64 actor/critic with clipped Double-Q, `gamma=1`, `reward_scale=100`, batch 256, two updates per environment step, target-network `tau=0.005`, `target_entropy=-2`, and initial alpha 1.0.
- The SAC backup uses `scale * r + gamma * (1-done) * (min Q_target(s',a') - alpha*log pi(a'|s'))`. This is mathematically plausible for finite-horizon soft SAC when terminal markers and `time-to-go` are correct. The prior diagnostics reported **zero terminal-mask mismatches** and passing shadow optimizer updates, not zero Bellman approximation error.
- Alpha of the tested SAC checkpoint is 0.01558, MBPO 0.01278; both above configured `min_alpha=0.001`. Small alpha may be worth ablation, but alpha hitting minimum is *not* the recorded failure mode. Q magnitudes on the order of tens are natural for reward scaled by 100 over up to 117 steps; Q-values alone do not demonstrate exploding Q.
- Replay is uniformly sampled with replacement from a FIFO buffer of capacity 50,000; Stage 4 sampling had approximately 99.7% unique transitions within a sampled 256-transition real batch. The 50,000 transition capacity covers about 427 daily episodes, a tradeoff that might affect coverage but is not proof of a bug. Simply increasing batch, replay or day repetition is not scientifically justified without a controlled ablation.
- The Stage 4 selected branch test has only 12 comparisons per agent from 4 days. Critic preference and deterministic real-return sign agreement was 7/12 for SAC and 2/12 for MBPO; these estimates are too small, temporally correlated, and have mismatched stochastic-soft versus deterministic objectives to label either critic broken.
- Both actors were updated in the shadow test. This verifies optimizer wiring, not good Q-policy calibration. The checkpoint alone cannot show global optimization convergence.

## MBPO model-specific concerns

- 500 training episodes (58,500 real transitions); 432 ensemble fits, 172,800 accepted generated one-step transitions over training and a final 1,600-transition synthetic replay pool. Approximately 217 real plus 39 imaginary samples are used in a 256-sample update once the model pool is populated.
- There is no configured epistemic-uncertainty rejection threshold (`uncertainty_max_std=None`), so a passing fit's rollouts are accepted without that gate; the model-skill gate remains active.
- On the sampled final training replay, the model beats a persistence power forecast by only **2.78% RMSE**. The prediction accuracy is not demonstrated on independent untouched test scenarios. Rotating holdouts across persistent network fits can reuse previously trained days as validation data in later iterations, rendering the per-fit holdout scores optimistic.
- These limitations do **not** negate the genuine model-free evaluation of the frozen MBPO-trained policy; they prevent a strong claim about model generalization or causal explanation of the observed performance difference.

## Recommended next tests and decision criteria

**P0 — Resolve provenance, no retraining:** Locate the original SAC CSV matching the exact checkpoint SHA; run `check_checkpoint_training_provenance.py` against candidate files. Save `git rev-parse HEAD`, the exact training/evaluation CLI, environment snapshot, and file SHA-256 checksums. The public `state-space-reduction` branch inspected at `b0e27b3e...` is missing `SAC_workspace/src/evaluation/` while the local experiment clearly used evaluation code; commit the reproducible evaluation source before publication.

**P1 — Validate critic on independent counterfactuals, not just training TD residual:** For fixed summer/spring states from held-out days (e.g. 2022-07-03), compare feasible `a_tilt in {-1,-0.5,0,0.5,1}` with fixed azimuth command and *the same pre-action state*. Evaluate exact first-step simulator reward and Monte Carlo continuation under the frozen policy, with common random numbers. Estimate both deterministic energy and entropy-regularized SAC soft returns in the same units, and report Q orderings/credible uncertainty. Do not compare Q values across fabricated inconsistent states or infeasible instantaneous tilt jumps.

**P2 — Policy behavior and representation:** Sample actor mean/std, boundary fractions, tilt/azimuth commands by season, irradiance regime and remaining horizon. Evaluate stochastic versus deterministic action deployment and sensitivity to initial orientation. Check observation aliasing: similar 6D observations with substantially different sun zenith/optimal orientations. Preserve 6D for the primary baseline; a richer-state ablation would diagnose information limitation but change the scientific task and should be reported separately.

**P3 — Training experiments:** After cleaning provenance, use 3–5 independent training seeds each for SAC/MBPO, a common fixed real-transition budget of 58,500 (500 episodes), and an extended SAC budget of 234,000 (2,000 episodes). Evaluate snapshots on a pre-specified development split and reserve an untouched final test split. Only subsequently test one-factor ablations of reward normalization / alpha scaling, replay size, batch size, and update-to-data ratio. Do not treat a larger batch or more repeats as the default fix.

**P4 — MBPO quality:** Fix a disjoint dynamics-validation subset *before all ensemble fits* (not rotating persistent fitted days); evaluate prediction bias, calibration, and 1-step kWh errors by month/solar angle. Compare MBPO with real-only SAC at matched steps and independent training seeds, plus an ablation without synthetic transitions.

**P5 — Publication limits:** Present current single-run paired results as *descriptive*, explicitly label the weather sample weighting and data-dependent post-training exclusions, and say gross DC energy with motor penalty set to zero. Do not infer confidence bounds from one observed day per nontrivial temporal bin or from one training seed. For real net gains, include a physically calibrated tracker power consumption model and repeat the matched comparison.

## Final interpretation

The fixed mount can outperform SAC in particular seasons and on most sampled individual days without contradicting optimal-control theory. Optimal feasible control can choose no movement; *an imperfectly learned SAC policy is not the optimum*. In this case there is direct physical evidence of SAC's too-steep learned orientation on spring/summer days, and direct evidence of a separately incorrect SAC training-history association in the Stage 5 reporting pipeline. The critic's role in producing the poor policy requires a calibrated counterfactual experiment. Do not rewrite the learning equations, change batch sizes or rerun expensive episodes before resolving the checkpoint–history provenance mismatch.
