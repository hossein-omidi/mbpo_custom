# Stage 5.1 — Independent critical review of two external assessments

**Date:** 2026-10-30  
**Scope:** The two user-provided critiques, Stage 4 `SEND_ME_SAC_MBPO_DEEP_DIAGNOSTIC.zip`, Stage 5 `SEND_ME_STAGE5_EVIDENCE.zip`, pinned `state-space-reduction` code as previously inspected at `b0e27b3ee6cbb03bbf7e84d26386498d02e19c81`, and the Stage 5 audit/test source bundled in the earlier diagnostic. No retraining or new pvlib simulation was executed.

## Overall judgment

The critics are strongest when they ask for clearer research claims and more robust audit *tests*. They are weakest when they infer **learning mechanisms** from small, non-equivalent TD/return probes or propose numerical reward alterations without deriving them from the SAC/MBPO equations. We should **fix the diagnostic tests and provenance checks now**, **not** change the actor, critic, replay capacity, entropy temperature or dynamics reward on this evidence.

### Critique A: test coverage — accepted/qualified/rejected

| Critique | Decision | Specific reasoning and action |
|---|---|---|
| Original seven Stage 5 tests check episode pairing, reward identity and action/tilt consistency | **Accept** | The tests operate on synthetic fixtures; passing establishes only these code paths. Preserved and extended. |
| Only 1,008 feasible trajectories validated, excluding 336 `poa_instant_ideal` paths | **Accept with correction** | True for the action contract: 1,008 = 336×3. The audit's *unweighted-day* summary contains **three** methods, while the attached original evaluation `summary.json` contains **four** methods. We now explicitly report the exclusion and run a separate gross-only oracle consistency contract on all 336 oracle paths. No claim of independently optimizing PVWatts is made. |
| Shadow-update/action replay/train-test overlap lack Stage 5 unit tests | **Accept as a coverage limitation** | Stage 4 **actually ran** a cloned optimizer step and checkpoint action reproduction; the Stage 5 unit suite had not recreated those operations. New negative tests ensure Stage 5 flags tampered Stage 4 status, replay claims, and overlap. These are **not** independent live checkpoint tests. Stage 4 replay checked steps 2–117 (38,976 actions = 336×116) per method, **not all 117 first actions**. The independent tool is bundled for local rerun with trusted `.pt` files. |
| Fixture used invalid UTC times like `00:060:00`; strings could pass | **Accept, confirmed diagnostic bug** | The old audit compared timestamp strings only. We replaced this with ISO parsing, explicit UTC offset, same scenario date, exact first 12:05 UTC, 117 monotonic 5-minute steps, and paired-clock equality. Fixture repaired and malformed/gapped times are rejected in tests. Original real data passes. |
| MBPO 2/12 versus SAC 7/12 sign agreement proves SAC has a better critic | **Reject** | Different critic networks, four days, correlated within-day points and different objectives (stochastic entropy-regularized Q vs deterministic continuation). Neither absolute accuracy nor bias nor actor-vs-critic blame can be inferred from the 12 signs. This diagnostic must be labeled exploratory. |
| Replay buffer 50k erases older SAC experience, so replay eviction is dominant cause | **Unproven** | FIFO capacity is 50,000 transitions, approximately 427.4 episodes. The final **428-episode approximate** SAC window contains **393 distinct scenario IDs**, and all four seasons. Recent-vs-full seasonal total-variation distance is ~0.0426 (MBPO ~0.0104). This does not rule out fine-grained state-distribution drift or catastrophic forgetting, but it is not a dramatic seasonal collapse. Replay size is an ablation candidate, not a fix. |
| Match training days to exactly MBPO's 451 unique days | **Reject as blanket prescription** | Comparing different *number of unique days* at different real-transition budgets confounds dataset exposure with algorithm. Instead use a prespecified common stream of scenario IDs and initial seeds through 500 episodes and compare snapshots at equal real steps; optionally test dataset-breadth with a distinct, controlled ablation. Never choose 451 cases after inspecting test performance. |
| POA ideal advantage includes no slew-rate constraint and is gross-only | **Accept** | It is an unattainable instantaneous theoretical upper bound, not another feasible tracking algorithm. Our added check verifies recorded per-step dominance and data consistency, **not actuator feasibility**. Consider a separate feasible slew-limited reference if paper needs a realizable upper comparator. |
| Test nonzero movement cost and weighted summary/oracle integrity | **Accept** | Added negative/positive tests for net = gross – movement cost, design-weighted means/differences, separate oracle gross and recorded-weather parity. |

### Critique B: scientific interpretation — accepted/qualified/rejected

| Critique | Decision | Specific reasoning and action |
|---|---|---|
| Every advantage should name checkpoint, seed and real-transition budget | **Accept** | Use “MBPO checkpoint trained for 58,500 real steps versus SAC checkpoint trained for 234,000 real steps, both evaluated on the same 336 stratified days.” One checkpoint per algorithm is not an algorithm-wide claim. |
| Same 6D observation disproves representation/partial-observability concerns | **Reject strong inference; accept priority guidance** | MBPO performance shows the 6D representation can support an effective policy in one run. It does **not** prove the observation makes the MDP Markov, nor establish that SAC can learn the same policy from identical state information and no learned model. Keep the representation fixed for primary comparisons; move enriched-state ablation after direct training/value diagnostics. |
| In-sample model “2.8% better than mean predictor” | **Reject baseline identification** | The model skill is approximately 2.78% **RMSE improvement versus last-value persistence**, not versus a mean predictor. Its in-sample nature and rotating/persistent ensemble holdouts are real limitations. Small improvement does not prove the synthetic model is irrelevant to a 15% policy improvement. |
| Synthetic absolute TD error 2.167 versus real 1.202 means synthetic rewards/targets are 80% too large | **Reject** | These are **mean absolute Bellman residuals**, not mean rewards or mean Q targets. Different state distributions, model bias and transition uncertainty can increase absolute residuals without any target scale mismatch or upward Q bias. Both real and synthetic transitions flow into the **same SACAgent** with `reward_scale=100`; the fake environment computes gross energy and movement cost in the same project units. **Do not divide synthetic rewards by 1.8 or normalize them to match TD residuals**: that changes the optimization objective and would itself create an artificial scale inconsistency. Instead check signed TD errors, true vs predicted kWh, next-state predictions and calibration on untouched dates. |
| Plot SD across selected dates must not be presented as independent training-seed SD | **Accept** | One trained seed per method. The original summary labels day-to-day descriptive SD. Add this distinction to every paper figure/table caption. |
| 50k of 234k transitions retained ⇒ SAC is provably overfitting | **Reject causation** | True arithmetic; not evidence of nonstationarity or critic overfit. Window contains 393 recent unique days across seasons. Test only after verifying correct checkpoint/history association. |
| `reward_scale=100` and alpha ~0.015 means exploration collapsed | **Unproven** | Alpha > floor 0.001 in both checkpoints; entropy target is in **nats per action** and should **not** be rescaled with reward magnitude. Matching proportional reward and alpha scaling largely preserves relative weighting, so the proposed `scale=10, alpha_init=.1` experiment does not isolate “more entropy”. To test entropy, vary alpha/target entropy **relative to reward** with one independent change and quantify resulting entropy/action coverage. |
| `SAC@58.5k worse than SAC@234k` implies degradation with more training | **Reject: logical direction reversed** | If early SAC is worse than later SAC, SAC **improved** with more steps, not degraded. Degradation requires `SAC@234k < SAC@58.5k` on a prespecified identical evaluation (with stochastic variation/seed controls). One run cannot prove convergence. |
| Increase replay to 200k, batch to 512, or simply repeat days now | **Reject immediate fixes; consider separate controlled ablations** | No observed correctness defect demands those values. Changes may improve or worsen learning; testing one factor at a time is defensible only after provenance and comparable evaluation are addressed. |
| Arbitrary synthetic reward rescaling based on TD-absolute-error ratio | **Strongly reject** | Invalid inference and risks corrupting reward meaning and SAC target consistency. No change made. |
| Trial matched real budgets and 3–5 training seeds | **Accept with design qualification** | Use a precommitted shared schedule over scenarios, common evaluation development/test partitions, and independent training seeds. Match real steps and ideally the update-to-real-step ratio. Report model compute cost separately. Do not use the 336 viewed test days for iterative tuning and then claim they remain untouched. |
| “The PVWatts model is not the cause; legacy counterfactual proves this” | **Only previously reported** | The legacy-vs-PVWatts counterfactual was discussed in prior reports; Stage 5 artifacts alone do not independently recreate its physics. Do not attribute all differences to the backend without the command, source data and per-day output of that separate test. |
| Evaluation initial pose 30/180 vs randomized training pose is necessarily out of distribution | **Reject necessity; test** | A fixed pose may lie within a training distribution. Check empirical initial-pose frequency and performance sensitivity before calling it OOD. |

## New Stage 5.1 full-data verification (offline)

The upgraded audit was executed on the **actual** supplied Stage 4 diagnostic and complete Stage 5 ZIP extracted data (not only toy data), including `evaluation/episodes.csv`, `evaluation/trajectories_detailed.csv`, `evaluation/scenario_selection.json`, and both training histories.

| Full-data check | Result |
|---|---|
| Strict 5-minute UTC timestamps and day identity | **PASS** for 1,008 feasible trajectories, 117,936 steps |
| Recorded weather / solar parity | **PASS**, 39,312 paired step groups across SAC, MBPO, fixed |
| Gross-only POA reference consistency, independent action contract | **PASS** for 336 oracle trajectories, 39,312 steps; does not prove independent PVWatts optimum |
| Independent recomputation of design-weighted sample means | **PASS**, 336 bins, weight sum 1,167 |
| MBPO weighted mean | 0.8575851613 kWh |
| SAC weighted mean | 0.7487482291 kWh |
| Fixed weighted mean | 0.7454190285 kWh |
| POA oracle weighted **gross** mean | 0.9124866745 kWh |
| Weighted MBPO minus fixed / SAC minus fixed | +0.1121661328 / +0.0033292005 kWh |
| Stage 4 checkpoint-action replay *reported* coverage | **336×116 = 38,976** for each actor; first action was not checked in that replay test |
| Train/eval scenario-ID overlaps | **0 for both histories**, as exact recorded scenario IDs; cannot establish meteorological or training-seed independence |
| Stage 4 to Stage 5 SAC alpha provenance | **FAIL**: 0.01558018196 in earlier checkpoint snapshot vs 0.02241805196 in Stage 5 SAC CSV |
| Stage 4 to Stage 5 SAC last-100-episode metrics | **FAIL**, return / critic loss / Q mean also diverge |
| MBPO history provenance against Stage 4 | **Matches reported snapshot** |
| Final approximately 428-episode SAC replay window | **393 distinct dates**, seasonal total-variation ~0.04257 vs entire 2,000-episode SAC history |
| Final approximately 428-episode MBPO replay window | **387 distinct dates**, seasonal total-variation ~0.01037 vs entire 500-episode MBPO history |
| New plus carried Stage 4 unit tests | **34 passed** (20 Stage 5.1 and 14 Stage 4, in local environment) |

A 50,000-transition FIFO buffer covers 427 whole 117-step episodes plus 41 transitions from the next older episode. The 428-episode CSV window is a conservative approximation to scenario coverage, not a reconstruction of the individual replay indices. Distinct scenario IDs and seasonal balance do **not** guarantee equal state-action coverage or critic generalization.

**Critical note on evidence level:** The new offline tests catch altered *saved reports*, not an independently recomputed SAC optimizer step or a live checkpoint-action match. Those remain obtainable by rerunning the bundled Stage 4 diagnostic on the user's own trusted `.pt` and simulator. Even if those checks pass again, they verify wiring and recorded action provenance—not policy optimality.

## Decisions / recommended controlled order

**Now (no model code changes):**

1. Identify the SAC history belonging to checkpoint SHA `749e28902aede9d64154bfb27431b3a47522f33f43e6e85015f51db1c28f1688`; run the trusted checkpoint-vs-CSV checker and preserve exact CLI/config/commit. If history cannot be recovered, do **not** interpret Stage 5 SAC optimizer plots as the evaluated policy's training trajectory.
2. Preserve the original 336-day test output as already viewed; designate a separate development sample for any tuning and a fresh truly untouched final test before paper inference. Include design-weighted summaries and conditional sampling-frame definition.
3. Re-execute the existing Stage 4 checkpoint shadow update and action replay, if needed, against the true local files; label partial first-action coverage.
4. Construct a **new** well-powered critic-calibration experiment with fixed pre-action observations, feasible actions and matched stochastic/entropy-regularized Q returns (same reward scale), repeated across many prespecified independent days. Measure signed and absolute value bias and actor choice independently; avoid selecting 100 correlated points from four days.
5. Evaluate SAC snapshots at 500 and 2,000 episodes, matched to MBPO 500 episodes on a prespecified common scenario stream for the first 500 episodes. Repeat across independent training seeds and report compute time. Correct the provenance before modifying config.
6. Only then run one-variable-at-a-time replay-capacity, alpha-relative-to-reward, or actor/critic-loss ablations; separately validate the MBPO ensemble on an invariant never-trained holdout and consider real-only MBPO control. Keep the primary 6D observation and physical simulator fixed.

**Do not currently modify** `SACAgent`, the synthetic reward, replay configuration, policy state vector, PVWatts physics, optimizer or training logs. Fixing the test suite and provenance protocol is justified; fixing network weights based on these reports is not.

## Paper-safe wording

> In a paired PVWatts/SAPM simulation involving 336 calendar-stratified evaluation days, the particular MBPO checkpoint trained on 58,500 real transitions yielded a design-weighted mean of 0.8576 kWh, compared with 0.7487 kWh for a model-free SAC checkpoint trained on 234,000 real transitions and 0.7454 kWh for a fixed mounting. SAC underperformed fixed mounting on 200 of 336 observed days, with particularly steep learned spring/summer tilt. The reported variations summarize days, not independent training runs. These are descriptive checkpoint-specific findings; algorithm-level reproducibility, MBPO model generalization, and the cause of SAC's poor learned policy remain to be established.

**Do not use the pasted reviewers' suggested CLI flags** `--total-steps`, `--config`, `--replay-size` or `--reward-scale` against the existing current training workflows; those flags are absent. The bundled command printer uses `--epochs` and always passes `--power-model pvwatts_sapm`. If future ablations require new flags, implement them explicitly in a separate branch with new config fingerprints rather than silently editing the experiment that produced the paper's baseline.
