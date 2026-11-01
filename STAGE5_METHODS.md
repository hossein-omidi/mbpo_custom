# Stage 5 — hypothesis-to-evidence decision framework

| Hypothesis | Strongest diagnostic evidence | What would refute/limit it | Do NOT conclude merely from |
|---|---|---|---|
| Current PV power makes actor greedy | Sensitivity of action to power at genuine states; real action-grid continuation worse than actor; repeated across seasons | Modest actor sensitivity, no consistent near-noon grid regret | P as an input feature or high P–reward correlation (both are physically expected) |
| Power normalization poorly scaled | Observed Pnorm distribution, actor gradients and power-channel model errors | No consistent impact in held-out states | Pnorm below 0.1 alone |
| Finite-horizon `tau` distorts policy | Actor sensitivity to tau, noon error by remaining steps, verified terminal/Q masking | Error not dependent on tau after controlling weather | Gamma=1 or tau present in state |
| Seasons mixed in training replay | Month counts, post-eviction FIFO coverage, matched budget report | Balanced training and replay seasonal coverage | Episode budget alone |
| Azimuth sign/angle bug | Exact reproduced physical azimuth/tilt and cosAOI, angle wrapping checks | Closed-form trigonometry agrees with pvlib on all sampled replay steps | Raw difference between 359.9° and 0.1° |
| Azimuth input seam harms actor | Near-seam physical states plus action jumps across two almost identical azimuths | No large actor jump near actual visited orientations | Off-manifold seam stress test alone |
| Incorrect UTC/DST/time duration | Parsed aware ISO; UTC five-minute cadence; fixed UTC window and NY local offsets; real daylight coverage | All physical/clock checks pass | Civil noon not equal solar noon |
| MBPO learned model inaccurate | Real held-out power/DNI/DHI RMSE vs persistence, bias, 1σ/2σ coverage by season/noon | Positive reproducible model skill and calibrated uncertainty on unseen states | In-sample model NLL or one replay RMSE |
| Synthetic experience biases critic | Held-out MBPO TD errors, model bias, critic/physical action ranking versus SAC | No specific model bias and no meaningful sampled action regret | Larger fake TD than real TD across different distributions |
| Critic misranks actions | Same-state Q grid + actual immediate/full deterministic return grid, enough independent dates | Critic ranking of action alternatives is consistent | One sign disagreement (soft Q and deterministic return are different objectives) |
| Actor fails to maximize learned critic | Chosen actor action vs best **tested** Q-grid action on same observation | Critic ranks chosen above grid | Gradient magnitude on one minibatch |
| Fixed panel beats MBPO at noon due to geometry | AOI and direct POA component explain measured DC power difference | PV power gap persists after replay with same physical weather and orientations | Seeing fixed above MBPO on a plot |

## Diagnostic interpretation hierarchy

1. **Integrity failures**: bad timestamp cadence, mismatched data/physics hash, missing input fields, action replay failure, inconsistent POA or reward. These are actual correctness defects. Stop and repair the specific contract before interpreting performance.
2. **Measured local suboptimality**: a frozen policy's selected near-noon action yields lower physical continuation energy than a feasible alternative action under identical weather. This establishes a state-specific suboptimal decision, not its training cause.
3. **Model-quality evidence**: held-out dynamics skill versus persistence, seasonal bias and uncertainty coverage. Weak/no skill supports an MBPO reliability concern, not proof that synthetic replay caused the observed gap.
4. **Design differences**: seasonal/UTC window, scalar azimuth seam, P and tau sensitivities, limited observation history. These are hypotheses about representations, not necessarily bugs.
5. **Scientific conclusions**: claim robust superiority only after budget-matched, independent training seeds, and untouched final evaluation. A good daily mean does not imply each interval must beat fixed.

## Important detail

The exported `trajectories_detailed.csv` may not contain `surface_albedo` even though Stage A's environment does. The diagnostic therefore replays a precommitted selection of actual NSRDB days to obtain time-aligned recorded albedo, POA components, cell temperature and power. It does not fill in invented albedo values.
