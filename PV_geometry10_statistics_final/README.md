# Geometry10 finalized scientific representation update (v3)

**Target:** `hossein-omidi/mbpo_custom`, `PostProcessing`, commit `e34371f7527eaf210186f20f2c917996a436cfd5` (2026-11-25). This is the consolidated replacement for `PV_geometry10_statistics_update.zip`, on top of the already-integrated Geometry10 audit release. It accepts the exact original target, the exact prior statistical update, or the final version. It refuses unknown local edits. Do not apply both installers sequentially unless the prior one is already installed.

## Apply in the original repository

Unzip this archive into the original repository root. From that root:

```bash
python PV_geometry10_statistics_final/install.py --root . --dry-run
python PV_geometry10_statistics_final/install.py --root . --apply
git diff --check
python -m pytest -q geometry10_tests/test_scientific_representation.py
python -m pytest -q geometry10_tests
```

The dry run checks the base commit ancestry, exact original/final file contents, dependency bytes, unique source anchors, Python compilation of the generated modules, and `git apply --check`. The installer generates one atomic Git patch in memory and does not overwrite unexpected local edits. Applying again is a no-op. No sibling repository, checkpoint, output directory or data backup is created. No training source or hyperparameter is changed. Inspect `git diff` before committing.

## Final review corrections (v3)

1. Reject impossible conditional power sample SD before checking derived pooled statistics: for M nonnegative draws, `sample_sd <= sqrt(M)*mean` (small absolute/relative floating-point tolerance). This follows from `sum(x*x) <= sum(x)**2`; it is a necessary bound, not sufficient to establish the original draws or their quantiles. The earlier validator accepted the injected 10,000 W SD for mean 11 W, M=3; this package rejects it even when the derived pooled and first-day tables have been recalculated to agree.
2. Require matching frozen training runtime identities (including training source and dependency fingerprints), exact algorithm-specific configuration digests within each algorithm, and common physical/shared SAC/protocol/budget metadata across methods. SAC and MBPO are not required to have the same algorithm-specific configuration. Apply this to standalone learning analysis, advanced learning, and direct MC evaluation. Existing runs with legitimately different frozen dependencies cannot be combined under one controlled-comparison claim.
3. Recalculate the existing diagnostic skill and RMSE/mean-predictive-SD ratios, counterfactual MAE, bias and mean disagreement, and equivalent-pose pair count and power discrepancy from persisted rows using the producer's fixed conventions. Do not interpret the RMSE/mean-SD ratio as a calibration proof. The epsilon-floor convention in the producer is unchanged; the zero-persistence-error training-gate edge case is not modified. Raw hierarchical draws are not newly retained.
4. Existing full-scope rich replay validates actual raw path means/SD and energy identities. For final claims about full held-out time-resolved or cumulative dispersion, run both rich collection and rich rendering with `--scope full` (same stage, seeds and trajectory tag). Representative plots are illustrative selected-day results only. Standard MC conditional moments remain sufficient for exact marginal pooling, not for independent raw-path verification.

No training source, training fingerprint, baseline, learning target, reward, or hyperparameter has changed.

## Essential changes

- Common full MC table validation before sealing and before any managed publication pathway: exact historical-day/draw/method/time grids, UTC, weather hash pairing, net energy, mean power integral, first-day moments, pooled mean/SD and daily/paired nested means, variances, quantiles and conditional Monte Carlo standard errors. No pooling over timestamps as independent observations.
- Final SAC and MBPO periodic-policy energy parity with MC realization 0 (gross, movement, net), including two-day smoke evaluation. Fixed and astronomical reference parity retained. Model-validation physical on-policy truth at realization 0 must also agree with final periodic gross/net returns; model predictions are not substituted for physical truth. Other model realizations are separate conditional diagnostic evaluations.
- Model validation exports predictive sample SD, empirical quantiles and observed/predicted reward. Recompute internal/external RMSE, MAE, bias, mean predictive SD, coverage and counterfactual errors from saved rows. Cross-check training synthetic exposure against the actual committed episode history. Unfitted ensemble is explicitly unavailable, with verified zero synthetic exposure; retain the seed in policy results and omit unavailable model plots/metrics (no manufactured zero errors or SDs).
- Rich trajectory replay loads the exact checkpoint hashes, rechecks stored power SD, per-step gross-energy units and oracle power/energy identity. Paired differences and cumulative trajectories are calculated before aggregation; existing correct estimators are not changed.
- Low-level sample-complexity/physical/model CLI requires sealed completed training runs. Test evaluation and legacy test report require explicit `--confirm-test`; direct MC default is validation. Recovery ledger is exported as committed interactions plus a known executed-interaction lower bound, not silently mixed into the sample-complexity endpoint. Unlogged work and lost timing remain unknown.
- Hash actual statistical dependencies in evaluation/report provenance, and numerical/figure output files in generated report manifests. Legacy cached reports with altered or missing outputs are rejected.
- Legacy bootstrap figures are labeled central resampling sensitivity bands, not calibrated population confidence intervals; nonattainment is shown; compute-cost plots show matched seeds and mean ± seed SD instead of an isolated bar. Boxplot outliers remain visible. Explicit SD scope in physical, model, training and dataset figures. Calendar categories are point estimates rather than falsely connected across missing months; heatmaps use nearest-cell rendering; diffuse-regime x positions are fixed across seeds even when a category is absent.

## Statistical conventions (unchanged)

| Figure/quantity | Correct statistic |
|---|---|
| Learning curves | Average validation days within seed, then mean and sample SD across independent seed means (`ddof=1`). No smoothing of the primary endpoint. |
| Pooled physical power | Empirical population SD across equal seed/day/draw cells. Reconstruct within-day population contribution from sample SD via `(M-1)/M`. |
| Paired comparison | Subtract MBPO–SAC in each matched cell before aggregation. |
| Cumulative energy | Integrate each original path before computing moments. |
| Single-day MC profile | Mean and sample SD across draws for explicitly identified seed/day. |
| Model RMSE | Mean of per-fitted-seed RMSEs; sample SD across fitted seeds only, not pooled transition RMSE. Fitted/selected seed coverage explicit. |
| Training diagnostics | 25-episode trailing mean within seed, then seed SD. Not controlled policy evaluation. |
| Calendar heatmap / occupancy / target attainment | Actual cell means/frequencies/counts. No SD band is necessary. Coverage and underlying numerical exports remain available. |
| Circular azimuth | Circular mean/resultant length, or actual points; never linear angular SD. |

A mean ± SD band is descriptive, not a confidence interval, range, calibrated field uncertainty or evidence of equivalence. A lower band can be negative for a nonnegative physical quantity; this does not imply negative observed power. Baselines have zero between-training-seed dispersion but may have nonzero weather/day variability. For a single seed, seed SD is undefined and is not drawn. Raw per-draw time profiles are not retained by standard MC; the exact conditional moments are sufficient for marginal pooled SD, but raw rich replay is required to independently audit per-draw temporal SD, paired instantaneous differences and cumulative covariance. Use full rich scope for claims about the full held-out panel; representative scope is explicitly a subset.

## Running the postprocessing

Use the actual frozen experiment name. Example:

```bash
EXP=geom10_final_pilot_v2
python -m PV_experiments.evaluate --name "$EXP" --mode smoke --draws 3
python -m PV_experiments.evaluate --name "$EXP" --mode validation --draws 10
python -m PV_experiments.evaluate --name "$EXP" --mode model
python -m PV_experiments.represent --name "$EXP" --kind all --stage validation --tag statistics_v3
python -m PV_experiments.advanced_report --name "$EXP" --kind all --stage validation --tag statistics_v3
python -m PV_experiments.represent --name "$EXP" --kind collect --stage validation --scope representative --trajectory-tag statistics_v3
python -m PV_experiments.represent --name "$EXP" --kind trajectories --stage validation --scope representative --trajectory-tag statistics_v3 --tag statistics_v3
```

For final full test evaluation, freeze the protocol first and use `--confirm-test` on all test evaluation and reporting commands. Use the predeclared final draw count, not the illustrative 10-draw pilot. `--kind all` representation needs at least two training seeds, a completed physical validation and model diagnostic status for each seed. Rich replay is optional, not a required extra evaluation for basic plots.

### Existing results

Training fingerprint sources have not changed, so successfully sealed training runs from the target commit need no retraining. The evaluation and report fingerprints intentionally change: old sealed evaluations and reports are **not silently blessed as newly verified results**. If those directories already exist, the manager refuses to overwrite them. Rename the old `evaluations` directory (and old `reports` directory if using the legacy `report` CLI) to an unused name after inspecting them, then regenerate the affected outputs from the same checkpoints. New `--tag`/`--trajectory-tag` values avoid collisions in representation. This patch does not automatically move, delete or regenerate results. The fresh MC evaluation is needed for a new evaluation seal; model validation also needs new exported SD/physical truth columns. This is a deliberate conservative choice, not a requirement to retrain policies.

## Verification boundary

The prior package documented its source-anchor checks; the three new call-site anchors were separately checked against the actual pinned GitHub files via the repository connector. The 30 local regression tests passed using constructed fixtures, inspected aggregation/statistics excerpts and the earlier audit helper/release payload. The previous impossible-SD acceptance was reproduced before the fix. Installer fixture tests include exact-prior upgrade and idempotence. The local execution environment could not resolve GitHub to clone the source. Local numerical tests therefore used the new validator code plus transcribed inspected aggregation/statistics function excerpts and the previously supplied audit helper, with independent constructed fixtures; the installer was tested on isolated Git fixtures. These checks are not a claim that the full patched project or its real NSRDB dataset was run. Execute the dry run and both pytest commands above on the real repository, then smoke/validation/model/representation in order. Do not proceed to final test if any check fails.

The training gate's degenerate zero-persistence-error edge case is deliberately not altered here: it would change training semantics and invalidate existing checkpoint provenance. The audit found no evidence that it occurred in the intended diverse whole-day holdout. It can be separately corrected before a new training campaign if actually observed. Other fundamental SAC/MBPO equations and the sample-efficiency target remain unchanged.
