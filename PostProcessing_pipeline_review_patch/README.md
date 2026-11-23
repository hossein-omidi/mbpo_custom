# PostProcessing — Simulator/training audit and truthful mean ± SD representation (incremental patch)

**Reviewed source:** `https://github.com/hossein-omidi/mbpo_custom/tree/PostProcessing`, current read-only branch commit `b0c01ec13e0568d0bf37476af5480bda8e3a40fa`. The previous **Monte Carlo critical-review V2** must already be installed; this patch builds on its exact source. It does not recreate or rewrite SAC, MBPO, pvlib, the NSRDB daily catalogue, or the uncertainty process.

## Priority findings

1. **Direct verifier import is broken.** Running `python PostProcessing_MC_patch/tools/verify_mc_environment.py ...` makes CPython search `.../tools` first; the repository root is absent and `from mbpo.env...` fails. This patch verifies a complete repo layout, then adds its root to Python's module path before importing project modules. Module-form execution `python -m PostProcessing_MC_patch.tools.verify_mc_environment` also works. It does **not** ignore missing dependencies or physical-simulator errors.
2. **Official sample-complexity learning-curve shading represents two-way 95% bootstrap percentile bands, not SD.** The existing bootstrap analysis is preserved. Additional **genuine mean ± 1 SD** plots are created from its exact [training-seed, validation-day, real-transition-checkpoint] matrices. The SD across independently trained seeds is separate from SD across historical days. Paired MBPO–SAC SD is computed **after subtracting matched seed/day outcomes**, not by combining the two individual SDs.
3. **The current single-run training-return chart is a diagnostic rolling mean, not a multi-seed mean ± SD estimate.** Its title is clarified; it must not be used as a published cross-seed learning curve.
4. **Existing nested MC power files calculate per-day mean and sample SD across independent weather realizations, which is correct.** An additional pooled 195-step mean ± descriptive SD curve combines held-out historical days and MC draws with an **exact empirical variance decomposition**, without treating neighboring time steps as independent samples.
5. **Frozen scenario split is already implemented.** The patch retains the current deterministic, month-stratified `train_ids`, `validation_ids` and `test_ids`. It adds a fail-fast check that *training* episode CSVs used in sample-complexity analysis contain only `train_ids`. This cannot retroactively certify old contaminated datasets; it prevents silently analyzing them.
6. **MBPO dynamics already separates weather from actions:** its weather ensemble predicts DNI/DHI increments without action inputs and its electrical-response ensemble conditions power changes on actions and predicted weather. SAC and MBPO real interactions use the same Stage A physical environment. No change to replay mixing or network optimization is justified from this audit.

The 2018–2024 historical catalogue, the 195 real steps per episode, the six-dimensional state and two-dimensional action, the positive movement-cost model, and the existing experiment protocol are not modified. The corrected report uses the same real-environment interaction checkpoints, never episode index or model-generated rollouts, to compare learning efficiency.

## Modified and added files

| File | Change |
|---|---|
| `PostProcessing_MC_patch/tools/verify_mc_environment.py` | Fix direct-file repository import path with explicit checkout validation. Preserve `python -m` support. |
| `MBPO_workspace/src/sample_complexity.py` | Validate training scenario membership; export descriptive SD analysis; explicitly distinguish existing bootstrap resampling bands from SD. |
| `MBPO_workspace/workflows/09_monte_carlo_evaluate.py` | Export across-day × across-draw power mean/SD and an additional shaded power curve. |
| `MBPO_workspace/src/plots.py` | Label raw single-run training plot as a rolling diagnostic only. |
| `MBPO_workspace/src/curve_statistics.py` | Pure [seed,day,checkpoint] statistics, exact pooled day/draw variance, CSV and figure exporters. |

The installer verifies the Git blob hashes of each modified source and also checks the **unchanged** experiment protocol, SAC trainer, MBPO trainer and action-independent weather / action-dependent power dynamics code. It refuses unknown code versions. No backups, Git commands or GitHub changes are made.

## Local installation

Extract the ZIP into your repository root, keeping the `PostProcessing_pipeline_review_patch/` directory. From the repository root:

```bash
python PostProcessing_pipeline_review_patch/apply_local.py --root . --dry-run
python PostProcessing_pipeline_review_patch/apply_local.py --root . --diff-file pipeline_review.diff
# inspect the diff, then choose ONE of:
python PostProcessing_pipeline_review_patch/apply_local.py --root . --apply
# OR manually: git apply --check pipeline_review.diff && git apply pipeline_review.diff
```

No backup directories are created or deleted. To remove the patch from a Git-controlled checkout you can inspect/revert its changes using your normal Git workflow. The installer does not invoke Git on your behalf.

## Verify the direct-file error is fixed

```bash
python PostProcessing_MC_patch/tools/verify_mc_environment.py \
  --uncertainty-spec PostProcessing_MC_patch/examples/hypothetical_uncertainty_5pct.json \
  --out outputs/monte_carlo_environment_audit.json
```

This invokes the actual 195-step physical simulator. It still **requires** installed pvlib/Gym dependencies and the annual NSRDB CSV files. The patch does not circumvent them. As an alternative in an unpatched checkout, you can already use `python -m PostProcessing_MC_patch.tools.verify_mc_environment ...` from the root.

## Paper-ready mean and shaded SD plots

After running multiple training seeds with the **same frozen validation scenario partition** and matching real-transition checkpoints:

```bash
python -m MBPO_workspace.workflows.05_sample_complexity \
  --sac-runs SAC_workspace/runs/sac_mc_seed0 SAC_workspace/runs/sac_mc_seed1 \
  --mbpo-runs MBPO_workspace/runs/mbpo_mc_seed0 MBPO_workspace/runs/mbpo_mc_seed1 \
  --outdir MBPO_workspace/evaluation/sample_complexity_mc
```

The existing workflow still requires **at least two runs per algorithm**; five or more independent training seeds are preferable for robust empirical comparisons. Newly generated files include:

- `learning_curve_mean_pm_seed_sd.png`: solid mean, shaded ±1 SD **across training-seed means** (each seed averaged over the same fixed validation days).
- `learning_curve_mean_pm_day_sd.png`: solid mean, shaded ±1 SD **across historical-day means** (each day averaged over training seeds).
- `paired_difference_mean_pm_seed_sd.png`: paired MBPO−SAC mean ±1 SD **across training-seed differences**.
- `learning_curve_descriptive_sd.csv` and `paired_difference_descriptive_sd.csv`: fully labeled source statistics and sample counts.

The original `learning_curve.png` and `paired_difference.png` preserve their **two-way bootstrap resampling bands** as separate figures. Because the validation dates were deterministically month-stratified, these bootstrap intervals should not be interpreted as guaranteed design-based 95% confidence coverage for future years or all NSRDB dates. They are best reported as bootstrap resampling sensitivity intervals. The SD plots are **descriptive**, not confidence intervals.

For the nested Monte Carlo test of trained controllers, use your existing command for `MBPO_workspace.workflows.09_monte_carlo_evaluate`. It additionally exports:

- `mc_pooled_power_mean_sd_by_utc_step.csv`: separate mean, descriptive pooled SD, between-day SD and within-weather-draw SD, all at the same 195 UTC steps.
- `mc_power_mean_pm_sd_over_heldout_days.png`: solid mean, shaded ±1 descriptive SD of the **historical-day × weather-draw mixture**.

Your existing `daily_power_mean_std_by_utc_step.csv` and `mc_first_day_power_mean_sd.png` are retained. Their SD comes from the independently simulated uncertain weather realizations **conditional on each selected historical day**, not from training seeds. No chronological/weather data are replaced by average inputs before pvlib.

## Formal definitions

For returns `R(seed,day,checkpoint)`:

- `mean = mean_seed mean_day R`;
- `sd_across_training_seed_means = sample_sd_seed(mean_day R)`;
- `sd_across_historical_day_means = sample_sd_day(mean_seed R)`;
- paired return difference: `D = R_MBPO − R_SAC` at identical seed, day, checkpoint, with `sd_seed(mean_day D)`.

For K historical days and M weather draws per day, time-dependent day mean `μ_e,t` and **sample** SD `s_e,t`:

`μ_t = mean_e μ_e,t`

`Var_empirical(t) = mean_e[(μ_e,t − μ_t)^2] + (M−1)/M · mean_e[s_e,t^2]`

`SD_pooled(t) = sqrt(Var_empirical(t))`

This is the exact empirical population variance of the K×M *simulated outputs* when all historical days and draws are weighted equally. It is **not** the Monte Carlo standard error of the mean, a parametric Gaussian forecast interval, a training-seed SD, or calibrated NSRDB uncertainty. The hypothetical 5% weather specification remains an unvalidated sensitivity assumption.

## Verification status

- Targeted additional offline tests: **10 passed**.
- Previous Monte Carlo V2 numerical/orchestration tests plus additional tests: **30 passed** in the same environment.
- Source-file replacement-anchor uniqueness and target Git blob SHA-1s: checked against read-only GitHub `PostProcessing`.
- Local mock installer: patch planning, syntax checking, file application, and idempotence verified.
- Test of direct-file verifier entry point: `--help` works without importing pvlib; invalid checkout produces an actionable error rather than the original ambiguous `ModuleNotFoundError`.
- **NOT executed here:** complete locally installed pvlib/Gym/NSRDB physical simulator, trained checkpoint evaluation, and actual multi-seed sample complexity curves. These must pass in your workstation before publication claims.

**No retraining is required solely because of this patch**: it modifies reporting and a verifier entry point, not the simulator dynamics, the episode split, the reward, or either actor/critic implementation. But any previous run with an actually different simulator or uncertainty model must not be pooled with these runs merely because its saved files have the same names.
