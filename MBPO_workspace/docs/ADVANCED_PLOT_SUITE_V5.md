# Advanced reproducible SAC / MBPO / fixed / POA plotting (v5)

This additive reporting suite adapts **visual concepts**, not the legacy TensorFlow/Ray dependencies, from `scripts/evaluate_agent.py`, `scripts/plot_training_progress.py`, `scripts/evaluate_fullyear_mc.py`, `scripts/diagnose_tracking.py`, `scripts/plot_paper_eval.py`, and `scripts/plot_rollout_trajectory.py`.

## Why the old figures were missing

`MBPO_workspace/workflows/03_evaluate.py` in v4 renders the small verified figure set but did not invoke `paper_reporting.py`. Also, the older `paper_reporting.expand_paths()` required *all six* controllers, whereas the configured v4 workflow deliberately evaluates just MBPO, SAC, fixed, and instantaneous POA. The v5 suite supports those four directly and never evaluates or displays classical dual-axis trackers.

## Run

```bash
# Recommended: from repo root, after an existing evaluation run
./plot_modern_results.sh MBPO_workspace/evaluation/verified \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_seed0/episodes.csv \
  --sac-training-csv SAC_workspace/runs/sac_seed1/episodes.csv

# Or
python -m MBPO_workspace.workflows.06_plot_suite \
  --eval-dir MBPO_workspace/evaluation/verified
```

New invocations of `MBPO_workspace.workflows.03_evaluate` will also call this suite automatically after creating `trajectories_detailed.csv` and the usual episode/stratified files. Set `--skip-advanced-suite` to save plotting time. Old evaluation folders work **without rerunning** PVlib or training. Existing `advanced/trajectories_rich.csv` from the legacy-compatible reporting adapter is reused when present. If only the plain `trajectories.csv` exists, action/solar plots are explicitly skipped because those values were never recorded.

## Generated reports

`advanced_suite/advanced_suite.pdf` (multi-page figure booklet), numbered PNG files, `figure_manifest.json` (provenance, available and skipped plots), and `curve_band_data.csv` (all plotted mean/spread values, hour, units, and number of independent sampled days).

The legacy features retained or adapted include UTC-aligned power and cumulative energy; 2D/3D panel paths; learned policy commanded actions; POA-minus-learned paired hourly power deficits; orientation occupancy; month-weighted energy and movement; observed-day ECDF; movement-energy relationships; and training episode return curves. Classical sun-pointing/slew trackers are never considered by the new suite.

## Meaning of *solid mean + shaded variability*

For each scenario/date \(d\) and post-step UTC time \(t\), first average multiple **action seeds** of that same date (if any): \(\bar y_{dt} = R^{-1}\sum_r y_{drt}\). The design weight \(w_d\) is `stratum.population_days` from `scenario_selection.json` (when present). The solid curve is

\[ \hat\mu_t = \frac{\sum_d w_d\bar y_{dt}}{\sum_d w_d}. \]

The translucent envelope is

\[ \hat\mu_t \pm \sqrt{\frac{\sum_d w_d(\bar y_{dt}-\hat\mu_t)^2}{\sum_d w_d}}. \]

This is a **descriptive weighted RMS dispersion over observed dates**, not an unbiased finite-population standard deviation, estimator standard error, confidence interval, or time-series forecast interval. It describes the weather-day variation visible in the chosen sample; its interpretation changes with selected dates. If only one distinct date is available, no shading is shown. Adjacent five-minute steps are correlated and never counted as independent observations. For paired contrasts, per-date differences are taken **before** calculating the mean and dispersion.

Monthly series are design-weighted point estimates over that month's eligible manifest days, conditional on the saved sampling design. **No 95% confidence bars** are added from just one date per sampling bin because within-bin sample variance is not identified. A repeated independent date-selection design or >=2 independently sampled dates per bin would be required to estimate it. No weather-regime inference, t-based CI from 117 step replicates, or interpolated missing raw actions is permitted.

The instantaneous POA oracle optimizes *gross* output and never enters a **net** energy plot. The POA-minus-MBPO *power* gap is gross-on-gross. Movement is the sum of absolute applied tilt and azimuth deltas; with reference movement_penalty=0, this is **not** electric actuator consumption. The 3D representative path is a single observed date, **not** a statistical band. Azimuth is circular; the suite avoids naive linear means or SD bands for 0°/360° wrap-around.

### Falsifiable checks

The loader rejects duplicate path keys, missing methods, incomplete 117-step paths, inconsistent 5-minute UTC timestamps, unmatched dates/seeds, inconsistent paired clocks, nonfinite energy/power measurements, and mismatched 5-minute power/energy accounting. When detailed fields are available, it verifies the feasible step reward identity and disallows an invented POA net reward. The suite does not discard incomplete dates to produce attractive curves: a whole missing metric is omitted with a reason in `figure_manifest.json`.

### Known limits

- Date stratification may reuse training dates. This is not a held-out generalization result without verified exclusion.
- With 7 correlated weather years and one selected date per bin, the suite intentionally avoids claiming design-based CI coverage.
- Only new evaluations can export genuinely measured actions and solar geometry; existing bare `trajectories.csv` cannot reconstruct those fields.
- Per-date Monte Carlo over stochastic policy action seeds measures conditional action randomness, not independently sampled weather variability or training-seed uncertainty.
- Trial-to-trial training uncertainty requires independently trained checkpoints and paired scenarios. The training moving-average curve from one experiment is **not** a confidence band.
