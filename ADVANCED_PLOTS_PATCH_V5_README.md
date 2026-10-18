# Advanced scientific visualization add-on (v5)

**Minimal additive patch** for the current SAC v2 + MBPO recovery v2 + stratified v3 + statistical v4/v4.1 installation. Apply from repository root; it DOES NOT touch neural-network architecture, PV physics, SAC/MBPO optimizer, datasets, or existing figures. It **replaces one evaluation workflow** to auto-create rich metadata and the advanced suite after your next successful run.

## Files

- `MBPO_workspace/src/advanced_suite.py` — standalone exporter, paired validation, 2D/3D scientific plots, observed-date mean/spread, PDF/PNG, machine-readable curve data.
- `MBPO_workspace/workflows/03_evaluate.py` — minimal entry-point integration; new `--skip-advanced-suite` option.
- `MBPO_workspace/workflows/06_plot_suite.py` — offline report regeneration.
- `plot_modern_results.sh` — Bash runner, no Conda activation, Ray or TensorFlow.
- `MBPO_workspace/tests/test_advanced_suite_v5.py` — numeric and end-to-end test coverage.
- `MBPO_workspace/docs/ADVANCED_PLOT_SUITE_V5.md` — detailed statistical meanings and assumptions.

## Installation

```bash
# Back up custom edits to MBPO_workspace/workflows/03_evaluate.py, if any.
unzip -l /path/to/solar_pv_advanced_scientific_plots_patch_v5.zip
unzip -o /path/to/solar_pv_advanced_scientific_plots_patch_v5.zip -d .
chmod +x plot_modern_results.sh
python -m unittest discover -s MBPO_workspace/tests -p 'test_*.py' -v
python -m unittest discover -s SAC_workspace/tests -p 'test_*.py' -v
```

**Immediately regenerate figures from a completed MBPO comparison run, WITHOUT retraining/evaluating again:**

```bash
./plot_modern_results.sh MBPO_workspace/evaluation/verified \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_seed0/episodes.csv \
  --sac-training-csv SAC_workspace/runs/sac_seed1/episodes.csv
```

Or `python -m MBPO_workspace.workflows.06_plot_suite --eval-dir MBPO_workspace/evaluation/verified`.

Artifacts are written under `MBPO_workspace/evaluation/verified/advanced_suite/`: PNG figures, combined PDF, `curve_band_data.csv` with plotted numbers and `figure_manifest.json` with source provenance and missing-data warnings.

**Compatibility:** Prefer `trajectories_detailed.csv` (new), then `advanced/trajectories_rich.csv` (old six-method export; filters out classical trackers), then plain `trajectories.csv`. A plain CSV has no action/solar/irradiance columns: these plots are skipped and listed in the manifest; they cannot be reconstructed from compressed data. A new `03_evaluate` automatically writes the richer trajectory export before creating plots, and `--skip-advanced-suite` can opt out of expensive figure creation.

**Statistics:** Solid lines are design-weighted mean over *distinct sampled dates* at the same UTC hour. Bands are **descriptive weighted observed-date root-mean-square deviations**; they are not SEM, confidence intervals, unbiased finite-population SD, or uncertainty about unseen future weather. Exact formulas and excluded inferences appear in `docs/ADVANCED_PLOT_SUITE_V5.md`. Gross POA is excluded from net plots. The trailing training curve average is smoothing, not between-seed uncertainty.

**Verified:** clean extraction over the earlier archived source packages, 34 earlier MBPO tests + one additional older-rich compatibility test (35 MBPO tests), 20 SAC tests with two environment-dependent skips; all offline plotting smoke tests use synthetic data, **not** verified real checkpoint energy results.
