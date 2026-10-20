# v5.3 — minimal evaluation workflow fix + read-only numerical audit

**Base:** v5.2 scientific representations. Copy the ZIP's repository-relative files into your repository. Back up locally customized versions of `03_evaluate.py` and `06_plot_suite.py` before replacing. Does not touch trained models or the core SAC/MBPO, pvlib, simulation or inference algorithms.

## Confirmed failure and corrections

1. `03_evaluate.py` ended with `KeyError: 'plots'` *after* successfully saving results. `advanced_suite.render_suite()` returns `figures` and `skipped`, not `plots`. The final summary uses those keys, including an empty compatible result when `--skip-advanced-suite` is used. All of the expensive evaluation before the erroneous final print was already complete.
2. Optional training-history CSVs supplied explicitly are now checked at argument parsing in both `03_evaluate.py` and `06_plot_suite.py`. A misspelled path fails immediately instead of silently omitting a requested training diagnostic.
3. New optional **read-only** `07_audit_scientific_results.py` independently recalculates completed evaluation sums and weighted means from saved CSV files. It does NOT alter the files.

## Audit your already completed 252-day evaluation — DO NOT rerun training or simulation

```bash
python -m MBPO_workspace.workflows.07_audit_scientific_results \
  --eval-dir MBPO_workspace/evaluation/scientific_v52_fresh \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_seed0/episodes.csv \
  --sac-training-csv SAC_workspace/runs/sac_seed0/episodes.csv
```

The command reports `PASS` only if the present artifacts agree within numerical tolerances. It checks:

- One-to-one completeness of each 117-step episode for all four methods, including paired initial orientation and UTC timestamps.
- 5-minute power-to-energy conversion, cumulative energy, and gross/net/movement identities.
- Exact sample-date coverage and population weights from `scenario_selection.json`.
- Independently computed calendar-bin-weighted method means and paired net-energy differences against `summary.json`.
- Per-scenario POA gross-energy dominance over feasible methods as a necessary check (NOT proof of global optimality).
- Weighted mean and observed-day RMS bands in `advanced_suite/curve_band_data.csv` for recorded basic power and energy curves.
- Optional counts of evaluation dates also present in each agent's training history. **Nonzero overlap means this is not a strictly held-out-date test.**

If `curve_band_data.csv` is absent, the audit explicitly reports `tested_curve_points: 0`; it does not fabricate curve validation. The audit cannot prove the algorithm converged, that the policy is globally optimal, that the real-world weather physics are accurate, or that confidence intervals are justified. It checks the *internal numerical consistency* of the saved measurements.

## Normal plotting (unchanged script)

```bash
./plot_modern_results.sh MBPO_workspace/evaluation/scientific_v52_fresh \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_seed0/episodes.csv \
  --sac-training-csv SAC_workspace/runs/sac_seed0/episodes.csv
```

## Checks for the replacement modules

```bash
python -m unittest MBPO_workspace.tests.test_workflow_audit_v53 -v
python -m unittest discover -s MBPO_workspace/tests -p 'test_*.py' -v
python -m unittest discover -s SAC_workspace/tests -p 'test_*.py' -v
```

Validated in a reconstructed v5.2 workspace with synthetic fixtures: 42 MBPO tests passed; 20 SAC tests passed or skipped (18 pass, 2 environmental skips). **User's real NSRDB result files were not available in this environment; use the audit command above to validate those measurements.**

## Statistical limits

A three-date-per-month scheme with one uniformly selected date per within-month bin supports design-weighted point estimates for the eligible 2018–2024 manifest. Its within-bin design variance is **not estimable** from one observed date per bin. Shaded weighted RMS dispersion in figures is *descriptive* across sampled dates, not SEM or a 95% CI. Training-date overlap means the paired result cannot be advertised as independent held-out-date generalization.
