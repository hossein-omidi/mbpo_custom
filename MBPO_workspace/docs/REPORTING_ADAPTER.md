# Complementary reporting patch (v1)

## What is in the uploaded `result.sh`?

`result.sh` is a **legacy orchestration script**. It resolves Ray trial directories, activates
Conda `mbpo`, calls `plot_training_progress.py`, runs old TensorFlow-policy evaluation, and then
calls `diagnose_tracking.py` and `verify_pv_state_space.py`. Its `do_eval` path contains
`rm -rf "$EVAL_OUT"`, which deletes previous output directories; **do not use it for modern
runs**. This patch neither replaces nor deletes it. The advanced calculations come from its
referenced source modules (below), not from the shell script itself.

## Reuse and modernization provenance

| Original project file / function | Modern implementation |
| --- | --- |
| `scripts/plot_paper_eval.py::plot_energy_decomposition` | gross, net, and actuator movement-cost decomposition |
| `::plot_daily_gain_distributions` | paired daily net-gain distributions, keyed by scenario+seed, never array index |
| `::plot_per_scenario_ratio` | learned net / slew-limited dual-axis **net**; no illegal ideal-oracle net ratio |
| `::plot_hourly_power_gap_heatmap` | 5-minute power deficit relative to feasible dual-axis tracker |
| `::plot_orientation_strategy_heatmap` | occupancy of physical panel tilt and azimuth |
| `::plot_movement_efficiency` | actual angular displacement versus net-energy yield |
| `scripts/diagnose_tracking.py::plot_actions_time_series` | normalized action time-series for MBPO, SAC, and dual-axis reference |
| `::plot_actions_3d` | time × normalized tilt and azimuth action |
| `::plot_orientation_3d` | time × panel pose overlaid with astronomical sun-pointing target |
| `scripts/plot_rollout_trajectory.py::plot_rollout` | UTC post-step timestamp convention and physical trajectory export |
| `scripts/plot_training_progress.py` | prior modern SAC/MBPO training plots remain in existing evaluator; no Ray progress loader |

All figures are adapted to `MBPO_workspace` + `SAC_workspace` actual evaluation paths.
The implementation imports existing `MBPO_workspace.src.evaluation` methods and
uses the `SAC_workspace` rollout information, which already originates from the
original pvlib `PVTrackingEnv`. It does **not** import legacy TensorFlow/Ray plotting
modules at runtime; those currently have heavyweight transitive dependencies and
assume legacy trial/policy formats.

## Minimal changed files

- `MBPO_workspace/src/paper_reporting.py` (new): validates paired paths, writes
  `trajectories_rich.csv`, regenerates PDF and advanced 2D/3D figures.
- `MBPO_workspace/workflows/04_advanced_report.py` (new): offline plot regeneration.
- `MBPO_workspace/workflows/03_evaluate.py` (minor extension): automatically calls
  rich reporting *after* the existing evaluated paths and original reports.
- `SAC_workspace/src/evaluation/evaluation.py` (one tuple extension): retains
  physical weather and solar metadata in `infos` for reporting; no SAC training changes.
- `MBPO_workspace/tests/test_advanced_reporting.py` (new): physical invariants and plots.

## Commands (from repository root, current modern Python environment)

```bash
python -m unittest discover -s SAC_workspace/tests -p 'test_*.py' -v
python -m unittest discover -s MBPO_workspace/tests -p 'test_*.py' -v

python -m MBPO_workspace.workflows.03_evaluate \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_seed0/latest.pt \
  --sac-checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_seed0/episodes.csv \
  --sac-training-csv SAC_workspace/runs/sac_seed0/episodes.csv \
  --outdir MBPO_workspace/evaluation/paired

# No simulator or checkpoint load: regenerate figures from previously saved rich CSV
python -m MBPO_workspace.workflows.04_advanced_report \
  --eval-dir MBPO_workspace/evaluation/paired
```

### Output

Under `MBPO_workspace/evaluation/paired/advanced/`:

- `trajectories_rich.csv` — 117 rows per method/scenario, including true
  normalized commanded actions, pre- and post-action panel pose, true UTC
  solar timestamps, irradiance/weather labels, physical W and kWh.
- `advanced_summary.json`, `figure_manifest.txt`, `advanced_paper_figures.pdf`.
- 11 advanced PNGs when condition labels are available (10 otherwise).

The original `summary.json`, `episodes.csv`, `trajectories.csv`,
`comparison_report.pdf`, SAC reference figures, and all `scripts/` legacy
modules remain untouched.

## Scientific safeguards and limitations

1. Every paired method must have the same 117 post-step timestamps, five-minute
   spacing, initial pose, and (when recorded) identical irradiance and solar
   geometry. Deviations raise errors rather than being smoothed or interpolated.
2. Net comparisons include **only** MBPO, SAC, fixed mounting, and slew-limited
   dual-axis, evaluated with the identical simulator reward. Instantaneous
   dual-axis and ideal POA are reported for gross energy only.
3. The hourly power gap uses the *feasible* dual-axis controller, not an ideal
   oracle with unknown movement cost.
4. Weather categories are taken from the original environment and summarized
   by each episode's most frequent 5-minute category. No weather labels are
   inferred from the controller's actions.
5. Angular movement plots show control effort, but when `movement_penalty=0`,
   angular movement is **not** an energy cost. Real motor consumption has not
   been modeled in that experiment.
6. The solar pointing angle is a heuristic diagnostic, not a globally optimal
   control target under slew and future horizons.
7. The original compact 6D observation can be partially observable. The new
   diagrams do not claim theoretical optimality or generalization to unseen
   scenarios. For paper-grade conclusions use disjoint evaluation dates and
   multiple independently trained policy seeds.
8. Only the plotting/diagnostic path has been augmented. This patch does not
   change MBPO ensemble optimization, replay ratios, entropy tuning, or
   physics. Long NSRDB training and end-to-end evaluation require verification
   in the actual repository environment.
