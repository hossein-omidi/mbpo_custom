# Reproducible calendar-stratified NSRDB evaluation (v3)

The evaluation horizon and physical simulator are unchanged. Each NSRDB episode
runs **117 native five-minute physical transitions**, with post-action timestamps
12:05–21:45 UTC. Do not label this as 24-hour daily generation.

## Sampling and protocol

* For every populated **year × month** manifest cell, sort eligible ISO date IDs.
* Partition each cell into `min(K, eligible_count)` consecutive temporal bins.
* Use a separate deterministic RNG seeded by `(selection_seed, year, month)`.
  Select **one uniformly random date from each bin**. Every sampled date has
  inclusion probability `1 / bin_eligible_count`; analysis weight is
  `bin_eligible_count`. These weights correct unequal bin/month sizes in the
  **point estimate** of the full eligible-manifest mean.
* `--all-scenarios` (legacy flag) now means **stratified sampling**;
  `--stratified-scenarios` is an exact alias. `--exhaustive-scenarios` is the
  explicit, potentially costly old exhaustive behavior.
* Default `--days-per-month 3` picks up to 3/year/month; seven complete years
  2018–2024 therefore give 252 selected dates from 2,557 calendar dates.
  The actual manifest may contain fewer eligible dates. Choose K=1 for up to 84
  evaluations, K=2 for up to 168, or K=5 for up to 420. All methods use the
  exact same sampled dates, deterministic evaluation seed and fixed initial
  panel orientation.
* The `scenario_selection.json` contains source manifest SHA256 fingerprint,
  individual selected IDs, full population weights, sampling settings,
  excluded training IDs and missing year/month cells. The CSV allows inspection.
* `--selection-plan other_run/scenario_selection.json` replays *exactly*
  the same sampled scenario IDs; it will reject a different manifest.
* `--exclude-training-days` removes scenarios recorded in ALL supplied policy
  training `episodes.csv` files **before** constructing the sample. SAC needs
  `--training-csv`; MBPO-vs-SAC needs both training CSVs. Check that those
  histories match the policy checkpoints: incomplete/incorrect logs make this
  separation invalid. This is observed-day disjointness, not a guaranteed
  climatologically independent new dataset.

## Commands

Create a lightweight plan without loading a policy:

```bash
python -m SAC_workspace.workflows.06_plan_scenarios \
  --outdir SAC_workspace/evaluation/planned_sample --days-per-month 3 --selection-seed 42
```

Evaluate standalone SAC using that plan (you can omit `--selection-plan`
and instead specify `--all-scenarios --days-per-month 3 --selection-seed 42`):

```bash
python -m SAC_workspace.workflows.04_evaluate \
  --checkpoint SAC_workspace/runs/sac_seed1/latest.pt \
  --training-csv SAC_workspace/runs/sac_seed1/episodes.csv \
  --selection-plan SAC_workspace/evaluation/planned_sample/scenario_selection.json \
  --outdir SAC_workspace/evaluation/sac_calendar252 --device cpu
```

Compare MBPO and SAC on exactly the same dates, optionally omit the
expensive ideal POA optimizer (sun-pointing gross reference remains):

```bash
python -m MBPO_workspace.workflows.03_evaluate \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_seed0/latest.pt \
  --sac-checkpoint SAC_workspace/runs/sac_seed1/latest.pt \
  --selection-plan SAC_workspace/evaluation/planned_sample/scenario_selection.json \
  --skip-poa-opt --device cpu \
  --outdir MBPO_workspace/evaluation/paired_calendar252
```

For observed-unseen dates (requires trustworthy **both** training histories):

```bash
python -m MBPO_workspace.workflows.03_evaluate \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_seed0/latest.pt \
  --sac-checkpoint SAC_workspace/runs/sac_seed1/latest.pt \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_seed0/episodes.csv \
  --sac-training-csv SAC_workspace/runs/sac_seed1/episodes.csv \
  --all-scenarios --exclude-training-days --days-per-month 3 \
  --skip-poa-opt --outdir MBPO_workspace/evaluation/heldout_calendar_sample
```

To regenerate the calendar/seasonal/irradiance/movement figures from CSV
without touching physics or any checkpoint:

```bash
python -m SAC_workspace.workflows.05_stratified_report \
  --eval-dir MBPO_workspace/evaluation/paired_calendar252
```

Legacy plotting and the MBPO advanced-report entrypoint work unchanged.
The modern SAC and MBPO plotting outputs retain their previous filenames.

## Saved output files

```text
<outdir>/
    scenario_selection.json       # exact sampling plan and weights
    selected_scenarios.csv        # date/year/month/temporal-bin/weight
    summary.json                  # existing unweighted sample summary
    episodes.csv                  # unchanged baseline compatibility
    trajectories.csv              # unchanged baseline compatibility
    stratified/
        episode_features.csv      # one row per paired day/method/seed, replot-ready
        stratified_statistics.json
        sampling_statistics.pdf
        01_sampling_coverage.png
        02_monthly_energy_and_movement.png
        03_seasonal_energy.png
        04_monthly_paired_gains.png
        05_movement_energy_scatter.png
        06_weather_conditions.png         # only when real weather labels exist
        07_irradiance_regimes.png         # if DNI/DHI/GHI available
```

`stratified_statistics.json` contains weighted average net/gross energy,
actual applied angular movement (not merely commanded action), modeled
movement cost, absolute tracking errors, power and irradiance, month/year/
season panels, paired gain vs feasible controllers, ideal **gross** gaps,
observed weather-condition and irradiance-ratio bands, and bootstrap intervals.

## Statistical caveats

* `summary.json`, preexisting energy-bar charts, and older PDF figures report
  **unweighted sample summaries**, as before. For population-day weighted
  comparisons use `stratified/stratified_statistics.json` and its calendar plots.
* The point estimate is design-weighted for the **manifest eligible-date
  population**; data may overlap with training. It is NOT automatically an
  independent generalization evaluation.
* The reported interval is a **descriptive year-cluster bootstrap**. With only
  seven source years and one date per temporal sub-bin it does NOT completely
  estimate within-bin random sampling uncertainty or account for every kind
  of serial weather correlation. It is not a publication-grade exact design
  confidence interval. Increase the date budget, repeat independent selection
  seeds, and/or use year-held-out evaluation for stronger inferential claims.
* Since the original reference has `movement_penalty=0`, current net reward
  equals gross PV generation and does NOT model true motor electricity.
* Weather-condition labels are observed post-hoc, not privileged inputs to
  the SAC/MBPO policies. If labels are missing, a separate sampled-data
  diffuse irradiance fraction `sum(DHI)/sum(GHI)` analysis uses explicit
  heuristic bins; these bins are NOT certified cloud-type classifications.
* Fixed-date / scenario-specific evaluation still works using `--scenarios`;
  the default reference-five evaluation is unchanged when no selector is set.
* The instantaneous sun-pointing and optimized POA controllers are **gross
  references only** and must never be treated as feasible net alternatives.
* Real pvlib/NSRDB end-to-end speed, reproducibility, and learned policy
  performance still require verification on the actual repository/data.
