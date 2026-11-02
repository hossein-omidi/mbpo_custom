# Full-day SAC + MBPO evaluation repair (source-only overlay)

## What was corrected

This package repairs the evaluation/reporting layer for the current full-day
NSRDB experiment: 08:45 UTC start, 196 weather timestamps, 195 five-minute
control transitions, ending at 01:00 UTC on the next date.

- Removes the old 117-step requirement from stratified reporting.
- Validates the actual 195-step post-action UTC grid, including midnight rollover.
- Records the full episode grid in deterministic sampling plans and fingerprints.
- Makes holdout/exclusion wording truthful in saved plans.
- Updates statistical estimand descriptions to the actual episode duration.
- Fixes MBPO heatmap indexing for all 195 transitions.
- Fixes SAC and MBPO time axes to display 24+ hours after midnight instead of
  jumping backward to 00:00.
- Adds a read-only run preflight and an offline report-rebuild utility.

No training algorithm, actor/critic, replay buffer, dynamics model, PV physics,
NSRDB source data, checkpoint, or reward is modified.

## Install

Extract at the repository root, preserving the folder structure:

```bash
unzip -o FULL_DAY_SAC_MBPO_EVALUATION_FIX_v2.zip -d .
```

Back up local changes first. The overlay does not contain or overwrite run
folders, checkpoints, datasets, or existing evaluation outputs.

## Step 1 — run regression tests

```bash
python -m pytest -q tests/test_full_day_evaluation_repair.py
```

Expected: 12 passed.

## Step 2 — read-only preflight of both existing training runs

```bash
python scripts/preflight_full_day_evaluation.py \
  --sac-run SAC_workspace/runs/sac_full_day_seed0_new \
  --mbpo-run MBPO_workspace/runs/mbpo_full_day_seed0_stageA2 \
  --days-per-month 4 \
  --selection-seed 42
```

This checks the checkpoint/history/config files, 195-step MDP metadata,
matching physical and NSRDB fingerprints, and the training-day exclusion plan.
It does not train, evaluate, or write files. If it fails, stop and resolve its
reported mismatch; do not bypass fingerprint checks.

## Step 3 — check whether the previous evaluation can be rebuilt offline

Locate saved detailed trajectories:

```bash
find MBPO_workspace/evaluation -maxdepth 4 -name trajectories_detailed.csv -print
```

Offline rebuilding is permitted only if the same output folder also contains
`scenario_selection.json` and `summary.json`, and the saved plan proves that
training dates were excluded. If those conditions hold, run (replace the path
with the exact folder found):

```bash
python scripts/rebuild_full_day_statistics.py \
  --eval-dir MBPO_workspace/evaluation/EXACT_EXISTING_FOLDER
```

It validates the saved four-method trajectories and rebuilds reports without
rerunning the PV simulator. It refuses incomplete trajectories, mismatched
UTC clocks, stale selection plans, or unverified training-date exclusions.

If the previous run did not exclude training dates, its results may still be
useful as exploratory results, but they must not be labelled held-out results.
Use Step 4 for a proper held-out evaluation.

## Step 4 — full paired held-out evaluation (plain command; no shell script)

Run this only after Step 2 passes. The output path must be new or empty.

```bash
python -m MBPO_workspace.workflows.03_evaluate \
  --power-model pvwatts_sapm \
  --wind-reference-height-m 2.0 \
  --gamma-pdc -0.004 \
  --sapm-mount open_rack_glass_polymer \
  --wind-powerlaw-exponent 0.14 \
  --sac-checkpoint SAC_workspace/runs/sac_full_day_seed0_new/latest.pt \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_full_day_seed0_stageA2/latest.pt \
  --sac-training-csv SAC_workspace/runs/sac_full_day_seed0_new/episodes.csv \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_full_day_seed0_stageA2/episodes.csv \
  --all-scenarios \
  --days-per-month 4 \
  --selection-seed 42 \
  --exclude-training-days \
  --outdir MBPO_workspace/evaluation/pvwatts_sapm_full_day_heldout_seed42_v2 \
  --seed 42 \
  --device cpu
```

Do not omit `--exclude-training-days`, either training CSV, or `--outdir`.
Do not concatenate the output path onto the `--device` value. The earlier
terminal paste showed `--device cpuO_workspace/...`; that is not a valid
substitute for the separate `--outdir` argument.

`--all-scenarios` means a reproducible calendar-stratified sample, not every
manifest day. With four days per month it selects up to four dates per
source-year/month after excluding all dates recorded in both training logs.
This is a logged-date holdout, not a leave-one-year-out experiment.

## Expected output checks

In `scenario_selection.json`, verify:

- `episode.start_time`: `08:45`
- `episode.periods`: `196`
- `episode.freq`: `5min`
- `episode.tz`: `UTC`
- `excluded_scenario_ids`: contains the training-day union
- `selected_scenario_ids`: disjoint from `excluded_scenario_ids`

In `summary.json`, verify `independent_training_day_exclusion` is `true`.
The stratified report should describe 195 transitions from 08:50 UTC through
01:00 UTC the next date.

## Verification limits

The overlay's Python syntax, 12 focused regression tests, and ZIP integrity
were verified in the build environment. Full integration with the user's WSL
NSRDB files and trusted checkpoints must be completed locally. Gym/CUDA
compatibility warnings are separate from these evaluation fixes.
