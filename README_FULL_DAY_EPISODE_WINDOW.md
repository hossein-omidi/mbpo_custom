# Full-day episode-window repair overlay

Target repository: `hossein-omidi/mbpo_custom`, branch `PVwatt_dev`  
Reviewed branch head: `9793dec1fcb37d12d8d6cd37318f4064e66d343c`

## What was already correct in the branch

The shared episode window and most of its data/evaluation plumbing were already present:

- UTC start: `08:45`
- 196 timestamps at native `5min` spacing
- 195 environment transitions
- Last timestamp: `01:00 UTC` on the following calendar date
- NSRDB manifest builder and runtime loader already join adjacent-year rows for Dec-31 episodes.
- SAC/MBPO training schedules derive their episode frequency from the shared 195-step constant.
- The current MBPO paired evaluator and advanced trajectory exporter are horizon-aware.

## What this repair fixes

The earlier update missed a hard-coded check in `SAC_workspace/src/environment/environment.py`:

```python
if self.horizon != 117:
    raise ValueError(...)
```

That check caused the exact MBPO traceback reported by the user. This overlay replaces it with validation against the configured timestamp count and the resolved environment grid.

The overlay also:
- updates `MBPO_workspace/representation_config.json` from 117 to 195 transitions;
- keeps the compact state at six features and corrects stale five-feature documentation;
- makes the physical-observation solar-time decoder roll over midnight safely;
- makes the Stage-A budget audit calculate transitions and replay-equivalent episodes from the resolved configuration;
- updates the NSRDB integration tests to use 195 transitions and to test adjacent-year data for Dec-31 episodes.

## Install

This is a **drop-in overlay**, not a patch file.

1. Extract this ZIP directly into the repository root.
2. Keep the folder structure and allow the included files to overwrite their matching files.

No `apply_full_day_episode_window.py` command is required. No code is committed to GitHub by this package. It does not overwrite datasets, manifests, checkpoints, training histories, or existing run directories.

## Dataset boundary

The NSRDB data loader uses real adjacent-year rows; it does not synthesize or interpolate weather. With the current 2018–2024 dataset, Dec-31 episodes for 2018–2023 can use the following year's CSV. The `2024-12-31` episode is correctly omitted unless an actual 2025 support CSV is present. This does not remove a season or affect the five fixed validation dates.

The failed MBPO run got past environment construction/reset and reached the old 117-step adapter check, so at least the selected manifest scenario was already loadable. The integration test in this overlay checks the included Dec-31 scenarios against the adjacent-year data.

## Training and old run directories

The old 117-step checkpoints and run snapshots are historical records. Do not resume them under the 195-step MDP.

The SAC error `FileExistsError` is independent: `SAC_workspace/runs/sac_full_day_seed0` already contains data. Keep it intact and use a new run directory, for example:

- `SAC_workspace/runs/sac_full_day_seed0_r1`
- `MBPO_workspace/runs/mbpo_full_day_seed0_r1`

The Gym deprecation and CUDA-driver messages are warnings, not the causes of either traceback. CPU execution remains valid.

## Verification

The compact regression test can be run with:

```bash
python -m pytest -q tests/test_full_day_episode_window.py
```

The NSRDB integration test requires the local canonical year CSVs and generated manifest:

```bash
python -m pytest -q tests/test_nsrdb_multiyear.py
```

The manifest should already report `start_time=08:45`, `periods=196`, and `freq=5min`. Do not regenerate it if those values and its 196-timestamp scenario rows are already present.

## Evaluation note

The current Stage-A MBPO paired evaluation (`MBPO_workspace/workflows/03_evaluate.py`) uses the horizon-aware MBPO evaluator. The separate legacy TensorFlow Monte-Carlo report script and the older Stage-5 diagnostic scripts contain older protocol assumptions and are not certified by this overlay.

The branch tree also contains `SAC_workspace/SAC_workspace_evaluation.zip`, while `SAC_workspace/src/evaluation/` is absent from the tracked tree. If that archive has not already been extracted in the local working copy, the standalone/paired evaluation imports need that separate package restored before evaluation. This is separate from the 117-step training crash.
