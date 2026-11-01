# Full-daylight UTC episode-window update

Target source layout: `hossein-omidi/mbpo_custom`, branch `PVwatt_dev`.

## Why this is not a one-number change

The old episode is `12:00–21:45 UTC` (118 timestamps, 117 transitions). Increasing only `periods` cannot restore the summer morning daylight omitted before 12:00 UTC. The fixed annual daylight envelope recommended here is:

- Start: `08:45 UTC`
- Last timestamp: `01:00 UTC` on the following calendar date
- Frequency: `5min`
- `periods = 196` timestamps
- `EPISODE_ACTION_STEPS = 195` transitions
- Duration: `16h 15min`

For New York (40.72 N, 74.01 W), approximate annual solar geometry over 2018–2024 gives earliest sunrise around 09:24 UTC and latest sunset around 00:31 UTC. The proposed window leaves about 39 minutes before earliest sunrise and 29 minutes after latest sunset. It covers apparent daylight, not civil twilight.

## What the updater changes

The updater safely updates shared train/evaluation defaults and the hard-coded horizon checks in the SAC/MBPO evaluation pipeline. It also makes the NSRDB manifest builder and runtime weather loader join the next year's native NSRDB rows for episodes crossing midnight. It does not fabricate or interpolate weather.

The updater changes source files only. It does not modify checkpoints, training logs, year CSVs, or the existing manifest.

## Apply

From the repository root, after extracting this ZIP:

```bash
python scripts/apply_full_day_episode_window.py --check-only
python scripts/apply_full_day_episode_window.py
python -m pytest -q tests/test_full_day_episode_window.py
python scripts/prepare_nsrdb_newyork_catalog.py --years 2018-2024
```

The manifest builder uses next-year files already present in `data/pv_weather/nsrdb` for Dec 31 episodes in 2018–2023. If `newyork_2025_5min.csv` is absent, the 2024-12-31 episode is explicitly reported and omitted because its final 00:00–01:00 UTC weather rows are not in the 2024 file. Do not synthesize those rows. The downloader in this branch is limited to 2018–2024; obtaining 2025 support data would require a separately verified source.

After regeneration, inspect the manifest:

```bash
python - <<'PY'
import json
p='data/pv_weather/nsrdb/newyork_multiyear_manifest.json'
m=json.load(open(p, encoding='utf-8'))
print('episode:', m['episode'])
print('scenarios:', len(m['scenarios']))
print('Dec 31 scenarios:', [s['scenario_id'] for s in m['scenarios'] if s['month']==12 and s['day']==31])
print('example:', m['scenarios'][0]['timestamps'][0], '->', m['scenarios'][0]['timestamps'][-1])
PY
```

Expected episode metadata: `start_time=08:45`, `periods=196`, `freq=5min`, `action_steps=195`. For complete 2018–2024 scenario coverage, the 2024-12-31 boundary needs actual 2025 NSRDB support rows; otherwise that single scenario is omitted with a warning.

## Important training consequences

1. The observation `tau` now runs across 195 actions instead of 117. This is a different finite-horizon MDP.
2. Old SAC and MBPO checkpoints must **not** be resumed or compared as if they were trained on the same MDP. Use new run directories and train both from scratch.
3. At 500 episodes, each algorithm now collects `500 × 195 = 97,500` real transitions, versus 58,500 previously.
4. `epoch_length` and MBPO `model_train_freq` are derived from `EPISODE_ACTION_STEPS` in the shared base config and therefore become 195 automatically.
5. The existing 50,000-transition replay capacity now holds about 256 full episodes instead of 427. This updater deliberately does not change replay or model hyperparameters because the requested change is the episode window only. Reassess buffer coverage after the first new training run.
6. Winter episodes contain several hours of night-time zero-irradiance steps because a fixed-length UTC window must cover the longest summer daylight envelope. This is the trade-off for keeping one fixed horizon and a consistent `tau` definition across seasons.

## Evaluation compatibility

The update makes `scripts/eval_utils.py`, `MBPO_workspace/src/evaluation.py`, and `MBPO_workspace/src/advanced_suite.py` horizon-aware, including the post-midnight unwrapped UTC plot axis. The previously supplied Stage 5 diagnostic ZIP hard-codes 117 steps and the old 12:00–21:45 window. **Do not run that old Stage 5 script on new checkpoints.** The next action-conditioned diagnostic must be updated to read the new horizon and permit timestamps to cross midnight.

## Safety

- Make a Git commit or backup before applying.
- Run `--check-only` first. It aborts without writing if any expected source block differs.
- The updater does not edit any user data, checkpoint, or manifest.
- Regenerate the manifest only after applying the source update.
