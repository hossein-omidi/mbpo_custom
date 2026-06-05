"""Stage 3 — NSRDB multi-year empirical weather scenarios (separate from PVGIS-TMY baseline).

Mode label: weather_scenario_mode = nsrdb_multiyear
Baseline TMY: conf1/conf2 with weather_scenario_mode = pvgis_tmy (weather_source=historical)

Lineage: MBPO/SAC hyperparameters from conf1 (main annual experiment); episode grid from
0.py (13:30 UTC, 79×7min30s = 78 actions); conf2 epoch-aligned model scheduling
(model_train_freq=78, rollout_schedule).

Stochasticity model (important):
  - Across episodes: sample one real historical scenario e = (year, month, day) ~ Uniform(manifest).
  - Within an episode: weather W_e(t) follows that scenario's NSRDB trajectory (5-min UTC
    time-interpolated onto the 7min30s control grid). It is NOT independently random at
    each 5-min or 7min30s step — temporal cloud/irradiance structure is preserved.
  - pvlib is deterministic given W_e and panel pose.

Requires catalog from offline SAM CSVs:
  python scripts/prepare_nsrdb_multiyear_catalog.py \\
    --weather-dataset-dir /home/user01/weather_dataset/weather_dataset \\
    --years 2018-2024 --copy-to-outdir

Train:
  ./train.sh run_nsrdb stage3_nsrdb --verify

Eval (frozen policy, scenario MC — result.sh auto-detects stage3_nsrdb):
  ./result.sh run_nsrdb --full
"""

import importlib
import os

from examples.config.pv_tracking._paths import default_log_dir

_base = importlib.import_module('examples.config.pv_tracking.conf1')

NSRDB_MANIFEST = 'data/pv_weather/nsrdb/albuquerque_multiyear_manifest.json'

# Episode grid (must match 0.py / manifest episode block / MBPO epoch_length=78).
EPISODE_TZ = 'UTC'
EPISODE_START_TIME = '13:30'
EPISODE_PERIODS = 79
EPISODE_FREQ = '7min30s'

# Seasonally spaced scenario IDs — must exist in manifest after prepare script.
STAGE3_NSRDB_VALIDATION_SCENARIO_IDS = [
    '2018-02-15',
    '2019-05-15',
    '2020-08-15',
    '2021-11-15',
    '2024-06-21',
]


def assert_nsrdb_validation_scenarios(manifest_path, scenario_ids):
    """Fail fast if manifest or validation scenario IDs are missing."""
    from mbpo.env.nsrdb_weather import load_scenario_manifest

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    path = manifest_path
    if not os.path.isabs(path):
        path = os.path.join(repo_root, manifest_path)
    manifest = load_scenario_manifest(path)
    ids = {s['scenario_id'] for s in manifest['scenarios']}
    missing = [sid for sid in scenario_ids if sid not in ids]
    if missing:
        raise ValueError(
            'NSRDB validation scenarios missing from manifest %s: %s' % (
                manifest_path, missing))
    episode = manifest.get('episode', {})
    for key, expected in (
        ('start_time', EPISODE_START_TIME),
        ('periods', EPISODE_PERIODS),
        ('freq', EPISODE_FREQ),
    ):
        actual = episode.get(key)
        if actual != expected:
            raise ValueError(
                'manifest episode.%s=%r != expected %r' % (key, actual, expected))
    site = manifest.get('site', {})
    if site.get('latitude') != 35.08 or site.get('longitude') != -106.65:
        raise ValueError('manifest site lat/lon mismatch: %s' % site)


assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, STAGE3_NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_stage3_nsrdb_multiyear_scenario_2026-06-03'
TRAINING_STAGE = 'stage3_nsrdb'

params = dict(_base.params)
params['config_version'] = CONFIG_VERSION
params['log_dir'] = default_log_dir()
params['kwargs'] = dict(_base.params['kwargs'])
params['kwargs'].update({
    'n_epochs': 2000,
    'n_initial_exploration_steps': 12000,
    # conf2 epoch-aligned model scheduling (real_ratio stays conf1=0.5).
    'model_train_freq': 78,
    'max_model_rollout_length': 25,
    'rollout_schedule': [20, 200, 1, 25],
    'n_train_repeat': 20,
    'eval_n_episodes': len(STAGE3_NSRDB_VALIDATION_SCENARIO_IDS),
    'q_loss_warning_threshold': 500.0,
})
params['environment_kwargs'] = dict(_base.params['environment_kwargs'])
params['environment_kwargs'].update({
    'latitude': 35.08,
    'longitude': -106.65,
    'tz': EPISODE_TZ,
    'start_time': EPISODE_START_TIME,
    'periods': EPISODE_PERIODS,
    'freq': EPISODE_FREQ,
    'start_date': '2018-01-01',
    'end_date': '2024-12-31',
    'randomize_day': False,
    'randomize_scenario': True,
    'randomize_initial_orientation': True,
    'weather_scenario_mode': 'nsrdb_multiyear',
    'weather_source': 'nsrdb_multiyear',
    'scenario_manifest': NSRDB_MANIFEST,
    'irradiance_perturbation_std': 0.0,
    'observation_noise_std': 0.0,
    'movement_penalty': 0.0,
    'observation_mode': 'physical',
})
params['evaluation_environment_kwargs'] = dict(
    _base.params['evaluation_environment_kwargs'])
params['evaluation_environment_kwargs'].update({
    'latitude': 35.08,
    'longitude': -106.65,
    'tz': EPISODE_TZ,
    'start_time': EPISODE_START_TIME,
    'periods': EPISODE_PERIODS,
    'freq': EPISODE_FREQ,
    'start_date': '2018-01-01',
    'end_date': '2024-12-31',
    'randomize_day': False,
    'randomize_scenario': False,
    'randomize_initial_orientation': False,
    'weather_scenario_mode': 'nsrdb_multiyear',
    'weather_source': 'nsrdb_multiyear',
    'scenario_manifest': NSRDB_MANIFEST,
    'fixed_eval_scenarios': list(STAGE3_NSRDB_VALIDATION_SCENARIO_IDS),
    'irradiance_perturbation_std': 0.0,
    'observation_noise_std': 0.0,
    'movement_penalty': 0.0,
    'observation_mode': 'physical',
})
params['evaluation_environment_kwargs'].pop('fixed_eval_dates', None)
