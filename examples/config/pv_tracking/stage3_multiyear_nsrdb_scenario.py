"""Stage 3 — NSRDB multi-year empirical weather (main recommended config).

weather_scenario_mode = nsrdb_multiyear
  e = (year, month, day) ~ Uniform(manifest), fixed W_e per episode.

Legacy PVGIS-TMY baseline: conf1 / conf2 (weather_scenario_mode = pvgis_tmy).

Data: local SAM CSVs data/pv_weather/nsrdb/nsrdb_{2018..2024}_utc_5min.csv
Reader: pvlib.iotools.read_psm3 (see tests/test1.py, mbpo.env.nsrdb_iotools)

Timing: native 5-min control grid synchronized with NSRDB (no interpolation).
  13:30–23:15 UTC → 118 timestamps, 117 transitions, interval_hours = 5/60.

Train:
  ./train.sh run_nsrdb stage3_nsrdb --verify

Eval:
  ./result.sh run_nsrdb --full
"""

import importlib
import os

from examples.config.pv_tracking._paths import default_log_dir

_base = importlib.import_module('examples.config.pv_tracking.0')

NSRDB_MANIFEST = 'data/pv_weather/nsrdb/albuquerque_multiyear_manifest.json'

EPISODE_TZ = 'UTC'
EPISODE_START_TIME = '13:30'
EPISODE_PERIODS = 118
EPISODE_FREQ = '5min'
EPISODE_ACTION_STEPS = EPISODE_PERIODS - 1  # 117

STAGE3_NSRDB_VALIDATION_SCENARIO_IDS = [
    '2018-02-15',
    '2019-05-15',
    '2020-08-15',
    '2021-11-15',
    '2024-06-21',
]


def assert_nsrdb_validation_scenarios(manifest_path, scenario_ids):
    from mbpo.env.nsrdb_weather import load_scenario_manifest

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    path = manifest_path if os.path.isabs(manifest_path) else os.path.join(repo_root, manifest_path)
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
        if episode.get(key) != expected:
            raise ValueError(
                'manifest episode.%s=%r != expected %r' % (key, episode.get(key), expected))
    site = manifest.get('site', {})
    if site.get('latitude') != 35.08 or site.get('longitude') != -106.65:
        raise ValueError('manifest site lat/lon mismatch: %s' % site)


assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, STAGE3_NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_stage3_nsrdb_multiyear_scenario_2026-06-06_5min'
TRAINING_STAGE = 'stage3_nsrdb'

params = dict(_base.params)
params['config_version'] = CONFIG_VERSION
params['log_dir'] = default_log_dir()
params['kwargs'] = dict(_base.params['kwargs'])
params['kwargs'].update({
    'n_epochs': 2000,
    'epoch_length': EPISODE_ACTION_STEPS,
    'n_initial_exploration_steps': 8000,
    'real_ratio': 0.75,
    'discount': 1,
    'model_train_freq': EPISODE_ACTION_STEPS,
    'max_model_rollout_length': 1,
    'rollout_schedule': [30, 400, 1, 1],
    'n_train_repeat': 2,
    'eval_n_episodes': len(STAGE3_NSRDB_VALIDATION_SCENARIO_IDS),
    'eval_deterministic': True,
    'q_loss_warning_threshold': 500.0,
    'monitor_metric': 'evaluation/return-average',
})
params['environment_kwargs'] = {
    'latitude': 35.08,
    'longitude': -106.65,
    'altitude': 1600.0,
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
}
params['evaluation_environment_kwargs'] = dict(params['environment_kwargs'])
params['evaluation_environment_kwargs'].update({
    'randomize_scenario': False,
    'randomize_initial_orientation': False,
    'fixed_eval_scenarios': list(STAGE3_NSRDB_VALIDATION_SCENARIO_IDS),
})
params['evaluation_environment_kwargs'].pop('fixed_eval_dates', None)
