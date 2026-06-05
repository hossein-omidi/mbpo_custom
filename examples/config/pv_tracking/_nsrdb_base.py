"""Shared NSRDB multi-year scenario settings for PVTracking MBPO-SAC configs."""

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

NSRDB_VALIDATION_SCENARIO_IDS = [
    '2018-02-15',
    '2019-05-15',
    '2020-08-15',
    '2021-11-15',
    '2024-06-21',
]

NSRDB_CONF_NAMES = frozenset({
    'stage3_nsrdb', 'conf1', 'conf2', 'conf3', 'conf4',
})


def assert_nsrdb_validation_scenarios(manifest_path=None, scenario_ids=None):
    from mbpo.env.nsrdb_weather import load_scenario_manifest

    manifest_path = manifest_path or NSRDB_MANIFEST
    scenario_ids = scenario_ids or NSRDB_VALIDATION_SCENARIO_IDS
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


def nsrdb_environment_kwargs(**overrides):
    """Training env: NSRDB multi-year e~(year,day), native 5-min pvlib physics."""
    kw = {
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
    kw.update(overrides)
    return kw


def nsrdb_evaluation_environment_kwargs(**overrides):
    """In-train eval: fixed NSRDB validation scenarios (cycles each epoch)."""
    kw = nsrdb_environment_kwargs(
        randomize_scenario=False,
        randomize_initial_orientation=False,
        fixed_eval_scenarios=list(NSRDB_VALIDATION_SCENARIO_IDS),
    )
    kw.update(overrides)
    kw.pop('fixed_eval_dates', None)
    return kw


def default_nsrdb_algo_kwargs(**overrides):
    """MBPO/SAC defaults aligned with stage3_multiyear_nsrdb_scenario (main reference)."""
    kw = dict(_base.params['kwargs'])
    kw.update({
        'epoch_length': EPISODE_ACTION_STEPS,
        'model_train_freq': EPISODE_ACTION_STEPS,
        'n_epochs': 2000,
        'n_initial_exploration_steps': 8000,
        'real_ratio': 0.75,
        'discount': 1.0,
        'max_model_rollout_length': 1,
        'rollout_schedule': [30, 400, 1, 1],
        'n_train_repeat': 2,
        'eval_n_episodes': len(NSRDB_VALIDATION_SCENARIO_IDS),
        'eval_deterministic': True,
        'q_loss_warning_threshold': 500.0,
        'monitor_metric': 'evaluation/return-average',
    })
    kw.update(overrides)
    return kw


def build_nsrdb_params(
        config_version,
        training_stage,
        algo_kwargs=None,
        train_env_kwargs=None,
        eval_env_kwargs=None):
    """Assemble full params dict for NSRDB training."""
    params = dict(_base.params)
    params['config_version'] = config_version
    params['training_stage'] = training_stage
    params['log_dir'] = default_log_dir()
    params['kwargs'] = default_nsrdb_algo_kwargs(**(algo_kwargs or {}))
    params['environment_kwargs'] = nsrdb_environment_kwargs(**(train_env_kwargs or {}))
    params['evaluation_environment_kwargs'] = nsrdb_evaluation_environment_kwargs(
        **(eval_env_kwargs or {}))
    return params
