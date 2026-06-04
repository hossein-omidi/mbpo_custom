"""conf1 — full-year annual-scenario MBPO-SAC (main experiment)."""

import importlib

from examples.config.pv_tracking._paths import default_log_dir
from examples.config.pv_tracking.verified_dates import assert_dates_in_historical_catalog

_stage2 = importlib.import_module('examples.config.pv_tracking.stage2_random_weather')

IRRADIANCE_PERTURBATION_STD = 0.0
OBSERVATION_NOISE_STD = 0.0

assert_dates_in_historical_catalog(['2020-06-21', '2020-12-07'])

CONFIG_VERSION = 'pv_tracking_conf1_fullyear_annual_2026-06'
TRAINING_STAGE = 'conf1'

params = dict(_stage2.params)
params['config_version'] = CONFIG_VERSION
params['log_dir'] = default_log_dir()
params['kwargs'] = dict(_stage2.params['kwargs'])
params['kwargs'].update({
    'n_epochs': 500,
    'real_ratio': 0.5,
    'n_initial_exploration_steps': 12000,
})
params['environment_kwargs'] = dict(_stage2.params['environment_kwargs'])
params['environment_kwargs'].update({
    'start_date': '2020-01-01',
    'end_date': '2020-12-31',
    'randomize_day': True,
    'randomize_initial_orientation': True,
    'weather_source': 'historical',
    'irradiance_perturbation_std': IRRADIANCE_PERTURBATION_STD,
    'observation_noise_std': OBSERVATION_NOISE_STD,
    'movement_penalty': 0.0,
    'observation_mode': 'physical',
})
params['environment_kwargs'].pop('excluded_dates', None)
params['evaluation_environment_kwargs'] = dict(
    _stage2.params['evaluation_environment_kwargs'])
params['evaluation_environment_kwargs'].update({
    'start_date': '2020-01-01',
    'end_date': '2020-12-31',
    'randomize_day': True,
    'randomize_initial_orientation': True,
    'weather_source': 'historical',
    'irradiance_perturbation_std': IRRADIANCE_PERTURBATION_STD,
    'observation_noise_std': OBSERVATION_NOISE_STD,
    'movement_penalty': 0.00,
})
params['evaluation_environment_kwargs'].pop('fixed_eval_dates', None)
