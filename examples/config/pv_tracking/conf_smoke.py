"""conf_smoke — short MBPO run for pipeline verification."""

import importlib

from examples.config.pv_tracking._paths import default_log_dir

_base = importlib.import_module('examples.config.pv_tracking.0')

CONFIG_VERSION = 'pv_tracking_conf_smoke_2026-06'
TRAINING_STAGE = 'conf_smoke'

params = dict(_base.params)
params['config_version'] = CONFIG_VERSION
params['log_dir'] = default_log_dir('smoke_runs')
params['kwargs'] = dict(_base.params['kwargs'])
params['kwargs'].update({
    'n_epochs': 4,
    'n_train_repeat': 1,
    'eval_n_episodes': 1,
    'model_train_freq': 39,
    'model_retain_epochs': 1,
    'max_model_t': 20,
    'rollout_batch_size': 32,
    'num_networks': 3,
    'num_elites': 2,
    'real_ratio': 0.8,
    'max_model_rollout_length': 5,
    'rollout_schedule': [0, 3, 2, 5],
    'save_every_epochs': 0,
    'n_initial_exploration_steps': 117,
})
params['environment_kwargs'] = dict(_base.params['environment_kwargs'])
params['environment_kwargs'].update({
    'start_date': '2020-06-01',
    'end_date': '2020-08-31',
    'randomize_day': True,
    'weather_source': 'historical',
    'observation_mode': 'physical',
    'movement_penalty': 0.0,
})
params['evaluation_environment_kwargs'] = dict(
    _base.params.get('evaluation_environment_kwargs', {}))
params['evaluation_environment_kwargs'].update({
    'start_date': '2020-06-01',
    'end_date': '2020-08-31',
    'randomize_day': False,
    'fixed_eval_dates': ['2020-06-21'],
    'weather_source': 'historical',
    'observation_mode': 'physical',
})
