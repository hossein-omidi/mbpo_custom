"""Short MBPO smoke test that must trigger synthetic rollouts.

Purpose:
  - exercise exact remaining_steps filtering
  - step rollout_length through 2, 3, 4, 5 across epochs
  - keep runtime small enough for local auditing
"""

import importlib

_base = importlib.import_module('examples.config.pv_tracking.0')

CONFIG_VERSION = 'pv_tracking_smoke_model_rollout_2026-05-26'
TRAINING_STAGE = 'smoke'

params = dict(_base.params)
params['config_version'] = CONFIG_VERSION
params['log_dir'] = '/home/ecer/PVRL/mbpo/smoke_runs'
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
    'n_initial_exploration_steps': 78,
})
params['environment_kwargs'] = dict(_base.params['environment_kwargs'])
params['environment_kwargs'].update({
    'start_date': '2020-06-01',
    'end_date': '2020-06-30',
    'weather_source': 'clearsky',
    'randomize_day': True,
    'movement_penalty': 0.0,
    'observation_mode': 'physical',
})
params['evaluation_environment_kwargs'] = {
    'fixed_eval_dates': ['2020-06-07', '2020-06-14'],
    'randomize_day': False,
    'weather_source': 'clearsky',
    'movement_penalty': 0.0,
}
