"""conf2 — full-year with conservative MBPO/SAC hyperparameters."""

import importlib

from examples.config.pv_tracking._paths import default_log_dir
from examples.config.pv_tracking.verified_dates import (
    STAGE3_VALIDATION_DATES,
    assert_dates_in_historical_catalog,
)

_base = importlib.import_module('examples.config.pv_tracking.conf1')

assert_dates_in_historical_catalog(STAGE3_VALIDATION_DATES)

CONFIG_VERSION = 'pv_tracking_conf2_fullyear_stable_2026-06'
TRAINING_STAGE = 'conf2'

params = dict(_base.params)
params['config_version'] = CONFIG_VERSION
params['log_dir'] = default_log_dir()
params['kwargs'] = dict(_base.params['kwargs'])
params['kwargs'].update({
    'n_epochs': 200,
    'n_initial_exploration_steps': 3000,
    'real_ratio': 0.80,
    'discount': 1,
    'target_entropy': 'auto',
    'min_alpha': 0.01,
    'n_train_repeat': 20,
    'model_train_freq': 78,
    'max_model_rollout_length': 25,
    'rollout_schedule': [20, 200, 1, 25],
    'eval_n_episodes': len(STAGE3_VALIDATION_DATES),
    'q_loss_warning_threshold': 500.0,
})
params['environment_kwargs'] = dict(_base.params['environment_kwargs'])
params['evaluation_environment_kwargs'] = dict(
    _base.params['evaluation_environment_kwargs'])
params['evaluation_environment_kwargs'].update({
    'fixed_eval_dates': list(STAGE3_VALIDATION_DATES),
    'randomize_day': False,
    'randomize_initial_orientation': False,
})
