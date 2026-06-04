"""conf4 — full‑year random environment + alternative (extreme) hyperparameters (parallel to conf3)."""

import importlib

from examples.config.pv_tracking._paths import default_log_dir
from examples.config.pv_tracking.verified_dates import assert_dates_in_historical_catalog

# Base environment configuration from stage2 (same as conf1)
_stage2 = importlib.import_module('examples.config.pv_tracking.stage2_random_weather')

IRRADIANCE_PERTURBATION_STD = 0.0
OBSERVATION_NOISE_STD = 0.0

# Ensure the full-year range exists in the historical catalog
assert_dates_in_historical_catalog(['2020-01-01', '2020-12-31'])

CONFIG_VERSION = 'pv_tracking_conf4_alternative_2026-06'   # FIXED: was 'conf3'
TRAINING_STAGE = 'conf4'

params = dict(_stage2.params)
params['config_version'] = CONFIG_VERSION
params['log_dir'] = default_log_dir()
params['kwargs'] = dict(_stage2.params['kwargs'])

# Environment kwargs (full‑year random, from conf1)
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

# Evaluation environment (same as training, no fixed dates – from conf1)
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
    'movement_penalty': 0.0,
})
params['evaluation_environment_kwargs'].pop('fixed_eval_dates', None)

# Alternative hyperparameters (as you specified)
params['kwargs'].update({
    'n_epochs': 200,
    'real_ratio': 0.1,                  # very low → heavy model reliance
    'n_initial_exploration_steps': 5000,
    'discount': 1,
    'target_entropy': 'auto',
    'min_alpha': 0.1,
    'n_train_repeat': 20,
    'model_train_freq': 78,
    'max_model_rollout_length': 1,       # almost no model rollouts
    'rollout_schedule': [20, 200, 1, 1], # constant rollout length 1
    'q_loss_warning_threshold': 500.0,
})