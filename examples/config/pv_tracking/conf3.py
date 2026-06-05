"""conf3 — full-year annual-scenario balanced MBPO-SAC.

Combines conf1 and conf2:

- Full-year historical annual-scenario MDP.
- Physical 11-D observation mode.
- randomize_day=True for training.
- No excluded calendar days.
- No synthetic irradiance or observation noise.
- Long initial exploration from conf1.
- More stable SAC/MBPO settings from conf2.
- Fixed validation dates only for in-training checkpoint monitoring.
- Final paper evaluation should still use annual MC evaluation, not these monitor dates.

This config is energy-only by default: movement_penalty=0.0.
For the final movement-aware paper run, set movement_penalty=0.001 in both
environment_kwargs and evaluation_environment_kwargs.
"""

import importlib

from examples.config.pv_tracking._paths import default_log_dir
from examples.config.pv_tracking.verified_dates import (
    STAGE3_VALIDATION_DATES,
    assert_dates_in_historical_catalog,
)

_base = importlib.import_module('examples.config.pv_tracking.conf1')

IRRADIANCE_PERTURBATION_STD = 0.0
OBSERVATION_NOISE_STD = 0.0

assert_dates_in_historical_catalog(['2020-06-21', '2020-12-07'])
assert_dates_in_historical_catalog(STAGE3_VALIDATION_DATES)

CONFIG_VERSION = 'pv_tracking_conf3_fullyear_balanced_2026-06'
TRAINING_STAGE = 'conf3'

params = dict(_base.params)
params['config_version'] = CONFIG_VERSION
params['log_dir'] = default_log_dir()

# ---------------------------------------------------------------------
# MBPO/SAC hyperparameters
# ---------------------------------------------------------------------
params['kwargs'] = dict(_base.params['kwargs'])
params['kwargs'].update({
    # Longer than conf2, stronger than quick test, still manageable.
    'n_epochs': 1000,

    # Keep conf1 exploration strength.
    'n_initial_exploration_steps': 12000,

    # Balanced real/model learning.
    # 0.5 worked well for exploration, but instability mainly came from long rollouts.
    'real_ratio': 0.8,

    # Keep finite-horizon energy objective.
    'discount': 1.0,

    # More robust SAC entropy handling.
    'target_entropy': 'auto',
    'min_alpha': 0.02,

    # Stronger policy/critic updates.
    'n_train_repeat': 20,
    'train_every_n_steps': 1,

    # Train model roughly once per episode.
    'model_train_freq': 117,

    # Safer MBPO trust region.
    # Do not use 25 here: too long for 117-step PV days.
    'max_model_rollout_length': 10,
    'rollout_schedule': [60, 600, 1, 10],

    # Keep inherited/model settings explicit for clarity.
    'rollout_batch_size': 400,
    'num_networks': 7,
    'num_elites': 5,

    # More stable checkpoint monitoring.
    'eval_n_episodes': len(STAGE3_VALIDATION_DATES),
    'eval_deterministic': True,

    # Safety diagnostics.
    'q_loss_warning_threshold': 500.0,
    'q_loss_stop_threshold': None,
})

# ---------------------------------------------------------------------
# Training environment: annual-scenario MDP
# ---------------------------------------------------------------------
params['environment_kwargs'] = dict(_base.params['environment_kwargs'])
params['environment_kwargs'].update({
    'start_date': '2020-01-01',
    'end_date': '2020-12-31',

    'randomize_day': True,
    'randomize_initial_orientation': True,

    'weather_source': 'historical',
    'irradiance_perturbation_std': IRRADIANCE_PERTURBATION_STD,
    'observation_noise_std': OBSERVATION_NOISE_STD,

    # Energy-only pretraining / tracking objective.
    # For movement-aware final paper run, change to 0.001.
    'movement_penalty': 0.0,

    'observation_mode': 'physical',
})
params['environment_kwargs'].pop('excluded_dates', None)
params['environment_kwargs'].pop('fixed_eval_dates', None)

# ---------------------------------------------------------------------
# In-training evaluation environment
# ---------------------------------------------------------------------
params['evaluation_environment_kwargs'] = dict(
    _base.params['evaluation_environment_kwargs'])

params['evaluation_environment_kwargs'].update({
    'start_date': '2020-01-01',
    'end_date': '2020-12-31',

    # Fixed dates are only for stable checkpoint monitoring.
    # Final reporting should use annual MC evaluation.
    'fixed_eval_dates': list(STAGE3_VALIDATION_DATES),
    'randomize_day': True,
    'randomize_initial_orientation': True,

    'weather_source': 'historical',
    'irradiance_perturbation_std': IRRADIANCE_PERTURBATION_STD,
    'observation_noise_std': OBSERVATION_NOISE_STD,

    # Must match train objective.
    'movement_penalty': 0.0,

    'observation_mode': 'physical',
})