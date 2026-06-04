"""Stage 3 — full-year annual-scenario RL with stabilized MBPO/SAC settings.

Same MDP as stage3_fullyear_random_clean_split (full-year historical weather,
physical 11-D obs, movement_penalty=0). Hyperparameters are tuned to fix the
Q_loss explosion and evaluation/return-average oscillation seen when
real_ratio=0.5 and max imagined rollout length reaches 15–25 on 78-step days.

Changes vs clean_split (see module docstring in git diff):
  - real_ratio 0.10 (MBPO paper scale; 0.5 over-weighted imagined transitions)
  - max_model_rollout_length 5, conservative rollout_schedule
  - discount 0.995 for mild TD damping on finite-horizon energy
  - eval_n_episodes=4 on seasonally spaced STAGE3_VALIDATION_DATES
  - target_entropy='auto', min_alpha=0.02
  - q_loss_warning_threshold for early visibility of critic divergence

Train:
  mbpo run_local examples.development \\
    --config=examples.config.pv_tracking.stage3_fullyear_stable_mbpo \\
    --temp-dir=$PWD/.ray_tmp/stage3_stable
"""

import importlib

from examples.config.pv_tracking.verified_dates import (
    STAGE3_VALIDATION_DATES,
    assert_dates_in_historical_catalog,
)

_stage3 = importlib.import_module(
    'examples.config.pv_tracking.stage3_fullyear_random_clean_split')

assert_dates_in_historical_catalog(STAGE3_VALIDATION_DATES)

CONFIG_VERSION = 'pv_tracking_stage3_fullyear_stable_mbpo_2026-06-03'
TRAINING_STAGE = 'stage3'

params = dict(_stage3.params)
params['config_version'] = CONFIG_VERSION
params['kwargs'] = dict(_stage3.params['kwargs'])
params['kwargs'].update({
    'n_epochs': 2000,
    'n_initial_exploration_steps': 12000,
    # MBPO paper uses ~0.05; 0.5 caused critic batches to be half model-imagined
    # data and Q_loss to diverge once rollout_length > ~10 (see training_Q_loss.png).
    'real_ratio': 0.10,
    'discount': 1,
    'target_entropy': 'auto',
    'min_alpha': 0.01,
    'n_train_repeat': 20,
    'model_train_freq': 78,
    'max_model_rollout_length': 10,
    'rollout_schedule': [60, 600, 1, 10],
    'eval_n_episodes': len(STAGE3_VALIDATION_DATES),
    'q_loss_warning_threshold': 500.0,
})
params['environment_kwargs'] = dict(_stage3.params['environment_kwargs'])
params['evaluation_environment_kwargs'] = dict(
    _stage3.params['evaluation_environment_kwargs'])
params['evaluation_environment_kwargs'].update({
    'fixed_eval_dates': list(STAGE3_VALIDATION_DATES),
    'randomize_day': False,
    'randomize_initial_orientation': False,
})
