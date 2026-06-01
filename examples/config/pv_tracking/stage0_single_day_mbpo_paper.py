"""Stage 0 — MBPO paper-aligned hyperparameters (single clearsky day).

Same MDP as stage0_single_day.py. Training knobs: real_ratio=0.5, n_train_repeat=30,
rollout_schedule=[20,200,1,5]. CPU / parallel runs: docs/TRAINING_PROTOCOL.md §0.

  PV_CPU_PROFILE=dual ./scripts/run_stage0_paper_trial.sh train
"""

import importlib

_baseline = importlib.import_module('examples.config.pv_tracking.stage0_single_day')

CONFIG_VERSION = 'pv_tracking_stage0_single_day_mbpo_paper_2026-05-31'
TRAINING_STAGE = 'stage0'

params = dict(_baseline.params)
params['config_version'] = CONFIG_VERSION
params['kwargs'] = dict(_baseline.params['kwargs'])
params['kwargs'].update({
    'n_epochs': 500,
    'n_initial_exploration_steps': 1950,
    'real_ratio': 0.5,
    'n_train_repeat': 30,
    'rollout_schedule': [20, 200, 1, 5],
})
params['environment_kwargs'] = dict(_baseline.params['environment_kwargs'])
params['evaluation_environment_kwargs'] = dict(
    _baseline.params['evaluation_environment_kwargs'])
