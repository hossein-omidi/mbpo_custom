"""Stage 0 — stationary proof (single clearsky day, no day randomization).

Full A→Z procedure and directories: docs/TRAINING_PROTOCOL.md
  - Clean start: §0 (fresh trial, no --restore)
  - Train: mbpo run_local … --config=examples.config.pv_tracking.stage0_single_day
  - Eval:  evaluation/pv_stage0_single_day  (--fixed-eval-dates 2020-06-21)
  - Gate:  scripts/diagnose_tracking.py --eval-dir evaluation/pv_stage0_single_day --gate
"""

import importlib

_stage1 = importlib.import_module('examples.config.pv_tracking.0')

CONFIG_VERSION = 'pv_tracking_stage0_single_day_physical_autoentropy_2026-05-25'
TRAINING_STAGE = 'stage0'

params = dict(_stage1.params)
params['config_version'] = CONFIG_VERSION
params['kwargs'] = dict(_stage1.params['kwargs'])
params['kwargs'].update({
    'n_epochs': 500,
    # One day × 39 steps × ~50 episodes of uniform exploration.
    'n_initial_exploration_steps': 1950,
})
params['environment_kwargs'] = dict(_stage1.params['environment_kwargs'])
params['environment_kwargs'].update({
    'start_date': '2020-06-21',
    'end_date': '2020-06-21',
    'randomize_day': False,
    'weather_source': 'clearsky',
    'movement_penalty': 0.0,
})
params['evaluation_environment_kwargs'] = {
    'start_date': '2020-06-21',
    'end_date': '2020-06-21',
    'randomize_day': False,
    'fixed_eval_dates': ['2020-06-21'],
    'weather_source': 'clearsky',
    'movement_penalty': 0.0,
}
