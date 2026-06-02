"""Stage 0 — stationary proof (single clearsky day, no day randomization).

Full A→Z procedure and directories: docs/TRAINING_PROTOCOL.md
  - Clean start: §0 (fresh trial, no --restore)
  - Train: mbpo run_local … --config=examples.config.pv_tracking.stage0_single_day
  - Eval:  evaluation/pv_stage0_single_day  (inherit config; verified clear day)
  - Gate:  scripts/diagnose_tracking.py --eval-dir evaluation/pv_stage0_single_day --gate

Power/energy: pvlib via PVTrackingEnv (same path for learned policy and baselines).
"""

import importlib

from examples.config.pv_tracking.verified_dates import STAGE0_CLEARSKY_DAY

_stage1 = importlib.import_module('examples.config.pv_tracking.0')

CONFIG_VERSION = 'pv_tracking_stage0_single_day_clearsky_verified_2026-06-02'
TRAINING_STAGE = 'stage0'

params = dict(_stage1.params)
params['config_version'] = CONFIG_VERSION
params['kwargs'] = dict(_stage1.params['kwargs'])
params['kwargs'].update({
    'n_epochs': 500,
    # One day × 78 steps × ~25 episodes of uniform exploration.
    'n_initial_exploration_steps': 3900,
})
params['environment_kwargs'] = dict(_stage1.params['environment_kwargs'])
params['environment_kwargs'].update({
    'start_date': STAGE0_CLEARSKY_DAY,
    'end_date': STAGE0_CLEARSKY_DAY,
    'randomize_day': False,
    'weather_source': 'clearsky',
})
params['evaluation_environment_kwargs'] = dict(_stage1.params['evaluation_environment_kwargs'])
params['evaluation_environment_kwargs'].update({
    'start_date': STAGE0_CLEARSKY_DAY,
    'end_date': STAGE0_CLEARSKY_DAY,
    'randomize_day': False,
    'fixed_eval_dates': [STAGE0_CLEARSKY_DAY],
    'weather_source': 'clearsky',
})
