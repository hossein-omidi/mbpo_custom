"""Stage 3 — full-year historical-weather robustness (same site / same physical state).

Full A->Z procedure is orchestrated by run_sequential_stages.sh.
This config preserves the working Stage 2 training setup and changes only:
  * TRAINING_STAGE / CONFIG_VERSION
  * n_epochs -> 500
  * full-year date range
  * six seasonally spaced fixed eval dates

Note: this file intentionally inherits the Stage 2 algorithm/training
hyperparameters. In particular, it does not introduce new restore logic or any
core pipeline changes.
"""

import importlib

_stage2 = importlib.import_module('examples.config.pv_tracking.stage2_random_weather')

STAGE3_FIXED_EVAL_DATES = [
    '2020-01-15',
    '2020-03-20',
    '2020-06-21',
    '2020-08-01',
    '2020-10-15',
    '2020-12-21',
]

CONFIG_VERSION = 'pv_tracking_stage3_fullyear_random_physical_2026-05-25'
TRAINING_STAGE = 'stage3'

params = dict(_stage2.params)
params['config_version'] = CONFIG_VERSION
params['kwargs'] = dict(_stage2.params['kwargs'])
params['kwargs'].update({
    'n_epochs': 500,
    'real_ratio': 0.9,
})
params['environment_kwargs'] = dict(_stage2.params['environment_kwargs'])
params['environment_kwargs'].update({
    'start_date': '2020-01-01',
    'end_date': '2020-12-31',
    'randomize_day': True,
    'weather_source': 'historical',
})
params['evaluation_environment_kwargs'] = dict(
    _stage2.params['evaluation_environment_kwargs'])
params['evaluation_environment_kwargs'].update({
    'start_date': '2020-01-01',
    'end_date': '2020-12-31',
    'fixed_eval_dates': STAGE3_FIXED_EVAL_DATES,
    'randomize_day': False,
    'weather_source': 'historical',
})
