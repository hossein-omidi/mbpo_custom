"""Stage 3 final full-year historical-weather config with clean date splits.

Training:
  - full year historical-weather physical observations
  - excludes validation and final-test dates from sampling

Validation:
  - fixed dates used only for checkpoint selection during training

Final test:
  - untouched dates used only in post-training evaluation commands
"""

import importlib

_stage2 = importlib.import_module('examples.config.pv_tracking.stage2_random_weather')

VALIDATION_DATES = [
    '2020-02-15',
    '2020-05-15',
    '2020-08-15',
    '2020-11-15',
]

FINAL_TEST_DATES = [
    '2020-01-15',
    '2020-03-20',
    '2020-06-21',
    '2020-09-22',
    '2020-10-15',
    '2020-12-21',
]

TRAIN_EXCLUDED_DATES = VALIDATION_DATES + FINAL_TEST_DATES

CONFIG_VERSION = 'pv_tracking_stage3_fullyear_random_clean_split_2026-05-26'
TRAINING_STAGE = 'stage3'

params = dict(_stage2.params)
params['config_version'] = CONFIG_VERSION
params['kwargs'] = dict(_stage2.params['kwargs'])
params['kwargs'].update({
    'n_epochs': 500,
    'real_ratio': 0.5,
})
params['environment_kwargs'] = dict(_stage2.params['environment_kwargs'])
params['environment_kwargs'].update({
    'start_date': '2020-01-01',
    'end_date': '2020-12-31',
    'randomize_day': True,
    'weather_source': 'historical',
    'excluded_dates': TRAIN_EXCLUDED_DATES,
    'movement_penalty': 0.0,
    'observation_mode': 'physical',
})
params['evaluation_environment_kwargs'] = dict(
    _stage2.params['evaluation_environment_kwargs'])
params['evaluation_environment_kwargs'].update({
    'start_date': '2020-01-01',
    'end_date': '2020-12-31',
    'excluded_dates': None,
    'fixed_eval_dates': VALIDATION_DATES,
    'randomize_day': False,
    'weather_source': 'historical',
    'movement_penalty': 0.0,
})
