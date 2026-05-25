"""Stage 2 — summer random-weather robustness (same site / same physical state).

Full A->Z procedure and directories: docs/TRAINING_PROTOCOL.md
  - Train: mbpo run_local … --config=examples.config.pv_tracking.stage2_random_weather
  - Eval:  evaluation/pv_stage2_random_weather_summer
  - Gate:  scripts/diagnose_tracking.py --eval-dir evaluation/pv_stage2_random_weather_summer --gate

This stage intentionally preserves the working Stage 1 training procedure and
hyperparameters. It changes only the experiment definition:
  * TRAINING_STAGE / CONFIG_VERSION
  * weather_source: clearsky -> random
  * canonical evaluation outdir/protocol target

Note: real_ratio remains 1.0 by design, so this stage still uses real-env SAC
style batches only. True model rollouts require a separate MBPO ablation config.
"""

import importlib

_stage1 = importlib.import_module('examples.config.pv_tracking.0')

CONFIG_VERSION = 'pv_tracking_stage2_random_weather_summer_physical_2026-05-25'
TRAINING_STAGE = 'stage2'

params = dict(_stage1.params)
params['config_version'] = CONFIG_VERSION
params['kwargs'] = dict(_stage1.params['kwargs'])
params['environment_kwargs'] = dict(_stage1.params['environment_kwargs'])
params['environment_kwargs'].update({
    'weather_source': 'random',
})
params['evaluation_environment_kwargs'] = dict(
    _stage1.params['evaluation_environment_kwargs'])
params['evaluation_environment_kwargs'].update({
    'weather_source': 'random',
})
