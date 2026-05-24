"""Legacy 15-D observation ablation (power_norm + time features).

Same MBPO hyperparameters as 0.py; only observation_mode differs.
Use: mbpo run_local examples.development --config=examples.config.pv_tracking.1 ...
"""

import importlib

_0 = importlib.import_module('examples.config.pv_tracking.0')

CONFIG_VERSION = 'pv_tracking_legacy_v1_2026-05-23'

params = dict(_0.params)
params['config_version'] = CONFIG_VERSION
params['environment_kwargs'] = dict(_0.params['environment_kwargs'])
params['environment_kwargs']['observation_mode'] = 'legacy'
