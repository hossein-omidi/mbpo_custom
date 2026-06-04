"""Stage 3 — full-year annual-scenario RL (empirical weather distribution).

Environmental variability comes from sampling days from the annual catalog. Each day
provides an exogenous GHI/DNI/DHI, temperature, wind, and solar-geometry trajectory.
pvlib deterministically maps weather, time, location, and panel orientation to power.

Training:
  - All calendar days in [start_date, end_date] (no excluded_dates).
  - randomize_day=True samples d from the full annual set each reset.
  - irradiance_perturbation_std=0 for the main paper config (optional bounded augmentation only).

Evaluation: frozen policy, Monte Carlo day sampling, independent eval seeds — see
docs/RL_EVAL_PROTOCOL.md and docs/STAGE3_FULLYEAR_WORKFLOW.md.
"""

import importlib

from examples.config.pv_tracking.verified_dates import assert_dates_in_historical_catalog

_stage2 = importlib.import_module('examples.config.pv_tracking.stage2_random_weather')

# Paper default: 0.0 — deterministic TMY trajectory per sampled day (no intra-day cloud model).
# Set >0 only for optional bounded irradiance augmentation (not dynamic cloud stochasticity).
IRRADIANCE_PERTURBATION_STD = 0.0
OBSERVATION_NOISE_STD = 0.0

assert_dates_in_historical_catalog(['2020-06-21', '2020-12-07'])

CONFIG_VERSION = 'pv_tracking_stage3_fullyear_rl_annual_scenario_2026-06-04'
TRAINING_STAGE = 'stage3'

params = dict(_stage2.params)
params['config_version'] = CONFIG_VERSION
params['kwargs'] = dict(_stage2.params['kwargs'])
params['kwargs'].update({
    'n_epochs': 2500,
    'real_ratio': 0.5,
    'n_initial_exploration_steps': 12000, # was 8000
})
params['environment_kwargs'] = dict(_stage2.params['environment_kwargs'])
params['environment_kwargs'].update({
    'start_date': '2020-01-01',
    'end_date': '2020-12-31',
    'randomize_day': True,
    'randomize_initial_orientation': True,
    'weather_source': 'historical',
    'irradiance_perturbation_std': IRRADIANCE_PERTURBATION_STD,
    'observation_noise_std': OBSERVATION_NOISE_STD,
    'movement_penalty': 0.00,
    'observation_mode': 'physical',
})
params['environment_kwargs'].pop('excluded_dates', None)
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
    'movement_penalty': 0.00,
})
params['evaluation_environment_kwargs'].pop('fixed_eval_dates', None)
