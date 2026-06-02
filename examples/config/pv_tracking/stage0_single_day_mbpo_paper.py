"""Stage 0 — MBPO paper-aligned + movement cost (single cloudy historical day).

MDP: 2020-06-21, physical 11-D, weather_source=historical (pvlib + PVGIS TMY, overcast episode).
Reward = energy_kwh - movement_penalty * (|a_tilt| + |a_azimuth|).

MBPO (Janner et al. 2019): real_ratio=0.5, n_train_repeat=30, rollout_schedule=[20,200,1,5].
SAC: discount=1.0, target_entropy='auto', min_alpha=0.0.

Train / monitor / eval:
  docs/STAGE0_MBPO_PAPER_RUN.md
  ./scripts/run_stage0_paper_trial.sh {train|plot|eval|gate}
"""

import importlib

from examples.config.pv_tracking.verified_dates import STAGE0_CLOUDY_HISTORICAL_DAY

_baseline = importlib.import_module('examples.config.pv_tracking.stage0_single_day')

CONFIG_VERSION = 'pv_tracking_stage0_single_day_mbpo_paper_cloudy_historical_2026-06-02'
TRAINING_STAGE = 'stage0'

# Historical TMY for this calendar day is predominantly overcast in the 13:30–23:15 UTC window
# (see scripts/verify_pv_state_space.py and verified_dates.py).
CLOUDY_EVAL_DATE = STAGE0_CLOUDY_HISTORICAL_DAY

params = dict(_baseline.params)
params['config_version'] = CONFIG_VERSION
params['kwargs'] = dict(_baseline.params['kwargs'])
params['kwargs'].update({
    'n_epochs': 400,
    'n_initial_exploration_steps': 6000,
    'discount': 1.0,
    'real_ratio': 1,
    'n_train_repeat': 30,
    'rollout_schedule': [20, 200, 1, 1],
})
params['environment_kwargs'] = dict(_baseline.params['environment_kwargs'])
params['environment_kwargs'].update({
    'start_date': CLOUDY_EVAL_DATE,
    'end_date': CLOUDY_EVAL_DATE,
    'randomize_day': False,
    'weather_source': 'historical',
})
params['evaluation_environment_kwargs'] = dict(
    _baseline.params.get('evaluation_environment_kwargs', {}))
params['evaluation_environment_kwargs'].update({
    'start_date': CLOUDY_EVAL_DATE,
    'end_date': CLOUDY_EVAL_DATE,
    'randomize_day': False,
    'fixed_eval_dates': [CLOUDY_EVAL_DATE],
    'weather_source': 'historical',
})
