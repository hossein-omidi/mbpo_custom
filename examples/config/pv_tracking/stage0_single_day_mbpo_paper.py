"""Stage 0 — MBPO paper-aligned + movement cost (single clearsky day).

MDP: 2020-06-21, physical 11-D, reward = energy_kwh - movement_penalty * (|a_tilt| + |a_azimuth|).

MBPO (Janner et al. 2019): real_ratio=0.5, n_train_repeat=30, rollout_schedule=[20,200,1,5].
SAC: discount=1.0, target_entropy='auto', min_alpha=0.0.

Train / monitor / eval:
  docs/STAGE0_MBPO_PAPER_RUN.md
  ./scripts/run_stage0_paper_trial.sh {train|plot|eval|gate}
"""

import importlib

_baseline = importlib.import_module('examples.config.pv_tracking.stage0_single_day')

CONFIG_VERSION = 'pv_tracking_stage0_single_day_mbpo_paper_movement_2026-06-01'
TRAINING_STAGE = 'stage0'

params = dict(_baseline.params)
params['config_version'] = CONFIG_VERSION
params['kwargs'] = dict(_baseline.params['kwargs'])
params['kwargs'].update({
    'n_epochs': 500,
    'n_initial_exploration_steps': 3900,
    'discount': 1.0,
    'real_ratio': 0.5,
    'n_train_repeat': 30,
    'rollout_schedule': [20, 200, 1, 5],
})
