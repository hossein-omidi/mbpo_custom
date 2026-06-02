# Bumped when training hyperparameters change; verify_training_config.py checks this.
# Stage 1 — summer i.i.d. days, clearsky, zero motion cost, hold-out-aligned training eval.
# A→Z procedure: docs/TRAINING_PROTOCOL.md  |  eval: evaluation/pv_stage1_clearsky_summer
from examples.config.pv_tracking.verified_dates import STAGE1_FIXED_EVAL_DATES

CONFIG_VERSION = 'pv_tracking_stage1_clearsky_physical_autoentropy_2026-06-02'
TRAINING_STAGE = 'stage1'

params = {
    'type': 'MBPO',
    'universe': 'gym',
    'domain': 'PVTracking',
    'task': 'v0',
    'config_version': CONFIG_VERSION,

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'pv_tracking',

    'kwargs': {
        'n_epochs': 500,
        'epoch_length': 78,
        'train_every_n_steps': 1,
        'n_train_repeat': 15,
        'eval_render_mode': None,
        'eval_n_episodes': 8,
        'eval_deterministic': True,

        'discount': 1,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 80,
        'model_retain_epochs': 5,
        'max_model_t': 120,

        'rollout_batch_size': 400,
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 4,

        'real_ratio': .5,
        'min_alpha': 0.01,
        'max_model_rollout_length': 2,
        'target_entropy': 'auto',
        'rollout_schedule': [200, 400, 1, 2],
        'save_every_epochs': 5,
        'early_stop_patience': None,
        'monitor_metric': 'evaluation/return-average',
        'q_loss_warning_threshold': None,
        'q_loss_stop_threshold': None,

        'n_initial_exploration_steps': 6000,
    },
    'environment_kwargs': {
        'start_date': '2020-06-01',
        'end_date': '2020-08-31',
        'tz': 'UTC',
        'start_time': '13:30',
        'periods': 79,
        'freq': '7min30s',
        'randomize_day': True,
        'randomize_initial_orientation': False,
        'weather_source': 'clearsky',
        'temperature': 25.0,
        'wind_speed': 2.0,
        'movement_penalty': 0.001,
        'observation_mode': 'physical',
    },
    # Training-time eval: fixed summer dates (not random December / full year).
    'evaluation_environment_kwargs': {
        'fixed_eval_dates': STAGE1_FIXED_EVAL_DATES,
        'randomize_day': False,
        'weather_source': 'clearsky',
        'movement_penalty': 0.001,
    },
}
