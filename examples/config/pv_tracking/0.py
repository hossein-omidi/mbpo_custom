# Minimal MBPO-SAC skeleton for PVTracking (episode grid: 5min, 117 transitions).
# NSRDB experiment configs inherit this structure via _nsrdb_base.py.
#
# Bump CONFIG_VERSION when changing default hyperparameters below.

CONFIG_VERSION = 'pv_tracking_mbpo_skeleton_2026-06-06'
TRAINING_STAGE = 'base'

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
        'epoch_length': 117,
        'train_every_n_steps': 1,
        'n_train_repeat': 10,
        'eval_render_mode': None,
        'eval_n_episodes': 5,
        'eval_deterministic': True,

        'discount': 0.995,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 117,
        'model_retain_epochs': 5,
        'max_model_t': 120,

        'rollout_batch_size': 400,
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 5,

        'real_ratio': 0.10,
        'min_alpha': 0.01,
        'max_model_rollout_length': 5,
        'target_entropy': 'auto',
        'rollout_schedule': [30, 400, 1, 5],
        'save_every_epochs': 5,
        'early_stop_patience': None,
        'monitor_metric': 'evaluation/return-average',
        'q_loss_warning_threshold': 500.0,
        'q_loss_stop_threshold': None,

        'n_initial_exploration_steps': 6000,
    },
    'environment_kwargs': {
        'tz': 'UTC',
        'start_time': '13:30',
        'periods': 118,
        'freq': '5min',
        'observation_mode': 'physical',
        'movement_penalty': 0.0,
    },
    'evaluation_environment_kwargs': {},
}
