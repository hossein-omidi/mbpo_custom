# Bumped when training hyperparameters change; verify_training_config.py checks this.
# Physical 11-D obs only (no legacy). Long-run production config.
CONFIG_VERSION = 'pv_tracking_v3_physical_2026-05-24'

params = {
    'type': 'MBPO',
    'universe': 'gym',
    'domain': 'PVTracking',
    'task': 'v0',
    'config_version': CONFIG_VERSION,

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'pv_tracking',

    # v3: 11-D physical obs; SAC entropy + more real env data (no legacy / no new obs dims).
    # Run: python scripts/verify_training_config.py before mbpo run_local.
    'kwargs': {
        'n_epochs': 250,
        # One PV day = 39 env steps (40 timestamps at 15 min UTC: 13:30 -> 23:15).
        'epoch_length': 39,
        'train_every_n_steps': 1,
        'n_train_repeat': 10,
        'eval_render_mode': None,
        'eval_n_episodes': 5,
        'eval_deterministic': True,

        'discount': 0.999,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 100,
        'model_retain_epochs': 5,
        'max_model_t': 120,

        'rollout_batch_size': 300,
        'deterministic': False,
        'num_networks': 5,
        'num_elites': 3,
        # More real-env SAC batches (less pessimistic imagined rollouts).
        'real_ratio': 0.9,
        # Higher floor + auto target entropy for 2-D actions (reduces late collapse).
        'min_alpha': 0.15,
        'max_model_rollout_length': 3,
        'target_entropy': 'auto',
        'rollout_schedule': [20, 150, 1, 3],
        'save_every_epochs': 5,
        'early_stop_patience': None,
        'monitor_metric': 'evaluation/return-average',
        'q_loss_warning_threshold': None,
        'q_loss_stop_threshold': None,

        # ~64 random episodes before policy learning.
        'n_initial_exploration_steps': 2500,
    },
    'environment_kwargs': {
        'start_date': '2020-01-01',
        'end_date': '2020-12-31',
        'tz': 'UTC',
        'start_time': '13:30',
        'periods': 40,
        'freq': '15min',
        'randomize_day': True,
        'randomize_initial_orientation': False,
        'weather_source': 'random',
        'temperature': 23.0,
        'wind_speed': 2.0,
        'movement_penalty': 0.0001,
        'observation_mode': 'physical',
    },
}
