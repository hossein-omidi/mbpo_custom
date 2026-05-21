params = {
    'type': 'MBPO',
    'universe': 'gym',
    'domain': 'PVTracking',
    'task': 'v0',

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'pv_tracking',

    # Balanced defaults: reasonable wall-clock on CPU while still learning a
    # useful tracking policy. Increase n_epochs (e.g. 200-500) for higher accuracy.
    'kwargs': {
        'n_epochs': 200,
        'epoch_length': 64,
        'train_every_n_steps': 1,
        'n_train_repeat': 10,
        'eval_render_mode': None,
        'eval_n_episodes': 5,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        # Train dynamics model every 100 env steps; cap wall time if needed.
        'model_train_freq': 100,
        'model_retain_epochs': 5,
        'max_model_t': 120,

        'rollout_batch_size': 300,
        'deterministic': False,
        'num_networks': 5,
        'num_elites': 3,
        'real_ratio': 0.5,
        'min_alpha': 0.05,
        'max_model_rollout_length': 3,
        'target_entropy': -2,
        'rollout_schedule': [20, 120, 1, 3],
        'save_every_epochs': 5,
        'early_stop_patience': 12,
        'monitor_metric': 'evaluation/return-average',
        'q_loss_warning_threshold': 0.1,
        'q_loss_stop_threshold': 1,

        # ~10 full PV episodes (63 steps) before SAC training starts.
        'n_initial_exploration_steps': 630,
    },
    'environment_kwargs': {
        'start_date': '2020-01-01',
        'end_date': '2020-12-31',
        'start_time': '06:00',
        'periods': 64,
        'freq': '15min',
        'randomize_day': True,
        'randomize_initial_orientation': True,
        'weather_source': 'random',
        'temperature': 23.0,
        'wind_speed': 2.0,
        'movement_penalty': 0.01,
    },
}
