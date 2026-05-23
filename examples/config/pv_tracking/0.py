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
        'n_epochs': 15,
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
        'real_ratio': 0.5,
        'min_alpha': 0.05,
        'max_model_rollout_length': 3,
        'target_entropy': -2,
        'rollout_schedule': [20, 120, 1, 3],
        'save_every_epochs': 5,
        'early_stop_patience': None,
        # Training-time Ray eval (not the same as scripts/evaluate_agent.py holdout).
        'monitor_metric': 'evaluation/return-average',
        'q_loss_warning_threshold': None,
        'q_loss_stop_threshold': None,

        'n_initial_exploration_steps': 390,
    },
    'environment_kwargs': {
        'start_date': '2020-01-01',
        'end_date': '2020-12-31',
        # Project time standard: UTC everywhere (pvlib, env, train, test, plots).
        # Daylight UTC grid 13:30-23:15, 15-min steps, 39 actions per day (35N/106W).
        'tz': 'UTC',
        'start_time': '13:30',
        'periods': 40,
        'freq': '15min',
        'randomize_day': True,
        'randomize_initial_orientation': True,
        'weather_source': 'random',
        'temperature': 23.0,
        'wind_speed': 2.0,
        'movement_penalty': 0.0001,
        'observation_mode': 'physical',
    },
}
