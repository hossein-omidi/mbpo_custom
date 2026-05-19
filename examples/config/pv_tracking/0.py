params = {
    'type': 'MBPO',
    'universe': 'gym',
    'domain': 'PVTracking',
    'task': 'v0',

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'pv_tracking',

    'kwargs': {
        'n_epochs': 5,
        'epoch_length': 64,
        'train_every_n_steps': 1,
        'n_train_repeat': 10,
        'eval_render_mode': None,
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 50,
        'model_retain_epochs': 1,
        'rollout_batch_size': 100,
        'deterministic': False,
        'num_networks': 5,
        'num_elites': 3,
        'real_ratio': 0.1,
        'target_entropy': -2,
        'max_model_t': None,
        'rollout_schedule': [1, 10, 1, 1],
        # ~10 full episodes (63 steps each) before policy training
        'n_initial_exploration_steps': 630,
    }
}
