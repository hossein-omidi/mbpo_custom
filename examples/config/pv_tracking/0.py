# Bumped when training hyperparameters change; verify_training_config.py checks this.
# Physical 11-D obs only. v4 targets higher daily energy (beat sun tracker on hold-out).
CONFIG_VERSION = 'pv_tracking_v4_beat_sun_2026-05-24'

params = {
    'type': 'MBPO',
    'universe': 'gym',
    'domain': 'PVTracking',
    'task': 'v0',
    'config_version': CONFIG_VERSION,

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'pv_tracking',

    # v4 rationale (physical 11-D, same pipeline):
    # - Sun tracker moves every step; learned policy was too static (entropy floor + motion penalty).
    # - Lower movement_penalty (env): same formula for learned + baselines at eval (fair).
    # - Higher min_alpha / target_entropy: more training stochasticity → less collapsed mean policy.
    # - More real data (real_ratio), longer MBPO imagined rollouts (horizon within one day).
    # - Longer train + ~100-day uniform exploration before policy learning.
    'kwargs': {
        'n_epochs': 300,
        'epoch_length': 39,
        'train_every_n_steps': 1,
        'n_train_repeat': 15,
        'eval_render_mode': None,
        'eval_n_episodes': 8,
        'eval_deterministic': True,

        'discount': 0.999,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 80,
        'model_retain_epochs': 5,
        'max_model_t': 120,

        'rollout_batch_size': 400,
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 4,

        'real_ratio': 0.95,
        'min_alpha': 0.2,
        'max_model_rollout_length': 5,
        # Slightly less negative than 'auto' (~-2 for 2-D) → keep policy entropy higher.
        'target_entropy': -1.0,
        # [warmup_epochs, ramp_epochs, min_rollout_len, max_rollout_len]
        'rollout_schedule': [30, 220, 2, 5],
        'save_every_epochs': 5,
        'early_stop_patience': None,
        'monitor_metric': 'evaluation/return-average',
        'q_loss_warning_threshold': None,
        'q_loss_stop_threshold': None,

        # ~100 random episode days before SAC (39 steps each).
        'n_initial_exploration_steps': 3900,
    },
    'environment_kwargs': {
        'start_date': '2020-01-01',
        'end_date': '2020-12-31',
        'tz': 'UTC',
        'start_time': '13:30',
        'periods': 40,
        'freq': '15min',
        'randomize_day': True,
        # Match eval; same init for learned + baselines.
        'randomize_initial_orientation': False,
        'weather_source': 'random',
        'temperature': 23.0,
        'wind_speed': 2.0,
        # Was 0.0001 — sun pays movement every step; lower penalty allows competitive motion
        # while keeping reward = energy_kwh - movement_cost for ALL methods (fair compare).
        'movement_penalty': 0.00005,
        'observation_mode': 'physical',
    },
}
