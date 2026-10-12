"""conf_compact — NSRDB multi-year with compact 5-component state space.

Finite-horizon MDP with 5-component state (zenith_norm, ghi_norm, rel_azimuth_sin, tilt_norm, tau).
Anchored to stage3_multiyear_nsrdb_scenario: discount=1.0, rollout length 1.

Key differences from conf3:
- Compact 5D state space (zenith_norm, ghi_norm, rel_azimuth_sin, tilt_norm, tau)
- Smaller networks (64-wide) for faster training
- Same physics-based reward and movement cost
"""

from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_conf_compact_nsrdb_2026'
TRAINING_STAGE = 'conf_compact'

# Compact networks for faster training with 5D state
_SAC_HIDDEN = (64, 64)

params = build_nsrdb_params(
    CONFIG_VERSION,
    TRAINING_STAGE,
    algo_kwargs={
        'n_epochs': 2000,
        'n_initial_exploration_steps': 8000,
        'real_ratio': 0.85,
        'discount': 1.0,
        'max_model_rollout_length': 1,
        'rollout_schedule': [30, 400, 1, 1],
        'n_train_repeat': 2,
        'model_retain_epochs': 4,
        'max_model_t': 60,
        'num_networks': 5,
        'num_elites': 3,
        'hidden_dim': 64,
    },
    policy_params_kwargs={
        'hidden_layer_sizes': (64, 64),
    },
    q_params_kwargs={
        'hidden_layer_sizes': (64, 64),
    },
    train_env_kwargs={
        'observation_mode': 'compact',
    },
    eval_env_kwargs={
        'observation_mode': 'compact',
        'fixed_eval_scenarios': None,  # For MC eval: use full manifest, not fixed scenarios
        'randomize_scenario': True,  # For MC eval: randomize scenarios
    },
    run_params_kwargs={
        'checkpoint_replay_pool': True,
    },
    replay_pool_params_kwargs={
        'kwargs': {
            'max_size': 50000,  # capped at 50k for 8GB node
        }
    },
)