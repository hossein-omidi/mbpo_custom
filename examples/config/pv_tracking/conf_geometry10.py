# PVWATT_DEV_REPAIR_V3
"""Geometry10 managed SAC/MBPO reference. Same physical/hyperparameter
configuration as conf_geometry10; ten observation features only. This config is
read by the managed Torch configuration loader, not the legacy Ray runner."""
from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_conf_geometry10_nsrdb_2026'
TRAINING_STAGE = 'conf_geometry10'

# Geometry10 state; hidden widths remain unchanged.
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
        'observation_mode': 'geometry10',
        'movement_penalty': 1.0,
        'movement_cost_mode': 'geometry',
    },
    eval_env_kwargs={
        'observation_mode': 'geometry10',
        'fixed_eval_scenarios': None,
        'randomize_scenario': True,
        'movement_penalty': 1.0,
        'movement_cost_mode': 'geometry',
    },
    run_params_kwargs={
        'checkpoint_replay_pool': True,
    },
    replay_pool_params_kwargs={
        'kwargs': {
            'max_size': 50000,
        }
    },
)
