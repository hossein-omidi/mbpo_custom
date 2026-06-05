"""conf1 — NSRDB fast training (shorter run, lighter MBPO updates).

Parallel NSRDB profile for quick iteration / hyperparameter search.
Same env and eval protocol as stage3_multiyear_nsrdb_scenario.
"""

from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_conf1_nsrdb_fast_2026-06-06'
TRAINING_STAGE = 'conf1'

params = build_nsrdb_params(
    CONFIG_VERSION,
    TRAINING_STAGE,
    algo_kwargs={
        'n_epochs': 400,
        'n_initial_exploration_steps': 4000,
        'real_ratio': 0.50,
        'discount': 1.0,
        'max_model_rollout_length': 1,
        'rollout_schedule': [20, 200, 1, 1],
        'n_train_repeat': 2,
        'rollout_batch_size': 200,
        'num_networks': 5,
        'num_elites': 3,
        'save_every_epochs': 10,
    },
)
