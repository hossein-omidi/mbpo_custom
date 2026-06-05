"""conf3 — NSRDB strong training (stable MBPO, balanced model/real mix).

Recommended for production runs: moderate real_ratio, 5-step rollouts, 15× SAC repeats.
"""

from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_conf3_nsrdb_strong_2026-06-06'
TRAINING_STAGE = 'conf3'

params = build_nsrdb_params(
    CONFIG_VERSION,
    TRAINING_STAGE,
    algo_kwargs={
        'n_epochs': 2000,
        'n_initial_exploration_steps': 12000,
        'real_ratio': 0.10,
        'discount': 0.995,
        'max_model_rollout_length': 5,
        'rollout_schedule': [30, 400, 1, 5],
        'n_train_repeat': 15,
        'rollout_batch_size': 400,
        'num_networks': 7,
        'num_elites': 5,
        'min_alpha': 0.01,
        'target_entropy': 'auto',
    },
)
