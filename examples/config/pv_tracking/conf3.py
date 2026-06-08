"""conf3 — NSRDB strong (stage3 base + higher real-data fraction).

Anchored to stage3_multiyear_nsrdb_scenario: rollout length 1, discount=1.0.
Variant: real_ratio=0.85 and moderate SAC repeats for a data-heavy schedule.
"""

from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_conf3_nsrdb_strong_2026-06-07'
TRAINING_STAGE = 'conf3'

params = build_nsrdb_params(
    CONFIG_VERSION,
    TRAINING_STAGE,
    algo_kwargs={
        'n_epochs': 3000,
        'n_initial_exploration_steps': 10000,
        'real_ratio': 0.85,
        'discount': 1.0,
        'max_model_rollout_length': 1,
        'rollout_schedule': [30, 400, 1, 1],
        'n_train_repeat': 5,
    },
)
