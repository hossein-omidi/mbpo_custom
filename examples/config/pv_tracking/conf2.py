"""conf2 — NSRDB slow (stage3 base + longer run and exploration).

Anchored to stage3_multiyear_nsrdb_scenario: real_ratio=0.75, rollout length 1.
Variant: more epochs, longer initial exploration, longer model retention.
"""

from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_conf2_nsrdb_slow_2026-06-07'
TRAINING_STAGE = 'conf2'

params = build_nsrdb_params(
    CONFIG_VERSION,
    TRAINING_STAGE,
    algo_kwargs={
        'n_epochs': 4000,
        'n_initial_exploration_steps': 12000,
        'real_ratio': 0.75,
        'discount': 1.0,
        'max_model_rollout_length': 1,
        'rollout_schedule': [30, 400, 1, 1],
        'n_train_repeat': 2,
        'model_retain_epochs': 7,
    },
)
