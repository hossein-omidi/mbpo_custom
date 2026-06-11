"""conf1 — NSRDB fast (stage3 base + moderate SAC repeats).

Anchored to stage3_multiyear_nsrdb_scenario: real_ratio=0.75, rollout length 1.
Variant: n_train_repeat=5 for faster policy improvement per epoch.
"""

from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_conf1_nsrdb_fast_2026-06-07'
TRAINING_STAGE = 'conf1'

params = build_nsrdb_params(
    CONFIG_VERSION,
    TRAINING_STAGE,
    algo_kwargs={
        'n_epochs': 2500,
        'n_initial_exploration_steps': 8000,
        'real_ratio': 0.85,
        'discount': 1.0,
        'max_model_rollout_length': 1,
        'rollout_schedule': [30, 400, 1, 1],
        'n_train_repeat': 5,
    },
)
