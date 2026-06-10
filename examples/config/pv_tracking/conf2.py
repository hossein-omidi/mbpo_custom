"""conf2 — NSRDB slow (stage3 base + longer run, softer MBPO ensemble).

Anchored to stage3_multiyear_nsrdb_scenario: discount=1.0, rollout length 1.
Variant vs stage3: more epochs, longer exploration, lower real_ratio (more
model data), longer model retention, smaller BNN ensemble (from 0.py skeleton),
higher min_alpha, and movement_penalty on train+eval for fair sun-tracker compare.
"""

from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_conf2_nsrdb_slow_2026-06-10'
TRAINING_STAGE = 'conf2'

# Shared across train / in-train eval / result.sh MC eval and baselines.
_MOVEMENT_PENALTY = 0.0021

params = build_nsrdb_params(
    CONFIG_VERSION,
    TRAINING_STAGE,
    algo_kwargs={
        'n_epochs': 4000,
        'n_initial_exploration_steps': 12000,
        'real_ratio': 0.9,
        'discount': 1.0,
        'max_model_rollout_length': 1,
        'rollout_schedule': [30, 400, 1, 1],
        'n_train_repeat': 2,
        'model_retain_epochs': 7,
        'num_networks': 5,
        'num_elites': 3,
        'min_alpha': 0.01,
    },
    train_env_kwargs={'movement_penalty': _MOVEMENT_PENALTY},
    eval_env_kwargs={'movement_penalty': _MOVEMENT_PENALTY},
)
