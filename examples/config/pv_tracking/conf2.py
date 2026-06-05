"""conf2 — NSRDB slow training (long horizon, conservative MBPO).

Low real_ratio, longer exploration, longer model rollouts — stable but slow.
Inspired by conservative MBPO schedules (cf. examples/config/ant/0.py).
"""

from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_conf2_nsrdb_slow_2026-06-06'
TRAINING_STAGE = 'conf2'

params = build_nsrdb_params(
    CONFIG_VERSION,
    TRAINING_STAGE,
    algo_kwargs={
        'n_epochs': 3000,
        'n_initial_exploration_steps': 15000,
        'real_ratio': 0.10,
        'discount': 0.995,
        'max_model_rollout_length': 5,
        'rollout_schedule': [30, 400, 1, 5],
        'n_train_repeat': 15,
        'rollout_batch_size': 400,
        'num_networks': 7,
        'num_elites': 5,
        'model_retain_epochs': 5,
        'save_every_epochs': 20,
    },
)
