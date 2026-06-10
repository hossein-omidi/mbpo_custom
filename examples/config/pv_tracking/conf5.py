"""conf5 — NSRDB advanced strong (A/B vs conf3: wider/deeper nets + model budget).

Anchored to stage3: rollout length 1, discount=1.0, eval noise-free.
Variant: high real_ratio, longer BNN training budget, full 7/5 ensemble,
wider dynamics model (hidden_dim=256), deeper actor/critic (256×3).
"""

from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_conf5_nsrdb_advanced_2026-06-10'
TRAINING_STAGE = 'conf5'

# Deeper actor + twin Q (default base uses 2×256).
_SAC_HIDDEN = (256, 256, 256)

params = build_nsrdb_params(
    CONFIG_VERSION,
    TRAINING_STAGE,
    algo_kwargs={
        'n_epochs': 3500,
        'n_initial_exploration_steps': 12000,
        'real_ratio': 0.85,
        'discount': 1.0,
        'max_model_rollout_length': 1,
        'rollout_schedule': [30, 400, 1, 1],
        'n_train_repeat': 5,
        'rollout_batch_size': 500,
        'model_retain_epochs': 8,
        'max_model_t': 180,
        'hidden_dim': 256,
        'num_networks': 7,
        'num_elites': 5,
        'min_alpha': 0.05,
    },
    policy_params_kwargs={
        'hidden_layer_sizes': _SAC_HIDDEN,
    },
    q_params_kwargs={
        'hidden_layer_sizes': _SAC_HIDDEN,
    },
)
