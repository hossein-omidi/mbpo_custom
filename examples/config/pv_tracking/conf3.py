"""conf3 — NSRDB strong (stage3 base + deep/wide nets, high real-data fraction).

Anchored to stage3_multiyear_nsrdb_scenario: rollout length 1, discount=1.0.
Variant vs 0.py skeleton: real_ratio=0.85, full 7/5 BNN ensemble, wider
dynamics model (hidden_dim=256), longer model-training budget, deeper
actor/critic (256×3), larger rollout batch — recommended production run.
"""

from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_conf3_nsrdb_strong_2026-06-12'
TRAINING_STAGE = 'conf3'

# Deeper actor + twin Q (0.py skeleton default is 2×256).
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
        # min_alpha/reward_scale from _nsrdb_base (0.001 / 100): run3 showed
        # min_alpha=0.05 makes the entropy bonus ~18x the energy signal.
    },
    policy_params_kwargs={
        'hidden_layer_sizes': _SAC_HIDDEN,
    },
    q_params_kwargs={
        'hidden_layer_sizes': _SAC_HIDDEN,
    },
)
