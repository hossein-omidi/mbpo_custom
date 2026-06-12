"""conf4 — NSRDB light-fast (smallest nets, short run, quick epochs).

Anchored to stage3: rollout length 1, discount=1.0, reward_scale/min_alpha from
_nsrdb_base (100 / 0.001).

Use for fast iteration / smoke runs before a full conf1 or conf2 training.
  - **Tiny nets**: BNN hidden_dim=64, 3/2 ensemble; SAC 64×1 (single hidden layer).
  - **Fixed pose**: 30° tilt, 180° azimuth (south) — no orientation RNG.
  - **Short budget**: 1500 epochs, 5k exploration, max_model_t=60 s.
  - **Light MBPO**: rollout_batch_size=250, model_retain_epochs=4,
    n_train_repeat=4 (fewer SAC passes → faster wall-clock per epoch).
  - **Mostly real data**: real_ratio=0.95 (minimal model-rollout work).
"""

from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_conf4_nsrdb_light_fast_2026-06-12'
TRAINING_STAGE = 'conf4'

# Single hidden layer — minimum viable SAC for 11-D / 2-D.
_SAC_HIDDEN = (64,)

_FIXED_ORIENTATION_KW = {
    'randomize_initial_orientation': False,
}

params = build_nsrdb_params(
    CONFIG_VERSION,
    TRAINING_STAGE,
    algo_kwargs={
        'n_epochs': 1500,
        'n_initial_exploration_steps': 5000,
        'real_ratio': 0.95,
        'discount': 1.0,
        'max_model_rollout_length': 1,
        'rollout_schedule': [20, 250, 1, 1],
        'n_train_repeat': 4,
        'rollout_batch_size': 250,
        'model_retain_epochs': 4,
        'max_model_t': 60,
        'hidden_dim': 64,
        'num_networks': 3,
        'num_elites': 2,
    },
    train_env_kwargs=dict(_FIXED_ORIENTATION_KW),
    eval_env_kwargs=dict(_FIXED_ORIENTATION_KW),
    policy_params_kwargs={
        'hidden_layer_sizes': _SAC_HIDDEN,
    },
    q_params_kwargs={
        'hidden_layer_sizes': _SAC_HIDDEN,
    },
)
