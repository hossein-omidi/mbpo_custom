"""conf1 — NSRDB simple-strong (compact nets, fixed pose, high real_ratio).

Anchored to stage3: rollout length 1, discount=1.0, reward_scale/min_alpha from
_nsrdb_base (100 / 0.001).

Design (11-D obs, 2-D act, 117-step days):
  - **Compact nets**: BNN hidden_dim=128, 5/3 ensemble; SAC actor+critic 128×2.
    Smaller than the 256-wide defaults — faster epochs, less critic noise, still
    ample capacity for this MDP.
  - **Fixed initial orientation**: randomize_initial_orientation=False → every
    reset starts at tilt=30°, azimuth=180° (south). Removes pose RNG so the
    policy learns weather tracking from one consistent mount reference.
  - **More SAC updates**: n_train_repeat=8 (vs 2 in stage3 base) for clearer
    gradients with reward_scale=100.
  - **High real data**: real_ratio=0.9 (best signal from run2 evidence).
"""

from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_conf1_nsrdb_simple_strong_2026-06-12'
TRAINING_STAGE = 'conf1'

# Compact actor + twin Q (framework default is 256×2).
_SAC_HIDDEN = (128, 128)

# Fixed mount at reset: 30° tilt, 180° azimuth (PVTrackingEnv defaults).
_FIXED_ORIENTATION_KW = {
    'randomize_initial_orientation': False,
}

params = build_nsrdb_params(
    CONFIG_VERSION,
    TRAINING_STAGE,
    algo_kwargs={
        'n_epochs': 4000,
        'n_initial_exploration_steps': 10000,
        'real_ratio': 0.9,
        'discount': 1.0,
        'max_model_rollout_length': 1,
        'rollout_schedule': [30, 400, 1, 1],
        'n_train_repeat': 8,
        'rollout_batch_size': 400,
        'model_retain_epochs': 6,
        'max_model_t': 120,
        'hidden_dim': 128,
        'num_networks': 5,
        'num_elites': 3,
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
