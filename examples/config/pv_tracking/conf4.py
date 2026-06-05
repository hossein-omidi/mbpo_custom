"""conf4 — NSRDB noisy training (observation + irradiance augmentation).

Same NSRDB scenario MDP with bounded noise for robustness experiments.
"""

from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_conf4_nsrdb_noisy_2026-06-06'
TRAINING_STAGE = 'conf4'

IRRADIANCE_PERTURBATION_STD = 0.05
OBSERVATION_NOISE_STD = 0.02

params = build_nsrdb_params(
    CONFIG_VERSION,
    TRAINING_STAGE,
    algo_kwargs={
        'n_epochs': 2000,
        'n_initial_exploration_steps': 10000,
        'real_ratio': 0.30,
        'discount': 0.995,
        'max_model_rollout_length': 3,
        'rollout_schedule': [30, 400, 1, 3],
        'n_train_repeat': 10,
    },
    train_env_kwargs={
        'irradiance_perturbation_std': IRRADIANCE_PERTURBATION_STD,
        'observation_noise_std': OBSERVATION_NOISE_STD,
    },
    eval_env_kwargs={
        'irradiance_perturbation_std': 0.0,
        'observation_noise_std': 0.0,
    },
)
