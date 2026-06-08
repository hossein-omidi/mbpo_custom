"""conf4 — NSRDB noisy (stage3 base + mild observation/irradiance augmentation).

Anchored to stage3_multiyear_nsrdb_scenario algo schedule; eval stays noise-free.
Variant: bounded train-time noise with extra epochs for robustness.
"""

from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_conf4_nsrdb_noisy_2026-06-07'
TRAINING_STAGE = 'conf4'

IRRADIANCE_PERTURBATION_STD = 0.03
OBSERVATION_NOISE_STD = 0.01

params = build_nsrdb_params(
    CONFIG_VERSION,
    TRAINING_STAGE,
    algo_kwargs={
        'n_epochs': 3000,
        'n_initial_exploration_steps': 10000,
        'real_ratio': 0.75,
        'discount': 1.0,
        'max_model_rollout_length': 1,
        'rollout_schedule': [30, 400, 1, 1],
        'n_train_repeat': 3,
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
