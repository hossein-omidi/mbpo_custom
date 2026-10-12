"""conf_sac — SAC-only baseline for NSRDB multi-year (fair comparison with MBPO).

Pure SAC baseline without model-based rollouts for fair comparison with MBPO.
Same finite-horizon MDP, same 5-component state space, same reward function.

Use for sample complexity comparison: SAC vs MBPO on identical tasks.
"""

from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_conf_sac_nsrdb_2026'
TRAINING_STAGE = 'conf_sac'

# SAC-only: same network sizes as MBPO for fair comparison
_SAC_HIDDEN = (64, 64)

params = build_nsrdb_params(
    CONFIG_VERSION,
    TRAINING_STAGE,
    algo_kwargs={
        'n_epochs': 2000,
        'n_initial_exploration_steps': 8000,
        'real_ratio': 1.0,  # Pure SAC: no model rollouts
        'discount': 1.0,
        'n_train_repeat': 2,
        'model_retain_epochs': 1,  # Not used for SAC
        'max_model_t': 60,
        'num_networks': 1,  # Not used for SAC
        'num_elites': 1,
    },
    policy_params_kwargs={
        'hidden_layer_sizes': (64, 64),
    },
    q_params_kwargs={
        'hidden_layer_sizes': (64, 64),
    },
    train_env_kwargs={
        'observation_mode': 'compact',
    },
    eval_env_kwargs={
        'observation_mode': 'compact',
        'fixed_eval_scenarios': None,  # For MC eval: use full manifest, not fixed scenarios
        'randomize_scenario': True,  # For MC eval: randomize scenarios
    },
)

# Override algorithm type to SAC (not MBPO)
params['kwargs']['type'] = 'SAC'