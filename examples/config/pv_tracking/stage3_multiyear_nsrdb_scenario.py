"""Stage 3 — NSRDB multi-year empirical weather (main reference config).

weather_scenario_mode = nsrdb_multiyear
  e = (year, month, day) ~ Uniform(manifest), fixed W_e per episode.

Data: data/pv_weather/nsrdb/nsrdb_{2018..2024}_utc_5min.csv
Timing: native 5-min control (118 timestamps, 117 transitions).

Train:
  ./train.sh run_nsrdb stage3_nsrdb --verify

Eval:
  ./result.sh run_nsrdb --full
"""

from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_stage3_nsrdb_multiyear_2026-06-06'
TRAINING_STAGE = 'stage3_nsrdb'

params = build_nsrdb_params(
    CONFIG_VERSION,
    TRAINING_STAGE,
    algo_kwargs={
        'n_epochs': 2000,
        'n_initial_exploration_steps': 8000,
        'real_ratio': 0.75,
        'discount': 1.0,
        'max_model_rollout_length': 1,
        'rollout_schedule': [30, 400, 1, 1],
        'n_train_repeat': 2,
    },
)
