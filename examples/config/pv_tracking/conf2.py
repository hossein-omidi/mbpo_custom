"""conf2 — NSRDB slow (stage3 base + longer run, softer MBPO ensemble).

Anchored to stage3_multiyear_nsrdb_scenario: discount=1.0, rollout length 1.
Variant vs stage3: more epochs, longer exploration, lower real_ratio (more
model data), longer model retention, smaller BNN ensemble (from 0.py skeleton),
higher min_alpha, and geometry-based movement cost on train+eval for fair
sun-tracker compare.

Movement cost (movement_cost_mode='geometry'):
  E_move [kWh] = movement_penalty * P_motor * (|Δtilt|/ω_tilt + |Δaz|/ω_az) / 3.6e6
  Defaults: P_motor=30 W, ω_tilt=1.5°/s, ω_az=2°/s (1 m² dual-axis tracker).
  At max step (5° + 10°) → ~0.07 Wh ≈ 1.2% of median 5-min harvest (NYC window).
  movement_penalty=1.0 is the unscaled physics estimate (not a tunable gain).
"""

from examples.config.pv_tracking._nsrdb_base import (
    NSRDB_MANIFEST,
    NSRDB_VALIDATION_SCENARIO_IDS,
    assert_nsrdb_validation_scenarios,
    build_nsrdb_params,
)

assert_nsrdb_validation_scenarios(NSRDB_MANIFEST, NSRDB_VALIDATION_SCENARIO_IDS)

CONFIG_VERSION = 'pv_tracking_conf2_nsrdb_slow_2026-06-12'
TRAINING_STAGE = 'conf2'

# Physics-derived actuator cost; scale=1.0 (see module docstring).
_MOVEMENT_ENV_KWARGS = {
    'movement_cost_mode': 'geometry',
    'movement_penalty': 1.0,
}

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
        # min_alpha/reward_scale from _nsrdb_base (0.001 / 100): the old 0.01
        # floor kept the entropy bonus ~3x the per-step energy reward.
    },
    train_env_kwargs=dict(_MOVEMENT_ENV_KWARGS),
    eval_env_kwargs=dict(_MOVEMENT_ENV_KWARGS),
)
