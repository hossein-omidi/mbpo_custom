"""conf2 — NSRDB slow finite-horizon (stage3 base + longer run, softer MBPO).

Finite-horizon MDP (native to this fork, no flags needed):
  M_H = (S x T, A, P, R, H, gamma), H = 117 (118 x 5-min, 12:00-21:45 UTC),
  gamma = 1 (undiscounted; bounded return G = sum_t r_t <= H * r_max).
  Augmented state x_t = (s_t, tau_t), tau = remaining_steps / H in [0, 1]
  (12-D physical obs). tau evolves deterministically (tau' = tau - 1/H),
  never learned: FakeEnv overrides the model's tau output; SAC targets use
  Q(s', tau-1, a'); terminal d = 1[tau = 1/H]; rollout filter tau > k.

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

CONFIG_VERSION = 'pv_tracking_conf2_nsrdb_slow_fh_2026-09-29'
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
    run_params_kwargs={
        'checkpoint_replay_pool': True,
    },
    replay_pool_params_kwargs={
        'kwargs': {
            'max_size': 200000,  # capped at 200k for 8GB node
        }
    },
)