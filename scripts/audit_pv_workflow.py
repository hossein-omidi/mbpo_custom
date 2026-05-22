#!/usr/bin/env python3
"""End-to-end smoke audit for PVTracking physical 11-D observations."""

from __future__ import print_function

import importlib.util
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import gym
import softlearning.environments.adapters.gym_adapter  # noqa: F401
from softlearning.environments.utils import get_environment_from_params

from mbpo.models.constructor import construct_model, format_samples_for_training
from mbpo.models.fake_env import FakeEnv
from mbpo.static.pv_tracking import (
    PHYSICAL_OBS_DIM,
    LEGACY_OBS_DIM,
    StaticFns,
    observation_mode_from_obs,
)
from mbpo.env.pv_tracking import PHYSICAL_OBS_LABELS


def load_config():
    path = os.path.join(_REPO_ROOT, 'examples', 'config', 'pv_tracking', '0.py')
    spec = importlib.util.spec_from_file_location('pv_cfg', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.params


def build_training_env(params):
    variant = {
        'environment_params': {
            'training': {
                'domain': 'PVTracking',
                'task': 'v0',
                'universe': 'gym',
                'kwargs': dict(params['environment_kwargs']),
            }
        }
    }
    return get_environment_from_params(variant['environment_params']['training'])


def audit_dynamics_model(env):
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    assert obs_dim == PHYSICAL_OBS_DIM

    model = construct_model(
        obs_dim=obs_dim, act_dim=act_dim, num_networks=2, num_elites=1)
    fake = FakeEnv(
        model,
        StaticFns,
        obs_low=env.observation_space.low,
        obs_high=env.observation_space.high,
    )

    obs = env.reset()
    act = np.zeros(act_dim, dtype=np.float32)
    samples = {
        'observations': obs[None],
        'actions': act[None],
        'next_observations': obs[None],
        'rewards': np.zeros((1, 1), dtype=np.float32),
    }
    inputs, outputs = format_samples_for_training(samples)
    assert inputs.shape == (1, obs_dim + act_dim)
    assert outputs.shape == (1, obs_dim + 1)

    assert fake._obs_low.shape[0] == obs_dim
    assert fake._obs_high.shape[0] == obs_dim
    # FakeEnv.step requires a trained BNN; shape wiring is verified above.


def main():
    params = load_config()
    env = build_training_env(params)
    obs = env.reset()
    assert params['environment_kwargs'].get('observation_mode') == 'physical'
    assert observation_mode_from_obs(obs) == 'physical'
    assert obs.shape == (PHYSICAL_OBS_DIM,)
    assert list(env.unwrapped.obs_feature_names) == list(PHYSICAL_OBS_LABELS)

    legacy = gym.make('PVTracking-v0', observation_mode='legacy')
    assert legacy.reset().shape == (LEGACY_OBS_DIM,)
    legacy.close()

    audit_dynamics_model(env)
    env.close()
    print('audit_pv_workflow: all checks passed (physical 11-D).')


if __name__ == '__main__':
    main()
