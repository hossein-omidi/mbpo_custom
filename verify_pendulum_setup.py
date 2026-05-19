#!/usr/bin/env python
"""Quick verification script for Pendulum configuration."""

import sys
import gym

# Test 1: Can we import gym and create a Pendulum environment?
try:
    env = gym.make('Pendulum-v0')
    print("✓ Successfully created Pendulum-v0 environment")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    env.close()
except Exception as e:
    print(f"✗ Failed to create Pendulum-v0: {e}")
    sys.exit(1)

# Test 2: Can we import and access the static functions?
try:
    import mbpo.static
    static_fns = mbpo.static['pendulum']
    print("✓ Successfully imported Pendulum static functions")
    print(f"  StaticFns class: {static_fns}")
except Exception as e:
    print(f"✗ Failed to import Pendulum static functions: {e}")
    sys.exit(1)

# Test 3: Test the termination function
try:
    import numpy as np
    obs = np.random.randn(5, 3)  # batch_size=5, obs_dim=3 (cos, sin, theta_dot)
    act = np.random.randn(5, 1)  # batch_size=5, action_dim=1
    next_obs = np.random.randn(5, 3)
    
    done = static_fns.termination_fn(obs, act, next_obs)
    print("✓ Successfully called termination_fn")
    print(f"  Output shape: {done.shape}")
    print(f"  Output dtype: {done.dtype}")
    assert done.shape == (5, 1), f"Expected shape (5, 1), got {done.shape}"
    assert done.dtype == bool or done.dtype == np.bool_, f"Expected bool dtype, got {done.dtype}"
except Exception as e:
    print(f"✗ Failed termination function test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Verify config file
try:
    from examples.config.ant import params as ant_config
    from examples.config.inverted_pendulum import params as pendulum_config
    
    print("✓ Successfully imported config files")
    assert ant_config['domain'] == 'Pendulum', f"Expected domain 'Pendulum', got {ant_config['domain']}"
    assert ant_config['task'] == 'v0', f"Expected task 'v0', got {ant_config['task']}"
    assert pendulum_config['domain'] == 'Pendulum', f"Expected domain 'Pendulum', got {pendulum_config['domain']}"
    assert pendulum_config['task'] == 'v0', f"Expected task 'v0', got {pendulum_config['task']}"
    print("  Config domains and tasks correctly set to Pendulum-v0")
except Exception as e:
    print(f"✗ Failed config import test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All verification tests passed!")
