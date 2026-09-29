"""MBPO environment registration — PVTracking only."""

import gym

MBPO_ENVIRONMENT_SPECS = (
    {
        'id': 'PVTracking-v0',
        'entry_point': 'mbpo.env.pv_tracking:PVTrackingEnv',
    },
)


def register_mbpo_environments():
    """Register PVTracking-v0 with gym."""
    for spec in MBPO_ENVIRONMENT_SPECS:
        gym.register(**spec)
    return tuple(s['id'] for s in MBPO_ENVIRONMENT_SPECS)
