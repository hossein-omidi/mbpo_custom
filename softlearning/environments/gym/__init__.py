"""Custom Gym environments for the PV tracking project.

Only PVTracking-v0 is registered. MuJoCo, MultiGoal, and multiworld
registrations have been removed — this project targets PV solar tracking
and does not require those environments or their dependencies.
"""

from mbpo.env import register_mbpo_environments


GYM_ENVIRONMENTS = ()  # populated by register_environments()


def register_environments():
    """Register all custom environments (PVTracking only)."""
    registered_mbpo_environments = register_mbpo_environments()
    return registered_mbpo_environments
