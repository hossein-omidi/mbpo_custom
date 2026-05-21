import numpy as np

# PVTracking-v0 observation layout (must match mbpo/env/pv_tracking.py).
# Indices of (sin, cos) pairs encoded as 2*pi * angle.
PV_CYCLIC_SLICES = (
    (1, 3),    # solar azimuth
    (8, 10),   # panel azimuth
    (11, 13),  # time of day
    (13, 15),  # day of year
)

# Default episode timing: 06:00 start, 64 x 15min timestamps, 63 actions.
DEFAULT_START_HOUR = 6.0
DEFAULT_NUM_ACTIONS = 63
DEFAULT_STEP_HOURS = 0.25
# Allow half a control step (7.5 min) below the computed episode end hour.
HORIZON_TIME_TOLERANCE_HOURS = DEFAULT_STEP_HOURS / 2.0


class StaticFns:
    PV_CYCLIC_SLICES = PV_CYCLIC_SLICES

    @staticmethod
    def episode_end_hour(
            start_hour=DEFAULT_START_HOUR,
            num_actions=DEFAULT_NUM_ACTIONS,
            step_hours=DEFAULT_STEP_HOURS):
        """Wall-clock hour of the final observation after the last action."""
        return start_hour + num_actions * step_hours

    @staticmethod
    def episode_end_angle(
            start_hour=DEFAULT_START_HOUR,
            num_actions=DEFAULT_NUM_ACTIONS,
            step_hours=DEFAULT_STEP_HOURS):
        """Time-of-day angle in obs (2*pi*hour/24) at episode end."""
        end_hour = StaticFns.episode_end_hour(start_hour, num_actions, step_hours)
        return 2.0 * np.pi * end_hour / 24.0

    @staticmethod
    def time_angle_from_obs(obs):
        """Raw atan2(sin, cos) for time-of-day indices (range (-pi, pi])."""
        obs = np.asarray(obs)
        if obs.ndim == 1:
            return np.arctan2(obs[11], obs[12])
        return np.arctan2(obs[:, 11], obs[:, 12])

    @staticmethod
    def time_of_day_from_obs(obs):
        """Decode time-of-day in hours [0, 24), matching PVTrackingEnv encoding."""
        angle = StaticFns.time_angle_from_obs(obs)
        angle = np.mod(angle, 2.0 * np.pi)
        hours = angle * 24.0 / (2.0 * np.pi)
        if np.ndim(hours) == 0:
            return float(hours)
        return hours

    @staticmethod
    def is_valid_rollout_start_obs(
            obs,
            margin_hours=HORIZON_TIME_TOLERANCE_HOURS,
            start_hour=DEFAULT_START_HOUR,
            num_actions=DEFAULT_NUM_ACTIONS,
            step_hours=DEFAULT_STEP_HOURS):
        """True for states that are not already at/past the real episode horizon."""
        time_of_day = StaticFns.time_of_day_from_obs(obs)
        end_hour = StaticFns.episode_end_hour(start_hour, num_actions, step_hours)
        cutoff = end_hour - margin_hours
        if np.ndim(time_of_day) == 0:
            return bool(time_of_day < cutoff)
        return time_of_day < cutoff

    @staticmethod
    def normalize_cyclic_pairs(obs):
        """Project each (sin, cos) block onto the unit circle."""
        obs = np.asarray(obs, dtype=np.float32)
        out = obs.copy()
        for start, end in PV_CYCLIC_SLICES:
            block = out[..., start:end]
            norms = np.linalg.norm(block, axis=-1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            out[..., start:end] = block / norms
        return out

    @staticmethod
    def postprocess_next_obs(next_obs, low=None, high=None):
        """Clip to bounds and renormalize cyclic features after a model step."""
        next_obs = np.asarray(next_obs, dtype=np.float32)
        if low is not None and high is not None:
            next_obs = np.clip(next_obs, low, high)
        return StaticFns.normalize_cyclic_pairs(next_obs)

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        finite = np.isfinite(next_obs).all(axis=-1)
        next_time = StaticFns.time_of_day_from_obs(next_obs)
        end_hour = StaticFns.episode_end_hour()
        at_horizon = next_time >= end_hour - HORIZON_TIME_TOLERANCE_HOURS

        done = (~finite) | at_horizon
        return done[:, None]
