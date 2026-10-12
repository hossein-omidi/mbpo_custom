import numpy as np
import pandas as pd
from pvlib.location import Location

# Observation layouts (must match mbpo/env/pv_tracking.py).
# Compact 6-component state space (6 components including tau)
COMPACT_OBS_DIM = 6
LEGACY_OBS_DIM = 16
PHYSICAL_OBS_DIM = 12
LEGACY_CYCLIC_SLICES = (
    (1, 3),    # solar azimuth
    (8, 10),   # panel azimuth
    (11, 13),  # time of day
    (13, 15),  # day of year
)

PHYSICAL_CYCLIC_SLICES = (
    (1, 3),    # solar azimuth
    (8, 10),   # panel azimuth
)

COMPACT_CYCLIC_SLICES = ()  # compact 5D has no sin/cos pairs
# Match PVTrackingEnv defaults (used only for solar-time decoding in StaticFns).
DEFAULT_LATITUDE = 40.72
DEFAULT_LONGITUDE = -74.01
DEFAULT_TZ = 'UTC'
DEFAULT_REFERENCE_DATE = '2020-06-21'

# Default episode timing (must match mbpo/env/pv_tracking.py).
from mbpo.env.pv_tracking import (
    DEFAULT_START_TIME,
    DEFAULT_START_HOUR,
    DEFAULT_EPISODE_STEPS,
    DEFAULT_FREQ,
    DEFAULT_STEP_HOURS,
)

DEFAULT_NUM_ACTIONS = DEFAULT_EPISODE_STEPS
HORIZON_TIME_TOLERANCE_HOURS = DEFAULT_STEP_HOURS / 2.0


def observation_mode_from_obs(obs):
    """Infer layout from trailing observation dimension."""
    dim = int(np.asarray(obs).shape[-1])
    if dim == LEGACY_OBS_DIM:
        return 'legacy'
    if dim == PHYSICAL_OBS_DIM:
        return 'physical'
    if dim == COMPACT_OBS_DIM:
        return 'compact'
    raise ValueError(
        'Unsupported PV observation dimension {} (expected {} legacy, {} physical, or {} compact).'.format(
            dim, LEGACY_OBS_DIM, PHYSICAL_OBS_DIM, COMPACT_OBS_DIM))

def cyclic_slices_for_obs(obs):
    mode = observation_mode_from_obs(obs)
    if mode == 'physical':
        return PHYSICAL_CYCLIC_SLICES
    if mode == 'compact':
        return COMPACT_CYCLIC_SLICES
    return LEGACY_CYCLIC_SLICES

def _angular_error_deg(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


class StaticFns:
    PV_CYCLIC_SLICES = LEGACY_CYCLIC_SLICES

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
        """Time-of-day angle in legacy obs (2*pi*hour/24) at episode end."""
        end_hour = StaticFns.episode_end_hour(start_hour, num_actions, step_hours)
        return 2.0 * np.pi * end_hour / 24.0

    @staticmethod
    def time_angle_from_obs(obs):
        """Raw atan2(sin, cos) for time-of-day indices.
        
        Compact: time angle computed from tau (index 4) and episode timing
        Legacy: indices (11,12) for time sin/cos
        Physical: indices (1,2) for solar azimuth sin/cos, no time angle
        """
        obs = np.asarray(obs)
        mode = observation_mode_from_obs(obs)
        if mode == 'legacy':
            if obs.ndim == 1:
                return np.arctan2(obs[11], obs[12])
            return np.arctan2(obs[:, 11], obs[:, 12])
        if mode == 'physical':
            # Physical obs has no time angle encoded
            raise ValueError("Physical observation mode has no time angle")
        if mode == 'compact':
            # Compact 6D: tau at index 5 (0-indexed), compute time from remaining steps
            if obs.ndim == 1:
                tau = obs[5]
            else:
                tau = obs[:, 5]
            # tau = remaining_steps / H, so remaining_steps = tau * H
            # current_step = H - remaining_steps = H * (1 - tau)
            # time_of_day = start_hour + current_step * step_hours
            from mbpo.env.pv_tracking import DEFAULT_EPISODE_STEPS, DEFAULT_START_HOUR, DEFAULT_STEP_HOURS
            H = DEFAULT_EPISODE_STEPS
            current_step = H * (1.0 - tau)
            time_of_day = DEFAULT_START_HOUR + current_step * DEFAULT_STEP_HOURS
            time_angle = 2.0 * np.pi * time_of_day / 24.0
            return time_angle
        raise ValueError("Unknown observation mode")

    @staticmethod
    def time_of_day_from_obs(obs):
        """Time of day in hours (UTC) from observation.
        
        Compact: computed from tau
        Legacy: from time sin/cos at indices (11,12)
        Physical: raises ValueError (no time encoded)
        """
        angle = StaticFns.time_angle_from_obs(obs)
        return angle * 24.0 / (2.0 * np.pi)

    @staticmethod
    def time_of_day_from_solar_obs(
            obs,
            latitude=DEFAULT_LATITUDE,
            longitude=DEFAULT_LONGITUDE,
            tz=DEFAULT_TZ,
            start_hour=DEFAULT_START_HOUR,
            num_actions=DEFAULT_NUM_ACTIONS,
            step_hours=DEFAULT_STEP_HOURS,
            reference_date=DEFAULT_REFERENCE_DATE):
        """Reconstruct time from solar geometry (physical obs)."""
        single = obs.ndim == 1
        if single:
            obs = obs.reshape(1, -1)

        zenith = obs[:, 0] * 180.0
        azimuth = np.rad2deg(np.arctan2(obs[:, 1], obs[:, 2])) % 360.0

        end_hour = StaticFns.episode_end_hour(start_hour, num_actions, step_hours)
        hour_grid = np.arange(
            start_hour,
            end_hour + step_hours * 0.5,
            step_hours,
            dtype=np.float64,
        )

        location = Location(latitude, longitude, tz=tz)
        timestamps = []
        for hour in hour_grid:
            hour_int = int(hour)
            minute = int(round((hour - hour_int) * 60.0))
            timestamps.append(
                pd.Timestamp(
                    year=pd.Timestamp(reference_date).year,
                    month=pd.Timestamp(reference_date).month,
                    day=pd.Timestamp(reference_date).day,
                    hour=hour_int,
                    minute=minute,
                    tz=tz,
                )
            )
        solar_position = location.get_solarposition(pd.DatetimeIndex(timestamps))
        grid_zenith = solar_position['zenith'].values
        grid_azimuth = solar_position['azimuth'].values % 360.0

        hours = np.empty(obs.shape[0], dtype=np.float64)
        for i in range(obs.shape[0]):
            zenith_err = np.abs(grid_zenith - zenith[i])
            azimuth_err = _angular_error_deg(grid_azimuth, azimuth[i])
            score = zenith_err + 0.25 * azimuth_err
            hours[i] = hour_grid[int(np.argmin(score))]

        if single:
            return float(hours[0])
        return hours

    @staticmethod
    def is_valid_rollout_start_obs(
            obs,
            required_remaining_steps=0,
            margin_hours=HORIZON_TIME_TOLERANCE_HOURS,
            start_hour=DEFAULT_START_HOUR,
            num_actions=DEFAULT_NUM_ACTIONS,
            step_hours=DEFAULT_STEP_HOURS):
        """True when enough real env steps remain for a model rollout.

        `required_remaining_steps=0` preserves the old meaning: current obs is not
        already at/past the real episode horizon. For MBPO physical observations,
        this method is only approximate because time is reconstructed from solar
        geometry. Prefer exact replay metadata when available.
        """
        if observation_mode_from_obs(obs) == 'physical':
            time_of_day = StaticFns.time_of_day_from_solar_obs(obs)
        else:
            time_of_day = StaticFns.time_of_day_from_obs(obs)
        end_hour = StaticFns.episode_end_hour(start_hour, num_actions, step_hours)
        remaining = np.rint((end_hour - np.asarray(time_of_day)) / step_hours).astype(np.int64)
        remaining = np.clip(remaining, 0, int(num_actions))
        valid = remaining > int(required_remaining_steps)
        if np.ndim(valid) == 0:
            return bool(valid)
        return valid

    @staticmethod
    def normalize_cyclic_pairs(obs):
        """Project each (sin, cos) block onto the unit circle."""
        obs = np.asarray(obs, dtype=np.float32)
        out = obs.copy()
        for start, end in cyclic_slices_for_obs(obs):
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
        if observation_mode_from_obs(next_obs) == 'physical':
            # Physical obs has no clock: do not infer horizon from sun geometry
            # (reference-date matching can terminate ~0.5-1.5 h early).
            # MBPO caps imagined depth via rollout_length (see _rollout_model).
            at_horizon = np.zeros_like(finite, dtype=bool)
        else:
            next_time = StaticFns.time_of_day_from_obs(next_obs)
            end_hour = StaticFns.episode_end_hour()
            at_horizon = next_time >= end_hour - HORIZON_TIME_TOLERANCE_HOURS

        done = (~finite) | at_horizon
        return done[:, None]
