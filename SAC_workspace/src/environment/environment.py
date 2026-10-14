"""
environment.py — SAC environment wrapper using same pvlib+NSRDB simulator.

This module wraps the existing MBPO PVTrackingEnv (which uses pvlib for
solar geometry and NSRDB for weather data) to provide a clean SAC interface.
The MDP settings (state space, action space, reward, dynamics) are identical.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional

# Import MBPO's environment directly (same simulator, same physics)
from mbpo.env.pv_tracking import PVTrackingEnv, DEFAULT_START_TIME, DEFAULT_PERIODS, DEFAULT_FREQ, DEFAULT_EPISODE_STEPS
from mbpo.static.pv_tracking import COMPACT_OBS_DIM, COMPACT_OBS_LABELS

logger = logging.getLogger(__name__)


class SACEnvironment:
    """
    SAC environment wrapper using the exact same pvlib+NSRDB simulator as MBPO.

    Key properties (identical to MBPO conf_sac / conf_compact):
    - State space: 6D compact [DNI/2000, DHI/2000, P/(2000·A·η), β/90°, (γ/180-1), (H-t)/H]
    - Action space: 2D continuous [delta_tilt, delta_azimuth] with max bounds
    - Episode horizon: 117 steps (5-minute intervals, 12:00-21:45 UTC)
    - Reward: Same as MBPO (power - movement_penalty)
    - Simulator: Same pvlib Location + NSRDB weather scenarios
    """

    def __init__(self, observation_mode: str = 'compact', seed: int = None,
                 **env_kwargs):
        # Initialize with same parameters as MBPO conf_sac/conf_compact
        default_kwargs = {
            'latitude': 40.72,
            'longitude': -74.01,
            'altitude': 12.0,
            'tz': 'UTC',
            'start_date': '2020-01-01',
            'end_date': '2020-12-31',
            'start_time': DEFAULT_START_TIME,
            'periods': DEFAULT_PERIODS,
            'freq': DEFAULT_FREQ,
            'efficiency': 0.18,
            'area': 1.0,
            'max_delta_tilt': 5.0,
            'max_delta_azimuth': 10.0,
            'randomize_day': True,
            'randomize_initial_orientation': True,
            'weather_source': 'clearsky',
            'randomize_scenario': False,
            'movement_penalty': 0.0,  # Same as SAC baseline
            'irradiance_perturbation_std': 0.0,
            'observation_noise_std': 0.0,
            'observation_mode': observation_mode,
        }
        # Merge user kwargs with defaults (user takes precedence)
        for k, v in env_kwargs.items():
            default_kwargs[k] = v

        self.env = PVTrackingEnv(**default_kwargs)
        self.observation_mode = observation_mode
        self.state_dim = COMPACT_OBS_DIM if observation_mode == 'compact' else 16
        self.action_dim = 2  # [delta_tilt, delta_azimuth]
        self.max_steps = DEFAULT_EPISODE_STEPS  # 117

    def reset(self) -> np.ndarray:
        """Reset environment and return initial 6D compact observation."""
        obs = self.env.reset()
        return self._normalize_obs(obs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action and return (next_obs, reward, done, info) with same MDP."""
        # Action is already in environment format
        next_obs_raw, reward, done, info = self.env.step(action)
        next_obs = self._normalize_obs(next_obs_raw)
        return next_obs, float(reward), bool(done), info

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation to 6D compact format."""
        if self.observation_mode == 'compact':
            # Should already be 6D
            return np.clip(np.array(obs, dtype=np.float32), -1.0, 5.0)[:6]
        else:
            # Convert to compact (same logic as representation module)
            from SAC_workspace.src.representation.representation import StateEncoder
            encoder = StateEncoder('compact')
            return encoder.encode(obs, self.observation_mode)

    def get_observation_space_info(self) -> Dict:
        """Get observation space details matching MBPO."""
        return {
            'state_dim': self.state_dim,
            'observation_mode': self.observation_mode,
            'feature_names': list(COMPACT_OBS_LABELS),
            'bounds_low': np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 0.0]),  # Approximate
            'bounds_high': np.array([5.0, 5.0, 5.0, 2.0, 2.0, 2.0]),  # Approximate
        }

    def get_action_space_info(self) -> Dict:
        """Get action space details matching MBPO."""
        return {
            'action_dim': self.action_dim,
            'action_bounds_low': np.array([-5.0, -10.0]),  # max_delta_tilt, max_delta_azimuth
            'action_bounds_high': np.array([5.0, 10.0]),
            'description': 'delta_tilt (deg/step), delta_azimuth (deg/step)',
        }

    def get_episode_length(self) -> int:
        """Get episode horizon length."""
        return self.max_steps

    def close(self):
        """Close environment."""
        pass


# Convenience class that directly uses the MBPO environment with minimal wrapper
class DirectPVEnv(SACEnvironment):
    """Direct wrapper with zero overhead — uses same env instance."""
    pass
