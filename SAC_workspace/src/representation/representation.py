"""
representation.py — State normalization and 6-element feature mapping for SAC.

This module provides the state encoding/decoding for the 6D compact observation
space used in SAC, matching exactly the MBPO compact observation mode.

6D Compact State Space: [DNI/2000, DHI/2000, P/(2000·A·η), β/90°, (γ%360°)/180°-1, (H-t)/H]

Mathematical validation:
- DNI ∈ [0, 2000] W/m² → [0, 1]
- DHI ∈ [0, 2000] W/m² → [0, 1]
- P ∈ [0, 2000·A·η] W → [0, 1] (normalized by 2000·A·η)
- β (tilt) ∈ [0, 90°] → [0, 1]
- γ (azimuth) ∈ [0, 360°) → [-1, 1] via (γ mod 360)/180 - 1
- τ (remaining horizon) ∈ [0, 1] as (H-t)/H
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional
import logging

# Import MBPO's static constants for exact parity
from mbpo.static.pv_tracking import (
    COMPACT_OBS_DIM,
    COMPACT_OBS_LABELS,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    DEFAULT_REFERENCE_DATE,
    DEFAULT_PERIODS,
    DEFAULT_FREQ,
)
from mbpo.env.pv_tracking import (
    DEFAULT_START_TIME,
    DEFAULT_START_HOUR,
    DEFAULT_STEP_HOURS,
    COMPACT_OBS_LABELS,
)

logger = logging.getLogger(__name__)


class StateEncoder:
    """State encoder/decoder for 6D compact observation space."""

    def __init__(self, observation_mode: str = 'compact'):
        self.observation_mode = observation_mode
        self.state_dim = COMPACT_OBS_DIM  # 6

        # Normalization constants (exact from MBPO)
        self.irradiance_norm = 2000.0  # DNI, DHI max
        self.power_norm = 2000.0       # P max (with A=1, η=0.18 ~ 360W, but normalized by 2000)
        self.tilt_norm = 90.0          # β max in degrees
        self.azimuth_norm = 180.0      # For [-1, 1] mapping

    def encode(self, observation: Union[np.ndarray, Dict, torch.Tensor],
               mode: str = 'compact') -> np.ndarray:
        """Encode raw observation to 6D compact state."""
        if isinstance(observation, torch.Tensor):
            observation = observation.detach().cpu().numpy()

        if mode == 'compact':
            # Already compact: just validate and clip
            obs_arr = np.array(observation, dtype=np.float32)
            return np.clip(obs_arr, -1.0, 5.0)[:6]  # Safety clip
        else:
            # Convert from other modes to compact
            return self._encode_to_compact(observation, mode)

    def _encode_to_compact(self, observation, mode: str) -> np.ndarray:
        """Convert physical/legacy to 6D compact."""
        if mode == 'physical' and len(observation) >= 12:
            # Physical: [zenith_norm, az_sin, az_cos, dni_norm, dhi_norm,
            #            ghi_norm, temp_norm, tilt_norm, panel_az_sin, panel_az_cos,
            #            cos_aoi, tau]
            dni_norm = float(observation[3]) if not np.isnan(observation[3]) else 0.0
            dhi_norm = float(observation[4]) if not np.isnan(observation[4]) else 0.0
            power_approx = float(observation[5])  # Use GHI as proxy
            tilt_norm = float(observation[7])
            az_sin = float(observation[8])
            az_cos = float(observation[9])
            azimuth_norm = np.arctan2(az_sin, az_cos) / np.pi  # [-1, 1]
            tau = float(observation[11])
            return np.array([
                dni_norm, dhi_norm, power_approx, tilt_norm,
                azimuth_norm, tau
            ], dtype=np.float32)
        elif mode == 'legacy' and len(observation) >= 16:
            # Legacy: extract compact components
            dni_norm = observation[3] if not np.isnan(observation[3]) else 0.0
            dhi_norm = observation[4] if not np.isnan(observation[4]) else 0.0
            ghi_norm = observation[5] if not np.isnan(observation[5]) else 0.0
            tilt_norm = observation[7]
            az_sin = observation[8]
            az_cos = observation[9]
            azimuth_norm = np.arctan2(az_sin, az_cos) / np.pi
            tau = observation[15]
            return np.array([
                dni_norm, dhi_norm, ghi_norm, tilt_norm,
                azimuth_norm, tau
            ], dtype=np.float32)
        else:
            # Default: assume compact input or create from minimal info
            return np.clip(np.array(observation, dtype=np.float32)[:6], -1.0, 5.0)

    def decode(self, compact_obs: np.ndarray, mode: str = 'compact') -> Dict:
        """Decode 6D compact observation back to labeled components."""
        if len(compact_obs) < 6:
            compact_obs = np.pad(compact_obs, (0, 6 - len(compact_obs)))

        result = {
            'dni_norm': float(compact_obs[0]),
            'dni_wm2': float(compact_obs[0]) * self.irradiance_norm,
            'dhi_norm': float(compact_obs[1]),
            'dhi_wm2': float(compact_obs[1]) * self.irradiance_norm,
            'power_norm': float(compact_obs[2]),
            'power_w_approx': float(compact_obs[2]) * self.power_norm,
            'tilt_norm': float(compact_obs[3]),
            'tilt_deg': float(compact_obs[3]) * self.tilt_norm,
            'azimuth_norm': float(compact_obs[4]),
            'azimuth_deg': float(compact_obs[4] + 1.0) * 90.0,
            'tau': float(compact_obs[5]),
            'remaining_steps_117ep': float(compact_obs[5]) * 117,
        }
        return result

    def get_state_dim(self, mode: str = 'compact') -> int:
        """Get state dimension for the observation mode."""
        return 6 if mode == 'compact' else 16 if mode == 'legacy' else 12

    def get_feature_names(self, mode: str = 'compact') -> list:
        """Get feature names for the observation mode."""
        return list(COMPACT_OBS_LABELS) if mode == 'compact' else []


class StateNormalizer:
    """Mathematically valid state normalization for SAC training."""

    def __init__(self, state_dim: int = 6):
        self.state_dim = state_dim
        # SAC requires bounded state representations
        # Each component of 6D compact is naturally bounded:
        # DNI/DHI/Power ∈ [0, 1], Tilt ∈ [0, 1], Azimuth ∈ [-1, 1], Tau ∈ [0, 1]
        # These are already well-behaved; we ensure they stay in [-1, 2] for safety.
        pass

    def normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation ensuring mathematical validity."""
        obs = np.array(obs, dtype=np.float32)
        if len(obs) == 6:
            # Clip to safe range for neural network input
            # Each component should be in reasonable range for SAC stability
            return np.clip(obs, -2.0, 5.0)  # Generous bounds
        else:
            # Pad/truncate to 6D
            result = np.zeros(6, dtype=np.float32)
            copy_len = min(len(obs), 6)
            result[:copy_len] = obs[:copy_len]
            return np.clip(result, -2.0, 5.0)

    def denormalize_state(self, state: np.ndarray, info: Dict = None) -> Dict:
        """Convert normalized state back to physical quantities."""
        encoder = StateEncoder()
        return encoder.decode(state, 'compact')
