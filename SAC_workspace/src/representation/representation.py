"""The six physical observation features are ALREADY normalized by PVTrackingEnv.

Never apply a second normalization and never substitute GHI for measured power.
"""
from __future__ import annotations
import numpy as np

FEATURE_NAMES = ('dni_norm', 'dhi_norm', 'power_norm', 'panel_tilt_norm', 'panel_azimuth_norm', 'tau')
LOW = np.array([0., 0., 0., 0., -1., 0.], dtype=np.float32)
HIGH = np.ones(6, dtype=np.float32)


def encode(observation) -> np.ndarray:
    state = np.asarray(observation, dtype=np.float32)
    if state.shape != (6,):
        raise ValueError(f'Expected exactly six compact features, got shape {state.shape}')
    if not np.isfinite(state).all():
        raise ValueError('Non-finite compact observation')
    # The reference environment clips its own irradiance/power; do not hide drift.
    if np.any(state < LOW - 1e-5) or np.any(state > HIGH + 1e-5):
        raise ValueError('Compact observation outside reference environment bounds')
    return np.ascontiguousarray(state)


def decode(observation) -> dict:
    s = encode(observation)
    return {'dni_wm2': float(s[0]*2000.), 'dhi_wm2': float(s[1]*2000.),
            'power_w': float(s[2]*2000.), 'panel_tilt_deg': float(s[3]*90.),
            'panel_azimuth_deg': float(((s[4]+1.)*180.) % 360.),
            'remaining_fraction': float(s[5])}


class StateEncoder:
    """Compatibility facade; compact observations are encoded at the source."""
    state_dim = 6
    feature_names = FEATURE_NAMES
    def encode(self, observation):
        return encode(observation)
    def decode(self, observation):
        return decode(observation)
