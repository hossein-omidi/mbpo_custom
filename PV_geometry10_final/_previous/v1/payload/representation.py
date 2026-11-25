"""Managed geometry10 observation encoder; physical values already normalized."""
from __future__ import annotations
import numpy as np
from mbpo.geometry10 import STATE_DIM, FEATURE_NAMES, LOW, HIGH, TAU, PANEL_AZ, check_batch, azimuth


def encode(observation):
    x=check_batch(observation)
    if x.shape!=(STATE_DIM,):
        raise ValueError(f'Expected one geometry10 observation, got {x.shape}')
    return np.ascontiguousarray(x)


def decode(observation):
    s=encode(observation)
    return {'dni_wm2':float(s[0]*2000.),'dhi_wm2':float(s[1]*2000.),
            'power_w':float(s[2]*2000.),'panel_tilt_deg':float(s[3]*90.),
            'panel_azimuth_deg':float(azimuth(s[PANEL_AZ])),
            'solar_zenith_deg':float(s[6]*180.),'solar_azimuth_deg':float(azimuth(s[7:9])),
            'remaining_fraction':float(s[TAU])}


class StateEncoder:
    state_dim=STATE_DIM
    feature_names=FEATURE_NAMES
    def encode(self,observation):return encode(observation)
    def decode(self,observation):return decode(observation)
