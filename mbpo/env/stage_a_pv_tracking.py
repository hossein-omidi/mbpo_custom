"""Narrow Stage-A extension of the *existing* PVTrackingEnv.

Uses original discrete-time kinematics, reward, observational POMDP interface,
NSRDB scenario sampler, and terminal/time conventions without modification.
The only physical change is using *recorded* per-timestep NSRDB albedo in the
canonical isotropic POA and PVWatts/SAPM backend. No on-line API, synthetic
weather, or additional observation/action channels are introduced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from mbpo.env.pv_tracking import PVTrackingEnv
from mbpo.env.pv_power_backend import compute_pv_power, PHYSICS_REVISION, DEFAULT_ALBEDO

ALBEDO_COLUMNS = ('surface_albedo', 'albedo', 'Surface Albedo', 'Surface_Albedo')


class StageAPVTrackingEnv(PVTrackingEnv):
    def __init__(self,*args,**kwargs):
        if (kwargs.get('weather_source')=='nsrdb_multiyear'
                or kwargs.get('weather_scenario_mode')=='nsrdb_multiyear'):
            for key in ('irradiance_perturbation_std','observation_noise_std'):
                if float(kwargs.get(key,0.0))!=0.0:
                    raise ValueError(f'Stage A requires zero synthetic NSRDB augmentation: {key}')
        super().__init__(*args,**kwargs)

    def _surface_albedo_at_current_time(self):
        if self.weather_source != 'nsrdb_multiyear':
            return DEFAULT_ALBEDO, 'fixed_non_nsrdb'
        # Weather is read from the local NSRDB year CSV and exactly aligned to
        # the 5-minute clock by the parent's existing resample/weather methods.
        frame = self.weather_profile
        if not isinstance(frame, pd.DataFrame):
            raise ValueError('NSRDB weather profile is not a DataFrame')
        if self.current_time not in frame.index:
            raise ValueError(f'Missing NSRDB albedo row at {self.current_time}')
        row = frame.loc[self.current_time]
        if isinstance(row, pd.DataFrame):
            raise ValueError('Duplicate NSRDB time labels: cannot assign unique albedo')
        cols = [c for c in ALBEDO_COLUMNS if c in frame.columns]
        if not cols:
            raise ValueError('NSRDB Surface Albedo is absent from the loaded CSV. '
                             'Stage A refuses to silently substitute 0.25; inspect local year files.')
        if len(cols) > 1:
            values = np.array([float(row[c]) for c in cols])
            if not np.allclose(values, values[0], rtol=0, atol=1e-8):
                raise ValueError('Conflicting NSRDB albedo columns')
        value = float(row[cols[0]])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Invalid NSRDB albedo {value} at {self.current_time}')
        return value, 'nsrdb_recorded'

    def _power_from_orientation(self, solar_zenith, solar_azimuth, tilt, azimuth):
        weather = self._current_weather()
        albedo,source = self._surface_albedo_at_current_time()
        result = compute_pv_power(
            tilt, azimuth, solar_zenith, solar_azimuth,
            weather['dni'], weather['ghi'], weather['dhi'],
            area=self.area, efficiency=self.efficiency,
            power_model=self.power_model,
            temp_air_c=weather['temperature'] if self.power_model=='pvwatts_sapm' else None,
            wind_speed_ms=weather['wind_speed'] if self.power_model=='pvwatts_sapm' else None,
            wind_reference_height_m=self.wind_reference_height_m,
            gamma_pdc=self.gamma_pdc, sapm_mount=self.sapm_mount,
            wind_powerlaw_exponent=self.wind_powerlaw_exponent,
            surface_albedo=albedo)
        self._last_power_result = result
        self._stage_a_last_albedo = albedo
        self._stage_a_albedo_source = source
        return float(result.power_w)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # Crucially, provide *measured interval* albedo to the paired baseline
        # evaluator; no independent choice of albedo is allowed for the oracle.
        info = dict(info)
        info['surface_albedo'] = float(self._stage_a_last_albedo)
        info['albedo_source'] = self._stage_a_albedo_source
        info['physics_revision'] = PHYSICS_REVISION
        return obs, reward, done, info
