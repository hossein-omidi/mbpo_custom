# POSTPROCESSING_TRAJECTORY_UNCERTAINTY_V1
"""Narrow Stage-A extension of the *existing* PVTrackingEnv.

Uses original discrete-time kinematics, reward, observational POMDP interface,
NSRDB scenario sampler, and terminal/time conventions without modification.
The only physical change is using *recorded* per-timestep NSRDB albedo in the
canonical isotropic POA and PVWatts/SAPM backend. Optional hypothesized
weather-error realizations are generated once per episode at reset; the
six-feature controller interface and physical power backend are unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from mbpo.env.pv_tracking import PVTrackingEnv
from mbpo.env.pv_power_backend import compute_pv_power, PHYSICS_REVISION, DEFAULT_ALBEDO
from mbpo.env.trajectory_uncertainty import (normalized_spec, realization_seed,
    spec_fingerprint, generate_weather_profile)
import hashlib
import json

ALBEDO_COLUMNS = ('surface_albedo', 'albedo', 'Surface Albedo', 'Surface_Albedo')


class StageAPVTrackingEnv(PVTrackingEnv):
    def __init__(self,*args,**kwargs):
        self.weather_uncertainty = normalized_spec(kwargs.pop('weather_uncertainty',None))
        self._requested_realization_id = None
        self._weather_realization_seed = None
        self._weather_realization_id = None
        self._weather_profile_sha256 = None
        if self.weather_uncertainty['enabled'] and not (
                kwargs.get('weather_source')=='nsrdb_multiyear' or
                kwargs.get('weather_scenario_mode')=='nsrdb_multiyear'):
            raise ValueError('Trajectory uncertainty requires historical NSRDB source')
        if (kwargs.get('weather_source')=='nsrdb_multiyear'
                or kwargs.get('weather_scenario_mode')=='nsrdb_multiyear'):
            for key in ('irradiance_perturbation_std','observation_noise_std'):
                if float(kwargs.get(key,0.0))!=0.0:
                    raise ValueError(f'Stage A requires zero synthetic NSRDB augmentation: {key}')
        super().__init__(*args,**kwargs)

    def set_weather_realization(self, realization_id=None):
        # Explicit identity for common-random-numbers evaluation. Training
        # passes None, drawing a new episode seed from the checkpointed RNG.
        if realization_id is not None and (
                isinstance(realization_id,bool) or
                not isinstance(realization_id,(int,np.integer)) or int(realization_id)<0):
            raise ValueError('realization_id must be a nonnegative integer or None')
        self._requested_realization_id = (None if realization_id is None else int(realization_id))

    def _apply_episode_weather_stochasticity(self):
        if not self.weather_uncertainty['enabled']:
            self._weather_realization_seed = None
            self._weather_realization_id = None
            self._weather_profile_sha256 = None
            return super()._apply_episode_weather_stochasticity()
        from mbpo.env.nsrdb_weather import scenario_id as nsrdb_scenario_id
        sid=nsrdb_scenario_id(self._current_scenario)
        identity=self._requested_realization_id
        if identity is None:
            rg=self.np_random
            seed=(int(rg.integers(0,2**63-1)) if hasattr(rg,'integers')
                  else int(rg.randint(0,2**31-1)))
        else:
            seed=realization_seed(self.weather_uncertainty,sid,identity)
        solar=self.location.get_solarposition(self.times)
        cosz=np.maximum(np.cos(np.deg2rad(np.asarray(solar['zenith'],dtype=float))),0.0)
        frame,diag=generate_weather_profile(
            self.weather_profile,cosz,self.weather_uncertainty,seed=seed)
        self.weather_profile=frame
        self._weather_realization_seed=int(seed)
        self._weather_realization_id=identity
        self._weather_uncertainty_diagnostics=diag
        # Hash entire exogenous sequence; independent of policy actions.
        fields=('dni','dhi','ghi','temperature','wind_speed','surface_albedo')
        present=[k for k in fields if k in frame]
        albedo_col=next((c for c in ALBEDO_COLUMNS if c in frame),None)
        if albedo_col is not None and albedo_col not in present:present.append(albedo_col)
        matrix=np.column_stack([np.asarray(frame[k],dtype='<f8') for k in present])
        self._weather_profile_sha256=hashlib.sha256(matrix.tobytes()).hexdigest()

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
        if self.weather_uncertainty['enabled'] and self.weather_uncertainty['albedo_sd']>0:
            info['albedo_source'] = 'nsrdb_recorded_with_assumed_uncertainty'
        info['weather_condition_source'] = 'original_nsrdb_label'
        info['weather_uncertainty_enabled'] = bool(self.weather_uncertainty['enabled'])
        info['weather_realization_seed'] = self._weather_realization_seed
        info['weather_realization_id'] = self._weather_realization_id
        info['weather_profile_sha256'] = self._weather_profile_sha256
        info['weather_uncertainty_spec_sha256'] = spec_fingerprint(self.weather_uncertainty)
        info['weather_uncertainty_clip_count'] = int(getattr(self,'_weather_uncertainty_diagnostics',{}).get('clip_count',0))
        info['weather_uncertainty_baseline_closure_max_wm2'] = getattr(
            self,'_weather_uncertainty_diagnostics',{}).get('max_baseline_closure_residual_wm2')
        return obs, reward, done, info
