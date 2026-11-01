"""Minimal adapter over PVTrackingEnv plus recorded-NSRDB-albedo Stage A subclass.

Normalized actions stay [-1,1]^2. Degrees and rewards are computed by the
original environment, not approximated or transformed inside this adapter.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from importlib import metadata
from SAC_workspace.src.representation.representation import encode


class SACEnvironment:
    def __init__(self, env_kwargs: dict):
        from mbpo.env.stage_a_pv_tracking import StageAPVTrackingEnv
        if env_kwargs.get('observation_mode') != 'compact':
            raise ValueError('Must use compact mode for reference parity')
        self.env = StageAPVTrackingEnv(**dict(env_kwargs))
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.horizon = int(self.env.num_action_steps)
        if self.horizon != 117 or self.observation_space.shape != (6,) or self.action_space.shape != (2,):
            raise ValueError('Resolved MDP does not match compact 117-step reference')
        if not np.array_equal(np.asarray(self.action_space.low), np.array([-1, -1])) or not np.array_equal(np.asarray(self.action_space.high), np.array([1, 1])):
            raise ValueError('Unexpected reference action bounds')
        self.steps = 0
        # Record the effective PVTrackingEnv parameters, including defaults
        # omitted by the reference config. This catches upstream physics drift.
        source = self.env
        location = source.location
        def value(name):
            return getattr(source, name, None)
        def library_version(name):
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                return None
        self.resolved_mdp = {
            'pvlib_version': library_version('pvlib'),
            'gym_version': library_version('gym'),
            'site_latitude': getattr(location, 'latitude', None),
            'site_longitude': getattr(location, 'longitude', None),
            'site_altitude': getattr(location, 'altitude', None),
            'tz': str(location.tz),
            'start_time': value('start_time'), 'periods': value('periods'),
            'freq': str(value('freq')),
            'weather_source': value('weather_source'),
            'area': value('area'), 'efficiency': value('efficiency'),
            'max_delta_tilt': value('max_delta_tilt'),
            'max_delta_azimuth': value('max_delta_azimuth'),
            'movement_penalty': value('movement_penalty'),
            'movement_cost_mode': value('movement_cost_mode'),
            'actuator_power_w': value('actuator_power_w'),
            'slew_rate_tilt_deg_s': value('slew_rate_tilt_deg_s'),
            'slew_rate_azimuth_deg_s': value('slew_rate_azimuth_deg_s'),
            'observation_mode': value('observation_mode'),
            'horizon': self.horizon,
        }
        if source.weather_source == 'nsrdb_multiyear':
            from mbpo.env.stage_a_data_contract import nsrdb_dataset_fingerprint
            self.resolved_mdp.update(nsrdb_dataset_fingerprint(source.scenario_manifest_path))
            self.resolved_mdp['nsrdb_albedo_contract'] = 'recorded_per_5min_strict'
        if source.power_model == 'pvwatts_sapm':
            import hashlib
            from pathlib import Path
            from mbpo.env import pv_power_backend, pvlib_physics, pv_tracking, stage_a_pv_tracking, stage_a_data_contract
            digest = hashlib.sha256()
            for module in (pv_power_backend, pvlib_physics, pv_tracking, stage_a_pv_tracking, stage_a_data_contract):
                digest.update(Path(module.__file__).read_bytes())
            self.resolved_mdp.update(
                power_model=source.power_model,
                albedo_model='nsrdb_recorded_strict',
                physics_revision=pv_power_backend.PHYSICS_REVISION,
                wind_reference_height_m=float(source.wind_reference_height_m),
                gamma_pdc=float(source.gamma_pdc),
                sapm_mount=source.sapm_mount,
                wind_powerlaw_exponent=float(source.wind_powerlaw_exponent),
                power_source_sha256=digest.hexdigest(),
                pv_power_backend_sha256=hashlib.sha256(
                    Path(pv_power_backend.__file__).read_bytes()).hexdigest(),
                pvlib_physics_sha256=hashlib.sha256(
                    Path(pvlib_physics.__file__).read_bytes()).hexdigest(),
            )

    def seed(self, seed: int) -> None:
        self.env.seed(seed)

    def reset(self, scenario_id: str | None = None) -> np.ndarray:
        self.steps = 0
        observation = encode(self.env.reset(scenario_id=scenario_id))
        if not np.isclose(observation[5], 1., atol=1e-6):
            raise RuntimeError('Finite-horizon observation must begin with tau=1')
        if str(self.env.location.tz).upper() != 'UTC':
            raise RuntimeError('Reference pvlib/NSRDB experiment must use UTC')
        return observation

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (2,) or not np.isfinite(a).all():
            raise ValueError(f'Action must be finite shape (2,), got {a}')
        if (np.abs(a) > 1.00001).any():
            raise ValueError('Policy generated action outside [-1,1]^2')
        obs, reward, done, info = self.env.step(np.clip(a, -1., 1.))
        self.steps += 1
        if self.steps > self.horizon or (self.steps == self.horizon) != bool(done):
            raise RuntimeError('Episode horizon or terminal signal disagrees with reference')
        observation = encode(obs)
        tau_expected = (self.horizon - self.steps) / self.horizon
        if not np.isclose(observation[5], tau_expected, atol=1e-6):
            raise RuntimeError('PVTrackingEnv compact tau feature does not match step count')
        if not np.isclose(reward, float(info['energy_kwh']) - float(info['movement_cost']), atol=1e-9):
            raise RuntimeError('Step reward differs from the reference net-energy definition')
        if getattr(self.env, 'power_model', 'legacy') == 'pvwatts_sapm':
            if info.get('power_model') != 'pvwatts_sapm':
                raise RuntimeError('PVWatts transition reported wrong backend')
            expected_energy = float(info['power_dc_w']) * float(self.env.interval_hours) / 1000.0
            if not np.isclose(expected_energy, float(info['energy_kwh']), rtol=0, atol=1e-10):
                raise RuntimeError('Power/energy identity failed for PVWatts')
            if not np.isclose(float(info['power']), float(info['power_dc_w']), rtol=0, atol=1e-10):
                raise RuntimeError('PVWatts DC power reporting mismatch')
            if info.get('cell_temperature_c') is None or info.get('wind_speed_10m_ms') is None:
                raise RuntimeError('Missing PVWatts cell temperature or wind metadata')
            # Normalized power is the *same* third feature in both physics modes.
            if not np.isclose(observation[2], np.clip(float(info['power_dc_w']) / 2000., 0, 1), atol=1e-6):
                raise RuntimeError('PVWatts power observation is inconsistent')
        expected_time = self.env.times[self.steps]
        actual_time = pd.Timestamp(info['timestamp_utc_iso'])
        if (actual_time != expected_time or str(info['timezone']).upper() != 'UTC'
                or expected_time.utcoffset().total_seconds() != 0):
            raise RuntimeError(f'Solar step UTC timestamp mismatch: {actual_time!s} != {expected_time!s}')
        if (self.env.weather_source == 'nsrdb_multiyear'
                and (info['weather_resampling'] != 'none_native_5min'
                     or not np.isclose(info['control_interval_minutes'], 5., atol=1e-6))):
            raise RuntimeError('NSRDB scenario is not using the reference native 5-minute grid')
        return observation, float(reward), bool(done), info

    @property
    def unwrapped(self):
        return self.env

    def close(self):
        self.env.close()
