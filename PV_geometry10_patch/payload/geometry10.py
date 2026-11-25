"""Geometry-aware state contract, independent of the RL and physics backends.

Only current observations enter the policy. The calendar context is used by
MBPO solely to compute the deterministic ephemeris of synthetic transitions.
"""
from __future__ import annotations
import numpy as np

MODE = 'geometry10'
STATE_DIM = 10
ACTION_DIM = 2
TAU = 9
PANEL_AZ = slice(4, 6)
SOLAR = slice(6, 9)
WEATHER_INPUTS = (0, 1, 6, 7, 8, 9)
WEATHER_DIM = len(WEATHER_INPUTS)
POWER_DIM = STATE_DIM + ACTION_DIM + 2
FEATURE_NAMES = ('dni_norm', 'dhi_norm', 'power_norm', 'panel_tilt_norm',
                 'panel_azimuth_sin', 'panel_azimuth_cos', 'solar_zenith_norm',
                 'solar_azimuth_sin', 'solar_azimuth_cos', 'tau')
LOW = np.array([0,0,0,0,-1,-1,0,-1,-1,0], dtype=np.float32)
HIGH = np.ones(STATE_DIM, dtype=np.float32)
IRRADIANCE_NORM = 2000.
POWER_NORM = 2000.


def circular(deg):
    a = np.deg2rad(deg)
    return np.stack((np.sin(a), np.cos(a)), axis=-1)


def azimuth(pair):
    a = np.asarray(pair)
    return np.mod(np.rad2deg(np.arctan2(a[..., 0], a[..., 1])), 360.)


def solar_features(zenith, solar_azimuth):
    z = np.asarray(zenith, dtype=float)
    a = np.asarray(solar_azimuth, dtype=float)
    if not np.isfinite(z).all() or not np.isfinite(a).all() or np.any((z<0)|(z>180)):
        raise ValueError('Invalid solar geometry')
    return np.concatenate((z[..., None]/180., circular(a)), axis=-1).astype(np.float32)


def check_batch(states):
    x = np.asarray(states, dtype=np.float32)
    if x.ndim < 1 or x.shape[-1] != STATE_DIM or not np.isfinite(x).all():
        raise ValueError('Expected finite geometry10 observation(s)')
    if np.any(x<LOW-1e-5) or np.any(x>HIGH+1e-5):
        raise ValueError('Geometry10 observation outside physical bounds')
    for block in (PANEL_AZ, slice(7,9)):
        if not np.allclose(np.linalg.norm(x[...,block], axis=-1), 1., atol=2e-5, rtol=0):
            raise ValueError('Invalid azimuth unit-circle encoding')
    return x


def build(dni, dhi, power, tilt, panel_azimuth, zenith, solar_azimuth, tau):
    raw = np.array([dni,dhi,power,tilt,panel_azimuth,zenith,solar_azimuth,tau], dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite physical observation')
    obs = np.array([np.clip(dni/IRRADIANCE_NORM,0,1),np.clip(dhi/IRRADIANCE_NORM,0,1),
                    np.clip(power/POWER_NORM,0,1),tilt/90., *circular(panel_azimuth),
                    *solar_features(zenith,solar_azimuth),tau], dtype=np.float32)
    return check_batch(obs)


def weather_features(states):
    return check_batch(states)[..., list(WEATHER_INPUTS)].astype(np.float32)


def angular_rmse_deg(a,b):
    d = (azimuth(a)-azimuth(b)+180.)%360.-180.
    return float(np.sqrt(np.mean(d*d)))


def pose_transition(states,actions,max_tilt,max_az,horizon):
    """Exact real-environment angular/time convention, including wrap and clipping."""
    x = check_batch(states)
    a = np.asarray(actions,dtype=np.float32)
    if x.ndim!=2 or a.shape!=(len(x),2) or not np.isfinite(a).all() or np.any(np.abs(a)>1.00001):
        raise ValueError('Invalid geometry10 action batch')
    if horizon<=0 or max_tilt<=0 or max_az<=0 or np.any(x[:,TAU]<=0):
        raise ValueError('Invalid horizon, limits, or terminal starting state')
    old_tilt=x[:,3].astype(float)*90.
    old_az=azimuth(x[:,PANEL_AZ].astype(float))
    new_tilt=np.clip(old_tilt+a[:,0].astype(float)*max_tilt,0,90.)
    new_az=(old_az+a[:,1].astype(float)*max_az)%360.
    dt=new_tilt-old_tilt
    da=(new_az-old_az+180.)%360.-180.
    remaining=np.rint(x[:,TAU].astype(float)*horizon).astype(int)
    if np.any(remaining<1) or np.any(remaining>horizon) or not np.allclose(x[:,TAU],remaining/horizon,atol=2e-6,rtol=0):
        raise ValueError('Remaining time does not match finite-horizon grid')
    tau=(remaining-1)/horizon
    return new_tilt/90.,circular(new_az),tau,dt,da,remaining==1


class SolarContext:
    """Cache ephemeris only; never read future meteorological/power observations.

    Uses the physical environment's own scenario date, clock grid, location and
    pvlib solar position convention. One-step rollouts require the originating
    real replay scenario ID. No context is added to the actor observation.
    """
    def __init__(self, reference_env):
        self.env=reference_env.unwrapped
        self.horizon=int(reference_env.horizon)
        self.cache={}

    def grid(self,sid):
        sid=str(sid)
        if sid not in self.cache:
            from mbpo.env.nsrdb_weather import scenario_episode_date
            if sid not in self.env._nsrdb_by_id:
                raise ValueError(f'Unknown geometry scenario {sid!r}')
            date=scenario_episode_date(self.env._nsrdb_by_id[sid],tz=self.env.location.tz)
            times=self.env._build_times(date)
            if len(times)!=self.horizon+1:
                raise ValueError('Ephemeris grid and episode horizon differ')
            sp=self.env.location.get_solarposition(times)
            self.cache[sid]=solar_features(sp['zenith'].to_numpy(),sp['azimuth'].to_numpy())
        return self.cache[sid]

    def next(self,states,scenario_ids):
        x=check_batch(states)
        if x.ndim!=2 or scenario_ids is None or len(scenario_ids)!=len(x):
            raise ValueError('One scenario ID per synthetic state is mandatory')
        remain=np.rint(x[:,TAU].astype(float)*self.horizon).astype(int)
        if np.any(remain<1) or np.any(remain>self.horizon) or not np.allclose(x[:,TAU],remain/self.horizon,atol=2e-6,rtol=0):
            raise ValueError('Invalid synthetic episode clock')
        step=self.horizon-remain
        ids=np.asarray(scenario_ids,dtype=str)
        result=np.empty((len(x),3),dtype=np.float32)
        for sid in np.unique(ids):
            mask=ids==sid
            grid=self.grid(sid)
            if not np.allclose(x[mask,SOLAR],grid[step[mask]],atol=3e-5,rtol=0):
                raise ValueError(f'Current replay solar geometry inconsistent with scenario/time: {sid}')
            result[mask]=grid[step[mask]+1]
        return result
