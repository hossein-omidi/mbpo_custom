"""Shared scalar PV DC backend for Stage A: isotropic POA, optional recorded albedo.

Retains the repository's 6-observation/2-action interface and the documented
PVWatts V5 + steady-state SAPM temperature approximation. This is NOT an IAM,
spectral, single-diode, inverter, or bifacial model. The actual NSRDB albedo is
passed by the StageAPVTrackingEnv adapter, never estimated from an RL state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from pvlib import atmosphere, irradiance, pvsystem, temperature
from mbpo.env.pvlib_physics import compute_poa_global, clip_panel_orientation

DEFAULT_ALBEDO = 0.25  # Explicit pvlib historic/default assumption; non-NSRDB only
PHYSICS_REVISION = 'stageA_nsrdb_recorded_albedo_v1'


@dataclass(frozen=True)
class PowerResult:
    power_w: float
    poa_global_wm2: float
    effective_irradiance_wm2: float
    cell_temperature_c: Optional[float]
    ambient_temperature_c: Optional[float]
    wind_speed_reference_ms: Optional[float]
    wind_speed_10m_ms: Optional[float]
    power_model: str
    surface_albedo: Optional[float] = None


def _finite(value: object, name: str) -> float:
    if value is None:
        raise ValueError(f'{name} is required')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def compute_pv_power(
    surface_tilt: float, surface_azimuth: float,
    solar_zenith: float, solar_azimuth: float,
    dni: float, ghi: float, dhi: float,
    *, area: float = 1.0, efficiency: float = 0.18,
    power_model: str = 'pvwatts_sapm',
    temp_air_c: Optional[float] = None,
    wind_speed_ms: Optional[float] = None,
    wind_reference_height_m: Optional[float] = None,
    gamma_pdc: float = -0.004,
    sapm_mount: str = 'open_rack_glass_polymer',
    wind_powerlaw_exponent: float = 0.14,
    surface_albedo: Optional[float] = None,
) -> PowerResult:
    """Calculate POA and DC power using the specified *measured* weather.

    A missing ``surface_albedo`` retains the historical pvlib default, but
    StageAPVTrackingEnv requires a recorded albedo for NSRDB scenarios.
    Numerical precision and current/future interval semantics are unchanged.
    """
    if power_model not in ('legacy', 'pvwatts_sapm'):
        raise ValueError(f'Unknown power model: {power_model!r}')
    area = _finite(area, 'area'); eta = _finite(efficiency, 'efficiency')
    if area <= 0 or not 0 < eta <= 1:
        raise ValueError('area must be positive; efficiency in (0,1]')
    for name, val in [('dni', dni), ('ghi', ghi), ('dhi', dhi)]:
        if _finite(val, name) < 0:
            raise ValueError(f'{name} cannot be negative')
    for name, val in [('surface_tilt', surface_tilt),
                      ('surface_azimuth', surface_azimuth),
                      ('solar_zenith', solar_zenith),
                      ('solar_azimuth', solar_azimuth)]:
        _finite(val, name)
    if surface_albedo is not None:
        albedo = _finite(surface_albedo, 'surface_albedo')
        if not 0 <= albedo <= 1:
            raise ValueError(f'Surface albedo must be in [0,1], got {albedo}')
        tilt, azimuth = clip_panel_orientation(surface_tilt, surface_azimuth)
        parts = irradiance.get_total_irradiance(
            surface_tilt=tilt, surface_azimuth=azimuth,
            solar_zenith=float(solar_zenith), solar_azimuth=float(solar_azimuth),
            dni=float(dni), ghi=float(ghi), dhi=float(dhi),
            albedo=albedo, model='isotropic')
        poa = _finite(parts['poa_global'], 'poa_global')
    else:
        albedo = DEFAULT_ALBEDO
        poa = _finite(compute_poa_global(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
            dni, ghi, dhi, model='isotropic'), 'poa_global')
    if poa < -1e-8:
        raise FloatingPointError('Negative POA from pvlib irradiance transposition')
    poa = max(0.0, poa)
    if power_model == 'legacy':
        return PowerResult(poa*area*eta,poa,poa,None,None,None,None,'legacy',albedo)

    air = _finite(temp_air_c,'temp_air_c')
    wind = _finite(wind_speed_ms,'wind_speed_ms')
    ref_h = _finite(wind_reference_height_m,'wind_reference_height_m')
    exponent = _finite(wind_powerlaw_exponent,'wind_powerlaw_exponent')
    gamma = _finite(gamma_pdc,'gamma_pdc')
    if wind < 0 or ref_h <= 0 or not 0 <= exponent <= 1:
        raise ValueError('Invalid wind/reference height/exponent')
    if not -.02 < gamma <= 0:
        raise ValueError('gamma_pdc must be in (-0.02,0]')
    try:
        params = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][sapm_mount]
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Unknown SAPM mounting preset {sapm_mount!r}') from exc
    wind10 = _finite(atmosphere.windspeed_powerlaw(
        wind, height_reference=ref_h, height_desired=10.0, exponent=exponent),
        'wind_speed_10m')
    tcell = _finite(temperature.sapm_cell(
        poa_global=poa, temp_air=air, wind_speed=wind10, **params),
        'cell_temperature_c')
    # Deliberately POA≈effective irradiance. IAM is NOT part of Stage A.
    dc = _finite(pvsystem.pvwatts_dc(
        effective_irradiance=poa, temp_cell=tcell,
        pdc0=1000.0*area*eta, gamma_pdc=gamma, temp_ref=25.0),
        'pvwatts_dc')
    return PowerResult(max(0.0,dc),poa,poa,tcell,air,wind,wind10,'pvwatts_sapm',albedo)
