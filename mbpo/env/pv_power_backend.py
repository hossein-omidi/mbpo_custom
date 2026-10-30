"""Opt-in DC-power calculation; Stage 1 does NOT modify PVTrackingEnv.

Both modes retain the repository's isotropic pvlib POA transposition.

* ``legacy``: exactly the existing model, POA * area * efficiency.
* ``pvwatts_sapm``: steady-state SAPM cell temperature and PVWatts V5 DC.

A deliberate physics boundary:
  - input wind speed must have a known measurement height; the height must be
    provided explicitly in the PVWatts mode.
  - no IAM/spectral/soiling/inverter correction: POA is used as an approximation
    to effective irradiance (NOT a complete PVWatts system chain).
  - no fallback temperature or wind values.

Requires the functions present in pvlib 0.13.1; no new neural-network interfaces.
This module is NOT activated by merely copying it into the repository.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from pvlib import atmosphere, pvsystem, temperature

from mbpo.env.pvlib_physics import compute_poa_global


@dataclass(frozen=True)
class PowerResult:
    """Scalar DC-model result; all physical units remain explicit."""
    power_w: float
    poa_global_wm2: float
    effective_irradiance_wm2: float
    cell_temperature_c: Optional[float]
    ambient_temperature_c: Optional[float]
    wind_speed_reference_ms: Optional[float]
    wind_speed_10m_ms: Optional[float]
    power_model: str


def _finite(value: object, name: str) -> float:
    if value is None:
        raise ValueError(f'{name} is required')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite numeric') from exc
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite numeric, got {value!r}')
    return result


def compute_pv_power(
    surface_tilt: float,
    surface_azimuth: float,
    solar_zenith: float,
    solar_azimuth: float,
    dni: float,
    ghi: float,
    dhi: float,
    *,
    area: float = 1.0,
    efficiency: float = 0.18,
    power_model: str = 'legacy',
    temp_air_c: Optional[float] = None,
    wind_speed_ms: Optional[float] = None,
    wind_reference_height_m: Optional[float] = None,
    gamma_pdc: float = -0.004,
    sapm_mount: str = 'open_rack_glass_polymer',
    wind_powerlaw_exponent: float = 0.14,
) -> PowerResult:
    """Evaluate one orientation/weather sample without changing simulator state.

    Parameters for ``pvwatts_sapm`` (examples, NOT module calibration):
      gamma_pdc=-0.004 [1/C], SAPM preset=open_rack_glass_polymer;
      Hellmann wind exponent=0.14.

    Wind speed is from the original data source at ``wind_reference_height_m``;
    the caller MUST supply that height explicitly. SAPM expects 10-m wind.
    ``pdc0 = 1000 * area * efficiency`` [W], preserving the legacy STC scale.
    Positive DC output is clamped to zero if a temperature correction would
    otherwise produce an unphysical negative wattage.

    Under ``legacy``, no weather-temperature or wind values are required, and
    the original POA-to-power arithmetic is preserved exactly.
    """
    if power_model not in ('legacy', 'pvwatts_sapm'):
        raise ValueError(f'Unknown power_model {power_model!r}')

    area = _finite(area, 'area')
    eta = _finite(efficiency, 'efficiency')
    if area <= 0 or not 0 < eta <= 1:
        raise ValueError('area must be > 0 and efficiency must be in (0, 1]')

    # Fail on bad weather rather than letting values silently enter the model.
    for name, val in (('dni', dni), ('ghi', ghi), ('dhi', dhi)):
        if _finite(val, name) < 0:
            raise ValueError(f'{name} must be nonnegative')
    for name, val in (('surface_tilt', surface_tilt),
                      ('surface_azimuth', surface_azimuth),
                      ('solar_zenith', solar_zenith),
                      ('solar_azimuth', solar_azimuth)):
        _finite(val, name)

    # Canonical project helper: same isotropic transposition and angle handling.
    poa = _finite(compute_poa_global(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
        dni, ghi, dhi, model='isotropic'), 'poa_global')
    if poa < 0:
        raise FloatingPointError('Negative post-transposition POA')

    if power_model == 'legacy':
        # Same arithmetic order as pvlib_physics.compute_panel_power_w.
        return PowerResult(poa * area * eta, poa, poa, None, None,
                           None, None, 'legacy')

    ta = _finite(temp_air_c, 'temp_air_c')
    wind = _finite(wind_speed_ms, 'wind_speed_ms')
    ref_h = _finite(wind_reference_height_m, 'wind_reference_height_m')
    exponent = _finite(wind_powerlaw_exponent, 'wind_powerlaw_exponent')
    gamma = _finite(gamma_pdc, 'gamma_pdc')
    if wind < 0 or ref_h <= 0 or not 0 <= exponent <= 1:
        raise ValueError('wind must be >= 0, wind height > 0, exponent in [0, 1]')
    if not -0.02 < gamma <= 0:
        raise ValueError('gamma_pdc must be non-positive and > -0.02 per C')

    try:
        params = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][sapm_mount]
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Invalid SAPM mounting preset {sapm_mount!r}') from exc

    # pvlib 0.13.1 exact signature: wind_speed_reference, height_reference,
    # height_desired, exponent=None, surface_type=None.
    wind_10m = _finite(atmosphere.windspeed_powerlaw(
        wind, height_reference=ref_h,
        height_desired=10.0, exponent=exponent), 'wind_speed_10m')
    cell = _finite(temperature.sapm_cell(
        poa_global=poa, temp_air=ta, wind_speed=wind_10m,
        **params), 'cell_temperature')

    # Approximation: no optical loss model in Stage 1, hence POA ~= G_eff.
    dc = _finite(pvsystem.pvwatts_dc(
        effective_irradiance=poa, temp_cell=cell,
        pdc0=1000.0 * area * eta, gamma_pdc=gamma,
        temp_ref=25.0), 'pvwatts_dc')
    return PowerResult(max(0.0, dc), poa, poa, cell, ta,
                       wind, wind_10m, 'pvwatts_sapm')
