"""Stand-alone additive backend contract tests; no NSRDB files required.

From the repo root:
  python -m pytest -q tests/test_pv_power_backend.py
"""
from __future__ import annotations

import inspect
import math
import sys

import numpy as np
import pytest
import pvlib

from mbpo.env.pv_power_backend import compute_pv_power
from mbpo.env.pvlib_physics import compute_panel_power_w

BASE = dict(surface_tilt=30.0, surface_azimuth=180.0,
            solar_zenith=35.0, solar_azimuth=185.0,
            dni=700.0, ghi=800.0, dhi=140.0,
            area=1.0, efficiency=0.18)
WEATHER = dict(temp_air_c=25.0, wind_speed_ms=2.0,
               wind_reference_height_m=2.0)


def test_runtime_api_compatibility():
    assert sys.version_info >= (3, 10)
    dc = inspect.signature(pvlib.pvsystem.pvwatts_dc).parameters
    assert {'effective_irradiance', 'temp_cell', 'pdc0', 'gamma_pdc', 'temp_ref'} <= set(dc)
    temp = inspect.signature(pvlib.temperature.sapm_cell).parameters
    assert {'poa_global', 'temp_air', 'wind_speed', 'a', 'b', 'deltaT'} <= set(temp)
    wind = inspect.signature(pvlib.atmosphere.windspeed_powerlaw).parameters
    assert {'wind_speed_reference', 'height_reference', 'height_desired', 'exponent'} <= set(wind)
    assert 'open_rack_glass_polymer' in pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']


@pytest.mark.parametrize('tilt,azimuth,zenith,sunaz', [
    (0., 0., 10., 180.), (30., 180., 30., 175.),
    (90., 0., 85., 300.), (40., 360., 94., 10.),
])
def test_exact_legacy_power_parity(tilt, azimuth, zenith, sunaz):
    b = {**BASE, 'surface_tilt': tilt, 'surface_azimuth': azimuth,
         'solar_zenith': zenith, 'solar_azimuth': sunaz}
    expected = compute_panel_power_w(**b)
    out = compute_pv_power(**b)
    assert out.power_model == 'legacy'
    assert out.power_w == expected
    assert out.power_w == out.poa_global_wm2 * b['area'] * b['efficiency']
    assert out.cell_temperature_c is None


def test_manual_independent_sapm_and_pvwatts_math():
    out = compute_pv_power(**BASE, **WEATHER, power_model='pvwatts_sapm')
    # Independent analytical equations (not calling pvlib's temp or power helpers).
    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
    v10 = WEATHER['wind_speed_ms'] * (10.0 / WEATHER['wind_reference_height_m'])**0.14
    tcell = (WEATHER['temp_air_c']
             + out.poa_global_wm2 * math.exp(params['a'] + params['b'] * v10)
             + out.poa_global_wm2 / 1000.0 * params['deltaT'])
    expected = (out.poa_global_wm2 / 1000.0 * 180.0 *
                (1.0 - 0.004 * (tcell - 25.0)))
    assert out.wind_speed_10m_ms == pytest.approx(v10, rel=1e-12)
    assert out.cell_temperature_c == pytest.approx(tcell, rel=1e-12)
    assert out.power_w == pytest.approx(expected, rel=1e-12)
    assert out.effective_irradiance_wm2 == out.poa_global_wm2


def test_stc_algebra_and_temperature_direction():
    assert pvlib.pvsystem.pvwatts_dc(1000., 25., 180., -0.004) == pytest.approx(180.)
    cold = compute_pv_power(**BASE, **{**WEATHER, 'temp_air_c': 5.},
                            power_model='pvwatts_sapm')
    hot = compute_pv_power(**BASE, **{**WEATHER, 'temp_air_c': 35.},
                           power_model='pvwatts_sapm')
    assert cold.poa_global_wm2 == hot.poa_global_wm2
    assert cold.cell_temperature_c < hot.cell_temperature_c
    assert cold.power_w > hot.power_w


def test_wind_direction_and_height_identity():
    calm = compute_pv_power(**BASE, **{**WEATHER, 'wind_speed_ms': 0.2},
                            power_model='pvwatts_sapm')
    windy = compute_pv_power(**BASE, **{**WEATHER, 'wind_speed_ms': 9.},
                             power_model='pvwatts_sapm')
    assert windy.cell_temperature_c < calm.cell_temperature_c
    assert windy.power_w > calm.power_w
    at_10m = compute_pv_power(**BASE, **{**WEATHER, 'wind_reference_height_m': 10.},
                              power_model='pvwatts_sapm')
    assert at_10m.wind_speed_10m_ms == pytest.approx(WEATHER['wind_speed_ms'])


def test_poa_identity_and_dark_case():
    old = compute_pv_power(**BASE)
    new = compute_pv_power(**BASE, **WEATHER, power_model='pvwatts_sapm')
    assert old.poa_global_wm2 == new.poa_global_wm2
    dark = compute_pv_power(**{**BASE, 'dni': 0., 'ghi': 0., 'dhi': 0.},
                            **WEATHER, power_model='pvwatts_sapm')
    assert dark.power_w == 0.
    assert dark.poa_global_wm2 == 0.


@pytest.mark.parametrize('extra,pattern', [
    ({'temp_air_c': None}, 'temp_air_c'),
    ({'wind_speed_ms': None}, 'wind_speed_ms'),
    ({'wind_reference_height_m': None}, 'wind_reference_height_m'),
    ({'wind_speed_ms': -1.}, 'wind'),
    ({'wind_reference_height_m': 0.}, 'height'),
    ({'gamma_pdc': float('nan')}, 'gamma_pdc'),
    ({'sapm_mount': 'bad_mount'}, 'mounting'),
])
def test_invalid_model_inputs_fail(extra, pattern):
    with pytest.raises(ValueError, match=pattern):
        compute_pv_power(**BASE, **{**WEATHER, **extra}, power_model='pvwatts_sapm')


def test_invalid_irradiance_and_model_fail():
    for key in ('dni', 'ghi', 'dhi'):
        with pytest.raises(ValueError, match=key):
            compute_pv_power(**{**BASE, key: -0.5})
    with pytest.raises(ValueError, match='power_model'):
        compute_pv_power(**BASE, power_model='unknown')


def test_output_units_and_no_fallback_weather():
    out = compute_pv_power(**BASE, **WEATHER, power_model='pvwatts_sapm')
    assert np.isfinite(out.power_w)
    assert 0 < out.power_w < 500
    assert out.ambient_temperature_c == WEATHER['temp_air_c']
    assert out.wind_speed_reference_ms == WEATHER['wind_speed_ms']
    with pytest.raises(ValueError):
        compute_pv_power(**BASE, power_model='pvwatts_sapm')
