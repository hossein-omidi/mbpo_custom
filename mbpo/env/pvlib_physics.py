"""Shared pvlib POA irradiance and panel power (used by env, eval, verification).

Angle conventions (uniform across NSRDB, pvlib, env state, observations, baselines)
---------------------------------------------------------------------------------
All angles are in **degrees** unless noted.

* **Panel surface tilt** (`surface_tilt` in pvlib): angle between the module
  plane and the **horizontal ground**, 0° = horizontal (face-up), 90° = vertical.
  Valid range for this project: **[0°, 90°]** — negative tilt (below horizontal)
  is non-physical for a ground mount and is clipped before every pvlib call.

* **Panel surface azimuth** (`surface_azimuth` in pvlib): compass bearing the
  module **faces** (normal projection on horizontal plane), **0° = North**,
  **90° = East**, **180° = South**, **270° = West**. Stored in **[0°, 360°)**.

* **Solar zenith** (pvlib / NSRDB ``Solar Zenith Angle``): angle from the
  vertical to the sun, 0° = overhead, 90° = horizon, **> 90°** = below horizon
  (possible at winter grid edges). Never used as a negative panel tilt.

* **Solar azimuth** (pvlib): compass bearing **to the sun**, same N/E/S/W rule.

* **Control actions** ``a = [a_tilt, a_az]`` in ``[-1, 1]``: normalized
  **increments** ``Δtilt = a_tilt * max_delta_tilt``,
  ``Δaz = a_az * max_delta_azimuth`` — **not** absolute angles. Negative
  ``a_tilt`` means "decrease tilt toward horizontal", not "negative tilt".

* **Observation** ``panel_tilt_norm = tilt_deg / 90`` ∈ [0, 1];
  ``solar_zenith_norm = clip(zenith_deg / 180, 0, 1)``.

POA transposition: ``pvlib.irradiance.get_total_irradiance`` (default
``model='isotropic'``) via :func:`compute_poa_global`. All env steps, baselines,
and the POA oracle use this same function — no hand-rolled transposition.

Dual-axis sun-tracker baseline: ``target_tilt = clip(solar_zenith, 0, 90)``,
``target_azimuth = solar_azimuth`` (standard astronomical dual-axis pointing).
This is **not** ``pvlib.tracking.singleaxis`` (single-axis hardware with one
rotation axis). Use :func:`single_axis_tracking_targets` only for the optional
``single_axis`` eval baseline.

Power model: POA_global × area × efficiency.
Temperature and wind_speed in NSRDB observations are exogenous state features;
they are not used in the current DC power formula (no pvlib temperature derate).
"""

import numpy as np
from pvlib.irradiance import get_total_irradiance

# Panel orientation limits (pvlib surface_tilt / surface_azimuth).
PANEL_TILT_DEG_MIN = 0.0
PANEL_TILT_DEG_MAX = 90.0
PANEL_AZIMUTH_DEG_MIN = 0.0
PANEL_AZIMUTH_DEG_MAX = 360.0  # stored as [0, 360)


def clip_panel_tilt_deg(tilt_deg):
    """Clip panel tilt to [0°, 90°] (pvlib surface_tilt, ground-mount)."""
    return float(np.clip(float(tilt_deg), PANEL_TILT_DEG_MIN, PANEL_TILT_DEG_MAX))


def wrap_panel_azimuth_deg(azimuth_deg):
    """Normalize panel azimuth to [0°, 360°) (pvlib surface_azimuth)."""
    return float(np.mod(float(azimuth_deg), PANEL_AZIMUTH_DEG_MAX))


def clip_panel_orientation(surface_tilt_deg, surface_azimuth_deg):
    """Return (tilt, azimuth) in the project's valid pvlib ranges."""
    return (
        clip_panel_tilt_deg(surface_tilt_deg),
        wrap_panel_azimuth_deg(surface_azimuth_deg),
    )


def panel_tilt_norm_from_deg(tilt_deg):
    """Observation feature panel_tilt_norm = tilt / 90."""
    return clip_panel_tilt_deg(tilt_deg) / PANEL_TILT_DEG_MAX


def panel_tilt_deg_from_norm(tilt_norm):
    """Decode panel_tilt_norm back to degrees."""
    return clip_panel_tilt_deg(float(tilt_norm) * PANEL_TILT_DEG_MAX)


def solar_zenith_norm_from_deg(zenith_deg):
    """Observation feature solar_zenith_norm (matches PVTrackingEnv)."""
    return float(np.clip(float(zenith_deg) / 180.0, 0.0, 1.0))


def sun_tracker_target_tilt_deg(solar_zenith_deg):
    """Dual-axis heuristic: face the sun, capped at vertical."""
    return clip_panel_tilt_deg(solar_zenith_deg)


def dual_axis_tracking_targets(solar_zenith_deg, solar_azimuth_deg):
    """Ideal dual-axis tracker surface angles (same convention as PVTrackingEnv).

    Returns ``(surface_tilt, surface_azimuth)`` for a full dual-axis mount that
    slews toward the sun each step. POA at these angles still uses
    :func:`compute_poa_global` → ``get_total_irradiance``.
    """
    return (
        sun_tracker_target_tilt_deg(solar_zenith_deg),
        wrap_panel_azimuth_deg(solar_azimuth_deg),
    )


def single_axis_tracking_targets(
        apparent_zenith,
        apparent_azimuth,
        axis_tilt=0.0,
        axis_azimuth=180.0,
        max_angle=90.0,
        backtrack=True,
        gcr=0.35):
    """Surface angles from ``pvlib.tracking.singleaxis`` (one-axis tracker).

    Uses apparent zenith/azimuth as required by pvlib. When the sun is below
    the horizon, pvlib returns NaN — fall back to ``(0°, axis_azimuth)``.
    """
    from pvlib import tracking

    result = tracking.singleaxis(
        apparent_zenith=apparent_zenith,
        apparent_azimuth=apparent_azimuth,
        axis_tilt=axis_tilt,
        axis_azimuth=axis_azimuth,
        max_angle=max_angle,
        backtrack=backtrack,
        gcr=gcr,
    )
    tilt = float(result['surface_tilt'])
    az = float(result['surface_azimuth'])
    if not np.isfinite(tilt) or not np.isfinite(az):
        return clip_panel_orientation(0.0, axis_azimuth)
    return clip_panel_orientation(tilt, az)


def compute_poa_global(
        surface_tilt,
        surface_azimuth,
        solar_zenith,
        solar_azimuth,
        dni,
        ghi,
        dhi,
        model='isotropic'):
    """Plane-of-array global irradiance [W/m²] — same path as PVTrackingEnv."""
    tilt, azimuth = clip_panel_orientation(surface_tilt, surface_azimuth)
    irradiance = get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=float(solar_zenith),
        solar_azimuth=float(solar_azimuth),
        dni=float(dni),
        ghi=float(ghi),
        dhi=float(dhi),
        model=model,
    )
    return max(float(irradiance['poa_global']), 0.0)


def verify_get_total_irradiance_parity(
        surface_tilt,
        surface_azimuth,
        solar_zenith,
        solar_azimuth,
        dni,
        ghi,
        dhi,
        model='isotropic',
        rtol=1e-9):
    """True when :func:`compute_poa_global` matches a direct pvlib call."""
    poa = compute_poa_global(
        surface_tilt, surface_azimuth,
        solar_zenith, solar_azimuth,
        dni, ghi, dhi, model=model)
    direct = get_total_irradiance(
        surface_tilt=clip_panel_tilt_deg(surface_tilt),
        surface_azimuth=wrap_panel_azimuth_deg(surface_azimuth),
        solar_zenith=float(solar_zenith),
        solar_azimuth=float(solar_azimuth),
        dni=float(dni),
        ghi=float(ghi),
        dhi=float(dhi),
        model=model,
    )
    expected = max(float(direct['poa_global']), 0.0)
    if expected < 1e-12 and poa < 1e-12:
        return True, 0.0
    return np.isclose(poa, expected, rtol=rtol, atol=1e-9), abs(poa - expected)


def compute_panel_power_w(
        surface_tilt,
        surface_azimuth,
        solar_zenith,
        solar_azimuth,
        dni,
        ghi,
        dhi,
        area=1.0,
        efficiency=0.18,
        model='isotropic'):
    """Panel DC power [W] from pvlib POA × area × efficiency."""
    poa = compute_poa_global(
        surface_tilt, surface_azimuth,
        solar_zenith, solar_azimuth,
        dni, ghi, dhi, model=model)
    return poa * float(area) * float(efficiency)


def energy_kwh_from_power_w(power_w, interval_hours):
    """Step energy [kWh] matching env.step reward units."""
    return float(power_w) * float(interval_hours) / 1000.0


# Dual-axis tracker defaults for area=1 m² (residential-scale, matches PVTrackingEnv).
# Motor draws electrical power only while slewing; duration = |Δangle| / slew rate.
DEFAULT_ACTUATOR_POWER_W = 30.0
DEFAULT_SLEW_RATE_TILT_DEG_S = 1.5
DEFAULT_SLEW_RATE_AZIMUTH_DEG_S = 2.0


def actuator_movement_cost_kwh(
        delta_tilt_deg,
        delta_azimuth_deg,
        actuator_power_w=DEFAULT_ACTUATOR_POWER_W,
        slew_rate_tilt_deg_s=DEFAULT_SLEW_RATE_TILT_DEG_S,
        slew_rate_azimuth_deg_s=DEFAULT_SLEW_RATE_AZIMUTH_DEG_S,
        scale=1.0):
    """Electrical actuator energy [kWh] for one control step.

    E = P_motor * (|Δtilt|/ω_tilt + |Δaz|/ω_az) / 3600 / 1000, scaled by ``scale``.
  At max env deltas (5° tilt + 10° azimuth) this is ~0.07 Wh — ~1% of a typical
  5-min harvest step on the NYC NSRDB window (comparable to literature parasitic
  losses for dual-axis trackers).
    """
    tilt_time_s = abs(float(delta_tilt_deg)) / float(slew_rate_tilt_deg_s)
    az_time_s = abs(float(delta_azimuth_deg)) / float(slew_rate_azimuth_deg_s)
    energy_kwh = float(actuator_power_w) * (tilt_time_s + az_time_s) / 3.6e6
    return float(scale) * energy_kwh
