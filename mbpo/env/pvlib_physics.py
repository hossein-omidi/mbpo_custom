"""Shared pvlib POA irradiance and panel power (used by env, eval, verification).

Power model: POA_global (isotropic sky) × area × efficiency.
Temperature and wind_speed in NSRDB observations are exogenous state features;
they are not used in the current DC power formula (no pvlib temperature derate).
"""

from pvlib.irradiance import get_total_irradiance


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
    irradiance = get_total_irradiance(
        surface_tilt=float(surface_tilt),
        surface_azimuth=float(surface_azimuth),
        solar_zenith=float(solar_zenith),
        solar_azimuth=float(solar_azimuth),
        dni=float(dni),
        ghi=float(ghi),
        dhi=float(dhi),
        model=model,
    )
    return max(float(irradiance['poa_global']), 0.0)


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
