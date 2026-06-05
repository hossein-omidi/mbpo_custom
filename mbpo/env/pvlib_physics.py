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
