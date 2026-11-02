"""Instantaneous dual-axis SUN-POINTING reference using shared PV physics.

This is a geometric sun-pointing comparator, not the POA/DC optimum and not an
actuator-feasible controller. For NSRDB, it must use the recorded per-step
surface albedo, exactly as the feasible controllers do.
"""
from __future__ import annotations

import numpy as np
from mbpo.env.pvlib_physics import dual_axis_tracking_targets
from mbpo.env.pv_power_backend import compute_pv_power


def selected_instantaneous_dual_axis(cfg, fixed_path, env=None):
    from SAC_workspace.src.environment.environment import SACEnvironment

    owns = env is None
    if owns:
        env = SACEnvironment(cfg.eval_env)

    p = {
        'method': 'dual_axis_instant_ideal',
        'scenario_id': fixed_path['scenario_id'],
        'seed': fixed_path['seed'],
        'initial_pose': list(fixed_path['initial_pose']),
        'reward': [], 'energy': [], 'power': [], 'cost': [],
        'tilt': [], 'azimuth': [], 'action_tilt': [], 'action_azimuth': [],
        'delta_tilt_deg': [], 'delta_azimuth_deg': [], 'infos': [],
    }
    try:
        u = getattr(env, 'unwrapped', env)
        infos = fixed_path.get('infos', [])
        horizon = int(env.horizon)
        if len(infos) != horizon or len(fixed_path.get('power', [])) != horizon:
            raise ValueError('Incomplete paired recorded weather for dual-axis reference')

        for info in infos:
            required = (
                'solar_zenith_deg', 'solar_azimuth_deg',
                'dni_wm2', 'ghi_wm2', 'dhi_wm2',
            )
            if any(info.get(key) is None for key in required):
                raise ValueError('Dual-axis reference is missing recorded solar/irradiance data')

            albedo = info.get('surface_albedo')
            albedo_source = info.get('albedo_source')
            if getattr(u, 'weather_source', None) == 'nsrdb_multiyear':
                if albedo is None or albedo_source != 'nsrdb_recorded':
                    raise ValueError(
                        'NSRDB dual-axis reference requires recorded per-step surface albedo'
                    )

            tilt, az = dual_axis_tracking_targets(
                info['solar_zenith_deg'], info['solar_azimuth_deg'])
            result = compute_pv_power(
                tilt, az,
                info['solar_zenith_deg'], info['solar_azimuth_deg'],
                info['dni_wm2'], info['ghi_wm2'], info['dhi_wm2'],
                area=u.area, efficiency=u.efficiency, power_model=u.power_model,
                temp_air_c=info.get('temperature_c', info.get('ambient_temperature_c')),
                wind_speed_ms=info.get('wind_speed_m_s'),
                wind_reference_height_m=u.wind_reference_height_m,
                gamma_pdc=u.gamma_pdc, sapm_mount=u.sapm_mount,
                wind_powerlaw_exponent=u.wind_powerlaw_exponent,
                surface_albedo=albedo,
            )
            power = float(result.power_w)
            if not np.isfinite(power) or power < 0:
                raise FloatingPointError('Invalid dual-axis reference DC power')
            energy = power * float(u.interval_hours) / 1000.0

            p['power'].append(power)
            p['energy'].append(energy)
            p['reward'].append(energy)
            p['cost'].append(0.0)
            p['tilt'].append(float(tilt))
            p['azimuth'].append(float(az))
            for key in ('action_tilt', 'action_azimuth',
                        'delta_tilt_deg', 'delta_azimuth_deg'):
                p[key].append(float('nan'))

            physical = dict(info)
            physical.update(
                power=power, power_dc_w=power,
                poa_global=float(result.poa_global_wm2),
                effective_irradiance_wm2=float(result.effective_irradiance_wm2),
                cell_temperature_c=result.cell_temperature_c,
                wind_speed_10m_ms=result.wind_speed_10m_ms,
                power_model=u.power_model, tilt=float(tilt), azimuth=float(az),
                surface_albedo=result.surface_albedo,
                albedo_source=albedo_source,
                cos_aoi=u._cos_angle_of_incidence(
                    info['solar_zenith_deg'], info['solar_azimuth_deg'], tilt, az),
            )
            p['infos'].append(physical)

        p['gross_energy_kwh'] = float(sum(p['energy']))
        p['net_energy_kwh'] = None
        p['movement_cost_kwh'] = None
        return p
    finally:
        if owns:
            env.close()
