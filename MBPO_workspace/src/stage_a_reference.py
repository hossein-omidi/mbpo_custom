"""Correct paired instantaneous gross-DC reference for Stage A.

The legacy isotropic-POA analytical maximum must receive the SAME per-step
NSRDB surface albedo as SAC, MBPO, and fixed. No extra environmental samples,
weather substitutions, controller actions or movement costs enter the oracle.
"""
from __future__ import annotations

import math
import numpy as np
from mbpo.env.pv_power_backend import compute_pv_power
from mbpo.env.baseline_contract import certify_pvwatts_monotonicity
from SAC_workspace.src.environment.environment import SACEnvironment


def isotropic_max_tilt_with_albedo(solar_zenith, dni, ghi, dhi, albedo):
    """Exact maximizer of monofacial isotropic POA with solar-facing azimuth.

    For sun above horizon and 0 <= beta <= 90 degrees,
    POA = DNI cos(z-beta) + DHI*(1+cos(beta))/2
                          + albedo*GHI*(1-cos(beta))/2
        = constant + C*cos(beta) + S*sin(beta).
    Direct beam is nonnegative on the optimization domain.
    """
    z,dn,gh,dh,rho=map(float,(solar_zenith,dni,ghi,dhi,albedo))
    if not all(map(math.isfinite,(z,dn,gh,dh,rho))):
        raise ValueError('Nonfinite POA oracle input')
    if not 0<=z<=180 or min(dn,gh,dh)<0 or not 0<=rho<=1:
        raise ValueError('Invalid solar geometry/irradiance/albedo in POA oracle')
    if z>90 and dn>1e-8:
        raise ValueError('Nonzero direct normal irradiance with sun below horizon')
    if z>90:dn=0.0
    zr=math.radians(z)
    C=dn*math.cos(zr)+(dh-rho*gh)/2.0
    S=dn*math.sin(zr)
    candidates=[0.0,math.pi/2]
    angle=math.atan2(S,C)
    if 0<angle<math.pi/2:candidates.append(angle)
    b=max(candidates,key=lambda x:C*math.cos(x)+S*math.sin(x))
    return math.degrees(b)


def ideal_instantaneous_poa(cfg, fixed_path, env=None):
    """Gross instantaneous no-slew POA/DC upper reference on matched NSRDB.

    Under verified monotone PVWatts/SAPM (or constant-efficiency legacy),
    maximizing isotropic POA also maximizes DC power. Does not optimize
    finite-horizon rewards or provide a feasible dual-axis actuator policy.
    """
    owns=env is None
    if owns:env=SACEnvironment(cfg.eval_env)
    p={'method':'poa_instant_ideal','scenario_id':fixed_path['scenario_id'],
       'seed':fixed_path['seed'],'initial_pose':list(fixed_path['initial_pose']),
       'reward':[],'energy':[],'power':[],'cost':[],
       'tilt':[],'azimuth':[],'action_tilt':[],'action_azimuth':[],
       'delta_tilt_deg':[],'delta_azimuth_deg':[],'infos':[]}
    try:
        u=getattr(env,'unwrapped',env)
        hours=float(u.interval_hours)
        infos=fixed_path['infos']
        if not infos or len(infos)!=len(fixed_path['energy']):
            raise ValueError('Paired fixed trajectory is missing or incomplete')
        for info in infos:
            for key in ('solar_zenith_deg','solar_azimuth_deg','dni_wm2','ghi_wm2',
                        'dhi_wm2','surface_albedo'):
                if info.get(key) is None:
                    raise ValueError(f'Paired NSRDB reference input missing: {key}')
            if u.weather_source=='nsrdb_multiyear' and info.get('albedo_source')!='nsrdb_recorded':
                raise ValueError('Cannot certify NSRDB oracle with synthetic/fixed albedo')
            zenith=float(info['solar_zenith_deg']);solaraz=float(info['solar_azimuth_deg'])
            tilt=isotropic_max_tilt_with_albedo(
                zenith,info['dni_wm2'],info['ghi_wm2'],info['dhi_wm2'],info['surface_albedo'])
            az=solaraz%360.0
            result=compute_pv_power(
                tilt,az,zenith,solaraz,
                info['dni_wm2'],info['ghi_wm2'],info['dhi_wm2'],
                area=u.area,efficiency=u.efficiency,power_model=u.power_model,
                temp_air_c=info.get('temperature_c',info.get('ambient_temperature_c')),
                wind_speed_ms=info.get('wind_speed_m_s'),
                wind_reference_height_m=u.wind_reference_height_m,
                gamma_pdc=u.gamma_pdc,sapm_mount=u.sapm_mount,
                wind_powerlaw_exponent=u.wind_powerlaw_exponent,
                surface_albedo=info['surface_albedo'])
            if u.power_model=='pvwatts_sapm':
                certify_pvwatts_monotonicity(
                    result,info.get('temperature_c',info.get('ambient_temperature_c')),
                    u.gamma_pdc)
            power=float(result.power_w)
            if len(fixed_path['power'])!=len(infos):
                raise ValueError('Missing paired fixed-power samples')
            reference=float(fixed_path['power'][len(p['power'])])
            if power+max(1e-5,1e-7*abs(reference))<reference:
                raise RuntimeError('Albedo-aware oracle fell below fixed DC power')
            energy=power*hours/1000.
            p['power'].append(power);p['energy'].append(energy)
            p['reward'].append(energy);p['cost'].append(0.)
            p['tilt'].append(tilt);p['azimuth'].append(az)
            for key in ('action_tilt','action_azimuth','delta_tilt_deg','delta_azimuth_deg'):
                p[key].append(float('nan'))
            metadata=dict(info)
            metadata.update(power=power,power_dc_w=power,
                poa_global=result.poa_global_wm2,
                effective_irradiance_wm2=result.effective_irradiance_wm2,
                cell_temperature_c=result.cell_temperature_c,
                wind_speed_10m_ms=result.wind_speed_10m_ms,
                power_model=u.power_model,tilt=tilt,azimuth=az,
                surface_albedo=float(info['surface_albedo']),
                albedo_source=info['albedo_source'],
                cos_aoi=(u._cos_angle_of_incidence(zenith,solaraz,tilt,az)
                           if hasattr(u,'_cos_angle_of_incidence') else None))
            p['infos'].append(metadata)
    finally:
        if owns:env.close()
    p['gross_energy_kwh']=float(sum(p['energy']))
    p['net_energy_kwh']=None
    p['movement_cost_kwh']=None
    return p
