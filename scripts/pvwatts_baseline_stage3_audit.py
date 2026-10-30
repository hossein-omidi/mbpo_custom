"""Read-only Stage 3 audit of fixed, sun-pointing and DC-optimum comparators.

Requires the project's NSRDB manifest/data and pvlib 0.13.1 in simple-worm.
Does not create or modify training/checkpoint/evaluation artifacts.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def independent_dc(info, tilt, azimuth, env):
    """Independent pvlib computation (does not call project power helper)."""
    from pvlib import atmosphere, temperature, pvsystem, irradiance
    u=env.unwrapped
    arr=irradiance.get_total_irradiance(
        surface_tilt=float(tilt),surface_azimuth=float(azimuth),
        solar_zenith=float(info['solar_zenith_deg']),
        solar_azimuth=float(info['solar_azimuth_deg']),
        dni=float(info['dni_wm2']),ghi=float(info['ghi_wm2']),
        dhi=float(info['dhi_wm2']),model='isotropic')
    poa=max(float(arr['poa_global']),0.)
    if u.power_model=='legacy':
        return poa*float(u.area)*float(u.efficiency),poa,None
    wind_10=float(atmosphere.windspeed_powerlaw(
        float(info['wind_speed_m_s']),height_reference=float(u.wind_reference_height_m),
        height_desired=10.,exponent=float(u.wind_powerlaw_exponent)))
    cell=float(temperature.sapm_cell(
        poa_global=poa,temp_air=float(info['ambient_temperature_c']),
        wind_speed=wind_10,
        **temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][u.sapm_mount]))
    watts=float(pvsystem.pvwatts_dc(poa,cell,
        1000.*float(u.area)*float(u.efficiency),float(u.gamma_pdc),temp_ref=25.))
    return max(watts,0.),poa,cell


def gate():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',default='2022-07-15')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--power-model',choices=['legacy','pvwatts_sapm'],default='pvwatts_sapm')
    p.add_argument('--wind-reference-height-m',type=float,default=2.0)
    p.add_argument('--gamma-pdc',type=float,default=-.004)
    p.add_argument('--sapm-mount',default='open_rack_glass_polymer')
    p.add_argument('--wind-powerlaw-exponent',type=float,default=.14)
    p.add_argument('--sac-checkpoint',default=None)
    p.add_argument('--mbpo-checkpoint',default=None)
    a=p.parse_args()
    from MBPO_workspace.src.config import load_config
    from MBPO_workspace.src.evaluation import ideal_instantaneous_poa, evaluate
    from SAC_workspace.src.environment.environment import SACEnvironment
    from SAC_workspace.src.evaluation.evaluation import (run_episode,ideal_instantaneous_dual_axis)
    from mbpo.env.baseline_contract import validate_paired_reference_group
    cfg=load_config(power_model=a.power_model,wind_reference_height_m=a.wind_reference_height_m,
        gamma_pdc=a.gamma_pdc,sapm_mount=a.sapm_mount,
        wind_powerlaw_exponent=a.wind_powerlaw_exponent)
    if bool(a.sac_checkpoint)!=bool(a.mbpo_checkpoint):
        p.error('Supply both checkpoints together for full SAC/MBPO validation')
    if a.sac_checkpoint:
        paths=evaluate(cfg,a.mbpo_checkpoint,a.sac_checkpoint,
            scenarios=[a.scenario],seeds=(a.seed,),device='cpu',
            include_poa=True,include_classical=True)
    else:
        env=SACEnvironment(cfg.sac.eval_env)
        try:
            fixed=run_episode(cfg.sac,'fixed_no_motion',a.scenario,a.seed,env=env)
            slew=run_episode(cfg.sac,'dual_axis_slew_limited',a.scenario,a.seed,env=env)
            sun=ideal_instantaneous_dual_axis(cfg.sac,fixed,env=env)
            ideal=ideal_instantaneous_poa(cfg.sac,fixed,env=env)
            paths=[fixed,slew,sun,ideal]
        finally:
            env.close()
    group={x['method']:x for x in paths}
    print('[PASS] Baseline pairing and upper-bound contract:',
          validate_paired_reference_group(group,require_oracle=True))
    env=SACEnvironment(cfg.sac.eval_env)
    try:
        for method,path in group.items():
            for i,info in enumerate(path['infos']):
                watts,poa,cell=independent_dc(info,path['tilt'][i],path['azimuth'][i],env)
                if not math.isclose(watts,float(path['power'][i]),rel_tol=1e-8,abs_tol=1e-5):
                    raise AssertionError(f'Independent DC parity: {method}, step {i}')
                if not math.isclose(poa,float(info['poa_global']),rel_tol=1e-8,abs_tol=1e-5):
                    raise AssertionError(f'Independent POA parity: {method}, step {i}')
                if cell is not None and not math.isclose(cell,float(info['cell_temperature_c']),
                       rel_tol=1e-8,abs_tol=1e-5):
                    raise AssertionError(f'Independent temperature parity: {method}, step {i}')
        print('[PASS] Independent pvlib POA, SAPM temperature, PVWatts DC:',
              sum(len(x['infos']) for x in paths),'post-step samples')
        # Independent azimuth/tilt grid checks on a representative hourly grid.
        oracle=group['poa_instant_ideal']
        for t in range(0,len(oracle['power']),12):
            info=oracle['infos'][t]
            sunaz=float(info['solar_azimuth_deg'])%360.
            for tilt in (0,15,30,45,60,75,90):
                for az in (sunaz,(sunaz+90.)%360.,(sunaz+180.)%360.,0.,180.):
                    candidate,_,_=independent_dc(info,tilt,az,env)
                    tol=max(1e-5,abs(candidate)*1e-7)
                    if candidate>oracle['power'][t]+tol:
                        raise AssertionError(f'Unconstrained DC oracle fails 2D grid: step={t}, '
                                             f'pose=({tilt},{az})')
        print('[PASS] Independent two-axis sampled grid below certified DC upper bound')
    finally:
        env.close()
    for name,path in group.items():
        print(f'{name:26s} gross={path["gross_energy_kwh"]:.6f} kWh '
              f'net={path["net_energy_kwh"]}')
    print('STAGE 3 BASELINE AUDIT: PASS (physical consistency and gross upper bound, '
          'not proof of controller quality or calibrated PV accuracy)')


if __name__=='__main__':
    try:
        gate()
    except Exception as error:
        print('STAGE 3 BASELINE AUDIT: FAIL:',type(error).__name__,str(error))
        raise
