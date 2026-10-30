#!/usr/bin/env python3
"""Read-only candidate DC-power comparison on actual 117-step NSRDB episodes.

The REAL shared PVTrackingEnv remains legacy; its recorded orientation/weather
trajectory is rescored offline using the opt-in new backend. This is a
trajectory sensitivity check, not a trained-policy or field-accuracy comparison.

From repo root (the test does not modify simulator, training files or runs):
  python tests/pvwatts_daily_sensitivity.py --wind-reference-height-m 2
  python tests/pvwatts_daily_sensitivity.py --wind-reference-height-m 2 --json /tmp/pvw_stage1.json

The wind reference height and power-law exponent are provisional assumptions
until independently confirmed from source metadata/site characterization.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path.cwd().resolve()
if not (ROOT / 'SAC_workspace').is_dir():
    raise SystemExit('Run from the mbpo_custom repository root')
sys.path.insert(0, str(ROOT))

from SAC_workspace.src.config.config import load_config
from SAC_workspace.src.environment.environment import SACEnvironment
from mbpo.env.pv_power_backend import compute_pv_power

DAYS = ('2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15')


def selected_action(step, horizon, policy):
    if policy == 'fixed':
        return np.zeros(2,dtype=np.float32)
    if policy == 'slew':
        return np.array([0.75*math.sin(2*math.pi*step/horizon),
                         0.80*math.cos(2*math.pi*step/horizon)],dtype=np.float32)
    raise ValueError(policy)


def run_day(cfg, day, policy, args):
    kwargs = dict(cfg.train_env)
    kwargs.update(randomize_initial_orientation=False,
                  irradiance_perturbation_std=0., observation_noise_std=0.,
                  movement_penalty=0.)
    # In particular, DO NOT modify power_model or any solver feature here.
    if 'power_model' in kwargs:
        raise RuntimeError('Comparison needs an unchanged legacy simulator, not an active PVWatts config')
    env = SACEnvironment(kwargs)
    try:
        env.seed(2026)
        env.reset(scenario_id=day)
        legacy_kwh, candidate_kwh = 0., 0.
        max_error_w = 0.
        max_poa_error = 0.
        tcells = []
        for step in range(env.horizon):
            action = selected_action(step, env.horizon, policy)
            _, _, done, info = env.step(action)
            u=env.unwrapped
            wind_ms = float(u.weather_profile.iloc[u.step_index]['wind_speed'])
            power = compute_pv_power(
                surface_tilt=float(info['tilt']),
                surface_azimuth=float(info['azimuth']),
                solar_zenith=float(info['solar_zenith_deg']),
                solar_azimuth=float(info['solar_azimuth_deg']),
                dni=float(info['dni_wm2']),
                ghi=float(info['ghi_wm2']),
                dhi=float(info['dhi_wm2']),
                area=float(u.area),efficiency=float(u.efficiency),
                power_model='pvwatts_sapm',
                temp_air_c=float(info['temperature_c']),
                wind_speed_ms=wind_ms,
                wind_reference_height_m=args.wind_reference_height_m,
                gamma_pdc=args.gamma_pdc,
                sapm_mount=args.sapm_mount,
                wind_powerlaw_exponent=args.wind_powerlaw_exponent)
            check_old = power.poa_global_wm2 * u.area * u.efficiency
            max_error_w=max(max_error_w,abs(check_old - float(info['power'])))
            max_poa_error=max(max_poa_error,abs(power.poa_global_wm2 - float(info['poa_global'])))
            if info['scenario_id'] != day:
                raise AssertionError(f'Scenario changed from {day}: {info["scenario_id"]}')
            if done != (step==env.horizon-1):
                raise AssertionError(f'Episode ended at wrong step: {step+1}')
            legacy_kwh+=float(info['energy_kwh'])
            candidate_kwh+=power.power_w*float(info['interval_hours'])/1000.
            tcells.append(power.cell_temperature_c)
        if env.horizon!=117 or max_error_w>1e-6 or max_poa_error>1e-6:
            raise AssertionError(f'Legacy simulator/POA contract violated on {day}/{policy}; '
                                 f'power_error={max_error_w}, poa_error={max_poa_error}')
        return {'scenario_id':day,'policy':policy,'steps':env.horizon,
                'legacy_kwh':legacy_kwh,'pvwatts_kwh':candidate_kwh,
                'difference_percent':100.*(candidate_kwh/legacy_kwh-1.) if legacy_kwh>0 else None,
                'mean_cell_temp_c':float(np.mean(tcells)),
                'max_legacy_power_error_w':max_error_w,
                'max_poa_error_wm2':max_poa_error}
    finally:
        env.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenarios',nargs='+',default=list(DAYS))
    ap.add_argument('--policies',nargs='+',choices=['fixed','slew'],default=['fixed','slew'])
    ap.add_argument('--wind-reference-height-m',type=float,required=True,
                    help='Wind measurement height in NSRDB file (often 2 m; verify)')
    ap.add_argument('--wind-powerlaw-exponent',type=float,default=0.14,
                    help='Assumed terrain-adjustment exponent; document/calibrate')
    ap.add_argument('--gamma-pdc',type=float,default=-0.004,
                    help='Example module temperature coefficient (1/C)')
    ap.add_argument('--sapm-mount',default='open_rack_glass_polymer',
                    help='Example SAPM temperature preset')
    ap.add_argument('--json',type=Path,default=None,
                    help='Optional output path; default produces no files')
    args=ap.parse_args()
    cfg=load_config(require_data=True)
    if len(set(args.scenarios))!=len(args.scenarios):
        ap.error('Duplicate scenarios')
    print('Provisional parameters: reference wind height={}m, exponent={}, gamma={}, mount={}'.format(
          args.wind_reference_height_m,args.wind_powerlaw_exponent,args.gamma_pdc,args.sapm_mount))
    print('day        policy   legacy_kWh    PVWatts_kWh      delta_%    mean_Tcell_C  max_old_err_W')
    rows=[]
    for day in args.scenarios:
        for policy in args.policies:
            r=run_day(cfg,day,policy,args)
            rows.append(r)
            print('{scenario_id} {policy:6s}   {legacy_kwh:.6f}      {pvwatts_kwh:.6f}     {difference_percent:+8.3f}      {mean_cell_temp_c:8.2f}     {max_legacy_power_error_w:.2e}'.format(**r))
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps({
           'experiment':'temperature-only sensitivity under legacy orientation trajectories',
           'status':'PASS',
           'assumptions':{'wind_reference_height_m':args.wind_reference_height_m,
                          'wind_powerlaw_exponent':args.wind_powerlaw_exponent,
                          'gamma_pdc':args.gamma_pdc,'sapm_mount':args.sapm_mount,
                          'effective_irradiance':'poa_global (no IAM/spectral)'},
           'rows':rows},indent=2,allow_nan=False),encoding='utf-8')
        print('Saved',args.json)
    print('STAGE 1 SENSITIVITY PASS: no simulator/train/evaluation source modified.')


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'STAGE 1 SENSITIVITY FAIL: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1)
