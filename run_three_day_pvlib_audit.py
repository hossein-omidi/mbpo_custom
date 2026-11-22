#!/usr/bin/env python3
"""Reproducible three-day Stage-A NSRDB/pvlib power-path audit.

Run from the root of hossein-omidi/mbpo_custom, PVwatt_dev:
  python run_three_day_pvlib_audit.py --start-date 2019-12-15 \
      --days 3 --outdir outputs/pvlib_three_day_audit

Requires original NSRDB 5-minute annual CSVs and manifest in repository,
plus repository's installed environment dependencies (pvlib, gym, etc.).

This invokes the *real* compact StageAPVTrackingEnv with zero control actions
for a fixed 30-degree/south-facing PV module, then calculates the instantaneous
sun-normal *non-actuated reference* using the same physics backend. It separately
checks backend values against direct pvlib calls. All actual outputs are from
NSRDB data. No artificial weather is substituted.
"""
from __future__ import annotations

import argparse
import json
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def _to_native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _append_row(scenario, env, fixed_power, fixed_result, direct_reference, *,
                reward=None, info=None, index=0, power_parity_error_w=0.0):
    from mbpo.env.pvlib_physics import dual_axis_tracking_targets
    from mbpo.env.pv_power_backend import compute_pv_power
    u = env.unwrapped
    now = pd.Timestamp(u.current_time)
    if str(now.tz) != 'UTC':
        raise AssertionError(f'Timestamps must be UTC; received {now}')
    src_weather = u.weather_profile.iloc[int(u.step_index)]
    weather = u._current_weather()
    alb, source = u._surface_albedo_at_current_time()
    if source != 'nsrdb_recorded':
        raise AssertionError(f'Expected recorded NSRDB albedo, got {source}')
    sol = u._solar_position(now)
    target_tilt, target_azimuth = dual_axis_tracking_targets(sol.zenith, sol.azimuth)
    ideal_result = compute_pv_power(
        target_tilt, target_azimuth, sol.zenith, sol.azimuth,
        weather['dni'], weather['ghi'], weather['dhi'],
        area=u.area, efficiency=u.efficiency, power_model=u.power_model,
        temp_air_c=weather['temperature'], wind_speed_ms=weather['wind_speed'],
        wind_reference_height_m=u.wind_reference_height_m,
        gamma_pdc=u.gamma_pdc, sapm_mount=u.sapm_mount,
        wind_powerlaw_exponent=u.wind_powerlaw_exponent,
        surface_albedo=alb,
    )
    # Irradiance closure is a diagnostic, not a strict identity for all satellite
    # estimates / temporal averages. It must be interpreted with source quality.
    cos_z = max(float(np.cos(np.deg2rad(sol.zenith))), 0.0)
    closure_error = weather['ghi'] - weather['dhi'] - weather['dni'] * cos_z
    return {
        'scenario_id': str(scenario), 'timestamp_utc': now.isoformat(),
        'within_episode_step': int(index),
        'dni_wm2': float(weather['dni']), 'dhi_wm2': float(weather['dhi']),
        'ghi_wm2': float(weather['ghi']),
        'temperature_c': float(weather['temperature']),
        'wind_speed_m_s': float(weather['wind_speed']),
        'albedo': float(alb), 'solar_zenith_deg': float(sol.zenith),
        'solar_azimuth_deg': float(sol.azimuth),
        'fixed_tilt_deg': float(u.tilt), 'fixed_azimuth_deg': float(u.azimuth),
        'poa_fixed_wm2': float(fixed_result.poa_global_wm2),
        'cell_temp_fixed_c': float(fixed_result.cell_temperature_c),
        'power_fixed_dc_w': float(fixed_power),
        'poa_sun_normal_wm2': float(ideal_result.poa_global_wm2),
        'power_sun_normal_dc_w': float(ideal_result.power_w),
        'closure_residual_wm2': float(closure_error),
        'direct_pvlib_power_parity_error_w': float(power_parity_error_w),
        'reward_kwh': None if reward is None else float(reward),
        'movement_cost_kwh': None if info is None else float(info['movement_cost']),
    }


def independent_pvlib_power(surface_tilt, surface_azimuth, zenith, azimuth,
                             weather, albedo, u):
    """Call pvlib independently instead of calling repository compute_pv_power.

    Verifies input wiring and algebraic parity with pvlib, NOT agreement with
    measured field power (which requires external field observations).
    """
    from pvlib import atmosphere, irradiance, pvsystem, temperature
    parts = irradiance.get_total_irradiance(
        surface_tilt=float(surface_tilt), surface_azimuth=float(surface_azimuth),
        solar_zenith=float(zenith), solar_azimuth=float(azimuth),
        dni=float(weather['dni']), ghi=float(weather['ghi']),
        dhi=float(weather['dhi']), albedo=float(albedo), model='isotropic',
    )
    poa = max(float(parts['poa_global']), 0.0)
    if u.power_model == 'legacy':
        return poa * float(u.area) * float(u.efficiency)
    if u.power_model != 'pvwatts_sapm':
        raise ValueError(f'Unexpected configured DC backend {u.power_model}')
    wind10 = atmosphere.windspeed_powerlaw(
        float(weather['wind_speed']),
        height_reference=float(u.wind_reference_height_m),
        height_desired=10.0, exponent=float(u.wind_powerlaw_exponent))
    params = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][u.sapm_mount]
    cell_temp = temperature.sapm_cell(
        poa_global=poa, temp_air=float(weather['temperature']),
        wind_speed=wind10, **params)
    power = pvsystem.pvwatts_dc(
        effective_irradiance=poa, temp_cell=cell_temp,
        pdc0=1000.0*float(u.area)*float(u.efficiency),
        gamma_pdc=float(u.gamma_pdc), temp_ref=25.0)
    return max(float(power), 0.0)


def _run_day(sid, env):
    env.reset(scenario_id=sid)
    u = env.unwrapped
    assert u.weather_source == 'nsrdb_multiyear'
    assert u.periods == 196 and env.horizon == 195, (
        f'The audited PVwatt_dev episode expects 196 timestamps / 195 transitions, '
        f'got {u.periods} / {env.horizon}. Review configuration and script.')
    assert u.power_model == 'pvwatts_sapm', (
        f'Unexpected power backend {u.power_model!r}. This script audits pvwatts_sapm.')
    if abs(u.tilt-30.0)>1e-8 or abs(u.azimuth-180.0)>1e-8:
        raise RuntimeError('Evaluation must begin at fixed 30-degree south orientation')
    rows=[]
    for step in range(env.horizon+1):
        if step == 0:
            rew, info = None, None
        else:
            _, rew, done, info = env.step(np.zeros(2,dtype=np.float32))
            if bool(done) != (step == env.horizon):
                raise AssertionError('Unexpected episode termination')
            if abs(float(info['movement_cost']))>1e-12:
                raise AssertionError('Zero movement must have zero movement charge')
        result = u._last_power_result
        weather = u._current_weather()
        alb,src = u._surface_albedo_at_current_time()
        sol = u._solar_position(u.current_time)
        parity = independent_pvlib_power(
            u.tilt,u.azimuth,sol.zenith,sol.azimuth,weather,alb,u)
        error = abs(float(u.last_power)-parity)
        if not np.isclose(u.last_power,parity,rtol=1e-8,atol=1e-6):
            raise AssertionError(
                f'pvlib direct parity failure {sid} step={step} error={error:.6g} W')
        rows.append(_append_row(sid,env,u.last_power,result,parity,
                 reward=rew, info=info,index=step,power_parity_error_w=error))
    return rows


def _figures(frame, outdir):
    grouped=list(frame.groupby('scenario_id',sort=True))
    def _dateaxis(ax):
        locator = mdates.HourLocator(interval=8, tz=timezone.utc)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M UTC', tz=timezone.utc))
        ax.grid(True,alpha=.25)
    fig,ax=plt.subplots(figsize=(13.0,5.4))
    for n,(sid,d) in enumerate(grouped):
        t=pd.to_datetime(d['timestamp_utc'],utc=True)
        for col,label in [('ghi_wm2','GHI'),('dni_wm2','DNI'),('dhi_wm2','DHI'),
                          ('poa_fixed_wm2','POA fixed, 30° South')]:
            ax.plot(t,d[col],label=label if n==0 else None,linewidth=1.55)
    ax.set_ylabel('Irradiance (W/m²)')
    ax.set_title('NSRDB PSM v4 irradiance / fixed-panel POA, 3 consecutive PV episodes')
    _dateaxis(ax);ax.legend(ncol=4,fontsize=9)
    fig.tight_layout();fig.savefig(outdir/'three_day_irradiance.png',dpi=200);plt.close(fig)

    fig,ax=plt.subplots(figsize=(13.0,5.4))
    for n,(sid,d) in enumerate(grouped):
        t=pd.to_datetime(d['timestamp_utc'],utc=True)
        ax.plot(t,d['power_fixed_dc_w'],label='Fixed 30° South' if n==0 else None,linewidth=1.7)
        ax.plot(t,d['power_sun_normal_dc_w'],label='Instantaneous sun-normal (no slew/cost)' if n==0 else None,
                linewidth=1.4,linestyle='--')
    ax.set_ylabel('PV DC power (W)')
    ax.set_title('PVWatts / SAPM module power under recorded NSRDB weather (not measured power)')
    _dateaxis(ax);ax.legend()
    fig.tight_layout();fig.savefig(outdir/'three_day_dc_power.png',dpi=200);plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(13,7.5),sharex=True)
    for n,(sid,d) in enumerate(grouped):
        t=pd.to_datetime(d['timestamp_utc'],utc=True)
        axes[0].plot(t,d['temperature_c'],label='Ambient' if n==0 else None)
        axes[0].plot(t,d['cell_temp_fixed_c'],label='Cell (SAPM)' if n==0 else None)
        axes[1].plot(t,d['wind_speed_m_s'],label='NSRDB wind' if n==0 else None)
    axes[0].set_ylabel('Temperature (°C)');axes[0].legend();axes[0].grid(alpha=.25)
    axes[1].set_ylabel('Wind speed (m/s)');axes[1].legend();_dateaxis(axes[1])
    fig.suptitle('Recorded meteorology and computed fixed-panel cell temperature')
    fig.tight_layout();fig.savefig(outdir/'three_day_temperature_wind.png',dpi=200);plt.close(fig)


def summarize(frame, horizon):
    summary={}
    for sid,d in frame.groupby('scenario_id',sort=True):
        d=d.sort_values('within_episode_step')
        dt_hours=5.0/60.0
        p=d['power_fixed_dc_w'].to_numpy(float)
        # Environment reward uses RIGHT endpoint power after each 5-min action:
        # sum over 195 intervals P(t_1)...P(t_195) × 5/60 /1000.
        right=float(np.sum(p[1:])*dt_hours/1000.0)
        trap=float((.5*p[0]+np.sum(p[1:-1])+.5*p[-1])*dt_hours/1000.0)
        rewards=float(d['reward_kwh'].dropna().sum())
        if not np.isclose(right,rewards,atol=1e-9,rtol=1e-9):
            raise AssertionError(f'Environment reward integration mismatch {sid}')
        ideal=float(np.sum(d['power_sun_normal_dc_w'].to_numpy(float)[1:])*dt_hours/1000.0)
        summary[sid]={
            'num_timestamps':int(len(d)), 'num_control_transitions':int(len(d)-1),
            'net_fixed_zero_motion_kwh':rewards,
            'gross_fixed_right_endpoint_kwh':right,
            'gross_fixed_trapezoidal_diagnostic_kwh':trap,
            'right_minus_trapezoidal_kwh':right-trap,
            'gross_sun_normal_no_actuator_kwh':ideal,
            'mean_ghi_wm2':float(d['ghi_wm2'].mean()),
            'mean_dni_wm2':float(d['dni_wm2'].mean()),
            'mean_dhi_wm2':float(d['dhi_wm2'].mean()),
            'median_abs_irradiance_closure_residual_wm2':float(d['closure_residual_wm2'].abs().median()),
            'max_abs_irradiance_closure_residual_wm2':float(d['closure_residual_wm2'].abs().max()),
            'max_independent_pvlib_power_parity_error_w':float(d['direct_pvlib_power_parity_error_w'].max()),
        }
        if len(d)!=horizon+1:
            raise AssertionError('Incomplete episode')
    return summary


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start-date',default='2019-12-15',help='YYYY-MM-DD, first of 3 or more consecutive NSRDB scenarios')
    ap.add_argument('--days',type=int,default=3,help='Number of consecutive scenario days; default 3')
    ap.add_argument('--outdir',default='outputs/pvlib_three_day_audit')
    args=ap.parse_args(argv)
    if args.days<1:ap.error('--days must be at least 1')
    start=pd.Timestamp(args.start_date)
    if start.tzinfo is not None:ap.error('Use YYYY-MM-DD without a timezone for scenario IDs')
    sids=[(start+pd.Timedelta(days=i)).date().isoformat() for i in range(args.days)]

    from SAC_workspace.src.config.config import load_config
    from SAC_workspace.src.environment.environment import SACEnvironment
    cfg=load_config(require_data=True)
    params=dict(cfg.eval_env)
    params.update(randomize_scenario=False,randomize_initial_orientation=False,
                  fixed_eval_scenarios=None)
    env=SACEnvironment(params)
    out=Path(args.outdir).resolve();out.mkdir(parents=True,exist_ok=True)
    try:
        available=set(env.unwrapped._nsrdb_by_id)
        missing=[sid for sid in sids if sid not in available]
        if missing:raise ValueError(f'Missing scenario IDs from NSRDB manifest: {missing}')
        rows=[]
        for sid in sids:rows.extend(_run_day(sid,env))
        frame=pd.DataFrame.from_records(rows)
        frame.to_csv(out/'three_day_pvlib_audit.csv',index=False)
        report={
            'source':'NSRDB local multi-year CSV, version from repository manifest',
            'site':{'latitude':env.unwrapped.location.latitude,
                    'longitude':env.unwrapped.location.longitude},
            'power_backend':env.unwrapped.power_model,
            'model_type':'recorded-weather conditional DC simulation; not field forecast verification',
            'poa_model':'isotropic, recorded 5-min NSRDB surface albedo',
            'sun_normal_comparison':'instantaneous geometric pointing, no slew constraint or motor cost',
            'periods':env.unwrapped.periods,'horizon':env.horizon,
            'interval_minutes':env.unwrapped.interval_hours*60,
            'physical_parameters':{
                'area_m2':env.unwrapped.area,'efficiency_STC':env.unwrapped.efficiency,
                'power_nominal_STC_W':1000*env.unwrapped.area*env.unwrapped.efficiency,
                'wind_reference_height_m':env.unwrapped.wind_reference_height_m,
                'sapm_mount':env.unwrapped.sapm_mount,
                'gamma_pdc_per_C':env.unwrapped.gamma_pdc,
                'wind_powerlaw_exponent':env.unwrapped.wind_powerlaw_exponent,
            },
            'days':summarize(frame,env.horizon),
            'validation_limitations':[
                'Direct pvlib parity checks implementation consistency, not accuracy against field data',
                'NSRDB time series are satellite/reanalysis model products',
                'Right-endpoint energy integration is a five-minute discrete approximation',
                'Instantaneous sun-normal output is not a feasible actuator-limited policy',
            ],
        }
        with (out/'three_day_audit_summary.json').open('w',encoding='utf8') as f:
            json.dump(report,f,indent=2,allow_nan=False,default=_to_native)
        _figures(frame,out)
        print(json.dumps({'outdir':str(out),'scenario_ids':sids,
                          'days':report['days']},indent=2))
    finally:
        env.close()


if __name__=='__main__':
    main()
