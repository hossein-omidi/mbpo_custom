#!/usr/bin/env python3
"""Read-only audit for measured SAC/MBPO PV tracker trajectories.

Audits commanded normalized actions against *post-step* panel orientation,
including [0,90] tilt clipping and [0,360) azimuth wrapping. It does not
estimate convergence, infer statistical confidence intervals, or change data.

Usage:
 python pv_orientation_diagnostics.py --eval-dir MBPO_workspace/evaluation/scientific_v52_fresh --day 2024-12-23
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

METHODS = ('mbpo', 'sac', 'fixed_no_motion')
REQUIRED = ('scenario_id','seed','method','step','power_w','energy_kwh','tilt_deg',
            'azimuth_deg','pre_tilt_deg','pre_azimuth_deg','action_tilt','action_azimuth')


def wrap_delta(x):
    """Signed shortest azimuth difference in degrees, in [-180,180)."""
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def expected_pose(pre_tilt, pre_azimuth, action_tilt, action_azimuth,
                  max_delta_tilt=5.0, max_delta_azimuth=10.0):
    return (np.clip(np.asarray(pre_tilt, float) + max_delta_tilt * np.asarray(action_tilt,float), 0., 90.),
            (np.asarray(pre_azimuth,float) + max_delta_azimuth * np.asarray(action_azimuth,float)) % 360.)


def check_group(g, *, tilt_step=5., az_step=10., tol=3e-4):
    """Return physical transition audit metrics for one complete controller/day/seed."""
    g = g.sort_values('step')
    if len(g) != 117 or not np.array_equal(g['step'].to_numpy(), np.arange(1,118)):
        raise ValueError('Incomplete or unordered 117-step episode')
    for col in REQUIRED[3:]:
        if not np.isfinite(g[col].to_numpy(dtype=float)).all():
            raise ValueError(f'Missing or nonfinite {col}')
    tilt=g['tilt_deg'].to_numpy(float); az=g['azimuth_deg'].to_numpy(float)
    pre_tilt=g['pre_tilt_deg'].to_numpy(float); pre_az=g['pre_azimuth_deg'].to_numpy(float)
    a_t=g['action_tilt'].to_numpy(float); a_a=g['action_azimuth'].to_numpy(float)
    if (np.abs(a_t)>1.+tol).any() or (np.abs(a_a)>1.+tol).any():
        raise ValueError('Normalized commanded actions exceed [-1,1]')
    if ((tilt < -tol)|(tilt > 90.+tol)|(az < -tol)|(az >= 360.+tol)).any():
        raise ValueError('Actual panel angle violates physical orientation bounds')
    if (np.abs(tilt[:-1]-pre_tilt[1:])>tol).any() or (np.abs(wrap_delta(az[:-1]-pre_az[1:]))>tol).any():
        raise ValueError('Pre-step pose does not agree with previous post-step pose')
    expected_t, expected_a=expected_pose(pre_tilt,pre_az,a_t,a_a,tilt_step,az_step)
    te=np.abs(tilt-expected_t);ae=np.abs(wrap_delta(az-expected_a))
    if te.max()>tol or ae.max()>tol:
        i=int(np.argmax(np.maximum(te,ae)))
        raise ValueError(f'Action/pose mismatch at step {i+1}: tilt_err={te[i]:.6g} deg, az_err={ae[i]:.6g} deg')
    if not np.allclose(g['energy_kwh'].to_numpy(float),g['power_w'].to_numpy(float)/12000.,atol=1e-6,rtol=1e-5):
        raise ValueError('Power to five-minute kWh conversion mismatch')
    # Outward commands at a hard tilt stop do not create additional rotation.
    clipped = np.abs(pre_tilt + tilt_step*a_t - expected_t)>tol
    signs=np.sign(a_t)
    adjacent_reversals=((np.abs(a_t[:-1])>=.5)&(np.abs(a_t[1:])>=.5)&(signs[:-1]!=signs[1:])).sum()
    return {'max_tilt_mapping_error_deg':float(te.max()),'max_azimuth_mapping_error_deg':float(ae.max()),
            'tilt_saturated_action_fraction':float(np.mean(np.abs(a_t)>=.95)),
            'azimuth_saturated_action_fraction':float(np.mean(np.abs(a_a)>=.95)),
            'tilt_hard_stop_fraction':float(np.mean((tilt<=tol)|(tilt>=90.-tol))),
            'tilt_outward_clipped_commands':int(clipped.sum()),
            'strong_adjacent_tilt_action_reversals':int(adjacent_reversals),
            'gross_energy_kwh':float(g['energy_kwh'].sum())}


def audit(root, day=None, tilt_step=5., az_step=10.):
    root=Path(root)
    f=root/'trajectories_detailed.csv'
    if not f.is_file():
        raise FileNotFoundError(f'Requires detailed action/pose measurements: {f}')
    df=pd.read_csv(f,low_memory=False)
    missing=set(REQUIRED)-set(df.columns)
    if missing:
        raise ValueError(f'Missing measured data columns: {sorted(missing)}')
    df['seed']=pd.to_numeric(df['seed'],errors='raise').astype(int)
    df['step']=pd.to_numeric(df['step'],errors='raise').astype(int)
    evaluations=[]
    failures=[]
    # Complete groups cannot be silently skipped.
    for (sid,seed,method),g in df[df['method'].isin(METHODS)].groupby(['scenario_id','seed','method'],sort=True):
        try:
            results=check_group(g,tilt_step=tilt_step,az_step=az_step)
            evaluations.append({'scenario_id':str(sid),'seed':int(seed),'method':method,**results})
        except (ValueError,KeyError) as e:
            failures.append({'scenario_id':str(sid),'seed':int(seed),'method':method,'error':str(e)})
    records=pd.DataFrame(evaluations)
    if records.empty:
        raise ValueError('No valid measured controller trajectories')
    if failures:
        raise ValueError(f'{len(failures)} trajectory failures; first: {failures[:5]}')
    first=df[(df['step']==1)&(df['method'].isin(METHODS))]
    pose=first.pivot(index=['scenario_id','seed'],columns='method',values=['pre_tilt_deg','pre_azimuth_deg'])
    for method in METHODS:
        if method not in pose['pre_tilt_deg'] or method not in pose['pre_azimuth_deg']:
            raise ValueError(f'Initial pose comparison is missing {method}')
        if ((np.abs(pose['pre_tilt_deg'][method]-pose['pre_tilt_deg']['fixed_no_motion'])>3e-4).any()
            or (np.abs(wrap_delta(pose['pre_azimuth_deg'][method]-pose['pre_azimuth_deg']['fixed_no_motion']))>3e-4).any()):
            raise ValueError(f'Method {method} uses a different paired initial orientation')
    # The net comparison is computed from actual measured rewards when available.
    if 'net_step_kwh' in df.columns:
        df['net_step_kwh']=pd.to_numeric(df['net_step_kwh'],errors='coerce')
    else:
        df['net_step_kwh']=np.nan
    if 'movement_cost_kwh' not in df.columns:
        raise ValueError('Missing movement cost: cannot compare net electrical energy')
    df['movement_cost_kwh']=pd.to_numeric(df['movement_cost_kwh'],errors='raise')
    feasible=df[df.method.isin(METHODS)].copy()
    reconstructed=feasible['energy_kwh'].to_numpy(float)-feasible['movement_cost_kwh'].to_numpy(float)
    observed=feasible['net_step_kwh'].to_numpy(float)
    if np.isfinite(observed).any() and (not np.isfinite(observed).all() or not np.allclose(observed,reconstructed,atol=1e-6)):
        raise ValueError('Recorded net step rewards are incomplete or not energy minus movement')
    feasible['net']=reconstructed
    totals=feasible.groupby(['scenario_id','seed','method'],sort=True)['net'].sum().unstack('method')
    if set(METHODS)-set(totals.columns) or totals[list(METHODS)].isna().any().any():
        raise ValueError('Each scenario/seed must have all three feasible methods')
    result={'source':str(f),'n_distinct_dates':int(totals.reset_index()['scenario_id'].nunique()),
            'groups_verified':len(records),
            'max_action_tilt_mapping_error_deg':float(records.max_tilt_mapping_error_deg.max()),
            'max_action_azimuth_mapping_error_deg':float(records.max_azimuth_mapping_error_deg.max()),
            'by_method':{},'worst_five_days_by_method':{}}
    for method in METHODS:
        rec=records[records.method==method]
        result['by_method'][method]={k:float(rec[k].mean()) for k in (
            'tilt_saturated_action_fraction','azimuth_saturated_action_fraction',
            'tilt_hard_stop_fraction','tilt_outward_clipped_commands',
            'strong_adjacent_tilt_action_reversals')}
    for method in ('mbpo','sac'):
        diff=totals[method]-totals['fixed_no_motion']
        result['worst_five_days_by_method'][method]=[
            {'scenario_id':str(sid),'seed':int(seed),'net_gain_kwh':float(v)}
            for (sid,seed),v in diff.nsmallest(5).items()]
    if 'poa_instant_ideal' in set(df.method):
        power=df[df.method.isin(('poa_instant_ideal',)+METHODS)].pivot(
            index=['scenario_id','seed','step'],columns='method',values='power_w')
        if (set(METHODS)|{'poa_instant_ideal'}) - set(power.columns):
            raise ValueError('POA comparison methods incomplete')
        if len(power)!=len(totals)*117 or power[list(METHODS)+['poa_instant_ideal']].isna().any().any():
            raise ValueError('POA oracle per-step pairing incomplete')
        for method in METHODS:
            gap=power['poa_instant_ideal']-power[method]
            violations=int((gap < -1e-3).sum())
            result.setdefault('poa_oracle',{})[method]={
                'oracle_power_below_feasible_step_count':violations,
                'mean_total_gross_gap_kwh':float(gap.groupby(level=['scenario_id','seed']).sum().mean()/12000.)}
            if violations:
                raise ValueError(f'Instant POA below feasible {method} at {violations} post-step times')
    if day:
        dayrows=records[records.scenario_id==day]
        if dayrows.empty:raise ValueError(f'Date not found in sampled evaluation: {day}')
        result['selected_day']=day
        result['selected_day_controller_mechanics']=dayrows.to_dict('records')
        dd=totals.reset_index(); dd=dd[dd.scenario_id==day]
        result['selected_day_net_kwh']=dd.to_dict('records')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir',required=True)
    p.add_argument('--day',default=None)
    p.add_argument('--max-delta-tilt',type=float,default=5.)
    p.add_argument('--max-delta-azimuth',type=float,default=10.)
    p.add_argument('--output-json',default=None)
    a=p.parse_args()
    if a.max_delta_tilt<=0 or a.max_delta_azimuth<=0:
        p.error('Positive per-step angular bounds required')
    result=audit(a.eval_dir,day=a.day,tilt_step=a.max_delta_tilt,az_step=a.max_delta_azimuth)
    s=json.dumps(result,indent=2,allow_nan=False)
    print(s)
    if a.output_json:
        path=Path(a.output_json);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(s+'\n',encoding='utf-8')
    print('\nPASS: action-to-pose mapping and measured physical trajectory checks. '
          'This is NOT a training-convergence or unseen-generalization certificate.')


if __name__=='__main__':main()
