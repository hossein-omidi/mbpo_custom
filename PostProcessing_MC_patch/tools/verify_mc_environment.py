#!/usr/bin/env python3
# POSTPROCESSING_MC_CURVE_STATS_AND_VERIFY_FIX_V1
"""Real-data Monte Carlo environment integration test (NO training required).

Run from the root of the PostProcessing repo after applying the patch:

  python PostProcessing_MC_patch/tools/verify_mc_environment.py \
      --uncertainty-spec PostProcessing_MC_patch/examples/hypothetical_uncertainty_5pct.json \
      --out outputs/monte_carlo_environment_audit.json

Requires the local 2018–2024 NSRDB files, pvlib, Gym, PyTorch dependencies
and existing valid Stage-A environment. Never downloads or modifies sources.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from datetime import timedelta
import numpy as np


def one_episode(env,sid,rid):
    import pandas as pd
    env.seed(819)
    obs=env.reset(scenario_id=sid,realization_id=rid)
    u=env.unwrapped
    if env.horizon!=195 or obs.shape!=(6,) or env.action_space.shape!=(2,):
        raise AssertionError('Wrong observation/action dimensions or 195-step horizon')
    signature=str(u._weather_profile_sha256)
    if not signature or signature=='None':raise AssertionError('Missing weather fingerprint')
    baseline=u.weather_profile.copy(deep=True)
    power=[];gross=[];cost=[];clock=[];albedo=[]
    for t in range(env.horizon):
        obs,reward,done,info=env.step(np.zeros(2,dtype=np.float32))
        expected=u.times[t+1]
        actual=pd.Timestamp(info['timestamp_utc_iso'])
        if expected!=actual or actual.utcoffset().total_seconds()!=0:
            raise AssertionError('UTC timestamp differs from configured 5-minute profile')
        if t and (actual-clock[-1]).total_seconds()!=300:
            raise AssertionError('Wrong five-minute interval')
        if bool(done)!=(t==env.horizon-1):raise AssertionError('Wrong terminal signal')
        if info['weather_profile_sha256']!=signature:
            raise AssertionError('Weather realization changed within an episode')
        P=float(info['power_dc_w']); E=float(info['energy_kwh']); C=float(info['movement_cost'])
        if not all(np.isfinite(v) for v in (P,E,C,float(reward))):
            raise AssertionError('Nonfinite power/energy/reward')
        if not np.isclose(E,P*u.interval_hours/1000.,rtol=0,atol=1e-10):
            raise AssertionError('Power-to-energy conversion incorrect')
        if not np.isclose(float(reward),E-C,rtol=0,atol=1e-10):
            raise AssertionError('Net-energy reward identity failed')
        if float(info['dhi_wm2'])>float(info['ghi_wm2'])+1e-8:
            raise AssertionError('Physical GHI>=DHI bound violated')
        if not 0<=float(info['surface_albedo'])<=1:
            raise AssertionError('Albedo out of physical bounds')
        if any(float(info[x])<0 for x in ('dni_wm2','dhi_wm2','ghi_wm2','wind_speed_m_s')):
            raise AssertionError('Negative irradiance or wind after perturbation')
        if not np.isfinite(obs).all():raise AssertionError('Nonfinite state')
        power.append(P);gross.append(E);cost.append(C);clock.append(actual);albedo.append(info['surface_albedo'])
    if len(clock)!=195 or str(clock[-1].date())==str(clock[0].date()):
        raise AssertionError('195-step midnight rollover was not observed')
    if not baseline.equals(u.weather_profile):
        raise AssertionError('Weather changed after reset during episode; expected frozen exogenous trajectory')
    return {
        'scenario_id':sid,'realization_id':rid,
        'weather_profile_sha256':signature,
        'weather_realization_seed':int(u._weather_realization_seed),
        'first_post_action_utc':clock[0].isoformat(),
        'last_post_action_utc':clock[-1].isoformat(),
        'total_gross_kwh':float(sum(gross)),
        'total_move_kwh':float(sum(cost)),
        'total_net_kwh':float(sum(gross)-sum(cost)),
        'power_sha256':hashlib.sha256(np.asarray(power,dtype='<f8').tobytes()).hexdigest(),
        'sample_count':len(power),
        'max_closure_residual_wm2':float(u._weather_uncertainty_diagnostics.get('max_post_closure_residual_wm2',0)),
        'projection_count':int(u._weather_uncertainty_diagnostics.get('clip_count',0)),
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uncertainty-spec',required=True)
    parser.add_argument('--scenario-id',default=None,
                        help='Optional TRAIN-only historical day; never inspect test days during tuning')
    parser.add_argument('--out',default='outputs/monte_carlo_environment_audit.json')
    args=parser.parse_args(argv)
    # Direct-file execution does not automatically add the repo root to
    # sys.path. Validate the expected checkout layout before importing the
    # project. `python -m PostProcessing_MC_patch.tools.verify_mc_environment`
    # also works and does not need this adjustment.
    import sys
    repo_root=Path(__file__).resolve().parents[2]
    if not (repo_root/'mbpo'/'env').is_dir() or not (repo_root/'SAC_workspace'/'src').is_dir():
        raise RuntimeError(f'This verifier must be placed inside a complete PostProcessing checkout: {repo_root}')
    if str(repo_root) not in sys.path:
        sys.path.insert(0,str(repo_root))
    from mbpo.env.trajectory_uncertainty import normalized_spec
    from SAC_workspace.src.config.config import load_config
    from SAC_workspace.src.environment.environment import SACEnvironment
    from SAC_workspace.src.training.experiment_protocol import build_scenario_protocol
    spec=normalized_spec(json.loads(Path(args.uncertainty_spec).read_text(encoding='utf-8')))
    if not spec['enabled']:parser.error('The sensitivity assumption must enable weather uncertainty')
    cfg=load_config(weather_uncertainty=spec,require_data=True)
    split=build_scenario_protocol(cfg.manifest,cfg.protocol)
    sid=args.scenario_id or split['train_ids'][len(split['train_ids'])//2]
    if sid not in split['train_ids']:
        parser.error('Use a training-only historical day for the preliminary simulator audit')
    kwargs=dict(cfg.train_env)
    kwargs.update(randomize_initial_orientation=False,randomize_scenario=False)
    env=SACEnvironment(kwargs)
    try:
        first=one_episode(env,sid,0)
        second=one_episode(env,sid,0)
        other=one_episode(env,sid,1)
    finally:env.close()
    for field in ('weather_profile_sha256','power_sha256','total_gross_kwh','total_net_kwh'):
        if first[field]!=second[field]:
            raise AssertionError('Fixed-day fixed-draw Monte Carlo replay is not deterministic: '+field)
    if first['weather_profile_sha256']==other['weather_profile_sha256']:
        raise AssertionError('Distinct Monte Carlo draw IDs generated an identical weather profile')
    # The unperturbed NSRDB profile is obtained independently from an env
    # with weather uncertainty disabled. Avoid changing or overwriting source
    # weather data; this is a diagnostic, not a fitted calibration check.
    nominal_kwargs=dict(cfg.train_env)
    nominal_kwargs.pop('weather_uncertainty',None)
    nominal_kwargs.update(randomize_initial_orientation=False,randomize_scenario=False)
    baseline=SACEnvironment(nominal_kwargs)
    try:
        baseline.seed(819)
        baseline.reset(scenario_id=sid)
        raw=baseline.unwrapped.weather_profile.copy(deep=True)
        physical=baseline.unwrapped
        nominal_energy=0.
        for t in range(baseline.horizon):
            _obs,_rew,done,info=baseline.step(np.zeros(2,dtype=np.float32))
            nominal_energy+=float(info['energy_kwh'])
        if not raw.equals(physical.weather_profile):
            raise AssertionError('Unperturbed NSRDB weather changed during nominal episode')
        baseline_profile={'nominal_gross_kwh':float(nominal_energy),
                          'nominal_input_count':len(raw),
                          'nominal_input_columns':list(raw.columns)}
    finally:baseline.close()
    report={'status':'passed','source':'real historical TRAIN day plus hypothetical uncertainty realizations',
        'unperturbed_baseline':baseline_profile,
        'scenario_id':sid,'fixed_draw_replay':True,'distinct_uncertainty_draws':True,
        'draw_zero':first,'draw_one':other,
        'limitations':'This checks the physical simulator and observed trajectory wiring; not SAC/MBPO learning or field accuracy'}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('PASS: 195 post-action 5-minute UTC transitions; replay, physics and net reward validated')
    print('Audit:',out.resolve())

if __name__=='__main__':main()
