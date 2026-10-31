#!/usr/bin/env python3
"""Read-only SAC-versus-fixed forensic analysis for the Stage-2/3 PVWatts task.

Uses *recorded post-step* weather for an immediate-action counterfactual.
That counterfactual is not an implementable policy or a full-horizon oracle.
Optionally reconstructs the *previous pre-action* 6D observation and verifies
that the requested SAC checkpoint generates the exported deterministic actions.
No checkpoints, models, configurations, reports or training data are modified.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

# Direct invocation (python scripts/<name>.py) makes Python search scripts/
# rather than the repository root. Resolve workspace imports using this script's
# installed location; do not depend on PYTHONPATH, cwd, or shell configuration.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import numpy as np

HORIZON=117
METHODS={'sac','mbpo','fixed_no_motion','poa_instant_ideal'}
TEMP_PRESETS={
    'open_rack_glass_polymer':(-3.56,-0.075,3.0),
    'open_rack_glass_glass':(-3.47,-0.0594,3.0),
    'close_mount_glass_glass':(-2.98,-0.0471,1.0),
    'insulated_back_glass_polymer':(-2.81,-0.0455,0.0),
}

def require(ok, why):
    if not ok: raise AssertionError(why)


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def f(row,key):
    value=row.get(key)
    require(value not in ('',None),f'Missing {key}: {row.get("scenario_id")}, step {row.get("step")}')
    val=float(value)
    require(math.isfinite(val),f'Non-finite {key}')
    return val


def isotropic_poa(solar_zenith,solar_azimuth,dni,dhi,ghi,tilt,azimuth):
    """pvlib default isotropic decomposition, albedo=.25; vectorized."""
    zen=np.deg2rad(solar_zenith); sa=np.deg2rad(solar_azimuth)
    beta=np.deg2rad(tilt); az=np.deg2rad(azimuth)
    cos_aoi=np.cos(zen)*np.cos(beta)+np.sin(zen)*np.sin(beta)*np.cos(sa-az)
    return np.maximum(0,dni*np.maximum(cos_aoi,0)
       +dhi*(1+np.cos(beta))/2 +ghi*.25*(1-np.cos(beta))/2)


def dc_from_poa(poa,ambient,wind10,model,eta,gamma,mount):
    if model=='legacy':return poa*eta, np.full(len(poa),np.nan)
    require(model=='pvwatts_sapm',f'Unsupported model {model}')
    require(mount in TEMP_PRESETS, f'Unsupported SAPM preset {mount}')
    a,b,dt=TEMP_PRESETS[mount]
    tcell=ambient+poa*(np.exp(a+b*wind10)+dt/1000.)
    return np.maximum(0.,poa*eta*(1.+gamma*(tcell-25))),tcell


def summarize_day(rows, models, args):
    # rows contain four methods from exactly the same scenario and evaluation seed
    sid=rows['sac'][0]['scenario_id']; sac=rows['sac'];fixed=rows['fixed_no_motion'];mbpo=rows.get('mbpo')
    def energy(method):return sum(f(row,'energy_kwh') for row in rows[method])
    vals={m:energy(m) for m in rows}
    require(all(len(v)==HORIZON for v in rows.values()),f'Incomplete day {sid}')
    weather_fields=('solar_zenith_deg','solar_azimuth_deg','dni_wm2',
                    'dhi_wm2','ghi_wm2','ambient_temperature_c','wind_speed_m_s')
    reference_weather={k:np.array([f(z,k) for z in fixed]) for k in weather_fields}
    for method,rr in rows.items():
        require([int(z['step']) for z in rr]==list(range(1,HORIZON+1)), f'Bad step sequence for {sid}/{method}')
        require([z['utc_iso'] for z in rr]==[z['utc_iso'] for z in fixed],f'Mismatched clock for {sid}/{method}')
        for k in weather_fields:
            measured=np.array([f(z,k) for z in rr])
            require(np.max(np.abs(measured-reference_weather[k]))<=1e-7,
                    f'Mismatched paired weather {sid}/{method}/{k}')
        for z in rr:
            e=f(z,'power_w')/12000.
            require(abs(e-f(z,'energy_kwh'))<1e-7,f'Energy mismatch {sid}/{method}')
            require(z['power_model']==models['model'],f'Physics model mismatch for {sid}/{method}')
            if method!='poa_instant_ideal':
                require(abs(f(z,'net_step_kwh')-(f(z,'energy_kwh')-f(z,'movement_cost_kwh')))<1e-7,
                        f'Reward mismatch {sid}/{method}')
        # Independent physical cross-check for EACH exported method, not just SAC.
        if method in ('sac','mbpo','fixed_no_motion','poa_instant_ideal'):
            azimuth=np.array([f(z,'azimuth_deg') for z in rr])
            tilt=np.array([f(z,'tilt_deg') for z in rr])
            poa=isotropic_poa(reference_weather['solar_zenith_deg'],reference_weather['solar_azimuth_deg'],
                reference_weather['dni_wm2'],reference_weather['dhi_wm2'],reference_weather['ghi_wm2'],tilt,azimuth)
            ambient=reference_weather['ambient_temperature_c']
            wind10=np.array([f(z,'wind_speed_10m_ms') for z in rr])
            predicted,cell=dc_from_poa(poa,ambient,wind10,models['model'],models['eta'],models['gamma'],models['mount'])
            require(np.max(np.abs(poa-np.array([f(z,'poa_global') for z in rr])))<1e-5,
                    f'POA reconstruction mismatch {sid}/{method}')
            require(np.max(np.abs(predicted-np.array([f(z,'power_w') for z in rr])))<1e-5,
                    f'PVWatts reconstruction mismatch {sid}/{method}')
            if models['model']=='pvwatts_sapm':
                require(np.max(np.abs(cell-np.array([f(z,'cell_temperature_c') for z in rr])))<1e-5,
                        f'SAPM temperature mismatch {sid}/{method}')
    for z in fixed:
        require(abs(f(z,'tilt_deg')-30.)<1e-7 and abs(f(z,'azimuth_deg')-180.)<1e-7,f'Fixed moved on {sid}')
        require(abs(f(z,'action_tilt'))<1e-8 and abs(f(z,'action_azimuth'))<1e-8,f'Fixed action on {sid}')
    def values(key):return np.array([f(z,key) for z in sac],float)
    solar_zenith=values('solar_zenith_deg');solar_az=values('solar_azimuth_deg')
    dni=values('dni_wm2');dhi=values('dhi_wm2');ghi=values('ghi_wm2')
    ambient=values('ambient_temperature_c');wind10=values('wind_speed_10m_ms')
    post_tilt=values('tilt_deg');post_az=values('azimuth_deg');pre_tilt=values('pre_tilt_deg')
    pa=isotropic_poa(solar_zenith,solar_az,dni,dhi,ghi,post_tilt,post_az)
    pred,tc=dc_from_poa(pa,ambient,wind10,models['model'],models['eta'],models['gamma'],models['mount'])
    require(np.max(np.abs(pa-values('poa_global'))) < 1e-5,f'Physical POA regression on {sid}')
    require(np.max(np.abs(pred-values('power_w'))) < 1e-5,f'Physical PVWatts power regression on {sid}')
    if models['model']=='pvwatts_sapm':
        require(np.max(np.abs(tc-values('cell_temperature_c')))<1e-5,f'Temperature mismatch on {sid}')
    # Single-step counterfactual: same PRE-action panel pose, same azimuth command,
    # replace tilt command with -1 (decrease by max 5 degrees). SAME next weather.
    cf_tilt=np.clip(pre_tilt-5.,0.,90.)
    cf_poa=isotropic_poa(solar_zenith,solar_az,dni,dhi,ghi,cf_tilt,post_az)
    cf_power,_=dc_from_poa(cf_poa,ambient,wind10,models['model'],models['eta'],models['gamma'],models['mount'])
    improvement=cf_power-values('power_w')
    day=ghi>=50.
    pos=(improvement > 1e-4) & day
    return {
        'scenario_id':sid,'month':int(sid[5:7]),'n_daylight_samples':int(day.sum()),
        'sac_kwh':vals['sac'],'fixed_kwh':vals['fixed_no_motion'],
        'mbpo_kwh':vals.get('mbpo'),
        'sac_minus_fixed_kwh':vals['sac']-vals['fixed_no_motion'],
        'sac_tilt_mean_daylight_deg':float(np.mean(post_tilt[day])) if day.any() else None,
        'sac_positive_tilt_action_daylight_fraction':float(np.mean((values('action_tilt')>0)[day])) if day.any() else None,
        'local_downward_tilt_better_count':int(pos.sum()),
        'local_downward_tilt_improvement_wh':float(np.sum(np.maximum(0,improvement[day]))/12.),
        'local_downward_tilt_net_delta_wh':float(np.sum(improvement[day])/12.),
    }


def check_actor_replay(eval_dir,checkpoint,models,all_paths,batch_size=8192):
    """Rebuild the exact previous 6D observations from step t-1 (omit step 1).

    This verifies that the current trusted checkpoint produces the same deterministic
    actions shown in the exported CSV. A mismatch indicates provenance/code drift.
    """
    from SAC_workspace.src.config.config import load_config
    from SAC_workspace.src.evaluation.evaluation import load_agent
    from mbpo.env.pv_tracking import POWER_NORM, IRRADIANCE_NORM
    cfg=load_config(require_data=False,power_model=models['model'],
        wind_reference_height_m=models['wind_reference_height_m'],
        gamma_pdc=models['gamma'],sapm_mount=models['mount'],
        wind_powerlaw_exponent=models['wind_powerlaw_exponent'])
    agent=load_agent(cfg,checkpoint,device='cpu')  # trusted local checkpoints only
    import torch
    states=[];targets=[]
    for key,methods in sorted(all_paths.items()):
        rr=methods['sac']
        for prev,now in zip(rr[:-1],rr[1:]):
            # Action now uses observation AFTER previous step, with tau=(117 - prev_step)/117.
            step=int(prev['step'])
            tilt=f(prev,'tilt_deg');az=f(prev,'azimuth_deg')
            x=[np.clip(f(prev,'dni_wm2')/IRRADIANCE_NORM,0,1),
               np.clip(f(prev,'dhi_wm2')/IRRADIANCE_NORM,0,1),
               np.clip(f(prev,'power_w')/POWER_NORM,0,1),
               tilt/90.,(az%360.)/180.-1.,(HORIZON-step)/HORIZON]
            states.append(x);targets.append([f(now,'action_tilt'),f(now,'action_azimuth')])
    st=np.asarray(states,dtype=np.float32); actions=np.asarray(targets,dtype=np.float32)
    preds=[];critic_gaps=[];stdtilt=[];stdaz=[];actor_at_tilt_bound=[]
    with torch.no_grad():
        for start in range(0,len(st),batch_size):
            ss=torch.as_tensor(st[start:start+batch_size],dtype=torch.float32)
            chosen,_=agent.actor(ss,deterministic=True,with_logprob=False)
            # Keep the same azimuth action and replace only tilt with the
            # maximum physically permitted downward movement (normalized -1).
            alternative=chosen.clone();alternative[:,0]=-1.
            q1,q2=agent.critic(ss,chosen)
            p1,p2=agent.critic(ss,alternative)
            critic_gaps.append((torch.minimum(q1,q2)-torch.minimum(p1,p2)).view(-1).cpu().numpy())
            mean,raw_log_std=agent.actor.net(ss).chunk(2,dim=-1)
            sigma=raw_log_std.clamp(-20.,2.).exp()
            stdtilt.append(sigma[:,0].cpu().numpy())
            stdaz.append(sigma[:,1].cpu().numpy())
            actor_at_tilt_bound.append((chosen[:,0]>0.98).cpu().numpy())
            preds.append(chosen.cpu().numpy())
    predicted=np.concatenate(preds)
    deviation=np.max(np.abs(predicted-actions),axis=1)
    qgap=np.concatenate(critic_gaps)
    replay_ok=bool(np.max(deviation)<=1e-4)
    return {'checked_actions':int(len(actions)),
            'max_abs_action_difference':float(deviation.max()),
            'p99_abs_action_difference':float(np.quantile(deviation,.99)),
            'fraction_abs_difference_gt_1e-4':float(np.mean(deviation>1e-4)),
            'passes_exact_action_replay':replay_ok,
            'critic_diagnostic':{
                'interpretable_for_saved_evaluation':replay_ok,
                'mean_Q_chosen_minus_max_downward':float(np.mean(qgap)),
                'fraction_Q_prefers_chosen':float(np.mean(qgap>=0)),
                'fraction_Q_prefers_downward':float(np.mean(qgap<0)),
                'p10_Q_gap':float(np.quantile(qgap,.1)),
                'p90_Q_gap':float(np.quantile(qgap,.9)),
                'mean_policy_pre_tanh_std_tilt':float(np.mean(np.concatenate(stdtilt))),
                'mean_policy_pre_tanh_std_azimuth':float(np.mean(np.concatenate(stdaz))),
                'fraction_deterministic_tilt_action_above_0_98':float(np.mean(np.concatenate(actor_at_tilt_bound))),
                'limitations':'Q is a learned, reward-scaled long-horizon estimate; Q preference is NOT proof that the alternative action is optimal.'},
            'note':'If action replay fails, first check checkpoint provenance; do not interpret Q diagnostics as belonging to this evaluation.'}


def train_data(path):
    if not path:return None
    rows=read_csv(path)
    require(len(rows)>0,f'Empty training history: {path}')
    last=rows[-1]
    return {'episodes':len(rows),'last_epoch':int(last['epoch']),
            'real_steps':int(last['env_steps']),
            'last_reported_return_kwh':float(last['return_kwh']),
            'last_alpha':float(last['alpha']),
            'last_model_replay_size':int(last['model_replay_size']) if 'model_replay_size' in last else None,
            'accepted_synthetic_total':sum(int(float(r['synthetic_generated'])) for r in rows) if 'synthetic_generated' in last else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir',required=True)
    p.add_argument('--sac-checkpoint',default=None,help='Trusted local SAC checkpoint used in this evaluation')
    p.add_argument('--sac-training-csv',default=None)
    p.add_argument('--mbpo-training-csv',default=None)
    p.add_argument('--outdir',default=None)
    args=p.parse_args(); root=Path(args.eval_dir)
    paths=read_csv(root/'trajectories_detailed.csv')
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    physics=summary['power_physics']; model=physics.get('model',summary.get('power_model'))
    models={'model':model,'gamma':float(physics.get('gamma_pdc',-.004)),
            'mount':physics.get('sapm_mount','open_rack_glass_polymer'),'eta':.18,
            'wind_reference_height_m':float(physics.get('wind_reference_height_m',2)),
            'wind_powerlaw_exponent':float(physics.get('wind_powerlaw_exponent',.14))}
    require(model=='pvwatts_sapm', 'This audit targets PVWatts–SAPM evaluation')
    all_paths=defaultdict(lambda:defaultdict(list))
    for row in paths: all_paths[(row['scenario_id'],int(row['seed']))][row['method']].append(row)
    require(len(all_paths)>0,'No episodes')
    report=[]
    for key,methods in sorted(all_paths.items()):
        require(METHODS<=set(methods),f'Missing method on {key}')
        for m in methods: methods[m].sort(key=lambda r:int(r['step']))
        report.append(summarize_day(methods,models,args))
    dls=sum(r['n_daylight_samples'] for r in report)
    totals={
      'evaluated_dates':len(report),'daylight_samples':dls,
      'fixed_beats_sac_days':sum(r['sac_minus_fixed_kwh']<0 for r in report),
      'sac_beats_fixed_days':sum(r['sac_minus_fixed_kwh']>0 for r in report),
      'locally_better_downward_tilt_daylight_fraction':sum(r['local_downward_tilt_better_count'] for r in report)/max(dls,1),
      'sac_mean_daylight_tilt_deg':float(np.average([r['sac_tilt_mean_daylight_deg'] for r in report if r['n_daylight_samples']],weights=[r['n_daylight_samples'] for r in report if r['n_daylight_samples']])),
      'sac_mean_positive_tilt_action_daylight_fraction':sum(r['sac_positive_tilt_action_daylight_fraction']*r['n_daylight_samples'] for r in report if r['n_daylight_samples'])/max(dls,1),
      'local_tilt_check':'One-step hindsight comparison, NOT a realizable or finite-horizon policy',
    }
    by_month={}
    for m in range(1,13):
        rows=[r for r in report if r['month']==m]
        if not rows:continue
        n=sum(r['n_daylight_samples'] for r in rows)
        by_month[str(m)]={'days':len(rows),'sac_minus_fixed_mean_kwh':float(np.mean([r['sac_minus_fixed_kwh'] for r in rows])),
                         'better_downward_tilt_fraction':sum(r['local_downward_tilt_better_count'] for r in rows)/n if n else None}
    result={'source':str(root.resolve()),'physics':models,'totals':totals,'monthly':by_month,
            'sac_training_history':train_data(args.sac_training_csv),
            'mbpo_training_history':train_data(args.mbpo_training_csv),
            'checkpoint_action_replay':check_actor_replay(root,args.sac_checkpoint,models,all_paths) if args.sac_checkpoint else 'NOT VERIFIED: no SAC checkpoint supplied'}
    out=Path(args.outdir or root/'sac_forensic');out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True),encoding='utf-8')
    with (out/'daily_diagnostics.csv').open('w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(report[0]));writer.writeheader();writer.writerows(report)
    print(json.dumps(result,indent=2,sort_keys=True))
    print('Saved:',out.resolve())
    if args.sac_checkpoint and not result['checkpoint_action_replay']['passes_exact_action_replay']:
        raise SystemExit('FAIL: recorded SAC actions do not match the given checkpoint')

if __name__=='__main__':main()
