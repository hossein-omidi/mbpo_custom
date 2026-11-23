"""Nested Monte Carlo test of *trained*, deterministic SAC and MBPO policies.

For every held-out historical day e and uncertainty realization m, identical
exogenous weather is supplied to both policies and to fixed / astronomical
baselines. The controller chooses actions sequentially within each complete
195-step NSRDB episode; policy state and power are NOT independently averaged
at each time step. The original PVLib physical backend computes every reward.

Inputs must be fresh checkpoints trained with the same weather uncertainty
assumptions and branch source code. Model-generated MBPO experience is never
used in this evaluation; it is always the original PV physical simulator.
"""
from __future__ import annotations
import argparse,csv,json,math,hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
from mbpo.env.trajectory_uncertainty import normalized_spec,spec_fingerprint

METHODS=('fixed_no_motion','sun_slew_limited','sac','mbpo')
PAIR_NAMES=(('mbpo','sac'),('mbpo','fixed_no_motion'),('sac','fixed_no_motion'),
            ('mbpo','sun_slew_limited'),('sac','sun_slew_limited'))


def action_for(method,env,obs,agent):
    if method=='fixed_no_motion':return np.zeros(2,dtype=np.float32)
    if method in ('mbpo','sac'):return agent.act(obs,deterministic=True)
    if method!='sun_slew_limited':raise ValueError(method)
    from mbpo.env.pvlib_physics import dual_axis_tracking_targets
    u=env.unwrapped
    # Physical energy is computed at post-action t+1, so target that time.
    nxt=u.times[min(u.step_index+1,len(u.times)-1)]
    sp=u._solar_position(nxt)
    if float(sp.zenith)>=90.:return np.zeros(2,dtype=np.float32)
    tilt,az=dual_axis_tracking_targets(float(sp.zenith),float(sp.azimuth))
    dt=np.clip((tilt-u.tilt)/u.max_delta_tilt,-1,1)
    da=((az-u.azimuth+180)%360-180)/u.max_delta_azimuth
    return np.array([dt,np.clip(da,-1,1)],dtype=np.float32)


def run_one(env,method,sid,draw,policy=None):
    obs=env.reset(scenario_id=sid,realization_id=int(draw))
    u=env.unwrapped
    first_pose=(float(u.tilt),float(u.azimuth))
    profile_sha=u._weather_profile_sha256
    if not profile_sha:raise RuntimeError('Missing controlled weather realization fingerprint')
    totals={'gross_kwh':0.,'movement_kwh':0.,'net_kwh':0.}
    times=[];power=[];weather=[];rewards=[]
    for t in range(env.horizon):
        a=action_for(method,env,obs,policy)
        obs,rew,done,info=env.step(a)
        if info.get('scenario_id')!=sid or info.get('weather_profile_sha256')!=profile_sha:
            raise RuntimeError('Historical day or weather realization changed mid-episode')
        if int(info.get('weather_realization_id'))!=int(draw):
            raise RuntimeError('Paired Monte Carlo realization mismatch')
        utc=datetime.fromisoformat(info['timestamp_utc_iso'].replace('Z','+00:00'))
        if utc.utcoffset().total_seconds()!=0:
            raise RuntimeError('Expected zero UTC offset')
        if times and (utc-times[-1]).total_seconds()!=300:
            raise RuntimeError('Non-chronological 5-minute power trajectory')
        times.append(utc)
        power.append(float(info['power_dc_w']))
        weather.append((float(info['dni_wm2']),float(info['dhi_wm2']),
                        float(info['ghi_wm2']),float(info['ambient_temperature_c']),
                        float(info['wind_speed_m_s']),float(info['surface_albedo'])))
        totals['gross_kwh']+=float(info['energy_kwh'])
        totals['movement_kwh']+=float(info['movement_cost'])
        totals['net_kwh']+=float(rew)
        rewards.append(float(rew))
        if bool(done)!=(t==env.horizon-1):raise RuntimeError('Unexpected episode termination')
    if not np.isclose(totals['gross_kwh']-totals['movement_kwh'],totals['net_kwh'],atol=1e-9):
        raise RuntimeError('Physical energy and reward accounting disagree')
    return {**totals,'pose':first_pose,'profile_sha256':profile_sha,
            'timestamps_utc':[s.isoformat() for s in times],
            'power_w':np.asarray(power),'weather':np.asarray(weather),
            'step_reward_kwh':np.asarray(rewards)}


def descriptive(values):
    x=np.asarray(values,dtype=float)
    if x.size==0 or not np.isfinite(x).all():raise ValueError('Missing/nonfinite numerical results')
    return {'n':int(len(x)),'mean':float(x.mean()),
            'sd_descriptive':float(x.std(ddof=1)) if len(x)>1 else 0.,
            'q05':float(np.quantile(x,.05)),'median':float(np.median(x)),
            'q95':float(np.quantile(x,.95)),
            'mc_se_mean':float(x.std(ddof=1)/math.sqrt(len(x))) if len(x)>1 else None}


def nested_summary(records,field,paired=None):
    """Equal weight to each historical day, then equal weight to each draw."""
    grouped=defaultdict(list)
    for row in records:
        if paired is None:
            if row['method']!=field:continue
            x=float(row['net_kwh'])
        else:
            if row['method']!=paired[0]:continue
            ref=paired[1]
            key=(row['scenario_id'],row['draw'],ref)
            if key not in field:raise ValueError('Missing common-realization paired reference')
            x=float(row['net_kwh'])-float(field[key])
        grouped[row['scenario_id']].append(x)
    if not grouped:raise ValueError('No nested Monte Carlo records')
    sizes={len(x) for x in grouped.values()}
    if len(sizes)!=1:raise ValueError('Draw counts differ across historical scenarios')
    daymeans=np.asarray([np.mean(v) for v in grouped.values()])
    pooled=np.concatenate([np.asarray(v) for v in grouped.values()])
    overall=float(daymeans.mean())
    vb=float(np.mean((daymeans-overall)**2))
    vw=float(np.mean([np.mean((np.asarray(v)-np.mean(v))**2) for v in grouped.values()]))
    res={'n_historical_days':len(grouped),'draws_per_day':next(iter(sizes)),
         'mean_net_kwh':overall,'between_day_variance_kwh2':vb,
         'within_day_uncertainty_variance_kwh2':vw,
         'total_descriptive_sd_kwh':math.sqrt(vb+vw),
         'pooled_quantiles_kwh':{'q05':float(np.quantile(pooled,.05)),
              'median':float(np.median(pooled)),'q95':float(np.quantile(pooled,.95))},
         'each_day':{sid:descriptive(x) for sid,x in grouped.items()}}
    res['note']='Within/between components are descriptive for assumed uncertainty, not a calibrated future-weather CI.'
    return res


def write_csv(path,rows):
    if not rows:raise ValueError('No output rows')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def load_policy(cfg,which,checkpoint,device):
    import torch
    from SAC_workspace.src.agents.sac_agent import SACAgent
    from SAC_workspace.src.training.trainer import config_hash
    from MBPO_workspace.src.config import config_digest
    from SAC_workspace.src.environment.environment import SACEnvironment
    data=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if data.get('schema')!=2:
        raise ValueError('Unrecognized checkpoint schema; train new policies with updated code')
    if which=='sac':
        if data.get('hash')!=config_hash(cfg.sac):
            raise ValueError('SAC checkpoint was not trained with this uncertainty configuration')
    elif data.get('config_hash')!=config_digest(cfg):
        raise ValueError('MBPO checkpoint was not trained with this uncertainty configuration')
    check_env=SACEnvironment(cfg.sac.eval_env)
    try:
        if data.get('resolved_mdp')!=check_env.resolved_mdp:
            raise ValueError('Checkpoint and current physical data/model fingerprint differ')
    finally:check_env.close()
    a=cfg.sac.algo
    agent=SACAgent(hidden_sizes=a['hidden_sizes'],lr=a['lr'],gamma=a['gamma'],
          tau=a['tau'],reward_scale=a['reward_scale'],alpha_init=a['alpha_init'],
          target_entropy=a['target_entropy'],min_alpha=a['min_alpha'],device=device)
    agent.load_state_dict(data['agent']);agent.actor.eval()
    return agent


def evaluate_paths(cfg,agents,scenario_ids,draws,selected_time_scenario):
    from SAC_workspace.src.environment.environment import SACEnvironment
    env=SACEnvironment(cfg.sac.eval_env)
    rows=[];profiles={};time_output=[]
    try:
        if not env.unwrapped.weather_uncertainty['enabled']:
            raise ValueError('Monte Carlo evaluator requires enabled uncertainty')
        for day_ix,sid in enumerate(scenario_ids):
            for rid in range(draws):
                paired_ref=None
                for method in METHODS:
                    result=run_one(env,method,sid,rid,agents.get(method))
                    signature=(result['profile_sha256'],tuple(result['timestamps_utc']),result['pose'])
                    if paired_ref is None:paired_ref=signature
                    if paired_ref!=signature:
                        raise RuntimeError(f'Controller weather/clock/pose mismatch on {sid} draw {rid}')
                    row={'scenario_id':sid,'draw':rid,'method':method,
                         'net_kwh':result['net_kwh'],'gross_kwh':result['gross_kwh'],
                         'movement_kwh':result['movement_kwh'],
                         'profile_sha256':result['profile_sha256']}
                    rows.append(row)
                    if sid==selected_time_scenario:
                        time_output.append((method,rid,result['timestamps_utc'],result['power_w']))
            print(f'Completed MC historical day {day_ix+1}/{len(scenario_ids)}: {sid}',flush=True)
    finally:env.close()
    return rows,time_output



def plot_diagnostics(out, rows, clock_rows, paired):
    """Publication-oriented descriptive figures; not inferential confidence bands."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import dates as mdates
    fig,ax=plt.subplots(figsize=(11,4.8))
    for method in METHODS:
        subset=[r for r in clock_rows if r['method']==method]
        times=[datetime.fromisoformat(r['time_utc']) for r in subset]
        mean=np.asarray([r['power_mean_w'] for r in subset]);sd=np.asarray([r['power_sd_w'] for r in subset])
        ax.plot(times,mean,label=method,linewidth=1.7)
        ax.fill_between(times,mean-sd,mean+sd,alpha=.12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M UTC'))
    ax.set_title('One held-out day: mean ± descriptive SD across weather draws')
    ax.set_ylabel('PV DC power (W)');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'mc_first_day_power_mean_sd.png',dpi=190);plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5))
    data=[[r['net_kwh'] for r in rows if r['method']==method] for method in METHODS]
    ax.boxplot(data,labels=METHODS,showfliers=False)
    ax.set_ylabel('Daily net DC energy (kWh)')
    ax.set_title('Assumed weather-error distribution over held-out historical days')
    ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(out/'mc_net_energy_boxplots.png',dpi=190);plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5))
    comparison=paired['mbpo_minus_sac']
    pairs=comparison['each_day']
    labels=list(pairs);values=[pairs[k]['mean'] for k in labels]
    ax.bar(range(len(labels)),values)
    if len(labels)<35:
        ax.set_xticks(range(len(labels)),labels,rotation=80,fontsize=7)
    else:
        ax.set_xticks([])
    ax.axhline(0,linewidth=.8)
    ax.set_ylabel('Paired net energy difference MBPO − SAC (kWh)')
    ax.set_title('Historical-day paired mean differences under common weather draws')
    fig.tight_layout();fig.savefig(out/'mc_paired_day_differences.png',dpi=190);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sac-checkpoint',required=True)
    p.add_argument('--mbpo-checkpoint',required=True)
    p.add_argument('--scenario-split-json',required=True)
    p.add_argument('--split-role',choices=('test','validation'),default='test')
    p.add_argument('--uncertainty-spec',required=True)
    p.add_argument('--draws',type=int,default=100)
    p.add_argument('--max-days',type=int,default=None,help='DEBUG subset only; never label as a full test')
    p.add_argument('--outdir',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--skip-figures',action='store_true')
    p.add_argument('--model-epochs',type=int,default=None,help='Match any training-time MBPO override')
    p.add_argument('--model-max-train-samples',type=int,default=None,help='Match any training-time MBPO override')
    args=p.parse_args(argv)
    if args.draws<2:p.error('At least two Monte Carlo draws are required for SD')
    if args.max_days is not None and args.max_days<1:p.error('max-days must be positive')
    spec=normalized_spec(json.loads(Path(args.uncertainty_spec).read_text(encoding='utf-8')))
    if not spec['enabled']:p.error('uncertainty-spec must have enabled=true')
    from MBPO_workspace.src.config import load_config
    from SAC_workspace.src.training.experiment_protocol import build_scenario_protocol
    overrides={}
    if args.model_epochs is not None:overrides['model_epochs']=args.model_epochs
    if args.model_max_train_samples is not None:overrides['model_max_train_samples']=args.model_max_train_samples
    cfg=load_config(weather_uncertainty=spec,model_overrides=overrides)
    split=json.loads(Path(args.scenario_split_json).read_text(encoding='utf-8'))
    expected=build_scenario_protocol(cfg.sac.manifest,cfg.sac.protocol)
    if split!=expected:raise ValueError('Frozen split differs from current manifest and training protocol')
    if set(split['train_ids'])&set(split['test_ids']):raise ValueError('Train/test leakage')
    days=list(split[f'{args.split_role}_ids'])
    population_size=len(days)
    if args.max_days is not None:days=days[:args.max_days]
    if not days:raise ValueError('Empty selected historical partition')
    agents={
        'sac':load_policy(cfg,'sac',args.sac_checkpoint,args.device),
        'mbpo':load_policy(cfg,'mbpo',args.mbpo_checkpoint,args.device),
    }
    rows,times=evaluate_paths(cfg,agents,days,args.draws,days[0])
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'all_day_draw_method.csv',rows)
    refs={(r['scenario_id'],r['draw'],r['method']):r['net_kwh'] for r in rows}
    methods={name:nested_summary(rows,name) for name in METHODS}
    paired={}
    for a,b in PAIR_NAMES:
        paired[f'{a}_minus_{b}']=nested_summary(rows,refs,(a,b))
    clock_rows=[]
    for method in METHODS:
        selected=sorted((r for r in times if r[0]==method),key=lambda v:v[1])
        if len(selected)!=args.draws:raise RuntimeError('Incomplete selected-day power profiles')
        grid=selected[0][2]
        if any(r[2]!=grid for r in selected):raise ValueError('UTC grid changes across uncertainty realizations')
        mat=np.stack([r[3] for r in selected],axis=0)
        for t,utc in enumerate(grid):
            clock_rows.append({'scenario_id':days[0],'method':method,'time_utc':utc,'step':t+1,
                'power_mean_w':float(mat[:,t].mean()),
                'power_sd_w':float(mat[:,t].std(ddof=1)),
                'power_q05_w':float(np.quantile(mat[:,t],.05)),
                'power_q95_w':float(np.quantile(mat[:,t],.95))})
    write_csv(out/'first_day_power_mean_sd.csv',clock_rows)
    report={'uncertainty_spec':spec,'uncertainty_spec_sha256':spec_fingerprint(spec),
        'scenario_protocol_sha256':split['protocol_sha256'],
        'split_role':args.split_role,'n_selected_days':len(days),'n_available_days':population_size,
        'partial_partition_evaluation':len(days)!=population_size,
        'draws_per_day':args.draws,'methods':methods,'paired_net_energy_differences':paired,
        'note':'Descriptive uncertainty under assumed weather-error distribution; no field-data calibration claimed.',
        'evaluation_model':'Original pvlib Stage A physical environment; no MBPO fake rollouts',
        'common_random_numbers':'Same uncertainty realization_id and weather profile SHA across paired controllers',
    }
    (out/'mc_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    if not args.skip_figures:
        plot_diagnostics(out,rows,clock_rows,paired)
    print('Monte Carlo summary:',(out/'mc_summary.json').resolve())

if __name__=='__main__':main()
