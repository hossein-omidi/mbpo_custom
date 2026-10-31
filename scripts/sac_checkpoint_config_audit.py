#!/usr/bin/env python3
"""Read-only reproducibility audit for paired PVWatts SAC/MBPO evaluation.

Validates the numerical report (not just plot labels), effective physical/config
settings, training-history and checkpoint integrity, and replay reward/normalization
contracts. Does not require a checkpoint to examine an exported evaluation folder.

Only open locally generated and trusted .pt files: torch.load uses pickle.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
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
REQUIRED=('sac','mbpo','fixed_no_motion','poa_instant_ideal')
FEASIBLE=REQUIRED[:3]
PHYSICS_KEYS=('power_model','wind_reference_height_m','gamma_pdc','sapm_mount',
              'wind_powerlaw_exponent','site_latitude','site_longitude',
              'start_time','periods','freq','area','efficiency','horizon',
              'max_delta_tilt','max_delta_azimuth','movement_penalty',
              'observation_mode','power_source_sha256')


def ensure(x,msg):
    if not x:raise AssertionError(msg)


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def scalar(x,field):
    try:v=float(x)
    except (ValueError,TypeError) as exc:raise AssertionError(f'{field}: missing/non-numeric {x!r}') from exc
    ensure(math.isfinite(v),f'{field}: non-finite {v}')
    return v

def canon_hash(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()

def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def model_fields(summary):
    pp=summary.get('power_physics',{})
    name=pp.get('model',summary.get('power_model'))
    ensure(name in ('legacy','pvwatts_sapm'),'Missing/invalid physical model in summary')
    if name=='pvwatts_sapm':
        for f in ('gamma_pdc','sapm_mount','wind_powerlaw_exponent','wind_reference_height_m'):
            ensure(pp.get(f) is not None,f'Missing power_physics.{f}')
    return {'power_model':name,'wind_reference_height_m':pp.get('wind_reference_height_m'),
            'gamma_pdc':pp.get('gamma_pdc'),'sapm_mount':pp.get('sapm_mount'),
            'wind_powerlaw_exponent':pp.get('wind_powerlaw_exponent')}


def sample_weighting(episodes,plan,summary,atol=1e-7):
    """Independent recomputation from raw daily episodes and the observed stratum weights."""
    required=set(REQUIRED)
    byday=defaultdict(dict)
    seeds=set()
    for row in episodes:
        sid=str(row['scenario_id']);method=str(row['method']);seed=int(row['seed'])
        ensure(method in required, f'Unexpected method {method}'); seeds.add(seed)
        key=(sid,seed)
        ensure(method not in byday[key],f'Duplicate method {method} for {key}')
        gross=scalar(row['gross_energy_kwh'],'gross_energy_kwh')
        net=scalar(row['net_energy_kwh'],'net_energy_kwh') if method!='poa_instant_ideal' else None
        movement=scalar(row['movement_cost_kwh'],'movement_cost_kwh') if method!='poa_instant_ideal' else None
        if net is not None: ensure(abs(gross-movement-net)<1e-8,f'Reward accounting mismatch for {key}, {method}')
        ensure(gross>=-1e-9,f'Negative gross energy {key}')
        byday[key][method]={'gross':gross,'net':net,'cost':movement,
                            'pose':(scalar(row['initial_tilt_deg'],'initial_tilt_deg'),scalar(row['initial_azimuth_deg'],'initial_azimuth_deg'))}
    ensure(byday,'Empty evaluation episodes')
    ensure(len(seeds)==1,'Expected one paired evaluation seed; multiple seeds require a declared aggregation protocol')
    for key,methods in byday.items():
        ensure(set(methods)==required,f'Unpaired methods in {key}')
        poses={tuple(round(x,8) for x in r['pose']) for r in methods.values()}
        ensure(len(poses)==1,f'Unequal initial panel pose in {key}')
        ensure(abs(methods['fixed_no_motion']['pose'][0]-30)<1e-9 and abs(methods['fixed_no_motion']['pose'][1]-180)<1e-9,'Unexpected fixed-start pose')
        ensure(methods['poa_instant_ideal']['gross']+1e-7 >= max(methods[m]['gross'] for m in FEASIBLE),f'Ideal reference does not dominate feasible energy on {key}')
    days={sid for sid,seed in byday}
    ensure(len(byday)==len(days),'Multiple evaluation seeds without explicit averaging')
    if plan is None:
        weights={sid:1 for sid in days};design='unweighted'
    else:
        ensure(plan['selected_count']==len(days),'Plan selected_count mismatch')
        ensure(set(plan['selected_scenario_ids'])==days,'Plan selected scenario IDs differ from episode file')
        rows=plan['strata'];weights={str(x['selected_scenario_id']):int(x['population_days']) for x in rows}
        ensure(len(weights)==len(rows),'Duplicate stratum selected IDs')
        ensure(set(weights)==days,'Selected scenario / strata mismatch')
        ensure(all(x>0 for x in weights.values()),'Invalid stratum weights')
        ensure(sum(weights.values())==int(plan['population_count']),'Invalid stratum represented day count')
        design='calendar_stratified'
    total=sum(weights.values());vals={}
    for m in REQUIRED:
        vals[m]={}
        for metric in ('gross','net','cost'):
            if m=='poa_instant_ideal' and metric!='gross':continue
            vals[m][metric]=sum(weights[sid]*byday[(sid,next(iter(seeds)))][m][metric] for sid in days)/total
            field={'gross':'mean_gross_kwh','net':'mean_net_kwh','cost':'mean_movement_cost_kwh'}[metric]
            given=summary['methods'][m][field]
            ensure(given is not None and abs(vals[m][metric]-float(given))<atol,
                   f'Summary {m}.{field} differs from weighted episodes: calc={vals[m][metric]} saved={given}')
    pairs=[('sac','fixed_no_motion'),('mbpo','fixed_no_motion'),('mbpo','sac')]
    gains={}
    for a,b in pairs:
        diff=vals[a]['net']-vals[b]['net']; denom=vals[b]['net']
        pct=100*diff/denom if denom>1e-10 else None
        saved=summary['paired_net_kwh'][f'{a}_minus_{b}']
        ensure(abs(diff-float(saved['mean_difference_kwh']))<atol,f'Incorrect weighted paired {a}-{b} gain')
        saved_pct=saved.get('ratio_of_weighted_means_percent')
        if saved_pct is not None:
            ensure(pct is not None and abs(pct-float(saved_pct))<1e-5,f'Incorrect reported gain percentage for {a}-{b}')
        gains[f'{a}_minus_{b}']={'weighted_difference_kwh':diff,
                                'ratio_of_weighted_means_percent':pct,
                                'winning_days':sum(byday[(sid,next(iter(seeds)))][a]['net']>byday[(sid,next(iter(seeds)))][b]['net'] for sid in days),
                                'evaluated_days':len(days)}
    return {'design':design,'evaluated_days':len(days),'represented_days':total,
            'seed':next(iter(seeds)),'means':vals,'paired_gains':gains,
            'selected_ids':days}


def config_evidence(model, summary, repo=None):
    from SAC_workspace.src.config.config import load_config as sac_load
    from MBPO_workspace.src.config import load_config as mbpo_load
    kw={k:v for k,v in model.items() if k!='power_model' and v is not None}
    kw['power_model']=model['power_model']
    sac=sac_load(require_data=False,**kw)
    mbpo=mbpo_load(require_data=False,**kw)
    ensure(sac.to_dict()==mbpo.sac.to_dict(),'SAC and MBPO reference configs diverge')
    for kind, env in [('training',sac.train_env),('evaluation',sac.eval_env)]:
        ensure(env.get('observation_mode')=='compact',f'Unexpected {kind} observation interface')
        ensure(env.get('movement_penalty')==0,f'Nonzero {kind} movement penalty; interpret net/gross separately')
        if model['power_model']=='pvwatts_sapm':
            for k in model:
                ensure(env.get(k)==model[k],f'Config {kind} mismatch in {k}')
    a=sac.algo
    ensure(a['gamma']==1.,'Changed finite-horizon gamma')
    ensure(a['hidden_sizes']==(64,64),'Changed hidden architecture')
    ensure(a['start_steps']>=0 and a['updates_per_step']>0 and a['reward_scale']>0,'Invalid SAC settings')
    # Note documentation vs code: this is a *mismatch*, not proof of a wrong power observation.
    # Runtime exact constants are authoritative; no new observations/sensors are added.
    from mbpo.env import pv_tracking
    norm=float(pv_tracking.POWER_NORM)
    irradiance_norm=float(pv_tracking.IRRADIANCE_NORM)
    ensure(norm>0 and irradiance_norm>0,'Invalid normalization constants')
    area=float(sac.train_env.get('area',1));eta=float(sac.train_env.get('efficiency',.18))
    from pathlib import Path as _Path
    # The adapter's hard-coded observation reconstruction must agree with the shared norm.
    # Locate it relative to the actual installed package, not to this diagnostic's file.
    adapter_src=(_Path(__import__('SAC_workspace.src.environment.environment',fromlist=['x']).__file__).read_text(encoding='utf-8'))
    if 'power_dc_w' in adapter_src and '/ 2000.' in adapter_src:
        ensure(abs(norm-2000.)<1e-9, 'SACEnvironment uses /2000 while simulator POWER_NORM has changed')
    return {'sac_mbpo_shared_config':True,'sac_algo':a,'mbpo_model_config':mbpo.model.__dict__,
            'train_env':sac.train_env,'eval_env':sac.eval_env,
            'normalization':{'actual_power_divisor_w':norm,
                'actual_irradiance_divisor_wm2':irradiance_norm,
                'documented_power_divisor_2000_area_eta_w':2000*area*eta,
                'documented_to_actual_divisor_ratio':norm/(2000*area*eta),
                'potential_issue':'Different from compact observation docstring; both real and synthetic models must use same constant.',
                'checkpoint_source_fingerprint_must_be_checked':True}}


def history_report(path,checkpoint_epoch=None,checkpoint_steps=None):
    if path is None:return None
    rows=read_csv(path)
    ensure(rows,f'Empty history {path}')
    prev_epoch=0;prev_steps=0;visited=set();scenarios=set()
    for row in rows:
        epoch=int(row['epoch']);steps=int(row['env_steps']);sid=str(row.get('scenario_id',''))
        ensure(sid not in ('', 'None'),f'Missing train scenario {path} epoch {epoch}')
        ensure(epoch==prev_epoch+1,f'Nonconsecutive training epoch {epoch} after {prev_epoch}')
        ensure(steps==prev_steps+HORIZON,f'Unexpected real interaction count at epoch {epoch}')
        for key in ('return_kwh','gross_energy_kwh','movement_cost_kwh'):
            scalar(row[key],f'{path}:{key}:{epoch}')
        ensure(abs(scalar(row['return_kwh'],'return')-
                   (scalar(row['gross_energy_kwh'],'gross')-scalar(row['movement_cost_kwh'],'movement')))<2e-5,
               f'Invalid real training reward accounting at episode {epoch}')
        scenarios.add(sid);prev_epoch=epoch;prev_steps=steps
    last=rows[-1]
    result={'episodes':len(rows),'real_steps':int(last['env_steps']),
            'unique_scenario_ids':len(scenarios),
            'last_alpha':float(last['alpha']),
            'unique_training_scenarios':scenarios,
            'training_date_repeats':len(rows)-len(scenarios)}
    if 'synthetic_generated' in last:
        result['synthetic_generated_total']=sum(int(float(r['synthetic_generated'])) for r in rows)
        result['episodes_with_model_optimizer_batch']=sum(
          math.isfinite(float(r['mean_model_batch'])) and float(r['mean_model_batch'])>0
          for r in rows if r.get('mean_model_batch','').strip().lower() not in ('','nan'))
        result['model_fits_latest']=int(last.get('model_fits',0))
    if checkpoint_epoch is not None:
        ensure(checkpoint_epoch <= result['episodes'],f'Checkpoint epoch exceeds history length {path}')
        ix=checkpoint_epoch-1
        ensure(ix>=0 and int(rows[ix]['env_steps'])==checkpoint_steps,'Checkpoint epoch/steps absent from history')
        result['checkpoint_matches_history_row']=True
        result['checkpoint_is_latest_history']=checkpoint_epoch==len(rows)
    return result


def _validate_replay(replay,norm,interval_hours,penalty,limit=50000):
    ensure(replay is not None,'Checkpoint lacks replay state')
    cap=int(replay['capacity']);cnt=int(replay['count']);head=int(replay['head'])
    ensure(cap>0 and 0<=cnt<=cap and 0<=head<cap,'Malformed replay metadata')
    ensure(cap<=5_000_000,'Unexpectedly large replay')
    ns=np.asarray(replay['next_states'],dtype=np.float64)
    rs=np.asarray(replay['rewards'],dtype=np.float64).reshape(-1)
    acts=np.asarray(replay['actions'],dtype=np.float64)
    done=np.asarray(replay['dones'],dtype=np.float64).reshape(-1)
    ensure(ns.shape==(cnt,6) and rs.shape==(cnt,) and acts.shape==(cnt,2) and done.shape==(cnt,),
           'Malformed replay data tensor shapes')
    ensure(np.isfinite(ns).all() and np.isfinite(rs).all() and np.isfinite(acts).all(),
           'Nonfinite replay data')
    ensure(np.all(np.abs(acts)<=1+1e-5),'Out-of-bounds replay actions')
    ensure(np.all((done==0)|(done==1)),'Non-binary replay terminal flag')
    if cnt==0:return {'count':0,'capacity':cap}
    ensure(np.all((ns[:,5]>=-1e-5)&(ns[:,5]<=1+1e-5)),'Invalid horizon feature')
    ensure(np.all((ns[:,2]>=-1e-6)&(ns[:,2]<=1+1e-6)),'Invalid power observation')
    # Only exact if the compact normalized power is not saturated; otherwise check non-negative gross.
    n_unsaturated=int(np.sum(ns[:,2]<1-1e-7))
    gross=ns[:,2]*norm*interval_hours/1000
    if penalty==0:
        ensure(np.allclose(rs[ns[:,2]<1-1e-7],gross[ns[:,2]<1-1e-7],atol=2e-6,rtol=2e-4),
               'Replay reward not reconstructed from next-state PV power and physical interval')
    if penalty>=0 and n_unsaturated:
        ensure(np.all(rs[ns[:,2]<1-1e-7]<=gross[ns[:,2]<1-1e-7]+2e-6),
               'Replay reward exceeds gross electrical energy on an unsaturated step')
    expected_done=ns[:,5]<=1e-7
    ensure(np.array_equal(done.astype(bool),expected_done),'Replay terminal and remaining-time mismatch')
    return {'count':cnt,'capacity':cap,'unsaturated_power_rows':n_unsaturated,
            'max_abs_reward_vs_next_power_kwh':float(np.max(np.abs(rs-gross))),
            'fraction_power_clipped':float(np.mean(ns[:,2]>=1-1e-7)),
            'terminal_count':int(np.sum(done))}


def inspect_checkpoint(path,kind,model,config,norm,history=None):
    import torch
    p=Path(path).resolve();ensure(p.is_file(),f'Missing {kind} checkpoint: {p}')
    data=torch.load(p,map_location='cpu',weights_only=False) # trusted local project checkpoints ONLY
    ensure(data.get('schema')==1,f'Invalid {kind} checkpoint schema')
    if kind=='sac':
        saved_config=config['sac_config']; expected_hash=canon_hash(saved_config)
        ensure(data.get('hash')==expected_hash,'SAC checkpoint does not match currently resolved SAC configuration')
    else:
        saved_config=config['mbpo_config'];expected_hash=canon_hash(saved_config)
        ensure(data.get('config_hash')==expected_hash,'MBPO checkpoint configuration hash mismatch')
        ensure(data.get('config')==saved_config,'MBPO checkpoint embedded config differs')
    saved_mdp=data.get('resolved_mdp');ensure(isinstance(saved_mdp,dict),'Checkpoint lacks resolved physical metadata')
    ensure(saved_mdp.get('power_model','legacy')==model['power_model'],'Checkpoint model differs from evaluation model')
    for field in ('wind_reference_height_m','gamma_pdc','sapm_mount','wind_powerlaw_exponent'):
        if model['power_model']=='pvwatts_sapm':ensure(saved_mdp.get(field)==model[field],f'Checkpoint/evaluation {field} mismatch')
    epoch=int(data['epoch']);steps=int(data['env_steps']);ensure(epoch>0 and steps==epoch*HORIZON,'Checkpoint epoch/step count mismatch')
    result={'path':str(p),'sha256_now':file_sha(p),'kind':kind,'checkpoint_epoch':epoch,
            'checkpoint_real_steps':steps,'seed':int(data['seed']),
            'config_hash_valid':True,
            'physical_config':{k:saved_mdp.get(k) for k in PHYSICS_KEYS}}
    # Check local sidecar metadata when present; even if the user passed a renamed snapshot.
    for name in ('config.json','resolved_mdp.json'):
        f=p.parent/name
        if f.is_file():
            sidecar=read_json(f)
            ensure(sidecar==(json.loads(json.dumps(saved_config)) if name=='config.json' else saved_mdp),f'Checkpoint and {f} differ')
            result['sidecar_'+name+'_validated']=True
        else:result['sidecar_'+name+'_validated']=False
    ag=data.get('agent');ensure(isinstance(ag,dict) and 'actor' in ag and 'critic' in ag,'Incomplete SAC actor/critic state')
    for field in ('gamma','tau','reward_scale','target_entropy','min_alpha'):
        ensure(abs(float(ag[field])-float(config['sac_algo'][field]))<1e-10,f'Checkpoint SAC loss parameter mismatch: {field}')
    logalpha=ag.get('log_alpha');ensure(logalpha is not None,'Missing learnable SAC temperature')
    alpha=math.exp(float(logalpha.reshape(-1)[0])); min_a=float(config['sac_algo']['min_alpha'])
    result['alpha_effective']=max(alpha,min_a)
    result['alpha_raw']=alpha
    result['alpha_at_floor']=alpha<=min_a*(1+1e-4)
    # Compare source bytes exactly with the fingerprint stored at training time.
    if model['power_model']=='pvwatts_sapm':
        from mbpo.env import pv_power_backend, pvlib_physics, pv_tracking
        sources=[Path(z.__file__) for z in (pv_power_backend,pvlib_physics,pv_tracking)]
        sh=hashlib.sha256()
        for fp in sources:sh.update(fp.read_bytes())
        ensure(sh.hexdigest()==saved_mdp.get('power_source_sha256'),
               f'{kind} checkpoint was built with different PVWatts/environment source bytes')
        for fp,key in zip(sources[:2],('pv_power_backend_sha256','pvlib_physics_sha256')):
            ensure(file_sha(fp)==saved_mdp.get(key),f'{kind} {key} does not match current source file')
        result['physical_source_bytes_match_checkpoint']=True
    if history:
        result['history']=history_report(history,epoch,steps)
        if result['history']['checkpoint_is_latest_history']:
            ensure(abs(result['alpha_effective']-result['history']['last_alpha'])<=2e-4,'Checkpoint alpha mismatch versus training history')
    if kind=='sac':
        result['replay']=_validate_replay(data.get('replay'),norm,5/60,float(config['sac_train_env']['movement_penalty']))
    else:
        result['real_replay']=_validate_replay(data.get('real_pool'),norm,5/60,float(config['sac_train_env']['movement_penalty']))
        result['model_replay']=_validate_replay(data.get('model_pool'),norm,5/60,float(config['sac_train_env']['movement_penalty']))
    return result


def audit(eval_dir,*,sac_checkpoint=None,mbpo_checkpoint=None,sac_history=None,mbpo_history=None,require_checkpoints=False):
    root=Path(eval_dir)
    summary=read_json(root/'summary.json');plan_file=root/'scenario_selection.json'
    plan=read_json(plan_file) if plan_file.is_file() else None
    episodes=read_csv(root/'episodes.csv')
    model=model_fields(summary)
    gains=sample_weighting(episodes,plan,summary)
    cfg_evidence=config_evidence(model,summary)
    from SAC_workspace.src.config.config import load_config as load_sac
    from MBPO_workspace.src.config import load_config as load_mbpo
    kw={k:v for k,v in model.items() if v is not None}
    sac=load_sac(require_data=False,**kw)
    mbpo=load_mbpo(require_data=False,**kw)
    params={'sac_config':sac.to_dict(),'mbpo_config':mbpo.to_dict(),
            'sac_algo':sac.algo,'sac_train_env':sac.train_env}
    reports={}
    if sac_checkpoint:reports['sac']=inspect_checkpoint(sac_checkpoint,'sac',model,params,cfg_evidence['normalization']['actual_power_divisor_w'],sac_history)
    if mbpo_checkpoint:reports['mbpo']=inspect_checkpoint(mbpo_checkpoint,'mbpo',model,params,cfg_evidence['normalization']['actual_power_divisor_w'],mbpo_history)
    if require_checkpoints:ensure(set(reports)=={'sac','mbpo'},'Both checkpoints required for provenance verdict')
    histories={}
    if sac_history:histories['sac']=history_report(sac_history)
    if mbpo_history:histories['mbpo']=history_report(mbpo_history)
    overlap={}
    for name,info in histories.items():
        d=info['unique_training_scenarios']; overlap[name]={'evaluated_days_seen_in_supplied_history':len(d & gains['selected_ids']),
          'example_overlap':sorted(d & gains['selected_ids'])[:12],
          'history_is_later_than_checkpoint': bool(name in reports and info['episodes']>reports[name]['checkpoint_epoch'])}
    budgets={}
    if set(reports)=={'sac','mbpo'}:
        sac_steps=reports['sac']['checkpoint_real_steps']
        mbpo_steps=reports['mbpo']['checkpoint_real_steps']
        budgets={'sac_checkpoint_real_steps':sac_steps,'mbpo_checkpoint_real_steps':mbpo_steps,
                 'equal_real_interaction_budget':sac_steps==mbpo_steps,
                 'mbpo_to_sac_real_interaction_ratio':mbpo_steps/sac_steps}
    elif set(histories)=={'sac','mbpo'}:
        budgets={'sac_history_real_steps':histories['sac']['real_steps'],
                 'mbpo_history_real_steps':histories['mbpo']['real_steps'],
                 'warning':'History lengths alone do not identify which checkpoint generated an older evaluation.'}
    verdict={'numerical_report_and_weighted_gains':'PASS',
             'checkpoint_provenance':'NOT VERIFIED: evaluation output did not record checkpoint SHA; later checkpoint hash cannot establish original evaluation identity',
             'training_history_overlap':'UNKNOWN' if not histories else ('WARNING' if any(v['evaluated_days_seen_in_supplied_history'] for v in overlap.values()) else 'NONE IN SUPPLIED HISTORY'),
             'observation_normalization':'DOCUMENTATION MISMATCH, NOT PROVEN TRAINING BUG'}
    return {'eval_dir':str(root.resolve()),'verdict':verdict,'physics':model,
            'gains':{k:v for k,v in gains.items() if k!='selected_ids'},
            'effective_configuration':cfg_evidence,
            'checkpoints':reports,'training_histories':{k:{j:v for j,v in h.items() if j!='unique_training_scenarios'} for k,h in histories.items()},
            'overlap':overlap,'training_budget_comparison':budgets,
            'evaluation_selection':{'method':'CLI checkpoint paths; no automatic best-checkpoint selection in paired evaluation',
               'eval_seed':gains['seed'],
               'training_random_initial_pose':bool(cfg_evidence['train_env'].get('randomize_initial_orientation')),
               'evaluation_random_initial_pose':bool(cfg_evidence['eval_env'].get('randomize_initial_orientation')),
               'same_obs_and_action_spaces':True},
            'limitations':[
              'The evaluation summary does not embed SAC/MBPO checkpoint SHA256 or checkpoint epoch. Hashes calculated now cannot prove which snapshot generated a historical report.',
              'The exact original checkpoints are required for actor-action replay. Use sac_policy_forensic.py for the definitive action check.',
              'No changes are made to model, reward, observation, simulation, evaluation or training.' ]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir',required=True)
    p.add_argument('--sac-checkpoint',default=None,help='Trusted local SAC .pt')
    p.add_argument('--mbpo-checkpoint',default=None,help='Trusted local MBPO .pt')
    p.add_argument('--sac-training-csv',default=None)
    p.add_argument('--mbpo-training-csv',default=None)
    p.add_argument('--out',default=None)
    p.add_argument('--require-checkpoints',action='store_true')
    args=p.parse_args()
    report=audit(args.eval_dir,sac_checkpoint=args.sac_checkpoint,mbpo_checkpoint=args.mbpo_checkpoint,
        sac_history=args.sac_training_csv,mbpo_history=args.mbpo_training_csv,
        require_checkpoints=args.require_checkpoints)
    out=Path(args.out or Path(args.eval_dir)/'sac_forensic'/'checkpoint_config_audit.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True,default=str),encoding='utf-8')
    print('Numerical report:',report['verdict']['numerical_report_and_weighted_gains'])
    for key,stat in report['gains']['paired_gains'].items():
        print(f"{key}: {stat['weighted_difference_kwh']:+.6f} kWh / {stat['ratio_of_weighted_means_percent']:+.3f}% ({stat['winning_days']} of {stat['evaluated_days']} days)")
    for name,ck in report['checkpoints'].items():
        print(f"{name}: checkpoint epoch={ck['checkpoint_epoch']}, real steps={ck['checkpoint_real_steps']}, alpha={ck['alpha_effective']:.6g}")
    print('Provenance:',report['verdict']['checkpoint_provenance'])
    print('Saved:',out)

if __name__=='__main__':main()
