"""Paired learned-policy and PV baseline evaluation; no model rollouts in eval.
Every *feasible* controller is stepped using the same PVTrackingEnv, NSRDB
scenario, UTC clock, and physical initial orientation. An unconstrained
instantaneous dual-axis reference reports GROSS energy only.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import csv
import json
import numpy as np
import torch
from SAC_workspace.src.environment.environment import SACEnvironment
from SAC_workspace.src.agents.sac_agent import SACAgent
from SAC_workspace.src.evaluation.evaluation import (
    run_episode, ideal_instantaneous_dual_axis, load_agent, _ci_clustered, reference_scenario_ids,
)
from MBPO_workspace.src.config import MBPOConfig

METHODS=('mbpo','sac','fixed_no_motion','dual_axis_slew_limited',
         'dual_axis_instant_ideal','poa_instant_ideal')
FEASIBLE=METHODS[:4]
PHYSICS_KEYS=('pvlib_version','gym_version','site_latitude','site_longitude','site_altitude',
    'tz','start_time','periods','freq','weather_source','area','efficiency',
    'max_delta_tilt','max_delta_azimuth','movement_penalty','movement_cost_mode',
    'actuator_power_w','slew_rate_tilt_deg_s','slew_rate_azimuth_deg_s',
    'observation_mode','horizon')
PVWATTS_PHYSICS_KEYS=('power_model','wind_reference_height_m','gamma_pdc',
    'sapm_mount','wind_powerlaw_exponent','power_source_sha256',
    'pv_power_backend_sha256','pvlib_physics_sha256')


def _check_physics(saved, actual):
    if not saved: raise ValueError('Checkpoint lacks effective MDP metadata')
    if (saved.get('power_model', 'legacy') != actual.get('power_model', 'legacy')):
        raise ValueError('Physical MDP mismatch: checkpoint and evaluator use different power models')
    keys = PHYSICS_KEYS + (PVWATTS_PHYSICS_KEYS if actual.get('power_model') == 'pvwatts_sapm' else ())
    for k in keys:
        if saved.get(k)!=actual.get(k):
            raise ValueError(f'Physical MDP mismatch for {k}: {saved.get(k)} != {actual.get(k)}')


def load_mbpo_agent(cfg: MBPOConfig, checkpoint,device='cpu'):
    """For evaluation only the MBPO-trained SAC actor/critic are needed."""
    data=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if data.get('schema')!=1 or data.get('config',{}).get('sac')!=cfg.sac.to_dict():
        raise ValueError('MBPO checkpoint does not match shared reference MDP')
    a=cfg.sac.algo
    agent=SACAgent(hidden_sizes=a['hidden_sizes'],lr=a['lr'],gamma=a['gamma'],
        tau=a['tau'],reward_scale=a['reward_scale'],alpha_init=a['alpha_init'],
        target_entropy=a['target_entropy'],min_alpha=a['min_alpha'],device=device)
    agent.load_state_dict(data['agent']);agent.actor.eval()
    agent.saved_mdp=data['resolved_mdp']
    return agent


def ideal_instantaneous_poa(cfg, fixed_path, env=None):
    """Instantaneous, actuator-unconstrained maximum POA under the project's
    default *isotropic* transposition model. The POA-optimal tilt is
    computed analytically, with azimuth facing the sun. Under PVWatts the
    temperature-corrected DC model must be monotone in POA; this is verified
    before reporting a certified instantaneous gross-DC upper bound. No
    action/slew limits apply and this is not a feasible controller.
    """
    from mbpo.env.baseline_contract import (
        isotropic_max_poa_tilt, certify_pvwatts_monotonicity)
    from mbpo.env.pv_power_backend import compute_pv_power
    owns = env is None
    if owns: env = SACEnvironment(cfg.eval_env)
    p={'method':'poa_instant_ideal','scenario_id':fixed_path['scenario_id'],
       'seed':fixed_path['seed'], 'initial_pose':list(fixed_path['initial_pose']),
       'reward':[],'energy':[],'power':[],'cost':[],'tilt':[],'azimuth':[],
       'action_tilt':[],'action_azimuth':[],'delta_tilt_deg':[],
       'delta_azimuth_deg':[],'infos':[]}
    try:
        u=getattr(env,'unwrapped',env)
        area=float(u.area);efficiency=float(u.efficiency)
        hours=float(getattr(u,'interval_hours',5./60.))
        infos=fixed_path['infos']
        if not infos or len(infos)!=len(fixed_path['energy']):
            raise ValueError('POA reference requires complete fixed-mount recorded weather')
        for info in infos:
            for key in ('solar_zenith_deg','solar_azimuth_deg','dni_wm2','ghi_wm2','dhi_wm2'):
                if info.get(key) is None: raise ValueError('POA reference weather missing '+key)
            az=float(info['solar_azimuth_deg'])%360.
            def physical(tilt):
                return compute_pv_power(float(tilt),az,info['solar_zenith_deg'],
                    info['solar_azimuth_deg'],info['dni_wm2'],info['ghi_wm2'],info['dhi_wm2'],
                    area=area, efficiency=efficiency, power_model=getattr(u,'power_model','legacy'),
                    temp_air_c=info.get('temperature_c',info.get('ambient_temperature_c')),
                    wind_speed_ms=info.get('wind_speed_m_s'),
                    wind_reference_height_m=getattr(u,'wind_reference_height_m',None),
                    gamma_pdc=getattr(u,'gamma_pdc',-0.004),
                    sapm_mount=getattr(u,'sapm_mount','open_rack_glass_polymer'),
                    wind_powerlaw_exponent=getattr(u,'wind_powerlaw_exponent',0.14))
            # The monofacial isotropic POA maximum is analytic, not a local
            # bounded-search approximation. With the current SAPM temperature
            # law, the PVWatts DC equation is monotone in POA over the
            # operating interval only when certified below.
            tilt=isotropic_max_poa_tilt(info['solar_zenith_deg'],
                info['dni_wm2'],info['ghi_wm2'],info['dhi_wm2'])
            selected=physical(tilt)
            if getattr(u,'power_model','legacy') == 'pvwatts_sapm':
                certify_pvwatts_monotonicity(
                    selected,info.get('temperature_c',info.get('ambient_temperature_c')),
                    getattr(u,'gamma_pdc',-0.004))
            power=float(selected.power_w)
            energy=power*hours/1000.
            # An unconstrained gross-power oracle must dominate the feasible
            # fixed orientation under IDENTICAL recorded weather. Catch
            # optimizer/model regressions before publishing an upper bound.
            fixed_power=fixed_path.get('power')
            if fixed_power is not None:
                if len(fixed_power) != len(infos):
                    raise ValueError('Incomplete paired fixed-power measurements')
                reference=float(fixed_power[len(p['power'])])
                if not np.isfinite(reference) or power + max(1e-5,1e-7*abs(reference)) < reference:
                    raise RuntimeError('POA gross oracle fell below paired fixed power')
            p['power'].append(power);p['energy'].append(energy)
            p['reward'].append(energy);p['cost'].append(0.)
            p['tilt'].append(float(tilt));p['azimuth'].append(float(az))
            for key in ('action_tilt','action_azimuth','delta_tilt_deg','delta_azimuth_deg'):
                p[key].append(float('nan'))
            metadata=dict(info)
            result=selected
            metadata.update(power=power,power_dc_w=power,
                poa_global=result.poa_global_wm2,
                effective_irradiance_wm2=result.effective_irradiance_wm2,
                cell_temperature_c=result.cell_temperature_c,
                wind_speed_10m_ms=result.wind_speed_10m_ms,
                power_model=getattr(u,'power_model','legacy'),
                tilt=float(tilt),azimuth=float(az),
                cos_aoi=(u._cos_angle_of_incidence(info['solar_zenith_deg'],
                    info['solar_azimuth_deg'],float(tilt),float(az))
                    if hasattr(u,'_cos_angle_of_incidence') else None))
            p['infos'].append(metadata)
    finally:
        if owns:env.close()
    p['gross_energy_kwh']=float(sum(p['energy']))
    p['net_energy_kwh']=None;p['movement_cost_kwh']=None
    return p


def evaluate(cfg:MBPOConfig, mbpo_checkpoint, sac_checkpoint, scenarios=None,
             seeds=(42,),device='cpu',include_poa=True,include_classical=True):
    from mbpo.env.nsrdb_weather import load_scenario_manifest
    from mbpo.env.baseline_contract import validate_paired_reference_group
    known={s['scenario_id'] for s in load_scenario_manifest(cfg.sac.manifest)['scenarios']}
    if scenarios is None: scenarios=reference_scenario_ids(cfg.sac)
    scenarios=list(scenarios);seeds=list(seeds)
    if not scenarios or len(set(scenarios))!=len(scenarios) or set(scenarios)-known:
        raise ValueError('Invalid, missing, or duplicate evaluation scenario IDs')
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Duplicate or missing evaluation seeds')
    if len(seeds)>1 and not cfg.sac.eval_env.get('randomize_initial_orientation',False):
        raise ValueError('Identical fixed-pose daily scenarios must not be counted as extra independent seeds')
    mbpo=load_mbpo_agent(cfg,mbpo_checkpoint,device=device)
    sac=load_agent(cfg.sac,sac_checkpoint,device=device)
    env=SACEnvironment(cfg.sac.eval_env)
    try:
        _check_physics(mbpo.saved_mdp,env.resolved_mdp)
        _check_physics(sac.checkpoint_resolved_mdp,env.resolved_mdp)
        paths=[]
        for sid in scenarios:
            for seed in seeds:
                # Reuse one NSRDB weather cache; reset to identical scenario/seed.
                fixed=run_episode(cfg.sac,'fixed_no_motion',sid,seed,env=env)
                p_sac=run_episode(cfg.sac,'sac',sid,seed,agent=sac,env=env)
                p_mbpo=run_episode(cfg.sac,'sac',sid,seed,agent=mbpo,env=env)
                p_mbpo['method']='mbpo'
                group=(p_mbpo,p_sac,fixed)
                if include_classical:
                    dual=run_episode(cfg.sac,'dual_axis_slew_limited',sid,seed,env=env)
                    ideal=ideal_instantaneous_dual_axis(cfg.sac,fixed,env=env)
                    group=group+(dual,ideal)
                if include_poa:group=group+(ideal_instantaneous_poa(cfg.sac,fixed,env=env),)
                expected=[i['timestamp_utc_iso'] for i in fixed['infos']]
                for p in group:
                    if (p['scenario_id'],p['seed'])!=(sid,seed):raise RuntimeError('Scenario mismatch in paired rollout')
                    if len(p['power'])!=env.horizon:raise RuntimeError('Incomplete evaluation horizon')
                    if not np.allclose(p['initial_pose'],fixed['initial_pose'],atol=1e-8):
                        raise RuntimeError('Unequal paired initial orientation')
                    if [i['timestamp_utc_iso'] for i in p['infos']]!=expected:
                        raise RuntimeError('Mismatch in five-minute UTC episode clocks')
                validate_paired_reference_group(
                    {p['method']:p for p in group}, require_oracle=include_poa)
                paths.extend(group)
    finally:
        env.close()
    return paths


def paired_summary(paths, plan=None):
    keys=defaultdict(dict)
    for p in paths:
        key=(p['scenario_id'],int(p['seed']))
        if p['method'] in keys[key]:raise ValueError('Duplicate evaluation method for a scenario')
        keys[key][p['method']]=p
    for key, group in keys.items():
        allowed=[{'mbpo','sac','fixed_no_motion'},
                 {'mbpo','sac','fixed_no_motion','poa_instant_ideal'},
                 set(METHODS),set(METHODS)-{'poa_instant_ideal'}]
        if set(group) not in allowed:raise ValueError(f'Missing paired methods for {key}')
    first_group=next(iter(keys.values()))
    first_path=next(iter(first_group.values()))
    evaluated_horizon=len(first_path['power'])
    out={'protocol':('Paired by NSRDB scenario, seed, initial pose and post-step UTC timestamps; '
           'real PVTrackingEnv; {} transitions; no model-generated actions in evaluation').format(evaluated_horizon),
         'scenario_count':len({k[0] for k in keys}),
         'paired_episodes':len(keys),'methods':{},'paired_net_kwh':{},
         'notes':['Instantaneous dual-axis is a sun-pointing GROSS-only reference, '
                  'not a globally optimized POA oracle, and not actuator-feasible.',
                  'Instantaneous DC reference maximizes single-step gross DC power for the '
                  'isotropic, monotone-PVWatts model (or legacy POA model); it is '
                  'unconstrained and not a finite-horizon net-energy policy.',
                  'Training and evaluation scenarios may overlap; do not claim a held-out result '
                  'without a disjoint training protocol.']}
    available=[m for m in METHODS if m in next(iter(keys.values()))]
    if plan is not None:
        from SAC_workspace.src.evaluation.statistical_inference import (
            day_summary_from_paths, stratified_point_estimate,
            build_valid_statistical_summary)
        verified=build_valid_statistical_summary(
            paths,plan,method_names=tuple(
                m for m in available if m in ('mbpo','sac','fixed_no_motion','poa_instant_ideal')))
        out['sampling_protocol']=plan['protocol']
        out['sampling_uncertainty']=verified['uncertainty']
        out['notes'].append('Population-weighted point estimates; no naive day-bootstrap CI for one-per-bin sampling.')
    for method in available:
        records=[g[method] for g in keys.values()]
        gross=np.asarray([p['gross_energy_kwh'] for p in records],float)
        net=np.asarray([p['net_energy_kwh'] for p in records if p['net_energy_kwh'] is not None],float)
        movement=np.asarray([p['movement_cost_kwh'] for p in records if p['movement_cost_kwh'] is not None],float)
        if plan is not None:
            day_gross=day_summary_from_paths(records,method,'gross_energy_kwh')
            mu_gross=stratified_point_estimate(day_gross,plan)
            day_net=(day_summary_from_paths(records,method,'net_energy_kwh')
                     if len(net) else None)
            mu_net=stratified_point_estimate(day_net,plan) if day_net is not None else None
            day_move=(day_summary_from_paths(records,method,'movement_cost_kwh')
                      if len(movement) else None)
            mu_move=stratified_point_estimate(day_move,plan) if day_move is not None else None
            day_net_values=list(day_net.values()) if day_net is not None else []
            descriptive_sd=float(np.std(day_net_values,ddof=1)) if len(day_net_values)>1 else None
        else:
            mu_gross=float(gross.mean());mu_net=float(net.mean()) if len(net) else None
            mu_move=float(movement.mean()) if len(movement) else None
            descriptive_sd=float(net.std(ddof=1)) if len(net)>1 else None
        out['methods'][method]={'count':len(records),
          'mean_gross_kwh':mu_gross,'mean_net_kwh':mu_net,
          'std_net_kwh':descriptive_sd,
          'std_interpretation':'unweighted sampled-day descriptive SD; not the standard error' if plan is not None else 'sample descriptive SD; not the standard error',
          'mean_movement_cost_kwh':mu_move}
    comparisons=[('mbpo','sac'),('mbpo','dual_axis_slew_limited'),
                 ('mbpo','fixed_no_motion'),('sac','dual_axis_slew_limited'),
                 ('sac','fixed_no_motion')]
    comparisons=[(a,b) for a,b in comparisons if a in available and b in available]
    for lhs,rhs in comparisons:
        clustered=defaultdict(list);ratios=[]
        for (sid,_),g in keys.items():
            left=g[lhs]['net_energy_kwh'];right=g[rhs]['net_energy_kwh']
            clustered[sid].append(left-right)
            if right>1e-10:ratios.append(100.*(left-right)/right)
        if plan is None:
            mean,lo,hi=_ci_clustered(clustered)
            out['paired_net_kwh'][f'{lhs}_minus_{rhs}']={
                'mean_difference_kwh':mean,'bootstrap_ci_95pct_kwh':[lo,hi],
                'mean_percent_gain':float(np.mean(ratios)) if ratios else None}
        else:
            from SAC_workspace.src.evaluation.statistical_inference import (
                day_summary_from_paths, stratified_point_estimate)
            a=day_summary_from_paths(paths,lhs,'net_energy_kwh')
            b=day_summary_from_paths(paths,rhs,'net_energy_kwh')
            differences={sid:a[sid]-b[sid] for sid in a}
            mu=stratified_point_estimate(differences,plan)
            denominator=stratified_point_estimate(b,plan)
            out['paired_net_kwh'][f'{lhs}_minus_{rhs}']={
                'mean_difference_kwh':mu,
                'bootstrap_ci_95pct_kwh':[None,None],
                'ci_status':'not estimated: one sampled day per temporal bin',
                'mean_percent_gain':None,
                'ratio_of_weighted_means_percent':100.*mu/denominator if denominator>1e-10 else None}
    return out


def write_outputs(paths,outdir,plan=None):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    stats=paired_summary(paths,plan=plan)
    models={info.get('power_model','legacy') for p in paths for info in p['infos']}
    if len(models)!=1:
        raise ValueError('Mixed electrical power models in the same evaluation output')
    stats['power_model']=next(iter(models))
    # Record the power physics in summaries, not only in checkpoint snapshots.
    fields=('wind_reference_height_m','gamma_pdc','sapm_mount','wind_powerlaw_exponent')
    declared={tuple((k,info.get(k)) for k in fields)
              for p in paths for info in p['infos']}
    if len(declared)!=1:
        raise ValueError('Inconsistent PV temperature/wind parameters across evaluation paths')
    stats['power_physics']={'model':stats['power_model'],
         **dict(next(iter(declared)))}
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    (out/'summary.json').write_text(json.dumps(clean(stats),indent=2,sort_keys=True,allow_nan=False),encoding='utf-8')
    episode_fields=['scenario_id','seed','method','gross_energy_kwh','movement_cost_kwh',
        'net_energy_kwh','season','initial_tilt_deg','initial_azimuth_deg']
    with (out/'episodes.csv').open('w',newline='',encoding='utf-8') as file:
        w=csv.DictWriter(file,fieldnames=episode_fields);w.writeheader()
        for p in paths:
            w.writerow(dict(scenario_id=p['scenario_id'],seed=p['seed'],method=p['method'],
                 gross_energy_kwh=p['gross_energy_kwh'],movement_cost_kwh=p['movement_cost_kwh'],
                 net_energy_kwh=p['net_energy_kwh'],season=p['infos'][0].get('season'),
                 initial_tilt_deg=p['initial_pose'][0],initial_azimuth_deg=p['initial_pose'][1]))
    traj_fields=['scenario_id','seed','method','step','utc_iso','power_w',
        'energy_kwh','cumulative_gross_kwh','tilt_deg','azimuth_deg','movement_cost_kwh']
    with (out/'trajectories.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=traj_fields);w.writeheader()
        for p in paths:
            e=0.
            for i,info in enumerate(p['infos']):
                e+=p['energy'][i]
                w.writerow(dict(scenario_id=p['scenario_id'],seed=p['seed'],method=p['method'],
                   step=i+1,utc_iso=info['timestamp_utc_iso'],power_w=p['power'][i],
                   energy_kwh=p['energy'][i],cumulative_gross_kwh=e,
                   tilt_deg=p['tilt'][i],azimuth_deg=p['azimuth'][i],
                   movement_cost_kwh=p['cost'][i]))
    return stats
