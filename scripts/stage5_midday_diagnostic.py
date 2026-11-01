#!/usr/bin/env python3
"""Read-only Stage 5 SAC/MBPO midday, clock, geometry, critic and model diagnosis.

No optimizer.step, no checkpoint writes, no weather generation, no actor mutation.
Only --outdir receives writes. Do not load untrusted torch checkpoint files.
Requires project's installed Stage-A physics and both real policy checkpoints.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
# Imports during the diagnostic must not generate __pycache__ files in the repo.
sys.dont_write_bytecode = True
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
H = 117
TZ = ZoneInfo('America/New_York')
REQUIRED = ('scenario_id','seed','method','step','utc_iso','power_w','energy_kwh',
            'tilt_deg','azimuth_deg','action_tilt','action_azimuth',
            'dni_wm2','dhi_wm2','solar_zenith_deg','solar_azimuth_deg')
METHODS = ('sac','mbpo','fixed_no_motion')


def demand(ok, msg):
    if not ok:
        raise ValueError(msg)


def finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def safely_json(x):
    if isinstance(x, dict):return {str(k):safely_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safely_json(v) for v in x]
    if isinstance(x,np.ndarray):return safely_json(x.tolist())
    if isinstance(x,(np.float32,np.float64,float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x,(np.integer,int)):return int(x)
    if isinstance(x,(bool,np.bool_)):return bool(x)
    return x


def save_csv(path, rows, columns=None):
    if not rows:return
    cols=columns or list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def parse_utc(x):
    """Require a real offset-aware timestamp; convert to UTC (never string-sort clocks)."""
    try:
        dt=datetime.fromisoformat(str(x).strip().replace('Z','+00:00'))
    except ValueError as exc:
        raise ValueError(f'Invalid ISO timestamp: {x!r}') from exc
    demand(dt.tzinfo is not None and dt.utcoffset() is not None,
           f'Naive UTC timestamp not acceptable: {x!r}')
    demand(dt.utcoffset()==timedelta(0),
           f'Expected exported UTC timestamps, got non-UTC offset: {x!r}')
    return dt.astimezone(timezone.utc)


def signed_angle(a,b):
    return (float(a)-float(b)+180.)%360.-180.


def solar_cos_aoi(beta,gamma,zenith,solaraz):
    b,z,d=np.deg2rad([beta,zenith,signed_angle(gamma,solaraz)])
    return float(np.cos(z)*np.cos(b)+np.sin(z)*np.sin(b)*np.cos(d))


def materialize_trajectories(path):
    file=Path(path)
    demand(file.is_file(),f'Missing paired detailed trajectory CSV: {file}')
    with file.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        demand(reader.fieldnames is not None,'Empty CSV header')
        missing=set(REQUIRED)-set(reader.fieldnames)
        demand(not missing,f'Trajectory missing required fields: {sorted(missing)}')
        rows=list(reader)
    demand(rows,'No evaluation trajectory records')
    groups=defaultdict(dict)
    for raw in rows:
        m=raw['method']
        if m not in METHODS+('poa_instant_ideal',):continue
        try:seed=int(raw['seed']);step=int(raw['step'])
        except (TypeError,ValueError) as exc:raise ValueError('Invalid seed/step in detailed trajectory') from exc
        key=(raw['scenario_id'],seed,m)
        slots=groups[key]
        demand(step not in slots,f'Duplicate detailed trajectory step: {key}/{step}')
        vals={col:finite(raw.get(col)) for col in raw}
        vals.update(scenario_id=key[0],seed=key[1],method=m,step=step,
                    datetime_utc=parse_utc(raw['utc_iso']))
        slots[step]=vals
    bypair=defaultdict(dict)
    for (sid,seed,m),slots in groups.items():
        demand(set(slots)==set(range(1,H+1)),f'Incomplete 117-step trajectory {(sid,seed,m)}')
        rr=[slots[j] for j in range(1,H+1)]
        times=[x['datetime_utc'] for x in rr]
        demand(all(t2-t1==timedelta(minutes=5) for t1,t2 in zip(times[:-1],times[1:])),
               f'UTC timestamp spacing not 5 minutes {(sid,seed,m)}')
        demand(times[0].strftime('%H:%M')=='12:05' and times[-1].strftime('%H:%M')=='21:45',
               f'Unexpected 12:05..21:45 UTC exported reward window {(sid,seed,m)}')
        demand(all(t.date().isoformat()==sid for t in times),f'UTC date not matching scenario: {sid}')
        demand(all(x['energy_kwh'] is not None and x['power_w'] is not None and
                   abs(x['energy_kwh']-x['power_w']/12000.)<=1e-5 for x in rr),
               f'Power-to-energy mismatch: {(sid,seed,m)}')
        demand(all(x['power_w']>=-1e-6 for x in rr),f'Negative power: {(sid,seed,m)}')
        bypair[(sid,seed)][m]=rr
    demand(bypair,'No evaluation trajectories found')
    for key,g in bypair.items():
        demand(set(METHODS).issubset(g),f'Missing paired SAC/MBPO/fixed: {key}; {sorted(g)}')
        for m in METHODS:
            demand(all(x['action_tilt'] is not None and x['action_azimuth'] is not None for x in g[m]),
                   f'Missing evaluated actions: {key}/{m}')
        master=[x['datetime_utc'] for x in g['fixed_no_motion']]
        for m,rr in g.items():
            demand([x['datetime_utc'] for x in rr]==master,
                   f'Unpaired UTC clock for {(key,m)}')
    return dict(bypair)


def identify_noon(rr):
    # Noon is defined by minimum solar zenith within the actual recorded 5-min grid.
    valid=[(x['solar_zenith_deg'],i) for i,x in enumerate(rr) if x['solar_zenith_deg'] is not None]
    demand(len(valid)==H,'Missing solar zenith for noon computation')
    idx=min(valid)[1]
    return idx


def clock_diagnosis(grouped,latitude=None,longitude=None):
    rows=[];failures=[]
    for (sid,seed),g in sorted(grouped.items()):
        rr=g['fixed_no_motion'];i=identify_noon(rr)
        noon=rr[i]['datetime_utc']
        native=noon.astimezone(TZ)
        first=rr[0]['datetime_utc'];last=rr[-1]['datetime_utc']
        fixed_start=first-timedelta(minutes=5)
        offset=native.utcoffset()
        noon_alt=90.-rr[i]['solar_zenith_deg']
        bad_geom=sum(1 for k in range(H) if abs(signed_angle(g['sac'][k]['solar_azimuth_deg'],rr[k]['solar_azimuth_deg']))>1e-5)
        day=datetime.fromisoformat(sid)
        local_midnight=datetime(day.year,day.month,day.day,tzinfo=TZ)
        next_midnight=datetime.fromordinal(day.date().toordinal()+1).replace(tzinfo=TZ)
        local_calendar_day_hours=(next_midnight.astimezone(timezone.utc)-local_midnight.astimezone(timezone.utc)).total_seconds()/3600.
        rows.append(dict(scenario_id=sid,seed=seed,utc_first_reward=first.isoformat(),
            utc_last_reward=last.isoformat(),utc_reset=fixed_start.isoformat(),
            ny_reset=fixed_start.astimezone(TZ).isoformat(),ny_noon=native.isoformat(),
            utc_grid_solar_noon=noon.isoformat(),offset_hours=offset.total_seconds()/3600.,
            local_calendar_day_hours=local_calendar_day_hours,
            episode_utc_duration_hours=H*5/60.,
            local_episode_wall_clock_duration_hours=(last.astimezone(TZ).replace(tzinfo=None)-fixed_start.astimezone(TZ).replace(tzinfo=None)).total_seconds()/3600.,
            dst=bool(native.dst()),utc_noon_step=i+1,
            noon_solar_zenith_deg=rr[i]['solar_zenith_deg'],
            noon_solar_altitude_deg=noon_alt,
            reward_window_hours=H*5./60.,
            clipped_daylight_energy_caution='Not a full 24h or sunrise-to-sunset integration',
            solar_geometry_mismatch_sac_count=bad_geom))
    return rows


def daylight_duration_by_day(scenario_date, latitude, longitude):
    """5-minute daylight coverage for local solar day; DST handled by UTC conversion.

    Use 04:00 UTC through next 04:00 UTC so a New York summer sunset
    shortly after 00:00 UTC belongs to the correct solar date.
    Daylight hours count is descriptive, not missing PV energy.
    """
    import pvlib
    date=pd.Timestamp(scenario_date, tz='UTC')
    time_grid=pd.date_range(date+pd.Timedelta(hours=4),
          date+pd.Timedelta(days=1,hours=4),freq='5min',inclusive='left')
    z=np.asarray(pvlib.solarposition.get_solarposition(
        time_grid, float(latitude),float(longitude))['apparent_zenith'],float)
    daylight=z<90.
    # Grid-sample coverage of interval-start weather; the simulator applies
    # interval-END weather at 12:05,...,21:45, a distinct convention.
    in_episode=(time_grid>=date+pd.Timedelta(hours=12)) & (time_grid<date+pd.Timedelta(hours=21,minutes=45))
    return dict(sunrise_approx_utc=time_grid[np.flatnonzero(daylight)[0]].isoformat() if daylight.any() else None,
        sunset_approx_utc=(time_grid[np.flatnonzero(daylight)[-1]]+pd.Timedelta(minutes=5)).isoformat() if daylight.any() else None,
        sunrise_approx_new_york=time_grid[np.flatnonzero(daylight)[0]].tz_convert('America/New_York').isoformat() if daylight.any() else None,
        sunset_approx_new_york=(time_grid[np.flatnonzero(daylight)[-1]]+pd.Timedelta(minutes=5)).tz_convert('America/New_York').isoformat() if daylight.any() else None,
        sunlit_hours=float(np.count_nonzero(daylight)*5./60.),
        in_episode_sunlit_hours=float(np.count_nonzero(daylight & in_episode)*5./60.),
        fraction_sunlit_hours_in_episode=float(np.count_nonzero(daylight & in_episode)/np.count_nonzero(daylight)) if daylight.any() else None,
        morning_sunlit_hours_excluded=float(np.count_nonzero(daylight & (time_grid<date+pd.Timedelta(hours=12)))*5./60.),
        evening_sunlit_hours_excluded=float(np.count_nonzero(daylight & (time_grid>=date+pd.Timedelta(hours=21,minutes=45)))*5./60.))


def replay_month_and_budget(checkpoints, overlap):
    out={}
    for method,ck in checkpoints.items():
        pool=ck.get('replay') if method=='sac' else ck.get('real_pool')
        if pool is None:
            out[method]={'status':'MISSING_REAL_REPLAY'};continue
        n=int(pool['count']);capacity=int(pool['capacity'])
        info=dict(real_transitions=int(ck['env_steps']),checkpoint_epoch=int(ck['epoch']),
            real_replay_count=n,real_replay_capacity=capacity,
            fraction_generated_transitions_currently_retained=n/max(1,int(ck['env_steps'])),
            equivalent_complete_days=n/117.)
        if method=='mbpo' and 'scenario_ids' in pool:
            names=pool['scenario_names'];ids=np.asarray(pool['scenario_ids'],dtype=int)
            info['months_in_buffer']=dict(sorted(Counter(names[i][5:7] for i in ids).items()))
            info['replay_validation_contract']=pool.get('validation_contract','not_saved')
            info['validation_roles']=dict(Counter(pool.get('validation_roles',{}).values()))
            info['model_fake_pool_count']=int(ck.get('model_pool',{}).get('count',0))
            info['mbpo_dynamics_fits']=int(ck.get('dynamics',{}).get('fit_count',0))
            info['dynamics_elites']=ck.get('dynamics',{}).get('elites',[])
        out[method]=info
    out['budgets_matched']=out['sac'].get('real_transitions')==out['mbpo'].get('real_transitions')
    return out


def evaluation_critic_td(grouped,agents,algo,max_rows,seed,noon_width):
    import torch
    rng=np.random.default_rng(seed)
    results=[]
    for method,agent in agents.items():
        candidates=[]
        for key,g in grouped.items():
            rr=g[method];noon=identify_noon(rr)
            for j in range(1,H-1):
                prev=rr[j-1];cur=rr[j];nxt=rr[j+1]
                s=np.array([prev['dni_wm2']/2000.,prev['dhi_wm2']/2000.,prev['power_w']/2000.,
                       prev['tilt_deg']/90.,(prev['azimuth_deg']%360.)/180-1.,(H-j)/H],np.float32)
                sn=np.array([cur['dni_wm2']/2000.,cur['dhi_wm2']/2000.,cur['power_w']/2000.,
                       cur['tilt_deg']/90.,(cur['azimuth_deg']%360.)/180-1.,(H-j-1)/H],np.float32)
                a=np.array([cur['action_tilt'],cur['action_azimuth']],np.float32)
                candidates.append((key,cur['step'],cur['energy_kwh'],s,a,sn,
                       abs(j-noon)*5<=noon_width,group_labels(cur['datetime_utc'])))
        indices=rng.choice(len(candidates),size=min(max_rows,len(candidates)),replace=False)
        batch=[candidates[int(i)] for i in indices]
        s=torch.tensor(np.array([r[3] for r in batch]));a=torch.tensor(np.array([r[4] for r in batch]));
        sn=torch.tensor(np.array([r[5] for r in batch]));reward=torch.tensor([[r[2]] for r in batch],dtype=torch.float32)
        # Nonterminal samples only; the current code deliberately samples 1..H-2.
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.manual_seed(seed+(0 if method=='sac' else 10000))
            next_a,next_lp=agent.actor(sn,deterministic=False,with_logprob=True)
            tq1,tq2=agent.critic_target(sn,next_a)
            q1,q2=agent.critic(s,a)
            target=agent.reward_scale*reward+agent.gamma*(torch.minimum(tq1,tq2)-agent.alpha*next_lp)
            tderr=(q1-target).cpu().numpy().reshape(-1)
            qgap=(q1-q2).cpu().numpy().reshape(-1)
        for i,row in enumerate(batch):
            results.append(dict(method=method,scenario_id=row[0][0],seed=row[0][1],step=row[1],
                season=row[7],is_solar_noon_band=bool(row[6]),
                abs_td1_scaled=float(abs(tderr[i])),td1_scaled=float(tderr[i]),
                twin_q_disagreement=float(qgap[i])))
    stats=[]
    for method in agents:
        for near in (True,False):
            sub=[r for r in results if r['method']==method and r['is_solar_noon_band']==near]
            if sub:stats.append(dict(method=method,noon_band=near,n=len(sub),
                abs_td1_mean=float(np.mean([x['abs_td1_scaled'] for x in sub])),
                abs_twin_gap_mean=float(np.mean([abs(x['twin_q_disagreement']) for x in sub]))))
    return results,stats


def group_labels(t):
    h=t.month
    season=('winter' if h in (12,1,2) else 'spring' if h in (3,4,5)
            else 'summer' if h in (6,7,8) else 'fall')
    return season


def time_profile(grouped,halfwidth=30):
    rows=[]
    for key,g in sorted(grouped.items()):
        ref=g['fixed_no_motion'];noon=identify_noon(ref)
        for i in range(H):
            if i<noon-halfwidth//5: phase='before_noon'
            elif i>noon+halfwidth//5: phase='after_noon'
            else:phase='solar_noon_band'
            base=g['fixed_no_motion'][i]
            for method in ('sac','mbpo'):
                p=g[method][i]
                co=solar_cos_aoi(p['tilt_deg'],p['azimuth_deg'],p['solar_zenith_deg'],p['solar_azimuth_deg'])
                refco=solar_cos_aoi(base['tilt_deg'],base['azimuth_deg'],base['solar_zenith_deg'],base['solar_azimuth_deg'])
                rows.append(dict(scenario_id=key[0],seed=key[1],method=method,step=i+1,
                    season=group_labels(p['datetime_utc']),phase=phase,
                    offset_from_grid_solar_noon_min=(i-noon)*5,
                    utc=p['datetime_utc'].isoformat(),local=p['datetime_utc'].astimezone(TZ).isoformat(),
                    power_diff_w=p['power_w']-base['power_w'],
                    energy_diff_wh=(p['energy_kwh']-base['energy_kwh'])*1000.,
                    tilt_deg=p['tilt_deg'],azimuth_deg=p['azimuth_deg'],
                    tilt_diff_from_fixed_deg=p['tilt_deg']-base['tilt_deg'],
                    relative_azimuth_error_deg=signed_angle(p['azimuth_deg'],p['solar_azimuth_deg']),
                    signed_cos_aoi=co,cos_aoi_front=max(0.,co),fixed_cos_aoi_front=max(0.,refco),
                    dni_wm2=p['dni_wm2'],dhi_wm2=p['dhi_wm2'],
                    poa_global_wm2=p.get('poa_global_wm2'),cell_temperature_c=p.get('cell_temperature_c')))
    return rows


def summary_profile(rows):
    d=defaultdict(list)
    for r in rows:d[(r['method'],r['season'],r['phase'])].append(r)
    out=[]
    for (m,s,phase),grp in sorted(d.items()):
        # Equal-day weighting: first sum each day's five-minute kWh differences,
        # then average across days. Rows are separate seed/day units.
        per=defaultdict(list)
        for r in grp:per[(r['scenario_id'],r['seed'])].append(r)
        gains=[sum(x['energy_diff_wh'] for x in rr) for rr in per.values()]
        out.append(dict(method=m,season=s,phase=phase,n_days=len(per),
            mean_wh_agent_minus_fixed=float(np.mean(gains)),
            median_wh_agent_minus_fixed=float(np.median(gains)),
            fraction_days_agent_outperforms_fixed=float(np.mean(np.asarray(gains)>0)),
            sum_wh_agent_minus_fixed=float(np.sum(gains)),
            mean_abs_azimuth_error_deg=float(np.mean([abs(x['relative_azimuth_error_deg']) for x in grp])),
            mean_panel_tilt_deg=float(np.mean([x['tilt_deg'] for x in grp])),
            mean_cos_aoi_front=float(np.mean([x['cos_aoi_front'] for x in grp]))))
    return out


def calendar_overlap(grouped, histories):
    eval_days={key[0] for key in grouped}
    result={}
    for method,path in histories.items():
        if not path:
            result[method]={'status':'NOT_SUPPLIED','heldout_certified':False};continue
        demand(Path(path).is_file(),f'Missing {method} training history: {path}')
        with open(path,newline='',encoding='utf-8-sig') as f:r=list(csv.DictReader(f))
        ids=[x.get('scenario_id','') for x in r]
        demand(all(len(x)>=10 for x in ids),f'Missing scenario IDs in {path}')
        overlap=sorted(eval_days.intersection(set(ids)))
        counts=Counter(x[5:7] for x in ids)
        result[method]=dict(n_training_episodes=len(ids),distinct_training_days=len(set(ids)),
            n_eval_overlap=len(overlap),overlap_days=overlap[:25],
            heldout_certified=len(overlap)==0,
            episodes_by_month=dict(sorted(counts.items())))
    return result


def training_history_metrics(paths,checkpoints):
    reports={};table=[]
    for method,path in paths.items():
        rows=list(csv.DictReader(open(path,newline='',encoding='utf-8-sig')))
        epoch=int(checkpoints[method]['epoch'])
        demand(len(rows)>=epoch,f'{method} training CSV shorter than checkpoint epoch')
        history=rows[:epoch]
        try:
            demand(int(history[-1]['epoch'])==epoch and int(history[-1]['env_steps'])==int(checkpoints[method]['env_steps']),
                   f'{method}: training CSV and checkpoint budget disagree')
        except (KeyError,TypeError) as exc:raise ValueError('Missing training provenance fields') from exc
        freq=Counter(r['scenario_id'][5:7] for r in history)
        replay_count=int((checkpoints[method].get('replay') if method=='sac' else checkpoints[method].get('real_pool'))['count'])
        # Complete episodes in FIFO + one partial. This is only an inferred
        # *recent scenario distribution*; exact SAC replay row IDs are not stored.
        tail=history[-math.ceil(replay_count/H):]
        tailfreq=Counter(r['scenario_id'][5:7] for r in tail)
        reports[method]=dict(epoch=epoch,real_steps=int(checkpoints[method]['env_steps']),
            all_training_month_counts=dict(sorted(freq.items())),
            last_replay_equivalent_episode_month_counts_approx=dict(sorted(tailfreq.items())),
            replay_month_caution='SAC FIFO does not store per-row scenario IDs; tail estimate assumes uninterrupted 117-step episodes.',
            alpha_final=finite(history[-1].get('alpha')),
            critic_loss_final=finite(history[-1].get('critic_loss')))
        for i,r in enumerate(history):
            table.append(dict(method=method,epoch=i+1,scenario_id=r.get('scenario_id'),
                month=r['scenario_id'][5:7],return_kwh=finite(r.get('return_kwh')),
                alpha=finite(r.get('alpha')),critic_loss=finite(r.get('critic_loss')),
                actor_loss=finite(r.get('actor_loss')),entropy=finite(r.get('entropy')),
                model_skill_power=finite(r.get('model_skill_power_vs_persistence')),
                mean_synthetic_batch=finite(r.get('mean_model_batch'))))
    return table,reports


def exact_policy_action_replay(grouped,agents):
    """Reproduce every saved SAC/MBPO action from the preceding observed state.

    The first step cannot be reconstructed from a previous output row, so the
    physical replay audits it on the selected branch days at reset.
    """
    import torch
    summary={}
    for method,agent in agents.items():
        states=[];targets=[]
        for key,g in grouped.items():
            rr=g[method]
            for j in range(1,H):
                prev,cur=rr[j-1],rr[j]
                states.append([prev['dni_wm2']/2000.,prev['dhi_wm2']/2000.,
                    prev['power_w']/2000.,prev['tilt_deg']/90.,
                    (prev['azimuth_deg']%360.)/180.-1.,(H-j)/H])
                targets.append([cur['action_tilt'],cur['action_azimuth']])
        states=np.asarray(states,np.float32);targets=np.asarray(targets,np.float32)
        predicted=[]
        with torch.no_grad():
            for k in range(0,len(states),4096):
                a,_=agent.actor(torch.from_numpy(states[k:k+4096]),
                        deterministic=True,with_logprob=False)
                predicted.append(a.cpu().numpy())
        delta=np.max(np.abs(np.concatenate(predicted)-targets),axis=1)
        summary[method]=dict(checked_actions=int(len(delta)),
          p99_abs_error=float(np.quantile(delta,.99)),
          maximum_abs_error=float(np.max(delta)),
          fraction_over_1e_4=float(np.mean(delta>1e-4)),
          pass_exact_policy_replay=bool(np.max(delta)<=1e-4))
        demand(summary[method]['pass_exact_policy_replay'],
               f'{method}: saved checkpoint does not reproduce the evaluation action sequence')
    return summary


def oracle_dominance_check(grouped):
    checked=0
    for key,g in grouped.items():
        if 'poa_instant_ideal' not in g:continue
        for j in range(H):
            upper=g['poa_instant_ideal'][j]['power_w']
            for method in METHODS:
                power=g[method][j]['power_w']
                demand(upper+max(1e-4,1e-6*abs(power))>=power,
                   f'Ideal POA/DC not an upper reference at {key}/step{j+1}/{method}')
                checked+=1
    return {'checked_power_comparisons':checked,
        'status':'PASS' if checked else 'NOT_EXPORTED_IN_DETAILED_CSV',
        'caution':'Oracle is instantaneous/unconstrained, not a slew-limited feasible tracking policy.'}


def infer_initial_obs(rr,env,scenario_id,seed):
    env.seed(int(seed))
    obs=env.reset(scenario_id=scenario_id)
    demand(len(obs)==6,'Expected compact six-state observation')
    return obs


def follow_recorded_to_branch(env,obs,rr,step):
    for i in range(step-1):
        a=np.array([rr[i]['action_tilt'],rr[i]['action_azimuth']],np.float32)
        obs,rew,done,info=env.step(a)
        demand(abs(float(info['power_dc_w'])-float(rr[i]['power_w']))<1e-3,
               f'Physical power replay mismatch at step {i+1}; check physics/checkpoint/dataset')
        demand(abs(float(info['tilt'])-float(rr[i]['tilt_deg']))<1e-5,
               f'Physical tilt replay mismatch step {i+1}')
        demand(abs(signed_angle(info['azimuth'],rr[i]['azimuth_deg']))<1e-5,
               f'Physical azimuth replay mismatch step {i+1}')
    return obs


def stratified_day_select(grouped,count,seed):
    rng=np.random.default_rng(seed)
    keys=sorted(grouped)
    demand(count<=len(keys),'Requested more days than available evaluation scenarios')
    months=defaultdict(list)
    for key in keys:months[int(key[0][5:7])].append(key)
    ordered=[]
    for m in (1,4,7,10):
        if months[m] and len(ordered)<count:
            ordered.append(months[m][int(rng.integers(len(months[m])))])
    remaining=[key for key in keys if key not in ordered]
    if len(ordered)<count:
        idx=rng.choice(len(remaining),size=count-len(ordered),replace=False)
        ordered.extend(remaining[int(i)] for i in np.atleast_1d(idx))
    return ordered


def choose_branch_steps(rr,offsets):
    noon=identify_noon(rr)
    return sorted({min(H,max(1,noon+1+int(delta))) for delta in offsets})


def actor_output(agent,obs):
    import torch
    with torch.no_grad():
        a=agent.act(np.asarray(obs,np.float32),deterministic=True)
    return np.asarray(a,np.float32)


def critic_values(agent,obs,actions):
    import torch
    s=torch.as_tensor(np.broadcast_to(np.asarray(obs,np.float32),(len(actions),6)).copy())
    a=torch.as_tensor(np.asarray(actions,np.float32))
    with torch.no_grad():
        q1,q2=agent.critic(s,a)
    return np.minimum(q1.cpu().numpy().reshape(-1),q2.cpu().numpy().reshape(-1))


def rollout_one(cfg,agent,rr,sid,seed,branch_step,first_action,full=True):
    from SAC_workspace.src.environment.environment import SACEnvironment
    env=SACEnvironment(cfg.eval_env)
    try:
        obs=infer_initial_obs(rr,env,sid,seed)
        obs=follow_recorded_to_branch(env,obs,rr,branch_step)
        pre=obs.copy()
        cur=np.asarray(first_action,np.float32)
        obs,reward,done,info=env.step(cur)
        first_reward=float(reward)
        total=first_reward
        for i in range(branch_step,H):
            if not full:break
            if done:break
            action=actor_output(agent,obs)
            obs,r,done,info=env.step(action)
            total+=float(r)
        return dict(first_reward_kwh=first_reward,return_kwh=total,
                    pre_obs=pre.tolist(),after_power_w=float(info['power_dc_w']) if not full else None,
                    first_pose_tilt_deg=float(pre[3])*90.,
                    first_pose_azimuth_deg=(float(pre[4])+1.)*180.)
    finally:
        env.close()


def angle_and_physics_checks(cfg,grouped,days,agents,seed):
    import torch
    from SAC_workspace.src.environment.environment import SACEnvironment
    from mbpo.env import pv_tracking
    checks=[]
    for sid,eval_seed in days:
        g=grouped[(sid,eval_seed)]
        env=SACEnvironment(cfg.eval_env)
        try:
            for method in METHODS:
                rr=g[method]
                obs=infer_initial_obs(rr,env,sid,eval_seed)
                for k in range(H):
                    a=np.zeros(2,np.float32) if method=='fixed_no_motion' else np.array([rr[k]['action_tilt'],rr[k]['action_azimuth']],np.float32)
                    previous=float(env.unwrapped.tilt)
                    prev_az=float(env.unwrapped.azimuth)
                    obs,r,done,info=env.step(a)
                    pose_err=abs(float(info['tilt'])-float(rr[k]['tilt_deg']))
                    az_err=abs(signed_angle(info['azimuth'],rr[k]['azimuth_deg']))
                    power_err=abs(float(info['power_dc_w'])-rr[k]['power_w'])
                    simulator_timestamp=parse_utc(info['timestamp_utc_iso'])
                    demand(simulator_timestamp==rr[k]['datetime_utc'],
                           f'Physical UTC clock versus detailed export mismatch {sid}/{method}/step{k+1}')
                    demand(abs(float(info['solar_zenith_deg'])-float(rr[k]['solar_zenith_deg']))<1e-5,
                           f'Physical solar zenith differs from evaluation CSV {sid}/{method}/step{k+1}')
                    demand(abs(signed_angle(info['solar_azimuth_deg'],rr[k]['solar_azimuth_deg']))<1e-5,
                           f'Physical solar azimuth differs from evaluation CSV {sid}/{method}/step{k+1}')
                    demand(pose_err<1e-4 and az_err<1e-4 and power_err<1e-3,
                           f'Real physical replay mismatch {sid}/{method}/step{k+1}')
                    expected_tilt=np.clip(previous+float(a[0])*env.unwrapped.max_delta_tilt,0,90)
                    expected_az=(prev_az+float(a[1])*env.unwrapped.max_delta_azimuth)%360
                    demand(abs(expected_tilt-float(info['tilt']))<1e-5,'Tilt sign/kinematics inconsistent')
                    demand(abs(signed_angle(expected_az,info['azimuth']))<1e-5,'Azimuth sign/wrap inconsistent')
                    z=float(info['solar_zenith_deg']);solaraz=float(info['solar_azimuth_deg'])
                    geo=solar_cos_aoi(info['tilt'],info['azimuth'],z,solaraz)
                    demand(abs(max(0,geo)-float(info['cos_aoi']))<5e-6,
                           'AOI cosine geometry inconsistent with pvlib')
                    if k==0 or k==H-1 or abs(k-identify_noon(rr))<=2:
                        record=dict(scenario_id=sid,seed=eval_seed,method=method,step=k+1,
                          tilt_deg=info['tilt'],azimuth_deg=info['azimuth'],
                          solar_zenith_deg=z,solar_azimuth_deg=solaraz,
                          dni_wm2=info['dni_wm2'],dhi_wm2=info['dhi_wm2'],ghi_wm2=info['ghi_wm2'],
                          cos_aoi=max(0,geo),poa_global_wm2=info['poa_global'],
                          surface_albedo=info.get('surface_albedo'),
                          albedo_source=info.get('albedo_source'),
                          power_dc_w=info['power_dc_w'],
                          output_energy_wh=float(r)*1000.,
                          cell_temperature_c=info.get('cell_temperature_c'))
                        if info.get('surface_albedo') is not None:
                            import pvlib
                            parts=pvlib.irradiance.get_total_irradiance(
                              surface_tilt=float(info['tilt']),surface_azimuth=float(info['azimuth']),
                              solar_zenith=z,solar_azimuth=solaraz,
                              dni=float(info['dni_wm2']),dhi=float(info['dhi_wm2']),ghi=float(info['ghi_wm2']),
                              albedo=float(info['surface_albedo']),model='isotropic')
                            calc=float(parts['poa_global'])
                            demand(abs(calc-float(info['poa_global']))<max(.05,1e-5*abs(calc)),
                                   'Albedo-matched pvlib POA does not reproduce simulator')
                            record.update(poa_direct=float(parts['poa_direct']),
                                          poa_sky_diffuse=float(parts['poa_sky_diffuse']),
                                          poa_ground_diffuse=float(parts['poa_ground_diffuse']),
                                          poa_recomputed_wm2=calc)
                        checks.append(record)
        finally:env.close()
    return checks


def power_and_time_sensitivity(grouped,agents,sample_per_method,seed):
    import torch
    rng=np.random.default_rng(seed)
    all_rows=[]
    for key,g in grouped.items():
        for m in ('sac','mbpo'):
            rr=g[m]
            for j in range(1,H):
                r=rr[j-1]
                all_rows.append((m,key,j,np.asarray([
                    min(1,max(0,r['dni_wm2']/2000.)),
                    min(1,max(0,r['dhi_wm2']/2000.)),
                    min(1,max(0,r['power_w']/2000.)),
                    r['tilt_deg']/90.,(r['azimuth_deg']%360.)/180.-1.,(H-j)/H],np.float32)))
    rows=[]
    for m in ('sac','mbpo'):
        pool=[r for r in all_rows if r[0]==m]
        ix=rng.choice(len(pool),size=min(sample_per_method,len(pool)),replace=False)
        agent=agents[m]
        for j in ix:
            _,(sid,eval_seed),step,s=pool[int(j)]
            x=torch.tensor(s[None],requires_grad=True,dtype=torch.float32)
            out,_=agent.actor(x,deterministic=True,with_logprob=False)
            grads=[]
            for k in range(2):
                grads.append(torch.autograd.grad(out[0,k],x,retain_graph=k==0)[0][0].detach().numpy().copy())
            original=out.detach().numpy()[0]
            for channel,label,dist in ((2,'power',.01),(5,'remaining_time',1./H),(4,'azimuth',.05)):
                change=s.copy()
                if channel==4:change[channel]=np.clip(change[channel]+dist,-1,1)
                else:change[channel]=np.clip(change[channel]+dist,0,1)
                compare=actor_output(agent,change)
                rows.append(dict(method=m,scenario_id=sid,seed=eval_seed,step=step,
                    season=group_labels(datetime.fromisoformat(sid)),power_norm=float(s[2]),
                    tilt_action=float(original[0]),azimuth_action=float(original[1]),
                    perturbed_feature=label,
                    perturbation=float(change[channel]-s[channel]),
                    delta_tilt_action=float(compare[0]-original[0]),
                    delta_azimuth_action=float(compare[1]-original[1]),
                    derivative_dtilt_dpower=float(grads[0][2]),
                    derivative_dazimuth_dpower=float(grads[1][2]),
                    derivative_dtilt_dtau=float(grads[0][5]),
                    derivative_dazimuth_dtau=float(grads[1][5])))
    return rows


def azimuth_seam(agents,grouped,seed,max_states=100):
    rng=np.random.default_rng(seed)
    all_states=[]
    for key,g in grouped.items():
        rr=g['fixed_no_motion']
        for r in rr:
            all_states.append([min(1,r['dni_wm2']/2000),min(1,r['dhi_wm2']/2000),
                               min(1,r['power_w']/2000),r['tilt_deg']/90.,0.,(H-r['step'])/H])
    ix=rng.choice(len(all_states),size=min(max_states,len(all_states)),replace=False)
    records=[]
    for m,agent in agents.items():
        for j in ix:
            s=np.array(all_states[int(j)],np.float32)
            s1=s.copy();s2=s.copy();s1[4]=(359.9/180)-1;s2[4]=(.1/180)-1
            a1=actor_output(agent,s1);a2=actor_output(agent,s2)
            records.append(dict(method=m,azimuth_1_deg=359.9,azimuth_2_deg=.1,
              physical_separation_deg=.2,
              actor_action_jump_linf=float(np.max(np.abs(a1-a2))),
              actor_tilt_jump=float(a1[0]-a2[0]),
              actor_azimuth_jump=float(a1[1]-a2[1])))
    return records


def dynamic_predict(state,obs,actions):
    import torch
    xa=np.concatenate([obs,actions],axis=1).astype(np.float32)
    xmean=np.asarray(state['xmean'],np.float32);xstd=np.asarray(state['xstd'],np.float32)
    ymean=np.asarray(state['ymean'],np.float32);ystd=np.asarray(state['ystd'],np.float32)
    from MBPO_workspace.src.dynamics import GaussianDeltaNet
    models=[];stds=[]
    with torch.no_grad():
        x=torch.from_numpy((xa-xmean)/xstd)
        for ix in state['elites']:
            net=GaussianDeltaNet(int(state['hidden_dim']))
            net.load_state_dict(state['nets'][int(ix)]);net.eval()
            mu,logstd=net(x)
            models.append(mu.cpu().numpy()*ystd+ymean)
            stds.append(np.exp(logstd.cpu().numpy())*ystd)
    models=np.asarray(models);stds=np.asarray(stds)
    mean=models.mean(axis=0)
    var=np.mean(stds**2,axis=0)+np.var(models,axis=0)
    return mean,np.sqrt(np.maximum(var,1e-12)),models.std(axis=0)


def dynamics_evaluation(grouped,mbpo_ck,train_overlap,max_rows,seed,noon_width=30):
    # Held-out from TRAINING only if training CSV supports exact no-overlap.
    # Must NOT claim pristine test status after using these days for investigation.
    dyn=mbpo_ck.get('dynamics')
    demand(dyn is not None and dyn.get('fitted'),'MBPO dynamics are missing/unfitted')
    from mbpo.env import pv_tracking
    observations=[];actions=[];targets=[];meta=[]
    rng=np.random.default_rng(seed)
    for key,g in grouped.items():
        rr=g['mbpo'];noon=identify_noon(rr)
        for k in range(H-1):
            r,t=rr[k],rr[k+1]
            s=np.array([np.clip(r['dni_wm2']/pv_tracking.IRRADIANCE_NORM,0,1),
                        np.clip(r['dhi_wm2']/pv_tracking.IRRADIANCE_NORM,0,1),
                        np.clip(r['power_w']/pv_tracking.POWER_NORM,0,1),
                        r['tilt_deg']/90,(r['azimuth_deg']%360)/180-1,(H-(k+1))/H],np.float32)
            nxt=np.array([np.clip(t['dni_wm2']/pv_tracking.IRRADIANCE_NORM,0,1),
                          np.clip(t['dhi_wm2']/pv_tracking.IRRADIANCE_NORM,0,1),
                          np.clip(t['power_w']/pv_tracking.POWER_NORM,0,1)],np.float32)
            a=np.array([t['action_tilt'],t['action_azimuth']],np.float32)
            observations.append(s);actions.append(a);targets.append(nxt-s[:3]);
            meta.append(dict(scenario_id=key[0],seed=key[1],step=k+2,
                  season=group_labels(t['datetime_utc']),
                  window='noon' if abs(k+1-noon)*5<=noon_width else 'off_noon',
                  daylight=bool(t['solar_zenith_deg']<90 and t['dni_wm2']>0)))
    idx=rng.choice(len(observations),size=min(max_rows,len(observations)),replace=False)
    obs=np.array([observations[i] for i in idx]);act=np.array([actions[i] for i in idx])
    true=np.array([targets[i] for i in idx]);mm=[meta[i] for i in idx]
    pred,std,disagree=dynamic_predict(dyn,obs,act)
    rows=[]
    for j,meta_row in enumerate(mm):
        r=dict(meta_row)
        for k,label in enumerate(('dni','dhi','power')):
            r.update({f'{label}_actual_delta_norm':float(true[j,k]),
                      f'{label}_pred_delta_norm':float(pred[j,k]),
                      f'{label}_error_norm':float(pred[j,k]-true[j,k]),
                      f'{label}_pred_std_norm':float(std[j,k]),
                      f'{label}_ensemble_disagreement_norm':float(disagree[j,k])})
        rows.append(r)
    summary=[]
    for phase in ('noon','off_noon','all'):
        for season in ('winter','spring','summer','fall','all'):
            ii=[j for j,r in enumerate(mm) if (phase=='all' or r['window']==phase) and
                (season=='all' or r['season']==season)]
            if len(ii)<10:continue
            for k,label in enumerate(('dni','dhi','power')):
                error=pred[ii,k]-true[ii,k]
                pers=true[ii,k]
                rmse=float(np.sqrt(np.mean(error**2)))
                p_rmse=float(np.sqrt(np.mean(pers**2)))
                z=np.abs(error)/np.maximum(std[ii,k],1e-8)
                summary.append(dict(phase=phase,season=season,channel=label,n=len(ii),
                    rmse_norm=rmse,persistence_rmse_norm=p_rmse,
                    skill_vs_persistence=None if p_rmse<1e-8 else 1-rmse/p_rmse,
                    signed_bias_norm=float(np.mean(error)),
                    empirical_1sigma_coverage=float(np.mean(z<=1)),
                    empirical_2sigma_coverage=float(np.mean(z<=2)),
                    median_ensemble_std_norm=float(np.median(std[ii,k])),
                    median_ensemble_disagreement_norm=float(np.median(disagree[ii,k]))))
    provenance={'heldout_from_training_certified':bool(train_overlap.get('mbpo',{}).get('heldout_certified')),
      'validation_warning':'These days are independent from training only if supplied MBPO training history covers the full checkpoint. They remain development diagnostics, not untouched final test days.',
      'source':'Real exported evaluation trajectory transitions; model never fit during this diagnostic.'}
    return rows,summary,provenance


def action_grid(counterfactual,agents,grouped,days,offsets,grid,full):
    # Physical experiments, frozen deterministic policy after ONE alternative action.
    # Same initial condition, same local measured weather, no learned-synthetic rollout.
    all_rows=[]
    from SAC_workspace.src.environment.environment import SACEnvironment
    cfg=counterfactual
    for sid,seed in days:
        g=grouped[(sid,seed)]
        for method in ('sac','mbpo'):
            agent=agents[method];rr=g[method]
            for step in choose_branch_steps(rr,offsets):
                pre_env=SACEnvironment(cfg.eval_env)
                try:
                    obs=infer_initial_obs(rr,pre_env,sid,seed)
                    obs=follow_recorded_to_branch(pre_env,obs,rr,step)
                finally:pre_env.close()
                chosen=actor_output(agent,obs)
                recorded=np.array([rr[step-1]['action_tilt'],rr[step-1]['action_azimuth']],np.float32)
                demand(np.max(np.abs(recorded-chosen))<1e-4,
                    f'Checkpoint action mismatch: {(sid,seed,method,step)}')
                actions=[chosen.copy()]+[np.array([a,b],np.float32) for a in grid for b in grid]
                q=critic_values(agent,obs,actions)
                for index,a in enumerate(actions):
                    physical=rollout_one(cfg,agent,rr,sid,seed,step,a,full=full)
                    all_rows.append(dict(scenario_id=sid,seed=seed,method=method,step=step,
                      utc=rr[step-1]['datetime_utc'].isoformat(),
                      local=rr[step-1]['datetime_utc'].astimezone(TZ).isoformat(),
                      season=group_labels(rr[step-1]['datetime_utc']),
                      sample_kind='actor' if index==0 else 'grid',
                      action_tilt=float(a[0]),action_azimuth=float(a[1]),
                      q_clipped_soft_scaled=float(q[index]),
                      first_reward_wh=float(physical['first_reward_kwh'])*1000.,
                      continuation_reward_wh=float(physical['return_kwh'])*1000.,
                      remaining_steps=H-step+1,
                      comparator='one_action_then_frozen_deterministic_policy' if full else 'immediate_only'))
    return all_rows


def grid_summaries(grid_rows):
    d=defaultdict(list)
    for r in grid_rows:d[(r['scenario_id'],r['seed'],r['method'],r['step'])].append(r)
    out=[]
    for key,rr in d.items():
        chosen=next(r for r in rr if r['sample_kind']=='actor')
        candidates=[r for r in rr if r['sample_kind']=='grid']
        best=max(candidates,key=lambda x:x['continuation_reward_wh'])
        bestq=max(candidates,key=lambda x:x['q_clipped_soft_scaled'])
        out.append(dict(scenario_id=key[0],seed=key[1],method=key[2],step=key[3],
            chosen_action_tilt=chosen['action_tilt'],chosen_action_azimuth=chosen['action_azimuth'],
            chosen_first_reward_wh=chosen['first_reward_wh'],
            chosen_remaining_return_wh=chosen['continuation_reward_wh'],
            best_grid_action_tilt=best['action_tilt'],best_grid_action_azimuth=best['action_azimuth'],
            best_grid_remaining_return_wh=best['continuation_reward_wh'],
            regret_to_best_tested_grid_wh=max(0.,best['continuation_reward_wh']-chosen['continuation_reward_wh']),
            best_grid_q_action_tilt=bestq['action_tilt'],best_grid_q_action_azimuth=bestq['action_azimuth'],
            critic_prefers_actor_to_best_physical_grid=bool(chosen['q_clipped_soft_scaled']>=best['q_clipped_soft_scaled']),
            caution='Grid candidate actions are not guaranteed globally optimal; critic soft Q vs deterministic returns are different objectives.'))
    return out


def render_diagnostic_plots(out,profile_rows,summary_rows,regrets):
    """Draw explanatory plots from diagnostic outputs; never source images/data."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return ['Matplotlib missing: graphical outputs skipped']
    messages=[]
    fig,ax=plt.subplots(figsize=(10,5))
    for method in ('sac','mbpo'):
        rr=[r for r in profile_rows if r['method']==method and abs(r['offset_from_grid_solar_noon_min'])<=120]
        offsets=sorted({r['offset_from_grid_solar_noon_min'] for r in rr})
        values=[np.mean([r['power_diff_w'] for r in rr if r['offset_from_grid_solar_noon_min']==x]) for x in offsets]
        ax.plot(offsets,values,marker='.',label=f'{method.upper()} minus fixed')
    ax.axhline(0,linewidth=1,linestyle='--',color='gray');ax.axvline(0,linewidth=1,linestyle=':',color='gray')
    ax.set(xlabel='Minutes from sampled solar noon',ylabel='Mean power difference (W)',
           title='Paired time-aligned solar-noon power comparison')
    ax.legend();fig.tight_layout();fig.savefig(out/'noon_power_comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    seasons=['winter','spring','summer','fall']
    for method in ('sac','mbpo'):
        v=[next((r['mean_wh_agent_minus_fixed'] for r in summary_rows if r['method']==method and r['season']==s and r['phase']=='solar_noon_band'),np.nan) for s in seasons]
        x=np.arange(4)+(0 if method=='sac' else .34)
        ax.bar(x,v,width=.34,label=method.upper())
    ax.set_xticks(np.arange(4)+.17,seasons)
    ax.axhline(0,linewidth=1,color='gray')
    ax.set(ylabel='Mean paired daylight-band energy difference (Wh / day)',
           title='Solar-noon band versus fixed, by season')
    ax.legend();fig.tight_layout();fig.savefig(out/'seasonal_noon_difference.png',dpi=150);plt.close(fig)
    if regrets:
        fig,ax=plt.subplots(figsize=(10,5))
        for method in ('sac','mbpo'):
            r=[x['regret_to_best_tested_grid_wh'] for x in regrets if x['method']==method]
            ax.hist(r,bins=16,alpha=.5,label=method.upper())
        ax.set(xlabel='Best tested grid improvement over actor (Wh)',ylabel='Branch count',
               title='Sampled finite-action policy gap (not global optimality)')
        ax.legend();fig.tight_layout();fig.savefig(out/'action_grid_regret.png',dpi=150);plt.close(fig)
    return messages


def run(args):
    import torch
    from SAC_workspace.src.config.config import load_config as sac_config
    from MBPO_workspace.src.config import load_config as mbpo_config
    from SAC_workspace.src.agents.sac_agent import SACAgent
    torch.set_num_threads(1)
    paths=[args.sac_checkpoint,args.mbpo_checkpoint,args.eval_dir]
    for p in paths:demand(Path(p).exists(),f'Missing required input: {p}')
    folder=Path(args.eval_dir)
    detail=folder/'trajectories_detailed.csv'
    grouped=materialize_trajectories(detail)
    oracle_check=oracle_dominance_check(grouped)
    clock=clock_diagnosis(grouped)
    profiles=time_profile(grouped,halfwidth=args.noon_width_min)
    profsum=summary_profile(profiles)
    overlap=calendar_overlap(grouped,{'sac':args.sac_training_csv,'mbpo':args.mbpo_training_csv})
    demand(all(x.get('heldout_certified') for x in overlap.values()),
      'Training-day overlap not certified: supply both full training histories and use genuinely held-out evaluation days.')
    # Loading torch pickle checkpoint is permitted only for user's own trusted files.
    sc=torch.load(args.sac_checkpoint,map_location='cpu',weights_only=False)
    mc=torch.load(args.mbpo_checkpoint,map_location='cpu',weights_only=False)
    p1=sc.get('resolved_mdp',{});p2=mc.get('resolved_mdp',{})
    for key in ('power_model','gamma_pdc','wind_reference_height_m','sapm_mount',
                'wind_powerlaw_exponent','power_source_sha256','horizon','start_time','freq',
                'nsrdb_manifest_sha256','nsrdb_year_file_sha256','albedo_model'):
        if key in p1 or key in p2:
            demand(p1.get(key)==p2.get(key),f'Checkpoint physics mismatch on {key}')
    demand(p1.get('power_model')=='pvwatts_sapm', 'Stage 5 requires matched PVWatts-SAPM checkpoints')
    kw={k:p1[k] for k in ('power_model','gamma_pdc','wind_reference_height_m','sapm_mount','wind_powerlaw_exponent')}
    sac_cfg=sac_config(require_data=True,**kw)
    mbpo_cfg=mbpo_config(require_data=True,**kw)
    demand(sac_cfg.to_dict()==mbpo_cfg.sac.to_dict(), 'SAC/MBPO MDP config mismatch')
    from SAC_workspace.src.environment.environment import SACEnvironment
    verify=SACEnvironment(sac_cfg.eval_env)
    try:
        for key in ('power_source_sha256','nsrdb_manifest_sha256','nsrdb_year_file_sha256'):
            saved=p1.get(key)
            if saved is not None:
                demand(saved==verify.resolved_mdp.get(key),
                   f'Current PV data/physics differs from saved checkpoint on {key}; do not bypass this.')
    finally:verify.close()
    modelspec=mbpo_cfg.model
    model_configuration=dict(real_ratio=float(modelspec.real_ratio),
        fake_ratio=1.-float(modelspec.real_ratio),ensemble=int(modelspec.num_models),
        elite_count=int(modelspec.num_elites),rollout_length=int(modelspec.rollout_length),
        uncertainty_max_std=modelspec.uncertainty_max_std,
        require_model_skill=bool(modelspec.require_model_skill),
        last_model_validation_metrics=mc.get('dynamics',{}).get('last_metrics'),
        model_checkpoints_are_not_trained_in_diagnostic=True)
    agents={}
    for method,checkpoint in (('sac',sc),('mbpo',mc)):
        algo=sac_cfg.algo
        a=SACAgent(hidden_sizes=algo['hidden_sizes'],lr=algo['lr'],gamma=algo['gamma'],
            tau=algo['tau'],reward_scale=algo['reward_scale'],alpha_init=algo['alpha_init'],
            target_entropy=algo['target_entropy'],min_alpha=algo['min_alpha'],device='cpu')
        a.load_state_dict(checkpoint['agent']);a.actor.eval();a.critic.eval();agents[method]=a
    action_replay=exact_policy_action_replay(grouped,agents)
    days=stratified_day_select(grouped,args.days,args.seed)
    verify=SACEnvironment(sac_cfg.eval_env)
    try:
        physical=verify.unwrapped
        lat=getattr(physical.location,'latitude',None);lon=getattr(physical.location,'longitude',None)
    finally:verify.close()
    demand(lat is not None and lon is not None,'Missing physical site coordinates for daylight diagnosis')
    daylight=[]
    for sid,eval_seed in sorted(days):
        x=dict(scenario_id=sid,seed=eval_seed,latitude=float(lat),longitude=float(lon))
        x.update(daylight_duration_by_day(sid,lat,lon))
        daylight.append(x)
    replay_summary=replay_month_and_budget({'sac':sc,'mbpo':mc},overlap)
    trainrows,trainmetrics=training_history_metrics(
        {'sac':args.sac_training_csv,'mbpo':args.mbpo_training_csv},
        {'sac':sc,'mbpo':mc})
    tdrows,tdsummary=evaluation_critic_td(grouped,agents,sac_cfg.algo,
                          args.model_rows,args.seed+5,args.noon_width_min)
    phase_check=angle_and_physics_checks(sac_cfg,grouped,days,agents,args.seed)
    sens=power_and_time_sensitivity(grouped,agents,args.sensitivity_states,args.seed+8)
    seams=azimuth_seam(agents,grouped,args.seed+9,max_states=args.seam_states)
    dynrows,dynsum,dynprov=dynamics_evaluation(grouped,mc,overlap,args.model_rows,args.seed+10,args.noon_width_min)
    grid=[];regrets=[]
    if args.grid:
        grid=action_grid(sac_cfg,agents,grouped,days,args.offsets,args.grid,full=not args.immediate_only)
        regrets=grid_summaries(grid)
    out=Path(args.outdir).resolve()
    eval_source=Path(args.eval_dir).resolve()
    demand(out!=ROOT and out!=eval_source and not any(out==Path(v).resolve() for v in paths),
           'Output folder must be separate from evaluation/input files and repository root')
    demand(not (out in eval_source.parents or out in ROOT.parents),
           'Output folder cannot be an ancestor of the evaluation directory or repository root')
    out.mkdir(parents=True,exist_ok=True)
    save_csv(out/'clock_and_noon.csv',clock)
    save_csv(out/'training_history_and_seasons.csv',trainrows)
    save_csv(out/'daylight_duration_utc_local.csv',daylight)
    save_csv(out/'heldout_critic_td_rows.csv',tdrows)
    save_csv(out/'heldout_critic_td_summary.csv',tdsummary)
    save_csv(out/'intraday_power_comparison.csv',profiles)
    save_csv(out/'intraday_summary.csv',profsum)
    save_csv(out/'physical_geometry_replay.csv',phase_check)
    save_csv(out/'power_tau_sensitivity.csv',sens)
    save_csv(out/'azimuth_seam.csv',seams)
    save_csv(out/'mbpo_dynamics_heldout_rows.csv',dynrows)
    save_csv(out/'mbpo_dynamics_heldout_summary.csv',dynsum)
    save_csv(out/'action_grid_branches.csv',grid)
    save_csv(out/'action_grid_regrets.csv',regrets)
    report={'schema':'pv_stage5_readonly_v1','read_only':True,
       'input_sha256':{'sac_checkpoint':sha(args.sac_checkpoint),
                      'mbpo_checkpoint':sha(args.mbpo_checkpoint),
                      'trajectories_detailed_csv':sha(detail),
                      'sac_training_csv':sha(args.sac_training_csv),
                      'mbpo_training_csv':sha(args.mbpo_training_csv)},
       'analysis_params':dict(seed=args.seed,days=args.days,grid=args.grid,
                              branch_offsets=args.offsets,noon_width_min=args.noon_width_min,
                              continuation_mode='immediate_only' if args.immediate_only else 'full_deterministic'),
       'n_evaluation_scenarios':len(grouped),
       'n_executed_physical_branch_days':len(days),
       'selected_day_ids':[dict(scenario_id=s,seed=seed) for s,seed in days],
       'checkpoint_budget':{'sac_steps':sc.get('env_steps'), 'mbpo_steps':mc.get('env_steps'),
                            'budget_matched':sc.get('env_steps')==mc.get('env_steps')},
       'data_provenance':overlap,
       'evaluation_action_replay':action_replay,
       'instantaneous_ideal_upper_reference':oracle_check,
       'replay_coverage_and_budget':replay_summary,
       'mbpo_model_configuration_and_saved_skill':model_configuration,
       'trained_agent_entropy_temperature':{
           m:dict(alpha=float(a.alpha),reward_scale=float(a.reward_scale),
                  alpha_over_reward_scale=float(a.alpha/a.reward_scale),
                  alpha_floor=float(a.min_alpha),target_entropy=float(a.target_entropy))
           for m,a in agents.items()},
       'initial_pose_shift':dict(train_randomize_initial_orientation=bool(sac_cfg.train_env.get('randomize_initial_orientation')),
                  eval_randomize_initial_orientation=bool(sac_cfg.eval_env.get('randomize_initial_orientation')),
                  warning='A fixed evaluation pose is a distribution shift if training initial orientation was randomized.'),
       'training_coverage_and_optimizer_history':trainmetrics,
       'selected_daylight_coverage':daylight,
       'evaluation_critic_td_summary':tdsummary,
       'interpretation':{'noon':'minimum solar zenith among sampled 5-minute UTC points, not legal clock noon',
                         'time_zone':'America/New_York IANA standard/DST applied to UTC measurements',
                         'window':'117 rewards 12:05–21:45 UTC; local window shifts with DST',
                         'power':'same physical source used; gross DC energy if movement_penalty=0',
                         'mbpo_validation':dynprov,
                         'causal_limit':'Sensitivity, branches, and model errors identify mechanisms, not definitive causal proof without retraining ablations.',
                         'selection':'day selection fixed by seed and calendar months before outcome inspection',
                         'azimuth_seam':'Counterfactual 359.9 vs 0.1 normalized azimuth states are off-policy stress tests; they may be physically inconsistent with the power feature. Large action jumps are not a physical angle bug by themselves.',
                         'ensemble_uncertainty':'Gaussian predictive variance and ensemble disagreement are separately exported; calibration is an empirical check, not a statistical guarantee.',
                         'profile':'unweighted descriptive day means, not population-weighted or multi-training-seed inference'},
       'intraday_comparison':profsum,
       'grid_regret_descriptive':{
         m:{'n':sum(r['method']==m for r in regrets),
            'fraction_grid_better_than_actor':float(np.mean([r['regret_to_best_tested_grid_wh']>1e-6 for r in regrets if r['method']==m])) if any(r['method']==m for r in regrets) else None,
            'mean_grid_regret_wh':float(np.mean([r['regret_to_best_tested_grid_wh'] for r in regrets if r['method']==m])) if any(r['method']==m for r in regrets) else None}
         for m in ('sac','mbpo')},
       'mbpo_dynamics_summary':dynsum}
    report['plot_warnings']=render_diagnostic_plots(out,profiles,profsum,regrets)
    (out/'stage5_report.json').write_text(json.dumps(safely_json(report),indent=2,allow_nan=False),encoding='utf-8')
    (out/'README_RESULTS.md').write_text('# Stage 5 read-only diagnostics\n\n'
      'This bundle contains data-derived reports from the frozen SAC/MBPO policies and local NSRDB environment.\n'
      'No training, optimizer step, checkpoint change, new weather generation or policy adjustment was done.\n'
      'Grid regret is relative to finite candidate actions only, not the global optimum.\n'
      'Solar noon is 5-minute-sampled minimum zenith. Sunrise/sunset outside the window cannot be inferred from a trajectory alone.\n'
      'Use findings for hypotheses; final algorithm comparison requires budget-matched, multi-seed, untouched evaluation.\n',encoding='utf-8')
    archive=out/'SEND_ME_STAGE5_MIDDAY_DIAGNOSTIC.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p!=archive:z.write(p,p.name)
    print('STAGE 5 READ-ONLY DIAGNOSIS FINISHED')
    print('Budget matched:',report['checkpoint_budget']['budget_matched'])
    print('Read-only ZIP for review:',archive)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--sac-checkpoint',required=True)
    p.add_argument('--mbpo-checkpoint',required=True)
    p.add_argument('--sac-training-csv',required=True)
    p.add_argument('--mbpo-training-csv',required=True)
    p.add_argument('--eval-dir',required=True)
    p.add_argument('--outdir',default='MBPO_workspace/evaluation/stage5_midday_diagnostic')
    p.add_argument('--days',type=int,default=8,help='Calendar-stratified scenario/seed pairs for physical replay')
    p.add_argument('--offsets',type=int,nargs='+',default=[-2,0,2],help='5-min step offsets from 5-min-grid solar noon')
    p.add_argument('--grid',type=float,nargs='+',default=[-1.,-.5,0.,.5,1.],help='Normalized tilt AND azimuth action candidates')
    p.add_argument('--immediate-only',action='store_true',help='Faster, but cannot test long-horizon optimality')
    p.add_argument('--sensitivity-states',type=int,default=300)
    p.add_argument('--seam-states',type=int,default=100)
    p.add_argument('--model-rows',type=int,default=4000)
    p.add_argument('--noon-width-min',type=int,default=30)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    demand(args.days>=1 and args.sensitivity_states>=1 and args.model_rows>=10,'Invalid sample counts')
    demand(args.noon_width_min>=5 and args.noon_width_min%5==0,'Noon window must be a positive multiple of 5 min')
    demand(all(-1<=a<=1 for a in args.grid),'Normalized action-grid values must be in [-1,1]')
    demand(len(set(args.grid))==len(args.grid),'Duplicate grid action values')
    demand(max(abs(x) for x in args.offsets)<=20,'Offsets are in 5-minute steps, not hours')
    run(args)


if __name__=='__main__':main()
