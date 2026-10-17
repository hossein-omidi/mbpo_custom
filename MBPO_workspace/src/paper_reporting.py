"""Ray/TF-free scientific figures adapted from the repository's historic plots.

Sources of plot ideas and verified semantics:
  scripts/plot_paper_eval.py: energy decomposition, gain distributions,
       orientation occupancy, power-gap heatmap, scenario ratios, movement yield.
  scripts/diagnose_tracking.py: signed tracking errors, action time-series,
       action paths in 3D and orientation versus physical solar target.
  scripts/plot_rollout_trajectory.py: UTC post-action time-axis convention.

Consumes the existing MBPO_workpace evaluation 'paths'. Exports sufficient
transition metadata for *offline* plot regeneration without reopening pvlib,
replaying environments, restoring a policy or loading legacy Ray/TensorFlow.
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from MBPO_workspace.src.evaluation import METHODS, FEASIBLE

LABELS = {
    'mbpo': 'MBPO (remastered)', 'sac': 'SAC (model-free)',
    'fixed_no_motion': 'Fixed mount', 'dual_axis_slew_limited': 'Dual axis (slew-limited)',
    'dual_axis_instant_ideal': 'Sun pointing (ideal; gross only)',
    'poa_instant_ideal': 'POA optimum (ideal; gross only)',
}
IDEALS = set(METHODS) - set(FEASIBLE)
FIELDS = (
    'scenario_id','seed','method','step','timestamp_utc_iso','clock_hour_utc',
    'season','weather_condition','solar_zenith_deg','solar_azimuth_deg',
    'solar_altitude_deg','dni_wm2','dhi_wm2','ghi_wm2',
    'power_w','energy_kwh','reward_kwh','movement_cost_kwh',
    'cumulative_gross_kwh','cumulative_net_kwh',
    'pre_tilt_deg','pre_azimuth_deg','tilt_deg','azimuth_deg',
    'action_tilt','action_azimuth','actual_delta_tilt_deg','actual_delta_azimuth_deg',
    'tilt_sun_target_deg','tilt_tracking_error_deg','azimuth_tracking_error_deg',
)
FLOATS = set(FIELDS)-{'scenario_id','seed','method','step','timestamp_utc_iso','season','weather_condition'}


def _hour(ts):
    dt=datetime.fromisoformat(str(ts).replace('Z','+00:00'))
    if dt.tzinfo is None or dt.utcoffset().total_seconds()!=0:
        raise ValueError('Expected an explicit UTC-aware post-step timestamp')
    return dt.hour+dt.minute/60+dt.second/3600


def _shortest(a):
    return (float(a)+180.)%360.-180.


def _fixed_float(value):
    if value is None or value=='': return float('nan')
    return float(value)


def expand_paths(paths):
    """Validate and flatten *already generated* trajectories into aligned rows.

    Every real policy is validated against kinematic action/energy identities.
    No synthetic/ideal trajectory is ever labeled as a feasible net controller.
    """
    paths=list(paths)
    if not paths: raise ValueError('No evaluation paths')
    groups=defaultdict(dict)
    rows=[]
    for path in paths:
        m=path['method']; sid=str(path['scenario_id']); seed=int(path['seed'])
        if m not in METHODS: raise ValueError('Unknown method '+m)
        if m in groups[(sid,seed)]:raise ValueError(f'Duplicate method for {sid}/{seed}: {m}')
        groups[(sid,seed)][m]=path
        n=len(path['infos'])
        if n != 117: raise ValueError(f'Expected 117 steps, got {n} for {m}/{sid}')
        for k in ('power','energy','reward','cost','tilt','azimuth','action_tilt',
                  'action_azimuth','delta_tilt_deg','delta_azimuth_deg'):
            if len(path[k])!=n:raise ValueError(f'Inconsistent {k} length in {m}/{sid}')
        is_ideal=m in IDEALS
        if is_ideal:
            if path['net_energy_kwh'] is not None or path['movement_cost_kwh'] is not None:
                raise ValueError('Ideal counterfactual cannot claim feasible net energy')
        else:
            if not np.isclose(float(path['net_energy_kwh']), sum(path['reward']), atol=1e-7):
                raise ValueError('Wrong net return')
            if not np.isclose(float(path['gross_energy_kwh'])-float(path['movement_cost_kwh']),
                              float(path['net_energy_kwh']), atol=1e-7):
                raise ValueError('Net must equal gross less actuator costs')
        pose=path['initial_pose']
        pre_tilt,pre_az=float(pose[0]),float(pose[1])
        gross=net=0.
        times=[]
        for i in range(n):
            info=path['infos'][i]
            ts=str(info['timestamp_utc_iso'])
            hour=_hour(ts);times.append(datetime.fromisoformat(ts.replace('Z','+00:00')))
            tilt=float(path['tilt'][i]);az=float(path['azimuth'][i])
            pw=float(path['power'][i]);en=float(path['energy'][i]);rw=float(path['reward'][i]);cost=float(path['cost'][i])
            if not all(np.isfinite(x) for x in (tilt,az,pw,en,rw,cost)):
                raise ValueError('Nonfinite physical measurement')
            if tilt < -1e-5 or tilt > 90.+1e-5 or az < -1e-5 or az >=360.+1e-5:
                raise ValueError('Out-of-bound panel pose')
            if not np.isclose(en,pw*(5./60.)/1000.,atol=1e-6,rtol=1e-5):
                raise ValueError('Energy/power/5-minute accounting mismatch')
            a_t=_fixed_float(path['action_tilt'][i]);a_a=_fixed_float(path['action_azimuth'][i])
            d_t=_fixed_float(path['delta_tilt_deg'][i]);d_a=_fixed_float(path['delta_azimuth_deg'][i])
            if not is_ideal:
                if not np.isfinite([a_t,a_a,d_t,d_a]).all() or max(abs(a_t),abs(a_a))>1.00001:
                    raise ValueError('Invalid executed normalized action')
                if not np.isclose(rw,en-cost,atol=1e-7):raise ValueError('Step reward identity failure')
                if not np.isclose(tilt, np.clip(pre_tilt+d_t,0,90),atol=1e-4):
                    raise ValueError('Tilt physical increment mismatch')
                if abs(_shortest(az-pre_az)-d_a)>1e-4:
                    raise ValueError('Azimuth physical increment mismatch')
            zen=_fixed_float(info.get('solar_zenith_deg'))
            sunaz=_fixed_float(info.get('solar_azimuth_deg'))
            target=float(np.clip(zen,0.,90.)) if np.isfinite(zen) else float('nan')
            row=dict(scenario_id=sid,seed=seed,method=m,step=i+1,timestamp_utc_iso=ts,
                     clock_hour_utc=hour,season=info.get('season') or '',
                     weather_condition=info.get('weather_condition') or '',
                     solar_zenith_deg=zen,solar_azimuth_deg=sunaz,
                     solar_altitude_deg=_fixed_float(info.get('solar_altitude_deg')) if info.get('solar_altitude_deg') is not None else 90.-zen,
                     dni_wm2=_fixed_float(info.get('dni_wm2')),
                     dhi_wm2=_fixed_float(info.get('dhi_wm2')),
                     ghi_wm2=_fixed_float(info.get('ghi_wm2')),
                     power_w=pw,energy_kwh=en,reward_kwh=rw,movement_cost_kwh=cost,
                     cumulative_gross_kwh=gross+en,cumulative_net_kwh=net+rw,
                     pre_tilt_deg=pre_tilt,pre_azimuth_deg=pre_az,
                     tilt_deg=tilt,azimuth_deg=az,action_tilt=a_t,action_azimuth=a_a,
                     actual_delta_tilt_deg=d_t,actual_delta_azimuth_deg=d_a,
                     tilt_sun_target_deg=target,tilt_tracking_error_deg=tilt-target,
                     azimuth_tracking_error_deg=_shortest(az-sunaz) if np.isfinite(sunaz) else float('nan'))
            rows.append(row)
            gross+=en;net+=rw;pre_tilt,pre_az=tilt,az
        for before,after in zip(times,times[1:]):
            if abs((after-before).total_seconds()-300.)>1e-6:
                raise ValueError('Nonchronological/non-five-minute NSRDB timestamps')
        if times[0].hour!=12 or times[0].minute!=5 or times[-1].hour!=21 or times[-1].minute!=45:
            raise ValueError('Episode time grid is not the 12:05–21:45 UTC post-action grid')
        if not np.isclose(gross,path['gross_energy_kwh'],atol=1e-7):
            raise ValueError('Gross energy aggregation mismatch')
    expected=set(METHODS) if any('poa_instant_ideal' in g for g in groups.values()) else set(METHODS)-{'poa_instant_ideal'}
    for key,g in groups.items():
        if set(g)!=expected: raise ValueError(f'Missing paired controllers: {key}')
        ref=g['fixed_no_motion'];times=[i['timestamp_utc_iso'] for i in ref['infos']]
        for m,path in g.items():
            if [i['timestamp_utc_iso'] for i in path['infos']]!=times:
                raise ValueError(f'Non-paired UTC timestamps for {key}: {m}')
            if not np.allclose(path['initial_pose'],ref['initial_pose'],atol=1e-7):
                raise ValueError(f'Non-paired initial orientation for {key}: {m}')
            for t,(info,reference) in enumerate(zip(path['infos'],ref['infos'])):
                for field in ('solar_zenith_deg','solar_azimuth_deg',
                              'dni_wm2','dhi_wm2','ghi_wm2'):
                    if info.get(field) is not None and reference.get(field) is not None:
                        if not np.isclose(float(info[field]),float(reference[field]),atol=1e-6):
                            raise ValueError(f'Mismatch in paired physical conditions ({field}) at step {t}')
    return rows


def _load_rows(path):
    with Path(path).open(newline='',encoding='utf-8') as f:
        rd=csv.DictReader(f)
        if not rd.fieldnames or set(rd.fieldnames)!=set(FIELDS):
            raise ValueError('Unsupported rich trajectory CSV format')
        rows=[]
        for x in rd:
            v={k:(_fixed_float(x[k]) if k in FLOATS else x[k]) for k in FIELDS}
            v['seed']=int(v['seed']);v['step']=int(v['step'])
            rows.append(v)
    return rows


def _validate_rows(rows):
    grouped=defaultdict(list)
    for row in rows:
        if row['method'] not in METHODS:raise ValueError('Unknown saved controller')
        grouped[(str(row['scenario_id']),int(row['seed']),row['method'])].append(row)
    if not grouped:raise ValueError('Rich trajectory CSV is empty')
    by_pair=defaultdict(dict)
    for (sid,seed,method), rr in grouped.items():
        if len(rr)!=117 or [r['step'] for r in rr]!=list(range(1,118)):
            raise ValueError('Corrupted/misordered trajectory rows')
        times=[datetime.fromisoformat(str(r['timestamp_utc_iso']).replace('Z','+00:00')) for r in rr]
        if any(t.tzinfo is None or t.utcoffset().total_seconds()!=0 for t in times):
            raise ValueError('Non UTC time')
        if any(abs((b-a).total_seconds()-300)>1e-5 for a,b in zip(times,times[1:])):
            raise ValueError('Bad 5-minute time sequence')
        if not np.allclose([r['cumulative_gross_kwh'] for r in rr],
                           np.cumsum([r['energy_kwh'] for r in rr]),atol=1e-6):
            raise ValueError('Corrupt gross cumulatives')
        if not np.allclose([r['cumulative_net_kwh'] for r in rr],
                           np.cumsum([r['reward_kwh'] for r in rr]),atol=1e-6):
            raise ValueError('Corrupt net cumulatives')
        if method in FEASIBLE:
            for r in rr:
                if not np.isclose(r['reward_kwh'],r['energy_kwh']-r['movement_cost_kwh'],atol=1e-6):
                    raise ValueError('Corrupt step net energy identity')
        by_pair[(sid,seed)][method]=rr
    for key, methods in by_pair.items():
        if set(methods)!=set(next(iter(by_pair.values()))):
            raise ValueError('Inconsistent method sets across pairs')
        clock=[r['timestamp_utc_iso'] for r in next(iter(methods.values()))]
        if any([r['timestamp_utc_iso'] for r in rr]!=clock for rr in methods.values()):
            raise ValueError('Paired timestamp mismatch')
    if not set(FEASIBLE).issubset(set(next(iter(by_pair.values())))):
        raise ValueError('Missing feasible comparison methods')
    return grouped, by_pair


def write_rich_csv(paths, outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    rows=expand_paths(paths)
    path=outdir/'trajectories_rich.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    return path


def _arr(rows,key):return np.asarray([r[key] for r in rows],dtype=float)


def _stats(rows):
    grouped,by_pair=_validate_rows(rows)
    results={}
    for m in METHODS:
        rr=[r for r in rows if r['method']==m]
        if not rr:continue
        episodes=[v for k,v in grouped.items() if k[2]==m]
        agg=lambda name:float(np.mean([sum(_arr(p,name)) for p in episodes]))
        result={'n':len(episodes),'mean_gross_kwh':agg('energy_kwh')}
        if m in FEASIBLE:
            result.update(mean_net_kwh=agg('reward_kwh'),mean_movement_cost_kwh=agg('movement_cost_kwh'),
                          mean_abs_motion_deg=float(np.mean([np.sum(np.abs(_arr(p,'actual_delta_tilt_deg'))+np.abs(_arr(p,'actual_delta_azimuth_deg'))) for p in episodes])))
        else:result.update(mean_net_kwh=None,mean_movement_cost_kwh=None,mean_abs_motion_deg=None)
        results[m]=result
    per_pair={}
    for key,methods in by_pair.items():
        per_pair[key]={m:{'net':float(_arr(rr,'reward_kwh').sum()) if m in FEASIBLE else None,
                          'gross':float(_arr(rr,'energy_kwh').sum())} for m,rr in methods.items()}
    return results,per_pair,grouped,by_pair


def _dominant_condition(rr):
    labels=[r['weather_condition'] for r in rr if r['weather_condition']]
    if not labels: return None
    unique,counts=np.unique(labels,return_counts=True)
    return str(unique[int(np.argmax(counts))])


def render_from_csv(rich_csv, outdir):
    """Regenerate advanced paper-style plots from saved CSV, no simulator needed."""
    rows=_load_rows(rich_csv)
    summary,per_pair,grouped,by_pair=_stats(rows)
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    figures=[]
    methods=[m for m in METHODS if m in summary]
    pairs=sorted(by_pair)
    with PdfPages(out/'advanced_paper_figures.pdf') as pdf:
        def save(fig,name):
            if name not in ('03_orientation_occupancy.png','04_hourly_power_gap.png'):
                fig.tight_layout()
            filename=out/name
            fig.savefig(filename,dpi=190,bbox_inches='tight')
            pdf.savefig(fig)
            plt.close(fig)
            figures.append(str(filename))

        # Legacy plot_paper_eval.plot_energy_decomposition semantics, feasible methods only.
        fig,ax=plt.subplots(figsize=(10,5))
        xs=np.arange(len(FEASIBLE));w=.36
        ax.bar(xs-w/2,[summary[m]['mean_gross_kwh'] for m in FEASIBLE],w,label='Gross')
        ax.bar(xs+w/2,[summary[m]['mean_net_kwh'] for m in FEASIBLE],w,label='Net')
        for i,m in enumerate(FEASIBLE):
            ax.text(i,summary[m]['mean_gross_kwh']+.003,
                    f"Motion {summary[m]['mean_movement_cost_kwh']*1000:.3f} Wh",
                    fontsize=8,ha='center')
        ax.set_xticks(xs);ax.set_xticklabels([LABELS[m] for m in FEASIBLE],rotation=12)
        ax.set_ylabel('Mean daily electrical energy (kWh)')
        ax.set_title('Energy accounting: gross − actuator cost = net (feasible controllers)')
        ax.legend();ax.grid(axis='y',alpha=.25);save(fig,'01_energy_decomposition.png')

        # Scenario pairing by ID, NOT by row index (repair legacy index pairing).
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for ax,m,title in zip(axes,('fixed_no_motion','dual_axis_slew_limited'),
                              ('SAC / MBPO vs fixed','SAC / MBPO vs dual axis')):
            for learn in ('mbpo','sac'):
                gain=[per_pair[k][learn]['net']-per_pair[k][m]['net'] for k in pairs]
                ax.hist(gain,bins=min(15,max(4,int(np.sqrt(len(gain))*2))),alpha=.55,
                        label=f'{LABELS[learn]} mean={np.mean(gain):.3f} kWh')
            ax.axvline(0,color='black',ls='--',lw=1);ax.legend(fontsize=8)
            ax.set_title(title);ax.set_xlabel('Paired daily net energy gain (kWh)')
            ax.set_ylabel('Number of daily scenarios');ax.grid(axis='y',alpha=.25)
        save(fig,'02_daily_gain_distribution.png')

        # Occupancy heatmap: use physical orientations rather than action commands.
        plotted=[m for m in ('mbpo','sac','dual_axis_slew_limited','fixed_no_motion') if m in summary]
        histograms=[]
        for m in plotted:
            data=[r for r in rows if r['method']==m]
            H,_,_=np.histogram2d(_arr(data,'azimuth_deg'),_arr(data,'tilt_deg'),
                  bins=[np.linspace(0,360,49),np.linspace(0,90,31)])
            histograms.append(H/H.sum()*100. if H.sum() else H)
        vmax=max(h.max() for h in histograms)
        fig,axes=plt.subplots(1,len(plotted),figsize=(4.5*len(plotted),4.5),sharey=True)
        for ax,m,H in zip(np.ravel(axes),plotted,histograms):
            im=ax.imshow(H.T,origin='lower',aspect='auto',extent=[0,360,0,90],
                         cmap='magma',vmin=0,vmax=max(vmax,1e-9))
            ax.set_title(LABELS[m]);ax.set_xlabel('Panel azimuth (°)')
        axes[0].set_ylabel('Panel tilt (°)')
        fig.colorbar(im,ax=list(np.ravel(axes)),label='% of observed steps',shrink=.72)
        save(fig,'03_orientation_occupancy.png')

        # Paired power gap vs *feasible* sun-tracking reference, 5-min exact alignment.
        fig,axes=plt.subplots(1,2,figsize=(13,max(4,min(13,2+len(pairs)*.12))),sharey=True)
        vmax=0.;arrays=[]
        for learn in ('mbpo','sac'):
            z=np.vstack([_arr(by_pair[k]['dual_axis_slew_limited'],'power_w')-
                         _arr(by_pair[k][learn],'power_w') for k in pairs])
            arrays.append(z)
            vmax=max(vmax,np.percentile(np.abs(z),98))
        for ax,z,learn in zip(axes,arrays,('mbpo','sac')):
            im=ax.imshow(z,aspect='auto',cmap='RdBu_r',vmin=-max(vmax,1e-9),
                         vmax=max(vmax,1e-9),interpolation='nearest')
            ax.set_title(f'Dual-axis power − {LABELS[learn]} power')
            ax.set_xlabel('Post-step UTC clock hour')
            pos=np.linspace(0,116,7,dtype=int);ax.set_xticks(pos)
            ax.set_xticklabels([f'{12+(i+1)*5/60:.2f}' for i in pos])
        if len(pairs)<=35:
            axes[0].set_yticks(np.arange(len(pairs)))
            axes[0].set_yticklabels([k[0] for k in pairs],fontsize=7)
        else:axes[0].set_ylabel('NSRDB scenario (sorted by ID)')
        fig.colorbar(im,ax=list(axes),shrink=.75,label='Power deficit (W)')
        save(fig,'04_hourly_power_gap.png')

        # Performance ratio vs slew-limited dual axis; no ideal reference net.
        fig,ax=plt.subplots(figsize=(11,5))
        valid=[k for k in pairs if per_pair[k]['dual_axis_slew_limited']['net']>1e-9]
        for m,marker in [('mbpo','o'),('sac','s'),('fixed_no_motion','^')]:
            vals=np.array([per_pair[k][m]['net']/per_pair[k]['dual_axis_slew_limited']['net'] for k in valid])
            ax.scatter(np.arange(len(valid)),vals,label=LABELS[m],marker=marker,s=24,alpha=.7)
        ax.axhline(1,color='black',ls='--',lw=1)
        ax.set_ylabel('Net kWh / feasible dual-axis net kWh')
        ax.set_xlabel('Matched NSRDB scenarios (sorted by ID)')
        ax.set_title('Per-scenario performance ratio — feasible dual-axis reference')
        if len(valid)<=25:
            ax.set_xticks(range(len(valid)));ax.set_xticklabels([k[0] for k in valid],rotation=90,fontsize=7)
        ax.legend();ax.grid(alpha=.25);save(fig,'05_ratio_ladder.png')

        # Angular motion includes actual applied delta (not requested saturated action).
        fig,ax=plt.subplots(figsize=(8,5))
        for m in FEASIBLE:
            xx,yy=[],[]
            for key in pairs:
                rr=by_pair[key][m]
                xx.append(float(np.sum(np.abs(_arr(rr,'actual_delta_tilt_deg'))+
                                       np.abs(_arr(rr,'actual_delta_azimuth_deg')))))
                yy.append(float(np.sum(_arr(rr,'reward_kwh'))))
            ax.scatter(xx,yy,s=20,alpha=.65,label=LABELS[m])
        ax.set_xlabel('Total applied angular movement (degrees / episode)')
        ax.set_ylabel('Net energy (kWh / episode)')
        ax.set_title('Motion versus net energy (movement penalty may be zero)')
        ax.legend();ax.grid(alpha=.25);save(fig,'06_movement_energy_scatter.png')

        # Tracking-action diagnostics & pre/post alignment from diagnose_tracking.py.
        sid,seed=pairs[0]
        fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True)
        t=_arr(by_pair[(sid,seed)]['sac'],'clock_hour_utc')
        for m in ('mbpo','sac','dual_axis_slew_limited'):
            rr=by_pair[(sid,seed)][m]
            axs[0].plot(t,_arr(rr,'action_tilt'),label=LABELS[m])
            axs[1].plot(t,_arr(rr,'action_azimuth'),label=LABELS[m])
            axs[2].plot(t,_arr(rr,'tilt_tracking_error_deg'),label=LABELS[m])
        axs[0].set_ylabel('Tilt action [-1,1]');axs[1].set_ylabel('Azimuth action [-1,1]')
        axs[2].set_ylabel('Post-step tilt − sun target (degrees)')
        axs[2].set_xlabel('UTC post-step hour')
        for ax in axs:ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.suptitle(f'Control actions and tilt error | scenario={sid}, seed={seed}')
        save(fig,'07_tracking_action_diagnostics.png')

        # 3D action space time path; no model-generated actions in evaluation.
        fig=plt.figure(figsize=(10,7));ax=fig.add_subplot(111,projection='3d')
        for m in ('mbpo','sac','dual_axis_slew_limited'):
            rr=by_pair[(sid,seed)][m]
            ax.plot(t,_arr(rr,'action_tilt'),_arr(rr,'action_azimuth'),label=LABELS[m])
        ax.set(xlabel='Post-step UTC hour',ylabel='Tilt control [-1,1]',
               zlabel='Azimuth control [-1,1]',title='3D normalized control trajectories')
        ax.legend(loc='upper left',bbox_to_anchor=(0.02,.98),fontsize=8);save(fig,'08_actions_3d.png')

        # Correct astronomical target uses post-action timestamp, sun-facing azimuth.
        fig=plt.figure(figsize=(10,7));ax=fig.add_subplot(111,projection='3d')
        for m in ('mbpo','sac','dual_axis_slew_limited'):
            rr=by_pair[(sid,seed)][m]
            ax.plot(t,_arr(rr,'tilt_deg'),_arr(rr,'azimuth_deg'),label=LABELS[m])
        rr=by_pair[(sid,seed)]['dual_axis_slew_limited']
        ax.plot(t,_arr(rr,'tilt_sun_target_deg'),_arr(rr,'solar_azimuth_deg'),
                label='Astronomical target (unconstrained)',ls=':',color='black')
        ax.set(xlabel='UTC hour',ylabel='Surface tilt (°)',zlabel='Surface azimuth (°)',
               title='Physical panel orientation vs solar pointing (3D)')
        ax.legend(loc='upper left',bbox_to_anchor=(0.02,.98),fontsize=8);save(fig,'09_orientation_vs_sun_3d.png')

        fig,ax=plt.subplots(figsize=(10,5))
        seasons=['winter','spring','summer','fall']
        for m in FEASIBLE:
            vv=[]
            for season in seasons:
                vals=[float(np.sum(_arr(rr,'reward_kwh'))) for (sid,sd,method),rr in grouped.items()
                      if method==m and rr[0]['season']==season]
                vv.append(np.mean(vals) if vals else np.nan)
            ax.plot(seasons,vv,marker='o',label=LABELS[m])
        ax.set_ylabel('Mean daily net energy (kWh)')
        ax.set_title('Seasonal energy (only seasons represented in evaluation)')
        ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.25)
        save(fig,'10_seasonal_energy.png')

        # Weather labels are optional; only render if collected from simulator,
        # never invent or infer cloud categories from insufficient metadata.
        conditions=sorted({r['weather_condition'] for r in rows if r['weather_condition']})
        if conditions:
            fig,ax=plt.subplots(figsize=(10,5))
            positions=np.arange(len(conditions));w=.19
            for j,m in enumerate(FEASIBLE):
                values=[]
                for cond in conditions:
                    z=[float(np.sum(_arr(rr,'reward_kwh'))) for (sid,sd,mm),rr in grouped.items()
                       if mm==m and _dominant_condition(rr)==cond]
                    values.append(np.mean(z) if z else np.nan)
                ax.bar(positions+(j-1.5)*w,values,w,label=LABELS[m])
            ax.set_xticks(positions);ax.set_xticklabels(conditions)
            ax.set_title('Weather-conditional net energy (episode diagnostic label)')
            ax.set_ylabel('Mean net kWh / episode');ax.legend(fontsize=8,ncol=2)
            ax.grid(axis='y',alpha=.25);save(fig,'11_weather_conditions.png')

    report={'source_csv':str(Path(rich_csv).resolve()),'episode_pairs':len(pairs),
            'scenarios':len({sid for sid,_ in pairs}),
            'methods':summary,'plots':[Path(p).name for p in figures],
            'protocol': 'UTC native 5min; 117 real simulator transitions per matched episode',
            'limitations':[
                'An instantaneous sun-pointing or POA ideal is gross-only; no feasible net-energy claim.',
                'Per-scenario ratios exclude zero-denominator days.',
                'Scenario overlap with training is possible; not a held-out claim.',
                'Solar pointing error uses post-step sun position and is diagnostic, not an optimal action label.',
                'Movement penalty may be zero: movement scatter is not an energy-efficiency ranking.',
                'Episode weather conditions are diagnostics only; they are not forecast data for the agent.',
            ]}
    (out/'advanced_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    (out/'figure_manifest.txt').write_text('\n'.join(report['plots'])+'\n',encoding='utf-8')
    return report


def create_advanced_report(paths, eval_dir):
    """Write rich export and figures without removing legacy or modern outputs."""
    outdir=Path(eval_dir)/'advanced'
    rich=write_rich_csv(paths,outdir)
    return render_from_csv(rich,outdir)
