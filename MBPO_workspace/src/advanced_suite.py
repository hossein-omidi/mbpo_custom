"""Offline POA-only scientific plotting. No training, simulator, Ray or TF imports.

Line: design-weighted mean over sampled NSRDB dates at a fixed post-step UTC clock.
Shading: weighted observed-day RMS dispersion, descriptive ONLY (not SE/CI).
Policy/action replicates on one date are averaged before across-date aggregation.
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

VISIBLE=('mbpo','sac','fixed_no_motion','poa_instant_ideal')
FEASIBLE=VISIBLE[:3]
LABELS={'mbpo':'MBPO','sac':'SAC','fixed_no_motion':'Fixed mount',
        'poa_instant_ideal':'Instant DC optimum (gross-only)'}
DETAIL=('net_step_kwh','action_tilt','action_azimuth','actual_delta_tilt_deg',
        'actual_delta_azimuth_deg','pre_tilt_deg','pre_azimuth_deg','solar_zenith_deg',
        'solar_azimuth_deg','dni_wm2','dhi_wm2','ghi_wm2','weather_condition',
        'power_model','poa_global','poa_global_wm2','power_dc_w','effective_irradiance_wm2',
        'cell_temperature_c','ambient_temperature_c','wind_speed_m_s','wind_speed_10m_ms',
        'wind_reference_height_m','gamma_pdc','sapm_mount','wind_powerlaw_exponent')
BASIC=('scenario_id','seed','method','step','utc_iso','power_w','energy_kwh',
       'cumulative_gross_kwh','tilt_deg','azimuth_deg','movement_cost_kwh')
ALL=BASIC+DETAIL
MEASURES={'power_w':'Power (W)','cumulative_gross_kwh':'Cumulative gross energy (kWh)',
          'cumulative_net_kwh':'Cumulative net energy (kWh)',
          'action_tilt':'Commanded normalized tilt action',
          'action_azimuth':'Commanded normalized azimuth action',
          'tilt_deg':'Panel tilt (degrees)'}


def export_detail(paths, outdir):
    """Optional lossless *observed* metadata export at evaluation time.

    A None field means unmeasured, not zero. Ideal commands are undefined.
    Every complete 117-step method-day-seed path is checked before export.
    """
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    fp=outdir/'trajectories_detailed.csv'
    seen=set()
    with fp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=ALL);w.writeheader()
        for p in paths:
            if p['method'] not in VISIBLE:continue
            key=(str(p['scenario_id']),int(p['seed']),p['method'])
            if key in seen:raise ValueError('Duplicate evaluation path '+str(key))
            seen.add(key)
            if any(len(p[z])!=117 for z in ('infos','power','energy','reward','cost','tilt','azimuth','action_tilt','action_azimuth','delta_tilt_deg','delta_azimuth_deg')):
                raise ValueError('Detailed export requires 117 complete steps: '+str(key))
            cum=0.
            for i,info in enumerate(p['infos']):
                power=float(p['power'][i]);energy=float(p['energy'][i]);cost=float(p['cost'][i]);cum+=energy
                if not np.isfinite([power,energy,cost]).all() or not np.isclose(energy,power/12000.,atol=1.e-6):
                    raise ValueError('Invalid power/energy relationship '+str(key))
                is_ideal=p['method']=='poa_instant_ideal'
                if not is_ideal:
                    if not np.isclose(float(p['reward'][i]),energy-cost,atol=1e-6):
                        raise ValueError('Step net-energy identity failure '+str(key))
                    a=np.array([p['action_tilt'][i],p['action_azimuth'][i],
                                p['delta_tilt_deg'][i],p['delta_azimuth_deg'][i]],float)
                    if not np.isfinite(a).all() or np.max(np.abs(a[:2]))>1.00001:
                        raise ValueError('Invalid normalized action or applied motion '+str(key))
                    old_tilt=float(p['initial_pose'][0]) if i==0 else float(p['tilt'][i-1])
                    old_az=float(p['initial_pose'][1]) if i==0 else float(p['azimuth'][i-1])
                    actual_tilt=float(p['tilt'][i]);actual_az=float(p['azimuth'][i])
                    if not np.isclose(actual_tilt,np.clip(old_tilt+a[2],0,90),atol=1e-5):
                        raise ValueError('Applied tilt motion does not match panel orientation '+str(key))
                    az_error=(actual_az-old_az-a[3]+180)%360-180
                    if abs(az_error)>1e-5:
                        raise ValueError('Applied azimuth motion does not match panel orientation '+str(key))
                row=dict(scenario_id=key[0],seed=key[1],method=key[2],step=i+1,
                         utc_iso=info['timestamp_utc_iso'],power_w=power,energy_kwh=energy,
                         cumulative_gross_kwh=cum,tilt_deg=float(p['tilt'][i]),
                         azimuth_deg=float(p['azimuth'][i]),movement_cost_kwh=cost,
                         net_step_kwh=None if is_ideal else float(p['reward'][i]),
                         action_tilt=None if is_ideal else float(p['action_tilt'][i]),
                         action_azimuth=None if is_ideal else float(p['action_azimuth'][i]),
                         actual_delta_tilt_deg=None if is_ideal else float(p['delta_tilt_deg'][i]),
                         actual_delta_azimuth_deg=None if is_ideal else float(p['delta_azimuth_deg'][i]),
                         pre_tilt_deg=float(p['initial_pose'][0]) if i==0 else float(p['tilt'][i-1]),
                         pre_azimuth_deg=float(p['initial_pose'][1]) if i==0 else float(p['azimuth'][i-1]),
                         solar_zenith_deg=info.get('solar_zenith_deg'),
                         solar_azimuth_deg=info.get('solar_azimuth_deg'),
                         dni_wm2=info.get('dni_wm2'),dhi_wm2=info.get('dhi_wm2'),
                         ghi_wm2=info.get('ghi_wm2'),weather_condition=info.get('weather_condition'),
                         power_model=info.get('power_model','legacy'),
                         poa_global=info.get('poa_global'),
                         poa_global_wm2=info.get('poa_global'),power_dc_w=info.get('power_dc_w'),
                         effective_irradiance_wm2=info.get('effective_irradiance_wm2'),
                         cell_temperature_c=info.get('cell_temperature_c'),
                         ambient_temperature_c=info.get('ambient_temperature_c'),
                         wind_speed_m_s=info.get('wind_speed_m_s'),
                         wind_speed_10m_ms=info.get('wind_speed_10m_ms'),
                         wind_reference_height_m=info.get('wind_reference_height_m'),
                         gamma_pdc=info.get('gamma_pdc'),
                         sapm_mount=info.get('sapm_mount'),
                         wind_powerlaw_exponent=info.get('wind_powerlaw_exponent'))
                w.writerow(row)
    return fp


def _float(v):
    if v is None or str(v).strip().lower() in ('','none','nan'):return np.nan
    return float(v)


def _time(s):
    d=datetime.fromisoformat(s.replace('Z','+00:00'))
    if d.utcoffset() is None or d.utcoffset().total_seconds()!=0:
        raise ValueError('All episode timestamps must include an explicit UTC offset')
    return d


def _read_eval(folder):
    folder=Path(folder)
    detailed=folder/'trajectories_detailed.csv'
    base=folder/'trajectories.csv'
    legacy_rich=folder/'advanced'/'trajectories_rich.csv'
    source=detailed if detailed.exists() else legacy_rich if legacy_rich.exists() else base
    if not source.exists():raise FileNotFoundError('Missing trajectories.csv or existing rich trajectory CSV in '+str(folder))
    rich=source!=base
    grouped=defaultdict(list)
    with source.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        fieldnames=set(reader.fieldnames or ())
        if source==legacy_rich:
            # Historic advanced output uses timestamp_utc_iso and reward_kwh.
            fieldnames=(fieldnames-{'timestamp_utc_iso','reward_kwh'})|{'utc_iso','net_step_kwh'}
        if not set(BASIC).issubset(fieldnames):
            raise ValueError('Incompatible saved trajectory format: '+str(source))
        for row in reader:
            method=row['method']
            if method not in VISIBLE:continue
            key=(str(row['scenario_id']),int(row['seed']),method)
            if source==legacy_rich:
                row['utc_iso']=row.get('timestamp_utc_iso')
                row['net_step_kwh']=row.get('reward_kwh') if method!='poa_instant_ideal' else None
            parsed={k:(_float(row.get(k)) if k not in ('scenario_id','method','utc_iso','weather_condition','power_model','sapm_mount') else row.get(k)) for k in ALL}
            parsed['scenario_id']=key[0];parsed['seed']=key[1];parsed['method']=method
            parsed['step']=int(row['step']);parsed['utc_iso']=str(row['utc_iso'])
            grouped[key].append(parsed)
    if not grouped:raise ValueError('No visible controller trajectories')
    keys=defaultdict(dict); clocks={}
    for (sid,seed,method),rr in grouped.items():
        if len(rr)!=117 or [int(row['step']) for row in rr]!=list(range(1,118)):
            raise ValueError('Expected 117 ordered steps per path: '+str((sid,seed,method)))
        dt=[_time(r['utc_iso']) for r in rr]
        if any(abs((b-a).total_seconds()-300)>1e-5 for a,b in zip(dt,dt[1:])):
            raise ValueError('Non-native five-minute clock: '+str((sid,seed,method)))
        h=np.array([x.hour+x.minute/60. for x in dt])
        if not np.isclose(h[0],12.+5./60.) or not np.isclose(h[-1],21.75):
            raise ValueError('Unsupported control window')
        for i,r in enumerate(rr):
            if not np.isfinite([r['power_w'],r['energy_kwh'],r['tilt_deg'],r['azimuth_deg'],r['movement_cost_kwh']]).all():
                raise ValueError('Nonfinite physical measurements')
            if not np.isclose(r['energy_kwh'],r['power_w']/12000.,rtol=1e-5,atol=1e-6):
                raise ValueError('Power/energy inconsistency')
        if not np.allclose(np.cumsum([r['energy_kwh'] for r in rr]),[r['cumulative_gross_kwh'] for r in rr],atol=1e-6):
            raise ValueError('Incorrect cumulative energy')
        if method=='poa_instant_ideal' and np.isfinite([r['net_step_kwh'] for r in rr]).any():
            raise ValueError('POA oracle has no feasible net-energy measurement')
        if method in FEASIBLE:
            net=np.asarray([r['net_step_kwh'] for r in rr],dtype=float)
            # Basic historical CSVs omit the entire net-step column; in that
            # case we can reconstruct it exactly from gross minus movement.
            # A partly populated column is damaged data, not a valid fallback.
            if np.isfinite(net).any() and not np.isfinite(net).all():
                raise ValueError('Partly missing/nonfinite net step rewards: '+str((sid,seed,method)))
            if np.isfinite(net).all() and not np.allclose(
                    net,[r['energy_kwh']-r['movement_cost_kwh'] for r in rr],atol=1e-6):
                raise ValueError('Invalid step reward')
        key=(sid,seed)
        keys[key][method]=rr
        if key in clocks and [r['utc_iso'] for r in rr]!=clocks[key]:
            raise ValueError('Nonpaired UTC clocks '+str(key))
        clocks[key]=[r['utc_iso'] for r in rr]
    for key,methods in keys.items():
        if set(methods)!=set(VISIBLE):raise ValueError('Incomplete or non-POA paired controller set '+str(key))
        if rich:
            reference=methods['fixed_no_motion']
            for method, rr in methods.items():
                for field in ('solar_zenith_deg','solar_azimuth_deg','dni_wm2','dhi_wm2','ghi_wm2'):
                    a=np.array([r[field] for r in rr]);b=np.array([r[field] for r in reference])
                    # If a rich export contains the field, it must be complete for
                    # both paired methods; missing values are not weather evidence.
                    if np.isfinite(a).any() != np.isfinite(b).any():
                        raise ValueError('Inconsistent weather field coverage: '+str((key,method,field)))
                    if np.isfinite(a).any():
                        if not np.isfinite(a).all() or not np.isfinite(b).all():
                            raise ValueError('Partially missing weather / solar geometry: '+str((key,method,field)))
                        if not np.allclose(a,b,atol=1e-5):
                            raise ValueError('Nonpaired recorded weather / solar geometry: '+str((key,method,field)))
    by_day=defaultdict(set)
    for sid,seed in keys:by_day[sid].add(seed)
    if len({tuple(sorted(v)) for v in by_day.values()})!=1:
        raise ValueError('Different action seed sets across evaluation days')
    return grouped,keys,source,rich


def _weights(folder, sids):
    fp=Path(folder)/'scenario_selection.json'
    if fp.exists():
        plan=json.loads(fp.read_text(encoding='utf-8'))
        w={s['selected_scenario_id']:int(s['population_days']) for s in plan['strata']}
        if set(w)!=sids or sum(w.values())!=int(plan['population_count']) or min(w.values())<1:
            raise ValueError('Saved trajectory coverage disagrees with selection plan')
        return w,'manifest_bin_population_days'
    return {sid:1 for sid in sids},'equal_weight_observed_scenarios_NO_manifest_claim'


def weighted_curve(curves_by_day, weights):
    """Mean and descriptive (biased for finite population) weighted RMS scatter.

    Curves must represent distinct scenario dates; combine seeds before calling.
    This dispersion is NOT estimator SE, a CI, or unbiased population variance.
    """
    if not curves_by_day or set(curves_by_day)!=set(weights):
        raise ValueError('Require exactly one valid curve per selected date')
    days=sorted(weights)
    a=np.asarray([curves_by_day[sid] for sid in days],float)
    w=np.asarray([weights[sid] for sid in days],float)
    if len(a.shape)!=2 or not np.isfinite(a).all() or not np.isfinite(w).all() or np.any(w<=0):
        raise ValueError('Invalid daily curves or weights')
    mean=np.average(a,axis=0,weights=w)
    if len(days)==1:return mean,None
    scatter=np.sqrt(np.average((a-mean)**2,axis=0,weights=w))
    return mean,scatter


def _day_curves(grouped,metric,methods):
    result={}
    for method in methods:
        per_day=defaultdict(list)
        for (sid,seed,m),rr in grouped.items():
            if m!=method:continue
            arr=np.array([r.get(metric,np.nan) for r in rr],float)
            if metric=='cumulative_net_kwh':
                if m not in FEASIBLE:continue
                if np.isnan(arr).all():
                    arr=np.cumsum(np.array([r['energy_kwh']-r['movement_cost_kwh'] for r in rr],float))
                else:arr=np.cumsum(np.array([r['net_step_kwh'] for r in rr],float))
            if np.isfinite(arr).all():per_day[sid].append(arr)
        if per_day:
            result[method]={sid:np.mean(seeds,axis=0) for sid,seeds in per_day.items()}
    return result


def _save_csv(path,rows,fields):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def render_suite(eval_dir, outdir=None, *, mbpo_training_csv=None, sac_training_csv=None):
    root=Path(eval_dir);out=Path(outdir) if outdir else root/'advanced_suite'
    out.mkdir(parents=True,exist_ok=True)
    grouped,pairs,source,rich=_read_eval(root)
    weights,weight_type=_weights(root,{key[0] for key in pairs})
    # All controller/date/seed pairs are already validated. Collapse seed replicas by day.
    results=[];omitted=[];curve_rows=[]
    hours=np.arange(1,118)*5./60.+12.
    with PdfPages(out/'advanced_suite.pdf') as pdf:
        def save(fig,filename):
            fig.tight_layout()
            fig.savefig(out/filename,dpi=165)
            pdf.savefig(fig)
            plt.close(fig)
            results.append(filename)
        def line_bands(filename,metric,methods,title,zero=False):
            curves=_day_curves(grouped,metric,methods)
            fig,ax=plt.subplots(figsize=(11,5.6))
            shown=0
            for method in methods:
                daycurves=curves.get(method)
                if daycurves is None or set(daycurves)!=set(weights):continue
                mu,sd=weighted_curve(daycurves,weights)
                ax.plot(hours,mu,linewidth=2.15,label=LABELS[method]+' — solid mean')
                color=ax.lines[-1].get_color()
                if sd is not None:
                    ax.fill_between(hours,mu-sd,mu+sd,color=color,alpha=.16)
                for j,(x,m) in enumerate(zip(hours,mu)):
                    curve_rows.append(dict(figure=filename,method=method,metric=metric,
                           clock_hour_utc=f'{x:.8f}',weighted_mean=m,
                           observed_day_rms_sd=None if sd is None else sd[j],
                           n_distinct_days=len(daycurves),represented_manifest_days=sum(weights.values())))
                shown+=1
            if not shown:
                plt.close(fig);omitted.append(filename+': measured field unavailable');return
            if zero:ax.axhline(0,color='black',linestyle='--',linewidth=1)
            ax.set(xlabel='UTC clock hour (post-action)\nShade = observed-day RMS dispersion, not SE/CI; time steps are not independent days',
                   ylabel=MEASURES.get(metric,metric),title=title)
            ax.grid(alpha=.22);ax.legend(fontsize=7,ncol=2,loc='upper left')
            save(fig,filename)

        line_bands('01_power_weighted_mean_spread.png','power_w',VISIBLE,
                   'PV power across NSRDB sampled days — observed weather variability')
        line_bands('02_cumulative_gross_weighted_mean_spread.png','cumulative_gross_kwh',VISIBLE,
                   'Cumulative gross DC generation — includes unconstrained upper bound')
        line_bands('03_cumulative_net_weighted_mean_spread.png','cumulative_net_kwh',FEASIBLE,
                   'Cumulative feasible net generation (no gross-only oracle)')

        # Paired POA gross power surplus and MBPO–SAC gross power differences.
        for metric,filename,comparison,title in (
            ('power_w','04_poa_minus_mbpo_power_band.png',('poa_instant_ideal','mbpo'),
             'Paired instantaneous DC upper-bound surplus over MBPO (gross)'),
            ('power_w','05_mbpo_minus_sac_power_band.png',('mbpo','sac'),
             'Paired physical power contrast: MBPO minus SAC')):
            series=_day_curves(grouped,metric,comparison)
            diff={sid:series[comparison[0]][sid]-series[comparison[1]][sid] for sid in weights}
            mu,sd=weighted_curve(diff,weights)
            fig,ax=plt.subplots(figsize=(11,5.5));ax.axhline(0,color='black',ls='--',lw=1)
            ax.plot(hours,mu,lw=2.2,label='Weighted paired-day mean')
            if sd is not None:ax.fill_between(hours,mu-sd,mu+sd,alpha=.20,label='Observed-day RMS spread, not CI')
            ax.set(xlabel='UTC clock hour (post-action)',ylabel='Gross power difference (W)',title=title)
            ax.grid(alpha=.22);ax.legend(fontsize=8)
            for j,h in enumerate(hours):
                curve_rows.append(dict(figure=filename,method='_minus_'.join(comparison),metric=metric,
                    clock_hour_utc=h,weighted_mean=mu[j],observed_day_rms_sd=None if sd is None else sd[j],
                    n_distinct_days=len(weights),represented_manifest_days=sum(weights.values())))
            save(fig,filename)

        if rich:
            for metric,name in (('action_tilt','06_normalized_tilt_actions_band.png'),
                                ('action_azimuth','07_normalized_azimuth_actions_band.png'),
                                ('tilt_deg','08_panel_tilt_band.png')):
                line_bands(name,metric,('mbpo','sac'),
                           'Controller dynamics across dates — '+MEASURES[metric])
        else:
            line_bands('08_panel_tilt_band.png','tilt_deg',('mbpo','sac'),
                       'Panel tilt across dates (no action fields in historical CSV)')
            omitted.extend(['06_normalized_tilt_actions_band.png: historical CSV lacks action_tilt',
                            '07_normalized_azimuth_actions_band.png: historical CSV lacks action_azimuth'])

        # Canonical UTC-aligned representative plots: never attach statistical uncertainty to one day.
        sid=sorted(weights)[0];seed=sorted({sd for s,sd in pairs if s==sid})[0]
        representative=pairs[(sid,seed)]
        fig,axs=plt.subplots(3,1,sharex=True,figsize=(11,9))
        for m in VISIBLE:
            rr=representative[m]
            axs[0].plot(hours,[r['power_w'] for r in rr],label=LABELS[m])
            axs[1].plot(hours,[r['cumulative_gross_kwh'] for r in rr],label=LABELS[m])
            axs[2].plot(hours,[r['tilt_deg'] for r in rr],label=LABELS[m])
        for ax,yl in zip(axs,('Power (W)','Cumulative gross energy (kWh)','Tilt (°)')):
            ax.set_ylabel(yl);ax.grid(alpha=.2)
        axs[0].legend(fontsize=8,ncol=2);axs[-1].set_xlabel('UTC clock hour (post-action)')
        fig.suptitle(f'Paired physical trajectory — {sid}, seed={seed} (single date; no error bars)')
        save(fig,'09_representative_day_trajectories.png')

        fig=plt.figure(figsize=(10,8));ax=fig.add_subplot(111,projection='3d')
        for m in ('mbpo','sac','poa_instant_ideal'):
            rr=representative[m]
            ax.plot(hours,[r['tilt_deg'] for r in rr],[r['azimuth_deg'] for r in rr],label=LABELS[m])
        ax.set(xlabel='UTC hour',ylabel='Tilt (°)',zlabel='Azimuth (°)',title=f'3D orientation paths — {sid}, one weather day')
        ax.legend();save(fig,'10_orientation_3d_selected_day.png')

        # Weighted orientation occupancy. Not circular arithmetic mean of azimuth.
        fig,axs=plt.subplots(1,3,figsize=(13,4.7),sharex=True,sharey=True)
        bins=(np.linspace(0,360,37),np.linspace(0,90,19))
        maps=[]
        for m in ('mbpo','sac','poa_instant_ideal'):
            xx=[];yy=[];ww=[]
            for sid in sorted(weights):
                seedrr=[rr for (s,seed,meth),rr in grouped.items() if s==sid and meth==m]
                for rr in seedrr:
                    xx.extend([r['azimuth_deg']%360. for r in rr]);yy.extend([r['tilt_deg'] for r in rr])
                    ww.extend([weights[sid]/(len(seedrr)*117.)]*117)
            H,_,_=np.histogram2d(xx,yy,bins=bins,weights=ww)
            maps.append(H/max(H.sum(),1.))
        vmax=max(h.max() for h in maps)
        for ax,m,h in zip(axs,('mbpo','sac','poa_instant_ideal'),maps):
            im=ax.imshow(h.T,origin='lower',aspect='auto',extent=(0,360,0,90),cmap='magma',vmin=0,vmax=vmax)
            ax.set_title(LABELS[m]);ax.set_xlabel('Panel azimuth (°)')
        axs[0].set_ylabel('Panel tilt (°)')
        fig.colorbar(im,ax=axs,shrink=.75,label='Weighted fraction of sampled control steps')
        fig.suptitle('Observed orientation occupancy (not a probabilistic confidence region)')
        # colorbar prevents usual tight_layout optimization; save as is
        fig.savefig(out/'11_orientation_occupancy.png',dpi=165,bbox_inches='tight');pdf.savefig(fig);plt.close(fig);results.append('11_orientation_occupancy.png')

        # From the same paired scenario, POA-minus-MBPO physical power differences.
        # Each row is ONE scenario-date, never an independent 5-minute sample.
        day_sids=sorted(weights)
        poa=_day_curves(grouped,'power_w',('poa_instant_ideal','mbpo'))
        gap=np.array([poa['poa_instant_ideal'][d]-poa['mbpo'][d] for d in day_sids])
        fig,ax=plt.subplots(figsize=(11,max(4.,min(13.,len(day_sids)*.07+2))))
        finite=np.abs(gap[np.isfinite(gap)]);vmax=max(1e-9,float(np.percentile(finite,98)))
        im=ax.imshow(gap,aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax,interpolation='nearest')
        ix=np.linspace(0,116,7).astype(int)
        ax.set_xticks(ix,[f'{hours[i]:.1f}' for i in ix]);ax.set(xlabel='UTC post-step hour',
            ylabel='Distinct sampled scenario day, sorted',title='Instantaneous DC upper bound minus MBPO DC power (W)')
        if len(day_sids)<=30:
            ax.set_yticks(np.arange(len(day_sids)),day_sids,fontsize=7)
        fig.colorbar(im,ax=ax,label='POA − MBPO power (W)')
        save(fig,'20_poa_minus_mbpo_day_hour_heatmap.png')

        features=root/'stratified'/'episode_features.csv'
        if features.exists():
            with features.open(newline='',encoding='utf-8') as f:features_rows=list(csv.DictReader(f))
            fr=defaultdict(list)
            for r in features_rows:
                if r['method'] in VISIBLE and r['scenario_id'] in weights:
                    fr[(r['scenario_id'],r['method'])].append(r)
            def feat(sid,m,k):
                if (sid,m) not in fr:return np.nan
                arr=[_float(r.get(k)) for r in fr[(sid,m)]]
                return float(np.mean(arr)) if arr and np.isfinite(arr).all() else np.nan
            def monthly(metric,name,selected,ylabel,title):
                fig,ax=plt.subplots(figsize=(10.5,5.5));count=0
                for method in selected:
                    xy=[]
                    for month in range(1,13):
                        days=[s for s in weights if int(s[5:7])==month]
                        vals=np.array([feat(s,method,metric) for s in days]);w=np.array([weights[s] for s in days])
                        xy.append(float(np.average(vals,weights=w)) if len(vals) and np.isfinite(vals).all() else np.nan)
                    if np.isfinite(xy).any():ax.plot(np.arange(1,13),xy,marker='o',lw=2,label=LABELS[method]);count+=1
                if not count:plt.close(fig);omitted.append(name+': missing full metric coverage');return
                ax.set(xlabel='Calendar month',ylabel=ylabel,title=title);ax.set_xticks(np.arange(1,13))
                ax.grid(alpha=.2);ax.legend(fontsize=8);save(fig,name)
            monthly('net_energy_kwh','12_monthly_weighted_net_energy.png',FEASIBLE,'Mean feasible net kWh / window',
                    'Year/month-bin weighted energy — NOT a confidence band')
            monthly('movement_total_deg','13_monthly_weighted_movement.png',FEASIBLE,'Abs. tilt + azimuth travel (degrees / window)',
                    'Population-bin weighted physical motion — no motor electricity implied')
            monthly('gross_energy_kwh','14_monthly_gross_poa.png',VISIBLE,'Mean gross kWh / window',
                    'Gross DC energy including unconstrained upper bound')

            # Scientific separation: POA oracle has GROSS only; a common net-energy
            # plot must NEVER include it. Both charts use the same sampled days and
            # exact calendar-bin population weights, not raw counts of episodes.
            seasons=[('winter',(12,1,2)),('spring',(3,4,5)),
                     ('summer',(6,7,8)),('fall',(9,10,11))]
            for metric,methods,filename,heading in (
                ('gross_energy_kwh',VISIBLE,'21_seasonal_weighted_energy.png',
                 'Seasonal gross DC energy: policies and instantaneous upper bound'),
                ('net_energy_kwh',FEASIBLE,'27_seasonal_weighted_net_energy.png',
                 'Seasonal feasible net PV energy: POA oracle excluded')):
                fig,ax=plt.subplots(figsize=(10.5,5.2));shown=0
                for method in methods:
                    ys=[]
                    for season,months_ in seasons:
                        ids=[d for d in weights if int(d[5:7]) in months_]
                        vals=np.array([feat(d,method,metric) for d in ids],float)
                        ws=np.array([weights[d] for d in ids],float)
                        ys.append(float(np.average(vals,weights=ws)) if len(ids) and np.isfinite(vals).all() else np.nan)
                    # A partially measured method has NO publishable seasonal series.
                    if np.isfinite(ys).all():
                        ax.plot([x[0] for x in seasons],ys,marker='o',lw=2,label=LABELS[method]);shown+=1
                    else:
                        omitted.append(filename+': incomplete '+method+' '+metric+' coverage')
                if shown==len(methods):
                    ax.set(ylabel='Weighted mean '+('gross' if metric.startswith('gross') else 'net')+
                           ' energy (kWh / 117-step window)',title=heading+' (no sampling CI)')
                    ax.legend(fontsize=8);ax.grid(alpha=.2);save(fig,filename)
                else:
                    plt.close(fig);omitted.append(filename+': incomplete comparable method coverage')

            # Weather groups are assigned after sampling: descriptive ONLY.
            groups_by_weather=defaultdict(list)
            for s in sorted(weights):
                recorded=fr.get((s,'fixed_no_motion'),[])
                labels=[r.get('irradiance_regime','unavailable') for r in recorded]
                group=labels[0] if labels and len(set(labels))==1 else 'unavailable'
                gap_kwh=feat(s,'mbpo','net_energy_kwh')-feat(s,'sac','net_energy_kwh')
                if np.isfinite(gap_kwh):groups_by_weather[group].append(gap_kwh)
            if groups_by_weather:
                fig,ax=plt.subplots(figsize=(11,5))
                names=sorted(groups_by_weather)
                ax.boxplot([groups_by_weather[n] for n in names],showmeans=True)
                ax.set_xticks(np.arange(1,len(names)+1))
                ax.set_xticklabels([f'{n}\n(n={len(groups_by_weather[n])})' for n in names])
                ax.axhline(0,color='black',ls='--',lw=1)
                ax.set(ylabel='MBPO − SAC paired net kWh',title='Observed sampled days by irradiance regime (post-sampling groups; no CI)')
                ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2)
                save(fig,'22_irradiance_regime_paired_gains.png')

            fig,ax=plt.subplots(figsize=(10,5));n=0
            for m in FEASIBLE:
                xs=np.array([feat(s,m,'movement_total_deg') for s in sorted(weights)])
                ys=np.array([feat(s,m,'net_energy_kwh') for s in sorted(weights)])
                ok=np.isfinite(xs)&np.isfinite(ys)
                if ok.any():ax.scatter(xs[ok],ys[ok],label=LABELS[m],s=18,alpha=.5);n+=1
            if n:
                ax.set(xlabel='Total applied angular travel (°/window)',ylabel='Feasible net energy (kWh/window)',
                       title='Observed movement–energy relation (not a causal effect)')
                ax.grid(alpha=.2);ax.legend();save(fig,'15_motion_vs_energy_scatter.png')
            else:plt.close(fig);omitted.append('15_motion_vs_energy_scatter.png: no measured movement')

            fig,ax=plt.subplots(figsize=(10.5,5.5));shown=0
            for m in FEASIBLE:
                values=[];labels=[]
                for sid in sorted(weights):
                    v=feat(sid,m,'net_energy_kwh')
                    if np.isfinite(v):values.append(v);labels.append(weights[sid])
                if len(values)==len(weights):
                    # Weighted empirical cumulative distribution over sampled dates.
                    order=np.argsort(values);val=np.array(values)[order];wt=np.array(labels,float)[order]
                    ax.step(val,np.cumsum(wt)/wt.sum(),where='post',label=LABELS[m]);shown+=1
            if shown:
                ax.set(xlabel='Net energy (kWh / window)',ylabel='Weighted observed-day cumulative fraction',
                       title='Weighted observed-day energy ECDF (descriptive, not sampling CDF CI)')
                ax.grid(alpha=.2);ax.legend();save(fig,'16_weighted_day_energy_ecdf.png')
            else:plt.close(fig);omitted.append('16_weighted_day_energy_ecdf.png: incomplete episode metrics')
        else:
            omitted.append('12–16 monthly/motion/ECDF panels: missing stratified/episode_features.csv')

        # The distribution of paired day-level net gains is unconditional on features.
        daily_net={m:_day_curves(grouped,'cumulative_net_kwh',(m,)).get(m) for m in ('mbpo','sac','fixed_no_motion')}
        if all(daily_net.values()):
            fig,ax=plt.subplots(figsize=(10,5.2))
            for left,right in [('mbpo','sac'),('mbpo','fixed_no_motion'),('sac','fixed_no_motion')]:
                d=np.array([daily_net[left][sid][-1]-daily_net[right][sid][-1] for sid in sorted(weights)],float)
                ax.hist(d,bins=max(3,min(14,int(np.sqrt(len(d))*2))),alpha=.35,label=f'{LABELS[left]} − {LABELS[right]}')
            ax.axvline(0,color='black',ls='--',lw=1)
            ax.set(xlabel='Matched-day net energy difference (kWh)',ylabel='Observed sampled-day count',
                   title='Distribution of paired net energy gains (no unpaired SEM or fake CI)')
            ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
            save(fig,'23_paired_day_gain_histograms.png')

        if rich:
            fig,axs=plt.subplots(2,1,sharex=True,figsize=(11,8))
            for m in ('mbpo','sac'):
                rr=representative[m]
                for ax,metric in ((axs[0],'action_tilt'),(axs[1],'action_azimuth')):
                    if np.isfinite([r[metric] for r in rr]).all():
                        ax.plot(hours,[r[metric] for r in rr],label=LABELS[m])
            axs[0].set_ylabel('Normalized tilt action');axs[1].set_ylabel('Normalized azimuth action')
            axs[1].set_xlabel('UTC clock hour (post-action)')
            for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
            fig.suptitle(f'Policy actions — {sid} (single paired day)')
            save(fig,'17_representative_action_controls.png')
            fig=plt.figure(figsize=(10,7));ax=fig.add_subplot(111,projection='3d')
            count=0
            for method in ('mbpo','sac'):
                rr=representative[method]
                a=np.array([r['action_tilt'] for r in rr]);b=np.array([r['action_azimuth'] for r in rr])
                if np.isfinite(a).all() and np.isfinite(b).all():
                    ax.plot(hours,a,b,label=LABELS[method],lw=2);count+=1
            if count:
                ax.set(xlabel='UTC hour',ylabel='Normalized tilt action',zlabel='Normalized azimuth action',
                       title=f'3D commanded actions — {sid} (one observed day)')
                ax.legend();save(fig,'24_actions_3d_selected_day.png')
            else:plt.close(fig);omitted.append('24_actions_3d_selected_day.png: measured action columns incomplete')
            fig,ax=plt.subplots(figsize=(10,5))
            solar=np.array([r['solar_zenith_deg'] for r in representative['fixed_no_motion']],float)
            if np.isfinite(solar).all():
                ax.plot(hours,solar,label='Solar zenith',ls='--',color='black')
                for m in ('mbpo','sac','poa_instant_ideal'):
                    ax.plot(hours,[r['tilt_deg'] for r in representative[m]],label=LABELS[m])
                ax.set(xlabel='UTC hour',ylabel='Tilt / solar zenith (°)',
                       title='Orientation versus sun geometry (diagnostic, not POA optimality test)')
                ax.grid(alpha=.2);ax.legend();save(fig,'18_tilt_vs_solar_geometry.png')
            else:plt.close(fig);omitted.append('18_tilt_vs_solar_geometry.png: solar zenith not available')
        else:omitted.append('17–18 policy/solar diagnostics: rich trajectory fields only generated by new evaluation runs')

        for method,train_file in (('mbpo',mbpo_training_csv),('sac',sac_training_csv)):
            if train_file and Path(train_file).exists():
                with Path(train_file).open(newline='',encoding='utf-8') as f:hist=list(csv.DictReader(f))
                if hist and {'env_steps','return_kwh'}<=set(hist[0]):
                    x=np.array([_float(r['env_steps']) for r in hist]);y=np.array([_float(r['return_kwh']) for r in hist]);ok=np.isfinite(x)&np.isfinite(y)
                    if ok.any():
                        fig,ax=plt.subplots(figsize=(11,5))
                        ax.plot(x[ok],y[ok],lw=.9,alpha=.30,label='Raw single-episode energy')
                        # Rolling mean across chronologically adjacent logged episodes is smoothing,
                        # NOT independent-replicate mean and must never get a CI ribbon.
                        w=min(20,int(ok.sum()))
                        if w>=2:
                            xx=x[ok];yy=y[ok]
                            ax.plot(xx[w-1:],np.convolve(yy,np.ones(w)/w,mode='valid'),lw=2,
                                    label=f'Trailing {w}-episode moving average (NOT uncertainty)')
                        ax.set(xlabel='REAL environment transitions',ylabel='Training episode net energy (kWh)',
                               title=f'{LABELS[method]} training: episodic weather variation + learning (no CI)')
                        ax.legend();ax.grid(alpha=.2)
                        save(fig,'19_training_'+method+'.png')
            else:omitted.append('19_training_'+method+'.png: training CSV not supplied')

        # Single-run optimizer diagnostics: raw point estimates/logged losses only.
        # These are NOT independent training-seed error bars.
        if mbpo_training_csv and Path(mbpo_training_csv).is_file():
            with Path(mbpo_training_csv).open(newline='',encoding='utf-8') as f:
                hist=list(csv.DictReader(f))
            if hist and 'env_steps' in hist[0]:
                available=[(k,label) for k,label in (
                    ('model_val_nll','Held-out ensemble NLL'),
                    ('model_val_power_rmse_norm','Held-out power RMSE (normalized)'),
                    ('model_replay_size','Synthetic replay size'),
                    ('critic_loss','Critic loss'),
                    ('actor_loss','Actor loss'),
                    ('alpha','Entropy temperature α')) if k in hist[0]]
                if available:
                    fig,axs=plt.subplots(3,2,figsize=(12,10))
                    x=np.array([_float(r['env_steps']) for r in hist],float)
                    for ax,(key,label) in zip(axs.flat,available):
                        y=np.array([_float(r[key]) for r in hist],float)
                        ok=np.isfinite(x)&np.isfinite(y)
                        ax.plot(x[ok],y[ok],linewidth=1.1);ax.set(xlabel='Real environment steps',ylabel=label)
                        ax.grid(alpha=.2)
                    for ax in list(axs.flat)[len(available):]:ax.axis('off')
                    fig.suptitle('MBPO dynamics / SAC optimization diagnostics (logged single-run values; no error bars)')
                    save(fig,'25_mbpo_model_optimizer_diagnostics.png')
        if sac_training_csv and Path(sac_training_csv).is_file():
            with Path(sac_training_csv).open(newline='',encoding='utf-8') as f:
                hist=list(csv.DictReader(f))
            if hist and 'env_steps' in hist[0]:
                avail=[(k,label) for k,label in (
                    ('critic_loss','Critic loss'),('actor_loss','Actor loss'),
                    ('entropy','Policy entropy'),('alpha','Entropy temperature α')) if k in hist[0]]
                if avail:
                    fig,axs=plt.subplots(2,2,figsize=(11,7));x=np.array([_float(r['env_steps']) for r in hist],float)
                    for ax,(key,label) in zip(axs.flat,avail):
                        y=np.array([_float(r[key]) for r in hist],float)
                        ok=np.isfinite(x)&np.isfinite(y)
                        ax.plot(x[ok],y[ok],linewidth=1.2);ax.set(xlabel='Real environment steps',ylabel=label)
                        ax.grid(alpha=.2)
                    for ax in list(axs.flat)[len(avail):]:ax.axis('off')
                    fig.suptitle('Standalone SAC training diagnostics (single-run logs)')
                    save(fig,'26_sac_optimizer_diagnostics.png')

    _save_csv(out/'curve_band_data.csv',curve_rows,
              ('figure','method','metric','clock_hour_utc','weighted_mean','observed_day_rms_sd',
               'n_distinct_days','represented_manifest_days'))
    summary=dict(source=str(source),rich_metadata_used=rich,weight_type=weight_type,
        sampled_unique_dates=len(weights),represented_dates=int(sum(weights.values())),
        assumptions=['Each curve is one scenario-day after averaging policy seeds',
        'At each UTC clock point the solid line is the weighted mean over sampled days',
        'Shading is weighted RMS observed-day scatter; not unbiased population SD, SEM, or CI',
        'Each 5-minute time point is not an independent sampling unit',
        'POA is unconstrained gross-only: never included in net-energy curves',
        'Panel azimuth is circular; no ordinary arithmetic azimuth mean/SD is plotted',
        'Monthly weighted means use selected day weights; no justified CI from one sample per bin',
        'Moving average of sequential training episodes is not training-seed uncertainty'],
        figures=results,skipped=omitted,curve_data='curve_band_data.csv')
    (out/'figure_manifest.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    return summary
