"""Calendar-based energy/effort heatmaps from frozen paired Monte Carlo records.
No annual extrapolation, no interpolation of unevaluated days, no gross-only
oracle presented as a feasible net-energy controller.
"""
from __future__ import annotations
import calendar
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize,TwoSlopeNorm,ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from .advanced_learning import finite,same,sha_file,write_json,save,csv_out

METHODS=('fixed_no_motion','sun_tracking','sac','mbpo')
LABELS={'fixed_no_motion':'Fixed mounting','sun_tracking':'Astronomical tracking','sac':'SAC','mbpo':'MBPO','mbpo_minus_sac':'MBPO − SAC','mbpo_minus_sun_tracking':'MBPO − astronomical'}
FIELDS=('gross_kwh','movement_kwh','net_kwh')
MONTHS=[calendar.month_abbr[m] for m in range(1,13)]


def load_calendar(cfg,seeds,stage,root):
    from PV_experiments.core import verify_pair,run_path
    from PV_experiments.report import validate_mc_source
    from PV_experiments.evaluate import existing_report_is_valid,_provenance,_require_matching_provenance
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Invalid seed selection')
    seeds=sorted(seeds);splits=[verify_pair(cfg,s,root) for s in seeds]
    if any(x!=splits[0] for x in splits[1:]):raise ValueError('Frozen scenario partitions differ across seeds')
    days=sorted(splits[0][f'{stage}_ids']);all_days=sum([list(splits[0][f'{role}_ids']) for role in ('train','validation','test')],[])
    if len(set(all_days))!=len(all_days) or not days:raise ValueError('Invalid frozen split')
    # Scenario IDs in the audited branch are ISO historical dates, no invented dates.
    for sid in all_days:
        if pd.Timestamp(sid).date().isoformat()!=sid:raise ValueError(f'Invalid historical date ID: {sid}')
    M=None;ref=None;baseline=None;sources={};allvalues=[];meta0=None
    for seed in seeds:
        for a in ('sac','mbpo'):
            p=run_path(cfg,a,seed,root);from PV_experiments.core import read_json
            meta=read_json(p/'run_metadata.json');keys=('scenario_protocol_sha256','resolved_mdp_sha256','shared_sac_config_sha256','horizon','energy_objective','movement_cost_mode','movement_penalty','power_model');sig=tuple(meta[k] for k in keys)
            if meta0 is None:meta0=sig
            elif sig!=meta0:raise ValueError('Physical/reward/learning config mismatch across seeds')
            if meta['movement_cost_mode']!='geometry' or float(meta['movement_penalty'])!=1.:raise ValueError('Expected enabled physical movement energy (geometry, gain 1)')
            sources[f'{a}_seed{seed}/run_metadata.json']=sha_file(p/'run_metadata.json');sources[f'{a}_seed{seed}/scenario_split.json']=sha_file(p/'scenario_split.json')
        directory,summary=validate_mc_source(cfg,stage,seed,root);m=int(summary['draws_per_day']);
        if M is None:M=m
        elif M!=m:raise ValueError('Draw count differs across seeds')
        if M<2:raise ValueError('At least two weather realizations required')
        if not existing_report_is_valid(cfg,stage,seed,M,directory,root):raise ValueError('Incomplete MC output')
        _require_matching_provenance(directory,_provenance(cfg,stage,seed,root,draws=M))
        fingerprint=summary['_validated_weather_hashes']
        if ref is None:ref=fingerprint
        elif ref!=fingerprint:raise ValueError('Weather fingerprints differ across seeds')
        p=directory/'all_day_draw_method.csv';d=pd.read_csv(p,dtype={'scenario_id':str,'profile_sha256':str});index=pd.MultiIndex.from_product([days,range(M),METHODS],names=['scenario_id','draw','method']);d=d.set_index(['scenario_id','draw','method'],verify_integrity=True)
        if len(d)!=len(index) or not d.index.isin(index).all():raise ValueError('Missing or extra day/draw/method records')
        d=d.reindex(index)
        expected_hash=np.repeat([fingerprint[(sid,rid)] for sid in days for rid in range(M)],len(METHODS))
        if not np.array_equal(d.profile_sha256.to_numpy(),expected_hash):raise ValueError('Raw fingerprint mismatch')
        v=finite(d[list(FIELDS)],'MC energies').reshape(len(days),M,len(METHODS),len(FIELDS));same(v[...,0]-v[...,1],v[...,2],'Gross minus movement equals net')
        if (v[...,:2]<-1e-10).any():raise ValueError('Negative gross or actuator energy')
        same(v[:,:,0,1],np.zeros((len(days),M)),'Fixed movement cost')
        if baseline is None:baseline=v[:,:,:2,:].copy()
        else:same(baseline,v[:,:,:2,:],'Baseline gross/net/actuator across seeds')
        if stage=='validation':
            for a in ('sac','mbpo'):
                rf=run_path(cfg,a,seed,root)/'periodic_evaluation/evaluation_references.csv';r=pd.read_csv(rf,dtype={'scenario_id':str}).set_index('scenario_id',verify_integrity=True)
                if set(r.index)!=set(days) or not (r.episode_steps==195).all():raise ValueError('Periodic reference coverage/horizon mismatch')
                for mi,prefix in [(0,'fixed'),(1,'dual_axis')]:
                    same(r.loc[days,[f'{prefix}_gross_energy_kwh',f'{prefix}_movement_cost_kwh',f'{prefix}_net_energy_kwh']].to_numpy(),v[:,0,mi,:],'Periodic baseline versus MC realization zero')
                sources[f'{a}_seed{seed}/periodic_evaluation/evaluation_references.csv']=sha_file(rf)
        allvalues.append(v)
        for fname in ('all_day_draw_method.csv','mc_summary.json','manager_provenance.json'):sources[f'seed{seed}/{fname}']=sha_file(directory/fname)
    return dict(seeds=seeds,days=days,catalogue_days=sorted(all_days),draws=M,values=np.stack(allvalues),sources=sources,stage=stage,protocol=splits[0]['protocol_sha256'])


def aggregate(d,metric,layout):
    """One cell: mean daily response, equal historical dates after averaging draws.
    Separately expose day variability, seed-mean SD and sample coverage.
    """
    if layout not in ('month_day','year_month'):raise ValueError(layout)
    days=pd.DatetimeIndex(d['days']);catalogue=pd.DatetimeIndex(d['catalogue_days']);S,D,M=d['values'].shape[:3]
    if len(days)!=D or len(set(days))!=D or len(set(catalogue))!=len(catalogue) or not set(days).issubset(set(catalogue)) or S!=len(d['seeds']) or M!=d['draws']:raise ValueError('Calendar grid/coverage mismatch')
    if metric in FIELDS:
        idx=FIELDS.index(metric);data={a:finite(d['values'][:,:,:,i,idx]) for i,a in enumerate(METHODS)}
    elif metric=='paired_net_gain_kwh':
        x=d['values'][...,2];data={name:x[:,:,:,3]-x[:,:,:,j] for name,j in [('mbpo_minus_sac',2),('mbpo_minus_sun_tracking',1)]}
    else:raise ValueError(metric)
    years=range(int(catalogue.year.min()),int(catalogue.year.max())+1)
    # Include calendar cells without eligible catalogue days (shown grey if valid).
    rowkeys=range(1,13) if layout=='month_day' else years
    colkeys=range(1,32) if layout=='month_day' else range(1,13)
    rows=[]
    for i in rowkeys:
        for j in colkeys:
            valid=j<=calendar.monthrange(2000,i)[1] if layout=='month_day' else True
            idx=np.flatnonzero((days.month==i)&(days.day==j) if layout=='month_day' else (days.year==i)&(days.month==j))
            eligible=int(np.sum((catalogue.month==i)&(catalogue.day==j) if layout=='month_day' else (catalogue.year==i)&(catalogue.month==j)))
            for a,x in data.items():
                base=dict(layout=layout,row=i,column=j,method=a,metric=metric,valid_calendar_cell=valid,n_historical_days=len(idx),n_catalogue_days=eligible,n_training_seeds=S,n_draws_per_day=M,mean_daily_kwh=None,sd_seed_means_kwh=None,sd_historical_day_means_kwh=None,sum_evaluated_daily_means_kwh=None)
                if len(idx):
                    sample=x[:,idx,:];seed_means=sample.mean(axis=(1,2));day_means=sample.mean(axis=(0,2));base.update(mean_daily_kwh=float(day_means.mean()),sd_seed_means_kwh=float(seed_means.std(ddof=1)) if S>1 else None,sd_historical_day_means_kwh=float(day_means.std(ddof=1)) if len(idx)>1 else None,sum_evaluated_daily_means_kwh=float(day_means.sum()))
                rows.append(base)
    return pd.DataFrame(rows)


def _panel_grid(table,methods,layout,metric,out):
    isdiff=metric=='paired_net_gain_kwh';rows=sorted(table.row.unique());cols=sorted(table.column.unique());shape=(len(rows),len(cols));valid=table[table.method==methods[0]].pivot(index='row',columns='column',values='valid_calendar_cell').loc[rows,cols].to_numpy(dtype=bool)
    allvals=finite(table.mean_daily_kwh.dropna(),'calendar cell values');lo=float(allvals.min());hi=float(allvals.max())
    if isdiff or lo<0:
        v=max(abs(lo),abs(hi),1e-9);norm=TwoSlopeNorm(vmin=-v,vcenter=0,vmax=v);cmap='RdBu_r'
    else:norm=Normalize(0,max(hi,1e-9));cmap='viridis'
    n=len(methods);nrows=2 if n==4 else 1;ncols=2
    fig,axs=plt.subplots(nrows,ncols,figsize=(12.3,6.4 if layout=='month_day' and n==4 else 4.1 if layout=='month_day' else 5.5 if n==4 else 3.6),squeeze=False,layout='constrained')
    cm=plt.get_cmap(cmap).copy();cm.set_bad((0,0,0,0))
    for ax,a in zip(axs.flat,methods):
        q=table[table.method==a];z=q.pivot(index='row',columns='column',values='mean_daily_kwh').loc[rows,cols].to_numpy(dtype=float)
        # True square cells; pale gray for unobserved valid dates, white for invalid calendar dates.
        bg=np.where(valid,1.,0.);ax.imshow(bg,cmap=ListedColormap(['white','#d9d9d9']),vmin=0,vmax=1,interpolation='nearest',aspect='equal')
        im=ax.imshow(np.ma.masked_invalid(z),cmap=cm,norm=norm,interpolation='nearest',aspect='equal')
        ax.set_title(LABELS[a],fontsize=10);ax.set_yticks(range(len(rows)),MONTHS if layout=='month_day' else [str(y) for y in rows]);ax.set_xticks(range(0,len(cols),5) if layout=='month_day' else range(12),[str(cols[i]) for i in range(0,len(cols),5)] if layout=='month_day' else MONTHS,fontsize=7);ax.tick_params(axis='y',labelsize=8,length=0);ax.tick_params(axis='x',length=0)
        ax.set_xticks(np.arange(-.5,shape[1],1),minor=True);ax.set_yticks(np.arange(-.5,shape[0],1),minor=True);ax.grid(which='minor',color='white',linewidth=.35);ax.tick_params(which='minor',bottom=False,left=False)
        for sp in ax.spines.values():sp.set_visible(False)
        if layout=='year_month':
            for i,j in np.ndindex(shape):
                if np.isfinite(z[i,j]):
                    rgba=im.cmap(im.norm(z[i,j]));luma=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2];label=f'{z[i,j]:.3f}' if metric=='movement_kwh' else f'{z[i,j]:.2f}'
                    ax.text(j,i,label,ha='center',va='center',fontsize=5.6,color='black' if luma>.52 else 'white')
    unit='Paired net energy difference (kWh / evaluated day)' if isdiff else {'net_kwh':'Mean net DC energy (kWh / evaluated day)','gross_kwh':'Mean gross DC energy (kWh / evaluated day)','movement_kwh':'Mean actuator energy (kWh / evaluated day)'}[metric]
    fig.colorbar(im,ax=axs.ravel().tolist(),orientation='vertical',shrink=.85,pad=.025,label=unit)
    fig.suptitle(('Month × day of month (across historical years)' if layout=='month_day' else 'Historical year × month')+' — '+metric.replace('_kwh','').replace('_',' '),fontsize=12)
    fig.legend(handles=[Patch(facecolor='#d9d9d9',label='Not evaluated'),Patch(facecolor='white',edgecolor='.7',label='Invalid calendar date')],loc='outside lower center',ncol=2,fontsize=8,frameon=False)
    save(fig,out,f'H_{layout}_{metric}')


def render_calendar(d,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);manifest=[]
    for layout in ('month_day','year_month'):
        for metric in ('net_kwh','gross_kwh','movement_kwh','paired_net_gain_kwh'):
            t=aggregate(d,metric,layout);methods=list(t.method.unique());t.to_csv(out/f'calendar_{layout}_{metric}.csv',index=False,float_format='%.12g');_panel_grid(t,methods,layout,metric,out)
            manifest.append(dict(figure=f'H_{layout}_{metric}',layout=layout,metric=metric,estimand='Equal historical-day mean after averaging equal weather draws and selected seed means. Pair before averaging for differences. No extrapolation.',scale='Shared across method panels; zero-centered for signed data.'))
    # One dedicated coverage heatmap shows how many eligible historical days enter each month/day block.
    t=aggregate(d,'net_kwh','month_day');t=t[t.method==METHODS[0]];n=t.pivot(index='row',columns='column',values='n_historical_days').to_numpy();valid=t.pivot(index='row',columns='column',values='valid_calendar_cell').to_numpy(dtype=bool)
    fig,ax=plt.subplots(figsize=(8.6,4),layout='constrained');im=ax.imshow(np.ma.masked_where(~valid,n),cmap='Blues',vmin=0,vmax=max(1,int(n.max())),aspect='equal',interpolation='nearest');ax.set(yticks=range(12),yticklabels=MONTHS,xticks=range(0,31,5),xticklabels=range(1,32,5),xlabel='Day of month',title='Historical-day coverage (evaluated dates across years)');fig.colorbar(im,ax=ax,label='Number of historical dates');ax.set_xticks(np.arange(-.5,31,1),minor=True);ax.set_yticks(np.arange(-.5,12,1),minor=True);ax.grid(which='minor',color='white',lw=.4);ax.tick_params(which='minor',bottom=False,left=False);save(fig,out,'H_coverage_month_day');manifest.append(dict(figure='H_coverage_month_day',estimand='Historical date count, not seed or draw count.'))
    write_json(out/'figure_manifest.json',manifest);write_json(out/'calendar_scope.json',dict(stage=d['stage'],seeds=d['seeds'],draws=d['draws'],historical_days=d['days'],catalogue_days_count=len(d['catalogue_days']),note='The full frozen partition is evaluated, but not the full calendar population. Net and gross are mean kWh per evaluated daily episode; not monthly or annual total. The oracle is excluded because its feasible net energy is undefined. Each CSV also records the sum over evaluated daily means only; no imputed days.'))
