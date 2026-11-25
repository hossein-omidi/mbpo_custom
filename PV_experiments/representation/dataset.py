"""Original NSRDB catalogue only, on the same native post-action UTC grid."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .common import Figures,save_csv,write_json,style,plt
from .statistics import season,finite,clock_grid


def read_dataset(cfg,split):
    from mbpo.env.nsrdb_weather import load_scenario_manifest,NsrdbYearCache,resample_weather_to_episode_times,scenario_episode_date
    from mbpo.env.stage_a_data_contract import nsrdb_dataset_fingerprint
    from SAC_workspace.src.training.experiment_protocol import build_scenario_protocol
    if build_scenario_protocol(cfg.sac.manifest,cfg.sac.protocol)!=split:raise ValueError('Catalogue differs from frozen split')
    manifest=load_scenario_manifest(cfg.sac.manifest); cache=NsrdbYearCache(manifest)
    lookup={sid:role for role in ('train','validation','test') for sid in split[f'{role}_ids']}
    rows=[];profiles=[];hours=1/12; columns=['ghi','dni','dhi','temperature','wind_speed']
    for row in sorted(manifest['scenarios'],key=lambda x:x['scenario_id']):
        sid=row['scenario_id']; date=scenario_episode_date(row); y=int(date.year)
        grid=clock_grid(str(date.date()),cfg.sac.eval_env['start_time'],int(cfg.sac.eval_env['periods'])-1)
        if 'timestamps' in row and not pd.DatetimeIndex(pd.to_datetime(row['timestamps'][1:],utc=True)).equals(grid):raise ValueError(f'Manifest timestamp mismatch: {sid}')
        year=cache.get_year(y)
        if grid[-1].year!=y:
            year=pd.concat([year,cache.get_year(grid[-1].year)]).sort_index();year=year[~year.index.duplicated(keep='last')]
        aligned=resample_weather_to_episode_times(year,grid)
        if not aligned.index.equals(grid):raise ValueError(f'Native UTC grid mismatch: {sid}')
        values=finite(aligned[columns]); ghi,dni,dhi,temp,wind=values.T
        if (values[:,[0,1,2,4]]<0).any():raise ValueError(f'Negative weather input {sid}')
        # Ratio of integrated irradiation, not mean of unstable dawn ratios.
        kt=np.nan
        if 'clearsky_ghi' in aligned:
            cs=finite(aligned.clearsky_ghi);kt=ghi.sum()/cs.sum() if cs.sum()>0 else np.nan
        rows.append(dict(scenario_id=sid,source_year=y,month=date.month,day=date.day,season=season(date.month),role=lookup[sid],ghi_kwh_m2=ghi.sum()*hours/1000,dni_kwh_m2=dni.sum()*hours/1000,dhi_kwh_m2=dhi.sum()*hours/1000,temperature_c=temp.mean(),wind_m_s=wind.mean(),diffuse_fraction=dhi.sum()/ghi.sum() if ghi.sum()>0 else np.nan,clearsky_ghi_ratio=kt))
        profiles.append(ghi)
    df=pd.DataFrame(rows)
    if df.scenario_id.duplicated().any() or set(df.scenario_id)!=set(lookup):raise ValueError('Catalogue coverage mismatch')
    return df,np.asarray(profiles),nsrdb_dataset_fingerprint(cfg.sac.manifest)


def render_dataset(df,profiles,out,fingerprint):
    figs=Figures(out);save_csv(out,'historical_day_weather.csv',df)
    cover=df.groupby(['source_year','month','role']).size().rename('n_days').reset_index();save_csv(out,'scenario_coverage.csv',cover)
    fig,ax=plt.subplots(figsize=(9,4)); tab=cover.groupby(['month','role']).n_days.sum().unstack(fill_value=0);tab.plot.bar(stacked=True,ax=ax);ax.set(xlabel='Calendar month',ylabel='Historical scenarios');style(ax);figs.save(fig,'D01_split_coverage','Are seasonal conditions represented in each partition?','Actual catalogue counts, not probability estimates.','scenario_coverage.csv')
    monthly=[];fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
    for ax,fields,unit in zip(axes,[('ghi_kwh_m2','dni_kwh_m2','dhi_kwh_m2'),('temperature_c',),('wind_m_s',)],['Irradiation (kWh/m² / episode)','Mean ambient temperature (°C)','Mean wind speed (m/s)']):
        for field in fields:
            stats=df.groupby('month')[field].agg(['mean','std','count']);x=stats.index.to_numpy();mean=stats['mean'].to_numpy();sd=stats['std'].to_numpy();ax.plot(x,mean,label=field.replace('_kwh_m2','').upper() if len(fields)>1 else field);ax.fill_between(x,mean-sd,mean+sd,alpha=.12);monthly.extend(dict(month=int(m),quantity=field,mean=v,sd=s,n_days=int(n)) for m,v,s,n in zip(x,mean,sd,stats['count']));ax.set_ylabel(unit);style(ax)
        if len(fields)>1:ax.legend()
    axes[-1].set(xlabel='Month',xticks=range(1,13));save_csv(out,'monthly_weather_statistics.csv',monthly);fig.suptitle('Historical-day mean ± sample SD');figs.save(fig,'D02_seasonal_weather','How does the physical forcing vary seasonally?','Daily integrals or daily average meteorological values; sample SD across historical days.','historical_day_weather.csv')
    fig,ax=plt.subplots(figsize=(9,4)); x=np.asarray(profiles); daykey=pd.to_datetime('2000-'+df.month.astype(str).str.zfill(2)+'-'+df.day.astype(str).str.zfill(2)).dt.dayofyear; mat=np.full((366,x.shape[1]),np.nan)
    for d in np.unique(daykey):mat[d-1]=x[daykey==d].mean(axis=0)
    im=ax.imshow(mat.T,aspect='auto',origin='lower',interpolation='nearest',extent=[.5,366.5,8.75+.5/12,8.75+(x.shape[1]+.5)/12]);fig.colorbar(im,ax=ax,label='Mean GHI (W/m²)');ax.set(xlabel='Calendar day (leap-year reference; Feb 29 retained)',ylabel='UTC hour (unwrapped)');figs.save(fig,'D03_annual_irradiance_heatmap','How does the daily irradiation envelope change over the year?','Equal historical-day mean at each calendar month/day and post-action UTC step. Missing days blank.','native NSRDB GHI grid')
    save_csv(out,'calendar_ghi_heatmap.csv',pd.DataFrame(mat,columns=[f'step_{i+1}' for i in range(x.shape[1])]).assign(calendar_day=np.arange(1,367)))
    fig,ax=plt.subplots(figsize=(8,4)); data=df.clearsky_ghi_ratio.dropna();field='clearsky_ghi_ratio' if len(data) else 'diffuse_fraction';data=df[field].dropna();ax.hist(data,bins=30);ax.set(xlabel='Integrated GHI / NSRDB clear-sky GHI' if field=='clearsky_ghi_ratio' else 'Integrated DHI / GHI',ylabel='Historical days');style(ax);figs.save(fig,'D04_weather_regime_distribution','How diverse are the historical radiation conditions?','One ratio per original historical episode. No claim of calibrated cloud classes.','historical_day_weather.csv')
    figs.finish();write_json(out/'dataset_manifest.json',dict(fingerprint=fingerprint,n_days=len(df),note='Original unperturbed NSRDB modelled weather; post-action episode window, not whole 24-hour daily totals. Calendar day is episode start date.'))
