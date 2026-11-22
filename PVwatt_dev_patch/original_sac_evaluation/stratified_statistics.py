"""Design-weighted analysis of paired SAC/MBPO NSRDB scenario samples.

The sampling frame is the manifest. Per-day evaluation is NOT a 24h yield,
and an instant orientation oracle is gross-only, never a feasible net reference.
Reports are independently regenerable from plan JSON + episode_features.csv.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from SAC_workspace.src.evaluation.scenario_sampling import validate_selection

IDEAL = {'dual_axis_instant_ideal', 'poa_instant_ideal'}
SEASONS = {12:'winter', 1:'winter',2:'winter',3:'spring',4:'spring',5:'spring',
           6:'summer',7:'summer',8:'summer',9:'fall',10:'fall',11:'fall'}
FIELDS = ('scenario_id','seed','method','year','month','season','stratum','population_days',
          'weather_condition','net_energy_kwh','gross_energy_kwh','movement_cost_kwh',
          'movement_tilt_deg','movement_azimuth_deg','movement_total_deg',
          'mean_power_w','mean_dni_wm2','mean_dhi_wm2','mean_ghi_wm2',
          'episode_diffuse_fraction','irradiance_regime','mean_solar_tilt_error_deg','mean_solar_azimuth_error_deg','n_steps')
FLOATS = {'population_days','net_energy_kwh','gross_energy_kwh','movement_cost_kwh',
          'movement_tilt_deg','movement_azimuth_deg','movement_total_deg',
          'mean_power_w','mean_dni_wm2','mean_dhi_wm2','mean_ghi_wm2',
          'episode_diffuse_fraction','mean_solar_tilt_error_deg','mean_solar_azimuth_error_deg'}


def _expected_horizon(plan):
    grid = plan.get('episode') or {}
    for key, expected in (('start_time', '08:45'), ('periods', 196),
                          ('freq', '5min'), ('tz', 'UTC')):
        if grid.get(key) != expected:
            raise ValueError(
                'Stratified statistics require the full-day 08:45 UTC / '
                '196-timestamp sampling plan; regenerate the selection plan'
            )
    return int(grid['periods']) - 1


def _time_grid_note(plan):
    ep = plan['episode']
    start = datetime.fromisoformat('2020-01-01T' + ep['start_time']).replace(tzinfo=timezone.utc)
    end = start + timedelta(minutes=_expected_horizon(plan) * 5)
    return (f"{_expected_horizon(plan)} transitions; {start:%H:%M} to "
            f"{end:%H:%M} UTC {'next day' if end.date() > start.date() else 'same day'}, "
            'not a 24-hour UTC calendar interval')


def _safe_mean(xs):
    # A partial measured weather series must not silently become a 'daily mean'
    # with a smaller denominator. Missing inputs are reported as unavailable.
    values=list(xs)
    if not values or any(x is None for x in values):return None
    vals=np.asarray(values,dtype=float)
    return float(vals.mean()) if np.isfinite(vals).all() else None


def _modal(xs):
    counts = Counter(str(v) for v in xs if v is not None and str(v).strip() and str(v).lower()!='none')
    return sorted(counts, key=lambda x:(-counts[x],x))[0] if counts else 'unavailable'


def make_episode_features(paths, plan):
    """Extract one row per method/day/seed; no simulator calls."""
    weights={s['selected_scenario_id']: s for s in plan['strata']}
    expected_steps = _expected_horizon(plan)
    rows=[]
    for p in paths:
        sid=str(p['scenario_id'])
        if sid not in weights:
            raise ValueError('Evaluation path is not part of selected NSRDB sample: ' + sid)
        s=weights[sid]; method=p['method']; ideal=method in IDEAL
        date_value=date.fromisoformat(sid)
        infos=p.get('infos',[])
        steps=len(infos)
        if steps != expected_steps or any(len(p.get(k, [])) != steps
                                         for k in ('energy', 'power', 'tilt', 'azimuth')):
            raise ValueError(
                f'Incomplete full-day trajectory for {sid} / {method}: '
                f'expected {expected_steps} transitions, got {steps}'
            )
        # Check the actual post-action UTC clock and the midnight rollover.
        expected_first = (datetime.fromisoformat(sid + 'T' + plan['episode']['start_time'])
                          .replace(tzinfo=timezone.utc) + timedelta(minutes=5))
        expected_last = expected_first + timedelta(minutes=(steps - 1) * 5)
        first = datetime.fromisoformat(str(infos[0].get('timestamp_utc_iso', '')))
        last = datetime.fromisoformat(str(infos[-1].get('timestamp_utc_iso', '')))
        if (first.utcoffset() != timedelta(0) or last.utcoffset() != timedelta(0)
                or first != expected_first or last != expected_last):
            raise ValueError(
                f'Post-action UTC episode grid mismatch for {sid}/{method}: '
                f'expected {expected_first.isoformat()} .. {expected_last.isoformat()}, '
                f'got {first.isoformat()} .. {last.isoformat()}'
            )
        if method not in IDEAL and (p['net_energy_kwh'] is None or p['movement_cost_kwh'] is None):
            raise ValueError('Feasible controller missing net energy or movement cost')
        if ideal and (p['net_energy_kwh'] is not None or p['movement_cost_kwh'] is not None):
            raise ValueError('Ideal orientation must be gross-only')
        tilt=np.asarray(p.get('delta_tilt_deg',[]),float)
        az=np.asarray(p.get('delta_azimuth_deg',[]),float)
        if ideal: tilt_move=az_move=None
        else:
            if len(tilt)!=steps or len(az)!=steps or not np.isfinite(tilt).all() or not np.isfinite(az).all():
                raise ValueError('Feasible controller has invalid actual applied motion')
            tilt_move=float(np.abs(tilt).sum())
            az_move=float(np.abs(az).sum())
        zen=np.asarray([i.get('solar_zenith_deg') if i.get('solar_zenith_deg') is not None else np.nan for i in infos],float)
        saz=np.asarray([i.get('solar_azimuth_deg') if i.get('solar_azimuth_deg') is not None else np.nan for i in infos],float)
        current_tilt=np.asarray(p['tilt'],float)
        current_az=np.asarray(p['azimuth'],float)
        err_tilt=np.abs(current_tilt - np.clip(zen,0,90))
        err_az=np.abs((current_az-saz+180)%360-180)
        dhi=np.asarray([i.get('dhi_wm2') if i.get('dhi_wm2') is not None else np.nan for i in infos],float)
        ghi=np.asarray([i.get('ghi_wm2') if i.get('ghi_wm2') is not None else np.nan for i in infos],float)
        ratio=(float(dhi.sum())/float(ghi.sum()) if np.isfinite(ghi).all()
               and np.isfinite(dhi).all() and float(ghi.sum())>1e-9 else None)
        regime=('unavailable' if ratio is None else
                'low diffuse (<0.30)' if ratio<0.30 else
                'mixed diffuse (0.30–0.60)' if ratio<0.60 else
                'high diffuse (>=0.60)')
        result={
            'scenario_id': sid,'seed': int(p['seed']),'method': method,
            'year':date_value.year,'month':date_value.month,'season':SEASONS[date_value.month],
            'stratum':s['stratum'],'population_days':int(s['population_days']),
            'weather_condition':_modal([i.get('weather_condition') for i in infos]),
            'net_energy_kwh':p['net_energy_kwh'], 'gross_energy_kwh':p['gross_energy_kwh'],
            'movement_cost_kwh':p['movement_cost_kwh'],
            'movement_tilt_deg':tilt_move,'movement_azimuth_deg':az_move,
            'movement_total_deg':(tilt_move+az_move if not ideal else None),
            'mean_power_w':_safe_mean(p['power']),
            'mean_dni_wm2':_safe_mean([i.get('dni_wm2') for i in infos]),
            'mean_dhi_wm2':_safe_mean([i.get('dhi_wm2') for i in infos]),
            'mean_ghi_wm2':_safe_mean([i.get('ghi_wm2') for i in infos]),
            'episode_diffuse_fraction':ratio,'irradiance_regime':regime,
            'mean_solar_tilt_error_deg':_safe_mean(err_tilt),
            'mean_solar_azimuth_error_deg':_safe_mean(err_az),
            'n_steps':steps,
        }
        rows.append(result)
    return rows


def _keyed(rows, plan):
    """Ensure no pseudoreplication and that ALL methods use identical days/seeds."""
    selected=set(plan['selected_scenario_ids'])
    groups=defaultdict(dict)
    for row in rows:
        sid=row['scenario_id'];seed=int(row['seed']);method=row['method']
        if sid not in selected:raise ValueError('Unselected scenario '+sid)
        if method in groups[(sid,seed)]:raise ValueError('Duplicate method/day/seed')
        groups[(sid,seed)][method]=row
    if not groups:raise ValueError('Empty evaluation features')
    seen_sids={sid for sid,_ in groups}
    if seen_sids!=selected:raise ValueError('Missing/extra selected scenario results')
    methods=set(next(iter(groups.values())))
    if any(set(g)!=methods for g in groups.values()):
        raise ValueError('Not all controllers evaluated on the same paired conditions')
    seeds_by_day=defaultdict(set)
    for sid,seed in groups:seeds_by_day[sid].add(seed)
    if len({tuple(sorted(x)) for x in seeds_by_day.values()})!=1:
        raise ValueError('Evaluation seeds differ by scenario')
    return groups,sorted(methods)


def _day_aggregate(rows, metric):
    """Average identical-seed policy replicates per DAY before any inference."""
    values=defaultdict(list)
    for r in rows:
        v=r.get(metric)
        if v is not None and np.isfinite(float(v)):
            values[r['scenario_id']].append(float(v))
    return {k:float(np.mean(v)) for k,v in values.items()}


def _weighted(values, plan, subgroup=None, bootstrap_seed=123, n_boot=1000):
    """Population-day weighted estimate; bootstrap YEAR clusters, not 5-min steps.

    Note: year-cluster intervals are descriptive with only seven years and do
    not fully quantify within-bin random sampling uncertainty. They should not
    be called guaranteed-coverage design-based confidence intervals.
    """
    sids=[];weights=[];years=[];v=[]
    eligible=[s for s in plan['strata'] if subgroup is None or subgroup(s)]
    if not eligible:return None
    # Never renormalize the population denominator after dropping a missing bin.
    # For an optional metric, unavailable coverage means the full-group mean is unavailable.
    if any(s['selected_scenario_id'] not in values or
           not np.isfinite(float(values[s['selected_scenario_id']])) for s in eligible):
        return None
    for s in eligible:
        sid=s['selected_scenario_id']
        sids.append(sid);weights.append(float(s['population_days']))
        years.append(int(s['year']));v.append(float(values[sid]))
    vals=np.asarray(v,dtype=float);w=np.asarray(weights,dtype=float)
    value=float(np.average(vals,weights=w))
    order=sorted(set(years));yidx=np.asarray(years,int)
    year_weights=np.array([w[yidx==y].sum() for y in order])
    year_sums=np.array([(w[yidx==y]*vals[yidx==y]).sum() for y in order])
    if len(order)>=2:
        rng=np.random.default_rng(bootstrap_seed)
        sample_idx=rng.integers(0,len(order),size=(n_boot,len(order)))
        boot=year_sums[sample_idx].sum(axis=1)/year_weights[sample_idx].sum(axis=1)
        interval=[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))]
    else: interval=[None,None]
    return {'weighted_mean':value,
            'year_cluster_bootstrap_95_interval':None,
            'exploratory_year_resampling_95_percentiles_NOT_design_CI':interval,
            'sampled_days':len(sids),'represented_days':int(sum(weights)),
            'sample_min':float(vals.min()),'sample_median':float(np.median(vals)),
            'sample_max':float(vals.max()),
            'sample_std':float(vals.std(ddof=1)) if len(vals)>1 else None,
            'sample_std_interpretation':('Unweighted dispersion of selected daily outcomes; '
                'NOT finite-manifest population SD, sampling SE, or CI.')}


def analyse_features(rows,plan,bootstrap_seed=123,n_boot=1000):
    if int(n_boot)<1:raise ValueError('n_boot must be positive')
    if int(plan['population_count'])!=sum(s['population_days'] for s in plan['strata']):
        raise ValueError('Population weights are inconsistent')
    groups,methods=_keyed(rows,plan)
    expected_steps = _expected_horizon(plan)
    if any(int(row.get('n_steps', -1)) != expected_steps for row in rows):
        raise ValueError('Stratified episode features have mixed/incorrect horizons')
    by_method={m:[r for r in rows if r['method']==m] for m in methods}
    features=('net_energy_kwh','gross_energy_kwh','movement_cost_kwh','movement_total_deg',
              'movement_tilt_deg','movement_azimuth_deg','mean_power_w',
              'mean_dni_wm2','mean_dhi_wm2','mean_ghi_wm2','episode_diffuse_fraction',
              'mean_solar_tilt_error_deg','mean_solar_azimuth_error_deg')
    bymetric={m:{k:_day_aggregate(by_method[m],k) for k in features} for m in methods}
    all_stats={}
    for m in methods:
        all_stats[m]={k:_weighted(bymetric[m][k],plan,bootstrap_seed=bootstrap_seed,n_boot=n_boot)
                      for k in features}
    feasible=[m for m in methods if m not in IDEAL]
    # Day-level paired differences, ensuring no incompatible net/oracle comparisons.
    comparisons={}
    for left in ('mbpo','sac'):
        if left not in feasible:continue
        for right in feasible:
            if left==right:continue
            a=bymetric[left]['net_energy_kwh'];b=bymetric[right]['net_energy_kwh']
            paired={sid:a[sid]-b[sid] for sid in (a.keys() & b.keys())}
            stats=_weighted(paired,plan,bootstrap_seed=bootstrap_seed,n_boot=n_boot)
            mu_right=_weighted(b,plan,bootstrap_seed=bootstrap_seed,n_boot=n_boot)
            stats['ratio_of_weighted_means_percent']=(100*stats['weighted_mean']/mu_right['weighted_mean']
                              if abs(mu_right['weighted_mean'])>1e-10 else None)
            stats['weighted_win_fraction']=float(np.average([float(paired[s['selected_scenario_id']]>0)
                      for s in plan['strata']],weights=[s['population_days'] for s in plan['strata']]))
            comparisons[f'{left}_minus_{right}']=stats
    # Gross-only ideal comparisons, separated from net (and never substituted).
    gross_to_ideals={}
    for left in feasible:
        for ideal in methods:
            if ideal not in IDEAL:continue
            x=bymetric[left]['gross_energy_kwh'];y=bymetric[ideal]['gross_energy_kwh']
            vals={sid:x[sid]-y[sid] for sid in x.keys() & y.keys()}
            gross_to_ideals[f'{left}_gross_minus_{ideal}_gross']=_weighted(
                 vals,plan,bootstrap_seed=bootstrap_seed,n_boot=n_boot)

    monthly={};seasonal={};yearly={};weather={};regimes={}
    for m in methods:
        monthly[m]={str(month): {k:_weighted(vals,plan,subgroup=lambda s,mo=month:s['month']==mo,
                                               bootstrap_seed=bootstrap_seed,n_boot=n_boot)
                                  for k,vals in bymetric[m].items() if k in ('net_energy_kwh','gross_energy_kwh','movement_total_deg')}
                    for month in range(1,13)}
        seasonal[m]={season: {k:_weighted(vals,plan,subgroup=lambda s,ss=season:SEASONS[s['month']]==ss,
                                                bootstrap_seed=bootstrap_seed,n_boot=n_boot)
                                  for k,vals in bymetric[m].items() if k in ('net_energy_kwh','gross_energy_kwh','movement_total_deg')}
                     for season in ('winter','spring','summer','fall')}
        yearly[m]={str(y): {k:_weighted(vals,plan,subgroup=lambda s,yy=y:s['year']==yy,
                                              bootstrap_seed=bootstrap_seed,n_boot=n_boot)
                                 for k,vals in bymetric[m].items() if k in ('net_energy_kwh','gross_energy_kwh')}
                    for y in plan['years']}
        # Weather class is NOT used as sampling stratum. Group counts are sample diagnostics,
        # not unbiased estimates of unknown all-manifest weather subgroup populations.
        cls=defaultdict(list)
        for r in by_method[m]:
            cls[r['weather_condition']].append(r)
        regimes_by_group=defaultdict(list)
        for r in by_method[m]:
            regimes_by_group[r['irradiance_regime']].append(r)
        regimes[m]={condition:{'sampled_episode_count':len(subset),
                        'unweighted_mean_net_kwh':_safe_mean([r['net_energy_kwh'] for r in subset]),
                        'unweighted_mean_gross_kwh':_safe_mean([r['gross_energy_kwh'] for r in subset]),
                        'unweighted_mean_motion_deg':_safe_mean([r['movement_total_deg'] for r in subset])}
                    for condition,subset in sorted(regimes_by_group.items())}
        weather[m]={condition: {'sampled_episode_count':len(subset),
                       'unweighted_mean_net_kwh':_safe_mean([r['net_energy_kwh'] for r in subset]),
                       'unweighted_mean_gross_kwh':_safe_mean([r['gross_energy_kwh'] for r in subset]),
                       'unweighted_mean_movement_deg':_safe_mean([r['movement_total_deg'] for r in subset])}
                    for condition,subset in sorted(cls.items())}
    return {
        'protocol':plan['protocol'], 'selected_scenarios':plan['selected_count'],
        'frame_scenarios':plan['population_count'],'sampling_fraction':plan['selected_count']/plan['population_count'],
        'missing_year_months':plan.get('missing_year_months', []),
        'horizon':_time_grid_note(plan),
        'weights':'Each chosen day represents the number of eligible manifest days in its year/month temporal bin.',
        'interval_warning':'Year-cluster bootstrap descriptive interval (only limited independent years); '
                          'does not fully capture within-bin selection variance or adjacent-date autocorrelation.',
        'training_overlap_warning':('Observed training-day IDs were excluded according to logged training CSV(s), '
                    'but this is not guaranteed unseen if training logs are incomplete or checkpoints differ.'
                    if plan.get('excluded_scenario_ids') else
                    'Sampling from the training manifest is NOT an independent held-out test.'),
        'observed_training_days_excluded':len(plan.get('excluded_scenario_ids', [])),
        'weather_warning':'Weather-condition groups are post hoc observed subsets; no guaranteed population class balance.',
        'motor_warning':'If movement_penalty=0, reported net=gross; physical motor electricity is not measured.',
        'methods':all_stats,'paired_net_differences':comparisons,'gross_only_ideal_gaps':gross_to_ideals,
        'monthly':monthly,'seasonal':seasonal,'yearly':yearly,'weather_conditions':weather,
        'irradiance_regimes':regimes,
        'irradiance_note':(
            f'Measured ratio = sum(DHI)/sum(GHI) over the {expected_steps} '
            'five-minute transitions; heuristic diffuse categories, not certified cloud types.'),
    }


def _clean(o):
    if isinstance(o,dict):return {k:_clean(v) for k,v in o.items()}
    if isinstance(o,list):return [_clean(v) for v in o]
    if isinstance(o,(float,np.floating)) and not np.isfinite(o):return None
    return o


def write_episode_features(paths,plan,outdir):
    rows=make_episode_features(paths,plan)
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    out=outdir/'episode_features.csv'
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    return rows


def load_episode_features(outdir):
    with (Path(outdir)/'episode_features.csv').open(newline='',encoding='utf-8') as f:
        out=[]
        for row in csv.DictReader(f):
            row={k:float(v) if k in FLOATS and v not in ('',None) else None if k in FLOATS
                 else int(v) if k in ('year','month','seed','n_steps') else v
                 for k,v in row.items()}
            out.append(row)
    return out



def monthly_paired_weighted_gains(rows, plan, left, right):
    """Design-weighted per-month paired net gains, averaging seeds within day.

    The population weight is attached to the sampled DATE, not a time step or
    policy seed. A missing eligible date makes that month unavailable.
    """
    a=_day_aggregate([r for r in rows if r['method']==left],'net_energy_kwh')
    b=_day_aggregate([r for r in rows if r['method']==right],'net_energy_kwh')
    paired={sid:a[sid]-b[sid] for sid in a.keys() & b.keys()}
    out={}
    for month in range(1,13):
        item=_weighted(paired,plan,subgroup=lambda s,mo=month:s['month']==mo,
                       n_boot=1)
        out[month]=item['weighted_mean'] if item is not None else None
    return out


def generate_figures(stats,rows,plan,outdir):
    """Read-only graphics on saved episode features; no physical simulations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    figures=[]
    methods=[m for m in ('mbpo','sac','dual_axis_slew_limited','fixed_no_motion') if m in stats['methods']]
    if not methods: raise ValueError('No feasible policy results to plot')
    months=np.arange(1,13)
    with PdfPages(outdir/'sampling_statistics.pdf') as pdf:
        def save(fig,name):
            fig.tight_layout()
            fig.savefig(outdir/name,dpi=165)
            pdf.savefig(fig)
            plt.close(fig)
            figures.append(name)
        ys=plan['years'];heat=np.zeros((len(ys),12),float)
        for s in plan['strata']:
            heat[ys.index(s['year']),s['month']-1]+=1
        fig,ax=plt.subplots(figsize=(11,3.8))
        im=ax.imshow(heat,cmap='Blues',aspect='auto',vmin=0)
        ax.set_xticks(range(12),[str(m) for m in months]);ax.set_yticks(range(len(ys)),ys)
        ax.set(xlabel='Calendar month',ylabel='Source year',title='NSRDB selected days per year and month')
        for y in range(len(ys)):
            for x in range(12):
                if heat[y,x]:ax.text(x,y,str(int(heat[y,x])),ha='center',va='center',fontsize=8)
        fig.colorbar(im,ax=ax,label='Selected dates')
        save(fig,'01_sampling_coverage.png')
        fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
        for m in methods:
            for ax,k,label in [(axes[0],'net_energy_kwh','Net'),(axes[1],'movement_total_deg','Actual applied angular motion')]:
                y=[(stats['monthly'][m][str(x)][k] or {}).get('weighted_mean',np.nan) for x in months]
                ax.plot(months,y,marker='o',label=m)
        axes[0].set(ylabel='Mean net energy (kWh / control window)',title='Calendar-month controller comparison (population-weighted)')
        axes[1].set(ylabel='Mean abs. angular movement (° / window)',xlabel='Calendar month')
        axes[1].set_xticks(months)
        for ax in axes:ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.2)
        save(fig,'02_monthly_energy_and_movement.png')
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        season_order=['winter','spring','summer','fall']
        for m in methods:
            for ax,k,title in [(axes[0],'net_energy_kwh','Seasonal net energy'),
                               (axes[1],'gross_energy_kwh','Seasonal gross energy')]:
                ax.plot(season_order,[(stats['seasonal'][m][s][k] or {}).get('weighted_mean',np.nan)
                                      for s in season_order],marker='o',label=m)
                ax.set(title=title,ylabel='kWh / control window');ax.grid(alpha=.2)
        for ax in axes:ax.legend(fontsize=8)
        save(fig,'03_seasonal_energy.png')
        fig,ax=plt.subplots(figsize=(11,5))
        pair_names=[p for p in stats['paired_net_differences'] if p.startswith('mbpo_minus_') or p.startswith('sac_minus_')]
        for name in pair_names:
            left,right=name.split('_minus_')
            monthly_gain=monthly_paired_weighted_gains(rows,plan,left,right)
            ax.plot(months,[monthly_gain[mo] if monthly_gain[mo] is not None
                            else float('nan') for mo in months],
                    marker='.',label=name.replace('_',' '))
        ax.axhline(0,ls='--',color='black',lw=1)
        ax.set(xlabel='Month',ylabel='Paired net gain (kWh / window)',title='Paired policy differences (population-weighted month means)')
        ax.set_xticks(months);ax.grid(alpha=.2)
        if pair_names:ax.legend(fontsize=7,ncol=2)
        save(fig,'04_monthly_paired_gains.png')
        fig,ax=plt.subplots(figsize=(10,5))
        for m in methods:
            rr=[r for r in rows if r['method']==m]
            ax.scatter([r['movement_total_deg'] for r in rr],[r['net_energy_kwh'] for r in rr],
                       s=15,alpha=.45,label=m)
        ax.set(xlabel='Total absolute tilt + azimuth travel (° / window)',
               ylabel='Net energy (kWh / window)',title='Tracking motion versus harvested energy')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        save(fig,'05_movement_energy_scatter.png')
        known=sorted({r['weather_condition'] for r in rows if r['weather_condition']!='unavailable'})
        if known:
            fig,ax=plt.subplots(figsize=(max(8,len(known)*2),4.8))
            ticks=np.arange(len(known));width=.75/len(methods)
            for i,m in enumerate(methods):
                vals=[(stats['weather_conditions'][m].get(c) or {}).get('unweighted_mean_net_kwh')
                      for c in known]
                ax.bar(ticks+(i-(len(methods)-1)/2)*width,vals,width,label=m)
            ax.set_xticks(ticks,known)
            ax.set(ylabel='Sample mean net (kWh / window)',title='Post-hoc weather condition comparisons (not population-stratified)')
            ax.legend(ncol=2,fontsize=8);ax.grid(axis='y',alpha=.2)
            save(fig,'06_weather_conditions.png')
        regimes=[r for r in ('low diffuse (<0.30)','mixed diffuse (0.30–0.60)',
                             'high diffuse (>=0.60)') if any(r in stats['irradiance_regimes'][m] for m in methods)]
        if regimes:
            fig,ax=plt.subplots(figsize=(11,5))
            ticks=np.arange(len(regimes));width=.75/len(methods)
            for i,m in enumerate(methods):
                vals=[(stats['irradiance_regimes'][m].get(c) or {}).get('unweighted_mean_net_kwh')
                      for c in regimes]
                ax.bar(ticks+(i-(len(methods)-1)/2)*width,vals,width,label=m)
            ax.set_xticks(ticks,regimes)
            ax.set(ylabel='Sample mean net (kWh / window)',
                   title='Measured irradiance diffuse fraction (DHI/GHI; descriptive sample subsets)')
            ax.legend(ncol=2,fontsize=8);ax.grid(axis='y',alpha=.2)
            save(fig,'07_irradiance_regimes.png')
    return figures


def render_statistical_report(outdir,plan=None,paths=None,n_boot=1000,bootstrap_seed=123,plots=True):
    """Supports initial evaluation and repeatable offline regeneration."""
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    if plan is None:
        plan=json.loads((outdir.parent/'scenario_selection.json').read_text(encoding='utf-8'))
    rows=write_episode_features(paths,plan,outdir) if paths is not None else load_episode_features(outdir)
    stats=analyse_features(rows,plan,bootstrap_seed=bootstrap_seed,n_boot=n_boot)
    files=generate_figures(stats,rows,plan,outdir) if plots else []
    stats['figures']=files
    (outdir/'stratified_statistics.json').write_text(json.dumps(_clean(stats),indent=2,allow_nan=False),encoding='utf-8')
    return stats
