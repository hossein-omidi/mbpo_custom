"""Explicit descriptive dispersion of paired SAC/MBPO validation trajectories.

Inputs are arrays [independent training seed, held-out historical scenario, real
interaction checkpoint]. All cells must be finite and paired across algorithms.

The total mean equals the equally weighted cell mean, but two different
sources of dispersion must remain distinct:
  * SD(seed-mean): variation between independently trained policies, after
    averaging each seed over the SAME frozen validation days.
  * SD(day-mean): variation between weather days, after averaging trained seeds.
The plotted ±1 SD bands are descriptive, not bootstrap CI or an estimate of
MC/weather-measurement uncertainty within the frozen validation day.
"""
from __future__ import annotations
import numpy as np


def summarize(matrix: np.ndarray) -> dict[str, np.ndarray]:
    x=np.asarray(matrix,dtype=float)
    if x.ndim!=3 or not all(n>0 for n in x.shape) or not np.isfinite(x).all():
        raise ValueError('Expected finite [seed, scenario_day, checkpoint] array')
    seeds,days,_=x.shape
    if seeds<2 or days<2:
        raise ValueError('At least 2 independently trained seeds and 2 historical days required for both descriptive SDs')
    seed_means=x.mean(axis=1)
    day_means=x.mean(axis=0)
    overall=seed_means.mean(axis=0)
    if not np.allclose(overall,day_means.mean(axis=0),rtol=1e-12,atol=1e-12):
        raise ArithmeticError('Mean over seeds/days is inconsistent')
    return {
        'mean':overall,
        'sd_across_training_seed_means':seed_means.std(axis=0,ddof=1),
        'sd_across_historical_day_means':day_means.std(axis=0,ddof=1),
        'n_training_seeds':seeds,'n_historical_days':days,
    }


def aggregate_mc_power_profiles(clock_rows: list[dict], draws: int) -> list[dict]:
    """Combine day-level time-resolved MC means and sample SDs exactly.

    Each historical day and each MC weather draw has equal weight. Every
    day-level SD is sample SD (ddof=1) across the M independent draws.
    The empirical population variance of all K*M *actual simulated powers*
    is: Var_d(mean_d) + ((M-1)/M)*mean_d(SD_d**2).
    This is a DESCRIPTIVE mixture variance, not a confidence interval.
    """
    from collections import defaultdict
    draws=int(draws)
    if draws<2:raise ValueError('At least two MC realizations per day needed for sample SD')
    groups=defaultdict(list)
    per_day_seen=set()
    for r in clock_rows:
        method=str(r['method']);sid=str(r['scenario_id']);step=int(r['step'])
        if step<1 or step>195:raise ValueError('Invalid 5-minute episode step')
        key=(method,sid,step)
        if key in per_day_seen:raise ValueError(f'Duplicate day/method/step {key}')
        per_day_seen.add(key)
        mu=float(r['power_mean_w']);sd=float(r['power_sd_w'])
        if not (np.isfinite(mu) and np.isfinite(sd)) or sd<0:
            raise ValueError('Nonfinite/negative per-day power mean or SD')
        groups[(method,step)].append((sid,mu,sd))
    methods=sorted({m for m,_ in groups})
    step_set=set(range(1,196))
    for m in methods:
        if {t for mm,t in groups if mm==m}!=step_set:
            raise ValueError(f'Incomplete 195-step power curve for {m}')
    day_set={sid for (m,sid,step) in per_day_seen}
    if not day_set:raise ValueError('No historical days')
    for m in methods:
        for t in step_set:
            if {sid for sid,_,_ in groups[(m,t)]}!=day_set:
                raise ValueError('Methods or time steps have different historical-day coverage')
    result=[]
    for m in methods:
        for step in range(1,196):
            rows=sorted(groups[(m,step)])
            day_mu=np.asarray([x[1] for x in rows],dtype=float)
            day_sd=np.asarray([x[2] for x in rows],dtype=float)
            mu=float(day_mu.mean())
            between=float(np.mean((day_mu-mu)**2))
            within=float((draws-1.)/draws * np.mean(day_sd**2))
            total=float(between+within)
            if total<0:raise ArithmeticError('Negative physical power variance')
            result.append({
                'method':m, 'step':step,
                'utc_hour_unwrapped':round(8.75+step/12,8),
                'n_historical_days':len(day_set),'draws_per_day':draws,
                'mean_power_w':mu,
                'sd_total_descriptive_w':float(np.sqrt(total)),
                'sd_between_historical_days_w':float(np.sqrt(between)),
                'sd_within_weather_draws_w':float(np.sqrt(within)),
                'variance_check_total_w2':total,
            })
    return result


def write_sample_complexity_mean_sd(outdir, steps, sac_matrix, mbpo_matrix, target):
    """Export genuinely matched real-step mean ± descriptive SD curves.

    The official two-way bootstrap CI figures are deliberately preserved by
    the caller. This function adds, but never relabels, 1 SD figures.
    """
    import csv
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    steps=np.asarray(steps,dtype=int)
    sac=np.asarray(sac_matrix,dtype=float);mbpo=np.asarray(mbpo_matrix,dtype=float)
    if sac.shape!=mbpo.shape or sac.shape[2]!=len(steps):
        raise ValueError('SAC/MBPO must use matched seed/day/checkpoint grids')
    if len(steps)<2 or np.any(np.diff(steps)<=0):
        raise ValueError('Real-environment interaction checkpoints must be strictly increasing')
    desc={'sac':summarize(sac),'mbpo':summarize(mbpo),
          'mbpo_minus_sac':summarize(mbpo-sac)}
    result_rows=[];difference_rows=[]
    for name in ('sac','mbpo'):
        d=desc[name]
        for i,step in enumerate(steps):
            result_rows.append({'algorithm':name,'env_steps':int(step),
                'mean_net_energy_kwh':float(d['mean'][i]),
                'sd_across_training_seed_means_kwh':float(d['sd_across_training_seed_means'][i]),
                'sd_across_historical_day_means_kwh':float(d['sd_across_historical_day_means'][i]),
                'n_training_seeds':int(d['n_training_seeds']),
                'n_validation_days':int(d['n_historical_days']),
                'weather_realization': 'fixed realization_id=0 per held-out validation day'})
    d=desc['mbpo_minus_sac']
    for i,step in enumerate(steps):
        difference_rows.append({'env_steps':int(step),'mean_difference_net_energy_kwh':float(d['mean'][i]),
            'sd_across_paired_training_seed_means_kwh':float(d['sd_across_training_seed_means'][i]),
            'sd_across_paired_historical_day_means_kwh':float(d['sd_across_historical_day_means'][i])})
    def write(name,rows):
        p=out/name
        with p.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return name
    files=[write('learning_curve_descriptive_sd.csv',result_rows),
           write('paired_difference_descriptive_sd.csv',difference_rows)]
    target=float(target)
    if not np.isfinite(target):raise ValueError('Nonfinite predefined energy target')
    images=[]
    for mode,sd_key,title,filename in (
        ('seeds','sd_across_training_seed_means',
         'Mean ± 1 SD across independently trained seed means (fixed validation days)',
         'learning_curve_mean_pm_seed_sd.png'),
        ('days','sd_across_historical_day_means',
         'Mean ± 1 SD across historical day means (averaged over training seeds)',
         'learning_curve_mean_pm_day_sd.png'),
    ):
        fig,ax=plt.subplots(figsize=(9,5.5))
        for name in ('sac','mbpo'):
            d=desc[name];mean=d['mean'];std=d[sd_key]
            line,=ax.plot(steps,mean,label=name.upper(),linewidth=2)
            ax.fill_between(steps,mean-std,mean+std,alpha=.19,color=line.get_color(),
                            label=name.upper()+' descriptive ±1 SD')
        ax.axhline(target,linestyle='--',linewidth=1.1,color='gray',label='Predefined target')
        ax.set(xlabel='Real environment transitions (not synthetic MBPO rollouts)',
               ylabel='Mean validation net energy (kWh / episode)',title=title)
        ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
        fig.savefig(out/filename,dpi=180);plt.close(fig);images.append(filename)
    d=desc['mbpo_minus_sac']
    fig,ax=plt.subplots(figsize=(9,5))
    mean=d['mean'];std=d['sd_across_training_seed_means']
    line,=ax.plot(steps,mean,label='Paired MBPO − SAC mean',linewidth=2)
    ax.fill_between(steps,mean-std,mean+std,alpha=.20,color=line.get_color(),
                    label='Descriptive ±1 SD across paired training seeds')
    ax.axhline(0,color='gray',linestyle='--',linewidth=1)
    ax.set(xlabel='Real environment transitions',ylabel='Paired net energy difference (kWh / episode)',
           title='Paired MBPO − SAC: mean ± 1 SD across training seeds')
    ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
    filename='paired_difference_mean_pm_seed_sd.png';fig.savefig(out/filename,dpi=180);plt.close(fig)
    images.append(filename)
    return {
        'data_files':files,'figure_files':images,
        'seed_sd_definition':'sample SD (ddof=1) across independently trained seed means, each averaged over the same fixed validation days',
        'day_sd_definition':'sample SD (ddof=1) across historical-day means, each averaged over the same set of trained seeds',
        'paired_sd_definition':'sample SD (ddof=1) across seed means of MBPO minus SAC, pairing identical seed and historical day',
        'not_included':'MC weather-draw uncertainty within a validation day; periodic validation freezes realization_id=0',
        'interpretation':'Shaded regions are descriptive ±1 SD, not confidence intervals; separate original bootstrap CI plots remain unchanged',
    }


def plot_pooled_mc_power_profiles(outdir, rows):
    """Plot full held-out historical-day × weather-draw mixture mean ± SD."""
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    arr=list(rows)
    if not arr:raise ValueError('No pooled Monte Carlo data')
    fig,ax=plt.subplots(figsize=(11,5.5))
    for method in sorted({r['method'] for r in arr}):
        records=sorted((r for r in arr if r['method']==method),key=lambda r:r['step'])
        if len(records)!=195:raise ValueError(f'Incomplete pooled MC power curve for {method}')
        x=np.asarray([float(r['utc_hour_unwrapped']) for r in records])
        mu=np.asarray([float(r['mean_power_w']) for r in records])
        sd=np.asarray([float(r['sd_total_descriptive_w']) for r in records])
        line,=ax.plot(x,mu,linewidth=1.8,label=method+' mean')
        ax.fill_between(x,mu-sd,mu+sd,color=line.get_color(),alpha=.14,
                        label=method+' ±1 descriptive SD')
    ax.set(xlabel='Unwrapped post-action UTC hour (25.00 = next-day 01:00)',
           ylabel='PV DC power (W)',
           title='Held-out days × weather draws: mean ± descriptive SD of PV power')
    ax.grid(alpha=.23);ax.legend(ncol=2,fontsize=8);fig.tight_layout()
    filename='mc_power_mean_pm_sd_over_heldout_days.png';fig.savefig(out/filename,dpi=180);plt.close(fig)
    return filename
