"""Strict descriptive scientific figures; no inferred CI from one day per bin."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from SAC_workspace.src.evaluation.statistical_inference import build_valid_statistical_summary, day_summary_from_paths

VISIBLE=('mbpo','sac','fixed_no_motion','poa_instant_ideal')
LABELS={'mbpo':'MBPO','sac':'SAC','fixed_no_motion':'Fixed','poa_instant_ideal':'Instant POA oracle (gross)'}

def render_verified(paths,plan,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    methods={p['method'] for p in paths}
    if not methods<=set(VISIBLE) or 'poa_instant_ideal' not in methods:
        raise ValueError('Only SAC/MBPO/fixed/POA oracle may enter the configured scientific figures')
    summary=build_valid_statistical_summary(paths,plan)
    (outdir/'statistical_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    figfiles=[]
    with PdfPages(outdir/'verified_figures.pdf') as pdf:
        def save(fig,name):
            fig.tight_layout();fig.savefig(outdir/name,dpi=160);pdf.savefig(fig);plt.close(fig);figfiles.append(name)
        names=[m for m in VISIBLE if m in methods]
        ys=[];sds=[]
        for m in names:
            rec=summary['methods'][m];ys.append(rec['weighted_mean']);sds.append(rec['sd_across_days'])
        fig,ax=plt.subplots(figsize=(10,5));xs=np.arange(len(names))
        ax.bar(xs,ys,alpha=.75)
        ax.set_xticks(xs,[LABELS[x] for x in names]);ax.set(ylabel='Mean energy (kWh / 117-step window)',title='Finite-manifest weighted mean; POA oracle gross-only')
        ax.text(.01,.98,'No confidence bars: one sampled date per stratum cannot estimate design variance.',transform=ax.transAxes,va='top',fontsize=8)
        ax.grid(axis='y',alpha=.2);save(fig,'01_weighted_mean_energy.png')
        # Distribution plots MUST compare one physical metric at a time.
        # Boxplot means are unweighted descriptive means, never population CIs.
        for metric,methods,filename,heading in (
            ('gross_energy_kwh',names,'02_day_distribution_boxplot.png',
             'Observed sampled-day GROSS PV energy (unweighted descriptive boxplots)'),
            ('net_energy_kwh',[m for m in names if m!='poa_instant_ideal'],
             '05_feasible_net_energy_boxplot.png',
             'Observed sampled-day feasible NET energy (no POA oracle)')):
            fig,ax=plt.subplots(figsize=(9,5))
            for index,m in enumerate(methods):
                values=day_summary_from_paths(paths,m,metric)
                ax.boxplot(list(values.values()),positions=[index],widths=.5,showmeans=True)
            ax.set_xticks(range(len(methods)),[LABELS[x] for x in methods])
            ax.set(ylabel='kWh / 117-step window',title=heading)
            ax.grid(axis='y',alpha=.2);save(fig,filename)
        fig,ax=plt.subplots(figsize=(9,5));pairs=[('mbpo','sac'),('mbpo','fixed_no_motion'),('sac','fixed_no_motion')]
        for j,(a,b) in enumerate(pairs):
            if a not in names or b not in names:continue
            d={sid:x-day_summary_from_paths(paths,b,'net_energy_kwh')[sid] for sid,x in day_summary_from_paths(paths,a,'net_energy_kwh').items()}
            ax.scatter(np.full(len(d),j),list(d.values()),s=8,alpha=.25)
            ax.plot([j-.2,j+.2],[summary['paired_gaps'][a+'_minus_'+b]['weighted_mean_difference_kwh']]*2,linewidth=2)
        ax.axhline(0,color='black',ls='--',lw=1)
        ax.set_xticks(range(len(pairs)),[a+' − '+b for a,b in pairs]);ax.set(ylabel='Matched-day net difference (kWh)',title='Paired daily contrasts; horizontal mark = design-weighted mean')
        ax.grid(axis='y',alpha=.2);save(fig,'03_paired_day_differences.png')
        fig,ax=plt.subplots(figsize=(10,5))
        months=range(1,13)
        for m in names:
            metric='gross_energy_kwh' if m=='poa_instant_ideal' else 'net_energy_kwh'
            v=day_summary_from_paths(paths,m,metric)
            vals=[]
            for month in months:
                strata=[s for s in plan['strata'] if s['month']==month]
                w=np.array([s['population_days'] for s in strata],float)
                y=np.array([v[s['selected_scenario_id']] for s in strata])
                vals.append(float(np.average(y,weights=w)) if len(y) else np.nan)
            ax.plot(months,vals,marker='o',label=LABELS[m])
        ax.set_xticks(list(months));ax.set(xlabel='Month',ylabel='Weighted mean energy (kWh / window)',title='Month-stratified energy: observed dates only, no unsupported CI')
        ax.legend();ax.grid(alpha=.2);save(fig,'04_monthly_weighted_energy.png')
    return {'figures':figfiles,'statistics':summary}
