"""Modern 2D/3D plotting and shared SAC plotting reuse; legacy scripts remain intact.
No TensorFlow, Ray, or legacy policy imports. Output uses actual saved physical
rollout metadata (timestamps, irradiance-driven power, and kWh).
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from MBPO_workspace.src.evaluation import METHODS,FEASIBLE

LABELS={'mbpo':'Remastered MBPO','sac':'Standalone SAC',
        'fixed_no_motion':'Fixed mount','dual_axis_slew_limited':'Dual axis (rate-limited)',
        'dual_axis_instant_ideal':'Ideal sun pointing (gross only)',
        'poa_instant_ideal':'Instant POA optimum (gross only)'}


def _utc_hours(path):
    from datetime import datetime
    ts=[datetime.fromisoformat(x['timestamp_utc_iso']) for x in path['infos']]
    if any(t.utcoffset().total_seconds()!=0 for t in ts):
        raise ValueError('Expected UTC-normalized NSRDB post-step timestamps')
    return np.array([t.hour+t.minute/60+t.second/3600 for t in ts])


def _history(path):
    if not path:return None
    with Path(path).open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    return rows or None


def make_figures(paths,outdir,mbpo_training_csv=None,sac_training_csv=None):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    # Direct reuse of SAC's fully-tested plotting system for its four methods:
    # gross/net reference comparison, aligned trajectories, per-scenario deltas,
    # seasonality, and actor/critic/entropy diagnostics.
    # Keep historical/rich SAC plot modules untouched. Some expect a `date`
    # field on every info record; evaluation fixture and older NSRDB builds
    # expose only scenario_id. Enrich the shared paths non-destructively.
    from SAC_workspace.src.evaluation.stratified_statistics import SEASONS
    for path in paths:
        for info in path.get('infos', []):
            if not info.get('date'):
                info['date'] = str(path['scenario_id'])
            if not info.get('season'):
                info['season'] = SEASONS[int(str(path['scenario_id'])[5:7])]
    from SAC_workspace.src.evaluation.plots import plot_results
    plot_results([p for p in paths if p['method'] in ('sac','fixed_no_motion','dual_axis_slew_limited','dual_axis_instant_ideal')],outdir/'sac_reference',sac_training_csv)
    available=[m for m in METHODS if any(p['method']==m for p in paths)]
    by={(p['scenario_id'],p['seed'],p['method']):p for p in paths}
    pairs=sorted({(p['scenario_id'],p['seed']) for p in paths})
    fig_paths=[]
    pdf=PdfPages(outdir/'comparison_report.pdf')

    def save(fig,name):
        p=outdir/name
        fig.tight_layout();fig.savefig(p,dpi=210)
        pdf.savefig(fig);plt.close(fig)
        fig_paths.append(p)

    # Gross energy can include the unconstrained ideal reference.
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar([LABELS[m] for m in available],
           [np.mean([p['gross_energy_kwh'] for p in paths if p['method']==m]) for m in available])
    ax.set_title('Gross energy — same NSRDB scenarios, same pvlib irradiance model')
    ax.set_ylabel('Mean gross energy (kWh / 117-step episode)')
    ax.tick_params(axis='x',rotation=17);ax.grid(axis='y',alpha=.2)
    save(fig,'01_gross_energy.png')

    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar([LABELS[m] for m in FEASIBLE],
           [np.mean([p['net_energy_kwh'] for p in paths if p['method']==m]) for m in FEASIBLE])
    ax.set_title('Net energy — only feasible controllers with identical reward')
    ax.set_ylabel('Mean net energy (kWh / 117-step episode)')
    ax.tick_params(axis='x',rotation=17);ax.grid(axis='y',alpha=.2)
    save(fig,'02_net_energy.png')

    fig,ax=plt.subplots(figsize=(11,5))
    refs=('sac','fixed_no_motion','dual_axis_slew_limited')
    for j,ref in enumerate(refs):
        dy=[by[(sid,sd,'mbpo')]['net_energy_kwh']-by[(sid,sd,ref)]['net_energy_kwh']
            for sid,sd in pairs]
        ax.scatter(np.full(len(dy),j),dy,alpha=.5,s=16)
        ax.plot([j-.2,j+.2],[float(np.mean(dy))]*2,linewidth=3)
    ax.axhline(0,color='black',linewidth=1,linestyle='--')
    ax.set_xticks(range(len(refs)));ax.set_xticklabels([f'MBPO − {LABELS[x]}' for x in refs])
    ax.set_ylabel('Paired net energy difference (kWh / episode)')
    ax.set_title('Scenario-level differences; do not treat adjacent days as independent weather draws')
    ax.grid(axis='y',alpha=.2)
    save(fig,'03_paired_net_differences.png')

    # Trajectory plots from the same scenario and post-step UTC timestamps.
    sid,seed=pairs[0]
    chosen=[by[(sid,seed,m)] for m in available]
    t=_utc_hours(chosen[0])
    fig,axs=plt.subplots(4,1,figsize=(11,12),sharex=True)
    for p in chosen:
        name=LABELS[p['method']]
        axs[0].plot(t,p['power'],label=name)
        axs[1].plot(t,np.cumsum(p['energy']),label=name)
        axs[2].plot(t,p['tilt'],label=name)
        axs[3].plot(t,p['azimuth'],label=name)
    for ax,ylabel in zip(axs,('Power (W)','Cumulative gross energy (kWh)',
                  'Tilt (degrees)','Azimuth (degrees)')):
        ax.set_ylabel(ylabel);ax.grid(alpha=.2)
    axs[0].legend(fontsize=8,ncol=2)
    axs[-1].set_xlabel('Post-action time (UTC hours)')
    fig.suptitle(f'Matched trajectory — {sid}, seed={seed}')
    save(fig,'04_physical_trajectories.png')

    fig=plt.figure(figsize=(10,7));ax=fig.add_subplot(111,projection='3d')
    for m in ('mbpo','sac','dual_axis_slew_limited'):
        p=by[(sid,seed,m)]
        ax.plot(np.asarray(p['tilt']),np.asarray(p['azimuth']),t,label=LABELS[m])
    ax.set_xlabel('Tilt (degrees)');ax.set_ylabel('Azimuth (degrees)')
    ax.set_zlabel('UTC clock hour')
    ax.set_title('3D solar tracking orientation path (real pvlib simulator)')
    ax.legend()
    save(fig,'05_orientation_3d.png')

    # Calendar grouping; scenario-level heatmap normalized to achievable dual-axis.
    sids=sorted({sid for sid,_ in pairs})
    grid=np.zeros((len(sids),3),float)
    for i,scenario in enumerate(sids):
        subset=[(s,seed) for s,seed in pairs if s==scenario]
        for j,m in enumerate(('mbpo','sac','fixed_no_motion')):
            diff=[by[(s,z,m)]['net_energy_kwh']-by[(s,z,'dual_axis_slew_limited')]['net_energy_kwh']
                  for s,z in subset]
            grid[i,j]=np.mean(diff)
    fig,ax=plt.subplots(figsize=(8,max(4.0,min(14.,len(sids)*.18+1))))
    lim=max(np.abs(grid).max(),1e-10)
    heat=ax.imshow(grid,cmap='coolwarm',vmin=-lim,vmax=lim,aspect='auto')
    fig.colorbar(heat,ax=ax,label='Net kWh relative to slew-limited dual axis')
    ax.set_xticks(range(3));ax.set_xticklabels(['MBPO','SAC','Fixed'])
    if len(sids)<=55:
        ax.set_yticks(range(len(sids)));ax.set_yticklabels(sids,fontsize=7)
    else:ax.set_ylabel('NSRDB scenario index (sorted)')
    ax.set_title('Paired per-scenario energy differences (not held-out unless disjoint)')
    save(fig,'06_scenario_energy_heatmap.png')

    fig,ax=plt.subplots(figsize=(10,5))
    seasons=('winter','spring','summer','fall')
    for m in FEASIBLE:
        season_means=[]
        for season in seasons:
            vals=[p['net_energy_kwh'] for p in paths if p['method']==m
                  and p['infos'][0].get('season')==season]
            season_means.append(np.mean(vals) if vals else np.nan)
        ax.plot(seasons,season_means,marker='o',label=LABELS[m])
    ax.set_title('Calendar-season net energy, paired NSRDB scenarios')
    ax.set_ylabel('Mean net energy (kWh / episode)')
    ax.legend(ncol=2);ax.grid(alpha=.2)
    save(fig,'07_seasonal_net_energy.png')

    histories={'mbpo':_history(mbpo_training_csv),'sac':_history(sac_training_csv)}
    if any(h is not None for h in histories.values()):
        fig,ax=plt.subplots(figsize=(10,5))
        for m,history in histories.items():
            if not history:continue
            x=np.array([int(r['env_steps']) for r in history]);y=np.array([float(r['return_kwh']) for r in history])
            ax.plot(x,y,alpha=.15)
            w=min(30,len(y))
            if w>1:ax.plot(x[w-1:],np.convolve(y,np.ones(w)/w,'valid'),label=LABELS[m])
        ax.set_xlabel('REAL environment transitions (imaginary rollout steps excluded)')
        ax.set_ylabel('Training episode net energy (kWh)')
        ax.set_title('Learning curves (weather distributions may differ by sampled episode)')
        ax.legend();ax.grid(alpha=.2)
        save(fig,'08_training_learning_curves.png')
    if histories['mbpo']:
        rows=histories['mbpo']
        x=np.array([int(r['env_steps']) for r in rows])
        fig,axs=plt.subplots(2,2,figsize=(11,7))
        for ax,k,title in zip(axs.flat,
                ('model_val_nll','model_val_power_rmse_norm','model_replay_size','critic_loss'),
                ('Held-out ensemble NLL','Held-out power prediction RMSE (normalized)',
                 'Synthetic replay size','SAC critic loss')):
            y=np.array([float(row[k]) for row in rows]);ok=np.isfinite(y)
            ax.plot(x[ok],y[ok]);ax.set_title(title);ax.set_xlabel('Real transitions');ax.grid(alpha=.2)
        save(fig,'09_dynamics_training_diagnostics.png')
    pdf.close()
    return fig_paths
