"""Publication-ready figures; no imports from TensorFlow/MBPO training."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from SAC_workspace.src.evaluation.evaluation import METHODS
from SAC_workspace.src.evaluation.time_axis import unwrapped_utc_hours

LABELS = {'sac':'Standalone SAC', 'dual_axis_slew_limited':'Dual-axis (slew-limited)',
          'fixed_no_motion':'Fixed (no motion)', 'dual_axis_instant_ideal':'Dual-axis (instantaneous)'}



def plot_results(paths, outdir, training_csv=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    method_paths = {m: [p for p in paths if p['method']==m] for m in METHODS}
    fig, ax = plt.subplots(figsize=(9.2, 5))
    labels = [LABELS[m] for m in METHODS]
    values = [np.mean([p['gross_energy_kwh'] for p in method_paths[m]]) for m in METHODS]
    ax.bar(labels, values)
    ax.set_ylabel('Mean generated energy (kWh / episode)')
    ax.set_title('Identical NSRDB scenarios — gross energy (ideal is actuator-unconstrained)')
    ax.tick_params(axis='x', labelrotation=15)
    ax.grid(axis='y', alpha=.22)
    fig.tight_layout()
    fig.savefig(outdir/'gross_energy_comparison.png', dpi=220)
    plt.close(fig)

    feasible_methods = tuple(m for m in METHODS if m != 'dual_axis_instant_ideal')
    vals = [np.mean([p['net_energy_kwh'] for p in method_paths[m]]) for m in feasible_methods]
    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.bar([LABELS[m] for m in feasible_methods], vals)
    ax.set_ylabel('Mean net reward (kWh / episode)')
    ax.set_title('Feasible controllers only — identical net-energy reward definition')
    ax.tick_params(axis='x', labelrotation=15)
    ax.grid(axis='y', alpha=.22)
    fig.tight_layout()
    fig.savefig(outdir/'net_energy_comparison.png', dpi=220)
    plt.close(fig)


    # Preserve paired scenario variability, rather than showing only mean bars.
    fig, ax = plt.subplots(figsize=(10, 5))
    fixed = {(p['scenario_id'], p['seed']):p for p in method_paths['sac']}
    for j, method in enumerate(('dual_axis_slew_limited','fixed_no_motion')):
        paired = [(p, fixed[(p['scenario_id'],p['seed'])]) for p in method_paths[method]]
        deltas = np.array([learned['net_energy_kwh']-ref['net_energy_kwh'] for ref, learned in paired])
        x = np.full(len(deltas), j, dtype=float)
        ax.scatter(x, deltas, alpha=.55, s=26, label=LABELS[method])
        ax.plot([j-.15,j+.15],[float(np.mean(deltas))]*2,linewidth=3)
    ax.axhline(0.,linestyle='--',linewidth=1)
    ax.set_xticks([0,1]);ax.set_xticklabels(['SAC minus dual axis','SAC minus fixed mount'])
    ax.set_ylabel('Paired net-energy difference (kWh / episode)')
    ax.set_title('Same NSRDB scenarios — each dot is one scenario/seed pair')
    ax.grid(axis='y',alpha=.25)
    fig.tight_layout()
    fig.savefig(outdir/'paired_net_energy_differences.png',dpi=220)
    plt.close(fig)

    first_scenario = paths[0]['scenario_id']
    first_seed = paths[0]['seed']
    same = [p for p in paths if p['scenario_id']==first_scenario and p['seed']==first_seed]
    # Draw the actual recorded post-step UTC clock, unwrapped across midnight.
    time = unwrapped_utc_hours(same[0]['infos'])
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for p in same:
        name = LABELS[p['method']]
        axes[0].plot(time, p['power'], label=name)
        axes[1].plot(time, np.cumsum(p['energy']), label=name)
        axes[2].plot(time, p['tilt'], label=name)
        axes[3].plot(time, p['azimuth'], label=name)
    for ax, ylab in zip(axes, ('Power (W)','Cumulative gross energy (kWh)','Panel tilt (degrees)','Panel azimuth (degrees)')):
        ax.set_ylabel(ylab)
        ax.grid(alpha=.23)
    axes[0].legend(loc='best', fontsize=8)
    axes[-1].set_xlabel('Unwrapped UTC hour (24+ means next UTC date)')
    fig.suptitle(f'Paired trajectory — {first_scenario}, seed={first_seed}')
    fig.tight_layout()
    fig.savefig(outdir/'matched_trajectory.png', dpi=220)
    plt.close(fig)

    # Seasonal mean net energy from paired scenarios; skip missing ideal net cost.
    season_groups = {}
    for method in METHODS:
        for p in method_paths[method]:
            if p['net_energy_kwh'] is None: continue
            season = p['infos'][0].get('season') or 'unknown'
            season_groups.setdefault(season, {}).setdefault(method, []).append(p['net_energy_kwh'])
    seasons = [season for season in ('winter','spring','summer','fall','unknown') if season in season_groups]
    if seasons:
        fig, ax = plt.subplots(figsize=(9.8,5))
        x = np.arange(len(seasons)); width = .18
        for i, method in enumerate(feasible_methods):
            y = [np.mean(season_groups.get(season,{}).get(method, [float('nan')])) for season in seasons]
            ax.bar(x+(i-1)*width, y, width, label=LABELS[method])
        ax.set_xticks(x); ax.set_xticklabels(seasons)
        ax.set_ylabel('Mean net reward (kWh / episode)')
        ax.set_title('Paired NSRDB scenario energy by season (not an unseen-data claim)')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.22)
        fig.tight_layout()
        fig.savefig(outdir/'seasonal_net_energy.png',dpi=220)
        plt.close(fig)

    if training_csv:
        with Path(training_csv).open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if rows:
            steps = np.array([int(r['env_steps']) for r in rows])
            net = np.array([float(r['return_kwh']) for r in rows])
            fig, ax = plt.subplots(figsize=(9.2,4.8))
            ax.plot(steps, net, alpha=.25, label='Episode net energy')
            window = min(30, len(net))
            if window > 1:
                x = np.convolve(net, np.ones(window)/window, mode='valid')
                ax.plot(steps[window-1:], x, label=f'{window}-episode moving average')
            ax.set_xlabel('Real simulator transitions')
            ax.set_ylabel('Training net reward (kWh / episode)')
            ax.set_title('Standalone SAC training progression')
            ax.legend()
            ax.grid(alpha=.23)
            fig.tight_layout()
            fig.savefig(outdir/'training_learning_curve.png', dpi=220)
            plt.close(fig)
            # Diagnostics after warm-up; the pretraining section has NaN losses.
            panels = [('critic_loss','Twin critic loss'),('actor_loss','Actor loss'),
                      ('entropy','Stochastic policy entropy'),('alpha','Entropy temperature')]
            fig, axes = plt.subplots(2,2,figsize=(11,7))
            for ax,(key,title) in zip(axes.flat,panels):
                values=np.array([float(r[key]) for r in rows], dtype=float)
                mask=np.isfinite(values)
                ax.plot(steps[mask],values[mask])
                ax.set_title(title)
                ax.set_xlabel('Real simulator transitions')
                ax.grid(alpha=.22)
            fig.tight_layout()
            fig.savefig(outdir/'training_diagnostics.png',dpi=220)
            plt.close(fig)

    plot_scientific_scenario_panels(paths, outdir)


def plot_scientific_scenario_panels(paths, outdir):
    """Scenario-matched plots using recorded pvlib/NSRDB trajectories only.

    This code performs no new simulations and imports no model-based RL code.
    Paired uncertainty always resamples entire scenario IDs (not time steps).
    """
    from SAC_workspace.src.evaluation.evaluation import _ci_clustered
    from datetime import datetime, timedelta
    outdir = Path(outdir)
    grouped = {}
    for p in paths:
        grouped.setdefault((p['scenario_id'], p['seed']), {})[p['method']] = p
    methods = ('dual_axis_slew_limited', 'fixed_no_motion')
    colors = ('C1', 'C2')

    # Scatter is paired, not an unpaired comparison of changing weather days.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    for ax, method, color in zip(axes, methods, colors):
        reference, learned = [], []
        for g in grouped.values():
            reference.append(g[method]['net_energy_kwh'])
            learned.append(g['sac']['net_energy_kwh'])
        reference = np.asarray(reference, dtype=float)
        learned = np.asarray(learned, dtype=float)
        low = min(float(np.min(reference)), float(np.min(learned)), 0.)
        high = max(float(np.max(reference)), float(np.max(learned)))
        pad = max((high-low)*.035, .005)
        ax.scatter(reference, learned, s=14, alpha=min(.6, max(.04, 65./len(reference))), color=color)
        ax.plot([low-pad,high+pad],[low-pad,high+pad],linestyle='--',color='gray',label='Equal net energy')
        fraction = float(np.mean(learned>reference))
        delta = float(np.mean(learned-reference))
        ax.set(title=f'SAC vs {LABELS[method]}\nMean difference {delta:+.4f} kWh; better on {fraction:.1%} of pairs',
               xlabel=f'{LABELS[method]} net energy (kWh)', ylabel='SAC net energy (kWh)',
               xlim=(low-pad,high+pad), ylim=(low-pad,high+pad))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.20)
    fig.tight_layout()
    fig.savefig(outdir/'paired_scenario_net_energy_scatter.png', dpi=220)
    plt.close(fig)

    def draw_stratified_gain(kind, outfile):
        # Day groups, rather than individual transitions, are independent samples.
        labels = ('winter','spring','summer','fall') if kind=='season' else tuple(range(1,13))
        fig, ax = plt.subplots(figsize=(11, 5.2))
        for offset, method, color in zip((-.12,.12), methods, colors):
            xx, yy, lo_err, hi_err = [], [], [], []
            for i, label in enumerate(labels):
                day_groups = {}
                for (sid, seed), g in grouped.items():
                    info = g['sac']['infos'][0]
                    value = (info['season'] if kind=='season' else int(info['date'][5:7]))
                    if value == label:
                        day_groups.setdefault(sid, []).append(
                            g['sac']['net_energy_kwh'] - g[method]['net_energy_kwh'])
                if not day_groups:
                    continue
                mean, low, high = _ci_clustered(day_groups, n_boot=1000)
                xx.append(i+offset); yy.append(mean)
                lo_err.append(mean-low if np.isfinite(low) else 0.)
                hi_err.append(high-mean if np.isfinite(high) else 0.)
            if xx:
                ax.errorbar(xx, yy, yerr=np.array([lo_err,hi_err]),
                            fmt='o-',capsize=3,label=f'SAC − {LABELS[method]}',color=color)
        ax.axhline(0.,color='gray',linestyle='--')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels if kind=='season' else ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))
        ax.set_ylabel('Paired mean difference in net energy (kWh/day)')
        ax.set_title(f'SAC energy gain by {kind}: scenario-clustered 95% bootstrap intervals\nIntervals omitted when only one scenario is available')
        ax.legend(fontsize=9);ax.grid(axis='y',alpha=.25)
        fig.tight_layout()
        fig.savefig(outdir/outfile,dpi=220)
        plt.close(fig)

    draw_stratified_gain('season','seasonal_paired_net_gain.png')
    draw_stratified_gain('month','monthly_paired_net_gain.png')

    # At most one deterministic representative from each available season,
    # rather than treating the first chronological scenario as representative.
    representative = {}
    for (sid, seed), g in grouped.items():
        season = g['sac']['infos'][0]['season']
        if season not in representative:
            representative[season] = (sid, seed, g)
    scenario_dir = outdir/'representative_days'
    scenario_dir.mkdir(parents=True,exist_ok=True)
    for season, (sid, seed, g) in sorted(representative.items()):
        fig, axes = plt.subplots(4,1,sharex=True,figsize=(11,10))
        for method in METHODS:
            p = g[method]
            time = unwrapped_utc_hours(p['infos'])
            axes[0].plot(time,p['power'],label=LABELS[method])
            axes[1].plot(time,np.cumsum(p['energy']),label=LABELS[method])
            axes[2].plot(time,p['tilt'],label=LABELS[method])
            axes[3].plot(time,p['azimuth'],label=LABELS[method])
        for ax, label in zip(axes,('Power (W)','Cumulative gross energy (kWh)',
                                    'Tilt (degrees)','Azimuth (degrees)')):
            ax.set_ylabel(label);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel('Unwrapped UTC hour (24+ means next UTC date)')
        fig.suptitle(f'{season.title()} day — {sid} / seed {seed}; same NSRDB irradiance')
        fig.tight_layout()
        fig.savefig(scenario_dir/f'{season}_{sid}.png',dpi=220)
        plt.close(fig)
