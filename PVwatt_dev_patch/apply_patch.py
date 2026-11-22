#!/usr/bin/env python3
"""PVwatt_dev repair: standalone local installer; never accesses or writes GitHub.

Stages the prior independent-evaluation/sample-complexity patch and critical
model/physics/replay/report corrections atomically with blob SHA verification.
Run dry-run first; run install only against inspected PVwatt_dev source.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path
import previous_stage as previous

SCRIPT = Path(__file__).resolve().parent
BASE = dict(previous.BASE_BLOBS)
BASE.update({
    'examples/config/pv_tracking/conf_compact.py':'b143822e433c5071773382c72e1df35f863f40f8',
    'mbpo/env/pv_tracking.py':'b03c1016e1f3c1f5ac3429e929d1162485b09931',
    'mbpo/env/pvlib_physics.py':'9a63cf714e5e25008981505c33f812497feb66ef',
    'MBPO_workspace/src/paper_reporting.py':'6988366999e1b484ad3e325c531761f4d1e9da01',
    'MBPO_workspace/src/plots.py':'a829f756184658ef52d3e1bbcebe898b57c36313',
    'MBPO_workspace/src/evaluation.py':'befa3cbe5f9f2acc39f9f62414c39bc98778b732',
    'SAC_workspace/src/evaluation/plots.py':'760e2ef531ff3735856bea145d8497d8dfa0287d',
})
# The attached src archives precede several light plotting-only commits. Exact
# anchors are still required; accept only these two inspected standalone blobs.
ALTERNATE_BASE = {
    'MBPO_workspace/src/paper_reporting.py': set(),
    'MBPO_workspace/src/plots.py': {'dbf5bb2e4327a4860ba5771e391c2a3960990ba7'},
}


def rep(src, old, new, name):
    return previous.replace_once(src,old,new,name)


def patch_compact(s):
    s=rep(s,"    train_env_kwargs={\n        'observation_mode': 'compact',\n    },",
          "    train_env_kwargs={\n        'observation_mode': 'compact',\n        'movement_penalty': 1.0,\n        'movement_cost_mode': 'geometry',\n    },",'positive train movement cost')
    return rep(s,"    eval_env_kwargs={\n        'observation_mode': 'compact',\n        'fixed_eval_scenarios': None,\n        'randomize_scenario': True,\n    },",
          "    eval_env_kwargs={\n        'observation_mode': 'compact',\n        'fixed_eval_scenarios': None,\n        'randomize_scenario': True,\n        'movement_penalty': 1.0,\n        'movement_cost_mode': 'geometry',\n    },",'positive evaluation movement cost')


def patch_physics(s):
    anchor='    tilt_time_s = abs(float(delta_tilt_deg)) / float(slew_rate_tilt_deg_s)\n    az_time_s = abs(float(delta_azimuth_deg)) / float(slew_rate_azimuth_deg_s)\n    energy_kwh = float(actuator_power_w) * (tilt_time_s + az_time_s) / 3.6e6\n    return float(scale) * energy_kwh'
    replacement='''    # This is movement-only electricity, not idle or holding consumption.
    params = (float(delta_tilt_deg), float(delta_azimuth_deg),
              float(actuator_power_w), float(slew_rate_tilt_deg_s),
              float(slew_rate_azimuth_deg_s), float(scale))
    if not np.isfinite(params).all():
        raise ValueError('Non-finite actuator-energy input')
    dt, da, power, tilt_rate, az_rate, gain = params
    if power <= 0 or tilt_rate <= 0 or az_rate <= 0 or gain < 0:
        raise ValueError('Actuator power/rates must be positive; gain nonnegative')
    return gain * power * (abs(dt) / tilt_rate + abs(da) / az_rate) / 3.6e6'''
    s=rep(s,anchor,replacement,'physical actuator energy')
    # Remove an unsubstantiated claim of general energy savings in the helper docstring.
    s=rep(s,'  At max env deltas (5° tilt + 10° azimuth) this is ~0.07 Wh — ~1% of a typical\n  5-min harvest step on the NYC NSRDB window (comparable to literature parasitic\n  losses for dual-axis trackers).',
          '    Power is consumed only during commanded motion. Both axes are assumed\n    to move sequentially with the same active power. The scale is a dimensionless\n    sensitivity factor; scale=1 gives the physical movement-only estimate.',
          'remove unverified actuator-cost claim')
    return s


def patch_environment(s):
    s=rep(s,'''        self.slew_rate_tilt_deg_s = float(slew_rate_tilt_deg_s)
        self.slew_rate_azimuth_deg_s = float(slew_rate_azimuth_deg_s)
''','''        self.slew_rate_tilt_deg_s = float(slew_rate_tilt_deg_s)
        self.slew_rate_azimuth_deg_s = float(slew_rate_azimuth_deg_s)
        movement_values = (self.movement_penalty, self.actuator_power_w,
                           self.slew_rate_tilt_deg_s, self.slew_rate_azimuth_deg_s)
        if not np.isfinite(movement_values).all():
            raise ValueError('Non-finite movement-cost parameters')
        if (self.movement_penalty < 0 or self.actuator_power_w <= 0
                or self.slew_rate_tilt_deg_s <= 0 or self.slew_rate_azimuth_deg_s <= 0):
            raise ValueError('Invalid physical actuator parameters')
''','movement configuration guard')
    s=rep(s,'''        previous_tilt = float(self.tilt)
        delta_tilt = float(action[0]) * self.max_delta_tilt
        delta_azimuth = float(action[1]) * self.max_delta_azimuth

        self.tilt = clip_panel_tilt_deg(self.tilt + delta_tilt)
        executed_delta_tilt = float(self.tilt - previous_tilt)
        self.azimuth = wrap_panel_azimuth_deg(self.azimuth + delta_azimuth)
''','''        previous_tilt = float(self.tilt)
        previous_azimuth = float(self.azimuth)
        delta_tilt = float(action[0]) * self.max_delta_tilt
        delta_azimuth = float(action[1]) * self.max_delta_azimuth

        self.tilt = clip_panel_tilt_deg(self.tilt + delta_tilt)
        executed_delta_tilt = float(self.tilt - previous_tilt)
        self.azimuth = wrap_panel_azimuth_deg(self.azimuth + delta_azimuth)
        executed_delta_azimuth = float(
            (self.azimuth - previous_azimuth + 180.0) % 360.0 - 180.0)
''','executed angular deltas')
    s=rep(s,'''                executed_delta_tilt,
                delta_azimuth,
''','''                executed_delta_tilt,
                executed_delta_azimuth,
''','physical applied movement')
    s=rep(s,'''            movement_cost = self.movement_penalty * (
                abs(delta_tilt) / self.max_delta_tilt
                + abs(delta_azimuth) / self.max_delta_azimuth
            )
''','''            movement_cost = self.movement_penalty * (
                abs(executed_delta_tilt) / self.max_delta_tilt
                + abs(executed_delta_azimuth) / self.max_delta_azimuth
            )
''','legacy applied movement')
    # Preserve original exported delta fields as requested command metadata and
    # provide correctly applied deltas separately. Evaluation needs actual motion.
    s=rep(s,'''            'delta_tilt_deg': float(delta_tilt),
            'delta_azimuth_deg': float(delta_azimuth),
''','''            'delta_tilt_deg': float(executed_delta_tilt),
            'delta_azimuth_deg': float(executed_delta_azimuth),
            'requested_delta_tilt_deg': float(delta_tilt),
            'requested_delta_azimuth_deg': float(delta_azimuth),
''','applied motion metadata')
    return s


def patch_replay(s):
    # Start from the verified stage's persistent scenario roles and fraction checks.
    s=previous.patch_replay(s)
    old='''    def rollout_states(self,n,seed=None):
        valid=np.flatnonzero((self.states[:self.count,5]>1e-7)
                                 &(self.dones[:self.count,0]<.5))
        if not len(valid):return np.empty((0,6),dtype=np.float32)
        rng=self.rng if seed is None else np.random.default_rng(seed)
        indices=rng.choice(valid,size=int(n),replace=len(valid)<int(n))
        return self.states[indices].copy()
'''
    new='''    def rollout_states(self,n,seed=None):
        # No holdout-day state is ever used to generate synthetic policy training.
        valid=np.flatnonzero((self.states[:self.count,5]>1e-7)
                                 &(self.dones[:self.count,0]<.5))
        valid=valid[np.asarray([
            self._validation_roles[self.scenario_names[int(self.scenario_ids[i])]] == 'train'
            for i in valid],dtype=bool)]
        if not len(valid):return np.empty((0,6),dtype=np.float32)
        rng=self.rng if seed is None else np.random.default_rng(seed)
        indices=rng.choice(valid,size=int(n),replace=len(valid)<int(n))
        return self.states[indices].copy()
'''
    s=rep(s,old,new,'train-only synthetic start states')
    s=rep(s,"state['validation_contract']='sticky_day_assignment_v2'",
          "state['validation_contract']='sticky_day_assignment_v3_train_only_rollouts'",
          'no-leak checkpoint contract')
    s=rep(s,"data.get('validation_contract')!='sticky_day_assignment_v2'",
          "data.get('validation_contract')!='sticky_day_assignment_v3_train_only_rollouts'",
          'no-leak checkpoint load')
    old_mix='''    n_real=int(batch_size*real_ratio)
    n_fake=batch_size-n_real
    if len(model)<n_fake or n_fake==0:
        return real.sample(batch_size),batch_size,0
    if n_real<=0 or len(real)<n_real:
        raise ValueError('Insufficient real transitions')
    rb,mb=real.sample(n_real),model.sample(n_fake)
    order=rng.permutation(batch_size)
    return {k:np.concatenate([rb[k],mb[k]],axis=0)[order] for k in rb},n_real,n_fake'''
    new_mix='''    batch_size = int(batch_size)
    real_ratio = float(real_ratio)
    if batch_size < 2 or not 0 < real_ratio < 1:
        raise ValueError('Invalid mixed minibatch proportions')
    desired_real = max(1, int(batch_size * real_ratio))
    desired_model = batch_size-desired_real
    n_model = min(desired_model, len(model))
    n_real = batch_size-n_model
    if len(real) < n_real:
        raise ValueError('Insufficient real replay transitions')
    if n_model == 0:
        return real.sample(batch_size),batch_size,0
    rb,mb=real.sample(n_real),model.sample(n_model)
    order=rng.permutation(batch_size)
    return {k:np.concatenate([rb[k],mb[k]],axis=0)[order] for k in rb},n_real,n_model'''
    s=rep(s,old_mix,new_mix,'partial synthetic minibatch mixing')
    return s


def patch_mbpo_config(s):
    s=previous.patch_mbpo_config(s)
    s=rep(s,"            'schema':'mbpo-torch-structured-v1'",
          "            'schema':'mbpo-torch-separated-exogenous-v3'",'architecture configuration digest')
    # Prior patch validates pool capacity; retain the one-step rollout constraint.
    return s


def patch_sac_config(s):
    s=previous.patch_sac_config(s)
    s=rep(s,"    if float(skeleton['discount']) != 1.0:",
          """    # Shared physical actuator objective in both training and evaluation.
    for name, env in (('training',nsrdb),('evaluation',eval_env)):
        if (not np_isclose(env.get('movement_penalty'),1.0)
                or env.get('movement_cost_mode') != 'geometry'):
            raise ValueError(f'{name} reference must enable physical movement cost')
    if float(skeleton['discount']) != 1.0:""",'aligned geometry-cost validation')
    return s


def patch_mbpo_trainer(s):
    s=previous.patch_mbpo_trainer(s)
    # The prior experiment stage already upgrades checkpoints to v2. Separate
    # dynamics model carries an explicit independent architecture schema v3.
    return s


def patch_paper(s):
    # Time from the first day window spans midnight. The fixed UTC envelope
    # starts 08:45; hours before that are necessarily from the next UTC day.
    s=rep(s,"    return dt.hour+dt.minute/60+dt.second/3600",
          "    clock = dt.hour+dt.minute/60+dt.second/3600\n    return clock + (24.0 if clock < 8.75 else 0.0)",
          'paper UTC rollover')
    start=s.index('def _stats(rows):')
    end=s.index('\ndef _dominant_condition',start)
    new='''def _stats(rows):
    """Descriptive equal-scenario means; NOT design-weighted annual estimates.

    Multiple seeds of the same weather day are first averaged, so unequal
    numbers of seeds do not change the sampling weight of a date.
    """
    grouped,by_pair=_validate_rows(rows)
    results={}
    for m in METHODS:
        episodes=[(k,v) for k,v in grouped.items() if k[2]==m]
        if not episodes:continue
        def mean_over_days(field,absolute_motion=False):
            days=defaultdict(list)
            for k,ep in episodes:
                if absolute_motion:
                    value=float(np.sum(np.abs(_arr(ep,'actual_delta_tilt_deg'))
                                      +np.abs(_arr(ep,'actual_delta_azimuth_deg'))))
                else:
                    value=float(_arr(ep,field).sum())
                days[str(k[0])].append(value)
            return float(np.mean([np.mean(values) for values in days.values()]))
        res={'n':len(episodes),'n_scenarios':len({k[0] for k,v in episodes}),
             'mean_gross_kwh':mean_over_days('energy_kwh')}
        if m in FEASIBLE:
            res.update(mean_net_kwh=mean_over_days('reward_kwh'),
                       mean_movement_cost_kwh=mean_over_days('movement_cost_kwh'),
                       mean_abs_motion_deg=mean_over_days(None,absolute_motion=True))
        else:
            res.update(mean_net_kwh=None,mean_movement_cost_kwh=None,mean_abs_motion_deg=None)
        results[m]=res
    per_pair={}
    for key,methods in by_pair.items():
        per_pair[key]={m:{'net':float(_arr(rr,'reward_kwh').sum()) if m in FEASIBLE else None,
                          'gross':float(_arr(rr,'energy_kwh').sum())} for m,rr in methods.items()}
    return results,per_pair,grouped,by_pair

'''
    s=s[:start]+new+s[end:]
    s=rep(s,'    pairs=sorted(by_pair)\n','''    pairs=sorted(by_pair)
    per_day=defaultdict(lambda:defaultdict(list))
    for (sid,seed),methods in per_pair.items():
        for method,vals in methods.items():
            per_day[sid][method].append(vals)
    balanced={sid:{method:{'net':float(np.mean([v['net'] for v in values if v['net'] is not None]))
                               if any(v['net'] is not None for v in values) else None,
                            'gross':float(np.mean([v['gross'] for v in values]))}
                      for method,values in methods.items()}
              for sid,methods in per_day.items()}
''','per-scenario paired aggregates')
    s=rep(s,'''                gain=[per_pair[k][learn]['net']-per_pair[k][m]['net'] for k in pairs]
                ax.hist(gain,bins=min(15,max(4,int(np.sqrt(len(gain))*2))),alpha=.55,
                        label=f'{LABELS[learn]} mean={np.mean(gain):.3f} kWh')''',
          '''                gain=[balanced[sid][learn]['net']-balanced[sid][m]['net'] for sid in sorted(balanced)]
                ax.hist(gain,bins=min(15,max(4,int(np.sqrt(len(gain))*2))),alpha=.55,
                        label=f'{LABELS[learn]} equal-day mean={np.mean(gain):.3f} kWh')''',
          'paired daily gains without seed oversampling')
    s=rep(s,'''        valid=[k for k in pairs if per_pair[k]['dual_axis_slew_limited']['net']>1e-9]
        for m,marker in [('mbpo','o'),('sac','s'),('fixed_no_motion','^')]:
            vals=np.array([per_pair[k][m]['net']/per_pair[k]['dual_axis_slew_limited']['net'] for k in valid])''',
          '''        valid=[sid for sid in sorted(balanced) if balanced[sid]['dual_axis_slew_limited']['net']>1e-9]
        for m,marker in [('mbpo','o'),('sac','s'),('fixed_no_motion','^')]:
            vals=np.array([balanced[sid][m]['net']/balanced[sid]['dual_axis_slew_limited']['net'] for sid in valid])''',
          'ratios averaged within day first')
    s=rep(s,'''                vals=[float(np.sum(_arr(rr,'reward_kwh'))) for (sid,sd,method),rr in grouped.items()
                      if method==m and rr[0]['season']==season]
                vv.append(np.mean(vals) if vals else np.nan)''',
          '''                days=defaultdict(list)
                for (sid,sd,method),rr in grouped.items():
                    if method==m and rr[0]['season']==season:
                        days[sid].append(float(np.sum(_arr(rr,'reward_kwh'))))
                vals=[np.mean(v) for v in days.values()]
                vv.append(np.mean(vals) if vals else np.nan)''',
          'scenario-balanced seasons in paper report')
    s=rep(s,'''                    z=[float(np.sum(_arr(rr,'reward_kwh'))) for (sid,sd,mm),rr in grouped.items()
                       if mm==m and _dominant_condition(rr)==cond]
                    values.append(np.mean(z) if z else np.nan)''',
          '''                    days=defaultdict(list)
                    for (sid,sd,mm),rr in grouped.items():
                        if mm==m and _dominant_condition(rr)==cond:
                            days[sid].append(float(np.sum(_arr(rr,'reward_kwh'))))
                    z=[np.mean(v) for v in days.values()]
                    values.append(np.mean(z) if z else np.nan)''',
          'scenario-balanced weather conditions in paper report')
    s=rep(s,'''        ax.set_title('Motion versus net energy (movement penalty may be zero)')''',
          '''        ax.set_title('Actuator movement versus net energy (descriptive)')''',
          'actuator-energy plot title')
    s=rep(s,"                'Movement penalty may be zero: movement scatter is not an energy-efficiency ranking.',",
          "                'Actuator consumption covers commanded motion only; no holding or standby power.',",
          'movement model caveat')
    s=rep(s,"            'methods':summary,'plots':[Path(p).name for p in figures],",
          "            'methods':summary,'plots':[Path(p).name for p in figures],\n            'aggregation':'descriptive equal-scenario means; seeds averaged within scenario; NOT design-weighted population estimates',",
          'paper reporting aggregation label')
    return s


def patch_plots(s):
    old='''def _history(path):
    if not path:return None
    with Path(path).open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    return rows or None
'''
    new=old+'''

def _balanced(paths,method,field):
    by_day=defaultdict(list)
    for p in paths:
        if p['method']==method:
            by_day[str(p['scenario_id'])].append(float(p[field]))
    if not by_day:
        raise ValueError(f'Method {method} has no evaluation trajectories')
    return float(np.mean([np.mean(v) for v in by_day.values()]))
'''
    s=rep(s,old,new,'MBPO plots equal-day aggregation helper')
    s=rep(s,"[np.mean([p['gross_energy_kwh'] for p in paths if p['method']==m]) for m in available]",
          "[_balanced(paths,m,'gross_energy_kwh') for m in available]",
          'MBPO gross bars')
    s=rep(s,"[np.mean([p['net_energy_kwh'] for p in paths if p['method']==m]) for m in FEASIBLE]",
          "[_balanced(paths,m,'net_energy_kwh') for m in FEASIBLE]",
          'MBPO net bars')
    s=rep(s,'''        dy=[by[(sid,sd,'mbpo')]['net_energy_kwh']-by[(sid,sd,ref)]['net_energy_kwh']
            for sid,sd in pairs]
        ax.scatter(np.full(len(dy),j),dy,alpha=.5,s=16)
        ax.plot([j-.2,j+.2],[float(np.mean(dy))]*2,linewidth=3)''',
          '''        day_differences=defaultdict(list)
        for sid,sd in pairs:
            day_differences[str(sid)].append(
                by[(sid,sd,'mbpo')]['net_energy_kwh']-by[(sid,sd,ref)]['net_energy_kwh'])
        dy=np.asarray([np.mean(v) for v in day_differences.values()],dtype=float)
        ax.scatter(np.full(len(dy),j),dy,alpha=.5,s=16)
        ax.plot([j-.2,j+.2],[float(np.mean(dy))]*2,linewidth=3)''',
          'MBPO paired gain weighting')
    s=rep(s,'''            vals=[p['net_energy_kwh'] for p in paths if p['method']==m
                  and p['infos'][0].get('season')==season]
            season_means.append(np.mean(vals) if vals else np.nan)''',
          '''            days=defaultdict(list)
            for p in paths:
                if p['method']==m and p['infos'][0].get('season')==season:
                    days[str(p['scenario_id'])].append(float(p['net_energy_kwh']))
            vals=[np.mean(v) for v in days.values()]
            season_means.append(np.mean(vals) if vals else np.nan)''',
          'MBPO seasonal weighting')
    s=rep(s,"ax.set_title('Gross energy — same NSRDB scenarios, same pvlib irradiance model')",
          "ax.set_title('Gross energy — descriptive equal-day averages')",'MBPO gross title')
    s=rep(s,"ax.set_title('Net energy — only feasible controllers with identical reward')",
          "ax.set_title('Net energy — descriptive equal-day averages')",'MBPO net title')
    return s


def patch_mbpo_evaluation(s):
    return rep(s,"if data.get('schema')!=1 or data.get('config',{}).get('sac')",
               "if data.get('schema')!=2 or data.get('config',{}).get('sac')",
               'MBPO actor evaluation schema-2 checkpoint')


def patch_sac_plots(s):
    # Some secondary SAC figures run from this module even when MBPO calls it.
    anchor="LABELS = {'sac':'Standalone SAC', 'dual_axis_slew_limited':'Dual-axis (slew-limited)',"
    if s.count(anchor)!=1:
        raise RuntimeError('SAC plotting source unexpectedly changed')
    s=rep(s,'''def plot_results(paths, outdir, training_csv=None):''',
          '''def _equal_day_mean(paths, field):
    from collections import defaultdict
    days=defaultdict(list)
    for p in paths:
        days[str(p['scenario_id'])].append(float(p[field]))
    return float(np.mean([np.mean(v) for v in days.values()]))


def plot_results(paths, outdir, training_csv=None):''','SAC plot summary helper')
    s=rep(s,"values = [np.mean([p['gross_energy_kwh'] for p in method_paths[m]]) for m in METHODS]",
          "values = [_equal_day_mean(method_paths[m], 'gross_energy_kwh') for m in METHODS]",
          'SAC plot gross bars')
    s=rep(s,"vals = [np.mean([p['net_energy_kwh'] for p in method_paths[m]]) for m in feasible_methods]",
          "vals = [_equal_day_mean(method_paths[m], 'net_energy_kwh') for m in feasible_methods]",
          'SAC plot net bars')
    s=rep(s,'''        fraction = float(np.mean(learned>reference))
        delta = float(np.mean(learned-reference))''',
          '''        from collections import defaultdict
        by_day=defaultdict(list)
        for (sid,seed),g in grouped.items():
            by_day[sid].append(g['sac']['net_energy_kwh']-g[method]['net_energy_kwh'])
        day_diffs=np.asarray([np.mean(v) for v in by_day.values()],dtype=float)
        fraction=float(np.mean(day_diffs>0))
        delta=float(np.mean(day_diffs))''',
          'SAC scientific plot equal-day pair summaries')
    s=rep(s,"'Same NSRDB scenarios — each dot is one scenario/seed pair'",
          "'Paired net energy; dots represent scenario/seed runs (descriptive)'",
          'SAC paired scatter label')
    # Do not label one-per-stratum bootstrap bands as inferential confidence intervals.
    s=rep(s,"ax.set_title(f'SAC energy gain by {kind}: scenario-clustered 95% bootstrap intervals\\nIntervals omitted when only one scenario is available')",
          "ax.set_title(f'SAC energy gain by {kind}: descriptive day-bootstrap bands\\nNot a design-based confidence interval')",
          'SAC descriptive bands')
    return s


SOURCE_TRANSFORMS={
    'SAC_workspace/src/training/trainer.py':previous.patch_sac_trainer,
    'MBPO_workspace/src/trainer.py':patch_mbpo_trainer,
    'SAC_workspace/src/config/config.py':patch_sac_config,
    'MBPO_workspace/src/config.py':patch_mbpo_config,
    'MBPO_workspace/src/replay.py':patch_replay,
    'MBPO_workspace/src/dynamics.py':lambda source:(SCRIPT/'dynamics_replacement.py').read_text(encoding='utf-8'),
    'SAC_workspace/workflows/01_setup_config.py':lambda s:previous.patch_setup_workflows(s,'sac'),
    'MBPO_workspace/workflows/01_setup_config.py':lambda s:previous.patch_setup_workflows(s,'mbpo'),
    'SAC_workspace/src/agents/sac_agent.py':previous.patch_agent_doc,
    'MBPO_workspace/workflows/03_evaluate.py':previous.patch_final_evaluator,
    'examples/config/pv_tracking/conf_compact.py':patch_compact,
    'mbpo/env/pv_tracking.py':patch_environment,
    'mbpo/env/pvlib_physics.py':patch_physics,
    'MBPO_workspace/src/paper_reporting.py':patch_paper,
    'MBPO_workspace/src/plots.py':patch_plots,
    'MBPO_workspace/src/evaluation.py':patch_mbpo_evaluation,
    'SAC_workspace/src/evaluation/plots.py':patch_sac_plots,
}


def sha_git(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode('ascii')+data).hexdigest()


def stage(root):
    staged={}
    for rel, transform in SOURCE_TRANSFORMS.items():
        path=root/rel
        if not path.is_file():
            if rel == 'SAC_workspace/src/evaluation/plots.py':
                # This source was supplied separately in SAC_src.zip and is
                # installed from new_files when absent from the Git branch.
                continue
            raise FileNotFoundError(f'Cannot install: existing source not found: {path}')
        data=path.read_bytes()
        sha=sha_git(data)
        permitted={BASE[rel],*ALTERNATE_BASE.get(rel,set())}
        if sha not in permitted:
            raise RuntimeError(f'Source revision mismatch: {rel} sha={sha}, expected {sorted(permitted)}')
        transformed=transform(data.decode('utf-8'))
        if transformed==data.decode('utf-8'):
            raise RuntimeError(f'No changes generated for {rel}')
        staged[rel]='# PVWATT_DEV_REPAIR_V3\n'+transformed
    for file in sorted((SCRIPT/'new_files').rglob('*')):
        if file.is_file() and file.suffix=='.py':
            rel=file.relative_to(SCRIPT/'new_files').as_posix()
            dest=root/rel
            wanted=file.read_text(encoding='utf-8')
            if rel in staged:
                if staged[rel] != '# PVWATT_DEV_REPAIR_V3\n'+wanted:
                    raise RuntimeError(f'Duplicate staged source has incompatible content: {rel}')
                continue
            if dest.is_file():
                if dest.read_text(encoding='utf-8')==wanted:
                    continue
                # Files supplied in SAC_src.zip may be absent from the Git
                # branch. If locally present, update only the exact audited
                # original archived version; leave divergent edits alone.
                previous_file = SCRIPT/'original_sac_evaluation'/Path(rel).name
                if (rel.startswith('SAC_workspace/src/evaluation/')
                        and previous_file.is_file()
                        and dest.read_bytes()==previous_file.read_bytes()):
                    staged[rel]=wanted
                    continue
                raise FileExistsError(f'Refusing to overwrite modified module: {rel}')
            staged[rel]=wanted
    for rel,content in staged.items():
        if rel.endswith('.py'):
            ast.parse(content,filename=rel)
    return staged


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='.',help='Root of checked-out PVwatt_dev repo')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    root=Path(args.root).resolve()
    if not (root/'MBPO_workspace/src/trainer.py').exists():
        p.error('Target must be the mbpo_custom repository root')
    staged=stage(root)
    if args.dry_run:
        print(f'DRY RUN OK: {len(staged)} complete Python source files verified; nothing written.')
        return
    backup=root/'_pvwatt_dev_repair_v3_backups'
    for rel in staged:
        original=root/rel
        if original.is_file():
            out=backup/rel
            out.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(original,out)
    # Verify and stage all files before writing. Local only, with pristine copies
    # preserved in _pvwatt_dev_repair_v3_backups/.
    for rel,content in staged.items():
        dest=root/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_name(dest.name+'.pvrepair.tmp')
        temp.write_text(content,encoding='utf-8')
        temp.replace(dest)
    (root/'PVWATT_DEV_REPAIR_MANIFEST.json').write_text(json.dumps({
        'revision':'v3','base_branch':'PVwatt_dev',
        'files':sorted(staged),'base_blob_shas':BASE,
        'note':'Checkpoint compatibility changed; new training runs are mandatory.',
    },indent=2),encoding='utf-8')
    print(f'INSTALLED: {len(staged)} local source files. GitHub untouched.')
    print('Backups:',backup)


if __name__=='__main__':
    main()
