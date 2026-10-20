"""Read-only numerical audit of a completed stratified MBPO/SAC PV evaluation.

Usage: python -m MBPO_workspace.workflows.07_audit_scientific_results --eval-dir PATH

This checks observed/exported identities; it does not prove RL convergence,
independent training/test separation, global POA optimality, or confidence coverage.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

METHODS = {'mbpo', 'sac', 'fixed_no_motion', 'poa_instant_ideal'}
FEASIBLE = METHODS - {'poa_instant_ideal'}


def _number(value, label):
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Missing or invalid {label}: {value!r}') from exc
    if not math.isfinite(result):
        raise ValueError(f'Nonfinite {label}: {value!r}')
    return result


def _equal(actual, expected, label, *, atol=1e-7, rtol=1e-5):
    if not math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol):
        raise ValueError(f'{label}: {actual:.12g} != {expected:.12g}')


def _read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f'Empty or headerless CSV: {path}')
        return list(reader)


def _training_overlap(csv_path, sampled):
    if csv_path is None:
        return None
    file = Path(csv_path)
    if not file.is_file():
        raise FileNotFoundError(f'Explicit training history missing: {file}')
    with file.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or 'scenario_id' not in reader.fieldnames:
            raise ValueError('Training CSV must contain scenario_id: '+str(file))
        seen = {r['scenario_id'] for r in reader if r.get('scenario_id')}
    return len(seen & sampled)


def audit(root, mbpo_training_csv=None, sac_training_csv=None):
    """Validate summary, per-day episodes, step-level trajectories and plot means.

    Raises ValueError for incomplete/inconsistent stored data. Returns a
    JSON-serializable audit summary. DOES NOT modify evaluation files.
    """
    root = Path(root)
    for name in ('scenario_selection.json', 'summary.json', 'episodes.csv', 'trajectories.csv'):
        if not (root / name).is_file():
            raise FileNotFoundError('Required completed-evaluation artifact missing: '+str(root / name))
    plan = json.loads((root / 'scenario_selection.json').read_text(encoding='utf-8'))
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    strata = plan['strata']
    weights = {str(s['selected_scenario_id']): _number(s['population_days'], 'stratum weight') for s in strata}
    if not weights or len(weights) != len(strata) or any(w < 1 or int(w) != w for w in weights.values()):
        raise ValueError('Empty, duplicate or noninteger scenario-selection weights')
    selected = set(weights)
    if set(plan['selected_scenario_ids']) != selected or len(plan['selected_scenario_ids']) != len(weights):
        raise ValueError('Selection plan ID/strata mismatch')
    population = _number(plan['population_count'], 'population_count')
    _equal(sum(weights.values()), population, 'Stratum population coverage', atol=0, rtol=0)
    if int(plan['selected_count']) != len(weights):
        raise ValueError('Selection count is inconsistent')

    episodes = {}
    for row in _read_csv(root / 'episodes.csv'):
        sid, method = row['scenario_id'], row['method']
        seed = int(row['seed'])
        key = (sid, seed, method)
        if sid not in selected or method not in METHODS or key in episodes:
            raise ValueError('Unknown, duplicated or nonselected episode: '+str(key))
        gross = _number(row['gross_energy_kwh'], 'episode gross kWh '+str(key))
        movement = 0.0 if method not in FEASIBLE else _number(row['movement_cost_kwh'], 'episode movement '+str(key))
        net = None if method not in FEASIBLE else _number(row['net_energy_kwh'], 'episode net kWh '+str(key))
        if method in FEASIBLE:
            _equal(net, gross - movement, 'Episode reward identity '+str(key))
        elif str(row.get('net_energy_kwh','')).strip().lower() not in ('', 'none', 'nan'):
            raise ValueError('Gross-only POA oracle must not contain feasible net energy: '+str(key))
        episodes[key] = {'gross':gross,'net':net,'movement':movement,
                         'initial_pose':(_number(row['initial_tilt_deg'],'initial tilt'),
                                         _number(row['initial_azimuth_deg'],'initial azimuth'))}
    group = defaultdict(dict)
    for (sid, seed, method), values in episodes.items():
        group[(sid, seed)][method] = values
    if not group or {sid for sid,_ in group} != selected:
        raise ValueError('Incomplete episode selection coverage')
    seed_sets=defaultdict(set)
    for (sid, seed), methods in group.items():
        if set(methods) != METHODS:
            raise ValueError(f'Incomplete paired episode controller set: {(sid,seed)}')
        ref = methods['fixed_no_motion']['initial_pose']
        if any(not np.allclose(m['initial_pose'],ref,atol=1e-8,rtol=0) for m in methods.values()):
            raise ValueError(f'Unequal initial panel pose: {(sid,seed)}')
        seed_sets[sid].add(seed)
    if len({tuple(sorted(v)) for v in seed_sets.values()}) != 1:
        raise ValueError('Action-seed sets vary across selected dates')

    paths = defaultdict(list)
    for row in _read_csv(root / 'trajectories.csv'):
        key=(row['scenario_id'],int(row['seed']),row['method'])
        if key not in episodes:
            raise ValueError('Unexpected trajectory path: '+str(key))
        step=int(row['step'])
        power=_number(row['power_w'], 'power W '+str(key))
        energy=_number(row['energy_kwh'], 'step energy '+str(key))
        cost=_number(row['movement_cost_kwh'], 'movement cost '+str(key))
        cum=_number(row['cumulative_gross_kwh'], 'cumulative energy '+str(key))
        _number(row['tilt_deg'], 'tilt '+str(key))
        _number(row['azimuth_deg'], 'azimuth '+str(key))
        _equal(energy, power/12000., 'Step energy = W*(5/60)/1000 '+str(key))
        dt=datetime.fromisoformat(row['utc_iso'].replace('Z','+00:00'))
        if dt.utcoffset() is None or dt.utcoffset().total_seconds()!=0:
            raise ValueError('Timestamp without explicit UTC: '+str(key))
        paths[key].append((step, dt, power, energy, cost, cum))
    if set(paths) != set(episodes):
        raise ValueError('Trajectory and episode methods/dates/seeds are not identical')
    clocks={}
    for key, samples in paths.items():
        if len(samples)!=117 or [s[0] for s in samples]!=list(range(1,118)):
            raise ValueError('Incorrect 117-step trajectory: '+str(key))
        timestamps=[s[1] for s in samples]
        if any((b-a).total_seconds()!=300 for a,b in zip(timestamps,timestamps[1:])):
            raise ValueError('Incorrect 5-min trajectory interval: '+str(key))
        if (timestamps[0].hour,timestamps[0].minute)!=(12,5) or (timestamps[-1].hour,timestamps[-1].minute)!=(21,45):
            raise ValueError('Incorrect 12:05-21:45 UTC evaluation window: '+str(key))
        sid,seed,_=key
        ck=(sid,seed)
        if ck in clocks and clocks[ck]!=timestamps:
            raise ValueError('Nonpaired UTC weather clocks: '+str(ck))
        clocks[ck]=timestamps
        gross=0.0;cost=0.0
        for _,_,_,e,c,cumulative in samples:
            gross+=e;cost+=c
            _equal(cumulative,gross,'Cumulative gross trajectory '+str(key),atol=1e-6)
        record=episodes[key]
        _equal(gross,record['gross'],'Episode-to-step gross energy '+str(key))
        if key[2] in FEASIBLE:
            _equal(cost,record['movement'],'Episode-to-step movement '+str(key))
            _equal(gross-cost,record['net'],'Episode-to-step net energy '+str(key))

    def weighted(method,metric):
        total=0.0
        for sid,w in weights.items():
            vals=[v[metric] for (day,_,m),v in episodes.items() if day==sid and m==method]
            if not vals or any(v is None for v in vals):
                raise ValueError('Incomplete daily measurements: '+str((sid,method,metric)))
            total+=w*float(np.mean(vals))
        return total/population

    for method in METHODS:
        rec=summary['methods'][method]
        _equal(_number(rec['mean_gross_kwh'],'summary gross'),weighted(method,'gross'),
               'Weighted summary gross '+method)
        if method in FEASIBLE:
            _equal(_number(rec['mean_net_kwh'],'summary net'),weighted(method,'net'),
                   'Weighted summary net '+method)
        elif rec.get('mean_net_kwh') is not None:
            raise ValueError('POA summary incorrectly contains net-energy estimate')
    for pair,value in summary.get('paired_net_kwh',{}).items():
        a,sep,b=pair.partition('_minus_')
        if sep and a in FEASIBLE and b in FEASIBLE:
            _equal(_number(value['mean_difference_kwh'],'paired difference'),
                   weighted(a,'net')-weighted(b,'net'),'Summary paired mean '+pair)

    # An unconstrained instantaneous gross POA optimum is an upper bound for
    # the *same* physical power model/weather, not for net reward or learning.
    for (sid,seed),methods in group.items():
        oracle=methods['poa_instant_ideal']['gross']
        for feasible in FEASIBLE:
            if oracle + 1e-5 < methods[feasible]['gross']:
                raise ValueError(f'POA gross bound violated for {(sid,seed,feasible)}')

    curve_file=root/'advanced_suite'/'curve_band_data.csv'
    n_curve_points=0
    if curve_file.is_file():
        # Independently recompute the exported weighted power/energy line means
        # and descriptive weighted RMS bands. Never interpret these as CIs.
        for row in _read_csv(curve_file):
            method=row['method'];metric=row['metric']
            if metric not in ('power_w','cumulative_gross_kwh','cumulative_net_kwh'):
                continue  # Extra action diagnostics are verified by rich-export checks.
            a,sep,b=method.partition('_minus_')
            if sep and a in METHODS and b in METHODS:
                pair=(a,b)
            elif method in METHODS:
                pair=(method,)
            else:
                continue
            step=int(round((float(row['clock_hour_utc'])-12)*12))
            if not 1<=step<=117:
                raise ValueError('Plot point is not on 5-minute native time grid')
            def day_value(sid,m):
                vals=[]
                for seed in seed_sets[sid]:
                    p=paths[(sid,seed,m)]
                    if metric=='power_w':v=p[step-1][2]
                    elif metric=='cumulative_gross_kwh':v=p[step-1][5]
                    else:v=sum(q[3]-q[4] for q in p[:step])
                    vals.append(v)
                return float(np.mean(vals))
            y=np.array([day_value(sid,pair[0])-(day_value(sid,pair[1]) if len(pair)==2 else 0)
                        for sid in weights],dtype=float)
            w=np.array(list(weights.values()),dtype=float)
            mu=float(np.average(y,weights=w))
            scatter=float(np.sqrt(np.average((y-mu)**2,weights=w))) if len(y)>1 else None
            _equal(_number(row['weighted_mean'],'plotted mean'),mu,
                   'Plotted design-weighted mean '+str((method,metric,step)),atol=1e-5)
            if scatter is not None:
                _equal(_number(row['observed_day_rms_sd'],'plotted RMS'),scatter,
                       'Plotted descriptive scatter '+str((method,metric,step)),atol=1e-5)
            elif str(row.get('observed_day_rms_sd','')).strip().lower() not in ('', 'none', 'nan'):
                raise ValueError('Single sampled date cannot have observed-day RMS band')
            n_curve_points+=1

    overlap={'mbpo':_training_overlap(mbpo_training_csv,selected),
             'sac':_training_overlap(sac_training_csv,selected)}
    return {'status':'PASS - internal numerical consistency, NOT proof of generalization',
            'sampled_days':len(weights),'represented_manifest_days':int(population),
            'paired_day_seed_groups':len(group),'trajectory_paths':len(paths),
            'tested_curve_points':n_curve_points,'training_day_overlap':overlap,
            'weighted_net_energy_kwh':{m:weighted(m,'net') for m in sorted(FEASIBLE)},
            'poa_oracle_weighted_gross_kwh':weighted('poa_instant_ideal','gross'),
            'sampling_ci_status':'Not estimable from one random date per nontrivial stratum',
            'limits':['Cannot certify checkpoint convergence, weather physics, true held-out generalization or confidence coverage',
                      'Training overlap counts require explicit training history paths',
                      'The POA check is necessary consistency, not a global optimality proof']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir',required=True)
    parser.add_argument('--mbpo-training-csv')
    parser.add_argument('--sac-training-csv')
    args=parser.parse_args()
    result=audit(args.eval_dir,args.mbpo_training_csv,args.sac_training_csv)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
