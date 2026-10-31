#!/usr/bin/env python3
"""Evidence-only Stage 5 audit, with no checkpoint loading or simulator mutation.
Run from any directory. Python 3.10+ standard library only.
The script intentionally does not automatically assign p-values or independence claims.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage5_additional_checks import (check_clock, physical_weather_parity, validate_oracle,
    weighted_crosscheck, training_provenance_check, replay_recency_diagnostic, train_eval_overlap)

METHODS = ('sac', 'mbpo', 'fixed_no_motion')

def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def asfloat(v, where):
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f'Missing/non-numeric {where}: {v!r}') from None
    if not math.isfinite(x):
        raise ValueError(f'Non-finite {where}: {v!r}')
    return x

def close(actual, expected, atol=1e-5, rtol=1e-6):
    return abs(actual-expected) <= atol + rtol * abs(expected)

def is_blank(x):
    return x is None or str(x).strip() in ('', 'None', 'nan')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read_diag(path):
    path = Path(path)
    if path.is_file() and path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            d = json.loads(z.read('deep_diagnostic.json'))
            with z.open('branched_real_continuations.csv') as f:
                import io
                branch = list(csv.DictReader(io.TextIOWrapper(f,encoding='utf-8')))
        return d, branch
    if path.is_dir():
        d = json.loads((path/'deep_diagnostic.json').read_text(encoding='utf-8'))
        b = path/'branched_real_continuations.csv'
        return d, read_csv(b) if b.is_file() else []
    raise ValueError(f'Expected Stage 4 diagnostic ZIP or directory: {path}')

def audit_deep(d, branch):
    if d.get('schema') != 'sac-mbpo-deep-v1':
        raise ValueError('Unrecognized Stage 4 deep diagnostic schema')
    cfg = d['config']
    c = cfg['shared_sac_algo']
    checks = d['replay_contract']
    lp = d['learning_probes']
    result = {
        'checkpoints': {k: {z:v for z,v in d['checkpoint'][k].items() if z != 'path'} for k in ('sac','mbpo')},
        'physics': cfg['physics'], 'algo': c,
        'shared_sac_agent': True,
        'observed_alpha': {k: lp[k]['alpha'] for k in ('sac_real','mbpo_real')},
        'in_sample_td_absolute_Q1_mean': {k:lp[k]['critic_TD_abs_error_Q1']['mean'] for k in ('sac_real','mbpo_real','mbpo_synthetic')},
        'shadow_status': {k:d['shadow_optimizer_steps'][k]['status'] for k in ('sac','mbpo')},
        'replay_total': {k:checks[k]['count'] for k in checks},
        'terminal_mismatch': {k:checks[k]['terminal_mismatch_count'] for k in checks},
        'evaluation_action_replay':d.get('evaluation_action_replay'),
        'limitations': d.get('important_limitations'),
        'dynamics_validation': {'in_sample_power_skill':d['mbpo_model_fit_to_real_replay']['replay_skill_vs_persistence'][2],
                                'rotating_holdout_warning':d['mbpo_mixture']['critical_caveat']},
        'day_branch': {}, 'issues': [],
        'stage4_claims_status': 'REPORTED_IN_STAGE4_NOT_INDEPENDENTLY_REPLAYED',
    }
    for k,v in result['terminal_mismatch'].items():
        if v:result['issues'].append(f'Terminal mask mismatch: {k}={v}')
    for k,v in result['shadow_status'].items():
        if v!='SHADOW_UPDATE_PASSED':result['issues'].append(f'Shadow optimizer failed: {k}={v}')
    for method in ('sac','mbpo'):
        r=result['evaluation_action_replay'].get(method) if isinstance(result['evaluation_action_replay'],dict) else None
        if not r or r.get('checkpoint_actions_match_saved_evaluation') is not True or int(r.get('checked_actions',0))<1:
            result['issues'].append(f'Action-replay parity not established for {method}')
        elif not (math.isfinite(float(r.get('max_abs_normalized_action_error',float('nan'))))
                  and float(r['max_abs_normalized_action_error'])<1e-4):
            result['issues'].append(f'Action-replay error too large or nonfinite: {method}')
        shadow=d['shadow_optimizer_steps'].get(method,{})
        metrics=shadow.get('training_step_metrics',{})
        if not metrics or any(not math.isfinite(float(x)) for x in metrics.values()):
            result['issues'].append(f'Shadow step metrics missing or nonfinite: {method}')

    if c.get('reward_scale') == 100 and min(result['observed_alpha'].values()) > c.get('min_alpha',0):
        result['issues'].append('Low alpha relative to reward may merit ablation; neither alpha is at configured minimum; not a proven bug')
    for method in ('sac','mbpo'):
        rr = [r for r in branch if r.get('method') == method]
        if not rr: continue
        energy = [asfloat(r['real_return_chosen_minus_downward_kwh'],'branch physical gap') for r in rr]
        q = [asfloat(r['Q_chosen_minus_downward_reward_scaled_soft'],'branch Q gap') for r in rr]
        signs = sum((x>0)==(y>0) for x,y in zip(energy,q) if abs(x)>1e-7 and abs(y)>1e-6)
        valid = sum(abs(x)>1e-7 and abs(y)>1e-6 for x,y in zip(energy,q))
        result['day_branch'][method] = {
            'points':len(rr),'distinct_days':len({r['scenario_id'] for r in rr}),
            'physical_chosen_minus_down_mean_kwh':statistics.mean(energy),
            'critic_chosen_minus_down_mean_scaled_Q':statistics.mean(q),
            'sign_agreement':signs/valid if valid else None,
            'warning':'Selected correlated within-day branches; deterministic energy and stochastic soft-Q optimize different objectives.'}
    return result

def audit_eval_episodes(rows):
    """Validate paired, physically feasible episode summaries. Does not mistake POA oracle for net energy."""
    if not rows: raise ValueError('No evaluation episodes')
    by = defaultdict(dict)
    for r in rows:
        method = r['method'];key = (r['scenario_id'], str(r['seed']))
        if method in by[key]:raise ValueError(f'Duplicate episode: {key}/{method}')
        if method in METHODS:
            e=asfloat(r['gross_energy_kwh'],f'{key}/{method}/gross')
            n=asfloat(r['net_energy_kwh'],f'{key}/{method}/net')
            cost=asfloat(r['movement_cost_kwh'],f'{key}/{method}/cost')
            if not close(e-cost,n,atol=1e-6):raise ValueError(f'Net energy identity failed for {key}/{method}')
            for col in ('initial_tilt_deg','initial_azimuth_deg'):
                asfloat(r[col],f'{key}/{method}/{col}')
            by[key][method]=r
    if not by:raise ValueError('No SAC/MBPO/fixed evaluation rows')
    pairs=[]
    for key,group in sorted(by.items()):
        if set(group)!=set(METHODS):raise ValueError(f'Missing paired agents or fixed baseline at {key}: {set(group)}')
        poses = {(round(float(v['initial_tilt_deg']),6),round(float(v['initial_azimuth_deg'])%360,6)) for v in group.values()}
        if len(poses)!=1:raise ValueError(f'Initial pose mismatch {key}: {poses}')
        x = {'scenario_id':key[0], 'seed':key[1],
             'initial_tilt_deg':next(iter(poses))[0], 'initial_azimuth_deg':next(iter(poses))[1]}
        for method in METHODS:x[f'{method}_net_kwh'] = float(group[method]['net_energy_kwh'])
        x['sac_minus_fixed_kwh']=x['sac_net_kwh']-x['fixed_no_motion_net_kwh']
        x['mbpo_minus_fixed_kwh']=x['mbpo_net_kwh']-x['fixed_no_motion_net_kwh']
        x['mbpo_minus_sac_kwh']=x['mbpo_net_kwh']-x['sac_net_kwh']
        pairs.append(x)
    byday=defaultdict(list)
    for r in pairs:byday[r['scenario_id']].append(r)
    comparisons={}
    for name in ('sac_minus_fixed','mbpo_minus_fixed','mbpo_minus_sac'):
        vals=[statistics.mean(r[f'{name}_kwh'] for r in rr) for rr in byday.values()]
        comparisons[name]={'daily_mean_kwh':statistics.mean(vals),
                           'positive_day_fraction':sum(v>0 for v in vals)/len(vals),
                           'negative_day_fraction':sum(v<0 for v in vals)/len(vals),
                           'n_distinct_days':len(vals),
                           'status':'DESCRIPTIVE; no unqualified bootstrap CI or inference from selected days'}
    method_mean={m:statistics.mean(statistics.mean(r[f'{m}_net_kwh'] for r in rr) for rr in byday.values()) for m in METHODS}
    return {'paired_episode_count':len(pairs), 'distinct_days':len(byday),
            'method_mean_kwh_unweighted_days':method_mean,
            'comparisons':comparisons,
            'sampling_warning':'Unweighted day means are descriptive; apply the original design weights if the sample is stratified.'},pairs

def audit_detailed(rows, pairs):
    by=defaultdict(list)
    allow=set(METHODS)
    for row in rows:
        if row['method'] in allow:
            by[(row['scenario_id'],str(row['seed']),row['method'])].append(row)
    if not by:raise ValueError('No detailed trajectories for SAC/MBPO/fixed')
    paired={(r['scenario_id'],str(r['seed'])):r for r in pairs}
    n_actions=0
    for key, rr in by.items():
        if (key[0],key[1]) not in paired: raise ValueError(f'Trajectory not in episodes.csv: {key}')
        rr.sort(key=lambda x:int(x['step']))
        if [int(r['step']) for r in rr]!=list(range(1,118)):
            raise ValueError(f'Incomplete 117-step trajectory: {key}')
        pose=paired[(key[0],key[1])]
        prev=(pose['initial_tilt_deg'],pose['initial_azimuth_deg'])
        seen_utc=[]
        total=0.
        for r in rr:
            tag=f'{key}/step={r["step"]}'
            if r.get('power_model') != 'pvwatts_sapm':raise ValueError(f'Missing/wrong PVWatts physics tag at {tag}')
            power=asfloat(r['power_w'],tag+'/power_w');energy=asfloat(r['energy_kwh'],tag+'/energy_kwh')
            if not close(energy,power/12000,atol=1e-7): raise ValueError(f'Power-to-energy mismatch at {tag}')
            net=asfloat(r['net_step_kwh'],tag+'/net_step_kwh')
            cost=asfloat(r['movement_cost_kwh'],tag+'/movement_cost')
            if not close(net,energy-cost,atol=1e-7):raise ValueError(f'Reward mismatch {tag}')
            tilt=asfloat(r['tilt_deg'],tag+'/tilt_deg'); az=asfloat(r['azimuth_deg'],tag+'/azimuth_deg')
            if not (0<=tilt<=90 and -1e-5<=az<=360.00001):raise ValueError(f'Physical orientation outside bounds {tag}')
            da=asfloat(r['action_azimuth'],tag+'/action_azimuth')
            dt=asfloat(r['action_tilt'],tag+'/action_tilt')
            if abs(da)>1.00001 or abs(dt)>1.00001:raise ValueError(f'Normalized action out of bounds {tag}')
            post_tilt=max(0,min(90,prev[0]+5*dt));post_az=(prev[1]+10*da)%360
            if not close(tilt,post_tilt,atol=5e-4):raise ValueError(f'Action/tilt inconsistency {tag}: {tilt} != {post_tilt}')
            azdelta=(az-post_az+180)%360-180
            if abs(azdelta)>5e-4:raise ValueError(f'Action/azimuth inconsistency {tag}: {azdelta}')
            if key[2]=='fixed_no_motion' and (abs(dt)>1e-6 or abs(da)>1e-6):
                raise ValueError(f'Fixed baseline moved: {tag}')
            prev=(tilt,az);total+=net
            seen_utc.append(r['utc_iso']);n_actions+=1
        check_clock(rr, key[0], f'Detailed {key}')
        if len(set(seen_utc))!=117:raise ValueError(f'Duplicate timestamps {key}')
        expected=float(pose[f'{key[2]}_net_kwh'])
        if not close(total,expected,atol=2e-5):raise ValueError(f'Trajectory-to-episode net mismatch {key}')
    expected_keys={(r['scenario_id'],r['seed'],m) for r in pairs for m in METHODS}
    if set(by)!=expected_keys:raise ValueError('Detailed trajectories missing paired controller-days')
    for key in {(k[0],k[1]) for k in by}:
        clock = {tuple(x['utc_iso'] for x in sorted(by[(key[0],key[1],m)],key=lambda a:int(a['step']))) for m in METHODS}
        if len(clock)!=1:raise ValueError(f'Inconsistent method timestamps {key}')
    weather=physical_weather_parity(by,{(p['scenario_id'],str(p['seed'])) for p in pairs})
    return {'validated_trajectories':len(by),'validated_transitions':n_actions,
            'validated_methods':list(METHODS), 'excluded_from_action_contract':['poa_instant_ideal'],
            'recorded_weather_parity':weather,
            'physical_contract':'PASS for SAC, MBPO, fixed: action, pose, power-to-energy, net reward, PVWatts label, 117 steps, parsed 5min UTC clock; recorded weather parity only'}

def audit_history(path, checkpoint, label):
    rows=read_csv(path)
    epoch=int(checkpoint['epoch'])
    if len(rows)<epoch:raise ValueError(f'{label}: training history shorter than checkpoint epoch')
    ids=[r['scenario_id'] for r in rows[:epoch]]
    if int(rows[epoch-1]['env_steps'])!=int(checkpoint['real_steps']):
        raise ValueError(f'{label}: history env steps do not match checkpoint')
    counts=Counter(ids)
    alpha=[asfloat(r['alpha'],f'{label}/alpha') for r in rows[:epoch]]
    return {'episodes':epoch,'distinct_training_days':len(counts),'repeated_episode_count':epoch-len(counts),
            'fraction_of_episodes_that_are_repeats':(epoch-len(counts))/epoch,
            'mean_day_repetitions':epoch/len(counts),'maximum_repetitions':max(counts.values()),
            'min_alpha':min(alpha),'final_alpha':alpha[-1]},set(ids)

def run(args):
    d,branches=read_diag(args.deep)
    report={'schema':'sac-mbpo-stage5-v1','validation_scope':'Static artifact audit; NO new training or physics execution',
            'input_deep_sha256':sha256(args.deep) if Path(args.deep).is_file() else None,
            'deep':audit_deep(d,branches),'evaluation_status':'NOT_SUPPLIED',
            'reproducibility_issues':[], 'claims':{}}
    pairs=[]
    training={}
    for name,path in (('sac',args.sac_training_csv),('mbpo',args.mbpo_training_csv)):
        if path:
            item,ids=audit_history(path,d['checkpoint'][name],name)
            training[name]=item
            training[name]['training_csv_sha256']=sha256(path)
            training[name]['training_ids']=sorted(ids)
            history_rows=read_csv(path)
            training[name]['replay_recency']=replay_recency_diagnostic(history_rows[:d['checkpoint'][name]['epoch']],
                capacity=int(d['config']['shared_sac_algo']['replay_size']), horizon=int(d['config']['physics']['horizon']))
            training[name]['provenance_comparison']=training_provenance_check(
                training[name],d['checkpoint'][name],d.get('training_history',{}).get(name))
            if training[name]['provenance_comparison']['issues']:
                report['reproducibility_issues'] += [name.upper()+': '+x for x in training[name]['provenance_comparison']['issues']]
            if name in d.get('training_history',{}):
                prior=d['training_history'][name].get('training_window_last_100',{})
                current=history_rows[:d['checkpoint'][name]['epoch']][-100:]
                divergences=[]
                for metric in ('return_kwh','critic_loss','q_mean'):
                    if metric not in prior or metric not in current[0]: continue
                    vals=[asfloat(r[metric],name+'/'+metric) for r in current if not is_blank(r[metric])]
                    if not vals:continue
                    observed=statistics.mean(vals); expected=float(prior[metric]['mean'])
                    if not close(observed,expected,atol=1e-4,rtol=1e-4):
                        divergences.append({'metric':metric,'stage5_history_mean':observed,'stage4_history_mean':expected})
                training[name]['last100_comparison_to_stage4']=divergences
                if divergences:
                    report['reproducibility_issues'].append(name.upper()+': last-100 training metrics differ from Stage4 checkpoint-matched history')
    if training:
        report['training']={name:{k:v for k,v in obj.items() if k!='training_ids'} for name,obj in training.items()}
    if args.eval_dir:
        root=Path(args.eval_dir)
        summ=root/'summary.json';ep=root/'episodes.csv'
        if not (summ.is_file() and ep.is_file()):
            raise ValueError(f'Evaluation requires summary.json and episodes.csv in {root}')
        data=json.loads(summ.read_text(encoding='utf-8'))
        if data.get('power_model')!='pvwatts_sapm':
            raise ValueError('Evaluation summary physics not PVWatts; do not mix legacy with PVWatts')
        episode_rows=read_csv(ep)
        result,pairs=audit_eval_episodes(episode_rows)
        # Stage 4 deliberately replayed actions 2..117, not the initial action.
        # Check expected coverage without falsely describing full first-action parity.
        replay_coverage={}
        for method in ('sac','mbpo'):
            obs=report['deep']['evaluation_action_replay'].get(method,{}) if isinstance(report['deep']['evaluation_action_replay'],dict) else {}
            got=int(obs.get('checked_actions',0))
            expected=len(pairs)*116
            replay_coverage[method]={'reported':got,'expected_post_first_actions':expected,
                'first_action_verified_on_every_day':False,'matches_expected':got==expected}
            if got!=expected:
                report['reproducibility_issues'].append(f'{method}: Stage4 checkpoint-action replay coverage inconsistent with evaluation count')
        result['stage4_action_replay_coverage']=replay_coverage
        detailed=root/'trajectories_detailed.csv'
        detailed_rows=read_csv(detailed) if detailed.exists() else None
        result['detailed']=audit_detailed(detailed_rows,pairs) if detailed_rows is not None else {'status':'MISSING; physics/action checks incomplete'}
        if detailed_rows is not None:
            result['oracle_separate_gross_only_contract']=validate_oracle(episode_rows,detailed_rows,pairs)
        plan=root/'scenario_selection.json'
        if not plan.is_file():plan=root/'selection_plan.json'
        if plan.is_file():
            result['weighted_official_summary_crosscheck']=weighted_crosscheck(pairs,
                json.loads(plan.read_text(encoding='utf-8')),data,episodes_rows=episode_rows)
        result['evaluation_summary_sha256']=sha256(summ)
        result['evaluation_episodes_sha256']=sha256(ep)
        result['detailed_sha256']=sha256(detailed) if detailed.exists() else None
        result['report_methods']=data.get('methods')
        result['sampling_protocol']=data.get('sampling_protocol')
        evaluation_ids={p['scenario_id'] for p in pairs}
        for name,obj in training.items():
            overlap=train_eval_overlap(obj['training_ids'],evaluation_ids)
            result[f'{name}_train_eval_day_overlap_count']=overlap['count']
            result[f'{name}_train_eval_overlap_ids']=overlap['sample_ids']
            if overlap['count']:
                report['reproducibility_issues'].append(f'{name} train/eval overlap: {overlap["count"]} dates. Not a fully unseen evaluation set.')
        if not training:report['reproducibility_issues'].append('No episode histories supplied: held-out status cannot be verified.')
        if data.get('sampling_protocol') or (root/'selection_plan.json').exists() or (root/'scenario_selection.json').exists():
            if 'weighted_official_summary_crosscheck' in result:
                result['sampling_interpretation']='Official design-weighted means independently reproduced; unweighted per-day summaries remain descriptive.'
            else:
                report['reproducibility_issues'].append('Stratified sample without reproducible weighted estimates; do not substitute unweighted descriptive means.')
        report['evaluation']=result
        report['evaluation_status']='AUDITED'
    else:
        report['reproducibility_issues'].append('Full evaluation directory not supplied: claims about fixed beating SAC and +15% MBPO cannot be independently verified from Stage 4 ZIP.')
    if report['deep']['issues']:
        report['reproducibility_issues'] += [f'Stage4 integrity claim: {issue}' for issue in report['deep']['issues'] if
            'Shadow optimizer failed' in issue or 'Action-replay' in issue or 'Shadow step' in issue]
    report['claims']={'sac_critic_broken':'NOT_PROVED',
        'sac_optimizer_wiring_broken':'NO_EVIDENCE_IN_SAMPLED_CHECKPOINT_DIAGNOSTICS',
        'mbpo_superiority_statistically_supported':'NEEDS_MULTI_TRAINING_SEEDS_AND_PROPER_HELDOUT_DESIGN',
        'fixed_outperforms_sac':'DESCRIPTIVE_ONLY_IF_FULL_EVALUATION_SHOWS_IT',
        'more_episodes_or_larger_batch_fix_sac':'NOT_DEMONSTRATED'}
    report['overall_status']='AUDITED_WITH_REPRODUCIBILITY_FLAGS' if report['reproducibility_issues'] else 'AUDITED_WITHOUT_DETECTED_ARTIFACT_CONFLICT'
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    (out/'stage5_audit.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8')
    if pairs:
        with (out/'paired_evaluation_days.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    (out/'SEND_BACK_NOTE.md').write_text('# Stage 5 evidence collection\n\n'
        'Send SEND_ME_STAGE5_EVIDENCE.zip. This file contains analysis results, not the original full trajectory data. '
        'For a deeper independent plot/math audit, also send evaluation/summary.json, episodes.csv, '
        'trajectories_detailed.csv, selection/stratification files, training histories, and exact launch commands.\n\n'
        'No optimizer settings or checkpoints were modified. Do not claim held-out generalization without comparing date IDs.\n',encoding='utf-8')
    archive=out/'SEND_ME_STAGE5_EVIDENCE.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for name in ('stage5_audit.json','SEND_BACK_NOTE.md','paired_evaluation_days.csv'):
            p=out/name
            if p.exists():z.write(p,name)
        if args.bundle_eval and args.eval_dir:
            root=Path(args.eval_dir)
            for name in ('summary.json','episodes.csv','trajectories_detailed.csv','selection_plan.json','scenario_selection.json'):
                p=root/name
                if p.is_file():z.write(p,'evaluation/'+name)
            for name,path in (('sac',args.sac_training_csv),('mbpo',args.mbpo_training_csv)):
                if path:z.write(path,f'training/{name}_episodes.csv')
        if args.bundle_plots and args.eval_dir:
            root=Path(args.eval_dir).resolve()
            plots=sorted(p for p in root.rglob('*.png') if p.is_file() and p.stat().st_size < 12_000_000)[:30]
            for p in plots:
                if root in p.resolve().parents:z.write(p,'figures/'+str(p.relative_to(root)))
    print(f'{report["overall_status"]}: {archive}; evaluation={report["evaluation_status"]}; reproducibility concerns={len(report["reproducibility_issues"])}')
    for issue in report['reproducibility_issues']:print('CAUTION:',issue)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deep',required=True,help='Stage 4 SEND_ME_SAC_MBPO_DEEP_DIAGNOSTIC.zip or extracted folder')
    parser.add_argument('--eval-dir',default=None,help='Original paired evaluation folder containing summary/episodes/rich trajectory CSV')
    parser.add_argument('--sac-training-csv',default=None)
    parser.add_argument('--mbpo-training-csv',default=None)
    parser.add_argument('--outdir',default='stage5_evidence_output')
    parser.add_argument('--bundle-eval',action='store_true',help='Opt in to including rich evaluation CSV and training CSV in the return ZIP')
    parser.add_argument('--bundle-plots',action='store_true',help='Opt in to including up to 30 PNG plots (max 12MB each) in the return ZIP')
    args=parser.parse_args()
    run(args)
if __name__=='__main__':main()
