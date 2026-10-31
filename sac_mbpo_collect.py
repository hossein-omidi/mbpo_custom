#!/usr/bin/env python3
"""SAC/MBPO project evidence collector (read-only; Python 3.10+).

Run from the root of mbpo_custom:
    python sac_mbpo_collect.py

This script does not train, run simulations, or overwrite existing project files.
Only trust locally created .pt files: torch.load(weights_only=False) executes pickle.
Disable checkpoint deserialization with --no-load-checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import traceback
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

VERSION = '2026-10-30.v2'
STAGE4 = {
    'sac': {'sha256': '749e28902aede9d64154bfb27431b3a47522f33f43e6e85015f51db1c28f1688',
            'epoch': 2000, 'env_steps': 234000, 'updates': 452000,
            'alpha': 0.01558018196374178,
            'last100': {'return_kwh': 0.743684, 'critic_loss': 10.505924, 'q_mean': 34.171192}},
    'mbpo': {'sha256': '69da37c2de95ce50e25f71a51181c2a89716e60fbc8d1d03346a3a04e178037a',
             'epoch': 500, 'env_steps': 58500, 'updates': 101000,
             'alpha': 0.012782987207174301},
}

# Explicitly approved small research source files only. Never copy secrets, checkpoints,
# weather datasets, git internals or environment files.
SOURCE_FILES = [
    'SAC_workspace/src/agents/sac_agent.py',
    'SAC_workspace/src/training/replay_buffer.py',
    'SAC_workspace/src/training/trainer.py',
    'SAC_workspace/src/config/config.py',
    'SAC_workspace/src/environment/environment.py',
    'SAC_workspace/src/evaluation/evaluation.py',
    'SAC_workspace/src/evaluation/scenario_sampling.py',
    'SAC_workspace/src/evaluation/stratified_statistics.py',
    'SAC_workspace/src/evaluation/statistical_inference.py',
    'SAC_workspace/workflows/03_train.py',
    'MBPO_workspace/src/dynamics.py',
    'MBPO_workspace/src/trainer.py',
    'MBPO_workspace/src/replay.py',
    'MBPO_workspace/src/evaluation.py',
    'MBPO_workspace/src/config.py',
    'MBPO_workspace/src/agents/mbpo_agent.py',
    'MBPO_workspace/workflows/02_train.py',
    'MBPO_workspace/workflows/03_evaluate.py',
    'MBPO_workspace/representation_config.json',
    'mbpo/env/pv_tracking.py',
    'mbpo/env/pvlib_physics.py',
    'mbpo/env/pv_power_backend.py',
]


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite_num(value):
    try: x=float(value)
    except (ValueError,TypeError,OverflowError):return None
    return x if math.isfinite(x) else None


def relative(root, path):
    try:return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:return '<outside-repository>'


def safe_path(root, path):
    """Prevent symlink traversal outside project for packaged files."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError: return False


def json_safe(x):
    if isinstance(x,dict):return {str(k):json_safe(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [json_safe(v) for v in x]
    if isinstance(x,(str,int,bool)) or x is None:return x
    if isinstance(x,float):return x if math.isfinite(x) else None
    return str(x)


def json_write(path, x):
    Path(path).write_text(json.dumps(json_safe(x),ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def run_git(root):
    result={'head':None,'branch':None,'dirty_file_count':None,'status':'UNAVAILABLE'}
    for key,args in [('head',['git','rev-parse','HEAD']),('branch',['git','branch','--show-current']),('status_lines',['git','status','--porcelain'])]:
        try:
            cp=subprocess.run(args,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              text=True,timeout=12,check=False)
            if cp.returncode==0:
                if key=='status_lines':result['dirty_file_count']=len(cp.stdout.splitlines())
                else:result[key]=cp.stdout.strip()
                result['status']='AVAILABLE'
        except (OSError,subprocess.TimeoutExpired): pass
    return result


def discover(root, area, name, cap=40):
    base=root/area
    if not base.is_dir():return []
    try:
        paths=sorted(p for p in base.rglob(name) if p.is_file() and safe_path(root,p))
        return paths[:cap]
    except (OSError,PermissionError): return []


def load_csv_summary(path):
    """Do not read large process traces into RAM; training CSV is small."""
    with path.open(encoding='utf-8-sig',newline='') as f:
        dr=csv.DictReader(f)
        fields=dr.fieldnames or []
        rows=list(dr)
    if not rows:raise ValueError('CSV contains no rows')
    last=rows[-1];num=len(rows)
    days=[r.get('scenario_id') or '' for r in rows]
    result={'columns':fields,'rows':num,'first':{k:last_val(rows[0],k) for k in ('epoch','env_steps','updates','scenario_id','alpha')},
            'last':{k:last_val(last,k) for k in ('epoch','env_steps','updates','scenario_id','alpha')},
            'distinct_scenario_ids':len(set(days)),
            'last100_means':{},'anomalies':[]}
    for name in ('return_kwh','critic_loss','actor_loss','alpha_loss','entropy','q_mean','target_mean'):
        vals=[finite_num(r.get(name)) for r in rows[-100:]]
        vals=[v for v in vals if v is not None]
        if vals:result['last100_means'][name]=statistics.mean(vals)
    try:
        ep=[int(r['epoch']) for r in rows]; steps=[int(r['env_steps']) for r in rows]
        if ep!=list(range(1,len(rows)+1)):
            result['anomalies'].append('Episode numbers are not continuous 1..N')
        if any(b<=a for a,b in zip(steps,steps[1:])):
            result['anomalies'].append('Non-increasing environment steps')
        if any(b-a!=117 for a,b in zip(steps,steps[1:])):
            result['anomalies'].append('One or more episodes are not 117 real steps')
        for r in rows:
            a=finite_num(r.get('alpha'))
            if a is None or a<=0:
                result['anomalies'].append('One or more alpha values invalid/non-positive');break
    except (ValueError,TypeError,KeyError) as e:
        result['anomalies'].append('Missing or inconsistent required CSV columns: '+str(e))
    result['season_count']={}
    for month,months in [('winter',{12,1,2}),('spring',{3,4,5}),('summer',{6,7,8}),('autumn',{9,10,11})]:
        result['season_count'][month]=sum(1 for v in days if len(v)>=7 and v[5:7].isdigit() and int(v[5:7]) in months)
    recent=days[-428:]
    result['last428_episodes']={'distinct_scenarios':len(set(recent)),'episodes':len(recent)}
    return result


def last_val(row,key):return row.get(key)


def flatten_checkpoint_meta(ck, kind):
    if not isinstance(ck,dict):raise ValueError('Checkpoint is not a dictionary')
    wanted=('schema','epoch','env_steps','updates','hash','config_hash')
    out={k:ck.get(k) for k in wanted if k in ck}
    mdp=ck.get('resolved_mdp')
    if isinstance(mdp,dict):
        out['resolved_mdp']={k:mdp.get(k) for k in ('power_model','observation_mode','horizon','periods',
          'max_delta_tilt','max_delta_azimuth','movement_penalty','gamma_pdc','sapm_mount',
          'wind_reference_height_m','wind_powerlaw_exponent','power_source_sha256') if k in mdp}
    a=ck.get('agent')
    if isinstance(a,dict):
        out['agent_keys']=sorted(str(k) for k in a.keys())
        if 'updates' in a:out['agent_updates']=int(a['updates'])
        if 'log_alpha' in a:
            try:
                val=a['log_alpha']
                if hasattr(val,'detach'):val=val.detach().float().reshape(-1)[0].item()
                else:val=float(val)
                raw=math.exp(float(val))
                out['agent_alpha_unclamped']=raw
                minimum=a.get('min_alpha')
                if minimum is None and 'config' in ck:
                    conf=ck.get('config')
                    if isinstance(conf,dict):
                        sec=conf.get('sac',conf)
                        if isinstance(sec,dict):
                            algo=sec.get('algo',sec)
                            if isinstance(algo,dict):minimum=algo.get('min_alpha')
                if minimum is None:minimum=0.001
                out['agent_alpha_effective']=max(raw,float(minimum))
                out['alpha_minimum_assumed_if_no_saved_value']=minimum if 'min_alpha' not in a else None
            except Exception as e:out['alpha_read_error']=str(e)
        for part in ('actor','critic','critic_target'):
            weights=a.get(part)
            if isinstance(weights,dict):
                out[part+'_parameter_tensors']=len(weights)
                try:
                    import torch
                    out[part+'_all_finite']=all(bool(torch.isfinite(v).all().item()) for v in weights.values() if torch.is_tensor(v))
                except Exception as e:out[part+'_finite_check_error']=type(e).__name__
    # Do not recursively serialize raw optimizer tensors, replay, configs, or model weights.
    if kind=='sac':rep=ck.get('replay')
    else:rep=ck.get('real_pool')
    if isinstance(rep,dict):
        out['real_buffer']={k:rep.get(k) for k in ('count','size','capacity','pos','idx') if isinstance(rep.get(k),(int,float,str))}
    if kind=='mbpo':
        syn=ck.get('model_pool')
        if isinstance(syn,dict):out['synthetic_buffer']={k:syn.get(k) for k in ('count','size','capacity') if isinstance(syn.get(k),(int,float,str))}
    return json_safe(out)


def checkpoint_info(path,kind,allow_load=True):
    checksum=sha256_file(path)
    result={'path':None,'sha256':checksum,'size_bytes':path.stat().st_size,
            'stage4_expected_hash_matches':checksum==STAGE4[kind]['sha256'],
            'status':'NOT_DESERIALIZED','metadata':{}}
    if not allow_load:
        result['note']='Checkpoint metadata deliberately not loaded (--no-load-checkpoints)'
        return result
    try:import torch
    except ImportError:
        result['note']='PyTorch not installed in active environment; hash-only check completed'
        return result
    try:
        # Never use this on files you did not create and trust. It is local, no network.
        ck=torch.load(path,map_location='cpu',weights_only=False)
        result['metadata']=flatten_checkpoint_meta(ck,kind)
        result['status']='READ'
        del ck
    except Exception as e:
        result['status']='LOAD_FAILED'
        result['note']=f'{type(e).__name__}: {str(e)[:500]}'
    return result


def compare_pair(c,h):
    if not c or not h:return {'status':'MISSING_FILE','issues':['checkpoint or history missing']}
    if c.get('status')!='READ':return {'status':'CHECKPOINT_NOT_READ','issues':[c.get('note','checkpoint not deserialized')]}
    m=c['metadata'];issues=[];checks={}
    for ckkey,hkey in (('epoch','epoch'),('env_steps','env_steps'),('agent_updates','updates')):
        if ckkey not in m or hkey not in h['last'] or h['last'][hkey] is None:
            checks[ckkey]='NOT_CHECKED'
        else:
            try:
                matched=int(m[ckkey])==int(h['last'][hkey])
                checks[ckkey]='MATCH' if matched else 'MISMATCH'
                if not matched:issues.append(f'{ckkey}: checkpoint {m[ckkey]} vs CSV {h["last"][hkey]}')
            except (ValueError,TypeError):checks[ckkey]='NOT_CHECKED'
    aa=finite_num(m.get('agent_alpha_effective'));ha=finite_num(h['last'].get('alpha'))
    if aa is None or ha is None:checks['alpha']='NOT_CHECKED'
    else:
        matches=math.isclose(aa,ha,abs_tol=1e-7,rel_tol=1e-5)
        checks['alpha']='MATCH' if matches else 'MISMATCH'
        if not matches:issues.append(f'alpha: checkpoint {aa:.10f} vs CSV {ha:.10f}')
    if issues:status='MISMATCH'
    elif all(v=='MATCH' for v in checks.values()):status='MATCH_ON_CHECKED_FIELDS'
    else:status='INCONCLUSIVE_INCOMPLETE_METADATA'
    return {'status':status,'checks':checks,'issues':issues,
            'limitation':'Matching final values do not prove that all prior episodes came from this checkpoint run'}


def compare_to_stage4(kind,c,h):
    expected=STAGE4[kind]
    if not c:return {'status':'NO_CHECKPOINT'}
    if c['sha256']!=expected['sha256']:
        return {'status':'DIFFERENT_CHECKPOINT_THAN_STAGE4',
                'expected_stage4_sha256':expected['sha256'], 'current_sha256':c['sha256']}
    out={'status':'CHECKPOINT_MATCHES_STAGE4','checkpoint_sha256':c['sha256'],'issues':[]}
    if not h:out['status']='HISTORY_MISSING';return out
    if not math.isclose(float(h['last']['alpha']),float(expected['alpha']),rel_tol=1e-5,abs_tol=1e-7):
        out['status']='STAGE4_CHECKPOINT_HISTORY_MISMATCH'
        out['issues'].append(f'Stage4 alpha {expected["alpha"]:.10f} vs CSV {float(h["last"]["alpha"]):.10f}')
    for key,expected_val in expected.get('last100',{}).items():
        actual=h['last100_means'].get(key)
        if actual is not None and not math.isclose(actual,expected_val,abs_tol=1e-3,rel_tol=1e-3):
            out['status']='STAGE4_CHECKPOINT_HISTORY_MISMATCH'
            out['issues'].append(f'Last-100 {key}: Stage4 ~{expected_val:.6f} vs CSV {actual:.6f}')
    return out


def read_eval_summary(path):
    d=json.loads(path.read_text(encoding='utf-8'))
    methods=d.get('methods') or {}
    def parse(vals):
        if not isinstance(vals,dict):return None
        return {k:vals[k] for k in ('count','mean_net_kwh','mean_gross_kwh','mean_movement_cost_kwh') if k in vals}
    return {'power_model':d.get('power_model'),'paired_episodes':d.get('paired_episodes'),
            'scenario_count':d.get('scenario_count'),
            'method_means':{k:parse(v) for k,v in methods.items()},
            'sampling_protocol':d.get('sampling_protocol')}


def list_sources(root):
    out=[]
    for rel in SOURCE_FILES:
        path=root/rel
        present=path.is_file() and safe_path(root,path)
        out.append({'file':rel,'exists':present,'size_bytes':path.stat().st_size if present else None,
                    'sha256':sha256_file(path) if present else None})
    return out


def local_inventory(root):
    return {'version':VERSION,'repository_name':root.name,'platform':platform.platform(),
            'python':sys.version.split()[0],'git':run_git(root),
            'source_inventory':list_sources(root),
            'note':'No secrets, raw weather data, checkpoints, optimizer state or private .env copied.'}


def output_dir(root, base):
    if base:
        p=Path(base).expanduser()
        if not p.is_absolute():p=root/p
        p.mkdir(parents=True,exist_ok=True)
        return p
    for i in range(100):
        path=root/('sac_mbpo_collect_output' if i==0 else f'sac_mbpo_collect_output_{i}')
        try:path.mkdir(exist_ok=False);return path
        except FileExistsError:continue
    raise RuntimeError('Cannot choose a fresh output folder')


def run(root, *, no_load_checkpoints=False, include_source=True, include_traces=False, outdir=None):
    root=Path(root).expanduser().resolve()
    if not (root/'SAC_workspace').is_dir() or not (root/'MBPO_workspace').is_dir():
        raise ValueError('Not at mbpo_custom repository root. Run: cd ~/projects/PVLIB_Paper/mbpo_custom')
    out=output_dir(root,outdir)
    report={'schema':'sac-mbpo-local-inventory-v2','tool_version':VERSION,
            'status':'COLLECTED','reproducibility_issues':[],
            'project':local_inventory(root),'runs':{},'evaluations':[],
            'privacy':{'checkpoint_weights_included':False,'weather_included':False,'secrets_included':False,
                       'source_included':bool(include_source),'trajectories_included':bool(include_traces)}}
    if not report['project']['git']['head']:
        report['reproducibility_issues'].append('Git commit hash not available; capture exact code revision before paper.')
    missing=[x['file'] for x in report['project']['source_inventory'] if not x['exists']]
    report['project']['unavailable_whitelisted_source_paths']=missing
    report['project']['missing_path_note']='These are probes for potentially relevant code files, not a pass/fail checklist; repository layouts may differ.'
    snapshots=[]
    for kind in ('sac','mbpo'):
        base='SAC_workspace/runs' if kind=='sac' else 'MBPO_workspace/runs'
        checkpoints=discover(root,base,'latest.pt',cap=40)
        checkpoint_entries=[]
        for p in checkpoints:
            print(f'  Checking {kind.upper()} checkpoint: {relative(root,p)}',flush=True)
            c=checkpoint_info(p,kind,allow_load=not no_load_checkpoints)
            c['path']=relative(root,p)
            # Directory-level pairing only; no accidental mixing by similar names.
            csvpath=p.parent/'episodes.csv'
            hist=None
            if csvpath.is_file() and safe_path(root,csvpath):
                try:
                    hist=load_csv_summary(csvpath)
                    hist['path']=relative(root,csvpath)
                    hist['sha256']=sha256_file(csvpath)
                    snapshots.append((csvpath, f'training/{kind}/{p.parent.name}_episodes.csv'))
                except Exception as e:
                    report['reproducibility_issues'].append(f'{kind} {relative(root,csvpath)}: CSV error {e}')
            pairing=compare_pair(c,hist)
            stage4=compare_to_stage4(kind,c,hist)
            checkpoint_entries.append({'checkpoint':c,'training_history':hist,'provenance':pairing,'stage4_comparison':stage4})
            if pairing['status']=='MISMATCH':
                report['reproducibility_issues'].append(f'{kind} {relative(root,p)}: checkpoint/history mismatch: '+', '.join(pairing['issues']))
            if stage4['status']=='STAGE4_CHECKPOINT_HISTORY_MISMATCH':
                report['reproducibility_issues'].append(f'{kind} {relative(root,p)}: expected Stage4 checkpoint found, but paired CSV disagrees: '+', '.join(stage4['issues']))
        report['runs'][kind]=checkpoint_entries
        if not checkpoints: report['reproducibility_issues'].append(f'No {kind} latest.pt found under {base}')
    ebase=root/'MBPO_workspace/evaluation'
    if ebase.is_dir():
        summaries=discover(root,'MBPO_workspace/evaluation','summary.json',cap=25)
        for path in summaries:
            item={'summary_path':relative(root,path)}
            try:
                item.update(read_eval_summary(path))
                ep=path.parent/'episodes.csv'
                selection=path.parent/'scenario_selection.json'
                if ep.is_file() and safe_path(root,ep):
                    item['episodes_count']=count_csv_rows(ep)
                    snapshots.append((ep, f'evaluation/{path.parent.name}_episodes.csv'))
                if selection.is_file() and safe_path(root,selection):
                    sd=json.loads(selection.read_text(encoding='utf-8'))
                    item['sampling_population_count']=sd.get('population_count')
                    item['sampling_selected_count']=len(sd.get('selected_scenario_ids',[]))
                for f in (path,selection):
                    if f.is_file() and safe_path(root,f):
                        snapshots.append((f,f'evaluation/{path.parent.name}_{f.name}'))
                detailed=path.parent/'trajectories_detailed.csv'
                if detailed.is_file():
                    item['detailed_trajectory_file_present']=True
                    item['detailed_trajectory_bytes']=detailed.stat().st_size
                    if include_traces and safe_path(root,detailed):
                        snapshots.append((detailed,f'evaluation/{path.parent.name}_trajectories_detailed.csv'))
            except Exception as e:
                item['scan_error']=f'{type(e).__name__}: {str(e)[:300]}'
            report['evaluations'].append(item)
    if not report['evaluations']:
        report['reproducibility_issues'].append('No evaluation/summary.json found; full comparison not scanned.')
    report['status']='COLLECTED_WITH_FLAGS' if report['reproducibility_issues'] else 'COLLECTED_NO_DETECTED_CONFLICT'
    resultpath=out/'PROJECT_DIAGNOSTIC.json'
    json_write(resultpath,report)
    txt=make_summary(report)
    (out/'READ_ME_FIRST.txt').write_text(txt,encoding='utf-8')
    zpath=out/'SEND_ME_SAC_MBPO_PROJECT_DIAGNOSTIC.zip'
    unique=set()
    with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.write(resultpath,'PROJECT_DIAGNOSTIC.json')
        z.write(out/'READ_ME_FIRST.txt','READ_ME_FIRST.txt')
        for path,member in snapshots:
            if member in unique or not path.is_file() or not safe_path(root,path):continue
            if path.stat().st_size>35*1024*1024:continue
            z.write(path,member);unique.add(member)
        if include_source:
            for item in report['project']['source_inventory']:
                if not item['exists'] or int(item['size_bytes'])>800000:continue
                p=root/item['file']
                if not safe_path(root,p):continue
                z.write(p,'source/'+item['file'])
    print('\n'+txt)
    print(f'OUTPUT_FOLDER: {out}')
    print(f'SEND_ME: {zpath}')
    return report,zpath


def count_csv_rows(path):
    with path.open(newline='',encoding='utf-8-sig') as f:
        return max(0,sum(1 for _ in csv.reader(f))-1)


def make_summary(report):
    lines=['SAC / MBPO LOCAL PROJECT DIAGNOSIS',
           'Status: '+report['status'],
           'Repository: '+report['project']['repository_name'],
           'Git HEAD: '+str(report['project']['git']['head']),
           'Absent whitelisted code paths (possibly optional): '+str(len(report['project']['unavailable_whitelisted_source_paths']))]
    for kind,runs in report['runs'].items():
        lines.append(f'{kind.upper()} checkpoint candidates: {len(runs)}')
        for r in runs:
            c=r['checkpoint'];lines.append(' - '+str(c['path']))
            lines.append('   Stage4 hash match: '+str(c['stage4_expected_hash_matches'])+
                         '; checkpoint/CSV: '+r['provenance']['status']+
                         '; Stage4 consistency: '+r['stage4_comparison']['status'])
            for issue in r['provenance'].get('issues',[]):lines.append('   ISSUE '+issue)
    lines.append('Evaluation summary folders: '+str(len(report['evaluations'])))
    lines.append('Collected flags: '+str(len(report['reproducibility_issues'])))
    for w in report['reproducibility_issues'][:25]:lines.append(' - '+w)
    lines.append('')
    lines.append('NO training was run. NO checkpoint, code, or data was modified.')
    lines.append('ZIP includes metadata, selected training/evaluation CSVs and approved source files only.')
    lines.append('Review the ZIP before uploading, especially if the source code is private.')
    return '\n'.join(lines)+'\n'


def self_test():
    """No external files and no torch dependency. Does not validate actual SAC."""
    from tempfile import TemporaryDirectory
    assert finite_num('nan') is None
    assert finite_num('0.02')==0.02
    fake={'last':{'epoch':'100','env_steps':'11700','updates':'10','alpha':'0.01558'}}
    ck={'status':'READ','metadata':{'epoch':100,'env_steps':11700,'agent_updates':10,'agent_alpha_effective':0.01558}}
    assert compare_pair(ck,fake)['status']=='MATCH_ON_CHECKED_FIELDS'
    fake['last']['alpha']='0.025'
    assert compare_pair(ck,fake)['status']=='MISMATCH'
    with TemporaryDirectory() as tmp:
        a=Path(tmp)/'a.txt';a.write_text('verify');assert len(sha256_file(a))==64
    print('SELF_TEST_PASS: collector integrity tests passed. No checkpoints loaded.')


def main():
    a=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a.add_argument('--repo-root',default='.',help='Repository root; default=current directory')
    a.add_argument('--outdir',default=None,help='Custom output directory (otherwise a new numbered folder)')
    a.add_argument('--no-load-checkpoints',action='store_true',help='Disable torch.load; hashes only (safer for untrusted files)')
    a.add_argument('--no-source',action='store_true',help='Do not include selected source code in upload ZIP')
    a.add_argument('--include-trajectories',action='store_true',help='Include large evaluation trajectories if present (optional)')
    a.add_argument('--self-test',action='store_true',help='Quick tests, no repository access')
    args=a.parse_args()
    if args.self_test:self_test();return
    try:run(args.repo_root,no_load_checkpoints=args.no_load_checkpoints,
            include_source=not args.no_source,include_traces=args.include_trajectories,
            outdir=args.outdir)
    except Exception as e:
        print(f'ERROR: {type(e).__name__}: {e}',file=sys.stderr)
        print('Verify you are inside ~/projects/PVLIB_Paper/mbpo_custom and run again.',file=sys.stderr)
        raise SystemExit(2)

if __name__=='__main__':main()
