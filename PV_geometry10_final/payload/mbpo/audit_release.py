"""Release-boundary checks and diagnostics for the managed Geometry10 workflow."""
from __future__ import annotations
import csv
import hashlib
from pathlib import Path
import numpy as np
from mbpo.audit_support import sha_file, read_json, atomic_json, expected_steps


def checkpoint_identity(data, algorithm, seed, epoch, steps):
    if algorithm not in ('sac','mbpo'):
        raise ValueError('Unknown checkpoint algorithm')
    # Legacy files have distinct schema keys; new checkpoints state their role explicitly.
    inferred = 'mbpo' if 'dynamics' in data and 'real_pool' in data else 'sac' if 'replay' in data and 'dynamics' not in data else None
    if (data.get('schema')!=2 or inferred!=algorithm or data.get('algorithm',inferred)!=algorithm
            or data.get('seed')!=seed or data.get('epoch')!=epoch or data.get('env_steps')!=steps):
        raise ValueError('Checkpoint algorithm/seed/epoch/interaction identity mismatch')


def load_verified_checkpoint(path, algorithm, *, run_dir=None, expected_seed=None, expected_epoch=None, expected_sha=None):
    """Load trusted local checkpoints, binding the exact bytes to run metadata when available."""
    import torch
    path=Path(path).resolve(); root=path.parent if run_dir is None else Path(run_dir).resolve()
    meta_path=root/'run_metadata.json'; meta=read_json(meta_path) if meta_path.is_file() else None
    if meta is not None:
        if meta.get('training_status')!='completed' or meta.get('algorithm')!=algorithm:
            raise ValueError('Expected a completed run for the requested algorithm')
        if expected_seed is not None and meta.get('seed')!=expected_seed: raise ValueError('Run seed differs from requested seed')
        if expected_epoch is not None and meta.get('final_epoch')!=expected_epoch: raise ValueError('Run epoch differs from requested budget')
        expected_seed=meta['seed']; expected_epoch=meta['final_epoch']; steps=meta['final_env_steps']
        if steps!=expected_epoch*meta['horizon']: raise ValueError('Invalid run interaction budget')
        sealed=meta.get('final_checkpoint_sha256')
        if expected_sha is not None and sealed is not None and sealed!=expected_sha: raise ValueError('Checkpoint digest differs from completed run')
        expected_sha=expected_sha or sealed
    elif expected_seed is not None or expected_epoch is not None or expected_sha is not None or run_dir is not None:
        raise ValueError('Managed checkpoint requires completed run metadata')
    else:
        # Direct legacy caller with no run metadata: identity checks available only for algorithm.
        steps=None
    with path.open('rb') as stream:
        def file_hash():
            stream.seek(0); h=hashlib.sha256()
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            return h.hexdigest()
        before=file_hash()
        if expected_sha is not None and before!=expected_sha:raise ValueError('Checkpoint bytes differ from expected digest')
        stream.seek(0); data=torch.load(stream,map_location='cpu',weights_only=False)
        if file_hash()!=before:raise ValueError('Checkpoint changed during load')
    if meta is None: expected_seed=data.get('seed');expected_epoch=data.get('epoch');steps=data.get('env_steps')
    checkpoint_identity(data,algorithm,expected_seed,expected_epoch,steps)
    if meta is not None:
        if data.get('scenario_protocol_sha256')!=meta.get('scenario_protocol_sha256'):
            raise ValueError('Checkpoint protocol differs from run metadata')
        if 'training_identity_sha256' in meta:
            h=sha_file(root/'training_identity.json')
            if h!=meta['training_identity_sha256'] or h!=data.get('training_identity_sha256'):
                raise ValueError('Checkpoint training identity differs from completed run')
    return data,before


def seal_evaluation(out, identity):
    """Bind every raw CSV/JSON consumed by reports, not only model inputs."""
    out=Path(out); mode=identity['mode']
    required=(('model_validation_summary.json','internal_holdout_predictions.csv','external_on_policy_predictions.csv','counterfactual_predictions.csv') if mode=='model' else ('mc_summary.json','all_day_draw_method.csv','daily_power_mean_std_by_utc_step.csv','mc_pooled_power_mean_sd_by_utc_step.csv','first_day_power_mean_sd.csv'))
    files={p.name:sha_file(p) for p in sorted(out.iterdir()) if p.is_file() and p.suffix in ('.csv','.json') and p.name!='manager_provenance.json'}
    if not set(required)<=set(files):raise ValueError('Missing evaluation source artifacts')
    # No missing/empty table can be sealed. Existing workflow verifies numerical identities.
    for name in required:
        p=out/name
        if p.suffix=='.csv':
            with p.open(newline='',encoding='utf-8') as f:
                r=csv.reader(f); header=next(r,None); row=next(r,None)
                if not header or not row or len(header)!=len(row):raise ValueError(f'Empty or malformed evaluation artifact: {p}')
    atomic_json(out/'manager_provenance.json',{'identity':identity,'files':files})


def require_evaluation_seal(out, identity):
    out=Path(out); recorded=read_json(out/'manager_provenance.json')
    if recorded.get('identity')!=identity or not recorded.get('files'):
        raise ValueError(f'{out}: evaluation inputs/checkpoint hashes differ from recorded provenance')
    current={p.name:sha_file(p) for p in sorted(out.iterdir()) if p.is_file() and p.suffix in ('.csv','.json') and p.name!='manager_provenance.json'}
    if current!=recorded['files']:raise ValueError(f'{out}: evaluation source artifacts changed')


def replay_at_confirmation(replay, rows):
    """Cumulative replay exposure up to confirmed target or fixed censoring budget."""
    index={(int(r['seed']),int(r['env_steps'])):r for r in replay}
    result=[]
    for row in rows:
        if row['algorithm']!='mbpo':continue
        step=int(row['restricted_steps']); key=(int(row['seed']),step)
        if key not in index:raise ValueError('Missing replay accounting at confirmation/budget')
        result.append(dict(index[key],confirmed=bool(row['confirmed']),accounting_endpoint='confirmation' if row['confirmed'] else 'censoring_budget'))
    return result


def equivalent_pose_metrics(rows, pose_tolerance=1e-5):
    """Power agreement for action aliases: same executed orientation, same weather."""
    groups={}
    for r in rows:groups.setdefault((r['scenario_id'],r['step']),[]).append(r)
    diff=[]; actual=[]
    for block in groups.values():
        for i,r in enumerate(block):
            for q in block[:i]:
                a=np.array([r[k] for k in ('next_tilt_norm','next_azimuth_sin','next_azimuth_cos')]); b=np.array([q[k] for k in ('next_tilt_norm','next_azimuth_sin','next_azimuth_cos')])
                if np.max(np.abs(a-b))<=pose_tolerance:
                    diff.append((r['predicted_next_power_norm']-q['predicted_next_power_norm'])*2000.)
                    actual.append((r['actual_next_power_norm']-q['actual_next_power_norm'])*2000.)
    return {'n_equivalent_pose_pairs':len(diff),'alias_predicted_power_rmse_w':float(np.sqrt(np.mean(np.square(diff)))) if diff else None,
            'alias_actual_power_max_abs_difference_w':float(np.max(np.abs(actual))) if actual else None,'pose_tolerance_normalized':pose_tolerance}


def verify_loaded_policies(summary, identity):
    roles=('mbpo',) if identity['mode']=='model' else ('sac','mbpo')
    expected={role:{'sha256':identity[role+'_checkpoint_sha256'],'seed':identity['seed'],'epoch':identity['final_epoch']} for role in roles}
    if summary.get('loaded_policies')!=expected:raise ValueError('Evaluation did not use exact intended checkpoint identities')


def check_snapshot(path, algorithm, seed, epoch, steps, updates, config_hash, protocol_hash, physics_hash):
    import torch
    s=torch.load(path,map_location='cpu',weights_only=False)
    meta=s['metadata']; expected={'algorithm':algorithm,'seed':seed,'epoch':epoch,'env_steps':steps,'updates':updates,
        'config_sha256':config_hash,'scenario_protocol_sha256':protocol_hash,'resolved_mdp_sha256':physics_hash}
    if any(meta.get(k)!=v for k,v in expected.items()) or not s.get('actor_state_dict'):
        raise ValueError('Periodic snapshot metadata differs from evaluation identity')
    if any(not torch.isfinite(v).all() for v in s['actor_state_dict'].values()):raise ValueError('Nonfinite actor snapshot')


def completed_training_files(run_dir, meta):
    """Seal training/report inputs and ensure exact scheduled evaluation completeness."""
    root=Path(run_dir); h=int(meta['horizon']); steps=expected_steps(meta)
    epochs=[int(s)//h for s in steps]
    files=['config.json','resolved_mdp.json','scenario_split.json','episodes.csv','training_identity.json',
           'periodic_evaluation/protocol.json','periodic_evaluation/evaluation_references.csv','periodic_evaluation/evaluations.csv','periodic_evaluation/evaluation_days.csv']
    evdir=root/'periodic_evaluation'; snapdir=root/'policy_snapshots'
    from mbpo.audit_support import digest
    def rows(name):
        with (evdir/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
    ag=rows('evaluations.csv'); daily=rows('evaluation_days.csv'); ids=set(meta['validation_scenario_ids'])
    if sorted(int(r['epoch']) for r in ag)!=epochs or set(int(r['epoch']) for r in daily)!=set(epochs):raise ValueError('Incomplete scheduled evaluation grid')
    with (root/'episodes.csv').open(newline='',encoding='utf-8') as f: history=list(csv.DictReader(f))
    if len(history)!=meta['final_epoch'] or any(int(r['epoch'])!=i or int(r['env_steps'])!=i*h for i,r in enumerate(history,1)):raise ValueError('Incomplete committed training history')
    for e in epochs:
        a=[r for r in ag if int(r['epoch'])==e]; d=[r for r in daily if int(r['epoch'])==e]
        if len(d)!=len(ids) or {r['scenario_id'] for r in d}!=ids or int(a[0]['updates'])!=int(history[e-1]['updates']):raise ValueError('Periodic evaluation day/update count mismatch')
        if any(r['algorithm']!=meta['algorithm'] or int(r['seed'])!=meta['seed'] or int(r['env_steps'])!=e*h for r in a+d):raise ValueError('Periodic evaluation run identity mismatch')
        snapshot=f"policy_snapshots/{meta['algorithm']}_seed{meta['seed']}_epoch{e:05d}.pt"
        marker=f'periodic_evaluation/complete_{e:05d}.json'
        expected={'epoch':e,'env_steps':e*h,'days':digest(d),'aggregate':digest(a),'snapshot_sha256':sha_file(root/snapshot)}
        if read_json(root/marker)!=expected:raise ValueError('Periodic evaluation completion marker mismatch')
        files.extend([snapshot,marker])
    return {name:sha_file(root/name) for name in files}


def check_final_checkpoint(run_dir, meta):
    """Verify final state identity and equality of final evaluated and saved actors once at completion."""
    import torch
    root=Path(run_dir); ck=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
    epoch=int(meta['final_epoch']); steps=int(meta['final_env_steps']); algorithm=meta['algorithm']; seed=int(meta['seed'])
    checkpoint_identity(ck,algorithm,seed,epoch,steps)
    if ck.get('training_identity_sha256')!=sha_file(root/'training_identity.json') or ck.get('scenario_protocol_sha256')!=meta['scenario_protocol_sha256']:
        raise ValueError('Final checkpoint provenance differs from run')
    from mbpo.audit_support import episode_commit
    if ck.get('episode_commit')!=episode_commit(root/'episodes.csv',epoch,steps,int(meta['horizon']),updates=ck['agent']['updates']):raise ValueError('Final checkpoint does not commit the current training history')
    path=root/'policy_snapshots'/f'{algorithm}_seed{seed}_epoch{epoch:05d}.pt'
    snapshot=torch.load(path,map_location='cpu',weights_only=False); actual=snapshot['actor_state_dict']; expected=ck['agent']['actor']
    if set(actual)!=set(expected) or any(not torch.equal(actual[k],expected[k]) for k in expected):raise ValueError('Final evaluated actor differs from checkpoint actor')


def stratified_coverage(actual, quantiles, next_states, metadata):
    """Descriptive empirical coverage per historical day and next-observation irradiance regime."""
    y=np.asarray(actual); q=np.asarray(quantiles); x=np.asarray(next_states)
    if q.shape!=(4,len(y),3) or len(metadata)!=len(y) or len(x)!=len(y):raise ValueError('Invalid stratified diagnostic arrays')
    # DNI+DHI is a transparent proxy, not horizontal irradiance. Defined in advance.
    proxy=(x[:,0]+x[:,1])*2000.
    ids=np.asarray([r['scenario_id'] for r in metadata]); groups={}
    for sid in sorted(set(ids)):groups['day:'+str(sid)]=ids==sid
    groups.update({'irradiance_proxy_0_to_100':proxy<100,'irradiance_proxy_100_to_500':(proxy>=100)&(proxy<500),'irradiance_proxy_500_plus':proxy>=500})
    out={}
    for label,mask in groups.items():
        if not np.any(mask):continue
        out[label]={'n':int(mask.sum()),'coverage90':np.mean((y[mask]>=q[1,mask])&(y[mask]<=q[2,mask]),axis=0).tolist(),'coverage95':np.mean((y[mask]>=q[0,mask])&(y[mask]<=q[3,mask]),axis=0).tolist()}
    return {'channels':['dni_norm','dhi_norm','power_norm'],'groups':out,'scope':'fixed held-out realization; irradiance proxy DNI+DHI W/m2; descriptive, not independent transitions'}
