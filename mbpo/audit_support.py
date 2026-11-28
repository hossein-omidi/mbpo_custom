"""Geometry10 reliability contracts. No physical or optimizer changes."""
from __future__ import annotations
import csv, hashlib, json, os, sys, tempfile
from pathlib import Path
from contextlib import contextmanager
from importlib import metadata
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_bytes(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.audit-', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_json(path, obj):
    atomic_bytes(path, json.dumps(obj, sort_keys=True, indent=2, allow_nan=False).encode())


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def runtime_identity(root=ROOT, kind='training'):
    """Hash active source bytes, not Git status, outputs or archived installers."""
    root = Path(root)
    physics = ['mbpo/geometry10.py', 'SAC_workspace/src/environment/environment.py',
               'SAC_workspace/src/representation/representation.py']
    physics += ['mbpo/env/'+name+'.py' for name in ('__init__','baseline_contract','historical_weather','nsrdb_iotools','nsrdb_weather','pv_power_backend','pv_tracking','pvlib_physics','stage_a_data_contract','stage_a_pv_tracking','trajectory_uncertainty')]
    policy = physics + ['SAC_workspace/src/agents/sac_agent.py']
    config = ['SAC_workspace/src/config/config.py','MBPO_workspace/src/config.py']
    config += ['examples/config/pv_tracking/'+name for name in ('0.py','_nsrdb_base.py','conf_geometry10.py')]
    common = policy + config + ['mbpo/audit_support.py','mbpo/audit_release.py']
    if kind == 'training':
        paths = common + ['SAC_workspace/src/training/trainer.py','SAC_workspace/src/training/replay_buffer.py',
            'SAC_workspace/src/training/experiment_protocol.py','MBPO_workspace/src/trainer.py',
            'MBPO_workspace/src/dynamics.py','MBPO_workspace/src/replay.py',
            'MBPO_workspace/workflows/08_monte_carlo_train.py','PV_experiments/core.py','PV_experiments/train.py',
            'PostProcessing_MC_patch/tools/verify_mc_environment.py']
    elif kind in ('physical','model','evaluation'):
        paths = (physics + config + ['mbpo/audit_support.py','mbpo/audit_release.py'] if kind=='physical' else common) + ['SAC_workspace/src/training/experiment_protocol.py']
        if kind!='physical': paths += ['PV_experiments/evaluate.py']
        if kind == 'model': paths += ['MBPO_workspace/src/dynamics.py','MBPO_workspace/src/replay.py',
            'SAC_workspace/src/training/replay_buffer.py','MBPO_workspace/workflows/06_model_validation.py']
        elif kind == 'evaluation': paths += ['MBPO_workspace/workflows/09_monte_carlo_evaluate.py']
        else: paths += ['PostProcessing_MC_patch/tools/verify_mc_environment.py']
    else: raise ValueError(kind)
    # The active config loader reads 0.py as a literal, not an executable legacy runner.
    versions = {}
    for name in ('numpy','pandas','torch','pvlib','gym','scipy'):
        try: versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError: versions[name] = None
    return {'schema':1,'kind':kind,'python':list(sys.version_info[:3]),'dependencies':versions,
            'sources':{p:sha_file(root/p) for p in sorted(set(paths))}}


def dataset_identity(root=ROOT):
    from SAC_workspace.src.config.config import load_config
    from mbpo.env.stage_a_data_contract import nsrdb_dataset_fingerprint
    cfg = load_config(require_data=True)
    path = Path(cfg.manifest)
    return nsrdb_dataset_fingerprint(path if path.is_absolute() else Path(root)/path)


def bind_run(outdir, resume, resolved_mdp):
    path = Path(outdir)/'training_identity.json'
    identity = {'runtime':runtime_identity(), 'physics':resolved_mdp}
    expected=os.environ.get('PV_AUDIT_EXPECTED_RUNTIME')
    if expected is not None and digest(identity['runtime'])!=expected:
        raise ValueError('Managed runtime differs from frozen plan before training')
    expected_data=os.environ.get('PV_AUDIT_EXPECTED_DATASET')
    if expected_data is not None:
        expected_data=json.loads(expected_data)
        if any(resolved_mdp.get(k)!=v for k,v in expected_data.items()):
            raise ValueError('Managed data differs from frozen plan before training')
    if resume:
        if not path.is_file() or read_json(path) != identity:
            raise ValueError('Training source/physics identity missing or changed; use a fresh experiment')
    else:
        if path.exists(): raise FileExistsError(path)
        atomic_json(path, identity)


def episode_commit(path, epoch, steps, horizon, *, updates=None):
    """Checkpoint commits exact CSV bytes; a zero-epoch checkpoint has empty history."""
    path=Path(path)
    if path.exists():
        with path.open('rb') as f:
            os.fsync(f.fileno()); data=f.read()
    else: data=b''
    _validate_episode_prefix(data, epoch, steps, horizon, updates=updates)
    return {'size':len(data), 'sha256':hashlib.sha256(data).hexdigest()}


def _validate_episode_prefix(data, epoch, steps, horizon, *, updates=None):
    if epoch < 0 or steps != epoch*horizon: raise ValueError('Invalid checkpoint horizon/count')
    if epoch == 0:
        if data or updates not in (None,0): raise ValueError('Epoch-zero checkpoint must have empty history and zero updates')
        return
    if not data.endswith(b'\n'): raise ValueError('Committed episode history has partial final row')
    rows=list(csv.DictReader(data.decode('utf-8').splitlines()))
    if len(rows)!=epoch: raise ValueError('Committed episode count differs from checkpoint')
    for i,row in enumerate(rows,1):
        if int(row['epoch'])!=i or int(row['env_steps'])!=i*horizon:
            raise ValueError('Nonchronological committed episode history')
    if updates is not None:
        counts=[int(row['updates']) for row in rows]
        if min(counts)<0 or any(a>b for a,b in zip(counts,counts[1:])) or counts[-1]!=updates:
            raise ValueError('Committed optimizer update count differs from checkpoint')


def recover_history(path, commit, epoch, steps, horizon, *, updates=None):
    path=Path(path); data=path.read_bytes() if path.exists() else b''
    if not isinstance(commit,dict): raise ValueError('Checkpoint lacks committed history hash; cannot safely resume')
    size=int(commit['size']); prefix=data[:size]
    if size<0 or len(prefix)!=size or hashlib.sha256(prefix).hexdigest()!=commit['sha256']:
        raise ValueError('Committed history prefix missing or corrupted; refusing recovery')
    _validate_episode_prefix(prefix, epoch, steps, horizon, updates=updates)
    tail=data[size:]
    ledger=path.parent/'recovery_events.json'
    rows=read_json(ledger) if ledger.exists() else []
    if tail:
        # Count only a contiguous sequence of fully written, chronological rows.
        n=0
        try:
            lines=tail.splitlines(keepends=True)
            header=prefix.splitlines()[0] if epoch else lines.pop(0)
            fields=next(csv.reader([header.decode('utf-8').strip()]))
            for line in lines:
                if not line.endswith(b'\n'): break
                values=next(csv.reader([line.decode('utf-8').strip()]))
                if len(values)!=len(fields): break
                row=dict(zip(fields,values)); j=epoch+n+1
                if int(row['epoch'])!=j or int(row['env_steps'])!=j*horizon: break
                n+=1
        except (ValueError,KeyError,IndexError,UnicodeError,csv.Error): pass
        event={'epoch':epoch,'discarded_tail_sha256':hashlib.sha256(tail).hexdigest(),
               'discarded_complete_csv_rows':n,'known_repeated_transitions':n*horizon,'status':'pending'}
        if not rows or rows[-1]!=event:
            rows.append(event); atomic_json(ledger,rows)
        if prefix: atomic_bytes(path,prefix)
        else: path.unlink(missing_ok=True)
    elif epoch==0 and path.exists(): path.unlink()
    if rows and rows[-1].get('status')=='pending':
        if rows[-1]['epoch']!=epoch: raise ValueError('Unresolved recovery transaction for another checkpoint')
        rows[-1]['status']='applied'; atomic_json(ledger,rows)


@contextmanager
def exclusive_lock(path):
    """Linux/WSL lock; never unlink the inode. Pass fd to the managed child."""
    import fcntl
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
    try:
        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError(f'Active job holds {path}') from exc
        yield fd
    finally:
        # Do not LOCK_UN: a child inheriting this open-file description may be alive.
        os.close(fd)


def expected_steps(meta):
    h=int(meta['horizon']); interval=int(meta['evaluation_interval_episodes']); epochs=int(meta['final_epoch'])
    if min(h,interval,epochs)<1 or h*epochs!=int(meta['final_env_steps']): raise ValueError('Invalid evaluation schedule')
    return np.array(sorted(set(range(interval,epochs+1,interval))|{epochs}),dtype=np.int64)*h


def target_event(curve, steps, target, sustain):
    y=np.asarray(curve,float); x=np.asarray(steps)
    if y.ndim!=1 or x.shape!=y.shape or not np.isfinite(y).all() or not np.isfinite(x).all() or np.any(np.diff(x)<=0) or not np.isfinite(target) or sustain<1: raise ValueError('Invalid target event inputs')
    streak=0
    for j,value in enumerate(y):
        streak=streak+1 if value>=target else 0
        if streak>=sustain: return int(x[j-sustain+1]),int(x[j])
    return None,None


def power_envelope(states, area, efficiency, gamma, *, cell_min=-60., residual_allowance=200.):
    """Operational diagnostic ONLY: isotropic POA, rho <= 1, finite assumed T/closure range.
    These assumptions are not guaranteed under Gaussian/lognormal weather. No filtering.
    """
    s=np.asarray(states,float); dni,dhi=s[:,0]*2000.,s[:,1]*2000.
    b=s[:,3]*np.pi/2; z=s[:,6]*np.pi
    cosaz=s[:,4]*s[:,7]+s[:,5]*s[:,8]
    cosi=np.cos(z)*np.cos(b)+np.sin(z)*np.sin(b)*cosaz
    ghi=dhi+dni*np.maximum(np.cos(z),0)+residual_allowance
    poa=dni*np.maximum(cosi,0)+dhi*(1+np.cos(b))/2+ghi*(1-np.cos(b))/2
    return area*efficiency*poa*max(0.,1+gamma*(cell_min-25.))


def envelope_counts(states, env):
    s=np.asarray(states,float)
    upper=power_envelope(s,float(env.area),float(env.efficiency),
                        0. if env.power_model=='legacy' else float(env.gamma_pdc))
    saturated=(s[:,:2]>=1.-1e-7).any(axis=1)
    violation=(s[:,2]*2000.>upper+1e-3)&~saturated
    return {'envelope_checked':int((~saturated).sum()),'envelope_violations':int(violation.sum()),
            'envelope_irradiance_saturated':int(saturated.sum())}


def counterfactual_metrics(rows, tolerance_kwh=1e-7):
    """One-step realized reward diagnostics, grouped by identical starting state."""
    groups={}
    for row in rows: groups.setdefault((row['scenario_id'],row['step']),[]).append(row)
    errors=[]; regret=[]; correct=total=0; ties=0
    for block in groups.values():
        y=np.array([r['actual_reward_kwh'] for r in block]); p=np.array([r['predicted_reward_kwh'] for r in block])
        hold=[i for i,r in enumerate(block) if r['action_type']=='hold']
        if len(hold)!=1: raise ValueError('Counterfactual group needs one hold action')
        j=hold[0]; others=np.arange(len(block))!=j
        errors.extend(((p-p[j])-(y-y[j]))[others]); regret.append(float(y.max()-y[np.argmax(p)]))
        for i in range(len(y)):
            for k in range(i):
                d=y[i]-y[k]; e=p[i]-p[k]
                if abs(d)<=tolerance_kwh: ties+=1; continue
                total+=1; correct+=int(abs(e)>tolerance_kwh and np.sign(d)==np.sign(e))
    return {'n_states':len(groups),'reward_difference_rmse_kwh':float(np.sqrt(np.mean(np.square(errors)))),
            'mean_one_step_regret_kwh':float(np.mean(regret)), 'pairwise_agreement':correct/total if total else None,
            'non_tied_pairs':total,'true_near_ties':ties,'tie_tolerance_kwh':tolerance_kwh}


def hierarchical(ensemble, states, actions, *, seed, samples=128, chunk=32,
                 envelope_template=None, env=None, envelope_totals=None):
    """Marginal clipped delta moments/quantiles using independent diagnostic RNG.
    Fixed seed and identical shape give common weather and member/noise draws across actions.
    """
    states=np.asarray(states,np.float32); actions=np.asarray(actions,np.float32)
    if samples<40 or chunk<1 or len(states)==0: raise ValueError('Invalid diagnostic sampling grid')
    rng=np.random.default_rng(seed); means=[]; sds=[]; quantiles=[]
    for i in range(0,len(states),chunk):
        x=states[i:i+chunk]; a=actions[i:i+chunk]; n=len(x)
        d,_=ensemble.predict_delta(np.tile(x,(samples,1)),np.tile(a,(samples,1)),stochastic=True,rng=rng,record=False)
        d=np.asarray(d,dtype=float).reshape(samples,n,3)
        if not np.isfinite(d).all(): raise FloatingPointError('Nonfinite diagnostic predictions')
        if envelope_totals is not None:
            if env is None or envelope_template is None: raise ValueError('Envelope diagnostics need next geometry and physics')
            nexts=np.tile(envelope_template[i:i+chunk],(samples,1))
            nexts[:,:3]=np.tile(x[:,:3],(samples,1))+d.reshape(-1,3)
            for k,v in envelope_counts(nexts,env).items(): envelope_totals[k]=envelope_totals.get(k,0)+v
        means.append(d.mean(axis=0)); sds.append(d.std(axis=0,ddof=1)); quantiles.append(np.quantile(d,[.025,.05,.95,.975],axis=0))
    return np.concatenate(means),np.concatenate(sds),np.concatenate(quantiles,axis=1)


def conditional_power_rmse(ensemble, states, actions, next_states):
    """Offline teacher-forced electrical-response diagnostic, never used for rollouts."""
    mu,_=ensemble._members(ensemble.power_nets,ensemble.power_elites,
        ensemble._power_features(states,actions,next_states[:,:2]),ensemble.power_xmean,
        ensemble.power_xstd,ensemble.power_ymean,ensemble.power_ystd)
    pred=np.clip(states[:,2]+mu.mean(0)[:,0],0.,1.)
    return float(np.sqrt(np.mean((pred-next_states[:,2])**2)))


def diagnostic_seed(*values):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,values)).encode()).digest()[:8],'little')


@contextmanager
def preserve_global_rng():
    """Periodic evaluation must not advance training's process-wide RNG streams."""
    import random, torch
    py=random.getstate(); nr=np.random.get_state(); tr=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try: yield
    finally:
        random.setstate(py); np.random.set_state(nr); torch.set_rng_state(tr)
        if cuda is not None: torch.cuda.set_rng_state_all(cuda)
