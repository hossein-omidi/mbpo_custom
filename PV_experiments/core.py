"""Frozen experiment settings, automatic discovery and safe execution.

No model algorithms or PV simulation equations are modified. External work is
performed exclusively through the existing project workflows. No backups,
network access, Git commands, or implicit replacement of existing artifacts.
"""
from __future__ import annotations
from mbpo.audit_support import runtime_identity, dataset_identity, exclusive_lock, digest
from mbpo.geometry10 import MODE, STATE_DIM

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA=1
DEFAULT_UNCERTAINTY='PostProcessing_MC_patch/examples/hypothetical_uncertainty_5pct.json'
DEFAULT_ROOT='outputs/paper_experiments'
PROJECT_ROOT=Path(__file__).resolve().parents[1]
IDENTIFIER=re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')


def require_repo(root: Path=PROJECT_ROOT):
    for relative in ('MBPO_workspace/workflows/08_monte_carlo_train.py',
                     'MBPO_workspace/workflows/09_monte_carlo_evaluate.py',
                     'MBPO_workspace/workflows/05_sample_complexity.py',
                     'MBPO_workspace/workflows/06_model_validation.py',
                     'SAC_workspace/src/training/trainer.py',
                     'MBPO_workspace/src/trainer.py'):
        if not (root/relative).is_file():
            raise RuntimeError(f'Missing required existing workflow: {root/relative}. Install in the PostProcessing repository root.')


def parse_seeds(value: str | None) -> list[int] | None:
    if value is None:return None
    if ':' in value:
        if value.count(':')!=1:raise ValueError('Seed range must be START:END (inclusive)')
        a,b=map(int,value.split(':'))
        if a<0 or b<a or b-a>99:raise ValueError('Use a nonnegative seed range of at most 100 seeds')
        return list(range(a,b+1))
    items=[int(x.strip()) for x in value.split(',')]
    if not items or any(x<0 for x in items) or len(set(items))!=len(items):
        raise ValueError('Seeds must be distinct nonnegative integers')
    return sorted(items)


def validate_name(name):
    if not isinstance(name,str) or not IDENTIFIER.fullmatch(name):
        raise ValueError('Experiment name must have 1–64 characters: letters, digits, underscore, hyphen; first character alphanumeric')
    return name


def utcnow():return datetime.now(timezone.utc).isoformat(timespec='seconds')


def canonical(data):return json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def sha256(data):
    if isinstance(data,str):data=data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):return json.loads(path.read_text(encoding='utf-8'))


def atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.pv_experiment_',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def base_path(name:str, root:Path=PROJECT_ROOT)->Path:
    validate_name(name)
    return root/DEFAULT_ROOT/name


def config_path(name:str,root:Path=PROJECT_ROOT)->Path:
    return base_path(name,root)/'experiment.json'


def load_experiment(name:str,root:Path=PROJECT_ROOT):
    require_repo(root)
    path=config_path(name,root)
    if not path.is_file():raise FileNotFoundError(f'Experiment not initialized: {path}. Run PV_experiments.train first.')
    cfg=read_json(path)
    if cfg.get('schema')!=SCHEMA or cfg.get('name')!=name or cfg.get('observation_mode')!=MODE or cfg.get('state_dim')!=STATE_DIM:
        raise ValueError('Invalid experiment configuration schema or name')
    if sorted(cfg.get('seeds',[]))!=cfg.get('seeds') or not cfg['seeds']:
        raise ValueError('Invalid experiment seed plan')
    frozen=base_path(name,root)/'uncertainty_spec.json'
    if not frozen.is_file() or sha256(canonical(read_json(frozen)))!=cfg.get('uncertainty_sha256'):
        raise ValueError('Frozen uncertainty specification changed or was removed. Never silently mix experimental assumptions.')
    if cfg.get('dataset_identity') is not None and cfg['dataset_identity']!=dataset_identity(root):
        raise ValueError('Current data differ from frozen experiment')
    return cfg


def initialize_experiment(name:str, *, seeds:list[int] | None=None, epochs:int | None=None,
    uncertainty_file:str | None=None, device:str='cpu',model_epochs:int | None=None,
    model_max_train_samples:int | None=None,root:Path=PROJECT_ROOT,dry_run:bool=False):
    require_repo(root)
    path=config_path(name,root)
    requested=sorted(seeds) if seeds is not None else None
    if path.exists():
        cfg=load_experiment(name,root)
        if not dry_run and (cfg.get('training_runtime')!=runtime_identity(root) or cfg.get('dataset_identity')!=dataset_identity(root)):
            raise ValueError('Experiment source/dataset changed or predates audit patch; use a new experiment name')
        if device is not None and cfg['device']!=device:
            raise ValueError('Device differs from frozen experiment configuration; use a new experiment name')
        for field,value in [('epochs',epochs),('model_epochs',model_epochs),
                            ('model_max_train_samples',model_max_train_samples)]:
            if value is not None and cfg.get(field)!=value:
                raise ValueError(f'Frozen experiment {field}={cfg.get(field)!r}; requested {value!r}. Use a new experiment name.')
        if requested is not None and not set(requested).issubset(cfg['seeds']):
            raise ValueError('Requested seeds must belong to the frozen experiment seed set')
        if uncertainty_file is not None:
            supplied=read_json((root/uncertainty_file).resolve())
            if sha256(canonical(supplied))!=cfg['uncertainty_sha256']:
                raise ValueError('Uncertainty settings differ from frozen experiment. Use a new experiment name.')
        return cfg
    seeds=requested if requested is not None else list(range(5))
    if not seeds or len(set(seeds))!=len(seeds) or any(x<0 for x in seeds):
        raise ValueError('A nonempty set of distinct nonnegative training seeds is required')
    epochs=2000 if epochs is None else int(epochs)
    if epochs<=0:raise ValueError('epochs must be positive')
    for label,val in [('model_epochs',model_epochs),('model_max_train_samples',model_max_train_samples)]:
        if val is not None and val<=0:raise ValueError(f'{label} must be positive')
    inp=uncertainty_file or DEFAULT_UNCERTAINTY
    p=(root/inp).resolve()
    if not p.is_file():raise FileNotFoundError(f'Missing uncertainty specification: {p}')
    spec=read_json(p)
    if spec.get('enabled') is not True:raise ValueError('Experiment uncertainty spec must explicitly enable perturbations')
    # Underlying original training workflow performs the complete mathematical
    # schema check. Freeze the exact JSON data, never silently copy a newer file.
    data={
        'schema':SCHEMA,'name':name,'created_utc':utcnow(),
        'observation_mode':MODE,'state_dim':STATE_DIM,
        'seeds':seeds,'epochs':epochs,'device':device or 'cpu',
        'model_epochs':model_epochs,'model_max_train_samples':model_max_train_samples,
        'uncertainty_sha256':sha256(canonical(spec)),
        'uncertainty_original_input':inp,
        'run_layout':'runs/{algorithm}_seed{seed}',
        'repository_workflows':{'training':'MBPO_workspace.workflows.08_monte_carlo_train',
             'evaluation':'MBPO_workspace.workflows.09_monte_carlo_evaluate',
             'sample_complexity':'MBPO_workspace.workflows.05_sample_complexity',
             'model_validation':'MBPO_workspace.workflows.06_model_validation'},
    }
    if not dry_run:
        data['training_runtime']=runtime_identity(root)
        data['dataset_identity']=dataset_identity(root)
        dest=base_path(name,root)
        dest.mkdir(parents=True,exist_ok=True)
        specpath=dest/'uncertainty_spec.json'
        if specpath.exists():raise FileExistsError(f'Unowned experiment settings file exists: {specpath}')
        atomic_json(specpath,spec)
        atomic_json(path,data)
    return data


def experiment_dir(cfg,root=PROJECT_ROOT):return base_path(cfg['name'],root)


def uncertainty_path(cfg,root=PROJECT_ROOT):return experiment_dir(cfg,root)/'uncertainty_spec.json'


def run_path(cfg,algorithm,seed,root=PROJECT_ROOT):
    if algorithm not in ('sac','mbpo'):raise ValueError('Unknown algorithm')
    if seed not in cfg['seeds']:raise ValueError('Unknown seed for frozen experiment')
    return experiment_dir(cfg,root)/'runs'/f'{algorithm}_seed{seed}'


def selected_seeds(cfg,seeds):
    if seeds is None:return list(cfg['seeds'])
    if not seeds or not set(seeds).issubset(cfg['seeds']):raise ValueError('Selected seeds are not a nonempty subset of experiment seed plan')
    return sorted(seeds)


def inspect_training_run(cfg,algorithm,seed,root=PROJECT_ROOT):
    d=run_path(cfg,algorithm,seed,root)
    if not d.exists():return 'missing',None
    p=d/'run_metadata.json'
    if not p.is_file():return 'incomplete',None
    meta=read_json(p)
    if meta.get('seed')!=seed or meta.get('algorithm')!=algorithm or meta.get('run_schema')!=2:
        raise ValueError(f'Run metadata identity mismatch: {p}')
    if meta.get('observation_mode')!=MODE or meta.get('state_dim')!=STATE_DIM:
        raise ValueError(f'Run uses an incompatible observation contract: {d}. Use a fresh experiment name.')
    identity_path=d/'training_identity.json'
    if cfg.get('training_runtime') is not None:
        identity=read_json(identity_path) if identity_path.is_file() else None
        if identity is None or identity.get('runtime')!=cfg['training_runtime'] or any(identity.get('physics',{}).get(k)!=v for k,v in cfg['dataset_identity'].items()):
            raise ValueError(f'Run source/dataset identity differs from frozen experiment: {d}')
    if meta.get('training_status')=='completed':
        if cfg.get('training_runtime') is not None:
            from mbpo.audit_support import sha_file
            if meta.get('final_checkpoint_sha256')!=sha_file(d/'latest.pt') or meta.get('training_identity_sha256')!=sha_file(identity_path):
                raise ValueError(f'Completed checkpoint or training identity changed: {d}')
            from mbpo.audit_release import completed_training_files
            if meta.get('training_output_sha256')!=completed_training_files(d,meta):
                raise ValueError(f'Completed training artifacts changed: {d}')
            if meta.get('final_env_steps')!=meta.get('horizon')*cfg['epochs']:
                raise ValueError(f'Completed run interaction budget differs: {d}')
        if meta.get('final_epoch')!=cfg['epochs']:
            raise ValueError(f'Completed run has unexpected training budget: {d}')
        for file in ('latest.pt','scenario_split.json','episodes.csv','config.json'):
            if not (d/file).is_file():raise ValueError(f'Completed run lacks {file}: {d}')
        return 'completed',meta
    return 'incomplete',meta


def verify_pair(cfg,seed,root=PROJECT_ROOT):
    sac_status,sac=inspect_training_run(cfg,'sac',seed,root)
    mbpo_status,mbpo=inspect_training_run(cfg,'mbpo',seed,root)
    if sac_status!='completed' or mbpo_status!='completed':
        raise ValueError(f'Seed {seed}: complete SAC and MBPO runs required (SAC={sac_status}; MBPO={mbpo_status})')
    shared=('scenario_protocol_sha256','resolved_mdp_sha256','shared_sac_config_sha256',
            'validation_scenario_ids','test_scenario_ids','horizon','state_dim','action_dim','observation_mode',
            'power_model','movement_penalty','movement_cost_mode','energy_objective')
    for k in shared:
        if sac.get(k)!=mbpo.get(k):raise ValueError(f'Seed {seed}: SAC and MBPO differ in {k}')
    if sac.get('horizon')!=195 or sac.get('state_dim')!=STATE_DIM or sac.get('observation_mode')!=MODE or sac.get('action_dim')!=2:
        raise ValueError('Training metadata incompatible with inspected 195-step geometry10, 2-action simulator')
    sp=read_json(run_path(cfg,'sac',seed,root)/'scenario_split.json')
    mp=read_json(run_path(cfg,'mbpo',seed,root)/'scenario_split.json')
    if sp!=mp or sp.get('protocol_sha256')!=sac.get('scenario_protocol_sha256'):
        raise ValueError(f'Seed {seed}: frozen scenario partitions are inconsistent')
    partitions=[set(sp.get(k,[])) for k in ('train_ids','validation_ids','test_ids')]
    if any(not group for group in partitions):
        raise ValueError('An empty train/validation/test scenario partition is invalid')
    if any(partitions[i]&partitions[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError('Train/validation/test scenario leakage detected')
    if set(sac.get('validation_scenario_ids',[]))!=partitions[1] or set(sac.get('test_scenario_ids',[]))!=partitions[2]:
        raise ValueError('Saved metadata scenario IDs disagree with frozen scenario protocol')
    return sp


def job_output(cfg,kind,seed=None,root=PROJECT_ROOT):
    name=kind if seed is None else f'{kind}_seed{seed}'
    return experiment_dir(cfg,root)/'logs'/f'{name}.log'


def execute(cfg,kind,command,*,root=PROJECT_ROOT,seed=None,dry_run=False):
    """Invoke a verified existing CLI, streaming output and recording a job manifest."""
    if not all(isinstance(x,str) and x for x in command):raise TypeError('Invalid argument vector')
    if command[0]!=sys.executable or command[1:2]!=['-m']:
        raise ValueError('Only local, module-based existing workflow execution is supported')
    print('COMMAND:', ' '.join(repr(x) if ' ' in x else x for x in command),flush=True)
    if dry_run:return
    directory=experiment_dir(cfg,root)
    log=job_output(cfg,kind,seed,root)
    log.parent.mkdir(parents=True,exist_ok=True)
    key=log.stem
    job_path=directory/'jobs'/f'{key}.json'
    with exclusive_lock(job_path.with_suffix('.lock')) as lock_fd:
        return _execute_locked(cfg,kind,command,root,seed,directory,log,job_path,lock_fd)


def _execute_locked(cfg,kind,command,root,seed,directory,log,job_path,lock_fd):
    status={'schema':SCHEMA,'kind':kind,'seed':seed,'experiment':cfg['name'],
            'command':command,'status':'running','started_utc':utcnow(),
            'log_relative_path':log.relative_to(directory).as_posix()}
    atomic_json(job_path,status)
    proc=None
    try:
        with log.open('a',encoding='utf-8') as f:
            f.write('\n--- managed invocation ---\n'); f.flush()
            child_env=os.environ.copy()
            if kind.startswith('train_') and cfg.get('training_runtime') is not None:
                child_env['PV_AUDIT_EXPECTED_RUNTIME']=digest(cfg['training_runtime'])
                child_env['PV_AUDIT_EXPECTED_DATASET']=canonical(cfg['dataset_identity'])
            proc=subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,bufsize=1,pass_fds=(lock_fd,),env=child_env)
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line,end='',flush=True)
                f.write(line);f.flush()
            rc=proc.wait()
        status.update(status='completed' if rc==0 else 'failed',returncode=rc,finished_utc=utcnow())
        atomic_json(job_path,status)
        if rc:raise RuntimeError(f'{kind} failed (exit code {rc}); see {log}')
    except BaseException:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait()
        if status['status']=='running':
            status.update(status='failed',finished_utc=utcnow())
            atomic_json(job_path,status)
        raise


def command_override(cfg):
    ret=[]
    if cfg.get('model_epochs') is not None:ret+=['--model-epochs',str(cfg['model_epochs'])]
    if cfg.get('model_max_train_samples') is not None:ret+=['--model-max-train-samples',str(cfg['model_max_train_samples'])]
    return ret


def relative_run(cfg,alg,seed,root=PROJECT_ROOT):return str(run_path(cfg,alg,seed,root))


def require_files(paths):
    for p in paths:
        if not Path(p).is_file():raise FileNotFoundError(f'Missing expected artifact: {p}')
