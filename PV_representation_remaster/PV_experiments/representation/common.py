from __future__ import annotations
import hashlib, json, os, tempfile, shutil
from pathlib import Path
from contextlib import contextmanager
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from .statistics import LABELS,COLORS


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def atomic_json(path,obj):
    path=Path(path);fd,name=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


@contextmanager
def collection_session(target):
    """Resumable, exclusive per-collection lock; unlike plots, keep completed work."""
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    lock=target.parent/('.'+target.name+'.lock');fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    try:
        target.mkdir(exist_ok=True)
        yield target
    finally:lock.unlink(missing_ok=True)


def save_csv(out,name,data):
    df=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
    if df.empty:raise ValueError(f'No records for {name}')
    df.to_csv(out/name,index=False,float_format='%.12g')


class Figures:
    def __init__(self,out):self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.manifest=[]
    def save(self,fig,name,question,estimand,source):
        if not getattr(fig,"_pv_layout_done",False):fig.tight_layout()
        try:
            for ext in ('pdf','png'):fig.savefig(self.out/f'{name}.{ext}',dpi=220,bbox_inches='tight')
        finally:plt.close(fig)
        self.manifest.append(dict(figure=name,question=question,estimand=estimand,source=source))
    def finish(self):write_json(self.out/'figure_manifest.json',self.manifest)


def style(ax):ax.grid(alpha=.22);ax.spines[['top','right']].set_visible(False)


def line(ax,x,y,method,sd=None):
    import numpy as np
    color=COLORS.get(method,'#333333');ax.plot(x,y,label=LABELS.get(method,method),color=color,lw=1.6,ls='--' if 'ideal' in method else '-')
    if sd is not None and np.isfinite(sd).all():ax.fill_between(x,y-sd,y+sd,color=color,alpha=.12,linewidth=0)


@contextmanager
def new_output(target):
    """No destructive reuse. Publish only complete output directories."""
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(f'Output exists: {target}; use a new --tag')
    lock=target.parent/('.'+target.name+'.lock')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.close(fd)
    tmp=None
    try:
        tmp=Path(tempfile.mkdtemp(prefix='.'+target.name+'-',dir=target.parent))
        yield tmp
        if target.exists():raise FileExistsError(target)
        tmp.rename(target)
    finally:
        if tmp is not None and tmp.exists():shutil.rmtree(tmp)
        lock.unlink(missing_ok=True)


def verify_runs(cfg,seeds,root):
    from PV_experiments.core import verify_pair,run_path
    from SAC_workspace.src.training.experiment_protocol import canonical_json_sha256
    splits=[verify_pair(cfg,s,root) for s in seeds]
    if any(x!=splits[0] for x in splits[1:]):raise ValueError('Frozen partitions differ across training seeds')
    shared=[]
    for seed in seeds:
        for alg in ('sac','mbpo'):
            p=run_path(cfg,alg,seed,root);m=read_json(p/'run_metadata.json')
            resolved=read_json(p/'resolved_mdp.json')
            if canonical_json_sha256(resolved)!=m['resolved_mdp_sha256']:raise ValueError('Resolved MDP metadata does not match recorded fingerprint')
            if m['movement_cost_mode']!='geometry' or m['movement_penalty']!=1.0:raise ValueError('Expected enabled physical movement cost (geometry, gain 1)')
            shared.append(tuple(m[k] for k in ('resolved_mdp_sha256','shared_sac_config_sha256','scenario_protocol_sha256','horizon','energy_objective','power_model')))
    if len(set(shared))!=1:raise ValueError('Run physics/configuration differ across seeds')
    return splits[0]
