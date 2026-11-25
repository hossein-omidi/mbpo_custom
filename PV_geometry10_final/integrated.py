#!/usr/bin/env python3
"""Build and apply one guarded Git patch to the existing Geometry10 checkout."""
from __future__ import annotations
import argparse, difflib, hashlib, subprocess, sys
from pathlib import Path
BASE='84322e2df3b786904631edfbaae84cf8b5d5aae3'
PAYLOAD=Path(__file__).resolve().parent/'payload'
PIN={
 'MBPO_workspace/src/dynamics.py':'d1b697a4706ada994486d1884c8019de7b5144db',
 'MBPO_workspace/src/replay.py':'1e6d188f8e6ff3f54f2c6cb6d45339086ad096cd',
 'MBPO_workspace/src/sample_complexity.py':'07323ab36a077da587fcf59911ee6321a90a9cfc',
 'MBPO_workspace/src/trainer.py':'364bfbf30b7fb4bf1b5527e0895607b607a92650',
 'MBPO_workspace/workflows/06_model_validation.py':'033cceab112d318745290e469a1812485dd22031',
 'PV_experiments/advanced_learning.py':'66ec214f4368f8c4a3a9ea7d58f89b6488617970',
 'PV_experiments/core.py':'a4addada9a8ce3f19531ff06a74ac6c386adc3d4',
 'PV_experiments/evaluate.py':'26f681b2159899877ee2fe0f23f1f2e7519990e2',
 'PV_experiments/report.py':'2f1bf852cdb7784abfff326e2ef3f1f86941843e',
 'PV_experiments/representation/training.py':'99aeebd7a09760cc323bfc19cc840dffe266bec3',
 'PV_experiments/train.py':'671f61eb0472568b2c7bb7c0c4d80f3a5ff7c464',
 'PostProcessing_MC_patch/tools/verify_mc_environment.py':'03b052c97009d3f008a076e2865360319eb5fe59',
 'SAC_workspace/src/training/experiment_protocol.py':'4c5c0dd0207ad1cd966e7855f87dfd20cb2005e3',
 'SAC_workspace/src/training/trainer.py':'99ef8bde096738a3c9a457ef259440037c35bd35',
 'SAC_workspace/workflows/04_evaluate.py':'283a3464e62bd84c66eafa6516fa0201800d9fd9',
 'MBPO_workspace/workflows/03_evaluate.py':'cd3b7bf7a645ce8b5479ed051a0f5c9804d64f7a',
}

def replace(s,a,b):
    if s.count(a)!=1: raise ValueError(f'Expected exactly one source anchor, found {s.count(a)}: {a[:160]!r}')
    return s.replace(a,b,1)

def between(s,a,b,new):
    start=s.index(a); end=s.index(b,start+len(a)); return s[:start]+new+s[end:]

def add_import(s,line): return replace(s,'from __future__ import annotations\n','from __future__ import annotations\n'+line+'\n')

def trainer(s,mbpo):
    s=add_import(s,'from mbpo.audit_support import bind_run, episode_commit, recover_history, envelope_counts, sha_file')
    p='self.history_path' if mbpo else 'self._episode_path'
    # Bind exact source/physics before restoring any state.
    if mbpo:
        s=replace(s,'            self.last_model_step = 0\n','            self.last_model_step = 0\n            bind_run(self.outdir, resume is not None, self.env.resolved_mdp)\n')
    else:
        s=replace(s,'            self.epoch, self.env_steps = 0, 0\n','            self.epoch, self.env_steps = 0, 0\n            bind_run(self.outdir, resume is not None, self.env.resolved_mdp)\n')
    s=replace(s,'                self.load(resume)\n','                self.load(resume)\n            if resume is None:\n                self.save()  # recoverable epoch-zero boundary\n')
    s=replace(s,"            'epoch': self.epoch, 'env_steps': self.env_steps,\n",f"            'epoch': self.epoch, 'env_steps': self.env_steps,\n            'episode_commit': episode_commit({p}, self.epoch, self.env_steps, self.env.horizon),\n            'training_identity_sha256': sha_file(self.outdir / 'training_identity.json'),\n")
    if mbpo:
        s=replace(s,'        torch.save(ck, tmp)\n        os.replace(tmp, path)\n',"        with tmp.open('wb') as f:\n            torch.save(ck,f); f.flush(); os.fsync(f.fileno())\n        os.replace(tmp,path)\n")
    else:
        s=replace(s,'        torch.save(data, tmp)\n        os.replace(tmp, dest)\n',"        with tmp.open('wb') as f:\n            torch.save(data,f); f.flush(); os.fsync(f.fileno())\n        os.replace(tmp,dest)\n")
    ck='ck' if mbpo else 'ckpt'
    anchor = '        self.agent.load_state_dict('+ck+"['agent'])\n"
    s=replace(s,anchor,f"        if {ck}.get('training_identity_sha256') != sha_file(self.outdir / 'training_identity.json'):\n            raise ValueError('Checkpoint training identity mismatch')\n"+anchor)
    a='        if not self.history_path.is_file():' if mbpo else '        if not self._episode_path.exists():'
    s=between(s,a,'    def train(self, epochs=None):',f'''        recover_history({p}, {ck}.get('episode_commit'), self.epoch,
                        self.env_steps, self.env.horizon)
        self.evaluator.reconcile(self.epoch, self.env.horizon)

''')
    target='target' if mbpo else 'until'
    s=replace(s,'self.epoch > 0 and self.epoch % eval_interval == 0\n',f'self.epoch > 0 and (self.epoch % eval_interval == 0 or self.epoch == {target})\n')
    if mbpo:
        # Distinguish numerical model fit, skill gate and actual replay use.
        s=replace(s,"        self.latest_model_metrics = {**stats, 'model_fit_seconds': fit_seconds,", "        self.latest_model_metrics = {**stats, 'model_fit_seconds': fit_seconds,\n                                     'model_fit_completed': 1, 'model_skill_pass': 0,")
        s=replace(s,'        states, scenario_ids = self.real_pool.rollout_states',"        self.latest_model_metrics['model_skill_pass'] = 1\n        states, scenario_ids = self.real_pool.rollout_states")
        s=replace(s,"        self.latest_model_metrics.update({\n            'model_attempted': int(len(states))", "        self.latest_model_metrics.update({\n            **self.fake_env.last_feasibility,\n            'model_attempted': int(len(states))")
        s=replace(s,'        fit_seconds = rollout_seconds = 0.0\n','        fit_seconds = rollout_seconds = 0.0\n        diagnostic_counts = dict.fromkeys((\'model_fit_attempted\',\'model_fit_completed\',\'model_skill_pass\',\'envelope_checked\',\'envelope_violations\',\'envelope_irradiance_saturated\',\'raw_weather_outside\',\'raw_power_outside\'), 0)\n        real_saturated = 0; real_envelope_states = []\n')
        s=replace(s,'            self.real_pool.add(state, action, reward, nxt, done, sid)\n', '''            self.real_pool.add(state, action, reward, nxt, done, sid)
            # Actual irradiance saturation is checked before decoding.
            if max(float(info['dni_wm2']), float(info['dhi_wm2'])) >= 2000.:
                real_saturated += 1
            else:
                real_envelope_states.append(nxt.copy())
''')
        s=replace(s,"                fit_seconds += float(self.latest_model_metrics.get('model_fit_seconds', 0.0))\n", "                diagnostic_counts['model_fit_attempted'] += 1\n                for key in diagnostic_counts:\n                    diagnostic_counts[key] += int(self.latest_model_metrics.get(key, 0))\n                fit_seconds += float(self.latest_model_metrics.get('model_fit_seconds', 0.0))\n")
        s=replace(s,'        self.epoch += 1\n',"        e = envelope_counts(np.asarray(real_envelope_states), self.env.unwrapped) if real_envelope_states else {'envelope_checked':0,'envelope_violations':0,'envelope_irradiance_saturated':0}\n        real_checked=e['envelope_checked']; real_violations=e['envelope_violations']; real_saturated+=e['envelope_irradiance_saturated']\n        self.epoch += 1\n")
        s=replace(s,"            'synthetic_samples_used': int(sum(nm)), 'real_samples_used': int(sum(nr)),\n", "            'synthetic_samples_used': int(sum(nm)), 'real_samples_used': int(sum(nr)),\n            **diagnostic_counts, 'real_envelope_checked':real_checked,\n            'real_envelope_violations':real_violations, 'real_irradiance_saturated':real_saturated,\n")
    return s

EVAL_METHODS='''    def _evaluation_manifest(self, epoch, env_steps):
        from mbpo.audit_support import digest, sha_file
        def rows_at(name):
            with (self.eval_dir/name).open(newline='', encoding='utf-8') as stream:
                return [r for r in csv.DictReader(stream) if int(r['epoch']) == int(epoch)]
        days=rows_at('evaluation_days.csv'); ag=rows_at('evaluations.csv')
        if len(ag)!=1 or len(days)!=len(self.validation_ids) or {r['scenario_id'] for r in days}!=set(self.validation_ids):
            raise ValueError('Incomplete periodic evaluation grid')
        for row in days+ag:
            if int(row['env_steps'])!=int(env_steps) or row['algorithm']!=self.algorithm or int(row['seed'])!=self.seed:
                raise ValueError('Periodic evaluation identity mismatch')
        values=np.asarray([float(r['net_energy_kwh']) for r in days])
        if not np.isfinite(values).all() or not np.isclose(values.mean(),float(ag[0]['mean_net_energy_kwh']),rtol=1e-8,atol=1e-10):
            raise ValueError('Periodic aggregate mismatch')
        snapshot=self.snapshot_dir/f"{self.algorithm}_seed{self.seed}_epoch{int(epoch):05d}.pt"
        return {'epoch':int(epoch),'env_steps':int(env_steps),'days':digest(days),'aggregate':digest(ag),'snapshot_sha256':sha_file(snapshot)}

    def has_evaluation(self, *, epoch: int, env_steps: int) -> bool:
        from mbpo.audit_support import read_json
        path=self.eval_dir/f'complete_{int(epoch):05d}.json'
        try:
            return read_json(path)==self._evaluation_manifest(epoch,env_steps)
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def reconcile(self, epoch, horizon):
        """Discard only artifacts beyond the committed checkpoint; verify earlier evaluations."""
        for name in ('evaluation_days.csv','evaluations.csv'):
            path=self.eval_dir/name
            if not path.is_file(): continue
            with path.open(newline='',encoding='utf-8') as stream:
                reader=csv.DictReader(stream); fields=reader.fieldnames; rows=list(reader)
            keep=[r for r in rows if int(r['epoch'])<=epoch]
            if len(keep)!=len(rows):
                import io
                from mbpo.audit_support import atomic_bytes
                output=io.StringIO(newline=''); writer=csv.DictWriter(output,fieldnames=fields)
                writer.writeheader(); writer.writerows(keep); atomic_bytes(path,output.getvalue().encode())
        for path in self.eval_dir.glob('complete_*.json'):
            if int(path.stem.split('_')[-1])>epoch: path.unlink()
        for path in self.snapshot_dir.glob(f'{self.algorithm}_seed{self.seed}_epoch*.pt'):
            if int(path.stem.rsplit('epoch',1)[1])>epoch: path.unlink()
        interval=int(self.protocol['evaluation_interval_episodes'])
        for earlier in range(interval,epoch,interval):
            if not self.has_evaluation(epoch=earlier,env_steps=earlier*horizon):
                raise ValueError(f'Committed historical evaluation {earlier} is missing or corrupt; cannot reconstruct from latest policy')

'''

def protocol(s):
    s=replace(s,'    write_json_atomic(path, data)\n\n\ndef _write_csv_replace',"    if status == 'completed' and (Path(run_dir)/'training_identity.json').is_file():\n        from mbpo.audit_support import sha_file\n        data['final_checkpoint_sha256']=sha_file(Path(run_dir)/'latest.pt')\n        data['training_identity_sha256']=sha_file(Path(run_dir)/'training_identity.json')\n    write_json_atomic(path, data)\n\n\ndef _write_csv_replace")
    s=between(s,'    def has_evaluation(', '    def evaluate(',EVAL_METHODS)
    s=replace(s,"            day_rows = []\n            for sid in self.validation_ids:\n                result = self._run_episode(sid, policy=policy, controller=\"policy\")\n                day_rows.append({\n                    \"algorithm\": self.algorithm, \"seed\": self.seed,\n                    \"epoch\": int(epoch), \"env_steps\": int(env_steps),\n                    \"scenario_id\": sid, **result,\n                })\n","            from mbpo.audit_support import preserve_global_rng\n            day_rows = []\n            with preserve_global_rng():\n                for sid in self.validation_ids:\n                    result = self._run_episode(sid, policy=policy, controller=\"policy\")\n                    day_rows.append({\n                        \"algorithm\": self.algorithm, \"seed\": self.seed,\n                        \"epoch\": int(epoch), \"env_steps\": int(env_steps),\n                        \"scenario_id\": sid, **result,\n                    })\n")
    s=replace(s,'        self._save_policy_snapshot(policy, epoch=epoch, env_steps=env_steps, updates=updates)\n        return aggregate', '''        self._save_policy_snapshot(policy, epoch=epoch, env_steps=env_steps, updates=updates)
        from mbpo.audit_support import atomic_json
        atomic_json(self.eval_dir/f'complete_{int(epoch):05d}.json', self._evaluation_manifest(epoch,env_steps))
        return aggregate''')
    return s


def dynamics(s):
    s=replace(s,'    def predict_delta(self, states, actions, stochastic=True):','    def predict_delta(self, states, actions, stochastic=True, *, rng=None, record=True):')
    start=s.index('    def predict_delta('); end=s.index('    def state_dict(',start)
    b=s[start:end]; b=replace(b,'        states, actions = self._check(states, actions)\n','        states, actions = self._check(states, actions)\n        rng = self.rng if rng is None else rng\n'); b=b.replace('self.rng.integers','rng.integers').replace('self.rng.normal','rng.normal')
    b=replace(b,'        next_weather = np.clip(states[:, :2]+dw, 0., 1.)\n','''        raw_weather = states[:, :2]+dw
        if not np.isfinite(raw_weather).all(): raise FloatingPointError('Nonfinite raw weather sample')
        next_weather = np.clip(raw_weather, 0., 1.)
''')
    b=replace(b,'        dp = np.clip(states[:, 2:3]+dp, 0., 1.)-states[:, 2:3]\n','''        raw_power = states[:, 2:3]+dp
        if not np.isfinite(raw_power).all(): raise FloatingPointError('Nonfinite raw power sample')
        dp = np.clip(raw_power, 0., 1.)-states[:, 2:3]
        if record:
            self.last_sample_diagnostics = {
                'raw_weather_outside':int(((raw_weather<0)|(raw_weather>1)).any(axis=1).sum()),
                'raw_power_outside':int(((raw_power<0)|(raw_power>1)).any(axis=1).sum())}
''')
    s=s[:start]+b+s[end:]
    s=replace(s,'        self.solar_context = SolarContext(reference_env)\n','        self.solar_context = SolarContext(reference_env)\n        self.reference_physics = reference_env.unwrapped\n        self.last_feasibility = {}\n')
    # Do not require mock integration environments to expose physics if they intentionally lack it.
    s=replace(s,'        if np.any(mask): check_batch(nxt[mask])\n','''        if np.any(mask): check_batch(nxt[mask])
        from mbpo.audit_support import envelope_counts
        e = self.reference_physics
        self.last_feasibility = dict(getattr(self.ensemble,'last_sample_diagnostics',{}))
        if all(hasattr(e,k) for k in ('area','efficiency','power_model')) and np.any(mask):
            self.last_feasibility.update(envelope_counts(nxt[mask],e))
''')
    return s


def replay(s):
    return replace(s,'        valid=np.flatnonzero((self.states[:self.count,TAU]>1e-7)\n                                 &(self.dones[:self.count,0]<.5))','        valid=np.flatnonzero(self.states[:self.count,TAU]>1e-7)')


def core(s):
    s=add_import(s,'from mbpo.audit_support import runtime_identity, dataset_identity, exclusive_lock, digest')
    s=replace(s,'        cfg=load_experiment(name,root)\n        if device', '''        cfg=load_experiment(name,root)
        if not dry_run and (cfg.get('training_runtime')!=runtime_identity(root) or cfg.get('dataset_identity')!=dataset_identity(root)):
            raise ValueError('Experiment source/dataset changed or predates audit patch; use a new experiment name')
        if device''')
    s=replace(s,'    if not dry_run:\n        dest=base_path(name,root)\n','''    if not dry_run:
        data['training_runtime']=runtime_identity(root)
        data['dataset_identity']=dataset_identity(root)
        dest=base_path(name,root)
''')
    s=replace(s,"    if meta.get('training_status')=='completed':\n",'''    identity_path=d/'training_identity.json'
    if cfg.get('training_runtime') is not None:
        identity=read_json(identity_path) if identity_path.is_file() else None
        if identity is None or identity.get('runtime')!=cfg['training_runtime'] or any(identity.get('physics',{}).get(k)!=v for k,v in cfg['dataset_identity'].items()):
            raise ValueError(f'Run source/dataset identity differs from frozen experiment: {d}')
    if meta.get('training_status')=='completed':
        if cfg.get('training_runtime') is not None:
            from mbpo.audit_support import sha_file
            if meta.get('final_checkpoint_sha256')!=sha_file(d/'latest.pt') or meta.get('training_identity_sha256')!=sha_file(identity_path):
                raise ValueError(f'Completed checkpoint or training identity changed: {d}')
            if meta.get('final_env_steps')!=meta.get('horizon')*cfg['epochs']:
                raise ValueError(f'Completed run interaction budget differs: {d}')
''')
    # Replace stale status as authoritative mutex with inherited process lock.
    s=replace(s,"    if job_path.exists() and read_json(job_path).get('status')=='running':\n        raise RuntimeError(f'Existing job is marked running: {job_path}. Inspect it before starting a duplicate.')\n",'''    with exclusive_lock(job_path.with_suffix('.lock')) as lock_fd:
        return _execute_locked(cfg,kind,command,root,seed,directory,log,job_path,lock_fd)


def _execute_locked(cfg,kind,command,root,seed,directory,log,job_path,lock_fd):
''')
    s=replace(s,"        with log.open('w',encoding='utf-8') as f:\n", "        with log.open('a',encoding='utf-8') as f:\n            f.write('\\n--- managed invocation ---\\n'); f.flush()\n")
    s=replace(s,"            proc=subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,\n","            child_env=os.environ.copy()\n            if kind.startswith('train_') and cfg.get('training_runtime') is not None:\n                child_env['PV_AUDIT_EXPECTED_RUNTIME']=digest(cfg['training_runtime'])\n                child_env['PV_AUDIT_EXPECTED_DATASET']=canonical(cfg['dataset_identity'])\n            proc=subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,\n")
    s=replace(s,'                                  text=True,bufsize=1)','                                  text=True,bufsize=1,pass_fds=(lock_fd,),env=child_env)')
    return s


def train(s):
    s=add_import(s,'from mbpo.audit_support import runtime_identity, dataset_identity, exclusive_lock')
    s=replace(s,'def run(args,root=PROJECT_ROOT):\n',"def run(args,root=PROJECT_ROOT):\n    if args.dry_run: return _run_locked(args,root)\n    with exclusive_lock(base_path(args.name,root).parent/('.'+args.name+'.training.lock')):\n        return _run_locked(args,root)\n\n\ndef _run_locked(args,root=PROJECT_ROOT):\n")
    s=replace(s,'        verified=False\n','        verified=False\n        identity={\'runtime\':runtime_identity(root,\'physical\'),\'dataset\':dataset_identity(root),\'uncertainty\':cfg[\'uncertainty_sha256\']}\n')
    s=replace(s,"and data.get('state_dim')==STATE_DIM)","and data.get('state_dim')==STATE_DIM and data.get('audit_identity')==identity)")
    s=replace(s,"            execute(cfg,'physical_preflight',command,root=root)\n","            execute(cfg,'physical_preflight',command,root=root)\n            data=read_json(audit)\n            if data.get('status')!='passed': raise RuntimeError('Physical preflight did not pass')\n            if data.get('audit_identity')!=identity: raise RuntimeError('Preflight source/dataset identity changed')\n")
    return s


def evaluate(s):
    s=add_import(s,'from mbpo.audit_support import runtime_identity, dataset_identity')
    s=replace(s,"    result={\n","    dataset=dataset_identity(root)\n    if cfg.get('dataset_identity') is not None and dataset!=cfg['dataset_identity']:\n        raise ValueError('Current dataset differs from frozen experiment')\n    for folder in (sac,mbpo):\n        resolved=read_json(folder/'resolved_mdp.json')\n        if any(resolved.get(k)!=v for k,v in dataset.items()):\n            raise ValueError('Current dataset differs from checkpoint physical identity')\n    result={\n")
    s=replace(s,"      'schema':1,'experiment':cfg['name'],'seed':int(seed),'mode':mode,\n", "      'schema':2,'experiment':cfg['name'],'seed':int(seed),'mode':mode,\n      'runtime':runtime_identity(root,'model' if mode=='model' else 'evaluation'), 'dataset':dataset,\n")
    return s


def report(s):
    s=replace(s,'    data=read_json(summary)\n','''    data=read_json(summary)
    from .evaluate import existing_report_is_valid,_provenance,_require_matching_provenance
    draws=int(data['draws_per_day'])
    if not existing_report_is_valid(cfg,stage,seed,draws,directory,root):
        raise ValueError('Incomplete Monte Carlo source')
    _require_matching_provenance(directory,_provenance(cfg,stage,seed,root,draws=draws))
''')
    s=replace(s,"for filename in ('mc_summary.json','all_day_draw_method.csv',", "for filename in ('manager_provenance.json','mc_summary.json','all_day_draw_method.csv',")
    s=replace(s,"for rel in ('run_metadata.json','scenario_split.json',", "for rel in ('run_metadata.json','scenario_split.json','episodes.csv',")
    s=replace(s,"and m.get('source_file_sha256')==input_hashes):","and m.get('source_file_sha256')==input_hashes and m.get('estimator_semantics')=='confirmed_v2'):")
    s=replace(s,"'seeds':seeds,'source_file_sha256':input_hashes,\n","'seeds':seeds,'source_file_sha256':input_hashes,'estimator_semantics':'confirmed_v2',\n")
    return s


def sample(s):
    s=add_import(s,'from mbpo.audit_support import target_event, expected_steps')
    s=between(s,'def _threshold_steps(', 'def _auc(','''def _threshold_steps(curve: np.ndarray, steps: np.ndarray, target: float, sustain: int) -> float | None:
    _, confirmed=target_event(curve,steps,target,sustain)
    return float(confirmed) if confirmed is not None else None


''')
    s=between(s,'    # Use exact common real-interaction checkpoints.', '    if len(steps) < 2:', '''    # Require all scheduled evaluations, not the intersection of incomplete runs.
    steps=expected_steps(ref_meta)
    for run in all_runs:
        if set(run['aggregate'])!=set(steps) or set(run['daily'])!=set(steps):
            raise ValueError(f"Missing or unexpected scheduled evaluations: {run['root']}")
''')
    s=replace(s,'            reached = _threshold_steps(curve, steps, target, int(sustain_checkpoints))\n','            onset, reached = target_event(curve, steps, target, int(sustain_checkpoints))\n')
    s=replace(s,'                "real_env_steps_to_target": reached,\n','                "real_env_steps_to_target": reached,\n                "onset_steps": onset, "confirmed_steps": reached,\n                "restricted_steps": reached if reached is not None else int(steps[-1]),\n')
    s=replace(s,'    reduction_summary = {\n','''    restricted = {a: [r['restricted_steps'] for r in per_seed_rows if r['algorithm']==a] for a in ('sac','mbpo')}
    reduction_summary = {
        "endpoint": "confirmed attainment; onset is retrospective",
        "restricted_steps_saved_mean": float(np.mean(np.asarray(restricted['sac'])-np.asarray(restricted['mbpo']))),
        "restricted_budget_relative_saving_percent": float(100*(1-np.mean(restricted['mbpo'])/np.mean(restricted['sac']))),
''')
    s=replace(s,'first common real-transition checkpoint at which mean validation net energy reaches target for the required consecutive checkpoints','confirmation at the final scheduled checkpoint of the first sustained sequence; retrospective onset reported separately')
    return s


def advanced(s):
    s=add_import(s,'from mbpo.audit_support import expected_steps, target_event as event')
    return between(s,'def expected_steps(', 'def load_learning(', '')


def verifier(s):
    s=replace(s,'    power=[];gross=[];cost=[];clock=[];albedo=[]\n','    power=[];gross=[];cost=[];clock=[];albedo=[]\n    from mbpo.audit_support import envelope_counts\n    checked=violations=saturated=0\n')
    s=replace(s,'        power.append(P);gross.append(E);cost.append(C);clock.append(actual);albedo.append(info[\'surface_albedo\'])\n', '''        power.append(P);gross.append(E);cost.append(C);clock.append(actual);albedo.append(info['surface_albedo'])
        if max(float(info['dni_wm2']),float(info['dhi_wm2']))>=2000.: saturated+=1
        else:
            counts=envelope_counts(obs[None,:],u)
            checked+=counts['envelope_checked']; violations+=counts['envelope_violations']
''')
    s=replace(s,"        'sample_count':len(power),\n", "        'sample_count':len(power),'real_envelope_checked':checked,'real_envelope_violations':violations,'real_irradiance_saturated':saturated,\n")
    s=replace(s,"    out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')","    from mbpo.audit_support import runtime_identity,dataset_identity,atomic_json\n    from PV_experiments.core import canonical,sha256\n    report['audit_identity']={'runtime':runtime_identity(repo_root,'physical'),'dataset':dataset_identity(repo_root),'uncertainty':sha256(canonical(json.loads(Path(args.uncertainty_spec).read_text(encoding='utf-8'))))}\n    atomic_json(out,report)")
    return s


def representation(s):
    s=replace(s,"    from .common import write_json\n    write_json(out/'learning_report_notes.json'", "    from .common import write_json\n    write_json(out/'learning_report_notes.json'")  # assert expected output route
    s=replace(s,"'First sustained evaluated checkpoint; open triangles mark last evaluated budget, not reaching time.'", "'Confirmed attainment (end of sustained sequence); open triangles mark last evaluated budget, not reaching time.'")
    s=replace(s,"figs=Figures(out); rows=[];cf=[];sources={}\n","figs=Figures(out); rows=[];cf=[];sources={}; control_rows=[]\n")
    s=replace(s,"        d=read_json(path/'model_validation_summary.json'); p=path/'external_on_policy_predictions.csv';f=pd.read_csv(p)\n","        d=read_json(path/'model_validation_summary.json'); p=path/'external_on_policy_predictions.csv';f=pd.read_csv(p)\n        control_rows.append(dict(seed=seed,**d['counterfactual_action_test'],**{'train_'+k:v for k,v in d['training_synthetic_data_accounting'].items() if isinstance(v,(int,float)) or v is None}))\n")
    s=replace(s,"    table=pd.DataFrame(rows);save_csv(out,'model_quality_by_seed.csv',table);cf=pd.concat(cf);save_csv(out,'counterfactual_results.csv',cf)\n","    table=pd.DataFrame(rows);save_csv(out,'model_quality_by_seed.csv',table);cf=pd.concat(cf);save_csv(out,'counterfactual_results.csv',cf)\n    save_csv(out,'model_control_diagnostics_by_seed.csv',control_rows)\n")
    return s

# Additional validation modifications are below.
def model_validation(s):
    # Additive diagnostic changes retain CSV names and existing output columns.
    s=replace(s,'def _channel_metrics(actual, predicted, total_std=None):','def _channel_metrics(actual, predicted, total_std=None, quantiles=None):')
    s=replace(s,'            out[name]["coverage_90"] = float(np.mean(np.abs(err) <= 1.6448536269514722 * sd))\n            out[name]["coverage_95"] = float(np.mean(np.abs(err) <= 1.959963984540054 * sd))', '''            if quantiles is None: raise ValueError('Empirical intervals required with diagnostic SD')
            q=np.asarray(quantiles)
            if q.shape!=(4,len(actual),3) or not np.isfinite(q).all(): raise ValueError('Invalid predictive quantiles')
            out[name]["coverage_90"] = float(np.mean((y>=q[1,:,j])&(y<=q[2,:,j])))
            out[name]["coverage_95"] = float(np.mean((y>=q[0,:,j])&(y<=q[3,:,j])))''')
    s=replace(s,'            sd = np.maximum(np.asarray(total_std, dtype=float)[:, j], 1e-8)',"            sd = np.asarray(total_std, dtype=float)[:, j]\n            if not np.isfinite(sd).all() or np.any(sd<0): raise ValueError('Invalid predictive standard deviation')")
    s=replace(s,'def _prediction_rows(actual, predicted, uncertainty, metadata):','def _prediction_rows(actual, predicted, uncertainty, metadata, quantiles=None):')
    s=replace(s,'            row[f"error_delta_{name}"] = float(predicted[i, j] - actual[i, j])\n','''            row[f"error_delta_{name}"] = float(predicted[i, j] - actual[i, j])
            if quantiles is not None:
                for k,label in enumerate(('q025','q05','q95','q975')):
                    row[f'{label}_delta_{name}']=float(quantiles[k,i,j])
''')
    s=replace(s,'        holdout_mean, _holdout_aleatoric, _holdout_epistemic, holdout_total, holdout_score = (\n            ensemble.predict_distribution(holdout_states, holdout_actions))\n        internal_metrics = _channel_metrics(holdout_actual_delta, holdout_mean, holdout_total)', '''        from mbpo.audit_support import hierarchical, diagnostic_seed, conditional_power_rmse, counterfactual_metrics, envelope_counts
        _m, _a, _e, _t, holdout_score = ensemble.predict_distribution(holdout_states, holdout_actions)
        holdout_mean,holdout_total,holdout_quantiles=hierarchical(ensemble,holdout_states,holdout_actions,seed=diagnostic_seed(seed,'internal'))
        internal_metrics = _channel_metrics(holdout_actual_delta, holdout_mean, holdout_total, holdout_quantiles)''')
    s=replace(s,'holdout_score, internal_meta)','holdout_score, internal_meta, holdout_quantiles)')
    s=replace(s,'        counterfactual_rows = []\n','        counterfactual_rows = []\n        external_means=[]; external_quantiles=[]; external_disagreement=[]\n        external_sample_envelope={}\n')
    s=replace(s,'            mean_delta, _aleatoric, _epistemic, total_std, _score = ensemble.predict_distribution(\n                states, actions)\n','''            mean_delta,total_std,quantiles=hierarchical(ensemble,states,actions,seed=diagnostic_seed(seed,sid,'external',args.realization_id),
                envelope_template=pred_next,env=env.unwrapped,envelope_totals=external_sample_envelope)
            # Exact reward is affine in clipped power, with fixed movement cost.
            mean_next=states[:,:3]+mean_delta
            pred_rewards=pred_rewards+(mean_next[:,2]-pred_next[:,2])*fake_env.power_norm*fake_env.interval_hours/1000.
            pred_next[:,:3]=mean_next
            external_means.append(mean_delta); external_quantiles.append(quantiles); external_disagreement.append(disagreement)
''')
    s=replace(s,'                    pnext = pred_nxt[0]\n','''                    # Same diagnostic seed and identical shape for each action at this state.
                    cf_mean,_,_=hierarchical(ensemble,replay_obs[None,:],candidate[None,:],
                        seed=diagnostic_seed(seed,sid,int(t),args.realization_id,'counterfactual'))
                    mean_next=replay_obs[:3]+cf_mean[0]
                    pred_reward=pred_reward+(mean_next[2]-pred_nxt[0,2])*fake_env.power_norm*fake_env.interval_hours/1000.
                    pred_nxt[0,:3]=mean_next
                    pnext = pred_nxt[0]
''')
    s=replace(s,'        predicted_delta, _a, _e, _total, disagreement = ensemble.predict_distribution(states, actions)\n        external_metrics = _channel_metrics(actual_delta, predicted_delta, total_std)','''        predicted_delta=np.concatenate(external_means,axis=0)
        quantiles=np.concatenate(external_quantiles,axis=1)
        disagreement=np.concatenate(external_disagreement,axis=0)
        external_metrics = _channel_metrics(actual_delta, predicted_delta, total_std, quantiles)
        external_metrics['conditional_power_rmse_norm_actual_next_weather']=conditional_power_rmse(ensemble,states,actions,next_states)
        external_metrics['real_operational_envelope']=envelope_counts(next_states,env.unwrapped)
        external_metrics['predicted_mean_operational_envelope']=envelope_counts(pred_next,env.unwrapped)
        external_metrics['hierarchical_sample_operational_envelope']=external_sample_envelope''')
    s=replace(s,'        training_generation = _load_training_stats(run_dir / "episodes.csv")\n','''        counterfactual_metrics.update(counterfactual_metrics_fn(counterfactual_rows))
        training_generation = _load_training_stats(run_dir / "episodes.csv")
'''.replace('counterfactual_metrics_fn','action_metrics'))
    # The existing local dict is also named counterfactual_metrics; alias the function.
    s=replace(s,'conditional_power_rmse, counterfactual_metrics, envelope_counts','conditional_power_rmse, counterfactual_metrics as action_metrics, envelope_counts')
    s=replace(s,'             for m in external_meta])','             for m in external_meta], quantiles)')
    s=replace(s,'            "validation_schema": 1,','            "validation_schema": 2,\n            "hierarchical_diagnostic_samples":128,\n            "feasibility_policy":"diagnostic only; cell_min=-60 C, GHI residual allowance=200 W/m2, rho_max=1; not guaranteed support bounds",')
    s=replace(s,'Predictive interval coverage uses the ensemble Gaussian-mixture moment-matched standard deviation; it is a diagnostic, not a guarantee of calibrated uncertainty.','Predictive coverage uses empirical quantiles of hierarchical clipped weather and power samples; total SD is not epistemic uncertainty. Ensemble disagreement remains the deterministic plug-in diagnostic.')
    s=replace(s,'"real_samples_used"):', '''"real_samples_used", "model_fit_attempted", "model_fit_completed", "model_skill_pass", "envelope_checked", "envelope_violations", "envelope_irradiance_saturated", "raw_weather_outside", "raw_power_outside", "real_envelope_checked", "real_envelope_violations", "real_irradiance_saturated"):''')
    s=replace(s,'    return result\n\n\ndef _run_policy_trajectory',"    fit=result.get('model_fit_completed'); passed=result.get('model_skill_pass')\n    result['skill_gate_pass_fraction']=passed/fit if fit and passed is not None else None\n    real=result.get('real_samples_used'); synthetic=result.get('synthetic_samples_used')\n    result['actual_synthetic_minibatch_fraction']=synthetic/(real+synthetic) if real is not None and synthetic is not None and real+synthetic>0 else None\n    return result\n\n\ndef _run_policy_trajectory")
    return s

def legacy(s):
    return replace(s,'import argparse\n',"if __name__ == '__main__':\n    raise SystemExit('This legacy evaluation CLI is unsupported for Geometry10. Use python -m PV_experiments.evaluate --name EXPERIMENT --mode validation.')\nimport argparse\n")

TRANSFORMS={
 'MBPO_workspace/src/dynamics.py':dynamics,
 'MBPO_workspace/src/replay.py':replay,
 'MBPO_workspace/src/sample_complexity.py':sample,
 'MBPO_workspace/src/trainer.py':lambda s:trainer(s,True),
 'MBPO_workspace/workflows/06_model_validation.py':model_validation,
 'PV_experiments/advanced_learning.py':advanced,
 'PV_experiments/core.py':core,
 'PV_experiments/evaluate.py':evaluate,
 'PV_experiments/report.py':report,
 'PV_experiments/representation/training.py':representation,
 'PV_experiments/train.py':train,
 'PostProcessing_MC_patch/tools/verify_mc_environment.py':verifier,
 'SAC_workspace/src/training/experiment_protocol.py':protocol,
 'SAC_workspace/src/training/trainer.py':lambda s:trainer(s,False),
 'SAC_workspace/workflows/04_evaluate.py':legacy,
 'MBPO_workspace/workflows/03_evaluate.py':legacy,
}

def blob(data): return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def git(root,*args,input=None,check=True):
    p=subprocess.run(['git','-C',str(root),*args],input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if check and p.returncode: raise RuntimeError(f'git {args}: {p.stderr.decode(errors="replace")}')
    return p


def build(root):
    # Only exact audited files; no fuzzy replacements or destructive reset.
    result={}
    for path,fn in TRANSFORMS.items():
        data=(root/path).read_bytes()
        if blob(data)!=PIN[path]: raise ValueError(f'{path}: source differs from audited Git blob {PIN[path]}')
        old=data.decode('utf-8'); new=fn(old)
        if new==old: raise ValueError(f'No change generated: {path}')
        compile(new,path,'exec'); result[path]=(old,new)
    for file in sorted(PAYLOAD.rglob('*')):
        if not file.is_file() or '__pycache__' in file.parts: continue
        path=file.relative_to(PAYLOAD).as_posix()
        if (root/path).exists(): raise FileExistsError(f'New file already exists: {path}')
        if git(root,'check-ignore','--',path,check=False).returncode==0: raise ValueError(f'New source is Git-ignored: {path}')
        new=file.read_text(encoding='utf-8')
        if path.endswith('.py'): compile(new,path,'exec')
        result[path]=(None,new)
    return result


def patch_bytes(changes):
    parts=[]
    for path,(old,new) in changes.items():
        parts.append(f'diff --git a/{path} b/{path}\n')
        if old is None: parts.append('new file mode 100644\n')
        parts.extend(difflib.unified_diff((old or '').splitlines(keepends=True),new.splitlines(keepends=True),fromfile=f'a/{path}' if old is not None else '/dev/null',tofile=f'b/{path}'))
    return ''.join(parts).encode()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.')
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--dry-run',action='store_true');mode.add_argument('--apply',action='store_true');mode.add_argument('--write-patch',metavar='PATH')
    args=p.parse_args();root=Path(args.root).resolve()
    if git(root,'rev-parse','HEAD').stdout.decode().strip()!=BASE: raise SystemExit(f'Expected exact audited HEAD {BASE}; no changes made.')
    if Path(git(root,'rev-parse','--show-toplevel').stdout.decode().strip()).resolve()!=root: raise SystemExit('Provide the Git repository root.')
    changes=build(root); patch=patch_bytes(changes)
    git(root,'apply','--check','--whitespace=error-all','-',input=patch)
    print(f'PASS: {len(changes)} target files, original Git blob hashes, exact anchors, Python compilation and git apply --check')
    if args.write_patch:
        out=Path(args.write_patch);out.write_bytes(patch);print('Patch written:',out.resolve())
    elif args.apply:
        git(root,'apply','--whitespace=error-all','-',input=patch)
        print('Applied directly; no backups, commits, pushes or alternate workspaces created.')
    else: print('Dry run: repository unchanged.')
    for path in changes: print(' ',path)

if __name__=='__main__':
    try: main()
    except Exception as e: print(f'ERROR: {e}',file=sys.stderr);sys.exit(1)
