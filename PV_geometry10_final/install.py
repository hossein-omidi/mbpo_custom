#!/usr/bin/env python3
"""Apply the complete audited Geometry10 update directly to an existing Git checkout."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import integrated as old

BASE=old.BASE
HERE=Path(__file__).resolve().parent
r=old.replace
add=old.add_import


def trainers(s,mbpo):
    s=old.trainer(s,mbpo)
    algo='mbpo' if mbpo else 'sac'; ck='ck' if mbpo else 'ckpt'
    p='self.history_path' if mbpo else 'self._episode_path'
    s=r(s,"            'episode_commit': episode_commit("+p+", self.epoch, self.env_steps, self.env.horizon),",f"            'algorithm': '{algo}',\n            'episode_commit': episode_commit({p}, self.epoch, self.env_steps, self.env.horizon, updates=self.agent.updates),")
    s=r(s,f"        recover_history({p}, {ck}.get('episode_commit'), self.epoch,\n                        self.env_steps, self.env.horizon)",f"        recover_history({p}, {ck}.get('episode_commit'), self.epoch,\n                        self.env_steps, self.env.horizon, updates=self.agent.updates)")
    s=r(s,f"        self.agent.load_state_dict({ck}['agent'])",f"        from mbpo.audit_release import checkpoint_identity\n        checkpoint_identity({ck},'{algo}',self.seed,{ck}['epoch'],{ck}['env_steps'])\n        if {ck}['env_steps']!={ck}['epoch']*self.env.horizon: raise ValueError('Checkpoint episode/interaction mismatch')\n        self.agent.load_state_dict({ck}['agent'])")
    return s


def protocol(s):
    s=old.protocol(s)
    # Validate snapshot content at creation, and commit reference/periodic artifacts at final completion.
    s=r(s,'        atomic_json(self.eval_dir/f\'complete_{int(epoch):05d}.json\', self._evaluation_manifest(epoch,env_steps))', '''        from mbpo.audit_release import check_snapshot
        check_snapshot(self.snapshot_dir/f"{self.algorithm}_seed{self.seed}_epoch{int(epoch):05d}.pt",self.algorithm,self.seed,epoch,env_steps,updates,self.config_sha256,self.protocol['protocol_sha256'],self.resolved_mdp_sha256)
        atomic_json(self.eval_dir/f'complete_{int(epoch):05d}.json', self._evaluation_manifest(epoch,env_steps))''')
    s=r(s,"        data['final_checkpoint_sha256']=sha_file(Path(run_dir)/'latest.pt')", "        from mbpo.audit_release import completed_training_files, check_final_checkpoint\n        check_final_checkpoint(run_dir,data)\n        data['training_output_sha256']=completed_training_files(run_dir,data)\n        data['final_checkpoint_sha256']=sha_file(Path(run_dir)/'latest.pt')")
    return s


def core(s):
    s=old.core(s)
    s=r(s,"            if meta.get('final_env_steps')!=meta.get('horizon')*cfg['epochs']:","            from mbpo.audit_release import completed_training_files\n            if meta.get('training_output_sha256')!=completed_training_files(d,meta):\n                raise ValueError(f'Completed training artifacts changed: {d}')\n            if meta.get('final_env_steps')!=meta.get('horizon')*cfg['epochs']:")
    # Require current dataset for managed reporting; training source compatibility is checked separately at resume.
    s=r(s,"    return cfg\n\n\ndef initialize_experiment", "    if cfg.get('dataset_identity') is not None and cfg['dataset_identity']!=dataset_identity(root):\n        raise ValueError('Current data differ from frozen experiment')\n    return cfg\n\n\ndef initialize_experiment")
    return s


def mc(s):
    s=r(s,'def load_policy(cfg,which,checkpoint,device,expected_protocol_sha256=None):','def load_policy(cfg,which,checkpoint,device,expected_protocol_sha256=None, *, expected_seed=None, expected_epoch=None, expected_sha=None):')
    s=r(s,"    data=torch.load(checkpoint,map_location='cpu',weights_only=False)","    from mbpo.audit_release import load_verified_checkpoint\n    data, loaded_sha=load_verified_checkpoint(checkpoint,which,expected_seed=expected_seed,expected_epoch=expected_epoch,expected_sha=expected_sha)")
    s=r(s,"    agent.load_state_dict(data['agent']);agent.actor.eval()","    agent.load_state_dict(data['agent']);agent.actor.eval()\n    agent.loaded_checkpoint_sha256=loaded_sha\n    agent.loaded_checkpoint_seed=int(data['seed']); agent.loaded_checkpoint_epoch=int(data['epoch'])")
    s=r(s,"    p.add_argument('--sac-checkpoint',required=True)","    p.add_argument('--sac-checkpoint',required=True)\n    p.add_argument('--expected-seed',type=int)\n    p.add_argument('--expected-epoch',type=int)\n    p.add_argument('--expected-sac-sha')\n    p.add_argument('--expected-mbpo-sha')")
    a="expected_protocol_sha256=split['protocol_sha256']),"
    if s.count(a)!=2:raise ValueError('Unexpected MC loader anchors')
    s=s.replace(a,"expected_protocol_sha256=split['protocol_sha256'], expected_seed=args.expected_seed, expected_epoch=args.expected_epoch, expected_sha=args.expected_sac_sha),",1)
    s=r(s,a,"expected_protocol_sha256=split['protocol_sha256'], expected_seed=args.expected_seed, expected_epoch=args.expected_epoch, expected_sha=args.expected_mbpo_sha),")
    s=r(s,'    rows,clock_rows=evaluate_paths(cfg,agents,days,args.draws,days[0])',"    if len({(a.loaded_checkpoint_seed,a.loaded_checkpoint_epoch) for a in agents.values()})!=1:\n        raise ValueError('SAC and MBPO checkpoint seed/epoch differ')\n    rows,clock_rows=evaluate_paths(cfg,agents,days,args.draws,days[0])")
    s=r(s,"    report={'uncertainty_spec':spec,", "    report={'loaded_policies':{k:{'sha256':a.loaded_checkpoint_sha256,'seed':a.loaded_checkpoint_seed,'epoch':a.loaded_checkpoint_epoch} for k,a in agents.items()},'uncertainty_spec':spec,")
    return s


def evaluate(s):
    s=old.evaluate(s)
    s=r(s,"'runtime':runtime_identity(root,'model' if mode=='model' else 'evaluation'), 'dataset':dataset,", "'runtime':runtime_identity(root,'model' if mode=='model' else 'evaluation'), 'dataset':dataset, 'final_epoch':int(cfg['epochs']),")
    s=old.between(s,'def _require_matching_provenance(', 'def _verify_model_validation_summary(','''def _require_matching_provenance(out,expected):
    from mbpo.audit_release import require_evaluation_seal
    require_evaluation_seal(out,expected)


''')
    s=r(s,"    if (d.get('seed')!=seed", "    if d.get('validation_schema')!=2: raise ValueError('Model diagnostic schema differs')\n    if (d.get('seed')!=seed")
    s=r(s,"    if mode=='smoke':cmd+=['--max-days','2']", "    if not dry_run:\n        cmd+=['--expected-seed',str(seed),'--expected-epoch',str(cfg['epochs']),'--expected-sac-sha',identity['sac_checkpoint_sha256'],'--expected-mbpo-sha',identity['mbpo_checkpoint_sha256']]\n    if mode=='smoke':cmd+=['--max-days','2']")
    # The model write has a deeper indentation; use full-line anchors rather than substring count.
    a="\n        atomic_json(out/'manager_provenance.json',identity)\n"
    s=r(s,a,"\n        from mbpo.audit_release import seal_evaluation, verify_loaded_policies\n        verify_loaded_policies(read_json(out/'mc_summary.json'),identity)\n        if _provenance(cfg,mode,seed,root,draws=draws)!=identity: raise ValueError('Evaluation inputs changed during execution')\n        seal_evaluation(out,identity)\n")
    s=r(s,"            cmd+=command_override(cfg)\n            execute(cfg,'model_validation'", "            if not args.dry_run: cmd+=['--expected-seed',str(seed),'--expected-epoch',str(cfg['epochs']),'--expected-sha',identity['mbpo_checkpoint_sha256']]\n            cmd+=command_override(cfg)\n            execute(cfg,'model_validation'")
    s=r(s,"                atomic_json(out/'manager_provenance.json',identity)","                from mbpo.audit_release import seal_evaluation, verify_loaded_policies\n                verify_loaded_policies(read_json(out/'model_validation_summary.json'),identity)\n                if _provenance(cfg,mode,seed,root,realization_id=args.realization_id)!=identity: raise ValueError('Model evaluation inputs changed during execution')\n                seal_evaluation(out,identity)")
    return s


def model_validation(s):
    s=old.model_validation(s)
    s=r(s,'    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)', '''    from mbpo.audit_release import load_verified_checkpoint, equivalent_pose_metrics
    ckpt,loaded_sha=load_verified_checkpoint(checkpoint_path,'mbpo',run_dir=run_dir,expected_seed=getattr(args,'expected_seed',None),expected_epoch=getattr(args,'expected_epoch',None),expected_sha=getattr(args,'expected_sha',None))''')
    s=r(s,'    parser.add_argument("--run-dir", required=True)', '    parser.add_argument("--run-dir", required=True)\n    parser.add_argument("--expected-seed", type=int)\n    parser.add_argument("--expected-epoch", type=int)\n    parser.add_argument("--expected-sha")')
    s=r(s,'                        "actual_next_power_norm": float(actual_nxt[2]),','''                        "next_tilt_norm":float(actual_nxt[3]),
                        "next_azimuth_sin":float(actual_nxt[4]),
                        "next_azimuth_cos":float(actual_nxt[5]),
                        "actual_next_power_norm": float(actual_nxt[2]),''')
    s=r(s,'        counterfactual_metrics.update(action_metrics(counterfactual_rows))','        counterfactual_metrics.update(action_metrics(counterfactual_rows))\n        counterfactual_metrics.update(equivalent_pose_metrics(counterfactual_rows))')
    s=r(s,"        external_metrics['conditional_power_rmse_norm_actual_next_weather']", "        from mbpo.audit_release import stratified_coverage\n        external_metrics['coverage_by_day_and_irradiance']=stratified_coverage(actual_delta,quantiles,next_states,external_meta)\n        external_metrics['conditional_power_rmse_norm_actual_next_weather']")
    s=r(s,'            "checkpoint": str(checkpoint_path),','            "checkpoint": str(checkpoint_path),\n            "loaded_policies":{"mbpo":{"sha256":loaded_sha,"seed":seed,"epoch":int(ckpt["epoch"])}},')
    return s


def preflight(s):
    s=old.verifier(s)
    s=r(s,'    for t in range(env.horizon):\n        obs,reward,done,info=env.step(np.zeros(2,dtype=np.float32))','''    from mbpo.geometry10 import pose_transition
    from mbpo.env.pvlib_physics import actuator_movement_cost_kwh
    moving_steps=0
    for t in range(env.horizon):
        action=np.array([.7,.8] if t%4==0 else [-.5,-.6] if t%4==1 else [0.,0.],dtype=np.float32)
        tilt,az,tau,dt,da,terminal=pose_transition(obs[None,:],action[None,:],u.max_delta_tilt,u.max_delta_azimuth,env.horizon)
        obs,reward,done,info=env.step(action)
        if not np.allclose(obs[3:6],np.r_[tilt,az[0]],rtol=0,atol=4e-5) or not np.isclose(obs[-1],tau[0],atol=1e-7) or bool(done)!=bool(terminal[0]):
            raise AssertionError('Scripted movement pose/horizon parity failed')
        expected_cost=actuator_movement_cost_kwh(float(dt[0]),float(da[0]),actuator_power_w=u.actuator_power_w,slew_rate_tilt_deg_s=u.slew_rate_tilt_deg_s,slew_rate_azimuth_deg_s=u.slew_rate_azimuth_deg_s,scale=u.movement_penalty)
        if not np.isclose(float(info['movement_cost']),expected_cost,rtol=0,atol=1e-8):raise AssertionError('Scripted executed-motion cost mismatch')
        moving_steps+=int(abs(dt[0])+abs(da[0])>1e-6)''')
    s=r(s,"        'sample_count':len(power),", "        'scripted_moving_steps':moving_steps,'sample_count':len(power),")
    s=r(s,"    for field in ('weather_profile_sha256','power_sha256','total_gross_kwh','total_net_kwh'):","    if first['scripted_moving_steps']<=0:raise AssertionError('No nonzero movement exercised')\n    for field in ('weather_profile_sha256','power_sha256','total_gross_kwh','total_net_kwh','scripted_moving_steps'):")
    return s


def advanced(s):
    s=old.advanced(s)
    s=r(s,"        rep['mbpo_replay_at_final_budget']=final.astype(object).where(pd.notna(final),None).to_dict(orient='records')", "        rep['mbpo_replay_at_final_budget']=final.astype(object).where(pd.notna(final),None).to_dict(orient='records')\n        from mbpo.audit_release import replay_at_confirmation\n        endpoint=replay_at_confirmation(d['replay'],r['rows'])\n        csv_out(out/'mbpo_replay_at_confirmation_or_budget.csv',endpoint)\n        rep['mbpo_replay_at_confirmation_or_budget']=endpoint")
    return s


def trajectories(s):
    return r(s,"expected_protocol_sha256=split['protocol_sha256']) for m in ('sac','mbpo')", "expected_protocol_sha256=split['protocol_sha256'],expected_seed=seed,expected_epoch=cfg['epochs']) for m in ('sac','mbpo')")


PIN=dict(old.PIN,**{'MBPO_workspace/workflows/09_monte_carlo_evaluate.py':'fd2fe2e740ce65b77bb73954743c0e24c2d6a42a','PV_experiments/representation/trajectories.py':'6d9b114bffbfdfacf8a3d885c2ed7c26f17f2f27'})
TRANSFORMS=dict(old.TRANSFORMS)
TRANSFORMS.update({'SAC_workspace/src/training/trainer.py':lambda s:trainers(s,False),'MBPO_workspace/src/trainer.py':lambda s:trainers(s,True),'SAC_workspace/src/training/experiment_protocol.py':protocol,'PV_experiments/core.py':core,'MBPO_workspace/workflows/09_monte_carlo_evaluate.py':mc,'PV_experiments/evaluate.py':evaluate,'MBPO_workspace/workflows/06_model_validation.py':model_validation,'PostProcessing_MC_patch/tools/verify_mc_environment.py':preflight,'PV_experiments/advanced_learning.py':advanced,'PV_experiments/representation/trajectories.py':trajectories})


def build(root):
    import json
    result={}
    prior_hashes=json.loads((HERE/'prior_payload_hashes.json').read_text())
    for path,fn in TRANSFORMS.items():
        base=old.git(root,'show',f'{BASE}:{path}').stdout
        if old.blob(base)!=PIN[path]:raise ValueError(f'Audited base blob differs: {path}')
        original=base.decode('utf-8'); final=fn(original); compile(final,path,'exec')
        file=root/path
        if file.is_symlink() or not file.is_file():raise ValueError(f'Invalid source file {path}')
        current=file.read_bytes().decode('utf-8')
        if current==final:continue
        previous=old.TRANSFORMS[path](original) if path in old.TRANSFORMS else original
        if current not in (original,previous):raise ValueError(f'Local source differs from audited or prior-patch version: {path}')
        result[path]=(current,final)
    for file in sorted((HERE/'payload').rglob('*')):
        if not file.is_file() or '__pycache__' in file.parts:continue
        path=file.relative_to(HERE/'payload').as_posix(); target=root/path
        final=file.read_text(encoding='utf-8')
        if path.endswith('.py'):compile(final,path,'exec')
        if target.is_symlink():raise ValueError(f'Symlink target: {path}')
        if target.exists():
            current=target.read_bytes().decode('utf-8')
            if current==final:continue
            if old.blob(current.encode())!=prior_hashes.get(path):raise ValueError(f'Unexpected local payload: {path}')
        else:
            current=None
            if old.git(root,'check-ignore','--',path,check=False).returncode==0:raise ValueError(f'New source is ignored: {path}')
        result[path]=(current,final)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.')
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--apply',action='store_true');g.add_argument('--dry-run',action='store_true');g.add_argument('--write-patch',metavar='PATH')
    args=p.parse_args(argv);root=Path(args.root).resolve()
    if Path(old.git(root,'rev-parse','--show-toplevel').stdout.decode().strip()).resolve()!=root:raise ValueError('Use Git repository root')
    if old.git(root,'merge-base','--is-ancestor',BASE,'HEAD',check=False).returncode!=0:raise ValueError(f'Expected audited Geometry10 commit {BASE} or descendant with exact supported source files')
    # Guard active unchanged interfaces as well as edited blobs. Other unrelated
    # workspace files and historical outputs are not touched.
    stable = ['mbpo/geometry10.py','SAC_workspace/src/environment/environment.py',
              'SAC_workspace/src/representation/representation.py','SAC_workspace/src/agents/sac_agent.py',
              'SAC_workspace/src/training/replay_buffer.py','SAC_workspace/src/config/config.py',
              'MBPO_workspace/src/config.py','MBPO_workspace/workflows/08_monte_carlo_train.py']
    stable += ['mbpo/env/'+n+'.py' for n in ('pv_tracking','pv_power_backend','pvlib_physics','stage_a_pv_tracking','stage_a_data_contract','trajectory_uncertainty','nsrdb_weather','baseline_contract')]
    stable += ['examples/config/pv_tracking/'+n for n in ('0.py','_nsrdb_base.py','conf_geometry10.py')]
    for name in stable:
        if (root/name).read_bytes()!=old.git(root,'show',f'{BASE}:{name}').stdout:
            raise ValueError(f'Active unchanged dependency differs from audited source: {name}')
    plan=build(root)
    if not plan:print('Already at finalized source version; no changes made.');return
    patch=old.patch_bytes(plan)
    old.git(root,'apply','--check','--whitespace=error-all','-',input=patch)
    print(f'PASS: {len(plan)} files; pinned base, exact supported source, compilation and git apply --check')
    if args.apply:
        old.git(root,'apply','--whitespace=error-all','-',input=patch)
        print('Applied in place. No backup, duplicate workspace, commit or push.')
    elif args.write_patch:Path(args.write_patch).write_bytes(patch)
    else:print('Dry run: no source files modified.')
    for path in plan:print(' ',path)

if __name__=='__main__':
    try:main()
    except Exception as e:print(f'ERROR: {e}',file=sys.stderr);sys.exit(1)
