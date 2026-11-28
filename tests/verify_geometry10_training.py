#!/usr/bin/env python3
"""Read-only acceptance audit for managed Geometry10 SAC/MBPO training runs.
Run from the original repository: python verify_geometry10_training.py --name EXP.
No training, model changes, or writes to experiment artifacts. Trusted local .pt only.
Checks committed bookkeeping, checkpoint/replay contracts, and actual synthetic usage.
A zero-use MBPO run is a valid fallback execution but fails the MBPO exposure criterion.
Passing is not a guarantee of convergence, field accuracy or model counterfactual skill.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(row, key):
    n = int(row[key])
    require(n >= 0, f'Negative {key}')
    return n


def finite(row, key):
    v = float(row[key]); require(math.isfinite(v), f'Nonfinite {key}'); return v


def close(a, b, msg, *, atol=1e-9, rtol=1e-8):
    require(math.isclose(a,b,abs_tol=atol,rel_tol=rtol), f'{msg}: {a} != {b}')


def audit_rows(rows, cfg, meta, split):
    """Validate every episode, including exact synthetic draws from pre/post-refit pools."""
    alg=meta['algorithm']; shared=cfg if alg=='sac' else cfg['sac']; a=shared['algo']
    h=int(meta['horizon']); E=int(meta['final_epoch']); warm=int(a['start_steps']); u=int(a['updates_per_step']); B=int(a['batch_size']); cap=int(a['replay_size'])
    require(h==195 and E>0 and len(rows)==E and int(meta['final_env_steps'])==E*h,'Invalid episode/horizon budget')
    require(warm>=B and cap>=B and u>0,'Unsupported warmup/replay schedule for exact audit')
    train=set(split['train_ids']); prev_updates=prev_pool=fits=attempts=passes=generated=used=real=violations=checked=raw_w=raw_p=0
    if alg=='mbpo':
        m=cfg['model']; freq=int(m['model_train_freq']); roll=int(m['rollout_batch_size']); maxfake=B-int(B*float(m['real_ratio'])); fakecap=min(int(m['model_pool_capacity']),int(m['model_retain_epochs'])*roll*int(m['rollout_length']))
        require(freq>0 and freq%h==0 and int(m['rollout_length'])==1 and bool(m['require_model_skill']), 'Unsupported model schedule or model-skill gate disabled')
    for i,r in enumerate(rows,1):
        try:
            steps=i*h; expected=u*max(0,steps-warm); du=expected-prev_updates
            require(count(r,'epoch')==i and count(r,'env_steps')==steps and count(r,'updates')==expected,'Episode/update grid mismatch')
            require(r['scenario_id'] in train,'Training scenario outside frozen training split')
            gross=finite(r,'gross_energy_kwh'); cost=finite(r,'movement_cost_kwh'); net=finite(r,'return_kwh')
            require(gross>=0 and cost>=0,'Negative gross or actuator energy'); close(gross-cost,net,'Physical energy identity')
            require(finite(r,'alpha')>=float(a['min_alpha'])-1e-8,'Entropy floor violated')
            for key in ('critic_loss','actor_loss','alpha_loss','entropy','q_mean','target_mean'):
                v=float(r[key]); require(math.isfinite(v) if du else math.isnan(v),f'Incorrect availability of {key}')
            if du: require(float(r['critic_loss'])>=0,'Negative critic MSE')
            if alg=='mbpo':
                scheduled=int(steps>warm and steps%freq==0); att=count(r,'model_fit_attempted'); fit=count(r,'model_fit_completed'); passed=count(r,'model_skill_pass')
                require(att==scheduled and 0<=passed<=fit<=att,'Fit schedule/gate accounting mismatch')
                attempts+=att; fits+=fit; passes+=passed
                require(count(r,'model_fits')==fits,'Cumulative ensemble fit count mismatch')
                require(count(r,'real_replay_size')==min(steps,cap),'Real replay capacity/count mismatch')
                if fit:
                    for key in ('model_holdout_scenarios','model_train_scenarios'):
                        require(finite(r,key)>=int(m['min_holdout_scenarios']), 'Fit lacked required distinct days')
                    sp=finite(r,'model_skill_power_vs_persistence'); sa=finite(r,'model_skill_all_channels_vs_persistence')
                    require(passed==int(sp>0 and sa>0),'Persistence gate result inconsistent')
                    finite(r,'model_val_nll')
                n=count(r,'synthetic_attempted'); accepted=count(r,'synthetic_accepted'); rejected=count(r,'synthetic_rejected')
                require(accepted+rejected==n and count(r,'synthetic_generated')==accepted and n in (0,roll) and (not n or passed),'Synthetic rollout generation inconsistent')
                pool=count(r,'model_replay_size')
                expected_pool=(0 if scheduled and not passed else min(fakecap,prev_pool+accepted))
                require(pool==expected_pool,'Synthetic pool did not follow gate/FIFO accounting')
                # Refits occur at the LAST transition of an episode, BEFORE its updates.
                last=u if steps>warm else 0
                expected_fake=(du-last)*min(prev_pool,maxfake)+last*min(pool,maxfake)
                f=count(r,'synthetic_samples_used'); rr=count(r,'real_samples_used')
                require(f==expected_fake and rr+f==du*B,'Actual mixed-batch draws differ from configured schedule')
                if du:
                    close(finite(r,'mean_model_batch'),f/du,'Mean synthetic samples per update')
                    close(finite(r,'mean_real_batch'),rr/du,'Mean real samples per update')
                generated+=accepted; used+=f; real+=rr; prev_pool=pool
                c=count(r,'envelope_checked'); v=count(r,'envelope_violations'); sat=count(r,'envelope_irradiance_saturated')
                require(v<=c and c+sat<=n,'Invalid synthetic envelope counters');checked+=c;violations+=v
                rw=count(r,'raw_weather_outside');rp=count(r,'raw_power_outside');require(rw<=n and rp<=n,'Invalid raw clipping counters');raw_w+=rw;raw_p+=rp
            prev_updates=expected
        except (KeyError,ValueError,TypeError) as e:
            raise ValueError(f'{alg} seed {meta["seed"]}, episode {i}: {e}') from e
    result=dict(algorithm=alg,seed=meta['seed'],episodes=E,committed_real_transitions=E*h,optimizer_updates=prev_updates)
    if alg=='mbpo':
        result.update(fit_attempts=attempts,completed_fits=fits,skill_passes=passes,accepted_synthetic_transitions=generated,synthetic_batch_draws=used,real_batch_draws=real,actual_synthetic_fraction=used/(used+real) if used+real else None,synthetic_exposure_status='nonzero' if used else 'fallback_only',synthetic_envelope_checked=checked,synthetic_envelope_violations=violations,raw_weather_clipped_rows=raw_w,raw_power_clipped_rows=raw_p)
    return result


def audit_checkpoint(root,cfg,meta,summary):
    import numpy as np
    import torch
    from mbpo.geometry10 import check_batch,pose_transition
    from mbpo.audit_support import sha_file
    from mbpo.audit_release import check_final_checkpoint
    require(sha_file(root/'latest.pt')==meta['final_checkpoint_sha256'],'Final checkpoint hash mismatch')
    check_final_checkpoint(root,meta)
    ck=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
    ag=ck['agent']; U=summary['optimizer_updates']; require(ag['updates']==U,'Final update count mismatch')
    def finite_tree(x):
        if torch.is_tensor(x): require(bool(torch.isfinite(x).all()),'Nonfinite checkpoint tensor')
        elif isinstance(x,dict):
            for v in x.values(): finite_tree(v)
        elif isinstance(x,(list,tuple)):
            for v in x: finite_tree(v)
    finite_tree(ag)
    for key,params in [('actor_opt',ag['actor']),('critic_opt',ag['critic']),('alpha_opt',{'alpha':ag['log_alpha']})]:
        opt=ag[key]; states=opt['state']
        require(len(states)==(len(params) if U else 0),f'{key}: optimizer parameter state missing')
        for v in states.values(): require(float(v['step'])==U,f'{key}: optimizer step mismatch')
    mdp=ck['resolved_mdp']; h=meta['horizon']; scale=float(mdp['movement_penalty']);power=float(mdp['actuator_power_w']);vt=float(mdp['slew_rate_tilt_deg_s']);va=float(mdp['slew_rate_azimuth_deg_s'])
    require(mdp['movement_cost_mode']=='geometry','Wrong reward contract')
    def replay(data,expected_count):
        n=int(data['count']); cap=int(data['capacity']); head=int(data['head']);require(n==expected_count and 0<=n<=cap and 0<=head<cap and (n==cap or head==n),'Invalid replay count/head')
        s=check_batch(data['states']); t=check_batch(data['next_states']); a=np.asarray(data['actions']);r=np.asarray(data['rewards']);d=np.asarray(data['dones']);require(s.shape==(n,10) and t.shape==s.shape and a.shape==(n,2) and r.shape==d.shape==(n,1),'Replay tensor shape mismatch')
        require(np.isfinite(r).all() and np.isin(d,[0,1]).all(),'Invalid replay reward/done')
        if n:
            tilt,pair,tau,dt,da,done=pose_transition(s,a,h)
            require(np.allclose(t[:,3],tilt,atol=1e-6,rtol=0) and np.allclose(t[:,4:6],pair,atol=2e-6,rtol=0),'Replay executed geometry mismatch')
            require(np.allclose(t[:,9],tau,atol=1e-6,rtol=0) and np.array_equal(d[:,0].astype(bool),done),'Replay terminal/time mismatch')
            expected=t[:,2].astype(float)/6-scale*power*(abs(dt)/vt+abs(da)/va)/3.6e6
            require(np.allclose(r[:,0],expected,atol=5e-8,rtol=1e-6),'Replay reward inconsistent with physical net-energy accounting')
    alg=meta['algorithm']; shared=cfg if alg=='sac' else cfg['sac']; real_count=min(summary['committed_real_transitions'],shared['algo']['replay_size'])
    if alg=='sac': replay(ck['replay'],real_count)
    else:
        from MBPO_workspace.src.replay import ScenarioReplayBuffer
        from MBPO_workspace.src.dynamics import DynamicsEnsemble
        rp=ck['real_pool'];replay(rp,real_count)
        with (root/'episodes.csv').open(newline='') as f:last=list(csv.DictReader(f))[-1]
        replay(ck['model_pool'],int(last['model_replay_size']))
        m=cfg['model'];rb=ScenarioReplayBuffer(shared['algo']['replay_size'],validation_fraction=m['model_holdout_fraction']);rb.load_state_dict(rp)
        split=json.loads((root/'scenario_split.json').read_text());require(set(rb.scenario_names)<=set(split['train_ids']),'Replay contains held-out policy days')
        dyn=ck['dynamics'];finite_tree(dyn);require(dyn['fit_count']==summary['completed_fits'] and bool(dyn['fitted'])==bool(dyn['fit_count']),'Checkpoint ensemble fit state mismatch')
        model=DynamicsEnsemble(m['num_models'],m['num_elites'],m['hidden_dim'],lr=m['model_lr']);model.load_state_dict(dyn)
        for elite in (model.weather_elites,model.power_elites):require(len(elite)==len(set(elite))==m['num_elites'],'Invalid ensemble elites')
        if model.fitted:
            for opt in model.weather_optimizers+model.power_optimizers:require(bool(opt.state),'Fitted ensemble optimizer state absent')
    summary['checkpoint_replay_and_optimizer_audit']='passed'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',required=True);args=p.parse_args()
    sys.path.insert(0,str(Path.cwd()))
    from PV_experiments.core import load_experiment,verify_pair,inspect_training_run,run_path
    from PV_experiments.scientific_checks import verify_run_comparability,recovery_accounting
    from mbpo.audit_support import runtime_identity
    plan=load_experiment(args.name); require(plan.get('training_runtime')==runtime_identity(),'Current training source/dependencies differ from frozen plan; audit using matching runtime')
    records=[];runs=[]
    for seed in plan['seeds']:
        split=verify_pair(plan,seed)
        for alg in ('sac','mbpo'):
            status,meta=inspect_training_run(plan,alg,seed);require(status=='completed','Incomplete run')
            root=run_path(plan,alg,seed);cfg=json.loads((root/'config.json').read_text())
            with (root/'episodes.csv').open(newline='') as f:rows=list(csv.DictReader(f))
            result=audit_rows(rows,cfg,meta,split);audit_checkpoint(root,cfg,meta,result)
            result.update(recovery_accounting(root,result['committed_real_transitions'],meta['horizon']))
            records.append(result);runs.append(dict(root=root,metadata=meta))
    verify_run_comparability(runs)
    for r in records:print(json.dumps(r,sort_keys=True,allow_nan=False))
    inactive=[r['seed'] for r in records if r['algorithm']=='mbpo' and r['synthetic_exposure_status']=='fallback_only']
    print('Structural training-chain audit: PASS')
    if inactive:
        print(f'MBPO exposure criterion: FAIL (zero synthetic minibatch draws in seeds {inactive}); these are fallback-only runs, not evidence of active model-based learning.',file=sys.stderr);return 2
    print('MBPO exposure criterion: PASS (nonzero usage in each seed; no claim of adequate model skill or convergence)');return 0

if __name__=='__main__':raise SystemExit(main())
