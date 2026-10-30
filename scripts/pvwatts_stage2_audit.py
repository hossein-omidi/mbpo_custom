"""Fail-closed Stage 2 gate against the REAL local NSRDB/PV lib environment.

Run after copying the Stage 2 source files, before new training:
  python scripts/pvwatts_stage2_audit.py

Checks both physics backends, SAC + MBPO loader parity, complete 117-step
SAC/MBPO warm-up trainer episodes, and selected-power evaluation reference.
No previous checkpoints or experiments are modified. Exits 0 only if ALL pass.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys
import tempfile
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))



def validate_paired_power_paths(paths, expected_methods, reference_env, compute_fn):
    """Fail-closed physical and bookkeeping check for every evaluation path.

    Real policies and counterfactual references expose different *info* dictionaries.
    Their canonical common contract is path['power'][i] and
    info['power_dc_w'], not info['power'] (the latter is only present for
    direct simulator steps and some ideal reference implementations).

    Recompute each orientation/weather sample through the selected canonical
    physics backend. This is stronger than comparing two copied output fields.
    The backend's independent pvlib equivalence is separately tested by the
    Stage 2 tests; this is a full evaluation-plumbing check.
    """
    import math

    seen = set()
    u = reference_env.unwrapped
    interval_hours = float(u.interval_hours)
    if not math.isfinite(interval_hours) or interval_hours <= 0:
        raise RuntimeError('Invalid reference simulator time interval')
    expected_model = str(u.power_model)
    tolerance_w = 1e-6

    for path in paths:
        name = path['method']
        if name in seen:
            raise RuntimeError('Duplicate evaluation method: ' + str(name))
        seen.add(name)
        for key in ('power', 'energy', 'reward', 'cost', 'tilt', 'azimuth', 'infos'):
            if len(path[key]) != reference_env.horizon:
                raise RuntimeError('Incomplete evaluation sequence: ' + name + '/' + key)

        for i, (info, power, energy, reward, cost, tilt, azimuth) in enumerate(zip(
                path['infos'], path['power'], path['energy'], path['reward'],
                path['cost'], path['tilt'], path['azimuth'])):
            context = f'{name} step {i + 1}'
            if info.get('power_model') != expected_model:
                raise RuntimeError('Evaluation mixed physical backends: ' + context)
            # Require the standard *evaluation* DC-power diagnostic; never
            # silently substitute the path power if the diagnostic is absent.
            required = ('power_dc_w', 'poa_global', 'cell_temperature_c',
                        'effective_irradiance_wm2', 'wind_speed_10m_ms',
                        'temperature_c', 'wind_speed_m_s', 'solar_zenith_deg',
                        'solar_azimuth_deg', 'dni_wm2', 'ghi_wm2', 'dhi_wm2')
            missing = [key for key in required if info.get(key) is None]
            if missing:
                raise RuntimeError(f'Missing evaluation physics data {context}: {missing}')
            numbers = (power, energy, reward, cost, tilt, azimuth,
                       *(info[key] for key in required))
            if not all(math.isfinite(float(x)) for x in numbers):
                raise RuntimeError('Non-finite evaluation power or weather: ' + context)
            if float(power) < -tolerance_w or float(info['poa_global']) < -tolerance_w:
                raise RuntimeError('Negative DC power or POA: ' + context)
            if abs(float(info['power_dc_w']) - float(power)) > tolerance_w:
                raise RuntimeError('Evaluation DC diagnostic disagrees with trajectory: ' + context)
            if info.get('power') is not None and abs(float(info['power']) - float(power)) > tolerance_w:
                raise RuntimeError('Optional simulator power diagnostic disagrees: ' + context)
            if abs(float(energy) - float(power) * interval_hours / 1000.0) > 1e-9:
                raise RuntimeError('Evaluation energy disagrees with DC power: ' + context)
            if abs(float(reward) - (float(energy) - float(cost))) > 1e-8:
                raise RuntimeError('Evaluation reward/energy/cost mismatch: ' + context)

            computed = compute_fn(
                tilt, azimuth, info['solar_zenith_deg'], info['solar_azimuth_deg'],
                info['dni_wm2'], info['ghi_wm2'], info['dhi_wm2'],
                area=u.area, efficiency=u.efficiency,
                power_model=expected_model,
                temp_air_c=info['temperature_c'],
                wind_speed_ms=info['wind_speed_m_s'],
                wind_reference_height_m=u.wind_reference_height_m,
                gamma_pdc=u.gamma_pdc, sapm_mount=u.sapm_mount,
                wind_powerlaw_exponent=u.wind_powerlaw_exponent)
            for label, measured, expected in (
                ('DC power', power, computed.power_w),
                ('POA', info['poa_global'], computed.poa_global_wm2),
                ('effective irradiance', info['effective_irradiance_wm2'], computed.effective_irradiance_wm2),
                ('cell temperature', info['cell_temperature_c'], computed.cell_temperature_c),
                ('wind speed at 10 m', info['wind_speed_10m_ms'], computed.wind_speed_10m_ms),
            ):
                if expected is None or abs(float(measured) - float(expected)) > 1e-6:
                    raise RuntimeError(f'{label} inconsistent with selected pvlib model: {context}')
    if seen != set(expected_methods):
        raise RuntimeError(f'Evaluation method coverage mismatch: expected {sorted(expected_methods)}, got {sorted(seen)}')
    return sum(len(path['power']) for path in paths)


def gate():
    import pvlib
    from mbpo.env.nsrdb_weather import load_scenario_manifest, default_manifest_path
    from SAC_workspace.src.config.config import load_config
    from MBPO_workspace.src.config import load_config as mbpo_load
    from SAC_workspace.src.environment.environment import SACEnvironment
    from SAC_workspace.src.training.trainer import SACTrainer
    from MBPO_workspace.src.trainer import MBPOTrainer
    from MBPO_workspace.src.evaluation import ideal_instantaneous_poa,_check_physics
    from SAC_workspace.src.evaluation.evaluation import run_episode
    from mbpo.env.pv_power_backend import compute_pv_power

    if tuple(int(x) for x in pvlib.__version__.split('.')[:2]) < (0,13):
        raise RuntimeError('pvlib >=0.13 is required for the upgraded backend')
    manifest=load_scenario_manifest(default_manifest_path())
    ids={row['scenario_id'] for row in manifest['scenarios']}
    scenario='2022-07-15'
    if scenario not in ids:raise RuntimeError('Required test scenario missing: '+scenario)
    opts=dict(power_model='pvwatts_sapm',wind_reference_height_m=2.,
              gamma_pdc=-.004,sapm_mount='open_rack_glass_polymer',wind_powerlaw_exponent=.14)
    old=load_config();new=load_config(**opts)
    new_mbpo=mbpo_load(**opts)
    if new.to_dict()!=new_mbpo.sac.to_dict():
        raise RuntimeError('SAC and MBPO physics/configuration mismatch')
    if 'power_model' in old.train_env or 'power_model' in old.eval_env:
        raise RuntimeError('Historical legacy environment contract changed')
    for name in ('train_env','eval_env'):
        if not all(getattr(new,name)[k]==v for k,v in opts.items()):
            raise RuntimeError('Physical model parameters did not reach ' + name)
    print('[PASS] Exact SAC/MBPO configuration parity; historical legacy unaffected')

    el=SACEnvironment({**old.eval_env,'fixed_eval_scenarios':[scenario]})
    en=SACEnvironment({**new.eval_env,'fixed_eval_scenarios':[scenario]})
    try:
        if (el.horizon != 117 or en.horizon != 117
                or el.observation_space.shape!=(6,) or en.observation_space.shape!=(6,)
                or el.action_space.shape!=(2,) or en.action_space.shape!=(2,)):
            raise RuntimeError('Unexpected SAC/MBPO simulator action/observation contract')
        if 'power_model' in el.resolved_mdp:
            raise RuntimeError('Legacy checkpoint metadata contract changed')
        if not all(en.resolved_mdp.get(k)==v for k,v in opts.items()):
            raise RuntimeError('New checkpoint physical metadata incomplete')
        try:_check_physics(el.resolved_mdp,en.resolved_mdp)
        except ValueError:pass
        else:raise RuntimeError('Old checkpoint not rejected under new model')
        for e in (el,en):
            e.seed(42);s=e.reset(scenario_id=scenario)
            for j in range(e.horizon):
                a=np.array([math.sin(j/11),math.cos(j/13)],dtype=np.float32)
                s,reward,done,info=e.step(a)
                u=e.unwrapped
                result=compute_pv_power(info['tilt'],info['azimuth'],
                    info['solar_zenith_deg'],info['solar_azimuth_deg'],
                    info['dni_wm2'],info['ghi_wm2'],info['dhi_wm2'],
                    area=u.area,efficiency=u.efficiency,
                    power_model=u.power_model,temp_air_c=info['temperature_c'],
                    wind_speed_ms=info['wind_speed_m_s'],
                    wind_reference_height_m=u.wind_reference_height_m,
                    gamma_pdc=u.gamma_pdc,sapm_mount=u.sapm_mount,
                    wind_powerlaw_exponent=u.wind_powerlaw_exponent)
                if abs(result.power_w-info['power'])>1e-7:raise RuntimeError('Backend mismatch')
                if abs(result.poa_global_wm2-info['poa_global'])>1e-7:raise RuntimeError('POA mismatch')
                if abs(reward-info['energy_kwh']+info['movement_cost'])>1e-9:
                    raise RuntimeError('Reward identity broken')
                if done != (j==116) or s.shape!=(6,):raise RuntimeError('Horizon/observation mismatch')
            print(f'[PASS] {u.power_model}: complete 117-step real pvlib episode and physics/reward parity')
        fixed=run_episode(new,'fixed_no_motion',scenario,42,env=en)
        oracle=ideal_instantaneous_poa(new,fixed,env=en)
        if len(oracle['infos'])!=117 or oracle['gross_energy_kwh']+1e-9<fixed['gross_energy_kwh']:
            raise RuntimeError('PVWatts ideal reference not consistent with selected power')
        for j,row in enumerate(oracle['infos']):
            if row['power_model']!='pvwatts_sapm' or abs(row['power']-oracle['power'][j])>1e-7:
                raise RuntimeError('Ideal reference reports old physics')
        print('[PASS] PVWatts physical evaluation reference dominates fixed mount on paired trajectory')
    finally:el.close();en.close()

    # One full trainer episode each. Start_steps is 8000, so no expensive learning
    # or model fits; training optimizer/synthetic fit is tested later in training gate.
    with tempfile.TemporaryDirectory(prefix='pvwatts_stage2_') as td:
        sac=SACTrainer(new,Path(td)/'sac',seed=211,device='cpu')
        mbpo=MBPOTrainer(new_mbpo,Path(td)/'mbpo',seed=211,device='cpu')
        try:
            if sac.env.unwrapped.power_model != 'pvwatts_sapm' or mbpo.env.unwrapped.power_model != 'pvwatts_sapm':
                raise RuntimeError('Trainers are not using the new PVWatts physics')
            if sac.env.resolved_mdp != mbpo.env.resolved_mdp:
                raise RuntimeError('SAC and MBPO trainers resolved different physical MDPs')
            x=sac.train_episode();y=mbpo.train_episode()
            for z in (x,y):
                if not np.isfinite(z['gross_energy_kwh']):raise RuntimeError('Nonfinite trainer gross energy')
            print('[PASS] SAC and MBPO actual 117-step trainer episodes, same new physical MDP')
            # Short-run checkpoints are NOT converged policies: this tests the
            # genuine model identity and paired-EVALUATION plumbing only.
            sac_ck=sac.save('stage2_sac_smoke.pt')
            mbpo_ck=mbpo.save('stage2_mbpo_smoke.pt')
            from MBPO_workspace.src.evaluation import evaluate as evaluate_pair, write_outputs
            from MBPO_workspace.src.advanced_suite import export_detail, _read_eval
            paths=evaluate_pair(new_mbpo,mbpo_ck,sac_ck,
                scenarios=[scenario],seeds=(42,),device='cpu',
                include_poa=True,include_classical=True)
            expected_methods={'mbpo','sac','fixed_no_motion',
                'dual_axis_slew_limited','dual_axis_instant_ideal','poa_instant_ideal'}
            if {p['method'] for p in paths}!=expected_methods:
                raise RuntimeError('Paired checkpoint evaluation omitted a method')
            checked = validate_paired_power_paths(paths, expected_methods, sac.env, compute_pv_power)
            if checked != 6 * 117:
                raise RuntimeError('Paired power check did not cover every simulated interval')
            report_dir=Path(td)/'evaluation'
            stats=write_outputs(paths,report_dir)
            detail=export_detail(paths,report_dir)
            if not detail.is_file() or 'power_physics' not in stats:
                raise RuntimeError('Power physics was not persisted to evaluation outputs')
            with detail.open(encoding='utf-8') as h:
                header=h.readline().strip().split(',')
            for key in ('power_model','poa_global_wm2','cell_temperature_c',
                        'wind_speed_m_s','wind_speed_10m_ms'):
                if key not in header: raise RuntimeError('Missing reporting field '+key)
            _read_eval(report_dir)  # read back the actual new report, not a mock fixture
            print('[PASS] New-physics checkpoint load, all six paired methods and reporting round-trip')
        finally:sac.env.close();mbpo.env.close()

    # Exercise the *learned* MBPO path with two real NSRDB days. This is a real
    # optimizer/ensemble/synthetic transition smoke test, not predictive-accuracy
    # evidence. All temporary state stays outside the repository.
    from MBPO_workspace.src.replay import mix_batches
    full_cfg=mbpo_load(model_overrides=dict(
        model_epochs=2,model_patience=2,model_batch_size=32,
        model_max_train_samples=234,rollout_batch_size=32), **opts)
    mixed_replay_verified = False
    with tempfile.TemporaryDirectory(prefix='pvwatts_stage2_model_') as td:
        trainer=MBPOTrainer(full_cfg,Path(td)/'mbpo',seed=42,device='cpu')
        try:
            for sid in ('2022-01-15','2022-07-15'):
                if sid not in ids:raise RuntimeError('Missing distinct learned-model test scenario '+sid)
                state=trainer.env.reset(scenario_id=sid)
                for i in range(trainer.env.horizon):
                    action=np.array([math.sin(i/17),math.cos(i/19)],dtype=np.float32)
                    nxt,reward,done,info=trainer.env.step(action)
                    trainer.real_pool.add(state,action,reward,nxt,done,info['scenario_id'])
                    state=nxt
            if len(trainer.real_pool) != 234:
                raise RuntimeError('Real MBPO replay did not receive two complete 117-step days')
            generated,rejected,uncertainty=trainer._fit_and_rollout()
            if not trainer.dynamics.fitted or trainer.dynamics.fit_count<1:
                raise RuntimeError('Actual MBPO dynamics ensemble did not fit')
            # Exercise fake environment mechanics independently of the
            # model-skill safety gate; do NOT insert untrusted transitions.
            states=trainer.real_pool.rollout_states(16)
            import torch
            with torch.no_grad():
                act,_=trainer.agent.actor(torch.as_tensor(states,dtype=torch.float32),
                                          with_logprob=False)
            fake_states,fake_rewards,fake_done,uncertainties,valid=trainer.fake_env.step(
                states,act.numpy().astype(np.float32),max_std=None)
            if (fake_states.shape!=(16,6) or fake_rewards.shape!=(16,)
                    or not np.isfinite(fake_states).all()
                    or not np.isfinite(fake_rewards).all()
                    or not np.isfinite(uncertainties).all()):
                raise RuntimeError('Learned synthetic transition is invalid')
            skipped=bool(trainer.latest_model_metrics.get('model_skipped'))
            if skipped and len(trainer.model_pool):
                raise RuntimeError('Model skill gate rejected fit but synthetic replay was populated')
            if not skipped and (generated<=0 or len(trainer.model_pool)<1):
                raise RuntimeError('Model skill gate passed but no replay transitions were generated')
            batch,n_real,n_model=mix_batches(trainer.real_pool,trainer.model_pool,
                32,full_cfg.model.real_ratio,trainer.rng)
            if n_real<=0 or (not skipped and n_model<=0) or (skipped and n_model!=0):
                raise RuntimeError('SAC batch sampling bypassed the model-skill gate')
            metrics=trainer.agent.update(batch)
            if not all(np.isfinite(v) for v in metrics.values()):
                raise RuntimeError('Nonfinite SAC optimizer update')
            print('[PASS] Real-data MBPO dynamics fit, learned synthetic step, '
                  f'skill-gated replay ({n_real} real / {n_model} model), '
                  'and SAC optimizer update')
            if skipped:
                print('[NOT VERIFIED] Learned ensemble did not satisfy the model-skill '
                      'gate in this 2-epoch mini-fit; mixed synthetic replay '
                      'was correctly blocked, not bypassed.')
            else:
                mixed_replay_verified = True
                print('[PASS] Full real/synthetic mixture exercised with '
                      f'{generated} safety-accepted generated transitions.')
            print('      Two-epoch model fitting is NOT a predictive-accuracy assessment.')
        finally:trainer.env.close()

    if not mixed_replay_verified:
        print('\nSTAGE 2 AUDIT PARTIALLY VERIFIED: physical evaluation, reports,')
        print('checkpoint interfaces and model-skill gate passed, but mixed model')
        print('replay was not exercised because the fitted ensemble was rejected.')
        print('This is not a software defect; repeat with a competent learned ensemble')
        print('before claiming the SAC/MBPO synthetic training mixture is validated.')
        return 3
    print('\nSTAGE 2 INTEGRATION AUDIT PASS — real physics, shared configs, checkpoint')
    print('evaluation, reporting, dynamics fitting, synthetic replay and guarded SAC updates.')
    print('Scientific accuracy and converged-policy performance require independent validation.')
    return 0


if __name__=='__main__':
    try: sys.exit(gate())
    except Exception as exc:
        print('\nSTAGE 2 AUDIT FAILED OR INCOMPLETE:',type(exc).__name__,str(exc),file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
