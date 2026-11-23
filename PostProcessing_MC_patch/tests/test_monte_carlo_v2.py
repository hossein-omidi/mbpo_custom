"""Additional blocking-defect regression tests for reviewed v2 Monte Carlo package.

Offline synthetic tests only; actual NSRDB, Gym and pvlib are needed separately.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace, ModuleType
from datetime import datetime,timedelta,timezone
import importlib.util,sys,tempfile,zipfile,hashlib
import numpy as np
import pytest
import pandas as pd

from test_monte_carlo_weather import W,E,APP,frame,spec


def test_empirical_total_variance_exact_and_finite_mc_bias_corrected():
    # Both days have different deterministic expected returns and identical
    # within-day error patterns. Sample means have finite-draw MC noise.
    rows=[]
    for sid,base in [('2024-01-02',10.),('2024-04-02',20.)]:
        for rid,error in [(0,-1.),(1,1.)]:
            rows.append({'scenario_id':sid,'draw':rid,'method':'sac','net_kwh':base+error})
    x=E.nested_summary(rows,'sac')
    assert x['mean_net_kwh']==pytest.approx(15.)
    assert x['between_day_variance_kwh2']==pytest.approx(25.)
    assert x['within_day_uncertainty_variance_kwh2']==pytest.approx(1.)
    assert x['total_empirical_variance_kwh2']==pytest.approx(26.)
    assert x['total_descriptive_sd_kwh']==pytest.approx(np.sqrt(26.))
    # Per-day unbiased variance 2, two draws each; observed between-day
    # empirical component contains a finite Monte Carlo bias of 0.5.
    assert x['within_day_unbiased_variance_kwh2']==pytest.approx(2.)
    assert x['between_day_variance_corrected_kwh2']==pytest.approx(24.5)
    assert x['mc_se_mean_fixed_days_kwh']==pytest.approx(np.sqrt(.5))


def test_duplicate_draws_and_mismatched_day_draws_rejected():
    rows=[{'scenario_id':'2024-01-02','draw':0,'method':'sac','net_kwh':1.},
          {'scenario_id':'2024-01-02','draw':0,'method':'sac','net_kwh':1.}]
    with pytest.raises(ValueError,match='Duplicate'):
        E.nested_summary(rows,'sac')
    rows=[{'scenario_id':'2024-01-02','draw':0,'method':'sac','net_kwh':1.},
          {'scenario_id':'2024-01-02','draw':1,'method':'sac','net_kwh':2.},
          {'scenario_id':'2024-02-02','draw':1,'method':'sac','net_kwh':3.},
          {'scenario_id':'2024-02-02','draw':2,'method':'sac','net_kwh':4.}]
    with pytest.raises(ValueError,match='Unmatched'):
        E.nested_summary(rows,'sac')


def test_uncertain_ghi_is_never_less_than_uncertain_dhi_and_projection_reported():
    # Severe original closure mismatch: DNI contributes to GHI but the source
    # has almost no GHI. Preserving the residual without a physical projection
    # can result in GHI < DHI under an uncertainty realization.
    f=frame(196)
    f['dni']=300.
    f['dhi']=100.
    f['ghi']=102.
    out,diag=W.generate_weather_profile(f,np.ones(len(f)),spec(dni_cv=.5,dhi_cv=.2),seed=9)
    assert (out['ghi'].to_numpy()>=out['dhi'].to_numpy()).all()
    assert diag['clip_count']>=diag['ghi_dhi_projection_count']
    assert diag['ghi_dhi_projection_count']>0
    assert diag['max_ghi_below_dhi_wm2']==pytest.approx(0.)
    pd.testing.assert_frame_equal(f,frame(196).assign(dni=300.,dhi=100.,ghi=102.))


def test_sun_slew_baseline_is_same_as_existing_current_solar_timestamp():
    class E0:
        def __init__(self):
            self.unwrapped=self
            self.current_time='current'
            self.times=['current','future']
            self.step_index=0
            self.tilt=0.;self.azimuth=180.;self.max_delta_tilt=5.;self.max_delta_azimuth=10.
            self.seen=[]
        def _solar_position(self,t):
            self.seen.append(t)
            return SimpleNamespace(zenith=45.,azimuth=180.)
    # Inject a stub for an optional physical-target module; this test verifies
    # ordering, not the solar geometry formula.
    target=ModuleType('mbpo.env.pvlib_physics')
    target.dual_axis_tracking_targets=lambda zen,az:(40.,180.)
    sys.modules['mbpo.env.pvlib_physics']=target
    e=E0();action=E.action_for('sun_slew_limited',e,None,None)
    assert e.seen==['current']
    assert action[0]==pytest.approx(1.)


def test_v1_upgrade_is_guarded_and_idempotent(tmp_path,monkeypatch):
    # Simulate an already applied v1 package. The 7 existing source modules
    # carry the v1 marker and exact required injected interface snippets.
    # The old V1 ZIP never modified model_validation.py, whose exact raw
    # branch content is separately guarded by the installer in real use.
    monkeypatch.setattr(APP,'PATCHERS',{k:v for k,v in APP.PATCHERS.items()
        if k!='MBPO_workspace/workflows/06_model_validation.py'})
    for rel,anchors in APP.PATCHED_REQUIRED.items():
        if rel=='MBPO_workspace/workflows/06_model_validation.py':continue
        dst=tmp_path/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_text(APP.MARKER+'\n'+'\n'.join(anchors))
    files={'mbpo/env/trajectory_uncertainty.py':'trajectory_uncertainty_v1.py',
           'MBPO_workspace/workflows/08_monte_carlo_train.py':'08_monte_carlo_train_v1.py',
           'MBPO_workspace/workflows/09_monte_carlo_evaluate.py':'09_monte_carlo_evaluate_v1.py'}
    for rel,fixture in files.items():
        dst=tmp_path/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_bytes(Path(__file__).parent.joinpath('fixtures_v1',fixture).read_bytes())
    planned=APP.prepare(tmp_path)
    assert set(planned)=={'mbpo/env/trajectory_uncertainty.py',
                          'MBPO_workspace/workflows/09_monte_carlo_evaluate.py'}
    for rel,(_,new) in planned.items():
        (tmp_path/rel).write_text(new)
    assert APP.prepare(tmp_path)=={}
    # Reject unrelated edits, rather than overwriting local user code.
    (tmp_path/'mbpo/env/trajectory_uncertainty.py').write_text('print("modified")')
    with pytest.raises(ValueError,match='refusing overwrite'):
        APP.prepare(tmp_path)


def test_full_nested_mc_orchestration_exports_every_historical_day(monkeypatch):
    """Exercise complete scenario×draw×method orchestration via a surrogate plant.

    This checks code wiring and statistics, not physical pvlib correctness.
    """
    import types
    class StubPolicy:
        def act(self,x,deterministic=False):
            assert deterministic
            return np.asarray([0.,0.],dtype=np.float32)
    class StubSolarEnv:
        horizon=4
        weather_uncertainty={'enabled':True}
        def __init__(self,kwargs):
            self.unwrapped=self
            self.max_delta_tilt=5.;self.max_delta_azimuth=10.
        def reset(self,scenario_id,realization_id):
            self.sid=scenario_id;self.rid=realization_id;self.step_index=0
            from datetime import date
            start=datetime.fromisoformat(scenario_id+'T08:45:00+00:00')
            self.times=[start+timedelta(minutes=5*j) for j in range(5)]
            self.current_time=self.times[0]
            self.tilt=30.;self.azimuth=180.
            self._weather_realization_seed=W.realization_seed(spec(),scenario_id,realization_id)
            self.weather,self._weather_uncertainty_diagnostics=W.generate_weather_profile(
                frame(5),np.ones(5)*.6,spec(),seed=self._weather_realization_seed)
            self._weather_profile_sha256=hashlib.sha256(
                self.weather.to_numpy().tobytes()).hexdigest()
            return np.array([.2,.2,.2,.3,0.,1.],dtype=np.float32)
        def _solar_position(self,t):
            return SimpleNamespace(zenith=35.,azimuth=180.)
        def step(self,action):
            self.step_index+=1;self.current_time=self.times[self.step_index]
            self.tilt=float(np.clip(self.tilt+action[0]*self.max_delta_tilt,0,90))
            self.azimuth=float((self.azimuth+action[1]*self.max_delta_azimuth)%360)
            row=self.weather.iloc[self.step_index]
            power=0.18*(float(row.dni)+float(row.dhi))
            energy=power/12000.
            return (np.array([row.dni/2000,row.dhi/2000,power/2000,self.tilt/90,
                        self.azimuth/180-1,(4-self.step_index)/4],dtype=np.float32),
                    energy,self.step_index==self.horizon,{
                        'scenario_id':self.sid,'weather_profile_sha256':self._weather_profile_sha256,
                        'weather_realization_id':self.rid,
                        'timestamp_utc_iso':self.current_time.isoformat(),
                        'power_dc_w':power,'energy_kwh':energy,'movement_cost':0.,
                        'dni_wm2':row.dni,'dhi_wm2':row.dhi,'ghi_wm2':row.ghi,
                        'ambient_temperature_c':row.temperature,
                        'wind_speed_m_s':row.wind_speed,'surface_albedo':row.surface_albedo})
        def close(self):pass
    mod=ModuleType('SAC_workspace.src.environment.environment')
    mod.SACEnvironment=StubSolarEnv
    monkeypatch.setitem(sys.modules,mod.__name__,mod)
    # Stub astronomy target imported by the baseline's policy function.
    stub=ModuleType('mbpo.env.pvlib_physics')
    stub.dual_axis_tracking_targets=lambda z,a:(30.,180.)
    monkeypatch.setitem(sys.modules,stub.__name__,stub)
    import hashlib
    cfg=SimpleNamespace(sac=SimpleNamespace(eval_env={}))
    rows,clock=E.evaluate_paths(cfg,{'sac':StubPolicy(),'mbpo':StubPolicy()},
        ['2024-06-21','2024-11-21'],draws=3,selected_time_scenario='2024-06-21')
    assert len(rows)==2*3*4
    assert len(clock)==2*4*4
    assert {r['scenario_id'] for r in clock}=={'2024-06-21','2024-11-21'}
    assert all(r['post_closure_max_wm2']>=0 for r in rows)
    by={(r['scenario_id'],r['draw'],r['method']):r for r in rows}
    for sid in ['2024-06-21','2024-11-21']:
        for rid in range(3):
            assert by[(sid,rid,'sac')]['profile_sha256']==by[(sid,rid,'mbpo')]['profile_sha256']
            assert by[(sid,rid,'sac')]['net_kwh']==pytest.approx(by[(sid,rid,'mbpo')]['net_kwh'])


def test_model_validation_counterfactual_replay_is_fixed_across_resets():
    import ast
    # Minimal source fragment exercising every unique anchored code path.
    original='''from pathlib import Path
import json

def _run_policy_trajectory(env, agent, scenario_id):
    obs = env.reset(scenario_id=scenario_id)
    return obs

def run_validation(args):
    cfg = load_config(
        model_overrides=model_overrides,
    )
    states, actions, next_states, rewards = _run_policy_trajectory(env, agent, sid)
    for sid in []:
        for t in []:
            for x in []:
                for z in []:
                    replay_obs = env.reset(scenario_id=sid)
    summary = {
            "validation_scenario_ids": scenario_ids,
            "interpretation": [
                "Counterfactual tests replay the same policy prefix to reconstruct an observed state, then execute alternative actions in the real simulator.",
            ],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states-per-scenario", type=int, default=6)
    if args.states_per_scenario < 1:
        raise ValueError("wrong states")
'''
    patched=APP.patch_model_validation(original)
    ast.parse(patched)
    assert 'weather_uncertainty=uncertainty_spec' in patched
    assert 'realization_id=args.realization_id' in patched
    assert 'realization_id=realization_id' in patched
    assert 'validation_realization_id' in patched


def test_contradictory_albedo_aliases_are_not_hidden_by_uncertainty_augmentation():
    f=frame();f['Surface Albedo']=.9
    with pytest.raises(ValueError,match='Conflicting NSRDB surface-albedo'):
        W.generate_weather_profile(f,np.ones(len(f))*.6,spec(),seed=3)


def test_mc_eval_rejects_changed_weather_even_if_profile_hash_is_same(monkeypatch):
    # Hash-only checking can miss a reporting bug that inadvertently changes
    # controller-specific weather fields. Verify the actual paired weather too.
    class FakeEnv:
        def __init__(self,kw):
            self.weather_uncertainty={'enabled':True}
            self.unwrapped=self
            self._weather_uncertainty_diagnostics={'clip_count':0,
                'max_baseline_closure_residual_wm2':0.,
                'max_post_closure_residual_wm2':0.}
            self.horizon=1
        def close(self):pass
    mod=ModuleType('SAC_workspace.src.environment.environment')
    mod.SACEnvironment=FakeEnv
    monkeypatch.setitem(sys.modules,mod.__name__,mod)
    def conflicting_weather(env,method,sid,rid,policy=None):
        return {'profile_sha256':'same-hash','pose':(30.,180.),
            'timestamps_utc':['2024-06-21T08:50:00+00:00'],
            'weather':np.asarray([[500.+(method=='sac'),100.,400.,18.,2.,.2]]),
            'power_w':np.array([1.]),'gross_kwh':.1,'movement_kwh':0.,
            'net_kwh':.1,'realization_seed':123}
    monkeypatch.setattr(E,'run_one',conflicting_weather)
    with pytest.raises(RuntimeError,match='meteorological values differ'):
        E.evaluate_paths(SimpleNamespace(sac=SimpleNamespace(eval_env={})),
                         {'sac':object(),'mbpo':object()},['2024-06-21'],2)
