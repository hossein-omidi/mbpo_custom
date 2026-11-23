"""Unit-level mathematical tests; no PVLib/Gym or actual NSRDB downloads needed."""
from __future__ import annotations
import importlib.util,json,sys,math,types
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PKG=Path(__file__).resolve().parents[1]


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

W=load(PKG/'new_files/mbpo/env/trajectory_uncertainty.py','mc_uncertainty')
# Avoid importing the physical simulator for numerical-only tests.
for package in ('mbpo','mbpo.env'):
    mod=types.ModuleType(package)
    mod.__path__=[]
    sys.modules.setdefault(package,mod)
sys.modules['mbpo.env.trajectory_uncertainty']=W
E=load(PKG/'new_files/MBPO_workspace/workflows/09_monte_carlo_evaluate.py','mc_eval')
APP=load(PKG/'apply_local.py','mc_applier')


def frame(n=196):
    return pd.DataFrame({'dni':np.full(n,500.),'dhi':np.full(n,100.),
           'ghi':np.full(n,400.),'temperature':np.full(n,18.),
           'wind_speed':np.full(n,2.),'surface_albedo':np.full(n,.2)})


def spec(**overrides):
    return W.normalized_spec({'enabled':True,'dni_cv':.05,'dhi_cv':.05,
        'temperature_sd_c':1.,'wind_cv':.05,'albedo_sd':.02,**overrides})


def test_zero_uncertainty_exact_and_immutable_inputs():
    original=frame();copy=original.copy(deep=True)
    out,diag=W.generate_weather_profile(original,np.full(len(original),.6),{},seed=3)
    pd.testing.assert_frame_equal(out,copy)
    pd.testing.assert_frame_equal(original,copy)
    assert not diag['enabled']


def test_reproducible_common_random_numbers_and_unique_draws():
    s=spec()
    seed1=W.realization_seed(s,'2024-07-03',5)
    seed2=W.realization_seed(s,'2024-07-03',5)
    seed3=W.realization_seed(s,'2024-07-03',6)
    assert seed1==seed2 and seed1!=seed3
    source=frame();z=np.full(len(source),.6)
    x,dx=W.generate_weather_profile(source,z,s,seed=seed1)
    y,dy=W.generate_weather_profile(source,z,s,seed=seed2)
    other,_=W.generate_weather_profile(source,z,s,seed=seed3)
    pd.testing.assert_frame_equal(x,y)
    assert not np.array_equal(x['dni'].values,other['dni'].values)
    pd.testing.assert_frame_equal(source,frame())
    assert dx['clip_count']==0


def test_preserved_closure_residual_and_bounds():
    source=frame();source.loc[:,'ghi']=420.  # deliberately nonzero baseline residual
    cosz=np.full(len(source),.6)
    output,diagnostics=W.generate_weather_profile(source,cosz,spec(),seed=19)
    base=source.ghi-source.dhi-source.dni*cosz
    closure=output.ghi-output.dhi-output.dni*cosz
    assert np.max(np.abs(base-closure))<1e-9
    assert diagnostics['max_post_closure_residual_wm2']==pytest.approx(20.)
    for col in ('dni','dhi','ghi','wind_speed'):
        assert (output[col]>=0).all() and np.isfinite(output[col]).all()
    assert output.surface_albedo.between(0,1).all()


def test_nighttime_zeros_remain_zero():
    f=frame();f.loc[:,'dni']=0.;f.loc[:,'dhi']=0.;f.loc[:,'ghi']=0.
    out,_=W.generate_weather_profile(f,np.zeros(len(f)),spec(),seed=4)
    assert (out[['dni','dhi','ghi']].values==0).all()


def test_covariance_and_ar1_statistics():
    s=spec(day_fraction=0.,ar1=.75)
    rng=np.random.default_rng(99)
    z=W._latent_path(s,30000,rng)
    corr=np.corrcoef(z.T)
    assert abs(corr[0,1]-.2)<.035
    assert abs(np.corrcoef(z[:-1,0],z[1:,0])[0,1]-.75)<.035
    assert abs(z[:,0].std()-1.)<.07


def test_mean_one_lognormal_multiplier_is_unbiased():
    s=spec(day_fraction=0.,ar1=0.)
    z=W._latent_path(s,60000,np.random.default_rng(881))[:,0]
    sigma=math.sqrt(math.log1p(.05**2))
    mult=np.exp(sigma*z-.5*sigma*sigma)
    assert abs(mult.mean()-1.)<.001
    assert abs(mult.std()-.05)<.001


def test_bad_correlation_and_bad_parameters_fail():
    with pytest.raises(ValueError,match='positive definite'):
        W.normalized_spec({'latent_correlation':[[1]*5]*5})
    with pytest.raises(ValueError,match='ar1'):
        W.normalized_spec({'ar1':1.})
    with pytest.raises(ValueError,match='dni_cv'):
        W.normalized_spec({'dni_cv':-0.1})
    with pytest.raises(ValueError,match='Unexpected uncertainty'):
        W.normalized_spec({'unverified_error_pct':5})
    with pytest.raises(ValueError,match='zero'):
        W.normalized_spec({'enabled':True})


def test_error_in_sun_vector_rejected():
    with pytest.raises(ValueError,match='cos_zenith'):
        W.generate_weather_profile(frame(),np.ones(195),spec(),seed=2)


def test_crn_nested_variance_law():
    rows=[]
    for sid in ('2019-07-04','2024-07-04'):
        for draw in range(3):
            base=10 if sid=='2019-07-04' else 20
            for method in E.METHODS:
                value=base+draw*(1 if method=='mbpo' else 0.5)
                rows.append({'scenario_id':sid,'draw':draw,'method':method,'net_kwh':value})
    result=E.nested_summary(rows,'mbpo')
    assert result['n_historical_days']==2 and result['draws_per_day']==3
    assert result['mean_net_kwh']==pytest.approx(16.)
    assert result['total_descriptive_sd_kwh']**2==pytest.approx(
        result['between_day_variance_kwh2']+result['within_day_uncertainty_variance_kwh2'])
    refs={(r['scenario_id'],r['draw'],r['method']):r['net_kwh'] for r in rows}
    diff=E.nested_summary(rows,refs,('mbpo','sac'))
    assert diff['mean_net_kwh']==pytest.approx(.5)


def test_marked_incremental_patch_and_fingerprints():
    assert len(APP.EXPECTED)==8
    assert set(APP.PATCHERS)==set(APP.EXPECTED)
    assert all(len(v)==40 for v in APP.EXPECTED.values())
    content=(PKG/'apply_local.py').read_text()
    assert 'shutil.copy' not in content
    assert "git push" not in content.split('def main():')[1]


def test_complete_closed_loop_episode_uses_same_correlated_weather_for_paired_policies():
    """End-to-end controller/realization orchestration with a synthetic plant.

    This verifies Monte Carlo *control flow*, not the actual pvlib physics.
    """
    import types,hashlib
    from datetime import datetime,timedelta,timezone
    solmod=types.ModuleType('mbpo.env.pvlib_physics')
    solmod.dual_axis_tracking_targets=lambda zen,az:(30.,180.)
    sys.modules['mbpo.env.pvlib_physics']=solmod
    cfg=spec()
    baseline=frame(n=5)
    class FakeEnv:
        horizon=4
        def __init__(self):
            self.unwrapped=self
            self.max_delta_tilt=5.;self.max_delta_azimuth=10.
            self.times=[datetime(2024,6,21,8,45,tzinfo=timezone.utc)+timedelta(minutes=5*k)
                        for k in range(5)]
        def _solar_position(self,t):return types.SimpleNamespace(zenith=40.,azimuth=180.)
        def reset(self,scenario_id,realization_id):
            self.sid=scenario_id;self.draw=realization_id;self.step_index=0
            self.current_time=self.times[0]
            seed=W.realization_seed(cfg,scenario_id,realization_id)
            self.profile,self._weather_uncertainty_diagnostics=W.generate_weather_profile(baseline,np.full(5,.6),cfg,seed=seed)
            self._weather_realization_seed=seed
            self._weather_profile_sha256=hashlib.sha256(self.profile.to_numpy().tobytes()).hexdigest()
            self.tilt=30.;self.azimuth=180.
            return np.array([.25,.05,.01,.333,0,1.],dtype=np.float32)
        def step(self,action):
            self.tilt=np.clip(self.tilt+float(action[0])*5,0,90)
            self.azimuth=(self.azimuth+float(action[1])*10)%360
            self.step_index+=1
            self.current_time=self.times[self.step_index]
            x=self.profile.iloc[self.step_index]
            power=(float(x.dni)+float(x.dhi))*(1-abs(self.tilt-30)/180.)*.18
            energy=power/12000.
            return (np.array([x.dni/2000.,x.dhi/2000.,power/2000.,self.tilt/90.,self.azimuth/180.-1.,
                              (4-self.step_index)/4],dtype=np.float32),
                    energy,self.step_index==4,{
                     'scenario_id':self.sid,'weather_profile_sha256':self._weather_profile_sha256,
                     'weather_realization_id':self.draw,'timestamp_utc_iso':self.times[self.step_index].isoformat(),
                     'power_dc_w':power,'energy_kwh':energy,'movement_cost':0.,
                     'dni_wm2':x.dni,'dhi_wm2':x.dhi,'ghi_wm2':x.ghi,
                     'ambient_temperature_c':x.temperature,'wind_speed_m_s':x.wind_speed,
                     'surface_albedo':x.surface_albedo})
    class FakePolicy:
        def act(self,obs,deterministic=False):
            assert deterministic
            return np.array([1.,0.],dtype=np.float32)
    env=FakeEnv()
    fixed=E.run_one(env,'fixed_no_motion','2024-06-21',5)
    sac=E.run_one(env,'sac','2024-06-21',5,FakePolicy())
    mbpo=E.run_one(env,'mbpo','2024-06-21',5,FakePolicy())
    sun=E.run_one(env,'sun_slew_limited','2024-06-21',5)
    assert len(sac['power_w'])==4
    assert fixed['profile_sha256']==sac['profile_sha256']==mbpo['profile_sha256']==sun['profile_sha256']
    assert np.array_equal(fixed['weather'],sac['weather'])
    assert sac['net_kwh']<fixed['net_kwh']
    assert sac['net_kwh']==pytest.approx(mbpo['net_kwh'])
    assert np.isclose(sun['net_kwh'],fixed['net_kwh'])
    # Repeating a draw does not depend on prior episode action histories.
    repeat=E.run_one(env,'fixed_no_motion','2024-06-21',5)
    assert np.array_equal(repeat['power_w'],fixed['power_w'])
