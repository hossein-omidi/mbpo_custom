"""Stage-A unit and production integration tests.

The unit-level mathematical oracle checks run without pvlib. The pvlib/NSRDB
integration tests are SKIPPED outside the user's real project environment;
they MUST pass in WSL before model retraining.
"""
from __future__ import annotations
import ast
import math
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def _load_isotropic_max_tilt():
    # Pure math function can be validated independently from pvlib installation.
    path=ROOT/'MBPO_workspace/src/stage_a_reference.py'
    tree=ast.parse(path.read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef)
                  and node.name=='isotropic_max_tilt_with_albedo')
    code=compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                 str(path),'exec')
    namespace={'math':math}
    exec(code,namespace)
    return namespace['isotropic_max_tilt_with_albedo']


def test_analytic_isotropic_poa_oracle_beats_dense_grid():
    f=_load_isotropic_max_tilt()
    for z,dni,ghi,dhi,rho in [(35,800,700,125,.09),
                               (70,900,800,80,.8),
                               (50,0,450,420,.07),
                               (89,20,500,400,.4),
                               (33,400,300,250,0)]:
        def poa(b):
            rad=math.radians(b)
            zrad=math.radians(z)
            return (dni*max(0.,math.cos(zrad-rad))+
                    dhi*(1+math.cos(rad))/2 + rho*ghi*(1-math.cos(rad))/2)
        tilt=f(z,dni,ghi,dhi,rho)
        assert 0<=tilt<=90
        samples=np.linspace(0,90,9001)
        assert poa(tilt)>=max(poa(float(v)) for v in samples)-1e-7


def test_invalid_albedo_oracle_rejected():
    f=_load_isotropic_max_tilt()
    for rho in [-.1,1.5,float('nan')]:
        with pytest.raises(ValueError):f(45,700,650,150,rho)
    with pytest.raises(ValueError):f(105,100,0,0,.2)


def test_actual_pvlib_backend_and_simulator():
    pytest.importorskip('pvlib')
    pytest.importorskip('gym')
    from mbpo.env.pv_power_backend import compute_pv_power
    from SAC_workspace.src.config.config import load_config
    from SAC_workspace.src.environment.environment import SACEnvironment
    from MBPO_workspace.src.stage_a_reference import ideal_instantaneous_poa
    from SAC_workspace.src.evaluation.evaluation import run_episode
    kw=dict(power_model='pvwatts_sapm',temp_air_c=24,wind_speed_ms=2,
            wind_reference_height_m=2,area=1,efficiency=.18)
    less=compute_pv_power(70,180,45,180,800,700,120,surface_albedo=.09,**kw)
    more=compute_pv_power(70,180,45,180,800,700,120,surface_albedo=.50,**kw)
    assert more.poa_global_wm2>less.poa_global_wm2
    assert more.power_w>=less.power_w
    assert more.effective_irradiance_wm2==more.poa_global_wm2
    cfg=load_config(power_model='pvwatts_sapm')
    env=SACEnvironment(cfg.eval_env)
    try:
        state=env.reset(scenario_id='2024-06-21')
        assert state.shape==(6,)
        nstate,reward,done,info=env.step(np.zeros(2,dtype=np.float32))
        assert nstate.shape==(6,) and not done
        assert info['albedo_source']=='nsrdb_recorded'
        frame=env.unwrapped.weather_profile
        row=frame.loc[env.unwrapped.current_time]
        original=next(float(row[k]) for k in ('surface_albedo','albedo','Surface Albedo')
                      if k in row)
        assert info['surface_albedo']==pytest.approx(original)
        assert reward==pytest.approx(info['energy_kwh']-info['movement_cost'])
        fixed=run_episode(cfg,'fixed_no_motion','2024-06-21',42,env=env)
        oracle=ideal_instantaneous_poa(cfg,fixed,env=env)
        assert len(oracle['power'])==117
        assert all(p+1e-5>=f for p,f in zip(oracle['power'],fixed['power']))
    finally:
        env.close()
