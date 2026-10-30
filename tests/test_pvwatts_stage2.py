"""Stage-2 end-to-end contract tests. Run AFTER the guarded installer --apply.

python -m pytest -q tests/test_pv_power_backend.py tests/test_pvwatts_stage2.py

Tests with real NSRDB data require local CSV/manifest. They SKIP (never PASS)
when the dataset is absent. Use scripts/pvwatts_stage2_audit.py for fail-closed gate.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from mbpo.env.pv_power_backend import compute_pv_power
from mbpo.env.pvlib_physics import compute_panel_power_w
from mbpo.env.nsrdb_weather import default_manifest_path
from SAC_workspace.src.config.config import load_config
from SAC_workspace.src.environment.environment import SACEnvironment
from MBPO_workspace.src.config import load_config as load_mbpo_config
from MBPO_workspace.src.evaluation import _check_physics


SAMPLE = '2022-07-15'
PV = dict(power_model='pvwatts_sapm', wind_reference_height_m=2.0,
          gamma_pdc=-0.004, sapm_mount='open_rack_glass_polymer',
          wind_powerlaw_exponent=0.14)


def _needs_dataset():
    if not Path(default_manifest_path()).is_file():
        pytest.skip('Local NSRDB catalog absent; cannot claim dataset integration PASS')


def test_shared_sac_and_mbpo_physical_config():
    for kw in ({}, PV):
        sac=load_config(require_data=False, **kw)
        mbpo=load_mbpo_config(require_data=False, **kw)
        assert sac.to_dict() == mbpo.sac.to_dict()
        for name in ('train_env','eval_env'):
            x=getattr(sac,name)
            if kw:
                for key,val in PV.items(): assert x[key]==val
            else:
                assert 'power_model' not in x  # true historical config compatibility
        assert sac.train_env['observation_mode']=='compact'
        assert sac.algo['hidden_sizes']==(64,64)


def test_legacy_real_simulator_checkpoint_metadata_unchanged():
    _needs_dataset()
    sac=load_config(require_data=True)
    env=SACEnvironment(sac.eval_env)
    try:
        assert env.unwrapped.power_model=='legacy'
        assert 'power_model' not in env.resolved_mdp
        assert len(env.resolved_mdp)>10
    finally:env.close()


def test_new_model_snapshot_and_guard():
    _needs_dataset()
    a=SACEnvironment(load_config(**PV).eval_env)
    b=SACEnvironment(load_config().eval_env)
    try:
        for key,val in PV.items(): assert a.resolved_mdp[key]==val
        with pytest.raises(ValueError,match='mismatch'):
            _check_physics(a.resolved_mdp,b.resolved_mdp)
        with pytest.raises(ValueError,match='mismatch'):
            _check_physics(b.resolved_mdp,a.resolved_mdp)
        assert a.horizon==117 and a.observation_space.shape==(6,)
        assert a.action_space.shape==(2,)
    finally:a.close();b.close()


@pytest.mark.parametrize('phys', [{},PV], ids=['legacy','pvwatts'])
def test_real_step_and_reset_use_same_backend(phys):
    _needs_dataset()
    cfg=load_config(**phys)
    options=dict(cfg.eval_env, fixed_eval_scenarios=[SAMPLE],
                 randomize_scenario=False, randomize_initial_orientation=False)
    env=SACEnvironment(options)
    try:
        env.seed(42);s=env.reset(scenario_id=SAMPLE)
        assert s.shape==(6,)
        assert s[5]==pytest.approx(1.0)
        # Direct recomputation at the exact reset timestamp. No step duplication.
        u=env.unwrapped
        ts=u.times[0]; pos=u._solar_position(ts); w=u._current_weather()
        expected=compute_pv_power(u.tilt,u.azimuth,float(pos.zenith),float(pos.azimuth),
              w['dni'],w['ghi'],w['dhi'],area=u.area,efficiency=u.efficiency,
              power_model=u.power_model,temp_air_c=w['temperature'],
              wind_speed_ms=w['wind_speed'],wind_reference_height_m=u.wind_reference_height_m,
              gamma_pdc=u.gamma_pdc,sapm_mount=u.sapm_mount,
              wind_powerlaw_exponent=u.wind_powerlaw_exponent)
        assert u.last_power==pytest.approx(expected.power_w,abs=1e-9)
        for step in range(117):
            action=np.array([math.sin(step/12.),math.cos(step/18.)],dtype=np.float32)
            s,r,done,info=env.step(action)
            expected=compute_pv_power(info['tilt'],info['azimuth'],
                      info['solar_zenith_deg'],info['solar_azimuth_deg'],
                      info['dni_wm2'],info['ghi_wm2'],info['dhi_wm2'],
                      area=u.area,efficiency=u.efficiency,
                      power_model=u.power_model,temp_air_c=info['temperature_c'],
                      wind_speed_ms=info['wind_speed_m_s'],
                      wind_reference_height_m=u.wind_reference_height_m,
                      gamma_pdc=u.gamma_pdc,sapm_mount=u.sapm_mount,
                      wind_powerlaw_exponent=u.wind_powerlaw_exponent)
            assert info['power']==pytest.approx(expected.power_w,abs=1e-8)
            assert info['power_dc_w']==pytest.approx(expected.power_w,abs=1e-8)
            assert info['poa_global']==pytest.approx(expected.poa_global_wm2,abs=1e-8)
            assert info['cell_temperature_c']==expected.cell_temperature_c
            assert info['energy_kwh']==pytest.approx(info['power']*5./60./1000.,abs=1e-12)
            assert r==pytest.approx(info['energy_kwh']-info['movement_cost'],abs=1e-12)
            assert (step==116)==bool(done)
            assert s.shape==(6,) and np.isfinite(s).all()
            assert s[5]==pytest.approx((116-step)/117.,abs=1e-6)
            assert info['power_model']==phys.get('power_model','legacy')
            if not phys:
                assert info['power']==pytest.approx(compute_panel_power_w(
                       info['tilt'],info['azimuth'],info['solar_zenith_deg'],
                       info['solar_azimuth_deg'],info['dni_wm2'],info['ghi_wm2'],
                       info['dhi_wm2'],area=u.area,efficiency=u.efficiency),abs=1e-8)
    finally:env.close()


def test_evaluation_fixed_and_selected_model_ideal():
    _needs_dataset()
    from MBPO_workspace.src.evaluation import ideal_instantaneous_poa
    from SAC_workspace.src.evaluation.evaluation import run_episode
    cfg=load_config(**PV)
    e=SACEnvironment({**cfg.eval_env,'fixed_eval_scenarios':[SAMPLE]})
    try:
        fixed=run_episode(cfg,'fixed_no_motion',SAMPLE,42,env=e)
        ideal=ideal_instantaneous_poa(cfg,fixed,env=e)
        assert len(ideal['power'])==117
        assert ideal['gross_energy_kwh']+1e-8>=fixed['gross_energy_kwh']
        for i,info in enumerate(ideal['infos']):
            assert info['power_model']=='pvwatts_sapm'
            assert info['power']==pytest.approx(ideal['power'][i],abs=1e-8)
            assert info['cell_temperature_c'] is not None
            assert info['poa_global']>=0
            assert ideal['power'][i]+1e-4 >= fixed['power'][i]
    finally:e.close()


def test_pvwatts_vs_independent_pvlib_oracle_and_code_fingerprint():
    """Independent direct pvlib path; must not reuse compute_pv_power as oracle."""
    _needs_dataset()
    import hashlib
    import pvlib
    from pvlib import atmosphere, temperature, pvsystem, irradiance
    cfg=load_config(**PV)
    env=SACEnvironment({**cfg.eval_env,'fixed_eval_scenarios':[SAMPLE]})
    try:
        u=env.unwrapped
        for key,source in (
            ('pv_power_backend_sha256','mbpo/env/pv_power_backend.py'),
            ('pvlib_physics_sha256','mbpo/env/pvlib_physics.py'),
        ):
            expected=hashlib.sha256((Path(__file__).resolve().parents[1]/source).read_bytes()).hexdigest()
            assert env.resolved_mdp[key]==expected
        env.reset(scenario_id=SAMPLE)
        for j in range(117):
            _,r,_,info=env.step(np.array([math.sin(j/13),math.cos(j/15)],np.float32))
            if j not in (0,16,44,69,96,116):continue
            irr=irradiance.get_total_irradiance(
                surface_tilt=info['tilt'],surface_azimuth=info['azimuth'],
                solar_zenith=info['solar_zenith_deg'],
                solar_azimuth=info['solar_azimuth_deg'],
                dni=info['dni_wm2'],ghi=info['ghi_wm2'],dhi=info['dhi_wm2'],
                model='isotropic')
            poa=max(float(irr['poa_global']),0.)
            wind10=float(atmosphere.windspeed_powerlaw(
                info['wind_speed_m_s'],height_reference=2.,height_desired=10.,exponent=.14))
            tcell=float(temperature.sapm_cell(
                poa_global=poa,temp_air=info['temperature_c'],wind_speed=wind10,
                **temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']))
            power=max(0.,float(pvsystem.pvwatts_dc(
                effective_irradiance=poa,temp_cell=tcell,
                pdc0=1000*u.area*u.efficiency,gamma_pdc=-.004,temp_ref=25.)))
            assert info['poa_global']==pytest.approx(poa,abs=1e-8)
            assert info['cell_temperature_c']==pytest.approx(tcell,abs=1e-9)
            assert info['power_dc_w']==pytest.approx(power,abs=1e-8)
            assert info['power']==pytest.approx(power,abs=1e-8)
            assert info['energy_kwh']==pytest.approx(power/12000.,abs=1e-10)
            assert r==pytest.approx(info['energy_kwh']-info['movement_cost'],abs=1e-10)
    finally:env.close()


def test_pvwatts_rejects_invalid_mount_and_wind_inputs():
    kwargs=dict(surface_tilt=30, surface_azimuth=180, solar_zenith=45,
        solar_azimuth=190,dni=700,ghi=810,dhi=150,area=1,efficiency=.18,
        power_model='pvwatts_sapm',temp_air_c=22,wind_speed_ms=3,
        wind_reference_height_m=2.)
    with pytest.raises(ValueError):
        compute_pv_power(**{**kwargs,'sapm_mount':'unknown'})
    with pytest.raises(ValueError):
        compute_pv_power(**{**kwargs,'wind_reference_height_m':None})
    with pytest.raises(ValueError):
        compute_pv_power(**{**kwargs,'wind_speed_ms':float('nan')})
