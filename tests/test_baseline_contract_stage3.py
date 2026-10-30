"""Regression and independent numerical checks for Stage 3 comparison baselines."""
from math import cos, radians
import numpy as np
import pytest
# Import the exact production file without requiring Gym/pvlib for pure math tests.
import importlib.util
from pathlib import Path
_spec=importlib.util.spec_from_file_location(
    'baseline_contract_stage3',
    Path(__file__).resolve().parents[1] / 'mbpo/env/baseline_contract.py')
_contract=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_contract)
isotropic_max_poa_tilt=_contract.isotropic_max_poa_tilt
certify_pvwatts_monotonicity=_contract.certify_pvwatts_monotonicity
validate_paired_reference_group=_contract.validate_paired_reference_group


def analytic_poa(z,dn,gh,dh,tilt):
    theta=radians(tilt)
    return (dn*max(0.,cos(radians(z)-theta)) +
            dh*(1+cos(theta))/2 + 0.25*gh*(1-cos(theta))/2)


def test_analytic_oracle_is_global_for_isotropic_model_random_cases():
    rng=np.random.default_rng(20261030)
    grid=np.linspace(0.,90.,1801)
    for _ in range(250):
        z=float(rng.uniform(0,180));dn=float(rng.uniform(0,1200))
        gh=float(rng.uniform(0,1250));dh=float(rng.uniform(0,700))
        chosen=isotropic_max_poa_tilt(z,dn,gh,dh)
        estimated=analytic_poa(z,dn,gh,dh,chosen)
        sampled=max(analytic_poa(z,dn,gh,dh,float(x)) for x in grid)
        assert 0 <= chosen <= 90
        assert estimated >= sampled-1e-8


def test_analytic_oracle_dominates_global_two_axis_grid():
    rng=np.random.default_rng(817)
    for _ in range(75):
        z=float(rng.uniform(0,180));dn=float(rng.uniform(0,1200))
        gh=float(rng.uniform(0,1250));dh=float(rng.uniform(0,700))
        sunaz=float(rng.uniform(0,360))
        selected=isotropic_max_poa_tilt(z,dn,gh,dh)
        upper=analytic_poa(z,dn,gh,dh,selected)
        for tilt in np.linspace(0.,90.,61):
            b=radians(float(tilt));zr=radians(z)
            for az in np.linspace(0.,360.,17,endpoint=False):
                incidence=(cos(zr)*cos(b)+np.sin(zr)*np.sin(b)*cos(radians(sunaz-float(az))))
                poa=(dn*max(0.,incidence)+dh*(1+cos(b))/2+
                     .25*gh*(1-cos(b))/2)
                assert upper+1e-8 >= poa


def test_oracle_boundary_cases():
    # Pure sky diffuse makes horizontal orientation optimal.
    assert isotropic_max_poa_tilt(40,0,0,100)==pytest.approx(0.)
    # Strong ground reflection can make vertical orientation optimal.
    assert isotropic_max_poa_tilt(40,0,1000,0)==pytest.approx(90.)
    assert isotropic_max_poa_tilt(120,0,0,0) in (0.,90.)
    for bad in ((10,-1,50,20),(190,100,100,30),(float('nan'),100,30,20)):
        with pytest.raises(ValueError):isotropic_max_poa_tilt(*bad)


class FakePower:
    def __init__(self,poa,cell):
        self.poa_global_wm2=poa;self.cell_temperature_c=cell


def test_pvwatts_monotonicity_positives_and_fail_closed():
    assert certify_pvwatts_monotonicity(FakePower(1000,48),25,-.004)>0
    with pytest.raises(ValueError,match='not monotone'):
        certify_pvwatts_monotonicity(FakePower(1000,400),25,-.004)
    with pytest.raises(ValueError,match='heating'):
        certify_pvwatts_monotonicity(FakePower(1000,10),25,-.004)


def make_path(name, power=110., model='pvwatts_sapm'):
    base={'solar_zenith_deg':35.,'solar_azimuth_deg':165.,
          'dni_wm2':800.,'ghi_wm2':750.,'dhi_wm2':150.,
          'ambient_temperature_c':30.,'wind_speed_m_s':2.,
          'wind_reference_height_m':2.,'gamma_pdc':-.004,
          'sapm_mount':'open_rack_glass_polymer','wind_powerlaw_exponent':.14,
          'power_model':model,'control_interval_minutes':5.}
    from datetime import datetime,timezone,timedelta
    initial=[30.,180.]
    infos=[]
    for i in range(117):
        info=dict(base)
        info['timestamp_utc_iso']=(datetime(2022,7,15,12,5,tzinfo=timezone.utc)+timedelta(minutes=5*i)).isoformat()
        info['power_dc_w']=power
        infos.append(info)
    ideal=name in ('poa_instant_ideal','dual_axis_instant_ideal')
    energy=power/12000.
    return dict(method=name,scenario_id='2022-07-15',seed=42,initial_pose=initial,
                power=[power]*117,energy=[energy]*117,reward=[energy]*117,
                cost=[0.]*117,tilt=[30.]*117,azimuth=[180.]*117,
                action_tilt=[float('nan') if ideal else 0.]*117,
                action_azimuth=[float('nan') if ideal else 0.]*117,
                delta_tilt_deg=[float('nan') if ideal else 0.]*117,
                delta_azimuth_deg=[float('nan') if ideal else 0.]*117,
                net_energy_kwh=None if ideal else 117*energy,
                gross_energy_kwh=117*energy,
                movement_cost_kwh=None if ideal else 0.,infos=infos)


def test_paired_baseline_contract_accepts_consistent_real_and_ideal():
    paths={k:make_path(k,p) for k,p in [('fixed_no_motion',110.),
              ('sac',120.),('mbpo',125.),('poa_instant_ideal',140.)]}
    result=validate_paired_reference_group(paths,require_oracle=True)
    assert result['power_model']=='pvwatts_sapm' and result['steps']==117


def corrupt_oracle_but_keep_accounting(paths):
    p=paths['poa_instant_ideal'];i=20
    p['power'][i]=115.
    p['energy'][i]=115./12000.
    p['reward'][i]=115./12000.
    p['infos'][i]['power_dc_w']=115.


@pytest.mark.parametrize('mutation,part',[
    (lambda p: p['mbpo']['infos'][8].__setitem__('wind_speed_m_s',3.),'Unpaired weather'),
    (lambda p: p['sac']['infos'][8].__setitem__('ambient_temperature_c',None),'Missing comparable'),
    (lambda p: p['mbpo']['infos'][1].__setitem__('power_model','legacy'),'Mixed power models'),
    (lambda p: p['fixed_no_motion']['tilt'].__setitem__(50,40.),'orientation changed'),
    (lambda p: p['fixed_no_motion']['action_tilt'].__setitem__(9,0.5),'action or movement'),
    (lambda p: corrupt_oracle_but_keep_accounting(p),'oracle underestimates'),
    (lambda p: p['sac']['infos'][8].__setitem__('power_dc_w',200.),'Power metadata mismatch'),
    (lambda p: p['poa_instant_ideal'].__setitem__('net_energy_kwh',1.),'feasible net'),
    (lambda p: p['sac']['infos'][8].__setitem__('timestamp_utc_iso','fake'),'UTC clock'),
])
def test_baseline_guard_rejects_mismatched_physics_and_false_oracles(mutation,part):
    paths={k:make_path(k,p) for k,p in [('fixed_no_motion',110.),
              ('sac',120.),('mbpo',125.),('poa_instant_ideal',140.)]}
    mutation(paths)
    with pytest.raises(ValueError,match=part):
        validate_paired_reference_group(paths,require_oracle=True)
