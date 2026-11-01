"""No network, training, checkpoint, or real-NSRDB access required for these tests."""
from __future__ import annotations
import csv
from datetime import datetime,timezone,timedelta
from pathlib import Path
import math
import numpy as np
import pytest
from scripts.stage5_midday_diagnostic import (
    parse_utc, signed_angle, solar_cos_aoi, materialize_trajectories,
    identify_noon, clock_diagnosis, time_profile, summary_profile,choose_branch_steps,
    calendar_overlap, stratified_day_select,
)


def fixture_csv(path,invalid_time=False,duplicate=False,different_clock=False):
    fields=['scenario_id','seed','method','step','utc_iso','power_w','energy_kwh',
      'tilt_deg','azimuth_deg','action_tilt','action_azimuth','dni_wm2','dhi_wm2',
      'solar_zenith_deg','solar_azimuth_deg','poa_global_wm2','cell_temperature_c']
    rows=[]
    for day in ('2024-01-15','2024-07-15'):
        for method in ('sac','mbpo','fixed_no_motion','poa_instant_ideal'):
            for step in range(1,118):
                dt=datetime.fromisoformat(day+'T12:00:00+00:00')+timedelta(minutes=5*step)
                if invalid_time and day=='2024-01-15' and method=='mbpo' and step==40:
                    stamp='2024-01-15T00:060:00+00:00'
                elif different_clock and day=='2024-01-15' and method=='mbpo' and step==40:
                    stamp=(dt+timedelta(minutes=1)).isoformat()
                else:stamp=dt.isoformat()
                z=20+abs(step-60)*.4
                p=180 if method=='sac' else 170 if method=='mbpo' else 160 if method=='fixed_no_motion' else 185
                rows.append(dict(scenario_id=day,seed=42,method=method,step=step,
                    utc_iso=stamp,power_w=p,energy_kwh=p/12000.,tilt_deg=30,azimuth_deg=180,
                    action_tilt='' if method=='poa_instant_ideal' else 0,
                    action_azimuth='' if method=='poa_instant_ideal' else 0,
                    dni_wm2=800,dhi_wm2=100,solar_zenith_deg=z,solar_azimuth_deg=180,
                    poa_global_wm2=900,cell_temperature_c=40))
    if duplicate:rows.append(rows[0].copy())
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def test_utc_strict_clock_and_dst():
    with pytest.raises(ValueError,match='Invalid ISO'):parse_utc('2024-01-15T00:117:00+00:00')
    with pytest.raises(ValueError,match='Naive'):parse_utc('2024-01-15T12:05:00')
    from zoneinfo import ZoneInfo
    ny=ZoneInfo('America/New_York')
    assert parse_utc('2024-01-15T12:00:00Z').astimezone(ny).strftime('%H:%M %z')=='07:00 -0500'
    assert parse_utc('2024-07-15T12:00:00Z').astimezone(ny).strftime('%H:%M %z')=='08:00 -0400'
    assert parse_utc('2024-01-15T21:45:00Z').astimezone(ny).strftime('%H:%M')=='16:45'
    assert parse_utc('2024-07-15T21:45:00Z').astimezone(ny).strftime('%H:%M')=='17:45'


def test_geometry_wrap_and_action_sign_math():
    assert abs(signed_angle(359.9,.1)+.2)<1e-9
    assert solar_cos_aoi(0,180,0,180)==pytest.approx(1.)
    assert solar_cos_aoi(0,180,90,180)==pytest.approx(0.,abs=1e-12)
    assert solar_cos_aoi(30,180,30,180)==pytest.approx(1.,abs=1e-12)
    assert solar_cos_aoi(30,0,30,180)==pytest.approx(.5,abs=1e-12)


def test_matched_117_step_series_and_noon(tmp_path):
    fn=tmp_path/'trajectories_detailed.csv'
    fixture_csv(fn)
    bypair=materialize_trajectories(fn)
    assert len(bypair)==2
    assert set(bypair[('2024-01-15',42)])=={'sac','mbpo','fixed_no_motion','poa_instant_ideal'}
    winter=bypair[('2024-01-15',42)]['fixed_no_motion']
    assert identify_noon(winter)==59
    assert choose_branch_steps(winter,[-2,0,2])==[58,60,62]
    clock=clock_diagnosis(bypair)
    assert {r['offset_hours'] for r in clock}=={-5.0,-4.0}
    assert {r['utc_noon_step'] for r in clock}=={60}
    summary=summary_profile(time_profile(bypair))
    assert len(summary)>0
    assert all(r['mean_wh_agent_minus_fixed']>0 for r in summary)
    selected=stratified_day_select(bypair,2,42)
    assert len(selected)==2


@pytest.mark.parametrize('bad',('invalid_time','duplicate','different_clock'))
def test_detailed_audit_rejects_corruption(tmp_path,bad):
    fn=tmp_path/'trajectories_detailed.csv'
    fixture_csv(fn,**{bad:True})
    with pytest.raises(ValueError):materialize_trajectories(fn)


def test_missing_training_history_not_certified(tmp_path):
    fn=tmp_path/'trajectories_detailed.csv';fixture_csv(fn)
    bypair=materialize_trajectories(fn)
    result=calendar_overlap(bypair,{'sac':None,'mbpo':None})
    assert not result['sac']['heldout_certified']
    assert not result['mbpo']['heldout_certified']


def test_training_overlap_detection(tmp_path):
    fn=tmp_path/'trajectories_detailed.csv';fixture_csv(fn)
    bypair=materialize_trajectories(fn)
    tr=tmp_path/'episodes.csv'
    tr.write_text('scenario_id,epoch,env_steps\n2024-01-15,1,117\n')
    x=calendar_overlap(bypair,{'sac':tr,'mbpo':tr})
    assert x['sac']['n_eval_overlap']==1
    assert x['mbpo']['heldout_certified'] is False


def test_real_pvlib_5min_daylight_optional():
    pytest.importorskip('pvlib')
    from scripts.stage5_midday_diagnostic import daylight_duration_by_day
    data=daylight_duration_by_day('2024-06-21',40.72,-74.01)
    assert data['sunlit_hours']>14
    assert 0<data['fraction_sunlit_hours_in_episode']<1
    assert data['morning_sunlit_hours_excluded']>0
    assert data['evening_sunlit_hours_excluded']>0


def test_nonpaired_clock_and_power_corruption(tmp_path):
    path=tmp_path/'trajectories_detailed.csv'
    fixture_csv(path)
    original=path.read_text()
    corrupted=original.replace(',180,0.015,',',180,0.032,',1)
    # Make an explicit single-field energy mistake instead of relying on text coincidence.
    rows=list(csv.DictReader(path.open()))
    rows[0]['energy_kwh']='100.0'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match='Power-to-energy mismatch'):
        materialize_trajectories(path)


def test_dst_transition_utc_cadence_survives():
    from zoneinfo import ZoneInfo
    ny=ZoneInfo('America/New_York')
    # Spring DST transition 2024-03-10 02:00 local jumps directly to 03:00.
    before=parse_utc('2024-03-10T06:55:00+00:00')
    after=parse_utc('2024-03-10T07:00:00+00:00')
    assert after-before==timedelta(minutes=5)
    assert before.astimezone(ny).hour==1
    assert after.astimezone(ny).hour==3
    # Fall DST repeats a local hour, but UTC still progresses in 5-minute steps.
    before=parse_utc('2024-11-03T05:55:00+00:00')
    after=parse_utc('2024-11-03T06:00:00+00:00')
    assert after-before==timedelta(minutes=5)
    assert before.astimezone(ny).hour==1
    assert after.astimezone(ny).hour==1
    assert before.astimezone(ny).utcoffset()!=after.astimezone(ny).utcoffset()


def test_oracle_upper_bound_contract(tmp_path):
    from scripts.stage5_midday_diagnostic import oracle_dominance_check
    fn=tmp_path/'trajectories_detailed.csv';fixture_csv(fn)
    grouped=materialize_trajectories(fn)
    assert oracle_dominance_check(grouped)['status']=='PASS'
    grouped[('2024-01-15',42)]['poa_instant_ideal'][40]['power_w']=100.0
    with pytest.raises(ValueError,match='upper reference'):
        oracle_dominance_check(grouped)
