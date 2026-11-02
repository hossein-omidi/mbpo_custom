"""Regression tests for 195-step SAC/MBPO paired full-day statistics and sampling."""
import copy
from datetime import datetime, timedelta, timezone

import pytest

from SAC_workspace.src.evaluation.scenario_sampling import (
    build_selection, validate_selection, frame_fingerprint,
)
from SAC_workspace.src.evaluation.stratified_statistics import (
    make_episode_features, analyse_features, _time_grid_note,
)


def manifest():
    days=['2018-01-02', '2018-01-03', '2018-02-04']
    return {
        'episode': {'start_time':'08:45','periods':196,'freq':'5min','tz':'UTC'},
        'scenarios':[{'scenario_id':d,'year':int(d[:4]),'month':int(d[5:7]),
                      'source_year':int(d[:4])} for d in days],
    }


def path(sid,method='mbpo', n=195):
    t0=datetime.fromisoformat(sid+'T08:45:00').replace(tzinfo=timezone.utc)
    info=[]
    for i in range(1,n+1):
        stamp=t0+timedelta(minutes=5*i)
        info.append({'timestamp_utc_iso':stamp.isoformat(),
                     'solar_zenith_deg':40.0,'solar_azimuth_deg':180.0,
                     'dni_wm2':400.0,'ghi_wm2':500.0,'dhi_wm2':100.0,
                     'weather_condition':'clear'})
    energy=[0.01]*n
    return {'scenario_id':sid,'seed':42,'method':method,
            'infos':info,'energy':energy,'power':[120.0]*n,
            'tilt':[30.0]*n,'azimuth':[180.0]*n,
            'delta_tilt_deg':[0.0]*n,'delta_azimuth_deg':[0.0]*n,
            'net_energy_kwh':sum(energy), 'gross_energy_kwh':sum(energy),
            'movement_cost_kwh':0.0}


def test_plan_describes_full_episode_and_exclusion_truthfully():
    m=manifest()
    plan=build_selection(m,days_per_month=4,selection_seed=42,excluded_ids=('2018-01-02',))
    assert plan['episode']['periods']==196
    assert '195' in plan['episode_note']
    assert 'excluded' in plan['sampling_note']
    ids=validate_selection(plan,m)
    assert '2018-01-02' not in ids
    assert plan['population_count']==2
    assert '01:00' in _time_grid_note(plan)


def test_manifest_clock_changes_invalidate_saved_plan():
    m=manifest();plan=build_selection(m,days_per_month=3)
    changed=copy.deepcopy(m)
    changed['episode']['periods']=118
    assert frame_fingerprint(changed)!=frame_fingerprint(m)
    with pytest.raises(ValueError,match='episode grid'):
        validate_selection(plan,changed)


def test_complete_195_step_midnight_trajectories_generate_stats():
    m=manifest();plan=build_selection(m,days_per_month=4)
    paths=[path(sid,meth) for sid in plan['selected_scenario_ids']
           for meth in ('sac','mbpo','fixed_no_motion')]
    rows=make_episode_features(paths,plan)
    assert all(int(r['n_steps'])==195 for r in rows)
    stats=analyse_features(rows,plan,n_boot=4)
    assert stats['selected_scenarios']==3
    assert '195' in stats['horizon'] and '01:00' in stats['horizon']
    assert stats['observed_training_days_excluded']==0


def test_corrupted_step_count_rejected():
    plan=build_selection(manifest(),days_per_month=3)
    with pytest.raises(ValueError,match='expected 195'):
        make_episode_features([path('2018-01-02',n=117)],plan)


def test_utc_midnight_misalignment_rejected():
    plan=build_selection(manifest(),days_per_month=3)
    p=path('2018-01-02')
    p['infos'][-1]['timestamp_utc_iso']='2018-01-02T01:00:00+00:00'
    with pytest.raises(ValueError,match='UTC episode grid'):
        make_episode_features([p],plan)


def test_invalid_manifest_start_time_rejected():
    m=manifest();m['episode']['start_time']='12:00'
    with pytest.raises(ValueError,match='full-day evaluation grid'):
        build_selection(m)


def test_offline_rebuild_accepts_saved_195_step_physical_trajectories(tmp_path):
    import csv
    from scripts.rebuild_full_day_statistics import observed_paths
    selected={'2018-01-02'}
    columns=['scenario_id','seed','method','step','utc_iso','power_w',
             'energy_kwh','movement_cost_kwh','tilt_deg','azimuth_deg',
             'actual_delta_tilt_deg','actual_delta_azimuth_deg',
             'solar_zenith_deg','solar_azimuth_deg','dni_wm2',
             'dhi_wm2','ghi_wm2','weather_condition']
    csvfile=tmp_path/'trajectories_detailed.csv'
    with csvfile.open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=columns);w.writeheader()
        dt=datetime(2018,1,2,8,45,tzinfo=timezone.utc)
        for method in ('mbpo','sac','fixed_no_motion','poa_instant_ideal'):
            for step in range(1,196):
                w.writerow({
                    'scenario_id':'2018-01-02','seed':42,'method':method,
                    'step':step,'utc_iso':(dt+timedelta(minutes=step*5)).isoformat(),
                    'power_w':120,'energy_kwh':0.01,'movement_cost_kwh':0,
                    'tilt_deg':30,'azimuth_deg':180,
                    'actual_delta_tilt_deg':'' if method=='poa_instant_ideal' else 0,
                    'actual_delta_azimuth_deg':'' if method=='poa_instant_ideal' else 0,
                    'solar_zenith_deg':40,'solar_azimuth_deg':180,
                    'dni_wm2':500,'dhi_wm2':100,'ghi_wm2':600,
                    'weather_condition':'clear'})
    paths=observed_paths(csvfile,selected)
    assert len(paths)==4
    assert all(len(p['infos'])==195 for p in paths)
    assert sum(p['net_energy_kwh'] is None for p in paths)==1
    plan=build_selection({'episode':manifest()['episode'],
                          'scenarios':[manifest()['scenarios'][0]]},days_per_month=1)
    rows=make_episode_features(paths,plan)
    assert {r['method'] for r in rows}=={'mbpo','sac','fixed_no_motion','poa_instant_ideal'}
    assert all(r['n_steps']==195 for r in rows)
    from SAC_workspace.src.evaluation.stratified_statistics import render_statistical_report
    report=render_statistical_report(tmp_path/'stratified',plan,paths=paths,n_boot=4)
    assert report['selected_scenarios']==1
    assert len(report['figures'])==7


def test_run_directory_preflight_requires_full_mdp_and_complete_history(tmp_path):
    import json
    import csv
    from scripts.preflight_full_day_evaluation import load_run, EXPECTED
    run=tmp_path/'sac_run'
    run.mkdir()
    import torch
    (run/'config.json').write_text('{}')
    mdp = EXPECTED | {'power_model':'pvwatts_sapm'}
    (run/'resolved_mdp.json').write_text(json.dumps(mdp))
    with (run/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=('scenario_id','epoch','env_steps'))
        w.writeheader()
        w.writerow({'scenario_id':'2018-01-02','epoch':1,'env_steps':195})
        w.writerow({'scenario_id':'2018-01-03','epoch':2,'env_steps':390})
    torch.save({'resolved_mdp':mdp,'seed':0,'epoch':2,'env_steps':390},run/'latest.pt')
    mdp, seen, n, seed=load_run(run,'SAC')
    assert n==2 and seen=={'2018-01-02','2018-01-03'} and seed==0
    assert mdp['horizon']==195
    with (run/'episodes.csv').open('a',newline='') as f:
        w=csv.writer(f); w.writerow(['2018-01-04', 3, 400])
    with pytest.raises(ValueError,match='expected 585'):
        load_run(run,'SAC')


def test_optional_mbpo_plot_clock_does_not_wrap_at_midnight():
    import ast
    from pathlib import Path
    import numpy as np
    source=Path(__file__).resolve().parents[1]/'MBPO_workspace/src/plots.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_utc_hours')
    module=ast.Module(body=[fn],type_ignores=[])
    env={'np':np}
    exec(compile(ast.fix_missing_locations(module),str(source),'exec'),env)
    values=[{'timestamp_utc_iso':s} for s in (
        '2018-01-02T23:55:00+00:00','2018-01-03T00:00:00+00:00',
        '2018-01-03T01:00:00+00:00')]
    hours=env['_utc_hours']({'infos':values})
    assert np.all(np.diff(hours)>0)
    assert hours.tolist()==pytest.approx([23+55/60,24.0,25.0])


def test_sac_plot_time_axis_is_unwrapped_across_midnight():
    from SAC_workspace.src.evaluation.time_axis import unwrapped_utc_hours

    infos = [
        {'timestamp_utc_iso': '2026-06-21T23:50:00+00:00'},
        {'timestamp_utc_iso': '2026-06-21T23:55:00+00:00'},
        {'timestamp_utc_iso': '2026-06-22T00:00:00+00:00'},
        {'timestamp_utc_iso': '2026-06-22T00:05:00+00:00'},
    ]
    hours = unwrapped_utc_hours(infos)
    assert hours[0] == pytest.approx(23 + 50/60)
    assert hours[2] == pytest.approx(24.0)
    assert hours[3] > 24.0
    assert (hours[1:] - hours[:-1]) == pytest.approx([5/60]*3)


def test_sac_plot_rejects_non_utc_or_non_monotone_clock():
    from SAC_workspace.src.evaluation.time_axis import unwrapped_utc_hours
    with pytest.raises(ValueError, match='UTC'):
        unwrapped_utc_hours([{'timestamp_utc_iso': '2026-06-21T08:50:00-04:00'}])
    with pytest.raises(ValueError, match='strictly increasing'):
        unwrapped_utc_hours([
            {'timestamp_utc_iso': '2026-06-21T08:55:00+00:00'},
            {'timestamp_utc_iso': '2026-06-21T08:50:00+00:00'},
        ])


def test_preflight_rejects_checkpoint_from_another_run(tmp_path):
    import json, csv, torch
    from scripts.preflight_full_day_evaluation import load_run, EXPECTED
    run=tmp_path/'run'; run.mkdir()
    mdp=EXPECTED | {'power_model':'pvwatts_sapm'}
    (run/'config.json').write_text('{}')
    (run/'resolved_mdp.json').write_text(json.dumps(mdp))
    with (run/'episodes.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=('scenario_id','epoch','env_steps')); w.writeheader()
        w.writerow({'scenario_id':'2018-01-02','epoch':1,'env_steps':195})
    torch.save({'resolved_mdp':mdp,'seed':0,'epoch':1,'env_steps':195},run/'latest.pt')
    torch.save({'resolved_mdp':mdp | {'start_time':'12:00'},'seed':0,'epoch':1,'env_steps':195},run/'latest.pt')
    with pytest.raises(ValueError,match='resolved_mdp differs'):
        load_run(run,'SAC')
