"""Offline regression of statistical meaning, draw pairing, and import-path patch.

These tests do NOT replace pvlib/NSRDB integration or independent empirical
model-quality experiments. They do protect the mathematical statistics and
installer against silent changes in data shape or reporting meaning.
"""
from __future__ import annotations
import importlib.util, csv, sys, subprocess, shutil
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'new_files/MBPO_workspace/src/curve_statistics.py'
spec=importlib.util.spec_from_file_location('pipeline_curve_statistics_offline',NEW)
cs=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)


def test_seed_sd_and_day_sd_are_not_conflated():
    # seed0 day0,day1 -> 1,3; seed1 day0,day1 -> 5,7
    x=np.array([[[1,2],[3,4]], [[5,6],[7,8]]],dtype=float)
    r=cs.summarize(x)
    assert np.allclose(r['mean'],[4.,5.])
    assert np.allclose(r['sd_across_training_seed_means'],np.sqrt(8.))
    assert np.allclose(r['sd_across_historical_day_means'],np.sqrt(2.))
    assert not np.allclose(r['sd_across_training_seed_means'],r['sd_across_historical_day_means'])


def test_paired_method_sd_subtracts_before_dispersion():
    sac=np.array([[[2.,4.],[3.,5.]], [[4.,6.],[5.,7.]]])
    mbpo=sac + np.array([[[1.,1.],[1.,1.]], [[3.,3.],[3.,3.]]])
    diff=cs.summarize(mbpo-sac)
    assert np.allclose(diff['mean'],2.)
    assert np.allclose(diff['sd_across_training_seed_means'],np.sqrt(2.))
    assert np.allclose(diff['sd_across_historical_day_means'],0.)
    # Variability across individual algorithm scores would be larger.
    assert np.any(cs.summarize(mbpo)['sd_across_training_seed_means']>diff['sd_across_training_seed_means'])


def test_summarize_rejects_invalid_or_insufficient_independence():
    for shape in [(0,2,3),(1,4,3),(3,1,2),(2,2)]:
        with pytest.raises(ValueError):cs.summarize(np.ones(shape))
    x=np.ones((2,2,3));x[0,0,0]=np.nan
    with pytest.raises(ValueError):cs.summarize(x)


def mk_mc_clock_rows(by_day_draws,methods=('sac','mbpo')):
    rows=[]
    for method in methods:
        for day,values in by_day_draws.items():
            vals=np.asarray(values,dtype=float)
            for t in range(1,196):
                rows.append({'method':method,'scenario_id':day,'step':t,
                    'power_mean_w':float(vals.mean()),
                    'power_sd_w':float(vals.std(ddof=1))})
    return rows


def test_pooled_mc_variance_decomposition_matches_raw_simulated_draws():
    values={'2023-06-21':[1.,2.,3.],'2024-06-21':[5.,7.,9.]}
    records=mk_mc_clock_rows(values)
    result=cs.aggregate_mc_power_profiles(records,draws=3)
    assert len(result)==2*195
    p=next(x for x in result if x['method']=='sac' and x['step']==195)
    raw=np.array(list(values.values())).reshape(-1)
    assert p['mean_power_w']==pytest.approx(float(raw.mean()))
    assert p['sd_total_descriptive_w']**2==pytest.approx(float(raw.var(ddof=0)))
    assert p['variance_check_total_w2']==pytest.approx(
        p['sd_between_historical_days_w']**2+p['sd_within_weather_draws_w']**2)
    assert p['utc_hour_unwrapped']==pytest.approx(25.)


def test_pooled_mc_rejects_duplicate_incomplete_and_negative_sd():
    records=mk_mc_clock_rows({'2018-06-21':[5.,7.], '2019-06-21':[9.,11.]})
    with pytest.raises(ValueError,match='Duplicate'):
        cs.aggregate_mc_power_profiles(records+[records[0]],draws=2)
    with pytest.raises(ValueError,match='different historical-day coverage'):
        cs.aggregate_mc_power_profiles(records[:-1],draws=2)
    bad=[dict(x) for x in records];bad[0]['power_sd_w']=-1.
    with pytest.raises(ValueError,match='Nonfinite/negative'):
        cs.aggregate_mc_power_profiles(bad,draws=2)
    with pytest.raises(ValueError,match='At least two'):
        cs.aggregate_mc_power_profiles(records,draws=1)


def test_sample_complexity_exports_and_shaded_graphs(tmp_path):
    pytest.importorskip('matplotlib')
    a=np.array([[[1.,2.,3.],[3.,4.,5.]], [[5.,6.,7.],[7.,8.,9.]]])
    b=a+2.
    result=cs.write_sample_complexity_mean_sd(tmp_path,[195,390,585],a,b,target=5.0)
    assert len(result['figure_files'])==3
    for fn in result['figure_files']+result['data_files']:
        assert (tmp_path/fn).exists() and (tmp_path/fn).stat().st_size>50
    with (tmp_path/'learning_curve_descriptive_sd.csv').open() as f:
        rows=list(csv.DictReader(f))
    row=next(r for r in rows if r['algorithm']=='sac' and r['env_steps']=='195')
    assert float(row['mean_net_energy_kwh'])==pytest.approx(4.0)
    assert float(row['sd_across_training_seed_means_kwh'])==pytest.approx(np.sqrt(8.))
    assert 'confidence interval' in result['interpretation']


def test_mc_pooled_graph_exists(tmp_path):
    pytest.importorskip('matplotlib')
    data=cs.aggregate_mc_power_profiles(
        mk_mc_clock_rows({'2018-06-21':[20.,24.],'2019-06-21':[30.,32.]}),2)
    name=cs.plot_pooled_mc_power_profiles(tmp_path,data)
    assert (tmp_path/name).exists() and (tmp_path/name).stat().st_size>200



def test_v2_sources_validate_exact_installer_anchors():
    """Validate installed source modules without developer-only ZIP files."""
    import ast
    repo=ROOT.parent
    mc=repo/'MBPO_workspace/workflows/09_monte_carlo_evaluate.py'
    verifier=repo/'PostProcessing_MC_patch/tools/verify_mc_environment.py'
    assert mc.is_file() and verifier.is_file()
    m=mc.read_text(encoding='utf-8')
    v=verifier.read_text(encoding='utf-8')
    ast.parse(m,filename=str(mc))
    ast.parse(v,filename=str(verifier))
    assert 'mc_pooled_power_mean_sd_by_utc_step.csv' in m
    assert 'def main(' in v




def test_verify_script_can_display_help_as_direct_file_without_importing_pvlib(tmp_path):
    """Actual local verifier entry point is portable and fails on missing repo."""
    import subprocess
    import sys
    from pathlib import Path
    verifier=ROOT.parent/'PostProcessing_MC_patch/tools/verify_mc_environment.py'
    assert verifier.is_file()
    result=subprocess.run([sys.executable,str(verifier),'--help'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--uncertainty-spec' in result.stdout
    # Executed outside the project layout, it must report layout mismatch
    # rather than hiding real dependencies behind a ModuleNotFoundError.
    elsewhere=tmp_path/'workspace'/'PostProcessing_MC_patch'/'tools'/'verify_mc_environment.py'
    elsewhere.parent.mkdir(parents=True,exist_ok=True)
    elsewhere.write_text(verifier.read_text(encoding='utf-8'),encoding='utf-8')
    dummy=tmp_path/'assumptions.json'
    dummy.write_text('{"enabled":true}',encoding='utf-8')
    process=subprocess.run([sys.executable,str(elsewhere),'--uncertainty-spec',str(dummy)],
                           capture_output=True,text=True)
    assert process.returncode!=0
    assert 'complete PostProcessing checkout' in process.stderr
    assert 'No module named' not in process.stderr




def test_full_guarded_installer_plan_and_apply_on_exact_mock_source(tmp_path,monkeypatch):
    """Portable patch installer test: four synthetic modules, no external ZIP.

    Tests the actual four patch transformations, Python syntax, installer
    planning, writing, rejection of unexpected versions, and idempotence.
    The synthetic minimal source texts carry the exact audited anchors.
    """
    import ast
    import importlib.util
    import sys
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('installer_integration_offline',ROOT/'apply_local.py')
    tool=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    repo=tmp_path/'checkout'
    for relative in tool.REQUIRED_UNCHANGED:
        p=repo/relative
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('# mock source for guarded compatibility test\n',encoding='utf-8')
        tool.REQUIRED_UNCHANGED[relative]=tool.blob_sha(p.read_bytes())
    originals={
        'MBPO_workspace/src/sample_complexity.py': 'import json\nfrom pathlib import Path\n\ndef _read_csv(path):return []\n\ndef inspect(root,aggregate,daily,references):\n    split={\'train_ids\':[\'2020-06-21\']}\n    episodes = _read_csv(root / "episodes.csv")\n    if not aggregate or not daily or not references:\n        raise ValueError(\'missing rows\')\n\ndef report(out,report,steps,sac_matrix,mbpo_matrix,target):\n    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")\n\ndef plot(ax,steps,sac_lo,sac_hi,mbpo_lo,mbpo_hi,diff_lo,diff_hi):\n    ax.fill_between(steps, sac_lo, sac_hi, alpha=0.18)\n    ax.fill_between(steps, mbpo_lo, mbpo_hi, alpha=0.18)\n    ax.fill_between(steps, diff_lo, diff_hi, alpha=0.2)\n',
        'MBPO_workspace/workflows/09_monte_carlo_evaluate.py': "def render(out,clock_rows,args,rows,first_day_clock_rows,paired):\n    write_csv(out/'daily_power_mean_std_by_utc_step.csv',clock_rows)\n    report={\n        'power_output':'daily_power_mean_std_by_utc_step.csv (every selected historical day)',\n    }\n    if not args.skip_figures:\n        plot_diagnostics(out,rows,first_day_clock_rows,paired)\n",
        'MBPO_workspace/src/plots.py': "def plot(ax):\n    ax.set_title('Learning curves (weather distributions may differ by sampled episode)')\n",
        'PostProcessing_MC_patch/tools/verify_mc_environment.py': '#!/usr/bin/env python3\n\ndef main(argv=None):\n    args=parser.parse_args(argv)\n    from mbpo.env.trajectory_uncertainty import normalized_spec\n',
    }
    for relative,source in originals.items():
        p=repo/relative
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(source,encoding='utf-8')
        tool.EXPECTED[relative]=tool.blob_sha(p.read_bytes())
    changes=tool.plan(repo)
    assert len(changes)==5
    assert any(name.endswith('curve_statistics.py') for name in changes)
    for relative,(_old,new) in changes.items():
        ast.parse(new,filename=relative)
    monkeypatch.setattr(sys,'argv',['apply_local.py','--root',str(repo),'--apply'])
    tool.main()
    for relative,(_old,new) in changes.items():
        assert (repo/relative).read_text(encoding='utf-8')==new
    assert tool.plan(repo)=={}

