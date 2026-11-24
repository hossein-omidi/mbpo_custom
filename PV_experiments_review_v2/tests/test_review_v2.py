"""Offline regression for high-priority experiment-manager validity checks."""
from __future__ import annotations
import importlib.util,sys,json,csv,ast
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
ROOT_MODULE=importlib.util.spec_from_file_location('pv_review_fixer',ROOT/'repair_tests.py')
fix=importlib.util.module_from_spec(ROOT_MODULE);ROOT_MODULE.loader.exec_module(fix)
INSTALLER_SPEC=importlib.util.spec_from_file_location('pv_review_installer',ROOT/'install.py')
inst=importlib.util.module_from_spec(INSTALLER_SPEC)
sys.path.insert(0,str(ROOT))
INSTALLER_SPEC.loader.exec_module(inst)

# Import actual replacement modules by file; their relative import needs the
# installed project's PV_experiments package, which is always present after
# installation. This is an integration test, not a duplicate model trainer.

def test_mock_plots_indent_fix_is_scoped_and_repeatable():
    from pathlib import Path
    original='''from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_full_guarded_installer_plan_and_apply_on_exact_mock_source(tmp_path,monkeypatch):
    originals={
      'MBPO_workspace/src/plots.py': "def plot(ax):\\n    ax.set_title('Learning curves (weather distributions may differ by sampled episode)')\\n",
    }
'''
    revised=fix.fix_pipeline_test(original)
    assert "def plot(ax):\\n        ax.set_title" in revised
    assert fix.fix_pipeline_test(revised)==revised
    assert "def plot(ax):\\n    ax.set_title" not in revised
    ast.parse(revised)
    broken=original.replace("def plot(ax):\\n    ax.set_title", "def plot(ax):\\n      ax.set_title")
    with pytest.raises(ValueError,match='Unexpected synthetic'):
        fix.fix_pipeline_test(broken)


def test_guarded_manager_patch_idempotent_and_unknown_changes_rejected(tmp_path):
    from shutil import copyfile
    previous=ROOT.parent/'PV_experiments_bundle'
    assert previous.is_dir(), 'Expected the previously extracted PV_experiments_bundle directory'
    repo=tmp_path/'repo'
    for module in ('evaluate','report'):
        dst=repo/'PV_experiments'/f'{module}.py';dst.parent.mkdir(parents=True,exist_ok=True)
        copyfile(previous/'PV_experiments'/f'{module}.py',dst)
    dst=repo/inst.TEST_MANAGER;dst.parent.mkdir(parents=True,exist_ok=True)
    copyfile(previous/'tests/test_experiment_manager.py',dst)
    mod=importlib.util.spec_from_file_location('previous_portable',previous/'portable_test_fix.py')
    prev=importlib.util.module_from_spec(mod);mod.loader.exec_module(prev)
    dst=repo/inst.TEST_PIP;dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text('from pathlib import Path\nROOT=Path(__file__).resolve().parents[1]\n\n'+prev.REPLACEMENT)
    changes=inst.plan(repo)
    assert {'PV_experiments/evaluate.py','PV_experiments/report.py',inst.TEST_PIP}.issubset(changes)
    # The previous bundle's own test may already have been corrected by V2.
    assert len(changes) in (3,4)
    for relative,(_,content) in changes.items():
        ast.parse(content)
        target=repo/relative
        target.write_text(content)
    assert inst.plan(repo)=={}
    target=repo/'PV_experiments/evaluate.py';target.write_text(target.read_text()+'\n# unknown change')
    with pytest.raises(ValueError,match='differs from the audited manager'):
        inst.plan(repo)


def test_raw_paired_mc_accepts_complete_grid_and_rejects_profile_disagreement(tmp_path):
    # The production report module is imported from the installed checkout,
    # not from this patch's independent directory.
    repo=ROOT.parent
    assert (repo/'PV_experiments/report.py').is_file(), 'The experiment manager has not been installed'
    sys.path.insert(0,str(repo))
    importlib.import_module('PV_experiments')
    s=importlib.util.spec_from_file_location('PV_experiments._review_report',repo/'PV_experiments/report.py')
    module=importlib.util.module_from_spec(s);s.loader.exec_module(module)
    _validate_raw_mc_file=module._validate_raw_mc_file
    path=tmp_path/'all_day_draw_method.csv'
    methods=('sac','mbpo','fixed_no_motion','sun_slew_limited')
    means={'sac':10.,'mbpo':11.,'fixed_no_motion':8.,'sun_slew_limited':9.}
    pairs={name:{'mean_net_kwh':means[name.split('_minus_')[0]]-means[name.split('_minus_')[1]]}
           for name in ('mbpo_minus_sac','mbpo_minus_fixed_no_motion',
                        'sac_minus_fixed_no_motion','mbpo_minus_sun_slew_limited',
                        'sac_minus_sun_slew_limited')}
    summary={'draws_per_day':2,'methods':{k:{'mean_net_kwh':v} for k,v in means.items()},
             'paired_net_energy_differences':pairs}
    def write(changed=False):
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=('scenario_id','draw','method','net_kwh','profile_sha256'))
            writer.writeheader()
            for rid in range(2):
                for method in methods:
                    writer.writerow({'scenario_id':'2024-01-02','draw':rid,'method':method,
                        'net_kwh':means[method],
                        'profile_sha256':f'profile-{rid}' if not changed or method!='mbpo' else 'wrong'})
    write();fingerprints,baselines=_validate_raw_mc_file(path,summary,{'2024-01-02'})
    assert len(fingerprints)==2 and len(baselines)==4
    write(changed=True)
    with pytest.raises(ValueError,match='weather realization mismatch'):
        _validate_raw_mc_file(path,summary,{'2024-01-02'})
    write();summary['methods']['sac']['mean_net_kwh']=77.
    with pytest.raises(ValueError,match='summary mean disagrees'):
        _validate_raw_mc_file(path,summary,{'2024-01-02'})


def test_checkpoint_identity_must_not_be_reused(tmp_path):
    repo=ROOT.parent
    assert (repo/'PV_experiments/evaluate.py').is_file(), 'The experiment manager has not been installed'
    sys.path.insert(0,str(repo))
    importlib.import_module('PV_experiments')
    s=importlib.util.spec_from_file_location('PV_experiments._review_evaluate',repo/'PV_experiments/evaluate.py')
    module=importlib.util.module_from_spec(s);s.loader.exec_module(module)
    _require_matching_provenance=module._require_matching_provenance
    result=tmp_path/'eval';result.mkdir()
    (result/'manager_provenance.json').write_text(json.dumps({'seed':0,'mbpo_checkpoint_sha256':'AAA'}))
    _require_matching_provenance(result,{'seed':0,'mbpo_checkpoint_sha256':'AAA'})
    with pytest.raises(ValueError,match='differ from recorded provenance'):
        _require_matching_provenance(result,{'seed':0,'mbpo_checkpoint_sha256':'BBB'})
