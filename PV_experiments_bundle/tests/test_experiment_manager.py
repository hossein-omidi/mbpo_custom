"""Unit and isolated fake-CLI integration tests; no PV data or Git access needed."""
from __future__ import annotations
import argparse,ast,json,os,sys,subprocess
from pathlib import Path
import pytest

BUNDLE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BUNDLE))
from PV_experiments import core,train,evaluate,report
from portable_test_fix import updated_test_text


def make_mock_repo(tmp_path):
    root=tmp_path/'repo';root.mkdir()
    for rel in ('MBPO_workspace/workflows/08_monte_carlo_train.py',
                'MBPO_workspace/workflows/09_monte_carlo_evaluate.py',
                'MBPO_workspace/workflows/05_sample_complexity.py',
                'MBPO_workspace/workflows/06_model_validation.py',
                'SAC_workspace/src/training/trainer.py',
                'MBPO_workspace/src/trainer.py'):
        p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# mock existing workflow\n')
    spec=root/core.DEFAULT_UNCERTAINTY;spec.parent.mkdir(parents=True,exist_ok=True)
    spec.write_text(json.dumps({'enabled':True,'dni_cv':.05,'seed':2026}))
    return root


def training_args(**overrides):
    args={'name':'mc_test','seeds':'0:1','epochs':3,'uncertainty_spec':None,'device':None,
          'model_epochs':None,'model_max_train_samples':None,'resume':False,
          'dry_run':False,'skip_verification':False}
    args.update(overrides);return argparse.Namespace(**args)


def evaluate_args(**overrides):
    args={'name':'mc_test','mode':'smoke','seeds':None,'draws':None,'realization_id':0,
          'confirm_test':False,'dry_run':False};args.update(overrides)
    return argparse.Namespace(**args)


def report_args(**overrides):
    args={'name':'mc_test','kind':'learning','stage':'validation','seeds':None,'dry_run':False}
    args.update(overrides);return argparse.Namespace(**args)


def make_completed_runs(root,cfg):
    split={'train_ids':['2018-01-01'],'validation_ids':['2018-01-02'],'test_ids':['2018-01-03'],
           'protocol_sha256':'split-sha'}
    for seed in cfg['seeds']:
        for alg in ('sac','mbpo'):
            d=core.run_path(cfg,alg,seed,root)
            d.mkdir(parents=True)
            metadata={'run_schema':2,'algorithm':alg,'seed':seed,
                'training_status':'completed','final_epoch':cfg['epochs'],
                'scenario_protocol_sha256':'split-sha','resolved_mdp_sha256':'mdp',
                'shared_sac_config_sha256':'common-sac','validation_scenario_ids':['2018-01-02'],
                'test_scenario_ids':['2018-01-03'],'horizon':195,'state_dim':6,'action_dim':2,
                'power_model':'pvwatts_sapm','movement_penalty':1.0,
                'movement_cost_mode':'geometry','energy_objective':'DC_kWh_less_configured_movement_cost'}
            core.atomic_json(d/'run_metadata.json',metadata)
            core.atomic_json(d/'scenario_split.json',split)
            for file in ('latest.pt','episodes.csv','config.json'):
                (d/file).write_text('mock')
            ev=d/'periodic_evaluation';ev.mkdir(parents=True)
            for filename in ('evaluations.csv','evaluation_days.csv','evaluation_references.csv'):
                (ev/filename).write_text('mock')
    return split


def test_seed_parsing_and_unsafe_names():
    assert core.parse_seeds('0:4')==[0,1,2,3,4]
    assert core.parse_seeds('4,0,2')==[0,2,4]
    for seeds in ('2:1','-1','1,1','1:500','10::12'):
        with pytest.raises(ValueError):core.parse_seeds(seeds)
    for invalid in ('../', '../secret','foo/bar','.hidden',''):
        with pytest.raises(ValueError):core.validate_name(invalid)


def test_experiment_frozen_settings_and_spec_mutation_rejected(tmp_path):
    root=make_mock_repo(tmp_path)
    cfg=core.initialize_experiment('mc_test',seeds=[0,1],epochs=3,root=root)
    assert (core.experiment_dir(cfg,root)/'uncertainty_spec.json').is_file()
    assert core.load_experiment('mc_test',root)==cfg
    assert core.initialize_experiment('mc_test',root=root)['epochs']==3
    with pytest.raises(ValueError,match='Frozen experiment epochs'):
        core.initialize_experiment('mc_test',epochs=2,root=root)
    with pytest.raises(ValueError,match='frozen experiment seed'):
        core.initialize_experiment('mc_test',seeds=[3],root=root)
    with pytest.raises(ValueError,match='Frozen uncertainty'):
        frozen=core.uncertainty_path(cfg,root)
        frozen.write_text(json.dumps({'enabled':True,'dni_cv':.6,'seed':2026}))
        core.load_experiment('mc_test',root)


def test_dry_run_does_not_write(tmp_path):
    root=make_mock_repo(tmp_path)
    cfg=core.initialize_experiment('dry_run',root=root,dry_run=True)
    assert cfg['epochs']==2000 and cfg['seeds']==[0,1,2,3,4]
    assert not core.experiment_dir(cfg,root).exists()


def test_split_leakage_and_checkpoint_checks(tmp_path):
    root=make_mock_repo(tmp_path)
    cfg=core.initialize_experiment('mc_test',seeds=[0,1],epochs=3,root=root)
    make_completed_runs(root,cfg)
    for seed in cfg['seeds']:assert core.verify_pair(cfg,seed,root)['protocol_sha256']=='split-sha'
    dest=core.run_path(cfg,'sac',0,root)/'scenario_split.json'
    split=core.read_json(dest);split['test_ids']=['2018-01-01']
    core.atomic_json(dest,split)
    with pytest.raises(ValueError):core.verify_pair(cfg,0,root)


def test_training_commands_autodiscover_dirs_and_resume_safeguard(tmp_path,monkeypatch):
    root=make_mock_repo(tmp_path)
    args=training_args()
    cmd=[]
    def fake_execute(cfg,kind,command,**kw):
        cmd.append((kind,command))
        if kind=='physical_preflight':
            core.atomic_json(core.experiment_dir(cfg,root)/'preflight'/'environment_audit.json',
                {'status':'passed','fixed_draw_replay':True})
        else:
            alg=command[command.index('--algorithm')+1]
            seed=int(command[command.index('--seed')+1]);d=core.run_path(cfg,alg,seed,root)
            d.mkdir(parents=True,exist_ok=True)
            core.atomic_json(d/'run_metadata.json',{'run_schema':2,'seed':seed,'algorithm':alg,
                'training_status':'completed','final_epoch':cfg['epochs']})
            for f in ('latest.pt','scenario_split.json','episodes.csv','config.json'):
                (d/f).touch()
    monkeypatch.setattr(train,'execute',fake_execute)
    train.run(args,root)
    assert [k for k,_ in cmd]==['physical_preflight','train_sac','train_mbpo','train_sac','train_mbpo']
    assert 'sac_seed0' in cmd[1][1][cmd[1][1].index('--run-dir')+1]
    cmd.clear()
    train.run(args,root)
    assert not cmd


def test_validation_evaluation_command_paths(tmp_path,monkeypatch):
    root=make_mock_repo(tmp_path)
    cfg=core.initialize_experiment('mc_test',seeds=[0,1],epochs=3,root=root)
    make_completed_runs(root,cfg)
    got=[]
    def fake_exec(cfg,kind,command,**kwargs):
        got.append((kind,command))
    monkeypatch.setattr(evaluate,'execute',fake_exec)
    # Isolate CLI routing; report-content verification is a separate function.
    monkeypatch.setattr(evaluate,'existing_report_is_valid',lambda *a,**kw:True)
    evaluate.run(evaluate_args(),root)
    assert len(got)==2
    assert all('--max-days' in cmd and '--draws' in cmd for _,cmd in got)
    assert '--sac-checkpoint' in got[0][1] and '--mbpo-checkpoint' in got[0][1]
    got.clear()
    with pytest.raises(ValueError,match='confirm-test'):
        evaluate.run(evaluate_args(mode='test'),root)
    with pytest.raises(ValueError,match='mode smoke'):
        evaluate.run(evaluate_args(mode='test',confirm_test=True),root)
    evaluate.run(evaluate_args(mode='model',realization_id=3),root)
    assert 'realization3' in got[0][1][got[0][1].index('--outdir')+1]


def test_reporting_autodiscovers_completed_seed_sets(tmp_path,monkeypatch):
    root=make_mock_repo(tmp_path)
    cfg=core.initialize_experiment('mc_test',seeds=[0,1],epochs=3,root=root)
    make_completed_runs(root,cfg)
    got=[]
    monkeypatch.setattr(report,'execute',lambda cfg,kind,command,**kwargs:got.append(command))
    report.run(report_args(dry_run=True),root)
    assert len(got)==1
    cmd=got[0]
    assert '--sac-runs' in cmd and '--mbpo-runs' in cmd
    assert 'sac_seed1' in ' '.join(cmd) and 'mbpo_seed1' in ' '.join(cmd)


def test_portable_replacement_does_not_depend_on_deleted_zip():
    old='''from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_full_guarded_installer_plan_and_apply_on_exact_mock_source(tmp_path,monkeypatch):
    import zipfile
    z=zipfile.ZipFile('/mnt/data/PostProcessing_MC_critical_review_v2_no_backup.zip')
    assert z is not None
'''
    new=updated_test_text(old)
    assert '/mnt/data/PostProcessing_MC_critical_review_v2_no_backup.zip' not in new
    assert 'tool.plan(repo)' in new
    ast.parse(new)
    assert updated_test_text(new)==new


def test_report_seed_dispersion_is_from_seed_level(tmp_path):
    assert report.summary_statistics([1.,3.])==(2.,pytest.approx(2**0.5))
    with pytest.raises(ValueError):report.summary_statistics([1.])


def test_true_subprocess_training_and_report_runners_create_manifest_no_backup(tmp_path):
    """Exercise `python -m` orchestration in fresh mock repository processes."""
    root=make_mock_repo(tmp_path)
    verifier=root/'PostProcessing_MC_patch/tools/verify_mc_environment.py'
    verifier.parent.mkdir(parents=True,exist_ok=True)
    verifier.write_text('''import sys,argparse,json,pathlib
p=argparse.ArgumentParser()
p.add_argument('--uncertainty-spec',required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'status':'passed','fixed_draw_replay':True}))
''')
    trainer=root/'MBPO_workspace/workflows/08_monte_carlo_train.py'
    trainer.write_text('''import argparse,json,pathlib
p=argparse.ArgumentParser()
for key in ('algorithm','uncertainty-spec','run-dir','device','seed','epochs'):p.add_argument('--'+key,required=True)
a=p.parse_args()
d=pathlib.Path(a.run_dir);d.mkdir(parents=True,exist_ok=True)
(d/'latest.pt').write_text('mock')
(d/'episodes.csv').write_text('mock')
(d/'config.json').write_text('{}')
split={'train_ids':['2018-01-01'],'validation_ids':['2018-01-02'],'test_ids':['2018-01-03'],'protocol_sha256':'split-sha'}
(d/'scenario_split.json').write_text(json.dumps(split))
data={'run_schema':2,'algorithm':a.algorithm,'seed':int(a.seed),'training_status':'completed','final_epoch':int(a.epochs),
  'scenario_protocol_sha256':'split-sha','resolved_mdp_sha256':'mdp','shared_sac_config_sha256':'common',
  'validation_scenario_ids':['2018-01-02'],'test_scenario_ids':['2018-01-03'],'horizon':195,'state_dim':6,'action_dim':2,
  'power_model':'pvwatts_sapm','movement_penalty':1.0,'movement_cost_mode':'geometry','energy_objective':'DC_kWh_less_configured_movement_cost'}
(d/'run_metadata.json').write_text(json.dumps(data))
periodic=d/'periodic_evaluation';periodic.mkdir(parents=True)
for fname in ('evaluations.csv','evaluation_days.csv','evaluation_references.csv'):
    (periodic/fname).write_text('mock')
print('Mock trainer finished:',a.algorithm,a.seed)
''')
    report_script=root/'MBPO_workspace/workflows/05_sample_complexity.py'
    report_script.write_text('''import argparse,json,pathlib
p=argparse.ArgumentParser()
p.add_argument('--sac-runs',nargs='+',required=True)
p.add_argument('--mbpo-runs',nargs='+',required=True)
p.add_argument('--outdir',required=True)
a=p.parse_args()
out=pathlib.Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
(out/'report.json').write_text(json.dumps({'algorithm_summary':[]}))
(out/'learning_curve_descriptive_sd.csv').write_text('algorithm,mean,sd\\nsac,1,0.1\\n')
(out/'learning_curve_mean_pm_seed_sd.png').write_bytes(b'png-stub')
print('Mock report generated',len(a.sac_runs),len(a.mbpo_runs))
''')
    train.run(training_args(),root)
    cfg=core.load_experiment('mc_test',root)
    for seed in (0,1):core.verify_pair(cfg,seed,root)
    assert len(list((core.experiment_dir(cfg,root)/'jobs').glob('*.json')))==5
    report.run(report_args(),root)
    assert (core.experiment_dir(cfg,root)/'reports/sample_complexity/report_manifest.json').is_file()
    assert not list(core.experiment_dir(cfg,root).rglob('*.bak'))
    assert not list(core.experiment_dir(cfg,root).rglob('*backup*'))
    # Safe second run: completed jobs and immutable reports are not replaced.
    train.run(training_args(),root)
    report.run(report_args(),root)


def test_installer_changes_only_explicit_legacy_tests_and_adds_manager(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('standalone_installer',BUNDLE/'install.py')
    installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)
    root=make_mock_repo(tmp_path)
    v=root/'PostProcessing_MC_patch/tools/verify_mc_environment.py'
    v.parent.mkdir(parents=True,exist_ok=True);v.write_text('# existing physical verifier\n')
    sc=root/'MBPO_workspace/src/sample_complexity.py'
    sc.parent.mkdir(parents=True,exist_ok=True);sc.write_text('# existing sample complexity\n')
    test=root/'PostProcessing_pipeline_review_patch/tests/test_pipeline_statistics.py'
    test.parent.mkdir(parents=True,exist_ok=True)
    source='''from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_full_guarded_installer_plan_and_apply_on_exact_mock_source(tmp_path,monkeypatch):
    import zipfile
    z=zipfile.ZipFile('/mnt/data/PostProcessing_MC_critical_review_v2_no_backup.zip')
    assert z is not None
'''
    test.write_text(source)
    changes=installer.plan(root)
    assert 'PostProcessing_pipeline_review_patch/tests/test_pipeline_statistics.py' in changes
    assert 'PV_experiments/train.py' in changes and 'PV_experiments/report.py' in changes
    installer.install(root,changes)
    assert installer.plan(root)=={}
    text=test.read_text()
    assert '/mnt/data/PostProcessing_MC_critical_review_v2_no_backup.zip' not in text
    assert not list(root.rglob('*.bak'))


def test_across_seed_mc_report_mean_sd_and_no_unintended_overwrite(tmp_path):
    pytest.importorskip('matplotlib')
    root=make_mock_repo(tmp_path)
    cfg=core.initialize_experiment('mc_test',seeds=[0,1],epochs=3,root=root)
    make_completed_runs(root,cfg)
    for seed in (0,1):
        directory=evaluate.evaluation_output(cfg,'test',seed,root)
        directory.mkdir(parents=True)
        methods={name:{'mean_net_kwh':10.+seed+(name=='mbpo')} for name in
                 ('sac','mbpo','fixed_no_motion','sun_slew_limited')}
        differences={key:{'mean_net_kwh':1.+seed} for key in
           ('mbpo_minus_sac','mbpo_minus_fixed_no_motion','sac_minus_fixed_no_motion',
            'mbpo_minus_sun_slew_limited','sac_minus_sun_slew_limited')}
        core.atomic_json(directory/'mc_summary.json',{
          'scenario_protocol_sha256':'split-sha','uncertainty_spec_sha256':'uncertain',
          'n_selected_days':2,'n_available_days':2,'draws_per_day':100,
          'split_role':'test','partial_partition_evaluation':False,
          'methods':methods,'paired_net_energy_differences':differences})
        (directory/'all_day_draw_method.csv').write_text('scenario_id,draw,method,net_kwh\n')
        with (directory/'mc_pooled_power_mean_sd_by_utc_step.csv').open('w') as f:
            f.write('method,step,mean_power_w,sd_total_descriptive_w\n')
            for name in methods:
                for t in range(1,196):f.write(f'{name},{t},{100.+seed+0.1*t},3.0\n')
    report.report_mc(cfg,[0,1],'test',root)
    out=core.experiment_dir(cfg,root)/'reports/monte_carlo_test'
    with (out/'net_energy_across_seeds.csv').open() as f:
        import csv
        rows=list(csv.DictReader(f))
    sac=next(r for r in rows if r['method']=='sac')
    assert float(sac['mean_over_training_seed_means_net_kwh'])==pytest.approx(10.5)
    assert float(sac['sd_over_training_seed_means_net_kwh'])==pytest.approx(0.5**.5)
    assert (out/'power_mean_pm_training_seed_sd.png').stat().st_size>500
    report.report_mc(cfg,[0,1],'test',root)
    with pytest.raises(FileExistsError,match='Do not overwrite'):
        report.report_mc(cfg,[0], 'test',root)
