"""Portable, narrow corrections to synthetic tests from the prior patch.

No project source code or backup file is involved. Every modification is
restricted to the exact old test function and independently syntax-checked.
"""
from __future__ import annotations
import ast
from textwrap import dedent

PLOT_TEST='test_full_guarded_installer_plan_and_apply_on_exact_mock_source'
EXPERIMENT_TEST='test_validation_evaluation_command_paths'
MC_TEST='test_across_seed_mc_report_mean_sd_and_no_unintended_overwrite'

MOCK_OLD = "def plot(ax):\\n    ax.set_title('Learning curves (weather distributions may differ by sampled episode)')\\n"
MOCK_FIXED = "def plot(ax):\\n        ax.set_title('Learning curves (weather distributions may differ by sampled episode)')\\n"

NEW_MC_TEST=dedent('''
def test_across_seed_mc_report_mean_sd_and_no_unintended_overwrite(tmp_path):
    pytest.importorskip('matplotlib')
    import csv
    root=make_mock_repo(tmp_path)
    cfg=core.initialize_experiment('mc_test',seeds=[0,1],epochs=3,root=root)
    make_completed_runs(root,cfg)
    methods=('sac','mbpo','fixed_no_motion','sun_slew_limited')
    days=('2018-01-03',)
    for seed in (0,1):
        directory=evaluate.evaluation_output(cfg,'test',seed,root)
        directory.mkdir(parents=True)
        returns={'sac':10.+seed,'mbpo':11.+seed,
                 'fixed_no_motion':8.,'sun_slew_limited':9.}
        summaries={name:{'mean_net_kwh':value} for name,value in returns.items()}
        pairs={name:{'mean_net_kwh':returns[name.split('_minus_')[0]]-
                     returns[name.split('_minus_')[1]]} for name in
               ('mbpo_minus_sac','mbpo_minus_fixed_no_motion','sac_minus_fixed_no_motion',
                'mbpo_minus_sun_slew_limited','sac_minus_sun_slew_limited')}
        core.atomic_json(directory/'mc_summary.json',{
          'scenario_protocol_sha256':'split-sha','uncertainty_spec_sha256':'uncertain',
          'n_selected_days':1,'n_available_days':1,'draws_per_day':100,
          'split_role':'test','partial_partition_evaluation':False,
          'methods':summaries,'paired_net_energy_differences':pairs})
        with (directory/'all_day_draw_method.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['scenario_id','draw','method','net_kwh','profile_sha256'])
            w.writeheader()
            for sid in days:
                for draw in range(100):
                    for method in methods:
                        w.writerow({'scenario_id':sid,'draw':draw,'method':method,
                            'net_kwh':returns[method],
                            'profile_sha256':f'same-weather-{sid}-{draw}'})
        with (directory/'mc_pooled_power_mean_sd_by_utc_step.csv').open('w') as f:
            f.write('method,step,mean_power_w,sd_total_descriptive_w\\n')
            for name in methods:
                for t in range(1,196):
                    f.write(f'{name},{t},{100.+seed+0.1*t},3.0\\n')
    report.report_mc(cfg,[0,1],'test',root)
    out=core.experiment_dir(cfg,root)/'reports/monte_carlo_test'
    with (out/'net_energy_across_seeds.csv').open() as f:
        rows=list(csv.DictReader(f))
    sac=next(r for r in rows if r['method']=='sac')
    assert float(sac['mean_over_training_seed_means_net_kwh'])==pytest.approx(10.5)
    assert float(sac['sd_over_training_seed_means_net_kwh'])==pytest.approx(0.5**.5)
    assert (out/'power_mean_pm_training_seed_sd.png').stat().st_size>500
    report.report_mc(cfg,[0,1],'test',root)
    with pytest.raises(FileExistsError,match='Do not overwrite'):
        report.report_mc(cfg,[0], 'test',root)
''')

def _replace_function(original:str,name:str,transform):
    tree=ast.parse(original)
    matches=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))
             and x.name==name]
    if len(matches)!=1:raise ValueError(f'Expected exactly one test {name}')
    node=matches[0]
    lines=original.splitlines(keepends=True)
    old=''.join(lines[node.lineno-1:node.end_lineno])
    new=transform(old)
    if old==new:return original
    result=''.join(lines[:node.lineno-1])+new.rstrip()+"\n"+''.join(lines[node.end_lineno:])
    ast.parse(result)
    return result


def fix_pipeline_test(original):
    def transform(old):
        if MOCK_FIXED in old:return old
        if old.count(MOCK_OLD)!=1:
            raise ValueError('Unexpected synthetic plots.py fixture: refusing speculative edit')
        return old.replace(MOCK_OLD,MOCK_FIXED,1)
    return _replace_function(original,PLOT_TEST,transform)


def fix_manager_tests(original):
    def change_validation(old):
        old_line="evaluate.run(evaluate_args(mode='model',realization_id=3),root)"
        new_line="evaluate.run(evaluate_args(mode='model',realization_id=3,dry_run=True),root)"
        if new_line in old:return old
        if old.count(old_line)!=1:raise ValueError('Unknown evaluation CLI mock test version')
        return old.replace(old_line,new_line,1)
    original=_replace_function(original,EXPERIMENT_TEST,change_validation)
    def change_mc(old):
        if "'profile_sha256':f'same-weather-{sid}-{draw}'" in old:
            return old
        if "(directory/'all_day_draw_method.csv').write_text('scenario_id,draw,method,net_kwh" not in old:
            raise ValueError('Unknown previous mock Monte Carlo fixture; refusing replacement')
        return NEW_MC_TEST
    return _replace_function(original,MC_TEST,change_mc)
