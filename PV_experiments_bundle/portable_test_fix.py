"""Fix previous local test's developer-container-only absolute ZIP dependency.

This makes no backups and does not change training, simulation or analysis code.
"""
from __future__ import annotations
import ast
from pathlib import Path
from textwrap import dedent

BROKEN_TEST='test_full_guarded_installer_plan_and_apply_on_exact_mock_source'
REPLACEMENT=dedent('''
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
        p.write_text('# mock source for guarded compatibility test\\n',encoding='utf-8')
        tool.REQUIRED_UNCHANGED[relative]=tool.blob_sha(p.read_bytes())
    originals={
        'MBPO_workspace/src/sample_complexity.py': '''+repr('''import json
from pathlib import Path

def _read_csv(path):return []

def inspect(root,aggregate,daily,references):
    split={'train_ids':['2020-06-21']}
    episodes = _read_csv(root / "episodes.csv")
    if not aggregate or not daily or not references:
        raise ValueError('missing rows')

def report(out,report,steps,sac_matrix,mbpo_matrix,target):
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

def plot(ax,steps,sac_lo,sac_hi,mbpo_lo,mbpo_hi,diff_lo,diff_hi):
    ax.fill_between(steps, sac_lo, sac_hi, alpha=0.18)
    ax.fill_between(steps, mbpo_lo, mbpo_hi, alpha=0.18)
    ax.fill_between(steps, diff_lo, diff_hi, alpha=0.2)
''')+''',
        'MBPO_workspace/workflows/09_monte_carlo_evaluate.py': '''+repr('''def render(out,clock_rows,args,rows,first_day_clock_rows,paired):
    write_csv(out/'daily_power_mean_std_by_utc_step.csv',clock_rows)
    report={
        'power_output':'daily_power_mean_std_by_utc_step.csv (every selected historical day)',
    }
    if not args.skip_figures:
        plot_diagnostics(out,rows,first_day_clock_rows,paired)
''')+''',
        'MBPO_workspace/src/plots.py': '''+repr('''def plot(ax):
    ax.set_title('Learning curves (weather distributions may differ by sampled episode)')
''')+''',
        'PostProcessing_MC_patch/tools/verify_mc_environment.py': '''+repr('''#!/usr/bin/env python3

def main(argv=None):
    args=parser.parse_args(argv)
    from mbpo.env.trajectory_uncertainty import normalized_spec
''')+''',
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
''')


def updated_test_text(original:str)->str:
    if '/mnt/data/PostProcessing_MC_critical_review_v2_no_backup.zip' not in original:
        return original
    tree=ast.parse(original)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==BROKEN_TEST]
    if len(functions)!=1:raise ValueError('Cannot identify exactly one legacy test function')
    fn=functions[0]
    block='\n'.join(original.splitlines()[fn.lineno-1:fn.end_lineno])
    if 'zipfile.ZipFile' not in block or '/mnt/data/' not in block:
        raise ValueError('Legacy test function was changed; refusing an unverified replacement')
    before=original.splitlines(keepends=True)[:fn.lineno-1]
    after=original.splitlines(keepends=True)[fn.end_lineno:]
    revised=''.join(before)+REPLACEMENT+'\n'+''.join(after)
    ast.parse(revised)
    return revised

PORTABLE_V2_SOURCE_CHECK=dedent('''
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
''')

PORTABLE_DIRECT_SCRIPT_CHECK=dedent('''
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
''')


def repair_legacy_tests(original:str)->str:
    """Replace ONLY historically known nonportable test bodies; idempotent."""
    tree=ast.parse(original)
    replacements={
      BROKEN_TEST:("/mnt/data/PostProcessing_MC_critical_review_v2_no_backup.zip",REPLACEMENT),
      'test_v2_sources_validate_exact_installer_anchors':(
          "/mnt/data/PostProcessing_MC_critical_review_v2_no_backup.zip",PORTABLE_V2_SOURCE_CHECK),
      'test_verify_script_can_display_help_as_direct_file_without_importing_pvlib':(
          "/mnt/data/mc_review_v2/tools/verify_mc_environment.py",PORTABLE_DIRECT_SCRIPT_CHECK),
    }
    lines=original.splitlines(keepends=True)
    changes=[]
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef) or node.name not in replacements:
            continue
        old_path,new_function=replacements[node.name]
        span=''.join(lines[node.lineno-1:node.end_lineno])
        if old_path not in span:continue
        if node.name==BROKEN_TEST and 'zipfile.ZipFile' not in span:
            raise ValueError('Previous installer test changed unexpectedly; not replacing')
        changes.append((node.lineno-1,node.end_lineno,new_function+'\n'))
    for start,end,text in sorted(changes,reverse=True):
        lines[start:end]=[text]
    revised=''.join(lines)
    ast.parse(revised)
    return revised
