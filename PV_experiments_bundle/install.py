#!/usr/bin/env python3
"""Install the local PV experiment manager and repair one nonportable test.

No Git changes, backups, deletions or overwriting of unknown source versions.
"""
from __future__ import annotations
import argparse,ast,difflib,os,tempfile
from pathlib import Path
from portable_test_fix import repair_legacy_tests

SOURCE=Path(__file__).resolve().parent
PACKAGE=SOURCE/'PV_experiments'
LEGACY='PostProcessing_pipeline_review_patch/tests/test_pipeline_statistics.py'


def plan(root:Path):
    if not (root/'MBPO_workspace/workflows/09_monte_carlo_evaluate.py').is_file() or not (root/'PostProcessing_MC_patch/tools/verify_mc_environment.py').is_file():
        raise ValueError('Expected repository + Monte Carlo V2 workflows not found. Extract this package in the PostProcessing root.')
    if not (root/'MBPO_workspace/src/sample_complexity.py').is_file():
        raise ValueError('Required sample-complexity workflow is absent')
    changes={}
    for p in PACKAGE.rglob('*.py'):
        target=root/p.relative_to(SOURCE)
        content=p.read_text(encoding='utf-8')
        ast.parse(content,filename=str(target))
        if target.exists():
            old=target.read_text(encoding='utf-8')
            if old==content:continue
            raise FileExistsError(f'Existing experiment-manager code differs: {target}. Refusing overwrite.')
        changes[p.relative_to(SOURCE).as_posix()]=('',content)
    test=root/LEGACY
    if test.exists():
        old=test.read_text(encoding='utf-8')
        new=repair_legacy_tests(old)
        if new!=old:
            changes[LEGACY]=(old,new)
    return changes


def print_diff(changes):
    out=[]
    for rel,(old,new) in sorted(changes.items()):
        out.append('diff --git a/'+rel+' b/'+rel+'\n')
        if not old:out.append('new file mode 100644\n')
        out.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
            fromfile='a/'+rel if old else '/dev/null',tofile='b/'+rel))
    return ''.join(out)


def install(root:Path,changes:dict):
    staged=[]
    try:
        for rel,(_old,new) in sorted(changes.items()):
            path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,
                    prefix='.pv_experiment_',suffix='.tmp',delete=False) as f:
                f.write(new);f.flush();os.fsync(f.fileno());staged.append((Path(f.name),path))
        for source,target in staged:os.replace(source,target)
    finally:
        for source,_ in staged:
            if source.exists():source.unlink()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,help='Existing PostProcessing repository checkout')
    p.add_argument('--apply',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--diff-file',default=None)
    a=p.parse_args(argv)
    if a.apply and a.dry_run:p.error('--apply cannot be combined with --dry-run')
    root=Path(a.root).expanduser().resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    changes=plan(root)
    print(f'Portable test repair + experiment manager changes: {len(changes)}')
    for rel,(old,_) in sorted(changes.items()):print(' ', 'MOD' if old else 'NEW',rel)
    if a.diff_file:
        path=Path(a.diff_file);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(print_diff(changes),encoding='utf-8');print('Reviewable diff:',path)
    if not a.apply:
        print('Dry-run finished. No files changed.')
        return
    install(root,changes)
    print('Applied locally. No backups or Git commands used.')

if __name__=='__main__':main()
