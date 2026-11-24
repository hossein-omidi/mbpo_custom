#!/usr/bin/env python3
"""Strict local corrective patch for PostProcessing experiment manager.

Updates only reviewed manager interfaces and three synthetic test functions.
No backups, no Git operations, no overwriting unknown source changes.
"""
from __future__ import annotations
import argparse,ast,difflib,hashlib,os,tempfile
from pathlib import Path
from repair_tests import fix_pipeline_test,fix_manager_tests

BASE=Path(__file__).resolve().parent
EXPECTED_SHA256={
    'PV_experiments/evaluate.py':'dc7cad5a7bd1bdc56363f776d4ef79be78a675bf518ff8cf08beffa221327c61',
    'PV_experiments/report.py':'35fe21adea9c15fae18c83c15c83337c0086c50a6353a3884d597110377c927a',
}
TEST_PIP='PostProcessing_pipeline_review_patch/tests/test_pipeline_statistics.py'
TEST_MANAGER='PV_experiments_bundle/tests/test_experiment_manager.py'


def sha256(b):return hashlib.sha256(b).hexdigest()


def plan(root):
    root=Path(root)
    changes={}
    for relative,expected in EXPECTED_SHA256.items():
        installed=root/relative
        source=BASE/relative
        if not installed.is_file():raise FileNotFoundError(f'Expected manager source is missing: {installed}')
        old=installed.read_text(encoding='utf-8')
        new=source.read_text(encoding='utf-8')
        if old==new:continue
        if sha256(old.encode('utf-8'))!=expected:
            raise ValueError(f'{relative}: installed version differs from the audited manager; no changes applied')
        ast.parse(new,filename=relative)
        changes[relative]=(old,new)
    for relative,fn in ((TEST_PIP,fix_pipeline_test),(TEST_MANAGER,fix_manager_tests)):
        installed=root/relative
        if not installed.is_file():raise FileNotFoundError(f'Expected installed test is missing: {installed}')
        old=installed.read_text(encoding='utf-8')
        new=fn(old)
        if old!=new:
            ast.parse(new,filename=relative)
            changes[relative]=(old,new)
    return changes


def diff(changes):
    result=[]
    for relative,(old,new) in sorted(changes.items()):
        result.append('diff --git a/'+relative+' b/'+relative+'\n')
        result.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                                           fromfile='a/'+relative,tofile='b/'+relative))
    return ''.join(result)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,help='Root of local PostProcessing checkout')
    g=p.add_mutually_exclusive_group()
    g.add_argument('--apply',action='store_true',help='Apply reviewed modifications locally')
    g.add_argument('--dry-run',action='store_true',help='Verify expected source versions only')
    p.add_argument('--diff-file',default=None,help='Optional unified-diff output')
    args=p.parse_args(argv)
    root=Path(args.root).expanduser().resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    changes=plan(root)
    print('Reviewed local modifications:',len(changes))
    for relative in sorted(changes):print('  MOD',relative)
    if args.diff_file:
        dest=Path(args.diff_file)
        if dest.exists():raise FileExistsError(f'Diff path already exists: {dest}')
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(diff(changes),encoding='utf-8')
    if not args.apply:
        print('No repository files changed. No backups created.')
        return
    staged=[]
    try:
        for relative,(_old,new) in sorted(changes.items()):
            dest=root/relative
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=dest.parent,
                                             prefix='.pv_repair_',suffix='.tmp',delete=False) as f:
                f.write(new)
                f.flush();os.fsync(f.fileno())
                staged.append((Path(f.name),dest))
        for src,dest in staged:
            os.replace(src,dest)
    finally:
        for src,_ in staged:
            if src.exists():src.unlink()
    print('Applied locally. No backup files, no Git commands and no GitHub writes.')

if __name__=='__main__':main()
