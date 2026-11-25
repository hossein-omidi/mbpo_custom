#!/usr/bin/env python3
"""Add scientific reporting modules to the audited PostProcessing checkout only.

No source replacement, no Git mutations, no backups, no existing result deletion.
Uses read-only git show only to verify comment/format-only local workflow variants.
"""
from __future__ import annotations
import argparse,ast,hashlib,os,tempfile
from pathlib import Path

BASE=Path(__file__).resolve().parent
EXPECTED={
 'PV_experiments/core.py':'8ac6ac75cc58148c914f7ff5c8cb1f82d1d1f5e2',
 'PV_experiments/evaluate.py':'26f681b2159899877ee2fe0f23f1f2e7519990e2',
 'PV_experiments/report.py':'2f1bf852cdb7784abfff326e2ef3f1f86941843e',
 'MBPO_workspace/workflows/09_monte_carlo_evaluate.py':'fd2fe2e740ce65b77bb73954743c0e24c2d6a42a',
 'MBPO_workspace/workflows/06_model_validation.py':'628825b968d05c1e2d27075b8f7ff672a785d17b',
 'MBPO_workspace/src/sample_complexity.py':'07323ab36a077da587fcf59911ee6321a90a9cfc',
 'MBPO_workspace/src/curve_statistics.py':'440cd49944781cff401f9f68a199083cbfcb1fc4',
 'MBPO_workspace/src/config.py':'5be07701ddfb39e14efecf4d846fe2a4c51ebee4',
 'SAC_workspace/src/environment/environment.py':'2b1f8e65e43a56ce2fe79356cdec1d97dfaa4176',
 'SAC_workspace/src/training/experiment_protocol.py':'caf11e2f6801e98ac6c8ed817117f3176ba0db8c',
 'SAC_workspace/src/config/config.py':'a0039ab99ca275e02feb15e10d54885b46dbc90a',
 'mbpo/env/baseline_contract.py':'0db6e7ab84c3fce497411220bdcf527455747941',
 'mbpo/env/pv_power_backend.py':'263ba194dfb7a30b9e4b526279ce1226920714c4',
 'mbpo/env/pv_tracking.py':'d0b42d5d651ae77efa4f4f282495540ce072cb5c',
 'mbpo/env/stage_a_pv_tracking.py':'92b32a769bf4502f699b4cb65f77bf7c254d13d1',
 'mbpo/env/nsrdb_weather.py':'01689dce6f15426a8537f9e46879744876afb3bd',
 'mbpo/env/stage_a_data_contract.py':'15be6f36220733e8af8f22e678d335e90ae28002',
 'mbpo/env/pvlib_physics.py':'f5df004cbc31ae0f1cf2b5deb3a522754611160c',
 'mbpo/env/trajectory_uncertainty.py':'dcc3bb43729525f2994a52e44068b2b1d30888e7',
 'mbpo/env/nsrdb_iotools.py':'75283a7c1ace9aab118af024ea76f39b77dbe9e7',
 'examples/config/pv_tracking/_nsrdb_base.py':'48ba072203796006908ff63f5b07b20917065b26',
 'examples/config/pv_tracking/conf_compact.py':'3e1ea399af0b7457353f352b8b756970e294b201',
}


def blob_sha(b):return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()

# The Monte Carlo workflow is READ-ONLY in these additive packages.  Accept a
# local comment/format-only variant when Git can provide the exact audited
# source.  We compare complete Python ASTs, not just imports/signatures:
# executable modifications must still be reviewed, not silently whitelisted.
MC_PATH = 'MBPO_workspace/workflows/09_monte_carlo_evaluate.py'
AUDITED_COMMIT = 'f0995849c98f07be2cbdab9202b49ab89ca50241'


def same_audited_python_ast(root, relative, content, audited_sha):
    """True only if current code is executable-AST-identical to a pinned Git blob.

    Git is used read-only; this function does not checkout, fetch, reset, or
    modify files.  A recognized reference must have the expected SHA-1.
    """
    if relative != MC_PATH:
        return False
    import subprocess
    for ref in (AUDITED_COMMIT, 'origin/PostProcessing', 'PostProcessing'):
        try:
            proc = subprocess.run(
                ['git', '-C', str(root), 'show', f'{ref}:{relative}'],
                capture_output=True, timeout=10, check=False)
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode or blob_sha(proc.stdout) != audited_sha:
            continue
        try:
            actual = ast.dump(ast.parse(content, filename=relative, type_comments=True),
                              include_attributes=False)
            reference = ast.dump(ast.parse(proc.stdout, filename=relative, type_comments=True),
                                 include_attributes=False)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f'Cannot validate {relative} Python syntax: {exc}') from exc
        if actual == reference:
            print(f'AST-IDENTICAL verified: {relative}; reference={ref}; '
                  'only comments/whitespace or equivalent formatting differ; workflow left untouched.')
            return True
        return False
    return False



def plan(root,expected=None):
    root=Path(root);expected=EXPECTED if expected is None else expected
    errors=[]
    for relative,digest in expected.items():
        p=root/relative
        if not p.is_file():errors.append(f'missing {relative}');continue
        b=p.read_bytes()
        if (blob_sha(b)!=digest and blob_sha(b.replace(b'\r\n',b'\n'))!=digest
                and not same_audited_python_ast(root,relative,b,digest)):
            errors.append(f'different source: {relative} (local Git blob SHA-1 {blob_sha(b)}; '
                          'executable AST cannot be verified as identical to the audited reference)')
    if errors:raise ValueError('Checkout does not match audited interfaces; nothing installed:\n'+'\n'.join(errors))
    changes={}
    for p in sorted((BASE/'PV_experiments').rglob('*.py')):
        rel=p.relative_to(BASE);target=root/rel;b=p.read_bytes();ast.parse(b,filename=str(rel))
        if target.exists():
            if target.read_bytes()==b:continue
            raise FileExistsError(f'Existing reporting module differs: {target}; no overwrite')
        changes[rel]=b
    return changes


def apply(root,changes):
    # Only create previously absent paths; a concurrent new file is never replaced.
    for rel,b in changes.items():
        path=Path(root)/rel;path.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix='.pv_representation-',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
            os.link(tmp,path)
        finally:os.unlink(tmp)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.');g=p.add_mutually_exclusive_group();g.add_argument('--apply',action='store_true');g.add_argument('--dry-run',action='store_true');a=p.parse_args(argv);root=Path(a.root).resolve();changes=plan(root)
    print(f'Audited interfaces: {len(EXPECTED)}; reference commit {AUDITED_COMMIT}')
    for rel in changes:print('NEW',rel)
    if a.apply:apply(root,changes);print(f'Installed {len(changes)} new files; existing repository files unchanged.')
    else:print(f'Dry run: {len(changes)} new files; nothing changed.')

if __name__=='__main__':main()
