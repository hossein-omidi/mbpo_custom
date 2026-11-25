"""Add the V2 representation modules; no replacement of existing files."""
from __future__ import annotations
import argparse,ast,hashlib,os,tempfile
from pathlib import Path
BASE=Path(__file__).resolve().parent
EXPECTED={
'PV_experiments/core.py':'8ac6ac75cc58148c914f7ff5c8cb1f82d1d1f5e2',
'PV_experiments/evaluate.py':'26f681b2159899877ee2fe0f23f1f2e7519990e2',
'PV_experiments/report.py':'2f1bf852cdb7784abfff326e2ef3f1f86941843e',
'MBPO_workspace/src/sample_complexity.py':'07323ab36a077da587fcf59911ee6321a90a9cfc',
'MBPO_workspace/src/curve_statistics.py':'440cd49944781cff401f9f68a199083cbfcb1fc4',
'SAC_workspace/src/training/experiment_protocol.py':'caf11e2f6801e98ac6c8ed817117f3176ba0db8c',
'SAC_workspace/src/training/trainer.py':'99ef8bde096738a3c9a457ef259440037c35bd35',
'MBPO_workspace/src/trainer.py':'31611b94ad92a0761d58061ab2cb158419189e9f',
'MBPO_workspace/workflows/09_monte_carlo_evaluate.py':'fd2fe2e740ce65b77bb73954743c0e24c2d6a42a',
}
def blob(b):return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
blob_sha = blob

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
    for rel,h in expected.items():
        p=root/rel
        if not p.is_file():errors.append(f'Missing: {rel}');continue
        b=p.read_bytes()
        if (blob(b)!=h and blob(b.replace(b'\r\n',b'\n'))!=h
                and not same_audited_python_ast(root,rel,b,h)):
            errors.append(f'Different audited source: {rel} (local Git blob SHA-1 {blob(b)}; '
                          'executable AST cannot be verified as identical to the audited reference)')
    if errors:raise ValueError('No files changed; incompatible checkout:\n'+'\n'.join(errors))
    result={}
    for p in sorted((BASE/'PV_experiments').glob('*.py')):
        rel=p.relative_to(BASE);b=p.read_bytes();ast.parse(b,filename=str(rel));target=root/rel
        if target.exists():
            if target.read_bytes()!=b:raise FileExistsError(f'Existing file differs, refusing overwrite: {rel}')
        else:result[rel]=b
    return result

def apply(root,changes):
    for rel,b in changes.items():
        p=Path(root)/rel;p.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.pv-report-',dir=p.parent)
        try:
            with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
            os.link(tmp,p)
        finally:os.unlink(tmp)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.');g=p.add_mutually_exclusive_group();g.add_argument('--dry-run',action='store_true');g.add_argument('--apply',action='store_true');a=p.parse_args(argv);root=Path(a.root).resolve();changes=plan(root)
    print('Audited source interfaces:',len(EXPECTED))
    for rel in changes:print('NEW',rel)
    if a.apply:apply(root,changes);print('Installed',len(changes),'new files; no backups, overwrites or Git writes.')
    else:print('Dry run:',len(changes),'new files; nothing changed.')
if __name__=='__main__':main()
