"""Fail-closed file-integrity gate after copying Stage-2 full replacement files.

This does NOT imply NSRDB, optimizer, or scientific model validation. Run
scripts/pvwatts_stage2_audit.py next for runtime acceptance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'STAGE2_DIRECT_COPY_MANIFEST.json'


def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()


def main():
    m=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if m.get('schema')!='pvwatts-stage2-direct-copy-v1':
        raise RuntimeError('Unrecognized replacement manifest')
    expected_files=m.get('files',{})
    if len(expected_files)!=13 or int(m.get('target_count',-1))!=13:
        raise RuntimeError('Replacement manifest is incomplete')
    for group,key in ((expected_files,'replacement_sha256'),(m['addons'],None)):
        for name,meta in group.items():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts:
                raise RuntimeError('Unsafe filename in manifest')
            target=ROOT/path
            if not target.is_file() or target.is_symlink():
                raise FileNotFoundError('Missing / symlinked installed file: '+name)
            expected=meta[key] if key else meta
            if sha(target.read_bytes())!=expected:
                raise RuntimeError('Installed source differs from verified replacement: '+name)
            if name.endswith('.py'):
                compile(target.read_text(encoding='utf-8'),name,'exec')
    for name in expected_files:
        if 'PVWATTS_STAGE2_INTEGRATED' not in (ROOT/name).read_text(encoding='utf-8'):
            raise RuntimeError('Missing Stage2 marker: '+name)
    print('PASS: all 13 full replacements and all addons match the generated manifest; Python source compiles.')
    print('NOT YET VERIFIED: actual NSRDB PVWatts SAC/MBPO training/evaluation. Run scripts/pvwatts_stage2_audit.py.')

if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,FileNotFoundError,KeyError,SyntaxError,OSError) as exc:
        print('STAGE2 DIRECT-COPY VERIFICATION FAIL:',exc,file=sys.stderr)
        raise SystemExit(2)
