"""Generate COMPLETE ready-to-copy Stage-2 replacement files from local repo originals.

This script DOES NOT change any existing repo source, dataset, policy checkpoint,
run, or evaluation output. It transforms the files in memory, compiles them,
then produces TWO separate ZIPs:

  PVWatts_Stage2_Direct_Replacements_<timestamp>.zip
  PVWatts_Stage2_Local_Originals_Backup_<timestamp>.zip

The reason for local generation is that the published GitHub tree differs from
some local SAC/MBPO modules. Copying remote full source files would silently
remove user's custom local changes. This preserves unmodified local content.

Usage (repo root, with v3 addon files extracted):
    python scripts/build_stage2_direct_replacements.py

To install AFTER inspecting outputs:
    unzip -o PVWatts_Stage2_Direct_Replacements_<timestamp>.zip -d .
    python scripts/verify_stage2_direct_copy.py
    python -m pytest -q tests/test_pv_power_backend.py tests/test_pvwatts_stage2.py
    python scripts/pvwatts_stage2_audit.py
"""
from __future__ import annotations

import datetime
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile
import tempfile
import os

ROOT=Path(__file__).resolve().parents[1]
PATCHER=ROOT/'scripts'/'apply_pvwatts_stage2.py'

ADDONS=(
    'mbpo/env/pv_power_backend.py',
    'mbpo/env/pv_power_cli.py',
    'MBPO_workspace/src/power_references.py',
    'scripts/apply_pvwatts_stage2.py',
    'scripts/pvwatts_stage1_preflight.py',
    'scripts/pvwatts_stage2_audit.py',
    'scripts/build_stage2_direct_replacements.py',
    'scripts/verify_stage2_direct_copy.py',
    'tests/test_pv_power_backend.py',
    'tests/test_pvwatts_stage2.py',
    'tests/test_pvwatts_stage2_patcher.py',
    'tests/pvwatts_daily_sensitivity.py',
    'tests/test_stage2_all_workflow_source.py',
    'tests/test_stage2_direct_copy_generator.py',
    'STAGE2_DIRECT_COPY_README.md',
)


def digest(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def load_patcher():
    if not PATCHER.is_file():
        raise FileNotFoundError('Missing v3 patcher source: '+str(PATCHER))
    spec=importlib.util.spec_from_file_location('_pvwatts_v3_source_transforms',PATCHER)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if len(mod.TARGETS)!=13 or set(mod.TARGETS)!=set(mod.PATCHES):
        raise RuntimeError('Unrecognized transformation target map')
    return mod


def zip_write(path:Path,files:dict[str,bytes]):
    if path.exists():
        raise FileExistsError(f'Output already exists; refusing overwrite: {path}')
    # Write to memory to avoid partial files on compilation or content errors.
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in sorted(files):
            if name.startswith('/') or '..' in Path(name).parts or '\\' in name:
                raise ValueError('Unsafe output entry '+name)
            z.writestr(name,files[name])
    payload=buf.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if z.testzip() is not None or set(z.namelist())!=set(files):
            raise RuntimeError('Output ZIP verification failed')
        for name,val in files.items():
            if z.read(name)!=val:
                raise RuntimeError('Output ZIP content mismatch '+name)
    with path.open('xb') as f:f.write(payload)


def main():
    p=load_patcher()
    # All 13 original sources must be recognizable. This is a strict safety
    # boundary: no partial source replacements, no dummy fallbacks, no overrides.
    originals, edited=p.check_files()
    if len(originals)!=13 or set(edited)!=set(originals):
        raise RuntimeError('Incomplete source transformation; refusing to export')
    for target,data in edited.items():
        compile(data.decode('utf-8'),target,'exec')
        if data == originals[target]:raise RuntimeError('Unchanged core file '+target)
    additions={}
    for name in ADDONS:
        f=ROOT/name
        if not f.is_file() or f.is_symlink():
            raise FileNotFoundError('Missing or symlinked addon: '+str(f))
        content=f.read_bytes()
        if name.endswith('.py'):compile(content.decode('utf-8'),name,'exec')
        additions[name]=content
    # Snapshot both original and proposed bytes. Check no mutation of inputs
    # from this point until outputs are written.
    src_hash={name:digest(data) for name,data in originals.items()}
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prepared=edited|additions
    manifest={
        'schema':'pvwatts-stage2-direct-copy-v1',
        'generated_utc':stamp,
        'source':'exact-local-files',
        'target_count':len(edited),
        'files':{name:{'original_sha256':src_hash[name],
                       'replacement_sha256':digest(edited[name])} for name in sorted(edited)},
        'addons':{name:digest(additions[name]) for name in sorted(additions)},
    }
    prepared['STAGE2_DIRECT_COPY_MANIFEST.json']=json.dumps(manifest,indent=2,sort_keys=True).encode()
    backup_files={'ORIGINAL_SOURCES/'+name:content for name,content in originals.items()}
    backup_files['BACKUP_MANIFEST.json']=json.dumps(manifest,indent=2,sort_keys=True).encode()
    rep_path=ROOT/f'PVWatts_Stage2_Direct_Replacements_{stamp}.zip'
    bak_path=ROOT/f'PVWatts_Stage2_Local_Originals_Backup_{stamp}.zip'
    if rep_path.exists() or bak_path.exists():raise FileExistsError('Time-stamped output already exists')
    if (ROOT/'PVWATTS_STAGE2_READY_TO_COPY').exists():
        raise FileExistsError('Ready-to-copy folder already exists; move it away before rebuilding')
    for name,expected in src_hash.items():
        if digest((ROOT/name).read_bytes())!=expected:
            raise RuntimeError('Repository source changed during preparation: '+name)
    zip_write(rep_path,prepared)
    try:zip_write(bak_path,backup_files)
    except BaseException:
        rep_path.unlink(missing_ok=True)
        raise
    for name,expected in src_hash.items():
        if digest((ROOT/name).read_bytes())!=expected:
            raise RuntimeError('Source changed unexpectedly: '+name)
    # Also materialize the complete files for an ordinary user-controlled
    # directory copy (no installer/patch command needed after preparation).
    ready = ROOT / 'PVWATTS_STAGE2_READY_TO_COPY'
    if ready.exists():
        raise FileExistsError('Ready-to-copy folder already exists; move it away before rebuilding: '+str(ready))
    with tempfile.TemporaryDirectory(prefix='.stage2-prepare-',dir=ROOT) as tmpdir:
        stage=Path(tmpdir)/'ready'
        for name, contents in prepared.items():
            output=stage/name
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_bytes(contents)
        if any((stage/name).read_bytes()!=data for name,data in prepared.items()):
            raise RuntimeError('Ready-to-copy file verification failed')
        os.replace(stage, ready)
    for name, expected in src_hash.items():
        if digest((ROOT/name).read_bytes())!=expected:
            raise RuntimeError('Original source changed during directory preparation: '+name)
    print('READY-TO-COPY FOLDER:',ready)
    print('Install complete replacement files by: cp -a PVWATTS_STAGE2_READY_TO_COPY/. .')
    print('The originals backup ZIP exists; keep it safe before copying.')
    print('COMPLETE replacement sources generated; original repository code UNCHANGED.')
    print(f'{len(edited)} integrated full source files, {len(additions)} addon/docs/test files.')
    print('REPLACEMENTS:',rep_path)
    print('ORIGINALS BACKUP:',bak_path)
    print('NEXT: inspect READY folder and backup; copy the complete files into repo root.')
    print('NEXT: python scripts/verify_stage2_direct_copy.py')
    print('NEXT: python -m pytest -q tests/test_pv_power_backend.py tests/test_pvwatts_stage2.py tests/test_pvwatts_stage2_patcher.py')
    print('NEXT: python scripts/pvwatts_stage2_audit.py')

if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError,OSError,SyntaxError) as exc:
        print('DIRECT-REPLACEMENT BUILD ABORTED; source files untouched:',exc,file=sys.stderr)
        raise SystemExit(2)
