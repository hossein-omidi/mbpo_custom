"""Check direct-copy generator with synthetic local source files, no NSRDB needed."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]


def _load(name,rel):
    spec=importlib.util.spec_from_file_location(name,ROOT/rel)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generator_produces_full_ready_copy_and_independent_backup(tmp_path,monkeypatch):
    build=_load('_stage2_builder_test','scripts/build_stage2_direct_replacements.py')
    verify=_load('_stage2_copy_verify_test','scripts/verify_stage2_direct_copy.py')
    originals, edited = {}, {}
    for n in range(13):
        name=f'workspace/module_{n:02d}.py'
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        old=f'VALUE = {n}\n'.encode()
        new=f'# PVWATTS_STAGE2_INTEGRATED\nVALUE = {n*10}\n'.encode()
        path.write_bytes(old);originals[name]=old;edited[name]=new

    class FakePatcher:
        def check_files(self):return originals,edited

    for name in ('scripts/demo.py','STAGE2_DIRECT_COPY_README.md'):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('VALUE=1\n' if name.endswith('.py') else 'Documentation\n')
    monkeypatch.setattr(build,'ROOT',tmp_path)
    monkeypatch.setattr(build,'ADDONS',('scripts/demo.py','STAGE2_DIRECT_COPY_README.md'))
    monkeypatch.setattr(build,'load_patcher',lambda:FakePatcher())
    build.main()
    replacements=next(tmp_path.glob('PVWatts_Stage2_Direct_Replacements_*.zip'))
    backup=next(tmp_path.glob('PVWatts_Stage2_Local_Originals_Backup_*.zip'))
    assert not replacements.samefile(backup)
    ready=tmp_path/'PVWATTS_STAGE2_READY_TO_COPY'
    assert ready.is_dir()
    assert all((ready/name).read_bytes()==data for name,data in edited.items())
    assert (ready/'STAGE2_DIRECT_COPY_MANIFEST.json').is_file()
    # Generation NEVER modifies original inputs.
    assert all((tmp_path/n).read_bytes()==v for n,v in originals.items())
    with zipfile.ZipFile(replacements) as z:
        assert z.testzip() is None
        assert all(z.read(n)==v for n,v in edited.items())
        assert 'STAGE2_DIRECT_COPY_MANIFEST.json' in z.namelist()
    with zipfile.ZipFile(backup) as z:
        assert z.testzip() is None
        assert all(z.read('ORIGINAL_SOURCES/'+n)==v for n,v in originals.items())
    # Simulate the user's direct copy step (explicit unpack operation).
    with zipfile.ZipFile(replacements) as z:z.extractall(tmp_path)
    monkeypatch.setattr(verify,'ROOT',tmp_path)
    monkeypatch.setattr(verify,'MANIFEST',tmp_path/'STAGE2_DIRECT_COPY_MANIFEST.json')
    verify.main()
    first=next(iter(edited))
    (tmp_path/first).write_text('VALUE=999\n')
    with pytest.raises(RuntimeError,match='differs'):
        verify.main()
