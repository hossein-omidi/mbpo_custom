"""Installer mechanics exercised against a small real Git checkout, not the PV runtime."""
from pathlib import Path
import pytest
import integrated as ap


def setup(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir();ap.git(root,'init','-q')
    src=root/'test.py';src.write_text('VALUE = 1\n');ap.git(root,'add','test.py')
    payload=tmp_path/'payload';payload.mkdir();(payload/'new.py').write_text('NEW = 2\n')
    monkeypatch.setattr(ap,'TRANSFORMS',{'test.py':lambda s:ap.replace(s,'VALUE = 1','VALUE = 3')})
    monkeypatch.setattr(ap,'PIN',{'test.py':ap.blob(src.read_bytes())});monkeypatch.setattr(ap,'PAYLOAD',payload)
    return root


def test_guarded_git_apply(tmp_path,monkeypatch):
    root=setup(tmp_path,monkeypatch);patch=ap.patch_bytes(ap.build(root))
    ap.git(root,'apply','--check','--whitespace=error-all','-',input=patch)
    assert (root/'test.py').read_text()=='VALUE = 1\n' and not (root/'new.py').exists()
    ap.git(root,'apply','--whitespace=error-all','-',input=patch)
    assert (root/'test.py').read_text()=='VALUE = 3\n' and (root/'new.py').read_text()=='NEW = 2\n'
    assert sorted(p.name for p in root.iterdir())==['.git','new.py','test.py']
    with pytest.raises(ValueError):ap.build(root)


def test_conflicting_dirty_source_refused(tmp_path,monkeypatch):
    root=setup(tmp_path,monkeypatch);(root/'test.py').write_text('VALUE = 9\n')
    with pytest.raises(ValueError,match='source differs'):ap.build(root)
    assert not (root/'new.py').exists()


def test_git_ignored_new_file_refused(tmp_path,monkeypatch):
    root=setup(tmp_path,monkeypatch);(root/'.gitignore').write_text('new.py\n')
    with pytest.raises(ValueError,match='Git-ignored'):ap.build(root)


def test_duplicate_anchor_refused():
    with pytest.raises(ValueError):ap.replace('aa','a','b')


def test_final_installer_handles_base_prior_and_final_without_overwriting(tmp_path,monkeypatch):
    import install as final
    import json
    root=tmp_path/'repo';root.mkdir();ap.git(root,'init','-q');ap.git(root,'config','user.name','Fixture');ap.git(root,'config','user.email','fixture@example.invalid')
    src=root/'x.py';src.write_text('VALUE = 1\n');ap.git(root,'add','x.py');ap.git(root,'commit','-qm','base')
    base=ap.git(root,'rev-parse','HEAD').stdout.decode().strip()
    pkg=tmp_path/'package';pkg.mkdir();(pkg/'payload').mkdir();(pkg/'payload'/'extra.py').write_text('VALUE = 2\n');(pkg/'prior_payload_hashes.json').write_text(json.dumps({}))
    monkeypatch.setattr(final,'BASE',base);monkeypatch.setattr(final,'HERE',pkg)
    monkeypatch.setattr(final,'PIN',{'x.py':ap.blob(src.read_bytes())});monkeypatch.setattr(final,'TRANSFORMS',{'x.py':lambda s:s.replace('1','3')});monkeypatch.setattr(ap,'TRANSFORMS',{'x.py':lambda s:s.replace('1','2')})
    plan=final.build(root);assert len(plan)==2
    src.write_text('VALUE = 2\n');patch=ap.patch_bytes(final.build(root));ap.git(root,'apply','--check','--whitespace=error-all','-',input=patch);ap.git(root,'apply','--whitespace=error-all','-',input=patch)
    assert src.read_text()=='VALUE = 3\n' and final.build(root)=={}
    src.write_text('VALUE = 9\n')
    with pytest.raises(ValueError,match='Local source differs'):final.build(root)
