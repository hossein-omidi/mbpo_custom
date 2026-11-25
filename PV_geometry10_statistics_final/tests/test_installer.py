import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('installer',Path(__file__).resolve().parents[1]/'install.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.fixture
def repo(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir();(root/'module.py').write_text('value=1\n');(root/'dep.py').write_text('a=1\n')
    m.git(root,'init','-q');m.git(root,'add','.');m.git(root,'-c','user.name=fixture','-c','user.email=fixture@example.com','commit','-qm','base')
    monkeypatch.setattr(m,'BASE',m.git(root,'rev-parse','HEAD').decode().strip());monkeypatch.setattr(m,'TRANSFORMS',{'module.py':lambda s:m.replace(s,'value=1','value=2')});monkeypatch.setattr(m,'DEPENDENCIES',('dep.py',))
    return root

def test_apply_and_idempotent(repo):
    patch,files=m.build(repo);assert 'module.py' in files
    m.git(repo,'apply','--check','-',input=patch);assert (repo/'module.py').read_text()=='value=1\n'
    m.git(repo,'apply','-',input=patch);assert (repo/'module.py').read_text()=='value=2\n'
    assert m.build(repo)==(b'',[])

def test_refuse_unexpected_local_edits(repo):
    (repo/'module.py').write_text('value=3\n')
    with pytest.raises(ValueError,match='Unexpected local edits'):m.build(repo)

def test_refuse_dependency_change(repo):
    (repo/'dep.py').write_text('a=2\n')
    with pytest.raises(ValueError,match='Unexpected dependency edits'):m.build(repo)

def test_replace_anchor_is_unique():
    with pytest.raises(ValueError):m.replace('aaaa','a','b')


def test_upgrade_previous_installer_and_payload(repo,monkeypatch):
    # Simulate exact previously installed source plus previous payload bytes.
    monkeypatch.setattr(m.previous,'TRANSFORMS',{'module.py':lambda s:m.replace(s,'value=1','value=9')})
    (repo/'module.py').write_text('value=9\n')
    # The archive is self-contained; a fixture of previous payload contents
    # can be constructed with a test-only payload manifest instead of the old ZIP.
    import hashlib,json
    package=repo.parent/'fixture_package';(package/'payload').mkdir(parents=True)
    (package/'payload'/'added.py').write_text('value=2\n')
    (repo/'added.py').write_text('value=1\n')
    (package/'previous_payload_sha256.json').write_text(json.dumps({'added.py':hashlib.sha256(b'value=1\n').hexdigest()}))
    monkeypatch.setattr(m,'HERE',package)
    patch,files=m.build(repo);assert set(files)=={'module.py','added.py'}
    m.git(repo,'apply','--check','-',input=patch);m.git(repo,'apply','-',input=patch)
    assert (repo/'module.py').read_text()=='value=2\n' and (repo/'added.py').read_text()=='value=2\n'
    assert m.build(repo)==(b'',[])
