"""Disposable Git fixtures: migration safety, not PV runtime validation."""
import importlib.util
from pathlib import Path
import subprocess
import pytest

spec=importlib.util.spec_from_file_location('geo_installer',Path(__file__).resolve().parents[1]/'install.py')
installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args]).decode()


def fixture(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    git(repo,'init','-q');git(repo,'config','user.name','fixture');git(repo,'config','user.email','fixture@example.invalid')
    (repo/'a.py').write_bytes(b'x=1\nlast=2')
    (repo/'b.py').write_bytes(b'y=1\n')
    git(repo,'add','a.py','b.py');git(repo,'commit','-qm','base')
    return repo


def migration(root,monkeypatch):
    monkeypatch.setattr(installer,'BASE',git(root,'rev-parse','HEAD').strip())
    monkeypatch.setattr(installer,'DEPENDENCIES',{})
    versions={
        'a.py':{'base':b'x=1\nlast=2','v1':b'x=2\nlast=2','v2':b'x=3\nlast=2','final':b'x=4\nlast=2'},
        'b.py':{'base':b'y=1\n','v1':b'y=2\n','v2':b'y=3\n','final':b'y=4\n'},
        'nested/new.py':{'base':None,'v1':b'z=1\n','v2':b'z=2\n','final':b'z=3\n'},
    }
    plan={k:v['final'] for k,v in versions.items()}
    monkeypatch.setattr(installer,'expected_versions',lambda root:(plan,versions))
    return plan,versions


def install(root,dry_run=False):
    plan,states=installer.make_plan(root)
    installer.apply_patch(root,installer.patch_bytes(root,plan),dry_run=dry_run)
    return plan,states


def test_clean_dry_run_and_inplace_patch(tmp_path,monkeypatch):
    root=fixture(tmp_path);plan,_=migration(root,monkeypatch)
    install(root,True)
    assert (root/'a.py').read_bytes()==b'x=1\nlast=2'
    assert not (root/'nested').exists()
    install(root)
    assert all((root/k).read_bytes()==v for k,v in plan.items())
    status=git(root,'status','--short','--untracked-files=all')
    assert ' M a.py' in status and ' M b.py' in status and '?? nested/new.py' in status
    assert not (tmp_path/'repo_geometry10').exists()
    assert git(root,'log','-1','--format=%s').strip()=='base'


@pytest.mark.parametrize('version',['v1','v2'])
def test_previous_version_upgrade(tmp_path,monkeypatch,version):
    root=fixture(tmp_path);plan,versions=migration(root,monkeypatch)
    for path,v in versions.items():
        p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v[version])
    _,states=install(root)
    assert set(states.values())=={version}
    assert all((root/k).read_bytes()==v for k,v in plan.items())


def test_mixed_versions_and_idempotence(tmp_path,monkeypatch):
    root=fixture(tmp_path);plan,versions=migration(root,monkeypatch)
    (root/'a.py').write_bytes(versions['a.py']['v1'])
    (root/'b.py').write_bytes(versions['b.py']['v2'])
    (root/'nested').mkdir();(root/'nested/new.py').write_bytes(plan['nested/new.py'])
    _,states=install(root)
    assert states=={'a.py':'v1','b.py':'v2','nested/new.py':'final'}
    _,states=install(root)
    assert set(states.values())=={'final'} and installer.patch_bytes(root,plan)==b''


def test_descendant_commit_with_recognized_staged_file(tmp_path,monkeypatch):
    root=fixture(tmp_path);plan,versions=migration(root,monkeypatch)
    (root/'a.py').write_bytes(versions['a.py']['v1']);git(root,'add','a.py');git(root,'commit','-qm','v1')
    (root/'b.py').write_bytes(versions['b.py']['v2']);git(root,'add','b.py')
    before=git(root,'ls-files','--stage')
    install(root)
    assert git(root,'ls-files','--stage')==before
    assert all((root/k).read_bytes()==v for k,v in plan.items())


def test_unrelated_local_changes_are_preserved(tmp_path,monkeypatch):
    root=fixture(tmp_path);migration(root,monkeypatch)
    (root/'notes.txt').write_text('tracked notes');git(root,'add','notes.txt');git(root,'commit','-qm','notes')
    (root/'notes.txt').write_text('unsaved work');(root/'untracked.txt').write_text('data')
    install(root)
    assert (root/'notes.txt').read_text()=='unsaved work'
    assert (root/'untracked.txt').read_text()=='data'


def test_conflicting_hunk_refuses_all_writes(tmp_path):
    root=fixture(tmp_path)
    patch=installer.patch_bytes(root,{'a.py':b'x=2\nlast=2\n','b.py':b'y=5\n','new.py':b'z=3\n'})
    (root/'b.py').write_bytes(b'y=99\n')
    with pytest.raises(subprocess.CalledProcessError):installer.apply_patch(root,patch,dry_run=False)
    assert (root/'a.py').read_bytes()==b'x=1\nlast=2'
    assert (root/'b.py').read_bytes()==b'y=99\n'
    assert not (root/'new.py').exists()


def test_unknown_target_rejected_without_writes(tmp_path,monkeypatch):
    root=fixture(tmp_path);migration(root,monkeypatch)
    (root/'b.py').write_bytes(b'y=99\n')
    with pytest.raises(ValueError,match='unrecognized working-tree'):install(root)
    assert (root/'a.py').read_bytes()==b'x=1\nlast=2'
    assert (root/'b.py').read_bytes()==b'y=99\n'
    assert not (root/'nested').exists()


def test_unknown_staged_target_is_rejected(tmp_path,monkeypatch):
    root=fixture(tmp_path);_,versions=migration(root,monkeypatch)
    (root/'a.py').write_bytes(b'x=99\n');git(root,'add','a.py')
    (root/'a.py').write_bytes(versions['a.py']['base'])
    with pytest.raises(ValueError,match='unrecognized staged'):install(root)


def test_ignored_new_file_rejected(tmp_path,monkeypatch):
    root=fixture(tmp_path);migration(root,monkeypatch)
    (root/'.gitignore').write_text('nested/\n')
    with pytest.raises(ValueError,match='ignored'):install(root)


def test_sibling_without_git_rejected(tmp_path,monkeypatch):
    root=fixture(tmp_path);migration(root,monkeypatch)
    sibling=root/'sibling_geometry10';sibling.mkdir()
    with pytest.raises(ValueError,match='original Git checkout'):installer.make_plan(sibling)
    assert not (sibling/'.git').exists()


def test_symlink_target_rejected(tmp_path,monkeypatch):
    root=fixture(tmp_path);migration(root,monkeypatch)
    (root/'nested').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):install(root)


def test_critical_dependency_drift_rejected(tmp_path,monkeypatch):
    root=fixture(tmp_path);migration(root,monkeypatch)
    monkeypatch.setattr(installer,'DEPENDENCIES',{'a.py':installer.blob_sha((root/'a.py').read_bytes())})
    (root/'a.py').write_bytes(b'x=99\n')
    with pytest.raises(ValueError,match='critical managed-pipeline dependency'):install(root)


def test_previous_transforms_generate_known_postimages(tmp_path,monkeypatch):
    """Use a real replacement rule from both releases, not a mocked classifier."""
    root=fixture(tmp_path)
    p='examples/config/pv_tracking/conf_compact.py'
    path=root/p;path.parent.mkdir(parents=True);path.write_text('"""Old config."""\n# Compact six-feature state; actor and critic dimensions remain unchanged.\nconfig={\'observation_mode\': \'compact\'}\n')
    git(root,'add',p);git(root,'commit','-qm','config')
    monkeypatch.setattr(installer,'BASE',git(root,'rev-parse','HEAD').strip())
    monkeypatch.setattr(installer,'PINS',{p:installer.blob_sha(path.read_bytes())})
    monkeypatch.setattr(installer,'DEPENDENCIES',{})
    plan,versions=installer.expected_versions(root)
    target=p.replace('conf_compact','conf_geometry10')
    assert b"'observation_mode': 'geometry10'" in plan[target]
    assert 'v1' in versions[target] and 'v2' in versions[target]
    assert versions['mbpo/geometry10.py']['v1']!=versions['mbpo/geometry10.py']['v2']
    assert versions['GEOMETRY10_README.md']['v1']!=versions['GEOMETRY10_README.md']['v2']
    assert path.read_bytes().startswith(b'"""Old config.')
