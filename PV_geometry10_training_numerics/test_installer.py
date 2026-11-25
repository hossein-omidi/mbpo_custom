import pytest
import install as mod


def test_installer_all_files_checked_before_write(tmp_path,monkeypatch):
    (tmp_path/'mbpo').mkdir();(tmp_path/'mbpo/geometry10.py').write_text('pass\n')
    (tmp_path/'geometry10_tests').mkdir()
    names=['a.py','b.py'];before=b'a=1\n';after=b'a=2\n'
    changes={n:(mod.blob(before),mod.blob(after),'a=1','a=2') for n in names}
    monkeypatch.setattr(mod,'CHANGES',changes)
    for n in names:(tmp_path/n).write_bytes(before)
    mod.install(tmp_path,False)
    assert all((tmp_path/n).read_bytes()==before for n in names)
    (tmp_path/'b.py').write_bytes(b'a=3\n')
    with pytest.raises(ValueError,match='unexpected Git blob'):mod.install(tmp_path,True)
    assert (tmp_path/'a.py').read_bytes()==before
    (tmp_path/'b.py').write_bytes(before)
    mod.install(tmp_path,True)
    assert all((tmp_path/n).read_bytes()==after for n in names)
    assert (tmp_path/mod.TEST).is_file()
    mod.install(tmp_path,True)
    assert all((tmp_path/n).read_bytes()==after for n in names)
