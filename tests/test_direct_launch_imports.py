"""Prove a direct Python script invocation can import its sibling repo packages.

This is a real child process with an unrelated current directory, not an AST
check or a PYTHONPATH workaround. The fake SAC_workspace is deliberately only
available in the script's parent repo, just as in the user's checkout.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('filename',[
    'sac_checkpoint_config_audit.py',
    'sac_policy_forensic.py',
])
def test_direct_script_bootstraps_repo_imports(tmp_path, filename):
    project = tmp_path / 'repo'
    script_dir = project / 'scripts'
    script_dir.mkdir(parents=True)
    shutil.copy2(SRC_ROOT / 'scripts' / filename, script_dir / filename)
    pkg = project / 'SAC_workspace' / 'src' / 'config'
    pkg.mkdir(parents=True)
    (pkg / 'config.py').write_text('ROOT_MARKER = "correct-repo"\n', encoding='utf-8')
    # runpy simulates execution of the script while avoiding its argparse main.
    # Then an absolute package import must resolve from the script's parent.
    code = '''import runpy
runpy.run_path(%r, run_name="test_import_only")
from SAC_workspace.src.config.config import ROOT_MARKER
print(ROOT_MARKER)
''' % str(script_dir / filename)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run(
        [sys.executable, '-c', code], cwd=tmp_path,
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'correct-repo'

@pytest.mark.parametrize('filename',[
    'sac_checkpoint_config_audit.py',
    'sac_policy_forensic.py',
])
def test_real_direct_file_help_works_with_unrelated_cwd(tmp_path, filename):
    script = SRC_ROOT / 'scripts' / filename
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run(
        [sys.executable, str(script), '--help'], cwd=tmp_path,
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert '--eval-dir' in result.stdout
