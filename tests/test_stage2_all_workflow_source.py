"""Every workflow source rewrite must generate syntactically valid Python.

Regression for malformed MBPO 01_setup_config.py path concatenation in v3.
"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('stage2_workflow_patcher', Path(__file__).resolve().parents[1]/'scripts'/'apply_pvwatts_stage2.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

FIXTURES = {
'SAC_workspace/workflows/01_setup_config.py': '''from pathlib import Path
from SAC_workspace.src.config.config import ROOT, load_config, write_snapshot

def main():
    c = load_config(require_data=True)
    dest = ROOT / 'SAC_workspace' / 'runs' / 'reference_config.json'
    write_snapshot(c,dest)
''',
'SAC_workspace/workflows/02_environment_replay.py': '''import numpy as np
from SAC_workspace.src.config.config import load_config

def main():
    config = load_config()
''',
'SAC_workspace/workflows/03_train.py': '''import argparse
from SAC_workspace.src.config.config import load_config

def main():
    p=argparse.ArgumentParser()
    args = p.parse_args()
    cfg = load_config()
''',
'SAC_workspace/workflows/04_evaluate.py': '''import argparse
from pathlib import Path
from SAC_workspace.src.config.config import load_config

def main():
    ap=argparse.ArgumentParser()
    args=ap.parse_args()
    cfg=load_config()
    paths=evaluate(cfg,args.checkpoint,scenario_ids=scenario_ids,seeds=args.seeds,device=args.device,include_classical=False)
    Path(args.outdir).mkdir(parents=True,exist_ok=True)
''',
'MBPO_workspace/workflows/01_setup_config.py': '''import json
from pathlib import Path
from MBPO_workspace.src.config import load_config
from SAC_workspace.src.config.config import ROOT

def main():
    c=load_config()
    p=ROOT/'MBPO_workspace/runs/reference_config.json'
    p.parent.mkdir(parents=True,exist_ok=True)
''',
'MBPO_workspace/workflows/02_train.py': '''import argparse
from MBPO_workspace.src.config import load_config

def main():
    p=argparse.ArgumentParser()
    args=p.parse_args()
    overrides={}
    c=load_config(model_overrides=overrides)
''',
'MBPO_workspace/workflows/03_evaluate.py': '''import argparse
from MBPO_workspace.src.config import load_config

def main():
    p=argparse.ArgumentParser()
    args=p.parse_args()
    cfg=load_config()
''',
}


def test_complete_workflow_transforms_compile():
    paths = [x for x in p.TARGETS if '/workflows/' in x]
    assert set(paths) == set(FIXTURES)
    for path in paths:
        result = p.patch_workflow(FIXTURES[path], path)
        compile(result, path, 'exec')
        assert 'add_power_cli_arguments(' in result
        assert 'power_cli_kwargs(args)' in result
        if path.endswith('MBPO_workspace/workflows/01_setup_config.py'):
            assert "ROOT/'MBPO_workspace'/'runs'/" in result
