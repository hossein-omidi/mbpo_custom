#!/usr/bin/env python3
"""Read-only preflight for SAC/MBPO repo: flag incomplete/mismatched trees before training."""
import argparse
import json
import subprocess
from pathlib import Path

REQUIRED = [
'SAC_workspace/src/agents/sac_agent.py',
'SAC_workspace/src/training/replay_buffer.py',
'SAC_workspace/src/training/trainer.py',
'SAC_workspace/src/config/config.py',
'SAC_workspace/src/evaluation/evaluation.py',
'SAC_workspace/src/evaluation/scenario_sampling.py',
'SAC_workspace/src/evaluation/stratified_statistics.py',
'SAC_workspace/src/evaluation/statistical_inference.py',
'MBPO_workspace/src/dynamics.py',
'MBPO_workspace/src/trainer.py',
'MBPO_workspace/src/evaluation.py',
'MBPO_workspace/src/replay.py',
'MBPO_workspace/workflows/02_train.py',
'MBPO_workspace/workflows/03_evaluate.py',
'MBPO_workspace/representation_config.json',
'SAC_workspace/workflows/03_train.py',
'mbpo/env/pv_tracking.py',
]
def check(root):
    root=Path(root).resolve()
    missing=[name for name in REQUIRED if not (root/name).is_file()]
    result={'repo_root':str(root),'missing_required_source_files':missing,'status':'PASS' if not missing else 'FAIL_INCOMPLETE_REPOSITORY'}
    try:
        head=subprocess.run(['git','rev-parse','HEAD'],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10,text=True)
        result['git_commit']=head.stdout.strip() if head.returncode==0 else None
        status=subprocess.run(['git','status','--porcelain'],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10,text=True)
        result['working_tree_modified_or_untracked_count']=len(status.stdout.splitlines()) if status.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):result['git_commit']=None
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo-root',default='.');p.add_argument('--output',default=None)
    a=p.parse_args();r=check(a.repo_root);print(json.dumps(r,indent=2))
    if a.output:Path(a.output).write_text(json.dumps(r,indent=2),encoding='utf-8')
    raise SystemExit(0 if r['status']=='PASS' else 2)
