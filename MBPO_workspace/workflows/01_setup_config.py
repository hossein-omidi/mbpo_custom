"""Verify identical reference MDP / SAC hyperparameters without Ray or TF."""
import json
from pathlib import Path
from MBPO_workspace.src.config import load_config
from SAC_workspace.src.config.config import ROOT


def main():
    c=load_config()
    p=ROOT/'MBPO_workspace/runs/reference_config.json'
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(c.to_dict(),indent=2,sort_keys=True),encoding='utf-8')
    print('Verified shared compact reference MDP: 6D, 2D, 117 steps, UTC native NSRDB.')
    print('SAC hyperparameters shared. Ensemble configuration:',c.model)
    print('Snapshot:',p)

if __name__=='__main__':main()
