"""Run from repository root: python -m SAC_workspace.workflows.01_setup_config"""
from pathlib import Path
from SAC_workspace.src.config.config import ROOT, load_config, write_snapshot


def main():
    c = load_config(require_data=True)
    dest = ROOT / 'SAC_workspace' / 'runs' / 'reference_config.json'
    write_snapshot(c, dest)
    print(f'VALIDATED: compact 6D, NSRDB manifest and reference files. Snapshot: {dest}')
    print(f'117 steps | gamma={c.algo["gamma"]} | reward_scale={c.algo["reward_scale"]}')

if __name__ == '__main__':
    main()
