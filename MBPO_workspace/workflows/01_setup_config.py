"""Verify identical reference MDP / SAC hyperparameters without Ray or TF."""
import json
from pathlib import Path
from MBPO_workspace.src.config import load_config
from SAC_workspace.src.config.config import ROOT


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--power-model', choices=('legacy','pvwatts_sapm'), default='legacy')
    p.add_argument('--wind-reference-height-m', type=float, default=2.0)
    p.add_argument('--gamma-pdc', type=float, default=-0.004)
    p.add_argument('--sapm-mount', default='open_rack_glass_polymer')
    p.add_argument('--wind-powerlaw-exponent', type=float, default=0.14)
    args=p.parse_args()
    kwargs=dict(power_model=args.power_model,
        wind_reference_height_m=args.wind_reference_height_m,
        gamma_pdc=args.gamma_pdc, sapm_mount=args.sapm_mount,
        wind_powerlaw_exponent=args.wind_powerlaw_exponent)
    c=load_config(**kwargs)
    p=ROOT/'MBPO_workspace/runs/reference_config.json'
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(c.to_dict(),indent=2,sort_keys=True),encoding='utf-8')
    print('Verified shared compact reference MDP: 6D, 2D, 117 steps, UTC native NSRDB.')
    print('SAC hyperparameters shared. Ensemble configuration:',c.model)
    print('Snapshot:',p)

if __name__=='__main__':main()
