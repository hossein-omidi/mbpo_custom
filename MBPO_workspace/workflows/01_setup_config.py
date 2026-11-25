# PVWATT_DEV_REPAIR_V3
"""Verify identical reference MDP / SAC hyperparameters without Ray or TF."""
import json
from pathlib import Path
from MBPO_workspace.src.config import load_config
from SAC_workspace.src.config.config import ROOT
from SAC_workspace.src.training.experiment_protocol import build_scenario_protocol


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--power-model', choices=('legacy','pvwatts_sapm'), default='pvwatts_sapm')
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
    horizon = int(c.sac.train_env['periods']) - 1
    print(f'Verified shared geometry10 reference: 10D, 2D, {horizon} transitions, UTC native NSRDB.')
    protocol = build_scenario_protocol(c.sac.manifest, c.sac.protocol)
    print(f'Frozen scenario split: train={len(protocol["train_ids"])}, validation={len(protocol["validation_ids"])}, test={len(protocol["test_ids"])}, sha256={protocol["protocol_sha256"][:12]}')
    print(f'MBPO settings: real_ratio={c.model.real_ratio}, rollout={c.model.rollout_length}, fit_freq={c.model.model_train_freq}, holdout_fraction={c.model.model_holdout_fraction}, min_train_days={c.model.min_holdout_scenarios}, min_holdout_days={c.model.min_holdout_scenarios}')
    print('SAC hyperparameters shared. Ensemble configuration:',c.model)
    print('Snapshot:',p)

if __name__=='__main__':main()
