# PVWATT_DEV_REPAIR_V3
"""Run from repository root: python -m SAC_workspace.workflows.01_setup_config"""
from pathlib import Path
from SAC_workspace.src.config.config import ROOT, load_config, write_snapshot
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
    c = load_config(require_data=True, **kwargs)
    dest = ROOT / 'SAC_workspace' / 'runs' / 'reference_config.json'
    write_snapshot(c, dest)
    print(f'VALIDATED: compact 6D, NSRDB manifest and reference files. Snapshot: {dest}')
    horizon = int(c.train_env['periods']) - 1
    print(f'{horizon} transitions | eval every {c.protocol["evaluation_interval_episodes"]} episodes | gamma={c.algo["gamma"]} | reward_scale={c.algo["reward_scale"]}')
    protocol = build_scenario_protocol(c.manifest, c.protocol)
    print(f'Frozen scenario split: train={len(protocol["train_ids"])}, validation={len(protocol["validation_ids"])}, test={len(protocol["test_ids"])}, sha256={protocol["protocol_sha256"][:12]}')

if __name__ == '__main__':
    main()
