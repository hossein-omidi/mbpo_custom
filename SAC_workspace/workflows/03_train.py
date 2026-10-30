"""Train using ONLY the existing pvlib+NSRDB environment transitions."""
import argparse
from SAC_workspace.src.config.config import ROOT, load_config
from SAC_workspace.src.training.trainer import SACTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default=str(ROOT / 'SAC_workspace/runs/sac_seed0'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=None, help='Final absolute epoch, not extra episodes')
    p.add_argument('--device', default='cpu')
    p.add_argument('--resume', default=None, help='Trusted checkpoint .pt from this workspace')
    # Single electrical backend selection shared by SAC and MBPO.
    p.add_argument('--power-model',choices=('legacy','pvwatts_sapm'),default='legacy')
    p.add_argument('--wind-reference-height-m',type=float,default=2.0)
    p.add_argument('--gamma-pdc',type=float,default=-0.004)
    p.add_argument('--sapm-mount',default='open_rack_glass_polymer')
    p.add_argument('--wind-powerlaw-exponent',type=float,default=0.14)
    args = p.parse_args()
    power_kwargs=dict(power_model=args.power_model,
        wind_reference_height_m=args.wind_reference_height_m,
        gamma_pdc=args.gamma_pdc, sapm_mount=args.sapm_mount,
        wind_powerlaw_exponent=args.wind_powerlaw_exponent)
    cfg = load_config(**power_kwargs)
    trainer = SACTrainer(cfg, args.run_dir, seed=args.seed, device=args.device, resume=args.resume)
    trainer.train(epochs=args.epochs)

if __name__ == '__main__': main()
