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
    args = p.parse_args()
    cfg = load_config()
    trainer = SACTrainer(cfg, args.run_dir, seed=args.seed, device=args.device, resume=args.resume)
    trainer.train(epochs=args.epochs)

if __name__ == '__main__': main()
