"""Standalone Torch MBPO: short imaginary transitions + shared SAC optimizer."""
import argparse
from SAC_workspace.src.config.config import ROOT
from MBPO_workspace.src.config import load_config
from MBPO_workspace.src.trainer import MBPOTrainer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',default=str(ROOT/'MBPO_workspace/runs/mbpo_seed0'))
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--epochs',type=int,default=None,help='Absolute final epoch')
    p.add_argument('--device',default='cpu')
    p.add_argument('--resume',default=None,help='Trusted MBPO checkpoint')
    p.add_argument('--model-epochs',type=int,default=None)
    p.add_argument('--model-max-train-samples',type=int,default=None)
    args=p.parse_args()
    overrides={}
    if args.model_epochs is not None:overrides['model_epochs']=args.model_epochs
    if args.model_max_train_samples is not None:overrides['model_max_train_samples']=args.model_max_train_samples
    c=load_config(model_overrides=overrides)
    t=MBPOTrainer(c,args.run_dir,seed=args.seed,device=args.device,resume=args.resume)
    t.train(epochs=args.epochs)

if __name__=='__main__':main()
