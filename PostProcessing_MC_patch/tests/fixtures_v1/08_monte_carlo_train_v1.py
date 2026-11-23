"""Train SAC or MBPO on complete correlated-uncertainty NSRDB trajectories.

Nothing is sampled per action: each episode is one sampled historical day and
one independently drawn physically constrained *whole-day* weather realization.
SAC/MBPO actors, critics, neural ensemble, buffers and optimizer unchanged.
Use distinct run directories. A validation realization is frozen (ID 0) for
comparison against deterministic baselines at every periodic checkpoint.

    python -m MBPO_workspace.workflows.08_monte_carlo_train --algorithm sac \
        --uncertainty-spec path/to/assumptions.json --run-dir ... --seed 0
"""
import argparse
import json
from pathlib import Path
from mbpo.env.trajectory_uncertainty import normalized_spec,spec_fingerprint


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm',required=True,choices=('sac','mbpo'))
    parser.add_argument('--uncertainty-spec',required=True)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=None,help='Absolute final epoch (not extra episodes)')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--resume',default=None,help='Trusted checkpoint created by this exact configuration')
    parser.add_argument('--model-epochs',type=int,default=None)
    parser.add_argument('--model-max-train-samples',type=int,default=None)
    args=parser.parse_args(argv)
    spec=normalized_spec(json.loads(Path(args.uncertainty_spec).read_text(encoding='utf-8')))
    if not spec['enabled']:
        parser.error('Uncertainty-augmented training requires enabled=true in assumptions')
    print('Weather uncertainty scenario: hypothetical; NOT calibrated NSRDB measurement accuracy')
    print('Parameter SHA256:',spec_fingerprint(spec))
    if args.algorithm=='sac':
        if args.model_epochs is not None or args.model_max_train_samples is not None:
            parser.error('MBPO-specific model options cannot be used for SAC')
        from SAC_workspace.src.config.config import load_config
        from SAC_workspace.src.training.trainer import SACTrainer
        cfg=load_config(weather_uncertainty=spec)
        trainer=SACTrainer(cfg,args.run_dir,seed=args.seed,device=args.device,resume=args.resume)
    else:
        from MBPO_workspace.src.config import load_config
        from MBPO_workspace.src.trainer import MBPOTrainer
        overrides={}
        if args.model_epochs is not None:overrides['model_epochs']=args.model_epochs
        if args.model_max_train_samples is not None:overrides['model_max_train_samples']=args.model_max_train_samples
        cfg=load_config(weather_uncertainty=spec,model_overrides=overrides)
        trainer=MBPOTrainer(cfg,args.run_dir,seed=args.seed,device=args.device,resume=args.resume)
    trainer.train(epochs=args.epochs)

if __name__=='__main__':main()
