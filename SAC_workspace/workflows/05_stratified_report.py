"""Regenerate calendar-stratified statistics and PNG/PDF without rerunning PVLIB.

python -m SAC_workspace.workflows.05_stratified_report --eval-dir SAC_workspace/evaluation/seasonal
python -m SAC_workspace.workflows.05_stratified_report --eval-dir MBPO_workspace/evaluation/paired
"""
import argparse
from pathlib import Path
from SAC_workspace.src.evaluation.stratified_statistics import render_statistical_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir',required=True)
    p.add_argument('--bootstrap-seed',type=int,default=123)
    p.add_argument('--bootstrap-draws',type=int,default=1000)
    p.add_argument('--no-plots',action='store_true')
    args=p.parse_args()
    root=Path(args.eval_dir)
    if args.bootstrap_draws<100:p.error('At least 100 bootstrap draws are required')
    if not (root/'scenario_selection.json').is_file():
        p.error('No scenario_selection.json in evaluation directory; evaluate with --all-scenarios first')
    if not (root/'stratified'/'episode_features.csv').is_file():
        p.error('No saved episode_features.csv; run the stratified evaluation once')
    result=render_statistical_report(root/'stratified',bootstrap_seed=args.bootstrap_seed,
                                    n_boot=args.bootstrap_draws,plots=not args.no_plots)
    print('Rebuilt sampled-scenario statistics:',root.resolve(),'| selected:',result['selected_scenarios'],
          '| figures:',len(result['figures']))

if __name__=='__main__':main()
