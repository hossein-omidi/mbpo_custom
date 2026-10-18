"""Explain and regenerate weighted scientific statistics from existing outputs.
For source trajectories, use the evaluation workflow so all metadata are checked.
"""
import argparse, json
from pathlib import Path
from SAC_workspace.src.evaluation.stratified_statistics import load_episode_features
from SAC_workspace.src.evaluation.statistical_inference import day_descriptives

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--eval-dir',required=True)
    args=ap.parse_args();root=Path(args.eval_dir)
    path=root/'stratified'/'episode_features.csv'
    if not path.exists():ap.error('Missing saved episode_features.csv; rerun evaluation with a selection plan')
    rows=load_episode_features(root/'stratified')
    methods=sorted(set(r['method'] for r in rows if r['method'] in ('mbpo','sac','fixed_no_motion','poa_instant_ideal')))
    for m in methods:
        metric='gross_energy_kwh' if m=='poa_instant_ideal' else 'net_energy_kwh'
        vals=[r[metric] for r in rows if r['method']==m]
        print(m,metric,day_descriptives(vals))
    print('Across-day SD is descriptive only. No inferential CI from one date per stratum.')
if __name__=='__main__':main()
