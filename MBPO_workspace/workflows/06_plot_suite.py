"""Render complete modern paper-style plot suite from saved evaluation artifacts."""
import argparse
from pathlib import Path
from MBPO_workspace.src.advanced_suite import render_suite

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir',required=True)
    p.add_argument('--outdir',default=None)
    p.add_argument('--mbpo-training-csv',default=None)
    p.add_argument('--sac-training-csv',default=None)
    a=p.parse_args()
    for flag,value in (('--mbpo-training-csv',a.mbpo_training_csv),
                       ('--sac-training-csv',a.sac_training_csv)):
        if value is not None and not Path(value).is_file():
            p.error(f'{flag}: file not found: {value}')
    result=render_suite(a.eval_dir,a.outdir,mbpo_training_csv=a.mbpo_training_csv,
                        sac_training_csv=a.sac_training_csv)
    print(f'Generated {len(result["figures"])} figures from {result["sampled_unique_dates"]} distinct days.')
    print('Source:',result['source'])
    print('Missing-data skips:',len(result['skipped']))
    for reason in result['skipped']:print('  SKIP:',reason)

if __name__=='__main__':main()
