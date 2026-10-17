"""Regenerate rich solar-tracking plots from saved CSV, without simulator/replay.

Usage:
  python -m MBPO_workspace.workflows.04_advanced_report \\
    --eval-dir MBPO_workspace/evaluation/paired
"""
import argparse
from pathlib import Path
from MBPO_workspace.src.paper_reporting import render_from_csv

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir',required=True,help='Directory from 03_evaluate')
    parser.add_argument('--rich-csv',default=None,help='Override rich trajectory CSV')
    parser.add_argument('--outdir',default=None,help='Regenerated figure destination')
    a=parser.parse_args()
    root=Path(a.eval_dir)
    input_path=Path(a.rich_csv) if a.rich_csv else root/'advanced'/'trajectories_rich.csv'
    out=Path(a.outdir) if a.outdir else root/'advanced'
    if not input_path.is_file():
        parser.error(f'{input_path} is missing. Run 03_evaluate once with the complementary patch.')
    report=render_from_csv(input_path,out)
    print('Advanced reports:',out.resolve(),'| paired episodes:',report['episode_pairs'],
          '| plots:',len(report['plots']))

if __name__=='__main__':main()
