"""Preview and save the exact calendar-balanced NSRDB evaluation scenarios.

No policy checkpoint, pvlib rollout, gradient update, Ray, or TensorFlow is
required. Run this before expensive evaluation to inspect year/month coverage.
"""
import argparse
from pathlib import Path
from SAC_workspace.src.config.config import load_config
from SAC_workspace.src.evaluation.scenario_sampling import build_selection,save_selection


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',default='SAC_workspace/evaluation/scenario_preview')
    p.add_argument('--days-per-month',type=int,default=3)
    p.add_argument('--selection-seed',type=int,default=42)
    p.add_argument('--exhaustive',action='store_true')
    p.add_argument('--exclude-training-csv',nargs='+',default=[],help='Training episodes.csv files to exclude (SAC and MBPO when comparing both)')
    args=p.parse_args()
    from mbpo.env.nsrdb_weather import load_scenario_manifest
    from SAC_workspace.src.evaluation.scenario_sampling import _training_seen
    manifest=load_scenario_manifest(load_config().manifest)
    try:
        excluded=_training_seen(args.exclude_training_csv) if args.exclude_training_csv else set()
        plan=build_selection(manifest,days_per_month=args.days_per_month,
                    selection_seed=args.selection_seed,exhaustive=args.exhaustive,
                    excluded_ids=excluded)
        save_selection(plan,args.outdir)
    except ValueError as exc:p.error(str(exc))
    print(f'{plan["selected_count"]} selected / {plan["population_count"]} eligible dates; '
          f'{len(excluded)} excluded training dates; {plan["year_month_strata_count"]} year-months')
    if plan['missing_year_months']:
        print('WARNING missing year-months:',', '.join(plan['missing_year_months']))
    print('Selection saved:',(Path(args.outdir)/'scenario_selection.json').resolve())
    print('Reuse with --selection-plan',str(Path(args.outdir)/'scenario_selection.json'))

if __name__=='__main__':main()
