"""Compare independent trained SAC, MBPO, and PV dual-axis references."""
import argparse
from pathlib import Path
from MBPO_workspace.src.config import load_config
from MBPO_workspace.src.evaluation import evaluate,write_outputs
from MBPO_workspace.src.plots import make_figures


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mbpo-checkpoint',required=True)
    p.add_argument('--sac-checkpoint',required=True)
    p.add_argument('--outdir',default='MBPO_workspace/evaluation/paired')
    from SAC_workspace.src.evaluation.scenario_sampling import add_selection_arguments
    add_selection_arguments(p)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--skip-poa-opt',action='store_true',help='Skip computationally costly per-step POA optimum')
    p.add_argument('--device',default='cpu')
    p.add_argument('--mbpo-training-csv',default=None)
    p.add_argument('--sac-training-csv',default=None)
    p.add_argument('--skip-advanced-suite',action='store_true',
                   help='Do not render the offline advanced PNG/PDF suite after evaluation')
    p.add_argument('--representation-config', default='MBPO_workspace/representation_config.json',
                   help='JSON profile controlling visible comparisons without deleting legacy trackers')
    args=p.parse_args()
    import json
    profile=json.loads(Path(args.representation_config).read_text())
    enabled=set(profile['active_methods'])
    poa_only=enabled=={'mbpo','sac','fixed_no_motion','poa_instant_ideal'}
    if not poa_only:
        p.error('Current verified profile supports exactly MBPO, SAC, fixed and POA oracle')
    if args.skip_poa_opt:
        p.error('POA-only representation requires POA optimization; remove --skip-poa-opt')
    cfg=load_config()
    from mbpo.env.nsrdb_weather import load_scenario_manifest
    from SAC_workspace.src.evaluation.scenario_sampling import resolve_cli_selection,save_selection
    try:
        scenarios,plan=resolve_cli_selection(args,load_scenario_manifest(cfg.sac.manifest),
            training_csvs=[x for x in (args.mbpo_training_csv,args.sac_training_csv) if x])
    except ValueError as exc:
        p.error(str(exc))
    if plan is not None:
        if args.exclude_training_days and not (args.mbpo_training_csv and args.sac_training_csv):
            p.error('Held-out comparisons require BOTH --mbpo-training-csv and --sac-training-csv')
        save_selection(plan,args.outdir)
        print(f'Selected {plan["selected_count"]} of {plan["population_count"]} dates; '
              f'protocol={plan["protocol"]}, years={plan["years"]}')
    paths=evaluate(cfg,args.mbpo_checkpoint,args.sac_checkpoint,
                   scenarios=scenarios,seeds=(args.seed,),device=args.device,include_poa=True,include_classical=False)
    stats=write_outputs(paths,args.outdir,plan=plan)
    # Export the measured actions, solar angles and irradiance now; future plots
    # can be regenerated entirely offline without reloading policy or pvlib.
    from MBPO_workspace.src.advanced_suite import export_detail
    export_detail(paths,args.outdir)
    figs=[]
    advanced={'plots': []}
    if plan is not None:
        from MBPO_workspace.src.verified_figures import render_verified
        verified=render_verified(paths,plan,Path(args.outdir)/'verified')
        figs=verified['figures']
    else:
        print('No selection plan: verified finite-population figures require --all-scenarios or --selection-plan')
    if plan is not None:
        from SAC_workspace.src.evaluation.stratified_statistics import render_statistical_report
        statistics=render_statistical_report(Path(args.outdir)/'stratified',plan,paths=paths)
        print('Stratified analyses:',len(statistics['figures']),'figures in',Path(args.outdir,'stratified').resolve())
    if not args.skip_advanced_suite:
        from MBPO_workspace.src.advanced_suite import render_suite
        advanced=render_suite(args.outdir,mbpo_training_csv=args.mbpo_training_csv,
                              sac_training_csv=args.sac_training_csv)
        print('Advanced suite:',len(advanced['figures']),'figures; skipped:',
              len(advanced['skipped']), 'under',Path(args.outdir,'advanced_suite').resolve())
    for k,v in stats['methods'].items():
        print(f'{k:26s} net={v["mean_net_kwh"]} gross={v["mean_gross_kwh"]:.6f} kWh')
    print('Paired report:',Path(args.outdir).resolve(),'; figures:',len(figs),'; additional paper figures:',len(advanced['plots']))

if __name__=='__main__':main()
