"""Paired SAC/MBPO/fixed/albedo-aware POA evaluation on real NSRDB rows.

No model synthetic transition is used in evaluation. Both agents, fixed pose,
and the unconstrained oracle see the identical timestep-specific measured
albedo, irradiance and SAPM temperature/wind physics.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from MBPO_workspace.src.config import load_config
from MBPO_workspace.src.evaluation import evaluate,write_outputs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mbpo-checkpoint',required=True)
    p.add_argument('--sac-checkpoint',required=True)
    p.add_argument('--outdir',default='MBPO_workspace/evaluation/stage_a_paired')
    from SAC_workspace.src.evaluation.scenario_sampling import add_selection_arguments
    add_selection_arguments(p)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--skip-poa-opt',action='store_true')
    p.add_argument('--device',default='cpu')
    p.add_argument('--mbpo-training-csv',default=None)
    p.add_argument('--sac-training-csv',default=None)
    p.add_argument('--skip-advanced-suite',action='store_true')
    p.add_argument('--representation-config',default='MBPO_workspace/representation_config.json')
    p.add_argument('--power-model',choices=('legacy','pvwatts_sapm'),default='pvwatts_sapm')
    p.add_argument('--wind-reference-height-m',type=float,default=2.0)
    p.add_argument('--gamma-pdc',type=float,default=-0.004)
    p.add_argument('--sapm-mount',default='open_rack_glass_polymer')
    p.add_argument('--wind-powerlaw-exponent',type=float,default=0.14)
    args=p.parse_args()
    for flag, val in (('--sac-training-csv',args.sac_training_csv),
                      ('--mbpo-training-csv',args.mbpo_training_csv)):
        if val is not None and not Path(val).is_file():
            p.error(f'{flag}: file not found: {val}')
    if args.exclude_training_days and not (args.mbpo_training_csv and args.sac_training_csv):
        p.error('Independent held-out selection requires BOTH training histories')
    profile=json.loads(Path(args.representation_config).read_text(encoding='utf-8'))
    if set(profile.get('active_methods',[]))!={'mbpo','sac','fixed_no_motion','poa_instant_ideal'}:
        p.error('Stage A supports the current four-method verified profile')
    if args.skip_poa_opt:
        p.error('The required POA upper reference cannot be skipped')
    cfg=load_config(power_model=args.power_model,
        wind_reference_height_m=args.wind_reference_height_m,
        gamma_pdc=args.gamma_pdc,sapm_mount=args.sapm_mount,
        wind_powerlaw_exponent=args.wind_powerlaw_exponent)
    from mbpo.env.nsrdb_weather import load_scenario_manifest
    from SAC_workspace.src.evaluation.scenario_sampling import resolve_cli_selection,save_selection
    scenarios,plan=resolve_cli_selection(
        args,load_scenario_manifest(cfg.sac.manifest),
        training_csvs=[x for x in (args.mbpo_training_csv,args.sac_training_csv) if x])
    if plan is not None:
        save_selection(plan,args.outdir)
        print('Selected',plan['selected_count'],'of',plan['population_count'],
              'days, protocol=',plan['protocol'])
    # Old MBPO _check_physics compares simulator source hashes but does not
    # compare annual NSRDB file hashes. Enforce the stronger Stage A dataset
    # contract for *both* trusted checkpoints before any expensive rollout.
    import torch
    from SAC_workspace.src.environment.environment import SACEnvironment
    check_env=SACEnvironment(cfg.sac.eval_env)
    try:
        expected=check_env.resolved_mdp
    finally:
        check_env.close()
    for ckpt_path in (args.sac_checkpoint,args.mbpo_checkpoint):
        ck=torch.load(ckpt_path,map_location='cpu',weights_only=False)
        saved=ck.get('resolved_mdp') or {}
        for key in ('nsrdb_manifest_sha256','nsrdb_year_file_sha256',
                    'nsrdb_albedo_contract','albedo_model','physics_revision'):
            if key not in saved or saved[key]!=expected.get(key):
                raise ValueError(f'Checkpoint NSRDB/physical fingerprint differs: {ckpt_path}: {key}')
    # The original evaluator runs only *real* SAC/MBPO/fixed trajectories;
    # replace ONLY the oracle with the recorded-albedo-aware analytical POA
    # solution, rather than silently using its old 0.25 fixed-albedo oracle.
    paths=evaluate(cfg,args.mbpo_checkpoint,args.sac_checkpoint,
                   scenarios=scenarios,seeds=(args.seed,),device=args.device,
                   include_poa=False,include_classical=False)
    from MBPO_workspace.src.stage_a_reference import ideal_instantaneous_poa
    from mbpo.env.baseline_contract import validate_paired_reference_group
    env=SACEnvironment(cfg.sac.eval_env)
    try:
        fixed=[row for row in paths if row['method']=='fixed_no_motion']
        oracles=[ideal_instantaneous_poa(cfg.sac,f,env=env) for f in fixed]
    finally:
        env.close()
    paths.extend(oracles)
    bypair=defaultdict(dict)
    for row in paths:
        key=(row['scenario_id'],int(row['seed']))
        if row['method'] in bypair[key]:
            raise ValueError(f'Duplicate evaluation method: {key}/{row["method"]}')
        bypair[key][row['method']]=row
    for key,group in bypair.items():
        validate_paired_reference_group(group,require_oracle=True)
        fixed_row=group['fixed_no_motion']
        fixed_albedo=[float(i['surface_albedo']) for i in fixed_row['infos']]
        for method,row in group.items():
            actual=[i.get('surface_albedo') for i in row['infos']]
            if len(actual)!=len(fixed_albedo) or any(
                    v is None or abs(float(v)-expected)>1e-10
                    for v,expected in zip(actual,fixed_albedo)):
                raise RuntimeError(f'Inconsistent NSRDB albedo for {key} / {method}')
    stats=write_outputs(paths,args.outdir,plan=plan)
    # Explicit gross-vs-net meaning: no simulated movement power is included
    # when the training MDP has movement_penalty=0.
    penalty=float(env.unwrapped.movement_penalty)
    out=Path(args.outdir)
    # The original summary writer already cleans nonfinite bootstrap values
    # to JSON null. Preserve that behavior rather than reserializing raw stats.
    report_path=out/'summary.json'
    report=json.loads(report_path.read_text(encoding='utf-8'))
    report['energy_objective']=(
        'gross_DC_kWh_no_actuator_consumption' if penalty==0 else
        'DC_kWh_less_modelled_actuator_consumption')
    report['albedo_contract']='recorded_NSRDB_surface_albedo_per_native_5min_sample'
    report['oracle_physics']='isotropic_POA_with_recorded_albedo_monotone_PVWatts_DC'
    report['independent_training_day_exclusion']=bool(args.exclude_training_days)
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True,
                              allow_nan=False),encoding='utf-8')
    from MBPO_workspace.src.advanced_suite import export_detail
    export_detail(paths,args.outdir)
    figs=[];advanced={'figures':[],'skipped':[]}
    if plan is not None:
        from MBPO_workspace.src.verified_figures import render_verified
        verified=render_verified(paths,plan,out/'verified')
        figs=verified['figures']
        from SAC_workspace.src.evaluation.stratified_statistics import render_statistical_report
        s=render_statistical_report(out/'stratified',plan,paths=paths)
        print('Statistical report figures:',len(s['figures']))
    if not args.skip_advanced_suite:
        from MBPO_workspace.src.advanced_suite import render_suite
        advanced=render_suite(args.outdir,mbpo_training_csv=args.mbpo_training_csv,
                               sac_training_csv=args.sac_training_csv)
    for method,data in stats['methods'].items():
        print(f'{method:26s} net={data["mean_net_kwh"]} gross={data["mean_gross_kwh"]:.6f} kWh')
    print('Stage A paired report:',out.resolve(),
          'verified figures:',len(figs),'advanced:',len(advanced['figures']))


if __name__=='__main__':
    main()
