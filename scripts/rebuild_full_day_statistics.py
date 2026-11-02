#!/usr/bin/env python3
"""Regenerate 195-step stratified + advanced reporting from saved real trajectories.

This is NOT an evaluation, training or physics simulator. It only processes
previously exported source data, validates the exact selection and UTC pairing,
and writes derived reports. It never edits checkpoints, source data or manifests.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

METHODS = {'mbpo', 'sac', 'fixed_no_motion', 'poa_instant_ideal'}
REQUIRED = {'scenario_id','seed','method','step','utc_iso','power_w','energy_kwh',
            'movement_cost_kwh','tilt_deg','azimuth_deg',
            'actual_delta_tilt_deg','actual_delta_azimuth_deg',
            'solar_zenith_deg','solar_azimuth_deg','dni_wm2','dhi_wm2','ghi_wm2'}


def numeric(row, field, *, required=True):
    value = row.get(field, '')
    if value in ('', None, 'None', 'nan'):
        if required: raise ValueError(f'Missing recorded physical field {field}')
        return float('nan')
    result=float(value)
    if not __import__('math').isfinite(result):
        if required: raise ValueError(f'Nonfinite physical field {field}')
        return float('nan')
    return result


def observed_paths(file, expected_ids):
    grouped=defaultdict(list)
    with Path(file).open(newline='', encoding='utf-8') as fp:
        reader=csv.DictReader(fp)
        fields=set(reader.fieldnames or ())
        missing=REQUIRED - fields
        if missing: raise ValueError(f'Detailed trajectory CSV missing columns: {sorted(missing)}')
        if 'weather_condition' not in fields:
            raise ValueError('Detailed trajectory CSV missing weather_condition')
        for row in reader:
            sid=str(row['scenario_id']);method=row['method']
            if method not in METHODS: raise ValueError(f'Unexpected method in detailed CSV: {method}')
            if sid not in expected_ids: raise ValueError(f'Unselected day in saved trajectories: {sid}')
            seed=int(row['seed']);step=int(row['step'])
            if not 1 <= step <= 195:
                raise ValueError(f'Invalid recorded episode step {step} for {sid}/{method}')
            grouped[(sid,seed,method)].append(row)
    if not grouped:raise ValueError('Detailed trajectory CSV is empty')
    selected={sid for sid,_,_ in grouped}
    if selected != expected_ids:
        raise ValueError(f'Saved trajectories do not contain the selected scenario set; '
                         f'missing={sorted(expected_ids-selected)[:6]}')
    keys=defaultdict(set)
    for sid,seed,method in grouped:keys[(sid,seed)].add(method)
    if any(v!=METHODS for v in keys.values()):
        raise ValueError('Saved data is not a complete 4-method paired evaluation')
    if len({tuple(sorted({seed for sid,seed in keys if sid==day})) for day in selected})!=1:
        raise ValueError('Evaluation seeds differ across saved scenario days')
    paths=[]
    for (sid,seed,method),rows in sorted(grouped.items()):
        if len(rows)!=195 or [int(r['step']) for r in rows]!=list(range(1,196)):
            raise ValueError(f'Incomplete, reordered or duplicated saved trajectory: {sid}/{method}')
        first = datetime.fromisoformat(sid+'T08:45:00').replace(tzinfo=timezone.utc)
        for ix,row in enumerate(rows,1):
            observed=datetime.fromisoformat(row['utc_iso'])
            expected=first+timedelta(minutes=ix*5)
            if observed != expected or observed.utcoffset()!=timedelta(0):
                raise ValueError(f'Incorrect saved post-action UTC time: {sid}/{method}/{ix}')
        vals={name:[numeric(r,field,required=True) for r in rows] for name,field in (
            ('power','power_w'),('energy','energy_kwh'),('tilt','tilt_deg'),
            ('azimuth','azimuth_deg'))}
        for step,(power,energy) in enumerate(zip(vals['power'],vals['energy']),1):
            if abs(energy-power/12000.0)>max(1e-6,abs(energy)*1e-6):
                raise ValueError(f'Incorrect recorded 5-min energy/power identity: {sid}/{method}/{step}')
        ideal = method=='poa_instant_ideal'
        costs=[numeric(r,'movement_cost_kwh',required=True) for r in rows]
        tilt_motion=[numeric(r,'actual_delta_tilt_deg',required=not ideal) for r in rows]
        az_motion=[numeric(r,'actual_delta_azimuth_deg',required=not ideal) for r in rows]
        if any(x<0 for x in costs):raise ValueError('Negative physical movement cost')
        infos=[]
        for r in rows:
            physical={'timestamp_utc_iso':r['utc_iso'],
                      'weather_condition':r.get('weather_condition') or '',
                      'solar_zenith_deg':numeric(r,'solar_zenith_deg'),
                      'solar_azimuth_deg':numeric(r,'solar_azimuth_deg'),
                      'dni_wm2':numeric(r,'dni_wm2'),
                      'dhi_wm2':numeric(r,'dhi_wm2'),
                      'ghi_wm2':numeric(r,'ghi_wm2')}
            infos.append(physical)
        paths.append({
            'scenario_id':sid,'seed':seed,'method':method,
            'infos':infos,**vals,
            'delta_tilt_deg':tilt_motion,'delta_azimuth_deg':az_motion,
            'net_energy_kwh':None if ideal else sum(e-c for e,c in zip(vals['energy'],costs)),
            'gross_energy_kwh':sum(vals['energy']),
            'movement_cost_kwh':None if ideal else sum(costs),
        })
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir',type=Path,required=True)
    p.add_argument('--skip-advanced',action='store_true')
    args=p.parse_args()
    folder=args.eval_dir
    for name in ('scenario_selection.json','summary.json','trajectories_detailed.csv'):
        if not (folder/name).is_file():
            p.error(f'Saved evaluation is incomplete: missing {folder/name}')
    old=json.loads((folder/'scenario_selection.json').read_text(encoding='utf-8'))
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if not old.get('excluded_scenario_ids') or summary.get('independent_training_day_exclusion') is not True:
        p.error('Saved output is not verified as an excluded-training-date selection. '
                'For a held-out scientific result, rerun the correctly flagged evaluator.')
    if old.get('protocol') not in ('calendar_stratified_utc_daytime', 'complete_manifest'):
        p.error('Saved evaluation uses an unsupported sampling protocol')
    from SAC_workspace.src.config.config import load_config
    from mbpo.env.nsrdb_weather import load_scenario_manifest
    from SAC_workspace.src.evaluation.scenario_sampling import build_selection, validate_selection
    cfg=load_config(power_model='pvwatts_sapm',wind_reference_height_m=2,
                    gamma_pdc=-0.004,sapm_mount='open_rack_glass_polymer',
                    wind_powerlaw_exponent=0.14)
    manifest=load_scenario_manifest(cfg.manifest)
    new=build_selection(manifest,
                        days_per_month=int(old['days_per_month']),
                        selection_seed=int(old['selection_seed']),
                        exhaustive=old['protocol']=='complete_manifest',
                        excluded_ids=old['excluded_scenario_ids'])
    if (new['selected_scenario_ids']!=old.get('selected_scenario_ids')
            or new['strata']!=old.get('strata')
            or new['population_count']!=old.get('population_count')):
        p.error('Saved scenario plan does not reproduce under the current manifest and exclusions')
    validate_selection(new,manifest)
    paths=observed_paths(folder/'trajectories_detailed.csv',set(new['selected_scenario_ids']))
    # Original saved sampling plan remains unchanged for reproducible provenance.
    (folder/'scenario_selection_full_day_reverified.json').write_text(
        json.dumps(new,indent=2),encoding='utf-8')
    from SAC_workspace.src.evaluation.stratified_statistics import render_statistical_report
    stats=render_statistical_report(folder/'stratified',new,paths=paths)
    print('Rebuilt stratified statistics from existing real-physics CSV:',
          len(paths),'complete trajectories',len(stats['figures']),'figures')
    if not args.skip_advanced:
        from MBPO_workspace.src.advanced_suite import render_suite
        result=render_suite(str(folder))
        print('Advanced figures:',len(result['figures']))
    print('No training, model, NSRDB or original trajectory files were modified.')


if __name__=='__main__':
    main()
