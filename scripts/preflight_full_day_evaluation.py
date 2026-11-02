#!/usr/bin/env python3
"""Read-only preflight for the paired 195-transition Stage-A evaluation.

Uses existing run metadata and the actual NSRDB manifest; no training, no
checkpoint writes, no weather edits and no evaluation output writes.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

EXPECTED = {'start_time': '08:45', 'periods': 196, 'freq': '5min', 'horizon': 195, 'tz': 'UTC'}
CHECK_FIELDS = ('start_time','periods','freq','horizon','tz','power_model',
                'nsrdb_manifest_sha256','nsrdb_year_file_sha256',
                'power_source_sha256','albedo_model','nsrdb_albedo_contract')


def load_run(run, label):
    run = Path(run)
    required = ['latest.pt', 'episodes.csv', 'resolved_mdp.json', 'config.json']
    missing = [p for p in required if not (run/p).is_file()]
    if missing:
        raise FileNotFoundError(f'{label} run {run}: missing {missing}')
    mdp = json.loads((run/'resolved_mdp.json').read_text(encoding='utf-8'))
    for key, value in EXPECTED.items():
        if mdp.get(key) != value:
            raise ValueError(f'{label}: expected MDP {key}={value!r}; found {mdp.get(key)!r}')
    with (run/'episodes.csv').open(newline='',encoding='utf-8') as fp:
        rd = csv.DictReader(fp)
        if not rd.fieldnames or 'scenario_id' not in rd.fieldnames:
            raise ValueError(f'{label}: episodes.csv lacks scenario_id column')
        rows = list(rd)
    if not rows or any(not row.get('scenario_id') for row in rows):
        raise ValueError(f'{label}: empty or incomplete episode history')
    if 'env_steps' not in rows[-1] or 'epoch' not in rows[-1]:
        raise ValueError(f'{label}: episodes.csv must record cumulative epoch and env_steps')
    try:
        steps = int(rows[-1]['env_steps'])
        final_epoch = int(rows[-1]['epoch'])
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{label}: invalid final epoch/env_steps') from exc
    if steps != 195*len(rows):
        raise ValueError(f'{label}: recorded {steps} env steps for {len(rows)} episodes; '
                         f'expected {195*len(rows)}. Investigate incomplete run/history.')
    if final_epoch != len(rows):
        raise ValueError(f'{label}: final CSV epoch={final_epoch}, but history has {len(rows)} rows')

    # A matching resolved_mdp.json is not sufficient: a stale latest.pt can
    # accidentally be paired with another run's CSV. Verify checkpoint-to-file
    # and checkpoint-to-history provenance before trusting either path.
    import torch
    checkpoint = torch.load(run/'latest.pt', map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f'{label}: latest.pt is not a recognized checkpoint dictionary')
    if checkpoint.get('resolved_mdp') != mdp:
        raise ValueError(f'{label}: latest.pt resolved_mdp differs from resolved_mdp.json')
    try:
        ck_epoch = int(checkpoint['epoch'])
        ck_steps = int(checkpoint['env_steps'])
        ck_seed = int(checkpoint['seed'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{label}: latest.pt lacks valid seed/epoch/env_steps provenance') from exc
    if ck_epoch != len(rows) or ck_epoch != final_epoch:
        raise ValueError(f'{label}: checkpoint epoch={ck_epoch} does not match CSV epoch={final_epoch}')
    if ck_steps != steps:
        raise ValueError(f'{label}: checkpoint env_steps={ck_steps} does not match CSV env_steps={steps}')
    return mdp, {r['scenario_id'] for r in rows}, len(rows), ck_seed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sac-run',required=True)
    p.add_argument('--mbpo-run',required=True)
    p.add_argument('--days-per-month',type=int,default=4)
    p.add_argument('--selection-seed',type=int,default=42)
    args=p.parse_args()
    sa, seen_sac, nsa, seed_sac = load_run(args.sac_run, 'SAC')
    mb, seen_mb, nmb, seed_mb = load_run(args.mbpo_run, 'MBPO')
    if seed_sac != seed_mb:
        raise ValueError(f'Paired SAC/MBPO checkpoints use different training seeds: {seed_sac} vs {seed_mb}')
    for key in CHECK_FIELDS:
        if sa.get(key) != mb.get(key):
            raise ValueError(f'Paired checkpoints have different recorded MDP {key}: '
                             f'SAC={sa.get(key)!r}, MBPO={mb.get(key)!r}')
    if sa.get('power_model') != 'pvwatts_sapm':
        raise ValueError('Evaluation command requests PVWatts/SAPM but runs differ')
    from SAC_workspace.src.config.config import load_config
    from mbpo.env.nsrdb_weather import load_scenario_manifest
    from SAC_workspace.src.evaluation.scenario_sampling import build_selection, validate_selection
    cfg = load_config(power_model='pvwatts_sapm',wind_reference_height_m=2,
                      gamma_pdc=-0.004,sapm_mount='open_rack_glass_polymer',
                      wind_powerlaw_exponent=0.14)
    manifest=load_scenario_manifest(cfg.manifest)
    plan=build_selection(manifest,days_per_month=args.days_per_month,
                         selection_seed=args.selection_seed,
                         excluded_ids=seen_sac | seen_mb)
    validate_selection(plan, manifest)
    if plan['episode']['periods'] != 196:
        raise ValueError('Selected NSRDB manifest is not the correct full-day episode')
    print('PASS: both local run directories contain checkpoints and complete episode histories')
    print(f'PASS: SAC episodes={nsa}, MBPO episodes={nmb}, each 195 transitions; shared training seed={seed_sac}')
    print('PASS: each latest.pt matches its own resolved_mdp.json and episodes.csv')
    print('PASS: compared policies share recorded MDP physics and complete full-day horizon')
    print(f"PASS: selection excludes {len(seen_sac|seen_mb)} observed training dates")
    print(f"PASS: selected={plan['selected_count']} / eligible={plan['population_count']} NSRDB dates")
    print('NOTE: logged-date exclusions cannot prove that all training-day provenance is complete.')


if __name__=='__main__':
    main()
