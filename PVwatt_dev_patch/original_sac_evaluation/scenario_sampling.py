"""Reproducible, calendar-stratified subsampling of the native NSRDB manifest.

Each populated year/month is divided into k chronological bins, then one date
is drawn uniformly per bin. Every sampled date represents its bin's population
count. This makes the Horvitz-Thompson population mean unbiased under the
sampling design, WITHOUT claiming that these are unseen training dates.

No pvlib, Ray, TensorFlow or weather loads are needed to build the sample.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def _day(row):
    sid = str(row['scenario_id'])
    try:
        day = date.fromisoformat(sid)
    except (ValueError, TypeError) as exc:
        raise ValueError('Expected ISO YYYY-MM-DD NSRDB scenario_id, got: ' + sid) from exc
    y = int(row.get('year', day.year))
    m = int(row.get('month', day.month))
    if (day.year, day.month) != (y, m):
        raise ValueError('Scenario year/month mismatch for ' + sid)
    return day


def frame_fingerprint(manifest):
    """Stable fingerprint of scenario IDs and their year/month source mapping."""
    frame = sorted((str(r['scenario_id']), int(r.get('source_year', r.get('year', _day(r).year))))
                   for r in manifest['scenarios'])
    episode = manifest.get('episode', {})
    # Fingerprint includes the episode time grid, not just scenario IDs.
    # This prevents replaying a 117-step plan as if it were a 195-step plan.
    required = ('start_time', 'periods', 'freq')
    if any(k not in episode for k in required):
        raise ValueError('Scenario manifest is missing the complete episode time grid')
    grid = {k: episode[k] for k in required}
    grid['tz'] = episode.get('tz', 'UTC')
    payload = {'scenarios': frame, 'episode': grid}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_selection(manifest, days_per_month=3, selection_seed=42, exhaustive=False, excluded_ids=()):
    """Return a serializable selection plan, with exact inclusion weights.

    ``exhaustive=True`` retains the original every-eligible-date benchmark.
    Defaults to reproducible temporal bins (not a weather-stratified claim).
    """
    days_per_month = int(days_per_month)
    selection_seed = int(selection_seed)
    if not 1 <= days_per_month <= 31:
        raise ValueError('days_per_month must be between 1 and 31')
    grouped = defaultdict(list)
    seen = set()
    excluded = set(excluded_ids)
    all_population = {str(r['scenario_id']) for r in manifest['scenarios']}
    if excluded - all_population:
        raise ValueError('Training exclusion list contains dates not in this manifest')
    ep = manifest.get('episode', {})
    required_grid = {
        'start_time': '08:45', 'periods': 196, 'freq': '5min', 'tz': 'UTC'
    }
    for key, expected in required_grid.items():
        actual = ep.get(key, 'UTC' if key == 'tz' else None)
        if actual != expected:
            raise ValueError(
                f'Invalid full-day evaluation grid: episode.{key}={actual!r}; '
                f'expected {expected!r}. Regenerate the real NSRDB manifest '
                'for the 195-transition episode; never substitute weather.'
            )

    for row in manifest['scenarios']:
        d = _day(row)
        sid = d.isoformat()
        if sid in seen:
            raise ValueError('Duplicate scenario_id in manifest: ' + sid)
        seen.add(sid)
        if sid not in excluded:
            grouped[(d.year, d.month)].append(sid)
    if not grouped:
        raise ValueError('Empty NSRDB manifest')

    strata, selected = [], []
    for (year, month), candidates in sorted(grouped.items()):
        candidates = sorted(candidates)
        n_bins = len(candidates) if exhaustive else min(days_per_month, len(candidates))
        bins = np.array_split(np.asarray(candidates, dtype=object), n_bins)
        rng = np.random.default_rng(np.random.SeedSequence([selection_seed, year, month]))
        for bin_index, bin_days in enumerate(bins, 1):
            options = bin_days.tolist()
            sid = options[0] if exhaustive else options[int(rng.integers(len(options)))]
            key = f'{year:04d}-{month:02d}-bin{bin_index:02d}'
            strata.append({'stratum': key, 'year': year, 'month': month,
                           'bin_index': bin_index, 'population_days': len(options),
                           'selected_scenario_id': sid,
                           'sampled_days': 1, 'date_start': options[0],
                           'date_end': options[-1]})
            selected.append(sid)

    population_size = len(seen)-len(excluded)
    yearlist=sorted({x[0] for x in grouped})
    missing_months=[f'{year:04d}-{month:02d}' for year in yearlist for month in range(1,13)
                    if (year,month) not in grouped]
    # A calendar day in New York spans two UTC dates during summer. The
    # episode description comes from the authoritative weather time grid.
    start = datetime.fromisoformat('2000-06-21T' + ep['start_time']).replace(tzinfo=timezone.utc)
    finish = start + timedelta(minutes=(int(ep['periods']) - 1) * 5)
    episode_note = (
        f"{int(ep['periods']) - 1} native five-minute control transitions "
        f"from {start:%H:%M} UTC to {finish:%H:%M} UTC "
        f"{'the next UTC date' if finish.date() != start.date() else 'the same UTC date'}; "
        'not a 24-hour calendar UTC production day.'
    )
    return {
        'schema_version': 1,
        'episode': {**required_grid},
        'protocol': 'complete_manifest' if exhaustive else 'calendar_stratified_utc_daytime',
        'selection_seed': selection_seed, 'days_per_month': days_per_month,
        'population_count': population_size, 'selected_count': len(selected),
        'frame_count_before_exclusions': len(all_population),
        'excluded_scenario_ids': sorted(excluded),
        'year_month_strata_count': len(grouped),
        'years': yearlist, 'missing_year_months': missing_months,
        'manifest_fingerprint_sha256': frame_fingerprint(manifest),
        'selected_scenario_ids': selected,
        'strata': strata,
        'sampling_note': (
            'One uniform random scenario from each chronological within-month bin; '
            'each selected scenario represents all eligible days in its bin. '
            'Weather class is observed after selection, not balanced by design. '
            + ('Scenario IDs in the supplied training logs have been excluded. '
               'This is a logged-date holdout, not proof of unseen years or complete training provenance.'
               if excluded else
               'No training-date exclusion was requested; evaluation days may overlap training.')
        ),
        'episode_note': episode_note,
    }


def validate_selection(plan, manifest):
    if plan.get('schema_version') != 1:
        raise ValueError('Unsupported NSRDB sampling plan schema')
    grid = manifest.get('episode', {})
    expected_grid = {k: grid.get(k, 'UTC' if k == 'tz' else None)
                     for k in ('start_time', 'periods', 'freq', 'tz')}
    if plan.get('episode') != expected_grid:
        raise ValueError('Selection plan was created for a different episode grid; generate a fresh plan')
    if plan.get('manifest_fingerprint_sha256') != frame_fingerprint(manifest):
        raise ValueError('NSRDB manifest changed since selection was saved')
    ids = plan['selected_scenario_ids']
    population = {r['scenario_id'] for r in manifest['scenarios']}
    if not ids or len(ids) != len(set(ids)) or not set(ids).issubset(population):
        raise ValueError('Invalid sample membership or duplicate dates')
    strata = plan['strata']
    if len(strata) != len(ids) or {s['selected_scenario_id'] for s in strata} != set(ids):
        raise ValueError('Inconsistent sampling plan strata')
    excluded=set(plan.get('excluded_scenario_ids', []))
    if not excluded.issubset(population):
        raise ValueError('Excluded date is outside the manifest')
    if excluded & set(ids):
        raise ValueError('Excluded training date appeared in evaluation selection')
    if sum(int(s['population_days']) for s in strata) != len(population)-len(excluded):
        raise ValueError('Population weights do not cover all eligible dates')
    if len(population)-len(excluded)!=plan['population_count']:
        raise ValueError('Eligible population count mismatch')
    # Do not silently trust a user-edited or stale selection plan.
    regenerated=build_selection(manifest,days_per_month=plan['days_per_month'],
                                selection_seed=plan['selection_seed'],
                                exhaustive=plan['protocol']=='complete_manifest',
                                excluded_ids=excluded)
    if ids != regenerated['selected_scenario_ids'] or strata != regenerated['strata']:
        raise ValueError('Selection plan was edited or no longer matches deterministic sampling')
    return ids


def save_selection(plan, root):
    """Save before evaluation for replay/reproducibility; does not delete old results."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    previous=root/'scenario_selection.json'
    if (root/'episodes.csv').exists() and not previous.exists():
        raise ValueError('Existing non-stratified evaluation results in --outdir; choose a new directory')
    if previous.exists() and (root/'episodes.csv').exists():
        before=json.loads(previous.read_text(encoding='utf-8'))
        if before.get('selected_scenario_ids') != plan['selected_scenario_ids'] or (
                before.get('manifest_fingerprint_sha256') != plan.get('manifest_fingerprint_sha256')):
            raise ValueError('Existing evaluation output uses a different scenario sample; choose a new --outdir')
    (root/'scenario_selection.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    columns = ('scenario_id','year','month','stratum','population_days','date_start','date_end')
    with (root/'selected_scenarios.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for s in plan['strata']:
            writer.writerow({'scenario_id': s['selected_scenario_id'],
                             'year': s['year'], 'month': s['month'],
                             'stratum': s['stratum'], 'population_days': s['population_days'],
                             'date_start': s['date_start'], 'date_end': s['date_end']})
    return root/'scenario_selection.json'


def add_selection_arguments(parser, *, singular_seed=False):
    """Same public CLI semantics for SAC and MBPO evaluation scripts."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--scenarios', nargs='+', default=None,
                       help='Evaluate exactly these manifest scenario IDs')
    group.add_argument('--all-scenarios', '--stratified-scenarios',
                       dest='stratified_scenarios', action='store_true',
                       help='Evaluate a reproducible within-month sample, NOT all days (default 3 dates per month/year)')
    group.add_argument('--exhaustive-scenarios', action='store_true',
                       help='Opt-in: evaluate EVERY eligible manifest date (potentially 2557 days)')
    group.add_argument('--selection-plan', default=None,
                       help='Replay the exact scenario_selection.json produced by another evaluation')
    parser.add_argument('--days-per-month', type=int, default=3,
                        help='Randomly choose this many spread-out dates per source year/month (default: 3)')
    parser.add_argument('--exclude-training-days', action='store_true',
                        help='Exclude all scenario IDs seen in the supplied policy training CSV(s); must provide CSVs for each compared policy')
    parser.add_argument('--selection-seed', type=int, default=42,
                        help='Separate deterministic scenario sampling seed; independent of policy seed')


def _training_seen(csvs):
    known=set()
    for filepath in csvs:
        path=Path(filepath)
        if not path.is_file():
            raise ValueError('Training CSV is missing: '+str(path))
        with path.open(newline='',encoding='utf-8') as f:
            reader=csv.DictReader(f)
            if reader.fieldnames is None or 'scenario_id' not in reader.fieldnames:
                raise ValueError('Training CSV lacks scenario_id column: '+str(path))
            known.update(row['scenario_id'] for row in reader if row.get('scenario_id'))
    return known


def resolve_cli_selection(args, manifest, training_csvs=()):
    """Return (scenario IDs or None, stratification plan or None)."""
    requested = [bool(args.scenarios), bool(args.stratified_scenarios),
                 bool(args.exhaustive_scenarios), bool(args.selection_plan)]
    if sum(requested)>1:
        raise ValueError('Choose one of --scenarios, --all-scenarios, --exhaustive-scenarios, --selection-plan')
    if args.selection_plan:
        if args.exclude_training_days:
            raise ValueError('--selection-plan already fixes exclusions; do not combine it with --exclude-training-days')
        plan=json.loads(Path(args.selection_plan).read_text(encoding='utf-8'))
        return validate_selection(plan,manifest),plan
    if args.exclude_training_days and not (args.stratified_scenarios or args.exhaustive_scenarios):
        raise ValueError('--exclude-training-days requires --all-scenarios or --exhaustive-scenarios')
    if args.stratified_scenarios or args.exhaustive_scenarios:
        if args.exclude_training_days and not training_csvs:
            raise ValueError('Held-out evaluation requires every compared policy training CSV')
        excluded=_training_seen(training_csvs) if args.exclude_training_days else set()
        plan=build_selection(manifest, days_per_month=args.days_per_month,
                             selection_seed=args.selection_seed,
                             exhaustive=args.exhaustive_scenarios,excluded_ids=excluded)
        return validate_selection(plan,manifest),plan
    return args.scenarios,None
