#!/usr/bin/env python3
"""Build NSRDB multi-year scenario manifest from offline SAM CSVs or API downloads.

Offline (recommended): place one SAM CSV per year (5-min UTC) and run:
  python scripts/prepare_nsrdb_multiyear_catalog.py \\
    --weather-dataset-dir /home/user01/weather_dataset/weather_dataset \\
    --years 2018-2024

Examples:
  python scripts/prepare_nsrdb_multiyear_catalog.py --import-csv 2018:/path/to/2018.csv
  python scripts/prepare_nsrdb_multiyear_catalog.py --years 2015-2020
  python scripts/prepare_nsrdb_multiyear_catalog.py --check-connectivity
  python scripts/prepare_nsrdb_multiyear_catalog.py --bootstrap-from-tmy-dev --years 2018,2019,2020
"""

from __future__ import print_function

import argparse
import json
import os
import shutil
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mbpo.env.nsrdb_weather import (
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_NSRDB_DATA_DIR,
    DEFAULT_WEATHER_DATASET_DIR,
    NSRDB_API_BASE,
    NSRDB_API_HOST,
    build_manifest_from_year_files,
    check_nsrdb_api_reachable,
    discover_year_csv_files,
    fetch_nsrdb_year_utc,
    import_psm3_csv_to_year_utc,
    write_manifest,
)


def parse_years(spec):
    years = []
    for part in spec.replace(' ', '').split(','):
        if '-' in part:
            a, b = part.split('-', 1)
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def parse_import_csv(specs):
    """Parse --import-csv 2015:/path/a.csv 2016:/path/b.csv"""
    out = []
    for spec in specs:
        if ':' not in spec:
            raise ValueError(
                'import-csv must be YEAR:/path/to/file.csv (got {!r})'.format(spec))
        year_s, path = spec.split(':', 1)
        out.append((int(year_s), path))
    return out


def save_year_csv(outdir, year, data, meta, interval_minutes=5):
    suffix = '5min' if int(interval_minutes) <= 5 else '15min'
    out_path = os.path.join(outdir, 'nsrdb_{}_utc_{}.csv'.format(year, suffix))
    meta_path = out_path.replace('.csv', '.metadata.json')
    data.to_csv(out_path)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({k: str(v) for k, v in meta.items()}, f, indent=2)
    print('  saved', out_path, 'rows=', len(data))
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        '--years', default='2018-2024',
        help='Comma list or range, e.g. 2018-2024')
    p.add_argument('--latitude', type=float, default=DEFAULT_LATITUDE)
    p.add_argument('--longitude', type=float, default=DEFAULT_LONGITUDE)
    p.add_argument('--outdir', default=str(DEFAULT_NSRDB_DATA_DIR))
    p.add_argument('--manifest', default=str(DEFAULT_MANIFEST_PATH))
    p.add_argument(
        '--weather-dataset-dir', default=None,
        help='Offline SAM CSV directory (one file per year, 5-min UTC)')
    p.add_argument('--interval', type=int, default=5,
                   help='Native CSV interval (minutes); episode grid still 7min30s')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--skip-download', action='store_true',
                   help='Only rebuild manifest from existing year CSVs')
    p.add_argument(
        '--check-connectivity', action='store_true',
        help='Test DNS/HTTPS to developer.nrel.gov and exit')
    p.add_argument(
        '--import-csv', nargs='+', default=[],
        metavar='YEAR:PATH',
        help='Import local NSRDB SAM CSV per year (offline / Viewer export)')
    p.add_argument(
        '--copy-to-outdir', action='store_true',
        help='Copy imported/offline CSVs into --outdir with canonical names')
    p.add_argument(
        '--bootstrap-from-tmy-dev', action='store_true',
        help='DEV ONLY: build year CSVs from bundled PVGIS TMY (not real NSRDB)')
    args = p.parse_args()

    if args.check_connectivity:
        ok, msg = check_nsrdb_api_reachable()
        print('Host:', NSRDB_API_HOST)
        print('API: ', NSRDB_API_BASE)
        print('Status:', msg)
        raise SystemExit(0 if ok else 1)

    years = parse_years(args.years)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    year_file_map = {}

    dataset_dir = args.weather_dataset_dir
    if dataset_dir is None and DEFAULT_WEATHER_DATASET_DIR.is_dir():
        dataset_dir = str(DEFAULT_WEATHER_DATASET_DIR)
    if dataset_dir:
        discovered = discover_year_csv_files(dataset_dir, years=years)
        print('[discover] %s -> %d year files' % (dataset_dir, len(discovered)))
        for year, path in discovered.items():
            year_file_map[year] = path
            if args.copy_to_outdir and not args.dry_run:
                dest = os.path.join(outdir, 'nsrdb_{}_utc_5min.csv'.format(year))
                if not os.path.isfile(dest):
                    shutil.copy2(path, dest)
                    print('[copy]', path, '->', dest)
                year_file_map[year] = dest

    api_key = os.environ.get('NSRDB_API_KEY') or os.environ.get('NREL_API_KEY')
    email = os.environ.get('NSRDB_EMAIL') or os.environ.get('NREL_EMAIL')

    if args.import_csv:
        for year, csv_path in parse_import_csv(args.import_csv):
            if year not in years:
                years.append(year)
            years = sorted(set(years))
            if args.dry_run:
                print('[dry-run] import', year, '<-', csv_path)
                year_file_map[year] = csv_path
                continue
            print('[import] year', year, '<-', csv_path)
            data, meta = import_psm3_csv_to_year_utc(csv_path)
            if args.copy_to_outdir:
                saved = save_year_csv(outdir, year, data, meta, args.interval)
                year_file_map[year] = saved
            else:
                year_file_map[year] = os.path.abspath(csv_path)

    if args.bootstrap_from_tmy_dev:
        from mbpo.env.historical_weather import load_historical_weather_catalog
        tmy = load_historical_weather_catalog(freq='15min')
        scales = {2018: 0.88, 2019: 1.0, 2020: 1.12, 2021: 0.95, 2022: 1.05, 2023: 0.92, 2024: 1.08}
        for year in years:
            if args.dry_run:
                print('[dry-run] bootstrap TMY ->', year)
                continue
            frame = tmy.copy()
            frame.index = frame.index.map(
                lambda ts: ts.replace(year=int(year)))
            scale = scales.get(int(year), 1.0)
            for col in ('ghi', 'dni', 'dhi'):
                frame[col] = frame[col] * scale
            saved = save_year_csv(outdir, year, frame, {'source': 'TMY_bootstrap_dev'}, 15)
            year_file_map[year] = saved

    elif not args.skip_download and not args.import_csv and not year_file_map:
        if not api_key or not email:
            raise SystemExit(
                'Set NSRDB_API_KEY and NSRDB_EMAIL, or use --weather-dataset-dir, '
                '--import-csv, or --bootstrap-from-tmy-dev.\n'
                'Register: https://developer.nrel.gov/signup/')
        ok, msg = check_nsrdb_api_reachable()
        if not ok:
            raise SystemExit(
                '{}\n\nWorkarounds:\n'
                '  1) Fix DNS/firewall so this machine resolves {}\n'
                '  2) Download SAM CSV from NSRDB Viewer, then:\n'
                '     python scripts/prepare_nsrdb_multiyear_catalog.py '
                '--weather-dataset-dir /path/to/csvs --years 2018-2024\n'
                '  3) Dev only: --bootstrap-from-tmy-dev'.format(msg, NSRDB_API_HOST))
        print('[connectivity]', msg)

        intervals_to_try = [args.interval]
        if args.interval == 5:
            intervals_to_try.extend([15, 60])
        elif args.interval == 15:
            intervals_to_try.append(60)

        for year in years:
            out_path = os.path.join(outdir, 'nsrdb_{}_utc_5min.csv'.format(year))
            if os.path.isfile(out_path):
                print('[skip] exists:', out_path)
                year_file_map[year] = out_path
                continue
            if args.dry_run:
                print('[dry-run] would download year', year)
                continue
            data, meta = None, None
            for interval in intervals_to_try:
                print('[download] NSRDB PSM3 year %s interval=%d min ...' % (year, interval))
                try:
                    data, meta = fetch_nsrdb_year_utc(
                        args.latitude, args.longitude, year, api_key, email,
                        interval=interval)
                    break
                except Exception as exc:
                    print('[warn] interval=%d failed: %s' % (interval, exc))
                    data, meta = None, None
            if data is None:
                raise SystemExit(
                    'Failed to download year {}. Try --import-csv or --weather-dataset-dir.'.format(
                        year))
            saved = save_year_csv(outdir, year, data, meta, args.interval)
            year_file_map[year] = saved

    if args.dry_run:
        print('Dry run complete. Year files:', sorted(year_file_map.keys()))
        return

    if not year_file_map:
        years_on_disk = discover_year_csv_files(outdir, years=years)
        year_file_map.update(years_on_disk)

    if not year_file_map:
        raise SystemExit(
            'No year CSV files found. Use --weather-dataset-dir, --import-csv, '
            'download, or --bootstrap-from-tmy-dev.')

    scenarios = build_manifest_from_year_files(
        outdir, sorted(year_file_map.keys()),
        latitude=args.latitude,
        longitude=args.longitude,
        year_file_map=year_file_map,
    )
    if not scenarios:
        raise SystemExit('No scenarios built; check episode window coverage in year CSVs.')

    manifest_path = write_manifest(
        args.manifest,
        scenarios,
        outdir,
        latitude=args.latitude,
        longitude=args.longitude,
        extra_meta={
            'interval_minutes': args.interval,
            'years_on_disk': sorted(year_file_map.keys()),
            'bootstrap_from_tmy_dev': bool(args.bootstrap_from_tmy_dev),
            'weather_dataset_dir': dataset_dir,
        },
    )
    by_mday = {}
    for s in scenarios:
        key = (int(s['month']), int(s['day']))
        by_mday.setdefault(key, set()).add(int(s['source_year']))
    multi = sum(1 for ys in by_mday.values() if len(ys) > 1)
    leap = [s['scenario_id'] for s in scenarios if int(s['month']) == 2 and int(s['day']) == 29]
    print('Manifest:', manifest_path)
    print('Scenarios:', len(scenarios), '| calendar days with >1 year:', multi)
    print('Feb 29 scenarios:', len(leap), leap[:3])
    print('Years:', sorted({int(s['source_year']) for s in scenarios}))


if __name__ == '__main__':
    main()
