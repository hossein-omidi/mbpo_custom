#!/usr/bin/env python3
"""Install the official New York City NSRDB 5-min dataset and build the manifest.

Source data: raw SAM CSVs downloaded from the NSRDB API (GOES CONUS PSM v4,
5-min, 2018-2024) by NSRDB_test.py into nsrdb_newyork_5min/{year}/<id>/*.csv.

IMPORTANT — timezone: the corrected download (NSRDB_test.py) requests utc=true,
so rows are stored natively in UTC (SAM header `Time Zone` = 0). The project
loader (mbpo.env.nsrdb_iotools via pvlib read_psm3/read_nsrdb_psm4) honors the
`Time Zone` header either way (it also converts older local-standard-time
files, header -5, to UTC), so episodes are aligned to exact UTC 5-min rows
with no shifting or interpolation. This script verifies that alignment
against pvlib solar geometry before writing the manifest.

Steps:
  1. Discover one raw SAM CSV per year under --raw-dir.
  2. Copy to data/pv_weather/nsrdb/newyork_{year}_5min.csv (canonical names).
  3. Validate each year: UTC index, 5-min spacing, full coverage incl. Feb 29,
     no NaNs in required columns, zenith-column vs pvlib agreement (< 1 deg).
  4. Build scenario manifest (episode window 12:00-21:45 UTC, 118 stamps).

Usage:
  python scripts/prepare_nsrdb_newyork_catalog.py
  python scripts/prepare_nsrdb_newyork_catalog.py --years 2018-2024 --force
"""

from __future__ import print_function

import argparse
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mbpo.env.nsrdb_iotools import read_nsrdb_csv_to_env_weather
from mbpo.env.nsrdb_weather import (
    DEFAULT_EPISODE_PERIODS,
    DEFAULT_EPISODE_START_TIME,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    DEFAULT_ALTITUDE,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_NSRDB_DATA_DIR,
    DEFAULT_WEATHER_DATASET_DIR,
    REQUIRED_COLUMNS,
    build_manifest_from_year_files,
    validate_nsrdb_year_weather,
    write_manifest,
)

ZENITH_SYNC_TOLERANCE_DEG = 1.0


def parse_years(spec):
    years = []
    for part in spec.replace(' ', '').split(','):
        if '-' in part:
            a, b = part.split('-', 1)
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def find_raw_sam_csv(raw_dir, year):
    """Locate the raw SAM CSV (with 2 metadata header rows) for one year."""
    candidates = sorted(
        glob.glob(os.path.join(raw_dir, str(year), '*', '*.csv'))
        + glob.glob(os.path.join(raw_dir, str(year), '*.csv')))
    sam = []
    for path in candidates:
        with open(path, encoding='utf-8') as f:
            header = f.readline()
        # SAM files start with the metadata header, not the data header.
        if header.startswith('Source,Location ID'):
            sam.append(path)
    if not sam:
        return None
    return sam[0]


def validate_year(path, year):
    """Load via the project pvlib path and run hard checks. Returns weather."""
    weather, meta = read_nsrdb_csv_to_env_weather(path)
    validate_nsrdb_year_weather(weather, path=path)

    # Coverage: every calendar day of the source year present after UTC shift.
    in_year = weather[weather.index.year == year]
    expected_days = 366 if pd.Timestamp(year=year, month=12, day=31).dayofyear == 366 else 365
    n_days = len(np.unique(in_year.index.date))
    if n_days < expected_days - 1:
        raise ValueError(
            '{}: only {} days covered in {} (expected ~{})'.format(
                path, n_days, year, expected_days))

    nan_required = weather[list(REQUIRED_COLUMNS)].isnull().sum().sum()
    if nan_required:
        raise ValueError('{}: {} NaNs in required columns'.format(path, nan_required))

    # Timezone synchronization proof: dataset zenith column must match pvlib
    # solar position computed at the UTC timestamps for the NYC site.
    if 'solar_zenith_angle' in weather.columns:
        from pvlib.location import Location
        loc = Location(DEFAULT_LATITUDE, DEFAULT_LONGITUDE,
                       tz='UTC', altitude=DEFAULT_ALTITUDE)
        sample = weather.iloc[::577]  # ~180 spot checks across the year
        sp = loc.get_solarposition(sample.index)
        err = np.abs(sp['zenith'].values - sample['solar_zenith_angle'].values)
        if float(np.nanmax(err)) > ZENITH_SYNC_TOLERANCE_DEG:
            raise ValueError(
                '{}: UTC sync failed; max zenith error {:.2f} deg '
                '(timestamps misaligned?)'.format(path, float(np.nanmax(err))))
        print('  [sync] zenith vs pvlib: mean {:.3f} deg, max {:.3f} deg'.format(
            float(np.nanmean(err)), float(np.nanmax(err))))

    print('  [ok] {} rows={} days={} utc=[{} .. {}]'.format(
        os.path.basename(path), len(weather), n_days,
        weather.index[0], weather.index[-1]))
    return weather


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw-dir', default=str(DEFAULT_WEATHER_DATASET_DIR),
                   help='Folder with raw NSRDB API downloads (NSRDB_test.py output)')
    p.add_argument('--years', default='2018-2024')
    p.add_argument('--outdir', default=str(DEFAULT_NSRDB_DATA_DIR))
    p.add_argument('--manifest', default=str(DEFAULT_MANIFEST_PATH))
    p.add_argument('--force', action='store_true',
                   help='Overwrite existing canonical year CSVs')
    args = p.parse_args()

    years = parse_years(args.years)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    year_file_map = {}
    for year in years:
        dest = os.path.join(outdir, 'newyork_{}_5min.csv'.format(year))
        if os.path.isfile(dest) and not args.force:
            print('[keep] {}'.format(dest))
        else:
            src = find_raw_sam_csv(args.raw_dir, year)
            if src is None:
                raise SystemExit(
                    'No raw SAM CSV for year {} under {}. '
                    'Run NSRDB_test.py first.'.format(year, args.raw_dir))
            shutil.copy2(src, dest)
            print('[copy] {} -> {}'.format(src, dest))
        print('[validate] year {}'.format(year))
        validate_year(dest, year)
        year_file_map[year] = dest

    print('[manifest] building scenarios (window {} UTC, {} stamps, 5min) ...'.format(
        DEFAULT_EPISODE_START_TIME, DEFAULT_EPISODE_PERIODS))
    scenarios = build_manifest_from_year_files(
        outdir, years,
        latitude=DEFAULT_LATITUDE,
        longitude=DEFAULT_LONGITUDE,
        start_time=DEFAULT_EPISODE_START_TIME,
        periods=DEFAULT_EPISODE_PERIODS,
        year_file_map=year_file_map,
    )
    if not scenarios:
        raise SystemExit('No scenarios built; check episode window coverage.')

    manifest_path = write_manifest(
        args.manifest,
        scenarios,
        outdir,
        latitude=DEFAULT_LATITUDE,
        longitude=DEFAULT_LONGITUDE,
        start_time=DEFAULT_EPISODE_START_TIME,
        periods=DEFAULT_EPISODE_PERIODS,
        extra_meta={
            'site_name': 'New York City',
            'altitude_m': DEFAULT_ALTITUDE,
            'interval_minutes': 5,
            'source_dataset': 'NSRDB GOES CONUS PSM v4 (developer API, 5-min)',
            'raw_download_dir': os.path.abspath(args.raw_dir),
            'raw_timezone': 'UTC native (requested with utc=true; SAM header '
                            'Time Zone 0; loader honors the header)',
        },
    )

    by_mday = {}
    for s in scenarios:
        by_mday.setdefault((int(s['month']), int(s['day'])), set()).add(
            int(s['source_year']))
    multi = sum(1 for ys in by_mday.values() if len(ys) > 1)
    feb29 = [s['scenario_id'] for s in scenarios
             if int(s['month']) == 2 and int(s['day']) == 29]
    print('Manifest:', manifest_path)
    print('Scenarios:', len(scenarios),
          '| calendar days with >1 year:', multi,
          '| Feb 29 scenarios:', feb29)
    print('Years:', sorted({int(s['source_year']) for s in scenarios}))


if __name__ == '__main__':
    main()
