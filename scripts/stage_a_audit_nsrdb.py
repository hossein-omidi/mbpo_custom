"""Read-only audit of local NSRDB annual PSM v4 CSVs and Stage-A inputs.

No API, no interpolation, no synthetic weather, no data modifications. Reports
annual completeness, cadence, Surface Albedo, Fill Flag, and irradiance closure.
The optional --year-csv tests one original downloaded SAM/NSRDB CSV directly.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import math
import calendar
import numpy as np
import pandas as pd


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):
            digest.update(data)
    return digest.hexdigest()


def csv_metadata(path):
    with open(path,encoding='utf-8-sig',newline='') as f:
        r=csv.reader(f);keys=next(r);values=next(r)
    if len(keys)!=len(values):raise ValueError(f'Invalid NSRDB metadata rows: {path}')
    return dict(zip(keys,values))


def audit_year(path,year=None):
    path=Path(path).resolve()
    if not path.is_file():raise FileNotFoundError(path)
    metadata=csv_metadata(path)
    raw=pd.read_csv(path,skiprows=2,low_memory=False)
    required={'Year','Month','Day','Hour','Minute','GHI','DHI','DNI',
              'Temperature','Wind Speed','Surface Albedo','Fill Flag'}
    missing=sorted(required-set(raw.columns))
    if missing:raise ValueError(f'{path}: missing NSRDB fields: {missing}')
    if not len(raw):raise ValueError(f'Empty annual NSRDB file: {path}')
    nums=['Year','Month','Day','Hour','Minute','GHI','DHI','DNI',
          'Temperature','Wind Speed','Surface Albedo','Fill Flag']
    if 'Solar Zenith Angle' in raw:nums.append('Solar Zenith Angle')
    for col in nums:
        raw[col]=pd.to_numeric(raw[col],errors='coerce')
        if not np.isfinite(raw[col]).all():
            raise ValueError(f'{path}: invalid/missing NSRDB values in {col}')
    years=raw['Year'].astype(int).unique()
    if len(years)!=1:raise ValueError(f'{path}: multiple year labels: {years}')
    data_year=int(years[0])
    if year is not None and data_year!=int(year):
        raise ValueError(f'{path}: metadata year {data_year} != {year}')
    expected=288*(366 if calendar.isleap(data_year) else 365)
    if len(raw)!=expected:
        raise ValueError(f'{path}: {len(raw)} rows, expected {expected} for 5-minute native annual data')
    times=pd.to_datetime(raw[['Year','Month','Day','Hour','Minute']].rename(
        columns={'Year':'year','Month':'month','Day':'day','Hour':'hour','Minute':'minute'}),errors='raise')
    if (times.diff().dropna()!=pd.Timedelta(minutes=5)).any():
        raise ValueError(f'{path}: duplicated/missing/irregular native 5-minute timestamps')
    if times.iloc[0]!=pd.Timestamp(data_year,1,1) or times.iloc[-1]!=(pd.Timestamp(data_year+1,1,1)-pd.Timedelta(minutes=5)):
        raise ValueError(f'{path}: annual timestamp boundaries mismatch')
    alb=raw['Surface Albedo'].to_numpy(float)
    if (alb<0).any() or (alb>1).any():
        raise ValueError(f'{path}: albedo outside [0,1]')
    for col in ('GHI','DHI','DNI','Wind Speed'):
        if (raw[col]<0).any():
            raise ValueError(f'{path}: negative {col} (silent clipping is not acceptable)')
    flags=Counter(int(x) for x in raw['Fill Flag'])
    closure=None
    if 'Solar Zenith Angle' in raw:
        cosz=np.clip(np.cos(np.deg2rad(raw['Solar Zenith Angle'].to_numpy(float))),0,1)
        discrepancy=raw['GHI'].to_numpy(float)-(raw['DHI'].to_numpy(float)+raw['DNI'].to_numpy(float)*cosz)
        daytime=cosz>.01
        closure={'abs_p95_wm2':float(np.quantile(np.abs(discrepancy[daytime]),.95)) if daytime.any() else None,
                 'fraction_daytime_abs_gt_50_wm2':float(np.mean(np.abs(discrepancy[daytime])>50)) if daytime.any() else None}
    return {'year':data_year,'file':str(path),'sha256':sha(path),
            'row_count':len(raw),'time_zone_metadata':metadata.get('Time Zone'),
            'latitude':metadata.get('Latitude'),'longitude':metadata.get('Longitude'),
            'surface_albedo_mean':float(alb.mean()),
            'surface_albedo_min':float(alb.min()),
            'surface_albedo_max':float(alb.max()),
            'fill_flag_counts':dict(sorted(flags.items())),
            'irradiance_closure_diagnostic':closure,
            'weather_synthesis_or_interpolation':'none'}


def audit_manifest(manifest_path):
    # Match nsrdb_weather.load_scenario_manifest's resolution of data_dir
    # relative to repository root without importing pvlib or downloading data.
    path=Path(manifest_path).resolve()
    manifest=json.loads(path.read_text(encoding='utf-8'))
    root=Path(__file__).resolve().parents[1]
    data=Path(manifest.get('data_dir',path.parent))
    data=data if data.is_absolute() else root/data
    years={}
    for row in manifest['scenarios']:
        year=int(row.get('source_year',row['year']))
        target=row.get('file_path') or row.get('year_file') or f'newyork_{year}_5min.csv'
        target=Path(target)
        target=target if target.is_absolute() else data/target
        if year in years and years[year].resolve()!=target.resolve():
            raise ValueError(f'Conflicting local files for {year}')
        years[year]=target
    return [audit_year(p,y) for y,p in sorted(years.items())]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    exclusive=p.add_mutually_exclusive_group(required=True)
    exclusive.add_argument('--manifest')
    exclusive.add_argument('--year-csv')
    p.add_argument('--out',default='stage_a_nsrdb_data_audit.json')
    args=p.parse_args()
    reports=audit_manifest(args.manifest) if args.manifest else [audit_year(args.year_csv)]
    sites={(r['latitude'],r['longitude']) for r in reports}
    if len(sites)>1:
        raise ValueError(f'Inconsistent NSRDB site metadata across years: {sites}')
    result={'schema':'stage_a_nsrdb_audit_v1','annual_files':reports,
            'dataset_policy':'local_historical_csv_no_interpolation_no_synthesis',
            'note':'Fill Flags are reported, not silently deleted or corrected. '
                   'NSRDB is a satellite/model-derived product, not raw ground truth.'}
    Path(args.out).write_text(json.dumps(result,indent=2,sort_keys=True),encoding='utf-8')
    print('VALIDATED',len(reports),'NSRDB years, files unchanged. Report:',Path(args.out).resolve())
    for r in reports:
        print(r['year'],r['row_count'],'albedo mean',round(r['surface_albedo_mean'],4),
              'Fill Flag:',r['fill_flag_counts'])

if __name__=='__main__':main()
