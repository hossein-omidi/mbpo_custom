"""Optional NSRDB PSM v4 downloader. Training/evaluation NEVER call this.

Security: no embedded credential. Set NLR_API_KEY and NSRDB_EMAIL in the
terminal or a private credential manager. Existing downloaded annual CSVs
are read directly by the simulator; there is no API dependency at runtime.
"""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import time
import zipfile
import requests
import pandas as pd

BASE_URL='https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download.json'
ATTRIBUTES=('ghi,dhi,dni,air_temperature,wind_speed,relative_humidity,'
            'solar_zenith_angle,surface_albedo,surface_pressure,cloud_type,'
            'clearsky_ghi,clearsky_dhi,clearsky_dni,fill_flag')

def safe_extract(z,folder):
    folder=Path(folder).resolve()
    for entry in z.infolist():
        candidate=(folder/entry.filename).resolve()
        if not candidate.is_relative_to(folder):
            raise ValueError(f'Unsafe zip entry path: {entry.filename}')
    z.extractall(folder)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--years',type=int,nargs='+',default=list(range(2018,2025)))
    p.add_argument('--lat',type=float,default=40.7128)
    p.add_argument('--lon',type=float,default=-74.0060)
    p.add_argument('--outdir',default='nsrdb_newyork_5min')
    p.add_argument('--force',action='store_true',help='Explicitly allow replacing downloaded year files')
    p.add_argument('--max-attempts',type=int,default=80)
    args=p.parse_args(argv)
    key=os.environ.get('NLR_API_KEY') or os.environ.get('NSRDB_API_KEY')
    email=os.environ.get('NSRDB_EMAIL')
    if not key or not email:
        p.error('Set NLR_API_KEY and NSRDB_EMAIL before downloading. No keys are stored in code.')
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    for year in args.years:
        if not 2018<=year<=2024:p.error('PSM v4 source scope is limited to 2018–2024')
        path=out/f'newyork_nsrdb_5min_{year}.zip'
        folder=out/str(year)
        if (path.exists() or folder.exists()) and not args.force:
            raise FileExistsError(f'{year} files exist; refusing to replace. Use --force explicitly.')
        payload={'wkt':f'POINT({args.lon} {args.lat})','attributes':ATTRIBUTES,
                 'names':str(year),'interval':'5','utc':'true','leap_day':'true',
                 'email':email,'api_key':key}
        response=requests.post(BASE_URL,params={'api_key':key},data=payload,timeout=65)
        response.raise_for_status()
        data=response.json()
        if data.get('errors'):
            raise RuntimeError(f'NSRDB API reported errors ({len(data["errors"])} entries)')
        url=data.get('outputs',{}).get('downloadUrl')
        if not url:raise RuntimeError('NSRDB response had no downloadUrl')
        content=None
        for attempt in range(args.max_attempts):
            r=requests.get(url,timeout=120)
            if r.status_code==403 or r.status_code>=500:
                time.sleep(15)
                continue
            r.raise_for_status()
            if not zipfile.is_zipfile(io.BytesIO(r.content)):
                raise ValueError('Returned data was not an NSRDB ZIP')
            content=r.content;break
        if content is None:raise TimeoutError('NSRDB ZIP not ready after attempts')
        if folder.exists():
            import shutil
            if not args.force:raise FileExistsError(folder)
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            safe_extract(z,folder)
            csvs=[n for n in z.namelist() if n.lower().endswith('.csv')]
            if len(csvs)!=1:raise ValueError(f'Expected one NSRDB annual CSV, found {csvs!r}')
            with z.open(csvs[0]) as f:df=pd.read_csv(f,skiprows=2)
        required={'Year','Month','Day','Hour','Minute','DNI','DHI','GHI',
                  'Temperature','Wind Speed','Surface Albedo','Fill Flag'}
        if not required<=set(df.columns):
            raise ValueError(f'NSRDB data missing {sorted(required-set(df.columns))}')
        if len(df) not in (105120,105408):
            raise ValueError(f'Unexpected annual native 5-min row count: {len(df)}')
        path.write_bytes(content)
        print(f'{year}: validated {len(df)} native 5-minute rows; ZIP saved: {path}')
    print('Finished. Run the existing catalog preparation script to use the local CSVs.')

if __name__=='__main__':main()
