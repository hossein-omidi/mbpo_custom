# Reference / smoke test: local NSRDB CSV → pvlib reader → POA power.
# Canonical implementation lives in mbpo.env.nsrdb_iotools and mbpo.env.pvlib_physics.

import glob
import os
import sys

import pandas as pd
import pvlib

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mbpo.env.nsrdb_iotools import read_nsrdb_csv_to_env_weather
from mbpo.env.pvlib_physics import compute_poa_global

DATA_DIR = os.path.join(_REPO, 'data/pv_weather/nsrdb')
FILE_PATTERN = os.path.join(DATA_DIR, 'newyork_*_5min.csv')

LATITUDE = 40.72
LONGITUDE = -74.01
TZ = 'UTC'
SURFACE_TILT = 30.0
SURFACE_AZIMUTH = 180.0


# Helper, not a pytest case (this file is a script: python tests/test1.py).
def check_pvlib_power_path(df):
    location = pvlib.location.Location(
        latitude=LATITUDE, longitude=LONGITUDE, tz=TZ, altitude=12, name='NewYork_test')

    sample = df.between_time('12:00', '21:45').head(20).copy()
    if sample.empty:
        raise ValueError('No data found in the expected UTC daytime window.')

    solpos = location.get_solarposition(sample.index)
    poa_vals = []
    for i, row in sample.iterrows():
        poa_vals.append(compute_poa_global(
            SURFACE_TILT, SURFACE_AZIMUTH,
            solpos.loc[i, 'zenith'], solpos.loc[i, 'azimuth'],
            row['dni'], row['ghi'], row['dhi']))
    sample['poa_global'] = poa_vals

    print('\nPVLIB POA TEST')
    print('--------------')
    print(sample[['ghi', 'dni', 'dhi', 'poa_global']].head())
    print('\nPOA global min/max: {:.2f} / {:.2f} W/m2'.format(
        sample['poa_global'].min(), sample['poa_global'].max()))


def main():
    files = sorted(glob.glob(FILE_PATTERN))
    if not files:
        raise RuntimeError('No files found with pattern: {}'.format(FILE_PATTERN))

    print('Found {} NSRDB files.'.format(len(files)))
    path = files[0]
    print('\nReading:', path)

    weather, meta = read_nsrdb_csv_to_env_weather(path)
    print('Reader used:', meta.get('_reader'))
    print('Shape:', weather.shape)
    print('Index start:', weather.index[0])
    print('Index end:', weather.index[-1])

    print('Weather check:')
    for col in ['ghi', 'dni', 'dhi', 'temperature', 'wind_speed']:
        if col in weather.columns:
            print('  {}: min={:.2f}, mean={:.2f}, max={:.2f}'.format(
                col, weather[col].min(), weather[col].mean(), weather[col].max()))
        else:
            print('  {}: not found'.format(col))

    check_pvlib_power_path(weather)


if __name__ == '__main__':
    main()
