#!/usr/bin/env python3
"""Download and cache the default one-location historical weather dataset.

This script fetches a PVGIS TMY dataset for the project's default site,
re-maps it to a canonical non-leap UTC year, upsamples it to 15-minute
resolution, and stores a ready-to-use CSV inside the repo.
"""

from pathlib import Path
import json

import pandas as pd
from pvlib.iotools import get_pvgis_tmy

from mbpo.env.historical_weather import (
    CANONICAL_YEAR,
    default_historical_weather_file,
)


LATITUDE = 35.0
LONGITUDE = -106.0


def main():
    weather, months_selected, inputs, meta = get_pvgis_tmy(
        LATITUDE, LONGITUDE, map_variables=True)
    weather = weather.rename(columns={'temp_air': 'temperature'})

    canonical_index = pd.DatetimeIndex([
        pd.Timestamp(
            year=CANONICAL_YEAR,
            month=timestamp.month,
            day=timestamp.day,
            hour=timestamp.hour,
            minute=timestamp.minute,
            tz='UTC',
        )
        for timestamp in weather.index.tz_convert('UTC')
    ])
    weather.index = canonical_index
    weather = weather[~weather.index.duplicated(keep='first')]
    weather = weather.sort_index()

    full_index = pd.date_range(
        '{}-01-01 00:00'.format(CANONICAL_YEAR),
        '{}-12-31 23:45'.format(CANONICAL_YEAR),
        freq='15min',
        tz='UTC',
    )
    weather = weather.reindex(full_index)
    numeric_cols = [c for c in weather.columns if pd.api.types.is_numeric_dtype(weather[c])]
    weather[numeric_cols] = weather[numeric_cols].interpolate(
        method='time', limit_direction='both')
    weather['dni'] = weather['dni'].clip(lower=0.0)
    weather['ghi'] = weather['ghi'].clip(lower=0.0)
    weather['dhi'] = weather['dhi'].clip(lower=0.0)
    weather['wind_speed'] = weather['wind_speed'].clip(lower=0.0)

    out_path = Path(default_historical_weather_file())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(out_path)

    metadata = {
        'source': 'PVGIS_TMY',
        'latitude': LATITUDE,
        'longitude': LONGITUDE,
        'canonical_year': CANONICAL_YEAR,
        'rows': int(len(weather)),
        'columns': list(weather.columns),
        'months_selected': months_selected,
        'inputs': inputs,
        'meta': meta,
    }
    metadata_path = out_path.with_suffix('.metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)

    print('Saved historical weather CSV to:', out_path)
    print('Saved metadata JSON to:', metadata_path)


if __name__ == '__main__':
    main()
