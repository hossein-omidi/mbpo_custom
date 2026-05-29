from pathlib import Path

import numpy as np
import pandas as pd

from pvlib.iotools import read_epw, read_psm3, read_tmy3


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORICAL_WEATHER_FILE = (
    REPO_ROOT / 'data' / 'pv_weather' / 'albuquerque_pvgis_tmy_utc_15min.csv')
CANONICAL_YEAR = 2021  # non-leap year for month/day lookup
DEFAULT_SITE_LATITUDE = 35.0
DEFAULT_SITE_LONGITUDE = -106.0


def default_historical_weather_file():
    return str(DEFAULT_HISTORICAL_WEATHER_FILE)


def using_default_site(latitude, longitude, atol=1e-6):
    return (
        abs(float(latitude) - DEFAULT_SITE_LATITUDE) <= atol
        and abs(float(longitude) - DEFAULT_SITE_LONGITUDE) <= atol)


def _canonicalize_index(index):
    canonical = []
    for timestamp in pd.DatetimeIndex(index).tz_convert('UTC'):
        canonical.append(pd.Timestamp(
            year=CANONICAL_YEAR,
            month=timestamp.month,
            day=timestamp.day,
            hour=timestamp.hour,
            minute=timestamp.minute,
            tz='UTC',
        ))
    return pd.DatetimeIndex(canonical)


def _normalize_weather_columns(frame):
    frame = frame.copy()
    rename_map = {
        'temp_air': 'temperature',
        'Temperature': 'temperature',
        'Tamb': 'temperature',
        'DryBulb': 'temperature',
        'wind_speed': 'wind_speed',
        'Wspd': 'wind_speed',
        'ghi': 'ghi',
        'GHI': 'ghi',
        'dhi': 'dhi',
        'DHI': 'dhi',
        'dni': 'dni',
        'DNI': 'dni',
        'relative_humidity': 'relative_humidity',
        'RHum': 'relative_humidity',
        'pressure': 'pressure',
        'Pres': 'pressure',
    }
    frame = frame.rename(columns=rename_map)
    required = ('dni', 'ghi', 'dhi', 'temperature', 'wind_speed')
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(
            'Historical weather file missing required columns {}. Got columns: {}'
            .format(missing, sorted(frame.columns)))
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    optional = ('relative_humidity', 'pressure')
    for column in optional:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame['dni'] = frame['dni'].clip(lower=0.0)
    frame['ghi'] = frame['ghi'].clip(lower=0.0)
    frame['dhi'] = frame['dhi'].clip(lower=0.0)
    frame['wind_speed'] = frame['wind_speed'].clip(lower=0.0)
    return frame


def _read_weather_file(weather_file):
    path = Path(weather_file)
    if not path.exists():
        raise FileNotFoundError(
            'Historical weather file not found: {}. '
            'Run scripts/prepare_default_historical_weather.py or pass '
            'weather_file=<path-to-epw/tmy3/psm3/csv>.'.format(path))
    suffixes = [s.lower() for s in path.suffixes]
    if path.suffix.lower() == '.epw':
        data, _ = read_epw(str(path))
    elif path.suffix.lower() in ('.tm3', '.tmy3'):
        data, _ = read_tmy3(str(path), map_variables=True)
    elif suffixes[-2:] == ['.psm3', '.csv']:
        data, _ = read_psm3(str(path), map_variables=True)
    else:
        try:
            data = pd.read_csv(str(path), index_col=0, parse_dates=True)
        except Exception:
            data, _ = read_psm3(str(path), map_variables=True)
        else:
            if not isinstance(data.index, pd.DatetimeIndex):
                data, _ = read_psm3(str(path), map_variables=True)
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError('Historical weather data must have a DatetimeIndex.')
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    else:
        data.index = data.index.tz_convert('UTC')
    return _normalize_weather_columns(data)


def load_historical_weather_catalog(weather_file=None, freq='15min'):
    weather_file = weather_file or default_historical_weather_file()
    weather = _read_weather_file(weather_file)

    weather.index = _canonicalize_index(weather.index)
    weather = weather[~weather.index.duplicated(keep='first')]
    weather = weather.sort_index()

    full_index = pd.date_range(
        '{}-01-01 00:00'.format(CANONICAL_YEAR),
        '{}-12-31 23:45'.format(CANONICAL_YEAR),
        freq=freq,
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
    return weather


def available_month_days(weather_catalog):
    return {
        (int(timestamp.month), int(timestamp.day))
        for timestamp in weather_catalog.index
    }


def classify_weather_conditions(location, times, weather_frame):
    clearsky = location.get_clearsky(times)
    ghi_clear = clearsky['ghi'].clip(lower=1.0).values
    dni_clear = clearsky['dni'].clip(lower=1.0).values
    ghi_ratio = np.clip(weather_frame['ghi'].values / ghi_clear, 0.0, 1.5)
    dni_ratio = np.clip(weather_frame['dni'].values / dni_clear, 0.0, 1.5)

    conditions = np.full(len(times), 'overcast', dtype=object)
    clear_mask = (ghi_ratio >= 0.75) & (dni_ratio >= 0.6)
    partial_mask = (~clear_mask) & (ghi_ratio >= 0.35)
    conditions[clear_mask] = 'clear'
    conditions[partial_mask] = 'partly_cloudy'
    return conditions


def build_weather_profile_from_catalog(location, times, weather_catalog):
    canonical = []
    for timestamp in pd.DatetimeIndex(times).tz_convert('UTC'):
        try:
            canonical.append(pd.Timestamp(
                year=CANONICAL_YEAR,
                month=timestamp.month,
                day=timestamp.day,
                hour=timestamp.hour,
                minute=timestamp.minute,
                tz='UTC',
            ))
        except ValueError as exc:
            raise ValueError(
                'Historical weather catalog does not include leap-day timestamps '
                'for {}. Choose non-leap fixed_eval_dates or provide an explicit '
                'weather_file with leap-day coverage.'.format(timestamp.date())
            ) from exc
    canonical_times = pd.DatetimeIndex(canonical)

    weather = weather_catalog.reindex(canonical_times)
    if weather.isnull().any().any():
        missing = weather[weather.isnull().any(axis=1)].index[:5]
        raise ValueError(
            'Historical weather catalog missing timestamps: {}'.format(
                [ts.isoformat() for ts in missing]))

    weather = weather.copy()
    weather.index = times
    weather['condition'] = classify_weather_conditions(location, times, weather)
    return weather
