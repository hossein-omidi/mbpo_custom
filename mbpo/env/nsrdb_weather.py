"""NSRDB PSM3 multi-year weather scenarios for PVTracking.

Each scenario e = (data_year, calendar month, day) maps to one fixed irradiance
trajectory W_e on the project UTC episode grid (13:30–23:15, 7min30s).

Data flow:
  - Offline SAM CSV (5-min UTC) from NSRDB Viewer or API download
  - Year cache + JSON manifest (scripts/prepare_nsrdb_multiyear_catalog.py)
  - Env samples scenario at reset; pvlib computes sun geometry and POA from W_e

Stochasticity across episodes: empirical scenario sampling e ~ p(e), not synthetic clouds.
"""

import calendar
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pvlib.iotools import get_psm3, read_psm3

from .historical_weather import classify_weather_conditions

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NSRDB_DATA_DIR = REPO_ROOT / 'data' / 'pv_weather' / 'nsrdb'
DEFAULT_MANIFEST_PATH = DEFAULT_NSRDB_DATA_DIR / 'albuquerque_multiyear_manifest.json'
DEFAULT_WEATHER_DATASET_DIR = Path('/home/user01/weather_dataset/weather_dataset')

DEFAULT_LATITUDE = 35.08
DEFAULT_LONGITUDE = -106.65
DEFAULT_TZ = 'UTC'

REQUIRED_COLUMNS = ('dni', 'ghi', 'dhi', 'temperature', 'wind_speed')
OPTIONAL_COLUMNS = (
    'relative_humidity',
    'cloud_type',
    'clearsky_ghi',
    'clearsky_dni',
    'clearsky_dhi',
    'solar_zenith_angle',
)
PSM3_ATTRIBUTES = (
    'air_temperature',
    'dhi',
    'dni',
    'ghi',
    'wind_speed',
)

NSRDB_API_HOST = 'developer.nrel.gov'
NSRDB_API_BASE = 'https://developer.nrel.gov'

YEAR_FILE_PATTERNS = (
    'albuquerque_{year}_utc_15min.csv',
    'albuquerque_{year}_utc_5min.csv',
    'nsrdb_{year}_utc_5min.csv',
    '*_{year}.csv',
    '*_{year}_*.csv',
)


def default_manifest_path():
    return str(DEFAULT_MANIFEST_PATH)


def _repo_relative(path):
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def normalize_psm3_dataframe(data, map_variables=True):
    """Map PSM3 / SAM columns to env column names."""
    frame = data.copy()
    rename = {
        'temp_air': 'temperature',
        'air_temperature': 'temperature',
        'Temperature': 'temperature',
        'Tamb': 'temperature',
        'wind_speed': 'wind_speed',
        'Wspd': 'wind_speed',
        'ghi': 'ghi',
        'GHI': 'ghi',
        'dhi': 'dhi',
        'DHI': 'dhi',
        'dni': 'dni',
        'DNI': 'dni',
        'ghi_clear': 'clearsky_ghi',
        'dni_clear': 'clearsky_dni',
        'dhi_clear': 'clearsky_dhi',
        'Clearsky GHI': 'clearsky_ghi',
        'Clearsky DNI': 'clearsky_dni',
        'Clearsky DHI': 'clearsky_dhi',
        'solar_zenith': 'solar_zenith_angle',
        'Solar Zenith Angle': 'solar_zenith_angle',
        'Cloud Type': 'cloud_type',
        'Relative Humidity': 'relative_humidity',
    }
    frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            'NSRDB data missing columns {} (have {})'.format(
                missing, sorted(frame.columns)))
    for col in REQUIRED_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    for col in OPTIONAL_COLUMNS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame['dni'] = frame['dni'].clip(lower=0.0)
    frame['ghi'] = frame['ghi'].clip(lower=0.0)
    frame['dhi'] = frame['dhi'].clip(lower=0.0)
    frame['wind_speed'] = frame['wind_speed'].clip(lower=0.0)
    return frame


def psm3_index_to_utc(index, metadata=None):
    """Convert PSM3 timestamps to UTC using metadata Time Zone when needed."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        return idx.tz_convert('UTC')
    tz_offset = None
    if metadata:
        tz_offset = metadata.get('Time Zone', metadata.get('local_time_zone'))
    if tz_offset is not None:
        try:
            offset_hours = float(tz_offset)
            return (idx - pd.to_timedelta(offset_hours, unit='h')).tz_localize('UTC')
        except (TypeError, ValueError):
            pass
    return idx.tz_localize('UTC')


def check_nsrdb_api_reachable(timeout=10):
    """Return (ok, message). Fails fast when DNS/network blocks developer.nrel.gov."""
    import socket
    try:
        socket.getaddrinfo(NSRDB_API_HOST, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return False, (
            'Cannot resolve {} ({!s}). This is a DNS/network issue on this machine, '
            'not an invalid API key. Fix outbound DNS or download SAM CSV files from '
            'the NSRDB Viewer on a networked machine and use '
            'prepare_nsrdb_multiyear_catalog.py --import-csv YEAR:PATH'.format(
                NSRDB_API_HOST, exc))
    try:
        import requests
        resp = requests.get(NSRDB_API_BASE, timeout=timeout)
        if resp.status_code >= 500:
            return False, 'NREL API returned HTTP {}'.format(resp.status_code)
    except Exception as exc:
        return False, 'Cannot reach {}: {!s}'.format(NSRDB_API_BASE, exc)
    return True, 'OK'


def fetch_nsrdb_year_utc(
        latitude,
        longitude,
        year,
        api_key,
        email,
        interval=15,
        timeout=300,
        leap_day=True,
        retries=3):
    """Download one calendar year of NSRDB PSM3 and return UTC-indexed DataFrame."""
    import time
    import requests

    ok, msg = check_nsrdb_api_reachable(timeout=min(30, timeout))
    if not ok:
        raise ConnectionError(msg)

    last_err = None
    for attempt in range(int(retries)):
        try:
            data, meta = get_psm3(
                latitude=latitude,
                longitude=longitude,
                api_key=api_key,
                email=email,
                names=str(int(year)),
                interval=int(interval),
                attributes=list(PSM3_ATTRIBUTES),
                map_variables=True,
                leap_day=leap_day,
                timeout=timeout,
            )
            data = normalize_psm3_dataframe(data, map_variables=False)
            data.index = psm3_index_to_utc(data.index, meta)
            data = data[~data.index.duplicated(keep='first')].sort_index()
            return data, meta
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_err = exc
            if attempt + 1 < retries:
                wait = 5 * (attempt + 1)
                print('[nsrdb] connection failed (attempt %d/%d), retry in %ds...' % (
                    attempt + 1, retries, wait))
                time.sleep(wait)
    raise ConnectionError(
        'NSRDB download failed after {} attempts for year {}. '
        'Last error: {!s}. If DNS fails, use --import-csv or fix network access to {}.'.format(
            retries, year, last_err, NSRDB_API_HOST))


def import_psm3_csv_to_year_utc(csv_path):
    """Load a local NSRDB SAM CSV (Viewer export or saved API file) as UTC yearly data."""
    path = _repo_relative(csv_path)
    if not path.is_file():
        raise FileNotFoundError('NSRDB CSV not found: {}'.format(path))
    data, meta = read_psm3(str(path), map_variables=True)
    data = normalize_psm3_dataframe(data, map_variables=False)
    data.index = psm3_index_to_utc(data.index, meta)
    drop_cols = [c for c in ('Year', 'Month', 'Day', 'Hour', 'Minute') if c in data.columns]
    if drop_cols:
        data = data.drop(columns=drop_cols)
    data = data[~data.index.duplicated(keep='first')].sort_index()
    return data, meta


def discover_year_csv_files(data_dir, years=None):
    """Find per-year NSRDB CSV files in a directory (SAM export naming)."""
    data_dir = _repo_relative(data_dir)
    if not data_dir.is_dir():
        return {}
    found = {}
    all_csv = sorted(data_dir.glob('*.csv'))
    for csv_path in all_csv:
        year_match = re.search(r'(?:^|_)(\d{4})(?:_|\.csv)', csv_path.name)
        if not year_match:
            continue
        year = int(year_match.group(1))
        if years is not None and year not in years:
            continue
        if year not in found:
            found[year] = str(csv_path)
    return found


def load_nsrdb_year_csv(path):
    """Load a prepared yearly UTC CSV (SAM export or catalog-prepared file)."""
    path = _repo_relative(path)
    if not path.is_file():
        raise FileNotFoundError('NSRDB year file not found: {}'.format(path))
    if path.suffix.lower() == '.csv':
        try:
            data, meta = read_psm3(str(path), map_variables=True)
            data = normalize_psm3_dataframe(data, map_variables=False)
            data.index = psm3_index_to_utc(data.index, meta)
            drop_cols = [c for c in ('Year', 'Month', 'Day', 'Hour', 'Minute') if c in data.columns]
            if drop_cols:
                data = data.drop(columns=drop_cols)
            return data[~data.index.duplicated(keep='first')].sort_index()
        except Exception as exc:
            last_err = exc
        else:
            last_err = None
        if last_err is not None:
            data = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
    else:
        raise ValueError('Unsupported NSRDB year file: {}'.format(path))
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError('NSRDB year file must have DatetimeIndex: {}'.format(path))
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    else:
        data.index = data.index.tz_convert('UTC')
    return normalize_psm3_dataframe(data, map_variables=False)


def resample_weather_to_episode_times(weather_utc, times):
    """Map native-interval NSRDB (e.g. 5-min UTC) onto the control episode grid (7min30s).

    Principle: agent samples at times[t]; weather W_e(t) is interpolated from the
    fixed scenario trajectory — not independent 5-min draws. pvlib uses the same
    times[t] for solar geometry. Energy integrates with dt = 7min30s (env.interval_hours).
    """
    times = pd.DatetimeIndex(times).tz_convert('UTC')
    subset = weather_utc.loc[
        (weather_utc.index >= times[0] - pd.Timedelta('1h'))
        & (weather_utc.index <= times[-1] + pd.Timedelta('1h'))
    ]
    if subset.empty:
        subset = weather_utc
    reindexed = subset.reindex(times, method='nearest', tolerance=pd.Timedelta('8min'))
    numeric_cols = [
        c for c in list(REQUIRED_COLUMNS) + list(OPTIONAL_COLUMNS)
        if c in subset.columns
    ]
    if reindexed[numeric_cols].isnull().any().any():
        reindexed[numeric_cols] = subset[numeric_cols].astype(float).interpolate(
            method='time', limit_direction='both').reindex(times)
    if reindexed[numeric_cols].isnull().any().any():
        missing_ts = reindexed[reindexed[numeric_cols].isnull().any(axis=1)].index[:3]
        raise ValueError(
            'NSRDB episode window has missing weather at {}'.format(
                [ts.isoformat() for ts in missing_ts]))
    return reindexed


def build_weather_profile_from_scenario(location, times, year_weather, scenario):
    """Build episode weather_profile for one manifest scenario row."""
    times = pd.DatetimeIndex(times).tz_convert('UTC')
    weather = resample_weather_to_episode_times(year_weather, times)
    weather = weather.copy()
    weather.index = times
    weather['condition'] = classify_weather_conditions(location, times, weather)
    return weather


def scenario_episode_date(scenario, tz=DEFAULT_TZ):
    """Calendar date for pvlib solar geometry (true data year)."""
    year = int(scenario.get('source_year', scenario['year']))
    return pd.Timestamp(
        year=year,
        month=int(scenario['month']),
        day=int(scenario['day']),
        tz=tz,
    )


def scenario_id(scenario):
    return scenario.get('scenario_id') or scenario.get('id') or '{year}-{month:02d}-{day:02d}'.format(
        year=int(scenario.get('source_year', scenario['year'])),
        month=int(scenario['month']),
        day=int(scenario['day']),
    )


def episode_timestamps_iso(date, start_time, periods, freq, tz=DEFAULT_TZ):
    """UTC ISO timestamp list for one episode grid."""
    times = pd.date_range(
        start='{} {}'.format(pd.Timestamp(date).date(), start_time),
        periods=int(periods),
        freq=freq,
        tz=tz,
    )
    return [ts.isoformat() for ts in times]


def load_scenario_manifest(manifest_path=None):
    """Load manifest JSON; returns dict with scenarios list and metadata."""
    manifest_path = _repo_relative(manifest_path or default_manifest_path())
    if not manifest_path.is_file():
        raise FileNotFoundError(
            'NSRDB scenario manifest not found: {}. Run '
            'scripts/prepare_nsrdb_multiyear_catalog.py'.format(manifest_path))
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)
    if 'scenarios' not in manifest:
        raise ValueError('Manifest must contain "scenarios" list: {}'.format(manifest_path))
    data_dir = manifest.get('data_dir', str(manifest_path.parent))
    data_dir = _repo_relative(data_dir)
    for row in manifest['scenarios']:
        sid = scenario_id(row)
        row['scenario_id'] = sid
        row['id'] = sid
        if 'source_year' not in row:
            row['source_year'] = int(row['year'])
        yf = row.get('file_path') or row.get('year_file')
        if yf and not os.path.isabs(yf):
            row['_year_path'] = str(_repo_relative(os.path.join(data_dir, yf)))
        elif yf:
            row['_year_path'] = str(Path(yf).resolve())
        else:
            yf = 'albuquerque_{year}_utc_15min.csv'.format(year=int(row['source_year']))
            row['_year_path'] = str(data_dir / yf)
    manifest['_data_dir'] = str(data_dir)
    manifest['_manifest_path'] = str(manifest_path)
    return manifest


class NsrdbYearCache:
    """Lazy per-year UTC weather loader."""

    def __init__(self, manifest):
        self._manifest = manifest
        self._cache = {}
        self._path_by_year = {}
        for row in manifest['scenarios']:
            year = int(row.get('source_year', row['year']))
            path = row.get('_year_path')
            if path and year not in self._path_by_year:
                self._path_by_year[year] = path

    def get_year(self, year):
        year = int(year)
        if year not in self._cache:
            path = self._path_by_year.get(year)
            if not path:
                rows = [s for s in self._manifest['scenarios'] if int(s.get('source_year', s['year'])) == year]
                if not rows:
                    raise KeyError('No scenarios for year {} in manifest'.format(year))
                path = rows[0]['_year_path']
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    'NSRDB year file missing: {} (year {})'.format(path, year))
            self._cache[year] = load_nsrdb_year_csv(path)
        return self._cache[year]


def index_scenarios(manifest):
    """Return scenarios list and lookup dicts."""
    scenarios = list(manifest['scenarios'])
    by_id = {scenario_id(s): s for s in scenarios}
    by_mday = {}
    for s in scenarios:
        key = (int(s['month']), int(s['day']))
        by_mday.setdefault(key, []).append(s)
    return scenarios, by_id, by_mday


def allowed_month_days(start_date, end_date, tz=DEFAULT_TZ):
    """Month-day pairs from a calendar range (label year ignored for NSRDB)."""
    days = pd.date_range(
        start=pd.Timestamp(start_date, tz=tz),
        end=pd.Timestamp(end_date, tz=tz),
        freq='D',
    )
    return {(int(d.month), int(d.day)) for d in days}


def filter_scenarios_by_date_range(scenarios, start_date, end_date, tz=DEFAULT_TZ):
    """Keep scenarios whose (month, day) appears in [start_date, end_date] calendar window."""
    allowed = allowed_month_days(start_date, end_date, tz=tz)
    return [
        s for s in scenarios
        if (int(s['month']), int(s['day'])) in allowed
    ]


def _episode_has_daylight(weather_profile, min_ghi=1.0):
    return float(weather_profile['ghi'].max()) > min_ghi


def build_manifest_from_year_files(
        data_dir,
        years,
        latitude=DEFAULT_LATITUDE,
        longitude=DEFAULT_LONGITUDE,
        start_time='13:30',
        periods=79,
        freq='7min30s',
        episode_validate=True,
        location=None,
        year_file_map=None):
    """Scan year CSVs and emit scenario rows with complete episode windows."""
    from pvlib.location import Location

    data_dir = _repo_relative(data_dir)
    location = location or Location(latitude, longitude, tz=DEFAULT_TZ)
    year_file_map = year_file_map or discover_year_csv_files(data_dir, years=years)

    scenarios = []
    for year in sorted(int(y) for y in years):
        year_path = year_file_map.get(year)
        if not year_path:
            for pattern in YEAR_FILE_PATTERNS:
                candidate = data_dir / pattern.format(year=year)
                matches = sorted(glob.glob(str(candidate)))
                if matches:
                    year_path = matches[0]
                    break
        if not year_path or not os.path.isfile(year_path):
            continue
        year_path = str(Path(year_path).resolve())
        weather = load_nsrdb_year_csv(year_path)
        for month in range(1, 13):
            max_day = calendar.monthrange(year, month)[1]
            for day in range(1, max_day + 1):
                try:
                    anchor = pd.Timestamp(year=year, month=month, day=day, tz='UTC')
                except ValueError:
                    continue
                times = pd.date_range(
                    start='{} {}'.format(anchor.date(), start_time),
                    periods=int(periods),
                    freq=freq,
                    tz='UTC',
                )
                sid = '{}-{:02d}-{:02d}'.format(year, month, day)
                row = {
                    'scenario_id': sid,
                    'id': sid,
                    'source_year': year,
                    'year': year,
                    'month': month,
                    'day': day,
                    'timestamps': episode_timestamps_iso(
                        anchor, start_time, periods, freq, tz='UTC'),
                    'valid_step_count': int(periods),
                    'file_path': year_path,
                    'year_file': os.path.basename(year_path),
                }
                if episode_validate:
                    try:
                        prof = resample_weather_to_episode_times(weather, times)
                    except ValueError:
                        continue
                    if prof[list(REQUIRED_COLUMNS)].isnull().any().any():
                        continue
                    if not _episode_has_daylight(prof):
                        continue
                scenarios.append(row)
    return scenarios


def episode_weather_diagnostics(weather_profile):
    """Diffuse/cloud diagnostics from a fixed episode weather trajectory."""
    ghi = weather_profile['ghi'].astype(float).values
    dhi = weather_profile['dhi'].astype(float).values
    dni = weather_profile['dni'].astype(float).values
    ghi_sum = float(np.sum(ghi))
    dhi_sum = float(np.sum(dhi))
    dni_sum = float(np.sum(dni))
    diffuse_fraction = dhi_sum / ghi_sum if ghi_sum > 1e-6 else 0.0
    dni_fraction = dni_sum / (dni_sum + dhi_sum + 1e-6)
    out = {
        'diffuse_fraction': diffuse_fraction,
        'dni_fraction': dni_fraction,
        'ghi_sum_wm2_min': ghi_sum,
        'dhi_sum_wm2_min': dhi_sum,
        'dni_sum_wm2_min': dni_sum,
    }
    if 'cloud_type' in weather_profile.columns:
        ct = weather_profile['cloud_type'].dropna()
        if len(ct):
            out['cloud_type_mode'] = float(ct.mode().iloc[0])
    if 'clearsky_ghi' in weather_profile.columns:
        cs = weather_profile['clearsky_ghi'].astype(float).sum()
        out['clearsky_ratio'] = ghi_sum / cs if cs > 1e-6 else 0.0
    return out


def write_manifest(
        path,
        scenarios,
        data_dir,
        latitude=DEFAULT_LATITUDE,
        longitude=DEFAULT_LONGITUDE,
        start_time='13:30',
        periods=79,
        freq='7min30s',
        source='NSRDB_PSM3',
        extra_meta=None):
    """Write manifest JSON."""
    path = _repo_relative(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    years = sorted({int(s.get('source_year', s['year'])) for s in scenarios})
    payload = {
        'source': source,
        'weather_scenario_mode': 'nsrdb_multiyear',
        'site': {
            'latitude': float(latitude),
            'longitude': float(longitude),
            'tz': DEFAULT_TZ,
        },
        'episode': {
            'start_time': start_time,
            'periods': int(periods),
            'freq': freq,
            'action_steps': int(periods) - 1,
        },
        'data_dir': str(_repo_relative(data_dir)),
        'years': years,
        'num_scenarios': len(scenarios),
        'scenarios': scenarios,
    }
    if extra_meta:
        payload['meta'] = extra_meta
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return str(path)
