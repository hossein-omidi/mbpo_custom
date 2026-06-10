"""NSRDB PSM3 multi-year weather scenarios for PVTracking.

Each scenario e = (data_year, calendar month, day) maps to one fixed irradiance
trajectory W_e on the project UTC episode grid (13:30–23:15, native 5min).

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

from pvlib.iotools import get_psm3

from .nsrdb_iotools import read_nsrdb_csv_to_env_weather, to_env_weather_frame

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

# Canonical offline SAM files (5-min UTC). Legacy 15-min names are ignored when 5-min exists.
YEAR_FILE_PATTERNS = (
    'nsrdb_{year}_utc_5min.csv',
    'albuquerque_{year}_utc_5min.csv',
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
            data = to_env_weather_frame(data, meta)
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
    """Load a local NSRDB SAM CSV via pvlib.iotools (read_nsrdb_psm4 / read_psm3)."""
    path = _repo_relative(csv_path)
    if not path.is_file():
        raise FileNotFoundError('NSRDB CSV not found: {}'.format(path))
    return read_nsrdb_csv_to_env_weather(str(path))


def discover_year_csv_files(data_dir, years=None):
    """Find per-year NSRDB 5-min SAM CSV files (prefers nsrdb_{year}_utc_5min.csv)."""
    data_dir = _repo_relative(data_dir)
    if not data_dir.is_dir():
        return {}
    found = {}
    if years is not None:
        year_list = [int(y) for y in years]
    else:
        year_list = None
        year_set = set()
        for csv_path in data_dir.glob('*.csv'):
            m = re.search(r'(?:^|_)(\d{4})(?:_|\.csv)', csv_path.name)
            if m:
                year_set.add(int(m.group(1)))
        year_list = sorted(year_set)
    for year in year_list:
        for pattern in YEAR_FILE_PATTERNS:
            matches = sorted(glob.glob(str(data_dir / pattern.format(year=year))))
            if matches:
                found[int(year)] = str(Path(matches[0]).resolve())
                break
    return found


def load_nsrdb_year_csv(path):
    """Load a prepared yearly UTC CSV via pvlib NSRDB readers (standard path)."""
    path = _repo_relative(path)
    if not path.is_file():
        raise FileNotFoundError('NSRDB year file not found: {}'.format(path))
    if path.suffix.lower() != '.csv':
        raise ValueError('Unsupported NSRDB year file: {}'.format(path))
    weather, _meta = read_nsrdb_csv_to_env_weather(str(path))
    validate_nsrdb_year_weather(weather, path=str(path))
    return weather


def validate_nsrdb_year_weather(weather, path=''):
    """Validate UTC 5-min NSRDB frame after pvlib reader (raises on failure)."""
    if not isinstance(weather.index, pd.DatetimeIndex):
        raise ValueError('NSRDB weather must have DatetimeIndex: {}'.format(path))
    if weather.index.tz is None:
        raise ValueError('NSRDB weather index must be timezone-aware UTC: {}'.format(path))
    if str(weather.index.tz) != 'UTC':
        raise ValueError('NSRDB weather index must be UTC: {}'.format(path))
    missing = [c for c in REQUIRED_COLUMNS if c not in weather.columns]
    if missing:
        raise ValueError('NSRDB missing columns {} in {}'.format(missing, path))
    if len(weather) < 2:
        raise ValueError('NSRDB year file too short: {}'.format(path))
    deltas = weather.index.to_series().diff().dropna()
    med_min = deltas.median().total_seconds() / 60.0
    if abs(med_min - 5.0) > 0.6:
        raise ValueError(
            'NSRDB native spacing {:.2f} min (expected ~5) in {}'.format(med_min, path))
    for col in ('ghi', 'dni', 'dhi'):
        vals = weather[col].astype(float)
        if vals.isnull().any():
            raise ValueError('NSRDB {} has NaNs in {}'.format(path, col))
        if (vals < 0).any():
            raise ValueError('NSRDB {} has negative {}'.format(path, col))
    return True


def diagnostic_weather_label(location, times, weather_frame):
    """Optional clear/partly_cloudy/overcast label for diagnostics only (not used in power)."""
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


def resample_weather_to_episode_times(weather_utc, times):
    """Align NSRDB 5-min UTC rows to episode timestamps (hard sync, no interpolation).

    One RL action interval = one NSRDB weather row = one 5-minute timestamp.
    Agent orientation at t uses weather and pvlib solar geometry at the same t.
    """
    times = pd.DatetimeIndex(times).tz_convert('UTC')
    if len(times) >= 2:
        step_min = times.to_series().diff().dropna().median().total_seconds() / 60.0
        if abs(step_min - 5.0) > 0.01:
            raise ValueError(
                'NSRDB native path requires 5-min episode grid; got {:.2f} min spacing'.format(
                    step_min))
    missing = times.difference(weather_utc.index)
    if len(missing):
        raise ValueError(
            'NSRDB episode missing exact 5-min rows at {}'.format(
                [ts.isoformat() for ts in missing[:5]]))
    aligned = weather_utc.loc[times].copy()
    numeric_cols = [
        c for c in list(REQUIRED_COLUMNS) + list(OPTIONAL_COLUMNS)
        if c in aligned.columns
    ]
    if aligned[numeric_cols].isnull().any().any():
        bad = aligned[aligned[numeric_cols].isnull().any(axis=1)].index[:3]
        raise ValueError(
            'NSRDB episode has NaN weather at {}'.format(
                [ts.isoformat() for ts in bad]))
    return aligned


def build_weather_profile_from_scenario(location, times, year_weather, scenario):
    """Build episode weather_profile for one manifest scenario row."""
    times = pd.DatetimeIndex(times).tz_convert('UTC')
    weather = resample_weather_to_episode_times(year_weather, times)
    weather = weather.copy()
    weather.index = times
    weather['condition'] = diagnostic_weather_label(location, times, weather)
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


def scenario_source_year(scenario):
    """NSRDB data-file year for a manifest row (may differ from label year)."""
    return int(scenario.get('source_year', scenario['year']))


def scenario_id(scenario):
    return scenario.get('scenario_id') or scenario.get('id') or '{year}-{month:02d}-{day:02d}'.format(
        year=scenario_source_year(scenario),
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
            yf = 'nsrdb_{year}_utc_5min.csv'.format(year=int(row['source_year']))
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


def scenario_row_weight(scenario):
    """Per-scenario sampling weight from manifest row, if present."""
    for key in ('probability', 'weight', 'sample_weight'):
        if key in scenario and scenario[key] is not None:
            try:
                value = float(scenario[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value >= 0.0:
                return value
    return 1.0


def normalized_scenario_probabilities(scenarios):
    """Empirical scenario distribution p(e_i) = w_i / sum_j w_j; uniform if no weights."""
    if not scenarios:
        return np.array([], dtype=np.float64)
    weights = np.array(
        [scenario_row_weight(s) for s in scenarios], dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        weights = np.ones(len(scenarios), dtype=np.float64)
        total = float(len(scenarios))
    probs = weights / total
    return probs / probs.sum()


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
        periods=118,
        freq='5min',
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
                    'date': anchor.date().isoformat(),
                    'start_time': start_time,
                    'end_time': times[-1].strftime('%H:%M'),
                    'timestamps': episode_timestamps_iso(
                        anchor, start_time, periods, freq, tz='UTC'),
                    'valid_step_count': int(periods),
                    'file_path': os.path.basename(year_path),
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
        periods=118,
        freq='5min',
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
