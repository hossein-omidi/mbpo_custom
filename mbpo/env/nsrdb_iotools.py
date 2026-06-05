"""Canonical local NSRDB CSV ingestion via pvlib.iotools (PSM3 / PSM4).

Matches the verified workflow in tests/test1.py:
  read_nsrdb_psm4(path) when available, else read_psm3(path, map_variables=True)
  → UTC DatetimeIndex → env column names (temperature, ghi, dni, dhi, wind_speed).

Episode control grid (native 5min) is aligned later by nsrdb_weather.resample_weather_to_episode_times.
"""

from __future__ import print_function

import pandas as pd
import pvlib

from pvlib.iotools import read_psm3

REQUIRED_IRRADIANCE = ('ghi', 'dni', 'dhi')


def unpack_pvlib_reader_result(result):
    """Accept (data, meta) or legacy (meta, data) return order."""
    a, b = result
    if isinstance(a, pd.DataFrame):
        return a, b
    return b, a


def read_nsrdb_file(path, map_variables=True):
    """Read a local NSRDB SAM CSV with the best available pvlib reader.

    Returns
    -------
    data : pd.DataFrame
        Raw pvlib-mapped frame (DatetimeIndex, ghi/dni/dhi/...).
    meta : dict
        NSRDB header metadata (latitude, longitude, Time Zone, ...).
    reader_name : str
        Which pvlib function succeeded ('read_nsrdb_psm4' or 'read_psm3').
    """
    path = str(path)
    errors = []

    read_psm4 = getattr(pvlib.iotools, 'read_nsrdb_psm4', None)
    if read_psm4 is not None:
        try:
            data, meta = unpack_pvlib_reader_result(read_psm4(path))
            return data, meta, 'read_nsrdb_psm4'
        except Exception as exc:
            errors.append(('read_nsrdb_psm4', str(exc)))

    try:
        data, meta = read_psm3(path, map_variables=map_variables)
        return data, meta, 'read_psm3'
    except Exception as exc:
        errors.append(('read_psm3', str(exc)))

    msg = 'Could not read NSRDB file {!r} with pvlib readers:\n'.format(path)
    for name, err in errors:
        msg += '  {}: {}\n'.format(name, err)
    raise RuntimeError(msg)


def psm3_index_to_utc(index, metadata=None):
    """Normalize reader index to UTC (honour metadata Time Zone when naive)."""
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


def site_from_nsrdb_metadata(meta):
    """Extract lat/lon from pvlib NSRDB metadata dict."""
    lat = meta.get('latitude', meta.get('Latitude'))
    lon = meta.get('longitude', meta.get('Longitude'))
    elev = meta.get('Elevation', meta.get('elevation'))
    out = {}
    if lat is not None:
        out['latitude'] = float(lat)
    if lon is not None:
        out['longitude'] = float(lon)
    if elev is not None:
        try:
            out['altitude'] = float(elev)
        except (TypeError, ValueError):
            pass
    return out


def to_env_weather_frame(data, metadata=None):
    """Map pvlib NSRDB frame → env weather columns + UTC index."""
    from mbpo.env.nsrdb_weather import normalize_psm3_dataframe

    frame = data.copy()
    frame.index = psm3_index_to_utc(frame.index, metadata)
    drop_cols = [c for c in ('Year', 'Month', 'Day', 'Hour', 'Minute') if c in frame.columns]
    if drop_cols:
        frame = frame.drop(columns=drop_cols)
    frame = normalize_psm3_dataframe(frame, map_variables=False)
    frame = frame[~frame.index.duplicated(keep='first')].sort_index()
    return frame


def read_nsrdb_csv_to_env_weather(path, map_variables=True):
    """Full pipeline: pvlib reader → UTC env weather DataFrame + metadata."""
    data, meta, reader_name = read_nsrdb_file(path, map_variables=map_variables)
    weather = to_env_weather_frame(data, meta)
    meta = dict(meta)
    meta['_reader'] = reader_name
    meta['_site'] = site_from_nsrdb_metadata(meta)
    return weather, meta
