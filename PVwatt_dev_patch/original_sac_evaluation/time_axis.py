"""UTC time-axis helpers for finite-horizon NSRDB episode plots."""
from __future__ import annotations
import numpy as np

def unwrapped_utc_hours(infos):
    """Return monotonically increasing UTC hours for post-action samples.

    Hours above 24 denote the following UTC date. This avoids plotting a
    full-day 08:45–01:00 episode as if its time axis ran backward at midnight.
    """
    from datetime import datetime, timedelta
    stamps = [datetime.fromisoformat(str(i['timestamp_utc_iso']).replace('Z', '+00:00'))
              for i in infos]
    if not stamps:
        raise ValueError('Cannot plot an empty evaluation trajectory')
    if any(t.utcoffset() != timedelta(0) for t in stamps):
        raise ValueError('Expected UTC-normalized post-action timestamps')
    deltas = np.asarray([(t - stamps[0]).total_seconds() for t in stamps], dtype=float)
    if not np.isfinite(deltas).all() or np.any(np.diff(deltas) <= 0):
        raise ValueError('Evaluation timestamps must be finite and strictly increasing')
    start_clock = stamps[0].hour + stamps[0].minute / 60. + stamps[0].second / 3600.
    return start_clock + deltas / 3600.

