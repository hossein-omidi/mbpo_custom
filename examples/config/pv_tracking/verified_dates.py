"""Canonical PVGIS TMY episode dates (Albuquerque 35N/106W).

Verified in scripts/verify_pv_state_space.py against the bundled historical
catalog (data/pv_weather/albuquerque_pvgis_tmy_utc_15min.csv).

Power and cumulative energy for the agent and all baselines (sun_tracking,
fixed_no_motion, optional POA oracle) come from PVTrackingEnv.step →
mbpo.env.pvlib_physics.compute_poa_global → pvlib.irradiance.get_total_irradiance.
Baselines only choose panel orientation; they do not bypass the env physics path.
Sun tracker uses dual-axis zenith/az targets (not pvlib.tracking.singleaxis).
"""

from mbpo.env.historical_weather import available_month_days, load_historical_weather_catalog

# Episode-window characterization on the project UTC grid (13:30–23:15 UTC).
CLEAR_HISTORICAL_DAY = '2020-06-15'   # high GHI, ~0% overcast in window
CLOUDY_HISTORICAL_DAY = '2020-06-21'  # low GHI / high overcast in window
WINTER_HISTORICAL_DAY = '2020-12-07'  # winter sun geometry + historical irradiance

VERIFIED_HISTORICAL_EPISODE_DATES = (
    CLEAR_HISTORICAL_DAY,
    CLOUDY_HISTORICAL_DAY,
    WINTER_HISTORICAL_DAY,
)

# Stage 0 single-day proofs
STAGE0_CLEARSKY_DAY = CLEAR_HISTORICAL_DAY
STAGE0_CLOUDY_HISTORICAL_DAY = CLOUDY_HISTORICAL_DAY

# Stage 1 summer hold-out (clearsky training; dates on site calendar)
STAGE1_FIXED_EVAL_DATES = [
    '2020-06-07',
    CLEAR_HISTORICAL_DAY,
    '2020-07-15',
    '2020-08-01',
]

# Stage 2 summer hold-out (historical weather; same calendar spread)
STAGE2_FIXED_EVAL_DATES = list(STAGE1_FIXED_EVAL_DATES)

# Stage 3 in-training validation (seasonally spaced, historical)
STAGE3_VALIDATION_DATES = [
    '2020-02-15',
    '2020-05-15',
    '2020-08-15',
    '2020-11-15',
]

# Stage 3 final test hold-out (never sampled during training in clean_split)
STAGE3_FINAL_TEST_DATES = [
    '2020-01-15',
    '2020-03-20',
    CLOUDY_HISTORICAL_DAY,
    '2020-09-22',
    '2020-10-15',
    WINTER_HISTORICAL_DAY,
]

# Stage 3 full-year (RL annual support — NOT training exclusions)
STAGE3_CONFIG_MODULE = 'examples.config.pv_tracking.stage3_multiyear_nsrdb_scenario'

# Optional fixed-calendar stress tests (reporting / diagnostics only; not train exclusions).
STAGE3_STRESS_TEST_DATES = sorted(set(STAGE3_VALIDATION_DATES + STAGE3_FINAL_TEST_DATES))
# Backward-compatible alias (deprecated name: "holdout").
STAGE3_HOLDOUT_DATES = STAGE3_STRESS_TEST_DATES

SEASON_ORDER = ('winter', 'spring', 'summer', 'fall')


def season_of_date(date_str):
    """Map YYYY-MM-DD to env season label (same rules as PVTrackingEnv.step)."""
    import pandas as pd
    day = int(pd.Timestamp(date_str).dayofyear)
    if 80 <= day <= 171:
        return 'spring'
    if 172 <= day <= 263:
        return 'summer'
    if 264 <= day <= 354:
        return 'fall'
    return 'winter'


def dates_by_season(date_list):
    """Group date strings by season; values sorted."""
    from collections import defaultdict
    groups = defaultdict(list)
    for d in date_list:
        groups[season_of_date(d)].append(d)
    return {s: sorted(groups[s]) for s in SEASON_ORDER if groups[s]}


def training_day_count(start='2020-01-01', end='2020-12-31', excluded=None):
    """Number of calendar days in the training sampler after exclusions."""
    import pandas as pd
    excluded = set(excluded or [])
    days = pd.date_range(start, end, freq='D')
    return sum(1 for d in days if str(d.date()) not in excluded)


# Legacy six-date eval set (superseded by clean_split hold-outs)
STAGE3_FIXED_EVAL_DATES = [
    '2020-01-15',
    '2020-03-20',
    CLOUDY_HISTORICAL_DAY,
    '2020-08-01',
    '2020-10-15',
    '2020-12-21',
]


def assert_dates_in_historical_catalog(dates):
    """Raise ValueError if any calendar day is missing from the bundled TMY file."""
    import pandas as pd

    catalog = load_historical_weather_catalog()
    available = available_month_days(catalog)
    missing = []
    for d in dates:
        ts = pd.Timestamp(d)
        if (int(ts.month), int(ts.day)) not in available:
            missing.append(d)
    if missing:
        raise ValueError(
            'Dates not in historical weather catalog: %s (available month-days: %d)' % (
                missing, len(available)))

