from pathlib import Path
import json
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]


def literal_assignment(path, name):
    source = (ROOT / path).read_text(encoding="utf-8")
    match = re.search(
        r'^\s*' + re.escape(name) + r"\s*=\s*(['\"])(.*?)\1\s*$|^\s*"
        + re.escape(name) + r"\s*=\s*(\d+)\s*$",
        source, re.MULTILINE)
    if not match:
        raise AssertionError(f'{name} literal assignment not found in {path}')
    return match.group(2) if match.group(1) else int(match.group(3))


def test_shared_episode_grid_is_195_transitions():
    path = 'examples/config/pv_tracking/_nsrdb_base.py'
    assert literal_assignment(path, 'EPISODE_START_TIME') == '08:45'
    assert literal_assignment(path, 'EPISODE_PERIODS') == 196
    assert literal_assignment(path, 'EPISODE_FREQ') == '5min'
    assert 'EPISODE_ACTION_STEPS = EPISODE_PERIODS - 1  # 195' in (
        ROOT / path).read_text(encoding='utf-8')


def test_environment_defaults_match_shared_grid():
    assert literal_assignment('mbpo/env/pv_tracking.py', 'DEFAULT_START_TIME') == '08:45'
    assert literal_assignment('mbpo/env/pv_tracking.py', 'DEFAULT_PERIODS') == 196
    assert literal_assignment('mbpo/env/nsrdb_weather.py', 'DEFAULT_EPISODE_START_TIME') == '08:45'
    assert literal_assignment('mbpo/env/nsrdb_weather.py', 'DEFAULT_EPISODE_PERIODS') == 196


def test_sac_adapter_derives_horizon_from_resolved_periods():
    source = (ROOT / 'SAC_workspace/src/environment/environment.py').read_text(encoding='utf-8')
    assert 'self.horizon != 117' not in source
    assert "expected_periods = int(env_kwargs.get('periods', self.env.periods))" in source
    assert 'expected_horizon = expected_periods - 1' in source
    assert 'timestamp_count != expected_periods' in source
    assert 'self.horizon != expected_horizon' in source


def test_compact_representation_is_still_six_features():
    source = (ROOT / 'SAC_workspace/src/representation/representation.py').read_text(encoding='utf-8')
    assert "FEATURE_NAMES = ('dni_norm', 'dhi_norm', 'power_norm', 'panel_tilt_norm', 'panel_azimuth_norm', 'tau')" in source
    profile = json.loads((ROOT / 'MBPO_workspace/representation_config.json').read_text(encoding='utf-8'))
    assert profile['statistical_display']['horizon'] == '195_native_five_minute_control_steps'
    config = (ROOT / 'examples/config/pv_tracking/conf_compact.py').read_text(encoding='utf-8')
    assert 'compact six-feature state space' in config


def test_current_mbpo_evaluation_is_horizon_agnostic():
    evaluation = (ROOT / 'MBPO_workspace/src/evaluation.py').read_text(encoding='utf-8')
    advanced = (ROOT / 'MBPO_workspace/src/advanced_suite.py').read_text(encoding='utf-8')
    assert 'evaluated_horizon=len(first_path[\'power\'])' in evaluation
    assert 'len(rr)!=117' not in advanced
    assert 'np.arange(1,118)' not in advanced
    assert '117-step window' not in advanced


def test_dataset_manifest_and_runtime_join_adjacent_year_for_dec31():
    source = (ROOT / 'mbpo/env/nsrdb_weather.py').read_text(encoding='utf-8')
    runtime = (ROOT / 'mbpo/env/pv_tracking.py').read_text(encoding='utf-8')
    assert 'episode_end_year > year' in source
    assert 'episode_weather = pd.concat([weather, next_year_weather])' in source
    assert "candidate = data_dir / pattern.format(year=year)" in source
    assert 'next_year_weather = self._nsrdb_year_cache.get_year(int(times[-1].year))' in runtime


def test_window_math():
    # 08:45 -> 01:00 next date = 16h15 = 975 minutes = 195 transitions.
    start = 8 * 60 + 45
    end = 24 * 60 + 60
    assert end - start == 975
    assert (end - start) // 5 == 195
    assert 195 + 1 == 196


def test_fixed_window_covers_solar_daylight_in_all_seasons():
    pd = pytest.importorskip('pandas')
    pytest.importorskip('pvlib')
    from pvlib.location import Location

    location = Location(40.72, -74.01, tz='UTC', altitude=12.0)
    dates = ('2020-03-20', '2020-06-21', '2020-09-22', '2020-12-21')
    for date in dates:
        midnight = pd.Timestamp(date, tz='UTC')
        window_start = midnight + pd.Timedelta(hours=8, minutes=45)
        window_end = window_start + pd.Timedelta(minutes=975)

        # Start at the episode boundary. Midnight UTC can still be the
        # previous local evening in New York and must not count as this day.
        solar_day_end = window_start + pd.Timedelta(days=1)
        scan = pd.date_range(window_start, solar_day_end, freq='5min')
        elevation = location.get_solarposition(scan)['apparent_elevation'].to_numpy()
        daylight = scan[elevation > 0.0]

        # Confirm the episode begins before sunrise and ends after sunset.
        assert elevation[0] <= 0.0, f'{date}: daylight begins before the episode'
        assert len(daylight) > 0, f'{date}: no daylight found in the target solar day'
        assert daylight[0] > window_start, f'{date}: sunrise is outside the episode'
        assert daylight[-1] <= window_end, f'{date}: sunset is after the episode end'
        assert elevation[-1] <= 0.0, f'{date}: daylight remains after the episode window'


def test_physical_clock_decoder_rolls_over_midnight():
    np = pytest.importorskip('numpy')
    pd = pytest.importorskip('pandas')
    pytest.importorskip('pvlib')
    from pvlib.location import Location
    from mbpo.static.pv_tracking import StaticFns

    location = Location(40.72, -74.01, tz='UTC', altitude=0.0)
    stamp = pd.Timestamp('2020-06-22 00:30', tz='UTC')
    solar = location.get_solarposition(pd.DatetimeIndex([stamp])).iloc[0]
    obs = np.zeros(12, dtype=np.float32)
    obs[0] = float(solar['zenith']) / 180.0
    az = np.deg2rad(float(solar['azimuth']))
    obs[1], obs[2] = np.sin(az), np.cos(az)
    hour = StaticFns.time_of_day_from_solar_obs(
        obs, reference_date='2020-06-21')
    assert hour == pytest.approx(24.5, abs=5.0 / 60.0)
