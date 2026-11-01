#!/usr/bin/env python3
"""Apply the minimal fixed full-daylight UTC episode-window update.

Target: hossein-omidi/mbpo_custom, PVwatt_dev source layout.
This updater is deliberately strict: it checks every expected old block before
writing any file. It does not modify checkpoints, NSRDB CSVs, or manifests.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str, edits: dict[str, str]) -> None:
    if path not in edits:
        source = (ROOT / path).read_text(encoding="utf-8")
    else:
        source = edits[path]
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one matching block in {path}, found {count}. "
            "No files have been written. Check that this is the reviewed PVwatt_dev source."
        )
    edits[path] = source.replace(old, new, 1)


def build_edits() -> dict[str, str]:
    edits: dict[str, str] = {}

    replace_once(
        "examples/config/pv_tracking/_nsrdb_base.py",
        """EPISODE_TZ = 'UTC'
# Daylight window centred on NYC solar noon (~16:56 UTC): 12:00–21:45 UTC.
EPISODE_START_TIME = '12:00'
EPISODE_PERIODS = 118
EPISODE_FREQ = '5min'
EPISODE_ACTION_STEPS = EPISODE_PERIODS - 1  # 117
""",
        """EPISODE_TZ = 'UTC'
# Fixed UTC envelope covers NYC apparent daylight across the annual cycle.
# 08:45 UTC through 01:00 UTC on the next date: 16 h 15 min.
# 196 timestamps at 5-min spacing = 195 control transitions.
EPISODE_START_TIME = '08:45'
EPISODE_PERIODS = 196
EPISODE_FREQ = '5min'
EPISODE_ACTION_STEPS = EPISODE_PERIODS - 1  # 195
""",
        edits,
    )

    replace_once(
        "mbpo/env/nsrdb_weather.py",
        """Each scenario e = (data_year, calendar month, day) maps to one fixed irradiance
trajectory W_e on the project UTC episode grid (12:00–21:45 UTC, native 5min).
""",
        """Each scenario e = (data_year, calendar month, day) maps to one fixed irradiance
trajectory W_e on the project UTC episode grid (08:45 UTC through 01:00 UTC
on the next date, native 5min).
""",
        edits,
    )
    replace_once(
        "mbpo/env/nsrdb_weather.py",
        """# Project episode window (UTC), centred on NYC solar noon (~16:56 UTC):
# 12:00–21:45 UTC = 118 timestamps / 117 transitions at 5 min.
DEFAULT_EPISODE_START_TIME = '12:00'
DEFAULT_EPISODE_PERIODS = 118
""",
        """# Fixed UTC envelope for NYC apparent daylight across the annual cycle:
# 08:45 UTC through 01:00 UTC on the next date = 196 timestamps / 195 transitions.
DEFAULT_EPISODE_START_TIME = '08:45'
DEFAULT_EPISODE_PERIODS = 196
""",
        edits,
    )

    replace_once(
        "mbpo/env/pv_tracking.py",
        """# Episode grid: periods timestamps at freq; env.step() count == periods - 1.
# Wall-clock labels use Location.tz (UTC) only — no civil-time conversion.
# Daylight window 12:00–21:45 UTC (NYC solar noon ~16:56 UTC):
# 585 min / 5 min = 117 transitions (118 timestamps).
# Step convention: action updates panel pose, then env advances to times[step_index];
# reward uses NSRDB weather + pvlib geometry at that same timestamp (post-action pose).
DEFAULT_START_TIME = '12:00'
DEFAULT_PERIODS = 118
DEFAULT_FREQ = '5min'
DEFAULT_EPISODE_STEPS = DEFAULT_PERIODS - 1  # 117 transitions per day
""",
        """# Episode grid: periods timestamps at freq; env.step() count == periods - 1.
# Wall-clock labels use Location.tz (UTC) only — no civil-time conversion.
# Fixed annual daylight envelope: 08:45 UTC through 01:00 UTC next date.
# 975 min / 5 min = 195 transitions (196 timestamps).
# Step convention: action updates panel pose, then env advances to times[step_index];
# reward uses NSRDB weather + pvlib geometry at that same timestamp (post-action pose).
DEFAULT_START_TIME = '08:45'
DEFAULT_PERIODS = 196
DEFAULT_FREQ = '5min'
DEFAULT_EPISODE_STEPS = DEFAULT_PERIODS - 1  # 195 transitions per episode
""",
        edits,
    )

    replace_once(
        "SAC_workspace/src/config/config.py",
        """    if (int(nsrdb['periods']) != 118 or nsrdb['start_time'] != '12:00'
            or nsrdb['freq'] != '5min' or nsrdb['tz'] != 'UTC'
            or int(eval_env['periods']) != 118 or eval_env['start_time'] != '12:00'
            or eval_env['freq'] != '5min' or eval_env['tz'] != 'UTC'):
        raise ValueError('Reference episode grid changed; verify horizon before proceeding')
""",
        """    expected_start = str(_constant(_BASE, 'EPISODE_START_TIME'))
    expected_periods = int(_constant(_BASE, 'EPISODE_PERIODS'))
    expected_freq = str(_constant(_BASE, 'EPISODE_FREQ'))
    expected_tz = str(_constant(_BASE, 'EPISODE_TZ'))
    for env_name, env in (('training', nsrdb), ('evaluation', eval_env)):
        if (int(env['periods']) != expected_periods
                or env['start_time'] != expected_start
                or env['freq'] != expected_freq or env['tz'] != expected_tz):
            raise ValueError(
                '{} episode grid differs from shared NSRDB reference: '
                'start_time={!r}, periods={!r}, freq={!r}, tz={!r}'.format(
                    env_name, env['start_time'], env['periods'], env['freq'], env['tz']))
""",
        edits,
    )

    # The manifest builder must join the following UTC year for Dec-31 episodes.
    replace_once(
        "mbpo/env/nsrdb_weather.py",
        """    scenarios = []
    for year in sorted(int(y) for y in years):
""",
        """    scenarios = []
    missing_boundary_scenarios = []
    for year in sorted(int(y) for y in years):
""",
        edits,
    )
    replace_once(
        "mbpo/env/nsrdb_weather.py",
        """                sid = '{}-{:02d}-{:02d}'.format(year, month, day)
                row = {
""",
        """                sid = '{}-{:02d}-{:02d}'.format(year, month, day)
                episode_weather = weather
                episode_end_year = int(times[-1].year)
                if episode_end_year > year:
                    adjacent_path = year_file_map.get(episode_end_year)
                    if not adjacent_path:
                        adjacent_map = discover_year_csv_files(
                            data_dir, years=[episode_end_year])
                        adjacent_path = adjacent_map.get(episode_end_year)
                    if not adjacent_path or not os.path.isfile(adjacent_path):
                        missing_boundary_scenarios.append(sid)
                        continue
                    next_year_weather = load_nsrdb_year_csv(adjacent_path)
                    episode_weather = pd.concat([weather, next_year_weather]).sort_index()
                    episode_weather = episode_weather[~episode_weather.index.duplicated(keep='last')]
                row = {
""",
        edits,
    )
    replace_once(
        "mbpo/env/nsrdb_weather.py",
        """                        prof = resample_weather_to_episode_times(weather, times)
""",
        """                        prof = resample_weather_to_episode_times(episode_weather, times)
""",
        edits,
    )
    replace_once(
        "mbpo/env/nsrdb_weather.py",
        """                scenarios.append(row)
    return scenarios
""",
        """                    scenarios.append(row)
    if missing_boundary_scenarios:
        print('[warning] omitted cross-year episodes without adjacent-year NSRDB data: {}'.format(
            ', '.join(missing_boundary_scenarios)))
    return scenarios
""",
        edits,
    )

    # Runtime episode loading must use the same adjacent-year rows as the manifest builder.
    replace_once(
        "mbpo/env/pv_tracking.py",
        """            year = scenario_source_year(self._current_scenario)
            year_weather = self._nsrdb_year_cache.get_year(year)
            weather = build_weather_profile_from_scenario(
                self.location, times, year_weather, self._current_scenario)
""",
        """            year = scenario_source_year(self._current_scenario)
            year_weather = self._nsrdb_year_cache.get_year(year)
            if len(times) and int(times[-1].year) > year:
                next_year_weather = self._nsrdb_year_cache.get_year(int(times[-1].year))
                year_weather = pd.concat([year_weather, next_year_weather]).sort_index()
                year_weather = year_weather[~year_weather.index.duplicated(keep='last')]
            weather = build_weather_profile_from_scenario(
                self.location, times, year_weather, self._current_scenario)
""",
        edits,
    )

    # Allow adjacent-year boundary data to be loaded even when that year is support-only.
    replace_once(
        "mbpo/env/nsrdb_weather.py",
        """            path = self._path_by_year.get(year)
            if not path:
                rows = [s for s in self._manifest['scenarios'] if int(s.get('source_year', s['year'])) == year]
                if not rows:
                    raise KeyError('No scenarios for year {} in manifest'.format(year))
                path = rows[0]['_year_path']
""",
        """            path = self._path_by_year.get(year)
            if not path:
                data_dir = Path(self._manifest.get('_data_dir', self._manifest.get('data_dir', '')))
                for pattern in YEAR_FILE_PATTERNS:
                    candidate = data_dir / pattern.format(year=year)
                    if candidate.is_file():
                        path = str(candidate)
                        break
            if not path:
                rows = [s for s in self._manifest['scenarios'] if int(s.get('source_year', s['year'])) == year]
                if not rows:
                    raise KeyError('No NSRDB year file for {} in manifest/data_dir'.format(year))
                path = rows[0]['_year_path']
""",
        edits,
    )

    # Keep the standalone evaluator's plotted time axis continuous across midnight.
    replace_once(
        "scripts/eval_utils.py",
        """    return [(t.hour + t.minute / 60.0 + t.second / 3600.0) for t in times[1:H+1]]
""",
        """    start_hour = DEFAULT_START_HOUR
    return [start_hour + (i + 1) * DEFAULT_STEP_HOURS for i in range(H)]
""",
        edits,
    )
    replace_once(
        "scripts/eval_utils.py",
        """    \"\"\"Force evaluation to use UTC daylight grid (12:00-21:45 UTC).\"\"\"
""",
        """    \"\"\"Force evaluation to use the shared fixed UTC daylight envelope.\"\"\"
""",
        edits,
    )

    # Make the detailed MBPO/SAC comparison exporter horizon-agnostic.
    replace_once(
        "MBPO_workspace/src/advanced_suite.py",
        """    Every complete 117-step method-day-seed path is checked before export.
""",
        """    Every complete method-day-seed path is checked against the resolved episode horizon.
""",
        edits,
    )
    replace_once(
        "MBPO_workspace/src/advanced_suite.py",
        """    seen=set()
    with fp.open('w',newline='',encoding='utf-8') as f:
""",
        """    seen=set()
    expected_horizon=None
    with fp.open('w',newline='',encoding='utf-8') as f:
""",
        edits,
    )
    replace_once(
        "MBPO_workspace/src/advanced_suite.py",
        """            if any(len(p[z])!=117 for z in ('infos','power','energy','reward','cost','tilt','azimuth','action_tilt','action_azimuth','delta_tilt_deg','delta_azimuth_deg')):
                raise ValueError('Detailed export requires 117 complete steps: '+str(key))
""",
        """            horizon=len(p['infos'])
            if expected_horizon is None:
                expected_horizon=horizon
            if horizon!=expected_horizon or any(len(p[z])!=horizon for z in ('power','energy','reward','cost','tilt','azimuth','action_tilt','action_azimuth','delta_tilt_deg','delta_azimuth_deg')):
                raise ValueError('Detailed export requires one consistent complete episode horizon: '+str(key))
""",
        edits,
    )
    replace_once(
        "MBPO_workspace/src/advanced_suite.py",
        """    keys=defaultdict(dict); clocks={}
    for (sid,seed,method),rr in grouped.items():
        if len(rr)!=117 or [int(row['step']) for row in rr]!=list(range(1,118)):
            raise ValueError('Expected 117 ordered steps per path: '+str((sid,seed,method)))
        dt=[_time(r['utc_iso']) for r in rr]
        if any(abs((b-a).total_seconds()-300)>1e-5 for a,b in zip(dt,dt[1:])):
            raise ValueError('Non-native five-minute clock: '+str((sid,seed,method)))
        h=np.array([x.hour+x.minute/60. for x in dt])
        if not np.isclose(h[0],12.+5./60.) or not np.isclose(h[-1],21.75):
            raise ValueError('Unsupported control window')
""",
        """    keys=defaultdict(dict); clocks={}; expected_horizon=None; expected_clock=None
    for (sid,seed,method),rr in grouped.items():
        horizon=len(rr)
        if expected_horizon is None:
            expected_horizon=horizon
        if horizon!=expected_horizon or [int(row['step']) for row in rr]!=list(range(1,horizon+1)):
            raise ValueError('Expected one consistent ordered episode horizon: '+str((sid,seed,method)))
        dt=[_time(r['utc_iso']) for r in rr]
        if any(abs((b-a).total_seconds()-300)>1e-5 for a,b in zip(dt,dt[1:])):
            raise ValueError('Non-native five-minute clock: '+str((sid,seed,method)))
        clock=[(x.hour,x.minute,x.second) for x in dt]
        if expected_clock is None:
            expected_clock=clock
        elif clock!=expected_clock:
            raise ValueError('Different UTC clock grid across evaluation paths')
""",
        edits,
    )
    replace_once(
        "MBPO_workspace/src/advanced_suite.py",
        """    hours=np.arange(1,118)*5./60.+12.
""",
        """    sample_path=next(iter(grouped.values()))
    sample_times=[_time(r['utc_iso']) for r in sample_path]
    episode_steps=len(sample_times)
    episode_start_hour=(sample_times[0].hour+sample_times[0].minute/60.)-5./60.
    hours=episode_start_hour+np.arange(1,episode_steps+1)*5./60.
""",
        edits,
    )
    replace_once(
        "MBPO_workspace/src/advanced_suite.py",
        """                    ww.extend([weights[sid]/(len(seedrr)*117.)]*117)
""",
        """                    ww.extend([weights[sid]/(len(seedrr)*episode_steps)]*episode_steps)
""",
        edits,
    )
    replace_once(
        "MBPO_workspace/src/advanced_suite.py",
        """                    ax.set(ylabel='Weighted mean '+('gross' if metric.startswith('gross') else 'net')+
                           ' energy (kWh / 117-step window)',title=heading+' (no sampling CI)')
""",
        """                    ax.set(ylabel='Weighted mean '+('gross' if metric.startswith('gross') else 'net')+
                           ' energy (kWh / {}-step window)'.format(episode_steps),
                           title=heading+' (no sampling CI)')
""",
        edits,
    )

    # Remove stale fixed-horizon wording from the paired evaluation provenance.
    replace_once(
        "MBPO_workspace/src/evaluation.py",
        """    out={'protocol':('Paired by NSRDB scenario, seed, initial pose and post-step UTC timestamps; '
           'real PVTrackingEnv; 117 transitions; no model-generated actions in evaluation'),
""",
        """    first_group=next(iter(keys.values()))
    first_path=next(iter(first_group.values()))
    evaluated_horizon=len(first_path['power'])
    out={'protocol':('Paired by NSRDB scenario, seed, initial pose and post-step UTC timestamps; '
           'real PVTrackingEnv; {} transitions; no model-generated actions in evaluation').format(evaluated_horizon),
""",
        edits,
    )

    replace_once(
        "MBPO_workspace/src/trainer.py",
        """        if len(episode_rewards)!=self.env.horizon or not done:
            raise RuntimeError('Incomplete 117-transition solar episode')
""",
        """        if len(episode_rewards)!=self.env.horizon or not done:
            raise RuntimeError('Incomplete {}-transition solar episode'.format(self.env.horizon))
""",
        edits,
    )

    # Stale prose only; the episode count itself is derived from the shared constants.
    replace_once(
        "scripts/prepare_nsrdb_newyork_catalog.py",
        """  4. Build scenario manifest (episode window 12:00-21:45 UTC, 118 stamps).
""",
        """   4. Build scenario manifest (episode window 08:45 UTC through 01:00 UTC next day, 196 stamps).
""",
        edits,
    )

    return edits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true',
                        help='Validate all source blocks without writing changes')
    args = parser.parse_args()
    edits = build_edits()
    if args.check_only:
        print('PASS: all expected PVwatt_dev source blocks found; no files changed.')
        return
    for rel, content in edits.items():
        target = ROOT / rel
        target.write_text(content, encoding='utf-8')
        print('UPDATED', rel)
    print('Episode window updated: 08:45 UTC to 01:00 UTC next day; 196 timestamps; 195 transitions.')
    print('Next required step: regenerate the NSRDB manifest with scripts/prepare_nsrdb_newyork_catalog.py.')


if __name__ == '__main__':
    main()
