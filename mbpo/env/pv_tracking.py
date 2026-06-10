import logging
import numpy as np
import gym
from gym import spaces
import pandas as pd

from pvlib.location import Location
from pvlib.irradiance import aoi

from mbpo.env.pvlib_physics import compute_panel_power_w

from .historical_weather import (
    available_month_days,
    build_weather_profile_from_catalog,
    default_historical_weather_file,
    load_historical_weather_catalog,
    using_default_site,
)
from .nsrdb_weather import (
    NsrdbYearCache,
    build_weather_profile_from_scenario,
    default_manifest_path,
    episode_weather_diagnostics,
    filter_scenarios_by_date_range,
    index_scenarios,
    load_scenario_manifest,
    normalized_scenario_probabilities,
    scenario_episode_date,
    scenario_id,
    scenario_source_year,
)

logger = logging.getLogger(__name__)

# Observation layouts (must stay in sync with mbpo/static/pv_tracking.py).
OBSERVATION_MODES = ('legacy', 'physical')
WEATHER_SOURCES = ('clearsky', 'historical', 'nsrdb_multiyear')
WEATHER_SCENARIO_MODES = ('pvgis_tmy', 'nsrdb_multiyear', 'clearsky')

LEGACY_OBS_LABELS = (
    'solar_zenith_norm',
    'solar_azimuth_sin',
    'solar_azimuth_cos',
    'dni_norm',
    'dhi_norm',
    'ghi_norm',
    'temperature_norm',
    'panel_tilt_norm',
    'panel_azimuth_sin',
    'panel_azimuth_cos',
    'power_norm',
    'time_of_day_sin',
    'time_of_day_cos',
    'day_of_year_sin',
    'day_of_year_cos',
)

PHYSICAL_OBS_LABELS = (
    'solar_zenith_norm',
    'solar_azimuth_sin',
    'solar_azimuth_cos',
    'dni_norm',
    'dhi_norm',
    'ghi_norm',
    'temperature_norm',
    'panel_tilt_norm',
    'panel_azimuth_sin',
    'panel_azimuth_cos',
    'cos_aoi',
)

IRRADIANCE_NORM = 2000.0
TEMPERATURE_NORM = 50.0
POWER_NORM = 2000.0

# Project time standard (pvlib Location.tz, info clocks, train/test/plots must match).
PV_TIMEZONE = 'UTC'

# Episode grid: periods timestamps at freq; env.step() count == periods - 1.
# Wall-clock labels use Location.tz (UTC) only — no civil-time conversion.
# Daylight window 13:30–23:15 UTC: 585 min / 5 min = 117 transitions (118 timestamps).
# Step convention: action updates panel pose, then env advances to times[step_index];
# reward uses NSRDB weather + pvlib geometry at that same timestamp (post-action pose).
DEFAULT_START_TIME = '13:30'
DEFAULT_PERIODS = 118
DEFAULT_FREQ = '5min'
DEFAULT_EPISODE_STEPS = DEFAULT_PERIODS - 1  # 117 transitions per day
DEFAULT_CONTROL_INTERVAL_MINUTES = 5.0


def episode_clock_hour(time_str):
    """Parse 'HH:MM' to fractional hour (UTC wall clock)."""
    hour, minute = map(int, str(time_str).split(':'))
    return hour + minute / 60.0


DEFAULT_START_HOUR = episode_clock_hour(DEFAULT_START_TIME)
_DEFAULT_STEP_HOURS = pd.Timedelta(DEFAULT_FREQ).total_seconds() / 3600.0
DEFAULT_STEP_HOURS = _DEFAULT_STEP_HOURS
DEFAULT_END_HOUR = DEFAULT_START_HOUR + (DEFAULT_PERIODS - 1) * DEFAULT_STEP_HOURS


class PVTrackingEnv(gym.Env):
    """RL scenario-sampling environment for single-day PV tracking episodes.

    Stochasticity is confined to ``reset()``: one historical day/scenario is drawn
    from the dataset (uniformly or via manifest weights). Within an episode the
    exogenous weather trajectory is fixed and deterministic conditional on that draw.
    Optional ``irradiance_perturbation_std`` / ``observation_noise_std`` apply
    episode-level augmentation at reset only (not per control step).
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        latitude=35.0,
        longitude=-106.0,
        altitude=1600.0,
        tz=PV_TIMEZONE,
        start_date='2020-01-01',
        end_date='2020-12-31',
        start_time=DEFAULT_START_TIME,
        periods=DEFAULT_PERIODS,
        freq=DEFAULT_FREQ,
        efficiency=0.18,
        area=1.0,
        max_delta_tilt=5.0,
        max_delta_azimuth=10.0,
        randomize_day=True,
        randomize_initial_orientation=True,
        weather_source='clearsky',
        weather_scenario_mode=None,
        weather_file=None,
        scenario_manifest=None,
        randomize_scenario=False,
        fixed_eval_scenarios=None,
        temperature=25.0,
        wind_speed=2.0,
        movement_penalty=0.01,
        fixed_eval_dates=None,
        excluded_dates=None,
        # Optional bounded irradiance augmentation (0 = paper default, deterministic profile per day).
        irradiance_perturbation_std=0.0,
        observation_noise_std=0.0,
        observation_mode='legacy',
        log_observations=False,
    ):
        self.location = Location(latitude, longitude, tz=tz, altitude=altitude)
        self.start_date = pd.Timestamp(start_date, tz=tz)
        self.end_date = pd.Timestamp(end_date, tz=tz)
        self.start_time = start_time
        self.periods = periods
        self.freq = freq
        self.efficiency = efficiency
        self.area = area
        self.max_delta_tilt = max_delta_tilt
        self.max_delta_azimuth = max_delta_azimuth
        self.randomize_day = randomize_day
        self.randomize_initial_orientation = randomize_initial_orientation
        if weather_scenario_mode is not None:
            mode = str(weather_scenario_mode).lower()
            if mode not in WEATHER_SCENARIO_MODES:
                raise ValueError(
                    'weather_scenario_mode must be one of {}, got {!r}.'.format(
                        WEATHER_SCENARIO_MODES, weather_scenario_mode))
            if mode == 'nsrdb_multiyear':
                weather_source = 'nsrdb_multiyear'
            elif mode == 'pvgis_tmy':
                weather_source = 'historical'
            elif mode == 'clearsky':
                weather_source = 'clearsky'
            self.weather_scenario_mode = mode
        else:
            if weather_source == 'nsrdb_multiyear':
                self.weather_scenario_mode = 'nsrdb_multiyear'
            elif weather_source == 'historical':
                self.weather_scenario_mode = 'pvgis_tmy'
            else:
                self.weather_scenario_mode = 'clearsky'
        self.weather_source = weather_source
        self.weather_file = weather_file
        self.scenario_manifest_path = scenario_manifest
        self.randomize_scenario = bool(randomize_scenario)
        self.temperature = temperature
        self.temperature_variation = 5.0
        self.wind_speed = wind_speed
        self.wind_speed_variation = 1.0
        self.movement_penalty = movement_penalty
        # Optional bounded augmentation on catalog irradiance (0 = deterministic weather(d)).
        self.irradiance_perturbation_std = float(irradiance_perturbation_std)
        self.observation_noise_std = float(observation_noise_std)
        if weather_source not in WEATHER_SOURCES:
            raise ValueError(
                'weather_source must be one of {}, got {!r}.'.format(
                    WEATHER_SOURCES, weather_source))
        if observation_mode not in OBSERVATION_MODES:
            raise ValueError(
                'observation_mode must be one of {}, got {!r}.'.format(
                    OBSERVATION_MODES, observation_mode))
        self.observation_mode = observation_mode
        self.log_observations = bool(log_observations)
        self._obs_log_steps_remaining = 0
        self.obs_feature_names = (
            PHYSICAL_OBS_LABELS
            if observation_mode == 'physical'
            else LEGACY_OBS_LABELS
        )
        self.fixed_eval_dates = None
        self.excluded_dates = None
        self._fixed_eval_date_index = 0
        self._rollout_seed = None
        self._eval_cycle_fixed_ids = False
        self._current_scenario = None
        self._nsrdb_year_cache = None
        self._nsrdb_scenarios = []
        self._nsrdb_by_id = {}
        self._nsrdb_by_mday = {}
        self._nsrdb_scenario_probs = None
        self._episode_obs_noise = None
        self.fixed_eval_scenarios = None
        self._fixed_eval_scenario_index = 0
        if fixed_eval_scenarios is not None:
            if isinstance(fixed_eval_scenarios, str):
                fixed_eval_scenarios = [fixed_eval_scenarios]
            self.fixed_eval_scenarios = list(fixed_eval_scenarios)
        if fixed_eval_dates is not None:
            if isinstance(fixed_eval_dates, (str, pd.Timestamp)):
                fixed_eval_dates = [fixed_eval_dates]
            self.fixed_eval_dates = [
                pd.Timestamp(date, tz=tz)
                for date in fixed_eval_dates
            ]
        if excluded_dates is not None:
            if isinstance(excluded_dates, (str, pd.Timestamp)):
                excluded_dates = [excluded_dates]
            self.excluded_dates = {
                pd.Timestamp(date, tz=tz).normalize()
                for date in excluded_dates
            }

        self.start_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='D',
            tz=tz,
        )
        self._historical_weather_catalog = None
        if self.weather_source == 'historical':
            if self.weather_file is None and not using_default_site(latitude, longitude):
                raise ValueError(
                    'Bundled historical weather is only available for the default '
                    'site (35.0, -106.0). Pass weather_file=... for other locations.')
            self._historical_weather_catalog = load_historical_weather_catalog(
                weather_file=self.weather_file or default_historical_weather_file(),
                freq=self.freq,
            )
            available = available_month_days(self._historical_weather_catalog)
            self.start_dates = pd.DatetimeIndex([
                date for date in self.start_dates
                if (int(date.month), int(date.day)) in available
            ])
        elif self.weather_source == 'nsrdb_multiyear':
            manifest_path = self.scenario_manifest_path or default_manifest_path()
            manifest = load_scenario_manifest(manifest_path)
            scenarios, by_id, by_mday = index_scenarios(manifest)
            scenarios = filter_scenarios_by_date_range(
                scenarios, self.start_date, self.end_date, tz=tz)
            if self.excluded_dates:
                scenarios = [
                    s for s in scenarios
                    if pd.Timestamp(
                        int(s['year']), int(s['month']), int(s['day']), tz=tz
                    ).normalize() not in self.excluded_dates
                ]
            if not scenarios:
                raise ValueError(
                    'No NSRDB scenarios in date range {}; check manifest {}'.format(
                        (self.start_date.date(), self.end_date.date()), manifest_path))
            self._nsrdb_manifest = manifest
            self._nsrdb_year_cache = NsrdbYearCache(manifest)
            self._nsrdb_scenarios = scenarios
            self._nsrdb_scenario_probs = normalized_scenario_probabilities(scenarios)
            self._nsrdb_by_id = by_id
            self._nsrdb_by_mday = {
                k: sorted(
                    [s for s in v if s in scenarios],
                    key=scenario_id)
                for k, v in by_mday.items()
            }
            if self.fixed_eval_scenarios:
                for sid in self.fixed_eval_scenarios:
                    if sid not in by_id:
                        raise ValueError(
                            'fixed_eval_scenarios id {!r} not in manifest'.format(sid))
            # Calendar days that have at least one scenario (for randomize_day path).
            day_keys = sorted({(int(s['month']), int(s['day'])) for s in scenarios})
            self.start_dates = pd.DatetimeIndex([
                pd.Timestamp(
                    year=self.start_date.year, month=m, day=d, tz=tz)
                for m, d in day_keys
            ])
            if self.randomize_scenario and self.randomize_day:
                logger.warning(
                    'nsrdb_multiyear: randomize_scenario=True takes precedence over '
                    'randomize_day (uniform over all scenarios).')
        if self.excluded_dates:
            self.start_dates = pd.DatetimeIndex([
                date for date in self.start_dates
                if date.normalize() not in self.excluded_dates
            ])
        if len(self.start_dates) == 0:
            raise ValueError('PVTrackingEnv start_date must be <= end_date.')

        self.interval_hours = pd.Timedelta(self.freq).total_seconds() / 3600.0

        self.tilt_limits = (0.0, 90.0)
        self.azimuth_limits = (0.0, 360.0)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        low, high = self._observation_bounds()
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        try:
            from gym.utils import seeding
            self.np_random, seed = seeding.np_random(seed)
        except Exception:
            self.np_random = np.random.RandomState(seed)
        self._rollout_seed = seed
        # Explicit seeding is for matched-scenario eval (seed % len(fixed list)).
        # In-train MBPO eval uses begin_evaluation_rollouts() to cycle all ids.
        self._eval_cycle_fixed_ids = False
        return [seed]

    def begin_evaluation_rollouts(self, n_episodes=None):
        """Reset counters so in-train eval visits each fixed id once per epoch.

        MBPO calls this before rollouts(n_episodes). Without it, env.seed() from
        __init__ pins one scenario via rollout_seed % len(fixed_eval_scenarios),
        making evaluation/return-std == 0 incorrectly.
        """
        self._fixed_eval_scenario_index = 0
        self._fixed_eval_date_index = 0
        self._eval_cycle_fixed_ids = True

    def _build_times(self, date):
        return pd.date_range(
            start=f"{date.date()} {self.start_time}",
            periods=self.periods,
            freq=self.freq,
            tz=self.location.tz,
        )

    @staticmethod
    def _clock_hour_from_timestamp(timestamp):
        return float(timestamp.hour + timestamp.minute / 60.0)

    def _episode_time_metadata(self, timestamp):
        """Wall-clock metadata in env timezone (project standard: UTC).

        No local-time conversion: when Location.tz is UTC, labels are taken
        directly from the pvlib DatetimeIndex timestamp.
        """
        env_hour = self._clock_hour_from_timestamp(timestamp)
        tz_name = str(self.location.tz)
        if tz_name.upper() in ('UTC', 'ETC/UTC', 'ETC/GMT'):
            utc_iso = timestamp.isoformat()
        else:
            utc_iso = timestamp.tz_convert('UTC').isoformat()
        return {
            'time': env_hour,
            'clock_hour': env_hour,
            'clock_hour_utc': env_hour,
            'clock_hour_env_tz': env_hour,
            'timezone': tz_name,
            'timestamp_utc_iso': utc_iso,
            'timestamp_env_iso': timestamp.isoformat(),
        }

    def _build_weather_profile(self, times):
        if self.weather_source == 'clearsky':
            clearsky = self.location.get_clearsky(times)
            weather = pd.DataFrame({
                'dni': clearsky['dni'].clip(lower=0.0),
                'ghi': clearsky['ghi'].clip(lower=0.0),
                'dhi': clearsky['dhi'].clip(lower=0.0),
                'temperature': np.full(len(times), self.temperature, dtype=np.float64),
                'wind_speed': np.full(len(times), self.wind_speed, dtype=np.float64),
                'condition': ['clear'] * len(times),
            }, index=times)
        elif self.weather_source == 'nsrdb_multiyear':
            if self._current_scenario is None:
                raise RuntimeError('NSRDB scenario not set before _build_weather_profile')
            year = scenario_source_year(self._current_scenario)
            year_weather = self._nsrdb_year_cache.get_year(year)
            weather = build_weather_profile_from_scenario(
                self.location, times, year_weather, self._current_scenario)
        else:
            weather = build_weather_profile_from_catalog(
                self.location, times, self._historical_weather_catalog)

        return weather

    def _select_nsrdb_scenario(self):
        """Sample one scenario e ~ p(e) once per episode reset.

        Weather within the episode is the fixed NSRDB trajectory for that scenario
        (see _build_weather_profile); it is not re-sampled at each control step.
        """
        if self.fixed_eval_scenarios is not None:
            if self._eval_cycle_fixed_ids:
                index = self._fixed_eval_scenario_index % len(self.fixed_eval_scenarios)
                self._fixed_eval_scenario_index += 1
            elif self._rollout_seed is not None:
                index = int(self._rollout_seed) % len(self.fixed_eval_scenarios)
            else:
                index = self._fixed_eval_scenario_index % len(self.fixed_eval_scenarios)
                self._fixed_eval_scenario_index += 1
            sid = self.fixed_eval_scenarios[index]
            return self._nsrdb_by_id[sid]
        if self.randomize_scenario or not self.randomize_day:
            index = int(self.np_random.choice(
                len(self._nsrdb_scenarios), p=self._nsrdb_scenario_probs))
            return self._nsrdb_scenarios[index]
        # randomize_day: pick calendar day, then uniform over years with that month/day.
        index = self.np_random.randint(len(self.start_dates))
        m = int(self.start_dates[index].month)
        d = int(self.start_dates[index].day)
        pool = self._nsrdb_by_mday.get((m, d), [])
        if not pool:
            index = int(self.np_random.choice(
                len(self._nsrdb_scenarios), p=self._nsrdb_scenario_probs))
            return self._nsrdb_scenarios[index]
        return pool[self.np_random.randint(len(pool))]

    def _resolve_nsrdb_scenario_for_date(self, date):
        """Deterministic scenario for a calendar date (no pool resampling at reset)."""
        ts = pd.Timestamp(date, tz=self.location.tz)
        pool = self._nsrdb_by_mday.get(
            (int(ts.month), int(ts.day)), self._nsrdb_scenarios)
        pool = sorted(pool, key=scenario_id)
        manifest_years = {scenario_source_year(s) for s in self._nsrdb_scenarios}
        if int(ts.year) in manifest_years:
            by_year = [s for s in pool if scenario_source_year(s) == int(ts.year)]
            if by_year:
                return by_year[0]
        return pool[0]

    def _apply_episode_weather_stochasticity(self):
        """Optional episode-level irradiance scale (one draw at reset, not per step).

        Default 0.0: weather(d) is fixed for the day; pvlib maps that trajectory
        deterministically. When > 0, a single lognormal scale is applied to the
        whole day's dni/dhi/ghi series.
        """
        if self.irradiance_perturbation_std <= 0.0:
            return
        scale = float(self.np_random.lognormal(
            mean=0.0, sigma=self.irradiance_perturbation_std))
        for col in ('dni', 'dhi', 'ghi'):
            self.weather_profile[col] = np.maximum(
                self.weather_profile[col].values * scale, 0.0)

    def _prepare_episode_observation_noise(self):
        """Pre-sample observation noise for every timestep at reset (not each step)."""
        if self.observation_noise_std <= 0.0:
            self._episode_obs_noise = None
            return
        obs_dim = int(self.observation_space.shape[0])
        self._episode_obs_noise = self.np_random.normal(
            0.0,
            self.observation_noise_std,
            size=(len(self.times), obs_dim),
        ).astype(np.float32)

    def _maybe_noise_observation(self, obs, step_index=None):
        if self.observation_noise_std <= 0.0 or self._episode_obs_noise is None:
            return obs
        idx = int(self.step_index if step_index is None else step_index)
        idx = min(max(idx, 0), len(self._episode_obs_noise) - 1)
        noise = self._episode_obs_noise[idx]
        return np.clip(
            obs + noise,
            self.observation_space.low,
            self.observation_space.high).astype(np.float32)

    def _current_weather(self):
        weather = self.weather_profile.iloc[self.step_index]
        return {
            'dni': float(weather['dni']),
            'dhi': float(weather['dhi']),
            'ghi': float(weather['ghi']),
            'temperature': float(weather['temperature']),
            'wind_speed': float(weather['wind_speed']),
        }

    def _solar_position(self, timestamp):
        solar_position = self.location.get_solarposition(timestamp)
        return solar_position.iloc[0]

    @property
    def num_action_steps(self):
        """Number of control steps per episode (one less than timestamp count)."""
        return max(len(self.times) - 1, 1)

    def _observation_bounds(self):
        if self.observation_mode == 'physical':
            low = np.array([
                0.0, -1.0, -1.0,
                0.0, 0.0, 0.0,
                -1.0,
                0.0, -1.0, -1.0,
                0.0,
            ], dtype=np.float32)
            high = np.array([
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0,
                1.0, 1.0, 1.0,
                1.0,
            ], dtype=np.float32)
        else:
            low = np.array([
                0.0, -1.0, -1.0,
                0.0, 0.0, 0.0,
                -1.0,
                0.0, -1.0, -1.0,
                0.0,
                -1.0, -1.0,
                -1.0, -1.0,
            ], dtype=np.float32)
            high = np.array([
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0,
                1.0, 1.0, 1.0,
                1.0,
                1.0, 1.0,
                1.0, 1.0,
            ], dtype=np.float32)
        return low, high

    def _cos_angle_of_incidence(self, solar_zenith, solar_azimuth, tilt, azimuth):
        incidence_deg = float(aoi(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=solar_zenith,
            solar_azimuth=solar_azimuth,
        ))
        return float(np.clip(np.cos(np.deg2rad(incidence_deg)), 0.0, 1.0))

    def _log_observation_vector(self, obs, phase):
        if not self.log_observations or self._obs_log_steps_remaining <= 0:
            return
        lines = ['PVTracking observation ({}, mode={}):'.format(
            phase, self.observation_mode)]
        for name, value in zip(self.obs_feature_names, np.asarray(obs).reshape(-1)):
            lines.append('  {:24s} {: .6f}'.format(name, float(value)))
        logger.info('\n'.join(lines))
        print('\n'.join(lines))
        self._obs_log_steps_remaining -= 1

    def _power_from_orientation(self, solar_zenith, solar_azimuth, tilt, azimuth):
        weather = self._current_weather()
        return compute_panel_power_w(
            tilt, azimuth,
            solar_zenith, solar_azimuth,
            weather['dni'], weather['ghi'], weather['dhi'],
            area=self.area,
            efficiency=self.efficiency,
        )

    def _build_observation(self, solar_zenith, solar_azimuth, weather, power):
        solar_azimuth_rad = np.deg2rad(solar_azimuth)
        panel_azimuth_rad = np.deg2rad(self.azimuth)

        def _norm_irradiance(value):
            return float(np.clip(value / IRRADIANCE_NORM, 0.0, 1.0))

        def _norm_temperature(value):
            return float(np.clip(value / TEMPERATURE_NORM, -1.0, 1.0))

        shared = [
            float(np.clip(solar_zenith / 180.0, 0.0, 1.0)),
            float(np.sin(solar_azimuth_rad)),
            float(np.cos(solar_azimuth_rad)),
            _norm_irradiance(weather['dni']),
            _norm_irradiance(weather['dhi']),
            _norm_irradiance(weather['ghi']),
            _norm_temperature(weather['temperature']),
            float(self.tilt / 90.0),
            float(np.sin(panel_azimuth_rad)),
            float(np.cos(panel_azimuth_rad)),
        ]

        if self.observation_mode == 'physical':
            cos_aoi = self._cos_angle_of_incidence(
                solar_zenith, solar_azimuth, self.tilt, self.azimuth)
            obs = np.array(shared + [cos_aoi], dtype=np.float32)
        else:
            time_of_day = self.current_time.hour + self.current_time.minute / 60.0
            time_angle = 2.0 * np.pi * time_of_day / 24.0
            day_angle = 2.0 * np.pi * float(self.current_time.dayofyear) / 365.0
            obs = np.array(
                shared + [
                    float(power / POWER_NORM),
                    float(np.sin(time_angle)),
                    float(np.cos(time_angle)),
                    float(np.sin(day_angle)),
                    float(np.cos(day_angle)),
                ],
                dtype=np.float32,
            )
        return obs

    def reset(self, date=None, scenario_id=None):
        self.step_index = 0
        self._episode_obs_noise = None
        if self.weather_source == 'nsrdb_multiyear':
            if scenario_id is not None:
                if scenario_id not in self._nsrdb_by_id:
                    raise ValueError(
                        'Unknown scenario_id {!r} for NSRDB manifest'.format(scenario_id))
                self._current_scenario = self._nsrdb_by_id[scenario_id]
            elif date is not None:
                self._current_scenario = self._resolve_nsrdb_scenario_for_date(date)
            else:
                self._current_scenario = self._select_nsrdb_scenario()
            self.current_date = scenario_episode_date(
                self._current_scenario, tz=self.location.tz)
        elif date is not None:
            self.current_date = pd.Timestamp(date, tz=self.location.tz)
        elif self.fixed_eval_dates is not None:
            if len(self.fixed_eval_dates) == 0:
                raise ValueError('fixed_eval_dates must contain at least one date.')
            if self._eval_cycle_fixed_ids:
                index = self._fixed_eval_date_index % len(self.fixed_eval_dates)
                self._fixed_eval_date_index += 1
            elif self._rollout_seed is not None:
                index = int(self._rollout_seed) % len(self.fixed_eval_dates)
            else:
                index = self._fixed_eval_date_index % len(self.fixed_eval_dates)
                self._fixed_eval_date_index += 1
            self.current_date = self.fixed_eval_dates[index]
        elif self.randomize_day and len(self.start_dates) > 1:
            index = self.np_random.randint(len(self.start_dates))
            self.current_date = self.start_dates[index]
        else:
            self.current_date = self.start_dates[0]

        self.times = self._build_times(self.current_date)
        self.weather_profile = self._build_weather_profile(self.times)
        self._apply_episode_weather_stochasticity()
        self._prepare_episode_observation_noise()
        self._episode_weather_diagnostics = episode_weather_diagnostics(
            self.weather_profile)

        if self.randomize_initial_orientation:
            self.tilt = float(self.np_random.uniform(*self.tilt_limits))
            self.azimuth = float(self.np_random.uniform(0.0, 360.0))
        else:
            self.tilt = 30.0
            self.azimuth = 180.0

        self.current_time = self.times[self.step_index]
        solar_position = self._solar_position(self.current_time)
        weather = self._current_weather()
        power = self._power_from_orientation(
            solar_position.zenith,
            solar_position.azimuth,
            self.tilt,
            self.azimuth,
        )

        self.last_power = power
        if self.log_observations:
            self._obs_log_steps_remaining = 2
        obs = self._build_observation(
            solar_position.zenith,
            solar_position.azimuth,
            weather,
            power,
        )
        self._log_observation_vector(obs, 'reset')
        return self._maybe_noise_observation(obs)

    def step(self, action):
        delta_tilt = float(action[0]) * self.max_delta_tilt
        delta_azimuth = float(action[1]) * self.max_delta_azimuth

        self.tilt = float(np.clip(self.tilt + delta_tilt, *self.tilt_limits))
        self.azimuth = float(np.mod(self.azimuth + delta_azimuth, 360.0))

        self.step_index += 1
        done = self.step_index >= len(self.times) - 1
        self.current_time = self.times[min(self.step_index, len(self.times) - 1)]

        solar_position = self._solar_position(self.current_time)
        weather = self._current_weather()
        power = self._power_from_orientation(
            solar_position.zenith,
            solar_position.azimuth,
            self.tilt,
            self.azimuth,
        )

        energy_kwh = power * self.interval_hours / 1000.0
        movement_cost = self.movement_penalty * (
            abs(delta_tilt) / self.max_delta_tilt
            + abs(delta_azimuth) / self.max_delta_azimuth
        )
        reward = energy_kwh - movement_cost

        obs = self._build_observation(
            solar_position.zenith,
            solar_position.azimuth,
            weather,
            power,
        )
        self._log_observation_vector(obs, 'step')
        self.last_power = power
        time_meta = self._episode_time_metadata(self.current_time)
        day_of_year = int(self.current_time.dayofyear)
        if 80 <= day_of_year <= 171:
            season = 'spring'
        elif 172 <= day_of_year <= 263:
            season = 'summer'
        elif 264 <= day_of_year <= 354:
            season = 'fall'
        else:
            season = 'winter'

        solar_alt = float(90.0 - solar_position.zenith)
        info = {
            'poa_global': float(power / self.efficiency / self.area),
            'power': float(power),
            'energy_kwh': float(energy_kwh),
            'movement_cost': float(movement_cost),
            'reward_energy': float(energy_kwh),
            'reward_movement': float(movement_cost),
            'tilt': float(self.tilt),
            'azimuth': float(self.azimuth),
            'date': str(self.current_time.date()),
            'day_of_year': day_of_year,
            'season': season,
            'weather_source': self.weather_source,
            'weather_scenario_mode': self.weather_scenario_mode,
            'scenario_id': (
                scenario_id(self._current_scenario)
                if self._current_scenario is not None else None),
            'scenario_year': (
                scenario_source_year(self._current_scenario)
                if self._current_scenario is not None else None),
            'weather_condition': self.weather_profile['condition'].iloc[self.step_index],
            'episode_diffuse_fraction': float(
                self._episode_weather_diagnostics.get('diffuse_fraction', 0.0)),
            'episode_dni_fraction': float(
                self._episode_weather_diagnostics.get('dni_fraction', 0.0)),
            'solar_zenith_deg': float(solar_position.zenith),
            'solar_azimuth_deg': float(solar_position.azimuth),
            'solar_altitude_deg': solar_alt,
            'cos_aoi': self._cos_angle_of_incidence(
                solar_position.zenith,
                solar_position.azimuth,
                self.tilt,
                self.azimuth,
            ),
            'observation_mode': self.observation_mode,
            'dni_wm2': float(weather['dni']),
            'dhi_wm2': float(weather['dhi']),
            'ghi_wm2': float(weather['ghi']),
            'temperature_c': float(weather['temperature']),
            'delta_tilt_deg': float(delta_tilt),
            'delta_azimuth_deg': float(delta_azimuth),
            'interval_hours': float(self.interval_hours),
            'freq': str(self.freq),
            'control_interval_minutes': float(self.interval_hours * 60.0),
            'weather_native_interval_minutes': (
                float(self._nsrdb_manifest.get('meta', {}).get('interval_minutes', 5))
                if self.weather_source == 'nsrdb_multiyear'
                and getattr(self, '_nsrdb_manifest', None) is not None
                else None),
            'weather_resampling': (
                'none_native_5min'
                if self.weather_source == 'nsrdb_multiyear' else None),
            'num_action_steps': int(self.num_action_steps),
            'rollout_seed': (
                int(self._rollout_seed)
                if self._rollout_seed is not None else None),
        }
        info.update(time_meta)
        info['irradiance_perturbation_std'] = float(self.irradiance_perturbation_std)
        info['observation_noise_std'] = float(self.observation_noise_std)

        return self._maybe_noise_observation(obs), float(reward), bool(done), info

    def render(self, mode='human'):
        print(
            f"date={self.current_time.date()} time={self.current_time.time()} "
            f"tilt={self.tilt:.1f} azimuth={self.azimuth:.1f} "
            f"power={self.last_power:.1f}"
        )

    def close(self):
        return
