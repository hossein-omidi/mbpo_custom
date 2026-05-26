import gzip
import pickle
import tempfile

import numpy as np
import pytest

from mbpo.env.pv_tracking import PVTrackingEnv
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool
from scripts.eval_utils import decode_pv_observation, normalize_angle_diff


REPRESENTATIVE_LOGITS = np.array([-10.0, -2.0, -0.1, 0.0, 0.1, 2.0, 10.0], dtype=np.float64)
ROUNDTRIP_ANGLES_DEG = [0, 1, 45, 90, 179, 180, 181, 270, 359]


def make_physical_env():
    return PVTrackingEnv(
        randomize_day=False,
        start_date='2020-06-21',
        end_date='2020-06-21',
        randomize_initial_orientation=False,
        weather_source='clearsky',
        movement_penalty=0.0,
        observation_mode='physical',
    )


def make_historical_env(start_date='2020-06-21', end_date='2020-06-21'):
    return PVTrackingEnv(
        randomize_day=False,
        start_date=start_date,
        end_date=end_date,
        randomize_initial_orientation=False,
        weather_source='historical',
        movement_penalty=0.0,
        observation_mode='physical',
    )


def advance_to_productive_sun(env, min_altitude_deg=20.0):
    obs = env.reset()
    action = np.zeros(2, dtype=np.float32)
    while True:
        solar = env._solar_position(env.current_time)
        altitude = 90.0 - float(solar.zenith)
        if altitude >= min_altitude_deg:
            return obs, solar
        obs, _, done, _ = env.step(action)
        assert not done, 'Episode ended before reaching productive sun.'


def test_tanh_preserves_sign_for_representative_logits():
    squashed = np.tanh(REPRESENTATIVE_LOGITS)
    for raw, out in zip(REPRESENTATIVE_LOGITS, squashed):
        if raw == 0.0:
            assert out == pytest.approx(0.0)
        else:
            assert np.sign(out) == np.sign(raw)


def test_physical_observation_uses_zenith_not_altitude():
    env = make_physical_env()
    try:
        obs = env.reset()
        solar = env._solar_position(env.current_time)
        assert obs[0] * 180.0 == pytest.approx(float(solar.zenith), abs=1e-5)

        _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info['solar_altitude_deg'] == pytest.approx(
            90.0 - info['solar_zenith_deg'], abs=1e-6)
    finally:
        env.close()


def test_azimuth_encoding_round_trip_matches_atan2_sin_cos_order():
    for angle in ROUNDTRIP_ANGLES_DEG:
        obs = np.zeros(11, dtype=np.float32)
        obs[1] = np.sin(np.deg2rad(angle))
        obs[2] = np.cos(np.deg2rad(angle))
        obs[8] = np.sin(np.deg2rad(angle))
        obs[9] = np.cos(np.deg2rad(angle))
        decoded = decode_pv_observation(obs)

        assert abs(normalize_angle_diff(decoded['solar_azimuth_deg'], angle)) < 1e-5
        assert abs(normalize_angle_diff(decoded['panel_azimuth_deg'], angle)) < 1e-5


def test_wrapped_azimuth_error_examples():
    assert normalize_angle_diff(10.0, 350.0) == pytest.approx(20.0)
    assert normalize_angle_diff(350.0, 10.0) == pytest.approx(-20.0)
    assert normalize_angle_diff(181.0, 179.0) == pytest.approx(2.0)
    assert normalize_angle_diff(179.0, 181.0) == pytest.approx(-2.0)


def test_action_sign_mapping_matches_documented_physical_directions():
    env = make_physical_env()
    try:
        action_expectations = [
            (np.array([1.0, 0.0], dtype=np.float32), 5.0, 0.0, 35.0, 180.0),
            (np.array([-1.0, 0.0], dtype=np.float32), -5.0, 0.0, 25.0, 180.0),
            (np.array([0.0, 1.0], dtype=np.float32), 0.0, 10.0, 30.0, 190.0),
            (np.array([0.0, -1.0], dtype=np.float32), 0.0, -10.0, 30.0, 170.0),
        ]
        for action, expected_dt, expected_da, expected_tilt, expected_az in action_expectations:
            env.reset()
            _, _, _, info = env.step(action)
            assert info['delta_tilt_deg'] == pytest.approx(expected_dt)
            assert info['delta_azimuth_deg'] == pytest.approx(expected_da)
            assert env.tilt == pytest.approx(expected_tilt)
            assert env.azimuth == pytest.approx(expected_az)
    finally:
        env.close()


def test_power_increases_when_moving_toward_direct_alignment():
    env = make_physical_env()
    try:
        _, solar = advance_to_productive_sun(env, min_altitude_deg=20.0)
        env.tilt = float(np.clip(float(solar.zenith) - 20.0, 0.0, 90.0))
        env.azimuth = float((float(solar.azimuth) - 30.0) % 360.0)

        base_power = env._power_from_orientation(
            float(solar.zenith), float(solar.azimuth), env.tilt, env.azimuth)
        toward_power = env._power_from_orientation(
            float(solar.zenith),
            float(solar.azimuth),
            min(env.tilt + env.max_delta_tilt, 90.0),
            (env.azimuth + env.max_delta_azimuth) % 360.0,
        )
        away_power = env._power_from_orientation(
            float(solar.zenith),
            float(solar.azimuth),
            max(env.tilt - env.max_delta_tilt, 0.0),
            (env.azimuth - env.max_delta_azimuth) % 360.0,
        )

        assert toward_power > base_power > away_power
    finally:
        env.close()


def test_cos_aoi_is_maximal_at_direct_alignment():
    env = make_physical_env()
    try:
        _, solar = advance_to_productive_sun(env, min_altitude_deg=20.0)
        aligned = env._cos_angle_of_incidence(
            float(solar.zenith), float(solar.azimuth),
            float(solar.zenith), float(solar.azimuth))
        perturbed = env._cos_angle_of_incidence(
            float(solar.zenith), float(solar.azimuth),
            min(float(solar.zenith) + 10.0, 90.0), (float(solar.azimuth) + 15.0) % 360.0)

        assert aligned == pytest.approx(1.0, abs=1e-6)
        assert aligned > perturbed
    finally:
        env.close()


def test_real_daily_episode_marks_terminal_on_final_transition():
    env = make_physical_env()
    try:
        obs = env.reset()
        action = np.zeros(2, dtype=np.float32)
        done = False
        steps = 0
        while not done:
            obs, _, done, _ = env.step(action)
            steps += 1
        assert steps == env.num_action_steps
        assert done is True
    finally:
        env.close()


def test_replay_remaining_steps_excludes_near_terminal_states_for_rollout_length():
    env = make_physical_env()
    try:
        obs = env.reset()
        action = np.zeros(2, dtype=np.float32)
        rollout_length = 5
        path = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminals': [],
            'next_observations': [],
        }
        for step in range(env.num_action_steps):
            next_obs, reward, done, _ = env.step(action)
            path['observations'].append(obs)
            path['actions'].append(action.copy())
            path['rewards'].append([reward])
            path['terminals'].append([done])
            path['next_observations'].append(next_obs)
            obs = next_obs
            if done:
                break
        for key in path:
            path[key] = np.asarray(path[key])

        pool = SimpleReplayPool(env.observation_space, env.action_space, max_size=256)
        pool.add_path(path)
        remaining_steps = pool.fields['remaining_steps'][:pool.size].squeeze(-1)
        expected = np.arange(env.num_action_steps, 0, -1, dtype=np.int32)
        assert np.array_equal(remaining_steps, expected)
        assert np.array_equal((remaining_steps > rollout_length),
                              (expected > rollout_length))
    finally:
        env.close()


def test_old_replay_experience_missing_remaining_steps_fails_loudly():
    env = make_physical_env()
    try:
        pool = SimpleReplayPool(env.observation_space, env.action_space, max_size=16)
        samples = {
            'observations': np.zeros((2, env.observation_space.shape[0]), dtype=np.float32),
            'actions': np.zeros((2, env.action_space.shape[0]), dtype=np.float32),
            'rewards': np.zeros((2, 1), dtype=np.float32),
            'terminals': np.zeros((2, 1), dtype=bool),
            'next_observations': np.zeros((2, env.observation_space.shape[0]), dtype=np.float32),
        }
        with tempfile.NamedTemporaryFile(suffix='.pkl.gz') as tmp:
            with gzip.open(tmp.name, 'wb') as f:
                pickle.dump(samples, f)
            with pytest.raises(Exception):
                pool.load_experience(tmp.name)
    finally:
        env.close()


def test_training_date_exclusion_is_enforced_programmatically():
    env = PVTrackingEnv(
        randomize_day=True,
        start_date='2020-06-01',
        end_date='2020-06-03',
        excluded_dates=['2020-06-02'],
        randomize_initial_orientation=False,
        weather_source='clearsky',
        movement_penalty=0.0,
        observation_mode='physical',
    )
    try:
        seen = set()
        for seed in range(20):
            env.seed(seed)
            env.reset()
            seen.add(str(env.current_date.date()))
        assert seen <= {'2020-06-01', '2020-06-03'}
        assert '2020-06-02' not in seen
    finally:
        env.close()


def test_historical_weather_changes_irradiance_and_power():
    hist_env = make_historical_env('2020-06-21', '2020-06-21')
    clear_env = make_physical_env()
    try:
        hist_env.reset()
        clear_env.reset()

        midday_index = len(hist_env.times) // 2
        hist_env.step_index = midday_index
        clear_env.step_index = midday_index
        hist_env.current_time = hist_env.times[midday_index]
        clear_env.current_time = clear_env.times[midday_index]

        hist_solar = hist_env._solar_position(hist_env.current_time)
        clear_solar = clear_env._solar_position(clear_env.current_time)

        hist_ghi = float(hist_env.weather_profile['ghi'].iloc[midday_index])
        clear_ghi = float(clear_env.weather_profile['ghi'].iloc[midday_index])
        hist_power = hist_env._power_from_orientation(
            float(hist_solar.zenith), float(hist_solar.azimuth),
            hist_env.tilt, hist_env.azimuth)
        clear_power = clear_env._power_from_orientation(
            float(clear_solar.zenith), float(clear_solar.azimuth),
            clear_env.tilt, clear_env.azimuth)

        assert hist_env.weather_profile['condition'].iloc[midday_index] in (
            'clear', 'partly_cloudy', 'overcast')
        assert hist_ghi < clear_ghi
        assert hist_power < clear_power
    finally:
        hist_env.close()
        clear_env.close()
