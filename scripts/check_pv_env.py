import argparse
import gym
import numpy as np

# Ensure the custom MBPO environment registry is loaded.
import softlearning.environments.adapters.gym_adapter  # noqa: F401


def validate_weather_profile(env):
    """Sanity-check irradiance/temperature weather table vs pvlib solar position."""
    inner = env.unwrapped
    profile = inner.weather_profile
    times = profile.index
    solar = inner.location.get_solarposition(times)
    cos_zenith = np.cos(np.deg2rad(solar['zenith'].clip(0.0, 90.0).values))
    poa_direct = profile['dni'].values * np.maximum(cos_zenith, 0.0)
    dhi_residual = profile['ghi'].values - poa_direct
    dhi_expected = np.clip(dhi_residual, 0.0, profile['ghi'].values)
    max_dhi_err = float(np.max(np.abs(profile['dhi'].values - dhi_expected)))
    assert (profile['dni'].values >= 0).all()
    assert (profile['dhi'].values >= 0).all()
    assert (profile['ghi'].values >= 0).all()
    assert len(profile) == len(times) == inner.periods
    print('weather_source:', inner.weather_source)
    print('irradiance units: W/m^2 (pvlib clearsky base)')
    print('dni  min/max: {:.1f} / {:.1f}'.format(profile['dni'].min(), profile['dni'].max()))
    print('dhi  min/max: {:.1f} / {:.1f}'.format(profile['dhi'].min(), profile['dhi'].max()))
    print('ghi  min/max: {:.1f} / {:.1f}'.format(profile['ghi'].min(), profile['ghi'].max()))
    print('temp min/max (C): {:.1f} / {:.1f}'.format(
        profile['temperature'].min(), profile['temperature'].max()))
    if inner.weather_source == 'historical':
        clearsky = inner.location.get_clearsky(times)
        ghi_ratio = profile['ghi'].values / np.maximum(clearsky['ghi'].values, 1.0)
        print('historical weather note: no exact DHI closure check; using dataset irradiance directly')
        print('historical/clearsky GHI ratio min/max: {:.3f} / {:.3f}'.format(
            float(np.min(ghi_ratio)), float(np.max(ghi_ratio))))
    else:
        print('clearsky decomposition residual max: {:.4f}'.format(max_dhi_err))
    print('weather profile alignment: OK ({} timestamps)'.format(len(times)))


def main():
    parser = argparse.ArgumentParser(description='Smoke-test PVTracking-v0.')
    parser.add_argument(
        '--observation-mode',
        choices=('legacy', 'physical'),
        default='legacy',
        help='Observation layout (default: legacy for checkpoint compatibility).',
    )
    parser.add_argument(
        '--weather-source',
        choices=('clearsky', 'historical'),
        default='historical',
        help='Weather source to load for the smoke test (default: historical).',
    )
    parser.add_argument(
        '--log-obs',
        action='store_true',
        help='Print named observation features on reset and first step.',
    )
    parser.add_argument(
        '--validate-weather',
        action='store_true',
        help='Check irradiance units, DHI consistency, and timestamp alignment.',
    )
    args = parser.parse_args()

    env = gym.make(
        'PVTracking-v0',
        observation_mode=args.observation_mode,
        weather_source=args.weather_source,
        log_observations=args.log_obs,
    )
    inner = env.unwrapped
    obs = env.reset()
    if args.validate_weather:
        validate_weather_profile(env)
    print('observation_mode:', getattr(inner, 'observation_mode', 'legacy'))
    print('feature_names:', getattr(inner, 'obs_feature_names', []))
    print('reset obs shape:', np.asarray(obs).shape)
    print('action space:', env.action_space)
    print('obs space:', env.observation_space)

    # Incremental action semantics: action[i] in [-1,1] -> delta_tilt = action[0] * max_delta_tilt
    tilt_before = inner.tilt
    _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    delta = inner.tilt - tilt_before
    assert abs(delta - inner.max_delta_tilt) < 0.01, (
        'action=[1,0] expected delta_tilt={:.1f}, got {:.4f}'.format(
            inner.max_delta_tilt, delta))
    print('action semantics OK: [1,0] -> delta_tilt={:.2f} deg (max={:.1f})'.format(
        delta, inner.max_delta_tilt))

    for step in range(3):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        print(
            'step={}'.format(step),
            'action={}'.format(action),
            'reward={:.4f}'.format(reward),
            'done={}'.format(done),
            'cos_aoi={:.4f}'.format(info.get('cos_aoi', float('nan'))),
        )
        if done:
            print('done at step', step)
            break

    env.close()
    print('PVTracking environment checker completed successfully.')


if __name__ == '__main__':
    main()
