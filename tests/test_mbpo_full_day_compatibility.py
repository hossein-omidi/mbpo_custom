from types import SimpleNamespace

import pytest


def _reference_env(horizon=195, periods=196):
    core = SimpleNamespace(
        max_delta_tilt=5.0,
        max_delta_azimuth=10.0,
        movement_penalty=0.01,
        movement_cost_mode='legacy',
        actuator_power_w=30.0,
        slew_rate_tilt_deg_s=1.5,
        slew_rate_azimuth_deg_s=2.0,
        interval_hours=5.0 / 60.0,
        num_action_steps=horizon,
        periods=periods,
    )
    return SimpleNamespace(unwrapped=core, horizon=horizon)


def test_structured_fake_environment_accepts_195_step_episode():
    pytest.importorskip('torch')
    pytest.importorskip('gym')
    from MBPO_workspace.src.dynamics import StructuredFakeEnv

    fake = StructuredFakeEnv(object(), _reference_env())
    assert fake.horizon == 195


def test_structured_fake_environment_rejects_mismatched_real_horizon():
    pytest.importorskip('torch')
    pytest.importorskip('gym')
    from MBPO_workspace.src.dynamics import StructuredFakeEnv

    with pytest.raises(ValueError, match='horizon'):
        StructuredFakeEnv(object(), _reference_env(horizon=195, periods=118))


def test_model_fit_frequency_must_land_on_complete_episode_boundary():
    pytest.importorskip('torch')
    from MBPO_workspace.src.trainer import validate_model_fit_frequency

    assert validate_model_fit_frequency(195, 195) == 195
    assert validate_model_fit_frequency(390, 195) == 390
    with pytest.raises(ValueError, match='multiple'):
        validate_model_fit_frequency(117, 195)


def test_paired_reference_contract_accepts_195_steps():
    from datetime import datetime, timedelta, timezone
    from mbpo.env.baseline_contract import validate_paired_reference_group

    n = 195
    start = datetime(2026, 6, 21, 8, 50, tzinfo=timezone.utc)
    step_energy = 100.0 * (5.0 / 60.0) / 1000.0
    infos = []
    for i in range(n):
        stamp = (start + timedelta(minutes=5 * i)).isoformat()
        infos.append({
            'timestamp_utc_iso': stamp,
            'power_model': 'legacy',
            'weather_source': 'nsrdb_multiyear',
            'solar_zenith_deg': 45.0,
            'solar_azimuth_deg': 180.0,
            'dni_wm2': 500.0,
            'ghi_wm2': 600.0,
            'dhi_wm2': 100.0,
            'surface_albedo': 0.22,
            'albedo_source': 'nsrdb_recorded',
            'power_dc_w': 100.0,
            'control_interval_minutes': 5.0,
        })
    path = {
        'method': 'fixed_no_motion',
        'scenario_id': '2026-06-21',
        'seed': 0,
        'initial_pose': [30.0, 180.0],
        'power': [100.0] * n,
        'energy': [step_energy] * n,
        'reward': [step_energy] * n,
        'cost': [0.0] * n,
        'tilt': [30.0] * n,
        'azimuth': [180.0] * n,
        'action_tilt': [0.0] * n,
        'action_azimuth': [0.0] * n,
        'delta_tilt_deg': [0.0] * n,
        'delta_azimuth_deg': [0.0] * n,
        'infos': infos,
        'gross_energy_kwh': step_energy * n,
        'net_energy_kwh': step_energy * n,
        'movement_cost_kwh': 0.0,
    }
    result = validate_paired_reference_group({'fixed_no_motion': path})
    assert result['steps'] == 195


def test_dual_axis_reference_passes_recorded_albedo(monkeypatch):
    from types import SimpleNamespace
    import MBPO_workspace.src.power_references as refs

    captured = {}

    def fake_compute(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            power_w=120.0,
            poa_global_wm2=700.0,
            effective_irradiance_wm2=700.0,
            cell_temperature_c=35.0,
            wind_speed_10m_ms=2.0,
            surface_albedo=kwargs['surface_albedo'],
        )

    monkeypatch.setattr(refs, 'compute_pv_power', fake_compute)
    core = SimpleNamespace(
        weather_source='nsrdb_multiyear',
        area=1.0, efficiency=0.18, power_model='pvwatts_sapm',
        wind_reference_height_m=2.0, gamma_pdc=-0.004,
        sapm_mount='open_rack_glass_polymer',
        wind_powerlaw_exponent=0.14, interval_hours=5.0 / 60.0,
        _cos_angle_of_incidence=lambda *args: 1.0,
    )
    env = SimpleNamespace(unwrapped=core, horizon=1)
    fixed = {
        'scenario_id': '2026-06-21', 'seed': 0, 'initial_pose': [30.0, 180.0],
        'power': [100.0],
        'infos': [{
            'solar_zenith_deg': 30.0, 'solar_azimuth_deg': 180.0,
            'dni_wm2': 600.0, 'ghi_wm2': 700.0, 'dhi_wm2': 100.0,
            'temperature_c': 25.0, 'wind_speed_m_s': 2.0,
            'surface_albedo': 0.31, 'albedo_source': 'nsrdb_recorded',
        }],
    }
    result = refs.selected_instantaneous_dual_axis(None, fixed, env=env)
    assert captured['surface_albedo'] == pytest.approx(0.31)
    assert result['infos'][0]['surface_albedo'] == pytest.approx(0.31)
