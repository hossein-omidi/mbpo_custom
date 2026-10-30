"""Tests the actual Stage 2 evaluation audit contract without external NSRDB files.

Run via: python -m pytest -q tests/test_pvwatts_stage2_audit_contract.py
These tests exercise the fail-closed checker, not the physical pvlib implementation;
the latter is covered by the existing 26 Stage 1/2 runtime tests and the real audit.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    '_stage2_audit_under_test', ROOT / 'scripts' / 'pvwatts_stage2_audit.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

METHODS = ('mbpo', 'sac', 'fixed_no_motion', 'dual_axis_slew_limited',
           'dual_axis_instant_ideal', 'poa_instant_ideal')


def fixture():
    physical = SimpleNamespace(
        interval_hours=5/60, power_model='pvwatts_sapm', area=1.0,
        efficiency=0.18, wind_reference_height_m=2.0,
        gamma_pdc=-0.004, sapm_mount='open_rack_glass_polymer',
        wind_powerlaw_exponent=0.14,
    )
    env = SimpleNamespace(unwrapped=physical, horizon=1)
    sample = {
        'power_model': 'pvwatts_sapm',
        'power_dc_w': 135.0,
        'poa_global': 800.0,
        'effective_irradiance_wm2': 800.0,
        'cell_temperature_c': 40.0,
        'wind_speed_10m_ms': 3.0,
        'temperature_c': 30.0,
        'wind_speed_m_s': 2.5,
        'solar_zenith_deg': 35.0,
        'solar_azimuth_deg': 170.0,
        'dni_wm2': 700.0,
        'ghi_wm2': 850.0,
        'dhi_wm2': 150.0,
    }
    energy = 135.0 / 12000.0
    paths = [{
        'method': method, 'power': [135.0], 'energy': [energy],
        'reward': [energy], 'cost': [0.0], 'tilt': [20.0],
        'azimuth': [170.0], 'infos': [dict(sample)],
    } for method in METHODS]
    def canonical(tilt, az, z, a, dni, ghi, dhi, **kw):
        assert (tilt, az, z, a, dni, ghi, dhi) == (
            20.0, 170.0, 35.0, 170.0, 700.0, 850.0, 150.0)
        assert kw['power_model'] == 'pvwatts_sapm'
        return SimpleNamespace(power_w=135.0, poa_global_wm2=800.0,
                               effective_irradiance_wm2=800.0,
                               cell_temperature_c=40.0, wind_speed_10m_ms=3.0)
    return paths, env, canonical


def test_all_six_evaluation_methods_valid_without_info_power_alias():
    paths, env, fn = fixture()
    assert all('power' not in p['infos'][0] for p in paths)
    assert AUDIT.validate_paired_power_paths(paths, METHODS, env, fn) == 6


def test_missing_canonical_dc_diagnostic_is_not_silently_substituted():
    paths, env, fn = fixture()
    del paths[0]['infos'][0]['power_dc_w']
    with pytest.raises(RuntimeError, match='Missing evaluation physics data'):
        AUDIT.validate_paired_power_paths(paths, METHODS, env, fn)


def test_disagreement_with_dc_diagnostic_rejected():
    paths, env, fn = fixture()
    paths[0]['infos'][0]['power_dc_w'] += 2
    with pytest.raises(RuntimeError, match='DC diagnostic disagrees'):
        AUDIT.validate_paired_power_paths(paths, METHODS, env, fn)


def test_backend_power_difference_rejected_even_if_output_fields_agree():
    paths, env, fn = fixture()
    paths[0]['power'][0] = paths[0]['infos'][0]['power_dc_w'] = 136.0
    paths[0]['energy'][0] = paths[0]['reward'][0] = 136.0 / 12000
    with pytest.raises(RuntimeError, match='DC power inconsistent with selected pvlib model'):
        AUDIT.validate_paired_power_paths(paths, METHODS, env, fn)


def test_mixed_model_rejected():
    paths, env, fn = fixture()
    paths[3]['infos'][0]['power_model'] = 'legacy'
    with pytest.raises(RuntimeError, match='mixed physical backends'):
        AUDIT.validate_paired_power_paths(paths, METHODS, env, fn)


def test_energy_reward_and_temperature_errors_rejected():
    for modification, expected_error in (
        (lambda p: p['energy'].__setitem__(0, p['energy'][0]+0.01), 'energy disagrees'),
        (lambda p: p['reward'].__setitem__(0, p['reward'][0]+0.01), 'reward/energy/cost'),
        (lambda p: p['infos'][0].__setitem__('cell_temperature_c', 40.5),
         'cell temperature inconsistent'),
    ):
        paths, env, fn = fixture()
        modification(paths[0])
        with pytest.raises(RuntimeError, match=expected_error):
            AUDIT.validate_paired_power_paths(paths, METHODS, env, fn)


def test_incomplete_method_coverage_and_duplicate_rejected():
    paths, env, fn = fixture()
    with pytest.raises(RuntimeError, match='method coverage mismatch'):
        AUDIT.validate_paired_power_paths(paths[:-1], METHODS, env, fn)
    with pytest.raises(RuntimeError, match='Duplicate evaluation method'):
        AUDIT.validate_paired_power_paths(paths + [paths[0]], METHODS, env, fn)
