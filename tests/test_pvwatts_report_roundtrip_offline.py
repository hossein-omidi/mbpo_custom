"""Offline output-contract regression on the user's exact local Stage 2 report module.

Intentionally uses complete 117-step fixture and real exporter/parser, not a
stub. Does NOT claim actual PV energy accuracy.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import importlib.util
import math


def _load_report_module():
    # This test is installed into repo/tests; read the real installed module
    # rather than a second private copy.
    path = Path(__file__).resolve().parents[1] / 'MBPO_workspace' / 'src' / 'advanced_suite.py'
    spec = importlib.util.spec_from_file_location('_stage2_reporting_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage2_detailed_csv_reimport_for_all_four_visible_methods(tmp_path):
    mod = _load_report_module()
    n = 117
    paths = []
    for method in ('mbpo', 'sac', 'fixed_no_motion', 'poa_instant_ideal'):
        ideal = method == 'poa_instant_ideal'
        power = 130.0
        energies = [power / 12000.0] * n
        infos = []
        for i in range(n):
            ts = datetime(2022, 7, 15, 12, 5, tzinfo=timezone.utc) + timedelta(minutes=5*i)
            infos.append(dict(
                timestamp_utc_iso=ts.isoformat(), solar_zenith_deg=50.0,
                solar_azimuth_deg=180.0, dni_wm2=700.0,
                ghi_wm2=800.0, dhi_wm2=100.0, weather_condition='clear',
                power_model='pvwatts_sapm', poa_global=800.0,
                power_dc_w=power, effective_irradiance_wm2=800.0,
                cell_temperature_c=40.0, ambient_temperature_c=30.0,
                wind_speed_m_s=2.0, wind_speed_10m_ms=2.5,
                wind_reference_height_m=2.0, gamma_pdc=-0.004,
                sapm_mount='open_rack_glass_polymer', wind_powerlaw_exponent=0.14,
            ))
        paths.append(dict(method=method,scenario_id='2022-07-15',seed=42,
            initial_pose=[30.0,180.0],
            power=[power]*n,energy=energies,reward=energies,
            cost=[0.0]*n,tilt=[30.0]*n,azimuth=[180.0]*n,
            action_tilt=[math.nan if ideal else 0.0]*n,
            action_azimuth=[math.nan if ideal else 0.0]*n,
            delta_tilt_deg=[math.nan if ideal else 0.0]*n,
            delta_azimuth_deg=[math.nan if ideal else 0.0]*n,infos=infos,
        ))
    fn = mod.export_detail(paths,tmp_path)
    assert fn.is_file()
    grouped,keys,source,rich=mod._read_eval(tmp_path)
    assert rich is True and source==fn
    assert len(grouped)==4 and len(keys)==1
    for (sid,seed,method),rows in grouped.items():
        assert len(rows)==117
        assert rows[0]['power_model']=='pvwatts_sapm'
        assert rows[0]['sapm_mount']=='open_rack_glass_polymer'
        assert rows[0]['wind_speed_10m_ms']==2.5
        assert rows[0]['cell_temperature_c']==40.0
