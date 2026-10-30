"""Pure-source surgical patch checks; no pvlib, NSRDB or torch needed."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest


def _load():
    path=Path(__file__).resolve().parents[1]/'scripts'/'apply_pvwatts_stage2.py'
    spec=importlib.util.spec_from_file_location('_stage2_install',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_all_target_files_have_patchers():
    p=_load()
    assert len(p.TARGETS)==13
    assert set(p.TARGETS)==set(p.PATCHES)
    assert len(p.TARGETS)==len(set(p.TARGETS))


def test_patch_env_syntax_and_guarded_source():
    p=_load()
    original='''from mbpo.env.pvlib_physics import (
    compute_panel_power_w,
)
class PVTrackingEnv:
    def __init__(self,
        wind_speed=2.0,
        movement_penalty=0.01,):
        self.wind_speed_variation = 1.0
    def _current_weather(self):
        return {'dni':0., 'ghi':0., 'dhi':0.,'temperature':25.,'wind_speed':2.}
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
        return power
    def reset(self):
        power = self._power_from_orientation(
            solar_position.zenith,
            solar_position.azimuth,
            self.tilt,
            self.azimuth,
        )
        return power
    def step(self):
        power = self._power_from_orientation(
            solar_position.zenith,
            solar_position.azimuth,
            self.tilt,
            self.azimuth,
        )
        info = {
            'poa_global': float(power / self.efficiency / self.area),
        }
        info.update(time_meta)
        return info
'''
    updated=p.patch_env(original)
    compile(updated,'<mock-v2-env>','exec')
    assert updated.count('power_result = self._power_result_from_orientation(')==2
    assert "'poa_global': float(power_result.poa_global_wm2)" in updated
    assert "'cell_temperature_c': power_result.cell_temperature_c" in updated
    assert updated.count('def _power_result_from_orientation(')==1
    assert updated.count('def _power_from_orientation(')==1
    with pytest.raises(RuntimeError,match='anchor'):
        p.patch_env(updated)


def test_sac_config_patch_compiles_simple_fixture():
    p=_load()
    original='''def load_config(require_data: bool = True) -> SACConfig:
    nsrdb={};eval_env={}
    for env in (nsrdb, eval_env):
        env['scenario_manifest'] = 'dataset.json'
    return nsrdb,eval_env
'''
    patched=p.patch_sac_config(original)
    compile(patched,'<mock-config>','exec')
    assert 'nsrdb.update(physics)' in patched
    assert 'eval_env.update(physics)' in patched
    assert "power_model: str = \"legacy\"" in patched


def test_patcher_refuses_unrecognized_sources():
    p=_load()
    for path,patch in p.PATCHES.items():
        with pytest.raises(RuntimeError):
            patch('')


def test_mbpo_selected_references_patch_source_and_compile():
    p=_load()
    original='''PHYSICS_KEYS=('pvlib_version','observation_mode','horizon')

def ideal_instantaneous_poa():
    from mbpo.env.pvlib_physics import compute_panel_power_w
    for info in []:
        def value(tilt):
                return compute_panel_power_w(float(tilt),az,info['solar_zenith_deg'],
                         info['solar_azimuth_deg'],info['dni_wm2'],
                         info['ghi_wm2'],info['dhi_wm2'],area=area,efficiency=efficiency)
        if True:
            power=float(value(tilt));energy=power*hours/1000.
            p['infos'].append(dict(info))

def paired_summary(paths,plan=None):
    out={'notes':['Instantaneous POA optimizer maximizes single-step gross power for the '
                  'isotropic transposition model, not global finite-horizon net energy.']}
    return out

def evaluate():
    if True:
                ideal=ideal_instantaneous_dual_axis(cfg.sac,fixed,env=env)

def write_outputs(paths,plan):
    stats=paired_summary(paths,plan=plan)
    return stats
'''
    s=p.patch_mbpo_evaluation(original)
    compile(s,'<MBPO mocked eval>','exec')
    assert 'selected_instantaneous_dual_axis(cfg.sac,fixed,env=env)' in s
    assert 'from mbpo.env.pv_power_backend import compute_pv_power' in s
    assert "stats['power_physics']=physics" in s
    assert 'cell_temperature_c=result.cell_temperature_c' in s


def test_mbpo_interpretation_accepts_reworded_local_note_and_compiles():
    p=_load()
    fixture="""def paired_summary(paths, plan=None):
    out={'notes': [
        'The old POA ideal was an instantaneous geometry-only comparator, '
        'not an actuator feasible net-energy optimum.',
        'No held-out estimate is implied.',
    ]}
    return out
"""
    altered=p.patch_summary_interpretation(fixture)
    compile(altered,'<local-notes-variant>','exec')
    assert 'selected-model single-step gross DC power' in altered
    assert 'No held-out estimate is implied' in altered
    assert 'geometry-only comparator' in altered
    assert altered.count('selected-model single-step gross DC power')==1
    with pytest.raises(RuntimeError,match='already present'):
        p.patch_summary_interpretation(altered)


def test_mbpo_interpretation_handles_single_quote_double_quote_and_wraps():
    p=_load()
    fixture="""def paired_summary(paths):
    out={'notes': [
        "Instantaneous POA optimizer used isotropic transposition model, "
        "so gross power is not globally net-energy optimal.",
        'Separate note.']}
    return out
"""
    altered=p.patch_summary_interpretation(fixture)
    compile(altered,'<old-notes-variant>','exec')
    assert 'isotropic transposition model' not in altered
    assert 'selected-model single-step gross DC power' in altered
    assert altered.count('selected-model single-step gross DC power')==1


def test_advanced_csv_parser_treats_power_model_as_text():
    p=_load()
    source='''DETAIL=('other',
        'solar_azimuth_deg','dni_wm2','dhi_wm2','ghi_wm2','weather_condition')
LABELS={'poa_instant_ideal':'Instant POA (gross-only)'}
parsed={k:(_float(row.get(k)) if k not in ('scenario_id','method','utc_iso','weather_condition') else row.get(k)) for k in ALL}
row=dict(
                         ghi_wm2=info.get('ghi_wm2'),weather_condition=info.get('weather_condition'))
'''
    after=p.patch_advanced(source)
    compile(after,'<advanced mock>','exec')
    assert "'weather_condition','power_model')" in after
    assert "power_model=info.get('power_model')" in after
    assert "'poa_instant_ideal':'Instant DC reference (gross-only)'" in after


def test_adapter_guard_compiles_and_records_fingerprints():
    p=_load()
    source='''class SACEnvironment:
    def __init__(self):
        self.resolved_mdp = {
        }

    def seed(self, seed: int) -> None:
        pass

    def step(self):
        return observation, float(reward), bool(done), info
'''
    patched=p.patch_adapter(source)
    compile(patched,'<adapter-mock>','exec')
    assert 'pv_power_backend_sha256' in patched
    assert 'pvlib_physics_sha256' in patched
    assert 'Real transition uses an unexpected DC backend' in patched
    assert 'DC power diagnostic disagrees with real power' in patched
    with pytest.raises(RuntimeError):p.patch_adapter(patched)


def test_stage2_audit_never_disables_model_skill_gate():
    audit=(Path(__file__).resolve().parents[1]/'scripts'/'pvwatts_stage2_audit.py').read_text()
    assert 'require_model_skill=False' not in audit
    assert "skipped=bool(trainer.latest_model_metrics.get('model_skipped'))" in audit
    assert "'skill-gated replay" in audit
    assert 'trainer.fake_env.step(' in audit
