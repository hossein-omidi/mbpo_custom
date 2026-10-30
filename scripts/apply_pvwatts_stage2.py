"""Fail-closed, transactional Stage-2 integration of the Stage-1 PVWatts backend.

  python scripts/apply_pvwatts_stage2.py --check
  python scripts/apply_pvwatts_stage2.py --apply

The script patches only recognizable source anchors. It compiles ALL prospective
Python files before changing any file; writes a complete ZIP of originals; uses
atomic replacement and restores originals on failure. It never touches datasets,
trained checkpoints, existing run directories, or saved evaluations.

No network is required. Review the printed diff summary and backups.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_ADDON_SHA256 = {
    'mbpo/env/pv_power_backend.py':
        '0eff58d4e589238b84ea7b64176a539217f7569e3aa386d85bdc90fc0adf08f9',
    'mbpo/env/pv_power_cli.py':
        '180ee59fc9fe2567f9f19f3e17410141edaac87749a2c8a170464c8176f5ef9d',
    'MBPO_workspace/src/power_references.py':
        '7842f215dd81a5190e68f9ff2cd355607914cb7330d347596865a46be03bd30c',
}

TARGETS = (
    'mbpo/env/pv_tracking.py',
    'SAC_workspace/src/config/config.py',
    'SAC_workspace/src/environment/environment.py',
    'MBPO_workspace/src/config.py',
    'MBPO_workspace/src/evaluation.py',
    'MBPO_workspace/src/advanced_suite.py',
    'SAC_workspace/workflows/01_setup_config.py',
    'SAC_workspace/workflows/02_environment_replay.py',
    'SAC_workspace/workflows/03_train.py',
    'SAC_workspace/workflows/04_evaluate.py',
    'MBPO_workspace/workflows/01_setup_config.py',
    'MBPO_workspace/workflows/02_train.py',
    'MBPO_workspace/workflows/03_evaluate.py',
)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def replace(s: str, old: str, new: str, desc: str, times: int = 1) -> str:
    observed = s.count(old)
    if observed != times:
        raise RuntimeError(f'{desc}: expected {times} unique source anchor(s), found {observed}. '
                           'Your local file differs; no files will be changed.')
    return s.replace(old, new)


def splice(s: str, start: str, end: str, new_middle: str, desc: str) -> str:
    if s.count(start) != 1 or s.count(end) != 1:
        raise RuntimeError(f'{desc}: cannot recognize source method boundaries; refusing change')
    x = s.index(start) + len(start)
    y = s.index(end, x)
    return s[:x] + new_middle + s[y:]


def patch_env(s: str) -> str:
    s = replace(s, 'from mbpo.env.pvlib_physics import (',
        'from mbpo.env.pv_power_backend import compute_pv_power\n\nfrom mbpo.env.pvlib_physics import (',
        'PVTrackingEnv import')
    s = replace(s, "        wind_speed=2.0,\n        movement_penalty=0.01,",
        "        wind_speed=2.0,\n"
        "        power_model='legacy',\n"
        "        wind_reference_height_m=None,\n"
        "        gamma_pdc=-0.004,\n"
        "        sapm_mount='open_rack_glass_polymer',\n"
        "        wind_powerlaw_exponent=0.14,\n"
        "        movement_penalty=0.01,", 'PVTrackingEnv constructor')
    s = replace(s, '        self.wind_speed_variation = 1.0\n',
        '        self.wind_speed_variation = 1.0\n'
        '        self.power_model = str(power_model)\n'
        '        self.wind_reference_height_m = wind_reference_height_m\n'
        '        self.gamma_pdc = float(gamma_pdc)\n'
        '        self.sapm_mount = str(sapm_mount)\n'
        '        self.wind_powerlaw_exponent = float(wind_powerlaw_exponent)\n'
        "        if self.power_model not in ('legacy', 'pvwatts_sapm'):\n"
        "            raise ValueError('Unknown power_model: ' + self.power_model)\n"
        "        if self.power_model == 'pvwatts_sapm' and wind_reference_height_m is None:\n"
        "            raise ValueError('PVWatts requires explicitly declared wind_reference_height_m')\n",
        'PVTrackingEnv physical attributes')
    start = '    def _power_from_orientation(self, solar_zenith, solar_azimuth, tilt, azimuth):\n'
    end = '    def _build_observation(self, solar_zenith, solar_azimuth, weather, power):\n'
    middle = (
        '        """Backward-compatible scalar power helper."""\n'
        '        return self._power_result_from_orientation(\n'
        '            solar_zenith, solar_azimuth, tilt, azimuth).power_w\n\n'
        '    def _power_result_from_orientation(self, solar_zenith, solar_azimuth, tilt, azimuth):\n'
        '        """One canonical backend for reset, step and recorded physical diagnostics."""\n'
        '        weather = self._current_weather()\n'
        '        return compute_pv_power(\n'
        '            tilt, azimuth, solar_zenith, solar_azimuth,\n'
        "            weather['dni'], weather['ghi'], weather['dhi'],\n"
        '            area=self.area, efficiency=self.efficiency,\n'
        '            power_model=self.power_model,\n'
        "            temp_air_c=weather['temperature'], wind_speed_ms=weather['wind_speed'],\n"
        '            wind_reference_height_m=self.wind_reference_height_m,\n'
        '            gamma_pdc=self.gamma_pdc, sapm_mount=self.sapm_mount,\n'
        '            wind_powerlaw_exponent=self.wind_powerlaw_exponent,\n'
        '        )\n\n'
    )
    s = splice(s, start, end, middle, 'PVTrackingEnv central power method')
    call = ('        power = self._power_from_orientation(\n'
            '            solar_position.zenith,\n'
            '            solar_position.azimuth,\n'
            '            self.tilt,\n'
            '            self.azimuth,\n'
            '        )\n')
    new_call = ('        power_result = self._power_result_from_orientation(\n'
                '            solar_position.zenith,\n'
                '            solar_position.azimuth,\n'
                '            self.tilt,\n'
                '            self.azimuth,\n'
                '        )\n'
                '        power = power_result.power_w\n')
    s = replace(s, call, new_call, 'PVTrackingEnv reset and step shared backend', times=2)
    s = replace(s, "'poa_global': float(power / self.efficiency / self.area),",
                "'poa_global': float(power_result.poa_global_wm2),", 'true POA info field')
    s = replace(s, '        info.update(time_meta)\n',
        '        info.update(time_meta)\n'
        "        info.update({\n"
        "            'power_model': self.power_model,\n"
        "            'power_dc_w': float(power_result.power_w),\n"
        "            'effective_irradiance_wm2': float(power_result.effective_irradiance_wm2),\n"
        "            'cell_temperature_c': power_result.cell_temperature_c,\n"
        "            'wind_speed_m_s': float(weather['wind_speed']),\n"
        "            'wind_speed_10m_ms': power_result.wind_speed_10m_ms,\n"
        "            'wind_reference_height_m': self.wind_reference_height_m,\n"
        "            'gamma_pdc': self.gamma_pdc if self.power_model != 'legacy' else None,\n"
        "            'sapm_mount': self.sapm_mount if self.power_model != 'legacy' else None,\n"
        "            'wind_powerlaw_exponent': self.wind_powerlaw_exponent if self.power_model != 'legacy' else None,\n"
        "        })\n", 'power diagnostics')
    return s


def patch_sac_config(s: str) -> str:
    old = 'def load_config(require_data: bool = True) -> SACConfig:'
    new = ('def load_config(require_data: bool = True, *, power_model: str = "legacy",\n'
           '                wind_reference_height_m: float = 2.0,\n'
           '                gamma_pdc: float = -0.004,\n'
           '                sapm_mount: str = "open_rack_glass_polymer",\n'
           '                wind_powerlaw_exponent: float = 0.14) -> SACConfig:')
    s = replace(s, old, new, 'SAC load_config signature')
    anchor = "    for env in (nsrdb, eval_env):\n        env['scenario_manifest']"
    new_anchor = (
        "    if power_model not in ('legacy', 'pvwatts_sapm'):\n"
        "        raise ValueError('Unsupported power model: ' + str(power_model))\n"
        "    if power_model == 'pvwatts_sapm':\n"
        "        # Explicit, provisional physical assumptions; keep legacy config untouched.\n"
        "        physics = dict(power_model=power_model,\n"
        "                       wind_reference_height_m=float(wind_reference_height_m),\n"
        "                       gamma_pdc=float(gamma_pdc), sapm_mount=str(sapm_mount),\n"
        "                       wind_powerlaw_exponent=float(wind_powerlaw_exponent))\n"
        "        # Validate before starting any costly experiment.\n"
        "        from mbpo.env.pv_power_backend import compute_pv_power\n"
        "        compute_pv_power(30., 180., 35., 180., 700., 800., 140.,\n"
        "                         temp_air_c=25., wind_speed_ms=2., **physics)\n"
        "        nsrdb.update(physics)\n"
        "        eval_env.update(physics)\n"
        "    for env in (nsrdb, eval_env):\n        env['scenario_manifest']"
    )
    s = replace(s, anchor, new_anchor, 'SAC shared train/eval physics configuration')
    return s


def patch_adapter(s: str) -> str:
    s = replace(s, "        }\n\n    def seed(self, seed: int) -> None:",
        "        }\n"
        "        # Preserve byte-for-byte legacy MDP metadata for prior checkpoints.\n"
        "        # New PVWatts checkpoints record all parameters affecting DC power.\n"
        "        if source.power_model != 'legacy':\n"
        "            for key in ('power_model', 'wind_reference_height_m',\n"
        "                        'gamma_pdc', 'sapm_mount', 'wind_powerlaw_exponent'):\n"
        "                self.resolved_mdp[key] = value(key)\n"
        "            # Code fingerprints forbid silent checkpoint reuse after physics changes.\n"
        "            import hashlib\n"
        "            from pathlib import Path\n"
        "            repo_root = Path(__file__).resolve().parents[3]\n"
        "            for key, relative in (\n"
        "                ('pv_power_backend_sha256', 'mbpo/env/pv_power_backend.py'),\n"
        "                ('pvlib_physics_sha256', 'mbpo/env/pvlib_physics.py'),\n"
        "            ):\n"
        "                self.resolved_mdp[key] = hashlib.sha256(\n"
        "                    (repo_root / relative).read_bytes()).hexdigest()\n"
        "\n    def seed(self, seed: int) -> None:", 'resolved MDP physics snapshot')
    # This validation runs in both SAC and MBPO real transitions.
    return replace(s,
        '        return observation, float(reward), bool(done), info\n',
        '        if info.get("power_model") != self.env.power_model:\n'
        '            raise RuntimeError("Real transition uses an unexpected DC backend")\n'
        '        if not np.isclose(float(info["power_dc_w"]), float(info["power"]),\n'
        '                          rtol=1e-10, atol=1e-10):\n'
        '            raise RuntimeError("DC power diagnostic disagrees with real power")\n'
        '        if (not np.isfinite(info["poa_global"])\n'
        '                or not np.isfinite(info["power_dc_w"])):\n'
        '            raise RuntimeError("Nonfinite POA or DC power")\n'
        '        return observation, float(reward), bool(done), info\n',
        'SAC/MBPO shared real-transition backend guard')


def patch_mbpo_config(s: str) -> str:
    return replace(s, 'def load_config(require_data=True, model_overrides=None) -> MBPOConfig:\n'
        '    sac = load_sac_config(require_data=require_data)',
        'def load_config(require_data=True, model_overrides=None, **power_kwargs) -> MBPOConfig:\n'
        '    sac = load_sac_config(require_data=require_data, **power_kwargs)',
        'MBPO shared config delegation')


def patch_summary_interpretation(source: str) -> str:
    """Update the *meaning* of the ideal DC reference, not an exact note spelling.

    The local evaluation file may have an edited or rewrapped notes paragraph.
    We validate the AST shape of paired_summary's notes list, modify one
    relevant entry if it exists, or add a clarification if not. All executable
    evaluation/power/checkpoint anchors remain strict in patch_mbpo_evaluation.
    """
    tree = ast.parse(source)
    summary = [node for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == 'paired_summary']
    if len(summary) != 1:
        raise RuntimeError('MBPO evaluation interpretation: paired_summary must exist uniquely')
    candidates = []
    for stmt in ast.walk(summary[0]):
        if (isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == 'out'
                and isinstance(stmt.value, ast.Dict)):
            for key, value in zip(stmt.value.keys, stmt.value.values):
                if isinstance(key, ast.Constant) and key.value == 'notes':
                    candidates.append(value)
    if len(candidates) != 1 or not isinstance(candidates[0], ast.List):
        raise RuntimeError('MBPO evaluation interpretation: cannot safely identify paired_summary notes list')
    notes = candidates[0]
    if not all(isinstance(item, ast.Constant) and isinstance(item.value, str)
               for item in notes.elts):
        raise RuntimeError('MBPO evaluation interpretation: notes contain non-literal text')
    new_note = ('Instantaneous reference maximizes selected-model single-step gross DC power, '
                'not global finite-horizon net energy; method ID poa_instant_ideal is retained '
                'for compatibility.')
    existing = [item for item in notes.elts if
                ('POA optimizer' in item.value or
                 ('instantaneous' in item.value.lower() and
                  'isotropic transposition model' in item.value.lower()))]
    if len(existing) > 1:
        raise RuntimeError('MBPO evaluation interpretation: ambiguous duplicate model descriptions')
    lines = source.splitlines(keepends=True)
    def loc(lineno: int, col: int) -> int:
        # AST column offsets are UTF-8 byte offsets. Repository source is UTF-8.
        return sum(len(line) for line in lines[:lineno-1]) + len(
            lines[lineno-1].encode('utf-8')[:col].decode('utf-8'))
    if existing:
        item = existing[0]
        start, end = loc(item.lineno, item.col_offset), loc(item.end_lineno, item.end_col_offset)
        return source[:start] + repr(new_note) + source[end:]
    if any('selected-model single-step gross DC power' in x.value for x in notes.elts):
        raise RuntimeError('Stage 2 model interpretation is already present; refusing duplicate patch')
    # No stale note exists: insert an unambiguous truthful note into the guarded
    # AST-identified report list. Do NOT silently skip the reporting dependency.
    close = loc(notes.end_lineno, notes.end_col_offset) - 1
    if source[close] != ']':
        raise RuntimeError('MBPO evaluation interpretation: invalid notes-list boundary')
    prior = source[:close].rstrip()
    comma = ', ' if notes.elts and not prior.endswith(',') else ' '
    return source[:close] + comma + repr(new_note) + source[close:]


def patch_mbpo_evaluation(s: str) -> str:
    s = replace(s, "'observation_mode','horizon')", "'observation_mode','horizon',\n"
        "    'power_model','wind_reference_height_m','gamma_pdc','sapm_mount',\n"
        "    'wind_powerlaw_exponent','pv_power_backend_sha256',\n"
        "    'pvlib_physics_sha256')", 'MBPO checkpoint physics identity')
    s = replace(s, '                ideal=ideal_instantaneous_dual_axis(cfg.sac,fixed,env=env)',
        '                from MBPO_workspace.src.power_references import selected_instantaneous_dual_axis\n'
        '                ideal=selected_instantaneous_dual_axis(cfg.sac,fixed,env=env)',
        'MBPO selected-model classical ideal')
    s = replace(s, '    from mbpo.env.pvlib_physics import compute_panel_power_w\n',
        '    from mbpo.env.pv_power_backend import compute_pv_power\n', 'MBPO ideal reference canonical backend')
    old = ("                return compute_panel_power_w(float(tilt),az,info['solar_zenith_deg'],\n"
           "                         info['solar_azimuth_deg'],info['dni_wm2'],\n"
           "                         info['ghi_wm2'],info['dhi_wm2'],area=area,efficiency=efficiency)\n")
    new = (
        "                return compute_pv_power(float(tilt),az,info['solar_zenith_deg'],\n"
        "                         info['solar_azimuth_deg'],info['dni_wm2'],\n"
        "                         info['ghi_wm2'],info['dhi_wm2'],area=area,efficiency=efficiency,\n"
        "                         power_model=u.power_model,\n"
        "                         temp_air_c=info.get('temperature_c'),\n"
        "                         wind_speed_ms=info.get('wind_speed_m_s'),\n"
        "                         wind_reference_height_m=u.wind_reference_height_m,\n"
        "                         gamma_pdc=u.gamma_pdc, sapm_mount=u.sapm_mount,\n"
        "                         wind_powerlaw_exponent=u.wind_powerlaw_exponent).power_w\n"
    )
    s = replace(s, old, new, 'MBPO ideal selected-model power')
    s = replace(s, '            power=float(value(tilt));energy=power*hours/1000.\n',
        "            result=compute_pv_power(float(tilt),az,info['solar_zenith_deg'],\n"
        "                         info['solar_azimuth_deg'],info['dni_wm2'],\n"
        "                         info['ghi_wm2'],info['dhi_wm2'],area=area,efficiency=efficiency,\n"
        "                         power_model=u.power_model, temp_air_c=info.get('temperature_c'),\n"
        "                         wind_speed_ms=info.get('wind_speed_m_s'),\n"
        "                         wind_reference_height_m=u.wind_reference_height_m,\n"
        "                         gamma_pdc=u.gamma_pdc, sapm_mount=u.sapm_mount,\n"
        "                         wind_powerlaw_exponent=u.wind_powerlaw_exponent)\n"
        "            power=float(result.power_w);energy=power*hours/1000.\n",
        'MBPO ideal physical reporting')
    s = replace(s, "            p['infos'].append(dict(info))\n",
        "            ideal_info=dict(info)\n"
        "            ideal_info.update(power=power, power_dc_w=power,\n"
        "                poa_global=float(result.poa_global_wm2),\n"
        "                effective_irradiance_wm2=float(result.effective_irradiance_wm2),\n"
        "                cell_temperature_c=result.cell_temperature_c,\n"
        "                wind_speed_10m_ms=result.wind_speed_10m_ms,\n"
        "                power_model=u.power_model, tilt=float(tilt), azimuth=float(az),\n"
        "                cos_aoi=u._cos_angle_of_incidence(\n"
        "                    info['solar_zenith_deg'],info['solar_azimuth_deg'],tilt,az))\n"
        "            p['infos'].append(ideal_info)\n", 'ideal reference info provenance')
    s = patch_summary_interpretation(s)
    s = replace(s,
      '    stats=paired_summary(paths,plan=plan)\n',
      "    stats=paired_summary(paths,plan=plan)\n"
      "    physics_rows=[p['infos'][0] for p in paths if p.get('infos')]\n"
      "    if physics_rows:\n"
      "        keys=('power_model','wind_reference_height_m','gamma_pdc',\n"
      "              'sapm_mount','wind_powerlaw_exponent')\n"
      "        physics={key:physics_rows[0].get(key) for key in keys}\n"
      "        if any(any(info.get(k)!=physics[k] for k in keys) for info in physics_rows):\n"
      "            raise ValueError('Mixed physical-model definitions in evaluation paths')\n"
      "        stats['power_physics']=physics\n",
      'paired report physical-model identity')
    return s


def patch_advanced(s: str) -> str:
    s = replace(s,
        "        'solar_azimuth_deg','dni_wm2','dhi_wm2','ghi_wm2','weather_condition')",
        "        'solar_azimuth_deg','dni_wm2','dhi_wm2','ghi_wm2','weather_condition',\n"
        "        'poa_global_wm2','effective_irradiance_wm2','cell_temperature_c',\n"
        "        'ambient_temperature_c','wind_speed_m_s','wind_speed_10m_ms',\n"
        "        'power_model')", 'advanced report extended schema')
    s = replace(s,
        "                         ghi_wm2=info.get('ghi_wm2'),weather_condition=info.get('weather_condition'))",
        "                         ghi_wm2=info.get('ghi_wm2'),weather_condition=info.get('weather_condition'),\n"
        "                         poa_global_wm2=info.get('poa_global'),\n"
        "                         effective_irradiance_wm2=info.get('effective_irradiance_wm2'),\n"
        "                         cell_temperature_c=info.get('cell_temperature_c'),\n"
        "                         ambient_temperature_c=info.get('temperature_c'),\n"
        "                         wind_speed_m_s=info.get('wind_speed_m_s'),\n"
        "                         wind_speed_10m_ms=info.get('wind_speed_10m_ms'),\n"
        "                         power_model=info.get('power_model'))",
        'advanced report export physical channels')
    s = replace(s,
        "('scenario_id','method','utc_iso','weather_condition')",
        "('scenario_id','method','utc_iso','weather_condition','power_model')",
        'advanced CSV new categorical physics column')
    s = replace(s, "'poa_instant_ideal':'Instant POA (gross-only)'",
        "'poa_instant_ideal':'Instant DC reference (gross-only)'",
        'advanced figures truthful reference label')
    return s


CLI_IMPORT = 'from mbpo.env.pv_power_cli import add_power_cli_arguments, power_cli_kwargs\n'

def patch_workflow(s: str, path: str) -> str:
    # Preserve original invocation semantics with default legacy; new experiments
    # explicitly specify --power-model pvwatts_sapm in all four workflow CLIs.
    if 'SAC_workspace/workflows/01_setup_config.py' == path:
        s=replace(s, 'from pathlib import Path\n', 'from pathlib import Path\nimport argparse\n'+CLI_IMPORT, 'SAC setup imports')
        s=replace(s, '    c = load_config(require_data=True)',
          '    parser=argparse.ArgumentParser()\n    add_power_cli_arguments(parser)\n    args=parser.parse_args()\n    c = load_config(require_data=True, **power_cli_kwargs(args))', 'SAC setup physics')
        s=replace(s,"    dest = ROOT / 'SAC_workspace' / 'runs' / 'reference_config.json'",
          "    dest = ROOT / 'SAC_workspace' / 'runs' / (\n"
          "        'reference_config_pvwatts.json' if args.power_model == 'pvwatts_sapm'\n"
          "        else 'reference_config.json')", 'SAC separate setup snapshot')
    elif 'SAC_workspace/workflows/02_environment_replay.py' == path:
        s=replace(s, 'import numpy as np\n', 'import numpy as np\nimport argparse\n'+CLI_IMPORT, 'SAC replay imports')
        s=replace(s, '    config = load_config()',
          '    parser=argparse.ArgumentParser()\n    add_power_cli_arguments(parser)\n    args=parser.parse_args()\n    config = load_config(**power_cli_kwargs(args))', 'SAC replay config')
    elif 'MBPO_workspace/workflows/01_setup_config.py' == path:
        s=replace(s, 'import json\n', 'import json\nimport argparse\n'+CLI_IMPORT, 'MBPO setup imports')
        s=replace(s, '    c=load_config()',
          '    parser=argparse.ArgumentParser()\n    add_power_cli_arguments(parser)\n    args=parser.parse_args()\n    c=load_config(**power_cli_kwargs(args))', 'MBPO setup physics')
        s=replace(s,"    p=ROOT/'MBPO_workspace/runs/reference_config.json'",
          "    p=ROOT/'MBPO_workspace'/'runs'/('reference_config_pvwatts.json'\n"
          "                                  if args.power_model == 'pvwatts_sapm'\n"
          "                                  else 'reference_config.json')", 'MBPO separate setup snapshot')
    else:
        # Five? main train/eval CLIs: SAC 03, SAC 04, MBPO 02, MBPO 03.
        if 'import argparse\n' not in s:
            raise RuntimeError(f'{path}: argparse import missing')
        s = replace(s, 'import argparse\n', 'import argparse\n'+CLI_IMPORT, f'{path} CLI import')
        if path.endswith('SAC_workspace/workflows/03_train.py'):
            anchor = "    args = p.parse_args()\n    cfg = load_config()"
            new = "    add_power_cli_arguments(p)\n    args = p.parse_args()\n    cfg = load_config(**power_cli_kwargs(args))"
        elif path.endswith('SAC_workspace/workflows/04_evaluate.py'):
            anchor = "    args=ap.parse_args()"
            new = "    add_power_cli_arguments(ap)\n    args=ap.parse_args()"
        elif path.endswith('MBPO_workspace/workflows/02_train.py'):
            anchor = '    args=p.parse_args()'
            new = '    add_power_cli_arguments(p)\n    args=p.parse_args()'
        elif path.endswith('MBPO_workspace/workflows/03_evaluate.py'):
            anchor = '    args=p.parse_args()'
            new = '    add_power_cli_arguments(p)\n    args=p.parse_args()'
        else:
            raise RuntimeError(f'Unexpected CLI file: {path}')
        s=replace(s, anchor, new, f'{path} arguments')
        if path.endswith('SAC_workspace/workflows/04_evaluate.py'):
            s=replace(s, '    cfg=load_config()', '    cfg=load_config(**power_cli_kwargs(args))', 'SAC evaluate config')
            s=replace(s, '    paths=evaluate(cfg,args.checkpoint,scenario_ids=scenario_ids,seeds=args.seeds,device=args.device,include_classical=False)',
              "    # Enforce the full physical MDP identity also for SAC-only evaluation.\n"
              "    from SAC_workspace.src.evaluation.evaluation import load_agent\n"
              "    from SAC_workspace.src.environment.environment import SACEnvironment as _PhysicsEnv\n"
              "    from MBPO_workspace.src.evaluation import _check_physics\n"
              "    _check_env = _PhysicsEnv(cfg.eval_env)\n"
              "    try:\n"
              "        _check_agent = load_agent(cfg,args.checkpoint,device=args.device)\n"
              "        _check_physics(_check_agent.checkpoint_resolved_mdp,_check_env.resolved_mdp)\n"
              "    finally: _check_env.close()\n"
              "    paths=evaluate(cfg,args.checkpoint,scenario_ids=scenario_ids,seeds=args.seeds,device=args.device,include_classical=False)",
              'SAC-only evaluation explicit physics guard')
            s=replace(s, "    Path(args.outdir).mkdir(parents=True,exist_ok=True)",
              "    Path(args.outdir).mkdir(parents=True,exist_ok=True)\n"
              "    (Path(args.outdir)/'power_physics.json').write_text(\n"
              "        json.dumps(ref_env.resolved_mdp,indent=2,sort_keys=True),encoding='utf-8')",
              'SAC evaluation physical-model provenance')
        if path.endswith('MBPO_workspace/workflows/02_train.py'):
            s=replace(s, '    c=load_config(model_overrides=overrides)',
              '    c=load_config(model_overrides=overrides, **power_cli_kwargs(args))', 'MBPO train config')
        if path.endswith('MBPO_workspace/workflows/03_evaluate.py'):
            s=replace(s, '    cfg=load_config()', '    cfg=load_config(**power_cli_kwargs(args))', 'MBPO evaluate config')
    return s


PATCHES = {
    'mbpo/env/pv_tracking.py': patch_env,
    'SAC_workspace/src/config/config.py': patch_sac_config,
    'SAC_workspace/src/environment/environment.py': patch_adapter,
    'MBPO_workspace/src/config.py': patch_mbpo_config,
    'MBPO_workspace/src/evaluation.py': patch_mbpo_evaluation,
    'MBPO_workspace/src/advanced_suite.py': patch_advanced,
    **{p: (lambda s, path=p: patch_workflow(s,path)) for p in TARGETS if '/workflows/' in p},
}


def check_files() -> tuple[dict[str, bytes],dict[str, bytes]]:
    # Reject incomplete, edited, or accidentally mixed Stage-2 extraction.
    # The manifest is regenerated with the ZIP; it is intentionally not trusted
    # as a cryptographic signature, but it catches installation mistakes.
    manifest_file=ROOT/'STAGE2_MANIFEST.json'
    if not manifest_file.is_file():
        raise RuntimeError('STAGE2_MANIFEST.json missing from extracted package')
    manifest=json.loads(manifest_file.read_text(encoding='utf-8'))
    if manifest.get('schema')!='pvwatts-stage2-safe-v4':
        raise RuntimeError('Unrecognized Stage-2 package schema')
    for name, expected in manifest.get('files',{}).items():
        if (not isinstance(name,str) or not isinstance(expected,str)
                or Path(name).is_absolute() or '..' in Path(name).parts):
            raise RuntimeError('Invalid Stage-2 package manifest entry')
        file=ROOT/name
        if not file.is_file() or sha(file.read_bytes())!=expected:
            raise RuntimeError(f'Stage-2 package file missing or modified: {name}')
    if len(manifest.get('files',{}))<11:
        raise RuntimeError('Stage-2 package manifest is incomplete')
    if set(PATCHES) != set(TARGETS):
        raise RuntimeError('Developer target manifest does not cover all target files')
    for required in ('mbpo/env/pv_power_backend.py', 'mbpo/env/pv_power_cli.py',
                     'tests/test_pv_power_backend.py', 'scripts/pvwatts_stage2_audit.py',
                     'MBPO_workspace/src/power_references.py'):
        if not (ROOT / required).is_file():
            raise FileNotFoundError(f'Required extracted Stage-2 package file missing: {required}')
    for addon, expected in REQUIRED_ADDON_SHA256.items():
        actual=sha((ROOT/addon).read_bytes())
        if actual != expected:
            raise RuntimeError(f'Additive Stage-2 dependency was modified or mismatched: {addon} '
                               f'(expected {expected[:12]}, got {actual[:12]}). '
                               'Re-extract the verified ZIP; existing local edits must be reconciled.')
    originals, edited = {}, {}
    incompatibilities = []
    for target in TARGETS:
        path = ROOT / target
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f'Target missing or symlinked: {target}')
        b = path.read_bytes()
        if b'\r\n' in b:
            raise RuntimeError(f'Refusing to rewrite CRLF-encoded source: {target}')
        source = b.decode('utf-8')
        try:
            compile(source, str(path), 'exec')
        except SyntaxError as exc:
            raise RuntimeError(f'Original source is already invalid: {target}: {exc}') from exc
        if 'PVWATTS_STAGE2_INTEGRATED' in source:
            raise RuntimeError(f'{target} is already Stage-2-integrated; do not apply twice')
        try:
            modified = PATCHES[target](source)
        except (RuntimeError, SyntaxError, ValueError) as exc:
            incompatibilities.append(f'{target}: {exc}')
            continue
        if modified == source:
            raise RuntimeError(f'No modifications generated for {target}')
        modified = '# PVWATTS_STAGE2_INTEGRATED (guard; installer is not idempotent)\n' + modified
        try:
            compile(modified, str(path), 'exec')
        except SyntaxError as exc:
            incompatibilities.append(f'{target}: generated source does not compile: {exc}')
            continue
        originals[target] = b
        edited[target] = modified.encode('utf-8')
    if incompatibilities:
        raise RuntimeError('Source incompatibilities (no changes made):\n  - ' +
                           '\n  - '.join(incompatibilities))
    return originals, edited


def save_backup(originals: dict[str,bytes], edited: dict[str,bytes]) -> Path:
    now = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest = ROOT / f'.pvwatts_stage2_backup_{now}.zip'
    if dest.exists():
        raise FileExistsError(f'Backup file already exists: {dest}')
    manifest = {'schema':1,'created_utc':now,'files':{
        p:{'original_sha256':sha(originals[p]),'installed_sha256':sha(edited[p])}
        for p in TARGETS}}
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json',json.dumps(manifest,indent=2,sort_keys=True))
        for p, content in originals.items():
            z.writestr('originals/'+p,content)
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
    return dest


def atomic_write(path: Path, data: bytes):
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.pvwatts-stage2-',delete=False) as h:
        temp=Path(h.name);h.write(data);h.flush();os.fsync(h.fileno())
    try:
        os.chmod(temp,path.stat().st_mode)
        os.replace(temp,path)
    finally:
        if temp.exists(): temp.unlink()


def apply(originals: dict[str,bytes], edited: dict[str,bytes]) -> Path:
    backup = save_backup(originals,edited)
    done=[]
    try:
        for name in TARGETS:
            path=ROOT/name
            if path.read_bytes()!=originals[name]:
                raise RuntimeError(f'Target changed since preflight: {name}')
            atomic_write(path,edited[name])
            done.append(name)
    except BaseException:
        for name in reversed(done):atomic_write(ROOT/name,originals[name])
        raise
    return backup


def restore(path: Path):
    with zipfile.ZipFile(path) as z:
        if z.testzip():raise RuntimeError('Backup archive corrupted')
        manifest=json.loads(z.read('manifest.json'))
        # Check EVERY file before performing any destructive restoration.
        originals={}
        for name, meta in manifest['files'].items():
            candidate=ROOT/name
            if name not in TARGETS or not candidate.is_file():
                raise RuntimeError(f'Unrecognized backup source: {name}')
            current=candidate.read_bytes()
            if sha(current)!=meta['installed_sha256']:
                raise RuntimeError(f'{name} has changed since Stage 2 installation. '
                                   'Refusing to overwrite user edits.')
            originals[name]=z.read('originals/'+name)
            if sha(originals[name])!=meta['original_sha256']:
                raise RuntimeError('Backup source checksum mismatch: '+name)
        for name,b in originals.items():atomic_write(ROOT/name,b)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true',help='Fail-closed dry-run (default)')
    mode.add_argument('--apply', action='store_true',help='Backup and atomically apply all changes')
    mode.add_argument('--restore-from',metavar='ZIP',help='Restore exact originals when no edits followed')
    parser.add_argument('--diff',action='store_true',help='Show a full unified source diff during --check')
    args=parser.parse_args()
    if args.restore_from:
        restore(Path(args.restore_from).resolve())
        print('Restored all original target files. Newly extracted Stage-1/2 additive files are retained.')
        return
    old,new=check_files()
    print(f'Stage 2 source preflight: {len(TARGETS)} targets recognized and all candidate files compile.')
    for name in TARGETS:
        print(f'  {name}: {sha(old[name])[:12]} -> {sha(new[name])[:12]}')
    if args.diff:
        for name in TARGETS:
            old_lines=old[name].decode('utf-8').splitlines(keepends=True)
            new_lines=new[name].decode('utf-8').splitlines(keepends=True)
            sys.stdout.writelines(difflib.unified_diff(
                old_lines,new_lines,fromfile='before/'+name,tofile='after/'+name))
    if args.apply:
        backup=apply(old,new)
        print(f'APPLIED: {len(TARGETS)} files; originals backed up to {backup}')
        print('NEXT: python -m pytest -q tests/test_pv_power_backend.py tests/test_pvwatts_stage2.py')
        print('NEXT: python scripts/pvwatts_stage2_audit.py')
    else:
        print('CHECK PASS (dry-run): no existing files were modified. Re-run with --apply.')


if __name__ == '__main__':
    try:main()
    except (ValueError,RuntimeError,OSError,SyntaxError) as exc:
        print('STAGE 2 ABORTED, NO INTENTIONAL PARTIAL EDITS:',exc,file=sys.stderr)
        sys.exit(2)
