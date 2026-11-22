"""Paired NSRDB evaluation against dual-axis sun pointing.

Both learned SAC and rate-limited dual-axis tracker use PVTrackingEnv.step,
with exactly the same action limits, energy accounting, day, seed and pose.
Ideal instantaneous dual-axis is a separate, non-actuator-limited reference.
It uses the SAME power/PVLIB transposition model (not an unrelated ModelChain).
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from SAC_workspace.src.config.config import SACConfig
from SAC_workspace.src.config import config as reference_config
from SAC_workspace.src.environment.environment import SACEnvironment
from SAC_workspace.src.agents.sac_agent import SACAgent

METHODS = ('sac', 'dual_axis_slew_limited', 'fixed_no_motion', 'dual_axis_instant_ideal')


def reference_scenario_ids(cfg: SACConfig) -> list[str]:
    """Evaluation-only scenario selection, independent of the training sampler.

    conf_compact intentionally sets fixed_eval_scenarios=None so its ordinary
    evaluation uses random scenarios. A reproducible paired benchmark instead
    defaults to the original reference's named validation scenarios.
    Read the reference source without importing the Ray/TensorFlow stack.
    """
    scenarios = cfg.eval_env.get('fixed_eval_scenarios')
    if scenarios is None or len(scenarios) == 0:
        scenarios = reference_config._constant(
            reference_config._BASE, 'NSRDB_VALIDATION_SCENARIO_IDS')
    if not isinstance(scenarios, (list, tuple)) or not scenarios:
        raise ValueError('Reference validation scenarios must be a nonempty sequence')
    if len(set(scenarios)) != len(scenarios):
        raise ValueError('Duplicate reference evaluation scenarios')
    return list(scenarios)


def resolve_scenario_ids(cfg: SACConfig) -> list[str]:
    """Backwards-compatible name used by the original SAC evaluation tests."""
    return reference_scenario_ids(cfg)


def _controller_action(env, method, agent):
    if method == 'sac':
        return agent.act(env.state, deterministic=True)
    if method == 'fixed_no_motion':
        return np.zeros(2, dtype=np.float32)
    if method != 'dual_axis_slew_limited':
        raise ValueError(f'Unknown env-step controller {method}')
    from mbpo.env.pvlib_physics import dual_axis_tracking_targets
    u = env.unwrapped
    # The environment awards energy at POST-action time; use the corresponding
    # next timestamp for astronomical sun pointing (privileged baseline input).
    next_time = u.times[min(u.step_index + 1, len(u.times)-1)]
    sp = u._solar_position(next_time)
    target_tilt, target_az = dual_axis_tracking_targets(sp.zenith, sp.azimuth)
    delta_tilt = np.clip((target_tilt - u.tilt) / u.max_delta_tilt, -1., 1.)
    shortest_az = (target_az - u.azimuth + 180.) % 360. - 180.
    delta_az = np.clip(shortest_az / u.max_delta_azimuth, -1., 1.)
    return np.array([delta_tilt, delta_az], dtype=np.float32)


def run_episode(cfg: SACConfig, method: str, scenario_id: str, seed: int, agent=None, env=None):
    if method not in METHODS[:3]:
        raise ValueError(f'Invalid policy method {method}')
    if method == 'sac' and agent is None:
        raise ValueError('SAC evaluation requires loaded agent')
    owns_env = env is None
    if owns_env:
        env = SACEnvironment(cfg.eval_env)
    env.seed(int(seed))
    path = {'method': method, 'scenario_id': scenario_id, 'seed': int(seed),
            'reward': [], 'power': [], 'energy': [], 'cost': [], 'tilt': [],
            'azimuth': [], 'action_tilt': [], 'action_azimuth': [],
            'delta_tilt_deg': [], 'delta_azimuth_deg': [], 'infos': []}
    try:
        env.state = env.reset(scenario_id=scenario_id)
        path['initial_pose'] = [float(env.unwrapped.tilt), float(env.unwrapped.azimuth)]
        for t in range(env.horizon):
            old_tilt, old_az = float(env.unwrapped.tilt), float(env.unwrapped.azimuth)
            action = _controller_action(env, method, agent)
            env.state, reward, done, info = env.step(action)
            path['reward'].append(float(reward))
            for dest, src in (('power','power'),('energy','energy_kwh'),('cost','movement_cost'),('tilt','tilt'),('azimuth','azimuth')):
                path[dest].append(float(info[src]))
            path['delta_tilt_deg'].append(float(env.unwrapped.tilt) - old_tilt)
            path['delta_azimuth_deg'].append((float(env.unwrapped.azimuth)-old_az+180.)%360.-180.)
            path['action_tilt'].append(float(action[0]))
            path['action_azimuth'].append(float(action[1]))
            # Keep date/season metadata present for both old and new plotters.
            # The manifest ID is the canonical NSRDB episode date.
            if not info.get('date'):
                info = dict(info, date=str(scenario_id))
            if not info.get('season'):
                from SAC_workspace.src.evaluation.stratified_statistics import SEASONS
                info = dict(info, season=SEASONS[int(str(scenario_id)[5:7])])
            path['infos'].append({k: info.get(k) for k in ('scenario_id','date','season','weather_source','weather_resampling','control_interval_minutes','solar_zenith_deg','solar_azimuth_deg',
                       'timestamp_utc_iso','timezone','cos_aoi','rollout_seed',
                       'weather_condition','episode_dni_fraction','episode_diffuse_fraction',
                       'dni_wm2','ghi_wm2','dhi_wm2','solar_altitude_deg',
                       'poa_global','power_dc_w','effective_irradiance_wm2',
                       'temperature_c','cell_temperature_c','ambient_temperature_c',
                       'wind_speed_m_s','wind_speed_10m_ms','power_model',
                       'wind_reference_height_m','gamma_pdc','sapm_mount',
                       'wind_powerlaw_exponent','surface_albedo','albedo_source','physics_revision')})
            if done != (t == env.horizon-1):
                raise RuntimeError('Unexpected terminal position')
    finally:
        if owns_env:
            env.close()
    path['net_energy_kwh'] = float(sum(path['reward']))
    path['gross_energy_kwh'] = float(sum(path['energy']))
    path['movement_cost_kwh'] = float(sum(path['cost']))
    if not np.isclose(path['net_energy_kwh'], path['gross_energy_kwh'] - path['movement_cost_kwh'], atol=1e-8):
        raise RuntimeError('Energy/reward mismatch')
    return path


def ideal_instantaneous_dual_axis(cfg: SACConfig, fixed_path, env=None):
    """Compute the gross-only, unconstrained sun-pointing reference offline.

    Reuse the *recorded post-action* weather and solar position from the paired
    fixed-mount trajectory. Never re-step/reseed the physical environment:
    that would be slower, needlessly repeat weather loading, and can introduce
    mismatched conditions in a counterfactual benchmark.
    """
    from mbpo.env.pvlib_physics import dual_axis_tracking_targets
    from mbpo.env.pv_power_backend import compute_pv_power
    owns_env = env is None
    if owns_env:
        if cfg is None:
            raise ValueError('cfg is required when no reference environment is passed')
        env = SACEnvironment(cfg.eval_env)
    try:
        physical = getattr(env, 'unwrapped', env)
        area = float(physical.area)
        efficiency = float(physical.efficiency)
        # Real PVTrackingEnv exposes interval_hours. Test doubles / wrappers
        # may omit it; infer it from the recorded control interval, not by
        # resetting/re-stepping the simulator. The reference grid is 5 min.
        native_interval = getattr(physical, 'interval_hours', None)
        if native_interval is None:
            minute = fixed_path['infos'][0].get('control_interval_minutes') if fixed_path.get('infos') else None
            if minute is None:
                if cfg is not None and str(cfg.eval_env.get('freq', '5min')) != '5min':
                    raise ValueError('Missing interval_hours for a non-five-minute environment')
                minute = 5
            interval_hours = float(minute) / 60.0
        else:
            interval_hours = float(native_interval)
        if not (np.isfinite(interval_hours) and interval_hours > 0):
            raise ValueError('Invalid physical control interval')
        infos = fixed_path['infos']
        n = len(infos)
        if n == 0 or n != len(fixed_path['energy']):
            raise ValueError('Fixed reference has missing or inconsistent weather trajectory')
        horizon = getattr(env, 'horizon', n)
        if horizon != n:
            raise ValueError('Fixed reference does not match the evaluation horizon')
        result = {'method': 'dual_axis_instant_ideal',
                  'scenario_id': fixed_path['scenario_id'], 'seed': fixed_path['seed'],
                  'initial_pose': list(fixed_path['initial_pose']),
                  'reward': [], 'energy': [], 'power': [], 'cost': [],
                  'tilt': [], 'azimuth': [], 'action_tilt': [],
                  'action_azimuth': [], 'delta_tilt_deg': [],
                  'delta_azimuth_deg': [], 'infos': []}
        required = ('solar_zenith_deg', 'solar_azimuth_deg',
                    'dni_wm2', 'ghi_wm2', 'dhi_wm2')
        for info in infos:
            missing = [k for k in required if info.get(k) is None]
            if missing:
                raise ValueError('Fixed reference missing PV weather fields: ' + ', '.join(missing))
            tilt, az = dual_axis_tracking_targets(
                float(info['solar_zenith_deg']), float(info['solar_azimuth_deg']))
            if (getattr(physical, 'weather_source', None) == 'nsrdb_multiyear'
                    and info.get('surface_albedo') is None):
                raise ValueError('NSRDB dual-axis reference is missing recorded surface_albedo')
            physical_result = compute_pv_power(
                tilt, az, float(info['solar_zenith_deg']),
                float(info['solar_azimuth_deg']), float(info['dni_wm2']),
                float(info['ghi_wm2']), float(info['dhi_wm2']),
                area=area, efficiency=efficiency,
                power_model=getattr(physical, 'power_model', 'legacy'),
                temp_air_c=info.get('temperature_c',info.get('ambient_temperature_c')),
                wind_speed_ms=info.get('wind_speed_m_s'),
                wind_reference_height_m=getattr(physical,'wind_reference_height_m',None),
                gamma_pdc=getattr(physical,'gamma_pdc',-0.004),
                sapm_mount=getattr(physical,'sapm_mount','open_rack_glass_polymer'),
                wind_powerlaw_exponent=getattr(physical,'wind_powerlaw_exponent',0.14),
                surface_albedo=info.get('surface_albedo'))
            power = float(physical_result.power_w)
            energy = power * interval_hours / 1000.0
            if not np.isfinite(power) or power < 0 or not np.isfinite(energy):
                raise ValueError('Nonphysical power from dual-axis reference')
            result['power'].append(power)
            result['energy'].append(energy)
            result['reward'].append(energy)   # plotting compatibility only
            result['cost'].append(0.0)        # bookkeeping only; no physical motor cost claim
            result['tilt'].append(float(tilt))
            result['azimuth'].append(float(az))
            for key in ('action_tilt','action_azimuth','delta_tilt_deg','delta_azimuth_deg'):
                result[key].append(float('nan'))
            metadata=dict(info)
            metadata.update(power=power,power_dc_w=power,
                poa_global=float(physical_result.poa_global_wm2),
                effective_irradiance_wm2=float(physical_result.effective_irradiance_wm2),
                cell_temperature_c=physical_result.cell_temperature_c,
                wind_speed_10m_ms=physical_result.wind_speed_10m_ms,
                power_model=getattr(physical,'power_model','legacy'),
                tilt=float(tilt),azimuth=float(az))
            result['infos'].append(metadata)
        result['gross_energy_kwh'] = float(sum(result['energy']))
        result['net_energy_kwh'] = None
        result['movement_cost_kwh'] = None
        return result
    finally:
        if owns_env:
            env.close()


def load_agent(cfg: SACConfig, checkpoint: str | Path, device='cpu') -> SACAgent:
    # Load only checkpoint files created locally by this training package.
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if data.get('schema') != 1:
        raise ValueError('Unsupported SAC checkpoint schema')
    from SAC_workspace.src.training.trainer import config_hash
    if data.get('hash') != config_hash(cfg):
        raise ValueError('SAC checkpoint configuration does not match evaluation environment')
    state = data['agent']
    agent = SACAgent(hidden_sizes=cfg.algo['hidden_sizes'], lr=cfg.algo['lr'],
                     gamma=cfg.algo['gamma'], tau=cfg.algo['tau'], reward_scale=cfg.algo['reward_scale'],
                     alpha_init=cfg.algo['alpha_init'], target_entropy=cfg.algo['target_entropy'], min_alpha=cfg.algo['min_alpha'], device=device)
    agent.load_state_dict(state)
    agent.actor.eval()
    agent.checkpoint_resolved_mdp = data.get('resolved_mdp')
    return agent


def evaluate(cfg: SACConfig, checkpoint: str | Path, scenario_ids=None, seeds=(42,), device='cpu', include_classical=True):
    if len(tuple(seeds)) > 1 and not cfg.eval_env.get('randomize_initial_orientation', False):
        raise ValueError('Reference evaluation fixes the initial pose; multiple seeds would only duplicate the same episode. Use one seed.')
    agent = load_agent(cfg, checkpoint, device=device)
    from mbpo.env.nsrdb_weather import load_scenario_manifest
    manifest = load_scenario_manifest(cfg.manifest)
    known = {row['scenario_id'] for row in manifest['scenarios']}
    if scenario_ids is None:
        scenario_ids = reference_scenario_ids(cfg)
    if not scenario_ids:
        raise ValueError('No evaluation scenarios specified')
    if len(scenario_ids) != len(set(scenario_ids)) or len(seeds) != len(set(seeds)):
        raise ValueError('Duplicate scenario IDs or evaluation seeds are not allowed')
    if set(scenario_ids) - known:
        raise ValueError(f'Unknown scenario IDs: {set(scenario_ids)-known}')
    # Instantiate once: avoid rebuilding the NSRDB cache for every policy/day.
    # Every controller reseeds, resets to the SAME explicit scenario, and
    # receives the same reference initial panel pose.
    paths = []
    env = SACEnvironment(cfg.eval_env)
    try:
        # Evaluation-specific randomness/initial-pose switches legitimately
        # differ from training. Physical model, energy units and actuator limits
        # must not change between the policy checkpoint and the evaluator.
        saved_mdp = agent.checkpoint_resolved_mdp
        if not saved_mdp:
            raise ValueError('Checkpoint lacks effective PV tracking model metadata')
        if saved_mdp.get('power_model','legacy') != env.resolved_mdp.get('power_model','legacy'):
            raise ValueError('SAC evaluation physical power model mismatch')
        if env.resolved_mdp.get('power_model') == 'pvwatts_sapm':
            for key in ('wind_reference_height_m','gamma_pdc','sapm_mount',
                        'wind_powerlaw_exponent','power_source_sha256',
                        'albedo_model','physics_revision',
                        'nsrdb_manifest_sha256','nsrdb_year_file_sha256',
                        'nsrdb_albedo_contract'):
                if saved_mdp.get(key) != env.resolved_mdp.get(key):
                    raise ValueError(f'SAC evaluator power parameter mismatch: {key}')
        physical_keys = ('pvlib_version','gym_version','site_latitude','site_longitude','site_altitude','tz',
                         'start_time','periods','freq','weather_source','area',
                         'efficiency','max_delta_tilt','max_delta_azimuth',
                         'movement_penalty','movement_cost_mode','actuator_power_w',
                         'slew_rate_tilt_deg_s','slew_rate_azimuth_deg_s',
                         'observation_mode','horizon')
        for key in physical_keys:
            if saved_mdp.get(key) != env.resolved_mdp.get(key):
                raise ValueError(f'Checkpoint vs evaluation PV simulator mismatch: {key}')
        for sid in scenario_ids:
            for seed in seeds:
                fixed = run_episode(cfg, 'fixed_no_motion', sid, int(seed), env=env)
                sac = run_episode(cfg, 'sac', sid, int(seed), agent, env=env)
                classic = []
                if include_classical:
                    classic = [run_episode(cfg, 'dual_axis_slew_limited', sid, int(seed), env=env),
                               ideal_instantaneous_dual_axis(cfg, fixed, env=env)]
                from mbpo.env.baseline_contract import validate_paired_reference_group
                validate_paired_reference_group(
                    {p['method']:p for p in [sac, fixed]+classic})
                paths.extend([sac, fixed] + classic)
                for p in [sac] + classic:
                    if len(p['energy']) != len(fixed['energy']) or p['scenario_id'] != fixed['scenario_id']:
                        raise AssertionError('Unpaired evaluation episode')
                    if ([i['timestamp_utc_iso'] for i in p['infos']]
                            != [i['timestamp_utc_iso'] for i in fixed['infos']]):
                        raise AssertionError('UTC timestamp grid mismatch across compared methods')
                    if not np.allclose(p['initial_pose'], fixed['initial_pose'], atol=1e-8):
                        raise AssertionError('Different starting panel pose for matched evaluations')
                    if len(p['energy']) != env.horizon:
                        raise AssertionError('Benchmark horizon mismatch')
    finally:
        env.close()
    return paths


def _ci_clustered(differences_by_scenario, seed=123, n_boot=2000):
    """Scenario-cluster bootstrap, resampling days (not correlated replicate rows)."""
    groups = np.array([np.mean(v) for v in differences_by_scenario.values()], dtype=np.float64)
    if not len(groups): return (float('nan'), float('nan'), float('nan'))
    mean = float(groups.mean())
    if len(groups) < 2: return mean, float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    draws = rng.choice(groups, size=(n_boot, len(groups)), replace=True).mean(axis=1)
    return mean, float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def summary(paths):
    bymethod, paired = defaultdict(list), defaultdict(dict)
    for p in paths:
        bymethod[p['method']].append(p)
        key = (p['scenario_id'], p['seed'])
        if p['method'] in paired[key]:
            raise ValueError(f'Duplicate {p["method"]} result for scenario {key}')
        paired[key][p['method']] = p
    for (sid, seed), methods in paired.items():
        if set(methods) != set(METHODS):
            raise ValueError(f'Incomplete or duplicated benchmark set for {sid} seed={seed}: {set(methods)}')
    horizon = len(next(iter(paths))['infos']) if paths else 0
    if horizon <= 0 or any(len(p.get('infos', [])) != horizon for p in paths):
        raise ValueError('SAC summary requires complete, equal-length trajectories')
    out = {'protocol': f'same NSRDB scenario, seed, pose, {horizon} post-step UTC timestamps',
           'scenario_count': len({p['scenario_id'] for p in paths}),
           'replicate_pairs': len(paired), 'methods': {}, 'paired_vs_sac': {}}
    for method, records in bymethod.items():
        vals = [p['net_energy_kwh'] for p in records if p['net_energy_kwh'] is not None]
        out['methods'][method] = {'n': len(vals), 'mean_net_kwh': float(np.mean(vals)) if vals else None,
            'std_net_kwh': float(np.std(vals, ddof=1)) if len(vals)>1 else None,
            'mean_gross_kwh': float(np.mean([p['gross_energy_kwh'] for p in records])),
            'mean_movement_cost_kwh': float(np.mean([p['movement_cost_kwh'] for p in records
                                                if p['movement_cost_kwh'] is not None]))
                                                if any(p['movement_cost_kwh'] is not None for p in records) else None,
            'mean_absolute_tilt_delta_deg': float(np.mean([np.mean(np.abs(p['delta_tilt_deg']))
                for p in records])) if all('delta_tilt_deg' in p for p in records) and method != 'dual_axis_instant_ideal' else None,
            'mean_absolute_azimuth_delta_deg': float(np.mean([np.mean(np.abs(p['delta_azimuth_deg']))
                for p in records])) if all('delta_azimuth_deg' in p for p in records) and method != 'dual_axis_instant_ideal' else None}
    for method in METHODS[1:]:
        group = defaultdict(list)
        for (sid, _), methods in paired.items():
            if methods['sac']['net_energy_kwh'] is not None and methods[method]['net_energy_kwh'] is not None:
                group[sid].append(methods['sac']['net_energy_kwh']-methods[method]['net_energy_kwh'])
        mean, lo, hi = _ci_clustered(group)
        per_scenario = []
        for (sid, _), methods in paired.items():
            a = methods['sac']['net_energy_kwh']
            b = methods[method]['net_energy_kwh']
            if a is not None and b is not None and b > 1e-12:
                per_scenario.append(100.0*(a-b)/b)
        out['paired_vs_sac'][method] = {
            'sac_minus_reference_net_kwh': mean,
            'bootstrap_95pct_ci': [lo, hi],
            'mean_per_scenario_net_gain_pct': float(np.mean(per_scenario)) if per_scenario else None,
            'valid_relative_gain_scenarios': len(per_scenario)}
    return out


def write_outputs(paths, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stats = summary(paths)
    def finite_json(obj):
        if isinstance(obj, float) and not np.isfinite(obj): return None
        if isinstance(obj, dict): return {k: finite_json(v) for k,v in obj.items()}
        if isinstance(obj, list): return [finite_json(v) for v in obj]
        return obj
    (outdir/'summary.json').write_text(json.dumps(finite_json(stats), indent=2, allow_nan=False), encoding='utf-8')
    fields = ['scenario_id','seed','method','net_energy_kwh','gross_energy_kwh','movement_cost_kwh']
    with (outdir/'episodes.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in paths: w.writerow({k:p[k] for k in fields})
    trajectory_fields = ['scenario_id','seed','method','step','timestamp_utc_iso','utc_hour','power_w','energy_kwh',
                         'reward_kwh','movement_cost_kwh','tilt_deg','azimuth_deg',
                         'normalized_action_tilt','normalized_action_azimuth','actual_delta_tilt_deg','actual_delta_azimuth_deg']
    with (outdir/'trajectories.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=trajectory_fields)
        writer.writeheader()
        for p in paths:
            for i in range(len(p['power'])):
                writer.writerow(dict(zip(trajectory_fields, [p['scenario_id'],p['seed'],p['method'],i+1,
                  p['infos'][i]['timestamp_utc_iso'], float(p['infos'][i]['timestamp_utc_iso'][11:13]) + float(p['infos'][i]['timestamp_utc_iso'][14:16]) / 60., p['power'][i],p['energy'][i],p['reward'][i],p['cost'][i],
                  p['tilt'][i],p['azimuth'][i],p['action_tilt'][i],p['action_azimuth'][i],
                  p['delta_tilt_deg'][i],p['delta_azimuth_deg'][i]])))
    return stats
