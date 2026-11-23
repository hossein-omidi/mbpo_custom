#!/usr/bin/env python3
"""PostProcessing Monte Carlo v2 uncertainty integration, no backups or Git writes.

Applies to the exact reviewed PostProcessing source or upgrades the exact v1
Monte Carlo new modules. Unknown local source changes are not overwritten.
"""
from __future__ import annotations
import argparse, ast, difflib, hashlib, os, tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
NEW=HERE/'new_files'
MARKER='# POSTPROCESSING_TRAJECTORY_UNCERTAINTY_V1'
EXPECTED={
 'mbpo/env/stage_a_pv_tracking.py':'e0e96b3b883a1057ecef50a41d75a0fdc73697f4',
 'SAC_workspace/src/environment/environment.py':'053009fe0bc9122e35a52f85d0b1217232a71e4a',
 'SAC_workspace/src/config/config.py':'8786e8764ad04dc4015cd449f1b0297d7932ed9f',
 'MBPO_workspace/src/config.py':'532646d169f2e640435c5c9d9ba2c226eb7906a2',
 'SAC_workspace/src/training/experiment_protocol.py':'33c72302f99453f92149a99c6dec898d98df000a',
 'SAC_workspace/src/training/trainer.py':'77188bc694ffee8d68b76afbbcf66e64109895e4',
 'MBPO_workspace/src/trainer.py':'e22305dbc81e49b2fc6309e148076c781a5e186e',
 'MBPO_workspace/workflows/06_model_validation.py':'85aabf4a79d7e62570858fe034fe3cd200a745a3',
}

# Exact original v1 new-module fingerprints. Permit a guarded update from the
# previous Monte Carlo ZIP without making users revert unrelated modifications.
PREVIOUS_V1_NEW={
 'mbpo/env/trajectory_uncertainty.py':'b0f11f50734e8bf3a94df2305336e5ecc1c0c3d7',
 'MBPO_workspace/workflows/09_monte_carlo_evaluate.py':'20208e3967a112c75a02b6adefb65f61a39633e1',
 'MBPO_workspace/workflows/08_monte_carlo_train.py':'70cdd8d54aff363646f8d3913bbcfcd94ee108d4',
}
PATCHED_REQUIRED={
 'mbpo/env/stage_a_pv_tracking.py':('def set_weather_realization(', 'generate_weather_profile(', 'weather_profile_sha256'),
 'SAC_workspace/src/environment/environment.py':('weather_uncertainty', 'set_weather_realization(', 'trajectory_uncertainty'),
 'SAC_workspace/src/config/config.py':('weather_uncertainty=None', "eval_env['weather_uncertainty']"),
 'MBPO_workspace/src/config.py':('weather_uncertainty=None', 'weather_uncertainty=weather_uncertainty'),
 'SAC_workspace/src/training/experiment_protocol.py':('realization_id=0',),
 'SAC_workspace/src/training/trainer.py':("'weather_realization_seed'",),
 'MBPO_workspace/src/trainer.py':("'weather_realization_seed'",),
 'MBPO_workspace/workflows/06_model_validation.py':('validation_realization_id', 'realization_id=args.realization_id', 'weather_uncertainty=uncertainty_spec'),
}

def blob(data):return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()

def change(s,old,new,where):
    n=s.count(old)
    if n!=1:
        raise ValueError(f'{where}: anchor occurrence {n} instead of 1: {old[:90]!r}')
    return s.replace(old,new,1)

def patch_stage(s):
    p='stage_a_pv_tracking.py'
    s=change(s,"from mbpo.env.pv_power_backend import compute_pv_power, PHYSICS_REVISION, DEFAULT_ALBEDO\n",
    "from mbpo.env.pv_power_backend import compute_pv_power, PHYSICS_REVISION, DEFAULT_ALBEDO\n"
    "from mbpo.env.trajectory_uncertainty import (normalized_spec, realization_seed,\n"
    "    spec_fingerprint, generate_weather_profile)\nimport hashlib\nimport json\n",p)
    s=change(s,
        'canonical isotropic POA and PVWatts/SAPM backend. No on-line API, synthetic\n'
        'weather, or additional observation/action channels are introduced.',
        'canonical isotropic POA and PVWatts/SAPM backend. Optional hypothesized\n'
        'weather-error realizations are generated once per episode at reset; the\n'
        'six-feature controller interface and physical power backend are unchanged.',p)
    s=change(s,"    def __init__(self,*args,**kwargs):\n        if (kwargs.get('weather_source')=='nsrdb_multiyear'",
    """    def __init__(self,*args,**kwargs):
        self.weather_uncertainty = normalized_spec(kwargs.pop('weather_uncertainty',None))
        self._requested_realization_id = None
        self._weather_realization_seed = None
        self._weather_realization_id = None
        self._weather_profile_sha256 = None
        if self.weather_uncertainty['enabled'] and not (
                kwargs.get('weather_source')=='nsrdb_multiyear' or
                kwargs.get('weather_scenario_mode')=='nsrdb_multiyear'):
            raise ValueError('Trajectory uncertainty requires historical NSRDB source')
        if (kwargs.get('weather_source')=='nsrdb_multiyear'""",p)
    new="""    def set_weather_realization(self, realization_id=None):
        # Explicit identity for common-random-numbers evaluation. Training
        # passes None, drawing a new episode seed from the checkpointed RNG.
        if realization_id is not None and (
                isinstance(realization_id,bool) or
                not isinstance(realization_id,(int,np.integer)) or int(realization_id)<0):
            raise ValueError('realization_id must be a nonnegative integer or None')
        self._requested_realization_id = (None if realization_id is None else int(realization_id))

    def _apply_episode_weather_stochasticity(self):
        if not self.weather_uncertainty['enabled']:
            self._weather_realization_seed = None
            self._weather_realization_id = None
            self._weather_profile_sha256 = None
            return super()._apply_episode_weather_stochasticity()
        from mbpo.env.nsrdb_weather import scenario_id as nsrdb_scenario_id
        sid=nsrdb_scenario_id(self._current_scenario)
        identity=self._requested_realization_id
        if identity is None:
            rg=self.np_random
            seed=(int(rg.integers(0,2**63-1)) if hasattr(rg,'integers')
                  else int(rg.randint(0,2**31-1)))
        else:
            seed=realization_seed(self.weather_uncertainty,sid,identity)
        solar=self.location.get_solarposition(self.times)
        cosz=np.maximum(np.cos(np.deg2rad(np.asarray(solar['zenith'],dtype=float))),0.0)
        frame,diag=generate_weather_profile(
            self.weather_profile,cosz,self.weather_uncertainty,seed=seed)
        self.weather_profile=frame
        self._weather_realization_seed=int(seed)
        self._weather_realization_id=identity
        self._weather_uncertainty_diagnostics=diag
        # Hash entire exogenous sequence; independent of policy actions.
        fields=('dni','dhi','ghi','temperature','wind_speed','surface_albedo')
        present=[k for k in fields if k in frame]
        albedo_col=next((c for c in ALBEDO_COLUMNS if c in frame),None)
        if albedo_col is not None and albedo_col not in present:present.append(albedo_col)
        matrix=np.column_stack([np.asarray(frame[k],dtype='<f8') for k in present])
        self._weather_profile_sha256=hashlib.sha256(matrix.tobytes()).hexdigest()

"""
    s=change(s,"    def _surface_albedo_at_current_time(self):",new+"    def _surface_albedo_at_current_time(self):",p)
    s=change(s,"        info['physics_revision'] = PHYSICS_REVISION\n        return obs, reward, done, info",
    """        info['physics_revision'] = PHYSICS_REVISION
        if self.weather_uncertainty['enabled'] and self.weather_uncertainty['albedo_sd']>0:
            info['albedo_source'] = 'nsrdb_recorded_with_assumed_uncertainty'
        info['weather_condition_source'] = 'original_nsrdb_label'
        info['weather_uncertainty_enabled'] = bool(self.weather_uncertainty['enabled'])
        info['weather_realization_seed'] = self._weather_realization_seed
        info['weather_realization_id'] = self._weather_realization_id
        info['weather_profile_sha256'] = self._weather_profile_sha256
        info['weather_uncertainty_spec_sha256'] = spec_fingerprint(self.weather_uncertainty)
        info['weather_uncertainty_clip_count'] = int(getattr(self,'_weather_uncertainty_diagnostics',{}).get('clip_count',0))
        info['weather_uncertainty_baseline_closure_max_wm2'] = getattr(
            self,'_weather_uncertainty_diagnostics',{}).get('max_baseline_closure_residual_wm2')
        return obs, reward, done, info""",p)
    return MARKER+'\n'+s

def patch_adapter(s):
    p='SAC environment adapter'
    s=change(s,"            'weather_source': value('weather_source'),",
    """            'weather_source': value('weather_source'),
            'weather_uncertainty': value('weather_uncertainty'),""",p)
    s=change(s,"from mbpo.env import pv_power_backend, pvlib_physics, pv_tracking, stage_a_pv_tracking, stage_a_data_contract",
    "from mbpo.env import pv_power_backend, pvlib_physics, pv_tracking, stage_a_pv_tracking, stage_a_data_contract, trajectory_uncertainty",p)
    s=change(s,"for module in (pv_power_backend, pvlib_physics, pv_tracking, stage_a_pv_tracking, stage_a_data_contract):",
    "for module in (pv_power_backend, pvlib_physics, pv_tracking, stage_a_pv_tracking, stage_a_data_contract, trajectory_uncertainty):",p)
    s=change(s,"    def reset(self, scenario_id: str | None = None) -> np.ndarray:\n        self.steps = 0\n        observation = encode(self.env.reset(scenario_id=scenario_id))",
    """    def reset(self, scenario_id: str | None = None, realization_id: int | None = None) -> np.ndarray:
        self.steps = 0
        self.env.set_weather_realization(realization_id)
        observation = encode(self.env.reset(scenario_id=scenario_id))""",p)
    s=change(s,
        "            self.resolved_mdp['nsrdb_albedo_contract'] = 'recorded_per_5min_strict'",
        """            self.resolved_mdp['nsrdb_albedo_contract'] = (
                'recorded_with_hypothetical_uncertainty'
                if source.weather_uncertainty['enabled'] and source.weather_uncertainty['albedo_sd']>0
                else 'recorded_per_5min_strict')""",p)
    s=change(s,"                albedo_model='nsrdb_recorded_strict',",
        """                albedo_model=('nsrdb_recorded_with_hypothetical_uncertainty'
                    if source.weather_uncertainty['enabled'] and source.weather_uncertainty['albedo_sd']>0
                    else 'nsrdb_recorded_strict'),""",p)
    return MARKER+'\n'+s

def patch_sac_cfg(s):
    p='SAC config'
    s=change(s,"                wind_powerlaw_exponent=0.14) -> SACConfig:",
     "                wind_powerlaw_exponent=0.14, weather_uncertainty=None) -> SACConfig:",p)
    s=change(s,"    skeleton = _literal_mapping_fields(_SKELETON, 'params'",
    """    if weather_uncertainty is not None:
        from mbpo.env.trajectory_uncertainty import normalized_spec
        spec = normalized_spec(weather_uncertainty)
        nsrdb['weather_uncertainty'] = dict(spec)
        eval_env['weather_uncertainty'] = dict(spec)
    skeleton = _literal_mapping_fields(_SKELETON, 'params'""",p)
    return MARKER+'\n'+s

def patch_mbpo_cfg(s):
    p='MBPO config'
    s=change(s,"                sapm_mount='open_rack_glass_polymer',wind_powerlaw_exponent=0.14)->MBPOConfig:",
      "                sapm_mount='open_rack_glass_polymer',wind_powerlaw_exponent=0.14,\n                weather_uncertainty=None)->MBPOConfig:",p)
    s=change(s,"        sapm_mount=sapm_mount,wind_powerlaw_exponent=wind_powerlaw_exponent)",
      "        sapm_mount=sapm_mount,wind_powerlaw_exponent=wind_powerlaw_exponent,\n        weather_uncertainty=weather_uncertainty)",p)
    return MARKER+'\n'+s

def patch_periodic(s):
    p='shared periodic evaluator'
    s=change(s,"        obs = self.env.reset(scenario_id=scenario_id)",
    """        # A fixed common uncertainty realization maintains identical
        # reference and learned-policy weather at every checkpoint.
        obs = self.env.reset(scenario_id=scenario_id, realization_id=0)""",p)
    return MARKER+'\n'+s


def patch_sac_trainer(s):
    p='SAC trainer logging'
    s=change(s,"            'scenario_id': scenario_id, 'return_kwh': float(sum(rewards)),",
      """            'scenario_id': scenario_id,
            'weather_realization_seed': self.env.unwrapped._weather_realization_seed,
            'weather_uncertainty_enabled': bool(self.env.unwrapped.weather_uncertainty['enabled']),
            'return_kwh': float(sum(rewards)),""",p)
    return MARKER+'\n'+s


def patch_mbpo_trainer(s):
    p='MBPO trainer logging'
    s=change(s,"            'scenario_id': scenario, 'return_kwh': float(sum(episode_rewards)),",
      """            'scenario_id': scenario,
            'weather_realization_seed': self.env.unwrapped._weather_realization_seed,
            'weather_uncertainty_enabled': bool(self.env.unwrapped.weather_uncertainty['enabled']),
            'return_kwh': float(sum(episode_rewards)),""",p)
    return MARKER+'\n'+s

def patch_model_validation(s):
    p='MBPO model validation: counterfactual replay under stochastic weather'
    s=change(s, 'def _run_policy_trajectory(env, agent, scenario_id):',
       'def _run_policy_trajectory(env, agent, scenario_id, realization_id=0):',p)
    s=change(s, '    obs = env.reset(scenario_id=scenario_id)',
       '    obs = env.reset(scenario_id=scenario_id, realization_id=realization_id)',p)
    s=change(s, '    cfg = load_config(\n        model_overrides=model_overrides,',
       '''    uncertainty_spec = None
    if args.uncertainty_spec is not None:
        from mbpo.env.trajectory_uncertainty import normalized_spec
        uncertainty_spec = normalized_spec(json.loads(
            Path(args.uncertainty_spec).read_text(encoding='utf-8')))
    cfg = load_config(
        weather_uncertainty=uncertainty_spec,
        model_overrides=model_overrides,''',p)
    s=change(s, 'states, actions, next_states, rewards = _run_policy_trajectory(env, agent, sid)',
       'states, actions, next_states, rewards = _run_policy_trajectory(\n                env, agent, sid, realization_id=args.realization_id)',p)
    s=change(s, '                    replay_obs = env.reset(scenario_id=sid)',
       '                    replay_obs = env.reset(\n                        scenario_id=sid, realization_id=args.realization_id)',p)
    s=change(s, '    parser.add_argument("--states-per-scenario", type=int, default=6)',
       '''    parser.add_argument("--states-per-scenario", type=int, default=6)
    parser.add_argument("--uncertainty-spec", default=None,
                        help="Use the same uncertainty assumptions as MC-trained MBPO")
    parser.add_argument("--realization-id", type=int, default=0,
                        help="Frozen weather realization for exact counterfactual replay")''',p)
    s=change(s, '    if args.states_per_scenario < 1:',
       '    if args.realization_id < 0:\n        parser.error("realization-id must be nonnegative")\n    if args.states_per_scenario < 1:',p)
    s=change(s, '            "validation_scenario_ids": scenario_ids,',
       '''            "validation_scenario_ids": scenario_ids,
            "validation_realization_id": int(args.realization_id),
            "weather_uncertainty": uncertainty_spec,''',p)
    s=change(s, '"Counterfactual tests replay the same policy prefix to reconstruct an observed state, then execute alternative actions in the real simulator.",',
       '"Counterfactual tests replay the same policy prefix under an identical fixed weather realization, then execute alternative actions in the real simulator.",',p)
    return MARKER+'\n'+s


PATCHERS={
 'mbpo/env/stage_a_pv_tracking.py':patch_stage,
 'SAC_workspace/src/environment/environment.py':patch_adapter,
 'SAC_workspace/src/config/config.py':patch_sac_cfg,
 'MBPO_workspace/src/config.py':patch_mbpo_cfg,
 'SAC_workspace/src/training/experiment_protocol.py':patch_periodic,
 'SAC_workspace/src/training/trainer.py':patch_sac_trainer,
 'MBPO_workspace/src/trainer.py':patch_mbpo_trainer,
 'MBPO_workspace/workflows/06_model_validation.py':patch_model_validation,
}

def prepare(root):
    modifications={}
    for rel,fun in PATCHERS.items():
        target=root/rel
        if not target.is_file():raise FileNotFoundError(target)
        original=target.read_bytes();content=original.decode('utf-8')
        if content.startswith(MARKER+'\n'):
            missing=[part for part in PATCHED_REQUIRED[rel] if part not in content]
            if missing:raise ValueError(f'{rel}: incomplete v1 MC installation: {missing}')
            continue
        sha=blob(original)
        if sha!=EXPECTED[rel]:
            raise ValueError(f'{rel}: unrecognized source Git blob {sha}; expected {EXPECTED[rel]}. Refusing overwrite.')
        modified=fun(content)
        ast.parse(modified,filename=rel)
        modifications[rel]=(content,modified)
    for source in NEW.rglob('*.py'):
        rel=source.relative_to(NEW).as_posix();dest=root/rel
        new=source.read_text(encoding='utf8')
        ast.parse(new,filename=rel)
        if dest.exists():
            oldbytes=dest.read_bytes();old=oldbytes.decode('utf8')
            if old==new:continue
            if blob(oldbytes)!=PREVIOUS_V1_NEW.get(rel):
                raise ValueError(f'Existing new module differs from audited v1: {dest} (refusing overwrite)')
            modifications[rel]=(old,new)
            continue
        modifications[rel]=('',new)
    return modifications

def diff(modifications):
    parts=[]
    for rel,(old,new) in sorted(modifications.items()):
        lines=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
            fromfile='a/'+rel if old else '/dev/null',tofile='b/'+rel))
        parts.append('diff --git a/'+rel+' b/'+rel+'\n'+('new file mode 100644\n' if not old else '')+lines)
    return ''.join(parts)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--apply',action='store_true')
    ap.add_argument('--diff-file')
    args=ap.parse_args()
    if args.apply and args.dry_run:ap.error('--apply and --dry-run are exclusive')
    root=Path(args.root).expanduser().resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    planned=prepare(root)
    print('Audited local PostProcessing modifications:',len(planned))
    for p in sorted(planned):print(' MOD' if planned[p][0] else ' NEW',p)
    if args.diff_file:
        dest=Path(args.diff_file);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(diff(planned),encoding='utf-8');print('Wrote:',dest)
    if not args.apply:
        print('No repository files changed. Use --apply only after inspecting the diff.')
        return
    staged=[]
    try:
        for rel,(_,new) in sorted(planned.items()):
            target=root/rel;target.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile('w',encoding='utf8',dir=target.parent,prefix='.montecarlo_',suffix='.tmp',delete=False) as f:
                f.write(new);staged.append((Path(f.name),target))
        for src,dst in staged:os.replace(src,dst)
    finally:
        for src,_ in staged:
            if src.exists():src.unlink()
    print('Local changes applied; no backups, no Git commands, no GitHub writes.')

if __name__=='__main__':main()
