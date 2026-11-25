#!/usr/bin/env python3
"""Install Geometry10 into the existing audited Git checkout.

No sibling workspace, backup, network access, automatic commit or data changes.
A complete Git patch is checked before applying; Git remains the rollback mechanism.
"""
from __future__ import annotations
import argparse, ast, difflib, hashlib, subprocess, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE='92a1b00e5a0cbd5f770514396c0a347f5b9aca6d'
PINS={
'mbpo/env/pv_tracking.py':'d0b42d5d651ae77efa4f4f282495540ce072cb5c',
'mbpo/static/pv_tracking.py':'0f54b4cc2a56d39142ba9a4d90d3877712bfab6c',
'examples/config/pv_tracking/conf_compact.py':'3e1ea399af0b7457353f352b8b756970e294b201',
'SAC_workspace/src/config/config.py':'a0039ab99ca275e02feb15e10d54885b46dbc90a',
'SAC_workspace/src/environment/environment.py':'2b1f8e65e43a56ce2fe79356cdec1d97dfaa4176',
'SAC_workspace/src/representation/representation.py':'70cbc632ee7568c5be4065c103778cbc47adb43e',
'SAC_workspace/src/agents/sac_agent.py':'98311c5a82742529d803740ffdd76a50339dd30c',
'SAC_workspace/src/training/replay_buffer.py':'1f4803d3979825c444f5686e95ada35cce7ea41b',
'SAC_workspace/src/training/experiment_protocol.py':'caf11e2f6801e98ac6c8ed817117f3176ba0db8c',
'MBPO_workspace/src/config.py':'5be07701ddfb39e14efecf4d846fe2a4c51ebee4',
'MBPO_workspace/src/replay.py':'821490dd2d997515834d2ed96bdeab1280898d6e',
'MBPO_workspace/src/dynamics.py':'edff92372c70637058138023e99eb3a3efdc90a1',
'MBPO_workspace/src/trainer.py':'31611b94ad92a0761d58061ab2cb158419189e9f',
'MBPO_workspace/workflows/06_model_validation.py':'628825b968d05c1e2d27075b8f7ff672a785d17b',
'PV_experiments/core.py':'8ac6ac75cc58148c914f7ff5c8cb1f82d1d1f5e2',
'PostProcessing_MC_patch/tools/verify_mc_environment.py':'934d8d69b25600c86a17051e0e2a72e89fa458f6',
'PV_experiments/train.py':'8489caf2c271eb784d442a3f95baadb3404fa49a',
'SAC_workspace/workflows/01_setup_config.py':'298c6f22821069167cced36586555868bcb8412f',
'MBPO_workspace/workflows/01_setup_config.py':'2d8c4260743faefad8744f43ecad8774cbbd4622',
'examples/config/pv_tracking/_nsrdb_base.py':'48ba072203796006908ff63f5b07b20917065b26',
}


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args])


def blob_sha(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def replace(s,old,new,n=1):
    actual=s.count(old)
    if actual!=n:
        raise ValueError(f'Expected {n} occurrence(s), found {actual}: {old[:110]!r}')
    return s.replace(old,new)


def add_import(s, statement):
    # Safe insertion after the future import; for files without one, use known
    # import numpy line to avoid inserting in a module docstring.
    marker='from __future__ import annotations\n'
    if marker in s:return replace(s,marker,marker+statement+'\n')
    return replace(s,'import numpy as np\n','import numpy as np\n'+statement+'\n')


def transform(path,s):
    r=replace
    if path=='mbpo/env/pv_tracking.py':
        s=r(s,"OBSERVATION_MODES = ('legacy', 'physical', 'compact')","OBSERVATION_MODES = ('legacy', 'physical', 'compact', 'geometry10')")
        s=r(s,"        self.fixed_eval_dates = None", "        if observation_mode == 'geometry10':\n            from mbpo.geometry10 import FEATURE_NAMES\n            if self.observation_noise_std != 0.0:\n                raise ValueError('Geometry10 requires unperturbed orientation and solar geometry; use weather uncertainty instead')\n            self.obs_feature_names = FEATURE_NAMES\n        self.fixed_eval_dates = None")
        s=r(s,'        """Return observation bounds based on observation_mode."""','        """Return observation bounds based on observation_mode."""\n        if self.observation_mode == \'geometry10\':\n            from mbpo.geometry10 import LOW, HIGH\n            return LOW.copy(), HIGH.copy()')
        s=r(s,"        if self.observation_mode == 'compact':\n            return self._build_compact_observation", "        if self.observation_mode == 'geometry10':\n            from mbpo.geometry10 import build\n            tau = (self.num_action_steps-self.step_index)/float(self.num_action_steps)\n            return build(weather['dni'], weather['dhi'], power, self.tilt, self.azimuth,\n                         solar_zenith, solar_azimuth, tau)\n        if self.observation_mode == 'compact':\n            return self._build_compact_observation")
        s=s.replace('P/(2000*A*eta)', 'P/2000')
        s=r(s,'This 6D state captures all information needed for optimal tracking\n        without explicit solar position - the agent learns from irradiance\n        and power feedback directly.', 'The compact state omits explicit solar geometry and is partially observable.\n        Geometry10 retains measured power and adds known solar geometry.')
    elif path=='mbpo/static/pv_tracking.py':
        s=r(s,"    if dim == COMPACT_OBS_DIM:\n        return 'compact'", "    if dim == 10:\n        return 'geometry10'\n    if dim == COMPACT_OBS_DIM:\n        return 'compact'")
        s=s.replace('or {} compact).','or {} compact, or 10 geometry10).')
        s=r(s,"    if mode == 'compact':\n        return COMPACT_CYCLIC_SLICES", "    if mode == 'geometry10':\n        return ((4,6),(7,9))\n    if mode == 'compact':\n        return COMPACT_CYCLIC_SLICES")
        s=r(s,"        if mode == 'compact':\n            # Compact 6D: tau at index 5 (0-indexed), compute unwrapped time.\n            if obs.ndim == 1:\n                tau = obs[5]\n            else:\n                tau = obs[:, 5]", "        if mode in ('compact','geometry10'):\n            tau = obs[..., 9 if mode == 'geometry10' else 5]")
        s=r(s,'        zenith = obs[:, 0] * 180.0\n        azimuth = np.rad2deg(np.arctan2(obs[:, 1], obs[:, 2])) % 360.0',"        if observation_mode_from_obs(obs) == 'geometry10':\n            return StaticFns.time_of_day_from_obs(obs)[0] if single else StaticFns.time_of_day_from_obs(obs)\n        zenith = obs[:, 0] * 180.0\n        azimuth = np.rad2deg(np.arctan2(obs[:, 1], obs[:, 2])) % 360.0")
    elif path=='SAC_workspace/src/config/config.py':
        s=r(s,"'examples/config/pv_tracking/conf_compact.py'","'examples/config/pv_tracking/conf_geometry10.py'")
        s=s.replace('_COMPACT','_REFERENCE').replace('compact MBPO task','geometry-aware MBPO task').replace('conf_compact overrides','conf_geometry10 overrides')
        s=r(s,"_REFERENCE = ROOT / 'examples/config/pv_tracking/conf_geometry10.py'","_REFERENCE = ROOT / 'examples/config/pv_tracking/conf_geometry10.py'\n_COMPACT = ROOT / 'examples/config/pv_tracking/conf_compact.py'  # legacy import compatibility")
        s=r(s,"if nsrdb.get('observation_mode') != 'compact' or eval_env.get('observation_mode') != 'compact':", "if nsrdb.get('observation_mode') != 'geometry10' or eval_env.get('observation_mode') != 'geometry10':")
        s=r(s,'SAC requires the same compact observation mode as MBPO reference','SAC and MBPO must use the same geometry10 reference')
    elif path=='MBPO_workspace/src/config.py':
        s=s.replace('_COMPACT','_REFERENCE').replace('conf_compact','conf_geometry10')
        s=r(s,'mbpo-torch-separated-exogenous-v3','mbpo-torch-separated-exogenous-geometry10-v4')
    elif path=='SAC_workspace/src/environment/environment.py':
        s=add_import(s,'from mbpo.geometry10 import MODE, STATE_DIM, TAU, FEATURE_NAMES, LOW, HIGH, POWER_NORM')
        s=r(s,"env_kwargs.get('observation_mode') != 'compact'","env_kwargs.get('observation_mode') != MODE")
        s=r(s,'Must use compact mode for reference parity','Must use geometry10 mode in this workspace')
        s=r(s,'self.observation_space.shape != (6,)','self.observation_space.shape != (STATE_DIM,)')
        s=r(s,'six-feature compact representation','geometry10 representation')
        s=r(s,'        if self.action_space.shape != (2,):',"        if (tuple(self.env.obs_feature_names) != FEATURE_NAMES or\n                not np.array_equal(self.observation_space.low, LOW) or\n                not np.array_equal(self.observation_space.high, HIGH)):\n            raise ValueError('Geometry10 feature order or observation bounds mismatch')\n        if self.action_space.shape != (2,):")
        s=s.replace('observation[5]','observation[TAU]').replace('compact tau feature','geometry10 tau feature')
        s=r(s,"            if not np.isclose(observation[2], np.clip(float(info['power_dc_w']) / 2000., 0, 1), atol=1e-6):","            if not np.isclose(observation[2], float(info['power_dc_w']) / POWER_NORM, atol=1e-6):")
        s=r(s,"            'observation_mode': value('observation_mode'),", "            'observation_mode': value('observation_mode'),\n            'observation_features': list(FEATURE_NAMES),\n            'observation_contract_sha256': __import__('hashlib').sha256(__import__('pathlib').Path(__import__('mbpo.geometry10',fromlist=['build']).__file__).read_bytes()).hexdigest(),")
    elif path=='SAC_workspace/src/representation/representation.py':
        s=(HERE/'payload/representation.py').read_text()
    elif path=='SAC_workspace/src/agents/sac_agent.py':
        s=add_import(s,'from mbpo.geometry10 import STATE_DIM, ACTION_DIM, check_batch')
        s=r(s,'def __init__(self, state_dim=6, action_dim=2','def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM',3)
        s=r(s,'(state_dim, action_dim) != (6, 2)','(state_dim, action_dim) != (STATE_DIM, ACTION_DIM)')
        s=r(s,"Reference contract expects state_dim=6, action_dim=2","Reference contract expects geometry10 state and 2 actions")
        s=r(s,'        self.device = torch.device(device)','        self.state_dim = int(state_dim)\n        self.device = torch.device(device)')
        s=r(s,'        s = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).reshape(1,6)',"        state = check_batch(state)\n        if state.shape != (self.state_dim,):\n            raise ValueError('Expected a single geometry10 observation')\n        s = torch.as_tensor(state, device=self.device).reshape(1,self.state_dim)")
        s=r(s,"return {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),","return {'state_dim': self.state_dim, 'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),")
        s=r(s,"        s, a = batch['states'], batch['actions']", "        s, a = batch['states'], batch['actions']\n        n = len(s)\n        if (s.shape != (n,self.state_dim) or batch['next_states'].shape != s.shape\n                or a.shape != (n,ACTION_DIM) or batch['rewards'].shape != (n,1)\n                or batch['dones'].shape != (n,1) or n == 0):\n            raise ValueError('Incompatible SAC batch dimensions')")
        s=r(s,'    def load_state_dict(self, state):','    def load_state_dict(self, state):\n        if state.get(\'state_dim\') != self.state_dim:\n            raise ValueError(\'Incompatible observation dimension in checkpoint\')')
    elif path=='SAC_workspace/src/training/replay_buffer.py':
        s=add_import(s,'from mbpo.geometry10 import STATE_DIM, ACTION_DIM')
        s=r(s,'state_dim=6, action_dim=2','state_dim=STATE_DIM, action_dim=ACTION_DIM')
        s=r(s,'(state_dim, action_dim) != (6,2)','(state_dim, action_dim) != (STATE_DIM,ACTION_DIM)')
        s=r(s,'        self.states = np.empty((capacity, 6), dtype=np.float32)','        self.state_dim = int(state_dim)\n        self.states = np.empty((capacity, self.state_dim), dtype=np.float32)')
        s=r(s,'self.next_states = np.empty((capacity, 6), dtype=np.float32)','self.next_states = np.empty((capacity, self.state_dim), dtype=np.float32)')
        s=r(s,"return {'capacity': self.capacity, 'count': self.count, 'head': self.head,","return {'state_dim': self.state_dim, 'capacity': self.capacity, 'count': self.count, 'head': self.head,")
        s=r(s,"        if data['capacity'] != self.capacity:","        if data.get('state_dim') != self.state_dim:\n            raise ValueError('Replay observation dimension differs from checkpoint')\n        if data['capacity'] != self.capacity:")
        s=r(s,"            getattr(self,k)[:self.count] = data[k]", "            expected = getattr(self,k)[:self.count].shape\n            if np.asarray(data[k]).shape != expected or not np.isfinite(data[k]).all():\n                raise ValueError(f'Corrupt replay field {k}: expected {expected}')\n            getattr(self,k)[:self.count] = data[k]\n        from mbpo.geometry10 import check_batch\n        check_batch(self.states[:self.count]); check_batch(self.next_states[:self.count])")
    elif path=='SAC_workspace/src/training/experiment_protocol.py':
        s=r(s,'STATE_DIM = 6\nACTION_DIM = 2','from mbpo.geometry10 import MODE, STATE_DIM, ACTION_DIM')
        s=r(s,'        "state_dim": STATE_DIM,\n','        "state_dim": STATE_DIM,\n        "observation_mode": MODE,\n')
        s=r(s,'{STATE_DIM} compact observation features','{STATE_DIM} geometry-aware observation features')
        s=r(s,'"state_dim": STATE_DIM, "action_dim": ACTION_DIM,','"state_dim": STATE_DIM, "action_dim": ACTION_DIM, "observation_mode": MODE,')
    elif path=='MBPO_workspace/src/replay.py':
        s=add_import(s,'from mbpo.geometry10 import STATE_DIM, TAU')
        s=r(s,'def rollout_states(self,n,seed=None):','def rollout_states(self,n,seed=None,*,with_scenarios=False):')
        s=r(s,'self.states[:self.count,5]','self.states[:self.count,TAU]')
        s=r(s,'        if not len(valid):return np.empty((0,6),dtype=np.float32)','        if not len(valid):\n            empty=np.empty((0,STATE_DIM),dtype=np.float32)\n            return (empty, []) if with_scenarios else empty')
        s=r(s,'        return self.states[indices].copy()', '        states=self.states[indices].copy()\n        if with_scenarios:\n            return states, [self.scenario_names[int(self.scenario_ids[i])] for i in indices]\n        return states')
    elif path=='MBPO_workspace/src/dynamics.py':
        s=add_import(s,'from mbpo.geometry10 import (STATE_DIM, WEATHER_DIM, POWER_DIM, TAU, PANEL_AZ, SOLAR,\n    weather_features, check_batch, pose_transition, SolarContext)')
        s=r(s,'DYNAMICS_SCHEMA = 3','DYNAMICS_SCHEMA = 4')
        s=r(s,"'schema': DYNAMICS_SCHEMA, 'num_models': self.num_models,","'schema': DYNAMICS_SCHEMA, 'state_dim': STATE_DIM, 'num_models': self.num_models,")
        s=r(s,"if data.get('schema') != DYNAMICS_SCHEMA:","if data.get('schema') != DYNAMICS_SCHEMA or data.get('state_dim') != STATE_DIM:")
        s=r(s,"            setattr(self, field, np.asarray(data[field],dtype=np.float32))","            saved = np.asarray(data[field],dtype=np.float32)\n            if saved.shape != getattr(self,field).shape or not np.isfinite(saved).all() or (field.endswith('std') and np.any(saved <= 0)):\n                raise ValueError(f'Invalid saved dynamics scaler: {field}')\n            setattr(self, field, saved)")
        s=r(s,'GaussianDeltaNet(hidden_dim, 3, 2)','GaussianDeltaNet(hidden_dim, WEATHER_DIM, 2)')
        s=r(s,'GaussianDeltaNet(hidden_dim, 10, 1)','GaussianDeltaNet(hidden_dim, POWER_DIM, 1)')
        for fn in ('zeros','ones'):
            s=r(s,f'np.{fn}(3, np.float32)',f'np.{fn}(WEATHER_DIM, np.float32)')
            s=r(s,f'np.{fn}(10, np.float32)',f'np.{fn}(POWER_DIM, np.float32)')
        s=r(s,'states.shape[1] != 6','states.shape[1] != STATE_DIM',2)
        s=r(s,'Expected 6D observation and 2D action transitions','Expected geometry10 observation and 2D action transitions')
        s=r(s,'states[:, 5] <= 0','states[:, TAU] <= 0')
        s=r(s,'        yw = (nxt[:, :2] - states[:, :2]).astype(np.float32)', '        check_batch(nxt)\n        yw = (nxt[:, :2] - states[:, :2]).astype(np.float32)')
        s=r(s,'np.column_stack((states[:, 0], states[:, 1], states[:, 5])).astype(np.float32)','weather_features(states)',2)
        s=r(s,'Dynamics expects finite (N,6) states and bounded (N,2) actions','Dynamics expects finite geometry10 states and bounded (N,2) actions')
        s=r(s,'        return np.stack(mus).astype(np.float32), np.stack(sigmas).astype(np.float32)',"        mus, sigmas = np.stack(mus).astype(np.float32), np.stack(sigmas).astype(np.float32)\n        if not np.isfinite(mus).all() or not np.isfinite(sigmas).all() or np.any(sigmas < 0):\n            raise FloatingPointError('Invalid ensemble distribution parameters')\n        return mus, sigmas")
        s=r(s,'        self.ensemble = ensemble\n','        self.ensemble = ensemble\n        self.solar_context = SolarContext(reference_env)\n')
        s=r(s,'def step(self, states, actions, max_std=None, stochastic=True):','def step(self, states, actions, max_std=None, stochastic=True, *, scenario_ids=None):')
        start=s.index('        if (states.ndim!=2 or states.shape[1]!=6')
        end=s.index('        gross = nxt[:,2]*self.power_norm',start)
        s=s[:start]+'''        states = check_batch(states)
        if states.ndim != 2 or not len(states):
            raise ValueError('Expected nonempty geometry10 rollout batch')
        if actions.shape != (len(states),2) or not np.isfinite(actions).all() or np.any(np.abs(actions)>1.00001):
            raise ValueError('Invalid synthetic action batch')
        actions = np.clip(actions, -1., 1.)
        # Exact ephemeris from scenario + remaining time; no weather leakage.
        next_solar = self.solar_context.next(states, scenario_ids)
        new_tilt, new_az, tau, executed_tilt, executed_az, done = pose_transition(
            states, actions, self.max_delta_tilt, self.max_delta_azimuth, self.horizon)
        delta,uncertainty = self.ensemble.predict_delta(states,actions,stochastic=stochastic)
        delta = np.asarray(delta,dtype=np.float32)
        uncertainty = np.asarray(uncertainty,dtype=np.float32)
        if delta.shape != (len(states),3) or uncertainty.shape != (len(states),):
            raise ValueError('Invalid ensemble prediction dimensions')
        # Reject raw NaN/Inf before clipping can hide them.
        finite_prediction = np.isfinite(delta).all(axis=1) & np.isfinite(uncertainty) & (uncertainty >= 0)
        nxt=np.empty_like(states)
        nxt[:,:3] = np.clip(states[:,:3]+delta,0.,1.)
        nxt[:,3] = new_tilt
        nxt[:,PANEL_AZ] = new_az
        nxt[:,SOLAR] = next_solar
        nxt[:,TAU] = tau
''' +s[end:]
        s=r(s,'        mask=np.isfinite(nxt).all(axis=1)&np.isfinite(reward)&np.isfinite(uncertainty)', '        mask=finite_prediction & np.isfinite(nxt).all(axis=1)&np.isfinite(reward)\n        if np.any(mask): check_batch(nxt[mask])')
        a=s.index('"""');b=s.index('"""',a+3)
        s=s[:a]+'''"""Geometry10 one-step MBPO: learned DNI/DHI/power deltas, exact pose,
clock, solar ephemeris and actuator-energy calculation. The exogenous model
uses current solar geometry and tau, never actions or future weather."""'''+s[b+3:]
    elif path=='MBPO_workspace/src/trainer.py':
        s=add_import(s,'from mbpo.geometry10 import TAU')
        s=r(s,'states = self.real_pool.rollout_states(m.rollout_batch_size)','states, scenario_ids = self.real_pool.rollout_states(m.rollout_batch_size, with_scenarios=True)')
        s=r(s,'states, actions, max_std=m.uncertainty_max_std, stochastic=True)','states, actions, max_std=m.uncertainty_max_std, stochastic=True, scenario_ids=scenario_ids)')
        s=r(s,'state[5]','state[TAU]')
    elif path=='MBPO_workspace/workflows/06_model_validation.py':
        s=add_import(s,'from mbpo.geometry10 import TAU, PANEL_AZ, SOLAR, angular_rmse_deg')
        s=r(s,'states, actions, max_std=None, stochastic=False)','states, actions, max_std=None, stochastic=False, scenario_ids=[sid]*len(states))')
        s=r(s,'replay_obs[None, :], candidate[None, :], max_std=None, stochastic=False)','replay_obs[None, :], candidate[None, :], max_std=None, stochastic=False, scenario_ids=[sid])')
        s=r(s,'"azimuth_rmse_norm": _rmse(pred_next[:, 4] - next_states[:, 4]),','"azimuth_rmse_norm": angular_rmse_deg(pred_next[:,PANEL_AZ], next_states[:,PANEL_AZ])/180.,\n            "azimuth_rmse_deg": angular_rmse_deg(pred_next[:,PANEL_AZ], next_states[:,PANEL_AZ]),\n            "solar_geometry_rmse": _rmse(pred_next[:,SOLAR]-next_states[:,SOLAR]),')
        s=r(s,'pred_next[:, 5] - next_states[:, 5]','pred_next[:, TAU] - next_states[:, TAU]')
        s=r(s,'orientation and remaining horizon are reconstructed by the structured fake environment.','orientation, solar ephemeris and remaining horizon are reconstructed by the structured fake environment.')
    elif path=='PV_experiments/core.py':
        s=add_import(s,'from mbpo.geometry10 import MODE, STATE_DIM')
        s=r(s,"    if cfg.get('schema')!=SCHEMA or cfg.get('name')!=name:","    if cfg.get('schema')!=SCHEMA or cfg.get('name')!=name or cfg.get('observation_mode')!=MODE or cfg.get('state_dim')!=STATE_DIM:")
        s=r(s,"        'schema':SCHEMA,'name':name,'created_utc':utcnow(),","        'schema':SCHEMA,'name':name,'created_utc':utcnow(),\n        'observation_mode':MODE,'state_dim':STATE_DIM,")
        s=r(s,"'horizon','state_dim','action_dim',","'horizon','state_dim','action_dim','observation_mode',")
        s=r(s,"sac.get('state_dim')!=6","sac.get('state_dim')!=STATE_DIM or sac.get('observation_mode')!=MODE")
        s=r(s,'195-step, 6-state, 2-action simulator','195-step geometry10, 2-action simulator')
        s=r(s,"    if meta.get('training_status')=='completed':","    if meta.get('observation_mode')!=MODE or meta.get('state_dim')!=STATE_DIM:\n        raise ValueError(f'Run uses an incompatible observation contract: {d}. Use a fresh experiment name.')\n    if meta.get('training_status')=='completed':")
    elif path=='PV_experiments/train.py':
        s=add_import(s,'from mbpo.geometry10 import MODE, STATE_DIM')
        s=r(s,"verified=data.get('status')=='passed' and data.get('fixed_draw_replay') is True", "verified=(data.get('status')=='passed' and data.get('fixed_draw_replay') is True and data.get('observation_mode')==MODE and data.get('state_dim')==STATE_DIM)")
    elif path=='SAC_workspace/workflows/01_setup_config.py':
        s=r(s,'VALIDATED: compact 6D, NSRDB manifest and reference files.', 'VALIDATED: geometry10, NSRDB manifest and reference files.')
    elif path=='MBPO_workspace/workflows/01_setup_config.py':
        s=r(s,'Verified shared compact reference MDP: 6D, 2D,','Verified shared geometry10 reference: 10D, 2D,')
    elif path=='examples/config/pv_tracking/_nsrdb_base.py':
        s=r(s,"'conf5', 'conf_compact', 'conf_sac',", "'conf5', 'conf_compact', 'conf_geometry10', 'conf_sac',")
    elif path=='PostProcessing_MC_patch/tools/verify_mc_environment.py':
        s=add_import(s,'from mbpo.geometry10 import MODE, STATE_DIM')
        s=r(s,"report={'status':'passed','source':","report={'status':'passed','observation_mode':MODE,'state_dim':STATE_DIM,'source':")
        s=r(s,'    if env.horizon!=195 or obs.shape!=(6,) or env.action_space.shape!=(2,):','    from mbpo.geometry10 import STATE_DIM, check_batch, SOLAR, solar_features\n    if env.horizon!=195 or obs.shape!=(STATE_DIM,) or env.action_space.shape!=(2,):')
        s=r(s,"        if not np.isfinite(obs).all():raise AssertionError('Nonfinite state')","        check_batch(obs)\n        if not np.allclose(obs[SOLAR],solar_features(info['solar_zenith_deg'],info['solar_azimuth_deg']),atol=1e-6,rtol=0):\n            raise AssertionError('Observed solar geometry disagrees with physical timestamp')")
    elif path=='examples/config/pv_tracking/conf_compact.py':
        # No changes to original file; generate new config at separate path.
        a=s.index('"""');b=s.index('"""',a+3)
        s=s[:a]+'''"""Geometry10 managed SAC/MBPO reference. Same physical/hyperparameter
configuration as conf_compact; ten observation features only. This config is
read by the managed Torch configuration loader, not the legacy Ray runner."""'''+s[b+3:]
        s=s.replace('conf_compact','conf_geometry10').replace("'observation_mode': 'compact'","'observation_mode': 'geometry10'")
        s=s.replace('Compact six-feature state; actor and critic dimensions remain unchanged.','Geometry10 state; hidden widths remain unchanged.')
    else:raise ValueError(f'Unknown transform: {path}')
    ast.parse(s,filename=path)
    return s


