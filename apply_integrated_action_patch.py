#!/usr/bin/env python3
# Apply the integrated Geometry10 instantaneous-target action contract.
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import tempfile

EXPECTED_BLOBS = {
    "mbpo/geometry10.py": "45b6df5f89a407133da76d7c8c999d0fdbcf58d4",
    "mbpo/env/pv_tracking.py": "686be049de3aef8320a8bc213950967d29efd998",
    "SAC_workspace/src/environment/environment.py": "f03bbf29ff1d5bc01b3e424c0a8b59555190a6ba",
    "SAC_workspace/src/agents/sac_agent.py": "77eee8ee022ed81c9524d3e1c0a2402be76d97c6",
    "SAC_workspace/src/training/replay_buffer.py": "4c96d43d018d6a0791c6003d2ee76257463d8dbe",
    "SAC_workspace/src/training/experiment_protocol.py": "56c6ae27ad5298cb041cc029b71bb3bb2b90e426",
    "MBPO_workspace/src/dynamics.py": "63d717f15c1caf086bbda9df67b74c7e44ac4a83",
    "MBPO_workspace/workflows/09_monte_carlo_evaluate.py": "32d321daa1aabeb07518037ab8e6d0da47aad66a",
    "MBPO_workspace/workflows/06_model_validation.py": "72f8f5cdb38d9ed5b9d74dfd3061730e0c9632e7",
    "mbpo/audit_support.py": "74d3543409543b6bddc6fb800fa61803d7b8908b",
    "PV_experiments/representation/statistics.py": "85815e9e384635e17832150469272c1fb08adfc1",
    "PV_experiments/representation/trajectories.py": "a0bf03e0c8f5c15d0b54fdb04d5ca950f431f569",
}
ACTION_SEMANTICS = "instantaneous_target_tilt_absolute_azimuth_relative_v1"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def git_output(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one source match, found {count}")
    return source.replace(old, new, 1)


def replace_regex(source: str, pattern: str, replacement: str, label: str) -> str:
    result, count = re.subn(pattern, replacement, source, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one regex match, found {count}")
    return result


def transform_geometry10(s: str) -> str:
    s = replace_once(
        s, "ACTION_DIM = 2\nTAU = 9",
        f"ACTION_DIM = 2\nACTION_SEMANTICS = '{ACTION_SEMANTICS}'\nTAU = 9",
        "Geometry10 action contract constant",
    )
    helper = r'''

def target_orientation_from_action(current_tilt_deg, current_azimuth_deg, actions):
    # Decode target actions: absolute tilt and relative circular azimuth.
    a = np.asarray(actions, dtype=np.float64)
    if (a.ndim < 1 or a.shape[-1] != ACTION_DIM
            or not np.isfinite(a).all() or np.any(np.abs(a) > 1.0)):
        raise ValueError("Actions must be finite and inside [-1, 1]")
    old_tilt = np.asarray(current_tilt_deg, dtype=np.float64)
    old_azimuth = np.asarray(current_azimuth_deg, dtype=np.float64)
    try:
        old_tilt, old_azimuth, a_tilt, a_azimuth = np.broadcast_arrays(
            old_tilt, old_azimuth, a[..., 0], a[..., 1])
    except ValueError as exc:
        raise ValueError("Current pose and action batch shapes are incompatible") from exc
    if (not np.isfinite(old_tilt).all() or not np.isfinite(old_azimuth).all()
            or np.any((old_tilt < 0.0) | (old_tilt > 90.0))
            or np.any((old_azimuth < 0.0) | (old_azimuth >= 360.0))):
        raise ValueError("Current panel pose is outside its physical domain")
    target_tilt = 45.0 * (a_tilt + 1.0)
    target_azimuth = np.mod(old_azimuth + 180.0 * a_azimuth, 360.0)
    delta_tilt = target_tilt - old_tilt
    delta_azimuth = (target_azimuth - old_azimuth + 180.0) % 360.0 - 180.0
    if np.ndim(target_tilt) == 0:
        return (float(target_tilt), float(target_azimuth),
                float(delta_tilt), float(delta_azimuth))
    return target_tilt, target_azimuth, delta_tilt, delta_azimuth


def action_for_target_orientation(current_tilt_deg, current_azimuth_deg,
                                  target_tilt_deg, target_azimuth_deg):
    # Encode a physical target pose using the Geometry10 action convention.
    current_tilt, current_azimuth, target_tilt, target_azimuth = np.broadcast_arrays(
        np.asarray(current_tilt_deg, dtype=np.float64),
        np.asarray(current_azimuth_deg, dtype=np.float64),
        np.asarray(target_tilt_deg, dtype=np.float64),
        np.asarray(target_azimuth_deg, dtype=np.float64))
    if not all(np.isfinite(v).all() for v in
               (current_tilt, current_azimuth, target_tilt, target_azimuth)):
        raise ValueError("Current and target poses must be finite")
    if (np.any((current_tilt < 0.0) | (current_tilt > 90.0))
            or np.any((target_tilt < 0.0) | (target_tilt > 90.0))
            or np.any((current_azimuth < 0.0) | (current_azimuth >= 360.0))
            or np.any((target_azimuth < 0.0) | (target_azimuth >= 360.0))):
        raise ValueError("Current and target poses must lie in the physical domain")
    delta_azimuth = (target_azimuth - current_azimuth + 180.0) % 360.0 - 180.0
    return np.stack((target_tilt / 45.0 - 1.0, delta_azimuth / 180.0),
                    axis=-1).astype(np.float32)


def hold_action(current_tilt_deg):
    # Return the command that leaves both panel angles unchanged.
    tilt = np.asarray(current_tilt_deg, dtype=np.float64)
    if not np.isfinite(tilt).all() or np.any((tilt < 0.0) | (tilt > 90.0)):
        raise ValueError("Current tilt must be finite and inside [0, 90] degrees")
    return np.stack((tilt / 45.0 - 1.0, np.zeros_like(tilt)),
                    axis=-1).astype(np.float32)
'''
    s = replace_once(
        s,
        "def azimuth(pair):\n    a = np.asarray(pair)\n    return np.mod(np.rad2deg(np.arctan2(a[..., 0], a[..., 1])), 360.)\n",
        "def azimuth(pair):\n    a = np.asarray(pair)\n    return np.mod(np.rad2deg(np.arctan2(a[..., 0], a[..., 1])), 360.)\n" + helper,
        "Geometry10 shared target-action helpers",
    )
    build = r'''def build(dni, dhi, power, tilt, panel_azimuth, zenith, solar_azimuth, tau):
    raw = np.array([dni, dhi, power, tilt, panel_azimuth, zenith, solar_azimuth, tau], dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite physical observation")
    if not 0.0 <= dni <= IRRADIANCE_NORM:
        raise ValueError(
            f"DNI {dni} W/m^2 outside geometry10 normalization range [0, {IRRADIANCE_NORM}]")
    if not 0.0 <= dhi <= IRRADIANCE_NORM:
        raise ValueError(
            f"DHI {dhi} W/m^2 outside geometry10 normalization range [0, {IRRADIANCE_NORM}]")
    if not 0.0 <= power <= POWER_NORM:
        raise ValueError(
            f"DC power {power} W outside geometry10 normalization range [0, {POWER_NORM}]")
    if not 0.0 <= tilt <= 90.0:
        raise ValueError(f"Panel tilt {tilt} degrees outside [0, 90]")
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"Remaining-horizon fraction {tau} outside [0, 1]")
    obs = np.array([
        dni / IRRADIANCE_NORM, dhi / IRRADIANCE_NORM, power / POWER_NORM,
        tilt / 90.0, *circular(panel_azimuth),
        *solar_features(zenith, solar_azimuth), tau,
    ], dtype=np.float32)
    return check_batch(obs)


def weather_features'''
    s = replace_regex(
        s,
        r"def build\(dni, dhi, power, tilt, panel_azimuth, zenith, solar_azimuth, tau\):\n.*?\n\n\ndef weather_features",
        build,
        "Geometry10 observation normalization",
    )
    pose = r'''def pose_transition(states, actions, horizon):
    # Exact instantaneous target-pose and finite-horizon transition.
    x = check_batch(states)
    a = np.asarray(actions, dtype=np.float32)
    if (x.ndim != 2 or a.shape != (len(x), ACTION_DIM)
            or not np.isfinite(a).all() or np.any(np.abs(a) > 1.0)):
        raise ValueError("Invalid geometry10 action batch")
    if (not np.isfinite(horizon) or horizon <= 0 or np.any(x[:, TAU] <= 0)):
        raise ValueError("Invalid horizon or terminal starting state")
    old_tilt = x[:, 3].astype(float) * 90.0
    old_azimuth = azimuth(x[:, PANEL_AZ].astype(float))
    new_tilt, new_azimuth, delta_tilt, delta_azimuth = target_orientation_from_action(
        old_tilt, old_azimuth, a)
    remaining = np.rint(x[:, TAU].astype(float) * horizon).astype(int)
    if (np.any(remaining < 1) or np.any(remaining > horizon)
            or not np.allclose(x[:, TAU], remaining / horizon, atol=2e-6, rtol=0)):
        raise ValueError("Remaining time does not match finite-horizon grid")
    tau = (remaining - 1) / horizon
    return (new_tilt / 90.0, circular(new_azimuth), tau,
            delta_tilt, delta_azimuth, remaining == 1)


class SolarContext'''
    s = replace_regex(
        s,
        r"def pose_transition\(states,actions,max_tilt,max_az,horizon\):\n.*?\n\n\nclass SolarContext",
        pose,
        "Geometry10 pose transition",
    )
    return s


def transform_pv_tracking(s: str) -> str:
    old = '''        previous_tilt = float(self.tilt)
        previous_azimuth = float(self.azimuth)
        delta_tilt = float(action[0]) * self.max_delta_tilt
        delta_azimuth = float(action[1]) * self.max_delta_azimuth

        self.tilt = clip_panel_tilt_deg(self.tilt + delta_tilt)
        executed_delta_tilt = float(self.tilt - previous_tilt)
        self.azimuth = wrap_panel_azimuth_deg(self.azimuth + delta_azimuth)
        executed_delta_azimuth = float(
            (self.azimuth - previous_azimuth + 180.0) % 360.0 - 180.0)
'''
    new = '''        action = np.asarray(action, dtype=np.float32)
        if (action.shape != (2,) or not np.isfinite(action).all()
                or np.any(np.abs(action) > 1.0)):
            raise ValueError("Action must be finite, shape (2,), and inside [-1, 1]")
        previous_tilt = float(self.tilt)
        previous_azimuth = float(self.azimuth)
        if self.observation_mode == 'geometry10':
            from mbpo.geometry10 import ACTION_SEMANTICS, target_orientation_from_action
            target_tilt, target_azimuth, delta_tilt, delta_azimuth = (
                target_orientation_from_action(previous_tilt, previous_azimuth, action))
            self.tilt = float(target_tilt)
            self.azimuth = float(target_azimuth)
            action_semantics = ACTION_SEMANTICS
        else:
            # Preserve the original incremental-action behavior for legacy modes.
            delta_tilt = float(action[0]) * self.max_delta_tilt
            delta_azimuth = float(action[1]) * self.max_delta_azimuth
            self.tilt = clip_panel_tilt_deg(self.tilt + delta_tilt)
            self.azimuth = wrap_panel_azimuth_deg(self.azimuth + delta_azimuth)
            action_semantics = 'legacy_incremental_v1'
        executed_delta_tilt = float(self.tilt - previous_tilt)
        executed_delta_azimuth = float(
            (self.azimuth - previous_azimuth + 180.0) % 360.0 - 180.0)
'''
    s = replace_once(s, old, new, "PVTrackingEnv action interpretation")
    s = replace_once(
        s,
        "            'observation_mode': self.observation_mode,\n",
        "            'observation_mode': self.observation_mode,\n"
        "            'action_semantics': action_semantics,\n",
        "PVTrackingEnv action-semantics provenance",
    )
    s = replace_once(
        s,
        "        self.max_delta_tilt = max_delta_tilt\n        self.max_delta_azimuth = max_delta_azimuth\n",
        "        # Retained for legacy incremental modes and legacy-cost normalization;\n"
        "        # Geometry10 target actions are not slew-limited by these values.\n"
        "        self.max_delta_tilt = max_delta_tilt\n"
        "        self.max_delta_azimuth = max_delta_azimuth\n",
        "PVTrackingEnv legacy increment compatibility",
    )
    return s


def transform_sac_environment(s: str) -> str:
    s = replace_once(
        s,
        "from mbpo.geometry10 import MODE, STATE_DIM, TAU, FEATURE_NAMES, LOW, HIGH, POWER_NORM",
        "from mbpo.geometry10 import (MODE, STATE_DIM, TAU, FEATURE_NAMES, LOW, HIGH,\n"
        "    POWER_NORM, ACTION_SEMANTICS)",
        "SACEnvironment action-contract import",
    )
    s = replace_once(
        s,
        "            'horizon': self.horizon,\n",
        "            'horizon': self.horizon,\n"
        "            'action_semantics': ACTION_SEMANTICS,\n",
        "SACEnvironment resolved action semantics",
    )
    s = replace_once(
        s,
        "        if (np.abs(a) > 1.00001).any():\n"
        "            raise ValueError('Policy generated action outside [-1,1]^2')\n"
        "        obs, reward, done, info = self.env.step(np.clip(a, -1., 1.))\n",
        "        if (np.abs(a) > 1.0).any():\n"
        "            raise ValueError('Policy generated action outside [-1,1]^2')\n"
        "        obs, reward, done, info = self.env.step(a)\n"
        "        if info.get('action_semantics') != ACTION_SEMANTICS:\n"
        "            raise RuntimeError('Physical environment action semantics differ from Geometry10 contract')\n",
        "SACEnvironment non-saturating action validation",
    )
    return s


def transform_sac_agent(s: str) -> str:
    return replace_once(
        s,
        "        return np.clip(a.squeeze(0).cpu().numpy(), -1., 1.).astype(np.float32)\n",
        "        # The actor's tanh already enforces the action domain; do not distort it again.\n"
        "        return a.squeeze(0).cpu().numpy().astype(np.float32)\n",
        "SACAgent redundant action clipping",
    )


def transform_replay_buffer(s: str) -> str:
    if s.count("1.00001") != 2:
        raise RuntimeError("ReplayBuffer: expected two action-bound validation tolerances")
    return s.replace("1.00001", "1.0")


def transform_dynamics(s: str) -> str:
    s = replace_once(s, "DYNAMICS_SCHEMA = 4", "DYNAMICS_SCHEMA = 5",
                     "Dynamics checkpoint semantic version")
    if s.count("1.00001") != 3:
        raise RuntimeError("Dynamics: expected three action-bound validation tolerances")
    s = s.replace("1.00001", "1.0")
    s = replace_once(
        s,
        "        actions = np.clip(actions, -1., 1.)\n"
        "        # Exact ephemeris from scenario + remaining time; no weather leakage.\n",
        "        # Actions are validated above; preserve the exact policy command.\n"
        "        # Exact ephemeris from scenario + remaining time; no weather leakage.\n",
        "StructuredFakeEnv redundant action clipping",
    )
    s = replace_once(
        s,
        "        new_tilt, new_az, tau, executed_tilt, executed_az, done = pose_transition(\n"
        "            states, actions, self.max_delta_tilt, self.max_delta_azimuth, self.horizon)\n",
        "        new_tilt, new_az, tau, executed_tilt, executed_az, done = pose_transition(\n"
        "            states, actions, self.horizon)\n",
        "StructuredFakeEnv shared instantaneous pose transition",
    )
    s = replace_once(
        s,
        "        nxt=np.empty_like(states)\n"
        "        nxt[:,:3] = np.clip(states[:,:3]+delta,0.,1.)\n",
        "        nxt=np.empty_like(states)\n"
        "        predicted_exogenous = states[:, :3] + delta\n"
        "        bounds_violation = ((predicted_exogenous < -1e-5)\n"
        "                            | (predicted_exogenous > 1.0 + 1e-5)).any(axis=1)\n"
        "        if np.any(finite_prediction & bounds_violation):\n"
        "            raise FloatingPointError('Dynamics returned an unprojected physical state')\n"
        "        # predict_delta performs the single physical projection; do not project twice here.\n"
        "        nxt[:, :3] = predicted_exogenous\n",
        "StructuredFakeEnv duplicate next-state projection",
    )
    return s


def transform_experiment_protocol(s: str) -> str:
    s = replace_once(
        s,
        "from mbpo.geometry10 import MODE, STATE_DIM, ACTION_DIM",
        "from mbpo.geometry10 import (MODE, STATE_DIM, ACTION_DIM, ACTION_SEMANTICS,\n"
        "    hold_action, action_for_target_orientation)",
        "PeriodicEvaluator action helpers",
    )
    s = replace_once(
        s,
        '            "policy_evaluation": "deterministic_actor_mean_tanh",\n',
        '            "policy_evaluation": "deterministic_actor_mean_tanh",\n'
        '            "action_semantics": ACTION_SEMANTICS,\n'
        '            "astronomical_baseline": "instantaneous_dual_axis_target",\n',
        "PeriodicEvaluator protocol action provenance",
    )
    old = '''            elif controller == "fixed":
                action = np.zeros(2, dtype=np.float32)
            elif controller == "dual_axis_slew_limited":
                if float(underlying.max_delta_tilt) <= 0 or float(underlying.max_delta_azimuth) <= 0:
                    raise ValueError("Positive action increments are required for the dual-axis baseline")
                solar = underlying._solar_position(underlying.current_time)
                if float(solar.zenith) >= 90.0:
                    action = np.zeros(2, dtype=np.float32)
                else:
                    from mbpo.env.pvlib_physics import dual_axis_tracking_targets
                    target_tilt, target_azimuth = dual_axis_tracking_targets(
                        float(solar.zenith), float(solar.azimuth))
                    delta_tilt = np.clip(float(target_tilt) - float(underlying.tilt),
                                         -underlying.max_delta_tilt, underlying.max_delta_tilt)
                    delta_azimuth = ((float(target_azimuth) - float(underlying.azimuth) + 180.0)
                                     % 360.0) - 180.0
                    delta_azimuth = np.clip(delta_azimuth,
                                            -underlying.max_delta_azimuth,
                                            underlying.max_delta_azimuth)
                    action = np.asarray([
                        delta_tilt / underlying.max_delta_tilt,
                        delta_azimuth / underlying.max_delta_azimuth,
                    ], dtype=np.float32)
'''
    new = '''            elif controller == "fixed":
                action = hold_action(float(underlying.tilt))
            elif controller == "dual_axis_instantaneous":
                solar = underlying._solar_position(underlying.current_time)
                if float(solar.zenith) >= 90.0:
                    action = hold_action(float(underlying.tilt))
                else:
                    from mbpo.env.pvlib_physics import dual_axis_tracking_targets
                    target_tilt, target_azimuth = dual_axis_tracking_targets(
                        float(solar.zenith), float(solar.azimuth))
                    action = action_for_target_orientation(
                        float(underlying.tilt), float(underlying.azimuth),
                        float(target_tilt), float(target_azimuth))
'''
    s = replace_once(s, old, new, "PeriodicEvaluator physical baselines")
    return replace_once(
        s,
        'dual = self._run_episode(sid, controller="dual_axis_slew_limited")',
        'dual = self._run_episode(sid, controller="dual_axis_instantaneous")',
        "PeriodicEvaluator baseline controller name",
    )


def transform_mc_evaluator(s: str) -> str:
    s = replace_once(
        s,
        "from mbpo.env.trajectory_uncertainty import normalized_spec,spec_fingerprint\n",
        "from mbpo.env.trajectory_uncertainty import normalized_spec,spec_fingerprint\n"
        "from mbpo.geometry10 import hold_action, action_for_target_orientation\n",
        "Monte Carlo action helpers",
    )
    s = s.replace("sun_slew_limited", "sun_tracking")
    old = '''def action_for(method,env,obs,agent):
    if method=='fixed_no_motion':return np.zeros(2,dtype=np.float32)
    if method in ('mbpo','sac'):return agent.act(obs,deterministic=True)
    if method!='sun_tracking':raise ValueError(method)
    from mbpo.env.pvlib_physics import dual_axis_tracking_targets
    u=env.unwrapped
    # Match the existing PeriodicEvaluator astronomical baseline: the
    # sun position at the CURRENT decision timestamp determines the action.
    # The plant then advances to t+1 and computes the resulting power.
    sp=u._solar_position(u.current_time)
    if float(sp.zenith)>=90.:return np.zeros(2,dtype=np.float32)
    tilt,az=dual_axis_tracking_targets(float(sp.zenith),float(sp.azimuth))
    dt=np.clip((tilt-u.tilt)/u.max_delta_tilt,-1,1)
    da=((az-u.azimuth+180)%360-180)/u.max_delta_azimuth
    return np.array([dt,np.clip(da,-1,1)],dtype=np.float32)
'''
    new = '''def action_for(method,env,obs,agent):
    u=env.unwrapped
    if method=='fixed_no_motion':
        return hold_action(float(u.tilt))
    if method in ('mbpo','sac'):
        return agent.act(obs,deterministic=True)
    if method!='sun_tracking':
        raise ValueError(method)
    from mbpo.env.pvlib_physics import dual_axis_tracking_targets
    # Current decision-time sun position defines the instantaneous target.
    # The physical environment advances to t+1 after applying that target.
    sp=u._solar_position(u.current_time)
    if float(sp.zenith)>=90.:
        return hold_action(float(u.tilt))
    tilt,az=dual_axis_tracking_targets(float(sp.zenith),float(sp.azimuth))
    return action_for_target_orientation(
        float(u.tilt),float(u.azimuth),float(tilt),float(az))
'''
    return replace_once(s, old, new, "Monte Carlo physical baseline actions")


def transform_statistics(s: str) -> str:
    s = s.replace("sun_slew_limited", "sun_tracking")
    return replace_once(s, "Astronomical (slew limited)", "Astronomical (instantaneous)",
                        "Representation baseline label")


def transform_trajectories(s: str) -> str:
    s = replace_once(
        s,
        "from .common import Figures,save_csv,write_json,atomic_json,read_json,sha,style,line,plt\n",
        "from .common import Figures,save_csv,write_json,atomic_json,read_json,sha,style,line,plt\n"
        "from mbpo.geometry10 import (ACTION_SEMANTICS, hold_action,\n"
        "    action_for_target_orientation, target_orientation_from_action)\n",
        "Trajectory action-contract imports",
    )
    s = s.replace("sun_slew_limited", "sun_tracking")
    recorder_step = r'''    def step(self,action):
        u=self.unwrapped
        previous=(float(u.tilt),float(u.azimuth))
        a=finite(action)
        if a.shape!=(2,) or (np.abs(a)>1.0).any():
            raise ValueError('Invalid command')
        if self.method=='fixed_no_motion':
            check_close(a,hold_action(previous[0]),'Fixed baseline hold action')
        if self.method=='sun_tracking':
            sp=u._solar_position(u.current_time)
            if sp.zenith>=90:
                expected=hold_action(previous[0])
            else:
                from mbpo.env.pvlib_physics import dual_axis_tracking_targets
                target_tilt,target_azimuth=dual_axis_tracking_targets(
                    float(sp.zenith),float(sp.azimuth))
                expected=action_for_target_orientation(
                    previous[0],previous[1],float(target_tilt),float(target_azimuth))
            check_close(a,expected,'Astronomical instantaneous target command',atol=1e-6)

        obs,reward,done,info=self.env.step(action)
        if info.get('action_semantics')!=ACTION_SEMANTICS:
            raise ValueError('Physical transition used an unexpected action contract')
        dt=float(info['delta_tilt_deg'])
        da=float(info['delta_azimuth_deg'])
        tilt=float(info['tilt'])
        az=float(info['azimuth'])
        if not (0<=tilt<=90 and 0<=az<360):
            raise ValueError('Panel orientation outside valid domain')

        target_tilt,target_azimuth,expected_dt,expected_da=target_orientation_from_action(
            previous[0],previous[1],a)
        check_close([dt,da],[expected_dt,expected_da],'Executed target displacement')
        check_close([tilt,shortest(az,target_azimuth)],[target_tilt,0.],
                    'Action-to-target pose mapping',atol=1e-5)

        # The target is instantaneous in the MDP; motor energy still reflects
        # physical travel time. The maximum possible sequential travel is
        # 90/tilt_rate + 180/azimuth_rate and must fit the five-minute interval.
        seconds=abs(dt)/u.slew_rate_tilt_deg_s+abs(da)/u.slew_rate_azimuth_deg_s
        if seconds>u.interval_hours*3600+1e-7:
            raise ValueError('Sequential actuator time exceeds interval')
        cost=u.movement_penalty*u.actuator_power_w*seconds/3.6e6
        if u.movement_cost_mode!='geometry':
            raise ValueError('Nonphysical movement cost mode')
        check_close(info['movement_cost'],cost,'Actuator energy',atol=1e-10)
        check_close(info['energy_kwh'],info['power_dc_w']*u.interval_hours/1000,
                    'Physical power integration',atol=1e-10)
        check_close(reward,info['energy_kwh']-cost,'Net reward',atol=1e-10)
        if self.method=='fixed_no_motion':
            check_close([tilt,az,dt,da,cost],[*self.pose,0,0,0],
                        'Fixed pose and cost')
        self.records.append([info['power_dc_w'],info['energy_kwh'],cost,reward,
            tilt,az,a[0],a[1],dt,da,info['poa_global'],info['cell_temperature_c']])
        self.info.append(dict(info))
        return obs,reward,done,info
'''
    s = replace_regex(
        s,
        r"    def step\(self,action\):\n.*?\n\n\ndef oracle_from_weather",
        recorder_step + "\n\n\ndef oracle_from_weather",
        "Trajectory recorder action/physics audit",
    )
    s = replace_once(
        s,
        ",max_delta_tilt_deg=float(u.max_delta_tilt),max_delta_azimuth_deg=float(u.max_delta_azimuth)",
        "",
        "Remove obsolete slew-limit metadata",
    )
    s = replace_once(
        s,
        "astronomical='current decision-time sun position, same slew bounds, darkness holds; no future-weather access'",
        "astronomical='current decision-time sun position, instantaneous dual-axis target, darkness holds; no future-weather access'",
        "Trajectory baseline description",
    )
    s = replace_once(
        s,
        "Commands timestamped at decision time; normalized requested increments, not executed angles.",
        "Commands timestamped at decision time; normalized tilt-target and relative-azimuth coordinates, not executed angles.",
        "Trajectory action interpretation description",
    )
    s = replace_once(
        s,
        "('Absolute tilt-target command','Relative azimuth-target command','Absolute executed tilt (deg)','Absolute executed azimuth (deg)')",
        "('Normalized tilt-target coordinate','Relative azimuth-target coordinate','Absolute executed tilt (deg)','Absolute executed azimuth (deg)')",
        "Trajectory action-axis labels",
    )
    s = replace_once(
        s,
        "methods=list(METHODS),sources=data['sources'],collector_sha256=sha(Path(__file__)))",
        "methods=list(METHODS),action_semantics=ACTION_SEMANTICS,sources=data['sources'],collector_sha256=sha(Path(__file__)))",
        "Trajectory collection provenance",
    )
    s = replace_once(
        s,
        "'methods':list(METHODS),'sources':data['sources']}",
        "'methods':list(METHODS),'action_semantics':ACTION_SEMANTICS,'sources':data['sources']}",
        "Trajectory rendering provenance",
    )
    s = replace_once(
        s,
        "fixed_pose_deg=[30,180],actuator_parameters=audits[0]['actuator_parameters']",
        "fixed_pose_deg=[30,180],action_semantics=ACTION_SEMANTICS,actuator_parameters=audits[0]['actuator_parameters']",
        "Trajectory audit action provenance",
    )
    return s


def transform_model_validation(s: str) -> str:
    s = replace_once(
        s,
        "from mbpo.geometry10 import TAU, PANEL_AZ, SOLAR, angular_rmse_deg",
        "from mbpo.geometry10 import TAU, PANEL_AZ, SOLAR, angular_rmse_deg, hold_action",
        "Counterfactual hold-action import",
    )
    helper = r'''def _counterfactual_actions(policy_action, state):
    # Action probes with independent tilt/azimuth effects under target semantics.
    hold = hold_action(float(state[3]) * 90.0)
    return [
        ("policy", np.asarray(policy_action, dtype=np.float32)),
        ("hold", hold),
        ("tilt_positive", np.asarray([1.0, 0.0], dtype=np.float32)),
        ("tilt_negative", np.asarray([-1.0, 0.0], dtype=np.float32)),
        ("azimuth_positive", np.asarray([hold[0], 0.5], dtype=np.float32)),
        ("azimuth_negative", np.asarray([hold[0], -0.5], dtype=np.float32)),
        ("diagonal_positive", np.asarray([1.0, 0.5], dtype=np.float32)),
        ("diagonal_negative", np.asarray([-1.0, -0.5], dtype=np.float32)),
    ]


def run_validation'''
    s = replace_regex(
        s,
        r"def _counterfactual_actions\(policy_action\):\n.*?\n\n\ndef run_validation",
        helper,
        "Counterfactual action probes",
    )
    return replace_once(
        s,
        "for action_name, candidate in _counterfactual_actions(actions[t]):",
        "for action_name, candidate in _counterfactual_actions(actions[t], states[t]):",
        "Counterfactual state-aware hold action",
    )


def transform_audit_support(s: str) -> str:
    return replace_once(
        s,
        "        zero=[i for i,r in enumerate(block) if r['action_type']=='zero']\n"
        "        if len(zero)!=1: raise ValueError('Counterfactual group needs one zero action')\n"
        "        j=zero[0]; others=np.arange(len(block))!=j\n",
        "        hold=[i for i,r in enumerate(block) if r['action_type']=='hold']\n"
        "        if len(hold)!=1: raise ValueError('Counterfactual group needs one hold action')\n"
        "        j=hold[0]; others=np.arange(len(block))!=j\n",
        "Counterfactual reference action semantics",
    )


TRANSFORMS = {
    "mbpo/geometry10.py": transform_geometry10,
    "mbpo/env/pv_tracking.py": transform_pv_tracking,
    "SAC_workspace/src/environment/environment.py": transform_sac_environment,
    "SAC_workspace/src/agents/sac_agent.py": transform_sac_agent,
    "SAC_workspace/src/training/replay_buffer.py": transform_replay_buffer,
    "SAC_workspace/src/training/experiment_protocol.py": transform_experiment_protocol,
    "MBPO_workspace/src/dynamics.py": transform_dynamics,
    "MBPO_workspace/workflows/09_monte_carlo_evaluate.py": transform_mc_evaluator,
    "MBPO_workspace/workflows/06_model_validation.py": transform_model_validation,
    "mbpo/audit_support.py": transform_audit_support,
    "PV_experiments/representation/statistics.py": transform_statistics,
    "PV_experiments/representation/trajectories.py": transform_trajectories,
}


def self_test_geometry10(source: str) -> None:
    with tempfile.TemporaryDirectory(prefix="geometry10-action-check-") as tmp:
        module_path = Path(tmp) / "geometry10_under_test.py"
        module_path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("geometry10_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)

        tilt, az, dt, da = module.target_orientation_from_action(30.0, 359.0, [0.0, 0.0])
        assert abs(tilt - 45.0) < 1e-10 and abs(az - 359.0) < 1e-10
        assert abs(dt - 15.0) < 1e-10 and abs(da) < 1e-10

        tilt, az, dt, da = module.target_orientation_from_action(30.0, 359.0, [0.5, 0.01])
        assert abs(tilt - 67.5) < 1e-10 and abs(az - 0.8) < 1e-8
        assert abs(dt - 37.5) < 1e-10 and abs(da - 1.8) < 1e-8

        hold = module.hold_action(30.0)
        decoded = module.target_orientation_from_action(30.0, 180.0, hold)
        assert abs(decoded[0] - 30.0) < 1e-6 and abs(decoded[1] - 180.0) < 1e-6
        assert abs(decoded[2]) < 1e-6 and abs(decoded[3]) < 1e-6

        target_action = module.action_for_target_orientation(30.0, 359.0, 90.0, 1.0)
        decoded = module.target_orientation_from_action(30.0, 359.0, target_action)
        assert abs(decoded[0] - 90.0) < 1e-5 and abs(decoded[1] - 1.0) < 1e-5

        state = module.build(500.0, 100.0, 100.0, 30.0, 359.0, 40.0, 100.0, 1.0)
        nxt = module.pose_transition(state[None, :], [[0.0, 0.0]], 195)
        assert abs(float(nxt[0][0]) - 0.5) < 1e-6
        assert abs(float(module.azimuth(nxt[1][0])) - 359.0) < 1e-5
        assert abs(float(nxt[2][0]) - 194.0 / 195.0) < 1e-8
        assert not bool(nxt[5][0])

        terminal = state.copy()
        terminal[-1] = 1.0 / 195.0
        nxt = module.pose_transition(terminal[None, :], [[-1.0, 1.0]], 195)
        assert bool(nxt[5][0]) and abs(float(nxt[2][0])) < 1e-12

        for bad in ([1.0001, 0.0], [0.0, float("nan")]):
            try:
                module.target_orientation_from_action(30.0, 180.0, bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid action was not rejected: {bad}")

        try:
            module.build(2001.0, 100.0, 100.0, 30.0, 180.0, 40.0, 100.0, 1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("Out-of-range DNI was silently saturated")


def verify_static_contract(changed: dict[str, str]) -> None:
    g = changed["mbpo/geometry10.py"]
    assert f"ACTION_SEMANTICS = '{ACTION_SEMANTICS}'" in g
    assert "def pose_transition(states, actions, horizon):" in g
    assert "a = np.clip" not in g
    assert "np.clip(dni" not in g and "np.clip(dhi" not in g

    env = changed["mbpo/env/pv_tracking.py"]
    assert "if self.observation_mode == 'geometry10':" in env
    assert "target_orientation_from_action(previous_tilt, previous_azimuth, action)" in env
    assert "'action_semantics': action_semantics" in env

    adapter = changed["SAC_workspace/src/environment/environment.py"]
    assert "self.env.step(a)" in adapter
    assert "np.clip(a" not in adapter
    assert "ACTION_SEMANTICS" in adapter

    agent = changed["SAC_workspace/src/agents/sac_agent.py"]
    assert "return a.squeeze(0).cpu().numpy().astype(np.float32)" in agent
    assert "return np.clip(a.squeeze" not in agent

    dyn = changed["MBPO_workspace/src/dynamics.py"]
    assert "DYNAMICS_SCHEMA = 5" in dyn
    assert "pose_transition(states, actions, self.horizon)" in dyn
    assert "actions = np.clip" not in dyn
    assert "nxt[:,:3] = np.clip" not in dyn

    for path in (
        "MBPO_workspace/workflows/09_monte_carlo_evaluate.py",
        "PV_experiments/representation/statistics.py",
        "PV_experiments/representation/trajectories.py",
    ):
        assert "sun_slew_limited" not in changed[path], f"Stale baseline name in {path}"
    protocol = changed["SAC_workspace/src/training/experiment_protocol.py"]
    assert "dual_axis_slew_limited" not in protocol
    assert "hold_action(float(underlying.tilt))" in protocol
    assert "action_for_target_orientation" in protocol
    mc = changed["MBPO_workspace/workflows/09_monte_carlo_evaluate.py"]
    assert "sun_slew_limited" not in mc
    assert "hold_action(float(u.tilt))" in mc
    assert "action_for_target_orientation" in mc
    trajectories = changed["PV_experiments/representation/trajectories.py"]
    assert "target_orientation_from_action" in trajectories
    assert "action_for_target_orientation" in trajectories
    assert "ACTION_SEMANTICS" in trajectories
    assert "action_type']=='zero'" not in changed["mbpo/audit_support.py"]
    assert "action_type']=='hold'" in changed["mbpo/audit_support.py"]
    assert '("hold", hold)' in changed["MBPO_workspace/workflows/06_model_validation.py"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the integrated target-action patch.")
    parser.add_argument("--repo", default=".", help="Path to the clean mbpo_custom checkout")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()

    if not (repo / ".git").exists() and not (repo / ".git").is_file():
        raise SystemExit(f"Not a Git checkout: {repo}")
    status = git_output(repo, "status", "--porcelain").decode("utf-8").strip()
    if status:
        raise SystemExit("Refusing to patch a dirty checkout. Create/switch to your new branch and clear existing changes first.")

    originals: dict[str, str] = {}
    worktree_originals: dict[str, bytes] = {}
    changed: dict[str, str] = {}
    for rel, expected_sha in EXPECTED_BLOBS.items():
        raw = git_output(repo, "show", f"HEAD:{rel}")
        actual_sha = git_blob_sha(raw)
        if actual_sha != expected_sha:
            raise SystemExit(
                f"Source mismatch for {rel}: expected {expected_sha}, got {actual_sha}. "
                "This patch is pinned to the reviewed PostProcessing snapshot; do not force it."
            )
        original = raw.decode("utf-8")
        originals[rel] = original
        worktree_originals[rel] = (repo / rel).read_bytes()
        changed[rel] = TRANSFORMS[rel](original)

    verify_static_contract(changed)
    for rel, content in changed.items():
        compile(content, str(repo / rel), "exec")
    self_test_geometry10(changed["mbpo/geometry10.py"])

    if args.dry_run:
        print("PASS: source hashes, exact replacements, static contract, syntax and action tests.")
        print("No files written.")
        return 0

    written: list[str] = []
    try:
        for rel, content in changed.items():
            target = repo / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".integrated-action.tmp")
            try:
                with tmp.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(content)
                os.replace(tmp, target)
                written.append(rel)
            finally:
                if tmp.exists():
                    tmp.unlink()
        subprocess.check_call(["git", "diff", "--check"], cwd=repo)
    except Exception:
        for rel in reversed(written):
            (repo / rel).write_bytes(worktree_originals[rel])
        raise

    print("PASS: integrated instantaneous-target action patch applied.")
    print("Changed files:")
    for rel in changed:
        print(f"  {rel}")
    print("No backup files, commits, pushes, or training runs were created.")
    print("Next: run the bundled unit tests and the smoke-test sequence in README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
