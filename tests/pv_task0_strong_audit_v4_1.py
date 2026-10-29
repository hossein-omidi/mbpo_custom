#!/usr/bin/env python3
"""Read-only V4.1 audit of NYC NSRDB scenario selection, physics and bookkeeping.

Run from the mbpo_custom repository root with the project environment active:
    python /path/to/pv_task0_strong_audit_v4_1.py
    python /path/to/pv_task0_strong_audit_v4_1.py --json audit_results.json
    python /path/to/pv_task0_strong_audit_v4_1.py --scenarios 2022-06-17 2022-07-15

Verifies scenario selection API, per-step pvlib AND hand-derived isotropic POA,
normalization, time window, cloud diagnostics, deterministic rollouts and movement
costs. Additionally, checks NumPy RNG compatibility and asserts executed-motion
actuator costs agree between the real environment and the MBPO fake environment. An actual runtime bug is a FAIL
with nonzero exit code, even when the rest of the audit passes. This script does
NOT modify the simulator, weather data, training code, or checkpoints.
Does NOT prove field accuracy, train/test separation, or checkpoint evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

# The script may live outside the repository (e.g., Downloads). Always run
# from the mbpo_custom repository root and make its packages importable.
REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "SAC_workspace").is_dir() or not (REPO_ROOT / "mbpo").is_dir():
    raise SystemExit("Run this script from the mbpo_custom repository root directory.")
sys.path.insert(0, str(REPO_ROOT))

from SAC_workspace.src.config.config import load_config
from SAC_workspace.src.environment.environment import SACEnvironment
from mbpo.env.nsrdb_weather import load_scenario_manifest, scenario_id
from mbpo.env.pv_tracking import IRRADIANCE_NORM, POWER_NORM, PVTrackingEnv
from mbpo.env.pvlib_physics import compute_panel_power_w

DEFAULT_SCENARIOS = (
    "2022-01-15", "2022-04-15", "2022-07-15", "2022-10-15",
)
EVALUATION_MODULES = (
    "SAC_workspace.src.evaluation.evaluation",
    "SAC_workspace.src.evaluation.plots",
    "SAC_workspace.src.evaluation.scenario_sampling",
    "SAC_workspace.src.evaluation.stratified_statistics",
    "SAC_workspace.src.evaluation.statistical_inference",
    "MBPO_workspace.src.evaluation",
)


def verify(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def near(actual, expected, label, *, atol=1e-6, rtol=1e-9):
    verify(
        np.isfinite(actual)
        and np.isfinite(expected)
        and np.isclose(actual, expected, atol=atol, rtol=rtol),
        f"{label}: actual={actual!r}, expected={expected!r}",
    )


def actions_for(step: int, policy: str) -> np.ndarray:
    if policy == "stationary":
        return np.array([0., 0.], dtype=np.float32)
    if policy == "north_slew":
        # From a deterministic south-facing 30deg/180deg start:
        # Steps 1..10: tilt +5deg, az +10deg; steps 11..18: az +10deg.
        # From step 18 onward, the panel is fixed at 80deg tilt, 0deg azimuth.
        if step < 10:
            return np.array([1., 1.], dtype=np.float32)
        if step < 18:
            return np.array([0., 1.], dtype=np.float32)
        return np.array([0., 0.], dtype=np.float32)
    if policy == "tilt_upper_saturate":
        # Starts at 30deg, reaches 90deg after 12 steps, then repeatedly
        # requests blocked +5deg tilt. Nonzero cost must stop at the limit.
        return np.array([1., 0.], dtype=np.float32)
    if policy == "tilt_lower_saturate":
        # Starts at 30deg, reaches 0deg after 6 steps, then repeatedly
        # requests blocked -5deg tilt. Nonzero cost must stop at the limit.
        return np.array([-1., 0.], dtype=np.float32)
    raise ValueError(policy)


def one_episode(cfg, sid: str, *, policy: str, movement_penalty: float, seed: int = 2026):
    kwargs = dict(cfg.train_env)
    kwargs.update(
        randomize_initial_orientation=False,
        randomize_scenario=False,
        irradiance_perturbation_std=0.0,
        observation_noise_std=0.0,
        movement_cost_mode="geometry",
        movement_penalty=movement_penalty,
    )
    env = SACEnvironment(kwargs)
    try:
        env.seed(seed)
        obs = env.reset(scenario_id=sid)  # Pins BOTH YEAR and MONTH/DAY.
        model = env.unwrapped
        verify(str(model.current_date.date()) == sid,
               f"reset selected wrong year/day: {model.current_date.date()} != {sid}")
        verify(env.horizon == 117, f"unexpected horizon {env.horizon}")
        verify(obs.shape == (6,) and env.action_space.shape == (2,),
               "compact observation/action shape mismatch")
        near(float(obs[5]), 1.0, "initial tau", atol=1e-6)
        near(float(model.tilt), 30., "fixed initial tilt")
        near(float(model.azimuth), 180., "fixed initial azimuth")

        prev_tilt, prev_azimuth = float(model.tilt), float(model.azimuth)
        prev_timestamp = None
        total_energy = 0.
        total_reward = 0.
        total_cost = 0.
        independent_energy = 0.
        max_poa_error = 0.
        max_manual_poa_error = 0.
        max_power_error = 0.
        max_aoi_cos_error = 0.
        max_ghi_residual = 0.
        first_step = {}
        last_step = {}
        trace = []

        for t in range(env.horizon):
            action = actions_for(t, policy)
            obs, reward, done, info = env.step(action)
            label = f"{sid}/{policy}/step {t+1}"

            verify(info["scenario_id"] == sid, f"{label}: scenario changed: {info['scenario_id']}")
            verify(info["scenario_year"] == int(sid[:4]), f"{label}: source year mismatch")
            verify(str(info["date"]) == sid, f"{label}: date mismatch")
            verify(info["weather_source"] == "nsrdb_multiyear", f"{label}: not NSRDB")
            verify(info["weather_resampling"] == "none_native_5min", f"{label}: wrong grid")
            near(float(info["control_interval_minutes"]), 5., f"{label} 5min", atol=1e-8)
            verify(obs.shape == (6,) and np.isfinite(obs).all(), f"{label}: invalid observation")
            verify(env.observation_space.contains(obs), f"{label}: observation out of bounds")
            verify(done == (t == env.horizon-1), f"{label}: premature/late done")

            timestamp = pd.Timestamp(info["timestamp_utc_iso"])
            verify(timestamp.utcoffset() == timedelta(0), f"{label}: not UTC")
            verify(str(timestamp.date()) == sid, f"{label}: timestamp date mismatch")
            if prev_timestamp is None:
                expected_first = pd.Timestamp(f"{sid} 12:05", tz="UTC")
                verify(timestamp == expected_first, f"{label}: wrong first timestamp {timestamp}")
            else:
                verify(timestamp-prev_timestamp == pd.Timedelta(minutes=5),
                       f"{label}: non-native 5-minute timestep")
            prev_timestamp = timestamp

            # Predict kinematics independent of the environment's next state.
            dtilt = float(action[0]) * float(model.max_delta_tilt)
            dazimuth = float(action[1]) * float(model.max_delta_azimuth)
            predicted_tilt = float(np.clip(prev_tilt+dtilt, 0., 90.))
            predicted_azimuth = (prev_azimuth+dazimuth) % 360.
            near(float(info["tilt"]), predicted_tilt, f"{label} tilt")
            near(float(info["azimuth"]), predicted_azimuth, f"{label} azimuth")
            near(float(model.tilt), predicted_tilt, f"{label} inner tilt")
            near(float(model.azimuth), predicted_azimuth, f"{label} inner azimuth")
            near(float(info["delta_tilt_deg"]), dtilt, f"{label} commanded tilt")
            near(float(info["delta_azimuth_deg"]), dazimuth, f"{label} commanded azimuth")
            # Capture executed tilt BEFORE updating the previous pose.
            # At a hard limit, a nonzero command can execute zero movement.
            executed_dtilt = predicted_tilt - prev_tilt
            prev_tilt, prev_azimuth = predicted_tilt, predicted_azimuth

            # Independent recomputation DIRECTLY from pvlib, not the project's
            # compute_poa_global or compute_panel_power_w helper functions.
            z = float(info["solar_zenith_deg"])
            saz = float(info["solar_azimuth_deg"])
            if t in (0, env.horizon-1):
                # Check that recorded solar metadata corresponds to the UTC clock
                # and location; this is a pvlib self-consistency test, NOT field truth.
                solar = pvlib.solarposition.get_solarposition(
                    pd.DatetimeIndex([timestamp]),
                    latitude=float(model.location.latitude),
                    longitude=float(model.location.longitude),
                    altitude=float(model.location.altitude),
                ).iloc[0]
                near(z, float(solar["zenith"]), f"{label} solar zenith", atol=0.05)
                near(saz, float(solar["azimuth"]), f"{label} solar azimuth", atol=0.05)
            dni = float(info["dni_wm2"])
            ghi = float(info["ghi_wm2"])
            dhi = float(info["dhi_wm2"])
            poa_parts = pvlib.irradiance.get_total_irradiance(
                surface_tilt=predicted_tilt,
                surface_azimuth=predicted_azimuth,
                solar_zenith=z,
                solar_azimuth=saz,
                dni=dni,
                ghi=ghi,
                dhi=dhi,
                albedo=0.25,      # pvlib default, used by the project implementation
                model="isotropic",
            )
            expected_poa = max(0., float(poa_parts["poa_global"]))

            # A genuinely separate closed-form calculation for isotropic-sky
            # transposition, not calling pvlib's irradiance functions at all.
            # POA = DNI*cos(AOI)_+ + DHI*(1+cos(beta))/2
            #     + GHI*albedo*(1-cos(beta))/2; same 0.25 albedo as project.
            b, zrad, diff = map(math.radians,
                                 (predicted_tilt, z, saz-predicted_azimuth))
            cos_beta = math.cos(b)
            cos_inc = (math.cos(zrad)*cos_beta
                       + math.sin(zrad)*math.sin(b)*math.cos(diff))
            manual_poa = max(0.,
                dni*max(0., cos_inc)
                + dhi*(1.+cos_beta)/2.
                + ghi*0.25*(1.-cos_beta)/2.)
            max_manual_poa_error = max(max_manual_poa_error,
                                       abs(float(info["poa_global"])-manual_poa))
            near(float(info["poa_global"]), manual_poa,
                 f"{label} handwritten isotropic POA", atol=1e-5)
            area = float(model.area)
            efficiency = float(model.efficiency)
            expected_power = expected_poa * area * efficiency
            expected_energy = expected_power * (5.0/60.0) / 1000.
            max_poa_error = max(max_poa_error, abs(float(info["poa_global"])-expected_poa))
            max_power_error = max(max_power_error, abs(float(info["power"])-expected_power))
            near(float(info["poa_global"]), expected_poa, f"{label} POA", atol=1e-6)
            near(float(info["power"]), expected_power, f"{label} power W", atol=1e-6)
            near(float(info["energy_kwh"]), expected_energy, f"{label} energy kWh", atol=1e-9)

            # Independent angle-of-incidence calculation.
            aoi_deg = float(pvlib.irradiance.aoi(predicted_tilt, predicted_azimuth, z, saz))
            expected_cos_aoi = float(np.clip(math.cos(math.radians(aoi_deg)), 0., 1.))
            max_aoi_cos_error = max(max_aoi_cos_error,
                                    abs(float(info["cos_aoi"])-expected_cos_aoi))
            near(float(info["cos_aoi"]), expected_cos_aoi,
                 f"{label} cos(AOI)", atol=1e-8)

            # Geometry-mode actuator energy uses ACTUAL executed tilt movement,
            # not requested tilt: this must remain valid when commands saturate.
            if movement_penalty == 0.:
                expected_cost = 0.
            else:
                seconds = (abs(executed_dtilt)/float(model.slew_rate_tilt_deg_s)
                           + abs(dazimuth)/float(model.slew_rate_azimuth_deg_s))
                expected_cost = (float(model.actuator_power_w)*seconds
                                 / 3600. / 1000. * movement_penalty)
            near(float(info["movement_cost"]), expected_cost,
                 f"{label} movement cost", atol=1e-10)
            near(float(reward), expected_energy-expected_cost,
                 f"{label} reward", atol=1e-8)
            near(float(obs[0]), float(np.clip(dni/IRRADIANCE_NORM,0.,1.)),
                 f"{label} DNI observation", atol=1e-6)
            near(float(obs[1]), float(np.clip(dhi/IRRADIANCE_NORM,0.,1.)),
                 f"{label} DHI observation", atol=1e-6)
            near(float(obs[2]), float(np.clip(expected_power/POWER_NORM,0.,1.)),
                 f"{label} power observation", atol=1e-6)
            near(float(obs[3]), predicted_tilt/90., f"{label} tilt observation", atol=1e-6)
            near(float(obs[4]), predicted_azimuth/180.-1.,
                 f"{label} azimuth observation", atol=1e-6)
            near(float(obs[5]), (env.horizon-t-1)/env.horizon,
                 f"{label} time observation", atol=1e-6)

            total_energy += float(info["energy_kwh"])
            total_reward += float(reward)
            total_cost += float(info["movement_cost"])
            independent_energy += expected_energy
            # This is only a diagnostic: NSRDB components are not guaranteed
            # to obey GHI = DNI*cos(zenith)+DHI exactly at every 5-minute sample.
            if z < 85. and ghi > 50.:
                residual = abs(ghi - (dni*max(0., math.cos(math.radians(z)))+dhi))
                max_ghi_residual = max(max_ghi_residual, residual)
            entry = dict(step=t+1, timestamp=timestamp.isoformat(),
                         tilt=predicted_tilt, azimuth=predicted_azimuth,
                         poa=float(info["poa_global"]), power=float(info["power"]),
                         reward=float(reward), weather_condition=str(info["weather_condition"]))
            trace.append(entry)
            if t == 0:
                first_step = entry
            last_step = entry

        near(total_energy, independent_energy, f"{sid}/{policy} cumulative energy", atol=1e-7)
        near(total_reward, total_energy-total_cost, f"{sid}/{policy} cumulative reward", atol=1e-8)
        verify(last_step["timestamp"] == pd.Timestamp(f"{sid} 21:45",tz="UTC").isoformat(),
               f"{sid}/{policy}: incorrect final timestamp")
        if policy == "stationary":
            near(float(last_step["tilt"]), 30., "stationary final tilt")
            near(float(last_step["azimuth"]), 180., "stationary final azimuth")
            near(total_cost, 0., "stationary movement cost")
        # Verify the reports are aggregating different variables over different
        # time windows, rather than treating one late cloud observation as
        # a contradiction of an episode-level summary.
        frame = model.weather_profile
        diagnostics = {}
        if "cloud_type" in frame:
            ct = frame["cloud_type"].dropna()
            if len(ct):
                mode = float(ct.mode().iloc[0])
                near(float(info["episode_cloud_type_mode"]), mode,
                     f"{sid} full-episode cloud type mode")
                diagnostics["episode_cloud_type_mode"] = mode
                diagnostics["last_cloud_type"] = info.get("cloud_type")
        if "clearsky_ghi" in frame:
            denom = float(frame["clearsky_ghi"].astype(float).sum())
            ratio = float(frame["ghi"].astype(float).sum())/denom if denom > 1e-6 else 0.
            near(float(info["episode_clearsky_ratio"]), ratio,
                 f"{sid} full-episode clearsky ratio", atol=1e-8)
            diagnostics["episode_clearsky_ratio"] = ratio
        # Reconstruct the time-local clear/partly_cloudy/overcast labels.
        sky = model.location.get_clearsky(model.times)
        rg = np.clip(frame["ghi"].to_numpy()/sky["ghi"].clip(lower=1.).to_numpy(), 0., 1.5)
        rd = np.clip(frame["dni"].to_numpy()/sky["dni"].clip(lower=1.).to_numpy(), 0., 1.5)
        expected_conditions = np.full(len(frame), "overcast", dtype=object)
        clear = (rg >= .75) & (rd >= .60)
        partial = (~clear) & (rg >= .35)
        expected_conditions[clear] = "clear"
        expected_conditions[partial] = "partly_cloudy"
        if not np.array_equal(expected_conditions, frame["condition"].to_numpy()):
            j = int(np.flatnonzero(expected_conditions != frame["condition"].to_numpy())[0])
            raise AssertionError(f"{sid}: cloud label mismatch at weather row {j}")
        diagnostics["last_weather_condition"] = str(info["weather_condition"])
        diagnostics["counts_by_condition"] = {
            kind: int(np.sum(expected_conditions == kind))
            for kind in ("clear", "partly_cloudy", "overcast")
        }

        # Diagnostic only: the selected window is intentionally shorter than
        # a full day. This counts daylight clock samples outside its bounds;
        # it is NOT a calculation of omitted annual kWh.
        full_utc = pd.date_range(f"{sid} 00:00", periods=288, freq="5min", tz="UTC")
        full_solar = model.location.get_solarposition(full_utc)
        daylight = np.asarray(full_solar["apparent_elevation"] > 0.)
        start = pd.Timestamp(f"{sid} 12:00", tz="UTC")
        end = pd.Timestamp(f"{sid} 21:45", tz="UTC")
        before = np.asarray(full_utc < start)
        after = np.asarray(full_utc > end)
        daylight_window = {
            "daylight_samples_inside_episode_clock": int(np.sum(daylight & ~(before | after))),
            "daylight_samples_before_episode_clock": int(np.sum(daylight & before)),
            "daylight_samples_after_episode_clock": int(np.sum(daylight & after)),
            "note": "UTC calendar-day counts; they may span two local solar days. Not omitted kWh.",
        }

        if policy == "north_slew":
            near(float(last_step["tilt"]), 80., "north final tilt")
            near(float(last_step["azimuth"]), 0., "north final azimuth")
            if movement_penalty > 0.:
                verify(total_cost > 0., "nonzero commanded motion did not cost actuator energy")
        return dict(
            scenario_id=sid, policy=policy, movement_penalty=movement_penalty,
            steps=len(trace), gross_energy_kwh=total_energy,
            movement_cost_kwh=total_cost, reward=total_reward,
            independent_pvlib_gross_kwh=independent_energy,
            maximum_poa_error_wm2=max_poa_error,
            maximum_hand_formula_poa_error_wm2=max_manual_poa_error,
            maximum_power_error_w=max_power_error,
            maximum_cos_aoi_error=max_aoi_cos_error,
            maximum_irradiance_closure_residual_wm2=max_ghi_residual,
            diagnostics=diagnostics, daylight_window=daylight_window,
            first=first_step, last=last_step, trace=trace,
        )
    finally:
        env.close()




def test_reported_july_step_fixture():
    """Independently re-evaluate numeric data in the uploaded July 2019 log."""
    print("=== Uploaded July 2019 step: disputed north-facing pose ===", flush=True)
    tilt, azimuth = 82.12758856959266, 359.084489095519
    solar_z, solar_az = 61.881181801041635, 275.0033340965587
    dni, dhi, ghi = 260., 192., 315.
    z, b, diff = map(math.radians, (solar_z, tilt, solar_az-azimuth))
    cos_aoi = (math.cos(z)*math.cos(b)
               + math.sin(z)*math.sin(b)*math.cos(diff))
    beam = dni*max(0., cos_aoi)
    sky = dhi*(1.+math.cos(b))/2.
    ground = ghi*0.25*(1.-math.cos(b))/2.
    analytic_poa = beam+sky+ground
    direct_poa = float(pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt, surface_azimuth=azimuth,
        solar_zenith=solar_z, solar_azimuth=solar_az,
        dni=dni, dhi=dhi, ghi=ghi,
        albedo=0.25, model="isotropic")['poa_global'])
    near(analytic_poa, 183.3383843515649,
         "uploaded July 2019 POA vs hand formula", atol=1e-7)
    near(direct_poa, analytic_poa,
         "uploaded July 2019 POA vs independent pvlib", atol=1e-7)
    near(analytic_poa*.18, 33.000909183281685,
         "uploaded July 2019 power vs hand formula", atol=1e-7)
    print(f"  PASS: north-facing reported pose -> {analytic_poa:.9f} W/m2, "
          f"{analytic_poa*.18:.9f} W; no one-step orientation mismatch", flush=True)
    return {"reported_poa_wm2": 183.3383843515649,
            "calculated_poa_wm2": analytic_poa,
            "calculated_power_w": analytic_poa*.18,
            "note": "One step only; no assertion about the separate episode-total energy."}

def _current_scenario_id(env: SACEnvironment):
    """Use same scenario-ID accessor the source uses; no inference from date range."""
    from mbpo.env.nsrdb_weather import scenario_id as sid_of
    return sid_of(env.unwrapped._current_scenario)


def test_scenario_selection(cfg, available, anchor: str):
    """Validate explicit selection, fixed-evaluation lists and flag semantics."""
    print("=== Scenario-ID selection and the reported 'dead flag' claim ===", flush=True)
    verify(anchor in available, f"Audit selection anchor {anchor!r} absent from catalog")
    base = dict(cfg.train_env)
    base.update(start_date=anchor, end_date=anchor,
                randomize_initial_orientation=False, randomize_day=False,
                randomize_scenario=False,
                fixed_eval_scenarios=None,
                irradiance_perturbation_std=0., observation_noise_std=0.)
    selection = {}
    source = inspect.getsource(PVTrackingEnv._select_nsrdb_scenario)
    verify("self.randomize_scenario or not self.randomize_day" in source,
           "scenario sampling source changed; review selection semantics again")
    print("  SOURCE VERIFIED: randomize_scenario IS consulted in _select_nsrdb_scenario", flush=True)

    # Fix 1: explicit scenario_id is forwarded by SACEnvironment.reset.
    env = SACEnvironment(base)
    try:
        env.seed(11)
        env.reset(scenario_id=anchor)
        verify(_current_scenario_id(env) == anchor, "explicit scenario_id ignored")
        _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        verify(info["scenario_id"] == anchor, "explicit scenario changed after step")
        selection["explicit_scenario_id"] = info["scenario_id"]
        print(f"  PASS explicit reset(scenario_id={anchor!r}) -> {info['scenario_id']}", flush=True)
    finally:
        env.close()

    # Separate raw API: reset(date=...) is available on PVTrackingEnv, although
    # it is NOT an argument of SACEnvironment.reset. Confirm exact-year match.
    raw = PVTrackingEnv(**base)
    try:
        raw.seed(11)
        raw.reset(date=anchor)
        _, _, _, info = raw.step(np.zeros(2, dtype=np.float32))
        verify(info["scenario_id"] == anchor,
               f"raw reset(date={anchor!r}) picked {info['scenario_id']}")
        selection["raw_date"] = info["scenario_id"]
        print(f"  PASS raw PVTrackingEnv.reset(date={anchor!r}) -> {info['scenario_id']}", flush=True)
    finally:
        raw.close()

    # Fix 2: a singleton fixed_eval_scenarios must select that exact year/day
    # even when the seed changes. No source change is required.
    env = SACEnvironment({**base, "fixed_eval_scenarios": [anchor]})
    try:
        chosen = []
        for seed in (0, 1, 2, 99):
            env.seed(seed)
            env.reset()
            chosen.append(_current_scenario_id(env))
        verify(chosen == [anchor]*len(chosen), f"singleton fixed list failed: {chosen}")
        selection["fixed_eval_singleton"] = chosen
        print(f"  PASS singleton fixed_eval_scenarios over four seeds: {chosen}", flush=True)
    finally:
        env.close()

    # A list of more than one fixed scenario is NOT automatically cycled by
    # repeated env.reset after seeding. The default uses seed modulo list size.
    candidates = list(dict.fromkeys([anchor, *[s for s in sorted(available)
                                                   if s != anchor][:3]]))
    if len(candidates) >= 2:
        env = SACEnvironment({**base, "fixed_eval_scenarios": candidates})
        try:
            env.seed(42)
            env.reset()
            chosen = _current_scenario_id(env)
            expected = candidates[42 % len(candidates)]
            verify(chosen == expected,
                   f"fixed list seed selection: got {chosen}, expected {expected}")
            env.unwrapped.begin_evaluation_rollouts()
            cycled = []
            for _ in candidates:
                env.reset()
                cycled.append(_current_scenario_id(env))
            verify(cycled == candidates, f"evaluation-cycle API selected {cycled}")
            selection["fixed_list_seeded"] = chosen
            selection["fixed_list_cycled"] = cycled
            print(f"  PASS multi-ID list: seed 42 -> {chosen}; explicit eval cycle -> {cycled}", flush=True)
        finally:
            env.close()

    # Crucial: false/false still chooses a scenario, but only within the
    # month/day-filtered catalog. False/true uses a different selection branch.
    # Neither config is a contract to pin the exact YEAR. Do not demand that
    # stochastic trials necessarily produce different years.
    for randomize_day, randomize_scenario in ((False, False), (True, False),
                                               (False, True), (True, True)):
        flag = f"randomize_day={randomize_day},randomize_scenario={randomize_scenario}"
        kwargs = {**base, "randomize_day": randomize_day,
                  "randomize_scenario": randomize_scenario}
        env = None
        try:
            # PVTrackingEnv.__init__ itself calls reset(). A broken sampling
            # branch can therefore fail before SACEnvironment is constructed.
            env = SACEnvironment(kwargs)
            seen = []
            for seed in (10, 11, 12, 13):
                env.seed(seed)
                env.reset()
                sid = _current_scenario_id(env)
                verify(sid in available, f"{flag}: unknown scenario {sid}")
                verify(sid[5:] == anchor[5:],
                       f"{flag}: expected month/day {anchor[5:]}, found {sid}")
                seen.append(sid)
            selection[flag] = {"status": "PASS", "observed_scenarios": seen}
            print(f"  PASS {flag}: {seen}", flush=True)
        except AttributeError as exc:
            # This is a known source bug: gym's seeding.np_random() may yield
            # numpy.random.Generator, which implements integers(), not randint().
            # Record the FAIL, but still run the expensive physics checks below.
            if "randint" not in str(exc) or not (randomize_day and not randomize_scenario):
                raise
            issue = {
                "id": "NSRDB_RANDOM_DAY_GENERATOR_RANDINT",
                "source": "mbpo/env/pv_tracking.py:_select_nsrdb_scenario",
                "status": "FAIL",
                "flags": flag,
                "exception": f"{type(exc).__name__}: {exc}",
                "remedy": "Use a Generator/RandomState-compatible randint/integers helper in the source",
            }
            selection.setdefault("runtime_issues", []).append(issue)
            selection[flag] = {"status": "FAIL", "issue_id": issue["id"]}
            print(f"  CONFIRMED RUNTIME BUG {flag}: {exc}", flush=True)
            print("    Other scenario-selection and physics checks will continue.", flush=True)
        finally:
            if env is not None:
                env.close()
    # Guard against a common scientifically invalid assumption:
    # start_date/end_date do not, by themselves, constrain source year.
    print("  INTERPRETATION: false does not mean deterministic fixed-year;"
          " start/end filter calendar month/day in NSRDB mode.", flush=True)
    return selection


def test_non_nsrdb_random_day_rng():
    """Exercise the separate non-NSRDB random-day branch using clear sky.

    This is a read-only functional test, not a simulation accuracy experiment.
    """
    print("=== Separate clear-sky random-day RNG compatibility check ===", flush=True)
    env = None
    try:
        env = PVTrackingEnv(
            weather_source="clearsky", weather_scenario_mode="clearsky",
            latitude=40.72, longitude=-74.01, tz="UTC",
            start_date="2022-07-15", end_date="2022-07-17",
            start_time="12:00", periods=118, freq="5min",
            randomize_day=True, randomize_initial_orientation=False,
            observation_mode="compact",
        )
        env.seed(43)
        env.reset()
        print(f"  PASS clear-sky random-day reset: {env.current_date.date()}", flush=True)
        return {"status": "PASS", "selected_date": str(env.current_date.date())}
    except AttributeError as exc:
        if "randint" not in str(exc):
            raise
        issue = {
            "id": "NON_NSRDB_RANDOM_DAY_GENERATOR_RANDINT",
            "source": "mbpo/env/pv_tracking.py:reset (randomize_day branch)",
            "status": "FAIL",
            "exception": f"{type(exc).__name__}: {exc}",
            "remedy": "Use Generator/RandomState-compatible integer sampling in reset",
        }
        print(f"  CONFIRMED RUNTIME BUG clear-sky random day: {exc}", flush=True)
        return issue
    finally:
        if env is not None:
            env.close()


def test_geometry_cost_at_saturated_tilt(cfg, anchor):
    """Regression: physical cost uses executed tilt in REAL and MBPO fake env.

    The model's solar-channel deltas are supplied from the actual real next state
    for this narrow contract test; it does NOT assess learned-model accuracy.
    """
    print("=== Real/MBPO-fake actuator boundary parity (geometry mode) ===", flush=True)
    from MBPO_workspace.src.dynamics import StructuredFakeEnv

    kwargs = dict(cfg.train_env)
    kwargs.update(randomize_initial_orientation=False,
                  irradiance_perturbation_std=0., observation_noise_std=0.,
                  movement_penalty=1., movement_cost_mode="geometry")
    env = SACEnvironment(kwargs)

    class KnownSolarDelta:
        def __init__(self, real_before, real_after):
            self.delta = np.asarray(real_after[:3] - real_before[:3], dtype=np.float32)

        def predict_delta(self, states, actions):
            verify(states.shape == (1, 6), "expected one state")
            verify(actions.shape == (1, 2), "expected one action")
            return self.delta[None, :], np.zeros(1, dtype=np.float32)

    def verify_fake_parity(before, after, action, reward, done, label):
        oracle = KnownSolarDelta(before, after)
        fake = StructuredFakeEnv(oracle, env)
        future, fake_rewards, fake_dones, uncertainty, mask = fake.step(
            np.asarray(before, dtype=np.float32)[None, :],
            np.asarray(action, dtype=np.float32)[None, :])
        verify(bool(mask[0]), f"{label}: fake transition was rejected")
        verify(bool(fake_dones[0]) == bool(done), f"{label}: wrong terminal signal")
        verify(np.allclose(future[0], after, atol=3e-6, rtol=0),
               f"{label}: real and fake next states differ: {future[0]} vs {after}")
        near(float(fake_rewards[0]), float(reward),
             f"{label}: real/fake reward disagreement", atol=2e-7, rtol=0)
        return float(fake_rewards[0])

    try:
        env.seed(7)
        obs = env.reset(scenario_id=anchor)
        move_up = np.array([1., 0.], dtype=np.float32)
        # 30 -> 90 degrees after 12 positive 5-degree movements.
        for _ in range(12):
            obs, _, _, info = env.step(move_up)
        near(float(info["tilt"]), 90., "expected upper tilt boundary")

        before = obs.copy()
        after, reward, done, info = env.step(move_up)
        upper_cost = float(info["movement_cost"])
        near(float(info["tilt"]), 90., "clipped upper tilt")
        near(float(info["delta_tilt_deg"]), 5., "requested upward tilt")
        near(upper_cost, 0., "no physical movement means no geometry energy", atol=1e-12)
        upper_fake_reward = verify_fake_parity(before, after, move_up, reward, done,
                                               "upper-tilt saturation")
        print("  PASS upper limit: requested +5deg, executed 0deg, "
              "real/fake cost=0", flush=True)

        # Move off the boundary with both tilt and azimuth movement.
        move_down_az = np.array([-1., 1.], dtype=np.float32)
        before = after.copy()
        after, reward, done, info = env.step(move_down_az)
        near(float(info["tilt"]), 85., "negative tilt from upper limit")
        expected_cost = (float(env.unwrapped.actuator_power_w) *
                         (5. / float(env.unwrapped.slew_rate_tilt_deg_s) +
                          10. / float(env.unwrapped.slew_rate_azimuth_deg_s)) /
                         3.6e6)
        near(float(info["movement_cost"]), expected_cost,
             "executed movement energy", atol=1e-12)
        verify_fake_parity(before, after, move_down_az, reward, done,
                           "tilt and azimuth movement")
        print(f"  PASS interior movement: real/fake actuator cost "
              f"{expected_cost:.12f} kWh", flush=True)

        move_down = np.array([-1., 0.], dtype=np.float32)
        # 85 -> 0 over 17 more steps.
        for _ in range(17):
            after, reward, done, info = env.step(move_down)
        near(float(info["tilt"]), 0., "expected lower tilt boundary")
        before = after.copy()
        after, reward, done, info = env.step(move_down)
        lower_cost = float(info["movement_cost"])
        near(float(info["tilt"]), 0., "clipped lower tilt")
        near(lower_cost, 0., "no motion below tilt limit means no cost", atol=1e-12)
        verify_fake_parity(before, after, move_down, reward, done,
                           "lower-tilt saturation")
        print("  PASS lower limit: requested -5deg, executed 0deg, "
              "real/fake cost=0", flush=True)

        return {
            "status": "PASS",
            "upper_limit_executed_movement_cost_kwh": upper_cost,
            "lower_limit_executed_movement_cost_kwh": lower_cost,
            "interior_movement_cost_kwh": float(expected_cost),
            "upper_limit_fake_reward": upper_fake_reward,
            "real_fake_contract": "known solar delta used to isolate kinematics/reward; learned model not assessed",
        }
    finally:
        env.close()


def test_random_initial_orientation_determinism(cfg, anchor):
    print("=== Random initial orientation: repeatability given seed and scenario ===", flush=True)
    base = dict(cfg.train_env)
    base.update(randomize_initial_orientation=True,
                irradiance_perturbation_std=0., observation_noise_std=0.,
                fixed_eval_scenarios=None)
    def sample(seed):
        env = SACEnvironment(base)
        try:
            env.seed(seed)
            initial = env.reset(scenario_id=anchor)
            pose = (float(env.unwrapped.tilt), float(env.unwrapped.azimuth))
            data = [tuple(float(v) for v in initial)]
            for action in ((0., 0.), (0.5, -0.5), (-1., 1.)):
                state, rew, done, info = env.step(np.asarray(action, dtype=np.float32))
                data.append((tuple(float(v) for v in state), float(rew),
                             info["scenario_id"], info["timestamp_utc_iso"]))
            return pose, data
        finally:
            env.close()
    one = sample(12345)
    two = sample(12345)
    other = sample(12346)
    verify(one == two, "same scenario and seed did not reproduce orientation/short trace")
    verify(one[0] != other[0], "different seeds unexpectedly produced identical pose")
    print(f"  PASS seed 12345 pose {one[0]}; repeat identical. Seed 12346 pose {other[0]}", flush=True)
    return {"same_seed_pose": one[0], "different_seed_pose": other[0]}


def audit_physics_scope(cfg):
    print("=== Code defaults versus experiment overrides; physics scope ===", flush=True)
    constructor = inspect.signature(PVTrackingEnv.__init__)
    default_penalty = constructor.parameters["movement_penalty"].default
    config_penalty = float(cfg.train_env["movement_penalty"])
    verify(np.isclose(default_penalty, .01),
           f"expected baseline default 0.01 for this branch, found {default_penalty}")
    verify(np.isclose(config_penalty, 0.),
           f"expected benchmark override 0.0, found {config_penalty}")
    p_args = list(inspect.signature(compute_panel_power_w).parameters)
    print(f"  ENV default movement_penalty={default_penalty}; benchmark override={config_penalty}", flush=True)
    print(f"  Implemented power formula inputs={p_args}", flush=True)
    print("  LIMITATION: This fixed-efficiency power function takes no temperature,"
          " wind-speed, IAM, inverter or module-datasheet inputs.", flush=True)
    print("  LIMITATION: This does not establish that 30deg/180deg is optimal."
          " It is merely the source code's deterministic initial pose.", flush=True)
    return {"code_default_movement_penalty": default_penalty,
            "experiment_movement_penalty": config_penalty,
            "power_function_parameters": p_args}


def test_rng_api_contract():
    """Both NumPy Generator and RandomState must support scalar integer draws."""
    print("=== NumPy RNG backward/forward compatibility ===", flush=True)
    from mbpo.env.pv_tracking import _rng_index
    for rng in (np.random.default_rng(314159), np.random.RandomState(314159)):
        draws = [_rng_index(rng, 7) for _ in range(50)]
        verify(all(isinstance(x, int) and 0 <= x < 7 for x in draws),
               f"invalid {type(rng).__name__} index results")
        verify(len(set(draws)) > 1, f"{type(rng).__name__} RNG appears constant")
        print(f"  PASS {type(rng).__name__}: sampled 50 valid indices", flush=True)
    return "PASS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS),
                        help="Exact IDs from NSRDB manifest, e.g. 2022-07-15")
    parser.add_argument("--json", type=Path, default=None,
                        help="Optional path for structured results and full step traces")
    args = parser.parse_args()

    print("=== Code and dependency checks ===", flush=True)
    for name in EVALUATION_MODULES:
        module = importlib.import_module(name)
        print(f"  IMPORT OK {module.__name__}", flush=True)
    cfg = load_config(require_data=True)
    manifest = load_scenario_manifest(cfg.manifest)
    available = {scenario_id(row) for row in manifest["scenarios"]}
    verify(len(args.scenarios) == len(set(args.scenarios)), "duplicate requested scenarios")
    missing = sorted(set(args.scenarios)-available)
    verify(not missing, f"IDs missing from manifest: {missing}; use --scenarios with valid IDs")
    manifest_sha256 = hashlib.sha256(Path(cfg.manifest).read_bytes()).hexdigest()
    print(f"  Manifest: {cfg.manifest} ({len(available)} unique scenarios)", flush=True)
    print(f"  Manifest SHA256: {manifest_sha256}", flush=True)
    rng_api = test_rng_api_contract()
    july_fixture = test_reported_july_step_fixture()
    audit_scope = audit_physics_scope(cfg)
    anchor = "2022-07-15" if "2022-07-15" in available else args.scenarios[0]
    selection = test_scenario_selection(cfg, available, anchor)
    clear_sky_rng = test_non_nsrdb_random_day_rng()
    random_initial = test_random_initial_orientation_determinism(cfg, anchor)
    print("=== Pinned seasonal rollouts (fixed 30deg/180deg pose) ===", flush=True)

    results = []
    for sid in args.scenarios:
        r = one_episode(cfg, sid, policy="stationary", movement_penalty=0.)
        results.append(r)
        print(f"  PASS {sid} 117/117  E={r['gross_energy_kwh']:.6f} kWh"
              f"  max POA err={r['maximum_poa_error_wm2']:.3g} W/m2"
              f"  manual formula err={r['maximum_hand_formula_poa_error_wm2']:.3g} W/m2", flush=True)
        d = r['daylight_window']
        print(f"    Sunlit 5-min samples outside UTC window:"
              f" before={d['daylight_samples_before_episode_clock']},"
              f" after={d['daylight_samples_after_episode_clock']}"
              f" (not an energy-loss estimate)", flush=True)

    chosen_sid = "2022-07-15" if "2022-07-15" in available else args.scenarios[0]
    print("=== Reproducibility check (same scenario, seed, orientation and policy) ===", flush=True)
    one = one_episode(cfg, chosen_sid, policy="stationary", movement_penalty=0.)
    two = one_episode(cfg, chosen_sid, policy="stationary", movement_penalty=0.)
    near(one["gross_energy_kwh"], two["gross_energy_kwh"],
         "repeat episode energy", atol=1e-12, rtol=0.)
    verify(one["trace"] == two["trace"], "step-by-step traces changed across identical reruns")
    print(f"  PASS repeat {chosen_sid}: all 117 step traces identical", flush=True)

    print("=== North-facing orientation reached by actual slew-limited actions ===", flush=True)
    north_free = one_episode(cfg, chosen_sid, policy="north_slew", movement_penalty=0.)
    north = one_episode(cfg, chosen_sid, policy="north_slew", movement_penalty=1.)
    near(north["gross_energy_kwh"], north_free["gross_energy_kwh"],
         "same-actions gross energy independent of actuator cost", atol=1e-12, rtol=0.)
    verify([ (t["poa"], t["power"], t["tilt"], t["azimuth"])
             for t in north["trace"] ] ==
           [ (t["poa"], t["power"], t["tilt"], t["azimuth"])
             for t in north_free["trace"] ],
           "enabling actuator cost changed weather/geometry/POA/power")
    near(north_free["movement_cost_kwh"], 0., "zero movement penalty")
    verify(north["movement_cost_kwh"] > 0., "geometry-mode cost absent")
    print("  PASS paired identical actions, zero-vs-one movement penalty:"
          " power and POA traces unchanged, only reward/cost differ", flush=True)
    results.extend((north_free, north))
    print(f"  PASS {chosen_sid} north-facing end pose "
          f"({north['last']['tilt']:.1f}deg, {north['last']['azimuth']:.1f}deg), "
          f"gross={north['gross_energy_kwh']:.6f} kWh, "
          f"cost={north['movement_cost_kwh']:.9f} kWh", flush=True)

    print("=== Full 117-step saturated-policy regression (nonzero movement penalty) ===", flush=True)
    upper = one_episode(cfg, chosen_sid, policy="tilt_upper_saturate", movement_penalty=1.)
    lower = one_episode(cfg, chosen_sid, policy="tilt_lower_saturate", movement_penalty=1.)
    near(float(upper["last"]["tilt"]), 90., "full-episode saturated upper tilt")
    near(float(lower["last"]["tilt"]), 0., "full-episode saturated lower tilt")
    # Exactly 60deg of upward travel from 30->90, or 30deg downward to 0;
    # repeated commands after reaching the limit execute zero tilt movement.
    # This validates the entire episode's cost, independently of the simulator.
    actuator_power = float(cfg.train_env.get("actuator_power_w", 30.))
    tilt_rate = float(cfg.train_env.get("slew_rate_tilt_deg_s", 1.5))
    expected_up = actuator_power*(60./tilt_rate)/3.6e6
    expected_down = actuator_power*(30./tilt_rate)/3.6e6
    near(float(upper["movement_cost_kwh"]), expected_up,
         "upper saturation total cost", atol=2e-10)
    near(float(lower["movement_cost_kwh"]), expected_down,
         "lower saturation total cost", atol=2e-10)
    results.extend((upper, lower))
    print(f"  PASS upper saturation: 117 steps, total cost={expected_up:.9f} kWh", flush=True)
    print(f"  PASS lower saturation: 117 steps, total cost={expected_down:.9f} kWh", flush=True)

    actuator_boundary = test_geometry_cost_at_saturated_tilt(cfg, chosen_sid)
    runtime_issues = list(selection.get("runtime_issues", []))
    if clear_sky_rng.get("status") == "FAIL":
        runtime_issues.append(clear_sky_rng)
    overall_status = "FAILED_CONFIRMED_RUNTIME_BUGS" if runtime_issues else "PASSED"
    payload = dict(
        status=overall_status, scenarios=args.scenarios, rng_api=rng_api,
        confirmed_runtime_issues=runtime_issues,
        scenario_selection=selection, non_nsrdb_random_day_rng=clear_sky_rng,
        actuator_boundary_regression=actuator_boundary,
        random_initial_orientation=random_initial,
        physics_scope=audit_scope, reported_july_fixture=july_fixture,
        manifest=str(cfg.manifest),
        manifest_sha256=manifest_sha256,
        pvlib_version=pvlib.__version__, numpy_version=np.__version__,
        note="Checks scenario selection, hand-derived isotropic POA and pvlib parity, not field accuracy or trained checkpoints.",
        results=results,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        print(f"  JSON trace: {args.json}", flush=True)
    if runtime_issues:
        print(f"\nAUDIT COMPLETED: {len(runtime_issues)} CONFIRMED RUNTIME BUG(S); "
              "OTHER COMPLETED CHECKS PASSED.", flush=True)
        for issue in runtime_issues:
            print(f"  FAIL {issue['id']}: {issue['exception']}", flush=True)
    else:
        print("\nALL EXTENDED TASK-0 V4.1 RUNTIME CHECKS PASSED", flush=True)
    print("LIMITATION: Physical field accuracy, annual-energy impact and trained-policy evaluation were NOT tested.")
    return 2 if runtime_issues else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nAUDIT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
