#!/usr/bin/env python3
"""Read-only Task-0 integration audit, complementary to pv_task0_strong_audit_v4_1.py.

Run from the root of the *local* mbpo_custom checkout, with the project's
Python environment active:

  python /path/to/pv_task0_integration_audit_v5.py
  python /path/to/pv_task0_integration_audit_v5.py --scenario 2022-07-15 \
      --sac-checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
      --mbpo-checkpoint MBPO_workspace/runs/mbpo_seed0/latest.pt \
      --json /tmp/pv_task0_integration.json

Tests: code provenance, local import paths, source NSRDB CSV parity, both real
trainer environment adapters with pvlib call spies, saved real/model replay
history, and an actual *paired policy checkpoint evaluation* against the same
PVTrackingEnv. It does NOT train new policies, validate NN model calibration,
or establish field/annual accuracy or out-of-sample statistical significance.

No tracked files, simulator code, checkpoints or run histories are changed.
Only temporary directories are used for trainer instantiation. JSON is written
ONLY when --json is explicitly supplied. Trusted local PyTorch checkpoints
only: the existing evaluation code uses torch.load(weights_only=False).

Exit codes: 0 = all required runtime tests passed; 1 = test failed;
            2 = incomplete (for example: missing checkpoint/skipped eval).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from unittest import mock

import numpy as np

EXPECTED_REMOTE_BRANCH_SHA = "eba24fd6551e3ab9c04699402acfd62c59540d36"
EXPECTED_MODULES = (
    "SAC_workspace.src.config.config",
    "SAC_workspace.src.environment.environment",
    "SAC_workspace.src.training.trainer",
    "SAC_workspace.src.evaluation.evaluation",
    "SAC_workspace.src.evaluation.scenario_sampling",
    "SAC_workspace.src.evaluation.statistical_inference",
    "MBPO_workspace.src.trainer",
    "MBPO_workspace.src.dynamics",
    "MBPO_workspace.src.evaluation",
    "mbpo.env.pv_tracking",
    "mbpo.env.pvlib_physics",
    "mbpo.env.nsrdb_weather",
)


def require(cond, message):
    if not cond:
        raise AssertionError(message)


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_git(*args):
    p = subprocess.run(["git", *args], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, timeout=20)
    if p.returncode:
        return None
    return p.stdout.strip()


class Audit:
    def __init__(self, root: Path):
        self.root = root
        self.records = []

    def record(self, name, status, detail=None):
        rec = {"check": name, "status": status, "detail": detail or {}}
        self.records.append(rec)
        print(f"[{status}] {name}", flush=True)
        if detail and status != "PASS":
            print("  ", str(detail)[:800], flush=True)
        return rec

    def run(self, name, func):
        try:
            details = func()
            return self.record(name, "PASS", details)
        except Exception as e:
            self.record(name, "FAIL", {"exception": f"{type(e).__name__}: {e}",
                                       "traceback": traceback.format_exc(limit=8)})
            return None

    def incomplete(self, name, reason):
        self.record(name, "INCOMPLETE", {"reason": reason})

    def summary(self):
        result = "FAIL" if any(x["status"] == "FAIL" for x in self.records) else \
                 "INCOMPLETE" if any(x["status"] == "INCOMPLETE" for x in self.records) else "PASS"
        return {"status": result, "repository_root": str(self.root),
                "tests": self.records,
                "limitations": [
                    "No model training or neural-dynamics calibration performed.",
                    "One or a few days of policy evaluation do not establish annual performance.",
                    "Matching stored weather does not independently prove NSRDB meteorological accuracy.",
                    "No claims of out-of-sample generalization without precommitted disjoint scenario splits.",
                ]}


def code_provenance(root):
    sha = repo_git("rev-parse", "HEAD")
    branch = repo_git("branch", "--show-current")
    changes = repo_git("status", "--short", "--untracked-files=all")
    assert sha is not None, "Cannot establish current Git revision. Run inside a Git checkout."
    mods = [] if not changes else changes.splitlines()
    return {"head_sha": sha, "branch": branch,
            "published_branch_sha_at_audit_creation": EXPECTED_REMOTE_BRANCH_SHA,
            "matches_published_revision": sha == EXPECTED_REMOTE_BRANCH_SHA,
            "dirty_worktree": bool(mods), "modified_or_untracked": mods[:100],
            "note": "A PASS here records local provenance. Dirty status or a different HEAD does not validate the published branch."}


def import_origin_check(root):
    modules = {}
    for dotted in EXPECTED_MODULES:
        m = importlib.import_module(dotted)
        source = inspect.getfile(m)
        src = Path(source).resolve()
        require(src.is_relative_to(root),
                f"{dotted} imported from OUTSIDE repository: {src}")
        modules[dotted] = str(src.relative_to(root))
    return {"resolved_imports": modules}


def effective_config_and_weather(root, scenario):
    from SAC_workspace.src.config.config import load_config
    from mbpo.env.nsrdb_weather import load_scenario_manifest, load_nsrdb_year_csv
    from SAC_workspace.src.environment.environment import SACEnvironment
    cfg = load_config(require_data=True)
    manifest = load_scenario_manifest(cfg.manifest)
    rows = manifest["scenarios"]
    identifiers = [s["scenario_id"] for s in rows]
    require(len(identifiers) == len(set(identifiers)), "Duplicate NSRDB scenario IDs")
    lookup = {r["scenario_id"]: r for r in rows}
    require(scenario in lookup, f"Scenario {scenario} not found in NSRDB manifest")
    require(cfg.train_env["weather_source"] == cfg.eval_env["weather_source"] == "nsrdb_multiyear",
            "Training/evaluation selected something other than NSRDB")
    require(cfg.train_env["observation_mode"] == cfg.eval_env["observation_mode"] == "compact",
            "Training/evaluation state representations differ")
    require(cfg.train_env["freq"] == cfg.eval_env["freq"] == "5min",
            "Training/evaluation intervals differ")
    dataset_file = Path(lookup[scenario]["_year_path"]).resolve()
    require(dataset_file.is_file(), f"Referenced NSRDB year file does not exist: {dataset_file}")
    data = load_nsrdb_year_csv(dataset_file)
    env = SACEnvironment(cfg.train_env)
    try:
        env.seed(713)
        env.reset(scenario_id=scenario)
        frame = env.unwrapped.weather_profile
        require(str(frame.index.tz) == "UTC", "Environment weather is not UTC")
        require(len(frame) == 118, f"Expected 118 weather samples, got {len(frame)}")
        require(len(data.index.intersection(frame.index)) == len(frame),
                "Source NSRDB file lacks exact episode timestamps")
        discrepancies = {}
        for channel in ("dni", "dhi", "ghi", "temperature", "wind_speed"):
            values = data.loc[frame.index, channel].to_numpy(dtype=float)
            used = frame[channel].to_numpy(dtype=float)
            require(np.all(np.isfinite(values)) and np.all(np.isfinite(used)),
                    f"{channel}: nonfinite source/episode sample")
            err = float(np.max(np.abs(values - used)))
            discrepancies[channel] = err
            require(np.allclose(used, values, rtol=0, atol=1e-9),
                    f"{channel} differs from NSRDB source year data: max deviation={err}")
    finally:
        env.close()
    return {"scenario_count": len(rows),
            "distinct_years": sorted({int(r["source_year"]) for r in rows}),
            "selected_scenario": scenario,
            "manifest_sha256": sha256_file(Path(cfg.manifest)),
            "source_year_file": str(dataset_file),
            "source_year_sha256": sha256_file(dataset_file),
            "weather_csv_max_difference": discrepancies,
            "caveat": "Same reader, independent file load; checks data plumbing, not weather measurement correctness."}


def real_training_path_test(scenario):
    """Instantiate actual trainers but do NOT call training or save to user runs."""
    from SAC_workspace.src.config.config import load_config as load_sac
    from MBPO_workspace.src.config import load_config as load_mbpo
    from SAC_workspace.src.training.trainer import SACTrainer
    from MBPO_workspace.src.trainer import MBPOTrainer
    from SAC_workspace.src.agents.sac_agent import SACAgent
    from mbpo.env.pv_tracking import PVTrackingEnv
    from mbpo.env import pv_tracking as tracking, pvlib_physics as physics

    scfg = load_sac(require_data=True)
    mcfg = load_mbpo(require_data=True)
    require(scfg.to_dict() == mcfg.sac.to_dict(), "Shared SAC/MBPO MDP and optimizer configuration differ")
    counts = {"pvtracking_power": 0, "pvlib_irradiance": 0}
    orig_power = tracking.compute_panel_power_w
    orig_irradiance = physics.get_total_irradiance

    def power_probe(*args, **kwargs):
        counts["pvtracking_power"] += 1
        return orig_power(*args, **kwargs)

    def irradiance_probe(*args, **kwargs):
        counts["pvlib_irradiance"] += 1
        return orig_irradiance(*args, **kwargs)

    with tempfile.TemporaryDirectory(prefix="pv_task0_train_probe_") as tmp:
        sac = None
        mbpo = None
        try:
            # Current trainer constructors produce only configs in temporary directories.
            sac = SACTrainer(scfg, Path(tmp) / "sac", seed=713, device="cpu")
            mbpo = MBPOTrainer(mcfg, Path(tmp) / "mbpo", seed=713, device="cpu")
            require(type(sac.env) is type(mbpo.env), "Different environment adapter classes")
            require(isinstance(sac.env.unwrapped, PVTrackingEnv) and
                    isinstance(mbpo.env.unwrapped, PVTrackingEnv),
                    "A trainer is not using original PVTrackingEnv")
            require(type(sac.agent) is type(mbpo.agent) is SACAgent,
                    "SAC and MBPO trainers do not share the same SACAgent class")
            require(mbpo.fake_env.ensemble is mbpo.dynamics,
                    "MBPO dynamics ensemble is not connected to fake environment")
            states = []
            with mock.patch.object(tracking, "compute_panel_power_w", power_probe), \
                 mock.patch.object(physics, "get_total_irradiance", irradiance_probe):
                for name, trainer in (("sac", sac), ("mbpo", mbpo)):
                    e = trainer.env
                    e.seed(970)
                    initial = e.reset(scenario_id=scenario)
                    path = []
                    for i, a in enumerate(((0.35, -0.40), (-0.80, 0.20), (0.0, 0.0))):
                        next_state, reward, done, info = e.step(np.asarray(a, np.float32))
                        require(info["weather_source"] == "nsrdb_multiyear", f"{name}: wrong weather source")
                        require(info["scenario_id"] == scenario, f"{name}: wrong scenario")
                        path.append((next_state.copy(), float(reward), bool(done),
                                     str(info["timestamp_utc_iso"]), float(info["power"]),
                                     float(info["dni_wm2"]), float(info["dhi_wm2"]), float(info["ghi_wm2"])))
                    states.append((initial.copy(), path))
                # Exercise the ACTUAL train_episode method, not just trainer.env.step.
                # The default 8,000-step warm-up ensures no gradient/model fitting
                # on these two short episodes; saved histories cover later stages.
                sac_result = sac.train_episode()
                mbpo_result = mbpo.train_episode()
                require(int(sac_result["env_steps"]) == 117 and
                        int(mbpo_result["env_steps"]) == 117,
                        "One actual training episode did not execute 117 real steps")
                require(int(sac_result["updates"]) == 0 and
                        int(mbpo_result["updates"]) == 0,
                        "This warm-up test unexpectedly started optimizer updates")
            require(counts["pvtracking_power"] >= 242 and
                    counts["pvlib_irradiance"] >= counts["pvtracking_power"],
                    f"Real trainer steps did not reach canonical pvlib physics: {counts}")
            require(np.allclose(states[0][0], states[1][0], atol=1e-7),
                    "Identical seed and scenario yielded different initial observations")
            for i, (a, b) in enumerate(zip(states[0][1], states[1][1])):
                require(np.allclose(a[0], b[0], atol=1e-7) and
                        np.isclose(a[1], b[1], atol=1e-9) and a[2:] == b[2:],
                        f"SAC/MBPO real transitions differ at sample {i+1}")
        finally:
            if sac is not None:
                sac.env.close()
            if mbpo is not None:
                mbpo.env.close()
    return {"manual_paired_steps_per_trainer": 3,
            "actual_training_episode_steps_per_trainer": 117,
            "tracked_function_calls": counts,
            "same_environment_type": True, "same_sac_agent_class": True,
            "note": "Actual train_episode() used with 8,000-step warmup; no optimizer/model fitting on test episodes."}


def history_model_usage_check(root):
    sac_path = root / "SAC_workspace/runs/sac_seed0/episodes.csv"
    mbpo_path = root / "MBPO_workspace/runs/mbpo_seed0/episodes.csv"
    for path in (sac_path, mbpo_path):
        require(path.is_file(), f"Missing committed training log: {path}")
    with sac_path.open(newline="") as f:
        sac = list(csv.DictReader(f))
    with mbpo_path.open(newline="") as f:
        mbpo = list(csv.DictReader(f))
    require(bool(sac) and bool(mbpo), "Empty training histories")
    for name, rows in (("SAC", sac), ("MBPO", mbpo)):
        steps = [int(r["env_steps"]) for r in rows]
        require(all(v == 117 * i for i, v in enumerate(steps, start=1)),
                f"{name}: training history is not a 117-step-per-episode sequence")
    fits = int(mbpo[-1]["model_fits"])
    generated = sum(int(r["synthetic_generated"]) for r in mbpo)
    used = [float(r["mean_model_batch"]) for r in mbpo if
            r.get("mean_model_batch", "nan").lower() not in ("", "nan", "none")]
    mixed = sum(1 for value in used if value > 0)
    require(fits > 0 and generated > 0 and mixed > 0,
            "MBPO history does not substantiate ensemble fitting, synthetic generation AND replay mixing")
    return {"sac_episodes": len(sac), "mbpo_episodes": len(mbpo),
            "mbpo_model_fits": fits, "mbpo_synthetic_generated_total": generated,
            "mbpo_epochs_with_synthetic_minibatches": mixed,
            "caveat": "Training log metadata alone does not validate neural model predictions or attribution."}



def saved_checkpoint_replay_test(root, checkpoint):
    """Inspect trusted saved MBPO training state, not only its log summaries."""
    import torch
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = ("schema", "epoch", "env_steps", "dynamics", "real_pool", "model_pool", "agent")
    require(all(k in ck for k in required), "MBPO checkpoint lacks training/replay fields")
    require(ck["schema"] == 1, "Unexpected checkpoint format")
    d, real, fake = ck["dynamics"], ck["real_pool"], ck["model_pool"]
    require(int(d["fit_count"]) > 0 and bool(d["fitted"]),
            "Saved MBPO dynamics model was never fitted")
    require(int(d["num_models"]) >= int(d["num_elites"]) >= 2,
            "MBPO ensemble metadata is invalid")
    nr, nf = int(real["count"]), int(fake["count"])
    require(nr > 0 and nf >= 0, "Invalid real/model replay counts")
    for name, buffer, n in (("real", real, nr), ("model", fake, nf)):
        for k, dim in (("states", 6), ("next_states", 6), ("actions", 2)):
            a = np.asarray(buffer[k])
            require(a.shape == (n, dim) and np.isfinite(a).all(),
                    f"{name} replay has corrupt {k}: {a.shape}")
        rewards = np.asarray(buffer["rewards"])
        dones = np.asarray(buffer["dones"])
        require(rewards.shape == dones.shape == (n, 1) and
                np.isfinite(rewards).all() and np.isfinite(dones).all(),
                f"{name} replay has invalid rewards/dones")
    require(len(real["scenario_ids"]) == nr and bool(real["scenario_names"]),
            "Missing per-scenario provenance for real replay")
    if nf:
        st = np.asarray(fake["states"], dtype=float)
        nxt = np.asarray(fake["next_states"], dtype=float)
        require(np.allclose(st[:, 5] - nxt[:, 5], 1. / 117., atol=2e-6, rtol=0),
                "Synthetic replay violates remaining-horizon transition")
        require(np.all(np.abs(np.asarray(fake["actions"])) <= 1.00001),
                "Synthetic replay has out-of-bounds actions")
    csv_path = root / "MBPO_workspace/runs/mbpo_seed0/episodes.csv"
    if checkpoint.resolve() == (root / "MBPO_workspace/runs/mbpo_seed0/latest.pt").resolve():
        require(csv_path.is_file(), "Matching MBPO episode log unavailable")
        with csv_path.open(newline="") as f:
            last = list(csv.DictReader(f))[-1]
        require(int(last["epoch"]) == int(ck["epoch"]) and
                int(last["env_steps"]) == int(ck["env_steps"]) and
                int(last["model_replay_size"]) == nf,
                "MBPO checkpoint replay count/episode differs from logged training history")
    return {"trained_ensemble_fits": int(d["fit_count"]),
            "ensemble_members": int(d["num_models"]),
            "elite_members": int(d["num_elites"]),
            "real_replay_size": nr, "synthetic_replay_size_at_checkpoint": nf,
            "caveat": "Valid saved replay data is not a test of predictive model skill."}


def paired_checkpoint_test(root, scenario, mbpo_checkpoint, sac_checkpoint):
    from MBPO_workspace.src.config import load_config
    from MBPO_workspace.src.evaluation import evaluate
    from MBPO_workspace.src.dynamics import StructuredFakeEnv
    from mbpo.env import pv_tracking as tracking, pvlib_physics as physics
    from pvlib.irradiance import get_total_irradiance as independent_poa

    mcfg = load_config(require_data=True)
    observed = {"real_step_count": 0, "pvlib_irradiance_count": 0}
    orig_step = tracking.PVTrackingEnv.step
    orig_poa = physics.get_total_irradiance

    def wrapped_step(self, action):
        observed["real_step_count"] += 1
        return orig_step(self, action)

    def wrapped_poa(*a, **kw):
        observed["pvlib_irradiance_count"] += 1
        return orig_poa(*a, **kw)

    def no_fake(*a, **kw):
        raise RuntimeError("Synthetic model was called during physical-policy evaluation")

    with mock.patch.object(tracking.PVTrackingEnv, "step", wrapped_step), \
         mock.patch.object(physics, "get_total_irradiance", wrapped_poa), \
         mock.patch.object(StructuredFakeEnv, "step", no_fake):
        paths = evaluate(mcfg, mbpo_checkpoint, sac_checkpoint,
                         scenarios=[scenario], seeds=(42,), device="cpu",
                         include_poa=True, include_classical=False)

    by = {p["method"]: p for p in paths}
    require(len(paths) == 4 and set(by) == {"mbpo", "sac", "fixed_no_motion", "poa_instant_ideal"},
            f"Unexpected evaluation controllers: {list(by)}")
    require(observed["real_step_count"] == 351,
            f"Expected exactly 117 real simulator steps for each of 3 physical policies; got {observed}")
    require(observed["pvlib_irradiance_count"] >= 351,
            f"Evaluation did not use pvlib POA computation for real steps: {observed}")

    ref = by["fixed_no_motion"]
    keys = ("dni_wm2", "dhi_wm2", "ghi_wm2", "solar_zenith_deg", "solar_azimuth_deg")
    times = [info["timestamp_utc_iso"] for info in ref["infos"]]
    require(len(times) == 117, "Fixed-mount reference has incomplete clock")
    checks = {"individual_pvlib_power_checks": 0, "poa_upper_bound_checks": 0}
    for name, p in by.items():
        require(len(p["infos"]) == len(p["power"]) == len(p["energy"]) == 117,
                f"{name}: not exactly 117 evaluation samples")
        require(p["scenario_id"] == scenario and p["seed"] == 42,
                f"{name}: wrong scenario or seed")
        require(np.allclose(p["initial_pose"], ref["initial_pose"], atol=1e-7),
                f"{name}: inconsistent initial panel pose")
        require([i["timestamp_utc_iso"] for i in p["infos"]] == times,
                f"{name}: different temporal grid")
        for j in range(117):
            info = p["infos"][j]
            f = ref["infos"][j]
            require(all(np.isclose(float(info[k]), float(f[k]), atol=1e-8, rtol=0)
                        for k in keys),
                    f"{name}: unequal physical weather/solar inputs at step {j+1}")
            power = float(p["power"][j])
            require(np.isclose(float(p["energy"][j]), power / 12000., atol=2e-8, rtol=0),
                    f"{name}: 5-minute power-to-energy discrepancy at step {j+1}")
            if name != "poa_instant_ideal":
                if "power" in info:
                    require(np.isclose(float(info["power"]), power, atol=2e-5, rtol=1e-6),
                            f"{name}: recorded real-step info disagrees with policy power")
                direct = independent_poa(
                    surface_tilt=float(p["tilt"][j]),
                    surface_azimuth=float(p["azimuth"][j]),
                    solar_zenith=float(info["solar_zenith_deg"]),
                    solar_azimuth=float(info["solar_azimuth_deg"]),
                    dni=float(info["dni_wm2"]), ghi=float(info["ghi_wm2"]),
                    dhi=float(info["dhi_wm2"]), model="isotropic")
                power_from_direct = max(0., float(direct["poa_global"])) * \
                    float(mcfg.sac.eval_env.get("area", 1.)) * float(mcfg.sac.eval_env.get("efficiency", 0.18))
                require(np.isclose(power, power_from_direct, atol=2e-5, rtol=1e-7),
                        f"{name}: policy power differs from independent pvlib at step {j+1}")
                checks["individual_pvlib_power_checks"] += 1
            else:
                max_other = max(float(by[m]["power"][j]) for m in
                                ("sac", "mbpo", "fixed_no_motion"))
                require(power + max(1e-5, 1e-6 * max_other) >= max_other,
                        f"POA oracle lower than a feasible policy at step {j+1}: {power} vs {max_other}")
                checks["poa_upper_bound_checks"] += 1
    fixed_tilt = np.asarray(ref["tilt"], float)
    fixed_az = np.asarray(ref["azimuth"], float)
    require(np.allclose(fixed_tilt, fixed_tilt[0], atol=1e-8) and
            np.allclose(fixed_az, fixed_az[0], atol=1e-8),
            "Fixed-mount baseline did not remain stationary")
    return {"scenario": scenario, "methods": list(by),
            "observed_invocations": observed, "checks": checks,
            "gross_kwh": {k: float(v["gross_energy_kwh"]) for k, v in by.items()},
            "net_kwh": {k: v["net_energy_kwh"] for k, v in by.items()},
            "meaning": "Real policy checkpoint execution, matched inputs and real pvlib power; not an out-of-sample estimate."}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="2022-07-15", help="NSRDB manifest scenario for paired evaluation")
    p.add_argument("--sac-checkpoint", default="SAC_workspace/runs/sac_seed0/latest.pt")
    p.add_argument("--mbpo-checkpoint", default="MBPO_workspace/runs/mbpo_seed0/latest.pt")
    p.add_argument("--skip-evaluation", action="store_true", help="Explicitly skip checkpoint evaluation (INCOMPLETE result)")
    p.add_argument("--json", type=Path, default=None, help="Optional structured report output")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path.cwd().resolve()
    if not all((root / x).is_dir() for x in ("SAC_workspace", "MBPO_workspace", "mbpo")):
        raise SystemExit("Run this script from the mbpo_custom checkout root.")
    sys.path.insert(0, str(root))
    audit = Audit(root)
    try:
        baseline_status = repo_git("status", "--porcelain", "--untracked-files=all")
        audit.run("Repository commit and local worktree provenance", lambda: code_provenance(root))
        imported = audit.run("All critical imports originate inside this checkout", lambda: import_origin_check(root))
        if imported is None:
            audit.incomplete("NSRDB source file parity", "Critical imports failed")
            audit.incomplete("Real SAC/MBPO trainer path", "Critical imports failed")
            audit.incomplete("Synthetic replay usage in saved training history", "Critical imports failed")
            audit.incomplete("Actual paired checkpoint evaluation", "Critical imports failed")
        else:
            weather = audit.run("NSRDB source file parity and hashes",
                                lambda: effective_config_and_weather(root, args.scenario))
            if weather is None:
                audit.incomplete("Real SAC/MBPO trainer path", "NSRDB source check failed")
                audit.incomplete("Actual paired checkpoint evaluation", "NSRDB source check failed")
            else:
                audit.run("Real SAC and MBPO trainers call same PVTrackingEnv and pvlib",
                          lambda: real_training_path_test(args.scenario))
            audit.run("MBPO saved history shows fitting, generation and mixture",
                      lambda: history_model_usage_check(root))
            if weather is not None:
                sc = (root / args.sac_checkpoint).resolve()
                mc = (root / args.mbpo_checkpoint).resolve()
                if mc.is_file():
                    audit.run("Trusted MBPO checkpoint contains valid model and real replay",
                              lambda: saved_checkpoint_replay_test(root, mc))
                else:
                    audit.incomplete("Trusted MBPO checkpoint replay inspection",
                                     "MBPO checkpoint file is missing")
                if args.skip_evaluation:
                    audit.incomplete("Actual paired checkpoint evaluation", "Disabled by --skip-evaluation")
                elif not (sc.is_file() and mc.is_file()):
                    audit.incomplete("Actual paired checkpoint evaluation",
                                     f"Missing trusted checkpoint(s): SAC={sc.is_file()}, MBPO={mc.is_file()}")
                else:
                    audit.run("Paired checkpoint evaluation on real pvlib; POA dominates per step",
                              lambda: paired_checkpoint_test(root, args.scenario, mc, sc))
        final_status = repo_git("status", "--porcelain", "--untracked-files=all")
        if baseline_status != final_status:
            audit.record("Project working tree unchanged by audit", "FAIL",
                         {"before": baseline_status, "after": final_status})
        else:
            audit.record("Project working tree unchanged by audit", "PASS")
    except Exception as exc:
        audit.record("Audit execution", "FAIL", {"exception": repr(exc), "traceback": traceback.format_exc()})
    summary = audit.summary()
    if args.json is not None:
        out = args.json.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        print("Saved JSON report:", out)
    print("\nTASK-0 INTEGRATION AUDIT:", summary["status"])
    print("(PASS = checks executed and passed; INCOMPLETE is never treated as PASS.)")
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[summary["status"]]


if __name__ == "__main__":
    sys.exit(main())
