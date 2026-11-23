# POSTPROCESSING_TRAJECTORY_UNCERTAINTY_V1
"""Reproducible scenario partitioning and deterministic periodic policy evaluation.

This module deliberately leaves SAC/MBPO learning updates unchanged. It makes the
training, validation, and final-test scenario roles explicit and shared by both
algorithms, then evaluates policies only in the original PV simulator.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_VERSION = "pv_nsrdb_month_stratified_v1"
STATE_DIM = 6
ACTION_DIM = 2


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _stable_key(seed: int, role: str, scenario_id: str) -> str:
    text = f"{int(seed)}|{role}|{scenario_id}".encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def build_scenario_protocol_from_rows(
    rows: list[dict], *, manifest_sha256: str, split_seed: int = 2026,
    validation_days_per_month: int = 1, test_days_per_month: int = 2,
    preferred_validation_ids: list[str] | tuple[str, ...] = (),
    evaluation_interval_episodes: int = 50, target_fraction: float = 0.90,
    sustain_checkpoints: int = 2, bootstrap_replicates: int = 2000,
) -> dict:
    """Create a deterministic month-stratified train/validation/test partition.

    Preferred validation dates are retained first (the five reference dates in
    _nsrdb_base.py). Remaining validation and test dates are selected by a
    stable SHA-256 ordering within each calendar month. All remaining scenarios
    are training-only. Scenario weights are renormalized over training IDs.
    """
    v_per_month = int(validation_days_per_month)
    t_per_month = int(test_days_per_month)
    if v_per_month < 1 or t_per_month < 1:
        raise ValueError("At least one validation and one test day per month are required")
    if int(evaluation_interval_episodes) < 1 or int(sustain_checkpoints) < 1 or int(bootstrap_replicates) < 100:
        raise ValueError("Invalid periodic-evaluation or bootstrap protocol settings")
    if not math.isfinite(float(target_fraction)) or not 0.0 < float(target_fraction) < 1.0:
        raise ValueError("target_fraction must be finite and strictly between zero and one")
    if not rows:
        raise ValueError("The NSRDB manifest contains no scenarios")

    by_month: dict[int, list[dict]] = {month: [] for month in range(1, 13)}
    by_id: dict[str, dict] = {}
    for raw in rows:
        row = dict(raw)
        sid = str(row.get("scenario_id") or row.get("id") or "").strip()
        if not sid:
            try:
                sid = f"{int(row.get('source_year', row['year'])):04d}-{int(row['month']):02d}-{int(row['day']):02d}"
            except Exception as exc:
                raise ValueError(f"Manifest scenario has no valid identifier: {row}") from exc
        try:
            month = int(row["month"])
            year = int(row.get("source_year", row.get("year")))
            day = int(row["day"])
            import datetime as _dt
            _dt.date(year, month, day)
        except Exception as exc:
            raise ValueError(f"Invalid calendar fields for scenario {sid!r}") from exc
        if not 1 <= month <= 12:
            raise ValueError(f"Invalid month for scenario {sid!r}")
        if sid in by_id:
            raise ValueError(f"Duplicate scenario ID in manifest: {sid}")
        row["scenario_id"] = sid
        row["month"] = month
        row["source_year"] = year
        by_id[sid] = row
        by_month[month].append(row)

    preferred = [str(x) for x in preferred_validation_ids]
    missing_preferred = sorted(set(preferred) - set(by_id))
    if missing_preferred:
        raise ValueError(f"Preferred validation scenarios missing from manifest: {missing_preferred}")
    preferred_by_month: dict[int, list[str]] = {m: [] for m in range(1, 13)}
    for sid in preferred:
        preferred_by_month[int(by_id[sid]["month"])].append(sid)

    validation_ids: list[str] = []
    test_ids: list[str] = []
    train_ids: list[str] = []
    for month in range(1, 13):
        candidates = by_month[month]
        minimum = v_per_month + 2  # validation + at least one test day + at least one training day
        if len(candidates) < minimum:
            raise ValueError(
                f"Month {month:02d} has {len(candidates)} scenarios; at least {minimum} "
                "are required for validation, test, and training partitions")
        preferred_month = sorted(preferred_by_month[month])
        if len(preferred_month) > v_per_month:
            raise ValueError(
                f"Month {month:02d} has {len(preferred_month)} preferred validation dates, "
                f"exceeding validation_days_per_month={v_per_month}")
        validation_month = list(preferred_month)
        remaining = [r for r in candidates if r["scenario_id"] not in set(validation_month)]
        remaining.sort(key=lambda r: _stable_key(split_seed, f"validation-{month}", r["scenario_id"]))
        while len(validation_month) < v_per_month:
            validation_month.append(remaining.pop(0)["scenario_id"])
        validation_ids.extend(validation_month)

        remaining.sort(key=lambda r: _stable_key(split_seed, f"test-{month}", r["scenario_id"]))
        actual_test_count = min(t_per_month, len(remaining) - 1)
        month_test = remaining[:actual_test_count]
        test_ids.extend(r["scenario_id"] for r in month_test)
        selected_test = {r["scenario_id"] for r in month_test}
        train_ids.extend(r["scenario_id"] for r in remaining[actual_test_count:])
        if selected_test.intersection(validation_month):
            raise RuntimeError("Internal scenario partition overlap")

    validation_ids = sorted(validation_ids)
    test_ids = sorted(test_ids)
    train_ids = sorted(train_ids)
    all_ids = set(by_id)
    if (set(train_ids) & set(validation_ids) or set(train_ids) & set(test_ids)
            or set(validation_ids) & set(test_ids)
            or set(train_ids) | set(validation_ids) | set(test_ids) != all_ids):
        raise RuntimeError("Scenario partition is not a disjoint exhaustive partition")

    try:
        from mbpo.env.nsrdb_weather import scenario_row_weight
        raw_weights = np.asarray([scenario_row_weight(by_id[sid]) for sid in train_ids], dtype=float)
    except Exception:
        def _fallback_weight(row):
            for key in ("probability", "weight", "sample_weight"):
                if row.get(key) is None:
                    continue
                try:
                    value = float(row[key])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value >= 0.0:
                    return value
            return 1.0
        raw_weights = np.asarray([_fallback_weight(by_id[sid]) for sid in train_ids], dtype=float)
    if (not np.isfinite(raw_weights).all()) or (raw_weights < 0).any():
        raise ValueError("Training scenario weights must be finite and nonnegative")
    if raw_weights.sum() <= 0:
        raw_weights = np.ones(len(train_ids), dtype=float)
    probabilities = raw_weights / raw_weights.sum()

    body = {
        "protocol_version": PROTOCOL_VERSION,
        "split_seed": int(split_seed),
        "manifest_sha256": str(manifest_sha256),
        "validation_days_per_month": v_per_month,
        "test_days_per_month": t_per_month,
        "preferred_validation_ids": preferred,
        "evaluation_interval_episodes": int(evaluation_interval_episodes),
        "target_fraction": float(target_fraction),
        "sustain_checkpoints": int(sustain_checkpoints),
        "bootstrap_replicates": int(bootstrap_replicates),
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "test_ids": test_ids,
        "train_probabilities": [float(x) for x in probabilities],
        "month_counts": {
            str(m): {
                "train": sum(int(by_id[s]["month"]) == m for s in train_ids),
                "validation": sum(int(by_id[s]["month"]) == m for s in validation_ids),
                "test": sum(int(by_id[s]["month"]) == m for s in test_ids),
            } for m in range(1, 13)
        },
    }
    body["protocol_sha256"] = canonical_json_sha256(body)
    return body


def build_scenario_protocol(manifest_path: str | Path, protocol_config: dict) -> dict:
    from mbpo.env.nsrdb_weather import load_scenario_manifest
    manifest_path = Path(manifest_path).resolve()
    manifest = load_scenario_manifest(manifest_path)
    return build_scenario_protocol_from_rows(
        manifest["scenarios"], manifest_sha256=file_sha256(manifest_path),
        split_seed=int(protocol_config.get("split_seed", 2026)),
        validation_days_per_month=int(protocol_config.get("validation_days_per_month", 1)),
        test_days_per_month=int(protocol_config.get("test_days_per_month", 2)),
        preferred_validation_ids=protocol_config.get("preferred_validation_ids", ()),
        evaluation_interval_episodes=int(protocol_config.get("evaluation_interval_episodes", 50)),
        target_fraction=float(protocol_config.get("target_fraction", 0.90)),
        sustain_checkpoints=int(protocol_config.get("sustain_checkpoints", 2)),
        bootstrap_replicates=int(protocol_config.get("bootstrap_replicates", 2000)),
    )


def persist_scenario_protocol(run_dir: str | Path, protocol: dict, *, resume: bool) -> Path:
    path = Path(run_dir) / "scenario_split.json"
    if resume:
        if not path.is_file():
            raise FileNotFoundError("Resume requires scenario_split.json from the original run")
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != protocol:
            raise ValueError("Scenario split differs from the original run; refusing resume")
    else:
        if path.exists():
            raise FileExistsError(f"Scenario split already exists: {path}")
        write_json_atomic(path, protocol)
    return path


def prepare_run_metadata(
    run_dir: str | Path, *, algorithm: str, seed: int, config_sha256: str,
    shared_sac_config_sha256: str, resolved_mdp: dict, protocol: dict,
    horizon: int, resume: bool,
) -> dict:
    run_dir = Path(run_dir)
    path = run_dir / "run_metadata.json"
    metadata = {
        "run_schema": 2,
        "algorithm": str(algorithm),
        "seed": int(seed),
        "config_sha256": str(config_sha256),
        "shared_sac_config_sha256": str(shared_sac_config_sha256),
        "resolved_mdp_sha256": canonical_json_sha256(resolved_mdp),
        "scenario_protocol_sha256": protocol["protocol_sha256"],
        "scenario_split_file": "scenario_split.json",
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "horizon": int(horizon),
        "power_model": resolved_mdp.get("power_model", "legacy"),
        "movement_penalty": float(resolved_mdp.get("movement_penalty", 0.0)),
        "movement_cost_mode": resolved_mdp.get("movement_cost_mode"),
        "energy_objective": (
            "gross_DC_kWh_no_actuator_consumption"
            if float(resolved_mdp.get("movement_penalty", 0.0)) == 0.0
            else "DC_kWh_less_configured_movement_cost"),
        "evaluation_protocol_version": PROTOCOL_VERSION,
        "evaluation_interval_episodes": int(protocol["evaluation_interval_episodes"]),
        "target_fraction": float(protocol["target_fraction"]),
        "sustain_checkpoints": int(protocol["sustain_checkpoints"]),
        "bootstrap_replicates": int(protocol["bootstrap_replicates"]),
        "validation_scenario_ids": list(protocol["validation_ids"]),
        "test_scenario_ids": list(protocol["test_ids"]),
        "training_status": "in_progress",
    }
    if resume:
        if not path.is_file():
            raise FileNotFoundError("Resume requires run_metadata.json from the original run")
        saved = json.loads(path.read_text(encoding="utf-8"))
        for key, value in metadata.items():
            if key == "training_status":
                continue
            if saved.get(key) != value:
                raise ValueError(f"Run metadata mismatch for {key}: {saved.get(key)!r} != {value!r}")
        return saved
    if path.exists():
        raise FileExistsError(f"Run metadata already exists: {path}")
    metadata["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json_atomic(path, metadata)
    return metadata


def finish_run_metadata(run_dir: str | Path, *, final_epoch: int, env_steps: int,
                        training_seconds: float, status: str = "completed") -> None:
    path = Path(run_dir) / "run_metadata.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update({"training_status": status, "final_epoch": int(final_epoch),
                 "final_env_steps": int(env_steps),
                 "training_loop_seconds": float(training_seconds),
                 "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    write_json_atomic(path, data)


def _write_csv_replace(path: Path, fieldnames: list[str], rows: list[dict], *, replace_key: str | None = None,
                       replace_value: Any = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old_rows = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            old_rows = list(reader)
    if replace_key is not None:
        old_rows = [r for r in old_rows if str(r.get(replace_key)) != str(replace_value)]
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerows(rows)
    os.replace(tmp, path)


class PeriodicEvaluator:
    """Evaluate deterministic policies and common baselines on fixed validation days."""

    def __init__(self, env, protocol: dict, run_dir: str | Path, *, algorithm: str,
                 seed: int, config_sha256: str, resolved_mdp: dict):
        self.env = env
        self.protocol = protocol
        self.run_dir = Path(run_dir)
        self.algorithm = str(algorithm)
        self.seed = int(seed)
        self.config_sha256 = str(config_sha256)
        self.resolved_mdp_sha256 = canonical_json_sha256(resolved_mdp)
        self.validation_ids = list(protocol["validation_ids"])
        if tuple(getattr(self.env.observation_space, "shape", ())) != (STATE_DIM,):
            raise ValueError(f"Periodic evaluator requires {STATE_DIM} compact observation features")
        if tuple(getattr(self.env.action_space, "shape", ())) != (ACTION_DIM,):
            raise ValueError(f"Periodic evaluator requires {ACTION_DIM} continuous actions")
        if bool(getattr(self.env.unwrapped, "randomize_initial_orientation", False)):
            raise ValueError("Periodic evaluation requires a fixed initial panel orientation for paired scenarios")
        self.eval_dir = self.run_dir / "periodic_evaluation"
        self.snapshot_dir = self.run_dir / "policy_snapshots"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.references_path = self.eval_dir / "evaluation_references.csv"
        self._ensure_references()
        write_json_atomic(self.eval_dir / "protocol.json", {
            "protocol_version": PROTOCOL_VERSION,
            "scenario_protocol_sha256": protocol["protocol_sha256"],
            "validation_scenario_ids": self.validation_ids,
            "evaluation_interval_episodes": int(protocol["evaluation_interval_episodes"]),
            "target_fraction": float(protocol["target_fraction"]),
            "sustain_checkpoints": int(protocol["sustain_checkpoints"]),
            "bootstrap_replicates": int(protocol["bootstrap_replicates"]),
            "policy_evaluation": "deterministic_actor_mean_tanh",
            "environment": "original StageAPVTrackingEnv; no learned model transitions",
            "energy_objective": (
                "gross_DC_kWh_no_actuator_consumption"
                if float(self.env.unwrapped.movement_penalty) == 0.0
                else "DC_kWh_less_configured_movement_cost"),
            "movement_cost_mode": str(self.env.unwrapped.movement_cost_mode),
        })

    def _run_episode(self, scenario_id: str, *, policy=None, controller: str = "fixed") -> dict:
        # A fixed common uncertainty realization maintains identical
        # reference and learned-policy weather at every checkpoint.
        obs = self.env.reset(scenario_id=scenario_id, realization_id=0)
        gross = 0.0
        movement = 0.0
        net = 0.0
        steps = 0
        underlying = self.env.unwrapped
        for _ in range(self.env.horizon):
            if controller == "policy":
                action = policy.act(obs, deterministic=True)
            elif controller == "fixed":
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
            else:
                raise ValueError(f"Unknown evaluation controller: {controller}")
            obs, reward, done, info = self.env.step(action)
            if str(info.get("scenario_id")) != str(scenario_id):
                raise RuntimeError("Evaluation environment changed the explicitly selected NSRDB scenario")
            gross += float(info["energy_kwh"])
            movement += float(info["movement_cost"])
            net += float(reward)
            steps += 1
            if done:
                break
        if steps != self.env.horizon:
            raise RuntimeError(f"Incomplete evaluation episode for {scenario_id}: {steps} steps")
        return {"scenario_id": scenario_id, "net_energy_kwh": net,
                "gross_energy_kwh": gross, "movement_cost_kwh": movement,
                "episode_steps": steps}

    def _ensure_references(self) -> None:
        if self.references_path.is_file():
            with self.references_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            ids = [r.get("scenario_id") for r in rows]
            if ids != self.validation_ids:
                raise ValueError("Saved evaluation baseline scenarios differ from scenario_split.json")
            for row in rows:
                for key in ("fixed_net_energy_kwh", "dual_axis_net_energy_kwh"):
                    if not math.isfinite(float(row[key])):
                        raise ValueError("Nonfinite saved evaluation reference")
            return
        rows = []
        for sid in self.validation_ids:
            fixed = self._run_episode(sid, controller="fixed")
            dual = self._run_episode(sid, controller="dual_axis_slew_limited")
            rows.append({
                "scenario_id": sid,
                "fixed_net_energy_kwh": fixed["net_energy_kwh"],
                "fixed_gross_energy_kwh": fixed["gross_energy_kwh"],
                "fixed_movement_cost_kwh": fixed["movement_cost_kwh"],
                "dual_axis_net_energy_kwh": dual["net_energy_kwh"],
                "dual_axis_gross_energy_kwh": dual["gross_energy_kwh"],
                "dual_axis_movement_cost_kwh": dual["movement_cost_kwh"],
                "episode_steps": fixed["episode_steps"],
            })
        fields = list(rows[0])
        _write_csv_replace(self.references_path, fields, rows)

    def has_evaluation(self, *, epoch: int, env_steps: int) -> bool:
        path = self.eval_dir / "evaluations.csv"
        if not path.is_file():
            return False
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if int(row.get("epoch", -1)) == int(epoch) and int(row.get("env_steps", -1)) == int(env_steps):
                    return True
        return False

    def evaluate(self, policy, *, epoch: int, env_steps: int, updates: int) -> dict:
        started = time.perf_counter()
        actor = getattr(policy, "actor", None)
        was_training = bool(actor.training) if actor is not None else None
        if actor is not None:
            actor.eval()
        try:
            day_rows = []
            for sid in self.validation_ids:
                result = self._run_episode(sid, policy=policy, controller="policy")
                day_rows.append({
                    "algorithm": self.algorithm, "seed": self.seed,
                    "epoch": int(epoch), "env_steps": int(env_steps),
                    "scenario_id": sid, **result,
                })
        finally:
            if actor is not None and was_training:
                actor.train()
        values = np.asarray([r["net_energy_kwh"] for r in day_rows], dtype=float)
        gross = np.asarray([r["gross_energy_kwh"] for r in day_rows], dtype=float)
        movement = np.asarray([r["movement_cost_kwh"] for r in day_rows], dtype=float)
        elapsed = time.perf_counter() - started
        aggregate = {
            "algorithm": self.algorithm, "seed": self.seed,
            "epoch": int(epoch), "env_steps": int(env_steps), "updates": int(updates),
            "n_scenarios": len(day_rows),
            "mean_net_energy_kwh": float(values.mean()),
            "sd_net_energy_kwh": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median_net_energy_kwh": float(np.median(values)),
            "mean_gross_energy_kwh": float(gross.mean()),
            "mean_movement_cost_kwh": float(movement.mean()),
            "evaluation_seconds": float(elapsed),
        }
        eval_dir = self.eval_dir
        day_fields = list(day_rows[0])
        _write_csv_replace(eval_dir / "evaluation_days.csv", day_fields, day_rows,
                           replace_key="epoch", replace_value=epoch)
        aggregate_fields = list(aggregate)
        _write_csv_replace(eval_dir / "evaluations.csv", aggregate_fields, [aggregate],
                           replace_key="epoch", replace_value=epoch)
        self._save_policy_snapshot(policy, epoch=epoch, env_steps=env_steps, updates=updates)
        return aggregate

    def _save_policy_snapshot(self, policy, *, epoch: int, env_steps: int, updates: int) -> Path:
        import torch
        path = self.snapshot_dir / f"{self.algorithm}_seed{self.seed}_epoch{int(epoch):05d}.pt"
        metadata = {
            "snapshot_schema": 1, "algorithm": self.algorithm, "seed": self.seed,
            "epoch": int(epoch), "env_steps": int(env_steps), "updates": int(updates),
            "config_sha256": self.config_sha256,
            "scenario_protocol_sha256": self.protocol["protocol_sha256"],
            "resolved_mdp_sha256": self.resolved_mdp_sha256,
            "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
            "policy_action": "tanh(actor_mean); deterministic evaluation",
        }
        state = {k: v.detach().cpu().clone() for k, v in policy.actor.state_dict().items()}
        tmp = path.with_suffix(".tmp")
        torch.save({"metadata": metadata, "actor_state_dict": state}, tmp)
        os.replace(tmp, path)
        return path
