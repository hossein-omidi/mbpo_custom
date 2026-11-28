# POSTPROCESSING_MC_CURVE_STATS_AND_VERIFY_FIX_V1
"""Empirical sample-complexity and compute-cost analysis for paired SAC/MBPO runs.

The independent variable is REAL PV simulator transitions. Synthetic MBPO samples
are reported separately and are never counted as physical environment samples.
"""
from __future__ import annotations
from mbpo.audit_support import target_event, expected_steps
from mbpo.geometry10 import ACTION_SEMANTICS

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _number(value, *, field: str, allow_nan: bool = False) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {field}: {value!r}") from exc
    if not allow_nan and not math.isfinite(out):
        raise ValueError(f"Nonfinite value for {field}: {value!r}")
    return out


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _load_run(run_dir: str | Path, expected_algorithm: str) -> dict:
    root = Path(run_dir).resolve()
    metadata = _read_json(root / "run_metadata.json")
    split = _read_json(root / "scenario_split.json")
    if split.get("protocol_sha256") != metadata.get("scenario_protocol_sha256"):
        raise ValueError(f"{root}: scenario_split.json fingerprint differs from run metadata")
    if list(split.get("validation_ids", [])) != list(metadata.get("validation_scenario_ids", [])):
        raise ValueError(f"{root}: validation scenario IDs differ from run metadata")
    if list(split.get("test_ids", [])) != list(metadata.get("test_scenario_ids", [])):
        raise ValueError(f"{root}: test scenario IDs differ from run metadata")
    if metadata.get("training_status") != "completed":
        raise ValueError(f"{root}: run is not marked completed; finish training before sample-complexity analysis")
    if metadata.get("algorithm") != expected_algorithm:
        raise ValueError(f"{root}: expected {expected_algorithm}, found {metadata.get('algorithm')}")
    if metadata.get("run_schema") != 2:
        raise ValueError(f"{root}: unsupported run schema; retrain with the verified evaluation patch")
    from PV_experiments.scientific_checks import verify_completed_run
    verify_completed_run(root,metadata)
    eval_root = root / "periodic_evaluation"
    aggregate = _read_csv(eval_root / "evaluations.csv")
    daily = _read_csv(eval_root / "evaluation_days.csv")
    references = _read_csv(eval_root / "evaluation_references.csv")
    episodes = _read_csv(root / "episodes.csv")
    allowed_training_days=set(map(str,split.get('train_ids',[])))
    if not allowed_training_days:
        raise ValueError(f'{root}: frozen split has no training scenarios')
    unexpected_training=[str(r.get('scenario_id','')) for r in episodes
                         if str(r.get('scenario_id','')) not in allowed_training_days]
    if unexpected_training:
        raise ValueError(f'{root}: training log includes non-training historical day IDs: '
                         f'{sorted(set(unexpected_training))[:10]}')
    if not aggregate or not daily or not references:
        raise ValueError(f"{root}: periodic evaluation files are empty")
    if len(episodes) != int(metadata.get("final_epoch", -1)):
        raise ValueError(f"{root}: episode history length differs from run_metadata.final_epoch")
    episode_steps = [int(r["env_steps"]) for r in episodes]
    horizon = int(metadata["horizon"])
    if (not episode_steps or episode_steps[-1] != int(metadata.get("final_env_steps", -1))
            or any((b - a) != horizon for a, b in zip([0] + episode_steps[:-1], episode_steps))):
        raise ValueError(f"{root}: episode interaction counts do not match the declared finite horizon")
    seed = int(metadata["seed"])
    episode_updates = {int(r["env_steps"]): int(r["updates"]) for r in episodes}
    for row in aggregate:
        if row.get("algorithm") != expected_algorithm or int(row.get("seed", -1)) != seed:
            raise ValueError(f"{root}: aggregate evaluation row has wrong algorithm/seed")
    for row in daily:
        if row.get("algorithm") != expected_algorithm or int(row.get("seed", -1)) != seed:
            raise ValueError(f"{root}: daily evaluation row has wrong algorithm/seed")
    validation_ids = list(metadata["validation_scenario_ids"])
    if [r["scenario_id"] for r in references] != validation_ids:
        raise ValueError(f"{root}: baseline scenario IDs do not match run metadata")
    step_to_days: dict[int, dict[str, float]] = {}
    for row in daily:
        step = int(row["env_steps"])
        sid = str(row["scenario_id"])
        bucket = step_to_days.setdefault(step, {})
        if sid in bucket:
            raise ValueError(f"{root}: duplicate daily evaluation at step={step}, scenario={sid}")
        bucket[sid] = _number(row["net_energy_kwh"], field="net_energy_kwh")
    step_to_aggregate = {}
    step_to_updates = {}
    for row in aggregate:
        step = int(row["env_steps"])
        epoch = int(row["epoch"])
        if step != epoch * horizon:
            raise ValueError(f"{root}: evaluation epoch/env_steps do not match the finite horizon")
        if step not in episode_updates or episode_updates[step] != int(row["updates"]):
            raise ValueError(f"{root}: evaluation gradient-update count does not match episode history at {step}")
        if step in step_to_aggregate:
            raise ValueError(f"{root}: duplicate aggregate evaluation at env_steps={step}")
        step_to_aggregate[step] = _number(row["mean_net_energy_kwh"], field="mean_net_energy_kwh")
        step_to_updates[step] = int(row["updates"])
        values = step_to_days.get(step, {})
        if set(values) != set(validation_ids):
            raise ValueError(f"{root}: incomplete validation days at env_steps={step}")
        if not np.isclose(np.mean(list(values.values())), step_to_aggregate[step], rtol=1e-7, atol=1e-9):
            raise ValueError(f"{root}: aggregate mean disagrees with daily evaluation rows at {step}")
    if set(step_to_days) != set(step_to_aggregate):
        raise ValueError(f"{root}: aggregate and daily evaluation checkpoints differ")
    return {
        "root": root, "metadata": metadata, "seed": seed,
        "aggregate": step_to_aggregate, "updates": step_to_updates, "daily": step_to_days,
        "references": references, "episodes": episodes,
    }


def _bootstrap_curves(matrix: np.ndarray, *, replicates: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-way cluster bootstrap over training seeds and validation scenario days.

    matrix dimensions: [training_seed, scenario_day, evaluation_checkpoint].
    The same sampled seed/day indices are used across algorithms for paired
    differences, preserving the crossed seed-by-scenario design.
    """
    if matrix.ndim != 3 or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite [seed, scenario, checkpoint] matrix")
    point = matrix.mean(axis=(0, 1))
    n_seed, n_day, _ = matrix.shape
    if n_seed < 2 or n_day < 2:
        raise ValueError("At least two independent training seeds and two scenario days are required")
    rng = np.random.default_rng(seed)
    boot = np.empty((int(replicates), matrix.shape[2]), dtype=float)
    for b in range(int(replicates)):
        si = rng.integers(0, n_seed, size=n_seed)
        di = rng.integers(0, n_day, size=n_day)
        boot[b] = matrix[si][:, di, :].mean(axis=(0, 1))
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    return point, lo, hi


def _threshold_steps(curve: np.ndarray, steps: np.ndarray, target: float, sustain: int) -> float | None:
    _, confirmed=target_event(curve,steps,target,sustain)
    return float(confirmed) if confirmed is not None else None


def _auc(curve: np.ndarray, steps: np.ndarray) -> float:
    if len(steps) < 2 or steps[-1] <= steps[0]:
        return float("nan")
    integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrator(curve, steps) / (steps[-1] - steps[0]))


def _training_compute(run: dict) -> dict:
    rows = run["episodes"]
    def total(name):
        vals = [_number(r[name], field=name, allow_nan=True) for r in rows if r.get(name) not in (None, "")]
        vals = [v for v in vals if math.isfinite(v)]
        return float(sum(vals)) if vals else None
    final_steps = max(int(r["env_steps"]) for r in rows)
    times=[_number(r["episode_seconds"],field="episode_seconds") for r in rows]
    if any(t<0 for t in times):raise ValueError("Negative training time")
    train_seconds=float(sum(times))
    eval_seconds = sum(_number(r["evaluation_seconds"], field="evaluation_seconds")
                       for r in _read_csv(run["root"] / "periodic_evaluation" / "evaluations.csv"))
    result = {
        "training_episode_seconds": train_seconds,
        "periodic_evaluation_seconds": float(eval_seconds),
        "measured_training_plus_evaluation_seconds": (
            float(train_seconds + eval_seconds) if train_seconds is not None else None),
        "training_transitions": final_steps,
        "training_transitions_per_episode_second": (
            float(final_steps / train_seconds) if train_seconds and train_seconds > 0 else None),
        "model_fit_seconds": total("model_fit_seconds"),
        "model_rollout_seconds": total("model_rollout_seconds"),
        "synthetic_attempted": total("synthetic_attempted"),
        "synthetic_accepted": total("synthetic_accepted"),
        "synthetic_rejected": total("synthetic_rejected"),
        "synthetic_samples_used": total("synthetic_samples_used"),
        "real_samples_used": total("real_samples_used"),
        "gradient_updates": int(max(int(r.get("updates", 0)) for r in rows)),
    }
    from PV_experiments.scientific_checks import recovery_accounting
    result.update(recovery_accounting(run["root"],final_steps,int(run["metadata"]["horizon"])))
    return result


def analyze_sample_complexity(
    sac_run_dirs: list[str | Path], mbpo_run_dirs: list[str | Path], outdir: str | Path,
    *, target_fraction: float | None = None, target_kwh: float | None = None,
    sustain_checkpoints: int | None = None, bootstrap_replicates: int | None = None,
    bootstrap_seed: int = 2026,
) -> dict:
    if len(sac_run_dirs) < 2 or len(mbpo_run_dirs) < 2:
        raise ValueError("At least two run directories per algorithm are required; five or more are recommended")
    if target_kwh is None and target_fraction is not None and not 0.0 < float(target_fraction) < 1.0:
        raise ValueError("target_fraction must be strictly between zero and one")
    sac_runs = [_load_run(p, "sac") for p in sac_run_dirs]
    mbpo_runs = [_load_run(p, "mbpo") for p in mbpo_run_dirs]
    for runs in (sac_runs, mbpo_runs):
        seeds = [r["seed"] for r in runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate training seed supplied")
    sac_runs.sort(key=lambda r: r["seed"])
    mbpo_runs.sort(key=lambda r: r["seed"])
    sac_seeds = [r["seed"] for r in sac_runs]
    mbpo_seeds = [r["seed"] for r in mbpo_runs]
    if sac_seeds != mbpo_seeds:
        raise ValueError(f"SAC and MBPO must use the same paired seed set: {sac_seeds} vs {mbpo_seeds}")

    all_runs = sac_runs + mbpo_runs
    from PV_experiments.scientific_checks import verify_run_comparability
    verify_run_comparability(all_runs)
    ref_meta = all_runs[0]["metadata"]
    target_fraction = float(ref_meta.get("target_fraction", 0.90)) if target_fraction is None else float(target_fraction)
    sustain_checkpoints = int(ref_meta.get("sustain_checkpoints", 2)) if sustain_checkpoints is None else int(sustain_checkpoints)
    bootstrap_replicates = int(ref_meta.get("bootstrap_replicates", 2000)) if bootstrap_replicates is None else int(bootstrap_replicates)
    if target_kwh is None and not 0.0 < target_fraction < 1.0:
        raise ValueError("target_fraction must be strictly between zero and one")
    if sustain_checkpoints < 1 or bootstrap_replicates < 100:
        raise ValueError("sustain_checkpoints must be positive and bootstrap_replicates at least 100")
    for run in all_runs[1:]:
        for key in ("scenario_protocol_sha256", "resolved_mdp_sha256", "shared_sac_config_sha256",
                    "validation_scenario_ids", "test_scenario_ids", "horizon", "state_dim", "action_dim",
                    "energy_objective", "movement_cost_mode", "movement_penalty", "power_model",
                    "target_fraction", "sustain_checkpoints", "bootstrap_replicates",
                    "evaluation_interval_episodes", "final_epoch", "final_env_steps"):
            if run["metadata"].get(key) != ref_meta.get(key):
                raise ValueError(f"Run comparability failure for {key}: {run['root']}")
    validation_ids = list(ref_meta["validation_scenario_ids"])
    if len(validation_ids) < 2:
        raise ValueError("Validation set is too small for uncertainty analysis")

    # Require all scheduled evaluations, not the intersection of incomplete runs.
    steps=expected_steps(ref_meta)
    for run in all_runs:
        if set(run['aggregate'])!=set(steps) or set(run['daily'])!=set(steps):
            raise ValueError(f"Missing or unexpected scheduled evaluations: {run['root']}")
    if len(steps) < 2:
        raise ValueError("Fewer than two common evaluation checkpoints across all runs")
    n_seed, n_day = len(sac_runs), len(validation_ids)
    updates_by_step = {}
    for step in steps:
        values = {int(run["updates"][int(step)]) for run in all_runs}
        if len(values) != 1:
            raise ValueError(f"SAC/MBPO gradient-update budgets differ at {int(step)} real transitions: {values}")
        updates_by_step[int(step)] = values.pop()

    def make_matrix(runs):
        matrix = np.empty((n_seed, n_day, len(steps)), dtype=float)
        for i, run in enumerate(runs):
            for j, step in enumerate(steps):
                daily = run["daily"][int(step)]
                missing = set(validation_ids) - set(daily)
                if missing:
                    raise ValueError(f"Missing validation days at {step}: {sorted(missing)}")
                matrix[i, :, j] = [daily[sid] for sid in validation_ids]
        return matrix

    sac_matrix = make_matrix(sac_runs)
    mbpo_matrix = make_matrix(mbpo_runs)
    sac_mean, sac_lo, sac_hi = _bootstrap_curves(
        sac_matrix, replicates=bootstrap_replicates, seed=bootstrap_seed)
    mbpo_mean, mbpo_lo, mbpo_hi = _bootstrap_curves(
        mbpo_matrix, replicates=bootstrap_replicates, seed=bootstrap_seed + 1)
    diff_matrix = mbpo_matrix - sac_matrix
    diff_mean, diff_lo, diff_hi = _bootstrap_curves(
        diff_matrix, replicates=bootstrap_replicates, seed=bootstrap_seed + 2)

    references = sac_runs[0]["references"]
    for run in all_runs[1:]:
        for r0, r1 in zip(references, run["references"]):
            for key in ("scenario_id", "fixed_net_energy_kwh", "dual_axis_net_energy_kwh"):
                if r0[key] != r1[key]:
                    raise ValueError(f"Evaluation baseline mismatch in {run['root']} for {key}")
    fixed_mean = float(np.mean([_number(r["fixed_net_energy_kwh"], field="fixed_net_energy_kwh") for r in references]))
    dual_mean = float(np.mean([_number(r["dual_axis_net_energy_kwh"], field="dual_axis_net_energy_kwh") for r in references]))
    if target_kwh is None:
        if dual_mean <= fixed_mean:
            raise ValueError(
                "Rate-limited dual-axis reference does not exceed fixed baseline on validation days; "
                "the relative target is undefined. Supply a pre-registered --target-kwh instead.")
        target = fixed_mean + float(target_fraction) * (dual_mean - fixed_mean)
        target_definition = f"fixed_mean + {target_fraction:.6g} * (dual_axis_mean - fixed_mean)"
    else:
        target = float(target_kwh)
        if not math.isfinite(target):
            raise ValueError("target_kwh must be finite")
        target_definition = "user-specified absolute validation net-energy target"

    per_seed_rows = []
    threshold_by_algorithm: dict[str, dict[int, float | None]] = {"sac": {}, "mbpo": {}}
    auc_by_algorithm: dict[str, list[float]] = {"sac": [], "mbpo": []}
    for algorithm, matrix in (("sac", sac_matrix), ("mbpo", mbpo_matrix)):
        for i, seed in enumerate(sac_seeds):
            curve = matrix[i].mean(axis=0)
            onset, reached = target_event(curve, steps, target, int(sustain_checkpoints))
            auc_value = _auc(curve, steps)
            threshold_by_algorithm[algorithm][seed] = reached
            auc_by_algorithm[algorithm].append(auc_value)
            per_seed_rows.append({
                "algorithm": algorithm, "seed": seed,
                "reached_target": reached is not None,
                "real_env_steps_to_target": reached,
                "onset_steps": onset, "confirmed_steps": reached,
                "restricted_steps": reached if reached is not None else int(steps[-1]),
                "last_evaluated_env_steps": int(steps[-1]),
                "auc_mean_net_kwh_per_episode": auc_value,
                "final_mean_net_energy_kwh": float(curve[-1]),
            })

    paired_reductions = []
    for seed in sac_seeds:
        ns = threshold_by_algorithm["sac"][seed]
        nm = threshold_by_algorithm["mbpo"][seed]
        if ns is not None and nm is not None and ns > 0:
            paired_reductions.append(100.0 * (1.0 - nm / ns))

    # Bootstrap AUC by resampling the crossed seed/day units.
    def auc_bootstrap(matrix, seed):
        rng = np.random.default_rng(seed)
        vals = np.empty(bootstrap_replicates, dtype=float)
        for b in range(bootstrap_replicates):
            si = rng.integers(0, n_seed, size=n_seed)
            di = rng.integers(0, n_day, size=n_day)
            curve = matrix[si][:, di, :].mean(axis=(0, 1))
            vals[b] = _auc(curve, steps)
        return [float(x) for x in np.quantile(vals, [0.025, 0.975])]

    curve_rows = []
    difference_rows = []
    for i, step in enumerate(steps):
        curve_rows.extend([
            {"algorithm": "sac", "env_steps": int(step), "gradient_updates": updates_by_step[int(step)], "mean_net_energy_kwh": sac_mean[i],
             "ci95_lower": sac_lo[i], "ci95_upper": sac_hi[i],
             "n_training_seeds": n_seed, "n_validation_days": n_day},
            {"algorithm": "mbpo", "env_steps": int(step), "gradient_updates": updates_by_step[int(step)], "mean_net_energy_kwh": mbpo_mean[i],
             "ci95_lower": mbpo_lo[i], "ci95_upper": mbpo_hi[i],
             "n_training_seeds": n_seed, "n_validation_days": n_day},
        ])
        difference_rows.append({"env_steps": int(step), "gradient_updates": updates_by_step[int(step)], "mbpo_minus_sac_kwh": diff_mean[i],
                                "ci95_lower": diff_lo[i], "ci95_upper": diff_hi[i]})

    summaries = []
    compute_rows = []
    for algorithm, runs, matrix, mean_curve, lo_curve, hi_curve in (
        ("sac", sac_runs, sac_matrix, sac_mean, sac_lo, sac_hi),
        ("mbpo", mbpo_runs, mbpo_matrix, mbpo_mean, mbpo_lo, mbpo_hi),
    ):
        reached_values = [v for v in threshold_by_algorithm[algorithm].values() if v is not None]
        auc_values = np.asarray(auc_by_algorithm[algorithm], dtype=float)
        auc_ci = auc_bootstrap(matrix, bootstrap_seed + (11 if algorithm == "sac" else 12))
        if len(reached_values) >= 2:
            threshold_rng = np.random.default_rng(bootstrap_seed + (21 if algorithm == "sac" else 22))
            threshold_boot = threshold_rng.choice(
                np.asarray(reached_values, dtype=float),
                size=(bootstrap_replicates, len(reached_values)), replace=True).mean(axis=1)
            threshold_ci = [float(x) for x in np.quantile(threshold_boot, [0.025, 0.975])]
        else:
            threshold_ci = None
        summaries.append({
            "algorithm": algorithm,
            "n_training_seeds": n_seed,
            "n_validation_days": n_day,
            "final_mean_net_energy_kwh": float(mean_curve[-1]),
            "final_ci95_lower": float(lo_curve[-1]),
            "final_ci95_upper": float(hi_curve[-1]),
            "mean_auc_net_kwh_per_episode": float(np.mean(auc_values)),
            "auc_ci95_lower": auc_ci[0], "auc_ci95_upper": auc_ci[1],
            "target_kwh_per_episode": float(target),
            "target_reached_seeds": len(reached_values),
            "target_reach_fraction": len(reached_values) / n_seed,
            "mean_real_steps_to_target_among_reached": float(np.mean(reached_values)) if reached_values else None,
            "sample_complexity_mean_ci95_among_reached": threshold_ci,
            "median_real_steps_to_target_among_reached": float(np.median(reached_values)) if reached_values else None,
        })
        for run in runs:
            compute = _training_compute(run)
            compute_rows.append({"algorithm": algorithm, "seed": run["seed"], **compute})

    restricted = {a: [r['restricted_steps'] for r in per_seed_rows if r['algorithm']==a] for a in ('sac','mbpo')}
    reduction_summary = {
        "endpoint": "confirmed attainment; onset is retrospective",
        "restricted_steps_saved_mean": float(np.mean(np.asarray(restricted['sac'])-np.asarray(restricted['mbpo']))),
        "restricted_budget_relative_saving_percent": float(100*(1-np.mean(restricted['mbpo'])/np.mean(restricted['sac']))),
        "paired_seed_count_reaching_target": len(paired_reductions),
        "mean_paired_sample_reduction_percent": float(np.mean(paired_reductions)) if paired_reductions else None,
        "median_paired_sample_reduction_percent": float(np.median(paired_reductions)) if paired_reductions else None,
        "paired_sample_reduction_ci95": (
            [float(x) for x in np.quantile(
                np.random.default_rng(bootstrap_seed + 13).choice(
                    paired_reductions, size=(bootstrap_replicates, len(paired_reductions)), replace=True
                ).mean(axis=1), [0.025, 0.975])]
            if len(paired_reductions) >= 2 else None
        ),
    }

    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "learning_curve.csv", curve_rows)
    _write_csv(out / "paired_difference.csv", difference_rows)
    _write_csv(out / "sample_complexity_by_seed.csv", per_seed_rows)
    _write_csv(out / "algorithm_summary.csv", summaries)
    _write_csv(out / "computational_cost_by_seed.csv", compute_rows)

    dual_field = ("dual_axis_instantaneous_validation_mean_kwh"
                  if ref_meta.get("action_semantics") == ACTION_SEMANTICS
                  else "dual_axis_slew_limited_validation_mean_kwh")
    report = {
        "analysis_schema": 1,
        "algorithms": ["sac", "mbpo"],
        "training_seeds": sac_seeds,
        "common_real_environment_steps": [int(x) for x in steps],
        "gradient_updates_at_common_steps": {str(k): int(v) for k, v in updates_by_step.items()},
        "validation_scenario_ids": validation_ids,
        "scenario_protocol_sha256": ref_meta["scenario_protocol_sha256"],
        "resolved_mdp_sha256": ref_meta["resolved_mdp_sha256"],
        "shared_sac_config_sha256": ref_meta["shared_sac_config_sha256"],
        "energy_objective": ref_meta.get("energy_objective"),
        "movement_cost_mode": ref_meta.get("movement_cost_mode"),
        "movement_penalty": ref_meta.get("movement_penalty"),
        "power_model": ref_meta.get("power_model"),
        "sample_complexity_definition": "confirmation at the final scheduled checkpoint of the first sustained sequence; retrospective onset reported separately",
        "target_definition": target_definition,
        "target_kwh_per_episode": float(target),
        "sustain_checkpoints": int(sustain_checkpoints),
        "fixed_validation_mean_kwh": fixed_mean,
        dual_field: dual_mean,
        "sample_reduction": reduction_summary,
        "algorithm_summary": summaries,
        "computational_cost": compute_rows,
        "uncertainty_method": "two-way percentile cluster bootstrap resampling training seeds and validation scenario days; paired indices for MBPO-SAC differences",
        "bootstrap_replicates": int(bootstrap_replicates),
        "limitations": [
            "Bootstrap intervals resample the observed training seeds and validation scenario days; they do not represent uncertainty over other sites, years, or weather-data providers.",
            "Real sample complexity counts only transitions collected from the original PV simulator.",
            "Synthetic transitions and synthetic minibatch draws are reported separately and are not counted as real samples.",
            "Wall-clock totals include logged episode and periodic evaluation time; they exclude process startup, dependency loading, and one-time baseline-reference construction.",
            "A target not reached within the measured budget is reported as not reached; no extrapolation is performed.",
            "AUC is normalized over the common observed interaction interval, not extrapolated to zero interactions.",
        ],
    }
    # Additional publication-oriented descriptive ±1 SD curves. Separate
    # SD across independent training seeds from SD across historical days.
    # The existing two-way bootstrap 95% CIs remain unchanged and are NOT SD.
    from MBPO_workspace.src.curve_statistics import write_sample_complexity_mean_sd
    report['descriptive_mean_sd_reporting'] = write_sample_complexity_mean_sd(
        out, steps, sac_matrix, mbpo_matrix, target)
    report['bootstrap_interval_caveat'] = (
        'Original two-way bootstrap intervals resample independent training seeds and a '
        'deterministically selected month-stratified set of validation days. They are '
        'resampling sensitivity intervals, not design-validated 95% coverage for '
        'all NSRDB days, future years, or weather-error realizations.')
    report['periodic_validation_weather'] = (
        'One frozen weather realization_id=0 per historical validation day; '
        'descriptive seed/day SD excludes within-day Monte Carlo draws')
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    _render_figures(out, steps, sac_mean, sac_lo, sac_hi, mbpo_mean, mbpo_lo, mbpo_hi,
                    diff_mean, diff_lo, diff_hi, per_seed_rows, compute_rows, target)
    return report


def _render_figures(out: Path, steps, sac_mean, sac_lo, sac_hi, mbpo_mean, mbpo_lo, mbpo_hi,
                    diff_mean, diff_lo, diff_hi, per_seed_rows, compute_rows, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(steps, sac_mean, label="SAC", linewidth=2)
    ax.fill_between(steps, sac_lo, sac_hi, alpha=0.18, label="SAC central 95% resampling band (sensitivity)")
    ax.plot(steps, mbpo_mean, label="MBPO", linewidth=2)
    ax.fill_between(steps, mbpo_lo, mbpo_hi, alpha=0.18, label="MBPO central 95% resampling band (sensitivity)")
    ax.axhline(target, linestyle="--", linewidth=1.2, label="Predefined target")
    ax.set(xlabel="Real PV simulator transitions", ylabel="Mean validation net energy (kWh / episode)",
           title="Empirical sample-efficiency learning curves")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(steps, diff_mean, linewidth=2, label="MBPO − SAC")
    ax.fill_between(steps, diff_lo, diff_hi, alpha=0.2, label="Paired central 95% resampling band (sensitivity)")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set(xlabel="Real PV simulator transitions", ylabel="Net-energy difference (kWh / episode)",
           title="Paired MBPO–SAC validation performance")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(out / "paired_difference.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for algorithm, marker in (("sac", "o"), ("mbpo", "s")):
        rows = [r for r in per_seed_rows if r["algorithm"] == algorithm and r["reached_target"]]
        if rows:
            ax.scatter([r["seed"] for r in rows], [r["real_env_steps_to_target"] for r in rows],
                       marker=marker, s=65, label=algorithm.upper())
        missed=[r for r in per_seed_rows if r['algorithm']==algorithm and not r['reached_target']]
        if missed:
            ax.scatter([r['seed'] for r in missed],[r['last_evaluated_env_steps'] for r in missed],marker=marker,facecolors='none',edgecolors='k',s=65,label=algorithm.upper()+' not confirmed (budget)')
    ax.set(xlabel="Training seed", ylabel="Confirmation steps or censored budget",
           title="Per-seed empirical sample complexity")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(out / "sample_complexity_by_seed.png", dpi=180); plt.close(fig)

    algos = ["sac", "mbpo"]
    mean_time = []
    for algorithm in algos:
        vals = [r["measured_training_plus_evaluation_seconds"] for r in compute_rows
                if r["algorithm"] == algorithm and r["measured_training_plus_evaluation_seconds"] is not None]
        if len(vals)!=sum(r["algorithm"]==algorithm for r in compute_rows):raise ValueError("Missing compute timing")
        mean_time.append(float(np.mean(vals)))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    paired={a:{r['seed']:r['measured_training_plus_evaluation_seconds'] for r in compute_rows if r['algorithm']==a} for a in algos}
    if set(paired['sac'])!=set(paired['mbpo']):raise ValueError('Unpaired timing seeds')
    for seed in sorted(paired['sac']):ax.plot([0,1],[paired[a][seed] for a in algos],'-o',alpha=.5,label=f'Seed {seed}')
    sd=[np.std(list(paired[a].values()),ddof=1) for a in algos] if len(paired['sac'])>1 else None
    ax.errorbar([0,1],mean_time,yerr=sd,fmt='D',color='k',capsize=3,label='Mean ± seed SD' if sd is not None else 'Mean')
    ax.set_xticks([0,1],['SAC','MBPO']);ax.legend(fontsize=8)
    ax.set(ylabel="Measured training + periodic evaluation time (s)",
           title="Computational cost per training seed")
    ax.grid(axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(out / "computational_cost.png", dpi=180); plt.close(fig)
