"""Validate the saved one-step MBPO ensemble on internal and external held-out days.

The external checks use the original PV simulator. Counterfactual action checks
replay the same policy prefix to reconstruct each observed state, then compare
several alternative actions against the real next transition.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from MBPO_workspace.src.config import load_config, config_digest
from MBPO_workspace.src.dynamics import DynamicsEnsemble, StructuredFakeEnv
from MBPO_workspace.src.replay import ScenarioReplayBuffer
from SAC_workspace.src.agents.sac_agent import SACAgent
from SAC_workspace.src.environment.environment import SACEnvironment
from SAC_workspace.src.training.experiment_protocol import canonical_json_sha256

CHANNELS = ("dni_norm", "dhi_norm", "power_norm")


def _csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _rmse(x):
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=float)))))


def _channel_metrics(actual, predicted, total_std=None):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = predicted - actual
    if actual.ndim != 2 or actual.shape[1] != 3 or predicted.shape != actual.shape:
        raise ValueError("Model-validation arrays must have shape [n, 3]")
    out = {"n_transitions": int(len(actual))}
    for j, name in enumerate(CHANNELS):
        y = actual[:, j]
        err = residual[:, j]
        persistence_rmse = _rmse(y)
        rmse = _rmse(err)
        out[name] = {
            "rmse": rmse,
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "persistence_rmse": persistence_rmse,
            "skill_vs_persistence": float(1.0 - rmse / max(persistence_rmse, 1e-12)),
        }
        if total_std is not None:
            sd = np.maximum(np.asarray(total_std, dtype=float)[:, j], 1e-8)
            out[name]["mean_predictive_sd"] = float(np.mean(sd))
            out[name]["rmse_to_mean_predictive_sd"] = float(rmse / max(float(np.mean(sd)), 1e-12))
            out[name]["coverage_90"] = float(np.mean(np.abs(err) <= 1.6448536269514722 * sd))
            out[name]["coverage_95"] = float(np.mean(np.abs(err) <= 1.959963984540054 * sd))
    return out


def _prediction_rows(actual, predicted, uncertainty, metadata):
    rows = []
    for i in range(len(actual)):
        row = dict(metadata[i])
        row["normalized_epistemic_disagreement"] = float(uncertainty[i])
        for j, name in enumerate(CHANNELS):
            row[f"actual_delta_{name}"] = float(actual[i, j])
            row[f"predicted_delta_{name}"] = float(predicted[i, j])
            row[f"error_delta_{name}"] = float(predicted[i, j] - actual[i, j])
        rows.append(row)
    return rows


def _load_training_stats(path: Path) -> dict:
    if not path.is_file():
        return {"available": False}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {"available": bool(rows), "episodes": len(rows)}
    for key in ("model_fit_seconds", "model_rollout_seconds", "synthetic_attempted",
                "synthetic_accepted", "synthetic_rejected", "synthetic_samples_used",
                "real_samples_used"):
        vals = []
        for row in rows:
            try:
                value = float(row.get(key, "nan"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                vals.append(value)
        result[key] = float(sum(vals)) if vals else None
    for key in ("model_train_scenarios", "model_holdout_scenarios"):
        observed = []
        for row in rows:
            try:
                value = float(row.get(key, "nan"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                observed.append(value)
        result[f"last_{key}"] = observed[-1] if observed else None
        result[f"max_{key}"] = max(observed) if observed else None
    attempted = result.get("synthetic_attempted")
    accepted = result.get("synthetic_accepted")
    result["synthetic_acceptance_rate"] = (
        float(accepted / attempted) if attempted is not None and attempted > 0 and accepted is not None else None)
    return result


def _run_policy_trajectory(env, agent, scenario_id):
    obs = env.reset(scenario_id=scenario_id)
    states, actions, next_states, rewards = [], [], [], []
    for _ in range(env.horizon):
        action = agent.act(obs, deterministic=True)
        nxt, reward, done, _info = env.step(action)
        states.append(np.asarray(obs, dtype=np.float32).copy())
        actions.append(np.asarray(action, dtype=np.float32).copy())
        next_states.append(np.asarray(nxt, dtype=np.float32).copy())
        rewards.append(float(reward))
        obs = nxt
        if done:
            break
    if len(states) != env.horizon:
        raise RuntimeError(f"Incomplete policy trajectory for {scenario_id}")
    return (np.asarray(states), np.asarray(actions), np.asarray(next_states),
            np.asarray(rewards, dtype=np.float32))


def _counterfactual_actions(policy_action):
    return [
        ("policy", np.asarray(policy_action, dtype=np.float32)),
        ("zero", np.asarray([0.0, 0.0], dtype=np.float32)),
        ("tilt_positive", np.asarray([1.0, 0.0], dtype=np.float32)),
        ("tilt_negative", np.asarray([-1.0, 0.0], dtype=np.float32)),
        ("azimuth_positive", np.asarray([0.0, 1.0], dtype=np.float32)),
        ("azimuth_negative", np.asarray([0.0, -1.0], dtype=np.float32)),
        ("diagonal_positive", np.asarray([1.0, 1.0], dtype=np.float32)),
        ("diagonal_negative", np.asarray([-1.0, -1.0], dtype=np.float32)),
    ]


def run_validation(args):
    run_dir = Path(args.run_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    outdir = Path(args.outdir).resolve()
    model_overrides = {}
    if args.model_epochs is not None:
        model_overrides["model_epochs"] = args.model_epochs
    if args.model_max_train_samples is not None:
        model_overrides["model_max_train_samples"] = args.model_max_train_samples
    cfg = load_config(
        model_overrides=model_overrides,
        power_model=args.power_model,
        wind_reference_height_m=args.wind_reference_height_m,
        gamma_pdc=args.gamma_pdc,
        sapm_mount=args.sapm_mount,
        wind_powerlaw_exponent=args.wind_powerlaw_exponent,
    )
    saved_cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    requested_cfg = json.loads(json.dumps(cfg.to_dict()))
    if saved_cfg != requested_cfg:
        raise ValueError("Current MBPO configuration differs from the selected run config.json")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ckpt.get("schema") != 2 or ckpt.get("config_hash") != config_digest(cfg):
        raise ValueError("Checkpoint schema/configuration mismatch; use a checkpoint from the verified patch")
    seed = int(ckpt["seed"])
    split = json.loads((run_dir / "scenario_split.json").read_text(encoding="utf-8"))
    from SAC_workspace.src.training.experiment_protocol import build_scenario_protocol
    expected_split = build_scenario_protocol(cfg.sac.manifest, cfg.sac.protocol)
    if split != expected_split or split.get("protocol_sha256") != ckpt.get("scenario_protocol_sha256"):
        raise ValueError("Checkpoint/scenario_split.json does not match the current manifest and experiment protocol")
    scenario_ids = list(split["validation_ids"])
    if args.max_scenarios is not None:
        scenario_ids = scenario_ids[:int(args.max_scenarios)]
    if not scenario_ids:
        raise ValueError("No validation scenarios selected")

    a = cfg.sac.algo
    agent = SACAgent(hidden_sizes=a["hidden_sizes"], lr=a["lr"], gamma=a["gamma"],
                     tau=a["tau"], reward_scale=a["reward_scale"],
                     alpha_init=a["alpha_init"], target_entropy=a["target_entropy"],
                     min_alpha=a["min_alpha"], device=args.device)
    agent.load_state_dict(ckpt["agent"])
    agent.actor.eval()

    m = cfg.model
    ensemble = DynamicsEnsemble(num_models=m.num_models, num_elites=m.num_elites,
                                hidden_dim=m.hidden_dim, lr=m.model_lr,
                                device=args.device, seed=seed + 100)
    ensemble.load_state_dict(ckpt["dynamics"])
    if not ensemble.fitted:
        raise ValueError("Checkpoint contains an unfitted dynamics ensemble")

    env = SACEnvironment(cfg.sac.eval_env)
    if env.resolved_mdp != ckpt.get("resolved_mdp"):
        env.close()
        raise ValueError("Evaluation simulator fingerprint differs from the MBPO checkpoint")
    fake_env = StructuredFakeEnv(ensemble, env)

    try:
        # Internal held-out transitions: whole NSRDB days reserved by the MBPO replay.
        real_pool = ScenarioReplayBuffer(capacity=a["replay_size"], seed=seed + 1,
                                         validation_fraction=m.model_holdout_fraction)
        real_pool.load_state_dict(ckpt["real_pool"])
        _train_idx, holdout_idx = real_pool.scenario_split(
            m.model_holdout_fraction, seed=seed)
        holdout_states = real_pool.states[holdout_idx]
        holdout_actions = real_pool.actions[holdout_idx]
        holdout_actual_delta = (real_pool.next_states[holdout_idx, :3]
                                - holdout_states[:, :3])
        holdout_mean, _holdout_aleatoric, _holdout_epistemic, holdout_total, holdout_score = (
            ensemble.predict_distribution(holdout_states, holdout_actions))
        internal_metrics = _channel_metrics(holdout_actual_delta, holdout_mean, holdout_total)
        internal_meta = [{"scenario_id": real_pool.scenario_names[int(real_pool.scenario_ids[i])],
                          "transition_index": int(i)} for i in holdout_idx]
        internal_rows = _prediction_rows(holdout_actual_delta, holdout_mean,
                                         holdout_score, internal_meta)

        # External validation on held-out days not used by the policy or dynamics fit.
        external_states, external_actions, external_next, external_rewards = [], [], [], []
        external_pred_next, external_pred_rewards, external_total_std = [], [], []
        external_meta = []
        counterfactual_rows = []
        for sid in scenario_ids:
            states, actions, next_states, rewards = _run_policy_trajectory(env, agent, sid)
            pred_next, pred_rewards, _done, disagreement, _mask = fake_env.step(
                states, actions, max_std=None, stochastic=False)
            mean_delta, _aleatoric, _epistemic, total_std, _score = ensemble.predict_distribution(
                states, actions)
            actual_delta = next_states[:, :3] - states[:, :3]
            external_states.append(states)
            external_actions.append(actions)
            external_next.append(next_states)
            external_rewards.append(rewards)
            external_pred_next.append(pred_next)
            external_pred_rewards.append(pred_rewards)
            external_total_std.append(total_std)
            external_meta.extend([{"scenario_id": sid, "step": int(t)} for t in range(len(states))])

            selected_steps = np.unique(np.linspace(
                0, env.horizon - 1, num=min(int(args.states_per_scenario), env.horizon), dtype=int))
            for t in selected_steps:
                for action_name, candidate in _counterfactual_actions(actions[t]):
                    replay_obs = env.reset(scenario_id=sid)
                    for prefix_action in actions[:int(t)]:
                        replay_obs, _r, _d, _i = env.step(prefix_action)
                    if not np.allclose(replay_obs, states[t], rtol=0, atol=1e-6):
                        raise RuntimeError(
                            f"Deterministic replay failed for {sid} step {t}; "
                            "counterfactual state reconstruction is not exact")
                    actual_nxt, actual_reward, actual_done, _info = env.step(candidate)
                    pred_nxt, pred_reward, pred_done, unc, accepted = fake_env.step(
                        replay_obs[None, :], candidate[None, :], max_std=None, stochastic=False)
                    pnext = pred_nxt[0]
                    counterfactual_rows.append({
                        "scenario_id": sid, "step": int(t), "action_type": action_name,
                        "action_tilt": float(candidate[0]), "action_azimuth": float(candidate[1]),
                        "actual_reward_kwh": float(actual_reward),
                        "predicted_reward_kwh": float(pred_reward[0]),
                        "reward_error_kwh": float(pred_reward[0] - actual_reward),
                        "actual_done": bool(actual_done), "predicted_done": bool(pred_done[0]),
                        "accepted_by_finite_mask": bool(accepted[0]),
                        "normalized_epistemic_disagreement": float(unc[0]),
                        "actual_next_power_norm": float(actual_nxt[2]),
                        "predicted_next_power_norm": float(pnext[2]),
                        "power_error_norm": float(pnext[2] - actual_nxt[2]),
                        "actual_next_dni_norm": float(actual_nxt[0]),
                        "predicted_next_dni_norm": float(pnext[0]),
                        "dni_error_norm": float(pnext[0] - actual_nxt[0]),
                        "actual_next_dhi_norm": float(actual_nxt[1]),
                        "predicted_next_dhi_norm": float(pnext[1]),
                        "dhi_error_norm": float(pnext[1] - actual_nxt[1]),
                    })

        states = np.concatenate(external_states, axis=0)
        actions = np.concatenate(external_actions, axis=0)
        next_states = np.concatenate(external_next, axis=0)
        pred_next = np.concatenate(external_pred_next, axis=0)
        rewards = np.concatenate(external_rewards, axis=0)
        pred_rewards = np.concatenate(external_pred_rewards, axis=0)
        total_std = np.concatenate(external_total_std, axis=0)
        actual_delta = next_states[:, :3] - states[:, :3]
        predicted_delta, _a, _e, _total, disagreement = ensemble.predict_distribution(states, actions)
        external_metrics = _channel_metrics(actual_delta, predicted_delta, total_std)
        reward_error = pred_rewards - rewards
        external_metrics["reward"] = {
            "rmse_kwh": _rmse(reward_error),
            "mae_kwh": float(np.mean(np.abs(reward_error))),
            "bias_kwh": float(np.mean(reward_error)),
        }
        external_metrics["known_kinematics"] = {
            "tilt_rmse_norm": _rmse(pred_next[:, 3] - next_states[:, 3]),
            "azimuth_rmse_norm": _rmse(pred_next[:, 4] - next_states[:, 4]),
            "tau_rmse": _rmse(pred_next[:, 5] - next_states[:, 5]),
        }
        external_metrics["ensemble_disagreement"] = {
            "mean_normalized_epistemic_std": float(np.mean(disagreement)),
            "p95_normalized_epistemic_std": float(np.quantile(disagreement, 0.95)),
        }
        cf_reward_error = np.asarray([r["reward_error_kwh"] for r in counterfactual_rows], dtype=float)
        cf_power_error = np.asarray([r["power_error_norm"] for r in counterfactual_rows], dtype=float)
        counterfactual_metrics = {
            "n_counterfactual_transitions": len(counterfactual_rows),
            "reward_rmse_kwh": _rmse(cf_reward_error),
            "reward_mae_kwh": float(np.mean(np.abs(cf_reward_error))),
            "reward_bias_kwh": float(np.mean(cf_reward_error)),
            "power_rmse_norm": _rmse(cf_power_error),
            "power_mae_norm": float(np.mean(np.abs(cf_power_error))),
            "mean_normalized_epistemic_disagreement": float(np.mean([
                r["normalized_epistemic_disagreement"] for r in counterfactual_rows])),
        }

        training_generation = _load_training_stats(run_dir / "episodes.csv")
        outdir.mkdir(parents=True, exist_ok=True)
        _csv(outdir / "internal_holdout_predictions.csv", internal_rows)
        _csv(outdir / "counterfactual_predictions.csv", counterfactual_rows)
        external_rows = _prediction_rows(
            actual_delta, predicted_delta, disagreement,
            [{"scenario_id": m["scenario_id"], "step": m["step"]}
             for m in external_meta])
        _csv(outdir / "external_on_policy_predictions.csv", external_rows)
        summary = {
            "validation_schema": 1,
            "checkpoint": str(checkpoint_path),
            "seed": seed,
            "scenario_protocol_sha256": split["protocol_sha256"],
            "validation_scenario_ids": scenario_ids,
            "model_architecture": {
                "num_models": m.num_models, "num_elites": m.num_elites,
                "hidden_dim": m.hidden_dim, "rollout_length": m.rollout_length,
            },
            "internal_whole_day_holdout": internal_metrics,
            "external_on_policy_one_step": external_metrics,
            "counterfactual_action_test": counterfactual_metrics,
            "training_synthetic_data_accounting": training_generation,
            "interpretation": [
                "Internal holdout metrics use whole NSRDB days reserved by ScenarioReplayBuffer and never used for ensemble fitting.",
                "External on-policy metrics use the separate policy-validation scenario partition.",
                "Counterfactual tests replay the same policy prefix to reconstruct an observed state, then execute alternative actions in the real simulator.",
                "The model predicts normalized DNI, DHI, and PV-power deltas; orientation and remaining horizon are reconstructed by the structured fake environment.",
                "Predictive interval coverage uses the ensemble Gaussian-mixture moment-matched standard deviation; it is a diagnostic, not a guarantee of calibrated uncertainty.",
                "This validates one-step prediction and action-conditioned reward reconstruction, not long-horizon free-running model accuracy.",
            ],
        }
        (outdir / "model_validation_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        _render_plots(outdir, external_rows, counterfactual_rows)
        return summary
    finally:
        env.close()


def _render_plots(outdir: Path, external_rows: list[dict], counterfactual_rows: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for channel, actual_key, predicted_key, label in (
        ("dni", "actual_delta_dni_norm", "predicted_delta_dni_norm", "DNI normalized delta"),
        ("dhi", "actual_delta_dhi_norm", "predicted_delta_dhi_norm", "DHI normalized delta"),
        ("power", "actual_delta_power_norm", "predicted_delta_power_norm", "Power normalized delta"),
    ):
        actual = np.asarray([float(r[actual_key]) for r in external_rows])
        predicted = np.asarray([float(r[predicted_key]) for r in external_rows])
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        ax.scatter(actual, predicted, s=9, alpha=0.35)
        lim = max(float(np.max(np.abs(actual))), float(np.max(np.abs(predicted))), 1e-6)
        ax.plot([-lim, lim], [-lim, lim], linestyle="--", linewidth=1)
        ax.set(xlabel=f"Actual {label}", ylabel=f"Predicted {label}",
               title=f"External held-out {channel.upper()} prediction")
        ax.grid(alpha=0.2); fig.tight_layout()
        fig.savefig(outdir / f"external_{channel}_predicted_vs_actual.png", dpi=170)
        plt.close(fig)
    if counterfactual_rows:
        errors = np.asarray([float(r["power_error_norm"]) for r in counterfactual_rows])
        uncertainty = np.asarray([float(r["normalized_epistemic_disagreement"]) for r in counterfactual_rows])
        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.scatter(uncertainty, np.abs(errors), s=18, alpha=0.45)
        ax.set(xlabel="Normalized ensemble disagreement", ylabel="Absolute counterfactual power error",
               title="Counterfactual error versus ensemble disagreement")
        ax.grid(alpha=0.2); fig.tight_layout()
        fig.savefig(outdir / "counterfactual_error_vs_disagreement.png", dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Trusted checkpoint; defaults to RUN_DIR/latest.pt")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--states-per-scenario", type=int, default=6)
    parser.add_argument("--model-epochs", type=int, default=None)
    parser.add_argument("--model-max-train-samples", type=int, default=None)
    parser.add_argument("--power-model", choices=("legacy", "pvwatts_sapm"), default="pvwatts_sapm")
    parser.add_argument("--wind-reference-height-m", type=float, default=2.0)
    parser.add_argument("--gamma-pdc", type=float, default=-0.004)
    parser.add_argument("--sapm-mount", default="open_rack_glass_polymer")
    parser.add_argument("--wind-powerlaw-exponent", type=float, default=0.14)
    args = parser.parse_args()
    if args.states_per_scenario < 1:
        parser.error("--states-per-scenario must be positive")
    run_dir = Path(args.run_dir)
    if args.checkpoint is None:
        args.checkpoint = str(run_dir / "latest.pt")
    if args.outdir is None:
        args.outdir = str(run_dir / "model_validation")
    summary = run_validation(args)
    print("Model validation written to:", args.outdir)
    print("External on-policy power RMSE:",
          summary["external_on_policy_one_step"]["power_norm"]["rmse"])
    print("Counterfactual power RMSE:", summary["counterfactual_action_test"]["power_rmse_norm"])


if __name__ == "__main__":
    main()
