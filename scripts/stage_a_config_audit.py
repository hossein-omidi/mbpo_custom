#!/usr/bin/env python3
"""Read-only audit of standalone SAC/MBPO settings and training budget."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SAC_workspace.src.config.config import load_config as load_sac_config
from MBPO_workspace.src.config import load_config as load_mbpo_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None,
                        help="Absolute final epoch; defaults to configured 2000.")
    parser.add_argument("--power-model", choices=("legacy", "pvwatts_sapm"),
                        default="pvwatts_sapm")
    parser.add_argument("--wind-reference-height-m", type=float, default=2.0)
    parser.add_argument("--gamma-pdc", type=float, default=-0.004)
    parser.add_argument("--sapm-mount", default="open_rack_glass_polymer")
    parser.add_argument("--wind-powerlaw-exponent", type=float, default=0.14)
    args = parser.parse_args()
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be positive")

    physics = dict(power_model=args.power_model,
                   wind_reference_height_m=args.wind_reference_height_m,
                   gamma_pdc=args.gamma_pdc, sapm_mount=args.sapm_mount,
                   wind_powerlaw_exponent=args.wind_powerlaw_exponent)
    sac = load_sac_config(require_data=False, **physics)
    mbpo = load_mbpo_config(require_data=False, **physics)
    if sac.to_dict() != mbpo.sac.to_dict():
        raise SystemExit("ERROR: SAC and MBPO do not share identical SAC/MDP configuration")

    a, m = sac.algo, mbpo.model
    epochs = int(a["epochs"] if args.epochs is None else args.epochs)
    horizon = 117
    steps = epochs * horizon
    warmup = int(a["start_steps"])
    updates_per_step = int(a["updates_per_step"])
    sac_updates = max(0, steps - warmup) * updates_per_step
    model_fits = max(0, steps // int(m.model_train_freq)
                     - warmup // int(m.model_train_freq))
    effective_model_pool = min(
        int(m.model_pool_capacity),
        int(m.model_retain_epochs) * int(m.rollout_batch_size) * int(m.rollout_length)
    )
    report = {
        "power_model": args.power_model,
        "same_sac_config_for_sac_and_mbpo": True,
        "epochs_requested_absolute": epochs,
        "config_default_epochs": int(a["epochs"]),
        "transitions_per_episode": horizon,
        "real_environment_transitions": steps,
        "random_exploration_transitions": min(steps, warmup),
        "random_exploration_equivalent_episodes": round(warmup / horizon, 2),
        "estimated_sac_gradient_updates": sac_updates,
        "replay_capacity_transitions": int(a["replay_size"]),
        "replay_capacity_equivalent_episodes": round(int(a["replay_size"]) / horizon, 2),
        "configured_eval_interval": a.get("eval_interval"),
        "standalone_trainer_runs_periodic_validation": False,
        "checkpoint_interval_episodes": int(a["checkpoint_interval"]),
        "sac": {
            "batch_size": int(a["batch_size"]),
            "updates_per_real_step": updates_per_step,
            "learning_rate": float(a["lr"]),
            "gamma": float(a["gamma"]),
            "reward_scale": float(a["reward_scale"]),
            "tau": float(a["tau"]),
            "hidden_sizes": list(a["hidden_sizes"]),
            "alpha_init": float(a["alpha_init"]),
            "min_alpha": float(a["min_alpha"]),
            "target_entropy": float(a["target_entropy"]),
        },
        "mbpo": {
            "real_ratio": float(m.real_ratio),
            "synthetic_ratio": 1.0 - float(m.real_ratio),
            "ensemble_models": int(m.num_models),
            "elite_models": int(m.num_elites),
            "hidden_dim": int(m.hidden_dim),
            "rollout_length": int(m.rollout_length),
            "model_train_frequency_transitions": int(m.model_train_freq),
            "estimated_model_fits_after_warmup": model_fits,
            "rollout_batch_size": int(m.rollout_batch_size),
            "effective_model_pool_capacity": effective_model_pool,
            "model_epochs_per_fit": int(m.model_epochs),
            "model_patience": int(m.model_patience),
            "model_batch_size": int(m.model_batch_size),
            "max_model_train_samples": int(m.model_max_train_samples),
            "whole_day_holdout_fraction": float(m.model_holdout_fraction),
        },
        "notes": [
            "--epochs is the absolute final episode count, not additional episodes.",
            "SAC update count is estimated from the current trainer schedule; the checkpoint records the exact count.",
            "MBPO fit count is estimated from its once-per-117-step schedule after warm-up.",
            "The standalone trainer does not run validation just because eval_interval exists in config.",
            "movement_penalty=0 means logged net reward equals gross DC energy, not actuator-adjusted net generation.",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
