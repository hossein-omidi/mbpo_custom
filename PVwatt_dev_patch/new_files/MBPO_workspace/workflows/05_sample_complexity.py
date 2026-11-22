"""Aggregate paired multi-seed SAC/MBPO periodic evaluations.

Example:
python -m MBPO_workspace.workflows.05_sample_complexity \\
  --sac-runs SAC_workspace/runs/sac_seed0 SAC_workspace/runs/sac_seed1 \\
  --mbpo-runs MBPO_workspace/runs/mbpo_seed0 MBPO_workspace/runs/mbpo_seed1 \\
  --outdir MBPO_workspace/analysis/sample_complexity
"""
from __future__ import annotations

import argparse
from MBPO_workspace.src.sample_complexity import analyze_sample_complexity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sac-runs", nargs="+", required=True,
                        help="SAC run directories; use the same seed set as MBPO")
    parser.add_argument("--mbpo-runs", nargs="+", required=True,
                        help="MBPO run directories; use the same seed set as SAC")
    parser.add_argument("--outdir", default="MBPO_workspace/analysis/sample_complexity")
    parser.add_argument("--target-fraction", type=float, default=None,
                        help="Override the pre-registered target fraction in run metadata")
    parser.add_argument("--target-kwh", type=float, default=None,
                        help="Optional pre-registered absolute validation net-energy target")
    parser.add_argument("--sustain-checkpoints", type=int, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=None,
                        help="Override the pre-registered bootstrap replicate count")
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    report = analyze_sample_complexity(
        args.sac_runs, args.mbpo_runs, args.outdir,
        target_fraction=args.target_fraction, target_kwh=args.target_kwh,
        sustain_checkpoints=args.sustain_checkpoints,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print("Sample-complexity analysis:", args.outdir)
    print("Target:", report["target_kwh_per_episode"], "kWh/episode")
    for row in report["algorithm_summary"]:
        print(row["algorithm"], "final=", row["final_mean_net_energy_kwh"],
              "reached=", row["target_reached_seeds"], "/", row["n_training_seeds"],
              "mean_steps=", row["mean_real_steps_to_target_among_reached"])
    print("Paired sample reduction:", report["sample_reduction"])


if __name__ == "__main__":
    main()
