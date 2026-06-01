# Stage 0 MBPO paper — commands

Config: `examples.config.pv_tracking.stage0_single_day_mbpo_paper`  
Use **`scripts/run_pv_eval.sh`** for monitor / midterm / final (runs steps in correct order).

```bash
cd /home/user01/mbpo_custom
conda activate mbpo

# While training
./scripts/run_pv_eval.sh monitor

# Quick snapshot (does not touch final report dir)
./scripts/run_pv_eval.sh midterm

# After training finishes
./scripts/run_pv_eval.sh final
```

**Movement cost error?** Run `evaluate_agent` first — `verify_movement_cost_fairness --eval-dir` needs `evaluation_summary.json` from that step. The script above does eval then verify.

Preflight (before train):

```bash
python scripts/verify_training_config.py --config examples.config.pv_tracking.stage0_single_day_mbpo_paper
python scripts/validate_pv_rollouts.py --config-path examples/config/pv_tracking/stage0_single_day_mbpo_paper.py
python scripts/verify_movement_cost_fairness.py --config-path examples/config/pv_tracking/stage0_single_day_mbpo_paper.py
```

Train:

```bash
python -m softlearning.scripts.console_scripts run_local examples.development \
  --config=examples.config.pv_tracking.stage0_single_day_mbpo_paper \
  --gpus=0 --trial-gpus=0 --cpus=14 --trial-cpus=6 \
  --temp-dir=$PWD/.ray_tmp/stage0_mbpo_paper
```
