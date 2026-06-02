# Stage 0 MBPO paper run (cloudy historical day)

**Config:** `examples.config.pv_tracking.stage0_single_day_mbpo_paper`  
**Day:** `2020-06-21` · **Weather:** `historical` (PVGIS TMY) · **Steps:** 78 · **Movement penalty:** from config (`0.0009` inherited)

Use the helper script and full protocol:

| Task | Command |
|------|---------|
| Preflight + train | `./scripts/run_stage0_paper_trial.sh train` |
| Training plots | `./scripts/run_stage0_paper_trial.sh plot` → `training_plots/stage0_mbpo_paper/` |
| Eval (inherit config) | `./scripts/run_stage0_paper_trial.sh eval` → `evaluation/pv_stage0_mbpo_paper_movement/` |
| Gate | `./scripts/run_stage0_paper_trial.sh gate` |

**Do not** override eval with `--eval-weather-source clearsky` — training uses historical irradiance.

Copy-paste workflow for all stages: [PV_SIMPLE_WORKFLOW.md](PV_SIMPLE_WORKFLOW.md)  
Full A→Z: [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md)
