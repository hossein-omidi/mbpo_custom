# Stage 1 clearsky (pointer)

Full **A→Z** procedure, directory names, and gates:

**[TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md)**

| Item | Value |
|------|--------|
| Config | `examples/config/pv_tracking/0.py` |
| Train | `mbpo run_local … --config=examples.config.pv_tracking.0` |
| Eval outdir (only) | `evaluation/pv_stage1_clearsky_summer` |
| Hold-out dates | `2020-06-07`, `2020-06-21`, `2020-07-15`, `2020-08-01` |
| Gates | `scripts/diagnose_tracking.py --eval-dir evaluation/pv_stage1_clearsky_summer --gate` |

Prerequisite: Stage 0 gates pass on `evaluation/pv_stage0_single_day/`.
