# MBPO Remaster v1 — Complementary Reporting Patch

**Apply after:** `solar_sac_v2_complete_verified.zip` and
`solar_mbpo_remaster_v1_complete.zip`. Extract the ZIP into the **repository root**.

This patch intentionally changes only the evaluation/reporting layer. It does
not change the original MBPO code, SAC mathematics, pvlib/NSRDB simulator,
MBPO's probabilistic ensemble, replay buffers, train hyperparameters, or any
legacy `scripts/` plotting source.

## Two existing files extended

1. `MBPO_workspace/workflows/03_evaluate.py` — automatically produces the
   advanced CSV and figures after its existing reports.
2. `SAC_workspace/src/evaluation/evaluation.py` — retains seven additional
   **already available** simulator `info` fields during evaluation (DNI, DHI,
   GHI, solar altitude, weather condition, and episode weather diagnostics).
   It does **not** change environment dynamics or SAC training.

## New files

- `MBPO_workspace/src/paper_reporting.py`: validated rich CSV exporter,
  plotting algorithms adapted from legacy project figures, offline regeneration.
- `MBPO_workspace/workflows/04_advanced_report.py`: regenerates advanced
  figures from `advanced/trajectories_rich.csv` without simulation or policy loads.
- `MBPO_workspace/tests/test_advanced_reporting.py`: scientific validation and
  PDF/2D/3D regression tests.
- `MBPO_workspace/docs/REPORTING_ADAPTER.md`: source attribution, exact commands,
  requirements, scientific caveats.

## Run

```bash
python -m unittest discover -s MBPO_workspace/tests -p 'test_*.py' -v
python -m unittest discover -s SAC_workspace/tests -p 'test_*.py' -v

python -m MBPO_workspace.workflows.03_evaluate \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_seed0/latest.pt \
  --sac-checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
  --outdir MBPO_workspace/evaluation/paired

python -m MBPO_workspace.workflows.04_advanced_report \
  --eval-dir MBPO_workspace/evaluation/paired
```

Existing evaluation outputs are not deleted. Rich outputs are created under
`MBPO_workspace/evaluation/paired/advanced/`.

See `docs/REPORTING_ADAPTER.md` for scientific interpretations and limitations.
