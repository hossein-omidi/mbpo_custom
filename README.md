# MBPO — PV tracking

Model-Based Policy Optimization (MBPO) + SAC for single-axis PV tracking with pvlib physics.

**Full guide (setup, testing, training, evaluation, customization):** [docs/GUIDE.md](docs/GUIDE.md)
**Protocol notes:** [docs/FULL_YEAR.md](docs/FULL_YEAR.md)

## Quick start

```bash
conda activate mbpo
pip install -e .

./train.sh run1 conf3 --cpus 4 --trial-cpus 2 --verify
./result.sh run1 --full
```

Outputs: `runs/run1/results/` (training plots + evaluation vs sun tracker and fixed mount).

## Core stack

- `mbpo/` — MBPO algorithm, PV environment (pvlib + NSRDB), BNN ensemble model
- `softlearning/` — SAC, replay pool, samplers (trimmed to the used subset)
- `examples/development/` — Ray Tune training entry; `examples/config/pv_tracking/` — named configs
- `scripts/` — evaluation, verification, plotting, data preparation
- `tests/` — pytest suite (`pytest tests/ -q`)

No dependency on Cursor or VS Code.
