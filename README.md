# MBPO — PV tracking

Model-Based Policy Optimization (MBPO) + SAC for single-axis PV tracking with pvlib physics.

**Documentation:** [docs/FULL_YEAR.md](docs/FULL_YEAR.md)

## Quick start

```bash
conda activate mbpo
pip install -e .

./train.sh run1 conf1 --cpus 4 --trial-cpus 2
./result.sh run1 --full
```

Outputs: `runs/run1/results/` (training plots + evaluation vs sun tracker and fixed mount).

## Core stack

- `mbpo/` — MBPO algorithm, PV environment, BNN model
- `softlearning/` — SAC, replay pool, samplers
- `examples/development/` — Ray Tune training entry
- `scripts/` — evaluation, verification, plotting

No dependency on Cursor or VS Code.
