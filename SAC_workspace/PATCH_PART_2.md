# Patch 2/3 — Independent, model-free SAC training

**Apply Part 1 before Part 2.** Only files inside `SAC_workspace` are changed.

```bash
python -m unittest discover -s SAC_workspace/tests -p 'test_sac_math.py' -v
python -m SAC_workspace.workflows.03_train --run-dir SAC_workspace/runs/seed0 --epochs 2
# full run (use new run-dir):
python -m SAC_workspace.workflows.03_train --run-dir SAC_workspace/runs/sac_seed0 --seed 0
```

The `--epochs 2` run is a smoke check and contains NO SAC gradient updates until
8000 initial exploration transitions are collected. Use a clean run directory
for a full run. Each episode is 117 transitions; `--epochs` is absolute target.

No BNN, dynamics networks, fake_env, or imaginary rollout buffer are imported.
Only use trusted, locally created checkpoint files for `--resume`.

Part 3 adds paired NSRDB evaluation against the fixed mount and dual-axis tracker.
