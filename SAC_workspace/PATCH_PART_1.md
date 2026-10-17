# Patch 1/3 — MDP foundation

Contains the independent SAC configuration reader, compact observation validator,
minimal existing PVTrackingEnv adapter, workflow 01–02, and contract unit tests.

Apply at your existing repository ROOT. Does not change `mbpo/` or `softlearning/`.

```bash
python -m unittest discover -s SAC_workspace/tests -p 'test_contract.py' -v
python -m SAC_workspace.workflows.01_setup_config
python -m SAC_workspace.workflows.02_environment_replay
```

The published GitHub branch does not include the generated NSRDB manifest.
If it is absent locally, first run the existing repository catalog preparation:
`python scripts/prepare_nsrdb_newyork_catalog.py` (do not use `--force`).
No synthetic or clear-sky fallback is allowed.

Part 2 supplies the independent SAC agent and training loop. Part 3 supplies
pvlib dual-axis evaluation, plotting and paired statistical summaries.
