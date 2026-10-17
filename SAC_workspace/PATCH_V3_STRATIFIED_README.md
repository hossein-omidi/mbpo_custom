# Calendar-stratified NSRDB evaluation add-on — installation and scope

Unzip this patch from the repository root **after** SAC v2 and MBPO Recovery
v2 have been installed. It changes only evaluation/reporting code, never any
SAC/MBPO actor, critic, ensemble, physical simulator, NSRDB weather inputs or
existing training checkpoints.

```bash
unzip -l solar_pv_calendar_stratified_evaluation_patch_v3.zip
unzip -o solar_pv_calendar_stratified_evaluation_patch_v3.zip -d .
python -m unittest discover -s SAC_workspace/tests -p 'test_*.py' -v
python -m unittest discover -s MBPO_workspace/tests -p 'test_*.py' -v
python -m SAC_workspace.workflows.06_plan_scenarios --outdir SAC_workspace/evaluation/sample_plan
```

See `SAC_workspace/docs/STRATIFIED_EVALUATION_PROTOCOL.md` for the full
experimental design, CLI options, figures, inference caveats and examples.

**Compatibility**: The patch intentionally does NOT include
`SAC_workspace/src/evaluation/plots.py`: your preexisting rich/paper plotter
is kept byte-for-byte unchanged. MBPO's plot wrapper fills date/season
metadata where absent, and the real SAC evaluator emits the same fields.
Default five-scenario evaluation still works. `--scenarios` still selects
explicit IDs. `--all-scenarios` NOW means a deterministic sample of three
spaced dates per month-year; `--exhaustive-scenarios` restores the original
complete catalog evaluation. `--selection-plan` reuses exact matched dates.

Science: mean net energy from a 9h45m window is not 24h or annual yield;
movement_penalty=0 does not measure physical motor electricity. Neither
uniformly sampled nor even observed-unseen NSRDB days guarantee new-year
independent generalization. The reported year-cluster bootstrap ranges are
exploratory, not formal finite-population design-based confidence intervals.

Validation in patch workspace: synthetic 2557-date selection (252 sampled),
method pairing, weighted mean, offline 2D/PDF, POA weather reuse, original
SAC and MBPO tests; full native NSRDB training/evaluation **not run here**.
