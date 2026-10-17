# Patch 3/3 — Dual-axis pvlib evaluation + results

**Apply Parts 1 and 2 first.** Uses the original project's
`dual_axis_tracking_targets`, `compute_panel_power_w`, NSRDB scenario
manifest and unmodified PVTrackingEnv. No MBPO learned dynamics/policies.

```bash
python -m unittest discover -s SAC_workspace/tests -v
python -m SAC_workspace.workflows.04_evaluate \
  --checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
  --training-csv SAC_workspace/runs/sac_seed0/episodes.csv \
  --outdir SAC_workspace/evaluation/seed0
```

The pvlib documentation at
https://pvlib-python.readthedocs.io/en/stable/gallery/solar-tracking/plot_dual_axis_tracking.html
uses solar zenith/azimuth for its dual-axis mount. This patch uses the
same geometry through the repository's existing physics functions; it does
not introduce ModelChain temperature/inverter assumptions absent from the
existing environment.

`dual_axis_slew_limited`: a realizable comparator that can move only at the
same action rate as SAC; its control has privileged access to sun position.

`dual_axis_instant_ideal`: an unconstrained instantaneous sun-pointing
reference, NOT a realizable trajectory and NOT a proven global POA optimum.
Its actuator energy is undefined, therefore its **net energy is not reported**.
Its gross energy is shown separately. Physical net-energy comparisons include
SAC, fixed-no-motion and slew-limited dual-axis only.

Default evaluation days are taken from the reference config and may overlap
training: evaluation is *paired, in-distribution*, NOT held-out generalization.
For a true held-out study, both learning procedures would need their training
scenario samplers filtered identically before retraining.

The checkpoint reader accepts trusted LOCAL `.pt` files only.

Full multi-year benchmark: add `--all-scenarios` to `04_evaluate`; this
uses all native NSRDB scenario IDs in the existing manifest.
