# PV statistical integrity + POA-only representation (v4)

## Installation and scope
Apply at repository root using `unzip -o solar_pv_verified_statistics_poa_only_patch_v4.zip -d .`. Back up locally edited files beforehand. Patch is additive where possible; changes evaluation orchestration, preserving legacy reference-controller code. No trainer, actor, dynamics model or pvlib solar physics change.

## Default visible comparison
`MBPO_workspace/representation_config.json` is the central representation selector. The supported verified profile shows: MBPO, SAC, fixed mount, instantaneous POA maximizing gross isotropic power. Both conventional trackers are omitted from evaluated methods by default in the SAC/MBPO workflow (but their original code remains available for legacy usage). POA is per-timestamp unconstrained ideal and cannot claim feasible net movement cost or optimal finite-horizon net reward.

## Verified statistics
Figures and summary files are under `<outdir>/verified/`. Day boxplots and descriptive sample SD quantify observed variability; no 95% CI under the original one-date-per-bin sampling scheme. Independent date-selection replications are required for sampling-design CIs, and independent training seeds for training stochasticity. See `MBPO_workspace/docs/VERIFIED_STATISTICAL_PROTOCOL.md`.

## Run
```
python -m MBPO_workspace.workflows.03_evaluate --mbpo-checkpoint MBPO_workspace/runs/mbpo_seed0/latest.pt --sac-checkpoint SAC_workspace/runs/sac_seed1/latest.pt --all-scenarios --days-per-month 3 --selection-seed 42 --outdir MBPO_workspace/evaluation/verified
python -m SAC_workspace.workflows.04_evaluate --checkpoint SAC_workspace/runs/sac_seed1/latest.pt --all-scenarios --days-per-month 3 --selection-seed 42 --outdir SAC_workspace/evaluation/verified
python -m unittest discover -s MBPO_workspace/tests -p 'test_*.py' -v
python -m unittest discover -s SAC_workspace/tests -p 'test_*.py' -v
```
The profile requires actual POA optimization: omit `--skip-poa-opt`.

## Git ignore safety
`.gitignore` was updated by **appending new patterns to the remote repository's existing ignore rules**, not replacing prior exemptions. Ignores runtime `runs` and `evaluation`, checkpoints and binary weights. Does not glob-ignore all PNGs/PDFs: manuscript figures remain versionable. Ignored tracked files remain tracked; for existing oversized committed files inspect `git ls-files`, `git status --short`, then narrowly use `git rm --cached -- <file>` if you wish to untrack without deleting local bytes. Avoid bulk deletion.

## Scientific limitations
- For the existing stratified selection plan (one day selected uniformly per temporal bin), a weighted finite-manifest mean is unbiased, but finite-population sampling variance is NOT identified from one observation in each bin. Any year-resampling interval is explicitly exploratory, not an actual coverage-guaranteed CI.
- Weather-class exploratory comparisons are poststratified and not calibrated to known class population weights.
- The default 117 five-minute steps are a control window, not full 24-hour production days.
- The energy model's movement penalty is zero by default: simulated net = gross, not measured actuator electricity.
- No long real-NSRDB training/evaluation was performed in this patch build. Local mock/unit smoke checks cannot certify absence of all runtime bugs.

## Uploaded legacy findings
`evaluate_fullyear_mc.py`'s `ci95` uses Gaussian 1.96 SEM regardless of small sample/stratification; returns zero when n<=1. Both are invalid inferential summaries for the new sampling design. `evaluate_agent_advanced.py`'s mean ± std time curves correctly represent replicated-trajectory spread, but fixed date + fixed initial pose + deterministic policy yields degenerate repeats, not uncertainty. `plot_training_progress.py` uses ±std from Ray reported batches and should remain descriptive. The sun-zenith tracking-direction gates in `diagnose_tracking.py` are not appropriate objectives for a maximized POA oracle.
