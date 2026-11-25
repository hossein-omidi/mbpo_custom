# Geometry10 — finalized in-place installer

**Audited base:** `hossein-omidi/mbpo_custom`, `PostProcessing`, commit `84322e2df3b786904631edfbaae84cf8b5d5aae3`. This package supersedes `Geometry10_Integrated_Audit_Patch.zip` and includes its complete source transforms plus the release hardening below. It does not reinstall Geometry10 from the older compact-state revision `92a1b00`.

## Apply

In your existing Linux/WSL Git repository, extract the ZIP (it contains `PV_geometry10_final/`) and run:

```bash
python PV_geometry10_final/install.py --root . --dry-run
python PV_geometry10_final/install.py --root . --apply
git diff --check
git status --short
```

The installer uses the audited base Git objects and exact source anchors, generates a single Git patch in memory, compiles every generated Python file, and requires `git apply --check --whitespace=error-all` before writing. It accepts the exact original source, the exact previous integrated patch output, or the exact final output (already applied). It rejects divergent edited files and unchanged active dependencies that differ from the audited checkout. It never makes backups, creates a sibling workspace, resets, stages, commits, or pushes. It modifies only the listed source and test files; existing datasets and run outputs remain untouched. Do not run the old `integrated.py` directly. For a manually generated unified diff, use `--write-patch /tmp/geometry10_final.patch` instead of `--apply`, then `git apply --check` and `git apply` that file. Native Windows locking is not supported; use WSL/Linux.

## Changes included

- All previous integrated fixes: recoverable epoch-zero checkpoint, exact chronological committed CSV prefix, periodic evaluation transaction markers, inherited exclusive job locks, raw nonfinite sample checks, terminal-producing rollout roots, shared confirmed-attainment semantics and exact scheduled grid, source/data/version provenance, preflight identity, model and Monte Carlo report provenance, diagnostic-only feasibility envelope, empirical hierarchical predictive quantiles with isolated diagnostic RNG, shared-weather one-step counterfactual diagnostics, actual replay accounting, and explicit deprecation of broken legacy CLIs.
- New release checks: checkpoint algorithm, seed, epoch and interaction count validated inside the actual loaded trusted checkpoint; expected digest sent to both managed evaluators; actual loaded policy identity returned and checked before sealing. Seal all evaluation CSV/JSON source artifacts and refuse changed files. For training, check committed optimizer update count, snapshot metadata, final evaluated actor equality with final checkpoint, exact completed evaluation grid, and hashes of committed report inputs.
- New diagnostics: preflight scripted nonzero motion, coverage by held-out historical day and DNI+DHI regime, equivalent-executed-pose prediction differences, synthetic replay contribution at confirmation or censoring budget. Keep the existing statistical files and both managed reporting paths.

## Deliberate scientific decisions

Keep Geometry10, SAC equations, structured one-step MBPO, physical reward, model skill gate, 85% configured real replay, and current perturbed historical-weather protocol. No hard physical feasibility filter is activated: the envelope is an operational diagnostic using assumed minimum cell temperature -60 C, GHI residual allowance 200 W/m2 and albedo upper bound 1, not a proven support bound. Finite boundary clipping retains the existing distribution; nonfinite samples fail fast. Do not use future actual meteorology for training. Hierarchical validation uses 128 samples per state; coverage is descriptive and the ensemble disagreement score remains distinct from total predictive SD. Immediate reward regret and equivalent-pose diagnostics are not guarantees of long-horizon optimality.

Use **new experiment names**. Even an existing exact previous-patch training run must not be resumed under the new source contracts. No old outputs are silently migrated. The report's nominal-weather evaluation, no-synthetic ablation, baseline changes, historical partition expansion, and test-yield equivalence/noninferiority margin are not software fixes: they require separate documented scientific decisions and matching comparisons. The present managed primary estimand remains conditional on the frozen perturbed weather protocol; the nominal preflight is not a nominal policy evaluation. Validation-target attainment does not establish test equivalence. A reported 100-epoch single paired-seed pilot is not evidence of a general sample-efficiency advantage. Use more paired seeds and never tune against test outcomes.

## Verify in your actual environment

```bash
python -m pytest -q geometry10_tests
python -m PV_experiments.train --name geom10_final_pilot_v2 --seeds 0:1 --epochs 100
# Interrupt between checkpoints, then resume using the same frozen name:
python -m PV_experiments.train --name geom10_final_pilot_v2 --resume
python -m PV_experiments.evaluate --name geom10_final_pilot_v2 --mode smoke --draws 3
python -m PV_experiments.evaluate --name geom10_final_pilot_v2 --mode validation --draws 10
python -m PV_experiments.evaluate --name geom10_final_pilot_v2 --mode model
python -m PV_experiments.report --name geom10_final_pilot_v2 --kind all
python -m PV_experiments.represent --name geom10_final_pilot_v2 --kind all
python -m PV_experiments.advanced_report --name geom10_final_pilot_v2 --kind all
```

Check recovery from interruptions during episode append and each evaluation artifact write in separate pilots, including a nonmultiple-final-epoch budget. Compare subsequent scenario stream, actor and evaluation metrics to an uninterrupted same-runtime reference (excluding wall time). The initial checkpoint follows environment setup, so an interruption before initial checkpoint creation still requires an explicit fresh run. Unlogged partial interactions cannot be reconstructed. Completed-history corruption is not silently repaired. An incomplete standalone evaluation is rejected rather than silently overwritten. The 100-epoch pilot gives only two scheduled evaluations and is not a convergence study; use longer separately named pilots if necessary. Check `mbpo_replay_at_confirmation_or_budget.csv`, model-validation summary and `model_control_diagnostics_by_seed.csv` before final training.

## Verification actually performed

The uploaded prior integrated package and example installer were inspected. New original source anchors and blob identities were checked using GitHub at the audited revision; source transformations have exact guards. Python compilation and **24 package-level data-free/real-Git fixture tests passed** including wrong-seed/wrong-epoch checkpoint with a matching digest, altered output files, final snapshot mismatch, recovery history update counts, replay exposure at confirmation/censoring, pose aliases, stratified coverage, and installation from the base or prior-patch fixture. These tests do not establish full physical-simulator or training integration. Network/DNS restrictions prevented materializing the complete Git repository locally; the final full-source `git apply --check`, existing Geometry10 tests, NSRDB/PVLib/Gym runtime and multi-seed interruption tests must be executed in the user's checkout. The separate runtime verification record cited in the new audit was not provided; its pilot numbers were not independently reproduced. Do not launch final 2,000-epoch runs until the local acceptance gates pass.

Package tests from the extracted installer directory:

```bash
cd PV_geometry10_final
PYTHONPATH="$PWD:$PWD/payload" python -m pytest -q tests/test_installer.py payload/geometry10_tests/test_audit_support.py payload/geometry10_tests/test_audit_release.py
```
