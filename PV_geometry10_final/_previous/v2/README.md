# Geometry10 v2 — in-place upgrade for managed Torch SAC and MBPO

**Base:** `hossein-omidi/mbpo_custom`, branch `PostProcessing`, exact commit `92a1b00e5a0cbd5f770514396c0a347f5b9aca6d`. This supersedes the sibling-workspace installer. Do not run both installers.

## Install directly into your existing checkout (WSL/Linux)

Extract this ZIP into the repository root. Activate your existing Python environment. Use an LF checkout, with no tracked local edits. This installer reads exact committed Git blobs, validates working-tree bytes and source replacement counts, compiles all generated Python source, constructs one Git patch, and checks it with `git apply --check` before making changes. It does **not** make a new checkout, copy data, make backup files, stage files, commit, push, or touch experiment outputs. New source files are checked against Git ignore rules. The installer is intentionally single-use on the audited base, not a migration for an already patched tree.

```bash
cd ~/projects/PVLIB_Paper/mbpo_custom
git status --short
git rev-parse HEAD
python PV_geometry10_inplace_v2/install.py --root . --dry-run
python PV_geometry10_inplace_v2/install.py --root . --apply
git status --short
git diff --stat
python -m pytest -q geometry10_tests
```

If the dry run fails, do not force installation: provide the traceback and `git rev-parse HEAD`. Do not apply to a dirty checkout. Use Git to review/commit the listed source files after tests pass. The patch directory itself need not be committed. `git diff` excludes newly created untracked files until staged; `git status --short` lists them. If rollback is needed, use your existing Git history to restore the changed tracked files and remove only the newly generated files listed by the installer. Existing result directories must not be deleted.

## Observation contract and scientific rationale

The active managed Torch pipeline uses `conf_geometry10.py` by default. The original `conf_compact.py` and native environment's legacy modes remain as historical definitions; the patched managed actor, replay and dynamics are intentionally Geometry10-only, not arbitrary-dimensional or dual-mode. Use the original Git revision for six-dimensional training/checkpoint replay.

| Index | Feature | Normalization |
|---:|---|---|
| 0 | DNI | /2000 W/m², bounded |
| 1 | DHI | /2000 W/m², bounded |
| 2 | Current DC power | /2000 W |
| 3 | Panel tilt | /90° |
| 4,5 | Panel azimuth | sine, cosine |
| 6 | Solar zenith | /180° |
| 7,8 | Solar azimuth | sine, cosine |
| 9 | Remaining episode fraction | (H−t)/H |

Current power is a measured *pre-action* quantity, useful as feedback for hidden temperature/albedo/wind effects and as the anchor for the power-delta prediction. The reward uses the subsequent generated power after the action. No future measured weather or electrical power is available to the policy. Retaining power is a controlled design choice, not a proof of optimality. The state remains partially observable. The new schema refuses actual measured power outside [0,2000] W, instead of silently clipping it and breaking synthetic reward reconstruction; this must be reconfigured consistently if panel capacity ever exceeds that normalization range. The native geometry mode rejects independent observation noise that would corrupt unit-circle or solar-context identities; existing correlated weather uncertainty remains enabled.

## MBPO mathematical and interface audit

The actor is 10→2 (Gaussian parameters before squashing); each Q network receives state plus action, 12 inputs. The learned weather distribution predicts two deltas from six inputs: DNI, DHI, solar zenith, solar sine/cosine, tau. Actions, panel orientation and power do not enter the weather predictor. The conditional power distribution predicts one delta from 14 inputs: current 10D state, action 2D, next irradiance 2D. True next irradiance is only a supervised conditional-training input; generated transitions use *predicted* next irradiance. Power prediction uses current solar geometry and is a one-step approximation, not an exact physical power calculation. Both ensemble members and scaler dimensions are checked. No change is made to the Gaussian ensemble training objective or persistence-skill gates.

The next panel tilt is clip(beta+5*a_beta,0,90), azimuth is (phi+10*a_phi) mod 360. Executed movement is computed after clipping, and azimuth uses the shortest circular displacement. The real and fake environment share actuator energy parameters and normalization. The remaining integer step is derived from tau and must lie on the exact horizon grid; the terminal state has tau=0. Replay rows retain the originating scenario ID. The fake environment caches only deterministic solar ephemeris from the same location, date, UTC grid and pvlib convention as the real environment, verifies current geometry, and selects next-step geometry. It does not read future irradiance, temperature or power. Scenario IDs are not policy inputs. One-step rollouts remain mandatory. Finite-precision angular reconstruction is checked with explicit tolerances. The synthetic reward is predicted next normalized power × 2000 × interval_hours / 1000 minus the existing executed-motion energy cost. Nonfinite raw model parameters/outputs cannot pass silently through clipping; invalid predictions are rejected or fail explicitly. This is still a learned approximation to the real physical transition, and model uncertainty/accuracy must be measured.

The active managed flow is `conf_geometry10.py → SACConfig / MBPOConfig → PVTrackingEnv / StageAPVTrackingEnv → SACEnvironment → actor, critic, replay → MBPO ensemble / StructuredFakeEnv → periodic evaluation → checkpoints → physical Monte Carlo evaluation → PV_experiments representation / advanced reports`. The model-validation workflow passes correct scenario context for external on-policy and counterfactual rollouts and reports circular azimuth and solar geometry error. Evaluation continues in the physical simulator, not in the synthetic model. Physical movement objective, original weather assumptions, scenario partitions, action limits, horizon, baselines, optimizer settings, model size/depth (except inputs) and training budget are unchanged. No added smoothness penalty or residual RL. The shared geometry schema and observation source fingerprints, dynamics schema, saved dimensions, and experiment/run mode metadata prevent mixing old and new artifacts. The preflight audit also records its observation mode.

## Local validation gates

Run the dedicated `geometry10_tests` suite after installing. The 11 pure numerical tests validate bounds, power inversion, wrap continuity, circular error, scenario/time context, terminal handling and action clipping. The six local integration tests cover actual config resolution, actor update and checkpoint round trip, ensemble fit/checkpoint/replay mixing, experiment CLI/manifest, synthetic reward/invalid prediction checks, and a paired physical rollout against native compact mode on a TRAIN-only NSRDB day. The physical test uses a stub delta solely to isolate exact transition/reward reconstruction; it is not a model skill test. All six require the installed checkout and some require pvlib/Gym/local NSRDB data. They have not been executed on the real checkout in the preparation environment. Do not start full training unless they pass. Some preexisting tests explicitly assert six-feature shapes or mock the old fake-env API; those are historical compact-contract tests and are not migrated by this patch. The archived Ray/TensorFlow paths are not certified. Also, the pinned Git tree lacks `SAC_workspace/src/evaluation/evaluation.py` imported by older standalone evaluators (likely affected by `evaluation/` ignore rules). The managed Monte Carlo evaluator does not depend on that missing module. The patch does not fabricate its contents or certify that standalone legacy route.

## Pilot after validation

Use fresh experiment names, never resume a 6D run. Old checkpoints contain different actor/ensemble shapes and source/config fingerprints. Old outputs remain in the repo but are intentionally incompatible with the new managed mode; revisit the original Git commit to replay those policies. The initial exploration budget is 8000 real steps; 100 epochs gives 19500 real steps per algorithm per seed, a useful pilot, not proof of convergence or superiority.

```bash
python -m PV_experiments.train --name geom10_pilot --seeds 0:1 --epochs 100 --dry-run
python -m PV_experiments.train --name geom10_pilot --seeds 0:1 --epochs 100
python -m PV_experiments.evaluate --name geom10_pilot --mode smoke --draws 3
python -m PV_experiments.evaluate --name geom10_pilot --mode validation --draws 3
# Run only after the MBPO model is fitted; inspect episode/model skill and synthetic accounting.
python -m PV_experiments.evaluate --name geom10_pilot --mode model
python -m PV_experiments.represent --name geom10_pilot --kind all
python -m PV_experiments.advanced_report --name geom10_pilot --kind all
python -m PV_experiments.represent --name geom10_pilot --kind collect
python -m PV_experiments.represent --name geom10_pilot --kind trajectories
```

Do not tune using the held-out test partition. A controlled 6D-versus-10D comparison requires matched seeds, real interaction budgets and weather assumptions. The new observation alone cannot guarantee superiority over astronomical tracking, elimination of chattering, or a fully Markov model. Model validation, learning curves, actual accepted synthetic transitions and individual control trajectories are necessary evidence. The present audit is a source-level and numerical verification, not a full execution certification.

## Preparation verification record

The branch HEAD and all original source pins were checked through GitHub; additional replacement anchors were checked against the pinned source. Eleven schema tests and four disposable Git installer-mechanics tests passed in the preparation environment. Python compilation of the package sources and tests passed. The Git tests exercised direct application, dry-run nonmutation, a conflicting hunk (no files modified), tracked-dirty refusal, and ignored-new-source refusal. To repeat the installer-only tests from the extracted directory before applying, run `python -m pytest -q PV_geometry10_inplace_v2/tests/test_installer.py`. Those fixture tests do not constitute a dry run against your actual checkout; that remains mandatory. The package contains no actual NSRDB data or trained model files. The full repository checkout, pvlib and Gym were unavailable in the preparation environment; full source transformation, real-data integration, training and representation must be verified locally as described above.
