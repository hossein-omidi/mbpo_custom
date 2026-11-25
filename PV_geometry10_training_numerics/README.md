# Geometry10 training numerics — minimal final update

Audited branch: `PostProcessing`, commit `3bd5873f01eefe0a7652c6e4444ca296994c75de`.
The previous statistical-integrity update is already integrated in this commit.

## Scope

Two changes only: (1) preserve the sampled standard-normal noise in the shared SAC squashed-Gaussian log probability instead of recovering it by subtracting nearly equal float32 numbers; (2) correct the sign of persistence improvement in the dynamics skill gate and align the external diagnostic and its validator. Near zero persistence RMSE the latter is a regularized improvement score, not an ordinary relative skill ratio. Positive means strictly smaller RMSE; zero means equality. No changes to the physical simulator, state representation, action bounds, model architecture, hyperparameters, training budget, evaluation partition, or statistical estimators.

The installer validates the exact original and revised Git blob hashes of the four active source files before writing; unexpected modifications fail closed. It adds one regression test module. It does not modify the Git index, delete prior outputs, or create backups. Dry-run first and apply only to a clean, inactive checkout. Files are staged and replaced individually, not a transaction across the whole filesystem.

## Install (from repository root)

```bash
python PV_geometry10_training_numerics/install.py --root . --dry-run
python PV_geometry10_training_numerics/install.py --root . --apply
git diff --check
python -m pytest -q geometry10_tests/test_training_numerics.py -rs
python -m pytest -q geometry10_tests
```

The real-environment test requires the existing Gym/PVLib environment, local NSRDB manifest and the existing uncertainty specification. A skip is not a full integration pass. It tests both azimuth wrap directions, tilt saturation, terminal handling, physical power continuity and reward accounting. No new dependencies are required beyond the project's existing test/runtime dependencies.

## Training provenance

The source fingerprint changes. For a new experiment use a new name and train both algorithms from scratch under the same source identity. Do not resume earlier runs, mix their outputs with new runs, or rewrite their provenance. Preserve existing completed experiments on their original pinned source; the discovered numerical edge cases do not establish that any existing run is invalid.

Suggested pilot (100 episodes per seed, enough to pass warmup; model utilization is not guaranteed):

```bash
python -m PV_experiments.train --name geom10_numerics_pilot --seeds 0:1 --epochs 100 --device cpu
python -m PV_experiments.evaluate --name geom10_numerics_pilot --mode smoke --draws 3
```

Inspect model-fit attempts/completions, skill-gate passes, actual synthetic batch use, finite losses, energy accounting and validation reports before launching a full run. An unfitted or unused model is not evidence of a successful model-based training intervention. If using a GPU, choose the supported device explicitly and keep settings consistent across algorithms. Use validation only for development; reserve the final test partition for the frozen experiment.

```bash
python -m PV_experiments.train --name geom10_numerics_final --seeds 0:4 --epochs 2000 --device cpu
```

## Verification limits

See VERIFICATION.txt. Source replacements were verified against full GitHub blobs; the local environment could not clone the entire checkout or run actual NSRDB/PVLib/Gym integration. The artifact does not claim full training convergence, optimal hyperparameters, or complete end-to-end validation. Run the complete tests on the target machine before expensive experiments.
