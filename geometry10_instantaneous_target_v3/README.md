# Geometry10 Instantaneous Target v3 — branch-independent installer

This installer applies the **v3 scientific changes with the original patch's Unicode encoding repaired**. No scientific code or action semantics were changed by the encoding repair. It does not check,
require, create, switch, or modify a Git branch. It does not check `HEAD` or
run `git status`.

## Install

1. Extract the ZIP directly into your repository workspace.
2. From the repository root, run:

```bash
python geometry10_instantaneous_target_v3/install.py --dry-run
python geometry10_instantaneous_target_v3/install.py --apply
```

The installer automatically finds the repository root from its own location.
If you extracted it elsewhere, supply the repository path with `--root`.

## What branch-independent means

You may run it from any branch. Compatibility is determined by
`git apply --check` against the files currently in your working tree—not by
the branch name or commit ID.

The current source files must still contain the code/context expected by v3.
If your branch has incompatible edits, the dry-run stops without changing
files. This is intentional: a patch cannot safely be applied to arbitrary,
unrelated source code merely because it is on a different branch.

## Installer behavior

- Checks the included encoding-repaired patch's SHA-256 to detect an altered or corrupted patch.
- Detects when the exact patch is already applied.
- Runs `git apply --check` before applying.
- Does not inspect Git status, branch name, or `HEAD`.
- Does not create backups, switch branches, commit changes, or modify the
  Git index. It changes only the working-tree files touched by the patch.

## After installation

```bash
git diff --check
python -m pytest -q geometry10_tests
python check_geometry10_components.py
python -m PV_experiments.train --name target_v3_smoke --seeds 0 --epochs 1 --device cpu
python -m PV_experiments.evaluate --name target_v3_smoke --mode smoke --draws 2
```

Use a fresh experiment name and checkpoints created with v3. The managed
`PV_experiments` workflows are the verified experiment entry points for the
specified project revision.
