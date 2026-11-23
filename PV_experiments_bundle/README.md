# PV SAC / MBPO — Portable test repair + managed experiments

Targets your existing `mbpo_custom` **PostProcessing** checkout **after** applying `PostProcessing_MC_patch` V2 and `PostProcessing_pipeline_review_patch`. **Incremental only**: no change to SAC, MBPO, pvlib, NSRDB data, physical reward, observation/action, sample-complexity mathematics, or Monte Carlo evaluator.

## 1. What went wrong in your test

The old `test_pipeline_statistics.py` contains a literal location from the tool developer's prior Linux container:

```python
zipfile.ZipFile('/mnt/data/PostProcessing_MC_critical_review_v2_no_backup.zip')
```

That ZIP is not expected to exist on your machine. The test failed **before** running any SAC, MBPO or pvlib code. Your separate physical environment test already reported `PASS: 195 post-action 5-minute UTC transitions` and therefore verified the smoke-test plant conditions in your own environment. The Gym warning is not the cause of this failure.

The installer replaces this faulty mock-installation test with a synthetic-source test using its **actual installer transformations**, rather than reading an external ZIP. It also repairs the two previously skipped tests that referenced old container file paths. The new tests check the modules installed in your checkout and fail clearly when a required module is absent. These are **tests**, not backups, and no old ZIP is downloaded or created.

## 2. Install safely — locally, no backups

Extract `PV_experiments_managed_runner_and_portable_test_fix.zip` to the **root** of `mbpo_custom` (so the extracted folder is `PV_experiments_bundle/`). From that root:

```bash
python PV_experiments_bundle/install.py --root . --dry-run
python PV_experiments_bundle/install.py --root . --apply
python -m pytest -q PostProcessing_MC_patch/tests
python -m pytest -q PostProcessing_pipeline_review_patch/tests
python -m pytest -q PV_experiments_bundle/tests
```

The installer adds a **new** `PV_experiments/` Python package and updates only the three known nonportable test functions in `PostProcessing_pipeline_review_patch/tests/test_pipeline_statistics.py`. Unknown or modified manager files are never overwritten. There are **no `.bak` files**, backup folders, Git commands, GitHub commits, hidden changes to existing learning code, or automatic deletion of old work. It uses short temporary files solely for atomic writes and removes them immediately.

## 3. Short experiment commands

All commands are run from the repository root. The first training command freezes its uncertainty JSON, epoch budget and planned seed set in a named experiment folder. These settings cannot subsequently be changed under the same experiment ID.

### Training (both algorithms and five paired seeds, automatically)

```bash
python -m PV_experiments.train --name mc_final --seeds 0:4 --epochs 2000
```

SAC and MBPO are trained separately, with exactly the same frozen NSRDB scenario split, uncertainty assumptions, and seed set. Runs are stored automatically at:

- `outputs/paper_experiments/mc_final/runs/sac_seed0/`, …, `sac_seed4/`
- `outputs/paper_experiments/mc_final/runs/mbpo_seed0/`, …, `mbpo_seed4/`

The default uncertainty file is the existing *hypothetical* `PostProcessing_MC_patch/examples/hypothetical_uncertainty_5pct.json`. To change it, specify `--uncertainty-spec PATH` **only when starting a new experiment name**. The default device is CPU; specify `--device cuda` on initial training when a supported CUDA device and matching PyTorch installation are present. Model overrides `--model-epochs` and `--model-max-train-samples` are also frozen when the experiment is created. The algorithm trainer, optimizer, and replay code are unchanged.

You may first use a small **separately named** pilot experiment:

```bash
python -m PV_experiments.train --name mc_pilot --seeds 0:1 --epochs 50
```

Do not reuse a pilot experiment name for different hyperparameters or budgets. This protects the scientific comparison and prevents accidental mixing of checkpoints.

### Evaluation (automatically find all completed seed pairs)

```bash
python -m PV_experiments.evaluate --name mc_final --mode smoke
python -m PV_experiments.evaluate --name mc_final --mode model
python -m PV_experiments.evaluate --name mc_final --mode validation
python -m PV_experiments.evaluate --name mc_final --mode test --confirm-test
```

- **smoke:** two **validation** days, three MC draws per day by default. Never reports these as full-split results.
- **model:** MBPO learned-model validation using a frozen realization (default ID 0); run additional IDs with `--realization-id 1` etc.
- **validation:** all validation days, 100 weather draws per day by default.
- **test:** entire frozen test partition, 100 draws per day, only when explicitly confirmed; requires a completed smoke test for each selected seed. Do not tune from test results.

Outputs are automatically placed in `outputs/paper_experiments/mc_final/evaluations/<mode>/seedN/` with logs, result CSVs, figures and saved JSON. The model-validation outputs are further separated by realization ID.

### Reporting and figures (discover all SAC/MBPO runs)

```bash
python -m PV_experiments.report --name mc_final --kind learning
python -m PV_experiments.report --name mc_final --kind monte-carlo --stage validation
python -m PV_experiments.report --name mc_final --kind monte-carlo --stage test
```

`learning` invokes the existing multi-seed sample-complexity procedure with all matching SAC/MBPO run paths, automatically. It preserves the previous bootstrap bands, generates separate **mean ± sample SD across training seeds** and **mean ± sample SD across historical validation days**, and reports real environment transition counts (not synthetic MBPO rollouts).

`monte-carlo` aggregates results across trained seeds, **after** each seed has been evaluated on the same held-out scenarios and random-weather draw IDs. It exports across-seed energy means/SDs, paired MBPO–SAC differences, time-resolved pooled power means/SDs and a shaded mean ± SD power curve. The SD is calculated across **independently trained seed-level means**, never by treating all MC draws as independent training runs. It does not describe this band as a confidence interval.

### Inspect current status

```bash
python -m PV_experiments.status --name mc_final
```

### Optional flags

- `--seeds 0:1` or `--seeds 0,2,4`: only a subset of the **pre-registered** seed set, if starting with a larger plan.
- `--draws 500`: more Monte Carlo realizations, provided the selected evaluation has not already created results with another draw count.
- `--resume`: resume a partially completed training run from its exact existing `latest.pt`; never fabricates a checkpoint or silently deletes an incomplete run.
- `--dry-run`: preview the commands without writing a new experiment.
- `--realization-id 1`: validate MBPO counterfactual predictions under another **fixed** weather realization.

## 4. What it automatically creates

```
outputs/paper_experiments/mc_final/
  experiment.json                  # immutable experiment settings and seed plan
  uncertainty_spec.json            # immutable chosen uncertainty input (not a backup)
  preflight/environment_audit.json # per-experiment physical verification
  jobs/                             # each command, status, return code, UTC times
  logs/                             # live-captured output for every action
  runs/sac_seed0/                   # generated by the unchanged SAC trainer
  runs/mbpo_seed0/                  # generated by the unchanged MBPO trainer
  evaluations/smoke/seed0/
  evaluations/model/seed0/realization0/
  evaluations/validation/seed0/
  evaluations/test/seed0/
  reports/sample_complexity/
  reports/monte_carlo_validation/
  reports/monte_carlo_test/
```

Run artifact paths and metadata are created automatically. The manager does **not** discover and silently adopt unrelated historical runs from other output folders: doing so could mix different physics and uncertainty configurations. Use a fresh experiment name for new managed experiments. Existing work is never deleted.

## 5. Safety and scientific checks

- The frozen uncertainty input is hashed and checked on every invocation.
- The training budget and planned seed list are fixed per experiment.
- Run metadata must indicate completion and matching epoch budget, checkpoints and scenario-split files.
- Paired SAC and MBPO runs are checked for matching physical simulator fingerprint, historical partition, horizon, dimensions, movement cost and objective.
- Historical train, validation and test scenario IDs must be nonempty and pairwise disjoint.
- The sample-complexity routine receives matched training seeds and real-interaction checkpoints.
- Monte Carlo comparisons require the same scenario protocol, draw count and uncertainty specification across all seeds.
- Report input files are SHA-256 fingerprinted: a changed source result will **not** silently reuse an outdated figure.
- Existing completed jobs and reports are skipped only when compatible; incomplete ones cause a clear error and remain untouched.
- Actual execution failures are logged and propagated; Python does not treat a failed subprocess as success.
- Training interruption terminates a child process instead of leaving an unnoticed trainer running after the manager exits.

**Verification scope:** the manager underwent isolated unit and subprocess orchestration tests, including exercising two mock SAC/MBPO seed pairs and actual file generation. These validate path handling and control flow, not PV physical accuracy or converged RL performance. The real `PostProcessing` environment verification must be run on your machine. Previous log indicates that it passed; this manager verifies it under each frozen named experiment before the first managed training command.
