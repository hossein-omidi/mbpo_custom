# PV representation remaster — unified result reporting

## Scope and installation

Target: `hossein-omidi/mbpo_custom`, `PostProcessing` commit `f0995849c98f07be2cbdab9202b49ab89ca50241`. This is a read-only source audit and an **additive reporting package**. It adds eight Python files to `PV_experiments/`; it does not replace or delete any trainer, simulator, evaluation, data, existing result, baseline, legacy plotting, or GitHub file. The older figure generators remain available for historical reproducibility but are deselected from the new canonical entry point. No backup archives or `.bak` files are created.

Extract `PV_representation_remaster.zip` in the repository root, retaining `PV_representation_remaster/`. Run:

```bash
python PV_representation_remaster/install.py --root . --dry-run
python PV_representation_remaster/install.py --root . --apply
python -m pytest -q PV_representation_remaster/tests
python -m PV_experiments.represent --help
```

The installer compares 22 actual source blobs to the audited branch (Git blob hashes; CRLF-only normalization accepted), parses all new Python files, refuses unknown versions or file replacement, and is idempotent. If the guard reports a difference, inspect that source rather than removing the guard. The installer is deliberately additive, not a new training patch. Your previously installed MC and experiment manager are prerequisites. `numpy`, `pandas`, `matplotlib` and existing project dependencies are required. No download or alternative weather dataset is introduced.

## Recommended commands

### Dataset — even before training

```bash
python -m PV_experiments.represent --kind dataset
```

Reads the actual current NSRDB catalogue and split builder; outputs `outputs/paper_experiments/dataset_catalog/v1/`. The managed version below verifies the dataset against the experiment's saved physical fingerprints.

### Existing managed experiments

Training and evaluation commands remain unchanged. Example:

```bash
python -m PV_experiments.train --name mc_final --seeds 0:4 --epochs 2000
python -m PV_experiments.evaluate --name mc_final --mode smoke
python -m PV_experiments.evaluate --name mc_final --mode model
python -m PV_experiments.evaluate --name mc_final --mode validation
```

Generate only the result family you need:

```bash
python -m PV_experiments.represent --name mc_final --kind training
python -m PV_experiments.represent --name mc_final --kind learning
python -m PV_experiments.represent --name mc_final --kind evaluation
python -m PV_experiments.represent --name mc_final --kind model
python -m PV_experiments.represent --name mc_final --kind all
```

`all` is strict: it includes dataset, training, learning, physical evaluation and model validation, and expects all those inputs. Learning requires at least two paired independent training seeds and two common evaluation checkpoints; for a pilot use enough episodes to reach the configured checkpoint schedule. It does not auto-train or evaluate a missing result. The default physical partition is `validation`. `--seeds 0:1` or `--seeds 0,2` selects a frozen seed subset explicitly. The mean/SD band is not produced when there is only one training seed. Use `--dry-run` to display selection and paths. Use `--tag v2` for a new report, because existing report directories are never overwritten.

### Advanced physical trajectories (one independent replay stage)

The existing evaluator intentionally stores per-day power means and SDs, not all raw control trajectories. No mathematically valid algorithm can recover exact cumulative-energy SD, paired intraday SD, or applied angular travel from these marginals alone. The new collector reuses the existing `09_monte_carlo_evaluate.run_one`, `load_policy`, real `SACEnvironment`, configuration loader and frozen split. A small observer delegates `reset` and `step` unchanged and records the existing metadata/actions. It independently checks energy, motion, UTC and baseline invariants. The collector is **not** an alternative training environment and never uses MBPO fake rollouts.

```bash
python -m PV_experiments.represent --name mc_final --kind collect
python -m PV_experiments.represent --name mc_final --kind trajectories
# Or generate all canonical figures, including already collected trajectories:
python -m PV_experiments.represent --name mc_final --kind all --with-trajectories
```

Default `representative` scope selects the middle chronological historical day in each of DJF, MAM, JJA and SON (at most four days, selected without using controller performance). All frozen weather draws and selected training seeds are replayed. `--scope full` is the entire selected evaluation partition; use it consistently for collection and rendering:

```bash
python -m PV_experiments.represent --name mc_final --kind collect --scope full
python -m PV_experiments.represent --name mc_final --kind all --with-trajectories --scope full --tag full_v1
```

The collector stores compressed NPZ files (one seed/day) with fields for all physical quantities and weather identities, then renders without replay. Work is checkpointed after each complete seed/day, with exact input/code fingerprints and an exclusive local lock. Re-running the exact collection resumes registered days or verifies completion. If a process is killed precisely between file publication and progress registration, it rejects that orphan rather than silently accepting it; inspect before removing just the incomplete unregistered file. A forcibly killed process may leave the named `.lock`; remove it only after confirming no collector is active. The full collector replays `S × D × M × 4 × 195` feasible simulator transitions plus the gross oracle; it can be expensive. Use representative scope first. `--trajectory-tag other` creates a separate collection if you change selection or code. No source evaluation results are overwritten.

### Final held-out result

After selecting hyperparameters using validation only:

```bash
python -m PV_experiments.evaluate --name mc_final --mode test --confirm-test
python -m PV_experiments.represent --name mc_final --kind evaluation --stage test --confirm-test
python -m PV_experiments.represent --name mc_final --kind collect --stage test --confirm-test
python -m PV_experiments.represent --name mc_final --kind all --stage test --with-trajectories --confirm-test
```

Test access requires explicit confirmation. `all` in test mode uses final test trajectories and energy statistics, but still uses the **validation** learning curves and **validation** model diagnostics (their own labels/manifest state this). The model-validation realization defaults to 0; `--realization-id N` selects an already managed result. Do not tune based on test plots.

## Output structure

```
outputs/paper_experiments/mc_final/
  runs/{sac,mbpo}_seedN/                 # unchanged
  evaluations/{validation,test}/seedN/ # unchanged
  evaluations/model/seedN/realization0/ # unchanged
  representation/validation/all_v1/
    dataset/     training/    learning/
    evaluation/  model/       trajectories/ (optional)
    representation_manifest.json
  representation_trajectories/validation/representative_v1/
    collection_plan.json
    collection_progress.json
    baseline_audit.json
    trajectory_manifest.json
    seedN/YYYY-MM-DD.npz
```

Each new rendering family exports PDF (vector plots where applicable), PNG (220 dpi), numerical tables, figure question/estimand/source manifest, and top-level provenance. Existing figures do not need to be deleted. No unverified sample results are included. All output paths are derived from the existing manager's experiment and seed discovery; no manual checkpoint paths.

## Figure selection and scientific intent

| Family | Canonical figures | Research question | Source |
|---|---|---|---|
| D01–D04 | Split coverage, monthly irradiation/temperature/wind, calendar GHI heatmap, radiation-regime distribution | What historical forcing and seasonal variation does the plant encounter? | Original native UTC NSRDB catalogue |
| T01–T02 | Exploratory return/critic/actor/entropy-coefficient diagnostics and accepted/used synthetic replay | Are optimization and model-generated experiences behaving as expected? | `episodes.csv` |
| L01–L03 | Controlled learning/paired difference, target with unreached seeds explicitly marked, paired AUC and measured compute cost | Is MBPO empirically more sample-efficient and at what computational cost? | Existing `sample_complexity.py` numerical outputs |
| E01 | Gross/net DC energy and actuator energy | Do generation gains survive movement consumption? | Daily MC records |
| E02 (month/season/year) | Net energy and effort | Do outcomes depend on calendar conditions? | Daily MC records |
| E03 | Paired daily-gain ECDF and gross-energy versus actuator-energy trade-off | Are gains consistent across historical days? | Daily MC records |
| E04 | Marginal intraday power, descriptive mixture spread versus seed SD | How do generation and variability evolve through the day? | Daily power means and sample SDs |
| E05 | Four representative seasonal power profiles, one seed and weather-draw SD | How sensitive is the same historical day's output to assumed perturbations? | Same daily power moments |
| M01 | One-step normalized delta RMSE versus persistence; action-counterfactual error versus ensemble disagreement | Does the learned model have prediction skill and useful action dependence? | Model validation outputs |
| R01–R02 | Actual selected-day power/tilt/azimuth and normalized action bands | How do the controllers move and why do they collect different power? | Recorded real paths |
| R03–R05 | Cumulative gross/net/actuator energy, paired power and net gaps, gross-only oracle deficit | When do energy and effort accumulate? How large is the remaining gross-performance gap? | Raw paths, paired before aggregation |
| R06–R09 | Orientation occupancy, diffuse-fraction-conditioned paired gains, monthly executed angular travel, day/time paired heatmap | Where and under what conditions do controllers behave differently? | Recorded real paths |

Many of these concepts originate in `advanced_suite.py`, `paper_reporting.py`, `scripts/plot_paper_eval.py`, `scripts/diagnose_tracking.py`, `scripts/verify_pv_state_space.py`, and the archived stratified statistics. The remaster keeps their most informative figure concepts while replacing the incompatible schema. The old full-year MC, TensorFlow/Ray loaders, stale `SAC_workspace.src.evaluation` imports, duplicate dashboards, ratio ladders, correlation-as-causal-power shares, and redundant 3D angle plots are not part of the canonical pipeline. Unconstrained astronomical ideal is also redundant with the required feasible astronomical controller and gross optimum. Architecture illustrations are outside scope.

### Statistical contract

Let `X[s,d,r,t]` be a response for training seed s, frozen historical day d, MC weather draw r, and time step t. Days and draws are equally weighted; each seed is independently trained, but all seeds share the same historical and weather realization grid. The code distinguishes:

* Sample SD across seed-level means, ddof=1. This is algorithm training variability, not a confidence interval, and is omitted with one seed.
* Empirical mixture SD across the crossed seed/day/draw set, ddof=0, describing the sampled distribution (including historical and assumed weather variability). The variance identity `V=Var_s(E[X|s])+E_s Var_d(E[X|s,d])+E_sd Var_r(X|s,d)` is checked for raw results. This is an ordered variance decomposition, not independent random-effects variance components.
* Within-day weather SD for a selected seed and fixed historical day (ddof=1 over M draws).
* For existing conditional mean `mu[s,d,t]` and sample SD `sd[s,d,t]`, the exact descriptive marginal pooled variance is the between-mean variation plus `(M-1)/M * mean(sd²)`. No assumptions about temporal or cross-controller correlation are introduced.
* Cumulative energy is calculated for each trajectory before aggregation. Paired differences are computed under the same seed/day/draw before statistics. No variance of a difference is reconstructed from marginal variances. Azimuth uses circular statistics and resultant length, not ordinary linear SD; the selected actual trajectories use scatter to avoid wrap artifacts.
* The existing sample-complexity estimator provides target-reaching steps, normalized common-range AUC, paired reduction, and bootstrap numerical intervals. The latter are sensitivity intervals over observed seed/day units, not established coverage for a larger weather population. Unreached runs are censored at observed budget in the plot, never assigned a fabricated reaching time. The existing estimator is run in a temporary output directory; its numerical outputs are retained, duplicate legacy images are not copied. `learning_report_notes.json` explains that canonical figures replace the image filenames referenced by the original estimator JSON.
* Month/season/year outputs are **mean daily energy on the selected held-out dates**, with actual coverage counts, not annual totals. Year is the episode anchor date, including December episodes ending on Jan 1. A full calendar-year yield would require a justified sampling design or complete independent-year evaluation; daily resets also omit overnight repositioning. No calendar extrapolation is performed. Dataset figures are original modelled NSRDB records, not weather-perturbed training scenarios or ground-truth measurements.
* E01 compares gross and net accounting for the same movement-cost-trained policies. It does not simulate or imply the performance of policies trained with a zero movement-cost reward. Such an ablation requires separate training.
* Relative gains use ratios of matched aggregate means, not averages of unstable low-energy daily ratios. Nonpositive reference means produce missing ratios.

### Baseline audit and fairness

Read the inspected sources: [current MC evaluator](https://github.com/hossein-omidi/mbpo_custom/blob/f0995849c98f07be2cbdab9202b49ab89ca50241/MBPO_workspace/workflows/09_monte_carlo_evaluate.py), [physical transition](https://github.com/hossein-omidi/mbpo_custom/blob/f0995849c98f07be2cbdab9202b49ab89ca50241/mbpo/env/pv_tracking.py), [reference geometry](https://github.com/hossein-omidi/mbpo_custom/blob/f0995849c98f07be2cbdab9202b49ab89ca50241/mbpo/env/baseline_contract.py), [shared power backend](https://github.com/hossein-omidi/mbpo_custom/blob/f0995849c98f07be2cbdab9202b49ab89ca50241/mbpo/env/pv_power_backend.py), [periodic evaluator](https://github.com/hossein-omidi/mbpo_custom/blob/f0995849c98f07be2cbdab9202b49ab89ca50241/SAC_workspace/src/training/experiment_protocol.py).

* Fixed: zero normalized commands, common reset tilt 30°, azimuth 180°. It is a well-defined fixed-mount comparator, not a site-optimized annual fixed tilt.
* Feasible astronomical: current decision-time zenith and azimuth, shortest wrapped azimuth change, same incremental action limits as SAC/MBPO, hold at night. It is a current-time controller, not a next-step predictive or optimal net-energy tracker. Solar position is predictable; a one-step look-ahead benchmark would be a separate stronger comparator, not silently relabeled as this existing baseline.
* SAC/MBPO: deterministic checkpoint policies in the same original environment, same starting pose, weather realization, UTC timestamps, power backend, actuator objective. No imaginary rollouts enter evaluation. The shared compact config enables movement penalty gain 1 in `geometry` mode.
* Net energy `E_net = sum(P_t/12000 - C_t)` (195 five-minute intervals, post-action DC watts). `C_t = gain * motor_power_W * (|delta_tilt|/tilt_rate + |delta_az|/az_rate)/3.6e6` kWh. It is based on executed, not requested, increments and assumes sequential motor operation. The observer verifies the formula, available interval time, valid poses, and action limits. It does not claim actual measured actuator consumption or include idle/holding losses.
* Instantaneous gross reference: the existing analytic `isotropic_max_poa_tilt` is used with the same perturbed DNI/GHI/DHI/albedo, solar geometry, temperature, wind and PVWatts/SAPM backend. Isotropic beam and sky/ground diffuse components are optimized over 0–90° tilt and sun-facing azimuth, then the existing `certify_pvwatts_monotonicity` certifies DC monotonicity. This is a valid gross-power bound only within the model's assumptions. The collector checks it against **every** feasible controller at every paired step. It is not an actuator-feasible oracle or an optimal sequential net-energy policy. Its movement cost and net energy are undefined, not zero. Old reference code rejects uncertainty-perturbed albedo by label; the independent adapter accepts both audited NSRDB recorded and explicitly perturbed labels and always passes the actual row value.
* The original current MC summaries are validated against frozen checkpoints, day/draw coverage, and common weather fingerprints. Gross/net arithmetic and power integrals are independently rechecked. Validation MC draw 0 must match the saved periodic fixed/astronomical energy references for every validation day and both algorithms. Full rich collection compares each reproduced daily energy, marginal power mean and sample SD to existing evaluation results; it records the minimum oracle margin, max motion time, max fixed cost and actual actuator parameters in `baseline_audit.json`.

### Verification status / remaining limitations

Read-only GitHub review pinned the branch head and exact source interfaces. Local tests used real NumPy/pandas/Matplotlib, synthetic data with actual current CSV shapes, mocked manager and environment entry points, and actual figure exports. They exercised nested moment identities, paired cumulative SD, UTC rollover, circular means, atomic output, installer guards, dataset/evaluation/training/learning/model renderers, observer cost checks and collection resume. **13 passed, 1 skipped** at package build. The skipped test uses the real repository's pvlib backend and analytical oracle against an orientation grid; pvlib and gym were not available in this runtime, and the real repository could not be cloned into the container. Run the tests in your actual project environment. This is not an assertion of trained-policy performance or completed physical evaluation. No training, real NSRDB full evaluation or real checkpoint trajectory collection ran here. The baseline source audit and regression tests are not a substitute for the real collector's fail-closed checks.


### Compatibility with locally edited Monte Carlo workflow (installer-only revision)

The installer still checks every audited source file and never modifies
`MBPO_workspace/workflows/09_monte_carlo_evaluate.py`. When that workflow's
hash differs, the installer now checks the **entire Python AST** against the
pinned audited file retrieved from your **local Git object database**. The
reference blob itself must have the original audited SHA-1. This accepts only
comment/whitespace/formatting-only differences, not executable Python changes.
Git is used read-only (`git show`); nothing is checked out or fetched.

If it still stops, your working file has an executable change or the audited
Git reference is unavailable locally. Do not delete or reset it. Review with:

```bash
git status --short -- MBPO_workspace/workflows/09_monte_carlo_evaluate.py
git diff -- MBPO_workspace/workflows/09_monte_carlo_evaluate.py
git diff f0995849c98f07be2cbdab9202b49ab89ca50241 -- MBPO_workspace/workflows/09_monte_carlo_evaluate.py
```

If the reference is missing, fetch it deliberately with
`git fetch origin PostProcessing`. Otherwise have the changed file reviewed
against the pinned reference before extending the installer. Do **not** bypass
this guard or overwrite the workflow. Neither the trainer nor any PV physics,
SAC, or MBPO source is modified by this package.
