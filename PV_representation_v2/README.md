# PV representation V2 — sample-efficiency and calendar reports

## Scope
Additive follow-up to the representation remaster. Three new modules in `PV_experiments/` (`advanced_report.py`, `advanced_learning.py`, `calendar_heatmaps.py`); no changes to existing training, physical environment, model fitting, weather generation, checkpoint or evaluation algorithms. No Git mutations or backups; the installer may use read-only `git show` to verify a comment-only local workflow variant. The earlier `PV_experiments.represent` modules were not present in the inspected public `PostProcessing` tree and the earlier ZIP was not accessible in this runtime; this update does not assume they have been installed or replace them. Use the new command as the canonical output for learning and calendar plots, rather than publishing duplicate legacy figures. Other figure families continue to use the previous `represent` command if installed.

The installer checks exact source blobs from the reviewed `PostProcessing` branch (9 interfaces). The directly inspected files include `MBPO_workspace/src/sample_complexity.py`, `MBPO_workspace/src/curve_statistics.py`, both trainers, `SAC_workspace/src/training/experiment_protocol.py`, `MBPO_workspace/workflows/09_monte_carlo_evaluate.py`, `PV_experiments/core.py`, `report.py` and `evaluate.py`. The old numerical run loader is reused; V2 computes a censor-aware report from those validated periodic evaluation records. The existing estimator's outputs are not overwritten, and its retrospective target onset remains available in V2.

## Installation
Extract the archive in repository root, preserving folder `PV_representation_v2/`:

```bash
python PV_representation_v2/install.py --root . --dry-run
python PV_representation_v2/install.py --root . --apply
python -m pytest -q PV_representation_v2/tests
python -m PV_experiments.advanced_report --help
```

No source is overwritten. If a hash differs, review that interface instead of disabling the guard. Requires existing managed workflow and NumPy, pandas, Matplotlib (3.7+ for outside legend layouts). No additional packages or retraining are normally necessary. Python 3.10+.

## Use

```bash
# After completed paired training (uses periodic validation CSVs; no final MC required):
python -m PV_experiments.advanced_report --name mc_final --kind learning

# After full validation MC evaluation:
python -m PV_experiments.evaluate --name mc_final --mode validation
python -m PV_experiments.advanced_report --name mc_final --kind calendar

# Both in one command (requires both input families):
python -m PV_experiments.advanced_report --name mc_final --kind all

# Final held-out calendar figures, after selecting settings on validation only:
python -m PV_experiments.evaluate --name mc_final --mode test --confirm-test
python -m PV_experiments.advanced_report --name mc_final --kind calendar --stage test --confirm-test
```

Default stage is validation. `--seeds 0:4` is inclusive; default is all frozen experiment seeds. `--tag v3` creates a new report without overwriting earlier output. `--dry-run` shows path and selected seed set. No need to specify checkpoint paths, CSV paths or runs manually. Reports are written atomically under `outputs/paper_experiments/<name>/representation/<stage>/advanced_<kind>_<tag>/` with `learning/`, `calendar/` and a source-hash manifest. Existing directories are rejected, never silently overwritten. `--kind all --stage test --confirm-test` still uses validation for the learning part. Use the previous `represent` entry point for dataset, model and trajectory families, if installed. Do not interpret test results as a hyperparameter tuning input.

## Scientific treatment of sample complexity

The current periodic evaluator evaluates deterministic policies on the same frozen validation days and weather realization 0. SAC and MBPO use the same recorded gradient-update budget at each original environment-transition count. The exact checkpoint schedule is reconstructed from `final_epoch`, horizon, and `evaluation_interval_episodes` (including the final checkpoint); the new report refuses missing or unexpected checkpoints rather than silently using their intersection. Deliberately resumed runs with extra off-schedule evaluations are also rejected; reconcile their experimental protocol before comparison. The reported target uses the saved reference fraction and the same validated baseline energies. If the astronomical mean is not above the fixed mean, the relative threshold is undefined and the command fails; a separate preregistered absolute target protocol would be required, not an after-the-fact target change.

For a training seed, define the validation mean over identical historical days at each checkpoint. The frozen target is `fixed_mean + target_fraction * (astronomical_mean - fixed_mean)`. The existing code returns the onset of a sustained crossing. With a sustain rule of two checkpoints, crossing at 10k and 20k is only confirmed at 20k; the report exports both `onset_steps` and `confirmed_steps`. It uses confirmation for the primary sample-saving calculation. No interpolation or extrapolation. This is a finite-budget attainment event, not a guarantee of permanent convergence.

* **Conditional percentage reduction:** `100*(1 - N_MBPO/N_SAC)` per seed pair where both confirmed; mean and sample SD of these values (SD undefined for one pair). Report number of eligible pairs. A negative reduction means MBPO used more samples. Speedup `N_SAC/N_MBPO` is also exported; it is not the same as percentage saving. This does not estimate unconditional reduction if some seeds failed.
* **All-seed fixed-budget comparison:** `R_alg=min(N_alg,B)` with nonconfirmed runs assigned observed budget B solely for the restricted-cost estimand. Report mean paired `R_SAC-R_MBPO` and percentage `100*(1-mean(R_MBPO)/mean(R_SAC))`. This is a measure of capped sample expenditure up to confirmation, not an imputed time-to-target. If both methods fail, a zero saving only means the same measured budget was exhausted. Always read attainment fractions alongside it.
* **Learning performance:** mean and sample SD across independent training-seed means on frozen validation days; paired differences are formed before aggregation. These are descriptive bands, not calibrated confidence intervals, and do not measure uncertainty across within-day MC draws. AUC is normalized over the first through last observed scheduled evaluation, not extrapolated to zero interactions. Report paired AUC gain and final performance gain separately. Five or more independent paired seeds recommended; two is only a minimum.
* **Compute:** sum logged episode and periodic validation time. MBPO model fitting and generation are already included inside episode durations and are not added again. Confirmation time is compared only for both-confirmed pairs. No claim of hardware-independent speedup. Startup, checkpoint I/O and one-time baseline construction are not included. Real simulator training transitions count as samples; separately report periodic validation transition counts (not training samples). Synthetic generated/accepted transitions and minibatch draws are separate; logged draws may reuse samples. V2 exports actual synthetic batch fraction at every checkpoint; it does not infer that nominal ratio was achieved.

Publication figures (PDF + 300 dpi PNG), each with CSV data and figure manifest:

| ID | Question |
|---|---|
| L01 | Mean ± seed SD net energy against external transitions; paired MBPO–SAC difference; frozen target and baseline levels |
| L02 | Fraction of all seeds confirming the target versus external transitions; individual confirmed/censored results |
| L03 | Conditional percentage reduction for both-confirmed pairs and all-seed restricted-step savings |
| L04 | Paired common-budget AUC and final validation performance |
| L05 | Logged computation versus sample budget and conditional time to confirmed target |

`sample_efficiency_summary.json` provides both sample-saving definitions, status counts, confirmed fractions (counts and denominator), paired AUC/final gain summaries and compute summary; `sample_complexity_by_seed.csv`, `paired_sample_reduction.csv`, `target_attainment.csv`, `learning_curve.csv`, `logged_time_by_checkpoint.csv` and `mbpo_replay_accounting.csv` retain the underlying numerical quantities.

## Calendar heatmaps

Two layouts: month × day-of-month (12×31 across historical years) and historical year × month (7×12 for the 2018–2024 catalogue). Four controller panels with equal-sized cells and a common numerical color scale. Each cell is the equal-day mean of seed- and MC-draw-averaged daily energy. The year×month layout annotates cell values. The code exports net and gross DC generation and actuator energy, and zero-centered signed paired net differences (MBPO−SAC and MBPO−astronomical). Separate date-count heatmap. No plot uses a separate rescaled colormap for each tracker. Physical zero is a color-coded value, gray is a valid but unevaluated date, and white is a nonexistent calendar date. February 29 retained as a valid calendar position. The CSV records evaluated and catalogue day counts, descriptive sample SD across seed means and day means where defined, and the sum over evaluated daily means (not a month total unless complete coverage). The primary color measure is kWh per evaluated day, not total monthly or annual energy; frozen held-out partition does not represent all calendar dates. Never extrapolate or fill dates. Day is the historical episode anchor date in UTC, including the post-action next-day hour. All four feasible trackers share one physical model and objective. The gross-only instantaneous oracle is deliberately excluded from net comparisons, as it has no feasible actuator cost.

Calendar loader reuses managed validation and provenance routines, rejects stale checkpoint hashes, inconsistent weather fingerprints, incomplete scenario/draw/controller grids, changed run physics, inconsistent fixed/astronomical gross/net/cost across seeds, nonzero fixed actuator energy, and violations of gross minus movement equals net. On validation data it also verifies periodic baseline energies against MC realization zero. It does not rerun physical models; rich trajectory/oracle checks remain in the earlier remaster.

## Verification and boundaries

Ten local pytest regression tests passed using synthetic data and current documented source interfaces. Tested complete/incomplete sustained crossings; both/one/neither method reaches; all-failure plots; exact paired SD; incomplete checkpoint rejection; time sums; replay accounting; calendar weighting/leap/missing/zero cases; provenance/energy rejection; PNG/PDF generation; guarded installer and atomic report publishing. The learning fixture test uses the actual old estimator's `_load_run` when installed in the real checkout; here it used a mock because network clone, old remaster archive and trained artifacts were unavailable. The previous remaster source was not present at its expected branch paths. Therefore this is an audited additive update, **not** a complete runtime re-certification of the previous package or of the trained policies. No actual reduction percentage has been established; actual experiment results are needed. Example images supplied separately use clearly marked synthetic data and are not publication evidence.


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
