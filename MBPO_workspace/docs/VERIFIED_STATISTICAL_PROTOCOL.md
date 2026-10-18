# Statistically honest figures and POA-only representation

## Estimand and unit
The target is the mean energy over **eligible manifest dates**, for the standardized 117-step midday-to-evening interval, not 24 h and not out-of-distribution future weather. Paired differences must be computed within the same date, seed, initial pose and UTC timeline. Repeated control steps (117) are NOT 117 independent observations. Replicated policy actions on the same weather date also do not create new independent weather dates.

## Weighted point estimate
With mutually exclusive temporal bins h, size N_h, one uniform random observed day Y_h per bin and total eligible days N, the unbiased expansion estimator is μ̂ = (Σ_h N_h Y_h)/N. For paired differences replace Y_h by Y_h(A)−Y_h(B), evaluated on the same scenario. All year/month figures repeat this weighting inside the relevant subset.

## Critical uncertainty limitation
With one observation per bin, the within-bin sampling variance S²_h is **not estimable**. Consequently a design-based standard error or 95% CI is NOT identifiable for the existing single-selection-plan experiment. `sample_sd_across_days` is a valid descriptive statistic across observed dates, not the standard error of μ̂ and not a confidence interval. Legacy ±SD lines describe variability, not uncertainty in estimated means. Gaussian `1.96 * SEM` from 117 temporally correlated steps, a pooled set of weather days, or repeated fixed-date deterministic rollouts is not justified. Year-cluster bootstrap across only seven years is exploratory and does NOT account for unmeasured within-bin sampling uncertainty.

To estimate sampling uncertainty without changing the sampler, run R >= 2 genuinely independently seeded **date-selection plans** covering the *same eligibility frame*, evaluate each plan under frozen identical policy checkpoints, calculate one design-weighted estimate per plan, and use the sample SD of the R independent *estimate values*, divided by sqrt(R), with a Student-t critical value (R−1 df). This is a Monte-Carlo repeated-design approximation; report selection seeds and R. For year-generalization, use independent held-out source years and avoid claiming the finite-manifest CI covers new years. For retraining uncertainty, train multiple policy seeds; use seed as independent training replicate and report separately from day-sampling uncertainty, preferably with hierarchical modeling when substantial.

## Gross and net
For physical controllers, net = gross − *modeled* movement cost. With movement_penalty=0 net = gross by construction, **not physical motor electricity**. The instantaneous POA reference maximizes per-step `compute_panel_power_w` under the project's isotropic model; it is gross-only and unconstrained, cannot be compared as net. The POA optimum need not face the sun because the sky/ground diffuse contributions influence optimal tilt; azimuth maximizes beam for the isotropic sky model. `pvlib.irradiance.poa_components` combines components but does not perform optimization.

## Figure semantics
- Bars: population-weighted point estimates, **no synthetic uncertainty whiskers**.
- Boxplots: sampled-day empirical distribution (quartiles, median, mean marker), NOT the population CI and NOT design weighted.
- Matched scatter: per-day paired difference; horizontal mark = weighted mean.
- Monthly lines: weighted means across selected dates per month; no inference band.
- Plot selectors: configured active methods MBPO, SAC, fixed, instantaneous POA oracle. Both classical tracking references remain implemented but are hidden in the default representation. Legacy Ray/TensorFlow plots remain unchanged.

## Legacy audit
`evaluate_fullyear_mc.py` computes `ci95` using 1.96 × s/sqrt(n) and returns zero for n=1; **zero uncertainty for n=1 is invalid**. `plot_training_progress.py` labels ±std over evaluation episodes correctly as SD, but not CI. `evaluate_agent_advanced.py` uses `np.std` default ddof=0 on replicates: appropriate as population description of observed replicates, not as unbiased sample SD or mean uncertainty. On fixed date+fixed orientation+deterministic policy, identical curves necessarily have zero spread: do not show this as evidence of confidence. `diagnose_tracking.py` uses tilt-vs-zenith and sun-pointing sign gates that are not principled performance targets for a POA-optimizing controller; disable these gates for POA research.

## Git safety
`.gitignore` prevents **new untracked** outputs from entering git. It does not untrack already committed data. Inspect with `git ls-files` and untrack only the desired exact paths via `git rm --cached -- <path>`; NEVER blindly run `git rm -r` on valuable datasets. Confirm using `git status --short` before committing. Do not erase local datasets or checkpoint directories.


## v4.1 audit corrections (2026-10-17)
- The original `summary.json` mistakenly used a naive scenario-bootstrap 95% interval
  when a stratified selection plan was in effect. With one draw per nontrivial bin,
  that is not a valid design CI. Under a plan, `summary.json` now writes
  design-weighted means and paired day contrasts, and makes its CI fields null.
- A partially missing metric must not be interpreted as a full-population mean:
  all selected bins must be present with finite results for the relevant
  method/metric/subgroup; otherwise the corresponding estimate is unavailable.
- The monthly paired-gains plot must use each selected date's population bin
  weight, not the equal-weight mean of sample dates. Replicate policy seeds
  are averaged within a date before computing the weighted monthly gap.
- All compared methods must contain the identical set of day/seed pairs, and
  duplicate method/day/seed records are rejected before drawing conclusions.
- For an exhaustive census of eligible manifest dates, finite-population
  *sampling* error is exactly zero; this says nothing about training-seed
  variation, weather-year generalization, simulation-model error or measurement
  error. Otherwise, one draw per nontrivial bin does not identify a design SE.

For the stratified finite-manifest average, with one uniformly selected date
from bin h, `Var(mu_hat) = sum_h (N_h/N)^2 (1 - 1/N_h) S_h^2`.
The unknown within-bin finite-population variances `S_h^2` cannot be estimated
from a single date in each nontrivial bin. No numerical 95% CI is claimed.
The unweighted sampled-day SD and the year-resampling sensitivity percentiles
in the auxiliary reports remain descriptive, not design SEs or CIs.
