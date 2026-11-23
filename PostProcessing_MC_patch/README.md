# PostProcessing — Monte Carlo uncertainty, critical-correction review (v2)

This is an **incremental, local, backup-free patch**, not a full repository replacement. It **supersedes v1**; it accepts the reviewed raw PostProcessing source or upgrades exact v1-installed Monte Carlo modules without altering unrelated user files. It targets `hossein-omidi/mbpo_custom` branch `PostProcessing` as inspected 2026-11-23. No GitHub changes, commits or pushes are made. Existing observation/action dimensions, nominal weather catalogue, actor/critic, one-step MBPO synthetic replay, scene split and physical PVWatts/SAPM equations are preserved.

## Scientific scope

**Do not describe the example 5% parameter as measured NSRDB accuracy.** NSRDB is modeled/satellite/reanalysis meteorology, and the 5% values provided here are *hypothetical sensitivity assumptions*. For scientific claims about actual measurement/model uncertainty, the distribution, correlations and magnitude require independent ground-station calibration or defensible external validation. A mathematically valid Monte Carlo propagation model does not establish that assumed errors are true.

### Two levels of stochasticity

1. **Historical daily scenario**: the training and test manifest's complete five-minute NSRDB day is selected once per episode. The present `PostProcessing` month-stratified train/validation/test IDs remain unchanged.
2. **Uncertainty realization conditional on the day**: an independent full-trajectory draw is generated once per episode. The five weather-channel Gaussian *latent* error processes use a correlated multivariate, stationary AR(1) series plus correlated whole-day offsets:

   `Z_t = sqrt(q) B + sqrt(1-q) U_t`,  `U_t = phi U_(t-1) + sqrt(1-phi²) eps_t`

   Here `B, eps_t ~ N(0,C)` and `U_0 ~ N(0,C)` independently, where C is a validated, positive-definite 5x5 correlation matrix. Every timestamp has marginal covariance C; day-level offsets and autoregressive innovations preserve temporal dependence.

   DNI, DHI and wind use mean-one lognormal multipliers `exp(sigma * Z - sigma²/2)`, with `sigma² = log(1+CV²)`. Temperature uses additive °C errors. Albedo is recorded from NSRDB and receives bounded additive errors only if explicitly enabled; boundary clipping may introduce bias, which is reported. Zero irradiance remains zero and the night does not spontaneously become sunny.

   DNI and DHI errors control the GHI change via `GHI' = GHI + (DNI'-DNI) cos(zenith)_+ + (DHI'-DHI)`. This **preserves** the source data's DNI–DHI–GHI closure residual (unless the sampled GHI is smaller than DHI and must be projected onto the physical lower bound). It neither independently samples GHI nor falsely claims to repair any original NSRDB closure errors. Irradiance/albedo projection counts and maximum original/resulting closure residual are included in evaluation diagnostics. An original NSRDB irradiance closure residual is **not** guaranteed to be zero, so the code does **not** claim exact radiative closure.

**This is a sensitivity model, not a fitted weather forecasting model.** Solar geometry and panel movements remain deterministic conditional on an uncertainty draw. The same compact six-dimensional observation contains the updated instantaneous DNI/DHI/power/pose/time. The extra latent weather variables remain unobserved, so the compact controller remains partially observable; the patch does not claim otherwise.

### Training

Both existing trainers construct the same `StageAPVTrackingEnv`, so the new `weather_uncertainty` configuration is injected into their shared SAC config and MBPO config. The environment draws one coherent stochastic weather day **at reset**; action-dependent power is then computed by the existing pvlib backend at each of the 195 successive control transitions. The two trainers add only per-episode uncertainty-realization metadata; neither the SAC loss nor MBPO ensemble/replay algorithm is rewritten.

- **SAC:** transitions from the physically evaluated perturbed trajectories enter its original off-policy replay.
- **MBPO:** transitions from those same kinds of perturbed trajectories enter real replay, and only the real transitions train/validate the weather→power neural ensemble. The existing ensemble uncertainty is about its learned transition model; it is **not** interpreted as calibrated NSRDB measurement uncertainty. One-step imagined model rollouts still enter SAC updates through the original model replay.
- **Sample complexity:** keep reporting *actual environment transitions*, not invented rollouts or synthetic weather samples. Draw generation increases computation time; it does not reduce the sample count by itself. The existing sample-complexity workflow is retained.
- **Periodic validation:** a frozen draw `realization_id=0` per validation scenario is used for both policies and reference baselines across every checkpoint, avoiding comparisons based on changing weather realizations. This is separate from the final multi-draw Monte Carlo test.

### Evaluation

The new `09_monte_carlo_evaluate.py` runner iterates `scenario_id × realization_id × controller`, using the **same entire weather profile SHA-256, UTC timestamps and initial panel pose** for each paired controller. It runs the full physical environment for SAC, MBPO, fixed and slew-limited sun tracking, with individual action decisions at each five-minute step. No real outputs are replaced with a timestep-wise average and no MBPO synthetic rollouts enter evaluation.

For each historical day e, M independent uncertainty draws produce energy `R(e,m)`. Day-wise means, descriptive SD and percentiles are recorded. **A separate time-resolved mean, SD and quantile profile is now exported for every tested historical day**, not only the first. Across days, the empirical variance identity is summarized descriptively as

`V_total = Var_day[E(R|day)] + E_day[Var(R|day)]`.

Paired differences (MBPO minus SAC, etc.) use the same day and realization for both policies. We report the finite-draw Monte Carlo SE of the estimated mean **conditional on the fixed historical days**, plus a correction for the finite-M contribution to estimated between-day variance. That standard error must not be described as uncertainty across training seeds or future climate. Three PNGs are generated: first-day power with descriptive SD, daily net-energy boxplots and paired day-mean differences.

These estimates describe uncertainty under **the stated assumed weather-error distribution**, not field accuracy or uncertainty about future climate. A 95% empirical interval is a distribution quantile interval, **not** a frequentist confidence interval. For quality assurance, compare a 100-draw run with 500 or 1000 draws and verify stability; leave the frozen test role untouched during tuning.

## Installation

Extract the archive into the root of your *local* `PostProcessing` checkout. It contains `PostProcessing_MC_patch/`.

```bash
python PostProcessing_MC_patch/apply_local.py --root . --dry-run
python PostProcessing_MC_patch/apply_local.py --root . --diff-file PostProcessing_MC.patch
# Inspect PostProcessing_MC.patch, then either:
python PostProcessing_MC_patch/apply_local.py --root . --apply
# or apply the diff yourself with git apply after checking it
```

The installer verifies exact Git blob SHA-1 values for each of the eight existing source files and unique replacement anchors; it refuses different versions rather than overwriting changes it does not know how to merge. It creates **no backups** and issues **no git commands**. New modules must not already exist with different contents.

**IMPORTANT:** You may apply v2 to the exact reviewed raw `PostProcessing` source or upgrade the exact v1 Monte Carlo package (its already-installed new files are verified by SHA-1). Any unrecognized source changes deliberately fail the guarded installer. If your local source diverged from the reviewed PostProcessing branch, reconcile the source manually before applying.

## Example assumed sensitivity model

The included `examples/hypothetical_uncertainty_5pct.json` defines **unvalidated** DNI, DHI and wind CVs of 5%, temperature SD of 1°C, albedo SD of 0.02 absolute, 50% of the latent variance from persistent day-level errors, AR(1) coefficient 0.90, and illustrative DNI–DHI latent correlation 0.2. Change and justify those values before making a scientific *uncertainty accuracy* claim. The calibration protocol is outside this patch.

Train both algorithms under the **same assumption file**, separate run directories and matching seeds:

```bash
python -m MBPO_workspace.workflows.08_monte_carlo_train \
  --algorithm sac --uncertainty-spec PostProcessing_MC_patch/examples/hypothetical_uncertainty_5pct.json \
  --run-dir SAC_workspace/runs/sac_mc_seed0 --seed 0 --epochs 2000

python -m MBPO_workspace.workflows.08_monte_carlo_train \
  --algorithm mbpo --uncertainty-spec PostProcessing_MC_patch/examples/hypothetical_uncertainty_5pct.json \
  --run-dir MBPO_workspace/runs/mbpo_mc_seed0 --seed 0 --epochs 2000
```

Use **fresh training runs**. The patched environment physics source hash and simulation assumption differ from old checkpoints; the installer does not mislabel old checkpoints as equivalent or allow resuming them under new dynamics.

For the independent, frozen test partition and 100 draws per day:

```bash
python -m MBPO_workspace.workflows.09_monte_carlo_evaluate \
  --sac-checkpoint SAC_workspace/runs/sac_mc_seed0/latest.pt \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_mc_seed0/latest.pt \
  --scenario-split-json SAC_workspace/runs/sac_mc_seed0/scenario_split.json \
  --split-role test --draws 100 \
  --uncertainty-spec PostProcessing_MC_patch/examples/hypothetical_uncertainty_5pct.json \
  --outdir MBPO_workspace/evaluation/monte_carlo_test
```

Check both run directories' `scenario_split.json` hashes match before running final tests. This evaluator checks that the requested split equals the manifest-derived frozen split and that source code and uncertainty assumptions match checkpoints. For a **small computational smoke test only**, add `--max-days 2 --draws 3`; such output is flagged as incomplete test coverage. Do not tune policies using the test split. The evaluation can be computationally substantial because every draw reruns all four policies over 195 real PV time steps. The result includes `daily_power_mean_std_by_utc_step.csv` for **every** selected historical day, `all_day_draw_method.csv`, `first_day_power_mean_sd.csv`, and `mc_summary.json`. The report includes irradiance/albedo projection counts and raw/post-projection irradiance closure residuals.

If MBPO training used custom `--model-epochs` or `--model-max-train-samples`, provide the same overrides during Monte Carlo evaluation for strict checkpoint matching.

## Offline tests and integration verification

```bash
python -m pytest -q PostProcessing_MC_patch/tests/test_monte_carlo_weather.py
```

The offline numerical tests verify input immutability, mean-one lognormal scaling, Gaussian correlation and temporal AR(1) behavior, residual preservation where physically feasible, GHI≥DHI projection, bounds, night zeros, seed-based common random numbers, invalid inputs, variance decomposition and finite-draw correction, multi-day MC orchestration, strict albedo aliases and v1 upgrade safeguards. Source anchors and Git blob hashes were checked against read-only GitHub `PostProcessing` for all eight touched modules, including model validation (each replacement anchor occurred once).

**Limitations of verification:** The tooling here does not have the annual NSRDB 2018–2024 CSV files or a full pvlib/Gym runtime. The entire patched training/evaluation pipeline has therefore **not** been executed on actual weather data. After applying locally, first run a two-day/three-draw test and verify power/energy consistency, weather hashes and UTC timestamps. Full scientific validation requires those end-to-end checks, Monte Carlo convergence and calibrated (or clearly assumed) error parameters.


## High-priority corrections in v2

1. **Counterfactual model validation under stochastic weather:** The existing
   `MBPO_workspace/workflows/06_model_validation.py` previously reset the same
   historical day without fixing its uncertainty realization. It would therefore
   reconstruct a *different* physical state and fail an exact-replay assertion
   or compare unrelated outcomes. This patch adds `--uncertainty-spec` and
   `--realization-id` and reuses one fixed realization for the initial trajectory
   and all counterfactual action replays.
2. **Astronomical baseline timing:** The Monte Carlo sun-tracking baseline
   now uses the current decision timestamp, matching the existing
   `PeriodicEvaluator` convention; its reward is still post-action at the next
   timestamp. The alternative next-step astronomical controller is a different
   benchmark and should not be mixed into the same results without labeling it.
3. **Variance interpretation:** The observed finite-sample variance decomposition
   is now explicitly distinguished from the ideal law of total variance. The
   report adds finite-M-corrected between-day variance and the conditional
   MC standard error of the overall average (for the fixed tested days).
4. **All-day power uncertainty:** Previously only the first evaluated day
   received 5-minute power mean/SD/quantiles. Now every selected historical day
   produces its own 195-step profile in the output CSV.
5. **Physical and provenance checks:** A sampled GHI below DHI is projected
   to the physical lower bound and counted, albedo column aliases are verified
   before perturbation, the exact UTC grid and per-step weather values are
   checked across paired controllers, and SAC/MBPO checkpoint protocol
   fingerprints are compared with the frozen test-split hash.

### Run model validation on an MC-trained MBPO checkpoint

```bash
python -m MBPO_workspace.workflows.06_model_validation \
  --run-dir MBPO_workspace/runs/mbpo_mc_seed0 \
  --checkpoint MBPO_workspace/runs/mbpo_mc_seed0/latest.pt \
  --uncertainty-spec PostProcessing_MC_patch/examples/hypothetical_uncertainty_5pct.json \
  --realization-id 0
```

The fixed realization permits exact action-prefix replay. This single-draw
model-validation diagnostic does **not** establish model accuracy throughout
all possible weather errors. For a broader robustness study, run separate
validation realizations using different IDs and distinct report directories.

### Critical deployment caveats

- Keep training and testing on the same frozen historical-day partition.
- Train both algorithms afresh with the same declared uncertainty model. An
  original nominal checkpoint is not an MC-trained checkpoint.
- Do not use the independent test split to select uncertainty parameters,
  models, hyperparameters or the number of Monte Carlo draws.
- **Do not claim measured 5% NSRDB errors, improved robustness, or lower
  sample complexity merely because the code generates uncertainty draws.**
  Those are experimental questions requiring calibration, multiple seeds,
  the saved performance curves and the results of the real pvlib runs.
- The expanded offline suite checks mathematics, evaluation wiring and
  installer compatibility. **Full pvlib/Gym + seven-year NSRDB execution is
  still untested in this runtime** because those dependencies and annual
  CSV files are unavailable here. The source patch is not yet certified
  as production-valid until local integration, sample-complexity and
  counterfactual-model-validation runs pass.


### Verify the physical simulator before investing in RL training

After patch installation, this checkpoint-free verifier requires only your
local yearly NSRDB files and installed pvlib/Gym environment:

```bash
python PostProcessing_MC_patch/tools/verify_mc_environment.py \
  --uncertainty-spec PostProcessing_MC_patch/examples/hypothetical_uncertainty_5pct.json \
  --out outputs/monte_carlo_environment_audit.json
```

It chooses a **training-only** historical day, tests two repeated full 195-step
trajectories with `realization_id=0`, checks that realization 1 differs, confirms
exact UTC and energy/reward accounting, and independently runs an unperturbed
reference episode without modifying any recorded NSRDB source weather.
It is an integration test, **not** proof of field accuracy or RL model skill.

**Important:** As the weather projection and physics module hashes have
changed since the original MC v1 package, **retrain both SAC and MBPO** under
v2. Do not reuse nominal or old v1 MC checkpoints as if their simulator were
identical to this reviewed version. Run this integration check before training,
then run MBPO model validation with a frozen realization ID before reporting
sample complexity or counterfactual prediction accuracy.
