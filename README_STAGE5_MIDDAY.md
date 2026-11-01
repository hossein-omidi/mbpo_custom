# Stage 5 — Read-only SAC/MBPO Midday Investigation

**Scope:** `PVwatt_dev` standalone SAC and MBPO, after the Stage A albedo-aware replacement. This is a **diagnostic-only** package. Extract into the existing repository root; no existing source file is replaced, no trained policy is adjusted, and no training/evaluation data is rewritten. Files created are solely in the `--outdir` folder (Python bytecode generation is disabled).

The diagnostic runs counterfactuals against **your actual installed local NSRDB + PVWatts/SAPM simulator**, not a fabricated weather generator. The code has been audited against the public `PVwatt_dev` source and Stage A interface; **the complete WSL run cannot be performed in the build environment because the user's checkpoints, multi-year manifest/data, pvlib, and Gym are not available here.** Unit tests of pure logic pass locally, and the true-NSRDB integration remains a required acceptance check on your machine.

## 1. Install by extraction only

Copy `STAGE5_READ_ONLY_MIDDAY_DIAGNOSTIC.zip` to your GitHub project root:

```bash
cd ~/projects/PVLIB_Paper/mbpo_custom
unzip -o STAGE5_READ_ONLY_MIDDAY_DIAGNOSTIC.zip -d .
```

This ZIP adds only:

- `scripts/stage5_midday_diagnostic.py`
- `tests/test_stage5_midday_diagnostic.py`
- `README_STAGE5_MIDDAY.md`
- `STAGE5_METHODS.md`
- `SHA256SUMS.txt`

Run tests before running the full diagnostic:

```bash
python -m pytest -q tests/test_stage5_midday_diagnostic.py
```

The pvlib test must run on WSL rather than skip. Run the diagnostic using Python from your `simple-worm` environment. No Bash files or patch installer are required.

## 2. Preflight: locate the real evaluation file

```bash
find MBPO_workspace/evaluation -maxdepth 3 -name trajectories_detailed.csv -print
```

The `--eval-dir` argument is the parent of `trajectories_detailed.csv`, **not** its `verified/` directory. `--eval-dir MBPO_workspace/evaluation/stage_a_seed1_heldoutverified` is usually an incorrectly concatenated console string. Check the actual directory on disk.

## 3. Recommended complete diagnostic (frozen checkpoints)

```bash
python scripts/stage5_midday_diagnostic.py \
  --sac-checkpoint SAC_workspace/runs/sac_stage_a_seed1/latest.pt \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_stage_a_seed1/latest.pt \
  --sac-training-csv SAC_workspace/runs/sac_stage_a_seed1/episodes.csv \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_stage_a_seed1/episodes.csv \
  --eval-dir MBPO_workspace/evaluation/stage_a_seed1_heldout \
  --days 8 \
  --offsets -2 0 2 \
  --grid -1 -0.5 0 0.5 1 \
  --noon-width-min 30 \
  --model-rows 4000 \
  --sensitivity-states 300 \
  --seed 42 \
  --outdir MBPO_workspace/evaluation/stage5_midday_diagnostic
```

**Shell note:** the `\` must be the last character on the line. Do not add a space after `\`.

This executes on selected calendar-stratified evaluation scenarios:

- exact five-minute UTC and `America/New_York` clock diagnostics;
- spring/fall DST offset and **23/24/25-hour local-calendar-day** behavior;
- actual solar noon inferred from minimum recorded solar zenith;
- sunrise/sunset and daylight covered/missed by the fixed 12:00–21:45 UTC horizon;
- complete intraday SAC/MBPO-versus-fixed comparison for every date in the paired CSV;
- geometry/POA/albedo/temperature replay on selected real-weather scenarios;
- near-noon 25-action grids of normalized tilt and azimuth, plus the actual actor action, with full deterministic continuation under the original actor;
- critic versus physically realized action rankings;
- sensitivity of the actor to power and remaining horizon (`tau`), explicitly marked as off-manifold if perturbed;
- 359.9° versus 0.1° azimuth-wrap representation stress checks;
- replay capacity, day repetitions, seasonal sampling and real-transition budgets;
- fixed-model MBPO ensemble one-step prediction versus persistence by season/noon period, including uncertainty and ensemble disagreement;
- Q/TD error probes on held-out **evaluation** states, distinct from the prior training-replay TD diagnostic.

The 25-action grid is not an exhaustive continuous optimizer; the best sampled candidate is a **best tested action**, not a global or feasible full-horizon oracle. The “full deterministic continuation” compares the same conditioned first action followed by the same frozen deterministic learned policy, not the exact entropy-regularized SAC soft-Q objective.

## 4. Faster end-to-end smoke run

Use this after the unit tests to verify your actual installed simulator before the complete grid:

```bash
python scripts/stage5_midday_diagnostic.py \
  --sac-checkpoint SAC_workspace/runs/sac_stage_a_seed1/latest.pt \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_stage_a_seed1/latest.pt \
  --sac-training-csv SAC_workspace/runs/sac_stage_a_seed1/episodes.csv \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_stage_a_seed1/episodes.csv \
  --eval-dir MBPO_workspace/evaluation/stage_a_seed1_heldout \
  --days 4 --offsets 0 --grid -1 0 1 \
  --model-rows 500 --sensitivity-states 50 \
  --outdir MBPO_workspace/evaluation/stage5_smoke
```

The full run tests many more near-noon alternatives. The smoke run is an integration check, not scientific evidence of an effect.

## 5. Outputs

Created inside the specified output folder:

| Output | Purpose |
|---|---|
| `stage5_report.json` | Master findings, checkpoint/data hashes, budgets, warnings |
| `clock_and_noon.csv` | Actual UTC timestamps, NY offsets, DST, noon timing and local day duration |
| `daylight_duration_utc_local.csv` | Approx. UTC sunrise/sunset, sunlit hours in/out of window |
| `intraday_power_comparison.csv` | Per-five-minute SAC/MBPO energy, orientation, AOI vs fixed |
| `intraday_summary.csv` | Equal-day descriptive seasonal/phase summaries |
| `physical_geometry_replay.csv` | Actual pvlib replay: cosine AOI, direct/sky/ground POA, recorded albedo |
| `power_tau_sensitivity.csv` | Actor sensitivity to current power and remaining horizon |
| `azimuth_seam.csv` | Neural-action jump across the 0/360° representation boundary |
| `mbpo_dynamics_heldout_rows.csv` | Model prediction bias/uncertainty on real evaluation transitions |
| `mbpo_dynamics_heldout_summary.csv` | Model RMSE vs persistence and uncertainty calibration |
| `heldout_critic_td_rows.csv` | TD residuals on actual evaluation transitions |
| `heldout_critic_td_summary.csv` | Noon versus off-noon TD residuals |
| `training_history_and_seasons.csv` | Episodic returns, critic/actor losses, alpha, model skill, season |
| `action_grid_branches.csv` | Actual actor plus alternative tilt/azimuth actions, physical continuation, critic Q |
| `action_grid_regrets.csv` | Sampled action-grid advantage over chosen actor action |
| `noon_power_comparison.png` | Time-aligned two-policy power difference against fixed |
| `seasonal_noon_difference.png` | Seasonal noon-band gains versus fixed |
| `action_grid_regret.png` | Sampled grid regret distributions |
| `README_RESULTS.md` | Limitations and interpretation caveats |
| `SEND_ME_STAGE5_MIDDAY_DIAGNOSTIC.zip` | **Upload this output ZIP back into ChatGPT** |

The ZIP includes *outputs only*, not original NSRDB CSVs, checkpoints, secrets or repository source files. All original inputs are fingerprinted; they are never edited.

## 6. The time-zone/duration hypothesis

The simulator uses the **same UTC 117-step period in every season** (reset at 12:00 UTC, post-action rewards 12:05 through 21:45 UTC). New York is 07:00–16:45 **EST** in winter, but 08:00–17:45 **EDT** in summer. The interval remains 9h45m in UTC, despite this one-hour civil-clock shift.

Solar noon around 16:55 UTC occurs near decision step 59–61, but the script identifies the precise five-minute noon *on each actual scenario*, not by assuming step 60 for every date.

The scientific issues differ:

- **Time-zone error:** incorrectly parsing UTC as local standard time or double-applying offsets. A 4–5 hour mismatch would be catastrophic. Exact physical replay and five-minute UTC checks detect this.
- **DST effects:** EST↔EDT changes wall-clock labels while UTC progression remains regular. A DST transition day may contain 23 or 25 local clock hours, but a fixed UTC 117-step episode is unaffected in elapsed duration.
- **Incomplete daylight:** the fixed UTC episode can omit sunlit morning and/or evening intervals. That affects interpretation of a “daily” energy total and season comparisons. The diagnostic reports daylight *hours* omitted, not speculative energy outside the observed data.
- **Noon location:** minimum solar zenith defines solar noon physically. We do not substitute 12:00 civil time or exact 17:00 UTC.

**No time-zone correction should be applied to the simulator unless the results demonstrate a specific error.** Changing the episode window changes the MDP and would require retraining, so first diagnose with the current contracts unchanged.

## 7. Scientific limitations

1. A solar-noon dip below fixed is mathematically possible even if full-day MBPO energy exceeds fixed. The policy optimizes cumulative reward under partial observability, not per-step dominance.
2. The compact state is the same six features for SAC and MBPO. Feature sufficiency is **not established** by a single checkpoint comparison. Small off-manifold input changes only measure sensitivity; a causal ablation requires new training.
3. A large grid-regret result is strong evidence of a *locally suboptimal tested action* in that policy context, not proof of a critic implementation bug. Critic Q is soft/stochastic reward-scaled return; the branch score is deterministic physical return.
4. The MBPO ensemble's member disagreement is not automatically calibrated uncertainty; its models are not trained on independent bootstrap datasets in the original public implementation. Evaluate the empirical calibration.
5. Model prediction RMSE uses held-out **policy evaluation trajectories**. The script demands both training CSVs and explicitly rejects training/evaluation overlap. Reusing diagnostic days to tune hyperparameters would invalidate their status as untouched final test days.
6. The selected eight branch days are calendar-balanced by precommitted random seed. The main time-profile summary uses all paired evaluation days and is not a multi-training-seed confidence interval.
7. The tool flags unequal real-transition budgets but does not fix them. Only budget-matched retraining and multiple independent seeds justify algorithm-level conclusions.
8. PyTorch checkpoint loading with `weights_only=False` can execute untrusted pickle code: use only **your own** locally generated trusted checkpoints.
9. The source may differ from the reviewed ZIP overlay if you changed your working tree after installation. Current MDP hash and replay checks fail closed to avoid false confidence.

**If the script fails**, send the first traceback and the printed file/config paths. Do not patch out failed physics, data-integrity, or action-replay assertions.
