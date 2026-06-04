# Stage 3 — full-year stochastic RL (simple guide)

Standalone full-year experiment: **empirical annual weather scenarios** (day sampling), deterministic pvlib physics, seed-based evaluation. Stages 0–2 are **not** required.

**Canonical config:** `examples/config/pv_tracking/stage3_fullyear_stable_mbpo.py`  
**Protocol:** [RL_EVAL_PROTOCOL.md](RL_EVAL_PROTOCOL.md)

**Baselines** = `sun_tracking` and `fixed_no_motion` (same env, same pvlib energy path).

---

## What the config does

| Item | Setting |
|------|---------|
| Training days | `2020-01-01` … `2020-12-31`, **all** calendar days (no `excluded_dates`) |
| Weather | `historical` TMY — exogenous trajectory per sampled day; pvlib deterministic |
| Perturbation | `irradiance_perturbation_std=0` (paper); optional bounded augmentation only |
| Episode | 78 steps, UTC `13:30`–`23:15`; i.i.d. day + stochastic ξ each reset |
| In-train checkpoint eval | Same annual support; random day + seeds (no fixed calendar hold-out) |
| Post-train eval | Frozen policy; independent `--eval-seed-base` rollouts over full year |
| Optional diagnostics | `STAGE3_STRESS_TEST_DATES` — fixed calendar panel for plots only |

---

## Annual-scenario formulation

Environmental variability is represented through sampling from an empirical annual weather distribution. Each sampled day provides an exogenous weather trajectory. pvlib deterministically computes the PV response to this weather trajectory and the policy-selected panel orientation. Stochasticity enters through episode-level day/weather-scenario sampling, not synthetic intra-day cloud-transition modeling.

See [RL_EVAL_PROTOCOL.md](RL_EVAL_PROTOCOL.md).

---

## Setup

```bash
cd /home/user01/mbpo_custom
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mbpo
pip install -e .   # once
chmod +x scripts/run_stage3_fullyear.sh
```

---

## 1. Verification (before training)

```bash
./scripts/run_stage3_fullyear.sh verify
```

Runs config merge, RL protocol checks (no calendar hold-outs), env contract, baseline pvlib path, and state-space sanity.

Reports: `verification/stage3_fullyear/preflight_report.txt`

---

## 2. Train

```bash
./scripts/run_stage3_fullyear.sh train
```

Trial path: `sequential_stage_artifacts/stage3_fullyear_trial_dir.txt`

---

## 3. Training plots

```bash
./scripts/run_stage3_fullyear.sh plot
```

Output: `training_plots/stage3_fullyear/`

---

## 4. Post-train evaluation

### 4a. Standard eval (seed rollouts + paper figures)

```bash
./scripts/run_stage3_fullyear.sh eval
```

40 rollouts, `--eval-seed-base 100000`, annual day support. Outputs under `evaluation/pv_stage3_fullyear/`:

| Output | Description |
|--------|-------------|
| `paper_figures/` | **Main paper plots** (see list below) |
| `paper_figures/paper_metrics.json` | mean/std/n per method |
| `evaluation_summary.json` | Aggregate stats |
| `rollouts/`, `baseline_rollouts/` | Full trajectory CSVs |

**Paper figures generated automatically:**

1. `annual_performance_bars.png` — gross energy, movement, net, reward (mean ± std)
2. `energy_decomposition.png` — gross − movement = net
3. `seasonal_net_energy.png`, `seasonal_movement_cost.png`
4. `representative_daily_<date>.png` — power, cumulative energy, tilt/azimuth, solar alt, actions, reward
5. `daily_gain_distributions.png` — MBPO-SAC − fixed / − sun
6. `movement_efficiency.png` — angular movement vs net energy
7. `mc_timeseries_power.png` — UTC-hour pooled mean ± 1 std
8. `training_diagnostics/` — optional curves from `progress.csv`

### 4b. Monte Carlo eval (season bars — paper figures)

```bash
MC_REPLICATES=8 ./scripts/run_stage3_fullyear.sh mc-eval
```

Default `--date-set annual`: independent seeds over full year; season summary ± SEM.

Output: `evaluation/pv_stage3_fullyear_mc/`

**Optional fixed-date stress panel** (diagnostics only):

```bash
./scripts/run_stage3_fullyear.sh mc-eval-stress
```

Uses `STAGE3_STRESS_TEST_DATES` — not the primary RL test set.

Both standard + MC:

```bash
./scripts/run_stage3_fullyear.sh all-eval
```

---

## 5. Manual commands

```bash
export TRIAL=$(cat sequential_stage_artifacts/stage3_fullyear_trial_dir.txt)
export CKPT="$TRIAL/best_eval_checkpoint"

python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_stage3_fullyear \
  --eval-protocol inherit \
  --num-rollouts 40 --eval-seed-base 100000 \
  --max-path-length 78 --compare-baselines

python scripts/evaluate_fullyear_mc.py "$CKPT" \
  --outdir evaluation/pv_stage3_fullyear_mc \
  --eval-protocol inherit \
  --date-set annual --num-rollouts 40 --eval-seed-base 100000 \
  --error-bars sem
```

---

## Evaluation methodology

1. **Independence by seed (ξ)**, not by excluding calendar days from training.
2. **Frozen policy** — no SAC / replay / BNN updates during eval scripts.
3. **Real environment only** — not the learned dynamics model.
4. **Season bars:** pool rollouts by sampled day-of-year season; SEM across episodes.
5. Optional stress mode: aligned + ensemble bands on fixed dates for interpretability.

BNN `holdout_ratio` on replay transitions is **dynamics validation**, not RL policy evaluation.

---

## Related docs

- [RL_EVAL_PROTOCOL.md](RL_EVAL_PROTOCOL.md) — audit summary and objective
- [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md) — general stage rules
- [PV_SIMPLE_WORKFLOW.md](PV_SIMPLE_WORKFLOW.md) — Stage 0 paper track

### Post-train helper

```bash
./run_stage3_posttrain.sh latest          # seed-based frozen-policy eval
./run_stage3_posttrain.sh latest stress   # optional fixed-date diagnostics only
```

