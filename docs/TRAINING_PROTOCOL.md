# PV tracking — training protocol (A→Z)

Phased procedure for PV tracking with MBPO/SAC. Each stage is a **standalone experiment**: its own config, fresh trial, own evaluation directory, and own weather/day-sampling contract. Stages do **not** share checkpoints or replay pools unless you explicitly pass `--restore` (only supported for continuing the **same** Stage 3 config).

**Recommended science order:** Stage 0 (single day) → Stage 1 (summer clearsky) → Stage 2 (summer historical) → Stage 3 (full year). You may run any stage alone without completing earlier stages.

**Rules**

- **One stage = one fresh train** — no `--restore` across stages ([§0](#0-clean-stage-isolation-every-run)).
- **One calendar day = one episode** — no multi-day episodes (**78** control steps; `periods=79`, `freq=7min30s`).
- Use the **canonical `evaluation/pv_stage*`** (or stage helper script) for that config — do not mix eval outputs between stages.
- **Power / energy:** pvlib via `PVTrackingEnv.step` for learned policy and baselines ([§1.1](#11-physics-and-baselines-pvlib)).
- **Post-train eval:** prefer `--eval-protocol inherit` so date, `weather_source`, and `movement_penalty` match the checkpoint config ([§6](#6-training-vs-evaluation-environments)).

Copy-paste shortcuts: [PV_SIMPLE_WORKFLOW.md](PV_SIMPLE_WORKFLOW.md).

---

## Stage independence (at a glance)

| Stage | Config | Train: days / weather | i.i.d. episodes? | In-train eval dates | Post-train eval dir |
|-------|--------|------------------------|------------------|----------------------|---------------------|
| **0 clearsky** | `stage0_single_day` | `2020-06-15` / clearsky | No (single day) | same day | `evaluation/pv_stage0_baseline/` |
| **0 cloudy+movement** | `stage0_single_day_mbpo_paper` | `2020-06-21` / historical | No | same day | `evaluation/pv_stage0_mbpo_paper_movement/` |
| **1** | `0.py` | summer `2020-06-01`…`08-31` / clearsky | Yes | 4 summer hold-outs | `evaluation/pv_stage1_clearsky_summer/` |
| **2** | `stage2_random_weather` | summer range / **historical** | Yes | 4 summer hold-outs | `evaluation/pv_stage2_random_weather_summer/` |
| **3** | `stage3_fullyear_random_clean_split` | full year / historical annual scenarios | Yes (all days) | random day + seeds (annual) | `evaluation/pv_stage3_fullyear/` |

Canonical dates live in `examples/config/pv_tracking/verified_dates.py` (PVGIS TMY, Albuquerque site).

---

## 0. Clean stage isolation (every run)

### Is poor performance caused by a previous stage’s data?

**No — when you follow the clean protocol.**

A normal `mbpo run_local` (without `--restore`) starts an empty replay pool, fresh networks, and a new `seed:…` trial. Old trials on disk are ignored unless you point `--restore` or eval at them.

**Contamination happens only if you:**

1. Pass **`--restore=…`** from another stage or MDP.
2. Evaluate the **wrong checkpoint** (newest `seed:*` while another stage is training).
3. Override eval with mismatched **`--eval-weather-source`** or dates (use **`inherit`** instead).

### Clean warmup (every stage)

**Step 0 — Environment**

```bash
cd /path/to/mbpo_custom
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mbpo
```

**Step 1 — Preflight**

```bash
python scripts/verify_training_config.py --config CONFIG_MODULE
python scripts/validate_pv_rollouts.py --config-path CONFIG_PATH
```

For movement-penalty configs also:

```bash
python scripts/verify_movement_cost_fairness.py --config-path CONFIG_PATH
```

**Step 2 — Train (no restore)**

```bash
mbpo run_local examples.development \
  --config=CONFIG_MODULE \
  --gpus=0 --trial-gpus=0 --cpus="$CPUS" --trial-cpus="$TRIAL_CPUS" \
  --temp-dir="$PWD/.ray_tmp/YOUR_LABEL"
```

Or use `./scripts/run_stage0_baseline_trial.sh` / `run_stage0_paper_trial.sh` for Stage 0 tracks.

**Step 3 — Bind eval to this trial**

```bash
export TRIAL=/full/path/to/seed:YOUR_TRIAL_DIR
export CKPT="$TRIAL/best_eval_checkpoint"

python -c "
import json, sys
v = json.load(open(sys.argv[1] + '/params.json'))
assert v.get('restore') in (None, ''), 'refusing eval: trial was restored'
print('OK:', v.get('config_version'))
" "$TRIAL"
```

**Step 4 — Evaluate** into the canonical stage outdir with `--eval-protocol inherit` ([§4–§7](#4-stage-0--clearsky-single-day)).

**Step 5 — Plot training metrics**

```bash
python scripts/plot_training_progress.py "$TRIAL" --outdir training_plots/STAGE_LABEL \
  --metrics evaluation/return-average training/return-average model/val_loss \
  policy/shifts-mean policy/actions-std alpha
```

**Step 6 — Gate**

```bash
python scripts/diagnose_tracking.py \
  --eval-dir evaluation/pv_stageSTAGE/ \
  --trial-dir "$TRIAL" \
  --progress-csv "$TRIAL/progress.csv" \
  --verify-env --gate
```

### Stage 3 exception

`resume_stage3_refinement.sh` may `--restore` to continue the **same** Stage 3 MDP only — not cross-stage transfer.

### CPU / parallel Stage 0 runs

On a 16-thread VM, avoid two jobs each using `nproc` workers. Use `scripts/pv_cpu_env.sh`:

| `PV_CPU_PROFILE` | `--cpus` | `--trial-cpus` | When |
|------------------|----------|----------------|------|
| `single` | 10 | 4 | One training job |
| `dual` | 6 | 2 | Baseline + paper in parallel |

Each job needs its own `--temp-dir`. Helpers: `run_stage0_baseline_trial.sh`, `run_stage0_paper_trial.sh` (set `PV_CPU_PROFILE=dual` for parallel runs).

---

## 1. Formal MDP (as implemented)

| Symbol | Meaning |
|--------|---------|
| **Episode** | One calendar day, UTC `13:30`–`23:15`, **78** steps (`periods=79`, `freq=7min30s`) |
| **State** | `physical` 11-D: solar angles, normalized irradiance, panel pose, `cos_aoi` |
| **Action** | `a ∈ [-1,1]²` → Δtilt ∈ [-5°,5°], Δazimuth ∈ [-10°,10°] per step |
| **Reward** | `r = energy_kwh - movement_penalty × (‖a_tilt‖₁ + ‖a_azimuth‖₁)` |
| **Termination** | After 78 steps; no next-day carry-over |
| **Day sampling** | `reset()`: one day from `[start_date, end_date]` if `randomize_day=True` (i.i.d. over catalog) |

### 1.1 Physics and baselines (pvlib)

- `energy_kwh` and `info['power']` come from **`get_total_irradiance`** (pvlib) in `mbpo/env/pv_tracking.py::_power_from_orientation`.
- **`sun_tracking`** and **`fixed_no_motion`** call `env.step` with orientation targets decoded from obs; they use the **same** pvlib path as the learned policy (`scripts/eval_utils.py::make_baseline_rollout`).
- Verify: `python scripts/verify_pv_state_space.py` → `verification/pv_state_space/`.

---

## 2. Directory layout

### 2.1 Training artifacts (Ray)

| Purpose | Path |
|---------|------|
| Trials | `~/ray_mbpo/PVTracking/pv_tracking/seed:<id>_<timestamp>/` |
| Metrics | `…/progress.csv` |
| Best checkpoint | `…/best_eval_checkpoint/` (`monitor_metric=evaluation/return-average`) |

### 2.2 Evaluation outputs (repo)

| Stage track | Canonical `--outdir` |
|-------------|----------------------|
| Stage 0 clearsky | `evaluation/pv_stage0_baseline/` |
| Stage 0 paper | `evaluation/pv_stage0_mbpo_paper_movement/` |
| Stage 1 | `evaluation/pv_stage1_clearsky_summer/` |
| Stage 2 | `evaluation/pv_stage2_random_weather_summer/` |
| Stage 3 final test | e.g. `evaluation/pv_stage3_final_test/` (define per study) |

Each eval run typically includes:

```text
evaluation/pv_stage0_baseline/           # example
├── evaluation_summary.json
├── eval_scenario_confirmation.txt
├── reward_time_analysis.txt
├── rollouts/rollout_*.csv
├── baseline_rollouts/sun_tracking/
├── baseline_rollouts/fixed_no_motion/
└── diagnostics/                         # diagnose_tracking.py
    ├── tracking_diagnosis.txt
    └── plots/
```

### 2.3 Config modules

| Stage | Module | Training MDP | Eval (inherit) |
|-------|--------|--------------|----------------|
| **0 clearsky** | `stage0_single_day.py` | `2020-06-15`, `clearsky`, `randomize_day=False` | same |
| **0 paper** | `stage0_single_day_mbpo_paper.py` | `2020-06-21`, `historical`, movement penalty | same |
| **1** | `0.py` | summer, `clearsky`, `randomize_day=True` | 4 dates in `verified_dates.STAGE1_FIXED_EVAL_DATES` |
| **2** | `stage2_random_weather.py` | summer, `historical`, i.i.d. | `STAGE2_FIXED_EVAL_DATES` |
| **3** | `stage3_fullyear_random_clean_split.py` | full year (all days) | seed-based eval; optional `STAGE3_STRESS_TEST_DATES` for diagnostics |

Confirm `epoch_length=78`, `periods=79` with `verify_training_config.py` after merge.

---

## 3. Phase gates (automated)

After `evaluate_agent.py --compare-baselines`:

| Gate | Default | Meaning |
|------|---------|---------|
| Energy | learned / sun ≥ **0.95** | Harvest vs sun tracker |
| Action L1 | ratio ≥ **0.50** (alt ≥ 5°) | Not quasi-static |
| Tilt error | mean \|tilt − zenith\| ≤ **10°** | Zenith tracking |
| vs fixed | learned energy > fixed | Not only static pose |

`diagnose_tracking.py --gate` → exit **0** = pass.

Cloudy / movement-penalty runs may need relaxed thresholds — document any override in your study notes.

---

## 4. Stage 0 — clearsky single day

**Goal:** Single-day clearsky proof vs `sun_tracking` on `2020-06-15`.

**Helper:** `./scripts/run_stage0_baseline_trial.sh {train|plot|eval|gate}`

### Preflight

```bash
python scripts/verify_training_config.py --config examples.config.pv_tracking.stage0_single_day
python scripts/validate_pv_rollouts.py --config-path examples/config/pv_tracking/stage0_single_day.py
```

### Train

```bash
./scripts/run_stage0_baseline_trial.sh train
```

### Verify `params.json`

```bash
export TRIAL=$(cat sequential_stage_artifacts/stage0_baseline_trial_dir.txt)
python -c "
import json, sys
v=json.load(open(sys.argv[1]+'/params.json'))
e=v['environment_params']['training']['kwargs']
assert 'stage0' in (v.get('config_version') or '')
assert e.get('randomize_day') is False
assert e.get('start_date')==e.get('end_date')=='2020-06-15'
assert e.get('weather_source')=='clearsky'
print('OK Stage 0 clearsky')
" "$TRIAL"
```

### Evaluate

```bash
./scripts/run_stage0_baseline_trial.sh eval
```

Equivalent:

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_stage0_baseline \
  --eval-protocol inherit \
  --max-path-length 78 \
  --compare-baselines \
  --num-rollouts 10
```

### Gate

```bash
./scripts/run_stage0_baseline_trial.sh gate
```

---

## 5. Stage 0 — cloudy historical + movement (paper track)

**Goal:** Single overcast historical day (`2020-06-21`), movement penalty, MBPO paper-style kwargs.

**Helper:** `./scripts/run_stage0_paper_trial.sh {train|plot|eval|gate|verify}`

| Artifact | Path |
|----------|------|
| Plots | `training_plots/stage0_mbpo_paper/` |
| Eval | `evaluation/pv_stage0_mbpo_paper_movement/` |

### Evaluate (must match training weather)

```bash
./scripts/run_stage0_paper_trial.sh eval
```

Uses **`--eval-protocol inherit`** only — **do not** pass `--eval-weather-source clearsky` (that was a common mismatch).

### Gate

```bash
./scripts/run_stage0_paper_trial.sh gate
```

---

## 6. Stage 1 — summer i.i.d. clearsky

**Goal:** Random summer days, clearsky irradiance model, fixed summer hold-outs for checkpoint selection and post-train eval.

**Hold-out dates:** `2020-06-07`, `2020-06-15`, `2020-07-15`, `2020-08-01` (`STAGE1_FIXED_EVAL_DATES`).

### Train

```bash
mbpo run_local examples.development \
  --config=examples.config.pv_tracking.0 \
  --gpus=0 --trial-gpus=0 --cpus=10 --trial-cpus=4 \
  --temp-dir=$PWD/.ray_tmp/stage1
```

### Evaluate

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_stage1_clearsky_summer \
  --eval-protocol inherit \
  --max-path-length 78 \
  --compare-baselines \
  --num-rollouts 10
```

### Gate

```bash
python scripts/diagnose_tracking.py \
  --eval-dir evaluation/pv_stage1_clearsky_summer \
  --trial-dir "$TRIAL" \
  --progress-csv "$TRIAL/progress.csv" \
  --verify-env --gate
```

---

## 7. Stage 2 — summer i.i.d. historical weather

**Goal:** Same summer day catalog and hold-outs as Stage 1, but **`weather_source=historical`** (PVGIS TMY). Isolates weather robustness before Stage 3.

**Config:** `stage2_random_weather.py` (filename legacy; weather is historical).

### Train

```bash
mbpo run_local examples.development \
  --config=examples.config.pv_tracking.stage2_random_weather \
  --temp-dir=$PWD/.ray_tmp/stage2
```

### Evaluate

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_stage2_random_weather_summer \
  --eval-protocol inherit \
  --max-path-length 78 \
  --compare-baselines \
  --num-rollouts 10
```

### Gate

```bash
python scripts/diagnose_tracking.py \
  --eval-dir evaluation/pv_stage2_random_weather_summer \
  --trial-dir "$TRIAL" --progress-csv "$TRIAL/progress.csv" \
  --verify-env --gate
```

---

## 8. Stage 3 — full-year stochastic RL

**Config:** `stage3_fullyear_random_clean_split.py`  
**Protocol:** [RL_EVAL_PROTOCOL.md](RL_EVAL_PROTOCOL.md) · **Workflow:** [STAGE3_FULLYEAR_WORKFLOW.md](STAGE3_FULLYEAR_WORKFLOW.md)

| Item | Setting |
|------|---------|
| Training support | All calendar days `2020-01-01`…`2020-12-31` (no `excluded_dates`) |
| Episode variability | `randomize_day` (annual weather scenarios); optional `randomize_initial_orientation` |
| Irradiance augmentation | `irradiance_perturbation_std=0` (paper default) |
| In-train checkpoint eval | Same annual support; independent seeds (no fixed calendar hold-out) |
| Post-train eval | Frozen policy; `--eval-seed-base` rollouts over full year |

### Train

```bash
./scripts/run_stage3_fullyear.sh train
# or:
mbpo run_local examples.development \
  --config=examples.config.pv_tracking.stage3_fullyear_random_clean_split \
  --temp-dir=$PWD/.ray_tmp/stage3
```

### Evaluate

```bash
./scripts/run_stage3_fullyear.sh eval
python scripts/evaluate_fullyear_mc.py "$CKPT" \
  --outdir evaluation/pv_stage3_fullyear_mc \
  --date-set annual --num-rollouts 40 --eval-seed-base 100000
```

Optional fixed-date stress panel: `--date-set stress_test` (not the primary RL test set).

---

## 9. Training vs evaluation environments

| Phase | Source | Controls |
|-------|--------|----------|
| **Training rollouts** | `environment_kwargs` | Day range, `randomize_day`, `weather_source`, penalty, ξ |
| **In-training best checkpoint** | `evaluation_environment_kwargs` | Same annual support as training; Stage 3 uses seeds not fixed dates |
| **Post-train science eval** | `evaluate_agent.py` | **`--eval-protocol inherit`**; independence via `--eval-seed-base`, not calendar hold-outs (Stage 3) |

`evaluation_environment_kwargs` in the config should use the **same** `weather_source` as training. `verify_training_config.py` errors on a train/eval weather mismatch.

---

## 10. Optional: imitation warm-start

```bash
python scripts/collect_sun_demonstrations.py \
  --config examples.config.pv_tracking.stage0_single_day \
  --out demonstration/pv_stage0_sun.npz \
  --num-episodes 200
```

BC pretrain is not wired into `mbpo run_local` by default.

---

## 11. What we do not claim

- No proof of global optimality on pvlib physics.
- `sun_tracking` is a strong heuristic; cloudy days and movement penalties change gate interpretation.
- Scripts provide **engineering verification** (config, pvlib path, fairness, gates), not optimality proofs.

---

## Quick reference

| Step | Stage 0 clearsky | Stage 0 paper | Stage 1 | Stage 2 | Stage 3 |
|------|------------------|---------------|---------|---------|---------|
| Config | `stage0_single_day` | `stage0_single_day_mbpo_paper` | `0` | `stage2_random_weather` | `stage3_fullyear_random_clean_split` |
| Helper | `run_stage0_baseline_trial.sh` | `run_stage0_paper_trial.sh` | — | — | — |
| Train weather | clearsky | historical | clearsky | historical | historical |
| Eval dir | `pv_stage0_baseline` | `pv_stage0_mbpo_paper_movement` | `pv_stage1_clearsky_summer` | `pv_stage2_random_weather_summer` | `pv_stage3_fullyear` |
| Eval flags | `inherit` | `inherit` | `inherit` | `inherit` | `inherit` + `--eval-seed-base` |
| Steps | 78 | 78 | 78 | 78 | 78 |

See also: [PV_SIMPLE_WORKFLOW.md](PV_SIMPLE_WORKFLOW.md), [PV_TRACKING_ROOT_CAUSES.md](PV_TRACKING_ROOT_CAUSES.md).
