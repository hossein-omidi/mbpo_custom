# PV tracking — training protocol (A→Z)

Phased procedure for proving the learned policy can match or beat `sun_tracking` on clearsky before adding weather or full-year complexity.

**Rules**

- Do **not** start Stage 1 until **Stage 0 gates pass**.
- Do **not** use full-year historical weather until **Stage 1 gates pass**.
- Do **not** chain calendar days into multi-day episodes (one day = one episode, 39 steps).
- Use **only** the directory names in [§2](#2-directory-layout-one-name-per-purpose) for each stage — do not reuse `evaluation/pv_daylight_utc` for Stage 0/1 science runs (that path is a legacy December hold-out).

---

## 1. Formal MDP (as implemented)

| Symbol | Meaning |
|--------|---------|
| **Episode** | One calendar day, UTC grid `13:30`–`23:15`, **39** control steps (`periods=40` timestamps) |
| **State** `s` | `physical` obs (11-D): solar angles, normalized irradiance, panel pose, `cos_aoi` |
| **Action** `a` | `a ∈ [-1,1]²` → Δtilt ∈ [-5°,5°], Δazimuth ∈ [-10°,10°] per 15 min |
| **Reward** | `r = energy_kwh - movement_penalty × (‖a_tilt‖₁ + ‖a_azimuth‖₁)` |
| **Termination** | After 39 steps; **no** carry-over to the next calendar day |
| **Day sampling** | Each `reset()`: one day from `[start_date, end_date]` if `randomize_day=True` |

---

## 2. Directory layout (one name per purpose)

All paths are relative to the **repo root** unless noted.

### 2.1 Training artifacts (Ray — outside git)

| Purpose | Path pattern | Created by |
|---------|----------------|------------|
| **All PV trials** | `~/ray_mbpo/PVTracking/pv_tracking/seed:<id>_<timestamp><hash>/` | `mbpo run_local` |
| **Merged variant** | `…/params.json` | Ray at trial start |
| **Metrics** | `…/progress.csv` | Training loop |
| **Best checkpoint (in-train eval)** | `…/best_eval_checkpoint/` | `monitor_metric` = `evaluation/return-average` |
| **Latest weights** | `…/latest_checkpoint/` | End of each epoch |

After each run, set:

```bash
export TRIAL=$(ls -td ~/ray_mbpo/PVTracking/pv_tracking/seed:*/ | head -1)
export TRIAL="${TRIAL%/}"
export CKPT="$TRIAL/best_eval_checkpoint"
```

Use **`$CKPT`** for all post-train eval and diagnosis. Do not evaluate a Stage 1 checkpoint with Stage 0 eval flags (or vice versa) without understanding [RC5](PV_TRACKING_ROOT_CAUSES.md).

### 2.2 Evaluation outputs (repo — gitignored except `.gitkeep`)

| Stage | **Canonical `--outdir`** | Must match config |
|-------|--------------------------|-------------------|
| **Stage 0** | `evaluation/pv_stage0_single_day/` | `stage0_single_day.py`, date `2020-06-21` |
| **Stage 1** | `evaluation/pv_stage1_clearsky_summer/` | `0.py`, four summer hold-out dates |
| Legacy / ad hoc | `evaluation/pv_daylight_utc/` | December hold-out — **not** Stage 0/1 gates |

Each eval run creates:

```text
evaluation/pv_stage0_single_day/          # example; use the row for your stage
├── evaluation_summary.json             # eval_config, deterministic flag, metrics
├── evaluation_summary.txt
├── eval_scenario_confirmation.txt
├── reward_time_analysis.txt
├── rollouts/rollout_*.csv              # learned policy
├── baseline_rollouts/
│   ├── sun_tracking/rollout_*.csv
│   └── fixed_no_motion/rollout_*.csv
└── diagnostics/                      # from diagnose_tracking.py
    ├── tracking_diagnosis.txt
    └── plots/
```

### 2.3 Config and optional demonstrations

| Purpose | Path |
|---------|------|
| Stage 0 config | `examples/config/pv_tracking/stage0_single_day.py` |
| Stage 1 config | `examples/config/pv_tracking/0.py` |
| Stage 2 config | `examples/config/pv_tracking/stage2_random_weather.py` (legacy filename; now historical weather) |
| Sun demos (optional) | `demonstration/pv_stage0_sun.npz` |

### 2.4 Config modules (by stage)

| Stage | Module | `CONFIG_VERSION` prefix | Training MDP | Post-train eval dates |
|-------|--------|-------------------------|--------------|------------------------|
| **0** | `stage0_single_day.py` | `pv_tracking_stage0_…` | Single day `2020-06-21`, `randomize_day=False` | `--fixed-eval-dates 2020-06-21` |
| **1** | `0.py` | `pv_tracking_stage1_…` | Summer `2020-06-01`…`2020-08-31`, `randomize_day=True` | `2020-06-07,2020-06-21,2020-07-15,2020-08-01` |
| **2** | `stage2_random_weather.py` | `pv_tracking_stage2_…` | Summer `2020-06-01`…`2020-08-31`, `randomize_day=True`, `weather_source='historical'` (legacy filename retained) | `2020-06-07,2020-06-21,2020-07-15,2020-08-01` |

Epoch counts and exploration steps live in the config files (not duplicated here). Always confirm with `verify_training_config.py` and `params.json` after the trial starts.

---

## 3. Phase gates (automated)

Run **after** `evaluate_agent.py --compare-baselines` into the **canonical** `evaluation/pv_stage*` directory for that stage.

| Gate | Default | Meaning |
|------|---------|---------|
| Energy | mean(learned) / mean(sun) ≥ **0.95** | Primary harvest metric |
| Action L1 | ratio ≥ **0.50** (productive sun, alt ≥ 5°) | Deploy policy is not quasi-static vs sun |
| Tilt error | mean \|tilt − zenith\| ≤ **10°** | Tracks zenith |
| vs fixed | learned energy > `fixed_no_motion` | Not only a static pose |

Exit code **0** = pass, **1** = fail (CI-friendly).

Failure analysis for a reference Stage 1 checkpoint: [PV_TRACKING_ROOT_CAUSES.md](PV_TRACKING_ROOT_CAUSES.md).

---

## 4. Stage 0 — procedure A→Z (single day)

**Goal:** On `2020-06-21`, clearsky, `movement_penalty=0`, learned policy passes gates vs `sun_tracking`.

### A — Preflight (repo)

```bash
cd /path/to/mbpo
python scripts/verify_training_config.py \
  --config examples.config.pv_tracking.stage0_single_day
python scripts/validate_pv_rollouts.py \
  --config-path examples/config/pv_tracking/stage0_single_day.py
```

**Pass:** `verify_training_config` prints `PASS`; rollout validator exits 0.

### B — Dry run (optional, no learning)

```bash
mbpo run_example_dry examples.development \
  --config=examples.config.pv_tracking.stage0_single_day \
  --gpus=0 --trial-gpus=0 --cpus=2 --trial-cpus=1
```

**Pass:** log shows `config_version` with `stage0`, `periods=40`, `observation_mode='physical'`.

### C — Train

```bash
mbpo run_local examples.development \
  --config=examples.config.pv_tracking.stage0_single_day \
  --gpus=0 --trial-gpus=0 --cpus=4 --trial-cpus=2
```

### D — Verify trial contract (`params.json`)

```bash
export TRIAL=$(ls -td ~/ray_mbpo/PVTracking/pv_tracking/seed:*/ | head -1)
export TRIAL="${TRIAL%/}"
export CKPT="$TRIAL/best_eval_checkpoint"

python scripts/verify_training_config.py \
  --config examples.config.pv_tracking.stage0_single_day
python -c "
import json
v=json.load(open('$TRIAL/params.json'))
print('config_version:', v.get('config_version'))
k=v['algorithm_params']['kwargs']
e=v['environment_params']['training']['kwargs']
assert 'stage0' in (v.get('config_version') or '').lower(), 'wrong config — not Stage 0'
assert e.get('randomize_day') is False, 'Stage 0 requires randomize_day=False'
assert e.get('start_date')==e.get('end_date')=='2020-06-21', 'wrong training day'
print('OK: Stage 0 params.json')
"
```

**Pass:** `config_version` contains `stage0`; `randomize_day=False`; dates `2020-06-21`.

### E — Evaluate (canonical outdir only)

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_stage0_single_day \
  --eval-protocol inherit \
  --max-path-length 39 \
  --compare-baselines \
  --eval-weather-source clearsky \
  --fixed-eval-dates 2020-06-21 \
  --num-rollouts 10
```

`--deterministic` is **on by default** (`tanh(μ)`); use `--stochastic` only for ablations.

**Pass:** `evaluation/pv_stage0_single_day/evaluation_summary.json` exists; `eval_config.fixed_eval_dates` = `["2020-06-21"]`; `movement_penalty` = `0.0`; `deterministic` = `true`.

### F — Diagnose and gate

```bash
python scripts/diagnose_tracking.py \
  --eval-dir evaluation/pv_stage0_single_day \
  --trial-dir "$TRIAL" \
  --progress-csv "$TRIAL/progress.csv" \
  --verify-env \
  --gate \
  --min-energy-ratio 0.95 \
  --min-action-ratio 0.5 \
  --max-tilt-error-deg 10
```

**Pass:** exit code **0**; `diagnostics/tracking_diagnosis.txt` ends with `GATE STATUS: PASS`.

### G — Advance

Only after **F** passes → start [Stage 1](#5-stage-1--procedure-az-summer-clearsky).

If gates fail with small actions / wrong sign, see [PV_TRACKING_ROOT_CAUSES.md](PV_TRACKING_ROOT_CAUSES.md) and [§7](#7-optional-imitation-warm-start).

---

## 5. Stage 1 — procedure A→Z (summer clearsky)

**Goal:** Summer hold-out dates, clearsky, gates pass on `evaluation/pv_stage1_clearsky_summer/`.

Hold-out dates (must match eval flags): `2020-06-07`, `2020-06-21`, `2020-07-15`, `2020-08-01` (`STAGE1_FIXED_EVAL_DATES` in `0.py`).

### A — Preflight

```bash
python scripts/verify_training_config.py \
  --config examples.config.pv_tracking.0
python scripts/validate_pv_rollouts.py \
  --config-path examples/config/pv_tracking/0.py
```

### B — Dry run (optional)

```bash
mbpo run_example_dry examples.development \
  --config=examples.config.pv_tracking.0 \
  --gpus=0 --trial-gpus=0 --cpus=2 --trial-cpus=1
```

### C — Train

```bash
mbpo run_local examples.development \
  --config=examples.config.pv_tracking.0 \
  --gpus=0 --trial-gpus=0 --cpus=4 --trial-cpus=2
```

### D — Verify trial contract

```bash
export TRIAL=$(ls -td ~/ray_mbpo/PVTracking/pv_tracking/seed:*/ | head -1)
export TRIAL="${TRIAL%/}"
export CKPT="$TRIAL/best_eval_checkpoint"

python -c "
import json
v=json.load(open('$TRIAL/params.json'))
print('config_version:', v.get('config_version'))
k=v['algorithm_params']['kwargs']
e=v['environment_params']['training']['kwargs']
assert 'stage1' in (v.get('config_version') or '').lower(), 'wrong config — not Stage 1'
assert e.get('randomize_day') is True, 'Stage 1 training uses randomize_day=True'
print('n_epochs', k.get('n_epochs'), 'min_alpha', k.get('min_alpha'))
print('OK: Stage 1 params.json')
"
```

### E — Evaluate

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_stage1_clearsky_summer \
  --eval-protocol inherit \
  --max-path-length 39 \
  --compare-baselines \
  --eval-weather-source clearsky \
  --fixed-eval-dates 2020-06-07,2020-06-21,2020-07-15,2020-08-01 \
  --num-rollouts 10
```

### F — Diagnose and gate

```bash
python scripts/diagnose_tracking.py \
  --eval-dir evaluation/pv_stage1_clearsky_summer \
  --trial-dir "$TRIAL" \
  --progress-csv "$TRIAL/progress.csv" \
  --verify-env \
  --gate \
  --min-energy-ratio 0.95 \
  --min-action-ratio 0.5 \
  --max-tilt-error-deg 10
```

**Pass:** exit code **0** on `evaluation/pv_stage1_clearsky_summer/`.

### G — Advance

Only after **F** passes → Stage 2 (historical weather), using
`examples/config/pv_tracking/stage2_random_weather.py` and a new evaluation
directory.

---

## 6. Training vs evaluation environments

| Phase | Source | What it controls |
|-------|--------|------------------|
| **Training rollouts** | `environment_kwargs` in config | Day catalog, `randomize_day`, weather |
| **In-training checkpoint pick** | `evaluation_environment_kwargs` | Fixed dates for `best_eval_checkpoint` |
| **Post-train science eval** | `evaluate_agent.py` CLI | `--fixed-eval-dates`, `--eval-weather-source`, `--outdir` |

Post-train eval must use the **canonical `evaluation/pv_stage*`** directory for that stage. `evaluate_agent.py` reads checkpoint `params.json` for defaults but **overrides** day sampling via CLI flags above.

`real_ratio=1.0` removes model-rollout bias; it does **not** fix small or wrong-signed deploy actions ([RC1–RC4](PV_TRACKING_ROOT_CAUSES.md)).

---

## 6.5 Stage 2 — procedure A->Z (summer historical weather)

**Goal:** Same summer hold-out dates, same physical observation/state contract,
but now historical weather in both training and post-train evaluation. This isolates
weather robustness before any Stage 3 full-year expansion.

**Config:** `examples/config/pv_tracking/stage2_random_weather.py` (legacy filename retained)

**Important:** Stage 2 intentionally preserves the current working training
procedure and hyperparameters from Stage 1. It inherits the MBPO settings from
Stage 1, including model-based rollouts; it is not a pure real-data SAC
ablation.

### A — Preflight

```bash
python scripts/verify_training_config.py \
  --config examples.config.pv_tracking.stage2_random_weather
python scripts/validate_pv_rollouts.py \
  --config-path examples/config/pv_tracking/stage2_random_weather.py
```

### B — Dry run (optional)

```bash
mbpo run_example_dry examples.development \
  --config=examples.config.pv_tracking.stage2_random_weather \
  --gpus=0 --trial-gpus=0 --cpus=2 --trial-cpus=1
```

### C — Train

```bash
mbpo run_local examples.development \
  --config=examples.config.pv_tracking.stage2_random_weather \
  --gpus=0 --trial-gpus=0 --cpus=4 --trial-cpus=2
```

### D — Verify trial contract

```bash
export TRIAL=$(ls -td ~/ray_mbpo/PVTracking/pv_tracking/seed:*/ | head -1)
export TRIAL="${TRIAL%/}"
export CKPT="$TRIAL/best_eval_checkpoint"

python -c "
import json
v=json.load(open('$TRIAL/params.json'))
print('config_version:', v.get('config_version'))
k=v['algorithm_params']['kwargs']
e=v['environment_params']['training']['kwargs']
assert 'stage2' in (v.get('config_version') or '').lower(), 'wrong config — not Stage 2'
assert e.get('randomize_day') is True, 'Stage 2 training uses randomize_day=True'
assert e.get('weather_source') == 'historical', 'Stage 2 training must use historical weather'
assert e.get('observation_mode') == 'physical', 'Stage 2 should stay on physical observations'
print('n_epochs', k.get('n_epochs'), 'min_alpha', k.get('min_alpha'), 'real_ratio', k.get('real_ratio'))
print('OK: Stage 2 params.json')
"
```

### E — Evaluate

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_stage2_random_weather_summer \
  --eval-protocol inherit \
  --max-path-length 39 \
  --compare-baselines \
  --eval-weather-source historical \
  --fixed-eval-dates 2020-06-07,2020-06-21,2020-07-15,2020-08-01 \
  --num-rollouts 10
```

### F — Diagnose and gate

```bash
python scripts/diagnose_tracking.py \
  --eval-dir evaluation/pv_stage2_random_weather_summer \
  --trial-dir "$TRIAL" \
  --progress-csv "$TRIAL/progress.csv" \
  --verify-env \
  --gate \
  --min-energy-ratio 0.95 \
  --min-action-ratio 0.5 \
  --max-tilt-error-deg 10
```

**Pass:** exit code **0** on `evaluation/pv_stage2_random_weather_summer/`.

### G — Advance

Only after **F** passes → define Stage 3 (full-year catalog) with a new config
and a new evaluation directory.

---

## 7. Optional: imitation warm-start

If Stage 0 gates fail on action magnitude or sign:

```bash
python scripts/collect_sun_demonstrations.py \
  --config examples.config.pv_tracking.stage0_single_day \
  --out demonstration/pv_stage0_sun.npz \
  --num-episodes 200
```

BC pretrain + SAC fine-tune is not wired into `mbpo run_local` yet; use demos for analysis or a future BC script.

---

## 8. Stage progression (summary)

```text
Stage 0  →  evaluation/pv_stage0_single_day/  →  diagnose --gate  →  PASS
Stage 1  →  evaluation/pv_stage1_clearsky_summer/  →  diagnose --gate  →  PASS
Stage 2  →  evaluation/pv_stage2_random_weather_summer/  →  diagnose --gate  →  PASS
Stage 3  →  full-year catalog (define new config + NEW eval dir name)
```

---

## 9. What we do **not** claim

- No proof that SAC/MBPO reaches a global optimum on pvlib.
- `sun_tracking` is a strong clearsky heuristic; beating it needs sufficient **deploy** action scale (`tanh(μ)` at eval).
- Scripts provide **engineering verification** (config merge, fairness, gates), not optimality proofs.

---

## Quick reference

| Step | Stage 0 | Stage 1 | Stage 2 |
|------|---------|---------|---------|
| Config | `stage0_single_day` | `0` | `stage2_random_weather` |
| Eval dir | `evaluation/pv_stage0_single_day` | `evaluation/pv_stage1_clearsky_summer` | `evaluation/pv_stage2_random_weather_summer` |
| Fixed dates | `2020-06-21` | four summer dates in `0.py` | same four summer dates |
| Eval weather | `clearsky` | `clearsky` | `random` |
| Gate command | `diagnose_tracking.py --eval-dir evaluation/pv_stage0_single_day --gate` | same with `pv_stage1_clearsky_summer` | same with `pv_stage2_random_weather_summer` |

See also: [evaluation/README.md](../evaluation/README.md), [STAGE1_CLEARSKY_TRAINING.md](STAGE1_CLEARSKY_TRAINING.md), [PV_TRACKING_ROOT_CAUSES.md](PV_TRACKING_ROOT_CAUSES.md).
