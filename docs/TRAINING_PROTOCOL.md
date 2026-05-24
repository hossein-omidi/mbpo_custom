# PV tracking — training protocol (phased verification)

This document defines the **MDP**, **sampling**, **phase gates**, and **tooling** for proving the learned policy can match or beat `sun_tracking` before adding weather/season complexity.

**Do not** use full-year random weather until **Stage 1 passes**.  
**Do not** chain calendar days into multi-day episodes unless you explicitly change the env.

---

## 1. Formal MDP (as implemented)

| Symbol | Meaning |
|--------|---------|
| **Episode** | One calendar day, UTC grid `13:30`–`23:15`, **39** control steps (`periods=40` timestamps) |
| **State** `s` | `physical` obs (11-D): solar angles, normalized irradiance, panel pose, `cos_aoi` |
| **Action** `a` | `a ∈ [-1,1]²` → Δtilt ∈ [-5°,5°], Δazimuth ∈ [-10°,10°] per 15 min |
| **Reward** | `r = energy_kwh - movement_penalty × (‖a_tilt‖₁ + ‖a_azimuth‖₁)` (normalized actions) |
| **Termination** | After 39 steps on that day; **no** carry-over to the next calendar day |
| **Day sampling** | Each `reset()`: pick one day from `[start_date, end_date]` (uniform if `randomize_day=True`) |

**Not** a single “year-long” trajectory: the date range is a **catalog of independent episodes**.

---

## 2. Config modules (by stage)

| Stage | Module | Purpose |
|-------|--------|---------|
| **0** | `examples/config/pv_tracking/stage0_single_day.py` | One day (`2020-06-21`), `randomize_day=False`, clearsky, `movement_penalty=0` |
| **1** | `examples/config/pv_tracking/0.py` | Summer `2020-06-01`…`2020-08-31`, i.i.d. days, clearsky, training eval on fixed summer dates |
| **Legacy obs** | `examples/config/pv_tracking/1.py` | Same hyperparameters, 15-D obs (retrain required) |

Verify before training:

```bash
python scripts/verify_training_config.py --config examples.config.pv_tracking.stage0_single_day
python scripts/verify_training_config.py --config examples.config.pv_tracking.0
python scripts/validate_pv_rollouts.py --config-path examples/config/pv_tracking/0.py
```

---

## 3. Phase gates (automated)

After `evaluate_agent.py` with `--compare-baselines`, run:

```bash
python scripts/diagnose_tracking.py \
  --eval-dir evaluation/pv_stage1_clearsky_summer \
  --gate \
  --min-energy-ratio 0.95 \
  --min-action-ratio 0.5 \
  --max-tilt-error-deg 10
```

| Gate | Default | Meaning |
|------|---------|---------|
| Energy | mean(learned) / mean(sun) ≥ **0.95** | Primary harvest metric |
| Action L1 | ratio ≥ **0.5** on productive steps (alt ≥ 5°) | Mean policy not collapsed |
| Tilt error | mean \|tilt − zenith\| ≤ **10°** | Orientation tracks sun |
| vs fixed | learned energy > fixed | Not a static pose trick |

Exit code **0** = pass, **1** = fail (CI-friendly).

Verified failure analysis for the reference checkpoint: [PV_TRACKING_ROOT_CAUSES.md](PV_TRACKING_ROOT_CAUSES.md).

**Stage 0** (single day): same gates on eval with  
`--fixed-eval-dates 2020-06-21 --eval-weather-source clearsky`.

**Stage 1** (summer): use dates in `STAGE1_FIXED_EVAL_DATES` in `0.py`.

---

## 4. Training vs evaluation environments

- **Training sampler**: uses `environment_kwargs` (random summer days in Stage 1).
- **In-training checkpoint metric**: uses `evaluation_environment_kwargs` when set (fixed summer dates in Stage 1) so `best_eval_checkpoint` matches hold-out science.
- **Post-train eval**: `evaluate_agent.py` reads checkpoint `params.json`; override with `--fixed-eval-dates` / `--eval-weather-source` as needed.

---

## 5. Evaluation commands (templates)

### Stage 0

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_stage0_single_day \
  --eval-protocol inherit --max-path-length 39 --deterministic \
  --compare-baselines --eval-weather-source clearsky \
  --fixed-eval-dates 2020-06-21
```

### Stage 1

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_stage1_clearsky_summer \
  --eval-protocol inherit --max-path-length 39 --deterministic \
  --compare-baselines --eval-weather-source clearsky \
  --fixed-eval-dates 2020-06-07,2020-06-21,2020-07-15,2020-08-01
```

See [evaluation/README.md](../evaluation/README.md) for expected gate footer text.

---

## 6. Optional: imitation warm-start (if SAC fails Stage 0)

1. Collect sun-tracker demonstrations:

```bash
python scripts/collect_sun_demonstrations.py \
  --config examples.config.pv_tracking.stage0_single_day \
  --out demonstration/pv_stage0_sun.npz --num-episodes 200
```

2. BC pretrain + SAC fine-tune: not wired into `mbpo run_local` yet; use demonstrations for analysis or a future BC script. Alternatives documented: **TD3** / **PPO** with fixed exploration noise (separate experiment branch).

---

## 7. What we do **not** claim

- No proof that SAC/MBPO converges to global optimum on pvlib.
- `sun_tracking` is a strong heuristic on clearsky; beating it requires reactive, large-enough mean actions.
- Scripts provide **engineering verification** (timing, config merge, gates), not mathematical optimality.

---

## 8. Stage progression

```text
Stage 0 (single day, stationary)  →  gates pass
Stage 1 (summer i.i.d., clearsky) →  gates pass
Stage 2 (random weather)          →  only after Stage 1
Stage 3 (full year catalog)       →  only after Stage 2; consider contextual policy
```
