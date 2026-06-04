# Full-Year Stage 3 — Simple Training & Evaluation

End-to-end workflow for **finite-horizon annual-scenario MBPO-SAC** PV tracking (paper path).

**Canonical config:** `examples/config/pv_tracking/stage3_fullyear_stable_mbpo.py`  
**Helper script:** `scripts/run_stage3_fullyear.sh`  
**Protocol details:** [RL_EVAL_PROTOCOL.md](RL_EVAL_PROTOCOL.md) · [STAGE3_FULLYEAR_WORKFLOW.md](STAGE3_FULLYEAR_WORKFLOW.md)

Legacy ablation (high `real_ratio=0.5`, unstable Q): `stage3_fullyear_random_clean_split.py`

---

## Commands you need (copy-paste)

```bash
cd /home/user01/mbpo_custom
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mbpo

./scripts/run_stage3_fullyear.sh status    # trial + best_eval_checkpoint

# --- Refresh eval only (keep training checkpoint; no stale plots) ---
rm -rf evaluation/pv_stage3_fullyear evaluation/pv_stage3_fullyear_mc
rm -rf training_plots/stage3_fullyear

./scripts/run_stage3_fullyear.sh plot      # training curves (E[R] ± σ)
./scripts/run_stage3_fullyear.sh all-eval  # eval + mc-eval (16 rollouts each)

cat evaluation/pv_stage3_fullyear/EVAL_STATISTICS.txt
cat evaluation/pv_stage3_fullyear/evaluation_summary.txt
```

**New training from scratch:** run `verify` then `train` (see §1–3). Do **not** delete `~/ray_mbpo/.../seed:*` if you only want fresh eval.

```bash
./scripts/run_stage3_fullyear.sh verify
./scripts/run_stage3_fullyear.sh train      # fresh only; no --restore
```

More rollouts: `NUM_ROLLOUTS=32 ./scripts/run_stage3_fullyear.sh eval`

---

## What this run does

- **Training:** sample a calendar day each episode from the full year (`2020-01-01`…`2020-12-31`); historical TMY weather; pvlib deterministic physics; `irradiance_perturbation_std=0`.
- **MBPO:** BNN on real replay transitions; short model rollouts from real states; SAC on mixed real/model batches (`real_ratio=0.10`, max imagined length 5).
- **Evaluation:** frozen policy on the **real** env only; Monte Carlo over random days with independent seeds (`eval_seed_base=100000`); **no** calendar hold-out as the main test.
- **Figures:** `evaluation/pv_stage3_fullyear/paper_figures/` (and MC variant under `pv_stage3_fullyear_mc/`).

---

## Rules (read once)

| Do | Don't |
|----|--------|
| Use `stage3_fullyear_stable_mbpo` | Use `run_sequential_stages.sh` for this experiment |
| Train **fresh** (no `--restore`) for a new paper run | Reuse old replay pools or mix Stage 1/2 checkpoints |
| Use `./scripts/run_stage3_fullyear.sh` for verify/train/eval | Rely on old `pv_stage3_final_clean_test` hold-out eval dirs |
| Post-train: `./scripts/run_stage3_fullyear.sh eval` or `all-eval` | Treat fixed-date stress tests as the main metric |

---

## 0. Every session — shell setup

```bash
cd /home/user01/mbpo_custom
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mbpo
chmod +x scripts/run_stage3_fullyear.sh
```

**First time only** (if env missing — see `environment/pv-env.yml`):

```bash
cd /home/user01/mbpo_custom
conda env create -f environment/pv-env.yml
conda activate mbpo
pip install 'pvlib==0.10.4' 'tables==3.7.0' --no-deps
pip install 'opencv-python-headless==4.2.0.34'
pip install -e viskit
pip install -e .
```

**Historical weather catalog** (once per machine, or after deleting `data/pv_weather/`):

```bash
python scripts/prepare_default_historical_weather.py
```

---

## 1. Optional — remove previous checkpoints & outputs

Only needed when you want a **clean fresh trial** (new paper run).  
**This does not delete other Ray experiments** outside PV tracking if you narrow the paths below.

### 1a. List existing Stage 3 trials (inspect first)

```bash
ls -lt ~/ray_mbpo/PVTracking/pv_tracking/seed:*/params.json 2>/dev/null | head -5
grep -l 'stage3_fullyear' ~/ray_mbpo/PVTracking/pv_tracking/seed:*/params.json 2>/dev/null | head
```

### 1b. Remove Stage 3 Ray trials (checkpoints + progress.csv)

Removes **all** trials under the PV tracking log root:

```bash
# Preview
ls -d ~/ray_mbpo/PVTracking/pv_tracking/seed:* 2>/dev/null | wc -l

# Delete (irreversible)
rm -rf ~/ray_mbpo/PVTracking/pv_tracking/seed:*
```

To delete **one** trial only:

```bash
rm -rf ~/ray_mbpo/PVTracking/pv_tracking/seed:YOUR_SEED_DIR
```

### 1c. Remove eval + plots only (usual — keeps Ray checkpoint)

Run before re-eval so old CSVs/figures do not mix with the new run:

```bash
cd /home/user01/mbpo_custom
rm -rf evaluation/pv_stage3_fullyear evaluation/pv_stage3_fullyear_mc
rm -rf training_plots/stage3_fullyear
```

Then: `./scripts/run_stage3_fullyear.sh plot` and `./scripts/run_stage3_fullyear.sh all-eval`.

### 1d. Full local wipe (eval + plots + artifact pointer + Ray temp)

Does **not** delete `~/ray_mbpo/.../seed:*` (see §1b for that).

```bash
cd /home/user01/mbpo_custom
rm -rf .ray_tmp/stage3_fullyear
rm -rf evaluation/pv_stage3_fullyear evaluation/pv_stage3_fullyear_mc
rm -rf evaluation/pv_stage3_fullyear_mc_stress evaluation/pv_stage3_posttrain
rm -rf evaluation/pv_stage3_final_clean_test evaluation/pv_stage3_validation_clean_split
rm -rf training_plots/stage3_fullyear training_plots/stage3_clean_split_latest
rm -rf verification/stage3_fullyear
rm -f sequential_stage_artifacts/stage3_fullyear_trial_dir.txt
```

### 1e. Delete training checkpoints too (brand-new paper run)

```bash
rm -rf ~/ray_mbpo/PVTracking/pv_tracking/seed:*
# then run §1d, then verify + train
```

Skip §1b–1e if you **continue** the same trial (`resume_stage3_refinement.sh`).

---

## 2. Verification (required before train)

Single command (runs preflight + state-space checks):

```bash
cd /home/user01/mbpo_custom
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mbpo

./scripts/run_stage3_fullyear.sh verify
```

**Pass:** exit code 0 and `verification/stage3_fullyear/preflight_report.txt` ends with `PASS`.

**Optional extra checks:**

```bash
python scripts/verify_training_config.py --config examples.config.pv_tracking.stage3_fullyear_stable_mbpo
python scripts/check_pv_env.py --observation-mode physical --weather-source historical --validate-weather
pytest tests/test_pv_tracking_audit.py -q
mbpo run_example_dry examples.development \
  --config=examples.config.pv_tracking.stage3_fullyear_stable_mbpo \
  --gpus=0 --trial-gpus=0 --cpus=2 --trial-cpus=1
```

---

## 3. Training (fresh run)

```bash
cd /home/user01/mbpo_custom
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mbpo

# Default CPUs from scripts/pv_cpu_env.sh (override if needed):
# export PV_CPU_PROFILE=full
./scripts/run_stage3_fullyear.sh train
```

Trial directory is saved to:

`sequential_stage_artifacts/stage3_fullyear_trial_dir.txt`

**Equivalent manual train** (same as helper):

```bash
mbpo run_local examples.development \
  --config=examples.config.pv_tracking.stage3_fullyear_stable_mbpo \
  --gpus=0 --trial-gpus=0 \
  --cpus=4 --trial-cpus=2 \
  --temp-dir=/home/user01/mbpo_custom/.ray_tmp/stage3_fullyear
```

Do **not** pass `--restore` for a new paper experiment.

### Set active trial (required for manual commands)

```bash
./scripts/run_stage3_fullyear.sh status    # shows Resolved TRIAL path
export TRIAL=$(cat sequential_stage_artifacts/stage3_fullyear_trial_dir.txt)
export CKPT="$TRIAL/best_eval_checkpoint"
```

**Logs:** `~/ray_mbpo/PVTracking/pv_tracking/seed:*/` (not empty `mbpo_runs/` unless you train there).

---

## 4. Monitor while training

**Training curves:**

```bash
./scripts/run_stage3_fullyear.sh plot
```

**Refresh every 60s:**

```bash
export TRIAL=$(cat sequential_stage_artifacts/stage3_fullyear_trial_dir.txt)
watch -n 60 "./scripts/run_stage3_fullyear.sh plot"
```

**Last 20 epochs** (`TRIAL` must be set):

```bash
export TRIAL=$(cat sequential_stage_artifacts/stage3_fullyear_trial_dir.txt)
python - "$TRIAL/progress.csv" <<'PY'
import pandas as pd, sys
df = pd.read_csv(sys.argv[1])
cols = [c for c in [
    'epoch', 'evaluation/return-average', 'training/return-average',
    'model/val_loss', 'model_rollout_length', 'real_batch_ratio',
    'alpha', 'Q_loss',
] if c in df.columns]
print(df[cols].tail(20).to_string(index=False))
PY
```

---

## 5. Evaluation (frozen policy, real env)

All methods use the **same** env contract, **T=78**, matched seeds, pvlib power path.

### 5a. Main paper eval (recommended)

**16 MC rollouts** (default), matched seeds, real env only. Re-run when training updates `best_eval_checkpoint`:

```bash
./scripts/run_stage3_fullyear.sh eval
```

**Outputs:** `evaluation/pv_stage3_fullyear/`

| Path | Content |
|------|---------|
| `EVAL_STATISTICS.txt` | E[R], σ (ddof=1), baselines, clock — not SAC entropy |
| `paper_figures/` | Return process E[R]±σ, bars, calendar-season plots |
| `evaluation_summary.txt` | Aggregates + **calendar** seasons |
| `evaluation_rewards.png` | Per-rollout R with E[R] ± σ band |
| `rollouts/`, `baseline_rollouts/` | CSVs (learned / sun / **fixed_no_motion** = action 0, zero movement) |

More rollouts:

```bash
NUM_ROLLOUTS=32 ./scripts/run_stage3_fullyear.sh eval
```

### 5b. Monte Carlo eval + season bars

```bash
./scripts/run_stage3_fullyear.sh mc-eval
# → evaluation/pv_stage3_fullyear_mc/
```

Both eval + MC:

```bash
./scripts/run_stage3_fullyear.sh all-eval
```

### 5c. Post-train helper (alternative to 5a)

```bash
./run_stage3_posttrain.sh latest eval
# → evaluation/pv_stage3_posttrain/
```

Modes: `eval` (default), `rank`, `mc`, `stress`, `all` — see script header.

### 5d. Optional diagnostic only (NOT main paper metric)

Fixed calendar stress panel:

```bash
./scripts/run_stage3_fullyear.sh mc-eval-stress
# → evaluation/pv_stage3_fullyear_mc_stress/
```

### 5e. Manual eval (same as helper)

```bash
python scripts/evaluate_agent.py "$CKPT" \
  --outdir evaluation/pv_stage3_fullyear \
  --eval-protocol inherit \
  --max-path-length 78 \
  --compare-baselines \
  --num-rollouts 16 \
  --max-rollout-plots 4 \
  --eval-seed-base 100000
```

---

## 6. Read results

```bash
cat evaluation/pv_stage3_fullyear/EVAL_STATISTICS.txt
cat evaluation/pv_stage3_fullyear/evaluation_summary.txt
cat evaluation/pv_stage3_fullyear/paper_figures/EVAL_PROTOCOL_README.txt
ls evaluation/pv_stage3_fullyear/paper_figures/
```

Optional diagnostics:

```bash
./scripts/run_stage3_fullyear.sh gate
cat evaluation/pv_stage3_fullyear/diagnostics/tracking_diagnosis.txt
```

---

## 7. Quick copy-paste workflow (full pipeline)

```bash
cd /home/user01/mbpo_custom
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mbpo

# --- Optional clean start ---
# rm -rf ~/ray_mbpo/PVTracking/pv_tracking/seed:*
# rm -rf .ray_tmp/stage3_fullyear evaluation/pv_stage3_fullyear* training_plots/stage3_fullyear verification/stage3_fullyear
# rm -f sequential_stage_artifacts/stage3_fullyear_trial_dir.txt

python scripts/prepare_default_historical_weather.py   # if needed

./scripts/run_stage3_fullyear.sh verify
./scripts/run_stage3_fullyear.sh train

export TRIAL=$(cat sequential_stage_artifacts/stage3_fullyear_trial_dir.txt)
export CKPT="$TRIAL/best_eval_checkpoint"

./scripts/run_stage3_fullyear.sh plot
./scripts/run_stage3_fullyear.sh all-eval

echo "Paper figures: evaluation/pv_stage3_fullyear/paper_figures/"
echo "MC figures:      evaluation/pv_stage3_fullyear_mc/paper_figures/"
```

---

## 8. Helper command reference

```bash
./scripts/run_stage3_fullyear.sh status          # show where trials live + Resolved TRIAL
./scripts/run_stage3_fullyear.sh verify          # preflight
./scripts/run_stage3_fullyear.sh train           # fresh MBPO train
./scripts/run_stage3_fullyear.sh plot            # training curves
./scripts/run_stage3_fullyear.sh eval            # main paper eval + paper_figures/
./scripts/run_stage3_fullyear.sh mc-eval         # annual MC + season plots
./scripts/run_stage3_fullyear.sh mc-eval-stress   # diagnostic fixed dates only
./scripts/run_stage3_fullyear.sh all-eval        # eval + mc-eval
./scripts/run_stage3_fullyear.sh gate            # optional pass/fail diagnostics
```

**Environment overrides:** `NUM_ROLLOUTS=16` (default), `MC_REPLICATES=16`, `MAX_ROLLOUT_PLOTS=4`, `MAX_PAIRED_PLOT_DAYS=6`, `RAY_ROOT=...`, `PV_CPU_PROFILE=full`

---

## 9. Config snapshot (paper)

| Setting | Value |
|---------|--------|
| Days | Full year, `randomize_day=True` |
| Weather | `historical` (PVGIS TMY) |
| Perturbation | `irradiance_perturbation_std=0` |
| Horizon | 78 steps (UTC 13:30–23:15) |
| `real_ratio` | 0.10 |
| `max_model_rollout_length` | 5 |
| `rollout_schedule` | `[60, 600, 1, 5]` |
| `discount` | 0.995 |
| `target_entropy` | `'auto'` |
| `min_alpha` | 0.02 |
| `n_epochs` | 2000 |
| In-train eval | `STAGE3_VALIDATION_DATES` (4 dates), `eval_n_episodes=4` |
| Post-train eval | No calendar hold-out; MC over random days + seeds |

Config version string contains `pv_tracking_stage3_fullyear_stable_mbpo`.


Full clean (new training — deletes checkpoints)

cd /home/user01/mbpo_custom
rm -rf evaluation/pv_stage3_fullyear evaluation/pv_stage3_fullyear_mc
rm -rf training_plots/stage3_fullyear
rm -rf .ray_tmp/stage3_fullyear verification/stage3_fullyear
rm -f sequential_stage_artifacts/stage3_fullyear_trial_dir.txt
rm -rf ~/ray_mbpo/PVTracking/pv_tracking/seed:*
./scripts/run_stage3_fullyear.sh verify
./scripts/run_stage3_fullyear.sh train


Commands (keep your checkpoint, fresh eval)

cd /home/user01/mbpo_custom
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mbpo
# Remove old eval + plots (does NOT delete training)
rm -rf evaluation/pv_stage3_fullyear evaluation/pv_stage3_fullyear_mc
rm -rf training_plots/stage3_fullyear
./scripts/run_stage3_fullyear.sh status
./scripts/run_stage3_fullyear.sh plot
./scripts/run_stage3_fullyear.sh all-eval
cat evaluation/pv_stage3_fullyear/EVAL_STATISTICS.txt
cat evaluation/pv_stage3_fullyear/evaluation_summary.txt

