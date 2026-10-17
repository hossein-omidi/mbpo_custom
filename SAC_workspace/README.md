
### 3.2 Full SAC training

```bash
python -m SAC_workspace.workflows.03_train \
  --run-dir SAC_workspace/runs/sac_seed0 \
  --seed 0 \
  --device cpu
```



### 4.3 Full MBPO training

```bash
python -m MBPO_workspace.workflows.02_train \
  --run-dir MBPO_workspace/runs/mbpo_seed0 \
  --seed 0 \
  --device cpu






6. Evaluate SAC using the new protocol
After the checkpoint exists:
python -m SAC_workspace.workflows.04_evaluate \
  --checkpoint SAC_workspace/runs/sac_seed1/latest.pt \
  --training-csv SAC_workspace/runs/sac_seed1/episodes.csv \
  --all-scenarios \
  --days-per-month 3 \
  --selection-seed 42 \
  --device cpu \
  --outdir SAC_workspace/evaluation/stratified


Important change: --all-scenarios now runs the stratified sampling protocol. To evaluate literally every eligible day as before, use --exhaustive-scenarios.
7. Apply exactly the same dates to MBPO and SAC
For a fair paired comparison, reuse the saved scenario plan:


python -m MBPO_workspace.workflows.03_evaluate \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_seed0/latest.pt \
  --sac-checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
  --selection-plan SAC_workspace/evaluation/sample_plan/scenario_selection.json \
  --skip-poa-opt \
  --device cpu \
  --outdir MBPO_workspace/evaluation/paired_stratified







# Solar Tracking RL — Training, Evaluation & Reporting Guide

**Workspaces:** `SAC_workspace` (model-free Soft Actor-Critic) and `MBPO_workspace` (PyTorch, Ray-free Model-Based Policy Optimization + shared SAC).  
**Version:** SAC v2 + MBPO recovery v2 + advanced reporting.  
**Run all commands from the `mbpo_custom/` repository root.**

## 1. Requirements and setup

Use your working modern Python environment (for example, `simple-worm`), **not** the legacy Python 3.6 `mbpo` environment. The modern training workflows do not require Ray or TensorFlow.

```bash
cd ~/projects/PVLIB_Paper/mbpo_custom
conda activate simple-worm
python --version
python -m pip install -r SAC_workspace/requirements-sac.txt
# Equivalent MBPO requirements file points to SAC requirements:
# python -m pip install -r MBPO_workspace/requirements-mbpo.txt
```

If dependencies and tests already work, **do not reinstall packages unnecessarily**. The NSRDB manifest and year-level CSVs must already exist. Only when the manifest is missing:

```bash
python scripts/prepare_nsrdb_newyork_catalog.py
```

Expected NSRDB manifest: `data/pv_weather/nsrdb/newyork_multiyear_manifest.json`.

Check the installed workspace files:

```bash
ls SAC_workspace/workflows/
ls MBPO_workspace/workflows/
ls data/pv_weather/nsrdb/newyork_multiyear_manifest.json
```

## 2. Verify installation, physics and tests

```bash
# Model-free SAC tests (including real pvlib/NSRDB parity where available)
python -m unittest discover -s SAC_workspace/tests -p 'test_*.py' -v

# Remastered MBPO and reporting tests
python -m unittest discover -s MBPO_workspace/tests -p 'test_*.py' -v

# Read reference MDP and SAC hyperparameters
python -m SAC_workspace.workflows.01_setup_config

# Run exactly 117 transitions against the real solar simulator
python -m SAC_workspace.workflows.02_environment_replay

# Verify MBPO uses the same compact reference environment
python -m MBPO_workspace.workflows.01_setup_config
```

These setup steps save reference snapshots to:

- `SAC_workspace/runs/reference_config.json`
- `MBPO_workspace/runs/reference_config.json`

The shared task uses a **6D compact observation**, **2D normalized incremental actions**, **117 steps per episode**, native **five-minute NSRDB weather**, and **UTC** timestamps (`12:00`–`21:45`). Both algorithms use the same original `PVTrackingEnv` and the same SAC optimizer implementation.

## 3. SAC training (model-free)

### 3.1 Short smoke test

Use a new run directory to avoid overwriting an existing experiment:

```bash
python -m SAC_workspace.workflows.03_train \
  --run-dir SAC_workspace/runs/sac_smoke_new \
  --seed 0 \
  --epochs 2 \
  --device cpu
```

Two episodes test environment interaction, logging and checkpoint creation; they do **not** test gradient learning because the reference warm-up is 8,000 real steps.

### 3.2 Full SAC training

```bash
python -m SAC_workspace.workflows.03_train \
  --run-dir SAC_workspace/runs/sac_seed0 \
  --seed 0 \
  --device cpu
```

The default final target is **2,000 episodes**, taken from the repository's compact reference configuration. For a shorter controlled run:

```bash
python -m SAC_workspace.workflows.03_train \
  --run-dir SAC_workspace/runs/sac_seed1 \
  --seed 1 \
  --epochs 300 \
  --device cpu
```

Do not restart a completed or existing directory without `--resume`. Choose a different `--run-dir` for each independent seed.

### 3.3 Resume SAC training

```bash
python -m SAC_workspace.workflows.03_train \
  --run-dir SAC_workspace/runs/sac_seed0 \
  --seed 0 \
  --resume SAC_workspace/runs/sac_seed0/latest.pt \
  --epochs 2000 \
  --device cpu
```

`--epochs` specifies the **absolute final episode**, not the number of additional episodes. Resume requires the same run directory, seed, configuration, simulator metadata and matching episode history. Use only checkpoints you created and trust.

## 4. MBPO training (model-based)

### 4.1 Short smoke test

```bash
python -m MBPO_workspace.workflows.02_train \
  --run-dir MBPO_workspace/runs/mbpo_smoke_new \
  --seed 0 \
  --epochs 2 \
  --device cpu
```

Two episodes test real data collection, run artifacts and checkpointing. The dynamics ensemble starts fitting only **after the 8,000-step exploration phase**.

### 4.2 Optional model-training integration test

This checks the model-fitting and real/model replay pipeline using a **separate test run**; settings are intentionally lighter than the research configuration:

```bash
python -m MBPO_workspace.workflows.02_train \
  --run-dir MBPO_workspace/runs/mbpo_model_smoke_new \
  --seed 0 \
  --epochs 70 \
  --model-epochs 1 \
  --model-max-train-samples 1000 \
  --device cpu
```

The model quality gate can reject synthetic samples when prediction quality is insufficient. Such rejection is logged and training continues with real samples; it does not automatically indicate a crash. **Do not compare this reduced-fidelity test run with production policies.**

### 4.3 Full MBPO training

```bash
python -m MBPO_workspace.workflows.02_train \
  --run-dir MBPO_workspace/runs/mbpo_seed0 \
  --seed 0 \
  --device cpu
```

The default final target is **2,000 episodes**. The architecture reuses the exact SAC actor/critic, but additionally fits a probabilistic ensemble and generates one-step synthetic transitions. The default model parameters come from the reference config and MBPO's `ModelConfig`.

For a separate seed or shorter target:

```bash
python -m MBPO_workspace.workflows.02_train \
  --run-dir MBPO_workspace/runs/mbpo_seed1 \
  --seed 1 \
  --epochs 300 \
  --device cpu
```

### 4.4 Resume MBPO training

```bash
python -m MBPO_workspace.workflows.02_train \
  --run-dir MBPO_workspace/runs/mbpo_seed0 \
  --seed 0 \
  --resume MBPO_workspace/runs/mbpo_seed0/latest.pt \
  --epochs 2000 \
  --device cpu
```

Resume checks the episode log, replay buffers, random-number-generator states, physical MDP and saved configuration. If a run originally used `--model-epochs` or `--model-max-train-samples` overrides, **repeat the identical overrides when resuming**. Never resume with a different seed or model configuration.

**Checkpoint warning:** By default, checkpoint files are saved every 50 episodes and when a requested run finishes normally. If a process is interrupted between checkpoints, `episodes.csv` can be ahead of `latest.pt`; the trainer deliberately rejects an inconsistent resume. Preserve a copy of both artifacts and reconcile the run history before resuming; do not simply delete or truncate results without inspection.

## 5. Evaluate SAC against physical baselines

Use a saved SAC `latest.pt`. Evaluation is deterministic and uses the actual NSRDB/pvlib environment (no learned dynamics).

### 5.1 Default five-scenario evaluation

```bash
python -m SAC_workspace.workflows.04_evaluate \
  --checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
  --training-csv SAC_workspace/runs/sac_seed0/episodes.csv \
  --outdir SAC_workspace/evaluation/sac_seed0 \
  --device cpu
```

### 5.2 Select exact NSRDB scenario IDs

```bash
python -m SAC_workspace.workflows.04_evaluate \
  --checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
  --scenarios 2018-02-15 2019-05-15 2024-06-21 \
  --seeds 42 \
  --outdir SAC_workspace/evaluation/selected_days
```

### 5.3 Evaluate the complete NSRDB manifest

```bash
python -m SAC_workspace.workflows.04_evaluate \
  --checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
  --training-csv SAC_workspace/runs/sac_seed0/episodes.csv \
  --all-scenarios \
  --outdir SAC_workspace/evaluation/all_scenarios
```

`--all-scenarios` and `--scenarios` are mutually exclusive. The reference evaluation uses a fixed initial pose, so the default single seed (`42`) is appropriate; supplying multiple seeds is rejected when it would duplicate identical evaluation conditions.

**SAC evaluation outputs:**

```text
SAC_workspace/evaluation/sac_seed0/
├── summary.json
├── episodes.csv
├── trajectories.csv
├── gross_energy_comparison.png
├── net_energy_comparison.png
├── paired_net_energy_differences.png
├── matched_trajectory.png
├── seasonal_net_energy.png
├── training_learning_curve.png      # when --training-csv is supplied
└── training_diagnostics.png         # when --training-csv is supplied
```

The energy plots report **net** yield for SAC, fixed mounting and slew-limited dual-axis tracking; unconstrained instantaneous solar pointing is shown in **gross** energy only. Some optional figures depend on the available scenarios and supplied training log.

## 6. Unified MBPO vs SAC vs physical baselines

**Requires both a trained MBPO checkpoint and a trained SAC checkpoint.** This is the primary scientific comparison workflow.

### 6.1 Default paired comparison (five reference scenarios)

```bash
python -m MBPO_workspace.workflows.03_evaluate \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_seed0/latest.pt \
  --sac-checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_seed0/episodes.csv \
  --sac-training-csv SAC_workspace/runs/sac_seed0/episodes.csv \
  --seed 42 \
  --outdir MBPO_workspace/evaluation/paired_seed0 \
  --device cpu
```

This compares:

1. `mbpo` — remastered model-based SAC policy.
2. `sac` — model-free SAC policy.
3. `fixed_no_motion` — fixed panel orientation.
4. `dual_axis_slew_limited` — physically feasible dual-axis tracker.
5. `dual_axis_instant_ideal` — instantaneous sun pointing, **gross only**.
6. `poa_instant_ideal` — instantaneous POA-optimized orientation, **gross only**.

All feasible controllers use the same scenario, initial panel orientation, post-step UTC timestamps, actuator limits, physical simulator, and reward accounting.

### 6.2 Specific scenarios

```bash
python -m MBPO_workspace.workflows.03_evaluate \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_seed0/latest.pt \
  --sac-checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
  --scenarios 2018-02-15 2019-05-15 2024-06-21 \
  --outdir MBPO_workspace/evaluation/selected_days
```

### 6.3 All scenarios with optimized POA reference

This can be computationally expensive:

```bash
python -m MBPO_workspace.workflows.03_evaluate \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_seed0/latest.pt \
  --sac-checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
  --all-scenarios \
  --outdir MBPO_workspace/evaluation/full_with_poa
```

### 6.4 Faster full-manifest evaluation

`--skip-poa-opt` omits **only** the additional numerical POA optimization. It retains the physically feasible dual-axis controller and instantaneous sun-pointing gross reference.

```bash
python -m MBPO_workspace.workflows.03_evaluate \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_seed0/latest.pt \
  --sac-checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_seed0/episodes.csv \
  --sac-training-csv SAC_workspace/runs/sac_seed0/episodes.csv \
  --all-scenarios \
  --skip-poa-opt \
  --outdir MBPO_workspace/evaluation/full_manifest
```

`--all-scenarios` and `--scenarios` cannot be combined. `--seed` is a single evaluation seed (default `42`), not an independent training seed.

## 7. Advanced reporting and regeneration

The unified MBPO evaluation **automatically generates** the standard comparison figures, a PDF, and the advanced 2D/3D reports. You do not have to run another command for a normal evaluation.

To **regenerate advanced plots from existing saved CSVs**, without reopening the simulator or loading either checkpoint:

```bash
python -m MBPO_workspace.workflows.04_advanced_report \
  --eval-dir MBPO_workspace/evaluation/paired_seed0
```

Save regenerated figures to a different directory:

```bash
python -m MBPO_workspace.workflows.04_advanced_report \
  --eval-dir MBPO_workspace/evaluation/paired_seed0 \
  --rich-csv MBPO_workspace/evaluation/paired_seed0/advanced/trajectories_rich.csv \
  --outdir MBPO_workspace/evaluation/paired_seed0/advanced_reprint
```

The source `trajectories_rich.csv` must first be generated by a successful modern MBPO `03_evaluate` run. This offline workflow does **not** generate evaluation data or compare new policy checkpoints.

### 7.1 Unified comparison output layout

```text
MBPO_workspace/evaluation/paired_seed0/
├── summary.json                        # paired statistics, confidence intervals
├── episodes.csv                         # one row per method/day
├── trajectories.csv                     # five-minute tracking trajectories
├── comparison_report.pdf
├── 01_gross_energy.png
├── 02_net_energy.png
├── 03_paired_net_differences.png
├── 04_physical_trajectories.png
├── 05_orientation_3d.png
├── 06_scenario_energy_heatmap.png
├── 07_seasonal_net_energy.png
├── 08_training_learning_curves.png      # when training CSVs are supplied
├── 09_dynamics_training_diagnostics.png # when MBPO training CSV is supplied
├── sac_reference/                       # reused SAC reference plots
└── advanced/
    ├── trajectories_rich.csv            # actions, pose, solar/weather metadata
    ├── advanced_summary.json
    ├── figure_manifest.txt
    ├── advanced_paper_figures.pdf
    ├── 01_energy_decomposition.png
    ├── 02_daily_gain_distribution.png
    ├── 03_orientation_occupancy.png
    ├── 04_hourly_power_gap.png
    ├── 05_ratio_ladder.png
    ├── 06_movement_energy_scatter.png
    ├── 07_tracking_action_diagnostics.png
    ├── 08_actions_3d.png
    ├── 09_orientation_vs_sun_3d.png
    ├── 10_seasonal_energy.png
    └── 11_weather_conditions.png         # if weather labels are present
```

### 7.2 View reports on WSL

```bash
ls -lh MBPO_workspace/evaluation/paired_seed0/
ls -lh MBPO_workspace/evaluation/paired_seed0/advanced/
explorer.exe "$(wslpath -w MBPO_workspace/evaluation/paired_seed0)"
```

## 8. Training files and diagnostics

For a completed SAC run:

```text
SAC_workspace/runs/sac_seed0/
├── config.json
├── resolved_mdp.json
├── episodes.csv
└── latest.pt
```

For a completed MBPO run:

```text
MBPO_workspace/runs/mbpo_seed0/
├── config.json
├── resolved_mdp.json
├── episodes.csv
└── latest.pt
```

Inspect training progress:

```bash
# The most recent logged episodes
 tail -n 10 SAC_workspace/runs/sac_seed0/episodes.csv
 tail -n 10 MBPO_workspace/runs/mbpo_seed0/episodes.csv

# Check saved model files and sizes
ls -lh SAC_workspace/runs/sac_seed0/latest.pt
ls -lh MBPO_workspace/runs/mbpo_seed0/latest.pt
```

SAC `episodes.csv` records `epoch`, `env_steps`, `updates`, `scenario_id`, `return_kwh`, `gross_energy_kwh`, `movement_cost_kwh`, `alpha`, `critic_loss`, `actor_loss`, `alpha_loss`, `entropy`, `q_mean`, and `target_mean`.

MBPO additionally records ensemble validation error, model fitting time, real/model replay sizes, held-out skill versus persistence, model-generated/rejected transition counts, and the real/model minibatch composition. When diagnosing MBPO, look at **model skill**, **synthetic_generated**, **synthetic_rejected**, and **mean_model_batch** together; model fitting does not imply that generated transitions were accepted.

**Checkpoint names:** The current workflows write `latest.pt` every 50 completed episodes and again at normal completion. They do not provide a command-line flag to change the checkpoint interval.

## 9. Command-line options (exact implemented interfaces)

### `SAC_workspace.workflows.01_setup_config`

No command-line options. Writes the SAC reference configuration snapshot.

### `SAC_workspace.workflows.02_environment_replay`

No command-line options. Runs the physical 117-step zero-action simulator check.

### `SAC_workspace.workflows.03_train`

| Flag | Default | Meaning |
|---|---|---|
| `--run-dir PATH` | `SAC_workspace/runs/sac_seed0` | Run output directory; must be new unless resuming |
| `--seed INT` | `0` | Independent SAC training random seed |
| `--epochs INT` | Reference `2,000` | **Absolute final episode** |
| `--device DEVICE` | `cpu` | PyTorch device, e.g. `cpu` or `cuda` if configured |
| `--resume PATH` | None | Resume trusted SAC checkpoint in matching run directory |
| `-h`, `--help` | — | Display CLI help |

### `SAC_workspace.workflows.04_evaluate`

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint PATH` | **Required** | Trained SAC `.pt` |
| `--outdir PATH` | `SAC_workspace/evaluation` | Evaluation output directory |
| `--seeds INT [INT ...]` | `42` | Evaluation reset seeds; duplicates may be rejected for fixed initial pose |
| `--scenarios ID [ID ...]` | Five reference scenarios | Explicit NSRDB scenario IDs |
| `--all-scenarios` | Off | Evaluate every manifest scenario; incompatible with `--scenarios` |
| `--device DEVICE` | `cpu` | PyTorch device |
| `--training-csv PATH` | None | Training CSV for learning/diagnostic plots |
| `-h`, `--help` | — | Display CLI help |

### `MBPO_workspace.workflows.01_setup_config`

No command-line options. Writes the MBPO reference configuration snapshot.

### `MBPO_workspace.workflows.02_train`

| Flag | Default | Meaning |
|---|---|---|
| `--run-dir PATH` | `MBPO_workspace/runs/mbpo_seed0` | Run output directory; must be new unless resuming |
| `--seed INT` | `0` | Independent MBPO training random seed |
| `--epochs INT` | Reference `2,000` | **Absolute final episode** |
| `--device DEVICE` | `cpu` | PyTorch device |
| `--resume PATH` | None | Resume trusted MBPO checkpoint in matching run directory |
| `--model-epochs INT` | `5` | Maximum ensemble training epochs per fit |
| `--model-max-train-samples INT` | `6000` | Maximum real replay transitions used for each model fit |
| `-h`, `--help` | — | Display CLI help |

The two model-related options override the saved model configuration and therefore must remain unchanged during checkpoint resume. They do **not** change the physical simulator or the underlying SAC optimizer.

### `MBPO_workspace.workflows.03_evaluate`

| Flag | Default | Meaning |
|---|---|---|
| `--mbpo-checkpoint PATH` | **Required** | Trained MBPO `.pt` |
| `--sac-checkpoint PATH` | **Required** | Trained SAC `.pt` |
| `--outdir PATH` | `MBPO_workspace/evaluation/paired` | Report destination |
| `--scenarios ID [ID ...]` | Five reference scenarios | Explicit scenario IDs |
| `--all-scenarios` | Off | Full NSRDB manifest; incompatible with `--scenarios` |
| `--seed INT` | `42` | Evaluation reset seed |
| `--skip-poa-opt` | Off | Omit expensive extra optimized POA gross-only reference |
| `--device DEVICE` | `cpu` | PyTorch device |
| `--mbpo-training-csv PATH` | None | MBPO learning and dynamics plot inputs |
| `--sac-training-csv PATH` | None | SAC learning and diagnostics plot inputs |
| `-h`, `--help` | — | Display CLI help |

### `MBPO_workspace.workflows.04_advanced_report`

| Flag | Default | Meaning |
|---|---|---|
| `--eval-dir PATH` | **Required** | Existing modern MBPO evaluation directory |
| `--rich-csv PATH` | `<eval-dir>/advanced/trajectories_rich.csv` | Override source rich CSV |
| `--outdir PATH` | `<eval-dir>/advanced` | Destination for regenerated figures |
| `-h`, `--help` | — | Display CLI help |

To see the actual help text:

```bash
python -m SAC_workspace.workflows.03_train --help
python -m SAC_workspace.workflows.04_evaluate --help
python -m MBPO_workspace.workflows.02_train --help
python -m MBPO_workspace.workflows.03_evaluate --help
python -m MBPO_workspace.workflows.04_advanced_report --help
```

## 10. Experimental protocol essentials

- Train SAC and MBPO on the **same source NSRDB distribution** and the **same simulator/state/action/reward contract**; their learning algorithms differ, not their physical task.
- Report **net kWh** for feasible controllers (`mbpo`, `sac`, `fixed_no_motion`, `dual_axis_slew_limited`). The current reference movement penalty is **zero**, so simulated net energy normally equals gross energy; angular motion remains a separate diagnostic.
- Report **gross kWh only** for unconstrained `dual_axis_instant_ideal` and `poa_instant_ideal`; neither is an actuator-feasible net-energy benchmark.
- Paired evaluation matches each policy by scenario, initial pose, and UTC timestamps. `summary.json` contains paired net differences and bootstrap confidence intervals.
- The five default reference dates and even `--all-scenarios` may overlap the training distribution. **Neither option creates a held-out test split.** A publication claim about unseen days needs a separately enforced scenario-level train/test partition for both methods.
- Use **independently trained seed runs** (`sac_seed0`, `sac_seed1`, `mbpo_seed0`, `mbpo_seed1`, etc.) for learning-variance analysis. The evaluation `--seed` is different from the training random seed.
- The uploaded legacy `result.sh` is for Ray/TensorFlow trial directories and can erase an existing evaluation output directory. **Do not use `./result.sh` for modern SAC/MBPO workspaces.** Use `workflows.04_evaluate`, `workflows.03_evaluate`, and `workflows.04_advanced_report` above.
- Never load `.pt` checkpoints from untrusted sources; the workflows deserialize full training state.

## 11. Shortest end-to-end command sequence

For an already installed project with a working NSRDB manifest:

```bash
cd ~/projects/PVLIB_Paper/mbpo_custom
conda activate simple-worm

# Validate
python -m unittest discover -s SAC_workspace/tests -p 'test_*.py' -v
python -m unittest discover -s MBPO_workspace/tests -p 'test_*.py' -v
python -m SAC_workspace.workflows.01_setup_config
python -m SAC_workspace.workflows.02_environment_replay
python -m MBPO_workspace.workflows.01_setup_config

# Train (skip a command if that run already exists and is complete)
python -m SAC_workspace.workflows.03_train --run-dir SAC_workspace/runs/sac_seed0 --seed 0
python -m MBPO_workspace.workflows.02_train --run-dir MBPO_workspace/runs/mbpo_seed0 --seed 0

# Paired comparison and automatic 2D/3D/PDF reports
python -m MBPO_workspace.workflows.03_evaluate \
  --mbpo-checkpoint MBPO_workspace/runs/mbpo_seed0/latest.pt \
  --sac-checkpoint SAC_workspace/runs/sac_seed0/latest.pt \
  --mbpo-training-csv MBPO_workspace/runs/mbpo_seed0/episodes.csv \
  --sac-training-csv SAC_workspace/runs/sac_seed0/episodes.csv \
  --outdir MBPO_workspace/evaluation/paired_seed0

# Later: regenerate only advanced plots from existing CSV
python -m MBPO_workspace.workflows.04_advanced_report \
  --eval-dir MBPO_workspace/evaluation/paired_seed0
```

