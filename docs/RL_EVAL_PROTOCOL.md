# RL protocol — bounded annual-scenario PV tracking

The PV tracking task is a **stochastic finite-horizon control problem over an empirical annual weather distribution**, not a supervised calendar hold-out problem and not an intra-day synthetic cloud simulator.

## Formulation (paper)

> Environmental variability is represented through sampling from an empirical annual weather distribution. Each sampled day provides an exogenous weather trajectory. pvlib deterministically computes the PV response to this weather trajectory and the policy-selected panel orientation. Thus, stochasticity enters through episode-level day/weather-scenario sampling rather than synthetic intra-day cloud-transition modeling.

## Objective

\[
J(\pi) = \mathbb{E}_{d \sim p(d)} \left[ \sum_{t=0}^{T-1} r(s_t, a_t \mid \text{weather}(d), t) \right]
\]

- **d** — calendar day sampled from full annual support (`randomize_day=True`, Stage 3: 2020-01-01…2020-12-31).
- **weather(d)** — exogenous episode trajectory from the PVGIS TMY catalog (GHI, DNI, DHI, temperature, wind) plus deterministic solar geometry from pvlib.
- **pvlib** — deterministic physics mapping (irradiance, orientation → POA, power); not a stochastic weather generator.
- **T** — fixed daylight grid (78 steps, UTC episode clock).
- **r** — net energy minus movement penalty (same pvlib path for agent and baselines).

Optional **bounded augmentation** (off in main config): `irradiance_perturbation_std > 0` applies a single episode-level lognormal scale to the catalog profile. This is not dynamic cloud motion or sensor/actuator noise.

## Annual environment support

| Item | Stage 3 (main config) |
|------|------------------------|
| Calendar days | All days in range; no `excluded_dates` |
| Day sampling | `randomize_day=True` each reset |
| Weather source | `historical` (empirical month/day from TMY CSV) |
| Intra-day cloud transitions | **Not modeled** |
| `irradiance_perturbation_std` | **0.0** (paper default) |
| `observation_noise_std` | **0.0** (disabled) |
| Initial panel pose | `randomize_initial_orientation=True` (MDP initial state; not weather ξ) |

## Evaluation (RL policy — not BNN validation)

1. **Frozen policy** — no SAC, replay, or BNN updates during eval scripts.
2. **Real `PVTrackingEnv`** — not the learned dynamics model.
3. **Monte Carlo over days** — `randomize_day=True`, independent `--eval-seed-base + i` rollouts; no held-out calendar dates required.
4. Optional **stress-test** fixed dates (`STAGE3_STRESS_TEST_DATES`) for diagnostic plots only.

## BNN holdout ≠ RL policy evaluation

`holdout_ratio` on replay **transitions** is for dynamics-model validation and elite selection only.

## MBPO (unchanged)

- BNN trains on real replay transitions.
- Model rollouts from real states with `remaining_steps` filtering.
- SAC on real + model batches (`real_ratio`).

## Commands

```bash
./scripts/run_stage3_fullyear.sh verify
./scripts/run_stage3_fullyear.sh train
./scripts/run_stage3_fullyear.sh eval
./scripts/run_stage3_fullyear.sh mc-eval
```

See [STAGE3_FULLYEAR_WORKFLOW.md](STAGE3_FULLYEAR_WORKFLOW.md).
