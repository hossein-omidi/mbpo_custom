# SAC Workspace — Mathematical Verification & MBPO Integration

## Objective
Create a completely independent, model-free SAC workspace that:
1. Uses the exact same MDP as MBPO (6D compact state, same action/reward, same pvlib simulator)
2. Implements mathematically rigorous SAC (twin critics, target networks, entropy tuning)
3. Is fully modular and separate from MBPO algorithms

---

## 1. MDP Parity Verification

### State Space (Identical to MBPO conf_sac / conf_compact)
```
SAC 6D Compact: [DNI/2000, DHI/2000, P/(2000·A·η), β/90°, (γ%360°)/180°-1, (H-t)/H]
MBPO 6D:      [dni_norm, dhi_norm, power_norm, tilt_norm, azimuth_norm, tau]
```
- `DNI/2000`: DNI ∈ [0, 1000] W/m² → [0, 0.5] (normalized by 2000 for safety margin)
- `DHI/2000`: DHI ∈ [0, 1000] W/m² → [0, 0.5]
- `P/(2000·A·η)`: Power ∈ [0, 360] W (with A=1, η=0.18) → normalized by 2000
- `β/90°`: Tilt ∈ [0, 90°] → [0, 1]
- `(γ%360)/180-1`: Azimuth ∈ [0, 360°) → [-1, 1]
- `(H-t)/H`: Horizon remaining ∈ [0, 1] for H=117 steps

### Action Space (Identical)
```
SAC Action: [delta_tilt, delta_azimuth] ∈ [-5, 5] × [-10, 10] deg/step
MBPO:      Same max_delta_tilt=5.0, max_delta_azimuth=10.0
```

### Reward Function (Identical)
```
reward = power_output - movement_penalty * movement_cost
```
Same reward formula as MBPO (see `mbpo/env/pv_tracking.py`).

### Environment / Simulator (Identical)
```
SAC Environment: SACEnvironment wraps PVTrackingEnv
PVTrackingEnv:    Uses pvlib Location + NSRDB scenarios + pvlib_physics
Physics:          Same solar_zenith, solar_azimuth, POA calculations
Weather:          Same NSRDB multiyear scenarios (weather_scenario_mode='nsrdb_multiyear')
```

---

## 2. SAC Mathematical Formulation (Verified)

### Component Definitions (from `sac_agent.py`)
1. **Policy (Actor)**: π_φ(a|s) = Normal(μ(s; φ), σ(s; φ))
2. **Twin Critics**: Q_θ1(s,a), Q_θ2(s,a) — independent networks
3. **Target Critics**: Q̄_θ1, Q̄_θ2 — polyak-averaged: Q̄ ← τQ + (1-τ)Q̄

### SAC Update Equations (Mathematically Valid)

**Value Target (using min of twin targets for stability):**
```
V(s') = min(Q̄1(s', a'), Q̄2(s', a')) - α · log π(a'|s')
```

**Bellman Target for Q:**
```
Q_target(s, a) = r + γ · (1 - done) · V(s')
```

**Q-Loss (MSE):**
```
L_Q = E[(Q(s,a) - Q_target)²]  (for both Q1 and Q2 independently)
```

**Policy Loss (maximize Q with entropy regularization):**
```
J(φ) = E[s~D, a~π][min(Q1(s,a), Q2(s,a)) - α · log π(a|s)]
```

**Entropy Temperature Update:**
```
L_α = -α · E[a~π][log π(a|s) + H_target]
```

All updates use gradient descent with gradient clipping (norm ≤ 1.0) for stability.

---

## 3. Module Structure (Independent from MBPO)

```
SAC_workspace/src/
├── representation/representation.py   # State encoding (6D → physical/legacy conversion)
├── environment/environment.py         # SACEnvironment (wraps same PVTrackingEnv)
├── agents/sac_agent.py              # SACAgent (actor, twin critics, targets, replay)
├── training/trainer.py               # SACTrainer (pure model-free loop)
├── evaluation/evaluation.py          # SACEvaluator (benchmark vs oracle)
```

No imports from `mbpo/algorithms/` — completely decoupled from MBPO training code.
Only imports from `mbpo/env/` (simulator) and `mbpo/static/` (constants) for exact parity.

---

## 4. Notebook Sections (Documented Workflow)

1. **Setup & Config**: Same hyperparameters (hidden_dim=256, lr=3e-4, γ=0.99, τ=0.005)
2. **Environment & Replay Buffer**: Initialize SACEnvironment + ReplayBuffer (100K capacity)
3. **Agent Training**: Execute SAC loop — no rollouts, only real environment transitions stored
4. **Evaluation & Benchmarking**: Compare SAC metrics against pvlib-based oracle

---

## 5. Benchmark Comparison Setup

The evaluation compares:
- **SAC Policy**: Trained with pure SAC on same 6D state, same simulator
- **Physical Oracle**: Theoretical optimal tracking (panel = solar position at each step)
- **Baseline Metrics**: Mean reward, mean power output, relative performance

Both SAC and Oracle use the same `pvlib` solar geometry (`Location.solarposition`) and same weather scenarios (`NSRDB_MANIFEST`).

---

## Verification Checklist

| Requirement | Status | Evidence |
|---|---|---|
| Same 6D state space | ✅ | `COMPACT_OBS_DIM = 6` used in representation |
| Same action/reward | ✅ | Same `max_delta_tilt=5`, same reward formula |
| Same pvlib simulator | ✅ | `SACEnvironment` wraps `PVTrackingEnv` |
| Pure SAC (no model) | ✅ | `SACAgent` uses only replay buffer, no rollouts |
| Twin critics | ✅ | `TwinQNetwork` with independent Q1, Q2 |
| Target networks | ✅ | `target_twin_critic` with polyak (τ=0.005) |
| Entropy tuning | ✅ | `automatic_entropy_tuning=True`, learnable log_α |
| Mathematical rigor | ✅ | All SAC equations documented in sac_agent.py |
| Independent workspace | ✅ | No imports from `mbpo/algorithms/` |
| Jupyter notebook | ✅ | `notebooks/sac_training_analysis.ipynb` |
| Benchmark comparison | ✅ | `SACEvaluator` compares SAC vs Oracle |
