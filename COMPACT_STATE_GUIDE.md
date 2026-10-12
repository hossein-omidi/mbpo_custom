# PV Tracking with Compact 6D State Space

## Overview

This repository implements a **compact 6-dimensional state space** for PV tracking control, replacing the original 12D/16D observation spaces. The new observation vector contains only the essential information needed for optimal tracking without explicit solar position:

```
o_t = [DNI, DHI, P, β, γ, τ]
```

Where:
- **DNI** - Direct Normal Irradiance (normalized: DNI/2000)
- **DHI** - Diffuse Horizontal Irradiance (normalized: DHI/2000)  
- **P** - Current PV Power (normalized: P/(2000·A·η))
- **β** - Panel Tilt (normalized: β/90°)
- **γ** - Panel Azimuth (normalized: (γ mod 360°)/180° - 1 ∈ [-1, 1])
- **τ** - Remaining Horizon (normalized: (H-t)/H)

## Key Design Decisions

1. **No Solar Position**: The agent does NOT receive solar zenith/azimuth. It learns tracking from irradiance and power feedback directly.
2. **No Relative Azimuth**: The agent receives absolute panel orientation (tilt, azimuth), not relative to sun.
3. **All Features Normalized**: All 6 features are in appropriate ranges for neural network training.
4. **Actions Unchanged**: Still two incremental commands [Δtilt, Δazimuth] ∈ [-1, 1]².

## Configurations

### conf_compact (MBPO)
- **Algorithm**: MBPO with model-based rollouts
- **real_ratio**: 0.85 (15% model rollouts)
- **Observation**: 6D compact
- **Networks**: 64-wide (smaller for faster training with lower-dimensional input)
- **Replay Pool**: 50k capped for 8GB nodes

### conf_sac (SAC-only Baseline)
- **Algorithm**: Pure SAC (real_ratio=1.0, no model rollouts)
- **Observation**: 6D compact (identical to MBPO)
- **Networks**: 64-wide (same architecture for fair comparison)
- **Purpose**: Sample complexity comparison with MBPO on identical task

## Running Experiments

### Preflight Checks
```bash
# Test MBPO config
python scripts/verify_preflight.py --config examples.config.pv_tracking.conf_compact

# Test SAC config  
python scripts/verify_preflight.py --config examples.config.pv_tracking.conf_sac
```

### Dry Run Comparison
```bash
# Quick comparison without training
python scripts/compare_mbpo_vs_sac.py \
  --mbpo-config examples.config.pv_tracking.conf_compact \
  --sac-config examples.config.pv_tracking.conf_sac \
  --dry-run
```

### Full Comparison
```bash
# Run with 3 seeds, 2000 epochs each
python scripts/compare_mbpo_vs_sac.py \
  --mbpo-config examples.config.pv_tracking.conf_compact \
  --sac-config examples.config.pv_tracking.conf_sac \
  --n-seeds 3 \
  --epochs 2000
```

## Fair Comparison Design

| Aspect | MBPO (conf_compact) | SAC (conf_sac) |
|--------|---------------------|----------------|
| Environment | Identical | Identical |
| Observation Space | 6D compact | 6D compact |
| Reward Function | Identical | Identical |
| Network Architecture | 64-wide | 64-wide |
| Episode Horizon | 117 steps | 117 steps |
| Training Data | Same scenarios | Same scenarios |
| Model Rollouts | Yes (15%) | No (real_ratio=1.0) |

The only difference is `real_ratio`: MBPO uses 0.85 (model rollouts), SAC uses 1.0 (pure real data).

## Observation Details

### Compact 6D (used by both configs)
| Index | Feature | Normalization | Range |
|-------|---------|---------------|-------|
| 0 | DNI | DNI / 2000 | [0, 1] |
| 1 | DHI | DHI / 2000 | [0, 1] |
| 2 | Power | P / (2000·A·η) | [0, 1] |
| 3 | Tilt | β / 90° | [0, 1] |
| 4 | Azimuth | (γ mod 360°)/180° - 1 | [-1, 1] |
| 5 | Tau | (H-t) / H | [0, 1] |

### Legacy Modes (preserved for backward compatibility)
- **Physical (12D)**: Solar geometry + irradiance + panel orientation + tau
- **Legacy (16D)**: Full state with time-of-day and day-of-year sin/cos

## Key Files Modified

| File | Changes |
|------|---------|
| `mbpo/env/pv_tracking.py` | New `_build_compact_observation()`, updated bounds, feature labels |
| `mbpo/static/pv_tracking.py` | `COMPACT_OBS_DIM=6`, updated `time_angle_from_obs()` for tau at index 5 |
| `scripts/eval_utils.py` | Updated `decode_pv_observation()` for 6D compact, fixed baselines |
| `scripts/validate_pv_rollouts.py` | Added `validate_static_fns_compact()` |
| `examples/config/pv_tracking/conf_compact.py` | MBPO config with compact 6D |
| `examples/config/pv_tracking/conf_sac.py` | SAC-only config with compact 6D |
| `examples/config/pv_tracking/conf_registry.py` | Registered both configs |

## Verification

All preflight checks pass for both configurations:
- ✅ Training config validation
- ✅ Movement cost fairness
- ✅ UTC uniformity  
- ✅ NSRDB multi-year scenario validation
- ✅ NSRDB timing sync
- ✅ PV rollout validation

## Expected Outcomes

The compact 6D space should:
1. **Train faster** - Lower dimensionality = fewer parameters, faster convergence
2. **Learn robust policies** - Agent learns from raw irradiance/power feedback (more transferable)
3. **Enable fair MBPO vs SAC comparison** - Same observation space, only algorithm differs
4. **Reduce overfitting** - No explicit solar position to memorize

## References

- Original 12D/16D spaces: `mbpo/env/pv_tracking.py` (legacy/physical modes)
- PV physics: `mbpo/env/pvlib_physics.py` (shared by env, eval, baselines)
- NSRDB weather: `mbpo/env/nsrdb_weather.py` (multi-year scenarios)