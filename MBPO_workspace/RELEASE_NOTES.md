# Release notes — Remastered MBPO v1

- This additive patch introduces `MBPO_workspace` without modifying/removing `mbpo/`, `softlearning/`, `scripts/`, `examples/` or the existing `SAC_workspace/`.
- Python 3.10+; no training dependency on Ray or TensorFlow. Depends on existing SAC_workspace v2 and its tested SAC-v2 actor-critic, environment wrapper, replay implementation and config reader.
- Original `conf_compact` 6D/2D finite-horizon MDP, fixed 117-step UTC NSRDB simulator, gamma=1 and learned reward scale=100 are shared.
- Probabilistic residual ensemble: same referenced model count, elite count, 1-step rollout, 85/15 replay ratio and model fitting cadence. Structured exact actuator + horizon + reward from reference simulator helpers.
- Scenario-disjoint model-validation split; retained optimizers and fixed first-fit normalizers; cautious validation-skill gate; bounded model replay.
- Paired SAC/MBPO/fixed/dual-axis benchmark, gross-only instant sun-pointing and isotropic maximum-POA reference. Figures: 2D, 3D, heatmap, seasons, PDF, training metrics; SAC v2 plots reused.
- Locally checked 11 unit/integration tests with mocked physical environment. **No real-NSRDB MBPO training, convergence, or energy benefit has been claimed.**
- Old TF/Ray scripts (including richer historical figures) remain available but are not automatically imported into the modern Python runtime. The included modern plot suite does not implement every historical figure.
- Port rationale and remaining experimental acceptance steps are documented in `docs/THEORY_AND_AUDIT.md`.
