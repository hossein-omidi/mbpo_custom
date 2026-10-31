# Stage 5.1 reproducibility check — actual user artifacts

The updated read-only audit was run on the supplied Stage 4 deep diagnostic ZIP, the original Stage 5 full evaluation `episodes.csv`, `trajectories_detailed.csv`, `summary.json`, `scenario_selection.json`, and the two training CSVs. This is **not** a new training or physics simulation.

```text
Unit tests in this diagnostic package:
20 passed in approximately 0.1 s
Original bundled Stage 4 tests:
14 passed in approximately 3 s
Total: 34 passed

Full-data offline audit:
AUDITED_WITH_REPRODUCIBILITY_FLAGS
- SAC final-alpha Stage 4 checkpoint/report vs Stage 5 CSV mismatch
- SAC last-100-window statistics vs Stage 4 checkpoint-matched history mismatch
```

## Positive checks

- 336 date/seed groups; all three feasible methods matched.
- 117,936 feasible-control trajectory rows and 39,312 matched cross-controller step groups; 5-min UTC grid and scenario-day identity pass.
- 336 × 117 = 39,312 oracle step rows independently checked under a **gross-only, instantaneous upper-bound** contract. No actuator feasibility claimed.
- Design-weighted estimates recomputed with the original strata file; population weight sum = **1,167** and 336 selected scenarios.
- Weighted means: fixed=0.7454190285, SAC=0.7487482291, MBPO=0.8575851613 kWh; gross oracle=0.9124866745 kWh.
- Replay coverage from Stage 4 snapshot: 38,976 post-first-step actions per learned actor, consistent with 336×116; not an independent re-execution from this offline audit.
- Exact scenario-ID train/eval intersections: 0 for both provided training histories. The selected sampling frame excludes recorded SAC training dates, hence the test frame is conditional on that run.
- Approximate last-428-episode coverage: SAC 393 unique days, MBPO 387; both span all seasons. Replay eviction exists by FIFO design but is **not proved to be the cause** of the SAC policy's steep tilt.

## Negative check

SAC checkpoint alpha as reported by Stage 4 is **0.0155801819637**, whereas the Stage 5 training CSV ends at **0.0224180519581**. Three other last-100 training indicators also disagree. The correct SAC training CSV must be identified before using its diagnostic learning curves. MBPO's history matches the Stage 4 reported snapshot.

## What is still not revalidated here

- Loading the user's original checkpoint files to recompute a shadow optimizer step or its action replay.
- First actions on every evaluation day (the Stage 4 checkpoint-action check covered subsequent 116 actions/day).
- Recomputing PVWatts power from independent raw NSRDB data in a live simulator.
- Analytically proving the oracle globally optimal from exported aggregate power values.
- Critic conditional-value calibration against long-horizon entropy-regularized ground truth.
- Multiple independent training seeds, algorithm-level error bars, or generalization to unobserved weather years.

The audited results therefore support **checkpoint-specific, descriptive energy performance**, not a confirmed SAC implementation bug or a proof of MBPO's universal superiority.
