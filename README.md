# Read-only PV orientation/action contract diagnostic

The script does not modify code, checkpoints, trajectories or other experiment data. It is intended for the post-v5.3, rich detailed trajectory exporter (`trajectories_detailed.csv`) and will fail explicitly on missing required measurements.

From your repository root, copy `pv_orientation_diagnostics.py` there (or invoke its absolute file location):

```bash
python pv_orientation_diagnostics.py \
  --eval-dir MBPO_workspace/evaluation/scientific_v52_fresh \
  --day 2024-12-23 \
  --output-json orientation_audit_2024-12-23.json
```

The program audits every recorded SAC, MBPO and fixed-panel transition in the evaluation, not only the selected day. `--day` selects additional reporting detail.

Contract checked, *independently from the graphical line charts*:

- Physical tilt in `[0,90]` degrees; panel azimuth in `[0,360)` degrees.
- Normalized actions within `[-1,1]`; per step requested tilt change = `5*action_tilt` degrees, requested azimuth change = `10*action_azimuth` degrees, unless the CLI limits are explicitly changed.
- Actual next tilt equals `clip(pre_tilt + requested_tilt_change, 0, 90)`.
- Actual next azimuth equals `(pre_azimuth + requested_azimuth_change) % 360`.
- Step-to-step pose continuity and equality of initial pose for paired controllers.
- Power-to-energy identity for five-minute windows: `energy_kwh = power_w/12000`.
- Recorded net step reward, if present, equals gross energy minus movement cost.
- Complete 117-step trajectories and full controller pairing.
- POA instantaneous reference, if present, never generates *less* gross power at the same step than a feasible orientation (necessary condition, not a proof of global optimality).

Diagnostics *not evidence of bugs by themselves*:

- Fraction of commands with absolute action >=0.95 (near the requested per-step slew limit).
- Fraction of physical post-step orientations at tilt=0 or tilt=90.
- Number of outward tilt commands partially/fully clipped at physical limits.
- Number of adjacent strong tilt-command sign reversals.
- Worst total net-energy deficit days compared with fixed mounting.

**Limits:** This is a strict action/orientation/evaluation contract test, not a test of SAC Q-function accuracy, MBPO counterfactual predictive validity, convergence, optimality, out-of-training-scenario generalization, or sampling uncertainty. For these latter claims you must inspect training checkpoints/replay and run independent evaluation experiments. The script uses only the supplied saved evaluation data; synthetic unit tests do not establish that the user's real files pass.

No training hyperparameters are changed. Legacy v5.3 scientific audit should be run separately for weighted means and statistical curves.
