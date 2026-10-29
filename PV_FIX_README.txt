PV simulator bug fixes + Task-0 V4 regression audit
===================================================
Target repository: hossein-omidi/mbpo_custom, state-space-reduction, eba24fd.

FILES
  apply_pv_fixes.py           Safe source-patcher (check first; apply second)
  pv_task0_strong_audit_v4.py Post-fix audit, checks real and fake reward parity

PURPOSE
  * Fix 3 uses of numpy Generator's missing .randint API in the PVTrackingEnv.
  * Fix geometry actuator cost when a commanded tilt step is mechanically clipped.
    Apply to both PVTrackingEnv and MBPO StructuredFakeEnv.
  * Preserve older legacy movement action penalty, zero-movement-penalty
    benchmark objective, weather pipeline, 117-step window and config files.
  * Test direct calendar-day scenario selection, real pvlib power parity,
    two RNG APIs, exact year pinning, movement and zero-cost boundary cases,
    and real/fake kinematics and rewards with a known true solar-state delta.

HOW TO RUN
  1. Put BOTH Python scripts into your mbpo_custom repository ROOT.
  2. Activate the conda simple-worm environment.
  3. Run:

    cd /home/ecer/projects/PVLIB_Paper/mbpo_custom
    git status --short
    python apply_pv_fixes.py --check
    python apply_pv_fixes.py --apply
    git diff --check
    python -m py_compile mbpo/env/pv_tracking.py MBPO_workspace/src/dynamics.py
    set -o pipefail
    python pv_task0_strong_audit_v4.py --json pv_audit_v4_results.json \
      2>&1 | tee pv_audit_v4_output.txt
    echo "Audit exit code: ${PIPESTATUS[0]}"

  4. Check that final output says:
       ALL EXTENDED TASK-0 V4 RUNTIME CHECKS PASSED
     and that audit exit code is 0.

BACKUPS / ROLLBACK
  --check never writes files.
  --apply creates backups under .pv_fix_backups/<UTC-timestamp>/.
  The original source texts are in that directory; if necessary restore
  those two paths manually using cp, or use git restore after checking
  whether you have existing uncommitted changes. Do not git restore blindly.

NOT TESTED
  We cannot run your actual conda environment and NSRDB files from this
  workspace. Only source-code inspection, syntax checks and small isolated
  functional fixture tests were performed here. The Python audit must run
  on your machine to verify that actual real/fake transitions agree.

SCIENTIFIC SCOPE
  This is not an annual-energy or pvlib physical calibration study, nor
  a full pretrained SAC/MBPO checkpoint-level evaluation. It intentionally
  does not change the simplified constant-efficiency power model or fixed
  daily UTC experimental window.
