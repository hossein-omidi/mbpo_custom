#!/usr/bin/env python3
"""Read-only preflight for the staged PVWatts backend.

Run from mbpo_custom root: python scripts/pvwatts_stage1_preflight.py
Does not need NSRDB data; does not create or modify any project file.
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
if not all((ROOT / p).exists() for p in ('mbpo/env/pv_tracking.py',
                                          'mbpo/env/pvlib_physics.py',
                                          'SAC_workspace/src/agents/sac_agent.py',
                                          'MBPO_workspace/src/dynamics.py')):
    raise SystemExit('ERROR: run from mbpo_custom repository root')
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-pvlib', default='0.13.1',
                        help='Target package version, default 0.13.1')
    parser.add_argument('--allow-different-pvlib', action='store_true',
                        help='Allow API-compatible other pvlib versions')
    args = parser.parse_args()

    import numpy as np
    import pvlib
    from pvlib import atmosphere, pvsystem, temperature
    from mbpo.env import pvlib_physics
    from mbpo.env.pv_power_backend import compute_pv_power
    from SAC_workspace.src.agents.sac_agent import SACAgent
    from SAC_workspace.src.config.config import load_config
    from SAC_workspace.src.representation.representation import encode
    from MBPO_workspace.src.dynamics import DynamicsEnsemble, StructuredFakeEnv

    if sys.version_info < (3, 10):
        raise RuntimeError(f'Python 3.10+ required; found {sys.version}')
    if pvlib.__version__ != args.expected_pvlib and not args.allow_different_pvlib:
        raise RuntimeError(f'pvlib {pvlib.__version__} != expected {args.expected_pvlib}; '
                           'use --allow-different-pvlib only after validating API parity')
    checks = [
        (pvsystem.pvwatts_dc, ('effective_irradiance','temp_cell','pdc0','gamma_pdc','temp_ref')),
        (temperature.sapm_cell, ('poa_global','temp_air','wind_speed','a','b','deltaT')),
        (atmosphere.windspeed_powerlaw, ('wind_speed_reference','height_reference','height_desired','exponent')),
        (pvlib_physics.compute_poa_global, ('surface_tilt','surface_azimuth','solar_zenith','solar_azimuth','dni','ghi','dhi')),
    ]
    for fn, required in checks:
        missing = set(required) - set(inspect.signature(fn).parameters)
        if missing:
            raise RuntimeError(f'Missing expected API parameters in {fn}: {missing}')
    cfg = load_config(require_data=False)
    if (cfg.train_env.get('observation_mode') != 'compact' or
        cfg.eval_env.get('observation_mode') != 'compact'):
        raise RuntimeError('Expected unchanged 6D compact observation configuration')
    if 'power_model' in cfg.train_env or 'power_model' in cfg.eval_env:
        raise RuntimeError('Stage 1 baseline already has power_model configured: '
                           'review before treating this as a read-only baseline comparison')
    if not callable(StructuredFakeEnv.step) or not callable(DynamicsEnsemble.predict_delta):
        raise RuntimeError('MBPO dynamics interface not available')
    if not callable(SACAgent.act) or not callable(encode):
        raise RuntimeError('SAC action/representation interface not available')

    sample = dict(surface_tilt=30., surface_azimuth=180.,solar_zenith=30.,
                  solar_azimuth=180.,dni=800.,ghi=900.,dhi=120.)
    legacy = compute_pv_power(**sample)
    modern = compute_pv_power(**sample,power_model='pvwatts_sapm',
                              temp_air_c=25.,wind_speed_ms=2.,wind_reference_height_m=2.)
    assert np.isfinite(legacy.power_w) and np.isfinite(modern.power_w)
    assert abs(legacy.poa_global_wm2-modern.poa_global_wm2) < 1e-10
    print('[PASS] Python:',sys.version.split()[0])
    print('[PASS] pvlib:',pvlib.__version__, 'at', Path(pvlib.__file__).resolve())
    print('[PASS] pvlib API: DC, SAPM temperature, wind height conversion, project POA')
    print('[PASS] SAC/MBPO modules import; unmodified 6D compact configuration')
    print('[PASS] Backend smoke test: legacy={:.5f} W, PVWatts={:.5f} W, POA={:.5f} W/m2'.format(
        legacy.power_w,modern.power_w,legacy.poa_global_wm2))
    print('STAGE 1 PREFLIGHT PASS. Simulator/training/evaluation remain on legacy physics.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'STAGE 1 PREFLIGHT FAIL: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1)
