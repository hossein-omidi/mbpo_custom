"""Scientific contract for paired, same-weather PV tracking references.

No training dependencies.  The isotropic upper-bound geometry is analytic;
PVWatts monotonicity is checked separately before a gross DC bound is claimed.
"""
from __future__ import annotations

import math
import numpy as np

_ALBEDO = 0.25  # pvlib.get_total_irradiance default, used by pvlib_physics.py
_WEATHER = ('solar_zenith_deg', 'solar_azimuth_deg',
            'dni_wm2', 'ghi_wm2', 'dhi_wm2')
_PVW = ('ambient_temperature_c', 'wind_speed_m_s',
        'wind_reference_height_m', 'gamma_pdc', 'sapm_mount',
        'wind_powerlaw_exponent')


def isotropic_max_poa_tilt(solar_zenith_deg, dni_wm2, ghi_wm2, dhi_wm2):
    """Globally maximize monofacial isotropic POA over tilt in [0, 90] degrees.

    The surface azimuth is the sun azimuth; for nonnegative DNI it maximizes
    the beam term at each tilt. The isotropic diffuse terms do not depend on
    azimuth. Account for the clipped beam and both diffuse components.
    """
    z, dn, gh, dh = map(float, (solar_zenith_deg, dni_wm2, ghi_wm2, dhi_wm2))
    if not all(math.isfinite(x) for x in (z, dn, gh, dh)):
        raise ValueError('Nonfinite solar/irradiance inputs')
    if not 0. <= z <= 180. or min(dn, gh, dh) < 0.:
        raise ValueError('Invalid zenith or negative irradiance')
    zr = math.radians(z)
    ground = _ALBEDO * gh
    A = dn * math.cos(zr) + (dh - ground) / 2.
    B = dn * math.sin(zr)

    def analytic_poa(tilt):
        b = math.radians(tilt)
        beam = dn * max(0., math.cos(zr - b))
        return beam + dh * (1. + math.cos(b)) / 2. + ground * (1. - math.cos(b)) / 2.

    candidates = [0., 90.]
    active_peak = math.degrees(math.atan2(B, A))
    if 0. <= active_peak <= 90.:
        candidates.append(active_peak)
    # Boundary at which the direct component first becomes nonzero.
    if 90. < z < 180.:
        candidates.append(z - 90.)
    best = max(candidates, key=analytic_poa)
    return float(best)


def certify_pvwatts_monotonicity(power_result, temp_air_c, gamma_pdc):
    """Fail closed if maximizing POA cannot be certified to maximize PVWatts DC.

    SAPM's steady-state T_cell is linear in POA for fixed air temperature and
    wind. PVWatts V5 is therefore quadratic in POA with a non-positive second
    derivative when gamma<=0. Its derivative is smallest at maximal POA.
    """
    g = float(power_result.poa_global_wm2)
    gamma = float(gamma_pdc)
    air = float(temp_air_c)
    cell = float(power_result.cell_temperature_c)
    if not all(math.isfinite(x) for x in (g, gamma, air, cell)):
        raise ValueError('Cannot certify non-finite PVWatts operating point')
    if g < 0 or gamma > 0:
        raise ValueError('Unsupported PVWatts monotonicity regime')
    heat_per_poa = (cell - air) / g if g > 1e-12 else 0.
    if heat_per_poa < -1e-10:
        raise ValueError('Cell temperature model violates expected heating law')
    minimum_gradient_factor = 1. + gamma * (air - 25.) + 2. * gamma * heat_per_poa * g
    if minimum_gradient_factor < -1e-9:
        raise ValueError('PVWatts DC is not monotone in POA for this interval: '
                         'sun-azimuth POA maximum is not a certified DC oracle')
    return minimum_gradient_factor


def validate_paired_reference_group(group, *, require_oracle=False):
    """Validate source weather, pose, fixed baseline and gross-bound semantics.

    `group` holds already-generated trajectory dicts keyed by method. This
    cannot change any model outputs and never forces a passing comparison.
    """
    if 'fixed_no_motion' not in group:
        raise ValueError('Missing fixed baseline')
    fixed = group['fixed_no_motion']
    n = len(fixed['power'])
    if n != 117:
        raise ValueError('Reference must contain exactly 117 samples')
    if require_oracle and 'poa_instant_ideal' not in group:
        raise ValueError('No certified gross oracle in paired group')
    model = fixed['infos'][0].get('power_model', 'legacy')
    if model not in ('legacy','pvwatts_sapm'):
        raise ValueError('Unsupported reference physics model')
    required = _WEATHER + (_PVW if model == 'pvwatts_sapm' else ())
    initial = np.asarray(fixed['initial_pose'], dtype=float)
    if initial.shape != (2,) or not np.isfinite(initial).all():
        raise ValueError('Invalid fixed-mount orientation')
    for t in range(n):
        if not np.allclose([fixed['tilt'][t],fixed['azimuth'][t]],initial,rtol=0,atol=1e-7):
            raise ValueError('Fixed-mount orientation changed during episode')
        if not np.allclose([fixed['action_tilt'][t],fixed['action_azimuth'][t],
                            fixed['delta_tilt_deg'][t],fixed['delta_azimuth_deg'][t]],
                           0.,rtol=0,atol=1e-7):
            raise ValueError('Fixed-mount action or movement was nonzero')
        if abs(float(fixed['cost'][t]))>1e-10:
            raise ValueError('Fixed-mount movement cost was nonzero')
    for name, path in group.items():
        if len(path['power']) != n or len(path['infos']) != n:
            raise ValueError(f'Incomplete paired episode for {name}')
        if (str(path['scenario_id']),int(path['seed'])) != (str(fixed['scenario_id']),int(fixed['seed'])):
            raise ValueError(f'Scenario/seed mismatch for {name}')
        if not np.allclose(np.asarray(path['initial_pose'],float),initial,rtol=0,atol=1e-7):
            raise ValueError(f'Initial orientation mismatch for {name}')
        for t,(observed, ref) in enumerate(zip(path['infos'], fixed['infos'])):
            if observed.get('timestamp_utc_iso') != ref.get('timestamp_utc_iso'):
                raise ValueError(f'UTC clock mismatch: {name}, step {t}')
            if observed.get('power_model','legacy') != model:
                raise ValueError(f'Mixed power models: {name}, step {t}')
            for field in required:
                a,b=observed.get(field),ref.get(field)
                if a is None or b is None:
                    raise ValueError(f'Missing comparable weather/physics {field}: {name}, step {t}')
                if field == 'sapm_mount':
                    equal = (a == b)
                else:
                    av,bv=float(a),float(b)
                    equal = math.isfinite(av) and math.isfinite(bv) and math.isclose(av,bv,rel_tol=1e-9,abs_tol=1e-6)
                if not equal:
                    raise ValueError(f'Unpaired weather/physics {field}: {name}, step {t}')
            power=float(path['power'][t]);energy=float(path['energy'][t])
            if not math.isfinite(power) or power < 0 or not math.isfinite(energy):
                raise ValueError(f'Invalid power or energy: {name}, step {t}')
            interval=float(ref.get('control_interval_minutes',5.))/60.
            if not math.isclose(energy,power*interval/1000.,rel_tol=1e-7,abs_tol=1e-8):
                raise ValueError(f'Power/energy mismatch: {name}, step {t}')
            diagnostic=observed.get('power_dc_w')
            if diagnostic is None or not math.isclose(float(diagnostic),power,rel_tol=1e-8,abs_tol=1e-5):
                raise ValueError(f'Power metadata mismatch: {name}, step {t}')
            is_ideal = name in ('poa_instant_ideal','dual_axis_instant_ideal')
            if is_ideal:
                if path.get('net_energy_kwh') is not None or path.get('movement_cost_kwh') is not None:
                    raise ValueError('Ideal counterfactual presented as feasible net energy')
            else:
                if not math.isclose(float(path['reward'][t]),energy-float(path['cost'][t]),
                                    rel_tol=1e-7,abs_tol=1e-8):
                    raise ValueError(f'Reward accounting mismatch: {name}, step {t}')
    if 'poa_instant_ideal' in group:
        ideal=group['poa_instant_ideal']
        for name,path in group.items():
            if name=='poa_instant_ideal':
                continue
            for t,(upper,actual) in enumerate(zip(ideal['power'],path['power'])):
                tol=max(1e-5,1e-7*abs(float(actual)))
                if float(upper)+tol < float(actual):
                    raise ValueError(f'Gross DC oracle underestimates {name} at step {t}')
    return {'methods':tuple(sorted(group)), 'steps':n, 'power_model':model,
            'fixed_tilt_deg':float(initial[0]), 'fixed_azimuth_deg':float(initial[1])}
