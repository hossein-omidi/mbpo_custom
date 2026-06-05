#!/usr/bin/env python3
"""Verify merged MBPO variant matches examples/config/pv_tracking/*.py.

Runs the same merge path as training (base.py + config file) and fails if
critical hyperparameters or environment kwargs differ from the config module.

Usage:
  python scripts/verify_training_config.py
  python scripts/verify_training_config.py --config examples.config.pv_tracking.1
  python scripts/verify_training_config.py --config examples.config.pv_tracking.stage3_multiyear_nsrdb_scenario
"""

from __future__ import print_function

import argparse
import importlib
import sys

_SCRIPT_DIR = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
_REPO_ROOT = __import__('os').path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Keys that must match the config file exactly after merge (training-critical).
ALGO_KEYS = (
    'n_epochs',
    'epoch_length',
    'n_initial_exploration_steps',
    'min_alpha',
    'target_entropy',
    'real_ratio',
    'rollout_schedule',
    'max_model_rollout_length',
    'n_train_repeat',
)
ENV_KEYS = (
    'tz',
    'start_time',
    'periods',
    'freq',
    'randomize_initial_orientation',
    'observation_mode',
    'movement_penalty',
    'weather_source',
    'randomize_day',
    'start_date',
    'end_date',
    'excluded_dates',
)


def load_config_module(config_path):
    mod = importlib.import_module(config_path)
    params = getattr(mod, 'params')
    return mod, dict(params)


def build_merged_variant(config_path, policy='gaussian'):
    from examples.utils import get_parser
    from examples.development import get_variant_spec

    parser = get_parser()
    argv = ['--config', config_path, '--policy', policy]
    args = parser.parse_args(argv)
    return get_variant_spec(args)


def check_pv_config_file(mod, file_params, errors):
    """Config-file checks that do not require Ray."""
    file_env = file_params.get('environment_kwargs', {})
    file_algo = file_params.get('kwargs', {})
    file_eval = file_params.get('evaluation_environment_kwargs', {})

    if not getattr(mod, 'CONFIG_VERSION', None):
        errors.append('CONFIG_VERSION missing on config module (bump when changing hyperparameters)')

    periods = int(file_env.get('periods', 40))
    epoch_length = file_algo.get('epoch_length')
    if epoch_length is not None and int(epoch_length) != periods - 1:
        errors.append(
            'epoch_length=%r must equal periods-1=%d (PV episode steps)' % (
                epoch_length, periods - 1))

    stage = getattr(mod, 'TRAINING_STAGE', None)
    if stage:
        version = getattr(mod, 'CONFIG_VERSION', '') or ''
        if stage not in version:
            errors.append(
                'TRAINING_STAGE=%r should appear in CONFIG_VERSION=%r for traceability' % (
                    stage, version))

    if file_eval.get('fixed_eval_dates'):
        start = file_env.get('start_date')
        end = file_env.get('end_date')
        for d in file_eval['fixed_eval_dates']:
            if start and d < start:
                errors.append('fixed_eval_dates %s before training start_date %s' % (d, start))
            if end and d > end:
                errors.append('fixed_eval_dates %s after training end_date %s' % (d, end))

    if file_env.get('start_date') and file_env.get('end_date'):
        if file_env['start_date'] > file_env['end_date']:
            errors.append('start_date > end_date in environment_kwargs')

    train_ws = file_env.get('weather_source')
    eval_ws = file_eval.get('weather_source', train_ws)
    if train_ws and eval_ws and train_ws != eval_ws:
        errors.append(
            'weather_source train=%r != eval=%r (use same pvlib irradiance path)' % (
                train_ws, eval_ws))

    if train_ws == 'historical' and file_eval.get('fixed_eval_dates'):
        try:
            from examples.config.pv_tracking.verified_dates import (
                assert_dates_in_historical_catalog,
            )
            assert_dates_in_historical_catalog(file_eval['fixed_eval_dates'])
        except ValueError as exc:
            errors.append(str(exc))

    if (file_env.get('start_date') == file_env.get('end_date')
            and file_env.get('randomize_day') is True
            and len(file_eval.get('fixed_eval_dates', [])) <= 1):
        print('  note: single-day catalog with randomize_day=True still samples that day only')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--config',
        default='examples.config.pv_tracking.0',
        help='Python module path (examples.config.pv_tracking.0)',
    )
    p.add_argument(
        '--file-only',
        action='store_true',
        help='Only run config-file PV checks (no Ray merge).',
    )
    args = p.parse_args()

    mod, file_params = load_config_module(args.config)
    file_algo = file_params.get('kwargs', {})
    file_env = file_params.get('environment_kwargs', {})

    errors = []
    check_pv_config_file(mod, file_params, errors)

    variant = None
    if not args.file_only:
        try:
            variant = build_merged_variant(args.config)
        except ImportError as exc:
            print('WARN: Ray merge skipped (%s); file-only checks only.' % exc)
        except Exception as exc:
            errors.append('merged variant build failed: %s' % exc)

    if variant is not None:
        algo = variant['algorithm_params']['kwargs']
        env = variant['environment_params']['training']['kwargs']
        eval_env = variant['environment_params'].get('evaluation', {})
        if isinstance(eval_env, dict) and 'kwargs' in eval_env:
            eval_kwargs = eval_env['kwargs']
        else:
            eval_kwargs = env

        for key in ALGO_KEYS:
            expected = file_algo.get(key)
            if expected is None:
                continue
            actual = algo.get(key)
            if actual != expected:
                errors.append(
                    'algorithm_params.kwargs.%s: merged=%r config_file=%r' % (
                        key, actual, expected))

        for key in ENV_KEYS:
            expected = file_env.get(key)
            if expected is None:
                continue
            actual = env.get(key)
            if actual != expected:
                errors.append(
                    'environment_params.training.kwargs.%s: merged=%r config_file=%r' % (
                        key, actual, expected))

        file_eval = file_params.get('evaluation_environment_kwargs') or {}
        for key, expected in file_eval.items():
            actual = eval_kwargs.get(key)
            if actual != expected:
                errors.append(
                    'environment_params.evaluation.kwargs.%s: merged=%r config_file=%r' % (
                        key, actual, expected))

        config_version = file_params.get('config_version')
        if config_version is not None:
            merged_version = variant.get('config_version')
            if merged_version != config_version:
                errors.append(
                    'config_version: merged=%r config_file=%r' % (
                        merged_version, config_version))

        eval_periods = int(eval_kwargs.get('periods', env.get('periods', 40)))
        if int(algo.get('epoch_length', 39)) != eval_periods - 1:
            errors.append('merged epoch_length != evaluation periods - 1')

    print('Config module: %s' % args.config)
    if hasattr(mod, 'CONFIG_VERSION'):
        print('CONFIG_VERSION: %s' % mod.CONFIG_VERSION)
    if hasattr(mod, 'TRAINING_STAGE'):
        print('TRAINING_STAGE: %s' % mod.TRAINING_STAGE)
    print('Training dates: %s .. %s  randomize_day=%s  weather=%s' % (
        file_env.get('start_date'),
        file_env.get('end_date'),
        file_env.get('randomize_day'),
        file_env.get('weather_source'),
    ))
    print('movement_penalty=%s  periods=%s  epoch_length=%s' % (
        file_env.get('movement_penalty'),
        file_env.get('periods'),
        file_algo.get('epoch_length'),
    ))
    if file_params.get('evaluation_environment_kwargs'):
        print('evaluation_environment_kwargs: %s' % (
            file_params['evaluation_environment_kwargs'],))

    if variant is not None:
        algo = variant['algorithm_params']['kwargs']
        env = variant['environment_params']['training']['kwargs']
        print('Merged n_epochs=%s min_alpha=%s real_ratio=%s exploration=%s' % (
            algo.get('n_epochs'),
            algo.get('min_alpha'),
            algo.get('real_ratio'),
            algo.get('n_initial_exploration_steps'),
        ))
        print('Merged env: observation_mode=%r randomize_initial_orientation=%r' % (
            env.get('observation_mode'),
            env.get('randomize_initial_orientation'),
        ))
        obs_dim = 11 if env.get('observation_mode') == 'physical' else 15
        print('Expected policy obs dim: %d' % obs_dim)

    if errors:
        print('\nFAIL — config / merge checks:')
        for e in errors:
            print('  - %s' % e)
        return 1

    print('\nPASS — %s' % (
        'config file checks' if variant is None else
        'merged training variant matches %s' % args.config))
    return 0


if __name__ == '__main__':
    sys.exit(main())
