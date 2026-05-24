#!/usr/bin/env python3
"""Verify merged MBPO variant matches examples/config/pv_tracking/*.py.

Runs the same merge path as training (base.py + config file) and fails if
critical hyperparameters or environment kwargs differ from the config module.

Usage:
  python scripts/verify_training_config.py
  python scripts/verify_training_config.py --config examples.config.pv_tracking.1
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
    # examples.development.get_variant_spec loads the config module and
    # runs the same merge path as mbpo run_local/run_example_dry.
    return get_variant_spec(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--config',
        default='examples.config.pv_tracking.0',
        help='Python module path (examples.config.pv_tracking.0)',
    )
    args = p.parse_args()

    mod, file_params = load_config_module(args.config)
    variant = build_merged_variant(args.config)

    algo = variant['algorithm_params']['kwargs']
    env = variant['environment_params']['training']['kwargs']
    file_algo = file_params.get('kwargs', {})
    file_env = file_params.get('environment_kwargs', {})

    errors = []
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

    config_version = file_params.get('config_version')
    if config_version is not None:
        merged_version = variant.get('config_version')
        if merged_version != config_version:
            errors.append(
                'config_version: merged=%r config_file=%r' % (
                    merged_version, config_version))

    print('Config module: %s' % args.config)
    if hasattr(mod, 'CONFIG_VERSION'):
        print('CONFIG_VERSION: %s' % mod.CONFIG_VERSION)
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
        print('\nFAIL — merged variant does not match config file:')
        for e in errors:
            print('  - %s' % e)
        return 1

    print('\nPASS — merged training variant matches %s' % args.config)
    return 0


if __name__ == '__main__':
    sys.exit(main())
