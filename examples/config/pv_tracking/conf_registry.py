"""Named training configs for CLI: train.sh <run> <conf>."""

from __future__ import print_function

from examples.config.pv_tracking._nsrdb_base import NSRDB_CONF_NAMES

# name -> (python module, config file basename, one-line description)
CONFIGS = {
    'stage3_nsrdb': (
        'examples.config.pv_tracking.stage3_multiyear_nsrdb_scenario',
        'stage3_multiyear_nsrdb_scenario.py',
        'NSRDB main reference (stage3_multiyear_nsrdb_scenario)',
    ),
    'conf1': (
        'examples.config.pv_tracking.conf1',
        'conf1.py',
        'NSRDB simple-strong — compact nets, fixed 30°S pose, real_ratio=0.9',
    ),
    'conf2': (
        'examples.config.pv_tracking.conf2',
        'conf2.py',
        'NSRDB slow — long run, conservative MBPO (parallel run)',
    ),
    'conf3': (
        'examples.config.pv_tracking.conf3',
        'conf3.py',
        'NSRDB strong — deep SAC + wide BNN, high real_ratio (recommended)',
    ),
    'conf4': (
        'examples.config.pv_tracking.conf4',
        'conf4.py',
        'NSRDB light-fast — 64-wide nets, short run, quick epochs',
    ),
    'conf5': (
        'examples.config.pv_tracking.conf5',
        'conf5.py',
        'NSRDB advanced — deep SAC + wide BNN, high real_ratio (A/B vs conf3)',
    ),
}


def is_nsrdb_conf(name):
    return (name or '').strip() in NSRDB_CONF_NAMES


def resolve_conf(name):
    key = (name or '').strip()
    if key not in CONFIGS:
        raise KeyError(
            'Unknown config %r. Available: %s' % (
                name, ', '.join(sorted(CONFIGS.keys()))))
    return CONFIGS[key]


def list_configs():
    for name, (module, path, desc) in sorted(CONFIGS.items()):
        print('  %-12s  %s' % (name, desc))
        print('               module=%s' % module)
