"""Named training configs for CLI: train.sh <run> <conf>."""

from __future__ import print_function

# name -> (python module, config file basename, one-line description)
CONFIGS = {
    'conf1': (
        'examples.config.pv_tracking.conf1',
        'conf1.py',
        'Full-year MBPO-SAC, historical weather, annual day sampling (main)',
    ),
    'conf2': (
        'examples.config.pv_tracking.conf2',
        'conf2.py',
        'Full-year with conservative MBPO hyperparameters (stable training)',
    ),
    'conf3': (
        'examples.config.pv_tracking.conf3',
        'conf3.py',
        'Summer historical weather (shorter curriculum / debug)',
    ),
    'conf_smoke': (
        'examples.config.pv_tracking.conf_smoke',
        'conf_smoke.py',
        'Short smoke run (model rollouts + remaining_steps audit)',
    ),
    'conf4': (                                   # <-- added
        'examples.config.pv_tracking.conf4',
        'conf4.py',
        'Full-year random environment + alternative hyperparams (parallel to conf3)',
    ),
}


def resolve_conf(name):
    key = (name or '').strip()
    if key not in CONFIGS:
        raise KeyError(
            'Unknown config %r. Available: %s' % (
                name, ', '.join(sorted(CONFIGS.keys()))))
    return CONFIGS[key]


def list_configs():
    for name, (module, path, desc) in sorted(CONFIGS.items()):
        print('  %-10s  %s' % (name, desc))
        print('             module=%s' % module)