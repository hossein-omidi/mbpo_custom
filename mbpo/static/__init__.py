import sys
import os
import importlib


def _import_static_fns(module_suffix, fns_name='StaticFns'):
    module = importlib.import_module('mbpo.static.' + module_suffix)
    return getattr(module, fns_name)


_STATIC_DIR = os.path.dirname(os.path.abspath(__file__))
_files = [f for f in os.listdir(_STATIC_DIR) if f.endswith('.py') and not f.startswith('__')]

# {pvtracking: StaticFns, ...} — keys match domain.lower() with underscores removed
static_fns = {
    f.replace('.py', '').replace('_', ''): _import_static_fns(f.replace('.py', ''))
    for f in _files
}

sys.modules[__name__] = static_fns
