"""NSRDB multi-year env tests (native 5-min control grid).

Entry point:
  python -m pytest tests/test_nsrdb_multiyear_env.py -q
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_spec = importlib.util.spec_from_file_location(
    'test_nsrdb_multiyear',
    os.path.join(_HERE, 'test_nsrdb_multiyear.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name in dir(_mod):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_mod, _name)
