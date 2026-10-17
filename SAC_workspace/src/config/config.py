"""Read the existing PV configuration without importing Ray or TensorFlow.

Only literal dictionaries from the committed configuration modules are loaded.
Failures are explicit; SAC must never silently run a different weather/MDP.
"""
from __future__ import annotations
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
_BASE = ROOT / 'examples/config/pv_tracking/_nsrdb_base.py'
_COMPACT = ROOT / 'examples/config/pv_tracking/conf_compact.py'
_SKELETON = ROOT / 'examples/config/pv_tracking/0.py'


def _parse(path: Path) -> ast.Module:
    if not path.is_file():
        raise FileNotFoundError(f'Required reference configuration missing: {path}')
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _literal(node, names=None):
    """Restricted interpreter for configuration literals and module constants."""
    names = names or {}
    if isinstance(node, ast.Constant): return node.value
    if isinstance(node, ast.Name):
        if node.id in names: return names[node.id]
        raise ValueError(f'Unknown reference config constant: {node.id}')
    if isinstance(node, ast.Dict):
        return {_literal(k, names): _literal(v, names) for k,v in zip(node.keys,node.values)}
    if isinstance(node, (ast.Tuple,ast.List,ast.Set)):
        seq = [_literal(v, names) for v in node.elts]
        return tuple(seq) if isinstance(node,ast.Tuple) else set(seq) if isinstance(node,ast.Set) else seq
    if isinstance(node, ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
        val = _literal(node.operand, names)
        return -val if isinstance(node.op,ast.USub) else +val
    if isinstance(node, ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
        x,y = _literal(node.left, names), _literal(node.right, names)
        if isinstance(node.op,ast.Add):return x+y
        if isinstance(node.op,ast.Sub):return x-y
        if isinstance(node.op,ast.Mult):return x*y
        return x/y
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('len','list','tuple') and len(node.args)==1 and not node.keywords:
        val = _literal(node.args[0], names)
        return {'len': len, 'list': list, 'tuple': tuple}[node.func.id](val)
    raise ValueError(f'Non-literal configuration expression: {ast.dump(node)[:140]}')


def _module_constants(path):
    known = {}
    for node in _parse(path).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            try: known[node.targets[0].id] = _literal(node.value, known)
            except (ValueError, TypeError, KeyError): pass
    return known


def _named_function_dict(path: Path, name: str, variable: str) -> dict:
    names = _module_constants(path)
    for node in _parse(path).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            for statement in node.body:
                if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in statement.targets):
                    if isinstance(statement.value, ast.Dict):
                        return _literal(statement.value, names)
    raise ValueError(f'Expected literal {name}.{variable} dict not found in {path}')


def _updated_dict(path: Path, name: str, variable: str = 'kw') -> dict:
    names = _module_constants(path)
    for node in _parse(path).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            for statement in node.body:
                if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                    call = statement.value
                    if isinstance(call.func, ast.Attribute) and call.func.attr == 'update' and isinstance(call.func.value, ast.Name) and call.func.value.id == variable and len(call.args) == 1 and isinstance(call.args[0], ast.Dict):
                        return _literal(call.args[0], names)
    raise ValueError(f'No literal {variable}.update dict found in {path}')


def _reference_eval_defaults(path: Path) -> dict:
    """Resolve the explicit reference *evaluation* builder, not training overrides.

    The legacy module imports TensorFlow through its configuration dependencies;
    this restricted AST reader inspects only the declared builder expression.
    Unknown AST constructs fail rather than silently choosing defaults.
    """
    names = _module_constants(path)
    func = next((n for n in _parse(path).body if isinstance(n, ast.FunctionDef)
                 and n.name == 'nsrdb_evaluation_environment_kwargs'), None)
    if func is None:
        raise ValueError('Reference NSRDB evaluation builder is missing')
    for stmt in func.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == 'kw'):
            call = stmt.value
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == 'nsrdb_environment_kwargs' and not call.args):
                raise ValueError('Reference NSRDB evaluation builder changed unexpectedly')
            overrides = {kw.arg: _literal(kw.value, names) for kw in call.keywords}
            base = _named_function_dict(path, 'nsrdb_environment_kwargs', 'kw')
            base.update(overrides)
            base.pop('fixed_eval_dates', None)
            return base
    raise ValueError('Cannot derive NSRDB evaluation defaults')


def _constant(path: Path, name: str) -> Any:
    names = _module_constants(path)
    if name not in names: raise ValueError(f'Cannot resolve {name} in {path}')
    return names[name]


def _call_kwargs(path: Path, call_name: str, key: str) -> dict:
    for node in _parse(path).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'params' for t in node.targets):
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == call_name:
                for kw in call.keywords:
                    if kw.arg == key:
                        return _literal(kw.value, _module_constants(path))
    return {}


def _literal_mapping_fields(path: Path, name: str, keys: set[str]) -> dict:
    """Extract only literal entries from a legacy dictionary containing Ray expressions."""
    for node in _parse(path).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            assert isinstance(node.value, ast.Dict)
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == 'kwargs' and isinstance(v, ast.Dict):
                    result = {}
                    for key_node, val_node in zip(v.keys, v.values):
                        if isinstance(key_node, ast.Constant) and key_node.value in keys:
                            result[key_node.value] = _literal(val_node, _module_constants(path))
                    return result
    return {}


@dataclass(frozen=True)
class SACConfig:
    train_env: dict
    eval_env: dict
    algo: dict
    manifest: Path
    config_paths: tuple[Path, ...]

    def to_dict(self) -> dict:
        return {'train_env': self.train_env, 'eval_env': self.eval_env,
                'algo': self.algo, 'manifest': str(self.manifest),
                'reference_configs': [str(p) for p in self.config_paths]}


def load_config(require_data: bool = True) -> SACConfig:
    """Materialize the effective *compact MBPO task* for an independent SAC run.

    Only optimizer and replay implementation are SAC-specific; environmental
    and finite-horizon task settings are inherited from the reference sources.
    """
    nsrdb = _named_function_dict(_BASE, 'nsrdb_environment_kwargs', 'kw')
    nsrdb.update(_call_kwargs(_COMPACT, 'build_nsrdb_params', 'train_env_kwargs'))
    eval_env = _reference_eval_defaults(_BASE)
    eval_env.update(_call_kwargs(_COMPACT, 'build_nsrdb_params', 'eval_env_kwargs'))
    for env in (nsrdb, eval_env):
        env['scenario_manifest'] = str((ROOT / _constant(_BASE, 'NSRDB_MANIFEST')).resolve())
    skeleton = _literal_mapping_fields(_SKELETON, 'params', {'discount', 'tau', 'reward_scale', 'n_train_repeat', 'n_epochs', 'n_initial_exploration_steps', 'min_alpha'})
    base_algo = _updated_dict(_BASE, 'default_nsrdb_algo_kwargs', 'kw')
    # The default NSRDB algorithm block is a literal dict, and conf_compact overrides it.
    skeleton.update({k: v for k, v in base_algo.items() if k in {'discount', 'tau', 'reward_scale', 'n_train_repeat', 'n_epochs', 'n_initial_exploration_steps', 'min_alpha'}})
    skeleton.update(_call_kwargs(_COMPACT, 'build_nsrdb_params', 'algo_kwargs'))
    policy = _call_kwargs(_COMPACT, 'build_nsrdb_params', 'policy_params_kwargs')
    critic = _call_kwargs(_COMPACT, 'build_nsrdb_params', 'q_params_kwargs')
    replay = _call_kwargs(_COMPACT, 'build_nsrdb_params', 'replay_pool_params_kwargs')
    if tuple(policy.get('hidden_layer_sizes', ())) != tuple(critic.get('hidden_layer_sizes', ())) or not policy:
        raise ValueError('Reference actor/critic hidden layer dimensions missing or inconsistent')
    if nsrdb.get('observation_mode') != 'compact' or eval_env.get('observation_mode') != 'compact':
        raise ValueError('SAC requires the same compact observation mode as MBPO reference')
    if float(skeleton['discount']) != 1.0:
        raise ValueError('Reference discount changed; audit finite-horizon objective first')
    # SAC-only settings are named separately; do not copy model-based settings.
    algo = {'epochs': int(skeleton['n_epochs']), 'start_steps': int(skeleton['n_initial_exploration_steps']),
            'updates_per_step': int(skeleton['n_train_repeat']), 'gamma': float(skeleton['discount']),
            'reward_scale': float(skeleton['reward_scale']), 'tau': float(skeleton['tau']),
            'hidden_sizes': tuple(policy['hidden_layer_sizes']), 'batch_size': 256,
            'replay_size': int(replay.get('kwargs', {}).get('max_size', 50000)),
            'lr': 3e-4, 'alpha_init': 1.0, 'target_entropy': -2.0,
            'min_alpha': float(skeleton['min_alpha']),
            'eval_interval': 10, 'checkpoint_interval': 50}
    if (int(nsrdb['periods']) != 118 or nsrdb['start_time'] != '12:00'
            or nsrdb['freq'] != '5min' or nsrdb['tz'] != 'UTC'
            or int(eval_env['periods']) != 118 or eval_env['start_time'] != '12:00'
            or eval_env['freq'] != '5min' or eval_env['tz'] != 'UTC'):
        raise ValueError('Reference episode grid changed; verify horizon before proceeding')
    manifest = Path(nsrdb['scenario_manifest'])
    if require_data:
        if not manifest.is_file():
            raise FileNotFoundError(f'NSRDB manifest is missing: {manifest}\n'
                'Run existing: python scripts/prepare_nsrdb_newyork_catalog.py\n'
                'Do NOT silently fall back to clear-sky.')
        from mbpo.env.nsrdb_weather import load_scenario_manifest
        loaded = load_scenario_manifest(manifest)
        ids = {str(s['scenario_id']) for s in loaded['scenarios']}
        if not set(_constant(_BASE, 'NSRDB_VALIDATION_SCENARIO_IDS')) <= ids:
            raise ValueError('NSRDB scenario manifest is missing reference validation days')
        episode = loaded.get('episode', {})
        site = loaded.get('site', {})
        for field, expected in (('start_time', nsrdb['start_time']),
                                ('periods', nsrdb['periods']), ('freq', nsrdb['freq'])):
            if episode.get(field) != expected:
                raise ValueError(f'Manifest episode {field}: {episode.get(field)!r} != {expected!r}')
        for field, expected in (('latitude', nsrdb['latitude']), ('longitude', nsrdb['longitude'])):
            if not np_isclose(site.get(field), expected):
                raise ValueError(f'Manifest site {field}: {site.get(field)!r} != {expected!r}')
        for row in loaded['scenarios']:
            path = Path(row['_year_path'])
            if not path.is_file():
                raise FileNotFoundError(f'NSRDB year data missing: {path}')
    return SACConfig(nsrdb, eval_env, algo, manifest, (_BASE, _COMPACT, _SKELETON))


def np_isclose(actual, expected) -> bool:
    # Avoid NumPy as a configuration-loading dependency.
    try:
        return abs(float(actual) - float(expected)) < 1e-8
    except (ValueError, TypeError):
        return False


def write_snapshot(config: SACConfig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding='utf-8')
