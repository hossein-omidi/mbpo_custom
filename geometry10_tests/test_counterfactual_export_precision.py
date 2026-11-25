"""Regression of the counterfactual CSV producer's actual row expression.
This deliberately isolates the export calculation, not simulator/model execution.
"""
import ast
import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PV_experiments.scientific_checks import same


def test_counterfactual_export_residuals_roundtrip():
    source = Path(__file__).resolve().parents[1] / 'MBPO_workspace/workflows/06_model_validation.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    rows = [n.args[0] for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == 'append'
            and isinstance(n.func.value, ast.Name) and n.func.value.id == 'counterfactual_rows'
            and len(n.args) == 1 and isinstance(n.args[0], ast.Dict)]
    assert len(rows) == 1
    expr = compile(ast.Expression(rows[0]), str(source), 'eval')
    rng = np.random.default_rng(2026)
    pairs = [(np.float32(.0001), np.float32(.05)),
             (np.float32(.000001), np.float32(.9)),
             (np.float32(.9), np.float32(.000001))]
    pairs += list(zip(rng.uniform(0, 1, 2048).astype(np.float32),
                      rng.uniform(0, 1, 2048).astype(np.float32)))
    records = []
    for t, (a, p) in enumerate(pairs):
        actual = np.array([a,a,a,.5,0,1,0,0,1,0], dtype=np.float32)
        predicted = actual.copy(); predicted[:3] = p
        row = eval(expr, {'float':float, 'int':int, 'bool':bool}, dict(
            sid='2020-01-01',t=t,action_name='zero',candidate=np.zeros(2),
            actual_reward=float(a)/6,pred_reward=np.array([float(p)/6]),
            actual_done=False,pred_done=np.array([False]),accepted=np.array([True]),
            unc=np.array([.01]),actual_nxt=actual,pnext=predicted))
        records.append(row)
    csv = pd.read_csv(io.StringIO(pd.DataFrame(records).to_csv(index=False)),float_precision='round_trip')
    for field in ('power','dni','dhi'):
        expected = csv[f'predicted_next_{field}_norm'] - csv[f'actual_next_{field}_norm']
        same(expected,csv[f'{field}_error_norm'],field+' export',atol=0,rtol=0)
    same(csv.predicted_reward_kwh-csv.actual_reward_kwh,csv.reward_error_kwh,'reward export',atol=0,rtol=0)
    # The first fixture reproduces the previous validator failure. A genuinely
    # corrupted residual must still be rejected by the unchanged verifier.
    old = float(np.float32(.05)-np.float32(.0001))
    with pytest.raises(ValueError, match='numerical/shape mismatch'):
        same(np.array([csv.power_error_norm.iloc[0]]),np.array([old]),'old float32 residual')
    wrong = csv.power_error_norm.copy(); wrong.iloc[0] += 1e-5
    with pytest.raises(ValueError, match='numerical/shape mismatch'):
        same(csv.predicted_next_power_norm-csv.actual_next_power_norm,wrong,'corrupted residual')
    # Check aggregate diagnostics from serialized operands, without relaxing tolerances.
    for field in ('power', 'dni', 'dhi'):
        expected = csv[f'predicted_next_{field}_norm'] - csv[f'actual_next_{field}_norm']
        recorded = csv[f'{field}_error_norm']
        def metrics(x):
            return np.array([np.sqrt(np.mean(x*x)), np.mean(np.abs(x)), np.mean(x)])
        same(metrics(expected), metrics(recorded), field+' summary',atol=0,rtol=0)
    same(csv.reward_error_kwh,csv.power_error_norm/6,'reward/power identity',atol=1e-15,rtol=0)
