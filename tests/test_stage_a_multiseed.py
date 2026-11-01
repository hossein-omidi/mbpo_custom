import csv
from pathlib import Path
import importlib.util
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('diag_stage_a',ROOT/'scripts/sac_mbpo_deep_diagnostic.py')
# The diagnostic script may depend on torch but never runs training when imported.
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_multiple_seed_rows_are_not_concatenated(tmp_path):
    p=tmp_path/'trajectories_detailed.csv'
    rows=[]
    for sid in ('2024-01-15','2024-07-19'):
        for seed in (42,43):
            for method in ('sac','mbpo'):
                rows.extend(dict(scenario_id=sid,seed=seed,method=method,step=t)
                            for t in range(1,mod.H+1))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['scenario_id','seed','method','step'])
        w.writeheader();w.writerows(rows)
    grouped=mod.read_evaluation_actions(tmp_path)
    assert len(grouped)==4
    for pair,methods in grouped.items():
        assert set(methods)=={'sac','mbpo'}
        assert len(methods['sac'])==mod.H
        assert all(int(r['seed'])==pair[1] for r in methods['mbpo'])


def test_incomplete_seed_pair_fails_loudly(tmp_path):
    p=tmp_path/'trajectories_detailed.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['scenario_id','seed','method','step'])
        w.writeheader();w.writerows(dict(scenario_id='2024-01-15',seed=42,method='sac',step=t)
                                   for t in range(1,mod.H+1))
    with pytest.raises((AssertionError,ValueError),match='Missing paired policy'):
        mod.read_evaluation_actions(tmp_path)
