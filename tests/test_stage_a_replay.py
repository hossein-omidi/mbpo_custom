"""Persistent MBPO ensemble validation membership and checkpoint roundtrip."""
import numpy as np
import pytest


def _buffer():
    from MBPO_workspace.src.replay import ScenarioReplayBuffer
    return ScenarioReplayBuffer(capacity=100,seed=47)


def _add(buffer,sid,n=6):
    s=np.array([.5,.2,.1,.4,0.,1.],dtype=np.float32)
    nxt=s.copy();nxt[5]-=1/117
    for _ in range(n):buffer.add(s,np.zeros(2,dtype=np.float32),.008,nxt,False,sid)


def test_stable_day_split_across_refits_and_roundtrip():
    pool=_buffer()
    for i in range(8):_add(pool,f'2024-01-{i+1:02d}')
    train1,held1=pool.scenario_split(.2,seed=1)
    train2,held2=pool.scenario_split(.2,seed=9812837)
    assert np.array_equal(train1,train2)
    assert np.array_equal(held1,held2)
    assert not set(pool.scenario_ids[train1]).intersection(pool.scenario_ids[held1])
    other=_buffer();other.load_state_dict(pool.state_dict())
    a,b=other.scenario_split(.2,seed=2)
    assert np.array_equal(a,train1) and np.array_equal(b,held1)
    for i in range(8,12):_add(pool,f'2024-01-{i+1:02d}')
    for key,role in other._validation_roles.items():
        assert pool._validation_roles[key]==role


def test_legacy_replay_checkpoint_refused_for_model_validation():
    pool=_buffer();_add(pool,'2024-01-01');_add(pool,'2024-01-02')
    old=pool.state_dict();old.pop('validation_contract')
    newer=_buffer()
    with pytest.raises(ValueError,match='stable validation contract'):
        newer.load_state_dict(old)
