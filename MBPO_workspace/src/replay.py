"""Separate real and imaginary buffers; scenario metadata tracks complete days.
Uses the same SAC_workspace transition schema and sampling implementation.
"""
from __future__ import annotations
import numpy as np
from SAC_workspace.src.training.replay_buffer import ReplayBuffer


class ScenarioReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, seed=0):
        super().__init__(capacity=capacity, seed=seed)
        self.scenario_ids = np.empty(self.capacity, dtype=np.int32)
        self.scenario_names = []
        self._scenario_index = {}

    def add(self, state, action, reward, next_state, done, scenario_id):
        if not scenario_id: raise ValueError('Real transitions require an NSRDB scenario ID')
        sid = str(scenario_id)
        if sid not in self._scenario_index:
            self._scenario_index[sid] = len(self.scenario_names)
            self.scenario_names.append(sid)
        slot = self.head
        super().add(state, action, reward, next_state, done)
        self.scenario_ids[slot] = self._scenario_index[sid]

    def scenario_split(self, fraction, seed):
        """Return train/holdout row indices, separated by complete scenario IDs.
        No weather trajectory is in both sets, even for repeated sampled days.
        """
        active = self.scenario_ids[:self.count]
        unique = np.unique(active)
        if len(unique) < 2: raise ValueError('Need >=2 distinct NSRDB scenarios for day-level holdout')
        rng = np.random.default_rng(seed)
        chosen = rng.permutation(unique)
        k = max(1, min(len(unique)-1, int(round(fraction*len(unique)))))
        holdout = np.flatnonzero(np.isin(active, chosen[:k]))
        train = np.flatnonzero(~np.isin(active, chosen[:k]))
        if not len(holdout) or not len(train): raise RuntimeError('Empty dynamics split')
        assert not set(active[train]).intersection(set(active[holdout]))
        return train, holdout

    def rollout_states(self, n, seed=None):
        # The six-component time-to-go is exact. Do not sample terminal states.
        valid = np.flatnonzero((self.states[:self.count, 5] > 1e-7)
                                   & (self.dones[:self.count, 0] < .5))
        if not len(valid): return np.empty((0,6),dtype=np.float32)
        rng = self.rng if seed is None else np.random.default_rng(seed)
        ix = rng.choice(valid,size=int(n),replace=len(valid)<n)
        return self.states[ix].copy()

    def state_dict(self):
        state = super().state_dict()
        state['scenario_ids'] = self.scenario_ids[:self.count].copy()
        state['scenario_names'] = list(self.scenario_names)
        return state

    def load_state_dict(self, data):
        super().load_state_dict(data)
        self.scenario_names = list(data['scenario_names'])
        self._scenario_index = {k:i for i,k in enumerate(self.scenario_names)}
        ids = np.asarray(data['scenario_ids'],dtype=np.int32)
        if ids.shape != (self.count,) or np.any(ids<0) or np.any(ids>=len(self.scenario_names)):
            raise ValueError('Corrupt scenario metadata in replay checkpoint')
        self.scenario_ids[:self.count] = ids


def mix_batches(real: ReplayBuffer, model: ReplayBuffer, batch_size: int,
                real_ratio: float, rng: np.random.Generator):
    """Exact real fraction when model replay is available; otherwise real-only.
    Keeps the sample source separated while training with the shared SACAgent.
    """
    n_real = int(batch_size * real_ratio)
    n_fake = batch_size - n_real
    if len(model) < n_fake or n_fake == 0:
        return real.sample(batch_size), batch_size, 0
    if n_real <= 0 or len(real) < n_real: raise ValueError('Insufficient real transitions')
    rb, mb = real.sample(n_real), model.sample(n_fake)
    perm = rng.permutation(batch_size)
    return {k: np.concatenate([rb[k],mb[k]],axis=0)[perm]
            for k in rb}, n_real, n_fake
