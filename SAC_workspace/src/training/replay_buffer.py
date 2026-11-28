"""Fixed-size FIFO replay memory using ONLY transitions from PVTrackingEnv.step."""
from __future__ import annotations
from mbpo.geometry10 import ACTION_SEMANTICS, STATE_DIM, ACTION_DIM
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim=STATE_DIM, action_dim=ACTION_DIM, seed=0):
        self.capacity = int(capacity)
        if self.capacity <= 0 or (state_dim, action_dim) != (STATE_DIM,ACTION_DIM):
            raise ValueError('Invalid reference replay buffer specification')
        self.state_dim = int(state_dim)
        self.states = np.empty((capacity, self.state_dim), dtype=np.float32)
        self.actions = np.empty((capacity, 2), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_states = np.empty((capacity, self.state_dim), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        self.count, self.head = 0, 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.count

    def add(self, state, action, reward, next_state, done):
        from SAC_workspace.src.representation.representation import encode
        state = encode(state)
        next_state = encode(next_state)
        action = np.asarray(action, dtype=np.float32)
        if (action.shape != (2,) or not np.isfinite(action).all()
                or (np.abs(action) > 1.0).any()):
            raise ValueError('Replay action must be finite and inside [-1,1]^2')
        if not np.isfinite(reward) or done not in (0, 1, False, True):
            raise ValueError('Invalid transition reward or terminal flag')
        i = self.head
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i, 0] = float(reward)
        self.next_states[i] = next_state
        self.dones[i, 0] = float(done)
        self.head = (self.head + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, batch_size):
        if self.count < batch_size:
            raise ValueError(f'Need {batch_size} transitions; have {self.count}')
        ix = self.rng.integers(0, self.count, size=batch_size)
        return {k: getattr(self, k)[ix].copy() for k in ('states','actions','rewards','next_states','dones')}

    def state_dict(self):
        return {'action_semantics': ACTION_SEMANTICS, 'state_dim': self.state_dim, 'capacity': self.capacity, 'count': self.count, 'head': self.head,
                'rng': self.rng.bit_generator.state,
                **{k: getattr(self,k)[:self.count].copy() for k in ('states','actions','rewards','next_states','dones')}}

    def load_state_dict(self, data):
        if data.get('action_semantics') != ACTION_SEMANTICS:
            raise ValueError('Checkpoint uses incompatible PV action semantics')
        if data.get('state_dim') != self.state_dim:
            raise ValueError('Replay observation dimension differs from checkpoint')
        if data['capacity'] != self.capacity:
            raise ValueError('Replay capacity differs from saved checkpoint')
        self.count, self.head = int(data['count']), int(data['head'])
        if not 0 <= self.count <= self.capacity or not 0 <= self.head < self.capacity:
            raise ValueError('Corrupt replay indices')
        for k in ('states','actions','rewards','next_states','dones'):
            expected = getattr(self,k)[:self.count].shape
            if np.asarray(data[k]).shape != expected or not np.isfinite(data[k]).all():
                raise ValueError(f'Corrupt replay field {k}: expected {expected}')
            getattr(self,k)[:self.count] = data[k]
        from mbpo.geometry10 import check_batch
        check_batch(self.states[:self.count]); check_batch(self.next_states[:self.count])
        if (self.count < self.capacity and self.head != self.count) or np.any(np.abs(self.actions[:self.count]) > 1.0) or not np.isin(self.dones[:self.count], [0., 1.]).all():
            raise ValueError('Invalid saved replay indices, actions or terminal flags')
        self.rng.bit_generator.state = data['rng']
