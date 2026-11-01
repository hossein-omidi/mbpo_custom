"""Real/synthetic replay pools with persistent, nonleaking day holdouts.

Previously each model refit used a fresh day split: a scenario held out for
model evaluation could be learned from on the next refit. Roles are now fixed
upon first encounter and saved in the replay checkpoint. This is an INTERNAL
model-validation split, not the independent final policy test set.
"""
from __future__ import annotations

import hashlib
import numpy as np
from SAC_workspace.src.training.replay_buffer import ReplayBuffer


class ScenarioReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, seed=0):
        super().__init__(capacity=capacity,seed=seed)
        self.scenario_ids=np.empty(self.capacity,dtype=np.int32)
        self.scenario_names=[]
        self._scenario_index={}
        self.validation_seed=int(seed)
        self._validation_roles={}  # scenario ID -> 'train' or 'holdout'

    def _assign_role(self, sid, fraction=0.2):
        if sid in self._validation_roles:return
        if len(self._validation_roles)==0:
            role='train'
        elif len(self._validation_roles)==1:
            role='holdout'
        else:
            raw=f'{self.validation_seed}|{sid}|mbpo-validation-v1'.encode()
            u=int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')/float(2**64)
            role='holdout' if u<float(fraction) else 'train'
        self._validation_roles[sid]=role

    def add(self,state,action,reward,next_state,done,scenario_id):
        if not scenario_id:raise ValueError('Real transitions require an NSRDB scenario ID')
        sid=str(scenario_id)
        if sid not in self._scenario_index:
            self._scenario_index[sid]=len(self.scenario_names)
            self.scenario_names.append(sid)
            self._assign_role(sid)
        slot=self.head
        super().add(state,action,reward,next_state,done)
        self.scenario_ids[slot]=self._scenario_index[sid]

    def scenario_split(self,fraction,seed):
        """Stable all-rows-of-a-day validation split, independent of refit RNG.

        ``seed`` remains in the method signature for dynamics API compatibility
        but is not used: the original replay initialization seed establishes
        persistent membership. Never reshuffle validation days across refits.
        """
        if not 0<fraction<0.5:raise ValueError('Holdout fraction must lie in (0,0.5)')
        active=self.scenario_ids[:self.count]
        unique=np.unique(active)
        if len(unique)<2:raise ValueError('Need >=2 distinct real NSRDB days')
        roles=np.array([self._validation_roles[self.scenario_names[int(i)]]
                        for i in active],dtype=object)
        train=np.flatnonzero(roles=='train')
        holdout=np.flatnonzero(roles=='holdout')
        if not len(train) or not len(holdout):
            raise ValueError('Replay window lacks one reserved role; more distinct days needed')
        if set(active[train]).intersection(set(active[holdout])):
            raise RuntimeError('Model validation leakage: same scenario in train/holdout')
        return train,holdout

    def rollout_states(self,n,seed=None):
        valid=np.flatnonzero((self.states[:self.count,5]>1e-7)
                                 &(self.dones[:self.count,0]<.5))
        if not len(valid):return np.empty((0,6),dtype=np.float32)
        rng=self.rng if seed is None else np.random.default_rng(seed)
        indices=rng.choice(valid,size=int(n),replace=len(valid)<int(n))
        return self.states[indices].copy()

    def state_dict(self):
        state=super().state_dict()
        state['scenario_ids']=self.scenario_ids[:self.count].copy()
        state['scenario_names']=list(self.scenario_names)
        state['validation_seed']=self.validation_seed
        state['validation_roles']=dict(self._validation_roles)
        state['validation_contract']='sticky_day_assignment_v1'
        return state

    def load_state_dict(self,data):
        super().load_state_dict(data)
        self.scenario_names=list(data['scenario_names'])
        self._scenario_index={name:i for i,name in enumerate(self.scenario_names)}
        ids=np.asarray(data['scenario_ids'],dtype=np.int32)
        if ids.shape!=(self.count,) or np.any(ids<0) or np.any(ids>=len(self.scenario_names)):
            raise ValueError('Corrupt scenario-ID metadata in replay checkpoint')
        self.scenario_ids[:self.count]=ids
        if data.get('validation_contract')!='sticky_day_assignment_v1':
            raise ValueError('Existing MBPO replay lacks stable validation contract; '
                             'train a new Stage A model rather than resuming old replay')
        self.validation_seed=int(data['validation_seed'])
        self._validation_roles=dict(data['validation_roles'])
        if set(self.scenario_names)!=set(self._validation_roles):
            raise ValueError('Invalid persisted MBPO scenario validation membership')
        if not set(self._validation_roles.values())<={'train','holdout'}:
            raise ValueError('Unknown MBPO model validation role')


def mix_batches(real:ReplayBuffer,model:ReplayBuffer,batch_size:int,
                real_ratio:float,rng:np.random.Generator):
    n_real=int(batch_size*real_ratio)
    n_fake=batch_size-n_real
    if len(model)<n_fake or n_fake==0:
        return real.sample(batch_size),batch_size,0
    if n_real<=0 or len(real)<n_real:
        raise ValueError('Insufficient real transitions')
    rb,mb=real.sample(n_real),model.sample(n_fake)
    order=rng.permutation(batch_size)
    return {k:np.concatenate([rb[k],mb[k]],axis=0)[order] for k in rb},n_real,n_fake
