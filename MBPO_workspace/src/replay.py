# PVWATT_DEV_REPAIR_V3
"""Real/synthetic replay pools with persistent, nonleaking day holdouts.

Previously each model refit used a fresh day split: a scenario held out for
model evaluation could be learned from on the next refit. Roles are now fixed
upon first encounter and saved in the replay checkpoint. This is an INTERNAL
model-validation split, not the independent final policy test set.
"""
from __future__ import annotations
from mbpo.geometry10 import STATE_DIM, TAU

import hashlib
import numpy as np
from SAC_workspace.src.training.replay_buffer import ReplayBuffer


class ScenarioReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, seed=0, validation_fraction=0.2):
        super().__init__(capacity=capacity,seed=seed)
        self.validation_fraction=float(validation_fraction)
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError('validation_fraction must lie in (0, 0.5)')
        self.scenario_ids=np.empty(self.capacity,dtype=np.int32)
        self.scenario_names=[]
        self._scenario_index={}
        self.validation_seed=int(seed)
        self._validation_roles={}  # scenario ID -> 'train' or 'holdout'

    def _assign_role(self, sid, fraction=None):
        if sid in self._validation_roles:return
        fraction=self.validation_fraction if fraction is None else float(fraction)
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

    @property
    def active_holdout_scenario_count(self):
        if self.count == 0:
            return 0
        active = np.unique(self.scenario_ids[:self.count])
        return sum(self._validation_roles.get(self.scenario_names[int(i)]) == 'holdout'
                   for i in active)

    @property
    def active_train_scenario_count(self):
        if self.count == 0:
            return 0
        active = np.unique(self.scenario_ids[:self.count])
        return sum(self._validation_roles.get(self.scenario_names[int(i)]) == 'train'
                   for i in active)

    def scenario_split(self,fraction,seed):
        """Stable all-rows-of-a-day validation split, independent of refit RNG.

        ``seed`` remains in the method signature for dynamics API compatibility
        but is not used: the original replay initialization seed establishes
        persistent membership. Never reshuffle validation days across refits.
        """
        if not 0<fraction<0.5:raise ValueError('Holdout fraction must lie in (0,0.5)')
        if not np.isclose(float(fraction),self.validation_fraction,rtol=0,atol=1e-12):
            raise ValueError('Dynamics holdout fraction differs from persistent replay assignment')
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

    def rollout_states(self,n,seed=None,*,with_scenarios=False):
        # No holdout-day state is ever used to generate synthetic policy training.
        valid=np.flatnonzero((self.states[:self.count,TAU]>1e-7)
                                 &(self.dones[:self.count,0]<.5))
        valid=valid[np.asarray([
            self._validation_roles[self.scenario_names[int(self.scenario_ids[i])]] == 'train'
            for i in valid],dtype=bool)]
        if not len(valid):
            empty=np.empty((0,STATE_DIM),dtype=np.float32)
            return (empty, []) if with_scenarios else empty
        rng=self.rng if seed is None else np.random.default_rng(seed)
        indices=rng.choice(valid,size=int(n),replace=len(valid)<int(n))
        states=self.states[indices].copy()
        if with_scenarios:
            return states, [self.scenario_names[int(self.scenario_ids[i])] for i in indices]
        return states

    def state_dict(self):
        state=super().state_dict()
        state['scenario_ids']=self.scenario_ids[:self.count].copy()
        state['scenario_names']=list(self.scenario_names)
        state['validation_seed']=self.validation_seed
        state['validation_roles']=dict(self._validation_roles)
        state['validation_fraction']=self.validation_fraction
        state['validation_contract']='sticky_day_assignment_v3_train_only_rollouts'
        return state

    def load_state_dict(self,data):
        super().load_state_dict(data)
        self.scenario_names=list(data['scenario_names'])
        self._scenario_index={name:i for i,name in enumerate(self.scenario_names)}
        ids=np.asarray(data['scenario_ids'],dtype=np.int32)
        if ids.shape!=(self.count,) or np.any(ids<0) or np.any(ids>=len(self.scenario_names)):
            raise ValueError('Corrupt scenario-ID metadata in replay checkpoint')
        self.scenario_ids[:self.count]=ids
        if data.get('validation_contract')!='sticky_day_assignment_v3_train_only_rollouts':
            raise ValueError('Existing MBPO replay lacks stable validation contract; '
                             'train a new Stage A model rather than resuming old replay')
        self.validation_seed=int(data['validation_seed'])
        self.validation_fraction=float(data['validation_fraction'])
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError('Corrupt saved validation fraction')
        self._validation_roles=dict(data['validation_roles'])
        if set(self.scenario_names)!=set(self._validation_roles):
            raise ValueError('Invalid persisted MBPO scenario validation membership')
        if not set(self._validation_roles.values())<={'train','holdout'}:
            raise ValueError('Unknown MBPO model validation role')


def mix_batches(real:ReplayBuffer,model:ReplayBuffer,batch_size:int,
                real_ratio:float,rng:np.random.Generator):
    batch_size = int(batch_size)
    real_ratio = float(real_ratio)
    if batch_size < 2 or not 0 < real_ratio < 1:
        raise ValueError('Invalid mixed minibatch proportions')
    desired_real = max(1, int(batch_size * real_ratio))
    desired_model = batch_size-desired_real
    n_model = min(desired_model, len(model))
    n_real = batch_size-n_model
    if len(real) < n_real:
        raise ValueError('Insufficient real replay transitions')
    if n_model == 0:
        return real.sample(batch_size),batch_size,0
    rb,mb=real.sample(n_real),model.sample(n_model)
    order=rng.permutation(batch_size)
    return {k:np.concatenate([rb[k],mb[k]],axis=0)[order] for k in rb},n_real,n_model
