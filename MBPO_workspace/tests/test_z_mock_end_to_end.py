"""Deterministic stand-in for external NSRDB/pvlib so Torch MBPO can be
integration-tested without claiming to execute actual physics in CI."""
import tempfile
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from SAC_workspace.src.config.config import SACConfig
from MBPO_workspace.src.config import ModelConfig,MBPOConfig
from MBPO_workspace.src.trainer import MBPOTrainer

class DummyWorld:
    max_delta_tilt=5.;max_delta_azimuth=10.;movement_penalty=0.
    movement_cost_mode='legacy';actuator_power_w=30.;slew_rate_tilt_deg_s=1.5
    slew_rate_azimuth_deg_s=2.;interval_hours=1/12
    def __init__(self):self.np_random=np.random.RandomState(0)

class DummyPV:
    horizon=117
    resolved_mdp={'testing':'mock-only; never scientific solar results'}
    def __init__(self,kwargs):
        self.unwrapped=DummyWorld();self.ep=0;self.stepn=0
    def seed(self,seed):self.unwrapped.np_random.seed(seed)
    def reset(self):
        self.ep+=1;self.stepn=0
        self.state=np.array([.5,.2,.12,.5,0.,1.],dtype=np.float32)
        return self.state.copy()
    def step(self,action):
        self.stepn+=1
        s=self.state.copy();s[0]=.5+.1*np.sin(self.stepn/30)
        s[1]=.2+.02*np.cos(self.stepn/30)
        s[3]=np.clip(s[3]+float(action[0])*5/90.,0,1)
        s[4]=((s[4]+1)*180+float(action[1])*10)%360/180.-1
        s[2]=np.clip(.12+.05*s[3]*s[0],0,1)
        s[5]=max(0.,1-self.stepn/117)
        self.state=s
        energy=float(s[2]*2000/12000)
        return s.copy(),energy,self.stepn==117,{
            'scenario_id':'2020-06-%02d'%(self.ep%20+1),
            'energy_kwh':energy,'movement_cost':0.}
    def close(self):pass


def mocked_physics():
    physics=types.ModuleType('mbpo.env.pvlib_physics')
    physics.energy_kwh_from_power_w=lambda p,h:float(p)*float(h)/1000.
    physics.actuator_movement_cost_kwh=lambda *args,**kwargs:0.
    envmod=types.ModuleType('mbpo.env.pv_tracking');envmod.POWER_NORM=2000.
    return patch.dict(sys.modules,{'mbpo':types.ModuleType('mbpo'),
       'mbpo.env':types.ModuleType('mbpo.env'),
       'mbpo.env.pvlib_physics':physics,'mbpo.env.pv_tracking':envmod})

class IntegrationTests(unittest.TestCase):
    def test_mock_training_synthetic_and_resume(self):
        import torch
        torch.set_num_threads(1)
        sac=SACConfig(train_env={'mock':True},eval_env={'mock':True},
                      algo={'epochs':2,'start_steps':140,'updates_per_step':1,
                       'gamma':1.,'tau':.005,'hidden_sizes':(8,8),'batch_size':32,
                       'replay_size':500,'lr':3e-4,'reward_scale':100.,
                       'alpha_init':1.,'target_entropy':-2.,'min_alpha':.001,
                       'checkpoint_interval':50},manifest=Path('dummy'),config_paths=tuple())
        model=ModelConfig(real_ratio=.75,num_models=2,num_elites=2,hidden_dim=8,
            rollout_length=1,model_train_freq=117,rollout_batch_size=40,
            model_pool_capacity=100,model_epochs=2,model_patience=2,require_model_skill=False,
            model_batch_size=64,model_max_train_samples=200)
        cfg=MBPOConfig(sac,model)
        with tempfile.TemporaryDirectory() as tmp, mocked_physics():
            t=MBPOTrainer(cfg,tmp,seed=5,env_factory=DummyPV)
            try:
                for _ in range(2):t.train_episode()
                self.assertGreater(t.dynamics.fit_count,0)
                self.assertGreater(len(t.model_pool),0)
                self.assertGreater(t.agent.updates,0)
                self.assertEqual(t.env_steps,234)
                self.assertEqual(t.real_pool.count,234)
                path=t.save()
            finally:t.env.close()
            t2=MBPOTrainer(cfg,tmp,seed=5,resume=path,env_factory=DummyPV)
            try:
                self.assertEqual(t2.env_steps,234)
                self.assertEqual(t2.agent.updates,t.agent.updates)
                self.assertEqual(len(t2.model_pool),len(t.model_pool))
                self.assertEqual(t2.dynamics.fit_count,1)
            finally:t2.env.close()

if __name__=='__main__':unittest.main()
