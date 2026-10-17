import copy
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from SAC_workspace.src.training.replay_buffer import ReplayBuffer
from MBPO_workspace.src.dynamics import DynamicsEnsemble,StructuredFakeEnv,gaussian_nll
from MBPO_workspace.src.replay import ScenarioReplayBuffer,mix_batches


def mkstate(tau=1., tilt=.4, az=0.,p=.1):
    return np.array([.4,.25,p,tilt,az,tau],dtype=np.float32)


class FakePredictor:
    def predict_delta(self, states, actions, stochastic=True):
        deltas=np.zeros((len(states),3),dtype=np.float32)
        deltas[:,2]=.1
        return deltas,np.zeros(len(states),dtype=np.float32)


class FakeUnderlying:
    max_delta_tilt=5.;max_delta_azimuth=10.;movement_penalty=0.
    movement_cost_mode='legacy';actuator_power_w=30.;slew_rate_tilt_deg_s=1.5
    slew_rate_azimuth_deg_s=2.;interval_hours=1./12.

class ModelTests(unittest.TestCase):
    def test_gaussian_nll_is_exact(self):
        mu=torch.zeros(4,3);ls=torch.zeros(4,3);target=torch.zeros(4,3)
        self.assertAlmostEqual(gaussian_nll(mu,ls,target).item(),1.5*np.log(2*np.pi),places=5)

    def test_group_level_holdout(self):
        b=ScenarioReplayBuffer(500,seed=6)
        for group in ['2018-01-01','2018-01-02','2019-01-01','2019-01-02']:
            for t in range(60):
                b.add(mkstate(tau=(117-t)/117),np.zeros(2),.01,
                      mkstate(tau=(116-t)/117),False,group)
        a,c=b.scenario_split(.25,seed=4)
        self.assertFalse(set(b.scenario_ids[a]).intersection(set(b.scenario_ids[c])))
        self.assertEqual(len(a)+len(c),len(b))
        data=b.state_dict();b2=ScenarioReplayBuffer(500);b2.load_state_dict(data)
        self.assertEqual(len(b2),len(b));self.assertEqual(b2.scenario_names,b.scenario_names)

    def test_mix_never_promotes_model_data_to_real_pool(self):
        real=ScenarioReplayBuffer(100,seed=2)
        model=ReplayBuffer(100,seed=3)
        for t in range(80):
            s=mkstate(tau=(117-t)/117)
            sn=mkstate(tau=(116-t)/117)
            real.add(s,np.zeros(2),.01,sn,False,'date')
            model.add(s,np.zeros(2),.02,sn,False)
        batch,nr,nm=mix_batches(real,model,20,.75,np.random.default_rng(4))
        self.assertEqual((nr,nm),(15,5))
        self.assertEqual(int(np.isclose(batch['rewards'],.02).sum()),5)

    def test_one_step_physical_projection_and_terminal_mask(self):
        physics=types.ModuleType('mbpo.env.pvlib_physics')
        physics.energy_kwh_from_power_w=lambda power_w,interval_hours:power_w*interval_hours/1000.
        physics.actuator_movement_cost_kwh=lambda *args,**kwargs:0.
        envmod=types.ModuleType('mbpo.env.pv_tracking');envmod.POWER_NORM=2000.
        dummy=types.SimpleNamespace(unwrapped=FakeUnderlying(),horizon=117)
        with patch.dict(sys.modules,{'mbpo':types.ModuleType('mbpo'),
                 'mbpo.env':types.ModuleType('mbpo.env'),
                 'mbpo.env.pvlib_physics':physics,'mbpo.env.pv_tracking':envmod}):
            fe=StructuredFakeEnv(FakePredictor(),dummy)
            s=np.stack([mkstate(tau=1/117,tilt=.99,az=.99,p=.5),
                        mkstate(tau=1.,tilt=.1,az=-.99,p=.2)])
            a=np.array([[1,1],[-1,-1]],dtype=np.float32)
            ns,r,d,uncertainty,valid=fe.step(s,a)
        np.testing.assert_array_equal(d,[True,False])
        self.assertAlmostEqual(float(ns[0,5]),0,places=6)
        self.assertAlmostEqual(float(ns[1,5]),1-1/117,places=6)
        self.assertAlmostEqual(float(ns[0,3]),1.,places=6)
        self.assertTrue(-1<=ns[0,4]<1)
        self.assertTrue(np.all(valid))
        self.assertAlmostEqual(float(r[0]),.6*2000/12000,places=6)

    def test_probabilistic_ensemble_fits_and_restores(self):
        b=ScenarioReplayBuffer(1000,seed=1)
        rng=np.random.default_rng(5)
        for i in range(400):
            s=mkstate(tau=(117-(i%100))/117,p=float(rng.uniform(.04,.2)))
            a=rng.uniform(-1,1,size=2).astype(np.float32)
            nxt=s.copy();nxt[:3]=np.clip(nxt[:3]+rng.normal(0,.01,3),0,1);nxt[-1]-=1/117
            b.add(s,a,float(nxt[2]*.001),nxt,False,f'2019-06-{1+i%8:02d}')
        ens=DynamicsEnsemble(num_models=2,num_elites=2,hidden_dim=8,seed=5)
        metrics=ens.fit(b,epochs=2,patience=2,max_samples=240,batch_size=64)
        self.assertTrue(np.isfinite(metrics['model_val_nll']))
        delta,unc=ens.predict_delta(b.states[:9],b.actions[:9])
        self.assertEqual(delta.shape,(9,3));self.assertTrue(np.isfinite(delta).all())
        state=ens.state_dict()
        ens2=DynamicsEnsemble(num_models=2,num_elites=2,hidden_dim=8,seed=14)
        ens2.load_state_dict(state)
        np.testing.assert_allclose(ens2.predict_delta(b.states[:9],b.actions[:9])[0],
                                   ens.predict_delta(b.states[:9],b.actions[:9])[0])

if __name__=='__main__':unittest.main()
