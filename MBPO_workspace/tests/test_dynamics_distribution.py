"""No PVLib or NSRDB data is needed for these probabilistic model unit tests."""
import unittest
import numpy as np
from MBPO_workspace.src.dynamics import DynamicsEnsemble


class DynamicsDistributionTests(unittest.TestCase):
    def setUp(self):
        self.model=DynamicsEnsemble(num_models=3,num_elites=2,hidden_dim=8,seed=11)
        self.model.fitted=True
        self.states=np.array([
            [0.2,0.1,0.3,0.4,-0.2,1.0],
            [0.4,0.2,0.5,0.6,0.1,0.5],
        ],dtype=np.float32)
        self.actions=np.array([[0.1,-0.2],[-0.4,0.3]],dtype=np.float32)

    def test_deterministic_prediction_matches_ensemble_mean(self):
        model=self.model
        mu,ale,epi,total,score=model.predict_distribution(self.states,self.actions)
        delta,unc=model.predict_delta(self.states,self.actions,stochastic=False)
        self.assertEqual(mu.shape,(2,3))
        self.assertEqual(ale.shape,(2,3))
        self.assertEqual(epi.shape,(2,3))
        self.assertEqual(total.shape,(2,3))
        self.assertEqual(score.shape,(2,))
        np.testing.assert_allclose(delta,mu,rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(unc,score,rtol=1e-6,atol=1e-7)
        self.assertTrue(np.all(total>=ale-1e-7))

    def test_weather_prediction_is_action_invariant(self):
        first,_=self.model.predict_delta(self.states,self.actions,stochastic=False)
        other=np.array([[-0.8,0.6],[0.9,-0.8]],dtype=np.float32)
        second,_=self.model.predict_delta(self.states,other,stochastic=False)
        np.testing.assert_array_equal(first[:,:2],second[:,:2])
        # Weather network explicitly receives only irradiance + time, not action.
        weather_x, weather_y, _, _ = self.model.make_xy(self._toy_replay(),np.arange(8))
        self.assertEqual(weather_x.shape[1],3)
        self.assertEqual(weather_y.shape[1],2)

    @staticmethod
    def _toy_replay():
        class Replay: pass
        r=Replay()
        r.states=np.tile(np.array([.2,.3,.1,.4,-.2,1.],dtype=np.float32),(8,1))
        r.actions=np.zeros((8,2),dtype=np.float32)
        r.next_states=r.states.copy()
        r.next_states[:,:2]+=.01
        return r

    def test_checkpoint_roundtrip_uses_new_architecture_schema(self):
        saved=self.model.state_dict()
        other=DynamicsEnsemble(num_models=3,num_elites=2,hidden_dim=8,seed=11)
        other.load_state_dict(saved)
        mu1=self.model.predict_delta(self.states,self.actions,stochastic=False)[0]
        mu2=other.predict_delta(self.states,self.actions,stochastic=False)[0]
        np.testing.assert_allclose(mu1,mu2,atol=0,rtol=0)
        with self.assertRaises(ValueError):
            other.load_state_dict({**saved,'schema':1})

if __name__=='__main__':unittest.main()
