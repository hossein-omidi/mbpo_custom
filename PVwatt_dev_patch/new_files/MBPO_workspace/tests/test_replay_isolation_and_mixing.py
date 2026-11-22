import unittest
import numpy as np
from MBPO_workspace.src.replay import ScenarioReplayBuffer,mix_batches
from SAC_workspace.src.training.replay_buffer import ReplayBuffer


class ModelReplayIsolation(unittest.TestCase):
    def test_holdout_days_are_not_synthetic_start_states(self):
        pool=ScenarioReplayBuffer(capacity=512,seed=5,validation_fraction=.2)
        for d in range(12):
            for t in range(12):
                # Distinct DNI value encodes the scenario ID in this fixture.
                state=np.array([d/20.,.4,.4,.5,0.,1.-t/20.],dtype=np.float32)
                next_state=state.copy();next_state[5]-=.01
                pool.add(state,np.zeros(2,dtype=np.float32),0.0,next_state,False,f'2024-01-{d+1:02d}')
        train,hold=pool.scenario_split(.2,seed=17)
        heldout={int(round(float(v)*20)) for v in pool.states[hold,0]}
        start=pool.rollout_states(100,seed=18)
        assert len(start)==100
        sampled={int(round(float(v)*20)) for v in start[:,0]}
        self.assertFalse(heldout.intersection(sampled))
        self.assertGreater(pool.active_train_scenario_count,0)
        self.assertGreater(pool.active_holdout_scenario_count,0)
        restored=ScenarioReplayBuffer(capacity=512,seed=5,validation_fraction=.2)
        restored.load_state_dict(pool.state_dict())
        np.testing.assert_array_equal(restored.rollout_states(30,seed=11),
                                      pool.rollout_states(30,seed=11))

    def test_partial_synthetic_batch_is_used_without_changing_size(self):
        real=ReplayBuffer(capacity=128,seed=1)
        model=ReplayBuffer(capacity=16,seed=2)
        for t in range(64):
            s=np.ones(6,dtype=np.float32)*t/100.0
            real.add(s,np.zeros(2,dtype=np.float32),0.0,s,False)
        for t in range(5):
            s=np.ones(6,dtype=np.float32)*.25
            model.add(s,np.zeros(2,dtype=np.float32),0.0,s,False)
        batch,n_real,n_model=mix_batches(real,model,32,.75,np.random.default_rng(5))
        self.assertEqual((n_real,n_model),(27,5))
        self.assertEqual(batch['states'].shape,(32,6))

if __name__=='__main__':unittest.main()
