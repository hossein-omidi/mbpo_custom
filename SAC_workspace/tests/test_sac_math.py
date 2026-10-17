import unittest
import numpy as np
import torch
from SAC_workspace.src.agents.sac_agent import SACAgent, SquashedGaussianActor
from SAC_workspace.src.training.replay_buffer import ReplayBuffer

class SACMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
    def test_tanh_log_probability_gradient(self):
        actor = SquashedGaussianActor()
        s = torch.randn(32, 6)
        a, logp = actor(s)
        self.assertTrue(torch.isfinite(logp).all())
        self.assertTrue((a.abs() < 1.).all())
        (-logp.mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters()))
    def test_log_density_matches_transformed_distribution(self):
        from torch.distributions import Normal, TransformedDistribution, TanhTransform
        actor = SquashedGaussianActor()
        s = torch.randn(64,6) * .2
        a, lp = actor(s)
        m, ls = actor.net(s).chunk(2,dim=-1)
        dist = TransformedDistribution(Normal(m,ls.clamp(-20,2).exp()), [TanhTransform(cache_size=1)])
        expected = dist.log_prob(a).sum(-1,keepdim=True)
        torch.testing.assert_close(lp,expected,atol=1e-4,rtol=1e-4)

    def test_terminal_mask_and_exact_reward_scaling(self):
        agent = SACAgent()
        batch={'next_states':torch.zeros(2,6), 'rewards':torch.tensor([[.2],[.3]]), 'dones':torch.ones(2,1)}
        target=agent.compute_target(batch)
        torch.testing.assert_close(target, torch.tensor([[20.],[30.]]))
    def test_update_is_finite_and_changes_targets(self):
        agent=SACAgent()
        buf=ReplayBuffer(1024,seed=2)
        for i in range(300):
            buf.add(np.zeros(6),np.zeros(2),.02,np.zeros(6),i%117==116)
        before=[p.clone() for p in agent.critic_target.parameters()]
        stats=agent.update(buf.sample(32))
        self.assertEqual(agent.updates,1)
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,agent.critic_target.parameters())))
        for old, current, new in zip(before, agent.critic.parameters(), agent.critic_target.parameters()):
            torch.testing.assert_close(new, old * (1-agent.tau) + current.detach() * agent.tau, atol=2e-7, rtol=1e-6)
    def test_entropy_temperature_respects_reference_minimum(self):
        agent = SACAgent(alpha_init=1., min_alpha=0.001)
        with torch.no_grad():
            agent.log_alpha.fill_(-30.)
        self.assertAlmostEqual(agent.alpha, 0.001, places=6)
        batch = {'next_states': torch.zeros(4, 6), 'rewards': torch.zeros(4, 1),
                 'dones': torch.zeros(4, 1)}
        torch.manual_seed(3)
        expected = agent.compute_target(batch)
        torch.manual_seed(3)
        with torch.no_grad():
            a, lp = agent.actor(batch['next_states'])
            q1, q2 = agent.critic_target(batch['next_states'], a)
            reference = torch.minimum(q1,q2) - 0.001*lp
        torch.testing.assert_close(expected, reference)

    def test_ring_buffer_restores_identical_sampling_state(self):
        a = ReplayBuffer(5, seed=88)
        for i in range(9):
            s = np.array([.1,.2,.05,.3,-.1, (117-(i+1))/117], dtype=np.float32)
            a.add(s, np.zeros(2, dtype=np.float32), i/2000., s, bool(i % 4 == 0))
        b = ReplayBuffer(5, seed=999)
        b.load_state_dict(a.state_dict())
        self.assertEqual((a.count, a.head), (b.count, b.head))
        for key in ('states','actions','rewards','next_states','dones'):
            np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
        for i in range(4):
            x,y=a.sample(3),b.sample(3)
            for key in x:
                np.testing.assert_array_equal(x[key],y[key])

    def test_buffer_requires_collected_size(self):
        b=ReplayBuffer(1000)
        with self.assertRaises(ValueError):b.sample(256)

if __name__=='__main__':unittest.main()
