"""Fast synthetic regression tests; no NSRDB or trained checkpoint required."""
import importlib.util
import math
import sys
import unittest
from pathlib import Path
import numpy as np

SRC = Path(__file__).with_name('pv_policy_quality_diagnostic.py')
spec = importlib.util.spec_from_file_location('quality_diag', SRC)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)

class FakeAgent:
    def act(self,s,deterministic=True):
        return np.zeros(2,dtype=np.float32)

class FakeEnv:
    def __init__(self):
        self.horizon=4
        self.unwrapped=self
        self.tilt=30.
        self.azimuth=180.
        self.i=0
    def seed(self,n):self.seed_value=int(n)
    def reset(self,scenario_id):
        self.i=0;self.tilt=30.;self.azimuth=180.
        return self._state()
    def _state(self):
        return np.array([.2,.4,.5,self.tilt/90.,(self.azimuth/180.)-1.,(self.horizon-self.i)/self.horizon],np.float32)
    def step(self,a):
        a=np.asarray(a)
        self.tilt=float(np.clip(self.tilt+5*a[0],0,90))
        self.azimuth=float((self.azimuth+10*a[1])%360)
        self.i+=1
        done=self.i==self.horizon
        reward=1.-float((a[0]-.5)**2)  # positive actor intervention is better
        return self._state(),reward,done,{}

class DiagnosticTests(unittest.TestCase):
    def test_action_clipping_dedup(self):
        cc=diag.candidates(np.array([1.,-1.]),.5)
        self.assertEqual(cc[0][0],'actor')
        self.assertEqual(len(cc),len({tuple(x.tolist()) for _,x in cc}))
        self.assertTrue(all(np.abs(x).max()<=1 for _,x in cc))
    def test_soft_q_correct_conditioned_first_action(self):
        lp=[math.log(.5),math.log(.25)]
        got=diag.soft_return([1,2,3],lp,gamma=1,reward_scale=100,alpha=2)
        self.assertAlmostEqual(got,600.-2*sum(lp))
        got=diag.soft_return([1,2,3],lp,gamma=.5,reward_scale=100,alpha=2)
        want=(100+100+75)-2*(.5*lp[0]+.25*lp[1])
        self.assertAlmostEqual(got,want)
    def test_bellman_terminal_mask(self):
        self.assertAlmostEqual(diag.soft_td_target(.05,False,10,-2,gamma=1,reward_scale=100,alpha=.5),16.)
        self.assertAlmostEqual(diag.soft_td_target(.05,True,10,-2,gamma=1,reward_scale=100,alpha=.5),5.)
    def test_order_agreement(self):
        self.assertEqual(diag.pair_agreement([0,1,2],[0,3,4]),(1.,3))
        self.assertEqual(diag.pair_agreement([0,1,2],[3,2,1]),(0.,3))
    def test_identical_deterministic_replay_and_local_gain(self):
        env=FakeEnv();agent=FakeAgent()
        baseline=diag.baseline_rollout(env,agent,'2024-12-23',42)
        self.assertEqual(len(baseline['states']),4)
        for t in range(4):
            rr,_=diag.intervene_rollout(env,agent,'2024-12-23',42,baseline,t,baseline['actions'][t])
            self.assertAlmostEqual(sum(rr),sum(baseline['rewards'][t:]))
        act=np.array([.5,0],np.float32)
        improved,_=diag.intervene_rollout(env,agent,'2024-12-23',42,baseline,2,act)
        self.assertGreater(sum(improved),sum(baseline['rewards'][2:]))
    def test_invalid_soft_dimensions(self):
        with self.assertRaises(ValueError):diag.soft_return([1,2],[],1,100,.1)
        with self.assertRaises(ValueError):diag.soft_return([1],[float('nan')],1,100,.1)
    def test_sd_replicates_not_population_ci(self):
        mean,sd=diag._mean_std([3,5,7])
        self.assertAlmostEqual(mean,5.)
        self.assertAlmostEqual(sd,2.)
        self.assertEqual(diag._mean_std([4])[1],None)

if __name__=='__main__':unittest.main(verbosity=2)

class PairedInferenceTests(unittest.TestCase):
    def _rows(self,q_a=2.,q_b=1., det_a=3.,det_b=2.):
        return [{'candidate':'a','critic_min_soft_scaled':q_a,'deterministic_to_go_kwh':det_a},
                {'candidate':'b','critic_min_soft_scaled':q_b,'deterministic_to_go_kwh':det_b}]
    def test_short_mc_is_not_significance_checked(self):
        samples={'a':[3.,4.,5.,6.,7.],'b':[1.,1.,1.,1.,1.]}
        rows,stats=diag.paired_candidate_statistics(self._rows(),samples)
        self.assertEqual(rows[0]['status'],'insufficient_stochastic_repeats')
        self.assertIsNone(rows[0]['mc_paired_ci95_low_scaled'])
        self.assertIsNone(stats['resolved_agreement_fraction'])
    def test_paired_uncertainty_eliminates_shared_noise(self):
        # Both candidates see identical shared stochastic noise. Their
        # paired difference has exactly zero variance, unlike unpaired SD.
        noise=np.arange(20)*100.
        a=noise+3.;b=noise+2.
        rows,stats=diag.paired_candidate_statistics(self._rows(),{'a':a,'b':b})
        self.assertEqual(rows[0]['status'],'resolved_conditional_difference')
        self.assertAlmostEqual(rows[0]['mc_paired_difference_scaled'],1.)
        self.assertAlmostEqual(rows[0]['mc_paired_difference_sd_scaled'],0.)
        self.assertEqual(stats['resolved_agreement_fraction'],1.)
    def test_physical_action_alias_not_counted_as_ranking_error(self):
        values=np.arange(20)*.15
        rows,stats=diag.paired_candidate_statistics(self._rows(q_a=10.,q_b=-10.,det_a=1.,det_b=1.),
                                                   {'a':values,'b':values})
        self.assertEqual(rows[0]['status'],'physically_equivalent_on_tested_day')
        self.assertEqual(stats['resolved_count'],0)
    def test_paired_ci_includes_zero_when_differences_are_uncertain(self):
        a=np.array([2.,-2.]*10)
        b=np.zeros(20)
        rows,stats=diag.paired_candidate_statistics(self._rows(),{'a':a,'b':b})
        self.assertEqual(rows[0]['status'],'unresolved_conditional_difference')
        self.assertIsNone(rows[0]['critic_order_correct_if_resolved'])
        self.assertEqual(stats['resolved_count'],0)
